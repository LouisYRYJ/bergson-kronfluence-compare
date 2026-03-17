"""Task definition for causal language modelling (bergson comparison).

Mirrors bergson's defaults:
  - loss_fn: cross-entropy
  - loss_reduction: "mean" (bergson default) — but kronfluence factor fitting
    uses the summed per-token loss internally, so we return summed loss here
    (matching the openwebtext example). The "mean" vs "sum" distinction in
    bergson controls how gradients are scaled; kronfluence handles this via
    its own factor computation pipeline.
  - Tracked modules: configurable via constructor so the caller can replicate
    bergson's --filter_modules behaviour by specifying which modules to INCLUDE.

The bergson run_hessian.sh filter_modules for the Qwen2.5-14B model:
    "*.mlp.*,*model.layers.[0-9].*,*model.layers.1[0-9].*,
     *model.layers.2[0-9].*,*model.layers.3[0-9].*,*model.layers.4[0-6].*,
     *.lm_head,*.lora_B.*"
This EXCLUDES all MLP, layers 0-46, lm_head, and lora_B.
For a 48-layer Qwen2.5-14B (layers 0-47), only layer 47 attention remains.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from kronfluence.task import Task

BATCH_TYPE = Dict[str, torch.Tensor]

# Qwen2.5-14B-Instruct has 48 layers (0-47).
# bergson's filter_modules excludes layers 0-46, all MLP, lm_head, lora_B.
# What remains: lora_A attention projections in layer 47 only.
# (The model is a LoRA adapter, so the actual Linear modules are .lora_A.default)
DEFAULT_TRACKED_MODULES = [
    "model.layers.47.self_attn.q_proj.lora_A.default",
    "model.layers.47.self_attn.k_proj.lora_A.default",
    "model.layers.47.self_attn.v_proj.lora_A.default",
    "model.layers.47.self_attn.o_proj.lora_A.default",
]


class LanguageModelingTask(Task):
    def __init__(self, tracked_modules: Optional[List[str]] = None):
        super().__init__()
        self._tracked_modules = tracked_modules

    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        # Flash/mem-efficient SDPA are disabled in compute_scores.py, so passing
        # attention_mask is safe (won't trigger a different kernel path) and needed
        # for EKFAC covariance factor computation with variable-length data.
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        labels = batch["labels"][..., 1:].contiguous().view(-1)
        if not sample:
            return F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)
        else:
            with torch.no_grad():
                probs = torch.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(probs, num_samples=1).flatten()
                sampled_labels[labels == -100] = -100
            return F.cross_entropy(logits, sampled_labels, ignore_index=-100, reduction="sum")

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        labels = batch["labels"][..., 1:].contiguous().view(-1)
        return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        return self._tracked_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        # Match bergson's valid_mask: position i is valid iff labels[i+1] != -100.
        # This ensures only completion positions contribute to covariance,
        # excluding prompt tokens (which have labels=-100).
        labels = batch["labels"]
        mask = torch.zeros_like(labels, dtype=torch.long)
        mask[:, :-1] = (labels[:, 1:] != -100).long()
        return mask
