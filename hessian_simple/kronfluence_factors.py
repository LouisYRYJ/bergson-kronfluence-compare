"""Compute EKFAC factors — minimal version for plain text models (no LoRA).

Matches bergson's hessian_pipeline.py setup:
  - Deterministic SDPA (flash/mem-efficient disabled)
  - TF32 disabled for exact fp32 reproducibility

Usage:
    python -m hessian_simple.kronfluence_factors \
        --model EleutherAI/pythia-14m-deduped \
        --dataset NeelNanda/pile-10k \
        --factor_strategy ekfac
"""

import argparse
import logging

import torch

# Force deterministic SDPA kernel — flash/mem-efficient kernels produce
# subtly different backward gradients depending on batch size.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.matmul.allow_tf32 = False

from transformers import default_data_collator

from common.kron_pipeline_simple import construct_model, get_dataset
from common.kron_task import LanguageModelingTask
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.utils.dataset import DataLoaderKwargs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute EKFAC factors (minimal, no LoRA)."
    )

    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model", type=str, required=True,
        help="HF hub id or local path to the model.",
    )
    parser.add_argument(
        "--torch_dtype", type=str, default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype.",
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="HF hub dataset name or local JSONL path.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--truncation", action="store_true", default=False)

    # ── Factor computation ───────────────────────────────────────────────────
    parser.add_argument("--factors_name", type=str, default="bergson_compare")
    parser.add_argument(
        "--factor_strategy", type=str, default="ekfac",
        choices=["identity", "diagonal", "kfac", "ekfac"],
    )
    parser.add_argument("--factor_batch_size", type=int, default=4)

    # ── Module tracking ──────────────────────────────────────────────────────
    parser.add_argument(
        "--tracked_modules", type=str, default=None,
        help="Comma-separated module names to track. If unset, tracks all.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir", type=str, default="./hessian_simple/results/kronfluence",
        help="Directory for storing analysis results.",
    )

    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)

    return parser.parse_args()


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    dtype = DTYPE_MAP[args.torch_dtype]

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_dataset = get_dataset(
        dataset_path=args.dataset,
        tokenizer_name_or_path=args.model,
        text_column=args.text_column,
        max_length=args.max_length,
        split=args.split,
        truncation=args.truncation,
    )
    logging.info("Dataset size: %d", len(train_dataset))

    # ── Model ────────────────────────────────────────────────────────────────
    model = construct_model(args.model, torch_dtype=dtype)

    # ── Task + tracked modules ───────────────────────────────────────────────
    if args.tracked_modules:
        tracked_modules = [m.strip() for m in args.tracked_modules.split(",")]
    else:
        # Track all nn.Linear under base_model (excludes embed/unembed layers)
        prefix = model.base_model_prefix
        tracked_modules = [
            f"{prefix}.{n}" if prefix else n
            for n, m in model.base_model.named_modules()
            if isinstance(m, torch.nn.Linear)
        ]
        logging.info("Auto-tracked %d modules (excluding embed/unembed)", len(tracked_modules))
    task = LanguageModelingTask(tracked_modules=tracked_modules)

    model = prepare_model(model, task)

    # ── Analyzer ─────────────────────────────────────────────────────────────
    analyzer = Analyzer(
        analysis_name=args.factors_name,
        model=model,
        task=task,
        profile=args.profile,
        output_dir=args.output_dir,
    )
    dataloader_kwargs = DataLoaderKwargs(
        num_workers=4, collate_fn=default_data_collator, pin_memory=True,
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # ── Factor arguments ─────────────────────────────────────────────────────
    factor_args = FactorArguments(
        strategy=args.factor_strategy,
        include_bias=False,
        use_empirical_fisher=True,
    )

    # ── Fit factors ──────────────────────────────────────────────────────────
    analyzer.fit_all_factors(
        factors_name=args.factors_name,
        dataset=train_dataset,
        per_device_batch_size=args.factor_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=args.overwrite,
    )
    logging.info(
        "Done. Factors saved to %s/%s/%s",
        args.output_dir, args.factors_name, args.factors_name,
    )


if __name__ == "__main__":
    main()
