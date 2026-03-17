"""Model and dataset construction for bergson comparison.

Mirrors the bergson run_hessian.sh configuration:
  - Auto-detects PEFT adapters: loads base model + applies adapter (like bergson)
  - Loads a prompt/completion JSONL dataset
  - Tokenises with chat template + prompt masking (labels=-100 on the prompt portion)
"""

import logging
from typing import List, Optional

import torch
from datasets import load_dataset
from torch import nn
from torch.utils import data
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def construct_model(
    model_name_or_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load a causal LM, auto-detecting PEFT adapters.

    If model_name_or_path contains a PEFT adapter_config.json, the base model
    is loaded first and the adapter is applied on top (mirroring bergson's
    setup_model_and_peft behaviour). Otherwise the path is loaded directly.
    """
    # Try to detect a PEFT adapter
    peft_config = None
    try:
        from peft import PeftConfig
        peft_config = PeftConfig.from_pretrained(model_name_or_path)
        logger.info(
            "Detected PEFT adapter at %s (base model: %s)",
            model_name_or_path,
            peft_config.base_model_name_or_path,
        )
    except (ValueError, ImportError):
        pass

    if peft_config is not None:
        from peft import PeftModel

        # Load the base model
        base_model_name = peft_config.base_model_name_or_path
        config = AutoConfig.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        # Apply the adapter
        model = PeftModel.from_pretrained(base_model, model_name_or_path)
        logger.info("Loaded PEFT model with adapter from %s", model_name_or_path)
    else:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

    return model


def get_dataset(
    dataset_path: str,
    tokenizer_name_or_path: str,
    prompt_column: str = "prompt",
    completion_column: str = "completion",
    max_length: int = 2048,
    split: str = "train",
    indices: Optional[List[int]] = None,
) -> data.Dataset:
    """Load a JSONL dataset with prompt/completion columns and tokenise it.

    Prompt tokens get labels=-100 so only the completion contributes to the loss,
    matching bergson's default behaviour.
    """
    raw_datasets = load_dataset("json", data_files=dataset_path, split=split)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        # Apply chat template to match bergson's tokenize() behaviour
        convo = [
            {"role": "user", "content": example[prompt_column]},
            {"role": "assistant", "content": example[completion_column]},
        ]
        formatted = tokenizer.apply_chat_template(convo, tokenize=False)
        enc = tokenizer(
            formatted,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Mask prompt tokens — only the assistant completion gets labels.
        # Find the completion text in the formatted string and map char→token.
        labels = [-100] * len(input_ids)
        ans = example[completion_column]
        start_char = formatted.rfind(ans)
        if start_char >= 0:
            end_char = start_char + len(ans)
            # Re-encode with offsets to get char→token mapping
            enc_off = tokenizer(
                formatted,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
            start_tok = enc_off.char_to_token(start_char)
            end_tok = enc_off.char_to_token(end_char - 1)
            if start_tok is not None and end_tok is not None:
                labels[start_tok : end_tok + 1] = input_ids[start_tok : end_tok + 1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized = raw_datasets.map(
        tokenize_function,
        batched=False,
        remove_columns=raw_datasets.column_names,
        load_from_cache_file=True,
        desc="Tokenising dataset",
    )

    if indices is not None:
        tokenized = tokenized.select(indices)

    return tokenized
