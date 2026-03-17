"""Minimal model and dataset construction — plain text, no LoRA."""

import logging
import os
from typing import List, Optional

import torch
from datasets import load_dataset
from torch import nn
from torch.utils import data
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def construct_model(
    model_name_or_path: str,
    torch_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Load a causal LM from HF hub or local path."""
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
    text_column: str = "text",
    max_length: int = 2048,
    split: str = "train",
    truncation: bool = False,
    indices: Optional[List[int]] = None,
) -> data.Dataset:
    """Load a plain-text dataset and tokenise for next-token prediction.

    All tokens contribute to the loss (no prompt masking).
    """
    if os.path.isfile(dataset_path):
        raw_datasets = load_dataset("json", data_files=dataset_path, split=split)
    else:
        raw_datasets = load_dataset(dataset_path, split=split)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        results = tokenizer(
            examples[text_column],
            add_special_tokens=True,
            truncation=truncation,
            padding=True,
            max_length=max_length,
        )
        results["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in ids]
            for ids in results["input_ids"]
        ]
        return results

    tokenized = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets.column_names,
        load_from_cache_file=True,
        desc="Tokenising dataset",
    )

    if indices is not None:
        tokenized = tokenized.select(indices)

    return tokenized
