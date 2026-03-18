"""Compute pairwise influence scores with identity strategy (no hessian).

Mirrors the bergson influence_pipeline.py setup: accepts HuggingFace dataset
identifiers (e.g. NeelNanda/pile-10k) and computes raw gradient dot products.

Usage:
    python -m raw_influence.kronfluence_scores \
        --query_dataset NeelNanda/pile-10k \
        --train_dataset NeelNanda/pile-10k \
        --query_size 32 \
        --train_size 128
"""

import argparse
import logging
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator

from common.kron_pipeline import construct_model
from common.kron_task import LanguageModelingTask
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.utils.dataset import DataLoaderKwargs

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
# Force deterministic math SDPA kernel — flash/mem-efficient kernels produce
# subtly different backward gradients depending on batch size.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)


def parse_args():
    parser = argparse.ArgumentParser(description="Influence scores with identity strategy.")

    parser.add_argument("--model", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--query_dataset", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--query_split", type=str, default="train")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--query_size", type=int, default=None)
    parser.add_argument("--train_size", type=int, default=None)

    parser.add_argument("--scores_name", type=str, default="identity_scores")
    parser.add_argument("--query_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--data_partitions", type=int, default=1)
    parser.add_argument("--module_partitions", type=int, default=1)
    parser.add_argument(
        "--aggregate_query_gradients",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sum query gradients into one vector before scoring.",
    )

    parser.add_argument("--output_dir", type=str, default="./raw_influence/results")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False)

    return parser.parse_args()


def get_hf_dataset(dataset_name, tokenizer, text_column, max_length, split="train", num_examples=None):
    """Load an HF dataset (hub ID or local file path) and tokenize for next-token prediction (labels = input_ids)."""
    import os
    if os.path.exists(dataset_name):
        # Local file: infer format from extension and load with data_files.
        ext = os.path.splitext(dataset_name)[-1].lower()
        fmt = "json" if ext in (".jsonl", ".json") else ext.lstrip(".")
        raw = load_dataset(fmt, data_files={split: dataset_name}, split=split)
    else:
        raw = load_dataset(dataset_name, split=split)
    if num_examples is not None:
        raw = raw.select(range(min(num_examples, len(raw))))

    column_names = raw.column_names

    def tokenize_fn(examples):
        results = tokenizer(examples[text_column], truncation=True, padding=True, max_length=max_length)
        results["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in ids]
            for ids in results["input_ids"]
        ]
        return results

    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc=f"Tokenising {dataset_name}",
    )
    return tokenized


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets.
    train_dataset = get_hf_dataset(
        args.train_dataset, tokenizer, args.text_column, args.max_length,
        split=args.train_split, num_examples=args.train_size,
    )
    query_dataset = get_hf_dataset(
        args.query_dataset, tokenizer, args.text_column, args.max_length,
        split=args.query_split, num_examples=args.query_size,
    )
    logging.info("Query dataset size: %d, Train dataset size: %d", len(query_dataset), len(train_dataset))

    # Model.
    model = construct_model(args.model, torch_dtype=torch.float32)
    # Match bergson's module tracking: bergson uses model.base_model for hook
    # registration, which excludes the LM head (embed_out). Get all nn.Linear
    # module names from base_model, prefixed with the base_model_prefix.
    prefix = model.base_model_prefix
    tracked = [
        f"{prefix}.{n}" if prefix else n
        for n, m in model.base_model.named_modules()
        if isinstance(m, torch.nn.Linear)
    ]
    logging.info("Tracking %d modules (excluding LM head): %s", len(tracked), tracked[:3])
    task = LanguageModelingTask(tracked_modules=tracked)
    model = prepare_model(model, task)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    model = accelerator.prepare_model(model)

    # Analyzer.
    analyzer = Analyzer(
        analysis_name="kronfluence",
        model=model,
        task=task,
        output_dir=args.output_dir,
        profile=args.profile,
    )
    dataloader_kwargs = DataLoaderKwargs(num_workers=4, collate_fn=default_data_collator, pin_memory=True)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Identity factors (no-op).
    factor_args = FactorArguments(strategy="identity")
    analyzer.fit_all_factors(
        factors_name="identity",
        dataset=train_dataset,
        per_device_batch_size=args.train_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=args.overwrite,
    )

    # Scores.
    score_args = ScoreArguments(
        data_partitions=args.data_partitions,
        module_partitions=args.module_partitions,
        aggregate_query_gradients=args.aggregate_query_gradients,
    )
    analyzer.compute_pairwise_scores(
        scores_name=args.scores_name,
        factors_name="identity",
        query_dataset=query_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        score_args=score_args,
        overwrite_output_dir=args.overwrite,
    )

    scores = analyzer.load_pairwise_scores(args.scores_name)["all_modules"]

    out_path = Path(args.output_dir) / "kronfluence_scores.pt"
    torch.save(scores.squeeze(), out_path)
    logging.info("Scores shape: %s, saved to %s", scores.shape, out_path)


if __name__ == "__main__":
    main()
