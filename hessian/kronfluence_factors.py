"""Compute EKFAC influence factors — kronfluence equivalent of bergson's hessian pipeline.

Usage:
    python -m hessian.kronfluence_factors \
        --model /path/to/model \
        --dataset /path/to/data.jsonl \
        --factor_strategy ekfac \
        --factors_name my_factors

The bergson script with  method=kfac + --ev_correction  maps to  strategy="ekfac"
in kronfluence (EKFAC = eigenvalue-corrected KFAC).
"""

import argparse
import logging
from datetime import timedelta

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from common.kron_pipeline import construct_model, get_dataset
from common.kron_task import LanguageModelingTask
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.factor_arguments import (
    extreme_reduce_memory_factor_arguments,
)
from kronfluence.utils.dataset import DataLoaderKwargs

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute influence factors (kronfluence equivalent of bergson hessian)."
    )

    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--torch_dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prompt_column", type=str, default="prompt")
    parser.add_argument("--completion_column", type=str, default="completion")
    parser.add_argument("--max_length", type=int, default=2048)

    # ── Factor computation ───────────────────────────────────────────────────
    parser.add_argument("--factors_name", type=str, default="bergson_compare")
    parser.add_argument(
        "--factor_strategy", type=str, default="ekfac",
        choices=["identity", "diagonal", "kfac", "ekfac"],
    )
    parser.add_argument("--factor_batch_size", type=int, default=4)

    # ── Module tracking ──────────────────────────────────────────────────────
    parser.add_argument("--tracked_modules", type=str, default=None)

    # ── Memory / partitioning ────────────────────────────────────────────────
    parser.add_argument("--covariance_module_partitions", type=int, default=2)
    parser.add_argument("--covariance_data_partitions", type=int, default=4)
    parser.add_argument("--lambda_module_partitions", type=int, default=4)
    parser.add_argument("--lambda_data_partitions", type=int, default=4)

    # ── Fisher / bias ──────────────────────────────────────────────────────
    parser.add_argument("--use_empirical_fisher", action="store_true", default=False)
    parser.add_argument("--include_bias", action="store_true", default=False)

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
        prompt_column=args.prompt_column,
        completion_column=args.completion_column,
        max_length=args.max_length,
        split=args.split,
    )
    logging.info("Dataset size: %d", len(train_dataset))

    # ── Model ────────────────────────────────────────────────────────────────
    model = construct_model(args.model, torch_dtype=dtype)

    # ── Task + tracked modules ───────────────────────────────────────────────
    tracked_modules = None
    if args.tracked_modules:
        tracked_modules = [m.strip() for m in args.tracked_modules.split(",")]
    task = LanguageModelingTask(tracked_modules=tracked_modules)

    model = prepare_model(model, task)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    model = accelerator.prepare_model(model)

    # ── Analyzer ─────────────────────────────────────────────────────────────
    analyzer = Analyzer(
        analysis_name="bergson_compare",
        model=model,
        task=task,
        profile=args.profile,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    dataloader_kwargs = DataLoaderKwargs(
        num_workers=4,
        collate_fn=collator,
        pin_memory=True,
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # ── Factor arguments ─────────────────────────────────────────────────────
    factor_args = extreme_reduce_memory_factor_arguments(
        strategy=args.factor_strategy,
        module_partitions=1,
        dtype=dtype,
    )
    factor_args.covariance_module_partitions = args.covariance_module_partitions
    factor_args.covariance_data_partitions = args.covariance_data_partitions
    factor_args.lambda_module_partitions = args.lambda_module_partitions
    factor_args.lambda_data_partitions = args.lambda_data_partitions
    factor_args.use_empirical_fisher = args.use_empirical_fisher
    factor_args.include_bias = args.include_bias

    # ── Fit factors ──────────────────────────────────────────────────────────
    analyzer.fit_all_factors(
        factors_name=args.factors_name,
        dataset=train_dataset,
        per_device_batch_size=args.factor_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=args.overwrite,
    )
    logging.info("Done. Factors saved under analysis_name='bergson_compare', factors_name='%s'", args.factors_name)


if __name__ == "__main__":
    main()
