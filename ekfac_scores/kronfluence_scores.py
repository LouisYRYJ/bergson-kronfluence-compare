"""Compute pairwise influence scores using pre-computed EKFAC factors.

Supports both plain-text datasets (--text_column) and prompt/completion
datasets (--prompt_column + --completion_column).

Usage:
    python -m ekfac_scores.kronfluence_scores \
        --model EleutherAI/pythia-14m-deduped \
        --query_dataset NeelNanda/pile-10k \
        --train_dataset NeelNanda/pile-10k \
        --factors_name ekfac_pythia14m \
        --scores_name ekfac_scores
"""

import argparse
import logging
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, default_data_collator

from common.kron_task import LanguageModelingTask
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import ScoreArguments
from kronfluence.utils.dataset import DataLoaderKwargs

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)


def parse_args():
    parser = argparse.ArgumentParser(description="Influence scores using pre-computed EKFAC factors.")

    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--torch_dtype", type=str, default="float32",
        choices=["float32", "float16", "bfloat16"],
    )

    # ── Datasets ─────────────────────────────────────────────────────────────
    parser.add_argument("--query_dataset", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--query_size", type=int, default=None)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--truncation", action="store_true", default=False)
    # Plain text mode (default)
    parser.add_argument("--text_column", type=str, default=None)
    # Prompt/completion mode
    parser.add_argument("--prompt_column", type=str, default=None)
    parser.add_argument("--completion_column", type=str, default=None)

    # ── Factors (pre-computed) ───────────────────────────────────────────────
    parser.add_argument("--factors_name", type=str, required=True)
    parser.add_argument(
        "--analysis_name", type=str, default=None,
        help="Analysis name used when fitting factors. Defaults to factors_name.",
    )

    # ── Module tracking ──────────────────────────────────────────────────────
    parser.add_argument("--tracked_modules", type=str, default=None)

    # ── Scores ───────────────────────────────────────────────────────────────
    parser.add_argument("--scores_name", type=str, default="ekfac_scores")
    parser.add_argument("--query_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--data_partitions", type=int, default=1)
    parser.add_argument("--module_partitions", type=int, default=1)
    parser.add_argument(
        "--aggregate_query_gradients",
        action=argparse.BooleanOptionalAction, default=True,
    )

    parser.add_argument("--output_dir", type=str, default="./ekfac_scores/results")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False)

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

    # ── Determine dataset mode ───────────────────────────────────────────────
    if args.prompt_column and args.completion_column:
        from common.kron_pipeline import construct_model, get_dataset
        get_ds = lambda path, size: get_dataset(
            dataset_path=path,
            tokenizer_name_or_path=args.model,
            prompt_column=args.prompt_column,
            completion_column=args.completion_column,
            max_length=args.max_length,
            split=args.split,
            indices=list(range(size)) if size else None,
        )
        use_seq2seq_collator = True
    else:
        from common.kron_pipeline_simple import construct_model, get_dataset
        text_col = args.text_column or "text"
        get_ds = lambda path, size: get_dataset(
            dataset_path=path,
            tokenizer_name_or_path=args.model,
            text_column=text_col,
            max_length=args.max_length,
            split=args.split,
            truncation=args.truncation,
            indices=list(range(size)) if size else None,
        )
        use_seq2seq_collator = False

    query_dataset = get_ds(args.query_dataset, args.query_size)
    train_dataset = get_ds(args.train_dataset, args.train_size)
    logging.info("Query: %d examples, Train: %d examples", len(query_dataset), len(train_dataset))

    # ── Model ────────────────────────────────────────────────────────────────
    model = construct_model(args.model, torch_dtype=dtype)

    tracked_modules = None
    if args.tracked_modules:
        tracked_modules = [m.strip() for m in args.tracked_modules.split(",")]
    else:
        prefix = model.base_model_prefix
        tracked_modules = [
            f"{prefix}.{n}" if prefix else n
            for n, m in model.base_model.named_modules()
            if isinstance(m, torch.nn.Linear)
        ]
        logging.info("Auto-tracked %d modules (excluding embed/unembed)", len(tracked_modules))
    task = LanguageModelingTask(tracked_modules=tracked_modules)

    model = prepare_model(model, task)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    model = accelerator.prepare_model(model)

    # ── Analyzer ─────────────────────────────────────────────────────────────
    analysis_name = args.analysis_name or "kronfluence"
    analyzer = Analyzer(
        analysis_name=analysis_name,
        model=model,
        task=task,
        output_dir=args.output_dir,
        profile=args.profile,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_seq2seq_collator:
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    else:
        collator = default_data_collator

    dataloader_kwargs = DataLoaderKwargs(num_workers=4, collate_fn=collator, pin_memory=True)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # ── Scores ───────────────────────────────────────────────────────────────
    score_args = ScoreArguments(
        data_partitions=args.data_partitions,
        module_partitions=args.module_partitions,
        aggregate_query_gradients=args.aggregate_query_gradients,
    )
    analyzer.compute_pairwise_scores(
        scores_name=args.scores_name,
        factors_name=args.factors_name,
        query_dataset=query_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        score_args=score_args,
        overwrite_output_dir=args.overwrite,
    )

    scores = analyzer.load_pairwise_scores(args.scores_name)["all_modules"]
    if args.aggregate_query_gradients:
        scores = scores / len(query_dataset)

    out_path = Path(args.output_dir) / "kronfluence_scores.pt"
    torch.save(scores.squeeze(), out_path)
    logging.info("Scores shape: %s, saved to %s", scores.shape, out_path)


if __name__ == "__main__":
    main()
