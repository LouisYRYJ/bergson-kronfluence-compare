"""Run the full influence pipeline end-to-end (bergson).

Given a query dataset, index dataset, and model, produces a single influence
array of shape [num_index_examples] by:

1. Reducing the query dataset gradients to a single mean vector.
2. Scoring each index example against that reduced query gradient.

Example:
    python -m raw_influence.bergson_pipeline results/raw_influence/bergson \
        --query_dataset NeelNanda/pile-10k \
        --index_dataset NeelNanda/pile-10k \
        --truncation
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

# Force deterministic math SDPA kernel — flash/mem-efficient kernels produce
# subtly different backward gradients depending on batch size.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.matmul.allow_tf32 = False

from simple_parsing import ArgumentParser, ConflictResolution

from bergson.build import build
from bergson.config import IndexConfig, PreprocessConfig, ScoreConfig
from bergson.score.score import score_dataset
from bergson.utils.worker_utils import validate_run_path


@dataclass
class PipelineConfig:
    """Extra arguments specific to the influence pipeline."""

    query_dataset: str
    """HuggingFace dataset identifier for the query set."""

    index_dataset: str
    """HuggingFace dataset identifier for the index set."""


def main():
    parser = ArgumentParser(
        description="Run the full influence pipeline (reduce + score).",
        conflict_resolution=ConflictResolution.EXPLICIT,
    )
    parser.add_arguments(PipelineConfig, dest="pipeline_cfg")
    parser.add_arguments(IndexConfig, dest="index_cfg")
    args = parser.parse_args()
    pipeline_cfg: PipelineConfig = args.pipeline_cfg
    index_cfg: IndexConfig = args.index_cfg

    run_path = Path(index_cfg.run_path)

    # Step 1: Reduce query gradients to a single mean vector
    query_cfg = deepcopy(index_cfg)
    query_cfg.run_path = str(run_path / "query")
    query_cfg.data.dataset = pipeline_cfg.query_dataset
    validate_run_path(query_cfg)

    print(f"Reducing query gradients: {pipeline_cfg.query_dataset}")
    preprocess_cfg = PreprocessConfig(aggregation="sum")
    build(query_cfg, preprocess_cfg)

    # Step 2: Score each index example against the reduced query gradient
    score_index_cfg = deepcopy(index_cfg)
    score_index_cfg.run_path = str(run_path / "scores")
    score_index_cfg.data.dataset = pipeline_cfg.index_dataset
    validate_run_path(score_index_cfg)

    score_cfg = ScoreConfig(query_path=str(run_path / "query"))
    print(f"Scoring index dataset: {pipeline_cfg.index_dataset}")
    score_dataset(score_index_cfg, score_cfg, PreprocessConfig())

    print(f"Scores path: {score_index_cfg.run_path}")


if __name__ == "__main__":
    main()
