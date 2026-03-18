"""Run the full EKFAC influence pipeline end-to-end (bergson).

Given a query dataset, index (training) dataset, and model, produces influence
scores by:

1. Building a mean query gradient (bergson build with aggregation=mean).
2. Fitting EKFAC factors on the training dataset (bergson hessian with
   method=kfac + ev_correction).
3. Applying the EKFAC inverse Hessian to the mean query gradient
   (apply_hessian.py).
4. Scoring each training example against the EKFAC-transformed query gradient
   (bergson score).

No intermediate per-example gradients are saved except the mean query gradient
from step 1.

Example:
    python -m ekfac_scores.bergson_pipeline ekfac_scores/results/bergson \
        --query_dataset NeelNanda/pile-10k \
        --index_dataset NeelNanda/pile-10k \
        --model EleutherAI/pythia-14m-deduped \
        --prompt_column text \
        --precision fp32 \
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
from bergson.config import HessianConfig, IndexConfig, PreprocessConfig, ScoreConfig
from bergson.hessians.apply_hessian import EkfacApplicator, EkfacConfig
from bergson.hessians.hessian_approximations import approximate_hessians
from bergson.score.score import score_dataset
from bergson.utils.worker_utils import validate_run_path


@dataclass
class PipelineConfig:
    """Extra arguments specific to the EKFAC influence pipeline."""

    query_dataset: str
    """HuggingFace dataset identifier or local path for the query set."""

    index_dataset: str
    """HuggingFace dataset identifier or local path for the index (training) set."""

    lambda_damp_factor: float = 0.1
    """Damping factor for EKFAC eigenvalue correction."""


def main():
    parser = ArgumentParser(
        description="Run the full EKFAC influence pipeline.",
        conflict_resolution=ConflictResolution.EXPLICIT,
    )
    parser.add_arguments(PipelineConfig, dest="pipeline_cfg")
    parser.add_arguments(IndexConfig, dest="index_cfg")
    parser.add_arguments(HessianConfig, dest="hessian_cfg")
    args = parser.parse_args()
    pipeline_cfg: PipelineConfig = args.pipeline_cfg
    index_cfg: IndexConfig = args.index_cfg
    hessian_cfg: HessianConfig = args.hessian_cfg

    run_path = Path(index_cfg.run_path)

    # ── Step 1: Build mean query gradient ─────────────────────────────────────
    print("=" * 60)
    print("Step 1: Building mean query gradient")
    print("=" * 60)
    query_cfg = deepcopy(index_cfg)
    query_cfg.run_path = str(run_path / "query")
    query_cfg.data.dataset = pipeline_cfg.query_dataset
    query_cfg.projection_dim = 0  # no random projection for EKFAC
    query_cfg.skip_preconditioners = True
    validate_run_path(query_cfg)

    preprocess_cfg = PreprocessConfig(aggregation="mean")
    build(query_cfg, preprocess_cfg)

    # ── Step 2: Fit EKFAC factors on training data ────────────────────────────
    print("=" * 60)
    print("Step 2: Fitting EKFAC factors on training data")
    print("=" * 60)
    hessian_index_cfg = deepcopy(index_cfg)
    hessian_index_cfg.run_path = str(run_path / "hessian")
    hessian_index_cfg.data.dataset = pipeline_cfg.index_dataset
    validate_run_path(hessian_index_cfg)

    # Force EKFAC
    hessian_cfg.method = "kfac"
    hessian_cfg.ev_correction = True

    approximate_hessians(hessian_index_cfg, hessian_cfg)

    # ── Step 3: Apply EKFAC to the mean query gradient ────────────────────────
    print("=" * 60)
    print("Step 3: Applying EKFAC to mean query gradient")
    print("=" * 60)
    hessian_method_path = str(run_path / "hessian" / hessian_cfg.method)
    query_gradient_path = str(run_path / "query")
    ekfac_query_path = str(run_path / "ekfac_query")

    ekfac_cfg = EkfacConfig(
        hessian_method_path=hessian_method_path,
        gradient_path=query_gradient_path,
        run_path=ekfac_query_path,
        lambda_damp_factor=pipeline_cfg.lambda_damp_factor,
    )
    applicator = EkfacApplicator(ekfac_cfg)
    applicator.compute_ivhp_sharded()

    # ── Step 4: Score training examples against EKFAC-transformed query ───────
    print("=" * 60)
    print("Step 4: Scoring training data against EKFAC-transformed query")
    print("=" * 60)
    score_index_cfg = deepcopy(index_cfg)
    score_index_cfg.run_path = str(run_path / "scores")
    score_index_cfg.data.dataset = pipeline_cfg.index_dataset
    score_index_cfg.projection_dim = 0  # no random projection for EKFAC
    score_index_cfg.skip_preconditioners = True
    validate_run_path(score_index_cfg)

    score_cfg = ScoreConfig(query_path=ekfac_query_path)
    score_dataset(score_index_cfg, score_cfg, PreprocessConfig())

    # Save flat scores for easy comparison
    from bergson.data import load_scores

    scores = load_scores(Path(score_index_cfg.run_path))
    flat = torch.from_numpy(scores.mmap["score_0"].astype("float32"))
    out_path = run_path.parent / "bergson_scores.pt"
    torch.save(flat, out_path)

    print("=" * 60)
    print(f"Done! Scores saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
