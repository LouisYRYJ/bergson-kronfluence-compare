"""Run the Hessian approximation pipeline (bergson).

Given a dataset and model, computes Kronecker-factored Hessian approximations
with deterministic SDPA backends for reproducibility.

Example:
    python -m common.bergson_hessian results/hessian \
        --dataset NeelNanda/pile-10k \
        --model EleutherAI/pythia-14m-deduped \
        --prompt_column text \
        --precision fp32 \
        --truncation
"""

import torch

# Force deterministic math SDPA kernel — flash/mem-efficient kernels produce
# subtly different backward gradients depending on batch size.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.matmul.allow_tf32 = False

from simple_parsing import ArgumentParser, ConflictResolution

from bergson.config import HessianConfig, IndexConfig
from bergson.hessians.hessian_approximations import approximate_hessians
from bergson.utils.worker_utils import validate_run_path


def main():
    parser = ArgumentParser(
        description="Run the Hessian approximation pipeline.",
        conflict_resolution=ConflictResolution.EXPLICIT,
    )
    parser.add_arguments(IndexConfig, dest="index_cfg")
    parser.add_arguments(HessianConfig, dest="hessian_cfg")
    args = parser.parse_args()
    index_cfg: IndexConfig = args.index_cfg
    hessian_cfg: HessianConfig = args.hessian_cfg

    validate_run_path(index_cfg)

    print(f"Computing Hessian approximation ({hessian_cfg.method})")
    print(f"  Model: {index_cfg.model}")
    print(f"  Dataset: {index_cfg.data.dataset}")
    print(f"  Precision: {index_cfg.precision}")
    print(f"  Hessian dtype: {hessian_cfg.hessian_dtype}")
    approximate_hessians(index_cfg, hessian_cfg)

    print(f"Hessian output path: {index_cfg.run_path}")


if __name__ == "__main__":
    main()
