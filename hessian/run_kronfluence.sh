#!/usr/bin/env bash
set -euo pipefail

# ── User Configuration (mirrors bergson's hessian config) ────────────────────
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DATASET="/mnt/ssd-1/louis/emergent_misalignment/data/merged_medical/merged_medical.jsonl"

# ── Data ─────────────────────────────────────────────────────────────────────
SPLIT="train"
PROMPT_COLUMN="prompt"
COMPLETION_COLUMN="completion"
MAX_LENGTH=2048

# ── Factor strategy ──────────────────────────────────────────────────────────
FACTOR_STRATEGY="ekfac"
FACTORS_NAME="ekfac"
ANALYSIS_NAME="kronfluence"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$DIR/results"
FACTOR_BATCH_SIZE=2
TORCH_DTYPE="float32"

# ── Module tracking (only layer 23 — matches bergson's FILTER_MODULES) ──────
TRACKED_MODULES="model.layers.23.self_attn.q_proj,model.layers.23.self_attn.k_proj,model.layers.23.self_attn.v_proj,model.layers.23.self_attn.o_proj,model.layers.23.mlp.gate_proj,model.layers.23.mlp.up_proj,model.layers.23.mlp.down_proj"

# ── Memory partitioning ─────────────────────────────────────────────────────
COVARIANCE_MODULE_PARTITIONS=1
COVARIANCE_DATA_PARTITIONS=1
LAMBDA_MODULE_PARTITIONS=1
LAMBDA_DATA_PARTITIONS=1

ENV_PREFIX="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"

# ── Build & run command ──────────────────────────────────────────────────────
CMD="python -m hessian.kronfluence_factors"
CMD+=" --model ${MODEL}"
CMD+=" --dataset ${DATASET}"
CMD+=" --split ${SPLIT}"
CMD+=" --prompt_column ${PROMPT_COLUMN}"
CMD+=" --completion_column ${COMPLETION_COLUMN}"
CMD+=" --max_length ${MAX_LENGTH}"
CMD+=" --factor_strategy ${FACTOR_STRATEGY}"
CMD+=" --factors_name ${FACTORS_NAME}"
CMD+=" --factor_batch_size ${FACTOR_BATCH_SIZE}"
CMD+=" --torch_dtype ${TORCH_DTYPE}"
CMD+=" --covariance_module_partitions ${COVARIANCE_MODULE_PARTITIONS}"
CMD+=" --covariance_data_partitions ${COVARIANCE_DATA_PARTITIONS}"
CMD+=" --lambda_module_partitions ${LAMBDA_MODULE_PARTITIONS}"
CMD+=" --lambda_data_partitions ${LAMBDA_DATA_PARTITIONS}"
CMD+=" --output_dir ${OUTPUT_DIR}"
CMD+=" --analysis_name ${ANALYSIS_NAME}"
CMD+=" --use_empirical_fisher"
CMD+=" --overwrite"

[[ -n "${TRACKED_MODULES}" ]] && CMD+=" --tracked_modules \"${TRACKED_MODULES}\""

mkdir -p "$DIR/results"
echo "Running: ${ENV_PREFIX} ${CMD}"
eval "${ENV_PREFIX} ${CMD}" 2>&1 | tee "$DIR/results/kronfluence.log"
