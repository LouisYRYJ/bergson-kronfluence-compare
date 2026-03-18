#!/usr/bin/env bash
set -euo pipefail

# ── User Configuration ───────────────────────────────────────────────────────
MODEL="EleutherAI/pythia-14m-deduped"
DATASET="NeelNanda/pile-10k"

# ── Data ─────────────────────────────────────────────────────────────────────
SPLIT="train"
TEXT_COLUMN="text"
MAX_LENGTH=2048
TRUNCATION="--truncation"

# ── Factor strategy ──────────────────────────────────────────────────────────
# bergson:  method=kfac + --ev_correction  →  kronfluence: strategy=ekfac
FACTOR_STRATEGY="ekfac"
FACTORS_NAME="ekfac"
ANALYSIS_NAME="kronfluence"
FACTOR_BATCH_SIZE=4
TORCH_DTYPE="float32"

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR="./hessian_simple/results"

# ── Module tracking ──────────────────────────────────────────────────────────
TRACKED_MODULES=""

ENV_PREFIX="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"

# ── Build & run command ──────────────────────────────────────────────────────
CMD="python -m hessian_simple.kronfluence_factors"
CMD+=" --model ${MODEL}"
CMD+=" --dataset ${DATASET}"
CMD+=" --split ${SPLIT}"
CMD+=" --text_column ${TEXT_COLUMN}"
CMD+=" --max_length ${MAX_LENGTH}"
CMD+=" --factor_strategy ${FACTOR_STRATEGY}"
CMD+=" --factors_name ${FACTORS_NAME}"
CMD+=" --factor_batch_size ${FACTOR_BATCH_SIZE}"
CMD+=" --torch_dtype ${TORCH_DTYPE}"
CMD+=" --output_dir ${OUTPUT_DIR}"
CMD+=" --analysis_name ${ANALYSIS_NAME}"
CMD+=" --overwrite"
CMD="${ENV_PREFIX} ${CMD}"

[[ -n "${TRACKED_MODULES}" ]] && CMD+=" --tracked_modules \"${TRACKED_MODULES}\""
[[ -n "${TRUNCATION}" ]]      && CMD+=" ${TRUNCATION}"

mkdir -p hessian_simple/results
echo "Running: ${CMD}"
eval "${CMD}" 2>&1 | tee hessian_simple/results/kronfluence.log
