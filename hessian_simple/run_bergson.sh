#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

# ── User Configuration ───────────────────────────────────────────────────────
MODEL="EleutherAI/pythia-14m-deduped"
DATASET="NeelNanda/pile-10k"
RUN_PATH="$DIR/results/bergson"

# ── Data ─────────────────────────────────────────────────────────────────────
PROMPT_COLUMN="text"
COMPLETION_COLUMN=""
CONVERSATION_COLUMN=""
TRUNCATION="--truncation"

# ── HessianConfig ────────────────────────────────────────────────────────────
METHOD="kfac"
HESSIAN_DTYPE="fp32"
EV_CORRECTION="--ev_correction"
USE_DATASET_LABELS="--use_dataset_labels"

# ── IndexConfig ──────────────────────────────────────────────────────────────
PRECISION="fp32"
TOKEN_BATCH_SIZE=2048
LOSS_FN="ce"
LOSS_REDUCTION="sum"
FSDP=""
FILTER_MODULES=""

ENV_PREFIX="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"

# ── Build & run command ──────────────────────────────────────────────────────
CMD="python -m common.bergson_hessian ${RUN_PATH}"
CMD+=" --model ${MODEL}"
CMD+=" --dataset ${DATASET}"
CMD+=" --prompt_column ${PROMPT_COLUMN}"
CMD+=" --method ${METHOD}"
CMD+=" --hessian_dtype ${HESSIAN_DTYPE}"
CMD+=" --precision ${PRECISION}"
CMD+=" --token_batch_size ${TOKEN_BATCH_SIZE}"
CMD+=" --loss_fn ${LOSS_FN}"
CMD+=" --loss_reduction ${LOSS_REDUCTION}"
CMD+=" --overwrite"

# Optional flags – appended only when non-empty
[[ -n "${COMPLETION_COLUMN}" ]]     && CMD+=" --completion_column ${COMPLETION_COLUMN}"
[[ -n "${CONVERSATION_COLUMN}" ]]   && CMD+=" --conversation_column ${CONVERSATION_COLUMN}"
[[ -n "${FILTER_MODULES}" ]]        && CMD+=" --filter_modules \"${FILTER_MODULES}\""
[[ -n "${EV_CORRECTION}" ]]         && CMD+=" ${EV_CORRECTION}"
[[ -n "${TRUNCATION}" ]]            && CMD+=" ${TRUNCATION}"
[[ -n "${FSDP}" ]]                  && CMD+=" ${FSDP}"
[[ -n "${USE_DATASET_LABELS}" ]]    && CMD+=" ${USE_DATASET_LABELS}"

mkdir -p hessian_simple/results
echo "Running: ${ENV_PREFIX} ${CMD}"
eval "${ENV_PREFIX} ${CMD}" 2>&1 | tee hessian_simple/results/bergson.log
