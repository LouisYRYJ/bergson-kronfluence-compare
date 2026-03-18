#!/usr/bin/env bash
set -euo pipefail

# ── User Configuration ───────────────────────────────────────────────────────
MODEL="EleutherAI/pythia-14m-deduped"
QUERY_DATASET="NeelNanda/pile-10k"
INDEX_DATASET="NeelNanda/pile-10k"
RUN_PATH="ekfac_scores/results/bergson"

# ── Data ─────────────────────────────────────────────────────────────────────
PROMPT_COLUMN="text"
COMPLETION_COLUMN=""
CONVERSATION_COLUMN=""
TRUNCATION="--truncation"

# ── HessianConfig ────────────────────────────────────────────────────────────
HESSIAN_DTYPE="fp32"
USE_DATASET_LABELS="--use_dataset_labels"

# ── IndexConfig ──────────────────────────────────────────────────────────────
PRECISION="fp32"
TOKEN_BATCH_SIZE=2048
LOSS_FN="ce"
LOSS_REDUCTION="sum"
FSDP=""
FILTER_MODULES=""

# ── EKFAC ────────────────────────────────────────────────────────────────────
LAMBDA_DAMP_FACTOR=0.1

ENV_PREFIX="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"

# ── Build & run command ──────────────────────────────────────────────────────
CMD="python -m ekfac_scores.bergson_pipeline ${RUN_PATH}"
CMD+=" --query_dataset ${QUERY_DATASET}"
CMD+=" --index_dataset ${INDEX_DATASET}"
CMD+=" --model ${MODEL}"
CMD+=" --prompt_column ${PROMPT_COLUMN}"
CMD+=" --hessian_dtype ${HESSIAN_DTYPE}"
CMD+=" --precision ${PRECISION}"
CMD+=" --token_batch_size ${TOKEN_BATCH_SIZE}"
CMD+=" --loss_fn ${LOSS_FN}"
CMD+=" --loss_reduction ${LOSS_REDUCTION}"
CMD+=" --lambda_damp_factor ${LAMBDA_DAMP_FACTOR}"
CMD+=" --overwrite"
CMD+=" --skip_preconditioners"

# Optional flags – appended only when non-empty
[[ -n "${COMPLETION_COLUMN}" ]]     && CMD+=" --completion_column ${COMPLETION_COLUMN}"
[[ -n "${CONVERSATION_COLUMN}" ]]   && CMD+=" --conversation_column ${CONVERSATION_COLUMN}"
[[ -n "${FILTER_MODULES}" ]]        && CMD+=" --filter_modules \"${FILTER_MODULES}\""
[[ -n "${USE_DATASET_LABELS}" ]]    && CMD+=" ${USE_DATASET_LABELS}"
[[ -n "${TRUNCATION}" ]]            && CMD+=" ${TRUNCATION}"
[[ -n "${FSDP}" ]]                  && CMD+=" ${FSDP}"

mkdir -p ekfac_scores/results
echo "Running: ${ENV_PREFIX} ${CMD}"
eval "${ENV_PREFIX} ${CMD}" 2>&1 | tee ekfac_scores/results/bergson.log
