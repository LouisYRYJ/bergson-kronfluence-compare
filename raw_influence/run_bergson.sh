#!/usr/bin/env bash
set -euo pipefail

# ── User Configuration ───────────────────────────────────────────────────────
MODEL="EleutherAI/pythia-14m-deduped"
QUERY_DATASET="NeelNanda/pile-10k"
INDEX_DATASET="NeelNanda/pile-10k"
RUN_PATH="raw_influence/results/bergson"

# ── Data ─────────────────────────────────────────────────────────────────────
PROMPT_COLUMN="text"
COMPLETION_COLUMN=""
CONVERSATION_COLUMN=""
TRUNCATION="--truncation"

# ── IndexConfig ──────────────────────────────────────────────────────────────
PRECISION="fp32"
PROJECTION_DIM=0
TOKEN_BATCH_SIZE=2048
LOSS_FN="ce"
LOSS_REDUCTION="sum"
FSDP=""
FILTER_MODULES=""

ENV_PREFIX="CUDA_VISIBLE_DEVICES=0"

# ── Build & run command ──────────────────────────────────────────────────────
CMD="python -m raw_influence.bergson_pipeline ${RUN_PATH}"
CMD+=" --query_dataset ${QUERY_DATASET}"
CMD+=" --index_dataset ${INDEX_DATASET}"
CMD+=" --model ${MODEL}"
CMD+=" --prompt_column ${PROMPT_COLUMN}"
CMD+=" --precision ${PRECISION}"
CMD+=" --projection_dim ${PROJECTION_DIM}"
CMD+=" --token_batch_size ${TOKEN_BATCH_SIZE}"
CMD+=" --loss_fn ${LOSS_FN}"
CMD+=" --loss_reduction ${LOSS_REDUCTION}"
CMD+=" --overwrite"
CMD+=" --skip_preconditioners"

# Optional flags – appended only when non-empty
[[ -n "${COMPLETION_COLUMN}" ]]     && CMD+=" --completion_column ${COMPLETION_COLUMN}"
[[ -n "${CONVERSATION_COLUMN}" ]]   && CMD+=" --conversation_column ${CONVERSATION_COLUMN}"
[[ -n "${FILTER_MODULES}" ]]        && CMD+=" --filter_modules \"${FILTER_MODULES}\""
[[ -n "${TRUNCATION}" ]]            && CMD+=" ${TRUNCATION}"
[[ -n "${FSDP}" ]]                  && CMD+=" ${FSDP}"

echo "Running: ${ENV_PREFIX} ${CMD}"
eval "${ENV_PREFIX} ${CMD}"
