#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

# ── User Configuration ───────────────────────────────────────────────────────
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DATASET="/mnt/ssd-1/louis/emergent_misalignment/data/merged_medical/merged_medical.jsonl"
RUN_PATH="$DIR/results/bergson"

# ── Data ─────────────────────────────────────────────────────────────────────
SPLIT="train"
PROMPT_COLUMN="prompt"
COMPLETION_COLUMN="completion"
CONVERSATION_COLUMN=""
TRUNCATION=""

# ── HessianConfig ────────────────────────────────────────────────────────────
METHOD="kfac"
HESSIAN_DTYPE="fp32"
EV_CORRECTION="--ev_correction"
USE_DATASET_LABELS="--use_dataset_labels"

# ── IndexConfig ──────────────────────────────────────────────────────────────
PRECISION="fp32"
TOKEN_BATCH_SIZE=4096
LOSS_FN="ce"
LOSS_REDUCTION="sum"
FSDP=""

# Only track last layer (23) — exclude layers 0-22 and lm_head
FILTER_MODULES="*layers.[0-9].*,*layers.1[0-9].*,*layers.2[0-2].*,*.lm_head"

ENV_PREFIX="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"

# ── Build & run command ──────────────────────────────────────────────────────
CMD="python -m common.bergson_hessian ${RUN_PATH}"
CMD+=" --model ${MODEL}"
CMD+=" --dataset ${DATASET}"
CMD+=" --split ${SPLIT}"
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
[[ -n "${EV_CORRECTION}" ]]         && CMD+=" ${EV_CORRECTION}"
[[ -n "${USE_DATASET_LABELS}" ]]    && CMD+=" ${USE_DATASET_LABELS}"
[[ -n "${TRUNCATION}" ]]            && CMD+=" ${TRUNCATION}"
[[ -n "${FSDP}" ]]                  && CMD+=" ${FSDP}"
[[ -n "${FILTER_MODULES}" ]]        && CMD+=" --filter_modules \"${FILTER_MODULES}\""

mkdir -p "$DIR/results"
echo "Running: ${ENV_PREFIX} ${CMD}"
eval "${ENV_PREFIX} ${CMD}" 2>&1 | tee "$DIR/results/bergson.log"
