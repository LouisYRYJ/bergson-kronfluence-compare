#!/usr/bin/env bash
set -euo pipefail

# ── User Configuration (mirrors bergson's run_bergson.sh) ────────────────────
MODEL="EleutherAI/pythia-14m-deduped"
QUERY_DATASET="NeelNanda/pile-10k"
TRAIN_DATASET="NeelNanda/pile-10k"
SCORES_NAME="identity_scores"

# ── Data ─────────────────────────────────────────────────────────────────────
QUERY_SPLIT="train"
TRAIN_SPLIT="train"
TEXT_COLUMN="text"
MAX_LENGTH=2048

# ── Sizes (0 = use full dataset) ────────────────────────────────────────────
QUERY_SIZE=0
TRAIN_SIZE=0

# ── Build & run command ──────────────────────────────────────────────────────
CMD="python -m raw_influence.kronfluence_scores"
CMD+=" --model ${MODEL}"
CMD+=" --query_dataset ${QUERY_DATASET}"
CMD+=" --train_dataset ${TRAIN_DATASET}"
CMD+=" --query_split ${QUERY_SPLIT}"
CMD+=" --train_split ${TRAIN_SPLIT}"
CMD+=" --text_column ${TEXT_COLUMN}"
CMD+=" --max_length ${MAX_LENGTH}"
CMD+=" --scores_name ${SCORES_NAME}"
CMD+=" --overwrite"

[[ "${QUERY_SIZE}" -gt 0 ]] && CMD+=" --query_size ${QUERY_SIZE}"
[[ "${TRAIN_SIZE}" -gt 0 ]] && CMD+=" --train_size ${TRAIN_SIZE}"

echo "Running: ${CMD}"
eval "${CMD}"
