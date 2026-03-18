#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
MODEL="EleutherAI/pythia-14m-deduped"
DATASET="NeelNanda/pile-10k"
TEXT_COLUMN="text"
MAX_LENGTH=2048

FACTORS_NAME="ekfac"
SCORES_NAME="ekfac_scores"
ANALYSIS_NAME="kronfluence"
OUTPUT_DIR="./ekfac_scores/results"
FACTOR_BATCH_SIZE=4
QUERY_BATCH_SIZE=4
TRAIN_BATCH_SIZE=8

# ── Step 1: Fit EKFAC factors ───────────────────────────────────────────────
echo "=== Step 1: Fitting EKFAC factors ==="
CMD="python -m hessian_simple.kronfluence_factors"
CMD+=" --model ${MODEL}"
CMD+=" --dataset ${DATASET}"
CMD+=" --text_column ${TEXT_COLUMN}"
CMD+=" --max_length ${MAX_LENGTH}"
CMD+=" --factors_name ${FACTORS_NAME}"
CMD+=" --factor_strategy ekfac"
CMD+=" --factor_batch_size ${FACTOR_BATCH_SIZE}"
CMD+=" --torch_dtype float32"
CMD+=" --output_dir ${OUTPUT_DIR}"
CMD+=" --analysis_name ${ANALYSIS_NAME}"
CMD+=" --truncation"
CMD+=" --overwrite"

mkdir -p ekfac_scores/results
LOG="ekfac_scores/results/kronfluence.log"

echo "Running: ${CMD}"
eval "${CMD}" 2>&1 | tee "${LOG}"

# ── Step 2: Compute scores ──────────────────────────────────────────────────
echo "=== Step 2: Computing EKFAC scores ==="
CMD="python -m ekfac_scores.kronfluence_scores"
CMD+=" --model ${MODEL}"
CMD+=" --query_dataset ${DATASET}"
CMD+=" --train_dataset ${DATASET}"
CMD+=" --text_column ${TEXT_COLUMN}"
CMD+=" --max_length ${MAX_LENGTH}"
CMD+=" --factors_name ${FACTORS_NAME}"
CMD+=" --analysis_name ${ANALYSIS_NAME}"
CMD+=" --scores_name ${SCORES_NAME}"
CMD+=" --output_dir ${OUTPUT_DIR}"
CMD+=" --query_batch_size ${QUERY_BATCH_SIZE}"
CMD+=" --train_batch_size ${TRAIN_BATCH_SIZE}"
CMD+=" --torch_dtype float32"
CMD+=" --truncation"
CMD+=" --overwrite"

echo "Running: ${CMD}"
eval "${CMD}" 2>&1 | tee -a "${LOG}"
