#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$DIR/results"

CUDA_VISIBLE_DEVICES=1 bash "$DIR/run_bergson.sh" > "$DIR/results/bergson.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 bash "$DIR/run_kronfluence.sh" > "$DIR/results/kronfluence.log" 2>&1 &

wait
