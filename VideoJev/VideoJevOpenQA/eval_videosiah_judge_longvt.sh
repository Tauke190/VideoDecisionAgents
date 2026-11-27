#!/usr/bin/env bash
# Usage: bash VideoJevOpenQA/eval_videosiah_judge_longvt.sh path/to/predictions.jsonl
set -euo pipefail
PRED="$(realpath "$1")"
OUT="$(dirname "$PRED")/judge_longvt"
cd "$(dirname "${BASH_SOURCE[0]}")/.."


# JUDGE_MODEL="Qwen/Qwen3-235B-A22B" BACKEND="local"  

JUDGE_MODEL="Qwen/Qwen3-8B" BACKEND="local"  



 # paper: JUDGE_MODEL="Qwen/Qwen3-235B-A22B" BACKEND="openai" + OPENAI_BASE_URL
source .venv/bin/activate
source scripts/env.sh
torchrun --standalone --nproc-per-node=1 scripts/judge_videosiah_longvt.py \
  --predictions "$PRED" --backend "$BACKEND" --model "$JUDGE_MODEL" --out "$OUT"
