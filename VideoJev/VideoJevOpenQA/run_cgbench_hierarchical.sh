#!/usr/bin/env bash
# CG-Bench QA: hierarchical sampling + Qwen multiple choice, then open-ended + judge on the same trees
# Usage: bash VideoJevOpenQA/run_cgbench_hierarchical.sh  (SAMPLE=100 by default)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source scripts/env.sh

QUESTIONS=${QUESTIONS:-data/benchmarks/cgbench/questions.jsonl}  # videos/ must sit next to it
SAMPLE=${SAMPLE:-100}
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
FRAMES=${FRAMES:-128}
CONF=${CONF:-0.8}
MAX_DEPTH=${MAX_DEPTH:-3}
PEAKS=${PEAKS:-3}
TOP=${TOP:-2}
MAX_NODES=${MAX_NODES:-10}
OUT=${OUT:-VideoJevOpenQA/runs/cgbench_qa_s${SAMPLE}}

[ "${REDO:-0}" = 1 ] && rm -rf "$OUT"

# 1. tree + multiple choice
torchrun --standalone --nproc-per-node=$NPROC VideoJevOpenQA/hierarchical_videosiah.py \
  --questions $QUESTIONS \
  --sample $SAMPLE \
  --frames $FRAMES \
  --conf-threshold $CONF \
  --max-depth $MAX_DEPTH \
  --peaks $PEAKS \
  --top-leaves $TOP \
  ${CLIPS:+--clips} \
  --max-nodes $MAX_NODES \
  --prompt mcq \
  --max-new-tokens 8 \
  --out $OUT

# 2. open-ended answers from the same trees
torchrun --standalone --nproc-per-node=$NPROC VideoJevOpenQA/hierarchical_videosiah.py \
  --questions $QUESTIONS \
  --sample $SAMPLE \
  --reuse $OUT/predictions.jsonl \
  --top-leaves $TOP \
  ${CLIPS:+--clips} \
  --prompt short \
  --out $OUT/open

# 3. judge open-ended
bash VideoJevOpenQA/eval_videosiah_judge_longvt.sh $OUT/open/predictions.jsonl

# 4. metrics
python VideoJevOpenQA/eval_cgbench_qa.py $OUT
