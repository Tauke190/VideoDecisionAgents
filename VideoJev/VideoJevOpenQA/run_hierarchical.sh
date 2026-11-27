#!/usr/bin/env bash
# Hierarchical sampling: VideoJev zooms until confident, Qwen3.5-4B answers from that window
# Usage: LIMIT=10 bash VideoJevOpenQA/run_hierarchical.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source scripts/env.sh

LIMIT=${LIMIT:-0}
NPROC=${NPROC:-3}
FRAMES=${FRAMES:-128}
CONF=${CONF:-0.8}
MAX_DEPTH=${MAX_DEPTH:-3}
ANSWER_FRAMES=${ANSWER_FRAMES:-16}
PEAKS=${PEAKS:-3}
PROMPT=${PROMPT:-short}
PAD=${PAD:-1}
TOP=${TOP:-2}
MAX_NODES=${MAX_NODES:-10}
OUT=${OUT:-VideoJevOpenQA/runs/hier_limit${LIMIT}_f${FRAMES}_c${CONF}_p${PEAKS}_${PROMPT}_pad${PAD}_top${TOP}}

[ "${REDO:-0}" = 1 ] && rm -rf "$OUT"

torchrun --standalone --nproc-per-node=$NPROC VideoJevOpenQA/hierarchical_videosiah.py \
  --limit $LIMIT \
  --frames $FRAMES \
  --conf-threshold $CONF \
  --max-depth $MAX_DEPTH \
  --peaks $PEAKS \
  --prompt $PROMPT \
  --pad $PAD \
  --top-leaves $TOP \
  ${CLIPS:+--clips} \
  --max-nodes $MAX_NODES \
  ${REUSE:+--reuse $REUSE} \
  --answer-frames $ANSWER_FRAMES \
  --out $OUT

# optional judge
if [ "${JUDGE:-0}" = 1 ]; then
  bash VideoJevOpenQA/eval_videosiah_judge_longvt.sh $OUT/predictions.jsonl
fi
