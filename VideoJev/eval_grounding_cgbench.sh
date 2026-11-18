source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
source scripts/env.sh
# Quick test: LIMIT=5 bash eval_grounding_cgbench.sh   -> 5 random videos (all their questions), SEED=0 by default
# FPS defaults to VideoJev's official grounding rate (4); override with FPS=2 etc.
torchrun --standalone --nproc-per-node="${NPROC:-1}" scripts/evaluate_hf.py \
  --checkpoint ZhuYuhan/VideoJev-4B \
  --benchmark custom --manifest data/benchmarks/cgbench/questions.jsonl --out "${OUT:-runs/cgbench}" \
  --grounding-context 60000 --grounding-fps "${FPS:-1}" --max-frames 512 \
  ${LIMIT:+--limit-videos "$LIMIT" --limit-seed "${SEED:-0}"} "$@"
