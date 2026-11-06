
source scripts/env.sh
torchrun --standalone --nproc-per-node="${NPROC:-1}" scripts/evaluate_hf.py \
  --checkpoint ZhuYuhan/VideoJev-4B \
  --benchmark custom --manifest data/benchmarks/cgbench/questions.jsonl --out "${OUT:-runs/cgbench}" \
  --qa-context 96000