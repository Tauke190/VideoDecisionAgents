#!/bin/bash
#SBATCH --job-name=judge_qwen235b
#SBATCH --output=/home/av354855/projects/VideoDecisionAgents/VideoJev/slurmlogs/judge_qwen235b-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem80
#SBATCH --gres=gpu:4
#SBATCH --qos=group2-2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=200G
# Usage: sbatch VideoJevOpenQA/judge_longvt_235b.slurm path/to/predictions.jsonl
# One-time setup (login node): uv venv ~/venvs/vllm --python 3.12 && uv pip install -p ~/venvs/vllm vllm==0.12.0
set -euo pipefail
PRED="$(realpath "$1")"
OUT="$(dirname "$PRED")/judge_longvt"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-235B-A22B-FP8}"   # bf16 Qwen/Qwen3-235B-A22B needs --gres=gpu:8
CLIENTS="${CLIENTS:-32}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
ROOT=/home/av354855/projects/VideoDecisionAgents/VideoJev

module load cuda
cd "$ROOT"
echo "Job started at $(date) on $(hostname)"
nvidia-smi

"${VLLM_VENV:-$HOME/venvs/vllm}/bin/vllm" serve "$JUDGE_MODEL" --served-model-name "$JUDGE_MODEL" \
  --tensor-parallel-size "$SLURM_GPUS_ON_NODE" --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --port "$PORT" > "slurmlogs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID' EXIT
until curl -sf "http://localhost:$PORT/health" > /dev/null; do
  kill -0 "$VLLM_PID" || { echo "vLLM died, see slurmlogs/vllm-$SLURM_JOB_ID.log" >&2; exit 1; }
  sleep 15
done
echo "vLLM ready at $(date)"

source .venv/bin/activate
source scripts/env.sh
OPENAI_BASE_URL="http://localhost:$PORT/v1" OPENAI_API_KEY=EMPTY \
  torchrun --standalone --nproc-per-node="$CLIENTS" scripts/judge_videosiah_longvt.py \
  --predictions "$PRED" --backend openai --model "$JUDGE_MODEL" --out "$OUT"
echo "Job finished at $(date)"
