#!/bin/bash
#SBATCH --job-name=cgbench_grounding
#SBATCH --output=/home/av354855/projects/VideoDecisionAgents/VideoJev/slurmlogs/cgbench_grounding-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem80
#SBATCH --gres=gpu:1
#SBATCH --qos=group2-2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G              # decoding up to 1024 frames per video

module load anaconda3
module load cuda
source /home/av354855/projects/VideoDecisionAgents/VideoJev/.venv/bin/activate

export MPLBACKEND=Agg        # stop matplotlib
export TQDM_MININTERVAL=60

echo "Job started at $(date) on $(hostname)"
nvidia-smi

cd /home/av354855/projects/VideoDecisionAgents/VideoJev
# Repo root on PYTHONPATH + VideoJev runtime env vars
source scripts/env.sh

num_procs="${NPROC:-${SLURM_GPUS_ON_NODE:-1}}"   # one process per GPU

# Model: ZhuYuhan/VideoJev-4B  ZhuYuhan/VideoJev-35B-A3B
checkpoint="${CHECKPOINT:-ZhuYuhan/VideoJev-4B}"

# Dataset: cgbench  1219 vid / 12129 q   (all grounding rows)
manifest=data/benchmarks/cgbench/questions.jsonl

fps_list="${FPS_LIST:-2}"            # space-separated, e.g. "1 2"
max_frames="${MAX_FRAMES:-1024}"     # uniform frame cap per video
ctx="${CTX:-262144}"                 # grounding context; 262144 keeps native 480x832 frames at 1024 frames
limit="${LIMIT:-}"                   # e.g. LIMIT=3 -> 3 random videos (all their questions) for a smoke test
seed="${SEED:-0}"

limit_args=""
[ -n "${limit}" ] && limit_args="--limit-videos ${limit} --limit-seed ${seed}"

for fps in ${fps_list}; do
  out=runs/cgbench_grounding/fps${fps}_f${max_frames}_ctx${ctx}${limit:+_lim${limit}}-${SLURM_JOB_ID}

  echo "=== VideoJev | cgbench | ${fps} fps | ${max_frames} frames | ctx ${ctx} | ${out} ==="
  torchrun --standalone --nproc-per-node=${num_procs} scripts/evaluate_hf.py \
      --checkpoint ${checkpoint} \
      --benchmark custom \
      --manifest ${manifest} \
      --out ${out} \
      --grounding-context ${ctx} \
      --grounding-fps ${fps} \
      --max-frames ${max_frames} \
      ${limit_args}

  echo "metrics: ${out}/custom/metrics.json"
done

echo "Job finished at $(date)"
