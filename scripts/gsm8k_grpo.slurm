#!/bin/bash
#SBATCH --job-name=gsm8k-grpo
#SBATCH --partition=gpuqs
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/gsm8k_grpo_%j.out

# From-scratch GRPO on GSM8K (Qwen2.5-0.5B-Instruct), benchmarked across
# single-GPU / DDP / FSDP on a 4-GPU coe-hpc3 node (cs00x a100:4).
# Model + dataset are pre-cached on the shared HF cache; compute nodes are air-gapped.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

source /etc/profile.d/lmod.sh
module load python3/3.12.12 ml/torch/2.6

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p logs results

nvidia-smi || true

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
STEPS="${STEPS:-150}"
G="${G:-8}"
MNT="${MNT:-200}"
OUT="results/gsm8k_grpo.json"
rm -f "${OUT}"

run () {  # run <strategy> <nproc>
  local strat="$1" nproc="$2"
  echo "=== strategy=${strat} (nproc=${nproc}) ==="
  torchrun --standalone --nproc_per_node="${nproc}" -m experiments.gsm8k_grpo \
    --strategy "${strat}" --model "${MODEL}" \
    --steps "${STEPS}" --group-size "${G}" --max-new-tokens "${MNT}" --out "${OUT}"
}

# Each strategy is guarded so one failure doesn't abort the rest or the plot.
run single 1 || echo "!! single strategy failed"     # baseline: full model + opt on one GPU
run ddp    4 || echo "!! ddp strategy failed"         # replicated: same per-GPU mem, ~4x throughput
run fsdp   4 || echo "!! fsdp strategy failed"        # ZeRO-3: optimizer state sharded -> ~1/N mem

python -m experiments.plot_gsm8k || true
mkdir -p figures && cp -f results/gsm8k_grpo.png figures/ || true   # refresh README figure
