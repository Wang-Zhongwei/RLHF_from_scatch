#!/bin/bash
#SBATCH --job-name=rlhf-parallel
#SBATCH --partition=gpuqs
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench_%j.out

# Model-parallel scaling benchmark on coe-hpc3 (needs a 4-GPU node, e.g. cs00x a100:4).
# Sweeps world_size 1 -> 4 and prints peak per-GPU memory + throughput so you can
# show the ~1/N memory curve that backs the "model-parallel" claim.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

source /etc/profile.d/lmod.sh
module load python3/3.12.12 ml/torch/2.6

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs results

nvidia-smi || true

for NPROC in 1 2 4; do
  echo "=== world_size=${NPROC} ==="
  torchrun --standalone --nproc_per_node=${NPROC} -m experiments.bench_parallel \
    --steps 20 --batch-size 8 --seq-len 128
done
