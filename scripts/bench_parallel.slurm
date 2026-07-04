#!/bin/bash
#SBATCH --job-name=rlhf-parallel
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench_%j.out

# Model-parallel scaling benchmark on coe-hpc3.
# Sweeps world_size 1 -> 4 and prints peak per-GPU memory + throughput so you can
# show the ~1/N memory curve that backs the "model-parallel" claim.
set -euo pipefail
module load cuda 2>/dev/null || true
source "${VENV:-.venv}/bin/activate"
mkdir -p logs results

for NPROC in 1 2 4; do
  echo "=== world_size=${NPROC} ==="
  torchrun --standalone --nproc_per_node=${NPROC} -m experiments.bench_parallel \
    --steps 20 --batch-size 8 --seq-len 128
done
