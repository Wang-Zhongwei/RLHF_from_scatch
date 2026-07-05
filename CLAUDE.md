# CLAUDE.md — project notes & reproduction

## Reproduce: reward model → alignment frontier → figures

The alignment experiments (reward modeling, the PPO/GRPO/DPO reward-vs-KL frontier,
sample efficiency, and FSDP scaling) run on the SJSU **coe-hpc3** cluster. End to end:

### 0. Environment (coe-hpc3)
- Submit Slurm jobs **from coe-hpc3** (`ssh coe-hpc3`). GPU partition is `gpuqs`
  (`--gres=gpu:a100:1`, or `gpu:h100:1`; 4-GPU nodes are `cs00[1-4]` = `a100:4`).
- Runtime comes from a module, no venv/pip:
  `module load python3/3.12.12 ml/torch/2.6` → torch 2.6+cu126, transformers, datasets,
  numpy, matplotlib. Use `python` from that module.
- **Compute nodes are air-gapped.** Jobs export
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1` and read a pre-populated
  shared HF cache (`~/.cache/huggingface`, on the shared NFS home).

### 1. Pre-cache models + dataset (once, from the login node coe-hpc1 — it has internet)
```bash
# coe-hpc1 only (login node); needs huggingface_hub (pip install --user huggingface_hub)
python scripts/predownload.py     # gpt2, gpt2-medium, and Dahoas/rm-static -> shared cache
```
Note: the login node's old glibc can't build `datasets`/`pyarrow`, so `predownload.py`
fetches the dataset *repo files* with `huggingface_hub`; the compute-side loader
(`rlhf.data._load_split`) reads those parquet shards directly (see `load_preference_dataset`).

### 2. Train the reward model  → `results/reward_model/`
```bash
sbatch scripts/train_reward.slurm                 # gpt2-medium, 10k pairs, 2 epochs, batch 8
# override via env, e.g.: sbatch --export=ALL,RM_MODEL=gpt2,RM_MAX=5000,RM_BS=8 scripts/train_reward.slurm
```
- Full-backbone Bradley-Terry on `Dahoas/rm-static`; reports held-out pairwise accuracy
  (≈ **0.68** for gpt2-medium; 0.5 = random) and peak VRAM/RAM.
- **Batch 16 OOMs on a 40 GB A100** — use `RM_BS=8` on A100-40GB (≈15 GB at batch 4),
  or an H100-80GB for batch 16. `RM_FREEZE=1` trains only a linear-probe head.

### 3. Alignment frontier + sample efficiency  → `results/frontier.json`, `results/sample_eff.json`
```bash
sbatch scripts/train_hpc.slurm    # loads results/reward_model (fails loudly if missing)
```
- exp1 = reward-vs-KL frontier for PPO / GRPO / DPO; exp2 = GRPO-vs-PPO reward-per-step + peak mem.
- 200 steps, group 8, `--rl-lr 5e-6`, rollouts on **128 train-split** prompts, eval on a
  **disjoint 32-prompt test split** (no train/eval leak).
- `kl_coef` (additive KL penalty; PPO & GRPO) and DPO `beta` (temperature) are **different
  knobs — swept separately** (`--kl-coefs` vs `--dpo-betas`).
- **PPO warms its value head for 300 steps** (`--value-warmup 300 --vf-lr 1e-3`); at 30 steps
  the critic's explained variance is negative (worse than a constant baseline).
- Re-run one method only (merges into the existing JSON):
  `python -m experiments.exp1_alignment_frontier --methods ppo --value-warmup 300 --out results/frontier.json`

### 4. FSDP scaling benchmark  → `results/bench_parallel.json`
```bash
sbatch --gres=gpu:a100:4 scripts/bench_parallel.slurm   # sweeps world_size 1,2,4
```
Peak memory/GPU falls (ZeRO-3 shards optimizer state), e.g. gpt2-medium 9.0 → 4.9 → 3.6 GB.

### 5. Render the results figure  → `figures/showcase.png` (committed, shown in README)
```bash
python -m experiments.plot_showcase       # reads results/{reward_model,frontier,sample_eff,bench_parallel}
cp results/showcase.png figures/showcase.png
```

## Layout of the alignment code
- `rlhf/` — from-scratch primitives: `reward.py` (RM + save/load), `ppo.py`, `grpo.py`,
  `preference.py` (DPO…), `data.py` (`load_preference_dataset`/`load_prompt_set`), `parallel/` (FSDP/TP).
- `experiments/common.py` — **shared harness**: `build_harness` (loads RM + policy + train/eval
  prompts, on GPU), `sample_group`, `score_sequences`, `eval_reward_and_kl`, and the shared
  value-head PPO (`ppo_value_head_update`, `value_head_warmup`). Both exp1 and exp2 import it.
- `experiments/train_reward_model.py`, `exp1_alignment_frontier.py`,
  `exp2_grpo_sample_efficiency.py`, `bench_parallel.py`, `plot_showcase.py`.

## Notes / gotchas
- The experiments run on GPU only if a real RM checkpoint exists (`--rm-dir`); with none they
  fall back to a synthetic tiny-gpt2 RM on CPU for hermetic CI (`pytest -q`, no downloads).
- The RM head is an unbounded linear scalar (standard); reward values are arbitrary-scale, so
  only *differences* across methods (scored by the same RM) are meaningful.
- `sacct` is disabled on this cluster — use `squeue` + the job logs (`logs/*_%j.out`) to track runs.
