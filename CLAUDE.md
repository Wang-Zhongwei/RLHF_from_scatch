# CLAUDE.md — project notes & reproduction

Two experiment tracks, each producing one figure:

- **Part 1 — Alignment on a learned reward model** → `results/alignment.png`
  (PPO/GRPO/DPO reward-vs-KL frontier + GRPO-vs-PPO sample efficiency).
- **Part 2 — GRPO on GSM8K with a verifiable reward** → `results/gsm8k_grpo.png`
  (from-scratch GRPO, benchmarked single-GPU / DDP / FSDP).

Both run on the SJSU **coe-hpc3** cluster.

## 0. Environment (coe-hpc3)
- Submit Slurm jobs **from coe-hpc3** (`ssh coe-hpc3`). GPU partition is `gpuqs`
  (`--gres=gpu:a100:1` / `gpu:h100:1`; 4-GPU nodes are `cs00[1-4]` = `a100:4`).
- Runtime comes from a module, no venv/pip:
  `module load python3/3.12.12 ml/torch/2.6` → torch 2.6+cu126, transformers 4.57, datasets,
  numpy, matplotlib. Use `python` from that module.
- **Compute nodes are air-gapped.** Jobs export
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1` and read a pre-populated
  shared HF cache (`~/.cache/huggingface`, on the shared NFS home).

## 1. Pre-cache models + datasets (once, from the login node coe-hpc1 — it has internet)
```bash
# coe-hpc1 only (login node); needs huggingface_hub on PATH (e.g. ~/.local/bin/hf).
python scripts/predownload.py
```
Fetches, into the shared HF cache: `gpt2`, `gpt2-medium`, `Qwen/Qwen2.5-0.5B-Instruct`
(models) and `Dahoas/rm-static`, `openai/gsm8k` (datasets). The login node's old glibc can't
build `datasets`/`pyarrow`, so this fetches the dataset *repo files* with `huggingface_hub`;
the compute-side loaders read the parquet shards directly (`rlhf.data._load_split`,
`rlhf.gsm8k.load_gsm8k`).

---

# Part 1 — Alignment on a learned reward model

### 1.1 Train the reward model  → `results/reward_model/`
```bash
sbatch scripts/train_reward.slurm                 # gpt2-medium, 10k pairs, 2 epochs, batch 8
# override via env, e.g.: sbatch --export=ALL,RM_MODEL=gpt2,RM_MAX=5000,RM_BS=8 scripts/train_reward.slurm
```
Full-backbone Bradley-Terry on `Dahoas/rm-static`; reports held-out pairwise accuracy
(≈ **0.68** for gpt2-medium; 0.5 = random) and peak VRAM/RAM. **Batch 16 OOMs on a 40 GB
A100** — use `RM_BS=8` on A100-40GB, or an H100-80GB for batch 16. `RM_FREEZE=1` trains only
a linear-probe head.

### 1.2 Alignment frontier + sample efficiency  → `results/frontier.json`, `results/sample_eff.json`
```bash
sbatch scripts/align.slurm    # loads results/reward_model (fails loudly if missing)
```
- exp1 = reward-vs-KL frontier for PPO / GRPO / DPO; exp2 = GRPO-vs-PPO reward-per-step.
- 200 steps, group 8, `--rl-lr 5e-6`, rollouts on **128 train-split** prompts, eval on a
  **disjoint 32-prompt test split** (no train/eval leak).
- `kl_coef` (additive KL penalty; PPO & GRPO) and DPO `beta` (temperature) are **different
  knobs — swept separately** (`--kl-coefs` vs `--dpo-betas`).
- **PPO warms its value head for 300 steps** (`--value-warmup 300 --vf-lr 1e-3`); its
  in-training explained variance still dips negative, evidence the critic is fragile.
- `align.slurm` renders the figure at the end (next step). Re-run one method only (merges into
  the JSON): `python -m experiments.exp1_alignment_frontier --methods ppo --out results/frontier.json`.

### 1.3 Figure  → `results/alignment.png`
```bash
python -m experiments.plot_alignment    # reads frontier.json + sample_eff.json
```
Three panels: **A** GRPO & PPO reward vs KL-penalty coefficient (higher penalty → less reward);
**B** DPO reward vs β (opposite direction — higher β chases reward); **C** GRPO vs PPO reward
per step (GRPO climbs; PPO's critic lags).

---

# Part 2 — GRPO on GSM8K (verifiable reward)

### 2.1 Run + figure  → `results/gsm8k_grpo.json`, `results/gsm8k_grpo.png`
```bash
sbatch scripts/gsm8k_grpo.slurm    # 4-GPU a100 node (cs00x); sweeps single / ddp / fsdp
```
From-scratch GRPO fine-tunes **Qwen2.5-0.5B-Instruct** on GSM8K, then re-runs the same loop
under three parallel strategies and renders `results/gsm8k_grpo.png` (via `plot_gsm8k`).

- **Verifiable reward, no reward model:** reward = "does the completion's final number match
  the gold answer" (`rlhf/gsm8k.py`), so the signal can't be gamed. This is the DeepSeek-R1
  recipe; the GRPO math itself is reused from `rlhf/grpo.py`.
- **`--strategy {single,ddp,fsdp}`** wraps the *same* training loop in nothing /
  DistributedDataParallel / FSDP (`rlhf/parallel/{ddp,fsdp}.py`) and reports peak memory + throughput.
- Results: held-out accuracy **32% → 46%** (single-GPU); peak mem/GPU single 17.7 / DDP 19.3 /
  **FSDP 11.2 GB** (ZeRO-3 shards optimizer state); DDP has the throughput lead.

### 2.2 Gotchas (encoded in the code)
- **Offline loading:** transformers 4.57 makes a network `model_info` call inside
  `from_pretrained` even under `HF_HUB_OFFLINE`. `rlhf/models.py:_resolve` resolves the local
  snapshot dir and loads from the path, sidestepping it on air-gapped nodes.
- **FSDP generation:** every forward is a collective all-gather, so all ranks must run identical
  forward counts. Rollouts use `sync_len=True` (fixed generation length, no data-dependent early
  stop) and equal per-rank eval shards (`rlhf/sampling.py`); otherwise NCCL aborts.

---

## Layout of the code
- `rlhf/` — from-scratch primitives: `reward.py` (RM + save/load), `ppo.py`, `grpo.py`,
  `preference.py` (DPO/IPO/KTO/ORPO/SimPO), `gsm8k.py` (GSM8K load + verifiable reward),
  `sampling.py` (FSDP-safe batched rollout + masked log-probs), `data.py`
  (`load_preference_dataset`/`load_prompt_set`), `parallel/` (`dist`, `ddp`, `fsdp`,
  `tensor_parallel`).
- `experiments/common.py` — **shared alignment harness**: `build_harness` (loads RM + policy +
  train/eval prompts), `sample_group`, `score_sequences`, `eval_reward_and_kl`, and the shared
  value-head PPO (`ppo_value_head_update`, `value_head_warmup`). Imported by exp1 and exp2.
- Experiments: `train_reward_model.py`, `exp1_alignment_frontier.py`,
  `exp2_grpo_sample_efficiency.py` (Part 1); `gsm8k_grpo.py` (Part 2);
  `tp_check.py` (multi-rank tensor-parallel correctness check, torchrun).
- Plotters: `plot_alignment.py` (→ `results/alignment.png`), `plot_gsm8k.py`
  (→ `results/gsm8k_grpo.png`).
- Scripts (`scripts/`): `predownload.py`, `train_reward.slurm`, `align.slurm`, `gsm8k_grpo.slurm`
  (the 4-GPU sweep also runs `experiments.tp_check`, the 4×A100 tensor-parallel check).

## Notes / gotchas
- Part 1 experiments run on GPU only if a real RM checkpoint exists (`--rm-dir`); with none they
  fall back to a synthetic tiny-gpt2 RM on CPU for hermetic CI (`pytest -q`, no downloads).
- The RM head is an unbounded linear scalar; reward values are arbitrary-scale, so only
  *differences* across methods (scored by the same RM) are meaningful. Part 2's reward is a
  0/1 correctness check, so its numbers are absolute (accuracy).
- `sacct` is disabled on this cluster — use `squeue` + the job logs (`logs/*_%j.out`).
