"""Experiment 2 — GRPO vs PPO sample efficiency.

Both methods optimize the same reward model from the same SFT start. We log mean
reward vs training step (and vs wall-clock), and peak memory. The expected story:
GRPO reaches comparable reward per step while carrying no value head / value
optimizer state, so it uses less memory and less compute per update.

    python -m experiments.exp2_grpo_sample_efficiency --steps 40 --out results/sample_eff.json

SKELETON: shares the online loop from exp1 but records a per-step reward trace.
Scale on GPU via --steps / --group-size and a larger model.
"""
import argparse
import json
import os

import torch

from rlhf import group_relative_advantages, grpo_loss
from experiments.common import (
    build_harness, sample_group, score_sequences, clone_policy, ppo_value_head_update,
)
from rlhf.parallel.fsdp import peak_memory_mb


def _seq_logps(model, seqs):
    lps = []
    for seq in seqs:
        ids = seq.unsqueeze(0)
        tgt = ids[0, 1:]
        rng = torch.arange(tgt.shape[0], device=ids.device)
        lps.append(model(ids).logits[0, :-1].log_softmax(-1)[rng, tgt].sum())
    return torch.stack(lps)


def run_grpo(policy, reference, reward_bundle, tokenizer, prompts, steps, group_size, mnt, lr=1e-5):
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    trace = []
    for step in range(steps):
        prompt = prompts[step % len(prompts)]
        seqs = sample_group(policy, tokenizer, prompt, group_size, mnt)
        rewards = score_sequences(reward_bundle, tokenizer, seqs)
        adv = group_relative_advantages(rewards)
        new_lp = _seq_logps(policy, seqs)
        out = grpo_loss(new_lp, new_lp.detach(), adv, kl_coef=0.0)
        opt.zero_grad(); out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        trace.append(rewards.mean().item())
    return trace


def run_ppo(policy, reference, reward_bundle, tokenizer, prompts, steps, group_size, mnt, lr=1e-5):
    # Real value-head PPO (same update as exp1, via the shared helper), so the two
    # experiments compare the *same* PPO. The value head is what GRPO drops.
    device = next(policy.parameters()).device
    hidden = getattr(policy.config, "n_embd", None) or policy.config.hidden_size
    value_head = torch.nn.Linear(hidden, 1).to(device)
    opt = torch.optim.AdamW(list(policy.parameters()) + list(value_head.parameters()), lr=lr)
    trace = []
    for step in range(steps):
        prompt = prompts[step % len(prompts)]
        trace.append(ppo_value_head_update(policy, value_head, reference, reward_bundle,
                                           tokenizer, prompt, opt, group_size, mnt, kl_coef=0.0))
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--rm-dir", default="results/reward_model")
    ap.add_argument("--dataset", default="Dahoas/rm-static")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--rl-lr", type=float, default=1e-5)
    ap.add_argument("--out", default="results/sample_eff.json")
    args = ap.parse_args()

    tokenizer, base, reference, reward_bundle, prompts = build_harness(
        rm_dir=args.rm_dir, dataset=args.dataset, model=args.model, n_prompts=args.n_prompts)

    results = {}
    for name, runner in [("grpo", run_grpo), ("ppo", run_ppo)]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        policy = clone_policy(base)
        trace = runner(policy, reference, reward_bundle, tokenizer, prompts,
                       args.steps, args.group_size, args.max_new_tokens, lr=args.rl_lr)
        results[name] = {"reward_trace": trace, "peak_mem_mb": peak_memory_mb()}
        print(f"[{name}] final reward={trace[-1]:+.3f} peak_mem={results[name]['peak_mem_mb']:.0f}MB")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
