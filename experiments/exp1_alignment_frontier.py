"""Experiment 1 — the reward-vs-KL alignment frontier.

Aligns one shared SFT checkpoint with PPO, GRPO, and DPO, then plots each on
mean-reward (y) vs KL-from-reference (x). The story: methods that reach higher
reward at lower KL sit up-and-to-the-left — they buy more alignment per unit of
distribution drift. Run over several KL/beta settings to trace each method's
curve.

    python -m experiments.exp1_alignment_frontier --steps 40 --out results/frontier.json

This is a SKELETON: the per-method aligners below implement a minimal online
loop that runs end-to-end on tiny-gpt2/CPU. Scale on GPU by raising
--steps/--group-size/--max-new-tokens and swapping the model in rlhf.models.
"""
import argparse
import json
import os

import torch

from rlhf import (
    grpo_loss, group_relative_advantages,
    ppo_loss, sequence_logprob, gae_advantages,
    dpo_loss, batch_sequence_logprob,
)
from experiments.common import (
    build_harness, sample_group, score_sequences, eval_reward_and_kl, clone_policy,
)


def align_with_grpo(policy, reference, reward_bundle, tokenizer, prompts,
                    steps, group_size, max_new_tokens, kl_coef, lr=1e-5, clip_eps=0.2):
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    for step in range(steps):
        prompt = prompts[step % len(prompts)]
        seqs = sample_group(policy, tokenizer, prompt, group_size, max_new_tokens)
        rewards = score_sequences(reward_bundle, tokenizer, seqs)
        adv = group_relative_advantages(rewards)          # critic-free baseline

        new_lp, old_lp, ref_lp = [], [], []
        for seq in seqs:
            ids = seq.unsqueeze(0)
            tgt = ids[0, 1:]
            new_lp.append(policy(ids).logits[0, :-1].log_softmax(-1)[torch.arange(tgt.shape[0]), tgt].sum())
            with torch.no_grad():
                old_lp.append(new_lp[-1].detach())
                ref_lp.append(reference(ids).logits[0, :-1].log_softmax(-1)[torch.arange(tgt.shape[0]), tgt].sum())
        out = grpo_loss(torch.stack(new_lp), torch.stack(old_lp), adv,
                        ref_logprobs=torch.stack(ref_lp), clip_eps=clip_eps, kl_coef=kl_coef)
        opt.zero_grad(); out["loss"].backward(); opt.step()
    return policy


def align_with_ppo(policy, reference, reward_bundle, tokenizer, prompts,
                   steps, group_size, max_new_tokens, kl_coef, lr=1e-5):
    # TODO(gpu): add a value head + GAE targets; here we use reward-to-go as the
    # advantage baseline so the skeleton runs without a separate critic net.
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    for step in range(steps):
        prompt = prompts[step % len(prompts)]
        seqs = sample_group(policy, tokenizer, prompt, group_size, max_new_tokens)
        rewards = score_sequences(reward_bundle, tokenizer, seqs)
        adv = rewards - rewards.mean()
        loss = 0.0
        for seq, a in zip(seqs, adv):
            ids = seq.unsqueeze(0)
            tgt = ids[0, 1:]
            lp = policy(ids).logits[0, :-1].log_softmax(-1)[torch.arange(tgt.shape[0]), tgt].sum()
            loss = loss - a.detach() * lp
        opt.zero_grad(); (loss / len(seqs)).backward(); opt.step()
    return policy


def align_with_dpo(policy, reference, reward_bundle, tokenizer, prompts,
                   steps, group_size, max_new_tokens, beta, lr=1e-5):
    # Build on-policy pairs: best vs worst sampled completion by reward model.
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    for step in range(steps):
        prompt = prompts[step % len(prompts)]
        seqs = sample_group(policy, tokenizer, prompt, group_size, max_new_tokens)
        rewards = score_sequences(reward_bundle, tokenizer, seqs)
        chosen, rejected = seqs[rewards.argmax()], seqs[rewards.argmin()]

        def logp(model, seq):
            ids = seq.unsqueeze(0)
            tgt = ids[0, 1:]
            return model(ids).logits[0, :-1].log_softmax(-1)[torch.arange(tgt.shape[0]), tgt].sum()

        with torch.no_grad():
            ref_c, ref_r = logp(reference, chosen), logp(reference, rejected)
        loss = dpo_loss(logp(policy, chosen).unsqueeze(0), logp(policy, rejected).unsqueeze(0),
                        ref_c.unsqueeze(0), ref_r.unsqueeze(0), beta=beta)
        opt.zero_grad(); loss.backward(); opt.step()
    return policy


METHODS = {
    "grpo": align_with_grpo,
    "ppo": align_with_ppo,
    "dpo": align_with_dpo,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.02, 0.1, 0.5])
    ap.add_argument("--out", default="results/frontier.json")
    args = ap.parse_args()

    tokenizer, base_policy, reference, reward_bundle, prompts = build_harness()
    r0, kl0 = eval_reward_and_kl(base_policy, reference, reward_bundle, tokenizer, prompts,
                                 args.group_size, args.max_new_tokens)
    frontier = {"base": {"reward": r0, "kl": kl0}, "methods": {}}

    for name, aligner in METHODS.items():
        frontier["methods"][name] = []
        for beta in args.betas:  # beta doubles as KL coef for PPO/GRPO
            policy = clone_policy(base_policy)
            aligner(policy, reference, reward_bundle, tokenizer, prompts,
                    args.steps, args.group_size, args.max_new_tokens, beta)
            reward, kl = eval_reward_and_kl(policy, reference, reward_bundle, tokenizer, prompts,
                                            args.group_size, args.max_new_tokens)
            frontier["methods"][name].append({"beta": beta, "reward": reward, "kl": kl})
            print(f"[{name}] beta={beta:<5} reward={reward:+.3f} kl={kl:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(frontier, f, indent=2)
    print(f"Wrote {args.out}. Plot with experiments/plot_frontier.py")


if __name__ == "__main__":
    main()
