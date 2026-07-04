"""Experiment 1 — the reward-vs-KL alignment frontier.

Aligns one shared SFT checkpoint with PPO, GRPO, and DPO, then plots each on
mean-reward (y) vs KL-from-reference (x). The story: methods that reach higher
reward at lower KL sit up-and-to-the-left — they buy more alignment per unit of
distribution drift. Run over several KL/beta settings to trace each method's
curve.

    # real reward model (GPU): train it first, then load via --rm-dir
    python -m experiments.train_reward_model --model gpt2-medium --out results/reward_model
    python -m experiments.exp1_alignment_frontier --rm-dir results/reward_model \
        --kl-coefs 0.0 0.01 0.05 0.1 --dpo-betas 0.05 0.1 0.3 0.5 --steps 200

With no --rm-dir checkpoint it falls back to the synthetic tiny-gpt2 RM (hermetic
CI). kl_coef (PPO/GRPO additive KL penalty) and beta (DPO temperature) are swept
separately because they are not the same quantity.
"""
import argparse
import json
import os

import torch

from rlhf import (
    grpo_loss, group_relative_advantages, dpo_loss, load_preference_dataset,
)
from experiments.common import (
    build_harness, sample_group, score_sequences, eval_reward_and_kl, clone_policy,
    ppo_value_head_update, value_head_warmup, save_aligned_policy,
)


def align_with_grpo(policy, reference, reward_bundle, tokenizer, prompts,
                    steps, group_size, max_new_tokens, kl_coef, lr=1e-5, clip_eps=0.2,
                    pref_pairs=None, save_dir=None):
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
            rng = torch.arange(tgt.shape[0], device=ids.device)
            # length-normalized (mean) seq log-prob keeps the k3 KL term O(1) per token;
            # summing makes exp(ref_sum - new_sum) overflow to nan with real models.
            new_lp.append(policy(ids).logits[0, :-1].log_softmax(-1)[rng, tgt].mean())
            with torch.no_grad():
                old_lp.append(new_lp[-1].detach())
                ref_lp.append(reference(ids).logits[0, :-1].log_softmax(-1)[rng, tgt].mean())
        out = grpo_loss(torch.stack(new_lp), torch.stack(old_lp), adv,
                        ref_logprobs=torch.stack(ref_lp), clip_eps=clip_eps, kl_coef=kl_coef)
        opt.zero_grad(); out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    if save_dir:
        save_aligned_policy(save_dir, policy, meta={"method": "grpo", "kl_coef": kl_coef})
    return policy


def align_with_ppo(policy, reference, reward_bundle, tokenizer, prompts,
                   steps, group_size, max_new_tokens, kl_coef, lr=1e-5,
                   clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, gamma=0.99, lam=0.95,
                   pref_pairs=None, value_warmup=0, vf_lr=1e-3, save_dir=None):
    # Faithful PPO: a learned value head supplies per-token GAE baselines and the
    # update is the scaffold's clipped surrogate + value loss + entropy bonus
    # (rlhf.ppo_loss), plus a k3 KL penalty to the frozen reference weighted by
    # kl_coef (=beta). The critic is exactly what GRPO drops for its group
    # baseline -- see rlhf/grpo.py and exp2's memory story.
    #
    # NOTE: PPO does not *need* a group -- V(s) is its baseline, so one completion
    # per step suffices. We sample a group here only for parity with GRPO (equal
    # rollout budget/step -> comparable frontier & sample efficiency); the group
    # supplies extra tokens, never the baseline.
    device = next(policy.parameters()).device
    hidden = getattr(policy.config, "n_embd", None) or policy.config.hidden_size
    value_head = torch.nn.Linear(hidden, 1).to(device)
    opt = torch.optim.AdamW(
        list(policy.parameters()) + list(value_head.parameters()), lr=lr)
    # Warm the critic (policy frozen, V -> MC returns) before any policy update, so early
    # GAE advantages use a calibrated baseline instead of a random head.
    warm = value_head_warmup(policy, value_head, reward_bundle, tokenizer, prompts,
                             value_warmup, group_size, max_new_tokens, vf_lr=vf_lr, gamma=gamma)
    if warm is not None:
        print(f"[ppo] value-head warmup ({value_warmup} steps) vf_loss={warm['vf_loss']:.4f} "
              f"EV(mean V)={warm['ev']:+.3f} (target MC={warm['target']:+.3f})", flush=True)
    for step in range(steps):
        prompt = prompts[step % len(prompts)]
        ppo_value_head_update(policy, value_head, reference, reward_bundle, tokenizer, prompt,
                              opt, group_size, max_new_tokens, kl_coef,
                              clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef,
                              gamma=gamma, lam=lam)
    if save_dir:
        save_aligned_policy(save_dir, policy, value_head, meta={"method": "ppo", "kl_coef": kl_coef})
    return policy


def _seq_logp(model, ids):
    """Sum of per-token log-probs of a token-id sequence under ``model``."""
    tgt = ids[0, 1:]
    rng = torch.arange(tgt.shape[0], device=ids.device)
    return model(ids).logits[0, :-1].log_softmax(-1)[rng, tgt].sum()


def _response_logp(model, prompt, response, tokenizer, device, max_prompt=384, max_resp=128):
    """Log-prob of ``response`` tokens only (prompt masked), standard DPO scoring.

    Tokenizes prompt and response separately so the boundary is exact and each is
    truncated independently — a long dialogue prompt can't crowd out the response.
    """
    p_ids = tokenizer(prompt).input_ids[-max_prompt:]        # keep end of (long) prompt
    r_ids = tokenizer(" " + response).input_ids[:max_resp]   # keep start of response
    ids = torch.tensor([p_ids + r_ids], device=device)
    tgt = ids[0, 1:]
    rng = torch.arange(tgt.shape[0], device=device)
    tok_lp = model(ids).logits[0, :-1].log_softmax(-1)[rng, tgt]
    return tok_lp[max(len(p_ids) - 1, 0):].sum()     # response region only


def align_with_dpo(policy, reference, reward_bundle, tokenizer, prompts,
                   steps, group_size, max_new_tokens, beta, lr=1e-5, pref_pairs=None,
                   save_dir=None):
    # True DPO on the dataset's real (chosen, rejected) pairs when available; falls
    # back to on-policy best/worst-by-RM pairs for the synthetic/CI path.
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    device = next(policy.parameters()).device
    for step in range(steps):
        if pref_pairs:
            ex = pref_pairs[step % len(pref_pairs)]
            lp_c = _response_logp(policy, ex["prompt"], ex["chosen"], tokenizer, device)
            lp_r = _response_logp(policy, ex["prompt"], ex["rejected"], tokenizer, device)
            with torch.no_grad():
                ref_c = _response_logp(reference, ex["prompt"], ex["chosen"], tokenizer, device)
                ref_r = _response_logp(reference, ex["prompt"], ex["rejected"], tokenizer, device)
        else:
            prompt = prompts[step % len(prompts)]
            seqs = sample_group(policy, tokenizer, prompt, group_size, max_new_tokens)
            rewards = score_sequences(reward_bundle, tokenizer, seqs)
            chosen, rejected = seqs[rewards.argmax()], seqs[rewards.argmin()]
            lp_c, lp_r = _seq_logp(policy, chosen.unsqueeze(0)), _seq_logp(policy, rejected.unsqueeze(0))
            with torch.no_grad():
                ref_c = _seq_logp(reference, chosen.unsqueeze(0))
                ref_r = _seq_logp(reference, rejected.unsqueeze(0))

        loss = dpo_loss(lp_c.unsqueeze(0), lp_r.unsqueeze(0),
                        ref_c.unsqueeze(0), ref_r.unsqueeze(0), beta=beta)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    if save_dir:
        save_aligned_policy(save_dir, policy, meta={"method": "dpo", "beta": beta})
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
    # Decoupled sweep knobs: kl_coef is an additive KL *penalty* (PPO/GRPO); dpo beta
    # is the DPO temperature. They are NOT the same quantity, so sweep them separately.
    ap.add_argument("--kl-coefs", type=float, nargs="+", default=None)
    ap.add_argument("--dpo-betas", type=float, nargs="+", default=None)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.02, 0.1, 0.5],
                    help="fallback sweep used for both when --kl-coefs/--dpo-betas are unset")
    ap.add_argument("--rm-dir", default="results/reward_model")
    ap.add_argument("--dataset", default="Dahoas/rm-static")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-train-prompts", type=int, default=128,
                    help="prompts for training rollouts (disjoint from eval)")
    ap.add_argument("--n-eval-prompts", type=int, default=32,
                    help="held-out prompts for reward/KL eval (disjoint from train)")
    ap.add_argument("--n-prompts", type=int, default=None,
                    help="back-compat: if set, overrides --n-train-prompts")
    ap.add_argument("--rl-lr", type=float, default=1e-5,
                    help="policy LR for the RL aligners; lower avoids reward-hacking collapse")
    ap.add_argument("--value-warmup", type=int, default=30,
                    help="PPO value-head warmup steps (critic pretrain before policy updates)")
    ap.add_argument("--vf-lr", type=float, default=1e-3, help="LR for the PPO value-head warmup")
    ap.add_argument("--save-dir", default=None,
                    help="if set, save each aligned policy (+ PPO value head) under results/<save-dir>/")
    ap.add_argument("--out", default="results/frontier.json")
    args = ap.parse_args()

    n_train = args.n_prompts or args.n_train_prompts

    kl_coefs = args.kl_coefs or args.betas
    dpo_betas = args.dpo_betas or args.betas
    sweeps = {"grpo": kl_coefs, "ppo": kl_coefs, "dpo": dpo_betas}

    tokenizer, base_policy, reference, reward_bundle, train_prompts, eval_prompts = build_harness(
        rm_dir=args.rm_dir, dataset=args.dataset, model=args.model,
        n_train_prompts=n_train, n_eval_prompts=args.n_eval_prompts)
    # Guarantee a full sweep of the training set before eval: round-robin covers every
    # train prompt once steps >= len(train_prompts); bump steps up if the caller asked for fewer.
    steps = args.steps
    if steps < len(train_prompts):
        print(f"[exp1] bumping --steps {steps} -> {len(train_prompts)} to cover the full "
              f"training set before eval", flush=True)
        steps = len(train_prompts)

    # True DPO trains on the dataset's real offline pairs (None -> on-policy fallback).
    pref_pairs = None
    if args.rm_dir and os.path.isdir(args.rm_dir):
        pref_pairs = load_preference_dataset(args.dataset, "train",
                                             max_examples=max(steps, 500))

    # Base and every method are scored on the held-out eval prompts, never the train set.
    r0, kl0 = eval_reward_and_kl(base_policy, reference, reward_bundle, tokenizer, eval_prompts,
                                 args.group_size, args.max_new_tokens)
    frontier = {"base": {"reward": r0, "kl": kl0}, "methods": {}}

    for name, aligner in METHODS.items():
        frontier["methods"][name] = []
        for coef in sweeps[name]:
            policy = clone_policy(base_policy)
            kw = dict(lr=args.rl_lr, pref_pairs=pref_pairs)
            if name == "ppo":
                kw.update(value_warmup=args.value_warmup, vf_lr=args.vf_lr)
            if args.save_dir:
                kw["save_dir"] = os.path.join("results", args.save_dir, f"{name}_{coef}")
            aligner(policy, reference, reward_bundle, tokenizer, train_prompts,
                    steps, args.group_size, args.max_new_tokens, coef, **kw)
            reward, kl = eval_reward_and_kl(policy, reference, reward_bundle, tokenizer, eval_prompts,
                                            args.group_size, args.max_new_tokens)
            frontier["methods"][name].append({"beta": coef, "reward": reward, "kl": kl})
            knob = "beta" if name == "dpo" else "kl_coef"
            print(f"[{name}] {knob}={coef:<5} reward={reward:+.3f} kl={kl:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(frontier, f, indent=2)
    print(f"Wrote {args.out}. Plot with experiments/plot_frontier.py")


if __name__ == "__main__":
    main()
