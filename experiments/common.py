"""Shared harness for the alignment experiments (exp1 / exp2).

Provides: a frozen reference policy, a trained reward model, group rollouts, and
the two scalar metrics every alignment method is scored on —

    mean reward  (reward model score of on-policy samples)
    KL(policy || reference)  (drift from the SFT starting point)

so PPO / GRPO / DPO can be plotted on the same reward-vs-KL axes.

By default (``rm_dir`` pointing at a saved checkpoint) the reward model is a real
backbone+head trained by ``experiments/train_reward_model.py`` and everything runs
on GPU. With no checkpoint it falls back to the tiny synthetic RM so CI stays
hermetic and offline.
"""
import copy
import json
import os

import torch

from rlhf import (
    load_tokenizer, load_model, set_pad_token_to_eos,
    build_synthetic_preference_dataset, reward_train_step,
    build_eval_prompt_set, reward_head_forward,
    load_reward_model, load_prompt_set,
    gae_advantages, policy_ratio, ppo_loss, k3_kl,
    compute_returns, value_function_loss,
)
from experiments.pipeline_demo import HiddenBackbone, build_pref_batch


def pick_device(device=None):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def build_harness(seed=0, rm_steps=30, rm_dir="results/reward_model",
                  dataset="Dahoas/rm-static", model=None,
                  n_train_prompts=128, n_eval_prompts=32, device=None):
    """Return (tokenizer, policy, reference, reward_bundle, train_prompts, eval_prompts).

    Training rollouts and held-out eval draw from **disjoint** prompt sets so a policy
    is never scored on the prompts it optimized: with a real RM they come from the
    dataset's ``train`` / ``test`` splits. If ``rm_dir`` holds a saved reward model,
    load it (real RM, real prompts, GPU); otherwise build the synthetic toy RM on
    tiny-gpt2 for hermetic/offline CI, where both sets are the tiny eval prompt set
    (the leak is irrelevant for the toy smoke, so the two lists are identical).
    """
    torch.manual_seed(seed)
    device = pick_device(device)
    real_rm = bool(rm_dir) and os.path.isdir(rm_dir)

    if real_rm:
        reward_bundle = load_reward_model(rm_dir, device)
        model_name = model or reward_bundle["meta"].get("model")
    else:
        model_name = model  # None -> DEFAULT_MODEL (tiny-gpt2)

    tokenizer = set_pad_token_to_eos(load_tokenizer(model_name) if model_name else load_tokenizer())

    # Keep the policy in eval mode (dropout off): gradients still flow for the RL
    # aligners, and a dropout-free forward makes KL(policy||reference) measure real
    # weight drift instead of dropout noise (base KL should read ~0, not inflated).
    policy = (load_model(model_name) if model_name else load_model()).to(device)
    policy.eval()
    reference = (load_model(model_name) if model_name else load_model()).to(device)
    for p in reference.parameters():
        p.requires_grad = False

    if real_rm:
        # Disjoint splits: train the aligners on ``train`` prompts, score on held-out ``test``.
        train_prompts = load_prompt_set(dataset, "train", n_train_prompts, seed)
        eval_prompts = load_prompt_set(dataset, "test", n_eval_prompts, seed)
    else:
        # Train a small reward head on synthetic preferences (frozen reference backbone).
        hidden = getattr(policy.config, "n_embd", None) or policy.config.hidden_size
        reward_head = torch.nn.Linear(hidden, 1).to(device)
        rm_opt = torch.optim.AdamW(reward_head.parameters(), lr=1e-3)
        backbone = HiddenBackbone(reference)
        rm_batch = {k: v.to(device) for k, v in
                    build_pref_batch(build_synthetic_preference_dataset(6, seed), tokenizer).items()}
        for _ in range(rm_steps):
            reward_train_step(backbone, reward_head, rm_batch, rm_opt, pad_id=tokenizer.pad_token_id)
        reward_bundle = {"model": reference, "weight": reward_head.weight, "bias": reward_head.bias}
        # Toy path: the leak is irrelevant for the smoke, so train == eval (identical lists).
        train_prompts = eval_prompts = build_eval_prompt_set()

    return tokenizer, policy, reference, reward_bundle, train_prompts, eval_prompts


@torch.no_grad()
def sample_group(policy, tokenizer, prompt, group_size, max_new_tokens=16, temperature=1.0):
    """Sample ``group_size`` completions for one prompt. Returns list of token-id tensors."""
    enc = tokenizer(prompt, return_tensors="pt").to(policy.device)
    out = policy.generate(
        **enc, do_sample=True, temperature=temperature,
        num_return_sequences=group_size, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    return [seq for seq in out]


def score_sequences(reward_bundle, tokenizer, sequences):
    """Reward-model score for each full token sequence. Returns a (G,) tensor.

    Scores the last *non-pad* token to match how the RM was trained
    (rlhf.reward_train_step). generate() right-pads early-finishing samples with
    eos, so the absolute last position is usually padding — scoring it would apply
    the RM at a position it never saw in training.
    """
    pad_id = tokenizer.pad_token_id
    scores = []
    for seq in sequences:
        with torch.no_grad():
            outputs = reward_bundle["model"](seq.unsqueeze(0), output_hidden_states=True)
        idx = max(int((seq != pad_id).sum().item()) - 1, 0)
        last_hidden = outputs.hidden_states[-1][0, idx, :]
        scores.append(reward_head_forward(last_hidden, reward_bundle["weight"], reward_bundle["bias"]))
    return torch.stack(scores).squeeze(-1)


@torch.no_grad()
def eval_reward_and_kl(policy, reference, reward_bundle, tokenizer, prompts,
                       group_size=4, max_new_tokens=16):
    """Mean on-policy reward and mean KL(policy||reference) over eval prompts."""
    rewards, kls = [], []
    for prompt in prompts:
        seqs = sample_group(policy, tokenizer, prompt, group_size, max_new_tokens)
        rewards.append(score_sequences(reward_bundle, tokenizer, seqs).mean().item())
        for seq in seqs:
            ids = seq.unsqueeze(0)
            plogits = policy(ids).logits[0, :-1]
            rlogits = reference(ids).logits[0, :-1]
            tgt = ids[0, 1:]
            rng = torch.arange(tgt.shape[0], device=ids.device)
            pk = plogits.log_softmax(-1)[rng, tgt]
            rk = rlogits.log_softmax(-1)[rng, tgt]
            # k1 estimator: mean(logpi - logref), the standard TRL/PPO approximate KL.
            # Linear (no exp blow-up from a single collapsed token) and positive once the
            # policy genuinely diverges; a near-reference policy may read slightly negative.
            kls.append((pk - rk).mean().item())
    return sum(rewards) / len(rewards), sum(kls) / len(kls)


def clone_policy(policy):
    return copy.deepcopy(policy)


def value_head_warmup(policy, value_head, reward_bundle, tokenizer, prompts, warmup_steps,
                      group_size, max_new_tokens, vf_lr=1e-3, gamma=0.99):
    """Pretrain the value head (policy frozen) before PPO's policy updates begin.

    Regresses V(s) toward Monte-Carlo returns (discounted terminal RM reward) -- a
    fixed target that doesn't bootstrap off the random critic -- so PPO starts with a
    calibrated baseline instead of noise. Trains only the value head, at its own
    (higher) LR. Returns ``{"vf_loss", "ev", "target"}`` measured on the last warmup
    batch -- ``ev`` is the mean value the critic predicts (its expected value), and
    ``target`` is the mean Monte-Carlo return it was regressed toward -- or None if
    disabled.
    """
    if warmup_steps <= 0:
        return None
    opt_v = torch.optim.AdamW(value_head.parameters(), lr=vf_lr)
    stats = None
    for step in range(warmup_steps):
        prompt = prompts[step % len(prompts)]
        with torch.no_grad():
            seqs = sample_group(policy, tokenizer, prompt, group_size, max_new_tokens)
            rewards = score_sequences(reward_bundle, tokenizer, seqs)
        values_l, returns_l = [], []
        for seq, seq_reward in zip(seqs, rewards):
            ids = seq.unsqueeze(0)
            tgt = ids[0, 1:]
            with torch.no_grad():
                hidden = policy(ids, output_hidden_states=True).hidden_states[-1][0, :-1]
                token_rewards = torch.zeros(tgt.shape[0], device=hidden.device)
                token_rewards[-1] = seq_reward
                mc = torch.as_tensor(compute_returns(token_rewards.cpu().numpy(), gamma),
                                     dtype=torch.float32, device=hidden.device)
            values_l.append(value_head(hidden).squeeze(-1))
            returns_l.append(mc)
        v_cat, r_cat = torch.cat(values_l), torch.cat(returns_l)
        loss = value_function_loss(v_cat, r_cat)
        opt_v.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)
        opt_v.step()
        stats = {"vf_loss": loss.item(), "ev": v_cat.mean().item(), "target": r_cat.mean().item()}
    return stats


def save_aligned_policy(out_dir, policy, value_head=None, meta=None):
    """Persist an aligned policy (+ optional PPO value head) under ``out_dir``.

    Off by default in the experiments (metrics-only); pass ``--save-dir`` to enable.
    Mirrors ``rlhf.save_reward_model``: config + a plain state_dict (reloadable with
    ``AutoModelForCausalLM.from_pretrained``), avoiding save_pretrained's deepspeed path.
    """
    os.makedirs(out_dir, exist_ok=True)
    policy.config.save_pretrained(out_dir)
    torch.save({k: v.detach().cpu() for k, v in policy.state_dict().items()},
               os.path.join(out_dir, "pytorch_model.bin"))
    if value_head is not None:
        torch.save({"weight": value_head.weight.detach().cpu(),
                    "bias": value_head.bias.detach().cpu()},
                   os.path.join(out_dir, "value_head.pt"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta or {}, f, indent=2)
    print("saved aligned policy ->", out_dir, flush=True)
    return out_dir


def ppo_value_head_update(policy, value_head, reference, reward_bundle, tokenizer, prompt,
                          opt, group_size, max_new_tokens, kl_coef,
                          clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, gamma=0.99, lam=0.95):
    """One value-head PPO update over a group sampled for ``prompt``; returns mean reward.

    Shared by exp1 (frontier) and exp2 (sample efficiency) so "PPO" is the *same*
    algorithm in both. The value head -- not the group mean -- is PPO's baseline:
    raw RM reward -> per-token GAE with V(s) -> whiten advantages -> clipped surrogate
    + value loss + entropy (rlhf.ppo_loss), plus a k3 KL penalty scaled by kl_coef.
    """
    device = next(policy.parameters()).device
    seqs = sample_group(policy, tokenizer, prompt, group_size, max_new_tokens)
    rewards = score_sequences(reward_bundle, tokenizer, seqs)
    new_lps, ref_lps, values_l, advs, returns_l, logits_l = [], [], [], [], [], []
    for seq, seq_reward in zip(seqs, rewards):
        ids = seq.unsqueeze(0)
        tgt = ids[0, 1:]
        idx = torch.arange(tgt.shape[0], device=device)
        out = policy(ids, output_hidden_states=True)
        logits = out.logits[0, :-1]
        new_lp = logits.log_softmax(-1)[idx, tgt]
        values = value_head(out.hidden_states[-1][0, :-1]).squeeze(-1)
        with torch.no_grad():
            ref_lp = reference(ids).logits[0, :-1].log_softmax(-1)[idx, tgt]
            token_rewards = torch.zeros(tgt.shape[0], device=device)
            token_rewards[-1] = seq_reward
            boot = torch.cat([values.detach(), values.new_zeros(1)])
            adv = torch.as_tensor(
                gae_advantages(token_rewards.cpu().numpy(), boot.cpu().numpy(), gamma, lam),
                dtype=torch.float32, device=device)
            returns = adv + values.detach()
        new_lps.append(new_lp); ref_lps.append(ref_lp); values_l.append(values)
        advs.append(adv); returns_l.append(returns); logits_l.append(logits)

    new_lp = torch.cat(new_lps); ref_lp = torch.cat(ref_lps)
    values = torch.cat(values_l); returns = torch.cat(returns_l); logits = torch.cat(logits_l)
    adv = torch.cat(advs)
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)          # PPO advantage whitening
    ratio = policy_ratio(new_lp, new_lp.detach())
    out_ppo = ppo_loss(ratio, adv, values, returns, logits,
                       clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef)
    kl = k3_kl(new_lp, ref_lp).mean()
    loss = out_ppo["loss"] + kl_coef * kl
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    opt.step()
    return rewards.mean().item()
