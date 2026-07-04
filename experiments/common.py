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
import os

import torch

from rlhf import (
    load_tokenizer, load_model, set_pad_token_to_eos,
    build_synthetic_preference_dataset, reward_train_step,
    build_eval_prompt_set, reward_head_forward,
    load_reward_model, load_prompt_set,
)
from experiments.pipeline_demo import HiddenBackbone, build_pref_batch


def pick_device(device=None):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def build_harness(seed=0, rm_steps=30, rm_dir="results/reward_model",
                  dataset="Dahoas/rm-static", model=None, n_prompts=16, device=None):
    """Return (tokenizer, policy, reference, reward_bundle, prompts).

    If ``rm_dir`` holds a saved reward model, load it (real RM, real prompts, GPU);
    otherwise build the synthetic toy RM on tiny-gpt2 for hermetic/offline CI.
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
        prompts = load_prompt_set(dataset, "test", n_prompts, seed)
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
        prompts = build_eval_prompt_set()

    return tokenizer, policy, reference, reward_bundle, prompts


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
