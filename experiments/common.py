"""Shared harness for the alignment experiments (exp1 / exp2).

Provides: a frozen reference policy, a trained reward model, group rollouts, and
the two scalar metrics every alignment method is scored on —

    mean reward  (reward model score of on-policy samples)
    KL(policy || reference)  (drift from the SFT starting point)

so PPO / GRPO / DPO can be plotted on the same reward-vs-KL axes.
"""
import copy

import torch

from rlhf import (
    load_tokenizer, load_model, set_pad_token_to_eos,
    build_synthetic_preference_dataset, reward_train_step,
    build_eval_prompt_set, sequence_logprob,
)
from experiments.pipeline_demo import HiddenBackbone, build_pref_batch


def build_harness(seed=0, rm_steps=30):
    """Return (tokenizer, policy, reference, reward_bundle, prompts)."""
    torch.manual_seed(seed)
    tokenizer = load_tokenizer()
    set_pad_token_to_eos(tokenizer)

    policy = load_model()
    policy.train()
    reference = load_model()          # frozen SFT snapshot
    for p in reference.parameters():
        p.requires_grad = False

    # Train a small reward head on synthetic preferences.
    hidden = getattr(policy.config, "n_embd", None) or policy.config.hidden_size
    reward_head = torch.nn.Linear(hidden, 1)
    rm_opt = torch.optim.AdamW(reward_head.parameters(), lr=1e-3)
    backbone = HiddenBackbone(reference)
    rm_batch = build_pref_batch(build_synthetic_preference_dataset(6, seed), tokenizer)
    for _ in range(rm_steps):
        reward_train_step(backbone, reward_head, rm_batch, rm_opt)
    reward_bundle = {"model": reference, "weight": reward_head.weight, "bias": reward_head.bias}

    return tokenizer, policy, reference, reward_bundle, build_eval_prompt_set()


@torch.no_grad()
def sample_group(policy, tokenizer, prompt, group_size, max_new_tokens=16, temperature=1.0):
    """Sample ``group_size`` completions for one prompt. Returns list of token-id tensors."""
    enc = tokenizer(prompt, return_tensors="pt")
    out = policy.generate(
        **enc, do_sample=True, temperature=temperature,
        num_return_sequences=group_size, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    return [seq for seq in out]


def score_sequences(reward_bundle, tokenizer, sequences):
    """Reward-model score for each full token sequence. Returns a (G,) tensor."""
    from rlhf import reward_head_forward
    scores = []
    for seq in sequences:
        with torch.no_grad():
            outputs = reward_bundle["model"](seq.unsqueeze(0), output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][0, -1, :]
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
            pk = plogits.log_softmax(-1)[torch.arange(tgt.shape[0]), tgt]
            rk = rlogits.log_softmax(-1)[torch.arange(tgt.shape[0]), tgt]
            kls.append((pk - rk).mean().item())
    return sum(rewards) / len(rewards), sum(kls) / len(kls)


def clone_policy(policy):
    return copy.deepcopy(policy)
