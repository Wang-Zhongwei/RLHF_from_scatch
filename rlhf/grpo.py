"""GRPO — Group Relative Policy Optimization (DeepSeekMath / DeepSeek-R1).

GRPO drops PPO's learned value function. Instead, for each prompt it samples a
*group* of G completions, scores them all with the reward model, and normalizes
rewards *within the group* to get advantages:

    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

The group mean acts as the baseline that a critic would otherwise provide, so
GRPO needs ~half the memory of PPO (no value head, no value optimizer state) —
which is exactly the sample/compute-efficiency story in
``experiments/exp2_grpo_sample_efficiency.py``.

The policy update reuses PPO's clipped surrogate (see :mod:`rlhf.ppo`) and adds
a KL penalty to a frozen reference policy, estimated with the low-variance,
always-positive k3 estimator from http://joschu.net/blog/kl-approx.html.
"""
import torch

from .ppo import clipped_surrogate, policy_ratio


def group_relative_advantages(rewards, eps=1e-4):
    """Normalize rewards within each group to zero mean / unit variance.

    Args:
        rewards: (G,) for a single prompt or (B, G) for B prompts each with a
            group of G sampled completions.
        eps: numerical floor on the group std.
    Returns:
        Advantages with the same shape as ``rewards``.
    """
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / (std + eps)


def k3_kl(policy_logprobs, ref_logprobs):
    """Unbiased, non-negative k3 KL estimate: exp(r) - r - 1, r = ref - policy."""
    r = ref_logprobs - policy_logprobs
    return r.exp() - r - 1


def grpo_loss(new_logprobs, old_logprobs, advantages,
              ref_logprobs=None, clip_eps=0.2, kl_coef=0.0):
    """GRPO per-sample loss: clipped surrogate + optional KL-to-reference penalty.

    Args:
        new_logprobs: sequence log-probs under the current policy, (N,).
        old_logprobs: log-probs under the behavior policy that sampled the group.
        advantages: group-relative advantages from ``group_relative_advantages``.
        ref_logprobs: log-probs under the frozen reference (for the KL term).
        clip_eps: PPO clip range.
        kl_coef: weight on the KL penalty (0.0 disables it).
    Returns:
        Scalar loss dict: {"policy_loss", "kl", "loss"}.
    """
    ratio = policy_ratio(new_logprobs, old_logprobs)
    policy_loss = clipped_surrogate(ratio, advantages, clip_eps)

    kl = torch.zeros((), dtype=policy_loss.dtype)
    if ref_logprobs is not None and kl_coef != 0.0:
        kl = k3_kl(new_logprobs, ref_logprobs).mean()

    return {
        "policy_loss": policy_loss,
        "kl": kl,
        "loss": policy_loss + kl_coef * kl,
    }
