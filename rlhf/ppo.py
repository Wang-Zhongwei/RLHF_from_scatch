"""PPO objective: log-probs, KL, GAE, clipped surrogate, value + entropy terms."""
import numpy as np
import torch


def sequence_logprob(logits, token_ids):
    """Sum log-probs of the selected tokens along the sequence dimension."""
    B = token_ids.shape[0]
    return logits.log_softmax(dim=-1)[torch.arange(B), token_ids].sum()


def per_token_kl(policy_logprobs, ref_logprobs):
    """Per-token KL estimate between policy and reference log-probs."""
    return policy_logprobs - ref_logprobs


def compute_returns(rewards, gamma=0.99):
    """Discounted return at each timestep as a 1D numpy array."""
    T = len(rewards)
    returns = np.zeros(T)
    for t in reversed(range(T)):
        returns[t] = rewards[t] + (0 if t == T - 1 else returns[t + 1] * gamma)
    return returns


def gae_advantages(rewards, values, gamma=0.99, lam=0.95):
    """GAE advantages of shape (T,) from rewards (T,) and values (T+1,)."""
    T = len(rewards)
    adv = np.zeros(T + 1)
    for t in reversed(range(T)):
        delta_t = rewards[t] + gamma * values[t + 1] - values[t]
        adv[t] = delta_t + gamma * lam * adv[t + 1]
    return adv[:-1]


def policy_ratio(new_logprobs, old_logprobs):
    """PPO importance ratio exp(new - old), elementwise."""
    return (new_logprobs - old_logprobs).exp()


def clipped_surrogate(ratio, advantages, clip_eps=0.2):
    """PPO clipped surrogate loss (scalar tensor to minimize)."""
    return -torch.min(
        ratio * advantages,
        torch.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages,
    ).mean()


def value_function_loss(values, returns):
    """MSE between predicted values and target returns."""
    return (values - returns).pow(2).mean()


def entropy_bonus(logits):
    """Mean categorical entropy of the distribution defined by ``logits``."""
    probs = logits.softmax(dim=-1)
    return -(probs * probs.log()).sum(dim=-1).mean()


def ppo_loss(ratio, advantages, values, returns, logits,
             clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """Full PPO loss: clipped surrogate + value loss - entropy bonus."""
    policy_loss = clipped_surrogate(ratio, advantages, clip_eps)
    value_loss = value_function_loss(values, returns)
    entropy = entropy_bonus(logits)
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "loss": loss,
    }


def kl_penalized_reward(reward, kl, beta=0.1):
    """Reward shaped by a KL penalty against a reference policy."""
    return reward - beta * kl
