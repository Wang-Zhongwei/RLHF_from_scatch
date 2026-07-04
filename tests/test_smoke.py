"""Fast, hermetic unit tests for the from-scratch math (no model downloads)."""
import numpy as np
import torch

from rlhf import (
    greedy_decode, top_k_filter, cross_entropy_loss, shift_logits_and_labels,
    pad_batch, collate_lm_batch, adamw_update, gae_advantages, compute_returns,
    policy_ratio, clipped_surrogate, dpo_loss, simpo_loss, pairwise_reward_loss,
    group_relative_advantages, grpo_loss,
)


def test_greedy_decode():
    assert greedy_decode(torch.tensor([0.1, 0.9, 0.3])) == 1


def test_top_k_filter_keeps_k():
    out = top_k_filter(torch.tensor([1.0, 2.0, 3.0, 4.0]), k=2)
    assert torch.isinf(out).sum() == 2


def test_cross_entropy_ignores_mask():
    logits = torch.randn(1, 3, 5)
    labels = torch.tensor([[-100, 2, 3]])
    sl, sll = shift_logits_and_labels(logits, labels)
    loss = cross_entropy_loss(sl, sll)
    assert loss.ndim == 0 and loss.item() >= 0


def test_pad_and_collate():
    batch = [{"input_ids": [1, 2], "labels": [1, 2]},
             {"input_ids": [3], "labels": [3]}]
    out = collate_lm_batch(batch, pad_id=0)
    assert out["input_ids"].shape == (2, 2)
    assert out["attention_mask"].tolist() == [[1, 1], [1, 0]]


def test_adamw_reduces_quadratic():
    x = torch.tensor([5.0], requires_grad=True)
    state = {}
    for _ in range(200):
        loss = (x ** 2).sum()
        (grad,) = torch.autograd.grad(loss, x)
        with torch.no_grad():
            adamw_update(x, grad, state, lr=0.1)
    assert abs(x.item()) < 0.5


def test_gae_matches_returns_when_values_zero():
    rewards = np.array([1.0, 1.0, 1.0])
    adv = gae_advantages(rewards, np.zeros(4), gamma=0.99, lam=1.0)
    ret = compute_returns(rewards, gamma=0.99)
    assert np.allclose(adv, ret, atol=1e-6)


def test_clipped_surrogate_sign():
    ratio = torch.ones(4)
    adv = torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert clipped_surrogate(ratio, adv).item() < 0  # loss is -advantage here


def test_group_relative_advantages_zero_mean():
    adv = group_relative_advantages(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert abs(adv.mean().item()) < 1e-5


def test_grpo_loss_runs():
    lp = torch.randn(4, requires_grad=True)
    adv = group_relative_advantages(torch.randn(4))
    out = grpo_loss(lp, lp.detach(), adv, ref_logprobs=torch.randn(4), kl_coef=0.1)
    out["loss"].backward()
    assert lp.grad is not None


def test_dpo_prefers_chosen():
    # higher chosen logps -> lower loss
    hi = dpo_loss(torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1.0]))
    lo = dpo_loss(torch.tensor([0.0]), torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([1.0]))
    assert hi.item() < lo.item()


def test_simpo_runs():
    loss = simpo_loss(torch.tensor([-2.0]), torch.tensor([-4.0]),
                      torch.tensor([5.0]), torch.tensor([5.0]))
    assert loss.item() >= 0
