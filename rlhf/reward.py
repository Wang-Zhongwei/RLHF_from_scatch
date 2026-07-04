"""Reward modeling: scalar head + Bradley-Terry pairwise training."""
import numpy as np
import torch
import torch.nn.functional as F


def reward_head_forward(hidden_state, weight, bias):
    """Map a final hidden state to a scalar reward via a linear projection."""
    w = weight.view(-1)  # accept both (D,) and (1, D)
    return hidden_state @ w + bias


def pairwise_reward_loss(chosen_reward, rejected_reward):
    """Bradley-Terry pairwise loss: mean(-log_sigmoid(chosen - rejected))."""
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()


def reward_bce_loss(chosen_reward, rejected_reward):
    """Symmetric BCE form of the pairwise reward loss (numpy reference)."""
    chosen_loss = np.logaddexp(0, -chosen_reward)
    rejected_loss = np.logaddexp(0, rejected_reward)
    return float(np.mean(0.5 * (chosen_loss + rejected_loss)))


def pairwise_accuracy(chosen_reward, rejected_reward):
    """Fraction of pairs where chosen strictly beats rejected."""
    return (chosen_reward > rejected_reward).mean(dtype=float)


def reward_train_step(model, reward_head, batch, optimizer, pad_id=0):
    """Forward chosen+rejected, score the last real token, step the optimizer.

    ``model`` is expected to be a callable returning hidden states of shape
    (B, T, D) — see ``experiments/pipeline_demo.py`` for the backbone adapter.
    """
    optimizer.zero_grad()

    chosen_hidden = model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
    )
    B = chosen_hidden.shape[0]
    last_chosen_idx = (batch["chosen_input_ids"] != pad_id).sum(dim=-1) - 1
    last_chosen_hidden = chosen_hidden[torch.arange(B), last_chosen_idx, :]

    rejected_hidden = model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
    )
    B = rejected_hidden.shape[0]
    last_rejected_idx = (batch["rejected_input_ids"] != pad_id).sum(dim=-1) - 1
    last_rejected_hidden = rejected_hidden[torch.arange(B), last_rejected_idx, :]

    chosen_reward = reward_head_forward(last_chosen_hidden, reward_head.weight, reward_head.bias)
    rejected_reward = reward_head_forward(last_rejected_hidden, reward_head.weight, reward_head.bias)

    loss = pairwise_reward_loss(chosen_reward, rejected_reward)
    acc = pairwise_accuracy(chosen_reward, rejected_reward).item()

    loss.backward()
    optimizer.step()
    return {"loss": loss.item(), "accuracy": acc}
