"""Core language-model loss primitives shared by every training objective."""
import torch


def shift_logits_and_labels(logits, labels):
    """Drop the last logit position and first label so token t predicts t+1."""
    return logits[:, :-1, :], labels[:, 1:]


def cross_entropy_loss(shift_logits, shift_labels):
    """Mean next-token cross-entropy, ignoring positions equal to -100."""
    log_probs = shift_logits.softmax(dim=-1).log()
    mask = shift_labels == -100
    safe_labels = shift_labels.clone()
    safe_labels[mask] = 0
    target_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(dim=-1))
    return -target_log_probs[~mask].mean()
