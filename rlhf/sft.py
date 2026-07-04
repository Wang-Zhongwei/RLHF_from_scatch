"""Supervised fine-tuning step + evaluation loop."""
import torch

from .losses import cross_entropy_loss, shift_logits_and_labels


def sft_train_step(model, batch, optimizer):
    """Run one SFT forward/backward/step and return the loss as a float."""
    optimizer.zero_grad()
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    ).logits
    shifted_logits, shifted_labels = shift_logits_and_labels(logits, batch["labels"])
    loss = cross_entropy_loss(shifted_logits, shifted_labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_loss(model, batches):
    """Mean LM loss over validation batches, no grad."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in batches:
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
            shifted_logits, shifted_labels = shift_logits_and_labels(logits, batch["labels"])
            total_loss += cross_entropy_loss(shifted_logits, shifted_labels).item()
            count += 1
    return total_loss / count
