"""Tokenization, label masking, padding/collation and batch iteration."""
import math
import random

import torch


def tokenize_example(tokenizer, text, max_length=64):
    """Encode ``text`` with truncation at ``max_length``, no padding."""
    return tokenizer.encode(text, padding=False, max_length=max_length)


def build_labels(input_ids):
    """Return next-token labels (a fresh copy of ``input_ids``)."""
    return list(input_ids)


def mask_prompt_labels(labels, prompt_length):
    """Mask the first ``prompt_length`` labels with -100 so loss ignores the prompt."""
    return [-100 if i < prompt_length else label for i, label in enumerate(labels)]


def pad_batch(sequences, pad_id):
    """Right-pad a list of token-id sequences to the longest length."""
    n_max = max(len(seq) for seq in sequences)
    return [seq + [pad_id] * (n_max - len(seq)) for seq in sequences]


def make_attention_mask(padded_ids, pad_id):
    """Same-shape 0/1 mask: 1 where token != pad_id."""
    return [[1 if token != pad_id else 0 for token in row] for row in padded_ids]


def collate_lm_batch(batch, pad_id):
    input_ids = torch.tensor(pad_batch([x["input_ids"] for x in batch], pad_id))
    labels = torch.tensor(pad_batch([x["labels"] for x in batch], -100))
    attention_mask = torch.where(input_ids == pad_id, 0, 1)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def iterate_minibatches(examples, batch_size, seed=0):
    """Yield shuffled minibatches of size ``batch_size`` (deterministic per seed)."""
    rng = random.Random(seed)
    n = len(examples)
    rng.shuffle(examples)
    start = 0
    while start < n:
        end = min(n, start + batch_size)
        yield examples[start:end]
        start = end


def train_val_split(examples, val_ratio=0.2, seed=0):
    rng = random.Random(seed)
    shuffled = list(examples)
    n = len(shuffled)
    rng.shuffle(shuffled)
    val_size = math.floor(val_ratio * n)
    train_size = n - val_size
    return shuffled[:train_size], shuffled[train_size:]
