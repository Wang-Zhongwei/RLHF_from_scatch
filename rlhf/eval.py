"""Evaluation: completion generation, reward scoring, and win-rate."""
import torch

from .decoding import generate_and_decode
from .reward import reward_head_forward


def generate_completions(model, tokenizer, prompts, max_new_tokens=16):
    """One greedy completion per prompt, preserving input order."""
    return [
        generate_and_decode(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        for prompt in prompts
    ]


def score_with_reward(reward_model, tokenizer, prompt, completion):
    """Scalar reward for a prompt+completion pair.

    ``reward_model`` is a bundle dict {"model", "weight", "bias"}.
    """
    inputs = tokenizer(prompt + completion, return_tensors="pt")
    with torch.no_grad():
        outputs = reward_model["model"](**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1][0, -1, :]
    score = reward_head_forward(last_hidden, reward_model["weight"], reward_model["bias"])
    return score.item()


def win_rate(reward_model, tokenizer, prompts, completions_a, completions_b):
    """Fraction of prompts where A outscores B under the reward model (ties = 0.5)."""
    a_win = 0.0
    for prompt, comp_a, comp_b in zip(prompts, completions_a, completions_b):
        score_a = score_with_reward(reward_model, tokenizer, prompt, comp_a)
        score_b = score_with_reward(reward_model, tokenizer, prompt, comp_b)
        a_win += 1 if score_a > score_b else 0 if score_a < score_b else 0.5
    return a_win / len(prompts)
