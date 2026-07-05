"""GSM8K loading + verifiable-reward helpers for GRPO.

The answer parsing is pure Python (no torch / no ``datasets``) so it is unit-testable
anywhere, including the air-gapped login node. ``load_gsm8k`` imports ``datasets`` /
``huggingface_hub`` lazily and reads the *pre-cached* hub snapshot fully offline (same
pattern as ``rlhf/data.py``), but is scoped to the ``main`` config so it never mixes in
the ``socratic`` split.

The verifiable reward is intentionally simple: parse the model's final number, compare it
to the gold answer. No learned reward model, no tokenizer mismatch -- the reward is a
function, which is exactly why the GRPO signal here is trustworthy.
"""
import re

# Matches integers/decimals with optional thousands separators, leading $ and sign.
_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _normalize_number(text):
    """Return the last numeric token in ``text`` as a float, or None."""
    if text is None:
        return None
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    tok = matches[-1].replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(tok)
    except ValueError:
        return None


def parse_gold(answer):
    """GSM8K gold: the number after the final ``####`` marker."""
    tail = answer.split("####")[-1] if "####" in answer else answer
    return _normalize_number(tail)


def extract_pred(text):
    """Model prediction: prefer ``\\boxed{...}``, then 'answer is/: X', else last number."""
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        v = _normalize_number(m.group(1))
        if v is not None:
            return v
    m = re.search(r"(?:final answer|answer|####)\s*(?:is|:|=)?\s*\$?(-?[\d,]*\.?\d+)",
                  text, re.IGNORECASE)
    if m:
        v = _normalize_number(m.group(1))
        if v is not None:
            return v
    return _normalize_number(text)


def is_correct(pred, gold, tol=1e-4):
    """Numeric match with a small relative tolerance (handles 72 vs 72.0)."""
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= tol * max(1.0, abs(gold))


SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, then give the final "
    "numeric answer on its own line in the form \\boxed{ANSWER}."
)


def format_prompt(question, tokenizer=None):
    """Render a GSM8K question into a prompt string.

    Uses the tokenizer's chat template when available (Qwen-Instruct), else a plain
    fallback so the pure-Python path stays dependency-free.
    """
    q = question.strip()
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\nProblem: {q}\nSolution:"


def load_gsm8k(split="train", n=None, seed=0):
    """Load GSM8K ``main`` as ``list[{"question","answer","gold"}]``, fully offline."""
    import glob
    import os

    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    snap = snapshot_download(repo_id="openai/gsm8k", repo_type="dataset",
                             local_files_only=True)
    files = sorted(glob.glob(os.path.join(snap, "main", f"{split}-*.parquet")))
    if not files:
        raise FileNotFoundError(f"gsm8k: no 'main/{split}' parquet shards under {snap}")
    ds = load_dataset("parquet", data_files=files, split="train")
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return [{"question": ex["question"].strip(),
             "answer": ex["answer"].strip(),
             "gold": parse_gold(ex["answer"])} for ex in ds]


if __name__ == "__main__":
    # Pure-Python self-test (runs on the login node; no torch/datasets needed).
    g = "Janet has 3 apples. She buys 5 more.\nSo she has 8.\n#### 8"
    assert parse_gold(g) == 8.0, parse_gold(g)
    assert parse_gold("#### 1,234") == 1234.0
    assert parse_gold("#### -5") == -5.0

    assert extract_pred("... therefore \\boxed{8}") == 8.0
    assert extract_pred("The final answer is 42.") == 42.0
    assert extract_pred("blah blah 7 then 8 then \\boxed{72}") == 72.0
    assert extract_pred("I compute 12 * 6 = 72\n\\boxed{72}") == 72.0
    assert extract_pred("no numbers here") is None
    assert extract_pred("first 3 then the answer: 19") == 19.0

    assert is_correct(8.0, 8.0)
    assert is_correct(72.0, 72)
    assert not is_correct(71.0, 72.0)
    assert not is_correct(None, 8.0)

    # Reward end-to-end: correct completion scores 1, wrong scores 0.
    def _reward(completion, gold):
        return 1.0 if is_correct(extract_pred(completion), gold) else 0.0
    assert _reward("steps... \\boxed{8}", parse_gold(g)) == 1.0
    assert _reward("steps... \\boxed{9}", parse_gold(g)) == 0.0

    print("rlhf/gsm8k.py self-test: OK")
