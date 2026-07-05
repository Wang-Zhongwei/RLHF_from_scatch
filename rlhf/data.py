"""Synthetic datasets + prompt templating for SFT, reward modeling and eval.

These tiny hand-written sets keep CI hermetic (no downloads). On GPU, swap
``build_synthetic_*`` for a real corpus loader with the same record schema.
"""
import random


def build_synthetic_instruction_dataset():
    return [
        {"prompt": "What is the capital of France?",
         "response": "The capital of France is Paris."},
        {"prompt": "Write a Python function to add two numbers.",
         "response": "def add(a, b):\n    return a + b"},
        {"prompt": "Who wrote Romeo and Juliet?",
         "response": "William Shakespeare wrote Romeo and Juliet."},
        {"prompt": "Explain what a neural network is.",
         "response": "A neural network is a machine learning model inspired by the human brain."},
    ]


def format_example(example):
    """Render one instruction example into a single training string."""
    return f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}"


def apply_template(examples):
    return [format_example(example) for example in examples]


def build_synthetic_preference_dataset(num_examples=8, seed=0):
    pool = [
        {"prompt": "What is the boiling point of water?",
         "chosen": "Water boils at 100 degrees Celsius at sea level.",
         "rejected": "Water boils when it gets hot enough."},
        {"prompt": "Who wrote Romeo and Juliet?",
         "chosen": "Romeo and Juliet was written by William Shakespeare.",
         "rejected": "I am not sure who wrote it."},
        {"prompt": "What is 2 + 2?",
         "chosen": "2 + 2 equals 4.",
         "rejected": "2 + 2 equals 5."},
        {"prompt": "What does HTTP stand for?",
         "chosen": "HTTP stands for HyperText Transfer Protocol.",
         "rejected": "HTTP is a computer term."},
        {"prompt": "What is the speed of light?",
         "chosen": "The speed of light in a vacuum is approximately 3x10^8 meters per second.",
         "rejected": "Light travels very fast."},
        {"prompt": "What is DNA?",
         "chosen": "DNA is a molecule that carries genetic instructions for living organisms.",
         "rejected": "DNA is found in cells."},
        {"prompt": "What is the capital of France?",
         "chosen": "The capital of France is Paris.",
         "rejected": "I do not know."},
        {"prompt": "What is 2 + 2?",
         "chosen": "2 + 2 equals 4.",
         "rejected": "2 + 2 equals 5."},
    ]
    rng = random.Random(seed)
    if num_examples <= len(pool):
        return rng.sample(pool, num_examples)
    indices = list(range(len(pool))) * (num_examples // len(pool) + 1)
    rng.shuffle(indices)
    return [pool[i] for i in indices[:num_examples]]


def format_preference(example):
    return {
        "chosen_text": example["prompt"] + " " + example["chosen"],
        "rejected_text": example["prompt"] + " " + example["rejected"],
    }


def build_eval_prompt_set():
    """Held-out instruction-style prompts for win-rate / frontier evaluation."""
    return [
        "Explain what a neural network is in simple terms.",
        "Write a short poem about the ocean.",
        "What are three tips for staying productive while working from home?",
        "Summarize the plot of Romeo and Juliet in two sentences.",
        "What is the difference between supervised and unsupervised learning?",
        "Give me a recipe for a simple pasta dish.",
    ]


# --- Real dataset loaders (GPU runs; require `datasets`, pre-cached offline) -----------
# Kept out of the hermetic-test path: `datasets` is imported lazily so CI that only
# touches the synthetic builders above never needs the dependency or a download.

def _pick(cols, *candidates):
    """Return the first candidate column name present in ``cols`` (case-insensitive)."""
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _load_split(name, split):
    """Load one split of a parquet-based HF dataset from the *local* hub cache.

    Air-gapped compute nodes can't let ``datasets`` resolve a dataset by repo id
    (it phones the Hub for the builder even in offline mode). Instead we resolve
    the cached snapshot via ``huggingface_hub`` and point the parquet builder
    straight at the shard files — fully offline.
    """
    import glob
    import os

    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    snap = snapshot_download(repo_id=name, repo_type="dataset", local_files_only=True)
    for pat in (os.path.join(snap, "data", f"{split}-*.parquet"),
                os.path.join(snap, f"{split}-*.parquet"),
                os.path.join(snap, "**", f"*{split}*.parquet")):
        files = sorted(glob.glob(pat, recursive=True))
        if files:
            return load_dataset("parquet", data_files=files, split="train")
    raise FileNotFoundError(f"{name}: no parquet shards for split '{split}' under {snap}")


def load_preference_dataset(name="Dahoas/rm-static", split="train",
                            max_examples=None, seed=0):
    """Load a real preference set as ``list[{"prompt","chosen","rejected"}]``.

    Matches the schema of ``build_synthetic_preference_dataset`` so the existing
    ``build_pref_batch`` / ``reward_train_step`` path works unchanged. ``chosen`` /
    ``rejected`` are response-only strings (the prompt is prepended downstream).
    """
    ds = _load_split(name, split)
    cols = ds.column_names
    p_col = _pick(cols, "prompt", "question", "instruction")
    c_col = _pick(cols, "chosen", "response_j", "chosen_response")
    r_col = _pick(cols, "rejected", "response_k", "rejected_response")
    if not (p_col and c_col and r_col):
        raise ValueError(f"{name}: could not map prompt/chosen/rejected from columns {cols}")
    if max_examples is not None and max_examples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_examples))
    out = []
    for ex in ds:
        prompt, chosen, rejected = ex[p_col], ex[c_col], ex[r_col]
        if not (isinstance(prompt, str) and isinstance(chosen, str) and isinstance(rejected, str)):
            continue
        prompt, chosen, rejected = prompt.strip(), chosen.strip(), rejected.strip()
        if not prompt or not chosen or not rejected or chosen == rejected:
            continue
        out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    if not out:
        raise ValueError(f"{name}[{split}]: no usable preference pairs after filtering")
    return out


def load_prompt_set(name="Dahoas/rm-static", split="test", n=64, seed=0):
    """Held-out prompts (``list[str]``) for PPO/GRPO rollouts and frontier eval."""
    ds = _load_split(name, split)
    p_col = _pick(ds.column_names, "prompt", "question", "instruction")
    if p_col is None:
        raise ValueError(f"{name}: no prompt column in {ds.column_names}")
    ds = ds.shuffle(seed=seed)
    prompts, seen = [], set()
    for ex in ds:
        p = ex[p_col]
        if not isinstance(p, str):
            continue
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            prompts.append(p)
        if len(prompts) >= n:
            break
    return prompts
