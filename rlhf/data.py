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
