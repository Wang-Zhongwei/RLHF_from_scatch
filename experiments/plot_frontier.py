"""Render the reward-vs-KL frontier (exp1) and the reward-vs-step curve (exp2).

    python -m experiments.plot_frontier results/frontier.json results/sample_eff.json
"""
import json
import sys


def plot_frontier(path):
    import matplotlib.pyplot as plt
    data = json.load(open(path))
    fig, ax = plt.subplots(figsize=(5, 4))
    b = data["base"]
    ax.scatter([b["kl"]], [b["reward"]], marker="x", s=80, color="black", label="base (SFT)")
    for name, points in data["methods"].items():
        points = sorted(points, key=lambda p: p["kl"])
        ax.plot([p["kl"] for p in points], [p["reward"] for p in points], marker="o", label=name.upper())
    ax.set_xlabel("KL(policy || reference)")
    ax.set_ylabel("mean reward")
    ax.set_title("Alignment frontier: reward vs KL drift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path.replace(".json", ".png"), dpi=150)
    print("wrote", path.replace(".json", ".png"))


def plot_sample_eff(path):
    import matplotlib.pyplot as plt
    data = json.load(open(path))
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, d in data.items():
        ax.plot(range(1, len(d["reward_trace"]) + 1), d["reward_trace"],
                label=f"{name.upper()} (peak {d['peak_mem_mb']:.0f}MB)")
    ax.set_xlabel("training step")
    ax.set_ylabel("mean reward")
    ax.set_title("Sample efficiency: GRPO vs PPO")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path.replace(".json", ".png"), dpi=150)
    print("wrote", path.replace(".json", ".png"))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        (plot_frontier if "frontier" in p else plot_sample_eff)(p)
