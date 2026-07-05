"""Render the GSM8K-GRPO project figure from results/gsm8k_grpo.json.

    python -m experiments.plot_gsm8k

Two panels:
  A. GSM8K held-out accuracy vs GRPO step (single-GPU) — the learning curve.
  B. Peak memory-per-GPU across single / DDP / FSDP, with throughput labels — the
     DDP-vs-FSDP systems tradeoff.
"""
import json
import os

RESULTS = "results"
SRC = os.path.join(RESULTS, "gsm8k_grpo.json")

# Validated categorical palette (dataviz skill).
C_BASE = "#2a78d6"   # blue  — baseline / single
C_DDP = "#eda100"    # yellow
C_FSDP = "#4a3aa7"   # violet
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _load():
    with open(SRC) as f:
        return json.load(f)


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _acc_xy(rec):
    tr = rec.get("acc_trace") or []
    return [p["step"] for p in tr], [100 * p["acc"] for p in tr]


def panel_accuracy(ax, data):
    # Show the single-GPU curve — the cleanest monotonic "GRPO improves" story
    # (DDP/FSDP evaluate on different held-out shards, so they aren't directly comparable).
    base = data.get("single") or data.get("ddp") or data.get("fsdp")
    if base:
        x, y = _acc_xy(base)
        ax.plot(x, y, marker="o", color=C_BASE, linewidth=2.2, label="GRPO")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("GSM8K held-out accuracy (%)")
    ax.set_title("A · GRPO improves GSM8K accuracy",
                 fontsize=11, fontweight="bold", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)


def panel_systems(ax, data):
    order = [("single", "single-GPU", C_BASE),
             ("ddp", "DDP (4 GPU)", C_DDP),
             ("fsdp", "FSDP (4 GPU)", C_FSDP)]
    labels, gb, colors, tps = [], [], [], []
    for key, label, color in order:
        if key in data:
            labels.append(label)
            gb.append(data[key]["peak_mem_mb"] / 1024)
            tps.append(data[key]["throughput_completions_s"])
            colors.append(color)
    bars = ax.bar(labels, gb, color=colors, width=0.6, zorder=3)
    for b, g, t in zip(bars, gb, tps):
        ax.text(b.get_x() + b.get_width() / 2, g + max(gb) * 0.02, f"{g:.1f} GB",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
        ax.text(b.get_x() + b.get_width() / 2, g / 2, f"{t:.1f}\ncmp/s",
                ha="center", va="center", fontsize=8, color="white")
    ax.set_ylim(0, max(gb) * 1.18 if gb else 1)
    ax.set_ylabel("peak memory per GPU (GB)")
    ax.set_title("B · DDP replicates, FSDP shards (ZeRO-3)\n"
                 "FSDP cuts per-GPU memory; DDP keeps throughput",
                 fontsize=11, fontweight="bold", color=INK, loc="left")
    _style(ax)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    data = _load()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
    panel_accuracy(axes[0], data)
    panel_systems(axes[1], data)

    fig.suptitle("From-scratch GRPO on GSM8K (Qwen2.5-0.5B): verifiable rewards, "
                 "benchmarked single-GPU vs DDP vs FSDP",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(RESULTS, "gsm8k_grpo.png")
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print("wrote", out)


if __name__ == "__main__":
    main()
