"""Render the recruiter-facing "RLHF from scratch — results" one-pager.

    python -m experiments.plot_showcase

Reads the real artifacts already on disk and draws a 2x2 figure to
`results/showcase.png`:

  A. Reward model learned real preferences   (held-out pairwise accuracy vs random)
  B. Alignment frontier                       (reward vs KL drift; GRPO dominates)
  C. Sample efficiency                        (reward vs step; GRPO rises, PPO flat)
  D. Model-parallel scaling                   (peak GPU memory vs #GPUs, FSDP ZeRO-3)

Panel D needs `results/bench_parallel.json` (produced by scripts/bench_parallel.slurm);
if it is missing the panel shows a "run bench_parallel" note and A-C still render.
"""
import json
import os

RESULTS = "results"

# Validated categorical palette (dataviz skill). Entity -> color, fixed, never cycled.
C_GRPO = "#2a78d6"   # slot 1 blue  — the winner
C_PPO = "#e34948"    # slot 6 red
C_DPO = "#1baf7a"    # slot 2 aqua
C_BASE = "#0b0b0b"   # ink
C_MEM = "#256abf"    # sequential blue for the memory bars
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

METHOD_COLOR = {"grpo": C_GRPO, "ppo": C_PPO, "dpo": C_DPO}


def _load(path):
    with open(path) as f:
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


def _ema(xs, alpha=0.2):
    out, m = [], xs[0]
    for x in xs:
        m = alpha * x + (1 - alpha) * m
        out.append(m)
    return out


def panel_reward_model(ax):
    meta = _load(os.path.join(RESULTS, "reward_model", "train_log.json"))
    acc = meta["val_pairwise_acc"]
    bars = ax.bar(["random\nguessing", "learned\nRM"], [0.5, acc],
                  color=["#c3c2b7", C_GRPO], width=0.6, zorder=3)
    for b, v in zip(bars, [0.5, acc]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--", zorder=2)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("held-out pairwise accuracy")
    ax.set_title(f"A · Reward model learned real preferences\n"
                 f"{meta['model']}, {meta['train_pairs']:,} preference pairs",
                 fontsize=11, fontweight="bold", color=INK, loc="left")
    _style(ax)


def panel_frontier(ax):
    data = _load(os.path.join(RESULTS, "frontier.json"))
    b = data["base"]
    ax.scatter([b["kl"]], [b["reward"]], marker="*", s=240, color=C_BASE,
               zorder=5, label="base (SFT)")
    for name in ("grpo", "ppo", "dpo"):  # fixed order
        pts = data["methods"][name]
        ax.scatter([p["kl"] for p in pts], [p["reward"] for p in pts],
                   s=70, color=METHOD_COLOR[name], zorder=4,
                   edgecolors="white", linewidths=0.8, label=name.upper())
    ax.set_xlabel("KL(policy || reference)   — drift from the base model")
    ax.set_ylabel("mean reward (RM score)")
    ax.set_title("B · Alignment frontier: more reward, minimal drift\n"
                 "GRPO sits highest — best reward per unit of KL",
                 fontsize=11, fontweight="bold", color=INK, loc="left")
    ax.annotate("better", xy=(0.06, 0.94), xycoords="axes fraction",
                fontsize=9, color=MUTED, ha="left", va="top")
    ax.annotate("", xy=(0.02, 0.98), xytext=(0.14, 0.86), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
    ax.legend(frameon=False, fontsize=9, loc="center left")
    _style(ax)


def panel_sample_eff(ax):
    data = _load(os.path.join(RESULTS, "sample_eff.json"))
    for name in ("grpo", "ppo"):
        d = data[name]
        steps = range(1, len(d["reward_trace"]) + 1)
        ax.plot(steps, d["reward_trace"], color=METHOD_COLOR[name], alpha=0.25, linewidth=1)
        ax.plot(steps, _ema(d["reward_trace"]), color=METHOD_COLOR[name], linewidth=2.2,
                label=f"{name.upper()}  ·  peak {d['peak_mem_mb'] / 1024:.1f} GB")
    ax.set_xlabel("training step")
    ax.set_ylabel("mean reward (RM score)")
    ax.set_title("C · Sample efficiency: GRPO climbs, PPO stalls\n"
                 "and GRPO does it with less memory",
                 fontsize=11, fontweight="bold", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)


def panel_scaling(ax):
    path = os.path.join(RESULTS, "bench_parallel.json")
    if not os.path.exists(path):
        ax.text(0.5, 0.5, "run scripts/bench_parallel.slurm\nto fill this panel",
                ha="center", va="center", fontsize=11, color=MUTED)
        ax.set_title("D · Model-parallel scaling (FSDP / ZeRO-3)",
                     fontsize=11, fontweight="bold", color=INK, loc="left")
        ax.set_axis_off()
        return
    data = _load(path)
    ws = sorted(int(k) for k in data)
    gb = [data[str(w)]["peak_mem_mb"] / 1024 for w in ws]
    tps = [data[str(w)]["throughput_tok_s"] for w in ws]
    bars = ax.bar([str(w) for w in ws], gb, color=C_MEM, width=0.6, zorder=3)
    for b, g, t in zip(bars, gb, tps):
        ax.text(b.get_x() + b.get_width() / 2, g + max(gb) * 0.02, f"{g:.1f} GB",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
        ax.text(b.get_x() + b.get_width() / 2, g / 2, f"{t:,.0f}\ntok/s",
                ha="center", va="center", fontsize=8, color="white")
    model = data[str(ws[0])].get("model", "model")
    ax.set_ylim(0, max(gb) * 1.18)
    ax.set_xlabel("number of GPUs (world size)")
    ax.set_ylabel("peak memory per GPU (GB)")
    ax.set_title(f"D · FSDP shards the model: memory/GPU falls\n"
                 f"ZeRO-3 on {model} — optimizer state sharded across GPUs",
                 fontsize=11, fontweight="bold", color=INK, loc="left")
    _style(ax)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.flat:
        ax.set_facecolor("#fcfcfb")
    panel_reward_model(axes[0, 0])
    panel_frontier(axes[0, 1])
    panel_sample_eff(axes[1, 0])
    panel_scaling(axes[1, 1])

    fig.suptitle("RLHF from scratch — reward modeling, alignment, and model-parallel scaling",
                 fontsize=15, fontweight="bold", color=INK, x=0.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(RESULTS, "showcase.png")
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print("wrote", out)


if __name__ == "__main__":
    main()
