"""Render the alignment figure (exp1 frontier + exp2 sample efficiency).

    python -m experiments.plot_alignment

Three panels from the already-computed JSON:
  A. GRPO & PPO: mean reward vs KL-penalty coefficient.
  B. DPO: mean reward vs beta -- the knob runs the OPPOSITE way (high beta chases reward).
  C. GRPO vs PPO: mean reward vs training step (GRPO climbs; PPO's critic lags).
"""
import json
import os

RESULTS = "results"
C_GRPO = "#2a78d6"   # blue
C_PPO = "#e34948"    # red
C_DPO = "#1baf7a"    # aqua
C_BASE = "#0b0b0b"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


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


def _ema(xs, a=0.15):
    out, m = [], xs[0]
    for x in xs:
        m = a * x + (1 - a) * m
        out.append(m)
    return out


def _title(ax, t):
    ax.set_title(t, fontsize=11, fontweight="bold", color=INK, loc="left")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    frontier = json.load(open(os.path.join(RESULTS, "frontier.json")))
    sample_eff = json.load(open(os.path.join(RESULTS, "sample_eff.json")))
    base_r = frontier["base"]["reward"]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.patch.set_facecolor("#fcfcfb")
    for a in ax:
        a.set_facecolor("#fcfcfb")

    # A -- GRPO & PPO reward vs KL coefficient (categorical x for even spacing).
    for name, color in (("grpo", C_GRPO), ("ppo", C_PPO)):
        pts = sorted(frontier["methods"][name], key=lambda p: p["beta"])
        xs = list(range(len(pts)))
        ax[0].plot(xs, [p["reward"] for p in pts], marker="o", color=color,
                   linewidth=2.2, label=name.upper())
        ax[0].set_xticks(xs)
        ax[0].set_xticklabels([f'{p["beta"]:g}' for p in pts])
    ax[0].axhline(base_r, color=MUTED, linestyle="--", linewidth=1)
    ax[0].text(0, base_r, " base (SFT)", color=MUTED, fontsize=8, va="bottom")
    ax[0].set_xlabel("KL-penalty coefficient")
    ax[0].set_ylabel("mean reward (RM score)")
    _title(ax[0], "A · Lower KL penalty → chase reward")
    ax[0].legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax[0])

    # B -- DPO reward vs beta: opposite direction (higher beta chases reward).
    pts = sorted(frontier["methods"]["dpo"], key=lambda p: p["beta"])
    xs = list(range(len(pts)))
    ax[1].plot(xs, [p["reward"] for p in pts], marker="o", color=C_DPO,
               linewidth=2.2, label="DPO")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f'{p["beta"]:g}' for p in pts])
    ax[1].axhline(base_r, color=MUTED, linestyle="--", linewidth=1)
    ax[1].text(0, base_r, " base (SFT)", color=MUTED, fontsize=8, va="top")
    ax[1].set_xlabel("DPO β")
    ax[1].set_ylabel("mean reward (RM score)")
    _title(ax[1], "B · DPO's β runs the opposite way")
    ax[1].legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax[1])

    # C -- reward vs step, GRPO vs PPO.
    for name, color in (("grpo", C_GRPO), ("ppo", C_PPO)):
        tr = sample_eff[name]["reward_trace"]
        steps = range(1, len(tr) + 1)
        ax[2].plot(steps, tr, color=color, alpha=0.2, linewidth=1)
        ax[2].plot(steps, _ema(tr), color=color, linewidth=2.4, label=name.upper())
    ax[2].set_xlabel("training step")
    ax[2].set_ylabel("mean reward (RM score)")
    _title(ax[2], "C · GRPO climbs; PPO's critic lags")
    ax[2].legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax[2])

    fig.tight_layout()
    out = os.path.join(RESULTS, "alignment.png")
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print("wrote", out)


if __name__ == "__main__":
    main()
