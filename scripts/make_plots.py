"""
make_plots.py — generate the two results figures for the presentation from the
actual eval counts (outputs/eval_v2_all.log). Wilson 95% CIs computed here, not
hardcoded.

    python scripts/make_plots.py   # -> outputs/plots/*.png
"""
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent.parent / "outputs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)


def wilson(k, n, z=1.96):
    """Wilson score 95% CI for k/n -> (low, high) as fractions."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


# ── data (from outputs/eval_v2_all.log) ───────────────────────────────────────
# best checkpoint (15k), place counts, n=20 per item
ITEMS = [("Cola can", 19, 20), ("Water bottle", 18, 20),
         ("Bread loaf", 15, 20), ("Milk carton", 12, 20)]
# per-checkpoint OVERALL place counts, n=80 each
CKPTS = [("5k", 55, 80), ("10k", 55, 80), ("15k", 64, 80), ("20k", 60, 80)]


# ── Figure 1: per-item place % with Wilson 95% CI error bars ──────────────────
def plot_per_item():
    names = [x[0] for x in ITEMS]
    pct = np.array([100 * k / n for _, k, n in ITEMS])
    los, his = [], []
    for _, k, n in ITEMS:
        lo, hi = wilson(k, n)
        los.append(pct[len(los)] - 100 * lo)
        his.append(100 * hi - pct[len(his)])
    err = [los, his]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(names))
    colors = ["#2e7d32" if p >= 75 else "#ef6c00" for p in pct]
    bars = ax.bar(x, pct, color=colors, width=0.6, zorder=3)
    ax.errorbar(x, pct, yerr=err, fmt="none", ecolor="#37474f",
                elinewidth=1.8, capsize=6, capthick=1.8, zorder=4)
    for xi, p in zip(x, pct):
        ax.text(xi, p + 4.5, f"{p:.0f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.axhline(80, ls="--", lw=1, color="#90a4ae", zorder=1)
    ax.text(len(names) - 0.5, 81.5, "overall 80%", ha="right", va="bottom",
            fontsize=9, color="#607d8b")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Task completion — in basket (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_title("Per-item task completion (best checkpoint, n=20 each)\nerror bars: Wilson 95% CI",
                 fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", color="#cfd8dc", zorder=0)
    fig.tight_layout()
    p = OUT / "per_item_success.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print("wrote", p)


# ── Figure 2: place % vs training step (checkpoint selection) ─────────────────
def plot_checkpoints():
    labels = [c[0] for c in CKPTS]
    pct = np.array([100 * k / n for _, k, n in CKPTS])
    los, his = [], []
    for _, k, n in CKPTS:
        lo, hi = wilson(k, n)
        los.append(pct[len(los)] - 100 * lo)
        his.append(100 * hi - pct[len(his)])
    err = [los, his]
    best = int(np.argmax(pct))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(labels))
    ax.errorbar(x, pct, yerr=err, fmt="-o", color="#1565c0", ecolor="#90caf9",
                elinewidth=1.8, capsize=6, capthick=1.8, markersize=8,
                lw=2.2, zorder=3)
    # highlight the selected (peak) checkpoint
    ax.scatter([x[best]], [pct[best]], s=260, facecolors="none",
               edgecolors="#2e7d32", linewidths=2.5, zorder=4)
    ax.annotate("selected", (x[best], pct[best]),
                textcoords="offset points", xytext=(0, 22),
                ha="center", fontsize=10, color="#2e7d32", fontweight="bold")
    # Value labels sit clear of the lower CI cap so they never overlap the whiskers.
    for xi, p, lo in zip(x, pct, los):
        ax.text(xi, p - lo - 2.0, f"{p:.0f}%", ha="center", va="top",
                fontsize=10, color="#37474f")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Training checkpoint (steps)", fontsize=11)
    ax.set_ylabel("Overall task completion (%)", fontsize=11)
    ax.set_ylim(50, 95)
    # No claim of over-training: paired McNemar tests find no significant difference
    # between any pair of checkpoints on task completion (all p >= 0.19).
    ax.set_title("Checkpoint selection by closed-loop task success (n=80 each)\n"
                 "error bars: Wilson 95% CI — differences between checkpoints are not significant",
                 fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", color="#cfd8dc", zorder=0)
    fig.tight_layout()
    p = OUT / "checkpoint_selection.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    plot_per_item()
    plot_checkpoints()
