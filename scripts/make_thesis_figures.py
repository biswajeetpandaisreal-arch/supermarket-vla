"""
make_thesis_figures.py — vector-style schematic figures for the thesis:
  1. fig_env_layout.png  — top-down environment layout (shelf, item y-positions, basket, base)
  2. fig_nearmiss.png    — side schematic of the 0% placement failure mode (near-miss)
  3. fig_pipeline.png    — experimental pipeline flow (expert -> demos -> train -> eval)

    .venv/bin/python scripts/make_thesis_figures.py   # -> outputs/plots/thesis/*.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

OUT = Path(__file__).parent.parent / "outputs" / "plots" / "thesis"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#12223a"; TEAL = "#127c74"; AMBER = "#c8721a"; GREY = "#5a6b7a"; RED = "#b3261e"


# ── 1. top-down environment layout (factual, from code constants) ─────────────
def env_layout():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-0.55, 0.75); ax.set_ylim(-0.55, 0.55)
    ax.set_xlabel("x  — depth toward shelf (m)"); ax.set_ylabel("y — lateral (m)")
    ax.set_aspect("equal")
    # shelf (back)
    ax.add_patch(Rectangle((0.53, -0.55), 0.16, 1.10, facecolor="#c9ced4", edgecolor=INK, lw=1.5))
    ax.text(0.61, 0.48, "shelf", ha="center", fontsize=10, rotation=90, color=INK)
    # items on the shelf front (x = PRODUCT_FRONT_X ~ 0.53), y positions from PRODUCTS
    items = [("milk", -0.26), ("cola", -0.13), ("bread", 0.00), ("cereal", 0.15), ("bottle", 0.34)]
    for name, y in items:
        col = "#9aa4ad" if name == "cereal" else TEAL
        ax.add_patch(Circle((0.50, y), 0.035, facecolor=col, edgecolor=INK, lw=1.2, zorder=3))
        ax.text(0.44, y, f"{name} (y={y:+.2f})", ha="right", va="center", fontsize=9, color=INK)
    ax.text(0.50, -0.42, "cereal excluded\n(grip slip)", ha="center", fontsize=7.5, color=GREY)
    # robot base + arm mount
    ax.add_patch(Circle((-0.30, 0.0), 0.12, facecolor="#e9edf1", edgecolor=INK, lw=1.5, zorder=2))
    ax.text(-0.30, 0.0, "base", ha="center", va="center", fontsize=9, color=INK)
    ax.add_patch(Circle((-0.05, 0.0), 0.03, facecolor=INK, zorder=3))
    ax.text(-0.05, -0.08, "arm mount", ha="center", fontsize=8, color=INK)
    # basket (on the base deck)
    ax.add_patch(FancyBboxPatch((0.24 - 0.13, -0.11), 0.26, 0.22, boxstyle="round,pad=0.005",
                 facecolor="#cfe3ff", edgecolor="#1565c0", lw=1.6, zorder=2))
    ax.text(0.24, -0.165, "basket", ha="center", va="center", fontsize=9, color="#1565c0")
    ax.set_title("Top-down environment layout (base parked during each pick)", fontsize=11)
    ax.grid(ls=":", color="#dfe4e9")
    fig.tight_layout(); p = OUT / "fig_env_layout.png"; fig.savefig(p, dpi=200); plt.close(fig)
    print("wrote", p)


# ── 2. near-miss failure schematic (side view) ────────────────────────────────
def nearmiss():
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Schematic of the 0% placement failure mode: near-miss at release",
                 fontsize=11)
    # basket (U-shape)
    bx, bw, bt, bh = 0.62, 0.22, 0.12, 0.26
    ax.plot([bx, bx, bx + bw, bx + bw], [bt + bh, bt, bt, bt + bh], color="#1565c0", lw=3)
    ax.text(bx + bw / 2, bt + 0.02, "basket", ha="center", fontsize=9, color="#1565c0")
    # gripper trajectory (carry then lower), releasing ~10 cm short of basket centre
    xs = [0.10, 0.22, 0.34, 0.44, 0.50, 0.52]
    ys = [0.80, 0.78, 0.70, 0.55, 0.44, 0.42]
    ax.plot(xs, ys, color=GREY, lw=2, marker="o", ms=4)
    ax.text(0.10, 0.84, "grasp & carry", fontsize=8.5, color=GREY)
    # release point
    ax.scatter([0.52], [0.42], s=90, facecolors="none", edgecolors=RED, lw=2, zorder=4)
    ax.annotate("release ~2 cm above rim\nbut ~10 cm SHORT",
                (0.52, 0.42), xytext=(0.16, 0.30), fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    # item falling short and rolling out
    ax.add_patch(Circle((0.55, 0.20), 0.028, facecolor=AMBER, edgecolor=INK, lw=1))
    ax.annotate("item drops short,\nrolls out of the tote",
                (0.57, 0.20), xytext=(0.66, 0.55), fontsize=9, color=AMBER,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.3))
    # ground line
    ax.plot([0.05, 0.95], [0.14, 0.14], color=INK, lw=1)
    fig.tight_layout(); p = OUT / "fig_nearmiss.png"; fig.savefig(p, dpi=200); plt.close(fig)
    print("wrote", p)


# ── 3. experimental pipeline flow ─────────────────────────────────────────────
def pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.2); ax.axis("off")
    stages = [("Scripted expert", "mink IK\n240 demos", TEAL),
              ("LeRobot dataset", "3 cams + state\n+ instruction", INK),
              ("Fine-tune SmolVLA", "freeze backbone,\ntrain action head", TEAL),
              ("Closed-loop eval", "success rate,\nWilson CIs", INK)]
    w, gap, h, y = 2.35, 0.55, 1.5, 0.85
    for i, (t, s, c) in enumerate(stages):
        x = 0.2 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                     facecolor=c, edgecolor="none"))
        ax.text(x + w / 2, y + h - 0.42, t, ha="center", va="center", fontsize=11.5,
                fontweight="bold", color="white")
        ax.text(x + w / 2, y + 0.42, s, ha="center", va="center", fontsize=9, color="#e8f5f2")
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + w, y + h / 2), (x + w + gap, y + h / 2),
                         arrowstyle="-|>", mutation_scale=20, lw=2.4, color=GREY))
    ax.text(5.5, 0.35, "the expert only teaches — at run-time the policy is fully in control",
            ha="center", fontsize=9.5, style="italic", color=GREY)
    fig.tight_layout(); p = OUT / "fig_pipeline.png"; fig.savefig(p, dpi=200); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    env_layout(); nearmiss(); pipeline()
