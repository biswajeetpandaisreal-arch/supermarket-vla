"""
make_journey_diagram.py — the "how I got here" progression figure for the
presentation (prior work → this work). Three stages with the real success
numbers: RoboCasa/GR00T (small models 0%) -> UR10e pilot (~53%) -> Supermarket
VLA (80%).

    python scripts/make_journey_diagram.py   # -> outputs/plots/journey_prior_work.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent.parent / "outputs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 5.6))
ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis("off")

STAGES = [
    dict(x=0.3, title="1 · RoboCasa + GR00T N1.6",
         lines=["Franka Panda + Omron base,", "kitchen tasks (pick → cabinet)",
                "GR00T = multi-billion params:", "powerful but won't fit 12 GB"],
         verdict="Small models tried:\nSmolVLA 0/20 · ACT 0/20",
         vc="#c62828", fc="#ffebee", ec="#c62828"),
    dict(x=4.65, title="2 · UR10e pilot (MuJoCo)",
         lines=["Simplified: UR10e + gripper,", "pick a cube (5 poses) → basket",
                "ACT vs SmolVLA compared;", "fixed the metrics & the data"],
         verdict="Small models CAN learn:\nSmolVLA ~53% · ACT ~47%",
         vc="#ef6c00", fc="#fff3e0", ec="#ef6c00"),
    dict(x=9.0, title="3 · Supermarket VLA (this work)",
         lines=["Language shopping list,", "pick named items → basket",
                "same recipe, scaled up:", "frozen backbone, data variety, rigor"],
         verdict="Task completion:\nSmolVLA 80%",
         vc="#2e7d32", fc="#e8f5e9", ec="#2e7d32"),
]

W, H = 3.7, 3.5
for s in STAGES:
    x = s["x"]
    ax.add_patch(FancyBboxPatch((x, 1.3), W, H,
                 boxstyle="round,pad=0.04,rounding_size=0.12",
                 fc=s["fc"], ec=s["ec"], lw=2.4, zorder=2))
    ax.text(x + W / 2, 4.45, s["title"], ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=s["ec"], zorder=3)
    for i, ln in enumerate(s["lines"]):
        ax.text(x + W / 2, 3.95 - i * 0.42, ln, ha="center", va="center",
                fontsize=9.8, color="#37474f", zorder=3)
    # verdict banner
    ax.add_patch(FancyBboxPatch((x + 0.25, 1.45), W - 0.5, 0.72,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc="white", ec=s["ec"], lw=1.6, zorder=3))
    ax.text(x + W / 2, 1.81, s["verdict"], ha="center", va="center",
            fontsize=9.6, fontweight="bold", color=s["vc"], zorder=4)

# arrows between stages, with a short label
def arrow(x0, x1, label):
    ax.add_patch(FancyArrowPatch((x0, 3.0), (x1, 3.0), arrowstyle="-|>",
                 mutation_scale=26, lw=3, color="#455a64", zorder=1))
    ax.text((x0 + x1) / 2, 3.35, label, ha="center", fontsize=9,
            color="#455a64", fontstyle="italic")

arrow(4.0, 4.65, "too hard /\nundeployable")
arrow(8.35, 9.0, "lessons\napplied")

# big success-number strip along the bottom
ax.text(6.5, 0.55, "success arc:   0%   →   ~53%   →   80%", ha="center",
        fontsize=15, fontweight="bold", color="#263238")

ax.set_title("How I got here: from a huge kitchen benchmark to a deployable supermarket VLA",
             fontsize=13.5, fontweight="bold", pad=10)
fig.tight_layout()
p = OUT / "journey_prior_work.png"
fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
print("wrote", p)
