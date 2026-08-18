"""
make_arch_diagram.py — SmolVLA architecture block diagram for the presentation
(Slide 3). Inputs -> frozen VLM backbone -> trained action expert -> action chunk
-> robot, with a closed-loop "re-plan" arrow.

    .venv/bin/python scripts/make_arch_diagram.py   # -> outputs/plots/smolvla_architecture.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath

OUT = Path(__file__).parent.parent / "outputs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

INPUT_C  = "#eceff1"; INPUT_E  = "#90a4ae"
FROZEN_C = "#e3f2fd"; FROZEN_E = "#1565c0"
TRAIN_C  = "#e8f5e9"; TRAIN_E  = "#2e7d32"
OUT_C    = "#fff3e0"; OUT_E    = "#ef6c00"
ROBOT_C  = "#f3e5f5"; ROBOT_E  = "#6a1b9a"

fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")


def box(x, y, w, h, title, sub, fc, ec, tag=None, tag_color=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.03,rounding_size=0.12",
                 fc=fc, ec=ec, lw=2.2, zorder=2))
    cx, cy = x + w / 2, y + h / 2
    ax.text(cx, cy + (0.30 if sub else 0), title, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=ec, zorder=3)
    if sub:
        ax.text(cx, cy - 0.32, sub, ha="center", va="center",
                fontsize=9.5, color="#455a64", zorder=3)
    if tag:
        ax.text(x + w - 0.12, y + h - 0.14, tag, ha="right", va="top",
                fontsize=10, fontweight="bold", color=tag_color, zorder=4)


def arrow(x0, y0, x1, y1, color="#37474f", lw=2.4, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                 mutation_scale=20, lw=lw, color=color, zorder=1))


# ── inputs (left) ─────────────────────────────────────────────────────────────
ax.text(1.5, 5.75, "INPUTS (each step)", ha="center", fontsize=10,
        fontweight="bold", color="#607d8b")
box(0.3, 4.30, 2.4, 0.95, "3 camera views", "wrist · scene · basket", INPUT_C, INPUT_E)
box(0.3, 3.15, 2.4, 0.95, "Instruction", "\"pick up the milk…\"", INPUT_C, INPUT_E)
box(0.3, 2.00, 2.4, 0.95, "Arm state", "6 joints + gripper", INPUT_C, INPUT_E)

# ── VLM backbone ──────────────────────────────────────────────────────────────
box(3.6, 2.65, 2.9, 1.9, "VLM backbone", "SmolVLM2-500M\n(vision-language)",
    FROZEN_C, FROZEN_E, tag="❄ FROZEN", tag_color=FROZEN_E)

# ── action expert ─────────────────────────────────────────────────────────────
box(7.2, 2.65, 2.9, 1.9, "Action expert", "flow-matching\ntransformer",
    TRAIN_C, TRAIN_E, tag="● TRAINED", tag_color=TRAIN_E)

# ── output chunk + robot ──────────────────────────────────────────────────────
box(10.7, 3.35, 2.1, 1.2, "50-step", "action chunk\n(joints + grip)", OUT_C, OUT_E)
box(10.7, 1.55, 2.1, 1.15, "Robot", "UR10e + gripper", ROBOT_C, ROBOT_E)

# ── arrows ────────────────────────────────────────────────────────────────────
for yin in (4.775, 3.625, 2.475):
    arrow(2.7, yin, 3.6, 3.6)                       # inputs -> backbone
arrow(6.5, 3.6, 7.2, 3.6)                            # backbone -> expert
ax.text(6.85, 3.85, "features", ha="center", fontsize=8.5, color="#607d8b")
arrow(10.1, 3.6, 10.7, 3.9)                          # expert -> chunk
arrow(11.75, 3.35, 11.75, 2.70)                      # chunk -> robot
ax.text(12.15, 3.0, "execute", ha="left", va="center", fontsize=8.5, color="#607d8b")

# closed-loop re-plan arrow: robot -> down -> left -> up into the inputs (clean elbows)
loop_c = "#6a1b9a"
verts = [(11.75, 1.55), (11.75, 0.95), (1.5, 0.95), (1.5, 1.72)]
ax.add_patch(PathPatch(MPath(verts, [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO]),
             fill=False, ec=loop_c, lw=1.8, zorder=1,
             capstyle="round", joinstyle="round"))
arrow(1.5, 1.72, 1.5, 2.00, color=loop_c, lw=1.8)          # arrowhead up into "Arm state"
ax.text(6.6, 1.16, "observe → re-plan  (closed loop)", ha="center",
        fontsize=9.5, color=loop_c, fontstyle="italic")

# ── legend ────────────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.3, 0.30), 0.35, 0.30, boxstyle="round,pad=0.02",
             fc=FROZEN_C, ec=FROZEN_E, lw=2))
ax.text(0.78, 0.45, "frozen (pretrained perception)", va="center", fontsize=9.5, color="#455a64")
ax.add_patch(FancyBboxPatch((5.1, 0.30), 0.35, 0.30, boxstyle="round,pad=0.02",
             fc=TRAIN_C, ec=TRAIN_E, lw=2))
ax.text(5.58, 0.45, "trained on our 240 demos (fits 12 GB)", va="center", fontsize=9.5, color="#455a64")

ax.set_title("SmolVLA architecture:  reads + looks  →  plans actions",
             fontsize=14, fontweight="bold", pad=12)
fig.tight_layout()
p = OUT / "smolvla_architecture.png"
fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
print("wrote", p)
