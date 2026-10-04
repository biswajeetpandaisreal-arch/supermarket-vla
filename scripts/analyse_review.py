"""
analyse_review.py — statistics for the examiner-review experiments.

Consumes the per-trial CSV written by eval_checkpoints.py --csv and produces:

  * per-item x per-checkpoint success with Wilson 95% CIs            (FIG-1)
  * placement conditional on grasp, and the grasp/placement overlap  (FIG-3)
  * McNemar exact tests between checkpoints (trials are paired)      (EXP-2)
  * re-scoring against the v1 narrow basket                          (EXP-4)

    python scripts/analyse_review.py outputs/review/exp1_instrumented_jitter.csv
"""
import sys
import math
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# v1 tote, from notes/REPORT_NOTES.md:48-50 — half-extents (x, y) and centre x.
V1_HALF, V1_CX = (0.11, 0.10), 0.28
V2_HALF, V2_CX = (0.15, 0.13), 0.24
# The published success box is hard-coded in rollout.py:149 and is STRICTER than
# the tote; it is expressed relative to the basket body, not to BASKET_HALF.
PUB_HALF = (0.09, 0.08)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k:2d}/{n:<2d} {100*k/n:5.1f}% ({lo:4.0f}-{hi:3.0f})"


def mcnemar_exact(a, b):
    """Exact two-sided McNemar on paired binary vectors. Returns (n01, n10, p)."""
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n01, n10, min(1.0, 2 * tail)


def main(path):
    df = pd.read_csv(path)
    df["checkpoint"] = df.checkpoint.astype(str).str.zfill(6)
    cks = sorted(df.checkpoint.unique())
    items = list(df.item.unique())
    print(f"loaded {len(df)} trials — {len(cks)} checkpoints x {len(items)} items\n")

    # ---- FIG-1: per-item x per-checkpoint, with Wilson CIs ----------------
    print("=" * 100)
    print("FIG-1  per-item x per-checkpoint (placement, with Wilson 95% CI)")
    print("=" * 100)
    print(f"{'ckpt':>7} | " + " | ".join(f"{i:^22}" for i in items) + " |      OVERALL")
    for ck in cks:
        cells = []
        for it in items:
            s = df[(df.checkpoint == ck) & (df.item == it)]
            cells.append(fmt(int(s.placement.sum()), len(s)))
        o = df[df.checkpoint == ck]
        print(f"{ck:>7} | " + " | ".join(cells) + " | " + fmt(int(o.placement.sum()), len(o)))

    print("\n" + "-" * 100)
    print("       grasp, same layout")
    for ck in cks:
        cells = []
        for it in items:
            s = df[(df.checkpoint == ck) & (df.item == it)]
            cells.append(fmt(int(s.grasp.sum()), len(s)))
        o = df[df.checkpoint == ck]
        print(f"{ck:>7} | " + " | ".join(cells) + " | " + fmt(int(o.grasp.sum()), len(o)))

    # ---- FIG-3: placement conditional on grasp ---------------------------
    print("\n" + "=" * 100)
    print("FIG-3  grasp / placement overlap  (tests whether placement implies grasp)")
    print("=" * 100)
    print(f"{'ckpt':>7} {'item':>14} {'grasp':>7} {'place':>7} {'g&p':>5} {'placed NOT grasped':>20} {'P(place|grasp)':>15}")
    for ck in cks:
        for it in items:
            s = df[(df.checkpoint == ck) & (df.item == it)]
            g, p = int(s.grasp.sum()), int(s.placement.sum())
            both = int(((s.grasp == 1) & (s.placement == 1)).sum())
            pne = int(((s.grasp == 0) & (s.placement == 1)).sum())
            cond = f"{100*both/g:5.1f}%" if g else "   n/a"
            flag = "  <-- ANOMALY" if pne else ""
            print(f"{ck:>7} {it:>14} {g:7d} {p:7d} {both:5d} {pne:20d} {cond:>15}{flag}")

    # ---- EXP-2: McNemar across checkpoints (paired by seed) ---------------
    print("\n" + "=" * 100)
    print("EXP-2  McNemar exact test between checkpoints (paired on seed)")
    print("=" * 100)
    for metric in ("placement", "grasp"):
        print(f"\n  --- {metric} ---")
        piv = df.pivot_table(index="seed", columns="checkpoint", values=metric)
        for c1, c2 in itertools.combinations(cks, 2):
            sub = piv[[c1, c2]].dropna()
            a, b = sub[c1].values.astype(int), sub[c2].values.astype(int)
            n01, n10, p = mcnemar_exact(a, b)
            sig = "SIGNIFICANT" if p < 0.05 else "not significant"
            print(f"  {c1} ({a.sum():2d}) vs {c2} ({b.sum():2d})  n={len(sub)}  "
                  f"discordant {c2}-only={n01:2d} {c1}-only={n10:2d}  p={p:.3f}  {sig}")

    # ---- EXP-4: re-score against the v1 narrow basket ---------------------
    print("\n" + "=" * 100)
    print("EXP-4  re-scoring the SAME rollouts against the v1 narrow tote")
    print("=" * 100)
    if not {"final_x", "final_y", "final_z"}.issubset(df.columns):
        print("  final item positions not present — cannot re-score")
        return
    bz = df.basket_z.iloc[0]

    def score(cx, hx, hy):
        return ((df.final_x - cx).abs() < hx) & ((df.final_y - 0.0).abs() < hy) \
               & (df.final_z > bz - 0.02) & (df.final_z < bz + 0.13)

    variants = [
        ("published box  (rollout.py:149, 0.18x0.16 m @ x=0.24)", PUB_HALF[0], PUB_HALF[1], V2_CX),
        ("v2 tote footprint (0.30x0.26 m @ x=0.24)",              V2_HALF[0], V2_HALF[1], V2_CX),
        ("v1 tote footprint (0.22x0.20 m @ x=0.28)",              V1_HALF[0], V1_HALF[1], V1_CX),
        ("v1 box, published ratio (0.13x0.12 m @ x=0.28)",        0.066, 0.062, V1_CX),
    ]
    print(f"{'criterion':<56} {'overall':>10}   per-checkpoint")
    for name, hx, hy, cx in variants:
        ok = score(cx, hx, hy)
        per = "  ".join(f"{c}:{100*ok[df.checkpoint == c].mean():5.1f}%" for c in cks)
        print(f"{name:<56} {100*ok.mean():9.1f}%   {per}")
    print("\n  note: the published criterion is NOT the tote footprint — it is hard-coded and")
    print("        stricter, so widening the tote did not by itself relax the metric.")

    # sanity: does the recomputed published box reproduce the logged flag?
    ok = score(V2_CX, *PUB_HALF)
    agree = (ok.astype(int) == df.placement).mean()
    print(f"  sanity check: recomputed published box agrees with logged placement on "
          f"{100*agree:.1f}% of trials")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/review/exp1_instrumented_jitter.csv")
