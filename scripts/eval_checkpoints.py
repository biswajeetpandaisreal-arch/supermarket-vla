"""
eval_checkpoints.py — rigorous closed-loop evaluation of one or more SmolVLA
checkpoints. For each checkpoint x item it runs N trials (fresh instruction +
seed each), tallies grasp and place success, and reports rates with Wilson 95%
confidence intervals. Checkpoints are ranked by overall PLACE success (the real
task metric — not training loss).

    python scripts/eval_checkpoints.py \
        --run outputs/train/smolvla_supermarket_v2 \
        --checkpoints 005000 010000 015000 020000 \
        --trials 20
"""
import sys
import csv
import math
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent))
from envs.supermarket_env import SupermarketEnv
from instruction_templates import instruction_for
from rollout import load_policy, run_episode

ITEMS = ["cola_can", "water_bottle", "milk_carton", "bread_loaf"]


def wilson(k, n, z=1.96):
    """Wilson score 95% CI for k successes in n trials -> (low, high) in %."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * (centre - half), 100 * (centre + half))


def trial_seed(base_seed, item_index, ep):
    """One seed per (item, trial). Identical across checkpoints, so trials are
    PAIRED and McNemar's test applies; distinct across trials, so each trial is a
    genuinely different scene once --jitter is on."""
    return base_seed + 1000 * item_index + ep


def eval_checkpoint(ckpt_path, items, trials, device, base_seed, jitter=0.0, rows=None, ckpt_name=""):
    policy, pre, post = load_policy(ckpt_path, device)
    env = SupermarketEnv(image_size=96)
    per_item = {}
    for i, item in enumerate(items):
        g = p = 0
        for ep in range(trials):
            s = trial_seed(base_seed, i, ep)
            rng = np.random.default_rng(s)
            torch.manual_seed(s)             # flow-matching noise was previously unseeded
            instr = instruction_for(item, rng)
            info = {}
            placed, grasped, _ = run_episode(env, policy, pre, post, device, item, instr,
                                             jitter=jitter, rng=rng, info=info)
            g += int(grasped)
            p += int(placed)
            if rows is not None:
                rows.append(dict(checkpoint=ckpt_name, item=item, trial=ep, seed=s,
                                 instruction=instr, grasp=int(grasped), placement=int(placed),
                                 **info))
        per_item[item] = {"grasp": g, "place": p, "n": trials}
        gl, gh = wilson(g, trials)
        pl, ph = wilson(p, trials)
        print(f"    {item:14s}  grasp {g:2d}/{trials} ({100*g//trials:3d}%  CI {gl:4.0f}-{gh:3.0f})"
              f"   place {p:2d}/{trials} ({100*p//trials:3d}%  CI {pl:4.0f}-{ph:3.0f})", flush=True)
    env.close()
    del policy
    torch.cuda.empty_cache()
    return per_item


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/train/smolvla_supermarket_v2",
                    help="training run dir containing checkpoints/")
    ap.add_argument("--checkpoints", nargs="+", default=["005000", "010000", "015000", "020000"])
    ap.add_argument("--items", nargs="+", default=ITEMS)
    ap.add_argument("--trials", type=int, default=20, help="trials per item")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="per-trial item x/y jitter (m), matching collect_data's 0.025; "
                         "0.0 reproduces the original fixed-scene protocol")
    ap.add_argument("--csv", default=None, help="write per-trial rows to this path")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path(args.run)
    N = args.trials * len(args.items)

    results = {}
    rows = [] if args.csv else None
    for ck in args.checkpoints:
        ckpt = run / "checkpoints" / ck / "pretrained_model"
        if not ckpt.exists():
            print(f"[skip] {ck}: not found at {ckpt}")
            continue
        print(f"\n=== checkpoint {ck}  ({args.trials} trials x {len(args.items)} items = {N}"
              f", jitter={args.jitter}) ===", flush=True)
        results[ck] = eval_checkpoint(str(ckpt), args.items, args.trials, device, args.seed,
                                      jitter=args.jitter, rows=rows, ckpt_name=ck)
        if rows is not None:                       # write as we go; a crash keeps finished work
            out = Path(args.csv); out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            print(f"    [csv] {len(rows)} trials -> {out}", flush=True)

    # ---- summary table, ranked by overall place rate ----
    print("\n" + "=" * 64)
    print(f"{'ckpt':>8} | {'grasp %':>18} | {'place %':>18}")
    print("-" * 64)
    ranking = []
    for ck, per in results.items():
        G = sum(v["grasp"] for v in per.values())
        P = sum(v["place"] for v in per.values())
        gl, gh = wilson(G, N)
        pl, ph = wilson(P, N)
        ranking.append((P, ck))
        print(f"{ck:>8} | {100*G/N:5.1f}  ({gl:4.0f}-{gh:3.0f}) | {100*P/N:5.1f}  ({pl:4.0f}-{ph:3.0f})")
    print("=" * 64)
    if ranking:
        best = max(ranking)[1]
        Pb = max(ranking)[0]
        pl, ph = wilson(Pb, N)
        print(f"BEST by place rate: checkpoint {best}  ->  {100*Pb/N:.1f}%  (95% CI {pl:.0f}-{ph:.0f}%)")
