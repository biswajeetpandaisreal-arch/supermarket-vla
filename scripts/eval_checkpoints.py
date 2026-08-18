"""
eval_checkpoints.py — rigorous closed-loop evaluation of one or more SmolVLA
checkpoints. For each checkpoint x item it runs N trials (fresh instruction +
seed each), tallies grasp and place success, and reports rates with Wilson 95%
confidence intervals. Checkpoints are ranked by overall PLACE success (the real
task metric — not training loss).

    MUJOCO_GL=egl .venv/bin/python scripts/eval_checkpoints.py \
        --run outputs/train/smolvla_supermarket_v2 \
        --checkpoints 005000 010000 015000 020000 \
        --trials 20
"""
import sys
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


def eval_checkpoint(ckpt_path, items, trials, device, base_seed):
    policy, pre, post = load_policy(ckpt_path, device)
    env = SupermarketEnv(image_size=96)
    per_item = {}
    for i, item in enumerate(items):
        rng = np.random.default_rng(base_seed + i)
        g = p = 0
        for ep in range(trials):
            instr = instruction_for(item, rng)
            placed, grasped, _ = run_episode(env, policy, pre, post, device, item, instr)
            g += int(grasped)
            p += int(placed)
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
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path(args.run)
    N = args.trials * len(args.items)

    results = {}
    for ck in args.checkpoints:
        ckpt = run / "checkpoints" / ck / "pretrained_model"
        if not ckpt.exists():
            print(f"[skip] {ck}: not found at {ckpt}")
            continue
        print(f"\n=== checkpoint {ck}  ({args.trials} trials x {len(args.items)} items = {N}) ===", flush=True)
        results[ck] = eval_checkpoint(str(ckpt), args.items, args.trials, device, args.seed)

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
