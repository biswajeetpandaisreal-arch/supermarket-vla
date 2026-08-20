"""
exp5_multiitem.py — quantify the multi-item orchestrator.

Thesis section 4.8 claims the orchestrator degrades on longer lists but reports no
numbers at all. This runs many randomised lists and records success BY POSITION in
the list, which is the actual measurement: position 1 is an empty basket (in
distribution), position 2+ sees a basket with items already in it (out of
distribution, since every training demo started from an empty basket).

Retries are disabled by default so that each position is a single clean Bernoulli
trial; the retry behaviour is a separate policy on top of that.

    MUJOCO_GL=egl .venv/bin/python scripts/exp5_multiitem.py \
        --lists 20 --lengths 2 3 --csv outputs/review/exp5_multiitem.csv
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
from instruction_templates import ITEM_NAMES
from rollout import load_policy, run_episode, return_to_home

ITEMS = ["cola_can", "water_bottle", "milk_carton", "bread_loaf"]
DEFAULT_CKPT = "outputs/train/smolvla_supermarket_v2/checkpoints/015000/pretrained_model"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def instruction_for_product(product):
    return f"pick up the {ITEM_NAMES[product]} and place it in the basket"


def run_list(env, policy, pre, post, device, products, rows, list_id, seed):
    """One shopping list. The scene is reset ONCE, so collected items accumulate in
    the basket exactly as the orchestrator runs it."""
    env.reset()
    n_in_basket = 0
    for pos, product in enumerate(products, 1):
        torch.manual_seed(seed + pos)
        instr = instruction_for_product(product)
        info = {}
        placed, grasped, _ = run_episode(env, policy, pre, post, device, product, instr,
                                         do_reset=False, info=info)
        rows.append(dict(list_id=list_id, list_len=len(products), position=pos,
                         item=product, instruction=instr,
                         items_in_basket_before=n_in_basket,
                         grasp=int(grasped), placement=int(placed),
                         final_x=info.get("final_x"), final_y=info.get("final_y"),
                         final_z=info.get("final_z")))
        print(f"  list{list_id:<3d} pos{pos} ({n_in_basket} in basket) {product:14s} "
              f"{'PLACED' if placed else ('grasped' if grasped else 'fail')}", flush=True)
        n_in_basket += int(placed)
    return n_in_basket


def summarise(rows):
    print("\n" + "=" * 78)
    print("EXP-5  success by POSITION in list (the distribution-shift gradient)")
    print("=" * 78)
    lens = sorted({r["list_len"] for r in rows})
    for L in lens:
        print(f"\n  --- lists of length {L} ---")
        sub = [r for r in rows if r["list_len"] == L]
        n_lists = len({r["list_id"] for r in sub})
        for pos in range(1, L + 1):
            s = [r for r in sub if r["position"] == pos]
            k, n = sum(r["placement"] for r in s), len(s)
            g = sum(r["grasp"] for r in s)
            lo, hi = wilson(k, n)
            print(f"    position {pos}:  place {k:2d}/{n:<2d} {100*k/n:5.1f}% "
                  f"(95% CI {lo:4.0f}-{hi:3.0f})   grasp {g:2d}/{n:<2d} {100*g/n:5.1f}%")
        full = sum(1 for lid in {r["list_id"] for r in sub}
                   if all(r["placement"] for r in sub if r["list_id"] == lid))
        lo, hi = wilson(full, n_lists)
        print(f"    COMPLETE list: {full}/{n_lists} {100*full/n_lists:5.1f}% (95% CI {lo:4.0f}-{hi:3.0f})")

    print("\n  --- by basket occupancy at the time of the pick ---")
    occ = sorted({r["items_in_basket_before"] for r in rows})
    for o in occ:
        s = [r for r in rows if r["items_in_basket_before"] == o]
        k, n = sum(r["placement"] for r in s), len(s)
        lo, hi = wilson(k, n)
        print(f"    {o} item(s) already in basket: {k:2d}/{n:<2d} {100*k/n:5.1f}% "
              f"(95% CI {lo:4.0f}-{hi:3.0f})")
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--lists", type=int, default=20, help="lists per length")
    ap.add_argument("--lengths", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, pre, post = load_policy(args.checkpoint, device)
    env = SupermarketEnv(image_size=96)
    rng = np.random.default_rng(args.seed)
    rows = []
    lid = 0
    for L in args.lengths:
        print(f"\n### lists of length {L} ({args.lists} lists)")
        for _ in range(args.lists):
            products = list(rng.choice(ITEMS, size=L, replace=False))
            run_list(env, policy, pre, post, device, products, rows, lid, args.seed + 100 * lid)
            lid += 1
    env.close()
    summarise(rows)
    if args.csv and rows:
        out = Path(args.csv); out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n[csv] {len(rows)} picks -> {out}")
