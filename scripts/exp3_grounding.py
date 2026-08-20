"""
exp3_grounding.py — does the policy LOCALISE the named item, or has it learned a
word -> trajectory lookup?

The published grounding test (thesis Table 4.2) shows the arm reaches a different
lateral position for each instruction. That rules out "ignores language", but not
the simpler hypothesis: items sit at fixed y, actions are absolute joint targets,
so the policy could map each word to a memorised reach with no visual localisation
at all. These modes separate the two.

    baseline : items at trained positions (rebuilds Table 4.2 with n and spread)
    swap     : milk <-> bread positions swapped, instructions UNCHANGED
               reaches NEW position -> visual grounding
               reaches USUAL position -> word->trajectory lookup
    unseen   : each item at lateral positions never used in training

Positions are overridden by mutating envs.supermarket_env.PRODUCTS before the env
is constructed (the MuJoCo XML is built in __init__), so no defaults are edited.

    MUJOCO_GL=egl .venv/bin/python scripts/exp3_grounding.py \
        --checkpoint outputs/train/smolvla_supermarket_v2/checkpoints/015000/pretrained_model \
        --mode swap --trials 12 --csv outputs/review/exp3_swap.csv
"""
import sys
import csv
import copy
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent))
import envs.supermarket_env as env_mod
from instruction_templates import instruction_for
from rollout import load_policy, run_episode

ITEMS = ["cola_can", "water_bottle", "milk_carton", "bread_loaf"]
TRAINED_Y = {p["name"]: p["y"] for p in env_mod.PRODUCTS}

# Lateral positions never used in training (trained set: -0.26 -0.13 0.00 0.15 0.34).
# Kept inside the trained span so this tests interpolation, not extrapolation past reach.
UNSEEN_Y = [-0.20, -0.06, 0.08, 0.22]


def set_positions(y_by_name):
    """Override product y positions. Returns the original PRODUCTS for restore."""
    original = copy.deepcopy(env_mod.PRODUCTS)
    for p in env_mod.PRODUCTS:
        if p["name"] in y_by_name:
            p["y"] = float(y_by_name[p["name"]])
    return original


def run_block(policy, pre, post, device, layout, targets, trials, base_seed, rows, tag):
    """One block: build an env at `layout`, then run `trials` per item in `targets`."""
    original = set_positions(layout)
    try:
        env = env_mod.SupermarketEnv(image_size=96)
        for i, item in enumerate(targets):
            actual_y = layout.get(item, TRAINED_Y[item])
            for ep in range(trials):
                s = base_seed + 1000 * i + ep
                rng = np.random.default_rng(s)
                torch.manual_seed(s)
                instr = instruction_for(item, rng)
                info = {}
                placed, grasped, _ = run_episode(env, policy, pre, post, device, item, instr,
                                                 jitter=0.0, rng=rng, info=info)
                reached = info.get("reached_lateral_y")
                # Which hypothesis does the reach match?
                d_actual = abs(reached - actual_y) if reached is not None else None
                d_trained = abs(reached - TRAINED_Y[item]) if reached is not None else None
                verdict = ""
                if reached is not None and abs(actual_y - TRAINED_Y[item]) > 0.05:
                    verdict = "actual" if d_actual < d_trained else "trained"
                rows.append(dict(block=tag, item=item, trial=ep, seed=s, instruction=instr,
                                 trained_y=TRAINED_Y[item], actual_y=actual_y,
                                 reached_y=reached, d_to_actual=d_actual, d_to_trained=d_trained,
                                 matches=verdict, grasp=int(grasped), placement=int(placed),
                                 final_x=info.get("final_x"), final_y=info.get("final_y"),
                                 final_z=info.get("final_z")))
                print(f"  [{tag}] {item:14s} ep{ep:<2d} actual_y={actual_y:+.3f} "
                      f"reached={reached if reached is None else round(reached, 3)} "
                      f"{verdict:8s} {'PLACED' if placed else ('grasped' if grasped else 'fail')}",
                      flush=True)
        env.close()
    finally:
        env_mod.PRODUCTS[:] = original      # always restore the module defaults


def summarise(rows):
    print("\n" + "=" * 74)
    by = {}
    for r in rows:
        by.setdefault((r["block"], r["item"]), []).append(r)
    print(f"{'block':10s} {'item':14s} {'actual_y':>9s} {'reached mean':>13s} {'sd':>6s} "
          f"{'min':>7s} {'max':>7s} {'n':>3s} {'place':>6s}")
    print("-" * 74)
    for (blk, item), rs in by.items():
        ys = [r["reached_y"] for r in rs if r["reached_y"] is not None]
        pl = sum(r["placement"] for r in rs)
        if ys:
            print(f"{blk:10s} {item:14s} {rs[0]['actual_y']:+9.3f} {np.mean(ys):13.3f} "
                  f"{np.std(ys):6.3f} {min(ys):7.3f} {max(ys):7.3f} {len(rs):3d} {pl:3d}/{len(rs)}")
        else:
            print(f"{blk:10s} {item:14s} {rs[0]['actual_y']:+9.3f} {'never closed':>13s} "
                  f"{'':6s} {'':7s} {'':7s} {len(rs):3d} {pl:3d}/{len(rs)}")
    verdicts = [r["matches"] for r in rows if r["matches"]]
    if verdicts:
        na, nt = verdicts.count("actual"), verdicts.count("trained")
        print("-" * 74)
        print(f"displaced-item reaches: matched ACTUAL position {na}/{len(verdicts)}, "
              f"matched TRAINED position {nt}/{len(verdicts)}")
        print("  -> visual grounding" if na > nt else "  -> word->trajectory lookup")
    print("=" * 74)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", choices=["baseline", "swap", "unseen", "all"], default="all")
    ap.add_argument("--items", nargs="+", default=ITEMS)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, pre, post = load_policy(args.checkpoint, device)
    rows = []

    if args.mode in ("baseline", "all"):
        print("\n### baseline — trained positions (rebuilds Table 4.2)")
        run_block(policy, pre, post, device, {}, args.items, args.trials, args.seed, rows, "baseline")

    if args.mode in ("swap", "all"):
        print("\n### swap — milk <-> bread, instructions unchanged")
        layout = {"milk_carton": TRAINED_Y["bread_loaf"], "bread_loaf": TRAINED_Y["milk_carton"]}
        run_block(policy, pre, post, device, layout, ["milk_carton", "bread_loaf"],
                  args.trials, args.seed + 100, rows, "swap")

    if args.mode in ("unseen", "all"):
        print("\n### unseen — lateral positions never used in training")
        for k, y in enumerate(UNSEEN_Y):
            for item in args.items:
                run_block(policy, pre, post, device, {item: y}, [item],
                          args.trials, args.seed + 200 + 10 * k, rows, f"unseen{y:+.2f}")

    summarise(rows)
    if args.csv and rows:
        out = Path(args.csv); out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n[csv] {len(rows)} trials -> {out}")
