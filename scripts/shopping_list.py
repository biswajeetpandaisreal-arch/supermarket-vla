"""
shopping_list.py — the shopping-list ORCHESTRATOR for the manipulation VLA.

Shows the shelf's item MENU, asks which you want, then the robot works through
your selection IN ORDER: for each item it runs the fine-tuned SmolVLA policy to
pick that product off the shelf and drop it in the basket. Items ACCUMULATE — the
scene is reset once at the start, not between items, so the basket fills up.
Selection is constrained to the displayed catalog (numbers or item names), so the
robot is never asked for something it can't pick.

(The base stays parked; driving between shelves is Part B / the navigation VLA.
This is the Part-A orchestrator: item selection -> a sequence of pick-and-places.)

    # interactive: shows the menu and asks what you want
    python scripts/shopping_list.py

    # non-interactive: give the selection directly (numbers or names)
    python scripts/shopping_list.py --list "1 3"

    # watch it live + read the model's plan (needs a display -> no MUJOCO_GL=egl):
    python scripts/shopping_list.py --list "milk, cola" --think --view
"""
import sys
import time
import argparse
from pathlib import Path

import re

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent))
from envs.supermarket_env import SupermarketEnv
from instruction_templates import ITEM_NAMES
from rollout import load_policy, run_episode, run_episode_explain

# default: the best checkpoint from the eval (15k, 80% task completion)
DEFAULT_CKPT = "outputs/train/smolvla_supermarket_v2/checkpoints/015000/pretrained_model"

# The shelf catalog, in menu order. This is the ONLY source of what's available —
# derived from the products the env actually has, so the menu can never offer an
# item the robot can't pick. (cereal is excluded: its grasp isn't reliable yet.)
MENU = ["milk_carton", "cola_can", "bread_loaf", "water_bottle"]
# convenience: accept the item's own words as a typed choice (e.g. "milk", "cola")
NAME_WORDS = {
    "milk_carton":  ["milk", "carton"],
    "cola_can":     ["cola", "coke", "soda"],
    "bread_loaf":   ["bread", "loaf"],
    "water_bottle": ["water", "bottle"],
}


def show_menu():
    print("\nAvailable items on the shelf:")
    for i, prod in enumerate(MENU, 1):
        print(f"   {i}. {ITEM_NAMES[prod]}")
    print()


def parse_selection(text):
    """Turn a menu selection into an ORDERED product list, preserving the order the
    user typed. Accepts numbers ('1 3'), item words ('milk, cola'), or a mix.
    Selection is matched ONLY against the displayed catalog — no free-form guessing.
    Returns (products, bad_tokens)."""
    tokens = [t for t in re.split(r"[,\s]+", text.strip().lower()) if t]
    products, bad = [], []
    for tok in tokens:
        if tok.isdigit() and 1 <= int(tok) <= len(MENU):     # pick by number
            products.append(MENU[int(tok) - 1])
            continue
        hit = next((p for p in MENU if any(w in tok for w in NAME_WORDS[p])), None)  # or by name
        (products.append(hit) if hit else bad.append(tok))
    return products, bad


def prompt_for_list():
    """Show the menu and read a selection, re-asking until it's valid."""
    show_menu()
    while True:
        text = input("What would you like? (numbers like '1 3' or names like 'milk, cola'): ")
        products, bad = parse_selection(text)
        if bad:
            print(f"   [!] didn't recognize: {', '.join(bad)} — use the numbers or names shown above.")
        if products:
            return products
        print("   Please choose at least one item from the list.")


def instruction_for_product(product):
    return f"pick up the {ITEM_NAMES[product]} and place it in the basket"


def collect_list(env, policy, pre, post, device, products, viewer=None, think=False, retries=2):
    """Run the whole list in order. Returns list of (product, placed) results.
    Each item gets up to `retries` extra attempts if it misses the basket — the
    scene is NOT reset between attempts, so an item still on the shelf is simply
    re-attempted (the basket keeps whatever was already collected)."""
    env.reset()                                   # one reset — items then accumulate
    runner = run_episode_explain if think else run_episode   # think = drive from + show the model's plan
    results = []
    for i, product in enumerate(products, 1):
        instr = instruction_for_product(product)
        name = ITEM_NAMES[product]
        print(f"[{i}/{len(products)}] {name:14s} — \"{instr}\"", flush=True)
        placed = False
        for attempt in range(1, retries + 2):     # 1 initial try + `retries` retries
            placed, grasped, _ = runner(env, policy, pre, post, device, product, instr,
                                        viewer=viewer, do_reset=False)   # keep prior items in the basket
            status = "IN BASKET ✓" if placed else ("grasped, missed basket ✗" if grasped else "failed to grasp ✗")
            tag = "" if attempt == 1 else f" (retry {attempt - 1}/{retries})"
            print(f"        -> {status}{tag}", flush=True)
            if placed or (viewer is not None and not viewer.is_running()):
                break
            if attempt <= retries:
                print(f"        ... missed, re-attempting {name}", flush=True)
        results.append((product, placed))
        if viewer is not None and not viewer.is_running():
            break
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=None, help='non-interactive selection, e.g. "1 3" or "milk, cola" (omit to be shown the menu)')
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--view", action="store_true", help="watch live in a GUI window (needs a display, no MUJOCO_GL=egl)")
    ap.add_argument("--think", action="store_true", help="stream a live readout of what the VLA is doing each step")
    ap.add_argument("--retries", type=int, default=2, help="extra attempts per item if it misses the basket (0 = no retries)")
    args = ap.parse_args()

    if args.list:                              # non-interactive: parse the given selection
        products, bad = parse_selection(args.list)
        if bad:
            print(f"[!] didn't recognize: {', '.join(bad)}")
        if not products:
            show_menu(); print("[!] Nothing valid selected. Exiting."); sys.exit(1)
    else:                                      # interactive: show the menu and ask
        products = prompt_for_list()
    print("\nShopping list:", " -> ".join(ITEM_NAMES[p] for p in products), "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, pre, post = load_policy(args.checkpoint, device)
    env = SupermarketEnv(image_size=96)

    if args.view:
        import mujoco.viewer
        env.reset()
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            v.cam.azimuth, v.cam.elevation, v.cam.distance = 150, -18, 2.6
            v.cam.lookat[:] = [0.4, -0.1, 0.9]
            results = collect_list(env, policy, pre, post, device, products, viewer=v, think=args.think, retries=args.retries)
            n_ok = sum(p for _, p in results)
            print(f"\nDelivered {n_ok}/{len(results)} items. Close the window to exit.")
            while v.is_running():
                v.sync(); time.sleep(0.05)
        env.close()
        sys.exit(0)

    results = collect_list(env, policy, pre, post, device, products, think=args.think, retries=args.retries)
    n_ok = sum(p for _, p in results)
    print("\n" + "=" * 40)
    print(f"List complete — delivered {n_ok}/{len(results)} items to the basket:")
    for product, placed in results:
        print(f"   {'✓' if placed else '✗'} {ITEM_NAMES[product]}")
    env.close()
