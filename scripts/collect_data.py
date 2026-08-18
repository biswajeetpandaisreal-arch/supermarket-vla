"""
collect_data.py — collect scripted pick-place demonstrations to pickle files,
ready for conversion to a LeRobotDataset (Phase 1 of the manipulation pipeline).

Records, per control step: wrist + scene RGB, 7-dim state (6 arm joints + gripper),
7-dim action (the executed command), and the episode instruction. Only SUCCESSFUL
episodes are saved. Per-episode position jitter varies the item placement.

    # pilot (a few per item):
    MUJOCO_GL=egl .venv/bin/python scripts/collect_data.py --n 3
    # full set:
    MUJOCO_GL=egl .venv/bin/python scripts/collect_data.py --n 80
"""
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from envs.supermarket_env import SupermarketEnv
import scripted_expert as se
from instruction_templates import instruction_for

WORKING_ITEMS = ["cola_can", "water_bottle", "milk_carton", "bread_loaf"]


class Recorder:
    """Captures (obs, state, action) each control step during an episode."""
    def __init__(self, instruction):
        self.instruction = instruction
        self.frames = []

    def record(self, env, action):
        state = np.append([env.data.qpos[a] for a in env.arm_qadr],
                          env.data.ctrl[env.gripper_act] / 255.0).astype(np.float32)
        self.frames.append({
            "observation.images.wrist":  env.render_camera("wrist_cam"),
            "observation.images.scene":  env.render_camera("scene_cam"),
            "observation.images.basket": env.render_camera("basket_cam"),
            "observation.state": state,
            "action": action,
            "instruction": self.instruction,
        })


def collect(items, n_per_item, jitter, out_dir, image_size, seed, max_attempts_factor=3):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    env = SupermarketEnv(image_size=image_size)
    totals = {}
    for item in items:
        d = out_dir / item
        d.mkdir(parents=True, exist_ok=True)
        saved, attempts = 0, 0
        cap = n_per_item * max_attempts_factor
        while saved < n_per_item and attempts < cap:
            attempts += 1
            rec = Recorder(instruction_for(item, rng))
            se._RECORDER = rec
            ok, _, _ = se.pick_place(env, item, capture=False, jitter=jitter, rng=rng)
            se._RECORDER = None
            if ok:
                with open(d / f"episode_{saved:03d}.pkl", "wb") as f:
                    pickle.dump(rec.frames, f)
                saved += 1
                print(f"  {item:14s} [{saved}/{n_per_item}] {len(rec.frames)} frames  \"{rec.instruction}\"")
        totals[item] = (saved, attempts)
        print(f"  -> {item}: saved {saved}/{n_per_item} in {attempts} attempts\n")
    env.close()
    print("Summary:", {k: f"{s}/{a}" for k, (s, a) in totals.items()})
    return totals


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", default=WORKING_ITEMS)
    ap.add_argument("--n", type=int, default=3, help="successful episodes per item")
    ap.add_argument("--jitter", type=float, default=0.025, help="item x/y position jitter (m)")
    ap.add_argument("--out", default=str(Path(__file__).parent.parent / "collected_data"))
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    collect(args.items, args.n, args.jitter, args.out, args.image_size, args.seed)
