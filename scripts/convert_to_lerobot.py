"""
convert_to_lerobot.py — convert collected_data/<item>/episode_*.pkl into a
LeRobotDataset (parquet + mp4) for SmolVLA training.

Each frame carries: action (7), observation.state (7), wrist + scene images
(video), and the per-episode `task` string. Frames are subsampled from the
100 Hz recording down to `--fps` so action chunks span a useful horizon.

    python scripts/convert_to_lerobot.py
"""
import sys
import argparse
import pickle
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SOURCE = Path(__file__).parent.parent / "collected_data"
STATE_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper"]
IMG = (96, 96, 3)

FEATURES = {
    "action":                    {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
    "observation.state":         {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
    "observation.images.wrist":  {"dtype": "video", "shape": IMG, "names": ["height", "width", "channels"]},
    "observation.images.scene":  {"dtype": "video", "shape": IMG, "names": ["height", "width", "channels"]},
    "observation.images.basket": {"dtype": "video", "shape": IMG, "names": ["height", "width", "channels"]},
}


def episode_files():
    return [f for d in sorted(SOURCE.glob("*")) if d.is_dir() for f in sorted(d.glob("episode_*.pkl"))]


def write_dataset(repo_id, root, files, subsample, fps):
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=FEATURES,
                               root=root, robot_type="ur10e", use_videos=True)
    for ep_path in files:
        with open(ep_path, "rb") as f:
            frames = pickle.load(f)
        task = frames[0]["instruction"]
        kept = frames[::subsample]
        for fr in kept:
            ds.add_frame({
                "action":                    fr["action"],
                "observation.state":         fr["observation.state"],
                "observation.images.wrist":  fr["observation.images.wrist"],
                "observation.images.scene":  fr["observation.images.scene"],
                "observation.images.basket": fr["observation.images.basket"],
                "task":                      task,
            })
        ds.save_episode()
        print(f"  {ep_path.parent.name}/{ep_path.name}  ({len(kept)} frames)  task=\"{task}\"")
    ds.finalize()
    return ds


def convert(repo_id, root, subsample, fps, val_holdout, val_root, max_per_item=None):
    by_item = {}
    for f in episode_files():
        by_item.setdefault(f.parent.name, []).append(f)

    train, val = [], []
    for item, fl in by_item.items():
        if max_per_item:
            fl = fl[:max_per_item]
        if val_holdout > 0 and len(fl) > val_holdout:
            train += fl[:-val_holdout]
            val += fl[-val_holdout:]
        else:
            train += fl

    print(f"Converting {len(train)} train + {len(val)} val episodes "
          f"(subsample={subsample} -> {fps} fps)\n")
    print("-- train --")
    tr = write_dataset(repo_id, root, train, subsample, fps)
    print(f"\nTrain dataset -> {tr.root}  ({tr.num_frames} frames, {tr.num_episodes} episodes)")
    if val:
        print("\n-- val (held out) --")
        vl = write_dataset(repo_id + "_val", val_root, val, subsample, fps)
        print(f"\nVal dataset -> {vl.root}  ({vl.num_frames} frames, {vl.num_episodes} episodes)")


if __name__ == "__main__":
    root_default = Path(__file__).parent.parent / "data" / "supermarket_manip_lerobot"
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="local/supermarket_manip")
    ap.add_argument("--root", default=str(root_default))
    ap.add_argument("--val-root", default=str(root_default) + "_val")
    ap.add_argument("--subsample", type=int, default=5, help="keep every Nth 100 Hz frame")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--val-holdout", type=int, default=8, help="episodes held out per item for validation")
    ap.add_argument("--max-per-item", type=int, default=None, help="cap episodes/item (for a quick test)")
    args = ap.parse_args()
    convert(args.repo_id, args.root, args.subsample, args.fps,
            args.val_holdout, args.val_root, args.max_per_item)
