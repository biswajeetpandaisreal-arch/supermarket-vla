"""
view_scene.py — open the live interactive MuJoCo viewer to watch the sim.

Run on a machine WITH a display, and do NOT set MUJOCO_GL=egl (that's offscreen):
    .venv/bin/python scripts/view_scene.py            # robot runs a little demo loop
    .venv/bin/python scripts/view_scene.py --static   # just hold the pose; orbit with the mouse

Mouse: drag to orbit, scroll to zoom, right-drag to pan.

The demo loop is collision-safe: hold the grasp pose (product in the wrist cam)
-> tuck the arm -> patrol the base along the shelf -> return.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.supermarket_env import SupermarketEnv, HOME_QPOS, TRANSIT_QPOS

CYCLE = 3500  # sim steps per demo loop


def demo_control(env, k):
    """Set arm target + base velocity for sim step k (loops every CYCLE)."""
    c = k % CYCLE
    if c < 700:                                   # hold grasp pose at a product
        env.set_arm_target(HOME_QPOS); env.set_base_velocity(0, 0, 0)
    elif c < 1200:                                # tuck the arm for travel
        env.set_arm_target(TRANSIT_QPOS); env.set_base_velocity(0, 0, 0)
    elif c < 2900:                                # patrol the base along the shelf
        env.set_arm_target(TRANSIT_QPOS)
        yt = 0.35 if c < 1750 else (-0.35 if c < 2350 else 0.0)
        err = yt - env.get_base_pose()[1]
        env.set_base_velocity(0, float(np.clip(3.0 * err, -0.6, 0.6)), 0)
    else:                                         # return to the grasp pose
        env.set_arm_target(HOME_QPOS); env.set_base_velocity(0, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true", help="hold the home pose instead of running the demo loop")
    args = ap.parse_args()

    env = SupermarketEnv(image_size=128)
    env.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 150, -20, 3.2
        viewer.cam.lookat[:] = [0.5, 0.0, 0.9]
        k = 0
        while viewer.is_running():
            if not args.static:
                demo_control(env, k)
            env.step_sim(1)
            viewer.sync()
            k += 1
    env.close()


if __name__ == "__main__":
    main()
