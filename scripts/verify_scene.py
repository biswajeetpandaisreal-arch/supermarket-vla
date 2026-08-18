"""
verify_scene.py — Stage 1 verification for the supermarket scene.

Checks, in order:
  1. Scene loads; prints DOF count and the actuator list.
  2. All expected bodies present (shelf, basket, mobile_base, every product).
  3. Renders and saves wrist-cam, scene-cam, and a wide "viewer-equivalent" view.
  4. Drives the base +/-0.5 m in x and y and +/-45 deg in yaw; confirms it moves.
  5. Steps physics 5 s; confirms no NaN and products stay on their shelf.

Run:  MUJOCO_GL=egl python scripts/verify_scene.py
Saves images to outputs/stage1/.
"""
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.supermarket_env import SupermarketEnv, SHELF_LEVELS, TRANSIT_QPOS

OUT = Path(__file__).parent.parent / "outputs" / "stage1"
DT = 0.002  # matches the model timestep


def drive_base(env, axis, target, kp=3.0, max_steps=3000):
    """Proportional drive of one base DOF to `target`; returns reached value."""
    idx = {"x": 0, "y": 1, "yaw": 2}[axis]
    for _ in range(max_steps):
        err = target - env.get_base_pose()[idx]
        if abs(err) < 0.008:
            break
        cmd = [0.0, 0.0, 0.0]
        cmd[idx] = float(np.clip(kp * err, -0.7, 0.7))
        env.set_base_velocity(*cmd)
        env.step_sim(1)
    env.set_base_velocity(0, 0, 0)
    return env.get_base_pose()[idx]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = SupermarketEnv(image_size=512)
    env.reset()

    # 1. DOF / actuators
    print("=" * 60)
    print(f"1. Scene loaded.  nq={env.model.nq}  nv(DOF)={env.model.nv}  nu={env.model.nu}")
    print("   Actuators:", [env.model.actuator(i).name for i in range(env.model.nu)])

    # 2. Body presence
    print("-" * 60)
    expected = ["mobile_base", "shelf", "basket", "base"] + env.product_names
    names = {env.model.body(i).name for i in range(env.model.nbody)}
    missing = [b for b in expected if b not in names]
    print(f"2. Bodies present: {len(expected) - len(missing)}/{len(expected)} expected"
          + (f"  MISSING: {missing}" if missing else "  (all present)"))

    # 3. Renders
    print("-" * 60)
    env.step_sim(200)  # settle
    wrist = env.render_camera("wrist_cam")
    scene = env.render_camera("scene_cam")
    cv2.imwrite(str(OUT / "wrist_cam.png"), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT / "scene_cam.png"), cv2.cvtColor(scene, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT / "viewer_equivalent.png"),
                cv2.cvtColor(np.concatenate([scene, wrist], axis=1), cv2.COLOR_RGB2BGR))
    print(f"3. Saved wrist_cam.png, scene_cam.png, viewer_equivalent.png -> {OUT}")

    # 4. Base motion
    print("-" * 60)
    # Tuck the arm and back the base off the shelf first: at the nominal x=0 the
    # platform front is ~0.28 m from origin and the shelf front is at 0.75 m, so
    # driving +0.5 m in x would ram the base into the shelf. Testing from x=-0.5
    # gives a collision-free +/-0.5 m of travel to confirm the actuators work.
    print("4. Base motion (arm tucked, backed off shelf; drive +/-0.5 m and +/-45 deg):")
    env.set_arm_pose(TRANSIT_QPOS)
    env.step_sim(150)
    drive_base(env, "x", -0.5)  # collision-free test origin
    for axis, tgt in [("x", 0.0), ("x", -0.5),
                      ("y", 0.5), ("y", -0.5), ("y", 0.0),
                      ("yaw", np.deg2rad(45)), ("yaw", -np.deg2rad(45)), ("yaw", 0.0)]:
        reached = drive_base(env, axis, tgt)
        unit = "deg" if axis == "yaw" else "m"
        t = np.rad2deg(tgt) if axis == "yaw" else tgt
        r = np.rad2deg(reached) if axis == "yaw" else reached
        ok = "OK" if abs(r - t) < (2.0 if axis == "yaw" else 0.02) else "FAIL"
        print(f"   base_{axis:3s} -> {t:+6.2f} {unit:3s}  reached {r:+6.2f}  [{ok}]")

    # 5. Stability: 5 s of physics, products must stay on their shelf. Arm is
    #    tucked clear of the shelf so this tests the SCENE settling, not the arm
    #    reaching into (and knocking) the product cluster.
    print("-" * 60)
    env.reset()
    env.set_arm_pose(TRANSIT_QPOS)
    rest = {n: env.body_pos(n).copy() for n in env.product_names}
    env.set_base_velocity(0, 0, 0)
    steps = int(5.0 / DT)
    env.step_sim(steps)
    nan = bool(np.isnan(env.data.qpos).any())
    shelf_floor = SHELF_LEVELS[0] - 0.05  # anything that fell off a shelf drops well below this
    print(f"5. Stepped {steps} steps (5 s).  NaN in qpos? {nan}")
    all_on = True
    for n in env.product_names:
        now = env.body_pos(n)
        drift = np.linalg.norm(now[:2] - rest[n][:2])
        on_shelf = now[2] > shelf_floor and drift < 0.10
        all_on = all_on and on_shelf
        print(f"   {n:14s} dz={now[2]-rest[n][2]:+.3f}  xy-drift={drift:.3f}  "
              f"[{'on shelf' if on_shelf else 'MOVED/FELL'}]")

    print("=" * 60)
    verdict = (not missing) and (not nan) and all_on
    print("STAGE 1 SELF-CHECK:", "PASS" if verdict else "REVIEW NEEDED")
    env.close()


if __name__ == "__main__":
    main()
