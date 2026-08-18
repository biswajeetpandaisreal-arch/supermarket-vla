"""
find_grasp.py — top-down grasp IK for shelf products, using **mink** (MuJoCo-native
differential IK) on our actual scene model.

Solves arm joint angles for a PRE_GRASP (hovering above) and GRASP (fingers around
the item) with the gripper pointing straight down. Only the arm moves — the AGV
base, gripper joints, and product free-bodies are held fixed via masked
integration. Used by the scripted expert to pick each product off the shelf.

    MUJOCO_GL=egl .venv/bin/python scripts/find_grasp.py --product cola_can
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import mujoco
import mink

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.supermarket_env import SupermarketEnv, HOME_QPOS, JOINT_NAMES, BASE_JOINTS

GRIPPER_LENGTH = 0.14     # wrist(attachment_site) -> fingertip distance; grasp target sits this far above the item
PRE_GRASP_UP = 0.15       # pre-grasp hover height above the grasp

# Rotation whose 3rd column (site z-axis) points straight down — a top-down grasp.
_R_DOWN = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])


def _topdown_target(target_pos, yaw_deg):
    yaw = 0.0 if yaw_deg is None else np.deg2rad(yaw_deg)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R = Rz @ _R_DOWN
    return mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), np.asarray(target_pos, float))


def solve_pose(env, target_pos, seed=None, yaw_deg=None, ori_cost=0.5, iters=200, dt=0.05):
    """mink differential IK: move ONLY the arm so the gripper reaches `target_pos`,
    preferring to point straight down (weight `ori_cost`; use ~0 for a
    position-dominant pose like a release, where exact tilt doesn't matter).
    Returns (arm_q, pos_err_m, zaxis_err_deg)."""
    m, d = env.model, env.data
    if seed is not None:
        for a, v in zip(env.arm_qadr, seed):
            d.qpos[a] = v
    mujoco.mj_forward(m, d)

    config = mink.Configuration(m)
    config.update(d.qpos)
    task = mink.FrameTask("attachment_site", "site", position_cost=1.0, orientation_cost=ori_cost, lm_damping=1e-2)
    task.set_target(_topdown_target(target_pos, yaw_deg))
    posture = mink.PostureTask(m, cost=1e-2)                 # pick a sane elbow/shoulder branch
    posture.set_target(d.qpos.copy())

    # The arm hangs off the AGV base, so the base joints DO move the gripper —
    # pin them (velocity 0) so the QP solves arm-only. (Gripper/product DOFs don't
    # affect the gripper frame, so the QP leaves them at rest on its own.)
    limits = [mink.ConfigurationLimit(m),                       # respect joint position limits (elbow ±π etc.)
              mink.VelocityLimit(m, {j: 0.0 for j in BASE_JOINTS})]
    for _ in range(iters):
        dq = mink.solve_ik(config, [task, posture], dt, solver="quadprog", damping=1e-3, limits=limits)
        config.integrate_inplace(dq, dt)
        err = task.compute_error(config)
        if np.linalg.norm(err[:3]) < 0.003 and (ori_cost < 0.2 or np.linalg.norm(err[3:]) < 0.02):
            break

    for a in env.arm_qadr:
        d.qpos[a] = config.q[a]
    mujoco.mj_forward(m, d)
    sid = m.site("attachment_site").id
    perr = np.linalg.norm(d.site_xpos[sid] - np.asarray(target_pos, float))
    zerr = np.degrees(np.arccos(np.clip(d.site_xmat[sid].reshape(3, 3)[:, 2] @ [0, 0, -1], -1, 1)))
    return np.array([config.q[a] for a in env.arm_qadr]), perr, zerr


def solve_grasp(env, product, gripper_length=GRIPPER_LENGTH, yaw_deg=None, seed=None):
    """Return (pre_grasp_q, grasp_q, pos_err, z_err) for a top-down grasp of `product`."""
    p = env.body_pos(product)
    grasp_pos = p + np.array([0, 0, gripper_length])
    grasp_q, perr, zerr = solve_pose(env, grasp_pos, seed=seed, yaw_deg=yaw_deg)
    pre_q, _, _ = solve_pose(env, grasp_pos + np.array([0, 0, PRE_GRASP_UP]), seed=grasp_q, yaw_deg=yaw_deg)
    return pre_q, grasp_q, perr, zerr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="cola_can")
    ap.add_argument("--gripper-length", type=float, default=GRIPPER_LENGTH)
    args = ap.parse_args()

    import cv2
    env = SupermarketEnv(image_size=512)
    env.reset()
    pre, grasp, perr, zerr = solve_grasp(env, args.product, gripper_length=args.gripper_length)
    print(f"{args.product}: GRASP pos err {perr*1000:.1f} mm, z-axis-down err {zerr:.1f} deg")
    print(f"  PRE_GRASP = {np.round(pre, 4).tolist()}")
    print(f"  GRASP     = {np.round(grasp, 4).tolist()}")
    # render the grasp pose
    env.set_arm_pose(grasp)
    env.step_sim(5)
    tiles = [env.render_camera("scene_cam"), env.render_camera("wrist_cam")]
    out = Path(__file__).parent.parent / "outputs" / "stage1" / f"grasp_{args.product}.png"
    cv2.imwrite(str(out), cv2.cvtColor(np.concatenate(tiles, axis=1), cv2.COLOR_RGB2BGR))
    print(f"  rendered -> {out}")
    env.close()
