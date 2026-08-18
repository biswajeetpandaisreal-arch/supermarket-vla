"""
scripted_expert.py — scripted top-down pick-from-shelf → place-in-basket for one
product. Foundation for manipulation demo collection (Phase 1).

Trajectory: HOME → PRE_GRASP → descend GRASP → close → lift → transport above
basket → release → retract. Grasp/place poses come from find_grasp (IK).

Smoke test (one episode, saves a key-frame strip + success verdict):
    MUJOCO_GL=egl .venv/bin/python scripts/scripted_expert.py --product cola_can
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from envs.supermarket_env import SupermarketEnv, HOME_QPOS, TRANSIT_QPOS, BASKET_HALF, PRODUCTS
from find_grasp import solve_grasp, solve_pose, GRIPPER_LENGTH

# per-item grasp tuning (None = use defaults)
_GRASP_YAW = {p["name"]: p.get("grasp_yaw") for p in PRODUCTS}   # finger-close yaw
_GRIP_LEN  = {p["name"]: p.get("grasp_gl") for p in PRODUCTS}    # grasp height (wrist->item)

OPEN, CLOSE = 0.0, 1.0
_VIEWER = None    # set in --view mode so run_phase renders each step to the live GUI
_RECORDER = None  # set during data collection to record (obs, state, action) per step
NOSLIP = 5        # noslip solver iterations during the grip (higher = firmer hold)


def _lerp(a, b, t):
    return a + (b - a) * t


def run_phase(env, target_q, g0, g1, n_steps):
    """Interpolate the arm from its current pose to target_q while ramping the
    gripper g0→g1, stepping physics along the way. Records a frame per step when
    a data-collection recorder is attached."""
    start = np.array([env.data.qpos[a] for a in env.arm_qadr])
    for i in range(n_steps):
        t = (i + 1) / n_steps
        arm_cmd = _lerp(start, target_q, t)
        grip_cmd = _lerp(g0, g1, t)
        env.set_arm_target(arm_cmd)
        env.data.ctrl[env.gripper_act] = grip_cmd * 255.0
        env.step_sim(5)
        if _RECORDER is not None:
            # action = the 7-dim command just executed (6 arm targets + gripper 0..1)
            _RECORDER.record(env, np.append(arm_cmd, grip_cmd).astype(np.float32))
        if _VIEWER is not None:
            _VIEWER.sync()
            time.sleep(0.03)      # watchable speed in the live viewer


def pick_place(env, product, capture=False, jitter=0.0, rng=None, grip_len=None):
    """Run one scripted pick-place episode. Returns (success, frames, ik_err).
    `jitter` (m) randomly offsets the target item's x/y each episode so the
    demos cover varied positions (the grasp IK adapts to the settled position).
    `grip_len` overrides the grasp height (smaller = grip lower on the item)."""
    # Settle the scene with the arm already at HOME BEFORE solving IK: mink is a
    # local solver, so solving the grasp from the actual start pose (HOME) gives
    # the matching arm branch — solving from a different pose lands a solution the
    # servo can't track accurately, and the gripper closes off-centre.
    env.reset()
    env.set_arm_pose(HOME_QPOS)
    env.step_sim(30)
    # noslip stabilizes a friction-only grip (smooth robosuite meshes slip otherwise);
    # disabled at release so the item drops cleanly into the basket.
    env.model.opt.noslip_iterations = NOSLIP

    if jitter > 0:
        rng = rng or np.random.default_rng()
        jadr = env.model.jnt_qposadr[env.model.body(product).jntadr[0]]
        jdof = env.model.jnt_dofadr[env.model.body(product).jntadr[0]]
        env.data.qpos[jadr]     += float(rng.uniform(-jitter, jitter))
        env.data.qpos[jadr + 1] += float(rng.uniform(-jitter, jitter))
        env.data.qvel[jdof:jdof + 6] = 0.0
        env.step_sim(30)            # re-settle at the jittered position

    basket = env.body_pos("basket")
    gp = env.body_pos(product)
    pre_q, grasp_q, perr, zerr = solve_grasp(env, product, yaw_deg=_GRASP_YAW.get(product),
                                             gripper_length=grip_len or _GRIP_LEN.get(product) or GRIPPER_LENGTH)
    # Frontal extraction: the middle shelf is boxed in (panel above), so we can't
    # lift up. Instead pull the item straight OUT the front at grasp height until
    # it clears the shelf front, then move over the basket and lower in.
    extract_q, _, _  = solve_pose(env, np.array([0.40, gp[1], gp[2] + 0.20]), seed=grasp_q, ori_cost=0.5)
    approach_q, _, _ = solve_pose(env, basket + np.array([0, 0, 0.50]), seed=HOME_QPOS, ori_cost=0.2)
    place_q, pperr, _ = solve_pose(env, basket + np.array([0, 0, 0.24]), seed=HOME_QPOS, ori_cost=0.05)
    env.set_arm_pose(HOME_QPOS)     # IK moved the arm; restore the HOME start pose
    env.step_sim(5)

    frames = []
    def snap(label):
        if not capture:
            return
        img = env.render_camera("scene_cam")
        cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        frames.append(img)

    snap("0 home")
    run_phase(env, pre_q,   OPEN,  OPEN,  40); snap("1 pre-grasp")
    run_phase(env, grasp_q, OPEN,  OPEN,  30); snap("2 grasp")
    run_phase(env, grasp_q, OPEN,  CLOSE, 25); snap("3 close")
    run_phase(env, grasp_q,   CLOSE, CLOSE, 15)
    run_phase(env, grasp_q,    CLOSE, CLOSE, 20)                       # extra grip settle before moving (helps tall items hold)
    run_phase(env, pre_q,      CLOSE, CLOSE, 45); snap("4 lift")       # lift straight up into the now-open bay (secures the grip)
    run_phase(env, extract_q,  CLOSE, CLOSE, 65); snap("5 extract")   # pull out the shelf front (gentle: less slip)
    run_phase(env, approach_q, CLOSE, CLOSE, 70); snap("5 approach")  # move over the basket (gentle)
    run_phase(env, place_q,    CLOSE, CLOSE, 40)
    run_phase(env, place_q,    CLOSE, CLOSE, 45); snap("6 at-basket") # settle so the arm actually reaches over the basket
    env.model.opt.noslip_iterations = 0                              # let it drop cleanly
    run_phase(env, place_q,     CLOSE, OPEN,  30); snap("7 release")  # open gripper — item drops in
    run_phase(env, place_q,     OPEN,  OPEN,  30)                     # hold over the tote while it falls
    env.step_sim(50)                                                 # let it settle in the tote
    run_phase(env, place_q,     OPEN,  OPEN,  15); snap("8 placed")   # final hold: item settled -- DEMO ENDS HERE

    # Clean episode boundary: the return-to-HOME is an environment RESET for the
    # NEXT episode, not part of the demonstration. Recording it taught v1 to move
    # away right after releasing (and knock the item back out), so exclude it.
    global _RECORDER
    _saved_recorder = _RECORDER
    _RECORDER = None
    run_phase(env, HOME_QPOS, OPEN, OPEN, 40)
    _RECORDER = _saved_recorder

    prod = env.body_pos(product)
    dx, dy, dz = abs(prod[0] - basket[0]), abs(prod[1] - basket[1]), prod[2] - basket[2]
    # Honest check: item actually settled INSIDE the basket footprint and low
    # (near the floor), not perched on the rim.
    success = dx < BASKET_HALF[0] - 0.02 and dy < BASKET_HALF[1] - 0.02 and -0.02 < dz < 0.15
    return success, frames, (perr, zerr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="cola_can")
    ap.add_argument("--view", action="store_true",
                    help="watch the pick-place live in the GUI (run on a display, NOT with MUJOCO_GL=egl)")
    args = ap.parse_args()

    if args.view:
        import mujoco.viewer
        env = SupermarketEnv(image_size=128)
        env.reset()
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            v.cam.azimuth, v.cam.elevation, v.cam.distance = 150, -18, 2.6
            v.cam.lookat[:] = [0.5, -0.1, 0.9]
            _VIEWER = v
            ok, _, (perr, zerr) = pick_place(env, args.product, capture=False)
            print(f"{args.product}: {'SUCCESS (in basket)' if ok else 'FAIL'} — "
                  f"close the window to exit. Inspect the final state.")
            while v.is_running():
                v.sync()
                time.sleep(0.05)
        env.close()
        sys.exit(0)

    env = SupermarketEnv(image_size=320)
    ok, frames, (perr, zerr) = pick_place(env, args.product, capture=True)
    print(f"{args.product}: grasp IK err {perr*1000:.1f} mm / {zerr:.1f} deg  →  "
          f"{'SUCCESS (in basket)' if ok else 'FAIL'}")
    # tile key frames into a grid
    cols = 4
    rows = [np.concatenate(frames[i:i + cols], axis=1) for i in range(0, len(frames), cols)]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    grid = np.concatenate(rows, axis=0)
    out = Path(__file__).parent.parent / "outputs" / "stage1" / f"pickplace_{args.product}.png"
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  key frames → {out}")
    env.close()
