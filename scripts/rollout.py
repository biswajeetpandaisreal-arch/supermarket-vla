"""
rollout.py — run the fine-tuned SmolVLA policy CLOSED-LOOP against SupermarketEnv
to see whether it actually completes the pick-and-place (not just that loss fell).

The policy observes wrist + scene cameras, arm state, and the task string, and
outputs 7-dim actions (6 arm joint targets + gripper) at ~20 Hz, which we apply.

    python scripts/rollout.py \
        --checkpoint outputs/train/smolvla_supermarket/checkpoints/020000/pretrained_model \
        --items cola_can bread_loaf --episodes 2 --capture
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from envs.supermarket_env import SupermarketEnv, HOME_QPOS, BASKET_HALF
from instruction_templates import instruction_for

ACTION_SUBSTEPS = 25   # mj steps per policy action (~20 Hz, matching the training subsample)


def return_to_home(env, viewer=None):
    """Reposition the arm to the exact trained HOME start pose between items.
    We SNAP (teleport) rather than glide on purpose: a smooth joint-space return
    sweeps the gripper across the shelf front and knocks edge items (e.g. the
    water bottle) out of place, breaking the next pick. The snap is instantaneous
    (mj_forward, no dynamics) so it never touches the shelf or the basket."""
    env.set_arm_pose(HOME_QPOS)
    env.step_sim(30)
    if viewer is not None:
        viewer.sync()


def load_policy(checkpoint, device):
    """Load a trained policy. The class is taken from the checkpoint's own
    config.json, so the same evaluation harness runs SmolVLA and the ACT baseline
    (ACT ignores the language field — that is the point of the comparison)."""
    import json
    from lerobot.policies.factory import make_pre_post_processors
    ptype = json.load(open(Path(checkpoint) / "config.json")).get("type", "smolvla")
    if ptype == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy as Policy
    else:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Policy
    policy = Policy.from_pretrained(checkpoint)
    policy.eval().to(device)
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=checkpoint)
    return policy, pre, post


def obs_batch(env, instruction, device):
    def img(a):
        return torch.from_numpy(a).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    gripper = env.data.ctrl[env.gripper_act] / 255.0
    state = np.append([env.data.qpos[a] for a in env.arm_qadr], gripper).astype(np.float32)
    return {
        "observation.images.wrist":  img(env.render_camera("wrist_cam")),
        "observation.images.scene":  img(env.render_camera("scene_cam")),
        "observation.images.basket": img(env.render_camera("basket_cam")),
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "task": [instruction],
    }


def vla_readout(env, product, a, basket, rest_z, grasped_once):
    """Human-readable view of what the policy is doing THIS step. The instruction
    and the action (grip command) are the model's actual output; `phase` is our
    plain-language interpretation of that action + the resulting scene state."""
    grip_cmd = float(np.clip(a[6], 0.0, 1.0))
    grip = "CLOSE" if grip_cmd > 0.5 else "OPEN "
    p = env.body_pos(product)
    lift = p[2] - rest_z
    horiz = float(np.hypot(p[0] - basket[0], p[1] - basket[1]))   # item's distance from basket centre
    over_basket = horiz < 0.12
    if not grasped_once and grip_cmd <= 0.5:
        phase = "REACHING for the item"
    elif not grasped_once and grip_cmd > 0.5:
        phase = "CLOSING grip on the item"
    elif grasped_once and grip_cmd > 0.5 and not over_basket:
        phase = "CARRYING to the basket"
    elif grasped_once and grip_cmd > 0.5 and over_basket:
        phase = "POSITIONING over the basket"
    else:
        phase = "RELEASING into the basket"
    return (f"  🧠 grip:{grip} lift:{lift:+.2f}m dist-to-basket:{horiz:.2f}m  | {phase}"
            f"   act[{a[0]:+.2f} {a[1]:+.2f} {a[2]:+.2f} {a[3]:+.2f} {a[4]:+.2f} {a[5]:+.2f} | g={grip_cmd:.2f}]")


JOINT_LABELS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def explain_plan(chunk, step0):
    """Turn SmolVLA's raw predicted action chunk into a readable view of the
    model's OWN plan. Everything here is decoded from `chunk` — the network's
    unnormalized output — not from the scene. `chunk`: (chunk_size, 7)."""
    g = chunk[:, 6]
    closed = g > 0.5
    spark = "".join("█" if c else "·" for c in closed)        # its planned grip timeline
    # grip-intention events inside the plan (when the model intends to grasp/release)
    events = []
    for i in range(1, len(closed)):
        if closed[i] and not closed[i - 1]:
            events.append(f"CLOSE→grasp at +{i}")
        if not closed[i] and closed[i - 1]:
            events.append(f"OPEN→release at +{i}")
    intent = "; ".join(events) if events else ("hold CLOSED" if closed[0] else "hold OPEN")
    # which joint it plans to move most, and by how much (planned travel over the chunk)
    travel = np.abs(np.diff(chunk[:, :6], axis=0)).sum(axis=0)
    j = int(np.argmax(travel))
    lines = [
        f"  🧠 SmolVLA re-planned @step {step0} — {len(chunk)}-step lookahead:",
        f"       grip plan : {spark}   (█=closed ·=open)",
        f"       intent    : {intent}",
        f"       arm plan  : most motion in {JOINT_LABELS[j]} (Δ{travel[j]:.2f} rad); total planned travel {travel.sum():.2f} rad",
    ]
    return "\n".join(lines)


def jitter_item(env, product, jitter, rng):
    """Randomly offset the target item's x/y, as collect_data does for the demos.
    Evaluation originally ran without this, so every trial saw an identical scene;
    supplying `jitter` makes eval trials vary in item pose the way training did."""
    jadr = env.model.jnt_qposadr[env.model.body(product).jntadr[0]]
    jdof = env.model.jnt_dofadr[env.model.body(product).jntadr[0]]
    env.data.qpos[jadr]     += float(rng.uniform(-jitter, jitter))
    env.data.qpos[jadr + 1] += float(rng.uniform(-jitter, jitter))
    env.data.qvel[jdof:jdof + 6] = 0.0
    env.step_sim(30)                      # re-settle at the jittered position


def tcp_pos(env):
    """Gripper centre — midpoint of the two finger pads."""
    return 0.5 * (env.body_pos("left_pad") + env.body_pos("right_pad"))


def run_episode(env, policy, pre, post, device, product, instruction, max_steps=180, capture=False, viewer=None, do_reset=True, think=False, think_every=8, jitter=0.0, rng=None, info=None):
    """`info`, if given, is filled in place with per-trial detail (start/final item
    pose, grasp-time gripper y and arm configuration) so evaluations can be
    re-scored later without re-simulating. Return value is unchanged."""
    if do_reset:
        env.reset()
    return_to_home(env, viewer)           # exact trained start pose; keeps collected items in the basket
    env.model.opt.noslip_iterations = 5
    if jitter > 0:
        jitter_item(env, product, jitter, rng or np.random.default_rng())
    policy.reset()
    basket = env.body_pos("basket").copy()
    start = env.body_pos(product).copy()
    rest_z = start[2]
    max_lift = 0.0
    closed_y = closed_q = None            # gripper y / arm pose at the first grip closure
    frames = []
    for step in range(max_steps):
        batch = pre(obs_batch(env, instruction, device))
        with torch.no_grad():
            action = policy.select_action(batch)
        a = post(action).squeeze(0).cpu().numpy()
        env.set_arm_target(a[:6])
        env.data.ctrl[env.gripper_act] = float(np.clip(a[6], 0.0, 1.0)) * 255.0
        if viewer is None:
            env.step_sim(ACTION_SUBSTEPS)
        else:
            for _ in range(5):                        # sync the live viewer through the action
                env.step_sim(5); viewer.sync(); time.sleep(0.02)
            if not viewer.is_running():
                break
        if closed_y is None and float(np.clip(a[6], 0.0, 1.0)) > 0.5:
            closed_y = float(tcp_pos(env)[1])         # where it reached when it decided to close
            closed_q = [float(env.data.qpos[adr]) for adr in env.arm_qadr]
        max_lift = max(max_lift, env.body_pos(product)[2] - rest_z)
        if think and step % think_every == 0:
            print(vla_readout(env, product, a, basket, rest_z, max_lift > 0.05), flush=True)
        if capture and step % 15 == 0:
            frames.append(env.render_camera("scene_cam"))
    p = env.body_pos(product)
    dx, dy, dz = abs(p[0] - basket[0]), abs(p[1] - basket[1]), p[2] - basket[2]
    placed = dx < 0.09 and dy < 0.08 and -0.02 < dz < 0.13
    grasped = max_lift > 0.05                 # item lifted clear of the shelf at some point
    if info is not None:
        info.update(start_x=start[0], start_y=start[1], start_z=start[2],
                    final_x=p[0], final_y=p[1], final_z=p[2],
                    basket_x=basket[0], basket_y=basket[1], basket_z=basket[2],
                    max_lift=max_lift, reached_lateral_y=closed_y,
                    joint_config_at_grasp="" if closed_q is None else " ".join(f"{v:.4f}" for v in closed_q))
    return placed, grasped, frames


def run_episode_explain(env, policy, pre, post, device, product, instruction,
                        n_replans=6, capture=False, viewer=None, do_reset=True):
    """Like run_episode, but drives control from the model's PREDICTED ACTION CHUNK
    and prints SmolVLA's own plan each time it re-plans — so you watch the model
    interpret the instruction (decide to grasp/release, plan the arm motion) rather
    than reading heuristics we wrote. Same (placed, grasped, frames) return."""
    if do_reset:
        env.reset()
    return_to_home(env, viewer)           # exact trained start pose; keeps collected items in the basket
    env.model.opt.noslip_iterations = 5
    policy.reset()
    basket = env.body_pos("basket").copy()
    rest_z = env.body_pos(product)[2]
    max_lift = 0.0
    frames = []
    n_exec = policy.config.n_action_steps            # execute the chunk, then re-plan (matches deployment)
    step = 0
    for _ in range(n_replans):
        batch = pre(obs_batch(env, instruction, device))
        chunk = post(policy.predict_action_chunk(batch)).squeeze(0).cpu().numpy()   # (chunk, 7): the model's plan
        print(explain_plan(chunk, step), flush=True)
        for a in chunk[:n_exec]:
            env.set_arm_target(a[:6])
            env.data.ctrl[env.gripper_act] = float(np.clip(a[6], 0.0, 1.0)) * 255.0
            if viewer is None:
                env.step_sim(ACTION_SUBSTEPS)
            else:
                for _ in range(5):
                    env.step_sim(5); viewer.sync(); time.sleep(0.02)
                if not viewer.is_running():
                    break
            max_lift = max(max_lift, env.body_pos(product)[2] - rest_z)
            if capture and step % 15 == 0:
                frames.append(env.render_camera("scene_cam"))
            step += 1
        if viewer is not None and not viewer.is_running():
            break
    p = env.body_pos(product)
    dx, dy, dz = abs(p[0] - basket[0]), abs(p[1] - basket[1]), p[2] - basket[2]
    placed = dx < 0.09 and dy < 0.08 and -0.02 < dz < 0.13
    grasped = max_lift > 0.05
    return placed, grasped, frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--items", nargs="+", default=["cola_can", "water_bottle", "milk_carton", "bread_loaf"])
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--view", action="store_true", help="watch one episode live (display, no MUJOCO_GL=egl)")
    ap.add_argument("--instruction", default=None, help="custom task string (which item to pick)")
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, pre, post = load_policy(args.checkpoint, device)
    rng = np.random.default_rng(args.seed)
    env = SupermarketEnv(image_size=96)

    if args.view:
        import mujoco.viewer
        item = args.items[0]
        instr = args.instruction or instruction_for(item, rng)
        print(f'Watching the policy on "{instr}"  (close the window to exit)')
        env.reset()
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            v.cam.azimuth, v.cam.elevation, v.cam.distance = 150, -18, 2.6
            v.cam.lookat[:] = [0.4, -0.1, 0.9]
            placed, grasped, _ = run_episode(env, policy, pre, post, device, item, instr, viewer=v)
            print("result:", "PLACED" if placed else ("grasped" if grasped else "fail"))
            while v.is_running():
                v.sync(); time.sleep(0.05)
        env.close()
        sys.exit(0)

    n_placed = n_grasped = n = 0
    strips = []
    for item in args.items:
        for ep in range(args.episodes):
            instr = instruction_for(item, rng)
            placed, grasped, frames = run_episode(env, policy, pre, post, device, item, instr, capture=args.capture)
            n_placed += placed
            n_grasped += grasped
            n += 1
            status = "PLACED" if placed else ("grasped" if grasped else "fail")
            print(f"  {item:14s} ep{ep}: {status:8s}  \"{instr}\"")
            if args.capture and frames:
                row = np.concatenate(frames[:8], axis=1)
                cv2.putText(row, f"{item} {status}", (4, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                strips.append(row)
    print(f"\nGrasped:          {n_grasped}/{n} ({100*n_grasped//max(n,1)}%)")
    print(f"Placed in basket: {n_placed}/{n} ({100*n_placed//max(n,1)}%)")
    if strips:
        w = max(s.shape[1] for s in strips)
        strips = [np.pad(s, ((0, 0), (0, w - s.shape[1]), (0, 0))) for s in strips]
        out = Path(__file__).parent.parent / "outputs" / "stage1" / "rollout.png"
        cv2.imwrite(str(out), cv2.cvtColor(np.concatenate(strips, axis=0), cv2.COLOR_RGB2BGR))
        print("rollout frames ->", out)
    env.close()
