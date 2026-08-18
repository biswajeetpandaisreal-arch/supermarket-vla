# DECISIONS.md — Supermarket VLA Robot

Running log of design decisions not fully specified by the build instructions.
One line of reasoning each.

## D0 — Separate project folder
Built in `projects/supermarket_vla/`, adapting the reference pipeline at
`projects/smolvla_ur10e/` by copying. Reference project is never modified
(per hard constraint). User requested a new folder.

## D1 — ⚠ VRAM: 12 GB actual vs 16–24 GB assumed  [NEEDS USER ACK]
`nvidia-smi` reports **RTX A2000 12 GB**, not the 16–24 GB the instructions'
Environment section states. This is not cosmetic: the reference project already
has `train_smolvla_vision_unfrozen.py` and `PROBLEMS_AND_SOLUTIONS.md #18/19`
documenting that the *full* unfrozen SmolVLA (403M params) will not fit 12 GB —
it forced batch_size=2 and "trained too noisily to learn anything". So Stage 4's
"batch size start 8, drop to 4 on OOM" is optimistic for full fine-tuning.
Planned path: reuse the vision-encoder-only unfreeze (86M trainable) which is the
known-good 12 GB config. Flagging before Stage 3 data collection because it
affects whether the whole plan is viable on this machine.

## D2 — UR16e not available locally → keep UR10e kinematics
`mujoco_menagerie` has `universal_robots_ur10e` and `ur5e`, but **no UR16e**.
Per instructions (Stage 1.1), we keep UR10e kinematics rather than downloading
meshes without asking. Reach/payload differ but are adequate for the shelf scene.

## D4 — Cloud VLA considered, deferred → staying on SmolVLA for now
User asked about cloud-based VLA. Findings: installed LeRobot 0.5.2 has native
`async_inference/` (policy_server + robot_client) = a ready remote-inference path,
and stronger VLAs are already available locally (`pi0`, `pi05`, `pi0_fast`,
`groot`, plus an `Isaac-GR00T` checkout). Three real paths:
  A) Cloud train / local infer (keeps offline-at-runtime; SmolVLA-base ~450M fits
     12 GB for *inference* — the 12 GB pain is a *training* problem).
  B) Cloud inference at runtime via `async_inference` with a bigger VLA (most
     capable; breaks the offline constraint; hourly GPU cost + network latency).
  C) Hosted commercial API (e.g. Gemini Robotics) — rejected: can't fine-tune on
     our own LeRobot demos, so it doesn't fit the train-on-sim-data workflow.
Decision: **user is exploring, not committing** — keep SmolVLA in the plan,
revisit model choice after the pipeline exists. Options A/B remain open and both
also mitigate D1 (VRAM).
OpenVLA asked about later: rejected as the local pick — (1) not a LeRobot policy
(needs its own stack, doesn't fit our collect→convert→lerobot_train→rollout
pipeline), (2) 7B doesn't fit 12 GB — inference needs 4-bit quant, fine-tuning
(even LoRA) ~16 GB+, infeasible locally. Designated in-pipeline upgrade path if
SmolVLA is outgrown = **π0/π0.5** (already in LeRobot); OpenVLA only viable with
cloud training.

## D5 — Mobile base: real Omron LD-60 AGV (robosuite), 3-DOF (Stage 1)
Uses robosuite's `omron_mobile_base.xml` (real AGV mesh) instead of a hand-made
box platform, at the user's request. It ships exactly the 3 mobile DOFs we need
— `joint_mobile_forward`/`side`/`yaw` with velocity actuators (kv 1000/1000/1500)
— plus a support column the arm mounts on (~0.70 m). We DROP its torso-lift joint
(`joint_torso_height`) + motor to keep the plan's 3-DOF base (hard constraint #4),
and lower its `frictionloss` 250→15 so our velocity-servo nav parks precisely
(250 stalled the base ~5 cm short). Arm nested at the base's "add robot here"
mount; basket mounted on the AGV deck. The AGV's collision pedestal already
floats ~2 mm above the floor, so no ground-friction drag (this had to be
hand-fixed on the old box platform — see history below). Supersedes the earlier
box-platform D5/D8.
Two follow-up fixes after visual review: (1) robosuite's `pedestal_feet_col`
collision box exactly encloses the AGV mesh and rendered as an opaque box hiding
it — made it invisible (alpha 0, collision kept). (2) robosuite parks the chassis
0.20 m behind the arm mount, so the arm perched on the front edge — shifted the
`wheeled_base` forward (−0.20→−0.05 m) to center the chassis under the arm.

## D10 — Supermarket surroundings from robosuite textures (Stage 1)
Made the scene read as a store aisle instead of a lone shelf, reusing robosuite
assets (per user's steer to look there first): floor = `light-gray-floor-tile`,
walls = `light-gray-plaster`, shelves = `steel-brushed` metal (gondola look) —
all robosuite textures wired the way `table_arena.xml` does. Added a second
gondola (`shelf_facing`) across the aisle. (Perimeter walls were added then
REMOVED at the user's request — "the box seems pretentious"; it's now an open
aisle, no enclosing walls.) Extra shelves + the working shelf's unused levels are stocked
with STATIC, non-colliding robosuite meshes (`decor_*`) so the store looks full
without adding DOF. robosuite has no shelf/cabinet object, so the shelf geometry
stays hand-built — only its skin is robosuite.

## D11 — Arm re-seated on the AGV column centre (Stage 1)
Arm base was at x=0 while the AGV column/chassis centre is at x=−0.05, so the arm
sat 5 cm proud of the column. Set `ARM_MOUNT_X=−0.05` so the arm sits squarely on
the column. (Shifts EE back ~5 cm; HOME wrist framing still shows products,
exact per-product grasp poses come in Stage 3.)

## D9 — Basket: raised tote in front of the arm (Stage 1)
Evolved from "flat on the deck" to a **raised tote on a short stand** at
base-relative (0.28, 0, 0.48), chosen from a brainstorm for: short place motion,
wrist-camera-visible drop, forgiving target, clear of the arm→shelf reach line.
IK-verified reachable within the REAL UR10e joint limits (0.1 mm EE error at the
drop point; note the elbow limit is ±π — a naive ±2π IK finds invalid configs).

## D12 — Project scope: two-VLA robot in a DYNAMIC store (major)
User's target (from the TIC-VLA paper, arxiv 2602.02459): a supermarket with
**moving shoppers that physically collide** with the robot; given a **text
shopping list**, the robot navigates the crowd, picks each item, and collects it.
**Two VLAs run sequentially: navigation VLA + manipulation VLA.** Stays in MuJoCo
(NOT Isaac Sim / TIC-VLA's stack) — we rebuild the dynamic-nav *capability* here.
Nav is the research-hard half; de-risked by building a reactive scripted navigator
first (baseline + demo source), then training the nav VLA on its demos — mirroring
the manipulation pipeline. Full revised plan in PLAN.md (Phases 1–6). Manipulation
= SmolVLA; two VLAs × 12 GB makes cloud-training the likely path (ties to D4/D1).

## D6 — Products: real robosuite grocery meshes (Stage 1)
Uses robosuite's textured pick-place meshes (`milk`, `can`, `bread`, `cereal`,
`bottle`) instead of hand-drawn primitives, at the user's request ("use an open
source available instead of creating"). Loaded by mirroring robosuite's own
object XML (mesh + texture + material) so they render exactly as in robosuite.
Rest height comes from each mesh's `bottom_site` z. Body names stay ours
(milk_carton, cola_can, …). 5 s settle → 0 xy-drift, no NaN (~2–3 cm settle drop
as meshes seat onto the shelf).

## D7 — Two arm poses: HOME (grasp view) + TRANSIT (navigation)
Retuned for the AGV mount (arm base ~0.70 m). `HOME_QPOS` frames the middle-shelf
products in the wrist camera (bread centered under the open gripper).
`TRANSIT_QPOS` folds the arm up over the AGV (EE at x≈0.21, clear of the shelf
front at 0.75) so the base can navigate without the arm colliding. Stability +
base-motion checks use TRANSIT.

## D8 — (superseded by D5) base floor-friction
Was: the old box platform dragged on the floor and had to be floated 6 cm up.
The Omron AGV's pedestal already floats ~2 mm above the floor, so this is moot.
Kept for history: any kinematic base MUST avoid resting on the floor plane or it
crawls under ground friction.

## D3 — Camera set: wrist + one scene camera
Reference env has 3 cameras (overhead/side/wrist). Instructions (Stage 1.5) ask
for wrist + one static scene camera facing the shelf. Plan: keep `wrist_cam`,
replace overhead/side with a single shelf-facing `scene_cam`; dataset drops to
2 image keys (was 3). Reduces per-frame data and VRAM at inference.
