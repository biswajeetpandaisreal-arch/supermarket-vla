# PLAN.md — Two-VLA Supermarket Robot in a Dynamic Store (MuJoCo)

## Vision
A mobile manipulator (Omron AGV + UR10e) works a supermarket **full of moving
shoppers**. Given a **text shopping list**, it navigates the aisles — physically
avoiding people who can block and collide with it — reaches each item's shelf,
picks it, and drops it in its basket, repeating until the list is done.

**Two VLAs, run sequentially:**
- **Manipulation VLA** — SmolVLA pick-and-place at the shelf (the proven half).
- **Navigation VLA** — language-guided driving through the dynamic store, reacting
  to moving shoppers (the research-hard half; TIC-VLA's capability, rebuilt in
  MuJoCo, not Isaac Sim).

Everything stays in **MuJoCo**, reusing the reference pipeline at
`../smolvla_ur10e/` and **robosuite** assets. We never modify the reference project.

## Order of work (user priority)
**Manipulation (SmolVLA) is PRIMARY — build and fine-tune it first**, on the
current single-aisle scene with the base parked. Navigation (dynamic store +
shoppers + nav VLA) comes after. Both VLAs are built the same way the reference
builds manipulation: **scripted expert → recorded demos → imitation-trained
policy → closed-loop.**

Each phase ends at a human verification gate (run the verify script, inspect
outputs, say "proceed"). Design choices go in `DECISIONS.md`.

---

## What already exists (scene — DONE)
`envs/supermarket_env.py` builds a MuJoCo scene: Omron LD-60 AGV + UR10e + 2F-85,
a metal shelf stocked with real robosuite grocery meshes, a raised basket on the
AGV deck (IK-reachable), wrist + scene cameras, robosuite textures, a facing
gondola. `verify_scene.py` passes. This is the manipulation workspace.

### Reuse map
| Need | Reuse from |
|------|-----------|
| Scene, AGV, arm, grasp friction | `envs/supermarket_env.py` (this project) |
| IK grasp-pose solver | `<ref>/scripts/find_perfect_grasp.py` |
| Grasp smoke test | `<ref>/scripts/test_grasp.py` |
| Scripted collection + phase structure | `<ref>/scripts/collect_data.py` |
| pkl → LeRobotDataset, language augment, val split | `<ref>/scripts/convert_to_lerobot.py` |
| VRAM-safe SmolVLA training (vision-only unfreeze) | `<ref>/scripts/train_smolvla_vision_unfrozen.py` |
| Closed-loop rollout / eval | `<ref>/scripts/rollout.py` |
| Meshes, Omron AGV, textures | robosuite |

---

# PART A — Manipulation (SmolVLA), PRIMARY

## Phase 1 — Scripted grasp expert + demo collection
Objective: scripted pick-from-shelf → place-in-basket demos, recorded to LeRobot
format. Base parked; arm-only (7 DOF).
- `scripts/find_grasp.py` (adapt `<ref>/find_perfect_grasp.py`): IK grasp/pre-grasp
  poses per product on the shelf.
- `scripts/scripted_expert.py` (adapt `<ref>/collect_data.py`): approach → grasp →
  lift → transport → release into the raised basket; keep only successful episodes.
- Per-episode randomization (seeded): target item, lateral position, shelf level
  (≥2 levels), distractors, small lighting / camera jitter. Instruction from a
  shared template module (rotate paraphrases).
- `scripts/convert_to_lerobot.py`: → `data/supermarket_manip_lerobot` (wrist + scene
  cameras, 7-dim state/action), deterministic val holdout.
- `verify_dataset.py`: open-loop replay one episode; dataset stats; 2×3 sample grid.
- PILOT 30 episodes → review, then full ~350–450 (60–80/item).
Gates: `PHASE 1a COMPLETE` (pilot), `PHASE 1b COMPLETE` (full set).

## Phase 2 — SmolVLA fine-tuning
Objective: fine-tune SmolVLA on the manipulation dataset (LeRobot entry point).
- Config mirroring the reference's successful run; **vision-only unfreeze** (86M)
  for 12 GB. ~20k steps; checkpoint + loss curves. Print wall-clock estimate after
  100 steps; stop for confirmation if >12 h (cloud/A100 decision is the user's).
- `verify_training.py`: load final ckpt, one forward pass, assert output shape
  (chunk, 7), de-normalized actions within joint limits.
Gate: `PHASE 2 COMPLETE`.

## Phase 3 — Closed-loop pick-place + VLA introspection viewer
Objective: run SmolVLA closed-loop AND see "what it's thinking."
- `scripts/rollout.py` (adapt `<ref>/rollout.py`): build obs (wrist+scene, state,
  task), query chunk, execute, re-query; strict placed-in-basket success.
- `scripts/introspect.py` — the "what is it thinking" viewer:
  1. **Predicted action chunk** overlaid as the intended gripper trajectory.
  2. **Attention/saliency heatmap** over the image given the instruction (where it
     looks — does "red can" light up the can).
  3. **Language-sensitivity probe** — swap instructions on one frame, watch the
     predicted action change.
  (SmolVLA is a direct action model — no native chain-of-thought text; a
  reasoning-VLA / ECoT-style narrator is a later optional add-on.)
- `verify_closed_loop.py`: 5 single-item episodes, wrist video, latency, success.
Gate: `PHASE 3 COMPLETE` — manipulation VLA working + introspectable.

---

# PART B — Navigation (after manipulation works)

## Phase 4 — Navigable store + moving colliding shoppers
Objective: a walkable store with shoppers that **physically collide** with the AGV.
- Extend the scene: gondolas forming aisles, open floor, start/park area.
- `envs/shoppers.py`: N **mocap** shopper agents (capsule people) on scripted
  looping paths — immovable kinematic obstacles that block/push the AGV on contact.
- `verify_store.py`: shoppers move; a shopper-into-AGV contact is DETECTED; stable.
Gate: `PHASE 4 COMPLETE`.

## Phase 5 — Scripted navigator → nav demos → Navigation VLA
- `shared/store_map.py`, `shared/instruction_templates.py` (shared with collection).
- `scripts/navigator.py`: reactive goal-seeking + moving-obstacle avoidance
  (potential-field / velocity-obstacle) from ground-truth shopper states → base vel.
- `scripts/collision_checker.py`: robot↔shopper contact + timeout.
- `verify_nav.py`: start→goal runs through the crowd; trajectory plots; collisions.
- `collect_nav_data.py` → `data/store_nav_lerobot`; train nav policy; `verify_nav_vla.py`.
Gate: `PHASE 5 COMPLETE`.

## Phase 6 — Orchestrator (shopping list) + two-VLA integration + eval
- `scripts/orchestrator.py`: parse list → per item {NAVIGATE (nav VLA) →
  MANIPULATE (SmolVLA) → CHECK} → next/retry/skip; base frozen during manip.
- `evaluate.py`: seeded held-out runs; metrics: list-completion, pick success,
  **shopper-collision rate**, run time, replans/run, CIs; scripted-nav vs nav-VLA.
- `results/eval_report.md`.
Gate: `PHASE 6 COMPLETE — project done`.

---

## Key risks / open decisions
1. **Nav VLA is the hard part** (Part B) — de-risked by the scripted navigator
   baseline (Phase 5) proving the loop before the VLA is trained.
2. **12 GB VRAM × two VLAs** (D1). Run sequentially (not both resident), but
   *training* two policies strains 12 GB → cloud-train likely. Decide at Phase 2/5.
3. **Shopper collision model** — mocap agents that block/push the AGV; tune for
   detectable contacts + stable physics (Phase 4).
4. **Interpretability** — SmolVLA has no native reasoning text; introspection =
   predicted-chunk + attention + language-probe (Phase 3). Reasoning-VLA optional.

## Model choices (running)
- Manipulation: **SmolVLA** (fits 12 GB, proven). Upgrade path: π0 (in LeRobot) via
  cloud. OpenVLA rejected (DECISIONS D4).
- Navigation: SmolVLA-style policy first; TIC-VLA-style latency-aware ideas noted
  for slow/cloud inference.
