# Supermarket Service Robot — a Two-VLA System in Simulation

A mobile manipulator that reads a **text shopping list**, navigates a supermarket
(eventually one full of moving shoppers), **picks each item off the shelf**, drops
it in an onboard basket, and delivers the collected list — driven by **two
vision-language-action (VLA) models**: one for **manipulation**, one for
**navigation**. Built in **MuJoCo**, trained locally on a **12 GB GPU**.

> This README is the poster-building reference. Companion docs:
> `PLAN.md` (roadmap), `DECISIONS.md` (chronological decisions), `REPORT_NOTES.md`
> (full write-up notes).

---

## TL;DR (elevator pitch)
Give the robot *"pick up the milk and place it in the basket"* and a fine-tuned
**SmolVLA** policy drives a UR10e arm on an Omron AGV to do exactly that — reading
the **language** to choose the right item among distractors, using **wrist + scene +
basket cameras** to act. A scripted expert generates demos; SmolVLA is fine-tuned on
a **12 GB** GPU by keeping its pretrained perception **frozen** and learning only the
**action mapping**. First policy already **grasps and grounds language correctly**;
current work closes the **placing** gap.

---

## 1. Motivation
- Retail/warehouse "pick from a list" is a real, hard robotics task: perception +
  language + mobile navigation + manipulation.
- **VLAs** (vision-language-action models) promise generalist robot control, but they
  are large. **Can a capable VLA be fine-tuned and run on a single consumer 12 GB
  GPU?** This project says: yes, for the manipulation half.

## 2. System
| Component | Choice |
|---|---|
| Simulator | **MuJoCo 3.9** (EGL headless rendering) |
| Mobile base | **Omron LD-60 AGV** (robosuite mesh) |
| Arm + gripper | **UR10e** + **Robotiq 2F-85** (MuJoCo Menagerie) |
| Products | Real textured **robosuite grocery meshes** (milk, can, bread, cereal, bottle) |
| Cameras | **wrist** (eye-in-hand) + **scene** (shelf-facing) + **basket** (drop target) |
| Policy | **SmolVLA** fine-tuned from `lerobot/smolvla_base` (LeRobot) |
| Hardware | **NVIDIA RTX A2000, 12 GB** (local, no cloud) |

*Custom-built scene* (not RoboCasa — that didn't suit local SmolVLA and is
kitchen-only). Store aisle: stocked metal shelf + facing gondola + tile floor.

## 3. Method (manipulation pipeline)
```
Scripted expert  →  demos (LeRobot)  →  SmolVLA fine-tune  →  closed-loop rollout
```
1. **Scripted expert** — top-down grasp via **mink** IK (MuJoCo-native), frontal
   extraction out of the shelf, carry, place. Per-item grasp tuning (yaw, height,
   position). Expert success ≥ 96–100% per item.
2. **Demos** — records wrist+scene+basket images (96×96), 7-DOF state & action, and a
   **language instruction** (6 paraphrase templates × items). Only *successful*
   episodes saved; per-episode position jitter; **episodes end at "placed"** (clean
   boundary). 240 episodes → LeRobot dataset (~20 fps, 208 train / 32 val).
3. **SmolVLA fine-tune** — **freeze** the pretrained vision+language backbone, train
   only the **action expert** (maps scene features + instruction → joint angles).
   Batch 8, 20 k steps, ~4 h, ~3 GB VRAM.

**Why freezing works:** the pretrained backbone (SmolVLM + community robot data)
already *sees* and *reads*; we only learn *how this robot acts* → 208 demos suffice.

## 4. Results

### First policy (v1)
- **Grasp: 75%** (9/12). **Place: 0%** (0/12). *(small sample — reported with the
  caveat that ≥20 trials/item + Wilson 95% CIs are needed for real claims.)*
- **Language grounding — proven.** Same scene, change only the instruction → the arm
  reaches the *named* item:

  | instruction | item location (y) | gripper reached (y) |
  |---|---|---|
  | "milk carton" | −0.26 | **−0.24** |
  | "loaf of bread" | 0.00 | **−0.01** |
  | "water bottle" | +0.34 | **+0.29** |

- **Diagnosis of place = near-misses, not failure.** Traced: the policy grasps,
  carries, and lowers the item to within **~2 cm** of the basket, then drops it
  **~10 cm short** (rolls out) and doesn't cleanly stop. So perception + language +
  grasp + transport all work; the gap is **final-drop precision + termination**.

### Targeted fixes (v2, in progress) — four changes, all aimed at placing
1. **Clean episode boundary** (demo ends at "placed" — removes post-place flailing).
2. **Basket camera** (dedicated drop-target view — the policy could previously barely
   see the basket).
3. **Wider tote** (forgives the ~10 cm miss).
4. *(Optional / shelved: vision-encoder unfreeze — memory-heavy + overfit risk.)*

## 5. Key contributions (poster bullets)
- A **complete, reproducible VLA manipulation pipeline** on a **12 GB** GPU:
  scripted expert → LeRobot dataset → SmolVLA fine-tune → closed-loop eval.
- **Language-conditioned item selection** demonstrated (reaches the correct named
  item among distractors).
- A **quantified failure analysis** turning "0% place" into an actionable diagnosis
  (near-miss precision + observation + episode-boundary bug), each addressed.
- **Honest ML methodology**: frozen-backbone rationale, normalization frozen with the
  checkpoint, checkpoint selection by *success rate* not loss, Wilson-CI reporting.

## 6. Figures (files on disk → poster use)
| File | Poster use |
|---|---|
| `outputs/poster/robot_hero_front.png` | **Hero image** — robot reaching into the shelf |
| `outputs/poster/robot_and_shelf.png` | System / workspace overview |
| `outputs/poster/grasp_closeup.png` | Manipulation detail (gripper + product) |
| `outputs/poster/wrist_pov.png` | "Robot's-eye" wrist view |
| `outputs/stage1/dataset_preview.png` | Data: wrist+scene filmstrip across a pick-place |
| `outputs/stage1/basket_visibility.png` | Motivation for the basket camera (poor visibility) |
| `outputs/stage1/basket_cam.png` | The fix: clear drop-target view |
| `outputs/stage1/pickplace_cola_can.png` | Method: scripted expert key frames |
| `outputs/stage1/rollout.png` | Result: closed-loop policy behaviour |

**Plots still to generate:** language-grounding bar chart (table in §4), success-rate
with Wilson CIs, v1-vs-v2 comparison, checkpoint-selection curve, system block diagram.

## 7. Suggested poster layout (4-column)
1. **Title + one-liner + hero image** (`robot_hero_front.png`) + Motivation.
2. **System & Method** — the pipeline arrow diagram, `pickplace_cola_can.png`,
   the 3-camera setup (`basket_cam.png`), "freeze backbone → learn action expert".
3. **Data & Training** — `dataset_preview.png`, dataset numbers, the "why 208 demos
   is enough" (frozen pretrained backbone) point.
4. **Results & Next** — language-grounding table/plot, grasp/place numbers with CIs,
   the near-miss diagnosis figure, and "Next: dynamic store + navigation VLA".

## 8. Repo (how to run)
```bash
# env deps live in ../smolvla_ur10e/.venv (symlinked as .venv); render needs EGL
MUJOCO_GL=egl .venv/bin/python scripts/verify_scene.py          # sanity-check the scene
MUJOCO_GL=egl .venv/bin/python scripts/scripted_expert.py --product cola_can   # one scripted pick-place
MUJOCO_GL=egl .venv/bin/python scripts/collect_data.py --n 60   # collect demos
MUJOCO_GL=egl .venv/bin/python scripts/convert_to_lerobot.py    # -> LeRobotDataset
.venv/bin/python -m lerobot.scripts.lerobot_train  ...          # fine-tune SmolVLA (see REPORT_NOTES)
MUJOCO_GL=egl .venv/bin/python scripts/rollout.py --checkpoint <ckpt> --view   # watch the policy
```
Key files: `envs/supermarket_env.py`, `scripts/{find_grasp,scripted_expert,collect_data,convert_to_lerobot,train_smolvla,rollout}.py`, `shared/instruction_templates.py`.

## 9. Status & next steps
- **Done:** scene, scripted expert (4/5 items), data pipeline, first SmolVLA fine-tune,
  failure diagnosis, v2 dataset + the four targeted fixes.
- **In progress:** v2 retraining (frozen-backbone config) + rigorous multi-checkpoint
  eval (5k/10k/15k/20k) by *closed-loop success rate*, ≥20 trials/item, Wilson 95% CIs.
- **Next (Part B / "Stage 2"): navigation** — dynamic store with **moving, colliding
  shoppers**, reactive scripted navigator → nav demos → **navigation VLA** →
  **orchestrator** that runs the full shopping list and delivers.

## 10. Constraints (context for the poster's "challenges")
- **12 GB VRAM** shaped every training decision (frozen backbone, batch tuning).
- **Local SmolVLA only** (no cloud); bigger VLAs (π0, OpenVLA, MolmoAct2) would need
  cloud/HPC — the designated upgrade path.
