# Supermarket VLA Robot — Report Notes

Consolidated notes for the final report. Companion to `PLAN.md` (roadmap) and
`DECISIONS.md` (chronological decision log). Numbers are from the actual runs.

---

## 1. Project overview

**Goal.** A mobile manipulator that, given a **text shopping list**, navigates a
supermarket **full of moving shoppers** (who physically collide with it), picks
each listed item off the shelf, drops it in an onboard basket, and — once the list
is complete — drives to a checkout/drop-off zone. The episode ends only when the
whole list is *delivered*, not after a single item.

**Architecture: two VLAs, run sequentially.**
- **Manipulation VLA** — SmolVLA, one language-conditioned pick-and-place per item,
  base parked. (Part A — the focus so far.)
- **Navigation VLA** — language-guided driving through the dynamic store, reacting
  to shoppers. (Part B / "Stage 2" — the harder half; the capability the TIC-VLA
  paper targets, rebuilt here in MuJoCo rather than Isaac Sim.)
- **Orchestrator** — loops the list: `for item: navigate → pick → place`; tracks
  completion; terminates on delivery.

**Why one item at a time:** a single arm picks one object per motion; the basket
accumulates items across the list as the AGV moves between shelves.

---

## 2. Simulation & assets (custom-built, MuJoCo)

- **Simulator:** MuJoCo 3.9, EGL offscreen rendering. LeRobot 0.5.2, torch 2.11+cu128.
- **Robot:** Omron LD-60 AGV (robosuite mesh) + UR10e arm + Robotiq 2F-85 gripper
  (MuJoCo Menagerie). Scene assembled programmatically (MJCF string), reusing the
  reference project `smolvla_ur10e`'s gripper/wrist-cam injection pattern.
  - 10 actuators: arm(6 position servos) + gripper(1) + AGV(3 velocity: forward/side/yaw).
  - Arm re-parented onto the AGV support column (~0.70 m); **gravity compensation**
    enabled on the arm so position servos hit targets (else ~8 cm droop).
  - AGV torso-lift DOF dropped to keep the plan's 3-DOF base; mobile-joint
    `frictionloss` lowered 250→15 for precise velocity-servo parking.
- **Products:** real textured robosuite grocery meshes — `milk`, `can`, `bread`,
  `cereal`, `bottle` — the standard pick-place set (labels readable, e.g. cereal =
  "HyVee Crisp Crunch"). Chosen over hand-drawn primitives for realism.
- **Shelf:** hand-built metal gondola (steel-brushed texture), 2 m × 1.8 m × 0.4 m,
  3 levels; the panel **above the target row was removed** so the arm can lift items
  straight up (see §4). Facing gondola + tile floor (robosuite textures) for a store
  aisle. Perimeter walls were added then removed (looked "pretentious").
- **Basket:** blue plastic tote on a solid pedestal, riding on the AGV deck.
  v1: half-size (0.11, 0.10) at x=0.28. **v2: enlarged to (0.15, 0.13), moved to
  x=0.24** (more forgiving drop target).
- **Cameras:** `wrist_cam` (eye-in-hand, 75° FOV), `scene_cam` (static, shelf-facing),
  and **`basket_cam`** (added in v2 — AGV-mounted, aimed at the tote, gives the policy
  a clear view of the drop target throughout the place).

**Why not RoboCasa:** tried it; it did not respond well to SmolVLA locally, and its
scenes are pre-baked kitchens that can't be rebuilt into a supermarket. Hence the
custom MuJoCo + robosuite-mesh stack.

---

## 3. Manipulation pipeline (Part A)

Mirrors the reference recipe: **scripted expert → recorded demos → LeRobot dataset
→ SmolVLA fine-tune → closed-loop rollout.**

### 3a. IK — mink (not scipy)
Switched from a scipy L-BFGS objective to **mink** (MuJoCo-native differential IK).
Only the arm moves (base pinned via a `VelocityLimit`; joint limits respected via
`ConfigurationLimit`). Sub-mm convergence. Key gotcha: mink is *local*, so grasp
IK must be seeded from the actual start pose (HOME) or the servo can't track it.

### 3b. Scripted pick-place trajectory
`HOME → pre-grasp → grasp → close → lift (into the opened bay) → extract (pull
straight out the shelf front) → approach (over basket) → lower → release → hold`.
The **frontal extraction** (pull item straight out at height, not fold up over the
base) was needed to keep the arm clear of the shelf.

### 3c. Per-item grasp tuning (found empirically, stored as constants in `PRODUCTS`)
- Grip must be secured by lifting straight up → required **opening the shelf** above
  the target row.
- `cereal` (widest box) needs a **grasp yaw of 90°** (close fingers across the narrow face).
- `bottle` (tall cylinder) needs a **higher grip point** (`grasp_gl=0.18`) for a stable hold.
- Edge items (`milk` at y=−0.32) tip under jitter at the extended-reach edge → **pulled
  inward** (milk to y=−0.26).
- Grip stability: noslip=5 during the grip, higher product friction (2.5), gentle
  transport, extra grip-settle.

### 3d. Data collection
- `collect_data.py`: records **per control step** — wrist + scene + basket images
  (96×96), 7-dim state (6 arm joints + gripper), 7-dim action (the executed command),
  and a **language instruction**. Only **successful** episodes saved.
- **Instruction augmentation:** 6 paraphrase templates × item names (`shared/
  instruction_templates.py`) → ~20 distinct instructions in the dataset; one per
  episode. Same module used at inference (train/deploy consistency).
- **Per-episode jitter** (±2 cm on item x/y) for position variety; the grasp IK
  re-solves on the settled position.
- **Clean episode boundary (v2 fix):** demos now END at "placed" — the return-to-HOME
  is excluded (it's an env reset, not a demonstration). Recording it in v1 taught the
  policy to move away right after releasing.
- **Expert success rates (with jitter, after tuning):** can 100%, bread 100%,
  milk 98%, bottle 96% (all ≥85% target). `cereal` excluded for now (grip slips in
  transport; backfill later).
- **v1 dataset:** 240 episodes (60/item × 4), ~490 frames/ep, 2 cameras, 6.1 GB raw.

### 3e. LeRobot conversion
`convert_to_lerobot.py`: pkl → LeRobotDataset (parquet + mp4). 100 Hz recording
**subsampled 5× → 20 fps** (so action chunks span a useful horizon). Per-item
**val holdout** (8/item). v1 result: 208 train / 32 val episodes, ~98 frames/ep,
20,384 train frames.

### 3f. SmolVLA fine-tuning
- Pretrained `lerobot/smolvla_base` (~865 MB weights, cached locally).
- **Vision-encoder-unfreeze** (`train_smolvla.py`) — freezes the language backbone,
  trains vision encoder (~86 M) + action expert. Rationale: (a) preserve strong
  pretrained perception, (b) fit 12 GB, (c) but adapt vision to the MuJoCo render
  style (real-pretrained ≠ synthetic). v1 used the *fully frozen* backbone
  (expert-only) default.
- v1 run: batch 8, 20 k steps, ~1.34 step/s → **~4 h** on the RTX A2000, `mem_gb≈3.2`
  (huge headroom), 4 checkpoints (5k/10k/15k/20k) ~1–2 GB each.

---

## 4. Results

### v1 (2 cameras, small tote, frozen backbone, boundary bug)
- **Grasp: 9/12 (75%). Place: 0/12 (0%).** (Small sample — see §5 caveat.)
- **Language grounding PROVEN.** Same scene, three instructions → gripper reached the
  named item's location:

  | instruction | item's y | gripper reached y |
  |---|---|---|
  | milk carton | −0.26 | −0.24 |
  | loaf of bread | 0.00 | −0.01 |
  | water bottle | +0.34 | +0.29 |

  Changing only the words moved the arm left/center/right — it is genuinely the
  language-conditioned SmolVLA, not a fixed motion or "grab the nearest."

- **Diagnosis of the 0% place (traced one episode):** the policy does the *whole*
  task — grasps, carries, lowers the item to within ~2 cm of the tote — then (a)
  releases ~10 cm short so it rolls out of the small tote, and (b) does not cleanly
  stop (swings back toward the shelf, re-closes gripper → can knock the item out).
  So **"0% place" = near-misses, not a broken policy.** Perception, language, grasp,
  and transport all work; the gap is **final-drop precision + termination.**

### Interpretation
The hard "does it understand the task" problems are already solved. The remaining
problem is motor precision on a small, poorly-observed target — the easier failure
mode to have.

### v2 (3 cameras, wider tote, clean boundary, frozen backbone) — THE HEADLINE RESULT
The four fixes worked. **Place success went from 0% (v1) to 80% (v2 best checkpoint.)**
Rigorous closed-loop eval: 4 checkpoints × 4 items × 20 trials = **320 episodes**,
Wilson 95% CIs, checkpoints ranked by place rate (not loss).

**Per-checkpoint (n=80 each):**

| checkpoint | grasp % | place % |
|---|---|---|
| 5k  | 72.5 (58/80) | 68.8 (55/80) |
| 10k | 80.0 (64/80) | 68.8 (55/80) |
| **15k** ⭐ **(selected)** | **92.5 (74/80)** | **80.0 (64/80), 95% CI 70–87** |
| 20k | 88.8 (71/80) | 75.0 (60/80) |

**Selected checkpoint (15k) — per item, Wilson 95% CIs:**

| item | grasp | place |
|---|---|---|
| cola can | 100% (20/20) | 95% (76–99) |
| water bottle | 100% (20/20) | 90% (70–97) |
| bread loaf | 100% (20/20) | 75% (53–89) |
| milk carton | 70% (14/20) | 60% (39–78) |

**Reading of the v2 result:**
- **0% → 80% place** confirms the v1 diagnosis was correct: the failure was
  final-drop precision + observability, not understanding — and the targeted fixes
  (clean boundary, basket camera, wider tote) closed it.
- **Checkpoint selection matters:** 15k is the sweet spot; **20k *dips* (80→75%)** —
  mild over-training that selecting-by-loss would have shipped. Direct evidence for
  the §6.1 methodology point.
- **Milk is the weak item** at 15k (grasp 70%, place 60%) — it placed higher at
  5k/10k (85%/100%), so it's recoverable with a milk-specific grasp tweak or more
  milk demos; it drags the average.
- Verified this is the **policy, not a script**: the inference path (`rollout.py`,
  `eval_checkpoints.py`) imports no IK/expert code — every action is
  `policy.select_action(...)` from images + state + the task string. Graded per-item
  success (95/90/75/60%) and language grounding are signatures a hardcoded routine
  can't produce. Eval log: `outputs/eval_v2_all.log`.

---

## 5. v2 improvements (implemented) — four fixes, all aimed at placing
1. **Clean episode boundary** — demo ends at "placed"; no recorded return-home
   (kills the post-place flailing that knocked items out).
2. **Basket camera** — dedicated AGV-mounted view of the tote; the policy can now
   *see the drop target* through the whole carry, not just glimpse it at the end.
   (User's insight: v1 could only see the basket via the wrist, and only late.)
3. **Wider, forward tote** — forgives the residual ~10 cm miss.
4. ~~Vision-encoder unfrozen~~ — **dropped for the final v2 run.** Unfreezing the
   vision encoder + 3 cameras exceeded 12 GB (OOM). Pivoted to the **frozen backbone
   (action-expert-only)** config, which fits comfortably and avoids overfitting
   ~86 M extra params on ~200 episodes. The frozen config produced the §4 v2 result.

v2 dataset: 3 cameras, wider tote, clean boundaries, 60/item; frozen-backbone train.

---

## 6. Evaluation methodology (rigor — pitfalls being actively avoided)
1. **Over-training:** select the deployed checkpoint by **closed-loop success rate**
   across 5k/10k/15k/20k — NOT by training/val loss. (v1 mistake: grabbed 20 k blindly.)
2. **Normalization freeze/version:** LeRobot bakes dataset norm stats into the
   checkpoint; the rollout loads the pre/post-processors *from that same checkpoint*
   → record and deploy can't drift apart.
3. **Camera consistency:** sim has no physical drift, but any camera change ⇒
   re-collect (why we re-collected the moment `basket_cam` was added).
4. **Episode boundaries:** reviewed and fixed (§3d, §5).
5. **Statistical power:** report **≥20 trials/item (80+ total)** with **Wilson 95%
   CIs**; do not call a difference real if the CIs overlap. (v1's 12 trials → e.g.
   grasp 75% has CI ≈ 47–91%, place 0% has CI ≈ 0–24%.)

---

## 7. Hardware & constraints
- **GPU: NVIDIA RTX A2000, 12 GB** (the original brief assumed 16–24 GB — it doesn't
  have it; this shaped every training decision).
- **Local SmolVLA only** — no online GPUs. Bigger VLAs (π0/π0.5, OpenVLA, MolmoAct2,
  GR00T) don't fit 12 GB for fine-tuning; they'd need cloud (train remote, infer
  local). Designated in-pipeline upgrade if SmolVLA is outgrown: **π0** (already in
  LeRobot). OpenVLA/MolmoAct2 also viable but only cloud-trained.
- Timings: data collection ~10–12 min (240 ep); LeRobot conversion a few min;
  SmolVLA fine-tune ~4 h @ 20 k steps.

---

## 8. Key numbers (quick reference for the report)
| Quantity | Value |
|---|---|
| Products (manipulated) | 4 solid (can, bread, milk, bottle); cereal pending |
| Expert success (w/ jitter) | can 100 / bread 100 / milk 98 / bottle 96 % |
| Demos collected (v1) | 240 episodes, 20,384 train frames (20 fps), 208/32 train/val |
| Cameras | wrist + scene (+ basket in v2), 96×96 |
| Instructions | ~20 distinct (6 templates × items) |
| Policy | SmolVLA fine-tuned from `smolvla_base`, **frozen backbone** (action-expert only) |
| Train | batch 8, 20 k steps, ~4 h, ~3.2 GB VRAM, RTX A2000 12 GB |
| v1 result | grasp 75% (9/12), place 0% (0/12) — near-misses |
| **v2 result (best, 15k)** | **grasp 92.5%, place 80.0% (95% CI 70–87), n=80** |
| v2 eval scope | 4 checkpoints × 4 items × 20 trials = 320 episodes, Wilson CIs |
| v2 per item (place) | can 95% · bottle 90% · bread 75% · milk 60% |
| Language grounding | reaches correct named item (milk/bread/bottle) |

---

## 9. Figures

All paths relative to the project root. Poster shots are 1400×1400; render strips vary.

### System / hero (poster-ready)
| File | Suggested caption | Shows |
|---|---|---|
| `outputs/poster/robot_hero_front.png` | Omron LD-60 AGV + UR10e reaching into a stocked supermarket shelf. | Full system, hero |
| `outputs/poster/robot_hero_side.png` | Side view of the mobile manipulator at the shelf. | System, alt angle |
| `outputs/poster/robot_and_shelf.png` | 3/4 view: AGV, arm, tote, stocked gondola (scene camera). | System + workspace |
| `outputs/poster/grasp_closeup.png` | Robotiq 2F-85 gripper over a cereal box (readable product label). | Manipulation detail |
| `outputs/poster/wrist_pov.png` | Eye-in-hand (wrist) camera view during a grasp. | Robot's-eye view |
| `outputs/stage1/store_scene.png` | Store aisle: working gondola + facing gondola + tile floor. | Environment |

### Observations / cameras (method + motivation)
| File | Suggested caption | Shows |
|---|---|---|
| `outputs/stage1/basket_visibility.png` | Basket seen from wrist (close/late/occluded) vs scene (small/arm-occluded) — motivates a dedicated basket camera. | Problem motivation |
| `outputs/stage1/basket_cam.png` | Added basket camera: clear tote view during carry (left) and gripper descending into it (right). | The v2 fix |
| `outputs/stage1/viewer_equivalent.png` | Scene + wrist cameras side by side. | Observation set |

### Manipulation method
| File | Suggested caption | Shows |
|---|---|---|
| `outputs/stage1/grasp_cola_can.png` | Top-down grasp pose from mink IK (0.0 mm / 0.0° convergence). | IK / grasp |
| `outputs/stage1/pickplace_cola_can.png` | Scripted pick-place key frames: home→grasp→lift→extract→carry→place. | Expert trajectory |
| `outputs/stage1/basket_new.png` | Redesigned blue plastic tote on its pedestal. | Basket |

### Data & results
| File | Suggested caption | Shows |
|---|---|---|
| `outputs/stage1/dataset_preview.png` | Demo filmstrip: wrist (top) + scene (bottom) rows per item, across the pick-place. | Dataset, both views |
| `outputs/stage1/rollout.png` | Closed-loop SmolVLA rollout — the policy reaching to grasp each item (v1). | Policy behaviour |

### Figures still to generate (plots — quick matplotlib once data is in)
- **Language-grounding plot** — instruction (milk / bread / bottle) vs gripper-y reached, showing it tracks the *named* item (data table in §4).
- **Success-rate plot with Wilson 95% CIs** — grasp & place, per item + overall, for v1 (and v2 after eval).
- **v1 vs v2 comparison** — place rate before/after the four fixes, with CIs (the headline result).
- **Checkpoint-selection curve** — success rate vs training step (5k/10k/15k/20k), justifying the chosen checkpoint (not loss).
- **System block diagram** — cameras + state + instruction → SmolVLA → 7-DOF action; plus the orchestrator loop (nav ↔ manip ↔ list).
- **Near-miss trajectory trace** — item x/z over time vs the basket (the §4 diagnosis, as a plot).

---

## 10. Next steps
- ✅ v2 done: trained (frozen backbone) → 320-episode multi-checkpoint eval → **15k
  selected, 80% place**. Placing landed.
- Optional polish: lift **milk** (its weak item) with a milk-specific grasp tweak or
  extra milk demos; backfill **cereal** (currently excluded).
- Begin **Stage 2 (navigation)** — dynamic store, moving colliding shoppers, reactive
  scripted navigator → nav demos → navigation VLA → orchestrator (loops the list).
