# Project Knowledge Base — Supermarket VLA (Mobile Manipulator Grasping with VLA Models)

A single, comprehensive reference for the entire project: what it is, why it exists,
how it works, every key number, the file layout, and where things stand. Written so
that a new collaborator (human or AI) can become fully productive from this file
alone.

---

## 1. One-paragraph summary
A mobile manipulator (UR10e arm + Robotiq 2F-85 gripper on an Omron LD-60 base),
simulated in MuJoCo, is given a natural-language instruction naming a grocery item
and must pick it off a shelf and place it in an onboard basket. A compact
Vision-Language-Action model, **SmolVLA** (~450M params), is fine-tuned from
scripted-expert demonstrations by **freezing its vision-language backbone and
training only the action expert**, so it fits a **12 GB** GPU. The policy reaches
**80% task completion (95% CI 70–87), 92.5% grasp**, over **320 closed-loop
trials**. The overarching vision is a two-VLA supermarket service robot (a
navigation VLA + this manipulation VLA + an orchestrator that runs a shopping
list); this project is the **manipulation half (Part A)**.

## 2. Origin & the two-VLA vision
- Inspired by a latency-aware navigation VLA paper (the user's reference). Goal: a
  robot that, given a **text shopping list**, navigates a store full of **moving,
  colliding shoppers**, picks each listed item, and delivers the whole list.
- **Architecture:** Manipulation VLA (SmolVLA, one item at a time, base parked) +
  Navigation VLA (Part B, not yet built) + Orchestrator (loops the list:
  navigate → pick → place → deliver).
- Manipulation was prioritised first; navigation is future work.

## 3. The research journey (the thesis arc)
1. **RoboCasa + GR00T N1.6.** RoboCasa is a MuJoCo/robosuite kitchen benchmark;
   its reference model is NVIDIA GR00T N1.6 (multi-billion params). GR00T reached
   **~54% zero-shot** but is too large to fine-tune/deploy on 12 GB.
2. **Small models fail RoboCasa.** SmolVLA, ACT and Pi0 were tried on the RoboCasa
   kitchen tasks ("pick X from counter, place in cabinet") and scored **0/20** each
   (see `~/thesis_scripts/eval_results.csv`, `eval_act_results.csv`).
3. **UR10e pilot.** Stepped back to a controlled fixed-base UR10e cube pick-and-place
   (5 positions). ACT ~47%, **SmolVLA ~53%**. Three methodology lessons emerged:
   a success-metric bug (inflated 80%→ real 40%), duplicated demos (no variety),
   and vision-unfreeze OOM. (Project: `~/projects/smolvla_ur10e`.)
4. **Supermarket VLA (this project).** Applied those lessons → **80%**.
   Success arc: **0% → ~53% → 80%**.

## 4. Environment (`envs/supermarket_env.py`)
- **Simulator:** MuJoCo 3.9, EGL offscreen rendering (headless). Frameworks:
  LeRobot 0.5.2, torch 2.11+cu128.
- **Robot:** Omron LD-60 base + UR10e (6-DOF) + Robotiq 2F-85. 10 actuators:
  6 arm position servos, 1 gripper, 3 base velocity (fwd/side/yaw). Base parked
  during picks. Gravity compensation on the arm subtree (else ~8 cm droop).
  Mobile-joint frictionloss 250→15 for precise parking.
- **Products (robosuite meshes):** milk, cola can, bread, cereal, water bottle.
  Lateral positions y (m): milk −0.26, cola −0.13, bread 0.00, cereal 0.15,
  bottle 0.34. **Cereal excluded** (slips in gripper).
- **Basket:** blue plastic tote on a pedestal on the base deck, ~(0.24, 0.00) m.
  Widened in v2 to be more forgiving.
- **Cameras (all 96×96):** `wrist_cam` (eye-in-hand, 75° FOV), `scene_cam`
  (shelf-facing), `basket_cam` (drop target, added in v2).
- Key constants: `HOME_QPOS`, `TRANSIT_QPOS`, `ARM_MOUNT_X=-0.05`,
  `BASKET_POS=(0.24,0,0.47)`, `BASKET_HALF=(0.15,0.13)`.

## 5. Scripted expert (`scripts/find_grasp.py`, `scripts/scripted_expert.py`)
- **IK:** `mink` (MuJoCo-native differential IK / QP). Base pinned via VelocityLimit,
  joint limits via ConfigurationLimit. Local solver → **seed grasp IK from HOME**.
  Top-down grasp (`_R_DOWN`), `GRIPPER_LENGTH=0.14`, `PRE_GRASP_UP=0.15`.
- **Trajectory:** home → pre-grasp → grasp → close → lift into opened bay →
  **frontal extraction** (pull straight out of shelf front) → approach over basket
  → lower → release → settle. No-slip iterations during transport, disabled at
  release.
- **Per-item tuning:** cereal grasp_yaw=90; bottle higher grip (grasp_gl=0.18);
  milk moved inward from reach edge.
- **Clean episode boundary (v2 fix):** demo ENDS at placement; return-to-home is
  excluded (recording it taught the policy to move away and knock items out).
- **Expert success (w/ jitter):** cola 100, bread 100, milk 98, bottle 96%.

## 6. Data (`scripts/collect_data.py`, `scripts/convert_to_lerobot.py`)
- **240 demonstrations** (60/item × 4 items), successes only.
- Per step: 3 images (wrist/scene/basket 96×96), 7-D state (6 joints + gripper),
  7-D action, language instruction.
- **6 paraphrase templates** (`shared/instruction_templates.py`), one/episode →
  6 × 4 items = **24 distinct instructions** (was previously recorded here as "~20
  paraphrases"; 20 was a miscount of the instruction total, not the template count).
  Same module used at inference (train/deploy consistency).
- Per-episode positional jitter (±2 cm); grasp IK re-solved on settled position.
- Recorded 100 Hz → subsampled 5× → **20 fps**. LeRobot format (parquet + mp4).
  Per-item val hold-out. Normalisation stats frozen with the checkpoint.
- v2 dataset: 208 train / 32 val episodes, 3 cameras.

## 7. Training (`scripts/train_smolvla.py` and LeRobot `lerobot_train`)
- Fine-tune from `lerobot/smolvla_base`. Backbone = SmolVLM-2 (~16 VLM layers);
  action expert = flow-matching transformer; predicts a **50-step action chunk**.
- **FROZEN backbone, train only the action expert** → fits ~3 GB, no overfit.
  (Vision-unfreeze exists as a monkeypatch but OOMs with 3 cameras; dropped.)
- Config: batch 8, 20k steps, ~4 h, RTX A2000 12 GB. Checkpoints every 5k
  (5k/10k/15k/20k). Command:
  ```
  .venv/bin/python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=local/supermarket_manip \
    --dataset.root=data/supermarket_manip_lerobot \
    --policy.type=smolvla --policy.pretrained_path=lerobot/smolvla_base \
    --policy.push_to_hub=false --output_dir=outputs/train/smolvla_supermarket_v2 \
    --batch_size=8 --steps=20000 --save_freq=5000 --eval_freq=0 --wandb.enable=false
  ```

## 8. Evaluation (`scripts/rollout.py`, `scripts/eval_checkpoints.py`)
- **Closed-loop:** policy sees 3 cams + state + instruction, outputs 7-D actions at
  ~20 Hz (ACTION_SUBSTEPS=25). Metrics: **grasp** (lifted clear) and **placement /
  task completion** (rests inside basket footprint, low).
- **Rigor:** ≥20 trials/item, fresh seed + paraphrase per trial, **Wilson 95% CIs**.
- **Checkpoint selection by task success, NOT loss.** Normalisation loaded from the
  same checkpoint.

## 9. RESULTS (headline numbers — memorise these)

> ⚠️ **These numbers are under review — see `REVIEW_REPORT.md` before quoting them.**
> They are measured on a **fixed object pose**: the evaluation path applies no
> positional jitter. Under the same ±2 cm jitter used to collect the training data,
> 15k placement is **58.8%** (95% CI 48–69), not 80%. Whether the headline changes is
> pending supervisor discussion; do not quote 80% without stating the protocol.
- **Overall (best checkpoint = 15k): task completion 80% (95% CI 70–87),
  grasp 92.5%, over 320 trials.**
- **Per item (n=20, placement):** cola **95%**, bottle **90%**, bread **75%**,
  milk **60%**. Grasp: cola/bottle/bread 100%, milk 70%.
- **Checkpoint selection (n=80 each, placement):** 5k 68.8%, 10k 68.8%,
  **15k 80.0%**, 20k 75.0% (20k dip = mild over-training).
- **0% → 80% story:** v1 placed 0% but near-missed (dropped ~10 cm short + didn't
  stop cleanly). Fixes: clean episode boundary, basket camera, wider tote.
- **Language grounding:** same scene, change only the instruction → arm reaches the
  named item. milk (y −0.26)→ reached −0.24; bread (0.00)→ −0.01; bottle (0.34)→ 0.29.
- **Interpretability:** SmolVLA plans 50 steps ahead; at frame 0 (arm at home) its
  plan already closes the gripper at +16 → it "decides to grasp" before reaching.
- **Orchestrator:** menu → collect items in order, basket accumulates, retry on
  miss. Known limitation: multi-item lists degrade — but **not** because of basket
  occupancy (EXP-5 refuted that: 0/1/2 items already in basket → 61.8/58.3/66.7%).
  Complete-list success is 45% (2-item) and 30% (3-item). Mechanism unresolved.

## 10. Known limitations
- **Evaluated on a fixed object pose.** `env.reset()` applies no jitter; under ±2 cm
  jitter placement falls to 58.8%. See `REVIEW_REPORT.md` §1.
- **No visual grounding.** The policy maps word → memorised trajectory: displace an
  item and it still reaches the trained position (216/216 trials). `REVIEW_REPORT.md` §2.
- Milk weakest (60% place / 70% grasp) — partly a **metric artefact**: `grasped` is
  `max_lift > 0.05 m` and milk's median lift is 0.054 m, so its grasp rate is a lower
  bound. `REVIEW_REPORT.md` §7.
- Multi-item lists < single picks — **not** explained by basket occupancy (EXP-5).
- Cereal excluded (grip slip).
- Simulation only (no sim-to-real claimed).
- Flow-matching stochastic → run-to-run variance (the eval sets no torch seed).
- **No training loss log exists** for the v2 run (`log_freq` set, stdout not captured).
- **π0 was never trained** — crashed before step 1 on a PaliGemma API error.

## 11. The ideal (unconstrained) system
Not "a bigger model" — some limits are physical (a huge model can't run a fast/safe
control loop near people). Ideal = **dual-system**: large reasoner (System 2) for
list + crowd, fast reflex policy (System 1) for control. Plus: whole-body control
(base+arm together), **tactile + depth sensing** (fixes grasping), diverse
co-trained data (generalises), imitation→RL (beats the demo ceiling). Model class if
unconstrained: **π0.5 / GR00T-class**. Current system = the deployable subset.

## 12. Repository layout (`~/projects/supermarket_vla`)
```
envs/supermarket_env.py          scene, robot, cameras, constants
scripts/find_grasp.py            mink IK (solve_grasp, solve_pose)
scripts/scripted_expert.py       pick-place trajectory + recorder hook
scripts/collect_data.py          demonstration recorder
scripts/convert_to_lerobot.py    pkl -> LeRobotDataset
scripts/train_smolvla.py         (vision-unfreeze monkeypatch; else use lerobot_train)
scripts/rollout.py               closed-loop eval (run_episode, --view, --think, --think plan readout)
scripts/eval_checkpoints.py      multi-checkpoint eval + Wilson CIs
scripts/shopping_list.py         menu orchestrator (--view, --think, --retries)
scripts/make_plots.py            per-item bar chart + checkpoint curve
scripts/make_arch_diagram.py     SmolVLA architecture diagram
scripts/make_journey_diagram.py  RoboCasa->UR10e->Supermarket progression
scripts/make_camera_views.py     the 3 camera views figure
scripts/make_thesis_figures.py   env layout, near-miss schematic, pipeline flow
scripts/build_ppt.py             builds the 25-slide presentation
scripts/patch_ppt_slides.py      patches ppt slides 10/22/23
scripts/patch_add_slides.py      adds ideal-system + thank-you slides
shared/instruction_templates.py  ITEM_NAMES, TEMPLATES, instruction_for()
thesis/                          Cranfield LaTeX thesis (main.tex + chapters/ + references.bib + figures/)
outputs/                         plots/, poster/, stage1/, train/, ppt_assets/
data/supermarket_manip_lerobot/  the LeRobot dataset
.venv -> ../smolvla_ur10e/.venv  shared virtualenv
Docs: PLAN.md, DECISIONS.md, RUNNING.md, REPORT_NOTES.md, README.md,
      PRESENTATION.md, PRESENTATION_SCRIPT.md, PROJECT_KNOWLEDGE.md (this file)
```

## 13. Environment / running notes
- **Interpreter:** always `.venv/bin/python` (symlink to `../smolvla_ur10e/.venv`,
  i.e. system python3.12). NOT conda `(base)`.
- **Rendering:** `MUJOCO_GL=egl` for headless (data, eval, figures). For the live
  viewer (`--view`) use a real display and do NOT set `MUJOCO_GL=egl`.
- **Offline-first:** the user historically worked offline; flag any step needing
  internet (pip installs, model downloads).
- **Never modify the reference project `~/projects/smolvla_ur10e`** (read-only).

## 14. Related project — `~/thesis_scripts` (the RoboCasa/pilot experiments)
- `train_smolvla.py`, `train_act.py`, `train_pi0.py`, `rollout_smolvla.py`,
  `batch_eval.py`, `eval_act.py`.
- Checkpoints: `smolvla_robocasa`, `act_robocasa`, `pi0_robocasa`,
  `smolvla_fixedbase`, `act_fixedbase`.
- `groot_eval_videos*` — GR00T evaluation videos on RoboCasa tasks.
- `eval_results.csv` (SmolVLA) & `eval_act_results.csv` (ACT): 0/20 on RoboCasa.

## 15. Supervisors / admin
- Author: **Biswajeet Panda**. Supervisors: **Dr Gilbert Tang, Prof Phil Webb**.
- Cranfield University, **MSc Robotics** (Oct 2025 – Sept 2026), Faculty of
  Engineering and Applied Sciences.
