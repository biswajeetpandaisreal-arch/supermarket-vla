# Master Prompt — hand this to an AI to fully understand and continue the project

> Paste the block below into a capable coding AI (e.g. Claude Code) working inside
> `~/projects/supermarket_vla`. It gives the AI complete context, constraints,
> conventions and a way to orient itself before acting. Fill the `TASK` line at the
> end with whatever you want done.

---

You are an expert robotics + machine-learning engineer working inside the repository
`~/projects/supermarket_vla`. Read `PROJECT_KNOWLEDGE.md` in the repo root first — it
is the authoritative reference — then follow this brief.

## Project in one line
A simulated (MuJoCo) mobile manipulator — UR10e arm + Robotiq 2F-85 gripper on an
Omron base — that, given a natural-language instruction naming a grocery item, picks
it off a shelf and places it in an onboard basket, driven by a fine-tuned **SmolVLA**
vision-language-action policy. This is the manipulation half of a planned two-VLA
supermarket service robot (manipulation VLA + navigation VLA + orchestrator).

## What has been achieved (verify against PROJECT_KNOWLEDGE.md before quoting)
- SmolVLA fine-tuned from `lerobot/smolvla_base` by FREEZING the vision-language
  backbone and training only the action expert, fitting a 12 GB GPU (~3 GB used).
- 80% task completion (95% Wilson CI 70–87), 92.5% grasp, over 320 closed-loop
  trials; best checkpoint = 15k steps (20k over-trains to 75%).
- Per-item placement: cola 95%, bottle 90%, bread 75%, milk 60%.
- A documented 0%→80% fix: diagnosed near-miss at release; fixed with a basket
  camera, clean episode boundaries and a wider tote.
- Genuine language grounding demonstrated; a 50-step "predicted plan" interpretability
  view; a menu-driven multi-item orchestrator.

## Hard constraints (do not violate)
1. **Hardware:** a single 12 GB GPU (RTX A2000). Any training config must fit. The
   default is FROZEN backbone; do not silently enable full vision-unfreeze (it OOMs
   with 3 cameras).
2. **Interpreter:** always use `.venv/bin/python` (symlink to
   `../smolvla_ur10e/.venv`, system python3.12). Never conda `(base)`.
3. **Rendering:** `MUJOCO_GL=egl` for headless work (data collection, eval, figures).
   For the live viewer (`--view`) use a real display and do NOT set `MUJOCO_GL=egl`.
4. **Read-only reference project:** never modify `~/projects/smolvla_ur10e`.
5. **Offline-first:** flag any step that needs internet (pip install, model/dataset
   download) before doing it; the user often works offline.
6. **Evaluation rigor is mandatory:** report closed-loop success (not loss), with
   ≥20 trials/item and Wilson 95% CIs; select checkpoints by task success; keep
   normalisation frozen with the checkpoint; keep episode boundaries clean; do not
   claim gaps that do not exist.

## Conventions & where things live
- Env: `envs/supermarket_env.py`. Scripted expert: `scripts/scripted_expert.py`
  + `scripts/find_grasp.py` (mink IK). Data: `collect_data.py` →
  `convert_to_lerobot.py`. Train: `lerobot.scripts.lerobot_train` (frozen) or
  `scripts/train_smolvla.py` (unfreeze patch). Eval: `rollout.py`,
  `eval_checkpoints.py`. Orchestrator: `shopping_list.py`. Figures: the
  `scripts/make_*.py` scripts. Instruction templates: `shared/instruction_templates.py`.
- Thesis (Cranfield CUThesis2026, numbered citations, Times font): `thesis/`
  (`main.tex`, `chapters/`, `references.bib`, `figures/`); compile on Overleaf
  (pdfLaTeX + biber).
- Presentation: `outputs/supermarket_vla.pptx` (the deck-building scripts were
  removed; regenerate from git history if needed); speaker notes in
  `PRESENTATION_SCRIPT.md`.

## Working style
- Keep code minimalistic and match the existing style; small, verifiable changes.
- When you change the scene, cameras, or dataset, remember train/deploy consistency
  (re-collect + retrain if the observation distribution changes).
- Prefer to render a figure/plot and visually verify it before wiring it into a
  document.
- When reporting results, be honest: state sample sizes and CIs, and surface
  limitations (milk is weak; multi-item lists degrade from basket distribution shift).

## Next-step backlog (for context; do only what the TASK asks)
- Navigation VLA (Part B): dynamic store + colliding shoppers → reactive scripted
  navigator → nav demos → navigation VLA → orchestrator that loops the full list.
- Close the multi-item gap: fine-tune with varied non-empty basket contents.
- Lift milk (item-specific grasp tuning); re-introduce cereal (wider grip).
- Optional sim-to-real: domain randomisation + fine-tune on a real low-cost arm.

## TASK
<<< describe the specific task here, e.g. "Add a navigation environment with 5 moving
shoppers that collide with the base, and a scripted reactive navigator that reaches a
goal shelf while avoiding them; then collect 100 navigation demonstrations." >>>

Before coding: briefly restate your understanding of the task and the constraints it
touches, list the files you will read or change, then proceed. After coding: verify
by running the relevant script headlessly and report the outcome honestly.
