# THESIS_REVIEW_TASKS.md

Task queue for Claude Code. Derived from an examiner-style review of the MSc thesis
*Mobile Manipulator Grasping of Everyday Objects Using Vision–Language–Action Models*.

**Repo:** `~/projects/supermarket_vla`
**Companion docs:** `thesis-review-action-plan.md` (full reasoning), `thesis-review-addendum.md` (updates from `PROJECT_KNOWLEDGE.md`)

---

## 0. CONTEXT FOR THE AGENT

Read `PROJECT_KNOWLEDGE.md` in the repo root before starting. Summary of what matters here:

- MuJoCo supermarket sim; UR10e + Robotiq 2F-85 on a parked Omron LD-60 base.
- SmolVLA fine-tuned with a **frozen** VLM backbone; only the action expert trains.
- Headline results: 80% task completion (95% CI 70–87), 92.5% grasp, n=80 for the 15k checkpoint.
- The LaTeX thesis lives in `thesis/` (`main.tex`, `chapters/`, `references.bib`, `figures/`).

### HARD RULES — do not violate

```
1. ALWAYS use .venv/bin/python. NEVER conda, NEVER bare `python`/`python3`.
2. NEVER modify ~/projects/smolvla_ur10e — it is a read-only reference project.
3. Headless work (data, eval, figures): export MUJOCO_GL=egl
   Live viewer (--view): do NOT set MUJOCO_GL. Requires a real display.
4. Assume OFFLINE. If a step needs internet (pip install, HF download), STOP and ask.
5. NEVER edit files under outputs/train/**/checkpoints/ — those are trained artefacts.
6. Before running any script, check its real interface:  <script> --help
   Do NOT invent flags. If a flag this file assumes does not exist, STOP and report.
7. Do not delete or overwrite existing eval CSVs. Write new results to NEW filenames.
8. Git: create a branch before any edits.  git checkout -b thesis-review-fixes
   Commit after each completed task with the task ID in the message.
```

### Execution order

```
PHASE 0  INV-1..INV-6   read-only investigations   ← DO THESE FIRST, then STOP
PHASE 1  TXT-1..TXT-8   thesis text fixes          ← no experiments needed
PHASE 2  EXP-1..EXP-6   simulation runs            ← gated on PHASE 0 findings
PHASE 3  FIG-1..FIG-4   figures, tables, appendix
```

**Phase 0 is a hard gate.** Several later tasks change depending on what Phase 0 finds.
Run all of Phase 0, write `REVIEW_FINDINGS.md`, then stop and report before Phase 1.

---

## PHASE 0 — READ-ONLY INVESTIGATIONS

No file modifications in this phase. Record every result in `REVIEW_FINDINGS.md` at the
repo root, using the template at the end of this section.

---

### INV-1 — Does a training loss log exist?

**Why:** thesis §4.6 claims *"selecting by training loss would have deployed the worse model."*
The training command used `--eval_freq=0 --wandb.enable=false`, so there may be no loss
curve at all. If none exists, the claim must be **deleted**, not softened.

```bash
cd ~/projects/supermarket_vla
ls -la outputs/train/smolvla_supermarket_v2/
find outputs/train/smolvla_supermarket_v2/ -maxdepth 2 -type f \
     \( -name "*.json" -o -name "*.csv" -o -name "*.log" -o -name "*.txt" -o -name "*.yaml" \) | head -50
```

If anything looks like a metrics log, inspect it:

```bash
find outputs/train/smolvla_supermarket_v2/ -name "*.json" | head -5 | xargs -I{} sh -c 'echo "=== {} ==="; head -c 2000 {}; echo'
```

**Record:** does a per-step loss series exist? At what interval? Path?

**Acceptance:** `REVIEW_FINDINGS.md` states either the log path + format, or "no loss log found".

---

### INV-2 — Are checkpoint eval trials seed-matched?

**Why:** if all four checkpoints were evaluated on the *same* seeds, the trials are paired and
**McNemar's test** applies — far more powerful than the unpaired two-proportion test, and it
may resolve the 15k-vs-20k comparison that currently is not significant (p ≈ 0.45).

```bash
cd ~/projects/supermarket_vla
sed -n '1,200p' scripts/eval_checkpoints.py
grep -n "seed\|Seed\|SEED\|random\|rng" scripts/eval_checkpoints.py
grep -n "seed" scripts/rollout.py | head -30
```

**Determine:** does the script build one seed list and reuse it across checkpoints, or does
it generate fresh seeds per checkpoint?

**Record:** PAIRED or UNPAIRED, plus the line numbers that prove it.

---

### INV-3 — Does per-item × per-checkpoint data already exist?

**Why:** thesis §4.3 claims *"the milk carton placed more reliably at earlier checkpoints"*
and §4.6 claims the 15k→20k drop *"is consistent across items"* — but only aggregates are
published. The 320 rollouts already contain the 4×4 breakdown.

```bash
cd ~/projects/supermarket_vla
find outputs/ -name "*.csv" -newermt "2026-01-01" | head -30
grep -rn "per_item\|per-item\|item.*checkpoint\|to_csv" scripts/eval_checkpoints.py
```

For any candidate CSV:

```bash
head -5 <path>; echo "---"; wc -l <path>
```

**Record:** path to a CSV with per-trial or per-item-per-checkpoint rows, or "not found —
must re-derive".

---

### INV-4 — v1 (narrow) basket dimensions

**Why:** this is the **single most important missing constant in the project**. The 0%→80%
improvement bundles two behaviour fixes with a *widened basket* — a change to the success
criterion, not the policy. Without the v1 footprint, that gain cannot be decomposed and
EXP-4 cannot run.

Current v2 value is `BASKET_HALF=(0.15, 0.13)` → 0.30 × 0.26 m.

```bash
cd ~/projects/supermarket_vla
git log --oneline -- envs/supermarket_env.py | head -30
git log -p -S "BASKET_HALF" -- envs/supermarket_env.py | grep -n "BASKET_HALF\|^commit\|^Date" | head -40
grep -rn "basket" DECISIONS.md PLAN.md REPORT_NOTES.md 2>/dev/null | head -30
grep -rn "BASKET_HALF\|basket.*wide\|widen" *.md 2>/dev/null | head -20
```

**Record:** the v1 `BASKET_HALF` value and the commit that changed it, or "unrecoverable".

**If unrecoverable:** flag it — the thesis must then state in §5.6 that the tolerance
contribution cannot be quantified retrospectively. That is an acceptable admission; silently
omitting it is not.

---

### INV-5 — π0 training configuration

**Why:** thesis §2.2 claims SmolVLA *"is the largest capable VLA that fine-tunes within a 12 GB
budget."* But `~/thesis_scripts/` contains `train_pi0.py` and a `pi0_robocasa` checkpoint.
π0 is ~3B params. If it trained on the 12 GB A2000, the claim is contradicted by your own
experiments.

```bash
ls -la ~/thesis_scripts/
sed -n '1,150p' ~/thesis_scripts/train_pi0.py
grep -n "freeze\|lora\|LoRA\|batch\|device\|cuda\|memory\|gradient_checkpoint" ~/thesis_scripts/train_pi0.py
ls -la ~/thesis_scripts/pi0_robocasa/ 2>/dev/null | head -20
ls ~/thesis_scripts/*.csv
```

**Determine:**
- (a) trained on different hardware, (b) trained with freezing/LoRA/fewer cameras, or (c) it fit on 12 GB
- whether a π0 RoboCasa **eval result** exists (the abstract asserts π0 failed but reports no number)

**Record:** the resolution, plus π0's RoboCasa score with n if found.

---

### INV-6 — Extract missing constants for Chapter 3 and Appendix B

**Why:** the thesis is missing the learning rate, optimiser, template count, and the numeric
success threshold — all of which block the "reproducible pipeline" contribution claim.

```bash
cd ~/projects/supermarket_vla

# exact instruction template count (thesis is self-contradictory: ~20 templates vs ~20 total instructions)
.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from shared.instruction_templates import TEMPLATES, ITEM_NAMES
print('TEMPLATES:', len(TEMPLATES))
print('ITEMS:', len(ITEM_NAMES), ITEM_NAMES)
print('TOTAL INSTRUCTIONS:', len(TEMPLATES)*len(ITEM_NAMES))
for t in TEMPLATES: print(' ', t)
"

# numeric success criterion — thesis only says 'inside the footprint and low'
grep -n "success\|BASKET_HALF\|placed\|in_basket\|footprint\|0\.4\|z <\|height" scripts/rollout.py | head -40

# LeRobot resolved defaults (LR, optimiser, schedule) — none were passed on the command line
.venv/bin/python -c "
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
c = SmolVLAConfig()
for k,v in sorted(vars(c).items()):
    if any(s in k.lower() for s in ['lr','optim','sched','decay','warmup','grad','beta','eps']):
        print(k,'=',v)
" 2>/dev/null || echo "IMPORT PATH DIFFERS — locate SmolVLAConfig manually"

.venv/bin/python -c "import lerobot, torch; print('lerobot', lerobot.__version__); print('torch', torch.__version__)"
```

**Record:** template count, exact success-criterion thresholds, LR/optimiser/schedule values,
library versions.

---

### Phase 0 output — write this file

Create `REVIEW_FINDINGS.md` at the repo root:

```markdown
# Phase 0 findings — <date>

## INV-1 training loss log
FOUND / NOT FOUND — path: …  format: …  interval: …

## INV-2 seed matching
PAIRED / UNPAIRED — evidence: eval_checkpoints.py lines …

## INV-3 per-item × checkpoint data
FOUND / NOT FOUND — path: …  columns: …

## INV-4 v1 basket dimensions
BASKET_HALF v1 = (…, …)   commit: …
or: UNRECOVERABLE

## INV-5 π0 configuration
Resolution: (a) different hardware / (b) freezing-LoRA / (c) fit in 12 GB
Details: …
π0 RoboCasa result: … (n = …)  or NOT FOUND

## INV-6 constants
TEMPLATES = …   ITEMS = …   total instructions = …
Success criterion: |x − 0.24| < …  ∧  |y| < …  ∧  z < …
LR = …  optimiser = …  schedule = …
lerobot = …  torch = …  mujoco = 3.9

## Consequences for later tasks
- EXP-4 is: RUNNABLE / BLOCKED (needs INV-4)
- EXP-2 is: RUNNABLE / NOT APPLICABLE (needs INV-2 = PAIRED)
- TXT-6 (§4.6 loss claim) is: REWRITE / DELETE (needs INV-1)
- TXT-7 (§2.2 12 GB claim) resolution: … (needs INV-5)
```

**STOP HERE. Report findings before starting Phase 1.**

---

## PHASE 1 — THESIS TEXT FIXES

All edits are in `thesis/`. Locate each target with `grep -rn` before editing — chapter
filenames are not assumed by this document.

```bash
cd ~/projects/supermarket_vla/thesis
ls chapters/
```

After each task: `git commit -m "TXT-n: <description>"`

---

### TXT-1 — Correct 240 → 208 training demonstrations  🔴 BLOCKER

`PROJECT_KNOWLEDGE.md` §6: *"v2 dataset: 208 train / 32 val episodes."*
240 is what was **collected**; 208 is what the model **trained on**. The thesis presents 240
as the training set throughout.

```bash
grep -rn "240" chapters/ main.tex
```

Replace each occurrence with the split, e.g.:

> "240 demonstrations were collected (60 per item across four items); 208 were used for
> training and 32 (8 per item) held out for validation."

**Locations to check:** abstract, §3.3, §3.4, §5.1, §6.1, §B.3.

**Note in the commit:** this slightly *strengthens* the argument — the result was obtained on
less data than claimed. Do not bury it.

**Acceptance:** no bare "240 demonstrations" remains where the training set is meant.

---

### TXT-2 — Fix the RoboCasa zero-shot / fine-tuned contradiction  🔴 BLOCKER

Three mutually incompatible statements exist. Evidence (`train_*.py` scripts and
`smolvla_robocasa` / `act_robocasa` / `pi0_robocasa` checkpoints) shows the models were
**fine-tuned**, not evaluated zero-shot. §5.3 is correct; the abstract is wrong.

Independent confirmation: thesis §2.3 states ACT *"cannot accept a free-form text
instruction"* — so ACT cannot be evaluated zero-shot at all.

```bash
grep -rn "zero-shot\|zero shot\|out of the box" chapters/ main.tex
```

- Abstract: `"failed zero-shot"` → `"failed even after fine-tuning on RoboCasa demonstrations"`
- §4.1: remove `"out of the box"`; state that models were fine-tuned on RoboCasa demos
- §4.1: add the **number of RoboCasa demonstrations** used (find in `~/thesis_scripts/`)
- Verify §5.3 now agrees with both

**Note:** GR00T's ~54% may genuinely have been zero-shot. If so, that asymmetry **must be
stated** — otherwise Fig. 4.1 compares fine-tuned small models against a zero-shot large
model without saying so.

---

### TXT-3 — Fix the milk-carton geometry claims  🔴 BLOCKER

Positions (KB §4): milk **−0.26**, bottle **+0.34**. KB §5: *"milk moved inward from reach
edge."* So milk is **not** at the far edge — the bottle is 8 cm further out and scores
100% grasp / 90% placement. Chapter 3 is right; Chapter 4 and 5 are wrong.

```bash
grep -rn "far lateral edge\|most extended\|near-full extension\|edge-of-reach" chapters/
```

- §4.3: delete *"sits at the far lateral edge of the shelf, where the arm is most extended"*
- §5.2 (weak item): revise the explanation
- §5.4: delete *"only reachable at near-full extension"* as the motivation for whole-body
  control — substitute reaching around obstacles / larger effective workspace, both already
  in the text

**Replacement explanation** (the expert hits 98% on milk with no special tuning, so this is a
*learned-policy* limitation, not kinematics): mesh geometry is less forgiving to an imprecise
learned grasp than a cylinder, and/or the wrist-camera viewpoint is oblique at that approach.
Confirm against EXP-6 before finalising wording.

---

### TXT-4 — Fix the +0.34 misattribution

§4.5 says: *"The small systematic undershoot on the far item (reaching +0.29 for a target at
+0.34) is consistent with the milk carton's edge-of-reach difficulty."*

**y = +0.34 is the water bottle.** Milk is at −0.26.

```bash
grep -rn "0.29\|+0.34\|0\\.34" chapters/
```

Also note the bottle scores 100% grasp, so a 5 cm offset there needs a benign explanation —
most likely the gripper-centre offset at the grasp pose. State that instead.

---

### TXT-5 — Fix the abstract's trial-count conflation

Current: *"Evaluated over 320 closed-loop trials … the policy achieves an 80% task-completion
rate (95% CI 70–87)."* The 80% and its CI are **n = 80** (one checkpoint); 320 is
4 checkpoints × 4 items × 20.

Replace with:

> "…achieves an 80% task-completion rate (95% Wilson CI 70–87, n = 80) for the selected
> checkpoint, drawn from 320 closed-loop rollouts across four checkpoints."

Also add to the abstract: **four items**, **208 training demonstrations**, and the words
**"in simulation"** where it currently says the model is "deployable".

---

### TXT-6 — Rewrite §4.6 (checkpoint selection)  🔴 BLOCKER

Two problems:

**(a) The differences are not significant.** Two-proportion tests on Table 4.3:

| Comparison | Counts | z | p |
|---|---|---|---|
| 15k vs 20k | 64/80 vs 60/80 | 0.76 | 0.45 |
| 15k vs 5k | 64/80 vs 55/80 | 1.63 | 0.10 |

§4.6 currently calls the 15k→20k drop *"evidence of mild over-training"* and *"the classic
signature of mild over-fitting"*. It also dismisses the 5k/10k comparison for overlapping
intervals, then accepts a comparison whose intervals overlap **more** — an inconsistent
standard an examiner will notice.

**(b) The loss claim may be unevidenced** — depends on INV-1.

**Branch on Phase 0 results:**

- **INV-2 = PAIRED** → run EXP-2 (McNemar's) first; it may rescue the claim. Rewrite from the result.
- **INV-2 = UNPAIRED** → rewrite to state the checkpoints are statistically indistinguishable:

> "All four checkpoints perform comparably; observed differences are within sampling noise at
> n = 80. The methodological point stands independently of the ranking: because closed-loop
> success and training loss are not monotonically related, selection must be made on the
> former. Here the choice was consequential only in that the final checkpoint was not the
> empirical best."

- **INV-1 = NOT FOUND** → **delete** *"Selecting by training loss, which typically continues
  to improve, would have deployed the worse model."* Do not soften it — with no loss log it
  is an unevidenced assertion about your own run.

---

### TXT-7 — Resolve the 12 GB / "largest VLA" claim

§2.2: *"the largest capable VLA that fine-tunes within a 12 GB budget."*
Appendix B reports peak usage **≈ 3 GB** — a 4× margin. And INV-5 may show π0 (~3B) trained
too.

Rewrite per INV-5 outcome:
- (a) π0 on other hardware → keep the claim, add the caveat
- (b) π0 with freezing/LoRA → *"largest that fine-tunes in this configuration"*, describe the π0 setup
- (c) π0 fit in 12 GB → **delete the claim**; reframe as: the ceiling constrained model
  selection and ruled out unfreezing the backbone with three camera streams

Add the memory table to Appendix B either way:

| Configuration | Peak GPU memory |
|---|---|
| SmolVLA, backbone frozen | ~3 GB |
| SmolVLA, vision unfrozen, 3 cameras | OOM |
| π0 | from INV-5 |

---

### TXT-8 — Honest-framing pass

Small wording changes, each a viva risk if left:

```bash
grep -rn "deployable\|mobile manipulator\|useful success rate" chapters/ main.tex
```

- **Abstract:** "deployable" → "deployable **in simulation**" (§5.2.0.0.4 already says no
  sim-to-real is claimed; the abstract contradicts its tone)
- **Abstract/title:** note the **base is parked** — §1.2 is honest, the title is not
- **RQ1 §1.4:** define "useful success rate" with an a-priori threshold. Under most realistic
  shelf-picking thresholds 80% overall and 60% on milk would **not** qualify. Saying so is
  maturity, not weakness — frame as "useful as a feasibility demonstration, not for deployment"
- **§4.2 / §5.5:** add one sentence that a simulated 80% is not directly comparable to
  Kiviranta's 80% on real SO-101 hardware
- **Fig. 4.1 caption:** add *"Note these rates are not comparable: task difficulty decreases
  left to right."* The three numbers span three different tasks, robots and embodiments

---

## PHASE 2 — EXPERIMENTS

Always check `--help` first. Write results to **new** filenames under `outputs/review/`.

```bash
cd ~/projects/supermarket_vla
export MUJOCO_GL=egl
mkdir -p outputs/review
```

**Universal logging requirement** — every run below must log, per trial:
`seed, item, full_instruction, checkpoint, grasp_bool, placement_bool,
final_item_xyz, reached_lateral_y, joint_config_at_grasp`

`final_item_xyz` is what makes EXP-4 a re-scoring rather than a re-run. Log it from now on
even where this file does not explicitly require it.

---

### EXP-3 — Position-swap grounding test  ⭐ HIGHEST VALUE

**Why:** §4.5's test rules out "policy ignores language" but **not** the hypothesis that the
policy learned a lookup table:

```
"milk"   → joint config reaching y ≈ −0.26 → close → carry to basket
"bread"  → joint config reaching y ≈  0.00 → close → carry to basket
"bottle" → joint config reaching y ≈ +0.34 → close → carry to basket
```

That policy does **zero visual localisation** and reproduces Table 4.2 exactly. It is the
*likely* hypothesis: items sit at fixed positions (±2 cm jitter), actions are absolute joint
targets, the backbone is frozen, and 208 demos is small.

This matters because §3.1.1 claims *"the policy must localise the named item from pixels
alone"* — which the current evidence does not support.

**EXP-3a — swap two items:**

```bash
grep -n "y\s*=\|ITEM_POS\|POSITIONS\|-0.26\|0.34" envs/supermarket_env.py | head -30
```

Add an override (a CLI flag or env var — **do not hard-edit the defaults**) that swaps
milk (−0.26) ↔ bread (0.00). Then:

```bash
.venv/bin/python scripts/rollout.py --help   # confirm flags first
# n >= 10 per instruction, swapped layout, instructions UNCHANGED
```

**Interpretation:**
- reaches the item's **new** position → genuine visual grounding → §4.5's strong claim is vindicated
- reaches the item's **usual** position → word→pose lookup → **also a strong result**:
  *"a compact VLA fine-tuned on a small, positionally-fixed dataset learns language-to-trajectory
  mapping rather than language-grounded visual selection"*

**Both outcomes improve the thesis. Do not treat the second as a failure.**

**EXP-3b — unseen positions:** place each item at 2–3 lateral positions never used in
training, n ≥ 10 each. Graceful degradation → visual localisation. Sharp collapse → lookup.
This also fills the untested generalisation claim in §5.3.

**Also fix Table 4.2 regardless:** add **n**, add spread (min–max or SD) per instruction, and
**add the cola can** — cola (−0.13) vs bread (0.00) is the tightest and most informative
discrimination, and its absence looks like an omission.

**Output:** `outputs/review/exp3_grounding.csv` + a summary in `REVIEW_FINDINGS.md`

---

### EXP-1 — Fresh-seed re-evaluation of the 15k checkpoint

**Why:** all four checkpoints were evaluated on the **same** 80 trials, the maximum was
selected, and that maximum was reported with a naive CI. That is a winner's curse — the
expected value of a maximum over four noisy estimates exceeds the true value of the selected
one, and the interval understates the uncertainty because the same data chose the number.

```bash
.venv/bin/python scripts/rollout.py --help
# 15k checkpoint only, 80 trials (20 per item), seeds DISJOINT from the original selection set
```

Confirm from INV-2/INV-3 which seeds the original evaluation used, and pick a disjoint range.

**Output:** `outputs/review/exp1_freshseed_15k.csv`

> ⚠️ If this returns lower than 80%, **report the lower figure**. A defensible 76% is worth
> more than an indefensible 80%. Update the abstract, §4.2, §6.1 — and `PROJECT_KNOWLEDGE.md`
> §9 and the presentation scripts, or the superseded number will resurface in the viva.

---

### EXP-2 — McNemar's test across checkpoints  *(only if INV-2 = PAIRED)*

```bash
.venv/bin/python - <<'PY'
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
df = pd.read_csv("<per-trial CSV from INV-3>")
a = df[df.checkpoint==15000].sort_values("seed").placement.values
b = df[df.checkpoint==20000].sort_values("seed").placement.values
tbl = [[((a==1)&(b==1)).sum(), ((a==1)&(b==0)).sum()],
       [((a==0)&(b==1)).sum(), ((a==0)&(b==0)).sum()]]
print(tbl); print(mcnemar(tbl, exact=True))
PY
```

Repeat for 15k vs 5k and 15k vs 10k. Feed the result into TXT-6.

---

### EXP-4 — Narrow-basket re-scoring  *(blocked until INV-4 resolves)*

**Why:** the 0%→80% story bundles two behaviour fixes (episode boundaries, basket camera)
with a **wider basket** — a change to the success criterion, not the policy. An item that
previously failed now passes with no change in robot behaviour. The thesis never separates
these, and §6.1 omits the basket change entirely.

If `final_item_xyz` was logged, this is **pure re-scoring, zero simulation**:

```bash
.venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_csv("<15k eval CSV with final item positions>")
V1_HALF = (?, ?)   # from INV-4
BX, BY = 0.24, 0.00
inside = (abs(df.final_x-BX) < V1_HALF[0]) & (abs(df.final_y-BY) < V1_HALF[1]) & (df.final_z < Z_THRESH)
print("v2 (published):", df.placement.mean())
print("v1 (narrow)   :", inside.mean())
PY
```

The gap is the portion of the 80-point gain attributable to **tolerance** rather than
**behaviour**. Report it in §4.4 and §5.6.

---

### EXP-5 — Multi-item orchestrator quantification

**Why:** §1.5 claims the orchestrator as a contribution *"together with an honest
characterisation of where it degrades"*, but §4.8 contains **no numbers at all** — no success
rate, no n, no list lengths. It is the only results section with no data.

```bash
.venv/bin/python scripts/shopping_list.py --help
# 2-item lists n>=20, 3-item lists n>=20, randomised item order
```

Log success **by position in list** and basket contents at each pick. The position-1 vs
position-2+ gradient **is** the result — it quantifies the distribution shift §4.8 currently
only asserts.

**Output:** `outputs/review/exp5_multiitem.csv`

---

### EXP-6 — Milk-carton diagnostic

Now narrower than originally scoped: the geometry refutes the reach hypothesis (bottle is
further out and scores better), and the scripted expert hits 98% on milk with no special
tuning. So the difficulty is in the **learned policy**, not the kinematics.

Compare, for milk vs bottle:
- scripted expert grasp pose vs learned policy grasp pose — how far off is the policy?
- wrist-camera view at the approach (render and inspect)
- milk placement rate **by checkpoint** (from INV-3 data)

Feeds the replacement explanation in TXT-3.

---

### EXP-7 — ACT baseline on the supermarket task  ⭐ THE MISSING CONTROL

**Why:** RQ2 asks whether gains come more from model capacity or from data/metrics. On the
supermarket task you **never vary capacity** — SmolVLA is the only policy ever run on it. All
capacity evidence comes from the pilot, a different task and robot. Yet the strong comparative
claim appears in eight places including the abstract.

Infrastructure exists: `~/thesis_scripts/train_act.py`, `eval_act.py`, and ACT has been
trained twice (`act_robocasa`, `act_fixedbase`).

```bash
sed -n '1,120p' ~/thesis_scripts/train_act.py
# train ACT on the SAME 208-episode LeRobot dataset, same 3 cameras, same action space
# ACT is language-blind: train per-item, or condition on item as a non-language signal
# STATE CLEARLY which approach was used
# evaluate with the identical protocol: 20 trials x 4 items, Wilson CIs, same success criterion
# log peak GPU memory -> feeds TXT-7
```

**Bonus:** if ACT matches SmolVLA per-item, that is **independent evidence the task does not
require language grounding** — which corroborates or contradicts EXP-3. Run both.

**Output:** `outputs/review/exp7_act_baseline.csv`

**If not run:** soften RQ2 everywhere to *"large gains were available from data and
observability corrections **at fixed model capacity**"*, and add to §5.2: *"No same-task
capacity comparison was performed; the model-versus-data conclusion rests on the pilot study,
which used a different task and embodiment."*

---

## PHASE 3 — FIGURES, TABLES, APPENDIX

### FIG-1 — Per-item × checkpoint table (from INV-3)

4 checkpoints × 4 items, with Wilson CIs. Add to §4.6. This evidences two claims currently
made without data (§4.3 milk-by-checkpoint, §4.6 "consistent across items").

### FIG-2 — Loss curve *(only if INV-1 found a log)*

Training loss vs steps, with 5k/10k/15k/20k marked. If the 32-episode validation split can be
scored offline, add validation loss too — *"validation loss also kept falling"* is a far
stronger version of §4.6's argument than training loss alone.

```bash
grep -n "def main\|argparse\|plt\." scripts/make_plots.py | head -20
```

### FIG-3 — Conditional placement column in Table 4.1

Placement given a successful grasp — this **changes the story usefully**:

| Item | Grasp | Placed | Placed \| grasped |
|---|---|---|---|
| Cola | 20/20 | 19/20 | 95% |
| Bottle | 20/20 | 18/20 | 90% |
| Bread | 20/20 | 15/20 | **75%** |
| Milk | 14/20 | 12/20 | **86%** |
| Overall | 74/80 | 64/80 | 86.5% |

Milk's *placement* is fine once grasped — **better than bread's**. Milk's problem is purely
the grasp; **bread is the real placement problem**. This sharpens the grasp/place argument in
§4.3 and §5.1, and gives cleaner future-work targets: contact sensing for milk (a grasp
problem, as §5.4 argues) and drop precision for bread.

### FIG-4 — Figure repairs

```bash
ls thesis/figures/
grep -n "def " scripts/make_thesis_figures.py | head -20
```

- **Fig. 3.3:** label collisions — "cola (y=−0.13)" overlaps "basket"; "cereal excluded"
  overlaps the shelf. This is the figure that reveals the milk/bottle geometry, so it must be legible.
- **Fig. 3.6:** blank black panel bottom-right — fill or reflow to 4×2
- **Fig. 3.5:** caption says "wrist and scene" — the **basket** camera, central to the §4.4
  fix, is missing. Add it, or note the sample predates it
- **Fig. 3.2:** shows the cereal box despite its exclusion — add a caption note
- **Fig. 4.1:** add the comparability caveat (see TXT-8)

### FIG-5 — Appendix B from INV-6

Fill Table B.1: **learning rate, optimiser, LR schedule, weight decay, gradient clipping,
image augmentation** — none are currently stated, and the LR alone makes the run
unreproducible. Add lerobot 0.5.2 / torch 2.11+cu128 / MuJoCo 3.9. Add basket geometry
(0.30 × 0.26 m, centre (0.24, 0.00), height 0.47 m), the numeric success threshold, the exact
template list, and `ACTION_SUBSTEPS = 25` (currently "roughly twenty-five"). Replace "default"
entries in Table 3.2 with real values (`GRIPPER_LENGTH=0.14`, `PRE_GRASP_UP=0.15`,
bottle `grasp_gl=0.18`, cereal `yaw=90`).

Add a **Code and data availability** statement — §1.5 claims a "complete, reproducible
pipeline" but there is no public repo, licence or dataset location.

### FIG-6 — Citation gaps

```bash
grep -c "cite" thesis/chapters/*.tex
```

- **§2.5 (robotic grasping) — entire section uncited.** Force closure, analytic vs
  learning-based grasp synthesis. A lit-review subsection with zero references stands out.
- §2.4: DAgger (Ross et al. 2011), behaviour cloning (Pomerleau 1988), covariate shift
- §2.7: domain randomisation (Tobin et al. 2017), LoRA (Hu et al. 2021)
- §2.9: flow matching (Lipman et al. 2023)
- Ch. 3 tooling: `mink`, robosuite (Zhu et al.), MuJoCo Menagerie, SmolVLM-2 — none cited
- Ref [10]: cite the **SmolVLA paper (arXiv 2506.01844)**, not just the HF model page
- Ref [8]: no arXiv ID or venue — an examiner may be unable to locate it

### FIG-7 — Repetition cull

*"Data and metrics mattered more than model choice"* appears in **§1.1, §1.5, §2.8, §4.1,
§4.9, §5.1, §5.5, §6.1 — eight times.* Repetition of a *conclusion* reads as insistence rather
than evidence, especially where RQ2 lacks a same-task test.

Also: no-slip solver iterations (§3.1, §3.1.1, §3.2.2); instruction templates (§3.3, §3.3.1,
near-verbatim); dual-system argument (§5.2.0.0.7, §5.4, §6.2.1); §6.2.1 largely restates §5.4.

State each once, cross-reference thereafter. Cuts ~10% of length.

### FIG-8 — Cosmetic

- Fourth-level headings render as `2.1.0.0.1 Action chunking.` and propagate into the ToC —
  use unnumbered `\paragraph` or restructure §5.2's eight instances
- Naming: "Pi0"/"π0"; "GR00T N1.6"/"N1.x"/"N1"; "basket"/"tote"
- Abbreviations: add **EGL**, remove **RGB-D** (unused), reconcile **VRAM** vs "GPU memory"
- Roadmap paragraphs for Chapters 2, 4, 5 (Chapter 3 does this well — mirror it)
- **§3.1.1: forward-reference the basket camera** — *"(added as a result of the failure
  analysis in §4.4)"*. Without it, Chapter 3 presents three cameras as the original design
  while §4.4 presents the third as a discovery, which looks like concealed iteration

---

## SEPARATE — NOT A CODE TASK

**Replace `thesis/` Appendix A.** The ethics letter is still the template:

```
Reference: CURES/67890/2026
Project ID: 12345
Title: Full Title of Individual Research Project or Thesis
```

This is the single most damaging item in the document and takes five minutes. **The agent
cannot do this — retrieve the real CURES letter from the Cranfield portal manually.**

---

## OPTIONAL — add to CLAUDE.md

```markdown
## Thesis review
An examiner-style review is in progress. Task queue: THESIS_REVIEW_TASKS.md
Phase 0 findings: REVIEW_FINDINGS.md
Work on branch thesis-review-fixes. Complete Phase 0 before Phase 1.
Headline numbers in PROJECT_KNOWLEDGE.md §9 may be superseded by EXP-1 —
check REVIEW_FINDINGS.md before quoting them.
```
