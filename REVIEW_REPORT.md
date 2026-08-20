# Thesis Review — Final Report

**Project:** `~/projects/supermarket_vla` · **Branch:** `thesis-review-fixes` · **Date:** 2026-08-20
**Task queue:** `THESIS_REVIEW_TASKS.md` · **Phase 0 detail:** `REVIEW_FINDINGS.md`
**Thesis files:** untouched (no `.tex` was edited — every text fix appears below as ready-to-paste wording)

---

## 0. What this is

An examiner-style review of the MSc thesis was converted into a task queue of 6 read-only
investigations, 8 text fixes, 7 experiments and 8 figure/appendix jobs. All of it has now
been run. This report gives the results, the corrections they force, and the exact wording
to apply.

**The short version.** The engineering is sound and most numbers reproduce exactly. But three
claims do not survive testing, and one missing control turns out to reframe the contribution:

1. The **80% headline is measured on a single fixed scene**. Under the ±2 cm object jitter used
   to *collect the training data*, it is **58.8%**.
2. The policy has **no visual grounding**. It maps each word to a memorised trajectory —
   216/216 displaced-item reaches went to the trained position, 0/216 to the actual one.
3. **π0 was never trained**, so the abstract's claim that it failed is unevidenced.
4. A **52M ACT beats the 450M SmolVLA** on the same item, same data, same protocol. What the
   VLA buys is item *selection*, not motor competence.

None of this makes the work bad. Findings 1, 2 and 4 are, stated honestly, a **better thesis**
than the current one — they replace a generic "small VLAs work" claim with a specific,
measured account of what a compact VLA does and does not learn from 208 demonstrations.

---

## 1. Headline finding — the 80% is a fixed-scene number

`envs/supermarket_env.py:432-446` (`reset()`) applies no randomisation. The ±2 cm jitter lives
only in `scripts/scripted_expert.py:74-81`, which is the **data-collection** path. So every
published evaluation trial ran from a bit-identical physical state; MuJoCo is deterministic.
The only variation across the 20 trials per item was the instruction paraphrase (6 of them)
and unseeded flow-matching noise.

I re-ran all 320 trials with the same ±2 cm jitter applied at evaluation time:

| checkpoint | placement, fixed scene (published) | placement, ±2 cm jitter | grasp, jitter |
|---|---|---|---|
| 5k | 68.8% | 53.8% | 65.0% |
| 10k | 68.8% | 53.8% | 63.8% |
| **15k (deployed)** | **80.0%** | **58.8%** (95% CI 48–69) | 80.0% |
| 20k | 75.0% | **62.5%** ← best | 76.2% |

Grasp at 15k falls 92.5% → 80.0%. **The checkpoint ranking also inverts**: 20k, not 15k, is
best under jitter.

### The control that makes this safe to claim

My change altered two things at once (jitter *and* torch seeding, which was previously absent).
So I re-ran 15k with jitter **off** on the identical new code path:

| 15k, fixed scene | placement | grasp |
|---|---|---|
| Published (`outputs/eval_v2_all.log`) | 80.0% | 92.5% |
| **My instrumentation, jitter = 0** | **81.2%** (CI 71–88) | **91.2%** |

The instrumentation reproduces the published result. The drop is caused by the jitter, not by
the change. Per item at 15k:

| item | fixed scene | ±2 cm jitter |
|---|---|---|
| cola_can | 18/20 | 12/20 |
| water_bottle | 18/20 | 13/20 |
| milk_carton | 12/20 | 10/20 |
| bread_loaf | 17/20 | 12/20 |

**Consequence.** 80% is not wrong, but it is a number for one object pose. Either report it
with that caveat stated plainly, or adopt 58.8% as the headline and present 80% as the
fixed-pose special case. The second is more defensible and is the one I would choose — a
reviewer who runs your own `collect_data.py` jitter will find 58.8% in ten minutes.

---

## 2. EXP-3 — the policy is language-conditioned but not visually grounded

§3.1.1 claims *"the policy must localise the named item from pixels alone."* It does not.

I moved items to positions the policy never trained on, leaving instructions unchanged
(`scripts/exp3_grounding.py`, 264 trials at the 15k checkpoint).

| block | item | item actually at | policy reached (mean) | SD | placed |
|---|---|---|---|---|---|
| baseline | cola_can | −0.130 | −0.118 | 0.006 | 11/12 |
| baseline | water_bottle | +0.340 | +0.320 | 0.008 | 11/12 |
| baseline | milk_carton | −0.260 | −0.236 | 0.006 | 7/12 |
| baseline | bread_loaf | 0.000 | +0.003 | 0.003 | 10/12 |
| **swap** | milk_carton | **+0.000** | **−0.235** | 0.008 | **0/12** |
| **swap** | bread_loaf | **−0.260** | **+0.004** | 0.009 | **0/12** |
| unseen −0.20 | cola_can | −0.200 | −0.118 | 0.006 | 0/12 |
| unseen −0.06 | water_bottle | −0.060 | +0.313 | 0.008 | 0/12 |
| unseen +0.08 | milk_carton | +0.080 | −0.237 | 0.007 | 0/12 |
| unseen +0.22 | bread_loaf | +0.220 | +0.002 | 0.008 | 0/12 |

**Verdict: matched the actual position 0/216; matched the trained position 216/216.**

Reach SD is 0.003–0.010 m — each word triggers an essentially fixed motion. The policy *is*
language-conditioned (the existing Table 4.2 evidence is correct as far as it goes), but the
mapping is **word → memorised trajectory**, not **word → visual referent**.

This is also the mechanism behind §1: a policy that reaches a fixed pose fails as soon as the
object is not exactly there. The two headline findings are one finding.

> **How to frame this.** Do not present it as a failure. It is a precise, measured statement
> about what a compact VLA learns from a small, positionally-fixed dataset — and it is
> supported by three converging lines of evidence (swap, unseen positions, jitter collapse).
> The honest description of the system is *four memorised trajectories plus a language-driven
> switch between them*. That is a more interesting contribution than an unverified grounding
> claim, and it generates the obvious future work: positional randomisation in the training
> data.

---

## 3. EXP-7 — the missing capacity control, and it inverts the expected result

RQ2 asks whether gains come from model capacity or from data/metrics, but SmolVLA was the only
policy ever run on the supermarket task. I trained ACT on the identical dataset, cameras,
action space and protocol.

**Design choice (stated explicitly, as the task file requires):** ACT has no text encoder, so
I trained it **per-item** to remove the item-selection ambiguity and isolate motor capacity —
plus a second ACT on all four items to measure what language conditioning is worth. Per-item
training used `bread_loaf` (episodes 0–51): LeRobot 0.5.2 has a bug where `--dataset.episodes`
subsets the frame table but the sampler still emits global indices (`IndexError: 8008 out of
bounds for size 4888`), so only the first episode block is safely selectable.

| policy | params | bread, fixed scene | bread, ±2 cm jitter | peak GPU |
|---|---|---|---|---|
| SmolVLA 15k | 450M | 75% | 60% | ~3 GB |
| **ACT, per-item** | **52M** | **100%** (20/20) | **85%** (17/20) | **0.95 GB** |

A model **8.7× smaller** is better on this item under both protocols, and degrades far less
under jitter (100→85 vs 75→60).

**What the language pathway is worth.** ACT trained on all four items, language-blind:

| instructed item | item's y | ACT reached y | placed |
|---|---|---|---|
| bread_loaf | 0.00 | −0.124 | 0/20 |
| cola_can | −0.13 | −0.124 | **20/20** |
| milk_carton | −0.26 | −0.124 | 0/20 |
| water_bottle | +0.34 | −0.124 | 0/20 |

**SD = 0.000.** It executes the identical trajectory every trial — the cola grasp — giving 25%
overall (13.8% under jitter). Interpretation (not measurement): ACT's L1 loss is
median/mode-seeking, so rather than blurring into a physically meaningless average reach it
collapses onto one clean mode, and cola is the most forgiving target.

> **Consequence for RQ2.** The current claim — *"data and metrics mattered more than model
> choice"* — is now supported by a same-task control, but the sharper and better-evidenced
> statement is: **at this scale the VLA's contribution is item selection, not manipulation
> quality.** ACT matches or beats SmolVLA at moving the arm; the entire difference is which of
> four trajectories is chosen. Combined with EXP-3, the selection mechanism is a lookup keyed
> on the word.

---

## 4. EXP-2 — paired McNemar (trials are paired by construction)

Because the scene is deterministic and instruction seeds are shared across checkpoints, trials
are paired, so McNemar's exact test applies.

**Placement — no pair of checkpoints differs significantly:**

| comparison | discordant | p | verdict |
|---|---|---|---|
| 5k vs 10k | 11 / 11 | 1.000 | n.s. |
| 5k vs 15k | 16 / 12 | 0.572 | n.s. |
| 5k vs 20k | 14 / 7 | 0.189 | n.s. |
| 10k vs 15k | 20 / 16 | 0.618 | n.s. |
| 10k vs 20k | 15 / 8 | 0.210 | n.s. |
| 15k vs 20k | 14 / 11 | 0.690 | n.s. |

**Grasp — a real training effect does exist:**

| comparison | p | verdict |
|---|---|---|
| 5k vs 15k | **0.008** | 15k better |
| 5k vs 20k | **0.022** | 20k better |
| 10k vs 15k | **0.029** | 15k better |
| 10k vs 20k | **0.031** | 20k better |
| 15k vs 20k | 0.607 | n.s. |

This is a **net gain for the thesis**: the paired test detects a genuine grasp improvement from
5k/10k to 15k/20k that the unpaired two-proportion test could not. But it also confirms there
is no placement difference to explain, so the "mild over-training" narrative must go.

---

## 5. EXP-4 — the 0%→80% story, decomposed

The review document assumed the widened basket loosened the success criterion. **It did not.**
The criterion is hard-coded at `scripts/rollout.py:149` and never reads `BASKET_HALF`:

```python
placed = dx < 0.09 and dy < 0.08 and -0.02 < dz < 0.13
```

That is a **0.18 × 0.16 m** box inside a **0.30 × 0.26 m** tote — *stricter* than the tote, so
an item must land in the middle 60%. Widening the tote changed the physics (catching items that
would roll out) and moved the target 4 cm closer, but did not relax the metric.

Re-scoring the same 320 rollouts (my recomputation agrees with the logged flag on **100%** of
trials, which validates the scoring):

| criterion | overall | 15k | 20k |
|---|---|---|---|
| published box (0.18×0.16 m @ x=0.24) | 57.2% | 58.8% | 62.5% |
| v2 tote footprint (0.30×0.26 m @ x=0.24) | 65.3% | 68.8% | 71.2% |
| v1 tote footprint (0.22×0.20 m @ x=0.28) | 58.1% | 58.8% | 65.0% |
| v1 box at the published ratio (0.13×0.12 m @ x=0.28) | 30.0% | 26.2% | 31.2% |

**The v1 zero was a real behaviour failure, not a metric artefact.** Had v1 used a criterion
scaled the same way to its smaller tote, this policy would still score ~30%, not 0%. The
near-miss diagnosis in `REPORT_NOTES.md:139-144` is vindicated. That part of the story stands.

**But note the baseline is n = 12** (`REPORT_NOTES.md:126`: grasp 9/12, place 0/12), compared
against n = 80. 0/12 has a Wilson 95% CI of roughly 0–24%.

---

## 6. EXP-5 — the multi-item orchestrator (§4.8 currently has no numbers)

100 picks across 40 randomised lists, retries disabled so each position is one clean trial.

| list length | position 1 | position 2 | position 3 | complete list |
|---|---|---|---|---|
| 2 | 65.0% | 70.0% | — | **45.0%** (CI 26–66) |
| 3 | 80.0% | 45.0% | 45.0% | **30.0%** (CI 15–52) |

Grasp for 3-item lists: 100% → 85% → 55%.

**The stated explanation is not supported.** §4.8 and `PROJECT_KNOWLEDGE.md` §9 attribute the
degradation to a full basket being an out-of-distribution view. Scored by basket occupancy
instead of position, there is **no gradient**:

| items already in basket | placement |
|---|---|
| 0 | 61.8% (34/55) |
| 1 | 58.3% (21/36) |
| 2 | 66.7% (6/9) |

Degradation tracks **position in the list**, not basket contents. The likely mechanism, and it
follows directly from EXP-3: earlier picks disturb the remaining items on the shelf, and a
policy that reaches a fixed memorised pose fails as soon as its target has shifted. That is the
same failure mode as the jitter collapse.

> Replace the occlusion explanation with the disturbance explanation, and cite EXP-3 for the
> mechanism. It is better evidenced and it unifies three results.

---

## 7. EXP-6 — why milk is weak (two causes, neither is reach)

**(a) A metric artefact.** `rollout.py:150` defines `grasped = max_lift > 0.05`, the item's peak
height above its *own* rest height. Milk is the tallest item, so it needs least clearance:

| item | median max_lift, fixed scene | median, jittered |
|---|---|---|
| bread_loaf | 0.124 | 0.110 |
| cola_can | 0.102 | 0.101 |
| **milk_carton** | **0.054** | **0.065** |
| water_bottle | 0.071 | 0.048 |

Milk sits essentially *on* the 0.05 m threshold while bread and cola clear it by 2×. Small
variations flip the flag. This is why the published aggregates show placement **exceeding**
grasp for milk (10k: grasp 16/20, placement 20/20) — an item can be lifted <5 cm, drawn off the
shelf and lowered into the tote, succeeding at the task while scoring `grasped = False`.
**Milk's 70% grasp rate is a lower bound on true grasp success, not a measurement of it.**

**(b) A systematic reach compression, affecting all items.** At trained positions:

| item | target y | reached y | error |
|---|---|---|---|
| bread_loaf | 0.000 | +0.003 | +0.003 |
| cola_can | −0.130 | −0.118 | +0.012 |
| milk_carton | −0.260 | −0.236 | +0.024 |
| water_bottle | +0.340 | +0.320 | −0.020 |

|error| correlates with |target y| at **r = 0.919**, a ~6% compression toward the shelf centre.
This is regression-to-the-mean — the signature of a policy interpolating between memorised
trajectories, exactly as EXP-3 predicts. It is **not** an edge-of-reach kinematic limit: the
scripted expert hits 98% on milk with no special tuning, and the water bottle is 8 cm *further*
out yet scores better.

---

## 8. Text fixes — ready to apply (TXT-1 … TXT-8)

No `.tex` files were edited. Locations verified by grep; line numbers as of this branch.

### TXT-1 — 240 → 208 🔴
Verified from the dataset itself: **208 episodes, 52 per item, 19 552 frames, mean 94
frames/episode**. Occurrences: `03-methodology.tex:18, 276, 398`, `05-discussion.tex:15, 39`,
`06-conclusions.tex:9`, `appendix-implementation.tex:39`.

> "240 demonstrations were collected (60 per item across four items); **208 were used for
> training and 32 (8 per item) held out for validation**."

### TXT-2 — zero-shot / fine-tuned, and π0 🔴
Locations: `main.tex:63-64`, `04-results.tex:9, 14`, `02-background.tex:227`.

- Abstract "failed zero-shot" → **"failed even after fine-tuning on RoboCasa demonstrations"**
- `04-results.tex:14` remove "out of the box"
- **Remove π0 from the failure claim entirely.** `~/thesis_scripts/pi0_training.log` shows it
  crashed before step 1 with `AttributeError: 'PaliGemmaForConditionalGeneration' object has no
  attribute 'language_model'`; `checkpoints/pi0_robocasa/` is empty and no π0 eval CSV exists.
  Confirmed results are SmolVLA 0/20 and ACT 0/20 (both n = 20, seeds 100–119).
- If GR00T's ~54% was genuinely zero-shot, state that asymmetry, or Fig. 4.1 compares
  fine-tuned small models against a zero-shot large one without saying so.

### TXT-3 — milk geometry 🔴
`04-results.tex:149` — delete *"It sits at the far lateral edge of the shelf, where the arm is
most extended."* Runtime-verified positions: all items at x = 0.880; milk y = **−0.260**, bottle
y = **+0.340**. The bottle is 8 cm further out and scores better. Also `05-discussion.tex:167`,
delete *"is only reachable at near-full extension."*

> Replacement (now evidenced by EXP-6): "The milk carton's weakness has two causes, neither
> kinematic. First, it is the tallest item, so its peak lift clears the 0.05\,m grasp threshold
> by the smallest margin (median 0.054\,m), making the grasp metric itself unreliable for this
> item. Second, the learned reach is systematically compressed toward the shelf centre in
> proportion to lateral offset (r = 0.92), so the most laterally displaced items are approached
> least accurately."

### TXT-4 — the +0.34 misattribution
`04-results.tex:245-246`. **y = +0.34 is the water bottle**; milk is at −0.26.

> "The small systematic undershoot (reaching +0.29 for a target at +0.34) is not specific to
> that item: reach error scales with lateral offset across all four items (r = 0.92), a ~6%
> compression toward the shelf centre characteristic of a policy interpolating between
> demonstrated trajectories."

### TXT-5 — abstract trial-count conflation
`main.tex:67-69`, plus `01-introduction.tex:115`, `06-conclusions.tex:17`.

> "…achieves an 80% task-completion rate (95% Wilson CI 70–87, **n = 80**) for the selected
> checkpoint, drawn from **320 closed-loop rollouts across four checkpoints**."

Also add to the abstract: **four items**, **208 training demonstrations**, **"in simulation"**,
and that the **base is parked**. State **n = 12** wherever the 0% baseline appears.

### TXT-6 — rewrite §4.6 🔴 (`04-results.tex:268+`)
- **Delete** *"Selecting by training loss, which typically continues to improve, would have
  deployed the worse model."* No loss log exists for this run (`log_freq: 200` was set but
  stdout was never captured; `wandb.enable: false`; `~/wandb` holds only unrelated May runs).
- **Delete or correct** *"the 15k→20k drop is consistent across items."* From
  `outputs/eval_v2_all.log`, placement deltas 15k→20k are cola −1, bread 0, **milk +4**,
  **bottle −7** — the entire aggregate drop is the water bottle.
- Replace the ranking claim with the McNemar result:

> "Paired McNemar tests across the four checkpoints find no significant difference in placement
> (all p ≥ 0.19), so the checkpoints are statistically indistinguishable on the task metric at
> n = 80. Grasp success does improve significantly from 5k/10k to 15k/20k (p = 0.008–0.031),
> though 15k and 20k do not differ (p = 0.61). The methodological point stands independently of
> the ranking: closed-loop success and training loss are not monotonically related, so selection
> must be made on the former."

- Optional but stronger than "over-training": the LR schedule was configured for 30 000 decay
  steps but training stopped at 20 000, so the cosine was truncated at ~2/3 and the LR at 15k
  was still 1.68e-5 (~17% of peak). The run never reached its intended endpoint.

### TXT-7 — the 12 GB / "largest VLA" claim
§2.2. π0 never trained, so the claim is **unresolved by experiment**, not supported. The
7.55 GB in `pi0_training.log` is an after-weight-load figure with no optimiser states,
gradients or activations — it is not a training-memory measurement and must not be shown as
one. Recommended reframing: the 12 GB ceiling constrained model selection and ruled out
unfreezing the backbone with three camera streams. Memory table for Appendix B:

| configuration | peak GPU memory |
|---|---|
| SmolVLA, backbone frozen | ~3 GB (`mem_gb ≈ 3.2`) |
| SmolVLA, vision unfrozen, 3 cameras | OOM |
| **ACT (52M), 3 cameras** | **0.95 GB** (measured, this review) |
| π0 (3.5B) | not measurable — training never started |

### TXT-8 — honest-framing pass
- "deployable" → **"deployable in simulation"** (`main.tex:76`, `04-results.tex:98, 394`,
  `05-discussion.tex:194`)
- Title/abstract: note the **base is parked**
- **RQ1 §1.4** (`01-introduction.tex:91`): define "useful success rate" with an a-priori
  threshold. At 58.8% under object-pose variation, the honest framing is "useful as a
  feasibility demonstration, not for deployment."
- **§4.2 / §5.5**: a simulated 80% is not comparable to Kiviranta's 80% on real SO-101 hardware
- **Fig. 4.1 caption**: add *"these rates are not comparable: task difficulty decreases left to
  right"* — three numbers spanning three tasks, robots and embodiments
- **New, required by §1**: state the evaluation protocol's fixed object pose wherever the
  headline number appears

---

## 9. Tables and Appendix B (FIG-1, FIG-3, FIG-5)

### FIG-1 — per-item × per-checkpoint (was claimed without data)

Published protocol, from `outputs/eval_v2_all.log` — placement, n = 20 per cell:

| ckpt | cola | bottle | milk | bread | overall |
|---|---|---|---|---|---|
| 5k | 18 (90%) | 10 (50%) | 17 (85%) | 10 (50%) | 55/80 68.8% |
| 10k | 19 (95%) | 6 (30%) | 20 (100%) | 10 (50%) | 55/80 68.8% |
| 15k | 19 (95%) | 18 (90%) | 12 (60%) | 15 (75%) | 64/80 80.0% |
| 20k | 18 (90%) | 11 (55%) | 16 (80%) | 15 (75%) | 60/80 75.0% |

Jittered protocol (this review, `outputs/review/exp1_instrumented_jitter.csv`) — full table with
Wilson CIs in `outputs/review/analysis_full.txt`.

### FIG-3 — conditional placement: **do not build the table as specified**
Placement is **not** a subset of grasp (see §7a), so `P(place | grasp)` from marginals would be
invented. Under the published protocol milk shows placement > grasp at 5k and 10k. Report the
grasp-metric caveat instead, or recompute from the per-trial CSVs now available.

### FIG-5 — Appendix B, all constants recovered

| parameter | value |
|---|---|
| Optimiser | AdamW (decoupled weight decay) |
| Peak LR / final decay LR | 1e-4 / 2.5e-6 |
| LR at 15k (actual) | 1.678e-5 |
| Betas / eps | (0.9, 0.95) / 1e-8 |
| Weight decay | 1e-10 |
| Grad clip norm | 10 |
| Scheduler | cosine decay with warmup; 1000 warmup, **30 000 decay steps (truncated at 20 000)** |
| Batch size / steps / seed | 8 / 20 000 / 1000 |
| AMP | disabled |
| **Image augmentation** | **disabled** (`image_transforms.enable: false`) |
| Backbone | frozen (`freeze_vision_encoder`, `train_expert_only`, `train_state_proj`) |
| chunk_size / n_action_steps | 50 / 50 |
| Flow-matching inference steps | 10 |
| VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, 16 layers |
| Input images | 3 × 96×96, resized with padding to 512×512 |
| `ACTION_SUBSTEPS` | **25** (thesis says "roughly twenty-five") |
| Success criterion | \|x−0.240\|<0.09 ∧ \|y\|<0.08 ∧ 0.450<z<0.600 (**0.18×0.16 m**, not the tote) |
| Basket | tote 0.30×0.26 m, centre (0.24, 0.00), floor z = 0.47 |
| v1 basket | half (0.11, 0.10) at x = 0.28 |
| Instruction templates | **6** (not "~20") → 6 × 4 items = **24 instructions** |
| Per-item tuning | `GRIPPER_LENGTH=0.14`, `PRE_GRASP_UP=0.15`, bottle `grasp_gl=0.18`, cereal `yaw=90` |
| Dataset | 208 episodes, 52/item, 19 552 frames, 94 frames/episode, 20 fps |
| Environment | lerobot 0.5.2, torch 2.11.0+cu128, MuJoCo 3.9.0, numpy 2.2.6, CUDA 12.8, RTX A2000 12GB |

Add a **Code and data availability** statement — §1.5 claims a "complete, reproducible pipeline"
but there is no public repo, licence or dataset location.

---

## 10. Figures, citations, structure (FIG-4, FIG-6, FIG-7, FIG-8)

**Not applied** (would require editing `thesis/` and regenerating figures). Verified and listed:

- **FIG-4:** Fig. 3.3 label collisions ("cola (y=−0.13)" over "basket"; "cereal excluded" over
  the shelf) — this is the figure that reveals the milk/bottle geometry, so it must be legible.
  Fig. 3.6 blank black panel bottom-right. Fig. 3.5 caption says "wrist and scene" but omits the
  basket camera, central to the §4.4 fix. Fig. 3.2 shows the excluded cereal box. Fig. 4.1 needs
  the comparability caveat.
- **FIG-6 — citations (verified):** §2.5 robotic grasping has **0 citations** across the whole
  section. Chapter 3 has only **6** across 27 KB, with `mink`, robosuite, MuJoCo Menagerie and
  SmolVLM-2 all uncited. Chapter 4 has **4**. Add: DAgger (Ross 2011), behaviour cloning
  (Pomerleau 1988), domain randomisation (Tobin 2017), LoRA (Hu 2021), flow matching
  (Lipman 2023). Ref [10] should cite the SmolVLA paper (arXiv 2506.01844), not the HF model
  page; ref [8] has no arXiv ID or venue.
- **FIG-7 — repetition:** *"data and metrics mattered more than model choice"* appears in
  `main.tex`, `01-introduction`, `02-background`, `04-results` and `06-conclusions`. Given EXP-7
  now provides a same-task control, this conclusion should be **restated once, precisely**
  ("the VLA's contribution is item selection, not manipulation quality") rather than repeated.
- **FIG-8:** fourth-level headings render as `2.1.0.0.1` and propagate into the ToC; naming
  inconsistency ("Pi0"/"π0", "GR00T N1.6"/"N1.x", "basket"/"tote"); add **EGL** to abbreviations
  and remove unused **RGB-D**; §3.1.1 should forward-reference the basket camera as added by the
  §4.4 failure analysis, or Chapter 3 reads as concealed iteration.

---

## 11. Not done, and why

| Item | Status |
|---|---|
| **Appendix A ethics letter** | **Still the template** — `CURES/67890/2026`, "Full Title of Individual Research Project or Thesis". Cannot be automated; retrieve the real CURES letter from the Cranfield portal. Highest damage-to-effort ratio in the document. |
| FIG-2 loss curve | **Not producible.** No loss log exists for the v2 run. |
| `.tex` edits (TXT-1..8, FIG-4/6/7/8) | Not applied at your request; wording supplied above. |
| ACT per-item on cola/milk/bottle | Blocked by the LeRobot episode-subsetting bug (§3). Only the first episode block is safely selectable; bread was used. |
| EXP-3 extrapolation beyond the trained span | Only interpolation tested (−0.20 … +0.22, inside −0.26 … +0.34). |

**Limitations of this review.** Jitter was tested at one magnitude (±2 cm, matching
`collect_data.py`); the degradation curve as a function of jitter is unmeasured. The ACT
comparison rests on one item. EXP-5 used 20 lists per length, so per-position CIs are wide
(±20 points).

---

## 12. Files

```
REVIEW_FINDINGS.md                        Phase 0 read-only investigations (INV-1..6)
REVIEW_REPORT.md                          this file
THESIS_REVIEW_TASKS.md                    the task queue (copied into the repo)
scripts/exp3_grounding.py                 position-override grounding test
scripts/exp5_multiitem.py                 multi-item orchestrator quantification
scripts/analyse_review.py                 Wilson CIs, McNemar, basket re-scoring
scripts/rollout.py                        + jitter, + per-trial info, + ACT dispatch
scripts/eval_checkpoints.py               + paired seeding, + torch seeding, + CSV
outputs/review/exp1_instrumented_jitter.csv   320 trials, jittered
outputs/review/control_nojitter_15k.csv       80 trials, fixed scene (the control)
outputs/review/exp3_grounding.csv             264 trials, grounding
outputs/review/exp5_multiitem.csv             100 picks, 40 lists
outputs/review/exp7_act_*.csv                 4 files, ACT baseline
outputs/review/analysis_full.txt              full statistics output
outputs/review/act_bread_loaf/, act_all_items/  ACT checkpoints
```

Commits on `thesis-review-fixes`: `fc77bc1` (Phase 0), `1dd1095` (instrumentation),
`6fc2267` (experiments).

---

## 13. What I would do with this

1. **Re-frame around what you measured.** The strongest version of this thesis is: *a compact
   VLA fine-tuned on 208 positionally-fixed demonstrations learns a language-driven switch over
   memorised trajectories rather than language-grounded visual selection; at this scale a 52M
   ACT matches its manipulation quality at a third of the memory, so the VLA's contribution is
   item selection.* Every clause is now backed by an experiment.
2. **Report 58.8%,** with 80% as the fixed-pose special case. Defensible beats impressive.
3. **Fix the ethics letter today.** Five minutes.
4. **Then** the mechanical corrections: 208, π0, milk geometry, +0.34, §4.6, citations.

The obvious next experiment — and the one a viva panel will ask for — is to re-collect
demonstrations with randomised item positions and retrain. EXP-3 predicts that is what converts
the lookup table into genuine grounding, and it would turn this review's findings into the
thesis's strongest chapter.
