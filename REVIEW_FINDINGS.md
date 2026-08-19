# Phase 0 findings — 2026-08-19

Read-only investigation for `THESIS_REVIEW_TASKS.md`. No files under `outputs/`,
`envs/`, `scripts/` or `thesis/` were modified. Branch: `thesis-review-fixes`.

Every claim below is backed by a path + line number or a command output.

---

## INV-1 — training loss log

**NOT FOUND.** There is no per-step loss series for the supermarket v2 run.

`outputs/train/smolvla_supermarket_v2/` contains only `checkpoints/`. The complete set
of non-weight files across all four checkpoints is:

```
pretrained_model/{config.json, train_config.json, policy_preprocessor.json, policy_postprocessor.json}
training_state/{training_step.json, optimizer_param_groups.json, scheduler_state.json}
```

None is a metrics log. Searched further and found nothing:
- no `*.log`, `nohup.out`, `*loss*` under `outputs/train/`
- no `wandb/` directory in the repo (`train_config.json` → `"wandb": {"enable": false}`)
- `~/wandb/` exists but holds only three runs dated **2026-05-21**, none named for the
  supermarket project; the v2 run trained **2026-07-27**. Unrelated.

`train_config.json` sets `"log_freq": 200`, so LeRobot **did** print a loss line every
200 steps — to stdout, which was never redirected to a file. The numbers existed at
run time and were not captured.

> **Consequence:** TXT-6 → **DELETE** the §4.6 sentence *"Selecting by training loss,
> which typically continues to improve, would have deployed the worse model."* There is
> no record of the loss for this run, so the claim cannot be evidenced. FIG-2 (loss
> curve) is **not producible** without retraining.
>
> Note: pilot logs in `~/thesis_scripts/` **do** contain loss series
> (`training.log`, `smolvla_fixedbase.log` ~1200 `loss=` lines, `act_fixedbase.log`).
> Those are the RoboCasa/fixed-base pilots — a different run, model and task. They
> cannot substitute for the supermarket curve and must not be relabelled as such.

---

## INV-2 — are checkpoint eval trials seed-matched?

**PAIRED — and more strongly than the review assumed, because the evaluation scene is
fully deterministic.**

Evidence:

| Location | Finding |
|---|---|
| `scripts/eval_checkpoints.py:86` | `eval_checkpoint(..., args.seed)` — the **same** `--seed` (default 1000) is passed for every checkpoint |
| `scripts/eval_checkpoints.py:47` | `rng = np.random.default_rng(base_seed + i)`, `i` = item index → identical instruction sequence per item across all four checkpoints |
| `scripts/rollout.py:118` | `run_episode(...)` takes **no seed argument** |
| `envs/supermarket_env.py:432-446` | `reset()` restores a fixed `qpos`, calls `mj_forward`, and applies **no randomisation of any kind** |
| `scripts/scripted_expert.py:74-81` | positional jitter (`±0.025 m`) lives **only** in `pick_place()` — the data-collection path |
| `scripts/rollout.py`, `eval_checkpoints.py` | **no** `torch.manual_seed` / `np.random.seed` anywhere |

### What this actually means — the important part

The published evaluation contains **no scene variation at all**. Every trial for a
given item starts from a bit-identical physical state; MuJoCo is deterministic. Across
the 20 trials per item, the only things that change are:

1. **the instruction paraphrase** (6 templates — see INV-6), and
2. **SmolVLA's flow-matching sampling noise**, which is unseeded.

The ±2 cm jitter described in `PROJECT_KNOWLEDGE.md` §6 is a **training-data** property.
It is not exercised at evaluation time. The `eval_checkpoints.py` docstring (line 3-4,
*"fresh instruction + seed each"*) is misleading: the seed advances the instruction
sampler only.

Two consequences for the thesis, both of which an examiner can reach from the source:

- **The Wilson CIs do not mean what the thesis implies.** n=20 per item is 20 draws over
  *paraphrase × model sampling noise* at one fixed object pose — not 20 draws over task
  instances. Any claim that the CI captures robustness to object placement is
  unsupported. This needs stating explicitly in §3.5/§4.2 wherever the protocol is
  described.
- **The evaluation is not reproducible.** With no torch seed, re-running
  `eval_checkpoints.py` today will not reproduce Table 4.3.

> **Consequence:** EXP-2 (McNemar's) is **valid in principle** — trials are paired by
> construction — but **cannot be run on existing data**, because per-trial outcomes were
> never saved (see INV-3). It requires a re-run with per-trial logging.
>
> EXP-1 as specified in the task file is **much weaker than intended**: "disjoint seeds"
> would only change the instruction paraphrases, not the scene. A meaningful fresh-seed
> re-evaluation must **add** per-episode pose jitter to the eval path (mirroring
> `scripted_expert.py:74-81`) — which makes it a new experiment, not a re-run.

---

## INV-3 — per-item × per-checkpoint data

**FOUND — as a log, not a CSV. No CSV exists anywhere under `outputs/`.**

`scripts/eval_checkpoints.py` has no `to_csv` / `csv` / `DataFrame` call; `per_item`
(line 45-54) is accumulated in memory, printed (line 57-58), and discarded. Per-trial
outcomes are never retained — only per-item counts.

The full 4×4 matrix survives in **`outputs/eval_v2_all.log`** (42 lines, 2026-07-31),
which is the source of the published Table 4.3:

| ckpt | cola grasp/place | bottle grasp/place | milk grasp/place | bread grasp/place | overall grasp | overall place |
|---|---|---|---|---|---|---|
| 5k  | 19/18 | 10/10 | 14/**17** | 15/10 | 72.5% | 68.8% |
| 10k | 20/19 | 11/6  | 16/**20** | 17/10 | 80.0% | 68.8% |
| 15k | 20/19 | 20/18 | 14/12 | 20/15 | 92.5% | 80.0% |
| 20k | 20/18 | 16/11 | 15/16 | 20/15 | 88.8% | 75.0% |

(all counts out of 20)

### Three things this table settles

**(a) §4.6's "the 15k→20k drop is consistent across items" is FALSE.** Placement deltas
15k→20k: cola −1, bread 0, milk **+4**, bottle **−7**. The entire aggregate drop
(64→60) is the **water bottle**; milk improves. This is a direct contradiction and must
be corrected regardless of what any re-run shows.

**(b) §4.3's "milk placed more reliably at earlier checkpoints" is TRUE and now
quantifiable:** milk placement = 17, 20, 12, 16 across 5k/10k/15k/20k.

**(c) A metric defect: placement is NOT a subset of grasp.** Milk at 10k scores
grasp 16/20 but placement **20/20**; at 5k, 14/20 grasp vs **17/20** placement. At least
four episodes were scored *placed but not grasped*.

Cause: `scripts/rollout.py:150` defines `grasped = max_lift > 0.05`, where `max_lift` is
the item's peak height **above its own shelf rest height**. An item can be lifted <5 cm,
drawn off the shelf front and lowered ~0.5 m into the tote — succeeding at the task
while scoring `grasped = False`. Milk is the tallest item (rest z = 1.110 m), so it
needs the least clearance and is the most exposed to this.

> **Consequence:** the "milk is a grasp problem" narrative (§4.3, §5.2, and FIG-3 in the
> task file) is **partly a metric artefact**. Milk's 70% grasp rate at 15k is a lower
> bound on true grasp success, not a measurement of it.
>
> **FIG-3 as specified cannot be built.** A `P(place | grasp)` column requires per-trial
> joint outcomes, which do not exist, and the aggregates prove the two events are not
> nested. Producing that table from the marginals would be inventing data. It needs the
> EXP-1/EXP-2 re-run with per-trial logging.

---

## INV-4 — v1 (narrow) basket dimensions

**RECOVERED — from documentation, not git.**

`REPORT_NOTES.md:48-50`:

> **Basket:** blue plastic tote on a solid pedestal, riding on the AGV deck.
> v1: half-size **(0.11, 0.10)** at **x = 0.28**. v2: enlarged to **(0.15, 0.13)**, moved
> to **x = 0.24** (more forgiving drop target).

So **two** things changed, not one: footprint 0.22×0.20 m → 0.30×0.26 m **and** centre
x 0.28 → 0.24 m (4 cm closer to the arm).

Git is no help here: the repository has **exactly one commit**
(`f5fdd40`, 2026-08-18) containing the entire project. There is no v1→v2 diff to inspect.

### The review document's premise for EXP-4 is wrong, and this matters

`THESIS_REVIEW_TASKS.md` assumes the widened basket loosened the success criterion. It
did not. The criterion is **hard-coded in `scripts/rollout.py:149`** and never reads
`BASKET_HALF`:

```python
placed = dx < 0.09 and dy < 0.08 and -0.02 < dz < 0.13
```

Verified at runtime (basket body at `[0.24, 0.00, 0.47]`):

```
success box    : |x−0.240| < 0.09 ,  |y| < 0.08 ,  0.450 < z < 0.600
success footprint : 0.18 × 0.16 m
v2 tote footprint : 0.30 × 0.26 m
```

The scoring box is **substantially stricter than the tote itself** — an item must land
in the middle 60% of the tote to count. So widening the tote did **not** relax the
metric. It changed the *physics* (a wider tote catches an item that would otherwise roll
out) and, via the 4 cm x-shift, moved the target closer to where the policy was already
dropping.

Whether the v1 `rollout.py` used a different hard-coded box is **unrecoverable** — one
commit, no history, and `REPORT_NOTES.md` does not record it.

> **Consequence:** EXP-4 is **not a pure re-scoring** and is **not blocked on INV-4**.
> It is blocked on data: `final_item_xyz` was never logged (no CSVs exist), so the
> v1-box counterfactual cannot be computed from existing results. It requires re-running
> the 15k checkpoint with final-position logging, then scoring against both boxes.
>
> Separately, §5.6 must state that the v1 success box is unrecoverable, so the
> tolerance-vs-behaviour split can only be estimated for the tote geometry, not for the
> scoring criterion.

### Additional finding — the 0% baseline is n = 12

`REPORT_NOTES.md:126`: *"**Grasp: 9/12 (75%). Place: 0/12 (0%).** (Small sample — see §5
caveat.)"*

The headline "0% → 80%" compares **12 trials** against **80 trials**. The thesis
presents both as comparable. The 0% needs its n stated everywhere it appears
(0/12 has a Wilson 95% CI of roughly 0–24%, so "0%" is itself an imprecise estimate).

### Additional finding — the language-grounding table is from v1

`REPORT_NOTES.md:127-137` places the grounding table (milk −0.26 → −0.24, bread 0.00 →
−0.01, bottle +0.34 → +0.29) under the **### v1** heading. That evidence therefore comes
from the **2-camera v1 model with the boundary bug**, not the 15k v2 checkpoint whose
results are headlined. If the thesis presents it as a property of the reported policy,
that is a mismatch — and it is cheap to fix by re-running the test on 15k as part of
EXP-3.

---

## INV-5 — π0 training configuration

**Resolution: none of (a)/(b)/(c). π0 was never trained at all.**

`~/thesis_scripts/pi0_training.log` (1.8 KB, complete) shows the run crashed **before
the first training step**:

```
Creating policy...
  Total: 3501.4M | Trainable: 578.0M (16.5%)
Loading pretrained pi0_base weights (direct download)...
  Transferred 340/777 tensors.
  GPU after load: 7.55 GB

Training pi0 for 20000 steps (batch=8)
Traceback (most recent call last):
  File ".../train_pi0.py", line 127, in <module>
    output = policy.forward(batch)
  ...
  File ".../paligemma_with_expert.py", line 229, in embed_language_tokens
    return self.paligemma.language_model.embed_tokens(tokens)
AttributeError: 'PaliGemmaForConditionalGeneration' object has no attribute 'language_model'
```

This is a **library API incompatibility** (a transformers version where PaliGemma's
submodule was renamed), not a memory failure. Corroborating evidence:

- `~/thesis_scripts/checkpoints/pi0_robocasa/` is **empty** — no `.pt` file. The other
  four pilot dirs all contain checkpoints (`act_robocasa/step_020000.pt`,
  `smolvla_robocasa/step_020000.pt`, `act_fixedbase/step_100000.pt`,
  `smolvla_fixedbase/step_060000.pt`).
- The traceback path is `~/miniconda/envs/robocasa/lib/python3.10/` — the pilot ran
  under a **conda env**, not the project venv.
- Only two eval CSVs exist (`eval_results.csv`, `eval_act_results.csv`). **There is no
  π0 eval result.**

### What this forces

- **The abstract's assertion that π0 failed on RoboCasa is unsupported and must be
  removed or restated.** π0 produced no result to fail with. The honest statement is
  that π0 could not be brought up in this environment because of a library
  incompatibility, so no π0 result is reported. Leaving the current wording is a
  serious viva risk: the log is one file away.
- **TXT-7:** the 12 GB question is *unresolved by experiment*, not resolved either way.
  π0 reached 7.55 GB **after weight load only** — with no optimiser states, activations
  or gradients allocated, since it never completed a forward pass. That figure is not a
  training-memory measurement and must not be presented as one.

Confirmed RoboCasa results (both n = 20, identical seeds 100–119, all `success=False`):

| model | RoboCasa result | source |
|---|---|---|
| SmolVLA | 0/20 | `~/thesis_scripts/eval_results.csv` |
| ACT | 0/20 | `~/thesis_scripts/eval_act_results.csv` |
| π0 | **no result — never trained** | `pi0_training.log`, empty checkpoint dir |

---

## INV-6 — constants for Chapter 3 and Appendix B

### Instruction templates — the thesis figure is wrong

```
TEMPLATES: 6
ITEMS:     5  ['milk_carton', 'cola_can', 'bread_loaf', 'cereal_box', 'water_bottle']
TOTAL:     30  (nominal)
```

The six templates in `shared/instruction_templates.py`:

```
pick up the {item} and place it in the basket
grab the {item} and put it in the basket
take the {item} and drop it in the basket
put the {item} into the basket
collect the {item} and place it in the basket
pick the {item} off the shelf and put it in the basket
```

**There are 6 paraphrase templates, not "~20".** For the four evaluated items the
instruction space is 6 × 4 = **24 distinct strings** (30 if cereal is counted, but cereal
is excluded from all experiments). `PROJECT_KNOWLEDGE.md` §6 and the thesis both say
"~20 instruction paraphrases" — the "20" appears to be the *total instruction count*
misread as a template count.

Combined with INV-2, this sharpens the picture: each item's 20 evaluation trials draw
from only **6 phrasings at one fixed object pose**, so paraphrases repeat ~3.3× each.

### Success criterion (exact) — `scripts/rollout.py:149` and `:193`

```python
placed  = dx < 0.09 and dy < 0.08 and -0.02 < dz < 0.13   # dx,dy = |Δ| to basket body; dz signed
grasped = max_lift > 0.05                                  # peak height above own rest height
```

Numerically, with the basket body at `[0.24, 0.00, 0.47]`:

```
|x − 0.240| < 0.09 m   ∧   |y − 0.000| < 0.08 m   ∧   0.450 m < z < 0.600 m
```

Thesis currently says only "inside the footprint and low". Note this is **0.18 × 0.16 m**,
*not* the tote footprint of 0.30 × 0.26 m — Appendix B must not present `BASKET_HALF` as
the criterion (see INV-4).

### Training hyperparameters — from `checkpoints/015000/pretrained_model/train_config.json`

| Parameter | Value |
|---|---|
| Optimiser | **AdamW** (decoupled weight decay) |
| Peak LR | **1e-4** |
| Betas / eps | (0.9, 0.95) / 1e-8 |
| Weight decay | **1e-10** |
| Grad clip norm | **10** |
| Scheduler | **cosine decay with warmup** |
| Warmup steps | **1000** |
| Decay steps | **30000** |
| Final decay LR | 2.5e-6 |
| LR actually reached at 15k | **1.678e-5** (`scheduler_state.json`) |
| Batch size | 8 |
| Steps | 20000 |
| Seed | **1000** |
| `use_amp` | **false** |
| **Image augmentation** | **DISABLED** (`image_transforms.enable: false`) |
| `freeze_vision_encoder` | true |
| `train_expert_only` | true |
| `train_state_proj` | true |
| chunk_size / n_action_steps | 50 / 50 |
| Flow-matching inference steps | 10 |
| VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, 16 layers |
| Input images | 3 × 96×96, **resized with padding to 512×512** |
| State/action norm | MEAN_STD; visual IDENTITY |
| `ACTION_SUBSTEPS` | **25** (`rollout.py:26`) — thesis says "roughly twenty-five" |

Two items worth calling out in Appendix B:

- **The LR schedule never completed.** `num_decay_steps = 30000` but training stopped at
  20000, so the cosine decay was truncated at ~2/3. At 15k the LR was still 1.68e-5,
  ~17% of peak. This is a legitimate alternative reading of the 15k-vs-20k difference
  and is more defensible than "mild over-training".
- **No image augmentation was used.** Relevant to the generalisation discussion in §5.3
  and to EXP-3 — with a fixed scene, no jitter at eval, and no augmentation, the
  lookup-table hypothesis is more plausible, not less.

### Environment

```
lerobot 0.5.2   torch 2.11.0+cu128   mujoco 3.9.0   numpy 2.2.6
CUDA 12.8       NVIDIA RTX A2000 12GB
```

### Item positions (runtime-verified) — confirms TXT-3 / TXT-4

| item | x | y | z |
|---|---|---|---|
| milk_carton | 0.880 | **−0.260** | 1.110 |
| cola_can | 0.880 | −0.130 | 1.085 |
| bread_loaf | 0.880 | 0.000 | 1.070 |
| cereal_box | 0.880 | 0.150 | 1.125 |
| water_bottle | 0.880 | **+0.340** | 1.107 |

All five sit at identical x = 0.880 m. The water bottle is **8 cm further from the
centreline** than the milk carton, and scores 100% grasp / 90% placement at 15k. TXT-3
and TXT-4 are confirmed: milk is not the most-extended item, and the +0.34 undershoot
belongs to the bottle.

---

## Consequences for later tasks

| Task | Status | Reason |
|---|---|---|
| **TXT-1** (240→208) | RUNNABLE | independent of Phase 0 |
| **TXT-2** (zero-shot) | RUNNABLE, **and widen** | must also fix the π0 claim (INV-5) |
| **TXT-3 / TXT-4** (milk geometry) | **CONFIRMED — RUNNABLE** | runtime positions verified |
| **TXT-5** (n=80 vs 320) | RUNNABLE | + state n=12 for the 0% baseline (INV-4) |
| **TXT-6** (§4.6) | **DELETE loss claim** (INV-1); rewrite ranking | + fix "consistent across items", which INV-3 disproves outright |
| **TXT-7** (12 GB) | **REWRITE — new resolution** | π0 never trained (INV-5); 7.55 GB is a load-only figure |
| **TXT-8** (framing) | RUNNABLE | + add the fixed-scene caveat from INV-2 |
| **EXP-1** (fresh seeds) | **RESCOPE** | needs pose jitter added to eval, not just new seeds (INV-2) |
| **EXP-2** (McNemar's) | **VALID but needs re-run** | paired by construction; per-trial data never saved (INV-3) |
| **EXP-3** (grounding) | **RUNNABLE — highest value, and stronger than assumed** | fixed scene + no augmentation + 6 templates make the lookup hypothesis more likely; existing grounding evidence is from v1 |
| **EXP-4** (narrow basket) | **RESCOPE — not blocked on INV-4** | v1 tote = (0.11, 0.10) @ x=0.28 recovered; but needs a re-run logging `final_item_xyz` |
| **EXP-5** (multi-item) | RUNNABLE | unchanged |
| **EXP-6** (milk diagnostic) | **RESCOPE** | must first separate the grasp-metric artefact (INV-3c) from real grasp failure |
| **EXP-7** (ACT baseline) | RUNNABLE | infrastructure confirmed in `~/thesis_scripts/` |
| **FIG-1** (per-item × ckpt) | **RUNNABLE NOW, zero simulation** | full 4×4 matrix in `outputs/eval_v2_all.log` |
| **FIG-2** (loss curve) | **NOT PRODUCIBLE** | no loss log (INV-1) |
| **FIG-3** (conditional placement) | **NOT PRODUCIBLE as specified** | place ⊄ grasp (INV-3c); needs per-trial data |
| **FIG-5** (Appendix B) | **RUNNABLE NOW** | all constants recovered above |

### The two highest-value items, in my judgement

1. **Add per-trial logging to `eval_checkpoints.py`** (`seed, item, instruction,
   checkpoint, grasp, place, final_item_xyz`). It is a small change and it single-handedly
   unblocks EXP-1, EXP-2, EXP-4, EXP-6 and FIG-3 — five tasks currently blocked by the
   same missing artefact.
2. **EXP-3**, which INV-2 and INV-6 have made considerably more pointed: a deterministic
   scene, no eval-time jitter, no image augmentation, 6 paraphrases and 208 episodes is
   close to the ideal setting for a policy to learn word→trajectory rather than
   word→visual-referent.

### Corrections to `THESIS_REVIEW_TASKS.md` itself

- EXP-4's premise (widened basket = loosened success criterion) is **incorrect**; the
  criterion is hard-coded and stricter than the tote.
- INV-4 is **not** the blocker for EXP-4; missing `final_item_xyz` logging is.
- FIG-3's table is **not derivable** from published aggregates.
- EXP-1's "disjoint seeds" would not vary the scene.

---

**STOP — Phase 0 complete. Awaiting go-ahead before Phase 1.**
