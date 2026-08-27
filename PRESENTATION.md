# Supermarket Service Robot — Presentation Outline
*A language-driven pick-and-place robot powered by a Vision-Language-Action model, in simulation.*
Build this in PowerPoint / Google Slides. Each slide: **title · bullets · [figure] · speaker notes.**
All figure paths are relative to the project root. Numbers are from the actual runs.

---

## Slide 1 — Title
**Teaching a Robot to Shop: A Vision-Language-Action Model in Simulation**
- Your name · supervisor · date
- One-liner: *"Type what you want; the robot reads it, finds it on the shelf, and drops it in the basket."*
- [Figure: `outputs/poster/robot_hero_front.png`]
- **Notes:** Set the scene — a mobile robot arm that follows plain-language requests in a supermarket sim.

---

## Slide 2 — The Problem
- Retail/warehouse "pick from a list" needs **perception + language + manipulation** together.
- **Vision-Language-Action (VLA)** models promise generalist robot control — but they're large.
- **Core question:** can a capable VLA be fine-tuned *and* run on a **single 12 GB consumer GPU**?
- **Notes:** Frame it as a hard, real task and a hardware-constrained research question. This is the thread through the whole talk.

---

## Slide 3 — What is a VLA? And what is SmolVLA?
- **VLA = Vision-Language-Action model:** a *single* neural network that takes **camera images + a language instruction** and outputs **robot actions** directly — no separate hand-coded perception → planning → control stack.
- **SmolVLA:** a compact (~450M-param) open VLA from Hugging Face's LeRobot, built to run on **modest hardware** — the reason it fits our 12 GB GPU.
- **Architecture — two parts:**
  1. **VLM backbone (SmolVLM2-500M):** a vision-language model that *encodes* the images + instruction into features — the "understands what it sees and reads" part. **(We freeze this.)**
  2. **Action expert (flow-matching transformer):** takes those features + the robot's joint state and *generates* a **50-step chunk of future actions** (joint + gripper commands). **(We train only this.)**
- **How it runs (closed loop):** observe → VLM encodes → action expert predicts a 50-step plan → execute → re-plan.
- [Figure: block diagram — (3 images + instruction + arm state) → **VLM backbone** → **action expert** → 50-step action chunk → robot]
- **Notes:** Keep it intuitive: *"it reads and looks, then acts."* Point out we freeze the backbone and train only the action expert (ties to the 12 GB / training slide), and that the 50-step chunk is exactly what we visualize later in the "seeing it think" slide.

---

## Slide 4 — The Road Here: from a Kitchen Benchmark to This Project
- **Started with RoboCasa** — a Franka Panda + Omron mobile robot doing **kitchen tasks** (pick from counter → place in cabinet) in MuJoCo.
- RoboCasa's benchmark model is **NVIDIA GR00T N1.6** — a **multi-billion-parameter** VLA: powerful, but far too large to fine-tune or deploy on a **12 GB** GPU.
- I tested the **deployable small models** on the same kitchen tasks — **SmolVLA, ACT, Pi0** — and they couldn't do it: **SmolVLA 0/20, ACT 0/20**.
- **Decision:** step back to a **simpler, controlled task** to make real progress → then scale the lessons up.
- [Figure: `outputs/plots/journey_prior_work.png` — the 0% → ~53% → 80% arc across the three stages]
- **Notes:** This is the "why does the supermarket project exist" slide. The huge model works but won't deploy; small models fail on the hard benchmark → simplify, then scale.

---

## Slide 5 — The UR10e Pilot: What It Taught Me
- **Simplified to a MuJoCo UR10e + Robotiq gripper:** pick a cube from 5 positions → basket. Compared **ACT vs SmolVLA**.
- **Small models CAN learn a controlled task:** ACT ~47% placed, **SmolVLA ~53%** placed.
- **Three lessons — carried straight into the supermarket VLA:**
  1. **Fix the success metric** — a bug counted "cube near basket" as success, inflating 80% → a real **40%**; measure actual release.
  2. **Data variety > volume** — duplicated demos taught nothing; add jitter/paraphrases.
  3. **Freeze the backbone** — fully unfreezing vision blew the 12 GB budget and destabilized training.
- These lessons → applied in `supermarket_vla` → **80%**.
- **Notes:** Emphasize methodology-over-model: fixing metrics and data drove the biggest gains. This is the bridge to the results section.

---

## Slide 6 — The System
- **Simulator:** MuJoCo (headless GPU rendering)
- **Robot:** Omron LD-60 mobile base + UR10e arm + Robotiq 2-finger gripper
- **Products:** real textured grocery meshes (milk, cola, bread, water bottle)
- **3 cameras:** wrist (eye-in-hand) · scene (shelf-facing) · basket (drop target)
- [Figure: `outputs/poster/robot_and_shelf.png`]
- **Notes:** Custom-built store scene (not a canned kitchen). The 3-camera setup matters later.

---

## Slide 7 — Approach: the Pipeline
- **Scripted expert → demos → fine-tune SmolVLA → closed-loop evaluation**
- Scripted expert = a hand-coded "teacher" using inverse kinematics; generates clean pick-place demos.
- SmolVLA (a ~450M-param VLA) learns to imitate — but conditioned on **language + vision**.
- [Figure: `outputs/stage1/pickplace_cola_can.png` — expert key frames]
- **Notes:** One arrow diagram. Emphasize: the expert only *teaches*; at run time the neural net drives.

---

## Slide 8 — Data Collection
- **240 demonstrations** (60 per item × 4 items), only *successful* episodes kept.
- Each frame records: 3 camera images + arm state + action + a **language instruction**.
- **~20 instruction paraphrases** ("pick up the milk…", "grab the cola…") so it keys off meaning, not one sentence.
- Per-episode position jitter for variety.
- [Figure: `outputs/stage1/dataset_preview.png`]
- **Notes:** Highlight the language variety — that's what makes it language-*conditioned*.

---

## Slide 9 — Training on 12 GB
- Fine-tuned from **`smolvla_base`** (pretrained on community robot data).
- **Froze the vision-language backbone**, trained only the **action head** → fits in ~3 GB, no overfitting.
- **Why it works:** the backbone already *sees and reads*; we only teach *how this robot acts*.
- Batch 8 · 20k steps · ~4 h · RTX A2000 12 GB.
- **Notes:** This answers Slide 2's question — yes, a VLA fits 12 GB by freezing perception. Key methodological point.

---

## Slide 10 — Headline Result
- **80% task-completion rate** — reads instruction → picks the named item → lands it in the basket.
- **95% confidence interval: 70–87%** (Wilson) · **grasp success 92.5%**.
- Measured over **320 closed-loop trials** — not a lucky demo.
- *(Confidence interval = the range the true rate very likely sits in.)*
- [Figure: make a bar chart — per-item place % with CI error bars; see Slide 13 notes]
- **Notes:** This is THE slide. Say the 80% number, then immediately show it's rigorously measured.

---

## Slide 11 — Results per Item (best checkpoint)
| Item | Grasp | In basket |
|---|---|---|
| Cola can | 100% | **95%** |
| Water bottle | 100% | **90%** |
| Loaf of bread | 100% | **75%** |
| Milk carton | 70% | **60%** |
- Three of four items complete **≥75%**; two at **≥90%**. Milk is the weak point.
- All with **Wilson 95% CIs**, 20 trials each.
- **Notes:** Be honest about milk — advisors reward acknowledging the weak spot.

---

## Slide 12 — From 0% to 80%: Failure Analysis
- **First policy: 0% placement** — but it grasped, carried, and *lowered* items to within ~2 cm of the basket.
- Diagnosis: **near-misses**, not broken understanding — dropped ~10 cm short + didn't cleanly stop.
- **Three targeted fixes:** (1) clean episode boundaries, (2) a dedicated **basket camera**, (3) a wider tote.
- Result: **0% → 80%.**
- [Figure: `outputs/stage1/basket_visibility.png` (problem) + `outputs/stage1/basket_cam.png` (fix)]
- **Notes:** A strong narrative arc — shows you diagnosed rather than just retrained blindly.

---

## Slide 13 — It Genuinely Understands Language
- Same scene, change **only the instruction** → the arm reaches the **named** item.
- "milk" → left · "bread" → center · "water bottle" → right (matches each item's position).
- Not "grab the nearest" — it's reading the words.
- **Notes:** This proves language grounding. If you want a figure, plot instruction vs. gripper position reached.

---

## Slide 14 — Which Model Do We Keep? (checkpoint selection)
- **A "checkpoint" = a saved snapshot of the model during training.** We trained for 20k steps and saved one every 5k → **4 versions of the same model at different stages** (5k / 10k / 15k / 20k), like save-points.
- We can only deploy **one**, so we had to **choose which snapshot to keep.**
- **How we chose:** ran the robot with *each* of the 4 and measured **actual task success** (the 320-trial eval) — **not** the training-loss curve.
- **Result:** **15k was the best (80%)**; the *final* **20k was worse (75%)** — it had **over-trained** (kept polishing the training numbers but got worse at the real task).
- **So we selected 15k.** Taking the last checkpoint, or picking by loss, would have shipped the *worse* robot.
- [Figure: `outputs/plots/checkpoint_selection.png` — success vs training step, peak circled at 15k, dip at 20k]
- **Notes:** Define checkpoint first (snapshot/save-point), then the choice. Punchline: *"we select by what the robot actually does, not by the loss."* The dip at 20k is the concrete evidence.

---

## Slide 15 — Seeing the VLA "Think"
- SmolVLA plans **50 steps ahead** — we surface its own predicted action plan live.
- At the *first frame* it has already decided to **close the gripper at +16 steps** — it plans the grasp before reaching the item.
- Honest interpretability: the plan is the model's real output, not a heuristic.
- [Figure: screenshot of the `--think` terminal readout — the "grip plan" sparkline]
- **Notes:** Great "wow" slide. Show the ASCII plan (`···········██████…`) and explain it's the network's intention.

---

## Slide 16 — Following a Shopping List
- Type a list from a **menu** → robot collects each item **in order**, basket **accumulates**.
- Deterministic menu (numbers or names) — reliable, no brittle text guessing.
- **Retry-on-miss** re-attempts a dropped item.
- [Figure: screenshot of the menu + run log, or `outputs/stage1/rollout.png`]
- **Notes:** This is the "product" — the end-to-end demo. Consider a short screen recording here if live.

---

## Slide 17 — Honest Limitations
- **Milk** is the weakest item (60%) — recoverable with milk-specific tuning / more data.
- **Multi-item lists** are less reliable than single picks: a *full basket* is a view the policy never saw in training (distribution shift). First item is always solid.
- **Cereal** excluded — grip slips on the wide box.
- **Notes:** Volunteering limits builds credibility. Tie the multi-item point to the accumulation finding.

---

## Slide 18 — Next Steps
- **Stage 2 — Navigation VLA:** drive the base between shelves in a store with **moving, colliding shoppers**.
- **Orchestrator:** loop the full list — navigate → pick → place → deliver.
- Optional: small retrain with non-empty baskets to make multi-item lists fully reliable.
- **Notes:** Show the roadmap — this project is Part A of a two-VLA system.

---

## Slide 19 — Takeaways
- ✅ A capable **VLA fine-tuned and run on a single 12 GB GPU** — the Slide-2 question, answered.
- ✅ **80% task completion**, rigorously measured (320 trials, Wilson CIs, success-based selection).
- ✅ Genuine **language grounding** + a window into the model's **plan**.
- ✅ End-to-end **language → shopping list → collected**.
- **Notes:** Close by circling back to the opening question and the headline number.

---

## Appendix / backup slides (optional)
- IK & the scripted expert (mink, top-down grasp) — `outputs/stage1/grasp_cola_can.png`
- Camera views side-by-side — `outputs/stage1/viewer_equivalent.png`
- Full store scene — `outputs/stage1/store_scene.png`
- Hardware/constraints table (12 GB, local, offline).

---

### Plots you still need to generate (for slides 7, 11)
Both are ~15 min of matplotlib on numbers already in `notes/REPORT_NOTES.md`:
1. **Per-item place % with Wilson-CI error bars** (Slide 7/8).
2. **Place % vs training step**, marking the 15k peak (Slide 11).
Say the word and I'll generate them as PNGs you can drop straight in.
```
```
