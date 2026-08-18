# Prompt to paste into Claude on your Windows laptop

> Copy everything between the lines below into Claude. Keep this file in the SAME
> folder as the `slideNN_*.png` images. Claude will generate `supermarket_vla.pptx`.

---------------------------------------------------------------------------------

You are helping me build a PowerPoint (.pptx) presentation. Use the `python-pptx`
library (install it if needed: `pip install python-pptx`). Write and run a Python
script that produces **`supermarket_vla.pptx`** in this folder.

**Global style:**
- 16:9 widescreen. Clean, academic, readable from the back of a room.
- Title ~32pt bold; body bullets ~18–20pt; keep to the bullets I give (don't invent content).
- Put the "Notes:" text in each slide's **speaker-notes** area, not on the slide.
- The image files are in THIS folder, named `slideNN_*.png`. Place each image on its
  slide, sized to fit with margins, not distorted. Slides without an image = text only.
- Use a consistent color accent (deep blue) for titles.

Build these 19 slides in order:

**Slide 1 — Title**
- Title: "Teaching a Robot to Shop: A Vision-Language-Action Model in Simulation"
- Subtitle: [my name] · [supervisor] · [date]
- One-liner: "Type what you want; the robot reads it, finds it on the shelf, and drops it in the basket."
- IMAGE: slide01_hero.png
- Notes: Set the scene — a mobile robot arm that follows plain-language requests in a supermarket sim.

**Slide 2 — The Problem**
- Retail/warehouse "pick from a list" needs perception + language + manipulation together.
- Vision-Language-Action (VLA) models promise generalist robot control — but they're large.
- Core question: can a capable VLA be fine-tuned AND run on a single 12 GB consumer GPU?
- Notes: Frame the hard task + the hardware-constrained research question; it's the thread through the talk.

**Slide 3 — What is a VLA? And what is SmolVLA?**
- VLA = Vision-Language-Action model: a single neural network that takes camera images + a language instruction and outputs robot actions directly — no separate hand-coded perception→planning→control stack.
- SmolVLA: a compact (~450M-param) open VLA from Hugging Face's LeRobot, built to run on modest hardware — the reason it fits our 12 GB GPU.
- Architecture — two parts: (1) VLM backbone (SmolVLM2-500M) encodes images + instruction into features — we FREEZE this; (2) Action expert (flow-matching transformer) takes those features + the arm state and generates a 50-step chunk of actions — we TRAIN only this.
- Runs closed-loop: observe → encode → predict 50-step plan → execute → re-plan.
- IMAGE: slide03_architecture.png
- Notes: "It reads and looks, then acts." We freeze the backbone, train only the action expert. The 50-step chunk reappears in the "seeing it think" slide.

**Slide 4 — The Road Here: from a Kitchen Benchmark to This Project**
- Started with RoboCasa — a Franka Panda + Omron mobile robot doing kitchen tasks (pick from counter → place in cabinet) in MuJoCo.
- RoboCasa's benchmark model is NVIDIA GR00T N1.6 — a multi-billion-parameter VLA: powerful, but far too large to fine-tune or deploy on a 12 GB GPU.
- I tested the deployable small models on the same kitchen tasks — SmolVLA, ACT, Pi0 — and they couldn't do it: SmolVLA 0/20, ACT 0/20.
- Decision: step back to a simpler, controlled task to make real progress → then scale the lessons up.
- IMAGE: slide04_journey.png
- Notes: The "why does the supermarket project exist" slide. Huge model works but won't deploy; small models fail on the hard benchmark → simplify, then scale.

**Slide 5 — The UR10e Pilot: What It Taught Me**
- Simplified to a MuJoCo UR10e + Robotiq gripper: pick a cube from 5 positions → basket. Compared ACT vs SmolVLA.
- Small models CAN learn a controlled task: ACT ~47% placed, SmolVLA ~53% placed.
- Three lessons — carried straight into the supermarket VLA:
  1. Fix the success metric — a bug counted "cube near basket" as success, inflating 80% → a real 40%; measure actual release.
  2. Data variety > volume — duplicated demos taught nothing; add jitter/paraphrases.
  3. Freeze the backbone — fully unfreezing vision blew the 12 GB budget and destabilized training.
- These lessons → applied in the supermarket VLA → 80%.
- (No image — text slide.)
- Notes: Methodology over model: fixing metrics and data drove the biggest gains. Bridge to the results section.

**Slide 6 — The System**
- Simulator: MuJoCo (headless GPU rendering)
- Robot: Omron LD-60 mobile base + UR10e arm + Robotiq 2-finger gripper
- Products: real textured grocery meshes (milk, cola, bread, water bottle)
- 3 cameras: wrist (eye-in-hand) · scene (shelf-facing) · basket (drop target)
- IMAGE: slide06_system.png
- Notes: Custom-built store scene (not a canned kitchen). The 3-camera setup matters later.

**Slide 7 — Approach: the Pipeline**
- Scripted expert → demos → fine-tune SmolVLA → closed-loop evaluation
- Scripted expert = a hand-coded "teacher" using inverse kinematics; generates clean pick-place demos.
- SmolVLA learns to imitate — but conditioned on language + vision.
- IMAGE: slide07_pipeline.png
- Notes: The expert only teaches; at run time the neural net drives.

**Slide 8 — Data Collection**
- 240 demonstrations (60 per item × 4 items), only successful episodes kept.
- Each frame records: 3 camera images + arm state + action + a language instruction.
- ~20 instruction paraphrases so it keys off meaning, not one sentence.
- Per-episode position jitter for variety.
- IMAGE: slide08_dataset.png
- Notes: Highlight the language variety — that's what makes it language-conditioned.

**Slide 9 — Training on 12 GB**
- Fine-tuned from `smolvla_base` (pretrained on community robot data).
- Froze the vision-language backbone, trained only the action head → fits in ~3 GB, no overfitting.
- Why it works: the backbone already sees and reads; we only teach how this robot acts.
- Batch 8 · 20k steps · ~4 h · RTX A2000 12 GB.
- (No image — text slide.)
- Notes: Answers Slide 2's question — a VLA fits 12 GB by freezing perception.

**Slide 10 — Headline Result**
- 80% task-completion rate — reads instruction → picks the named item → lands it in the basket.
- 95% confidence interval: 70–87% (Wilson) · grasp success 92.5%.
- Measured over 320 closed-loop trials — not a lucky demo.
- (Confidence interval = the range the true rate very likely sits in.)
- IMAGE: slide10_per_item_results.png
- Notes: THE slide. Say 80%, then show it's rigorously measured.

**Slide 11 — Results per Item (best checkpoint)**
- Make a 3-column table: Item | Grasp | In basket
  - Cola can | 100% | 95%
  - Water bottle | 100% | 90%
  - Loaf of bread | 100% | 75%
  - Milk carton | 70% | 60%
- Three of four items complete ≥75%; two at ≥90%. Milk is the weak point.
- All with Wilson 95% CIs, 20 trials each.
- (No image — the table is the visual.)
- Notes: Be honest about milk — acknowledging the weak spot builds credibility.

**Slide 12 — From 0% to 80%: Failure Analysis**
- First policy: 0% placement — but it grasped, carried, and lowered items to within ~2 cm of the basket.
- Diagnosis: near-misses, not broken understanding — dropped ~10 cm short + didn't cleanly stop.
- Three targeted fixes: (1) clean episode boundaries, (2) a dedicated basket camera, (3) a wider tote.
- Result: 0% → 80%.
- IMAGES: slide12a_basket_problem.png (left, "problem") and slide12b_basket_fix.png (right, "fix") — place side by side.
- Notes: A strong narrative arc — diagnosed rather than blindly retrained.

**Slide 13 — It Genuinely Understands Language**
- Same scene, change only the instruction → the arm reaches the named item.
- "milk" → left · "bread" → center · "water bottle" → right (matches each item's position).
- Not "grab the nearest" — it's reading the words.
- (No image — text slide.)
- Notes: This proves language grounding.

**Slide 14 — Which Model Do We Keep? (checkpoint selection)**
- A "checkpoint" = a saved snapshot of the model during training. We trained 20k steps, saved one every 5k → 4 versions of the same model (5k/10k/15k/20k), like save-points.
- We can only deploy one, so we had to choose which snapshot to keep.
- How we chose: ran the robot with each of the 4 and measured actual task success (320-trial eval) — NOT the training-loss curve.
- Result: 15k was best (80%); the final 20k was worse (75%) — it had over-trained.
- So we selected 15k. Taking the last checkpoint, or picking by loss, would have shipped the worse robot.
- IMAGE: slide14_checkpoint_selection.png
- Notes: Define checkpoint first (snapshot/save-point), then the choice. Punchline: "we select by what the robot actually does, not by the loss."

**Slide 15 — Seeing the VLA "Think"**
- SmolVLA plans 50 steps ahead — we surface its own predicted action plan live.
- At the first frame it has already decided to close the gripper at +16 steps — it plans the grasp before reaching the item.
- Honest interpretability: the plan is the model's real output, not a heuristic.
- Add a MONOSPACE text box on the slide showing this example plan readout:
      grip plan : ················██████████████████████████████████
      intent    : CLOSE -> grasp at +16
      (·=open  █=closed ; the model's own 50-step gripper plan at frame 0)
- (No image file — use the monospace text box above.)
- Notes: The "wow" slide. Explain the sparkline is the network's intention.

**Slide 16 — Following a Shopping List**
- Type a list from a menu → robot collects each item in order, basket accumulates.
- Deterministic menu (numbers or names) — reliable, no brittle text guessing.
- Retry-on-miss re-attempts a dropped item.
- IMAGE: slide16_shopping_list.png
- Notes: The end-to-end "product". A short screen recording works well here if presenting live.

**Slide 17 — Honest Limitations**
- Milk is the weakest item (60%) — recoverable with milk-specific tuning / more data.
- Multi-item lists are less reliable than single picks: a full basket is a view the policy never saw in training (distribution shift). First item is always solid.
- Cereal excluded — grip slips on the wide box.
- (No image — text slide.)
- Notes: Volunteering limits builds credibility.

**Slide 18 — Next Steps**
- Stage 2 — Navigation VLA: drive the base between shelves in a store with moving, colliding shoppers.
- Orchestrator: loop the full list — navigate → pick → place → deliver.
- Optional: small retrain with non-empty baskets to make multi-item lists fully reliable.
- (No image — text slide.)
- Notes: Show the roadmap — this is Part A of a two-VLA system.

**Slide 19 — Takeaways**
- A capable VLA fine-tuned and run on a single 12 GB GPU — the Slide-2 question, answered.
- 80% task completion, rigorously measured (320 trials, Wilson CIs, success-based selection).
- Genuine language grounding + a window into the model's plan.
- End-to-end: language → shopping list → collected.
- (No image — text slide.)
- Notes: Close by circling back to the opening question and the headline number.

After building it, save `supermarket_vla.pptx` and tell me if any image file was missing.
