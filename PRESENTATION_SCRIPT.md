# Presentation Script — *Mobile Manipulator Grasping of Everyday Objects Using VLA Models*
**Presenter:** Biswajeet Panda · **Target:** 15 minutes · **Deck:** `supermarket_vla (1).pptx` (25 slides)

> How to use this: the **SAY** lines are what you actually speak — read them aloud a
> couple of times, then present from memory. Timings add up to ~14.5 min, leaving a
> small buffer. Keep momentum; don't read bullets off the slide.

**Delivery tips**
- Speak to the *idea* of each slide, not every bullet/block.
- Land three numbers hard and slowly: **0% → 80%**, **92.5% grasp**, **320 trials**.
- The story arc: *big model works but won't deploy → I made a small one work → here's how, honestly measured → here's the ideal it points to.* Keep returning to it.
- Pause after each result. Silence sells confidence.

---

## ⏱ Timing plan
| Section | Slides | Time |
|---|---|---|
| Framing (problem + what's a VLA) | 1–4 | ~3:00 |
| The journey (why this project) | 5–7 | ~2:15 |
| System & method | 8–12 | ~3:15 |
| Results & rigour | 13–18 | ~3:45 |
| Demo & limits | 19–21 | ~1:40 |
| Vision & close | 22–25 | ~2:10 |
| **Total** | | **~14:45** |

*If running long, the Ideal-System slide (23) is the natural place to compress to ~30s.*

---

## Slide 1 — Title  *(~20s)*
**SAY:** "Good morning. My project is *Mobile Manipulator Grasping of Everyday Objects Using Vision-Language-Action models*. In one line: I built a robot, in simulation, that you can give a plain-English shopping list, and it finds each item on a shelf, picks it up, and drops it in its basket — driven entirely by a neural network. I'm supervised by Dr Tang and Professor Webb."

## Slide 2 — Contents  *(~15s)*
**SAY:** "I'll cover why this is hard, what a VLA is, the journey that led here, the system and how I trained it, then the results — which I've tried hard to measure honestly — and finish with limitations, the ideal system, and next steps."

## Slide 3 — The Problem  *(~45s)*
**SAY:** "Picking an item off a shelf sounds trivial, but it's actually three hard problems at once. **Perception** — the robot has to see and identify the right object among clutter. **Language** — it has to understand *which* item you asked for. And **manipulation** — it has to physically grasp it and place it. Traditionally each of these is a separate hand-built system. Vision-Language-Action models promise to do all three in a single network — but, as we'll see, they tend to be very large."
**(Point at the three cards, left to right.)**

## Slide 4 — What is a VLA? / SmolVLA  *(~60s)*
**SAY:** "So what is a VLA? It's one neural network: camera images and a text instruction go in, and robot actions come out directly — no separate perception or planning stack. The model I use is **SmolVLA** — a compact, open model, about 450 million parameters, built to run on modest hardware. That's the key: it fits a single consumer GPU. Architecturally it's two parts. First, a **vision-language backbone** — this is where the 'understanding' lives; it reads the instruction and looks at the images. I **freeze** this part. Second, an **action expert** that takes those features plus the robot's joint angles and predicts the next 50 steps of motion. I train *only* this part. It runs in a loop — observe, plan 50 steps, execute, re-plan."
**(Trace the diagram: inputs → backbone → action expert → robot.)**

## Slide 5 — The Road Here  *(~50s)*
**SAY:** "Now, why a supermarket, and why SmolVLA specifically? This is the journey. I started with **RoboCasa** — a standard benchmark of a robot doing kitchen tasks. Its reference model is NVIDIA's **GR00T N1.6**, a multi-billion-parameter model. It works — about 54% zero-shot — but it is far too large to fine-tune or deploy on the hardware I have, a 12-gigabyte GPU. So I tried the *deployable* small models — SmolVLA, ACT, Pi0 — on those same kitchen tasks. They scored **zero out of twenty**. The benchmark was simply too hard for small models out of the box. So I made a decision: step back to a controlled task, get the method right, then scale back up."

## Slide 6 — Three Robots, One Progression  *(~30s)*
**SAY:** "Visually, that's the whole thesis in one slide. The huge model on RoboCasa — undeployable. A simplified UR10e pick-and-place, where the small model finally worked — about 53%. And the supermarket system this led to, at 80%. Same recipe, scaled up."
**(Sweep left to right along the arrows.)**

## Slide 7 — The UR10e Pilot  *(~50s)*
**SAY:** "That middle step taught me the lessons that made everything else work. In the pilot, a UR10e arm picked a cube from five positions into a basket, and I compared ACT and SmolVLA — they reached about 47 and 53 percent. But the important findings weren't about the model; they were about method. One: I found a **success-metric bug** — it was counting an item held *above* the basket as a success, inflating a real 40% to a reported 80%. Two: my demonstrations were accidentally **duplicated** — no variety to learn from. And three: fully unfreezing the vision encoder blew the memory budget. Fixing the *metric and the data* mattered more than the model choice — and I carried all three lessons straight into the supermarket project."

## Slide 8 — The System  *(~40s)*
**SAY:** "Here's the system I built. It's MuJoCo — a physics simulator — with an Omron mobile base, a UR10e arm, and a two-finger gripper. The products are real textured 3D grocery meshes, not blocks. And crucially there are three cameras: a wrist camera on the hand, a scene camera facing the shelf, and a basket camera looking at the drop target."

## Slide 9 — What the Robot Sees  *(~35s)*
**SAY:** "These are those three views — exactly what the policy receives every step, alongside the instruction. The wrist view for the close-up grasp, the scene view for where things are, and the basket view so it knows where to drop. These three images plus the text are the *entire* input to the network — nothing else, no hand-coded object positions."

## Slide 10 — The Pipeline  *(~40s)*
**SAY:** "The method is a four-step recipe — you can follow the flow across the slide. I write a **scripted expert** — a hand-coded controller using inverse kinematics — that performs clean pick-and-places. I record those as **demonstrations**. I **fine-tune SmolVLA** to imitate them, freezing the backbone and training only the action head. Then I **evaluate closed-loop**. The key point: the expert only *teaches*; at run-time the neural network is fully in control."
**(Trace the four boxes left to right.)**

## Slide 11 — Data Collection  *(~40s)*
**SAY:** "I collected 240 demonstrations — 60 per item across four items — keeping only successful episodes. Every frame stores the three images, the arm state, the action, and a language instruction. And I deliberately used about 20 **paraphrases** — 'pick up the milk', 'grab the cola', and so on — so the model keys off the *meaning*, not one memorised sentence. I also jitter each item's position for variety. That variety point is the lesson from the pilot."

## Slide 12 — Training on 12 GB  *(~50s)*
**SAY:** "Training is where the hardware constraint bites. I fine-tune from the pretrained SmolVLA base. The trick that makes it fit: I **freeze the vision-language backbone** and train only the action head. This does two things — it fits in about 3 gigabytes with room to spare, and it avoids overfitting, because I'm not retraining 400 million parameters on only 240 demonstrations. The intuition: the backbone already knows how to see and read; I only need to teach it *how this particular robot moves*. Batch of 8, 20,000 steps, about four hours on an RTX A2000."

## Slide 13 — Headline Result  *(~60s)*  ⭐
**SAY:** "So — the headline. The policy completes the full task — read the instruction, pick the named item, land it in the basket — **80% of the time**. That's a 95% confidence interval of 70 to 87. Grasping alone is 92.5%. And this isn't one lucky demo — it's measured over **320 closed-loop trials**. For anyone less familiar: a confidence interval is just the range the true rate very likely sits in, given the number of trials — so I'm not over-reading a small sample."
**(Pause. Let 80% land.)**

## Slide 14 — Results per Item  *(~35s)*
**SAY:** "Broken down by item: cola and water are 95 and 90 percent, bread 75, and milk is the weak one at 60. Three of the four are at or above 75%. Grasping is essentially solved for three of them at 100%. Every number here has a Wilson confidence interval over 20 trials — and I'll be honest about milk in a moment."

## Slide 15 — From 0% to 80%  *(~60s)*
**SAY:** "This is the result I'm most proud of, because of *how* I got it. My first policy placed **zero percent** — but when I traced it, it was actually grasping, carrying, and lowering the item to within two centimetres of the basket, then dropping it about ten centimetres short. So it wasn't a broken policy — it *understood the task*; it just had a precision-and-timing problem at the very end. That diagnosis pointed to three specific fixes: clean episode boundaries in the data, a dedicated basket camera so it can see the target, and a slightly wider basket. That took it from zero to 80. The point is I *diagnosed* rather than blindly retrained."

## Slide 16 — It Understands Language  *(~40s)*
**SAY:** "A fair challenge is: is the language actually doing anything, or is it just going to one memorised spot? Here's the test. Same scene, I change *only* the instruction word. Say 'milk' and it reaches left, where the milk is; say 'bread', it goes centre; say 'water bottle', it reaches right. The target moves with the *word*. That's genuine language grounding, not a fixed trajectory."

## Slide 17 — Which Model Do We Keep?  *(~55s)*
**SAY:** "A methodological point I want to highlight. During training, the model is saved every 5,000 steps — so I end up with four snapshots, called checkpoints. You can only deploy one. The lazy default is to take the last one. Instead, I ran the actual robot with all four and measured task success — not training loss. And it mattered: the 15,000-step checkpoint was best at 80%, while the *final* 20,000-step one was actually **worse**, at 75% — it had started to over-train. If I'd picked by the loss curve, I'd have shipped the worse robot. So I select by what the robot actually does."

## Slide 18 — Seeing the VLA "Think"  *(~45s)*
**SAY:** "One thing I find genuinely interesting: I can surface the model's own plan. Remember it predicts 50 steps ahead. At the very first frame — arm still at home, nowhere near the item — its plan *already* shows the gripper closing at step 16. In other words, it has decided *to grasp* before it has even reached the object. And this isn't me interpreting — the bar is literally the network's output. It's a small window into what the policy is planning."

## Slide 19 — Following a Shopping List  *(~40s)*
**SAY:** "Finally, I tie it together with an orchestrator. You pick items from a menu, and the robot collects them in order — the basket accumulating as it goes. The menu keeps the input reliable, and if it misses an item, it retries. So it's a genuine end-to-end system: language in, collected items out."

## Slide 20 — Closed-Loop Rollout  *(~20s)*
**SAY:** "And here's what that looks like frame by frame — the policy reaching in and collecting each item: cola, bread, milk, water. Every frame here is the network in control."
**(Let the image do the work; move on.)**

## Slide 21 — Honest Limitations  *(~40s)*
**SAY:** "I want to be upfront about limits. Milk is my weakest item at 60% — recoverable with a bit of item-specific tuning. Multi-item lists are less reliable than single picks, because once the basket has something in it, that's a scene the policy never saw in training — a distribution shift. And I've excluded cereal for now because its wide box slips in the gripper. None of these are hidden — they're the honest edges of the result."

## Slide 22 — Next Steps  *(~35s)*
**SAY:** "This is actually only the *manipulation* half of a two-part vision. The next stage is a **navigation VLA** — driving the base between shelves in a store with moving, colliding shoppers — and an **orchestrator** that loops the full list: navigate, pick, place, deliver. And a small retrain with non-empty baskets would close that multi-item robustness gap."
**(Point at the three blocks.)**

## Slide 23 — If There Were No Constraints: the Ideal System  *(~45s)*  🆕
**SAY:** "A natural question is: if you had *unlimited* compute, what would you build? And the interesting answer is — not simply a bigger model. Even unconstrained, some limits are *physical*: a giant model can't run a fast control loop safely around people. So the ideal is a **dual-system** design — a large reasoner that plans the list and reads the crowd, paired with a fast reflex policy for real-time control. On top of that, four changes: control the base and arm as **one whole body**; add **tactile sensing** — because touch, not vision, is what finally fixes grasping; train on **far more diverse data** so it generalises to any store; and use **reinforcement learning** to push past what demonstrations can teach. The nice framing is that my current system is the *deployable subset* of that ideal — the same recipe, shrunk to a single GPU."
**(If short on time, cut to: "Unconstrained, the answer isn't a bigger model — it's a dual-system design, tactile sensing to fix grasping, and far more diverse data. My system is the deployable subset of that ideal.")**

## Slide 24 — Takeaways  *(~40s)*
**SAY:** "To wrap up. I showed a capable VLA can be fine-tuned and run on a single 12-gigabyte GPU. It reaches 80% task completion, measured properly — 320 trials, confidence intervals, and checkpoint selection by real success. It genuinely grounds language, and I can even see it plan. And it's a complete pipeline: from a typed list to collected items."
**(Point across the four blocks.)**

## Slide 25 — Thank You  *(~10s)*
**SAY:** "That's my project — thank you very much for listening, and I'm very happy to take any questions."

---

# 🎯 Supervisor Q&A — anticipated questions & strong answers

### On the model & training internals (likely straight off slides 4, 12, 23)
**Q: If you freeze the vision part, how does it still work?**
A: 'Frozen' means I don't *change* those weights — not that it's off. Every step it still runs a full forward pass: it looks at the images, reads the instruction, and outputs features. It's good at that because it was **pretrained on massive image-and-text data** before my project even starts. I only train the small action expert on top, which maps those features to *this* robot's movements. The eyes and language understanding are already trained; I teach the motor cortex.

**Q: Why is 240 demonstrations enough — isn't that tiny?**
A: Because of that division of labour. The model arrives already knowing what the objects are and what the words mean, from pretraining. My demos only teach the *motor mapping* for this specific robot — a much smaller learning problem. That's exactly why so few demos suffice and why it fits 12 GB.

**Q: What does 'batch of 8' mean?**
A: During training I show the model 8 examples at once, average the error across them into one update, and step. Averaging over 8 gives a stabler gradient than one at a time, and the GPU does the 8 in parallel. 8 was the largest that fit in 12 GB; 20,000 steps means 20,000 such updates.

**Q: With no constraints, wouldn't you just use the biggest model?**
A: No — and that's the point of the ideal-system slide. Even unconstrained, a giant model can't run a fast, safe control loop around people — that's a *physical* limit, not a compute one. So the ideal is a dual-system: a big reasoner for planning plus a fast reflex policy for control, with tactile sensing to fix grasping and diverse data to generalise. Scale alone isn't the answer.

**Q: Ideally, which model for this task?**
A: Under my constraints, SmolVLA is the correct choice and the results show it's sufficient. If compute opened up, I'd move to **π0.5** — it's designed for exactly this mobile-manipulation, list-driven setting, it generalises to new environments, and it's already in the same framework (LeRobot), so my pipeline carries over.

### On model choice
**Q: Why SmolVLA and not a bigger model like GR00T or OpenVLA?**
A: Deployability. The bigger models need far more than 12 GB even to *fine-tune*, and a core aim was to see whether a capable VLA can run on one consumer GPU. GR00T works on RoboCasa but I can't fine-tune or deploy it locally. SmolVLA is the largest capable VLA that fits — and it's enough for this task.

**Q: GR00T got 54% and your small models got 0% — is that a fair comparison?**
A: It's not a like-for-like *capability* claim, and I don't present it as one. GR00T is pretrained on large robot datasets close to that domain, so 54% zero-shot is expected. My point is narrower and about *deployability*: on the hardware available, the models I *can* run scored zero — which is exactly why I moved to a controlled task rather than claiming small models are as good.

### On results and rigour
**Q: 80% — is that a big enough sample to trust?**
A: The headline is 320 closed-loop trials; per item it's 20 each, and I report Wilson 95% confidence intervals throughout. So milk at 60% has a wide interval and I treat it cautiously; cola at 95% is tight. I've tried not to over-read small samples — that's why the CIs are on every figure.

**Q: Selecting the checkpoint by evaluation success — isn't that overfitting to your test set?**
A: Fair challenge. I select by *closed-loop success rate* rather than loss, and each trial uses fresh seeds, instruction paraphrases, and positional jitter — so it's not the exact training episodes. The honest caveat is that a fully separate locked test set would be even cleaner; with more time I'd add one that's never used for selection.

**Q: Is the '0% to 80%' real, or did you just fix a bug and call it a result?**
A: Both, and I'm explicit about that. The 0% was a genuine policy that near-missed every time. The improvement came from a *diagnosis*: I traced the failure to final-drop precision and observability, and each fix targets that. The value isn't the number jump — it's identifying the actual failure mode instead of blindly retraining.

**Q: Flow-matching is stochastic — how repeatable are these numbers?**
A: That's why I run 320 trials rather than a handful — the rates average over the sampling noise. Run-to-run there's variance, especially on weak items, which is exactly what the confidence intervals capture. I flag it as a known limitation.

### On the language / "understanding"
**Q: How do you know it's using the language and not memorising positions?**
A: The controlled test on the grounding slide: I hold the scene fixed and change only the instruction word, and the arm goes to the *named* item — left for milk, centre for bread, right for water. A fixed trajectory couldn't do that. The graded per-item success rates are also the signature of a learned controller, not a lookup.

**Q: Isn't this just imitation learning — what does the VLA add over ACT?**
A: ACT can't take a free-form instruction — it's trained per task from scratch, no language. The VLA adds language conditioning (one policy, many item instructions) and a pretrained visual prior, which is what let it work on my small dataset where a from-scratch policy would need far more data.

### On limitations, realism, and scope
**Q: This is simulation — will it transfer to a real robot?**
A: I don't claim sim-to-real yet; it's out of scope for this stage. But the pipeline is deliberately real-robot-shaped — it uses LeRobot and real hardware models (UR10e, Robotiq), so the same code path could fine-tune on real demonstrations. A real SO-101 arm is a candidate next step.

**Q: Why is milk the weak item?**
A: It sits at the far edge of the reliable grasp zone and its geometry is less forgiving, so its grasp rate drops to 70%. It placed higher at earlier checkpoints, so it's recoverable with milk-specific tuning or a few more milk demos — I'd rather report it honestly than hide it.

**Q: The multi-item list is less reliable — why, and how would you fix it?**
A: Distribution shift: the policy trained with an *empty* basket every episode, so once an item is in there the camera view is something it never saw. The clean fix is a small batch of demos with varied basket contents — that teaches it to ignore what's already collected.

**Q: What's the actual contribution — you're using an existing model?**
A: The contribution is the *engineering and methodology*, not a new architecture: a reproducible pipeline that gets a VLA working under a hard 12 GB constraint, a rigorous evaluation (CIs, success-based checkpoint selection), and a documented failure-diagnosis that turned 0% into 80% — plus the finding that fixing metrics and data variety matters more than model size at this scale.

**Q: Why MuJoCo and not RoboCasa or Isaac Sim?**
A: RoboCasa is kitchen-only and didn't respond well to SmolVLA locally; I needed a custom supermarket scene. MuJoCo gives accurate physics, is lightweight, and is the same engine family the benchmarks use — the practical choice for a controlled, reproducible environment on my hardware.

**Q: How will the navigation VLA handle moving shoppers in real time?**
A: That's the hard part of the next stage, and it's why the ideal-system slide stresses a fast reflex policy. The plan mirrors this one — a scripted reactive navigator generates avoidance demonstrations, then a second VLA learns language-guided navigation, with an orchestrator sequencing navigate-pick-place. Real-time reactivity and the execution-horizon trade-off are the open questions.

---

### Two-sentence fallback if you blank on any question
"That's a good question — my honest answer is I'd need to check the exact number, but the *direction* is [X]. What I can say confidently is [the thing you do know]." — Then bridge to a result you're sure of. **Never bluff a number.**
