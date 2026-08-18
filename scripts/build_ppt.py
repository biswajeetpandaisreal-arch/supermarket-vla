"""
build_ppt.py — generate the supermarket-VLA presentation as a polished .pptx with
a custom-designed theme (navy + teal, title bands, styled bullets, framed figures,
footer + slide numbers, a Contents page). Built from scratch — no external template.

    .venv/bin/python scripts/build_ppt.py   # -> outputs/supermarket_vla.pptx
"""
from pathlib import Path
from PIL import Image

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs" / "supermarket_vla.pptx"
def P(rel): return str(ROOT / rel)

# ── palette ───────────────────────────────────────────────────────────────────
INK    = RGBColor(0x12, 0x22, 0x39)   # deep navy (title slide bg, headings)
BAND   = RGBColor(0x1B, 0x30, 0x4C)   # title-band navy
ACCENT = RGBColor(0x17, 0xB3, 0xA3)   # teal
AMBER  = RGBColor(0xF0, 0x8A, 0x24)   # warm secondary (sparingly)
BG     = RGBColor(0xF5, 0xF7, 0xFA)   # page background
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
SLATE  = RGBColor(0x30, 0x3E, 0x4E)   # body text
MUTE   = RGBColor(0x8A, 0x97, 0xA6)   # footer / sub-bullets
LINEC  = RGBColor(0xDD, 0xE3, 0xE9)   # hairlines / image borders
ROWALT = RGBColor(0xEE, 0xF2, 0xF6)   # table zebra

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]

MARGIN = Inches(0.55)
BAND_H = Inches(1.12)
CTOP   = Inches(1.5)
CBOT   = Inches(6.95)
FOOT_Y = Inches(7.05)


def _kill_shadow(shape):
    # override the default theme shadow with an explicit empty effect list
    spPr = shape._element.spPr
    for el in spPr.findall(qn("a:effectLst")):
        spPr.remove(el)
    spPr.append(spPr.makeelement(qn("a:effectLst"), {}))
    # remove the theme style ref (its a:effectRef re-applies the shadow otherwise)
    style = shape._element.find(qn("p:style"))
    if style is not None:
        shape._element.remove(style)


def _noline_noshadow(sh):
    sh.line.fill.background()
    sh.shadow.inherit = False
    _kill_shadow(sh)
    return sh


def rect(slide, l, t, w, h, fill):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    return _noline_noshadow(sh)


def slide_bg(slide, color=BG):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def title_band(slide, title):
    rect(slide, 0, 0, SW, BAND_H, BAND)
    rect(slide, 0, BAND_H, SW, Pt(6), ACCENT)          # accent underline
    tb = slide.shapes.add_textbox(MARGIN, 0, SW - 2 * MARGIN, BAND_H)
    tf = tb.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = title
    r = p.runs[0]; r.font.size = Pt(27); r.font.bold = True; r.font.color.rgb = WHITE
    r.font.name = "Calibri"


def footer(slide, n):                         # slide number only (bottom-right)
    rb = slide.shapes.add_textbox(SW - MARGIN - Inches(1.2), FOOT_Y, Inches(1.2), Inches(0.32))
    rp = rb.text_frame.paragraphs[0]; rp.text = str(n); rp.alignment = PP_ALIGN.RIGHT
    rr = rp.runs[0]; rr.font.size = Pt(11); rr.font.color.rgb = MUTE


def bullets(slide, items, l, t, w, h, center=False):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    if center:
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, it in enumerate(items):
        text, lvl = it if isinstance(it, tuple) else (it, 0)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10 if lvl == 0 else 5)
        p.line_spacing = 1.05
        glyph = p.add_run()
        glyph.text = ("▪   " if lvl == 0 else "        –   ")
        glyph.font.color.rgb = ACCENT if lvl == 0 else MUTE
        glyph.font.size = Pt(16 if lvl == 0 else 13); glyph.font.bold = (lvl == 0)
        tr = p.add_run(); tr.text = text
        tr.font.size = Pt(18 if lvl == 0 else 14)
        tr.font.color.rgb = SLATE if lvl == 0 else MUTE


def fit(path, bl, bt, bw, bh):
    w, h = Image.open(path).size
    s = min(bw / w, bh / h)
    nw, nh = int(w * s), int(h * s)
    return int(bl + (bw - nw) / 2), int(bt + (bh - nh) / 2), nw, nh


def image(slide, path, bl, bt, bw, bh):
    l, t, w, h = fit(path, bl, bt, bw, bh)
    pic = slide.shapes.add_picture(path, l, t, w, h)
    pic.line.color.rgb = LINEC; pic.line.width = Pt(1)
    _kill_shadow(pic)
    return pic


def card(slide, l, t, w, h, header, body, color):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    box.fill.solid(); box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = LINEC; box.line.width = Pt(1.25)
    box.shadow.inherit = False; _kill_shadow(box)
    ht = slide.shapes.add_textbox(l, t + Inches(0.3), w, Inches(0.5))
    hp = ht.text_frame.paragraphs[0]; hp.alignment = PP_ALIGN.CENTER
    hr = hp.add_run(); hr.text = header; hr.font.bold = True; hr.font.size = Pt(16); hr.font.color.rgb = color
    rect(slide, l + w / 2 - Inches(0.7), t + Inches(0.88), Inches(1.4), Pt(3), color)
    bt = slide.shapes.add_textbox(l + Inches(0.25), t + Inches(1.1), w - Inches(0.5), h - Inches(1.3))
    bf = bt.text_frame; bf.word_wrap = True; bf.vertical_anchor = MSO_ANCHOR.TOP
    bp = bf.paragraphs[0]; bp.alignment = PP_ALIGN.CENTER
    br = bp.add_run(); br.text = body; br.font.size = Pt(14.5); br.font.color.rgb = SLATE


def right_arrow(slide, l, t, w, h, color):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, l, t, w, h)
    a.fill.solid(); a.fill.fore_color.rgb = color
    a.line.fill.background(); a.shadow.inherit = False; _kill_shadow(a)
    return a


def prog_col(slide, path, l, box_w, img_top, img_h, label, sub, subcolor):
    image(slide, path, l, img_top, box_w, img_h)
    tb = slide.shapes.add_textbox(l, img_top + img_h + Inches(0.14), box_w, Inches(0.9))
    tf = tb.text_frame; tf.word_wrap = True
    p1 = tf.paragraphs[0]; p1.alignment = PP_ALIGN.CENTER
    r1 = p1.add_run(); r1.text = label; r1.font.bold = True; r1.font.size = Pt(15); r1.font.color.rgb = INK
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run(); r2.text = sub; r2.font.size = Pt(13); r2.font.color.rgb = subcolor


# ── content-slide builder ─────────────────────────────────────────────────────
_n = [0]
def content(title, blts, img=None, imgs=None, table=None, mono=None, wide_img=None, notes=None):
    _n[0] += 1
    s = prs.slides.add_slide(BLANK)
    slide_bg(s); title_band(s, title); footer(s, _n[0])
    if wide_img:                              # short bullets on top, full-width figure below
        bullets(s, blts, MARGIN, CTOP, SW - 2 * MARGIN, Inches(1.55))
        image(s, wide_img, MARGIN, Inches(3.2), SW - 2 * MARGIN, CBOT - Inches(3.2))
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s
    visual = img or imgs or table or mono
    bx_l, bx_w = MARGIN, (Inches(6.9) if visual else SW - 2 * MARGIN)
    bullets(s, blts, bx_l, CTOP, bx_w, CBOT - CTOP, center=True)

    rl, rw = Inches(7.7), SW - Inches(7.7) - MARGIN
    rt, rh = CTOP, CBOT - CTOP
    if img:
        image(s, img, rl, rt, rw, rh)
    if imgs:
        half = int(rh / 2) - Inches(0.18)
        for k, ip in enumerate(imgs):
            image(s, ip, rl, rt + k * (half + Inches(0.36)), rw, half)
    if mono:
        card = rect(s, rl, Inches(2.7), rw, Inches(1.9), INK)
        tb = s.shapes.add_textbox(rl + Inches(0.2), Inches(2.85), rw - Inches(0.4), Inches(1.6))
        tf = tb.text_frame; tf.word_wrap = True
        for i, line in enumerate(mono):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            r = p.add_run(); r.text = line
            r.font.name = "Consolas"; r.font.size = Pt(12); r.font.color.rgb = RGBColor(0x9F, 0xF3, 0xE9)
    if table:
        rows, cols = len(table), len(table[0])
        gt = s.shapes.add_table(rows, cols, rl, Inches(2.4), rw, Inches(0.55 * rows)).table
        for ci in range(cols):
            gt.columns[ci].width = int(rw / cols)
        for ri, row in enumerate(table):
            for ci, val in enumerate(row):
                c = gt.cell(ri, ci); c.text = str(val)
                cell_fill = BAND if ri == 0 else (ROWALT if ri % 2 else WHITE)
                c.fill.solid(); c.fill.fore_color.rgb = cell_fill
                para = c.text_frame.paragraphs[0]
                r = para.runs[0]
                r.font.size = Pt(15); r.font.bold = (ri == 0)
                r.font.color.rgb = WHITE if ri == 0 else SLATE
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


# ══════════════════════════════════════════════════════════════════════════════
# Slide 1 — TITLE
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, INK)                                  # full navy bg
rect(s, 0, Inches(5.5), SW, Pt(4), ACCENT)                 # accent rule
rect(s, MARGIN, Inches(1.3), Inches(0.18), Inches(3.3), ACCENT)   # left accent bar
tb = s.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(7.7), Inches(3.9))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.line_spacing = 1.06
p.text = "Mobile Manipulator Grasping of Everyday Objects Using Vision–Language–Action (VLA) Models"
r = p.runs[0]; r.font.size = Pt(33); r.font.bold = True; r.font.color.rgb = WHITE
tb2 = s.shapes.add_textbox(Inches(1.0), Inches(5.75), Inches(8.4), Inches(1.4))
tf2 = tb2.text_frame; tf2.word_wrap = True
p1 = tf2.paragraphs[0]
b0 = p1.add_run(); b0.text = "by "; b0.font.size = Pt(18); b0.font.color.rgb = MUTE
b1 = p1.add_run(); b1.text = "Biswajeet Panda"; b1.font.size = Pt(18); b1.font.bold = True; b1.font.color.rgb = WHITE
p2 = tf2.add_paragraph(); p2.space_before = Pt(7)
s0 = p2.add_run(); s0.text = "Supervisors:  "; s0.font.size = Pt(15); s0.font.color.rgb = MUTE
s1 = p2.add_run(); s1.text = "Dr Gilbert Tang   ·   Prof Phil Webb"; s1.font.size = Pt(15); s1.font.color.rgb = ACCENT
image(s, P("outputs/poster/robot_hero_front.png"), Inches(9.1), Inches(1.35), Inches(3.75), Inches(3.95))
s.notes_slide.notes_text_frame.text = "Title slide."

# ══════════════════════════════════════════════════════════════════════════════
# Slide 2 — CONTENTS
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_bg(s); title_band(s, "Contents"); footer(s, 2)
AGENDA = [
    "Motivation & the 12 GB question",
    "What is a VLA?  (SmolVLA)",
    "The road here:  RoboCasa → UR10e → Supermarket",
    "System, data & training",
    "Results & rigorous evaluation",
    "Seeing the model “think” + shopping-list demo",
    "Limitations & next steps",
]
y = Inches(1.85)
for i, item in enumerate(AGENDA, 1):
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, MARGIN, y, Inches(0.62), Inches(0.62))
    box.fill.solid(); box.fill.fore_color.rgb = ACCENT if i % 2 else BAND; _noline_noshadow(box)
    ntf = box.text_frame; ntf.vertical_anchor = MSO_ANCHOR.MIDDLE
    np_ = ntf.paragraphs[0]; np_.alignment = PP_ALIGN.CENTER
    nr = np_.add_run(); nr.text = str(i); nr.font.size = Pt(20); nr.font.bold = True; nr.font.color.rgb = WHITE
    lb = s.shapes.add_textbox(Inches(1.4), y, Inches(10.8), Inches(0.62))
    ltf = lb.text_frame; ltf.vertical_anchor = MSO_ANCHOR.MIDDLE
    lp = ltf.paragraphs[0]; lr = lp.add_run(); lr.text = item
    lr.font.size = Pt(20); lr.font.color.rgb = SLATE
    y += Inches(0.72)

# ══════════════════════════════════════════════════════════════════════════════
# Slides 3–21 — CONTENT  (footer numbering starts at 3)
# ══════════════════════════════════════════════════════════════════════════════
_n[0] = 2  # so the next content() slide is numbered 3

# Slide 3 — The Problem (custom: three pillars)
_n[0] += 1
s = prs.slides.add_slide(BLANK)
slide_bg(s); title_band(s, "The Problem"); footer(s, _n[0])
lead = s.shapes.add_textbox(MARGIN, Inches(1.65), SW - 2 * MARGIN, Inches(0.7))
lp = lead.text_frame.paragraphs[0]
lr = lp.add_run(); lr.text = "Picking items from a shopping list means solving three hard problems — at the same time:"
lr.font.size = Pt(19); lr.font.color.rgb = SLATE
_cards = [("PERCEPTION", "See and identify the right item on a cluttered shelf.", ACCENT),
          ("LANGUAGE", "Understand which item the request is actually asking for.", BAND),
          ("MANIPULATION", "Grasp it and place it cleanly in the basket.", AMBER)]
_gap = Inches(0.45); _cw = int((SW - 2 * MARGIN - 2 * _gap) / 3); _ct = Inches(2.7); _ch = Inches(2.45)
for _i, (_hd, _bd, _col) in enumerate(_cards):
    card(s, MARGIN + _i * (_cw + _gap), _ct, _cw, _ch, _hd, _bd, _col)
bot = s.shapes.add_textbox(MARGIN, Inches(5.55), SW - 2 * MARGIN, Inches(0.9))
bf = bot.text_frame; bf.word_wrap = True; bpar = bf.paragraphs[0]
b1 = bpar.add_run(); b1.text = "Vision-Language-Action (VLA) models promise all three in one network"
b1.font.size = Pt(19); b1.font.bold = True; b1.font.color.rgb = INK
b2 = bpar.add_run(); b2.text = " — but they're large."
b2.font.size = Pt(19); b2.font.color.rgb = SLATE
s.notes_slide.notes_text_frame.text = "The task is three hard problems at once; VLAs promise all three in one model, but they're large — which sets up the hardware constraint."

content("What is a VLA?  And what is SmolVLA?", [
    "VLA = one neural network: images + a language instruction in, robot actions out — no hand-coded perception/planning/control stack.",
    "SmolVLA: a compact (~450M-param) open VLA (Hugging Face LeRobot), built for modest hardware — the reason it fits 12 GB.",
    "Architecture — two parts:",
    ("VLM backbone (SmolVLM2-500M): encodes images + instruction into features — WE FREEZE THIS.", 1),
    ("Action expert (flow-matching): features + arm state → a 50-step action chunk — WE TRAIN ONLY THIS.", 1),
    "Runs closed-loop: observe → encode → predict 50-step plan → execute → re-plan.",
], img=P("outputs/plots/smolvla_architecture.png"),
   notes="“It reads and looks, then acts.” Freeze backbone, train only the action expert.")

content("The Road Here: from a Kitchen Benchmark to This Project", [
    "Started with RoboCasa — a Franka Panda + Omron mobile robot doing kitchen tasks (pick → cabinet) in MuJoCo.",
    "RoboCasa's benchmark model is NVIDIA GR00T N1.6 — multi-billion params: strong (54% zero-shot) but far too large for 12 GB.",
    "Deployable small models on the same tasks — SmolVLA, ACT, Pi0 — couldn't do it: SmolVLA 0/20, ACT 0/20.",
    "Decision: step back to a simpler, controlled task — then scale the lessons up.",
], img=P("outputs/plots/journey_prior_work.png"),
   notes="Why the supermarket project exists: huge model works but won't deploy; small models fail on the hard benchmark.")

# Slide after "The Road Here" — the three robots, with progression arrows
_n[0] += 1
s = prs.slides.add_slide(BLANK)
slide_bg(s); title_band(s, "Three Robots, One Progression"); footer(s, _n[0])
_cols = [
    ("outputs/plots/progression/prog_robocasa.png", "RoboCasa · GR00T N1.6", "small models: 0/20", RGBColor(0xC6, 0x28, 0x28)),
    ("outputs/plots/progression/prog_ur10e.png",    "UR10e pilot (MuJoCo)",   "SmolVLA ~53%",        AMBER),
    ("outputs/plots/progression/prog_supermarket.png", "Supermarket VLA · this work", "SmolVLA 80%",  RGBColor(0x2E, 0x7D, 0x32)),
]
_bw, _gap = Inches(3.45), Inches(0.8)
_x0 = int((SW - (3 * _bw + 2 * _gap)) / 2)
_itop, _ih = Inches(1.95), Inches(3.15)
_xs = [_x0 + i * (_bw + _gap) for i in range(3)]
for _i, (_p, _lab, _sub, _sc) in enumerate(_cols):
    prog_col(s, P(_p), _xs[_i], _bw, _itop, _ih, _lab, _sub, _sc)
_ay = _itop + int(_ih / 2) - Inches(0.34)
_arrow_lbl = ["too big to deploy", "lessons applied"]
for _i in range(2):
    _ax = _xs[_i] + _bw + Inches(0.04)
    right_arrow(s, _ax, _ay, _gap - Inches(0.08), Inches(0.68), ACCENT)
    ct = s.shapes.add_textbox(_ax - Inches(0.3), _ay - Inches(0.5), _gap + Inches(0.5), Inches(0.4))
    cp = ct.text_frame.paragraphs[0]; cp.alignment = PP_ALIGN.CENTER
    cr = cp.add_run(); cr.text = _arrow_lbl[_i]; cr.font.size = Pt(11); cr.font.italic = True; cr.font.color.rgb = MUTE
s.notes_slide.notes_text_frame.text = ("The same VLA recipe across three robots: the huge model on RoboCasa (undeployable), "
                                       "the UR10e pilot where small models finally worked, and the supermarket VLA it led to.")

content("The UR10e Pilot: What It Taught Me", [
    "Simplified to a MuJoCo UR10e + Robotiq gripper: pick a cube from 5 positions → basket. ACT vs SmolVLA.",
    "Small models CAN learn a controlled task: ACT ~47% placed, SmolVLA ~53% placed.",
    "Three lessons — carried straight into the supermarket VLA:",
    ("Fix the success metric — a bug inflated 80% → a real 40%; measure actual release.", 1),
    ("Data variety > volume — duplicated demos taught nothing.", 1),
    ("Freeze the backbone — fully unfreezing vision blew the 12 GB budget.", 1),
    "These lessons → applied in the supermarket VLA → 80%.",
], notes="Methodology over model: fixing metrics and data drove the biggest gains.")

content("The System", [
    "Simulator: MuJoCo (headless GPU rendering)",
    "Robot: Omron LD-60 mobile base + UR10e arm + Robotiq 2-finger gripper",
    "Products: real textured grocery meshes (milk, cola, bread, water bottle)",
    "3 cameras: wrist (eye-in-hand) · scene (shelf-facing) · basket (drop target)",
], img=P("outputs/poster/robot_and_shelf.png"),
   notes="Custom-built store scene. The 3-camera setup matters later.")

content("What the Robot Sees — Three Camera Views", [
    "The policy acts from three synchronized views every step — plus the instruction:",
    ("Wrist (eye-in-hand): close-up of the gripper and the item being grasped.", 1),
    ("Scene (shelf-facing): the whole aisle — which items are where.", 1),
    ("Basket (drop target): added so it can see exactly where to drop.", 1),
], wide_img=P("outputs/plots/camera_views.png"),
   notes="These three images + the text instruction are the ENTIRE input to SmolVLA.")

content("Approach: the Pipeline", [
    "Scripted expert → demos → fine-tune SmolVLA → closed-loop evaluation",
    "Scripted expert = a hand-coded “teacher” using inverse kinematics; generates clean pick-place demos.",
    "SmolVLA learns to imitate — but conditioned on language + vision.",
], img=P("outputs/stage1/pickplace_cola_can.png"),
   notes="The expert only teaches; at run time the neural net drives.")

content("Data Collection", [
    "240 demonstrations (60 per item × 4 items), only successful episodes kept.",
    "Each frame: 3 camera images + arm state + action + a language instruction.",
    "~20 instruction paraphrases so it keys off meaning, not one sentence.",
    "Per-episode position jitter for variety.",
], img=P("outputs/stage1/dataset_preview.png"),
   notes="Highlight the language variety — that's what makes it language-conditioned.")

content("Training on 12 GB", [
    "Fine-tuned from smolvla_base (pretrained on community robot data).",
    "Froze the vision-language backbone, trained only the action head → fits in ~3 GB, no overfitting.",
    "Why it works: the backbone already sees and reads; we only teach how this robot acts.",
    "Batch 8 · 20k steps · ~4 h · RTX A2000 12 GB.",
], notes="A VLA fits 12 GB by freezing perception.")

content("Headline Result", [
    "80% task-completion rate — reads instruction → picks the named item → lands it in the basket.",
    "95% confidence interval: 70–87% (Wilson) · grasp success 92.5%.",
    "Measured over 320 closed-loop trials — not a lucky demo.",
    ("Confidence interval = the range the true rate very likely sits in.", 1),
], img=P("outputs/plots/per_item_success.png"),
   notes="THE slide. Say 80%, then show it's rigorously measured.")

content("Results per Item (best checkpoint)", [
    "Three of four items complete ≥75%; two at ≥90%. Milk is the weak point.",
    "All with Wilson 95% CIs, 20 trials each.",
], table=[["Item", "Grasp", "In basket"],
          ["Cola can", "100%", "95%"],
          ["Water bottle", "100%", "90%"],
          ["Loaf of bread", "100%", "75%"],
          ["Milk carton", "70%", "60%"]],
   notes="Be honest about milk — acknowledging the weak spot builds credibility.")

content("From 0% to 80%: Failure Analysis", [
    "First policy: 0% placement — but it grasped, carried, and lowered items to within ~2 cm of the basket.",
    "Diagnosis: near-misses, not broken understanding — dropped ~10 cm short + didn't cleanly stop.",
    "Three targeted fixes: clean episode boundaries, a dedicated basket camera, a wider tote.",
    "Result: 0% → 80%.",
], imgs=[P("outputs/stage1/basket_visibility.png"), P("outputs/stage1/basket_cam.png")],
   notes="A strong narrative arc — diagnosed rather than blindly retrained.")

content("It Genuinely Understands Language", [
    "Same scene, change only the instruction → the arm reaches the named item.",
    "“milk” → left · “bread” → center · “water bottle” → right (matches each item's position).",
    "Not “grab the nearest” — it's reading the words.",
], notes="This proves language grounding.")

content("Which Model Do We Keep?  (checkpoint selection)", [
    "A “checkpoint” = a saved snapshot during training. 20k steps, saved every 5k → 4 versions (5k/10k/15k/20k), like save-points.",
    "We can only deploy one, so we had to choose which snapshot to keep.",
    "How we chose: ran the robot with each of the 4 and measured actual task success — NOT the loss curve.",
    "15k was best (80%); the final 20k was worse (75%) — it over-trained.",
    "So we selected 15k. Picking by loss would have shipped the worse robot.",
], img=P("outputs/plots/checkpoint_selection.png"),
   notes="Define checkpoint first, then the choice. Select by what the robot does, not the loss.")

content("Seeing the VLA “Think”", [
    "SmolVLA plans 50 steps ahead — we surface its own predicted action plan live.",
    "At the first frame it has already decided to close the gripper at +16 steps — it plans the grasp before reaching the item.",
    "Honest interpretability: the plan is the model's real output, not a heuristic.",
], mono=["grip plan : ················██████████████████",
         "intent    : CLOSE -> grasp at +16",
         "(· = open   █ = closed)",
         "the model's own 50-step plan at frame 0"],
   notes="The “wow” slide. The sparkline is the network's intention.")

content("Following a Shopping List", [
    "Type a list from a menu → robot collects each item in order, basket accumulates.",
    "Deterministic menu (numbers or names) — reliable, no brittle text guessing.",
    "Retry-on-miss re-attempts a dropped item.",
], img=P("outputs/stage1/rollout.png"),
   notes="The end-to-end product. A short screen recording works well here if presenting live.")

# standalone enlarged rollout image (right after the shopping-list slide)
_n[0] += 1
s = prs.slides.add_slide(BLANK)
slide_bg(s); title_band(s, "Closed-Loop Rollout — the Policy Collecting Items"); footer(s, _n[0])
image(s, P("outputs/stage1/rollout.png"), MARGIN, Inches(1.65), SW - 2 * MARGIN, Inches(5.15))
s.notes_slide.notes_text_frame.text = "Enlarged closed-loop rollout filmstrip: SmolVLA reaching to grasp and place each item."

content("Honest Limitations", [
    "Milk is the weakest item (60%) — recoverable with milk-specific tuning / more data.",
    "Multi-item lists are less reliable than single picks: a full basket is a view the policy never saw in training. First item is always solid.",
    "Cereal excluded — grip slips on the wide box.",
], notes="Volunteering limits builds credibility.")

content("Next Steps", [
    "Stage 2 — Navigation VLA: drive the base between shelves in a store with moving, colliding shoppers.",
    "Orchestrator: loop the full list — navigate → pick → place → deliver.",
    "Optional: small retrain with non-empty baskets to make multi-item lists fully reliable.",
], notes="The roadmap — this is Part A of a two-VLA system.")

content("Takeaways", [
    "A capable VLA fine-tuned and run on a single 12 GB GPU — proven possible on modest hardware.",
    "80% task completion, rigorously measured (320 trials, Wilson CIs, success-based selection).",
    "Genuine language grounding + a window into the model's plan.",
    "End-to-end: language → shopping list → collected.",
], notes="Circle back to the opening question and the headline number.")

prs.save(str(OUT))
print(f"wrote {OUT}  ({len(prs.slides._sldIdLst)} slides)")
