"""
supermarket_env.py — MuJoCo scene for the supermarket VLA robot.

A UR10e + Robotiq 2F-85 mounted on a KINEMATIC mobile base (actuated slide-x,
slide-y, hinge-yaw — no wheel contact physics), facing a single supermarket
shelf stocked with real textured grocery meshes (robosuite's milk/can/bread/
cereal/bottle objects), with a basket riding on the base platform.

Adapted from the pick-and-place env of my earlier UR10e project
(github.com/biswajeetpandaisreal-arch/VLA_UR10e). The programmatic
MJCF-assembly and gripper/wrist-camera injection patterns are reused; the arm is
reparented under a mobile base and the table/single-cube is replaced by a
shelf + products.

Actuator layout (model.ctrl indexing):
    [0:6]  arm position servos  (shoulder_pan, shoulder_lift, elbow, wrist_1..3)
    [6]    gripper              (2F-85 fingers_actuator, 0..255)
    [7:10] base velocity        (base_x, base_y, base_yaw)

Run directly to build the scene and dump basic info.
"""
import os
import re
import importlib.util
from pathlib import Path

import numpy as np
import gymnasium as gym
import mujoco


def _menagerie_dir() -> Path:
    """MuJoCo Menagerie checkout: $MUJOCO_MENAGERIE_DIR if set, otherwise the copy
    robot_descriptions downloads on first use (~/.cache/robot_descriptions)."""
    if os.environ.get("MUJOCO_MENAGERIE_DIR"):
        return Path(os.environ["MUJOCO_MENAGERIE_DIR"])
    from robot_descriptions import ur10e_mj_description
    return Path(ur10e_mj_description.PACKAGE_PATH).parent


MENAGERIE = _menagerie_dir()
UR10E_DIR = MENAGERIE / "universal_robots_ur10e"
GRIPPER_DIR = MENAGERIE / "robotiq_2f85"

# robosuite ships the standard textured pick-place grocery meshes; we reuse them
# rather than hand-modelling primitives. Located without importing robosuite
# (find_spec avoids its noisy import side effects).
_RS_ROOT = Path(importlib.util.find_spec("robosuite").origin).parent
RS_OBJ_DIR = _RS_ROOT / "models" / "assets" / "objects"
RS_MESH_DIR = RS_OBJ_DIR / "meshes"
RS_TEX_DIR = _RS_ROOT / "models" / "assets" / "textures"
# Omron LD-60 AGV mobile base (real robosuite mesh) instead of a plain box platform.
RS_BASE_DIR = _RS_ROOT / "models" / "assets" / "bases"
RS_BASE_MESH_DIR = RS_BASE_DIR / "meshes" / "omron_mobile_base"

# ── Scene geometry (SI units, metres) ──────────────────────────────────────────
# The arm mounts on top of the Omron AGV's support column (~0.70 m). The AGV's
# 3 mobile joints (forward/side/yaw) already float 2 cm above the floor, so the
# base moves without ground friction.
ARM_MOUNT_OFFSET = 0.0            # extra height of the arm base above the AGV support mount
ARM_MOUNT_X = -0.05              # x of the arm base, matching the AGV support-column centre
# Basket is a raised tote on a short stand in front of the arm column: a short,
# wrist-camera-visible place motion with a forgiving drop. Base-relative; rides
# with the AGV. z is the basket-floor height; a stand connects it to the deck.
BASKET_POS = (0.24, 0.00, 0.47)   # nudged forward toward where the policy tends to drop
BASKET_HALF = (0.15, 0.13)        # larger, more forgiving footprint (half-size x, y)
DECK_TOP_Z = 0.38                 # AGV deck top (stand spans deck -> basket floor)

# Shelf: ~2 m wide (y), 1.8 m tall (z), 0.4 m deep (x). Front face toward the robot (+x side).
SHELF_FRONT_X = 0.75       # x of the shelf's front (robot-facing) edge
SHELF_DEPTH   = 0.40
SHELF_HALF_W  = 1.00       # half-width in y  → 2 m wide
SHELF_HEIGHT  = 1.80
PANEL_HALF_T  = 0.02       # half-thickness of shelf panels
SHELF_LEVELS  = [0.50, 1.00, 1.50]   # top-surface z of the 3 product shelves

SHELF_BACK_X   = SHELF_FRONT_X + SHELF_DEPTH        # 1.15
SHELF_CENTER_X = SHELF_FRONT_X + SHELF_DEPTH / 2.0  # 0.95

# ── Store surroundings: make it read as a supermarket aisle, not a lone shelf. ──
# A second gondola faces the robot across the aisle; extra shelves are stocked
# with static (decorative) products. No enclosing walls (open aisle).
FACING_SHELF_CX = -1.55                        # gondola across the aisle (opens toward robot)

# ── Products: real textured robosuite grocery meshes, each a free body with a
# unique name.  obj = robosuite object; mesh/tex/material fields mirror the
# robosuite object XML so the meshes look exactly as they do in robosuite.
#   bottom = z of the mesh's lowest point (from robosuite's bottom_site), used to
#            rest the object on the shelf surface.
RS_SPECS = {
    "milk":   dict(mesh="milk.msh",   scale="0.9 0.9 0.9", tex="ceramic.png", reflectance="0.5", texrepeat="1 1",   texuniform="true",  bottom=0.085),
    "can":    dict(mesh="can.msh",    scale="1 1 1",       tex="soda.png",    reflectance="0.7", texrepeat="5 5",   texuniform="true",  bottom=0.060),
    "bread":  dict(mesh="bread.stl",  scale="0.8 0.8 0.8", tex="bread.png",   reflectance="0.7", texrepeat="15 15", texuniform="true",  bottom=0.045),
    "cereal": dict(mesh="cereal.msh", scale="1 1 1",       tex="cereal.png",  reflectance="0.5", texrepeat="1 1",   texuniform="false", bottom=0.100),
    "bottle": dict(mesh="bottle.stl", scale="1 1 1",       tex="glass.png",   reflectance="0.5", texrepeat="5 5",   texuniform="true",  bottom=0.082),
}
PRODUCTS = [
    # y positions kept within the arm's reliable grasp zone (|y|<=~0.28) so that
    # per-episode jitter never pushes an item into the marginal extended-reach edge.
    dict(name="milk_carton",  obj="milk",   level=1, y=-0.26),
    dict(name="cola_can",     obj="can",    level=1, y=-0.13),
    dict(name="bread_loaf",   obj="bread",  level=1, y= 0.00),
    dict(name="cereal_box",   obj="cereal", level=1, y= 0.15, grasp_yaw=90),  # wide box: close fingers across the narrow face
    dict(name="water_bottle", obj="bottle", level=1, y= 0.34, grasp_gl=0.18),  # grip higher on the body for a stable hold
]
PRODUCT_FRONT_X = SHELF_FRONT_X + 0.13  # rest well inside the open front edge (easy front-edge access)

# Home arm pose (radians) — points the wrist/camera toward the middle-shelf
# product cluster. Tuned against the wrist-camera render in verify_scene.py.
# Tuned for the AGV-mounted arm (base ~0.70 m): HOME frames the middle-shelf
# products in the wrist camera; TRANSIT folds the arm up over the AGV, clear of
# the shelf, so the base can navigate without the arm colliding.
HOME_QPOS = np.array([2.9114, -1.2500, 1.3500, -1.9000, -1.5708, 0.0000], dtype=np.float32)
TRANSIT_QPOS = np.array([2.9114, -2.4000, 2.0000, -1.0000, -1.5708, 0.0000], dtype=np.float32)

JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
# Omron AGV mobile DOFs (forward=x, side=y, yaw=z) and their velocity actuators.
BASE_JOINTS = ["joint_mobile_forward", "joint_mobile_side", "joint_mobile_yaw"]
BASE_ACTS   = ["actuator_mobile_forward", "actuator_mobile_side", "actuator_mobile_yaw"]


# ── XML helpers (copied from the reference env; read-only reuse) ────────────────
def _load_dir(path: Path, suffixes=(".obj", ".stl", ".png", ".jpg", ".xml")) -> dict:
    return {f.name: f.read_bytes() for f in path.rglob("*") if f.suffix in suffixes}

def _strip_keyframe(xml: str) -> str:
    return re.sub(r'<keyframe>.*?</keyframe>', '', xml, flags=re.DOTALL)

def _patch_gripper(xml: str) -> str:
    for name in ["visual", "collision"]:
        xml = xml.replace(f'class="{name}"',      f'class="2f85_{name}"')
        xml = xml.replace(f'childclass="{name}"', f'childclass="2f85_{name}"')
    for attr in ["name=", "body1=", "body2="]:
        xml = xml.replace(f'{attr}"base"', f'{attr}"2f85_base"')
    xml = xml.replace('name="black"',     'name="2f85_black"')
    xml = xml.replace('material="black"', 'material="2f85_black"')
    return xml

def _extract(xml: str, tag: str) -> str:
    m = re.search(rf'<{tag}>(.*?)</{tag}>', xml, re.DOTALL)
    return m.group(1) if m else ""

def _extract_full(xml: str, tag: str) -> str:
    m = re.search(rf'<{tag}>.*</{tag}>', xml, re.DOTALL)
    return m.group(0) if m else ""


class SupermarketEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, image_size=256, render_mode="rgb_array", home_qpos=None,
                 base_start=(0.0, 0.0, 0.0)):
        super().__init__()
        self.image_size = image_size
        self.render_mode = render_mode
        self.home_qpos = np.array(HOME_QPOS if home_qpos is None else home_qpos, dtype=np.float32)
        self.base_start = np.array(base_start, dtype=np.float32)

        scene_xml, assets = self._build_scene()
        self.model = mujoco.MjModel.from_xml_string(scene_xml, assets=assets)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=image_size, width=image_size)
        self._viewer = None

        # Cache actuator / joint addressing so callers never hardcode indices.
        self.arm_act = [self.model.actuator(n).id for n in
                        ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]]
        self.gripper_act = self.model.actuator("fingers_actuator").id
        self.base_act = [self.model.actuator(n).id for n in BASE_ACTS]
        self.arm_qadr = [self.model.joint(n).qposadr[0] for n in JOINT_NAMES]
        self.base_qadr = [self.model.joint(n).qposadr[0] for n in BASE_JOINTS]
        self.product_names = [p["name"] for p in PRODUCTS]

        # Gravity-compensate the arm + gripper (like a real UR controller) so the
        # position servos hit their targets instead of drooping ~8 cm under the
        # arm's own weight — otherwise the gripper never reaches the commanded
        # grasp/place pose. Products/shelf/basket are NOT compensated (items must
        # still fall into the basket).
        base_id = self.model.body("base").id
        arm_subtree = {base_id}
        for i in range(self.model.nbody):                 # bodies are ordered parent-before-child
            if self.model.body_parentid[i] in arm_subtree:
                arm_subtree.add(i)
        for i in arm_subtree:
            self.model.body_gravcomp[i] = 1.0

    # ── Scene assembly ─────────────────────────────────────────────────────────
    def _build_scene(self):
        ur10e_xml = _strip_keyframe((UR10E_DIR / "ur10e.xml").read_text())

        # Wrist camera + gripper injection (mirrors the reference env exactly).
        pos_match  = re.search(r'<site name="attachment_site"[^>]*pos="([^"]+)"', ur10e_xml)
        quat_match = re.search(r'<site name="attachment_site"[^>]*quat="([^"]+)"', ur10e_xml)
        site_pos   = pos_match.group(1) if pos_match else "0 0.0922 0"
        site_quat  = quat_match.group(1) if quat_match else "-1 1 0 0"

        gripper_xml  = _patch_gripper((GRIPPER_DIR / "2f85.xml").read_text())
        gripper_xml  = gripper_xml.replace('forcerange="-5 5"', 'forcerange="-20 20"')
        gripper_body = re.search(r'(<body name="base_mount".*?</body>)\s*</worldbody>',
                                 gripper_xml, re.DOTALL).group(1)
        gripper_body = gripper_body.replace(
            'name="base_mount" pos="0 0 0.007"',
            f'name="base_mount" pos="{site_pos}" quat="{site_quat}"')
        # Eye-in-hand camera rigidly attached inside the gripper base body (FOV ~75°).
        gripper_body = gripper_body.replace(
            '<site name="pinch" pos="0 0 0.145" type="sphere" group="5" rgba="0.9 0.9 0.9 1" size="0.005"/>',
            '<site name="pinch" pos="0 0 0.145" type="sphere" group="5" rgba="0.9 0.9 0.9 1" size="0.005"/>\n'
            '        <camera name="wrist_cam" pos="0.06 0.06 0.05" zaxis="0.06 0.06 -0.095" fovy="75"/>')
        ur10e_xml = ur10e_xml.replace('<site name="attachment_site"',
                                      gripper_body + '\n        <site name="attachment_site"')

        # Reparent: pull out the arm's <body name="base"> subtree (drop the
        # menagerie spotlight) and mount it ARM_MOUNT_Z above the mobile base.
        arm_worldbody = _extract(ur10e_xml, "worldbody")
        arm_body = re.sub(r'<light[^>]*/>', '', arm_worldbody, count=1).strip()
        arm_body = arm_body.replace(
            '<body name="base" quat="0 0 0 -1" childclass="ur10e">',
            f'<body name="base" pos="{ARM_MOUNT_X} 0 {ARM_MOUNT_OFFSET}" quat="0 0 0 -1" childclass="ur10e">', 1)

        ur10e_default = _extract_full(ur10e_xml, "default")
        ur10e_asset   = _extract(ur10e_xml, "asset")
        ur10e_actuator = _extract(ur10e_xml, "actuator")

        # Gripper defaults with grippier friction (reference tuning).
        gripper_defaults = _extract_full(gripper_xml, "default").replace(
            "<default>",
            '<default>\n    <geom friction="2.0 0.5 0.1" solref="0.004 1" solimp="0.95 0.999 0.001 0.5 2"/>',
            1)
        gripper_defaults = gripper_defaults.replace('friction="0.7"', 'friction="3.0 0.5 0.1" condim="4"')
        gripper_defaults = gripper_defaults.replace('friction="0.6"', 'friction="3.0 0.5 0.1" condim="4"')
        gripper_defaults = gripper_defaults.replace(
            'solimp="0.95 0.99 0.001" solref="0.004 1" priority="1"',
            'solimp="0.95 0.999 0.001 0.5 2" solref="0.004 1" priority="1"')

        assets = {**_load_dir(UR10E_DIR), **_load_dir(GRIPPER_DIR)}
        # Pull in the robosuite mesh + texture bytes each product needs.
        for spec in RS_SPECS.values():
            assets[spec["mesh"]] = (RS_MESH_DIR / spec["mesh"]).read_bytes()
            assets[spec["tex"]] = (RS_TEX_DIR / spec["tex"]).read_bytes()
        # robosuite arena textures for the floor / walls / shelving (store look).
        for tex in ("light-gray-floor-tile.png", "steel-brushed.png"):
            assets[tex] = (RS_TEX_DIR / tex).read_bytes()

        # ── Omron AGV base: mount the arm on it, drop the torso-lift DOF (keep the
        #    plan's 3-DOF forward/side/yaw base), and add the basket on its deck. ──
        omron = (RS_BASE_DIR / "omron_mobile_base.xml").read_text()
        omron = omron.replace('meshes/omron_mobile_base/', '')  # reference meshes by basename
        for f in RS_BASE_MESH_DIR.glob("*.obj"):
            assets[f.name] = f.read_bytes()
        omron_asset = _extract(omron, "asset")
        omron_actuator = _extract(omron, "actuator")
        omron_actuator = re.sub(r'<motor[^>]*joint="joint_torso_height"[^>]*/>', '', omron_actuator)
        base_body = _extract(omron, "worldbody")
        base_body = base_body.replace('<body name="base" pos="0 0 0">',
                                      '<body name="mobile_base" pos="0 0 0">', 1)
        base_body = re.sub(r'<joint name="joint_torso_height"[^>]*/>', '', base_body)  # rigid column
        # robosuite sets frictionloss=250 on the mobile joints (tuned for their
        # controllers); too high for our simple velocity-servo nav to position
        # precisely — it stalls ~5 cm short. Lower it so the base parks cleanly.
        base_body = base_body.replace('frictionloss="250"', 'frictionloss="15"')
        # Center the AGV chassis under the arm column: robosuite parks the chassis
        # 0.20 m behind the mount, leaving the arm perched on the front edge. Shift
        # it forward so the arm sits squarely on the deck.
        base_body = base_body.replace('<body name="wheeled_base" pos="-0.20 0 0.192">',
                                      '<body name="wheeled_base" pos="-0.05 0 0.192">')
        # The pedestal collision box exactly encloses the Omron visual mesh and
        # renders as an opaque box hiding it — make it invisible (alpha 0) while
        # keeping its collision so the real AGV mesh shows.
        base_body = base_body.replace(
            'type="box" name="pedestal_feet_col" density="10"/>',
            'type="box" name="pedestal_feet_col" density="10" group="3" rgba="0.2 0.2 0.2 0"/>')
        base_body = base_body.replace('<!-- add robot here -->', arm_body)
        # basket + a dedicated camera aimed at it (rides on the AGV), so the policy
        # always has a clear view of the drop target during placing.
        basket_cam = ('<camera name="basket_cam" pos="0.24 -0.42 0.88" '
                      'mode="targetbody" target="basket"/>')
        base_body = base_body.replace('<body name="mobile_base" pos="0 0 0">',
                                      '<body name="mobile_base" pos="0 0 0">\n      '
                                      + self._basket_xml() + basket_cam, 1)

        scene_xml = f"""
<mujoco model="supermarket_vla">
  <compiler angle="radian" meshdir="." autolimits="true"/>
  <option integrator="implicitfast" timestep="0.002" cone="elliptic" impratio="10" solver="Newton" noslip_iterations="0"/>
  <visual><global offwidth="1600" offheight="1600"/></visual>
  {ur10e_default}
  {gripper_defaults}
  <asset>
    {ur10e_asset}
    {_extract(gripper_xml, "asset")}
    <texture type="skybox" builtin="gradient" rgb1="0.85 0.88 0.92" rgb2="0.4 0.45 0.5" width="512" height="512"/>
    <texture name="floor_tex" type="2d" file="light-gray-floor-tile.png"/>
    <material name="groundplane" texture="floor_tex" texrepeat="6 6" texuniform="true" reflectance="0.1"/>
    <texture name="shelf_tex" type="cube" file="steel-brushed.png"/>
    <material name="shelf_mat" texture="shelf_tex" reflectance="0.3" specular="0.4" shininess="0.4"/>
    <material name="tote_mat" rgba="0.14 0.32 0.52 1" specular="0.7" shininess="0.7" reflectance="0.2"/>
    <material name="tote_rim_mat" rgba="0.09 0.20 0.34 1" specular="0.7" shininess="0.6"/>
    <material name="pedestal_mat" rgba="0.22 0.23 0.26 1" specular="0.5" shininess="0.4"/>
    {self._product_assets_xml()}
    {omron_asset}
  </asset>

  <worldbody>
    <light pos="0.6 0 3.0" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <light pos="0.0 -1.2 2.4" dir="0.3 0.5 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="5 5 0.05" material="groundplane"/>

    <!-- Static camera facing the shelf (3/4 view over the robot's shoulder). -->
    <camera name="scene_cam" pos="-0.9 -1.6 1.9" mode="targetbody" target="shelf"/>

    {self._shelf_xml("shelf", SHELF_CENTER_X, back_plus_x=True, levels=[SHELF_LEVELS[0], SHELF_LEVELS[1]])}
    {self._shelf_xml("shelf_facing", FACING_SHELF_CX, back_plus_x=False)}
    {self._products_xml()}
    {self._decor_products_xml()}

    <!-- Omron LD-60 AGV base (robosuite mesh) with the arm + basket mounted on it. -->
    {base_body}
  </worldbody>

  <tendon>{_extract(gripper_xml, "tendon")}</tendon>
  <actuator>
    {ur10e_actuator}
    {_extract(gripper_xml, "actuator")}
    {omron_actuator}
  </actuator>
  <equality>{_extract(gripper_xml, "equality")}</equality>
  <contact>{_extract(gripper_xml, "contact")}</contact>
</mujoco>
"""
        return scene_xml, assets

    def _shelf_xml(self, name="shelf", cx=SHELF_CENTER_X, back_plus_x=True, levels=None):
        """One shelf unit built from box primitives: back panel, two sides,
        product shelves at `levels`, plus top and bottom panels. `back_plus_x`
        puts the solid back on the +x side (shelf opens toward -x) or vice-versa.
        Omitting a level's panel opens that bay (so the arm can lift items out)."""
        if levels is None:
            levels = SHELF_LEVELS
        w, t = SHELF_HALF_W, PANEL_HALF_T
        depth_half = SHELF_DEPTH / 2.0
        back_x = (depth_half - t) if back_plus_x else -(depth_half - t)
        parts = [f'<body name="{name}" pos="{cx} 0 0">']
        # back panel (thin in x, at the back)
        parts.append(f'  <geom type="box" material="shelf_mat" '
                     f'pos="{back_x:.3f} 0 {SHELF_HEIGHT/2:.3f}" '
                     f'size="{t} {w} {SHELF_HEIGHT/2:.3f}"/>')
        # side panels (thin in y)
        for sy in (-w + t, w - t):
            parts.append(f'  <geom type="box" material="shelf_mat" '
                         f'pos="0 {sy:.3f} {SHELF_HEIGHT/2:.3f}" '
                         f'size="{depth_half} {t} {SHELF_HEIGHT/2:.3f}"/>')
        # horizontal shelves: bottom, the product levels, and a top
        for z in [PANEL_HALF_T] + levels + [SHELF_HEIGHT - PANEL_HALF_T]:
            parts.append(f'  <geom type="box" material="shelf_mat" '
                         f'pos="0 0 {z:.3f}" size="{depth_half} {w} {t}" '
                         f'friction="1 0.05 0.05"/>')
        parts.append('</body>')
        return "\n    ".join(parts)

    def _basket_xml(self):
        """Raised plastic tote on a solid pedestal, riding on the AGV (child of
        mobile_base). z of BASKET_POS is the tote-floor height."""
        x, y, z = BASKET_POS
        hx, hy = BASKET_HALF
        wall, wh = 0.008, 0.055                   # wall thickness (half) and wall height (half)
        rim = 2 * wh                              # rim sits at the top of the walls
        floor_bottom = z - 0.01
        ped_hz = (floor_bottom - DECK_TOP_Z) / 2.0
        ped_cz = -(z - (DECK_TOP_Z + ped_hz))     # relative to the tote body origin
        g = lambda pos, size, mat: f'        <geom type="box" pos="{pos}" size="{size}" material="{mat}"/>\n'
        return (
            f'<body name="basket" pos="{x} {y} {z}">\n'
            # solid pedestal stand (dark), deck -> tote floor
            + g(f"0 0 {ped_cz:.3f}", f"{hx-0.03:.3f} {hy-0.03:.3f} {ped_hz:.3f}", "pedestal_mat")
            # tote floor + four plastic walls
            + f'        <geom type="box" size="{hx} {hy} 0.008" material="tote_mat"/>\n'
            + g(f"0 {-hy+wall:.3f} {wh}", f"{hx} {wall} {wh}", "tote_mat")
            + g(f"0 { hy-wall:.3f} {wh}", f"{hx} {wall} {wh}", "tote_mat")
            + g(f"{-hx+wall:.3f} 0 {wh}", f"{wall} {hy} {wh}", "tote_mat")
            + g(f"{ hx-wall:.3f} 0 {wh}", f"{wall} {hy} {wh}", "tote_mat")
            # darker rim around the top edge for a finished look
            + g(f"0 {-hy:.3f} {rim:.3f}", f"{hx} 0.006 0.007", "tote_rim_mat")
            + g(f"0 { hy:.3f} {rim:.3f}", f"{hx} 0.006 0.007", "tote_rim_mat")
            + g(f"{-hx:.3f} 0 {rim:.3f}", f"0.006 {hy} 0.007", "tote_rim_mat")
            + g(f"{ hx:.3f} 0 {rim:.3f}", f"0.006 {hy} 0.007", "tote_rim_mat")
            + '      </body>')

    def _decor_products_xml(self):
        """Static (non-colliding, visual-only) robosuite meshes stocking the other
        shelves so the store looks full. Not manipulated — no freejoint."""
        objs = list(RS_SPECS.keys())
        dh = SHELF_DEPTH / 2.0
        # (shelf_center_x, front_x, levels, y-positions)
        layouts = [
            (FACING_SHELF_CX, FACING_SHELF_CX + dh - 0.10, [0, 1, 2], [-0.75, -0.4, -0.05, 0.3, 0.65]),
            # working shelf: only the lower bay has decor (level 1 holds the target
            # items; the bay above level 1 is now open so the arm can lift them out).
            (SHELF_CENTER_X,  PRODUCT_FRONT_X,             [0],       [-0.75, -0.4, 0.4, 0.75]),
        ]
        parts, i = [], 0
        for cx, fx, levels, ys in layouts:
            for lvl in levels:
                for y in ys:
                    obj = objs[i % len(objs)]
                    s = RS_SPECS[obj]
                    z = SHELF_LEVELS[lvl] + PANEL_HALF_T + s["bottom"] + 0.002
                    parts.append(
                        f'<body name="decor_{i}" pos="{fx:.3f} {y:.3f} {z:.3f}">'
                        f'<geom type="mesh" mesh="{obj}_mesh" material="{obj}_mat" '
                        f'contype="0" conaffinity="0" group="1"/></body>')
                    i += 1
        return "\n    ".join(parts)

    def _product_assets_xml(self):
        """One mesh + texture + material per product, mirroring robosuite's own
        object XML so the meshes render exactly as they do in robosuite."""
        parts = []
        for obj, s in RS_SPECS.items():
            parts.append(f'<mesh name="{obj}_mesh" file="{s["mesh"]}" scale="{s["scale"]}"/>')
            parts.append(f'<texture name="{obj}_tex" type="2d" file="{s["tex"]}"/>')
            parts.append(f'<material name="{obj}_mat" texture="{obj}_tex" '
                         f'reflectance="{s["reflectance"]}" texrepeat="{s["texrepeat"]}" '
                         f'texuniform="{s["texuniform"]}"/>')
        return "\n    ".join(parts)

    def _products_xml(self):
        parts = []
        for p in PRODUCTS:
            s = RS_SPECS[p["obj"]]
            surf = SHELF_LEVELS[p["level"]] + PANEL_HALF_T   # top of that shelf panel
            z = surf + s["bottom"] + 0.005                   # rest the mesh's base on the surface
            parts.append(
                f'<body name="{p["name"]}" pos="{PRODUCT_FRONT_X:.3f} {p["y"]:.3f} {z:.3f}">\n'
                f'      <freejoint/>\n'
                f'      <geom type="mesh" mesh="{p["obj"]}_mesh" material="{p["obj"]}_mat" '
                f'density="100" condim="4" friction="2.5 0.3 0.1" '
                f'solref="0.001 1" solimp="0.998 0.998 0.001"/>\n'
                f'    </body>')
        return "\n    ".join(parts)

    # ── Simulation control ──────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        for adr, q in zip(self.arm_qadr, self.home_qpos):
            self.data.qpos[adr] = q
        for adr, q in zip(self.base_qadr, self.base_start):
            self.data.qpos[adr] = q
        # Arm position servos hold the home pose; base velocity servos hold still.
        for act, q in zip(self.arm_act, self.home_qpos):
            self.data.ctrl[act] = q
        self.data.ctrl[self.gripper_act] = 0.0
        for act in self.base_act:
            self.data.ctrl[act] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def set_arm_pose(self, qpos):
        """Snap the arm to a joint configuration and command the servos to hold it."""
        for adr, act, q in zip(self.arm_qadr, self.arm_act, qpos):
            self.data.qpos[adr] = q
            self.data.ctrl[act] = q
        mujoco.mj_forward(self.model, self.data)

    def set_arm_target(self, qpos):
        """Command the arm servos toward a pose (no teleport); the arm moves there
        smoothly under the position controllers as the sim steps."""
        for act, q in zip(self.arm_act, qpos):
            self.data.ctrl[act] = q

    def set_base_velocity(self, vx=0.0, vy=0.0, wyaw=0.0):
        self.data.ctrl[self.base_act[0]] = vx
        self.data.ctrl[self.base_act[1]] = vy
        self.data.ctrl[self.base_act[2]] = wyaw

    def step_sim(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def render_camera(self, camera):
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def _get_obs(self):
        return {"base_pose": self.get_base_pose()}

    # ── Introspection helpers ───────────────────────────────────────────────────
    def get_base_pose(self):
        return np.array([self.data.qpos[a] for a in self.base_qadr], dtype=np.float64)

    def body_pos(self, name):
        return self.data.body(name).xpos.copy()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
        self.renderer.close()


if __name__ == "__main__":
    env = SupermarketEnv()
    env.reset()
    print(f"Model built OK — nq={env.model.nq}  nv={env.model.nv}  nu={env.model.nu}")
    print("Actuators:", [env.model.actuator(i).name for i in range(env.model.nu)])
    print("Products :", env.product_names)
    print("Base pose:", env.get_base_pose())
    env.close()
