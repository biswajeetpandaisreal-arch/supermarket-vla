"""
make_camera_views.py — render the THREE camera views the policy sees (wrist, scene,
basket) at a representative moment, tiled side by side with labels. For the slides.

    python scripts/make_camera_views.py   # -> outputs/plots/camera_views.png
"""
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.supermarket_env import SupermarketEnv, HOME_QPOS

OUT = Path(__file__).parent.parent / "outputs" / "plots" / "camera_views.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

env = SupermarketEnv(image_size=480)          # render big for a crisp slide
env.reset()
env.set_arm_pose(HOME_QPOS)
env.step_sim(30)

views = [("WRIST  (eye-in-hand)", "wrist_cam"),
         ("SCENE  (shelf-facing)", "scene_cam"),
         ("BASKET  (drop target)", "basket_cam")]

tiles = []
for label, cam in views:
    img = env.render_camera(cam)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    # label bar
    bar = np.full((46, w, 3), (43, 30, 18), np.uint8)   # dark navy (BGR)
    cv2.putText(bar, label, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    tile = np.vstack([bar, img])
    tile = cv2.copyMakeBorder(tile, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(230, 227, 221))
    tiles.append(tile)

gap = np.full((tiles[0].shape[0], 16, 3), 255, np.uint8)
row = tiles[0]
for t in tiles[1:]:
    row = np.hstack([row, gap, t])
row = cv2.copyMakeBorder(row, 18, 18, 18, 18, cv2.BORDER_CONSTANT, value=(255, 255, 255))
cv2.imwrite(str(OUT), row)
print("wrote", OUT, row.shape)
env.close()
