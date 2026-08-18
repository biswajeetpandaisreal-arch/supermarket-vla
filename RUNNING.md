# Running this project

Python deps (mujoco, lerobot, torch+cu128) live in the reference project's venv,
reused here via a `.venv` symlink → `../smolvla_ur10e/.venv`.

**Do not use the `(base)` conda env — it has none of these packages.**

Two ways to run:

```bash
# 1. one-off (recommended): use the venv python directly, with EGL rendering
MUJOCO_GL=egl .venv/bin/python scripts/verify_scene.py

# 2. or activate the venv for the session
source .venv/bin/activate
MUJOCO_GL=egl python scripts/verify_scene.py
```

`MUJOCO_GL=egl` is required for offscreen (headless) rendering. Omit it only if
you have a display and want the interactive viewer.
