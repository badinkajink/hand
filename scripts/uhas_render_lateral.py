#!/usr/bin/env python3
"""Render what `yaw` does to the fingers, so the claim can be looked at instead of argued.

Two rows per hand, both viewed straight down the palm normal (a top-down view is the only
one in which abduction and flexion are visually separable):

  row 1  q = 0            -- fingers fully extended. The yaw axis is colinear with the
                             finger, so sweeping it is a pure ROLL: the tips do not move.
  row 2  the open keyframe -- the pose the hand actually operates at. The same sweep now
                             swings the tips sideways across the palm: real ABDUCTION.

Columns are yaw = lo, mid, hi applied to ALL THREE fingers at once.

    .venv/bin/python scripts/uhas_render_lateral.py --out docs/uhas/figs/lateral_sweep.png

Pair it with scripts/uhas_lateral_authority.py, which puts numbers on the same thing.
"""

from __future__ import annotations

import argparse
import pathlib

import mujoco
import numpy as np
from PIL import Image, ImageDraw

FINGERS = ("thumb", "index", "middle")
W, H = 520, 460


def render(model, data, cam) -> np.ndarray:
    with mujoco.Renderer(model, height=H, width=W) as r:
        mujoco.mj_forward(model, data)
        r.update_scene(data, camera=cam)
        return r.render()


def tip_xy(model, data, palm_R, palm_p) -> list[np.ndarray]:
    out = []
    for f in FINGERS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{f}_tip")
        out.append(palm_R.T @ (data.xpos[bid] - palm_p))
    return out


def build(xml: pathlib.Path, label: str, out: pathlib.Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.15]
    cam.distance = 0.42
    cam.azimuth = 90.0
    cam.elevation = -89.0  # straight down the palm normal

    # poses: q=0 and every keyframe named "open"
    poses: list[tuple[str, np.ndarray]] = [("q = 0  (fully extended)", np.zeros(model.nq))]
    for k in range(model.nkey):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k) or "") == "open":
            poses.append(("open keyframe  (operating pose)", model.key_qpos[k].copy()))

    yaw_adr, yaw_rng = [], []
    for f in FINGERS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_yaw")
        yaw_adr.append(model.jnt_qposadr[jid])
        yaw_rng.append(model.jnt_range[jid])

    cols = 3
    sheet = Image.new("RGB", (W * cols, H * len(poses) + 34 * len(poses) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), f"{label}:  yaw sweep, viewed down the palm normal", fill="black")

    for r_i, (pose_name, qpos) in enumerate(poses):
        y0 = 30 + r_i * (H + 34)
        # measure tip travel for the caption
        spans = []
        for c_i, frac in enumerate((0.0, 0.5, 1.0)):
            data.qpos[:] = qpos
            for adr, (lo, hi) in zip(yaw_adr, yaw_rng):
                data.qpos[adr] = lo + frac * (hi - lo)
            img = render(model, data, cam)
            sheet.paste(Image.fromarray(img), (c_i * W, y0 + 24))
            lab = ("yaw = min", "yaw = 0 (mid)", "yaw = max")[c_i]
            draw.text((c_i * W + 10, y0 + 6), lab, fill="black")

        # tip excursion between the two extremes, in the palm plane
        pb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        ends = []
        for frac in (0.0, 1.0):
            data.qpos[:] = qpos
            for adr, (lo, hi) in zip(yaw_adr, yaw_rng):
                data.qpos[adr] = lo + frac * (hi - lo)
            mujoco.mj_kinematics(model, data)
            ends.append(tip_xy(model, data, data.xmat[pb].reshape(3, 3), data.xpos[pb]))
        spans = [np.linalg.norm(a[:2] - b[:2]) * 1e3 for a, b in zip(*ends)]
        cap = (f"{pose_name}   |   in-palm-plane tip travel, min->max yaw:  "
               + "  ".join(f"{f}={s:.0f}mm" for f, s in zip(FINGERS, spans)))
        draw.text((10, y0 + 8 + H + 4), cap, fill="black")

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out}  ({sheet.size[0]}x{sheet.size[1]})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="assets/mjcf/baseline/hand_morphology_actuated.xml")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/uhas/figs/lateral_sweep.png"))
    a = ap.parse_args()
    build(pathlib.Path(a.xml), a.label, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
