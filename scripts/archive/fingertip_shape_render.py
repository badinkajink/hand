#!/usr/bin/env python3
"""Tile every fingertip shape variant into one close-up image, so the family can be LOOKED at.

    MUJOCO_GL=egl uv run python scripts/fingertip_shape_render.py \
        --scene results/phase1/landscape/m05_ik_cem/frozen_scene.xml \
        --out docs/experiments/20260818-fingertip_shapes/tips.png

A tip is 5-12 mm on a hand that is 120 mm across, so a whole-scene render shows nothing about
it and a numbers-only comparison of contact counts cannot tell "the ridge straddles the shaft"
from "the ridge is buried inside the distal capsule". This renders each variant zoomed on ONE
fingertip against the shaft it has to hold, from two directions: `along` looks down the shaft
(shows how the tip wraps it) and `across` looks perpendicular (shows the contact length).

It also prints, per variant, the fingertip-to-shaft gap at the grasp keyframe. A shape whose
gap goes NEGATIVE is already interpenetrating before the grip closes, and any later contact
number from it is measuring an initial-condition bug, not a design.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from morphohand.studies.scene_mutate import TIP_SHAPES, Scene  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _render(model, data, lookat, azimuth, elevation, distance, w, h):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    with mujoco.Renderer(model, height=h, width=w) as r:
        r.update_scene(data, camera=cam, scene_option=opt)
        return r.render()


def tip_gap(model, data, tip_body: str) -> float:
    """Smallest surface distance between the tip's geoms and the object, in mm."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tip_body)
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screwdriver_medium")
    tips = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
    objs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == obj]
    best = 1e9
    fromto = np.zeros(6)
    for g1 in tips:
        for g2 in objs:
            best = min(best, mujoco.mj_geomDistance(model, data, g1, g2, 1.0, fromto))
    return best * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path,
                    default=ROOT / "results/phase1/landscape/m05_ik_cem/frozen_scene.xml")
    ap.add_argument("--keyframe", default="open_ik")
    ap.add_argument("--tip", default="index_tip")
    ap.add_argument("--radius", type=float, default=0.005)
    ap.add_argument("--half-length", type=float, default=0.006)
    ap.add_argument("--distance", type=float, default=0.045)
    ap.add_argument("--shapes", nargs="*", default=list(TIP_SHAPES))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tile_w, tile_h, pad = 420, 320, 26
    cols = 4
    rows = (len(args.shapes) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * (2 * tile_h + pad)), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    work = ROOT / "results/phase1/fingertip/scenes"

    for i, shape in enumerate(args.shapes):
        path = Scene(args.scene).set_tip_shape(
            shape, r=args.radius, h=args.half_length).write(work / f"tip_{shape}.xml")
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(
            model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe))
        mujoco.mj_forward(model, data)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.tip)
        lookat = np.array(data.xpos[bid])
        gap = tip_gap(model, data, args.tip)
        ngeom = int((model.geom_bodyid == bid).sum())
        print(f"{shape:15s} geoms/tip {ngeom}  tip-to-shaft gap {gap:6.2f} mm"
              f"{'   <-- ALREADY TOUCHING' if gap <= 0 else ''}")

        # `along` looks down the shaft (world -Y); `across` looks perpendicular to it.
        a = _render(model, data, lookat, 90, -10, args.distance, tile_w, tile_h)
        b = _render(model, data, lookat, 0, -20, args.distance, tile_w, tile_h)
        cx, cy = (i % cols) * tile_w, (i // cols) * (2 * tile_h + pad)
        sheet.paste(Image.fromarray(a), (cx, cy + pad))
        sheet.paste(Image.fromarray(b), (cx, cy + pad + tile_h))
        draw.text((cx + 8, cy + 6), f"{shape}   gap {gap:.1f} mm", fill=(240, 240, 240))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"\nwrote {args.out}  (top row of each pair = along the shaft, bottom = across it)")


if __name__ == "__main__":
    main()
