"""Render a MuJoCo scene to PNG(s) so an agent can *look* at it.

Closes the visual loop when editing MJCF geometry / keyframes: instead of guessing from
numbers, render the pose from several canonical viewpoints, tile them into one image, and
read it back. Also prints a numeric report (fingertip world positions, object pose, active
contacts) so the picture and the numbers can be cross-checked.

Run (headless):
  MUJOCO_GL=egl uv run python scripts/mj_snap.py \
    --scene assets/mjcf/scene_screwdriver_medium_flat.xml \
    --keyframe open --out /tmp/snap.png

Settle/close the fingers before looking:
  ... --ctrl-from-keyframe open --steps 300
  ... --ctrl "0 2 -0.707  0 1.25 1  0 1.25 1" --steps 300

Views are free-camera presets (azimuth/elevation) around a lookat point that defaults to the
object body, falling back to the palm.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# azimuth, elevation
VIEWS: dict[str, tuple[float, float]] = {
    "front": (180.0, -10.0),   # looking down -x  (sees the y-z plane)
    "side": (90.0, -10.0),     # looking down -y  (sees the x-z plane)
    "top": (180.0, -89.0),     # straight down    (sees the x-y plane)
    "iso": (135.0, -25.0),
    "iso2": (225.0, -25.0),
    "back": (0.0, -10.0),
    "under": (180.0, 25.0),
}

TIP_BODIES = ("thumb_tip", "index_tip", "middle_tip")


def _body_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]


def _guess_lookat(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    names = _body_names(model)
    for candidate in names:
        if candidate and any(k in candidate for k in ("screwdriver", "cube", "object", "drill", "prism")):
            return np.array(data.body(candidate).xpos, dtype=float)
    for candidate in ("palm", "palm_pose"):
        if candidate in names:
            return np.array(data.body(candidate).xpos, dtype=float)
    return np.array([0.0, 0.0, 0.1])


def _object_body(model: mujoco.MjModel) -> str | None:
    for name in _body_names(model):
        if name and any(k in name for k in ("screwdriver", "cube", "object", "drill", "prism")):
            return name
    return None


def _parse_vec(raw: str | None) -> np.ndarray | None:
    if raw is None:
        return None
    return np.array([float(v) for v in raw.replace(",", " ").split()], dtype=float)


def _label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 8 * len(text) + 10, 20], fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 0))
    return img


def render(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    views: list[str],
    width: int,
    height: int,
    distance: float,
    lookat: np.ndarray,
    contacts: bool,
) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(opt)
        if contacts:
            opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        for view in views:
            az, el = VIEWS[view]
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.azimuth, cam.elevation, cam.distance = az, el, distance
            cam.lookat[:] = lookat
            renderer.update_scene(data, camera=cam, scene_option=opt)
            frames.append(_label(Image.fromarray(renderer.render()), view))
    return frames


def tile(frames: list[Image.Image], cols: int) -> Image.Image:
    rows = (len(frames) + cols - 1) // cols
    w, h = frames[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))
    for i, frame in enumerate(frames):
        sheet.paste(frame, ((i % cols) * w, (i // cols) * h))
    return sheet


def report(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    lines: list[str] = []
    names = _body_names(model)
    obj = _object_body(model)
    if obj:
        pos = data.body(obj).xpos
        mat = data.body(obj).xmat.reshape(3, 3)
        axis = mat[:, 2]  # cylinder long axis is local z
        lines.append(
            f"object {obj}: pos {np.round(pos, 4)}  long-axis {np.round(axis, 3)}  "
            f"|cos(z)| {abs(float(axis[2])):.3f}"
        )
    for tip in TIP_BODIES:
        if tip in names:
            tp = data.body(tip).xpos
            extra = ""
            if obj:
                extra = f"  d(obj-center) {np.linalg.norm(tp - data.body(obj).xpos) * 1000:6.1f} mm"
            lines.append(f"tip {tip:11s} {np.round(tp, 4)}{extra}")
    lines.append(f"ncon = {data.ncon}")
    seen: dict[tuple[str, str], float] = {}
    for i in range(data.ncon):
        con = data.contact[i]
        b1 = names[model.geom_bodyid[con.geom1]]
        b2 = names[model.geom_bodyid[con.geom2]]
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        key = (b1, b2)
        seen[key] = seen.get(key, 0.0) + float(np.linalg.norm(force[:3]))
    for (b1, b2), f in sorted(seen.items(), key=lambda kv: -kv[1]):
        lines.append(f"  contact {b1:14s} <-> {b2:22s} |f| {f:7.3f} N")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--keyframe", default="open")
    ap.add_argument("--qpos", default=None, help="explicit qpos vector (overrides keyframe)")
    ap.add_argument("--ctrl", default=None, help="explicit ctrl vector")
    ap.add_argument("--ctrl-from-keyframe", default=None, help="take ctrl from this keyframe")
    ap.add_argument("--steps", type=int, default=0, help="physics steps to run before rendering")
    ap.add_argument("--views", default="front,side,top,iso")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--distance", type=float, default=0.45)
    ap.add_argument("--lookat", default=None, help="x y z (default: object body)")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--contacts", action="store_true", help="draw contact points + forces")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)

    if args.qpos is not None:
        qpos = _parse_vec(args.qpos)
        assert qpos is not None
        data.qpos[: len(qpos)] = qpos
    elif args.keyframe:
        try:
            mujoco.mj_resetDataKeyframe(model, data, model.key(args.keyframe).id)
        except Exception as exc:  # keyframe missing -> default pose
            print(f"[warn] keyframe '{args.keyframe}' unusable ({exc}); using qpos0")

    if args.ctrl_from_keyframe:
        data.ctrl[:] = model.key(args.ctrl_from_keyframe).ctrl
    if args.ctrl is not None:
        ctrl = _parse_vec(args.ctrl)
        assert ctrl is not None
        data.ctrl[: len(ctrl)] = ctrl

    mujoco.mj_forward(model, data)
    for _ in range(args.steps):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    lookat = _parse_vec(args.lookat)
    if lookat is None:
        lookat = _guess_lookat(model, data)

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise SystemExit(f"unknown views {unknown}; choose from {sorted(VIEWS)}")

    frames = render(model, data, views, args.width, args.height, args.distance, lookat, args.contacts)
    sheet = tile(frames, args.cols)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)

    print(f"[snap] {args.scene.name} keyframe={args.keyframe} steps={args.steps} -> {args.out}")
    print(f"[snap] lookat {np.round(lookat, 4)} views {views}")
    print(report(model, data))


if __name__ == "__main__":
    main()
