"""A seconds-per-design REORIENTABILITY metric for `real_v1`, and the landscape it draws.

The program has wanted a cheap design score for a long time and the honest answer so far has been
no. `reorient_primitive.py feasibility` asks whether a hand can put its fingertips on a recorded
schedule and every design scores 0.2-2.2 mm on a 12.5 mm shaft -- reachability is SATURATED, so
it cannot rank anything (REORIENT_PRIMITIVE.txt section 5). The UHAS authority metric ranked the
grip and correlated -0.03 with goal approach.

What was missing is that a reorient is not a reachability question, it is a BUDGET question. A
fixed-contact rotation of theta about the pinch axis drives the descending pair contact down by
straddle * sin(theta), which its finger pays for by EXTENDING, and the mount-to-pad chain is
68.11 mm long. So the score is

    ceiling = asin(extension_left / straddle)

which needs one forward-kinematics call, and it ordered the four trained designs correctly
(rv04_mid 2.5 deg / 0.9 observed, rv00_wide 8.6 / 1.1, rv05_manual 9.6 / 1.7,
rv03_narrowy 10.2 / 4.0). This script computes it over the 6-dim XY workspace and, for each
design, also runs the OPEN-LOOP CARRY that the ceiling is supposed to predict -- so the metric is
reported next to the thing it claims to stand in for, not on its own.

Everything here is CPU and takes ~1 min per design. No CEM, no policy, no seed.

    MUJOCO_GL=egl uv run python scripts/real_v1_reorient_landscape.py --grid 3 \
        --out docs/experiments/20260828-real_v1_landscape/landscape.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import probe_real_v1_carry as pc  # noqa: E402
from morphohand.sampling.morphology import (  # noqa: E402
    morph_to_array, real_v1_compact_design, real_v1_mount_positions,
)
from morphohand.tools.morphology_xml import MorphologyValues  # noqa: E402

BASE_HAND = ROOT / "assets/mjcf/real_v1/real_hand.xml"
BASE_SCENE = ROOT / "assets/mjcf/real_v1/scenes/scene_screwdriver_medium.xml"
GEN = ROOT / "assets/mjcf/experimental/20260828-real_v1_landscape"


def gen_scene(vec) -> Path:
    GEN.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/generate_morphology_xml.py"),
                    "--base-hand-xml", str(BASE_HAND), "--base-scene-xml", str(BASE_SCENE),
                    "--output-dir", str(GEN),
                    "--thumb", *map(str, vec[0:3]), "--index", *map(str, vec[3:6]),
                    "--middle", *map(str, vec[6:9])],
                   check=True, capture_output=True, text=True, timeout=180)
    return sorted(GEN.glob("scene_*.xml"), key=lambda p: p.stat().st_mtime)[-1]


def evaluate(scene: Path, straddle: float, axis_ks, turn_steps: int, lift: float,
             obj: str) -> dict:
    """Extension budget at the fitted grasp, its ceiling, and the carry the ceiling predicts."""
    # The extension budget must come from the FITTED pose, not the scene's inherited `open_ik`
    # keyframe -- the generator carries the base design's pose forward, which belongs to
    # different mounts (the same trap that replaced rv05_manual's authored grasp overnight).
    built = pc._grip_from_fit(scene, straddle, 0.0, 0.004, obj)
    if built is None:
        return {"pose": False}
    ext = _extension(built[0], built[1])
    best = None
    for k in axis_ks:
        r = pc.carry(scene, lift, turn_steps, 400, np.radians(-90.0), 0.0, obj, 0.5, False,
                     straddle=straddle, label=scene.stem[:16], axis_k=k, linear_anchor=True)
        if r is None:
            return {"pose": False}
        if best is None or (r["ok"], r["final_cos"]) > (best["ok"], best["final_cos"]):
            best = r
    return {"pose": True, "carry": best, "extend_mm": ext}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=3, help="knob values per axis (thumb-x, pair-y)")
    ap.add_argument("--straddle", type=float, default=0.040)
    ap.add_argument("--axis-k", default="0.15,0.25,0.35,0.5")
    ap.add_argument("--turn-steps", type=int, default=550)
    ap.add_argument("--lift", type=float, default=0.10)
    ap.add_argument("--object-body", default="screwdriver_medium")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ks = [float(v) for v in args.axis_k.split(",")]
    # The workspace collapses to two knobs once the fitter re-centres palm X: thumb<->pair X
    # separation and index<->middle Y separation (see real_v1_pipeline.py).
    grid = np.linspace(0.0, 1.0, args.grid)
    rows = []
    print(f"{'design':16} {'Xsep':>6} {'Ysep':>6} {'extend':>7} {'ceiling':>8} "
          f"{'peak':>6} {'final':>6} {'z':>7} {'con':>4} {'N':>7}  ok")
    for xt in grid:
        for yt in grid:
            vec = [round(float(v), 4)
                   for v in morph_to_array(real_v1_compact_design(xt, xt, yt))]
            mounts = real_v1_mount_positions(MorphologyValues(*vec))
            xsep = abs(mounts["index"][0] - mounts["thumb"][0]) * 1000
            ysep = abs(mounts["index"][1] - mounts["middle"][1]) * 1000
            name = f"x{xt:.2f}_y{yt:.2f}"
            try:
                scene = gen_scene(vec)
            except Exception as exc:
                print(f"{name:16} generate failed: {type(exc).__name__}")
                continue
            ev = evaluate(scene, args.straddle, ks, args.turn_steps, args.lift, args.object_body)
            if not ev["pose"]:
                print(f"{name:16} {xsep:6.0f} {ysep:6.0f}   -- no pose at this straddle --")
                rows.append({"design": name, "x_sep_mm": xsep, "y_sep_mm": ysep, "pose": False})
                continue
            c = ev["carry"]
            ext = ev["extend_mm"]
            ceiling = float(np.degrees(np.arcsin(min(1.0, ext / 1000.0 / args.straddle))))
            rows.append({"design": name, "x_sep_mm": xsep, "y_sep_mm": ysep, "pose": True,
                         "extend_mm": ext, "ceiling_deg": round(ceiling, 1), "carry": c})
            print(f"{name:16} {xsep:6.0f} {ysep:6.0f} {ext:6.1f}m {ceiling:7.1f}d "
                  f"{c['peak_cos']:6.3f} {c['final_cos']:6.3f} {c['final_z']:7.4f} "
                  f"{c['contacts']:4d} {c['force_N']:7.2f}  {'OK' if c['ok'] else 'dropped'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
    return 0


def _extension(m, qpos) -> float:
    """mm of radial reach left at a fitted grasp, minimum over the three fingers.

    68.11 mm is the straight-chain mount-to-pad reach (yaw link 20.75 + 20.75 + 26.61). What is
    left of it is what a fixed-contact rotation about the pinch axis has to spend."""
    import mujoco
    from morphohand.tools.keyframe_ik import FINGERS
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    worst = 1e9
    for f in FINGERS:
        now = float(np.linalg.norm(d.body(f"{f}_tip").xpos - d.body(f"{f}_yaw_frame").xpos))
        worst = min(worst, 0.06811 - now)
    return round(worst * 1000, 1)


if __name__ == "__main__":
    sys.exit(main())
