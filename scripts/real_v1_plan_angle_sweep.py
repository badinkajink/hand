#!/usr/bin/env python3
"""The bench maneuver's commanded angle on the calibrated plant.

    python3 scripts/real_v1_plan_angle_sweep.py --out docs/experiments/20260916-tip_gait

Replays a family of re-exported plans (same design, grasp, pivot and budget; commanded turn
60..110 deg) on the bench setting (fixed palm, tool on the 100 mm post) per plant, and
records the tool's signed cos, pad forces and retention at the end of the hold. The
deployed plan is included as the reference. Answers whether the compliant hand's 50 deg tip
follows a larger commanded arc or is the end of the mechanism.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from real_v1_tip_gait import HANDS, plant_scene  # noqa: E402
from real_v1_turn_mechanism import replay  # noqa: E402


def plate_variant(scene: Path, z_mm: float) -> Path:
    """The palm plate at `z_mm` above the mounting plane (the built hand: 25; every scene
    generated before 2026-09-16: 0). Only the plate moves."""
    if z_mm == 0:
        return scene
    import xml.etree.ElementTree as ET
    out = scene.with_name(scene.stem + f"__plate{z_mm:g}.xml")
    if out.exists():
        return out
    root = ET.parse(scene).getroot()
    for body in root.iter("body"):
        if body.get("name") == "palm_pose":
            for g in body.findall("geom"):
                if g.get("size", "").startswith("0.085"):
                    g.set("pos", f"0 0 {z_mm / 1000:g}")
    ET.ElementTree(root).write(out, encoding="unicode")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plans", default="docs/experiments/20260902-residual-bench/deploy/rv05_manual_b85_plan.json,"
                    "docs/experiments/20260916-tip_gait/deploy/rv05_manual_a60_plan.json,"
                    "docs/experiments/20260916-tip_gait/deploy/rv05_manual_a75_plan.json,"
                    "docs/experiments/20260916-tip_gait/deploy/rv05_manual_a90_plan.json,"
                    "docs/experiments/20260916-tip_gait/deploy/rv05_manual_a110_plan.json")
    ap.add_argument("--hands", default="", help="deployed plans by D-number instead of --plans")
    ap.add_argument("--plates", default="0", help="palm plate heights above the mounting plane, mm")
    ap.add_argument("--plants", default="fast,cal")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jitter-xy", type=float, default=0.001)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--tag", default="plan_angle_sweep", help="output JSON stem")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    scenes = a.out / "scenes"
    scenes.mkdir(exist_ok=True)
    rows = []
    out = a.out / (a.tag + ".json")
    plans = [ROOT / pp for pp in a.plans.split(",")] if not a.hands else [
        ROOT / "docs/experiments" / HANDS[h][1] / "deploy" / f"{HANDS[h][0]}_plan.json" for h in a.hands.split(",")]
    hand_of = {HANDS[h][0]: h for h in HANDS}
    for plan_p in plans:
        plan = json.loads(plan_p.read_text())
        tag = plan_p.name.replace("_plan.json", "")
        traj = plan_p.with_name(f"{tag}_traj.csv")
        base = Path(plan["meta"]["scene"])
        for pname in a.plants.split(","):
          for plate in (float(v) for v in a.plates.split(",")):
            scene = plate_variant(plant_scene(base, scenes, plan["meta"].get("design", tag), pname), plate)
            for s in range(a.seeds):
                vid = (a.out / "videos" / f"{tag}__{pname}_pl{plate:g}__s{s}.mp4") if a.video else None
                r = replay(scene, plan, traj if traj.exists() else None, post=True, gravity=True,
                           seed=1000 + s, jitter_xy=a.jitter_xy if s else 0.0, video=vid,
                           cam_view=(0.24, 110.0, -5.0))
                row = {"plan": tag, "hand": hand_of.get(tag, tag), "plate_mm": plate,
                       "angle_cmd": plan["meta"]["angle_deg"], "plant": pname, "seed": s,
                       **{k: r[k] for k in ("cos_grip", "cos_turn_end", "cos_end", "turn_deg", "z_grip",
                                            "z_end", "dropped", "f_pad_end", "n_pad_end", "f_post_end")},
                       "f_pad_end_by": r["trace"][-1]["f_pad"], "q_err_end": r["q_err_end"]}
                rows.append(row)
                print(f"{tag:18s} {pname:5s} plate{plate:<3g} s{s}: cos grip {r['cos_grip']:+.2f} -> end {r['cos_end']:+.2f} "
                      f"({np.degrees(np.arccos(np.clip(r['cos_end'], -1, 1))):.0f} deg from vertical)  "
                      f"pads {r['n_pad_end']} {r['f_pad_end']:.2f} N  {'DROPPED' if r['dropped'] else 'held'}",
                      flush=True)
                out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
