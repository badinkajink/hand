#!/usr/bin/env python3
"""Take chosen designs through every gate that stands between a screen row and a bench run.

    python3 scripts/real_v1_promote_designs.py \
        --table docs/experiments/20260830-real_v1-budget-rescreen/selected/selected_table.json \
        --manifest docs/experiments/20260830-real_v1-sobol4096/hardware_manifest.json \
        --generated-dir assets/mjcf/experimental/20260830-real_v1-sobol4096 \
        --designs sv1_u2699,sv1_w3408 \
        --out-dir docs/experiments/20260830-real_v1-budget-rescreen/deploy

THE GATES, in the order a failure is cheapest to find:

  1. export        the design's operating point becomes mounts + four set-points + a 50 Hz CSV
  2. servo range   `HandPlan.validate` -- can the driver be TOLD to reach every set-point
  3. gantry travel the same call, against the measured rails
  4. clearance     modelled finger-to-finger distance along BOTH the chord and the CSV

Passing all four is necessary, not sufficient: nothing here models the servo bodies, horns or
brackets, so a plan with a small margin between two PROXIMAL segments is worth less than the
same margin between two tips. The clearance report names the two bodies for that reason.

`--ship <dir>` copies the plans that pass all four into the station's plans directory. Plans
that fail are still written to --out-dir, with the reason, so the failure is inspectable.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/manta/host"))

import real_v1_deploy_envelope as de  # noqa: E402
import real_v1_design_search as ds  # noqa: E402
from manta_hand.plan import HandPlan  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, required=True,
                    help="rows with design, straddle_mm, thumb_axial_mm, depth_*_mm, squeeze_mm, "
                         "axis_k, angle_deg and (optionally) budget_rad")
    ap.add_argument("--manifest", type=Path, required=True, help="for the design vectors")
    ap.add_argument("--generated-dir", type=Path, required=True)
    ap.add_argument("--designs", default=None, help="comma list; default every row in --table")
    ap.add_argument("--designs-file", type=Path, default=None)
    ap.add_argument("--budget", type=float, default=None,
                    help="override every row's budget_rad, radians")
    ap.add_argument("--object", default="medium")
    ap.add_argument("--bench-height", type=float, default=0.100, help="metres")
    ap.add_argument("--post-y", type=float, default=-35.0, help="mm")
    ap.add_argument("--pad-width-mm", type=float, default=21.1)
    ap.add_argument("--pad-len-mm", type=float, default=14.8)
    ap.add_argument("--turn-steps", type=int, default=550)
    ap.add_argument("--min-clearance-mm", type=float, default=5.0)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ship", type=Path, default=None,
                    help="copy the plans that pass every gate into this directory")
    a = ap.parse_args()

    table = {r["design"]: r for r in json.loads(a.table.read_text())}
    vectors = {r["design"]: tuple(r["vector_m"])
               for r in json.loads(a.manifest.read_text())["designs"]}
    if a.designs_file:
        names = [t.strip() for t in a.designs_file.read_text().replace("\n", ",").split(",")
                 if t.strip()]
    elif a.designs:
        names = [t.strip() for t in a.designs.split(",") if t.strip()]
    else:
        names = sorted(table)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    verdicts = []
    for design in names:
        row = table.get(design)
        if row is None:
            verdicts.append({"design": design, "verdict": "no row in --table"})
            print(f"{design:14s} no row in {a.table.name}")
            continue
        budget = a.budget if a.budget is not None else row.get("budget_rad", 0.5)
        tag = f"{design}_b{round(budget * 100):03d}"
        scene = de.object_scene(ds.scene_for(vectors[design], a.generated_dir), a.object,
                                a.bench_height, a.post_y / 1000.0, True,
                                a.pad_len_mm / 1000.0, a.pad_width_mm / 1000.0, False)
        cmd = [sys.executable, "scripts/real_v1_export_plan.py",
               "--design", design, "--tag", tag, "--object", a.object,
               "--scene", str(scene), "--design-table", str(a.table),
               "--straddle-mm", str(row["straddle_mm"]),
               "--thumb-axial-mm", str(row["thumb_axial_mm"]),
               "--squeeze-mm", str(row.get("squeeze_mm", 10.0)),
               "--axis-k", str(row["axis_k"]), "--angle-deg", str(row["angle_deg"]),
               "--turn-steps", str(a.turn_steps), "--budget", str(budget),
               "--bench-height", str(a.bench_height), "--post-y", str(a.post_y),
               "--flat-pads", "--pad-width-mm", str(a.pad_width_mm),
               "--out", str(a.out_dir)]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode or not (a.out_dir / f"{tag}_plan.json").exists():
            verdicts.append({"design": design, "tag": tag, "budget_rad": budget,
                             "verdict": "export failed",
                             "detail": (r.stdout + r.stderr).strip()[-300:]})
            print(f"{tag:22s} EXPORT FAILED")
            continue

        bad = [str(v) for v in HandPlan.from_json(a.out_dir / f"{tag}_plan.json").validate()]
        clear = subprocess.run(
            [sys.executable, "scripts/real_v1_trajectory_clearance.py",
             "--deploy-dir", str(a.out_dir), "--plan", tag, "--substeps", str(a.substeps)],
            cwd=ROOT, capture_output=True, text=True)
        margins = []
        for line in clear.stdout.splitlines():
            parts = line.split()
            if len(parts) > 2 and parts[0] in ("chord", "csv"):
                margins.append((parts[0], float(parts[1]), parts[3]))
        worst = min((m[1] for m in margins), default=None)
        ok = not bad and worst is not None and worst >= a.min_clearance_mm
        verdicts.append({
            "design": design, "tag": tag, "budget_rad": budget, "scene": str(scene),
            "straddle_mm": row["straddle_mm"], "thumb_axial_mm": row["thumb_axial_mm"],
            "axis_k": row["axis_k"], "angle_deg": row["angle_deg"],
            "servo_violations": bad,
            "clearance_mm": {m[0]: m[1] for m in margins},
            "clearance_pair": margins[0][2] if margins else None,
            "verdict": "ship" if ok else ("servo range" if bad else "clearance"),
        })
        print(f"{tag:22s} clearance {worst if worst is not None else float('nan'):+6.1f} mm  "
              f"{'0 violations' if not bad else bad[0]}  -> {verdicts[-1]['verdict']}")

    (a.out_dir / "promotion.json").write_text(json.dumps(verdicts, indent=1) + "\n")
    shipped = [v for v in verdicts if v.get("verdict") == "ship"]
    if a.ship:
        a.ship.mkdir(parents=True, exist_ok=True)
        for v in shipped:
            for suffix in ("plan.json", "traj.csv", "poses.txt", "build.txt"):
                shutil.copy2(a.out_dir / f"{v['tag']}_{suffix}", a.ship / f"{v['tag']}_{suffix}")
        print(f"\ncopied {len(shipped)} plans into {a.ship}")
    print(f"\n{len(shipped)}/{len(verdicts)} pass every gate; wrote {a.out_dir}/promotion.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
