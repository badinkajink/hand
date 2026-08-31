#!/usr/bin/env python3
"""Render confirmed torque-capped proof-lift/hold rollouts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import real_v1_deploy_envelope as de  # noqa: E402
import real_v1_design_search as ds  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--generated-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    table = {row["design"]: row for row in json.loads(args.table.read_text())}
    manifest = json.loads(args.manifest.read_text())
    vectors = {row["design"]: tuple(row["vector_m"]) for row in manifest["designs"]}
    results = json.loads(args.results.read_text())
    eligible = [row for row in results
                if row.get("nom_kept", 0) >= max(1, 0.6 * row.get("n_nom", 1))
                and row.get("nom_cos", 0.0) >= 0.7
                and (row.get("min_finger_clearance_mm") or -1.0) >= 5.0]
    eligible.sort(key=lambda row: (row.get("ens_win", -1.0), row["nom_cos"],
                                   -row.get("max_proof_slip_mm", 1e9)), reverse=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for result in eligible[:args.top]:
        design = result["design"]
        row = table[design]
        scene = de.object_scene(ds.scene_for(vectors[design], args.generated_dir), "medium",
                                0.100, -0.035, True, 0.0148, 0.0211, False)
        plan = de.make_plan(scene, straddle=row["straddle_mm"] / 1000.0,
                            depth=row["depth_fit_mm"] / 1000.0,
                            thumb_axial=row["thumb_axial_mm"] / 1000.0,
                            squeeze=row["squeeze_mm"] / 1000.0,
                            axis_k=row["axis_k"], angle_deg=row["angle_deg"], lift=0.10,
                            budget=0.5, turn_steps=550, hold_squeeze=0.0, bench=True)
        if plan is None:
            rendered.append({"design": design, "error": "fit failed during render"})
            continue
        chosen = None
        for seed in range(10):
            trial = de.execute(
                scene, plan, hold_steps=2500, seed=seed, selfcollision=True,
                load_target_units=250, load_gain=0.0024, capture_steps=400,
                proof_lift=0.060, proof_lift_steps=700, proof_max_slip=0.010,
                turn_torque_limit_nm=de.SCS0009_OVERLOAD_TORQUE_NM,
                hold_torque_limit_nm=de.SCS0009_RATED_TORQUE_NM)
            if (trial["ok"] and trial["final_cos"] >= 0.7
                    and (trial["min_finger_clearance_mm"] or -1.0) >= 5.0):
                chosen = seed
                break
        if chosen is None:
            rendered.append({"design": design, "error": "no passing render seed"})
            continue
        video = args.out_dir / f"{design}_retention.mp4"
        trial = de.execute(
            scene, plan, hold_steps=2500, seed=chosen, video=video, selfcollision=True,
            load_target_units=250, load_gain=0.0024, capture_steps=400,
            proof_lift=0.060, proof_lift_steps=700, proof_max_slip=0.010,
            turn_torque_limit_nm=de.SCS0009_OVERLOAD_TORQUE_NM,
            hold_torque_limit_nm=de.SCS0009_RATED_TORQUE_NM)
        rendered.append({"design": design, "seed": chosen, "video": str(video), **trial})
        print(f"rendered {video}", flush=True)

    (args.out_dir / "renders.json").write_text(json.dumps(rendered, indent=2) + "\n")
    print(f"wrote {len(rendered)} render records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
