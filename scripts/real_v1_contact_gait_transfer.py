#!/usr/bin/env python3
"""Replay contact-gait plans across tips, contact models, and grip jitters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from real_v1_contact_gait_search import Scene, hold_vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--include-2mm", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sources = [json.loads(path.read_text()) for path in args.plans]
    plans = {"hold": hold_vector().tolist()}
    for source in sources:
        label = source["tip"] + "_" + source["contact_model"]
        if source.get("pad_spacing_mm", 1.0) != 1.0:
            label += f'_{source["pad_spacing_mm"]:g}mm'
        if source.get("forced_release"):
            label += "_release"
        for method in ("direct_gradient", "cem", "hybrid_seed", "hybrid"):
            plans[label + "_" + method] = source["candidates"][method]["x"]
    rows = []
    destinations = [("sphere", "pt", 1.0), ("tpu6", "pt", 1.0),
                    ("tpu6", "padsT", 1.0)]
    if args.include_2mm:
        destinations.append(("tpu6", "padsT", 2.0))
    for tip, contact_model, spacing in destinations:
        for seed in args.seeds:
            scene = Scene(tip, contact_model, seed, pad_spacing_mm=spacing)
            destination = tip + "_" + contact_model + ("_2mm" if spacing == 2.0 else "")
            for label, x in plans.items():
                row = scene.evaluate(x, trace=True)
                rows.append({"destination":destination, "seed":seed,
                             "plan":label, **{key:row[key] for key in (
                                 "score", "vertical_cos", "cos_gain", "tool_z_m",
                                 "min_tool_z_m", "finger_normal_force_N", "held_final",
                                 "index_release_ms", "middle_release_ms",
                                 "index_recontact", "middle_recontact")}})
            print(f"{destination} seed {seed} complete", flush=True)
    output = {"mujoco_version":scene.mujoco.__version__,
              "source_files":[str(path) for path in args.plans],
              "seeds":args.seeds,"plans":plans,"rows":rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(f"Saved {len(rows)} rollouts to {args.out}", flush=True)


if __name__ == "__main__":
    main()
