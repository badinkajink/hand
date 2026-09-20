#!/usr/bin/env python3
"""A morphology run whose frozen scene carries the screw-tip mesh on the tool's -z end.

The calibrated morphology runs (`make_cal_morphology_run.py`) train on a plain 25 mm cylinder;
the chain's tool and the real screwdriver have the tip (+1 g, the centre of mass 2 mm towards the
tip). `policy_transfer_probe.variant_scene(..., "tipmesh", ...)` already builds that scene for the
zero-shot column; this copies a run directory (summary.json, best_rollout.npz, the plant scenes)
and puts the tip-mesh scene in as frozen_scene.xml, so a trainer can be pointed at it.

  uv run --extra rl python scripts/make_tipmesh_morphology_run.py \\
      --src results/phase1/real_v1/20260916-sv1_u0308_b050_cal --out results/phase1/real_v1/20260920-sv1_u0308_b050_cal_tip
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from policy_transfer_probe import variant_scene
    src, out = a.src.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith("frozen_scene__c"):
            shutil.copy(f, out / f.name)
    variant_scene(src / "frozen_scene.xml", "tipmesh", out)          # writes out/frozen_scene.xml
    s = json.load(open(out / "summary.json"))
    s["frozen_scene_xml"] = str(out / "frozen_scene.xml")
    s["tool"] = "screwdriver_medium with the screw-tip mesh (build_screw_scene tip)"
    s["derived_from"] = str(src)
    json.dump(s, open(out / "summary.json", "w"), indent=1)
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(out / "frozen_scene.xml"))
    b = m.body("screwdriver_medium")
    print(f"{out}: tool mass {float(b.mass[0]) * 1000:.2f} g, {m.nmesh} mesh, geoms on the tool "
          f"{sum(1 for g in range(m.ngeom) if m.geom_bodyid[g] == b.id)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
