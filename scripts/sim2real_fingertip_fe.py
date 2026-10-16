#!/usr/bin/env python3
"""Finite elements of the 85A magnet blocks for the deployment-readiness study (docs/handoff/
20261016-sim2real_fingertip_readiness.md, steps 3-4).

scripts/tpu_tip_magnet_layouts.py's model of the whole block (two 0.45 mm walls and four 0.2 mm skins of solid TPU around
every outline, the pockets' included, over a homogenized gyroid core; bonded NdFeB magnets; back and mount faces fixed) on
its grid, with the 85A recipes of scripts/tpu_tip_magnet.py (solid modulus 12 MPa, NinjaFlex 85A's tensile modulus, the
proxy for Siraya's unmeasured value; Poisson 0.45; gyroid 30 % or 50 %) and the 3 x 2 mm layouts of
scripts/metric_magnet_prints.py: four, five, and four with one wall at each pocket. Loads as in that script: the
screwdriver shaft at the contact point (0.5, 1, 3 N: approach and pad modulus E*) and 1 N patches at 15 face positions
(normal, along y, along z: face compliance and each magnet's motion).

With the magnet effectively rigid (160 GPa against 12 MPa) every displacement scales as 1 / E_s, so E* at another solid
modulus is E* (E_s / 12 MPa); `scale` tabulates that.

Rows: docs/experiments/20261016-sim2real_fingertip_readiness/fe/fem.jsonl (one fsynced row per layout and recipe).

    ~/.claude/bin/resguard.sh run --mem 10G --cpu 500 -- taskset -c 14-17 ~/miniconda3/envs/femtip/bin/python \\
        scripts/sim2real_fingertip_fe.py run --recipes w85_30 w85_50 --layouts four five four_w1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import metric_magnet_prints as MP  # noqa: E402

D = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness/fe"
ROWS = D / "fem.jsonl"
DEPTH = 5.0


def done():
    if not ROWS.exists():
        return set()
    return {(r["layout"], r["recipe"]) for r in (json.loads(l) for l in open(ROWS))}


def cmd_run(a):
    import tpu_tip_magnet as TM
    import tpu_tip_magnet_layouts as TL
    have = done()
    D.mkdir(parents=True, exist_ok=True)
    for recipe in a.recipes:
        for lay in a.layouts:
            walls = 1 if lay.endswith("_w1") else 2
            base = lay[:-3] if walls == 1 else lay
            name = f"{base}_d3x2_d{DEPTH:g}" + ("_w1" if walls == 1 else "")
            if (name, recipe) in have:
                print(name, recipe, "done", flush=True)
                continue
            TL.LAYOUTS[name] = MP.fe_layout(name)
            TL.RECIPE = TM.RECIPES[recipe]
            real = TL.classes
            TL.classes = lambda l, w=walls: MP.classes_walls(l, w)
            tmp = Path(tempfile.mkdtemp(dir=ROOT / "logs/20261016-sim2real_fingertip_readiness")) / "row.jsonl"
            try:
                TL.run(name, out=tmp)
            finally:
                TL.classes = real
            row = json.loads(tmp.read_text().splitlines()[-1])
            row.update(recipe=recipe, recipe_params=TM.RECIPES[recipe], pocket_walls=walls,
                       script="scripts/sim2real_fingertip_fe.py (tpu_tip_magnet_layouts.run)")
            with open(ROWS, "a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            tmp.unlink()
            tmp.parent.rmdir()
            print(f"{name} {recipe}: E* {row['E_best_MPa']:.2f} MPa, approach at 1 N {row['per_force']['1']['delta_um']:.1f} um",
                  flush=True)


def cmd_scale(a):
    rows = [json.loads(l) for l in open(ROWS)]
    E_s = {"w85_30": 12.0, "w85_50": 12.0, "w95_20": 30.0}
    for r in rows:
        e0 = E_s[r["recipe"]]
        print(r["layout"], r["recipe"], " ".join(f"E_s {e:g}: E* {r['E_best_MPa'] * e / e0:.2f}" for e in a.moduli))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--recipes", nargs="+", default=["w85_30", "w85_50"])
    r.add_argument("--layouts", nargs="+", default=["four", "five", "four_w1"])
    s = sub.add_parser("scale")
    s.add_argument("--moduli", nargs="+", type=float, default=[6.0, 9.0, 12.0, 18.0, 24.0])
    a = ap.parse_args()
    {"run": cmd_run, "scale": cmd_scale}[a.cmd](a)


if __name__ == "__main__":
    main()
