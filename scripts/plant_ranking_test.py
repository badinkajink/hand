#!/usr/bin/env python3
"""Does the corrected plant make the simulator RANK the hands the bench ranked?

    python3 scripts/plant_ranking_test.py --trials 10 --out ranking.json

The transfer study's first pass found the simulator ordering eight built hands at
spearman +0.36 (p = 0.43) against their measured alignment -- no relationship at all, on seven
measurable designs.  That was scored against the shipped plant, whose finger actuators are ~60x
stiffer than the bench's and which never drops the shaft.  This re-scores the same designs, the
same plans, under the plant calibrated in docs/experiments/20260902-servo-sysid/, and asks
whether the ordering moves.

It is a result either way.  If the correlation rises, the simulator's failure to predict the
bench was a plant error and is now partly fixed.  If it does not, the plant was not what the
ranking was missing, and that is worth knowing before any more design search is run against it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plant_drop_gate as G  # noqa: E402

# measured on the bench, docs/experiments/20260901-real_v1-transfer-firstpass
BENCH = {"sv1_w6689_b060": 0.826, "sv1_w2360_b075": 0.797, "sv1_u1364_b080": 0.489,
         "sv1_u0060_b75": 0.912, "sv1_u0308_b050": 0.625, "rv05_manual_b85": 0.553,
         "sv1_w0099_b100": 0.721}


def spearman(x, y):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for p, i in enumerate(s):
            r[i] = p
        return r
    rx, ry = rk(x), rk(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy-dir", type=Path,
                    default=ROOT / "docs/experiments/20260829-real_v1_deploy/deploy")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--jitter-xy", type=float, default=0.004)
    ap.add_argument("--jitter-yaw", type=float, default=0.09)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    variants = {"shipped": ["--kp", "30", "--forcerange", "10", "--frictionloss", "0", "--no-mass"],
                "corrected": ["--kp", "0.5", "--forcerange", "0.35", "--frictionloss", "0.0035"]}
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for tag, bench_cos in BENCH.items():
            plan = json.loads((a.deploy_dir / f"{tag}_plan.json").read_text())
            base = Path(plan["meta"]["scene"])
            row = {"bench": bench_cos}
            for vname, flags in variants.items():
                scene = Path(td) / f"{tag}_{vname}.xml"
                subprocess.run([sys.executable, str(ROOT / "scripts/apply_measured_plant.py"),
                                "--scene", str(base), "--out", str(scene)] + flags,
                               check=True, capture_output=True)
                ts = []
                for s in range(a.trials):
                    try:
                        ts.append(G.run_trial(scene, plan, a.jitter_xy, a.jitter_yaw, 2000 + s,
                                              traj=a.deploy_dir / f"{tag}_traj.csv"))
                    except Exception:
                        pass
                held = [t for t in ts if not t["dropped"]]
                row[vname] = {
                    "n": len(ts), "retention": len(held) / max(1, len(ts)),
                    "cos_held": float(np.mean([t["cos"] for t in held])) if held else 0.0}
            out[tag] = row
            print(f"{tag:<18} bench {bench_cos:.3f} | shipped {row['shipped']['cos_held']:.3f} "
                  f"(ret {row['shipped']['retention']:.2f}) | corrected "
                  f"{row['corrected']['cos_held']:.3f} (ret {row['corrected']['retention']:.2f})",
                  flush=True)

    tags = list(out)
    b = [out[t]["bench"] for t in tags]
    print(f"\nn = {len(tags)} designs")
    stats = {}
    for v in ("shipped", "corrected"):
        s = [out[t][v]["cos_held"] for t in tags]
        rho = spearman(b, s)
        pear = float(np.corrcoef(b, s)[0, 1])
        stats[v] = {"spearman": rho, "pearson": pear}
        print(f"  {v:<10} spearman {rho:+.3f}   pearson {pear:+.3f}")
    print(f"\n(the transfer first-pass reported spearman +0.36, p = 0.43, on these same seven)")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"designs": out, "stats": stats}, indent=1))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
