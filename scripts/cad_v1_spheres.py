#!/usr/bin/env python3
"""Collision spheres for the cad_v1 finger: MorphIt on every CAD part, combined per body.

Each part mesh of assets/mjcf/experimental/20261013-cad_v1/meshes/ (metres, in its body frame) is written in
millimetres and packed by MorphIt (Nechyporenko et al., 2025; clone and venv in /home/humanoid/Programs/MorphIt, driven
through its hand_trials/pack_link.py) at its own sphere budget. A body's collision geometry is the union of its parts'
spheres: the carriage with the yaw servo, the yaw link with the MCP servo, the MCP link with the PIP servo, the DIP
link. Packing link and servo apart keeps MorphIt's inside tests off the faces they share. The TPU tip keeps the
repository's 1 mm pads and is not packed here.

Fidelity per body is measured against the body's meshes with MorphIt's hand_trials/sphere_fidelity.py definitions
(coverage, protrusion, surface fit; see the result page's Terms and metrics).

Outputs, in the model folder: spheres.json {body: [[x, y, z, r], ...]} (metres, body frame, index finger) and
spheres_metrics.json. Then rebuild the model with
    .venv/bin/python scripts/build_cad_v1_finger.py --spheres assets/mjcf/experimental/20261013-cad_v1/spheres.json

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 500 -- .venv/bin/python scripts/cad_v1_spheres.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_cad_v1_finger import BODY_PARTS, OUT  # noqa: E402

MORPHIT = Path("/home/humanoid/Programs/MorphIt")
PY = MORPHIT / "venv-morphit/bin/python"
WORK = REPO / "logs/20261013-cad_v1_spheres_work"  # MorphIt runs, logs and per-part metrics (gitignored)
# spheres requested per part. Owner, 2026-10-13: "it's ok if the collision mesh is bigger than the visual mesh ... speed is
# more important", so the default is MorphIt-B (volume-covering, larger spheres) at small budgets; --tight packs
# MorphIt-S at four times the budget for a close surface fit.
BUDGET = {"carriage": 24, "servo_yaw": 12, "yaw_link": 32, "servo_mcp": 12, "mcp_link": 40, "servo_pip": 12, "dip_link": 20}


def run(cmd, log):
    with open(log, "w") as f:
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(MORPHIT))
    if r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(map(str, cmd))}; see {log}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="MorphIt-B")
    ap.add_argument("--tight", action="store_true", help="MorphIt-S at four times the budgets, written as spheres_tight*.json")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale-budget", type=float, default=1.0, help="multiply every part's budget")
    ap.add_argument("--set", nargs="*", default=[], help="MorphIt config overrides passed to pack_link.py")
    args = ap.parse_args()
    stem = "spheres"
    if args.tight:
        args.variant, args.scale_budget, stem = "MorphIt-S", 4.0 * args.scale_budget, "spheres_tight"
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)
    body_of = {p: b for b, ps in BODY_PARTS.items() for p in ps if p != "tip"}
    spheres = {b: [] for b in set(body_of.values())}
    part_rows = {}
    for p, b in body_of.items():
        m = trimesh.load(OUT / "meshes" / f"{p}.stl")
        m.apply_scale(1000.0)
        stl = WORK / f"{p}.stl"
        m.export(stl)
        n = max(4, int(round(BUDGET[p] * args.scale_budget)))
        t0 = time.time()
        run([PY, MORPHIT / "hand_trials/pack_link.py", "--mesh", stl, "--budgets", n, "--variant", args.variant,
             "--iterations", args.iterations, "--seed", args.seed, "--out", WORK, "--threads", 4, *(["--set", *args.set] if args.set else [])],
            WORK / "logs" / f"{p}.log")
        dt = time.time() - t0
        js = sorted((WORK / "spheres").glob(f"{p}_{args.variant}_n{n}*.json"), key=lambda q: q.stat().st_mtime)[-1]
        S = json.loads(js.read_text())
        C = np.asarray(S["centers"] if "centers" in S else S["centres"], float)
        R = np.asarray(S["radii"], float).ravel()
        spheres[b] += [[*c, float(r)] for c, r in zip(C, R)]
        part_rows[p] = dict(body=b, requested=n, spheres=int(len(R)), seconds=round(dt, 1), json=str(js))
        print(f"{p}: {len(R)} spheres in {dt:.0f} s", flush=True)
    (OUT / f"{stem}.json").write_text(json.dumps(spheres))
    # fidelity per part: each part's spheres against that part's mesh (one closed shell). A body's link and servo
    # share flush faces, and a signed distance to the two shells concatenated misjudges points inside the servo near
    # those faces, so bodies are summarised from their parts: protrusion is bounded by the worst part's.
    rows = {}
    for p, v in part_rows.items():
        mj = WORK / f"part_{p}_metrics.jsonl"
        if mj.exists():
            mj.unlink()
        run([PY, MORPHIT / "hand_trials/sphere_fidelity.py", "--mesh", WORK / f"{p}.stl", "--spheres", v["json"], "--out", mj],
            WORK / "logs" / f"fidelity_{p}.log")
        r = json.loads(mj.read_text().splitlines()[-1])
        rows[p] = dict(body=v["body"], spheres=v["spheres"], coverage=r["coverage"], protrusion_p95_mm=r["protrusion_p95_mm"],
                       protrusion_max_mm=r["protrusion_max_mm"], fit_1mm=r["fit_within_1p0mm"], fit_0p5mm=r["fit_within_0p5mm"],
                       seconds=v["seconds"], volume_mm3=r["mesh_volume_mm3"], area_mm2=r["mesh_area_mm2"])
        print(p, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in rows[p].items()}, flush=True)
    bodies = {}
    for b in sorted(spheres):
        ps = [rows[p] for p in rows if rows[p]["body"] == b]
        vol = sum(r["volume_mm3"] for r in ps)
        area = sum(r["area_mm2"] for r in ps)
        bodies[b] = dict(spheres=sum(r["spheres"] for r in ps), coverage=sum(r["coverage"] * r["volume_mm3"] for r in ps) / vol,
                         protrusion_p95_mm=max(r["protrusion_p95_mm"] for r in ps), protrusion_max_mm=max(r["protrusion_max_mm"] for r in ps),
                         fit_1mm=sum(r["fit_1mm"] * r["area_mm2"] for r in ps) / area, fit_0p5mm=sum(r["fit_0p5mm"] * r["area_mm2"] for r in ps) / area,
                         seconds=sum(r["seconds"] for r in ps))
    meta = dict(variant=args.variant, iterations=args.iterations, seed=args.seed, budgets=BUDGET, scale_budget=args.scale_budget,
                overrides=args.set, parts=rows, bodies=bodies)
    (OUT / f"{stem}_metrics.json").write_text(json.dumps(meta, indent=1))
    print("wrote", OUT / f"{stem}.json", OUT / f"{stem}_metrics.json")


if __name__ == "__main__":
    main()
