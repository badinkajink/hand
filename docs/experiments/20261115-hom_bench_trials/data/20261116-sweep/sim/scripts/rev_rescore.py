"""Re-score of twist grasps on the real object (hand-4a, 2026-11-16 13:5x). The corvus session asked to "re-score the twist
landscape (grid2dff and grid2dgr at the two talk layouts, then wider) on the real object at mu ~0.5 with the server's
plant, and say which grasps/layouts rank best now". The owner (relayed by the corvus session) said the sphere carries rubbery tape
0.5-1 mm thick, so mu 0.5 is a fit and mu 1.0 is reported beside it. The variants test whether a tape layer (the ball's
radius 20.75 mm in the plant, the controller keeping its 20 mm model) or a softer tangential contact (impratio below the
scenes' 100) reproduce the film grasps' climb at mu 1.0.

Closed-loop HOM through rev_closed.run (hom_bench.simulate on sr2_hom_replay's CPU plant, seeds 0-2) on the stem scene of
rev_stem (ball 21.0 g, stem 36.9 mm, block 6.5 g, a table under the block until the lift), with the server's plant
(kp 0.55 N m/rad, damping 0.011, tau_max 0.18 N m). A run that fails in HOM's least squares (the sphere already dropped)
is kept with its error and scores J 0. Rows are appended and fsynced as they finish.

    OMP_NUM_THREADS=1 .venv/bin/python rev_rescore.py OUT.jsonl --grasps grids --grids grid2dff grid2dgr \\
        --calls 329 162 --mu 0.5 1.0 [--workers 6]
    OMP_NUM_THREADS=1 .venv/bin/python rev_rescore.py OUT.jsonl --grasps talk4 --mu 1.0 --tape-mm 0.75 --impratio 1
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path.home() / "Programs/hand"
D = ROOT / "logs/20261116-sim_real_reversal"
sys.path.insert(0, str(D)); sys.path.insert(0, str(ROOT / "scripts/hardware")); sys.path.insert(0, str(ROOT / "scripts"))
SERVER = {"kp": 0.55, "damping": 0.011}                 # hom_server's "measured" plant, kp0.55_kv0_fr0.18_fl0_dp0.011


def scene(tape_mm=0.0, impratio=None):
    """The stem scene (rev_stem.build's geometry) with the ball's radius grown by tape_mm (the stem, block and table moved
    down with the ball's bottom; the centre stays at the scene's start) and, if given, another impratio."""
    import rev_stem
    base = rev_stem.build()
    if not tape_mm and impratio is None:
        return base
    x = Path(base).read_text()
    r0, r = 0.020, 0.020 + tape_mm / 1000.0
    stem, bh = 0.03689, 0.017
    old = '<geom type="sphere" size="0.020000"'
    assert x.count(old) == 1
    x = x.replace(old, f'<geom type="sphere" size="{r:.6f}"')
    for name, (a, b) in (("obj_stem", (-r0 + 0.001, -(r0 + stem))), ("obj_block", (-(r0 + stem), -(r0 + stem) - bh))):
        pat = re.compile(rf'name="{name}" type="cylinder" fromto="0 {a:.5f} 0 0 {b:.5f} 0"')
        assert len(pat.findall(x)) == 1, name
        dy = r - r0
        x = pat.sub(f'name="{name}" type="cylinder" fromto="0 {a - dy:.5f} 0 0 {b - dy:.5f} 0"', x)
    m = re.search(r'<body name="post" pos="0 0 ([-0-9.]+)">', x)
    assert m
    x = x.replace(m.group(0), f'<body name="post" pos="0 0 {float(m.group(1)) - (r - r0):.5f}">')
    if impratio is not None:
        x, n = re.subn(r'impratio="[^"]*"', f'impratio="{impratio:g}"', x)
        assert n == 1
    out = D / "scenes" / f"sph040_stem_block_tape{tape_mm:g}_imp{impratio if impratio is not None else 100:g}.xml"
    out.write_text(x)
    return out


_G = {}


def init(path):
    import sr2_hom_replay as RP
    orig = RP.ref_scene

    def rs(shape, d):
        pads, pt = orig(shape, d)
        return (Path(path), pt) if shape == "sphere" else (pads, pt)
    RP.ref_scene = rs
    import rev_closed as RC
    RC._G["RP"] = RP
    _G["RC"] = RC


def run(job):
    return _G["RC"].run(job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--grasps", choices=("grids", "talk4"), default="grids")
    ap.add_argument("--grids", nargs="+", default=["grid2dff", "grid2dgr"])
    ap.add_argument("--calls", type=int, nargs="+", default=[329, 162])
    ap.add_argument("--mu", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tape-mm", type=float, default=0.0)
    ap.add_argument("--impratio", type=float, default=None)
    ap.add_argument("--tau-max", type=float, default=None, help="the servos' torque limit (N m; the plant's 0.18)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N jobs (a size check)")
    a = ap.parse_args()
    path = scene(a.tape_mm, a.impratio)
    import rev_closed as RC
    if a.grasps == "talk4":
        grasps = [(n, s, c, g) for (n, s, c, g) in RC.GRASPS]
    else:
        grasps = [(f"{s}_c{c}_g{g}", s, c, g) for s in a.grids for c in a.calls for g in range(16)]
    out = Path(a.out)
    done = set()
    if out.exists():                                        # resume: skip (grasp, seed, mu) already written on this scene
        for x in out.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                if r.get("scene") == path.name:
                    done.add((r["grasp"], r["seed"], json.dumps(r["params"], sort_keys=True)))
    extra_p = {"tau_max": a.tau_max} if a.tau_max else {}
    jobs = [(n, s, c, g, seed, {**SERVER, "mu": mu, **extra_p}) for mu in a.mu for (n, s, c, g) in grasps
            for seed in a.seeds if (n, seed, json.dumps({**SERVER, "mu": mu, **extra_p}, sort_keys=True)) not in done]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"scene {path.name}, {len(jobs)} runs ({len(done)} done before), {a.workers} workers", flush=True)
    import multiprocessing as mp
    extra = {"scene": path.name, "tape_mm": a.tape_mm, "impratio": a.impratio if a.impratio is not None else 100.0}
    with mp.get_context("fork").Pool(a.workers, initializer=init, initargs=(str(path),)) as pool, \
            out.open("a") as fh:
        for k, r in enumerate(pool.imap_unordered(run, jobs), 1):
            fh.write(json.dumps({**r, **extra}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if k % 24 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
