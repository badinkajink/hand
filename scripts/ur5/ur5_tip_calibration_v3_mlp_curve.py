#!/usr/bin/env python3
"""Held-out error of the MLP ensemble on the v3 tips against the number of training holds, on the board GP's splits of
the data curves (owner, 2026-11-11: "updated charts describing the v3 fingertips error against number of points (loss
curve, with GP and MLP)").

The splits are those of scripts/ur5/ur5_tip_calibration_v3_data_curve.py: per tip the holds are grouped by point, each
seed holds out a fifth of the points, and the model is fitted on nested fractions of the other points. The GP rows are
read from the data curves of 2026-11-02 (v3 IDs 1 to 3 on the finger) and 2026-11-04 (v3 IDs 4 to 6 on the inverted rig,
pooled sessions); each MLP row is checked against the GP row of its seed and fraction for the same number of training
holds. The MLP is ur5_tip_calibration_predict.MLP (five members, two hidden layers of 64 SiLU units, AdamW), with the
step count chosen on a random fifth of the training rows as in ur5_tip_calibration_gp_vs_mlp.py, and the same outputs as
the GP: the contact point, the force and the torsion about the normal. Rows are appended and fsynced as they finish.

    ~/.claude/bin/resguard.sh run --mem 2G --cpu 200 -- env OMP_NUM_THREADS=1 .venv/bin/python \\
        scripts/ur5/ur5_tip_calibration_v3_mlp_curve.py run --tip 1
    .venv/bin/python scripts/ur5/ur5_tip_calibration_v3_mlp_curve.py collect
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import ur5_tip_calibration_predict as PR  # noqa: E402
import ur5_tip_calibration_v3_data_curve as V  # noqa: E402

DATA = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data"
GP_CURVES = {"finger": DATA / "20261102-v3_data_curve.json", "rig": DATA / "20261104-v3_id456_data_curve_pooled.json"}
ROWS = DATA / "20261111-v3_mlp_curve_rows"
OUT = DATA / "20261111-v3_gp_mlp_curve.json"


def fit_mlp(X, Y, floor, threads):
    """PR.MLP fitted with `threads` torch threads (its fit sets two), so that six tips train side by side."""
    import torch
    m = PR.MLP()
    orig = torch.set_num_threads
    torch.set_num_threads = lambda n: orig(threads)       # PR.MLP.fit sets two threads
    try:
        return m.fit(X, Y, floor)
    finally:
        torch.set_num_threads = orig


def cmd_run(a):
    tip = str(a.tip)
    rig = tip not in ("1", "2", "3")
    G = json.loads(GP_CURVES["rig" if rig else "finger"].read_text())
    g = G["tips"][tip]
    V.KINDS[:] = G["kinds"]
    H = V.load(g["sessions"])
    Y = np.c_[H["p"], H["f"], H["tn"]]
    assert len(Y) == g["holds"], (len(Y), g["holds"])
    floor = PR.floors(0)[:15]
    gp = {(r["seed"], r["fraction"]): r for r in g["rows"]}
    ROWS.mkdir(parents=True, exist_ok=True)
    path = ROWS / f"tip{tip}.jsonl"
    done = set()
    if path.exists():
        done = {(r["seed"], r["fraction"]) for r in map(json.loads, path.read_text().splitlines())}
    Gall = np.unique(H["group"])
    Gt = np.unique(H["group"][H["kind"].astype(str) == "twist"])
    Go = np.setdiff1d(Gall, Gt)
    t0 = time.time()
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)                  # the data curve's split, step for step
        gt, go = rng.permutation(Gt), rng.permutation(Go)
        kt, ko = max(1, len(gt) // 5), len(go) // 5
        test = np.isin(H["group"], np.r_[gt[:kt], go[:ko]])
        pool = rng.permutation(np.r_[gt[kt:], go[ko:]])
        for fr in G["fractions"]:
            train = np.isin(H["group"], pool[:max(2, int(round(fr * len(pool))))])
            ref = gp[(seed, fr)]
            assert int(train.sum()) == ref["n_train"], (tip, seed, fr, int(train.sum()), ref["n_train"])
            if (seed, fr) in done:
                continue
            t = time.time()
            perm = np.random.default_rng(1000 + seed).permutation(np.flatnonzero(train))
            m = fit_mlp(H["d_board"][perm], Y[perm], floor, a.threads)
            e = V.errors(H, m.predict(H["d_board"][test]), test)
            row = dict(tip=tip, seed=seed, fraction=fr, n_train=int(train.sum()), steps=int(m.steps),
                       seconds=round(time.time() - t, 1), **e)
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(f"ID {tip} seed {seed} fraction {fr:.2f}: {row['n_train']} holds, {row['steps']} steps; MLP "
                  f"{e['location_mm']:.2f} mm (GP {ref['location_mm']:.2f}), magnitude {e['magnitude']:.3f} "
                  f"(GP {ref['magnitude']:.3f}); {row['seconds']:.0f} s, total {time.time() - t0:.0f} s", flush=True)


def cmd_collect(a):
    out = dict(what=__doc__.split("\n\n")[1], sets={})
    for name, path in GP_CURVES.items():
        G = json.loads(path.read_text())
        s = dict(gp_source=str(path.relative_to(ROOT)), fractions=G["fractions"], kinds=G["kinds"], tips={})
        for tip, g in G["tips"].items():
            p = ROWS / f"tip{tip}.jsonl"
            mrows = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
            med = {}
            for fr in G["fractions"]:
                gr = [r for r in g["rows"] if r["fraction"] == fr]
                mr = [r for r in mrows if r["fraction"] == fr]
                seeds = sorted({r["seed"] for r in mr})
                gr = [r for r in gr if r["seed"] in seeds] if seeds else gr
                med[f"{fr:g}"] = {m: {k: float(np.median([r[k] for r in rr])) for k in
                                      ("n_train", "location_mm", "direction_deg", "magnitude")} if rr else None
                                  for m, rr in (("gp", gr), ("mlp", mr))}
                med[f"{fr:g}"]["seeds"] = seeds
            s["tips"][tip] = dict(holds=g["holds"], groups=g["groups"], sessions=g["sessions"], mlp_rows=mrows,
                                  gp_rows=[r for r in g["rows"]], median=med)
        out["sets"][name] = s
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    for name, s in out["sets"].items():
        for tip, d in s["tips"].items():
            line = []
            for fr, m in d["median"].items():
                if m["mlp"]:
                    line.append(f"{fr}: {m['gp']['n_train']:.0f} holds GP {m['gp']['location_mm']:.2f} MLP "
                                f"{m['mlp']['location_mm']:.2f}")
            print(f"{name} ID {tip}: " + "; ".join(line))
    print("->", OUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tip", type=int, required=True)
    r.add_argument("--seeds", type=int, default=4)
    r.add_argument("--threads", type=int, default=1)
    sub.add_parser("collect")
    a = ap.parse_args()
    {"run": cmd_run, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
