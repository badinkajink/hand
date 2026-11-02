#!/usr/bin/env python3
"""How a HOM design landscape changes with the finger servo (2026-11-02; owner: "i wonder if we could also characterize the
change in landscape between the idealized 4nmrad plant, the current scs0009 calibration, and the ideal sts3032 calibration.
there seems to be no transfer between layouts, controls optimized for one plant to another. is that really true?").

Reads the full grids at the default objects (625 symmetric tripods, up to 16 grasps per layout, seeds 0-2, GPU evaluator)
of each plant in PLANTS and compares every pair on the layouts both reached:
  - Spearman's rank correlation of the layouts' best-of-16 scores, and the share of one plant's best tenth of layouts
    that is in the other's best tenth;
  - each plant's best layout scored on the other plant (the score lost by using the other plant's choice);
  - the grasp: the share of layouts whose best grasp is the same on both, and the median score lost on the second plant
    by keeping the first plant's best grasp;
  - the median best-of-16 score and the share of layouts above 0.5 on each plant.

    python3 scripts/hom_plant_transfer.py      # table on stdout; rows/20261102-plant_transfer.json
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SCALE_HAND", "cad_v2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_landscape as L  # noqa: E402

ROWS = ROOT / "docs/experiments/20261029-foundational_pose_review/rows"
DEFAULT = (("twist", 40), ("roll", 40), ("pinpush", 25), ("press", 25))
PLANTS = (("work plant, 10 MPa", "grid2dff_v4b"),          # kp 4 N m/rad, 1 N m (10-30)
          ("SCS0009, 2.5 MPa", "grid2dff_v4b_sv25"),      # kp 0.72, 0.18 N m (11-02)
          ("STS3032 tuned, 2.5 MPa", "grid2dff_v4b_k21"))  # kp 2.1, 0.515 N m (11-02, the owner's estimate at 7 V)


def grasp_means(task, path):
    """{call: {grasp: mean score over seeds}} from hom_gpu.py rows."""
    out = {}
    for x in (path.read_text().splitlines() if path.exists() else []):
        if x.strip():
            r = json.loads(x)
            out.setdefault(int(r["call"]), {}).setdefault(int(r.get("grasp", 0)), []).append(L.rollout_score(task, r))
    return {c: {g: float(np.mean(v)) for g, v in gs.items()} for c, gs in out.items()}


def rank(v):
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    r[o] = np.arange(len(v))
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(np.asarray(a)), rank(np.asarray(b)))[0, 1]) if len(a) > 2 else float("nan")


def compare(task, ga, gb):
    calls = sorted(set(ga) & set(gb))
    best_a = np.array([max(ga[c].values()) for c in calls])
    best_b = np.array([max(gb[c].values()) for c in calls])
    k = max(1, len(calls) // 10)
    top_a = {calls[i] for i in np.argsort(-best_a)[:k]}
    top_b = {calls[i] for i in np.argsort(-best_b)[:k]}
    ca, cb = calls[int(np.argmax(best_a))], calls[int(np.argmax(best_b))]
    same, lost = [], []
    for c in calls:
        gA = max(ga[c], key=ga[c].get)
        gB = max(gb[c], key=gb[c].get)
        same.append(gA == gB)
        if gA in gb[c]:
            lost.append(gb[c][gB] - gb[c][gA])
    xy = {c: (40 + 5 * (c // 25), 50 + 5 * (c % 25)) for c in (ca, cb)}    # the lattice, x_sep major (hom_ea.lattice)
    return {"layouts": len(calls), "spearman": round(spearman(best_a, best_b), 3),
            "top_tenth_kept": round(len(top_a & top_b) / k, 3),
            "best_a": {"call": ca, "xy": xy[ca], "on_a": round(max(ga[ca].values()), 3), "on_b": round(max(gb[ca].values()), 3)},
            "best_b": {"call": cb, "xy": xy[cb], "on_b": round(max(gb[cb].values()), 3), "on_a": round(max(ga[cb].values()), 3)},
            "same_best_grasp": round(float(np.mean(same)), 3),
            "median_loss_keeping_a_grasp_on_b": round(float(np.median(lost)), 3) if lost else None,
            "median_a": round(float(np.median(best_a)), 3), "median_b": round(float(np.median(best_b)), 3),
            "above_half_a": round(float(np.mean(best_a > 0.5)), 3), "above_half_b": round(float(np.mean(best_b > 0.5)), 3)}


def main():
    out = {}
    for task, d in DEFAULT:
        grids = {name: grasp_means(task, ROWS / src / f"{task}_d{d}.jsonl") for name, src in PLANTS}
        grids = {n: g for n, g in grids.items() if g}
        for (na, ga), (nb, gb) in itertools.combinations(grids.items(), 2):
            r = compare(task, ga, gb)
            out[f"{task}_d{d}|{na}|{nb}"] = r
            print(f"{task:8s} {na:24s} -> {nb:24s} n {r['layouts']:3d}  rho {r['spearman']:+.2f}  top10% kept "
                  f"{r['top_tenth_kept']:.2f}  best(A) {r['best_a']['xy']} {r['best_a']['on_a']:.2f}->{r['best_a']['on_b']:.2f}  "
                  f"best(B) {r['best_b']['xy']} {r['best_b']['on_b']:.2f} (on A {r['best_b']['on_a']:.2f})  same grasp "
                  f"{r['same_best_grasp']:.2f}  med {r['median_a']:.2f}/{r['median_b']:.2f}  >0.5 "
                  f"{r['above_half_a']:.2f}/{r['above_half_b']:.2f}", flush=True)
    p = ROWS / "20261102-plant_transfer.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
