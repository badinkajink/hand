#!/usr/bin/env python3
"""How a HOM design landscape changes with the finger servo and the pad modulus (2026-11-02; owner: "i wonder if we could
also characterize the change in landscape between the idealized 4nmrad plant, the current scs0009 calibration, and the
ideal sts3032 calibration. there seems to be no transfer between layouts, controls optimized for one plant to another. is
that really true?"; then: "seems like we need to queue up modulus studies across a range as well").

Reads the full grids at the default objects (625 symmetric tripods, up to 16 grasps per layout, seeds 0-2, GPU evaluator)
of one set of grids (SETS: the plants, the pad moduli on the SCS0009 plant, or plant x modulus at 2.5 and 10 MPa) and
compares every pair on the layouts both reached:
  - Spearman's rank correlation of the layouts' best-of-16 scores, and the share of one grid's best tenth of layouts
    that is in the other's best tenth;
  - each grid's best layout scored on the other grid (the score lost by using the other grid's choice);
  - the grasp: the share of layouts whose best grasp is the same on both, and the median score lost on the second grid
    by keeping the first grid's best grasp;
  - the median best-of-16 score and the share of layouts above 0.5 on each grid.
Then, for the grids of each subset in ROBUST, one design for all of them, chosen three ways:
  - own: each grid's best layout and grasp (a hand and grasp for each plant or modulus);
  - one layout: the layout whose worst grid scores highest, its grasp re-picked on each grid (the plant or modulus
    known, as after a calibration);
  - one layout and grasp: the layout and grasp whose worst grid scores highest (the plant or modulus unknown within the
    subset).

    python3 scripts/hom/hom_plant_transfer.py                  # plants: rows/20261102-plant_transfer.json
    python3 scripts/hom/hom_plant_transfer.py --set modulus    # rows/20261102-modulus_transfer.json
    python3 scripts/hom/hom_plant_transfer.py --set factorial  # rows/20261102-plant_modulus_factorial.json
    python3 scripts/hom/hom_plant_transfer.py --set stiffness --score track   # rows/20261110-servo_stiffness_track.json

--score track ranks by hom_landscape.track_score (2026-11-09, the owner's go): only grids whose rows carry the tracking
fields enter (rows from 11-10 on); --score rollout, the default, is hom_landscape.rollout_score of the earlier pages.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SCALE_HAND", "cad_v2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_landscape as L  # noqa: E402
from sr2hand.numerics.stats import spearman

ROWS = ROOT / "docs/experiments/20261029-foundational_pose_review/rows"
DEFAULT = (("twist", 40), ("roll", 40), ("pinpush", 25), ("press", 25))
WORK10, WORK25 = ("work plant, 10 MPa", "grid2dff_v4b"), ("work plant, 2.5 MPa", "grid2dff_v4b_w25")   # kp 4 N m/rad, 1 N m
SCS = {1.25: "grid2dff_v4b_svE1p25", 2.5: "grid2dff_v4b_sv25", 5: "grid2dff_v4b_svE5", 10: "grid2dff_v4b_svE10"}
SCS = {e: (f"SCS0009, {e:g} MPa", src) for e, src in SCS.items()}                                   # kp 0.72, 0.18 N m
STS25 = ("STS3032 tuned, 2.5 MPa", "grid2dff_v4b_k21")       # kp 2.1, 0.515 N m (the owner's estimate at 7 V)
STS = {1.25: "grid2dff_v4b_k21E1p25", 2.5: "grid2dff_v4b_k21", 5: "grid2dff_v4b_k21E5", 10: "grid2dff_v4b_k21E10"}
STS = {e: (f"STS3032, {e:g} MPa", src) for e, src in STS.items()}
STSMU = {0.4: "grid2dff_v4b_k21mu0p4", 0.7: "grid2dff_v4b_k21mu0p7", 1.0: "grid2dff_v4b_k21", 2.4: "grid2dff_v4b_k21mu2p4"}
STSMU = {mu: (f"STS3032, friction {mu:g}", src) for mu, src in STSMU.items()}   # at 2.5 MPa (owner 11-02 17:4x)
# post-reset grids (2026-11-10): the SCS0009 at the measured 0.55 N m/rad (kp0.55_kv0_fr0.18_fl0_dp0.011) and the 0.72
# grids rerun with the tracking fields, both on MuJoCo 3.6.0 and the 2.5 MPa scenes of grid2dff_v4b_sv25
SV72, SV72R = ("SCS0009 0.72, stored", "grid2dff_v4b_sv25"), ("SCS0009 0.72, rerun", "grid2dff_v4b_sv25r")
SV55 = ("SCS0009 0.55", "grid2dff_v4b_sv55")
SETS = {"plants": (WORK10, SCS[2.5], STS25),
        "modulus": (SCS[1.25], SCS[2.5], SCS[5], SCS[10]),
        "factorial": (WORK10, WORK25, SCS[10], SCS[2.5]),
        "sts_modulus": (STS[1.25], STS[2.5], STS[5], STS[10]),
        "sts_friction": (STSMU[0.4], STSMU[0.7], STSMU[1.0], STSMU[2.4]),
        "stiffness": (SV72, SV72R, SV55)}
# subsets that get one design for all their grids; None = every grid of the set that has rows
ROBUST = {"plants": (None, (SCS[2.5][0], STS25[0])),
          "modulus": ((SCS[1.25][0], SCS[2.5][0], SCS[5][0]), None),          # the calibration's band, then 1.25-10 MPa
          "factorial": ((WORK25[0], SCS[2.5][0]), (SCS[2.5][0], SCS[10][0])),
          "sts_modulus": ((STS[1.25][0], STS[2.5][0], STS[5][0]), None),
          "sts_friction": ((STSMU[0.4][0], STSMU[0.7][0], STSMU[1.0][0]), None),   # bare PLA to grip tape, then all
          "stiffness": ((SV72R[0], SV55[0]),)}
OUT = {"plants": "20261102-plant_transfer.json", "modulus": "20261102-modulus_transfer.json",
       "factorial": "20261102-plant_modulus_factorial.json", "sts_modulus": "20261102-sts_modulus_transfer.json",
       "sts_friction": "20261102-sts_friction_transfer.json", "stiffness": "20261110-servo_stiffness.json"}
SCORE = {"rollout": L.rollout_score, "track": L.track_score}


def grasp_means(task, path, score="rollout"):
    """{call: {grasp: mean score over seeds}} from hom_gpu.py rows; {} for --score track on rows without the tracking
    fields."""
    out, tracked = {}, False
    for x in (path.read_text().splitlines() if path.exists() else []):
        if x.strip():
            r = json.loads(x)
            tracked = tracked or "track_err_rms_deg" in r
            out.setdefault(int(r["call"]), {}).setdefault(int(r.get("grasp", 0)), []).append(SCORE[score](task, r))
    if score == "track" and not tracked:
        return {}
    return {c: {g: float(np.mean(v)) for g, v in gs.items()} for c, gs in out.items()}


def xy(c):
    return (40 + 5 * (c // 25), 50 + 5 * (c % 25))    # the lattice, x_sep major (hom_ea.lattice)


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
    return {"layouts": len(calls), "spearman": round(spearman(best_a, best_b), 3),
            "top_tenth_kept": round(len(top_a & top_b) / k, 3),
            "best_a": {"call": ca, "xy": xy(ca), "on_a": round(max(ga[ca].values()), 3), "on_b": round(max(gb[ca].values()), 3)},
            "best_b": {"call": cb, "xy": xy(cb), "on_b": round(max(gb[cb].values()), 3), "on_a": round(max(ga[cb].values()), 3)},
            "same_best_grasp": round(float(np.mean(same)), 3),
            "median_loss_keeping_a_grasp_on_b": round(float(np.median(lost)), 3) if lost else None,
            "median_a": round(float(np.median(best_a)), 3), "median_b": round(float(np.median(best_b)), 3),
            "above_half_a": round(float(np.mean(best_a > 0.5)), 3), "above_half_b": round(float(np.mean(best_b > 0.5)), 3)}


def robust(grids):
    """One design for every grid of `grids` ({name: {call: {grasp: score}}}), on the layouts all of them reached."""
    names = list(grids)
    calls = sorted(set.intersection(*(set(g) for g in grids.values())))
    best = {n: {c: max(grids[n][c].values()) for c in calls} for n in names}
    lay = max(calls, key=lambda c: min(best[n][c] for n in names))

    def fixed(c):
        gs = sorted(set.intersection(*(set(grids[n][c]) for n in names)))
        return max(((min(grids[n][c][g] for n in names), g) for g in gs), default=(-1.0, None))

    fx = max(calls, key=lambda c: fixed(c)[0])
    fg = fixed(fx)[1]
    return {"grids": names, "layouts": len(calls),
            "own": {n: {"xy": xy(max(calls, key=best[n].get)), "score": round(max(best[n].values()), 3)} for n in names},
            "one_layout": {"xy": xy(lay), "grasp": {n: max(grids[n][lay], key=grids[n][lay].get) for n in names},
                           "score": {n: round(best[n][lay], 3) for n in names}},
            "one_layout_and_grasp": {"xy": xy(fx), "grasp": fg,
                                     "score": {n: round(grids[n][fx][fg], 3) for n in names} if fg is not None else None}}


def main():
    ap = argparse.ArgumentParser(description="pairwise transfer and one design across grids of the HOM landscapes")
    ap.add_argument("--set", choices=sorted(SETS), default="plants")
    ap.add_argument("--score", choices=sorted(SCORE), default="rollout")
    a = ap.parse_args()
    out = {"set": a.set, "score": a.score, "grids": {n: src for n, src in SETS[a.set]}, "pairs": {}, "robust": {}}
    for task, d in DEFAULT:
        grids = {name: grasp_means(task, ROWS / src / f"{task}_d{d}.jsonl", a.score) for name, src in SETS[a.set]}
        grids = {n: g for n, g in grids.items() if g}
        for (na, ga), (nb, gb) in itertools.combinations(grids.items(), 2):
            r = compare(task, ga, gb)
            out["pairs"][f"{task}_d{d}|{na}|{nb}"] = r
            print(f"{task:8s} {na:24s} -> {nb:24s} n {r['layouts']:3d}  rho {r['spearman']:+.2f}  top10% kept "
                  f"{r['top_tenth_kept']:.2f}  best(A) {r['best_a']['xy']} {r['best_a']['on_a']:.2f}->{r['best_a']['on_b']:.2f}  "
                  f"best(B) {r['best_b']['xy']} {r['best_b']['on_b']:.2f} (on A {r['best_b']['on_a']:.2f})  same grasp "
                  f"{r['same_best_grasp']:.2f}  med {r['median_a']:.2f}/{r['median_b']:.2f}  >0.5 "
                  f"{r['above_half_a']:.2f}/{r['above_half_b']:.2f}", flush=True)
        for sub in ROBUST[a.set]:
            names = list(grids) if sub is None else list(sub)
            if len(names) < 2 or any(n not in grids for n in names):
                continue
            r = robust({n: grids[n] for n in names})
            out["robust"][f"{task}_d{d}|{'|'.join(names)}"] = r
            fs = r["one_layout_and_grasp"]["score"] or {}
            print(f"{task:8s} one design for {', '.join(names)} ({r['layouts']} layouts)")
            for n in names:
                print(f"           {n:24s} own {r['own'][n]['score']:.2f} at {r['own'][n]['xy']}   one layout "
                      f"{r['one_layout']['score'][n]:.2f} at {r['one_layout']['xy']} (grasp {r['one_layout']['grasp'][n]})   "
                      f"one layout and grasp {fs.get(n, float('nan')):.2f} at {r['one_layout_and_grasp']['xy']} "
                      f"(grasp {r['one_layout_and_grasp']['grasp']})", flush=True)
    p = ROWS / OUT[a.set]
    if a.score != "rollout":
        p = p.with_name(f"{p.stem}_{a.score}.json")
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
