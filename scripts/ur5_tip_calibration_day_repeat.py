#!/usr/bin/env python3
"""Last night's predictors on presses made a day later (QUEUE.md R21; owner, 2026-10-23: "we can run the 40 minute test
then"): every predictor of scripts/ur5_tip_calibration_predict.py trained on all presses of one session (the training
folder, the analysis output of sphere2) and scored on every press of a later session at the same targets (the test
folder, the analysis output of the day repeat placed with the same registration), against the training session's own
held-out score; and the change of the board's response between the two days at the same target and force level, against
the change between the training session's own repeats of its reference press.

Both folders come from scripts/ur5_tip_calibration_analysis.py. The physical model and the snap of predicted points use
the training session's rest fit (calib_rest.json), as a calibration made one night and used the next day would.

Same target and level: a hold of the later session is matched to the training session's hold of the same press id and
force level (a repeated reference press to the training session's ref-00). Reported are the board change's difference
over its size, |d_2 - d_1| / |d_1| (the 15 components, uT), the normal force's ratio f_2 / f_1, and the difference of
the registered indentation at that level, d_2 - d_1 (mm).

    .venv/bin/python scripts/ur5_tip_calibration_day_repeat.py <training folder> <test folder> [--out day_repeat.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import fingertip_wrench_model as W  # noqa: E402
import ur5_tip_calibration_predict as PR  # noqa: E402


def base_id(i):
    return "ref-00" if i.startswith("ref-") else i


def same_target(H1, H2):
    """Pairs (j1, j2) of holds at the same press target and force level."""
    key1 = {}
    for j, (i, L) in enumerate(zip(H1["id"], np.round(H1["level_N"], 3))):
        key1.setdefault((base_id(str(i)), float(L)), []).append(j)
    pairs = []
    for j2, (i, L) in enumerate(zip(H2["id"], np.round(H2["level_N"], 3))):
        k = (base_id(str(i)), float(L))
        if k in key1:
            pairs.append((key1[k][0], j2))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("train")
    ap.add_argument("test")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    H, _, cal0 = PR.load(a.train)
    Ht, _, _ = PR.load(a.test)
    basis = W.load_basis(PR.BASIS)
    phys0 = PR.Physical(cal0, basis)
    est_snap = phys0.estimator()
    te = np.arange(len(Ht["id"]))
    out = dict(train=str(a.train), test=str(a.test), n_train_presses=int(len(np.unique(H["id"]))),
               n_train_holds=int(len(H["id"])), n_test_presses=int(len(np.unique(Ht["id"]))), n_test_holds=int(len(te)),
               viewer=PR.scores(Ht, te, *phys0.predict(Ht, te)))
    row, ph = PR.models(H, np.arange(len(H["id"])), Ht, te, cal0, basis, est_snap)
    out["day_later"] = row
    own = Path(a.train) / "predict/predict.json"
    if own.exists():
        last = json.loads(own.read_text())["curve"][-1]
        out["same_night"] = {k: last[k] for k in ("N", "physical", "gp:board", "mlp:board", "physical+gp:board") if k in last}
    pairs = same_target(H, Ht)
    if pairs:
        j1, j2 = (np.array(x) for x in zip(*pairs))
        d1, d2 = H["d_board"][j1], Ht["d_board"][j2]
        rel = np.linalg.norm(d2 - d1, axis=1) / np.maximum(np.linalg.norm(d1, axis=1), 1e-9)
        fr = Ht["f_n"][j2] / H["f_n"][j1]
        dd = Ht["depth"][j2] - H["depth"][j1]
        lv = H["level_N"][j1]
        out["same_target"] = dict(
            n=int(len(pairs)), board_rel_change=dict(median=float(np.median(rel)), p90=float(np.percentile(rel, 90))),
            force_ratio=dict(median=float(np.median(fr))), depth_change_mm=dict(median=float(np.median(dd)),
                                                                                 p90_abs=float(np.percentile(np.abs(dd), 90))),
            by_level={f"{L:g}": dict(n=int((lv == L).sum()), board_rel_change=float(np.median(rel[lv == L])),
                                     depth_change_mm=float(np.median(dd[lv == L])))
                      for L in sorted(set(lv.tolist()))},
            refs=[dict(id=str(Ht["id"][b]), level_N=float(lv[k]), depth_change_mm=float(dd[k]), board_rel_change=float(rel[k]))
                  for k, b in enumerate(j2) if str(Ht["id"][b]).startswith("ref-")])
    # the same night's spread: the training session's own repeats of the reference press against its ref-00
    r1 = [(j1, j2) for j1, j2 in same_target(H, H) if str(H["id"][j2]).startswith("ref-") and str(H["id"][j2]) != "ref-00"]
    if r1:
        a1, a2 = (np.array(x) for x in zip(*r1))
        rr = np.linalg.norm(H["d_board"][a2] - H["d_board"][a1], axis=1) / np.maximum(np.linalg.norm(H["d_board"][a1], axis=1), 1e-9)
        ddr = H["depth"][a2] - H["depth"][a1]
        lv1 = H["level_N"][a1]
        out["same_night_refs"] = dict(n=int(len(r1)), by_level={f"{L:g}": dict(
            n=int((lv1 == L).sum()), board_rel_change=float(np.median(rr[lv1 == L])),
            depth_change_mm=float(np.median(ddr[lv1 == L]))) for L in sorted(set(lv1.tolist()))})
    print(f"trained on {out['n_train_presses']} presses ({out['n_train_holds']} holds) of {a.train}; scored on "
          f"{out['n_test_presses']} presses ({out['n_test_holds']} holds) of {a.test}")
    print(f"  viewer's estimator: {out['viewer']['loc_mm']['median']:.2f} mm, {out['viewer']['force_N']['median']:.3f} N")
    for k in ("physical", "gp:board", "mlp:board", "physical+gp:board"):
        r = row[k]
        s = out.get("same_night", {}).get(k)
        print(f"  {k:18s} a day later {r['loc_mm']['median']:.2f} mm, {r['force_N']['median']:.3f} N"
              + (f"; same night (held out, N = {out['same_night']['N']}) {s['loc_mm']['median']:.2f} mm, "
                 f"{s['force_N']['median']:.3f} N" if s else ""))
    if pairs:
        st = out["same_target"]
        print(f"  same target and level ({st['n']} holds): board change differs by {st['board_rel_change']['median'] * 100:.1f} % "
              f"of its size (median; p90 {st['board_rel_change']['p90'] * 100:.1f} %); force ratio {st['force_ratio']['median']:.3f}; "
              f"indentation {st['depth_change_mm']['median'] * 1e3:+.0f} um (median)")
        if "same_night_refs" in out:
            print("  the same night's reference presses against its ref-00 (median by level): " + "; ".join(
                f"{L} N board {v['board_rel_change'] * 100:.1f} %, indentation {v['depth_change_mm'] * 1e3:+.0f} um"
                for L, v in out["same_night_refs"]["by_level"].items()))
            print("  the later session's holds against the same target last night (median by level): " + "; ".join(
                f"{L} N board {v['board_rel_change'] * 100:.1f} %, indentation {v['depth_change_mm'] * 1e3:+.0f} um"
                for L, v in st["by_level"].items()))
        for r in st["refs"]:
            print(f"    {r['id']} at {r['level_N']:g} N: indentation {r['depth_change_mm'] * 1e3:+.0f} um, board {r['board_rel_change'] * 100:.1f} %")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
