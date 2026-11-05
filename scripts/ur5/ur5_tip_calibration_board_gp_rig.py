#!/usr/bin/env python3
"""The contact viewer's board GP for v3 IDs 1 to 3 with the inverted rig's presses of IDs 3 to 6 added (owner,
2026-11-05 00:4x: "the calibration rig is off, the hand is on the robot, there's no chance to calibrate these other
fingers anytime soon. i think we can live with the noisy calibration from ids3-6"). The rig reached the pad's mount end,
which tip mode did not (QUEUE R33).

The model is the viewer's (ur5_tip_calibration_board_gp: one GP shared by every board, inputs the 15 board components,
outputs the contact point in mm and the force in N, presses and slides). Training: the tip-mode holds of IDs 1 to 3 (the
sessions of 20261102-board_gp_v3_id1_id3.npz), ID 3's rig holds and 1,500 rig holds of each of IDs 4 to 6 drawn at
random (seed 0), 14,185 holds in all.

  validate  a fifth of each tip's tip-mode point groups held out (2 seeds): the viewer's recipe on tip mode alone against
            the same with IDs 4 to 6's rig holds added, both without ID 3's rig holds; tested on each tip's held-out holds
            and on ID 3's rig holds at the mount end (1 mm or more from every ID 3 tip-mode point)
  fit       the model on all of the training holds, saved in the viewer's format

    ~/.claude/bin/resguard.sh run --mem 7G --cpu 500 -- env OPENBLAS_NUM_THREADS=4 taskset -c 4-7 .venv/bin/python \\
        scripts/ur5/ur5_tip_calibration_board_gp_rig.py validate|fit
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401
import ur5_tip_calibration_board_gp as BG  # noqa: E402
import ur5_tip_calibration_rig_coverage as RC  # noqa: E402
import ur5_tip_calibration_v3_data_curve as V  # noqa: E402

KINDS = ["press", "slide"]
DATA = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data"
OUT = DATA / "20261105-board_gp_v3_id1_id3_rig_id3_id6.npz"
VAL = DATA / "20261105-board_gp_rig_validation.json"


def sets():
    tips = {k: RC.load(V.TIPS[k], KINDS, f"tip{k}") for k in (1, 2, 3)}
    rng = np.random.default_rng(0)
    rig = {}
    for j in (4, 5, 6):
        h = RC.load(RC.RIG[j], KINDS, f"rig{j}")
        i = np.sort(rng.choice(len(h["p"]), min(1500, len(h["p"])), replace=False))
        rig[j] = {f: v[i] for f, v in h.items()}
    rig[3] = RC.load(RC.RIG[3], KINDS, "rig3")
    return tips, rig


def XY(hs):
    return (np.concatenate([h["d_board"] for h in hs]), np.concatenate([np.c_[h["p"], h["f"]] for h in hs]))


def validate():
    tips, rig = sets()
    d3 = cKDTree(np.unique(np.round(tips[3]["p"], 3), axis=0)).query(rig[3]["p"])[0]
    end = d3 >= 1.0
    r3end = {f: v[end] for f, v in rig[3].items()}
    out = {"mount_end_holds": int(end.sum()), "rows": []}
    for seed in range(2):
        tr, te = [], {}
        for k, h in tips.items():
            g = np.random.default_rng(900 + seed).permutation(np.unique(h["group"]))
            test = np.isin(h["group"], g[: len(g) // 5])
            tr.append({f: v[~test] for f, v in h.items()})
            te[k] = {f: v[test] for f, v in h.items()}
        for name, hs in (("tip_mode", tr), ("tip_mode_and_rig_ids4_6", tr + [rig[4], rig[5], rig[6]])):
            X, Y = XY(hs)
            t0 = time.time()
            m = BG.fit(X, Y, RC.FLOOR)
            row = dict(seed=seed, model=name, n_train=int(len(X)), ell=m["ell"], noise=m["nz"],
                       seconds=round(time.time() - t0, 1))
            for k, h in te.items():
                row[f"id{k}"] = RC.errs(h, BG.predict(m, h["d_board"]), np.ones(len(h["p"]), bool))
            row["id3_mount_end"] = RC.errs(r3end, BG.predict(m, r3end["d_board"]), np.ones(len(r3end["p"]), bool))
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
            VAL.write_text(json.dumps(out, indent=1))


def fit():
    tips, rig = sets()
    hs = [tips[1], tips[2], tips[3], rig[3], rig[4], rig[5], rig[6]]
    names = ["tip1", "tip2", "tip3", "rig3", "rig4", "rig5", "rig6"]
    X, Y = XY(hs)
    t0 = time.time()
    m = BG.fit(X, Y, RC.FLOOR)
    print(f"fitted on {len(X)} holds in {time.time() - t0:.1f} s: length {m['ell']:.3f}, noise {m['nz']}", flush=True)
    sessions = list(V.TIPS[1]) + list(V.TIPS[2]) + list(V.TIPS[3]) + [s for j in (3, 4, 5, 6) for s in RC.RIG[j]]
    np.savez_compressed(OUT, **{k: np.asarray(v) for k, v in m.items()}, sessions=np.array(sessions),
                        kinds=np.array(KINDS), n_holds=len(X), sets=np.array(names),
                        n_by_set=np.array([len(h["p"]) for h in hs]),
                        rig_sample="1,500 rig holds of each of IDs 4 to 6, numpy default_rng(0) choice per tip in order 4, 5, 6")
    print("->", OUT)


if __name__ == "__main__":
    {"validate": validate, "fit": fit}[sys.argv[1]]()
