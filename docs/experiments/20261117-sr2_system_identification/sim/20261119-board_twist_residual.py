"""Can the fingertip boards sense torsion, a twist about the contact normal? (the owner, 2026-11-19 18:4x: "pose =
location, direction, and ideally orientation, although all of our testing has said that rotational components are
impossible to detect").

The UR5 sphere sessions of the hand's three tips (10-28 to 10-30) end with twist groups: at one place and normal load
(1, 2.5, 5 or 6 N) the arm turns the tip about the contact normal 0, 5, 10, 15 (and 20) deg and back, holding at each.
The HEX-E's moment about the normal (the holds' tn) wanders by +-4 N mm from hold to hold without following the turn,
so it cannot be the truth here; the turn angle is.

Per group: a local linear model of the field, d = a + J f (15 components; f the HEX-E's force in the pad frame), fitted
to the same session's presses and slides within 1.2 mm of the group's place; its residual RMS is the repeat noise. The
twist holds' residuals at each turn minus the mean residual at turn 0 (the first and last holds of the group, which
brackets any drift): its size, its part common to the five chips (a field fixed in the lab turning with the board shows
there) and the rest (chip to chip). Torsion of the pad would grow with the load: a stuck patch's torque with a^3, about 5x
from 1 to 5 N, a slipping one's with F^(4/3), about 8x. A field fixed in the lab, seen by a turning board, grows with the
turn alone.

    /usr/bin/python3 sim/20261119-board_twist_residual.py [--out data/s3/20261119-board_twist_residual.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
SESSIONS = {1: "logs/20261028-spherev3id1_twists", 2: "logs/20261029-spherev3id2", 3: "logs/20261030-spherev3id3"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=1.2)
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261119-board_twist_residual.json"))
    a = ap.parse_args()
    rows = []
    for tip, s in SESSIONS.items():
        z = np.load(ROOT / s / "holds.npz", allow_pickle=True)
        k, ids = z["kind"].astype(str), z["id"].astype(str)
        ok = np.isfinite(z["d_board"]).all(1) & np.isfinite(z["p"]).all(1) & np.isfinite(z["f"]).all(1)
        D, P, F, T, TN, FN = z["d_board"], z["p"], z["f"], z["turn_deg"], z["tn"], z["f_n"]
        base = np.where(ok & np.isin(k, ["press", "slide"]))[0]
        for g in sorted(set(ids[(k == "twist") & ok])):
            ii = np.where((ids == g) & ok)[0]
            near = base[np.linalg.norm(P[base] - P[ii].mean(0), axis=1) < a.radius]
            if len(near) < 25:
                continue
            A = np.c_[np.ones(len(near)), F[near]]
            sol, *_ = np.linalg.lstsq(A, D[near], rcond=None)
            noise = float(np.sqrt(np.mean((D[near] - A @ sol) ** 2)))
            R = D[ii] - np.c_[np.ones(len(ii)), F[ii]] @ sol
            t = T[ii]
            r0 = R[t == 0].mean(0)
            for th in sorted(set(t[t > 0])):
                dR = (R[t == th].mean(0) - r0).reshape(5, 3)
                m = dR.mean(0)
                rows.append(dict(tip=tip, group=g, load_N=round(float(FN[ii].mean()), 2), turn_deg=float(th),
                                 n_turn=int((t == th).sum()), n_zero=int((t == 0).sum()), n_local=int(len(near)),
                                 noise_uT=round(noise, 3), size_uT=round(float(np.linalg.norm(dR)), 3),
                                 common_uT=round(float(np.linalg.norm(m)), 3),
                                 chip_to_chip_uT=round(float(np.sqrt(np.mean(np.sum((dR - m) ** 2, 1)))), 3),
                                 expected_noise_uT=round(noise * float(np.sqrt(15 * (1 / (t == th).sum()
                                                                                      + 1 / (t == 0).sum()))), 3),
                                 tn_Nmm=round(float(TN[ii][t == th].mean() - TN[ii][t == 0].mean()), 2),
                                 field_uT=round(float(np.linalg.norm(D[ii], axis=1).mean()), 1)))
    out = dict(what=__doc__.split("\n\n")[0].replace("\n", " "), script=str(Path(__file__).resolve().relative_to(ROOT)),
               rows=rows)
    by = {}
    for r in rows:
        by.setdefault((r["turn_deg"], round(r["load_N"])), []).append(r["size_uT"])
    out["by_turn_and_load"] = [dict(turn_deg=t, load_N=l, n=len(v), size_uT_median=round(float(np.median(v)), 2))
                               for (t, l), v in sorted(by.items())]
    Path(a.out).write_text(json.dumps(out, indent=1))
    for r in out["by_turn_and_load"]:
        print(r)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
