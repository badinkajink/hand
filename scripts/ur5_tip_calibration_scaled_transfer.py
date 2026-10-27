#!/usr/bin/env python3
"""Whether the v1 tips' presses serve v2 ID 1 (five N52 magnets) once each tip's board change is brought to v2 ID 1's
scale by the ratio of the magnets' strengths k fitted at rest (calib_rest.json). Owner, 2026-10-26: contact density
first. Unscaled, a GP of the v1 tips missed v2 ID 1's contacts by 2.52 mm over 1.5 N (lab, data/20261027-v1_v2_tips.json).

  transfer  a GP trained on the v1 tips' presses (cb1t, cb1r, cb2t, cb3t), unscaled and scaled, on v2 ID 1's presses.
  thinned   v2 ID 1's own presses in five folds by action, its training thinned to 12, 24 and 48 actions or kept whole,
            alone and with the scaled v1 presses added: location error (mm, along the surface) over and under 1.5 N
            and the force error over 1.5 N (N).

    .venv/bin/python scripts/ur5_tip_calibration_scaled_transfer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_tip_calibration_cb1r_checks as K  # noqa: E402

D = K.EXP / "data"
V1 = ("20261023-203938-cb1t", "20261025-114653-cb1r", "20261025-152920-cb2t", "20261026-134522-cb3t")
V2 = "20261026-192158-v2id1"


def presses(s):
    H = K.load(D / s)
    return K.sub(H, H["kind"] == "press"), json.loads((D / s / "calib_rest.json").read_text())["k"]


def score(T, Y):
    hi = T["f_n"] >= 1.5
    e = K.loc_err(T, Y[:, :3])
    fe = np.linalg.norm(Y[:, 3:] - T["f"], axis=1)
    return dict(over_1p5N=K.med(e[hi]), under_1p5N=K.med(e[~hi]), force_over_1p5N=K.med(fe[hi]))


def run():
    T, k2 = presses(V2)
    raw, scaled, ks = [], [], {V2: k2}
    for s in V1:
        H, k = presses(s)
        ks[s] = k
        raw.append(H)
        Hs = dict(H)
        Hs["d_board"] = H["d_board"] * (k2 / k)
        scaled.append(Hs)
    R, A = K.cat(*raw), K.cat(*scaled)
    res = dict(k=ks, n_v1_holds=int(len(A["id"])), n_v2_holds=int(len(T["id"])),
               v1_raw=score(T, K.gp(R).predict(T["d_board"])), v1_scaled=score(T, K.gp(A).predict(T["d_board"])))
    ids = T["id"].astype(str)
    u = np.unique(ids)
    rng = np.random.default_rng(0)
    fold = {a: i % 5 for i, a in enumerate(rng.permutation(u))}
    f = np.array([fold[a] for a in ids])
    res["n_actions"] = int(len(u))
    res["thinned"] = {}
    for n_own in (12, 24, 48, None):
        Y1, Y2 = np.full((len(ids), 6), np.nan), np.full((len(ids), 6), np.nan)
        for kf in range(5):
            tr = [a for a in u if fold[a] != kf]
            if n_own:
                tr = list(rng.choice(tr, size=min(n_own, len(tr)), replace=False))
            m, te = np.isin(ids, tr), f == kf
            Y1[te] = K.gp(K.sub(T, m)).predict(T["d_board"][te])
            Y2[te] = K.gp(K.cat(K.sub(T, m), A)).predict(T["d_board"][te])
        res["thinned"][str(n_own or "all")] = dict(own=score(T, Y1), own_and_v1_scaled=score(T, Y2))
    out = D / V2 / "scaled_transfer.json"
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    run()
