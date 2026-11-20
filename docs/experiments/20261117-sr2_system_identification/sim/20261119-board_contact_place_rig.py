"""How well the fingertip boards place a contact on the pad, where the place is known: the UR5 calibrations of the v3
tips held out session by session and tip by tip (the owner, 2026-11-19 18:1x: "again the forces are much lower priority
than contact pose").

The trials' estimator takes the contact point from the board GP (tip_contacts.BoardGP: the 15 field components to the
point and the 3D force, pooled over the v3 tips' UR5 sessions and the inverted rig's;
20261105-board_gp_v3_id1_id3_rig_id3_id6.npz). Here the same recipe (its 15 sessions, presses and slides, 1,500 rig holds
of each of IDs 4-6 drawn by default_rng(0), its length 1.94 and noise 0.3, the board's input floors) is fitted again
without one session, or without every session of one of the hand's tips (1 thumb, 2 middle, 3 index; tip 3's rig
session too), and predicts the holds left out: presses, slides, rolls and twists, at the place, normal and force that
the arm and the HEX-E gave them. Beside it:
  deployed   the deployed GP itself: in sample on its own sessions, and out of sample on the tall-sphere sessions of
             tips 2 and 3 (logs/20261029-tallv3id2, 20261030-tallv3id3), which it never saw
  pattern    the normal-only pattern match of sim/20261119-board_force_pattern_match.py: the best cosine of the field
             with the same tip's near-normal presses of 1-4 N in the training sessions, the median place of the 5 best
  force      the pad's point whose outward normal opposes the hold's force: where a frictionless contact would sit
Errors per hold: the place (the estimate projected onto the pad, mm), its parts along z and y, the normal's angle
(deg), the force's direction (deg) and its size ratio. Summaries by force, by lean |f_t|/f_n, by region along z and by
kind. The pad is sr2hand.control.surfaces.HalfEllipsoid(8.5, 7.4, 11, 22) posed in the holds' pad frame (apex at
(8.5, 0, 10.86), z from the proximal mount face).

    OPENBLAS_NUM_THREADS=4 /usr/bin/python3 sim/20261119-board_contact_place_rig.py [--out data/s3/20261119-board_contact_place_rig.json]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve, cholesky

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/ur5"), str(ROOT / "src")]
import _paths  # noqa: E402,F401
import ur5_tip_calibration_board_gp as BG  # noqa: E402
import ur5_tip_calibration_predict as PR  # noqa: E402
from sr2hand.control.surfaces import HalfEllipsoid, Posed  # noqa: E402

DATA = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data"
GP_FILE = DATA / "20261105-board_gp_v3_id1_id3_rig_id3_id6.npz"
RIG = "docs/experiments/20261030-inverted_rig_seat_block/data/holds/"
SETS = {"tip1": ["logs/20261028-spherev3id1_twists", "logs/20261028-rodv3id1"],
        "tip2": ["logs/20261029-spherev3id2", "logs/20261029-rodv3id2"],
        "tip3": ["logs/20261030-spherev3id3", "logs/20261030-rodv3id3"],
        "rig3": [RIG + "20261104-145215-id3pts11"],
        "rig4": [RIG + "20261102-220337-s3v4", RIG + "20261103-202017-rod32_dense"],
        "rig5": [RIG + "20261102-182710-s2v5", RIG + "20261103-144808-s2sph2", RIG + "20261103-172647-rod21_dense"],
        "rig6": [RIG + "20261102-145010-s1v6b", RIG + "20261103-122020-s1sph2b", RIG + "20261103-214753-rod13_dense"]}
TIP = {"tip1": 1, "tip2": 2, "tip3": 3, "rig3": 3, "rig4": 4, "rig5": 5, "rig6": 6}
TALL = {"logs/20261029-tallv3id2": 2, "logs/20261030-tallv3id3": 3}
EVAL_KINDS = ("press", "slide", "roll", "twist")
AX, AY, AZD, AZP, ZA = 8.5, 7.4, 11.0, 22.0, 10.86
PAD = Posed(HalfEllipsoid(AX, AY, AZD, AZP), pos=(0.0, 0.0, ZA), R=np.diag([1.0, -1.0, -1.0]))
F_BINS = [(0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 99.0)]
LEAN_BINS = [(0.0, 0.15), (0.15, 0.35), (0.35, 9.0)]
Z_BINS = [(-1.0, 6.0, "proximal end, z < 6"), (6.0, ZA, "proximal flat, 6 to 10.9"), (ZA, 16.0, "distal of apex, 10.9 to 16"),
          (16.0, 23.0, "distal flank and end, z > 16")]


def session_set(s):
    return next(k for k, v in SETS.items() if s in v)


def load(session, kinds, need_n=True):
    z = np.load(ROOT / session / "holds.npz", allow_pickle=True)
    k = z["kind"].astype(str)
    ok = np.isin(k, kinds) & np.isfinite(z["d_board"]).all(1) & np.isfinite(z["p"]).all(1) & np.isfinite(z["f"]).all(1)
    if need_n:
        ok &= np.isfinite(z["n"]).all(1)
    return dict(d=z["d_board"][ok], p=z["p"][ok], f=z["f"][ok], n=z["n"][ok], kind=k[ok])


def cat(hs):
    return {k: np.concatenate([h[k] for h in hs]) for k in hs[0]}


def training(exclude=()):
    """The deployed GP's training holds less the sessions in exclude: (X, Y, tip per hold, the holds)."""
    rng = np.random.default_rng(0)
    parts, tips = [], []
    for s, sessions in SETS.items():
        hs = [load(x, ("press", "slide"), need_n=False) for x in sessions if x not in exclude]
        if not hs:
            if s in ("rig4", "rig5", "rig6"):
                raise ValueError("the rig's sampled sets are never left out")
            continue
        h = cat(hs)
        if s in ("rig4", "rig5", "rig6"):
            i = rng.choice(len(h["d"]), 1500, replace=False)
            h = {k: v[i] for k, v in h.items()}
        parts.append(h)
        tips.append(np.full(len(h["d"]), TIP[s]))
    H = cat(parts)
    return H["d"], np.c_[H["p"], H["f"]], np.concatenate(tips), H


def fit_fixed(X, Y, floor, ell, nz):
    """BG.fit's GP at a given length and noise (no grid): the factor is formed in place."""
    mx, sx = X.mean(0), PR.scale(X, floor)
    my, sy = Y.mean(0), Y.std(0) + 1e-9
    Z, T = (X - mx) / sx, (Y - my) / sy
    K = BG.sqdist(Z, Z)
    K *= -0.5 / ell ** 2
    np.exp(K, out=K)
    K[np.diag_indices_from(K)] = 1.0 + nz ** 2
    L = cholesky(K, lower=True, overwrite_a=True, check_finite=False)
    a = cho_solve((L, True), T, check_finite=False)
    del K, L
    return dict(mx=mx, sx=sx, my=my, sy=sy, Z=Z, a=a, ell=float(ell))


def predict(m, X, chunk=2000):
    return np.vstack([BG.predict(m, X[i:i + chunk]) for i in range(0, len(X), chunk)])


def support(m):
    """The pad's surface point (pad frame) whose outward normal is m (unit)."""
    m = np.asarray(m, float) / np.linalg.norm(m)
    ax = np.array([AX, AY, AZD if m[2] >= 0 else AZP])
    p = ax ** 2 * m / math.sqrt(float(np.sum(ax ** 2 * m ** 2)))
    return np.array([p[0], p[1], ZA + p[2]])


def on_pad(p):
    _, q, n = PAD.closest(np.asarray(p, float))
    return q, n


def ang(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))))


def library(H, tips, tip):
    """The near-normal presses of 1-4 N of one tip: places and fields per newton."""
    fn = -np.sum(H["f"] * H["n"], 1)
    ft = np.linalg.norm(H["f"] + fn[:, None] * H["n"], axis=1)
    m = (tips == tip) & (H["kind"] == "press") & (fn >= 1.0) & (fn <= 4.0) & (ft < 0.15 * fn) & np.isfinite(H["n"]).all(1)
    return H["p"][m], H["d"][m] / fn[m][:, None]


def pattern_place(d, P, G, top=5):
    cos = G @ d / (np.linalg.norm(G, axis=1) * np.linalg.norm(d) + 1e-12)
    idx = np.argsort(-cos)[:top]
    return np.median(P[idx], 0), float(cos[idx[0]])


def errors(h, Yhat, P_lib=None, G_lib=None):
    """Per hold: the true place, force, lean and region, and each estimator's errors."""
    rows = []
    for i in range(len(h["d"])):
        p, f, n = h["p"][i], h["f"][i], h["n"][i]
        fn = -float(f @ n)
        ft = float(np.linalg.norm(f + fn * n))
        q_true, n_true = on_pad(p)
        r = dict(kind=str(h["kind"][i]), z=float(p[2]), y=float(p[1]), f=float(np.linalg.norm(f)), fn=fn,
                 lean=ft / max(fn, 1e-6))
        q, nq = on_pad(Yhat[i, :3])
        r.update(gp_e=float(np.linalg.norm(q - q_true)), gp_dz=float(q[2] - q_true[2]), gp_dy=float(q[1] - q_true[1]),
                 gp_en=ang(nq, n_true), gp_ef=ang(Yhat[i, 3:], f), gp_ratio=float(np.linalg.norm(Yhat[i, 3:])
                                                                                  / max(np.linalg.norm(f), 1e-6)))
        # shear: the force's part along the pad's surface, true at the true place, estimated at the estimated place
        fh = Yhat[i, 3:]
        st = f - float(f @ n_true) * n_true
        sh = fh - float(fh @ nq) * nq
        r.update(ft=float(np.linalg.norm(st)), sh_e=float(np.linalg.norm(sh - st)),
                 sh_ratio=float(np.linalg.norm(sh) / max(np.linalg.norm(st), 1e-6)),
                 sh_ang=ang(sh, st) if np.linalg.norm(st) > 0.2 else None,
                 fn_ratio=float(-(fh @ nq)) / max(fn, 1e-6))
        qf = support(-f)
        r.update(force_e=float(np.linalg.norm(qf - q_true)), force_dz=float(qf[2] - q_true[2]))
        if P_lib is not None and len(P_lib) and np.linalg.norm(h["d"][i]) >= 5.0:
            pp, c = pattern_place(h["d"][i], P_lib, G_lib)
            qp, nqp = on_pad(pp)
            r.update(pat_e=float(np.linalg.norm(qp - q_true)), pat_dz=float(qp[2] - q_true[2]), pat_en=ang(nqp, n_true),
                     pat_cos=c, pat_minus_force_dz=float(qp[2] - qf[2]), pat_to_force=float(np.linalg.norm(qp - qf)))
        rows.append(r)
    return rows


def stats(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    return dict(n=int(len(v)), median=round(float(np.median(v)), 3), p90=round(float(np.percentile(v, 90)), 3))


def summary(rows, est="gp"):
    keys = {"place_mm": f"{est}_e", "normal_deg": f"{est}_en", "force_dir_deg": f"{est}_ef", "force_ratio": f"{est}_ratio"}

    def block(sel):
        out = {k: stats([abs(r[v]) if k != "force_ratio" else r[v] for r in sel if v in r]) for k, v in keys.items()}
        out["dz_signed_median"] = stats([r[f"{est}_dz"] for r in sel if f"{est}_dz" in r])
        return out

    S = {"all": block(rows)}
    if est == "gp":
        sheared = [r for r in rows if r.get("sh_ang") is not None]
        S["shear"] = {"error_N": stats([r["sh_e"] for r in rows]), "direction_deg_over_0.2N": stats([r["sh_ang"] for r in sheared]),
                      "size_ratio_over_0.2N": stats([r["sh_ratio"] for r in sheared]),
                      "normal_force_ratio": stats([r["fn_ratio"] for r in rows]),
                      "by_shear_N": {f"{lo}-{hi}": {"direction_deg": stats([r["sh_ang"] for r in sheared if lo <= r["ft"] < hi]),
                                                    "error_N": stats([r["sh_e"] for r in rows if lo <= r["ft"] < hi])}
                                     for lo, hi in ((0.0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 9.0))}}
    S["by_force_N"] = {f"{lo}-{hi}": block([r for r in rows if lo <= r["f"] < hi]) for lo, hi in F_BINS}
    S["by_lean"] = {f"{lo}-{hi}": block([r for r in rows if lo <= r["lean"] < hi]) for lo, hi in LEAN_BINS}
    S["by_region"] = {lab: block([r for r in rows if lo <= r["z"] < hi]) for lo, hi, lab in Z_BINS}
    S["by_kind"] = {k: block([r for r in rows if r["kind"] == k]) for k in EVAL_KINDS}
    return S


def thin(rows, keys=("kind", "z", "y", "f", "lean", "gp_e", "gp_dz", "gp_en", "gp_ef", "gp_ratio", "force_dz",
                     "pat_e", "pat_dz", "pat_cos", "ft", "sh_e", "sh_ang", "fn_ratio")):
    """The held-out holds as columns (lists per key, None where a hold lacks the key), rounded, to keep the file small."""
    return {k: [None if r.get(k) is None else (round(r[k], 2) if isinstance(r[k], float) else r[k]) for r in rows]
            for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261119-board_contact_place_rig.json"))
    ap.add_argument("--quick", action="store_true", help="the deployed GP and one held-out session only")
    a = ap.parse_args()
    dep = np.load(GP_FILE)
    ell, nz = float(dep["ell"]), float(dep["nz"])
    floor = PR.floors(0)[:15]
    m_dep = {k: dep[k] for k in ("mx", "sx", "my", "sy", "Z", "a")} | {"ell": ell}
    out = {"what": __doc__.split("\n\n")[0].replace("\n", " "),
           "script": str(Path(__file__).resolve().relative_to(ROOT)), "gp": dict(ell=ell, noise=nz), "runs": []}
    t0 = time.time()
    # the deployed GP: in sample on the hand tips' sessions, out of sample on the tall sessions
    for s in SETS["tip1"] + SETS["tip2"] + SETS["tip3"] + SETS["rig3"] + list(TALL):
        h = load(s, EVAL_KINDS)
        rows = errors(h, predict(m_dep, h["d"]))
        run = dict(protocol="deployed, out of sample" if s in TALL else "deployed, in sample", held_out=[s],
                   tip=TALL.get(s) or TIP[session_set(s)], n=len(rows), summary=summary(rows))
        out["runs"].append(run)
        print(f"{run['protocol']:26s} {s.split('/')[-1]:32s} n {len(rows):5d}: place median "
              f"{run['summary']['all']['place_mm']['median']} mm (p90 {run['summary']['all']['place_mm']['p90']}), "
              f"normal {run['summary']['all']['normal_deg']['median']} deg  [{time.time() - t0:.0f} s]", flush=True)
    holdouts = [("session", [s]) for s in SETS["tip1"] + SETS["tip2"] + SETS["tip3"] + SETS["rig3"]]
    holdouts += [("tip", SETS["tip1"]), ("tip", SETS["tip2"]), ("tip", SETS["tip3"] + SETS["rig3"])]
    if a.quick:
        holdouts = holdouts[4:5]
    for proto, excl in holdouts:
        X, Y, tips, H = training(excl)
        m = fit_fixed(X, Y, floor, ell, nz)
        tip = TIP[session_set(excl[0])]
        P_lib, G_lib = library(H, tips, tip)
        for s in excl:
            h = load(s, EVAL_KINDS)
            rows = errors(h, predict(m, h["d"]), P_lib, G_lib)
            run = dict(protocol=f"{proto} left out", held_out=excl, session=s, tip=tip, n_train=int(len(X)),
                       n_library=int(len(P_lib)), n=len(rows), summary=summary(rows),
                       pattern=summary([r for r in rows if "pat_e" in r], "pat") if any("pat_e" in r for r in rows)
                       else None, holds=thin(rows) if proto == "session" else None)
            out["runs"].append(run)
            sa = run["summary"]["all"]
            pa = (run["pattern"] or {}).get("all", {}).get("place_mm") or {}
            print(f"{proto + ' left out':26s} {s.split('/')[-1]:32s} n {len(rows):5d} (train {len(X)}, library "
                  f"{len(P_lib)}): GP place median {sa['place_mm']['median']} mm (p90 {sa['place_mm']['p90']}), normal "
                  f"{sa['normal_deg']['median']} deg, force dir {sa['force_dir_deg']['median']} deg; pattern place "
                  f"{pa.get('median')} mm  [{time.time() - t0:.0f} s]", flush=True)
        del m, X, Y, H
        Path(a.out).write_text(json.dumps(out, indent=None, separators=(",", ":")))
    Path(a.out).write_text(json.dumps(out, indent=None, separators=(",", ":")))
    print("wrote", a.out, f"{Path(a.out).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
