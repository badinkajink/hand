#!/usr/bin/env python3
"""Magnetic crosstalk between the three fingertip boards on the SR2 hand, at the home layout (the night of 2026-11-04)
and at the stepper layouts of 2026-11-05 (the s1 layouts and the HOM study's), from the lab's sweeps
(docs/experiments/20261104-sr2_hand_crosstalk; the owner, 18:2x on 11-05: "this crosstalk analysis is high priority").
sr2_crosstalk_night.py analysed the night alone; this script takes any of the data sets and adds the layouts.

Readings. Each step's dwell gives every board's mean over its plausible records (|B| on every chip within 30 % of the
board's median and every component under 2e4 uT; misframed records), thermally corrected with the dwell's mean die
temperature (the calibration's kappa). Joints are the station's telemetry over the dwell (median; the command where no
sample falls in it). Mounts follow the run's mounts steps in the palm frame (home at a run's start). A step's change d
is its reading less the zero pose's at the same mounts, interpolated in time between the zero dwells at those mounts:
the reading a viewer zeroed at the layout's zero pose would see.

Analyses (each in the output JSON):
  noise       each zero dwell left out and predicted from its neighbours at the same mounts
  layouts     the change of every board at each layout move (the zero pose before and after), and the stepper steps
  own         each board's change predicted from its own finger's three angles by a GP (ur5_tip_calibration_board_gp),
              or from them and its mount, or from all nine angles and all mounts; tested by five random folds, by
              leaving each layout out, and by training at the home layout alone
  neighbour   what the own-angle GP leaves, against the distance to the nearest other pad, and the magnet model's share
              of it: the field of the other tips' four N52 magnets at the board's chips (fingertip_crosstalk.reading;
              each tip's calibration of 2026-11-02 for its chips, poles and strength), the pad frames from the cad_v2
              hand's forward kinematics (lab/handgeo.py), with one gain per receiving board and source tip fitted
  viewer      the contact viewer's decision (wrench_viewer: the board's estimator at its defaults, a chi-square above
              25, at least 0.25 N, and the test for a field from outside the pad) on each pose's change with no contact
              on any tip: raw, after the own-angle GP, after it and the magnet model, and after the nine-angle GP
Forces are changes divided by the tip's change per newton (median |d_board| / |f| over its UR5 face presses at 1 to 3 N).

    .venv/bin/python scripts/fingertip/sr2_crosstalk_hand.py [--sets 20261104 20261105] [--out NAME] [--no-viewer]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import fingertip_crosstalk as XT  # noqa: E402
import fingertip_wrench_model as W  # noqa: E402
import ur5_tip_calibration_board_gp as BG  # noqa: E402
import ur5_tip_calibration_v3_data_curve as V  # noqa: E402

EXP = ROOT / "docs/experiments/20261104-sr2_hand_crosstalk"
sys.path.insert(0, str(EXP / "lab"))
from handgeo import F, J, ORIGIN, Hand  # noqa: E402

DATA = EXP / "data"
CALIB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data/20261105-viewer_inputs/calib"
BOARD_GP = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data/20261105-board_gp_v3_id1_id3_rig_id3_id6.npz"
MST = timezone(timedelta(hours=-7))
BOARD = {"middle": 0, "index": 1, "thumb": 2}          # record order: mux channels 0, 1, 3
TIP_ID = {"thumb": 1, "index": 3, "middle": 2}
FID = {"thumb": "0", "index": "1", "middle": "2"}
YAW_SIGN = {"thumb": 1.0, "index": -1.0, "middle": -1.0}
HOME = {f: (ORIGIN[f][0] + x, ORIGIN[f][1] + y) for f, (x, y) in
        {"thumb": (-30.0, -55.0), "index": (30.0, 30.0), "middle": (30.0, -30.0)}.items()}   # handgeo.HOME_LOCAL
REC = np.dtype([("t", "<f8"), ("r", "<f4", (60,))])
SETS = {
    "20261104": dict(dir=DATA / "20261104-night", runs=["run1_home", "run1b_home_rest", "run1c_home_rest2", "run2_random_s1"],
                     logs=["boards/20261104-185340-mux.bin", "boards/20261104-191407-mux.bin"],
                     index_lost=datetime(2026, 11, 4, 19, 43, 52, tzinfo=MST).timestamp()),
    # the lab's push after hom1 (runs found by their events; run3_from32 left out: its middle X stage stayed at home
    # while the log moved it 45 mm)
    "20261105": dict(dir=DATA / "20261105-day", runs=None, logs=None, index_lost=None, skip=["run3_from32"]),
}
PALM_GAP = 5.0                                          # mm: mounts this close are one layout


def opener(p):
    p = Path(p)
    if not p.exists() and Path(str(p) + ".gz").exists():
        p = Path(str(p) + ".gz")
    return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)


def run_dirs(cfg):
    d = cfg["dir"]
    if cfg["runs"] is not None:
        return [d / r for r in cfg["runs"]]
    out = []
    for e in sorted(d.rglob("events.jsonl*")):
        if any(s in str(e.parent) for s in cfg.get("skip", [])):
            continue
        with opener(e) as fh:
            t0 = json.loads(fh.readline())["t"]
        out.append((t0, e.parent))
    return [p for _, p in sorted(out)]


def log_files(cfg):
    if cfg["logs"] is not None:
        return [cfg["dir"] / f for f in cfg["logs"]]
    return sorted((cfg["dir"] / "boards").glob("*.bin*"))


def telemetry(run):
    ts, q = [], []
    with opener(run / "telemetry.jsonl") as fh:
        for line in fh:
            s = json.loads(line)
            sv = s.get("servos") or {}
            if not all(FID[f] in sv for f in F):
                continue
            ts.append(s.get("t_sample") or s["t_recv"])
            q.append([YAW_SIGN[f] * sv[FID[f]]["aa"] if j == "yaw" else sv[FID[f]]["fe1" if j == "mcp" else "fe2"]
                      for f in F for j in J])
    return np.array(ts), np.array(q, float)


def steps(cfg, name):
    """Every completed step of the set's runs in time order, with its window, measured joints and mounts."""
    out, mounts, torque_off = [], dict(HOME), False
    for run in run_dirs(cfg):
        tq, q = telemetry(run)
        with opener(run / "events.jsonl") as fh:
            ev = [json.loads(line) for line in fh]
        for e in ev:
            if e.get("kind") != "step":
                continue
            if e["op"] == "mounts" and e.get("mounts"):
                src = e.get("mount_positions") or e["mounts"]
                mounts = {**mounts, **{f: (float(v["x"]), float(v["y"])) for f, v in src.items()}}
            if e["op"] == "torque":
                torque_off = e.get("state") == "off"
            if e.get("t_end") is None or e.get("t_settled") is None:
                continue
            dwell = e["t_end"] - e["t_settled"]
            keep = {"joints": min(2.5, max(dwell - 0.5, 0.5)), "torque": 15.0, "mounts": 10.0}[e["op"]]
            if e["family"].startswith("baseline"):
                keep = min(60.0, dwell - 1.0)
            t0, t1 = e["t_end"] - keep, e["t_end"]
            k = (tq >= t0) & (tq <= t1)
            qc = np.array([e["joints"][f][j] for f in F for j in J]) if e.get("joints") else np.zeros(9)
            out.append(dict(set=name, run=run.name, step=e["step"], op=e["op"], family=e["family"], t0=t0, t1=t1,
                            q_cmd=qc, q=np.median(q[k], axis=0) if k.any() else qc, n_tel=int(k.sum()),
                            mounts=dict(mounts), torque_off=torque_off, plan_clear=e.get("clearance_mm")))
    return out


NWIN, WREC = 12, 8                                       # 50 ms frames per dwell, records per frame (160 records/s)


def dwell_means(cfg, S):
    """Each step's mean reading per board (15 uT, thermally corrected), its spread and record count, and NWIN frames of
    WREC consecutive plausible records spread over the dwell: what the viewer reads every 50 ms."""
    kap = W.load_thermal()
    kap = kap if kap is not None else np.full(5, -0.0067)
    W0 = np.array([s["t0"] for s in S])
    W1 = np.array([s["t1"] for s in S])
    acc = {k: np.zeros((len(S), 3) + sh) for k, sh in (("B", (5, 3)), ("B2", (5, 3)), ("T", (5,)))}
    cnt = np.zeros((len(S), 3))
    frames = [[[] for _ in range(3)] for _ in S]
    usual = None
    for path in log_files(cfg):
        raw = gzip.open(path).read() if path.suffix == ".gz" else path.read_bytes()
        z = np.frombuffer(raw, REC, count=len(raw) // REC.itemsize)
        for i0 in range(0, len(z), 400_000):
            blk = z[i0:i0 + 400_000]
            t = blk["t"]
            r = blk["r"].reshape(-1, 3, 5, 4).astype(float)
            B, T = r[..., 1:], r[..., 0]
            mag = np.linalg.norm(B, axis=3)
            if usual is None:
                usual = np.nanmedian(np.where(np.isfinite(mag).all((1, 2))[:, None, None], mag, np.nan), axis=0)
            ok = ((np.abs(mag - usual) <= 0.3 * usual).all(2) & np.isfinite(r).all((2, 3))
                  & (np.abs(np.nan_to_num(B, nan=0.0, posinf=1e30, neginf=1e30)) <= 2e4).all((2, 3)))
            if cfg["index_lost"] is not None:
                after = t >= cfg["index_lost"]
                idx_zero = (r[:, BOARD["index"]] == 0).all((1, 2))
                ok[after, BOARD["index"]] = False
                bad = after & ~idx_zero                    # misaligned records after the index board was lost
                ok[bad] = False
            sel = np.flatnonzero((W1 >= t[0]) & (W0 <= t[-1]))
            for i in sel:
                a, b = np.searchsorted(t, [W0[i], W1[i]])
                if b <= a:
                    continue
                for bd in range(3):
                    m = ok[a:b, bd]
                    if m.any():
                        acc["B"][i, bd] += B[a:b, bd][m].sum(0)
                        acc["B2"][i, bd] += (B[a:b, bd][m] ** 2).sum(0)
                        acc["T"][i, bd] += T[a:b, bd][m].sum(0)
                        cnt[i, bd] += m.sum()
                        iv = a + np.flatnonzero(m)
                        if len(iv) >= WREC:
                            for k0 in np.linspace(0, len(iv) - WREC, NWIN).astype(int):
                                sl = iv[k0:k0 + WREC]
                                frames[i][bd].append(W.thermal_correct(B[sl, bd].mean(0), T[sl, bd].mean(0), kap).ravel())
        del z, raw
    for i, s in enumerate(S):
        s["b"], s["sd"], s["n"] = {}, {}, {}
        for f in F:
            n = cnt[i, BOARD[f]]
            if n >= 5:
                mB = acc["B"][i, BOARD[f]] / n
                sd = np.sqrt(np.maximum(acc["B2"][i, BOARD[f]] / n - mB ** 2, 0))
                s["b"][f] = W.thermal_correct(mB, acc["T"][i, BOARD[f]] / n, kap).ravel()
                s["sd"][f] = float(np.linalg.norm(sd))
                s["n"][f] = int(n)
                if frames[i][BOARD[f]]:
                    s.setdefault("w", {})[f] = np.array(frames[i][BOARD[f]][:NWIN])


def pct(v, p):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return round(float(np.percentile(v, p)), 2) if len(v) else None


def key(m):
    return tuple(int(round(m[f][i] / PALM_GAP)) for f in F for i in (0, 1))


def is_zero(s):
    return (s["op"] == "joints" and (s["family"] in ("zero", "via_zero") or s["family"].startswith("baseline"))
            and not s["torque_off"] and np.allclose(s["q_cmd"], 0))


def baselines(S):
    """Zero dwells by (set, mounts), the noise of each left out, and every step's change d against them."""
    Z = {}
    for s in S:
        if is_zero(s):
            for f, b in s["b"].items():
                Z.setdefault((s["set"], key(s["mounts"]), f), []).append((0.5 * (s["t0"] + s["t1"]), b))
    Z = {k: (np.array([x[0] for x in v]), np.array([x[1] for x in v])) for k, v in Z.items()}
    noise = {}
    for (st, _, f), (tz, bz) in Z.items():
        for i in range(1, len(tz) - 1):
            pred = bz[i - 1] + (bz[i + 1] - bz[i - 1]) * (tz[i] - tz[i - 1]) / (tz[i + 1] - tz[i - 1])
            noise.setdefault((st, f), []).append(float(np.linalg.norm(bz[i] - pred)))
    for s in S:
        s["d"] = {}
        for f, b in s["b"].items():
            z = Z.get((s["set"], key(s["mounts"]), f))
            if z is None:
                continue
            tm = 0.5 * (s["t0"] + s["t1"])
            s["d"][f] = b - np.array([np.interp(tm, z[0], z[1][:, c]) for c in range(15)])
    return Z, {f"{st} {f}": dict(n=len(v), median_uT=pct(v, 50), p95_uT=pct(v, 95)) for (st, f), v in noise.items()}


def layout_name(S):
    """A layout's name for each step: the family of its run's last mounts step, or home."""
    names, last = [], "home"
    for s in S:
        if s["op"] == "mounts" and s["family"].startswith("layout_"):
            last = s["family"][len("layout_"):]
        if key(s["mounts"]) == key(HOME):
            last = "home"
        s["layout"] = last
        names.append(last)
    return names


def layout_moves(S, upn):
    """The change of each board across each layout move: the last zero dwell before it against the first after."""
    rows = []
    zs = [i for i, s in enumerate(S) if is_zero(s)]
    for i, s in enumerate(S):
        if s["op"] != "mounts":
            continue
        before = [j for j in zs if j < i and S[j]["set"] == s["set"]]
        after = [j for j in zs if j > i and S[j]["set"] == s["set"] and key(S[j]["mounts"]) == key(s["mounts"])]
        if not before or not after:
            continue
        a, b = S[before[-1]], S[after[0]]
        if b["t0"] - a["t1"] > 300:
            continue
        moved = {f: round(float(np.hypot(b["mounts"][f][0] - a["mounts"][f][0], b["mounts"][f][1] - a["mounts"][f][1])), 1)
                 for f in F}
        ch = {f: round(float(np.linalg.norm(b["b"][f] - a["b"][f])), 1) for f in F if f in a["b"] and f in b["b"]}
        rows.append(dict(set=s["set"], family=s["family"], moved_mm=moved, change_uT=ch,
                         change_N={f: round(v / upn[f], 2) for f, v in ch.items()}, gap_s=round(b["t0"] - a["t1"], 1)))
    return rows


def tip_uT_per_N(tip):
    V.KINDS[:] = ["press"]
    H = V.load(V.TIPS[tip])
    fm = np.linalg.norm(H["f"], axis=1)
    k = (fm >= 1.0) & (fm <= 3.0)
    return float(np.median(np.linalg.norm(H["d_board"][k], axis=1) / fm[k]))


# the pad frame P of the calibrations (tpu_tip_design: x to the face, z from the tip's base to its end) in the hand's
# tip frame (handgeo's <f>_tip: origin on the finger's axis at the pad's centre plane, z toward the PIP, the face at
# x = face_sign * 10.55 mm): x_P = s x_tip - 2.05, y_P = -s y_tip, z_P = Z_C - z_tip
FACE_SIGN = {"thumb": 1.0, "index": -1.0, "middle": -1.0}


def T_tip_P(f):
    s = FACE_SIGN[f]
    T = np.eye(4)
    T[:3, :3] = np.diag([s, -s, -1.0])
    T[:3, 3] = [s * 2.05, 0.0, W.TD.Z_C]
    return T


class Kin:
    def __init__(self):
        self.h = Hand()
        self.TP = {f: T_tip_P(f) for f in F}

    def pads(self, mounts, q):
        """Pad frames T (4, 4) per finger in the scene's frame, mm."""
        local = {f: (mounts[f][0] - ORIGIN[f][0], mounts[f][1] - ORIGIN[f][1]) for f in F}
        self.h.set(local, {f: {j: float(q[3 * i + k]) for k, j in enumerate(J)} for i, f in enumerate(F)})
        out = {}
        for f in F:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = self.h.tip_frame(f), self.h.tip(f)
            out[f] = T @ self.TP[f]
        return out


def own_models(S, upn, names):
    """Own-pose GPs per board: inputs, tests and cross-validated predictions."""
    res, pred = {}, {}
    rng = np.random.default_rng(0)
    for f in F:
        rows = [i for i, s in enumerate(S) if s["op"] == "joints" and not is_zero(s) and not s["torque_off"]
                and f in s["d"] and np.isfinite(s["q"]).all() and not s["family"].startswith("test")]
        if len(rows) < 30:
            continue
        own = [3 * F.index(f) + k for k in range(3)]
        q = np.array([S[i]["q"] for i in rows])
        mt = np.array([[S[i]["mounts"][g][c] for g in F for c in (0, 1)] for i in rows])
        Y = np.array([S[i]["d"][f] for i in rows])
        lay = np.array([names[i] for i in rows])
        mi = [2 * F.index(f), 2 * F.index(f) + 1]
        inputs = {"own_angles": (q[:, own], np.full(3, 5.0)),
                  "own_angles_mount": (np.c_[q[:, own], mt[:, mi]], np.r_[np.full(3, 5.0), 5.0, 5.0]),
                  "all_angles_mounts": (np.c_[q, mt], np.r_[np.full(9, 5.0), np.full(6, 5.0)])}
        folds = np.array_split(rng.permutation(len(rows)), 5)
        out = dict(rows=len(rows), layouts={l: int((lay == l).sum()) for l in np.unique(lay)},
                   change_uT=dict(median=pct(np.linalg.norm(Y, axis=1), 50), p95=pct(np.linalg.norm(Y, axis=1), 95),
                                  max=round(float(np.linalg.norm(Y, axis=1).max()), 1)))
        P = {}
        for name, (X, fl) in inputs.items():
            cv = np.full_like(Y, np.nan)
            for te in folds:
                tr = np.setdiff1d(np.arange(len(rows)), te)
                cv[te] = BG.predict(BG.fit(X[tr], Y[tr], fl), X[te])
            P[name] = cv
            e = np.linalg.norm(cv - Y, axis=1)
            r = {"five_folds": dict(median_uT=pct(e, 50), p95_uT=pct(e, 95), median_N=round(pct(e, 50) / upn[f], 3),
                                    p95_N=round(pct(e, 95) / upn[f], 3))}
            lolo = np.full_like(Y, np.nan)
            for l in np.unique(lay):
                te = lay == l
                if te.sum() < 20 or (~te).sum() < 30:
                    continue
                lolo[te] = BG.predict(BG.fit(X[~te], Y[~te], fl), X[te])
                el = np.linalg.norm(lolo[te] - Y[te], axis=1)
                r[f"left_out {l}"] = dict(n=int(te.sum()), median_uT=pct(el, 50), p95_uT=pct(el, 95),
                                          median_N=round(pct(el, 50) / upn[f], 3))
            if np.isfinite(lolo).any():
                P[f"{name}_left_out_layout"] = lolo
                el = np.linalg.norm(lolo - Y, axis=1)
                r["left_out_layout_all"] = dict(n=int(np.isfinite(el).sum()), median_uT=pct(el, 50), p95_uT=pct(el, 95),
                                                median_N=round(pct(el, 50) / upn[f], 3), p95_N=round(pct(el, 95) / upn[f], 3))
            home = lay == "home"
            if home.sum() >= 30 and (~home).sum() >= 20:
                ho = np.full_like(Y, np.nan)
                ho[~home] = BG.predict(BG.fit(X[home], Y[home], fl), X[~home])
                P[f"{name}_home_only"] = ho
                eh = np.linalg.norm(ho[~home] - Y[~home], axis=1)
                r["home_only_to_layouts"] = dict(n=int((~home).sum()), median_uT=pct(eh, 50), p95_uT=pct(eh, 95),
                                                 median_N=round(pct(eh, 50) / upn[f], 3), p95_N=round(pct(eh, 95) / upn[f], 3))
            out[name] = r
        res[f] = out
        pred[f] = dict(rows=rows, Y=Y, P=P, lay=lay)
    return res, pred


def magnet_model(S, pred, upn):
    """The other tips' magnets at each board's chips (change from the layout's zero pose), and its fit to what the
    own-angle GP leaves."""
    kin = Kin()
    cal = {f: W.Calib.load(CALIB / f"20261102-v3_id_{TIP_ID[f]}-calib.json") for f in F}
    row = W.fe_row("four_05R4")
    model = {f: W.Model(cal[f], row) for f in F}
    src = {f: XT.source(row, cal[f].poles, cal[f].k) for f in F}
    zero_pads = {}
    out, field = {}, {}
    for f, pr in pred.items():
        others = [g for g in F if g != f]
        M = np.zeros((len(pr["rows"]), 2, 15))
        dist = np.zeros(len(pr["rows"]))
        for n, i in enumerate(pr["rows"]):
            s = S[i]
            Tq = kin.pads(s["mounts"], s["q"])
            k = key(s["mounts"])
            if k not in zero_pads:
                zero_pads[k] = kin.pads(s["mounts"], np.zeros(9))
            T0 = zero_pads[k]
            for j, g in enumerate(others):
                rel = np.linalg.inv(Tq[f]) @ Tq[g]
                rel0 = np.linalg.inv(T0[f]) @ T0[g]
                M[n, j] = (XT.reading(model[f], src[g], rel[:3, :3], rel[:3, 3])
                           - XT.reading(model[f], src[g], rel0[:3, :3], rel0[:3, 3]))
            c = Tq[f][:3, 3]
            dist[n] = min(np.linalg.norm(Tq[g][:3, 3] - c) for g in others)
        R = pr["Y"] - pr["P"]["own_angles"]
        big = np.linalg.norm(M.sum(1), axis=1) >= 5.0           # the model's field above the noise (3.3 uT)
        if big.sum() >= 20:
            A = M[big].transpose(0, 2, 1).reshape(-1, 2)        # (rows * 15, 2)
            gain = np.linalg.lstsq(A, R[big].ravel(), rcond=None)[0]
        else:
            gain = np.ones(2)
        fit = (M * gain[None, :, None]).sum(1)
        nominal = M.sum(1)
        bins = ((0, 25), (25, 35), (35, 50), (50, 70), (70, 999))
        rows = {}
        for a, b in bins:
            k = (dist >= a) & (dist < b)
            if not k.any():
                continue
            e0 = np.linalg.norm(R[k], axis=1)
            e1 = np.linalg.norm(R[k] - fit[k], axis=1)
            rows[f"{a}-{b} mm"] = dict(n=int(k.sum()), model_uT_median=pct(np.linalg.norm(M[k].sum(1), axis=1), 50),
                                       model_uT_max=round(float(np.linalg.norm(M[k].sum(1), axis=1).max()), 1),
                                       left_uT_median=pct(e0, 50), left_uT_p95=pct(e0, 95),
                                       after_model_uT_median=pct(e1, 50), after_model_uT_p95=pct(e1, 95))
        tot = float((R ** 2).sum())
        out[f] = dict(rows_model_over_5uT=int(big.sum()), gain_fitted=bool(big.sum() >= 20),
                      gain=dict(zip(others, np.round(gain, 3).tolist())),
                      explained_share_fitted=round(1 - float(((R - fit) ** 2).sum()) / tot, 3),
                      explained_share_nominal=round(1 - float(((R - nominal) ** 2).sum()) / tot, 3),
                      by_distance=rows)
        field[f] = dict(M=M, gain=gain, dist=dist)
    return out, field


def viewer(S, pred, field, upn):
    """Contacts the viewer would show with nothing touching any tip: its decision on each pose's dwell mean (no noise)
    and on the pose's 50 ms frames (the chips' noise as the viewer sees it), for the change raw and after each model;
    a model's prediction for the pose is taken from every frame of it."""
    basis = W.load_basis(W.basis_path("four_05R4"))
    row = W.fe_row("four_05R4")
    out = {}
    for f, pr in pred.items():
        cal = W.Calib.load(CALIB / f"20261102-v3_id_{TIP_ID[f]}-calib.json")
        est = W.SurfaceEstimator(W.Model(cal, row), basis, floor_uT=2.0)
        est.set_noise_scale(1 / np.sqrt(0.05 * 140))
        ff = W.ForeignField(est, 25.0)

        def shown(d):
            e = est.estimate(d)
            if not (e["contact"] and e["e"] > 25.0 and e["f_n"] >= 0.25):
                return 0.0, False
            if ff.lam(d, e) > ff.on:
                return 0.0, True
            return float(e["f_n"]), False

        Y = pr["Y"]
        model = {"raw": np.zeros_like(Y), "own_angles": pr["P"]["own_angles"],
                 "all_angles_mounts": pr["P"]["all_angles_mounts"]}
        if f in field:
            model["own_angles_and_magnets"] = pr["P"]["own_angles"] + (field[f]["M"] * field[f]["gain"][None, :, None]).sum(1)
        for extra in ("own_angles_left_out_layout", "own_angles_mount_left_out_layout", "own_angles_home_only"):
            if extra in pr["P"]:
                model[extra] = pr["P"][extra]
        r = {}
        for name, P in model.items():
            fn_d, fn_w, pose_any, nfr, foreign = [], [], [], 0, 0
            for n, i in enumerate(pr["rows"]):
                if not np.isfinite(P[n]).all():
                    fn_d.append(np.nan)
                    pose_any.append(False)
                    continue
                v, _ = shown(Y[n] - P[n])
                fn_d.append(v)
                w = S[i].get("w", {}).get(f)
                hit = False
                if w is not None:
                    for x in Y[n] + (w - S[i]["b"][f]) - P[n]:
                        v, rej = shown(x)
                        fn_w.append(v)
                        foreign += rej
                        hit |= v > 0
                    nfr += len(w)
                pose_any.append(hit)
            fn_d, fn_w = np.array(fn_d), np.array(fn_w)
            r[name] = dict(poses=int(np.isfinite(fn_d).sum()), dwell_shown=int((fn_d > 0).sum()),
                           dwell_force_N_median=pct(fn_d[fn_d > 0], 50),
                           frames=nfr, frames_shown=int((fn_w > 0).sum()),
                           frames_shown_share=round(float((fn_w > 0).mean()), 4) if nfr else None,
                           poses_with_a_frame_shown=int(np.sum(pose_any)),
                           frames_rejected_as_foreign=int(foreign),
                           frame_force_N_median=pct(fn_w[fn_w > 0], 50),
                           frame_force_N_max=round(float(fn_w.max()), 2) if nfr and (fn_w > 0).any() else None)
            pr.setdefault("shown", {})[name] = fn_d
        out[f] = r
    return out


class BoardGP:
    """wrench_viewer.BoardGP.predict and App.from_board_gp (the viewer's "board gp" mode), which this python cannot
    import (the viewer needs aiohttp): the pooled GP's point and force, weighted by w against the physical estimate's,
    and the press along the normal of the surface candidate nearest the point."""

    def __init__(self, path=BOARD_GP, m=256):
        from scipy.linalg import cho_factor, cho_solve
        self.cf, self.cs = cho_factor, cho_solve
        z = np.load(path)
        self.mx, self.sx, self.my, self.sy, self.Z, self.a = (z[k] for k in ("mx", "sx", "my", "sy", "Z", "a"))
        self.ell, self.nz = float(z["ell"]), float(z["nz"])
        self.m = min(m, len(self.Z) - 1)
        self.sq = (self.Z ** 2).sum(1)

    def predict(self, d):
        q = (np.asarray(d, float) - self.mx) / self.sx
        k = np.exp(-0.5 * ((self.Z - q) ** 2).sum(1) / self.ell ** 2)
        y = k @ self.a * self.sy + self.my
        s = np.argpartition(-k, self.m)[:self.m]
        Ks = np.exp(-0.5 * np.maximum(self.sq[s][:, None] + self.sq[s][None] - 2 * self.Z[s] @ self.Z[s].T, 0) / self.ell ** 2)
        w = float(np.clip(k[s] @ self.cs(self.cf(Ks + self.nz ** 2 * np.eye(self.m), lower=True), k[s]), 0.0, 1.0))
        return y[:3], y[3:], w

    def press(self, est, tree, e, d):
        pg, fg, w = self.predict(d)
        p = w * pg + (1 - w) * np.asarray(e["p"], float)
        f = w * fg + (1 - w) * np.asarray(e["f"], float)
        return float(-f @ est.Cn[int(tree.query(p)[1])])


def inject(S, pred, upn, n_press=400, seed=0):
    """Real presses on the hand's residual: UR5 presses of each tip (its own tip-mode holds, 0.25 to 3 N) added to 50 ms
    frames of the hand, through the viewer's decision. Frames from the zero dwells (the chips' noise alone) against
    frames of the moving hand after the own-angle GP and raw. Reports the share detected and the error of the shown
    press force against the press's normal force."""
    from scipy.spatial import cKDTree
    basis = W.load_basis(W.basis_path("four_05R4"))
    row = W.fe_row("four_05R4")
    rng = np.random.default_rng(seed)
    gp = BoardGP()
    out = {}
    for f, pr in pred.items():
        V.KINDS[:] = ["press"]
        H = V.load(V.TIPS[TIP_ID[f]])
        fn_true = -np.sum(H["f"] * H["n"], 1)
        k = np.flatnonzero((fn_true >= 0.25) & (fn_true <= 3.0))
        pick = rng.choice(k, min(n_press, len(k)), replace=False)
        cal = W.Calib.load(CALIB / f"20261102-v3_id_{TIP_ID[f]}-calib.json")
        est = W.SurfaceEstimator(W.Model(cal, row), basis, floor_uT=2.0)
        est.set_noise_scale(1 / np.sqrt(0.05 * 140))
        ff = W.ForeignField(est, 25.0)
        tree = cKDTree(est.C)
        noise = np.concatenate([s["w"][f] - s["b"][f] for s in S if is_zero(s) and f in s.get("w", {})])
        hand = {name: np.concatenate([pr["Y"][n] + (S[i]["w"][f] - S[i]["b"][f]) - P[n]
                                      for n, i in enumerate(pr["rows"]) if f in S[i].get("w", {})])
                for name, P in (("raw", np.zeros_like(pr["Y"])), ("own_angles", pr["P"]["own_angles"]))}
        r = {}
        for name, base in (("zero_pose_noise", noise), *hand.items()):
            fr = base[rng.integers(0, len(base), len(pick))]
            det, err, err_gp = [], [], []
            for x, j in zip(fr, pick):
                d = x + H["d_board"][j]
                e = est.estimate(d)
                c = bool(e["contact"] and e["e"] > 25.0 and e["f_n"] >= 0.25) and not ff.lam(d, e) > ff.on
                det.append(c)
                if c:
                    err.append((e["f_n"] - fn_true[j]) / fn_true[j])
                    err_gp.append((gp.press(est, tree, e, d) - fn_true[j]) / fn_true[j])
            r[name] = dict(presses=len(pick), detected_share=round(float(np.mean(det)), 3))
            for tag, ee in (("physical", np.array(err)), ("board_gp", np.array(err_gp))):
                r[name][tag] = dict(force_error_median_pct=pct(100 * ee, 50), force_error_abs_median_pct=pct(100 * np.abs(ee), 50),
                                    force_error_abs_p90_pct=pct(100 * np.abs(ee), 90))
        out[f] = dict(true_force_N_median=pct(fn_true[pick], 50), **r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=["20261104"])
    ap.add_argument("--out", default=None, help="output stem in data/ (default 20261105-sr2_crosstalk_hand_<sets>)")
    ap.add_argument("--no-viewer", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    S = []
    for name in a.sets:
        cfg = SETS[name]
        Ss = steps(cfg, name)
        dwell_means(cfg, Ss)
        S += Ss
        print(f"{name}: {len(Ss)} steps, {time.time() - t0:.0f} s", flush=True)
    upn = {f: tip_uT_per_N(TIP_ID[f]) for f in F}
    names = layout_name(S)
    Z, noise = baselines(S)
    R = dict(what=__doc__.split("\n\n")[1], sets=a.sets, steps=len(S), uT_per_N={f: round(v, 2) for f, v in upn.items()},
             noise=noise, layouts=layout_moves(S, upn))
    R["own"], pred = own_models(S, upn, names)
    print(f"own models: {time.time() - t0:.0f} s", flush=True)
    R["neighbour"], field = magnet_model(S, pred, upn)
    print(f"magnet model: {time.time() - t0:.0f} s", flush=True)
    if not a.no_viewer:
        R["viewer"] = viewer(S, pred, field, upn)
        print(f"viewer: {time.time() - t0:.0f} s", flush=True)
        R["presses_on_the_hand"] = inject(S, pred, upn)
        print(f"presses: {time.time() - t0:.0f} s", flush=True)
    stem = a.out or f"20261105-sr2_crosstalk_hand_{'_'.join(a.sets)}"
    (DATA / f"{stem}.json").write_text(json.dumps(R, indent=1))
    np.savez_compressed(DATA / f"{stem}_rows.npz", **{
        f"{f}_{k}": v for f, pr in pred.items() for k, v in
        (("rows", np.array(pr["rows"])), ("Y", pr["Y"]), ("lay", pr["lay"].astype(str)),
         *((f"P_{n}", p) for n, p in pr["P"].items()),
         *((f"shown_{n}", v) for n, v in pr.get("shown", {}).items()),
         *((("M", field[f]["M"]), ("dist", field[f]["dist"])) if f in field else ()))},
        q=np.array([s["q"] for s in S]), t=np.array([0.5 * (s["t0"] + s["t1"]) for s in S]),
        mounts=np.array([[s["mounts"][f][c] for f in F for c in (0, 1)] for s in S]),
        set=np.array([s["set"] for s in S]), run=np.array([s["run"] for s in S]),
        family=np.array([s["family"] for s in S]), layout=np.array(names))
    print(json.dumps(R, indent=1))
    print("->", DATA / f"{stem}.json")


if __name__ == "__main__":
    main()
