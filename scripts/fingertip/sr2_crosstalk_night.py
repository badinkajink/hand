#!/usr/bin/env python3
"""Crosstalk between the fingertips on the SR2 hand, from the lab's sweep of 2026-11-04 (18:57 to 20:34; runbook and data
in docs/experiments/20261104-sr2_hand_crosstalk, data/20261104-night).

Each step's dwell gives every board's mean reading (15 components, thermally corrected as in the UR5 calibration), and
its joints as measured by the station's telemetry (commanded where no sample falls in the window). The zero-pose dwells,
one every 12 poses, give each board's drift, interpolated in time; a step's change is its reading less that baseline.
Then:
  noise     the zero dwells left out one at a time and predicted from their neighbours (the drift floor)
  torque    servo torque off and on at the zero pose
  stepper   each stepper axis 5 mm out and back, the fingers straight
  families  each finger alone through its yaw, mcp and pip and two grids: the change on its own board (self term:
            the Earth's field turned with the tip, the finger's own servos) and on the others' (crosstalk)
  pair      the thumb and the middle finger toward each other: the change on both boards against the CAD clearance
  subtract  each board's change predicted from the joint angles by a GP (ur5_tip_calibration_board_gp), tested on the 327
            random three-finger poses: trained on the structured steps alone, and five folds over the random poses with
            the structured steps in every training set; inputs the board's own three angles, or all nine
Forces are the change divided by the tip's change per newton: the median of |d_board| / |f| over its UR5 face presses
at 1 to 3 N (tip mode).

    .venv/bin/python scripts/fingertip/sr2_crosstalk_night.py
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import fingertip_wrench_model as W  # noqa: E402
import ur5_tip_calibration_board_gp as BG  # noqa: E402
import ur5_tip_calibration_v3_data_curve as V  # noqa: E402

N = ROOT / "docs/experiments/20261104-sr2_hand_crosstalk/data/20261104-night"
OUT = ROOT / "docs/experiments/20261104-sr2_hand_crosstalk/data/20261104-night_crosstalk_analysis"
F = ("thumb", "index", "middle")
J = ("yaw", "mcp", "pip")
CHANNELS = (0, 1, 3)                                  # record order in the three-board logs
BOARD = {"middle": 0, "index": 1, "thumb": 2}          # index into CHANNELS: middle ch0, index ch1, thumb ch3
TIP_ID = {"thumb": 1, "index": 3, "middle": 2}
FID = {"thumb": "0", "index": "1", "middle": "2"}
YAW_SIGN = {"thumb": 1.0, "index": -1.0, "middle": -1.0}
RUNS = ("run1_home", "run1b_home_rest", "run1c_home_rest2", "run2_random_s1")
LOGS = ("boards/20261104-185340-mux.bin", "boards/20261104-191407-mux.bin")
MST = timezone(timedelta(hours=-7))
T_INDEX_LOST = datetime(2026, 11, 4, 19, 43, 52, tzinfo=MST).timestamp()
ZERO_FAM = ("zero", "baseline", "baseline_end", "via_zero")


def boards():
    """Times and thermally corrected readings (n, 3, 15) of the three boards, with a validity mask per board."""
    dt = np.dtype([("t", "<f8"), ("r", "<f4", (60,))])
    t, r = [], []
    for f in LOGS:
        z = np.fromfile(N / f, dt)
        t.append(z["t"])
        r.append(z["r"].reshape(-1, 3, 5, 4).astype(float))
    t, r = np.concatenate(t), np.concatenate(r)
    o = np.argsort(t, kind="stable")
    t, r = t[o], r[o]
    B, T = r[..., 1:], r[..., 0]
    mag = np.linalg.norm(B, axis=3)                                       # (n, 3, 5)
    before = t < T_INDEX_LOST
    usual = np.median(mag[before & np.isfinite(mag).all((1, 2))], axis=0)  # (3, 5)
    near = ((np.abs(mag - usual) <= 0.3 * usual).all(2) & np.isfinite(r).all((2, 3))
            & (np.abs(np.nan_to_num(B, nan=0.0, posinf=1e30, neginf=1e30)) <= 2e4).all((2, 3)))   # misframed records (lab, 11-05)
    ok = near.copy()
    after = ~before
    idx_zero = (r[:, BOARD["index"]] == 0).all((1, 2))
    good_after = after & idx_zero & near[:, BOARD["middle"]] & near[:, BOARD["thumb"]]
    ok[after] = False
    ok[good_after, BOARD["middle"]] = True
    ok[good_after, BOARD["thumb"]] = True
    kap = W.load_thermal()
    kap = kap if kap is not None else np.full(5, -0.0067)
    Bc = np.empty_like(B)
    for b in range(3):                                    # die temperature: a 3 s running mean, as in the calibration
        Tb = np.where(ok[:, b, None], T[:, b], np.nan)
        Tm = np.empty_like(Tb)
        for c in range(5):
            v = Tb[:, c]
            g = np.isfinite(v)
            cs = np.r_[0.0, np.cumsum(np.where(g, v, 0.0))]
            cn = np.r_[0, np.cumsum(g)]
            i0 = np.searchsorted(t, t - 3.0)
            i1 = np.arange(1, len(t) + 1)
            Tm[:, c] = (cs[i1] - cs[i0]) / np.maximum(cn[i1] - cn[i0], 1)
        Bc[:, b] = W.thermal_correct(B[:, b], Tm, kap)
    return t, Bc.reshape(len(t), 3, 15), ok, dict(records=int(len(t)), kept=ok.sum(0).tolist(),
                                                  kept_after_index_lost=int(good_after.sum()))


def telemetry(run):
    ts, q = [], []
    for line in open(N / run / "telemetry.jsonl"):
        s = json.loads(line)
        sv = s.get("servos") or {}
        if not all(FID[f] in sv for f in F):
            continue
        ts.append(s.get("t_sample") or s["t_recv"])
        q.append([YAW_SIGN[f] * sv[FID[f]]["aa"] if j == "yaw" else sv[FID[f]]["fe1" if j == "mcp" else "fe2"]
                  for f in F for j in J])
    return np.array(ts), np.array(q, float)


def steps():
    out = []
    for run in RUNS:
        tq, q = telemetry(run)
        for line in open(N / run / "events.jsonl"):
            e = json.loads(line)
            if e.get("kind") != "step" or e.get("t_end") is None or e.get("t_settled") is None:
                continue
            dwell = e["t_end"] - e["t_settled"]
            keep = {"joints": min(2.5, max(dwell - 0.5, 0.5)), "torque": 15.0, "mounts": 10.0}[e["op"]]
            if e["family"] in ("baseline", "baseline_end"):
                keep = min(60.0, dwell - 1.0)
            t0, t1 = e["t_end"] - keep, e["t_end"]
            k = (tq >= t0) & (tq <= t1)
            qc = np.array([e["joints"][f][j] for f in F for j in J]) if e.get("joints") else np.full(9, np.nan)
            out.append(dict(run=run, step=e["step"], op=e["op"], family=e["family"], t0=t0, t1=t1,
                            q_cmd=qc, q=np.median(q[k], axis=0) if k.any() else qc, n_tel=int(k.sum()),
                            state=e.get("state"), mounts=e.get("mounts"), clearance=e.get("clearance_mm")))
    return out


def tip_uT_per_N(tip):
    V.KINDS[:] = ["press"]
    H = V.load(V.TIPS[tip])
    fm = np.linalg.norm(H["f"], axis=1)
    k = (fm >= 1.0) & (fm <= 3.0)
    return float(np.median(np.linalg.norm(H["d_board"][k], axis=1) / fm[k]))


def pct(v, p):
    return round(float(np.percentile(v, p)), 2) if len(v) else None


def main():
    t, B, ok, rec = boards()
    S = steps()
    for s in S:                                           # each board's mean over the dwell
        k = (t >= s["t0"]) & (t <= s["t1"])
        s["b"], s["sd"], s["n"] = {}, {}, {}
        for f in F:
            kb = k & ok[:, BOARD[f]]
            if kb.sum() >= 5:
                s["b"][f] = B[kb, BOARD[f]].mean(0)
                s["sd"][f] = float(np.linalg.norm(B[kb, BOARD[f]].std(0)))
                s["n"][f] = int(kb.sum())
    R = {"records": rec, "steps": len(S)}
    upn = {f: tip_uT_per_N(TIP_ID[f]) for f in F}
    R["uT_per_N"] = {f: round(v, 2) for f, v in upn.items()}

    # drift: the zero dwells (torque on, mounts home), interpolated in time per board
    torque_off = False
    zero = {f: [] for f in F}
    for s in S:
        if s["op"] == "torque":
            torque_off = s["state"] == "off"
        s["torque_off"] = torque_off
        if s["op"] == "joints" and s["family"] in ZERO_FAM and not torque_off and np.allclose(s["q_cmd"], 0):
            for f in s["b"]:
                zero[f].append((0.5 * (s["t0"] + s["t1"]), s["b"][f], s["sd"][f]))
    base, noise = {}, {}
    for f in F:
        tz = np.array([z[0] for z in zero[f]])
        bz = np.array([z[1] for z in zero[f]])
        base[f] = (tz, bz)
        loo = [np.linalg.norm(bz[i] - (bz[i - 1] + (bz[i + 1] - bz[i - 1]) * (tz[i] - tz[i - 1]) / (tz[i + 1] - tz[i - 1])))
               for i in range(1, len(tz) - 1)]
        noise[f] = dict(zero_dwells=len(tz), within_dwell_sd_uT=round(float(np.median([z[2] for z in zero[f]])), 2),
                        left_out_zero_uT_median=pct(loo, 50), left_out_zero_uT_p95=pct(loo, 95),
                        drift_span_uT=round(float(np.linalg.norm(bz.max(0) - bz.min(0))), 1),
                        hours=round(float((tz[-1] - tz[0]) / 3600), 2))
    R["noise"] = noise

    def change(s, f):
        tz, bz = base[f]
        if f not in s["b"] or not len(tz):
            return None
        tm = 0.5 * (s["t0"] + s["t1"])
        bl = np.array([np.interp(tm, tz, bz[:, c]) for c in range(15)])
        return s["b"][f] - bl

    for s in S:
        s["d"] = {f: change(s, f) for f in F}
        s["d"] = {f: v for f, v in s["d"].items() if v is not None}

    # torque off / on
    R["torque"] = {f: [round(float(np.linalg.norm(s["d"][f])), 1) for s in S if s["op"] == "torque" and f in s["d"]
                       and s["state"] == "off"] for f in F}
    # steppers
    st = []
    for s in S:
        if s["op"] == "mounts" and s["mounts"]:
            (mf, xy), = s["mounts"].items()
            st.append(dict(finger=mf, mount=xy, change_uT={f: round(float(np.linalg.norm(s["d"][f])), 1) for f in s["d"]}))
    R["stepper"] = st

    # single-finger families: self and cross terms
    fam = {}
    for mf in F:
        for kind in ("yaw", "mcp", "pip", "mcp_pip", "yaw_mcp"):
            ss = [s for s in S if s["family"] == f"{mf}_{kind}" and s["op"] == "joints"]
            if not ss:
                continue
            row = {"steps": len(ss)}
            for f in F:
                v = [np.linalg.norm(s["d"][f]) for s in ss if f in s["d"]]
                if v:
                    row[f] = dict(n=len(v), median_uT=pct(v, 50), max_uT=round(float(np.max(v)), 1),
                                  max_N=round(float(np.max(v) / upn[f]), 2))
            fam[f"{mf}_{kind}"] = row
    R["families"] = fam

    # thumb and middle toward each other
    pair = [s for s in S if s["family"].startswith("tm_") and s["op"] == "joints"]
    rows = []
    for s in pair:
        if "thumb" in s["d"] and "middle" in s["d"]:
            rows.append((s["clearance"], float(np.linalg.norm(s["d"]["thumb"])), float(np.linalg.norm(s["d"]["middle"]))))
    rows = np.array(rows, float)
    bands = ((0, 20), (20, 30), (30, 50), (50, 999))
    R["pair"] = {f"{a}-{b} mm": dict(n=int(((rows[:, 0] >= a) & (rows[:, 0] < b)).sum()),
                                     thumb_max_uT=round(float(rows[(rows[:, 0] >= a) & (rows[:, 0] < b), 1].max()), 1),
                                     middle_max_uT=round(float(rows[(rows[:, 0] >= a) & (rows[:, 0] < b), 2].max()), 1))
                 for a, b in bands if ((rows[:, 0] >= a) & (rows[:, 0] < b)).any()}

    # the tips' frames in the CAD hand (the lab's handgeo.py) at each step's measured joints
    sys.path.insert(0, str(N.parents[1] / "lab"))
    from handgeo import HOME_LOCAL, Hand
    h = Hand()

    def frames(q):
        h.set(HOME_LOCAL, {f: {j: float(q[3 * i + k]) for k, j in enumerate(J)} for i, f in enumerate(F)})
        return {f: (h.tip_frame(f).copy(), h.tip(f).copy()) for f in F}

    R0 = frames(np.zeros(9))

    def ambient(fr, f, pos):
        """Features of a uniform field fixed to the hand, seen in the tip's frame: the entries of R(q)^T - R(0)^T, with the
        tip's displacement (mm) for a field that varies across the hand."""
        x = (fr[f][0].T - R0[f][0].T).ravel()
        return np.r_[x, (fr[f][1] - R0[f][1]) / 10.0, 1.0] if pos else np.r_[x, 1.0]

    def lin_fit(A, Y):
        return np.linalg.lstsq(A, Y, rcond=None)[0]

    # subtraction from the joint angles, tested on the random poses
    sub, pair_res = {}, {}
    rng = np.random.default_rng(0)
    for f in ("thumb", "middle"):
        own = [3 * F.index(f) + i for i in range(3)]
        X, Y, fam_, cl, FR = [], [], [], [], []
        for s in S:
            if s["op"] != "joints" or f not in s["d"] or s["torque_off"] or not np.isfinite(s["q"]).all():
                continue
            if s["family"] in ZERO_FAM or s["family"].startswith(("test",)):
                continue
            X.append(s["q"])
            Y.append(s["d"][f])
            fam_.append(s["family"])
            cl.append(s["clearance"] if s["clearance"] is not None else np.nan)
            FR.append(frames(s["q"]))
        X, Y, fam_, cl = np.array(X), np.array(Y), np.array(fam_), np.array(cl, float)
        is_rand = fam_ == "random"
        alone = np.char.startswith(fam_.astype(str), f + "_")          # this finger alone, the others straight
        raw = np.linalg.norm(Y[is_rand], axis=1)
        out = dict(structured=int((~is_rand).sum()), random=int(is_rand.sum()), alone=int(alone.sum()),
                   change_uT=dict(median=pct(raw, 50), p95=pct(raw, 95), max=round(float(raw.max()), 1)),
                   change_N=dict(median=round(pct(raw, 50) / upn[f], 2), p95=round(pct(raw, 95) / upn[f], 2)))
        ri = np.flatnonzero(is_rand)
        folds = np.array_split(rng.permutation(ri), 5)
        pos_ = {j: k for k, j in enumerate(ri)}
        Arot = np.array([ambient(fr, f, False) for fr in FR])
        Apos = np.array([ambient(fr, f, True) for fr in FR])

        def score(name, fit, pred):
            e1 = np.linalg.norm(pred(fit(~is_rand), is_rand) - Y[is_rand], axis=1)
            e3 = np.linalg.norm(pred(fit(alone), is_rand) - Y[is_rand], axis=1)
            e2 = np.empty(len(ri))
            for te in folds:
                trm = np.ones(len(X), bool)
                trm[te] = False
                tem = ~trm
                e = np.linalg.norm(pred(fit(trm), tem) - Y[tem], axis=1)
                for j, v in zip(np.flatnonzero(tem), e):
                    e2[pos_[j]] = v
            for tag, e in (("structured_only", e1), ("alone_only", e3), ("five_folds", e2)):
                out[f"{name}_{tag}"] = dict(median_uT=pct(e, 50), p95_uT=pct(e, 95), max_uT=round(float(e.max()), 1),
                                            median_N=round(pct(e, 50) / upn[f], 3), p95_N=round(pct(e, 95) / upn[f], 3),
                                            over_0p25N=int((e / upn[f] > 0.25).sum()))

        for name, cols in (("gp_own_angles", own), ("gp_all_angles", list(range(9)))):
            fl = np.full(len(cols), 5.0)
            score(name, lambda tr, c=cols, fl=fl: BG.fit(X[tr][:, c], Y[tr], fl),
                  lambda m, te, c=cols: BG.predict(m, X[te][:, c]))
        for name, A in (("field_turned", Arot), ("field_turned_and_moved", Apos)):
            score(name, lambda tr, A=A: lin_fit(A[tr], Y[tr]), lambda W_, te, A=A: A[te] @ W_)
        # what is left once this finger's own angles are accounted for (GP on its three angles): the neighbours' share,
        # against the CAD clearance between the fingers. The thumb-middle steps are predicted from a GP trained on every
        # other step; the random poses from their five folds.
        tm = np.char.startswith(fam_.astype(str), "tm_")
        fl = np.full(3, 5.0)
        res = np.full(len(X), np.nan)
        m = BG.fit(X[~tm][:, own], Y[~tm], fl)
        res[tm] = np.linalg.norm(BG.predict(m, X[tm][:, own]) - Y[tm], axis=1)
        for te in folds:
            trm = np.ones(len(X), bool)
            trm[te] = False
            m = BG.fit(X[trm][:, own], Y[trm], fl)
            res[te] = np.linalg.norm(BG.predict(m, X[te][:, own]) - Y[te], axis=1)
        m = BG.fit(X[~alone][:, own], Y[~alone], fl)
        res[alone] = np.linalg.norm(BG.predict(m, X[alone][:, own]) - Y[alone], axis=1)
        rows = {}
        for lab, sel in (("pair", tm), ("random", is_rand)):
            for a, b in ((0, 15), (15, 20), (20, 30), (30, 50), (50, 999)):
                k = sel & (cl >= a) & (cl < b)
                if k.any():
                    rows[f"{lab} {a}-{b} mm"] = dict(n=int(k.sum()), median_uT=pct(res[k], 50),
                                                     max_uT=round(float(res[k].max()), 1),
                                                     max_N=round(float(res[k].max() / upn[f]), 2))
        k = alone
        rows["alone (fit)"] = dict(n=int(k.sum()), median_uT=pct(res[k], 50), max_uT=round(float(res[k].max()), 1))
        pair_res[f] = rows
        sub[f] = out
    R["subtract"] = sub
    R["neighbour_share"] = pair_res
    OUT.with_suffix(".json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
