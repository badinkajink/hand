#!/usr/bin/env python3
"""Crosstalk between v3 tips on the fingertip holder, measured on the UR5 and compared with the dipole model (owner,
2026-10-27: "a super complete calibration suite + crosstalk analysis"; the doc-writing session takes the results from
docs/experiments/20261027-v3_tip_calibration_and_crosstalk/feed.jsonl).

A session's streams: robot.bin (flange pose, 125 Hz), board.bin (the robot tip v3 ID 1's board) and held.bin (the two
held v3 tips' boards behind the mux, one channel each), with events.jsonl for the actions. The robot pad's frame P in B
is T_BE T_EM T_MP (the session's registration); each held pad's is T_BI ur5_cb3_cell.tiphold_T_IF(i) T_FP, with T_BI the
holder's frame measured by touch-offs (lab/cell.json's holder entry, or --frame). W is the robot apex's height above a
held apex (mm, along the holder's vertical; the hovers' gap), and the lateral offset is the robot apex's place across
it, in the holder's frame.

The model (--model nominal, the default) is the one of fingertip_crosstalk.py budget, fixed before these data: every
board's chips as v2 ID 1's rest fit placed them (same board, cassette and pad depth; its calib_rest.json), every tip's
four N52 magnets at nominal places with the checkerboard's poles (1, -1, -1, 1) and v2 ID 1's strength k. With --model
rest, each board's chips and its tip's magnets come from that board's own reading far from the other tips, fitted with
the chip order fixed (fit_rest); on 2026-10-27 those fits left 0.10 to 0.23 mT but put k at 0.40, 2.03 and 2.97 for three
identical tips, trading it against the chips' offset along x (-3.0 to +3.6 mm). The field of one tip's magnets at another's chips is fingertip_crosstalk.reading at their relative pose. A
rest fit leaves each tip's overall polarity to the sign of its chips' frame, so the comparison fits one gain g per
receiving board (its sign is that polarity; |g| near 1 means the model's strength holds).

Hovers (the map): each hover's neighbour field is the mean of a board's 15 components over the dwell at the pose less
the mean of the two dwells at its reference 40 mm above, before and after (which cancels a linear drift); the model's
the same from the robot's measured poses in those windows. Each hover is reported with its W, offset, station and heading
(the robot finger's long axis in the holder's x-y plane, deg; a twist changes it) from the robot's measured pose, and a
repeat ("-rep") with its first run's.

Touch-offs: the approach of each touch-off, from the top of its travel to the contact onset (touches.json's T_BE; the
deepest point when there was none), less 0.3 mm. Changes are taken from the top of the travel; the model's change from
the same samples. Reported by receiving board: the change's norm over the 15 components binned by W, measured and
model; g, and the share of the measured change's energy that g times the model explains, 1 - sum|M - g P|^2 / sum|M|^2,
over the samples at W under 10 mm and per bin (the change is taken from the far reading, so zero is its reference;
most samples lie near contact, where the approach is slow); and the norm left after subtracting it.

    .venv/bin/python scripts/fingertip_crosstalk_holder.py SESSION [SESSION ...] [--frame REG] [--stem NAME]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

os.environ.setdefault("UR5_MOUNT", "v7")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_crosstalk as XT  # noqa: E402
import fingertip_holder as FH  # noqa: E402
import fingertip_wrench_model as W  # noqa: E402
import ur5_cb3_cell as CELL  # noqa: E402

LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
V3 = ROOT / "docs/experiments/20261027-v3_tip_calibration_and_crosstalk"
HELD_DT = np.dtype([("t", "<f8"), ("b", "<u1"), ("r", "<f4", (20,))])
BOARD_DT = np.dtype([("t", "<f8"), ("r", "<f4", (20,))])
APEX_P = np.array([8.5, 0.0, 10.86])
T_PF = np.array([[-1.0, 0, 0, -2.05], [0, 1.0, 0, 0], [0, 0, -1.0, -77.05], [0, 0, 0, 1.0]])   # pad frame P in F
NEAR_W = 10.0


def load(sd):
    sd = Path(sd)
    ev = [json.loads(s) for s in (sd / "events.jsonl").read_text().splitlines() if s.strip()]
    reg = json.loads((sd / "registration.json").read_text())
    R = np.fromfile(sd / "robot.bin", dtype="<f8").reshape(-1, 15)
    T = np.zeros((len(R), 4, 4))
    T[:, :3, :3] = Rotation.from_rotvec(R[:, 11:14]).as_matrix()
    T[:, :3, 3] = R[:, 8:11] * 1e3
    T[:, 3, 3] = 1.0
    touches = json.loads((sd / "touches.json").read_text()) if (sd / "touches.json").exists() else []
    return dict(name=sd.name, ev=ev, reg=reg, t_r=R[:, 0], T_BE=T, board=np.fromfile(sd / "board.bin", dtype=BOARD_DT),
                held=np.fromfile(sd / "held.bin", dtype=HELD_DT), touches={t["id"]: t for t in touches})


def holder_frame(path=None):
    if path is None:
        cell = json.loads((LAB / "cell.json").read_text())
        path = LAB / cell["fixtures"]["holder"]["registration"]
    return np.array(json.loads(Path(path).read_text())["T_BI"], float), str(Path(path).relative_to(ROOT))


def stations():
    return {("+y" if CELL.TIPHOLD["stations"][i][0] > 0 else "-y"): i for i in range(len(CELL.TIPHOLD["stations"]))}


def windows(ses, prefix):
    out, open_ = [], {}
    for e in ses["ev"]:
        if e.get("event") == "start" and str(e.get("id", "")).startswith(prefix):
            open_[e["id"]] = e["t"]
        elif e.get("event") == "end" and e.get("id") in open_:
            out.append((e["id"], open_.pop(e["id"]), e["t"], e.get("status")))
    return out


PERM, S = (0, 2, 4, 1, 3), 1      # the 5X board's stream order and sign, as v2 ID 1's rest fit found them (calib_rest.json)


def fit_rest(r15, free=False):
    """The four-magnet model fitted to a board's reading at rest: pole pattern, k, offsets, the chips' common turn and
    the background field, with the board's chip order fixed (the checkerboard's symmetry lets a free fit swap chips that
    the rest field cannot tell apart); free=True searches the order as fingertip_wrench_model.calibrate does."""
    from scipy.optimize import least_squares
    b = np.asarray(r15, float).reshape(5, 3)
    if free:
        cal, fits = W.calibrate(b, case="four_05R4", verbose=False)
        return cal, fits[1]["rms_mT"] if len(fits) > 1 else float("nan")
    row = W.fe_row("four_05R4")
    fits = []
    for pl in [(1,) + c for c in __import__("itertools").product((1, -1), repeat=3)]:
        best = None
        for th0 in (0.0, 90.0, 180.0, 270.0):
            r = least_squares(lambda q: W._resid(b, row, pl, S, PERM, q), [1.0, 0.0, 0.0, 0.0, th0, 0, 0, 0],
                              bounds=([0.1, -3, -4, -4, th0 - 60, -100, -100, -100], [4.0, 8, 4, 4, th0 + 60, 100, 100, 100]))
            if best is None or r.cost < best.cost:
                best = r
        fits.append((float(np.sqrt(np.mean(best.fun ** 2))), pl, best.x))
    fits.sort(key=lambda f: f[0])
    rms, pl, x = fits[0]
    cal = W.Calib(case="four_05R4", poles=tuple(pl), s=S, theta0=float(x[4] % 360), perm=PERM, k=float(x[0]),
                  dx=float(x[1]), dy=float(x[2]), dz=float(x[3]), g_uT=tuple(float(v) for v in x[5:8]), rms_mT=rms)
    return cal, fits[1][0]


def predict(cal_recv, cal_src, T_B_Precv, T_B_Psrc):
    """Field (uT, (n, 15)) of the source tip's magnets at the receiving board's chips, for source pad frames (n, 4, 4)."""
    model = W.Model(cal_recv, W.fe_row("four_05R4"))
    src = XT.source(W.fe_row("four_05R4"), cal_src.poles, cal_src.k)
    T = np.linalg.inv(T_B_Precv)[None] @ T_B_Psrc
    return np.array([XT.reading(model, src, Ti[:3, :3], Ti[:3, 3]) for Ti in T])


def nearest(t_src, t_query):
    i = np.clip(np.searchsorted(t_src, t_query), 1, len(t_src) - 1)
    return np.where(np.abs(t_src[i - 1] - t_query) < np.abs(t_src[i] - t_query), i - 1, i)


def nominal_calib():
    c = W.Calib.load(XT.V2 / "calib_rest.json")
    return W.Calib(case="four_05R4", poles=XT.FOUR_POLES, s=c.s, theta0=c.theta0, perm=c.perm, k=c.k, dx=c.dx, dy=c.dy,
                   dz=c.dz, g_uT=(0.0, 0.0, 0.0), rms_mT=float("nan"))


def analyse(sessions, frame_path, stem, model="nominal"):
    T_BI, frame_rel = holder_frame(frame_path)
    st = stations()
    T_B_Pheld = {k: T_BI @ CELL.tiphold_T_IF(i=i) @ np.linalg.inv(T_PF) for k, i in st.items()}
    apex_I = {k: FH.held_apex(CELL, i) for k, i in st.items()}
    out = dict(script="scripts/fingertip_crosstalk_holder.py", made=time.strftime("%Y-%m-%d %H:%M"), frame=frame_rel, model=model,
               sessions=[], touches=[], boards={})
    samples = {}                                  # receiving board -> list of (W, lateral, meas (15), model (15), touch)
    for sd in sessions:
        ses = load(sd)
        T_EP = np.array(ses["reg"]["T_EM"]) @ np.array(ses["reg"]["T_MP"])
        T_B_Prob = ses["T_BE"] @ T_EP[None]
        ap_B = T_B_Prob[:, :3, :3] @ APEX_P + T_B_Prob[:, :3, 3]
        ap_I = (ap_B - T_BI[:3, 3]) @ T_BI[:3, :3]                     # the robot apex in the holder's frame
        H = ses["held"]
        chans = sorted(int(c) for c in np.unique(H["b"]))
        # far samples: the robot apex 60 mm or more from both held apexes
        far_r = np.min([np.linalg.norm(ap_I - a, axis=1) for a in apex_I.values()], axis=0) > 60.0
        t_far = ses["t_r"][far_r]
        rest = {}
        for c in chans:
            m = H["b"] == c
            k = far_r[nearest(ses["t_r"], H["t"][m])]
            rest[f"held{c}"] = np.median(H["r"][m][k].reshape(-1, 5, 4)[:, :, 1:], axis=0).ravel() if k.sum() > 50 else None
        k = far_r[nearest(ses["t_r"], ses["board"]["t"])]
        rest["robot"] = np.median(ses["board"]["r"][k].reshape(-1, 5, 4)[:, :, 1:], axis=0).ravel() if k.sum() > 50 else None
        cals = {}
        for b, r15 in rest.items():
            if r15 is None:
                continue
            if b in out["boards"]:
                cals[b] = W.Calib(**{**out["boards"][b]["calib"], "poles": tuple(out["boards"][b]["calib"]["poles"]),
                                     "perm": tuple(out["boards"][b]["calib"]["perm"]),
                                     "g_uT": tuple(out["boards"][b]["calib"]["g_uT"])})
                continue
            cal, rms_next = fit_rest(r15)
            if model == "nominal":
                cal = nominal_calib()
            cals[b] = cal
            out["boards"][b] = dict(calib={k2: (list(v) if isinstance(v, tuple) else v) for k2, v in cal.__dict__.items()
                                           if k2 != "b_rest_uT"}, rest_uT=np.round(r15, 1).tolist(),
                                    rms_mT=round(cal.rms_mT, 4), rms_next_mT=round(rms_next, 4), session=ses["name"])
            print(f"{ses['name']} {b}: rest fit rms {cal.rms_mT * 1e3:.0f} uT (next pattern {rms_next * 1e3:.0f}), poles "
                  f"{cal.poles}, k {cal.k:.3f}", flush=True)
        out["sessions"].append(dict(name=ses["name"], channels=chans, far_samples=int(far_r.sum()),
                                    t0=float(ses["t_r"][0]), t1=float(ses["t_r"][-1])))
        if "robot" not in cals:
            continue
        # each station's held channel: the one whose reading changes most from its far reading while the robot apex is
        # within 6 mm of that station's apex (and by 5 uT or more); a channel holds one station
        score = {}
        for c in chans:
            b = f"held{c}"
            if b not in cals:
                continue
            m = H["b"] == c
            jr = nearest(ses["t_r"], H["t"][m])
            rr = H["r"][m].reshape(-1, 5, 4)[:, :, 1:].reshape(-1, 15)
            for k, ap in apex_I.items():
                nk = np.linalg.norm(ap_I[jr] - ap, axis=1) < 6.0
                if nk.sum() > 50:
                    score[(b, k)] = round(float(np.linalg.norm(np.median(rr[nk], axis=0) - rest[b])), 2)
        taken = {v["station"] for v in out["boards"].values() if v.get("station")}
        for (b, k), v in sorted(score.items(), key=lambda kv: -kv[1]):
            if v > 5.0 and k not in taken and not out["boards"][b].get("station"):
                out["boards"][b]["station"] = k
                out["boards"][b]["station_evidence_uT"] = {kk: vv for (bb, kk), vv in score.items() if bb == b}
                taken.add(k)
        model_r = W.Model(cals["robot"], W.fe_row("four_05R4"))
        for tid, a_, b_, status in windows(ses, "hold-touch"):
            idx = np.where((ses["t_r"] >= a_) & (ses["t_r"] <= b_))[0]
            if len(idx) < 50:
                continue
            dist = {k: np.linalg.norm(ap_I[idx] - v, axis=1) for k, v in apex_I.items()}
            stk = min(dist, key=lambda k: dist[k].min())
            Wz = ap_I[idx, 2] - apex_I[stk][2]
            top = idx[int(np.argmax(Wz))]
            tc = ses["touches"].get(tid)
            if tc is not None and tc.get("T_BE") is not None:
                Tc = np.array(tc["T_BE"]) @ T_EP
                Wc = float(((Tc[:3, :3] @ APEX_P + Tc[:3, 3] - T_BI[:3, 3]) @ T_BI[:3, :3])[2] - apex_I[stk][2])
            else:
                Wc = float(Wz.min())
            after = idx[idx >= top]
            Wa = ap_I[after, 2] - apex_I[stk][2]
            hit = np.where(Wa <= Wc + 0.3)[0]
            reach = after[hit[0]] if len(hit) else after[-1]
            appr = idx[(idx >= top) & (idx <= reach)]
            if len(appr) < 20:
                continue
            lat = ap_I[appr, :2] - apex_I[stk][:2]
            rec = dict(session=ses["name"], id=tid, station=stk, status=status, W_top_mm=round(float(Wz.max()), 2),
                       W_onset_mm=round(Wc, 3), lateral_at_onset_mm=np.round(lat[-1], 2).tolist(), boards={})
            t_a, t_b = ses["t_r"][appr[0]], ses["t_r"][appr[-1]]
            for c in chans:                                       # the held boards see the robot tip's magnets
                b = f"held{c}"
                if b not in cals or out["boards"][b].get("station") != stk:
                    continue
                hm = (H["b"] == c) & (H["t"] >= t_a) & (H["t"] <= t_b)
                rr = H["r"][hm].reshape(-1, 5, 4)[:, :, 1:].reshape(-1, 15)
                j = appr[np.clip(np.searchsorted(ses["t_r"][appr], H["t"][hm]), 0, len(appr) - 1)]
                pr = predict(cals[b], cals["robot"], T_B_Pheld[stk], T_B_Prob[j])
                p0 = predict(cals[b], cals["robot"], T_B_Pheld[stk], T_B_Prob[appr[:1]])[0]
                r0 = rr[:100].mean(0)
                samples.setdefault(b, []).append((ap_I[j, 2] - apex_I[stk][2], ap_I[j, :2] - apex_I[stk][:2], rr - r0,
                                                  pr - p0, tid))
                rec["boards"][b] = dict(n=int(len(rr)),
                                        meas_at_onset_uT=round(float(np.linalg.norm((rr - r0)[-50:].mean(0))), 2),
                                        model_at_onset_uT=round(float(np.linalg.norm((pr - p0)[-50:].mean(0))), 2))
            # the robot board sees both held tips' magnets
            src = [(out["boards"][b]["station"], cals[b]) for b in cals
                   if b.startswith("held") and out["boards"][b].get("station")]
            bm = (ses["board"]["t"] >= t_a) & (ses["board"]["t"] <= t_b)
            rb = ses["board"]["r"][bm].reshape(-1, 5, 4)[:, :, 1:].reshape(-1, 15)
            if len(rb) > 50 and src:
                j = appr[np.clip(np.searchsorted(ses["t_r"][appr], ses["board"]["t"][bm]), 0, len(appr) - 1)]
                prs, p0 = np.zeros((len(j), 15)), np.zeros(15)
                for k, cs in src:
                    sq = XT.source(W.fe_row("four_05R4"), cs.poles, cs.k)
                    for n_, Ti in enumerate(np.linalg.inv(T_B_Prob[j]) @ T_B_Pheld[k][None]):
                        prs[n_] += XT.reading(model_r, sq, Ti[:3, :3], Ti[:3, 3])
                    Ti = np.linalg.inv(T_B_Prob[appr[0]]) @ T_B_Pheld[k]
                    p0 += XT.reading(model_r, sq, Ti[:3, :3], Ti[:3, 3])
                r0 = rb[:70].mean(0)
                samples.setdefault("robot", []).append((ap_I[j, 2] - apex_I[stk][2], ap_I[j, :2] - apex_I[stk][:2],
                                                        rb - r0, prs - p0, tid))
                rec["boards"]["robot"] = dict(n=int(len(rb)),
                                              meas_at_onset_uT=round(float(np.linalg.norm((rb - r0)[-50:].mean(0))), 2),
                                              model_at_onset_uT=round(float(np.linalg.norm((prs - p0)[-50:].mean(0))), 2))
            out["touches"].append(rec)
            print(f"{ses['name']} {tid} ({stk}): W {Wz.max():.1f} -> {Wc:.2f} mm, lateral {np.round(lat[-1], 2).tolist()}; "
                  + "; ".join(f"{b} {v['meas_at_onset_uT']:.1f} vs model {v['model_at_onset_uT']:.1f} uT"
                              for b, v in rec["boards"].items()), flush=True)
        # hovers: the dwell at the pose against the reference dwells
        held_st = {out["boards"][b]["station"]: b for b in cals if b.startswith("held") and out["boards"][b].get("station")}
        for e in ses["ev"]:
            rec = e.get("rec") if e.get("event") == "end" and e.get("status") == "ok" else None
            if not rec or "t_pose" not in rec:
                continue
            win = {k: rec[k] for k in ("t_ref0", "t_pose", "t_ref1")}
            jp = np.where((ses["t_r"] >= win["t_pose"][0]) & (ses["t_r"] <= win["t_pose"][1]))[0]
            jr = np.where(((ses["t_r"] >= win["t_ref0"][0]) & (ses["t_r"] <= win["t_ref0"][1]))
                          | ((ses["t_r"] >= win["t_ref1"][0]) & (ses["t_r"] <= win["t_ref1"][1])))[0]
            if len(jp) < 10 or len(jr) < 10:
                continue
            pa = ap_I[jp].mean(0)
            stk = min(apex_I, key=lambda k: np.linalg.norm(pa - apex_I[k]))
            Tp = T_B_Prob[jp[len(jp) // 2]]
            Tr = T_B_Prob[jr[len(jr) // 2]]
            zP = (np.linalg.inv(T_BI) @ Tp)[:3, 2]                  # the robot finger's long axis in the holder's frame
            yaw = float(np.degrees(np.arctan2(zP[1], zP[0])))
            hv = dict(session=ses["name"], id=e["id"], station=stk, W_mm=round(float(pa[2] - apex_I[stk][2]), 3),
                      lateral_mm=np.round(pa[:2] - apex_I[stk][:2], 2).tolist(), heading_deg=round(yaw, 2),
                      peak_N=round(float(rec.get("peak_N", 0.0)), 3), boards={})

            def mean_in(t, r, w):
                m = (t >= w[0]) & (t <= w[1])
                return r[m].mean(0) if m.sum() > 5 else None

            for b in [bb for bb in cals if bb.startswith("held")]:
                c = int(b[4:])
                m = H["b"] == c
                t_h, r_h = H["t"][m], H["r"][m].reshape(-1, 5, 4)[:, :, 1:].reshape(-1, 15)
                mp, m0, m1 = (mean_in(t_h, r_h, win[k]) for k in ("t_pose", "t_ref0", "t_ref1"))
                if mp is None or m0 is None or m1 is None:
                    continue
                st_b = out["boards"][b].get("station")
                if st_b is None:
                    continue
                pp = predict(cals[b], cals["robot"], T_B_Pheld[st_b], np.stack([Tp, Tr]))
                d = mp - 0.5 * (m0 + m1)
                hv["boards"][b] = dict(station=st_b, meas=np.round(d, 2).tolist(), meas_uT=round(float(np.linalg.norm(d)), 2),
                                       model_uT=round(float(np.linalg.norm(pp[0] - pp[1])), 2),
                                       model=np.round(pp[0] - pp[1], 2).tolist(),
                                       ref_drift_uT=round(float(np.linalg.norm(m1 - m0)), 2))
            rb_t, rb_r = ses["board"]["t"], ses["board"]["r"].reshape(-1, 5, 4)[:, :, 1:].reshape(-1, 15)
            mp, m0, m1 = (mean_in(rb_t, rb_r, win[k]) for k in ("t_pose", "t_ref0", "t_ref1"))
            if mp is not None and m0 is not None and m1 is not None and held_st:
                pm = np.zeros(15)
                for k, b in held_st.items():
                    sq = XT.source(W.fe_row("four_05R4"), cals[b].poles, cals[b].k)
                    for sign, T_ in ((1.0, Tp), (-1.0, Tr)):
                        Ti = np.linalg.inv(T_) @ T_B_Pheld[k]
                        pm += sign * XT.reading(model_r, sq, Ti[:3, :3], Ti[:3, 3])
                d = mp - 0.5 * (m0 + m1)
                hv["boards"]["robot"] = dict(meas=np.round(d, 2).tolist(), meas_uT=round(float(np.linalg.norm(d)), 2),
                                             model_uT=round(float(np.linalg.norm(pm)), 2), model=np.round(pm, 2).tolist(),
                                             ref_drift_uT=round(float(np.linalg.norm(m1 - m0)), 2))
            out.setdefault("hovers", []).append(hv)
    # fits and bins per receiving board
    edges = np.r_[np.arange(-1.0, 5.0, 0.5), np.arange(5.0, 21.0, 1.0)]
    for b, rows in samples.items():
        Wv = np.concatenate([r[0] for r in rows])
        M = np.concatenate([r[2] for r in rows])
        P = np.concatenate([r[3] for r in rows])
        near = Wv < NEAR_W
        g = float(np.sum(M[near] * P[near]) / np.sum(P[near] ** 2)) if near.any() else float("nan")
        res = M - g * P
        r2 = 1.0 - float(np.sum(res[near] ** 2) / np.sum(M[near] ** 2)) if near.any() else float("nan")
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (Wv >= lo) & (Wv < hi)
            if s.sum() < 10:
                continue
            bins.append(dict(W_mm=round(float(Wv[s].mean()), 2), n=int(s.sum()),
                             meas_uT=round(float(np.linalg.norm(M[s].mean(0))), 2),
                             model_uT=round(float(np.linalg.norm(P[s].mean(0))), 2),
                             gmodel_uT=round(float(np.linalg.norm(g * P[s].mean(0))), 2),
                             left_uT=round(float(np.linalg.norm(res[s].mean(0))), 2),
                             explained=round(1.0 - float(np.sum(res[s] ** 2) / np.sum(M[s] ** 2)), 3)))
        out["boards"].setdefault(b, {}).update(gain=round(g, 3), explained_near=round(r2, 3), n_near=int(near.sum()),
                                               bins=bins)
        print(f"{b}: g {g:+.3f}, explains {r2:.3f} of the change's energy over {int(near.sum())} samples at W < "
              f"{NEAR_W:g} mm", flush=True)
    data = V3 / "data" / f"{stem}.json"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps(out, indent=1) + "\n")
    print("->", data.relative_to(ROOT))
    figure(out, samples, V3 / "media" / f"{stem}.png")
    return out


def figure(out, samples, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    boards = [b for b in ("held0", "held3", "held1", "held2", "robot") if b in out["boards"] and out["boards"][b].get("bins")]
    if not boards:
        return
    fig, axs = plt.subplots(1, len(boards), figsize=(5.2 * len(boards), 4.2), squeeze=False)
    for ax, b in zip(axs[0], boards):
        bins = out["boards"][b]["bins"]
        Wb = [x["W_mm"] for x in bins]
        v = out["boards"][b]
        ax.plot(Wb, [x["meas_uT"] for x in bins], "o-", color="k", ms=4, label="measured")
        ax.plot(Wb, [x["model_uT"] for x in bins], "s--", color="tab:blue", ms=3, label="dipole model")
        ax.plot(Wb, [x["left_uT"] for x in bins], "^:", color="tab:red", ms=3,
                label=f"left after {v['gain']:+.2f} x model ({100 * v['explained_near']:.0f} % explained)")
        st = out["boards"][b].get("station", "")
        who = "robot tip's board (v3 ID 1)" if b == "robot" else f"held board, {st} station"
        ax.set_title(f"{who}", fontsize=13)
        ax.set_xlabel("W, robot apex above the held apex (mm)", fontsize=13)
        ax.set_ylabel("change, norm of 15 components (uT)", fontsize=13)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print("->", path.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--frame", default=None, help="the holder's registration.json (default: lab/cell.json's holder entry)")
    ap.add_argument("--stem", default="20261027-crosstalk_touchoffs")
    ap.add_argument("--model", default="nominal", choices=("nominal", "rest"))
    a = ap.parse_args()
    analyse(a.sessions, a.frame, a.stem, a.model)


if __name__ == "__main__":
    main()
