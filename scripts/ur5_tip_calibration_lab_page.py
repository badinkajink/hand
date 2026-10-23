#!/usr/bin/env python3
"""Page of the first calibration runs of the fingertip on the lab's UR5 (2026-10-22): the runs, the HEX-E's link, the
wrist's stop at 19:59 and the checks it led to, the twist that reached the homing key, the pad's touch-offs and presses
of sphere2, and the board's rest reading against the tool's orientation.

Reads the run folders in docs/experiments/20261022-ur5_tip_calibration/lab/ (events, registration, streams), the plan
and the overhead camera's frames; writes the figures to media/ and
docs/experiments/20261022-ur5_tip_calibration/20261022-ur5_tip_calibration_lab_runs.html.

    /usr/bin/python3 scripts/ur5_tip_calibration_lab_page.py [--film logs/20261022-ur5_camera/sphere2]
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import ur5_tip_calibration_page as P  # noqa: E402   (esc, link, code, fig, film, table, plt_setup)

D = ROOT / "docs/experiments/20261022-ur5_tip_calibration"
LAB = D / "lab"
MEDIA = D / "media"
PAGE = D / "20261022-ur5_tip_calibration_lab_runs.html"
TEMPLATE = ROOT / "scripts/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
RUNS = ["20261022-192715-first", "20261022-195420-first2", "20261022-202551-sphere1", "20261022-204657-tcp1",
        "20261022-205059-tcp2", "20261022-210852-sphere2"]
PARTS = {"20261022-210852-sphere2": [("sphere2", "preflight,homing,pad"), ("sphere2r", "preflight,pad,register,pivot"),
                                     ("sphere2g", "preflight,grid,roll,twist")]}
SPHERE2G_END = "ended at 23:27 after the twists"
FT_DT = np.dtype([("t", "<f8"), ("seq", "<f8"), ("status", "<f8"), ("F", "<f8", (3,)), ("T", "<f8", (3,))])
BOARD_DT = np.dtype([("t", "<f8"), ("r", "<f4", (20,))])


def hm(t):
    return datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S")


def events(run):
    return [json.loads(s) for s in open(LAB / run / "events.jsonl")]


def ft(run):
    p = LAB / run / "ft.bin"
    return np.fromfile(p, FT_DT) if p.exists() else np.zeros(0, FT_DT)


def robot(run):
    p = LAB / run / "robot.bin"
    return np.fromfile(p, "<f8").reshape(-1, 15) if p.exists() else np.zeros((0, 15))


# ------------------------------------------------------------------------------------------------ numbers
def run_rows():
    """One row per run: start, stages asked, actions ended ok and aborted. A resumed session gives one row per run, cut
    at its preflights, with the names and stages of PARTS."""
    out = []
    for run in RUNS:
        ev = events(run)
        ses = json.loads((LAB / run / "session.json").read_text())
        cuts = [i for i, e in enumerate(ev) if e.get("event") == "start" and e.get("id") == "preflight"] or [0]
        parts = PARTS.get(run, [(run.split("-")[-1], ses.get("stages", ""))])
        for (name, stages), i0, i1 in zip(parts, cuts, cuts[1:] + [len(ev)]):
            part = ev[i0:i1]
            ok = [e["id"] for e in part if e.get("event") == "end" and e.get("status") == "ok"
                  and e.get("id") not in ("preflight", "homing")]
            ab = [(e["id"], e.get("why", "")) for e in part if e.get("event") == "end" and e.get("status") == "abort"]
            out.append(dict(run=run, name=name, start=hm(part[0]["t"])[:5], stages=stages, ok=len(ok), aborted=len(ab),
                            aborts=ab))
    return out


def part_name(run, t):
    """The name of the run that was going at time t: a resumed session's part from PARTS, cut at its preflights."""
    parts = PARTS.get(run)
    if not parts:
        return run.split("-")[-1]
    starts = [e["t"] for e in events(run) if e.get("event") == "start" and e.get("id") == "preflight"]
    k = max([0] + [i for i, s in enumerate(starts) if s <= t + 1.0])
    return parts[min(k, len(parts) - 1)][0]


def gaps(run, min_s=0.05):
    """Gaps of the HEX-E stream over min_s: start, length, change of Fz, and whether the sensor restarted (it came back
    untared: the status's 0x40000 bit newly set, or Fz past 100 N from before). A gap in which a new run of the program
    began (a resumed session) is marked between=True and is no fault."""
    f = ft(run)
    if len(f) < 2:
        return []
    dt = np.diff(f["t"])
    starts = [e["t"] for e in events(run) if e.get("event") == "start" and e.get("id") == "preflight"]
    out = []
    for i in np.nonzero(dt > min_s)[0]:
        if i == 0:
            continue
        t0, t1 = float(f["t"][i]), float(f["t"][i + 1])
        dFz = float(f["F"][i + 1, 2] - f["F"][i, 2])
        between = any(t0 - 1.0 <= s <= t1 + 1.0 for s in starts)
        untared = bool(int(f["status"][i + 1]) & 0x40000) and not int(f["status"][i]) & 0x40000
        out.append(dict(t=t0, s=t1 - t0, between=between, restart=None if between else bool(untared or abs(dFz) > 100.0),
                        dFz=dFz))
    t_end = max(e["t"] for e in events(run))
    if f["t"][-1] < t_end - 1.0:                                  # the stream ended before the run: a last, open gap
        out.append(dict(t=float(f["t"][-1]), s=float(t_end - f["t"][-1]), restart=None, dFz=None, open=True))
    return out


def earth_field(run="20261022-195420-first2"):
    """A uniform world field fitted to the board's rest readings at the gravity stations (each chip: an offset and a
    rotation; the field shared), after the chips' temperature correction (fingertip_wrench_model.thermal_correct)."""
    import fingertip_wrench_model as W
    kap = W.load_thermal()
    recs = [e["rec"] for e in events(run) if e.get("event") == "end" and e.get("status") == "ok"
            and str(e.get("id", "")).startswith("grav")]
    R = np.array([r["R_BE"] for r in recs])
    Bd = np.array([r["board"] for r in recs]).reshape(-1, 5, 4)
    T, B = Bd[:, :, 0], Bd[:, :, 1:]
    Bc = W.thermal_correct(B, T, kap)
    n = len(R)

    def procrustes(M):
        U, _, Vt = np.linalg.svd(M)
        return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt

    w = np.array([0.0, 0.0, -50.0])
    for _ in range(300):
        U_ = np.einsum("nji,j->ni", R, w)
        A, b0 = [], []
        for k in range(5):
            sol, *_ = np.linalg.lstsq(np.c_[U_, np.ones(n)], Bc[:, k], rcond=None)
            Ak = procrustes(sol[:3].T)
            A.append(Ak)
            b0.append((Bc[:, k] - U_ @ Ak.T).mean(0))
        M = np.vstack([A[k] @ R[i].T for k in range(5) for i in range(n)])
        rhs = np.concatenate([Bc[i, k] - b0[k] for k in range(5) for i in range(n)])
        w_new, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        if np.linalg.norm(w_new - w) < 1e-7:
            w = w_new
            break
        w = w_new
    U_ = np.einsum("nji,j->ni", R, w)
    res = np.array([[Bc[i, k] - b0[k] - A[k] @ U_[i] for k in range(5)] for i in range(n)])
    ang = [math.degrees(math.acos(np.clip((np.trace(A[0].T @ A[k]) - 1) / 2, -1, 1))) for k in range(5)]
    return dict(n=n, w=w.tolist(), w_norm=float(np.linalg.norm(w)), dip=float(math.degrees(math.atan2(-w[2], math.hypot(w[0], w[1])))),
                spread=np.ptp(Bc, axis=0).max(1).tolist(), rms=np.sqrt((res ** 2).mean(axis=(0, 2))).tolist(),
                rms_all=float(np.sqrt((res ** 2).mean())), chip_deg=ang, T_range=[float(T.min()), float(T.max())])


def self_map():
    """Least distance (mm) of the HEX-E's cable port, its body and the mount and finger to the robot's forearm and wrist
    links, over joint 4 and joint 6, for each joint 5 (ur5_cb3_cell.self_clearance)."""
    import ur5_cb3_cell as CELL
    C = CELL.C
    T_EM = C.Tm(C.rot(C.EZ, -92.15), [0.0, 0.0, 35.0])
    base = np.radians([48.3, -86.25, 109.55, 0.0, 0.0, 0.0])
    q5s = np.arange(-180, 181, 10)
    out = {k: [] for k in ("port", "hex_e", "tool")}
    for q5 in q5s:
        worst = dict(port=1e9, hex_e=1e9, tool=1e9)
        for q4 in np.arange(-180, 181, 30):
            for q6 in np.arange(-180, 180, 30):
                q = base.copy()
                q[3], q[4], q[5] = np.radians([q4, q5, q6])
                sc = CELL.self_clearance(q, T_EM)
                for k in worst:
                    worst[k] = min(worst[k], sc[k])
        for k in out:
            out[k].append(worst[k])
    at_stop = CELL.self_clearance(np.radians([82.8, -55.9, 80.0, -30.3, 173.5, 1.1]), T_EM)
    return dict(q5=q5s.tolist(), **out, at_stop=at_stop)


def twist_trace(run="20261022-205059-tcp2", act="twist-0-1.0N"):
    """The HEX-E's force (sensor frame, from its value 1 s before the turn) and joint 5 along tcp2's first twist, and the
    mount's least clearance to the cell along it (the session's registration)."""
    import ur5_cb3_cell as CELL
    C = CELL.C
    ev = events(run)
    t0 = [e["t"] for e in ev if e.get("id") == act and e.get("event") == "start"][0]
    t1 = [e["t"] for e in ev if e.get("id") == act and e.get("event") == "end"][0]
    f, r = ft(run), robot(run)
    m = (f["t"] > t0) & (f["t"] < t1 + 1.0)
    mr = (r[:, 0] > t0) & (r[:, 0] < t1 + 1.0)
    q5 = np.degrees(r[mr, 6])
    reg = json.loads((LAB / run / "registration.json").read_text())
    T_BI, T_EM = np.array(reg["T_BI"]), np.array(reg["T_EM"])
    lab_cell()
    ts, mount = [], []
    for i in np.nonzero(mr)[0][::25]:
        cl = CELL.clearance(r[i, 2:8], CELL.SIM.fk(r[i, 2:8]), T_BI, T_EM, "sphere")
        ts.append(r[i, 0] - t0)
        mount.append(min(cl["adapter"], cl["fork"], cl["clamp_plate"]))
    F = f["F"][m]
    k_turn = int(np.argmax(np.abs(np.diff(q5)) > 0.02)) if len(q5) > 1 else 0
    t_turn = r[mr, 0][k_turn] - t0
    F0 = F[(f["t"][m] - t0 > t_turn - 1.5) & (f["t"][m] - t0 < t_turn - 0.5)].mean(0)
    dF = F - F0
    k = int(np.argmax(np.linalg.norm(dF, axis=1)))
    return dict(t=(f["t"][m] - t0).tolist(), dF=dF.tolist(), tq=(r[mr, 0] - t0).tolist(), q5=q5.tolist(),
                t_turn=float(t_turn), peak_N=float(np.linalg.norm(dF[k])), t_peak=float(f["t"][m][k] - t0),
                tm=ts, mount=mount, why=[e.get("why") for e in ev if e.get("id") == act and e.get("event") == "end"][0])


def turn_rows(run="20261022-210852-sphere2", every_s=1.0):
    """Each roll and twist of run: how it ended, its press, the turn about its move's axis from the pressed pose (deg),
    the change of the HEX-E's force from its tare (N, the press included), joint 5's range (deg), and the mount's least
    clearance to the cell and the homing key in the model (mm, every every_s along the action, with the session's
    registration)."""
    import ur5_cb3_cell as CELL
    import ur5_tip_calibration as C
    lab_cell()
    ev = events(run)
    f, r = ft(run), robot(run)
    reg = json.loads((LAB / run / "registration.json").read_text())
    T_BI, T_EM = np.array(reg["T_BI"]), np.array(reg["T_EM"])
    plan = {a["id"]: a for a in json.loads((LAB / run / "plan_resume_2152.json").read_text())["actions"]}
    starts = {e["id"]: e["t"] for e in ev if e.get("event") == "start"}
    out = []
    for e in ev:
        if e.get("event") != "end" or not str(e.get("id", "")).startswith(("roll-", "twist-")):
            continue
        rec, t0, t1 = e.get("rec") or {}, starts[e["id"]], e["t"]
        mr, mf = (r[:, 0] >= t0) & (r[:, 0] <= t1), (f["t"] >= t0) & (f["t"] <= t1)
        row = dict(id=e["id"], kind=e["id"].split("-")[0], status=e.get("status"), why=e.get("why", ""), s=float(t1 - t0),
                   press_N=plan.get(e["id"], {}).get("press_N"))
        if mr.sum() < 10 or not rec.get("moves"):
            out.append(row)
            continue
        axis = np.array(rec["moves"][0]["axis_B"])
        R_stop = np.array(rec["T_stop"])[:3, :3]
        rr = r[mr][::25]
        rv = np.array([C.log_so3(C.ur_to_T(v[8:14])[:3, :3] @ R_stop.T) for v in rr])
        ang = np.degrees(rv @ axis)                                # about the first move's axis, for the figure
        tare = np.array(rec["tare"])[:3]
        dF = np.linalg.norm(f["F"][mf] - tare, axis=1)
        q5 = np.degrees(r[mr, 6])
        step = max(1, int(round(every_s * 125)))
        mount = [min(cl["adapter"], cl["fork"], cl["clamp_plate"]) for cl in
                 (CELL.clearance(v[2:8], CELL.SIM.fk(v[2:8]), T_BI, T_EM, "sphere") for v in r[mr][::step])]
        tp = float(rec.get("t_move0", t0))                         # the figure's time 0: the press's start
        ka, kf = rr[:, 0] >= tp - 2.0, f["t"][mf][::10] >= tp - 2.0
        row.update(turn=float(np.degrees(np.linalg.norm(rv, axis=1)).max()), dF_max=float(dF.max()),
                   q5=(float(q5.min()), float(q5.max())),
                   mount=float(min(mount)), ta=(rr[ka, 0] - tp).tolist(), ang=ang[ka].tolist(),
                   tf=(f["t"][mf][::10][kf] - tp).tolist(), dF=dF[::10][kf].tolist())
        out.append(row)
    return out


def lab_cell():
    """The cell model as set in the lab: the 2 by 4's block and the rod beside the sphere."""
    import ur5_cb3_cell as CELL
    C = CELL.C
    CELL.BOARD["block_mm"], CELL.BOARD["block_half_mm"] = 38.0, (55.0, 500.0)
    CELL.OTHER[:] = [(C.Tm(C.rot(C.EZ, -94.07), [-289.5, -585.5, 70.3]), "rod")]


def touch_rows(run="20261022-210852-sphere2"):
    """Each touch-off's onset from the expected (mm) in the order run, with its region, time and target on the pad
    (P frame, mm, from the run's plans)."""
    tgt = {}
    for pf in sorted((LAB / run).glob("plan*.json")):
        for a in json.loads(pf.read_text())["actions"]:
            tgt.setdefault(a["id"], a.get("target", {}).get("p"))
    out = []
    for e in events(run):
        if e.get("event") == "end" and e.get("status") == "ok" and str(e.get("id", "")).startswith("pad-"):
            on = e["rec"].get("onset")
            if on:
                out.append(dict(id=e["id"], region=e["id"].split("-")[1], t=e["t"], off=on["s0"] - on["expected_mm"],
                                p=on["p"], rms_mN=on["rms_N"] * 1e3, target=tgt.get(e["id"])))
    return out


def press_rows(run="20261022-210852-sphere2", stages=("grid", "pivot")):
    """Each press's push (N, the program's estimate along the approach) against the flange's travel past its rest (mm),
    from the streams, with the levels it held."""
    import ur5_tip_calibration as C
    ev = events(run)
    reg = json.loads((LAB / run / "registration.json").read_text())
    K = np.array(reg.get("K", np.eye(3).tolist()))
    f, r = ft(run), robot(run)
    out = []
    for e in ev:
        if e.get("event") != "end" or e.get("status") != "ok" or e.get("id", "").split("-")[0] not in ("grid", "ref", "pivot"):
            continue
        rec = e["rec"]
        if not rec.get("t_move0") or not rec.get("T_rest"):
            continue
        T_rest = np.array(rec["T_rest"])
        t0, t1 = rec["t_move0"], rec["t_stop"]
        m = (f["t"] >= t0) & (f["t"] <= t1)
        mr = (r[:, 0] >= t0 - 0.05) & (r[:, 0] <= t1 + 0.05)
        if m.sum() < 20 or mr.sum() < 5:
            continue
        d = -np.array(reg["T_BI"])[:3, 2]                        # the program's approach: down the indenter's z
        P = np.array([C.ur_to_T(v[8:14])[:3, 3] for v in r[mr]])
        s = np.interp(f["t"][m], r[mr, 0], (P - T_rest[:3, 3]) @ d)
        tare = np.array(rec["tare"])[:3]
        push = np.array([-(T_rest[:3, :3] @ np.linalg.solve(K, v - tare)) @ d for v in f["F"][m]])
        out.append(dict(id=e["id"], t=e["t"], s=s.tolist(), push=push.tolist(), tf=f["t"][m].tolist(),
                        levels=[lv["N"] for lv in rec.get("levels", [])]))
    return out


def smooth15(v):
    """The mean over 15 samples (0.15 s), the same length as the input, without the zeros of np.convolve's edges."""
    v = np.asarray(v, float)
    c = np.convolve(v, np.ones(15), mode="same")
    n = np.convolve(np.ones(len(v)), np.ones(15), mode="same")
    return c / n


def precontact(pr, run="20261022-210852-sphere2"):
    """Each grid press's push along the approach after 1.5 mm of travel, before the pad meets the sphere, and the
    board's change from its rest there and where the pad's own contact adds 0.4 N to that push."""
    ev = {e["id"]: e for e in events(run) if e.get("event") == "end" and e.get("status") == "ok"}
    plan = {a["id"]: a for a in json.loads((LAB / run / "plan_resume_2152.json").read_text())["actions"]}
    b = np.fromfile(LAB / run / "board.bin", BOARD_DT)
    tb, rb = b["t"], b["r"].reshape(-1, 5, 4).astype(float)[:, :, 1:]
    out = []
    for x in pr:
        if not x["id"].startswith("grid-") or x["id"] not in ev:
            continue
        rec = ev[x["id"]]["rec"]
        s, tf, p = np.array(x["s"]), np.array(x["tf"]), smooth15(x["push"])
        r0 = rb[(tb >= rec["t_rest0"][0]) & (tb <= rec["t_rest0"][1])].mean(0)

        def board(k):
            mb = (tb > tf[k] - 0.15) & (tb < tf[k] + 0.15)
            return float(np.linalg.norm(rb[mb].mean(0) - r0))
        k15 = int(np.argmin(abs(s - 1.5)))
        pre = float(p[k15])
        k2 = int(np.argmax((s > 1.6) & (p >= max(pre, 0.0) + 0.4)))
        out.append(dict(id=x["id"], pre=pre, b_pre=board(k15), s_pad=float(s[k2]), b_pad=board(k2),
                        joint6=plan.get(x["id"], {}).get("joint6_deg"), region=plan.get(x["id"], {}).get("target", {}).get("region")))
    return out


def ref_rows(pr):
    """The grid's reference presses in order, from press_rows: the flange's travel past its rest pose where the push,
    smoothed over 15 samples, passes 0.1 N (the onset) and 2 N, and the depth between (as lab/track_refs.py)."""
    out = []
    for x in pr:
        if not x["id"].startswith("ref-"):
            continue
        s, push = np.array(x["s"]), np.convolve(x["push"], np.ones(15) / 15, mode="same")
        on, top = float(s[np.argmax(push > 0.1)]), float(s[np.argmax(push > 1.95)])
        out.append(dict(id=x["id"], t=x["t"], onset=on, depth=top - on))
    return out


# ------------------------------------------------------------------------------------------------ figures
def fig_self_map(sm, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.plot(sm["q5"], sm["port"], "-o", ms=3, color="#b03030", label="Cable port")
    ax.plot(sm["q5"], sm["hex_e"], "-s", ms=3, color="#304080", label="HEX-E body")
    ax.plot(sm["q5"], sm["tool"], "-^", ms=3, color="#2a7a2a", label="Mount and finger")
    ax.axhline(15.0, color="0.4", lw=1, ls="--")
    ax.axvspan(30, 100, color="0.85", zorder=0, label="Joint 5 window (30 to 100 deg)")
    ax.axvline(173.5, color="#b03030", lw=1, ls=":")
    ax.text(170, 52, "the stop\nof 19:59", ha="right", va="top", fontsize=12, color="#b03030")
    ax.set_xlabel("Joint 5 (deg)")
    ax.set_ylabel("Least distance to the arm (mm)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 75)
    ax.legend(loc="lower center", fontsize=11, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_twist(tw, out):
    plt = P.plt_setup()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True)
    t, dF = np.array(tw["t"]), np.array(tw["dF"])
    for j, (lab, c) in enumerate(zip(("Fx", "Fy", "Fz"), ("#b03030", "#304080", "#2a7a2a"))):
        a1.plot(t, dF[:, j], lw=1, color=c, label=lab)
    a1.axvline(tw["t_turn"], color="0.5", lw=1, ls=":")
    a1.set_ylabel("Force from before\nthe turn (N)")
    a1.legend(loc="lower left", fontsize=11, ncol=3)
    a2.plot(tw["tq"], tw["q5"], color="k", lw=1.2, label="Joint 5")
    a2.set_ylabel("Joint 5 (deg)")
    a2b = a2.twinx()
    a2b.plot(tw["tm"], tw["mount"], color="#b06010", lw=1.2, label="Mount to the cell (model)")
    a2b.set_ylabel("Mount's clearance (mm)", color="#b06010")
    a2b.set_ylim(0, max(12, max(tw["mount"]) * 1.05))
    a2.set_xlabel("Time from the action's start (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_turns(rows, out):
    """The twists of sphere2g: the turn from the pressed pose and the change of the HEX-E's force against time."""
    plt = P.plt_setup()
    tw = [x for x in rows if x["kind"] == "twist" and "ang" in x]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True)
    for k, x in enumerate(tw):
        c = plt.cm.viridis(k / max(1, len(tw) - 1))
        a1.plot(x["ta"], x["ang"], lw=1.2, color=c, label=x["id"])
        a2.plot(x["tf"], x["dF"], lw=0.8, color=c)
    a1.set_ylabel("Turn from the\npressed pose (deg)")
    a1.legend(loc="upper left", fontsize=9, ncol=2)
    a2.set_ylabel("HEX-E's force from\nits tare (N)")
    a2.set_xlabel("Time from the start of the press (s)")
    a2.set_ylim(0, max(4.0, max(max(x["dF"]) for x in tw) * 1.1) if tw else 4.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_touches(rows, out):
    plt = P.plt_setup()
    fig, (ax, ay) = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True, gridspec_kw=dict(width_ratios=(1.25, 1)))
    col = dict(face="#304080", edge="#2a7a2a", flank="#b03030", tip="#806020")
    t0 = rows[0]["t"] if rows else 0.0
    for reg in ("face", "edge", "flank", "tip"):
        rr = [x for x in rows if x["region"] == reg]
        if rr:
            ax.plot([(x["t"] - t0) / 60 for x in rr], [x["off"] for x in rr], "o", ms=6, color=col[reg], label=reg.capitalize())
            ry = [x for x in rr if x.get("target")]
            ay.plot([x["target"][1] for x in ry], [x["off"] for x in ry], "o", ms=6, color=col[reg])
    for a in (ax, ay):
        a.axhline(0, color="0.4", lw=1)
        a.axhspan(-1, 1, color="0.9", zorder=0)
    ax.set_xlabel("Time from the first touch-off (min)")
    ax.set_ylabel("Onset from the expected (mm)")
    ax.legend(loc="upper left", fontsize=11, ncol=3)
    ay.set_xlabel("Target across the pad, $y_P$ (mm)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_presses(pr, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    t0 = min(x["t"] for x in pr)
    span = max(60.0, max(x["t"] for x in pr) - t0) / 60.0
    for x in pr:
        if not x["id"].startswith("ref-"):
            ax.plot(x["s"], smooth15(x["push"]), lw=0.7, color=plt.cm.viridis((x["t"] - t0) / 60.0 / span))
    for k, x in enumerate(r for r in pr if r["id"].startswith("ref-")):
        ax.plot(x["s"], smooth15(x["push"]), lw=1.4, color="k", label="Reference presses" if k == 0 else None)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, span))
    fig.colorbar(sm, ax=ax, label="Minutes from the first press")
    ax.set_xlabel("Flange's travel past the rest pose (mm)")
    ax.set_ylabel("Push on the pad (N)")
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def make_film(frames_dir, out_mp4, fps=12):
    """The overhead time-lapse (one frame every 2 s) as an H.264 film, and its middle frame as the poster."""
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        return False
    lst = out_mp4.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve()}'\nduration {1.0 / fps:.4f}\n" for p in frames))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf",
                    "scale=640:-2,format=yuv420p", "-c:v", "libx264", "-crf", "28", "-preset", "slow", str(out_mp4)],
                   check=True)
    lst.unlink()
    from PIL import Image
    Image.open(frames[len(frames) // 2]).save(out_mp4.with_suffix(".png"))
    return True


# ------------------------------------------------------------------------------------------------ text
def drift_text(dr):
    """The board's drift over sphere2's rests, and the two zeros of a press compared at its rest after."""
    if not dr:
        return ""
    return (f"<p>Over sphere2's {dr['minutes']:.0f} minutes, {dr['n']} rests out of contact gave the rest reading against time. "
            f"Their raw spread is {dr['raw_span']:.0f} µT on the worst component, from the tool's orientation; less the "
            f"Earth-field term of the fit above it is {dr['less_span']:.0f} µT. The chips' temperature moved "
            f"{dr['T_span']:.2f} °C. What is left drifts {dr['slope_norm']:.2f} µT per minute over all 15 components, a few "
            "tenths of a µT over one press. A press itself moves the rest by more. Over "
            f"{dr['n_pairs']} presses and touch-offs, the rest after a press differs from the rest before it by "
            f"{dr['local_med']:.1f} µT in the median and {dr['local_p90']:.1f} µT at the 90th percentile. A model of the rest that knows the time "
            f"since the zero misses the rest after by {dr['model_med']:.1f} µT in the median, about as much. Each press's "
            "zero is therefore taken from both of its rests, interpolated to the time of each hold, as the analysis does for "
            "the HEX-E.</p>")


def grid_text(R):
    """The grid's presses and aborts in sphere2g, and the reference presses (Table 3)."""
    pr, rf = R["press"], R.get("refs", [])
    grid = [x for x in pr if x["id"].startswith("grid-")]
    if not grid:
        return ""
    ab = [a for r in R["runs"] if r["name"] == "sphere2g" for a, why in r["aborts"] if a.startswith("grid-")]
    out = (f"<p>The grid pressed {len(grid)} points spread over the pad (Figure 4). It pressed the face to 5 N, the edges "
           "to 3 N and the flanks to 2 N. Each press held at 0.25, 0.5, 1 and 2 N on the way, and the face presses also "
           f"at 3.5 N. {len(ab)} press{'es' if len(ab) != 1 else ''} aborted"
           + (f" ({', '.join(ab)})" if ab else "") + ". A reference press, at one point of the pad's face to 2 N, came "
           "first, then about every 15 presses, and last. The same press each time shows whether the pad, or the chain "
           "behind it, moved over the night (Table 3).</p>")
    if len(rf) < 2:
        return out
    on, dp = np.array([x["onset"] for x in rf]), np.array([x["depth"] for x in rf])
    minutes = (rf[-1]["t"] - rf[0]["t"]) / 60.0
    out += ("<p>The workstation's first analysis found the chain from the board to the HEX-E softer than the model "
            "(QUEUE.md R16). Over the whole session, the flange travels 0.33 mm plus 0.20 mm per newton more than the FE "
            "pad indents, 0.73 mm at 2 N. Something between them gives: the DIP link's clamp, the printed mount or the "
            "arm's joints. A clamp that crept would show as a later onset or "
            "a growing depth. The session was set to stop if the onset moved 0.3 mm from the first reference press or the "
            f"depth grew 0.1 mm. Over {minutes:.0f} min, the onset moved at most {np.abs(on - on[0]).max():.3f} mm from "
            f"the first and the depth at most {np.abs(dp - dp[0]).max():.3f} mm.</p>")
    rows = [[P.code(x["id"]), hm(x["t"]), f"{x['onset']:.3f}", f"{x['depth']:.3f}", f"{x['onset'] - on[0]:+.3f}",
             f"{x['depth'] - dp[0]:+.3f}"] for x in rf]
    out += P.table(["Press", "Time", "Onset (mm)", "Depth to 2 N (mm)", "Onset change (mm)", "Depth change (mm)"],
                   rows, num_cols=(2, 3, 4, 5), caption="The reference presses of sphere2g, at one point of the pad's face "
                   "to 2 N, against the first.", n=3)
    pc = R.get("pre", [])
    bad = [x for x in pc if x["pre"] > 0.2]
    calm = [x for x in pc if abs(x["pre"]) < 0.1]
    if bad and calm:
        j6 = [x["joint6"] for x in bad if x.get("joint6") is not None]
        out += (f"<p>In {len(bad)} presses ({', '.join(sorted(x['id'] for x in bad))}) the HEX-E read "
                f"{min(x['pre'] for x in bad):.2f} to {max(x['pre'] for x in bad):.2f} N before the pad met the sphere. "
                "The load rose with the travel from the rest pose (the yellow lines that rise early in Figure 4). "
                f"The board did not feel it. After 1.5 mm of travel the board had moved {min(x['b_pre'] for x in bad):.1f} to "
                f"{max(x['b_pre'] for x in bad):.1f} µT from its rest, as in the presses without that load "
                f"({np.median([x['b_pre'] for x in calm]):.1f} µT in the median). The pad's own 0.4 N moves it "
                f"{np.median([x['b_pad'] for x in calm]):.1f} µT. The load therefore went through the HEX-E's cable or "
                "another part of the tool, and the HEX-E overstates the pad's force in those presses. All of them "
                f"pressed the pad's far flank" + (f" with joint 6 at {min(j6):.0f} to {max(j6):.0f} deg, the most "
                                                   "turned wrist of the grid" if j6 else "") + ".</p>")
    return out


def turns_text(R, F):
    """The rolls and twists of sphere2g (Table 4, Figure 5)."""
    tr = R.get("turns", [])
    if not tr:
        return ""
    done = [x for x in tr if "turn" in x]
    n_ok = sum(1 for x in tr if x["status"] == "ok")
    out = (f"<p>After the grid came {sum(x['kind'] == 'roll' for x in tr)} rolls at 1.5 N and "
           f"{sum(x['kind'] == 'twist' for x in tr)} twists at 1 and 2.5 N (Table 4). A roll tips the pressed pad 3 and "
           "6 deg each way about a level line through the sphere's top, so the contact rolls over the pad. Each roll "
           "does this about two such lines, a quarter turn apart. A twist turns the pad about the vertical through the "
           "top, one way to 5, 10 and 15 deg and back, which loads the pad in torsion. "
           f"{n_ok} of {len(tr)} ended ok.")
    if done:
        out += (f" The HEX-E's force moved at most {max(x['dF_max'] for x in done):.1f} N from its tare, the press "
                f"included. Joint 5 kept between {min(x['q5'][0] for x in done):.0f} and {max(x['q5'][1] for x in done):.0f} "
                "deg. In the model, with the session's registration, the mount stayed at least "
                f"{min(x['mount'] for x in done):.1f} mm from the cell and the homing key (Figure 5). In tcp2 the fork "
                "had reached the key.")
    out += "</p>"
    rows = [[P.code(x["id"]), f"{x['press_N']:.1f}" if x.get("press_N") else "", x["status"],
             f"{x['turn']:.1f}" if "turn" in x else "",
             f"{x['dF_max']:.2f}" if "dF_max" in x else "", f"{x['q5'][0]:.0f} to {x['q5'][1]:.0f}" if "q5" in x else "",
             f"{x['mount']:.1f}" if "mount" in x else "", f"{x['s']:.0f}"] for x in tr]
    out += P.table(["Action", "Press (N)", "Ended", "Largest turn (deg)", "Largest force change (N)", "Joint 5 (deg)",
                    "Mount's clearance (mm)", "Length (s)"], rows, num_cols=(1, 3, 4, 5, 6, 7),
                   caption="The rolls and twists of sphere2g. The turn is the flange's from the pressed pose; the force "
                   "change counts the press itself; the clearance is the model's, along the whole action.", n=4)
    if any(x["kind"] == "twist" and "ang" in x for x in tr):
        out += P.fig(5, F["turns"], "The twists of sphere2g. Top: the turn from the pressed pose. Bottom: the change of "
                     "the HEX-E's force from its tare, the press included.")
    return out


def sections(R, F):
    """(id, heading, html) of every section, and the lede and the summary, from the numbers R and the figure files F."""
    esc, table, fig, code = P.esc, P.table, P.fig, P.code
    S = []
    runs, ef, sm, tw, tr, pr, gp, rg = R["runs"], R["earth"], R["self"], R["twist"], R["touch"], R["press"], R["gaps"], R["reg"]
    offs = np.array([x["off"] for x in tr])
    gp = [g for g in gp if not g.get("between")]               # the gaps within the runs
    n_restart = sum(1 for g in gp if g["restart"])
    n_pause = sum(1 for g in gp if g["restart"] is False)
    S.append(("terms", "Terms and metrics", """<dl class="terms">
<dt>Onset from the expected</dt><dd>Where the pad first pushed on the sphere during a touch-off, along the approach, less where
the plan's geometry put that contact (mm; negative is earlier). It is fitted to the push against travel as
$F = A\\,(s - s_0)^p$ for $s > s_0$. Table 2 of the runbook expects it within 1 mm, and the session stops past 2 mm.</dd>
<dt>Onset and depth of a reference press</dt><dd>The onset is the flange's travel past the press's rest pose, along the
approach, where the push on the pad passes 0.1 N. The depth is its further travel to 2 N. Both are in mm, with the push
averaged over 15 samples (0.15 s) first.</dd>
<dt>Gravity residual</dt><dd>The root mean square (N) of the HEX-E's forces over the stations of a gravity sweep, less the
fit $f = A\\,g_E + o_f + v\\,(t - \\bar t)$. Here $g_E$ is the unit gravity in the flange frame, $A$ the sensor's map,
$o_f$ its offset and $v$ its drift. Table 2 of the runbook expects under 0.02 N.</dd>
<dt>Pause and restart of the HEX-E</dt><dd>A gap in the stream of samples over 50 ms. In a pause the sensor's sample
counter (about 1 kHz) advances by the gap's length, so the sensor sampled on and only the answers stopped. In a restart
it begins again near zero and the sensor returns untared.</dd>
<dt>Joint 5 window</dt><dd>The range of the robot's fifth joint (wrist 2) that every move and target must keep, 30 to
100 deg. Joint 5 at 0 or 180 deg is the wrist's singularity, and past about 90 deg the tool can fold back onto the
forearm.</dd>
<dt>Self-clearance</dt><dd>The least distance (mm) from the HEX-E, its cable port, the mount and the finger to the robot's
forearm and wrist links, from the URDF's meshes and the CAD. The plan needs 15 mm (10 mm for the HEX-E's body, which
stands 34.8 mm from wrist 2 in every pose).</dd>
<dt>Cell clearance</dt><dd>The least distance (mm) from each part of the tool to the indenters, the 2 by 4, its board and
the table in the cell model. The plan needs 6 mm for the HEX-E, 3 mm for the mount and 0.8 mm for the finger.</dd>
</dl>"""))
    rows = [[code(r["name"]), r["start"], esc(r["stages"].replace(",", ", ")), str(r["ok"]), str(r["aborted"]), esc(r["outcome"])]
            for r in runs]
    S.append(("runs", "Runs of the night", (
        f"<p>The program ran {len(runs)} times between {runs[0]['start']} and {runs[-1]['start']}, all at 30 % speed (Table 1). "
        "Each run started with the recorded hand seating on the sphere's key in place of homing, so nobody had to be at the "
        "robot. The folders of the runs, with their streams, printouts and overhead frames, are in "
        f"{code('docs/experiments/20261022-ur5_tip_calibration/lab/')}; {code('lab/NOTES.md')} gives each run's story. "
        f"The {P.link(D / '20261022-ur5_tip_calibration_protocol.html', 'protocol page')}, called the runbook below, "
        "describes the program, its plan and the thresholds of its checks. The workstation's "
        f"{P.link(D / '20261022-ur5_tip_ground_truth_and_predictors.html', 'page of ground truth and predictors')} "
        "analyses sphere2's data. It covers the pad's compliance against the FE model, the board's force against the "
        "HEX-E's, the pivots, the board's estimator, and predictors trained on the presses.</p>"
        + table(["Run", "Start", "Stages", "Actions ok", "Aborted", "How it ended"], rows, num_cols=(1, 3, 4),
                caption="The runs of 2026-10-22 on the lab's UR5.", n=1))))
    grow = [[hm(g["t"]), f"{g['s']:.2f}", "open" if g.get("open") else ("restart" if g["restart"] else "pause"),
             "" if g["dFz"] is None else f"{g['dFz']:+.1f}", P.code(g["run"])] for g in gp if g["s"] > 0.3 or g["restart"] or g.get("open")]
    S.append(("link", "Losses of the HEX-E's link", (
        f"<p>Within the runs, the HEX-E's stream stopped {len(gp)} times for over 50 ms (Table 2). {n_restart} were "
        f"restarts of the sensor, both in gravity sweeps while joint 6 turned. {sum(1 for g in gp if g.get('open'))} "
        "times the stream died for the rest of its run: soon after each restart, and at the stop of 19:59 (Section 4). "
        "Each time the box listed no sensor for 1 to 20 min. The link also failed once with the arm still: at tcp1's "
        "preflight the box listed no sensor at all. "
        "Each time the box listed it again, the sensor read its old untared zero, 651 N in Fz, with its usual noise. The failures "
        "therefore lay in its link, at the connector or in the cable. After the first restart its tare was lost, and two stations of the "
        "first gravity sweep were read 645 N higher. Every action with a gap over 0.5 s is therefore aborted now.</p>"
        f"<p>The other {n_pause} were pauses, with the sensor's counter running on. Most came from the computer. The reader "
        "of the HEX-E wrote ft.bin and called fsync every 0.5 s from its own thread. On the lab computer's SSD an fsync "
        "blocked for 0.1 to 0.4 s, and an action with no sample for 100 ms is aborted. Eight of sphere2's first 28 "
        "touch-offs were lost that way. After the fsync moved out (commit d1ef3216), one more pause came during a 160 MB "
        "copy. The copy's dirty pages throttled the reader's own write. The reader now hands its rows to a writer thread "
        "(96abeec0). Three pauses of 1.4 to 2.8 s came from the box itself, which did not answer the reads.</p>"
        + table(["Time", "Length (s)", "Kind", "Fz after less before (N)", "Run"], grow, num_cols=(0, 1, 3),
                caption="The HEX-E's gaps of over 0.3 s, and every restart.", n=2))))
    S.append(("wrist", "The wrist's stop at 19:59 and the checks of the arm against itself", (
        "<p>In first2 a travel of the mount stage turned joint 5 from 140 to 173.6 deg, 6 deg from the wrist's "
        "singularity. The owner saw the wrist pressing the HEX-E's cable port and stopped the robot. With the port in the "
        f"model, it stood {sm['at_stop']['port']:.1f} mm from the forearm at that pose (Figure 1).</p>"
        "<p>Three gaps let the travel through. The plan's builder solved its poses on its own IK branch, with joint 1 at "
        "−92 deg. The robot sat at 59 deg, and the program solves each move nearest the robot's joints. Its wrist limit "
        "was 7 deg from the singularity. And no check put the tool against the arm. The builder now takes the seating's "
        f"joints ({code('--q-seed')}), keeps joint 5 inside the window and needs the self-clearance. The program refuses an IK "
        "answer whose forward kinematics land more than 1 mm or 0.5 deg off the target. It also stops any move that "
        "takes joint 5 out of the window. On the robot's branch the mount stage's poses sit on the flipped wrist, so that "
        "stage was left out.</p>"
        + fig(1, F["self"], "The least distance to the robot's forearm and wrist links at each joint 5, over joints 4 and "
              "6. Red: the cable port. Blue: the HEX-E's body. Green: the mount and finger. The grey band is the program's "
              "window for joint 5. The dashed line is the 15 mm the plan needs, and the dotted line is the pose of the "
              "stop at 19:59."))))
    S.append(("twist", "The twist that reached the homing key", (
        f"<p>In tcp2 a twist turned the pad, pressed on the sphere at 1 N, about the vertical through the contact. On the way "
        f"to −20 deg the load rose to {tw['peak_N']:.1f} N and the action aborted. In the cell model the mount's fork came "
        f"within {min(tw['mount']):.1f} mm of the sphere's homing key at that pose (Figure 2). The builder had checked the "
        "twist's pressed pose alone, and the simulator models only the pad's contact. The builder now checks every 0.5 deg "
        "step of each slide, roll and twist, and each twist turns one way, to 15 deg and back.</p>"
        "<p>Before the load came, the turn itself added 0.3 to 0.6 N along the finger at 5 to 20 deg. Its axis passed off "
        "the pad's contact across the finger, by 0.25 to 1 mm, which the flank touch-offs of the registration measure.</p>"
        + fig(2, F["twist"], "tcp2's first twist. Top: the HEX-E's force in its own frame from its value just before the "
              "turn. Bottom: joint 5 (black) and the mount's least clearance to the cell in the model (orange)."))))
    ty = np.array([x["target"][1] for x in tr if x.get("target")])
    to = np.array([x["off"] for x in tr if x.get("target")])
    cy = np.polyfit(ty, to, 1)
    S.append(("segment", "Touch-offs, registration and presses of sphere2", (
        f"<p>sphere2 touched the pad on the sphere {len(tr)} times. The onsets came {np.median(offs):+.2f} mm from the "
        f"expected in the median, {offs.min():+.2f} to {offs.max():+.2f} mm in all (Figure 3). They follow where each "
        "touch-off was on the pad, not when it came. Across the pad they rise "
        f"{-cy[0]:.2f} mm for each mm toward −y<sub>P</sub>, and that line leaves "
        f"{float((to - np.polyval(cy, ty)).std()):.2f} mm. Each run worked across the pad from +y<sub>P</sub> to "
        "−y<sub>P</sub>, so the onsets rise in time as well. The registration takes the pattern into the pad's pose on "
        "the flange, and the part common to every touch into the sphere's centre.</p>"
        + rg + fig(3, F["touch"], "The onset of each touch-off from the expected, by region of the pad: against the time "
                   "from the first touch-off (left) and against the target's place across the pad (right). The grey band "
                   "is the runbook's 1 mm.")
        + (fig(4, F["press"], "The push on the pad against the flange's travel past its rest pose, for each pivot and grid "
               "press of the session, coloured by time; the reference presses in black.") if pr else "")
        + grid_text(R))))
    if R.get("turns"):
        S.append(("turns", "Rolls and twists of sphere2g", turns_text(R, F)))
    S.append(("board", "The board's rest reading against the tool's orientation", (
        f"<p>At first2's {ef['n']} gravity stations the board's rest reading moved by up to {max(ef['spread']):.0f} µT on a "
        f"chip as the tool turned. A uniform field fitted to all five chips leaves {ef['rms_all']:.1f} µT rms "
        f"({min(ef['rms']):.1f} to {max(ef['rms']):.1f} µT by chip). The field is {ef['w_norm']:.1f} µT, {ef['dip']:.0f} deg "
        "below the horizontal: Earth's field at the bench. The fitted axes of the chips stand "
        + ", ".join(f"{a:.1f}" for a in ef["chip_deg"]) + " deg from chip 0's, a quarter turn apart as on the board. The "
        "chips' temperature correction of 0.65 to 0.70 % per °C was applied first.</p>"
        "<p>A press tared at its own rest pose takes Earth's field into its zero. A roll or a twist turns the pad while it is "
        "pressed, and its board reading then needs the orientation term of this fit subtracted.</p>"
        + drift_text(R.get("drift")))))
    S.append(("future", "Future work", """<ul>
<li>The HEX-E's link (QUEUE.md R10). Inspect the connector and the port, and clamp the cable 3 to 5 cm from the
connector. Then flex it with the arm still while <code>lab/sensor_watch.py</code> logs. It passes with no gap over
100 ms and no restart.</li>
<li>The HEX-E's force gain (R5): a weighed mass at rest, in place of a gravity sweep, gives the gain on each axis to
about 1 %. first2's sweep gave gains of 1.135, 1.085 and 0.78 of their mean, which a cable that twists with joint 6
would also give.</li>
<li>The pad's surface offset: the presses meet the sphere 0.3 to 1.2 mm earlier than the plan's geometry. A
registration term for a uniform offset of the pad would separate it from the sphere's height.</li>
<li>The give between the board and the HEX-E (R16). The flange travels 0.33 mm plus 0.20 mm per newton more than the
FE pad indents. Per newton it is 0.21 mm on the face, 0.17 mm on the edges and 0.14 mm on the flanks. Check by hand
whether the DIP link moves in the fork, with a dial indicator on the link if one is at hand. The arm's joints under load
are in the chain too.</li>
<li>The load off the pad in four flank presses (Section 6). Watch the HEX-E's cable and the tool's parts near the
sphere while the arm repeats grid-066 slowly. Then decide whether the plan's flank presses need another wrist turn or
the cable another route.</li>
<li>The actions left out: grid-007 and the touch-offs pad-flank-04, -08 and -09, which aborted on pauses of the
HEX-E's stream, and the stages not run tonight (gravity, mount, final). A resume of the session runs them with the
program's queued writer.</li>
<li>The board's drift. The session's rests give the rest reading against time, temperature and orientation. Subtract
the orientation term of the Earth-field fit above, then fit the drift in time.</li>
</ul>"""))
    grid = [x for x in pr if x["id"].startswith("grid-")]
    rf = R.get("refs", [])
    ron = [x["onset"] - rf[0]["onset"] for x in rf] if rf else [0.0]
    rdp = [x["depth"] - rf[0]["depth"] for x in rf] if rf else [0.0]
    turns_ok = sum(1 for x in R.get("turns", []) if x["status"] == "ok")
    n_bad = sum(1 for x in R.get("pre", []) if x["pre"] > 0.2)
    lede = (f"The first segment of the fingertip's calibration data was taken on the lab's UR5 on the evening of 22 October "
            f"2026. In sphere2, from 21:08 to 23:27, the pad touched the sphere {len(tr)} times for the registration. It "
            f"then took {len(grid)} presses over the pad, {len(rf)} reference presses and {turns_ok} rolls and twists. "
            f"Over {(rf[-1]['t'] - rf[0]['t']) / 60:.0f} min, the reference presses kept their onset within "
            f"{max(map(abs, ron)):.2f} mm and their depth within {max(map(abs, rdp)):.2f} mm, so the pad and its clamp "
            f"held still. In {n_bad} flank presses the HEX-E carried a load that missed the pad, and the analysis leaves "
            "them out. Earlier runs met three faults. The HEX-E's link failed, a travel pressed the HEX-E's cable port "
            "against the forearm, and a twist swung the mount onto the homing key. The program and the plan's builder now "
            "check for each.")
    summary = (f"First segment of the fingertip calibration on the lab UR5 (sphere2, 21:08 to 23:27): {len(tr)} touch-offs "
               f"registered, {len(grid)} presses over the pad, {turns_ok} rolls and twists, reference presses steady "
               f"within {max(map(abs, ron)):.2f} mm; {n_bad} flank presses carried a load off the pad. The HEX-E link's "
               "failures, the wrist's stop at the forearm and the twist on the homing key led to the program's checks.")
    return S, lede, summary


def registration_text(run="20261022-210852-sphere2"):
    """The registration's printout in prose, from the session's notes."""
    reg = json.loads((LAB / run / "registration.json").read_text())
    n = reg.get("notes", {}).get("register", {})
    if not n:
        return "<p>The session's registration had not run.</p>"
    tcp = n.get("tcp_change_mm_deg", [float("nan")] * 2)
    return (f"<p>The registration fitted the pad's frame on the flange and the sphere's centre to {n.get('n_pad', 0)} touch-offs, "
            f"with {n.get('rms_pad_mm', float('nan')) * 1e3:.0f} µm rms left (the runbook asks under 50 µm). It moved the TCP "
            f"{tcp[0]:.2f} mm and {tcp[1]:.2f} deg from the nominal and the sphere's centre {n.get('centre_change_mm', float('nan')):.2f} mm "
            "from homing's. Most of the TCP's change is a tilt of the pad in its seat, about 2 deg about two axes. The "
            "mount also sits turned about 1.3 deg past its measured −92.15 deg. With this registration, two of the plan's "
            "least clearances fell under their margins: 0.57 mm for the finger (margin 0.8 mm) and 2.27 mm for the mount "
            "(margin 3 mm). The four actions under the margins, grid-050 and three slides, were left out of the rest of "
            "the session.</p>")


# ------------------------------------------------------------------------------------------------ page
def build(film_dir=None):
    MEDIA.mkdir(parents=True, exist_ok=True)
    cache = LAB / "20261022-lab_self_map.json"                 # 40 s to compute; the tool's geometry is fixed
    if cache.exists():
        sm = json.loads(cache.read_text())
    else:
        sm = self_map()
        cache.write_text(json.dumps(sm, default=float))
    R = dict(runs=run_rows(), earth=earth_field(), self=sm, twist=twist_trace(), touch=touch_rows(),
             press=press_rows(), reg=registration_text(), drift=board_drift())
    R["refs"] = ref_rows(R["press"])
    R["pre"] = precontact(R["press"])
    R["turns"] = turn_rows()
    outcome = {"first": "HEX-E restarted in gravity; 3 aborts", "first2": "e-stop by the owner in the mount stage",
               "sphere1": "HEX-E restarted in gravity; 3 aborts", "tcp1": "no HEX-E at preflight",
               "tcp2": "twist aborted at 10 N (fork on the key); interrupted",
               "sphere2": "three aborts in a row (the reader's fsync); resumed",
               "sphere2r": "crash in the pivot check after the registration; resumed",
               "sphere2g": SPHERE2G_END}
    for r in R["runs"]:
        r["outcome"] = outcome.get(r["name"], "")
    R["gaps"] = [dict(g, run=part_name(run, g["t"])) for run in RUNS for g in gaps(run)]
    F = dict(self=MEDIA / "20261022-lab_tool_against_arm.png", twist=MEDIA / "20261022-lab_twist_on_the_key.png",
             touch=MEDIA / "20261022-lab_touch_onsets.png", press=MEDIA / "20261022-lab_presses.png",
             turns=MEDIA / "20261022-lab_twists.png")
    if any(x["kind"] == "twist" and "ang" in x for x in R["turns"]):
        fig_turns(R["turns"], F["turns"])
    fig_self_map(R["self"], F["self"])
    fig_twist(R["twist"], F["twist"])
    fig_touches(R["touch"], F["touch"])
    if R["press"]:
        fig_presses(R["press"], F["press"])
    film_html = ""
    mp4 = MEDIA / "20261022-lab_sphere2_overhead.mp4"
    if film_dir and make_film(film_dir, mp4):
        film_html = P.film(0, mp4, mp4.with_suffix(".png"), "The overhead camera over sphere2's first touch-offs, 21:13 to "
                           "21:19: one frame every 2 s, played at 12 frames a second. The pad touches the sphere on the right-hand "
                           "fixture; the left-hand fixture holds the rod.").replace("Figure 0. ", "")
    S, lede, summary = sections(R, F)
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "UR5 lab calibration runs",
                 "H1": "Fingertip calibration runs on the lab UR5 with the OptoForce HEX-E, 22 October 2026",
                 "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-22", "LEDE": lede,
                 "BYLINE": "Runs and analysis by the lab computer's Claude Code session and the workstation's, for the owner.",
                 "FILM": film_html, "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    # the lab computer has no TeX: the workstation renders this list into TEX_CACHE (texsvg.render)
    (LAB / "20261022-lab_page_formulas.json").write_text(json.dumps(items, indent=0))
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = P.TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary)
    PAGE.write_text(page)
    (LAB / "20261022-lab_page_numbers.json").write_text(json.dumps(
        {k: v for k, v in R.items() if k in ("runs", "earth", "gaps", "touch")}, indent=1, default=float))
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


def board_drift(run="20261022-210852-sphere2", field_run="20261022-195420-first2"):
    """The board's rest reading at every action's rests in run, corrected for the chips' temperature, less the
    orientation term of the Earth-field fit to field_run's gravity stations (A_k R_BE^T w with the fit's chip rotations
    A_k and field w, refitted here), against time. Two zeros of a press are compared at the rest after it: the rest
    before it (taken at the same pose) and a model of the rests (offset, orientation term and a drift linear in time)."""
    import fingertip_wrench_model as W
    import ur5_tip_calibration as C
    kap = W.load_thermal()
    # the field fit, with the chips' rotations, from field_run
    recs = [e["rec"] for e in events(field_run) if e.get("event") == "end" and e.get("status") == "ok"
            and str(e.get("id", "")).startswith("grav")]
    Rg = np.array([r["R_BE"] for r in recs])
    Bg = np.array([r["board"] for r in recs]).reshape(-1, 5, 4)
    Bgc = W.thermal_correct(Bg[:, :, 1:], Bg[:, :, 0], kap)

    def procrustes(M):
        U, _, Vt = np.linalg.svd(M)
        return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt

    w = np.array([0.0, 0.0, -50.0])
    for _ in range(300):
        U_ = np.einsum("nji,j->ni", Rg, w)
        A = []
        for k in range(5):
            sol, *_ = np.linalg.lstsq(np.c_[U_, np.ones(len(Rg))], Bgc[:, k], rcond=None)
            A.append(procrustes(sol[:3].T))
        b0 = [(Bgc[:, k] - U_ @ A[k].T).mean(0) for k in range(5)]
        M = np.vstack([A[k] @ Rg[i].T for k in range(5) for i in range(len(Rg))])
        rhs = np.concatenate([Bgc[i, k] - b0[k] for k in range(5) for i in range(len(Rg))])
        w_new, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        done = np.linalg.norm(w_new - w) < 1e-7
        w = w_new
        if done:
            break
    # the rests of run
    b = np.fromfile(LAB / run / "board.bin", BOARD_DT)
    tb, rb = b["t"], b["r"].reshape(-1, 5, 4).astype(float)
    r = robot(run)
    rows = []
    for e in events(run):
        if e.get("event") != "end" or e.get("status") != "ok" or not isinstance(e.get("rec"), dict):
            continue
        rec = e["rec"]
        for key in ("t_rest0", "t_rest1"):
            if not rec.get(key):
                continue
            t0, t1 = rec[key]
            m = (tb >= t0) & (tb <= t1)
            if m.sum() < 20:
                continue
            i = min(np.searchsorted(r[:, 0], 0.5 * (t0 + t1)), len(r) - 1)
            R_BE = C.ur_to_T(r[i, 8:14])[:3, :3]
            Bc = W.thermal_correct(rb[m, :, 1:], rb[m, :, 0], kap).mean(0)          # (5, 3)
            orient = np.array([A[k] @ R_BE.T @ w for k in range(5)])
            rows.append(dict(id=e["id"], which=key, t=0.5 * (t0 + t1), T=rb[m, :, 0].mean(0).tolist(),
                             raw=Bc.ravel().tolist(), less_orient=(Bc - orient).ravel().tolist()))
    if not rows:
        return None
    t = np.array([x["t"] for x in rows])
    X = np.array([x["less_orient"] for x in rows])
    Xr = np.array([x["raw"] for x in rows])
    tm = (t - t[0]) / 60.0
    # drift model: offset + slope per component, fitted to the rests
    G = np.c_[np.ones_like(tm), tm]
    coef, *_ = np.linalg.lstsq(G, X, rcond=None)
    res = X - G @ coef
    # zeros of each press, judged at its rest after: the rest before (same pose) against the model at that time
    pairs = []
    by = {}
    for x in rows:
        by.setdefault(x["id"], {})[x["which"]] = x
    for k, v in by.items():
        if "t_rest0" in v and "t_rest1" in v:
            a, bb = v["t_rest0"], v["t_rest1"]
            after = np.array(bb["less_orient"])
            local = np.array(a["less_orient"])
            model = coef[0] + coef[1] * (bb["t"] - t[0]) / 60.0
            pairs.append(dict(id=k, local=float(np.linalg.norm(after - local)), model=float(np.linalg.norm(after - model)),
                              dt_s=float(bb["t"] - a["t"])))
    span_orient = float(np.ptp(Xr, axis=0).max())
    return dict(n=len(rows), minutes=float(tm[-1]), slope_uT_min=coef[1].reshape(5, 3).tolist(),
                slope_norm=float(np.linalg.norm(coef[1])), res_rms=float(np.sqrt((res ** 2).mean())),
                raw_span=span_orient, less_span=float(np.ptp(X, axis=0).max()), w=w.tolist(),
                T_span=float(np.ptp([x["T"] for x in rows], axis=0).max()),
                local_med=float(np.median([p["local"] for p in pairs])) if pairs else None,
                model_med=float(np.median([p["model"] for p in pairs])) if pairs else None,
                local_p90=float(np.percentile([p["local"] for p in pairs], 90)) if pairs else None,
                model_p90=float(np.percentile([p["model"] for p in pairs], 90)) if pairs else None,
                n_pairs=len(pairs), t=tm.tolist(), X=X.tolist(), Xr=Xr.tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", default=None, help="folder of the overhead time-lapse's JPEG frames")
    build(ap.parse_args().film)
