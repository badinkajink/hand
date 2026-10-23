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
    """One row per run: start, stages asked, actions ended ok and aborted, how it ended."""
    out = []
    for run in RUNS:
        ev = events(run)
        ses = json.loads((LAB / run / "session.json").read_text())
        ok = [e["id"] for e in ev if e.get("event") == "end" and e.get("status") == "ok" and e.get("id") not in ("preflight", "homing")]
        ab = [(e["id"], e.get("why", "")) for e in ev if e.get("event") == "end" and e.get("status") == "abort"]
        out.append(dict(run=run, name=run.split("-")[-1], start=ses.get("started", "")[11:16], stages=ses.get("stages", ""),
                        ok=len(ok), aborted=len(ab), aborts=ab))
    return out


def gaps(run, min_s=0.05):
    """Gaps of the HEX-E stream over min_s: start, length, whether the sample counter restarted, change of Fz."""
    f = ft(run)
    if len(f) < 2:
        return []
    dt = np.diff(f["t"])
    out = []
    for i in np.nonzero(dt > min_s)[0]:
        if i == 0:
            continue
        s0, s1 = int(f["seq"][i]), int(f["seq"][i + 1])
        adv = (s1 - s0) % 65536
        restart = abs(adv - dt[i] * 1000.0) > max(50.0, 0.2 * dt[i] * 1000.0)
        out.append(dict(t=float(f["t"][i]), s=float(dt[i]), restart=bool(restart), dFz=float(f["F"][i + 1, 2] - f["F"][i, 2])))
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
    CELL.BOARD["block_mm"], CELL.BOARD["block_half_mm"] = 38.0, (55.0, 500.0)
    CELL.OTHER[:] = [(C.Tm(C.rot(C.EZ, -94.07), [-289.5, -585.5, 70.3]), "rod")]
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


def touch_rows(run="20261022-210852-sphere2"):
    """Each touch-off's onset from the expected (mm) in the order run, with its region and time."""
    out = []
    for e in events(run):
        if e.get("event") == "end" and e.get("status") == "ok" and str(e.get("id", "")).startswith("pad-"):
            on = e["rec"].get("onset")
            if on:
                out.append(dict(id=e["id"], region=e["id"].split("-")[1], t=e["t"], off=on["s0"] - on["expected_mm"],
                                p=on["p"], rms_mN=on["rms_N"] * 1e3))
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
        d = np.array(rec.get("d_B") or (-T_rest[:3, :3] @ np.array([0, 0, 1.0])))
        P = np.array([C.ur_to_T(v[8:14])[:3, 3] for v in r[mr]])
        s = np.interp(f["t"][m], r[mr, 0], (P - T_rest[:3, 3]) @ d)
        tare = np.array(rec["tare"])[:3]
        push = np.array([-(T_rest[:3, :3] @ np.linalg.solve(K, v - tare)) @ d for v in f["F"][m]])
        out.append(dict(id=e["id"], s=s.tolist(), push=push.tolist(), levels=[lv["N"] for lv in rec.get("levels", [])]))
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


def fig_touches(rows, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    col = dict(face="#304080", edge="#2a7a2a", flank="#b03030", tip="#806020")
    t0 = rows[0]["t"] if rows else 0.0
    for reg in ("face", "edge", "flank", "tip"):
        rr = [x for x in rows if x["region"] == reg]
        if rr:
            ax.plot([(x["t"] - t0) / 60 for x in rr], [x["off"] for x in rr], "o", ms=6, color=col[reg], label=reg.capitalize())
    ax.axhline(0, color="0.4", lw=1)
    ax.axhspan(-1, 1, color="0.9", zorder=0)
    ax.set_xlabel("Time from the first touch-off (min)")
    ax.set_ylabel("Onset from the expected (mm)")
    ax.legend(loc="lower right", fontsize=11, ncol=4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_presses(pr, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for k, x in enumerate(pr):
        ax.plot(x["s"], x["push"], lw=1, color=plt.cm.viridis(k / max(1, len(pr) - 1)), label=x["id"] if k < 12 else None)
    ax.set_xlabel("Flange's travel past the rest pose (mm)")
    ax.set_ylabel("Push on the pad (N)")
    ax.legend(loc="upper left", fontsize=9, ncol=3)
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
def sections(R, F):
    """(id, heading, html) of every section, and the lede and the summary, from the numbers R and the figure files F."""
    esc, table, fig, code = P.esc, P.table, P.fig, P.code
    S = []
    runs, ef, sm, tw, tr, pr, gp, rg = R["runs"], R["earth"], R["self"], R["twist"], R["touch"], R["press"], R["gaps"], R["reg"]
    offs = np.array([x["off"] for x in tr])
    n_restart = sum(1 for g in gp if g["restart"])
    n_pause = sum(1 for g in gp if g["restart"] is False)
    S.append(("terms", "Terms and metrics", """<dl class="terms">
<dt>Onset from the expected</dt><dd>Where the pad first pushed on the sphere during a touch-off, along the approach, less where
the plan's geometry put that contact (mm; negative is earlier). It is fitted to the push against travel as
$F = A\\,(s - s_0)^p$ for $s > s_0$. Table 2 of the runbook expects it within 1 mm, and the session stops past 2 mm.</dd>
<dt>Gravity residual</dt><dd>The root mean square of the HEX-E's forces, over the stations of a gravity sweep, less the fit
$f = A\\,g_E + o_f + v\\,(t - \\bar t)$, with $g_E$ the unit gravity in the flange frame, $A$ the sensor's map,
$o_f$ its offset and $v$ its drift (N). Table 2 expects under 0.02 N.</dd>
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
    rows = [[r["name"], r["start"], esc(r["stages"].replace(",", ", ")), str(r["ok"]), str(r["aborted"]), esc(r["outcome"])]
            for r in runs]
    S.append(("runs", "Runs of the night", (
        f"<p>The program ran {len(runs)} times between {runs[0]['start']} and {runs[-1]['start']}, all at 30 % speed (Table 1). "
        "Each run started with the recorded hand seating on the sphere's key in place of homing, so nobody had to be at the "
        "robot. The folders of the runs, with their streams, printouts and overhead frames, are in "
        f"{code('docs/experiments/20261022-ur5_tip_calibration/lab/')}; {code('lab/NOTES.md')} gives each run's story.</p>"
        + table(["Run", "Start", "Stages", "Actions ok", "Aborted", "How it ended"], rows, num_cols=(1, 3, 4),
                caption="The runs of 2026-10-22 on the lab's UR5.", n=1))))
    grow = [[hm(g["t"]), f"{g['s']:.2f}", "open" if g.get("open") else ("restart" if g["restart"] else "pause"),
             "" if g["dFz"] is None else f"{g['dFz']:+.1f}", esc(g["run"])] for g in gp if g["s"] > 0.3 or g["restart"] or g.get("open")]
    S.append(("link", "Losses of the HEX-E's link", (
        f"<p>The HEX-E's stream stopped {len(gp)} times for over 50 ms. {n_restart} were restarts of the sensor. Each "
        "restart came while joint 6 was turning, or with the arm still, and the box then listed no sensor for 1 to 20 min. "
        f"{n_pause} were pauses with the counter running on (Table 2). The sensor read its old untared zero, 651 N in Fz, "
        "and its usual noise each time it came back. So its link failed, at the connector or in the cable, and the "
        "sensing element did not. After the first restart its tare was lost: two stations of the first gravity sweep were "
        "read 645 N higher, which is why every action with a gap over 0.5 s is now aborted.</p>"
        "<p>The pauses had another cause. The reader of the HEX-E wrote ft.bin and called fsync every 0.5 s from its own "
        "thread. On the lab computer's SSD an fsync blocked for 0.1 to 0.4 s, and a touch-off with no sample for 100 ms is "
        "aborted. Eight of sphere2's first 28 touch-offs were lost that way. With the fsync moved to a thread of its own "
        "(commit d1ef3216), the resumed run had no pause.</p>"
        + table(["Time", "Length (s)", "Kind", "Fz after less before (N)", "Run"], grow, num_cols=(0, 1, 3),
                caption="The HEX-E's gaps of over 0.3 s, and every restart.", n=2))))
    S.append(("wrist", "The wrist's stop at 19:59 and the checks of the arm against itself", (
        "<p>In first2 a travel of the mount stage turned joint 5 from 140 to 173.6 deg, 6 deg from the wrist's "
        "singularity. The owner saw the wrist pressing the HEX-E's cable port and stopped the robot. With the port in the "
        f"model, it stood {sm['at_stop']['port']:.1f} mm from the forearm at that pose (Figure 1).</p>"
        "<p>Three gaps let the travel through. The plan's builder solved its poses on its own IK branch, with joint 1 at "
        "−92 deg, while the robot sat at 59 deg, and the program solves its moves nearest the robot's joints. Its wrist limit "
        "was 7 deg from the singularity. And no check put the tool against the arm. The builder now takes the seating's "
        f"joints ({code('--q-seed')}), keeps joint 5 inside the window and needs the self-clearance. The program refuses an IK "
        "answer that its forward kinematics place more than 1 mm or 0.5 deg off the target, and it stops any move that "
        "takes joint 5 out of the window. On the robot's branch the mount stage's poses sit on the flipped wrist, so that "
        "stage was left out.</p>"
        + fig(1, F["self"], "The least distance from the cable port (red), the HEX-E's body (blue) and the mount and finger "
              "(green) to the robot's forearm and wrist links, over joints 4 and 6 at each joint 5. The grey band is the "
              "window the program keeps; the dashed line is the 15 mm the plan needs; the dotted line is the pose of the stop "
              "at 19:59."))))
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
    S.append(("segment", "Touch-offs, registration and presses of sphere2", (
        f"<p>sphere2 touched the pad on the sphere {len(tr)} times. The onsets came {np.median(offs):+.2f} mm from the "
        f"expected in the median, {offs.min():+.2f} to {offs.max():+.2f} mm in all (Figure 3). An early contact that is "
        "nearly the same at every touch reads as a higher sphere, and the registration takes it into the sphere's "
        "centre.</p>" + rg + fig(3, F["touch"], "The onset of each touch-off from the expected, in the order run, by region "
                                 "of the pad. The grey band is Table 2's 1 mm.")
        + (fig(4, F["press"], "The push on the pad against the flange's travel past its rest pose, for each grid and pivot "
               "press of the session.") if pr else ""))))
    S.append(("board", "The board's rest reading against the tool's orientation", (
        f"<p>At first2's {ef['n']} gravity stations the board's rest reading moved by up to {max(ef['spread']):.0f} µT on a "
        f"chip as the tool turned. A uniform field fitted to all five chips leaves {ef['rms_all']:.1f} µT rms "
        f"({min(ef['rms']):.1f} to {max(ef['rms']):.1f} µT by chip). The field is {ef['w_norm']:.1f} µT, {ef['dip']:.0f} deg "
        "below the horizontal: Earth's field at the bench. The fitted axes of the chips stand "
        + ", ".join(f"{a:.1f}" for a in ef["chip_deg"]) + " deg from chip 0's, a quarter turn apart as on the board. The "
        "chips' temperature correction of 0.65 to 0.70 % per °C was applied first.</p>"
        "<p>A press tared at its own rest pose takes Earth's field into its zero. A roll or a twist turns the pad while it is "
        "pressed, and its board reading then needs the orientation term of this fit subtracted.</p>")))
    S.append(("future", "Future work", """<ul>
<li>The HEX-E's link (QUEUE.md R10): inspect the connector and the port, clamp the cable 3 to 5 cm from the connector, and
flex it with the arm still while <code>lab/sensor_watch.py</code> logs. It passes with no gap over 100 ms and no
restart.</li>
<li>The HEX-E's force gain (R5): a weighed mass at rest, in place of a gravity sweep, gives the gain on each axis to
about 1 %. first2's sweep gave gains of 1.135, 1.085 and 0.78 of their mean, which a cable that twists with joint 6
would also give.</li>
<li>The pad's surface offset: the onsets and the presses meet the sphere 0.3 to 1.2 mm early. A registration term for a
uniform offset of the pad would separate it from the sphere's height.</li>
<li>The board's drift: the rests of the session give the rest reading against time, temperature and orientation; the
Earth-field fit above gives the orientation term to subtract before the drift in time is fitted.</li>
</ul>"""))
    lede = (f"The calibration program ran {len(runs)} times on the lab's UR5 on the evening of 22 October 2026. The HEX-E's "
            f"link failed {n_restart} times and paused {n_pause} times; its sensing read its old zero and noise each time "
            f"it came back. In sphere2 the pad touched the sphere {len(tr)} times, {np.median(offs):+.2f} mm from the "
            "expected in the median. A travel that took the wrist toward its singularity pressed the HEX-E's cable port "
            "against the forearm, and a twist swung the mount onto the homing key; the plan's builder and the program now "
            "check both.")
    summary = (f"Six runs of the fingertip calibration on the lab UR5: the HEX-E's link failed {n_restart} times; "
               f"{len(tr)} touch-offs {np.median(offs):+.2f} mm from the expected in the median; the wrist's stop and the "
               "twist on the homing key led to joint 5, self-collision and turn-path checks.")
    return S, lede, summary


def registration_text(run="20261022-210852-sphere2"):
    """The registration's printout in prose, from the session's notes."""
    reg = json.loads((LAB / run / "registration.json").read_text())
    n = reg.get("notes", {}).get("register", {})
    if not n:
        return "<p>The session's registration had not run.</p>"
    tcp = n.get("tcp_change_mm_deg", [float("nan")] * 2)
    return (f"<p>The registration fitted the pad's frame on the flange and the sphere's centre to {n.get('n_pad', 0)} touch-offs, "
            f"with {n.get('rms_pad_mm', float('nan')) * 1e3:.0f} µm rms left (Table 2: under 50 µm). It moved the TCP "
            f"{tcp[0]:.2f} mm and {tcp[1]:.2f} deg from the nominal and the sphere's centre {n.get('centre_change_mm', float('nan')):.2f} mm "
            "from homing's. Most of the TCP's change is a tilt of the pad in its seat, about 2 deg about two axes, with the "
            "mount turned about 1.3 deg past its measured −92.15 deg. With this registration the plan's least clearances fell "
            "to 0.57 mm for the finger and 2.27 mm for the mount, under their margins of 0.8 and 3 mm, so the four actions "
            "under them (grid-050 and three slides) were left out of the rest of the session.</p>")


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
    outcome = {"first": "HEX-E restarted in gravity; 3 aborts", "first2": "e-stop by the owner in the mount stage",
               "sphere1": "HEX-E restarted in gravity; 3 aborts", "tcp1": "no HEX-E at preflight",
               "tcp2": "twist aborted at 10 N (fork on the key); interrupted", "sphere2": "resumed twice; see Section 6"}
    for r in R["runs"]:
        r["outcome"] = outcome.get(r["name"], "")
    R["gaps"] = [dict(g, run=run.split("-")[-1]) for run in RUNS for g in gaps(run)]
    F = dict(self=MEDIA / "20261022-lab_tool_against_arm.png", twist=MEDIA / "20261022-lab_twist_on_the_key.png",
             touch=MEDIA / "20261022-lab_touch_onsets.png", press=MEDIA / "20261022-lab_presses.png")
    fig_self_map(R["self"], F["self"])
    fig_twist(R["twist"], F["twist"])
    fig_touches(R["touch"], F["touch"])
    if R["press"]:
        fig_presses(R["press"], F["press"])
    film_html = ""
    mp4 = MEDIA / "20261022-lab_sphere2_overhead.mp4"
    if film_dir and make_film(film_dir, mp4):
        film_html = P.film(0, mp4, mp4.with_suffix(".png"), "The overhead camera over sphere2's touch-offs, one frame every "
                           "2 s played at 12 frames a second (21:13 to 21:19). The pad touches the sphere on the right-hand "
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
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = P.TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary)
    PAGE.write_text(page)
    (LAB / "20261022-lab_page_numbers.json").write_text(json.dumps(
        {k: v for k, v in R.items() if k in ("runs", "earth", "gaps", "touch")}, indent=1, default=float))
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", default=None, help="folder of the overhead time-lapse's JPEG frames")
    build(ap.parse_args().film)


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
