#!/usr/bin/env python3
"""Ground truth of a calibration session of the fingertip on the UR5 (src/morphohand/driver/reskin5x/host/
ur5_tip_calibration.py), and the contact estimator scored against it.

For every hold of the session (a press's force level, a turn of a slide, roll or twist) the ground truth comes from the
robot and the HEX-E through the session's registration (registration.json): the flange's mean pose during the hold, the
HEX-E's deflection under the measured wrench (data sheet compliance), the pad's frame on the flange and the sphere's
centre give where the sphere stands in the pad frame, hence the contact point (the pad's closest surface point), its
normal and the indentation; the HEX-E's change from its zero (the rests before and after the action, interpolated in
time: the sensor drifts), less the change of the payload's weight
with the tool's orientation (the gravity fit), through its fitted 3 x 3 map, gives the force in the pad frame, its press
along the normal, its shear and, from the torques about the contact point, the torsion about the normal. The board's
change is the hold's mean reading (each chip corrected for its die temperature) against the rests before and after the
press, interpolated to the hold's time. The estimator (fingertip_wrench_model.SurfaceEstimator, as the viewer runs it:
magnets fitted to the first rest with the board's chip frames kept, 2 uT floor) reads each change.

In a simulated session (sim_truth.bin) the ground truth is compared with the simulator's own contact: its force, centre of
pressure, normal and torsion, which tests the measurement chain itself.

    .venv/bin/python scripts/ur5_tip_calibration_analysis.py logs/20261022-tip_calibration/<session> [--out DIR]
        [--zero before]     the rest before each action alone as the HEX-E's zero (the analysis up to 2026-10-22 19:40)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "src/morphohand/driver/reskin5x/host"
sys.path.insert(0, str(HOST))
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_tip_calibration as C  # noqa: E402
import ur5_tip_calibration_sim as SIM  # noqa: E402
import fingertip_wrench_model as W  # noqa: E402

DATA = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data"
BASIS = DATA / "20261020-b_v2_surface_basis.npz"
BOARD_CAL = DATA / "20261020-calib_pad_as_found.json"
OUT = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data"
ZERO = "both"           # the HEX-E's zero: "both" rests interpolated in time (default), or the rest "before" alone (--zero)
GRAVITY_FROM = None     # a session (or its registration.json) whose gravity fit a session without one takes (--gravity-from)


def borrow_gravity(reg, src):
    """A session run without the gravity stage takes the fit of another session with the same tool and HEX-E: the
    payload's maps A and Bt, the HEX-E's axes K and Kt, and its turn about the tool axis (the rotation of T_ES). The
    other session's offsets and drift are not used, since every hold's zero comes from its own rests. On 2026-10-22
    first2's fit predicted the five stations of sphere1 to 63 mN rms with only the offset free; its own residual was
    65 mN."""
    p = Path(src)
    p = p / "registration.json" if p.is_dir() else p
    d = json.loads(p.read_text())
    if d.get("grav") is None:
        raise SystemExit(f"{p} has no gravity fit")
    reg.grav = {k: np.array(v, float) for k, v in d["grav"].items()}
    reg.K, reg.Kt = np.array(d["K"], float), np.array(d["Kt"], float)
    reg.T_ES = reg.T_ES.copy()
    reg.T_ES[:3, :3] = np.array(d["T_ES"], float)[:3, :3]
    return reg


def load(sd):
    sd = Path(sd)
    ev = [json.loads(l) for l in open(sd / "events.jsonl") if l.strip()]
    ft = np.fromfile(sd / "ft.bin", "<f8")
    ft = ft[: len(ft) // 9 * 9].reshape(-1, 9)                    # a copy made mid-run can cut the last row
    bd = np.fromfile(sd / "board.bin", C.BoardStream.DTYPE)
    rb = np.fromfile(sd / "robot.bin", "<f8")
    rb = rb[: len(rb) // 15 * 15].reshape(-1, 15)
    truth = None
    if (sd / "sim_truth.bin").exists():
        truth = np.fromfile(sd / "sim_truth.bin", "<f8").reshape(-1, 19)
    return dict(plan=json.loads((sd / "plan.json").read_text()), reg=json.loads((sd / "registration.json").read_text()),
                ev=ev, ft=ft, t_b=bd["t"], r_b=bd["r"].reshape(-1, 5, 4).astype(float), rb=rb, truth=truth,
                sim=json.loads((sd / "sim_truth.json").read_text()) if (sd / "sim_truth.json").exists() else None)


class Session:
    def __init__(self, sd):
        self.d = load(sd)
        self.geo = C.Geometry(self.d["plan"])
        reg = C.Registration(self.geo).load(self.d["reg"])
        if reg.grav is None:
            if not GRAVITY_FROM:
                raise SystemExit(f"{sd}: no gravity fit in this session; pass --gravity-from <a session with one>")
            reg = borrow_gravity(reg, GRAVITY_FROM)
            self.d["reg"].setdefault("notes", {})["gravity_from"] = str(GRAVITY_FROM)
        self.reg = reg
        self.T_EP = reg.T_EP()
        self.c_B = C.apply(reg.T_BI, self.geo.c_I)
        self.Kinv = np.linalg.inv(reg.K)
        self.Ktinv = np.linalg.inv(reg.Kt)
        g = reg.grav
        self.A, self.Bt = g["A"], g["Bt"]
        self.o_E = reg.T_ES[:3, 3]                              # the HEX-E's moment origin in E
        self.R_SE = reg.T_ES[:3, :3].T
        kap = W.load_thermal()
        self.kappa = kap if kap is not None else np.full(5, -0.0067)
        b = self.d["r_b"][:, :, 1:]
        T = self.d["r_b"][:, :, 0]
        c = np.cumsum(np.r_[np.zeros((1, 5)), T], axis=0)       # each chip's die temperature over the 3 s before a record
        i0 = np.searchsorted(self.d["t_b"], self.d["t_b"] - 3.0)
        i1 = np.arange(1, len(T) + 1)
        Tm = (c[i1] - c[i0]) / (i1 - i0)[:, None]
        self.b = W.thermal_correct(b, Tm, self.kappa).reshape(-1, 15)
        self.T_chip = T
        self.ends = {e["id"]: e for e in self.d["ev"] if e.get("event") == "end"}
        self.acts = {a["id"]: a for a in self.d["plan"]["actions"]}
        g = self.geo
        Cd, Nd = W.subdivide(g.pts, g.nrm, g.tris, 3)               # about 0.12 mm apart, the estimator's candidates finer
        self.Cd, self.Nd = Cd, Nd / np.linalg.norm(Nd, axis=1, keepdims=True)
        self.Sd = self.Cd + g.R * self.Nd                           # where the sphere's centre stands touching each point
        sim = self.d["sim"]
        if sim is not None:
            self.T_EP_true = np.array(sim["T_EP"])
            self.c_true = np.array(sim["c_B"])
            self.o_P_true = C.apply(C.inv(self.T_EP_true) @ np.array(sim["T_ES"]), np.zeros(3))
            R_PS = (C.inv(self.T_EP_true) @ np.array(sim["T_ES"]))[:3, :3]
            cw = SIM.HEX_COMPLIANCE
            self.Cw_true = (R_PS, np.array(cw["r_rad_Nmm"]), np.array(cw["f_mm_N"]))

    def contact_on_surface(self, c_P):
        """The pad's surface point (and normal) whose touching sphere's centre lies nearest c_P: the contact point
        without the snapping of the FE mesh's vertices (about 1 mm apart)."""
        k = int(np.argmin(np.linalg.norm(self.Sd - c_P, axis=1)))
        return self.Cd[k], self.Nd[k]

    def win(self, t, x, t0, t1):
        m = (t >= t0) & (t <= t1)
        return x[m]

    def pose(self, t0, t1):
        rows = self.win(self.d["rb"][:, 0], self.d["rb"], t0, t1)
        return C.mean_pose([C.ur_to_T(r[8:14]) for r in rows]) if len(rows) else None

    def rest_board(self, rec, t):
        r0 = self.win(self.d["t_b"], self.b, *rec["t_rest0"]).mean(axis=0)
        r1 = self.win(self.d["t_b"], self.b, *rec["t_rest1"]).mean(axis=0) if rec.get("t_rest1") else r0
        ta, tb = np.mean(rec["t_rest0"]), np.mean(rec["t_rest1"]) if rec.get("t_rest1") else np.mean(rec["t_rest0"]) + 1
        return r0 + (r1 - r0) * (t - ta) / max(tb - ta, 1e-6), float(np.linalg.norm(r1 - r0))

    def truth(self, t0, t1):
        tr = self.d["truth"]
        if tr is None:
            return None
        rows = tr[(tr[:, 0] >= t0) & (tr[:, 0] <= t1)]
        rows = rows[rows[:, 12] > 0]
        if not len(rows):
            return None
        f = rows[:, 1:4].mean(axis=0)
        cop = rows[:, 4:7].mean(axis=0)
        n = rows[:, 7:10].mean(axis=0)
        # the sphere's centre in the true pad frame, the pad moved by the HEX-E's deflection under the true wrench
        T_BE = C.mean_pose([C.ur_to_T(r[13:19]) for r in rows])
        c = C.apply(C.inv(T_BE @ self.T_EP_true), self.c_true)
        R_PS, cr, ct = self.Cw_true
        x = cop
        tau = np.cross(x - self.o_P_true, f)
        dw = R_PS @ (cr * (R_PS.T @ tau))
        dt = R_PS @ (ct * (R_PS.T @ f))
        c = c - (dt + np.cross(dw, c - self.o_P_true))
        return dict(f=f, cop=cop, n=n / np.linalg.norm(n), tn=float(rows[:, 10].mean()), kind=int(rows[0, 12]), c=c)

    def zero(self, rec, t):
        """The HEX-E's reading at rest at time t of an action: the means of the rest before it and the rest after it
        (both at the action's rest pose, out of contact), interpolated in time, since the sensor's zero drifts (the lab's
        untared Fz moved 17 N in 45 min after a power cycle, 2026-10-22). Without a rest after, the rest before alone.
        Returns the zero and the change from the first rest to the second (6, N and N m)."""
        z0 = np.array(rec["tare"], float)
        if ZERO == "before" or not rec.get("t_rest1"):
            return z0, np.zeros(6)
        ft = self.d["ft"]
        w0 = self.win(ft[:, 0], ft[:, 3:9], *rec["t_rest0"])
        w1 = self.win(ft[:, 0], ft[:, 3:9], *rec["t_rest1"])
        if not len(w0) or not len(w1):
            return z0, np.zeros(6)
        z0, z1 = w0.mean(axis=0), w1.mean(axis=0)
        ta, tb = np.mean(rec["t_rest0"]), np.mean(rec["t_rest1"])
        return z0 + (z1 - z0) * (t - ta) / max(tb - ta, 1e-6), z1 - z0

    def ground_truth(self, rec, t0, t1, T_rest_pose):
        """The hold's ground truth (dict) from the robot, the HEX-E and the registration."""
        T_BE = self.pose(t0, t1)
        w = self.win(self.d["ft"][:, 0], self.d["ft"][:, 3:9], t0, t1)
        if T_BE is None or not len(w):
            return None
        w = w.mean(axis=0)
        tare, dz = self.zero(rec, 0.5 * (t0 + t1))
        R0, R1 = T_rest_pose[:3, :3], T_BE[:3, :3]
        g0, g1 = R0.T @ np.array([0.0, 0.0, -1.0]), R1.T @ np.array([0.0, 0.0, -1.0])
        df = w[:3] - tare[:3] - self.A @ (g1 - g0)
        dt = w[3:] - tare[3:] - self.Bt @ (g1 - g0)
        f_E = self.Kinv @ df
        t_E = self.Ktinv @ dt * 1e3                                 # N mm about the HEX-E's origin
        # the HEX-E's deflection under the contact wrench, in its own frame at its origin
        f_S, t_S = self.R_SE @ f_E, self.R_SE @ t_E
        cw = SIM.HEX_COMPLIANCE
        dw = np.array(cw["r_rad_Nmm"]) * t_S
        dtr = np.array(cw["f_mm_N"]) * f_S
        T_ES = self.reg.T_ES
        T_BP = T_BE @ T_ES @ C.Tm(C.exp_so3(dw), dtr) @ C.inv(T_ES) @ self.T_EP
        T_BP0 = T_BE @ self.T_EP
        c_P = C.apply(C.inv(T_BP), self.c_B)
        d, _, _ = self.geo.surface_distance(c_P)
        _, n_c = self.contact_on_surface(c_P)
        p_c = c_P - self.geo.R * n_c              # continuous in c_P: the sphere's point along the normal
        R_PE = self.T_EP[:3, :3].T
        f_P = R_PE @ f_E
        fn = float(-f_P @ n_c)
        ft = f_P + fn * n_c
        p_E = C.apply(self.T_EP, p_c)
        tc_E = t_E - np.cross(p_E - self.o_E, f_E)
        tn = float(tc_E @ (self.T_EP[:3, :3] @ n_c))
        return dict(T_BE=T_BE, c_P=c_P, p=p_c, n=n_c, depth=float(self.geo.R - d), f=f_P, f_n=fn, f_t=ft, tn=tn,
                    deflection_mm=float(np.linalg.norm(C.apply(C.inv(T_BP0), C.apply(T_BP, p_c)) - p_c)),
                    ft_rest_drift_N=float(np.linalg.norm(dz[:3])))

    def holds(self):
        """Every hold of every press and turn with its ground truth and the board's change."""
        out = []
        for aid, e in self.ends.items():
            if e.get("status") != "ok" or aid not in self.acts:
                continue
            a, rec = self.acts[aid], e.get("rec") or {}
            if a["kind"] not in ("press", "slide", "roll", "twist") or not rec.get("levels"):
                continue
            T_rest = np.array(rec["T_rest"]) if rec.get("T_rest") is not None else None
            if T_rest is None:
                continue
            windows = [(lv["t"], dict(level_N=lv["N"], turn_deg=0.0, dir_deg=None)) for lv in rec["levels"]]
            for mv in rec.get("moves", []):
                for tu in mv["turns"]:
                    windows.append((tu["t"], dict(level_N=a.get("press_N"), turn_deg=tu["deg"], dir_deg=mv.get("dir_deg"))))
            for (t0, t1), meta in windows:
                gt = self.ground_truth(rec, t0, t1, T_rest)
                if gt is None:
                    continue
                bb = self.win(self.d["t_b"], self.b, t0, t1)
                if not len(bb):
                    continue
                rest, drift = self.rest_board(rec, 0.5 * (t0 + t1))
                row = dict(id=aid, stage=a["stage"], kind=a["kind"], role=a.get("role"), region=a["target"].get("region"),
                           target=np.array(a["target"]["p"], float), twist=a.get("twist", 0.0), t=0.5 * (t0 + t1),
                           n_rec=len(bb), d_board=bb.mean(axis=0) - rest, rest_drift_uT=drift, **meta, **gt)
                row["truth"] = self.truth(t0, t1)
                out.append(row)
        out.sort(key=lambda r: r["t"])
        return out


def covered(C_, N_):
    """Candidates that something can touch on DIP link b v4: not behind the cover's 3 mm walls (x_t below -1.95 on the
    flanks) and not under cap TW15H3 (x_t below 3.2 from z_t 12.8 on)."""
    return (C_[:, 0] >= -1.9) & ~((C_[:, 2] >= 12.8) & (C_[:, 0] <= 3.2))


def estimator(ses, holds, mask=None):
    """The viewer's estimator: magnets fitted to the rest before the first reference press (chip frames kept); mask
    keeps only the candidates it passes (covered: the ones the cover and the cap leave open)."""
    ref = [h for h in holds if h.get("role") == "ref"]
    first = ses.ends[ref[0]["id"]]["rec"] if ref else ses.ends[holds[0]["id"]]["rec"]
    b_rest = ses.win(ses.d["t_b"], ses.b, *first["t_rest0"]).mean(axis=0).reshape(5, 3)
    board = W.Calib.load(BOARD_CAL)
    cal = W.calibrate_frames_fixed(b_rest, board)
    model = W.Model(cal)
    est = W.SurfaceEstimator(model, W.load_basis(BASIS), floor_uT=2.0, mask=mask)
    return cal, est


def tangential(v, n):
    return v - (v @ n) * n


def evaluate(holds, est, key="est"):
    for h in holds:
        est.set_noise_scale(1.0 / math.sqrt(max(h["n_rec"], 1)))
        h[key] = est.estimate(h["d_board"])
    return holds


def est_scores(holds, key):
    """Contact point error (mm) and relative normal-force error of an estimator, by the press's force level."""
    out = {}
    for lv in sorted({h["level_N"] for h in holds if h["kind"] == "press" and h["level_N"] is not None}):
        hh = [h for h in holds if h["kind"] == "press" and h["level_N"] == lv and h[key].get("contact")]
        if not hh:
            continue
        e = [float(np.linalg.norm(tangential(np.array(h[key]["p"]) - h["p"], h["n"]))) for h in hh]
        r = [abs(h[key]["f_n"] / h["f_n"] - 1.0) for h in hh if h["f_n"] > 0.1]
        out[f"{lv:g}"] = dict(n=len(hh), loc_median_mm=float(np.median(e)), loc_p90_mm=float(np.percentile(e, 90)),
                              far_2mm=float(np.mean(np.array(e) > 2.0)), rel_force_median=float(np.median(r)) if r else None)
    return out


def summarize(ses, holds):
    S = {}
    reg = ses.d["reg"]
    S["registration"] = reg.get("notes", {})
    m_ok = [h for h in holds if h["est"].get("contact")]
    def stats(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return dict(n=int(len(x)), median=float(np.median(x)) if len(x) else None, p90=float(np.percentile(x, 90)) if len(x) else None,
                    mean=float(np.mean(x)) if len(x) else None)
    # ground truth against the simulator's contact
    gt_vs = [h for h in holds if h["truth"] is not None and h["truth"]["kind"] == 1]
    if gt_vs:
        S["ground_truth_vs_sim"] = dict(
            location_mm=stats([np.linalg.norm(tangential(h["c_P"] - h["truth"]["c"], h["n"])) for h in gt_vs]),
            depth_mm=stats([abs(float((h["c_P"] - h["truth"]["c"]) @ h["n"])) for h in gt_vs]),
            normal_force_N=stats([abs(h["f_n"] - float(-h["truth"]["f"] @ h["truth"]["n"])) for h in gt_vs]),
            force_vector_N=stats([np.linalg.norm(h["f"] - h["truth"]["f"]) for h in gt_vs]),
            shear_N=stats([np.linalg.norm(h["f_t"] - tangential(h["truth"]["f"], h["truth"]["n"])) for h in gt_vs]),
            torsion_Nmm=stats([abs(h["tn"] - h["truth"]["tn"]) for h in gt_vs if h["kind"] == "twist"]),
            relative_normal_force=stats([abs(h["f_n"] / float(-h["truth"]["f"] @ h["truth"]["n"]) - 1.0) for h in gt_vs
                                         if float(-h["truth"]["f"] @ h["truth"]["n"]) > 0.4]))
    # estimator against the ground truth
    if m_ok:
        S["estimator_vs_ground_truth"] = dict(
            detected=f"{len(m_ok)}/{len(holds)}",
            location_mm=stats([np.linalg.norm(tangential(np.array(h["est"]["p"]) - h["p"], h["n"])) for h in m_ok]),
            normal_force_N=stats([abs(h["est"]["f_n"] - h["f_n"]) for h in m_ok]),
            relative_normal_force=stats([abs(h["est"]["f_n"] / h["f_n"] - 1.0) for h in m_ok if h["f_n"] > 0.4]))
        by = {}
        for r in ("face", "edge", "flank"):
            hh = [h for h in m_ok if h["region"] == r and h["stage"] == "grid"]
            if hh:
                x = np.array([h["f_n"] for h in hh])
                y = np.array([h["est"]["f_n"] for h in hh])
                by[r] = dict(location_mm=stats([np.linalg.norm(tangential(np.array(h["est"]["p"]) - h["p"], h["n"])) for h in hh]),
                             force_scale=float((x @ y) / (x @ x)))
        S["by_region"] = by
        # the board against the registration along the pad: with every touch at twist 0 the contact point along the
        # finger (z_P) rests on the homing's seating, and the presses then share one offset; the estimator's mean
        # location error at 3 N and over (where its own bias is small) measures it. A check, not a correction: the
        # ground truth stays free of the board. Simulator, 2026-10-22 (truth k4 of QUEUE.md R9): offset 0.325 mm along
        # z_P, board minus registration 0.306 mm, the estimator's own bias -0.019 mm
        hi = [h for h in m_ok if h["kind"] == "press" and (h["level_N"] or 0.0) >= 2.9]
        if len(hi) >= 5:
            d = np.array([tangential(np.array(h["est"]["p"]) - h["p"], h["n"]) for h in hi])
            S["board_minus_registration_mm"] = dict(levels="3 N and over", n=len(hi), mean=d.mean(0), sd=d.std(0),
                                                    se=d.std(0) / math.sqrt(len(hi)))
    gt_t = [h for h in holds if h["truth"] is not None and h["truth"]["kind"] == 1 and h["kind"] == "press"]
    if gt_t:                                                       # simulated sessions: the registration's own offset
        d = np.array([tangential((h["truth"]["c"] - ses.geo.R * h["n"]) - h["p"], h["n"]) for h in gt_t])
        S["registration_offset_vs_truth_mm"] = dict(n=len(gt_t), mean=d.mean(0), sd=d.std(0))
    S["estimator_by_level"] = est_scores(holds, "est")
    if holds and "est_masked" in holds[0]:
        S["estimator_masked_by_level"] = est_scores(holds, "est_masked")
    S["rest_drift_uT"] = stats([h["rest_drift_uT"] for h in holds])
    S["ft_rest_drift_N"] = stats([h["ft_rest_drift_N"] for h in holds])
    S["zero"] = ZERO
    ref = [h for h in holds if h.get("role") == "ref" and abs((h["level_N"] or 0) - 1.0) < 1e-9]
    if len(ref) > 1:
        d0 = ref[0]["d_board"]
        S["reference_press_1N"] = [dict(t_min=round((h["t"] - ref[0]["t"]) / 60, 1), change_uT=float(np.linalg.norm(h["d_board"])),
                                        from_first_uT=float(np.linalg.norm(h["d_board"] - d0)), f_n=h["f_n"]) for h in ref]
    S["holds"] = len(holds)
    return S


def pivot_check(ses, N=None):
    """The program's pivot check (stage_pivot) from the session's files. The board's change at N newtons of push for
    the four offset presses gives its map J (uT per mm along the face's t1 and t2). Through J, each press's change against
    the reference press places it on the face, and the registration's own prediction of the same place checks it. On
    2026-10-22 (sphere2r) the program's own check lost the reference press's board samples from its 180 s ring buffer."""
    pv = ses.d["plan"].get("pivot")
    if not pv:
        return None
    N = float(pv.get("level_N", 1.5)) if N is None else float(N)
    t1, t2 = np.array(pv["t1"], float), np.array(pv["t2"], float)
    d_B = -(ses.reg.T_BI[:3, :3] @ C.EZ)
    ft = ses.d["ft"]

    def level(rec):
        if not rec.get("t_rest0") or not rec.get("t_rest1") or not rec.get("levels"):
            return None
        r0 = ses.win(ses.d["t_b"], ses.b, *rec["t_rest0"]).mean(axis=0)
        r1 = ses.win(ses.d["t_b"], ses.b, *rec["t_rest1"]).mean(axis=0)
        ta, tb = np.mean(rec["t_rest0"]), np.mean(rec["t_rest1"])
        R_BE = np.array(rec["T_stop"])[:3, :3]
        pts = []
        for lv in rec["levels"]:
            vb = ses.win(ses.d["t_b"], ses.b, *lv["t"])
            vf = ses.win(ft[:, 0], ft[:, 3:6], *lv["t"])
            if not len(vb) or not len(vf):
                continue
            rest = r0 + (r1 - r0) * (np.mean(lv["t"]) - ta) / max(tb - ta, 1e-6)
            f_E = np.linalg.solve(ses.reg.K, vf.mean(axis=0) - np.array(rec["tare"])[:3])
            pts.append((float(-(R_BE @ f_E) @ d_B), vb.mean(axis=0) - rest))
        if not pts:
            return None
        pts.sort(key=lambda x: x[0])
        if len(pts) == 1:
            return pts[0][1] * N / pts[0][0]
        (f1, b1), (f2, b2) = pts[-2], pts[-1]
        return b1 + (b2 - b1) * (N - f1) / (f2 - f1)

    recs = [(ses.acts[aid], e["rec"]) for aid, e in ses.ends.items()
            if e.get("status") == "ok" and ses.acts.get(aid, {}).get("stage") == "pivot" and e.get("rec")]
    ref = [v for v in recs if v[0].get("role") == "ref"]
    offs = [v for v in recs if v[0].get("role") == "offset"]
    if not ref or len(offs) < 3:
        return None
    b0 = level(ref[0][1])
    rows, cols = [], []
    for a, r in offs:
        bl = level(r)
        if bl is not None:
            rows.append(a["offset_P"])
            cols.append(bl - b0)
    J = np.linalg.lstsq(np.array(rows, float), np.array(cols), rcond=None)[0].T          # (15, 2)
    q0 = C.apply(C.inv(np.array(ref[0][1]["T_rest"]) @ ses.T_EP), ses.c_B)
    out = dict(level_N=N, t1=t1, t2=t2, presses=[])
    for a, r in recs:
        if a.get("role") == "ref":
            continue
        bl = level(r)
        if bl is None:
            continue
        dp = np.linalg.lstsq(J, bl - b0, rcond=None)[0]
        q1 = C.apply(C.inv(np.array(r["T_rest"]) @ ses.T_EP), ses.c_B)
        pred = np.array([(q1 - q0) @ t1, (q1 - q0) @ t2])
        out["presses"].append(dict(id=a["id"], role=a["role"], twist=float(a.get("twist", 0.0)), offset_P=a["offset_P"],
                                   dp_board=dp, dp_registration=pred, board_minus_registration=dp - pred,
                                   resid_uT=float(np.linalg.norm(J @ dp - (bl - b0)))))
    return out


def save(holds, path):
    keys = ["t", "level_N", "turn_deg", "twist", "f_n", "depth", "tn", "rest_drift_uT", "n_rec", "deflection_mm", "ft_rest_drift_N"]
    arr = {k: np.array([h[k] if h[k] is not None else np.nan for h in holds], float) for k in keys}
    for k in ("p", "n", "f", "f_t", "c_P", "target", "d_board"):
        arr[k] = np.array([h[k] for h in holds], float)
    arr["id"] = np.array([h["id"] for h in holds])
    arr["kind"] = np.array([h["kind"] for h in holds])
    arr["region"] = np.array([h["region"] or "" for h in holds])
    arr["est_contact"] = np.array([bool(h["est"].get("contact")) for h in holds])
    arr["est_p"] = np.array([h["est"].get("p", [np.nan] * 3) for h in holds], float)
    arr["est_f"] = np.array([h["est"].get("f", [np.nan] * 3) for h in holds], float)
    arr["est_f_n"] = np.array([h["est"].get("f_n", np.nan) for h in holds], float)
    if holds and "est_masked" in holds[0]:
        arr["estm_p"] = np.array([h["est_masked"].get("p", [np.nan] * 3) for h in holds], float)
        arr["estm_f_n"] = np.array([h["est_masked"].get("f_n", np.nan) for h in holds], float)
    if holds and holds[0]["truth"] is not None:
        arr["true_f"] = np.array([h["truth"]["f"] if h["truth"] else [np.nan] * 3 for h in holds], float)
        arr["true_cop"] = np.array([h["truth"]["cop"] if h["truth"] else [np.nan] * 3 for h in holds], float)
        arr["true_n"] = np.array([h["truth"]["n"] if h["truth"] else [np.nan] * 3 for h in holds], float)
        arr["true_tn"] = np.array([h["truth"]["tn"] if h["truth"] else np.nan for h in holds], float)
        arr["true_c"] = np.array([h["truth"]["c"] if h["truth"] else [np.nan] * 3 for h in holds], float)
    np.savez_compressed(path, **arr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--out", default=None)
    ap.add_argument("--zero", choices=("both", "before"), default="both", help="the HEX-E's zero for each hold: the rests "
                    "before and after the action interpolated in time, or the rest before alone (the analysis before 2026-10-22 20:00)")
    ap.add_argument("--gravity-from", default=None, help="a session (or its registration.json) whose gravity fit a "
                    "session run without the gravity stage takes")
    a = ap.parse_args()
    global ZERO, GRAVITY_FROM
    ZERO = a.zero
    GRAVITY_FROM = a.gravity_from
    ses = Session(a.session)
    holds = ses.holds()
    cal, est = estimator(ses, holds)
    evaluate(holds, est)
    _, estm = estimator(ses, holds, mask=covered)
    evaluate(holds, estm, "est_masked")
    S = summarize(ses, holds)
    S["estimator_calibration"] = dict(k=cal.k, poles=cal.poles, dx=cal.dx, dy=cal.dy, dz=cal.dz, rms_mT=cal.rms_mT)
    S["pivot_check"] = pivot_check(ses)
    for pr in (S["pivot_check"] or {}).get("presses", []):
        print(f"pivot {pr['id']} ({pr['role']}, twist {pr['twist']:+.0f} deg): the board places it "
              f"{pr['dp_board'][0]:+.3f}, {pr['dp_board'][1]:+.3f} mm from the reference, the registration "
              f"{pr['dp_registration'][0]:+.3f}, {pr['dp_registration'][1]:+.3f} (fit residual {pr['resid_uT']:.1f} uT)")
    out = Path(a.out or OUT / Path(a.session).name)
    out.mkdir(parents=True, exist_ok=True)
    save(holds, out / "holds.npz")
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=C.jsonable))
    print(json.dumps({k: v for k, v in S.items() if k not in ("reference_press_1N",)}, indent=1, default=C.jsonable)[:4000])
    print(f"{len(holds)} holds -> {out}")


if __name__ == "__main__":
    main()
