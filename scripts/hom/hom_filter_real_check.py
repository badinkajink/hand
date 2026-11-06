#!/usr/bin/env python3
"""The contact filter of the HOM controller (src/sr2hand/control/hom.py ContactFilter, arXiv 2609.25619 Eqs. 4-6) run
on the inverted rig's rod rolls, against the contact point that the robot and the HEX-E give (owner, 2026-11-06: "how
well would this work on real? can we check with our calibration data?").

Data. The rod night of 11-03/04 pressed a rod of radius 12.5 mm on the pads of v3 IDs 5, 4 and 6 (seats 2, 3 and 1) and
rolled it about its contact line by +3, +6, -3, -6 and 0 deg after a press to 4 N, and again after a press to 1.5 N
(holds of 0.4 s; the turn about the line runs at the robot's own rate, about 0.4 deg/s). Per robot sample (125 Hz) the
rod's axis is placed in the pad frame P as scripts/ur5/ur5_probe_calibration_analysis.py places it per hold (flange
pose, HEX-E deflection under the measured wrench, T_EM, the joint rod registration's c_M and axis_M). The analysis labels
the contact as the point of the pad's mesh nearest the axis; per sample that label snaps to the mesh's vertex rows, 1 mm
apart along z_P, and stays put over up to 0.5 mm of the contact's travel. The truth here is the same label on the smooth
pad, the fitted HalfEllipsoid's point nearest the axis; the mesh label is kept for comparison (vs_mesh_label). Both are
interpolated to the record times of the held board (159 records/s). Each record's change from the action's rests (die
temperature corrected, as the analysis) goes through a board GP (ur5_tip_calibration_board_gp.fit) trained on the tip's
press, sweep, offset and slide holds of its rig sessions (ur5_tip_calibration_rig_coverage.RIG), without any hold of a
roll action.

Analytic pad. The pad's free surface in P (the plan's surface mesh) is fitted by a rigid motion of
surfaces.HalfEllipsoid(8.5, 7.4, 11, 22 mm) (frame: x outward at the apex, y across, z along the finger with z < 0
distal), starting from x_e = x_P, y_e = -y_P, z_e = 10.86 mm - z_P. The fit uses the dome (x_P > 0.5 mm, off the flat
distal end face); the residuals of all vertices and of the simulation's sphere pads of the v4b tip are reported too.

Filter. The pad frame P (in m) is the world; a stand-in for hom.Mirror gives the pad's pose (fixed), the rod's pose per
record (its material frame, z along its axis) and the joint axis (the pad's y). The measured point is turned into
generalized contact frames by hom.Mirror.gcf_from_cop (projections on the analytic pad and on the rod,
surfaces.Cylinder(12.5 mm, 10 mm)), and hom.ContactFilter runs with FilterParams() (k_min 5 1/s, k_max 100 1/s,
alpha 0.02 m/s) in three settings:
  raw       the measured point projected on the pad
  kmin      V_ref = 0, as in the HOM tasks: the gain stays at k_min, a low-pass of time constant 0.2 s
  roll      V_ref = the rod's pure roll about its contact line at the measured turn rate: angular part -omega_rod in the
            GCF {L1} axes (spin, roll_y, roll_z), zero linear part; roll_slide_rates turns it into geodesic velocities
            that move the frames ahead of the correction, and Eq. 6 raises the gain with their speed
kmin and roll also run at 50 Hz (the latest record at each tick) and, as a check of the filter alone, on the truth itself.

Metrics, per roll and pooled by press force: the distance along the pad between estimate and truth (the tangential part
at the truth's normal, mm), RMS and median over the holds (each 0.4 s plateau after its first 0.1 s, and the press's
level hold) and over the moving frames (between plateaus, the turn faster than 0.05 deg/s); the jitter (RMS step between
consecutive frames from the level hold on, mm); the lag (the time shift of the truth that minimizes the RMS of the
estimate's error along the roll direction, searched over -0.5 to 1 s); the truth's travel per roll; the raw estimate's
mean error vector at the holds (bias).

    ~/.claude/bin/resguard.sh run --mem 4G --cpu 500 -- env NUMPY_MADVISE_HUGEPAGE=0 OPENBLAS_NUM_THREADS=4 \\
        taskset -c 4-7 .venv/bin/python scripts/hom/hom_filter_real_check.py run --seats 2 3 1
    .venv/bin/python scripts/hom/hom_filter_real_check.py figure      (reads only the saved rows and traces)
2026-11-06: 22 rolls in 22 min, 2.9 GB at most.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
JOINT = ROOT / "docs/experiments/20261030-inverted_rig_seat_block/data/20261104-strip_table_pads_rod_joint_registration_final_grid05.json"
HOLDS = ROOT / "docs/experiments/20261030-inverted_rig_seat_block/data/holds"
OUT = ROOT / "docs/experiments/20261029-hom_control_review"
ROWS = OUT / "rows/20261106-filter_real_check.json"
TRACES = OUT / "rows/20261106-filter_real_traces.npz"
FIG = OUT / "media/20261106-filter_real_roll.png"
SIM_PADS = ROOT / "logs/20261030-hand_object_scale_cad_v2/scenes/cyl020.0/x092.5_y102.5_pads_bv2.xml"
SEATS = {2: dict(session="20261103-172647-rod21", tip=5, plans="seat2_v3id5"),
         3: dict(session="20261103-202017-rod32", tip=4, plans="seat3_v3id4"),
         1: dict(session="20261103-214753-rod13", tip=6, plans="seat1_v3id6")}
PLAN_PARTS = ("sweep8", "sweep8_fit", "half_a", "half_b", "extras_night")      # the night's plans of each seat
TRAIN_KINDS = ["press", "sweep", "offset", "slide"]
ELL_MM = (8.5, 7.4, 11.0, 22.0)          # surfaces.HalfEllipsoid(ax, ay, az, azp) of the v4b pad (tcd_ell7.4x8.5x11x22)
APEX_Z_P = 10.86                         # mm: the pad's apex plane in P (the envelope's mount face at z_P = 0)
ROD_HALF_LEN_M = 0.010
PRE_S = 1.0                              # s of the press before its level hold: the filter's start
SKIP_S = 0.1                             # s at the start of each plateau left out of the hold metrics
MOVING_DEG_S = 0.05                      # deg/s: a frame between holds is moving when the turn is faster
CHUNK = 4096                             # records per GP prediction
RATE_CTRL = 50.0                         # Hz, the controller rate of the resampled runs
LAGS = np.arange(-0.5, 1.0001, 0.005)    # s, the truth's shifts searched for the lag
SETTINGS = ("raw", "kmin", "roll", "kmin50", "roll50")


# ------------------------------------------------------------------------------------------------ the analytic pad
def fit_pad(pts, nrm):
    """Rigid motion (R_Pe, t_Pe) of the HalfEllipsoid frame in P (mm) fitted to the pad's dome vertices; the signed
    distances of all vertices and of the dome to the fitted surface."""
    from scipy.optimize import least_squares
    from sr2hand.control.surfaces import HalfEllipsoid
    from sr2hand.geometry.rotations import exp_so3, log_so3
    ell = HalfEllipsoid(*ELL_MM)
    R0 = np.diag([1.0, -1.0, -1.0])
    dome = (pts[:, 0] > 0.5) & (np.abs(nrm[:, 2]) < 0.9)

    def sdf(x, P):
        R, t = exp_so3(x[:3]) @ R0, x[3:]
        return np.array([ell.closest(q)[0] for q in (P - t) @ R])
    x0 = np.r_[0.0, 0.0, 0.0, 0.0, 0.0, APEX_Z_P]
    sol = least_squares(lambda x: sdf(x, pts[dome]), x0, x_scale=np.r_[0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    R, t = exp_so3(sol.x[:3]) @ R0, sol.x[3:]
    d_all, d_dome, d0 = sdf(sol.x, pts), sdf(sol.x, pts[dome]), sdf(x0, pts[dome])
    front = pts[:, 0] > 0.5
    summ = dict(
        R_Pe=np.round(R, 6).tolist(), t_Pe_mm=np.round(t, 4).tolist(),
        turn_from_nominal_deg=round(math.degrees(float(np.linalg.norm(log_so3(R0.T @ R)))), 3),
        shift_from_nominal_mm=np.round(t - x0[3:], 4).tolist(),
        dome=dict(n=int(dome.sum()), rms_mm=round(float(np.sqrt(np.mean(d_dome ** 2))), 4),
                  max_abs_mm=round(float(np.abs(d_dome).max()), 4),
                  rms_at_nominal_mm=round(float(np.sqrt(np.mean(d0 ** 2))), 4)),
        all_vertices=dict(n=int(len(pts)), rms_mm=round(float(np.sqrt(np.mean(d_all ** 2))), 4),
                          max_abs_mm=round(float(np.abs(d_all).max()), 4)),
        front_incl_end_face=dict(n=int(front.sum()), rms_mm=round(float(np.sqrt(np.mean(d_all[front] ** 2))), 4)),
        behind_centre_plane=dict(n=int((~front).sum()), rms_mm=round(float(np.sqrt(np.mean(d_all[~front] ** 2))), 4)
                                 if (~front).any() else None))
    return R, t, summ


def sim_pads_check():
    """The simulation's 1 mm sphere pads of the v4b tip (the HOM study's cached cad_v2 scene, thumb, palmar sign +1)
    against hom_tasks.tip_surface's HalfEllipsoid, posed at (R - ax, 0, 0) in the tip frame."""
    from sr2hand.control.surfaces import HalfEllipsoid, Posed
    if not SIM_PADS.exists():
        return dict(skipped=f"{SIM_PADS.relative_to(ROOT)} not found")
    root = ET.parse(SIM_PADS).getroot()
    b = root.find(".//body[@name='thumb_tip']")
    g = [x for x in b.findall("geom") if x.get("name", "").startswith("thumb_pad")]
    C = np.array([[float(v) for v in x.get("pos").split()] for x in g])
    rs = float(g[0].get("size"))
    surf = Posed(HalfEllipsoid(*(np.array(ELL_MM) * 1e-3)), pos=(0.01055 - 0.0085, 0.0, 0.0))
    d = np.array([surf.closest(c)[0] for c in C])            # centres: rs inside the surface where the pad is on it
    dome = (C[:, 0] > 0.01055 - 0.0085) & (C[:, 2] > -0.01055 + 2 * rs)
    e = d + rs                                                # the sphere's outer point against the surface
    return dict(file=str(SIM_PADS.relative_to(ROOT)), n=int(len(C)), rs_mm=rs * 1e3,
                centre_depth_mm=dict(median=round(float(-np.median(d)) * 1e3, 4)),
                outer_point_rms_mm=round(float(np.sqrt(np.mean(e ** 2))) * 1e3, 4),
                outer_point_rms_dome_mm=round(float(np.sqrt(np.mean(e[dome] ** 2))) * 1e3, 4),
                outer_point_max_abs_dome_mm=round(float(np.abs(e[dome]).max()) * 1e3, 4),
                n_dome=int(dome.sum()))


# ------------------------------------------------------------------------------------------------ the session
def load_session(seat):
    import ur5_probe_calibration_analysis as PA
    s = SEATS[seat]
    PA.JOINT[:] = [str(JOINT)]
    PA.PLANS[:] = [str(p) for p in (LAB / f"20261103-plan_probe_table_rod_{s['plans']}_{k}.json" for k in PLAN_PARTS)
                   if p.exists()]
    return PA.ProbeSession(LAB / s["session"])


def plateaus(rec):
    """[(t0, t1, label)] of a roll: the press's level hold, then each turn's hold."""
    out = [(rec["levels"][0]["t"][0], rec["levels"][0]["t"][1], "press")]
    for mv in rec["moves"]:
        for tu in mv["turns"]:
            out.append((tu["t"][0], tu["t"][1], f"{tu['deg']:+g}"))
    return out


def contact_batch(ses, c_P, a_P):
    """ProbeSession.contact for many rod poses: the pad's point (ses.Cd, the plan's mesh subdivided) nearest the axis
    within half_len of c_P along it (index, distance)."""
    g = ses.geo
    k_out, d_out = np.zeros(len(c_P), int), np.zeros(len(c_P))
    for i in range(0, len(c_P), 64):
        c, a = c_P[i:i + 64], a_P[i:i + 64]
        r = ses.Cd[None] - c[:, None]
        y = np.einsum("nmk,nk->nm", r, a)
        rp = r - y[..., None] * a[:, None]
        dist = np.where(np.abs(y) <= g.half_len, np.linalg.norm(rp, axis=2), np.inf)
        k = np.argmin(dist, axis=1)
        k_out[i:i + 64], d_out[i:i + 64] = k, dist[np.arange(len(k)), k]
    return k_out, d_out


def smooth_contact(c_P, a_P, p_seed, R_Pe, t_Pe, half_len):
    """The same label on the smooth pad: the fitted HalfEllipsoid's point nearest the rod's axis (mm, P), its outward
    normal and the axis' distance from it, by Brent's method along the axis within half_len of c_P, started from the
    mesh's nearest point p_seed. The mesh's nearest vertex snaps to its rows 1 mm apart along z_P; this point does not."""
    from scipy.optimize import minimize_scalar
    from sr2hand.control.surfaces import HalfEllipsoid
    ell = HalfEllipsoid(*ELL_MM)
    P, N, D = np.zeros_like(c_P), np.zeros_like(c_P), np.zeros(len(c_P))
    for i in range(len(c_P)):
        ce, ae = R_Pe.T @ (c_P[i] - t_Pe), R_Pe.T @ a_P[i]
        s0 = float((p_seed[i] - c_P[i]) @ a_P[i])
        lo, hi = max(-half_len, s0 - 3.0), min(half_len, s0 + 3.0)
        r = minimize_scalar(lambda s: ell.closest(ce + s * ae)[0], bounds=(lo, hi), method="bounded",
                            options=dict(xatol=1e-4))
        d, q, n = ell.closest(ce + r.x * ae)
        P[i], N[i], D[i] = R_Pe @ q + t_Pe, R_Pe @ n, d
    return P, N, D


def roll_trace(ses, a, R_Pe, t_Pe):
    """Per robot sample and per board record of one roll: the truth (the analysis's label on the mesh, and the same
    label on the fitted smooth pad), the rod's pose and turn, the board's change."""
    import ur5_probe_calibration_analysis as PA
    from sr2hand.geometry.rotations import exp_so3, frame_from_axis, log_so3, tf_from_ur
    C, SIM = PA.C, PA.SIM
    rec = ses.ends[a["id"]]["rec"]
    pl = plateaus(rec)
    t_a, t_z = pl[0][0] - PRE_S, pl[-1][1]
    rb, ft = ses.d["rb"], ses.d["ft"]
    m = (rb[:, 0] >= t_a - 0.3) & (rb[:, 0] <= t_z + 0.3)
    tr = rb[m, 0]
    T_BE = tf_from_ur(rb[m, 8:14], 1e3)
    w = np.column_stack([np.interp(tr, ft[:, 0], ft[:, 3 + j]) for j in range(6)])
    za, zb = ses.zero(rec, t_a)[0], ses.zero(rec, t_z)[0]            # the zero is linear in time between the rests
    tare = za + (zb - za) * ((tr - t_a) / (t_z - t_a))[:, None]
    f_E = (w[:, :3] - tare[:, :3]) @ ses.Kinv.T
    t_E = (w[:, 3:] - tare[:, 3:]) @ ses.Ktinv.T * 1e3
    cw = SIM.HEX_COMPLIANCE
    dw = np.array(cw["r_rad_Nmm"]) * (t_E @ ses.R_SE.T)
    dtr = np.array(cw["f_mm_N"]) * (f_E @ ses.R_SE.T)
    D = np.tile(np.eye(4), (len(tr), 1, 1))
    D[:, :3, :3], D[:, :3, 3] = exp_so3(dw), dtr
    T_ES, T_EM = ses.reg.T_ES, ses.reg.T_EM
    T_PM = ses.T_PB @ T_BE @ T_ES @ D @ C.inv(T_ES) @ T_EM
    c_P = T_PM[:, :3, :3] @ ses.c_M + T_PM[:, :3, 3]
    a_P = T_PM[:, :3, :3] @ ses.axis_M
    k, dist = contact_batch(ses, c_P, a_P)
    p_s, n_s, d_s = smooth_contact(c_P, a_P, ses.Cd[k], R_Pe, t_Pe, ses.geo.half_len)
    R_PE = (ses.T_PB @ T_BE)[:, :3, :3]
    f_P = -np.einsum("nij,nj->ni", R_PE, f_E)
    # the turn: the flange's rotation from the pressed pose about the move's axis (deg), and its rate (rad/s)
    mv = rec["moves"][0]
    ax_B = np.asarray(mv["axis_B"], float)
    ax_B /= np.linalg.norm(ax_B)
    R_stop = np.asarray(rec["T_stop"], float)[:3, :3]
    th = log_so3(T_BE[:, :3, :3] @ R_stop.T) @ ax_B
    ker = np.ones(13) / 13.0                                          # 0.1 s at 125 Hz
    th_s = np.convolve(np.pad(th, 6, mode="edge"), ker, mode="valid")
    th_dot = np.gradient(th_s, tr)
    ax_P = ses.T_PB[:3, :3] @ ax_B
    R_Mrod = frame_from_axis(ses.axis_M, (1.0, 0.0, 0.0), which=2)   # the rod's material frame in M, z along its axis
    rob = dict(t=tr, p=p_s, n=n_s, depth=ses.geo.R - d_s, p_mesh=ses.Cd[k], n_mesh=ses.Nd[k],
               depth_mesh=ses.geo.R - dist, c_P=c_P, a_P=a_P, R_rod=T_PM[:, :3, :3] @ R_Mrod, f_P=f_P,
               theta=np.degrees(th), theta_dot=th_dot, ax_P=ax_P)
    # the held board's records over the window and their change from the action's rests (linear in time)
    mb = (ses.d["t_b"] >= t_a) & (ses.d["t_b"] <= t_z)
    tb = ses.d["t_b"][mb]
    ra, rz = ses.rest_board(rec, t_a)[0], ses.rest_board(rec, t_z)[0]
    rest = ra + (rz - ra) * ((tb - t_a) / (t_z - t_a))[:, None]
    brd = dict(t=tb, d_board=ses.b[mb] - rest)
    return dict(id=a["id"], press_N=float(a["press_N"]), plateaus=pl, t0=pl[0][0], rob=rob, brd=brd, rec=rec,
                target=np.asarray(a["target"]["p"], float))


def check_against_holds(tr, seat):
    """Each plateau's mean of the per-sample labels against the analysis's hold (scripts/ur5/
    ur5_probe_calibration_analysis.py, the _dense folder), mm: [(the mesh label, the smooth label)]."""
    z = np.load(HOLDS / f"{SEATS[seat]['session']}_dense" / "holds.npz", allow_pickle=True)
    ids, tt = z["id"].astype(str), z["t"]
    out = []
    for t0, t1, lab in tr["plateaus"]:
        j = np.flatnonzero((ids == tr["id"]) & (np.abs(tt - 0.5 * (t0 + t1)) < 1e-3))
        if not len(j):
            continue
        m = (tr["rob"]["t"] >= t0) & (tr["rob"]["t"] <= t1)
        out.append((float(np.linalg.norm(tr["rob"]["p_mesh"][m].mean(0) - z["p"][j[0]])),
                    float(np.linalg.norm(tr["rob"]["p"][m].mean(0) - z["p"][j[0]]))))
    return out


# ------------------------------------------------------------------------------------------------ the GP
def train_gp(tip):
    """The board GP on the tip's rig holds of the kinds TRAIN_KINDS, without any hold of a roll action (a roll's level
    hold before it turns is a hold of kind press in the analysis)."""
    import ur5_tip_calibration_board_gp as BG
    import ur5_tip_calibration_rig_coverage as RC
    H = RC.load(RC.RIG[tip], TRAIN_KINDS, f"rig{tip}")
    keep = np.char.find(H["id"].astype(str), "-roll") < 0
    H = {k: v[keep] for k, v in H.items()}
    t0 = time.time()
    m = BG.fit(H["d_board"], np.c_[H["p"], H["f"]], RC.FLOOR)
    info = dict(sessions=list(RC.RIG[tip]), kinds=TRAIN_KINDS, n_holds=int(len(H["p"])),
                roll_action_holds_left_out=int((~keep).sum()), length=m["ell"], noise=m["nz"],
                fit_s=round(time.time() - t0, 1))
    return m, H, info


def gp_predict(m, X):
    import ur5_tip_calibration_board_gp as BG
    return np.concatenate([BG.predict(m, X[i:i + CHUNK]) for i in range(0, len(X), CHUNK)]) if len(X) else np.zeros((0, 6))


def roll_holds_check(m, seat):
    """The GP on the roll holds' own 0.4 s means (held out of its training): location error, mm (holds of 1 N and
    over), as ur5_tip_calibration_rig_coverage.errs."""
    import ur5_tip_calibration_rig_coverage as RC
    z = np.load(HOLDS / f"{SEATS[seat]['session']}_dense" / "holds.npz", allow_pickle=True)
    k = z["kind"].astype(str) == "roll"
    Hh = dict(p=z["p"][k], f=z["f"][k], n=z["n"][k])
    P = gp_predict(m, z["d_board"][k])
    return RC.errs(Hh, P, np.ones(k.sum(), bool))


# ------------------------------------------------------------------------------------------------ the filter
class Bench:
    """Stand-in for hom.Mirror on the rig: world = the pad frame P (m), the pad fixed at its origin, the rod posed per
    record; gcf_from_cop is hom.Mirror's own."""

    def __init__(self, pad_surface, rod_surface, joint_axis):
        from sr2hand.control import hom as H
        self.fingers = {"pad": H.FingerSpec("pad", [], "pad", "pad", "pad", pad_surface)}
        self.obj = H.ObjectSpec("rod", "rod", rod_surface)
        self.o, self.Ro, self.axis = np.zeros(3), np.eye(3), np.asarray(joint_axis, float)

    def obj_pose(self):
        return self.o.copy(), self.Ro.copy()

    def body_pose(self, f):
        return np.zeros(3), np.eye(3)

    def joint_axis(self, f):
        return self.axis.copy()

    def gcf_from_cop(self, f, cop):
        from sr2hand.control import hom as H
        return H.Mirror.gcf_from_cop(self, f, cop)


def run_filter(bench, meas, rod_o, rod_R, omega, mode, dts):
    """hom.ContactFilter over the measured points meas (n, 3; m, in P), the rod posed at (rod_o, rod_R) for each, step
    dts[i]: the estimate on the pad (n, 3) and the finger-side gain k1 (n). mode 'raw' (the measured point projected on
    the pad, no filter), 'kmin' (V_ref = 0) or 'roll' (V_ref = the pure roll: -omega in {L1}'s axes, linear part 0)."""
    from sr2hand.control import hom as H
    flt = H.ContactFilter(bench.fingers["pad"], bench.obj, H.FilterParams())
    est, gain = np.zeros_like(meas), np.zeros(len(meas))
    V = np.zeros(6)
    for i in range(len(meas)):
        bench.o, bench.Ro = rod_o[i], rod_R[i]
        g = bench.gcf_from_cop("pad", meas[i])
        if mode == "raw":
            est[i] = g.p1
            continue
        V[:3] = g.R1.T @ (-omega[i]) if mode == "roll" else 0.0
        est[i] = flt.update(bench, "pad", g, V, float(dts[i]))[0]
        gain[i] = flt.k[1]
    return est, gain


# ------------------------------------------------------------------------------------------------ metrics
def tangential(d, n):
    return d - np.sum(d * n, 1)[:, None] * n


def masks(t, pl, rate_deg_s):
    """Frames at the holds (each plateau after its first SKIP_S) and moving frames (between plateaus, the turn's
    smoothed rate above MOVING_DEG_S)."""
    hold, between = np.zeros(len(t), bool), np.zeros(len(t), bool)
    for t0, t1, _ in pl:
        hold |= (t >= t0 + SKIP_S) & (t <= t1)
    for (_, a1, _), (b0, _, _) in zip(pl[:-1], pl[1:]):
        between |= (t > a1) & (t < b0)
    return hold, between & (np.abs(rate_deg_s) > MOVING_DEG_S)


def lag_of(t, s_est, t_true, s_true, win):
    """The shift tau (s) of the truth minimizing the RMS of s_est(t) - s_true(t - tau) over the window, that RMS, and
    the RMS at tau = 0."""
    r = np.array([np.sqrt(np.mean((s_est[win] - np.interp(t[win] - tau, t_true, s_true)) ** 2)) for tau in LAGS])
    k = int(np.argmin(r))
    return float(LAGS[k]), float(r[k]), float(r[np.argmin(np.abs(LAGS))])


def rms(x):
    return round(float(np.sqrt(np.mean(x ** 2))), 4) if len(x) else None


def med(x):
    return round(float(np.median(x)), 4) if len(x) else None


def metrics(t, est, truth, nrm, hold, move, s_dir, p0, t_true, s_true):
    """Errors along the pad (mm) of one estimate against the truth at the same times: the tangential part at the
    truth's normal. Jitter: the RMS step between consecutive frames from the press's level hold on."""
    e = np.linalg.norm(tangential(est - truth, nrm), axis=1)
    win = (t >= T_WIN[0]) & (t <= T_WIN[1])
    step = np.linalg.norm(np.diff(est[win], axis=0), axis=1)
    lag, lag_rms, rms0 = lag_of(t, (est - p0) @ s_dir, t_true, s_true, win)
    return dict(hold_rms=rms(e[hold]), hold_median=med(e[hold]), move_rms=rms(e[move]), move_median=med(e[move]),
                jitter_rms=rms(step), lag_ms=round(lag * 1e3, 1), roll_dir_rms_at_lag_mm=round(lag_rms, 4),
                roll_dir_rms_at_0_mm=round(rms0, 4), n_hold=int(hold.sum()), n_move=int(move.sum()),
                _e_hold=e[hold], _e_move=e[move], _step=step)


def ticks(t_rec, t0, t1, rate):
    """The controller's ticks at rate over [t0, t1] and the index of the latest record at each."""
    tk = np.arange(t0, t1, 1.0 / rate)
    j = np.searchsorted(t_rec, tk, side="right") - 1
    ok = j >= 0
    return tk[ok], j[ok]


def at(t, t_src, x):
    """x (n, k) sampled at t_src, linearly interpolated to t."""
    x = np.asarray(x, float)
    return np.column_stack([np.interp(t, t_src, x[:, j]) for j in range(x.shape[1])])


T_WIN = [0.0, 0.0]          # the analysis window of the roll in hand: the press's level hold to the last hold's end


def analyse_roll(tr, m_gp, bench):
    """GP, truth at the records, the settings and their metrics for one roll; returns (row, trace, pooled errors)."""
    rob, brd, pl = tr["rob"], tr["brd"], tr["plateaus"]
    tb, tr_ = brd["t"], rob["t"]
    T_WIN[:] = [pl[0][0], pl[-1][1]]
    P = gp_predict(m_gp, brd["d_board"])
    gp_p, gp_f = P[:, :3], P[:, 3:6]
    truth, truth_mesh, f_true, c_rec = at(tb, tr_, rob["p"]), at(tb, tr_, rob["p_mesh"]), at(tb, tr_, rob["f_P"]), \
        at(tb, tr_, rob["c_P"])
    nrm = at(tb, tr_, rob["n"])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    R_rec = rob["R_rod"][np.clip(np.searchsorted(tr_, tb), 0, len(tr_) - 1)]
    theta = np.interp(tb, tr_, rob["theta"])
    rate = np.interp(tb, tr_, rob["theta_dot"])
    om = rate[:, None] * rob["ax_P"][None]                                   # the rod's angular velocity in P, rad/s
    # the roll direction: from the -6 deg hold's truth to the +6 deg hold's, on the tangent plane at the press
    hm = {lab: rob["p"][(tr_ >= t0 + SKIP_S) & (tr_ <= t1)].mean(0) for t0, t1, lab in pl}
    hm_mesh = {lab: rob["p_mesh"][(tr_ >= t0 + SKIP_S) & (tr_ <= t1)].mean(0) for t0, t1, lab in pl}
    p0 = hm["press"]
    n0 = rob["n"][(tr_ >= pl[0][0]) & (tr_ <= pl[0][1])].mean(0)
    n0 /= np.linalg.norm(n0)
    s_dir = tangential((hm["+6"] - hm["-6"])[None], n0[None])[0]
    s_dir /= np.linalg.norm(s_dir)
    s_true = (rob["p"] - p0) @ s_dir
    w_r = (tr_ >= pl[0][0]) & (tr_ <= pl[-1][1])
    travel = float(s_true[w_r].max() - s_true[w_r].min())
    hold, move = masks(tb, pl, np.degrees(rate))
    # the settings at the record rate (dt each record's own interval) and at 50 Hz (the latest record at each tick)
    dt_rec = np.r_[np.median(np.diff(tb)), np.diff(tb)]
    ests, gains = {}, {}
    for mode in ("raw", "kmin", "roll"):
        e, g = run_filter(bench, gp_p * 1e-3, c_rec * 1e-3, R_rec, om, mode, dt_rec)
        ests[mode], gains[mode] = e * 1e3, g
    tk, jk = ticks(tb, tb[0], tb[-1], RATE_CTRL)
    c_k, R_k = at(tk, tr_, rob["c_P"]), rob["R_rod"][np.clip(np.searchsorted(tr_, tk), 0, len(tr_) - 1)]
    rate_k = np.interp(tk, tr_, rob["theta_dot"])
    for mode in ("kmin", "roll"):
        e, g = run_filter(bench, gp_p[jk] * 1e-3, c_k * 1e-3, R_k, rate_k[:, None] * rob["ax_P"][None], mode,
                          np.full(len(tk), 1.0 / RATE_CTRL))
        ests[mode + "50"], gains[mode + "50"] = e * 1e3, g
    fed = {}                                         # the filter alone: the truth itself as the measured point
    for mode in ("kmin", "roll"):
        fed[mode] = run_filter(bench, truth * 1e-3, c_rec * 1e-3, R_rec, om, mode, dt_rec)[0] * 1e3
    truth_k, nrm_k = at(tk, tr_, rob["p"]), at(tk, tr_, rob["n"])
    nrm_k /= np.linalg.norm(nrm_k, axis=1, keepdims=True)
    hold_k, move_k = masks(tk, pl, np.degrees(rate_k))
    met = {}
    for s in SETTINGS:
        if s.endswith("50"):
            met[s] = metrics(tk, ests[s], truth_k, nrm_k, hold_k, move_k, s_dir, p0, tr_, s_true)
        else:
            met[s] = metrics(tb, ests[s], truth, nrm, hold, move, s_dir, p0, tr_, s_true)
    for s in ("kmin", "roll"):
        met[f"{s}_truthfed"] = metrics(tb, fed[s], truth, nrm, hold, move, s_dir, p0, tr_, s_true)
    met["gp_unprojected"] = metrics(tb, gp_p, truth, nrm, hold, move, s_dir, p0, tr_, s_true)
    vs_mesh = {}                                     # the same estimates against the analysis's label on the mesh
    for s in ("raw", "kmin", "roll"):
        e = np.linalg.norm(tangential(ests[s] - truth_mesh, nrm), axis=1)
        vs_mesh[s] = dict(hold_rms=rms(e[hold]), move_rms=rms(e[move]))
    lab_d = np.linalg.norm(tangential(rob["p_mesh"] - rob["p"], rob["n"]), axis=1)[w_r]
    bias = (ests["raw"] - truth)[hold].mean(0)
    row = dict(id=tr["id"], press_N=tr["press_N"], t0=float(tr["t0"]), n_records=int(len(tb)),
               record_rate_hz=round(float(1.0 / np.median(np.diff(tb))), 1), n_ticks_50hz=int(len(tk)),
               travel_mm=round(travel, 3), s_dir_P=np.round(s_dir, 4).tolist(), p0_P_mm=np.round(p0, 3).tolist(),
               hold_truth_along_roll_mm={lab: round(float((v - p0) @ s_dir), 3) for lab, v in hm.items()},
               hold_mesh_label_along_roll_mm={lab: round(float((v - p0) @ s_dir), 3) for lab, v in hm_mesh.items()},
               mesh_label_vs_smooth_mm=dict(rms=rms(lab_d), max=round(float(lab_d.max()), 4)),
               depth_hold_mm=round(float(np.median(rob["depth"][masks(tr_, pl, np.zeros(len(tr_)))[0]])), 3),
               turn_rate_deg_s=dict(mean_moving=round(float(np.degrees(np.mean(np.abs(rate[move])))), 3),
                                    p95=round(float(np.degrees(np.percentile(np.abs(rate[move]), 95))), 3)),
               contact_speed_mm_s=round(float(np.mean(np.abs(np.interp(tb[move] + 0.25, tr_, s_true)
                                                             - np.interp(tb[move] - 0.25, tr_, s_true)) / 0.5)), 4),
               truth_jitter_rms_mm=rms(np.linalg.norm(np.diff(truth[(tb >= pl[0][0]) & (tb <= pl[-1][1])], axis=0),
                                                      axis=1)),
               f_n_hold_N=round(float(-np.median(np.sum(f_true[hold] * nrm[hold], 1))), 3),
               gp_f_n_hold_N=round(float(-np.median(np.sum(gp_f[hold] * nrm[hold], 1))), 3),
               raw_bias_mm=np.round(bias, 4).tolist(),
               raw_bias_tangential_mm=round(float(np.linalg.norm(tangential(bias[None], n0[None])[0])), 4),
               gain_roll=dict(median_moving=round(float(np.median(gains["roll"][move])), 3) if move.any() else None,
                              max=round(float(gains["roll"].max()), 3)),
               metrics={s: {k: v for k, v in d.items() if not k.startswith("_")} for s, d in met.items()},
               vs_mesh_label=vs_mesh)
    trace = dict(t=tb - tr["t0"], truth=truth, truth_mesh=truth_mesh, normal=nrm, gp=gp_p, gp_f=gp_f, f_true=f_true,
                 theta=theta, rate=np.degrees(rate), raw=ests["raw"], kmin=ests["kmin"], roll=ests["roll"],
                 k_roll=gains["roll"], kmin_truthfed=fed["kmin"], roll_truthfed=fed["roll"], hold=hold, move=move,
                 t50=tk - tr["t0"], truth50=truth_k, kmin50=ests["kmin50"], roll50=ests["roll50"], hold50=hold_k,
                 move50=move_k, s_dir=s_dir, p0=p0,
                 plateaus=np.array([[a - tr["t0"], b - tr["t0"]] for a, b, _ in pl]))
    pooled = {s: (met[s]["_e_hold"], met[s]["_e_move"], met[s]["_step"]) for s in met}
    return row, trace, pooled


def pooled_summary(pool):
    """RMS and median over the pooled frames of every roll of one force."""
    out = {}
    for s, parts in pool.items():
        eh, em, st = (np.concatenate([p[k] for p in parts]) for k in range(3))
        out[s] = dict(hold_rms=rms(eh), hold_median=med(eh), move_rms=rms(em), move_median=med(em),
                      jitter_rms=rms(st), n_hold=int(len(eh)), n_move=int(len(em)))
    return out


# ------------------------------------------------------------------------------------------------ run
def summarize(rows, pools, keyf):
    """Pooled errors and median lags, travel and bias per group of rolls."""
    out = {}
    for g, pool in pools.items():
        rs = [r for r in rows if keyf(r) == g]
        out[g] = dict(n_rolls=len(rs), n_records=int(sum(r["n_records"] for r in rs)), pooled=pooled_summary(pool),
                      lag_ms_median={s: float(np.median([r["metrics"][s]["lag_ms"] for r in rs])) for s in rs[0]["metrics"]},
                      travel_mm_median=float(np.median([r["travel_mm"] for r in rs])),
                      contact_speed_mm_s_median=float(np.median([r["contact_speed_mm_s"] for r in rs])),
                      truth_jitter_rms_mm_median=float(np.median([r["truth_jitter_rms_mm"] for r in rs])),
                      mesh_label_vs_smooth_rms_mm_median=float(np.median([r["mesh_label_vs_smooth_mm"]["rms"] for r in rs])),
                      raw_bias_mm_mean=np.round(np.mean([r["raw_bias_mm"] for r in rs], 0), 4).tolist(),
                      raw_bias_tangential_mm_median=float(np.median([r["raw_bias_tangential_mm"] for r in rs])),
                      gain_roll_max=float(np.max([r["gain_roll"]["max"] for r in rs])),
                      vs_mesh_label={s: dict(hold_rms_median=float(np.median([r["vs_mesh_label"][s]["hold_rms"] for r in rs])),
                                             move_rms_median=float(np.median([r["vs_mesh_label"][s]["move_rms"] for r in rs])))
                                     for s in ("raw", "kmin", "roll")})
    return out


def run(seats, only=None, rows_path=ROWS, traces_path=TRACES):
    import ur5_probe_calibration_analysis as PA
    from sr2hand.control.surfaces import Cylinder, HalfEllipsoid, Posed
    t_start = time.time()
    rows, traces, pools, pools_seat, seat_info = [], {}, {}, {}, {}
    fit_summary = None
    for seat in seats:
        ses = load_session(seat)
        print(f"seat {seat}: session loaded ({time.time() - t_start:.0f} s)", flush=True)
        R_Pe, t_Pe, fs = fit_pad(ses.geo.pts, ses.geo.nrm)
        if fit_summary is None:
            fit_summary = fs
            print("pad fit:", json.dumps(fs), flush=True)
        pad = Posed(HalfEllipsoid(*(np.array(ELL_MM) * 1e-3)), pos=t_Pe * 1e-3, R=R_Pe)
        rod = Posed(Cylinder(ses.geo.R * 1e-3, ROD_HALF_LEN_M))
        bench = Bench(pad, rod, R_Pe[:, 1])
        m_gp, Htr, gp_info = train_gp(SEATS[seat]["tip"])
        gp_info["roll_holds_0p4s_means"] = roll_holds_check(m_gp, seat)
        print(f"seat {seat}: GP on {gp_info['n_holds']} holds ({gp_info['roll_action_holds_left_out']} roll-action holds "
              f"left out) in {gp_info['fit_s']} s, length {gp_info['length']:.2f}, noise {gp_info['noise']}; on the roll "
              f"holds' 0.4 s means {gp_info['roll_holds_0p4s_means']}", flush=True)
        ends = ses.ends
        acts = sorted((a for a in ses.acts.values() if a["kind"] == "roll" and a["id"] in ends
                       and ends[a["id"]].get("status") == "ok"), key=lambda a: ends[a["id"]]["t"])
        if only:
            acts = acts[:only]
        checks = []
        for k, a in enumerate(acts):
            t1 = time.time()
            tr = roll_trace(ses, a, R_Pe, t_Pe)
            checks += check_against_holds(tr, seat)
            row, trace, pooled = analyse_roll(tr, m_gp, bench)
            row = dict(seat=seat, tip=SEATS[seat]["tip"], **row)
            d_tr = np.min(np.linalg.norm(Htr["p"][None] - trace["truth"][trace["hold"]][::10, None], axis=2), axis=1)
            row["truth_to_nearest_training_point_mm"] = dict(median=round(float(np.median(d_tr)), 3),
                                                             max=round(float(d_tr.max()), 3))
            rows.append(row)
            traces[f"s{seat}_{k}"] = trace
            for s, v in pooled.items():
                pools.setdefault(f"{row['press_N']:g}N", {}).setdefault(s, []).append(v)
                pools_seat.setdefault(f"seat{seat}_{row['press_N']:g}N", {}).setdefault(s, []).append(v)
            mt = row["metrics"]
            print(f"  {a['id']} ({row['press_N']:g} N): travel {row['travel_mm']:.2f} mm at {row['contact_speed_mm_s']} "
                  f"mm/s, {row['n_records']} records; hold rms raw {mt['raw']['hold_rms']} kmin {mt['kmin']['hold_rms']} "
                  f"roll {mt['roll']['hold_rms']}; move rms raw {mt['raw']['move_rms']} kmin {mt['kmin']['move_rms']} "
                  f"roll {mt['roll']['move_rms']}; jitter raw {mt['raw']['jitter_rms']} kmin {mt['kmin']['jitter_rms']}; "
                  f"lag raw {mt['raw']['lag_ms']} kmin {mt['kmin']['lag_ms']} roll {mt['roll']['lag_ms']} ms (truth-fed "
                  f"{mt['kmin_truthfed']['lag_ms']} / {mt['roll_truthfed']['lag_ms']}); gain {row['gain_roll']}; "
                  f"{time.time() - t1:.0f} s", flush=True)
        ck = np.array(checks)
        seat_info[str(seat)] = dict(
            session=SEATS[seat]["session"], tip=SEATS[seat]["tip"], gp=gp_info, n_rolls=len(acts),
            plans=[Path(p).name for p in PA.PLANS], record_rate_hz=round(float(1.0 / np.median(np.diff(ses.d["t_b"]))), 1),
            label_vs_analysis_holds_mm=dict(n=len(ck), mesh_median=round(float(np.median(ck[:, 0])), 4),
                                            mesh_max=round(float(ck[:, 0].max()), 4),
                                            smooth_median=round(float(np.median(ck[:, 1])), 4),
                                            smooth_max=round(float(ck[:, 1].max()), 4)) if len(ck) else None)
        del ses
    summary = dict(
        what=__doc__.split("\n\n")[0], created=time.strftime("%Y-%m-%d %H:%M"), command="run --seats " +
        " ".join(map(str, seats)), seats=seat_info, pad_fit=fit_summary, sim_pads=sim_pads_check(),
        filter=dict(k_min=5.0, k_max=100.0, alpha_m_s=0.02, source="reference", settings=list(SETTINGS) +
                    ["kmin_truthfed", "roll_truthfed", "gp_unprojected"]),
        truth="the pad's point nearest the rod's axis on the fitted HalfEllipsoid (vs_mesh_label: on the plan's mesh, "
              "the analysis's own label)",
        controller_rate_hz=RATE_CTRL, skip_s=SKIP_S, pre_s=PRE_S, moving_deg_s=MOVING_DEG_S,
        joint_registration=str(JOINT.relative_to(ROOT)), n_rolls=len(rows),
        n_records=int(sum(r["n_records"] for r in rows)),
        by_force=summarize(rows, pools, lambda r: f"{r['press_N']:g}N"),
        by_seat_force=summarize(rows, pools_seat, lambda r: f"seat{r['seat']}_{r['press_N']:g}N"),
        seconds=round(time.time() - t_start, 1))
    Path(rows_path).parent.mkdir(parents=True, exist_ok=True)
    Path(rows_path).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1) + "\n")
    arr = {}
    for k, tr in traces.items():
        for f, v in tr.items():
            v = np.asarray(v)
            arr[f"{k}/{f}"] = v.astype(np.float32) if v.dtype == np.float64 else v
    arr["keys"] = np.array(list(traces))
    arr["ids"] = np.array([r["id"] for r in rows])
    arr["seat"] = np.array([r["seat"] for r in rows])
    arr["press_N"] = np.array([r["press_N"] for r in rows])
    np.savez_compressed(traces_path, **arr)
    print(json.dumps({g: v["pooled"] for g, v in summary["by_force"].items()}, indent=1))
    print("->", rows_path, traces_path, f"({time.time() - t_start:.0f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("run", "figure"))
    ap.add_argument("--seats", nargs="+", type=int, default=[2])
    ap.add_argument("--only", type=int, default=0, help="the first N rolls of each seat (a quick check)")
    ap.add_argument("--rows", default=str(ROWS))
    ap.add_argument("--traces", default=str(TRACES))
    ap.add_argument("--fig", default=str(FIG))
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.seats, a.only or None, a.rows, a.traces)
    else:
        figure(a.rows, a.traces, a.fig)


def figure(rows_path=ROWS, traces_path=TRACES, fig_path=FIG, roll_id="probe-face-776-r90-roll", seat=2):
    """From the saved rows and traces only: (a) one roll at 4 N, the contact's coordinate along the roll against time
    (truth, raw GP estimate, the k_min filter, the commanded-roll filter) over the rod's turn; (b) the RMS error along the
    pad at the holds and while moving, per setting and press force, pooled over the rolls."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = json.loads(Path(rows_path).read_text())
    Z = np.load(traces_path)
    rows = D["rows"]
    k = next((i for i, r in enumerate(rows) if r["id"] == roll_id and r["seat"] == seat),
             next(i for i, r in enumerate(rows) if r["press_N"] == 4.0))
    key = str(Z["keys"][k])
    g = lambda f: Z[f"{key}/{f}"].astype(float)        # noqa: E731
    t, s_dir, p0 = g("t"), g("s_dir"), g("p0")
    s = {f: (g(f) - p0) @ s_dir for f in ("truth", "raw", "kmin", "roll")}
    col = dict(truth="#000000", raw="#8c8c8c", kmin="#2a78d6", roll="#eb6834")
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "legend.fontsize": 11, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(8.2, 4.3), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0], height_ratios=[3.0, 1.0], hspace=0.08, wspace=0.42,
                          left=0.085, right=0.985, top=0.83, bottom=0.14)
    ax, axt, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    axt.sharex(ax)
    for a0, a1 in g("plateaus"):
        for a in (ax, axt):
            a.axvspan(a0, a1, color="#e6e6e6", lw=0, zorder=0)
    ax.plot(t[::3], s["raw"][::3], ".", ms=2.0, color=col["raw"], alpha=0.45, zorder=1, label="Raw GP estimate")
    ax.plot(t, s["kmin"], color=col["kmin"], lw=2.6, zorder=3, label=r"$k_{\min}$ filter")
    ax.plot(t, s["roll"], color=col["roll"], lw=1.3, zorder=4, label="Commanded-roll filter")
    ax.plot(t, s["truth"], color=col["truth"], lw=1.6, zorder=5, label="Truth")
    ax.set_ylim(-2.3, 2.3)
    ax.set_ylabel("Contact along the roll (mm)")
    ax.tick_params(labelbottom=False)
    axt.plot(t, g("theta"), color="#000000", lw=1.2)
    axt.set_ylim(-8, 8)
    axt.set_yticks([-6, 0, 6])
    axt.set_ylabel("Turn (deg)")
    axt.set_xlabel("Time from the press's level hold (s)")
    axt.set_xlim(t[0], t[-1])
    # (b) RMS error along the pad, pooled per press force
    B = D["summary"]["by_force"]
    names = [("raw", "Raw"), ("kmin", r"$k_{\min}$"), ("roll", "Roll")]
    yt, yl, y = [], [], 0.0
    for force in ("1.5N", "4N"):
        if force not in B:
            continue
        P = B[force]["pooled"]
        for nm, lab in names[::-1]:
            axb.plot([P[nm]["hold_rms"]], [y + 0.16], "o", ms=8, color=col[nm], zorder=3)
            axb.plot([P[nm]["move_rms"]], [y - 0.16], "o", ms=8, mfc="white", mec=col[nm], mew=2.0, zorder=3)
            yt.append(y)
            yl.append(lab)
            y += 1.0
        axb.text(0.0, y - 0.35, f"Press {force[:-1]} N ({B[force]['n_rolls']} rolls)", fontsize=11, ha="left",
                 va="bottom", transform=axb.get_yaxis_transform())
        y += 1.0
    axb.set_yticks(yt)
    axb.set_yticklabels(yl)
    axb.set_ylim(-0.7, y - 0.5)
    xmax = max(max(B[f]["pooled"][n][m] for n, _ in names for m in ("hold_rms", "move_rms")) for f in B)
    axb.set_xlim(0.0, 0.1 * math.ceil(xmax * 10 + 0.5))
    axb.set_xlabel("RMS error along the pad (mm)")
    axb.plot([], [], "o", ms=8, color="#555555", label="At holds")
    axb.plot([], [], "o", ms=8, mfc="white", mec="#555555", mew=2.0, label="Moving")
    axb.legend(loc="lower left", bbox_to_anchor=(-0.02, 0.99), ncol=2, frameon=False, handletextpad=0.1,
               columnspacing=0.8, borderaxespad=0.0)
    for a in (ax, axt, axb):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="x" if a is axb else "y", color="#e0e0e0", lw=0.8)
        a.set_axisbelow(True)
    h, l_ = ax.get_legend_handles_labels()
    order = [3, 0, 1, 2]
    fig.legend([h[i] for i in order], [l_[i] for i in order], loc="upper left", ncol=4, frameon=False,
               bbox_to_anchor=(0.06, 1.0), markerscale=4, handlelength=1.6, columnspacing=1.2)
    fig.text(0.012, 0.85, "a", fontsize=13, fontweight="bold", va="top")
    fig.text(0.60, 0.85, "b", fontsize=13, fontweight="bold", va="top")
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=100)
    plt.close(fig)
    print("->", fig_path, f"({key}: {rows[k]['id']}, seat {rows[k]['seat']}, {rows[k]['press_N']:g} N)")


if __name__ == "__main__":
    main()
