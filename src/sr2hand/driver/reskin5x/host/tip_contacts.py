#!/usr/bin/env python3
"""Contact estimates of the SR2 hand's three magnetic fingertips, for HOM on the bench (owner, 2026-11-16: "lets make
sure the next time we run trials we have the fingertip sensing going. ideally with pose tracking, closed loop control
too. that can be churned away on overnight to be ready for evaluation").

The boards. The hand's v3 fingertips (four N52 magnets in pad 05R4) sit on ReSkin 5X boards behind one I2C mux: v3 ID 2
on the middle finger (mux channel 0), v3 ID 3 on the index (channel 1) and v3 ID 1 on the thumb (channel 3)
(docs/experiments/20261104-sr2_hand_crosstalk/README.md, "The setup"). The splitter reskin5x_mux.py reads the mux, logs
every record to logs/<date>-three_finger_mux/<start>-mux.bin and writes each board's records to a pseudo-terminal
linked at /tmp/reskin5x_mux/ch<channel>. The live feed here (Outlets) reads those three pseudo-terminals. They get each
batch of records as the splitter reads it from the mux, about every 50 ms. The .bin log reaches the disk in 8 KB
blocks, about every 0.2 s; replays read it (MuxLog). No other program may read the pseudo-terminals during a trial,
or the two readers split the records between them.

The estimate, per board, is the viewer's (wrench_viewer.App with --board-gp, mode "board gp"). Every reading is divided
by its chip's gain at the chip's temperature over the last 3 s (fingertip_wrench_model.thermal_correct). The change d is
the mean of the last 50 ms less the rest reading. The physical model (fingertip_wrench_model.SurfaceEstimator on the FE
surface basis of pad 05R4, the tip's magnets as fitted at rest on 2026-11-02) finds the contact point and force. A
contact counts when its explained chi-square exceeds 25 and its press reaches 0.25 N, or 0.15 N once it counts. The
pooled board GP of the UR5 calibration then gives the point and the force, weighted by w against the physical model's;
w is one minus the GP's posterior variance of its standardized outputs, 1 at a calibration hold and 0 far from all of
them. That GP was fitted on the presses of v3 IDs 1-3 on the UR5 and of IDs 3-6 on the inverted rig. The viewer's
re-zero after 3 s at rest is left out: within a trial it would take a held press into the rest.

The own-pose term (OwnPose, on unless own_pose=False) is not the viewer's. A board's reading changes with its own
finger's pose by up to 166 uT, 5.8 N to the viewer, with no contact (docs/experiments/20261104-sr2_hand_crosstalk). A GP
on the finger's three joint angles, fitted to the crosstalk sweep of 2026-11-04/05, predicts that change; the
prediction's difference between the rest's pose and the present pose is taken off the reading's change before the
estimate. In the press of 2026-11-16 13:19 the middle finger's board changed by 28-31 uT during the approach, 10 mm from
the cylinder, and the term predicted 35-37 uT of it.

The rest is the mean reading of each board over --rest-s with the fingers still at the trial's open pose, before the
close. With refit, a new fit of the magnets to that rest (fingertip_wrench_model.calibrate_frames_fixed, the chip
frames of the saved calibration kept) competes with the saved model. The saved model is kept when, with a field from
outside the pad fitted to the difference, it explains the rest at least as well (the viewer's rule, wrench_viewer.App
.calibrate).

Frames. The estimate is in the pad frame of the fingertip calibrations (tpu_tip_design: x toward the palmar face, z from
the tip's mount end to its tip end, the face at x = 8.5 mm and the pad's centre at z = 10.86 mm). The hand's tip frame
<f>_tip (tool_chain.SR2Hand: origin on the finger's axis at the pad's centre plane, z toward the PIP, the face at
x = face_sign 10.55 mm) holds a pad-frame point at x_tip = s (x + 2.05), y_tip = -s y, z_tip = 10.86 - z, with s the
finger's face_sign (T_tip_pad; scripts/fingertip/sr2_crosstalk_hand.py T_tip_P). The tool chain at the measured joints
and the plan's mounts (meta.hom.d6_P_mm) then places it in the palm frame P. The link's StateMsg.cop takes, per finger
(thumb, index, middle), x, y, z in P (mm) and the normal force (N), NaN without a contact.

    /usr/bin/python3 tip_contacts.py logger                   the splitter's state; --start starts it when it is not running
    /usr/bin/python3 tip_contacts.py rest --plan PLAN         at the open pose: the rest, the fits, then --seconds of live
                                                              estimates in P (a check before a trial; nothing moves)
    .venv/bin/python tip_contacts.py replay --mux LOG.bin --corvus TRIAL_corvus.jsonl --plan PLAN --object cylinder
                                                              a recorded trial through the estimator, period by period
"""
from __future__ import annotations

import argparse
import json
import math
import os
import select
import subprocess
import sys
import termios
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import fingertip_wrench_model as W  # noqa: E402
from sr2hand.robot.tool_chain import SR2Hand  # noqa: E402

FINGERS = ("thumb", "index", "middle")
CHANNEL = {"middle": 0, "index": 1, "thumb": 3}          # mux channel of each finger's board
TIP_ID = {"thumb": 1, "index": 3, "middle": 2}           # the v3 tips on the hand since 2026-11-04
DATA = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data"
CALIB = DATA / "20261105-viewer_inputs/calib"            # 20261102-v3_id_<n>-calib.json: each tip's magnets at rest
BOARD_GP = DATA / "20261105-board_gp_v3_id1_id3_rig_id3_id6.npz"
CROSSTALK_ROWS = ROOT / ("docs/experiments/20261104-sr2_hand_crosstalk/data/"
                         "20261105-sr2_crosstalk_hand_20261104_20261105_rows.npz")
OWN_POSE = ROOT / "docs/experiments/20261115-hom_bench_trials/data/20261117-tip_contacts/20261117-own_pose_gp.npz"
CASE, DX_RANGE = "four_05R4", (0.43, 1.03)               # the v3 pads; the board offset's bound (wrench_viewer.DX_RANGE)
LINKS = Path("/tmp/reskin5x_mux")
SPLITTER = HERE / "reskin5x_mux.py"
WIN_S, KEEP_S = 0.05, 3.0                 # s: the averaged window, and the span of the temperatures (wrench_viewer)
THRESHOLD, MIN_FORCE, HYSTERESIS, FLOOR_UT = 25.0, 0.25, 0.6, 2.0          # the viewer's defaults
MUX_RATE = 160.6                          # records/s of the three-board mux (11-16)
FRAME = 4 * 4 * 5 + 2                     # one board's record on its pseudo-terminal: 5 x (t x y z) f4 and \r\n
STREAM_DTYPE = np.dtype([("t", "<f8"), ("r", "<f4", (20,))])     # wrench_viewer.read_stream's format


def valid(r):
    """Which records (n, 5, 4) have every chip reading (wrench_viewer.App.records: no zeroed chip, finite and in range,
    die temperatures between -40 and 125 C)."""
    with np.errstate(invalid="ignore", over="ignore"):
        return (~np.all(r[:, :, 1:] == 0, axis=2).any(axis=1) & np.isfinite(r).all(axis=(1, 2))
                & (np.abs(r) < 1e5).all(axis=(1, 2)) & (r[:, :, 0] > -40).all(axis=1) & (r[:, :, 0] < 125).all(axis=1))


def T_tip_pad(finger, hand=None):
    """The pad frame of the fingertip calibrations in the hand's tip frame <f>_tip (4 x 4, mm): x_tip = s (x + 2.05),
    y_tip = -s y, z_tip = Z_C - z, s the finger's face_sign; the pad's apex (8.5, 0, 10.86) goes to (s 10.55, 0, 0)."""
    g = (hand or SR2Hand.load()).fingers[finger]
    s = float(g.face_sign)
    T = np.eye(4)
    T[:3, :3] = np.diag([s, -s, -1.0])
    T[:3, 3] = [s * (g.reach_mm - W.TD.X_FACE), 0.0, W.TD.Z_C]
    return T


def sim_deg(joints_servo):
    """Servo degrees (3, 3), rows thumb, index, middle and columns aa, fe1, fe2 (StateMsg.joints) -> simulator degrees
    {finger: (yaw, mcp, pip)} (manta_hand.plan: sim = servo / JOINT_SIGN)."""
    host = str(ROOT / "src/sr2hand/driver/manta/host")
    if host not in sys.path:
        sys.path.insert(0, host)
    from manta_hand import plan as P
    J = np.asarray(joints_servo, float).reshape(3, 3)
    col = {"aa": 0, "fe1": 1, "fe2": 2}
    return {f: tuple(J[k, col[P.SIM_JOINT_TO_SERVO[j]]] / P.JOINT_SIGN[(f, j)] for j in ("yaw", "mcp", "pip"))
            for k, f in enumerate(FINGERS)}


class BoardGP:
    """The pooled GP from a board's change to the contact point (pad frame, mm) and force (N), as saved by
    scripts/ur5/ur5_tip_calibration_board_gp.py: wrench_viewer.BoardGP without the viewer's web server (the
    workstation's venv has no aiohttp)."""

    def __init__(self, path=BOARD_GP, m=256):
        z = np.load(path)
        self.mx, self.sx, self.my, self.sy, self.Z, self.a = (z[k] for k in ("mx", "sx", "my", "sy", "Z", "a"))
        self.ell, self.nz = float(z["ell"]), float(z["nz"])
        self.m = min(m, len(self.Z) - 1)
        self.sq = (self.Z ** 2).sum(1)
        self.n = int(z["n_holds"])

    def predict(self, d):
        """Contact point (mm), force (N) and w, given the m holds of largest kernel weight."""
        q = (np.asarray(d, float) - self.mx) / self.sx
        k = np.exp(-0.5 * ((self.Z - q) ** 2).sum(1) / self.ell ** 2)
        y = k @ self.a * self.sy + self.my
        s = np.argpartition(-k, self.m)[:self.m]
        Ks = np.exp(-0.5 * np.maximum(self.sq[s][:, None] + self.sq[s][None] - 2 * self.Z[s] @ self.Z[s].T, 0)
                    / self.ell ** 2)
        w = float(np.clip(k[s] @ cho_solve(cho_factor(Ks + self.nz ** 2 * np.eye(self.m), lower=True), k[s]), 0.0, 1.0))
        return y[:3], y[3:], w


class OwnPose:
    """Each board's change with its own finger's pose: a GP from the finger's yaw, MCP and PIP (simulator degrees) to
    the 15 components (uT), fitted by fit_own_pose on the crosstalk sweep of 2026-11-04/05 (the change from the zero
    pose of each layout; scripts/fingertip/sr2_crosstalk_hand.py own_models, input "own_angles"). There the five-fold
    error was 5.4-6.1 uT median against a change of 66-69 uT."""

    def __init__(self, path=OWN_POSE):
        z = np.load(path)
        self.m = {f: {k: z[f"{f}_{k}"] for k in ("mx", "sx", "my", "sy", "Z", "a", "ell")} for f in FINGERS}

    def predict(self, f, q):
        m = self.m[f]
        Zq = (np.asarray(q, float) - m["mx"]) / m["sx"]
        k = np.exp(-0.5 * ((m["Z"] - Zq) ** 2).sum(1) / float(m["ell"]) ** 2)
        return k @ m["a"] * m["sy"] + m["my"]


def fit_own_pose(out=OWN_POSE, rows=CROSSTALK_ROWS):
    """The OwnPose GPs from the crosstalk sweep's rows (the workstation: ur5_tip_calibration_board_gp.fit)."""
    import ur5_tip_calibration_board_gp as BG
    z = np.load(rows)
    q = z["q"]
    save, info = {}, {}
    for k, f in enumerate(FINGERS):
        X, Y = q[z[f"{f}_rows"]][:, 3 * k:3 * k + 3], z[f"{f}_Y"]
        m = BG.fit(X, Y, np.full(3, 5.0))
        for key in ("mx", "sx", "my", "sy", "Z", "a"):
            save[f"{f}_{key}"] = np.asarray(m[key])
        save[f"{f}_ell"], save[f"{f}_nz"] = np.float64(m["ell"]), np.float64(m["nz"])
        info[f] = dict(n=len(X), ell=round(m["ell"], 3), nz=m["nz"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **save, source=str(Path(rows).relative_to(ROOT)))
    return info


class Tip:
    """One fingertip's estimator: its board's model, its rest and the decision to count a contact."""

    def __init__(self, finger, basis, row, gp=None, kappa=None, calib=None, floor=FLOOR_UT, threshold=THRESHOLD,
                 min_force=MIN_FORCE, own=None):
        self.finger, self.basis, self.row, self.gp, self.kappa = finger, basis, row, gp, kappa
        self.own, self.q_rest = own, None
        self.floor, self.threshold, self.min_force = floor, threshold, min_force
        self.calib_path = Path(calib or CALIB / f"20261102-v3_id_{TIP_ID[finger]}-calib.json")
        self.cal0 = W.Calib.load(self.calib_path)
        self.use(self.cal0)
        self.saved = (self.cal, self.model, self.est, self.tree)
        self.base, self.shown, self.rest_info = None, False, None

    def use(self, cal):
        self.cal = cal
        self.model = W.Model(cal, self.row)
        self.est = W.SurfaceEstimator(self.model, self.basis, floor_uT=self.floor)
        self.est.set_noise_scale(1 / np.sqrt(WIN_S * 140))
        self.tree = cKDTree(self.est.C)

    def fields(self, r, r_T):
        """Fields (n, 15) uT of records r (n, 5, 4), each chip divided by its gain at its mean temperature over r_T."""
        b = np.asarray(r, float)[:, :, 1:]
        if self.kappa is not None:
            b = W.thermal_correct(b, np.asarray(r_T, float)[:, :, 0].mean(0), self.kappa)
        return b.reshape(len(r), 15)

    def kept_fit(self, model, b_rest):
        """rms (mT) of model's rest against b_rest with a field from outside the pad fitted to the difference
        (wrench_viewer.App.kept_fit)."""
        r = b_rest.ravel() - np.ravel(model.rest_uT(g_uT=(0.0, 0.0, 0.0)))
        A = W.far_field_basis(model)
        r = r - A @ np.linalg.lstsq(A, r, rcond=None)[0]
        return float(np.sqrt(np.mean(r ** 2))) / 1e3

    def set_rest(self, r, refit=True, q=None):
        """The rest from records r (n, 5, 4) at the open pose, the finger's joints there q (yaw, mcp, pip, simulator
        degrees; for the own-pose term); with refit the viewer's choice between the saved model and a new fit. Returns
        what was done."""
        r = np.asarray(r, float)
        r = r[valid(r)]
        if len(r) < 20:
            raise RuntimeError(f"{self.finger}: {len(r)} valid records for the rest")
        b_rest = self.fields(r, r).mean(0)
        self.cal, self.model, self.est, self.tree = self.saved
        self.q_rest = None if q is None or not np.all(np.isfinite(q)) else np.asarray(q, float)
        info = dict(records=len(r), level_mT=round(float(np.linalg.norm(b_rest)) / 1e3, 3),
                    calib=self.calib_path.name, T_C=np.round(r[:, :, 0].mean(0), 2).tolist(),
                    q_rest=None if self.q_rest is None else np.round(self.q_rest, 2).tolist())
        info["saved_fit_mT"] = round(self.kept_fit(self.model, b_rest), 4)
        if refit:
            t0 = time.time()
            cal = W.calibrate_frames_fixed(b_rest.reshape(5, 3), self.cal0, case=CASE, dx_range=DX_RANGE)
            info.update(new_fit_mT=round(cal.rms_mT, 4), new_k=round(cal.k, 3), refit_s=round(time.time() - t0, 2))
            if info["saved_fit_mT"] > cal.rms_mT:
                self.use(cal)
                info["model"] = "new fit"
            else:
                info["model"] = "saved, re-zeroed"
        else:
            info["model"] = "saved, re-zeroed"
        self.base, self.shown, self.rest_info = b_rest, False, info
        return info

    def estimate(self, t, r, q=None):
        """The contact from records (t, r) up to the estimate's time, the last KEEP_S of them, with the finger's joints q
        then (simulator degrees): dict with contact, and with one p, n (pad frame, mm), f (N, pad frame), f_n, w (the
        GP's weight; None without the GP). With the own-pose model and both poses known, the change the model predicts
        between the rest's pose and q is taken off the reading's change first."""
        r = np.asarray(r, float)
        ok = valid(r)
        t, r = np.asarray(t, float)[ok], r[ok]
        if len(t) < 3 or self.base is None:
            return dict(contact=False, why="no rest" if self.base is None else "no records")
        sel = t > t[-1] - WIN_S
        d = self.fields(r[sel], r).mean(0) - self.base
        own = None
        if self.own is not None and self.q_rest is not None and q is not None and np.all(np.isfinite(q)):
            own = self.own.predict(self.finger, q) - self.own.predict(self.finger, self.q_rest)
            d = d - own
        e = self.est.estimate(d)
        lim = self.min_force * (HYSTERESIS if self.shown else 1.0)
        contact = bool(e["contact"] and e["e"] > self.threshold and e["f_n"] >= lim)
        self.shown = contact
        out = dict(t=float(t[-1]), contact=contact, d_uT=round(float(np.linalg.norm(d)), 1), e=round(e.get("e", 0.0), 1),
                   n_win=int(sel.sum()), own_uT=None if own is None else round(float(np.linalg.norm(own)), 1))
        if not contact:
            return out
        p, f, w = np.asarray(e["p"], float), np.asarray(e["f"], float), None
        if self.gp is not None:
            pg, fg, w = self.gp.predict(d)
            p, f = w * pg + (1 - w) * p, w * fg + (1 - w) * f
        k = int(self.tree.query(p)[1])
        n = self.est.Cn[k]
        out.update(p=self.est.C[k].tolist(), n=n.tolist(), f=f.tolist(), f_n=float(-f @ n),
                   p_phys=np.round(e["p"], 3).tolist(), f_n_phys=round(e["f_n"], 3), w=None if w is None else round(w, 3))
        return out


class TipContacts:
    """The three fingertips on the hand: rests and estimates per board from a feed (Outlets or MuxLog), and the
    contacts in P at the measured joints."""

    def __init__(self, feed, d6_P_mm, gp=True, own_pose=True, calib=None, hand=None):
        basis, row, kappa = W.load_basis(W.basis_path(CASE)), W.fe_row(CASE), W.load_thermal()
        board_gp = BoardGP() if gp else None
        own = OwnPose() if own_pose else None
        self.feed, self.hand = feed, hand or SR2Hand.load()
        self.tips = {f: Tip(f, basis, row, board_gp, kappa, (calib or {}).get(f), own=own) for f in FINGERS}
        d6 = np.asarray(d6_P_mm, float)
        self.mounts = {"thumb": d6[0:2], "index": d6[2:4], "middle": d6[4:6]}
        self.T_tp = {f: T_tip_pad(f, self.hand) for f in FINGERS}
        self.lock, self.latest, self.worker, self.compute_ms = threading.Lock(), None, None, deque(maxlen=2000)

    @staticmethod
    def _q(joints_servo):
        if joints_servo is None or not np.all(np.isfinite(np.asarray(joints_servo, float))):
            return {}
        return sim_deg(joints_servo)

    def rest(self, t_end, seconds, refit=True, joints_servo=None):
        """Each board's rest over [t_end - seconds, t_end] (s, corvus's clock), the joints there (servo degrees, (3, 3))
        for the own-pose term."""
        q, out = self._q(joints_servo), {}
        for f, tip in self.tips.items():
            _t, r = self.feed.window(f, t_end, seconds)
            out[f] = tip.set_rest(r, refit, q.get(f))
        return out

    def rest_state(self):
        """Each board's rest, its pose and its model (the new fit's parameters, or None for the saved one), as JSON:
        load_rest_state takes it in another process (pinch_pickup.py's lift and slip after its grasp)."""
        from dataclasses import asdict
        return {f: dict(base=None if tip.base is None else np.asarray(tip.base).tolist(),
                        q_rest=None if tip.q_rest is None else np.asarray(tip.q_rest).tolist(),
                        cal=None if tip.cal is tip.cal0 else asdict(tip.cal), info=tip.rest_info)
                for f, tip in self.tips.items()}

    def load_rest_state(self, state):
        for f, s in state.items():
            tip = self.tips[f]
            tip.cal, tip.model, tip.est, tip.tree = tip.saved
            if s.get("cal"):
                c = dict(s["cal"])
                for key in ("poles", "perm", "g_uT"):
                    c[key] = tuple(c[key])
                tip.use(W.Calib(**c))
            tip.base = None if s.get("base") is None else np.asarray(s["base"], float)
            tip.q_rest = None if s.get("q_rest") is None else np.asarray(s["q_rest"], float)
            tip.rest_info, tip.shown = s.get("info"), False

    def estimate(self, t_end=None, joints_servo=None):
        """{finger: Tip.estimate} from each board's records up to t_end (the newest without), at the joints then."""
        q = self._q(joints_servo)
        return {f: tip.estimate(*self.feed.window(f, t_end, KEEP_S), q.get(f)) for f, tip in self.tips.items()}

    def to_P(self, est, joints_servo):
        """(cop (12,) for StateMsg.cop, {finger: record}) of the estimates est at the measured joints (servo degrees,
        (3, 3)): each contact's point (mm), normal and force (N) in P; NaN and None for a finger without one. Without
        joints a contact keeps its pad-frame values, p_P None and NaN in cop."""
        cop, rec = np.full(12, np.nan), {}
        q = self._q(joints_servo) or None
        for k, f in enumerate(FINGERS):
            e = est.get(f) or {}
            if not e.get("contact"):
                rec[f] = None
                continue
            if q is None:
                rec[f] = dict(p_P=None, f_n=round(e["f_n"], 4), p_pad=np.round(e["p"], 3).tolist(), w=e.get("w"),
                              e=e["e"], t=e["t"])
                continue
            c, _ = self.hand.box(f)
            m = self.mounts[f]
            p_tip = (self.T_tp[f] @ np.r_[e["p"], 1.0])[:3]
            T = self.hand.T_P_contact(f, m[0] - c[0], m[1] - c[1], *(math.radians(v) for v in q[f]),
                                      contact_tip_mm=p_tip)
            R = T[:3, :3] @ self.T_tp[f][:3, :3]                  # pad frame -> P
            cop[4 * k:4 * k + 3], cop[4 * k + 3] = T[:3, 3], e["f_n"]
            rec[f] = dict(p_P=np.round(T[:3, 3], 3).tolist(), n_P=np.round(R @ e["n"], 4).tolist(),
                          f_P=np.round(R @ e["f"], 4).tolist(), f_n=round(e["f_n"], 4), p_pad=np.round(e["p"], 3).tolist(),
                          w=e.get("w"), e=e["e"], t=e["t"])
        return cop, rec

    # ------------------------------------------------------------------------------------------- live
    def start(self, period=0.02, joints=None):
        """Estimate in a thread of its own, every period at most, at the joints that joints() gives (servo degrees,
        (3, 3), or None); get() gives the newest result."""
        self.stop_evt = threading.Event()

        def loop():
            while not self.stop_evt.is_set():
                t0 = time.perf_counter()
                try:
                    est = self.estimate(None, None if joints is None else joints())
                except Exception as e:                    # noqa: BLE001  the record says so; the loop goes on
                    est = {"error": repr(e)[:200]}
                dt = time.perf_counter() - t0
                with self.lock:
                    self.latest = (time.time(), est)
                    self.compute_ms.append(1e3 * dt)
                time.sleep(max(0.002, period - dt))
        self.worker = threading.Thread(target=loop, daemon=True, name="tips")
        self.worker.start()

    def get(self):
        """(time it was made, estimates) of the newest estimate, or (nan, {})."""
        with self.lock:
            return self.latest if self.latest is not None else (float("nan"), {})

    def stop(self):
        if self.worker is not None:
            self.stop_evt.set()
            self.worker.join(timeout=2.0)


# ------------------------------------------------------------------------------------------------ feeds

class Outlets(threading.Thread):
    """The splitter's three pseudo-terminals, read in one thread. Each record is stamped with corvus's clock when it is
    read, the earlier records of one read spread back at the mux's MUX_RATE, kept for `keep` s, and, after record(),
    appended to a stream file per finger in wrench_viewer's format (read_stream reads it)."""

    def __init__(self, links=LINKS, keep=10.0):
        super().__init__(daemon=True, name="outlets")
        self.keep, self.lock, self.stop_evt = keep, threading.Lock(), threading.Event()
        self.fds, self.paths = {}, {}
        for f in FINGERS:
            path = Path(links) / f"ch{CHANNEL[f]}"
            fd = os.open(str(path), os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY)
            termios.tcflush(fd, termios.TCIFLUSH)         # what waited unread is old
            self.fds[fd], self.paths[f] = f, str(path)
        self.buf = {f: bytearray() for f in FINGERS}
        self.synced = dict.fromkeys(FINGERS, False)
        self.t = {f: deque() for f in FINGERS}
        self.r = {f: deque() for f in FINGERS}
        self.count, self.files, self.error = dict.fromkeys(FINGERS, 0), {}, None

    def _parse(self, f, chunk):
        b = self.buf[f]
        b += chunk
        recs = []
        while len(b) >= 2 * FRAME:
            if not self.synced[f]:
                if b[FRAME - 2:FRAME] == b"\r\n" and b[2 * FRAME - 2:2 * FRAME] == b"\r\n":
                    self.synced[f] = True
                else:
                    del b[:1]
                    continue
            if b[FRAME - 2:FRAME] != b"\r\n":
                self.synced[f] = False
                continue
            recs.append(np.frombuffer(bytes(b[:FRAME - 2]), "<f4").reshape(5, 4))
            del b[:FRAME]
        return recs

    def run(self):
        while not self.stop_evt.is_set():
            try:
                ready, _, _ = select.select(list(self.fds), [], [], 0.1)
            except (OSError, ValueError) as e:
                self.error = repr(e)
                return
            now = time.time()
            for fd in ready:
                f = self.fds[fd]
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    continue
                except OSError as e:
                    self.error = f"{f}: {e!r}"
                    continue
                recs = self._parse(f, chunk)
                if not recs:
                    continue
                ts = now - (len(recs) - 1 - np.arange(len(recs))) / MUX_RATE
                with self.lock:
                    self.t[f].extend(ts)
                    self.r[f].extend(recs)
                    self.count[f] += len(recs)
                    while self.t[f] and now - self.t[f][0] > self.keep:
                        self.t[f].popleft()
                        self.r[f].popleft()
                    fh = self.files.get(f)
                    if fh is not None:
                        a = np.zeros(len(recs), STREAM_DTYPE)
                        a["t"], a["r"] = ts, np.array([x.ravel() for x in recs], np.float32)
                        fh.write(a.tobytes())

    def window(self, f, t_end=None, span=KEEP_S):
        """(t, r (n, 5, 4)) of finger f's records in (t_end - span, t_end] (the newest without t_end)."""
        with self.lock:
            t = np.array(self.t[f])
            r = np.array(self.r[f]) if self.r[f] else np.zeros((0, 5, 4))
        if not len(t):
            return t, r
        te = t[-1] if t_end is None else t_end
        sel = (t > te - span) & (t <= te)
        return t[sel], r[sel].astype(float)

    def rates(self, seconds=1.0):
        """{finger: valid records a second over the last `seconds`}."""
        now, out = time.time(), {}
        for f in FINGERS:
            t, r = self.window(f, now, seconds)
            out[f] = float(valid(r).sum()) / seconds if len(t) else 0.0
        return out

    def record(self, stem, since=None):
        """From now on every record goes to <stem>_<finger>.bin; with since (s, corvus's clock) the kept records from
        then are written first. Returns the paths."""
        paths = {}
        with self.lock:
            for f in FINGERS:
                paths[f] = f"{stem}_{f}.bin"
                fh = open(paths[f], "wb")
                if since is not None and self.t[f]:
                    t = np.array(self.t[f])
                    sel = t >= since
                    if sel.any():
                        a = np.zeros(int(sel.sum()), STREAM_DTYPE)
                        a["t"] = t[sel]
                        a["r"] = np.array([x.ravel() for x, s in zip(self.r[f], sel) if s], np.float32)
                        fh.write(a.tobytes())
                self.files[f] = fh
        return paths

    def close(self):
        self.stop_evt.set()
        self.join(timeout=1.0)
        with self.lock:
            for fh in self.files.values():
                fh.close()
            self.files = {}
        for fd in list(self.fds):
            os.close(fd)
        self.fds = {}


class MuxLog:
    """A splitter's log (reskin5x_mux.py's .bin: host time f8, then per board in channel order 5 x (t x y z) f4), read
    between t_from and t_to (s, corvus's clock) through a memory map, with Outlets' window()."""

    def __init__(self, path, t_from=None, t_to=None, channels=(0, 1, 3)):
        nb = len(channels)
        dt = np.dtype([("t", "<f8"), ("r", "<f4", (nb, 5, 4))])
        mm = np.memmap(path, dt, mode="r")
        t = np.asarray(mm["t"])
        sel = np.ones(len(t), bool)
        if t_from is not None:
            sel &= t >= t_from
        if t_to is not None:
            sel &= t <= t_to
        self.path, self.t = str(path), t[sel]
        r = np.asarray(mm["r"][sel])
        self.r = {f: r[:, list(channels).index(CHANNEL[f])] for f in FINGERS}
        del mm

    def window(self, f, t_end=None, span=KEEP_S):
        te = self.t[-1] if t_end is None else t_end
        sel = (self.t > te - span) & (self.t <= te)
        return self.t[sel], self.r[f][sel].astype(float)


def read_stream(path):
    """(t (n,), r (n, 5, 4)) of a stream file written by Outlets.record (wrench_viewer.read_stream's format)."""
    a = np.fromfile(path, STREAM_DTYPE)
    return a["t"], a["r"].reshape(-1, 5, 4).astype(float)


class Streams:
    """A trial's stream files (Outlets.record) as a feed with window()."""

    def __init__(self, paths):
        self.t, self.rr = {}, {}
        for f, p in paths.items():
            self.t[f], self.rr[f] = read_stream(p)

    def window(self, f, t_end=None, span=KEEP_S):
        t = self.t[f]
        te = t[-1] if t_end is None else t_end
        sel = (t > te - span) & (t <= te)
        return t[sel], self.rr[f][sel]


# ------------------------------------------------------------------------------------------------ the splitter

def splitter_pids():
    """PIDs of running reskin5x_mux.py processes (from /proc, so no pattern of ours can match itself)."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(a.endswith(b"reskin5x_mux.py") for a in argv[1:3]):
            out.append(int(d.name))
    return out


def outlets_up(links=LINKS):
    """Whether every finger's link exists and points at an existing pseudo-terminal."""
    for f in FINGERS:
        p = Path(links) / f"ch{CHANNEL[f]}"
        if not p.is_symlink() or not Path(os.path.realpath(p)).exists():
            return False
    return True


def ensure_splitter(start=True, wait_s=10.0, links=LINKS):
    """(pids, how) of the splitter: attached to a running one, or started detached (its own session, its output in
    logs/<date>-three_finger_mux/<date>-splitter.log) and waited for until its links are up. Raises RuntimeError when
    none runs and start is False, or when a started one brings up no links within wait_s."""
    pids = splitter_pids()
    if pids and outlets_up(links):
        return pids, "attached"
    if not start:
        raise RuntimeError("the splitter reskin5x_mux.py is not running (tip_contacts.py logger --start starts it)")
    if pids:
        raise RuntimeError(f"reskin5x_mux.py runs (pids {pids}) without its links under {links}")
    log_dir = ROOT / "logs" / time.strftime("%Y%m%d-three_finger_mux")
    log_dir.mkdir(parents=True, exist_ok=True)
    out = open(log_dir / time.strftime("%Y%m%d-splitter.log"), "a")
    subprocess.Popen([sys.executable, str(SPLITTER), "--links", str(links), "--stat-s", "10"], cwd=str(ROOT),
                     stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        time.sleep(0.5)
        if splitter_pids() and outlets_up(links):
            time.sleep(1.0)                               # the first records
            return splitter_pids(), "started"
    raise RuntimeError(f"reskin5x_mux.py started but no links under {links} after {wait_s:g} s: see {out.name}")


def check_streams(outlets, seconds=1.0, min_rate=100.0):
    """None when every board streams at min_rate valid records a second or more, else the reason."""
    time.sleep(seconds)
    rates = outlets.rates(seconds)
    low = {f: round(v, 1) for f, v in rates.items() if v < min_rate}
    if low:
        return f"boards below {min_rate:g} valid records/s: {low} (all: " + \
            ", ".join(f"{f} {v:.0f}" for f, v in rates.items()) + ")"
    return None


# ------------------------------------------------------------------------------------------------ commands

def load_plan_d6(path):
    return json.loads(Path(path).read_text())["meta"]["hom"]["d6_P_mm"]


def cmd_logger(a):
    pids = splitter_pids()
    print(f"splitter pids {pids}; links {'up' if outlets_up() else 'down'} under {LINKS}", flush=True)
    if a.start:
        pids, how = ensure_splitter(True)
        print(f"splitter {how}: pids {pids}", flush=True)
    if outlets_up():
        o = Outlets()
        o.start()
        why = check_streams(o, 2.0)
        print("streams: " + (why or "all three boards at " + ", ".join(f"{f} {v:.0f}/s" for f, v in o.rates(2.0).items())),
              flush=True)
        o.close()
    return 0


def cmd_rest(a):
    """The rest at the open pose, then live estimates in P at the station's joints (no motion)."""
    if a.links:                                           # a test feed (play), not the splitter
        o = Outlets(a.links)
    else:
        ensure_splitter(start=a.start)
        o = Outlets()
    o.start()
    why = check_streams(o, 1.0)
    if why:
        o.close()
        raise SystemExit(why)
    time.sleep(max(0.0, a.rest_s - 1.0))

    def joints():
        """The station's joints (servo degrees, (3, 3)), or None without --station."""
        if not a.station:
            return None
        import urllib.request
        sv = json.loads(urllib.request.urlopen(a.station.rstrip("/") + "/api/v1/state", timeout=3).read())
        tel = (sv.get("telemetry") or {}).get("servos") or {}
        return np.array([[tel[str(k)][j] for j in ("aa", "fe1", "fe2")] for k in range(3)], float)
    tc = TipContacts(o, load_plan_d6(a.plan), gp=not a.no_gp, own_pose=not a.no_own_pose)
    info = tc.rest(time.time(), a.rest_s, refit=not a.no_refit, joints_servo=joints())
    for f, v in info.items():
        print(f"{f}: {json.dumps(v)}", flush=True)
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        time.sleep(0.5)
        j = joints()
        est = tc.estimate(None, j)
        cop, rec = tc.to_P(est, j) if j is not None else (None, {})
        print(f"{time.time() - t0:5.1f} s  " + "  ".join(
            f"{f} " + (f"{e['f_n']:.2f} N at pad {np.round(e['p'], 1).tolist()}"
                       + ("" if not rec.get(f) else f" P {rec[f]['p_P']}") if e.get("contact") else f"none (|d| {e.get('d_uT')} uT)")
            for f, e in est.items()), flush=True)
    o.close()
    return 0


def play(mux, t_from, seconds=None, links=LINKS, stop=None, quiet=False):
    """A splitter's log written into pseudo-terminals at its own pace, as the splitter writes them (a test feed for the
    live path; no mux is read): the records from t_from (s, the log's clock) for `seconds`, each board's to the
    pseudo-terminal linked at links/ch<channel>, in batches every 50 ms. Runs until the log's end or stop is set."""
    import tty
    log = MuxLog(mux, t_from, None if seconds is None else t_from + seconds)
    outs = {}
    for f in FINGERS:
        master, slave = os.openpty()
        tty.setraw(slave)
        os.set_blocking(master, False)
        link = Path(links) / f"ch{CHANNEL[f]}"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.ttyname(slave))
        outs[f] = (master, slave)
    t_log0, t_wall0, i, n = float(log.t[0]), time.time(), 0, len(log.t)
    if not quiet:
        print(f"playing {n} records ({log.t[-1] - t_log0:.1f} s) of {Path(mux).name} into {links}", flush=True)
    try:
        while i < n and not (stop is not None and stop.is_set()):
            j = int(np.searchsorted(log.t, t_log0 + time.time() - t_wall0, side="right"))
            j = max(j, i)
            for f in FINGERS:
                b = b"".join(np.asarray(log.r[f][k], "<f4").tobytes() + b"\r\n" for k in range(i, j))
                try:
                    os.write(outs[f][0], b)
                except (BlockingIOError, OSError):
                    pass
            i = j
            time.sleep(0.05)
    finally:
        for master, slave in outs.values():
            os.close(master)
            os.close(slave)


def _obj_gap(p, obj, shape, R_mm):
    """mm from the object's surface to point p (P): a sphere of radius R_mm about obj's centre, or a cylinder of
    radius R_mm about obj's z axis."""
    c = np.asarray(obj["p_PO"], float)
    if shape == "sphere":
        return float(np.linalg.norm(np.asarray(p) - c)) - R_mm
    from sr2hand.geometry.rotations import quat_to_mat
    ax = quat_to_mat(np.asarray(obj["q_PO"], float))[:, 2]
    v = np.asarray(p) - c
    return float(np.linalg.norm(v - (v @ ax) * ax)) - R_mm


def replay(rows, feed, d6, rest_end, rest_s, shape=None, R_mm=None, refit=True, gp=True, own_pose=True, joints_of=None,
           obj_of=None, rest_joints=None):
    """Every row of a record through the estimator at its time: rows with t_ms (corvus's clock), joints (servo degrees,
    9) and obj; rest_joints (servo degrees, 9) the joints of the rest (default: the first row's). Returns (rest info,
    per-row records)."""
    tc = TipContacts(feed, d6, gp=gp, own_pose=own_pose)
    jo = joints_of or (lambda row: row.get("joints"))
    if rest_joints is None:
        rest_joints = next((jo(r) for r in rows if jo(r) is not None), None)
    info = tc.rest(rest_end, rest_s, refit=refit,
                   joints_servo=None if rest_joints is None else np.asarray(rest_joints, float).reshape(3, 3))
    out = []
    for row in rows:
        t = row["t_ms"] / 1000.0
        j = jo(row)
        j = None if j is None else np.asarray(j, float).reshape(3, 3)
        a = time.perf_counter()
        est = tc.estimate(t, j)
        b = time.perf_counter()
        cop, rec = tc.to_P(est, j)
        c = time.perf_counter()
        obj = obj_of(row) if obj_of else row.get("obj")
        gaps = {f: (None if rec.get(f) is None or obj is None or shape is None
                    else round(_obj_gap(rec[f]["p_P"], obj, shape, R_mm), 3)) for f in FINGERS}
        out.append(dict(t=row.get("t"), t_ms=row["t_ms"], est_ms=round(1e3 * (b - a), 3), to_P_ms=round(1e3 * (c - b), 3),
                        tips=rec, gap_mm=gaps, d_uT={f: est[f].get("d_uT") for f in FINGERS},
                        e={f: est[f].get("e") for f in FINGERS}, own_uT={f: est[f].get("own_uT") for f in FINGERS},
                        age_ms={f: (None if "t" not in est[f] else round(row["t_ms"] - 1e3 * est[f]["t"], 1)) for f in FINGERS},
                        cop=[None if not np.isfinite(v) else round(float(v), 4) for v in cop]))
    return info, out


def cmd_replay(a):
    """A recorded trial (hom_trial.py's corvus record) and the splitter's log of its span, period by period."""
    lines = [json.loads(x) for x in Path(a.corvus).read_text().splitlines() if x.strip()]
    head = lines[0]
    rows = [x for x in lines[1:] if "seq" in x]
    t0 = head["t0_ms"] / 1000.0
    rest_end = t0 + a.rest_end
    feed = MuxLog(a.mux, rest_end - a.rest_s - KEEP_S, rows[-1]["t_ms"] / 1000.0 + 1.0)
    plan = json.loads(Path(a.plan).read_text())
    R_mm = a.radius if a.radius else 1e3 * plan["meta"]["hom"]["setup"]["r_m"]
    info, out = replay(rows, feed, plan["meta"]["hom"]["d6_P_mm"], rest_end, a.rest_s, a.object, R_mm,
                       refit=not a.no_refit, gp=not a.no_gp, own_pose=not a.no_own_pose)
    summary = summarize(out, info)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(json.dumps({"kind": "header", "corvus": a.corvus, "mux": a.mux, "plan": a.plan, "object": a.object,
                                 "R_mm": R_mm, "rest_end_s": a.rest_end, "rest_s": a.rest_s, "gp": not a.no_gp,
                                 "own_pose": not a.no_own_pose, "refit": not a.no_refit, "rest": info,
                                 "summary": summary}) + "\n")
            for r in out:
                fh.write(json.dumps(r) + "\n")
    print(json.dumps({"rest": info, "summary": summary}, indent=1))
    return 0


def summarize(out, info=None):
    """Per finger: periods with a contact, their force and their gap to the object (quantiles); the estimate's time."""
    s = {}
    for f in FINGERS:
        on = [r for r in out if r["tips"].get(f)]
        F = np.array([r["tips"][f]["f_n"] for r in on]) if on else np.zeros(0)
        g = np.array([r["gap_mm"][f] for r in on if r["gap_mm"][f] is not None])
        s[f] = dict(periods=len(out), with_contact=len(on), first_t=on[0]["t"] if on else None,
                    f_n_N=None if not len(F) else np.round(np.percentile(F, [10, 50, 90]), 3).tolist(),
                    gap_mm=None if not len(g) else np.round(np.percentile(g, [10, 50, 90]), 2).tolist(),
                    gap_abs_median_mm=None if not len(g) else round(float(np.median(np.abs(g))), 2))
    est = np.array([r["est_ms"] for r in out])
    s["est_ms"] = np.round(np.percentile(est, [50, 95, 100]), 2).tolist() if len(est) else None
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("logger", help="the splitter's state and its three streams")
    p.add_argument("--start", action="store_true", help="start the splitter when it is not running")
    p = sub.add_parser("rest", help="the rest at the open pose and live estimates (no motion)")
    p.add_argument("--plan", required=True, help="the trial's plan (its meta.hom.d6_P_mm: the mounts in P)")
    p.add_argument("--rest-s", type=float, default=1.0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--station", default=None, help="the station's URL: its joints place the contacts in P")
    p.add_argument("--start", action="store_true", help="start the splitter when it is not running")
    p.add_argument("--no-refit", action="store_true", help="keep the saved models, re-zeroed (no new fit at rest)")
    p.add_argument("--no-gp", action="store_true", help="the physical model alone, without the board GP")
    p.add_argument("--no-own-pose", action="store_true", help="without the own-pose term")
    p.add_argument("--links", default=None, help="test only: the pseudo-terminals of play, without the splitter's check")
    p = sub.add_parser("replay", help="a recorded trial through the estimator")
    p.add_argument("--corvus", required=True, help="hom_trial.py's record (<stamp>_<task>_corvus.jsonl)")
    p.add_argument("--mux", required=True, help="the splitter's .bin over the trial")
    p.add_argument("--plan", required=True)
    p.add_argument("--object", choices=("sphere", "cylinder"), required=True)
    p.add_argument("--radius", type=float, default=None, help="mm (default: the plan's meta.hom.setup.r_m)")
    p.add_argument("--rest-end", type=float, default=-0.2, help="s of the trial's clock at which the rest ends")
    p.add_argument("--rest-s", type=float, default=1.0)
    p.add_argument("--no-refit", action="store_true")
    p.add_argument("--no-gp", action="store_true")
    p.add_argument("--no-own-pose", action="store_true", help="without the own-pose term")
    p.add_argument("--out", default=None, help="JSON lines: a header with the rest and the summary, then one per period")
    p = sub.add_parser("own-pose-fit", help="the OwnPose GPs from the crosstalk sweep's rows (workstation)")
    p.add_argument("--out", default=str(OWN_POSE))
    p = sub.add_parser("play", help="a splitter's log into pseudo-terminals at its own pace (a test feed)")
    p.add_argument("--mux", required=True)
    p.add_argument("--from", dest="t_from", type=float, required=True, help="s of the log's clock (corvus's epoch)")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--links", required=True, help="a folder of its own (never the splitter's /tmp/reskin5x_mux)")
    a = ap.parse_args()
    if a.cmd == "own-pose-fit":
        print(json.dumps(fit_own_pose(a.out)))
        return 0
    if a.cmd == "play":
        if Path(a.links).resolve() == LINKS.resolve():
            raise SystemExit(f"{LINKS} is the splitter's: give the test feed a folder of its own")
        play(a.mux, a.t_from, a.seconds, a.links)
        return 0
    return {"logger": cmd_logger, "rest": cmd_rest, "replay": cmd_replay}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
