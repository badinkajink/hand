"""Live contact position, force and pad deformation from the ReSkin 5X board, in the browser.

The server reads the board's stream (reskin5x_plot.Reader) or replays a recording, calibrates the measurement model of
scripts/fingertip_wrench_model.py from the reading at rest (the board's chip frames from --board-cal kept; poles, magnet
strength and offsets fitted), and every 33 ms estimates the contact anywhere on the pad's free surface, sides included
(fingertip_wrench_model.SurfaceEstimator on the FE surface basis of scripts/fingertip_wrench_fe.py --surface), from the
mean reading of the last 50 ms. It serves a three.js page that draws the pad's surface deformed by the FE basis
(scaled), coloured by the pressure of the sphere pads (stiffness E* a_i / h_i, h_i the pad's depth below the FE model's
node centroid along its normal: builder v2's rule) under a sphere of radius R pressed at the estimated point along the
surface normal, the force arrow, the 1-sigma position ellipse in the tangent plane, the chips and magnets, and traces.

Every record of the stream also goes to logs/20261020-contact_estimation/stream/<start>-stream.bin (88 bytes a record:
host time as float64, then 5 x [t x y z] as float32; read_stream() reads it back), so no session is lost; key r writes
the stretch between two presses of r as a CSV in the format of reskin5x_plot.py --out.

Pad changes (2026-10-22). The fitted poles, strength and offsets belong to one pad, so the viewer calibrates again by
itself when the pad changes. The magnitude of the reading over its 15 components is 0.16 mT with no pad on the board and
9 to 13 mT with the pads of 2026-10-22. Below 0.5 mT no contact is shown, and once a pad has been on the board and still
for 3 s it is calibrated. After the board is plugged in again it is calibrated as well, since the pad may have been
changed while it was unplugged. A re-zero that would move the rest by more than 10 % of the calibrated rest's
magnitude also calibrates, because the pad was set down in another place or pose. The pad's model is then kept, and
the reading only re-zeroed, when that model with a field from outside the pad (a uniform field and a linear gradient
across the board, fingertip_wrench_model.far_field_basis) fits the new rest at least as well as the new calibration
does (2026-10-25). Another fingertip laid against this one moved a rest by 0.94 mT, 11 %: the kept model fitted it at
0.29 mT and a new calibration at 0.43 mT, which had taken the neighbour's field into the magnets' strength (0.49 to 0.64
of N52). A reseated tip (0.95 against 0.33 mT) and a pad lifted off (0.75 against 0.13 mT) went to the new calibration.
--calib starts from a saved calibration in the same way, so a pad calibrated apart keeps its model when the viewer is
started with other fingertips against it. All 32 pole patterns are searched
(fingertip_wrench_model.calibrate_frames_fixed), so each magnet may be inserted either way. The status line gives the
poles row by row, tip end first, and every calibration is saved to logs/20261020-contact_estimation/calib/<time>-calib.json.

Drift (2026-10-21). The chips run without their on-chip temperature compensation, and their sensitivity falls by 0.65 to
0.70 % per C of die temperature: the board warms by about 1.3 C in the minutes after it is plugged in, which moves the
rest reading by 70 uT, and the room's temperature moves it over hours. Every reading is divided by its chip's gain at
the chip's own temperature (averaged over the last 3 s), with the coefficients fitted on the night of 2026-10-20
(fingertip_wrench_model.load_thermal; --no-thermal turns it off). The correction and the estimator's nuisances leave 8 to
14 uT of drift, so the estimator adds a noise floor (--floor, 2 uT a component) to the chips' noise,
and a contact is shown only above --min-force (0.25 N; once shown, down to 0.6 of it). After a touch the pad can come to
rest at a new place in its seat; with --auto-zero (default 3 s) the baseline is taken again once the reading has stayed
as still as at rest for that long (fingertip_wrench_model.RestZero), which also takes in a press held that still.

Fields from outside the pad (2026-10-23). Steel touched to the face changes the reading by 200 to 360 uT, and the
estimator explains that change as a press of 9 to 17 N on a side wall. With --foreign-gate (key f) every contact is
tested against a field from outside the pad (fingertip_wrench_model.ForeignField: a uniform field and a linear gradient
across the board) and not shown when that field explains the reading better by more than --foreign-lambda (25) in
chi-square. Steel that changes the reading by more than about 100 uT is then never shown as a contact; at 40 to 90 uT,
as an object comes near, some frames still show about 3.5 N. A press made with a steel object is rejected where the
object's field dominates the reading. The
test rejected none of 737 press and turn holds on the UR5 (docs/experiments/20261023-foreign_field_rejection).

Correction learned on the UR5 (2026-10-25). --correction names a UR5 session's data folder (holds.npz, calib_rest.json
under docs/experiments/20261022-ur5_tip_calibration/data). At start a Gaussian process is fitted to that session's
press holds: its inputs are the physical estimate's contact point and press, its outputs their errors against the
robot's ground truth (scripts/fingertip_second_tip_transfer.py, train_sphere2). Inputs and outputs are in the pad's
units, so the correction applies to another pad of the same design once its model is rebuilt from its rest. A shown
contact then moves to the corrected point, put back on the nearest surface candidate, and takes the corrected press;
the decision to show it stays with the physical estimate. On cb1t's own held-out presses at 2 N and over it cut the
miss from 1.16 to 0.55 mm; below 1 N the model and its correction fail
(docs/experiments/20261022-ur5_tip_calibration/20261024-checkerboard_fingertip_on_the_ur5.html, Section 7). sphere2's
correction moved the second fingertip's points further from the reference press on 2026-10-23 (3.71 against 3.07 mm),
so on another pad it is a trial (key g).

Several fingertips and the pooled GP (2026-10-26). With --mux the viewer reads a board running
firmware/reskin5x_mux_stream and gives every board on the mux a row of the page: its pad on the left, its readout on
the right, with --names, --calib and --correction given board by board (- for none) and the shared settings in the
bar at the top. --board-gp adds an estimate of its own, the GP from the board's change to the contact point and force
fitted on UR5 presses and slides of several tips (scripts/ur5_tip_calibration_board_gp.py; on ID 1's and ID 2's holds,
the calibration coordinator's pooled model of 2026-10-26). Each panel's menu picks the estimate shown: the physical
model, the physical model with the correction, or the GP; g moves every board to its next one. The decision to show a
contact stays with the physical estimate; the GP gives the point (put on the nearest surface candidate) and the force.
No UR5 hold pressed harder than 5.1 N, and 3 % of them pressed over 4 N. Far from every hold the GP's estimate returns
to the holds' mean, a press of 1.06 N: on the owner's presses of 2026-10-26 it read 1.1 to 1.3 N while the physical
model read 4 to 25 N. The GP's point and force are therefore weighted by w, one minus the GP's posterior variance of
its standardized outputs, and the physical estimate's by 1 - w. w is 1 at a hold and 0 far from all of them; it is
conditioned on the 256 holds of largest kernel weight, which leaves it 0.011 to 0.015 under the value on all 3,813 on
average. On those presses w was 0.8 to 0.98 between 1 and 4 N and 0 to 0.07 above 5 N. The panel's contact line gives
w, and the table gives the force's direction and its angle from the inward normal.

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/wrench_viewer.py [--port /dev/ttyACM0] [--http 8790]
        [--board-cal docs/experiments/20261020-fingertip_contact_estimation/data/20261020-calib_pad_as_found.json]
        [--poles 1 1 1 1 1] [--replay rec.csv] [--no-log] [--foreign-gate] [--correction <UR5 session data folder>]
        [--calib logs/20261020-contact_estimation/calib/<time>-calib.json]
    ... --mux /dev/serial/by-id/<mux board> --names "ID 1" "ID 2" --calib <cal 1> <cal 2> --correction <folder 1> <folder 2>
        --board-gp docs/experiments/20261022-ur5_tip_calibration/data/20261026-board_gp_id1_id2.npz
then open http://localhost:8790. Keys (every board): b new baseline, c recalibrate at rest (keep the pad untouched),
r record, z re-zero at rest on or off, f the test for fields from outside the pad on or off, g the next estimate.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from aiohttp import web, WSMsgType
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
import fingertip_wrench_model as W  # noqa: E402
from reskin5x_plot import Reader, MuxReader, ADDRS  # noqa: E402

DATA = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data"
BASIS = DATA / "20261020-b_v2_surface_basis.npz"
BOARD_CAL = DATA / "20261020-calib_pad_as_found.json"
REC_DIR = ROOT / "logs/20261020-contact_estimation"
WIN_S, FRAME_S, REST_S = 0.05, 1 / 30, 2.0
PAD_OFF_MT, PAD_ON_MT = 0.5, 1.0   # field magnitude over the 15 components: 0.16 mT with no pad, 9-13 mT with the pads of 10-22
REST_MOVED = 0.1                   # a rest this fraction of the calibrated rest's magnitude away is calibrated again
                                   # (seat shifts of 10-20 moved it 62-104 uT, about 1 %; the pad set down in another
                                   # pose at 15:22:59 on 10-22 moved it 14.3 mT, 110 %)
STREAM_DTYPE = np.dtype([("t", "<f8"), ("r", "<f4", (20,))])


def read_stream(path):
    """(host time (n,), records (n, 5, 4)) of a stream log written by the viewer."""
    a = np.fromfile(path, STREAM_DTYPE)
    return a["t"], a["r"].reshape(-1, 5, 4).astype(float)


class Replay(threading.Thread):
    """Plays a recording with Reader's interface: a reskin5x_plot.py --out CSV (host time, 5 x [t x y z]) or an .npz with
    t (n,) and r (n, 5, 4), up to `seconds` of it, `speed` times faster than it was recorded. Times stay those of the
    recording, so the viewer's windows (50 ms, 3 s) span the same stretch of it at any speed."""

    def __init__(self, path, keep, speed=1.0, seconds=None):
        super().__init__(daemon=True)
        if str(path).endswith(".npz"):
            z = np.load(path)
            self.t_rec, self.r_rec = z["t"] - z["t"][0], z["r"].astype(float)
        else:
            d = np.loadtxt(path)
            self.t_rec, self.r_rec = d[:, 0] - d[0, 0], d[:, 1:].reshape(-1, 5, 4)
        if seconds:
            keep_rows = self.t_rec <= seconds
            self.t_rec, self.r_rec = self.t_rec[keep_rows], self.r_rec[keep_rows]
        self.keep, self.speed, self.lock, self.stop, self.t0, self.all = keep, speed, threading.Lock(), False, None, []

    def run(self):
        self.t0 = time.time()
        while not self.stop:
            time.sleep(0.01)
            if (time.time() - self.t0) * self.speed > self.t_rec[-1]:
                self.t0 = time.time()                       # loop

    def snapshot(self):
        if self.t0 is None:
            return np.zeros(0), np.zeros((0, 5, 4))
        now = (time.time() - self.t0) * self.speed
        sel = (self.t_rec <= now) & (self.t_rec > now - self.keep)
        return self.t_rec[sel] + self.t0, self.r_rec[sel]


class Pads:
    """The sphere pads at the basis points: area a_i, depth h_i below the pad's centroid along the pad's normal (at
    least 1 mm), stiffness E* a_i / h_i (N/mm), pressed by a sphere of radius R along the surface normal."""

    def __init__(self, X, normals, areas, E_star, centroid):
        self.X, self.n, self.a = X, normals, areas
        self.h = np.maximum(np.einsum("ij,ij->i", normals, X - centroid), 1.0)
        self.k = E_star * self.a / self.h

    def press(self, q, n, f_press, R):
        """Pressure (kPa) of each pad when a sphere of radius R touching the surface at q (normal n) carries f_press N."""
        c = np.asarray(q) + R * np.asarray(n)
        gap = np.linalg.norm(self.X - c, axis=1) - R
        lo, hi = 0.0, 3.0
        for _ in range(40):
            dlt = 0.5 * (lo + hi)
            F = (self.k * np.maximum(dlt - gap, 0)).sum()
            lo, hi = (dlt, hi) if F < f_press else (lo, dlt)
        return 1e3 * self.k * np.maximum(dlt - gap, 0) / self.a


def region(n):
    """Name of the part of the pad whose outward normal is n (tip frame)."""
    if n[0] > 0.7:
        return "palmar face"
    if n[2] > 0.6:
        return "tip end"
    if n[2] < -0.6:
        return "mount end"
    side = "+y side" if n[1] > 0 else "-y side"
    return side if n[0] < 0.3 else f"face edge, {side}"


class BoardGP:
    """The GP from the board's change to the contact point (mm) and the 3D force (N), pad frame, pooled over UR5 sessions
    and saved by scripts/ur5_tip_calibration_board_gp.py (--board-gp)."""

    def __init__(self, path, m=256):
        z = np.load(path)
        self.mx, self.sx, self.my, self.sy, self.Z, self.a = (z[k] for k in ("mx", "sx", "my", "sy", "Z", "a"))
        self.ell, self.nz = float(z["ell"]), float(z["nz"])
        self.m = min(m, len(self.Z) - 1)
        self.sq = (self.Z ** 2).sum(1)
        self.name = "GP of " + ", ".join(s.split("-")[-1] for s in z["sessions"].astype(str))
        self.n = int(z["n_holds"])

    def predict(self, d):
        """Contact point (mm), force (N) and the weight w = 1 - the posterior variance of the standardized outputs, given
        the m holds of largest kernel weight (docstring, "Several fingertips and the pooled GP")."""
        q = (np.asarray(d, float) - self.mx) / self.sx
        k = np.exp(-0.5 * ((self.Z - q) ** 2).sum(1) / self.ell ** 2)
        y = k @ self.a * self.sy + self.my
        s = np.argpartition(-k, self.m)[:self.m]
        Ks = np.exp(-0.5 * np.maximum(self.sq[s][:, None] + self.sq[s][None] - 2 * self.Z[s] @ self.Z[s].T, 0) / self.ell ** 2)
        w = float(np.clip(k[s] @ cho_solve(cho_factor(Ks + self.nz ** 2 * np.eye(self.m), lower=True), k[s]), 0.0, 1.0))
        return y[:3], y[3:], w


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


class App:
    def __init__(self, a, src=None, name=None, calib=None, correction=None, board_gp=None):
        """One board: src a Reader-like source (default: --replay or --port), name its label (stream logs and
        calibrations get it as a suffix), calib a saved calibration to start from, correction a UR5 session folder
        for the GP correction, board_gp a BoardGP shared by every board."""
        self.a = a
        self.name, self.calib_arg, self.correction_arg, self.board_gp = name, calib, correction, board_gp
        self.tag = f"-{slug(name)}" if name else ""
        self.basis = W.load_basis(a.basis)
        self.row = W.fe_row(str(self.basis["case"]))
        self.tris = self.basis["tris"]
        self.pads = Pads(self.basis["pts"], self.basis["normals"], self.basis["pad_area"], float(self.row["E_best_MPa"]),
                         self.basis["node_centroid"])
        self.tree = cKDTree(self.basis["pts"])
        self.disp_u = self.basis["disp_u"]
        if src is not None:
            self.src = src
        elif a.replay:
            self.src = Replay(a.replay, keep=3.0, speed=a.speed, seconds=a.replay_seconds)
        else:
            self.src = Reader(a.port, keep=3.0)
        self.src.start()
        self.board = W.Calib.load(a.board_cal)
        self.kappa = None if a.no_thermal else W.load_thermal(a.thermal)
        self.floor, self.min_force, self.shown_prev = a.floor, a.min_force, False
        self.rz = W.RestZero(hold_s=a.auto_zero if a.auto_zero > 0 else 3.0)
        self.auto = a.auto_zero > 0
        self.foreign, self.ff, self.n_foreign = a.foreign_gate, None, 0
        self.corr_on, self.gp, self.corr_name, self.n_corr = False, None, None, 0
        self.mode, self.n_board_gp = ("board gp" if board_gp is not None else "physical"), 0
        self._ctree, self._ctree_est = None, None
        if correction:
            threading.Thread(target=self.train_correction, args=(correction,), daemon=True).start()
        self.T_cal = None
        self.est = self.base = self.cal = self.rest_ref = None
        self.pad_off, self.pad_still, self.cal_lock = False, W.RestZero(hold_s=3.0), threading.Lock()
        self.clients, self.R_ind, self.status = set(), 8.0, "starting"
        self.rec, self.rec_rows = False, []
        self.scene_dirty = False
        self.seen_reconnects, self.t_rebase = 0, None
        self.log = None
        if not a.replay and not a.no_log:
            (REC_DIR / "stream").mkdir(parents=True, exist_ok=True)
            self.log_path = REC_DIR / "stream" / (time.strftime("%Y%m%d-%H%M%S") + f"{self.tag}-stream.bin")
            self.log = open(self.log_path, "ab")
            print(f"stream log {self.log_path.relative_to(ROOT)}", flush=True)
        self.t_flush = time.time()

    # ---------------------------------------------------------------- calibration
    def records(self):
        """The last 3 s of records without those of a chip that read zeros or values out of range (framing errors when
        the board starts)."""
        t, r = self.src.snapshot()
        if not len(t):
            return t, r
        ok = (~np.all(r[:, :, 1:] == 0, axis=2).any(axis=1) & np.isfinite(r).all(axis=(1, 2))
              & (np.abs(r) < 1e5).all(axis=(1, 2)) & (r[:, :, 0] > -40).all(axis=1) & (r[:, :, 0] < 125).all(axis=1))
        return t[ok], r[ok]

    def fields(self, rr, r_T):
        """Fields (n, 15) uT of records rr (n, 5, 4), divided by each chip's gain at its temperature averaged over the
        records r_T (the last 3 s)."""
        b = rr[:, :, 1:]
        if self.kappa is not None:
            b = W.thermal_correct(b, r_T[:, :, 0].mean(0), self.kappa)
        return b.reshape(len(rr), 15)

    def saved_model(self, path):
        """(calibration, model, estimator, steel test) of a saved calibration (--calib)."""
        cal = W.Calib.load(path)
        model = W.Model(cal, self.row)
        est = W.SurfaceEstimator(model, self.basis, floor_uT=self.floor)
        est.set_noise_scale(1 / np.sqrt(WIN_S * 140))
        return cal, model, est, W.ForeignField(est, self.a.foreign_lambda)

    def kept_fit(self, model, b_rest):
        """rms (mT) of `model`'s rest against b_rest with a field from outside the pad (uniform and linear gradient
        across the board) fitted to the difference."""
        r = b_rest.ravel() - np.ravel(model.rest_uT(g_uT=(0.0, 0.0, 0.0)))
        A = W.far_field_basis(model)
        r = r - A @ np.linalg.lstsq(A, r, rcond=None)[0]
        return float(np.sqrt(np.mean(r ** 2))) / 1e3

    def calibrate(self, why=None, prev=None):
        """Fit the pad at rest. prev = (calibration, model, estimator, steel test) of the pad before a rest move, or of
        a saved calibration (--calib): kept, with the reading re-zeroed, when it fits the new rest with a field from
        outside the pad at least as well as the new calibration (docstring, "Pad changes")."""
        if not self.cal_lock.acquire(blocking=False):         # a calibration is already running
            return
        try:
            self.status = (f"{why}: " if why else "") + f"calibrating, keep the pad untouched for {REST_S:.0f} s"
            print(self.status, flush=True)
            time.sleep((REST_S + 0.2) / getattr(self.src, "speed", 1.0))   # a replay runs on the recording's clock
            t, r = self.records()
            if len(t) < 10:
                self.status = "no data from the board: not calibrated (c tries again)"
                return
            sel = t > t[-1] - REST_S
            b_rest = self.fields(r[sel], r).mean(0).reshape(5, 3)
            if np.linalg.norm(b_rest) < PAD_ON_MT * 1e3:
                self.pad_off, self.est, self.scene_dirty = True, None, True
                self.status = "no pad on the board: a pad is calibrated once it has been on the board and still for 3 s"
                print(self.status, flush=True)
                return
            self.T_cal = r[:, :, 0].mean(0)
            t0 = time.time()
            cal = W.calibrate_frames_fixed(b_rest, self.board, poles=tuple(self.a.poles) if self.a.poles else None)
            if prev is not None:
                kept = self.kept_fit(prev[1], b_rest)
                if kept <= cal.rms_mT:
                    self.cal, self.model, self.est, self.ff = prev
                    self.base, self.rest_ref, self.pad_off = b_rest.ravel(), b_rest.ravel(), False
                    self.rz.t_still = None
                    self.scene_dirty = True
                    self.status = (f"the pad's model kept (poles {self.pole_rows(self.cal.poles)}, strength {self.cal.k:.2f} "
                                   f"of N52): with a field from outside the pad it fits this rest at {kept:.2f} mT, a new "
                                   f"calibration at {cal.rms_mT:.2f} mT; the reading is re-zeroed")
                    print(self.status, flush=True)
                    return
                print(f"a new calibration fits this rest better than the pad's model with a field from outside the pad "
                      f"({cal.rms_mT:.2f} against {kept:.2f} mT)", flush=True)
            self.cal, self.rest_ref = cal, b_rest.ravel()
            self.model = W.Model(cal, self.row)
            est = W.SurfaceEstimator(self.model, self.basis, floor_uT=self.floor)
            est.set_noise_scale(1 / np.sqrt(WIN_S * 140))
            self.ff = W.ForeignField(est, self.a.foreign_lambda)
            self.est, self.base, self.pad_off = est, b_rest.ravel(), False
            self.rz.t_still = None
            self.scene_dirty = True                           # the pump sends the fitted poles and chip labels
            saved = ""
            if not self.a.replay and not self.a.no_log:
                (REC_DIR / "calib").mkdir(parents=True, exist_ok=True)
                path = REC_DIR / "calib" / (time.strftime("%Y%m%d-%H%M%S") + f"{self.tag}-calib.json")
                cal.save(path)
                saved = f"; saved to {path.relative_to(ROOT)}"
            self.status = (f"calibrated in {time.time() - t0:.1f} s: poles {self.pole_rows(cal.poles)}, magnet strength "
                           f"{cal.k:.2f} of N52, offset ({cal.dx:+.2f}, {cal.dy:+.2f}, {cal.dz:+.2f}) mm, rms "
                           f"{cal.rms_mT:.2f} mT (next pattern {cal.rms_next_mT:.2f} mT)")
            if cal.rms_next_mT < 1.5 * cal.rms_mT:          # 2.0-4.1 with the pads of 10-22 in their seat
                self.status += "; another pattern fits almost as well, so the pad may not sit in its seat"
            print(self.status + saved, flush=True)
        finally:
            self.cal_lock.release()

    def start_calibration(self, why, keep=False):
        """Calibrate in the background after the pad or the board changed, showing no contacts until it is done; with
        keep, the pad's present model is a candidate to keep."""
        prev = (self.cal, self.model, self.est, self.ff) if keep and self.est is not None else None
        self.est, self.scene_dirty = None, True
        self.status = f"{why}: calibrating, keep the pad untouched for {REST_S:.0f} s"
        threading.Thread(target=self.calibrate, args=(why, prev), daemon=True).start()

    def pole_rows(self, poles):
        """The poles row by row along the finger, tip end first, each row from -y to +y: '+ + / + / - -'."""
        c, _ = W.magnets(self.row)
        rows = {}
        for j in np.lexsort((c[:, 1], -c[:, 2])):
            rows.setdefault(round(float(c[j, 2]), 1), []).append("+" if poles[j] > 0 else "-")
        return "tip end " + " / ".join(" ".join(v) for v in rows.values()) + " mount end"

    def baseline(self):
        t, r = self.records()
        if len(t):
            self.base = self.fields(r[t > t[-1] - 0.5], r).mean(0)
            self.rz.t_still = None

    def set_floor(self, uT):
        self.floor = max(0.0, float(uT))
        if self.est is not None:
            self.est.floor = self.floor
            self.est.set_noise_scale(1 / np.sqrt(WIN_S * 140))

    def train_correction(self, folder):
        """The GP correction of the physical estimate learned on a UR5 session's press holds (docstring, "Correction
        learned on the UR5")."""
        import ur5_tip_calibration_predict as PR
        t0 = time.time()
        H, _, cal0 = PR.load(folder)
        Pv, Fv = PR.Physical(cal0, W.load_basis(PR.BASIS)).predict(H, np.arange(len(H["id"])))
        ok = np.all(np.isfinite(Pv), axis=1) & np.isfinite(Fv)
        self.gp = PR.GP().fit(np.c_[Pv, Fv][ok], np.c_[H["p"][ok] - Pv[ok], H["f_n"][ok] - Fv[ok]],
                              np.array([0.2, 0.2, 0.2, 0.05]))
        self.corr_name = Path(folder).name
        if self.mode == "physical":
            self.mode = "correction"
        self.scene_dirty = True                               # the page's list of estimates gains the correction
        print(f"UR5 correction of {self.corr_name} learned on {int(ok.sum())} press holds in {time.time() - t0:.1f} s "
              "(g turns it off)", flush=True)

    def corrected(self, e):
        """The estimate e with its contact point and press corrected by the GP; the point is put back on the nearest
        surface candidate, whose normal and tangents replace the estimate's, and the shear keeps its components."""
        if self._ctree_est is not self.est:
            self._ctree, self._ctree_est = cKDTree(self.est.C), self.est
        c = self.gp.predict(np.r_[e["p"], e["f_n"]][None])[0]
        k = int(self._ctree.query(np.asarray(e["p"]) + c[:3])[1])
        n, t1, t2 = self.est.Cn[k], self.est.T1[k], self.est.T2[k]
        fn = max(float(e["f_n"] + c[3]), 0.0)
        f = -fn * n + e["f_t"][0] * t1 + e["f_t"][1] * t2
        return dict(e, p=self.est.C[k].tolist(), n=n.tolist(), t1=t1.tolist(), t2=t2.tolist(), f=f.tolist(), f_n=fn,
                    index=k)

    def from_board_gp(self, e, d):
        """The pooled GP's contact point and force for the change d, weighted by w against the physical estimate e's (w
        from BoardGP.predict); the point is put on the nearest surface candidate, whose normal and tangents give the
        press and the shear."""
        if self._ctree_est is not self.est:
            self._ctree, self._ctree_est = cKDTree(self.est.C), self.est
        pg, fg, w = self.board_gp.predict(d)
        p = w * pg + (1 - w) * np.asarray(e["p"], float)
        f = w * fg + (1 - w) * np.asarray(e["f"], float)
        k = int(self._ctree.query(p)[1])
        n, t1, t2 = self.est.Cn[k], self.est.T1[k], self.est.T2[k]
        return dict(e, p=self.est.C[k].tolist(), n=n.tolist(), t1=t1.tolist(), t2=t2.tolist(), f=f.tolist(),
                    f_n=float(-f @ n), f_t=[float(f @ t1), float(f @ t2)], index=k, gp_w=w)

    def settings(self):
        th = (f"temperature correction on (chip gains -{-self.kappa.max() * 100:.2f} to -{-self.kappa.min() * 100:.2f} % "
              f"per C)" if self.kappa is not None else "temperature correction off")
        az = f"re-zero after {self.rz.hold_s:.0f} s at rest: on, {self.rz.count} so far" if self.auto else "re-zero at rest: off"
        ff = (f"fields from outside the pad rejected above {self.a.foreign_lambda:.0f} in chi-square, {self.n_foreign} frames "
              "so far" if self.foreign else "fields from outside the pad: not tested")
        return f"{th}; noise floor {self.floor:.2f} uT; shown above {self.min_force:.2f} N; {az}; {ff}; {self.mode_text()}"

    def modes(self):
        """The estimates this board can show: the physical model, with the UR5 correction once it is learned, and the
        pooled GP on the board's change when --board-gp is given."""
        return ["physical"] + (["correction"] if self.gp is not None else []) + (["board gp"] if self.board_gp else [])

    def mode_names(self):
        return {"physical": "physical model", "correction": f"physical model + correction of {self.corr_name}",
                "board gp": f"{self.board_gp.name} ({self.board_gp.n} holds)" if self.board_gp else ""}

    def mode_text(self):
        names = self.mode_names()
        extra = (", correction being learned" if self.correction_arg and self.gp is None else "")
        return f"estimate: {names[self.mode]} (g cycles{extra})"

    def cycle_mode(self, mode=None):
        ms = self.modes()
        self.mode = mode if mode in ms else ms[(ms.index(self.mode) + 1) % len(ms)] if self.mode in ms else ms[0]

    def drain(self):
        """Records since the last frame into the stream log and, while recording, into the recording."""
        if not hasattr(self.src, "all"):
            return
        with self.src.lock:
            rows = self.src.all[:]
            self.src.all.clear()
        if self.log is not None and rows:
            a = np.zeros(len(rows), STREAM_DTYPE)
            a["t"] = [t for t, _ in rows]
            a["r"] = np.array([r.ravel() for _, r in rows], np.float32)
            a.tofile(self.log)
            if time.time() - self.t_flush > 1.0:
                self.log.flush()
                self.t_flush = time.time()
        if self.rec:
            self.rec_rows.extend(rows)

    # ---------------------------------------------------------------- per frame
    def frame(self):
        self.drain()
        if not getattr(self.src, "connected", True):
            return dict(type="status", status="board unplugged: waiting for it on USB")
        if getattr(self.src, "reconnects", 0) != self.seen_reconnects:
            # the board restarts when it is plugged in again, and the pad may have been changed meanwhile: a new
            # calibration once REST_S of records have arrived
            self.seen_reconnects, self.t_rebase = self.src.reconnects, time.time() + REST_S + 0.5
            self.status = f"board plugged in again on {self.src.port}: keep the pad untouched, calibration in {REST_S:.0f} s"
            print(self.status, flush=True)
        t, r = self.src.snapshot()
        if self.t_rebase is not None and time.time() > self.t_rebase and len(t) > 10:
            self.t_rebase = None
            self.start_calibration("board plugged in again")
        if len(t) < 3 or self.t_rebase is not None:
            return dict(type="status", status=self.status)
        t, r = self.records()
        if len(t) < 3:
            return dict(type="status", status="no data from a chip")
        sel = t > t[-1] - WIN_S
        if sel.sum() == 0:
            return dict(type="status", status="no data from a chip")
        b_now = self.fields(r[sel], r).mean(0)
        level = float(np.linalg.norm(b_now)) / 1e3
        if level < PAD_OFF_MT:
            # the pad is off the board: the next pad is calibrated once it has been on the board and still for 3 s
            self.pad_still.t_still = None
            if not self.pad_off:
                self.pad_off, self.est, self.scene_dirty = True, None, True
                self.status = (f"no pad on the board ({level:.2f} mT): put one on and leave it untouched; it is "
                               f"calibrated after {self.pad_still.hold_s:.0f} s at rest")
                print(self.status, flush=True)
            return dict(type="status", status=self.status)
        if self.pad_off:
            if level > PAD_ON_MT:
                last = t > t[-1] - self.pad_still.span_s
                if self.pad_still.update(t[last], self.fields(r[last], r), np.zeros(15)) is not None:
                    self.pad_off = False
                    self.start_calibration("a pad was put on the board")
                else:
                    self.status = f"a pad is on the board: it is calibrated after {self.pad_still.hold_s:.0f} s at rest"
            return dict(type="status", status=self.status)
        if self.est is None:
            return dict(type="status", status=self.status)
        d = b_now - self.base
        if self.auto:
            last = t > t[-1] - self.rz.span_s
            nb = self.rz.update(t[last], self.fields(r[last], r), self.base)
            ref = self.rest_ref if self.rest_ref is not None else np.ravel(self.cal.b_rest_uT)
            if nb is not None and np.linalg.norm(nb - ref) > REST_MOVED * np.linalg.norm(ref):
                # the pad was set down in another place or pose: a new calibration in place of a re-zero
                self.start_calibration(f"the rest moved {np.linalg.norm(nb - ref) / 1e3:.1f} mT from the calibrated one",
                                       keep=True)
                return dict(type="status", status=self.status)
            if nb is not None:
                d = d + self.base - nb
                self.base = nb
                self.status = f"re-zeroed at rest at {time.strftime('%H:%M:%S')}: the rest had moved {self.rz.last_uT:.0f} uT"
        e = self.est.estimate(d)
        dT = (r[:, :, 0].mean(0) - self.T_cal) if self.T_cal is not None else np.zeros(5)
        out = dict(type="frame", t=float(t[-1]), status=self.status, settings=self.settings(), db=np.round(d, 1).tolist(),
                   chi2=round(e["chi2_none"], 1), e=round(e.get("e", 0.0), 1), rate=float(len(t) / (t[-1] - t[0] + 1e-9)),
                   g=np.round(e["g_uT"], 1).tolist(), eps=e["eps"], dT=np.round(dT, 2).tolist(),
                   still=round(self.rz.spread, 1) if np.isfinite(self.rz.spread) else None)
        lim = self.min_force * (0.6 if self.shown_prev else 1.0)
        contact = e["contact"] and e["e"] > self.a.threshold and e["f_n"] >= lim
        if contact and self.foreign and self.ff is not None:
            lam = self.ff.lam(d, e)
            out["lam"] = round(lam, 1)
            if lam > self.ff.on:
                contact, out["foreign"] = False, True
                self.n_foreign += 1
                out["status"] = (f"a field from outside the pad (steel or a magnet near the tip?): the reading changed "
                                 f"{np.linalg.norm(d):.0f} uT and a field across the board explains it better than a "
                                 f"contact by {lam:.0f} in chi-square, so no contact is shown (f turns the test off)")
        if contact and self.mode == "correction" and self.gp is not None:
            e = self.corrected(e)
            self.n_corr += 1
            out["corrected"] = True
        elif contact and self.mode == "board gp" and self.board_gp is not None:
            e = self.from_board_gp(e, d)
            self.n_board_gp += 1
            out["gp_w"] = round(e["gp_w"], 2)
        out["mode"] = self.mode
        self.shown_prev = bool(contact)
        out["contact"] = bool(contact)
        if contact:
            p, n, t1, t2, f = (np.array(e[k]) for k in ("p", "n", "t1", "t2", "f"))
            cov = np.array(e["cov"])
            Rl = np.stack([-n, t1, t2])                       # press, shear along t1, shear along t2
            cf = Rl @ cov[:3, :3] @ Rl.T
            sd = np.sqrt(np.abs(np.r_[np.diag(cf), np.diag(cov[3:, 3:])]))
            ev, evec = np.linalg.eigh(cov[3:, 3:])            # ascending: evec[:, 1] the long axis in (t1, t2)
            pres = self.pads.press(p, n, e["f_n"], self.R_ind)
            dist, idx = self.tree.query(p, k=6)
            w = np.where(self.basis["normals"][idx] @ n > 0.3, 1 / np.maximum(dist, 1e-3) ** 2, 0.0)
            w /= w.sum()
            U = np.einsum("p,pkvc,k->vc", w, self.disp_u[idx].astype(np.float32), f)
            out.update(p=np.round(p, 3).tolist(), n=np.round(n, 4).tolist(), t1=np.round(t1, 4).tolist(),
                       t2=np.round(t2, 4).tolist(), f=np.round(f, 3).tolist(), fn=round(e["f_n"], 3),
                       ft=np.round(e["f_t"], 3).tolist(), sd=np.round(sd, 3).tolist(),
                       ell=[float(np.sqrt(max(ev[1], 0))), float(np.sqrt(max(ev[0], 0))),
                            float(np.arctan2(evec[1, 1], evec[0, 1]))],
                       u=np.round(U.ravel(), 1).tolist(), pr=np.round(pres, 1).tolist(), pmax=float(pres.max()),
                       where=region(n))
        return out

    def scene(self):
        c, m = W.magnets(self.row)
        chips = self.model.P if self.est is not None else W.chip_points()
        return dict(type="scene", verts=np.round(self.basis["pts"], 3).tolist(), tris=self.tris.tolist(),
                    magnets=[dict(c=c[j].tolist(), d=self.row["magnets"][j][0], t=self.row["magnets"][j][1],
                                  pole=int(self.cal.poles[j]) if self.est is not None else 1) for j in range(len(c))],
                    chips=[dict(p=np.round(chips[i], 3).tolist(), addr=ADDRS[i],
                                U=W.U_ORDER[self.cal.perm[i]] if self.est is not None else "") for i in range(5)],
                    threshold=self.a.threshold, R=self.R_ind, x_cut=float(self.basis["x_cut"]), min_force=self.min_force,
                    floor=self.floor, auto=self.auto, hold_s=self.rz.hold_s, thermal=self.kappa is not None,
                    foreign=self.foreign, name=self.name, mode=self.mode, modes=self.modes(), mode_names=self.mode_names())


async def handle(app, cmd):
    """One command of the page for one board."""
    c = cmd.get("cmd")
    if c == "baseline":
        app.baseline()
    elif c == "calibrate":
        await asyncio.get_running_loop().run_in_executor(None, app.calibrate)
    elif c == "indenter":
        app.R_ind = float(cmd["R"])
    elif c == "minforce":
        app.min_force = max(0.0, float(cmd["N"]))
    elif c == "floor":
        await asyncio.get_running_loop().run_in_executor(None, app.set_floor, float(cmd["uT"]))
    elif c == "autozero":
        app.auto = bool(cmd.get("on", not app.auto))
        app.rz.t_still = None
        app.scene_dirty = True
    elif c in ("mode", "correction"):
        app.cycle_mode(cmd.get("mode"))
        app.scene_dirty = True
    elif c == "foreign":
        app.foreign = bool(cmd.get("on", not app.foreign))
        app.scene_dirty = True
    elif c == "record" and hasattr(app.src, "all"):
        if not app.rec:
            app.rec_rows = []
            app.rec, app.status = True, "recording the stream (r stops)"
        else:
            app.rec = False
            rows = app.rec_rows
            REC_DIR.mkdir(parents=True, exist_ok=True)
            path = REC_DIR / (time.strftime("%Y%m%d-%H%M%S") + f"{app.tag}-viewer_session.csv")
            np.savetxt(path, np.array([np.r_[t, r.ravel()] for t, r in rows]), fmt="%.4f",
                       header="host_time " + " ".join(f"{q}{c}" for c in range(5) for q in "txyz"))
            app.status = f"recorded {len(rows)} records ({rows[-1][0] - rows[0][0]:.0f} s) to {path.relative_to(ROOT)}" \
                if rows else "nothing recorded"
            print(app.status, flush=True)


async def ws_handler(request):
    apps, clients = request.app["apps"], request.app["clients"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)
    await ws.send_str(json.dumps(dict(type="boards", names=[ap.name or "fingertip" for ap in apps])))
    for k, ap in enumerate(apps):
        await ws.send_str(json.dumps(dict(ap.scene(), board=k)))
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            cmd = json.loads(msg.data)
            b = cmd.get("board")
            for app in (apps if b is None or b == "all" else apps[int(b):int(b) + 1] if int(b) >= 0 else []):
                await handle(app, cmd)
    finally:
        clients.discard(ws)
    return ws


async def pump(app_web):
    apps, clients = app_web["apps"], app_web["clients"]
    while True:
        t0 = time.time()
        if clients:
            for k, app in enumerate(apps):
                if app.scene_dirty:
                    app.scene_dirty = False
                    sc = json.dumps(dict(app.scene(), board=k))
                    await asyncio.gather(*(c.send_str(sc) for c in list(clients)), return_exceptions=True)
                msg = json.dumps(dict(app.frame(), board=k))
                await asyncio.gather(*(c.send_str(msg) for c in list(clients)), return_exceptions=True)
        else:
            for app in apps:
                app.drain()
        await asyncio.sleep(max(0.0, FRAME_S - (time.time() - t0)))


async def index(request):
    return web.FileResponse(HERE / "wrench_viewer.html")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None, help="serial port of the QT Py (one board, firmware/reskin5x_stream)")
    ap.add_argument("--mux", default=None, help="serial port of a board running firmware/reskin5x_mux_stream: every board on "
                    "the mux gets its own view and panel")
    ap.add_argument("--names", nargs="+", default=None, help="labels of the boards, in the order of the mux channels")
    ap.add_argument("--board-gp", default=None, help="a pooled GP from the board's change to the contact point and force "
                    "(scripts/ur5_tip_calibration_board_gp.py): an estimate of its own, shown from the start (g cycles)")
    ap.add_argument("--http", type=int, default=8790)
    ap.add_argument("--basis", default=str(BASIS), help="FE surface basis (scripts/fingertip_wrench_fe.py --surface)")
    ap.add_argument("--board-cal", default=str(BOARD_CAL), help="a full calibration of this board (chip frames)")
    ap.add_argument("--poles", type=int, nargs=5, default=None, help="known poles, e.g. 1 1 1 1 1 for N52 inserted one way")
    ap.add_argument("--threshold", type=float, default=25.0, help="explained chi-square above which a contact is shown")
    ap.add_argument("--min-force", type=float, default=0.25, help="pressing force (N) above which a contact is shown")
    ap.add_argument("--floor", type=float, default=2.0, help="noise floor (uT a component) added to the chips' noise")
    ap.add_argument("--auto-zero", type=float, default=3.0, help="re-zero after this long at rest (s); 0 turns it off")
    ap.add_argument("--foreign-gate", action="store_true", help="show no contact when a field from outside the pad (steel, "
                                                                "a magnet) explains the reading better (key f)")
    ap.add_argument("--foreign-lambda", type=float, default=25.0, help="chi-square margin of that test")
    ap.add_argument("--calib", nargs="+", default=None, help="one per board (- for none): a saved calibration (logs/20261020-contact_estimation/calib/*.json) to "
                    "start from: kept when, with a field from outside the pad, it fits the rest at start at least as well "
                    "as a new calibration")
    ap.add_argument("--correction", nargs="+", default=None, help="one per board (- for none): a UR5 session's data folder (holds.npz, calib_rest.json): its GP "
                    "correction of the physical estimate, on from the start; key g turns it off and on")
    ap.add_argument("--thermal", default=str(W.THERMAL), help="the chips' temperature coefficients (study thermal)")
    ap.add_argument("--no-thermal", action="store_true", help="do not correct the readings for the chips' temperature")
    ap.add_argument("--replay", default=None, help="a recording to play instead of the board (reskin5x_plot.py --out "
                                                   "CSV, or .npz with t and r)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay this many times faster than recorded")
    ap.add_argument("--replay-seconds", type=float, default=None, help="replay only the first this many seconds")
    ap.add_argument("--no-log", action="store_true", help="do not write the stream log")
    a = ap.parse_args()
    gp = BoardGP(a.board_gp) if a.board_gp else None

    def pick(values, k):
        return None if not values or k >= len(values) or values[k] in ("-", "none") else values[k]
    if a.mux:
        mr = MuxReader(a.mux, keep=3.0)
        names = a.names or []
        apps = [App(a, src=b, name=names[k] if k < len(names) else f"channel {b.channel}", calib=pick(a.calib, k),
                    correction=pick(a.correction, k), board_gp=gp) for k, b in enumerate(mr.boards)]
        mr.start()
        print(f"{mr.port}: boards on mux channels {mr.channels}: " + ", ".join(ap.name for ap in apps), flush=True)
    else:
        apps = [App(a, name=pick(a.names, 0), calib=pick(a.calib, 0), correction=pick(a.correction, 0), board_gp=gp)]
    for app in apps:
        prev = app.saved_model(app.calib_arg) if app.calib_arg else None
        threading.Thread(target=app.calibrate, kwargs=dict(prev=prev), daemon=True).start()
    web_app = web.Application()
    web_app["apps"], web_app["clients"] = apps, set()
    web_app.router.add_get("/", index)
    web_app.router.add_get("/ws", ws_handler)

    async def start_pump(wa):
        wa["pump"] = asyncio.create_task(pump(wa))
    web_app.on_startup.append(start_pump)
    print(f"open http://localhost:{a.http}", flush=True)
    web.run_app(web_app, host="127.0.0.1", port=a.http, print=None)


if __name__ == "__main__":
    main()
