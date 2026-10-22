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
magnitude also calibrates, because the pad was set down in another place or pose. All 32 pole patterns are searched
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

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/wrench_viewer.py [--port /dev/ttyACM0] [--http 8790]
        [--board-cal docs/experiments/20261020-fingertip_contact_estimation/data/20261020-calib_pad_as_found.json]
        [--poles 1 1 1 1 1] [--replay rec.csv] [--no-log]
then open http://localhost:8790. Keys: b new baseline, c recalibrate at rest (keep the pad untouched), r record,
z re-zero at rest on or off.
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
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
import fingertip_wrench_model as W  # noqa: E402
from reskin5x_plot import Reader, ADDRS  # noqa: E402

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


class App:
    def __init__(self, a):
        self.a = a
        self.basis = W.load_basis(a.basis)
        self.row = W.fe_row(str(self.basis["case"]))
        self.tris = self.basis["tris"]
        self.pads = Pads(self.basis["pts"], self.basis["normals"], self.basis["pad_area"], float(self.row["E_best_MPa"]),
                         self.basis["node_centroid"])
        self.tree = cKDTree(self.basis["pts"])
        self.disp_u = self.basis["disp_u"]
        if a.replay:
            self.src = Replay(a.replay, keep=3.0, speed=a.speed, seconds=a.replay_seconds)
        else:
            self.src = Reader(a.port, keep=3.0)
        self.src.start()
        self.board = W.Calib.load(a.board_cal)
        self.kappa = None if a.no_thermal else W.load_thermal(a.thermal)
        self.floor, self.min_force, self.shown_prev = a.floor, a.min_force, False
        self.rz = W.RestZero(hold_s=a.auto_zero if a.auto_zero > 0 else 3.0)
        self.auto = a.auto_zero > 0
        self.T_cal = None
        self.est = self.base = self.cal = None
        self.pad_off, self.pad_still, self.cal_lock = False, W.RestZero(hold_s=3.0), threading.Lock()
        self.clients, self.R_ind, self.status = set(), 8.0, "starting"
        self.rec, self.rec_rows = False, []
        self.scene_dirty = False
        self.seen_reconnects, self.t_rebase = 0, None
        self.log = None
        if not a.replay and not a.no_log:
            (REC_DIR / "stream").mkdir(parents=True, exist_ok=True)
            self.log_path = REC_DIR / "stream" / time.strftime("%Y%m%d-%H%M%S-stream.bin")
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

    def calibrate(self, why=None):
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
            self.cal = cal
            self.model = W.Model(cal, self.row)
            est = W.SurfaceEstimator(self.model, self.basis, floor_uT=self.floor)
            est.set_noise_scale(1 / np.sqrt(WIN_S * 140))
            self.est, self.base, self.pad_off = est, b_rest.ravel(), False
            self.rz.t_still = None
            self.scene_dirty = True                           # the pump sends the fitted poles and chip labels
            saved = ""
            if not self.a.replay and not self.a.no_log:
                (REC_DIR / "calib").mkdir(parents=True, exist_ok=True)
                path = REC_DIR / "calib" / time.strftime("%Y%m%d-%H%M%S-calib.json")
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

    def start_calibration(self, why):
        """Calibrate in the background after the pad or the board changed, showing no contacts until it is done."""
        self.est, self.scene_dirty = None, True
        self.status = f"{why}: calibrating, keep the pad untouched for {REST_S:.0f} s"
        threading.Thread(target=self.calibrate, args=(why,), daemon=True).start()

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

    def settings(self):
        th = (f"temperature correction on (chip gains -{-self.kappa.max() * 100:.2f} to -{-self.kappa.min() * 100:.2f} % "
              f"per C)" if self.kappa is not None else "temperature correction off")
        az = f"re-zero after {self.rz.hold_s:.0f} s at rest: on, {self.rz.count} so far" if self.auto else "re-zero at rest: off"
        return f"{th}; noise floor {self.floor:.2f} uT; shown above {self.min_force:.2f} N; {az}"

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
            ref = np.ravel(self.cal.b_rest_uT)
            if nb is not None and np.linalg.norm(nb - ref) > REST_MOVED * np.linalg.norm(ref):
                # the pad was set down in another place or pose: a new calibration in place of a re-zero
                self.start_calibration(f"the rest moved {np.linalg.norm(nb - ref) / 1e3:.1f} mT from the calibrated one")
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
                    floor=self.floor, auto=self.auto, hold_s=self.rz.hold_s, thermal=self.kappa is not None)


async def ws_handler(request):
    app = request.app["app"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    app.clients.add(ws)
    await ws.send_str(json.dumps(app.scene()))
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            cmd = json.loads(msg.data)
            if cmd.get("cmd") == "baseline":
                app.baseline()
            elif cmd.get("cmd") == "calibrate":
                await asyncio.get_running_loop().run_in_executor(None, app.calibrate)
            elif cmd.get("cmd") == "indenter":
                app.R_ind = float(cmd["R"])
            elif cmd.get("cmd") == "minforce":
                app.min_force = max(0.0, float(cmd["N"]))
            elif cmd.get("cmd") == "floor":
                await asyncio.get_running_loop().run_in_executor(None, app.set_floor, float(cmd["uT"]))
            elif cmd.get("cmd") == "autozero":
                app.auto = bool(cmd.get("on", not app.auto))
                app.rz.t_still = None
                app.scene_dirty = True
            elif cmd.get("cmd") == "record" and hasattr(app.src, "all"):
                if not app.rec:
                    app.rec_rows = []
                    app.rec, app.status = True, "recording the stream (r stops)"
                else:
                    app.rec = False
                    rows = app.rec_rows
                    REC_DIR.mkdir(parents=True, exist_ok=True)
                    path = REC_DIR / time.strftime("%Y%m%d-%H%M%S-viewer_session.csv")
                    np.savetxt(path, np.array([np.r_[t, r.ravel()] for t, r in rows]), fmt="%.4f",
                               header="host_time " + " ".join(f"{q}{c}" for c in range(5) for q in "txyz"))
                    app.status = f"recorded {len(rows)} records ({rows[-1][0] - rows[0][0]:.0f} s) to {path.relative_to(ROOT)}" \
                        if rows else "nothing recorded"
                    print(app.status, flush=True)
    finally:
        app.clients.discard(ws)
    return ws


async def pump(app_web):
    app = app_web["app"]
    while True:
        t0 = time.time()
        if app.clients:
            if app.scene_dirty:
                app.scene_dirty = False
                sc = json.dumps(app.scene())
                await asyncio.gather(*(c.send_str(sc) for c in list(app.clients)), return_exceptions=True)
            msg = json.dumps(app.frame())
            await asyncio.gather(*(c.send_str(msg) for c in list(app.clients)), return_exceptions=True)
        else:
            app.drain()
        await asyncio.sleep(max(0.0, FRAME_S - (time.time() - t0)))


async def index(request):
    return web.FileResponse(HERE / "wrench_viewer.html")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None, help="serial port of the QT Py")
    ap.add_argument("--http", type=int, default=8790)
    ap.add_argument("--basis", default=str(BASIS), help="FE surface basis (scripts/fingertip_wrench_fe.py --surface)")
    ap.add_argument("--board-cal", default=str(BOARD_CAL), help="a full calibration of this board (chip frames)")
    ap.add_argument("--poles", type=int, nargs=5, default=None, help="known poles, e.g. 1 1 1 1 1 for N52 inserted one way")
    ap.add_argument("--threshold", type=float, default=25.0, help="explained chi-square above which a contact is shown")
    ap.add_argument("--min-force", type=float, default=0.25, help="pressing force (N) above which a contact is shown")
    ap.add_argument("--floor", type=float, default=2.0, help="noise floor (uT a component) added to the chips' noise")
    ap.add_argument("--auto-zero", type=float, default=3.0, help="re-zero after this long at rest (s); 0 turns it off")
    ap.add_argument("--thermal", default=str(W.THERMAL), help="the chips' temperature coefficients (study thermal)")
    ap.add_argument("--no-thermal", action="store_true", help="do not correct the readings for the chips' temperature")
    ap.add_argument("--replay", default=None, help="a recording to play instead of the board (reskin5x_plot.py --out "
                                                   "CSV, or .npz with t and r)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay this many times faster than recorded")
    ap.add_argument("--replay-seconds", type=float, default=None, help="replay only the first this many seconds")
    ap.add_argument("--no-log", action="store_true", help="do not write the stream log")
    a = ap.parse_args()
    app = App(a)
    threading.Thread(target=app.calibrate, daemon=True).start()
    web_app = web.Application()
    web_app["app"] = app
    web_app.router.add_get("/", index)
    web_app.router.add_get("/ws", ws_handler)

    async def start_pump(wa):
        wa["pump"] = asyncio.create_task(pump(wa))
    web_app.on_startup.append(start_pump)
    print(f"open http://localhost:{a.http}", flush=True)
    web.run_app(web_app, host="127.0.0.1", port=a.http, print=None)


if __name__ == "__main__":
    main()
