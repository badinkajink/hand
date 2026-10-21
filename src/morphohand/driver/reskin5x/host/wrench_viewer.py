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

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/wrench_viewer.py [--port /dev/ttyACM0] [--http 8790]
        [--board-cal docs/experiments/20261020-fingertip_contact_estimation/data/20261020-calib_pad_as_found.json]
        [--poles 1 1 1 1 1] [--replay rec.csv] [--no-log]
then open http://localhost:8790. Keys: b new baseline, c recalibrate at rest (keep the pad untouched), r record.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
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
STREAM_DTYPE = np.dtype([("t", "<f8"), ("r", "<f4", (20,))])


def read_stream(path):
    """(host time (n,), records (n, 5, 4)) of a stream log written by the viewer."""
    a = np.fromfile(path, STREAM_DTYPE)
    return a["t"], a["r"].reshape(-1, 5, 4).astype(float)


class Replay(threading.Thread):
    """Plays a reskin5x_plot.py --out recording (host time, 5 x [t x y z]) in real time, with Reader's interface."""

    def __init__(self, path, keep):
        super().__init__(daemon=True)
        d = np.loadtxt(path)
        self.t_rec, self.r_rec = d[:, 0] - d[0, 0], d[:, 1:].reshape(-1, 5, 4)
        self.keep, self.lock, self.stop, self.t0, self.all = keep, threading.Lock(), False, None, []

    def run(self):
        self.t0 = time.time()
        while not self.stop:
            time.sleep(0.01)
            if time.time() - self.t0 > self.t_rec[-1]:
                self.t0 = time.time()                       # loop

    def snapshot(self):
        if self.t0 is None:
            return np.zeros(0), np.zeros((0, 5, 4))
        now = time.time() - self.t0
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
            self.src = Replay(a.replay, keep=3.0)
        else:
            self.src = Reader(a.port or sorted(glob.glob("/dev/ttyACM*"))[0], keep=3.0)
        self.src.start()
        self.board = W.Calib.load(a.board_cal)
        self.est = self.base = None
        self.clients, self.R_ind, self.status = set(), 8.0, "starting"
        self.rec, self.rec_rows = False, []
        self.log = None
        if not a.replay and not a.no_log:
            (REC_DIR / "stream").mkdir(parents=True, exist_ok=True)
            self.log_path = REC_DIR / "stream" / time.strftime("%Y%m%d-%H%M%S-stream.bin")
            self.log = open(self.log_path, "ab")
            print(f"stream log {self.log_path.relative_to(ROOT)}", flush=True)
        self.t_flush = time.time()

    # ---------------------------------------------------------------- calibration
    def calibrate(self):
        self.status = f"calibrating: keep the pad untouched for {REST_S:.0f} s"
        time.sleep(REST_S + 0.2)
        t, r = self.src.snapshot()
        sel = t > t[-1] - REST_S
        b_rest = r[sel][:, :, 1:].mean(0)
        t0 = time.time()
        cal = W.calibrate_frames_fixed(b_rest, self.board, poles=tuple(self.a.poles) if self.a.poles else None)
        self.cal = cal
        self.model = W.Model(cal, self.row)
        est = W.SurfaceEstimator(self.model, self.basis)
        est.set_noise_scale(1 / np.sqrt(WIN_S * 140))
        self.est, self.base = est, b_rest.ravel()
        self.status = (f"calibrated in {time.time() - t0:.1f} s: poles {cal.poles}, magnet strength {cal.k:.2f} of N52, "
                       f"offset ({cal.dx:+.2f}, {cal.dy:+.2f}, {cal.dz:+.2f}) mm, rms {cal.rms_mT:.2f} mT")
        print(self.status, flush=True)

    def baseline(self):
        t, r = self.src.snapshot()
        if len(t):
            self.base = r[t > t[-1] - 0.5][:, :, 1:].mean(0).ravel()

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
        t, r = self.src.snapshot()
        if self.est is None or len(t) < 3:
            return dict(type="status", status=self.status)
        sel = t > t[-1] - WIN_S
        rr = r[sel][:, :, 1:]
        good = ~np.all(rr == 0, axis=2).any(axis=1)
        if good.sum() == 0:
            return dict(type="status", status="no data from a chip")
        d = rr[good].mean(0).ravel() - self.base
        e = self.est.estimate(d)
        out = dict(type="frame", t=float(t[-1]), status=self.status, db=np.round(d, 1).tolist(),
                   chi2=round(e["chi2_none"], 1), e=round(e.get("e", 0.0), 1), rate=float(len(t) / (t[-1] - t[0] + 1e-9)),
                   g=np.round(e["g_uT"], 1).tolist(), eps=e["eps"])
        contact = e["contact"] and e["e"] > self.a.threshold
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
                    threshold=self.a.threshold, R=self.R_ind, x_cut=float(self.basis["x_cut"]))


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
                for c in list(app.clients):
                    await c.send_str(json.dumps(app.scene()))
            elif cmd.get("cmd") == "indenter":
                app.R_ind = float(cmd["R"])
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
    ap.add_argument("--replay", default=None, help="a reskin5x_plot.py --out recording to play instead of the board")
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
