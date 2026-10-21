"""Live contact position, force and pad deformation from the ReSkin 5X board, in the browser.

The server reads the board's stream (reskin5x_plot.Reader) or replays a recording, calibrates the measurement model of
scripts/fingertip_wrench_model.py from the reading at rest (the board's chip frames from --board-cal kept; poles, magnet
strength and offsets fitted), and every 33 ms estimates the contact from the mean reading of the last 50 ms. It serves a
three.js page that draws the pad's palmar face deformed by the FE basis (scaled), coloured by the contact pressure of
Winkler pads on the face's nodes (stiffness E* a_i / h_i, builder v2's rule, under a sphere of radius R pressed at the
estimated point), the force arrow, the 1-sigma position ellipse, the chips and magnets, and traces.

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/wrench_viewer.py [--port /dev/ttyACM0] [--http 8790]
        [--board-cal docs/experiments/20261020-fingertip_contact_estimation/data/20261020-calib_pad_as_found.json] [--poles 1 1 1 1 1] [--replay rec.csv]
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
from scipy.spatial import Delaunay, cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
import fingertip_wrench_model as W  # noqa: E402
from reskin5x_plot import Reader, ADDRS  # noqa: E402

BASIS = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-b_v2_patch_basis.npz"
BOARD_CAL = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-calib_pad_as_found.json"
REC_DIR = ROOT / "logs/20261020-contact_estimation"
WIN_S, FRAME_S, REST_S = 0.05, 1 / 30, 2.0


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
    """Winkler pads on the display nodes: area a_i (a third of the adjacent triangles), depth h_i below the face nodes'
    centroid along x, stiffness E* a_i / h_i (N/mm), pressed by a sphere of radius R along the face normal."""

    def __init__(self, X, tris, E_star, centroid_x):
        self.X = X
        P = X[tris]
        area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
        self.a = np.zeros(len(X))
        for c in range(3):
            np.add.at(self.a, tris[:, c], area / 3)
        self.h = np.maximum(X[:, 0] - centroid_x, 0.5)
        self.k = E_star * self.a / self.h
        n = np.zeros_like(X)
        nt = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        nt *= np.sign(nt[:, :1])                             # outward: +x
        for c in range(3):
            np.add.at(n, tris[:, c], nt)
        self.n = n / np.linalg.norm(n, axis=1, keepdims=True)
        self.tree = cKDTree(X[:, 1:])

    def surface(self, y, z):
        """Face point and normal at (y, z): inverse-distance blend of the 4 nearest nodes."""
        d, i = self.tree.query([y, z], k=4)
        w = 1 / np.maximum(d, 1e-3) ** 2
        w /= w.sum()
        p = np.r_[(w * self.X[i, 0]).sum(), y, z]
        n = (w[:, None] * self.n[i]).sum(0)
        return p, n / np.linalg.norm(n)

    def press(self, y, z, f_press, R):
        """Pressure (kPa) of each pad when a sphere of radius R at (y, z) carries f_press N along the normal."""
        q, n = self.surface(y, z)
        c = q + R * n
        gap = np.linalg.norm(self.X - c, axis=1) - R
        lo, hi = 0.0, 2.0
        for _ in range(40):
            dlt = 0.5 * (lo + hi)
            F = (self.k * np.maximum(dlt - gap, 0)).sum()
            lo, hi = (dlt, hi) if F < f_press else (lo, dlt)
        pen = np.maximum(dlt - gap, 0)
        return 1e3 * self.k * pen / self.a, q, n


class App:
    def __init__(self, a):
        self.a = a
        self.basis = W.load_basis(BASIS)
        self.row = W.fe_row(str(self.basis["case"]))
        X = self.basis["disp_X"]
        tri = Delaunay(X[:, 1:]).simplices
        P = X[tri]
        edge = np.max([np.linalg.norm(P[:, i] - P[:, (i + 1) % 3], axis=1) for i in range(3)], axis=0)
        self.tris = tri[edge < 1.6]
        self.pads = Pads(X, self.tris, float(self.row["E_best_MPa"]), float(self.basis["node_centroid"][0]))
        self.tree = cKDTree(self.basis["pts"])
        if a.replay:
            self.src = Replay(a.replay, keep=3.0)
        else:
            self.src = Reader(a.port or sorted(glob.glob("/dev/ttyACM*"))[0], keep=3.0)
        self.src.start()
        self.board = W.Calib.load(a.board_cal)
        self.est = self.base = None
        self.clients, self.R_ind, self.status = set(), 8.0, "starting"
        self.rec = False
        self.trace = []

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
        est = W.Estimator(self.model, self.basis)
        est.set_noise_scale(1 / np.sqrt(WIN_S * 140))
        self.est, self.base = est, b_rest.ravel()
        self.status = (f"calibrated in {time.time() - t0:.1f} s: poles {cal.poles}, magnet strength {cal.k:.2f} of N52, "
                       f"offset ({cal.dx:+.2f}, {cal.dy:+.2f}, {cal.dz:+.2f}) mm, rms {cal.rms_mT:.2f} mT")
        print(self.status, flush=True)

    def baseline(self):
        t, r = self.src.snapshot()
        if len(t):
            self.base = r[t > t[-1] - 0.5][:, :, 1:].mean(0).ravel()

    # ---------------------------------------------------------------- per frame
    def frame(self):
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
                   chi2=round(e["chi2_none"], 1), e=round(e.get("e", 0.0), 1), rate=float(len(t) / (t[-1] - t[0] + 1e-9)))
        contact = e["contact"] and e["e"] > self.a.threshold
        out["contact"] = bool(contact)
        if e["contact"]:
            out.update(g=np.round(e["g_uT"], 1).tolist(), eps=e["eps"])
        if contact:
            f = np.array(e["f"])
            cov = np.array(e["cov"])
            sd = np.sqrt(np.abs(np.diag(cov)))
            ev, evec = np.linalg.eigh(cov[3:, 3:])
            p, q, n = self.pads.press(e["y"], e["z"], f[0], self.R_ind)
            dist, idx = self.tree.query([e["y"], e["z"]], k=4)
            w = 1 / np.maximum(dist, 1e-3) ** 2
            w /= w.sum()
            U = np.einsum("p,pkvc,k->vc", w, self.basis["disp_u"][idx], f)
            out.update(y=e["y"], z=e["z"], f=np.round(f, 3).tolist(), sd=np.round(sd, 3).tolist(), q=np.round(q, 3).tolist(),
                       n=np.round(n, 4).tolist(), ell=[float(np.sqrt(max(ev[0], 0))), float(np.sqrt(max(ev[1], 0))),
                                                         float(np.arctan2(evec[1, 1], evec[0, 1]))],
                       u=np.round(U.ravel(), 1).tolist(), p=np.round(p, 1).tolist(), pmax=float(p.max()))
        if not self.rec and hasattr(self.src, "all"):
            with self.src.lock:
                self.src.all.clear()
        return out

    def scene(self):
        c, m = W.magnets(self.row)
        chips = self.model.P if self.est is not None else W.chip_points()
        return dict(type="scene", verts=np.round(self.basis["disp_X"], 3).tolist(), tris=self.tris.tolist(),
                    magnets=[dict(c=c[j].tolist(), d=self.row["magnets"][j][0], t=self.row["magnets"][j][1],
                                  pole=int(self.cal.poles[j]) if self.est is not None else 1) for j in range(len(c))],
                    chips=[dict(p=np.round(chips[i], 3).tolist(), addr=ADDRS[i],
                                U=W.U_ORDER[self.cal.perm[i]] if self.est is not None else "") for i in range(5)],
                    threshold=self.a.threshold, R=self.R_ind, x_back=float(chips[:, 0].mean() + 0.42))


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
                    with app.src.lock:
                        app.src.all.clear()
                    app.rec, app.status = True, "recording the stream (r stops)"
                else:
                    app.rec = False
                    with app.src.lock:
                        rows = list(app.src.all)
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
        await asyncio.sleep(max(0.0, FRAME_S - (time.time() - t0)))


async def index(request):
    return web.FileResponse(HERE / "wrench_viewer.html")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None, help="serial port of the QT Py")
    ap.add_argument("--http", type=int, default=8790)
    ap.add_argument("--board-cal", default=str(BOARD_CAL), help="a full calibration of this board (chip frames)")
    ap.add_argument("--poles", type=int, nargs=5, default=None, help="known poles, e.g. 1 1 1 1 1 for N52 inserted one way")
    ap.add_argument("--threshold", type=float, default=25.0, help="explained chi-square above which a contact is shown")
    ap.add_argument("--replay", default=None, help="a reskin5x_plot.py --out recording to play instead of the board")
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
