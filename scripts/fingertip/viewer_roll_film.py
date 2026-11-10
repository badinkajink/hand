#!/usr/bin/env python3
"""The fingertip viewer (src/sr2hand/driver/reskin5x/host/wrench_viewer.py) replayed on a roll of the UR5 calibration,
beside the roll as the robot recorded it, to show what the UR5 calibration changes (owner, 2026-11-10: "a video of the
magnetic fingertip viewer in sync with our snapshots of a roll, to demonstrate the effect of the calibration").

The roll. The rod night of 11-03/04 on the inverted rig (scripts/hom/hom_filter_real_check.py, the HOM review page's
Figure 8): seat 2 holds v3 ID 5 pad up; the robot presses a rod of radius 12.5 mm on the pad's face to 4 N, then turns
it about its contact line to +6, -6 and back to 0 deg at about 0.4 deg/s, holding 0.4 s at every 3 deg. The overhead
camera was off that night, so the roll is drawn from the robot's own record: the rod's pose in the pad frame P per robot
sample (flange pose, the HEX-E's deflection under the measured wrench, the joint rod registration), as the filter check
places it. The truth is the fitted pad's point nearest the rod's axis and the HEX-E's force.

The viewer. Two copies of the viewer's own App run on the held board's raw records of the same stretch, on a clock this
script sets: each calibrates the physical model from the rest before the press (as the viewer does live), and then
estimates every 1/30 s of the recording from the last 50 ms. One shows the physical model alone; the other the board
GP of the UR5 calibration (the viewer's --board-gp), fitted here on v3 ID 5's press, sweep, offset and slide holds of its
rig sessions with no hold of a roll action (the filter check's train_gp), saved in the viewer's format. The page is the
viewer's own wrench_viewer.html, with its WebSocket replaced by a feed this script fills and the truth drawn in green.

    resguard run --mem 5G -- .venv/bin/python scripts/fingertip/viewer_roll_film.py prepare   (session, truth, GP)
    .venv/bin/python scripts/fingertip/viewer_roll_film.py frames                              (the viewer's messages)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

OUT = ROOT / "docs/experiments/20261110-fingertip_viewer_roll"
DATA = OUT / "data"
MEDIA = OUT / "media"
WORK = ROOT / "logs/20261110-viewer_roll_film"
VIEWER = ROOT / "src/sr2hand/driver/reskin5x/host"
SEAT, TIP, ROLL = 2, 5, "probe-face-776-r90-roll"
PRE_S, POST_S = 14.0, 3.0              # s of the record kept before the press's level hold and after the roll's end
STEM = f"20261110-v3id{TIP}_rod_roll"


def prepare(a):
    """The roll's raw board records, the truth per robot sample and the GP without rolls, into DATA."""
    import hom_filter_real_check as FC
    t0 = time.time()
    ses = FC.load_session(SEAT)
    print(f"session loaded in {time.time() - t0:.0f} s", flush=True)
    R_Pe, t_Pe, fs = FC.fit_pad(ses.geo.pts, ses.geo.nrm)
    act = ses.acts[ROLL]
    tr = FC.roll_trace(ses, act, R_Pe, t_Pe)
    rec = tr["rec"]
    pl = tr["plateaus"]
    t_lo, t_hi = pl[0][0] - PRE_S, pl[-1][1] + POST_S
    tb, rb = ses.d["t_b"], ses.d["r_b"]
    m = (tb >= t_lo) & (tb <= t_hi)
    rob = tr["rob"]
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA / f"{STEM}_board.npz", t=tb[m], r=rb[m].astype(np.float32))
    # the robot's whole window too, with the rod's pose, so that the drawing starts before the press
    rb_all = ses.d["rb"]
    mr = (rb_all[:, 0] >= t_lo) & (rb_all[:, 0] <= t_hi)
    print(f"{m.sum()} board records over {t_hi - t_lo:.1f} s; {mr.sum()} robot samples; trace window "
          f"{rob['t'][0] - t_lo:.1f} to {rob['t'][-1] - t_lo:.1f} s", flush=True)
    keys = ("t", "p", "n", "depth", "c_P", "a_P", "R_rod", "f_P", "theta", "theta_dot")
    np.savez_compressed(DATA / f"{STEM}_truth.npz", **{k: np.asarray(rob[k]) for k in keys},
                        plateaus=np.array([[p0, p1] for p0, p1, _ in pl]), t_lo=t_lo, t_hi=t_hi,
                        R_Pe=R_Pe, t_Pe=t_Pe, rod_R_mm=float(ses.geo.R), rod_half_len_mm=float(ses.geo.half_len),
                        target=tr["target"], press_N=tr["press_N"])
    json.dump(dict(rec_keys=sorted(rec.keys()), plateaus=[[float(p0 - t_lo), float(p1 - t_lo), lab] for p0, p1, lab in pl],
                   pad_fit=fs), open(DATA / f"{STEM}_window.json", "w"), indent=1)
    # the pad's surface mesh in P (the analysis's), for the drawing
    np.savez_compressed(DATA / f"{STEM}_pad_mesh.npz", pts=np.asarray(ses.geo.pts), nrm=np.asarray(ses.geo.nrm))
    del ses
    t1 = time.time()
    m_gp, Htr, info = FC.train_gp(TIP)
    np.savez_compressed(DATA / f"20261110-board_gp_v3id{TIP}_no_rolls.npz",
                        **{k: np.asarray(v) for k, v in m_gp.items()},
                        sessions=np.array([Path(s).name for s in info["sessions"]]), n_holds=info["n_holds"])
    info = dict(info, sessions=[Path(s).name for s in info["sessions"]])
    json.dump(info, open(DATA / f"20261110-board_gp_v3id{TIP}_no_rolls.json", "w"), indent=1)
    print(f"GP on {info['n_holds']} holds ({info['roll_action_holds_left_out']} roll holds out) in "
          f"{time.time() - t1:.0f} s; total {time.time() - t0:.0f} s", flush=True)


class Clock:
    """A source for the viewer's App with Reader's interface whose present time this script sets: snapshot() gives the
    records of the last `keep` s before `now`. speed is large, so that App.calibrate's wait for the rest is a few ms."""

    def __init__(self, t, r, keep=3.0):
        self.t_rec, self.r_rec, self.keep, self.now, self.speed = t, r, keep, float(t[0]), 1e4

    def start(self):
        pass

    def snapshot(self):
        sel = (self.t_rec <= self.now) & (self.t_rec > self.now - self.keep)
        return self.t_rec[sel], self.r_rec[sel]


def viewer_args(**kw):
    """The viewer's command-line defaults for the v3 tips (wrench_viewer.main, --case four_05R4), without its log."""
    import fingertip_wrench_model as W
    sys.path.insert(0, str(VIEWER))
    import wrench_viewer as WV
    case = "four_05R4"
    a = dict(case=case, basis=str(W.basis_path(case)), dx_range=WV.DX_RANGE.get(case), board_cal=str(WV.BOARD_CAL),
             poles=None, threshold=25.0, min_force=0.25, floor=2.0, auto_zero=3.0, foreign_gate=False, foreign_lambda=25.0,
             thermal=str(W.THERMAL), no_thermal=False, replay=None, speed=1.0, replay_seconds=None, no_log=True)
    a.update(kw)
    return SimpleNamespace(**a)


def rest_end(t, r, quiet_uT=8.0):
    """The last record time before the board's reading leaves its first second's mean by more than quiet_uT (norm over
    the 15 components, 0.1 s means): the end of the rest before the press."""
    b = r[:, :, 1:].reshape(len(t), 15)
    b0 = b[t < t[0] + 1.0].mean(0)
    k = ((t - t[0]) // 0.1).astype(int)
    for i in range(k.max() + 1):
        sel = k == i
        if sel.sum() and np.linalg.norm(b[sel].mean(0) - b0) > quiet_uT:
            return float(t[sel][0]) - 0.2
    return float(t[-1])


def truth_at(T, tq):
    """The truth (point, normal, force in P) at times tq, from the robot samples; NaN outside them."""
    t = T["t"]
    out = {}
    for k in ("p", "n", "f_P"):
        v = np.column_stack([np.interp(tq, t, T[k][:, j], left=np.nan, right=np.nan) for j in range(3)])
        out[k] = v
    out["theta"] = np.interp(tq, t, T["theta"], left=np.nan, right=np.nan)
    return out


def frames(a):
    """The viewer's messages at every film frame: the physical model and the board GP, each from the viewer's App on the
    roll's raw records, with the truth; plus the per-frame estimates in one table for the traces."""
    sys.path.insert(0, str(VIEWER))
    import wrench_viewer as WV
    B = np.load(DATA / f"{STEM}_board.npz")
    T = dict(np.load(DATA / f"{STEM}_truth.npz"))
    t, r = B["t"].astype(float), B["r"].astype(float)
    t_lo = float(T["t_lo"])
    t_cal = rest_end(t, r)
    print(f"rest until {t_cal - t_lo:.2f} s of the window", flush=True)
    gp = WV.BoardGP(str(DATA / f"20261110-board_gp_v3id{TIP}_no_rolls.npz"))
    apps = {}
    for key, g in (("physical", None), ("gp", gp)):
        src = Clock(t, r)
        app = WV.App(viewer_args(auto_zero=a.auto_zero), src=src, name=key, board_gp=g)
        src.now = t_cal
        app.calibrate()
        if a.calib:                                   # a saved calibration in place of the rest fit (the viewer's --calib)
            app.cal, app.model, app.est, app.ff = app.saved_model(a.calib)
            app.status = f"saved calibration {Path(a.calib).name}: k {app.cal.k:.2f}, theta0 {app.cal.theta0:.1f}"
        print(key, app.status, flush=True)
        apps[key] = app
    t0, t1 = float(T["t"][0]), float(T["t"][-1])
    if a.from_rest:
        t0 = t_cal + 0.5
    tq = np.arange(t0, t1, a.step)
    msgs = {k: [] for k in apps}
    rows = []
    tr = truth_at(T, tq)
    for i, tt in enumerate(tq):
        row = dict(t=float(tt - t_lo))
        for key, app in apps.items():
            app.src.now = float(tt)
            m = app.frame()
            if m.get("type") == "frame":
                m["t"] = float(tt - t_lo)
            if np.isfinite(tr["p"][i]).all():
                m["truth"] = dict(p=np.round(tr["p"][i], 3).tolist(), n=np.round(tr["n"][i], 4).tolist(),
                                  f=np.round(tr["f_P"][i], 3).tolist(), theta=round(float(tr["theta"][i]), 3))
            msgs[key].append(m)
            row[key] = dict(contact=bool(m.get("contact")), p=m.get("p"), f=m.get("f"), fn=m.get("fn"),
                            gp_w=m.get("gp_w"), status=m.get("status", "")[:120])
        row["truth"] = msgs["physical"][-1].get("truth")
        rows.append(row)
    scenes = {k: app.scene() for k, app in apps.items()}
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / f"frames_step{a.step:g}{'_az' if a.auto_zero else ''}{'_calib' if a.calib else ''}.json"
    json.dump(dict(scenes=scenes, msgs=msgs, rows=rows, t_cal=t_cal - t_lo, step=a.step,
                   status={k: app.status for k, app in apps.items()},
                   rezero={k: app.rz.count for k, app in apps.items()}), open(out, "w"))
    # the errors along the pad against the truth
    for key in apps:
        e = [np.linalg.norm(np.subtract(rw[key]["p"], rw["truth"]["p"])) for rw in rows
             if rw[key]["contact"] and rw["truth"] is not None]
        nf = sum(1 for rw in rows if rw["truth"] is not None and not rw[key]["contact"])
        print(f"{key}: {len(e)} frames with a contact, {nf} without; distance to the truth median "
              f"{np.median(e):.2f} mm, 90th percentile {np.percentile(e, 90):.2f} mm; re-zeros {apps[key].rz.count}",
              flush=True)
    print(f"wrote {out.relative_to(ROOT)}", flush=True)


NAMES = ["Physical model, fitted to the rest reading", "GP of the UR5 calibration (no roll in its training)"]
TRUTH_JS = r"""
window.__truth = (k, tr) => {
  const b = boards[k]; if (!b || !b.face) return;
  if (!b.tg) {
    const g = new THREE.Group(), col = 0x00a33c;
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.5, 24, 16), new THREE.MeshPhongMaterial({ color: col }));
    const arr = new THREE.ArrowHelper(new THREE.Vector3(0, -1, 0), new THREE.Vector3(), 4, col, 1.2, 0.8);
    g.add(dot); g.add(arr); b.scene.add(g); b.tg = { g, dot, arr };
  }
  const { g, dot, arr } = b.tg;
  if (!tr) { g.visible = false; return; }
  g.visible = true;
  const q = T(tr.p), F = T(tr.f), L = Math.min(F.length() * +ar.value, 25), dir = F.clone().normalize();
  dot.position.copy(T(tr.p.map((v, c) => v + 0.35 * tr.n[c])));
  arr.position.copy(q.clone().sub(dir.clone().multiplyScalar(L))); arr.setDirection(dir);
  arr.setLength(Math.max(L, 0.5), Math.min(1.2, 0.4 * L + 0.3), Math.min(0.8, 0.3 * L + 0.2));
};
window.__cam = (k, pos, tgt) => { const b = boards[k]; b.camera.position.set(...pos); b.controls.target.set(...tgt); b.controls.update(); };
window.__ready = true;
"""


def harness_html():
    """The viewer's own page with three changes for a replay on a set clock: its WebSocket is a stub that this script
    feeds (window.__feed), the truth is drawn in green on every board (window.__truth), and the cameras can be set."""
    s = (VIEWER / "wrench_viewer.html").read_text()
    subs = [("ws = new WebSocket(`ws://${location.host}/ws`);",
             "ws = { readyState: 0, send() {}, close() {} }; window.__feed = s => ws.onmessage({ data: s });"),
            ("Arrow: force on the pad.", "Blue arrow: the estimated force on the pad. Green: the UR5's contact and the "
                                         "HEX-E's force."),
            ("connect();\n", "connect();\n" + TRUTH_JS)]
    for old, new in subs:
        if old not in s:
            raise SystemExit(f"wrench_viewer.html changed: {old[:40]!r} not found")
        s = s.replace(old, new, 1)
    return s


async def cdp_capture(url, jobs, size, out_dir, cams):
    """Load url in headless Chrome, run each job's JavaScript and save a screenshot per job."""
    import base64
    import subprocess
    import tempfile
    import urllib.request
    import websockets
    port = 9333
    prof = tempfile.mkdtemp(prefix="viewer_film_chrome_")
    proc = subprocess.Popen(["google-chrome", "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
                             f"--window-size={size[0]},{size[1]}", "--hide-scrollbars", "--no-first-run",
                             "--no-default-browser-check", "--enable-unsafe-swiftshader", "--use-angle=swiftshader",
                             "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1))
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                await asyncio_sleep(0.1)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            n = [0]

            async def call(method, **params):
                n[0] += 1
                mid = n[0]
                await ws.send(json.dumps(dict(id=mid, method=method, params=params)))
                while True:
                    r = json.loads(await ws.recv())
                    if r.get("id") == mid:
                        if "error" in r:
                            raise RuntimeError(f"{method}: {r['error']}")
                        return r.get("result", {})

            async def js(expr, wait=True):
                r = await call("Runtime.evaluate", expression=expr, awaitPromise=wait, returnByValue=True)
                if r.get("exceptionDetails"):
                    raise RuntimeError(f"page error: {r['exceptionDetails']}")
                return r.get("result", {}).get("value")

            await call("Emulation.setDeviceMetricsOverride", width=size[0], height=size[1], deviceScaleFactor=1, mobile=False)
            await call("Page.enable")
            await call("Page.navigate", url=url)
            for _ in range(300):
                if await js("window.__ready === true && typeof window.__feed === 'function'"):
                    break
                await asyncio_sleep(0.1)
            else:
                raise RuntimeError("the page did not load (three.js from unpkg?)")
            raf = "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))"
            for i, job in enumerate(jobs):
                await js(job, wait=False)
                if i == 0:
                    for k, (pos, tgt) in enumerate(cams):
                        await js(f"window.__cam({k}, {json.dumps(pos)}, {json.dumps(tgt)})")
                await js(raf)
                shot = await call("Page.captureScreenshot", format="png")
                (out_dir / f"v{i:05d}.png").write_bytes(base64.b64decode(shot["data"]))
                if i % 50 == 0:
                    print(f"  captured {i + 1}/{len(jobs)}", flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


async def asyncio_sleep(s):
    import asyncio
    await asyncio.sleep(s)


def capture(a):
    """Screenshots of the viewer's page, one per film frame, fed with the messages of frames()."""
    import asyncio
    import datetime as dt
    import functools
    import http.server
    import threading
    D = json.load(open(WORK / a.frames))
    T = np.load(DATA / f"{STEM}_truth.npz")
    t_lo = float(T["t_lo"])
    site = WORK / "site"
    site.mkdir(parents=True, exist_ok=True)
    (site / "harness.html").write_text(harness_html())
    out_dir = WORK / "viewer"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("v*.png"):
        f.unlink()
    keys = ["physical", "gp"]
    n = len(D["msgs"]["physical"]) if a.n is None else min(a.n, len(D["msgs"]["physical"]))
    jobs = []
    first = [json.dumps(dict(type="boards", names=NAMES))]
    for k, key in enumerate(keys):
        sc = dict(D["scenes"][key], board=k, name=NAMES[k])
        first.append(json.dumps(sc))
    pending = list(first)
    pending.append(None)                               # the arrow scale, set once the page has its controls
    for i in range(n):
        lines, pending = pending, []
        trs = []
        for k, key in enumerate(keys):
            m = dict(D["msgs"][key][i], board=k)
            tr = m.pop("truth", None)
            clock = dt.datetime.fromtimestamp(t_lo + m.get("t", 0.0)).strftime("%H:%M:%S")
            turn = f", the rod turned {tr['theta']:+.1f} deg" if tr else ""
            m["status"] = (f"Replay of the UR5 roll of 4 November ({clock}), v3 ID 5 pad up on seat 2: "
                           f"{m.get('t', 0.0):.1f} s{turn}. Green: the robot's contact and the HEX-E's force.")
            lines.append(json.dumps(m))
            trs.append(tr)
        if D["rows"][i]["t"] < a.t_start:              # fed, so that the page's traces start at the rest, not captured
            pending = lines
            continue
        body = "".join(f"window.__feed({json.dumps(s)});" if s is not None else
                       f"document.getElementById('ar').value = {a.arrow:g}; "
                       f"document.getElementById('arl').textContent = '{a.arrow:g}';" for s in lines)
        body += "".join(f"window.__truth({k}, {json.dumps(tr)});" for k, tr in enumerate(trs))
        jobs.append(body)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    handler.log_message = lambda *x, **y: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/harness.html"
    cams = [tuple(json.loads(a.cam)) for _ in keys]
    t0 = time.time()
    try:
        asyncio.run(cdp_capture(url, jobs, (a.width, a.height), out_dir, cams))
    finally:
        httpd.shutdown()
    print(f"{len(jobs)} screenshots in {out_dir.relative_to(ROOT)}, {time.time() - t0:.0f} s", flush=True)


def section(pts, tris, y0):
    """The pad surface's section by the plane y = y0: an ordered polyline (n, 2) of (x, z), mm, closed along the back."""
    segs = []
    for tri in tris:
        v = pts[tri]
        s = v[:, 1] - y0
        cut = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if s[i] * s[j] < 0:
                w = s[i] / (s[i] - s[j])
                cut.append(v[i] + w * (v[j] - v[i]))
        if len(cut) == 2:
            segs += cut
    q = np.array(segs)[:, [0, 2]]
    c = np.array([q[:, 0].min(), q[:, 1].mean()])
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]
    _, keep = np.unique(np.round(q, 3), axis=0, return_index=True)
    return q[np.sort(keep)]


def compose(a):
    """The film: on the left the roll drawn from the robot's record (the rod on the pad across its axis, the contact
    magnified) and the distance of each estimate from the robot's contact over the roll; on the right the viewer."""
    import subprocess
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    from PIL import Image
    import fingertip_wrench_model as W
    D = json.load(open(WORK / a.frames))
    T = np.load(DATA / f"{STEM}_truth.npz")
    tR = T["t"] - float(T["t_lo"])
    basis = W.load_basis(W.basis_path("four_05R4"))
    rows = D["rows"]
    keep = [i for i, rw in enumerate(rows) if rw["t"] >= a.t_start]
    n_cap = len(sorted((WORK / "viewer").glob("v*.png")))
    if n_cap != len(keep):
        raise SystemExit(f"{n_cap} viewer screenshots for {len(keep)} frames from {a.t_start} s: run capture first")
    tt = np.array([rows[i]["t"] for i in keep])

    def dist(key):
        return np.array([np.linalg.norm(np.subtract(rows[i][key]["p"], rows[i]["truth"]["p"]))
                         if rows[i][key]["contact"] and rows[i]["truth"] else np.nan for i in keep])
    d_ph, d_gp = dist("physical"), dist("gp")
    th = np.interp(tt, tR, T["theta"])
    y0 = float(np.median(T["p"][:, 1]))
    prof = section(basis["pts"], basis["tris"], y0)
    poly = np.vstack([prof, [[basis["x_cut"], prof[-1, 1]], [basis["x_cut"], prof[0, 1]]]])
    R = float(T["rod_R_mm"])
    k0 = 0
    spokes0 = []                                       # three spokes in the rod's own frame, one toward the contact
    c0, p0 = T["c_P"][k0], T["p"][k0]
    u = (p0 - c0) / np.linalg.norm(p0 - c0)
    ax0 = T["a_P"][k0] / np.linalg.norm(T["a_P"][k0])
    for ang in (0.0, 120.0, 240.0):
        r = np.radians(ang)
        v = u * np.cos(r) + np.cross(ax0, u) * np.sin(r)
        spokes0.append(T["R_rod"][k0].T @ v)
    frames_dir = WORK / "film"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("f*.png"):
        f.unlink()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 15,
                         "mathtext.fontset": "dejavuserif"})
    G, B, K = "#00a33c", "#1f4fd8", "#c0392b"
    from scipy.spatial import ConvexHull
    yz = basis["pts"][:, [1, 2]]
    hull = yz[ConvexHull(yz).vertices]
    mc, _ = W.magnets(W.fe_row("four_05R4"))
    t_hold = float(T["plateaus"][0][0] - T["t_lo"])          # the press's level hold: time zero of the traces
    xs = tt - t_hold

    def est(i, key):
        rw = rows[i][key]
        return np.asarray(rw["p"], float) if rw["contact"] else None
    for j, i in enumerate(keep):
        if a.only and j not in a.only:
            continue
        t = rows[i]["t"]
        k = int(np.clip(np.searchsorted(tR, t), 0, len(tR) - 1))
        c, p, f = T["c_P"][k], T["p"][k], T["f_P"][k]
        fig = plt.figure(figsize=(7.2, 10.8), dpi=100)
        fig.text(0.04, 0.972, "A rod rolled on a sensing fingertip", fontsize=23, weight="bold", va="top")
        fig.text(0.04, 0.932, "UR5 calibration rig, 4 November: v3 ID 5 pad up, a rod of radius\n12.5 mm pressed to 4 N "
                 "and turned to +6, -6 and back to 0 deg.\nReplayed at 4x speed beside the fingertip viewer (right).",
                 fontsize=14.5, va="top", color="#333333", linespacing=1.35)
        # (a) the roll across the rod, the pad's face up: z along the finger to the right, x the face's normal up
        ax = fig.add_axes([0.035, 0.455, 0.585, 0.355])
        ax.add_patch(Polygon(poly[:, [1, 0]], closed=True, fc="#f3dfc1", ec="#8a6a3c", lw=1.6))
        ax.add_patch(Circle((c[2], c[0]), R, fc="#dfe3e8", ec="#555555", lw=1.6))
        for s0 in spokes0:
            v = T["R_rod"][k] @ s0
            ax.plot([c[2], c[2] + R * v[2]], [c[0], c[0] + R * v[0]], color="#6b7280", lw=1.4)
        sc = 0.8                                        # mm per N for the force arrow
        ax.annotate("", xy=(p[2], p[0]), xytext=(p[2] - sc * f[2], p[0] - sc * f[0]),
                    arrowprops=dict(arrowstyle="-|>", color=G, lw=2.4, mutation_scale=18), zorder=6)
        sel = (tR > t - 4.0) & (tR <= t)
        ax.plot(T["p"][sel, 2], T["p"][sel, 0], color=G, lw=4, alpha=0.45, solid_capstyle="round", zorder=5)
        ax.plot(p[2], p[0], "o", color=G, ms=9, mec="white", mew=1.3, zorder=7)
        ax.set_xlim(3.0, 18.0)
        ax.set_ylim(0.5, 15.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.0, 1.035, "(a) The roll, from the robot's record", transform=ax.transAxes, fontsize=15)
        ax.text(0.02, 0.97, f"turn {th[j]:+.1f}°", transform=ax.transAxes, fontsize=15, va="top")
        ax.text(16.6, 10.2, "rod", ha="center", fontsize=14, color="#374151")
        ax.text(4.2, 3.2, "pad", ha="left", fontsize=14, color="#6b4f2a")
        ax.plot([14.5, 16.5], [1.6, 1.6], color="k", lw=2.2)
        ax.text(15.5, 2.0, "2 mm", ha="center", fontsize=12)
        ax.text(0.98, 0.97, f"{np.linalg.norm(f):.1f} N", transform=ax.transAxes, fontsize=14, va="top", ha="right",
                color=G)
        # (b) the three contact points seen from the palm
        axf = fig.add_axes([0.655, 0.455, 0.325, 0.355])
        axf.add_patch(Polygon(hull, closed=True, fc="#f7ecdb", ec="#8a6a3c", lw=1.4))
        for m0 in mc:
            axf.add_patch(Circle((m0[1], m0[2]), 1.5, fc="none", ec="#9ca3af", lw=1.0, ls="--"))
        lo = max(0, j - 12)                             # the last 2 s of each estimate
        for key, col, mk in (("physical", K, "x"), ("gp", B, "+")):
            pts = [est(keep[q], key) for q in range(lo, j + 1)]
            pts = np.array([q for q in pts if q is not None])
            if len(pts):
                axf.plot(pts[:, 1], pts[:, 2], color=col, lw=1.2, alpha=0.35)
            e = est(i, key)
            if e is not None:
                axf.plot(e[1], e[2], mk, color=col, ms=13, mew=3.0, zorder=6)
        axf.plot(T["p"][sel, 1], T["p"][sel, 2], color=G, lw=3, alpha=0.45)
        axf.plot(p[1], p[2], "o", color=G, ms=9, mec="white", mew=1.2, zorder=7)
        axf.set_xlim(-8.6, 8.6)
        axf.set_ylim(-0.3, 22.2)
        axf.set_aspect("equal")
        axf.axis("off")
        axf.text(0.0, 1.035, "(b) From the palm", transform=axf.transAxes, fontsize=15)
        axf.text(0.5, -0.02, "tip end up; dashed: magnets", transform=axf.transAxes, fontsize=11.5, ha="center",
                 va="top", color="#4b5563")
        # the distances now
        y_txt = 0.415
        for key, col, lab in (("physical", K, "× physical model"), ("gp", B, "+ GP of the UR5 calibration")):
            e = est(i, key)
            val = f"{np.linalg.norm(e - p):.2f} mm from the robot's contact" if e is not None else "no contact shown"
            fig.text(0.06, y_txt, f"{lab}: {val}", fontsize=14, color=col)
            y_txt -= 0.03
        fig.text(0.06, y_txt, "● the robot's contact and the HEX-E's force", fontsize=14, color=G)
        # (c) the distance of each estimate from the robot's contact over the roll
        ax2 = fig.add_axes([0.13, 0.15, 0.83, 0.16])
        cut = 8.0
        for d, col in ((d_ph, K), (d_gp, B)):
            ax2.plot(xs, np.minimum(d, cut), ".", color=col, ms=3.5, alpha=0.25)
            ax2.plot(xs[:j + 1], np.minimum(d[:j + 1], cut), ".", color=col, ms=5)
        ax2.axvline(t - t_hold, color="k", lw=1.0)
        ax2.set_ylim(0, cut + 0.3)
        ax2.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
        ax2.set_ylabel("Distance (mm)", fontsize=14)
        ax2.tick_params(labelbottom=False, labelsize=12)
        med_ph, med_gp = np.nanmedian(d_ph[:j + 1]), np.nanmedian(d_gp[:j + 1])
        ax2.text(0.0, 1.07, "(c) Distance from the robot's contact over the roll", transform=ax2.transAxes,
                 fontsize=14.5)
        if np.isfinite(med_ph):
            ax2.text(0.99, 0.80, f"median so far {med_ph:.1f} and {med_gp:.2f} mm", transform=ax2.transAxes,
                     fontsize=12.5, ha="right", va="top", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ax2.set_yticks([0, 2, 4, 6, 8])
        ax2.set_yticklabels(["0", "2", "4", "6", "≥8"])
        ax3 = fig.add_axes([0.13, 0.055, 0.83, 0.075], sharex=ax2)
        ax3.plot(xs, th, color="#6b7280", lw=1.6)
        ax3.axvline(t - t_hold, color="k", lw=1.0)
        ax3.set_ylim(-8, 8)
        ax3.set_yticks([-6, 0, 6])
        ax3.set_ylabel("Turn (°)", fontsize=13)
        ax3.set_xlabel("Time from the press's level hold (s)", fontsize=13)
        ax3.tick_params(labelsize=12)
        for axx in (ax2, ax3):
            axx.spines[["top", "right"]].set_visible(False)
        fig.canvas.draw()
        left = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        right = np.asarray(Image.open(WORK / "viewer" / f"v{j:05d}.png").convert("RGB"))
        out = np.full((1080, 1920, 3), 255, np.uint8)
        out[:, :720] = left[:1080, :720]
        out[:right.shape[0], 720:720 + right.shape[1]] = right[:1080, :1200]
        out[:, 719] = 0
        Image.fromarray(out).save(frames_dir / f"f{j:05d}.png")
        if j % 50 == 0:
            print(f"  composed {j + 1}/{len(keep)}", flush=True)
    if a.only:
        return
    MEDIA.mkdir(parents=True, exist_ok=True)
    dst = MEDIA / f"{STEM}_viewer_film.mp4"
    last = len(keep) - 1
    for h in range(1, 38):                              # 1.5 s on the last frame
        (frames_dir / f"f{last + h:05d}.png").write_bytes((frames_dir / f"f{last:05d}.png").read_bytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25", "-i", str(frames_dir / "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high", "-crf", "23", "-preset", "slow",
                    "-movflags", "+faststart", str(dst)], check=True)
    both = np.flatnonzero(np.isfinite(d_ph) & np.isfinite(d_gp))
    jp = int(both[np.argmin(np.abs(tt[both] - a.poster_t))])
    Image.open(frames_dir / f"f{jp:05d}.png").save(dst.with_suffix(".jpg"), quality=90)
    print(f"wrote {dst.relative_to(ROOT)} ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)
    s = dict(t_start=a.t_start, n_frames=len(keep), step_s=D["step"],
             physical=dict(median_mm=float(np.nanmedian(d_ph)), p90_mm=float(np.nanpercentile(d_ph, 90)),
                           frames_with_contact=int(np.isfinite(d_ph).sum())),
             gp=dict(median_mm=float(np.nanmedian(d_gp)), p90_mm=float(np.nanpercentile(d_gp, 90)),
                     frames_with_contact=int(np.isfinite(d_gp).sum())))
    json.dump(s, open(DATA / f"{STEM}_film_summary.json", "w"), indent=1)
    print(json.dumps(s), flush=True)


def table(a):
    """One row per film frame into DATA: the truth and the three estimates (the viewer's own rest fit with the board's
    stored chip frames, the full rest calibration of this board, the GP), for the page (scripts/fingertip/
    viewer_roll_page.py)."""
    T = np.load(DATA / f"{STEM}_truth.npz")
    t_hold = float(T["plateaus"][0][0] - T["t_lo"])
    tR = T["t"] - float(T["t_lo"])
    A = json.load(open(WORK / "frames_step0.16_az.json"))
    B = json.load(open(WORK / "frames_step0.16_az_calib.json"))
    out = []
    for ra, rb in zip(A["rows"], B["rows"]):
        if rb["t"] < a.t_start or rb["truth"] is None:
            continue
        r3 = (lambda v: None if v is None else [round(float(x), 3) for x in v])
        est = {}
        for name, src in (("physical_stored_frames", ra["physical"]), ("physical", rb["physical"]), ("gp", rb["gp"])):
            est[name] = dict(contact=src["contact"], p=r3(src["p"]), fn=None if src["fn"] is None else round(src["fn"], 3))
        out.append(dict(t=round(rb["t"] - t_hold, 3), theta=round(float(np.interp(rb["t"], tR, T["theta"])), 3),
                        p=r3(rb["truth"]["p"]), f=r3(rb["truth"]["f"]), **est))
    path = DATA / f"{STEM}_frames.json"
    json.dump(dict(what="per film frame: time from the press's level hold (s), the rod's turn (deg), the robot's contact "
                        "p (mm, pad frame) and the HEX-E's force f (N, on the pad), and each estimate shown by the viewer",
                   t_hold_window_s=t_hold, step_s=B["step"], t_cal_window_s=B["t_cal"], rows=out), open(path, "w"))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e3:.0f} kB, {len(out)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    fr = sub.add_parser("frames")
    fr.add_argument("--step", type=float, default=0.16, help="s of the recording between film frames (4x at 25 fps)")
    fr.add_argument("--auto-zero", type=float, default=3.0, help="the viewer's --auto-zero (0 turns it off)")
    fr.add_argument("--from-rest", action="store_true", help="start at the end of the rest, before the press")
    fr.add_argument("--calib", default=None, help="a saved calibration for the physical model (the viewer's --calib)")
    cp = sub.add_parser("capture")
    cp.add_argument("--frames", default="frames_step0.16_az_calib.json", help="frames() output in logs/")
    cp.add_argument("--n", type=int, default=None, help="only the first n frames")
    cp.add_argument("--width", type=int, default=1200)
    cp.add_argument("--height", type=int, default=1080)
    cp.add_argument("--cam", default="[[33.3, 21.0, -21.5], [10.4, 3.0, -1.0]]", help="camera position and target "
                    "(three's axes: tip z, x, y); the viewer's default is [[44, 26, 30], [10.5, 1, 0]]")
    cp.add_argument("--t-start", type=float, default=14.0, help="s of the window: earlier frames are fed, not captured")
    cp.add_argument("--arrow", type=float, default=2.0, help="the page's arrow scale, mm/N")
    co = sub.add_parser("compose")
    co.add_argument("--frames", default="frames_step0.16_az_calib.json")
    co.add_argument("--t-start", type=float, default=14.0)
    co.add_argument("--poster-t", type=float, default=31.3, help="s of the window: the poster is the frame nearest it "
                    "with a contact in both rows")
    co.add_argument("--only", type=int, nargs="*", default=None, help="compose only these frames (a look), no film")
    tb = sub.add_parser("table")
    tb.add_argument("--t-start", type=float, default=14.0)
    a = ap.parse_args()
    dict(prepare=prepare, frames=frames, capture=capture, compose=compose, table=table)[a.cmd](a)


if __name__ == "__main__":
    main()
