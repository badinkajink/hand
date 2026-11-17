#!/usr/bin/env python3
"""The magnetic fingertips' contact estimates beside the camera on the hand's trials of 2026-11-16 (owner, 2026-11-16
23:5x: "can we make a video that synchronizes the magnetic fingertip visualization with a real trial from today? it can be
the twist ..., the press, the pick up from dropped, (maybe all of them) just so i can pick and choose"; "please do a
center crop on the core manipulation ... and then show just the critical visualization of the contact sensors").

Three boards sit under the SR2 hand's v3 tips (mux channels 0 middle, 1 index, 3 thumb; logs/20261116-three_finger_mux/
*-mux.bin, copied from corvus). Each board's contact comes from the trials' estimator (src/sr2hand/driver/reskin5x/host/
tip_contacts.py Tip, as hom_trial.py runs it from 2026-11-17; owner, 2026-11-17: "yes, please re-render those films"):
the tip's calibration of 11-02 or a new fit at the rest before the close, whichever fits the rest better, the UR5 board
GP for the point and the force, and the change its own finger's pose makes to the reading taken off. The viewer's page
(wrench_viewer.py) draws each tip from it. Until 02:2x on 11-17 the films drew the viewer's own uncalibrated estimate, the
physical model fitted at the rest. This script crops the page's 3D views and sets them beside a crop of the trial's
camera, frame by frame on corvus's clock, which the mux records, the trial records and the cameras' frame times carry.

    ~/miniconda3/bin/python scripts/fingertip/hand_trial_tips_film.py frames --trial press     (the viewer's messages)
    .venv/bin/python scripts/fingertip/hand_trial_tips_film.py capture --trial press          (headless Chrome)
    .venv/bin/python scripts/fingertip/hand_trial_tips_film.py compose --trial press          (the film)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

BENCH = ROOT / "docs/experiments/20261115-hom_bench_trials"
MUX = ROOT / "logs/20261116-three_finger_mux"
RAW = ROOT / "logs/20261116-fingertip_film/raw"
WORK = ROOT / "logs/20261116-fingertip_film"
MEDIA = BENCH / "media"
VIEWER = ROOT / "src/sr2hand/driver/reskin5x/host"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
NB, CHIPS = 3, 5                                   # boards on the mux log, chips per board
BOARD = {"thumb": 2, "index": 1, "middle": 0}       # the log's board index of each finger (channels 3, 1 and 0)
SHOW = ("thumb", "index", "middle")
FPS = 25


def record_t0(stem):
    """The trial record's start (corvus's clock, s) and its last row's time."""
    rows = [json.loads(x) for x in (BENCH / "runs_corvus" / f"{stem}.jsonl").read_text().splitlines() if x.strip()]
    return rows[0]["t0_ms"] / 1e3, max(r["t_ms"] for r in rows if "t_ms" in r) / 1e3


def telemetry_t0(name):
    rows = [json.loads(x) for x in (BENCH / "data/20261116-pinch_pickup" / name).read_text().splitlines() if x.strip()]
    return rows[0]["t"]


def trial(name):
    """A trial's film: its mux log, the rest before the close, and its segments (video stem, from, to on corvus's clock,
    speed), with the camera and its crop (x, y, w, h of the 960 x 540 frame)."""
    if name == "press":
        t0, t1 = record_t0("20261116-131927_press_corvus")
        return dict(mux="20261116-131831-mux.bin", rest=t0 - 0.3, cam="832412070344", crop=(190, 0, 450, 540),
                    title="Press of the 25 mm cylinder, 16 November 13:19",
                    sub="Layout (70, 110) mm, the thumb-index pinch at the axis, HOM on the tracked pose; real time.",
                    segs=[("20261116-131927_trial_press_x70_y110_lat0", t0 - 0.5, t1 + 0.5, 1.0)])
    if name == "twist":
        t0, t1 = record_t0("20261116-161252_twist_corvus")
        return dict(mux="20261116-154014-mux.bin", rest=t0 - 0.3, cam="832412070344", crop=(40, 0, 450, 540),
                    title="Twist of the 40 mm sphere, 16 November 16:12",
                    sub="Layout (105, 70) mm, grasp 4, HOM on the tracked pose; the servos lost power mid-lift. Real time.",
                    segs=[("20261116-161252_trial_gr_g4_blind1", t0 - 0.5, t1 + 0.5, 1.0)])
    if name == "pickup":
        tg = telemetry_t0("20261116-154328_pinch_grasp_telemetry.jsonl")
        tl = telemetry_t0("20261116-154429_pinch_lift_telemetry.jsonl")
        ts = telemetry_t0("20261116-154534_pinch_slip_telemetry.jsonl")
        p = "20261116-pinch_pickup_thumb_middle_plan"
        return dict(mux="20261116-154014-mux.bin", rest=None, rest_after=tg, cam="832412070344", crop=(30, 0, 450, 540),
                    title="Pick-up of the lying 40 mm sphere, 16 November 15:43",
                    sub="Thumb against middle: the close (4x), the lift that stands the sphere up (2x), the hang and "
                        "the squeeze lowered in steps (2x).",
                    segs=[(f"20261116-154323_pinch_grasp_{p}", tg, tg + 40.0, 4.0),
                          (f"20261116-154425_pinch_lift_{p}", tl, tl + 22.0, 2.0),
                          (f"20261116-154531_pinch_slip_{p}", ts, ts + 16.0, 2.0)])
    raise SystemExit(f"unknown trial {name}")


def film_times(T):
    """The film's frames: per frame the segment index and the record time (corvus's clock, s)."""
    out = []
    for k, (_stem, a, b, speed) in enumerate(T["segs"]):
        n = int(round((b - a) / speed * FPS))
        out += [(k, a + i * speed / FPS) for i in range(n)]
    return out


def mux_window(T, lo, hi):
    dt = np.dtype([("t", "<f8"), ("r", "<f4", (NB * CHIPS * 4,))])
    d = np.fromfile(MUX / T["mux"], dtype=dt)
    sel = (d["t"] >= lo) & (d["t"] <= hi)
    return d["t"][sel].astype(float), d["r"][sel].reshape(-1, NB, CHIPS, 4).astype(float)


def first_change(t, r, after, quiet_uT=8.0):
    """The first time after `after` that any board's field leaves its mean over the second after `after` by quiet_uT
    (0.1 s means): the close."""
    b = r[:, :, :, 1:].reshape(len(t), NB, 15)
    base = b[(t >= after) & (t < after + 1.0)].mean(0)
    for tt in np.arange(after + 1.0, t[-1], 0.1):
        sel = (t >= tt) & (t < tt + 0.1)
        if sel.sum() and np.max(np.linalg.norm(b[sel].mean(0) - base, axis=1)) > quiet_uT:
            return float(tt)
    return float(t[-1])


# ------------------------------------------------------------------------------------------------ frames
def joints_of(T, a_trial):
    """The fingers' joints over the trial (simulator degrees per finger, yaw, mcp, pip) and their times (corvus's clock,
    s): the trial record's servo readings (tip_contacts.sim_deg), or the pick-up's telemetry."""
    import tip_contacts as TC
    if a_trial == "pickup":
        rows = []
        for n_ in ("20261116-154328_pinch_grasp_telemetry.jsonl", "20261116-154429_pinch_lift_telemetry.jsonl",
                   "20261116-154534_pinch_slip_telemetry.jsonl"):
            rows += [json.loads(x) for x in (BENCH / "data/20261116-pinch_pickup" / n_).read_text().splitlines()
                     if x.strip()]
        rows = [r for r in rows if r.get("joints")]
        tq = np.array([r["t"] for r in rows])
        q = {f: np.array([[r["joints"][f][j] for j in ("yaw", "mcp", "pip")] for r in rows], float) for f in SHOW}
        return tq, q
    stem = {"press": "20261116-131927_press_corvus", "twist": "20261116-161252_twist_corvus"}[a_trial]
    rows = [json.loads(x) for x in (BENCH / "runs_corvus" / f"{stem}.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if "t_ms" in r and r.get("joints") is not None]
    tq = np.array([r.get("t_joints_ms", r["t_ms"]) for r in rows]) / 1e3
    sd = [TC.sim_deg(np.asarray(r["joints"], float)) for r in rows]
    return tq, {f: np.array([s_[f] for s_ in sd], float) for f in SHOW}


def tip_message(app, m, e):
    """The viewer's message m for its page with the trial estimator's contact e in place of the viewer's own: the point,
    normal and force, and from them the page's pressure and deformation (wrench_viewer.App.frame's)."""
    import wrench_viewer as WV
    m = {k: v for k, v in m.items() if k not in ("p", "n", "t1", "t2", "f", "fn", "ft", "sd", "ell", "u", "pr", "pmax",
                                                "where", "gp_w", "corrected")}
    m["contact"] = bool(e.get("contact"))
    if not m["contact"]:
        return m
    p, n, f = (np.asarray(e[k], float) for k in ("p", "n", "f"))
    fn = float(e["f_n"])
    t1 = np.cross(n, [0.0, 0.0, 1.0])
    t1 = t1 if np.linalg.norm(t1) > 1e-3 else np.cross(n, [0.0, 1.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    pres = app.pads.press(p, n, fn, app.R_ind)
    dist, idx = app.tree.query(p, k=6)
    w = np.where(app.basis["normals"][idx] @ n > 0.3, 1 / np.maximum(dist, 1e-3) ** 2, 0.0)
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
    U = np.einsum("p,pkvc,k->vc", w, app.disp_u[idx].astype(np.float32), f)
    m.update(p=np.round(p, 3).tolist(), n=np.round(n, 4).tolist(), t1=np.round(t1, 4).tolist(), t2=np.round(t2, 4).tolist(),
             f=np.round(f, 3).tolist(), fn=round(fn, 3), ft=[round(float(f @ t1), 3), round(float(f @ t2), 3)],
             sd=[0.0] * 6, ell=[0.35, 0.35, 0.0], u=np.round(U.ravel(), 1).tolist(), pr=np.round(pres, 1).tolist(),
             pmax=float(pres.max()), where=WV.region(n), gp_w=e.get("w"))
    return m


def frames(a):
    """Every film frame's contact per board from the trial estimator (src/sr2hand/driver/reskin5x/host/tip_contacts.py,
    as the trials of 2026-11-17 on: each tip's calibration of 11-02 against a new fit at the rest, the UR5 board GP, the
    own-pose term from the finger's measured joints), drawn by the viewer's page: the viewer's App gives the page's
    other fields."""
    from viewer_roll_film import Clock, viewer_args
    sys.path.insert(0, str(VIEWER))
    import tip_contacts as TC
    import wrench_viewer as WV
    T = trial(a.trial)
    ft = film_times(T)
    lo = min(s_[1] for s_ in T["segs"]) - 6.0
    hi = max(s_[2] for s_ in T["segs"]) + 1.0
    t, r = mux_window(T, lo - (60.0 if T.get("rest_after") else 0.0), hi)
    rest = T["rest"]
    if rest is None:                                   # the pick-up: the rest just before the close
        rest = first_change(t, r, T["rest_after"]) - 0.3
    tq, Q = joints_of(T, a.trial)

    def q_at(f, tt):
        return Q[f][int(np.argmin(np.abs(tq - tt)))]

    basis, row, kappa = TC.W.load_basis(TC.W.basis_path(TC.CASE)), TC.W.fe_row(TC.CASE), TC.W.load_thermal()
    gp, own = TC.BoardGP(), TC.OwnPose()
    print(f"{a.trial}: {len(ft)} frames, {len(t)} records, rest at {time.strftime('%H:%M:%S', time.localtime(rest))}",
          flush=True)
    apps, tips, info = {}, {}, {}
    for f in SHOW:
        src = Clock(t, r[:, BOARD[f]], keep=3.0)
        app = WV.App(viewer_args(auto_zero=0.0), src=src, name=f, board_gp=None)
        src.now = rest
        app.calibrate()
        apps[f] = app
        tips[f] = TC.Tip(f, basis, row, gp, kappa, None, own=own)
        sel = (t > rest - 1.0) & (t <= rest)
        info[f] = tips[f].set_rest(r[sel, BOARD[f]], refit=True, q=q_at(f, rest))
        print(f, {k: info[f][k] for k in ("model", "saved_fit_mT", "new_fit_mT", "calib") if k in info[f]}, flush=True)
    msgs = {f: [] for f in SHOW}
    rows = []
    for k, tt in ft:
        row_ = dict(seg=k, t=round(tt, 3))
        sel = (t > tt - TC.KEEP_S) & (t <= tt)
        for f in SHOW:
            apps[f].src.now = float(tt)
            e = tips[f].estimate(t[sel], r[sel, BOARD[f]], q=q_at(f, tt))
            m_ = tip_message(apps[f], apps[f].frame(), e)
            msgs[f].append(m_)
            row_[f] = dict(contact=bool(e.get("contact")), fn=e.get("f_n"), p=e.get("p"), w=e.get("w"),
                           own_uT=e.get("own_uT"), d_uT=e.get("d_uT"))
        rows.append(row_)
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / f"{a.trial}_frames.json"
    json.dump(dict(scenes={f: app.scene() for f, app in apps.items()}, msgs=msgs, rows=rows, rest=rest, rest_info=info,
                   estimator="tip_contacts.Tip: saved calibration or new fit, board GP, own pose"), open(out, "w"))
    for f in SHOW:
        n_ = sum(rw[f]["contact"] for rw in rows)
        fn = [rw[f]["fn"] for rw in rows if rw[f]["contact"]]
        print(f"{f}: contact in {n_} of {len(rows)} frames" + (f", normal force median {np.median(fn):.2f} N, max "
                                                              f"{max(fn):.2f} N" if fn else ""), flush=True)
    print(f"wrote {out.relative_to(ROOT)}", flush=True)


# ------------------------------------------------------------------------------------------------ capture
def capture(a):
    """Screenshots of the viewer's page with the three boards, one per film frame."""
    import asyncio
    import functools
    import http.server
    import threading
    from viewer_roll_film import cdp_capture, harness_html
    D = json.load(open(WORK / f"{a.trial}_frames.json"))
    site = WORK / "site"
    site.mkdir(parents=True, exist_ok=True)
    (site / "harness.html").write_text(harness_html())
    out_dir = WORK / f"{a.trial}_viewer"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("v*.png"):
        f.unlink()
    n = len(D["rows"]) if a.n is None else min(a.n, len(D["rows"]))
    names = [f.capitalize() for f in SHOW]
    first = [json.dumps(dict(type="boards", names=names))]
    first += [json.dumps(dict(D["scenes"][f], board=k, name=names[k])) for k, f in enumerate(SHOW)]
    jobs = []
    for i in range(0, n, a.every):
        lines = first + [None] if i == 0 else []
        for k, f in enumerate(SHOW):
            m = dict(D["msgs"][f][i], board=k)
            m["status"] = ""
            lines.append(json.dumps(m))
        jobs.append("".join(f"window.__feed({json.dumps(s)});" if s is not None else
                            f"document.getElementById('ar').value = {a.arrow:g}; "
                            f"document.getElementById('arl').textContent = '{a.arrow:g}';" for s in lines))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    handler.log_message = lambda *x, **y: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/harness.html"
    cams = [board_cam(D, f, a.dist) for f in SHOW]
    t0 = time.time()
    try:
        asyncio.run(cdp_capture(url, jobs, (a.width, a.height), out_dir, cams))
    finally:
        httpd.shutdown()
    print(f"{len(jobs)} screenshots in {out_dir.relative_to(ROOT)}, {time.time() - t0:.0f} s", flush=True)


def board_cam(D, f, dist):
    """A camera (position, target in the page's frame) on the board's contacts: dist mm out along their median normal
    (component-wise; a mean drifts to the tip end when a few contacts sit there), tilted a little toward the finger's
    axis, aimed at their median point; the palmar face when the board shows none."""
    ms = [m for m in D["msgs"][f] if m.get("contact")]
    face = [m for m in ms if m.get("where") != "tip end"]
    end = [m for m in ms if m.get("where") == "tip end"]
    if len(ms) >= 5 and min(len(face), len(end)) >= 0.25 * len(ms):
        # the estimate jumps between the palmar face and the tip end (a weak contact, two fits): between the two, farther
        p = 0.5 * (np.median([m["p"] for m in face], 0) + np.median([m["p"] for m in end], 0))
        n = np.median([m["n"] for m in face], 0) + np.median([m["n"] for m in end], 0)
        dist *= 1.35
    elif len(ms) >= 5:
        p = np.median([m["p"] for m in ms], 0)
        n = np.median([m["n"] for m in ms], 0)
    else:
        p, n = np.array([7.0, 0.0, 12.0]), np.array([1.0, 0.0, 0.3])
    n = n / np.linalg.norm(n)
    v = n + np.array([0.0, 0.15, 0.25])                  # tilted toward the tip end and one side
    c = p + dist * v / np.linalg.norm(v)
    page = (lambda q: [round(float(q[2]), 2), round(float(q[0]), 2), round(float(q[1]), 2)])   # tip -> three (z, x, y)
    return page(c), page(p)


# ------------------------------------------------------------------------------------------------ compose
def video_frames(stem, cam):
    """The segment's video frames as files (extracted once) and their host times (s), row k = the mp4's frame k."""
    import csv
    d = WORK / "video" / f"{stem}_{cam}"
    if not d.exists():
        d.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(RAW / f"{stem}_{cam}.mp4"), "-vsync", "0", "-q:v", "2",
                        str(d / "%05d.jpg")], check=True)
    rows = list(csv.DictReader(open(RAW / f"{stem}_{cam}_frames.csv")))
    files = sorted(d.glob("*.jpg"))
    n = min(len(rows), len(files))
    return files[:n], np.array([float(r["t_host_ms"]) / 1e3 for r in rows[:n]])


def trace_image(D, T, size, ylabel="estimated normal force (N)"):
    """The three fingers' estimated normal force over the film (0 without a contact), drawn once; returns the image and
    the x pixel of each film frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    rows = D["rows"]
    x = np.arange(len(rows)) / FPS
    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    for f, c in zip(SHOW, COLS):
        y = [rw[f]["fn"] if rw[f]["contact"] else 0.0 for rw in rows]
        ax.plot(x, y, color=c, lw=2.2, label=f)
    for k in range(1, len(T["segs"])):
        b = next(i for i, rw in enumerate(rows) if rw["seg"] == k) / FPS
        ax.axvline(b, color="#9ca3af", lw=1, ls=(0, (4, 3)))
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, max(1.0, 1.15 * max([rw[f]["fn"] or 0 for rw in rows for f in SHOW if rw[f]["contact"]] or [1])))
    ax.set_xlabel("film time (s)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#eef0f3")
    fig.tight_layout()
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    x0, x1 = (ax.transData.transform((v, 0))[0] for v in (0, x[-1]))
    plt.close(fig)
    return img, (x0, x1), ax


COLS = ("#d97706", "#7c3aed", "#0f766e")             # thumb, index, middle
SEG_NAMES = {"pickup": ("the close, 4x", "the lift, 2x", "the hang and the slip, 2x")}


def compose(a):
    """The film: the camera's crop on the left, the viewer's three tips and the force traces on the right."""
    from PIL import Image, ImageDraw, ImageFont
    T = trial(a.trial)
    D = json.load(open(WORK / f"{a.trial}_frames.json"))
    rows = D["rows"]
    shots = sorted((WORK / f"{a.trial}_viewer").glob("v*.png"))
    if len(shots) != len(rows):
        raise SystemExit(f"{len(shots)} viewer screenshots for {len(rows)} frames: capture first")
    vids = [video_frames(stem, T["cam"]) for stem, *_ in T["segs"]]
    W_, H_ = 1920, 1080
    F = {s: ImageFont.truetype(FONT, s) for s in (17, 18, 20, 24)}
    FB = {s: ImageFont.truetype(FONT_B, s) for s in (24, 30)}
    cw = 500                                           # a tip's cell on the right
    bw, bh = VIEW_BOX(0)[2] - VIEW_BOX(0)[0], VIEW_BOX(0)[3] - VIEW_BOX(0)[1]
    vh = int(round(cw * bh / bw))
    cells = [(920, 140), (1430, 140), (920, 140 + vh + 44)]      # thumb, index, middle; the traces in the fourth
    trace, (x0, x1), _ax = trace_image(D, T, (cw, vh + 34))
    out = WORK / f"{a.trial}_film"
    out.mkdir(exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    x, y, w, h = T["crop"]
    names = SEG_NAMES.get(a.trial)
    head = wrap(T["sub"] + " Each tip: the trials' contact estimate from its magnet board (the tip's UR5 calibration and "
                "board GP, its own finger's crosstalk taken off), drawn by the fingertip viewer. Colour: pad pressure to "
                "300 kPa; blue arrow: the estimated force.", F[17], 975)
    for i, rw in enumerate(rows):
        im = Image.new("RGB", (W_, H_), "white")
        files, tv = vids[rw["seg"]]
        j = int(np.argmin(np.abs(tv - rw["t"])))
        cam = Image.open(files[j]).convert("RGB").crop((x, y, x + w, y + h)).resize((900, 1080), Image.LANCZOS)
        im.paste(cam, (0, 0))
        dr = ImageDraw.Draw(im)
        lab = "the hand" + (f": {names[rw['seg']]}" if names else ", real time")
        dr.rectangle((0, 0, 900, 40), fill="white")
        dr.text((12, 7), lab, font=F[24], fill="#141414")
        dr.text((930, 12), T["title"], font=FB[30], fill="#0f766e")
        for k, line in enumerate(head):
            dr.text((930, 54 + 21 * k), line, font=F[17], fill="#374151")
        shot = Image.open(shots[i]).convert("RGB")
        for k, f in enumerate(SHOW):
            view = shot.crop(VIEW_BOX(k)).resize((cw, vh), Image.LANCZOS)
            cx, cy = cells[k]
            im.paste(view, (cx, cy + 34))
            dr.rectangle((cx, cy + 34, cx + cw - 1, cy + 34 + vh - 1), outline="#d1d5db")
            m = rw[f]
            txt = f"{f}: {m['fn']:.1f} N" if m["contact"] and m["fn"] is not None else f"{f}: no contact shown"
            dr.text((cx + 2, cy + 2), txt, font=FB[24], fill=COLS[k])
        tx, ty = 1430, cells[2][1]
        im.paste(trace, (tx, ty))
        px = tx + x0 + (x1 - x0) * i / max(1, len(rows) - 1)
        dr.line((px, ty + 8, px, ty + vh - 30), fill="#111827", width=2)
        im.save(out / f"{i:05d}.png")
    film = MEDIA / f"20261116-fingertips_{a.trial}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(out / "%05d.png"), "-c:v", "libx264",
                    "-crf", "20", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(film)],
                   check=True)
    k = len(rows) // 2
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{k / FPS:.2f}", "-i", str(film), "-frames:v", "1", "-q:v", "3",
                    str(film.with_suffix(".jpg"))], check=True)
    print(f"wrote {film.relative_to(ROOT)} ({len(rows) / FPS:.1f} s)", flush=True)


def wrap(text, font, width):
    """The text in lines no wider than width pixels."""
    lines, cur = [], ""
    for w_ in text.split():
        t = (cur + " " + w_).strip()
        if font.getlength(t) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w_
    return lines + [cur]


# ------------------------------------------------------------------------------------------------ palm
# owner, 2026-11-17 00:1x: "for the press fingertips video, can we try being fancy and visualizing the fingertips overlaid
# on the fingertips, with a eye-in-palm perspective? the cylinder can be semi-transparent". The pad frame of the fingertip
# calibrations (the viewer's and the estimator's: x out of the palmar face, y across, z from the mount end to the tip
# end, mm) onto the model's tip body by tip_contacts.T_tip_pad: x_tip = s (x + 2.05), y_tip = -s y, z_tip = 10.86 - z,
# s the finger's face sign (+1 thumb, -1 index and middle). Until 01:1x on 11-17 the films used a map fitted here on
# the pad centres (the same turn and z, without the 2.05 mm along x).
PLAN = {"press": ("docs/experiments/20261115-hom_bench_trials/plans/20261116-press_x70_y110_lat0_plan.json",
                  "20261116-131927_press_corvus", 25.0)}


def tip_map(f):
    """(R, t): the pad frame (mm) in the model's tip body (mm), tip_contacts.T_tip_pad."""
    sys.path.insert(0, str(VIEWER))
    import tip_contacts as TC
    Tm = TC.T_tip_pad(f)
    return Tm[:3, :3], Tm[:3, 3]


def viewer_to_body(f, p_mm):
    """A point of the pad frame (mm) in the model's tip body (m)."""
    R, t = tip_map(f)
    return 1e-3 * (R @ np.asarray(p_mm, float) + t)


PAD_GREY = (0.84, 0.85, 0.89, 0.28)                    # untouched pads translucent, so a patch behind them shows


def pressure_rgba(kpa):
    """Pressure to 300 kPa from light to dark purple (the pad zoom's: darker, more force), light grey below 2 kPa."""
    w = np.clip(kpa / 300.0, 0.0, 1.0)[:, None] ** 0.6
    c = np.c_[(1 - w) * np.array([[0.78, 0.66, 0.98]]) + w * np.array([[0.33, 0.04, 0.70]]), np.ones(len(kpa))]
    c[kpa < 2.0] = PAD_GREY
    return c


def pad_sizes(base, kpa):
    """Pressed pads drawn 1.8 times larger than the rest."""
    return base * np.where(kpa >= 2.0, 1.8, 1.0)[:, None]


CHAIN = ("mount", "yaw_frame", "mcp_frame", "len_frame", "pip_frame")   # hom_abstract_film.CHAIN
LINK_RGBA = np.array([0.69, 0.71, 0.75, 1.0], np.float32)                 # the abstract films' LINK, #B0B5BF
JOINT_RGBA = np.array([0.49, 0.23, 0.93, 1.0], np.float32)                # their NEW, #7C3AED


def add_geom(scn, kind, rgba, width, p0, p1):
    """A sphere at p0 (radius width) or a connector from p0 to p1 (radius width) appended to the scene."""
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    if p1 is None:
        mujoco.mjv_initGeom(g, kind, np.array([width, width, width]), np.asarray(p0, float), np.eye(3).ravel(), rgba)
    else:
        mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
        mujoco.mjv_connector(g, kind, width, np.asarray(p0, float), np.asarray(p1, float))
    scn.ngeom += 1


def palm(a):
    """The press replayed in the sphere-pad scene from a camera in the palm: the measured joints, the tracked cylinder
    (translucent), each tip's pads coloured by its board's estimated pressure and the estimated force as an arrow; beside
    the camera's crop, with the force traces."""
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial import cKDTree
    from sr2hand.control import hom_server as HS
    HS.configure("measured", "cad_v2")
    from sr2hand.control import hom_bench as HB
    from sr2hand.control.link import StateMsg
    plan, rec_stem, d_mm = PLAN[a.trial]
    b = HB.HomBench.from_plan(str(ROOT / plan), a.trial, d_mm)
    m = mujoco.MjModel.from_xml_path(str(b.p_pads))
    mp = b.mirror.m                                    # HomBench's own model: the palm and the mounts as the bench set them
    for name in ["palm_pose"] + [f"{f}_mount" for f in SHOW]:
        m.body_pos[m.body(name).id] = mp.body_pos[mp.body(name).id]
        m.body_quat[m.body(name).id] = mp.body_quat[mp.body(name).id]
    dd = mujoco.MjData(m)
    T = trial(a.trial)
    D = json.load(open(WORK / f"{a.trial}_frames.json"))
    rows = D["rows"]
    rec = [json.loads(x) for x in (BENCH / "runs_corvus" / f"{rec_stem}.jsonl").read_text().splitlines() if x.strip()]
    rec = [r for r in rec if "t_ms" in r and r.get("joints") is not None]
    tj = np.array([r.get("t_joints_ms", r["t_ms"]) for r in rec]) / 1e3
    to = np.array([(r.get("obj") or {}).get("t_capture_ms", r["t_ms"]) for r in rec]) / 1e3
    # the scene: pads only on the tips (no tip meshes), the cylinder translucent, the post hidden
    m.vis.global_.offwidth, m.vis.global_.offheight = 1400, 1200
    obj = m.body("screwdriver_medium").id
    pads, tree = {}, {}
    for f in SHOW:
        tb = m.body(f"{f}_tip").id
        g = np.array([i for i in range(m.ngeom) if m.geom_bodyid[i] == tb and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE])
        pads[f] = g
        V = np.array(D["scenes"][f]["verts"])
        Rf, tf = tip_map(f)
        Q = (Rf.T @ (1e3 * m.geom_pos[g] - tf).T).T     # pad centres in the pad frame
        tree[f] = cKDTree(V).query(Q)[1]
        m.geom_rgba[g] = PAD_GREY
        m.geom_size[g] *= 1.25                         # drawn a little larger, so that the pads read as a surface
    size0 = {f: m.geom_size[pads[f]].copy() for f in SHOW}
    padset = set(np.concatenate(list(pads.values())).tolist())
    for i in range(m.ngeom):
        if m.geom_bodyid[i] == obj:
            m.geom_rgba[i] = (0.35, 0.60, 0.95, 0.13)
        elif m.geom_bodyid[i] == m.body("post").id or m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            m.geom_rgba[i] = (0, 0, 0, 0)
        elif i not in padset:                          # the links are drawn as the abstract films' (owner: "the dip
            m.geom_rgba[i] = (0, 0, 0, 0)              # link clutters everything")
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    opt.geomgroup[2] = 1
    for k in range(len(opt.sitegroup)):
        opt.sitegroup[k] = 0
    for t in range(m.ntex):                            # a white sky, as the abstract films
        if m.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            n = int(m.tex_width[t] * m.tex_height[t] * m.tex_nchannel[t])
            m.tex_data[m.tex_adr[t]:m.tex_adr[t] + n] = 255
    W_, H_ = 1020, 680
    ren = mujoco.Renderer(m, H_, W_)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    m.vis.global_.fovy = a.fovy

    def pose(tt):
        k = int(np.argmin(np.abs(tj - tt)))
        q = HB.sim_from_servo(StateMsg(seq=0, t_ms=0.0, joints=np.array(rec[k]["joints"], float)).joints_dict())
        for f in SHOW:
            for i, j in enumerate(("yaw", "mcp", "pip")):
                dd.qpos[m.jnt_qposadr[m.joint(f"{f}_{j}").id]] = q[f][i]
        o = rec[int(np.argmin(np.abs(to - tt)))].get("obj") or {}
        if o.get("p_PO") is not None:
            dd.qpos[0:3] = b.palm + 1e-3 * np.asarray(o["p_PO"], float)
            dd.qpos[3:7] = np.asarray(o["q_PO"], float)
        mujoco.mj_kinematics(m, dd)

    # the camera: in the palm, 8 mm out of it, looking at the three tips' centre at the first frame with a contact
    i0 = next((i for i, rw in enumerate(rows) if any(rw[f]["contact"] for f in SHOW)), 0)
    pose(rows[i0]["t"])
    tips = np.mean([dd.xpos[m.body(f"{f}_tip").id] for f in SHOW], 0)
    fwd = tips - b.palm
    fwd /= np.linalg.norm(fwd)
    eye = b.palm + a.eye_mm * 1e-3 * fwd
    cam.lookat[:] = tips
    cam.distance = float(np.linalg.norm(tips - eye))
    cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    cam.elevation = float(np.degrees(np.arcsin(fwd[2])))
    # the y sign of the map, checked: the mapped contacts should lie on the cylinder's surface
    chk = {f: [] for f in SHOW}
    for i, rw in enumerate(rows[::5]):
        pose(rw["t"])
        R, c = dd.xmat[obj].reshape(3, 3), dd.xpos[obj]
        for f in SHOW:
            mm = D["msgs"][f][5 * i]
            if mm.get("contact"):
                tb = m.body(f"{f}_tip").id
                x = dd.xpos[tb] + dd.xmat[tb].reshape(3, 3) @ viewer_to_body(f, mm["p"])
                u = R.T @ (x - c)
                chk[f].append(abs(np.hypot(u[0], u[1]) - 1e-3 * d_mm / 2) * 1e3)
    print("mapped contact to the cylinder's surface, median mm: " + ", ".join(
        f"{f} {np.median(v):.1f}" for f, v in chk.items() if v), flush=True)
    fy = 0.5 * H_ / np.tan(np.radians(a.fovy) / 2)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    def project(x):
        """A world point's pixel in the palm camera's image."""
        dv = np.asarray(x) - eye
        zc = dv @ fwd
        return (float(W_ / 2 + fy * (dv @ right) / zc), float(H_ / 2 - fy * (dv @ up) / zc))

    labels = []
    out = WORK / f"{a.trial}_palm"
    out.mkdir(exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    for i, rw in enumerate(rows):
        pose(rw["t"])
        arrows = []
        for f in SHOW:                                 # the pads' colours first: update_scene copies them
            mm = D["msgs"][f][i]
            g = pads[f]
            if mm.get("contact") and mm.get("pr") is not None:
                kp = np.asarray(mm["pr"], float)[tree[f]]
                m.geom_rgba[g] = pressure_rgba(kp)
                m.geom_size[g] = pad_sizes(size0[f], kp)
                tb = m.body(f"{f}_tip").id
                Rb = dd.xmat[tb].reshape(3, 3)
                x = dd.xpos[tb] + Rb @ viewer_to_body(f, mm["p"])
                F = Rb @ (tip_map(f)[0] @ np.asarray(mm["f"], float))
                if np.linalg.norm(F) > 1e-3:
                    arrows.append((x, F))
            else:
                m.geom_rgba[g] = PAD_GREY
                m.geom_size[g] = size0[f]
        ren.update_scene(dd, cam, opt)
        for f in SHOW:                                 # the abstract hand: grey links, joints at yaw, MCP and PIP
            ch = [dd.xpos[m.body(f"{f}_{k}").id].copy() for k in CHAIN]
            pc = dd.geom_xpos[pads[f]].mean(axis=0)
            d_ = pc - ch[-1]
            ch.append(ch[-1] + d_ * max(0.0, 1.0 - 0.012 / np.linalg.norm(d_)))   # stops 12 mm short of the pads' centre
            for p0, p1 in zip(ch[:-1], ch[1:]):
                if np.linalg.norm(p1 - p0) > 1e-4:
                    add_geom(ren.scene, mujoco.mjtGeom.mjGEOM_CAPSULE, LINK_RGBA, 2.2e-3, p0, p1)
            for j in (1, 2, 4):
                add_geom(ren.scene, mujoco.mjtGeom.mjGEOM_SPHERE, JOINT_RGBA, 3.4e-3, ch[j], None)
        for x, F in arrows:                            # the estimated force on the pad, ending at the contact
            L = 1e-3 * a.arrow * np.linalg.norm(F)
            if ren.scene.ngeom < ren.scene.maxgeom:
                gg = ren.scene.geoms[ren.scene.ngeom]
                mujoco.mjv_initGeom(gg, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                                    np.array([0.05, 0.30, 1.0, 1.0], np.float32))
                mujoco.mjv_connector(gg, mujoco.mjtGeom.mjGEOM_ARROW, 0.8e-3, x - L * F / np.linalg.norm(F), x)
                ren.scene.ngeom += 1
        Image.fromarray(ren.render()).save(out / f"r{i:05d}.png")
        labels.append({f: project(dd.xpos[m.body(f"{f}_tip").id]) for f in SHOW})
    ren.close()
    print(f"{len(rows)} renders in {out.relative_to(ROOT)}", flush=True)
    palm_film(a, T, D, out, labels)


def palm_film(a, T, D, rdir, labels):
    """The palm film: the camera's crop on the left; on the right the palm view, each tip labelled with its estimated
    normal force, and the force traces under it."""
    from PIL import Image, ImageDraw, ImageFont
    rows = D["rows"]
    vids = [video_frames(stem, T["cam"]) for stem, *_ in T["segs"]]
    F = {s_: ImageFont.truetype(FONT, s_) for s_ in (17, 24)}
    FB = {s_: ImageFont.truetype(FONT_B, s_) for s_ in (22, 30)}
    trace, (x0, x1), _ax = trace_image(D, T, (1020, 240), ylabel="force (N)")
    head = wrap(T["title"].split(", ", 1)[1] + ". Replayed in the simulator from the trial's record: the measured joints "
                "and the tracked cylinder (blue), "
                "seen from a camera in the palm. Each tip's pads are coloured by its magnet board's estimated pressure "
                "(the trials' estimator: the tip's UR5 calibration and board GP, its own finger's crosstalk taken off; "
                "darker: more pressure), and the estimated force is the blue arrow.", F[17], 975)
    out = WORK / f"{a.trial}_palm_film"
    out.mkdir(exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    x, y, w, h = T["crop"]
    for i, rw in enumerate(rows):
        im = Image.new("RGB", (1920, 1080), "white")
        files, tv = vids[rw["seg"]]
        j = int(np.argmin(np.abs(tv - rw["t"])))
        im.paste(Image.open(files[j]).convert("RGB").crop((x, y, x + w, y + h)).resize((900, 1080), Image.LANCZOS), (0, 0))
        dr = ImageDraw.Draw(im)
        dr.rectangle((0, 0, 900, 40), fill="white")
        dr.text((12, 7), "the hand, real time", font=F[24], fill="#141414")
        dr.text((930, 12), T["title"].split(",")[0] + ", seen from the palm", font=FB[30], fill="#0f766e")
        for k, line in enumerate(head):
            dr.text((930, 54 + 21 * k), line, font=F[17], fill="#374151")
        y0 = 60 + 21 * len(head)
        im.paste(Image.open(rdir / f"r{i:05d}.png").convert("RGB"), (900, y0))
        for k, f in enumerate(SHOW):
            u, v = labels[i][f]
            m_ = rw[f]
            txt = f"{f} {m_['fn']:.1f} N" if m_["contact"] and m_["fn"] is not None else f"{f}"
            tx, ty = 900 + u + (18 if u < 510 else -18 - FB[22].getlength(txt)), y0 + v - 46
            tx, ty = min(max(tx, 905), 1910 - FB[22].getlength(txt)), min(max(ty, y0 + 4), y0 + 646)
            dr.rectangle((tx - 4, ty - 2, tx + FB[22].getlength(txt) + 4, ty + 26), fill="white")
            dr.text((tx, ty), txt, font=FB[22], fill=COLS[k])
        ty0 = 1080 - 240
        im.paste(trace, (900, ty0))
        px = 900 + x0 + (x1 - x0) * i / max(1, len(rows) - 1)
        dr.line((px, ty0 + 8, px, ty0 + 240 - 30), fill="#111827", width=2)
        im.save(out / f"{i:05d}.png")
    film = MEDIA / f"20261116-fingertips_{a.trial}_palm.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(out / "%05d.png"), "-c:v", "libx264",
                    "-crf", "20", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(film)],
                   check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{len(rows) / 2 / FPS:.2f}", "-i", str(film), "-frames:v", "1",
                    "-q:v", "3", str(film.with_suffix(".jpg"))], check=True)
    print(f"wrote {film.relative_to(ROOT)} ({len(rows) / FPS:.1f} s)", flush=True)


def VIEW_BOX(k):
    """The k-th board's 3D view in the viewer's 1200 x 1800 page, its middle 590 px, below its name line and above its
    pressure legend (the boards' panels start every 582 px below the page's controls)."""
    return (110, 582 * k + 84, 700, 582 * k + 588)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("frames", "capture", "compose", "palm"):
        p = sub.add_parser(c)
        p.add_argument("--trial", required=True, choices=("press", "twist", "pickup"))
    cp = sub.choices["capture"]
    cp.add_argument("--width", type=int, default=1200)
    cp.add_argument("--height", type=int, default=1800)
    cp.add_argument("--dist", type=float, default=30.0, help="the cameras' distance from the contacts, mm")
    cp.add_argument("--arrow", type=float, default=2.0)
    cp.add_argument("--n", type=int, default=None)
    cp.add_argument("--every", type=int, default=1)
    pp = sub.choices["palm"]
    pp.add_argument("--fovy", type=float, default=62.0)
    pp.add_argument("--eye-mm", type=float, default=8.0, help="the camera's distance out of the palm toward the tips")
    pp.add_argument("--arrow", type=float, default=4.0, help="mm per N")
    a = ap.parse_args()
    return {"frames": frames, "capture": capture, "compose": compose, "palm": palm}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
