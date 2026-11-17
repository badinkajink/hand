#!/usr/bin/env python3
"""The magnetic fingertips' contact estimates beside the camera on the hand's trials of 2026-11-16 (owner, 2026-11-16
23:5x: "can we make a video that synchronizes the magnetic fingertip visualization with a real trial from today? it can be
the twist ..., the press, the pick up from dropped, (maybe all of them) just so i can pick and choose"; "please do a
center crop on the core manipulation ... and then show just the critical visualization of the contact sensors").

Three boards sit under the SR2 hand's v3 tips (mux channels 0 middle, 1 index, 3 thumb; logs/20261116-three_finger_mux/
*-mux.bin, copied from corvus). Each board runs through the fingertip viewer's own App (src/sr2hand/driver/reskin5x/host/
wrench_viewer.py, case four_05R4) with the physical model fitted to the rest just before the fingers close, as the viewer
does live, and no re-zero during the trial. Nothing on the hand calibrates these tips: the estimates are the viewer's
uncalibrated ones, and a moving finger's magnets reach the other boards (the SR2 crosstalk page). The viewer's page draws
each tip; this script crops its 3D views and sets them beside a crop of the trial's camera, frame by frame on corvus's
clock, which the mux records, the trial records and the cameras' frame times all carry.

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
def frames(a):
    """The viewer's messages for every film frame, one App per board on the trial's raw records."""
    from viewer_roll_film import Clock, viewer_args
    sys.path.insert(0, str(VIEWER))
    import wrench_viewer as WV
    T = trial(a.trial)
    ft = film_times(T)
    lo = min(s[1] for s in T["segs"]) - 6.0
    hi = max(s[2] for s in T["segs"]) + 1.0
    t, r = mux_window(T, lo - (60.0 if T.get("rest_after") else 0.0), hi)
    rest = T["rest"]
    if rest is None:                                   # the pick-up: the rest just before the close
        rest = first_change(t, r, T["rest_after"]) - 0.3
    print(f"{a.trial}: {len(ft)} frames, {len(t)} records, rest at {time.strftime('%H:%M:%S', time.localtime(rest))}",
          flush=True)
    apps = {}
    for f in SHOW:
        src = Clock(t, r[:, BOARD[f]], keep=3.0)
        app = WV.App(viewer_args(auto_zero=0.0), src=src, name=f, board_gp=None)
        src.now = rest
        app.calibrate()
        print(f, app.status, flush=True)
        apps[f] = app
    msgs = {f: [] for f in SHOW}
    rows = []
    for k, tt in ft:
        row = dict(seg=k, t=round(tt, 3))
        for f, app in apps.items():
            app.src.now = float(tt)
            m = app.frame()
            msgs[f].append(m)
            row[f] = dict(contact=bool(m.get("contact")), fn=m.get("fn"), p=m.get("p"), db=float(np.linalg.norm(
                m.get("db", [0.0]))) if m.get("db") is not None else None)
        rows.append(row)
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / f"{a.trial}_frames.json"
    json.dump(dict(scenes={f: app.scene() for f, app in apps.items()}, msgs=msgs, rows=rows, rest=rest,
                   status={f: app.status for f, app in apps.items()}), open(out, "w"))
    for f in SHOW:
        n = sum(rw[f]["contact"] for rw in rows)
        fn = [rw[f]["fn"] for rw in rows if rw[f]["contact"]]
        print(f"{f}: contact in {n} of {len(rows)} frames" + (f", normal force median {np.median(fn):.2f} N, max "
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


def trace_image(D, T, size):
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
    ax.set_ylabel("estimated normal force (N)")
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
    head = wrap(T["sub"] + " Each tip: the fingertip viewer's estimate from its board, the physical model fitted at rest "
                "before the close. Colour: pad pressure to 300 kPa; blue arrow: the estimated force. Not calibrated on "
                "the hand.", F[17], 975)
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


def VIEW_BOX(k):
    """The k-th board's 3D view in the viewer's 1200 x 1800 page, its middle 590 px, below its name line and above its
    pressure legend (the boards' panels start every 582 px below the page's controls)."""
    return (110, 582 * k + 84, 700, 582 * k + 588)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("frames", "capture", "compose"):
        p = sub.add_parser(c)
        p.add_argument("--trial", required=True, choices=("press", "twist", "pickup"))
    cp = sub.choices["capture"]
    cp.add_argument("--width", type=int, default=1200)
    cp.add_argument("--height", type=int, default=1800)
    cp.add_argument("--dist", type=float, default=30.0, help="the cameras' distance from the contacts, mm")
    cp.add_argument("--arrow", type=float, default=2.0)
    cp.add_argument("--n", type=int, default=None)
    cp.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    return {"frames": frames, "capture": capture, "compose": compose}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
