#!/usr/bin/env python3
"""A 2x4 video array of the eight transfer-study hands holding a reorientation on the bench.

One panel per hand, D1..D8 in reading order, from the tracker's own IR tape (640x360 at
~3.8 fps, `logs/tracker/*_track_frames`) with the measured net turn overlaid bottom-right at
the tracker's rate (~27 Hz, `*_track.csv`). Crop, tone mapping and the angle convention are
the filmstrip's (`real_v1_filmstrip.py`), so the video and `fig_filmstrip_designs` read the
same way: the number is the shaft's turn from its reading at the first frame, and it says
"tag lost" whenever no detection lies within TOL_S of the instant.

The trial shown for each hand is its best-aligned HELD trial: the operator's verdict is
HELD and, among those with a complete trace, cos_hold is the highest. g12 (D4) has no
complete trace -- its tag leaves view optically at 1.4-2.3 s on every trial -- so it shows
the HELD trial whose tag stayed in view longest. Trials admitted to the alignment pool from
another session (`extra`) have no tape and are never candidates.

The tape is half-scale, so the panels are an upscale: --scale 3 gives 990x900 panels and a
4020x1836 array. A 1920-wide copy is written next to it for slide software that refuses 4K.
"""
import argparse, csv, glob, json, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_v1_transfer_figures import bench                              # noqa: E402
from real_v1_transfer_ranking import DESIGN_ID                          # noqa: E402
from real_v1_filmstrip import CROP, LO_PCT, HI_PCT, _tone, tape         # noqa: E402

OUT = "docs/experiments/20260921-bench_video_array"
TOL_S = 0.2                     # a detection this close to the instant is "the reading"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
#: overlay type sizes in tape pixels (x scale): the hand label is a tenth of the panel
#: height and the angle a sixth, so both read from the back of a room.
LABEL_PT, DEG_PT, CLOCK_PT = 30, 52, 20
#: chip colour. Black boxes vanish into a greyscale IR frame; the ranking figures'
#: bench blue (#1f4e79) lifted a step for the projector. White on it is 7.2:1.
BOX = (36, 86, 160, 255)


def readings(run_tag):
    """Every tracker row as (t, deg or None); rows without a detection carry None."""
    p = glob.glob(f"logs/tracker/*-{run_tag}_track.csv")
    if not p:
        return []
    return [(float(r["t"]), float(r["deg"]) if r.get("cos") else None)
            for r in csv.DictReader(open(p[0]))]


def deg_at(rows, t, tol=TOL_S):
    """The detection nearest t within tol, else None. A single missed detection between
    two good ones does not flash "tag lost" at 27 Hz; a real loss does."""
    near = [(abs(x - t), d) for x, d in rows if d is not None and abs(x - t) <= tol]
    return min(near)[1] if near else None


def pick(B):
    """design -> (trial, n_held, n_trials), best-aligned HELD trial with tape."""
    out = {}
    for dsg, g in B.items():
        own = [t for t in g if not t["extra"]]
        nh = sum(t["outcome"] == "HELD" for t in own)
        held = [t for t in own if t["outcome"] == "HELD" and tape(t["tag"])[0]]
        complete = [t for t in held if t["cos_hold"] is not None]
        if complete:
            run = max(complete, key=lambda t: t["cos_hold"])
        else:
            def seen_until(t):
                r = [x for x, d in readings(t["tag"]) if d is not None]
                return r[-1] if r else -1.0
            run = max(held, key=seen_until)
        out[dsg] = (run, nh, len(own))
    return out


class Panel:
    def __init__(self, dsg, run, nh, n, scale):
        self.label = f"{DESIGN_ID[dsg]}   {nh}/{n} held"
        self.run = run
        fs, self.ts = tape(run["tag"])
        raw = [np.asarray(Image.open(f).convert("L").crop(CROP), dtype=float) for f in fs]
        lo, hi = np.percentile(np.concatenate([a.ravel() for a in raw]), [LO_PCT, HI_PCT])
        W, H = (CROP[2] - CROP[0]) * scale, (CROP[3] - CROP[1]) * scale
        self.size = (W, H)
        self.frames = [Image.fromarray(_tone(a, lo, hi).astype(np.uint8)).convert("RGB")
                       .resize((W, H), Image.LANCZOS) for a in raw]
        self.rows = readings(run["tag"])
        self.base = deg_at(self.rows, self.ts[0], tol=0.35)
        self.t_end = self.ts[-1]

    def frame_at(self, t):
        k = max([i for i, x in enumerate(self.ts) if x <= t] or [0])
        return self.frames[k]

    def turn_at(self, t):
        d = deg_at(self.rows, t)
        if d is None or self.base is None:
            return None
        return self.base - d


def compose(panels, t, scale, fonts, clock, gutter):
    W, H = panels[0].size
    ncol = 4
    sheet = Image.new("RGB", (ncol * W + (ncol + 1) * gutter, 2 * H + 3 * gutter), "white")
    f_lab, f_deg, f_clk = fonts
    for i, P in enumerate(panels):
        im = P.frame_at(t).copy()
        dr = ImageDraw.Draw(im, "RGBA")
        pad = 10 * scale
        # hand label, top-left
        _chip(dr, (pad, pad), P.label, f_lab, anchor="la", fill=(255, 255, 255, 255))
        # net turn, bottom-right
        turn = P.turn_at(t)
        if turn is None:        # at the angle's size "tag lost" would span the panel
            _chip(dr, (W - pad, H - pad), "tag lost", f_lab, anchor="rd",
                  fill=(221, 221, 221, 255))
        else:
            _chip(dr, (W - pad, H - pad), f"{turn:+.0f}°", f_deg, anchor="rd",
                  fill=(255, 255, 255, 255))
        if clock:
            _chip(dr, (pad, H - pad), f"{t:.1f} s", f_clk, anchor="ld",
                  fill=(221, 221, 221, 255))
        x = gutter + (i % ncol) * (W + gutter)
        y = gutter + (i // ncol) * (H + gutter)
        sheet.paste(im, (x, y))
    return sheet


def _chip(dr, xy, text, font, anchor, fill):
    """White text on an opaque blue box, so the annotation reads against a grey frame."""
    x, y = xy
    l, t, r, b = dr.textbbox((x, y), text, font=font, anchor=anchor)
    m = font.size * 0.22
    dr.rectangle((l - m, t - m, r + m, b + m), fill=BOX)
    dr.text((x, y), text, font=font, fill=fill, anchor=anchor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=3, help="upscale of the 330x300 crop")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback rate; 0.5 plays the 5.8 s run over 11.6 s")
    ap.add_argument("--clock", action="store_true", help="elapsed time, bottom-left")
    ap.add_argument("--poster-t", type=float, default=1.7,
                    help="instant of the still written beside the video")
    ap.add_argument("--out", default=f"{OUT}/20260921-eight_hands_held_turn_2x4")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    B = bench(10)
    P = pick(B)
    panels = [Panel(dsg, *P[dsg], a.scale) for dsg in sorted(B, key=lambda d: DESIGN_ID[d])]
    fonts = tuple(ImageFont.truetype(FONT, pt * a.scale) for pt in (LABEL_PT, DEG_PT, CLOCK_PT))
    gutter = 4 * a.scale
    t_end = max(p.t_end for p in panels)
    n = int(round(t_end * a.fps / a.speed)) + 1
    W, H = compose(panels, 0.0, a.scale, fonts, a.clock, gutter).size
    W, H = W - W % 2, H - H % 2                      # yuv420p wants even dimensions

    manifest = dict(runs=[dict(design=DESIGN_ID[p.run["design"]], run_id=p.run["run_id"],
                               outcome=p.run["outcome"], cos_hold=p.run["cos_hold"],
                               deg_turned=p.run["deg"], slip_mm=p.run["slip"],
                               base_deg=p.base, label=p.label, n_frames=len(p.frames))
                         for p in panels],
                    crop=CROP, scale=a.scale, fps=a.fps, speed=a.speed, size=[W, H],
                    t_end_s=t_end, tol_s=TOL_S)
    json.dump(manifest, open(a.out + ".json", "w"), indent=1)

    mp4 = a.out + ".mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
    poster = None
    for k in range(n):
        t = k / a.fps * a.speed
        im = compose(panels, t, a.scale, fonts, a.clock, gutter).crop((0, 0, W, H))
        ff.stdin.write(im.tobytes())
        if poster is None and t >= a.poster_t:
            poster = im
    ff.stdin.close(); ff.wait()
    (poster or im).save(a.out + "_poster.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4,
                    "-vf", "scale=1920:-2:flags=lanczos", "-c:v", "libx264", "-preset", "slow",
                    "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    a.out + "_1920.mp4"], check=True)
    for p in panels:
        r = p.run
        print(f"{p.label:18s} {r['run_id']}  cos_hold={r['cos_hold']}  "
              f"deg={r['deg']:.1f}  slip={r['slip']:.1f} mm  base={p.base:.1f}")
    print(f"wrote {mp4} ({W}x{H}, {n} frames at {a.fps} fps), {a.out}_1920.mp4, "
          f"{a.out}_poster.png, {a.out}.json")


if __name__ == "__main__":
    main()
