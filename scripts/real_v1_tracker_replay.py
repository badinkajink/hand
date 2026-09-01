#!/usr/bin/env python3
"""Watch a bench run back: the taped IR frames, captioned with the run's own trace.

    # every taped run that has not been rendered yet
    scripts/real_v1_tracker_replay.py --all

    # one run, by id, both outputs
    scripts/real_v1_tracker_replay.py 20260831-092657 --mp4 --filmstrip 8

A run only has pixels if it was recorded with `--video-hz` (see `real_v1_tag_tracker.py`);
without it there is a CSV and nothing to replay. The tape is deliberately dumb -- raw IR
frames named with their time -- and every judgement about what to draw lives here, offline,
so a bad caption costs a re-render and never a run.

TWO OUTPUTS, FOR TWO DIFFERENT QUESTIONS.
  --mp4         the turn at watchable speed, for "what actually happened". Contact loss,
                the pivot moving, a finger walking off the pad -- things that are motion.
  --filmstrip   evenly spaced tiles in one PNG, for "what did these two hands do
                differently". A still comparison is what goes in a paper, and phase-aligned
                tiles are how two runs of different length get compared at all.

INTERPRETER. Needs OpenCV, which on this workstation is in ~/miniconda3, not the uv env.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from morphohand.bench import replay as RP  # noqa: E402

try:
    import cv2
except ImportError:                                 # pragma: no cover - environment, not logic
    raise SystemExit("this needs OpenCV; try  /home/humanoid/miniconda3/bin/python "
                     + " ".join(sys.argv)) from None

TRACKER_LOGS = ROOT / "logs/tracker"

INK = (236, 236, 236)
DIM = (150, 150, 150)
GOOD = (120, 220, 130)
WARN = (90, 170, 250)
LOST = (90, 90, 240)
PEAK = (250, 210, 120)


def resolve(token: str) -> tuple[Path, Path]:
    """(trace csv, frames dir) from a csv path, a frames dir, or any part of a run id."""
    p = Path(token)
    if p.is_dir() and p.name.endswith("_frames"):
        return p.with_name(p.name[: -len("_frames")] + ".csv"), p
    if p.suffix == ".csv" and p.exists():
        return p, p.with_name(p.stem + "_frames")
    hits = sorted(TRACKER_LOGS.glob(f"*{token}*_track.csv")) if TRACKER_LOGS.is_dir() else []
    if len(hits) == 1:
        return hits[0], hits[0].with_name(hits[0].stem + "_frames")
    if not hits:
        raise SystemExit(f"no trace matches {token!r} under {TRACKER_LOGS}")
    raise SystemExit(f"{token!r} matches {len(hits)} traces: "
                     + ", ".join(h.stem for h in hits))


def summary_of(trace: Path) -> dict:
    side = trace.with_name(trace.stem + "_SUMMARY.json")
    return json.loads(side.read_text()) if side.exists() else {}


def _scaled(w: int):
    """Type sizes that hold at both 640x360 and full resolution."""
    s = max(0.38, w / 1280.0 * 0.62)
    return s, max(1, int(round(w / 640.0)))


def annotate(img, cap: RP.Caption, *, title: str, duration_s: float, gaps=()):
    """Burn one frame's own numbers onto it. Returns a new BGR image."""
    out = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = out.copy()
    h, w = out.shape[:2]
    fs, th = _scaled(w)
    pad = int(round(w * 0.015))

    top = 0
    if title:
        band = out[: int(h * 0.055) + 2 * pad].copy()
        cv2.rectangle(band, (0, 0), (w, band.shape[0]), (0, 0, 0), -1)
        out[: band.shape[0]] = cv2.addWeighted(band, 0.55, out[: band.shape[0]], 0.45, 0)
        cv2.putText(out, title, (pad, int(band.shape[0] * 0.68)), cv2.FONT_HERSHEY_DUPLEX,
                    fs * 0.85, INK, th, cv2.LINE_AA)
        top = band.shape[0]

    # A scrim under the text. IR frames of a lit bench are mostly WHITE, and light grey type
    # on a white tabletop is type nobody can read -- which was the state of the first live
    # tape. The panel is sized to the block it covers so it never grows into the picture.
    lines = cap.lines()
    y = top + int(pad * 2.2) + int(fs * 26)
    block_h = int(fs * 34) * len(lines) + (int(w * 0.30 * 0.055) + int(fs * 32)
                                           if cap.seen else 0)
    x1, y1 = pad // 2 + int(w * 0.34), y - int(fs * 30) + block_h + pad
    panel = out[max(top, y - int(fs * 30) - pad // 2):y1, pad // 2:x1]
    if panel.size:
        panel[:] = cv2.addWeighted(np.zeros_like(panel), 0.62, panel, 0.38, 0)

    colour = LOST if not cap.seen else (PEAK if cap.peak else INK)
    for line in lines:
        cv2.putText(out, line, (pad, y), cv2.FONT_HERSHEY_DUPLEX, fs, colour, th, cv2.LINE_AA)
        y += int(fs * 34)

    if cap.seen:
        _meter(out, cap.cos, x0=pad, y0=y, width=int(w * 0.30), fs=fs, th=th)
    _timeline(out, cap.t_s, duration_s, gaps, pad=pad, fs=fs, th=th)
    return out


def _meter(img, cos, *, x0, y0, width, fs, th):
    """cos on [-1, +1] as a bar. The centre tick is horizontal; the right end is straight up."""
    h = max(4, int(width * 0.055))
    cv2.rectangle(img, (x0, y0), (x0 + width, y0 + h), DIM, 1, cv2.LINE_AA)
    mid = x0 + width // 2
    cv2.line(img, (mid, y0 - 2), (mid, y0 + h + 2), DIM, 1, cv2.LINE_AA)
    fill = int(round((float(np.clip(cos, -1, 1)) + 1.0) / 2.0 * width))
    lo, hi = (mid, x0 + fill) if fill >= width // 2 else (x0 + fill, mid)
    cv2.rectangle(img, (lo, y0 + 1), (hi, y0 + h - 1), GOOD if cos > 0 else WARN, -1)
    cv2.putText(img, "flat", (mid - int(fs * 26), y0 + h + int(fs * 26)),
                cv2.FONT_HERSHEY_DUPLEX, fs * 0.6, DIM, th, cv2.LINE_AA)
    cv2.putText(img, "up", (x0 + width - int(fs * 22), y0 + h + int(fs * 26)),
                cv2.FONT_HERSHEY_DUPLEX, fs * 0.6, DIM, th, cv2.LINE_AA)


def _timeline(img, t, duration_s, gaps, *, pad, fs, th):
    """Where this frame sits in the run, with the stretches where the tag was lost in red."""
    h, w = img.shape[:2]
    y = h - pad - max(3, int(w * 0.006))
    x0, x1 = pad, w - pad
    span = max(duration_s, 1e-6)
    cv2.line(img, (x0, y), (x1, y), DIM, max(1, th - 1), cv2.LINE_AA)
    for g0, g1 in gaps:
        a = x0 + int((x1 - x0) * float(g0) / span)
        b = x0 + int((x1 - x0) * float(g1) / span)
        cv2.line(img, (a, y), (max(b, a + 1), y), LOST, max(2, th + 1), cv2.LINE_AA)
    cv2.circle(img, (x0 + int((x1 - x0) * min(t / span, 1.0)), y), max(3, int(w * 0.007)),
               INK, -1, cv2.LINE_AA)


def crop_box(shape, spec: str | None):
    """`x0,y0,x1,y1` as FRACTIONS of the frame -> pixel slice. Fractions rather than pixels so
    one crop survives a change of `--video-scale`."""
    h, w = shape[:2]
    if not spec:
        return 0, 0, w, h
    f = [float(v) for v in spec.split(",")]
    if len(f) != 4 or not all(0.0 <= v <= 1.0 for v in f) or f[0] >= f[2] or f[1] >= f[3]:
        raise SystemExit("--crop takes x0,y0,x1,y1 as fractions in [0,1] with x0<x1, y0<y1")
    return int(f[0] * w), int(f[1] * h), int(f[2] * w), int(f[3] * h)


def render(trace: Path, frames_dir: Path, *, want_mp4: bool, tiles: int, fps: float,
           out_dir: Path | None, force: bool, crop: str | None = None) -> int:
    paths = RP.tape_frames(frames_dir)
    if not paths:
        print(f"  {trace.stem}: no taped frames in {frames_dir.name}/ -- recorded without "
              "--video-hz")
        return 1
    rows = RP.read_trace(trace)
    times = [RP.frame_time_s(p) for p in paths]
    caps = RP.captions(times, rows)
    doc = summary_of(trace)
    summ = doc.get("summary", {})
    run = doc.get("run_id") or trace.stem
    duration = float(summ.get("duration_s") or (times[-1] if times else 1.0))
    gaps = [tuple(g) for g in summ.get("gaps", [])]
    c0, c1 = summ.get("cos_start"), summ.get("cos_hold")
    head = (f"{run}   cos {c0:+.2f} -> {c1:+.2f}" if c0 is not None and c1 is not None
            else run)
    out_dir = out_dir or trace.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    if want_mp4:
        mp4 = out_dir / f"{trace.stem}_replay.mp4"
        if force or not mp4.exists():
            first = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
            cx0, cy0, cx1, cy1 = crop_box(first.shape, crop)
            h, w = cy1 - cy0, cx1 - cx0
            vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not vw.isOpened():
                print(f"  {run}: OpenCV would not open an mp4 writer; skipping the video")
            else:
                for p, cap in zip(paths, caps):
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    vw.write(annotate(img[cy0:cy1, cx0:cx1], cap, title=head,
                                      duration_s=duration, gaps=gaps))
                vw.release()
                made.append(mp4)

    if tiles > 0:
        strip = out_dir / f"{trace.stem}_filmstrip.png"
        if force or not strip.exists():
            picks = RP.evenly_spaced(times, tiles)
            imgs = []
            for i in picks:
                img = cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    x0, y0, x1, y1 = crop_box(img.shape, crop)
                    imgs.append(annotate(img[y0:y1, x0:x1], caps[i], title="",
                                         duration_s=duration, gaps=gaps))
            if imgs:
                cv2.imwrite(str(strip), _tile(imgs, header=head))
                made.append(strip)

    for m in made:
        print(f"  {m}  ({len(paths)} taped frames, {sum(c.seen for c in caps)} with the tag)")
    return 0 if made else 1


def _tile(imgs, header: str = "", cols: int | None = None):
    """A contact sheet, titled once. Columns default near-square, never taller than wide.

    The run's identity belongs to the sheet, not to each tile: repeated across eight tiles it
    is eight copies of the one thing that does not vary, taking the space the tile needs for
    the numbers that do.
    """
    n = len(imgs)
    cols = cols or min(n, max(2, int(round(np.sqrt(n * 1.6)))))
    rows = int(np.ceil(n / cols))
    h, w = imgs[0].shape[:2]
    fs, th = _scaled(w)
    bar = int(h * 0.10) if header else 0
    sheet = np.zeros((bar + rows * h, cols * w, 3), np.uint8)
    if header:
        cv2.putText(sheet, header, (int(w * 0.015), int(bar * 0.68)),
                    cv2.FONT_HERSHEY_DUPLEX, fs, INK, th, cv2.LINE_AA)
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        sheet[bar + r * h:bar + (r + 1) * h, c * w:(c + 1) * w] = im
    for r in range(1, rows):
        sheet[bar + r * h - 1:bar + r * h + 1, :] = 40
    for c in range(1, cols):
        sheet[bar:, c * w - 1:c * w + 1] = 40
    return sheet


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="*", help="a trace CSV, a _frames dir, or part of a run id")
    p.add_argument("--all", action="store_true",
                   help=f"every taped run under {TRACKER_LOGS.relative_to(ROOT)}")
    p.add_argument("--mp4", action="store_true", help="write the annotated video")
    p.add_argument("--filmstrip", type=int, nargs="?", const=8, default=0, metavar="TILES",
                   help="write a contact sheet of TILES evenly spaced frames (default 8)")
    p.add_argument("--fps", type=float, default=8.0,
                   help="playback rate of the mp4; the tape's own rate plays back in realtime")
    p.add_argument("--crop", default=None, metavar="X0,Y0,X1,Y1",
                   help="keep only this box, as fractions of the frame -- the bench fills a "
                        "fraction of the camera's view and a paper figure wants the hand")
    p.add_argument("--out-dir", type=Path, default=None, help="default: beside the trace")
    p.add_argument("--force", action="store_true", help="re-render outputs that already exist")
    a = p.parse_args()

    if not a.mp4 and not a.filmstrip:
        a.mp4 = True
        a.filmstrip = 8

    jobs = [resolve(t) for t in a.run]
    if a.all:
        for d in sorted(TRACKER_LOGS.glob("*_frames")):
            job = (d.with_name(d.name[: -len("_frames")] + ".csv"), d)
            if job[0].exists() and job not in jobs:
                jobs.append(job)
    if not jobs:
        raise SystemExit("nothing to replay: name a run, or --all once some runs are taped")

    bad = 0
    for trace, frames in jobs:
        print(trace.stem)
        bad += render(trace, frames, want_mp4=a.mp4, tiles=a.filmstrip, fps=a.fps,
                      out_dir=a.out_dir, force=a.force, crop=a.crop)
    return 1 if bad == len(jobs) else 0


if __name__ == "__main__":
    sys.exit(main())
