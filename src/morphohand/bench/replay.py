"""Putting the run back on screen: a tape of IR frames, re-joined to its own trace.

The tracker's CSV says the shaft turned 55.9 degrees. It cannot say whether the middle
finger walked off the pad on the way, whether the tool pivoted about a contact or about
the floor, or which of two hands with the same final cosine got there by carrying versus
by flinging. Those are the questions a *picture* answers, and until this module the run
folder held no pixels at all -- only rows.

WHAT THE TAPE IS. A run optionally writes one JPEG every 1/`video_hz` seconds into
`<trace stem>_frames/`, named `INDEX_MILLISECONDS.jpg` so a frame carries its own position
on the trace's clock. Individual files, not a container: the bench service stops a run with
SIGINT and a video file that was never released is unplayable, whereas every frame written
before the interrupt survives. Assembly is this module's job, offline, where it is free to
fail and be re-run.

WHAT IS PURE AND WHAT IS NOT. Everything here is numpy-and-stdlib: name a frame, read a
trace back, decide which row a frame belongs to, choose the tiles of a filmstrip. The
drawing needs OpenCV, which on this workstation lives only in the conda interpreter, so it
stays in `scripts/real_v1_tracker_replay.py`. The split is so the join -- the part that can
silently mislabel a frame with another moment's numbers -- is unit-testable without a camera.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

#: Frames are named so that sorting them lexically is sorting them in time, and so that a
#: frame separated from its folder still knows when it was taken.
FRAME_RE = re.compile(r"^(\d{5})_(\d{8})\.jpg$")

#: A frame is labelled with a trace row only if one was recorded near enough in time. The
#: tape is sampled at a few Hz off a 30 Hz stream, so a genuine match is within a frame or
#: two; anything further away means the tag was LOST there, and saying so is the point.
MATCH_TOLERANCE_S = 0.25


def frame_name(index: int, t: float) -> str:
    """`INDEX_MILLISECONDS.jpg`, zero-padded so lexical order is temporal order."""
    if t < 0:
        raise ValueError("frame times are seconds since the trace started, never negative")
    return f"{index:05d}_{int(round(t * 1000.0)):08d}.jpg"


def frame_time_s(name: str | Path) -> float:
    """The seconds-since-start a tape frame was taken, recovered from its file name."""
    m = FRAME_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"{Path(name).name!r} is not a tape frame name (INDEX_MILLIS.jpg)")
    return int(m.group(2)) / 1000.0


def tape_frames(directory: str | Path) -> list[Path]:
    """Every tape frame in `directory`, in time order. Foreign files are ignored, not an error
    -- a run folder accumulates renders, and a stray PNG must not break the replay."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if FRAME_RE.match(p.name)), key=lambda p: p.name)


def read_trace(path: str | Path) -> list[dict]:
    """The tracker's CSV as floats, with blanks (a LOST frame, or a withheld x/y) as None.

    Blank is not zero. `x_bench_mm` is empty for every run whose heading was never resolved,
    and a reader that coerced that to 0.0 would put the shaft at the palm centre.
    """
    rows: list[dict] = []
    with Path(path).open(newline="") as f:
        for raw in csv.DictReader(f):
            row: dict = {}
            for k, v in raw.items():
                if k is None:
                    continue
                s = (v or "").strip()
                try:
                    row[k] = float(s) if s else None
                except ValueError:
                    row[k] = s
            if row.get("t") is not None:
                rows.append(row)
    return rows


def seen_rows(rows: list[dict]) -> list[dict]:
    """The rows where the cylinder tag was actually detected."""
    return [r for r in rows if r.get("cos") is not None]


def nearest_row(rows: list[dict], t: float, tolerance_s: float = MATCH_TOLERANCE_S):
    """The trace row closest in time to `t`, or None if the nearest one is too far away.

    Returning None is a measurement, not a failure: it is how a replay frame gets stamped
    TAG LOST instead of inheriting the numbers from whenever the tag was last seen.
    """
    best, best_dt = None, math.inf
    for r in seen_rows(rows):
        dt = abs(float(r["t"]) - t)
        if dt < best_dt:
            best, best_dt = r, dt
    return best if best_dt <= tolerance_s else None


def evenly_spaced(times: list[float], tiles: int) -> list[int]:
    """Indices of `tiles` frames spread evenly over the trace's DURATION, not over its count.

    Spacing by count would give a stalled stretch of the run the same number of tiles as the
    turn, and the turn is the part anyone is looking at. Endpoints are always included, so a
    filmstrip always shows the pose it started in and the pose it ended in.
    """
    if tiles <= 0 or not times:
        return []
    if tiles == 1 or len(times) == 1:
        return [0]
    if tiles >= len(times):
        return list(range(len(times)))
    t0, t1 = times[0], times[-1]
    if t1 <= t0:
        return list(range(tiles))
    picked: list[int] = []
    for k in range(tiles):
        want = t0 + (t1 - t0) * k / (tiles - 1)
        best = min(range(len(times)), key=lambda i: abs(times[i] - want))
        if best not in picked:
            picked.append(best)
    return sorted(picked)


def rescore(rows: list[dict], **kw):
    """Re-summarise a trace straight from its CSV, without the camera that produced it.

    The summariser is not frozen -- `cos_hold` changed on 2026-08-31 when it turned out to be
    anchored to the last DETECTION rather than to the end of the recording -- and a study that
    can only score a trial at the instant it was captured has to re-run the bench to adopt a
    fix. Every field the summary needs is in the CSV, so it does not.
    """
    from morphohand.bench import tags as T

    readings = [
        T.Reading(t=float(r["t"]), cos_up=float(r["cos"]), deg_from_up=float(r["deg"]),
                  z_bench_mm=float(r["z_bench_mm"]), radial_mm=float(r["radial_mm"]),
                  tag_z_bench_mm=float(r["tag_z_bench_mm"]), margin=float(r["margin"]),
                  range_mm=float(r["range_mm"]),
                  xy_bench_mm=((float(r["x_bench_mm"]), float(r["y_bench_mm"]))
                               if r.get("x_bench_mm") is not None else None))
        for r in seen_rows(rows)]
    kw.setdefault("total_frames", len(rows))
    kw.setdefault("trace_end_s", float(rows[-1]["t"]) if rows else None)
    return T.summarise(readings, **kw)


@dataclass(frozen=True)
class Caption:
    """What gets burned onto one replay frame. Split out so the wording is testable."""
    t_s: float
    seen: bool
    cos: float | None = None
    deg: float | None = None
    z_bench_mm: float | None = None
    peak: bool = False

    def lines(self) -> list[str]:
        if not self.seen:
            return [f"t {self.t_s:5.2f}s", "TAG LOST"]
        return [f"t {self.t_s:5.2f}s",
                f"cos {self.cos:+.3f}{'  PEAK' if self.peak else ''}",
                f"{self.deg:5.1f} deg from up",
                f"z {self.z_bench_mm:6.1f} mm"]


def captions(frame_times: list[float], rows: list[dict],
             tolerance_s: float = MATCH_TOLERANCE_S) -> list[Caption]:
    """One caption per tape frame, with the peak-cosine frame marked.

    The peak is marked on the tape frame NEAREST the peak, which is not usually the frame the
    peak was measured on -- the trace runs at 30 Hz and the tape at a few. Marking the nearest
    is honest for a replay; reading the peak's value off the tape would not be.
    """
    matched = [nearest_row(rows, t, tolerance_s) for t in frame_times]
    peak_i = None
    have = [(i, r) for i, r in enumerate(matched) if r is not None]
    if have:
        peak_i = max(have, key=lambda ir: float(ir[1]["cos"]))[0]
    out = []
    for i, (t, r) in enumerate(zip(frame_times, matched)):
        if r is None:
            out.append(Caption(t_s=t, seen=False))
        else:
            out.append(Caption(t_s=t, seen=True, cos=float(r["cos"]),
                               deg=float(r["deg"]), z_bench_mm=float(r["z_bench_mm"]),
                               peak=(i == peak_i)))
    return out
