"""Re-joining a tape of frames to the trace that was recorded alongside it.

The failure this file exists to prevent is a quiet one: a replay frame captioned with
another moment's numbers. Nobody reviewing a video checks the arithmetic behind the text
burned onto it -- that is the whole reason the text is there -- so the join has to be
right before anyone looks.
"""
from __future__ import annotations

import csv

import pytest

from morphohand.bench import replay as R


def write_trace(path, rows):
    fields = ["t", "cos", "deg", "z_bench_mm", "z_sim_mm", "radial_mm",
              "x_bench_mm", "y_bench_mm", "tag_z_bench_mm", "margin", "range_mm"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def seen(t, cos, z=80.0):
    return {"t": t, "cos": cos, "deg": 0.0, "z_bench_mm": z, "margin": 60.0,
            "range_mm": 300.0}


def test_a_frame_carries_its_own_time_and_sorts_by_it():
    names = [R.frame_name(i, t) for i, t in enumerate([0.0, 0.25, 1.5, 12.75, 100.0])]
    assert [R.frame_time_s(n) for n in names] == [0.0, 0.25, 1.5, 12.75, 100.0]
    # Lexical order must BE temporal order: the renderer sorts by name, and a tape that
    # sorted 10 before 2 would replay the run out of sequence.
    assert sorted(names) == names


def test_a_foreign_file_is_not_a_frame(tmp_path):
    with pytest.raises(ValueError):
        R.frame_time_s("probe_ir.png")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "0001_00000100.png").write_text("x")
    good = tmp_path / R.frame_name(1, 0.1)
    good.write_bytes(b"")
    assert R.tape_frames(tmp_path) == [good]


def test_a_blank_cell_stays_blank(tmp_path):
    """x/y is EMPTY for every run whose heading was never resolved. Read as 0.0 it would put
    the shaft at the palm centre, which is a pose, so nothing downstream would notice."""
    p = tmp_path / "t.csv"
    write_trace(p, [seen(0.0, 0.1), {"t": 0.1}])
    rows = R.read_trace(p)
    assert rows[0]["x_bench_mm"] is None
    assert rows[0]["cos"] == 0.1
    assert rows[1]["cos"] is None
    assert R.seen_rows(rows) == [rows[0]]


def test_a_frame_with_no_detection_nearby_is_not_captioned_with_a_stale_one(tmp_path):
    p = tmp_path / "t.csv"
    write_trace(p, [seen(0.0, 0.10), seen(0.05, 0.11), seen(2.0, 0.90)])
    rows = R.read_trace(p)
    assert R.nearest_row(rows, 0.06)["cos"] == 0.11
    # 1.0 s sits in the middle of a dropout. Silently borrowing the 0.11 reading from a
    # second earlier -- or the 0.90 from a second later -- is how a replay comes to show a
    # confident number over a frame with no tag in it.
    assert R.nearest_row(rows, 1.0) is None


def test_tiles_are_spaced_by_time_not_by_frame_count():
    """A run that stalls produces many frames of nothing happening. Spacing by index would
    spend the filmstrip on the stall; the turn is the part anyone is looking at."""
    times = [0.0] + [10.0 + 0.01 * i for i in range(200)]      # one frame, then a long stall
    picks = R.evenly_spaced(times, 4)
    assert picks[0] == 0 and picks[-1] == len(times) - 1
    chosen = [times[i] for i in picks]
    assert chosen[1] > 4.0                                     # not glued to the dense end


def test_the_endpoints_are_always_shown():
    times = [i * 0.25 for i in range(41)]
    for tiles in (1, 2, 3, 8, 41, 99):
        picks = R.evenly_spaced(times, tiles)
        assert picks == sorted(set(picks))
        assert picks[0] == 0
        if tiles > 1:
            assert picks[-1] == len(times) - 1


def test_the_peak_is_marked_once_and_a_dropout_says_so(tmp_path):
    p = tmp_path / "t.csv"
    write_trace(p, [seen(0.0, 0.0), seen(0.5, 0.5), seen(1.0, 0.95), seen(3.0, 0.90)])
    rows = R.read_trace(p)
    caps = R.captions([0.0, 0.5, 1.0, 2.0, 3.0], rows)
    assert [c.seen for c in caps] == [True, True, True, False, True]
    assert sum(c.peak for c in caps) == 1
    assert caps[2].peak
    assert "TAG LOST" in caps[3].lines()
    assert "cos +0.950  PEAK" in caps[2].lines()


def test_a_tape_with_no_matching_trace_still_captions_every_frame(tmp_path):
    """An empty or unreadable trace must not lose the pictures: a run whose CSV is missing
    its detections is exactly the run someone wants to LOOK at."""
    p = tmp_path / "t.csv"
    write_trace(p, [{"t": 0.0}, {"t": 0.1}])
    caps = R.captions([0.0, 0.25, 0.5], R.read_trace(p))
    assert len(caps) == 3 and not any(c.seen for c in caps)
    assert not any(c.peak for c in caps)
