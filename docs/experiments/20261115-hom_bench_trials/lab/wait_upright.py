"""Wait until the sphere stands upright and still on the table, then run a command (2026-11-20 12:5x: a twist dropped the
sphere on its side, the pick-up failed, and the owner, gone home, was asked for someone to stand it up; the series should
go on as soon as it stands, without waiting for a message).

Every --every-s the tracker (hom_setup.Measure, the trials' cameras and calibration) measures the object for 1 s; a check
passes when the block's axis is within --max-tilt of vertical, its bottom within --seat-mm of the table, and its centre
within --still-mm of the previous check's. After --steady passing checks in a row (the person who stood it up has let
go and stepped away) the command after '--' runs, and this program exits with its status. One JSON line per check to
--checks. Nothing moves; the cameras are open only during a check.

    /usr/bin/python3 wait_upright.py --map ../maps/20261116-sphere_fitted_map_O.json -- /usr/bin/python3 -u hom_series.py ...
"""
import argparse
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "src")]


def check(map_path, h, table_z):
    """(tilt from vertical deg, block bottom above the table mm, centre in B) or raises when not seen."""
    import hom_setup as HS
    a = types.SimpleNamespace(map=map_path, object="sphere", video_dir=None, plan=None)
    m = HS.Measure(a, video=None, map_file=None)
    try:
        _p, _q, info = m.pose(1.0, settle=0.0)
    finally:
        m.stop()
    tilt = math.degrees(math.acos(float(np.clip(HS.quat_to_mat(np.asarray(info["q_B"], float))[2, 1], -1.0, 1.0))))
    c = np.asarray(info["p_B"], float)
    return tilt, float(c[2]) - h - table_z, c


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("give the command to run after '--'")
    k = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--every-s", type=float, default=30.0)
    ap.add_argument("--steady", type=int, default=3, help="passing checks in a row (with --every-s 30: 60-90 s still)")
    ap.add_argument("--max-tilt", type=float, default=8.0)
    ap.add_argument("--seat-mm", type=float, default=4.0)
    ap.add_argument("--still-mm", type=float, default=1.5)
    ap.add_argument("--table-z", type=float, default=3.0)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--checks", default=str(ROOT / "logs/20261116-hom_trials/wait_upright.jsonl"))
    a = ap.parse_args(argv[:k])
    cmd = argv[k + 1:]
    import hom_setup as HS
    h = HS.map_height(a.map)
    print(f"{time.strftime('%H:%M:%S')} waiting for the sphere upright (<{a.max_tilt:g} deg) and still "
          f"({a.steady} checks {a.every_s:g} s apart within {a.still_mm:g} mm); then: {' '.join(cmd)[:200]}", flush=True)
    t_end, run, last = time.time() + 3600.0 * a.hours, 0, None
    while time.time() < t_end:
        try:
            tilt, bottom, c = check(a.map, h, a.table_z)
            moved = None if last is None else float(np.linalg.norm(c - last))
            ok = tilt < a.max_tilt and abs(bottom) < a.seat_mm and (moved is not None and moved < a.still_mm)
            last, err = c, None
        except BaseException as e:                          # noqa: BLE001  (not seen: SystemExit from Measure)
            tilt = bottom = moved = None
            ok, last, err = False, None, repr(e)[:160]
        run = run + 1 if ok else 0
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "tilt_deg": None if tilt is None else round(tilt, 1),
               "bottom_mm": None if bottom is None else round(bottom, 1),
               "moved_mm": None if moved is None else round(moved, 2), "ok": ok, "in_a_row": run, "error": err}
        with open(a.checks, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{time.strftime('%H:%M:%S')} " + (err if err else f"tilt {tilt:.1f} deg, bottom {bottom:+.1f} mm, moved "
              f"{'-' if moved is None else f'{moved:.2f}'} mm") + f": {'ok' if ok else 'not yet'} ({run}/{a.steady})",
              flush=True)
        if run >= a.steady:
            print(f"{time.strftime('%H:%M:%S')} UPRIGHT AND STILL: running the command", flush=True)
            return subprocess.run(cmd, cwd=HERE).returncode
        time.sleep(a.every_s)
    print("gave up", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
