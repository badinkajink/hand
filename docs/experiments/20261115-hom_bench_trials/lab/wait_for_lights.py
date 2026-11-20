"""Wait for the lab's lights, then run a command (2026-11-19 night: the lights went off at 21:10, the tracker found no
tag and the ranked layouts' series stopped at its first recentre; the lights go off in the evening at times that move,
20:12-21:40 since 11-04).

From --not-before on, every --every-s the tracker runs --check-s with the object's map (hom_trial.start_tracker, the
trials' cameras and calibration): each camera's mean grey level over its frames, and the share of the tracker's samples
that carry the object's pose. A check is lit when every camera's mean is at least its --min-luma (lit about 20 for
836612071918 and 29 for 832412070344 at the tracker's fixed exposure, dark 16 for both on 11-19) and the object is seen
in at least --min-seen of the samples. After --steady lit checks in a row the command after '--' runs, its output going
to this program's output, and this program exits with its status. One JSON line per check to --checks. The cameras are
open only during a check; nothing moves.

    /usr/bin/python3 wait_for_lights.py --map ../maps/20261116-sphere_fitted_map_O.json [--not-before 07:00] \\
        -- /usr/bin/python3 -u hom_series.py ... --go
"""
import argparse
import datetime as dt
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "src")]


def check(map_path, seconds):
    """(mean grey level per camera, share of samples with the object's pose, samples)."""
    import hom_trial as HT
    lum, lock, got = {}, threading.Lock(), []

    def on(smp):
        m = smp.msg
        if m is not None:
            with lock:
                got.append(bool(m.n_tags > 0 and np.all(np.isfinite(m.p_PO))))

    class _A:
        map = map_path
    arm, tracker = HT.start_tracker(_A, on)

    def frame(sn):
        def of(_sn, y, det):
            with lock:
                lum.setdefault(sn, []).append(float(y[::8, ::8].mean()))
        return of
    for w in tracker.workers:
        w.on_frame = frame(w.sn)
    try:
        time.sleep(1.5 + seconds)
    finally:
        tracker.stop()
        arm.stop()
    with lock:
        means = {sn: round(float(np.mean(v)), 1) for sn, v in lum.items()}
        seen = round(sum(got) / len(got), 3) if got else 0.0
        return means, seen, len(got)


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("give the command to run after '--'")
    k = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--not-before", default="07:00", help="local time HH:MM; checks start then")
    ap.add_argument("--every-s", type=float, default=300.0)
    ap.add_argument("--check-s", type=float, default=3.0)
    ap.add_argument("--steady", type=int, default=3)
    ap.add_argument("--min-luma", nargs="+", default=["836612071918=18.3", "832412070344=22"])
    ap.add_argument("--min-seen", type=float, default=0.3)
    ap.add_argument("--checks", default=str(ROOT / "logs/20261116-hom_trials/wait_for_lights.jsonl"))
    a = ap.parse_args(argv[:k])
    import claude_login                                    # the login of the session watching the run (2026-11-20)
    print(claude_login.line(), flush=True)
    cmd = argv[k + 1:]
    floor = {x.split("=")[0]: float(x.split("=")[1]) for x in a.min_luma}
    hh, mm = (int(x) for x in a.not_before.split(":"))
    now = dt.datetime.now()
    start = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if start < now - dt.timedelta(hours=12):
        start += dt.timedelta(days=1)
    print(f"{now:%H:%M:%S} waiting for {start:%Y-%m-%d %H:%M}, then checks every {a.every_s:g} s; the command: "
          f"{' '.join(cmd)[:300]}", flush=True)
    while dt.datetime.now() < start:
        time.sleep(min(60.0, max(1.0, (start - dt.datetime.now()).total_seconds())))
    run = 0
    while True:
        try:
            means, seen, n = check(a.map, a.check_s)
            lit = bool(means) and all(means.get(sn, 0.0) >= f for sn, f in floor.items()) and seen >= a.min_seen
            err = None
        except Exception as e:                              # noqa: BLE001
            means, seen, n, lit, err = {}, 0.0, 0, False, repr(e)[:200]
        run = run + 1 if lit else 0
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "luma": means, "seen": seen, "samples": n, "lit": lit,
               "in_a_row": run, "error": err}
        with open(a.checks, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{time.strftime('%H:%M:%S')} luma {means}, object seen {seen:.2f} of {n}: "
              f"{'lit' if lit else 'dark'} ({run}/{a.steady})" + (f" {err}" if err else ""), flush=True)
        if run >= a.steady:
            print(f"{time.strftime('%H:%M:%S')} LIGHTS ON: running the command", flush=True)
            return subprocess.run(cmd, cwd=HERE).returncode
        time.sleep(a.every_s)


if __name__ == "__main__":
    sys.exit(main())
