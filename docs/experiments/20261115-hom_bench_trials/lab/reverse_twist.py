"""Play a twist trial's finger targets backwards on the hand (owner at the bench, 2026-11-20 11:2x: "can we try reversing
controls here. just for fun", then "play the twist backward. let's just yolo it, we dont need a dry run").

From the trial's corvus record: the targets it streamed (target_sim_deg), last first, each clipped to the servos' safe
range (sr2hand.control.servo_limits, as hom_trial does), streamed to the station at --slow times the trial's period; the
arm stays still. The object is tracked: its pose in P at the start and the end, and every --every-s in between, to
--out. Both cameras film into --video-dir. Without --go it prints the first and last targets against the measured pose.

    /usr/bin/python3 reverse_twist.py RECORD --plan PLAN --map MAP --video-dir DIR --go [--slow 2]
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "src")]
import hom_setup as HS  # noqa: E402
import hom_trial as HT  # noqa: E402
from sr2hand.control import servo_limits as SL  # noqa: E402

FINGERS, JOINTS = ("thumb", "index", "middle"), ("yaw", "mcp", "pip")


def frame_of(v):
    T = np.asarray(v, float).reshape(3, 3)
    return {f: dict(zip(JOINTS, map(float, T[k]))) for k, f in enumerate(FINGERS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("record")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--object", default="sphere")
    ap.add_argument("--slow", type=float, default=2.0, help="times the trial's period between frames")
    ap.add_argument("--every-s", type=float, default=0.2)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--station", default="http://10.99.99.2:8765")
    ap.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path(a.record).read_text().splitlines() if '"seq"' in x]
    rows = [r for r in rows if "target_sim_deg" in r]
    period = float(np.median(np.diff([r["t"] for r in rows])))
    targets = [frame_of(r["target_sim_deg"]) for r in reversed(rows)]
    st = HS.Station(a.station, a.token_file)
    j = HS.sim_joints(st.state())
    first, last = targets[0], targets[-1]
    print(f"{len(targets)} targets, the trial's period {1e3 * period:.1f} ms, played every {1e3 * period * a.slow:.1f} ms "
          f"({len(targets) * period * a.slow:.1f} s)")
    print("   " + HS.show_pose("measured", j))
    print("   " + HS.show_pose("first (the trial's last)", first))
    print("   " + HS.show_pose("last (the trial's first)", last))
    print(f"   largest gap from the measured pose to the first target: "
          f"{max(abs(first[f][k] - j[f][k]) for f in j for k in j[f]):.1f} deg")
    if not a.go:
        return 0
    a2 = types.SimpleNamespace(map=a.map, object=a.object, video_dir=a.video_dir, plan=a.plan, go=True)
    meas = HS.Measure(a2, video=HS.recorder(a2, "reverse"), map_file=None)
    log, p0, p1 = [], None, None
    station = HT.Station(a.station, a.token_file, live=True)
    try:
        p0, q0, info0 = meas.pose(0.5)
        HS.report("before", p0, q0, *HS.nominal(HS.load_plan(a.plan)), HS.load_plan(a.plan)["meta"]["hom"]["shape"],
                  info0)
        station.start_stream()
        t0, t_log, clipped = time.time(), 0.0, set()
        for k, fr in enumerate(targets):
            c = SL.clip_sim(fr)
            clipped.update(c)
            station.frame(fr, None)
            if time.time() - t_log >= a.every_s:
                s = meas.latest(3, 0.5)
                log.append({"k": k, "t": round(time.time() - t0, 3), "p_PO": None if s is None else
                            np.round(np.asarray(s[0], float), 2).tolist()})
                t_log = time.time()
            time.sleep(period * a.slow)
        time.sleep(1.0)
        try:
            # the start's connection sits idle through the playback and the station closes it (11:27, 2026-11-20)
            station.c = station._conn()
            station.end_stream()
        except Exception as e:                              # noqa: BLE001
            print(f"   ending the stream: {e!r} (its 1 s lease lapses on its own)", flush=True)
            station.stream_token = None
        p1, q1, info1 = meas.pose(0.5)
        plan = HS.load_plan(a.plan)
        HS.report("after", p1, q1, *HS.nominal(plan), plan["meta"]["hom"]["shape"], info1)
        print(f"   moved in P: {np.round(np.asarray(p1) - np.asarray(p0), 2).tolist()} mm; frames posted "
              f"{station.n_frames}, errors {station.errors}; clipped {sorted(clipped)}", flush=True)
        j2 = HS.sim_joints(HS.Station(a.station, a.token_file).state())
        print("   " + HS.show_pose("measured after", j2))
    finally:
        station.stream_token = None
        meas.stop()
        out = a.out or str(Path(a.record).with_name(Path(a.record).stem.replace("_corvus", "") + "_reverse.json"))
        Path(out).write_text(json.dumps({"record": a.record, "slow": a.slow, "path": log,
                                         "p_PO_before": None if p0 is None else np.asarray(p0).tolist(),
                                         "p_PO_after": None if p1 is None else np.asarray(p1).tolist()}, indent=1) + "\n")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
