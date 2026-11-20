"""A series of HOM trials on the SR2 hand, unattended (owner, 2026-11-16 13:25: "you have authority to independently
experiment now"; the sphere "we can reliably not drop"): for each entry PLAN:PORT:LABEL, the established cycle of
hom_setup.py and hom_trial.py, each step a subprocess whose output is checked before the next runs:

    clear    only for a new layout (pose_sweep.py clear, 2026-11-19): the palm straight up until the fingers, straight
             and in the sweep's poses, clear the object's top by 25 mm; after letgo they can hang beside it
    layout   only when the station's mounts are not the plan's (the gantries move with the fingers straight)
    sweep    only for a new layout with --sweep (pose_sweep.py sweep): 12 free-air poses in about 50 s while the
             boards record, the data for the own-pose term's dependence on the layout
    ready    the tracked recentre and the pre-grasp ramp; READY, or NOT READY within --max-off mm and --max-yaw deg
             (a finger's open pose nudging the object), else the series stops
    trial    hom_trial.py trial --live --arm --ft with --max-drift-mm, recorded into --video-dir
    putdown  the tracked touchdown, the seat and release checks, then letgo (release, straighten, lift): LET GO

Any other outcome (a refusal, a stopped move, a lost object, torque off) stops the series with the hand where it is. One
summary line per trial goes to --out (JSON lines): the corvus record, the object's twist range about the vertical, its
largest rise along z_P (the push into the hand) and its drift in P, from the corvus record alone.

    /usr/bin/python3 hom_series.py --object sphere --map ../maps/20261116-sphere_fitted_map_O.json \\
        --video-dir DIR --out SUMMARY.jsonl PLAN:47003:g10_lat18 PLAN2:47010:g10_lat15 ... [--go]
"""
import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
from sr2hand.geometry.rotations import quat_to_mat  # noqa: E402

PY = "/usr/bin/python3"
TRIALS = ROOT / "logs/20261116-hom_trials"


def run(cmd, timeout):
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr
    keep = [x for x in out.splitlines() if x.strip() and "video:" not in x]
    print("\n".join("   " + x[:220] for x in keep[-8:]), flush=True)
    return r.returncode, out


def summary(record):
    """Twist range (deg), the largest rise along z_P (mm), the final drift in P (mm) and the periods, from a corvus
    record."""
    rows = [json.loads(x) for x in Path(record).read_text().splitlines() if '"seq"' in x]
    R0 = p0 = None
    tw, dz, dr = [], [], []
    for r in rows:
        o = r.get("obj")
        q = np.asarray((o or {}).get("q_PO") or [np.nan] * 4, float)
        if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 0.5 or not np.all(np.isfinite(o.get("p_PO") or [np.nan])):
            continue                                       # a period without a tracked pose
        R = quat_to_mat(np.asarray(o["q_PO"], float))
        p = np.asarray(o["p_PO"], float)
        if R0 is None:
            R0, p0 = R, p
        D = R0.T @ R
        tw.append(math.degrees(math.atan2(D[0, 2] - D[2, 0], D[0, 0] + D[2, 2])))
        dz.append(p[2] - p0[2])
        dr.append(float(np.linalg.norm(p - p0)))
    stops = [r.get("stop") for r in rows if r.get("stop")]
    return {"periods": len(rows), "twist_min_deg": round(min(tw), 1) if tw else None,
            "twist_max_deg": round(max(tw), 1) if tw else None, "rise_max_mm": round(max(dz), 1) if dz else None,
            "drift_end_mm": round(dr[-1], 1) if dr else None, "stop": stops[0] if stops else None}


def layout_differs(plan):
    """Whether the station's mounts differ from the plan's (hom_setup.mounts_off, 1 mm)."""
    sys.path.insert(0, str(HERE))
    import hom_setup as HS
    st = HS.Station("http://10.99.99.2:8765", str(Path.home() / ".sr2_station_token"))
    return bool(HS.mounts_off(HS.load_plan(plan), st.state()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("entries", nargs="+", help="PLAN:PORT:LABEL")
    ap.add_argument("--object", default="sphere")
    ap.add_argument("--map", required=True)
    ap.add_argument("--task", default="twist")
    ap.add_argument("--controller-host", default="100.65.73.59")
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-drift-mm", type=float, default=30.0)
    ap.add_argument("--max-off", type=float, default=3.0)
    ap.add_argument("--max-yaw", type=float, default=5.0)
    ap.add_argument("--upright-deg", type=float, default=8.0,
                    help="after a tilted release, go on when the object stands within this of vertical")
    ap.add_argument("--sweep", action="store_true",
                    help="the free-air pose sweep (pose_sweep.py) at each new layout, after its gantry move")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    common = ["--object", a.object, "--map", a.map]
    for k, e in enumerate(a.entries):
        plan, port, label = e.split(":")
        print(f"\n=== {time.strftime('%H:%M:%S')} trial {k + 1}/{len(a.entries)}: {label} ({plan}, port {port})",
              flush=True)
        if not a.go:
            continue
        # a new layout (2026-11-19): after letgo the straight fingers may hang beside the object, so the palm goes up
        # until they clear it before the gantries move (pose_sweep.py clear), and with --sweep the fingers then run
        # the free-air pose sweep at the new mounts
        new_layout = layout_differs(plan)
        if new_layout:
            rc, out = run([PY, "pose_sweep.py", "clear", "--plan", plan, "--map", a.map, "--go"], 180)
            if rc != 0 or "CLEAR OK" not in out:
                sys.exit(f"STOP: the palm could not clear the object before {label}'s layout")
        # layout loads the plan into the station; its gantries move only where the mounts differ
        rc, out = run([PY, "hom_setup.py", "layout", "--plan", plan, "--go", "--video-dir", a.video_dir], 300)
        if rc != 0 or "applied True" not in out:
            sys.exit(f"STOP: layout failed for {label}")
        if new_layout and a.sweep:
            rc, out = run([PY, "pose_sweep.py", "sweep", "--plan", plan, "--map", a.map, "--go"], 300)
            if rc != 0 or "SWEEP DONE" not in out:
                sys.exit(f"STOP: the free-air sweep failed at {label}'s layout")
        rc, out = run([PY, "hom_setup.py", "ready", "--plan", plan, *common, "--go", "--video-dir", a.video_dir], 300)
        m = re.search(r"(READY for the trial|NOT READY): ([\d.]+) mm, ([-+\d.]+) deg", out)
        if rc != 0 or m is None:
            sys.exit(f"STOP: ready failed for {label}")
        if m.group(1) == "NOT READY" and (float(m.group(2)) > a.max_off or abs(float(m.group(3))) > a.max_yaw):
            sys.exit(f"STOP: {label} not ready ({m.group(2)} mm, {m.group(3)} deg)")
        rc, out = run([PY, "hom_trial.py", "trial", "--task", a.task, "--controller", f"{a.controller_host}:{port}",
                       *common, "--station", "http://10.99.99.2:8765", "--live", "--arm", "--ft", "--out-dir",
                       str(TRIALS), "--video-dir", a.video_dir, "--label", label, "--max-drift-mm",
                       f"{a.max_drift_mm:g}"], 200)
        rec = re.search(r"-> (\S+_corvus\.jsonl)\s*$", out, re.M)
        if rc != 0 or rec is None or "trial done" not in out:
            sys.exit(f"STOP: trial failed for {label}")
        s = summary(rec.group(1))
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": label, "plan": plan, "port": int(port),
               "record": rec.group(1), "ready": [m.group(1), float(m.group(2)), float(m.group(3))], **s}
        print(f"   {label}: twist {s['twist_min_deg']} .. {s['twist_max_deg']} deg, rise {s['rise_max_mm']} mm, "
              f"drift {s['drift_end_mm']} mm, stop {s['stop']}", flush=True)
        rc, out = run([PY, "hom_setup.py", "putdown", "--plan", plan, *common, "--go", "--video-dir", a.video_dir], 400)
        row["putdown"] = "LET GO" if "LET GO" in out else out.strip().splitlines()[-1][:200]
        if rc == 0 and "LET GO" in out:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
        if rc != 0 or "LET GO" not in out:
            if "is not seated" in out:
                # the object held tilted past --seat-deg: let go of it rather than hold it for hours under the servos'
                # load, though it may topple (13:41, 2026-11-16), and stop: a person must stand it up again
                rc2, out2 = run([PY, "hom_setup.py", "letgo", "--plan", plan, *common, "--tilt-ok", "--go",
                                 "--video-dir", a.video_dir], 300)
                # released at 19.0 deg the sphere settled at 5.2 deg (13:50); at 21.6 deg it toppled (13:41): go on
                # when it stands within --upright-deg after the release
                t = re.findall(r"released: .*?tilt ([\d.]+) deg", out2)
                row["putdown"] = f"tilted; letgo --tilt-ok; settled at {t[-1] if t else '?'} deg"
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                if "LET GO" in out2 and t and float(t[-1]) <= a.upright_deg:
                    print(f"   {label}: released tilted, settled at {t[-1]} deg: going on", flush=True)
                    continue
                sys.exit(f"STOP: {label} ended tilted; released with letgo --tilt-ok "
                         f"({'LET GO' if 'LET GO' in out2 else 'letgo did not finish'}, tilt after "
                         f"{t[-1] if t else '?'} deg): stand the object up to go on")
            sys.exit(f"STOP: putdown did not finish for {label}: {row['putdown']}")
    print(f"\n=== {time.strftime('%H:%M:%S')} series done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
