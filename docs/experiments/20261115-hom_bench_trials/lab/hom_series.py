"""A series of HOM trials on the SR2 hand, unattended (owner, 2026-11-16 13:25: "you have authority to independently
experiment now"; the sphere "we can reliably not drop"): for each entry PLAN:PORT:LABEL, the established cycle of
hom_setup.py and hom_trial.py, each step a subprocess whose output is checked before the next runs:

    gate     before each trial, the station's servo diagnostics (no motion): all nine with torque, and with a servo at
             --hot-c (55 C) or more the series waits, the hand idle, until all are below --cool-c (40 C), the owner's
             thermal rule. With --serve, the plan's HOM server is started on the workstation before its first trial
             and stopped after its last, one server at a time
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
import os
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
WS = "humanoid@100.65.73.59"
KEY = str(Path.home() / ".ssh/id_wxie")
SERVE = "bash ~/Programs/hand/logs/20261116-hom_setup/hom_serve_budget.sh"


def ws(cmd, timeout=90):
    """A command on the workstation (the HOM servers' host)."""
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-i", KEY, WS, cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def server_pids(port):
    """The HOM server listening on port (the bracket keeps pgrep from matching its own shell)."""
    _, out = ws(f'pgrep -f "[h]om_server serve .*--listen {port}( |$)"', 30)
    return [int(x) for x in out.split() if x.isdigit()]


def server_start(a, plan, port):
    """The plan's HOM server, started before its first trial (2026-11-19 night: one at a time, 410 MB each, the
    workstation 6 GB short of free memory under the twist ranking's jobs). None when it answers, else the reason."""
    if server_pids(port):
        return None
    task, d, mass, budget = a.serve.split()
    rel = f"docs/experiments/20261115-hom_bench_trials/plans/{Path(plan).name}"
    tag = "s_" + Path(plan).stem.replace("_plan", "")[:48]
    _, out = ws(f"{SERVE} {task} {d} {rel} {port} {tag} {mass} {budget}", 120)
    for _ in range(10):
        if "HOM server:" in out:
            print(f"   server {port}: {[x for x in out.splitlines() if 'HOM server:' in x][-1][:150]}", flush=True)
            return None
        time.sleep(5)
        _, out = ws(f"grep -v -i warn ~/Programs/hand/logs/20261116-hom_setup/20261116-server_{tag}.log | tail -3", 30)
    return out.strip()[-300:]


def server_stop(port):
    for pid in server_pids(port):
        ws(f"kill {pid}", 30)
        print(f"   server {port} stopped (pid {pid})", flush=True)


def control(path):
    """(hold, skip prefixes) from the series' control file, read before every trial (2026-11-20 00:0x: the owner decides in
    the morning whether the layouts without a feasible twist run): a line 'hold' pauses the series before its next
    trial, the hand idle, until the line is gone; a line 'skip PREFIX' skips every entry whose label starts with PREFIX
    (A2bad, A3drop, b2touch, ...). '#' starts a comment. No file: run everything."""
    try:
        lines = [x.split("#")[0].strip() for x in Path(path).read_text().splitlines()]
    except OSError:
        return False, []
    return "hold" in lines, [x.split(None, 1)[1].strip() for x in lines if x.startswith("skip ") and len(x.split(None, 1)) > 1]


def servo_gate(a):
    """The owner's thermal rule (pause at 55 C, go on below 40 C) and torque on all nine servos, from the station's
    diagnostics (one read of each servo, no motion) before each trial. Above --hot-c the series waits with the hand
    where letgo left it (fingers straight, the palm lifted, nothing held), reading every 60 s, until every servo is below
    --cool-c. Returns (None or the reason to stop, the hottest servo's temperature)."""
    sys.path.insert(0, str(HERE))
    import hom_setup as HS
    st = HS.Station("http://10.99.99.2:8765", str(Path.home() / ".sr2_station_token"))
    t0, waited = time.time(), False
    while True:
        try:
            d = st.call("POST", "/api/v1/servos/diagnostics")
        except Exception as e:                             # noqa: BLE001
            return f"no servo diagnostics ({repr(e)[:120]})", None
        T = {k: float(v) for k, v in (d.get("temperature") or {}).items() if v is not None}
        off = [k for k, v in (d.get("torque_enable") or {}).items() if not v]
        if off or d.get("no_answer") or len(T) < 9:
            return f"servos without torque {off}, no answer {d.get('no_answer')}, temperatures {T}", None
        hot = max(T.values())
        if hot < (a.cool_c if waited else a.hot_c):
            if waited:
                print(f"   {time.strftime('%H:%M:%S')} servos at most {hot:.0f} C: going on", flush=True)
            return None, hot
        if not waited:
            print(f"   {time.strftime('%H:%M:%S')} a servo at {hot:.0f} C ({T}): waiting until all are below "
                  f"{a.cool_c:g} C", flush=True)
        waited = True
        if time.time() - t0 > a.cool_wait_s:
            return f"servos still at {hot:.0f} C after {a.cool_wait_s / 60:.0f} min", hot
        time.sleep(60)


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


def pickup(a, plan, label):
    """Stand a toppled sphere up again by the controlled-slip pick-up (pinch_pickup.py, the skill's steps; 2026-11-19
    19:28 after g13 at (105, 70) toppled it): plan at the station's layout (the trial's plan as the base), contacts 5,
    4, 6 then 3 deg above the equator until the touch keeps 8 mm off the table on both fingers and the joints 4 deg from
    their limits; then grasp, lift, slip, putdown (--level-max 0) and unturn, each checked for its own success line.
    Returns None when the sphere stands again, else the reason."""
    pm = ["--map", a.map]
    best = None
    for lat in ("5", "4", "6", "3"):
        rc, out = run([PY, "pinch_pickup.py", "plan", *pm, "--gap", "8", "--lat-deg", lat, "--base-plan", plan,
                       "--turn"], 240)
        m_margin = re.search(r"joint margin ([\d.]+) deg", out)
        tables = [float(x) for x in re.findall(r"(?:thumb|index|middle) table ([\d.]+)", out.split("at touch:")[-1].split("\n")[0])] \
            if "at touch:" in out else []
        m_plan = re.search(r"plan -> (\S+_plan\.json)", out)
        if rc == 0 and m_margin and m_plan and tables and float(m_margin.group(1)) >= 4.0 and min(tables) >= 8.0:
            best = m_plan.group(1)
            break
    if best is None:
        return "no pick-up plan keeps 8 mm off the table and 4 deg from the joint limits"
    for step, token, extra, tmo in (("grasp", "GRASPED", [], 400), ("lift", "LIFTED", [], 400),
                                    ("slip", "SLIP DONE", [], 500)):
        rc, out = run([PY, "pinch_pickup.py", step, "--plan", best, *pm, "--go", "--video-dir", a.video_dir], tmo)
        if rc != 0 or token not in out:
            return f"the pick-up's {step} did not finish"
    rc, out = run([PY, "hom_setup.py", "putdown", "--plan", best, "--object", "sphere", *pm, "--level-max", "0", "--go",
                   "--video-dir", a.video_dir], 450)
    if rc != 0 or "LET GO" not in out:
        return "the pick-up's putdown did not finish"
    rc, out = run([PY, "pinch_pickup.py", "unturn", "--plan", best, *pm, "--go", "--video-dir", a.video_dir], 300)
    if rc != 0 or ("UNTURNED" not in out and "no turn" not in out.lower()):
        return "the pick-up's unturn did not finish"
    print(f"   {label}: the sphere picked up and set down upright ({Path(best).name})", flush=True)
    return None


def station_plan_file():
    """The file name of the plan the station holds (hom_setup.py layout records it in the plan's meta.plan_file); plans
    can share a design name and differ in their poses (the mcpstop variants of 11-17)."""
    sys.path.insert(0, str(HERE))
    import hom_setup as HS
    pl = HS.Station("http://10.99.99.2:8765", str(Path.home() / ".sr2_station_token")).state().get("plan") or {}
    return (pl.get("meta") or {}).get("plan_file")


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
    ap.add_argument("--ready-args", default="",
                    help="extra arguments for ready, e.g. \"--watch-mm 3\": the descent's watch stopped at a 1.98 mm "
                         "apparent move while the object stood still in B (19:18, 2026-11-19; fingers hiding tags)")
    ap.add_argument("--putdown-args", default="",
                    help="extra arguments for putdown and letgo, e.g. \"--no-centre\" while another object stands at the "
                         "table's centre (2026-11-19)")
    ap.add_argument("--pickup", action="store_true",
                    help="a sphere that toppled at a release is picked up and stood up again (pinch_pickup.py) and the "
                         "series goes on; without it the series stops")
    ap.add_argument("--sweep", action="store_true",
                    help="the free-air pose sweep (pose_sweep.py) at each new layout, after its gantry move")
    ap.add_argument("--sweep-args", default="",
                    help="extra arguments for pose_sweep.py sweep, e.g. \"--ff-mm 30\": at (70, 90) the thumb and index "
                         "MCPs stalled 6-7.5 deg short at curl 40/40 with the model's fingers 23 mm apart (21:14, 11-19)")
    ap.add_argument("--sweep-only", action="store_true",
                    help="clear, layout and the free-air sweep at each entry's layout, no trial (2026-11-19 21:1x: the "
                         "lab's lights go off at night and the tracker sees nothing; the sweep needs no camera)")
    ap.add_argument("--serve", default="",
                    help="'TASK D MASS_G BUDGET', e.g. 'twist 40 28.8 0.56': start each plan's HOM server on the "
                         "workstation before its first trial and stop it after its last (hom_serve_budget.sh)")
    ap.add_argument("--hot-c", type=float, default=55.0, help="pause before a trial with a servo this hot")
    ap.add_argument("--cool-c", type=float, default=40.0, help="go on when every servo is below this")
    ap.add_argument("--cool-wait-s", type=float, default=5400.0)
    ap.add_argument("--cam-mode", default="1920x1080@30", help="the tracker's camera mode WxH@FPS (960x540@60)")
    ap.add_argument("--cam-exposure", type=float, default=None,
                    help="the cameras' exposure in 0.1 ms (50 = 5 ms); the calibration's 166 without it")
    ap.add_argument("--level-hand-deg", type=float, default=2.0,
                    help="z_P's tilt above which the hand is levelled before the recentre (clear, place, approach --fine)")
    ap.add_argument("--control", default=str(HERE / "series_control.txt"),
                    help="the control file read before every trial: 'hold' pauses, 'skip PREFIX' skips labels")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    import claude_login                                    # the login of the session watching the run (2026-11-20)
    print(claude_login.line(), flush=True)
    # the tracker's camera mode for every step this series runs (hom_trial.cam_mode reads it; 2026-11-20, owner: "id
    # like to prioritize (within reason) the 60fps tracking"), written into each row
    os.environ["HOM_CAM_MODE"] = a.cam_mode
    if a.cam_exposure:
        os.environ["HOM_CAM_EXPOSURE"] = f"{a.cam_exposure:g}"
    else:
        os.environ.pop("HOM_CAM_EXPOSURE", None)
    camera = f"{a.cam_mode}, exposure {a.cam_exposure or 166:g}"
    print(f"camera: {camera}", flush=True)
    common = ["--object", a.object, "--map", a.map]
    served = None                                          # the port whose HOM server this series started last
    for k, e in enumerate(a.entries):
        plan, port, label = e.split(":")
        print(f"\n=== {time.strftime('%H:%M:%S')} trial {k + 1}/{len(a.entries)}: {label} ({plan}, port {port})",
              flush=True)
        hold, skips = control(a.control)
        if a.go and hold:
            print(f"   {time.strftime('%H:%M:%S')} held by {a.control}: waiting", flush=True)
            while control(a.control)[0]:
                time.sleep(30)
            print(f"   {time.strftime('%H:%M:%S')} hold lifted", flush=True)
            hold, skips = control(a.control)
        if any(label.startswith(x) for x in skips):
            print(f"   skipped: {label} ({a.control}: skip)", flush=True)
            continue
        if not a.go:
            continue
        if a.serve and served is not None and served != port and not a.sweep_only:
            server_stop(served)
            served = None
        why, servo_c = servo_gate(a)
        if why:
            sys.exit(f"STOP before {label}: {why}")
        if a.serve and not a.sweep_only:
            why = server_start(a, plan, port)
            if why:
                sys.exit(f"STOP: the HOM server for {label} did not start: {why}")
            served = port
        # a new layout (2026-11-19): after letgo the straight fingers may hang beside the object, so the palm goes up
        # until they clear it before the gantries move (pose_sweep.py clear), and with --sweep the fingers then run
        # the free-air pose sweep at the new mounts
        new_layout = layout_differs(plan)
        if new_layout:
            rc, out = run([PY, "pose_sweep.py", "clear", "--plan", plan, "--map", a.map, "--go"], 180)
            if rc != 0 or "CLEAR OK" not in out:
                sys.exit(f"STOP: the palm could not clear the object before {label}'s layout")
        # layout loads the plan into the station; its gantries move only where the mounts differ. Skipped when the
        # station holds this plan at its mounts already (a plan's second and third trials)
        if new_layout or station_plan_file() != Path(plan).name:
            rc, out = run([PY, "hom_setup.py", "layout", "--plan", plan, "--go", "--video-dir", a.video_dir], 300)
            if rc != 0 and "neither the zero nor the open pose" in out:
                # the fingers left partly straight by letgo's partial release (2026-11-20 12:1x): straightened here, the
                # palm already clear of the object (zero refuses while the fingers are round it), then layout again
                rc, out = run([PY, "hom_setup.py", "zero", "--plan", plan, *common, "--go", "--video-dir", a.video_dir],
                              120)
                if rc != 0:
                    sys.exit(f"STOP: the fingers could not be straightened before {label}'s layout")
                rc, out = run([PY, "hom_setup.py", "layout", "--plan", plan, "--go", "--video-dir", a.video_dir], 300)
            if rc != 0 or "applied True" not in out:
                sys.exit(f"STOP: layout failed for {label}")
        if new_layout and (a.sweep or a.sweep_only):
            rc, out = run([PY, "pose_sweep.py", "sweep", "--plan", plan, "--map", a.map, *a.sweep_args.split(), "--go"],
                          300)
            if rc != 0 or "SWEEP DONE" not in out:
                sys.exit(f"STOP: the free-air sweep failed at {label}'s layout")
        if a.sweep_only:
            continue
        # the hand level before the recentre (2026-11-20: a putdown that levels a tilted object tilts the hand, and
        # letgo levels it again only within its --max-level, so t2 and t3 at (90, 135) g3 and t2 at (105, 100) g15
        # started 12-25 deg off the planned grasp): above --level-hand-deg the fingers are lifted clear of the object,
        # the hand levelled about the palm in steps at the grasp's height + 80 mm (place), and moved over the object at
        # that height (approach --fine, no descent); the recentre then descends as after a new layout
        rc, out = run([PY, "hom_setup.py", "status"], 60)
        t = re.search(r"z_P tilted ([\d.]+) deg from base z", out)
        if rc != 0 or t is None:
            sys.exit(f"STOP: no arm pose before {label}'s recentre")
        if float(t.group(1)) > a.level_hand_deg:
            print(f"   the hand is tilted {float(t.group(1)):.1f} deg: lifted clear, levelled and moved over the object "
                  "first", flush=True)
            for cmd, ok in (([PY, "pose_sweep.py", "clear", "--plan", plan, "--map", a.map, "--go"], "CLEAR OK"),
                            ([PY, "hom_setup.py", "place", "--plan", plan, "--map", a.map, "--max-level", "45", "--go",
                              "--video-dir", a.video_dir], "palm at ["),
                            ([PY, "hom_setup.py", "approach", "--plan", plan, *common, "--fine", "--fine-max", "150",
                              "--go", "--video-dir", a.video_dir], "after the lateral move")):
                rc, out = run(cmd, 240)
                if rc != 0 or ok not in out:
                    sys.exit(f"STOP: levelling the hand before {label}'s recentre: {cmd[1]} {cmd[2]} did not finish")
        rc, out = run([PY, "hom_setup.py", "ready", "--plan", plan, *common, *a.ready_args.split(), "--go",
                       "--video-dir", a.video_dir], 300)
        if rc != 0 and "too far to move" in out:
            # an object put back by hand away from the hand (2026-11-20 11:5x, 98 mm off; the owner wants resets
            # automated): the fingers lifted clear of it, the hand moved over it at that height, and ready again
            print(f"   {label}: the object stands too far for the recentre: lifted clear and moved over it first",
                  flush=True)
            for cmd, ok in (([PY, "pose_sweep.py", "clear", "--plan", plan, "--map", a.map, "--go"], "CLEAR OK"),
                            ([PY, "hom_setup.py", "approach", "--plan", plan, *common, "--fine", "--fine-max", "150",
                              "--go", "--video-dir", a.video_dir], "after the lateral move")):
                rc, out = run(cmd, 240)
                if rc != 0 or ok not in out:
                    sys.exit(f"STOP: moving over the object before {label}'s recentre: {cmd[1]} {cmd[2]} did not finish")
            rc, out = run([PY, "hom_setup.py", "ready", "--plan", plan, *common, *a.ready_args.split(), "--go",
                           "--video-dir", a.video_dir], 300)
        m = re.search(r"(READY for the trial|NOT READY): ([\d.]+) mm, ([-+\d.]+) deg", out)
        if rc != 0 or m is None:
            sys.exit(f"STOP: ready failed for {label}")
        tl = re.findall(r"ready: object in P .*?tilt ([\d.]+) deg", out)
        ready_tilt = float(tl[-1]) if tl else None              # the object's axis against z_P at the trial's start
        if ready_tilt is not None and ready_tilt > 5.0:
            print(f"   {label}: the object stands {ready_tilt:.1f} deg off the hand's axis at the start", flush=True)
        if m.group(1) == "NOT READY" and (float(m.group(2)) > a.max_off or abs(float(m.group(3))) > a.max_yaw):
            sys.exit(f"STOP: {label} not ready ({m.group(2)} mm, {m.group(3)} deg)")
        # --plan: the trial's fingertip estimator places the boards' contacts with the plan's mounts (since 11-17)
        rc, out = run([PY, "hom_trial.py", "trial", "--task", a.task, "--controller", f"{a.controller_host}:{port}",
                       *common, "--plan", plan, "--station", "http://10.99.99.2:8765", "--live", "--arm", "--ft", "--out-dir",
                       str(TRIALS), "--video-dir", a.video_dir, "--label", label, "--max-drift-mm",
                       f"{a.max_drift_mm:g}"], 200)
        rec = re.search(r"-> (\S+_corvus\.jsonl)\s*$", out, re.M)
        if rc != 0 or rec is None or "trial done" not in out:
            sys.exit(f"STOP: trial failed for {label}")
        s = summary(rec.group(1))
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": label, "plan": plan, "port": int(port),
               "record": rec.group(1), "ready": [m.group(1), float(m.group(2)), float(m.group(3))],
               "ready_tilt_deg": ready_tilt, "camera": camera, "servo_c_before": servo_c, **s}
        print(f"   {label}: twist {s['twist_min_deg']} .. {s['twist_max_deg']} deg, rise {s['rise_max_mm']} mm, "
              f"drift {s['drift_end_mm']} mm, stop {s['stop']}", flush=True)
        rc, out = run([PY, "hom_setup.py", "putdown", "--plan", plan, *common, *a.putdown_args.split(), "--go",
                       "--video-dir", a.video_dir], 400)
        row["putdown"] = "LET GO" if "LET GO" in out else out.strip().splitlines()[-1][:200]
        if "the object is not in the hand" in out:
            # dropped during the trial (2026-11-20 12:36): the fingers straightened (zero refuses while they are round
            # the object) and the hand lifted clear, then the pick-up when one plans at this layout, else a stop with
            # the hand safe
            row["putdown"] = "dropped in the trial: " + out.strip().splitlines()[-1][:150]
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            rc, out = run([PY, "hom_setup.py", "zero", "--plan", plan, *common, "--go", "--video-dir", a.video_dir], 120)
            if rc != 0:
                sys.exit(f"STOP: {label} dropped the object, and the fingers could not be straightened over it")
            rc, out = run([PY, "pose_sweep.py", "clear", "--plan", plan, "--map", a.map, "--max-lift", "150", "--go"], 180)
            why = pickup(a, plan, label) if a.pickup and a.object == "sphere" else "no --pickup"
            if why is None:
                continue
            sys.exit(f"STOP: {label} dropped the object; the fingers straight and the hand lifted; the pick-up: {why}")
        if "LET GO" not in out and "the levelling stopped" in out:
            # 2026-11-20 09:21: the levelling turn stopped on the object moving 2.18 mm against the hand (limit 2), and
            # the series stopped with the sphere held 49 mm above the table until 10:38 (one servo at 52 C, three
            # flagging overload). The same putdown without the levelling set it down and let go at 10:39
            print(f"   {label}: the levelling stopped; the putdown again without levelling", flush=True)
            rc, out = run([PY, "hom_setup.py", "putdown", "--plan", plan, *common, *a.putdown_args.split(),
                           "--level-max", "0", "--go", "--video-dir", a.video_dir], 400)
            row["putdown"] = "levelling stopped; without levelling: " + (
                "LET GO" if "LET GO" in out else out.strip().splitlines()[-1][:160])
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
                # the block's tilt from vertical after letgo (not its tilt in P, which a tilted hand inflates)
                t = (re.findall(r"settled: the block's axis ([\d.]+) deg from vertical", out2)
                     or re.findall(r"released: .*?tilt ([\d.]+) deg", out2))
                row["putdown"] = f"tilted; letgo --tilt-ok; settled at {t[-1] if t else '?'} deg"
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                if "LET GO" in out2 and t and float(t[-1]) <= a.upright_deg:
                    print(f"   {label}: released tilted, settled at {t[-1]} deg: going on", flush=True)
                    continue
                if a.pickup and "LET GO" in out2 and a.object == "sphere":
                    why = pickup(a, plan, label)
                    if why is None:
                        continue
                    sys.exit(f"STOP: {label} toppled and the pick-up failed: {why}")
                sys.exit(f"STOP: {label} ended tilted; released with letgo --tilt-ok "
                         f"({'LET GO' if 'LET GO' in out2 else 'letgo did not finish'}, tilt after "
                         f"{t[-1] if t else '?'} deg): stand the object up to go on")
            sys.exit(f"STOP: putdown did not finish for {label}: {row['putdown']}")
    if a.go and a.serve and served is not None:
        server_stop(served)
    print(f"\n=== {time.strftime('%H:%M:%S')} series done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
