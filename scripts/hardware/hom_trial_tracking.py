#!/usr/bin/env python3
"""Tracking of the live HOM trials on the SR2 hand, one row per trial (2026-11-16, for the lab session on corvus that
looks for the cause of the poor transfer from simulation): HOM's tracking score as hom_tasks.metrics defines it, the
object's motion in the palm frame P as the sim-real reversal study defines it, the gap between HOM's targets and the
measured joints, the tracker's coverage, and how each trial ended.

Inputs. The HOM server's records (src/sr2hand/control/hom_server.py), docs/experiments/20261115-hom_bench_trials/runs/
<stamp>_<task>_d<d>_trial<n>.jsonl: a header, one line per 50 Hz control period and an end line. corvus's records of
the same trials (runs_corvus/, written by lab/hom_trial.py) are matched by the time of the first state. They give two
things that the server's records lack: corvus's stop reason (its drift guard, the controller's stop) and the number of
targets that reached the station's stream. The plan named in the header gives the layout, the grid and grasp
(meta.hom.pick) and the contact latitude below the object's equator (meta.hom.v[6]; for the press it is the pinching
pads' latitude, and v[7] is the pusher's).

Tracking score. track = max(0, 1 - err / ref), as hom_tasks.metrics and hom_tasks.track_errors compute it in simulation,
over the task's periods from T_LIFT + 0.4 = 2.0 s to the schedule's end t_end (7.8 s for the twist, 5.5 s for the
press). The server's 0.5 s tail after t_end is left out, as the simulated runs end at t_end.
  Sphere tasks (twist, roll): err is the RMS angle between the tracked rotation since the task's start,
      R(q_k) R(q_0)^-1 in P, and the commanded rotation. The commanded rotation is the record's ref.w_ref integrated
      over the periods before each one (sr2hand.geometry.rotations.quat_integrate), and ref is its RMS angle. In
      simulation the measured rotation is the object's angular velocity integrated once per period; the bench has the
      orientation itself.
  Cylinder tasks (press, pinpush): err is the RMS angle between the tracked axis (body z) and the start axis turned by
      -theta_ref about P's x axis (the record's ref.theta_ref, deg); ref is the RMS of |theta_ref|.
A trial that ended before t_end is scored over its periods from 2.0 s, and its window is marked partial. A trial that
ended before 2.0 s is not scored. The contact factor of hom_landscape.track_score is left out, because no record has
the fingertip boards' contacts.

Object motion, as logs/20261116-sim_real_reversal/rev_features.py computes it, so that the numbers compare with that
study's simulated runs. It is measured from the first tracked sample over the whole record: the rise along z_P (mm),
the twist about z_P (the z component of the rotation vector of R(q) R(q_first)^-1, deg), the tilt (the norm of that
vector's x and y components, deg) and the lateral drift (the distance in P's x-y plane, mm). Each is given at the
record's end and at its extreme; the 3-D distance is corvus's drift-guard quantity.

Joint gap: the target minus the measured angle per finger and servo joint (aa, fe1, fe2) in servo degrees, over the
task's periods whose joints came from the servo bus. For fe1 and fe2 a positive gap is a target flexed past the
measured angle. Given per joint are the median and the largest magnitude, and per trial the share of those periods in
which any joint's gap exceeds 10 deg. Tracker coverage over the task's periods: the share of periods whose state
brought a new capture without the LOST flag, the median tag count, the share solved from one camera, and the median
age of the pose.

    .venv/bin/python scripts/hardware/hom_trial_tracking.py                   # every trial from 11:06; JSON and table
    .venv/bin/python scripts/hardware/hom_trial_tracking.py validate --record SIL.jsonl --metrics RES.json --label L

validate scores a simulated trial and compares err and ref with that run's own hom_tasks.metrics. The simulated trial is
the HOM server against the simulated hand on the server's plant; the server writes its record as on the bench, and
sim-hand prints the metrics (ports other than the bench servers' 47002-47014):

    .venv/bin/python -m sr2hand.control.hom_server serve --task twist --d 40 --plant measured --plan PLAN \
        --listen 47302 --reply 127.0.0.1:47301 --log-dir DIR --once &
    .venv/bin/python -m sr2hand.control.hom_server sim-hand --task twist --d 40 --plant measured --plan PLAN \
        --port-out 47302 --port-in 47301 > RES.json

The comparison is added to the JSON's validation list, and later runs of the scoring keep that list. On 2026-11-16 the
reference agreed exactly on four plans. err differed by 0.00 to 0.43 deg. The difference is the error of hom_tasks'
integration of the object's angular velocity sampled once per period: on the same states that integration departed from
the orientation by a median 0.27 to 1.95 deg, and hom_tasks.track_errors fed the exact rotation of each period gave this
script's err.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from sr2hand.control.link import DRY_RUN, FINGERS, SERVO_JOINTS  # noqa: E402
from sr2hand.control.trial_score import T_TASK, phase_w  # noqa: E402
from sr2hand.geometry.rotations import quat_integrate, quat_to_mat, to_scipy  # noqa: E402
from sr2hand.tracking.stream import LOST, SINGLE  # noqa: E402

ROOT = _paths.ROOT
BENCH = ROOT / "docs/experiments/20261115-hom_bench_trials"
RUNS = BENCH / "runs"
CORVUS = BENCH / "runs_corvus"
OUT = ROOT / "logs/20261116-sim_real_reversal/20261116-trial_tracking.json"
SINCE = "20261116-110600"        # the first trial of 2026-11-16 (11:06:09)
GAP_DEG = 10.0                   # deg: a joint's gap counted in gap_share
DRIFT_GUARD_MM = 20.0            # mm: hom_trial.py --max-drift-mm's default, for trials without corvus's record
MATCH_MS = 500.0                 # ms: largest difference of the first states' times for a corvus record to match
IN_PROGRESS_S = 60.0             # s: a record without an end line and newer than this is still being written
CYLINDER = ("press", "pinpush")
JOINTS = [f"{f}.{j}" for f in FINGERS for j in SERVO_JOINTS]


# ------------------------------------------------------------------------------------------------ records

def read_jsonl(path) -> list[dict]:
    """The lines of a JSON-lines file that parse; a record still being written may end in a partial line."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_server(path):
    """(header, periods, end line or None) of a hom_server record."""
    lines = read_jsonl(path)
    head = lines[0] if lines and lines[0].get("kind") == "header" else {}
    end = next((x for x in lines if x.get("kind") == "end"), None)
    return head, [x for x in lines if "kind" not in x and "t" in x], end


def rel(p) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def corvus_index() -> dict:
    """{t0_ms: path} of corvus's records, t0_ms being the time of the record's first state."""
    idx = {}
    for p in sorted(CORVUS.glob("*_corvus.jsonl")):
        with open(p) as fh:
            try:
                h = json.loads(fh.readline())
            except json.JSONDecodeError:
                continue
        if h.get("kind") == "header" and h.get("t0_ms") is not None:
            idx[float(h["t0_ms"])] = p
    return idx


def corvus_info(head: dict, idx: dict) -> dict | None:
    """corvus's record of the trial whose first state was sent at the server header's t0_ms: corvus's stop reason, the
    targets posted to the station's stream against the states sent, the station's error and the arm's lift."""
    t0 = head.get("t0_ms")
    if t0 is None or not idx:
        return None
    k = min(idx, key=lambda x: abs(x - float(t0)))
    if abs(k - float(t0)) > MATCH_MS:
        return None
    lines = read_jsonl(idx[k])
    h = lines[0]
    end = next((x for x in lines if x.get("kind") == "end"), None) or {}
    rows = [x for x in lines if "kind" not in x and "seq" in x]
    stop = next(({"t_s": x.get("t"), "reason": x["stop"]} for x in rows if x.get("stop")), None)
    lift = end.get("lift") or {}
    states = end.get("states", len(rows))
    posted = end.get("frames_posted")
    return {"file": rel(idx[k]), "live": h.get("live"), "arm": h.get("arm"), "states": states,
            "frames_posted": posted,
            "posted_share": round(posted / states, 3) if posted is not None and states else None,
            "stop": stop, "station_error": end.get("station_error"), "frame_errors": end.get("frame_errors"),
            "stream_restarts": end.get("stream_restarts"),
            "lift_start_s": round((float(lift["t_start_ms"]) - float(t0)) / 1000.0, 3)
            if lift.get("t_start_ms") is not None else None,
            "lift_dz_mm": lift.get("dz_mm"), "lift_max_dF_N": lift.get("max_dF_N"), "lift_stopped": lift.get("stopped")}


def plan_info(head: dict) -> dict:
    """The plan named in the header: its family (generalist, film, gr = ground-clear, lat = latitude variant, press),
    layout, grid and grasp, and the contact latitude below the equator (meta.hom.v[6], deg)."""
    path = head.get("plan")
    if not path:
        return {"path": None, "family": "none", "label": "no plan"}
    p = Path(path) if Path(path).is_absolute() else ROOT / path
    if not p.is_file():
        return {"path": path, "family": "missing", "label": "plan file missing"}
    h = json.loads(p.read_text()).get("meta", {}).get("hom") or {}
    pick = h.get("pick") or ""
    m = re.match(r"(grid\w+) grasp (\d+)", pick)
    grid, grasp = (m.group(1), int(m.group(2))) if m else (None, None)
    name = p.name
    if h.get("task") in CYLINDER:
        family = f"{h['task']} " + ("lat0" if "lat0" in name else (h.get("role") or ""))
    elif "_gr_plan" in name:
        family = "gr"
    elif re.search(r"_lat\d+_plan", name):
        family = "lat"
    elif h.get("role") == "generalist":
        family = "generalist"
    elif grid == "grid2dff":
        family = "film"
    else:
        family = h.get("role") or "other"
    v = h.get("v") or []
    lat = math.degrees(v[6]) if len(v) > 6 else None
    xy = h.get("xy_mm")
    who = f"g{grasp}" if grasp is not None else (pick.split()[0] if pick else "?")
    short = {"press lat0": "press lat0", "press generalist": "press S3", "pinpush generalist": "pinpush S3"}
    label = short.get(family, f"{family} {who}") + (f" ({xy[0]:g},{xy[1]:g})" if xy else "")
    started = head.get("started")
    changed = None
    if started:
        t_start = dt.datetime.strptime(started, "%Y-%m-%dT%H:%M:%S%z").timestamp()
        changed = bool(p.stat().st_mtime > t_start + 1.0)
    return {"path": path, "family": family, "label": label, "role": h.get("role"), "layout_mm": xy,
            "call": h.get("call"), "pick": pick, "grid": grid, "grasp": grasp,
            "lat_deg": None if lat is None else round(lat, 2),
            "lat_pusher_deg": round(math.degrees(v[7]), 2) if h.get("task") in CYLINDER and len(v) > 7 else None,
            "modified_after_start": changed}


# ------------------------------------------------------------------------------------------------ scores

def rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def track(task: str, rows: list[dict], prm: dict, h_last: float) -> dict:
    """err, ref and track = max(0, 1 - err / ref) over the window's periods (see the module docstring); for the sphere
    tasks also the commanded and the tracked rotation about z_P at the window's end (deg)."""
    t = np.array([float(r["t"]) for r in rows])
    Q = np.array([r["obj_quat"] for r in rows], float)
    out = {}
    if task in CYLINDER:
        th = np.array([(r.get("ref") or {}).get("theta_ref", np.nan) for r in rows], float)
        if not np.isfinite(th).all():
            return {"error": "a period of the window has no theta_ref"}
        A = quat_to_mat(Q)[:, :, 2]
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        a_ref = Rotation.from_rotvec(np.outer(-np.radians(th), [1.0, 0.0, 0.0])).apply(A[0])
        e = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", A, a_ref), -1.0, 1.0)))
        ref = np.abs(th)
        out["theta_ref_end_deg"] = round(float(th[-1]), 2)
    else:
        W = np.array([(r.get("ref") or {}).get("w_ref") if (r.get("ref") or {}).get("w_ref") is not None
                      else phase_w(r.get("phase"), prm) for r in rows], float)          # rad/s in P
        h = np.diff(t, append=t[-1] + h_last)                                         # each period's length
        Rr = to_scipy(quat_integrate(W * h[:, None]))           # commanded rotation before each period
        Rm = to_scipy(Q) * to_scipy(Q[0]).inv()                 # tracked rotation since the window's start, in P
        e = np.degrees((Rr.inv() * Rm).magnitude())
        ref = np.degrees(Rr.magnitude())
        out["twist_ref_end_deg"] = round(float(np.degrees(Rr[-1].as_rotvec()[2])), 2)
        out["twist_task_end_deg"] = round(float(np.degrees(Rm[-1].as_rotvec()[2])), 2)
    err, rf = rms(e), rms(ref)
    out.update(err_rms_deg=round(err, 2), ref_rms_deg=round(rf, 2),
               score=round(max(0.0, 1.0 - err / rf), 3) if rf > 0 else None)
    return out


def motion(rows: list[dict]) -> dict:
    """rev_features.feats over the whole record, with the extremes of the tilt and the lateral and 3-D drift."""
    P = np.array([r["obj_pos_mm"] if r.get("obj_pos_mm") is not None else [np.nan] * 3 for r in rows], float)
    Q = np.array([r["obj_quat"] if r.get("obj_quat") is not None else [np.nan] * 4 for r in rows], float)
    ok = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
    ok[ok] &= np.linalg.norm(Q[ok], axis=1) > 0.5
    if not ok.any():
        return {}
    P, Q = P[ok], Q[ok]
    R = to_scipy(Q)
    rv = (R * R[0].inv()).as_rotvec()                       # rotation from the first tracked sample, in P
    tw = np.degrees(rv[:, 2])
    tilt = np.degrees(np.linalg.norm(rv[:, :2], axis=1))
    dz = P[:, 2] - P[0, 2]
    lat = np.hypot(P[:, 0] - P[0, 0], P[:, 1] - P[0, 1])
    d3 = np.linalg.norm(P - P[0], axis=1)
    r1 = lambda x: round(float(x), 1)  # noqa: E731
    return {"rise_max_mm": r1(dz.max()), "rise_end_mm": r1(dz[-1]), "twist_end_deg": r1(tw[-1]),
            "twist_max_deg": r1(tw.max()), "twist_min_deg": r1(tw.min()), "tilt_end_deg": r1(tilt[-1]),
            "tilt_max_deg": r1(tilt.max()), "lat_end_mm": r1(lat[-1]), "lat_max_mm": r1(lat.max()),
            "drift_end_mm": r1(d3[-1]), "drift_max_mm": r1(d3.max())}


def joint_gap(rows: list[dict]) -> dict:
    """Target minus measured angle per servo joint over the periods whose joints came from the bus (servo degrees)."""
    bus = [r for r in rows if r.get("joints_src") == "bus"]
    if not bus:
        return {"n_bus": 0}
    J = np.array([[r["joints"][f][j] for f in FINGERS for j in SERVO_JOINTS] for r in bus], float)
    T = np.array([[r["targets"][f][j] for f in FINGERS for j in SERVO_JOINTS] for r in bus], float)
    G = T - J
    per = {n: {"median_deg": round(float(np.nanmedian(G[:, k])), 1),
               "max_abs_deg": round(float(np.nanmax(np.abs(G[:, k]))), 1),
               "share_over_10": round(float(np.mean(np.abs(G[:, k]) > GAP_DEG)), 3)} for k, n in enumerate(JOINTS)}
    return {"n_bus": len(bus), "joints": per,
            "share_over_10": round(float(np.mean(np.nanmax(np.abs(G), axis=1) > GAP_DEG)), 3)}


def tracker(rows: list[dict], sel: np.ndarray) -> dict:
    """Coverage of the tracker over the periods sel of the trial: a period has a fresh pose when its state brought a
    capture newer than the previous period's and without the LOST flag."""
    tc = np.array([r.get("t_capture_ms") if r.get("t_capture_ms") is not None else np.nan for r in rows], float)
    tm = np.array([float(r["t_ms"]) for r in rows])
    fl = np.array([int(r.get("obj_flags") or 0) for r in rows])
    nt = np.array([int(r.get("n_tags") or 0) for r in rows])
    new = np.isfinite(tc) & np.r_[True, np.diff(tc) != 0]
    fresh = new & ((fl & LOST) == 0)
    if not sel.any():
        return {}
    return {"fresh_share": round(float(fresh[sel].mean()), 3), "lost_share": round(float(((fl & LOST) != 0)[sel].mean()), 3),
            "single_camera_share": round(float(((fl & SINGLE) != 0)[sel].mean()), 3),
            "n_tags_median": float(np.median(nt[sel])),
            "pose_age_median_ms": round(float(np.nanmedian((tm - tc)[sel])), 1)}


def ending(rows, end, cor, mode, fragment, in_progress) -> dict:
    """How the trial ended: end of schedule, drift guard, stream lost, fragment, dry run (the fingers did not move)."""
    server = (end or {}).get("reason")
    t_last = float(rows[-1]["t"]) if rows else None
    if in_progress:
        return {"class": "in progress", "short": "running", "detail": "no end line yet", "t_s": t_last}
    if fragment:
        return {"class": "fragment", "short": "fragment", "t_s": t_last,
                "detail": f"opened by the server after a gap in corvus's states (first seq {rows[0].get('seq')}); "
                          "its clock and HOM's schedule restarted at 0"}
    out = {"server_reason": server, "t_s": t_last}
    stop = (cor or {}).get("stop")
    if server == "end of schedule" or (stop and stop["reason"] == "controller"):
        out.update({"class": "end of schedule", "short": "schedule"})
    elif server and server.startswith("controller error"):
        out.update({"class": "controller error", "short": "ctl error", "detail": server})
    elif stop and stop["reason"].startswith("object drifted"):
        m = re.search(r"drifted ([\d.]+) mm .*limit ([\d.]+)", stop["reason"])
        out.update({"class": "drift guard", "short": f"drift {m.group(1)}/{m.group(2)}" if m else "drift",
                    "detail": stop["reason"]})
    elif stop:
        out.update({"class": "corvus stop", "short": "corvus", "detail": stop["reason"]})
    elif cor is not None:
        out.update({"class": "stream lost", "short": "lost",
                    "detail": f"{server}; corvus: station error {cor.get('station_error')}"})
    else:                                             # no corvus record: corvus's drift guard from the record's poses
        mo = motion(rows)
        d = mo.get("drift_end_mm")
        if d is not None and d > DRIFT_GUARD_MM:
            out.update({"class": "drift guard", "short": f"drift {d:.1f}?",
                        "detail": f"inferred: {server}, last pose {d:.1f} mm from the first (corvus's record missing)"})
        else:
            out.update({"class": "stream lost", "short": "lost", "detail": f"{server} (corvus's record missing)"})
    if mode == "dry run":
        out["short"] = "dry, " + out["short"]
    return out


def score_record(path, cidx=None) -> dict:
    """Every quantity of the module docstring for one server record."""
    path = Path(path)
    head, rows, end = load_server(path)
    m = re.match(r"(\d{8}-\d{6})_", path.name)
    rec = {"stamp": m.group(1) if m else path.stem, "file": rel(path), "task": head.get("task"),
           "d_mm": head.get("d_mm"), "started": head.get("started"), "plant": head.get("plant_spec"),
           "git": head.get("git"), "t_end_s": head.get("t_end"), "plan": plan_info(head)}
    if not rows:
        return {**rec, "scored": False, "unscorable": "no periods"}
    dt_ = float(head.get("dt") or 0.02)
    t = np.array([float(r["t"]) for r in rows])
    t_end = float(head.get("t_end") or t[-1] + dt_)
    status = [int(r.get("status") or 0) for r in rows]
    mode = "dry run" if all(s & DRY_RUN for s in status) else "live"
    fragment = int(rows[0].get("seq", 0)) != 0
    in_progress = end is None and time.time() - path.stat().st_mtime < IN_PROGRESS_S
    cor = corvus_info(head, cidx if cidx is not None else corvus_index())
    src = Counter(r.get("joints_src") for r in rows)
    steps = np.round(np.diff(t), 3)
    rec.update({"mode": mode, "periods": len(rows), "periods_server": (end or {}).get("periods"),
                "duration_s": round(float(t[-1]), 2), "joints_src": dict(src),
                "skipped_periods": int(np.sum(np.round(steps[steps > 1.5 * dt_] / dt_) - 1)) if len(steps) else 0,
                "corvus": cor, "ended": ending(rows, end, cor, mode, fragment, in_progress),
                "motion": motion(rows)})
    win = np.flatnonzero((t >= T_TASK - 1e-9) & (t < t_end - 1e-9))
    notes = []
    if mode == "dry run":
        notes.append("dry run: the targets were not sent to the hand")
    if cor and cor.get("posted_share") is not None and cor["posted_share"] < 0.9:
        notes.append(f"targets posted to the stream in {cor['frames_posted']} of {cor['states']} periods")
    if rec["plan"].get("modified_after_start"):
        notes.append("the plan file was modified after the trial started")
    if fragment or in_progress or len(win) < 2:
        why = ("fragment of the trial before it" if fragment else "record still being written" if in_progress
               else f"ended at t = {t[-1]:.2f} s, before the task's start at {T_TASK:.1f} s")
        return {**rec, "scored": False, "unscorable": why, "notes": notes}
    rw = [rows[k] for k in win]
    partial = bool(t[win[-1]] < t_end - dt_ - 1e-9)
    if partial:
        notes.append(f"partial window: {t[win[0]]:.2f}-{t[win[-1]]:.2f} s of {T_TASK:.1f}-{t_end - dt_:.2f} s")
    prm = head.get("prm") or {}
    sel = np.zeros(len(rows), bool)
    sel[win] = True
    rec.update({"scored": True, "notes": notes,
                "window": {"t0_s": float(t[win[0]]), "t1_s": float(t[win[-1]]), "periods": len(win), "partial": partial},
                "track": track(rec["task"], rw, prm, dt_), "joint_gap": joint_gap(rw),
                "tracker_task": tracker(rows, sel), "tracker_trial": tracker(rows, np.ones(len(rows), bool))})
    return rec


# ------------------------------------------------------------------------------------------------ output

def load_out() -> dict:
    try:
        return json.loads(OUT.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_out(obj: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, OUT)


def fmt(x, f="{:.1f}", none="-"):
    return none if x is None else f.format(x)


def print_tables(trials: list[dict]) -> None:
    print("Trials (track = 1 - err/ref over 2.0 s to the schedule's end; motion from the first tracked sample; "
          "gap and tracker over the task)")
    hdr = (f"{'time':8s} {'plan':22s} {'lat':>5s} {'dur':>5s} {'ended':17s} {'n':>4s} {'bus/ref':>8s} {'track':>6s} "
           f"{'err':>5s} {'ref':>5s} {'twist e/max':>12s} {'rise max/e':>11s} {'tilt max/e':>11s} {'lat':>5s} "
           f"{'gap>10':>6s} {'fresh':>5s} {'tags':>4s}")
    print(hdr)
    for r in trials:
        p, mo, tr = r["plan"], r.get("motion") or {}, r.get("track") or {}
        src = r.get("joints_src") or {}
        tk = r.get("tracker_task") or {}
        print(f"{r['stamp'][9:11]}:{r['stamp'][11:13]}:{r['stamp'][13:15]} {p.get('label', '?')[:22]:22s} "
              f"{fmt(p.get('lat_deg')):>5s} {fmt(r.get('duration_s'), '{:.2f}'):>5s} "
              f"{(r.get('ended') or {}).get('short', '?')[:17]:17s} {r.get('periods', 0):4d} "
              f"{str(src.get('bus', 0)) + '/' + str(src.get('reference', 0)):>8s} "
              f"{fmt(tr.get('score'), '{:.3f}'):>6s} {fmt(tr.get('err_rms_deg')):>5s} {fmt(tr.get('ref_rms_deg')):>5s} "
              f"{fmt(mo.get('twist_end_deg'), '{:+.1f}') + '/' + fmt(mo.get('twist_max_deg'), '{:+.1f}'):>12s} "
              f"{fmt(mo.get('rise_max_mm')) + '/' + fmt(mo.get('rise_end_mm')):>11s} "
              f"{fmt(mo.get('tilt_max_deg')) + '/' + fmt(mo.get('tilt_end_deg')):>11s} {fmt(mo.get('lat_end_mm')):>5s} "
              f"{fmt((r.get('joint_gap') or {}).get('share_over_10'), '{:.2f}'):>6s} "
              f"{fmt(tk.get('fresh_share'), '{:.2f}'):>5s} {fmt(tk.get('n_tags_median'), '{:.0f}'):>4s}")
    print("\nJoint gap over the task, target - measured (servo deg), median/largest magnitude; fe1, fe2 > 0: target "
          "flexed past the joint")
    print(f"{'time':8s} {'plan':22s} " + " ".join(f"{n.replace('thumb', 'th').replace('index', 'in').replace('middle', 'mi'):>9s}"
                                                  for n in JOINTS))
    for r in trials:
        g = (r.get("joint_gap") or {}).get("joints")
        if not g:
            continue
        print(f"{r['stamp'][9:11]}:{r['stamp'][11:13]}:{r['stamp'][13:15]} {r['plan'].get('label', '?')[:22]:22s} "
              + " ".join(f"{g[n]['median_deg']:+.0f}/{g[n]['max_abs_deg']:.0f}".rjust(9) for n in JOINTS))
    print(f"\nShare of the task's bus periods with the joint's gap over {GAP_DEG:g} deg (%)")
    for r in trials:
        g = (r.get("joint_gap") or {}).get("joints")
        if not g:
            continue
        print(f"{r['stamp'][9:11]}:{r['stamp'][11:13]}:{r['stamp'][13:15]} {r['plan'].get('label', '?')[:22]:22s} "
              + " ".join(f"{100 * g[n]['share_over_10']:.0f}".rjust(9) for n in JOINTS))
    notes = [(r["stamp"], r.get("unscorable"), r.get("notes") or []) for r in trials]
    if any(u or n for _s, u, n in notes):
        print("\nNotes")
        for s, u, n in notes:
            for x in ([f"not scored: {u}"] if u else []) + n:
                print(f"  {s}: {x}")


def cmd_score(a) -> int:
    cidx = corvus_index()
    paths = sorted(p for p in RUNS.glob("*_trial*.jsonl") if p.name[:15] >= a.since)
    trials = [score_record(p, cidx) for p in paths]
    old = load_out()
    obj = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "script": rel(__file__),
           "runs": rel(RUNS), "corvus": rel(CORVUS), "since": a.since,
           "definitions": __doc__.split("\n\n")[1:5], "validation": old.get("validation", []), "trials": trials}
    write_out(obj)
    print_tables(trials)
    print(f"\n{len(trials)} trials, {sum(r.get('scored', False) for r in trials)} scored -> {rel(OUT)}")
    return 0


def cmd_validate(a) -> int:
    """A simulated trial's record scored here against its own hom_tasks.metrics (sim-hand's printed result)."""
    head, rows, _end = load_server(a.record)
    res = None
    for line in Path(a.metrics).read_text().splitlines():
        try:
            x = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(x, dict) and "track_err_rms_deg" in x:
            res = x
    if res is None:
        raise SystemExit(f"{a.metrics}: no line with track_err_rms_deg")
    t = np.array([float(r["t"]) for r in rows])
    t_end = float(head["t_end"])
    win = np.flatnonzero((t >= T_TASK - 1e-9) & (t < t_end - 1e-9))
    mine = track(head["task"], [rows[k] for k in win], head.get("prm") or {}, float(head.get("dt") or 0.02))
    e_s, r_s = res["track_err_rms_deg"], res["track_ref_rms_deg"]
    entry = {"label": a.label, "record": str(a.record), "plan": head.get("plan"), "task": head.get("task"),
             "plant": head.get("plant_spec"), "periods": len(win),
             "simulation": {"err_rms_deg": e_s, "ref_rms_deg": r_s, "score": round(max(0.0, 1 - e_s / r_s), 3)},
             "record": {k: mine[k] for k in ("err_rms_deg", "ref_rms_deg", "score")},
             "diff_err_deg": round(mine["err_rms_deg"] - e_s, 2), "diff_ref_deg": round(mine["ref_rms_deg"] - r_s, 2)}
    entry["within_0.1_deg"] = bool(abs(entry["diff_err_deg"]) <= 0.1 and abs(entry["diff_ref_deg"]) <= 0.1)
    if a.note:
        entry["note"] = a.note
    obj = load_out()
    obj["validation"] = [v for v in obj.get("validation", []) if v.get("label") != a.label] + [entry]
    obj.setdefault("trials", [])
    write_out(obj)
    print(json.dumps(entry))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default=SINCE, help="first trial stamp scored (YYYYMMDD-HHMMSS)")
    sub = ap.add_subparsers(dest="cmd")
    v = sub.add_parser("validate", help="a simulated trial's record against its own hom_tasks.metrics")
    v.add_argument("--record", required=True, help="hom_server's record of a sim-hand run")
    v.add_argument("--metrics", required=True, help="sim-hand's printed hom_tasks.metrics (JSON)")
    v.add_argument("--label", required=True)
    v.add_argument("--note", default=None, help="a sentence kept with the comparison")
    a = ap.parse_args()
    return cmd_validate(a) if a.cmd == "validate" else cmd_score(a)


if __name__ == "__main__":
    sys.exit(main())
