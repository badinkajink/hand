"""Scores of a recorded HOM trial on the SR2 hand: the inputs of hom_landscape.track_score and J, computed from the
stream that hom_bench.simulate writes and the bench records in the same format.

A trial file is JSON lines. The first line is {"header": {...}} (or hom_server.py's {"kind": "header", ...}); each later
line is one 20 ms control period:
  t            s since the run started
  joints       per finger aa, fe1, fe2 in servo degrees (hom_bench.servo_from_sim)
  obj_pos_mm   the object's position in the palm frame {P}, mm
  obj_quat     the object's orientation in {P}, wxyz
  contacts     per finger [centre of pressure in {P} in mm, normal force in N]; a finger without contact is absent,
               and a sample without the field had no fingertip reading
  targets      per finger aa, fe1, fe2 in servo degrees, as sent
  phase        the schedule's segment (hom_tasks.Schedule), as the controller named it
The header names the task, d_mm, the layout's mounts (mounts_mm: per finger x and y in {P}, mm), the grasp, the plant,
the seed and dt. Optional: prm (hom_tasks.PRM's values where the run changed them), object_mass_kg (default the
simulated object's 25 g) and gravity_P (gravity in {P}, m/s^2; default (0, 0, -9.81), the simulator's palm faces down).
The replay of scripts/hardware/sr2_hom_replay.py also reads palm_m ({P}'s origin in the model's world, m),
target_delay_s, post_off_s and hold_s from the header, and t_joints and t_obj (the times the joints were read and the
tracker's frame was captured, s, on the clock of t) from a sample.

score_stream follows hom_tasks.metrics over the task's samples, t - t0 >= T_LIFT + 0.4 = 2.0 s:
  track_err_rms_deg     RMS of the angle between the object's orientation and the commanded one
  track_ref_rms_deg     RMS of the commanded rotation's angle from the task's start
  contact_all_frac      share of the task's samples on which every finger's pad force exceeds f_min (0.05 N, as in
                        simulation); the rail roll counts its rollers only. contact_all_frac_025 uses 0.25 N, the
                        force below which the fingertip boards report no contact
  dropped               the object fell more than 15 mm along gravity between the first and the last sample
  fingers_end           fingers whose pad force at the last sample is at least half the object's weight
  held                  not dropped, and fingers_end >= 2 (with no fingertip reading at the last sample: not dropped)
  J                     hom_landscape.track_score: held x max(0, 1 - err / ref) x contact_all_frac; J_025 takes the
                        share at 0.25 N and J_nc no contact factor. J is None when no sample has a fingertip reading.
The commanded rotation is the task schedule's. For the cylinder tasks (pinpush, press) the commanded axis is the axis at
the task's start turned by the schedule's pinch angle about {P}'s x axis (a sample's ref.theta_ref when the record has
it), compared with the tracked axis by hom_tasks.track_errors. For the sphere tasks (roll, twist) the commanded angular
velocity of each sample (its ref.w_ref, else that of its phase) is integrated over the sample intervals and compared
with the tracked rotation q_k q_0^-1.

hom_tasks.metrics computes the same quantities from the simulator's state, with two differences. For the spheres it
integrates the object's angular velocity sampled once per period, where the stream has the orientation itself. Its
fingers_end counts each finger's force on all its bodies, links included; the stream has the pads' forces only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from sr2hand.geometry.rotations import quat_integrate, quat_to_mat, to_scipy

FINGERS = ("thumb", "index", "middle")
F_MIN, F_BOARD = 0.05, 0.25         # N: the simulation's contact threshold; the fingertip boards' reporting floor
FALL_MM = 15.0                      # mm: hom_tasks.metrics' drop
T_TASK = 2.0                        # s: hom_tasks.T_LIFT + 0.4, the task's start
MASS_KG = 0.025                     # kg: the simulated object (hand_object_scale_sim.MASS)


def _tasks():
    """hom_tasks (the schedule, PRM and track_errors), imported through hom_bench's paths as HomBench imports it."""
    from . import hom_bench as HB
    HB._paths()
    import hom_tasks as T
    return T


# ------------------------------------------------------------------------------------------------ trial files

def load_trial(path) -> tuple[dict, list[dict]]:
    """(header, samples) of a trial file: JSON lines whose first line is {"header": ...} or, as hom_server.py records a
    trial, {"kind": "header", ...} (its {"kind": "end", ...} line is dropped); a JSON object {"header": ...,
    "stream": [...]}; or a bare stream (header {})."""
    text = Path(path).read_text()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:                          # JSON lines
        obj = None
    if isinstance(obj, dict) and "stream" in obj:
        return obj.get("header", {}), obj["stream"]
    lines = [json.loads(x) for x in text.splitlines() if x.strip()]
    header = {}
    if lines and "header" in lines[0] and "t" not in lines[0]:
        header, lines = lines[0]["header"], lines[1:]
    elif lines and lines[0].get("kind") == "header":
        header, lines = {k: v for k, v in lines[0].items() if k != "kind"}, lines[1:]
    return header, [x for x in lines if "kind" not in x]


def write_trial(path, header: dict, stream: list[dict]) -> None:
    """A trial file: the header line, then one line per sample (numpy values as lists)."""
    def enc(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(json.dumps({"header": header}, default=enc) + "\n")
        for smp in stream:
            fh.write(json.dumps(smp, default=enc) + "\n")


# ------------------------------------------------------------------------------------------------ commanded rotation

def phase_w(phase: str, prm: dict) -> np.ndarray | None:
    """The sphere tasks' commanded angular velocity (rad/s, {P}) of a phase as hom_tasks.Schedule names it: 'wy +',
    'wz - 2' (w_step_dps about that axis), 'twist ramp' (ramp_dps about z), any other phase zero; None without one."""
    if not isinstance(phase, str):
        return None
    w = np.zeros(3)
    if phase == "twist ramp":
        w[2] = math.radians(prm["ramp_dps"])
    elif len(phase) >= 4 and phase[0] == "w" and phase[1] in "xyz" and phase[2] == " " and phase[3] in "+-":
        w["xyz".index(phase[1])] = (1.0 if phase[3] == "+" else -1.0) * math.radians(prm["w_step_dps"])
    return w


def _w_ref(task, samples, t, prm, T, header):
    """Per sample the commanded angular velocity: the schedule's reference when the record has it (hom_server.py's
    ref.w_ref), else from its phase, else from the schedule at its time. The last can pick the neighbouring segment
    at a boundary: the controller summed its time per period, and the stream's t is rounded to 0.1 ms (on the twist
    of 2026-11-15 that moved the reference's RMS from 59.40 to 58.69 deg)."""
    sch = None
    out = np.zeros((len(samples), 3))
    for k, smp in enumerate(samples):
        ref = smp.get("ref") or {}
        w = np.asarray(ref["w_ref"], float) if ref.get("w_ref") is not None else phase_w(smp.get("phase"), prm)
        if w is None:
            if sch is None:
                r = float(header.get("d_mm", 40.0)) / 2000.0
                stub = {"r": r, "n_out": {f: np.array([1.0, 0.0, 0.0]) for f in FINGERS}}
                sch = T.Schedule(task, stub, prm)
            _cmd, info = sch(T_TASK + t[k], {"quat": np.asarray(smp["obj_quat"], float)})
            w = np.asarray(info.get("w_ref", np.zeros(3)), float)
        out[k] = w
    return out


def track_errors(task, samples, t, prm, header, T=None):
    """Per task sample (deg): the angle between the tracked orientation and the commanded one, and the commanded
    rotation's angle from the task's start; t is each sample's time from the task's start (s). Samples without a
    finite quaternion are left out of both; (None, None) when the first task sample has none."""
    T = T or _tasks()
    Q = np.array([s.get("obj_quat") or [np.nan] * 4 for s in samples], float)
    ok = np.isfinite(Q).all(1)
    if not len(Q) or not ok[0]:
        return None, None
    if task in ("pinpush", "press"):
        sch = T.Schedule(task, None, prm)
        th = np.array([(s.get("ref") or {}).get("theta_ref", sch.angle(max(0.0, tt))[0]) for s, tt in zip(samples, t)],
                      float)
        R = quat_to_mat(Q[ok])
        rows = [{"axis": R[i][:, 2].tolist(), "theta_ref_deg": float(th[k])} for i, k in enumerate(np.flatnonzero(ok))]
        e, ref = T.track_errors(task, rows, None)
        return np.asarray(e, float), np.asarray(ref, float)
    W = _w_ref(task, samples, t, prm, T, header)
    dt = np.diff(t, append=t[-1] + (np.median(np.diff(t)) if len(t) > 1 else 0.0))
    Rr = to_scipy(quat_integrate(W * dt[:, None]))          # before each sample, as hom_tasks.track_errors
    Rm = to_scipy(Q[ok]) * to_scipy(Q[0]).inv()
    e = np.degrees((Rr[np.flatnonzero(ok)].inv() * Rm).magnitude())
    return e, np.degrees(Rr[np.flatnonzero(ok)].magnitude())


# ------------------------------------------------------------------------------------------------ scores

def _pad_forces(smp) -> np.ndarray | None:
    c = smp.get("contacts")
    if c is None:
        return None
    return np.array([float(c[f][1]) if f in c and c[f] is not None else 0.0 for f in FINGERS])


def score_stream(stream: list[dict], header: dict, f_min: float = F_MIN, f_board: float = F_BOARD,
                 fall_mm: float = FALL_MM) -> dict:
    """track_score's inputs (held, track_err_rms_deg, track_ref_rms_deg, contact_all_frac) and J of a recorded stream,
    with the drop, the loaded fingers at the end, the three-pad share at the boards' floor and J at that share (see
    the module docstring). Rounded as hom_tasks.metrics rounds them."""
    T = _tasks()
    task = header["task"]
    prm = {**T.PRM, **(header.get("prm") or {})}
    t_all = np.array([float(s["t"]) for s in stream])
    t_all = t_all - t_all[0]
    g = np.asarray(header.get("gravity_P") or (0.0, 0.0, -9.81), float)
    mass = float(header.get("object_mass_kg") or MASS_KG)
    out = {"task": task, "n": len(stream), "f_min": f_min, "f_board": f_board}

    P = np.array([s.get("obj_pos_mm") or [np.nan] * 3 for s in stream], float)
    okp = np.flatnonzero(np.isfinite(P).all(1))
    fall = float((P[okp[-1]] - P[okp[0]]) @ (g / np.linalg.norm(g))) if len(okp) else None
    out["fall_mm"] = round(fall, 2) if fall is not None else None
    out["dropped"] = bool(fall is not None and fall > fall_mm)
    F_end = _pad_forces(stream[-1])
    if F_end is None:
        out["fingers_end"] = None
        out["held"] = not out["dropped"]
    else:
        out["fingers_end"] = int((F_end >= mass * np.linalg.norm(g) / 2).sum())
        out["held"] = (not out["dropped"]) and out["fingers_end"] >= 2

    task_idx = np.flatnonzero(t_all >= T_TASK - 1e-9)
    if not len(task_idx):
        return {**out, "track_err_rms_deg": None, "track_ref_rms_deg": None, "contact_all_frac": None,
                "contact_all_frac_025": None, "J": None, "J_025": None, "J_nc": 0.0}
    ts = [stream[k] for k in task_idx]
    tt = t_all[task_idx] - T_TASK
    out["n_task"] = len(ts)
    out["metrics_dt"] = round(float(np.median(np.diff(tt))), 6) if len(tt) > 1 else None
    e, ref = track_errors(task, ts, tt, prm, header, T)
    if e is not None and np.isfinite(e).all() and len(e):
        out["track_err_rms_deg"] = round(float(np.sqrt(np.mean(e ** 2))), 2)
        out["track_ref_rms_deg"] = round(float(np.sqrt(np.mean(ref ** 2))), 2)
    else:
        out["track_err_rms_deg"] = out["track_ref_rms_deg"] = None

    rail = int(prm.get("rail", 0)) if task == "roll" else 0
    fs = [FINGERS.index(f) for f in (T.RAIL_ROLLERS[rail] if rail else FINGERS)]
    F = [_pad_forces(s) for s in ts]
    F = np.array([x[fs] for x in F if x is not None])
    for key, thr in (("contact_all_frac", f_min), ("contact_all_frac_025", f_board)):
        out[key] = round(float(np.mean((F > thr).all(1))), 3) if len(F) else None

    def j(share):
        if not out["held"] or out["track_err_rms_deg"] is None or not out["track_ref_rms_deg"]:
            return 0.0
        s = max(0.0, 1.0 - out["track_err_rms_deg"] / out["track_ref_rms_deg"])
        return None if share is None else float(s * share)

    out["J"] = j(out["contact_all_frac"])
    out["J_025"] = j(out["contact_all_frac_025"])
    out["J_nc"] = j(1.0)
    return out


def score_file(path, **kw) -> dict:
    header, stream = load_trial(path)
    return score_stream(stream, header, **kw)
