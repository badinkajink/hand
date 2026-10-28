#!/usr/bin/env python3
"""What each part of v2 ID 1's UR5 calibration suite cost in robot time and what it added to the GP's estimates, as the
basis of a shorter suite for the four-magnet v3 tips (owner, 2026-10-27: "can you intelligently figure out how important
this is? some kind of binary search + information gain optimization?").

  times   From each session's events.jsonl (each action's start and end, the record's hold windows) and robot.bin (the
          flange pose and joints at 124 Hz), every action's time split into travel between actions, the fast and slow
          approach, the holds at the force levels, the turns (motion and holds), the retreat and the rest windows. The
          moving share of travel, fast approach and retreat comes from the robot stream (speed over 0.24 s windows above
          0.3 mm/s or 0.05 deg/s). A travel is lifted by the program's rule (a turn over 20 deg or a move over 60 mm).
          The turn motion is fitted per turn segment as c0 + b n sqrt(L theta): n the path's waypoints (step_deg 0.5),
          theta the angle per waypoint, L the flange's lever (mm per rad, from the hold poses). The stop and restart
          around a force level come from a regression of the slow window on the number of levels. Savings: (a) travel,
          fast approach and retreat at speed slider 1.0, their moving time times 0.30; (b) a travel lift of 30 mm; (c)
          slide holds of 0.2 s, waypoints every 1.4 deg (the path stays within 0.01 mm of the arc at a lever of 128 mm),
          turn_v 0.004 raised to 0.016 (every turn_v from 0.016 to 0.4 took the same time); (d) the holds at 0.5, 2 and
          5 N left out; (e) the time per slide.
  value   Gaussian processes (scripts/ur5_tip_calibration_predict.GP through ur5_tip_calibration_cb1r_checks.gp, the
          board's 15 components to the contact point and 3D force) on the sphere session 20261026-192158-v2id1 in five
          folds. Folds by action follow scripts/ur5_tip_calibration_direction.py for the presses (their ids permuted with
          seed 0) and permute the turn actions' ids with seed 1. Folds by point hold out together every action whose
          median contact point lies within 0.75 mm of another's (union-find, as ur5_tip_calibration_density_curve.groups).
          Errors at the held-out holds: location (mm, along the surface), angle between the estimated and measured force
          (deg), force error (N, norm of the vector difference), and for slides the shear's direction error (deg, holds
          with a measured shear of 0.2 N or more) and vector error (N); medians by region (face, edge, flank) at 1.5 N
          and over and under 1.5 N, and pooled over the regions. Variants: the training kinds (presses; presses and
          slides; all), also at the presses' length and noise; slides halved by direction, amplitude and count, and on
          alternate slides, while the slide test's pooled medians grow by at most 10 % over "all"; force levels left out
          and tested on (presses-only GP); press families left out (fast grid, rate set, hard presses, grid; folds by
          point, presses-only GP); the reduced suite and its option. Each ratio to the reference has the 10th and 90th
          percentiles of a bootstrap over the test actions.
  info    The GP's predictive variance (latent, standardized output units, prior 1) from the kernel, length and noise
          that the all-kinds fits select, at the held-out holds; its Spearman correlation with the location error. Per
          group of actions (kind and region; family): the mean variance at the other held-out holds with and without the
          group in training and the rise per robot hour without it; a greedy order of the kind and region groups by the
          variance lowered per robot second.
  plan    The robot time of v2 ID 1's suite as it ran plus R29's actions that did not run (stages 1 to 4 of
          docs/experiments/20261027-v3_tip_calibration_and_crosstalk/README.md), with each program change alone and
          together (PROGRAM), and of the reduced suite (SUITE: keep, reduce or cut per part) and its option (OPTION).

    ~/.claude/bin/resguard.sh run --mem 6G --cpu 500 -- env NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4 \\
        .venv/bin/python scripts/ur5_tip_calibration_suite_value.py [--parts times,value,info,plan] [--quick]

Writes docs/experiments/20261022-ur5_tip_calibration/data/20261027-suite_value_v2id1.json and caches each fold's
predictions in logs/20261027-suite_value_cache/ (a rerun refits only what changed). From scratch: about 145 fits of up to
3,100 holds (2.3 GB at the peak, 10 to 25 s each), and 10 min in all; logs/20261027-suite_value.log has the run of
2026-10-27.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_tip_calibration_cb1r_checks as K  # noqa: E402

EXP = K.EXP
LAB, DATA = EXP / "lab", EXP / "data"
SESSIONS = dict(sphere="20261026-192158-v2id1", rod="20261027-095303-rodv2id1", tall="20261027-143529-tallv2id1")
OUT = DATA / "20261027-suite_value_v2id1.json"
CACHE = ROOT / "logs/20261027-suite_value_cache"
R29_PLAN = LAB / "20261027-plan_tall_lab_r29_v2id1.json"
SLIDER = 0.30                       # the plans' speed_slider
STEP_DEG = 0.5                      # run_turns' waypoint spacing
TOL = 0.10                          # the halving stops when a pooled median grows by more than this
SHEAR_MIN_N = 0.2                   # holds whose measured shear direction is scored
LOG = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)


def med(x):
    return K.med(x)


def rnd(x, k=3):
    if isinstance(x, dict):
        return {a: rnd(b, k) for a, b in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(b, k) for b in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), k)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ================================================================================================ the time model
def plan_actions(session):
    """Every action of the session's plan files (plan.json and the resumes), by id; the first file that has it."""
    acts = {}
    for f in sorted(glob.glob(str(LAB / session / "plan*.json"))):
        p = json.loads(Path(f).read_text())
        for a in p["actions"]:
            acts.setdefault(a["id"], dict(a, _travel=p.get("travel", {})))
    return acts


def records(session):
    """(start event, end event, plan action) of each action that ended ok, in time order, and the aborted ones."""
    ev = [json.loads(line) for line in (LAB / session / "events.jsonl").read_text().splitlines() if line.strip()]
    acts = plan_actions(session)
    starts, out, bad = {}, [], []
    for e in ev:
        if e["event"] == "start":
            starts[e["id"]] = e
        elif e["event"] == "end" and e["id"] in starts:
            s = starts[e["id"]]
            if s.get("kind") not in ("press", "slide", "twist", "roll", "touch"):
                continue
            if e.get("status") != "ok" or not isinstance(e.get("rec"), dict):
                bad.append(dict(id=e["id"], kind=s.get("kind"), dur=e["t"] - s["t"], why=e.get("why")))
                continue
            out.append((s, e, acts.get(e["id"], {})))
    return out, bad


class Robot:
    """The robot stream (t_host t_ctrl q0..q5 x y z rx ry rz safety_mode), sorted by host time."""

    def __init__(self, session):
        R = np.fromfile(LAB / session / "robot.bin", dtype="<f8")
        R = R.reshape(-1, 15)
        R = R[np.argsort(R[:, 0])]
        self.t, self.p, self.w = R[:, 0].copy(), R[:, 8:11] * 1e3, R[:, 11:14].copy()
        self.q6 = np.degrees(R[:, 7])
        del R

    @staticmethod
    def rotm(W):
        th = np.linalg.norm(W, axis=1)
        k = W / np.maximum(th, 1e-12)[:, None]
        Km = np.zeros((len(W), 3, 3))
        Km[:, 0, 1], Km[:, 0, 2], Km[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
        Km[:, 1, 2], Km[:, 2, 0], Km[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
        s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
        return np.eye(3)[None] + s * Km + (1 - c) * Km @ Km

    def last_motion(self, t0, t1, W=30):
        """The time of the last sample between t0 and t1 at which the flange still moved (as in window)."""
        i0, i1 = np.searchsorted(self.t, t0), np.searchsorted(self.t, t1)
        if i1 - i0 <= W + 1:
            return None
        t, p = self.t[i0:i1], self.p[i0:i1]
        Rm = self.rotm(self.w[i0:i1])
        dt = t[W:] - t[:-W]
        v = np.linalg.norm(p[W:] - p[:-W], axis=1) / dt
        tr = np.einsum("kij,kij->k", Rm[:-W], Rm[W:])
        om = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))) / dt
        mv = np.nonzero((v > 0.3) | (om > 0.05))[0]
        return float(t[mv[-1] + W]) if len(mv) else None

    def window(self, t0, t1, W=30):
        """(moving s, legs) between t0 and t1: legs are runs of motion of 0.2 s or more, as (duration s, length mm,
        rise mm, mean speed mm/s)."""
        i0, i1 = np.searchsorted(self.t, t0), np.searchsorted(self.t, t1)
        if i1 - i0 <= W + 1:
            return 0.0, []
        t, p = self.t[i0:i1], self.p[i0:i1]
        Rm = self.rotm(self.w[i0:i1])
        dt = t[W:] - t[:-W]
        v = np.linalg.norm(p[W:] - p[:-W], axis=1) / dt
        tr = np.einsum("kij,kij->k", Rm[:-W], Rm[W:])
        om = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))) / dt
        mv = (v > 0.3) | (om > 0.05)
        c = np.arange(len(mv)) + W // 2                         # the window's centre sample
        dts = np.diff(t)
        moving = float(np.sum(dts[c[mv] - 1])) if mv.any() else 0.0
        legs, k = [], 0
        while k < len(mv):
            if not mv[k]:
                k += 1
                continue
            j = k
            while j < len(mv) and mv[j]:
                j += 1
            a, b = c[k], min(c[j - 1], len(t) - 1)
            if t[b] - t[a] >= 0.2:
                d = p[b] - p[a]
                legs.append((float(t[b] - t[a]), float(np.linalg.norm(d)), float(d[2]), float(np.linalg.norm(d) / (t[b] - t[a]))))
            k = j
        return moving, legs


def action_phases(s, e, a, robot=None):
    """One action's time split into phases (s)."""
    r = e["rec"]
    t0 = s["t"]
    d = dict(id=e["id"], kind=s["kind"], stage=s.get("stage"), region=(a.get("target") or {}).get("region"),
             dur=e["t"] - t0, joint6=a.get("joint6_deg"))
    d["travel"] = r["t_rest0"][0] - t0
    d["rest"] = (r["t_rest0"][1] - r["t_rest0"][0]) + (r["t_rest1"][1] - r["t_rest1"][0])
    fa = r.get("fast") or {}
    d["fast"] = (fa["t"][1] - fa["t"][0]) if fa.get("t") and fa["t"][1] else 0.0
    lv = r.get("levels") or []
    holds = [x["t"][1] - x["t"][0] for x in lv]
    inner = [h for x, h in zip(lv, holds) if x["t"][1] <= r["t_stop"] + 1e-6]
    d["n_levels"], d["level_hold"] = len(lv), float(sum(holds))
    d["levels"] = [float(x["N"]) for x in lv]
    d["slow"] = r["t_stop"] - r["t_move0"] - sum(inner)          # the slow approach with its stops
    last = lv[-1]["t"][1] if lv and lv[-1]["t"][1] > r["t_stop"] else r["t_stop"]
    d["turn_move"] = d["turn_hold"] = 0.0
    d["n_turn_holds"] = 0
    for mv in r.get("moves", []):
        for tu in mv["turns"]:
            d["turn_move"] += tu["t"][0] - last
            d["turn_hold"] += tu["t"][1] - tu["t"][0]
            d["n_turn_holds"] += 1
            last = tu["t"][1]
    d["retreat"] = r["t_rest1"][0] - last
    d["other"] = d["dur"] - sum(d[k] for k in ("travel", "rest", "fast", "slow", "level_hold", "turn_move", "turn_hold",
                                               "retreat"))
    # the slow approach's length: from the fast approach's end to the stop, along the approach
    if r.get("T_rest") is not None and r.get("T_stop") is not None:
        dist = float(np.linalg.norm(np.array(r["T_stop"])[:3, 3] - np.array(r["T_rest"])[:3, 3])) - float(fa.get("mm", 0.0))
        d["slow_mm"] = dist
    d["v_real_mm_s"] = float(a.get("v", 0.0005 if s["kind"] == "press" else 0.0004)) * 1e3 * SLIDER
    if s["kind"] == "touch":
        d["v_real_mm_s"] = float(a.get("v", 0.00015)) * 1e3 * SLIDER
    if robot is not None:
        d["travel_moving"], legs = robot.window(t0, r["t_rest0"][0])
        d["legs"] = legs
        d["fast_moving"] = robot.window(fa["t"][0], fa["t"][1])[0] if d["fast"] else 0.0
        d["retreat_moving"] = robot.window(last, r["t_rest1"][0])[0]
        # the travel's rule (ur5_tip_calibration.travel): lifted when the tool turns more than turn_deg or moves more
        # than far_mm from the pose where the travel starts to the action's rest pose
        cfg = dict(turn_deg=20.0, far_mm=60.0, lift_mm=60.0)
        cfg.update(a.get("_travel") or {})
        i = min(np.searchsorted(robot.t, t0), len(robot.t) - 1)
        j = min(np.searchsorted(robot.t, r["t_rest0"][0]), len(robot.t) - 1)      # the rest pose, where it ends
        R0, R1 = robot.rotm(robot.w[[i, j]])
        ang = math.degrees(math.acos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))
        d["travel_turn_deg"], d["travel_mm"] = ang, float(np.linalg.norm(robot.p[j] - robot.p[i]))
        d["lifted"] = bool(ang > cfg["turn_deg"] or d["travel_mm"] > cfg["far_mm"])
        k0, k1 = np.searchsorted(robot.t, r["t_rest0"][1]), np.searchsorted(robot.t, r["t_rest1"][0])
        d["joint6_max"] = float(robot.q6[k0:k1].max()) if k1 > k0 else None
    return d


def turn_segments(s, e, a, robot=None):
    """Each turn segment of a slide, twist or roll: (kind, turn_v, n waypoints, angle per waypoint deg, lever mm/rad,
    duration s, and with the robot stream the settle: from the flange's last motion to the hold's start, s)."""
    r = e["rec"]
    tv = float(a.get("turn_v", (a.get("_travel") or {}).get("turn_v", 0.004)))
    lv = r.get("levels") or []
    last = lv[-1]["t"][1] if lv else r["t_stop"]
    T_prev = np.array(r["T_stop"])
    out = []
    for mv in r.get("moves", []):
        cur = 0.0
        for tu in mv["turns"]:
            dd = abs(tu["deg"] - cur)
            n = max(1, int(math.ceil(dd / STEP_DEG - 1e-9)))
            T = np.array(tu["T_BE"])
            lever = float(np.linalg.norm(T[:3, 3] - T_prev[:3, 3]) / max(math.radians(dd), 1e-9)) if dd > 0.05 else np.nan
            row = dict(id=e["id"], kind=s["kind"], turn_v=tv, n=n, theta=dd / n, deg=dd, lever=lever, dur=tu["t"][0] - last,
                       hold=tu["t"][1] - tu["t"][0])
            if robot is not None:
                lm = robot.last_motion(last, tu["t"][0])
                row["settle"] = (tu["t"][0] - lm) if lm is not None else None
            out.append(row)
            last, cur, T_prev = tu["t"][1], tu["deg"], T
    return out


def fit_turn_model(segs):
    """T = c0 + b n sqrt(L theta) (s), least squares over the segments with turn_v of 0.016 or more (the faster turn
    speeds all took the same time), and the same for turn_v 0.004."""
    res = {}
    for name, sel in (("fast", lambda x: x["turn_v"] >= 0.0159), ("tv004", lambda x: x["turn_v"] < 0.0159)):
        S = [x for x in segs if sel(x) and np.isfinite(x["lever"])]
        if len(S) < 5:
            continue
        X = np.array([[1.0, x["n"] * math.sqrt(x["lever"] * math.radians(x["theta"]))] for x in S])
        y = np.array([x["dur"] for x in S])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        res[name] = dict(c0_s=float(coef[0]), b=float(coef[1]), n_segments=len(S), rms_s=float(np.sqrt(np.mean((y - pred) ** 2))),
                         median_rel_err=float(np.median(np.abs(y - pred) / y)))
    return res


def turn_time(coef, degs_list, lever, step_deg=STEP_DEG):
    """Motion time (s) of a turn action's moves under the fitted model, c0 + b n sqrt(L theta) per segment, and its
    number of holds; each move turns through its angles and back to 0."""
    tm, nh = 0.0, 0
    for degs in degs_list:
        cur = 0.0
        for d in list(degs) + [0.0]:
            dd = abs(d - cur)
            n = max(1, int(math.ceil(dd / step_deg - 1e-9)))
            tm += coef["c0_s"] + coef["b"] * n * math.sqrt(lever * math.radians(dd / n))
            nh += 1
            cur = d
    return tm, nh


def action_time(p, T, cfg):
    if cfg.get("amp_alt") and p["kind"] == "slide":
        c = {k: v for k, v in cfg.items() if k != "amp_alt"}
        return 0.5 * (_action_time(p, T, dict(c, amp1=True)) + _action_time(p, T, dict(c, amp2only=True)))
    return _action_time(p, T, cfg)


def _action_time(p, T, cfg):
    """An action's robot time (s) under the changes in cfg, from its measured phases:
      slider1      travel, fast approach and (unless retreat is False) retreat at speed slider 1.0: their moving time
                   times 0.30 (the plans set v and turn_v 0.30 times lower, so presses and turns keep their speeds)
      lift30       the travel's lift at 30 mm instead of 60
      drop_levels  force levels left out of each press (its top level stays): the hold and the stop of each
      slide_hold   the hold at each slide step (s)
      step_deg     the turn path's waypoint spacing (deg)
      fast_turns   turn_v 0.004 raised to 0.016 or more (all of those took the same time)
      dirs, amp1   a slide's directions kept, and its first amplitude alone"""
    t = p["dur"]
    if cfg.get("slider1"):
        t -= (1 - SLIDER) * (p.get("travel_moving", 0.0) + p.get("fast_moving", 0.0))
        if cfg.get("retreat", True):
            t -= (1 - SLIDER) * p.get("retreat_moving", 0.0)
    if cfg.get("lift30"):
        t -= lift_saving(p) * (SLIDER if cfg.get("slider1") else 1.0)
    if p["kind"] == "press" and cfg.get("drop_levels"):
        n = min(int(cfg["drop_levels"]), max(0, p["n_levels"] - 1))
        t -= n * (T["level_overhead"]["per_level_s"] + p["level_hold"] / max(p["n_levels"], 1))
    if p.get("moves") and p.get("lever"):
        moves = p["moves"]
        if p["kind"] == "slide":
            if cfg.get("dirs") is not None:
                moves = [m for m in moves if any(abs(m["dir"] - x) < 1e-6 for x in cfg["dirs"])] or moves
            if cfg.get("amp1"):
                moves = [dict(m, degs=m["degs"][:1]) for m in moves]
            elif cfg.get("amp2only"):
                moves = [dict(m, degs=m["degs"][1:2] or m["degs"][:1]) for m in moves]
        model = T["turn_model"]
        slow = p["turn_v"] is not None and p["turn_v"] < 0.0159
        c_now = model["tv004"] if slow else model["fast"]
        c_new = model["fast"] if (cfg.get("fast_turns") or not slow) else model["tv004"]
        m_now, nh_now = turn_time(c_now, [m["degs"] for m in p["moves"]], p["lever"])
        m_new, nh_new = turn_time(c_new, [m["degs"] for m in moves], p["lever"], cfg.get("step_deg", STEP_DEG))
        hold = cfg["slide_hold"] if (p["kind"] == "slide" and cfg.get("slide_hold") is not None) else p["hold_s"]
        t += p["turn_move"] * (m_new / m_now - 1.0)
        t += nh_new * hold - p["turn_hold"]
    return t


def level_overhead(P):
    """Per force level: the stop and restart around its hold (s), from slow window - inner holds - slow mm / speed =
    c0 + c n_levels over presses at the grid's 0.15 mm/s."""
    rows = [p for p in P if p["kind"] == "press" and abs(p["v_real_mm_s"] - 0.15) < 1e-6 and p.get("slow_mm") is not None
            and p["n_levels"] >= 1]
    y = np.array([p["slow"] - p["slow_mm"] / p["v_real_mm_s"] for p in rows])
    X = np.c_[np.ones(len(rows)), [p["n_levels"] for p in rows]]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(c0_s=float(coef[0]), per_level_s=float(coef[1]), n=len(rows),
                rms_s=float(np.sqrt(np.mean((y - X @ coef) ** 2))))


def lift_saving(p, new_mm=30.0, old_mm=60.0, k=2.0):
    """Seconds saved at slider 0.30 by a lift of new_mm on a lifted travel (up to the lift, across, down to 10 mm above
    the indenter, the last approach): its up and down legs shorten by up to old - new mm, never below the 10 mm
    minimum; with the legs from the robot stream when it shows four or more, else at the legs' planned speeds (up
    0.03 k, down 0.02 k m/s, times the slider)."""
    if not p.get("lifted"):
        return 0.0
    cut = old_mm - new_mm
    legs = p.get("legs") or []
    if len(legs) >= 4:
        s = 0.0
        for leg in (legs[0], legs[-2]):
            if leg[1] > 10.5 and leg[3] > 1.0:
                s += min(cut, leg[1] - 10.0) / leg[3]
        return s
    return cut / (30.0 * k * SLIDER) + cut / (20.0 * k * SLIDER)


def session_times(name, session, use_robot=True):
    recs, bad = records(session)
    robot = Robot(session) if use_robot else None
    P = [action_phases(s, e, a, robot) for s, e, a in recs]
    segs = [x for s, e, a in recs if s["kind"] in ("slide", "twist", "roll") for x in turn_segments(s, e, a, robot)]
    lev = collections.defaultdict(list)
    for x in segs:
        if np.isfinite(x["lever"]):
            lev[x["id"]].append(x["lever"])
    for p, (s, e, a) in zip(P, recs):
        p["moves"] = [dict(dir=float(m.get("dir_deg", 0.0)), degs=list(m["degs"])) for m in a.get("moves", [])]
        p["lever"] = float(np.median(lev[p["id"]])) if lev.get(p["id"]) else None
        p["turn_v"] = float(a.get("turn_v", (a.get("_travel") or {}).get("turn_v", 0.004))) if p["moves"] else None
        p["hold_s"] = float(a.get("hold_s", 0.4)) if p["moves"] else None
        p["family"] = coarse_label(p["kind"], p["id"])
    del robot
    return P, segs, bad


def summarize_times(name, P, bad):
    keys = ("dur", "travel", "rest", "fast", "slow", "level_hold", "turn_move", "turn_hold", "retreat", "other")
    out = dict(session=SESSIONS[name], aborted=dict(n=len(bad), min=sum(b["dur"] for b in bad) / 60))
    by = collections.defaultdict(list)
    for p in P:
        by[p["kind"]].append(p)
    out["by_kind"] = {}
    for k, v in sorted(by.items()):
        row = dict(n=len(v), **{f"{c}_min": sum(x[c] for x in v) / 60 for c in keys},
                   median_s=float(np.median([x["dur"] for x in v])))
        for c in ("travel_moving", "fast_moving", "retreat_moving"):
            if c in v[0]:
                row[f"{c}_min"] = sum(x[c] for x in v) / 60
        row["lifted_travels"] = int(sum(1 for x in v if x.get("lifted")))
        out["by_kind"][k] = row
    out["total_h"] = sum(p["dur"] for p in P) / 3600
    return out


def savings(P):
    """Robot minutes saved by each change, over the actions P of one session."""
    s = {}
    mov = sum(p.get("travel_moving", 0) + p.get("fast_moving", 0) for p in P)
    ret = sum(p.get("retreat_moving", 0) for p in P)
    s["a_slider_travel_fast_min"] = (1 - SLIDER) * mov / 60
    s["a_slider_retreat_min"] = (1 - SLIDER) * ret / 60
    lift = sum(lift_saving(p) for p in P)
    s["b_lift30_at_slider_0p30_min"] = lift / 60
    s["b_lift30_at_slider_1_min"] = SLIDER * lift / 60
    s["n_lifted_travels"] = int(sum(1 for p in P if p.get("lifted")))
    sl = [p for p in P if p["kind"] == "slide"]
    s["c_slide_holds_0p2_min"] = sum(0.2 / 0.4 * p["turn_hold"] for p in sl) / 60
    return s


def level_saving(P, T, drop=(0.5, 2.0, 5.0)):
    """Minutes saved by leaving out the holds at the levels drop (N) where they are not a press's top level: each
    costs its hold and the stop and restart around it."""
    n, sec = 0, 0.0
    for p in P:
        if p["kind"] != "press" or not p.get("levels"):
            continue
        for lv in p["levels"][:-1]:
            if any(abs(lv - x) < 1e-6 for x in drop):
                n += 1
                sec += T["level_overhead"]["per_level_s"] + p["level_hold"] / max(p["n_levels"], 1)
    return dict(n_holds=n, min=sec / 60)


def times_part(quick=False):
    res, segs_all, P_all = {}, [], {}
    for name, ses in SESSIONS.items():
        t0 = time.time()
        P, segs, bad = session_times(name, ses)
        P_all[name] = P
        segs_all += segs
        res[name] = summarize_times(name, P, bad)
        log(f"times {name}: {len(P)} actions, {res[name]['total_h']:.2f} h in actions ({time.time() - t0:.0f} s)")
    model = fit_turn_model(segs_all)
    res["turn_model"] = model
    # the turn speeds measured: mean deg/s over the motion, by kind and turn_v
    sp = collections.defaultdict(list)
    for x in segs_all:
        sp[(x["kind"], x["turn_v"])].append((x["deg"], x["dur"]))
    res["turn_speed_deg_s"] = {f"{k}/turn_v {tv:g}": dict(n=len(v), mean_deg_s=sum(a for a, _ in v) / sum(b for _, b in v),
                                                          planned_deg_s=math.degrees(tv) * SLIDER)
                               for (k, tv), v in sorted(sp.items())}
    st = [x["settle"] for x in segs_all if x.get("settle") is not None]
    res["turn_settle_s"] = dict(n=len(st), median=float(np.median(st)), mean=float(np.mean(st)))
    lv = level_overhead([p for P in P_all.values() for p in P])
    res["level_overhead"] = lv
    res["savings"] = {name: savings(P) for name, P in P_all.items()}
    for name, P in P_all.items():
        res["savings"][name]["d_levels_0.5_2_5_dropped"] = level_saving(P, res)
        sl = [p for p in P if p["kind"] == "slide"]
        res["savings"][name]["e_per_slide_s"] = dict(as_run=float(np.mean([p["dur"] for p in sl])) if sl else None)
    # per action of each kind: the phases' medians (s)
    for name, P in P_all.items():
        med_s = {}
        for k in sorted(set(p["kind"] for p in P)):
            v = [p for p in P if p["kind"] == k]
            med_s[k] = {c: float(np.mean([x[c] for x in v])) for c in ("dur", "travel", "fast", "slow", "level_hold",
                                                                       "turn_move", "turn_hold", "retreat", "rest")}
        res[name]["mean_s_per_action"] = med_s
    return res, P_all, segs_all


# ================================================================================================ the holds and folds
def sphere_holds():
    """The sphere session's holds with each turn hold's direction (deg, from the events' windows), the amplitude index
    (0 the press hold, 1, 2, 3 the first to third amplitude, 9 the return to 0) and the action's family."""
    H = K.load(DATA / SESSIONS["sphere"])
    ok = np.isfinite(H["d_board"]).all(1) & np.isfinite(H["p"]).all(1) & np.isfinite(H["f"]).all(1)
    H = K.sub(H, ok)
    recs, _ = records(SESSIONS["sphere"])
    win = []
    for s, e, a in recs:
        for mv in e["rec"].get("moves", []):
            for j, tu in enumerate(mv["turns"]):
                win.append((tu["t"][0], tu["t"][1], e["id"], float(mv.get("dir_deg", 0.0)), j, tu["deg"]))
    win.sort()
    t0 = np.array([w[0] for w in win])
    n = len(H["id"])
    H["dir_deg"] = np.full(n, np.nan)
    H["amp"] = np.zeros(n, int)
    turn = H["kind"] != "press"
    k = np.searchsorted(t0, H["t"]) - 1
    for i in np.nonzero(turn)[0]:
        if k[i] >= 0 and H["t"][i] <= win[k[i]][1] and win[k[i]][2] == H["id"][i]:
            w = win[k[i]]
            H["dir_deg"][i] = w[3]
            H["amp"][i] = 9 if abs(w[5]) < 1e-9 else w[4] + 1
    fam = []
    for s in H["id"].astype(str):
        x = re.sub(r"-t[+-]\d+", "", s)
        x = re.sub(r"\d+(\.\d+)?", "#", x)
        fam.append(x)
    H["family"] = np.array(fam)
    dur = {e["id"]: e["t"] - s["t"] for s, e, a in recs}
    H["act_s"] = np.array([dur.get(str(i), np.nan) for i in H["id"]])
    return H


def action_folds(H, k=5):
    """Five folds by action: the presses as in ur5_tip_calibration_direction.cv (their ids permuted with seed 0), the
    turn actions' ids permuted with seed 1."""
    a = H["id"].astype(str)
    f = np.full(len(a), -1)
    pr = H["kind"] == "press"
    u = np.random.default_rng(0).permutation(np.unique(a[pr]))
    m = {x: i % k for i, x in enumerate(u)}
    f[pr] = [m[x] for x in a[pr]]
    u = np.random.default_rng(1).permutation(np.unique(a[~pr]))
    m = {x: i % k for i, x in enumerate(u)}
    f[~pr] = [m[x] for x in a[~pr]]
    return f


def point_groups(H, tol=0.75):
    """Actions whose median contact points lie within tol mm of each other, joined (union-find)."""
    ids = H["id"].astype(str)
    acts = np.unique(ids)
    c = np.array([np.median(H["p"][ids == x], axis=0) for x in acts])
    parent = list(range(len(acts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    d = np.linalg.norm(c[:, None] - c[None], axis=2)
    for i, j in zip(*np.nonzero(np.triu(d < tol, 1))):
        parent[find(i)] = find(j)
    g = {x: find(i) for i, x in enumerate(acts)}
    return np.array([g[x] for x in ids])


def point_folds(H, k=5):
    g = point_groups(H)
    u = np.random.default_rng(0).permutation(np.unique(g))
    m = {x: i % k for i, x in enumerate(u)}
    return np.array([m[x] for x in g]), g


# ================================================================================================ GP fits with a cache
FITS = dict(n=0, cached=0)


class GPAt(K.PR.GP):
    """ur5_tip_calibration_predict.GP with its length and noise given instead of chosen over the grid."""

    def __init__(self, ell, nz):
        super().__init__(center=True)
        self.ell_at, self.nz_at = float(ell), float(nz)

    def fit(self, X, Y, floor=None):
        self.mx, self.sx = X.mean(0), K.PR.scale(X, floor)
        self.my, self.sy = Y.mean(0), Y.std(0) + 1e-9
        Z, T = (X - self.mx) / self.sx, (Y - self.my) / self.sy
        sq = np.sum(Z ** 2, 1)
        D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
        L = np.linalg.cholesky(np.exp(-0.5 * D2 / self.ell_at ** 2) + self.nz_at ** 2 * np.eye(len(Z)))
        self.a = np.linalg.solve(L.T, np.linalg.solve(L, T))
        self.ell, self.nz, self.Z = self.ell_at, self.nz_at, Z
        return self


def fold_predict(H, train, folds, tag, quick=False, at=None):
    """Predictions [p, f] at every hold from the fold model that held it out, training on train & ~fold; the GP of
    ur5_tip_calibration_cb1r_checks.gp, or with at = (length, noise) the same GP at that length and noise."""
    CACHE.mkdir(parents=True, exist_ok=True)
    Y = np.full((len(H["id"]), 6), np.nan)
    sel = []
    for q in sorted(set(folds.tolist())):
        if q < 0:
            continue
        te = folds == q
        tr = train & ~te
        if tr.sum() < 10:
            continue
        h = hashlib.sha1(np.packbits(tr).tobytes() + np.packbits(te).tobytes() + str(at).encode()).hexdigest()[:16]
        f = CACHE / f"{tag}-f{q}-{h}.npz"
        if f.exists():
            z = np.load(f)
            Y[te] = z["Y"]
            sel.append((float(z["ell"]), float(z["nz"])))
            FITS["cached"] += 1
            continue
        t0 = time.time()
        Ht = K.sub(H, tr)
        if at is None:
            g = K.gp(Ht)
        else:
            g = GPAt(*at).fit(Ht["d_board"], np.c_[Ht["p"], Ht["f"]], K.PR.floors(0)[:15])
        Yq = g.predict(H["d_board"][te])
        Y[te] = Yq
        np.savez(f, Y=Yq, ell=g.ell, nz=g.nz, n_train=tr.sum())
        sel.append((float(g.ell), float(g.nz)))
        FITS["n"] += 1
        log(f"    fit {tag} fold {q}: {int(tr.sum())} holds, ell {g.ell:.2f}, noise {g.nz:.2f}, {time.time() - t0:.0f} s"
            f" (fits so far {FITS['n']}, cached {FITS['cached']})")
        if quick:
            break
    return Y, sel


# ================================================================================================ scores
def errors(H, Y):
    P, F = Y[:, :3], Y[:, 3:]
    e = dict(loc=K.loc_err(H, P), angle=K.angle(F, H["f"]), force=np.linalg.norm(F - H["f"], axis=1))
    ft, fe = K.shear(H["f"], H["n"]), K.shear(F, H["n"])
    sa = K.angle(fe, ft)
    sa[np.linalg.norm(ft, axis=1) < SHEAR_MIN_N] = np.nan
    e["shear_angle"] = sa
    e["shear_err"] = np.linalg.norm(fe - ft, axis=1)
    for k in e:
        e[k] = np.where(np.isfinite(Y).all(1), e[k], np.nan)
    return e


METRICS = ("loc", "angle", "force", "shear_angle", "shear_err")


def table(H, E, m, metrics=METRICS):
    """Medians by region and band, and pooled, over holds m."""
    hi = H["f_n"] >= 1.5
    out = {}
    for reg in ("face", "edge", "flank", "all"):
        r = m if reg == "all" else m & (H["region"].astype(str) == reg)
        for band, b in (("over_1p5N", hi), ("under_1p5N", ~hi)):
            k = r & b
            out[f"{reg}/{band}"] = dict(n=int(k.sum()), **{x: med(E[x][k]) for x in metrics})
    return out


def ratio_band(H, E1, E0, m, metric, band, n_boot=200, seed=0):
    """Ratio of the pooled median of E1 to E0 over holds m in band, with the 10th and 90th percentiles of a bootstrap
    over the test actions."""
    hi = H["f_n"] >= 1.5
    k = m & (hi if band == "over_1p5N" else ~hi) & np.isfinite(E1[metric]) & np.isfinite(E0[metric])
    if k.sum() < 10:
        return None
    a = H["id"].astype(str)[k]
    x1, x0 = E1[metric][k], E0[metric][k]
    r = float(np.median(x1) / np.median(x0))
    u, inv = np.unique(a, return_inverse=True)
    rng = np.random.default_rng(seed)
    idx = [np.nonzero(inv == j)[0] for j in range(len(u))]
    rs = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[j] for j in rng.integers(0, len(u), len(u))])
        rs.append(np.median(x1[pick]) / np.median(x0[pick]))
    return [r, float(np.percentile(rs, 10)), float(np.percentile(rs, 90))]


def compare(H, E1, E0, tests):
    """Ratios variant / reference of the pooled medians, per test set, metric and band."""
    out, worst = {}, (0.0, None)
    for tname, m in tests.items():
        mets = ("loc", "angle", "force", "shear_angle") if tname == "slide" else ("loc", "angle", "force")
        for met in mets:
            for band in ("over_1p5N", "under_1p5N"):
                rb = ratio_band(H, E1, E0, m, met, band)
                if rb is None:
                    continue
                out[f"{tname}/{met}/{band}"] = rb
                if rb[0] - 1 > worst[0]:
                    worst = (rb[0] - 1, f"{tname}/{met}/{band}")
    return out, dict(growth=worst[0], at=worst[1])


# ================================================================================================ the value part
def value_part(H, quick=False):
    res = dict(n_holds=int(len(H["id"])), n_actions={k: int(len(np.unique(H["id"][H["kind"] == k])))
                                                       for k in ("press", "slide", "twist", "roll")})
    fa = action_folds(H)
    fp, groups = point_folds(H)
    res["n_point_groups"] = int(len(np.unique(groups)))
    kind = H["kind"].astype(str)
    P, S, T, R = (kind == "press"), (kind == "slide"), (kind == "twist"), (kind == "roll")
    ALL = np.ones(len(kind), bool)
    tests = dict(press=P, slide=S, twist=T, roll=R)
    cache_E = {}
    V = res["variants"] = {}

    def run(name, train, folds="action", at=None):
        key = (name, folds)
        if key not in cache_E:
            Y, sel = fold_predict(H, train, fa if folds == "action" else fp, f"{folds}-{name}", quick, at)
            cache_E[key] = (errors(H, Y), sel)
        return cache_E[key]

    def entry(name, train, folds, note, test_sets, ref=None, crit=None, at=None, extra=None):
        """A variant's tables on its test sets, and its ratios to the reference variant (over the sets crit)."""
        E, sel = run(name, train, folds, at)
        row = dict(note=note, folds=folds, n_train_holds=int(train.sum()), ell_noise=sorted(set(sel)),
                   tests={t: table(H, E, m) for t, m in test_sets.items()})
        if ref is not None:
            E0, _ = cache_E[(ref, folds)]
            row["reference"] = f"{ref}/{folds}"
            row["vs_ref"], row["growth"] = compare(H, E, E0, crit if crit is not None else test_sets)
        if extra:
            row.update(extra)
        V[f"{name}/{folds}"] = row
        if ref is not None:
            log(f"  {name}/{folds}: growth {row['growth']['growth']:+.3f} at {row['growth']['at']} (vs {ref})")
        return row

    # (i) the training kinds, folds by action and by point; every test set
    for folds in ("action", "point"):
        for name, tr in (("presses", P), ("presses+slides", P | S), ("all", ALL)):
            entry(name, tr, folds, "training kinds", tests)
        for name in ("presses", "presses+slides"):
            V[f"{name}/{folds}"]["vs_all"] = compare(H, cache_E[(name, folds)][0], cache_E[("all", folds)][0], tests)[0]
        V[f"all/{folds}"]["vs_presses"] = compare(H, cache_E[("all", folds)][0], cache_E[("presses", folds)][0],
                                                  dict(press=P))[0]
    if quick:
        return res, cache_E, fa, fp
    # the same training sets at the presses' own length and noise (the grid's choice moves with the mix of kinds)
    ell_p = np.median([x[0] for x in cache_E[("presses", "action")][1]])
    nz_p = np.median([x[1] for x in cache_E[("presses", "action")][1]])
    for name, tr in (("presses+slides@press_hyper", P | S), ("all@press_hyper", ALL)):
        entry(name, tr, "action", f"as {name.split('@')[0]}, at the presses' length {ell_p:.2f} and noise {nz_p:.2f}",
              tests, ref="presses", crit=dict(press=P), at=(ell_p, nz_p))

    # (ii) slides subsampled, the other kinds as in "all"; halved while the slide test's pooled medians grow by at most
    # TOL over "all" (the press test is reported beside it)
    def slide_variant(name, keep, note, folds="action"):
        tr = ~S | (S & keep)
        row = entry(name, tr, folds, note, dict(press=P, slide=S), ref="all", crit=dict(slide=S),
                    extra=dict(n_slide_holds=int((S & keep).sum()), n_slide_actions=int(len(np.unique(H["id"][S & keep])))))
        row["vs_ref_press"] = compare(H, cache_E[(name, folds)][0], cache_E[("all", folds)][0], dict(press=P))[0]
        return row["growth"]["growth"]

    d, amp = H["dir_deg"], H["amp"]
    first = amp == 0
    search = res["halving"] = {}
    out = []
    for name, keep, note in (("dirs4", np.isin(d, [0, 90, 180, 270]) | first, "4 of 8 directions (0, 90, 180, 270 deg)"),
                             ("dirs2", np.isin(d, [0, 180]) | first, "2 of 8 directions (0 and 180 deg)"),
                             ("dirs2x", np.isin(d, [0, 90]) | first, "2 of 8 directions (0 and 90 deg)")):
        g = slide_variant(name, keep, note)
        out.append([name, g])
        if name == "dirs4" and g > TOL:
            break
    search["direction"] = out
    out = []
    for name, keep, note in (("amp1", np.isin(amp, [0, 1, 9]), "the first amplitude (1.146 deg, 0.1 mm) and the returns"),
                             ("amp2only", np.isin(amp, [0, 2, 9]), "the second amplitude (2.292 deg, 0.2 mm) and the returns")):
        out.append([name, slide_variant(name, keep, note)])
    search["amplitude"] = out
    sid = np.random.default_rng(0).permutation(np.unique(H["id"][S].astype(str)))
    out = []
    for n_keep in (50, 25, 12):
        g = slide_variant(f"count{n_keep}", np.isin(H["id"].astype(str), sid[:n_keep]), f"{n_keep} of the 100 slides (seed 0)")
        out.append([f"count{n_keep}", g])
        if g > TOL:
            break
    search["count"] = out
    # alternating: each slide keeps half of its moves, and the slides together keep all 8 directions and both amplitudes
    alt = alternating(H)
    out = []
    for name, note in (("dirs4alt", "4 directions per slide, 0/90/180/270 and 45/135/225/315 on alternate slides"),
                       ("ampalt", "one amplitude per slide, the first and the second on alternate slides"),
                       ("dirs4alt_ampalt", "4 directions and one amplitude per slide, alternating over four slides")):
        out.append([name, slide_variant(name, alt[name], note)])
    search["alternating"] = out

    # (iii) force levels left out of the press holds and tested on (presses-only GP, the presses' reference)
    lv = np.round(H["level_N"], 3)
    for name, drop in (("levels_keep_0.25_1_3.5", [0.5, 2.0, 5.0]), ("levels_keep_0.5_2_5", [0.25, 1.0, 3.5])):
        m = P & np.isin(lv, drop)
        entry(name, P & ~m, "action", f"press holds at {drop} N left out of training and tested on", dict(dropped=m),
              ref="presses", crit=dict(press=m))
    # press families left out of the presses-only GP, folds by point (their points stay covered by the others)
    fam = H["family"].astype(str)
    for name, out_m, note in (
            ("no_fastgrid", P & np.char.startswith(fam, "fast"), "the 54 fast-grid presses (0.6 and 4 mm/s) left out"),
            ("no_rate", P & np.char.startswith(fam, "rate"), "the 18 rate-set presses left out"),
            ("no_hard", P & np.char.startswith(fam, "hard"), "the 28 hard presses (to 8 N) left out"),
            ("no_fastgrid_rate", P & (np.char.startswith(fam, "fast") | np.char.startswith(fam, "rate")),
             "the fast-grid and rate presses left out"),
            ("no_grid", P & np.char.startswith(fam, "grid"),
             "the 52 grid presses (0.15 mm/s) left out; their targets' fast-grid presses (0.6 mm/s) stay")):
        slow_fast = dict(press_grid=P & np.char.startswith(fam, "grid"), press_fastgrid=P & np.char.startswith(fam, "fast-grid"))
        row = entry(name, P & ~out_m, "point", note, dict(press=P, press_over_3N=P & (H["f_n"] >= 3.0), **slow_fast),
                    ref="presses", crit=dict(press=P))
        for t, m in slow_fast.items():
            row[f"vs_ref_{t}"] = compare(H, cache_E[(name, "point")][0], cache_E[("presses", "point")][0], dict(press=m))[0]
        row["vs_ref_over_3N"] = compare(H, cache_E[(name, "point")][0], cache_E[("presses", "point")][0],
                                        dict(press=P & (H["f_n"] >= 3.0)))[0]
    res["fits"] = dict(FITS)
    return res, cache_E, fa, fp


def combo_part(H, cache_E, fa, fp, keep, name, note):
    """A combined reduction of the slides, in both fold types, against "all"."""
    kind = H["kind"].astype(str)
    P, S = kind == "press", kind == "slide"
    out = {}
    for folds, F in (("action", fa), ("point", fp)):
        tr = ~S | (S & keep)
        Y, sel = fold_predict(H, tr, F, f"{folds}-{name}")
        E1 = errors(H, Y)
        cache_E[(name, folds)] = (E1, sel)
        E0 = cache_E[("all", folds)][0]
        row = dict(note=note, folds=folds, n_train_holds=int(tr.sum()), ell_noise=sorted(set(sel)),
                   n_slide_actions=int(len(np.unique(H["id"][S & keep]))), n_slide_holds=int((S & keep).sum()),
                   tests=dict(press=table(H, E1, P), slide=table(H, E1, S)), reference=f"all/{folds}")
        row["vs_ref"], row["growth"] = compare(H, E1, E0, dict(slide=S))
        row["vs_ref_press"] = compare(H, E1, E0, dict(press=P))[0]
        out[f"{name}/{folds}"] = row
        log(f"  {name}/{folds}: growth {row['growth']['growth']:+.3f} at {row['growth']['at']} (vs all)")
    return out


# ================================================================================================ information gain
class Kern:
    """The squared-exponential kernel of the GP on standardized boards, with the length and noise selected by the
    all-kinds fits (their median), over every hold of the session."""

    def __init__(self, H, ell, nz):
        X = H["d_board"]
        self.mx = X.mean(0)
        self.sx = K.PR.scale(X, K.PR.floors(0)[:15])
        Z = (X - self.mx) / self.sx
        sq = np.sum(Z ** 2, 1)
        D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
        self.K = np.exp(-0.5 * D2 / ell ** 2)
        del D2
        self.ell, self.nz = ell, nz

    def var(self, tr, ev):
        """Latent predictive variance (prior 1) at the holds ev given the holds tr."""
        from scipy.linalg import solve_triangular
        tr, ev = np.nonzero(tr)[0], np.nonzero(ev)[0]
        if len(tr) == 0:
            return np.ones(len(ev))
        L = np.linalg.cholesky(self.K[np.ix_(tr, tr)] + self.nz ** 2 * np.eye(len(tr)))
        Vm = solve_triangular(L, self.K[np.ix_(tr, ev)], lower=True, check_finite=False)
        return np.clip(1.0 - np.sum(Vm ** 2, 0), 0.0, 1.0)


FAMILY_RULES = (("press", r"^(ref-|final-|hard-ref|hard-final|rod-ref|rod-final|tall-ref)", "refs"),
                ("press", r"^(fast-grid|fast\d-grid|rod-fast)", "fast grid"), ("press", r"^(rate-|rod-rate)", "rate set"),
                ("press", r"^pivot", "pivot"), ("press", r"^grid-", "grid"), ("press", r"^dense-", "dense"),
                ("press", r"^twin-", "twin"), ("press", r"^hard-", "hard"), ("press", r"^(rod-|tall-)", "grid"),
                ("slide", r"^slide-dense", "slide dense"), ("slide", r"^slide-twin", "slide twin"),
                ("slide", r"^slide-\d", "slide grid"), ("slide", r"^shear-", "shear"), ("slide", r"^fast-slide", "fast slide"),
                ("slide", r"^hard-", "hard slide"), ("slide", r"-tv\d+", "fast slide"), ("slide", r".", "slide grid"),
                ("twist", r"(-tv\d+|^fast-)", "twist rates"), ("twist", r"^hard-", "hard twist"), ("twist", r".", "twist"),
                ("roll", r"(-tv\d+|^fast-)", "roll rates"), ("roll", r".", "roll"), ("touch", r".", "touch-off"))


def coarse_label(kind, aid):
    """The action's family, "kind/name", by the plans' id patterns."""
    for k, rx, name in FAMILY_RULES:
        if k == kind and re.search(rx, str(aid)):
            return f"{k}/{name}"
    return f"{kind}/other"


def family_groups(H):
    lab = np.array([coarse_label(k, a) for k, a in zip(H["kind"].astype(str), H["id"].astype(str))])
    return collections.OrderedDict((name, lab == name) for name in sorted(set(lab.tolist())))


def info_part(H, cache_E, fa, fp):
    kind, reg = H["kind"].astype(str), H["region"].astype(str)
    act_s = dict(zip(H["id"].astype(str), H["act_s"]))
    secs = lambda m: float(sum(act_s[a] for a in np.unique(H["id"][m].astype(str))))  # noqa: E731
    G = collections.OrderedDict()
    for k in ("press", "slide", "twist", "roll"):
        for r in ("face", "edge", "flank"):
            m = (kind == k) & (reg == r)
            if m.any():
                G[f"{k}/{r}"] = m
    FG = family_groups(H)
    res = {}
    from scipy.stats import spearmanr
    for folds, F in (("action", fa), ("point", fp)):
        sel = cache_E[("all", folds)][1]
        ell, nz = float(np.median([x[0] for x in sel])), float(np.median([x[1] for x in sel]))
        t0 = time.time()
        Kr = Kern(H, ell, nz)
        R = res[folds] = dict(ell=ell, noise=nz)
        base_v = np.full(len(kind), np.nan)
        for q in range(5):
            te = F == q
            base_v[te] = Kr.var(~te, te)
        E = cache_E[("all", folds)][0]
        ok = np.isfinite(E["loc"])
        R["sd_vs_error"] = dict(spearman_loc=float(spearmanr(np.sqrt(base_v[ok]), E["loc"][ok])[0]),
                                spearman_force=float(spearmanr(np.sqrt(base_v[ok]), E["force"][ok])[0]),
                                mean_var=float(np.mean(base_v)), by_kind={k: float(np.mean(base_v[kind == k])) for k in G and
                                                                          ("press", "slide", "twist", "roll")})
        log(f"  info/{folds}: kernel ell {ell:.2f}, noise {nz:.2f}; mean variance {np.mean(base_v):.4f}; spearman(sd, loc) "
            f"{R['sd_vs_error']['spearman_loc']:.2f} ({time.time() - t0:.0f} s)")
        for label, groups in (("kind_region", G), ("family", FG)):
            rows = {}
            for gname, gm in groups.items():
                v = np.full(len(kind), np.nan)
                for q in range(5):
                    te = (F == q) & ~gm
                    v[te] = Kr.var(~(F == q) & ~gm, te)
                ev = ~gm
                sec = secs(gm)
                dv = float(np.nanmean(v[ev]) - np.nanmean(base_v[ev]))
                row = dict(n_actions=int(len(np.unique(H["id"][gm]))), n_holds=int(gm.sum()), robot_h=sec / 3600,
                           mean_var_with=float(np.nanmean(base_v[ev])), mean_var_without=float(np.nanmean(v[ev])),
                           rise=dv, rise_per_robot_h=dv / (sec / 3600))
                for k in ("press", "slide"):
                    e2 = ev & (kind == k)
                    row[f"rise_at_{k}"] = float(np.nanmean(v[e2]) - np.nanmean(base_v[e2])) if e2.any() else None
                rows[gname] = row
            order = sorted(rows, key=lambda g: -rows[g]["rise_per_robot_h"])
            R[f"drop_one_{label}"] = dict(rows=rows, order_per_robot_h=order)
            log(f"  info/{folds} drop one {label}: " + ", ".join(f"{g} {rows[g]['rise_per_robot_h']:.4f}/h" for g in order))
        # greedy: from no training, add the kind/region group that lowers the mean variance most per robot second
        cur, v_cur, steps, left = np.zeros(len(kind), bool), np.ones(len(kind)), [], list(G)
        while left:
            best = None
            for gname in left:
                v = np.full(len(kind), np.nan)
                for q in range(5):
                    te = F == q
                    v[te] = Kr.var((cur | G[gname]) & ~te, te)
                gain, sec = float(np.mean(v_cur) - np.mean(v)), secs(G[gname])
                if best is None or gain / sec > best[1] / best[2]:
                    best = (gname, gain, sec, v)
            gname, gain, sec, v = best
            cur |= G[gname]
            v_cur = v
            left.remove(gname)
            steps.append(dict(group=gname, gain=gain, robot_h=sec / 3600, gain_per_robot_h=gain / (sec / 3600),
                              mean_var_after=float(np.mean(v)), mean_var_after_press=float(np.mean(v[kind == "press"])),
                              mean_var_after_slide=float(np.mean(v[kind == "slide"]))))
        R["greedy_kind_region"] = steps
        log(f"  info/{folds} greedy: " + " -> ".join(f"{x['group']} ({x['gain_per_robot_h']:.3f}/h)" for x in steps))
        del Kr
    return res


# ================================================================================================ the suite's time
CHANGES = dict(slider1=dict(slider1=True), lift30=dict(lift30=True), slide_hold_0p2=dict(slide_hold=0.2),
               step_1p4=dict(step_deg=1.4, fast_turns=True), fast_turns=dict(fast_turns=True),
               all_program=dict(slider1=True, lift30=True, slide_hold=0.2, step_deg=1.4, fast_turns=True))
PROGRAM = CHANGES["all_program"]

# The reduced suite for v3: (stage, session, part = family label, decision, how many of the measured actions, the
# changes beyond PROGRAM, the variant of the value part that measures the change). "estimate" rows stand for R29's
# actions that did not run on the tall indenter, timed as the named sphere family's actions.
SUITE = [
    ("1 sphere pass 1", "sphere", "touch/touch-off", "keep", 1.0, {}, "registration"),
    ("2 sphere", "sphere", "press/grid", "keep", 1.0, {}, "density curve"),
    ("2 sphere", "sphere", "press/dense", "keep", 1.0, {}, "density curve"),
    ("2 sphere", "sphere", "press/twin", "keep", 1.0, {}, "density curve"),
    ("2 sphere", "sphere", "press/hard", "keep", 1.0, {}, "no_hard/point"),
    ("2 sphere", "sphere", "press/refs", "keep", 1.0, {}, "drift"),
    ("2 sphere", "sphere", "press/fast grid", "keep", 1.0, {}, "no_fastgrid/point"),
    ("2 sphere", "sphere", "press/rate set", "reduce", 6, {}, "no_rate/point"),
    ("2 sphere", "sphere", "slide/*", "keep", 1.0, {}, "halving"),
    ("2 sphere", "sphere", "twist/twist", "keep", 1.0, {}, "twist_board"),
    ("2 sphere", "sphere", "twist/hard twist", "keep", 1.0, {}, "twist_board"),
    ("2 sphere", "sphere", "twist/twist rates", "cut", 0.0, {}, "turn speeds"),
    ("2 sphere", "sphere", "roll/roll", "keep", 1.0, {}, "all vs presses+slides"),
    ("2 sphere", "sphere", "roll/roll rates", "cut", 0.0, {}, "turn speeds"),
    ("3 rod", "rod", "touch/touch-off", "keep", 1.0, {}, "registration"),
    ("3 rod", "rod", "press/pivot", "keep", 1.0, {}, "registration"),
    ("3 rod", "rod", "press/grid", "keep", 1.0, {}, "rod_tips"),
    ("3 rod", "rod", "press/refs", "keep", 1.0, {}, "drift"),
    ("3 rod", "rod", "press/fast grid", "cut", 0.0, {}, "rate set"),
    ("3 rod", "rod", "press/rate set", "cut", 0.0, {}, "no_rate/point"),
    ("3 rod", "rod", "slide/slide grid", "keep", 1.0, {}, "rod_tips"),
    ("3 rod", "rod", "slide/fast slide", "cut", 0.0, {}, "turn speeds"),
    ("3 rod", "rod", "roll/roll", "keep", 1.0, {}, "rod_tips"),
    ("3 rod", "rod", "roll/roll rates", "cut", 0.0, {}, "turn speeds"),
    ("3 rod", "rod", "twist/twist", "keep", 1.0, {}, "rod_tips"),
    ("3 rod", "rod", "twist/twist rates", "cut", 0.0, {}, "turn speeds"),
    ("4 tall", "tall", "touch/touch-off", "keep", 1.0, {}, "registration"),
    ("4 tall", "tall", "press/*", "keep", 1.0, {}, "r29_tall_vs_sphere"),
    ("4 tall", "estimate:tall_presses", "press/*", "keep", 15, {}, "r29_tall_vs_sphere"),
    ("4 tall", "estimate:shared_points", "press/grid", "keep", 12, {}, "r29 shear check"),
    ("4 tall", "estimate:sphere", "slide/slide twin", "keep", 1.0, {}, "halving"),
    ("4 tall", "estimate:sphere", "twist/twist", "reduce", 4, {}, "twist_board"),
]
# the shorter option: every slide at one amplitude, the first and the second on alternate slides
OPTION = dict(slide=dict(amp_alt=True))
# R29's actions past joint 6 +45 deg that did not run (QUEUE R29: 15 in all, 8 presses and 7 slides, the presses first)
R29_PAST_45 = {"estimate:sphere/slide/slide twin": 7}
# R29 as planned (its 122 actions; 45 presses ran): the counts of the estimate rows at full size
R29_FULL = {"estimate:tall_presses": 15, "estimate:shared_points": 12, "estimate:sphere/slide/slide twin": 50,
            "estimate:sphere/twist/twist": 12}


def select(P_all, session, part):
    ses = session.split(":")[-1] if session.startswith("estimate:") else session
    if ses in ("tall_presses", "shared_points"):
        ses = "tall"
    kind, fam = part.split("/", 1)
    return [p for p in P_all[ses] if p["kind"] == kind and (fam == "*" or p["family"] == part)]


def r29_twist_like(P_all):
    """The sphere's twists shaped as R29's (to 20 deg and back, 1 s holds, turn_v 0.016): their measured actions."""
    out = [p for p in P_all["sphere"] if p["kind"] == "twist" and p.get("hold_s") == 1.0 and p.get("turn_v") == 0.016]
    return out or [p for p in P_all["sphere"] if p["kind"] == "twist"]


def r29_slide_like(P_all):
    """The sphere's slides shaped as R29's (1 N, eight directions at 0.1 and 0.2 mm, turn_v 0.04, 0.4 s holds)."""
    return [p for p in P_all["sphere"] if p["kind"] == "slide" and p.get("turn_v") == 0.04 and len(p["moves"]) == 8
            and all(len(m["degs"]) == 2 for m in p["moves"])]


def part_actions(P_all, session, part):
    if session == "estimate:sphere" and part == "twist/twist":
        return r29_twist_like(P_all)
    if session == "estimate:sphere" and part.startswith("slide/"):
        return r29_slide_like(P_all)
    return select(P_all, session, part)


def plan_part(T, P_all):
    """The robot time (h) of v2 ID 1's suite as it ran plus R29's remainder (the v3 plan's stages 1 to 4), with each
    program change alone and together, and of the reduced suite."""
    out = {}
    # the suite as run, by stage and part, and with each program change
    rows = []
    for stage, ses, part, dec, n, extra, ref in SUITE:
        acts = part_actions(P_all, ses, part)
        if not acts:
            continue
        n_full = R29_FULL.get(ses, R29_FULL.get(f"{ses}/{part}", len(acts))) if ses.startswith("estimate") else len(acts)
        mean = lambda cfg: float(np.mean([action_time(p, T, cfg) for p in acts]))  # noqa: E731
        row = dict(stage=stage, session=ses, part=part, n=n_full, h=n_full * mean({}) / 3600,
                   **{f"h_{c}": n_full * mean(cfg) / 3600 for c, cfg in CHANGES.items()})
        n_keep = int(round(n * n_full)) if isinstance(n, float) else int(n)
        cfg = dict(PROGRAM, **extra)
        row.update(decision=dec, n_keep=n_keep, h_reduced=n_keep * mean(cfg) / 3600, value=ref,
                   s_per_action=mean({}), s_per_action_reduced=mean(cfg))
        if ses.startswith("estimate"):
            past = R29_PAST_45.get(f"{ses}/{part}", 0)
            s_past = row["s_per_action_reduced"]
        else:
            pa = [p for p in acts if (p.get("joint6_max") or -999.0) > 45.0]
            past = len(pa)
            s_past = float(np.mean([action_time(p, T, cfg) for p in pa])) if pa else 0.0
        row["n_past_j6_45"] = past
        row["n_past_j6_45_kept"] = int(round(past * n_keep / max(n_full, 1)))
        row["h_past_j6_45_kept"] = row["n_past_j6_45_kept"] * s_past / 3600
        rows.append(row)
    out["parts"] = rows
    tot = lambda k: float(sum(r[k] for r in rows))  # noqa: E731
    out["total_h"] = dict(as_run=tot("h"), **{c: tot(f"h_{c}") for c in CHANGES}, reduced=tot("h_reduced"),
                          reduced_past_j6_45=tot("h_past_j6_45_kept"))
    out["n_past_j6_45_kept"] = int(sum(r["n_past_j6_45_kept"] for r in rows))
    # the reduced suite without the turn path's larger steps (the turn model's extrapolation past 0.5 deg is unmeasured)
    cfg0 = dict(PROGRAM)
    cfg0.pop("step_deg")
    h = 0.0
    for stage, ses, part, dec, n, extra, ref in SUITE:
        acts = part_actions(P_all, ses, part)
        if not acts:
            continue
        n_full = R29_FULL.get(ses, R29_FULL.get(f"{ses}/{part}", len(acts))) if ses.startswith("estimate") else len(acts)
        n_keep = int(round(n * n_full)) if isinstance(n, float) else int(n)
        h += n_keep * float(np.mean([action_time(p, T, dict(cfg0, **extra)) for p in acts])) / 3600
    out["total_h"]["reduced_without_step_1p4"] = h
    h = 0.0
    for stage, ses, part, dec, n, extra, ref in SUITE:
        acts = part_actions(P_all, ses, part)
        if not acts:
            continue
        n_full = R29_FULL.get(ses, R29_FULL.get(f"{ses}/{part}", len(acts))) if ses.startswith("estimate") else len(acts)
        n_keep = int(round(n * n_full)) if isinstance(n, float) else int(n)
        ex = dict(extra, **OPTION["slide"]) if part.startswith("slide") else extra
        h += n_keep * float(np.mean([action_time(p, T, dict(PROGRAM, **ex)) for p in acts])) / 3600
    out["total_h"]["option_slides_alternate_amplitude"] = h
    by_stage = collections.OrderedDict()
    for r in rows:
        b = by_stage.setdefault(r["stage"], dict(h=0.0, h_all_program=0.0, h_reduced=0.0))
        for k in b:
            b[k] += r[k]
    out["by_stage"] = by_stage
    # the actions past joint 6 +45 deg in the measured sessions (the owner's presence)
    out["past_j6_45"] = {}
    for name, P in P_all.items():
        past = [p for p in P if (p.get("joint6_max") or -999) > 45.0]
        out["past_j6_45"][name] = dict(n=len(past), h=sum(p["dur"] for p in past) / 3600,
                                       h_all_program=sum(action_time(p, T, PROGRAM) for p in past) / 3600,
                                       by_family=dict(collections.Counter(p["family"] for p in past)))
    log("plan: suite as run %.2f h; with the program changes %.2f h; reduced %.2f h" % (
        out["total_h"]["as_run"], out["total_h"]["all_program"], out["total_h"]["reduced"]))
    for r in rows:
        log(f"  {r['stage']:16s} {r['session'][:22]:22s} {r['part']:22s} {r['decision']:6s} {r['n']:4d} -> {r['n_keep']:4d}  "
            f"{r['h']:.2f} h -> {r['h_all_program']:.2f} h (program) -> {r['h_reduced']:.2f} h")
    return out


def alternating(H):
    """Slide holds kept when each slide keeps half of its moves: by the slide's place k in the seed-0 permutation of the
    slides, directions 0/90/180/270 (k even) or 45/135/225/315 (k odd); the first amplitude (k even) or the second
    (k odd); and both halvings with k mod 4."""
    S = H["kind"] == "slide"
    sid = np.random.default_rng(0).permutation(np.unique(H["id"][S].astype(str)))
    k = {x: i for i, x in enumerate(sid)}
    kk = np.array([k.get(x, -1) for x in H["id"].astype(str)])
    d, amp = H["dir_deg"], H["amp"]
    first = amp == 0
    axis = np.isin(d, [0, 90, 180, 270])
    diag = np.isin(d, [45, 135, 225, 315])
    a1, a2 = np.isin(amp, [1, 9]), np.isin(amp, [2, 9])
    even, q = (kk % 2) == 0, kk % 4
    return dict(dirs4alt=first | (even & axis) | (~even & diag),
                ampalt=first | (even & a1) | (~even & a2),
                dirs4alt_ampalt=first | ((q == 0) & axis & a1) | ((q == 1) & diag & a2) | ((q == 2) & diag & a1)
                | ((q == 3) & axis & a2))


def combos(H):
    """The combined slide reductions tested in both fold types: (name, holds kept, note)."""
    S = H["kind"] == "slide"
    d, amp = H["dir_deg"], H["amp"]
    first = amp == 0
    sid = np.random.default_rng(0).permutation(np.unique(H["id"][S].astype(str)))
    four = np.isin(d, [0, 90, 180, 270]) | first
    one = np.isin(amp, [0, 1, 9])
    return [("dirs4_amp1", four & one, "4 directions at the first amplitude"),
            ("dirs4_amp1_count50", four & one & np.isin(H["id"].astype(str), sid[:50]),
             "4 directions at the first amplitude, 50 of the 100 slides")]


def suite_mask(H, session="sphere", option=False):
    """The holds of the session that the reduced suite (SUITE) keeps: each part's kept share of its actions (seed 0
    permutation, the same as the count variants), with a slide's directions and amplitude as its row says; with option,
    the slides at alternate amplitudes (the variant ampalt)."""
    lab = np.array([coarse_label(k, a) for k, a in zip(H["kind"].astype(str), H["id"].astype(str))])
    ids_all = H["id"].astype(str)
    keep = np.zeros(len(lab), bool)
    for stage, ses, part, dec, n, extra, ref in SUITE:
        if ses != session or dec == "cut":
            continue
        kind, fam = part.split("/", 1)
        m = (H["kind"] == kind) & ((lab == part) if fam != "*" else np.ones(len(lab), bool))
        ids = np.unique(ids_all[m])
        n_keep = int(round(n * len(ids))) if isinstance(n, float) else int(n)
        mm = m & np.isin(ids_all, np.random.default_rng(0).permutation(ids)[:n_keep])
        if extra.get("dirs") is not None:
            mm &= np.isin(H["dir_deg"], extra["dirs"]) | (H["amp"] == 0)
        if extra.get("amp1"):
            mm &= np.isin(H["amp"], [0, 1, 9])
        if option and kind == "slide":
            mm &= alternating(H)["ampalt"]
        keep |= mm
    return keep


def reduced_part(H, cache_E, fa, fp):
    """The reduced suite's sphere holds as training: the presses-only GP on its presses against "presses", and the
    all-kinds GP on all its holds against "all"; the suite in both fold types, its shorter option (slides at alternate
    amplitudes) by action."""
    kind = H["kind"].astype(str)
    P, S, T, R = (kind == "press"), (kind == "slide"), (kind == "twist"), (kind == "roll")
    out = {}
    for label, option, fold_list in (("reduced", False, (("action", fa), ("point", fp))), ("option", True, (("action", fa),))):
        keep = suite_mask(H, option=option)
        for folds, F in fold_list:
            for name, tr, ref, crit, tests in (
                    (f"{label}_presses", keep & P, "presses", dict(press=P), dict(press=P)),
                    (f"{label}_all", keep, "all", dict(slide=S, twist=T, roll=R), dict(press=P, slide=S, twist=T, roll=R))):
                Y, sel = fold_predict(H, tr, F, f"{folds}-{name}")
                E1 = errors(H, Y)
                cache_E[(name, folds)] = (E1, sel)
                E0 = cache_E[(ref, folds)][0]
                row = dict(note=f"the {label} suite's sphere holds in training", folds=folds, n_train_holds=int(tr.sum()),
                           n_train_actions=int(len(np.unique(H["id"][tr]))), ell_noise=sorted(set(sel)),
                           tests={t: table(H, E1, m) for t, m in tests.items()}, reference=f"{ref}/{folds}")
                row["vs_ref"], row["growth"] = compare(H, E1, E0, crit)
                if name.endswith("_all"):
                    row["vs_ref_press"] = compare(H, E1, E0, dict(press=P))[0]
                out[f"{name}/{folds}"] = row
                log(f"  {name}/{folds}: {int(tr.sum())} holds; growth {row['growth']['growth']:+.3f} at {row['growth']['at']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", default="times,value,info,plan")
    ap.add_argument("--quick", action="store_true", help="one fold of the training-kind variants only (a check)")
    ap.add_argument("--skip-reduced", action="store_true", help="leave out the reduced suite's fits")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    parts = a.parts.split(",")
    res = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    t0 = time.time()
    T = P_all = segs = None
    if "times" in parts or "plan" in parts:
        T, P_all, segs = times_part()
        res["times"] = T
    if any(p in parts for p in ("value", "info")):
        H = sphere_holds()
        V, cache_E, fa, fp = value_part(H, a.quick)
        res["value"] = V
        if not a.quick:
            for name, keep, note in combos(H):
                V["variants"].update(combo_part(H, cache_E, fa, fp, keep, name, note))
            if not a.skip_reduced:
                V["variants"].update(reduced_part(H, cache_E, fa, fp))
            V["fits"] = dict(FITS)
        if "info" in parts and not a.quick:
            res["info"] = info_part(H, cache_E, fa, fp)
    if "plan" in parts and T is not None:
        res["plan"] = plan_part(T, P_all)
    res.pop("run", None)
    res.setdefault("runs", {})[a.parts] = dict(seconds=time.time() - t0, fits=dict(FITS), date=time.strftime("%Y-%m-%d %H:%M"))
    Path(a.out).write_text(json.dumps(rnd(res, 4), indent=1) + "\n")
    log(f"-> {a.out} ({time.time() - t0:.0f} s; fits {FITS})")


if __name__ == "__main__":
    main()
