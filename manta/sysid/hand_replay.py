#!/usr/bin/env python3
"""Sim-to-real gate for policy C (the policy that uses load feedback): replay recorded hand sweeps in
MuJoCo with the fitted BAM (Better Actuator Models) SCS0009 actuator, and compare simulated position
and load with the real hand, joint by joint.

    bam_pendulum/../.venv-bam/bin/python hand_replay.py [--models m1 m4 m6] [--plot hand_replay.png]

Data: full-range sweeps (full_range_sweep.py, pass 3, 2026-10-29): one joint at a time ramps at 2 deg/s
over its range and back, P 15 / D 15, logged at ~500 Hz; every other servo is unpowered (held by its
gearbox) at its start angle, so those joints are pinned there with equality constraints.
Model: hand-main real_v3 (CAD masses), hand pointing down (gravity -z as in the sweeps), contacts off
(free air), design-search joints (x, y, len) removed. Joint mapping: driver finger 0/1/2 = thumb /
index / middle, aa/fe1/fe2 = yaw/mcp/pip, MuJoCo angle = driver angle (both: + = flexion, ranges
agree; zero offsets assumed 0, the v3 ROM is not calibrated).
Load: real register -> duty = -load/1000 (same convention as the pendulum rig; checked per joint by
the sign of corr(load, target - position)). Simulated duty comes from the actuator's control law.
Friction / bias split as in README: friction = (load going down - load going up)/2, bias = mean.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bam_pendulum"))
import bam_scs0009  # noqa: E402

bam_scs0009.register()
from bam.model import load_model  # noqa: E402
from bam.mujoco import MujocoController  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "../../assets/mjcf/real_v3/real_hand.xml")   # hand repo root, two levels up
FITS = os.path.join(HERE, "bam_pendulum/fits_best")
RAW = [os.path.join(HERE, f"full_range_pass3_raw_2026-10-29_{s}.txt") for s in ("a", "b")]
SCALES = os.path.join(HERE, "hand_friction_scales.json")
FSCALE: dict = {}     # joint -> friction scale (--use-scales)
FINGER = {0: "thumb", 1: "index", 2: "middle"}
JOINT = {"aa": "yaw", "fe1": "mcp", "fe2": "pip"}
DT = 0.002
QSTEP = None       # set by --quantize
VIN = 5.0          # hand supply; the health check read 5.0-5.3 V at rest


def parse_sweeps(raw=RAW):
    """-> list of dicts per joint: finger, joint, start (deg), t, phase, target, pos, load arrays."""
    out = []
    for f in raw:
        cur = None
        for line in open(f):
            m = re.match(r"finger (\d) (\w+) \(servo (\d)\): start ([-\d.]+)", line)
            if m:
                cur = {"finger": int(m[1]), "joint": m[2], "sid": int(m[3]), "start": float(m[4]), "rows": []}
                out.append(cur)
            elif cur and line[:1].isdigit() and line.count(",") == 4:
                t, ph, tg, p, ld = line.strip().split(",")
                cur["rows"].append((float(t), ph, float(tg), float(p), float(ld)))
    for s in out:
        r = s.pop("rows")
        s["t"] = np.array([x[0] for x in r]); s["phase"] = np.array([x[1] for x in r])
        s["target"] = np.array([x[2] for x in r]); s["pos"] = np.array([x[3] for x in r])
        s["load"] = np.array([x[4] for x in r])
        # drop single-sample read tears (README: position spikes, not motion)
        bad = np.abs(s["pos"] - np.convolve(s["pos"], np.ones(9) / 9, "same")) > 3
        s["pos"][bad] = np.interp(s["t"][bad], s["t"][~bad], s["pos"][~bad])
    return out


def build(active: str, rest: dict[str, float]):
    spec = mujoco.MjSpec.from_file(XML)
    for a in list(spec.actuators):
        spec.delete(a)
    for k in list(spec.keys):
        spec.delete(k)
    for j in list(spec.joints):
        if j.name.endswith(("_x", "_y", "_len")):
            spec.delete(j)
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    for j in spec.joints:
        j.damping = np.zeros(3); j.frictionloss = 0.0; j.armature = 0.0
        j.limited = mujoco.mjtLimited.mjLIMITED_FALSE
        if j.name != active:
            e = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=j.name)
            e.data[:5] = [rest[j.name], 0, 0, 0, 0]
    spec.add_actuator(name=active, target=active, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    m = spec.compile()
    return m, mujoco.MjData(m)


def rollout(model, sweep, rest):
    name = f"{FINGER[sweep['finger']]}_{JOINT[sweep['joint']]}"
    m, d = build(name, rest)
    for jn, q in rest.items():
        d.qpos[m.joint(jn).qposadr[0]] = q
    act = model.actuator
    act.kp, act.kd, act.vin = 15, 15, VIN
    act.quantize_step = QSTEP
    model.reset()
    k_f = FSCALE.get(name, 1.0)
    base = type(model).compute_frictions
    model.compute_frictions = lambda mt, et, dq: (lambda fl, dm: (fl * k_f, dm))(*base(model, mt, et, dq))
    ctrl = MujocoController(model, name, m, d)
    t = sweep["t"] - sweep["t"][0]
    goal = np.radians(sweep["target"])
    d.qpos[ctrl.qpos_indexes] = np.radians(sweep["pos"][0])
    mujoco.mj_forward(m, d)
    ctrl.reset()
    ts = np.arange(0.0, t[-1], DT)
    q = np.empty(len(ts)); duty = np.empty(len(ts))
    delay = model.command_delay.value
    for k, tt in enumerate(ts):
        ctrl.set_q_target(name, np.interp(max(tt - delay, 0.0), t, goal))
        ctrl.update()
        duty[k] = np.atleast_1d(act.duty_cycle)[0]
        mujoco.mj_step(m, d)
        q[k] = d.qpos[ctrl.qpos_indexes][0]
    return np.degrees(np.interp(t, ts, q)), 1000 * np.interp(t, ts, duty)


def split(pos, load, phase):
    """friction and bias (load units) in 10-deg bins, from the up / down legs."""
    up, dn = phase == "up", phase == "down"
    lo, hi = np.percentile(pos[up | dn], [5, 95])
    fr, bi = [], []
    for a in np.arange(lo, hi, 10):
        mu = up & (pos >= a) & (pos < a + 10); md = dn & (pos >= a) & (pos < a + 10)
        if mu.sum() > 50 and md.sum() > 50:
            lu, ld = load[mu].mean(), load[md].mean()
            fr.append((ld - lu) / 2); bi.append((ld + lu) / 2)
    return np.array(fr), np.array(bi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["m1", "m4", "m6"])
    ap.add_argument("--plot", default=None)
    ap.add_argument("--quantize", action="store_true", help="firmware sees whole encoder counts")
    ap.add_argument("--pass", dest="pas", type=int, default=3, help="sweep pass to replay (2 or 3)")
    ap.add_argument("--calibrate", action="store_true", help="save per-joint friction scale = real/sim friction (last model)")
    ap.add_argument("--use-scales", action="store_true", help="apply hand_friction_scales.json")
    a = ap.parse_args()
    global QSTEP
    QSTEP = np.radians(300 / 1024) if a.quantize else None
    raw = [os.path.join(HERE, f"full_range_pass{a.pas}_raw_2026-10-29_{s}.txt") for s in ("a", "b")]
    sweeps = parse_sweeps(raw)
    if a.use_scales:
        import json
        FSCALE.update(json.load(open(SCALES)))
        print("friction scales:", {k: round(v, 2) for k, v in FSCALE.items()})
    cal = {}
    rest = {f"{FINGER[s['finger']]}_{JOINT[s['joint']]}": np.radians(s["start"]) for s in sweeps}
    models = {n: load_model(os.path.join(FITS, f"{n}.json")) for n in a.models}

    print("position MAE (deg) | load: corr, MAE  | friction real/sim | gravity+bias real/sim (range of 10-deg bins, load units)")
    results = {}
    for s in sweeps:
        jn = f"{FINGER[s['finger']]}_{JOINT[s['joint']]}"
        e = s["target"] - s["pos"]
        moving = np.isin(s["phase"], ("up", "down"))
        sgn = -np.sign(np.corrcoef(s["load"][moving], e[moving])[0, 1])   # expect +1: duty = -load/1000
        real_load = -sgn * s["load"]     # in duty*1000 units, + = pushing toward + angle... see docstring
        fr_r, bi_r = split(s["pos"], real_load, s["phase"])
        line = f"{jn:12s} (load sign {'ok' if sgn > 0 else 'FLIPPED'})"
        for n, model in models.items():
            q, sim_load = rollout(model, s, rest)
            k = np.ones(25) / 25
            rl, sl = np.convolve(real_load, k, "same"), np.convolve(sim_load, k, "same")
            pm = np.mean(np.abs(q - s["pos"])[moving])
            corr = np.corrcoef(rl[moving], sl[moving])[0, 1]
            lm = np.mean(np.abs(rl - sl)[moving])
            fr_s, bi_s = split(q, sim_load, s["phase"])
            results[(jn, n)] = (s, q, sim_load)
            cal[jn] = FSCALE.get(jn, 1.0) * float(np.mean(fr_r) / np.mean(fr_s))
            line += (f"\n    {n.upper()}: pos {pm:5.2f} | load corr {corr:4.2f} mae {lm:4.0f} | "
                     f"friction {np.mean(fr_r):4.0f}/{np.mean(fr_s):4.0f} | "
                     f"bias {bi_r.min():+4.0f}..{bi_r.max():+4.0f} / {bi_s.min():+4.0f}..{bi_s.max():+4.0f}")
        print(line, flush=True)

    if a.calibrate:
        import json
        json.dump(cal, open(SCALES, "w"), indent=1)
        print("saved", SCALES, {k: round(v, 2) for k, v in cal.items()})

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n0 = a.models[-1]
        fig, axes = plt.subplots(len(sweeps), 2, figsize=(14, 2.2 * len(sweeps)))
        for row, s in zip(axes, sweeps):
            jn = f"{FINGER[s['finger']]}_{JOINT[s['joint']]}"
            _, q, sl = results[(jn, n0)]
            t = s["t"] - s["t"][0]
            row[0].plot(t, s["target"], "k:", lw=1); row[0].plot(t, s["pos"], "k", lw=1.2)
            row[0].plot(t, q, "tab:blue", lw=1); row[0].set_ylabel(f"{jn}\ndeg", fontsize=8)
            k = np.ones(25) / 25
            sgn = -np.sign(np.corrcoef(s["load"], s["target"] - s["pos"])[0, 1])
            row[1].plot(t, np.convolve(-sgn * s["load"], k, "same"), "k", lw=1, label="real load")
            row[1].plot(t, np.convolve(sl, k, "same"), "tab:blue", lw=1, label=f"sim {n0.upper()}")
            row[1].set_ylabel("load", fontsize=8)
        axes[0, 0].set_title("position: target (dotted), real (black), sim (blue)", fontsize=9)
        axes[0, 1].set_title("load (PWM x 1000), 50 ms smoothing", fontsize=9); axes[0, 1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(a.plot, dpi=90)
        print("plot ->", a.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
