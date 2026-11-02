#!/usr/bin/env python3
"""Scan the firmware D-term scale on high-D double-pendulum logs, and compare simulated vs real
load (PWM duty x 1000) on the hand's setting P 15 / D 15.  ../.venv-bam/bin/python d_gain_check.py"""
import glob

import mujoco
import numpy as np

import double_replay as R
from bam.model import load_model
from bam.mujoco import MujocoController

OFF = np.radians([-R.HANG_DEG[0], -R.HANG_DEG[1]])


def rollout_with_load(model, log):
    model.reset()
    mm = mujoco.MjModel.from_xml_string(R.mjcf(*OFF)); md = mujoco.MjData(mm)
    act = model.actuator
    act.kp, act.vin, act.kd = log["kp"], log["vin"], log["kd"]
    ctrl = MujocoController(model, ["j1", "j2"], mm, md)
    t, q, goal = log["t"] - log["t"][0], log["q"], log["goal"]
    md.qpos[:] = q[0]; mujoco.mj_forward(mm, md); ctrl.reset()
    ts = np.arange(0.0, t[-1], R.DT); pos = np.empty((len(ts), 2)); duty = np.empty((len(ts), 2))
    for k, tt in enumerate(ts):
        tg = max(tt - model.command_delay.value, 0.0)
        for j, n in enumerate(("j1", "j2")):
            ctrl.set_q_target(n, np.interp(tg, t, goal[:, j]))
        ctrl.update(); duty[k] = act.duty_cycle; mujoco.mj_step(mm, md); pos[k] = md.qpos
    return (np.stack([np.interp(t, ts, pos[:, j]) for j in range(2)], 1),
            np.stack([np.interp(t, ts, duty[:, j]) for j in range(2)], 1))


tune = [R.load_double(f) for f in sorted(glob.glob("data_raw/double_v2_tune/*.json"))]
tune = [l for l in tune if l["kd"] >= 45]
print("D-term scale scan (M4), mean joint error over the high-D logs (deg):")
for g in (0.0, 0.001, 0.00125, 0.0015, 0.00175, 0.002):
    m = load_model("fits_best/m4.json"); m.d_gain.value = g
    e = np.mean([np.degrees(np.mean(np.abs(rollout_with_load(m, l)[0] - l["q"])[l["path_mask"]], 0)) for l in tune], 0)
    print(f"  {g:.5f}: joint 1 {e[0]:.3f}, joint 2 {e[1]:.3f}")

import json
print("\nLoad, P 15 / D 15 logs (the hand's setting): real vs simulated PWM x 1000")
for name in ("m1", "m4"):
    m = load_model(f"fits_best/{name}.json")
    for f in sorted(glob.glob("data_raw/double_v2_kd15/*kp15_kd15.json")):
        l = R.load_double(f)
        raw = json.load(open(f)); real = -np.array([e["load"] for e in raw["entries"]], float)
        sim_q, sim_d = rollout_with_load(m, l); sim = 1000 * sim_d
        pm = l["path_mask"]
        out = []
        for j in range(2):
            r, s = real[pm, j], sim[pm, j]
            # compare after 50 ms smoothing (the register is quantized and noisy)
            k = np.ones(13) / 13; rs, ss = np.convolve(r, k, "same"), np.convolve(s, k, "same")
            out.append(f"j{j+1} corr {np.corrcoef(rs, ss)[0,1]:.2f} mae {np.mean(np.abs(rs-ss)):4.0f} (real |mean| {np.mean(np.abs(rs)):4.0f})")
        print(f"  {name.upper()} {l['name'][16:-15]:14s} " + " | ".join(out))
