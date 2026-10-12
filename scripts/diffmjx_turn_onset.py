#!/usr/bin/env python3
"""Forward parity and gradient cost of the DiffMJX fork at the turn onset of the D1 continuation (2 mm packed pads).

The measurement Codex named as the next step on 2026-10-07, with the stop rule of the audit of 2026-10-11:

  1. parity    the fork and CPU MuJoCo 3.3.1 (the fork's own MuJoCo) run 100 ms from the held state of the 2 mm packed
               pads under the first-phase controls of the 2 mm free hybrid plan of 2026-10-07 and of the hold plan,
               sampled every 10 ms. Parity holds when, for the turn plan at 100 ms, the fork's orientation gain is
               within PARITY_GAIN_REL of CPU MuJoCo's and its tool position within PARITY_POS_M.
  2. gradient  only if parity holds: the forward-mode gradient of the 100 ms vertical cosine with respect to the five
               plan values active before 120 ms (three yaw offsets, two opening amplitudes), compilation included,
               against CPU central differences at a 1e-2 step. A step along each direction is scored in CPU MuJoCo.
  3. search    only if one useful gradient costs at most STOP_RATIO times the CPU finite differences.

Run in the isolated DiffMJX environment on the CPU (JAX_PLATFORM_NAME=cpu); the result goes to
docs/experiments/20261011-diffmjx_audit/turn_onset.json, written after each stage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import real_v1_d1_continuation as d1  # noqa: E402

OUT = ROOT / "docs/experiments/20261011-diffmjx_audit/turn_onset.json"
PLAN = ROOT / "docs/experiments/20261007-contact-gait/20261007-padsT2-seed1-256.json"
HORIZON = 100           # ms, inside the first plan phase (0-120 ms) and the opening window (0-180 ms)
SAMPLE = 10             # ms between parity samples
ACTIVE = [0, 1, 2, 9, 10]   # plan values that act before 120 ms
FD_STEP = 1e-2
PROBE = 0.05            # plan units of the step along each gradient direction
PARITY_GAIN_REL = 0.10
PARITY_POS_M = 0.5e-3
STOP_RATIO = 100.0


def save(result: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(result, indent=1, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, OUT)


def cpu_trace(scene, x, horizon=HORIZON, every=SAMPLE):
    """Samples of vertical cosine, tool position and per-finger normal force every `every` ms from the held state."""
    data = d1.clone_data(scene.model, scene.held)
    data.ctrl[:] = scene.ctrl(x, 0)
    out = []
    t0 = time.perf_counter()
    for k in range(horizon):
        scene.mujoco.mj_step(scene.model, data)
        if (k + 1) % every == 0:
            out.append(sample(scene, data.qpos, d1.contact_summary(scene.model, data)["finger_normal_force_N"], k + 1))
    return out, time.perf_counter() - t0


def sample(scene, qpos, forces, t_ms):
    q = np.asarray(qpos)
    return {"t_ms": t_ms, "vertical_cos": float(d1.vertical_cos(q[scene.qadr + 3:scene.qadr + 7])),
            "tool_pos_m": q[scene.qadr:scene.qadr + 3].tolist(),
            "finger_normal_force_N": {f: float(v) for f, v in forces.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan-method", default="hybrid")
    args = parser.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    scene = d1.Scene("tpu6", "padsT", 1, pad_spacing_mm=2.0)
    plans = {"turn": np.asarray(json.loads(PLAN.read_text())["candidates"][args.plan_method]["x"], float),
             "hold": d1.hold_vector()}
    result = {"stage": "parity", "scene": str(scene.scene.relative_to(ROOT)), "plan_file": str(PLAN.relative_to(ROOT)),
              "plan_method": args.plan_method, "mujoco_version": mujoco.__version__, "jax_version": jax.__version__,
              "jax_backend": jax.default_backend(), "horizon_ms": HORIZON, "sample_ms": SAMPLE,
              "criterion": {"gain_rel": PARITY_GAIN_REL, "position_m": PARITY_POS_M, "stop_ratio": STOP_RATIO},
              "n_pads": scene.meta["n_pads"], "initial_vertical_cos": scene.cos0, "plans": {}}

    # ---------------------------------------------------------------- 1. forward parity
    mx, md, put_model_s = d1.mjx_setup(scene.model, scene.held)
    chunk = jax.jit(d1.mjx_rollout(mx, SAMPLE))
    t0 = time.perf_counter()
    for name, x in plans.items():
        ctrl = jnp.asarray(scene.ctrl(x, 0))
        state = md.replace(ctrl=ctrl)
        fork, chunk_times = [], []
        for i in range(HORIZON // SAMPLE):
            state, dt = d1.timed(chunk, state)
            chunk_times.append(dt)
            fork.append(sample(scene, state.qpos, {f: v["normal_force_N"] for f, v in
                                                   d1.mjx_finger_forces(scene.model, state).items()}, (i + 1) * SAMPLE))
        cpu, cpu_s = cpu_trace(scene, x)
        result["plans"][name] = {"x": x.tolist(), "cpu": cpu, "fork": fork, "cpu_rollout_s": cpu_s,
                                 "fork_chunk_s": chunk_times}
        print(name, "cpu gain", cpu[-1]["vertical_cos"] - scene.cos0, "fork gain", fork[-1]["vertical_cos"] - scene.cos0,
              flush=True)
    result["fork_first_chunk_s"] = result["plans"]["turn"]["fork_chunk_s"][0]
    result["fork_warm_100ms_s"] = float(sum(result["plans"]["hold"]["fork_chunk_s"]))
    result["put_model_s"] = put_model_s
    result["parity_wall_s"] = time.perf_counter() - t0
    turn = result["plans"]["turn"]
    gain_cpu = turn["cpu"][-1]["vertical_cos"] - scene.cos0
    gain_fork = turn["fork"][-1]["vertical_cos"] - scene.cos0
    pos_err = float(np.max(np.abs(np.subtract(turn["fork"][-1]["tool_pos_m"], turn["cpu"][-1]["tool_pos_m"]))))
    result["parity"] = {"gain_cpu": gain_cpu, "gain_fork": gain_fork,
                        "gain_rel_error": abs(gain_fork - gain_cpu) / abs(gain_cpu),
                        "tool_position_error_m": pos_err}
    result["parity"]["holds"] = bool(result["parity"]["gain_rel_error"] <= PARITY_GAIN_REL and pos_err <= PARITY_POS_M)
    save(result)
    print("parity", result["parity"], flush=True)
    if not result["parity"]["holds"]:
        result["stage"] = "stopped: forward parity fails"
        result["decision"] = "stop"
        save(result)
        return

    # ---------------------------------------------------------------- 2. one gradient
    result["stage"] = "gradient"
    base = plans["turn"]
    ctrl0, low, high = (jnp.asarray(a) for a in (scene.ctrl0, scene.low, scene.high))
    yaw, flex = jnp.asarray(scene.yaw_ids), {f: jnp.asarray(v) for f, v in scene.flex_ids.items()}
    rollout = d1.mjx_rollout(mx, HORIZON)

    def plan_ctrl_jax(x):
        c = ctrl0.at[yaw].set(jnp.clip(ctrl0[yaw] + d1.YAW_SCALE_RAD * x[0:3], low[yaw], high[yaw]))
        for i, f in enumerate(("index", "middle")):
            c = c.at[flex[f]].set(ctrl0[flex[f]] + 0.5 * (x[9 + i] + 1.0) * (low[flex[f]] - ctrl0[flex[f]]))
        return c

    def objective(theta):
        x = jnp.asarray(base).at[jnp.asarray(ACTIVE)].set(theta)
        final = rollout(md.replace(ctrl=plan_ctrl_jax(x)))
        return d1.vertical_cos(final.qpos[scene.qadr + 3:scene.qadr + 7])

    theta0 = jnp.asarray(base[ACTIVE])
    grad_fn = jax.jit(jax.jacfwd(objective))
    ad, first_s = d1.timed(grad_fn, theta0)
    ad_warm_s = d1.timed(grad_fn, theta0)[1]
    ad = np.asarray(ad)

    def cpu_objective(theta):
        x = base.copy()
        x[ACTIVE] = theta
        data = d1.clone_data(scene.model, scene.held)
        data.ctrl[:] = scene.ctrl(x, 0)
        for _ in range(HORIZON):
            mujoco.mj_step(scene.model, data)
        return [float(d1.vertical_cos(data.qpos[scene.qadr + 3:scene.qadr + 7]))]

    t0 = time.perf_counter()
    fd = d1.central_difference(cpu_objective, base[ACTIVE], np.eye(len(ACTIVE)), FD_STEP)[0]
    fd_s = time.perf_counter() - t0
    c0 = cpu_objective(base[ACTIVE])[0]
    probe = {}
    for name, g in (("ad", ad), ("cpu_fd", fd)):
        step = base[ACTIVE] + PROBE * g / np.linalg.norm(g)
        step = np.clip(step, -1.0, 1.0)
        probe[name] = cpu_objective(step)[0] - c0
    ratio = (first_s + 0.0) / fd_s
    result.update(stage="gradient done", gradient={
        "active": ACTIVE, "ad": ad.tolist(), "cpu_fd": fd.tolist(), "fd_step": FD_STEP,
        "cosine": float(np.dot(ad, fd) / (np.linalg.norm(ad) * np.linalg.norm(fd))),
        "first_call_s": first_s, "warm_s": ad_warm_s, "cpu_fd_s": fd_s, "cpu_fd_evals": 2 * len(ACTIVE),
        "ratio_first": ratio, "ratio_warm": ad_warm_s / fd_s, "probe_step": PROBE,
        "cpu_gain_along_ad": probe["ad"], "cpu_gain_along_fd": probe["cpu_fd"]})
    result["decision"] = "stop" if ratio > STOP_RATIO else "search"
    if ratio > STOP_RATIO:
        result["stage"] = "stopped: one gradient costs more than the stop ratio"
    save(result)
    print(json.dumps(result["gradient"], indent=1), flush=True)


if __name__ == "__main__":
    main()
