#!/usr/bin/env python3
"""Contact-model comparison bed, task 1: tangential pull-to-slip and pre-slip creep of a pinched tool.

The rig is the two-pad pinch of the 10-01 study (scripts/hom_contact_rig.py): two real_v1 fingertip
spheres (r 10.55 mm, 20 g) on force-controlled rails along world x squeeze the real_v1 screwdriver
(cylinder r 12.5 mm, 100 mm, 24.5 g), mu 1, gravity off. Per case (model, pinch N per pad, step dt):

  ramp   settle 0.4 s at N, then pull the tool along its own axis at 2 N/s until gross slip
         (tool speed > 10 mm/s) and 80 ms beyond. Onset is the vertex of a quadratic fit of the
         slip speed over the 40 ms after detection (the time the slip acceleration was zero), so
         the reported onset force does not carry the detection threshold's lag.
  hold   fresh rig, settle, ramp at 2 N/s to 50 % of the measured onset force, hold 1 s; creep
         rate is the slope of the axial displacement over the last 0.8 s of the hold.

Effective mu = F_onset / (2 N): two pads in parallel, each pressed with N. Slip speed is the mean
over the first 50 ms after onset (rigid Coulomb, equal static and kinetic mu: 34 mm/s) with the
coefficient of variation of the per-step speed as a stick-slip flag. Sliding mu is
(F - m a) / (2 N), a from a quadratic fit of the displacement over 5-60 ms after onset. Physics cost is the wall time of
the rig's step call alone (for mj:point4s it includes the per-step Python mu_t rewrite).

Model specs are the rig strings that the 10-01 torsion rows were measured with; `chain` is the
hom_chain.py spec the model stands for.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/contact_bed_pull.py --models mj:point3 --N 1 --dt 1          # one case
    $PY scripts/contact_bed_pull.py --out docs/experiments/20261005-contact_bed/pull_slip.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_contact_rig as H  # noqa: E402

MODELS = {   # chain spec (hom_chain.make_plant) -> rig spec (hom_contact_rig.make_rig), as in 10-01 torsion.jsonl
    "mj:point3": "mj:point3:ir100",
    "mj:point4s": "mj:point4s:fit0.000996044x0.2498:ir100",
    "mj:spheres:s1:rs0.75:tr0.02": "mj:spheres:s1:rs0.75:ir100:tr0.02",
    "drake:hydro:rt0.01": "drake:hydro:E1e7:r1:rt0.01",
}
RATE = 2.0             # N/s
T_SETTLE = 0.4
V_DETECT = 0.010       # m/s, gross-slip detection
V_FLOOR = 0.0002       # m/s, onset margin above the creep law
T_POST = 0.080         # s run on after detection
T_HOLD = 1.0
OUT = ROOT / "docs/experiments/20261005-contact_bed/pull_slip.jsonl"


def new_rig(spec, dt):
    H.DT = dt                      # the rig reads its module DT for the MJCF timestep and its step count
    return H.make_rig(spec, d_cg=0.0, gravity=False)


def settle(rig, N):
    rig.set_pad_force(N)
    rig.set_tool_wrench(np.zeros(3), np.zeros(3))
    rig.step(T_SETTLE)
    st = rig.tool_state()
    c = rig.contacts()
    return st["pos"].copy(), st["axis"].copy(), {"N_L": c["L"]["N"], "N_R": c["R"]["N"],
                                                 "n_L": c["L"]["n"], "n_R": c["R"]["n"]}


def drive(rig, dt, p0, a0, force_of_t, stop, t_max):
    """Apply F(t) along a0 each step; returns arrays t, F, u (axial displacement), v, step wall times."""
    t0 = rig.t
    T, F, U, V, W = [], [], [], [], []
    u_prev = 0.0
    n = int(round(t_max / dt))
    for k in range(n):
        tt = (k + 1) * dt
        f = force_of_t(tt)
        rig.set_tool_wrench(f * a0, np.zeros(3))
        w0 = time.perf_counter()
        rig.step(dt)
        W.append(time.perf_counter() - w0)
        p = rig.tool_state()["pos"]
        u = float((p - p0) @ a0)
        v = (u - u_prev) / dt
        u_prev = u
        T.append(tt), F.append(f), U.append(u), V.append(v)
        if not math.isfinite(u) or abs(u) > 0.045 or stop(tt, u, v):
            break
    return np.array(T), np.array(F), np.array(U), np.array(V), np.array(W), rig.t - t0


def smooth_v_at(T, V, t, half=0.010):
    m = np.abs(T - t) <= half
    return float(V[m].mean()) if m.any() else float("nan")


def run_case(chain, N, dt_ms):
    spec = MODELS[chain]
    dt = dt_ms * 1e-3
    t_wall = time.time()
    row = {"exp": "pull_slip", "chain_spec": chain, "spec": spec, "N": N, "dt_ms": dt_ms, "rate_N_s": RATE,
           "m_tool_kg": H.M_TOOL, "mu": H.MU, "gravity": False}

    # ---- ramp to gross slip
    rig = new_rig(spec, dt)
    p0, a0, c0 = settle(rig, N)
    row["settle"] = c0
    row["info"] = getattr(rig, "info", {})
    det = {}

    def stop(tt, u, v):
        if "t" not in det and v > V_DETECT:
            det["t"] = tt
        return "t" in det and tt > det["t"] + T_POST
    t_max = 2.5 * 2 * H.MU * N / RATE + 0.3
    T, F, U, V, W, _ = drive(rig, dt, p0, a0, lambda tt: RATE * tt, stop, t_max)
    row["us_per_step_median"] = float(np.median(W) * 1e6)
    row["us_per_step_mean"] = float(np.mean(W) * 1e6)
    row["n_steps_ramp"] = int(len(W))
    if "t" not in det:
        row.update(slipped=False, F_max=float(F[-1]), u_end_mm=float(U[-1] * 1e3))
    else:
        td = det["t"]
        i_d = int(np.searchsorted(T, td - 1e-9))
        # creep law v = c F fitted on the ramp between 30 and 80 % of the detection force; onset is the
        # first step of the run that ends at detection in which v stays above 2 c F + V_FLOOR
        mc = (F >= 0.3 * F[i_d]) & (F <= 0.8 * F[i_d])
        c = float((V[mc] * F[mc]).sum() / max((F[mc] ** 2).sum(), 1e-12)) if mc.any() else 0.0
        i = i_d
        while i > 0 and V[i - 1] > 2 * max(c, 0.0) * F[i - 1] + V_FLOOR:
            i -= 1
        j = max(i - 1, 0)                         # last step on the creep law
        t0, F_on, u_pre = float(T[j]), float(F[j]), float(U[j])
        # slip: mean speed over the first 50 ms; sliding friction from a quadratic fit of the
        # displacement over 5-60 ms after onset (robust to stick-slip oscillation)
        u50 = float(np.interp(t0 + 0.050, T, U))
        mw = (T >= t0 + 0.005) & (T <= t0 + 0.060)
        if mw.sum() >= 4:
            acc = 2 * np.polyfit(T[mw], U[mw], 2)[0]
            F_slide = float(F[mw].mean()) - H.M_TOOL * acc
        else:
            F_slide = float("nan")
        vw = V[(T > t0) & (T <= t0 + 0.050)]
        row.update(slipped=True, t_detect=td, F_detect=RATE * td, creep_c_mm_s_per_N=c * 1e3,
                   t_onset=t0, F_onset=F_on, mu_eff=F_on / (2 * N), u_pre_mm=u_pre * 1e3,
                   v_at_half_onset_mm_s=smooth_v_at(T, V, 0.5 * t0) * 1e3,
                   v_at_090_onset_mm_s=smooth_v_at(T, V, 0.9 * t0) * 1e3,
                   v_slip_mean50_mm_s=(u50 - u_pre) / 0.050 * 1e3,
                   v_slip_mean50_coulomb_mm_s=RATE * 0.050 ** 2 / (6 * H.M_TOOL) * 1e3,
                   v_slip_cv50=float(vw.std() / max(abs(vw.mean()), 1e-12)) if len(vw) > 2 else float("nan"),
                   mu_slide=F_slide / (2 * N))
    stride = max(1, int(round(0.005 / dt)))
    row["ramp_cols"] = ["t", "F", "u_mm", "v_mm_s"]
    row["ramp"] = [[round(float(a), 4), round(float(b), 4), float(c * 1e3), float(d * 1e3)]
                   for a, b, c, d in zip(T[::stride], F[::stride], U[::stride], V[::stride])]
    del rig

    # ---- hold at 50 % of onset
    if row.get("slipped"):
        F_h = 0.5 * row["F_onset"]
        rig = new_rig(spec, dt)
        p0, a0, _ = settle(rig, N)
        t_r = F_h / RATE
        T, F, U, V, W2, _ = drive(rig, dt, p0, a0, lambda tt: min(RATE * tt, F_h), lambda *a: False,
                                  t_r + T_HOLD)
        mh = T >= t_r + 0.2
        slope = np.polyfit(T[mh], U[mh], 1)[0] if mh.sum() > 3 else float("nan")
        u_hs = float(np.interp(t_r, T, U))
        row.update(F_hold=F_h, creep_mm_s=float(slope * 1e3), hold_disp_mm=float((U[-1] - u_hs) * 1e3),
                   u_at_hold_start_mm=u_hs * 1e3)
        stride = max(1, int(round(0.010 / dt)))
        row["hold_cols"] = ["t", "F", "u_mm"]
        row["hold"] = [[round(float(a), 4), round(float(b), 4), float(c * 1e3)]
                       for a, b, c in zip(T[::stride], F[::stride], U[::stride])]
        W = np.concatenate([W, W2])
        row["us_per_step_median"] = float(np.median(W) * 1e6)
        row["us_per_step_mean"] = float(np.mean(W) * 1e6)
    row["wall_s"] = time.time() - t_wall
    row["when"] = time.strftime("%Y-%m-%d %H:%M")
    return row


def done(path):
    have = set()
    if Path(path).exists():
        for line in open(path):
            try:
                r = json.loads(line)
                have.add((r["chain_spec"], r["N"], r["dt_ms"]))
            except Exception:
                pass
    return have


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--N", nargs="+", type=float, default=[0.5, 1.0, 3.0])
    ap.add_argument("--dt", nargs="+", type=float, default=[1.0, 5.0], help="physics step, ms")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    have = done(a.out)
    for model in a.models:
        for dt in a.dt:
            for N in a.N:
                if (model, N, dt) in have:
                    continue
                r = run_case(model, N, dt)
                H.append_row(a.out, r)
                keys = ("mu_eff", "u_pre_mm", "creep_mm_s", "v_slip_mean50_mm_s", "v_slip_cv50", "mu_slide", "us_per_step_median",
                        "wall_s")
                print(model, N, dt, {k: round(r[k], 4) for k in keys if k in r}, flush=True)


if __name__ == "__main__":
    main()
