#!/usr/bin/env python3
"""CPU MuJoCo against MuJoCo-Warp on the 1 mm sphere-pad holding fixture (2026-10-05).

The fixture and contact mapping are those of the 'legacy' rows of scripts/gpu_scaling_mjwarp.py:
distributed_contact_transfer.build (SR2 thumb-index pinch, 1 mm pads, 1 ms or 0.5 ms step) with
gpu_scaling_mjwarp.apply_legacy (positive solref (tc, 1), solimp d0 = 1 - 1/(tc^2 K diag),
impratio 100), the mjlab MujocoCfg of those rows (implicitfast, elliptic, dense, Newton solver,
200 iterations, tolerance 1e-8, 50 line-search iterations, gravity off) and the shared guide:
20 N/m centring force, 2e-5 N m s rotational damping and, in the 'pulse' scenario, the two
0.012 N m raised-cosine torque pulses at 0.30-0.45 s and 0.45-0.60 s about the pinch axis.

Both backends run one world from the same initial state. CPU: mujoco.mj_step with the guide
set through distributed_contact_transfer.wrench. GPU: mjlab Simulation at nworld 1, the
distributed_contact_gpu.guide kernel and mjw.step captured as one CUDA graph per 10 ms block,
exactly as gpu_scaling_mjwarp.run_point. Every 10 ms both are read through the same metric
function on an MjData (the GPU world copied with mjw.get_data_into):

  normal force per pad (sum of contact normal forces), contact torque about the pinch axis u,
  torsion capacity mu * sum(r_i f_i) about the pressure-centre axis and about the pinch point P,
  tool slip relative to the fingertip pair (tool centre minus the tip-sphere midpoint, in the
  plane normal to u, from t = 0), its component along the tool axis, tool twist about u,
  tool displacement from P, active contacts, constraint rows and solver iterations.

The two backends differ in one documented way: MuJoCo-Warp 3.6 reads contact inverse weights
of the WELDED bodies (constraint.py, body_weldid before body_invweight0) and CPU 3.6 reads the
geom bodies; the fingertip is welded to its pip link, so the GPU pads see diag 41.80 against
43.14 on CPU and are 3.2 % stiffer per unit penetration under the legacy positive solref.

Rows: docs/experiments/20261005-gpu_scaling/cpu_gpu_consistency.jsonl (one per backend and
scenario, then one comparison row per scenario), fsynced as they land; existing keys skipped.
Run with the repo uv env: uv run --extra rl --extra gpu python scripts/pads_cpu_gpu_consistency.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from distributed_contact_transfer import build, command, wrench  # noqa: E402

OUT = ROOT / "docs/experiments/20261005-gpu_scaling/cpu_gpu_consistency.jsonl"
SCENARIOS = {
    "pulse_dt1": dict(timestep=0.001, torque_peak=0.012, duration=0.8),
    "pulse_dt0.5": dict(timestep=0.0005, torque_peak=0.012, duration=0.8),
    "hold_dt1": dict(timestep=0.001, torque_peak=0.0, duration=0.5),
}
SAMPLE_TIMES = (0.1, 0.3, 0.375, 0.45, 0.5, 0.525, 0.6, 0.8)
SPACING = 1.0


def git_rev():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def mujoco_cfg(m, dt):
    from mjlab.sim import MujocoCfg

    return MujocoCfg(timestep=dt, integrator="implicitfast", impratio=m.opt.impratio, cone="elliptic",
                     jacobian="dense", solver="newton", iterations=200, tolerance=1e-8, ls_iterations=50,
                     gravity=(0.0, 0.0, 0.0))


def fixture(sc):
    from gpu_scaling_mjwarp import apply_legacy

    m, d, p, meta = build(dict(spacing_mm=SPACING, timestep=sc["timestep"], policy="compiled",
                               torque_peak=sc["torque_peak"], duration=sc["duration"]))
    mapping = apply_legacy(m, SPACING)
    mujoco.mj_forward(m, d)
    return m, d, meta, mapping


def weights(m):
    out = {}
    for n in ("thumb_tip", "index_tip", "tool"):
        b = m.body(n).id
        w = int(m.body_weldid[b])
        out[n] = dict(weld_body=m.body(w).name, invweight0_geom_body=float(m.body_invweight0[b, 0]),
                      invweight0_weld_body=float(m.body_invweight0[w, 0]))
    out["diag_cpu_geom_bodies"] = out["thumb_tip"]["invweight0_geom_body"] + out["tool"]["invweight0_geom_body"]
    out["diag_gpu_weld_bodies"] = out["thumb_tip"]["invweight0_weld_body"] + out["tool"]["invweight0_weld_body"]
    out["predicted_gpu_over_cpu_stiffness"] = out["diag_cpu_geom_bodies"] / out["diag_gpu_weld_bodies"]
    return out


class Metric:
    def __init__(self, m, meta, d0):
        self.m, self.meta = m, meta
        self.P, self.u, self.a = (np.array(meta[k]) for k in ("P", "u", "a"))
        self.pad = np.full(m.ngeom, -1)
        for g in range(m.ngeom):
            n = m.geom(g).name
            if n.startswith("thumb_pad"):
                self.pad[g] = 0
            elif n.startswith("index_pad"):
                self.pad[g] = 1
        self.tool_geom = m.geom("tool").id
        self.bt, self.bi, self.bo = (m.body(n).id for n in ("thumb_tip", "index_tip", "tool"))
        self.s0 = self.rel(d0)
        self.q0 = d0.xquat[self.bo].copy()

    def rel(self, d):
        s = d.xpos[self.bo] - 0.5 * (d.xpos[self.bt] + d.xpos[self.bi])
        return s - (s @ self.u) * self.u

    def __call__(self, d):
        m, u, P = self.m, self.u, self.P
        f6 = np.zeros(6)
        fpad = [0.0, 0.0]
        pos, fn, mu, nrm = [], [], [], []
        torque = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if c.efc_address < 0:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            k = max(self.pad[c.geom[0]], self.pad[c.geom[1]])
            if k >= 0:
                fpad[k] += f6[0]
            frame = np.array(c.frame).reshape(3, 3)
            force = frame.T @ f6[:3]
            if c.geom[1] != self.tool_geom:  # force on the tool
                force = -force
            torque += float(np.cross(np.array(c.pos) - P, force) @ u)
            pos.append(np.array(c.pos))
            fn.append(f6[0])
            mu.append(c.friction[0])
            nrm.append(frame[0])
        pos, fn, mu, nrm = (np.array(x, dtype=float) for x in (pos, fn, mu, nrm))

        def capacity(center):
            if not len(fn):
                return 0.0
            arm = np.cross(u, pos - center)
            vt = arm - np.sum(arm * nrm, axis=1)[:, None] * nrm
            return float(np.sum(mu * fn * np.linalg.norm(vt, axis=1)))

        cen = (fn[:, None] * pos).sum(0) / fn.sum() if len(fn) and fn.sum() > 0 else P
        ds = self.rel(d) - self.s0
        q = np.zeros(4)
        mujoco.mju_negQuat(q, self.q0)
        r = np.zeros(4)
        mujoco.mju_mulQuat(r, d.xquat[self.bo], q)
        twist = np.degrees(2.0 * np.arctan2(r[1:] @ u, r[0]))
        return dict(
            time=float(d.time), fn_thumb_N=fpad[0], fn_index_N=fpad[1], contact_torque_Nm=torque,
            torsion_capacity_Nm=capacity(cen), torsion_capacity_about_P_Nm=capacity(P),
            slip_mm=float(np.linalg.norm(ds) * 1e3), slip_axial_mm=float(ds @ self.a * 1e3),
            tool_twist_deg=float(twist), tool_disp_mm=float(np.linalg.norm(d.xpos[self.bo] - P) * 1e3),
            contacts=int(len(fn)), contacts_loaded=int((fn > 0).sum()), nefc=int(d.nefc),
            solver_niter=int(d.solver_niter[0]),
        )


def run_cpu(sc):
    m, d, meta, mapping = fixture(sc)
    mujoco_cfg(m, sc["timestep"]).apply(m)  # the options mjlab applies before put_model
    mujoco.mj_forward(m, d)
    metric = Metric(m, meta, d)
    dt = m.opt.timestep
    every = round(0.01 / dt)
    cfg = dict(torque_peak=sc["torque_peak"])
    trace, ticks = [], []
    status = "complete"
    for k in range(round(sc["duration"] / dt)):
        wrench(m, d, meta, command(float(d.time), cfg))
        tick = time.perf_counter()
        mujoco.mj_step(m, d)
        ticks.append(time.perf_counter() - tick)
        if not np.isfinite(d.qpos).all():
            status = "failed"
            break
        if (k + 1) % every == 0:
            trace.append(metric(d))
    return dict(status=status, mapping=mapping, weights=weights(m), trace=trace,
                us_per_step_median=float(np.median(ticks) * 1e6), steps=len(ticks),
                opt=dict(timestep=dt, impratio=float(m.opt.impratio), cone=int(m.opt.cone),
                         integrator=int(m.opt.integrator), solver=int(m.opt.solver),
                         iterations=int(m.opt.iterations), tolerance=float(m.opt.tolerance),
                         ls_iterations=int(m.opt.ls_iterations), ls_tolerance=float(m.opt.ls_tolerance)))


def run_gpu(sc):
    import mujoco_warp as mjw
    import warp as wp
    from mjlab.sim import Simulation, SimulationCfg

    from distributed_contact_gpu import guide

    m, d, meta, mapping = fixture(sc)
    dt = m.opt.timestep
    cfg = SimulationCfg(nconmax=256, njmax=1024, mujoco=mujoco_cfg(m, dt))
    sim = Simulation(1, cfg, m, "cuda:0")  # applies cfg.mujoco to m in place
    wd, wm = sim.wp_data, sim.wp_model
    q0 = np.tile(d.qpos, (1, 1)).astype(np.float32)

    def reset():
        wd.qpos.assign(q0)
        wd.qvel.zero_()
        wd.qacc_warmstart.zero_()
        wd.time.zero_()
        sim.forward()
        wp.synchronize()

    wd.ctrl.assign(np.tile(d.ctrl, (1, 1)).astype(np.float32))
    reset()
    block = round(0.01 / dt)
    inp = [wd.qpos, wd.qvel, wd.time, wd.xfrc_applied, meta["tqa"], meta["tva"], meta["tool"],
           wp.vec3(meta["P"]), wp.vec3(meta["u"]), sc["torque_peak"]]
    wp.launch(guide, dim=1, inputs=inp)
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        for _ in range(block):
            wp.launch(guide, dim=1, inputs=inp)
            mjw.step(wm, wd)
    reset()
    dg = mujoco.MjData(m)
    mjw.get_data_into(dg, m, wd, world_id=0)
    metric = Metric(m, meta, dg)
    trace, ticks = [], []
    status = "complete"
    for _ in range(round(sc["duration"] / 0.01)):
        wp.synchronize()
        tick = time.perf_counter()
        wp.capture_launch(cap.graph)
        wp.synchronize()
        ticks.append(time.perf_counter() - tick)
        mjw.get_data_into(dg, m, wd, world_id=0)
        if not np.isfinite(dg.qpos).all():
            status = "failed"
            break
        trace.append(metric(dg))
    del cap, sim
    return dict(status=status, mapping=mapping, weights=weights(m), trace=trace,
                us_per_step_median=float(np.median(ticks) / block * 1e6), steps=block * len(ticks),
                versions=dict(warp=wp.__version__))


FIELDS = ("fn_thumb_N", "fn_index_N", "contact_torque_Nm", "torsion_capacity_Nm", "torsion_capacity_about_P_Nm",
          "slip_mm", "slip_axial_mm", "tool_twist_deg", "tool_disp_mm", "contacts")


def compare(cpu, gpu, sc):
    tc = {round(x["time"], 4): x for x in cpu["trace"]}
    tg = {round(x["time"], 4): x for x in gpu["trace"]}
    common = sorted(set(tc) & set(tg))
    at = {}
    for t in SAMPLE_TIMES:
        if t > sc["duration"] + 1e-9:
            continue
        k = min(common, key=lambda x: abs(x - t))
        at[f"{k:.3f}"] = {f: dict(cpu=tc[k][f], gpu=tg[k][f], diff=tg[k][f] - tc[k][f],
                                  rel=(tg[k][f] - tc[k][f]) / tc[k][f] if tc[k][f] else None) for f in FIELDS}
    maxdiff = {}
    for f in FIELDS:
        dif = np.array([tg[k][f] - tc[k][f] for k in common])
        ref = np.array([abs(tc[k][f]) for k in common])
        maxdiff[f] = dict(max_abs_diff=float(np.max(np.abs(dif))), at_time=float(common[int(np.argmax(np.abs(dif)))]),
                          cpu_range=[float(np.min([tc[k][f] for k in common])), float(np.max([tc[k][f] for k in common]))],
                          rms_diff=float(np.sqrt(np.mean(dif**2))), cpu_abs_mean=float(np.mean(ref)))
    return dict(samples=at, trace_max_diff=maxdiff, n_common=len(common))


def write(row, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--backends", nargs="+", default=["cpu", "gpu"], choices=["cpu", "gpu"])
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.out.read_text().splitlines()] if args.out.exists() else []
    have = {r["key"]: r for r in rows if r.get("status") == "complete"}
    base = dict(task="cpu_gpu_consistency", model="mj_pads1_legacy",
                rig_spec=f"mj:spheres:s{SPACING}:rs0.75:tr0.03:ir100 + apply_legacy (gpu_scaling_mjwarp 'legacy')",
                fixture="SR2 thumb-index pinch, distributed_contact_transfer.build; 20 N/m centring guide",
                nworld=1, script="scripts/pads_cpu_gpu_consistency.py", git_rev=git_rev(),
                mujoco=mujoco.__version__)
    for name in args.scenarios:
        sc = SCENARIOS[name]
        res = {}
        for backend, fn in (("cpu", run_cpu), ("gpu", run_gpu)):
            k = f"{name}_{backend}"
            if k in have:
                res[backend] = have[k]
                continue
            if backend not in args.backends:
                res[backend] = {}
                continue
            row = dict(base, key=k, scenario=name, backend=backend, dt_ms=sc["timestep"] * 1e3,
                       torque_peak_Nm=sc["torque_peak"], duration_s=sc["duration"])
            try:
                row.update(fn(sc))
            except Exception as e:  # noqa: BLE001
                row.update(status="error", error=repr(e)[:600], traceback=traceback.format_exc()[-2000:])
            write(row, args.out)
            res[backend] = row
            end = row.get("trace", [{}])[-1] if row.get("trace") else {}
            print(k, row["status"], {f: round(end.get(f, float("nan")), 5) for f in FIELDS}, row.get("error", ""),
                  flush=True)
        k = f"{name}_compare"
        if k in have or res["cpu"].get("status") != "complete" or res["gpu"].get("status") != "complete":
            continue
        row = dict(base, key=k, scenario=name, backend="compare", dt_ms=sc["timestep"] * 1e3,
                   torque_peak_Nm=sc["torque_peak"], duration_s=sc["duration"], status="complete",
                   weights=res["cpu"]["weights"], cpu_us_per_step=res["cpu"]["us_per_step_median"],
                   gpu_us_per_step_1world=res["gpu"]["us_per_step_median"])
        row.update(compare(res["cpu"], res["gpu"], sc))
        write(row, args.out)
        end = list(row["samples"].values())[-1]
        print(k, {f: (round(v["cpu"], 5), round(v["gpu"], 5)) for f, v in end.items()}, flush=True)


if __name__ == "__main__":
    main()
