#!/usr/bin/env python3
"""Newton hydroelastic stability probe for the SR2 fingertip pinch (2026-10-05).

Three rigs, smallest first, each attempt appended as one fsynced JSON line to
docs/experiments/20261005-newton_probe/attempts.jsonl:

  sphere : one hydroelastic sphere (R 10.55 mm) resting on a fixed hydroelastic box
           under a constant body force, then a slow torque ramp about the normal
           (spin onset torque vs mu*sum(r_i f_i) from the contact set).
  repro  : Codex's SR2 pinch (distributed_contact_newton.setup), stepped one step at a
           time without graph capture so the first nonfinite step is located.
  pinch  : the same pinch held 0.5 s with no torque, then a slow twist ramp.

Run with logs/20261004-contact-transfer/venv/bin/python, WARP_CACHE_PATH=$(mktemp -d).
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.geometry import HydroelasticSDF

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/experiments/20261005-newton_probe/attempts.jsonl"
R_PAD = 0.01055


def log(row):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({k: row[k] for k in row if k not in ("trace",)}, default=float)[:900], flush=True)


def hydro_cfg(kh, voxel, mu, kf, density):
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = density
    cfg.mu = mu
    cfg.kf = kf
    cfg.mu_torsional = 0.0
    cfg.mu_rolling = 0.0
    cfg.margin = 0.0
    cfg.gap = 0.0
    cfg.is_hydroelastic = True
    cfg.kh = kh
    cfg.sdf_target_voxel_size = voxel
    cfg.sdf_narrow_band_range = (-0.003, 0.003)
    cfg.sdf_padding = 0.003
    return cfg


def make_solver(model, c):
    if c.get("solver", "mujoco") == "xpbd":
        return newton.solvers.SolverXPBD(model, iterations=c.get("iterations", 10))
    return newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        disable_sensors=True,
        njmax=c.get("njmax", 4096),
        nconmax=c.get("nconmax", 2048),
        iterations=c.get("iterations", 100),
        ls_iterations=c.get("ls_iterations", 50),
        cone=c.get("cone", "elliptic"),
        integrator=c.get("integrator", "implicitfast"),
        impratio=c.get("impratio", 1.0),
        tolerance=c.get("tolerance", 1e-8),
    )


@wp.kernel
def push(bodyf: wp.array[wp.spatial_vector], b: int, force: wp.vec3, torque: wp.vec3):
    bodyf[b] = wp.spatial_vector(force, torque)


def contact_set(model, state, cc, center, axis):
    """Newton-side pressure model: f_i = k_i * max(0, -d_i); torsion arm about axis."""
    n = int(cc.rigid_contact_count.numpy()[0])
    if n == 0:
        return dict(n=0, fn_sum=0.0, torsion_mu_sum=0.0, min_dist=0.0, k_max=0.0)
    dist = wp.zeros(cc.rigid_contact_max, dtype=float)
    pts = wp.zeros(cc.rigid_contact_max, dtype=wp.vec3)
    newton.eval_rigid_contact_kinematics(model, state, cc, out_distance=dist, out_point0_world=pts)
    d = dist.numpy()[:n]
    k = cc.rigid_contact_stiffness.numpy()[:n]
    f = k * np.maximum(0.0, -d)
    p = pts.numpy()[:n]
    nrm = cc.rigid_contact_normal.numpy()[:n]
    arm = np.cross(np.asarray(axis), p - np.asarray(center))
    vt = arm - (arm * nrm).sum(axis=1)[:, None] * nrm
    fr = cc.rigid_contact_friction.numpy()[:n]
    fr = np.where(fr > 0.0, fr, 1.0)
    return dict(
        n=n,
        fn_sum=float(f.sum()),
        fn_axis=float(np.sum(f * np.abs(nrm @ np.asarray(axis)))),
        torsion_sum=float(np.sum(f * np.linalg.norm(vt, axis=1))),
        torsion_mu_sum=float(np.sum(f * fr * np.linalg.norm(vt, axis=1))),
        min_dist=float(d.min()),
        k_max=float(k.max()),
        k_sum=float(k.sum()),
    )


def solver_forces(solver, P, u):
    """Contact forces the MuJoCo-Warp solve actually applied (mj_contactForce per contact)."""
    if not hasattr(solver, "mjw_data"):
        return {}
    import mujoco
    import mujoco_warp as mjw

    mjw.get_data_into(solver.mj_data, solver.mj_model, solver.mjw_data)
    m, d = solver.mj_model, solver.mj_data
    f6 = np.zeros(6)
    fn, tors, pads = 0.0, 0.0, {}
    for i, con in enumerate(d.contact[: d.ncon]):
        if con.efc_address < 0:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        n = np.array(con.frame[:3])
        arm = np.cross(np.asarray(u), np.array(con.pos) - np.asarray(P))
        vt = arm - (arm @ n) * n
        fn += f6[0]
        tors += con.friction[0] * f6[0] * np.linalg.norm(vt)
        key = m.geom(con.geom[1]).name.split("/")[-1]
        pads[key] = pads.get(key, 0.0) + float(f6[0])
    return dict(solver_fn_sum=float(fn), solver_torsion_mu_sum=float(tors), solver_fn_per_pad=pads)


# ---------------------------------------------------------------- rig 1: sphere on a box
def run_sphere(c):
    g = 9.81
    F = c.get("force", 3.0)
    kh_s, kh_b = c["kh"], c["kh"] * c.get("box_ratio", 100.0)
    dt = c["dt"]
    vol = 4.0 / 3.0 * np.pi * R_PAD**3
    mass = c.get("mass", 0.03)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    box = hydro_cfg(kh_b, c["voxel"], c.get("mu", 1.0), c.get("kf", 1000.0), 0.0)
    b.add_shape_box(body=-1, xform=wp.transform((0.0, 0.0, -0.005), wp.quat_identity()), hx=0.02, hy=0.02, hz=0.005, cfg=box)
    z0 = R_PAD - c.get("pen0", 0.0)
    body = b.add_body(xform=wp.transform((0.0, 0.0, z0), wp.quat_identity()), mass=mass,
                      inertia=wp.mat33(np.eye(3) * 0.4 * mass * R_PAD**2), lock_inertia=True)
    sph = hydro_cfg(kh_s, c["voxel"], c.get("mu", 1.0), c.get("kf", 1000.0), 0.0)
    b.add_shape_sphere(body, radius=R_PAD, cfg=sph)
    model = b.finalize(device="cuda:0")
    hc = HydroelasticSDF.Config(reduce_contacts=c.get("reduce", False), anchor_contact=True)
    pipe = newton.CollisionPipeline(model, reduce_contacts=c.get("reduce", False), rigid_contact_max=4096,
                                    broad_phase="explicit", sdf_hydroelastic_config=hc)
    solver = make_solver(model, c)
    s0, s1 = model.state(), model.state()
    ctrl = model.control()
    cc = pipe.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    settle = c.get("settle", 0.3)
    ramp_rate = c.get("torque_rate", 0.02)  # N m / s
    t_end = settle + c.get("ramp_time", 1.0)
    nsteps = int(round(t_end / dt))
    check = max(1, int(round(c.get("check_dt", 0.001) / dt)))
    trace, status, fail, onset = [], "complete", None, None
    elapsed = 0.0
    kh_eff = kh_s * kh_b / (kh_s + kh_b)
    delta = np.sqrt(F / (np.pi * kh_eff * R_PAD))
    a = np.sqrt(2 * R_PAD * delta)
    oracle_tau = c.get("mu", 1.0) * F * 8.0 / 15.0 * a
    static = None
    tick = time.perf_counter()
    for i in range(nsteps):
        t = i * dt
        tau = ramp_rate * (t - settle) if t > settle else 0.0
        s0.clear_forces()
        wp.launch(push, dim=1, inputs=[s0.body_f, body, wp.vec3(0.0, 0.0, -F), wp.vec3(0.0, 0.0, tau)])
        pipe.collide(s0, cc)
        solver.step(s0, s1, ctrl, cc, dt)
        s0, s1 = s1, s0
        if (i + 1) % check == 0 or i == nsteps - 1:
            wp.synchronize()
            elapsed += time.perf_counter() - tick
            q = s0.body_q.numpy()[body]
            v = s0.body_qd.numpy()[body]
            if not (np.isfinite(q).all() and np.isfinite(v).all()) or abs(q[2] - R_PAD) > 0.005 or np.abs(v).max() > 200:
                status, fail = "failed", dict(time=t + dt, phase="settle" if t < settle else "torque_ramp",
                                              q=q.tolist(), v=v.tolist())
                break
            row = dict(t=t + dt, z_pen_mm=(R_PAD - q[2]) * 1e3, wz=float(v[5]), vz=float(v[2]), tau=tau,
                       ncon=int(cc.rigid_contact_count.numpy()[0]))
            trace.append(row)
            if static is None and t + dt >= settle:
                static = contact_set(model, s0, cc, q[:3] - np.array([0, 0, R_PAD]), (0.0, 0.0, 1.0))
                static["pen_mm"] = row["z_pen_mm"]
                static["vz_end_settle"] = row["vz"]
            if t > settle and onset is None and abs(v[5]) > c.get("spin_w", 0.5):
                onset = dict(time=t + dt, torque=tau, wz=float(v[5]))
                break
            tick = time.perf_counter()
    simulated = len(trace) * check * dt
    return dict(
        status=status, failure=fail, static=static, onset=onset, oracle_winkler=dict(delta_mm=delta * 1e3, a_mm=a * 1e3, torsion=oracle_tau),
        us_per_step_nograph=elapsed / max(1, len(trace) * check) * 1e6, simulated_s=simulated,
        trace=trace[:: max(1, len(trace) // 40)],
    )


# ---------------------------------------------------------------- rig 2/3: Codex SR2 pinch
def pinch_setup(c, directory):
    sys.path.insert(0, str(ROOT / "scripts"))
    import distributed_contact_newton as D

    cfg = dict(voxel_mm=c.get("voxel_mm", 0.5), timestep=c["dt"], mapping=c.get("mapping", "native"),
               friction_gain=c.get("kf", 10.0), tool_stiffness_ratio=c.get("tool_ratio", 100.0),
               reduce_contacts=c.get("reduce", True), impratio=c.get("impratio", 100.0))
    if c.get("kh_pad") is not None or c.get("solver_override"):
        # Rebuild with our own kh and solver settings by patching after Codex's setup.
        pass
    model, solver, pipe, s0, s1, ctrl, cc, meta = D.setup(cfg, directory)
    return D, model, solver, pipe, s0, s1, ctrl, cc, meta


def run_pinch(c, directory):
    directory.mkdir(parents=True, exist_ok=True)
    import newton.solvers  # noqa

    sys.path.insert(0, str(ROOT / "scripts"))
    import distributed_contact_newton as D

    # Patch the shape kh and the solver constructor before D.setup builds them.
    kh_pad = c.get("kh_pad", 1e7 / R_PAD)
    orig_solver = newton.solvers.SolverMuJoCo
    overrides = {k: c[k] for k in ("iterations", "ls_iterations", "cone", "impratio", "integrator", "tolerance") if k in c}

    class Patched(orig_solver):
        def __init__(self, model, **kw):
            kw.update(overrides)
            super().__init__(model, **kw)

    D.newton.solvers.SolverMuJoCo = Patched
    orig_finalize = newton.ModelBuilder.finalize

    def finalize(self, *a, **kw):
        for i, label in enumerate(self.shape_label):
            name = label.split("/")[-1]
            if name in ("thumb_tipsphere", "index_tipsphere"):
                self.shape_material_kh[i] = kh_pad
            elif name == "tool":
                self.shape_material_kh[i] = kh_pad * c.get("tool_ratio", 100.0)
            if name in ("thumb_tipsphere", "index_tipsphere", "tool") and "margin" in c:
                self.shape_margin[i] = c["margin"]
                self.shape_gap[i] = c.get("gap", 0.0)
        return orig_finalize(self, *a, **kw)

    newton.ModelBuilder.finalize = finalize
    orig_add_mjcf = newton.ModelBuilder.add_mjcf

    def add_mjcf(self, xml, *a, **kw):
        if c.get("solref_fix", False):
            # MuJoCo reads solref="0.006" as (0.006, 1); Newton's importer pads it to (0.006, 0),
            # dampratio 0 -> k = 1/(tc^2 dr^2) = inf for any contact that falls back to geom solref.
            xml = xml.replace('solref="0.006"', 'solref="%s"' % c.get("fallback_solref", "0.006 1"))
        return orig_add_mjcf(self, xml, *a, **kw)

    newton.ModelBuilder.add_mjcf = add_mjcf
    try:
        cfg = dict(voxel_mm=c.get("voxel_mm", 0.5), timestep=c["dt"], mapping=c.get("mapping", "native"),
                   friction_gain=c.get("kf", 10.0), tool_stiffness_ratio=c.get("tool_ratio", 100.0),
                   reduce_contacts=c.get("reduce", True), impratio=c.get("impratio", 100.0))
        if c.get("static_squeeze_m"):
            cfg["static_squeeze_m"] = c["static_squeeze_m"]
        model, solver, pipe, s0, s1, ctrl, cc, meta = D.setup(cfg, directory)
    finally:
        newton.ModelBuilder.add_mjcf = orig_add_mjcf
        newton.ModelBuilder.finalize = orig_finalize
        D.newton.solvers.SolverMuJoCo = orig_solver
    if c.get("solver") == "xpbd":
        solver = newton.solvers.SolverXPBD(model, iterations=c.get("iterations", 10))
    dt = c["dt"]
    tool = meta["newton_tool"]
    P, u = np.array(meta["P"]), np.array(meta["u"])
    hold = c.get("hold", 0.5)
    rate = c.get("torque_rate", 0.0)
    t_end = hold + c.get("ramp_time", 0.0)
    nsteps = int(round(t_end / dt))
    check = max(1, int(round(c.get("check_dt", 0.0005) / dt)))
    first = c.get("first_steps", 40)
    trace, status, fail, onset, static = [], "complete", None, None, None
    elapsed = 0.0
    q_prev = None
    tick = time.perf_counter()
    for i in range(nsteps):
        t = i * dt
        tau = rate * (t - hold) if t > hold else 0.0
        s0.clear_forces()
        if c.get("center_spring", True):
            q = None
            wp.launch(D.drive, dim=1, inputs=[s0.body_q, s0.body_qd, s0.body_f, solver.mjw_data.time if hasattr(solver, "mjw_data") else wp.zeros(1, dtype=float), tool, wp.vec3(P), wp.vec3(u), 0.0])
        if tau:
            # add the twist torque on top of Codex's centring spring (drive overwrote body_f)
            bf = s0.body_f.numpy()
            bf[tool, 3:] += tau * u
            s0.body_f.assign(bf)
        pipe.collide(s0, cc)
        solver.step(s0, s1, ctrl, cc, dt)
        s0, s1 = s1, s0
        dense = c.get("dense", [0, first])
        in_dense = dense[0] <= i < dense[1]
        if in_dense or (i + 1) % check == 0 or i == nsteps - 1:
            wp.synchronize()
            elapsed += time.perf_counter() - tick
            qt = s0.body_q.numpy()[tool]
            vt = s0.body_qd.numpy()[tool]
            allq = s0.body_q.numpy()
            disp = float(np.linalg.norm(qt[:3] - P))
            row = dict(step=i + 1, t=t + dt, disp_mm=disp * 1e3, w_axis=float(vt[3:] @ u) if np.isfinite(vt).all() else None,
                       vmax=float(np.abs(vt).max()) if np.isfinite(vt).all() else None,
                       ncon=int(cc.rigid_contact_count.numpy()[0]), tau=tau)
            if in_dense:
                cs = contact_set(model, s0, cc, P, u)
                row.update(min_dist_mm=cs["min_dist"] * 1e3, k_max=cs["k_max"], k_sum=cs.get("k_sum", 0.0), fn_sum=cs["fn_sum"])
                if hasattr(solver, "mjw_data"):
                    md = solver.mjw_data
                    na = int(md.nacon.numpy()[0])
                    if na:
                        sr = md.contact.solref.numpy()[:na]
                        dd = md.contact.dist.numpy()[:na]
                        row.update(mj_ncon=na, solref_tc_min=float(sr[:, 0].min()), mj_dist_min_mm=float(dd.min() * 1e3),
                                   mj_dist_max_mm=float(dd.max() * 1e3), niter=int(md.solver_niter.numpy().max()))
                        try:
                            ef = md.efc.force.numpy()
                            row["efc_absmax"] = float(np.abs(ef).max())
                        except Exception:
                            pass
            trace.append(row)
            if not (np.isfinite(allq).all() and np.isfinite(vt).all()) or disp > 0.025 or np.abs(vt).max() > 2000:
                status = "failed"
                fail = dict(step=i + 1, time=t + dt, phase="hold" if t < hold else "twist", disp_mm=disp * 1e3,
                            finite=bool(np.isfinite(allq).all() and np.isfinite(vt).all()),
                            vmax=row["vmax"])
                break
            if static is None and t + dt >= hold:
                static = contact_set(model, s0, cc, P, u)
                static["disp_mm"] = disp * 1e3
                static.update(solver_forces(solver, P, u))
            if t > hold and onset is None and abs(row["w_axis"]) > c.get("spin_w", 0.5):
                onset = dict(time=t + dt, torque=tau, w=row["w_axis"])
                break
            tick = time.perf_counter()
    nmeas = (trace[-1]["step"] if trace else 0)
    timing = None
    if status == "complete" and c.get("timing_steps", 400):
        nt = c.get("timing_steps", 400)
        bf = s0.body_f.numpy(); bf[tool] = 0.0
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(nt):
            pipe.collide(s0, cc)
            solver.step(s0, s1, ctrl, cc, dt)
            s0, s1 = s1, s0
        wp.synchronize()
        timing = dict(us_per_step_nograph=(time.perf_counter() - t0) / nt * 1e6, steps=nt)
        try:
            with wp.ScopedCapture() as cap:
                for _ in range(20):
                    pipe.collide(s0, cc)
                    solver.step(s0, s1, ctrl, cc, dt)
                    s0, s1 = s1, s0
            wp.capture_launch(cap.graph)
            wp.synchronize()
            t0 = time.perf_counter()
            for _ in range(nt // 20):
                wp.capture_launch(cap.graph)
            wp.synchronize()
            timing["us_per_step_graph"] = (time.perf_counter() - t0) / (nt // 20 * 20) * 1e6
        except Exception as e:
            timing["graph_error"] = repr(e)[:200]
        qf = s0.body_q.numpy()[tool]
        timing["finite_after"] = bool(np.isfinite(qf).all())
    return dict(status=status, failure=fail, static=static, onset=onset, timing=timing,
                us_per_step_incl_readback=elapsed / max(1, nmeas) * 1e6, simulated_s=nmeas * dt,
                kh_pad=kh_pad, kh_tool=kh_pad * c.get("tool_ratio", 100.0),
                trace=[x for x in trace if "k_max" in x] + [x for x in trace if "k_max" not in x][:: max(1, len(trace) // 30)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rig", choices=["sphere", "pinch"])
    ap.add_argument("--cfg", action="append", default=[], help="JSON dict, repeatable")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    wp.init()
    for s in args.cfg:
        c = json.loads(s)
        t0 = time.perf_counter()
        row = dict(rig=args.rig, tag=args.tag, config=c, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
        try:
            if args.rig == "sphere":
                r = run_sphere(c)
            else:
                r = run_pinch(c, ROOT / "logs/20261005-newton_probe" / (args.tag or "pinch"))
            row.update(r)
        except Exception as e:
            row.update(status="error", error=repr(e), tb=traceback.format_exc()[-1500:])
        row["wall_s"] = time.perf_counter() - t0
        log(row)


if __name__ == "__main__":
    main()
