#!/usr/bin/env python3
"""Sampling MPC on the four HOM tasks of hom_tasks.py: the baseline beside the HOM controller on the same CAD-model
hand, servo plant, pads, objects, grasps and object placements.

Every run starts as a HOM run: the open pose, the approach at v_sep = 20 mm/s, the commanded pad forces and the lift
(hom_tasks.Schedule and the HOM controller until t = 2.0 s). From the task phase on, the controller is either the HOM
controller (controller "hom", which reproduces the rows of hom_tasks.py and adds the command log) or pinned iCEM
(controller "mpc"): the planners of 2026-10-14 (hand_object_scale_planners.py, config icem_pin: 64 rollouts of 0.5 s,
two iterations of 32, three knots, knot noise 0.08 rad with AR(1) correlation 0.7 across knots, the two best elites
kept, the first knot pinned to the applied command), replanned every 10 ms (100 Hz, the HOM controller's rate) on the
plant's own pads model with mujoco.rollout; between replannings the servo targets follow the planned spline at the
1 ms physics step. The planner's model is the plant's, compiled again with one contact sensor per finger (net force
between the tip body, which carries the pads, and the object).

Task costs, per 10 ms sample over the horizon (the last sample weighted w_term = 10 times, plus w_du times the squared
knot-to-knot change of the targets, as in the 2026-10-14 planners). The references are hom_tasks.Schedule's, tabulated
at the controller's ticks when the task starts:
  pinpush, press  ((theta - theta_ref) / 2 deg)^2, theta the pinch angle (hom_tasks.pinch_angle of the cylinder's axis
           a, about palm x) since the task start and theta_ref Schedule.angle's command; ((psi - psi_0) / 5 deg)^2,
           psi = asin(a_x) the rotation off the pinch axis; per finger clip((0.2 N - F_f) / 0.2 N, 0, 1), F_f the
           finger's net force on the object
  roll     |w - w_ref|^2 / (10 deg/s)^2, w the object's angular velocity (world); |p - p_ref|^2 / (5 mm)^2, p_ref the
           task-start centre plus the integral of Schedule's v_ref (the centre rolling along the thumb); per finger
           clip((0.5 N - F_f) / 0.5 N, 0, 1)
  twist    as roll with p_ref the task-start centre
  all      10 [object more than 15 mm below its height at the task start]
Rows carry the fields hom_tasks.rollout writes (cop_obj_mm, theta_ref_deg, w_ref_dps, v_ref_mms), so hom_tasks.metrics
scores both arms, and, from the 10 ms samples of the task phase, the smoothness metrics of
hand_object_scale_mpc.smooth_metrics (command rate, command reversals, angular-velocity jitter, three-pad share).
Commits before 2026-10-22 scored the earlier task definitions (axial displacement for pin-push, wall force for press).

    .venv/bin/python scripts/hom_tasks_mpc.py run --tasks pinpush --d 25 --seeds 0 --controllers mpc
    .venv/bin/python scripts/hom_tasks_mpc.py run --controllers mpc hom --seeds 0 --shard 0/2
    .venv/bin/python scripts/hom_tasks_mpc.py run --tasks pinpush press --d 25 40 --seeds 0 1 2 --controllers hom mpc --film
Rows: docs/experiments/20261021-hom_framework/rows/mpc_<task>.jsonl, one fsynced line per rollout, resumable per
(task, size, seed, controller). Films (--film, seed 0): media/20261021-mpc_<task>_<object>.mp4 and, when the HOM run
of the same cell is filmed in the same call, media/20261021-hom_vs_mpc_<task>_<object>.mp4 with a poster .jpg.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import hand_object_scale_mpc as M  # noqa: E402
import hand_object_scale_sim as S  # noqa: E402
import hom_tasks as T  # noqa: E402
from hom_tasks import Plant, Schedule, build, metrics, open_pose  # noqa: E402
from morphohand.control import hom as H  # noqa: E402

FINGERS = S.FINGERS
T_TASK = T.T_LIFT + 0.4
PANELS = ROOT / "logs/20261021-hom_tasks/panels"
# pinned iCEM of 2026-10-14 (hand_object_scale_planners.BASE + CONFIGS["icem_pin"]) at 100 Hz replanning
MPC = {**M.DEFAULTS, **M.PLANNER_DEFAULTS,
       **dict(method="icem", iters=2, pin0=1.0, nsample=64, horizon=0.5, dt_plan=0.01, w_theta=1.0, sig_theta_deg=2.0,
              w_off=1.0, sig_off_deg=5.0, w_omega=1.0, sig_omega_dps=10.0, w_drift=1.0, sig_drift=0.005, w_contact=1.0,
              f_min_push=0.2, f_min_tripod=0.5, w_drop=10.0)}
INT_KEYS = ("nsample", "knots", "cost_every", "iters", "keep", "log_every")


def planner_model(plant: Plant, path: Path):
    """The plant's model compiled again from its XML with one contact sensor per finger (net force between the tip
    body and the object); the post never collides; the palm as on the plant."""
    mj = plant.mj
    root = ET.parse(path).getroot()
    sens = root.find("sensor")
    if sens is None:
        sens = ET.SubElement(root, "sensor")
    for f in FINGERS:
        ET.SubElement(sens, "contact", name=f"mpc_{f}", body1=f"{f}_tip", body2=S.OBJ, data="force", reduce="netforce",
                      num="1")
    tmp = path.with_name(f"{path.stem}.mpcsens.{os.getpid()}.xml")
    ET.ElementTree(root).write(tmp)
    try:
        m = mj.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()
    m.body_pos[m.body("palm_pose").id] = plant.m.body_pos[plant.m.body("palm_pose").id]
    post = m.geom("post").id
    m.geom_contype[post] = m.geom_conaffinity[post] = 0
    if m.nq != plant.m.nq or m.nv != plant.m.nv or m.nu != plant.m.nu:
        raise RuntimeError("the sensor model changed the state size")
    return m


class _PlantView:
    """The attributes of a hom_turn3.MjPlant that hand_object_scale_mpc.Planner reads, on a hom_tasks.Plant."""

    def __init__(self, plant: Plant):
        self.mj, self.m, self.d = plant.mj, plant.m, plant.d
        self.act = np.concatenate([plant.act[f] for f in FINGERS])
        self.qa, self.va, self.tb = plant.qa, plant.va, plant.tb


def quat_rotate(q, v):
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4) (w, x, y, z)."""
    w, r = q[..., :1], q[..., 1:]
    c = np.cross(r, v)
    return v + 2.0 * w * c + 2.0 * np.cross(r, c)


class TaskPlanner(M.Planner):
    """hand_object_scale_mpc.Planner (its sampling, iCEM update, pinning and shift) with the task costs of this
    module; `ref` holds the references at the cost samples of the current replanning."""

    def __init__(self, plant: Plant, path: Path, task: str, prm: dict, nthread: int):
        from mujoco import rollout
        view = _PlantView(plant)
        mj = plant.mj
        self.mj, self.plant, self.prm, self.task = mj, view, prm, task
        self.rng = np.random.default_rng(0)
        self.m = m = planner_model(plant, path)
        self.pool = rollout.Rollout(nthread=nthread)
        self.datas = [mj.MjData(m) for _ in range(nthread)]
        self.spec = mj.mjtState.mjSTATE_FULLPHYSICS
        self.x0 = np.empty(mj.mj_stateSize(m, self.spec))
        self.dt = float(m.opt.timestep)
        self.nh = int(round(prm["horizon"] / self.dt))
        self.knot_t = np.linspace(0.0, prm["horizon"], int(prm["knots"]))
        tk = np.arange(self.nh) * self.dt
        self.W = np.stack([np.interp(tk, self.knot_t, np.eye(len(self.knot_t))[k]) for k in range(len(self.knot_t))], 1)
        self.lo, self.hi = m.actuator_ctrlrange[view.act, 0], m.actuator_ctrlrange[view.act, 1]
        self.qa = 1 + int(view.qa)                 # FULLPHYSICS = [time, qpos, qvel, act, ...]
        self.wa = 1 + m.nq + int(view.va) + 3      # the free joint's angular velocity (body frame)
        self.ctrl_full = np.zeros((int(prm["nsample"]), self.nh, m.nu))
        self.ref = None
        self.start(0)

    def start(self, seed: int):
        """A new rollout: no nominal plan, no kept elites, the sampling stream of this seed."""
        self.nominal = self.t_nom = self.elites = None
        self.rng = np.random.default_rng([int(seed), 20261021])
        self.ctrl_full[:] = self.plant.d.ctrl

    def cost(self, states, samples, p_ref, z0, sens=None):
        prm, ref = self.prm, self.ref
        ce = prm["cost_every"]
        X = states[:, ce - 1::ce]
        q = X[..., self.qa:self.qa + 7]
        p, qw = q[..., :3], q[..., 3:]
        run = prm["w_drop"] * (p[..., 2] < ref["z0"] - 0.015)
        F = np.linalg.norm(sens[:, ce - 1::ce].reshape(sens.shape[0], -1, len(FINGERS), 3), axis=-1)
        if self.task in ("pinpush", "press"):
            w_, x, y, z_ = qw[..., 0], qw[..., 1], qw[..., 2], qw[..., 3]
            a = np.stack([2 * (x * z_ + w_ * y), 2 * (y * z_ - w_ * x), 1 - 2 * (x * x + y * y)], -1)   # body z
            th = np.degrees(np.arctan2(a[..., 2], -a[..., 1])) - ref["th0"]
            dth = (th - ref["theta_ref"] + 180.0) % 360.0 - 180.0
            run = run + prm["w_theta"] * (dth / prm["sig_theta_deg"]) ** 2
            psi = np.degrees(np.arcsin(np.clip(a[..., 0], -1.0, 1.0))) - ref["psi0"]
            run = run + prm["w_off"] * (psi / prm["sig_off_deg"]) ** 2
            run = run + prm["w_contact"] * np.clip((prm["f_min_push"] - F) / prm["f_min_push"], 0.0, 1.0).sum(-1)
        else:
            ww = quat_rotate(qw, X[..., self.wa:self.wa + 3])
            run = run + prm["w_omega"] * np.sum((ww - ref["w_ref"]) ** 2, -1) / math.radians(prm["sig_omega_dps"]) ** 2
            run = run + prm["w_drift"] * np.sum((p - ref["p_ref"]) ** 2, -1) / prm["sig_drift"] ** 2
            run = run + prm["w_contact"] * np.clip((prm["f_min_tripod"] - F) / prm["f_min_tripod"], 0.0, 1.0).sum(-1)
        c = run[:, :-1].sum(1) + prm["w_term"] * run[:, -1]
        return c + prm["w_du"] * np.sum(np.diff(samples, axis=1) ** 2, (1, 2))


# ------------------------------------------------------------------------------------------------ references

def ref_table(task: str, sch: Schedule, ticks: list, i0: int, s0: dict) -> dict:
    """hom_tasks.Schedule's references at the controller ticks ticks[i0:] (the task phase; the same float times as the
    control loop): the pinch angle's command (deg), the object's reference angular velocity (world, rad/s), and the
    centre's reference path, the task-start centre plus the left Riemann sum of v_ref (hom_tasks.metrics' path)."""
    still = {"quat": np.array([1.0, 0.0, 0.0, 0.0])}
    th, w, v = [], [], []
    for t in ticks[i0:]:
        _cmd, info = sch(t, still)
        th.append(info.get("theta_ref", 0.0))
        w.append(info.get("w_ref", np.zeros(3)))
        v.append(info.get("v_ref", np.zeros(3)))
    dt = T.DT_C
    V = np.array(v)
    R0 = H.quat_to_mat(s0["quat"])
    a0 = R0[:, 2]
    return {"z0": float(s0["pos"][2]), "th0": T.pinch_angle(a0), "psi0": math.degrees(math.asin(max(-1.0, min(1.0, a0[0])))),
            "theta": np.array(th), "w": np.array(w),
            "p": s0["pos"][None] + np.vstack([np.zeros(3), np.cumsum(V[:-1], 0) * dt])}


def horizon_ref(task: str, tab: dict, j: int, planner: TaskPlanner) -> dict:
    """The references at the planner's cost samples when it replans at task tick j: sample k lies (k + 1) * cost_every
    physics steps (10 ms) ahead, at tick j + k + 1; past the task's end the last value holds."""
    nk = planner.nh // planner.prm["cost_every"]
    idx = np.minimum(j + 1 + np.arange(nk), len(tab["theta"]) - 1)
    ref = {"z0": tab["z0"], "th0": tab["th0"], "psi0": tab["psi0"]}
    if task in ("pinpush", "press"):
        ref["theta_ref"] = tab["theta"][idx]
    else:
        ref["w_ref"] = tab["w"][idx]
        ref["p_ref"] = tab["p"][idx] if task == "roll" else np.broadcast_to(tab["p"][0], (nk, 3))
    return ref


# ------------------------------------------------------------------------------------------------ rollout

def task_row(task, s, info):
    """One 10 ms row with the fields hom_tasks.rollout writes and hom_tasks.metrics reads."""
    R = H.quat_to_mat(s["quat"])
    row = {"t": None, "phase": info["phase"], "pos_mm": (1e3 * s["pos"]).round(3).tolist(),
           "w_dps": np.degrees(s["w"]).round(2).tolist(), "v_mms": (1e3 * s["v"]).round(2).tolist(),
           "axis": R[:, 2].round(5).tolist(), "F": {f: round(s["cop"].get(f, (None, 0.0))[1], 3) for f in FINGERS},
           "F_link": {f: round(max(0.0, s["Ftot"][f] - s["cop"].get(f, (None, 0.0))[1]), 3) for f in FINGERS},
           "cop_obj_mm": {f: (1e3 * (R.T @ (p_ - s["pos"]))).round(3).tolist() for f, (p_, _F) in s["cop"].items()}}
    if "w_ref" in info:
        row["w_ref_dps"] = np.degrees(info["w_ref"]).round(2).tolist()
        row["v_ref_mms"] = (1e3 * info["v_ref"]).round(3).tolist()
    if "theta_ref" in info:
        row["theta_ref_deg"] = round(info["theta_ref"], 3)
    return row


def rollout(task, setup, plant, mirror, q_open, seed, controller, planner=None, film=None, label=""):
    """One task under HOM until the task phase and under `controller` ("hom" or "mpc") from it on."""
    prm = T.PRM
    hprm = H.HomParams(dt=T.DT_C)
    sch = Schedule(task, setup, prm)
    plant.reset(q_open, seed)
    ctl = H.HomController(mirror, hprm, healthy={f: np.asarray(q_open[f]) for f in FINGERS})
    act = np.concatenate([plant.act[f] for f in FINGERS])
    dt = T.DT_C
    ticks, tk = [], 0.0                          # the loop's tick times, accumulated as the loop accumulates them
    while tk < sch.t_end - 1e-9:
        ticks.append(tk)
        tk += dt
    t, s0, lifted, i, j0, tab = 0.0, None, False, 0, None, None
    rows, frames, log, costs = [], [], [], []
    n_plan, t_plan = 0, 0.0
    w0 = time.perf_counter()
    while t < sch.t_end - 1e-9:
        s = plant.state()
        if t >= T.T_LIFT and not lifted:
            plant.post_off()
            lifted = True
        if s0 is None and t >= T_TASK - 1e-9:
            s0, j0 = s, i
            if controller == "mpc":
                if abs(planner.prm["cost_every"] * planner.dt - dt) > 1e-12 or ticks[i] != t:
                    raise RuntimeError("the planner's cost samples are not the controller's ticks")
                tab = ref_table(task, sch, ticks, i, s0)
        cmd, info = sch(t, s)
        u_applied = plant.d.ctrl[act].copy()
        if t >= T_TASK - 1e-9:
            log.append((float(t), u_applied, s["w"].copy(), H.quat_to_mat(s["quat"]),
                        np.array([s["cop"].get(f, (None, 0.0))[1] for f in FINGERS])))
        if controller == "mpc" and t >= T_TASK - 1e-9:
            if n_plan == 0:
                planner.start(seed)
            j = i - j0
            if (abs(info.get("theta_ref", 0.0) - tab["theta"][j]) > 1e-9
                    or not np.allclose(info.get("w_ref", np.zeros(3)), tab["w"][j])):
                raise RuntimeError(f"reference mismatch at t {t:.2f}")
            planner.ref = horizon_ref(task, tab, j, planner)
            tp = time.perf_counter()
            u, c_best, _c_nom = planner.plan(float(plant.d.time), None, None)
            t_plan += time.perf_counter() - tp
            n_plan += 1
            costs.append(round(float(c_best), 3))
            for k in range(int(round(dt / planner.dt))):
                plant.d.ctrl[act] = u[k]
                plant.mj.mj_step(plant.m, plant.d)
            t += dt
            plant.advance(t)
        else:
            for f in info.get("reanchor", ()):
                ctl.reanchor(f, s["q"][f])
            meas = H.Measurement(q=s["q"], obj_pos=s["pos"], obj_quat=s["quat"], cop=dict(s["cop"]))
            q_cmd, _ci = ctl.step(meas, cmd)
            plant.set_targets(q_cmd)
            t += dt
            plant.advance(t)
        row = task_row(task, s, info)
        row["t"] = round(t - dt, 3)
        row["u"] = [round(float(v), 4) for v in u_applied]
        rows.append(row)
        if film is not None and (len(rows) - 1) % 4 == 0:
            frames.append(film.frame(plant, f"{label}{task}, {setup['shape']} {setup['d_mm']:g} mm, seed {seed}",
                                     f"{info['phase']:<16s} t {t:4.2f} s  " + T.film_line(task, s, info)))
        if not np.all(np.isfinite(plant.d.qpos)):
            raise FloatingPointError(f"non-finite state at t {t:.3f}")
        i += 1
    res = metrics(task, setup, rows, s0, plant.state())
    sm = M.smooth_metrics(log, S.MASS * 9.81, lambda R: 0.0, t_from=T_TASK)
    sm.pop("peak_rate", None)
    res.update(sm, wall_s=round(time.perf_counter() - w0, 2), n_ticks=len(rows))
    if controller == "mpc":
        res.update(plan_s=round(t_plan, 2), n_plan=n_plan, plan_ms_mean=round(1e3 * t_plan / max(1, n_plan), 1),
                   cost_med=float(np.median(costs)) if costs else None)
    return res, rows, frames


# ------------------------------------------------------------------------------------------------ CLI

def _done(path: Path) -> set:
    out = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok":
                out.add((r["task"], r["d_mm"], r["seed"], r["controller"]))
    return out


def _append(path: Path, row: dict) -> None:
    """One fsynced line, written in one call under an exclusive lock: parallel jobs append to the same rows file."""
    import fcntl
    data = (json.dumps(row) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


def tile(clips, stem):
    import reorient_backends_films as F
    F.W, F.H = 640, 480
    n = F.tile_stream(clips, T.MEDIA / f"{stem}.mp4", T.MEDIA / f"{stem}.jpg", cols=len(clips), crf=22)
    print(f"tiled {n} frames -> {stem}.mp4", flush=True)


def cmd_run(a) -> int:
    import mujoco
    prm = dict(MPC)
    for kv in a.set:
        k, v = kv.split("=")
        prm[k] = v if isinstance(prm.get(k), str) else float(v)
    for k in INT_KEYS:
        prm[k] = int(prm[k])
    rows_dir = Path(a.rows_dir) if a.rows_dir else T.ROWS
    rows_dir.mkdir(parents=True, exist_ok=True)
    T.MEDIA.mkdir(parents=True, exist_ok=True)
    PANELS.mkdir(parents=True, exist_ok=True)
    jobs = [(task, float(d), sd, c) for task in a.tasks for d in (a.d or T.SIZES[task]) for sd in a.seeds
            for c in a.controllers]
    k0, n0 = (int(v) for v in a.shard.split("/"))
    jobs = [j for i, j in enumerate(jobs) if i % n0 == k0]
    done = {task: _done(rows_dir / f"mpc_{task}.jsonl") for task in a.tasks}
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done[j[0]] or (a.film and j[2] == a.seeds[0])]
    print(f"{len(jobs)} runs in shard {a.shard}; mpc {prm}", flush=True)
    cell = None
    clips = {}
    for task, d_mm, seed, ctrl in jobs:
        if cell is None or cell[0] != (task, d_mm):
            try:
                p_pads, p_pt, setup = build(task, d_mm)
            except Exception as e:
                print(task, d_mm, "build failed:", e, flush=True)
                cell = None
                continue
            m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
            S.set_palm(m_pt, setup["palm"])
            fs, os_ = T.sr2_specs(setup)
            mirror = H.Mirror(m_pt, fs, os_)
            q_open, _ = open_pose(setup, mujoco.MjModel.from_xml_path(str(p_pt)))
            plant = Plant(p_pads, setup)
            cell = ((task, d_mm), setup, mirror, q_open, plant, p_pads, {})
        _, setup, mirror, q_open, plant, p_pads, planners = cell
        planner = None
        if ctrl == "mpc":
            if "mpc" not in planners:
                planners["mpc"] = TaskPlanner(plant, p_pads, task, prm, a.nthread)
            planner = planners["mpc"]
        film = T.Film(plant, setup) if (a.film and seed == a.seeds[0]) else None
        obj = f"{setup['shape']}{d_mm:g}"
        row = {"task": task, "shape": setup["shape"], "d_mm": d_mm, "seed": seed, "controller": ctrl, "variant": "base",
               "design": {k: setup["design"][k] for k in ("tag", "h", "score")},
               "prm": {k: (list(v) if isinstance(v, tuple) else v) for k, v in T.PRM.items()},
               "mpc": prm if ctrl == "mpc" else None, "nthread": a.nthread, "when": time.strftime("%Y-%m-%d %H:%M")}
        try:
            res, trace, frames = rollout(task, setup, plant, mirror, q_open, seed, ctrl, planner, film,
                                         "HOM: " if ctrl == "hom" else "MPC: ")
            row.update(res, status="ok")
            row["trace"] = trace[::2]
            if film is not None and frames:
                import hom_contact_rig as HR
                fp = (T.MEDIA / f"20261021-mpc_{task}_{obj}.mp4") if ctrl == "mpc" else \
                    (PANELS / f"20261021-hom_{task}_{obj}_panel.mp4")
                HR.write_mp4(frames, fp, fps=25)
                row["film"] = str(fp.relative_to(ROOT))
                clips.setdefault((task, obj), {})[ctrl] = fp
        except Exception as e:
            row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
        if film is not None:
            # close now: Renderer.close frees its GL context before its mjr context, so a renderer collected after the
            # next one exists deletes that one's buffers (the first MPC film of 21:58 came out black)
            film.r.close()
        if (task, d_mm, seed, ctrl) not in done[task] or film is not None:
            _append(rows_dir / f"mpc_{task}.jsonl", row)
        brief = {k: row.get(k) for k in ("held", "theta_rmse_deg", "gain_min", "off_axis_max_deg", "pusher_slip_max_mm",
                                         "w_rmse_ratio", "centre_err_max_mm", "ramp_three_pads_max_deg", "drift_max_mm",
                                         "cmd_rate_rms", "cmd_reversals", "omega_jitter", "wall_s", "plan_ms_mean", "error")
                 if row.get(k) is not None}
        print(task, d_mm, ctrl, seed, brief, flush=True)
        cc = clips.get((task, obj), {})
        if "hom" in cc and "mpc" in cc:
            tile([cc["hom"], cc["mpc"]], f"20261021-hom_vs_mpc_{task}_{obj}")
            clips.pop((task, obj))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--tasks", nargs="+", default=list(T.TASKS))
    p.add_argument("--d", nargs="+", type=float)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--controllers", nargs="+", default=["mpc"], choices=["mpc", "hom"])
    p.add_argument("--nthread", type=int, default=6, help="rollout threads of the planner")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--film", action="store_true", help="film the first seed of each cell")
    p.add_argument("--set", nargs="*", default=[], help="planner and cost parameters k=v (MPC)")
    p.add_argument("--rows-dir", default=None, help="rows directory (default the study's rows/)")
    a = ap.parse_args()
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
