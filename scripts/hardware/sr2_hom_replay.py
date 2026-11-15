#!/usr/bin/env python3
"""Open-loop replay of recorded HOM trials of the SR2 hand in the CPU MuJoCo plant of scripts/hom/hom_tasks.py, and a
fit of that plant's parameters to the recorded joint angles and object poses. The owner ruled on 2026-10-06 that the
plant is fitted to servo-level and trajectory quantities, never to task outcomes such as turn angles or drops.

Trials are the files of sr2hand.control.trial_score (a header line, then one sample per 20 ms period in the stream
format of hom_bench.simulate). The plant is the inner-loop benchmark's (scripts/hom/inner_loop_plant.py): the CAD hand
with the v4b tip (SCALE_HAND=cad_v2), the SCS0009 servo plant (HOS_PLANT=servo) and the 2.5 MPa pads of the scene cache
logs/20261102-hom_servo_plant/scenes. Each object has one scene, the cell of the reference layout (92.5 x 102.5 mm,
hom_gpu.REF) built by hand_object_scale_sim.cell_scenes, the builder under hom_tasks.build and HomBench. A trial's
mounts move the three mount bodies of the compiled model as hom_gpu.Scene does, and its palm_m places {P}. The
parameters are set on the compiled model, so every candidate replays on one compiled scene:
  kp            servo gain, N m/rad               actuator gainprm and biasprm
  damping       finger joint damping, N m s/rad   dof_damping
  tau_max       servo torque limit, N m           actuator forcerange
  deadband_deg  servo dead band, deg              above 0 the finger actuators become motors with torque kp (target - q)
                                                  outside the band and none inside it, within tau_max, set at every
                                                  physics step (hom_torque.DeadbandPlant, hom_gpu.servo_deadband)
  frictionloss  finger joint friction, N m        dof_frictionloss
  mu            contact friction                  every geom's sliding friction, as the scene builder sets MU
  E_MPa         pad modulus, MPa                  the pads' solref time constant pad_tr(E) / 2. Each pad's solimp d0
                                                  is the same at every modulus (reorient_backends.pad_tr), and a scene
                                                  rebuilt at 5 MPa (logs/20261102-hom_servo_plant/scenes_E5) differs
                                                  from the 2.5 MPa scene in that time constant only, so no rebuild is
                                                  needed
The defaults are the scene's plant (kp 0.72, damping 0.0144, tau_max 0.18, no dead band or joint friction), mu 1.0 and
E 2.5 MPa.

Replay. The plant starts at rest from the first sample's joint angles and object pose, receives each sample's targets at
the sample's time plus the header's target_delay_s (default 0), and loses its support post when the schedule's lift
begins (the first sample past 'squeeze'; the header's post_off_s overrides it, a negative value removes the post from
the start). Up to the header's hold_s (s) the object is put back at rest at its start pose after every period
(hom_tasks.PLANT_VARIANTS['hold'], used by the cylinder tasks' runs). The plant's state is compared with the record at
the samples' times (a sample's t_joints and t_obj when it has them): joint RMS over the nine joints (deg), object
position RMS in {P} (mm) and orientation RMS (deg), over every sample after the first. The first sample's object pose
must be tracked.

Fit. CMA-ES (pycma) over the chosen parameters in coordinates scaled to [0, 1] (log scale for kp, damping, tau_max, mu
and E_MPa) within BOUNDS, from the default plant. It minimizes the training trials' mean cost, joint RMS / 1 deg +
position RMS / 1 mm + orientation RMS / 1 deg (--weights), and reports the same cost on held-out trials (every third
trial in name order unless --test names them). Each generation's candidates are replayed by a pool of worker processes.

  .venv/bin/python scripts/hardware/sr2_hom_replay.py synth --task twist --d 40 --call 202 --grasp 12 --seeds 0 1 \\
        --set kp=0.55 mu=0.8 --out logs/20261115-sr2_hom_replay/trials
  .venv/bin/python scripts/hardware/sr2_hom_replay.py score TRIAL...
  .venv/bin/python scripts/hardware/sr2_hom_replay.py replay TRIAL... [--truth | --set kp=0.6 mu=0.9]
  .venv/bin/python scripts/hardware/sr2_hom_replay.py fit TRIAL... --workers 6 --popsize 10 --gens 30 --out FIT.json
synth makes trials with hom_bench.simulate on a layout and grasp of the toy's grid (fp_sr2_toy rows grid2dff_*_v4b,
the layouts of hom_gpu --layouts grid2dff_size) at a chosen plant; its header records the plant and hom_tasks.metrics'
scores of the run. Every command that loads a scene runs under ~/.claude/bin/resguard.sh run (one worker is about
0.5 GB per object scene).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""                 # CPU MuJoCo only
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")    # no huge-page compaction stalls on this workstation

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import inner_loop_plant  # noqa: E402,F401  SCALE_HAND, HOS_PLANT, HOS_SCENES, HOM_ROWS_TAG before the scene modules
import mujoco  # noqa: E402
import fp_hom_check as C  # noqa: E402
import fp_sr2_toy as FP  # noqa: E402
import hand_object_scale_sim as S  # noqa: E402
import hom_tasks as T  # noqa: E402
import reorient_backends as RB  # noqa: E402
from sr2hand.control import hom_bench as HB  # noqa: E402
from sr2hand.control import trial_score as TS  # noqa: E402
from scipy.spatial.transform import Slerp  # noqa: E402
from sr2hand.geometry.rotations import from_scipy, quat_angle, quat_mul, quat_yaw, to_scipy  # noqa: E402

FINGERS = T.FINGERS
OUT = ROOT / "logs/20261115-sr2_hom_replay"
TOY_ROWS = C.ROWS                                     # docs/experiments/20261029-foundational_pose_review/rows
REF = (92.5, 102.5)                                   # hom_gpu.REF: the layout whose cell scenes are compiled
POSE = {"roll": "tripod", "twist": "tripod", "pinpush": "pinch", "press": "press"}
PAD_RE = re.compile(r"(thumb|index|middle)_pad\d+")
PRE_LIFT = ("open", "close", "squeeze", "no object")  # phases before the post is removed ('no object': hom_server.py
                                                      # before the first tracked pose)
_P0 = RB.plant_args(S.PLANT)
DEFAULT = {"kp": _P0["kp"], "damping": _P0["damping"], "tau_max": _P0["forcerange"], "deadband_deg": 0.0,
           "frictionloss": _P0["frictionloss"], "mu": S.MU, "E_MPa": RB.E_TPU / 1e6}
# (low, high, log scale) of each parameter in the fit. kp: the SR2 sysid's 0.55-0.72 N m/rad inside; damping: time
# constants damping / kp of 4-110 ms at kp 0.72; tau_max: the SCS0009's 0.18 N m inside; dead band: the measured
# 0.59 deg inside; mu: TPU on bare PLA (0.37-0.44 on the UR5) up to grip tape; E: the UR5 fit's 2.47 MPa inside.
BOUNDS = {"kp": (0.3, 2.0, True), "damping": (0.003, 0.08, True), "tau_max": (0.08, 0.5, True),
          "deadband_deg": (0.0, 1.5, False), "frictionloss": (0.0, 0.02, False), "mu": (0.3, 2.5, True),
          "E_MPa": (1.0, 10.0, True)}


# ------------------------------------------------------------------------------------------------ scenes and plant

def ref_scene(shape: str, d_mm: float) -> tuple[Path, Path]:
    """The reference layout's pads and pt scenes for one object (from the cache, built on first use)."""
    sc = S.cell_scenes(dict(C.cell_of(*REF)), shape, float(d_mm), models=("pads", "pt"))
    return Path(sc["pads"][0]), Path(sc["pt"][0])


def ref_xy() -> dict:
    """Palm-frame (x, y) mounts (m) of the reference layout, whose scenes are compiled."""
    M = FP.mounts6(FP.layout2(*REF))
    return {f: np.asarray(M[f][:2], float) for f in FINGERS}


def move_mounts(m, mounts_xy: dict | None, base: dict) -> None:
    """The mount bodies at a layout's (x, y) (m), from their compiled positions less the reference layout's."""
    if mounts_xy is None:
        return
    for f in FINGERS:
        m.body_pos[m.body(f"{f}_mount").id] = base[f] + np.array([mounts_xy[f][0], mounts_xy[f][1], 0.0])


def mount_base(m) -> dict:
    return {f: m.body_pos[m.body(f"{f}_mount").id].copy() - np.array([*ref_xy()[f], 0.0]) for f in FINGERS}


class SR2Plant(T.Plant):
    """hom_tasks.Plant on the reference cell's pads scene with a layout's mounts and a plant parameter set (module
    docstring). With rec set to a list, every advance appends (time, finger joint angles, object qpos)."""

    def __init__(self, path, setup, mounts_xy=None, params=None, hold_s=0.0):
        super().__init__(path, setup)
        m = self.m
        self.acts = np.concatenate([self.act[f] for f in FINGERS])
        self.qadr9 = np.concatenate([self.qadr[f] for f in FINGERS])
        self.vadr9 = np.concatenate([self.vadr[f] for f in FINGERS])
        self.pads = np.array([g for g in range(m.ngeom) if PAD_RE.fullmatch(m.geom(g).name or "")], int)
        self.base = mount_base(m)
        self.ctrlrange0 = m.actuator_ctrlrange[self.acts].copy()
        self.mass0 = (float(m.body_mass[self.tb]), m.body_inertia[self.tb].copy())
        self.gravity0 = m.opt.gravity.copy()
        self.hold_s = hold_s
        self.target = None
        self.rec = None
        move_mounts(m, mounts_xy, self.base)
        self.set_params(params or {})

    def set_params(self, params: dict) -> None:
        p = {**DEFAULT, **params}
        m, mj = self.m, self.mj
        self.kp, self.tau, self.band = float(p["kp"]), float(p["tau_max"]), math.radians(p["deadband_deg"])
        m.dof_damping[self.vadr9] = p["damping"]
        m.dof_frictionloss[self.vadr9] = p["frictionloss"]
        m.geom_friction[:, 0] = p["mu"]
        m.geom_solref[self.pads, 0] = float(f"{RB.pad_tr(p['E_MPa'] * 1e6) / 2:.6g}")   # as replace_tips writes it
        for k, a in enumerate(self.acts):
            m.actuator_gaintype[a] = mj.mjtGain.mjGAIN_FIXED
            m.actuator_gainprm[a, :] = 0.0
            m.actuator_biasprm[a, :] = 0.0
            m.actuator_forcelimited[a] = 1
            m.actuator_forcerange[a] = (-self.tau, self.tau)
            if self.band > 0:                            # a motor; advance sets its torque
                m.actuator_gainprm[a, 0] = 1.0
                m.actuator_biastype[a] = mj.mjtBias.mjBIAS_NONE
                m.actuator_ctrlrange[a] = (-self.tau, self.tau)
            else:                                        # the position servo: kp (ctrl - q), kv 0
                m.actuator_gainprm[a, 0] = self.kp
                m.actuator_biastype[a] = mj.mjtBias.mjBIAS_AFFINE
                m.actuator_biasprm[a, 1] = -self.kp
                m.actuator_ctrlrange[a] = self.ctrlrange0[k]
        self.params = p

    def set_object(self, mass_kg=None, gravity=None) -> None:
        """The object's mass (inertia scaled with it) and gravity in {P}; None restores the scene's."""
        m = self.m
        k = 1.0 if mass_kg is None else float(mass_kg) / self.mass0[0]
        m.body_mass[self.tb] = self.mass0[0] * k
        m.body_inertia[self.tb] = self.mass0[1] * k
        m.opt.gravity[:] = self.gravity0 if gravity is None else np.asarray(gravity, float)

    def reset(self, q0, seed):
        super().reset(q0, seed)
        self._start(q0)

    def reset_to(self, q0: dict, pos, quat) -> None:
        """At rest at the given finger joint angles (rad) and object pose (model world frame)."""
        m, d = self.m, self.d
        self.mj.mj_resetData(m, d)
        m.geom_contype[self.post], m.geom_conaffinity[self.post] = self.post0
        d.qpos[self.qa:self.qa + 3] = pos
        d.qpos[self.qa + 3:self.qa + 7] = quat
        for f in FINGERS:
            d.qpos[self.qadr[f]] = q0[f]
            d.ctrl[self.act[f]] = q0[f]
        self.obj0 = d.qpos[self.qa:self.qa + 7].copy()
        self._start(q0)
        self.mj.mj_forward(m, d)

    def _start(self, q0):
        self.target = np.concatenate([np.asarray(q0[f], float) for f in FINGERS])
        if self.band > 0:
            self.d.ctrl[self.acts] = 0.0

    def set_targets(self, q_cmd):
        self.target = np.concatenate([np.asarray(q_cmd[f], float) for f in FINGERS])
        if self.band <= 0:
            super().set_targets(q_cmd)

    def advance(self, t1: float):
        """hom_tasks.Plant.advance, with the dead band's torques at every physics step and the record."""
        m, d, mj = self.m, self.d, self.mj
        while d.time < t1 - 1e-9:
            if self.band > 0:
                e = self.target - d.qpos[self.qadr9]
                d.ctrl[self.acts] = np.clip(np.where(np.abs(e) > self.band, self.kp * e, 0.0), -self.tau, self.tau)
            mj.mj_step(m, d)
        if self.hold_s and d.time <= self.hold_s + 1e-9:
            d.qpos[self.qa:self.qa + 7] = self.obj0
            d.qvel[self.va:self.va + 6] = 0.0
            mj.mj_forward(m, d)
        if self.rec is not None:
            self.record()

    def record(self):
        d = self.d
        self.rec.append((float(d.time), d.qpos[self.qadr9].copy(), d.qpos[self.qa:self.qa + 7].copy()))


class _Tasks:
    """hom_tasks for hom_bench.simulate with its Plant replaced (bench.T is read for Plant, Schedule and metrics)."""

    def __init__(self, plant_factory):
        self.Plant = plant_factory

    def __getattr__(self, k):
        return getattr(T, k)


# ------------------------------------------------------------------------------------------------ synthetic trials

def toy_setup(task: str, d_mm: float, call: int, grasp: int = 0, src: str = "grid2dff"):
    """(hom_tasks setup, mounts (x, y) in m, layout record) of one layout and grasp of the toy's rows for this object,
    as hom_gpu.size_layouts, tripod_setup and pinch_setup build them (hom_gpu imports warp; this script does not)."""
    shape = T.SHAPE[task]
    FP.set_sizes(**({"sph_d_mm": d_mm} if shape == "sphere" else {"cyl_d_mm": d_mm}))
    name = FP.size_name(src, shape, d_mm)
    rows = (json.loads(x) for x in (TOY_ROWS / f"{name}.jsonl").read_text().splitlines() if x.strip())
    row = next(r for r in rows if int(r["call"]) == int(call))
    rec = row["fp"].get(POSE[task]) or {}
    if not rec.get("ok"):
        raise ValueError(f"{name} call {call}: no {POSE[task]} pose")
    g = (rec.get("grasps") or [rec])[grasp]
    M = FP.mounts6(row["d"])
    v = np.asarray(g["v"], float)
    if shape == "sphere":
        obj, _tg, n, _pc, _ax = FP.geometry("tripod", v, M)
        c = np.asarray(obj["centre"], float)
        n_out = {f: np.asarray(n[f], float) for f in FINGERS}
        r = d_mm / 2000.0
        setup = {"task": task, "shape": "sphere", "d_mm": d_mm, "r": r, "hl": 0.0,
                 "palm": np.array([-c[0], -c[1], S.Z_OBJ - c[2]]), "quat": np.array(T.SPHERE_Q), "design": {},
                 "targets": {f: c + (r + T.K.R) * n_out[f] for f in FINGERS}, "n_out": n_out, "mounts": M, "psi": 0.0}
    else:
        obj, tg, n, _pc, _ax = FP.geometry("pinch", v, M)
        c, psi = np.asarray(obj["centre"], float), float(v[3])
        setup = {"task": task, "shape": "cylinder", "d_mm": FP.CYL_D * 1000, "r": FP.CYL_D / 2, "hl": FP.CYL_L / 2,
                 "palm": np.array([-c[0], -c[1], S.Z_OBJ - c[2]]), "quat": quat_mul(quat_yaw(psi), T.SPHERE_Q),
                 "design": {}, "targets": tg, "n_out": {f: np.asarray(n[f], float) for f in FINGERS}, "mounts": M,
                 "psi": psi}
    mxy = {f: np.asarray(M[f][:2], float) for f in FINGERS}
    layout = {"rows": name, "call": int(call), "grasp": int(grasp), "d6_mm": [float(x) for x in row["d"]]}
    return setup, mxy, layout


def toy_bench(task, d_mm, call, grasp, dt=HB.DT_BENCH):
    """HomBench (controller, Mirror and schedule of hom_tasks) on a toy layout and grasp: the reference scenes with the
    mounts moved, as hom_gpu.Scene.prepare and hom_torque.run set them up."""
    setup, mxy, layout = toy_setup(task, d_mm, call, grasp)
    p_pads, p_pt = ref_scene(setup["shape"], d_mm)
    m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
    og = int(np.flatnonzero(m_pt.geom_bodyid == m_pt.body(S.OBJ).id)[0])
    want = [setup["r"]] if setup["shape"] == "sphere" else [setup["r"], setup["hl"]]
    if not np.allclose(m_pt.geom_size[og][:len(want)], want, atol=1e-6):
        raise RuntimeError(f"object size {m_pt.geom_size[og]} is not {want}")
    move_mounts(m_pt, mxy, mount_base(m_pt))
    q_open, _ = T.open_pose(setup, m_pt)
    S.set_palm(m_pt, setup["palm"])
    return HB.HomBench(task, d_mm, setup, m_pt, q_open, dt=dt, p_pads=p_pads), mxy, layout


def header_of(bench, mxy, layout, params, seed, hold_s):
    s = bench.setup
    return {"task": bench.task, "d_mm": bench.d_mm, "shape": s["shape"], "seed": seed, "dt": bench.prm.dt,
            "hand": T.K.HAND, "scene_plant": S.PLANT, "scenes": str(S.SCENES.relative_to(ROOT)), "ref_layout_mm": REF,
            "mounts_mm": {f: [round(1e3 * float(x), 4) for x in mxy[f]] for f in FINGERS},
            "palm_m": [float(x) for x in s["palm"]], "layout": layout,
            "grasp": {"centre_P_mm": [round(-1e3 * float(x), 4) for x in s["palm"][:2]] +
                      [round(1e3 * float(S.Z_OBJ - s["palm"][2]), 4)],
                      "n_out": {f: [round(float(x), 6) for x in s["n_out"][f]] for f in FINGERS},
                      "psi": float(s.get("psi", 0.0))},
            "plant": {k: float(v) for k, v in {**DEFAULT, **params}.items()}, "hold_s": hold_s, "prm": {},
            "object_mass_kg": S.MASS, "gravity_P": [0.0, 0.0, -9.81], "target_delay_s": 0.0}


def cmd_synth(a) -> int:
    import hom_landscape as L
    params = parse_set(a.set)
    out = Path(a.out)
    for seed in a.seeds:
        bench, mxy, layout = toy_bench(a.task, a.d, a.call, a.grasp)
        hold_s = T.T_CLOSE if (a.hold is None and T.SHAPE[a.task] == "cylinder") else float(a.hold or 0.0)
        bench.T = _Tasks(lambda path, setup: SR2Plant(path, setup, mxy, params, hold_s))
        w0 = time.perf_counter()
        stream, res = HB.simulate(bench, seed)
        wall = time.perf_counter() - w0
        h = header_of(bench, mxy, layout, params, seed, hold_s)
        h["sim"] = {**{k: res.get(k) for k in ("held", "dropped", "fingers_end", "track_err_rms_deg",
                                               "track_ref_rms_deg", "contact_all_frac", "metrics_dt")},
                    "J": L.track_score(a.task, res)}
        h["made"] = {"by": "scripts/hardware/sr2_hom_replay.py synth", "when": time.strftime("%Y-%m-%d %H:%M"),
                     "wall_s": round(wall, 2)}
        tag = "_".join(f"{k}{v:g}" for k, v in sorted(params.items())) or "default"
        path = out / f"{a.task}_d{a.d:g}_c{a.call}g{a.grasp}_{tag}_s{seed}.jsonl"
        TS.write_trial(path, h, stream)
        print(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: {len(stream)} samples, "
              f"{wall:.1f} s, J {h['sim']['J']:.4f}, held {res.get('held')}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ replay

def _sample_time(s, key_ms):
    """A sample's measurement time on the clock of t (s): t_joints or t_obj, or hom_server.py's ms stamp against the
    state's t_ms, else t."""
    key = {"t_joints_ms": "t_joints", "t_capture_ms": "t_obj"}[key_ms]
    if s.get(key) is not None:
        return float(s[key])
    if s.get(key_ms) is not None and s.get("t_ms") is not None and math.isfinite(float(s[key_ms])):
        return float(s["t"]) + 1e-3 * (float(s[key_ms]) - float(s["t_ms"]))
    return float(s["t"])


def prepare_trial(header: dict, stream: list[dict]) -> dict:
    """A trial's arrays for replay: times from the first sample (s), recorded joints (sim deg, n x 9), object pose in
    {P} (mm, wxyz), targets (sim rad per finger), the post's removal and the measurement times. In a record of
    hom_server.py, joints whose source was not the servo bus (joints_src) and object poses of a tracker frame already
    used by an earlier sample are left out of the comparison, and the replay starts at the first sample with both
    measured."""
    def measured(s):
        return (s.get("joints_src", "bus") == "bus" and s.get("obj_pos_mm") is not None
                and s.get("obj_quat") is not None and np.isfinite(np.asarray(s["obj_pos_mm"], float)).all()
                and np.isfinite([float(v) for f in FINGERS for v in s["joints"][f].values()]).all())
    k0 = next((k for k, s in enumerate(stream) if measured(s)), None)
    if k0 is None:
        raise ValueError("no sample with both the joints and the object's pose measured")
    stream = stream[k0:]
    t = np.array([float(s["t"]) for s in stream])
    t0 = t[0]
    q = np.array([np.concatenate([HB.sim_from_servo(s["joints"])[f] for f in FINGERS]) for s in stream])
    q[[s.get("joints_src", "bus") != "bus" for s in stream]] = np.nan
    pos = np.array([s.get("obj_pos_mm") or [np.nan] * 3 for s in stream], float)
    quat = np.array([s.get("obj_quat") or [np.nan] * 4 for s in stream], float)
    cap = [s.get("t_capture_ms") for s in stream]
    for k in range(1, len(stream)):
        if cap[k] is not None and cap[k] == cap[k - 1]:
            pos[k] = quat[k] = np.nan
    if header.get("post_off_s") is not None:
        post_k = int(np.searchsorted(t - t0, float(header["post_off_s"]) - 1e-9))
    else:
        post_k = next((k for k, s in enumerate(stream) if s.get("phase") not in PRE_LIFT), len(stream))
        if all("phase" not in s for s in stream):
            post_k = int(np.searchsorted(t - t0, T.T_LIFT - 1e-9))
    palm = np.asarray(header.get("palm_m") or (np.array([0.0, 0.0, S.Z_OBJ]) - 1e-3 * pos[0]), float)
    mxy = header.get("mounts_mm")
    return {"header": header, "task": header["task"], "shape": T.SHAPE[header["task"]], "d_mm": float(header["d_mm"]),
            "t": t - t0, "q_deg": np.degrees(q), "pos_mm": pos, "quat": quat,
            "t_joints": np.array([_sample_time(s, "t_joints_ms") for s in stream]) - t0,
            "t_obj": np.array([_sample_time(s, "t_capture_ms") for s in stream]) - t0,
            "targets": [HB.sim_from_servo(s["targets"]) for s in stream], "post_k": post_k,
            "q0": HB.sim_from_servo(stream[0]["joints"]), "palm": palm,
            "mounts_xy": None if mxy is None else {f: 1e-3 * np.asarray(mxy[f], float) for f in FINGERS},
            "hold_s": float(header.get("hold_s") or 0.0), "delay": float(header.get("target_delay_s") or 0.0),
            "mass": header.get("object_mass_kg"), "gravity": header.get("gravity_P")}


class Replayer:
    """One SR2Plant per object scene, reused across replays."""

    def __init__(self):
        self.plants = {}

    def plant(self, tr) -> SR2Plant:
        key = (tr["shape"], tr["d_mm"])
        if key not in self.plants:
            p_pads, _ = ref_scene(*key)
            self.plants[key] = SR2Plant(p_pads, {"palm": tr["palm"], "quat": np.array(T.SPHERE_Q)})
        return self.plants[key]

    def replay(self, tr: dict, params: dict) -> dict:
        return replay(self.plant(tr), tr, params)


def replay(plant: SR2Plant, tr: dict, params: dict) -> dict:
    """The plant driven open loop by a trial's targets: its joints (sim deg), object position in {P} (mm) and
    quaternion at the samples' times, and the errors against the record (module docstring)."""
    m, d = plant.m, plant.d
    plant.set_params(params)
    move_mounts(m, tr["mounts_xy"] or ref_xy(), plant.base)
    plant.set_object(tr["mass"], tr["gravity"])
    S.set_palm(m, tr["palm"])
    plant.hold_s = tr["hold_s"]
    t = tr["t"]
    if not (np.isfinite(tr["pos_mm"][0]).all() and np.isfinite(tr["quat"][0]).all()):
        raise ValueError("the first sample's object pose is not tracked: no start state for the replay")
    plant.reset_to(tr["q0"], tr["palm"] + 1e-3 * tr["pos_mm"][0], tr["quat"][0])
    bad0 = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
    plant.rec = []
    plant.record()
    w0 = time.perf_counter()
    for k in range(len(t)):
        ta = t[k] + tr["delay"]
        if ta > d.time + 1e-9:
            plant.advance(ta)
        if k == tr["post_k"]:
            plant.post_off()
        plant.set_targets(tr["targets"][k])
    if t[-1] > d.time + 1e-9:                            # with a target delay, the record must reach the last sample
        plant.advance(t[-1])
    wall = time.perf_counter() - w0
    rt = np.array([r[0] for r in plant.rec])
    rq = np.degrees(np.array([r[1] for r in plant.rec]))
    rx = np.array([r[2] for r in plant.rec])
    plant.rec = None
    diverged = bool(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > bad0 or not np.isfinite(rx).all())
    out = {"wall_s": wall, "diverged": diverged, "n": len(t)}
    if diverged:
        return {**out, "joint_rms_deg": None, "pos_rms_mm": None, "rot_rms_deg": None}
    tj, to = np.clip(tr["t_joints"], rt[0], rt[-1]), np.clip(tr["t_obj"], rt[0], rt[-1])
    q_sim = np.stack([np.interp(tj, rt, rq[:, i]) for i in range(9)], 1)
    p_sim = np.stack([np.interp(to, rt, rx[:, i]) for i in range(3)], 1)
    p_sim = 1e3 * (p_sim - tr["palm"])
    quat_sim = from_scipy(Slerp(rt, to_scipy(rx[:, 3:7]))(to))
    ok = np.isfinite(tr["quat"]).all(1)                  # untracked samples are left out
    er = np.full(len(t), np.nan)
    er[ok] = np.degrees(quat_angle(quat_sim[ok], tr["quat"][ok]))
    sel = slice(1, None)                                 # the first sample is the initial condition
    eq = (q_sim - tr["q_deg"])[sel]
    ep = np.linalg.norm(p_sim - tr["pos_mm"], axis=1)[sel]
    er = er[sel]
    return {**out, "joint_rms_deg": float(np.sqrt(np.nanmean(eq ** 2))),
            "pos_rms_mm": float(np.sqrt(np.nanmean(ep ** 2))), "rot_rms_deg": float(np.sqrt(np.nanmean(er ** 2))),
            "joint_max_deg": float(np.nanmax(np.abs(eq))), "pos_max_mm": float(np.nanmax(ep)),
            "rot_max_deg": float(np.nanmax(er)), "q_sim_deg": q_sim, "pos_sim_mm": p_sim, "quat_sim": quat_sim}


def header_params(header: dict) -> dict:
    """The plant a trial ran on as replay parameters: the header's plant dict (synth), or a plant spec string
    (hom_server.py's plant_spec or plant, as reorient_backends.plant_args reads it)."""
    p = header.get("plant")
    if isinstance(p, dict):
        return {k: float(v) for k, v in p.items() if k in DEFAULT}
    spec = header.get("plant_spec") or p
    if not spec:
        return {}
    spec = {"servo": S.SERVO_PLANT, "work": S.WORK_PLANT, "measured": "kp0.55_kv0_fr0.18_fl0_dp0.011"}.get(spec, spec)
    a = RB.plant_args(spec)
    return {"kp": a["kp"], "damping": a.get("damping", DEFAULT["damping"]), "tau_max": a["forcerange"],
            "frictionloss": a.get("frictionloss", 0.0)}


def cost_of(r: dict, weights=(1.0, 1.0, 1.0), penalty=1e3) -> float:
    """joint RMS / w_q (deg) + position RMS / w_p (mm) + orientation RMS / w_r (deg); penalty for a diverged replay."""
    if r.get("diverged") or r.get("joint_rms_deg") is None:
        return penalty
    return r["joint_rms_deg"] / weights[0] + r["pos_rms_mm"] / weights[1] + r["rot_rms_deg"] / weights[2]


def parse_set(items) -> dict:
    out = {}
    for kv in items or []:
        k, v = kv.split("=")
        if k not in DEFAULT:
            raise SystemExit(f"unknown parameter {k}; one of {', '.join(DEFAULT)}")
        out[k] = float(v)
    return out


def _brief(r):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
            if not isinstance(v, np.ndarray)}


def cmd_score(a) -> int:
    for p in a.trials:
        header, stream = TS.load_trial(p)
        sc = TS.score_stream(stream, header, f_min=a.f_min)
        sim = header.get("sim") or {}
        line = {k: sc.get(k) for k in ("held", "dropped", "fingers_end", "track_err_rms_deg", "track_ref_rms_deg",
                                        "contact_all_frac", "contact_all_frac_025", "J", "J_025", "J_nc")}
        if "J" in sim:
            line["J_sim"] = sim["J"]
            line["dJ"] = round(sc["J"] - sim["J"], 4) if sc["J"] is not None else None
            line["sim"] = {k: sim.get(k) for k in ("track_err_rms_deg", "track_ref_rms_deg", "contact_all_frac",
                                                   "fingers_end")}
        print(Path(p).name, json.dumps(line), flush=True)
    return 0


def cmd_replay(a) -> int:
    rep = Replayer()
    params = parse_set(a.set)
    rows = []
    for p in a.trials:
        header, stream = TS.load_trial(p)
        tr = prepare_trial(header, stream)
        prm = {**header_params(header), **params} if a.truth else params
        r = rep.replay(tr, prm)
        row = {"trial": Path(p).name, "params": {**DEFAULT, **prm}, **_brief(r), "cost": cost_of(r)}
        rows.append(row)
        print(row["trial"], json.dumps({k: row[k] for k in row if k not in ("trial", "params")}), flush=True)
        if a.out_dir:
            o = Path(a.out_dir) / (Path(p).stem + "_replay.jsonl")
            o.parent.mkdir(parents=True, exist_ok=True)
            with open(o, "w") as fh:
                fh.write(json.dumps({"header": {"replay_of": str(p), "params": row["params"]}}) + "\n")
                for k in range(len(tr["t"])):
                    q = np.radians(r["q_sim_deg"][k]).reshape(3, 3)
                    fh.write(json.dumps({"t": float(tr["t"][k]),
                                         "joints": HB.servo_from_sim({f: q[i] for i, f in enumerate(FINGERS)}),
                                         "obj_pos_mm": r["pos_sim_mm"][k].tolist(),
                                         "obj_quat": r["quat_sim"][k].tolist()}) + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ fit

_W = {}


def _init_worker(paths, weights):
    _W.update(rep=Replayer(), weights=weights,
              trials={p: prepare_trial(*TS.load_trial(p)) for p in paths})


def _job(job):
    ci, path, params = job
    try:
        r = _W["rep"].replay(_W["trials"][path], params)
        r = _brief(r)
    except Exception as e:                               # noqa: BLE001  a failed replay costs the penalty
        r = {"diverged": True, "error": f"{type(e).__name__}: {e}"}
    r["cost"] = cost_of(r, _W["weights"])
    return ci, path, r


def to_unit(name, v):
    lo, hi, lg = BOUNDS[name]
    return (math.log(v / lo) / math.log(hi / lo)) if lg else (v - lo) / (hi - lo)


def from_unit(name, x):
    lo, hi, lg = BOUNDS[name]
    x = min(1.0, max(0.0, float(x)))
    return lo * (hi / lo) ** x if lg else lo + (hi - lo) * x


def split(paths, test=None):
    """(train, test): the named test trials, or every third trial in name order."""
    paths = sorted(paths, key=lambda p: Path(p).name)
    if test:
        te = [p for p in paths if any(Path(p).name == Path(t).name for t in test)]
    else:
        te = paths[2::3]
    return [p for p in paths if p not in te], te


def cmd_fit(a) -> int:
    import multiprocessing as mp

    import cma
    names = a.params
    for n in names:
        if n not in BOUNDS:
            raise SystemExit(f"unknown parameter {n}")
    for kv in a.bound or []:
        k, v = kv.split("=")
        lo, hi = map(float, v.split(","))
        BOUNDS[k] = (lo, hi, BOUNDS[k][2])
    fixed = parse_set(a.set)
    train, test = split(a.trials, a.test)
    paths = train + test
    heads = {p: TS.load_trial(p)[0] for p in paths}
    truths = [h.get("plant") if isinstance(h.get("plant"), dict) else None for h in heads.values()]
    truth = truths[0] if truths[0] is not None and all(x == truths[0] for x in truths) else None
    dur = {p: len(TS.load_trial(p)[1]) for p in paths}
    weights = tuple(a.weights)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".jsonl")
    if a.start == "header":                              # the plant the first training trial ran on
        prior = header_params(heads[train[0]])
    else:
        prior = json.loads(Path(a.start).read_text())["fitted"] if a.start else {}
    start = {**DEFAULT, **prior, **fixed}
    x0 = [to_unit(n, start[n]) for n in names]

    def params_of(x):
        return {**fixed, **{n: from_unit(n, xi) for n, xi in zip(names, x)}}

    def append(row):
        with open(log, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    t_start = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init_worker, initargs=(paths, weights)) as pool:
        def evaluate(cands, which):
            jobs = sorted([(ci, p, c) for ci, c in enumerate(cands) for p in which], key=lambda j: -dur[j[1]])
            per = [dict() for _ in cands]
            for ci, p, r in pool.imap_unordered(_job, jobs, chunksize=1):
                per[ci][Path(p).name] = r
            return [float(np.mean([r["cost"] for r in d_.values()])) for d_ in per], per

        print(f"plant {S.PLANT}, scenes {S.SCENES.relative_to(ROOT)}; fitting {names} on {len(train)} trials, "
              f"{len(test)} held out; truth {truth}", flush=True)
        (c0_tr,), (per0_tr,) = evaluate([params_of(x0)], train)
        (c0_te,), (per0_te,) = evaluate([params_of(x0)], test) if test else ((None,), ({},))
        append({"gen": 0, "start": params_of(x0), "train_cost": c0_tr, "test_cost": c0_te, "train": per0_tr,
                "test": per0_te, "wall_s": round(time.perf_counter() - t_start, 1)})
        print(f"start: train cost {c0_tr:.3f}, test cost {c0_te}", flush=True)
        es = cma.CMAEvolutionStrategy(x0, a.sigma0, {"bounds": [[0.0] * len(names), [1.0] * len(names)],
                                                     "popsize": a.popsize, "seed": a.seed, "verbose": -9})
        best = (c0_tr, params_of(x0))
        hist = []
        gen = 0
        while not es.stop() and gen < a.gens and time.perf_counter() - t_start < 60 * a.minutes:
            gen += 1
            X = es.ask()
            cands = [params_of(x) for x in X]
            costs, per = evaluate(cands, train)
            es.tell(X, costs)
            i = int(np.argmin(costs))
            if costs[i] < best[0]:
                best = (costs[i], cands[i])
            row = {"gen": gen, "best_cost": best[0], "gen_best": costs[i], "gen_median": float(np.median(costs)),
                   "best": best[1], "sigma": float(es.sigma), "wall_s": round(time.perf_counter() - t_start, 1)}
            hist.append(row)
            append(row)
            print(f"gen {gen:3d}  best {best[0]:.4f}  gen best {costs[i]:.4f}  median {np.median(costs):.4f}  "
                  f"sigma {es.sigma:.3f}  {row['wall_s']:.0f} s  " +
                  " ".join(f"{n} {best[1][n]:.4g}" for n in names), flush=True)
        (cb_tr,), (perb_tr,) = evaluate([best[1]], train)
        (cb_te,), (perb_te,) = evaluate([best[1]], test) if test else ((None,), ({},))
        if truth is not None:
            (ct_tr,), _ = evaluate([truth], train)
            (ct_te,), _ = evaluate([truth], test) if test else ((None,), None)
        else:
            ct_tr = ct_te = None
    wall = time.perf_counter() - t_start
    res = {"trials": {"train": [Path(p).name for p in train], "test": [Path(p).name for p in test]},
           "params": names, "fixed": fixed, "bounds": {n: BOUNDS[n] for n in names}, "weights": weights,
           "popsize": es.popsize, "sigma0": a.sigma0, "workers": a.workers, "generations": gen,
           "evaluations": gen * es.popsize * len(train), "stop": {k: str(v) for k, v in es.stop().items()},
           "start": params_of(x0), "fitted": best[1], "truth": truth,
           "cost": {"train_start": c0_tr, "test_start": c0_te, "train_fitted": cb_tr, "test_fitted": cb_te,
                    "train_truth": ct_tr, "test_truth": ct_te},
           "per_trial": {"start": {**per0_tr, **per0_te}, "fitted": {**perb_tr, **perb_te}},
           "wall_s": round(wall, 1), "plant": S.PLANT, "scenes": str(S.SCENES.relative_to(ROOT)),
           "when": time.strftime("%Y-%m-%d %H:%M")}
    if truth is not None:
        res["error"] = {n: {"fitted": best[1][n], "truth": truth[n], "start": params_of(x0)[n],
                            "rel_err": (best[1][n] - truth[n]) / truth[n] if truth[n] else None,
                            "abs_err": best[1][n] - truth[n]} for n in names}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("fitted", "truth", "cost", "wall_s", "generations")}, indent=1), flush=True)
    if truth is not None:
        for n in names:
            e = res["error"][n]
            print(f"  {n:13s} start {e['start']:.4g}  fitted {e['fitted']:.4g}  truth {e['truth']:.4g}"
                  + (f"  ({100 * e['rel_err']:+.1f} %)" if e["rel_err"] is not None else f"  ({e['abs_err']:+.3g})"))
    return 0


# ------------------------------------------------------------------------------------------------ CLI

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("synth", help="trials made by hom_bench.simulate at a chosen plant")
    p.add_argument("--task", required=True, choices=["pinpush", "press", "roll", "twist"])
    p.add_argument("--d", type=float, required=True, help="object diameter, mm")
    p.add_argument("--call", type=int, required=True, help="layout index of the toy's grid rows")
    p.add_argument("--grasp", type=int, default=0, help="index in the layout's grasp set")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--set", nargs="*", default=[], help="plant parameters k=v (default the scene's plant)")
    p.add_argument("--hold", type=float, default=None, help="hold_s; default 0.8 s for the cylinder tasks, else 0")
    p.add_argument("--out", default=str(OUT / "trials"))
    p = sub.add_parser("score", help="trial_score.score_stream of each trial, beside the simulator's J when known")
    p.add_argument("trials", nargs="+")
    p.add_argument("--f-min", type=float, default=TS.F_MIN)
    p = sub.add_parser("replay", help="each trial replayed open loop at one parameter set")
    p.add_argument("trials", nargs="+")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--truth", action="store_true",
                   help="each trial at the plant its header names (synth's plant, or hom_server.py's plant_spec)")
    p.add_argument("--out-dir", default=None, help="write each replayed trajectory as <trial>_replay.jsonl")
    p.add_argument("--json", default=None, help="write the rows to this file")
    p = sub.add_parser("fit", help="CMA-ES over plant parameters on the trials' joint and object trajectories")
    p.add_argument("trials", nargs="+")
    p.add_argument("--params", nargs="+", default=list(BOUNDS), choices=list(BOUNDS))
    p.add_argument("--set", nargs="*", default=[], help="fixed values of the parameters not fitted")
    p.add_argument("--bound", nargs="*", default=[], help="name=lo,hi")
    p.add_argument("--test", nargs="*", default=None, help="held-out trial files (default every third)")
    p.add_argument("--weights", nargs=3, type=float, default=[1.0, 1.0, 1.0], metavar=("DEG", "MM", "DEG_ROT"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--popsize", type=int, default=10)
    p.add_argument("--sigma0", type=float, default=0.25)
    p.add_argument("--start", default=None,
                   help="start from an earlier fit's JSON (its fitted values), or 'header': the first training trial's "
                        "plant (hom_server.py's plant_spec, e.g. the measured 0.55 N m/rad)")
    p.add_argument("--gens", type=int, default=30)
    p.add_argument("--minutes", type=float, default=25.0, help="wall-time budget of the search")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=str(OUT / "fit.json"))
    a = ap.parse_args()
    return {"synth": cmd_synth, "score": cmd_score, "replay": cmd_replay, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
