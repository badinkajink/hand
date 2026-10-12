#!/usr/bin/env python3
"""The real_v1 D1 continuation after the supported grasp: scene, held state, plan, score, metrics and MJX glue.

Shared by the DiffMJX probes of 2026-10-07 (real_v1_diffmjx_*.py), the contact-gait search, transfer and Warp
replay (real_v1_contact_gait_*.py) and the audit of 2026-10-11 (diffmjx_audit_probes.py, diffmjx_turn_onset.py).
It imports no JAX at module level, so it loads in the project environment (MuJoCo 3.6) and in the isolated DiffMJX
environment of docs/experiments/20261007-diffmjx/README.md (MuJoCo 3.3.1, JAX 0.11.2).

Scene. reorient_backends.build_scene with the deployed D1 plan, the calibrated 'fast' plant and the bed numerics:
1 ms step, implicitfast, elliptic cone, Newton with 100 iterations, 50 line-search iterations and tolerance 1e-10,
impratio 100. The palm display plate never collides (MJX 3.6 has no cylinder-box routine; the plate has no contact
in this grasp). With contact_model 'padsT' the fingertip blocks become visual: the pads touch only the tool and the
blocks touch nothing (the blocks stay at least 28 mm apart in every 2026-10-07 plan, diffmjx_audit_probes.py).

Held state. The tool at the plan's pose with seed jitter, the grip pose held 0.8 s on the post (supported scene),
then cloned into the scene without the post.

Plan. 13 values in [-1, 1]: three yaw-target offsets (x 0.35 rad) for each of the phases 0-120, 120-240 and
240-400 ms; the index and middle opening amplitudes until 180 ms (0 at -1, the actuator's lower limit at +1); their
closing offsets (x 0.30 rad) after 180 ms. The forced-release variant fixes the index opening at +1.

Score. Final minus held vertical cosine, minus 70 per metre of tool descent beyond 3 mm, minus 0.7 per newton by
which each finger's final normal force falls short of 0.5 N. Held: descent at most 8 mm and every finger's final
normal force at least 0.5 N. A finger's normal force sums its contacts with the tool over all of its bodies.
"""
from __future__ import annotations

import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reorient_backends as rb  # noqa: E402

SCENES = ROOT / "assets/mjcf/experimental/20261007-diffmjx"
FINGERS = rb.FINGERS
OBJ = rb.OBJ
T_GRIP = rb.T_GRIP

HORIZON_MS = 400
PHASE_EDGES_MS = (120, 240, 400)
OPEN_UNTIL_MS = 180
CONTROL_CHANGE_MS = (0, 120, 180, 240)
N_PARAM = 13
YAW_SCALE_RAD = 0.35
CLOSE_SCALE_RAD = 0.30
FREE_DESCENT_M = 0.003
HELD_DESCENT_M = 0.008
MIN_FORCE_N = 0.5
DESCENT_WEIGHT = 70.0
FORCE_WEIGHT = 0.7
RELEASE_FORCE_N = 0.05
RECONTACT_MIN_MS = 20
TRACE_EVERY_MS = 10
METRIC_NAMES = ("tool_vertical_cos", "tool_z_m", "tool_omega_bodyx_rad_s", "tool_constraint_torque_bodyx_Nm")
# The fork's contact-from-distance impedance as the 2026-10-07 probes set it: (d0, d_width, width, midpoint, power)
CFD_SOLIMP = (0.0, 0.01, 0.05, 1.0, 2.0)


# ---------------------------------------------------------------------------------- scene and held state

def palm_plate(root: ET.Element) -> ET.Element:
    plate = root.find('.//body[@name="palm_pose"]/geom')
    assert plate is not None and plate.get("type") == "box", "palm_pose's first geom is not the display plate"
    return plate


def make_scene(hand: str = "D1", numerics: str = "bed", impratio: float = 100.0, no_post: bool = False,
               tool_capsule: bool = False, tip: str = "sphere", contact_model: str = "pt",
               pad_spacing_m: float = 0.001, out_dir: Path = SCENES) -> tuple[Path, dict]:
    """MJCF path and metadata of the continuation scene (see the module docstring)."""
    scene, meta = rb.build_scene(hand, tip, contact_model, "fast", numerics, impratio, "scene", out_dir,
                                 pad_s=pad_spacing_m)
    root = ET.parse(scene).getroot()
    plate = palm_plate(root)
    plate.set("contype", "0")
    plate.set("conaffinity", "0")
    if contact_model == "padsT":
        for geom in root.iter("geom"):
            if geom.get("name", "").endswith("_tipvis"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
    if tool_capsule:
        tool_geom = root.find(f'.//body[@name="{OBJ}"]/geom')
        assert tool_geom is not None and tool_geom.get("type") == "cylinder"
        tool_geom.set("type", "capsule")
    if no_post:
        post = root.find('.//body[@name="tool_platform"]/geom')
        assert post is not None
        post.set("contype", "0")
        post.set("conaffinity", "0")
    tag = "_mjx" + ("_capsule" if tool_capsule else "") + ("_no_post" if no_post else "")
    adapted = scene.with_name(scene.stem + tag + ".xml")
    tmp = adapted.with_name(adapted.name + ".tmp")
    ET.ElementTree(root).write(tmp)
    tmp.replace(adapted)
    return adapted, meta


def grip_state(model, meta: dict, hand: str, seed: int):
    """MjData after the plan's grip pose has been held T_GRIP on the post."""
    import mujoco

    data = mujoco.MjData(model)
    qadr = model.jnt_qposadr[model.body_jntadr[model.body(OBJ).id]]
    dx, dy, dyaw = rb.jitter(seed)
    data.qpos[qadr:qadr + 3] = np.asarray(meta["tool7"][:3]) + [dx, dy, 0.0]
    data.qpos[qadr + 3:qadr + 7] = rb._yaw_quat(meta["tool7"][3:], dyaw)
    for finger in FINGERS:
        for joint in rb.JOINTS:
            name = f"{finger}_{joint}"
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = meta["q0"][name]
    plan, traj = rb.load_plan(hand)
    target = rb.schedule(plan, traj)[0][1]
    for name, value in target.items():
        data.ctrl[model.actuator("a_" + name).id] = value
    while data.time < T_GRIP - 1e-9:
        mujoco.mj_step(model, data)
    return data


def clone_data(model, source):
    """A fresh MjData with source's state, warm start and controls, after mj_forward."""
    import mujoco

    data = mujoco.MjData(model)
    for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
        getattr(data, name)[:] = getattr(source, name)
    data.time = source.time
    mujoco.mj_forward(model, data)
    return data


def held_scene(tip: str = "tpu6", contact_model: str = "padsT", seed: int = 1, pad_spacing_mm: float = 1.0,
               tool_capsule: bool = False, hand: str = "D1", out_dir: Path = SCENES):
    """(scene path, metadata, model without the post, held MjData): the grip runs in the supported scene."""
    import mujoco

    spacing = pad_spacing_mm * 0.001
    scene, meta = make_scene(hand, no_post=True, tool_capsule=tool_capsule, tip=tip,
                             contact_model=contact_model, pad_spacing_m=spacing, out_dir=out_dir)
    supported, _ = make_scene(hand, tip=tip, contact_model=contact_model, pad_spacing_m=spacing, out_dir=out_dir)
    model = mujoco.MjModel.from_xml_path(str(scene))
    held = clone_data(model, grip_state(mujoco.MjModel.from_xml_path(str(supported)), meta, hand, seed))
    return scene, meta, model, held


def tool_addresses(model) -> tuple[int, int]:
    """qpos and qvel addresses of the tool's free joint."""
    joint = model.body_jntadr[model.body(OBJ).id]
    return int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])


_GEOM_TABLES: dict = {}


def _geom_tables(model) -> tuple[int, np.ndarray]:
    """(palm plate geom id, finger index of each geom with -1 off the fingers), cached per model."""
    key = id(model)
    if key not in _GEOM_TABLES or _GEOM_TABLES[key][0] is not model:
        body = model.body("palm_pose").id
        plate = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == body)
        finger = np.full(model.ngeom, -1)
        for g in range(model.ngeom):
            name = model.body(model.geom_bodyid[g]).name
            for j, f in enumerate(FINGERS):
                if name.startswith(f + "_"):
                    finger[g] = j
        _GEOM_TABLES[key] = (model, plate, finger)
    return _GEOM_TABLES[key][1], _GEOM_TABLES[key][2]


def palm_plate_geom(model) -> int:
    return _geom_tables(model)[0]


def finger_of_geom(model) -> np.ndarray:
    """Finger index (FINGERS order) of each geom's body, -1 for geoms not on a finger."""
    return _geom_tables(model)[1]


def contact_summary(model, data) -> dict:
    """Per-finger normal force and smallest contact distance on the tool, palm-plate contacts, total contacts."""
    import mujoco

    obj = model.body(OBJ).id
    plate = palm_plate_geom(model)
    finger = finger_of_geom(model)
    force = {f: 0.0 for f in FINGERS}
    min_dist = {f: None for f in FINGERS}
    plate_contacts = 0
    wrench = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        g0, g1 = (int(v) for v in contact.geom)
        if g0 == plate or g1 == plate:
            plate_contacts += 1
        tool0, tool1 = model.geom_bodyid[g0] == obj, model.geom_bodyid[g1] == obj
        if not (tool0 or tool1):
            continue
        j = finger[g1 if tool0 else g0]
        if j < 0:
            continue
        f = FINGERS[j]
        mujoco.mj_contactForce(model, data, i, wrench)
        force[f] += max(0.0, float(wrench[0]))
        dist = float(contact.dist)
        min_dist[f] = dist if min_dist[f] is None else min(min_dist[f], dist)
    return {"finger_normal_force_N": force, "finger_min_dist_m": min_dist,
            "palm_plate_contacts": plate_contacts, "total_contacts": int(data.ncon)}


# ---------------------------------------------------------------------------------- metrics

def vertical_cos(quat):
    """Cosine between the tool's body z axis (the shaft) and world z, from a (w, x, y, z) quaternion. Works on
    NumPy and JAX arrays."""
    return 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)


def metric_values(qpos, qvel, qfrc_constraint, qadr: int, vadr: int) -> tuple:
    """METRIC_NAMES of one state. The free joint's angular velocity and its constraint torque are in the tool's
    body frame, so the last two are about the tool's body x axis."""
    return (vertical_cos(qpos[qadr + 3:qadr + 7]), qpos[qadr + 2], qvel[vadr + 3], qfrc_constraint[vadr + 3])


def cpu_metrics(model, start, steps: int, ids=None, delta=0.0) -> np.ndarray:
    """METRIC_NAMES after `steps` CPU steps from `start` with `delta` added to ctrl[ids]."""
    import mujoco

    data = clone_data(model, start)
    if ids is not None:
        data.ctrl[ids] += delta
    for _ in range(steps):
        mujoco.mj_step(model, data)
    qadr, vadr = tool_addresses(model)
    return np.array(metric_values(data.qpos, data.qvel, data.qfrc_constraint, qadr, vadr))


def central_difference(fn, x0, directions, eps: float) -> np.ndarray:
    """Columns (fn(x0 + eps v) - fn(x0 - eps v)) / (2 eps) for each direction v."""
    x0 = np.asarray(x0, float)
    return np.column_stack([(np.asarray(fn(x0 + eps * v)) - np.asarray(fn(x0 - eps * v))) / (2 * eps)
                            for v in np.asarray(directions, float)])


# ---------------------------------------------------------------------------------- plan and score

def hold_vector(forced_release: bool = False) -> np.ndarray:
    x = np.zeros(N_PARAM)
    x[9:11] = -1.0
    if forced_release:
        x[9] = 1.0
    return x


def apply_forced_release(x, forced: bool) -> np.ndarray:
    x = np.asarray(x, float).copy()
    if forced:
        x[9] = 1.0
    return x


def plan_phase(k: int) -> int:
    return int(k >= PHASE_EDGES_MS[0]) + int(k >= PHASE_EDGES_MS[1])


def plan_ctrl(x, k: int, ctrl0, low, high, yaw_ids, flex_ids) -> np.ndarray:
    """Control vector the plan x applies from step k (one of CONTROL_CHANGE_MS) until the next change."""
    phase = plan_phase(k)
    ctrl = np.array(ctrl0, dtype=float, copy=True)
    ctrl[yaw_ids] = np.clip(ctrl0[yaw_ids] + YAW_SCALE_RAD * x[3 * phase:3 * phase + 3], low[yaw_ids], high[yaw_ids])
    for i, finger in enumerate(("index", "middle")):
        ids = flex_ids[finger]
        if k < OPEN_UNTIL_MS:
            amplitude = 0.5 * (x[9 + i] + 1.0)
            ctrl[ids] = ctrl0[ids] + amplitude * (low[ids] - ctrl0[ids])
        else:
            ctrl[ids] = np.clip(ctrl0[ids] + CLOSE_SCALE_RAD * x[11 + i], low[ids], high[ids])
    return ctrl


def score_and_held(cos0: float, z0: float, final_cos: float, min_z: float, forces: dict) -> dict:
    descent = max(0.0, z0 - min_z - FREE_DESCENT_M)
    shortfall = sum(max(0.0, MIN_FORCE_N - forces[f]) for f in FINGERS)
    return {"score": float(final_cos - cos0 - DESCENT_WEIGHT * descent - FORCE_WEIGHT * shortfall),
            "cos_gain": float(final_cos - cos0),
            "held_final": bool(min_z >= z0 - HELD_DESCENT_M and min(forces.values()) >= MIN_FORCE_N)}


def release_stats(trace: list, forces: dict) -> dict:
    """Release durations (10 ms samples below RELEASE_FORCE_N) and recontact flags of the index and middle."""
    out = {}
    for finger in ("index", "middle"):
        ms = int(sum(item["finger_normal_force_N"][finger] < RELEASE_FORCE_N for item in trace) * TRACE_EVERY_MS)
        out[f"{finger}_release_ms"] = ms
        out[f"{finger}_recontact"] = bool(ms >= RECONTACT_MIN_MS and forces[finger] >= MIN_FORCE_N)
    return out


class Scene:
    """CPU evaluator of 13-value plans from the held D1 state (scripts/real_v1_contact_gait_search.py)."""

    def __init__(self, tip: str, contact_model: str, seed: int = 1, forced_release: bool = False,
                 pad_spacing_mm: float = 1.0):
        import mujoco

        self.mujoco = mujoco
        self.scene, self.meta, self.model, self.held = held_scene(tip, contact_model, seed, pad_spacing_mm)
        self.qadr, self.vadr = tool_addresses(self.model)
        self.yaw_ids = np.array([self.model.actuator(f"a_{finger}_yaw").id for finger in FINGERS])
        self.flex_ids = {finger: np.array([self.model.actuator(f"a_{finger}_{joint}").id
                                           for joint in ("mcp", "pip")])
                         for finger in ("index", "middle")}
        self.ctrl0 = self.held.ctrl.copy()
        self.z0 = float(self.held.qpos[self.qadr + 2])
        self.cos0 = self.vertical_cos(self.held)
        self.low = self.model.actuator_ctrlrange[:, 0].copy()
        self.high = self.model.actuator_ctrlrange[:, 1].copy()
        self.calls = 0
        self.forced_release = forced_release
        self.pad_spacing_mm = pad_spacing_mm

    def vertical_cos(self, data) -> float:
        return float(vertical_cos(data.qpos[self.qadr + 3:self.qadr + 7]))

    def ctrl(self, x, k: int) -> np.ndarray:
        return plan_ctrl(x, k, self.ctrl0, self.low, self.high, self.yaw_ids, self.flex_ids)

    def evaluate(self, x, trace: bool = False, horizon_ms: int = HORIZON_MS, callback=None) -> dict:
        """Score row of plan x; trace adds 10 ms samples and release statistics; callback(k, data) runs after
        every step."""
        x = np.asarray(x, float)
        assert x.shape == (N_PARAM,)
        x = apply_forced_release(x, self.forced_release)
        self.calls += 1
        data = clone_data(self.model, self.held)
        min_z = self.z0
        contact_trace = []
        for k in range(horizon_ms):
            if k in CONTROL_CHANGE_MS:
                data.ctrl[:] = self.ctrl(x, k)
            self.mujoco.mj_step(self.model, data)
            min_z = min(min_z, float(data.qpos[self.qadr + 2]))
            if callback is not None:
                callback(k, data)
            if trace and (k + 1) % TRACE_EVERY_MS == 0:
                contact = contact_summary(self.model, data)
                contact_trace.append({"t_s": round((k + 1) * 0.001, 3),
                                      "vertical_cos": self.vertical_cos(data),
                                      "tool_z_m": float(data.qpos[self.qadr + 2]),
                                      "finger_normal_force_N": contact["finger_normal_force_N"]})
        forces = contact_summary(self.model, data)["finger_normal_force_N"]
        final_cos = self.vertical_cos(data)
        row = {**score_and_held(self.cos0, self.z0, final_cos, min_z, forces), "vertical_cos": final_cos,
               "tool_z_m": float(data.qpos[self.qadr + 2]), "min_tool_z_m": min_z,
               "finger_normal_force_N": forces, "x": x.tolist()}
        if trace:
            row["contact_trace"] = contact_trace
            row.update(release_stats(contact_trace, forces))
        return row

    def state_at(self, x, k_end: int):
        """MjData after k_end steps of plan x (controls applied as in evaluate)."""
        x = apply_forced_release(x, self.forced_release)
        data = clone_data(self.model, self.held)
        for k in range(k_end):
            if k in CONTROL_CHANGE_MS:
                data.ctrl[:] = self.ctrl(x, k)
            self.mujoco.mj_step(self.model, data)
        return data


# ---------------------------------------------------------------------------------- MJX (native 3.6 or the fork)

def mjx_setup(model, state, cfd: bool = False, scan_loop: bool = False, soft_collision: bool = False,
              fork_extras: bool = True):
    """(mx, md) for the installed MJX. The fork receives its switches (contact from distance with CFD_SOLIMP, the
    scan-loop solver, smooth collision); with 64-bit JAX the contact indices are cast to int64 so that a scan
    carry keeps its dtype."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    t0 = time.perf_counter()
    mx = mjx.put_model(model)
    put_model_s = time.perf_counter() - t0
    if hasattr(mx.opt, "cfd_enable"):
        options = dict(cfd_enable=cfd, cfd_solimp=jnp.array(CFD_SOLIMP), scan_loop=scan_loop)
        if fork_extras:
            options.update(col_soft_enable=soft_collision, softjax_mode="smooth", col_softness=0.01)
        mx = mx.replace(opt=mx.opt.replace(**options))
    elif cfd or scan_loop or soft_collision:
        raise RuntimeError("contact from distance, scan loop and smooth collision need the DiffMJX fork")
    md = mjx.put_data(model, state)
    if jax.config.read("jax_enable_x64"):
        if hasattr(md, "_impl"):
            md = md.tree_replace({f"_impl.contact.{k}": getattr(md._impl.contact, k).astype(np.int64)
                                  for k in ("geom1", "geom2", "geom")})
        else:
            md = md.replace(contact=md.contact.replace(**{k: getattr(md.contact, k).astype(jnp.int64)
                                                          for k in ("geom1", "geom2", "geom")}))
    return mx, md, put_model_s


def mjx_contact(data):
    """Contact and constraint force arrays of native MJX 3.6 (data._impl) or the fork (data)."""
    impl = data._impl if hasattr(data, "_impl") else data
    return impl.contact, impl.efc_force


def mjx_finger_gaps(model, data) -> dict:
    """Smallest signed distance between each finger and the tool over all candidate contacts."""
    contact, _ = mjx_contact(data)
    obj = model.body(OBJ).id
    finger = finger_of_geom(model)
    gaps = {f: None for f in FINGERS}
    for (g0, g1), dist in zip(np.asarray(contact.geom), np.asarray(contact.dist)):
        g0, g1 = int(g0), int(g1)
        tool0, tool1 = model.geom_bodyid[g0] == obj, model.geom_bodyid[g1] == obj
        if not (tool0 or tool1):
            continue
        j = finger[g1 if tool0 else g0]
        if j >= 0:
            f = FINGERS[j]
            gaps[f] = float(dist) if gaps[f] is None else min(gaps[f], float(dist))
    return gaps


def mjx_finger_forces(model, data) -> dict:
    """Per-finger count of contacts carrying force and their summed normal force, and the smallest distance."""
    contact, efc_force = mjx_contact(data)
    obj = model.body(OBJ).id
    finger = finger_of_geom(model)
    forces = np.asarray(efc_force)
    out = {f: {"n": 0, "normal_force_N": 0.0, "min_dist_m": None} for f in FINGERS}
    for (g0, g1), dist, address in zip(np.asarray(contact.geom), np.asarray(contact.dist),
                                       np.asarray(contact.efc_address)):
        g0, g1 = int(g0), int(g1)
        tool0, tool1 = model.geom_bodyid[g0] == obj, model.geom_bodyid[g1] == obj
        if not (tool0 or tool1):
            continue
        j = finger[g1 if tool0 else g0]
        if j < 0:
            continue
        row = out[FINGERS[j]]
        row["min_dist_m"] = float(dist) if row["min_dist_m"] is None else min(row["min_dist_m"], float(dist))
        if 0 <= address < len(forces) and forces[address] > 1e-6:
            row["n"] += 1
            row["normal_force_N"] += float(forces[address])
    return out


def mjx_rollout(mx, steps: int):
    """jit-able function state -> state after `steps` mjx.step calls under lax.scan."""
    import jax
    from mujoco import mjx

    def rollout(state):
        return jax.lax.scan(lambda carry, _: (mjx.step(mx, carry), None), state, None, length=steps)[0]
    return rollout


def timed(fn, *args):
    """(result, seconds) of fn(*args) with every output array ready."""
    import jax

    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    return out, time.perf_counter() - t0
