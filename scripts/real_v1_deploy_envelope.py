#!/usr/bin/env python3
"""Does the shippable schedule survive a hand that is not the simulated hand?

    uv run python scripts/real_v1_deploy_envelope.py --designs rv04_mid,ax_asym-10 --smoke
    uv run python scripts/real_v1_deploy_envelope.py --mode both --out docs/experiments/...

WHAT IS ACTUALLY BEING DEPLOYED. The 49 reorienting hands in `20260828-real_v1_search` all turn
the shaft OPEN LOOP: close to a fitted grasp, ramp the palm up, then run a straight line in joint
space from the grasp anchor to one end pose (`--linear-anchor`). That is a fixed list of joint
set-points streamed to position servos -- it needs no object-pose estimate, no camera and no
policy, which is exactly why it is the thing that can go on hardware tomorrow. The trained
anchored B policies need an observation vector the prototype does not have yet.

So the deployable is a TRAJECTORY, and the sim2real question is the one this file measures:
plan the trajectory on the nominal hand, then execute that same trajectory, unchanged, on a hand
/ tool / servo that differs from it. Re-fitting on the perturbed scene would measure whether the
PIPELINE can adapt, which is not what happens when the plan is already loaded on the robot.

WHY NOT `sim2real_robustness_sweep.py`. That one drives the m05 inline hand's a10->b33 pair
through mjwarp on the GPU. This hand has no trained deployable, runs on CPU MuJoCo, and its
perturbations include ones that sweep has no knob for (servo gain, torque ceiling, servo
calibration bias, mount placement error). Same intent, different subject.

THE AXES, and why each is here. Grouped by who is wrong -- the tool, the hand, or the servo:

  place_*     where the tool lands relative to the palm. The search jittered spawn by 0.5 mm sd;
              a human putting a screwdriver on a table in front of a gantry is off by millimetres
              and by degrees of yaw, which was never sampled at all.
  mass        the printed PLA tool. 24.5 g in the scene = ~40% effective infill; solid PLA is
              2.5x that and 2x is already known to drop the shaft.
  radius      printed shaft tolerance and the nominal-vs-actual diameter.
  friction    a material guess. PLA on TPU is not mu 2.4, and on this hand LOWER friction helped.
  solimp/     contact stiffness. The pads are printed TPU; the scene's compliance is a solver
  solref      setting, not a measurement.
  kp/kv       the servos are modelled kp=30 kv=0.5, which came from nowhere. A real servo is
              softer and laggier, and the grip force IS kp x tracking error.
  torque      forcerange +-10 N.m in the scene. A hobby servo saturates well below that.
  damping     gearbox friction, which the model does not have at all.
  ctrl_bias   servo zero-offset / calibration error, per joint, in degrees. This is the single
              most certain hardware error -- nobody's servo horn is on the spline exactly right.
  mount       the gantry mounts are positioned by hand; the built hand is not the drawn hand.

`ensemble` mode draws all of them at once from hardware-plausible distributions, because on the
real bench everything is wrong simultaneously, and one-at-a-time margins do not compose.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mujoco  # noqa: E402

import probe_real_v1_carry as pc  # noqa: E402
import real_v1_design_search as ds  # noqa: E402
from morphohand.studies.scene_mutate import Scene  # noqa: E402
from morphohand.tools.keyframe_ik import FINGERS  # noqa: E402

OBJ = "screwdriver_medium"
TABLE = ROOT / "docs/experiments/20260828-real_v1_search/table.json"
DATE = "20260829"
SCRATCH = ROOT / f"assets/mjcf/experimental/{DATE}-deploy_envelope"

# SCS0009 at 6 V, from Feetech's product specification.  The hardware's present-load field is
# not calibrated torque, but the bench protection experiment establishes that its 0..1000 scale
# tracks the percentage registers: overload_torque=80 trips near 800 and protective_torque=20
# produces an exact 200 plateau (and 40 produces 400).  This therefore is a useful screening
# proxy, not a metrology claim.
KGCM_TO_NM = 9.80665 / 100.0
SCS0009_STALL_TORQUE_NM = 2.3 * KGCM_TO_NM       # 0.2256 N m, published peak stall
SCS0009_RATED_TORQUE_NM = 0.7 * KGCM_TO_NM       # 0.0686 N m, published rated torque
SCS0009_OVERLOAD_TORQUE_NM = 0.8 * SCS0009_STALL_TORQUE_NM
SCS0009_PROTECTIVE_TORQUE_NM = 0.2 * SCS0009_STALL_TORQUE_NM

# The keepers from the 108-hand search, best-cell mean cos in brackets. rv05_manual is left out:
# its winning grasp is an authored pose, not one `_grip_from_fit` reproduces.
DEFAULT_DESIGNS = "rv04_mid,ax_asym-10,r08,g10,ax_asym+10,rv00_wide,rv03_narrowy"

# The tools. `medium` is the shipped 25 x 100 mm uniform cylinder every result to date is about.
# The two stubbies are a Stanley #2 stubby's catalogue dimensions with two different builds:
# printed in one PLA piece, or a printed handle on a steel shank (which is what a bought one is,
# and is 3x the mass with the extra mass at the far end of the lever).
OBJECTS = {
    "medium": {},
    "stubby_pla": dict(handle_density=500.0, shaft_density=500.0),
    "stubby_steel": dict(handle_density=500.0, shaft_density=7850.0),
    "stubby_solid": dict(handle_density=1240.0, shaft_density=7850.0),
}


def object_scene(base: Path, obj: str, bench: float = 0.0, post_y: float = 0.0,
                 flat_pads: bool = False, pad_len: float = 0.0148,
                 pad_width: float = 0.0148, flat_links: bool = False) -> Path:
    """The design's scene holding `obj`, optionally standing it on a bench platform."""
    if obj == "medium" and bench <= 0.0 and not flat_pads:
        return base
    tag = (obj + (f"__bench{bench*1000:.0f}py{post_y*1000:+.0f}" if bench > 0 else "")
           + (f"__flat{pad_len*1000:.0f}w{pad_width*1000:.0f}"
              f"{'L' if flat_links else 'T'}" if flat_pads else ""))
    out = SCRATCH / f"{base.stem}__{tag}.xml"
    if not out.exists():
        SCRATCH.mkdir(parents=True, exist_ok=True)
        sc = Scene(base)
        if flat_pads:
            sc.set_finger_flat_pads(pad_len=pad_len, width=pad_width, links=flat_links)
        if obj != "medium":
            sc.set_object_stubby(**OBJECTS[obj])
        if bench > 0.0:
            sc.set_object_platform(bench, post_y=post_y)
        tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
        sc.write(tmp)
        os.replace(tmp, out)
    return out


# --------------------------------------------------------------------------------------------
# CAN THE SERVOS BE TOLD TO DO THIS?  Clearance says the modelled fingers miss each other; it
# says nothing about whether the command is legal.  sv1_w0116 cleared 11.6 mm, scored the best
# cosine in the whole 128-hand pilot, and its GRIP asks middle yaw for 84 deg against a 70 deg
# cap -- nothing noticed until the plan was exported, four stages later.  The budget is exactly
# what moves a joint further, so the check belongs beside it, in the screen.
# --------------------------------------------------------------------------------------------
_QPOSADR: dict[str, dict[str, int]] = {}


def servo_shortfall(scene: Path, plan: dict, budget: float) -> tuple[float, str]:
    """Worst servo-range overrun of this plan's set-points, in degrees. 0.0 = commandable.

    Checks the same three poses `real_v1_export_plan.py` writes and `HandPlan.validate` gates
    -- open, grip, end of turn -- so a design that passes here passes there. Returns (0.0, "")
    when the driver package is unavailable, which keeps the screen runnable without it.
    """
    try:
        sys.path.insert(0, str(ROOT / "src/morphohand/driver/manta/host"))
        from manta_hand.plan import joint_violation
    except Exception:
        return 0.0, ""
    key = str(scene)
    if key not in _QPOSADR:
        m = mujoco.MjModel.from_xml_path(key)
        _QPOSADR[key] = {j: int(m.joint(j).qposadr[0])
                         for js in FINGERS.values() for j in js}
    adr = _QPOSADR[key]
    qpos = np.asarray(plan["open_qpos"])
    poses = {"open": {j: float(np.degrees(qpos[adr[j]])) for j in adr},
             "grip": {j: float(np.degrees(plan["anchor"][j])) for j in adr},
             "turn_end": {j: float(np.degrees(plan["anchor"][j]
                                              + float(np.clip(plan["delta"][j],
                                                              -budget, budget)))) for j in adr}}
    if plan.get("squeeze_delta"):
        poses["resqueeze"] = {j: float(np.degrees(plan["anchor"][j] + plan["squeeze_delta"][j]))
                              for j in adr}
    worst, who = 0.0, ""
    for name, vals in poses.items():
        for finger, joints in FINGERS.items():
            for j in joints:
                v = joint_violation(finger, j.rpartition("_")[2], vals[j])
                if v is not None and v.short > worst:
                    worst, who = v.short, f"{name}:{finger}_{v.axis}"
    return round(worst, 2), who


# --------------------------------------------------------------------------------------------
# the plan: a joint-space trajectory, computed once, on the nominal hand
# --------------------------------------------------------------------------------------------
def make_plan(scene: Path, *, straddle: float, depth: float | None, thumb_axial: float,
              squeeze: float, axis_k: float, angle_deg: float, lift: float,
              budget: float, turn_steps: int, hold_squeeze: float = 0.0,
              squeeze_steps: int = 200, bench: bool = False) -> dict | None:
    """Everything a servo needs, and nothing that depends on the physics being right.

    Mirrors `probe_real_v1_carry.carry(..., linear_anchor=True)` exactly, but stops at the
    trajectory instead of rolling it out, so the same plan can be replayed on any scene.
    """
    built = pc._grip_from_fit(scene, straddle, 0.0, squeeze, OBJ, depth, thumb_axial)
    if built is None:
        return None
    m, open_qpos, grip, depth_mm = built
    d = mujoco.MjData(m)
    d.qpos[:] = open_qpos
    d.qvel[:] = 0.0
    d.ctrl[:] = grip
    acts = pc._finger_act(m)
    pz_a = next(k for k in range(m.nu) if m.actuator(k).name == "a_palm_pz")
    mujoco.mj_forward(m, d)
    for _ in range(250):
        mujoco.mj_step(m, d)
    if bench:
        for _ in range(400):
            mujoco.mj_step(m, d)
    else:
        pz0 = float(d.ctrl[pz_a])
        for k in range(200):
            d.ctrl[pz_a] = pz0 + lift * (k + 1) / 200
            mujoco.mj_step(m, d)
        for _ in range(200):
            mujoco.mj_step(m, d)

    tip0 = {f: d.body(f"{f}_tip").xpos.copy() for f in FINGERS}
    centroid = np.mean([tip0[f] for f in FINGERS], axis=0)
    span = abs(tip0["index"][1] - tip0["middle"][1]) / 2.0
    centroid[2] += axis_k * span
    q0 = {j: float(d.qpos[m.jnt_qposadr[m.joint(j).id]]) for j in acts}

    mik = mujoco.MjModel.from_xml_path(str(scene))
    dik = mujoco.MjData(mik)
    dik.qpos[:] = d.qpos
    dik.qvel[:] = 0.0
    mujoco.mj_forward(mik, dik)
    R = pc._rotx(np.radians(angle_deg))
    for f in FINGERS:
        pc.ik_finger(mik, dik, f, centroid + R @ (tip0[f] - centroid), iters=400)
    end = {j: float(dik.qpos[mik.jnt_qposadr[mik.joint(j).id]]) for j in acts}

    # THE RE-SQUEEZE, PRECOMPUTED. These are position servos, so the grip force IS the commanded-
    # minus-actual error; as the shaft creeps down through the pads over the turn that error
    # shrinks and the force decays toward zero (measured: 16.1 N -> 0 over 0.8 s). `hold_squeeze`
    # is one more set-point that puts the pads back INSIDE the rotated shaft's surface.
    # `probe_real_v1_carry` solves it from the LIVE state at the end of the turn, which a
    # streamed trajectory cannot do, so it is solved here on the nominal hand's own end-of-turn
    # state and shipped as a second joint-space ramp. That makes it open-loop, which is the whole
    # premise -- and it means a hand that ends the turn somewhere else gets the nominal squeeze.
    sq_delta = None
    if hold_squeeze > 0.0:
        dsim = mujoco.MjData(m)
        dsim.qpos[:] = d.qpos
        dsim.qvel[:] = d.qvel
        dsim.ctrl[:] = d.ctrl
        mujoco.mj_forward(m, dsim)
        for k in range(1, turn_steps + 1):
            u = k / turn_steps
            for j, a in acts.items():
                dsim.ctrl[a] = float(grip[a]) + float(np.clip((end[j] - q0[j]) * u,
                                                             -budget, budget))
            mujoco.mj_step(m, dsim)
        o = dsim.body(OBJ).xpos.copy()
        ax = dsim.body(OBJ).xmat.reshape(3, 3)[:, 2]
        dik.qpos[:] = dsim.qpos
        dik.qvel[:] = 0.0
        mujoco.mj_forward(mik, dik)
        for f in FINGERS:
            t = dsim.body(f"{f}_tip").xpos.copy()
            v = (t - o) - float((t - o) @ ax) * ax          # radial component only
            n = float(np.linalg.norm(v))
            if n < 1e-6:
                continue
            pc.ik_finger(mik, dik, f, t - (v / n) * hold_squeeze, iters=200)
        sq_end = {j: float(dik.qpos[mik.jnt_qposadr[mik.joint(j).id]]) for j in acts}
        sq_delta = {j: sq_end[j] - q0[j] for j in acts}

    return {
        "scene": str(scene), "straddle": straddle, "depth": depth, "bench": bench,
        "thumb_axial": thumb_axial, "squeeze": squeeze, "axis_k": axis_k,
        "angle_deg": angle_deg, "lift": lift, "budget": budget, "turn_steps": turn_steps,
        "grip_depth_mm": float(depth_mm),
        "open_qpos": open_qpos.tolist(), "grip_ctrl": np.asarray(grip).tolist(),
        # the shippable part: per finger joint, the hold set-point and the ramp it moves through
        "anchor": {j: float(grip[a]) for j, a in acts.items()},
        "delta": {j: end[j] - q0[j] for j in acts},
        "hold_squeeze": hold_squeeze, "squeeze_steps": squeeze_steps,
        "squeeze_delta": sq_delta,
    }


# --------------------------------------------------------------------------------------------
# execution: the same trajectory, on whatever hand it is given
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# FORCE REGULATION. The open-loop plan commands joint ANGLES, and on position servos the grip
# force is whatever the commanded-minus-actual error happens to be -- which is why a 2 degree
# zero error, or a shaft 6% thinner than drawn, changes the grip without changing the plan.
#
# The phase decomposition says where this can and cannot help: at 1-2 degrees of joint error the
# grasp is never whiffed (100% grasped, 100% lifted) and 61% of the lifted runs lose the shaft
# DURING THE TURN. So gating the close on contact force -- the obvious fix -- buys nothing at
# small errors, because the close was already fine. What is missing is regulation THROUGH the
# turn, and that is what this does: per finger, nudge the flexion joints until the measured
# normal force sits in a band. It needs no object pose, only per-finger normal force, which on
# hardware is servo current -- so it is as deployable as the open-loop plan itself.
# WHY IT IS A JACOBIAN SQUEEZE AND NOT JUST "FLEX MORE". Traced through g14's open-loop turn,
# per-finger normal force runs 14/8/6 N at the start and reaches 0.1/0.0/0.0 N by 70% of the way
# round; the last third of the rotation is gravity settling a shaft the hand is no longer really
# holding. The first version of this regulator answered that by adding flexion to mcp and pip,
# wound to its full authority, and still measured 0.4 N -- because at that configuration more
# flexion curls the fingertip ALONG the shaft rather than into it. So the correction has to be
# taken in task space: move each pad toward the pinch axis, and solve for the joint step with the
# finger's own Jacobian.
#
# THIS IS STILL DEPLOYABLE. The direction comes from forward kinematics on measured joint angles
# and the pad centroid -- no object pose, no camera. The trigger is per-finger normal force,
# which on a servo is current. Both are things the prototype already has.
def _squeeze_dirs(m, d, acts) -> dict[str, np.ndarray]:
    """Per finger, the joint-space direction that moves its pad toward the pinch axis.

    The pinch axis is the line through the three pads' centroid along the shaft (world Y), so
    "inward" is the pad-to-centroid vector with its Y component removed. Nothing here reads the
    object.
    """
    tips = {f: d.body(f"{f}_tip").xpos.copy() for f in FINGERS}
    centroid = np.mean(list(tips.values()), axis=0)
    out = {}
    jacp = np.zeros((3, m.nv))
    for f, joints in FINGERS.items():
        v = centroid - tips[f]
        v[1] = 0.0                                   # radial only: never slide along the shaft
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            out[f] = np.zeros(len(joints))
            continue
        mujoco.mj_jacBody(m, d, jacp, None, m.body(f"{f}_tip").id)
        cols = [m.jnt_dofadr[m.joint(j).id] for j in joints]
        J = jacp[:, cols]                            # 3 x 3
        dq = np.linalg.lstsq(J, v / n, rcond=None)[0]
        nn = float(np.linalg.norm(dq))
        out[f] = dq / nn if nn > 1e-9 else dq
    return out


def _force_step(m, d, acts, trim: dict, target: float, gain: float, authority: float,
                obj: str) -> None:
    """One step of per-finger normal-force regulation. Updates `trim` in place.

    A deadband of +-30% of the set-point, because the first version had none and integrated
    itself into the stop within a fifth of the turn.
    """
    pf = pc._per_finger_contact(m, d, obj)
    dirs = _squeeze_dirs(m, d, acts)
    for f, joints in FINGERS.items():
        fn = float(pf[f]["fn"])
        if abs(fn - target) <= 0.3 * target:
            continue
        step = float(np.clip(gain * (target - fn), -0.0006, 0.0006))
        for j, u in zip(joints, dirs[f]):
            trim[j] = float(np.clip(trim[j] + step * float(u), -authority, authority))


def _servo_load_units(m, d, acts) -> dict[str, float]:
    """Napkin SCS0009 present-load proxy from simulated actuator torque.

    A value of 1000 corresponds to published stall torque.  The real register contains gearbox
    friction, inertial load, deadband and protection plateaus, so this is intentionally used as
    a broad control/ranking signal only.
    """
    return {
        f: float(np.clip(max(abs(float(d.actuator_force[acts[j]])) for j in joints)
                         / SCS0009_STALL_TORQUE_NM * 1000.0, 0.0, 1000.0))
        for f, joints in FINGERS.items()
    }


def _load_step(m, d, acts, trim: dict, target: float, gain: float,
               authority: float) -> None:
    """Move each pad inward/outward to maintain a per-finger servo-load band.

    The direction comes only from measured joint positions and hand FK; the trigger is the
    SCS0009-like load proxy.  It does not read object pose or simulator contact force.
    """
    loads = _servo_load_units(m, d, acts)
    dirs = _squeeze_dirs(m, d, acts)
    for f, joints in FINGERS.items():
        error = target - loads[f]
        if abs(error) <= 0.3 * target:
            continue
        step = float(np.clip(gain * error / 1000.0, -0.0006, 0.0006))
        for j, u in zip(joints, dirs[f]):
            trim[j] = float(np.clip(trim[j] + step * float(u), -authority, authority))


def _set_servo_torque_limit(m, acts, limit_nm: float) -> None:
    """Cap the nine finger actuators without touching the palm gantry actuators."""
    if limit_nm <= 0.0:
        return
    for actuator in set(acts.values()):
        m.actuator_forcelimited[actuator] = 1
        m.actuator_forcerange[actuator] = (-limit_nm, limit_nm)


def execute(scene: Path, plan: dict, *, place=(0.0, 0.0, 0.0), yaw: float = 0.0,
            ctrl_bias: dict | None = None, hold_steps: int = 800, seed: int = 0,
            jitter: float = 0.0005, video: Path | None = None,
            force_target: float = 0.0, force_gain: float = 0.0015,
            force_band: float = 0.45, force_phase: str = "all",
            selfcollision: bool = False, load_target_units: float = 0.0,
            load_gain: float = 0.0024, capture_steps: int = 0,
            proof_lift: float = 0.0, proof_lift_steps: int = 500,
            proof_max_slip: float = 0.010,
            turn_torque_limit_nm: float = 0.0, hold_torque_limit_nm: float = 0.0) -> dict:
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(plan["open_qpos"])
    d.qvel[:] = 0.0

    # WHERE THE TOOL IS. `place` is the deliberate placement error in metres, `jitter` the
    # irreducible one -- both land on the object's freejoint, which is the only honest place for
    # them: the palm is a gantry and knows where it is, the tool on the table does not.
    adr = int(m.jnt_qposadr[m.body(OBJ).jntadr[0]])
    rng = np.random.default_rng(seed)
    d.qpos[adr + 0] += place[0] + (rng.normal(0.0, jitter) if jitter > 0 else 0.0)
    d.qpos[adr + 1] += place[1] + (rng.normal(0.0, jitter) if jitter > 0 else 0.0)
    d.qpos[adr + 2] += place[2]
    if abs(yaw) > 1e-9:
        q = d.qpos[adr + 3:adr + 7].copy()
        qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, qz, q)
        d.qpos[adr + 3:adr + 7] = out

    acts = pc._finger_act(m)
    _set_servo_torque_limit(m, acts, turn_torque_limit_nm)
    pz_a = next(k for k in range(m.nu) if m.actuator(k).name == "a_palm_pz")
    bias = ctrl_bias or {}
    d.ctrl[:] = np.asarray(plan["grip_ctrl"])
    for j, a in acts.items():
        d.ctrl[a] = plan["anchor"][j] + bias.get(j, 0.0)
    mujoco.mj_forward(m, d)

    vid, vcam, frames = None, None, []
    if video is not None:
        vid = mujoco.Renderer(m, height=480, width=640)
        vcam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, vcam)
        vcam.azimuth, vcam.elevation, vcam.distance = 0.0, -12.0, 0.40
    step_i = 0
    clr_pairs, clr_owner = pc._cross_finger_pairs(m) if selfcollision else ([], {})
    clr_min, clr_at, clr_pair = float("inf"), None, None

    def _after_step():
        nonlocal step_i, clr_min, clr_at, clr_pair
        step_i += 1
        if clr_pairs and step_i % 10 == 0:
            dist, who = pc._min_cross_clearance(m, d, clr_pairs, clr_owner)
            if dist is not None and dist < clr_min:
                clr_min, clr_at, clr_pair = dist, step_i, who
        if vid is not None and step_i % 8 == 0:
            vcam.lookat[:] = d.body(OBJ).xpos
            vid.update_scene(d, vcam)
            frames.append(vid.render())

    def _run(n, before=None):
        for k in range(n):
            if before is not None:
                before(k)
            mujoco.mj_step(m, d)
            _after_step()

    _run(250)
    if plan.get("bench"):
        # FIXED PALM. The tool already sits at working height on a platform, so there is no
        # lift; the schedule is close -> settle -> turn. The palm still has to be commanded to
        # hold station, which it does by simply not being moved.
        _run(400)
    else:
        pz0 = float(d.ctrl[pz_a])
        _run(200, lambda k: d.ctrl.__setitem__(pz_a, pz0 + plan["lift"] * (k + 1) / 200))
        _run(200)
    lifted = {"cos": pc._cos(m, d, OBJ), "z": float(d.body(OBJ).xpos[2]),
              "contacts": pc._contacts_hand(m, d, OBJ)}

    budget, turn = plan["budget"], plan["turn_steps"]
    # The regulator rides ON TOP of the planned trajectory: the plan still says where the fingers
    # go, the regulator only trims each finger's flexion to hold its share of the load. Its
    # authority is capped at `force_band` rad so it can never rewrite the turn.
    # WHEN THE LOOP IS ALLOWED TO ACT. Regulating through the turn holds the shaft better and
    # turns it worse: the last third of the rotation is the shaft SETTLING into vertical under
    # gravity against a grip that has decayed to a fraction of a newton, so a hand that keeps
    # squeezing keeps it where it was. `hold` therefore leaves the turn open-loop exactly as
    # planned and only closes the loop once the command has finished -- which is the phase where
    # the open-loop schedule is left holding 0.2 N and loses the tool to any disturbance.
    trim = {j: 0.0 for j in acts}
    if force_target > 0.0 and force_phase != "hold":
        for _ in range(200):                      # settle the grip into the band before turning
            _force_step(m, d, acts, trim, force_target, force_gain, force_band, OBJ)
            for j, a in acts.items():
                d.ctrl[a] = plan["anchor"][j] + trim[j] + bias.get(j, 0.0)
            mujoco.mj_step(m, d)
            _after_step()
    for k in range(1, turn + 1):
        u = k / turn
        for j, a in acts.items():
            delta = plan["delta"][j] * u
            d.ctrl[a] = (plan["anchor"][j] + float(np.clip(delta, -budget, budget))
                         + trim[j] + bias.get(j, 0.0))
        if force_target > 0.0 and force_phase != "hold" and k % 5 == 0:
            _force_step(m, d, acts, trim, force_target, force_gain, force_band, OBJ)
            for j, a in acts.items():
                d.ctrl[a] = (plan["anchor"][j]
                             + float(np.clip(plan["delta"][j] * u, -budget, budget))
                             + trim[j] + bias.get(j, 0.0))
        mujoco.mj_step(m, d)
        _after_step()

    if plan.get("squeeze_delta"):
        sq_start = {j: float(d.ctrl[a]) for j, a in acts.items()}
        for k in range(1, int(plan["squeeze_steps"]) + 1):
            u = k / plan["squeeze_steps"]
            for j, a in acts.items():
                tgt = plan["anchor"][j] + plan["squeeze_delta"][j]
                d.ctrl[a] = (sq_start[j] + (tgt - sq_start[j]) * u)
                d.ctrl[a] = float(np.clip(d.ctrl[a], plan["anchor"][j] - budget,
                                          plan["anchor"][j] + budget)) + bias.get(j, 0.0)
            mujoco.mj_step(m, d)
            _after_step()

    # The turn may use short-duration torque below the 80% protection threshold, but a long
    # capture must live near the published rated torque. A single static MJCF limit cannot
    # express that phase change, so switch the nine actuator ceilings here.
    _set_servo_torque_limit(m, acts, hold_torque_limit_nm)

    def _hold_regulator():
        if load_target_units > 0.0:
            _load_step(m, d, acts, trim, load_target_units, load_gain, force_band)
        elif force_target > 0.0:
            _force_step(m, d, acts, trim, force_target, force_gain, force_band, OBJ)

    def _apply_trim():
        for joint, actuator in acts.items():
            d.ctrl[actuator] += trim[joint] - applied_trim[joint]
            applied_trim[joint] = trim[joint]

    applied_trim = dict(trim)
    peak = pc._cos(m, d, OBJ)
    min_z = float(d.body(OBJ).xpos[2])
    # Give the proprioceptive clamp time to establish a load band before testing whether the
    # object is genuinely captured. This still sees only nine positions + nine load proxies.
    for k in range(capture_steps):
        if k % 5 == 0:
            _hold_regulator()
            _apply_trim()
        mujoco.mj_step(m, d)
        min_z = min(min_z, float(d.body(OBJ).xpos[2]))
        peak = max(peak, pc._cos(m, d, OBJ))
        _after_step()

    # PROOF LIFT. A cylinder balanced against a finger or left on the support can pass an image-
    # based final-angle score. Raise the whole palm; only an actually captured object follows.
    proof_start_z = float(d.body(OBJ).xpos[2])
    pz_start = float(d.ctrl[pz_a])
    for k in range(proof_lift_steps if proof_lift > 0.0 else 0):
        d.ctrl[pz_a] = pz_start + proof_lift * (k + 1) / proof_lift_steps
        if k % 5 == 0:
            _hold_regulator()
            _apply_trim()
        mujoco.mj_step(m, d)
        min_z = min(min_z, float(d.body(OBJ).xpos[2]))
        peak = max(peak, pc._cos(m, d, OBJ))
        _after_step()
    proof_end_z = float(d.body(OBJ).xpos[2])
    proof_rise = proof_end_z - proof_start_z
    proof_lift_ok = proof_lift <= 0.0 or proof_rise >= 0.8 * proof_lift
    min_z_free = float(d.body(OBJ).xpos[2])
    for hk in range(hold_steps):
        if (force_target > 0.0 or load_target_units > 0.0) and hk % 5 == 0:
            _hold_regulator()
            _apply_trim()
        mujoco.mj_step(m, d)
        min_z = min(min_z, float(d.body(OBJ).xpos[2]))
        min_z_free = min(min_z_free, float(d.body(OBJ).xpos[2]))
        c = pc._cos(m, d, OBJ)
        peak = peak if abs(peak) >= abs(c) else c
        _after_step()
    nh, foh = pc._contacts_hand(m, d, OBJ)
    proof_slip = max(0.0, proof_end_z - min_z_free)
    proof_hold_ok = proof_lift <= 0.0 or proof_slip <= proof_max_slip
    # ON THE BENCH, "held" IS NOT ENOUGH. The tool stands on a post rather than being lifted
    # clear, so a rollout can bring it upright while it is still resting on that post -- the
    # same floor-assisted reorient `REORIENT_PRIMITIVE.txt` measured as 46-69% of both reference
    # policies' alignment, reintroduced by the bench. So the post is counted as a contact and a
    # run that ends touching it does not pass.
    post = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "tool_post")
    on_post = 0
    if post >= 0:
        oid = m.body(OBJ).id
        for c in range(d.ncon):
            g1, g2 = d.contact.geom1[c], d.contact.geom2[c]
            if (g1 == post and m.geom_bodyid[g2] == oid) or \
               (g2 == post and m.geom_bodyid[g1] == oid):
                on_post += 1
    if vid is not None and frames:
        import imageio.v3 as iio
        video.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(video, np.stack(frames), fps=40)
    return {
        "lifted_z": round(lifted["z"], 4), "lifted_contacts": lifted["contacts"][0],
        "peak_cos": round(float(peak), 3), "final_cos": round(pc._cos(m, d, OBJ), 3),
        "final_z": round(float(d.body(OBJ).xpos[2]), 4), "min_z_hold": round(min_z, 4),
        "contacts_hand": nh, "force_hand_N": round(foh, 2), "on_post": on_post,
        "load_units": {f: round(v, 1) for f, v in _servo_load_units(m, d, acts).items()},
        "proof_lift_mm": round(proof_lift * 1000.0, 1),
        "proof_rise_mm": round(proof_rise * 1000.0, 1),
        "proof_lift_ok": proof_lift_ok,
        "proof_slip_mm": round(proof_slip * 1000.0, 1),
        "proof_max_slip_mm": round(proof_max_slip * 1000.0, 1),
        "proof_hold_ok": proof_hold_ok,
        "min_z_free": round(min_z_free, 4),
        "turn_torque_limit_nm": turn_torque_limit_nm,
        "hold_torque_limit_nm": hold_torque_limit_nm,
        "min_finger_clearance_mm": (None if clr_min == float("inf")
                                    else round(clr_min * 1000, 2)),
        "clearance_at_step": clr_at,
        "clearance_pair": None if clr_pair is None else "-".join(clr_pair),
        "force_target_N": force_target,
        "trim_max_deg": round(float(np.degrees(max(abs(v) for v in trim.values()))), 2),
        # Same `ok` as the search: held, not merely upright -- a shaft standing on the table
        # reads cos 1.0, and one that comes up and then slides out reads a fine final z.
        "ok": bool(nh >= 1 and min_z > lifted["z"] - 0.02 and on_post == 0
                   and proof_lift_ok
                   and proof_hold_ok),
    }


# --------------------------------------------------------------------------------------------
# perturbations: a name -> (scene mutation, execution kwargs)
# --------------------------------------------------------------------------------------------
def mutate(base: Path, spec: dict, out_dir: Path) -> Path:
    """Write (once) the scene for a mutation spec. Identity spec returns the base scene."""
    keys = {k: v for k, v in spec.items() if k in
            ("mass", "radius", "friction", "solimp", "solref", "kp", "kv",
             "torque", "damping", "mount_mm", "mount_seed")}
    if not keys:
        return base
    tag = "_".join(f"{k}{v}" for k, v in sorted(keys.items())).replace(".", "d")
    # An ensemble draw sets ten continuous knobs at once and the readable tag runs past the
    # filesystem's 255-byte name limit, so long specs get hashed. The spec itself is recorded
    # in the result row; the filename only has to be unique.
    if len(tag) > 90:
        tag = "ens" + hashlib.md5(tag.encode()).hexdigest()[:16]
    out = out_dir / f"{base.stem}__{tag}.xml"
    if out.exists():
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    s = Scene(base)
    if "mass" in keys:
        s.scale_object_density(keys["mass"])
    if "radius" in keys:
        s.scale_object_radius(keys["radius"])
    if "friction" in keys:
        s.scale_friction(keys["friction"])
    if "solimp" in keys:
        s.set_solimp(min(0.97, keys["solimp"] - 0.02), keys["solimp"])
    if "solref" in keys:
        s.set_solref(keys["solref"])
    if "kp" in keys or "kv" in keys:
        s.scale_actuator_gain(keys.get("kp", 1.0), keys.get("kv", keys.get("kp", 1.0)))
    if "torque" in keys:
        s.scale_actuator_force(keys["torque"])
    if "damping" in keys:
        s.scale_joint_damping(keys["damping"])
    if "mount_mm" in keys:  # noqa: SIM102 (kept adjacent to its comment)
        # The built hand is not the drawn hand: each mount is off by up to `mount_mm`,
        # independently in x and y, which is what positioning three gantry blocks by hand costs.
        rng = np.random.default_rng(1000 + int(keys.get("mount_seed", 0)))
        m0 = mujoco.MjModel.from_xml_path(str(base))
        mounts = {}
        for f in FINGERS:
            p = m0.body(f"{f}_mount").pos
            e = rng.uniform(-1, 1, 2) * keys["mount_mm"] * 1e-3
            mounts[f] = (float(p[0] + e[0]), float(p[1] + e[1]))
        s.set_mounts(mounts)
    # ATOMIC. Eighteen workers share this cache and the first 24 rollouts of the first run died
    # on `ParseXML: empty file` -- one worker was reading the path while another was still
    # writing it. Write private, then rename, which is atomic on the same filesystem.
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    s.write(tmp)
    os.replace(tmp, out)
    return out


def dense_place_points() -> list[tuple[str, str, dict]]:
    """A fine map of the placement basin, which the coarse grid could not resolve.

    At +-3 mm the coarse sweep read 0% on most designs and 100% on one, which is not a wall --
    it is a basin whose centre is not where the plan put it. A trajectory is planned for one
    object pose and the fitter always centres the pinch on the object's mid-length; if the
    basin's centre is 3 mm off that, then the shipped plan is starting at the edge of its own
    tolerance and half the placement error budget is being given away for free.
    """
    pts = [("baseline", "as-simulated", {})]
    for mm in range(-8, 9):
        pts.append(("fine_x", f"{mm:+d}mm", {"place": (mm / 1000.0, 0.0, 0.0)}))
        pts.append(("fine_y", f"{mm:+d}mm", {"place": (0.0, mm / 1000.0, 0.0)}))
    for deg in range(-20, 21, 2):
        pts.append(("fine_yaw", f"{deg:+d}deg", {"yaw": np.radians(deg)}))
    # HEIGHT ONLY MATTERS WITHOUT A Z STAGE. With the palm lift, the fitter picks the palm's
    # height against the tool lying on the table and the grip depth comes out right by
    # construction. With a fixed palm and the tool on a platform, the platform's height IS the
    # grip depth, and an error in it is an error in the squeeze -- which nothing downstream
    # corrects. It is the tolerance the bench adds that the lift schedule never had.
    for mm in (-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4):
        pts.append(("fine_z", f"{mm:+.1f}mm", {"place": (0.0, 0.0, mm / 1000.0)}))
    return pts


def axis_points() -> list[tuple[str, str, dict]]:
    """(axis, label, spec). One physical property moves per point; baseline sits in every axis."""
    pts: list[tuple[str, str, dict]] = [("baseline", "as-simulated", {})]
    for v in (-0.006, -0.003, 0.003, 0.006):
        pts.append(("place_x", f"{v*1000:+.0f}mm", {"place": (v, 0.0, 0.0)}))
        pts.append(("place_y", f"{v*1000:+.0f}mm", {"place": (0.0, v, 0.0)}))
    for v in (-15.0, -7.0, 7.0, 15.0):
        pts.append(("place_yaw", f"{v:+.0f}deg", {"yaw": np.radians(v)}))
    for v in (0.5, 0.75, 1.5, 2.0, 2.5):
        pts.append(("mass", f"x{v}", {"mass": v}))
    for v in (0.88, 0.94, 1.06, 1.12):
        pts.append(("radius", f"x{v}", {"radius": v}))
    for v in (0.4, 0.6, 1.4, 2.0):
        pts.append(("friction", f"x{v}", {"friction": v}))
    for v in (0.99, 0.997, 0.999):
        pts.append(("solimp", f"dmax{v}", {"solimp": v}))
    for v in (0.003, 0.012, 0.020):
        pts.append(("solref", f"tc{v}", {"solref": v}))
    for v in (0.25, 0.5, 2.0, 4.0):
        pts.append(("servo_kp", f"x{v}", {"kp": v}))
    for v in (0.25, 0.4, 0.6, 2.0):
        pts.append(("servo_torque", f"x{v}", {"torque": v}))
    for v in (3.0, 10.0):
        pts.append(("servo_damping", f"x{v}", {"damping": v}))
    for v in (1.0, 2.0, 4.0, 7.0):
        pts.append(("ctrl_bias", f"{v:.0f}deg", {"bias_deg": v}))
    for v in (1.0, 2.0, 4.0):
        pts.append(("mount_err", f"{v:.0f}mm", {"mount_mm": v}))
    return pts


def ensemble_draw(rng: np.random.Generator, level: float) -> dict:
    """Everything wrong at once, at `level` x the plausible hardware error.

    Distributions are deliberately not the one-at-a-time grid: they are what a bench actually
    looks like -- placement gaussian in mm and degrees, material properties log-uniform because
    a friction coefficient is a multiplicative guess, servo bias gaussian per joint.
    """
    lg = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))  # noqa: E731
    mid = lambda lo, hi: 1.0 + (lg(lo, hi) - 1.0) * level                   # noqa: E731
    return {
        "place": (float(rng.normal(0, 0.003 * level)), float(rng.normal(0, 0.003 * level)), 0.0),
        "yaw": float(rng.normal(0, np.radians(8.0 * level))),
        "mass": mid(0.6, 2.0), "radius": mid(0.92, 1.08),
        "friction": mid(0.45, 1.8), "solimp": float(np.clip(
            0.995 + rng.uniform(-0.01, 0.004) * level, 0.975, 0.9992)),
        "solref": float(np.clip(0.006 * mid(0.6, 2.5), 0.002, 0.03)),
        "kp": mid(0.3, 2.5), "torque": mid(0.35, 1.5), "damping": mid(1.0, 6.0),
        "bias_deg": float(abs(rng.normal(0, 2.0 * level))),
        "mount_mm": float(abs(rng.normal(0, 1.5 * level))),
        "mount_seed": int(rng.integers(0, 10_000)),
    }


# --------------------------------------------------------------------------------------------
# cell mode: the operating point itself is a free variable
# --------------------------------------------------------------------------------------------
# THE WINNERS ARE RESONANCES. The search picked each design's cell by the highest held mean cos
# over a grid of pivot heights, and then measured that rv05_manual reads 0.991 and 0.000 at
# NEIGHBOURING pivot heights. A peak that narrow is a simulation artefact as far as hardware is
# concerned: axis_k is a commanded geometry and the built hand realises it with millimetres of
# error. So the shippable operating point is not the highest cell, it is the centre of the widest
# plateau -- which is what `--mode cell` measures, by scoring every cell under a small ensemble
# of wrong hands instead of under the nominal one.
def cell_grid(k_lo: float, k_hi: float, k_n: int, angles: list[float]) -> list[tuple]:
    return [(round(float(k), 3), float(a))
            for k in np.linspace(k_lo, k_hi, k_n) for a in angles]


def _cell_task(job: tuple) -> dict:
    (tag, scene, row, axis_k, angle, n_nom, n_ens, level, hold_steps,
     turn_steps, hold_squeeze, bench, selfcollision, retention_cfg, ensemble_seed,
     budget) = job
    # THE SQUEEZE IS PART OF THE GRASP, and it stops being a free constant once the pads are
    # flat. `fit` places the pad CENTRE at object_radius + PAD_RADIUS + gap - squeeze, which is
    # exact for a sphere and wrong for a flat face: the face's closest approach to the shaft is
    # along its own normal, and the finger meets the shaft at 20-35 deg to that normal, so the
    # same command leaves the face (R + 10.55)(1/cos t - 1) mm short. Measured on g14, grip
    # force falls 12.5 N -> 0.7 N at identical joint angles. Sweeping it is how that is paid for.
    squeeze = row.get("squeeze_mm", 4.0) / 1000.0
    scene = Path(scene)
    t0 = time.time()
    plan = make_plan(scene, straddle=row["straddle_mm"] / 1000.0,
                     depth=None if row["depth_req_mm"] is None else row["depth_req_mm"] / 1000.0,
                     thumb_axial=row["thumb_axial_mm"] / 1000.0, squeeze=squeeze,
                     axis_k=axis_k, angle_deg=angle, lift=0.10, budget=budget,
                     turn_steps=turn_steps, hold_squeeze=hold_squeeze, bench=bench)
    if plan is None:
        return {"design": tag, "axis_k": axis_k, "angle_deg": angle, "pose": False,
                "straddle_mm": row["straddle_mm"], "thumb_axial_mm": row["thumb_axial_mm"],
                "turn_steps": turn_steps, "hold_squeeze_mm": round(hold_squeeze * 1000, 1),
                "squeeze_mm": row.get("squeeze_mm", 4.0), "budget_rad": budget}
    servo_short, servo_worst = servo_shortfall(scene, plan, budget)
    nom, ens = [], []
    for rep in range(n_nom):
        nom.append(execute(scene, plan, hold_steps=hold_steps, seed=rep,
                           selfcollision=selfcollision, **retention_cfg))
    rng = np.random.default_rng(ensemble_seed)
    for i in range(n_ens):
        spec = ensemble_draw(rng, level)
        pert = mutate(scene, spec, SCRATCH)
        r = np.random.default_rng(7000 + i)
        bias = {j: float(np.radians(spec["bias_deg"]) * r.choice([-1.0, 1.0]))
                for js in FINGERS.values() for j in js}
        try:
            ens.append(execute(pert, plan, place=spec["place"], yaw=spec["yaw"],
                               ctrl_bias=bias, hold_steps=hold_steps, seed=100 + i,
                               **retention_cfg))
        except Exception:
            ens.append({"ok": False, "final_cos": 0.0, "min_z_hold": 0.0})
    held = lambda rs: [x["final_cos"] if x["ok"] else 0.0 for x in rs]  # noqa: E731
    clearances = [x["min_finger_clearance_mm"] for x in nom
                  if x.get("min_finger_clearance_mm") is not None]
    return {
        "design": tag, "axis_k": axis_k, "angle_deg": angle, "pose": True,
        "straddle_mm": row["straddle_mm"], "thumb_axial_mm": row["thumb_axial_mm"],
        "turn_steps": turn_steps, "hold_squeeze_mm": round(hold_squeeze * 1000, 1),
        "squeeze_mm": row.get("squeeze_mm", 4.0), "budget_rad": budget,
        "servo_short_deg": servo_short, "servo_worst": servo_worst,
        "nom_cos": round(float(np.mean(held(nom))), 3),
        "nom_sd": round(float(np.std(held(nom))), 3),
        "nom_kept": sum(1 for x in nom if x["ok"]), "n_nom": len(nom),
        "nom_proof_lift": sum(1 for x in nom if x.get("proof_lift_ok", True)),
        "nom_proof_hold": sum(1 for x in nom if x.get("proof_hold_ok", True)),
        "max_proof_slip_mm": round(max((x.get("proof_slip_mm", 0.0) for x in nom),
                                       default=0.0), 1),
        "proof_lift_mm": retention_cfg.get("proof_lift", 0.0) * 1000.0,
        "proof_max_slip_mm": retention_cfg.get("proof_max_slip", 0.0) * 1000.0,
        "load_target_units": retention_cfg.get("load_target_units", 0.0),
        "turn_torque_limit_nm": retention_cfg.get("turn_torque_limit_nm", 0.0),
        "hold_torque_limit_nm": retention_cfg.get("hold_torque_limit_nm", 0.0),
        "terminal_load_units": {
            f: round(float(np.mean([x.get("load_units", {}).get(f, 0.0) for x in nom])), 1)
            for f in FINGERS},
        "min_finger_clearance_mm": min(clearances) if clearances else None,
        "ens_cos": round(float(np.mean(held(ens))), 3) if ens else None,
        "ens_sd": round(float(np.std(held(ens))), 3) if ens else None,
        "ens_kept": sum(1 for x in ens if x["ok"]), "n_ens": len(ens),
        # the number to ship on: fraction of WRONG hands that both keep the shaft and stand it up
        "ens_win": round(sum(1 for x in ens if x["ok"] and x["final_cos"] >= 0.7)
                         / max(1, len(ens)), 3),
        "secs": round(time.time() - t0, 1),
    }


def _task(job: tuple) -> dict:
    design, scene, plan, axis, label, spec, rep, hold_steps = job
    t0 = time.time()
    scene = Path(scene)
    pert = mutate(scene, spec, SCRATCH)
    bias = {}
    if spec.get("bias_deg"):
        # A per-joint zero offset with a random SIGN: a common-mode bias would just be a
        # slightly different grasp, and it is the differential error that breaks a grip.
        r = np.random.default_rng(7000 + rep * 97 + hash(label) % 997)
        bias = {j: float(np.radians(spec["bias_deg"]) * r.choice([-1.0, 1.0]))
                for js in FINGERS.values() for j in js}
    try:
        out = execute(pert, plan, place=spec.get("place", (0.0, 0.0, 0.0)),
                      yaw=spec.get("yaw", 0.0), ctrl_bias=bias, hold_steps=hold_steps,
                      seed=rep, force_target=float(spec.get("force_target", 0.0)),
                      force_gain=float(spec.get("force_gain", 0.0015)),
                      force_phase=str(spec.get("force_phase", "all")))
    except Exception as exc:  # a mutated scene that will not compile is a result, not a crash
        out = {"error": f"{type(exc).__name__}: {exc}", "ok": False, "final_cos": 0.0,
               "peak_cos": 0.0, "min_z_hold": 0.0, "contacts_hand": 0}
    return {"design": design, "axis": axis, "label": label, "rep": rep,
            "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()},
            "secs": round(time.time() - t0, 1), **out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", default=DEFAULT_DESIGNS)
    ap.add_argument("--designs-file", type=Path, default=None,
                    help="optional newline/comma-separated design names; avoids an oversized "
                         "command line for population-scale screens")
    ap.add_argument("--design-table", type=Path, default=TABLE,
                    help="joined per-design table; defaults to the original 108-hand search")
    ap.add_argument("--design-manifest", type=Path, default=None,
                    help="optional real-v1 sampler manifest adding design vectors not in the "
                         "legacy named sets")
    ap.add_argument("--generated-dir", type=Path, default=ds.GEN,
                    help="where rigid scenes for manifest designs are generated")
    ap.add_argument("--flat-pads", action="store_true",
                    help="use the BUILT finger cross-section (14.8 mm across, flat face at "
                         "10.55 mm) instead of the shipped 21.1 mm round capsules")
    ap.add_argument("--pad-width-mm", type=float, default=14.8,
                    help="the flat pad's width ACROSS the finger. The built part is 14.8; a wider "
                         "printed tip is a free change, so this is a design knob, not a fact.")
    ap.add_argument("--flat-links", action="store_true",
                    help="also flatten the phalanges. Off by default: measured contact at a "
                         "settled grasp is entirely on the tips, so the links are left round.")
    ap.add_argument("--pad-len-mm", type=float, default=14.8,
                    help="the fingertip pad's extent along the finger")
    ap.add_argument("--post-y", type=float, default=0.0,
                    help="mm. Where along the tool the bench support sits. 0 is under its middle, "
                         "which is where the descending end of a -90 deg turn ends up.")
    ap.add_argument("--bench-height", type=float, default=0.0,
                    help="metres. >0 stands the tool on a platform at that height and runs the "
                         "schedule with a FIXED palm (no lift), which is the prototype bench.")
    ap.add_argument("--object", default="medium", choices=tuple(OBJECTS),
                    help="which tool the hand is holding; the plan is fitted to it")
    ap.add_argument("--straddle-mm", default=None,
                    help="override the table's straddle (the stubby's handle is a different "
                         "diameter and length, so the search's straddle need not fit it)")
    ap.add_argument("--thumb-axial-mm", default=None,
                    help="comma list; in --mode cell it is swept with the pivot height")
    ap.add_argument("--mode", default="both", choices=("axes", "ensemble", "both", "cell"))
    ap.add_argument("--cell-k", default="0.05,0.55,11", help="axis_k lo,hi,n for --mode cell")
    ap.add_argument("--cell-angles", default="-80,-90,-100,-110")
    ap.add_argument("--cell-nom", type=int, default=3)
    ap.add_argument("--cell-ens", type=int, default=8)
    ap.add_argument("--cell-level", type=float, default=1.0)
    ap.add_argument("--turn-steps", default="550", help="comma list; swept in --mode cell")
    ap.add_argument("--budget", default="0.5",
                    help="comma list of per-joint |delta| caps in RADIANS, swept in --mode "
                         "cell. 0.5 rad = 28.6 deg is Policy B's RESIDUAL ACTION BUDGET and "
                         "was inherited here by accident: every screen and every ranking in "
                         "this program so far was measured at it. A design holds only inside "
                         "a contiguous BAND of clips and drops on both sides, so screening at "
                         "one value scores each hand at a point that may be nowhere near its "
                         "own band -- see docs/experiments/20260830-real_v1-sobol128/deploy/.")
    ap.add_argument("--squeeze-mm", default="4.0",
                    help="comma list; how far inside the shaft surface the pads are driven")
    ap.add_argument("--hold-squeeze-mm", default="0", help="comma list; swept in --mode cell")
    ap.add_argument("--repeats", type=int, default=3,
                    help="rollouts per point. The good cells are narrow resonances and CPU "
                         "contact solves vary with the settle, so n=1 is not a measurement.")
    ap.add_argument("--draws", type=int, default=40, help="ensemble draws per design per level")
    ap.add_argument("--force-target", default="0",
                    help="comma list of per-finger normal-force set-points, N. 0 = the plain "
                         "open-loop plan. Anything else closes a loop on servo current alone.")
    ap.add_argument("--force-gain", type=float, default=0.0015)
    ap.add_argument("--ensemble-seed", type=int, default=20260829,
                    help="RNG for the ensemble draws. A cell picked as the best of thousands on "
                         "16 draws is partly picked on noise, so the number quoted for it has to "
                         "come from a DIFFERENT set of draws than the one that selected it.")
    ap.add_argument("--force-phase", default="all",
                    help="comma list of all|hold — whether the current loop acts through the "
                         "turn or only once the turn has finished")
    ap.add_argument("--levels", default="0.5,1.0", help="ensemble severity multipliers")
    ap.add_argument("--hold-steps", type=int, default=800)
    ap.add_argument("--load-target-units", type=float, default=0.0,
                    help="per-finger SCS0009-like load target; 1000 ~= published stall torque")
    ap.add_argument("--load-gain", type=float, default=0.0024)
    ap.add_argument("--capture-steps", type=int, default=0,
                    help="post-turn load-clamp settling steps before the proof lift")
    ap.add_argument("--proof-lift-mm", type=float, default=0.0,
                    help="raise the palm after capture; the object must follow to pass")
    ap.add_argument("--proof-lift-steps", type=int, default=700)
    ap.add_argument("--proof-max-slip-mm", type=float, default=10.0,
                    help="maximum vertical object slip during the post-lift free hold")
    ap.add_argument("--turn-torque-limit-nm", type=float, default=0.0,
                    help=f"absolute finger torque cap; SCS0009 80%% overload = "
                         f"{SCS0009_OVERLOAD_TORQUE_NM:.4f} N m")
    ap.add_argument("--hold-torque-limit-nm", type=float, default=0.0,
                    help=f"absolute long-hold cap; SCS0009 rated = "
                         f"{SCS0009_RATED_TORQUE_NM:.4f} N m")
    ap.add_argument("--selfcollision", action="store_true",
                    help="trace dynamic finger clearance over nominal cell rollouts")
    ap.add_argument("--dense-place", action="store_true",
                    help="replace the one-at-a-time axes with a 1 mm / 2 deg placement map")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", type=Path,
                    default=ROOT / f"docs/experiments/{DATE}-real_v1_deploy/envelope.json")
    ap.add_argument("--cells-from", type=Path, default=None,
                    help="a cells_*.json. Each design's operating point is taken from its best "
                         "cell there instead of from the 108-hand search table. Needed whenever "
                         "the schedule has changed -- the lift-tuned cell drops the tool on the "
                         "fixed-palm bench, so a tolerance map built on it measures nothing.")
    ap.add_argument("--smoke", action="store_true", help="baseline only, n=1, no pool")
    args = ap.parse_args()

    table = {r["design"]: r for r in json.loads(args.design_table.read_text())}
    # ON THE BENCH THE GRIP DEPTH HAS TO BE STATED. With the tool on the table `fit` searches a
    # palm-height window around the palm body's own z; with the tool 100 mm up, that window does
    # not contain a reachable pose and every design reports "no pose". The design's own fitted
    # depth is the right thing to ask for, and it is already in the table.
    if args.bench_height > 0:
        for r in table.values():
            if r.get("depth_req_mm") is None and r.get("depth_fit_mm"):
                r["depth_req_mm"] = r["depth_fit_mm"]
    if args.cells_from and args.cells_from.exists():
        best: dict[str, dict] = {}
        for c in json.loads(args.cells_from.read_text()):
            if not c.get("pose") or c.get("nom_kept", 0) < c.get("n_nom", 1) / 2:
                continue
            k = c["design"]
            if k not in best or (c.get("ens_win", 0), c["nom_cos"]) > \
                                (best[k].get("ens_win", 0), best[k]["nom_cos"]):
                best[k] = c
        for k, c in best.items():
            if k in table:
                table[k] = dict(table[k], axis_k=c["axis_k"], angle_deg=c["angle_deg"],
                                straddle_mm=c["straddle_mm"],
                                thumb_axial_mm=c["thumb_axial_mm"],
                                squeeze_mm=c.get("squeeze_mm", 4.0))
        print(f"operating points taken from {args.cells_from.name} for {len(best)} designs")

    design_text = args.designs_file.read_text() if args.designs_file else args.designs
    designs = [t.strip() for t in design_text.replace("\n", ",").split(",") if t.strip()]
    vecs = ds.design_set("all")
    if args.design_manifest:
        manifest = json.loads(args.design_manifest.read_text())
        vecs.update({r["design"]: tuple(r["vector_m"]) for r in manifest["designs"]})
    missing = sorted(set(designs) - set(vecs))
    if missing:
        raise SystemExit(f"design vectors missing for: {', '.join(missing)}")

    if args.mode == "cell":
        lo, hi, n = args.cell_k.split(",")
        grid = cell_grid(float(lo), float(hi), int(n),
                         [float(v) for v in args.cell_angles.split(",")])
        jobs = []
        # THE GRASP GEOMETRY IS PART OF THE OPERATING POINT, not a constant. The stubby's
        # 50.8 mm handle cannot take the 32 mm straddle the 100 mm shaft was fitted at (`fit`
        # caps spread at half_len - 5 mm), and where the thumb sits along the handle is the knob
        # that decided the whole 108-hand search. Both are swept with the pivot height.
        straddles = ([float(v) for v in str(args.straddle_mm).split(",")]
                     if args.straddle_mm else [None])
        thumb_axials = ([float(v) for v in str(args.thumb_axial_mm).split(",")]
                        if args.thumb_axial_mm else [None])
        # CONTROLLER knobs, free on hardware: how slowly the turn is commanded, and whether the
        # pads are re-seated at the top of it. Neither is a property of the hand.
        turn_stepss = [int(v) for v in str(args.turn_steps).split(",")]
        budgets = [float(v) for v in str(args.budget).split(",")]
        squeezes = [float(v) for v in str(args.squeeze_mm).split(",")]
        hold_squeezes = [float(v) / 1000.0 for v in str(args.hold_squeeze_mm).split(",")]
        retention_cfg = {
            "load_target_units": args.load_target_units,
            "load_gain": args.load_gain,
            "capture_steps": args.capture_steps,
            "proof_lift": args.proof_lift_mm / 1000.0,
            "proof_lift_steps": args.proof_lift_steps,
            "proof_max_slip": args.proof_max_slip_mm / 1000.0,
            "turn_torque_limit_nm": args.turn_torque_limit_nm,
            "hold_torque_limit_nm": args.hold_torque_limit_nm,
        }
        for tag in designs:
            row = table.get(tag)
            if row is None or not row.get("graspable"):
                print(f"{tag:14} no row / not graspable — skipped")
                continue
            scene = object_scene(ds.scene_for(vecs[tag], args.generated_dir), args.object,
                                 args.bench_height, args.post_y / 1000.0,
                                 args.flat_pads, args.pad_len_mm / 1000.0,
                                 args.pad_width_mm / 1000.0, args.flat_links)
            for st in straddles:
                for ta in thumb_axials:
                    r2 = dict(row)
                    if st is not None:
                        r2["straddle_mm"] = st
                    if ta is not None:
                        r2["thumb_axial_mm"] = ta
                    for sq in squeezes:
                      r2 = dict(r2, squeeze_mm=sq)
                      for k, a in grid:
                        for ts in turn_stepss:
                            for hs in hold_squeezes:
                              for bg in budgets:
                                jobs.append((tag, str(scene), r2, k, a, args.cell_nom,
                                             args.cell_ens, args.cell_level, args.hold_steps,
                                             ts, hs, args.bench_height > 0.0,
                                             args.selfcollision, retention_cfg,
                                             args.ensemble_seed, bg))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        done = json.loads(args.out.read_text()) if args.out.exists() else []
        seen = {(r["design"], r.get("straddle_mm"), r.get("thumb_axial_mm"), r["axis_k"],
                 r["angle_deg"], r.get("turn_steps"), r.get("hold_squeeze_mm"),
                 r.get("squeeze_mm", 4.0), r.get("budget_rad", 0.5)) for r in done}
        jobs = [j for j in jobs
                if (j[0], j[2]["straddle_mm"], j[2]["thumb_axial_mm"], j[3], j[4], j[9],
                    round(j[10] * 1000, 1), j[2].get("squeeze_mm", 4.0), j[15]) not in seen]
        print(f"{len(jobs)} cells ({args.cell_nom} nominal + {args.cell_ens} ensemble each) "
              f"on {args.workers} workers")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(_cell_task, jobs, chunksize=1)):
                done.append(r)
                if i % 25 == 0 or i == len(jobs) - 1:
                    el = time.time() - t0
                    print(f"  [{i+1}/{len(jobs)}] {el/60:5.1f} min, "
                          f"{el/(i+1)*(len(jobs)-i-1)/60:5.1f} left — {r['design']} "
                          f"k{r['axis_k']} a{r['angle_deg']:.0f} b{r.get('budget_rad')} "
                          f"nom {r.get('nom_cos')} kept {r.get('nom_kept')}/{r.get('n_nom')} "
                          f"servo+{r.get('servo_short_deg')} "
                          f"ens_win {r.get('ens_win')}", flush=True)
                    args.out.write_text(json.dumps(done, indent=1))
        args.out.write_text(json.dumps(done, indent=1))
        print(f"wrote {args.out}  ({len(done)} cells, {(time.time()-t0)/60:.1f} min)")
        return 0

    plans: dict[str, tuple[Path, dict]] = {}
    for tag in designs:
        row = table.get(tag)
        if row is None or not row.get("graspable"):
            print(f"{tag:14} no row / not graspable — skipped")
            continue
        scene = object_scene(ds.scene_for(vecs[tag], args.generated_dir), args.object,
                             args.bench_height, args.post_y / 1000.0,
                                 args.flat_pads, args.pad_len_mm / 1000.0,
                                 args.pad_width_mm / 1000.0, args.flat_links)
        row = dict(row)
        if args.straddle_mm:
            row["straddle_mm"] = float(str(args.straddle_mm).split(",")[0])
        if args.thumb_axial_mm:
            row["thumb_axial_mm"] = float(str(args.thumb_axial_mm).split(",")[0])
        t0 = time.time()
        plan = make_plan(scene, straddle=row["straddle_mm"] / 1000.0,
                         depth=None if row["depth_req_mm"] is None else row["depth_req_mm"] / 1000.0,
                         thumb_axial=row["thumb_axial_mm"] / 1000.0,
                         squeeze=row.get("squeeze_mm", 4.0) / 1000.0,
                         axis_k=row["axis_k"], angle_deg=row["angle_deg"], lift=0.10,
                         budget=float(str(args.budget).split(",")[0]),
                         bench=args.bench_height > 0.0,
                         turn_steps=int(str(args.turn_steps).split(",")[0]),
                         hold_squeeze=float(str(args.hold_squeeze_mm).split(",")[0]) / 1000.0)
        if plan is None:
            print(f"{tag:14} the fit found no pose — skipped")
            continue
        plans[tag] = (scene, plan)
        pj = args.out.parent / "plans" / f"{tag}.json"
        pj.parent.mkdir(parents=True, exist_ok=True)
        pj.write_text(json.dumps(plan, indent=1))
        print(f"{tag:14} planned in {time.time()-t0:5.1f}s  straddle {row['straddle_mm']:.1f}mm "
              f"thAx {row['thumb_axial_mm']:.0f}mm k {row['axis_k']} ang {row['angle_deg']:.0f} "
              f"-> {pj}")

    jobs: list[tuple] = []
    for tag, (scene, plan) in plans.items():
        if args.smoke:
            jobs.append((tag, str(scene), plan, "baseline", "as-simulated", {}, 0, args.hold_steps))
            continue
        if args.mode in ("axes", "both"):
            for ft in (float(v) for v in args.force_target.split(",")):
                for axis, label, spec in (dense_place_points() if args.dense_place
                                          else axis_points()):
                    spec = dict(spec, force_target=ft, force_gain=args.force_gain)
                    ax = axis if ft == 0 else f"{axis}_F{ft:g}"
                    for rep in range(args.repeats):
                        jobs.append((tag, str(scene), plan, ax, label, spec, rep,
                                     args.hold_steps))
        if args.mode in ("ensemble", "both"):
            for ft, fp_ in itertools.product(
                    (float(v) for v in args.force_target.split(",")),
                    args.force_phase.split(",")):
                for lv in (float(v) for v in args.levels.split(",")):
                    # SAME DRAWS FOR EVERY FORCE TARGET: the seed is reset per arm so the
                    # regulator is compared against the open loop on the identical set of wrong
                    # hands, not on a fresh sample of them.
                    rng = np.random.default_rng(args.ensemble_seed)
                    for i in range(args.draws):
                        spec = ensemble_draw(rng, lv)
                        spec["force_target"] = ft
                        spec["force_gain"] = args.force_gain
                        spec["force_phase"] = fp_
                        jobs.append((tag, str(scene), plan,
                                     f"ensemble{lv}_F{ft:g}{'' if ft == 0 else fp_[0]}",
                                     f"draw{i:02d}", spec, i, args.hold_steps))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: list[dict] = []
    if args.out.exists():
        done = json.loads(args.out.read_text())
        seen = {(r["design"], r["axis"], r["label"], r["rep"]) for r in done}
        jobs = [j for j in jobs if (j[0], j[3], j[4], j[6]) not in seen]
        print(f"resuming: {len(done)} already recorded, {len(jobs)} to run")

    print(f"{len(jobs)} rollouts on {args.workers} workers")
    t0 = time.time()
    if args.smoke or args.workers <= 1:
        for i, j in enumerate(jobs):
            r = _task(j)
            done.append(r)
            print(f"  {r['design']:14} {r['axis']:14} {r['label']:12} "
                  f"cos {r['final_cos']:+.3f} z {r['min_z_hold']:.3f} ok {r['ok']} "
                  f"({r['secs']}s)")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(_task, jobs, chunksize=1)):
                done.append(r)
                if i % 250 == 0 or i == len(jobs) - 1:
                    el = time.time() - t0
                    eta = el / (i + 1) * (len(jobs) - i - 1)
                    print(f"  [{i+1}/{len(jobs)}] {el/60:5.1f} min elapsed, "
                          f"{eta/60:5.1f} min left — last {r['design']} {r['axis']} "
                          f"{r['label']} cos {r['final_cos']:+.3f} ok {r['ok']}", flush=True)
                    args.out.write_text(json.dumps(done, indent=1))
    args.out.write_text(json.dumps(done, indent=1))
    print(f"wrote {args.out}  ({len(done)} rows, {(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
