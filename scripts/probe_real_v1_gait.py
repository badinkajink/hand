"""Can three fingers SPIN the shaft about its own axis once it is standing on the ground?

Every reorientation result in this program ends the same way: the shaft is vertical, the grip is
fixed, and the contacts have not moved on the material since the grasp closed
(docs/experiments/REORIENT_PRIMITIVE.txt; probe_real_v1_carry.py reaches cos 0.996 that way).
Turning a screwdriver is the opposite problem -- unbounded rotation about the tool's own axis,
which three fingers with ~12 mm of tangential stroke cannot deliver in one go. It needs a GAIT:
advance the contacts, let go, put them back, advance again.

Letting go is what the program has never done, and the reason is that a released shaft falls. The
premise this probe tests is that STANDING THE SHAFT ON THE GROUND removes that failure mode: a
100 mm x 12.5 mm cylinder resting on its end face is statically stable up to a tilt of
atan(12.5/50) = 14.0 deg, so the fingers may leave it entirely and find it where they left it.

    ground contact is the enabling constraint AND the load.

Pressing the shaft down buys stability during the release and costs torque during the twist: the
rim contacts resist spin with roughly mu * N * r_obj, and N is set by how hard the fingers push
the shaft into the floor. `--press` sweeps exactly that, through zero (just touching) into
NEGATIVE values, where the fingers lift the shaft clear of the floor and the gait is in-air --
the control condition, and the one the ground-support premise predicts will fail.

WHAT IT COMMANDS. Three pads on a ring of radius r_obj + r_pad - squeeze around the standing
shaft, at azimuths thumb=180, index=+60, middle=-60 deg (the tripod probe_real_v1_vertical_hold.py
uses). A cycle is:

    TWIST    all three azimuths advance by --stroke, carrying the shaft with them
    RELEASE  the pads retract radially by --release
    RETURN   the azimuths run back to the start while retracted
    REGRASP  the pads close again

with `--relay` doing RELEASE/RETURN/REGRASP one finger at a time so the shaft is never fully
free. The joint targets are solved ONCE into a (radius, azimuth) table and then played back, so
a rollout is pure mj_step and the command stream is a set-point list of the same shape the bench
already runs (real_v1_export_plan.py). Nothing here is a policy and nothing here reads the
object.

WHAT IT SCORES. Net rotation about the shaft's own axis, integrated from the body's angular
velocity so it is UNWRAPPED and can exceed 360 deg -- a cylinder has no yaw datum to read off its
quaternion. Reported per cycle and as an efficiency against the commanded stroke. Alongside it,
because a spinning shaft that has walked out of the hand or fallen over is not a result:
tilt from vertical, xy drift, bottom height, ground normal force and the per-finger friction-cone
utilisation.

    uv run python scripts/probe_real_v1_gait.py \
        --morph-run results/phase1/real_v1/rv05_manual_stored \
        --press -2,0,2,4 --stroke 30 --cycles 8 --repeats 4
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from morphohand.tools.keyframe_ik import FINGERS, TIPS, ik_finger  # noqa: E402

# Thumb opposite the pair, pair at +/-60 deg: a tripod on the circle the standing shaft affords.
# Identical to probe_real_v1_vertical_hold.py -- do not re-derive it, the two must agree.
AZIMUTH = {"thumb": np.pi, "index": np.pi / 3, "middle": -np.pi / 3}
PAD_RADIUS = 0.010550
CONTROL_DECIMATION = 10          # 10 sim steps at dt 0.002 = one 50 Hz control step


# --------------------------------------------------------------------------------------- scene

def _obj_geom(m, obj: str):
    gid = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(obj).id][0]
    return float(m.geom_size[gid, 0]), float(m.geom_size[gid, 1])   # radius, half length


def _stand(m, d, obj: str, cx: float, cy: float, gap: float = 0.0):
    """Put the shaft upright on the floor at (cx, cy), bottom face `gap` above z=0."""
    _, half = _obj_geom(m, obj)
    adr = m.jnt_qposadr[m.body(obj).jntadr[0]]
    d.qpos[adr:adr + 3] = [cx, cy, half + gap]
    d.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]     # local +z along world +z
    dadr = m.jnt_dofadr[m.body(obj).jntadr[0]]
    d.qvel[dadr:dadr + 6] = 0.0


def _palm_to(m, d, z: float):
    """Hold the palm plane at world height `z`, everything else at the keyframe pose."""
    body_z = float(m.body("palm_pose").pos[2])
    for j, val in (("palm_pz", z - body_z),):
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = val
    for a in range(m.nu):
        name = m.actuator(a).name
        if name.startswith("a_palm_"):
            jid = m.actuator_trnid[a, 0]
            d.ctrl[a] = float(d.qpos[m.jnt_qposadr[jid]])


def _finger_act(m):
    return {j: next(k for k in range(m.nu) if m.actuator(k).name == f"a_{j}")
            for joints in FINGERS.values() for j in joints}


# ------------------------------------------------------------------------------- measurements

def _ground(m, d, obj: str):
    """Contacts between the shaft and the world body (the floor plane). Count and normal force."""
    n, tot = 0, 0.0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        names = [m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name]
        if obj in names and "world" in names:
            mujoco.mj_contactForce(m, d, i, f6)
            n += 1
            tot += abs(float(f6[0]))
    return n, tot


def _hand(m, d, obj: str):
    """Object contacts against ANY hand body, and the pad-only subset.

    Pad-only is the grasp metric; whole-hand is the retention metric. probe_real_v1_carry.py
    scored six cells `dropped` on a pad-only count while the shaft sat in the air on the middle
    phalanges, so both are reported here.
    """
    nh, fh, np_, fp_ = 0, 0.0, 0, 0.0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        names = [m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name]
        if obj not in names:
            continue
        other = names[0] if names[1] == obj else names[1]
        if other.split("_")[0] not in ("thumb", "index", "middle", "palm"):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        nh += 1
        fh += abs(float(f6[0]))
        if other in TIPS.values():
            np_ += 1
            fp_ += abs(float(f6[0]))
    return nh, fh, np_, fp_


def _util(m, d, obj: str) -> dict:
    """Per-finger friction-cone utilisation |f_t| / (mu f_n) at the pad."""
    out = {f: {"fn": 0.0, "ft": 0.0, "util": 0.0} for f in FINGERS}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        names = {m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name}
        if obj not in names:
            continue
        for f in FINGERS:
            if TIPS[f] in names:
                mujoco.mj_contactForce(m, d, i, f6)
                fn, ft = abs(float(f6[0])), float(np.linalg.norm(f6[1:3]))
                out[f]["fn"] += fn
                out[f]["ft"] += ft
                out[f]["util"] = ft / (float(c.friction[0]) * fn) if fn > 1e-6 else 0.0
    return out


def _axis_tilt(m, d, obj: str):
    """Shaft axis in world, and its tilt from vertical in degrees."""
    ax = d.body(obj).xmat.reshape(3, 3)[:, 2].copy()
    return ax, float(np.degrees(np.arccos(np.clip(abs(ax[2]), -1.0, 1.0))))


# ------------------------------------------------------------------------------------- command

def _ring_table(m, centre, z, palm_z, radii, phis, iters=200):
    """Solve each pad onto (radius, azimuth) ONCE; the gait then plays joint targets back.

    Doing IK inside the rollout costs an mj_forward per iteration per finger per control step and
    dominates the probe. It is also the wrong model of the hardware, which runs a set-point list.
    Returns {finger: array[len(radii), len(phis), 3]} of joint angles, plus the worst residual.
    """
    mik = mujoco.MjModel.from_xml_path(m_path_of(m))
    dik = mujoco.MjData(mik)
    mujoco.mj_resetDataKeyframe(mik, dik, mik.key("open_ik").id)
    _palm_to(mik, dik, palm_z)
    mujoco.mj_forward(mik, dik)

    table = {f: np.zeros((len(radii), len(phis), 3)) for f in FINGERS}
    worst = 0.0
    per_r = [0.0] * len(radii)
    for f, joints in FINGERS.items():
        qadr = [mik.jnt_qposadr[mik.joint(j).id] for j in joints]
        for ri, r in enumerate(radii):
            # Warm start each radius from the nominal azimuth, then walk the sweep in order so
            # consecutive solves are millimetres apart and 200 iterations is generous.
            for pi, phi in enumerate(phis):
                a = AZIMUTH[f] + phi
                tgt = np.array([centre[0] + r * np.cos(a), centre[1] + r * np.sin(a), z])
                res = ik_finger(mik, dik, f, tgt, iters=iters)
                worst = max(worst, res)
                per_r[ri] = max(per_r[ri], res)
                table[f][ri, pi] = [float(dik.qpos[q]) for q in qadr]
    return table, worst, per_r


_MODEL_PATH = {}
_GRIP_DEPTH = [0.045]


def m_path_of(m):
    return _MODEL_PATH["path"]


def _lookup(table, f, ri, phi, phis):
    """Linear interpolation of the joint table along the azimuth axis."""
    x = np.interp(phi, phis, np.arange(len(phis)))
    i0, i1 = int(np.floor(x)), min(int(np.floor(x)) + 1, len(phis) - 1)
    w = x - i0
    return table[f][ri, i0] * (1 - w) + table[f][ri, i1] * w


# --------------------------------------------------------------------------------------- gait

def gait(scene: Path, press_mm: float, stroke_deg: float, cycles: int, squeeze: float,
         release_mm: float, grip_z: float, grip_depth: float, obj: str, centre_x: float = 0.004,
         relay: bool = False, twist_steps: int = 120, move_steps: int = 60,
         settle_steps: int = 400, approach_steps: int = 200, jitter: float = 0.0,
         seed: int = 0, no_floor: bool = False, pad_radius: float | None = None, video: Path | None = None, film: Path | None = None,
         trace: bool = False, cam=(120.0, -18.0, 0.32)):
    _MODEL_PATH["path"] = str(scene)
    _GRIP_DEPTH[0] = grip_depth
    m = mujoco.MjModel.from_xml_path(str(scene))
    floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    pad_r = PAD_RADIUS if pad_radius is None else float(pad_radius)
    if pad_radius is not None:
        # Resize the pad spheres in place. The tips are spheres centred on the tip body, so the
        # contact stays on the tip and no reach renormalisation is needed -- unlike the
        # fingertip-shape sweep, where six of eight candidates were buried inside the distal
        # capsule and never touched the object.
        for f in FINGERS:
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == m.body(TIPS[f]).id:
                    m.geom_size[g, 0] = pad_r
                    m.geom_rbound[g] = pad_r
    d = mujoco.MjData(m)
    r_obj, half = _obj_geom(m, obj)

    mujoco.mj_resetDataKeyframe(m, d, m.key("open_ik").id)
    palm_z = grip_z + grip_depth
    _palm_to(m, d, palm_z)
    mujoco.mj_forward(m, d)
    centre = d.body("palm_pose").xpos.copy()[:2] + np.array([centre_x, 0.0])

    rng = np.random.default_rng(seed)
    _stand(m, d, obj, centre[0] + (rng.normal(0, jitter) if jitter else 0.0),
           centre[1] + (rng.normal(0, jitter) if jitter else 0.0))
    mujoco.mj_forward(m, d)

    # The commanded ring sits `squeeze` INSIDE the shaft surface: these are position servos and
    # the grip force IS the commanded-minus-actual error. Commanding the pad exactly onto the
    # surface zeroes it and the hand lets go (REORIENT_PRIMITIVE.txt, the rule that cost the
    # carry probe three wrong answers).
    r_grip = r_obj + pad_r - squeeze
    r_open = r_obj + pad_r + release_mm / 1000.0
    r_wide = r_obj + pad_r + 0.018
    gear = r_grip / r_obj
    stroke = np.radians(stroke_deg)
    phis = np.linspace(0.0, stroke, 25)
    # press>0 commands the ring BELOW the material point it grips, pushing the shaft into the
    # floor through the grip; press<0 lifts it clear and the gait becomes in-air.
    z_ring = grip_z - press_mm / 1000.0
    table, ik_res, per_r = _ring_table(m, centre, z_ring, palm_z, [r_grip, r_open, r_wide], phis)
    # The approach happens at grip_z; the press is a ramp from that pose to the same ring at
    # z_ring. With press = 0 the two tables coincide and the ramp is a no-op.
    app, _, _ = (table, None, None) if abs(press_mm) < 1e-9 else \
        _ring_table(m, centre, grip_z, palm_z, [r_grip, r_open, r_wide], phis)

    acts = _finger_act(m)
    open_ctrl = {j: float(d.ctrl[a]) for j, a in acts.items()}

    vid, vcam, frames, shots = None, None, [], []
    if video is not None or film is not None:
        vid = mujoco.Renderer(m, height=480, width=640)
        vcam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, vcam)
        vcam.azimuth, vcam.elevation, vcam.distance = cam

    step_i = [0]
    spin = [0.0]                 # integrated rotation about the shaft's OWN axis, unwrapped
    spin_w = [0.0]               # the same integral onto WORLD +z: the roll/twist control
    rows: list = []
    peak_tilt = [0.0]
    lost = [False]

    def _cmd(f, ri, phi):
        q = _lookup(table, f, ri, phi, phis)
        for k, j in enumerate(FINGERS[f]):
            d.ctrl[acts[j]] = float(q[k])

    def _integrate():
        """Every sim step: the shaft's rotation about its own axis, and onto world +z."""
        ax = d.body(obj).xmat.reshape(3, 3)[:, 2]
        dadr = m.jnt_dofadr[m.body(obj).jntadr[0]]
        w_world = d.body(obj).xmat.reshape(3, 3) @ d.qvel[dadr + 3:dadr + 6]
        spin[0] += float(np.dot(w_world, ax)) * m.opt.timestep
        spin_w[0] += float(w_world[2]) * m.opt.timestep

    def _sample():
        ax, tilt = _axis_tilt(m, d, obj)
        # A cylinder is rotationally symmetric, so its quaternion carries no yaw datum; the
        # rotation is integrated from the body's angular velocity in `_integrate`, at the sim
        # rate. This sampler only carries the health checks.
        peak_tilt[0] = max(peak_tilt[0], tilt)
        pos = d.body(obj).xpos
        if tilt > 45.0 or pos[2] < half * 0.5:
            lost[0] = True

    phase = {"twist": 0.0, "release": 0.0, "return": 0.0, "regrasp": 0.0, "settle": 0.0}
    phase_contact = {k: [] for k in phase}
    phase_ground = {k: [] for k in phase}
    phase_any = {k: [] for k in phase}

    def _run(n, before=None, tag=None):
        s0 = spin[0]
        for k in range(n):
            if before is not None and step_i[0] % CONTROL_DECIMATION == 0:
                before(k)
            mujoco.mj_step(m, d)
            _integrate()
            step_i[0] += 1
            if step_i[0] % CONTROL_DECIMATION == 0:
                _sample()
                if tag is not None:
                    _h = _hand(m, d, obj)
                    phase_contact[tag].append(_h[2])
                    phase_any[tag].append(_h[0])
                    phase_ground[tag].append(_ground(m, d, obj)[0])
            if vid is not None and step_i[0] % 12 == 0:
                vcam.lookat[:] = [centre[0], centre[1], half]
                vid.update_scene(d, vcam)
                _mark(vid.scene, m, d, obj, r_obj, half)
                frames.append(vid.render())
        if tag is not None:
            phase[tag] += spin[0] - s0

    # --- settle the standing shaft with the hand open, then close onto the ring
    _run(settle_steps // 2)
    ctrl0 = {j: float(d.ctrl[a]) for j, a in acts.items()}
    wide = {f: _lookup(app, f, 2, 0.0, phis) for f in FINGERS}
    tgt_app = {f: _lookup(app, f, 0, 0.0, phis) for f in FINGERS}
    tgt0 = {f: _lookup(table, f, 0, 0.0, phis) for f in FINGERS}

    def _reach(k):
        u = (k + 1) / (approach_steps // 2)
        for f in FINGERS:
            for kk, j in enumerate(FINGERS[f]):
                d.ctrl[acts[j]] = ctrl0[j] * (1 - min(1.0, u)) + float(wide[f][kk]) * min(1.0, u)

    def _close(k):
        u = min(1.0, (k + 1) / (approach_steps // 2))
        for f in FINGERS:
            for kk, j in enumerate(FINGERS[f]):
                d.ctrl[acts[j]] = float(wide[f][kk]) * (1 - u) + float(tgt_app[f][kk]) * u

    _run(approach_steps // 2, _reach)      # out to the standoff ring, clear of the shaft
    _run(settle_steps // 4)
    _run(approach_steps // 2, _close)      # then straight in, radially, at the ring height
    _run(settle_steps // 2)

    def _press(k):
        u = min(1.0, (k + 1) / approach_steps)
        for f in FINGERS:
            for kk, j in enumerate(FINGERS[f]):
                d.ctrl[acts[j]] = float(tgt_app[f][kk]) * (1 - u) + float(tgt0[f][kk]) * u

    _run(approach_steps, _press)           # THE DESCENT: load the shaft onto the floor (or lift it)
    _run(settle_steps)
    spin[0] = 0.0                      # zero the counter once the grasp is established
    spin_w[0] = 0.0
    lost[0] = False
    peak_tilt[0] = 0.0
    if no_floor:                          # the in-air control, applied once the grip is set
        m.geom_contype[floor_gid] = 0
        m.geom_conaffinity[floor_gid] = 0
    grip = _hand(m, d, obj)
    gnd = _ground(m, d, obj)
    start = {"z": float(d.body(obj).xpos[2]),
             "xy": d.body(obj).xpos[:2].copy(),
             "tilt": _axis_tilt(m, d, obj)[1],
             "hand_contacts": grip[0], "hand_force_N": round(grip[1], 2),
             "pad_contacts": grip[2], "pad_force_N": round(grip[3], 2),
             "ground_contacts": gnd[0], "ground_force_N": round(gnd[1], 3)}

    # --- the gait
    per_cycle = []
    for c in range(cycles):
        s0 = spin[0]
        _run(twist_steps, lambda k: [_cmd(f, 0, stroke * min(1.0, (k + 1) / twist_steps))
                                     for f in FINGERS], tag="twist")
        if relay:
            # One finger at a time: the shaft is never released by more than one contact, so the
            # gait does not depend on it standing unaided.
            for f in FINGERS:
                _run(move_steps, lambda k, f=f: _cmd(f, 1, stroke), tag="release")
                _run(move_steps, lambda k, f=f: _cmd(
                    f, 1, stroke * max(0.0, 1 - (k + 1) / move_steps)), tag="return")
                _run(move_steps, lambda k, f=f: _cmd(f, 0, 0.0), tag="regrasp")
        else:
            _run(move_steps, lambda k: [_cmd(f, 1, stroke) for f in FINGERS], tag="release")
            _run(move_steps, lambda k: [_cmd(f, 1, stroke * max(0.0, 1 - (k + 1) / move_steps))
                                        for f in FINGERS], tag="return")
            _run(move_steps, lambda k: [_cmd(f, 0, 0.0) for f in FINGERS], tag="regrasp")
        _run(move_steps, tag="settle")
        ax, tilt = _axis_tilt(m, d, obj)
        nh, fh, npd, fpd = _hand(m, d, obj)
        ng, fg = _ground(m, d, obj)
        row = {"cycle": c + 1,
               "spin_deg": round(np.degrees(spin[0]), 2),
               "spin_world_deg": round(np.degrees(spin_w[0]), 2),
               "gain_deg": round(np.degrees(spin[0] - s0), 2),
               "tilt_deg": round(tilt, 2),
               "z": round(float(d.body(obj).xpos[2]), 4),
               "drift_mm": round(float(np.linalg.norm(d.body(obj).xpos[:2] - start["xy"])) * 1000, 2),
               "hand_contacts": nh, "hand_force_N": round(fh, 2),
               "pad_contacts": npd, "pad_force_N": round(fpd, 2),
               "ground_contacts": ng, "ground_force_N": round(fg, 3)}
        if trace:
            row["util"] = {f: round(v["util"], 3) for f, v in _util(m, d, obj).items()}
        per_cycle.append(row)
        rows.append(row)
        if film is not None and vid is not None:
            vcam.lookat[:] = [centre[0], centre[1], half]
            vid.update_scene(d, vcam)
            _mark(vid.scene, m, d, obj, r_obj, half)
            shots.append(vid.render())
        if lost[0]:
            break

    gains = [r["gain_deg"] for r in per_cycle]
    ax, tilt = _axis_tilt(m, d, obj)
    nh, fh, npd, fpd = _hand(m, d, obj)
    ng, fg = _ground(m, d, obj)
    out = {
        "scene": scene.parent.name if scene.name.endswith(".xml") else scene.name,
        "press_mm": press_mm, "stroke_deg": stroke_deg, "cycles_run": len(per_cycle),
        "cycles_asked": cycles, "squeeze_mm": squeeze * 1000, "release_mm": release_mm,
        "grip_z": grip_z, "grip_depth": grip_depth, "relay": relay, "seed": seed,
        "no_floor": no_floor, "move_steps": move_steps, "twist_steps": twist_steps,
        "ik_residual_mm": round(ik_res * 1000, 2),
        "ik_residual_grip_mm": round(per_r[0] * 1000, 2),
        "ik_residual_open_mm": round(per_r[1] * 1000, 2),
        "centre_x": centre_x,
        "start": {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else
                      [round(float(x), 4) for x in v]) for k, v in start.items()},
        "spin_deg": round(np.degrees(spin[0]), 2),
        "spin_world_deg": round(np.degrees(spin_w[0]), 2),
        "roll_fraction": round(1.0 - abs(spin_w[0]) / abs(spin[0]), 3)
                         if abs(spin[0]) > 1e-6 else 0.0,
        "phase_deg": {k: round(np.degrees(v), 2) for k, v in phase.items()},
        "phase_pad_contacts": {k: (round(float(np.mean(v)), 2) if v else 0.0)
                               for k, v in phase_contact.items()},
        "phase_ground_contacts": {k: (round(float(np.mean(v)), 2) if v else 0.0)
                                  for k, v in phase_ground.items()},
        "phase_hand_contacts": {k: (round(float(np.mean(v)), 2) if v else 0.0)
                                for k, v in phase_any.items()},
        "gain_mean_deg": round(float(np.mean(gains)), 2) if gains else 0.0,
        "gain_sd_deg": round(float(np.std(gains)), 2) if gains else 0.0,
        "gain_last3_deg": round(float(np.mean(gains[-3:])), 2) if gains else 0.0,
        "pad_radius": round(pad_r, 6), "gear_ratio": round(gear, 4),
        "deg_per_cycle": round(float(np.mean(gains)), 2) if gains else 0.0,
        "efficiency": round(float(np.mean(gains)) / stroke_deg, 3) if gains else 0.0,
        # the fraction of the slip-free transmission actually delivered
        "transmission": round(float(np.mean(gains)) / (gear * stroke_deg), 3)
                        if gains else 0.0,
        "final_tilt_deg": round(tilt, 2), "peak_tilt_deg": round(peak_tilt[0], 2),
        "final_z": round(float(d.body(obj).xpos[2]), 4),
        "drift_mm": round(float(np.linalg.norm(d.body(obj).xpos[:2] - start["xy"])) * 1000, 2),
        "hand_contacts": nh, "hand_force_N": round(fh, 2),
        "ground_contacts": ng, "ground_force_N": round(fg, 3),
        # `ok` asks the physics, not the cosine: still upright, still on its own footprint, still
        # in the hand. A shaft that has fallen over reads a large spin as it topples.
        "ok": bool(not lost[0] and tilt < 14.0 and nh >= 1 and
                   (no_floor or abs(float(d.body(obj).xpos[2]) - half) < 0.010)),
        "cycles": per_cycle,
    }
    if video is not None and frames:
        import imageio.v2 as imageio
        video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(video), frames, fps=40)
        out["video"] = str(video)
    if film is not None and shots:
        _tile(shots, film)
        out["film"] = str(film)
    return out


def _mark(scn, m, d, obj: str, r_obj: float, half: float):
    """Two body-fixed dots on the shaft surface, so a rotation is visible in the picture."""
    R = d.body(obj).xmat.reshape(3, 3)
    p0 = d.body(obj).xpos
    for k, (az, rgba) in enumerate(((0.0, (0.95, 0.25, 0.20, 1.0)),
                                    (np.pi, (0.20, 0.45, 0.95, 1.0)))):
        if scn.ngeom >= scn.maxgeom:
            return
        local = np.array([r_obj * np.cos(az), r_obj * np.sin(az), 0.25 * half])
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.0035, 0.0, 0.0]), p0 + R @ local,
                            np.eye(3).flatten(), np.array(rgba, np.float32))
        scn.ngeom += 1


def _tile(shots, path: Path, cols: int = 4):
    import imageio.v2 as imageio
    rows = int(np.ceil(len(shots) / cols))
    h, w, _ = shots[0].shape
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, s in enumerate(shots):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = s
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), canvas)


# --------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--morph-run", type=Path, action="append", default=None,
                    help="CEM run dir holding frozen_scene.xml; repeatable")
    ap.add_argument("--scene", type=Path, action="append", default=None,
                    help="scene XML directly; repeatable")
    ap.add_argument("--press", default="0.0",
                    help="mm the commanded ring sits below the grip height. >0 presses the "
                         "shaft into the floor, <0 lifts it clear (in-air control). Comma list.")
    ap.add_argument("--stroke", default="30", help="deg of azimuth per twist. Comma list.")
    ap.add_argument("--squeeze", type=float, default=0.004,
                    help="m the ring is commanded inside the shaft surface (the grip force)")
    ap.add_argument("--release", type=float, default=6.0,
                    help="mm the pads retract radially during the return stroke")
    ap.add_argument("--centre-x", type=float, default=0.004,
                    help="m the pad ring (and so the standing shaft) sits +x of the palm "
                         "origin. The mounts are asymmetric; 0 asks the thumb for 19.5 mm of "
                         "reach and the pair for 36.9 mm.")
    ap.add_argument("--grip-z", type=float, default=0.075,
                    help="m above the floor the pad ring grips the standing shaft")
    ap.add_argument("--grip-depth", type=float, default=0.045,
                    help="m the pad ring sits below the palm plane. The vertical-hold probe's "
                         "0.0615 puts the chain at 95%% extension with no workspace left.")
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--relay", action="store_true",
                    help="release/return/regrasp one finger at a time")
    ap.add_argument("--twist-steps", type=int, default=120)
    ap.add_argument("--move-steps", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=1,
                    help="spawn-jittered repeats. ONE ROLLOUT IS NOT A MEASUREMENT.")
    ap.add_argument("--jitter", type=float, default=0.0005)
    ap.add_argument("--object-body", default="screwdriver_medium")
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--film", type=Path, default=None)
    ap.add_argument("--pad-radius", type=float, default=None,
                    help="m; override the fingertip sphere radius. It sets the gait's gear "
                         "ratio (r_obj + r_pad - squeeze) / r_obj.")
    ap.add_argument("--no-floor", action="store_true",
                    help="take the ground away ONCE THE GRASP IS SET and run the identical "
                         "command stream: the in-air control for the ground-support premise")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    scenes = []
    for r in (args.morph_run or []):
        scenes.append((r.name, r / "frozen_scene.xml"))
    for s in (args.scene or []):
        scenes.append((s.parent.name if s.name == "frozen_scene.xml" else s.stem, s))
    if not scenes:
        ap.error("need --morph-run or --scene")

    presses = [float(x) for x in str(args.press).split(",")]
    strokes = [float(x) for x in str(args.stroke).split(",")]

    results = []
    print(f"{'design':22} {'press':>6} {'strk':>5} {'n':>3} "
          f"{'spin':>8} {'gain/cy':>8} {'trans':>6} {'tilt':>6} {'drift':>7} {'gndN':>7}  verdict")
    for (name, scene), press, stroke in itertools.product(scenes, presses, strokes):
        cells = []
        for rep in range(args.repeats):
            r = gait(scene, press, stroke, args.cycles, args.squeeze, args.release,
                     args.grip_z, args.grip_depth, args.object_body,
                     centre_x=args.centre_x, relay=args.relay, no_floor=args.no_floor,
                     pad_radius=args.pad_radius,
                     twist_steps=args.twist_steps, move_steps=args.move_steps,
                     jitter=args.jitter if args.repeats > 1 else 0.0, seed=rep,
                     video=args.video if rep == 0 else None,
                     film=args.film if rep == 0 else None, trace=args.trace)
            r["design"] = name
            cells.append(r)
            results.append(r)
        ok = sum(c["ok"] for c in cells)
        sp = np.array([c["spin_deg"] for c in cells])
        gm = np.array([c["gain_mean_deg"] for c in cells])
        print(f"{name:22} {press:6.1f} {stroke:5.0f} {len(cells):3d} "
              f"{sp.mean():7.1f}d {gm.mean():7.2f}d "
              f"{np.mean([c['transmission'] for c in cells]):6.3f} "
              f"{np.mean([c['final_tilt_deg'] for c in cells]):5.1f}d "
              f"{np.mean([c['drift_mm'] for c in cells]):6.1f}m "
              f"{np.mean([c['ground_force_N'] for c in cells]):6.3f}N  "
              f"{ok}/{len(cells)} ok")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
