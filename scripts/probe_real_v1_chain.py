"""The whole task in one rollout: grasp -> lift -> reorient -> stand it down -> gait.

Two results in this program have never met. `probe_real_v1_carry.py` takes the shaft off the
table and stands it vertical in the hand (cos 0.996, open loop, raised pivot). `probe_real_v1_gait.py`
spins a shaft that is ALREADY standing on the floor through 5 revolutions in 40 cycles. Each
begins where the other has no claim: the carry ends with the shaft in the air on three pads at
110 mm, and the gait begins with it planted on the ground at 50 mm, placed there by `_stand()`.

The seam between them is the whole question, and it is not a formality:

  * the carry's grasp is a PINCH ACROSS the shaft, chosen by `fit_real_v1_pose` for a horizontal
    cylinder; the gait's grasp is a RING AROUND it. They are different contact sets at different
    heights, and the gait's ring may not be reachable from wherever the carry parks the palm.
  * the carry ends TILTED. 0.996 is 5.1 deg, and a standing cylinder is only stable to 14.0.
    Setting a 5 deg shaft down on its rim edge is a topple, not a placement, so the palm has to
    take the tilt out first -- which it can, because a 6-DOF palm holding a rigid object can put
    that object anywhere: T_palm_new = T_obj_desired . T_obj_old^-1 . T_palm_old. That single
    line is the "re-pose to vertical" step, and it is also exactly the end-effector command an
    arm would be given, which is why it is written as a rigid transfer and not as a tilt hack.
  * the shaft has to be LOWERED ONTO the floor through a compliant grip. The fingers are position
    servos; the press is delivered through them, and it is the same variable `--press` the gait
    study swept (window: -6..+6 mm, +8 tips it over).

WHAT IS NEW HERE, beyond joining two probes. Standing the shaft on the ground does not only make
the release safe (that was the gait result) -- it makes the HAND free. Once the shaft supports
itself the hand can let go entirely, move to wherever it likes, and take the shaft again in a
different grasp. `--reindex full` does that: open, drive the palm to the gait's canonical pose,
close on the ring. `--reindex none` refuses the luxury and gaits from whatever grip the carry
left, with the ring solved at the carry's own palm pose. The difference between the two is the
cost of NOT having a floor to put the object down on, and it is a number this probe reports.

    uv run --extra rl python scripts/probe_real_v1_chain.py \
        --morph-run results/phase1/real_v1/rv05_manual_stored --cycles 8 --video out.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import palm_driver as pd  # noqa: E402
import probe_real_v1_gait as pg  # noqa: E402
from morphohand.tools.keyframe_ik import FINGERS, TIPS, ik_finger  # noqa: E402

DEBUG = []
CONTROL_DECIMATION = pg.CONTROL_DECIMATION


# ------------------------------------------------------------------------------------ geometry

def _rotx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _align(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector u onto unit vector v (Rodrigues)."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    s, c = float(np.linalg.norm(w)), float(np.dot(u, v))
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / s
    return np.eye(3) + np.sin(np.arccos(np.clip(c, -1, 1))) * K + (1 - c) * (K @ K)


def _rigid_palm_pose(m, d, obj: str, R_des: np.ndarray, p_des: np.ndarray):
    """World palm pose that would put a RIGIDLY HELD object at (R_des, p_des).

    T_palm_new = T_obj_des . T_obj_now^-1 . T_palm_now. The grip is not actually rigid -- the
    pads are position servos on a compliant contact and the shaft creeps -- which is why this is
    called in a loop rather than once (see `_servo`).
    """
    R_obj = d.body(obj).xmat.reshape(3, 3)
    p_obj = d.body(obj).xpos
    R_palm = d.body("palm_pose").xmat.reshape(3, 3)
    p_palm = d.body("palm_pose").xpos
    dR = R_des @ R_obj.T
    return dR @ R_palm, p_des + dR @ (p_palm - p_obj)


def _hold_ring(m, mik, dik, d, obj: str, acts: dict, r_ring: float) -> dict:
    """Finger targets that put every pad on a ring of radius `r_ring` about the shaft's axis,
    each at the axial station the CURRENT COMMAND already puts it at.

    Two traps here, and both have cost this program a session before under other names.

    ABSOLUTE, not a relative push. Grip force on a position servo is commanded-minus-actual and
    it bleeds: through a re-pose the shaft settles into the pads and the error is spent.
    Re-pushing each pad a further `depth` inward restores force but compounds without bound,
    because the pad never reaches the commanded point. A fixed interference against the shaft's
    current axis is idempotent.

    MEASURED FROM THE COMMAND, not from the achieved pose. These joints are ~60x too stiff in
    the shipped model but they still deflect under load, and that deflection IS the grip force.
    Reading the pad's ACHIEVED position and re-solving the ring from there hands back exactly
    the deflection, so every reissue of the command washes out a little more preload: measured
    18.4 -> 18.8 -> 18.4 -> 16.7 -> 11.6 -> 4.2 -> 0.25 N over seven reissues, with the object
    doing nothing unusual in between. Same family as `achieved_fraction` in the bench work.
    """
    o = d.body(obj).xpos.copy()
    ax = d.body(obj).xmat.reshape(3, 3)[:, 2]
    dik.qpos[:] = d.qpos
    for a_i in range(m.nu):                       # the commanded configuration, not the achieved
        jid = m.actuator_trnid[a_i, 0]
        if jid >= 0 and m.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE,
                                            mujoco.mjtJoint.mjJNT_SLIDE):
            dik.qpos[mik.jnt_qposadr[jid]] = float(d.ctrl[a_i])
    dik.qvel[:] = 0.0
    mujoco.mj_forward(mik, dik)
    for f in FINGERS:
        t = dik.body(TIPS[f]).xpos.copy()
        rel = t - o
        s_ax = float(rel @ ax)
        v = rel - s_ax * ax
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            continue
        ik_finger(mik, dik, f, o + s_ax * ax + (v / n) * r_ring, iters=200)
    return {j: float(dik.qpos[mik.jnt_qposadr[mik.joint(j).id]]) for j in acts}


def _finger_act(m) -> dict:
    return {j: next(k for k in range(m.nu) if m.actuator(k).name == f"a_{j}")
            for joints in FINGERS.values() for j in joints}


# ------------------------------------------------------------------------------------- the run

def chain(morph_run: Path, obj: str = "screwdriver_medium",
          lift: float = 0.10, angle_deg: float = -90.0, axis_k: float = 0.25,
          turn_steps: int = 550, budget: float = 0.5, hold_steps: int = 500,
          gap: float = 0.002, press_mm: float = 2.0, carry_squeeze: float = 0.0,
          repose_steps: int = 800, repose_iters: int = 8,
          descend_steps: int = 400, descend_iters: int = 1,
          press_steps: int = 300, settle_steps: int = 400,
          stand_order: str = "ground", airgrip: str = "cradle",
          reindex: str = "full", centre_x: float = 0.004, grip_depth: float = 0.050,
          ring_z: float | None = None,
          stroke_deg: float = 30.0, cycles: int = 8, squeeze: float = 0.002,
          release_mm: float = 6.0, twist_steps: int = 120, move_steps: int = 60,
          approach_steps: int = 200, pad_radius: float | None = None,
          jitter: float = 0.0, seed: int = 0, no_floor_gait: bool = False,
          arm_ik: Path | None = None, arm_scene: Path | None = None,
          video: Path | None = None, film: Path | None = None,
          cam=(120.0, -18.0, 0.36), video_every: int = 12, trace: bool = False) -> dict:
    scene = Path(arm_scene) if arm_scene is not None else \
        morph_run / ("arm_scene.xml" if arm_ik is not None else "frozen_scene.xml")
    pg._MODEL_PATH["path"] = str(scene)
    m = mujoco.MjModel.from_xml_path(str(scene))
    if pad_radius is not None:
        for f in FINGERS:
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == m.body(TIPS[f]).id:
                    m.geom_size[g, 0] = float(pad_radius)
                    m.geom_rbound[g] = float(pad_radius)
    pad_r = pg.PAD_RADIUS if pad_radius is None else float(pad_radius)
    d = mujoco.MjData(m)
    r_obj, half = pg._obj_geom(m, obj)
    floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    key = m.key("open_ik").id
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.ctrl[:] = m.key_ctrl[key]
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        adr = m.jnt_qposadr[m.body(obj).jntadr[0]]
        d.qpos[adr + 0] += float(rng.normal(0.0, jitter))
        d.qpos[adr + 1] += float(rng.normal(0.0, jitter))
    closed = np.load(morph_run / "best_rollout.npz")["best_finger_ctrl"]
    anchor = {j: float(closed[i * 3 + k])
              for i, (f, js) in enumerate(FINGERS.items()) for k, j in enumerate(js)}

    acts = _finger_act(m)
    palm = pd.make(m, d, arm_ik)
    for j, a in acts.items():
        d.ctrl[a] = anchor[j]
    mujoco.mj_forward(m, d)

    mik = mujoco.MjModel.from_xml_path(str(scene))
    dik = mujoco.MjData(mik)

    # ------------------------------------------------------------------ instrumentation
    step_i = [0]
    spin = [0.0]           # rotation about the shaft's own axis, integrated at the SIM rate:
    spin_w = [0.0]         # sampling this at the control rate aliases a 500 Hz signal and
    marks: list = []       # invents tens of degrees per cycle in phases with no hand contact.
    vid = vcam = None
    frames: list = []
    if video is not None or film is not None:
        vid = mujoco.Renderer(m, height=480, width=640)
        vcam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, vcam)
        vcam.azimuth, vcam.elevation, vcam.distance = cam

    def _integrate():
        ax = d.body(obj).xmat.reshape(3, 3)[:, 2]
        dadr = m.jnt_dofadr[m.body(obj).jntadr[0]]
        w_world = d.body(obj).xmat.reshape(3, 3) @ d.qvel[dadr + 3:dadr + 6]
        spin[0] += float(np.dot(w_world, ax)) * m.opt.timestep
        spin_w[0] += float(w_world[2]) * m.opt.timestep

    def _run(n, before=None, every_step=False):
        for k in range(n):
            if before is not None and (every_step or step_i[0] % CONTROL_DECIMATION == 0):
                before(k)
            mujoco.mj_step(m, d)
            _integrate()
            step_i[0] += 1
            if vid is not None and step_i[0] % video_every == 0:
                vcam.lookat[:] = d.body(obj).xpos
                vid.update_scene(d, vcam)
                pg._mark(vid.scene, m, d, obj, r_obj, half)
                frames.append(vid.render())

    def _snap(name: str) -> dict:
        nh, fh, npd, fpd = pg._hand(m, d, obj)
        ng, fg = pg._ground(m, d, obj)
        ax, tilt = pg._axis_tilt(m, d, obj)
        p = d.body(obj).xpos
        return {"phase": name, "cos": round(float(d.body(obj).xmat[8]), 4),
                "tilt_deg": round(tilt, 2), "z": round(float(p[2]), 4),
                "xy_mm": [round(float(p[0]) * 1000, 1), round(float(p[1]) * 1000, 1)],
                "hand_contacts": nh, "hand_force_N": round(fh, 2),
                "pad_contacts": npd, "pad_force_N": round(fpd, 2),
                "ground_contacts": ng, "ground_force_N": round(fg, 3),
                "spin_deg": round(float(np.degrees(spin[0])), 2),
                # Where each pad sits on the shaft, in the shaft's own frame: axial station from
                # the centre (mm, + toward the top) and radial distance from the axis (mm). A
                # station past +/- half_len means the pad is on the END CAP, where the grasp
                # wedges instead of pinching and extinguishes itself as the shaft slides.
                "pad_s_mm": {f: round(float((d.body(TIPS[f]).xpos - p) @
                                            d.body(obj).xmat.reshape(3, 3)[:, 2]) * 1000, 1)
                             for f in FINGERS},
                "pad_r_mm": {f: round(float(np.linalg.norm(
                    (d.body(TIPS[f]).xpos - p) - float((d.body(TIPS[f]).xpos - p) @
                     d.body(obj).xmat.reshape(3, 3)[:, 2]) *
                     d.body(obj).xmat.reshape(3, 3)[:, 2])) * 1000, 1) for f in FINGERS}}

    seams = []
    if film is not None:
        def _shot():
            vcam.lookat[:] = d.body(obj).xpos
            vid.update_scene(d, vcam)
            pg._mark(vid.scene, m, d, obj, r_obj, half)
            marks.append(vid.render())
    else:
        def _shot():
            return None

    # ---------------------------------------------------------------- 1. grasp, lift, settle
    _run(250)
    R_c, p_c = palm.cmd_pose()
    u0 = palm.read()
    u1 = palm.solve(R_c, p_c + np.array([0.0, 0.0, lift]))[0]
    _run(200, lambda k: palm.write(u0 + (u1 - u0) * (k + 1) / 200), every_step=True)
    _run(200)
    seams.append(_snap("lifted"))
    _shot()

    # ------------------------------------------------------------------------- 2. the carry
    # Verbatim `probe_real_v1_carry.py --linear-anchor`: the pivot is raised axis_k half-straddles
    # above the contact plane so the descending finger spends RETRACTION (38-47 mm available)
    # instead of EXTENSION (1.3-9.6 mm available), and the anchor is swept in a straight line in
    # joint space, which is what LerpFingerActionCfg.hold_target_ctrl does on the bench.
    angle = np.radians(angle_deg)
    tip0 = {f: d.body(TIPS[f]).xpos.copy() for f in FINGERS}
    centroid = np.mean([tip0[f] for f in FINGERS], axis=0)
    span = abs(tip0["index"][1] - tip0["middle"][1]) / 2.0
    centroid[2] += axis_k * span
    q0 = {j: float(d.qpos[m.jnt_qposadr[m.joint(j).id]]) for j in acts}
    dik.qpos[:] = d.qpos
    dik.qvel[:] = 0.0
    mujoco.mj_forward(mik, dik)
    R = _rotx(angle)
    carry_ik = 0.0
    for f in FINGERS:
        carry_ik = max(carry_ik, ik_finger(mik, dik, f, centroid + R @ (tip0[f] - centroid),
                                           iters=400))
    end = {j: float(dik.qpos[mik.jnt_qposadr[mik.joint(j).id]]) for j in acts}

    def _turn(k):
        u = (k + 1) / turn_steps
        for j, a in acts.items():
            d.ctrl[a] = anchor[j] + float(np.clip((end[j] - q0[j]) * u, -budget, budget))

    _run(turn_steps, _turn, every_step=True)
    _run(hold_steps)
    seams.append(_snap("reoriented"))
    _shot()
    carry_ok = seams[-1]["hand_contacts"] >= 1 and seams[-1]["z"] > seams[0]["z"] - 0.02

    # ------------------------------------------------- 3. re-pose to vertical, then set it down
    #
    # The palm holds a rigid object, so asking for an object pose fully determines the palm pose:
    # T_palm_new = T_obj_des . T_obj_now^-1 . T_palm_now. That is the whole "re-pose to vertical"
    # step, and it is written as an end-effector command because that is what an arm will be
    # given when the floating palm is replaced.
    #
    # ONE such transfer is not enough, and the reason is worth stating because it looks like a
    # bug: the grip is NOT rigid. Applied open-loop from a carry that ended 18.9 deg off
    # vertical, a single transfer left the shaft at 26.2 deg -- WORSE than it started -- because
    # the shaft creeps inside the pads through the move and the correction is computed once,
    # against a pose that no longer holds. Past 14.0 deg a standing cylinder is a topple, so the
    # release then dropped it. Iterating the same transfer against the MEASURED object pose
    # converges instead, and this is not a sim luxury: the bench has an object-pose sensor
    # (two AprilTags, 0.017 deg / 0.03 mm rms) whose whole point was that no controller yet used
    # it. `--repose open` runs the single-shot version as the control.
    r_hold = r_obj + pad_r - carry_squeeze
    r_grip = r_obj + pad_r - squeeze
    r_open = r_obj + pad_r + release_mm / 1000.0
    r_wide = r_obj + pad_r + 0.018
    gear = r_grip / r_obj
    stroke = np.radians(stroke_deg)
    phis = np.linspace(0.0, stroke, 25)
    if carry_squeeze > 0.0:
        # RE-GRIP BEFORE MOVING, and keep re-gripping through the move (see `_servo`). Pad force
        # through the carry: 5.6 N at the grasp, 19.1 N at the top of the turn, and then a
        # monotone bleed to 0 across the re-pose if nothing puts it back.
        sq = _hold_ring(m, mik, dik, d, obj, acts, r_hold)
        f0 = {j: float(d.ctrl[a]) for j, a in acts.items()}
        _run(200, lambda k: [d.ctrl.__setitem__(
            a, f0[j] + (sq[j] - f0[j]) * min(1.0, (k + 1) / 200))
            for j, a in acts.items()], every_step=True)
        _run(settle_steps // 4)
        seams.append(_snap("regripped"))
        _shot()

    def _ring_targets(z_ring: float, r_ring: float):
        """Tripod targets on a ring about the shaft's OWN axis, at world height `z_ring`.

        The gait's grasp, solved against a shaft that is standing in the air rather than on the
        floor and may still be a few degrees off vertical -- so the ring is built in the shaft's
        frame, not the world's. Returns (joint targets, worst IK residual in m).
        """
        o = d.body(obj).xpos.copy()
        av = d.body(obj).xmat.reshape(3, 3)[:, 2].copy()
        av = av * (1.0 if av[2] >= 0 else -1.0)
        e1 = np.array([1.0, 0.0, 0.0]) - av[0] * av
        n1 = float(np.linalg.norm(e1))
        e1 = e1 / n1 if n1 > 1e-9 else np.array([1.0, 0.0, 0.0])
        e2 = np.cross(av, e1)
        p_ring = o + ((z_ring - o[2]) / av[2]) * av if abs(av[2]) > 1e-6 else o
        dik.qpos[:] = d.qpos
        dik.qvel[:] = 0.0
        mujoco.mj_forward(mik, dik)
        worst = 0.0
        for f in FINGERS:
            az = pg.AZIMUTH[f]
            tgt = p_ring + r_ring * (np.cos(az) * e1 + np.sin(az) * e2)
            worst = max(worst, ik_finger(mik, dik, f, tgt, iters=300))
        return {j: float(dik.qpos[mik.jnt_qposadr[mik.joint(j).id]]) for j in acts}, worst

    def _low_point():
        """World point where the tilted shaft would touch a floor: the low edge of its bottom rim."""
        R = d.body(obj).xmat.reshape(3, 3)
        c = d.body(obj).xpos.copy()
        av = R[:, 2] * (1.0 if R[2, 2] >= 0 else -1.0)
        u = np.array([0.0, 0.0, 1.0]) - av[2] * av
        n = float(np.linalg.norm(u))
        return c - half * av - (r_obj * (u / n) if n > 1e-9 else np.zeros(3))

    def _move_u(u1, steps_each, hold=True, settle=None):
        """Ramp the palm command (and, if holding, the pad ring) to one target vector."""
        c0 = palm.read()
        sq = _hold_ring(m, mik, dik, d, obj, acts, r_hold) if (hold and carry_squeeze > 0) else None
        f0 = {j: float(d.ctrl[a]) for j, a in acts.items()}

        def _mv(k):
            v = min(1.0, (k + 1) / steps_each)
            palm.write(c0 + (np.asarray(u1, float) - c0) * v)
            if sq is not None:
                for j, a in acts.items():
                    d.ctrl[a] = f0[j] * (1 - v) + sq[j] * v

        _run(steps_each, _mv, every_step=True)
        _run(settle_steps // 4 if settle is None else settle)

    def _move(R, p_, steps_each, hold=True, settle=None):
        _move_u(palm.solve(R, p_)[0], steps_each, hold=hold, settle=settle)

    def _lost():
        return pg._hand(m, d, obj)[2] == 0 and float(d.body(obj).xpos[2]) < half * 0.8

    # The gain removes 98% of the initial error in `iters` corrections whatever `iters` is, so
    # the arm below varies the number of MEASUREMENTS and nothing else, and iters=1 is an honest
    # open-loop control rather than a differently-tuned controller. Proportional and never
    # scheduled: a reference ramp made `delta` change sign whenever the plant ran behind, and on
    # a 20 N grip with 150 steps of lag it always does -- the palm oscillated to rx -0.49 rad and
    # threw the shaft from a state that had already reached 3.0 deg.
    def _gain(iters):
        return 1.0 - 0.02 ** (1.0 / max(1, iters))

    def _descend(name, iters, steps_each, z_low_goal):
        """Lower the shaft, orientation untouched, until its low edge is `z_low_goal` off the floor.

        THE ONE AXIS WHERE FEEDBACK IS A MISTAKE, and it took a sweep to believe it: 1 correction
        stands the shaft 4/4 at every descent speed from 0.2 s to 1.6 s, 8 corrections stand it
        1/4. The reason is that the carry's terminal grasp LEAKS -- `probe_real_v1_carry` measures
        16.1 -> 13.2 -> 10.9 -> 6.0 -> 1.3 -> 0 N holding and doing nothing else, shaft on the
        table by 1.6 s -- and the shaft creeps down through the pads at about 1.5 mm/s while it
        does. Each extra correction adds a dwell, and every dwell is paid in grip. The set-down
        is a race, so it is run as ONE continuous move. (The tilt is the opposite case: there
        feedback is what makes the seam work at all. Height error here is the grip slipping, and
        lowering the palm further does not put it back.)
        """
        z0 = float(_low_point()[2])
        for it in range(iters):
            R_obj = d.body(obj).xmat.reshape(3, 3).copy()
            po = d.body(obj).xpos.copy()
            dz = (z0 + (z_low_goal - z0) * (it + 1) / iters) - float(_low_point()[2])
            _move(*_rigid_palm_pose(m, d, obj, R_obj, po + np.array([0.0, 0.0, dz])), steps_each)
            if DEBUG:
                print("   ", {**_snap(name), "dz_mm": round(dz * 1000, 1),
                              "z_low_mm": round(float(_low_point()[2]) * 1000, 1),
                              "palm_z": round(float(palm.cmd_pose()[1][2]), 4)})
            if _lost():
                break
        seams.append(_snap(name))
        _shot()

    def _upright(name, iters, steps_each, about_foot: bool):
        """Rotate the shaft to vertical, about its own centre or about the point it stands on.

        ABOUT THE FOOT is the difference between standing a rod up and juggling it. Corrected in
        mid-air the shaft's whole weight hangs on friction at three pads while the palm swings
        under it, and the grip bleeds -- 18.5 N at correction 1, 0.56 N at correction 8, shaft
        gone. Set the foot down first and the floor carries the weight through the entire
        correction; the fingers only have to keep the shaft from falling over, which is the job
        the gait study already showed they can do.
        """
        g = _gain(iters)
        for _ in range(iters):
            R_obj = d.body(obj).xmat.reshape(3, 3).copy()
            av = R_obj[:, 2] * (1.0 if R_obj[2, 2] >= 0 else -1.0)
            th = float(np.arccos(np.clip(av[2], -1.0, 1.0)))
            w = np.cross(av, np.array([0.0, 0.0, 1.0]))
            nw = float(np.linalg.norm(w))
            delta = g * th
            if nw < 1e-9 or delta < 1e-9:
                break
            k = w / nw
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R_corr = np.eye(3) + np.sin(delta) * K + (1 - np.cos(delta)) * (K @ K)
            po = d.body(obj).xpos.copy()
            piv = _low_point() if about_foot else po
            _move(*_rigid_palm_pose(m, d, obj, R_corr @ R_obj, piv + R_corr @ (po - piv)),
                  steps_each)
            if DEBUG:
                print("   ", {**_snap(name), "delta_deg": round(float(np.degrees(delta)), 2),
                              "z_low_mm": round(float(_low_point()[2]) * 1000, 1)})
            if _lost():
                break
        seams.append(_snap(name))
        _shot()

    # THE CARRY DOES NOT END IN A GRIP. At the top of a raised-pivot turn the shaft is commonly
    # cradled against the middle phalanges with the pads off it entirely -- pad force 0.0 N,
    # which `probe_real_v1_carry` documents and scores as held, correctly, because it is. A
    # cradle survives a hold; it does not survive a 61 mm descent, which drops the shaft on 1-3
    # of every 4 spawn-jittered seeds however slowly it is flown. Converting the cradle into the
    # gait's own tripod ring FIRST, while the shaft is vertical and still in the air, is the one
    # move that makes the descent uneventful -- and it costs nothing new, because that ring is
    # the grasp the gait was going to take anyway.
    air_ik = float("nan")
    if airgrip == "ring":
        z_air = float(d.body("palm_pose").xpos[2]) - grip_depth
        tgt_air, air_ik = _ring_targets(z_air, r_obj + pad_r - squeeze)
        f0 = {j: float(d.ctrl[a]) for j, a in acts.items()}
        _run(approach_steps, lambda k: [d.ctrl.__setitem__(
            a, f0[j] + (tgt_air[j] - f0[j]) * min(1.0, (k + 1) / approach_steps))
            for j, a in acts.items()], every_step=True)
        _run(settle_steps)
        seams.append(_snap("air_grip"))
        _shot()

    R0 = d.body(obj).xmat.reshape(3, 3)
    repose_deg = round(float(np.degrees(np.arccos(np.clip(abs(R0[2, 2]), -1, 1)))), 2)
    if stand_order == "ground":
        _descend("set_down", descend_iters, max(1, descend_steps // descend_iters), gap)
        _upright("upright", repose_iters, max(1, repose_steps // repose_iters), True)
    else:
        _upright("upright", repose_iters, max(1, repose_steps // repose_iters), False)
        _descend("set_down", descend_iters, max(1, descend_steps // descend_iters), gap)

    # The press is delivered THROUGH the grip, exactly as in the gait study, where it has a
    # measured window: -6..+6 mm holds, +8 levers the shaft off its own footprint.
    R_c, p_c = palm.cmd_pose()
    _move(R_c, p_c - np.array([0.0, 0.0, press_mm / 1000.0]), press_steps,
          hold=False, settle=settle_steps)
    seams.append(_snap("pressed"))
    _shot()
    stood = seams[-1]
    stood_ok = bool(stood["ground_contacts"] >= 1 and stood["tilt_deg"] < 14.0
                    and abs(stood["z"] - half) < 0.010)

    # ------------------------------------------------------------------ 4. take the gait grasp
    centre = d.body(obj).xpos[:2].copy()
    z_ring = float(np.mean([d.body(TIPS[f]).xpos[2] for f in FINGERS])) if ring_z is None \
        else float(ring_z)

    if reindex == "full":
        # THE FLOOR MAKES THE HAND FREE. Let go completely, drive the palm to the pose the gait
        # study validated, and take the shaft again as a ring. Nothing else in this program can
        # do this: every other release in the repertoire drops the object.
        z_ring = float(half + 0.025) if ring_z is None else float(ring_z)
        u_tgt, ep_re, er_re = palm.solve(
            np.eye(3), np.array([centre[0] - centre_x, centre[1], z_ring + grip_depth]))
        table, ik_res, per_r = pg._ring_table(
            m, centre, z_ring, palm.joint_dict(u_tgt), [r_grip, r_open, r_wide], phis)
        open_ctrl = {j: float(m.key_ctrl[key][a]) for j, a in acts.items()}
        cur_f = {j: float(d.ctrl[a]) for j, a in acts.items()}
        _run(approach_steps // 2, lambda k: [
            d.ctrl.__setitem__(a, cur_f[j] * (1 - min(1.0, (k + 1) / (approach_steps // 2)))
                               + open_ctrl[j] * min(1.0, (k + 1) / (approach_steps // 2)))
            for j, a in acts.items()], every_step=True)
        _run(settle_steps // 4)
        seams.append(_snap("released"))
        _shot()
        _move_u(u_tgt, repose_steps, hold=False, settle=settle_steps // 2)
        seams.append(_snap("reindexed"))
        _shot()
    else:
        # No luxury: solve the ring at the palm pose the carry actually left, and regrasp from
        # the carry's own finger pose. If this is unreachable the residual says so directly.
        table, ik_res, per_r = pg._ring_table(
            m, centre, z_ring, palm.joint_dict(palm.read()), [r_grip, r_open, r_wide], phis)

    wide = {f: pg._lookup(table, f, 2, 0.0, phis) for f in FINGERS}
    grip_t = {f: pg._lookup(table, f, 0, 0.0, phis) for f in FINGERS}
    cur_f = {j: float(d.ctrl[a]) for j, a in acts.items()}

    def _reach(k):
        u = min(1.0, (k + 1) / (approach_steps // 2))
        for f in FINGERS:
            for kk, j in enumerate(FINGERS[f]):
                d.ctrl[acts[j]] = cur_f[j] * (1 - u) + float(wide[f][kk]) * u

    def _close(k):
        u = min(1.0, (k + 1) / (approach_steps // 2))
        for f in FINGERS:
            for kk, j in enumerate(FINGERS[f]):
                d.ctrl[acts[j]] = float(wide[f][kk]) * (1 - u) + float(grip_t[f][kk]) * u

    _run(approach_steps // 2, _reach, every_step=True)
    _run(settle_steps // 4)
    _run(approach_steps // 2, _close, every_step=True)
    _run(settle_steps)
    seams.append(_snap("gait_grip"))
    _shot()
    grip_snap = seams[-1]
    grip_ok = bool(grip_snap["pad_contacts"] >= 2 and grip_snap["tilt_deg"] < 14.0)

    # ------------------------------------------------------------------------------ 5. the gait
    if no_floor_gait:
        m.geom_contype[floor_gid] = 0
        m.geom_conaffinity[floor_gid] = 0
    spin[0] = spin_w[0] = 0.0
    start_xy = d.body(obj).xpos[:2].copy()
    lost = [False]

    def _cmd(f, ri, phi):
        q = pg._lookup(table, f, ri, phi, phis)
        for k, j in enumerate(FINGERS[f]):
            d.ctrl[acts[j]] = float(q[k])

    per_cycle = []
    for c in range(cycles):
        s0 = spin[0]
        _run(twist_steps, lambda k: [_cmd(f, 0, stroke * min(1.0, (k + 1) / twist_steps))
                                     for f in FINGERS])
        _run(move_steps, lambda k: [_cmd(f, 1, stroke) for f in FINGERS])
        _run(move_steps, lambda k: [_cmd(f, 1, stroke * max(0.0, 1 - (k + 1) / move_steps))
                                    for f in FINGERS])
        _run(move_steps, lambda k: [_cmd(f, 0, 0.0) for f in FINGERS])
        _run(move_steps)
        _, tilt = pg._axis_tilt(m, d, obj)
        nh, fh, npd, fpd = pg._hand(m, d, obj)
        ng, fg = pg._ground(m, d, obj)
        p = d.body(obj).xpos
        if tilt > 45.0 or float(p[2]) < half * 0.5:
            lost[0] = True
        row = {"cycle": c + 1, "spin_deg": round(float(np.degrees(spin[0])), 2),
               "gain_deg": round(float(np.degrees(spin[0] - s0)), 2),
               "tilt_deg": round(tilt, 2), "z": round(float(p[2]), 4),
               "drift_mm": round(float(np.linalg.norm(p[:2] - start_xy)) * 1000, 2),
               "hand_contacts": nh, "hand_force_N": round(fh, 2),
               "pad_contacts": npd, "pad_force_N": round(fpd, 2),
               "ground_contacts": ng, "ground_force_N": round(fg, 3)}
        if trace:
            row["util"] = {f: round(v["util"], 3) for f, v in pg._util(m, d, obj).items()}
        per_cycle.append(row)
        _shot()
        if lost[0]:
            break

    gains = [r["gain_deg"] for r in per_cycle]
    _, tilt = pg._axis_tilt(m, d, obj)
    nh, fh, npd, fpd = pg._hand(m, d, obj)
    ng, fg = pg._ground(m, d, obj)
    seams.append(_snap("gaited"))
    _shot()
    out = {
        "run": morph_run.name, "object": obj, "reindex": reindex,
        "angle_deg": angle_deg, "axis_k": axis_k, "turn_steps": turn_steps,
        "hold_steps": hold_steps, "descend_steps": descend_steps, "lift": lift,
        "gap_mm": gap * 1000, "press_mm": press_mm, "grip_depth": grip_depth,
        "carry_squeeze_mm": carry_squeeze * 1000,
        "repose_iters": repose_iters, "descend_iters": descend_iters,
        "stand_order": stand_order, "airgrip": airgrip,
        "air_ik_mm": None if air_ik != air_ik else round(air_ik * 1000, 2),
        "centre_x": centre_x, "ring_z": round(z_ring, 4), "squeeze_mm": squeeze * 1000,
        "release_mm": release_mm, "stroke_deg": stroke_deg, "cycles_asked": cycles,
        "cycles_run": len(per_cycle), "no_floor_gait": no_floor_gait, "seed": seed,
        "carry_ik_residual_mm": round(carry_ik * 1000, 2),
        "repose_deg": repose_deg,
        "ring_ik_mm": round(ik_res * 1000, 2),
        "ring_ik_grip_mm": round(per_r[0] * 1000, 2),
        "ring_ik_open_mm": round(per_r[1] * 1000, 2),
        "pad_radius": round(pad_r, 6), "gear_ratio": round(gear, 4),
        "wrist": palm.kind,
        "arm_ik_pos_mm": round(getattr(palm, "worst_pos", 0.0) * 1000, 3),
        "arm_ik_rot_deg": round(float(np.degrees(getattr(palm, "worst_rot", 0.0))), 3),
        "arm_ik_fails": int(getattr(palm, "fails", 0)),
        "seams": seams,
        "carry_ok": bool(carry_ok), "stood_ok": stood_ok, "grip_ok": grip_ok,
        "spin_deg": round(float(np.degrees(spin[0])), 2),
        "turns": round(float(np.degrees(spin[0])) / 360.0, 3),
        "gain_mean_deg": round(float(np.mean(gains)), 2) if gains else 0.0,
        "gain_sd_deg": round(float(np.std(gains)), 2) if gains else 0.0,
        "transmission": round(float(np.mean(gains)) / (gear * stroke_deg), 3) if gains else 0.0,
        "final_tilt_deg": round(tilt, 2),
        "final_z": round(float(d.body(obj).xpos[2]), 4),
        "drift_mm": round(float(np.linalg.norm(d.body(obj).xpos[:2] - start_xy)) * 1000, 2),
        "hand_contacts": nh, "hand_force_N": round(fh, 2),
        "ground_contacts": ng, "ground_force_N": round(fg, 3),
        # The chain is only a chain if every seam held. `ok` is the AND of the four, so a run
        # that reoriented beautifully and then dropped the shaft on the way down reads False.
        "ok": bool(carry_ok and stood_ok and grip_ok and not lost[0] and tilt < 14.0
                   and nh >= 1 and (no_floor_gait or abs(float(d.body(obj).xpos[2]) - half) < 0.010)
                   and len(per_cycle) == cycles),
        "cycles": per_cycle,
    }
    if video is not None and frames:
        import imageio.v2 as imageio
        video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(video), frames, fps=40)
        out["video"] = str(video)
    if film is not None and marks:
        pg._tile(marks, film, cols=4)
        out["film"] = str(film)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--morph-run", type=Path, action="append",
                    default=None, help="CEM run dir holding frozen_scene.xml; repeatable")
    ap.add_argument("--arm-ik", type=Path, default=None,
                    help="the arm-only IK model written by build_real_v1_arm_scene.py. Given, "
                         "the palm's six commands are produced by a UR5e instead of the "
                         "floating gantry, and --morph-run must point at a run whose "
                         "arm_scene.xml exists.")
    ap.add_argument("--arm-scene", type=Path, default=None,
                    help="the task scene to roll out; defaults to <run>/arm_scene.xml when an "
                         "IK model is given, else the run's frozen_scene.xml")
    ap.add_argument("--arm", action="store_true",
                    help="shorthand: use <run>/arm_scene.xml and <run>/arm_ik.xml")
    ap.add_argument("--object-body", default="screwdriver_medium")
    ap.add_argument("--lift", type=float, default=0.10)
    ap.add_argument("--angle-deg", type=float, default=-90.0)
    ap.add_argument("--axis-k", type=float, default=0.25)
    ap.add_argument("--turn-steps", type=int, default=550)
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--hold-steps", type=int, default=500,
                    help="settle after the turn. The shaft is still standing UP in the "
                         "grip here: cos 0.837 at the last commanded step, 0.946 after "
                         "300, 0.996 after 500. Cutting it short hands the seam a 19 deg "
                         "tilt instead of a 5 deg one, and 14.0 is where a standing "
                         "cylinder topples.")
    ap.add_argument("--gap", type=float, default=0.002,
                    help="m the shaft's bottom face is left above the floor by the re-pose, "
                         "before the press closes it")
    ap.add_argument("--press", default="2.0",
                    help="mm the palm drives down after the shaft lands. Comma list.")
    ap.add_argument("--carry-squeeze", type=float, default=0.0,
                    help="m the pads are pushed back into the shaft after the turn, before the "
                         "re-pose. These are position servos; the error is the grip force.")
    ap.add_argument("--repose-iters", type=int, default=8,
                    help="rigid-transfer corrections used to bring the shaft upright. 1 is the "
                         "OPEN-LOOP control: one transfer computed against a grip that then "
                         "creeps. On the bench the measurement is the AprilTag rig.")
    ap.add_argument("--repose-steps", type=int, default=800)
    ap.add_argument("--descend-iters", type=int, default=1,
                    help="corrections on the way down. MORE IS WORSE: the carry's "
                         "terminal grasp leaks (16.1 N -> 0 in 1.6 s, and the shaft "
                         "creeps ~1.5 mm/s through it), so every measurement pause "
                         "costs grip. 1 continuous move = 4/4 seeds, 8 = 1/4.")
    ap.add_argument("--descend-steps", type=int, default=400)
    ap.add_argument("--airgrip", default="cradle", choices=("ring", "cradle"),
                    help="cradle = descend on whatever grasp the carry left; ring = try to "
                         "convert it into the gait's tripod FIRST, in mid-air. `ring` is a "
                         "control and it fails 4/4: the ring solves to 1.1 mm, but a linear "
                         "joint-space move between two grasps of the same object passes through "
                         "a configuration that holds neither, and in mid-air there is nothing "
                         "under the shaft. Changing grasp needs the floor.")
    ap.add_argument("--stand-order", default="ground", choices=("ground", "air"),
                    help="ground = set the tilted shaft's foot on the floor, then rotate it "
                         "upright about that foot; air = stand it up in mid-air first")
    ap.add_argument("--reindex", default="full", choices=("full", "none"),
                    help="full = let go on the floor and retake the gait's ring grasp; "
                         "none = gait from the grip the carry left")
    ap.add_argument("--centre-x", type=float, default=0.004)
    ap.add_argument("--grip-depth", type=float, default=0.050)
    ap.add_argument("--ring-z", type=float, default=None)
    ap.add_argument("--stroke", type=float, default=30.0)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--squeeze", type=float, default=0.002)
    ap.add_argument("--release", type=float, default=6.0)
    ap.add_argument("--twist-steps", type=int, default=120)
    ap.add_argument("--move-steps", type=int, default=60)
    ap.add_argument("--pad-radius", type=float, default=None)
    ap.add_argument("--no-floor-gait", action="store_true",
                    help="delete the floor once the gait grasp is set: the in-air control")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--jitter", type=float, default=0.0005)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--film", type=Path, default=None)
    ap.add_argument("--cam", default="120,-18,0.36")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = args.morph_run or [ROOT / "results/phase1/real_v1/rv05_manual_stored"]
    arm_ik = args.arm_ik or ((runs[0] / "arm_ik.xml") if args.arm else None)
    cam = tuple(float(v) for v in args.cam.split(","))
    rows = []
    print(f"{'run':22} {'idx':5} {'press':>6} {'ringIK':>7} {'repose':>7} "
          f"{'turns':>6} {'deg/cy':>7} {'tilt':>6} {'drift':>7} {'cyc':>5}  ok")
    for run in runs:
        for press in (float(v) for v in str(args.press).split(",")):
            for rep in range(args.repeats):
                r = chain(run, obj=args.object_body, lift=args.lift, angle_deg=args.angle_deg,
                          axis_k=args.axis_k, turn_steps=args.turn_steps, budget=args.budget,
                          hold_steps=args.hold_steps, gap=args.gap, press_mm=press,
                          carry_squeeze=args.carry_squeeze,
                          repose_iters=args.repose_iters, repose_steps=args.repose_steps,
                          descend_iters=args.descend_iters, descend_steps=args.descend_steps,
                          stand_order=args.stand_order, airgrip=args.airgrip,
                          reindex=args.reindex, centre_x=args.centre_x,
                          grip_depth=args.grip_depth, ring_z=args.ring_z,
                          stroke_deg=args.stroke, cycles=args.cycles, squeeze=args.squeeze,
                          release_mm=args.release, twist_steps=args.twist_steps,
                          move_steps=args.move_steps, pad_radius=args.pad_radius,
                          no_floor_gait=args.no_floor_gait, arm_ik=arm_ik,
                          arm_scene=args.arm_scene,
                          jitter=args.jitter if rep or args.repeats > 1 else 0.0,
                          seed=rep, video=args.video if rep == 0 else None,
                          film=args.film if rep == 0 else None, cam=cam, trace=args.trace)
                r["rep"] = rep
                rows.append(r)
                print(f"{r['run']:22} {r['reindex']:5} {press:6.1f} {r['ring_ik_grip_mm']:7.2f} "
                      f"{r['repose_deg']:7.2f} {r['turns']:6.2f} {r['gain_mean_deg']:7.2f} "
                      f"{r['final_tilt_deg']:6.2f} {r['drift_mm']:7.2f} "
                      f"{r['cycles_run']:2d}/{r['cycles_asked']:<2d} "
                      f"{'OK' if r['ok'] else '--'}"
                      f"{'' if r['carry_ok'] else ' carry'}"
                      f"{'' if r['stood_ok'] else ' stand'}"
                      f"{'' if r['grip_ok'] else ' grip'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
