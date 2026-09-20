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


def _rot_lerp(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    """Rotation `t` of the way from R0 to R1 along the geodesic."""
    dR = R0.T @ R1
    c = float(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(c))
    sn = np.sin(th)
    if th < 1e-7 or abs(sn) < 1e-7:
        return R1.copy() if t >= 1.0 else R0.copy()
    w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2.0 * sn)
    a = th * float(t)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return R0 @ (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K))


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


def _pad_frame(m, d, obj: str) -> dict:
    """Each pad's ACHIEVED station and radius in the shaft's own frame, mm-free (metres).

    station = signed distance along the shaft's long axis from its centre; radius = distance
    from the axis. Together they say where on the shaft the pad is sitting, which is the
    quantity that changes when the shaft slides through a grip rather than when the grip
    loosens.
    """
    o = d.body(obj).xpos.copy()
    ax = d.body(obj).xmat.reshape(3, 3)[:, 2]
    out = {}
    for f in FINGERS:
        rel = d.body(TIPS[f]).xpos - o
        sst = float(rel @ ax)
        v = rel - sst * ax
        out[f] = (sst, float(np.linalg.norm(v)))
    return out


def _regrasp_cmd(m, mik, dik, d, obj: str, acts: dict, ref: dict) -> dict:
    """Put every pad back where it was in the SHAFT's frame, from the commanded pose.

    `_squeeze_cmd` presses each pad `depth` further in along the current radius. That restores
    force but not position: if the shaft has slid axially through the grasp -- which is what a
    41 N grip decaying to 0.33 N over a lift actually is -- pressing harder holds the shaft in
    its new, wrong place. This drives each pad back to the (station, radius) it held at closure,
    so the correction is where the pad went rather than how hard it is pushing.
    """
    o = d.body(obj).xpos.copy()
    ax = d.body(obj).xmat.reshape(3, 3)[:, 2]
    dik.qpos[:] = d.qpos
    for a_i in range(m.nu):
        jid = m.actuator_trnid[a_i, 0]
        if jid >= 0 and m.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE,
                                            mujoco.mjtJoint.mjJNT_SLIDE):
            dik.qpos[mik.jnt_qposadr[jid]] = float(d.ctrl[a_i])
    dik.qvel[:] = 0.0
    mujoco.mj_forward(mik, dik)
    for f in FINGERS:
        rel = dik.body(TIPS[f]).xpos - o
        v = rel - float(rel @ ax) * ax
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            continue
        s_ref, r_ref = ref[f]
        ik_finger(mik, dik, f, o + s_ref * ax + r_ref * (v / n), iters=200)
    return {j: float(dik.qpos[mik.jnt_qposadr[mik.joint(j).id]]) for j in acts}


def _squeeze_cmd(m, mik, dik, d, obj: str, acts: dict, depth, from_achieved: bool = False) -> dict:
    """Push every pad `depth` further into the shaft, radially, FROM THE COMMANDED POSE.

    `from_achieved` measures from the achieved pose instead, which is the right reference when
    the command is not a millimetre of deflection but a radian: an RL residual frozen at its
    clip leaves the thumb 76 deg past its achieved angle with the actuator on its 0.35 N m
    ceiling (2026-09-20, D6 on the calibrated plant), and every grip composed from that command
    -- the relay's pinned pads, the gait table's rebase -- inherits the saturation. Re-seeding
    from the achieved angles with `depth` = the plans' 10 mm gives the servo a command it can
    deflect toward, at 2.5-4.5 N on the pads rather than the ceiling.

    `depth` is one number for every finger or a `{finger: depth}` dict; a NEGATIVE depth pulls
    the pad radially OUT of the shaft, which is how a driver finger is relieved before a turn
    on the compliant plant (see `turn_relief`).

    The carry's own re-squeeze (`probe_real_v1_carry --hold-squeeze`) with one correction, and
    the correction is the whole point. Grip force on a position servo is commanded-minus-actual,
    so measuring the pad's ACHIEVED position and pushing `depth` from there hands back exactly
    the deflection that was carrying the load: reissued, it bled 18.4 -> 16.7 -> 11.6 -> 4.2 ->
    0.25 N with the object doing nothing. Measuring from the command adds `depth` of real
    interference. Same family as `achieved_fraction` in the bench work.

    Applied ONCE, because it is relative: repeating it drives the pads arbitrarily deep, and the
    pad never reaches the commanded point so nothing stops it.
    """
    o = d.body(obj).xpos.copy()
    ax = d.body(obj).xmat.reshape(3, 3)[:, 2]
    dik.qpos[:] = d.qpos
    if not from_achieved:
        for a_i in range(m.nu):
            jid = m.actuator_trnid[a_i, 0]
            if jid >= 0 and m.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE,
                                                mujoco.mjtJoint.mjJNT_SLIDE):
                dik.qpos[mik.jnt_qposadr[jid]] = float(d.ctrl[a_i])
    dik.qvel[:] = 0.0
    mujoco.mj_forward(mik, dik)
    for f in FINGERS:
        df = float(depth[f]) if isinstance(depth, dict) else float(depth)
        if df == 0.0:
            continue
        t = dik.body(TIPS[f]).xpos.copy()
        rel = t - o
        v = rel - float(rel @ ax) * ax
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            continue
        ik_finger(mik, dik, f, t - (v / n) * df, iters=200)
    return {j: float(dik.qpos[mik.jnt_qposadr[mik.joint(j).id]]) for j in acts}


def _slide_cmd(m, mik, dik, d, obj: str, acts: dict, slide: dict) -> dict:
    """Move named pads ALONG the shaft's axis, from the commanded pose: `slide` is
    {finger: metres}, positive = away from the shaft's centre (toward the near end), which is
    how a pad is taken OFF a shaft when it has no radial room to retract -- the yaw joint has
    the travel the flexion joints lack at 95% extension. Same IK-from-command rule as
    `_squeeze_cmd`, for the same reason."""
    o = d.body(obj).xpos.copy()
    ax = d.body(obj).xmat.reshape(3, 3)[:, 2]
    dik.qpos[:] = d.qpos
    for a_i in range(m.nu):
        jid = m.actuator_trnid[a_i, 0]
        if jid >= 0 and m.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE,
                                            mujoco.mjtJoint.mjJNT_SLIDE):
            dik.qpos[mik.jnt_qposadr[jid]] = float(d.ctrl[a_i])
    dik.qvel[:] = 0.0
    mujoco.mj_forward(mik, dik)
    for f, sl in slide.items():
        if not sl:
            continue
        t = dik.body(TIPS[f]).xpos.copy()
        sgn = 1.0 if float((t - o) @ ax) >= 0.0 else -1.0
        ik_finger(mik, dik, f, t + sgn * float(sl) * ax, iters=200)
    return {j: float(dik.qpos[mik.jnt_qposadr[mik.joint(j).id]]) for j in acts}


def _support(m, d, obj: str):
    """Contacts between the tool and whatever is holding it up -- floor, table, or seat.

    NOT `probe_real_v1_gait._ground`, which asks for the body named "world". A countersink is a
    separate body, so on a screw scene that test reads zero contacts for a tool that is sitting
    firmly in its seat, and every "is it standing" gate downstream fails on a run that worked.
    Support here is anything the tool touches that is not the hand.
    """
    n, tot = 0, 0.0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        names = [m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name]
        if obj not in names:
            continue
        other = names[0] if names[1] == obj else names[1]
        if other.split("_")[0] in ("thumb", "index", "middle", "palm"):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        n += 1
        tot += abs(float(f6[0]))
    return n, tot


def _finger_act(m) -> dict:
    return {j: next(k for k in range(m.nu) if m.actuator(k).name == f"a_{j}")
            for joints in FINGERS.values() for j in joints}


# ------------------------------------------------------------------------------------- the run

def chain(morph_run: Path, obj: str = "screwdriver_medium",
          lift: float = 0.10, angle_deg: float = -90.0, axis_k: float = 0.25,
          turn_steps: int = 550, budget: float = 0.5, hold_steps: int = 500,
          gap: float = 0.002, press_mm: float = 2.0, carry_squeeze: float = 0.0,
          turn_squeeze: float = 0.0, turn_relief: float = 0.0,
          repose_steps: int = 800, repose_iters: int = 8,
          descend_steps: int = 400, descend_iters: int = 1,
          press_steps: int = 300, settle_steps: int = 400,
          transport_steps: int = 600, transport_iters: int = 1,
          stand_order: str = "ground", airgrip: str = "cradle",
          reindex: str = "full", relay_gait: bool = False, track_frac: float = 1.0,
          clear: float = 0.08,
          ring_az: str = "canonical",
          relay_squeeze: float = 0.0015,
          tilt_deg: float = 0.0, tilt_dir: float = 0.0, screw_torque: float = 0.0,
          centre_x: float = 0.004, grip_depth: float = 0.050,
          gait_depth: float | None = None,
          ring_z: float | None = None,
          stroke_deg: float = 30.0, cycles: int = 8, squeeze: float = 0.002,
          release_mm: float = 6.0, twist_steps: int = 120, move_steps: int = 60,
          approach_steps: int = 200, pad_radius: float | None = None,
          jitter: float = 0.0, seed: int = 0, no_floor_gait: bool = False,
          anchor_ctrl: dict | None = None,
          load_target: float = 0.0, load_gain: float = 0.0024, reg_band: float = 0.45,
          reg_every: int = 5, force_target: float = 0.0, force_gain: float = 0.0015,
          force_rate: float = 0.0006,
          track_gain: float = 0.0, track_rate: float = 0.01, track_every: int = 25,
          angle_gain: float = 0.0, angle_rate: float = 0.01, angle_from: str = "pressed",
          regrip_ref: str = "command", press_regrip: float = 0.0,
          regrasp: bool = False, regrasp_steps: int = 150,
          arm_ik: Path | None = None, scene_path: Path | None = None,
          place_xy=None, place_err=(0.0, 0.0), seat_z: float | None = None,
          tip_len: float = 0.0, seat_aim: str = "centre",
          video: Path | None = None, film: Path | None = None,
          cam=(120.0, -18.0, 0.36), video_every: int = 12, trace: bool = False,
          video_size=(640, 480), cam_look=None,
          gait_scan: str = "grip",
          step_hook=None, ctx: dict | None = None,
          turn_ctrl=None, close_ease_steps: int = 0, lift_ramp: int = 200,
          post_lift_settle: int = 200, arm_kp_scale: float = 1.0,
          finger_gravcomp: float | None = None) -> dict:
    """`turn_ctrl(k, m, d, acts, anchor, sq0) -> {joint: ctrl} | None`, if given, replaces the
    anchor sweep during the turn phase: it is asked every CONTROL_DECIMATION sim steps of the
    `turn_steps` turn and its last answer is held in between (a 50 Hz policy in the loop; the
    2026-09-19 chain test of the RL reorient policies). Everything before and after the turn --
    grasp, arm lift, hold, re-pose, set-down, gait -- runs as it always has, except that
    `close_ease_steps > 0` replaces the snap to the anchor with the RL environment's closure: an
    ease-out-quad lerp from the keyframe's open finger pose to the anchor over that many sim steps,
    the arm lift (`lift_ramp` sim steps) starting when the closure completes, and `post_lift_settle`
    sim steps of hold before the turn. The RL policies train on 240 / 80 / 260 (residual active at
    sim step 580), and the chain's own 0 / 200 / 200 arrives at the turn with a different grasp.

    `arm_kp_scale` multiplies the position gain of the six UR5e servos (kv unchanged). The
    menagerie servo (kp 2000 / kv 400 on the large joints, 500 / 100 on the wrist) is overdamped
    with its slow pole at kp/kv = 5 rad/s, so 0.52 s after an 80-step lift the palm still hangs
    4.9 mm below its command (wrist_1 10 mrad; measured 2026-09-19 on the D6 chain) and only
    reaches 0.04 mm after a further second. A UR5e told its payload holds ~0.1 mm at the end of
    a move, so the seam should be read on an arm that is where it was told to be: at 10 the slow
    pole is 50 rad/s and the palm is within 0.1 mm of its command 0.1 s after the lift.

    `finger_gravcomp`, if given, overrides the gravity compensation of the twelve finger bodies.
    `--payload-gravcomp` in the arm-scene builder compensates the whole hand subtree, fingers
    included, which relieves the finger SERVOS of their links' weight; on the bench the arm's
    payload compensation carries the hand at the flange and each finger servo still carries its
    own links, as the RL training scene has it (gravcomp 0 on the fingers). 0 restores that;
    with the arm stiffened the 2 N the fingers weigh moves the palm by ~0.01 mm."""
    scene = Path(scene_path) if scene_path is not None else \
        morph_run / ("arm_scene.xml" if arm_ik is not None else "frozen_scene.xml")
    pg._MODEL_PATH["path"] = str(scene)
    m = mujoco.MjModel.from_xml_path(str(scene))
    if arm_ik is not None and arm_kp_scale != 1.0:
        for n in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"):
            a = m.actuator(n).id
            m.actuator_gainprm[a, 0] *= arm_kp_scale     # kp
            m.actuator_biasprm[a, 1] *= arm_kp_scale     # -kp * qpos; biasprm[2] = -kv stays
    if finger_gravcomp is not None:
        for f in FINGERS:
            for seg in ("yaw_frame", "mcp_frame", "pip_frame", "tip"):
                b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{f}_{seg}")
                if b >= 0:
                    m.body_gravcomp[b] = float(finger_gravcomp)
    if pad_radius is not None:
        for f in FINGERS:
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == m.body(TIPS[f]).id:
                    m.geom_size[g, 0] = float(pad_radius)
                    m.geom_rbound[g] = float(pad_radius)
    pad_r = pg.PAD_RADIUS if pad_radius is None else float(pad_radius)
    d = mujoco.MjData(m)
    if ctx is not None:
        ctx["m"], ctx["d"] = m, d               # for a step_hook that needs the state before the turn
    r_obj, half = pg._obj_geom(m, obj)
    # Where the tool comes to rest once it is standing. On a plane that is its own half length;
    # in a countersink it is set by where the two cone radii match, and every "is it still
    # standing" test in this probe is written against it rather than against `half`.
    rest_z = float(half if seat_z is None else seat_z)
    floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # WHICH END GOES DOWN. A plain cylinder stands on either end, so every axis in this probe
    # was folded to point up and every tilt taken as arccos|cos|. A screwdriver cannot: its tip
    # is the -z end of the body, the countersink only accepts that end, and a tool standing on
    # its HANDLE reads as a perfect 0.00 deg stand under the folded convention -- which is what
    # a chain that leaves the rotation to the arm produces, because the arm's rigid transfer
    # picks the end to raise from the sign of R[2,2] and on a horizontal tool that sign is
    # numerical noise. With a tip, the axis is signed and 180 deg is 180 deg.
    tipped = tip_len > 0.0

    def _av() -> np.ndarray:
        R = d.body(obj).xmat.reshape(3, 3)
        return R[:, 2].copy() if tipped else R[:, 2] * (1.0 if R[2, 2] >= 0 else -1.0)

    def _tilt() -> float:
        return float(np.degrees(np.arccos(np.clip(_av()[2], -1.0, 1.0))))

    key = m.key("open_ik").id
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.ctrl[:] = m.key_ctrl[key]
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        adr = m.jnt_qposadr[m.body(obj).jntadr[0]]
        d.qpos[adr + 0] += float(rng.normal(0.0, jitter))
        d.qpos[adr + 1] += float(rng.normal(0.0, jitter))
    if anchor_ctrl is not None:
        # A Sobol-sampled hand has no CEM run: its grasp comes from `_grip_from_fit`, keyed by
        # joint name because the fit is solved on the design scene and replayed on the arm one.
        anchor = {j: float(anchor_ctrl[j]) for js in FINGERS.values() for j in js}
    else:
        closed = np.load(morph_run / "best_rollout.npz")["best_finger_ctrl"]
        anchor = {j: float(closed[i * 3 + k])
                  for i, (f, js) in enumerate(FINGERS.items()) for k, j in enumerate(js)}

    acts = _finger_act(m)
    open_q = {j: float(d.qpos[m.jnt_qposadr[m.joint(j).id]]) for j in acts}
    palm = pd.make(m, d, arm_ik)
    for j, a in acts.items():
        d.ctrl[a] = open_q[j] if close_ease_steps > 0 else anchor[j]
    mujoco.mj_forward(m, d)

    # CLOSED-LOOP GRIP, THROUGH THE WHOLE CHAIN. The deployed bench maneuver runs this only in
    # its hold phase, because regulating through the turn holds the shaft better and turns it
    # worse -- the last third of the rotation is the shaft settling into vertical against a grip
    # that has decayed, and a hand that keeps squeezing keeps it where it was. Here that trade is
    # taken deliberately: the settle is worth up to 30 deg on one hand and -17 on another, so it
    # is not something a chain can be built on, and a grasp that does not fail is worth more than
    # an alignment that arrives by luck.
    trim = {j: 0.0 for j in acts}
    reg = load_target > 0.0 or force_target > 0.0 or track_gain > 0.0 or angle_gain > 0.0
    angle_on = [angle_from == "start"]
    qadr = {j: m.jnt_qposadr[m.joint(j).id] for j in acts}
    if reg:
        import real_v1_deploy_envelope as de
    # WHERE THE PADS ARE, not how hard they push. `_regrasp_cmd` restores each pad's (station,
    # radius) in the SHAFT's frame and the chain fires it once, after the lift. Every deployed
    # hand slides 8.7-14.8 mm through the turn against the reference's 7.0 and keeps sliding at
    # every seam after it, so the grasp is migrating the whole run -- which a force loop cannot
    # see, because a pad that has walked 15 mm down the shaft can still be pushing 9 N. Filled
    # in once the grasp has settled; the tracker is a no-op until then.
    pad_track: dict = {"ref": None}

    def _regulate():
        # Two signals for the same job. The servo-load proxy is what the bench can actually
        # read; measured contact normal force is what the grip IS, and at the fractions of a
        # newton this chain's fitted grasp produces the load proxy barely moves, so the force
        # arm is the one that can tell whether closing the loop is worth anything at all.
        if force_target > 0.0:
            de._force_step(m, d, acts, trim, force_target, force_gain, reg_band, obj,
                           rate=force_rate)
        elif load_target > 0.0:
            de._load_step(m, d, acts, trim, load_target, load_gain, reg_band)
        # A THIRD signal, and the only one that answers the tool moving THROUGH the grasp:
        # drive the commanded pose toward the one that puts every pad back at the station and
        # radius it held at closure, on the tool's CURRENT frame. Slewed and clipped like the
        # others so it cannot overwrite the phase's own set-point in one tick.
        if track_gain > 0.0 and pad_track["ref"] is not None \
                and step_i[0] % track_every == 0:
            rg = _regrasp_cmd(m, mik, dik, d, obj, acts, pad_track["ref"])
            for j, a in acts.items():
                e = rg[j] - float(d.ctrl[a])
                trim[j] = float(np.clip(trim[j] + float(np.clip(track_gain * e,
                                                                -track_rate, track_rate)),
                                        -reg_band, reg_band))
        # A FOURTH, and the only one that reads what the bench's own servo bus reports: the
        # achieved joint angle. The SCS0009 is a spring (kp 0.5 on the calibrated plant) with
        # no integral term, so every finger that meets a load stops short of its set-point --
        # 25-36 deg behind through the 2026-09-16 handover and gait, with no actuator on its
        # ceiling. This arm commands the set-point plus `angle_gain` times the shortfall,
        # slewed `angle_rate` rad per tick and clipped to `reg_band`, which is the host-side
        # integral the servo lacks. It is armed from `angle_from` (the press, so the RL turn
        # runs as trained) and its falsifier is `sat` rising before `sp_err_deg` closes.
        if angle_gain > 0.0 and angle_on[0]:
            for j, a in acts.items():
                sp = float(d.ctrl[a]) - trim[j]
                want = float(np.clip(angle_gain * (sp - float(d.qpos[qadr[j]])), -reg_band, reg_band))
                trim[j] = float(trim[j] + np.clip(want - trim[j], -angle_rate, angle_rate))

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
        vid = mujoco.Renderer(m, height=int(video_size[1]), width=int(video_size[0]))
        vcam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, vcam)
        vcam.azimuth, vcam.elevation, vcam.distance = cam
        # The arm scene is 1.5 m across and the task is a 0.1 m corner of it, so the default
        # free camera frames the robot and renders the hand as a few pixels.
        if cam_look is not None:
            vcam.lookat[:] = cam_look

    # SUPPORT BOOKKEEPING. Everything after the tool is standing is judged on whether the
    # HAND ever handed it to the world. Armed at the start of the handover and sampled at the
    # control rate: `min_pads` is the worst instant, `free_frac` the share of the window in
    # which no pad was on the tool at all -- the share during which the only thing keeping it
    # where the chain put it is the floor's willingness to leave it there.
    watch = [False]
    sup = {"n": 0, "free": 0, "one": 0, "gnd": 0, "min": 3, "sum": 0}

    # A RESISTING TORQUE ABOUT THE TOOL'S OWN AXIS. A countersink is a friction brake, not a
    # thread: it gives back mu*N*r and nothing more, so on a plane or in a cone the load the
    # gait works against is whatever the press happens to buy. `--screw-torque` puts a named
    # one in, as a COULOMB brake -- it opposes whatever motion there is and does no work when
    # the tool is still, which is what a thread does and what a constant applied torque does
    # not (a constant torque spins the tool up the moment the fingers leave it).
    #
    # Two forms were tried and BOTH have a failure the other does not, so the probe reports a
    # guard rather than pretending one of them is safe everywhere. tau = clip(-I w / dt, +-t_r)
    # is dissipative by construction but useless here: the tool's axial inertia is 1.9e-6 kg
    # m^2 and dt is 2 ms, so the clip stops binding below 0.002 N m and every torque above that
    # returns the SAME answer -- the rotation is carried by the pads through contact, not by
    # the shaft's own momentum, and a brake sized by that momentum cannot oppose it.
    # -t_r*tanh(w/w0) does oppose it, and is what is used, but a brake that big on a body that
    # light can overshoot zero inside one step once the tool leaves the pads: at 0.05 N m the
    # shaft read +/-450 deg/cycle, past the 1.684 gear ceiling, i.e. the brake was pumping it.
    # `brake_pumped` flags exactly that, and a flagged cell is not a measurement.
    bid_obj = m.body(obj).id
    dadr_obj = m.jnt_dofadr[m.body(obj).jntadr[0]]
    brake = [0.0]
    BRAKE_W0 = 0.2

    def _integrate():
        ax = d.body(obj).xmat.reshape(3, 3)[:, 2]
        dadr = m.jnt_dofadr[m.body(obj).jntadr[0]]
        w_world = d.body(obj).xmat.reshape(3, 3) @ d.qvel[dadr + 3:dadr + 6]
        spin[0] += float(np.dot(w_world, ax)) * m.opt.timestep
        spin_w[0] += float(w_world[2]) * m.opt.timestep

    def _run(n, before=None, every_step=False):
        for k in range(n):
            if before is not None and (every_step or step_i[0] % CONTROL_DECIMATION == 0):
                # Strip the trim before the phase writes its set-point and put it back after,
                # so a phase that commands only SOME fingers (the relay walks one pad at a
                # time) neither loses the trim on the others nor gets it applied twice.
                if reg:
                    for j, a in acts.items():
                        d.ctrl[a] -= trim[j]
                before(k)
                if reg:
                    for j, a in acts.items():
                        d.ctrl[a] += trim[j]
            if reg and step_i[0] % reg_every == 0:
                pre = dict(trim)
                _regulate()
                for j, a in acts.items():
                    d.ctrl[a] += trim[j] - pre[j]
            if brake[0] > 0.0:
                R = d.body(obj).xmat.reshape(3, 3)
                ax = R[:, 2]
                w = float(np.dot(R @ d.qvel[dadr_obj + 3:dadr_obj + 6], ax))
                d.xfrc_applied[bid_obj, 3:6] = -brake[0] * np.tanh(w / BRAKE_W0) * ax
            mujoco.mj_step(m, d)
            _integrate()
            if step_hook is not None:
                step_hook(step_i[0])
            step_i[0] += 1
            if watch[0] and step_i[0] % CONTROL_DECIMATION == 0:
                npd = pg._hand(m, d, obj)[2]
                sup["n"] += 1
                sup["sum"] += npd
                sup["min"] = min(sup["min"], npd)
                sup["free"] += npd == 0
                sup["one"] += npd <= 1
                sup["gnd"] += _support(m, d, obj)[0] > 0
            if vid is not None and step_i[0] % video_every == 0:
                if cam_look is None:
                    vcam.lookat[:] = d.body(obj).xpos
                vid.update_scene(d, vcam)
                pg._mark(vid.scene, m, d, obj, r_obj, half)
                frames.append(vid.render())

    # DID THE TOOL MOVE IN THE HAND. A HELD reorient is one where it did not: the tool's pose
    # in the PALM's frame IS the grasp, and any change in it is the shaft rolling or sliding
    # through the pads. `roll_deg` is the total rotation of the tool relative to the palm since
    # the grasp settled, `slide_mm` the translation. Both stay at zero for a rigid carry however
    # far the arm moves, so they separate alignment the hand DELIVERED from alignment the shaft
    # helped itself to -- which the world-frame `tilt_deg` cannot.
    grip_ref: dict = {}

    def _grip_mark():
        Rp = d.body("palm_pose").xmat.reshape(3, 3)
        grip_ref["R"] = Rp.T @ d.body(obj).xmat.reshape(3, 3)
        grip_ref["p"] = Rp.T @ (d.body(obj).xpos - d.body("palm_pose").xpos)

    def _rel() -> dict:
        if not grip_ref:
            return {"roll_deg": 0.0, "slide_mm": 0.0}
        Rp = d.body("palm_pose").xmat.reshape(3, 3)
        dR = grip_ref["R"].T @ (Rp.T @ d.body(obj).xmat.reshape(3, 3))
        c = (float(np.trace(dR)) - 1.0) / 2.0
        po = Rp.T @ (d.body(obj).xpos - d.body("palm_pose").xpos)
        return {"roll_deg": round(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))), 2),
                "slide_mm": round(float(np.linalg.norm(po - grip_ref["p"])) * 1000, 1)}

    def _snap(name: str) -> dict:
        nh, fh, npd, fpd = pg._hand(m, d, obj)
        ng, fg = _support(m, d, obj)
        ax, tilt = _av(), _tilt()
        p = d.body(obj).xpos
        return {"phase": name, "t": round(step_i[0] * float(m.opt.timestep), 3),
                **_rel(),
                "cos": round(float(d.body(obj).xmat[8]), 4),
                "tilt_deg": round(tilt, 2), "z": round(float(p[2]), 4),
                "xy_mm": [round(float(p[0]) * 1000, 1), round(float(p[1]) * 1000, 1)],
                "hand_contacts": nh, "hand_force_N": round(fh, 2),
                "pad_contacts": npd, "pad_force_N": round(fpd, 2),
                "ground_contacts": ng, "ground_force_N": round(fg, 3),
                "spin_deg": round(float(np.degrees(spin[0])), 2),
                "palm_z": round(float(d.body("palm_pose").xpos[2]), 4),
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
                     d.body(obj).xmat.reshape(3, 3)[:, 2])) * 1000, 1) for f in FINGERS},
                # THE SERVO STATE, per finger: the largest commanded-minus-achieved angle across
                # the finger's joints, and how many of its actuators are on their force ceiling.
                # On the shipped plant (kp 30, +-10 N m) both read ~0 and the seam says nothing;
                # on the bench-calibrated plant (kp 0.5, +-0.35) they are the difference between
                # a finger that is holding and one that is stalled.
                "q_err_deg": {f: round(max(abs(float(np.degrees(
                    d.ctrl[acts[j]] - d.qpos[m.jnt_qposadr[m.joint(j).id]])))
                    for j in FINGERS[f] if j in acts), 1) for f in FINGERS},
                "sat": {f: sum(1 for j in FINGERS[f] if j in acts and
                               abs(float(d.actuator_force[acts[j]])) >=
                               0.98 * float(m.actuator_forcerange[acts[j], 1]))
                        for f in FINGERS},
                # The PHASE's set-point minus the achieved angle (the servo error above includes
                # whatever the regulator has added on top), and what it has added.
                "sp_err_deg": {f: round(max(abs(float(np.degrees(
                    d.ctrl[acts[j]] - trim[j] - d.qpos[qadr[j]])))
                    for j in FINGERS[f] if j in acts), 1) for f in FINGERS},
                "trim_deg": {f: round(max(abs(float(np.degrees(trim[j])))
                                         for j in FINGERS[f] if j in acts), 1) for f in FINGERS}}

    seams = []
    if film is not None:
        def _shot():
            if cam_look is None:
                vcam.lookat[:] = d.body(obj).xpos
            vid.update_scene(d, vcam)
            pg._mark(vid.scene, m, d, obj, r_obj, half)
            marks.append(vid.render())
    else:
        def _shot():
            return None

    # ---------------------------------------------------------------- 1. grasp, lift, settle
    if close_ease_steps > 0:
        def _close(k):
            t = min(1.0, k / float(close_ease_steps))
            al = 1.0 - (1.0 - t) ** 2
            for j, a in acts.items():
                d.ctrl[a] = (1.0 - al) * open_q[j] + al * anchor[j]
        _run(close_ease_steps, _close, every_step=True)
    else:
        _run(250)
    pad_ref = _pad_frame(m, d, obj)          # where the pads sit once the grasp has settled
    pad_track["ref"] = pad_ref
    _grip_mark()
    R_c, p_c = palm.cmd_pose()
    u0 = palm.read()
    u1 = palm.solve(R_c, p_c + np.array([0.0, 0.0, lift]))[0]
    _run(lift_ramp, lambda k: palm.write(u0 + (u1 - u0) * (k + 1) / lift_ramp), every_step=True)
    _run(post_lift_settle)
    seams.append(_snap("lifted"))
    if angle_from == "lifted":
        angle_on[0] = True
    _shot()

    # THE PADS MOVED, SO PUT THEM BACK. The grasp closes at ~41 N and is holding 0.33 N by the
    # top of the lift: the shaft creeps through the pads, the position error that IS the grip
    # force bleeds away, and every phase after this is measured through a grasp that has already
    # collapsed. One extra set-point, still open loop, referenced to where each pad sat in the
    # shaft's frame at closure rather than to how hard it was pushing.
    if regrasp:
        rg = _regrasp_cmd(m, mik, dik, d, obj, acts, pad_ref)
        st = {j: float(d.ctrl[a]) for j, a in acts.items()}
        _run(regrasp_steps,
             lambda k: [d.ctrl.__setitem__(a, st[j] + (rg[j] - st[j]) * (k + 1) / regrasp_steps)
                        for j, a in acts.items()], every_step=True)
        _run(100)
        seams.append(_snap("regrasped_lift"))
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
    if ctx is not None:
        # What the turn is made of, for a per-step probe: the achieved tips it rotates, the
        # pivot, the rigid-IK end pose and the grasp command it is added to.
        ctx["turn"] = {"tip0": {f: tip0[f].tolist() for f in FINGERS}, "centroid": centroid.tolist(),
                       "angle": float(angle), "q0": dict(q0), "end": dict(end), "anchor": dict(anchor),
                       "step0": int(step_i[0]), "turn_steps": int(turn_steps), "budget": float(budget),
                       "carry_ik_mm": float(carry_ik * 1e3)}

    # GRIP THROUGH THE TURN, or do not. `--carry-squeeze` closes the hand AFTER the hold, by
    # which time the shaft has already rolled itself upright inside a loose grasp; this closes
    # it BEFORE, as a constant joint-space offset carried through the sweep, so the shaft
    # arrives where the fingers put it. The two flags therefore ask different questions: one
    # buys time for the transport, the other suppresses the settle.
    sq0 = {j: 0.0 for j in acts}
    if turn_squeeze > 0.0:
        sq = _squeeze_cmd(m, mik, dik, d, obj, acts, turn_squeeze)
        sq0 = {j: sq[j] - float(d.ctrl[a]) for j, a in acts.items()}
    # RELIEVE THE DRIVERS, KEEP THE HOLDER. On the bench-calibrated plant (kp 0.5, 0.35 N m
    # ceiling) a 10 mm squeeze on all three fingers parks every yaw and mcp servo on its torque
    # ceiling holding the grip, and the turn then has no torque left to move anything: 0-10 deg
    # at every budget from 0.5 to 1.3 rad (docs/experiments/20260916-plant_budget). The bench
    # found the same stall on 2026-08-29 and the grip that turned was thumb firm / middle free
    # (loads 435/210/0). Index and middle are the fingers that travel in this turn -- they
    # straddle the shaft and describe the arcs about the pivot -- so they are the ones pulled
    # `turn_relief` radially out of the shaft, from the COMMANDED pose, as a constant offset
    # carried through the sweep. The thumb keeps the full squeeze and its moment arm.
    # A dict relieves named fingers by named amounts (m) -- {"middle": 0.02} opens the middle
    # 20 mm and keeps thumb + index as a two-point pinch, about which the tool is free to swing.
    if isinstance(turn_relief, dict):
        radial = {f: -float(v) for f, v in turn_relief.items() if not isinstance(v, (list, tuple))}
        axial = {f: float(v[1]) for f, v in turn_relief.items() if isinstance(v, (list, tuple))}
        if radial:
            rl = _squeeze_cmd(m, mik, dik, d, obj, acts, {f: radial.get(f, 0.0) for f in FINGERS})
            for j, a in acts.items():
                sq0[j] += rl[j] - float(d.ctrl[a])
        if axial:
            # from the pose the radial part leaves, so both compose
            for j, a in acts.items():
                d.ctrl[a] += sq0[j]
            sl = _slide_cmd(m, mik, dik, d, obj, acts, axial)
            for j, a in acts.items():
                d.ctrl[a] -= sq0[j]
                sq0[j] += sl[j] - (float(d.ctrl[a]) + sq0[j])
    elif turn_relief > 0.0:
        rl = _squeeze_cmd(m, mik, dik, d, obj, acts,
                          {"thumb": 0.0, "index": -turn_relief, "middle": -turn_relief})
        for j, a in acts.items():
            sq0[j] += rl[j] - float(d.ctrl[a])

    turn_cache: dict = {}

    def _turn(k):
        if turn_ctrl is not None:
            if k % CONTROL_DECIMATION == 0:
                c = turn_ctrl(k, m, d, acts, anchor, sq0)
                if c is not None:
                    turn_cache.update(c)
            for j, a in acts.items():
                if j in turn_cache:
                    d.ctrl[a] = turn_cache[j]
            return
        u = (k + 1) / turn_steps
        for j, a in acts.items():
            d.ctrl[a] = (anchor[j] + sq0[j]
                         + float(np.clip((end[j] - q0[j]) * u, -budget, budget)))

    _run(turn_steps, _turn, every_step=True)
    # THE SEAM THAT SEPARATES COMMAND FROM SLIP. Everything between this snapshot and the next
    # happens with the finger commands frozen, so the alignment gained in between is the shaft
    # rolling itself upright inside the grasp and nothing else.
    seams.append(_snap("turned"))
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
    r_grip = r_obj + pad_r - squeeze
    r_open = r_obj + pad_r + release_mm / 1000.0
    r_wide = r_obj + pad_r + 0.018
    gear = r_grip / r_obj
    stroke = np.radians(stroke_deg)
    phis = np.linspace(0.0, stroke, 25)
    if carry_squeeze > 0.0:
        # RE-GRIP BEFORE MOVING. The carry's terminal hold LEAKS: the shaft creeps down through
        # the pads at about 1.5 mm/s and the pad force decays to zero in ~1.6 s, so everything
        # after the turn is a race. On the flat floor one continuous set-down fits inside the
        # budget; an insertion (level, carry across, level, go down) does not, and this is what
        # buys the time.
        sq = _squeeze_cmd(m, mik, dik, d, obj, acts, carry_squeeze,
                          from_achieved=(regrip_ref == "achieved"))
        f0 = {j: float(d.ctrl[a]) - trim[j] for j, a in acts.items()}
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
        av = _av()
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
        """Lowest point of the tool: the bottom rim's low edge, or the cone apex if it is lower."""
        c = d.body(obj).xpos.copy()
        av = _av()
        u = np.array([0.0, 0.0, 1.0]) - av[2] * av
        n = float(np.linalg.norm(u))
        rim = c - half * av - (r_obj * (u / n) if n > 1e-9 else np.zeros(3))
        if tip_len <= 0.0:
            return rim
        apex = c - (half + tip_len) * av
        return apex if apex[2] < rim[2] else rim

    def _move_u(u1, steps_each, hold=True, settle=None):
        """Ramp the palm command (and, if holding, the pad ring) to one target vector."""
        c0 = palm.read()

        def _mv(k):
            v = min(1.0, (k + 1) / steps_each)
            palm.write(c0 + (np.asarray(u1, float) - c0) * v)

        _run(steps_each, _mv, every_step=True)
        _run(settle_steps // 4 if settle is None else settle)

    def _move(R, p_, steps_each, hold=True, settle=None):
        """Rigid palm move, INTERPOLATED IN CARTESIAN SPACE and re-solved at the control rate.

        Ramping the palm's JOINTS between two poses has the same shape as ramping its POSE only
        when the two are close. A 90 deg wrist reorientation is not close: the joint-space line
        between the two arm configurations swings the hand through an arc the Cartesian path
        never visits, and it drags the tool along it. That is what put the tool on the table at
        35 deg off vertical with the arm reporting an 0.008 mm IK residual -- the arm went
        exactly where it was told, by a route nobody chose, and no number of extra corrections
        moved it because each one re-walked the same arc. Interpolating the pose and re-solving
        each control step (the driver seeds every solve from the last, so the joint path stays
        continuous) makes the move the straight line the command says it is.
        """
        R0, p0 = palm.cmd_pose()
        R0, p0 = np.asarray(R0, float).copy(), np.asarray(p0, float).copy()
        R1, p1 = np.asarray(R, float), np.asarray(p_, float)
        n = max(1, steps_each)

        def _mv(k):
            t = min(1.0, (k + CONTROL_DECIMATION) / n)
            palm.write(palm.solve(_rot_lerp(R0, R1, t), p0 + (p1 - p0) * t)[0])

        _run(n, _mv)
        palm.write(palm.solve(R1, p1)[0])
        _run(settle_steps // 4 if settle is None else settle)

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

    def _descend(name, iters, steps_each, z_low_goal, foot_xy=None):
        """Lower the shaft, orientation untouched, until its low edge is `z_low_goal` off the floor.

        With `foot_xy` the descent also walks the MEASURED low point over that xy (the seat's
        centre), so the apex of a tool that is still leaning in the grip comes down on the hole
        and not beside it; the correction is `_gain(iters)` of the offset per iteration, as in
        `_upright`, and with iters=1 it is one aim taken at the top of the move.

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
        g = _gain(iters)
        for it in range(iters):
            R_obj = d.body(obj).xmat.reshape(3, 3).copy()
            po = d.body(obj).xpos.copy()
            low = _low_point()
            dz = (z0 + (z_low_goal - z0) * (it + 1) / iters) - float(low[2])
            dxy = (np.zeros(2) if foot_xy is None
                   else g * (np.asarray(foot_xy, float) - low[:2]))
            _move(*_rigid_palm_pose(m, d, obj, R_obj, po + np.array([dxy[0], dxy[1], dz])), steps_each)
            if DEBUG:
                print("   ", {**_snap(name), "dz_mm": round(dz * 1000, 1),
                              "z_low_mm": round(float(_low_point()[2]) * 1000, 1),
                              "palm_z": round(float(palm.cmd_pose()[1][2]), 4)})
            if _lost():
                break
        seams.append(_snap(name))
        _shot()

    def _transport(name, iters, steps_each):
        """Carry the tool sideways to the seat, at constant height and constant orientation.

        A SEPARATE PHASE, not folded into the descent, and the reason is the cone. Doing both at
        once drags the shaft in the grip -- a 1 deg exit tilt became 8 deg by the time the tip
        reached the mouth -- and a tilted cone entering a matched cone binds at 3 mm of a 10 mm
        seat and stops. Transport, re-level, then go straight down.
        """
        xy0 = d.body(obj).xpos[:2].copy()
        tgt = np.array([place_xy[0] + place_err[0], place_xy[1] + place_err[1]])
        for it in range(iters):
            R_obj = d.body(obj).xmat.reshape(3, 3).copy()
            po = d.body(obj).xpos.copy()
            want = xy0 + (tgt - xy0) * (it + 1) / iters
            _move(*_rigid_palm_pose(m, d, obj, R_obj,
                                    np.array([want[0], want[1], po[2]])), steps_each)
            if _lost():
                break
        seams.append(_snap(name))
        _shot()

    def _stage(name, steps_each, iters: int = 1):
        """Stand the tool vertical and carry it over the seat in the SAME move.

        The alternative -- level, transport, level again -- is four phases and about 7.6 s of
        hand-time, and the carry's terminal grasp is gone in 1.6 s (12.8 N at the top of the
        turn, 4.8 after the first levelling, 2.5 after the transport, 0 during the second).
        The rigid transfer can express both at once because it is a pose command, not a
        sequence, so the whole staging move is a single ramp and the budget is spent on one
        thing instead of three.

        ITERATED AGAINST THE MEASURED POSE, and that is the difference between a tool that
        enters the countersink and one that lands on the annulus beside it. The grip is
        compliant, so one open-loop transfer leaves several degrees of the levelling undone --
        5.0 deg measured on g12 -- and the tool then arrives over the socket pointing somewhere
        else. A 60 mm tip radius turns 5 deg into 5.2 mm of lateral tip offset against a 6 mm
        capture radius, and 16 deg (what the descent grew it to) into 16.5 mm, which misses.
        Dwells cost grip, which is why the DESCENT stays one continuous move; this phase is in
        the air with nothing under the tool, so it is the one that can afford them.
        """
        g = _gain(iters)
        for _ in range(iters):
            R_obj = d.body(obj).xmat.reshape(3, 3).copy()
            av = _av()
            th = float(np.arccos(np.clip(av[2], -1.0, 1.0)))
            w = np.cross(av, np.array([0.0, 0.0, 1.0]))
            nw = float(np.linalg.norm(w))
            if nw > 1e-9 and th > 1e-9:
                k = w / nw
                delta = g * th
                K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                R_des = (np.eye(3) + np.sin(delta) * K + (1 - np.cos(delta)) * (K @ K)) @ R_obj
            else:
                R_des = R_obj
            po = d.body(obj).xpos.copy()
            want = np.array([place_xy[0] + place_err[0], place_xy[1] + place_err[1]])
            # WHAT IS AIMED AT THE HOLE. `centre` carries the tool's body centre over the seat
            # and leaves the apex wherever the residual lean puts it, (half + tip) * sin(theta)
            # to the side -- 12 mm at the 14 deg the calibrated plant's grip stages at, twice
            # the seat's capture radius. `tip` aims the measured low point instead, so the
            # apex lands in the hole at whatever lean survives the levelling, and the lean is
            # taken out afterwards about the seated apex, with the floor carrying the weight.
            ref = _low_point()[:2] if seat_aim == "tip" else po[:2]
            xy = po[:2] + (want - ref) * g
            _move(*_rigid_palm_pose(m, d, obj, R_des,
                                    np.array([xy[0], xy[1], po[2]])), steps_each)
            if DEBUG:
                print("   ", {**_snap(name), "th_deg": round(float(np.degrees(th)), 2),
                              "apex_off_mm": round(float(np.linalg.norm(want - _low_point()[:2])) * 1000, 1)})
            if _lost():
                break
        seams.append(_snap(name))
        _shot()

    def _upright(name, iters, steps_each, about_foot: bool, foot_xy=None):
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
            av = _av()
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
            p_new = piv + R_corr @ (po - piv)
            if foot_xy is not None:
                # WALK THE FOOT TO THE SOCKET AS IT COMES UP. Standing a screwdriver TIP DOWN
                # cannot happen on a flat table -- the 10 mm cone has nowhere to go -- so the
                # seated version of this move has to arrive at the hole, and where the foot
                # ends up is not where it started: the contact migrates from the bottom rim to
                # the apex somewhere past 50 deg, which shifts it by the tool's own radius. A
                # correction on the MEASURED low point each iteration lands it without that
                # geometry having to be derived, and it is the same measurement the tilt
                # correction already uses.
                p_new = p_new + g * np.array([foot_xy[0] - piv[0], foot_xy[1] - piv[1], 0.0])
            _move(*_rigid_palm_pose(m, d, obj, R_corr @ R_obj, p_new), steps_each)
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
        f0 = {j: float(d.ctrl[a]) - trim[j] for j, a in acts.items()}
        _run(approach_steps, lambda k: [d.ctrl.__setitem__(
            a, f0[j] + (tgt_air[j] - f0[j]) * min(1.0, (k + 1) / approach_steps))
            for j, a in acts.items()], every_step=True)
        _run(settle_steps)
        seams.append(_snap("air_grip"))
        _shot()

    R0 = d.body(obj).xmat.reshape(3, 3)
    repose_deg = round(float(np.degrees(np.arccos(np.clip(abs(R0[2, 2]), -1, 1)))), 2)
    # Where the low point has to end up: `gap` above a plane, or `gap` above the depth the
    # seated tool's apex reaches in a countersink.
    z_low_goal = gap + (0.0 if tip_len <= 0.0 else rest_z - half - tip_len)
    if stand_order == "pivot":
        # NEVER OFF A SURFACE. Put the tool back down on the table LYING, then pivot it up about
        # its own foot, walking that foot into the socket as it rises. The reason is statics and
        # not preference: this grasp holds 0.4 N on a 24 g tool in mid-air, so a 90 deg mid-air
        # reorientation drops it whether the fingers or the arm do the rotating, on the UR5e and
        # on a floating palm alike. With the table under it the tool's weight is not the pads'
        # problem and they only have to keep it from falling over, which is the job the gait
        # study already showed they can do.
        _descend("set_down", descend_iters, max(1, descend_steps // descend_iters), gap)
        _upright("upright", repose_iters, max(1, repose_steps // repose_iters), True,
                 foot_xy=(None if place_xy is None else
                          (place_xy[0] + place_err[0], place_xy[1] + place_err[1])))
    elif place_xy is not None:
        # Insertion, in two moves. Levelling matters because a tilted cone binds in a matched
        # cone -- an 8 deg entry stops 3 mm into a 10 mm seat -- and the lateral carry has to
        # happen at height, but neither can be afforded its own phase.
        _stage("staged", max(1, transport_steps // max(1, transport_iters)), transport_iters)
        want = (place_xy[0] + place_err[0], place_xy[1] + place_err[1])
        _descend("set_down", descend_iters, max(1, descend_steps // descend_iters), z_low_goal,
                 foot_xy=(want if seat_aim == "tip" else None))
        if seat_aim == "tip" and not _lost():
            # The seat holds the apex; the lean the grip kept through the staging comes out
            # here, as a rotation about that apex and not about the tool's centre.
            _upright("seated", repose_iters, max(1, repose_steps // repose_iters), True,
                     foot_xy=want)
    elif stand_order == "ground":
        _descend("set_down", descend_iters, max(1, descend_steps // descend_iters), z_low_goal)
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
    if press_regrip > 0.0:
        # RE-REFERENCE THE GRIP ONCE THE SEAT CARRIES THE TOOL. The carry's command (an RL
        # residual frozen at its clip, or the fitter's anchor plus squeeze) is what held the
        # tool through the arm's re-pose, and on the calibrated plant that command is on the
        # servo's ceiling: 70-79 deg past the achieved angle at `turned`, `sat` 1-2 per finger.
        # Re-referencing it right after the turn (`regrip_ref="achieved"`) gives a 3-4 N grip
        # the levelling loses on two hands of three. Here the tool is in its seat and the
        # floor has the weight, so the command can be re-seeded from the achieved angles with
        # `press_regrip` of interference and the handover and gait start from a set-point the
        # servo can deflect toward instead of one it is already stalled against.
        sq = _squeeze_cmd(m, mik, dik, d, obj, acts, press_regrip, from_achieved=True)
        f0 = {j: float(d.ctrl[a]) - trim[j] for j, a in acts.items()}
        n_rr = max(1, settle_steps // 4)
        _run(n_rr, lambda k: [d.ctrl.__setitem__(
            a, f0[j] + (sq[j] - f0[j]) * min(1.0, (k + 1) / n_rr))
            for j, a in acts.items()], every_step=True)
        _run(settle_steps // 4)
        seams.append(_snap("re_referenced"))
        _shot()
    if angle_from == "pressed":
        angle_on[0] = True
    stood = seams[-1]
    stood_ok = bool(stood["ground_contacts"] >= 1 and stood["tilt_deg"] < 14.0
                    and abs(stood["z"] - rest_z) < 0.010)

    # A STANDING TOOL IS NOT A FIXTURED ONE. Everything downstream of this line in the shipped
    # chain assumed the tool stays exactly where it was set down for as long as the hand is off
    # it; on a bench nothing enforces that. `--tilt-deg` names the assumption and prices it: a
    # steady lateral load of tan(theta) of the tool's own weight, which is what a table out of
    # level by theta delivers, or a cable, or a nudge that does not go away. The free-standing
    # prediction is exact and worth stating before the sweep runs it -- a cylinder on its end
    # face topples when the overturning moment beats the restoring one, tan(theta) > r/half,
    # i.e. at atan(12.5/50) = 14.0 deg, the same number the gait study quotes for its stability
    # margin. A seat moves that threshold; so, differently, does never letting go.
    watch[0] = True
    brake[0] = float(screw_torque)
    if tilt_deg > 0.0:
        bid = m.body(obj).id
        az = np.radians(tilt_dir)
        f_lat = float(m.body_mass[bid]) * abs(float(m.opt.gravity[2])) * np.tan(np.radians(tilt_deg))
        d.xfrc_applied[bid, 0] = f_lat * np.cos(az)
        d.xfrc_applied[bid, 1] = f_lat * np.sin(az)
        _run(settle_steps // 2)
        seams.append(_snap("loaded"))
        _shot()

    # ---------------------------------------------------------- 3b. the palm move, and its cost
    def _cmd_qpos():
        """Load the scratch model with the COMMANDED configuration, not the achieved one."""
        dik.qpos[:] = d.qpos
        for a_i in range(m.nu):
            jid = m.actuator_trnid[a_i, 0]
            if jid >= 0 and m.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE,
                                                mujoco.mjtJoint.mjJNT_SLIDE):
                dik.qpos[mik.jnt_qposadr[jid]] = float(d.ctrl[a_i])
        dik.qvel[:] = 0.0
        mujoco.mj_forward(mik, dik)

    def _tool_frame():
        """Origin and orthonormal triad of the tool: axis up, plus two radial directions."""
        o = d.body(obj).xpos.copy()
        av = _av()
        e1 = np.array([1.0, 0.0, 0.0]) - av[0] * av
        n1 = float(np.linalg.norm(e1))
        e1 = e1 / n1 if n1 > 1e-9 else np.array([1.0, 0.0, 0.0])
        return o, av, e1, np.cross(av, e1)

    def _cyl(pt, frame):
        """A world point as (axial station, radius, azimuth) on the tool."""
        o, av, e1, e2 = frame
        rel = np.asarray(pt, float) - o
        st = float(rel @ av)
        v = rel - st * av
        return st, float(np.linalg.norm(v)), float(np.arctan2(v @ e2, v @ e1))

    def _uncyl(sra, frame):
        o, av, e1, e2 = frame
        st, r, az = sra
        return o + st * av + r * (np.cos(az) * e1 + np.sin(az) * e2)

    def _fly(u_tgt, steps, ends=None, iters=40):
        """Fly the palm to `u_tgt` with the pads pinned to the world points they hold now.

        The relay handover wants two things at once that a rigid hand cannot have: the palm
        over the tool, where the gait's ring is reachable, and the tool where it was set down.
        With two fingers loaded, moving the wrist drags the object -- across a plane that is
        drift, and out of a countersink that is the whole insertion undone.

        Resolving it is the FINGERS' job, not the wrist's. The palm follows the ramp, and at
        every control step each finger is re-solved to keep its pad on the same world point, so
        the wrist's motion is absorbed inside the hand and the contacts never move on the
        material. Pinned and seeded from the COMMAND, because on a position servo the pad's
        achieved pose already contains the deflection that is carrying the load: hold *that*
        fixed and the grip is handed back a millimetre at a time (18.4 N -> 0.25 N over seven
        reissues of a constant command, measured).
        """
        u0 = palm.read()
        _cmd_qpos()
        fr0 = _tool_frame()
        a0 = {f: _cyl(dik.body(TIPS[f]).xpos, fr0) for f in FINGERS}
        # `ends` turns the pin into a SLIDE. Pinning asks the hand to absorb the whole wrist
        # move with the contacts frozen, and this hand cannot -- it runs 15 mm short and the
        # pads come off. Sliding spends the same move going somewhere useful: each pad walks
        # from where the carry left it to its gait-ring station, and the interpolation is done
        # in the tool's own cylindrical coordinates so the path stays ON THE SURFACE. A straight
        # line in world space between two points on a cylinder is a chord: it passes inside the
        # tool, so the pads are commanded into the material and the transition jams.
        a1 = None if ends is None else {f: _cyl(ends[f], fr0) for f in FINGERS}
        if a1 is not None:
            for f in FINGERS:  # take the short way round
                a1[f] = (a1[f][0], a1[f][1],
                         a0[f][2] + float(np.arctan2(np.sin(a1[f][2] - a0[f][2]),
                                                     np.cos(a1[f][2] - a0[f][2]))))
        worst = [0.0]

        def _mv(k):
            u = min(1.0, (k + 1) / steps)
            palm.write(u0 + (np.asarray(u_tgt, float) - u0) * u)
            _cmd_qpos()
            fr = _tool_frame()
            for f in FINGERS:
                if a1 is None:
                    tgt = _uncyl(a0[f], fr)
                else:
                    tgt = _uncyl(tuple(a0[f][i] + (a1[f][i] - a0[f][i]) * u for i in range(3)), fr)
                worst[0] = max(worst[0], ik_finger(mik, dik, f, tgt, iters=iters))
                for kk, j in enumerate(FINGERS[f]):
                    d.ctrl[acts[j]] = float(dik.qpos[mik.jnt_qposadr[mik.joint(j).id]])

        _run(steps, _mv)
        _run(settle_steps // 4)
        return worst[0]

    def _cart_leg(f, sra0, sra1, n, iters=60):
        """One finger, one straight leg in the tool's cylindrical coordinates."""
        def _mv(k):
            u = min(1.0, (k + 1) / n)
            fr = _tool_frame()
            tgt = _uncyl(tuple(sra0[i] + (sra1[i] - sra0[i]) * u for i in range(3)), fr)
            _cmd_qpos()
            ik_finger(mik, dik, f, tgt, iters=iters)
            for j in FINGERS[f]:
                d.ctrl[acts[j]] = float(dik.qpos[mik.jnt_qposadr[mik.joint(j).id]])

        _run(n, _mv)

    def _relay_finger(f, end_pt, n):
        """Walk ONE pad from where it is to `end_pt`: off the tool, around it, back on.

        The move has to be Cartesian. Ramping the finger's JOINTS from its carry pose to its
        ring pose is the obvious implementation and it is the one that knocked the tool over on
        every fraction of the wrist move tried: the two poses straddle the shaft, so the
        straight line between them in joint space sweeps the pad THROUGH the material, and the
        tool leaves before the finger arrives. It is the same defect as the mid-air ring
        regrasp, which fails 6/6 for the same reason and was read as a mid-air problem.

        Three legs instead, in the tool's own cylindrical frame, so the pad is never commanded
        inside the surface: retract radially to the open radius at the station it already holds,
        travel to the new station and azimuth out there, then close back in. The other two
        fingers are not touched, so the tool is held throughout by the pair.
        """
        _cmd_qpos()
        fr = _tool_frame()
        s0, r0, az0 = _cyl(dik.body(TIPS[f]).xpos, fr)
        s1, r1, az1 = _cyl(end_pt, fr)
        az1 = az0 + float(np.arctan2(np.sin(az1 - az0), np.cos(az1 - az0)))
        _cart_leg(f, (s0, r0, az0), (s0, r_open, az0), max(1, n // 3))
        _cart_leg(f, (s0, r_open, az0), (s1, r_open, az1), max(1, n))
        _cart_leg(f, (s1, r_open, az1), (s1, r1, az1), max(1, n // 2))

    def _ring_here(centre_xy, z_r, palm_u, radii, phis_):
        """The gait's joint table, on the azimuths the PADS ACTUALLY OCCUPY.

        `pg._ring_table` puts the tripod at fixed WORLD bearings -- thumb at pi, index at pi/3,
        middle at -pi/3 -- which is the tripod a palm looking DOWN at a standing tool makes, and
        that is the pose the gait study validated. A chain that stands the tool up by pivoting
        it on the table does not leave the palm there: it leaves it on edge beside the tool, a
        quarter turn away, and the canonical ring is then 14.7 mm outside the fingers' reach.
        The handover commands that residual as interference, the pads arrive at 57-91 N, and the
        tool they were sent to take is shoved 17-26 deg off vertical by the taking of it.

        The gait needs a tripod that SWEEPS; it does not need one on a particular bearing. Built
        on the pads' own azimuths the ring is reachable by construction, and the stroke, the
        radii and the table's shape are untouched.
        """
        mikr = mujoco.MjModel.from_xml_path(str(scene))
        dikr = mujoco.MjData(mikr)
        mujoco.mj_resetDataKeyframe(mikr, dikr, mikr.key("open_ik").id)
        pg._palm_to(mikr, dikr, palm.joint_dict(palm_u))
        mujoco.mj_forward(mikr, dikr)
        az = {f: float(np.arctan2(dikr.body(TIPS[f]).xpos[1] - centre_xy[1],
                                  dikr.body(TIPS[f]).xpos[0] - centre_xy[0])) for f in FINGERS}
        tbl = {f: np.zeros((len(radii), len(phis_), 3)) for f in FINGERS}
        worst, per = 0.0, [0.0] * len(radii)
        for f, joints in FINGERS.items():
            qadr = [mikr.jnt_qposadr[mikr.joint(j).id] for j in joints]
            for ri, r in enumerate(radii):
                for pi, phi in enumerate(phis_):
                    a = az[f] + phi
                    tgt = np.array([centre_xy[0] + r * np.cos(a),
                                    centre_xy[1] + r * np.sin(a), z_r])
                    res = ik_finger(mikr, dikr, f, tgt, iters=200)
                    worst = max(worst, res)
                    per[ri] = max(per[ri], res)
                    tbl[f][ri, pi] = [float(dikr.qpos[q]) for q in qadr]
        return tbl, worst, per

    # ------------------------------------------------------------------ 4. take the gait grasp
    centre = d.body(obj).xpos[:2].copy()
    if ring_z is not None:
        z_ring = float(ring_z)
    elif reindex in ("relay", "none") and ring_az == "pads":
        # PUT THE RING WHERE THE HAND CAN HOLD IT. The mean pad height is where the pads are,
        # not where they can close: after the tool is stood up on the table the palm is on edge
        # beside it, and at the pads' own height the worst finger is 14.0 mm short of the grip
        # radius. It then hovers there -- 0.02 N, one pad on the tool, a gait that grips
        # nothing -- or, from a pose that can reach, arrives with 14 mm of commanded
        # interference and shoves the tool over. Scanning the shaft for the height whose ring
        # the fingers can actually close on costs three IK solves per candidate and removes
        # both failures. The gait does not care which band of the shaft it holds.
        best = (float("inf"), float(np.mean([d.body(TIPS[f]).xpos[2] for f in FINGERS])))
        for zc in np.arange(rest_z - half + 0.010, rest_z + half - 0.005, 0.005):
            _, w_, _ = _ring_here(centre, float(zc), palm.read(), [r_grip], [0.0])
            if w_ < best[0]:
                best = (w_, float(zc))
        z_ring = best[1]
    else:
        z_ring = float(np.mean([d.body(TIPS[f]).xpos[2] for f in FINGERS]))
    # Where the gait wants the palm: centred over the tool, `grip_depth` above the ring. Each
    # mode below either gets there or pays for staying -- and the payment is visible in one
    # number, the ring's IK residual, because a ring solved from the wrong palm pose is a ring
    # the fingers cannot reach.
    # 25 mm above the tool's own centre, MEASURED. The seated height is what it should be when
    # the insertion went in all the way; when it did not -- and a tool the hand stood up in a
    # countersink can be 10 mm proud -- a ring referenced to `rest_z` is 10 mm off the tool it
    # is meant to hold, which the gait palm scan below then cannot fix.
    z_gait = float(d.body(obj).xpos[2] + 0.025) if ring_z is None else float(ring_z)
    # HOW HIGH THE GAIT'S PALM SITS ABOVE ITS RING, and it is not the fit's grip depth. The fit
    # reports the palm height that pinches a tool LYING DOWN; the gait needs the height from
    # which three fingers can close on a ring around a tool STANDING UP, and on this hand those
    # are 64.5 mm and 45 mm. Handed the fit's number the ring solves 13.0 mm out of reach, the
    # close commands that as interference and knocks the tool over; at 45 mm the residual is
    # 0.43 mm and the same chain runs. Scanned rather than named, because it is a property of
    # each hand's reach, and the residual is the thing being minimised anyway.
    gd_use = float(grip_depth if gait_depth is None else gait_depth)
    if gait_depth is None and reindex in ("full", "regrip"):
        _wp, _wr, _nf = palm.worst_pos, palm.worst_rot, palm.fails
        best_gd = (float("inf"), gd_use)
        for gd in np.arange(0.030, 0.0721, 0.005):
            u_c = palm.solve(np.eye(3), np.array([centre[0] - centre_x, centre[1],
                                                  z_gait + float(gd)]))[0]
            jd = palm.joint_dict(u_c)
            # WHICH RING THE HEIGHT IS CHOSEN FOR. The close happens on `r_grip`, but the
            # palm DESCENDS onto the standing tool with the fingers at `r_open`, and it is
            # that descent that knocks the tool over -- 53 of 58 handover failures are
            # already past 14 deg at `reindexed`, having stood at 1.5 deg through `released`.
            # A height that solves the closed ring and not the open one drives the open pads
            # through the tool on the way down.
            w_ = pg._ring_table(m, centre, z_ring, jd, [r_grip], [0.0])[1]
            if gait_scan != "grip":
                w_o = pg._ring_table(m, centre, z_ring, jd, [r_open], [0.0])[1]
                w_ = w_o if gait_scan == "open" else max(w_, w_o)
            if w_ < best_gd[0]:
                best_gd = (w_, float(gd))
        gd_use = best_gd[1]
        palm.worst_pos, palm.worst_rot, palm.fails = _wp, _wr, _nf
    u_gait, ep_re, er_re = palm.solve(
        np.eye(3), np.array([centre[0] - centre_x, centre[1], z_gait + gd_use]))
    R_now, p_now = palm.cmd_pose()
    R_gait, p_gait = palm.fk(u_gait)
    palm_move_mm = round(float(np.linalg.norm(p_gait - p_now)) * 1000, 2)
    palm_move_deg = round(float(np.degrees(np.arccos(np.clip(
        (np.trace(R_gait @ R_now.T) - 1) / 2, -1.0, 1.0)))), 2)
    hold_ik_mm = float("nan")

    if reindex in ("track", "slide"):
        # THE HANDOVER WITHOUT A RELEASE. Fly the palm to the gait's own pose while all three
        # fingers hold their contacts fixed in the world, then walk them onto the ring one at a
        # time. This is the only mode that gets both the reachable ring AND an unbroken grasp;
        # `full` buys the ring by letting go, `relay` keeps the grasp and cannot reach.
        # HOW MUCH OF THE WRIST MOVE CAN THE HAND ABSORB? `--track-frac` sweeps it. At 0 the
        # palm does not move and the fingers keep every contact, but the ring is solved from
        # the pose the carry happened to leave and the pads cannot reach it cleanly; at 1 the
        # palm is where the gait wants it and the ring is exact, but the fingers run out of
        # reach on the way and the tool is briefly nobody's. The ring target itself is held
        # fixed across the sweep -- same centre, same height -- so the residual curve is the
        # wrist move's doing and nothing else's.
        z_ring = z_gait
        u0_go = palm.read()
        u_go = u0_go + (np.asarray(u_gait, float) - u0_go) * float(track_frac)
        ends = None if reindex == "track" else {
            f: np.array([centre[0] + r_grip * np.cos(pg.AZIMUTH[f]),
                         centre[1] + r_grip * np.sin(pg.AZIMUTH[f]), z_ring]) for f in FINGERS}
        hold_ik_mm = _fly(u_go, repose_steps, ends=ends) * 1000
        R_go, p_go = palm.fk(u_go)
        palm_move_mm = round(float(np.linalg.norm(p_go - p_now)) * 1000, 2)
        palm_move_deg = round(float(np.degrees(np.arccos(np.clip(
            (np.trace(R_go @ R_now.T) - 1) / 2, -1.0, 1.0)))), 2)
        seams.append(_snap("tracked"))
        _shot()
        centre = d.body(obj).xpos[:2].copy()
        table, ik_res, per_r = pg._ring_table(
            m, centre, z_ring, palm.joint_dict(u_go), [r_grip, r_open, r_wide], phis)
    elif reindex in ("full", "regrip"):
        # THE FLOOR MAKES THE HAND FREE. Let go completely, drive the palm to the pose the gait
        # study validated, and take the shaft again as a ring. Nothing else in this program can
        # do this: every other release in the repertoire drops the object.
        z_ring, u_tgt = z_gait, u_gait
        table, ik_res, per_r = pg._ring_table(
            m, centre, z_ring, palm.joint_dict(u_tgt), [r_grip, r_open, r_wide], phis)
        open_ctrl = {j: float(m.key_ctrl[key][a]) for j, a in acts.items()}
        cur_f = {j: float(d.ctrl[a]) - trim[j] for j, a in acts.items()}
        _run(approach_steps // 2, lambda k: [
            d.ctrl.__setitem__(a, cur_f[j] * (1 - min(1.0, (k + 1) / (approach_steps // 2)))
                               + open_ctrl[j] * min(1.0, (k + 1) / (approach_steps // 2)))
            for j, a in acts.items()], every_step=True)
        _run(settle_steps // 4)
        seams.append(_snap("released"))
        _shot()
        # OVER THE TOP, NOT THROUGH. The re-index is the largest rotation the wrist makes in the
        # whole chain -- from wherever the reorient parked the palm, on edge beside the tool, to
        # the gait's own pose looking down at it -- and it is where the chain is lost: 29 of 36
        # runs that stood the tool are past 14 deg by the end of this move, with the hand OPEN
        # and the ring not yet closed, mean tilt 1.15 deg before it and 41.4 after. A single
        # Cartesian leg rotates and translates at once, which walks the open fingers through the
        # space the standing tool occupies. Three legs instead: straight up to a height where
        # the fingertips clear the tool's top, across and around at that height, then straight
        # down onto the gait pose. Same endpoint, and nothing passes through the tool.
        p_g = np.array([centre[0] - centre_x, centre[1], z_gait + gd_use])
        R_c, p_c = palm.cmd_pose()
        z_hi = max(float(p_c[2]), float(p_g[2])) + clear
        n3 = max(1, repose_steps // 3)
        _move(R_c, np.array([p_c[0], p_c[1], z_hi]), n3, hold=False, settle=settle_steps // 8)
        _move(np.eye(3), np.array([p_g[0], p_g[1], z_hi]), n3, hold=False,
              settle=settle_steps // 8)
        _move(np.eye(3), p_g, n3, hold=False, settle=settle_steps // 2)
        seams.append(_snap("reindexed"))
        _shot()
    else:
        # No luxury: solve the ring at the palm pose the carry actually left, and regrasp from
        # the carry's own finger pose. If this is unreachable the residual says so directly.
        _rt = _ring_here if ring_az == "pads" else (
            lambda c, z_, u, rr, ph: pg._ring_table(m, c, z_, palm.joint_dict(u), rr, ph))
        table, ik_res, per_r = _rt(centre, z_ring, palm.read(), [r_grip, r_open, r_wide], phis)

    wide = {f: pg._lookup(table, f, 2, 0.0, phis) for f in FINGERS}
    grip_t = {f: pg._lookup(table, f, 0, 0.0, phis) for f in FINGERS}
    n_app = max(1, approach_steps // 2)

    def _ramp(f, dst, n):
        """Drive ONE finger from wherever its command is to `dst`, leaving the others alone."""
        src = [float(d.ctrl[acts[j]]) - trim[j] for j in FINGERS[f]]

        def _mv(k):
            u = min(1.0, (k + 1) / n)
            for kk, j in enumerate(FINGERS[f]):
                d.ctrl[acts[j]] = src[kk] * (1 - u) + float(dst[kk]) * u

        _run(n, _mv, every_step=True)

    if reindex in ("relay", "track", "slide", "regrip"):
        # THE HANDOVER, ONE FINGER AT A TIME. `full` and `none` both take the ring grasp with
        # all three fingers at once, which means both pass through a moment with every pad off
        # the tool -- `full` deliberately (it opens the hand and flies the palm somewhere else),
        # `none` incidentally (its reach phase sends all three out to the wide radius together).
        # Either way the tool is briefly the floor's problem, and that is the assumption being
        # removed here: move one finger to its ring station, close it, and only then release the
        # next. Two pads are on the tool at every instant, so the transition itself is a grasp
        # the hand could hold, not a release it happens to survive.
        #
        # It costs the palm move. `full` re-centres the palm over the tool before regrasping,
        # which is what makes its ring solve cleanly; a relay cannot, because moving the palm
        # while two fingers are loaded drags the tool across the seat it was just set into. So
        # the relay inherits `none`'s ring -- solved at whatever pose the carry parked the palm
        # -- and the IK residual it reports is the price of not being allowed to let go.
        _az = {f: pg.AZIMUTH[f] for f in FINGERS} if ring_az != "pads" else {
            f: float(np.arctan2(d.body(TIPS[f]).xpos[1] - centre[1],
                                d.body(TIPS[f]).xpos[0] - centre[0])) for f in FINGERS}
        ring_pt = {f: np.array([centre[0] + r_grip * np.cos(_az[f]),
                                centre[1] + r_grip * np.sin(_az[f]), z_ring])
                   for f in FINGERS}
        for f in FINGERS:
            if reindex != "slide":
                # A slide has already walked the pads onto the ring, so it only trims onto the
                # table's joint targets; everything else has to get there one finger at a time.
                _relay_finger(f, ring_pt[f], n_app)
            _run(settle_steps // 4)
            seams.append(_snap(f"handover_{f}"))
            _shot()
        _run(settle_steps)
        # SET THE GRIP EXPLICITLY, ONCE, ONCE ALL THREE ARE ON. The Cartesian legs land each pad
        # on the ring, which is nominally `squeeze` inside the surface -- but the tool moves as
        # each finger arrives, so the pad that got there first has been relieved by the time the
        # third lands (7.6 -> 4.9 -> 0.3 N, measured). Arrival order cannot set a grip. One
        # relative squeeze from the COMMANDED pose does, and it is the same primitive the carry
        # uses: measure from the command, not from the achieved pose that already contains the
        # deflection carrying the load.
        if relay_squeeze > 0.0:
            sq = _squeeze_cmd(m, mik, dik, d, obj, acts, relay_squeeze)
            f0 = {j: float(d.ctrl[a]) - trim[j] for j, a in acts.items()}
            _run(n_app, lambda k: [d.ctrl.__setitem__(
                a, f0[j] + (sq[j] - f0[j]) * min(1.0, (k + 1) / n_app))
                for j, a in acts.items()], every_step=True)
            _run(settle_steps // 2)
            seams.append(_snap("handover_grip"))
            _shot()
        # REBASE THE GAIT TABLE ON WHERE THE FINGERS ACTUALLY ARE. The table is solved from a
        # palm pose the relay never reaches, so its phi=0 entry is a few millimetres off the
        # ring; snapping onto it after a clean Cartesian arrival commands that error as
        # interference, and on a position servo interference IS grip -- 20 N of it, enough to
        # pick the tool up off the floor it is supposed to be standing on. Offsetting the whole
        # table by the difference keeps the sweep the gait study validated and starts it from
        # the grasp the hand has, which is the one that is holding the tool.
        for f in FINGERS:
            off = np.array([float(d.ctrl[acts[j]]) - trim[j] for j in FINGERS[f]]) - table[f][0, 0]
            table[f] += off
    else:
        cur_f = {j: float(d.ctrl[a]) - trim[j] for j, a in acts.items()}

        def _reach(k):
            u = min(1.0, (k + 1) / n_app)
            for f in FINGERS:
                for kk, j in enumerate(FINGERS[f]):
                    d.ctrl[acts[j]] = cur_f[j] * (1 - u) + float(wide[f][kk]) * u

        def _close(k):
            u = min(1.0, (k + 1) / n_app)
            for f in FINGERS:
                for kk, j in enumerate(FINGERS[f]):
                    d.ctrl[acts[j]] = float(wide[f][kk]) * (1 - u) + float(grip_t[f][kk]) * u

        _run(n_app, _reach, every_step=True)
        _run(settle_steps // 4)
        _run(n_app, _close, every_step=True)
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
    z_gait0 = float(d.body(obj).xpos[2])
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
        if relay_gait:
            # RELEASE / RETURN / REGRASP ONE FINGER AT A TIME. The other two stay closed at the
            # end of the stroke, so the tool is held by two pads through the whole recovery and
            # the gait stops depending on it standing unaided. It is the same schedule as the
            # relay handover, run once per cycle instead of once per task, which is why the two
            # are not really separate mechanisms: a handover is a relay whose target ring moved.
            for f in FINGERS:
                _run(move_steps, lambda k, f=f: _cmd(f, 1, stroke))
                _run(move_steps, lambda k, f=f: _cmd(
                    f, 1, stroke * max(0.0, 1 - (k + 1) / move_steps)))
                _run(move_steps, lambda k, f=f: _cmd(f, 0, 0.0))
        else:
            _run(move_steps, lambda k: [_cmd(f, 1, stroke) for f in FINGERS])
            _run(move_steps, lambda k: [_cmd(f, 1, stroke * max(0.0, 1 - (k + 1) / move_steps))
                                        for f in FINGERS])
            _run(move_steps, lambda k: [_cmd(f, 0, 0.0) for f in FINGERS])
        _run(move_steps)
        _, tilt = pg._axis_tilt(m, d, obj)
        nh, fh, npd, fpd = pg._hand(m, d, obj)
        ng, fg = _support(m, d, obj)
        p = d.body(obj).xpos
        if tilt > 45.0 or float(p[2]) < rest_z * 0.5:
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
    ng, fg = _support(m, d, obj)
    seams.append(_snap("gaited"))
    _shot()
    # ---------------------------------------------------------- the reorientation, on its own
    # The chain's gates are an AND, so a hand that reorients perfectly and then loses the
    # handover reads exactly like a hand that never turned the tool. These four fields score
    # the REORIENTATION alone, on the two things that can go wrong with it: the tool leaving
    # the hand, and the turn coming up short.
    #   reorient_deg  how much of the 90 deg lying->standing turn the tool actually made, read
    #                 at `upright` (the pivot's own result) before the press touches it.
    #   drop_stage    the first seam at which the tool is on the floor with nothing holding it.
    #                 Everything up to `pressed` is supposed to be held or supported, so any
    #                 name here is a drop; None means the tool was never loose.
    #   reorient_ok   turned to within 14 deg of vertical AND never dropped getting there.
    _ph = {s_["phase"]: s_ for s_ in seams}
    _up = _ph.get("upright", _ph.get("staged"))
    reorient_deg = None if _up is None else round(90.0 - float(_up["tilt_deg"]), 2)
    drop_stage = None
    for s_ in seams:
        if s_["phase"] == "released":
            break
        # loose = lying on the floor (centre at the shaft RADIUS, not the half-length -- a
        # tool standing on its end sits at `half` and is not dropped) with no hand on it
        if s_["hand_contacts"] < 1 and s_["z"] < r_obj + 0.004:
            drop_stage = s_["phase"]
            break
    # A REORIENTATION IS ONLY A REORIENTATION IF THE HAND IS STILL ON THE TOOL. Without this
    # clause the metric scores a tool that was set down and released and happened to settle
    # vertical: over 256 table-stand runs, 116 of the 177 that passed `stood_ok` had ZERO pads
    # on the tool at `upright` and only 4 had two. `stood_ok` tests ground contact and tilt and
    # never tested the grasp, and `reorient_deg` inherited that. Same defect as reading peak_cos
    # without final_z, and as a dropped shaft reading vertical in a countersink.
    upright_pads = 0 if _up is None else int(_up.get("pad_contacts") or 0)
    upright_force = 0.0 if _up is None else float(_up.get("pad_force_N") or 0.0)
    held_reorient = bool(upright_pads >= 2 and upright_force > 1.0)
    reorient_ok = bool(reorient_deg is not None and reorient_deg >= 76.0
                       and drop_stage is None and held_reorient)

    out = {
        "run": morph_run.name, "object": obj, "reindex": reindex,
        "palm_move_mm": palm_move_mm, "palm_move_deg": palm_move_deg,
        "track_frac": track_frac, "relay_squeeze_mm": round(relay_squeeze * 1000, 2),
        "hold_ik_mm": None if hold_ik_mm != hold_ik_mm else round(hold_ik_mm, 2),
        "relay_gait": bool(relay_gait), "tilt_deg": tilt_deg, "tilt_dir": tilt_dir,
        "screw_torque": screw_torque,
        # The support window: from the moment the tool is standing to the end of the gait.
        "sup_steps": sup["n"], "min_pads": None if not sup["n"] else int(sup["min"]),
        "mean_pads": None if not sup["n"] else round(sup["sum"] / sup["n"], 2),
        "free_frac": None if not sup["n"] else round(sup["free"] / sup["n"], 3),
        "one_frac": None if not sup["n"] else round(sup["one"] / sup["n"], 3),
        "ground_frac": None if not sup["n"] else round(sup["gnd"] / sup["n"], 3),
        "angle_deg": angle_deg, "axis_k": axis_k, "turn_steps": turn_steps,
        "hold_steps": hold_steps, "descend_steps": descend_steps, "lift": lift,
        "gap_mm": gap * 1000, "press_mm": press_mm, "grip_depth": grip_depth,
        "carry_squeeze_mm": carry_squeeze * 1000,
        "load_target": load_target, "force_target": force_target, "reg_band": reg_band,
        "angle_gain": angle_gain, "angle_rate": angle_rate, "angle_from": angle_from,
        "regrip_ref": regrip_ref, "carry_squeeze_mm": carry_squeeze * 1000,
        "press_regrip_mm": press_regrip * 1000,
        "regrasp": bool(regrasp),
        "trim_max_deg": round(float(np.degrees(max(abs(v) for v in trim.values()))), 2),
        "turn_squeeze_mm": turn_squeeze * 1000, "turn_relief_mm": ({f: (v * 1000 if not isinstance(v, (list, tuple)) else [v[0], v[1] * 1000]) for f, v in turn_relief.items()} if isinstance(turn_relief, dict) else turn_relief * 1000),
        # HOW MUCH THE TOOL MOVED IN THE HAND, worst case from the settled grasp to the
        # moment it stands. Past "pressed" the chain is deliberately changing grasp, so the
        # measure stops there. A held carry reads ~0 on both however far the arm travels.
        "roll_max_deg": round(max([s_["roll_deg"] for s_ in seams[:1 + max(
            (i for i, s_ in enumerate(seams) if s_["phase"] == "pressed"), default=0)]],
            default=0.0), 2),
        "slide_max_mm": round(max([s_["slide_mm"] for s_ in seams[:1 + max(
            (i for i, s_ in enumerate(seams) if s_["phase"] == "pressed"), default=0)]],
            default=0.0), 1),
        # The controlled slip, in degrees of alignment the hand did not command.
        "tilt_turned_deg": round(float(
            [s for s in seams if s["phase"] == "turned"][0]["tilt_deg"]), 2),
        "tilt_settled_deg": round(float(
            [s for s in seams if s["phase"] == "reoriented"][0]["tilt_deg"]), 2),
        "settle_deg": round(float(
            [s for s in seams if s["phase"] == "turned"][0]["tilt_deg"]
            - [s for s in seams if s["phase"] == "reoriented"][0]["tilt_deg"]), 2),
        "repose_iters": repose_iters, "descend_iters": descend_iters,
        "stand_order": stand_order, "airgrip": airgrip, "seat_aim": seat_aim,
        "seat_z": round(rest_z, 5), "tip_len_mm": round(tip_len * 1000, 2),
        "apex_seat_offset_mm": round(float(np.linalg.norm(
            _low_point()[:2] - np.asarray(place_xy, float))) * 1000, 2)
            if place_xy is not None else None,
        "place_xy": None if place_xy is None else [round(float(v), 5) for v in place_xy],
        "place_err_mm": [round(float(v) * 1000, 2) for v in place_err],
        "seat_offset_mm": round(float(np.linalg.norm(
            d.body(obj).xpos[:2] - np.asarray(place_xy, float))) * 1000, 2)
            if place_xy is not None else None,
        "air_ik_mm": None if air_ik != air_ik else round(air_ik * 1000, 2),
        "centre_x": centre_x, "ring_z": round(z_ring, 4),
        "gait_depth_mm": round(gd_use * 1000, 1), "squeeze_mm": squeeze * 1000,
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
        "reorient_deg": reorient_deg, "drop_stage": drop_stage,
        "upright_pads": upright_pads, "upright_force_N": round(upright_force, 2),
        "held_reorient": held_reorient,
        "reorient_ok": reorient_ok,
        "spin_deg": round(float(np.degrees(spin[0])), 2),
        "turns": round(float(np.degrees(spin[0])) / 360.0, 3),
        "gain_mean_deg": round(float(np.mean(gains)), 2) if gains else 0.0,
        "gain_sd_deg": round(float(np.std(gains)), 2) if gains else 0.0,
        "transmission": round(float(np.mean(gains)) / (gear * stroke_deg), 3) if gains else 0.0,
        # A brake that pumps instead of resisting shows up as a turn past the no-slip ceiling.
        # Flagged rather than silently dropped: the flag IS the validity limit of the knob.
        "brake_pumped": bool(screw_torque > 0.0 and gains
                             and abs(float(np.mean(gains))) > 1.05 * gear * stroke_deg),
        "final_tilt_deg": round(tilt, 2),
        "final_z": round(float(d.body(obj).xpos[2]), 4),
        # Axial descent through the grasp, per cycle. With the floor there this is ~0; with it
        # deleted it separates a hand that DROPPED the tool from one that is still holding it
        # and lowering it, which the `ok` gate cannot tell apart.
        "z_gait0": round(z_gait0, 4),
        "slip_mm_per_cycle": round((z_gait0 - float(d.body(obj).xpos[2])) * 1000
                                   / max(1, len(per_cycle)), 2),
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
    ap.add_argument("--scene-path", type=Path, default=None,
                    help="the task scene to roll out; defaults to <run>/arm_scene.xml when an "
                         "IK model is given, else the run's frozen_scene.xml")
    ap.add_argument("--screw", type=Path, default=None,
                    help="a build_screw_scene.py JSON. Sets the scene, the seat "
                         "coordinates, the seated height and the tip length together, "
                         "because getting one of them wrong is silent.")
    ap.add_argument("--transport-steps", type=int, default=600)
    ap.add_argument("--transport-iters", type=int, default=1)
    ap.add_argument("--place-err", default="0,0",
                    help="mm of deliberate lateral error in the insertion target")
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
    ap.add_argument("--turn-squeeze", type=float, default=0.0,
                    help="metres of extra radial closure held through the reorienting turn. "
                         "Suppresses the shaft's settle to vertical; --carry-squeeze, which "
                         "closes after the hold, does not.")
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
    ap.add_argument("--seat-aim", default="centre", choices=("centre", "tip"),
                    help="what the seated set-down carries over the socket: the tool's body centre "
                         "(shipped) or its measured apex, which is then stood up about the seat")
    ap.add_argument("--angle-gain", type=float, default=0.0,
                    help="achieved-joint-angle loop: command = set-point + gain * (set-point - achieved), "
                         "the host-side integral the SCS0009 lacks; 0 = off (shipped)")
    ap.add_argument("--angle-rate", type=float, default=0.01, help="rad per regulator tick the angle trim may move")
    ap.add_argument("--angle-from", default="pressed", choices=("start", "lifted", "pressed"),
                    help="the seam after which the angle loop is armed")
    ap.add_argument("--regrip-ref", default="command", choices=("command", "achieved"),
                    help="what the post-turn regrip's squeeze is measured from: the command (shipped; a "
                         "millimetre of deflection on the kp-30 plant) or the achieved joint angles")
    ap.add_argument("--press-regrip", type=float, default=0.0,
                    help="mm of pad interference to re-seed every finger command from the ACHIEVED angles "
                         "once the tool is pressed into its seat; 0 = keep the carry's command (shipped)")
    ap.add_argument("--stand-order", default="ground", choices=("ground", "air", "pivot"),
                    help="ground = set the tilted shaft's foot on the floor, then rotate it "
                         "upright about that foot; air = stand it up in mid-air first")
    ap.add_argument("--reindex", default="full",
                    choices=("full", "regrip", "slide", "track", "relay", "none"),
                    help="how the carry's grasp becomes the gait's. full = let go on the "
                         "floor, re-centre the palm and retake the ring; relay = walk the "
                         "fingers onto the ring ONE AT A TIME with the palm parked, so two "
                         "pads are on the tool at every instant; none = release and retake "
                         "all three together at the carry's palm pose. full and none both "
                         "pass through zero contacts and so both need the tool to stay put "
                         "on its own.")
    ap.add_argument("--clear", type=float, default=0.08,
                    help="m the re-index lifts the open hand above the gait pose before it "
                         "translates across. 0 collapses the three legs back into the single "
                         "Cartesian move that walks the open fingers through the standing tool.")
    ap.add_argument("--relay-squeeze", type=float, default=0.0015,
                    help="m of radial interference commanded once a relay handover has all "
                         "three pads on the ring. Arrival order cannot set a grip; this can.")
    ap.add_argument("--track-frac", type=float, default=1.0,
                    help="fraction of the wrist move to the gait's palm pose that a track/slide "
                         "handover actually makes. 0 keeps every contact and cannot reach the "
                         "ring; 1 reaches the ring and drops the contacts on the way.")
    ap.add_argument("--relay-gait", action="store_true",
                    help="run the gait's release/return/regrasp one finger at a time as well, "
                         "so the whole task after the carry never drops below two pads")
    ap.add_argument("--tilt-deg", type=float, default=0.0,
                    help="steady lateral load on the tool once it is standing, as the table "
                         "tilt that would produce it: |F| = m g tan(theta). A free-standing "
                         "cylinder topples past atan(r/half) = 14.0 deg whatever the hand is "
                         "doing elsewhere; this is the knob that stops the probe assuming a "
                         "set-down tool stays where it was set down.")
    ap.add_argument("--screw-torque", type=float, default=0.0,
                    help="N.m of Coulomb brake about the tool's own axis, armed once the tool "
                         "is standing. A cone gives back mu*N*r and no more; this is the load "
                         "a thread would add on top, and sweeping it to zero turns is the "
                         "stall torque.")
    ap.add_argument("--tilt-dir", type=float, default=0.0,
                    help="azimuth of that load in world degrees (0 = +x, away from the palm)")
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
    ap.add_argument("--cam", default="120,-18,0.36",
                    help="azimuth,elevation,distance")
    ap.add_argument("--cam-look", default=None, help="x,y,z the camera points at")
    ap.add_argument("--video-size", default="640,480")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = args.morph_run or [ROOT / "results/phase1/real_v1/rv05_manual_stored"]
    arm_ik = args.arm_ik or ((runs[0] / "arm_ik.xml") if args.arm else None)
    screw = json.loads(args.screw.read_text()) if args.screw else {}
    scene_path = args.scene_path or (Path(screw["scene"]) if screw else None)
    pe = [float(v) / 1000.0 for v in args.place_err.split(",")]
    cam = tuple(float(v) for v in args.cam.split(","))
    rows = []
    print(f"{'run':22} {'idx':5} {'press':>6} {'ringIK':>7} {'repose':>7} "
          f"{'turns':>6} {'deg/cy':>7} {'tilt':>6} {'drift':>7} {'free':>5} {'cyc':>5}  ok")
    for run in runs:
        for press in (float(v) for v in str(args.press).split(",")):
            for rep in range(args.repeats):
                r = chain(run, obj=args.object_body, lift=args.lift, angle_deg=args.angle_deg,
                          axis_k=args.axis_k, turn_steps=args.turn_steps, budget=args.budget,
                          hold_steps=args.hold_steps, gap=args.gap, press_mm=press,
                          carry_squeeze=args.carry_squeeze,
                          turn_squeeze=args.turn_squeeze,
                          repose_iters=args.repose_iters, repose_steps=args.repose_steps,
                          descend_iters=args.descend_iters, descend_steps=args.descend_steps,
                          stand_order=args.stand_order, airgrip=args.airgrip,
                          seat_aim=args.seat_aim, angle_gain=args.angle_gain,
                          angle_rate=args.angle_rate, angle_from=args.angle_from,
                          regrip_ref=args.regrip_ref, press_regrip=args.press_regrip / 1000.0,
                          reindex=args.reindex, relay_gait=args.relay_gait,
                          clear=args.clear,
                          track_frac=args.track_frac, relay_squeeze=args.relay_squeeze,
                          tilt_deg=args.tilt_deg, tilt_dir=args.tilt_dir,
                          screw_torque=args.screw_torque,
                          centre_x=args.centre_x,
                          grip_depth=args.grip_depth, ring_z=args.ring_z,
                          stroke_deg=args.stroke, cycles=args.cycles, squeeze=args.squeeze,
                          release_mm=args.release, twist_steps=args.twist_steps,
                          move_steps=args.move_steps, pad_radius=args.pad_radius,
                          no_floor_gait=args.no_floor_gait, arm_ik=arm_ik,
                          scene_path=scene_path,
                          place_xy=screw.get('socket_xy'), place_err=pe,
                          seat_z=screw.get('seat_z'), tip_len=screw.get('tip_len', 0.0),
                          transport_steps=args.transport_steps,
                          transport_iters=args.transport_iters,
                          jitter=args.jitter if rep or args.repeats > 1 else 0.0,
                          seed=rep, video=args.video if rep == 0 else None,
                          film=args.film if rep == 0 else None, cam=cam, trace=args.trace,
                          video_size=tuple(int(v) for v in args.video_size.split(",")),
                          cam_look=(None if args.cam_look is None else
                                    tuple(float(v) for v in args.cam_look.split(","))))
                r["rep"] = rep
                rows.append(r)
                print(f"{r['run']:22} {r['reindex']:5} {press:6.1f} {r['ring_ik_grip_mm']:7.2f} "
                      f"{r['repose_deg']:7.2f} {r['turns']:6.2f} {r['gain_mean_deg']:7.2f} "
                      f"{r['final_tilt_deg']:6.2f} {r['drift_mm']:7.2f} "
                      f"{(r['free_frac'] if r['free_frac'] is not None else 0):5.2f} "
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
