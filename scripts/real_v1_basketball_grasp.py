"""Fit and probe a basketball grasp on a `real_v1` mount layout.

THE GRASP. A size-7 basketball is 240 mm across on a hand whose thumb and pair mounts sit at
most 160 mm apart and whose fingers reach 78.66 mm. No pinch exists: all three pads land on
the ball's upper cap and the grasp is a FRICTION CAP GRASP from above. With a pad at polar
angle theta from the ball's top pole, the pad's normal force N pushes the ball down and out by
N cos(theta) and its friction can pull it up by at most mu N sin(theta), so a pad only carries
weight once

    tan(theta) > 1 / mu        (mu 1.0 -> theta > 45 deg;  mu 0.6 -> theta > 59 deg)

and the three pads together must clear the ball's weight with what is left over. Wrapping
further round the ball (larger theta) helps until the fingers run out of reach or the ball's
top runs into the plate, so the fit scans theta and the palm height and reports what is
reachable; the probe then closes, lifts and holds, under the shipped actuator and under the
measured one (kp 0.5, 0.35 N m -- docs/experiments/20260902-servo-sysid/drop_gate.json), over
a friction sweep. Geometric reach and a held lift are reported separately because they are
different questions.

MOUNT LAYOUTS. The hand is the CAD hand (assets/mjcf/real_v1/real_hand.xml); a design is six
gantry coordinates inside REAL_V1_WORKSPACE and is baked with the shared generator, so every
scene here is one the machine can be set to. `wide` parks all three gantries at the far ends
of their travel (thumb -80, pair +80/+-85 mm), the widest tripod the hand offers.

    MUJOCO_GL=egl uv run python scripts/real_v1_basketball_grasp.py fit --design wide
    MUJOCO_GL=egl uv run python scripts/real_v1_basketball_grasp.py sweep --out <dir>
    MUJOCO_GL=egl uv run python scripts/real_v1_basketball_grasp.py film --design wide --mu 1.0

`fit --write` writes `open_ik` (the grasp, pads at pad-contact distance) and `open` (the same
directions backed off by --gap) into the generated scene, which is the CEM seed / RL reset
layout every other real_v1 scene uses.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from morphohand.sampling.morphology import (  # noqa: E402
    REAL_V1_MOUNTS,
    REAL_V1_WORKSPACE,
    MorphologyValues,
    clip_morphology,
)
from morphohand.tools.keyframe_ik import (  # noqa: E402
    FINGERS,
    TIPS,
    actuator_ctrl_from_qpos,
    ik_finger,
    inject_keyframe,
)
from morphohand.tools.morphology_xml import create_rigid_hand_and_scene_xmls  # noqa: E402

HAND = ROOT / "assets/mjcf/real_v1/real_hand.xml"
SCENE = ROOT / "assets/mjcf/real_v1/scenes/scene_basketball.xml"
BALL = "basketball"
PAD_RADIUS = 0.010550
LIMIT_MARGIN = 0.05           # rad off a joint stop, as in fit_real_v1_pose
FINGER_NAMES = ("thumb", "index", "middle")

# Official sizes: circumference -> radius, and the mid-range mass. Size 7 is the men's ball.
BALL_SIZES = {
    "7": dict(radius=0.120, mass=0.620),
    "6": dict(radius=0.116, mass=0.540),
    "5": dict(radius=0.110, mass=0.480),
    "3": dict(radius=0.089, mass=0.300),
}

# Gantry coordinates in mm (palm frame), the readable form; offsets from REAL_V1_MOUNTS are
# what the generator takes. g12 is the bench's best-retaining deployed hand (D4).
DESIGNS = {
    "wide": dict(thumb=(-80.0, 0.0), index=(80.0, 85.0), middle=(80.0, -85.0)),
    "nominal": dict(thumb=(-50.0, 0.0), index=(50.0, 55.0), middle=(50.0, -55.0)),
    "g12": dict(thumb=(-42.5, 0.0), index=(42.5, 40.0), middle=(42.5, -40.0)),
}

PLANTS = {
    "shipped": dict(kp=30.0, kv=0.5, forcerange=10.0, frictionloss=0.0),
    "measured": dict(kp=0.5, kv=0.6, forcerange=0.35, frictionloss=0.0035),
}

# Contact models. `default` is what every scene in this repo runs (pyramidal cone, impratio 1).
# `grasp` is the setting the MuJoCo documentation gives for grasping: elliptic cone with the
# frictional impedance ten times the normal one. On a cap grasp the difference is the whole
# result -- with `default` the friction constraint is as soft as the normal one, so the pads
# lose their 0.5-1 mm of penetration before the friction has built the force it needs and the
# ball never leaves the floor, even at 50 N per pad and mu 2.4. A pinch does not care because
# its normals oppose each other; a cap grasp has to carry the normals' downward component on
# friction alone.
CONTACT = {
    "default": dict(cone=0, impratio=1.0),
    "grasp": dict(cone=1, impratio=10.0),
}


# --- design -> scene ----------------------------------------------------------------------

def design_values(mounts_mm: dict[str, tuple[float, float]]) -> MorphologyValues:
    off = {f: ((mounts_mm[f][0] * 1e-3) - REAL_V1_MOUNTS[f][0],
               (mounts_mm[f][1] * 1e-3) - REAL_V1_MOUNTS[f][1]) for f in FINGER_NAMES}
    mv = MorphologyValues(
        thumb_x=off["thumb"][0], thumb_y=off["thumb"][1], thumb_len=0.0,
        index_x=off["index"][0], index_y=off["index"][1], index_len=0.0,
        middle_x=off["middle"][0], middle_y=off["middle"][1], middle_len=0.0,
    )
    clipped = clip_morphology(mv, REAL_V1_WORKSPACE)
    for k in ("thumb_x", "thumb_y", "index_x", "index_y", "middle_x", "middle_y"):
        if abs(getattr(clipped, k) - getattr(mv, k)) > 1e-9:
            raise ValueError(f"{k}={getattr(mv, k)*1e3:.1f} mm is outside REAL_V1_WORKSPACE")
    return mv


def make_scene(design: str, mounts_mm: dict, radius: float, mass: float, out_dir: Path) -> Path:
    """Bake the layout through the shared generator, then size the ball."""
    # One file per layout AND ball: the generator names by morphology alone, and a second
    # size would overwrite the first's file under the same name.
    tag = f"{design}_r{radius*1e3:.0f}"
    _, scene = create_rigid_hand_and_scene_xmls(HAND, SCENE, design_values(mounts_mm), out_dir,
                                                 hand_prefix=f"hand_{tag}",
                                                 scene_prefix=f"scene_{tag}")
    text = scene.read_text()
    text = re.sub(rf'(<body name="{BALL}" pos=")[^"]*(")', rf'\g<1>0 0 {radius:.4f}\g<2>', text)
    text = re.sub(r'(<geom type="sphere" size=")[^"]*(" mass=")[^"]*(")',
                  rf'\g<1>{radius:.4f}\g<2>{mass:.3f}\g<3>', text)
    scene.write_text(text)
    m = mujoco.MjModel.from_xml_path(str(scene))
    gid = _ball_geom(m)
    assert abs(m.geom_size[gid, 0] - radius) < 1e-6, "ball radius did not take"
    return scene


def _ball_geom(m: mujoco.MjModel) -> int:
    bid = m.body(BALL).id
    return next(g for g in range(m.ngeom) if m.geom_bodyid[g] == bid)


# --- geometry -------------------------------------------------------------------------------

def _mounts_world(m, d) -> dict[str, np.ndarray]:
    return {f: d.body(f"{f}_mount").xpos.copy() for f in FINGER_NAMES}


def _set_palm(m, d, px, py, pz):
    for name, val in (("palm_px", px), ("palm_py", py), ("palm_pz", pz)):
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = val


def centre_palm(m, d, ball_pos: np.ndarray) -> tuple[float, float]:
    """Palm x/y at which the three mounts are equidistant from the ball's axis.

    A cap grasp wants the pads at the same polar angle, and each finger reaches its pad from
    its own mount, so the mounts should ring the axis at one radius. Solved on the radial
    distances directly (least squares over a grid, then refined) so an asymmetric deployed
    layout gets the same treatment as the symmetric ones.
    """
    ball_xy = ball_pos[:2]
    d.qpos[:] = 0
    d.qpos[:7] = [ball_pos[0], ball_pos[1], ball_pos[2], 1, 0, 0, 0]
    _set_palm(m, d, 0.0, 0.0, 0.0)
    mujoco.mj_forward(m, d)
    mounts = {f: (d.body(f"{f}_mount").xpos - d.body("palm_pose").xpos)[:2] for f in FINGER_NAMES}

    def spread(px, py):
        r = [np.linalg.norm(mounts[f] + np.array([px, py]) - ball_xy) for f in FINGER_NAMES]
        return float(np.var(r))

    best = (spread(0.0, 0.0), 0.0, 0.0)
    for step in (0.010, 0.002, 0.0004):
        cx, cy = best[1], best[2]
        for px in np.arange(cx - 5 * step, cx + 5 * step + 1e-9, step):
            for py in np.arange(cy - 5 * step, cy + 5 * step + 1e-9, step):
                v = spread(px, py)
                if v < best[0]:
                    best = (v, float(px), float(py))
    return best[1], best[2]


def squeeze_margin(q: list[float], rng: list) -> float:
    """Distance (rad) to the nearest stop that would block the SQUEEZE.

    The pads press the ball inward, which is positive MCP/PIP flexion, so only the flexion
    stop matters for those two: a finger parked on its -15 deg extension stop has all of its
    inward authority, and outboard targets can only be reached by leaning on that stop. Yaw
    is checked on both sides."""
    yaw, mcp, pip = q
    (ylo, yhi), (_, mhi), (_, phi) = rng
    return float(min(yaw - ylo, yhi - yaw, mhi - mcp, phi - pip))


JOINT_SPACING = 0.020750
PAD_CENTRE = 0.026610
CAP_YAW, CAP_MID, CAP_DIST = 0.022900, 0.022900, 0.016060
FLEX_SIGN = {"thumb": -1.0, "index": 1.0, "middle": 1.0}
YAW_RANGE = (-1.483530, 1.483530)
MCP_RANGE = (-0.261799, 1.605703)
PIP_RANGE = (-0.314159, 1.605703)
PLATE_UNDERSIDE = 0.025 - 0.0015          # plate underside above the mounting plane


def _rx(a):
    c, s_ = np.cos(a), np.sin(a)
    out = np.zeros(a.shape + (3, 3)); out[..., 0, 0] = 1
    out[..., 1, 1] = c; out[..., 1, 2] = -s_; out[..., 2, 1] = s_; out[..., 2, 2] = c
    return out


def _ry(a):
    c, s_ = np.cos(a), np.sin(a)
    out = np.zeros(a.shape + (3, 3)); out[..., 1, 1] = 1
    out[..., 0, 0] = c; out[..., 0, 2] = s_; out[..., 2, 0] = -s_; out[..., 2, 2] = c
    return out


def finger_fk(finger: str, yaw, mcp, pip):
    """Pad centre and the three link segments (mount frame) for arrays of joint angles.

    Same chain as the MJCF: yaw about +x at the mount, MCP and PIP about (0, s, 0) with s = -1
    on the thumb, joints 20.75 mm apart along -z, pad centre 26.61 mm below the PIP. Checked
    against mj_forward to 1e-9 in tests/test_basketball_reach.py.
    """
    s_ = FLEX_SIGN[finger]
    Ry_pip, Ry_mcp, Rx = _ry(s_ * pip), _ry(s_ * mcp), _rx(yaw)
    down = np.array([0.0, 0.0, -1.0])
    pip_in_mcp = np.array([0, 0, -JOINT_SPACING])
    mcp_in_yaw = np.array([0, 0, -JOINT_SPACING])
    # points in the MCP frame
    pad_mcp = pip_in_mcp + (Ry_pip @ (PAD_CENTRE * down))
    dist_end_mcp = pip_in_mcp + (Ry_pip @ (CAP_DIST * down))
    mid_end_mcp = CAP_MID * down
    # into the yaw frame
    def to_yaw(v):
        return mcp_in_yaw + np.einsum("...ij,...j->...i", Ry_mcp, np.broadcast_to(v, Ry_mcp.shape[:-1]))
    pad = np.einsum("...ij,...j->...i", Rx, to_yaw(pad_mcp))
    pip_j = np.einsum("...ij,...j->...i", Rx, to_yaw(np.broadcast_to(pip_in_mcp, Ry_pip.shape[:-1])))
    dist_end = np.einsum("...ij,...j->...i", Rx, to_yaw(dist_end_mcp))
    mcp_j = np.einsum("...ij,...j->...i", Rx, np.broadcast_to(mcp_in_yaw, Rx.shape[:-1]))
    mid_end = np.einsum("...ij,...j->...i", Rx, to_yaw(np.broadcast_to(mid_end_mcp, Ry_pip.shape[:-1])))
    yaw_end = np.einsum("...ij,...j->...i", Rx, np.broadcast_to(CAP_YAW * down, Rx.shape[:-1]))
    zero = np.zeros_like(pad)
    return pad, ((zero, yaw_end), (mcp_j, mid_end), (pip_j, dist_end))


def _seg_dist(c, a, b):
    """Distance from point c to segments a->b (broadcast over leading dims)."""
    ab = b - a
    t = np.clip(np.einsum("...i,...i->...", c - a, ab) / np.maximum(np.einsum("...i,...i->...", ab, ab), 1e-12), 0, 1)
    return np.linalg.norm(c - (a + t[..., None] * ab), axis=-1)


def reach_map(mounts: dict[str, np.ndarray], centre: np.ndarray, R: float,
              touch_tol: float = 0.0015, link_clear: float = -0.0005,
              n_yaw: int = 69, n_mcp: int = 55, n_pip: int = 38) -> dict[str, dict]:
    """Every joint-grid pose per finger whose pad touches the ball with its links clear.

    Returns, per finger, arrays of theta (polar angle from the top pole), azimuth about the
    ball's vertical axis, and the joint angles. The grid is the finger's full ROM at 2.5 deg /
    2 deg / 3 deg, so a pose the map does not contain is one the finger cannot make to
    within a couple of degrees.
    """
    yaw = np.linspace(*YAW_RANGE, n_yaw)
    mcp = np.linspace(*MCP_RANGE, n_mcp)
    pip = np.linspace(*PIP_RANGE, n_pip)
    Y, M, P = np.meshgrid(yaw, mcp, pip, indexing="ij")
    out = {}
    for f in FINGER_NAMES:
        pad, segs = finger_fk(f, Y, M, P)
        pad_w = pad + mounts[f]
        rel = pad_w - centre
        dist = np.linalg.norm(rel, axis=-1)
        touching = np.abs(dist - (R + PAD_RADIUS)) < touch_tol
        clear = np.ones_like(touching)
        for a, b in segs:
            clear &= _seg_dist(centre, a + mounts[f], b + mounts[f]) - (R + PAD_RADIUS) > link_clear
        marg = np.minimum.reduce([Y - YAW_RANGE[0], YAW_RANGE[1] - Y, MCP_RANGE[1] - M, PIP_RANGE[1] - P])
        ok = touching & clear & (marg > LIMIT_MARGIN)
        theta = np.degrees(np.arccos(np.clip(rel[..., 2] / np.maximum(dist, 1e-9), -1, 1)))
        az = np.degrees(np.arctan2(rel[..., 1], rel[..., 0]))
        out[f] = {"theta": theta[ok], "az": az[ok], "yaw": Y[ok], "mcp": M[ok], "pip": P[ok]}
    return out


def _sep(a, b):
    d = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return d


def choose_contacts(rm: dict, theta: float, theta_tol: float = 2.0, az_bin: float = 5.0,
                    thumb_az: float = 180.0, thumb_tol: float = 7.5):
    """One pose per finger at polar angle theta: thumb on its own meridian, pair mirrored.

    Under position control each pad's normal force is its own position error times kp, so
    with one squeeze command the three forces come out about equal in magnitude, and equal
    forces balance only when the pair sits symmetric about the thumb at +-60 deg from the
    ball's +x axis. The thumb is held within `thumb_tol` of `thumb_az` (its yaw near zero),
    index and middle are taken as a mirror pair, and the pair azimuth nearest +-60 deg wins.
    A thumb leaning 40 deg off its meridian was tried first: its pad ended up pushing the ball
    sideways into the pair and the ball rolled out from under it. Returns
    (min separation deg, {finger: (yaw, mcp, pip, az, theta)}) or None.
    """
    sel_t = (np.abs(rm["thumb"]["theta"] - theta) < theta_tol) & \
            (_sep(rm["thumb"]["az"], thumb_az) < thumb_tol)
    if not sel_t.any():
        return None
    it = np.flatnonzero(sel_t)
    i_t = it[np.argmin(_sep(rm["thumb"]["az"][sel_t], thumb_az) + np.abs(rm["thumb"]["theta"][sel_t] - theta))]
    best = None
    for f, sign in (("index", 1.0), ("middle", -1.0)):
        sel = np.abs(rm[f]["theta"] - theta) < theta_tol
        if not sel.any():
            return None
        idx = np.flatnonzero(sel)
        az = rm[f]["az"][sel] * sign          # fold the middle finger onto the index's side
        ok = (az > 30.0) & (az < 110.0)
        if not ok.any():
            return None
        cost = np.abs(az - 60.0) + 0.5 * np.abs(rm[f]["theta"][sel] - theta)
        cost[~ok] = np.inf
        bins = np.round(az / az_bin) * az_bin
        pick = {}
        for i in np.argsort(cost):
            if not np.isfinite(cost[i]):
                break
            pick.setdefault(float(bins[i]), idx[i])
        if best is None:
            best = pick
        else:
            common = sorted(set(best) & set(pick), key=lambda b: abs(b - 60.0))
            if not common:
                return None
            b = common[0]
            i_i, i_m = best[b], pick[b]
    pose = {"thumb": tuple(float(rm["thumb"][k][i_t]) for k in ("yaw", "mcp", "pip", "az", "theta")),
            "index": tuple(float(rm["index"][k][i_i]) for k in ("yaw", "mcp", "pip", "az", "theta")),
            "middle": tuple(float(rm["middle"][k][i_m]) for k in ("yaw", "mcp", "pip", "az", "theta"))}
    azs = [pose[f][3] for f in FINGER_NAMES]
    sep = min(_sep(azs[0], azs[1]), _sep(azs[0], azs[2]), _sep(azs[1], azs[2]))
    return float(sep), pose


def _choose_contacts_free(rm: dict, theta: float, theta_tol: float = 2.0, az_bin: float = 5.0):
    """The unconstrained selection (widest minimum separation); kept for the reach report."""
    cand = {}
    for f in FINGER_NAMES:
        sel = np.abs(rm[f]["theta"] - theta) < theta_tol
        if not sel.any():
            return None
        az = rm[f]["az"][sel]
        bins = np.round(az / az_bin) * az_bin
        keep = {}
        for i in np.argsort(np.abs(rm[f]["theta"][sel] - theta)):
            b = float(bins[i])
            if b not in keep:
                keep[b] = i
        idx = np.flatnonzero(sel)
        cand[f] = [(b, idx[i]) for b, i in keep.items()]
    best = None
    A = np.array([b for b, _ in cand["thumb"]]); I = np.array([b for b, _ in cand["index"]])
    Mm = np.array([b for b, _ in cand["middle"]])
    sep = np.minimum(np.minimum(_sep(A[:, None, None], I[None, :, None]),
                                _sep(A[:, None, None], Mm[None, None, :])),
                     _sep(I[None, :, None], Mm[None, None, :]))
    k = np.unravel_index(np.argmax(sep), sep.shape)
    best = float(sep[k])
    pose = {}
    for f, kk in zip(FINGER_NAMES, k):
        _, i = cand[f][kk]
        pose[f] = tuple(float(rm[f][key][i]) for key in ("yaw", "mcp", "pip", "az", "theta"))
    return best, pose


def fit(scene: Path, theta_deg: float, squeeze: float, gap: float,
        pz_lo: float = -0.15, pz_hi: float = 0.10, pz_step: float = 0.0025,
        min_sep: float = 90.0, seated_only: bool = False) -> dict | None:
    """Palm pose + finger angles for the cap grasp at one theta.

    The mounting plane is scanned from high to low; at each height the reach map says which
    (theta, azimuth) contacts each finger can make with its links clear of the ball, and
    `choose_contacts` picks the three with the widest azimuth spread. A height is feasible
    when that spread is at least `min_sep`. The LOWEST feasible height wins because it puts
    the ball's top nearest the plate; the SEATED height (ball touching the plate's underside)
    is tried explicitly and reported as such, since a ball pressed into the plate cannot sag
    and a floating one can. The grip pose (pads `squeeze` inside the surface, the ctrl
    target) and the open pose (`gap` outside) are IK'd from the chosen touch pose.
    """
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    R = float(m.geom_size[_ball_geom(m), 0])
    centre = np.array([0.0, 0.0, R])
    px, py = centre_palm(m, d, centre)
    d.qpos[:] = 0
    d.qpos[:7] = [0, 0, R, 1, 0, 0, 0]
    _set_palm(m, d, px, py, 0.0)
    mujoco.mj_forward(m, d)
    palm0 = d.body("palm_pose").xpos.copy()
    # world x/y of each mount with the palm centred over the ball; z relative to the mounting
    # plane so the height scan can add the plane's height back in
    mounts0 = {f: d.body(f"{f}_mount").xpos - np.array([0.0, 0.0, palm0[2]]) for f in FINGER_NAMES}
    ring = float(np.mean([np.linalg.norm(mounts0[f][:2] - centre[:2]) for f in FINGER_NAMES]))
    base_pz = float(palm0[2])
    seated_pz = (centre[2] + R) - PLATE_UNDERSIDE - base_pz

    heights = list(np.arange(pz_hi, pz_lo - 1e-9, -pz_step))
    if seated_only:
        heights = [seated_pz]
    elif pz_lo <= seated_pz <= pz_hi:
        heights.append(seated_pz)
    feasible = None
    scan = {}
    for pz in heights:
        if pz < seated_pz - 1e-6:
            scan[round(float(pz), 4)] = "plate"
            continue
        mounts = {f: mounts0[f] + np.array([0, 0, base_pz + pz]) for f in FINGER_NAMES}
        rm = reach_map(mounts, centre, R)
        ch = choose_contacts(rm, theta_deg)
        if ch is None:
            scan[round(float(pz), 4)] = "reach"
            continue
        sep, pose = ch
        scan[round(float(pz), 4)] = f"sep {sep:.0f}"
        if sep >= min_sep and (feasible is None or pz < feasible[0] - 1e-9):
            feasible = (float(pz), sep, pose)
    if feasible is None:
        return None
    pz, sep, pose = feasible

    d.qpos[:] = 0
    d.qpos[:7] = [0, 0, R, 1, 0, 0, 0]
    _set_palm(m, d, px, py, pz)
    for f, joints in FINGERS.items():
        for j, v in zip(joints, pose[f][:3]):
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
    mujoco.mj_forward(m, d)
    q_touch = d.qpos.copy()
    tips = {f: d.body(TIPS[f]).xpos.copy() for f in FINGER_NAMES}
    cl = clearance(m, d)
    normals = {f: (tips[f] - centre) / np.linalg.norm(tips[f] - centre) for f in FINGER_NAMES}
    mount_z = float(d.body("palm_pose").xpos[2])
    plate_gap = float((mount_z + PLATE_UNDERSIDE) - (centre[2] + R))
    # grip: each pad driven `squeeze` closer to the ball's centre along whatever joint motion
    # reduces that distance; open: `gap` further away. Not a point along the normal -- see
    # depth_pose.
    res_g, marg = depth_pose(m, d, q_touch, centre, R + PAD_RADIUS - squeeze)
    q_grip = d.qpos.copy()
    res_o, _ = depth_pose(m, d, q_touch, centre, R + PAD_RADIUS + gap)
    q_open = d.qpos.copy()
    grip_ctrl = actuator_ctrl_from_qpos(m, _data_at(m, q_grip))
    open_ctrl = actuator_ctrl_from_qpos(m, _data_at(m, q_open))
    return {
        "theta_deg": theta_deg, "radius": R, "squeeze": squeeze, "gap": gap,
        "palm": [px, py, pz], "mount_plane_z": mount_z, "ball_top_z": float(centre[2] + R),
        "plate_gap_mm": plate_gap * 1e3, "seated": bool(abs(plate_gap) < 1e-4),
        "ring_radius_mm": ring * 1e3, "min_sep_deg": sep,
        "contact_az_deg": {f: pose[f][3] for f in FINGER_NAMES},
        "contact_theta_deg": {f: pose[f][4] for f in FINGER_NAMES},
        "joints_deg": {f: [float(np.degrees(v)) for v in pose[f][:3]] for f in FINGER_NAMES},
        "grip_residual_mm": {f: r * 1e3 for f, r in res_g.items()},
        "open_residual_mm": {f: r * 1e3 for f, r in res_o.items()},
        "margin_rad": marg,
        "tips": {f: t.tolist() for f, t in tips.items()},
        "clearance": cl,
        "q_touch": q_touch.tolist(), "q_grip": q_grip.tolist(), "q_open": q_open.tolist(),
        "grip_ctrl": list(grip_ctrl), "open_ctrl": list(open_ctrl),
        "scan": {str(k): v for k, v in scan.items()},
    }


def clearance(m, d, tol: float = 5e-4) -> dict:
    """Ball contacts at the current pose, split into pads (wanted), links (a finger lying on
    the ball instead of touching it with its pad) and the plate (the ball seated)."""
    names = [m.body(i).name for i in range(m.nbody)]
    pads, links, plate = [], [], []
    for c in d.contact[: d.ncon]:
        b1, b2 = names[m.geom_bodyid[c.geom1]], names[m.geom_bodyid[c.geom2]]
        if BALL not in (b1, b2):
            continue
        other = b2 if b1 == BALL else b1
        if other.endswith("_tip"):
            pads.append(other)
        elif other == "palm_pose":
            plate.append(float(c.dist))
        elif c.dist < -tol:
            links.append((other, float(c.dist)))
    return {"pads": pads, "links": links, "plate": plate}


def depth_pose(m, d, q_start, centre: np.ndarray, target_dist: float, step: float = 0.01,
               iters: int = 300):
    """From `q_start`, move each pad's centre to `target_dist` from the ball's centre.

    Steepest descent on the distance with the tip Jacobian, in small joint steps clipped to
    the ROM. A point target `squeeze` along the contact normal is the wrong squeeze here: at
    the wrap angles this hand reaches the normal points two-thirds DOWNWARD and the pads are
    at full extension, so no such point is reachable and the IK returns the touch pose plus a
    few hundredths of a radian -- 10-17 N of normal force that vanish after 1 mm of palm
    rise. MCP flexion alone takes the same pad 2.8 mm into the ball per 0.05 rad, which is
    what this finds. Returns (|residual| m, squeeze margin rad) per finger; leaves d.qpos."""
    d.qpos[:] = q_start
    mujoco.mj_forward(m, d)
    res, marg = {}, {}
    jacp = np.zeros((3, m.nv))
    for f, joints in FINGERS.items():
        jids = [m.joint(j).id for j in joints]
        qadr = [m.jnt_qposadr[j] for j in jids]
        dadr = [m.jnt_dofadr[j] for j in jids]
        rng = [m.jnt_range[j] for j in jids]
        bid = m.body(TIPS[f]).id
        for _ in range(iters):
            p = d.body(bid).xpos
            n = p - centre
            dist = float(np.linalg.norm(n))
            err = dist - target_dist
            if abs(err) < 2e-4:
                break
            mujoco.mj_jacBody(m, d, jacp, None, bid)
            g = jacp[:, dadr].T @ (n / dist)          # d(dist)/dq
            if np.linalg.norm(g) < 1e-9:
                break
            dq = -np.sign(err) * step * g / np.linalg.norm(g)
            for k, a in enumerate(qadr):
                d.qpos[a] = np.clip(d.qpos[a] + dq[k], rng[k][0], rng[k][1])
            mujoco.mj_forward(m, d)
        p = d.body(bid).xpos
        res[f] = abs(float(np.linalg.norm(p - centre)) - target_dist)
        q = [float(d.qpos[a]) for a in qadr]
        marg[f] = squeeze_margin(q, rng)
    mujoco.mj_forward(m, d)
    return res, marg


def solve_from(m, d, q_start, targets: dict[str, np.ndarray], iters: int = 60):
    """LOCAL IK each finger from `q_start` to `targets`; leaves the result in d.qpos.

    Small damped steps so the solution stays on the touch pose's configuration branch. A
    full-strength solve from the same start jumped the thumb from a hooked pose (PIP +53 deg)
    to a straight one (PIP -5 deg) for a 20 mm inward target; driven there, the thumb pressed
    5.8 N against the pair's 52 and 58 N and the ball rolled out sideways."""
    d.qpos[:] = q_start
    mujoco.mj_forward(m, d)
    res, marg = {}, {}
    for f, joints in FINGERS.items():
        res[f] = ik_finger(m, d, f, targets[f], iters=iters, lam=1e-2, step=0.15)
        q = [float(d.qpos[m.jnt_qposadr[m.joint(j).id]]) for j in joints]
        rng = [m.jnt_range[m.joint(j).id] for j in joints]
        marg[f] = squeeze_margin(q, rng)
    mujoco.mj_forward(m, d)
    return res, marg


def _data_at(m, qpos):
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    return d


# --- plant / friction -----------------------------------------------------------------------

def apply_plant(m: mujoco.MjModel, plant: dict) -> None:
    """The nine finger position actuators only; the six palm ones stay the arm's kp 4000."""
    for i in range(m.nu):
        name = m.actuator(i).name
        if name.startswith("a_palm"):
            continue
        m.actuator_gainprm[i, 0] = plant["kp"]
        m.actuator_biasprm[i, 1] = -plant["kp"]
        m.actuator_biasprm[i, 2] = -plant["kv"]
        m.actuator_forcerange[i] = [-plant["forcerange"], plant["forcerange"]]
    if plant["frictionloss"] > 0:
        for f, joints in FINGERS.items():
            for j in joints:
                jid = m.joint(j).id
                m.dof_frictionloss[m.jnt_dofadr[jid]] = plant["frictionloss"]


def apply_friction(m: mujoco.MjModel, mu: float) -> None:
    """One sliding coefficient for every hand<->ball pair. MuJoCo takes the max of the pair, so
    the ball AND every finger geom are set, else the pads' 2.4 wins the max."""
    bid = m.body(BALL).id
    for g in range(m.ngeom):
        bn = m.body(m.geom_bodyid[g]).name
        if m.geom_bodyid[g] == bid or bn.startswith(FINGER_NAMES) or bn == "palm_pose":
            m.geom_friction[g, 0] = mu


# --- probe ----------------------------------------------------------------------------------

def pad_forces(m, d) -> dict:
    """Per-pad normal force and friction-cone utilisation on the ball, plus the ball's
    non-pad contacts."""
    names = [m.body(i).name for i in range(m.nbody)]
    out = {f: {"N": 0.0, "util": 0.0} for f in FINGER_NAMES}
    others = {}
    w = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = names[m.geom_bodyid[c.geom1]], names[m.geom_bodyid[c.geom2]]
        if BALL not in (b1, b2):
            continue
        other = b2 if b1 == BALL else b1
        mujoco.mj_contactForce(m, d, i, w)
        n, t = abs(float(w[0])), float(np.linalg.norm(w[1:3]))
        mu = float(max(m.geom_friction[c.geom1, 0], m.geom_friction[c.geom2, 0]))
        if other.endswith("_tip"):
            f = other[:-4]
            out[f]["N"] += n
            out[f]["util"] = max(out[f]["util"], t / (mu * n) if n > 1e-6 else 0.0)
        else:
            others[other] = others.get(other, 0.0) + n
    return {"pads": out, "others": others}


def grip_command(m, fitted: dict, overdrive: float) -> np.ndarray:
    """The finger ctrl: touch pose + overdrive x (grip - touch), clipped to the ROM.

    (grip - touch) is the closing direction that takes each pad toward the ball's centre
    (`depth_pose`); overdrive 1 commands the pad `squeeze` inside the surface, and larger
    values command further along the same direction. On the measured servo (kp 0.5) the
    force is kp times the position error until the 0.35 N m cliff, so overdrive 1 is
    ~0.15 N m and ~3 N at the pad, and overdrive >= 4 is the cliff at ~5-6 N -- what a bench
    plan that commands a deep grip pose actually delivers. Curling MCP+PIP instead was tried:
    the pair's flexion plane is the palm's x-z plane, "inboard" for a pad at azimuth 60 deg
    is half y, so their pads slid across the ball and off it."""
    ctrl = np.array(fitted["grip_ctrl"])
    touch, grip = np.array(fitted["q_touch"]), np.array(fitted["q_grip"])
    for i in range(m.nu):
        if m.actuator(i).name.startswith("a_palm"):
            continue
        jid = m.actuator_trnid[i, 0]
        a = m.jnt_qposadr[jid]
        ctrl[i] = float(np.clip(touch[a] + overdrive * (grip[a] - touch[a]), *m.jnt_range[jid]))
    return ctrl


def probe(scene: Path, fitted: dict, plant: dict, mu: float, lift: float = 0.10,
          settle_s: float = 2.0, ramp_s: float = 1.5, hold_s: float = 2.0,
          contact: dict | None = None, overdrive: float = 1.0,
          frames_at: tuple[float, ...] = (), renderer=None, cam=None) -> dict:
    """Close from `open` toward the grip, lift the palm, hold. Reports whether the ball came
    with the hand and stayed, and the pad forces at the end of the hold.

    The settle is 2 s because the measured servo's time constant is kv/kp = 1.2 s; at 0.5 s
    its fingers had not reached the ball and every measured-plant row read 0 N. The lift is
    a smoothstep of `lift` over `ramp_s` (peak 6L/T^2 = 0.27 m/s^2 at the defaults), the way
    the arm moves on the bench; a linear position ramp has a velocity step at its start
    that no grasp on a 620 g ball survives."""
    m = mujoco.MjModel.from_xml_path(str(scene))
    apply_plant(m, plant)
    apply_friction(m, mu)
    c = contact or CONTACT["grasp"]
    m.opt.cone = c["cone"]
    m.opt.impratio = c["impratio"]
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    pz_a = next(k for k in range(m.nu) if m.actuator(k).name == "a_palm_pz")
    grip_ctrl = grip_command(m, fitted, overdrive)
    pz0 = float(grip_ctrl[pz_a])

    d.qpos[:] = fitted["q_open"]
    d.qvel[:] = 0
    d.ctrl[:] = grip_ctrl
    mujoco.mj_forward(m, d)
    z0 = float(d.body(BALL).xpos[2])
    rel0 = d.body(BALL).xpos - d.body("palm_pose").xpos

    n_settle, n_ramp, n_hold = int(settle_s / dt), int(ramp_s / dt), int(hold_s / dt)
    total = n_settle + n_ramp + n_hold
    frames, want = [], sorted(int(t / dt) for t in frames_at)
    trace = []
    after_settle = None
    for k in range(total):
        if k >= n_settle:
            u = min(1.0, (k - n_settle + 1) / n_ramp)
            d.ctrl[pz_a] = pz0 + lift * (3 * u * u - 2 * u ** 3)
        mujoco.mj_step(m, d)
        if k == n_settle - 1:
            after_settle = pad_forces(m, d)
        if k % 25 == 0:
            pf = pad_forces(m, d)
            trace.append({"t": k * dt, "ball_z": float(d.body(BALL).xpos[2] - z0),
                          "N": [pf["pads"][f]["N"] for f in FINGER_NAMES]})
        if want and k == want[0] and renderer is not None:
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())
            want.pop(0)
    rise = float(d.body(BALL).xpos[2] - z0)
    rel = d.body(BALL).xpos - d.body("palm_pose").xpos
    slip = float(np.linalg.norm(rel - rel0))
    pf = pad_forces(m, d)
    held = rise > lift - 0.02 and slip < 0.02
    return {
        "held": bool(held), "rise_mm": rise * 1e3, "slip_mm": slip * 1e3,
        "pads_end": pf["pads"], "others_end": pf["others"],
        "pads_settle": after_settle["pads"] if after_settle else None,
        "n_pads_end": sum(1 for f in FINGER_NAMES if pf["pads"][f]["N"] > 0.05),
        "trace": trace, "frames": frames,
    }


# --- filmstrip ------------------------------------------------------------------------------

def film(scene: Path, fitted: dict, plant: dict, mu: float, out: Path, label: str,
         width: int = 480, height: int = 360, n: int = 6, contact: dict | None = None,
         overdrive: float = 1.0) -> dict:
    from PIL import Image, ImageDraw
    m = mujoco.MjModel.from_xml_path(str(scene))
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = 135.0, -20.0, 0.75
    cam.lookat[:] = [0.0, 0.0, fitted["radius"] + 0.06]
    times = np.linspace(0.0, 5.45, n)
    with mujoco.Renderer(m, height=height, width=width) as r:
        res = probe(scene, fitted, plant, mu, contact=contact, overdrive=overdrive,
                    frames_at=tuple(times), renderer=r, cam=cam)
    sheet = Image.new("RGB", (width * n, height + 22), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for i, fr in enumerate(res["frames"]):
        sheet.paste(Image.fromarray(fr), (i * width, 22))
        dr.text((i * width + 6, 5), f"t={times[i]:.1f}s", fill=(255, 255, 0))
    verdict = "HELD" if res["held"] else "DROPPED"
    text = f"{label}   {verdict}  rise {res['rise_mm']:.0f} mm"
    dr.text((width * n - 8 - dr.textlength(text), 5), text,
            fill=(120, 255, 120) if res["held"] else (255, 120, 120))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    res.pop("frames")
    return res


# --- CLI ------------------------------------------------------------------------------------

def _design_mounts(args) -> tuple[str, dict]:
    if args.mounts:
        v = [float(x) for x in args.mounts.replace(",", " ").split()]
        return args.design, dict(thumb=(v[0], v[1]), index=(v[2], v[3]), middle=(v[4], v[5]))
    return args.design, DESIGNS[args.design]


def cmd_fit(args) -> int:
    design, mounts = _design_mounts(args)
    size = BALL_SIZES[args.size]
    scene = make_scene(design, mounts, size["radius"], size["mass"], args.out)
    print(f"scene {scene.relative_to(ROOT)}  ball r={size['radius']*1e3:.0f} mm m={size['mass']*1e3:.0f} g")
    print(f"{'theta':>6} {'mount_z':>8} {'plate_gap':>9} {'ring':>6} {'sep':>5} {'az(t/i/m) deg':>18} "
          f"{'joints thumb yaw/mcp/pip deg':>28} {'grip res mm':>12}")
    best = None
    for th in args.theta:
        f = fit(scene, th, args.squeeze, args.gap, seated_only=args.seated)
        if f is None:
            print(f"{th:6.1f}   unreachable")
            continue
        az, jt, rg = f["contact_az_deg"], f["joints_deg"]["thumb"], f["grip_residual_mm"]
        print(f"{th:6.1f} {f['mount_plane_z']*1e3:8.1f} {f['plate_gap_mm']:9.1f} {f['ring_radius_mm']:6.1f} "
              f"{f['min_sep_deg']:5.0f} {az['thumb']:6.0f}/{az['index']:5.0f}/{az['middle']:5.0f}   "
              f"{jt[0]:8.1f}/{jt[1]:6.1f}/{jt[2]:6.1f}   {max(rg.values()):6.2f}"
              f"{'  SEATED' if f['seated'] else ''}")
        if args.probe:
            for pname in args.plant:
                for mu in args.mu:
                    p = probe(scene, f, PLANTS[pname], mu, contact=CONTACT[args.contact],
                              overdrive=args.overdrive)
                    N = p["pads_end"]
                    print(f"        {pname:9} mu={mu:.1f}  {'HELD ' if p['held'] else 'DROP '} rise {p['rise_mm']:6.1f} mm "
                          f"slip {p['slip_mm']:5.1f} mm  N t/i/m {N['thumb']['N']:.2f}/{N['index']['N']:.2f}/{N['middle']['N']:.2f}"
                          f"  others {', '.join(f'{k} {v:.1f}' for k, v in p['others_end'].items()) or '-'}")
        if best is None or th > best["theta_deg"]:
            best = f
    if args.write and best is not None:
        m = mujoco.MjModel.from_xml_path(str(scene))
        inject_keyframe(scene, "open_ik", " ".join(f"{v:.6g}" for v in best["q_grip"]),
                        " ".join(f"{v:.6g}" for v in best["grip_ctrl"]))
        inject_keyframe(scene, "open", " ".join(f"{v:.6g}" for v in best["q_open"]),
                        " ".join(f"{v:.6g}" for v in best["open_ctrl"]))
        print(f"wrote open_ik (theta {best['theta_deg']:.0f}) and open into {scene.relative_to(ROOT)}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"design": design, "mounts_mm": mounts, "size": args.size,
                                         "fit": best}, indent=1))
    return 0


def cmd_sweep(args) -> int:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["design"], r["size"], r["theta_deg"], r["plant"], r["mu"], r["squeeze"],
                      r.get("contact", "grasp"), r.get("overdrive", 1.0)))
    fits_path = out / "fits.json"
    fits = json.loads(fits_path.read_text()) if fits_path.exists() else {}
    t0 = time.time()
    with rows_path.open("a") as fh:
        for design in args.designs:
            mounts = DESIGNS[design]
            for size in args.sizes:
                sz = BALL_SIZES[size]
                scene = make_scene(design, mounts, sz["radius"], sz["mass"], out / "scenes")
                for th in args.theta:
                    for sq in args.squeeze:
                        key = f"{design}|{size}|{th:g}|{sq:g}"
                        if key not in fits:
                            f = fit(scene, th, sq, args.gap)
                            fits[key] = None if f is None else {k: v for k, v in f.items() if k != "scan"}
                            fits_path.write_text(json.dumps(fits))
                        f = fits[key]
                        if f is None:
                            print(f"{design:8} size {size} theta {th:4.0f} sq {sq*1e3:.0f}  unreachable", flush=True)
                            continue
                        for cname in args.contact:
                          for pname in args.plants:
                            for od in (args.overdrive if pname == "measured" else [1.0]):
                              for mu in args.mu:
                                k = (design, size, th, pname, mu, sq, cname, od)
                                if k in done:
                                    continue
                                p = probe(scene, f, PLANTS[pname], mu, contact=CONTACT[cname], overdrive=od)
                                p.pop("frames")
                                row = {"design": design, "size": size, "radius": sz["radius"], "mass": sz["mass"],
                                       "theta_deg": th, "squeeze": sq, "plant": pname, "mu": mu,
                                       "contact": cname, "overdrive": od,
                                       "plate_gap_mm": f["plate_gap_mm"], "mount_plane_z": f["mount_plane_z"],
                                       "seated": f["seated"], "min_sep_deg": f["min_sep_deg"],
                                       "contact_az_deg": f["contact_az_deg"], "joints_deg": f["joints_deg"],
                                       **{kk: vv for kk, vv in p.items() if kk != "trace"}}
                                fh.write(json.dumps(row) + "\n")
                                fh.flush()
                                os.fsync(fh.fileno())
                                N = p["pads_end"]
                                print(f"{design:8} size {size} theta {th:4.0f} sq {sq*1e3:.0f} {cname:7} {pname:8} od {od:.0f} mu {mu:.1f}  "
                                      f"{'HELD' if p['held'] else 'DROP'} rise {p['rise_mm']:6.1f} slip {p['slip_mm']:5.1f} "
                                      f"N {N['thumb']['N']:.1f}/{N['index']['N']:.1f}/{N['middle']['N']:.1f}  "
                                      f"[{time.time()-t0:5.0f}s]", flush=True)
    return 0


def cmd_film(args) -> int:
    design, mounts = _design_mounts(args)
    size = BALL_SIZES[args.size]
    scene = make_scene(design, mounts, size["radius"], size["mass"], args.out / "scenes")
    f = fit(scene, args.theta[0], args.squeeze, args.gap)
    if f is None:
        print("unreachable")
        return 1
    for pname in args.plant:
        for mu in args.mu:
            label = f"{design} size{args.size} th{args.theta[0]:.0f} {pname} mu{mu:.1f}"
            png = args.out / f"film_{design}_s{args.size}_th{args.theta[0]:.0f}_{pname}_od{args.overdrive:.0f}_mu{mu:.1f}.png"
            res = film(scene, f, PLANTS[pname], mu, png, label, contact=CONTACT[args.contact],
                       overdrive=args.overdrive)
            print(f"{label}: {'HELD' if res['held'] else 'DROPPED'} rise {res['rise_mm']:.1f} mm -> {png}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--design", default="wide", help="name in DESIGNS, or a label for --mounts")
        p.add_argument("--mounts", default=None, help="tx ty ix iy mx my in mm (palm frame)")
        p.add_argument("--size", default="7", choices=sorted(BALL_SIZES))
        p.add_argument("--theta", type=float, nargs="+", default=[60.0])
        p.add_argument("--squeeze", type=float, default=0.006)
        p.add_argument("--gap", type=float, default=0.008)
        p.add_argument("--plant", nargs="+", default=["shipped", "measured"], choices=sorted(PLANTS))
        p.add_argument("--mu", type=float, nargs="+", default=[1.0])
        p.add_argument("--contact", default="grasp", choices=sorted(CONTACT))
        p.add_argument("--overdrive", type=float, default=1.0)
        p.add_argument("--out", type=Path, default=ROOT / "assets/mjcf/experimental/20260916-basketball")

    p = sub.add_parser("fit"); common(p)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--seated", action="store_true", help="only the height at which the ball touches the plate")
    p.add_argument("--write", action="store_true")
    p.add_argument("--json", type=Path, default=None)
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("sweep")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--designs", nargs="+", default=["wide", "nominal", "g12"])
    p.add_argument("--sizes", nargs="+", default=["7", "6", "5", "3"])
    p.add_argument("--theta", type=float, nargs="+", default=[40, 45, 50, 55, 60, 65, 70, 75])
    p.add_argument("--squeeze", type=float, nargs="+", default=[0.006])
    p.add_argument("--gap", type=float, default=0.008)
    p.add_argument("--plants", nargs="+", default=["shipped", "measured"])
    p.add_argument("--mu", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.5, 2.4])
    p.add_argument("--contact", nargs="+", default=["grasp"], choices=sorted(CONTACT))
    p.add_argument("--overdrive", type=float, nargs="+", default=[1.0, 4.0, 8.0],
                   help="measured plant only; the shipped plant runs at 1")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("film"); common(p)
    p.set_defaults(fn=cmd_film)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
