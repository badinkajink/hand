"""The reorient JUDGE: a fixed, scripted reorientation run on whatever grasp a policy delivers.

A grasp is good iff this primitive can stand the cylinder up from it without dropping it. The
judge is deliberately not learned -- a trained reorienter is seed-dominated (hand repo CLAUDE.md
lesson 7), and a judge that changes run to run cannot rank grasps.

Algorithm (identical for every grasp; only its inputs -- the achieved contacts -- differ):
  1. ANCHORS. At handoff, each pad-face site's position and normal are expressed in the object
     frame: the material points the pads are holding.
  2. PATH. Rotate the object about the world x axis (the pinch axis) through the anchors'
     centroid raised by PIVOT_H, in N_WAY steps up to THETA_TARGET. PIVOT_H > 0 makes the shaft
     slide down through the grip as it stands, which is what lets the descending finger avoid
     extending (real_v1 MECHANISM.md). Both directions are solved; the one with the smaller IK
     residual is used.
  3. IK. For every waypoint the pads track their rotated anchors, pushed SQUEEZE into the
     object along their normals so the servos keep gripping. Palm fixed.
  4. EXECUTE. Joint targets are smoothly interpolated through the waypoints over T_TURN, then
     held for T_SETTLE. The physics decides what actually happens.

Verdict per episode:
    held      the object never touched the table during the judge and kept >= 2 finger contacts
              through the final settle
    tilt      the cylinder axis' final elevation above horizontal, deg (90 = vertical)
    success   held and tilt >= TILT_PASS
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import kin, scene

THETA_TARGET = np.radians(90.0)
N_WAY = 9
PIVOT_H = 0.008
PALM_FREE = True         # the UR5 may move the palm in z during the turn (2026-10-29)
SQUEEZE = 0.004
T_TURN = 2.0
T_SETTLE = 0.6
DROP_Z = 0.04            # object centre height (m) below which it has been dropped
TILT_PASS = 45.0         # deg
# Reachability cap (2026-10-29): the turn stops at the last waypoint whose IK miss is <= RES_MAX.
# Without it the 90-degree path asked for pad positions 13-16 mm out of reach from every table
# grasp, the pads drifted off mid-turn, and every grasp failed the same way.
RES_MAX = 0.003          # m

# MODE (2026-10-30): "bench" = the D1 bench turn, the one shipped to and run on the real hand
# (docs/experiments/20260831-real_v1-sobol8192/deploy/sv1_w6689_b050_poses.txt): fixed joint deltas
# added to whatever grasp the policy delivers, ramped over T_BENCH, palm held. "track" = the
# pad-tracking IK turn above (not trackable by 3-joint fingers past ~10 deg, see 2026-10-29 notes).
MODE = "bench"
#                       thumb yaw/mcp/pip       index yaw/mcp/pip        middle yaw/mcp/pip
BENCH_DQ = np.radians([24.01 - 32.50, 39.45 - 39.58, 1.88 - 0.00,
                       -34.42 + 23.15, 24.48 - 32.25, 0.00 - 18.87,
                       28.04 + 0.61, 28.96 - 24.92, 69.32 - 40.68])
T_BENCH = 1.1


@dataclass
class Verdict:
    held: bool
    tilt_deg: float
    max_tilt_deg: float
    min_z: float
    min_contacts: int
    ik_residual_mm: float
    direction: int
    reach_deg: float = 0.0     # how far the capped turn was planned

    @property
    def success(self) -> bool:
        return self.held and self.tilt_deg >= TILT_PASS


def _quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def object_axis(d, ix) -> np.ndarray:
    """World direction of the cylinder's axis (its local z)."""
    return _quat2mat(d.qpos[ix["obj_qpos"] + 3:ix["obj_qpos"] + 7])[:, 2]


def tilt_deg(d, ix) -> float:
    return float(np.degrees(np.arcsin(np.clip(abs(object_axis(d, ix)[2]), 0, 1))))


def finger_contacts(m, d, ix) -> int:
    og = ix["obj_geom"]
    hit = set()
    for c in d.contact[:d.ncon]:
        if og not in (c.geom1, c.geom2):
            continue
        b = m.geom_bodyid[c.geom2 if c.geom1 == og else c.geom1]
        for f, bs in ix["finger_bodies"].items():
            if b in bs:
                hit.add(f)
    return len(hit)


def plan_reorient(K: kin.Kinematics, d, ix):
    """Joint waypoints (N_WAY+1, 9) for the scripted turn from the current held state."""
    K.d.qpos[:] = d.qpos
    mujoco.mj_kinematics(K.m, K.d)
    obj_p = d.qpos[ix["obj_qpos"]:ix["obj_qpos"] + 3].copy()
    obj_R = _quat2mat(d.qpos[ix["obj_qpos"] + 3:ix["obj_qpos"] + 7])
    pads = d.site_xpos[ix["pad_site"]].copy()
    nrm = d.site_xmat[ix["pad_site"]].reshape(-1, 3, 3)[:, :, 2].copy()
    loc_p = (pads - obj_p) @ obj_R                      # anchors in the object frame
    loc_n = nrm @ obj_R
    pivot = pads.mean(0) + np.array([0, 0, PIVOT_H])
    q0 = d.qpos[ix["joint_qpos"]].copy()
    palm = float(d.qpos[ix["palm_qpos"]])
    best = None
    for sgn in (+1, -1):
        # start from the COMMANDED targets: a position servo grips through its tracking error,
        # so starting from the measured angles would release the squeeze before the turn
        qs, res, q = [d.ctrl[ix["joint_act"]].copy()], [], q0
        pz, pk = [float(d.ctrl[ix["palm_act"]])], palm
        for k in range(1, N_WAY + 1):
            th = sgn * THETA_TARGET * k / N_WAY
            Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
            p_k = Rx @ (obj_p - pivot) + pivot
            R_k = Rx @ obj_R
            tgt = loc_p @ R_k.T + p_k
            n_k = loc_n @ R_k.T
            sol = K.track_ik(tgt + n_k * SQUEEZE, n_k, q, pk, palm_free=PALM_FREE)
            if sol.residual > RES_MAX:
                break
            q, pk = sol.q, sol.palm_z
            qs.append(q)
            pz.append(pz[0] + (pk - palm))
            res.append(sol.residual)
        # prefer the direction that reaches further, then the smaller miss
        key = (-len(res), float(np.mean(res)) if res else 0.0)
        if best is None or key < best[3]:
            best = (np.array(qs), key[1], sgn, key, np.array(pz))
    qs, score, sgn, _, pz = best
    if len(qs) == 1:                                  # nothing reachable: hold still
        qs, pz = np.vstack([qs, qs]), np.r_[pz, pz]
    return qs, score, sgn, pz


def _bench_plan(m, d, ix):
    """Waypoints for the bench turn: commanded grasp -> grasp + BENCH_DQ (clipped to joint range)."""
    q0 = d.ctrl[ix["joint_act"]].copy()
    lo, hi = ix["jnt_range"][:, 0], ix["jnt_range"][:, 1]
    q1 = np.clip(q0 + BENCH_DQ, lo, hi)
    pz = float(d.ctrl[ix["palm_act"]])
    return np.vstack([q0, q1]), 0.0, 0, np.array([pz, pz])


def run(m, d, ix, K: kin.Kinematics, dt: float = 0.02, frames: list | None = None,
        renderer=None, cam=None) -> Verdict:
    """Execute the judge from the current (held) state of `d`, in place."""
    if MODE == "bench":
        qs, res, sgn, pz = _bench_plan(m, d, ix)
    else:
        qs, res, sgn, pz = plan_reorient(K, d, ix)
    t_turn = T_BENCH if MODE == "bench" else T_TURN
    nsub = int(round(dt / m.opt.timestep))
    n_turn, n_settle = int(t_turn / dt), int(T_SETTLE / dt)
    min_z, min_c, max_tilt, floor_hit = np.inf, 9, 0.0, False
    for k in range(n_turn + n_settle):
        s = min(1.0, k / n_turn) * (len(qs) - 1)
        i = min(int(s), len(qs) - 2)
        a = s - i
        a = a * a * (3 - 2 * a)
        d.ctrl[ix["joint_act"]] = qs[i] + (qs[i + 1] - qs[i]) * a
        d.ctrl[ix["palm_act"]] = pz[i] + (pz[i + 1] - pz[i]) * a
        for _ in range(nsub):
            mujoco.mj_step(m, d)
        z = float(d.qpos[ix["obj_qpos"] + 2])
        min_z = min(min_z, z)
        if k >= n_turn:                      # grip judged over the final settle only
            min_c = min(min_c, finger_contacts(m, d, ix))
        if not floor_hit and k > 5:
            og, fg = ix["obj_geom"], ix["floor_geom"]
            floor_hit = any({c.geom1, c.geom2} == {og, fg} for c in d.contact[:d.ncon])
        max_tilt = max(max_tilt, tilt_deg(d, ix))
        if frames is not None and renderer is not None and k % 2 == 0:
            renderer.update_scene(d, cam)
            frames.append(renderer.render())
    # 2026-10-30: was `min_z > DROP_Z`, which failed grasps that stood the cylinder up (its centre
    # drops ~50 mm as it swings below the grip) without it ever touching the table.
    held = not floor_hit and min_c >= 2
    reach = (float("nan") if MODE == "bench" else
             float(np.degrees(THETA_TARGET) * (len(qs) - 1) / N_WAY) if not np.allclose(qs[0], qs[-1]) else 0.0)
    return Verdict(held, tilt_deg(d, ix), max_tilt, min_z, int(min_c), res * 1000, sgn, reach)
