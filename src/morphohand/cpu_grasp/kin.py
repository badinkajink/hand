"""Inverse kinematics on the real_v3 + cylinder scene: the grasp keyframe and the reorient judge.

Two problems, one damped Gauss-Newton solver over (9 finger joints, palm_z):

  grasp_ik     put every pad's contact face ON the cylinder (a little inside it, so the servo
               tracking error makes the squeeze) with its normal pointing at the cylinder axis.
               Where along the shaft and how high on the circle the contact lands is free --
               that is the slack 3 joints per finger need -- with weak preferences for the
               equator (a grasp above it lets the shaft creep out, real_v1 SLIP.md) and for the
               mount's own y (no splayed fingers).
  track_ik     put each pad on an explicit world point with an explicit normal. The reorient
               judge uses it: rotate the achieved contact points with the object and follow.

Joint limits are enforced by projection after every step; `margin` reports how close the
solution sits to them. Finite-difference Jacobians on mj_kinematics: ~1 ms per solve.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import scene

Y = np.array([0.0, 1.0, 0.0])


@dataclass
class IKResult:
    q: np.ndarray            # 9 finger joints
    palm_z: float            # palm_z joint value (offset from the model's palm height)
    residual: float          # rms of the position residuals, m
    normal_err_deg: float    # worst pad-normal misalignment
    margin: float            # min distance to a joint limit, rad
    contacts: np.ndarray     # (3, 3) pad-face points, world
    normals: np.ndarray      # (3, 3) pad-face outward normals, world


class Kinematics:
    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.d = mujoco.MjData(model)
        self.ix = scene.indices(model)
        self.lo = self.ix["jnt_range"][:, 0].copy()
        self.hi = self.ix["jnt_range"][:, 1].copy()
        # every colliding geom on a finger's moving links, for the floor-clearance term
        moving = set().union(*self.ix["finger_bodies"].values())
        self.fgeoms = np.array([g for g in range(model.ngeom)
                                if model.geom_bodyid[g] in moving and model.geom_contype[g]])
        self.floor_clear = 0.001

    def finger_min_z(self) -> float:
        """Lowest point of any finger collision geom at the current FK state."""
        d, m, g = self.d, self.m, self.fgeoms
        z = d.geom_xpos[g, 2].copy()
        sph = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        z[sph] -= m.geom_size[g[sph], 0]
        for k in np.where(~sph)[0]:                                   # boxes
            R = d.geom_xmat[g[k]].reshape(3, 3)
            z[k] -= np.abs(R[2]) @ m.geom_size[g[k]]
        return float(z.min())

    def _fk(self, p):
        d, ix = self.d, self.ix
        d.qpos[ix["joint_qpos"]] = p[:9]
        d.qpos[ix["palm_qpos"]] = p[9]
        mujoco.mj_kinematics(self.m, d)
        s = ix["pad_site"]
        pos = d.site_xpos[s].copy()
        nrm = d.site_xmat[s].reshape(-1, 3, 3)[:, :, 2].copy()
        return pos, nrm

    def _solve(self, residual_fn, p0, iters=80, lam=1e-3, palm_free=True):
        p = p0.copy()
        eps = 1e-5
        for _ in range(iters):
            r = residual_fn(*self._fk(p))
            J = np.empty((len(r), 10))
            for k in range(10):
                dp = p.copy()
                dp[k] += eps
                J[:, k] = (residual_fn(*self._fk(dp)) - r) / eps
            if not palm_free:
                J[:, 9] = 0
            step = np.linalg.solve(J.T @ J + lam * np.eye(10), -J.T @ r)
            p = p + np.clip(step, -0.2, 0.2)
            p[:9] = np.clip(p[:9], self.lo, self.hi)
            if np.linalg.norm(step) < 1e-7:
                break
        return p

    def _result(self, p, pos_res):
        pos, nrm = self._fk(p)
        margin = float(np.min(np.minimum(p[:9] - self.lo, self.hi - p[:9])))
        return pos, nrm, IKResult(p[:9].copy(), float(p[9]), float(np.sqrt(np.mean(pos_res(pos) ** 2))),
                                  0.0, margin, pos, nrm)

    # --------------------------------------------------------------------------------------
    def grasp_ik(self, obj_pos, radius, squeeze=0.0015, q0=None, palm0=0.0, w_eq=0.3, w_y=0.05,
                 palm_free=True, face_margin=0.002, w_floor=3.0, target_y=None) -> IKResult:
        """Pinch the cylinder lying along y with its axis through `obj_pos`.

        Each pad FACE must be tangent to the cylinder (minus `squeeze`): its plane at distance
        r from the axis, its normal perpendicular to the axis and pointing at it. The tangent
        line may land anywhere on the face that is at least `face_margin` from its edge -- a
        real pad touches with whatever part of its face meets the shaft, and on a small shaft
        lying on a table that is the lower part, because the fingertip below the pad has to
        clear the table.
        """
        c = np.asarray(obj_pos, float)
        mount_y = np.array([self.m.body(f"{f}_mount").pos[1] for f in scene.FINGERS])
        if target_y is not None:          # explicit contact positions along the shaft (world y)
            mount_y = np.asarray(target_y, float)
        target_r = radius - squeeze
        ps = self.ix["pad_site"]
        half = self.m.site_size[ps].copy()
        half[:, 2] = 0.0                                         # in-plane extents only

        def face(pos, nrm):
            """Tangent-line point on each face, its in-face offset, and the face half-extent."""
            e = np.cross(nrm, Y)
            e /= np.linalg.norm(e, axis=1, keepdims=True)
            a = c + np.outer((pos - c) @ Y, Y)                   # axis point level with each pad
            sd = ((a - pos) * nrm).sum(1)                        # axis distance along the normal
            t = a - nrm * sd[:, None]                            # tangent line, on the face plane
            off = ((t - pos) * e).sum(1)
            ext = np.array([np.abs(self.d.site_xmat[sid].reshape(3, 3).T @ e[i]) @ half[i]
                            for i, sid in enumerate(ps)])         # face reach along e
            return t, sd, off, ext

        def res(pos, nrm):
            t, sd, off, ext = face(pos, nrm)
            lim = ext - face_margin
            r_floor = w_floor * max(0.0, self.floor_clear - self.finger_min_z())
            r_dist = sd - target_r                               # tangent (minus squeeze)
            r_perp = 0.02 * (nrm @ Y)                            # normal perpendicular to the axis
            r_face = 3.0 * np.maximum(0.0, np.abs(off) - lim)    # contact line on the face
            r_mid = 0.05 * off                                   # ... preferably mid-face
            r_eq = w_eq * 0.02 * nrm[:, 2]                       # prefer an equatorial contact
            r_y = w_y * (t[:, 1] - mount_y)                      # prefer the mount's y
            return np.concatenate([r_dist, r_perp, r_face, r_mid, r_eq, r_y, [r_floor]])

        p0 = np.concatenate([q0 if q0 is not None else np.tile([0, 0.6, 0.6], 3), [palm0]])
        p = self._solve(res, p0, palm_free=palm_free)
        pos, nrm = self._fk(p)
        t, sd, off, ext = face(pos, nrm)
        margin = float(np.min(np.minimum(p[:9] - self.lo, self.hi - p[:9])))
        out = IKResult(p[:9].copy(), float(p[9]), float(np.sqrt(np.mean((sd - target_r) ** 2))),
                       float(np.degrees(np.max(np.abs(np.arcsin(np.clip(nrm @ Y, -1, 1)))))), margin, t, nrm)
        out.face_offset = off
        out.face_ok = bool(np.all(np.abs(off) <= ext - face_margin + 1e-4))
        out.floor_z = self.finger_min_z()
        out.contact_elev_deg = np.degrees(np.arcsin(np.clip(-nrm[:, 2], -1, 1)))   # + = contact above equator
        return out

    def track_ik(self, targets, normals, q0, palm0, palm_free=False, w_n=0.02) -> IKResult:
        """Each pad face onto targets[i] with outward normal along normals[i] (world)."""
        targets, normals = np.asarray(targets), np.asarray(normals)

        def res(pos, nrm):
            return np.concatenate([(pos - targets).ravel(), w_n * (nrm - normals).ravel()])

        p = self._solve(res, np.concatenate([q0, [palm0]]), palm_free=palm_free)
        pos, nrm, out = self._result(p, lambda pos: np.linalg.norm(pos - targets, axis=1))
        out.normal_err_deg = float(np.degrees(np.max(np.arccos(np.clip((nrm * normals).sum(1), -1, 1)))))
        return out
