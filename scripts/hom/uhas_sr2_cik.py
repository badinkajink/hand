#!/usr/bin/env python3
"""The sphere controller of UHAS (Casas et al., "Cross-Embodiment Robot Manipulation via a Unified Hand Action Space",
arXiv 2607.03570, 2026; code in docs/uhas/UHAS_sim) for the SR2 hand's layouts, in torch and batched over MuJoCo-Warp
worlds that each carry their own layout (the inner-loop benchmark, owner 2026-11-12).

UHAS builds its controller from a hand's URDF with tools that run beside Isaac Lab (process_urdf). The SR2 layouts share
one finger and differ only in where the three fingers are mounted, so the same construction is computed here for every
world at once:

  sphere (paper Section 3.1): the palm centre c is the mean of the finger roots (the yaw joints' anchors) and l the mean
      distance from c to the fingertips (the pads' distal ends) in the open hand, every joint at zero and every finger along
      the palm's outward normal n. The sphere's radius is r = 2 l / pi and its centre c + r n. Its z axis is n and its x
      axis points from the thumb's root to the midpoint of the index's and the middle's roots.
  deformation (Section 3.3; multi_env_cfg.py: one driving plane per finger, driving vectors at polar angles 50 and
      120 deg, offsets in [-2, 2] radii, 1,000 Fibonacci samples): each finger's sphere has the radius rho(phi) r, with rho
      interpolated in the polar angle between the finger's two offsets and constant beyond them. A sample of negative
      radius is dropped.
  cascade IK (Section 3.4; sphere_torch_utils.torch_solve_for_A_joints_ohne_interpolation and torch_solve_for_B_joint):
      the lateral joint (yaw) from a table of the fingertip's azimuth on the undeformed sphere over the yaw's range, then
      each encompassing joint in turn (mcp, then pip). For an encompassing joint the samples of the finger's deformed
      sphere qualify when they lie within 0.1 r of the joint's plane of motion, at least the link's reach from the joint,
      and outside the quadrant behind the joint on the extended side. The joint takes the angle that points the link at
      the first qualifying sample met from the extended side, the smallest such angle, and its upper limit when none
      qualifies. The link of the mcp ends at the pip's anchor and the link of the pip at the pad's contact point.
  actions (multi_manipulation_env._pre_physics_step): nine numbers in [-1, 1]. The three lateral offsets are scaled by the
      table's range on either side of the finger's anchor azimuth (the azimuth at zero yaw), and the six driving-vector
      offsets are scaled to [-2, 2].

The finger points UHAS observes (two of seven points from root to tip, the middle one and the tip) are computed here too.

    .venv/bin/python scripts/hom/uhas_sr2_cik.py check      # the torch kinematics against MuJoCo's, and postures
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
from sr2hand.rl.math import quat_from_axis_angle, quat_to_mat  # noqa: E402

FINGERS = ("thumb", "index", "middle")
JOINTS = ("yaw", "mcp", "pip")
VECTOR_PHIS = (math.radians(50.0), math.radians(120.0))      # multi_env_cfg.vector_phis
OFFSET_LO, OFFSET_HI = -2.0, 2.0                             # min_vector_offsets, max_vector_offsets (sphere radii)
N_SURF = 1000                                                 # n_sphere_surface_samples
Z_BAND = 0.1                                                  # the finger-width band, in sphere radii
N_YAW = 121                                                   # samples of the lateral table over the yaw's range


def fibonacci(n, device):
    """UHAS's torch_sample_fibonacci_points_sphere: the azimuth and polar angle of n points, and their unit vectors."""
    i = torch.arange(n, device=device, dtype=torch.float32)
    phi = torch.acos(1 - 2 * i / n)
    theta = (2 * math.pi * i / ((1 + math.sqrt(5.0)) / 2)) % (2 * math.pi)
    theta[-1], phi[-1] = math.pi, 0.0
    u = torch.stack([torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi)], -1)
    return theta, phi, u


class Fingers:
    """The SR2 finger chains of a compiled model. Every body of a finger has the identity orientation in its parent, so a
    finger is four offsets (mount to yaw, yaw to mcp, mcp to pip, pip to tip), three joint axes with their ranges, and two
    pad points in the tip's frame: the distal end and the contact point, the pad point farthest toward the palm's centre
    at mid-height."""

    def __init__(self, m, device="cuda"):
        off, ax, lo, hi, end, pad = [], [], [], [], [], []
        for f in FINGERS:
            ids = [m.body(f"{f}_{n}").id for n in ("yaw_frame", "mcp_frame", "len_frame", "pip_frame", "tip")]
            for b in ids:
                if not np.allclose(m.body_quat[b], [1.0, 0.0, 0.0, 0.0]):
                    raise ValueError(f"{m.body(b).name} is rotated in its parent; Fingers assumes it is not")
            yaw, mcp, ln, pip, tip = ids
            off.append([m.body_pos[yaw], m.body_pos[mcp], m.body_pos[ln] + m.body_pos[pip], m.body_pos[tip]])
            jid = [m.joint(f"{f}_{j}").id for j in JOINTS]
            if not np.allclose(m.jnt_pos[jid], 0.0):
                raise ValueError(f"{f}: a joint is off its body's origin")
            ax.append(m.jnt_axis[jid])
            lo.append(m.jnt_range[jid, 0])
            hi.append(m.jnt_range[jid, 1])
            pads = [g for g in range(m.ngeom) if m.geom_bodyid[g] == tip and "_pad" in (m.geom(g).name or "")]
            P, rad = m.geom_pos[pads], float(m.geom_size[pads[0], 0])
            down = m.body_pos[tip] / np.linalg.norm(m.body_pos[tip])
            flex = np.cross(m.jnt_axis[jid[2]], m.body_pos[tip])
            flex /= np.linalg.norm(flex)
            end.append(P[np.argmax(P @ down)] + rad * down)
            mid = P[np.abs(P @ down - (P @ down).mean()) < 0.002]
            pad.append(mid[np.argmax(mid @ flex)] + rad * flex)
        t = (lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=device))
        self.off, self.axis, self.lo, self.hi, self.end, self.pad = map(t, (off, ax, lo, hi, end, pad))
        self.device = device

    def fk(self, mounts, q):
        """The finger points and frames in the palm frame. mounts (W, 3, 3): each finger's mount position; q (W, 3, 3):
        the joint angles by (finger, joint). Returns the anchors of yaw, mcp and pip, the tip body, the pad's distal end
        and contact point (each (W, 3, 3)) and the frames after each joint (three (W, 3, 3, 3))."""
        W = q.shape[0]
        Rj = quat_to_mat(quat_from_axis_angle(self.axis.expand(W, 3, 3, 3), q))
        p_yaw = mounts + self.off[:, 0]
        R1 = Rj[:, :, 0]
        p_mcp = p_yaw + (R1 @ self.off[:, 1, :, None])[..., 0]
        R2 = R1 @ Rj[:, :, 1]
        p_pip = p_mcp + (R2 @ self.off[:, 2, :, None])[..., 0]
        R3 = R2 @ Rj[:, :, 2]
        p_tip = p_pip + (R3 @ self.off[:, 3, :, None])[..., 0]
        return {"yaw": p_yaw, "mcp": p_mcp, "pip": p_pip, "tip": p_tip,
                "end": p_tip + (R3 @ self.end[..., None])[..., 0], "contact": p_tip + (R3 @ self.pad[..., None])[..., 0],
                "R": (R1, R2, R3)}

    def points(self, k):
        """UHAS's homogeneous observation points: of seven points spaced evenly along each finger from its root (the yaw
        anchor) through the mcp and pip anchors to the distal end, the middle one and the last. Returns (W, 3, 2, 3)."""
        P = torch.stack([k["yaw"], k["mcp"], k["pip"], k["end"]], 2)            # (W, 3, 4, 3)
        seg = (P[:, :, 1:] - P[:, :, :-1]).norm(dim=-1)                          # (W, 3, 3)
        s = torch.cat([torch.zeros_like(seg[..., :1]), seg.cumsum(-1)], -1)      # arc length at each vertex
        half = 0.5 * s[..., -1:]
        i = torch.searchsorted(s.contiguous(), half.contiguous()).clamp(1, 3) - 1
        a = torch.gather(s, 2, i)
        w = ((half - a) / torch.gather(seg, 2, i).clamp_min(1e-9))[..., None]
        P0 = torch.gather(P, 2, i[..., None].expand(-1, -1, -1, 3))
        P1 = torch.gather(P, 2, (i + 1)[..., None].expand(-1, -1, -1, 3))
        return torch.stack([(P0 + w * (P1 - P0))[:, :, 0], P[:, :, -1]], 2)


def b_angle(P, keep, p0, axis, e, r):
    """UHAS's rule for one encompassing joint of one finger in every world (torch_solve_for_B_joint). P (W, N, 3): the
    finger's deformed-sphere samples; keep (W, N): samples of nonnegative radius; p0 (W, 3): the joint's anchor; axis
    (W, 3): its axis; e (W, 3): the link's end with this joint at zero; r (W,): the sphere's radius (all in the palm frame).
    Returns the joint angle (W,) that points the link at the first qualifying sample met from the extended side, and
    whether any sample qualified."""
    u = e - p0
    u = u - (u * axis).sum(-1, keepdim=True) * axis
    reach = u.norm(dim=-1)
    xh = u / reach[:, None]
    yh = torch.cross(axis, xh, dim=-1)                             # where a positive angle moves the link
    v = P - p0[:, None]
    x, y, z = (v * xh[:, None]).sum(-1), (v * yh[:, None]).sum(-1), (v * axis[:, None]).sum(-1)
    ok = keep & (torch.hypot(x, y) >= reach[:, None]) & (z.abs() <= Z_BAND * r[:, None]) & ~((x < 0) & (y < 0))
    ang = torch.atan2(y, x).masked_fill(~ok, float("inf")).min(1).values
    return ang, ok.any(1)


class SphereController:
    """UHAS's canonical sphere, deformation and cascade IK for W worlds, each with its own finger mounts (W, 3, 3) in
    the palm frame."""

    def __init__(self, fingers: Fingers, mounts):
        self.F, dev = fingers, fingers.device
        self.mounts = torch.as_tensor(mounts, dtype=torch.float32, device=dev)
        W = self.W = self.mounts.shape[0]
        k = self.F.fk(self.mounts, torch.zeros(W, 3, 3, device=dev))
        c = k["yaw"].mean(1)
        n = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(W, 3)    # the palm's outward normal: the fingers hang
        if ((k["end"].mean(1) - c) * n).sum(-1).min() <= 0:
            raise ValueError("the open fingertips do not lie along -z of the palm frame")
        self.r = 2.0 * (k["end"] - c[:, None]).norm(dim=-1).mean(1) / math.pi
        self.c = c + self.r[:, None] * n
        x = 0.5 * (k["yaw"][:, 1] + k["yaw"][:, 2]) - k["yaw"][:, 0]
        x = x - (x * n).sum(-1, keepdim=True) * n
        x = x / x.norm(dim=-1, keepdim=True)
        self.R = torch.stack([x, torch.cross(n, x, dim=-1), n], -1)     # columns: the sphere's axes in the palm frame
        self.theta_s, self.phi_s, self.u = fibonacci(N_SURF, dev)
        w = ((self.phi_s - VECTOR_PHIS[0]) / (VECTOR_PHIS[1] - VECTOR_PHIS[0])).clamp(0.0, 1.0)
        self.w_phi = w                                                    # interpolation weight of the second vector
        self.U = (self.R[:, None] @ self.u[None, :, :, None])[..., 0]     # (W, N, 3): the samples' directions
        self._table()

    PER_WORLD = ("mounts", "r", "c", "R", "U", "th0", "dth", "dmin", "dmax", "q_ref")

    def subset(self, idx):
        """A controller for the worlds idx of this one (their tensors copied)."""
        out = SphereController.__new__(SphereController)
        out.F, out.theta_s, out.phi_s, out.u, out.w_phi, out.yaw_grid = (self.F, self.theta_s, self.phi_s, self.u,
                                                                          self.w_phi, self.yaw_grid)
        for k in self.PER_WORLD:
            setattr(out, k, getattr(self, k)[idx].clone())
        out.W = int(out.r.shape[0])
        return out

    def copy_rows(self, src, dst_idx, src_idx):
        """Worlds dst_idx of this controller take the tensors of src's worlds src_idx (a world reset to a new layout)."""
        for k in self.PER_WORLD:
            getattr(self, k)[dst_idx] = getattr(src, k)[src_idx]

    def surface(self, rho):
        """The samples of the fingers' deformed spheres: rho (W, 3, N) in sphere radii -> (W, 3, N, 3)."""
        return self.c[:, None, None] + (rho * self.r[:, None, None])[..., None] * self.U[:, None]

    def _encompass(self, yaw, P, keep):
        """mcp, then pip, of every finger by UHAS's rule, with the yaw angles (W, 3) and the samples P (W, 3, N, 3)."""
        W, dev = self.W, self.F.device
        q = torch.zeros(W, 3, 3, device=dev)
        q[..., 0] = yaw
        for j, (anchor, end) in ((1, ("mcp", "pip")), (2, ("pip", "contact"))):
            k = self.F.fk(self.mounts, q)                                 # this joint and the next at zero
            axis = (k["R"][j - 1] @ self.F.axis[:, j, :, None])[..., 0]
            for f in range(3):
                ang, ok = b_angle(P[:, f], keep[:, f], k[anchor][:, f], axis[:, f], k[end][:, f], self.r)
                lo, hi = self.F.lo[f, j], self.F.hi[f, j]
                q[:, f, j] = torch.where(ok, ang.clamp(lo, hi), hi)
        return q

    def _table(self):
        """The lateral table: the fingertip's azimuth on the undeformed sphere over the yaw's range, its value at zero
        yaw (the finger's anchor) and the range of azimuth offsets on either side."""
        dev, W = self.F.device, self.W
        P = self.surface(torch.ones(W, 3, N_SURF, device=dev))
        keep = torch.ones(W, 3, N_SURF, dtype=torch.bool, device=dev)
        self.yaw_grid = torch.stack([torch.linspace(float(self.F.lo[f, 0]), float(self.F.hi[f, 0]), N_YAW, device=dev)
                                     for f in range(3)])                  # (3, G)
        th = torch.zeros(W, 3, N_YAW, device=dev)
        for g in range(N_YAW):
            q = self._encompass(self.yaw_grid[:, g].expand(W, 3), P, keep)
            th[..., g] = self.azimuth(self.F.fk(self.mounts, q)["end"])
        zero = torch.argmin(self.yaw_grid.abs(), 1)                      # (3,)
        th0 = th[:, torch.arange(3, device=dev), zero]                    # (W, 3)
        d = torch.remainder(th - th0[..., None] + math.pi, 2 * math.pi) - math.pi
        self.th0, self.dth = th0, d
        self.dmin, self.dmax = d.min(-1).values, d.max(-1).values
        self.q_ref = self._encompass(torch.zeros(W, 3, device=dev), P, keep)

    def azimuth(self, p):
        """Azimuth (W, 3) of points p (W, 3, 3) about the sphere's z axis, in its frame."""
        v = ((p - self.c[:, None])[..., None, :] @ self.R[:, None])[..., 0, :]
        return torch.atan2(v[..., 1], v[..., 0])

    def to_sphere(self, p):
        """Points (W, ..., 3) in the palm frame -> the sphere's frame, in sphere radii."""
        sh = p.shape
        v = (p.reshape(self.W, -1, 3) - self.c[:, None]) @ self.R
        return (v / self.r[:, None, None]).reshape(sh)

    def solve(self, a):
        """Joint angles (W, 3, 3) for actions a (W, 9) in [-1, 1]: three lateral offsets, then two driving-vector offsets
        per finger."""
        a = a.clamp(-1.0, 1.0)
        lat, vec = a[:, :3], a[:, 3:].reshape(self.W, 3, 2)
        off = OFFSET_LO + 0.5 * (vec + 1.0) * (OFFSET_HI - OFFSET_LO)
        rho = off[..., :1] * (1.0 - self.w_phi) + off[..., 1:] * self.w_phi    # (W, 3, N)
        dth = torch.where(lat < 0, -lat * self.dmin, lat * self.dmax)
        g = torch.argmin((self.dth - dth[..., None]).abs(), -1)                  # (W, 3)
        yaw = torch.gather(self.yaw_grid.expand(self.W, 3, N_YAW), 2, g[..., None])[..., 0]
        return self._encompass(yaw, self.surface(rho), rho >= 0)


def cmd_check():
    """The torch kinematics against MuJoCo's on random angles and layouts, and a film strip of postures."""
    os.environ.setdefault("SCALE_HAND", "cad_v2")
    import mujoco
    import inner_loop_plant  # noqa: F401  the talk's plant, set before hom_gpu reads it
    import hom_gpu as HG
    sc = HG.Scene("sphere", 40.0)
    m = mujoco.MjModel.from_xml_path(str(sc.pads_path))
    d = mujoco.MjData(m)
    F = Fingers(m, "cpu")
    rng = np.random.default_rng(0)
    pb = m.body("palm_pose").id
    err = []
    for _ in range(20):
        xs, ys = rng.uniform(40, 160), rng.uniform(50, 170)
        bm = sc.body_mounts(HG.mount_xy(HG.sym6(xs, ys)))
        q = rng.uniform(F.lo.numpy(), F.hi.numpy())
        for i, f in enumerate(FINGERS):
            m.body_pos[m.body(f"{f}_mount").id] = bm[f]
            for j, jn in enumerate(JOINTS):
                d.qpos[m.jnt_qposadr[m.joint(f"{f}_{jn}").id]] = q[i, j]
        mujoco.mj_kinematics(m, d)
        Rp, pp = d.xmat[pb].reshape(3, 3), d.xpos[pb]
        k = F.fk(torch.tensor(np.array([bm[f] for f in FINGERS]), dtype=torch.float32)[None],
                 torch.tensor(q, dtype=torch.float32)[None])
        for i, f in enumerate(FINGERS):
            for name, body in (("mcp", "mcp_frame"), ("pip", "pip_frame"), ("tip", "tip")):
                ref = Rp.T @ (d.xpos[m.body(f"{f}_{body}").id] - pp)
                err.append(float(np.linalg.norm(k[name][0, i].numpy() - ref)))
    print(f"FK against MuJoCo, 20 layouts x 3 fingers x 3 bodies: max error {1e3 * max(err):.4f} mm")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(cmd_check())
