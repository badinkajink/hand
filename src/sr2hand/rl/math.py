"""Shared batched quaternion math (torch, wxyz convention).

Single home for the quat helpers that were re-implemented in mjlab_terms and
imitation (CODEBASE_AUDIT.md step 3).
"""
from __future__ import annotations

import torch


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two batched wxyz quaternions, shapes (B, 4)."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors `v` (B, 3) by batched wxyz quats `q` (B, 4)."""
    # q * (0, v) * q^-1 via v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    qw = q[..., 0]
    qvec = q[..., 1:]
    t = 2.0 * torch.cross(qvec, v, dim=-1)
    return v + qw.unsqueeze(-1) * t + torch.cross(qvec, t, dim=-1)


def quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors `v` (B, 3) by the INVERSE of batched wxyz quats `q` (B, 4)."""
    return quat_rotate(quat_conj(q), v)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of wxyz quaternions, the inverse of a unit quaternion."""
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def tilt_cos(q: torch.Tensor) -> torch.Tensor:
    """R(q)[2, 2] of unit wxyz quaternions: the cosine between the body's z axis and the world's z axis."""
    _, x, y, _ = q.unbind(dim=-1)
    return 1.0 - 2.0 * (x * x + y * y)


def rotate_axis(q: torch.Tensor, axis: tuple[float, float, float]) -> torch.Tensor:
    """R(q) @ axis for unit wxyz quaternions (B, 4) and one body-local axis, as (B, 3)."""
    qw, qx, qy, qz = q.unbind(dim=-1)
    ax, ay, az = axis
    return torch.stack([(1 - 2 * (qy * qy + qz * qz)) * ax + 2 * (qx * qy - qw * qz) * ay + 2 * (qx * qz + qw * qy) * az,
                        2 * (qx * qy + qw * qz) * ax + (1 - 2 * (qx * qx + qz * qz)) * ay + 2 * (qy * qz - qw * qx) * az,
                        2 * (qx * qz - qw * qy) * ax + 2 * (qy * qz + qw * qx) * ay + (1 - 2 * (qx * qx + qy * qy)) * az],
                       dim=-1)
