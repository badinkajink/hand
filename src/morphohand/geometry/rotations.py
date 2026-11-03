"""Rotations, quaternions and rigid transforms for the whole repository, over scipy.spatial.transform.Rotation.

Conventions: quaternions are (w, x, y, z), the order of MuJoCo and Drake; products are Hamilton; rotations are active
(R @ v turns v); angles are radians; transforms are 4 x 4 matrices [R t; 0 1] in the caller's length unit. Every
function takes one value or a batch over leading axes. Quaternions returned by a conversion have w >= 0; a product
keeps the sign its factors give.

The module imports numpy and scipy only. It runs in every environment of the repository (femtip has no MuJoCo) and on
the lab computer corvus, whose scipy 1.12 has no `scalar_first`, so the component order is changed by indexing. It has
no package-relative import, so a program that runs alone can carry a copy beside it. Torch versions of the quaternion
functions are in morphohand.rl.math. tests/test_geometry_rotations.py checks every function against scipy and, where
MuJoCo is installed, against its mju_* functions.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

__all__ = [
    "wxyz_to_xyzw", "xyzw_to_wxyz", "to_scipy", "from_scipy", "quat_normalize", "quat_canonical", "quat_to_mat",
    "mat_to_quat", "quat_mul", "quat_conj", "quat_apply", "quat_apply_inv", "quat_from_axis_angle", "quat_from_rotvec",
    "quat_to_rotvec", "quat_yaw", "quat_axis", "tilt_cos", "quat_angle", "quat_slerp", "quat_random",
    "rot_x", "rot_y", "rot_z", "exp_so3", "log_so3", "rot_axis_angle", "rot_angle", "heading", "rot_between",
    "frame_from_axis", "tangent_basis", "skew", "vee", "project_to_so3", "euler_to_mat", "mat_to_euler",
    "tf", "tf_inv", "tf_apply", "tf_compose", "tf_from_pose", "tf_to_pose", "tf_from_ur", "tf_to_ur", "perturb_body",
    "turn_about", "pose_error", "kabsch", "umeyama",
]

_EPS = 1e-12


def _arr(x):
    return np.asarray(x, dtype=float)


def _batched(fn, x, tail):
    """Apply fn, which takes an (N, *tail) array, to x of shape (..., *tail); return shape (..., *out_tail)."""
    x = _arr(x)
    lead = x.shape[:x.ndim - len(tail)]
    out = fn(x.reshape((-1,) + tuple(tail)))
    return out.reshape(lead + out.shape[1:])


# ------------------------------------------------------------------------------------------------------- quaternions
def wxyz_to_xyzw(q):
    """(w, x, y, z) -> (x, y, z, w), the order of scipy, warp and Newton."""
    return _arr(q)[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(q):
    """(x, y, z, w) -> (w, x, y, z)."""
    return _arr(q)[..., [3, 0, 1, 2]]


def to_scipy(q):
    """A scipy Rotation from wxyz quaternions of shape (4,) or (N, 4); scipy normalizes them."""
    return Rotation.from_quat(wxyz_to_xyzw(q))


def from_scipy(r):
    """wxyz quaternions of a scipy Rotation, with w >= 0."""
    return quat_canonical(xyzw_to_wxyz(r.as_quat()))


def quat_normalize(q):
    q = _arr(q)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), _EPS)


def quat_canonical(q):
    """The same rotation with w >= 0."""
    q = _arr(q)
    return np.where(q[..., :1] < 0.0, -q, q)


def quat_to_mat(q):
    """Rotation matrices (..., 3, 3) of wxyz quaternions (..., 4); a quaternion that is not unit is normalized."""
    return _batched(lambda x: to_scipy(x).as_matrix(), q, (4,))


def mat_to_quat(R):
    """wxyz quaternions (w >= 0) of rotation matrices; a matrix that is not orthonormal gives its nearest rotation."""
    return _batched(lambda x: from_scipy(Rotation.from_matrix(project_to_so3(x))), R, (3, 3))


def quat_mul(a, b):
    """Hamilton product a * b of wxyz quaternions: the rotation b, then a (as mju_mulQuat)."""
    a, b = np.broadcast_arrays(_arr(a), _arr(b))
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_conj(q):
    """The conjugate, which is the inverse of a unit quaternion."""
    return _arr(q) * np.array([1.0, -1.0, -1.0, -1.0])


def quat_apply(q, v):
    """v turned by q: R(q) @ v, broadcast over leading axes."""
    return np.einsum("...ij,...j->...i", quat_to_mat(q), _arr(v))


def quat_apply_inv(q, v):
    """v turned by the inverse of q: R(q).T @ v (a world vector in q's frame)."""
    return np.einsum("...ji,...j->...i", quat_to_mat(q), _arr(v))


def quat_from_rotvec(w):
    """wxyz quaternions of rotation vectors (axis times angle, rad)."""
    return _batched(lambda x: from_scipy(Rotation.from_rotvec(x)), w, (3,))


def quat_to_rotvec(q):
    """Rotation vectors (axis times angle, rad, angle in [0, pi]) of wxyz quaternions."""
    return _batched(lambda x: to_scipy(x).as_rotvec(), q, (4,))


def quat_from_axis_angle(axis, angle):
    """wxyz quaternion of a turn by angle (rad) about axis (any length but zero)."""
    axis = _arr(axis)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    return quat_from_rotvec(axis * _arr(angle)[..., None])


def quat_yaw(yaw):
    """wxyz quaternion of a turn by yaw (rad) about the world z axis; compose as quat_mul(quat_yaw(a), q)."""
    yaw = _arr(yaw)
    z = np.zeros_like(yaw)
    return quat_canonical(np.stack([np.cos(yaw / 2), z, z, np.sin(yaw / 2)], axis=-1))


def quat_axis(q, k=2):
    """Column k of R(q): the body's x (0), y (1) or z (2) axis in the world."""
    return quat_to_mat(q)[..., :, k]


def tilt_cos(q):
    """R(q)[2, 2], the cosine between the body's z axis and the world's z axis."""
    return quat_to_mat(q)[..., 2, 2]


def quat_angle(a, b):
    """Geodesic angle (rad, in [0, pi]) between the rotations a and b."""
    return np.linalg.norm(quat_to_rotvec(quat_mul(quat_conj(quat_normalize(a)), quat_normalize(b))), axis=-1)


def quat_slerp(a, b, t):
    """Spherical interpolation from a (t = 0) to b (t = 1) along the shorter arc, for scalar or array t."""
    s = Slerp([0.0, 1.0], to_scipy(np.stack([_arr(a), _arr(b)])))
    return from_scipy(s(np.clip(_arr(t), 0.0, 1.0)))


def quat_random(n, seed=None):
    """n uniformly distributed rotations as wxyz quaternions."""
    return from_scipy(Rotation.random(n, seed))


# ----------------------------------------------------------------------------------------------- rotation matrices
def rot_x(a):
    """Rotation matrices about x by a (rad)."""
    return _batched(lambda x: Rotation.from_euler("x", x).as_matrix(), _arr(a)[..., None], (1,))


def rot_y(a):
    return _batched(lambda x: Rotation.from_euler("y", x).as_matrix(), _arr(a)[..., None], (1,))


def rot_z(a):
    return _batched(lambda x: Rotation.from_euler("z", x).as_matrix(), _arr(a)[..., None], (1,))


def exp_so3(w):
    """Rotation matrix of a rotation vector (Rodrigues)."""
    return _batched(lambda x: Rotation.from_rotvec(x).as_matrix(), w, (3,))


def log_so3(R):
    """Rotation vector of a rotation matrix, angle in [0, pi]. The matrix is first replaced by its nearest rotation:
    near a half turn the formula of the trace turns a 7-decimal rounding into a degree (2026-11-02, the UR5 probe)."""
    return _batched(lambda x: Rotation.from_matrix(project_to_so3(x)).as_rotvec(), R, (3, 3))


def rot_axis_angle(axis, angle):
    """Rotation matrix of a turn by angle (rad) about axis (any length but zero)."""
    axis = _arr(axis)
    return exp_so3(axis / np.linalg.norm(axis, axis=-1, keepdims=True) * _arr(angle)[..., None])


def rot_angle(Ra, Rb):
    """Angle (rad) of the rotation that takes Ra to Rb."""
    return np.linalg.norm(log_so3(np.swapaxes(_arr(Ra), -1, -2) @ _arr(Rb)), axis=-1)


def heading(R):
    """Yaw (rad) of the body's x axis in the world's xy plane, atan2(R[1, 0], R[0, 0])."""
    R = _arr(R)
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def rot_between(a, b):
    """The least rotation taking direction a to direction b. For opposite directions it is a half turn about an axis
    perpendicular to a (a rotation, never the reflection -I)."""
    a, b = np.broadcast_arrays(_arr(a), _arr(b))
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    v = np.cross(a, b)
    s = np.linalg.norm(v, axis=-1)
    c = np.sum(a * b, axis=-1)
    axis = np.where((s > 1e-10)[..., None], v / np.maximum(s, _EPS)[..., None], 0.0)
    flip = (s <= 1e-10) & (c < 0.0)
    if np.any(flip):  # a perpendicular axis, from whichever world axis is least along a
        other = np.where((np.abs(a[..., 0]) < 0.9)[..., None], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        p = np.cross(a, other)
        axis = np.where(flip[..., None], p / np.linalg.norm(p, axis=-1, keepdims=True), axis)
    return exp_so3(axis * np.arctan2(s, c)[..., None])


def frame_from_axis(axis, hint=(1.0, 0.0, 0.0), which=2):
    """Right-handed frame (columns) whose column `which` is along axis and whose next column (cyclically) is the part
    of hint perpendicular to axis; a hint along axis is replaced by the world axis least along it."""
    z = _arr(axis)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    h = np.broadcast_to(_arr(hint), z.shape)
    x = h - np.sum(h * z, axis=-1, keepdims=True) * z
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    least = np.eye(3)[np.argmin(np.abs(z), axis=-1)]
    alt = least - np.sum(least * z, axis=-1, keepdims=True) * z
    x = np.where(n > 1e-9, x / np.maximum(n, _EPS), alt / np.linalg.norm(alt, axis=-1, keepdims=True))
    y = np.cross(z, x)
    cols = {2: (x, y, z), 0: (z, x, y), 1: (y, z, x)}[which]
    return np.stack(cols, axis=-1)


def tangent_basis(n):
    """Two unit vectors (t1, t2) with (t1, t2, n) a right-handed orthonormal frame."""
    F = frame_from_axis(n, hint=(1.0, 0.0, 0.0), which=2)
    return F[..., :, 0], F[..., :, 1]


def skew(v):
    """The matrix S with S @ u = v x u."""
    v = _arr(v)
    z = np.zeros(v.shape[:-1])
    return np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1),
                     np.stack([v[..., 2], z, -v[..., 0]], -1),
                     np.stack([-v[..., 1], v[..., 0], z], -1)], axis=-2)


def vee(S):
    """The vector v of a skew matrix S = skew(v) (from its antisymmetric part)."""
    S = _arr(S)
    return 0.5 * np.stack([S[..., 2, 1] - S[..., 1, 2], S[..., 0, 2] - S[..., 2, 0], S[..., 1, 0] - S[..., 0, 1]], -1)


def project_to_so3(M):
    """The rotation nearest to M in the Frobenius norm (SVD, determinant +1)."""
    U, _, Vt = np.linalg.svd(_arr(M))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones(U.shape[:-1])
    D[..., -1] = np.where(d == 0, 1.0, d)
    return (U * D[..., None, :]) @ Vt


def euler_to_mat(angles, seq):
    """Rotation matrices from Euler angles (rad) in scipy's convention: lower case 'xyz' turns about the fixed axes
    (URDF rpy is 'xyz'), upper case 'XYZ' about the body's own axes."""
    return _batched(lambda x: Rotation.from_euler(seq, x).as_matrix(), angles, (len(seq),))


def mat_to_euler(R, seq):
    return _batched(lambda x: Rotation.from_matrix(project_to_so3(x)).as_euler(seq), R, (3, 3))


# ----------------------------------------------------------------------------------------------- rigid transforms
def tf(R=None, t=None):
    """4 x 4 transform from a rotation matrix and a translation (either may be left out)."""
    lead = np.broadcast_shapes(*([_arr(R).shape[:-2]] if R is not None else []),
                               *([_arr(t).shape[:-1]] if t is not None else []))
    T = np.broadcast_to(np.eye(4), lead + (4, 4)).copy()
    if R is not None:
        T[..., :3, :3] = R
    if t is not None:
        T[..., :3, 3] = t
    return T


def tf_inv(T):
    """Inverse of a rigid transform, [R.T, -R.T t]."""
    T = _arr(T)
    Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    return tf(Rt, -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3]))


def tf_apply(T, P):
    """Points P (..., 3) mapped by T: R @ p + t."""
    T = _arr(T)
    return np.einsum("...ij,...j->...i", T[..., :3, :3], _arr(P)) + T[..., :3, 3]


def tf_compose(*Ts):
    """T1 @ T2 @ ... ."""
    out = _arr(Ts[0])
    for T in Ts[1:]:
        out = out @ _arr(T)
    return out


def tf_from_pose(p, q):
    """Transform of a position and a wxyz quaternion."""
    return tf(quat_to_mat(q), p)


def tf_to_pose(T):
    """(position, wxyz quaternion with w >= 0) of a transform."""
    T = _arr(T)
    return T[..., :3, 3].copy(), mat_to_quat(T[..., :3, :3])


def tf_from_ur(pose, scale=1e3):
    """UR pose [x, y, z (m), rotation vector (rad)] to a transform, its translation times scale (mm by default)."""
    pose = _arr(pose)
    return tf(exp_so3(pose[..., 3:6]), pose[..., :3] * scale)


def tf_to_ur(T, scale=1e-3):
    """Transform to a UR pose [x, y, z, rotation vector]; scale turns its translation unit into metres."""
    T = _arr(T)
    return np.concatenate([T[..., :3, 3] * scale, log_so3(T[..., :3, :3])], axis=-1)


def perturb_body(T, d):
    """T followed by the motion d = [rotation vector (rad), translation] in T's own frame."""
    d = _arr(d)
    return _arr(T) @ tf(exp_so3(d[..., :3]), d[..., 3:6])


def turn_about(T, c, axis, angle):
    """Transform T turned by angle (rad) about the line through point c along axis, all in the world frame."""
    R = rot_axis_angle(axis, angle)
    c = _arr(c)
    return tf(R, c - R @ c) @ _arr(T)


def pose_error(Ta, Tb):
    """(distance between the origins, angle in rad between the rotations) of two transforms."""
    Ta, Tb = _arr(Ta), _arr(Tb)
    return np.linalg.norm(Ta[..., :3, 3] - Tb[..., :3, 3], axis=-1), rot_angle(Ta[..., :3, :3], Tb[..., :3, :3])


# ---------------------------------------------------------------------------------------------------- registration
def kabsch(A, B, w=None):
    """(R, t) minimizing sum_i w_i |R a_i + t - b_i|^2 over rotations, for point sets A, B of shape (N, 3)."""
    A, B = _arr(A), _arr(B)
    w = np.ones(len(A)) if w is None else _arr(w)
    ca, cb = w @ A / w.sum(), w @ B / w.sum()
    r, _ = Rotation.align_vectors(B - cb, A - ca, weights=w)
    R = r.as_matrix()
    return R, cb - R @ ca


def umeyama(A, B):
    """(s, R, t) minimizing sum_i |s R a_i + t - b_i|^2 (Umeyama 1991), for point sets A, B of shape (N, 3)."""
    A, B = _arr(A), _arr(B)
    ca, cb = A.mean(0), B.mean(0)
    Ac, Bc = A - ca, B - cb
    U, S, Vt = np.linalg.svd(Bc.T @ Ac / len(A))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (Ac ** 2).sum(1).mean())
    return s, R, cb - s * R @ ca
