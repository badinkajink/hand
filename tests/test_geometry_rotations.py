"""morphohand.geometry.rotations against scipy and MuJoCo, including the cases the audit of 2026-11-03 found wrong in
hand-written copies: half turns, antiparallel vectors, a rounded matrix near 180 deg, and yaw composed on the wrong
side of a flat tool's quaternion."""
import ast
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from morphohand.geometry import rotations as G

RNG = np.random.default_rng(20261103)
Q = G.quat_random(200, 7)
TOL = 1e-12

try:
    import mujoco
except ImportError:  # the femtip environment
    mujoco = None
needs_mujoco = pytest.mark.skipif(mujoco is None, reason="MuJoCo is not installed")


def test_imports_numpy_and_scipy_only():
    tree = ast.parse(Path(G.__file__).read_text())
    mods = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert mods == {"numpy", "scipy"}


def test_quaternion_order_and_matrix_match_scipy():
    R = Rotation.from_quat(Q[:, [1, 2, 3, 0]]).as_matrix()
    assert np.allclose(G.quat_to_mat(Q), R, atol=TOL)
    assert np.allclose(G.quat_to_mat(G.mat_to_quat(R)), R, atol=TOL)
    assert np.all(G.mat_to_quat(R)[:, 0] >= 0)
    assert np.allclose(G.xyzw_to_wxyz(G.wxyz_to_xyzw(Q)), Q)


def test_batch_shapes():
    q = Q[:6].reshape(2, 3, 4)
    assert G.quat_to_mat(q).shape == (2, 3, 3, 3)
    assert G.mat_to_quat(G.quat_to_mat(q)).shape == (2, 3, 4)
    assert G.quat_to_rotvec(q).shape == (2, 3, 3)
    assert G.rot_z(np.zeros((2, 3))).shape == (2, 3, 3, 3)
    assert G.quat_to_mat(Q[0]).shape == (3, 3) and G.rot_x(0.3).shape == (3, 3)


def test_product_conjugate_and_apply():
    a, b = Q[:100], Q[100:]
    ab = G.quat_mul(a, b)
    ref = (Rotation.from_quat(a[:, [1, 2, 3, 0]]) * Rotation.from_quat(b[:, [1, 2, 3, 0]])).as_matrix()
    assert np.allclose(G.quat_to_mat(ab), ref, atol=TOL)
    assert np.allclose(G.quat_mul(a, G.quat_conj(a)), [1, 0, 0, 0], atol=TOL)
    v = RNG.normal(size=(100, 3))
    assert np.allclose(G.quat_apply(a, v), np.einsum("nij,nj->ni", G.quat_to_mat(a), v), atol=TOL)
    assert np.allclose(G.quat_apply_inv(a, G.quat_apply(a, v)), v, atol=1e-10)


@needs_mujoco
def test_product_apply_and_axis_angle_match_mujoco():
    for a, b in zip(Q[:50], Q[50:100]):
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, a, b)
        assert np.allclose(G.quat_mul(a, b), out, atol=TOL)
        v, r = RNG.normal(size=3), np.zeros(3)
        mujoco.mju_rotVecQuat(r, v, a)
        assert np.allclose(G.quat_apply(a, v), r, atol=1e-12)
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, a)
        assert np.allclose(G.quat_to_mat(a), m.reshape(3, 3), atol=TOL)
        axis, ang = RNG.normal(size=3), RNG.uniform(-math.pi, math.pi)
        qm = np.zeros(4)
        mujoco.mju_axisAngle2Quat(qm, axis / np.linalg.norm(axis), ang)
        assert G.quat_angle(G.quat_from_axis_angle(axis, ang), qm) < 1e-12


def test_exp_and_log_including_half_turns():
    w = RNG.normal(size=(200, 3))
    w *= (RNG.uniform(0, math.pi, 200) / np.linalg.norm(w, axis=1))[:, None]
    assert np.allclose(G.exp_so3(w), Rotation.from_rotvec(w).as_matrix(), atol=TOL)
    assert np.allclose(G.log_so3(G.exp_so3(w)), w, atol=1e-9)
    for axis in ([1, 0, 0], [0, 0, 1], [1, 1, 0], [1, -2, 3]):
        a = np.array(axis, float) / np.linalg.norm(axis)
        R = G.rot_axis_angle(a, math.pi)
        assert np.allclose(G.exp_so3(G.log_so3(R)), R, atol=1e-9)
        assert abs(np.linalg.norm(G.log_so3(R)) - math.pi) < 1e-9


def test_log_of_a_rounded_matrix_near_a_half_turn():
    """The UR5 probe straight down at 178.6 deg: a plan rounded to 7 decimals must not move the rotation by a degree."""
    R = G.euler_to_mat([math.radians(178.6), 0.2, -0.4], "xyz")
    w = G.log_so3(np.round(R, 7))
    assert G.rot_angle(G.exp_so3(w), R) < 1e-5


def test_rot_between_is_a_rotation_for_every_pair():
    a, b = RNG.normal(size=(100, 3)), RNG.normal(size=(100, 3))
    R = G.rot_between(a, b)
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    assert np.allclose(np.einsum("nij,nj->ni", R, unit(a)), unit(b), atol=1e-12)
    assert np.allclose(G.rot_between([0, 0, 1], [0, 0, 2]), np.eye(3), atol=TOL)
    for v in ([-1, 0, 0], [0, 0, 1], [0.3, -0.2, 0.9]):
        R = G.rot_between(v, -np.array(v))
        assert np.isclose(np.linalg.det(R), 1.0) and np.allclose(R @ v, -np.array(v), atol=1e-12)


def test_axis_rotations_euler_and_heading():
    a = 0.7
    c, s = math.cos(a), math.sin(a)
    assert np.allclose(G.rot_x(a), [[1, 0, 0], [0, c, -s], [0, s, c]], atol=TOL)
    assert np.allclose(G.rot_y(a), [[c, 0, s], [0, 1, 0], [-s, 0, c]], atol=TOL)
    assert np.allclose(G.rot_z(a), [[c, -s, 0], [s, c, 0], [0, 0, 1]], atol=TOL)
    ang = RNG.uniform(-1.2, 1.2, size=(50, 3))
    for seq in ("xyz", "XYZ", "zyx"):
        assert np.allclose(G.euler_to_mat(ang, seq), Rotation.from_euler(seq, ang).as_matrix(), atol=TOL)
        assert np.allclose(G.mat_to_euler(G.euler_to_mat(ang, seq), seq), ang, atol=1e-9)
    assert np.isclose(G.heading(G.rot_z(2.5)), 2.5)


def test_yaw_composed_in_the_world_frame_turns_a_flat_tool():
    """plant_drop_gate's jitter composed q * dq, a turn about the tool's own shaft: the heading did not move."""
    q0 = np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])        # the plans' flat screwdriver
    h0 = G.heading(G.quat_to_mat(q0))
    q1 = G.quat_mul(G.quat_yaw(0.1), q0)
    assert np.isclose(G.heading(G.quat_to_mat(q1)) - h0, 0.1)
    assert np.isclose(G.tilt_cos(q1), G.tilt_cos(q0))


def test_vertical_cosine_and_axes():
    x, y = Q[:, 1], Q[:, 2]
    assert np.allclose(G.tilt_cos(Q), 1 - 2 * (x * x + y * y), atol=TOL)
    assert np.allclose(G.quat_axis(Q, 0), G.quat_to_mat(Q)[:, :, 0])


def test_frames_skew_and_projection():
    n = RNG.normal(size=(100, 3))
    for which in (0, 1, 2):
        F = G.frame_from_axis(n, which=which)
        assert np.allclose(np.swapaxes(F, -1, -2) @ F, np.eye(3), atol=1e-12)
        assert np.allclose(np.linalg.det(F), 1.0)
        assert np.allclose(F[:, :, which], n / np.linalg.norm(n, axis=1, keepdims=True), atol=1e-12)
    F = G.frame_from_axis([1, 0, 0], hint=[2, 0, 0])                    # hint along the axis
    assert np.allclose(F.T @ F, np.eye(3), atol=1e-12)
    t1, t2 = G.tangent_basis(n)
    assert np.allclose(np.cross(t1, t2), n / np.linalg.norm(n, axis=1, keepdims=True), atol=1e-12)
    v, u = RNG.normal(size=(20, 3)), RNG.normal(size=(20, 3))
    assert np.allclose(np.einsum("nij,nj->ni", G.skew(v), u), np.cross(v, u))
    assert np.allclose(G.vee(G.skew(v)), v)
    R = G.quat_to_mat(Q[:20])
    assert np.allclose(G.project_to_so3(R + 1e-4 * RNG.normal(size=R.shape)), R, atol=1e-3)
    assert np.allclose(np.linalg.det(G.project_to_so3(RNG.normal(size=(20, 3, 3)))), 1.0)


def test_transforms_and_ur_poses():
    R, t = G.quat_to_mat(Q[:10]), RNG.normal(size=(10, 3))
    T = G.tf(R, t)
    assert np.allclose(G.tf_compose(T, G.tf_inv(T)), np.eye(4), atol=1e-12)
    p = RNG.normal(size=(10, 3))
    assert np.allclose(G.tf_apply(T, p), np.einsum("nij,nj->ni", R, p) + t)
    pos, q = G.tf_to_pose(T)
    assert np.allclose(G.tf_from_pose(pos, q), T, atol=1e-12)
    ur = np.concatenate([RNG.normal(size=3) * 0.3, G.log_so3(R[0])])
    assert np.allclose(G.tf_to_ur(G.tf_from_ur(ur)), ur, atol=1e-12)
    assert np.allclose(G.tf_from_ur(ur)[:3, 3], ur[:3] * 1e3)
    d = RNG.normal(size=6) * 0.1
    assert np.allclose(G.perturb_body(T[0], d), T[0] @ G.tf(G.exp_so3(d[:3]), d[3:]))
    c = np.array([1.0, 2.0, 3.0])
    Tc = G.turn_about(G.tf(t=c), c, [0, 1, 1], 0.8)                     # a frame at c turned about a line through c
    assert np.allclose(Tc[:3, 3], c) and np.isclose(G.rot_angle(np.eye(3), Tc[:3, :3]), 0.8)
    dist, ang = G.pose_error(T[0], G.perturb_body(T[0], [0, 0, 0.2, 0, 0, 0]))
    assert np.isclose(dist, 0.0) and np.isclose(ang, 0.2)


def test_slerp_and_angle():
    a, b = Q[0], Q[1]
    assert np.allclose(G.quat_to_mat(G.quat_slerp(a, b, 0.0)), G.quat_to_mat(a), atol=1e-12)
    assert np.allclose(G.quat_to_mat(G.quat_slerp(a, b, 1.0)), G.quat_to_mat(b), atol=1e-12)
    mid = G.quat_slerp(a, b, 0.5)
    assert np.isclose(G.quat_angle(a, mid), G.quat_angle(a, b) / 2)
    assert np.isclose(G.quat_angle(a, -a), 0.0, atol=1e-7)             # q and -q are one rotation


def test_registration():
    R, t = G.quat_to_mat(Q[3]), np.array([0.1, -0.2, 0.3])
    A = RNG.normal(size=(30, 3))
    B = A @ R.T + t
    R1, t1 = G.kabsch(A, B)
    assert np.allclose(R1, R, atol=1e-10) and np.allclose(t1, t, atol=1e-10)
    R2, t2 = G.kabsch(A, B, w=RNG.uniform(0.5, 2.0, 30))
    assert np.allclose(R2, R, atol=1e-10) and np.allclose(t2, t, atol=1e-10)
    s, R3, t3 = G.umeyama(A, 1.7 * A @ R.T + t)
    assert np.isclose(s, 1.7) and np.allclose(R3, R, atol=1e-10) and np.allclose(t3, t, atol=1e-10)
