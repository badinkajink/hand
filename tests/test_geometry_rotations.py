"""morphohand.geometry.rotations against MuJoCo's own quaternion and rotation functions, on random rotations and on the
half turns and opposite vectors where hand-written copies went wrong (audit of 2026-11-03)."""
import math

import numpy as np
import pytest

from morphohand.geometry import rotations as G

mujoco = pytest.importorskip("mujoco")
RNG = np.random.default_rng(20261103)
Q = np.concatenate([G.quat_random(100, 7), G.quat_from_axis_angle(RNG.normal(size=(20, 3)), np.full(20, math.pi))])


def test_products_rotations_and_matrices_match_mujoco():
    for a, b in zip(Q, Q[::-1]):
        ab, v, r, m = np.zeros(4), RNG.normal(size=3), np.zeros(3), np.zeros(9)
        mujoco.mju_mulQuat(ab, a, b)
        mujoco.mju_rotVecQuat(r, v, a)
        mujoco.mju_quat2Mat(m, a)
        assert np.allclose(G.quat_mul(a, b), ab, atol=1e-12)
        assert np.allclose(G.quat_apply(a, v), r, atol=1e-12)
        assert np.allclose(G.quat_to_mat(a), m.reshape(3, 3), atol=1e-12)
        qm = np.zeros(4)
        mujoco.mju_mat2Quat(qm, m)
        assert G.quat_angle(G.mat_to_quat(m.reshape(3, 3)), qm) < 1e-7


def test_axis_angle_and_rotation_vectors_match_mujoco():
    for axis, ang in zip(RNG.normal(size=(60, 3)), np.r_[RNG.uniform(-math.pi, math.pi, 50), np.full(10, math.pi)]):
        unit = axis / np.linalg.norm(axis)
        qm, w = np.zeros(4), np.zeros(3)
        mujoco.mju_axisAngle2Quat(qm, unit, ang)
        assert G.quat_angle(G.quat_from_axis_angle(axis, ang), qm) < 1e-7
        mujoco.mju_quat2Vel(w, qm, 1.0)
        assert np.allclose(G.quat_apply(G.quat_from_rotvec(w), unit), G.quat_apply(qm, unit), atol=1e-9)


def test_rot_between_matches_mujoco_including_opposite_vectors():
    z = np.array([0.0, 0.0, 1.0])
    for v in [*RNG.normal(size=(40, 3)), [0, 0, 1.0], [0, 0, -1.0], [1e-12, 0, -1.0]]:
        v = np.asarray(v) / np.linalg.norm(v)
        qm = np.zeros(4)
        mujoco.mju_quatZ2Vec(qm, v)
        R = G.rot_between(z, v)
        assert np.isclose(np.linalg.det(R), 1.0)                       # a rotation, never the reflection -I
        assert np.allclose(R @ z, G.quat_apply(qm, z), atol=1e-9)
