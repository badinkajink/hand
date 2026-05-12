# pyright: reportMissingImports=false

import numpy as np
import pytest

from morphohand.optimization.force_closure import (
    ContactWrench,
    _build_friction_cone_wrenches,
    force_closure_metrics,
    normal_balance,
    q1_distance_convex_hull,
    wrench_spread,
)


def test_contact_wrench_normalizes():
    cw = ContactWrench(point=[0, 0, 0], normal=[2.0, 0.0, 0.0], finger_idx=0)
    np.testing.assert_allclose(cw.normal, [1.0, 0.0, 0.0])


def test_normal_balance_opposed_pair_is_zero():
    contacts = [
        ContactWrench([1, 0, 0], [-1, 0, 0], 0),
        ContactWrench([-1, 0, 0], [1, 0, 0], 1),
    ]
    assert normal_balance(contacts) == pytest.approx(0.0, abs=1e-9)


def test_normal_balance_aligned_pair_is_two():
    contacts = [
        ContactWrench([0, 0, 0], [1, 0, 0], 0),
        ContactWrench([1, 0, 0], [1, 0, 0], 1),
    ]
    assert normal_balance(contacts) == pytest.approx(2.0)


def test_wrench_spread_collinear_is_smaller_than_orthogonal():
    com = np.zeros(3)
    collinear = [
        ContactWrench([1, 0, 0], [-1, 0, 0], 0),
        ContactWrench([-1, 0, 0], [1, 0, 0], 1),
    ]
    orthogonal = [
        ContactWrench([1, 0, 0], [-1, 0, 0], 0),
        ContactWrench([0, 1, 0], [0, -1, 0], 1),
        ContactWrench([0, 0, 1], [0, 0, -1], 2),
    ]
    W_col = np.stack(
        [np.concatenate([c.normal, np.cross(c.point - com, c.normal)]) for c in collinear],
        axis=1,
    )
    W_orth = np.stack(
        [np.concatenate([c.normal, np.cross(c.point - com, c.normal)]) for c in orthogonal],
        axis=1,
    )
    assert wrench_spread(W_orth) > wrench_spread(W_col)


def test_q1_distance_single_contact_is_norm():
    W = np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    assert q1_distance_convex_hull(W) == pytest.approx(1.0, abs=1e-6)


def test_q1_distance_opposed_pair_reaches_zero_via_convex_hull():
    W = np.array(
        [
            [1.0, -1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    # convex hull of (1,0..) and (-1,0..) contains origin → distance 0
    assert q1_distance_convex_hull(W) == pytest.approx(0.0, abs=1e-6)


def test_q1_distance_misaligned_pair_is_positive():
    W = np.array(
        [
            [1.0, 1.0],
            [0.0, 0.1],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    assert q1_distance_convex_hull(W) > 0.5


def test_friction_cone_wrenches_shape():
    contacts = [
        ContactWrench([1, 0, 0], [-1, 0, 0], 0),
        ContactWrench([-1, 0, 0], [1, 0, 0], 1),
    ]
    W = _build_friction_cone_wrenches(contacts, np.zeros(3), mu=0.3, n_edges=4)
    assert W.shape == (6, 8)


def test_force_closure_metrics_three_finger_wrap_better_than_two_aligned():
    com = np.zeros(3)
    bad = [
        ContactWrench([1, 0, 0], [1, 0, 0], 0),
        ContactWrench([1.05, 0, 0], [1, 0, 0], 1),
    ]
    good = [
        ContactWrench([1, 0, 0], [-1, 0, 0], 0),
        ContactWrench([-0.5, 0.87, 0], [0.5, -0.87, 0], 1),
        ContactWrench([-0.5, -0.87, 0], [0.5, 0.87, 0], 2),
    ]
    m_bad = force_closure_metrics(bad, com)
    m_good = force_closure_metrics(good, com)
    assert m_good.score > m_bad.score
    assert m_good.q1_distance < m_bad.q1_distance
    assert m_good.normal_balance < m_bad.normal_balance + 1e-6
    assert m_good.fingers_engaged == 3
    assert m_bad.fingers_engaged == 2


def test_force_closure_metrics_handles_empty():
    m = force_closure_metrics([], np.zeros(3))
    assert m.n_contacts == 0
    assert m.score == -float("inf")
