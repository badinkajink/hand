# pyright: reportMissingImports=false

import numpy as np
import pytest

from morphohand.optimization.contact_targets import (
    ContactTarget,
    ContactTargetSet,
    score_contact_targets,
    world_targets,
)


def _identity_pose():
    return np.zeros(3), np.eye(3)


def test_contact_target_normalizes_normal():
    t = ContactTarget(name="a", local_pos=[0, 0, 0], local_normal=[3.0, 0.0, 0.0])
    np.testing.assert_allclose(t.local_normal, [1.0, 0.0, 0.0])


def test_contact_target_rejects_zero_normal_and_bad_finger():
    with pytest.raises(ValueError):
        ContactTarget(name="a", local_pos=[0, 0, 0], local_normal=[0, 0, 0])
    with pytest.raises(ValueError):
        ContactTarget(name="a", local_pos=[0, 0, 0], finger="pinky")
    with pytest.raises(ValueError):
        ContactTarget(name="a", local_pos=[0, 0, 0], radius=-0.01)


def test_world_targets_translates_and_rotates():
    target_set = ContactTargetSet(
        object_body="obj",
        patches=[
            ContactTarget(name="a", local_pos=[0.1, 0.0, 0.0]),
            ContactTarget(name="b", local_pos=[0.0, 0.1, 0.0]),
        ],
    )
    pos = np.array([1.0, 2.0, 3.0])
    # rotate 90deg about Z: x -> y, y -> -x
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    world_pos, _ = world_targets(target_set, pos, rot)
    np.testing.assert_allclose(world_pos[0], [1.0, 2.1, 3.0], atol=1e-9)
    np.testing.assert_allclose(world_pos[1], [0.9, 2.0, 3.0], atol=1e-9)


def test_score_fixed_assignment_perfect_match_gives_one_per_patch():
    target_set = ContactTargetSet(
        object_body="obj",
        patches=[
            ContactTarget(name="a", local_pos=[0.0, 0.0, 0.0], finger="thumb"),
            ContactTarget(name="b", local_pos=[0.05, 0.0, 0.0], finger="index"),
            ContactTarget(name="c", local_pos=[0.0, 0.05, 0.0], finger="middle"),
        ],
    )
    obj_pos, obj_rot = _identity_pose()
    tips = np.array([
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.0, 0.05, 0.0],
    ])
    b = score_contact_targets(target_set, tips, obj_pos, obj_rot)
    assert b.assignment == [0, 1, 2]
    assert b.total_reward == pytest.approx(3.0, abs=1e-9)
    assert b.mean_distance == pytest.approx(0.0, abs=1e-9)


def test_score_reward_decays_with_distance():
    target_set = ContactTargetSet(
        object_body="obj",
        patches=[ContactTarget(name="a", local_pos=[0, 0, 0], radius=0.01, finger="thumb")],
    )
    obj_pos, obj_rot = _identity_pose()

    def reward_at(d):
        tips = np.array([[d, 0, 0], [10, 0, 0], [10, 1, 0]])
        return score_contact_targets(target_set, tips, obj_pos, obj_rot).total_reward

    assert reward_at(0.0) == pytest.approx(1.0)
    assert reward_at(0.005) == pytest.approx(1.0)
    assert 0.0 < reward_at(0.02) < 1.0
    assert reward_at(0.05) == pytest.approx(0.0)


def test_score_auto_assigns_unassigned_patches():
    target_set = ContactTargetSet(
        object_body="obj",
        patches=[
            ContactTarget(name="a", local_pos=[0.1, 0.0, 0.0]),
            ContactTarget(name="b", local_pos=[0.0, 0.1, 0.0]),
        ],
    )
    obj_pos, obj_rot = _identity_pose()
    # thumb closest to b, index closest to a, middle far
    tips = np.array([
        [0.0, 0.11, 0.0],
        [0.11, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    b = score_contact_targets(target_set, tips, obj_pos, obj_rot)
    # patch a should pair with index (tip 1); patch b with thumb (tip 0)
    assert b.assignment == [1, 0]


def test_score_respects_object_rotation():
    target_set = ContactTargetSet(
        object_body="obj",
        patches=[ContactTarget(name="a", local_pos=[0.1, 0.0, 0.0], radius=0.005, finger="thumb")],
    )
    # 180deg rot about Z moves the patch from x=+0.1 to x=-0.1
    rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
    tips_aligned_pre = np.array([[0.1, 0, 0], [10, 0, 0], [10, 1, 0]])
    tips_aligned_post = np.array([[-0.1, 0, 0], [10, 0, 0], [10, 1, 0]])
    obj_pos = np.zeros(3)
    pre = score_contact_targets(target_set, tips_aligned_pre, obj_pos, np.eye(3))
    post_aligned = score_contact_targets(target_set, tips_aligned_post, obj_pos, rot)
    post_misaligned = score_contact_targets(target_set, tips_aligned_pre, obj_pos, rot)
    assert pre.total_reward == pytest.approx(1.0)
    assert post_aligned.total_reward == pytest.approx(1.0)
    assert post_misaligned.total_reward == pytest.approx(0.0, abs=1e-6)


def test_empty_target_set_returns_zero():
    target_set = ContactTargetSet(object_body="obj", patches=[])
    tips = np.zeros((3, 3))
    b = score_contact_targets(target_set, tips, np.zeros(3), np.eye(3))
    assert b.total_reward == 0.0
    assert b.mean_distance == 0.0


def test_from_dict_parses_yaml_shaped_payload():
    raw = {
        "object_body": "power_drill",
        "patches": [
            {"name": "trigger", "finger": "index", "local_pos": [0.02, -0.01, 0.04], "radius": 0.012},
            {"name": "barrel", "finger": "thumb", "local_pos": [-0.02, 0.0, 0.05]},
        ],
    }
    ts = ContactTargetSet.from_dict(raw)
    assert ts.object_body == "power_drill"
    assert len(ts.patches) == 2
    assert ts.patches[0].finger == "index"
    assert ts.patches[1].radius == pytest.approx(0.012)
