"""The scheduled finger anchor: grasp set-point through the swing, hold set-point once aligned.

Why it exists: the residual is bounded, so the SET-POINT decides what the policy can reach, and on
the opposed hand the grasp pose and the three-finger hold are 1.296 rad apart at `thumb_pip`.
Every time-invariant way of spanning that gap failed — uniform residual 1.5 NaN'd at iteration 0,
a statically re-centred thumb parked in the floor at 34.8 N, and an asymmetric budget let the
thumb wreck the grasp (drop at step 78 against 485). They fail for one reason: the thumb's useful
pose and its harmful poses are the same neighbourhood, separated by WHEN.

The blend arithmetic is exercised directly rather than through mjlab so it runs without a GPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

GRASP = torch.tensor([0.333, 2.019, -1.281, 0.024, 1.670, -1.033, -0.120, 1.358, -0.647])
HOLD = torch.tensor([-0.013, 1.460, 0.016, 0.107, 1.554, -0.739, -0.202, 1.142, 0.008])


def _step(switched, cos, thresh, blend_steps):
    """One sim step of the latch-and-blend, lifted out of LerpFingerAction.apply_actions."""
    up = (cos >= thresh).float()
    switched = torch.clamp(switched + up / max(1.0, float(blend_steps)), max=1.0)
    beta = switched.unsqueeze(-1)
    offset = (1.0 - beta) * GRASP.unsqueeze(0) + beta * HOLD.unsqueeze(0)
    return switched, offset


def test_anchor_stays_on_the_grasp_pose_while_the_shaft_is_still_turning():
    """Paying out the hold pose during the swing is what parks the thumb in the floor."""
    sw = torch.zeros(1)
    for _ in range(200):
        sw, offset = _step(sw, torch.tensor([0.2]), 0.7, 60)
    assert torch.allclose(offset[0], GRASP)


def test_anchor_arrives_at_the_hold_pose_after_the_blend():
    sw = torch.zeros(1)
    for _ in range(60):
        sw, offset = _step(sw, torch.tensor([0.95]), 0.7, 60)
    assert torch.allclose(offset[0], HOLD, atol=1e-6)


def test_the_switch_latches_so_a_wobble_does_not_yank_the_hand_back():
    """Un-latching would step every finger target at once — the discontinuity the single-stage
    recipe exists to avoid, and the one that ejects the shaft at the A->B seam."""
    sw = torch.zeros(1)
    for _ in range(60):
        sw, _ = _step(sw, torch.tensor([0.95]), 0.7, 60)
    for _ in range(120):                      # shaft wobbles back below the gate
        sw, offset = _step(sw, torch.tensor([0.1]), 0.7, 60)
    assert torch.allclose(offset[0], HOLD, atol=1e-6)


def test_the_move_is_gradual_not_a_step_change():
    sw = torch.zeros(1)
    seen = []
    for _ in range(60):
        sw, offset = _step(sw, torch.tensor([0.95]), 0.7, 60)
        seen.append(float(offset[0, 2]))       # thumb_pip, the joint that travels 1.296 rad
    jumps = [abs(b - a) for a, b in zip(seen, seen[1:])]
    assert max(jumps) < 0.05, "each step must move thumb_pip a fraction of its 1.296 rad travel"
    assert seen[-1] > seen[0], "and it must actually get there"


def test_per_env_latching_is_independent():
    """Envs reorient at different steps; one env crossing the gate must not move another's."""
    sw = torch.zeros(2)
    for _ in range(60):
        sw, offset = _step(sw, torch.tensor([0.95, 0.10]), 0.7, 60)
    assert torch.allclose(offset[0], HOLD, atol=1e-6)
    assert torch.allclose(offset[1], GRASP)
