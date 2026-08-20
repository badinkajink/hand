"""Per-joint residual scale on the LerpFinger action.

Why it exists: the opposed hand's three-finger hold sits 1.296 rad from the closed set-point at
`thumb_pip` against a ±0.5 rad residual, so the thumb cannot be commanded there at all — but
raising the scale to 1.5 for every joint NaN'd the scene at iteration 0 on both noise levels
tried, because it also hands that authority to the opposed pair during the gravity swing the
recipe is built to keep gentle. The thumb reads 0 N through the lift and the swing, so it can
have the extra range on its own.

These tests exercise the arithmetic directly rather than through mjlab, so they run without a
GPU: the property that matters is that each joint gets its own multiplier and that a wrong-length
vector is rejected loudly rather than silently broadcast.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _apply(raw, scale, offset):
    """The line under test, lifted out of LerpFingerAction.apply_actions."""
    if not isinstance(scale, (int, float)):
        scale = torch.as_tensor(scale, dtype=torch.float32)
    return raw * scale + offset


def test_scalar_scale_is_unchanged_behaviour():
    raw = torch.ones(2, 9)
    offset = torch.zeros(2, 9)
    out = _apply(raw, 0.5, offset)
    assert torch.allclose(out, torch.full((2, 9), 0.5))


def test_per_joint_scale_gives_each_joint_its_own_authority():
    """thumb 1.5 / index 0.5 / middle 0.7, in joint_names order."""
    scale = [1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7]
    out = _apply(torch.ones(1, 9), scale, torch.zeros(1, 9))[0]
    assert out[:3].tolist() == [1.5, 1.5, 1.5]
    assert out[3:6].tolist() == [0.5, 0.5, 0.5]
    assert out[6:].tolist() == pytest.approx([0.7, 0.7, 0.7])


def test_thumb_can_reach_the_hold_the_pair_scale_could_not():
    """The concrete number this was built for: 1.296 rad at thumb_pip."""
    scale = [1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7]
    reach = _apply(torch.ones(1, 9), scale, torch.zeros(1, 9))[0]
    assert reach[2] >= 1.296, "thumb_pip must be able to reach the measured hold excursion"
    assert reach[8] >= 0.654, "middle_pip must be able to reach its measured excursion"
    assert reach[3] == pytest.approx(0.5), "the opposed pair must keep the swing's proven budget"


def test_wrong_length_vector_is_rejected():
    """Silent broadcasting would hand the opposed pair the thumb's authority — the exact
    configuration that NaN'd the scene — so a mismatched vector must raise, not fan out."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from morphohand.rl.actions import resolve_residual_scale

    assert resolve_residual_scale(0.5, 9) == 0.5
    assert len(resolve_residual_scale([1.5] * 9, 9)) == 9
    with pytest.raises(ValueError, match="one scalar or one value per joint"):
        resolve_residual_scale([1.5, 0.5], 9)
