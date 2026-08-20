"""`chuck_pose_match` — the reward that states the hold as a whole-hand configuration.

The term exists because a thumb-only brace reward asks for a motion whose immediate consequence
is a drop: on the opposed pair, a thumb press against fingers still at +-90 deg ejects the shaft
(six scripted engages, six ejections), so PPO refusing it is correct, and `thumb_brace_force`
read 0.0000 for the whole of r7. These tests pin the two properties that make the replacement
different: it saturates only when ALL THREE fingertips are in place, and it is silent until the
shaft is actually reoriented.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from morphohand.rl.terms_reward import _CHUCK_REF, chuck_pose_match  # noqa: E402


class _Env:
    device = "cpu"


def _patch(monkeypatch, tips: np.ndarray, cos: float) -> None:
    """Stand in for the two model-backed helpers so the term's own math is what is measured."""
    fake = types.ModuleType("morphohand.rl.imitation")
    fake.fingertips_in_object_frame = lambda env, name="cube", *a, **k: torch.as_tensor(
        tips, dtype=torch.float32).unsqueeze(0)
    monkeypatch.setitem(sys.modules, "morphohand.rl.imitation", fake)
    monkeypatch.setattr("morphohand.rl.terms_reward._alignment_cos",
                        lambda *a, **k: torch.tensor([cos]))


@pytest.fixture
def ref_npz(tmp_path):
    _CHUCK_REF.clear()
    pose = np.array([[-0.011, 0.000, 0.005],      # thumb
                     [0.006, 0.010, 0.006],       # index
                     [0.006, -0.010, 0.006]])     # middle
    p = tmp_path / "chuck_pose.npz"
    np.savez(p, hold_pose=pose)
    return p, pose


def test_saturates_when_the_hand_is_in_the_recorded_configuration(monkeypatch, ref_npz):
    p, pose = ref_npz
    _patch(monkeypatch, pose, cos=0.95)
    assert float(chuck_pose_match(_Env(), ref_npz=str(p))) == pytest.approx(1.0, abs=1e-5)


def test_silent_until_the_shaft_is_reoriented(monkeypatch, ref_npz):
    """Paying for the hold pose during the swing would buy a clamp, and the swing needs a
    loose pinch — the gate is what keeps this term from destroying the rotation."""
    p, pose = ref_npz
    _patch(monkeypatch, pose, cos=0.3)
    assert float(chuck_pose_match(_Env(), ref_npz=str(p), align_thresh=0.7)) == 0.0


def test_one_idle_finger_cannot_collect_it(monkeypatch, ref_npz):
    """The r7 failure mode in reward form: two fingers right, the thumb parked where it starts.
    A term that reduced over the best contacts would still pay here; this one must not."""
    p, pose = ref_npz
    both_right = pose.copy()
    _patch(monkeypatch, both_right, cos=0.95)
    full = float(chuck_pose_match(_Env(), ref_npz=str(p)))

    thumb_parked = pose.copy()
    thumb_parked[0] += np.array([-0.03, 0.02, 0.0])      # ~36 mm away, a stowed thumb
    _patch(monkeypatch, thumb_parked, cos=0.95)
    partial = float(chuck_pose_match(_Env(), ref_npz=str(p)))
    assert partial < 0.25 * full


def test_decays_smoothly_so_there_is_a_gradient_to_follow(monkeypatch, ref_npz):
    p, pose = ref_npz
    out = []
    for d in (0.000, 0.002, 0.005, 0.010):
        _patch(monkeypatch, pose + d, cos=0.95)
        out.append(float(chuck_pose_match(_Env(), ref_npz=str(p))))
    assert out == sorted(out, reverse=True)
    assert out[1] > 0.1, "a 2 mm error must still be worth chasing, not already flat"
