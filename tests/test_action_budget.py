"""The action-budget pre-flight, which exists because three runs were misread without it.

Each of r7, r8 and the r8 smoke paid a reward for a hand configuration the policy could not
command, read `0.0000` for the whole run, and was interpreted as PPO declining the behaviour.
The two properties pinned here are the two that were actually wrong in practice: the check has to
FAIL when a single joint overflows (not when the average does), and it has to read the sustained
hold rather than the approach, which is 4x closer and passes.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_action_budget.py"
SCENE = ROOT / "results/phase1/perp_thumb_engage/sp25_manual/frozen_scene.xml"

pytestmark = pytest.mark.skipif(
    not SCENE.exists(), reason="sp25 opposed frozen scene not present")


def _budget(demo: Path, scale: float, last: int = 0) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT), "--scene", str(SCENE),
           "--closed-keyframe", "closed_manual", "--demo-npz", str(demo),
           "--residual-scale", str(scale)]
    if last:
        cmd += ["--last", str(last)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    """A two-phase rollout: an approach that stays near the set-point, then a hold far from it.

    Built rather than recorded so the test does not depend on a 4000-step MuJoCo run, but shaped
    like the real one — the excursion that matters only appears in the second half.
    """
    import mujoco
    sys.path.insert(0, str(ROOT / "src"))
    from morphohand.rl.deploy import finger_ctrl_from_keyframe

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    centre = np.asarray(finger_ctrl_from_keyframe(SCENE, "closed_manual")).reshape(-1)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("closed_manual").id)
    adr = [model.jnt_qposadr[model.joint(j).id] for j in
           ("thumb_yaw", "thumb_mcp", "thumb_pip", "index_yaw", "index_mcp", "index_pip",
            "middle_yaw", "middle_mcp", "middle_pip")]

    q = np.tile(data.qpos.copy(), (200, 1))
    q[:100, adr] = centre + 0.2                 # approach: comfortably inside any sane budget
    q[100:, adr] = centre + 0.2
    q[100:, adr[2]] = centre[2] + 1.3           # hold: thumb_pip 1.3 rad out
    p = tmp_path_factory.mktemp("budget") / "demo.npz"
    np.savez(p, qpos=q)
    return p


def test_fails_when_one_joint_overflows(demo):
    out = _budget(demo, scale=0.5, last=100)
    assert out.returncode == 1, out.stdout
    assert "UNREACHABLE" in out.stdout
    assert "thumb_pip" in out.stdout


def test_passes_once_the_budget_covers_the_worst_joint(demo):
    out = _budget(demo, scale=1.5, last=100)
    assert out.returncode == 0, out.stdout
    assert "REACHABLE" in out.stdout


def test_reading_the_approach_instead_of_the_hold_hides_it(demo):
    """The exact mistake that produced the wrong reachability claim: averaged over the whole
    rollout, or read from first contact, the overflow is diluted and the check passes."""
    import re

    def needed(out: str) -> float:
        return float(re.search(r"demonstration is ([0-9.]+) rad", out).group(1))

    hold = needed(_budget(demo, scale=0.5, last=100).stdout)
    diluted = needed(_budget(demo, scale=0.5).stdout)         # whole rollout, approach included
    assert hold == pytest.approx(1.30, abs=0.01)
    assert diluted < hold * 0.75, (
        "averaging the approach into the hold must UNDERSTATE the excursion — that is the trap, "
        f"hold needs {hold} rad, the whole-rollout mean asks for only {diluted}")


def test_reports_the_scale_that_would_work(demo):
    out = _budget(demo, scale=0.5, last=100)
    assert "1.30" in out.stdout
