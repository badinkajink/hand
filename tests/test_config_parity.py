"""The launcher pre-flight must catch real drift and must NOT abort on new config fields.

Both halves are load-bearing and both have failed in production. The check exists because four
experiments died to omitted launcher flags; it then cost an idle GPU night by aborting two runs
25 s in over `friction_dr_scale`, a field added after the reference ran, sitting at its own
default, with `friction_dr: False` beside it. A parity check that cries wolf gets `--allow`d
reflexively, which is the same as not having one.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "assert_config_parity.py"


def _run_dir(tmp_path: Path, name: str, env: dict, ppo: dict | None = None) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump({"env": env, "ppo": ppo or {}}))
    return d


def _parity(run: Path, ref: Path, *allow: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT), "--run", str(run), "--reference", str(ref), "--wait", "1"]
    for a in allow:
        cmd += ["--allow", a]
    return subprocess.run(cmd, capture_output=True, text=True)


BASE = {"num_envs": 3072, "lift_delta_z": 0.14, "open_finger_from_keyframe": True}


def test_identical_configs_pass(tmp_path):
    ref = _run_dir(tmp_path, "ref", BASE)
    new = _run_dir(tmp_path, "new", BASE)
    assert _parity(new, ref).returncode == 0


def test_the_flag_drift_it_exists_to_catch(tmp_path):
    """The r5 failure: --lift-delta-z omitted, trained at 0.05 against the reference's 0.14."""
    ref = _run_dir(tmp_path, "ref", BASE)
    new = _run_dir(tmp_path, "new", {**BASE, "lift_delta_z": 0.05})
    out = _parity(new, ref)
    assert out.returncode == 1
    assert "lift_delta_z" in out.stdout


def test_new_field_at_its_dataclass_default_is_not_divergence(tmp_path):
    """`friction_dr_scale` is a tuple, so the old falsy-value rule aborted on it."""
    ref = _run_dir(tmp_path, "ref", BASE)
    new = _run_dir(tmp_path, "new", {**BASE, "friction_dr": False,
                                     "friction_dr_scale": [0.55, 1.15]})
    out = _parity(new, ref)
    assert out.returncode == 0, out.stdout
    assert "new field" in out.stdout


def test_new_field_MOVED_off_its_default_is_still_divergence(tmp_path):
    """Turning a feature on must break parity even though the reference predates the field."""
    ref = _run_dir(tmp_path, "ref", BASE)
    new = _run_dir(tmp_path, "new", {**BASE, "friction_dr": True,
                                     "friction_dr_scale": [0.2, 3.0]})
    out = _parity(new, ref)
    assert out.returncode == 1
    assert "friction_dr" in out.stdout


def test_allow_makes_an_intended_delta_pass(tmp_path):
    ref = _run_dir(tmp_path, "ref", BASE)
    new = _run_dir(tmp_path, "new", {**BASE, "thumb_brace_weight": 8.0})
    assert _parity(new, ref, "env.thumb_brace_weight").returncode == 0
