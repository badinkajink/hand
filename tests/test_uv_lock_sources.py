"""Lightweight sanity tests on `pyproject.toml` so the uv migration
doesn't silently regress.

Run via `pytest tests/test_uv_lock_sources.py`. No network, no env activation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import tomllib  # py3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def proj() -> dict:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def test_pyproject_exists(proj):
    assert proj["project"]["name"] == "morphohand"


def test_uv_managed(proj):
    assert proj.get("tool", {}).get("uv", {}).get("managed") is True


def test_required_extras_present(proj):
    extras = proj["project"].get("optional-dependencies", {})
    for name in ("gpu", "rl", "dev"):
        assert name in extras, f"missing [project.optional-dependencies].{name}"


def test_editable_sources_point_to_external(proj):
    sources = proj.get("tool", {}).get("uv", {}).get("sources", {})
    for pkg in ("mujoco-warp", "comfree_warp"):
        assert pkg in sources, f"missing source for {pkg}"
        spec = sources[pkg]
        assert spec.get("editable") is True, f"{pkg} should be editable"
        path = ROOT / spec["path"]
        assert path.exists(), f"editable source path {path} does not exist"
        assert (path / "pyproject.toml").exists(), f"no pyproject under {path}"


def test_torch_routed_through_cu128_index(proj):
    sources = proj.get("tool", {}).get("uv", {}).get("sources", {})
    assert "torch" in sources, "torch should be sourced from pytorch-cu128 index"
    assert sources["torch"].get("index") == "pytorch-cu128"
    indices = proj.get("tool", {}).get("uv", {}).get("index", [])
    assert any(i.get("name") == "pytorch-cu128" for i in indices), (
        "pytorch-cu128 index must be declared"
    )


def test_mjlab_pin_supports_mujoco_3_6(proj):
    """mjlab 1.3+ requires mujoco>=3.7 which conflicts with our external mujoco_warp 3.6.
    Pin to <1.3 until external is bumped."""
    rl = proj["project"]["optional-dependencies"]["rl"]
    mjlab_dep = next((d for d in rl if d.startswith("mjlab")), None)
    assert mjlab_dep is not None, "mjlab missing from [rl]"
    # Crude check: should pin <1.3 (e.g. ==1.2.0 or >=1.0,<1.3).
    assert "1.2" in mjlab_dep or "<1.3" in mjlab_dep, (
        f"mjlab pin '{mjlab_dep}' may not be compatible with mujoco 3.6"
    )
