"""Coverage and provenance checks for the hardware-configurable real-v1 sampler."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "real_v1_design_search", ROOT / "scripts/real_v1_design_search.py")
SEARCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SEARCH)

DEPLOY_SPEC = importlib.util.spec_from_file_location(
    "real_v1_deploy_envelope", ROOT / "scripts/real_v1_deploy_envelope.py")
DEPLOY = importlib.util.module_from_spec(DEPLOY_SPEC)
assert DEPLOY_SPEC.loader is not None
DEPLOY_SPEC.loader.exec_module(DEPLOY)


def test_sobol_population_is_prefix_stable_and_inside_real_workspace():
    designs_128, metadata_128 = SEARCH.sobol_designs(128, seed=20260830)
    designs_64, metadata_64 = SEARCH.sobol_designs(64, seed=20260830)

    assert len(designs_128) == 128
    for tag_64, vector_64 in designs_64.items():
        tag_128 = next(tag for tag, meta in metadata_128.items()
                       if meta["sobol_index"] == metadata_64[tag_64]["sobol_index"])
        assert vector_64 == designs_128[tag_128]

    workspace = SEARCH.REAL_V1_WORKSPACE
    bounds = (
        (workspace.thumb.x_min, workspace.thumb.x_max),
        (workspace.thumb.y_min, workspace.thumb.y_max), (0.0, 0.0),
        (workspace.index.x_min, workspace.index.x_max),
        (workspace.index.y_min, workspace.index.y_max), (0.0, 0.0),
        (workspace.middle.x_min, workspace.middle.x_max),
        (workspace.middle.y_min, workspace.middle.y_max), (0.0, 0.0),
    )
    for vector in designs_128.values():
        assert all(lo <= value <= hi for value, (lo, hi) in zip(vector, bounds))


def test_wide_stratum_only_moves_pair_x_outward():
    _, metadata = SEARCH.sobol_designs(128, seed=20260830,
                                       wide_fraction=0.25, wide_power=0.75)
    wide = [meta for meta in metadata.values() if meta["source"] == "sobol_wide_bias"]
    uniform = [meta for meta in metadata.values() if meta["source"] == "sobol_uniform"]

    assert 28 <= len(wide) <= 36
    assert len(wide) + len(uniform) == 128
    for meta in wide:
        raw = np.asarray(meta["sobol_u_raw"])
        used = np.asarray(meta["sobol_u_used"])
        assert used[2] >= raw[2]
        assert used[4] >= raw[4]
        assert np.array_equal(used[[0, 1, 3, 5]], raw[[0, 1, 3, 5]])
    for meta in uniform:
        assert meta["sobol_u_used"] == meta["sobol_u_raw"]


def test_sobol_set_keeps_known_anchors_including_deployment_baseline():
    population = SEARCH.design_set("sobol", sobol_count=128)
    assert len(population) == 134
    assert set(SEARCH.known_designs()).issubset(population)
    assert "g12" in population


def test_scs0009_napkin_torque_envelope_matches_published_ratings():
    assert np.isclose(DEPLOY.SCS0009_STALL_TORQUE_NM, 0.22555295)
    assert np.isclose(DEPLOY.SCS0009_RATED_TORQUE_NM, 0.06864655)
    assert np.isclose(DEPLOY.SCS0009_OVERLOAD_TORQUE_NM,
                      0.8 * DEPLOY.SCS0009_STALL_TORQUE_NM)
    assert np.isclose(DEPLOY.SCS0009_PROTECTIVE_TORQUE_NM,
                      0.2 * DEPLOY.SCS0009_STALL_TORQUE_NM)
