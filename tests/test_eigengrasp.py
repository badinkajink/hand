# pyright: reportMissingImports=false

import numpy as np
import pytest

from morphohand.optimization.eigengrasp import (
    FINGER_CTRL_DIM,
    SynergyBasis,
    bounds_in_subspace,
    fit_synergy_basis,
    hand_designed_basis,
)


def test_hand_designed_basis_unit_norm():
    basis = hand_designed_basis()
    assert basis.components.shape == (3, FINGER_CTRL_DIM)
    norms = np.linalg.norm(basis.components, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-9)


def test_hand_designed_basis_roundtrip_preserves_subspace_components():
    basis = hand_designed_basis()
    coeffs = np.array([0.5, -0.3, 0.2])
    full = basis.reconstruct(coeffs)
    recovered = basis.project(full)
    np.testing.assert_allclose(recovered, coeffs, atol=1e-9)


def test_fit_synergy_basis_recovers_dominant_direction():
    rng = np.random.default_rng(0)
    true_dir = np.zeros(FINGER_CTRL_DIM)
    true_dir[[1, 2, 4, 5, 7, 8]] = 1.0
    true_dir /= np.linalg.norm(true_dir)
    coeffs = rng.normal(0, 1.0, size=200)
    noise = rng.normal(0, 0.01, size=(200, FINGER_CTRL_DIM))
    samples = np.outer(coeffs, true_dir) + noise

    basis = fit_synergy_basis(samples, n_components=2)
    leading = basis.components[0]
    overlap = float(np.abs(np.dot(leading, true_dir)))
    assert overlap > 0.98, f"leading component should align with planted direction (got overlap={overlap})"
    assert basis.explained_variance_ratio is not None
    assert basis.explained_variance_ratio[0] > 0.95


def test_fit_synergy_basis_validation():
    with pytest.raises(ValueError):
        fit_synergy_basis(np.zeros((1, FINGER_CTRL_DIM)), n_components=1)
    with pytest.raises(ValueError):
        fit_synergy_basis(np.zeros((10, FINGER_CTRL_DIM)), n_components=FINGER_CTRL_DIM + 1)


def test_bounds_in_subspace_brackets_zero():
    basis = hand_designed_basis()
    lo_full = -np.ones(FINGER_CTRL_DIM)
    hi_full = np.ones(FINGER_CTRL_DIM)
    lo, hi = bounds_in_subspace(basis, lo_full, hi_full, n_samples=1024, seed=1)
    assert np.all(lo < 0) and np.all(hi > 0)
    assert np.all(hi > lo)


def test_synergy_basis_project_reconstruct_low_rank():
    basis = hand_designed_basis()
    full = basis.mean + 0.7 * basis.components[0] - 0.2 * basis.components[1]
    recon = basis.project_then_reconstruct(full)
    np.testing.assert_allclose(recon, full, atol=1e-9)
