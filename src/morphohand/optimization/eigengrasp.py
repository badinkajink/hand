"""Synergy (eigengrasp) subspace for the 9D finger control vector.

The 9D control (yaw, mcp, pip) x 3 fingers has strong inter-joint correlations
during grasping. Following Ciocarlie & Allen (RSS 2007), we project to a small
basis that captures the principal axes of variation in collected grasps.

This module provides:
- ``SynergyBasis``: mean + components, with ``project`` / ``reconstruct``;
- ``fit_synergy_basis_from_csvs``: PCA over historical CEM candidates;
- ``hand_designed_basis``: a 3D fallback (open/close, lateral spread, thumb
  opposition) usable when no candidate data exists yet.

The CEM strategy in ``phase1_strategy_synergy_cem`` consumes any
``SynergyBasis`` and runs Gaussian search in the K-dim coefficient space,
projecting back to 9D for evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FINGER_CTRL_DIM = 9


@dataclass
class SynergyBasis:
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray | None = None

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    @property
    def n_full(self) -> int:
        return int(self.components.shape[1])

    def project(self, full_ctrl: np.ndarray) -> np.ndarray:
        centered = np.asarray(full_ctrl, dtype=np.float64) - self.mean
        return centered @ self.components.T

    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        return self.mean + np.asarray(coeffs, dtype=np.float64) @ self.components

    def project_then_reconstruct(self, full_ctrl: np.ndarray) -> np.ndarray:
        return self.reconstruct(self.project(full_ctrl))


def fit_synergy_basis(
    samples: np.ndarray,
    n_components: int,
) -> SynergyBasis:
    """Fit a PCA basis on (N, D) finger-control samples."""
    X = np.asarray(samples, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"samples must be (N, D); got shape {X.shape}")
    n, d = X.shape
    if n_components < 1 or n_components > d:
        raise ValueError(f"n_components must be in [1, {d}]; got {n_components}")
    if n < 2:
        raise ValueError("Need at least 2 samples to fit a basis")

    mean = X.mean(axis=0)
    Xc = X - mean
    # SVD: Xc = U S Vt, components are rows of Vt
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    components = vt[:n_components]
    variances = (s ** 2) / max(1, n - 1)
    total = float(variances.sum()) if variances.sum() > 0 else 1.0
    explained = (variances[:n_components] / total).astype(np.float64)
    return SynergyBasis(mean=mean, components=components, explained_variance_ratio=explained)


def _load_ctrls_from_candidate_csv(csv_path: Path) -> np.ndarray:
    """Extract per-task finger control vectors from a multitask candidate CSV."""
    import csv as _csv

    out: list[list[float]] = []
    with csv_path.open("r") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            tc_raw = row.get("task_ctrl_json")
            if not tc_raw:
                continue
            try:
                tc = json.loads(tc_raw)
            except json.JSONDecodeError:
                continue
            for vec in tc.values():
                if isinstance(vec, list) and len(vec) == FINGER_CTRL_DIM:
                    try:
                        out.append([float(v) for v in vec])
                    except (TypeError, ValueError):
                        continue
    if not out:
        raise ValueError(f"No finger control vectors found in {csv_path}")
    return np.asarray(out, dtype=np.float64)


def fit_synergy_basis_from_csvs(
    csv_paths: list[Path],
    n_components: int = 3,
) -> SynergyBasis:
    """Aggregate finger control vectors from multitask CSVs and fit a basis."""
    chunks: list[np.ndarray] = []
    for p in csv_paths:
        try:
            chunks.append(_load_ctrls_from_candidate_csv(Path(p)))
        except ValueError:
            continue
    if not chunks:
        raise ValueError("No usable candidate CSVs provided")
    samples = np.concatenate(chunks, axis=0)
    return fit_synergy_basis(samples, n_components=n_components)


def _gram_schmidt(rows: np.ndarray) -> np.ndarray:
    """Orthonormalize rows of `rows` via classical Gram-Schmidt."""
    out = np.zeros_like(rows)
    for i, v in enumerate(rows):
        w = v.copy()
        for j in range(i):
            w = w - np.dot(w, out[j]) * out[j]
        n = np.linalg.norm(w)
        if n < 1e-12:
            raise ValueError("hand-designed basis vectors are linearly dependent")
        out[i] = w / n
    return out


def hand_designed_basis() -> SynergyBasis:
    """A small interpretable basis usable before any data exists.

    Conceptual axes (orthonormalized via Gram-Schmidt for projection sanity):
      0: flex-all-fingers (mcp and pip together across all three fingers)
      1: thumb-vs-others opposition (thumb flex vs index+middle flex)
      2: lateral spread (thumb yaw open vs index+middle yaw narrow)

    Joint layout (per evaluator): [t_yaw, t_mcp, t_pip, i_yaw, i_mcp, i_pip,
    m_yaw, m_mcp, m_pip].
    """
    flex_all = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=np.float64)
    thumb_opp = np.array([0, 1, 1, 0, -1, -1, 0, -1, -1], dtype=np.float64)
    spread = np.array([1, 0, 0, -1, 0, 0, -1, 0, 0], dtype=np.float64)
    raw = np.stack([flex_all, thumb_opp, spread], axis=0)
    comps = _gram_schmidt(raw)
    mean = np.zeros(FINGER_CTRL_DIM, dtype=np.float64)
    return SynergyBasis(
        mean=mean,
        components=comps,
        explained_variance_ratio=np.array([np.nan, np.nan, np.nan], dtype=np.float64),
    )


def bounds_in_subspace(
    basis: SynergyBasis,
    lo_full: np.ndarray,
    hi_full: np.ndarray,
    n_samples: int = 8192,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-coefficient bounds by Monte-Carlo over the full-ctrl box.

    The full-ctrl box maps to a polytope in coefficient space; we use a
    conservative axis-aligned envelope estimated from random uniform samples
    inside the box, projected through the basis.
    """
    rng = np.random.default_rng(seed)
    full_samples = rng.uniform(low=lo_full, high=hi_full, size=(n_samples, lo_full.size))
    coeffs = (full_samples - basis.mean) @ basis.components.T
    lo = coeffs.min(axis=0)
    hi = coeffs.max(axis=0)
    # Add a small margin so init mean isn't on a boundary.
    margin = 0.05 * (hi - lo + 1e-9)
    return lo - margin, hi + margin
