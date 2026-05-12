# Synergy (eigengrasp) subspace CEM

## What this is

A drop-in replacement for the raw-9D-control CEM search. Instead of sampling
Gaussians over `(yaw, mcp, pip) × 3` directly, we project to a small basis fit
on existing CEM candidates (PCA / "eigengrasps") and search in the
coefficient space, then reconstruct back to 9D when evaluating.

The 9D control of this hand is heavily correlated — fingers tend to flex
together, thumb opposes the others, lateral spread couples the three yaws.
A 4-dim basis fit on 13 historical multitask CSVs already captures ~90% of
the variance:

| K | cumulative explained variance |
|---|-------------------------------|
| 1 | 0.56 |
| 2 | 0.72 |
| 3 | 0.81 |
| 4 | 0.90 |

For trajectories of `P` phases, total search dimension drops from `9P` to
`KP` — e.g. a 4-phase trajectory: 36 → 16 with K=4.

## Files

- `src/morphohand/optimization/eigengrasp.py` — `SynergyBasis` dataclass,
  PCA fit (`fit_synergy_basis`, `fit_synergy_basis_from_csvs`), interpretable
  fallback (`hand_designed_basis`), MC-estimated bounds in subspace.
- `src/morphohand/optimization/phase1_strategy_synergy_cem.py` —
  `optimize_finger_controls_synergy`, `optimize_finger_control_trajectory_synergy`.
- `scripts/fit_synergy_basis.py` — CLI to fit/save/inspect a basis as `.npz`.
- `tests/test_eigengrasp.py` — unit tests for basis fit, roundtrip, bounds.

## How to use

```bash
# 1. Fit a basis on all historical multitask CSVs
uv run python scripts/fit_synergy_basis.py \
  --csv-glob 'results/phase1/**/all_candidates_multitask.csv' \
  --n-components 4 \
  --output results/synergy_basis_k4.npz

# 2. Inspect
uv run python scripts/fit_synergy_basis.py --inspect results/synergy_basis_k4.npz
```

```python
from morphohand.optimization.eigengrasp import SynergyBasis
from morphohand.optimization.phase1_strategy_synergy_cem import (
    optimize_finger_controls_synergy,
    optimize_finger_control_trajectory_synergy,
)
from morphohand.optimization.phase1_strategy_cem import Phase1OptimizationConfig

basis = SynergyBasis(mean=..., components=...)  # or load_basis(path)
result = optimize_finger_controls_synergy(
    evaluator=evaluator,
    cfg=Phase1OptimizationConfig(population=20, iterations=24),
    basis=basis,
)
```

## What to check experimentally

A meaningful comparison vs the raw-9D CEM should hold population and iteration
budget constant and compare `best_score` curves (and wall-clock per
`best-score-per-eval`). The expected pattern: synergy CEM converges in fewer
evaluations on the same scene/keyframe, because each sample is "in the
manifold of plausible grasps" rather than a random combination of joint
positions that may be physically meaningless.

Two open knobs worth scanning:
- `n_components` (2 / 3 / 4): trades coverage for search dim.
- Whether to refit the basis per-object vs. use a single global basis.
