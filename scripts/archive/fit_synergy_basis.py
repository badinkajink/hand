"""Fit and save a synergy (eigengrasp) basis from existing multitask CSVs.

Usage::

    uv run python scripts/fit_synergy_basis.py \
        --csv-glob 'results/phase1/**/all_candidates_multitask.csv' \
        --n-components 4 \
        --output results/synergy_basis_k4.npz

Run a sanity report on the saved basis::

    uv run python scripts/fit_synergy_basis.py --inspect results/synergy_basis_k4.npz
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from morphohand.optimization.eigengrasp import (
    SynergyBasis,
    fit_synergy_basis_from_csvs,
)


def save_basis(basis: SynergyBasis, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=basis.mean,
        components=basis.components,
        explained_variance_ratio=(
            basis.explained_variance_ratio
            if basis.explained_variance_ratio is not None
            else np.array([])
        ),
    )


def load_basis(path: Path) -> SynergyBasis:
    data = np.load(path)
    evr = data["explained_variance_ratio"]
    return SynergyBasis(
        mean=data["mean"],
        components=data["components"],
        explained_variance_ratio=(evr if evr.size else None),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv-glob", default="results/phase1/**/all_candidates_multitask.csv")
    p.add_argument("--n-components", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("results/synergy_basis_k4.npz"))
    p.add_argument("--inspect", type=Path, default=None,
                   help="Load an existing basis and print its summary.")
    args = p.parse_args()

    if args.inspect is not None:
        basis = load_basis(args.inspect)
        print(f"Loaded {args.inspect}")
        print(f"  K = {basis.n_components}, D = {basis.n_full}")
        print(f"  mean = {basis.mean.round(4).tolist()}")
        if basis.explained_variance_ratio is not None:
            evr = basis.explained_variance_ratio
            print(f"  explained_variance_ratio = {evr.round(4).tolist()}")
            print(f"  cumulative = {np.cumsum(evr).round(4).tolist()}")
        for i, comp in enumerate(basis.components):
            print(f"  comp[{i}] = {comp.round(3).tolist()}")
        return

    csvs = sorted(Path().glob(args.csv_glob))
    if not csvs:
        raise SystemExit(f"No CSVs matched glob: {args.csv_glob}")
    print(f"Fitting basis on {len(csvs)} CSVs, n_components={args.n_components}")
    basis = fit_synergy_basis_from_csvs(csvs, n_components=args.n_components)
    save_basis(basis, args.output)

    print(f"Saved basis to {args.output}")
    print(f"  explained_variance_ratio = {basis.explained_variance_ratio.round(4).tolist()}")
    print(f"  cumulative = {np.cumsum(basis.explained_variance_ratio).round(4).tolist()}")


if __name__ == "__main__":
    main()
