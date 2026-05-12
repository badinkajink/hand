"""Side-by-side comparison of grasp specification methods.

Runs CEM on the same scene/keyframe under five conditions:

  baseline      -- raw 9D finger-control CEM (existing behaviour)
  synergy_k3    -- CEM in a 3D synergy subspace fit from historical CSVs
  synergy_k4    -- CEM in a 4D synergy subspace fit from historical CSVs
  contact_map   -- baseline CEM, augmented objective with target contact patches
  force_closure -- baseline CEM, augmented objective with FC energy

Each condition is run with several seeds at matched evaluation budget; we
log convergence (best score so far per iteration) and a small handful of
diagnostic metrics from the best evaluated grasp.

Output: a single JSON to ``results/method_comparison/summary.json`` and a
human-readable Markdown table at ``results/method_comparison/results.md``.
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from morphohand.optimization.contact_targets import ContactTargetSet
from morphohand.optimization.eigengrasp import SynergyBasis, fit_synergy_basis_from_csvs
from morphohand.optimization.phase1_common import (
    Phase1EvalConfig,
    Phase1GraspEvaluator,
)
from morphohand.optimization.phase1_strategy_cem import (
    Phase1OptimizationConfig,
    optimize_finger_controls,
)
from morphohand.optimization.phase1_strategy_synergy_cem import (
    optimize_finger_controls_synergy,
)


@dataclass
class RunResult:
    method: str
    seed: int
    best_score: float
    best_iter: int
    history_best: list[float]
    diagnostics: dict[str, float]
    wall_time_s: float
    best_finger_ctrl: np.ndarray | None = None
    oracle_score: float = float("nan")


def _make_cfg(extras: dict[str, float] | None = None) -> Phase1EvalConfig:
    base = dict(
        settle_steps=120,
        lift_steps=80,
        hold_steps=40,
        lift_delta_z=0.05,
        lift_ramp_steps=40,
    )
    if extras:
        base.update(extras)
    return Phase1EvalConfig(**base)


_OPTIM_ITERATIONS = 12
_OPTIM_POPULATION = 24


def _make_optim_cfg(seed: int) -> Phase1OptimizationConfig:
    return Phase1OptimizationConfig(
        iterations=_OPTIM_ITERATIONS,
        population=_OPTIM_POPULATION,
        elite_fraction=0.25,
        sigma_init=0.20,
        seed=seed,
        log_every=0,
    )


def _make_evaluator(
    scene_xml: Path,
    keyframe: str,
    cfg: Phase1EvalConfig,
    contact_targets: ContactTargetSet | None = None,
) -> Phase1GraspEvaluator:
    return Phase1GraspEvaluator(
        scene_xml=scene_xml,
        keyframe=keyframe,
        cfg=cfg,
        contact_target_set=contact_targets,
    )


def _extract_history(history: list[dict[str, float]]) -> list[float]:
    return [float(row["best_score_so_far"]) for row in history]


def _diagnostics(metrics: dict[str, float]) -> dict[str, float]:
    keep = [
        "score",
        "cube_lift",
        "cube_tip_contacts",
        "cube_z_drop_from_peak",
        "mean_tip_distance",
        "all_finger_contact_persistence",
        "contact_target_reward",
        "contact_target_mean_distance",
        "fc_q1_distance",
        "fc_wrench_spread",
        "fc_normal_balance",
        "fc_fingers_engaged",
    ]
    return {k: float(metrics.get(k, float("nan"))) for k in keep}


def run_baseline(scene_xml: Path, keyframe: str, seed: int) -> RunResult:
    cfg = _make_cfg()
    ev = _make_evaluator(scene_xml, keyframe, cfg)
    t0 = time.perf_counter()
    out = optimize_finger_controls(ev, _make_optim_cfg(seed))
    wall = time.perf_counter() - t0
    return RunResult(
        method="baseline",
        seed=seed,
        best_score=float(out["best_score"]),
        best_iter=int(np.argmax(_extract_history(out["history"]))),
        history_best=_extract_history(out["history"]),
        diagnostics=_diagnostics(out["best_metrics"]),
        wall_time_s=wall,
        best_finger_ctrl=np.asarray(out["best_finger_ctrl"], dtype=np.float64),
    )


def run_synergy(scene_xml: Path, keyframe: str, seed: int, basis: SynergyBasis) -> RunResult:
    cfg = _make_cfg()
    ev = _make_evaluator(scene_xml, keyframe, cfg)
    t0 = time.perf_counter()
    out = optimize_finger_controls_synergy(ev, _make_optim_cfg(seed), basis)
    wall = time.perf_counter() - t0
    return RunResult(
        method=f"synergy_k{basis.n_components}",
        seed=seed,
        best_score=float(out["best_score"]),
        best_iter=int(np.argmax(_extract_history(out["history"]))),
        history_best=_extract_history(out["history"]),
        diagnostics=_diagnostics(out["best_metrics"]),
        wall_time_s=wall,
        best_finger_ctrl=np.asarray(out["best_finger_ctrl"], dtype=np.float64),
    )


def run_contact_map(
    scene_xml: Path,
    keyframe: str,
    seed: int,
    targets: ContactTargetSet,
) -> RunResult:
    cfg = _make_cfg({
        "objective_weight_contact_target_reward": 10.0,
        "objective_weight_contact_target_distance_penalty": 20.0,
    })
    ev = _make_evaluator(scene_xml, keyframe, cfg, contact_targets=targets)
    t0 = time.perf_counter()
    out = optimize_finger_controls(ev, _make_optim_cfg(seed))
    wall = time.perf_counter() - t0
    return RunResult(
        method="contact_map",
        seed=seed,
        best_score=float(out["best_score"]),
        best_iter=int(np.argmax(_extract_history(out["history"]))),
        history_best=_extract_history(out["history"]),
        diagnostics=_diagnostics(out["best_metrics"]),
        wall_time_s=wall,
        best_finger_ctrl=np.asarray(out["best_finger_ctrl"], dtype=np.float64),
    )


def run_force_closure(scene_xml: Path, keyframe: str, seed: int) -> RunResult:
    cfg = _make_cfg({"objective_weight_force_closure": 3.0})
    ev = _make_evaluator(scene_xml, keyframe, cfg)
    t0 = time.perf_counter()
    out = optimize_finger_controls(ev, _make_optim_cfg(seed))
    wall = time.perf_counter() - t0
    return RunResult(
        method="force_closure",
        seed=seed,
        best_score=float(out["best_score"]),
        best_iter=int(np.argmax(_extract_history(out["history"]))),
        history_best=_extract_history(out["history"]),
        diagnostics=_diagnostics(out["best_metrics"]),
        wall_time_s=wall,
        best_finger_ctrl=np.asarray(out["best_finger_ctrl"], dtype=np.float64),
    )


def run_combined(
    scene_xml: Path,
    keyframe: str,
    seed: int,
    basis: SynergyBasis,
    targets: ContactTargetSet,
) -> RunResult:
    cfg = _make_cfg({
        "objective_weight_contact_target_reward": 10.0,
        "objective_weight_contact_target_distance_penalty": 20.0,
        "objective_weight_force_closure": 3.0,
    })
    ev = _make_evaluator(scene_xml, keyframe, cfg, contact_targets=targets)
    t0 = time.perf_counter()
    out = optimize_finger_controls_synergy(ev, _make_optim_cfg(seed), basis)
    wall = time.perf_counter() - t0
    return RunResult(
        method=f"combined_k{basis.n_components}+ct+fc",
        seed=seed,
        best_score=float(out["best_score"]),
        best_iter=int(np.argmax(_extract_history(out["history"]))),
        history_best=_extract_history(out["history"]),
        diagnostics=_diagnostics(out["best_metrics"]),
        wall_time_s=wall,
        best_finger_ctrl=np.asarray(out["best_finger_ctrl"], dtype=np.float64),
    )


def oracle_evaluate(
    results: list[RunResult],
    scene_xml: Path,
    keyframe: str,
) -> None:
    """Re-score each method's best grasp under the *baseline* objective.

    The per-method objectives differ (contact_map and FC add positive terms,
    so a higher in-method score does not imply a better grasp). We rescore
    every method's best control under the original baseline objective and
    attach that as `oracle_score` for an apples-to-apples comparison.
    """
    oracle_cfg = _make_cfg()  # plain baseline weights
    oracle = _make_evaluator(scene_xml, keyframe, oracle_cfg)
    for r in results:
        if r.best_finger_ctrl is None:
            continue
        s, _ = oracle.evaluate(r.best_finger_ctrl)
        r.oracle_score = float(s)


def plot_convergence(results: list[RunResult], path: Path) -> None:
    """Per-method mean convergence curve of best-score-so-far."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    by_method: dict[str, list[list[float]]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r.history_best)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, hists in sorted(by_method.items()):
        H = np.asarray(hists, dtype=np.float64)
        mean = H.mean(axis=0)
        lo = H.min(axis=0)
        hi = H.max(axis=0)
        x = np.arange(1, mean.size + 1)
        line, = ax.plot(x, mean, label=method, linewidth=2.0)
        ax.fill_between(x, lo, hi, alpha=0.15, color=line.get_color())
    ax.set_xlabel("CEM iteration")
    ax.set_ylabel("best in-method score so far")
    ax.set_title("Convergence (in-method objective; not apples-to-apples)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_oracle_bars(results: list[RunResult], path: Path) -> None:
    """Bar chart of oracle (baseline-objective) score per method, with min/max range."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    by_method: dict[str, list[float]] = {}
    for r in results:
        if not np.isnan(r.oracle_score):
            by_method.setdefault(r.method, []).append(r.oracle_score)

    methods = sorted(by_method.keys(), key=lambda m: -float(np.mean(by_method[m])))
    means = [float(np.mean(by_method[m])) for m in methods]
    los = [float(np.min(by_method[m])) for m in methods]
    his = [float(np.max(by_method[m])) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(methods))
    ax.bar(xs, means, color="#4c78a8", alpha=0.85)
    for i, (lo, hi) in enumerate(zip(los, his)):
        ax.plot([i, i], [lo, hi], color="black", linewidth=1.4)
        ax.plot([i - 0.1, i + 0.1], [lo, lo], color="black", linewidth=1.4)
        ax.plot([i - 0.1, i + 0.1], [hi, hi], color="black", linewidth=1.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("oracle (baseline-objective) score")
    ax.set_title("Each method's best grasp re-scored under the baseline objective")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def aggregate(results: list[RunResult]) -> list[dict[str, Any]]:
    by_method: dict[str, list[RunResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    rows: list[dict[str, Any]] = []
    for method, runs in by_method.items():
        scores = np.asarray([r.best_score for r in runs], dtype=np.float64)
        oracle = np.asarray([r.oracle_score for r in runs], dtype=np.float64)
        walls = np.asarray([r.wall_time_s for r in runs], dtype=np.float64)
        iters = np.asarray([r.best_iter for r in runs], dtype=np.int64)
        diag_keys = sorted({k for r in runs for k in r.diagnostics})
        diag_mean = {
            k: float(np.nanmean([r.diagnostics.get(k, float("nan")) for r in runs]))
            for k in diag_keys
        }
        rows.append(
            {
                "method": method,
                "n_seeds": int(len(runs)),
                "best_score_mean": float(scores.mean()),
                "best_score_std": float(scores.std(ddof=0)),
                "best_score_min": float(scores.min()),
                "best_score_max": float(scores.max()),
                "oracle_score_mean": float(np.nanmean(oracle)),
                "oracle_score_std": float(np.nanstd(oracle, ddof=0)),
                "oracle_score_min": float(np.nanmin(oracle)),
                "oracle_score_max": float(np.nanmax(oracle)),
                "wall_time_s_mean": float(walls.mean()),
                "best_iter_mean": float(iters.mean()),
                "diagnostics_mean": diag_mean,
            }
        )
    rows.sort(key=lambda r: r["oracle_score_mean"], reverse=True)
    return rows


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Method comparison: power_drill_short_proximal / open_flat\n")
    lines.append(
        f"Matched CEM budget ({_OPTIM_ITERATIONS} iters x {_OPTIM_POPULATION} pop = "
        f"{_OPTIM_ITERATIONS * _OPTIM_POPULATION} evals/seed); "
        "settle/lift/hold = 120/80/40 sim steps; backend=MuJoCo native.\n"
    )
    lines.append("\n## Summary (sorted by oracle score, baseline objective)\n")
    lines.append(
        "Note: `in-method` scores are NOT directly comparable across methods because "
        "each method optimizes a different objective. The `oracle` column re-scores "
        "the best grasp from each run under the baseline objective for an "
        "apples-to-apples comparison.\n\n"
    )
    lines.append(
        "| Method | seeds | oracle (baseline obj) mean ± std | min | max | "
        "in-method score | wall (s/seed) | cube_lift | tip_contacts | fc_q1 |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        d = r["diagnostics_mean"]
        lines.append(
            f"| `{r['method']}` | {r['n_seeds']} "
            f"| {r['oracle_score_mean']:.3f} ± {r['oracle_score_std']:.3f} "
            f"| {r['oracle_score_min']:.3f} | {r['oracle_score_max']:.3f} "
            f"| {r['best_score_mean']:.3f} "
            f"| {r['wall_time_s_mean']:.2f} "
            f"| {d.get('cube_lift', float('nan')):.4f} "
            f"| {d.get('cube_tip_contacts', float('nan')):.1f} "
            f"| {d.get('fc_q1_distance', float('nan')):.3f} |\n"
        )

    lines.append("\n## Diagnostics means\n")
    diag_keys = sorted({k for r in rows for k in r["diagnostics_mean"]})
    lines.append("| Method | " + " | ".join(diag_keys) + " |\n")
    lines.append("|---|" + "---:|" * len(diag_keys) + "\n")
    for r in rows:
        d = r["diagnostics_mean"]
        vals = " | ".join(
            f"{d.get(k, float('nan')):.3f}" if not np.isnan(d.get(k, float("nan"))) else "—"
            for k in diag_keys
        )
        lines.append(f"| `{r['method']}` | {vals} |\n")

    path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-xml",
        default="assets/mjcf/scene_power_drill_short_proximal.xml",
        type=Path,
    )
    parser.add_argument("--keyframe", default="open_flat")
    parser.add_argument(
        "--contact-targets",
        default="assets/contact_targets/power_drill_short_proximal.yaml",
        type=Path,
    )
    parser.add_argument(
        "--csv-glob",
        default="results/phase1/**/all_candidates_multitask.csv",
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument(
        "--output-dir",
        default="results/method_comparison",
        type=Path,
    )
    parser.add_argument(
        "--methods",
        default="baseline,synergy_k3,synergy_k4,contact_map,force_closure,combined",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    global _OPTIM_ITERATIONS, _OPTIM_POPULATION
    _OPTIM_ITERATIONS = int(args.iterations)
    _OPTIM_POPULATION = int(args.population)
    print(
        f"CEM budget: iterations={_OPTIM_ITERATIONS}, population={_OPTIM_POPULATION} "
        f"({_OPTIM_ITERATIONS * _OPTIM_POPULATION} evals/seed)"
    )

    methods = set(m.strip() for m in args.methods.split(","))
    csvs = sorted(Path().glob(args.csv_glob))
    print(f"Fitting synergy bases on {len(csvs)} CSVs...")
    basis_k3 = fit_synergy_basis_from_csvs(csvs, n_components=3)
    basis_k4 = fit_synergy_basis_from_csvs(csvs, n_components=4)
    print(
        f"  K=3 cum var = {float(np.sum(basis_k3.explained_variance_ratio)):.3f}; "
        f"K=4 cum var = {float(np.sum(basis_k4.explained_variance_ratio)):.3f}"
    )
    targets = ContactTargetSet.from_yaml(args.contact_targets)
    print(f"Loaded {len(targets.patches)} contact target patches for {targets.object_body}")

    results: list[RunResult] = []
    for seed in range(args.seeds):
        print(f"\n=== seed {seed} ===")
        if "baseline" in methods:
            r = run_baseline(args.scene_xml, args.keyframe, seed)
            print(f"  baseline      score={r.best_score:+.3f}  wall={r.wall_time_s:.2f}s")
            results.append(r)
        if "synergy_k3" in methods:
            r = run_synergy(args.scene_xml, args.keyframe, seed, basis_k3)
            print(f"  synergy_k3    score={r.best_score:+.3f}  wall={r.wall_time_s:.2f}s")
            results.append(r)
        if "synergy_k4" in methods:
            r = run_synergy(args.scene_xml, args.keyframe, seed, basis_k4)
            print(f"  synergy_k4    score={r.best_score:+.3f}  wall={r.wall_time_s:.2f}s")
            results.append(r)
        if "contact_map" in methods:
            r = run_contact_map(args.scene_xml, args.keyframe, seed, targets)
            print(f"  contact_map   score={r.best_score:+.3f}  wall={r.wall_time_s:.2f}s")
            results.append(r)
        if "force_closure" in methods:
            r = run_force_closure(args.scene_xml, args.keyframe, seed)
            print(f"  force_closure score={r.best_score:+.3f}  wall={r.wall_time_s:.2f}s")
            results.append(r)
        if "combined" in methods:
            r = run_combined(args.scene_xml, args.keyframe, seed, basis_k4, targets)
            print(f"  combined      score={r.best_score:+.3f}  wall={r.wall_time_s:.2f}s")
            results.append(r)

    print("\n=== Oracle re-evaluation (baseline objective) ===")
    oracle_evaluate(results, args.scene_xml, args.keyframe)
    for r in results:
        print(f"  {r.method:<28s} seed={r.seed} oracle_score={r.oracle_score:+.3f}")

    plot_convergence(results, args.output_dir / "convergence.png")
    plot_oracle_bars(results, args.output_dir / "oracle_scores.png")

    summary = aggregate(results)
    raw = {
        "scene_xml": str(args.scene_xml),
        "keyframe": args.keyframe,
        "seeds": args.seeds,
        "n_runs": len(results),
        "summary": summary,
        "runs": [
            {
                "method": r.method,
                "seed": r.seed,
                "best_score": r.best_score,
                "oracle_score": r.oracle_score,
                "best_iter": r.best_iter,
                "history_best": r.history_best,
                "diagnostics": r.diagnostics,
                "wall_time_s": r.wall_time_s,
            }
            for r in results
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(raw, indent=2))
    write_markdown(summary, args.output_dir / "results.md")
    print(f"\nWrote {args.output_dir/'summary.json'} and {args.output_dir/'results.md'}")


if __name__ == "__main__":
    main()
