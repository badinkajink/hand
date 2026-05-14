"""Object eval suite for comparing grasp synthesis methods.

A benchmark-driven harness that runs each registered method across every
benchmark (scene/keyframe/contact-target combination) at matched CEM budget
and several seeds, then aggregates per-benchmark and overall.

Outputs
-------
- ``results/eval_suite/<run_tag>/summary.json``       per-run details
- ``results/eval_suite/<run_tag>/leaderboard.md``     cross-benchmark table
- ``results/eval_suite/<run_tag>/per_benchmark.md``   one section per benchmark
- ``results/eval_suite/<run_tag>/videos/``            best-grasp rollouts (MP4)
- ``results/eval_suite/<run_tag>/oracle_scores.png``  cross-benchmark bar chart

Usage
-----
List benchmarks::

    uv run python scripts/eval_suite.py --list

Run the full sweep (baseline + contact_map by default)::

    uv run python scripts/eval_suite.py --seeds 3

Restrict to a subset::

    uv run python scripts/eval_suite.py --benchmarks cube,prism --methods baseline

Skip GIFs when iterating::

    uv run python scripts/eval_suite.py --no-gifs
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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
from morphohand.sampling.scene import freeze_scene_for_eval


# ---------- benchmarks ---------------------------------------------------------


@dataclass(frozen=True)
class Benchmark:
    name: str
    scene_xml: Path
    keyframe: str
    contact_targets_path: Path
    description: str


BENCHMARKS: list[Benchmark] = [
    Benchmark(
        name="cube",
        scene_xml=Path("assets/mjcf/scene.xml"),
        keyframe="open",
        contact_targets_path=Path("assets/contact_targets/cube.yaml"),
        description="40mm cube; pinch across X axis",
    ),
    Benchmark(
        name="prism",
        scene_xml=Path("assets/mjcf/scene_prism.xml"),
        keyframe="open",
        contact_targets_path=Path("assets/contact_targets/prism.yaml"),
        description="22x68x18mm prism long along Y; pinch grip across X",
    ),
    Benchmark(
        name="screwdriver_medium_flat",
        scene_xml=Path("assets/mjcf/scene_screwdriver_medium.xml"),
        keyframe="open_flat",
        contact_targets_path=Path("assets/contact_targets/screwdriver_medium_open_flat.yaml"),
        description="12mm cyl screwdriver horizontal; wrap grip",
    ),
    Benchmark(
        name="screwdriver_medium_vertical",
        scene_xml=Path("assets/mjcf/scene_screwdriver_medium.xml"),
        keyframe="open_vertical",
        contact_targets_path=Path("assets/contact_targets/screwdriver_medium_open_vertical.yaml"),
        description="12mm cyl screwdriver vertical",
    ),
    Benchmark(
        name="screwdriver_medium_90vertical",
        scene_xml=Path("assets/mjcf/scene_screwdriver_medium.xml"),
        keyframe="open_90vertical",
        contact_targets_path=Path("assets/contact_targets/screwdriver_medium_open_90vertical.yaml"),
        description="12mm cyl screwdriver 90deg rotated",
    ),
    Benchmark(
        name="screwdriver_small_flat",
        scene_xml=Path("assets/mjcf/scene_screwdriver_small.xml"),
        keyframe="open_flat",
        contact_targets_path=Path("assets/contact_targets/screwdriver_small_open_flat.yaml"),
        description="4mm thin screwdriver horizontal; hardest object",
    ),
    Benchmark(
        name="power_drill",
        scene_xml=Path("assets/mjcf/scene_power_drill.xml"),
        keyframe="open_flat",
        contact_targets_path=Path("assets/contact_targets/power_drill_open_flat.yaml"),
        description="Power drill at open_flat",
    ),
    Benchmark(
        name="power_drill_short_proximal",
        scene_xml=Path("assets/mjcf/scene_power_drill_short_proximal.xml"),
        keyframe="open_flat",
        contact_targets_path=Path("assets/contact_targets/power_drill_short_proximal.yaml"),
        description="Power drill with short-proximal hand offset (active target)",
    ),
]


# ---------- run shell + results ------------------------------------------------


@dataclass
class RunResult:
    benchmark: str
    method: str
    seed: int
    best_score: float
    best_iter: int
    history_best: list[float]
    diagnostics: dict[str, float]
    wall_time_s: float
    best_finger_ctrl: np.ndarray | None = None
    oracle_score: float = float("nan")


def _baseline_cfg(extras: dict[str, float] | None = None) -> Phase1EvalConfig:
    """Reduced timing for tractable sweep; identical for every method/benchmark."""
    base: dict[str, Any] = dict(
        settle_steps=120,
        lift_steps=80,
        hold_steps=40,
        lift_delta_z=0.05,
        lift_ramp_steps=40,
    )
    if extras:
        base.update(extras)
    return Phase1EvalConfig(**base)


def _optim_cfg(seed: int, iterations: int, population: int) -> Phase1OptimizationConfig:
    return Phase1OptimizationConfig(
        iterations=iterations,
        population=population,
        elite_fraction=0.25,
        sigma_init=0.20,
        seed=seed,
        log_every=0,
    )


_DIAG_KEYS = (
    "score",
    "cube_lift",
    "cube_tip_contacts",
    "cube_z_drop_from_peak",
    "cube_xy_drift",
    "cube_axis_tilt",
    "mean_tip_distance",
    "all_finger_contact_persistence",
    "min_finger_contact_persistence",
    "contact_target_reward",
    "contact_target_mean_distance",
    "fc_q1_distance",
    "fc_wrench_spread",
    "fc_normal_balance",
    "fc_fingers_engaged",
)


def _diagnostics(metrics: dict[str, float]) -> dict[str, float]:
    return {k: float(metrics.get(k, float("nan"))) for k in _DIAG_KEYS}


def _history_best(history: list[dict[str, float]]) -> list[float]:
    return [float(row["best_score_so_far"]) for row in history]


# ---------- methods ------------------------------------------------------------


@dataclass
class MethodContext:
    iterations: int
    population: int
    synergy_basis: SynergyBasis | None


def _make_evaluator(
    bench: Benchmark,
    cfg: Phase1EvalConfig,
    contact_targets: ContactTargetSet | None,
) -> Phase1GraspEvaluator:
    # Always evaluate against the frozen scene — morph DOFs in the base XML
    # would drift otherwise. _frozen_scene_for caches per (scene, keyframe).
    return Phase1GraspEvaluator(
        scene_xml=_frozen_scene_for(bench),
        keyframe=bench.keyframe,
        cfg=cfg,
        contact_target_set=contact_targets,
    )


def run_baseline(bench: Benchmark, seed: int, ctx: MethodContext) -> RunResult:
    ev = _make_evaluator(bench, _baseline_cfg(), None)
    t0 = time.perf_counter()
    out = optimize_finger_controls(ev, _optim_cfg(seed, ctx.iterations, ctx.population))
    return _wrap_run(bench, "baseline", seed, out, time.perf_counter() - t0)


def run_contact_map(bench: Benchmark, seed: int, ctx: MethodContext) -> RunResult:
    targets = ContactTargetSet.from_yaml(bench.contact_targets_path)
    cfg = _baseline_cfg({
        "objective_weight_contact_target_reward": 10.0,
        "objective_weight_contact_target_distance_penalty": 20.0,
    })
    ev = _make_evaluator(bench, cfg, targets)
    t0 = time.perf_counter()
    out = optimize_finger_controls(ev, _optim_cfg(seed, ctx.iterations, ctx.population))
    return _wrap_run(bench, "contact_map", seed, out, time.perf_counter() - t0)


def run_force_closure(bench: Benchmark, seed: int, ctx: MethodContext) -> RunResult:
    cfg = _baseline_cfg({"objective_weight_force_closure": 3.0})
    ev = _make_evaluator(bench, cfg, None)
    t0 = time.perf_counter()
    out = optimize_finger_controls(ev, _optim_cfg(seed, ctx.iterations, ctx.population))
    return _wrap_run(bench, "force_closure", seed, out, time.perf_counter() - t0)


def run_synergy_k4(bench: Benchmark, seed: int, ctx: MethodContext) -> RunResult:
    if ctx.synergy_basis is None:
        raise RuntimeError("synergy_k4 requested without a fitted basis")
    ev = _make_evaluator(bench, _baseline_cfg(), None)
    t0 = time.perf_counter()
    out = optimize_finger_controls_synergy(
        ev, _optim_cfg(seed, ctx.iterations, ctx.population), ctx.synergy_basis
    )
    return _wrap_run(bench, "synergy_k4", seed, out, time.perf_counter() - t0)


_METHODS: dict[str, Callable[[Benchmark, int, MethodContext], RunResult]] = {
    "baseline": run_baseline,
    "contact_map": run_contact_map,
    "force_closure": run_force_closure,
    "synergy_k4": run_synergy_k4,
}


def _wrap_run(
    bench: Benchmark,
    method: str,
    seed: int,
    out: dict[str, Any],
    wall: float,
) -> RunResult:
    return RunResult(
        benchmark=bench.name,
        method=method,
        seed=seed,
        best_score=float(out["best_score"]),
        best_iter=int(np.argmax(_history_best(out["history"]))),
        history_best=_history_best(out["history"]),
        diagnostics=_diagnostics(out["best_metrics"]),
        wall_time_s=wall,
        best_finger_ctrl=np.asarray(out["best_finger_ctrl"], dtype=np.float64),
    )


# ---------- oracle re-evaluation + GIF rendering ------------------------------


def oracle_rescore(results: list[RunResult]) -> None:
    """Re-score each run's best grasp under the *baseline* objective."""
    cache: dict[str, Phase1GraspEvaluator] = {}
    for r in results:
        if r.best_finger_ctrl is None:
            continue
        bench = _benchmark_by_name(r.benchmark)
        key = f"{bench.scene_xml}::{bench.keyframe}"
        if key not in cache:
            cache[key] = _make_evaluator(bench, _baseline_cfg(), None)
        s, _ = cache[key].evaluate(r.best_finger_ctrl)
        r.oracle_score = float(s)


def render_best_videos(
    results: list[RunResult],
    output_dir: Path,
    frame_stride: int = 6,
) -> dict[tuple[str, str], Path]:
    """Render an MP4 rollout for the best seed of each (benchmark, method)."""
    videos: dict[tuple[str, str], Path] = {}
    best_per_pair: dict[tuple[str, str], RunResult] = {}
    for r in results:
        if r.best_finger_ctrl is None:
            continue
        key = (r.benchmark, r.method)
        prev = best_per_pair.get(key)
        if prev is None or r.oracle_score > prev.oracle_score:
            best_per_pair[key] = r

    for (bench_name, method_name), r in sorted(best_per_pair.items()):
        bench = _benchmark_by_name(bench_name)
        ev = _make_evaluator(bench, _baseline_cfg(), None)
        out_path = output_dir / f"{bench_name}__{method_name}.mp4"
        try:
            ev.render_rollout(r.best_finger_ctrl, out_path, frame_stride=frame_stride)
            videos[(bench_name, method_name)] = out_path
            print(f"  rendered {out_path.name}")
        except Exception as exc:
            print(f"  video render failed for {bench_name}/{method_name}: {exc}")
    return videos


def _benchmark_by_name(name: str) -> Benchmark:
    for b in BENCHMARKS:
        if b.name == name:
            return b
    raise KeyError(name)


# REQUIRED: every scene path that reaches Phase1GraspEvaluator goes through
# this resolver, which guarantees a frozen XML (morph joints baked out).
# Pointing the evaluator at a raw base scene lets the morph DOFs drift and
# silently invalidates the experiment. See feedback memory
# "Always freeze the scene before grasp evaluation".
_FROZEN_SCENE_CACHE: dict[tuple[Path, str], Path] = {}
_FROZEN_DIR: Path | None = None


def _set_frozen_scenes_dir(path: Path) -> None:
    global _FROZEN_DIR
    _FROZEN_DIR = Path(path)
    _FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    _FROZEN_SCENE_CACHE.clear()


def _frozen_scene_for(bench: Benchmark) -> Path:
    if _FROZEN_DIR is None:
        raise RuntimeError(
            "Frozen-scene cache dir not initialized; call _set_frozen_scenes_dir() "
            "before constructing any evaluator. (Frozen scenes are mandatory — see "
            "feedback memory 'Always freeze the scene before grasp evaluation'.)"
        )
    key = (bench.scene_xml.resolve(), bench.keyframe)
    cached = _FROZEN_SCENE_CACHE.get(key)
    if cached is not None and cached.exists():
        return cached
    frozen_path = _FROZEN_DIR / f"{bench.name}.frozen.xml"
    freeze_scene_for_eval(bench.scene_xml, bench.keyframe, frozen_path)
    _FROZEN_SCENE_CACHE[key] = frozen_path
    return frozen_path


# ---------- aggregation + reporting -------------------------------------------


def aggregate(results: list[RunResult]) -> dict[str, Any]:
    """Group results by (benchmark, method) and compute summary stats."""
    by_pair: dict[tuple[str, str], list[RunResult]] = {}
    for r in results:
        by_pair.setdefault((r.benchmark, r.method), []).append(r)

    rows: list[dict[str, Any]] = []
    for (bench, method), runs in by_pair.items():
        oracle = np.asarray([r.oracle_score for r in runs], dtype=np.float64)
        in_method = np.asarray([r.best_score for r in runs], dtype=np.float64)
        wall = np.asarray([r.wall_time_s for r in runs], dtype=np.float64)
        diag = {
            k: float(np.nanmean([r.diagnostics.get(k, float("nan")) for r in runs]))
            for k in _DIAG_KEYS
        }
        rows.append(
            {
                "benchmark": bench,
                "method": method,
                "n_seeds": int(len(runs)),
                "oracle_score_mean": float(np.nanmean(oracle)),
                "oracle_score_std": float(np.nanstd(oracle, ddof=0)),
                "oracle_score_min": float(np.nanmin(oracle)),
                "oracle_score_max": float(np.nanmax(oracle)),
                "in_method_score_mean": float(in_method.mean()),
                "wall_time_s_mean": float(wall.mean()),
                "diagnostics_mean": diag,
            }
        )

    # cross-benchmark deltas: best contact_map vs baseline per benchmark
    deltas: list[dict[str, Any]] = []
    bench_names = sorted({r["benchmark"] for r in rows})
    method_names = sorted({r["method"] for r in rows})
    if "baseline" in method_names:
        for b in bench_names:
            baseline_row = next((r for r in rows if r["benchmark"] == b and r["method"] == "baseline"), None)
            if baseline_row is None:
                continue
            for m in method_names:
                if m == "baseline":
                    continue
                cmp_row = next((r for r in rows if r["benchmark"] == b and r["method"] == m), None)
                if cmp_row is None:
                    continue
                deltas.append(
                    {
                        "benchmark": b,
                        "method": m,
                        "delta_oracle_mean": cmp_row["oracle_score_mean"]
                        - baseline_row["oracle_score_mean"],
                        "baseline_oracle_mean": baseline_row["oracle_score_mean"],
                        "method_oracle_mean": cmp_row["oracle_score_mean"],
                    }
                )

    return {"rows": rows, "deltas": deltas}


# ---------- reporting ---------------------------------------------------------


def write_leaderboard(agg: dict[str, Any], path: Path) -> None:
    rows = agg["rows"]
    deltas = agg["deltas"]

    bench_names = sorted({r["benchmark"] for r in rows})
    method_names = sorted({r["method"] for r in rows})

    lines: list[str] = []
    lines.append("# Eval suite leaderboard\n\n")
    lines.append(
        "All scores are re-evaluated under the **baseline objective** for "
        "apples-to-apples comparison. Higher is better.\n\n"
    )

    lines.append("## Oracle score per (benchmark, method)\n\n")
    lines.append("| Benchmark | " + " | ".join(method_names) + " |\n")
    lines.append("|---|" + "---:|" * len(method_names) + "\n")
    for b in bench_names:
        row = [f"`{b}`"]
        for m in method_names:
            cell = next((r for r in rows if r["benchmark"] == b and r["method"] == m), None)
            if cell is None:
                row.append("—")
            else:
                row.append(
                    f"{cell['oracle_score_mean']:.2f} ± {cell['oracle_score_std']:.2f}"
                )
        lines.append("| " + " | ".join(row) + " |\n")

    if deltas:
        lines.append("\n## Δ vs baseline per benchmark\n\n")
        comp_methods = sorted({d["method"] for d in deltas})
        lines.append("| Benchmark | " + " | ".join(f"Δ {m}" for m in comp_methods) + " |\n")
        lines.append("|---|" + "---:|" * len(comp_methods) + "\n")
        for b in bench_names:
            row = [f"`{b}`"]
            for m in comp_methods:
                d = next((d for d in deltas if d["benchmark"] == b and d["method"] == m), None)
                if d is None:
                    row.append("—")
                else:
                    sign = "+" if d["delta_oracle_mean"] >= 0 else ""
                    row.append(f"{sign}{d['delta_oracle_mean']:.2f}")
            lines.append("| " + " | ".join(row) + " |\n")

        # mean Δ across benchmarks
        lines.append("\n### Mean Δ across benchmarks\n\n")
        for m in comp_methods:
            ds = [d["delta_oracle_mean"] for d in deltas if d["method"] == m]
            if not ds:
                continue
            arr = np.asarray(ds, dtype=np.float64)
            lines.append(
                f"- **{m}**: mean Δ = {arr.mean():+.2f}, "
                f"median Δ = {float(np.median(arr)):+.2f}, "
                f"wins {int(np.sum(arr > 0))}/{len(arr)} benchmarks\n"
            )

    path.write_text("".join(lines))


def write_per_benchmark(
    agg: dict[str, Any],
    videos: dict[tuple[str, str], Path],
    path: Path,
) -> None:
    rows = agg["rows"]
    bench_names = sorted({r["benchmark"] for r in rows})
    method_names = sorted({r["method"] for r in rows})

    lines: list[str] = []
    lines.append("# Per-benchmark details\n\n")
    for b in bench_names:
        bench = _benchmark_by_name(b)
        lines.append(f"## `{b}` — {bench.description}\n\n")
        lines.append(f"- scene: `{bench.scene_xml}`\n")
        lines.append(f"- keyframe: `{bench.keyframe}`\n")
        lines.append(f"- contact targets: `{bench.contact_targets_path}`\n\n")

        lines.append("| Method | oracle (baseline obj) | cube_lift | tip_contacts | "
                     "all_finger_persist | fc_q1 |\n")
        lines.append("|---|---:|---:|---:|---:|---:|\n")
        for m in method_names:
            row = next((r for r in rows if r["benchmark"] == b and r["method"] == m), None)
            if row is None:
                continue
            d = row["diagnostics_mean"]
            lines.append(
                f"| `{m}` "
                f"| {row['oracle_score_mean']:.2f} ± {row['oracle_score_std']:.2f} "
                f"| {d.get('cube_lift', float('nan')):.4f} "
                f"| {d.get('cube_tip_contacts', float('nan')):.1f} "
                f"| {d.get('all_finger_contact_persistence', float('nan')):.2f} "
                f"| {d.get('fc_q1_distance', float('nan')):.3f} |\n"
            )

        video_lines: list[str] = []
        for m in method_names:
            key = (b, m)
            if key in videos:
                rel = videos[key].relative_to(path.parent)
                video_lines.append(
                    f'<video src="{rel}" width="280" autoplay loop muted playsinline></video>'
                )
        if video_lines:
            lines.append("\n" + " ".join(video_lines) + "\n\n")
        lines.append("\n")

    path.write_text("".join(lines))


def write_summary_chart(agg: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows = agg["rows"]
    bench_names = sorted({r["benchmark"] for r in rows})
    method_names = sorted({r["method"] for r in rows})

    x = np.arange(len(bench_names))
    width = 0.8 / max(1, len(method_names))

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(bench_names)), 5))
    for i, m in enumerate(method_names):
        means = []
        errs = []
        for b in bench_names:
            row = next((r for r in rows if r["benchmark"] == b and r["method"] == m), None)
            if row is None:
                means.append(0.0)
                errs.append(0.0)
            else:
                means.append(row["oracle_score_mean"])
                errs.append(row["oracle_score_std"])
        ax.bar(x + (i - (len(method_names) - 1) / 2) * width, means, width=width,
               yerr=errs, capsize=2.5, label=m, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(bench_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("oracle (baseline-objective) score")
    ax.set_title("Eval suite — oracle scores per benchmark / method")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_delta_chart(agg: dict[str, Any], path: Path) -> None:
    deltas = agg["deltas"]
    if not deltas:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    bench_names = sorted({d["benchmark"] for d in deltas})
    methods = sorted({d["method"] for d in deltas})
    x = np.arange(len(bench_names))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(bench_names)), 5))
    for i, m in enumerate(methods):
        ys = []
        for b in bench_names:
            d = next((d for d in deltas if d["benchmark"] == b and d["method"] == m), None)
            ys.append(d["delta_oracle_mean"] if d else 0.0)
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, ys, width=width, label=m, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bench_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Δ oracle score vs baseline")
    ax.set_title("Eval suite — improvement over baseline (positive = better)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------- driver ------------------------------------------------------------


def _list_benchmarks() -> None:
    for b in BENCHMARKS:
        print(f"  {b.name:40s} {b.scene_xml.name} @ {b.keyframe}  -- {b.description}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list benchmarks and exit")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument(
        "--methods",
        default="baseline,contact_map",
        help="comma-separated method names; choices: " + ",".join(sorted(_METHODS)),
    )
    parser.add_argument("--benchmarks", default="", help="comma-separated benchmark names; empty = all")
    parser.add_argument(
        "--csv-glob",
        default="results/phase1/**/all_candidates_multitask.csv",
        help="CSVs used to fit the synergy basis when synergy_k4 is requested",
    )
    parser.add_argument("--no-gifs", action="store_true")
    parser.add_argument("--gif-stride", type=int, default=6)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/eval_suite")
    )
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()

    if args.list:
        _list_benchmarks()
        return

    bench_filter = {s.strip() for s in args.benchmarks.split(",") if s.strip()}
    benches = [b for b in BENCHMARKS if not bench_filter or b.name in bench_filter]
    method_names = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in method_names if m not in _METHODS]
    if unknown:
        raise SystemExit(f"unknown method(s): {unknown}; valid: {sorted(_METHODS)}")

    run_tag = args.run_tag or datetime.utcnow().strftime("run_%Y%m%dT%H%M%S")
    out_dir = args.output_root / run_tag
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    # MANDATORY: prepare frozen scenes for every benchmark before any
    # evaluator is constructed. See feedback memory on frozen-scene protocol.
    _set_frozen_scenes_dir(out_dir / "frozen_scenes")
    for bench in benches:
        _frozen_scene_for(bench)  # populate cache up-front so log is clean
    print(f"frozen scenes -> {out_dir / 'frozen_scenes'}")

    print(f"run_tag = {run_tag}")
    print(f"benchmarks ({len(benches)}): {[b.name for b in benches]}")
    print(f"methods ({len(method_names)}): {method_names}")
    print(f"budget: {args.iterations} iters x {args.population} pop = {args.iterations * args.population} evals/seed; seeds={args.seeds}")

    synergy_basis: SynergyBasis | None = None
    if "synergy_k4" in method_names:
        csvs = sorted(Path().glob(args.csv_glob))
        print(f"Fitting synergy basis K=4 on {len(csvs)} CSVs...")
        synergy_basis = fit_synergy_basis_from_csvs(csvs, n_components=4)
        print(f"  cum var = {float(np.sum(synergy_basis.explained_variance_ratio)):.3f}")

    ctx = MethodContext(
        iterations=args.iterations,
        population=args.population,
        synergy_basis=synergy_basis,
    )

    results: list[RunResult] = []
    for bench in benches:
        print(f"\n=== {bench.name} ===")
        for method in method_names:
            fn = _METHODS[method]
            for seed in range(args.seeds):
                t0 = time.perf_counter()
                r = fn(bench, seed, ctx)
                results.append(r)
                print(
                    f"  {method:12s} seed={seed} score={r.best_score:+7.3f} "
                    f"in {time.perf_counter() - t0:5.2f}s"
                )

    print("\n=== Oracle re-evaluation ===")
    oracle_rescore(results)
    for r in results:
        print(
            f"  {r.benchmark:30s} {r.method:12s} seed={r.seed} oracle={r.oracle_score:+7.3f}"
        )

    agg = aggregate(results)

    print("\n=== Reporting ===")
    summary = {
        "run_tag": run_tag,
        "iterations": args.iterations,
        "population": args.population,
        "seeds": args.seeds,
        "methods": method_names,
        "benchmarks": [b.name for b in benches],
        "rows": agg["rows"],
        "deltas": agg["deltas"],
        "runs": [
            {
                "benchmark": r.benchmark,
                "method": r.method,
                "seed": r.seed,
                "best_score": r.best_score,
                "oracle_score": r.oracle_score,
                "best_iter": r.best_iter,
                "wall_time_s": r.wall_time_s,
                "history_best": r.history_best,
                "diagnostics": r.diagnostics,
            }
            for r in results
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_summary_chart(agg, out_dir / "scores.png")
    write_delta_chart(agg, out_dir / "deltas.png")

    videos: dict[tuple[str, str], Path] = {}
    if not args.no_gifs:
        print("Rendering rollout videos...")
        videos = render_best_videos(results, out_dir / "videos", frame_stride=args.gif_stride)

    write_leaderboard(agg, out_dir / "leaderboard.md")
    write_per_benchmark(agg, videos, out_dir / "per_benchmark.md")
    print(f"\nWrote outputs under {out_dir}")


if __name__ == "__main__":
    main()
