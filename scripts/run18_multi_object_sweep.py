"""Run 18: Multi-object morphology sweep on the short-proximal hand.

Six eval tasks (one keyframe per scene), one shared morphology candidate
population. For each morphology:
  - bake it into each task's rigid scene
  - construct a Phase1GraspEvaluator
  - evaluate with that task's foundational ctrl (single-shot, no per-morph
    CEM adaptation -- foundational ctrl is computed ONCE per task before
    the sweep)
  - record score + diagnostics

Outputs per task into `<output_root>/<tag>/<task_label>/`:
  - `all_candidates_<task>.csv`  -- one row per morphology

Plus a cross-task summary in `<output_root>/<tag>/all_candidates_multi.csv`
with per-task score columns and an aggregate (mean across tasks).

Foundational ctrl per task is produced by running
`scripts/phase1_optimize_grasp.py` once per task; outputs land under
`<output_root>/<tag>/foundational/<task_label>/`. Reused if present.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _python_cmd() -> list[str]:
    """Return the command prefix to invoke python in a sibling subprocess.

    Prefers `uv run python` (project-managed env) so all downstream scripts
    pick up the same uv-resolved interpreter and extras. Falls back to
    `sys.executable` if uv is not on PATH (e.g. inside an already-activated
    venv with no uv available).
    """
    import shutil
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "run", "python"]
    return [sys.executable]

import numpy as np  # noqa: E402

from morphohand.optimization.phase1_common import Phase1EvalConfig, Phase1GraspEvaluator  # noqa: E402
from morphohand.optimization.phase1_grasp import Phase1OptimizationConfig  # noqa: E402
from morphohand.sampling.adapt import adapt_foundational_ctrl  # noqa: E402
from morphohand.sampling.foundational import load_foundational_poses  # noqa: E402
from morphohand.sampling.morphology import (  # noqa: E402
    MorphologyBounds,
    morph_distance,
    morph_suffix,
    parse_morphology_from_keyframe,
    sample_morphologies,
)
from morphohand.sampling.scene import freeze_scene_for_eval, write_rigid_scene_with_object_size  # noqa: E402
from morphohand.optimization.contact_targets import ContactTargetSet  # noqa: E402


# ---- task definitions ----------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    label: str
    scene_xml: Path
    keyframe: str
    object_body: str
    # optional task-specific overrides
    contact_targets_yaml: Path | None = None
    pivot_steps: int = 0
    pivot_ramp_steps: int = 80
    pivot_delta_rx: float = 0.0
    pivot_delta_ry: float = 0.0
    pivot_delta_rz: float = 0.0
    lift_delta_z: float = 0.05
    lift_ramp_steps: int = 100


def default_tasks() -> list[Task]:
    return [
        Task(
            label="cube",
            scene_xml=ROOT / "assets/mjcf/scene_cube_short_proximal.xml",
            keyframe="open_short_manual",
            object_body="cube",
            contact_targets_yaml=ROOT / "assets/contact_targets/cube.yaml",
        ),
        Task(
            label="prism",
            scene_xml=ROOT / "assets/mjcf/scene_prism_short_proximal.xml",
            keyframe="open_short_manual",
            object_body="prism",
            contact_targets_yaml=ROOT / "assets/contact_targets/prism.yaml",
        ),
        Task(
            label="power_drill",
            scene_xml=ROOT / "assets/mjcf/scene_power_drill_short_proximal.xml",
            keyframe="open_flat_gripping",
            object_body="power_drill",
            contact_targets_yaml=ROOT / "assets/contact_targets/power_drill_short_proximal.yaml",
            # Slower pivot than run17 (240/200 instead of 180/120) to reduce impulses.
            pivot_steps=240,
            pivot_ramp_steps=200,
            pivot_delta_rx=-1.6,
            pivot_delta_ry=-1.4,
            pivot_delta_rz=1.6018,
            lift_delta_z=0.110,
            lift_ramp_steps=180,
        ),
        Task(
            label="screwdriver_medium_flat",
            scene_xml=ROOT / "assets/mjcf/scene_screwdriver_medium_flat_short_proximal.xml",
            keyframe="open_short_manual",
            object_body="screwdriver_medium",
            contact_targets_yaml=ROOT / "assets/contact_targets/screwdriver_medium_open_flat.yaml",
        ),
        Task(
            label="screwdriver_medium_vertical",
            scene_xml=ROOT / "assets/mjcf/scene_screwdriver_medium_vertical_short_proximal.xml",
            keyframe="open_short_manual",
            object_body="screwdriver_medium",
            contact_targets_yaml=ROOT / "assets/contact_targets/screwdriver_medium_open_vertical.yaml",
        ),
        Task(
            label="screwdriver_medium_90vert",
            scene_xml=ROOT / "assets/mjcf/scene_screwdriver_medium_short_proximal.xml",
            keyframe="open_90vertical_manual",
            # NOTE: scene_screwdriver_medium uses body name "cube" for its primary object
            object_body="cube",
            contact_targets_yaml=ROOT / "assets/contact_targets/screwdriver_medium_open_90vertical.yaml",
        ),
        Task(
            label="screwdriver_small_flat",
            scene_xml=ROOT / "assets/mjcf/scene_screwdriver_small_flat_short_proximal.xml",
            keyframe="open_short_manual",
            object_body="screwdriver_small",
            contact_targets_yaml=ROOT / "assets/contact_targets/screwdriver_small_open_flat.yaml",
        ),
    ]


# ---- foundational pass --------------------------------------------------------------

def run_foundational(task: Task, out_dir: Path, iterations: int, population: int, seed: int) -> Path:
    """Run a single CEM pass on `task` via phase1_optimize_grasp.py. Reuses any
    existing summary.json under `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # phase1_optimize_grasp.py auto-names a `run_YYYY...` subdir; we look for any.
    existing = sorted(out_dir.glob("*/summary.json"))
    if existing:
        print(f"[foundational/{task.label}] reusing {existing[0].parent.name}")
        return existing[0]

    args = [
        *_python_cmd(), str(ROOT / "scripts/phase1_optimize_grasp.py"),
        "--scene-xml", str(task.scene_xml),
        "--keyframe", task.keyframe,
        "--iterations", str(iterations),
        "--population", str(population),
        "--elite-fraction", "0.25",
        "--sigma-init", "0.10",
        "--output-dir", str(out_dir),
        "--seed", str(seed),
        # objective shape consistent with run15/17 (anchor + trajectory FC):
        "--objective-weight-cube-yaw-drift-penalty", "3.0",
        "--objective-weight-cube-axis-tilt-penalty", "2.0",
        "--objective-weight-cube-ang-drift-penalty", "1.0",
        "--objective-weight-finger-ctrl-anchor", "0.5",
        "--objective-weight-min-finger-persistence", "6.0",
        "--objective-weight-trajectory-fc-q1-penalty", "2.0",
        "--objective-weight-trajectory-fc-min-fingers-reward", "1.0",
        "--trajectory-fc-sample-count", "8",
        "--lift-delta-z", str(task.lift_delta_z),
        "--lift-ramp-steps", str(task.lift_ramp_steps),
        "--pivot-steps", str(task.pivot_steps),
        "--pivot-ramp-steps", str(task.pivot_ramp_steps),
        "--pivot-delta-rx", str(task.pivot_delta_rx),
        "--pivot-delta-ry", str(task.pivot_delta_ry),
        "--pivot-delta-rz", str(task.pivot_delta_rz),
    ]
    if task.contact_targets_yaml is not None and task.contact_targets_yaml.exists():
        args += [
            "--contact-targets-yaml", str(task.contact_targets_yaml),
            "--objective-weight-contact-target-reward", "10.0",
            "--objective-weight-contact-target-distance-penalty", "20.0",
        ]
    print(f"\n[foundational/{task.label}] launching CEM ...")
    subprocess.run([str(x) for x in args], check=True)
    existing = sorted(out_dir.glob("*/summary.json"))
    return existing[0]


def load_foundational_ctrl(task: Task, foundational_dir: Path) -> np.ndarray:
    poses = load_foundational_poses(foundational_dir, keyframe_name=task.keyframe)
    return np.asarray(poses[0].finger_ctrl, dtype=np.float64).copy()


# ---- evaluator construction --------------------------------------------------------

def build_eval_cfg(task: Task,
                   contact_target_reward: float = 10.0,
                   contact_target_distance_penalty: float = 20.0,
                   min_finger_persistence: float = 6.0) -> Phase1EvalConfig:
    return Phase1EvalConfig(
        settle_steps=120,
        lift_steps=80,
        hold_steps=40,
        lift_delta_z=task.lift_delta_z,
        lift_ramp_steps=task.lift_ramp_steps,
        pivot_steps=task.pivot_steps,
        pivot_ramp_steps=task.pivot_ramp_steps,
        pivot_delta_rx=task.pivot_delta_rx,
        pivot_delta_ry=task.pivot_delta_ry,
        pivot_delta_rz=task.pivot_delta_rz,
        objective_weight_cube_yaw_drift_penalty=3.0,
        objective_weight_cube_axis_tilt_penalty=2.0,
        objective_weight_cube_ang_drift_penalty=1.0,
        objective_weight_finger_ctrl_anchor=0.5,
        objective_weight_min_finger_persistence=min_finger_persistence,
        objective_weight_trajectory_fc_q1_penalty=2.0,
        objective_weight_trajectory_fc_min_fingers_reward=1.0,
        trajectory_fc_sample_count=8,
        objective_weight_contact_target_reward=contact_target_reward if task.contact_targets_yaml is not None else 0.0,
        objective_weight_contact_target_distance_penalty=contact_target_distance_penalty if task.contact_targets_yaml is not None else 0.0,
    )


# ---- sweep loop --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run18_multi_object")
    ap.add_argument("--output-root", type=Path, default=ROOT / "results/phase1")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--foundational-iterations", type=int, default=100)
    ap.add_argument("--foundational-population", type=int, default=52)
    ap.add_argument("--foundational-seed", type=int, default=0)
    ap.add_argument("--x-perturb", type=float, default=0.012)
    ap.add_argument("--y-perturb", type=float, default=0.012)
    ap.add_argument("--len-perturb", type=float, default=0.012)
    ap.add_argument("--x-min", type=float, default=-0.03)
    ap.add_argument("--x-max", type=float, default=0.03)
    ap.add_argument("--y-min", type=float, default=-0.03)
    ap.add_argument("--y-max", type=float, default=0.03)
    ap.add_argument("--len-min", type=float, default=0.0)
    ap.add_argument("--len-max", type=float, default=0.035)
    ap.add_argument("--base-morph-task", default="power_drill",
                    help="Which task's keyframe morphology becomes the base for sampling")
    ap.add_argument("--skip-foundational", action="store_true",
                    help="Assume foundational ctrl exists under tag/foundational/*")
    # Per-candidate / per-interval ctrl adaptation. Fully compatible with the
    # contact-map objective -- the adapt CEM uses the same evaluator and
    # therefore the same contact-target weights.
    ap.add_argument(
        "--adapt-mode",
        choices=["none", "interval-initial-fp", "interval-open", "sparse-per-morph", "local-perturbation"],
        default="none",
        help="Primary adapt mode (run on EVERY candidate)."
    )
    ap.add_argument("--fp-refresh-interval", type=int, default=40,
                    help="Refresh ctrl every N candidates (interval modes only)")
    ap.add_argument("--interval-adapt-iterations", type=int, default=16)
    ap.add_argument("--interval-adapt-population", type=int, default=36)
    ap.add_argument("--interval-adapt-elite-fraction", type=float, default=0.25)
    ap.add_argument("--interval-adapt-sigma-init", type=float, default=0.09)
    ap.add_argument("--adapt-seed", type=int, default=0)
    # Secondary adapt (run6's "sparse" — runs every candidate alongside interval)
    ap.add_argument("--sparse-adapt-mode",
                    choices=["none", "sparse-per-morph", "local-perturbation"],
                    default="none",
                    help="Extra per-morph refinement applied AFTER interval refresh.")
    ap.add_argument("--sparse-adapt-iterations", type=int, default=2)
    ap.add_argument("--sparse-adapt-population", type=int, default=10)
    ap.add_argument("--sparse-adapt-elite-fraction", type=float, default=0.25)
    ap.add_argument("--sparse-adapt-sigma-init", type=float, default=0.06)
    # Objective weight overrides (defaults match run17/baseline run18)
    ap.add_argument("--contact-target-reward", type=float, default=10.0)
    ap.add_argument("--contact-target-distance-penalty", type=float, default=20.0)
    ap.add_argument("--min-finger-persistence", type=float, default=6.0)
    args = ap.parse_args()

    out_dir = args.output_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run18] output dir: {out_dir}")

    tasks = default_tasks()

    # 1) foundational pass per task
    foundational_root = out_dir / "foundational"
    foundational_ctrl: dict[str, np.ndarray] = {}
    fp_t0 = time.perf_counter()
    for task in tasks:
        task_fp_dir = foundational_root / task.label
        if not args.skip_foundational:
            run_foundational(
                task, task_fp_dir,
                iterations=args.foundational_iterations,
                population=args.foundational_population,
                seed=args.foundational_seed,
            )
        foundational_ctrl[task.label] = load_foundational_ctrl(task, task_fp_dir)
    print(f"[run18] foundational pass done in {time.perf_counter() - fp_t0:.1f}s")

    # 2) sample morphologies (around the base task's keyframe morphology)
    base_task = next(t for t in tasks if t.label == args.base_morph_task)
    base_morph = parse_morphology_from_keyframe(base_task.scene_xml, keyframe_name=base_task.keyframe)
    bounds = MorphologyBounds(
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        len_min=args.len_min, len_max=args.len_max,
    )
    rng = np.random.default_rng(args.seed)
    candidates = sample_morphologies(
        base=base_morph,
        sample_count=args.samples,
        rng=rng,
        bounds=bounds,
        x_perturb=args.x_perturb,
        y_perturb=args.y_perturb,
        len_perturb=args.len_perturb,
    )
    candidates.sort(key=lambda m: morph_distance(m, base_morph))
    print(f"[run18] sampled {len(candidates)} morphologies (base from {base_task.label})")

    # 3) build per-task evaluators lazily; build per-task contact_target_set
    contact_target_sets: dict[str, ContactTargetSet | None] = {}
    eval_cfgs: dict[str, Phase1EvalConfig] = {}
    for task in tasks:
        eval_cfgs[task.label] = build_eval_cfg(
            task,
            contact_target_reward=args.contact_target_reward,
            contact_target_distance_penalty=args.contact_target_distance_penalty,
            min_finger_persistence=args.min_finger_persistence,
        )
        if task.contact_targets_yaml is not None and task.contact_targets_yaml.exists():
            contact_target_sets[task.label] = ContactTargetSet.from_yaml(task.contact_targets_yaml)
        else:
            contact_target_sets[task.label] = None

    # 4) sweep
    gen_dir = out_dir / "generated_mjcf"
    gen_dir.mkdir(parents=True, exist_ok=True)
    per_task_rows: dict[str, list[dict]] = {t.label: [] for t in tasks}
    cross_rows: list[dict] = []

    # Per-task running ctrl that gets refreshed by `adapt_foundational_ctrl`
    # when adapt-mode != "none". Starts at the foundational ctrl. Contact-map
    # synthesis flows through because the adapt CEM uses the SAME evaluator
    # (built per-morphology below) and therefore the same contact_target_set
    # + objective weights.
    current_ctrl: dict[str, np.ndarray] = {
        t.label: foundational_ctrl[t.label].copy() for t in tasks
    }
    adapt_cfg = Phase1OptimizationConfig(
        iterations=args.interval_adapt_iterations,
        population=args.interval_adapt_population,
        elite_fraction=args.interval_adapt_elite_fraction,
        sigma_init=args.interval_adapt_sigma_init,
        seed=args.adapt_seed,
    )
    sparse_adapt_cfg = Phase1OptimizationConfig(
        iterations=args.sparse_adapt_iterations,
        population=args.sparse_adapt_population,
        elite_fraction=args.sparse_adapt_elite_fraction,
        sigma_init=args.sparse_adapt_sigma_init,
        seed=args.adapt_seed,
    )
    adapt_seconds: dict[str, float] = {t.label: 0.0 for t in tasks}
    adapt_refresh_count: dict[str, int] = {t.label: 0 for t in tasks}
    sparse_adapt_seconds: dict[str, float] = {t.label: 0.0 for t in tasks}

    sweep_t0 = time.perf_counter()
    for idx, morph in enumerate(candidates):
        per_task_score: dict[str, float] = {}
        per_task_diag: dict[str, dict] = {}
        for task in tasks:
            scene_out = gen_dir / f"{task.label}_{morph_suffix(morph)}.xml"
            write_rigid_scene_with_object_size(
                base_scene_xml=task.scene_xml,
                output_scene_xml=scene_out,
                morphology=morph,
                object_body_name=task.object_body,
                size_xyz=None,
            )
            evaluator = Phase1GraspEvaluator(
                scene_xml=scene_out,
                keyframe=task.keyframe,
                cfg=eval_cfgs[task.label],
                contact_target_set=contact_target_sets[task.label],
                backend="mujoco",
            )

            # Adaptation pass: refreshes current_ctrl[task] using THIS
            # morphology's evaluator (same contact-map objective).
            refined_ctrl, refresh_triggered, adapt_secs, _ = adapt_foundational_ctrl(
                mode=args.adapt_mode,
                candidate_idx=idx,
                interval_ctrl=current_ctrl[task.label],
                evaluator=evaluator,
                adapt_cfg=adapt_cfg,
                refresh_interval=args.fp_refresh_interval,
            )
            current_ctrl[task.label] = refined_ctrl
            if refresh_triggered:
                adapt_refresh_count[task.label] += 1
                adapt_seconds[task.label] += adapt_secs

            # Secondary sparse pass — runs every candidate with a smaller CEM
            # (mirrors run6's two-stage adapt). Same evaluator/contact targets.
            if args.sparse_adapt_mode != "none":
                refined_ctrl, _, sparse_secs, _ = adapt_foundational_ctrl(
                    mode=args.sparse_adapt_mode,
                    candidate_idx=idx,
                    interval_ctrl=current_ctrl[task.label],
                    evaluator=evaluator,
                    adapt_cfg=sparse_adapt_cfg,
                    refresh_interval=1,
                )
                current_ctrl[task.label] = refined_ctrl
                sparse_adapt_seconds[task.label] += sparse_secs

            score, diag = evaluator.evaluate(current_ctrl[task.label])
            per_task_score[task.label] = float(score)
            per_task_diag[task.label] = diag
            per_task_rows[task.label].append({
                "candidate_id": idx,
                "thumb_x": morph.thumb_x, "thumb_y": morph.thumb_y, "thumb_len": morph.thumb_len,
                "index_x": morph.index_x, "index_y": morph.index_y, "index_len": morph.index_len,
                "middle_x": morph.middle_x, "middle_y": morph.middle_y, "middle_len": morph.middle_len,
                "score": float(score),
                **{k: float(v) for k, v in diag.items()},
            })

        aggregate = float(np.mean(list(per_task_score.values())))
        cross_rows.append({
            "candidate_id": idx,
            "thumb_x": morph.thumb_x, "thumb_y": morph.thumb_y, "thumb_len": morph.thumb_len,
            "index_x": morph.index_x, "index_y": morph.index_y, "index_len": morph.index_len,
            "middle_x": morph.middle_x, "middle_y": morph.middle_y, "middle_len": morph.middle_len,
            "score_mean": aggregate,
            **{f"score_{t.label}": per_task_score[t.label] for t in tasks},
        })

        if (idx + 1) % 50 == 0 or idx == len(candidates) - 1:
            elapsed = time.perf_counter() - sweep_t0
            eta = elapsed * (len(candidates) - idx - 1) / max(1, idx + 1)
            print(
                f"[run18] [{idx+1}/{len(candidates)}] mean={aggregate:+.2f} "
                f"per-task={ {k: f'{v:+.2f}' for k,v in per_task_score.items()} }  "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )

    # 5) write CSVs
    for task in tasks:
        rows = per_task_rows[task.label]
        path = out_dir / task.label / "all_candidates.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with (out_dir / "all_candidates_multi.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cross_rows[0].keys()))
        w.writeheader()
        w.writerows(cross_rows)

    sweep_secs = time.perf_counter() - sweep_t0
    if args.adapt_mode != "none" or args.sparse_adapt_mode != "none":
        print(f"\n[run18] adapt={args.adapt_mode} interval={args.fp_refresh_interval}  "
              f"sparse={args.sparse_adapt_mode}")
        for t in tasks:
            print(f"  {t.label:30s} interval_refreshes={adapt_refresh_count[t.label]:4d} "
                  f"interval_secs={adapt_seconds[t.label]:.1f} "
                  f"sparse_secs={sparse_adapt_seconds[t.label]:.1f}")
    print(f"\n[run18] DONE in {sweep_secs:.1f}s sweep (adapt={args.adapt_mode}, sparse={args.sparse_adapt_mode}).")
    print(f"        outputs under {out_dir}")


if __name__ == "__main__":
    main()
