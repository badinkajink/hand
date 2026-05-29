"""Drive the multi-morphology Path-A pipeline (CEM + eval + conditional finetune).

For each picked candidate (output of multimorph_pick_candidates.py):
  1. If no foundational CEM run for it: invoke scripts/phase1_optimize_grasp.py
  2. Run scripts/rl_eval_object.py against the picked base policy
  3. If lift_success_6cm < threshold: run scripts/rl_train_cube.py with
     --init-actor-checkpoint <base_policy_ckpt> for finetune-iters.

Per docs/rl/multimorphology.md (Path A). Uses subprocess so the same
config matches what we'd run by hand, and so a failure in any single
candidate doesn't kill the rest.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def log(msg: str, log_path: Path | None) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [multimorph] {msg}"
    print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(line + "\n")


def run_cmd(cmd: list[str], log_path: Path, env: dict | None = None) -> int:
    log(f"$ {' '.join(cmd)}", log_path)
    with log_path.open("a") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                               env=({**os.environ, **env} if env else None))


def cem_for_candidate(scene_xml: Path, parent_dir: Path, log_path: Path) -> int:
    """Re-run CEM on the candidate's morphology scene. Outputs end up at
    parent_dir/foundational/. Matches the run18_final foundational hyperparams
    (24 iters, pop=40). Wall-time: ~100 s on the 4070 Ti."""
    parent_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "--extra", "rl", "--extra", "gpu", "python",
        str(ROOT / "scripts" / "phase1_optimize_grasp.py"),
        "--scene-xml", str(scene_xml),
        "--keyframe", "open_short_manual",
        "--backend", "mujoco",
        "--optimizer", "cem",
        "--iterations", "24",
        "--population", "40",
        "--output-dir", str(parent_dir),
        "--tag", "foundational",
    ]
    return run_cmd(cmd, log_path, env={"MUJOCO_GL": "egl"})


def eval_policy(ckpt: Path, foundational: Path, eval_args: dict, log_path: Path) -> dict | None:
    cmd = [
        "uv", "run", "--extra", "rl", "--extra", "gpu", "python",
        str(ROOT / "scripts" / "rl_eval_object.py"),
        "--checkpoint", str(ckpt),
        "--foundational-run", str(foundational),
        "--object-body-name", "cube",
        "--x-jitter", str(eval_args["x_jitter"]),
        "--y-jitter", str(eval_args["y_jitter"]),
        "--yaw-jitter", str(eval_args["yaw_jitter"]),
        "--finger-residual-scale", "0.5",
        "--num-envs", "64",
        "--pose-grid", "1x1x1",
    ]
    rc = run_cmd(cmd, log_path, env={"MUJOCO_GL": "egl"})
    if rc != 0:
        return None
    iter_num = ckpt.stem.split("_")[-1]
    metrics_path = ckpt.parent.parent / f"eval_{iter_num}" / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open() as f:
        return json.load(f)


def finetune_policy(base_ckpt: Path, foundational: Path, tag: str,
                    eval_args: dict, finetune_iters: int, log_path: Path) -> Path | None:
    timesteps = finetune_iters * 1024 * 24
    cmd = [
        "uv", "run", "--extra", "rl", "--extra", "gpu", "python",
        str(ROOT / "scripts" / "rl_train_cube.py"),
        "--morphology-run", str(foundational),
        "--object-body-name", "cube",
        "--tag", tag,
        "--num-envs", "1024",
        "--total-timesteps", str(timesteps),
        "--init-actor-checkpoint", str(base_ckpt),
        "--cube-spawn-x-jitter", str(eval_args["x_jitter"]),
        "--cube-spawn-y-jitter", str(eval_args["y_jitter"]),
        "--cube-spawn-yaw-jitter", str(eval_args["yaw_jitter"]),
        "--dr-anneal-iters", "0",
        "--tracking-anneal-iters", "0",
        "--tracking-final-scale", "0.0",
        "--finger-residual-scale", "0.5",
        "--finger-close-easing", "ease_out_quad",
        "--contact-gate-stability-rewards",
        "--enable-lift-terminations",
        "--object-xy-drift-weight=-30",
        "--object-orientation-drift-weight=-20",
        "--finger-drift-weight=-10",
        "--no-wandb",
    ]
    rc = run_cmd(cmd, log_path, env={"MUJOCO_GL": "egl"})
    if rc != 0:
        return None
    candidates = sorted((ROOT / "results" / "rl").glob(f"*-{tag}"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def find_latest_ckpt(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("tensorboard/model_*.pt"),
                   key=lambda p: int(re.search(r"model_(\d+)\.pt", p.name).group(1)))
    return ckpts[-1] if ckpts else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--picks-json", type=Path, required=True,
                   help="JSON output from multimorph_pick_candidates.py")
    p.add_argument("--base-policy-ckpt", type=Path, required=True,
                   help="e.g. results/rl/<cube_stable_v1>/tensorboard/model_1400.pt")
    p.add_argument("--existing-foundational-id0", type=Path, required=True,
                   help="Foundational dir for candidate_id=0 (reuse, don't rerun CEM).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to put per-candidate CEM dirs + summary.")
    p.add_argument("--log-path", type=Path, required=True)
    p.add_argument("--finetune-threshold", type=float, default=0.80,
                   help="lift_success_6cm < this triggers finetune.")
    p.add_argument("--finetune-iters", type=int, default=250)
    p.add_argument("--eval-x-jitter", type=float, default=0.02)
    p.add_argument("--eval-y-jitter", type=float, default=0.005)
    p.add_argument("--eval-yaw-jitter", type=float, default=0.52)
    args = p.parse_args()

    with args.picks_json.open() as f:
        picks = json.load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_args = dict(x_jitter=args.eval_x_jitter, y_jitter=args.eval_y_jitter,
                     yaw_jitter=args.eval_yaw_jitter)
    summary = []

    for pick in picks:
        cid = pick["candidate_id"]
        label = pick["rank_label"]
        cand_dir = args.out_dir / f"candidate_{cid:04d}_{label}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== candidate {cid} ({label}) d={pick['distance']:.4f} ===", args.log_path)

        # --- (1) Foundational CEM (or reuse for id=0) ---
        if cid == 0:
            foundational = args.existing_foundational_id0
            log(f"reusing existing foundational: {foundational}", args.log_path)
        else:
            cem_dir = cand_dir / "foundational"
            if (cem_dir / "summary.json").exists():
                log(f"CEM already done at {cem_dir}", args.log_path)
            else:
                scene_xml = Path(pick["scene_xml"])
                if not scene_xml.exists():
                    log(f"SKIP — scene XML missing: {scene_xml}", args.log_path)
                    summary.append(dict(pick, status="missing_scene"))
                    continue
                log(f"running CEM on {scene_xml.name}", args.log_path)
                rc = cem_for_candidate(scene_xml, cand_dir, args.log_path)
                if rc != 0 or not (cem_dir / "summary.json").exists():
                    log(f"CEM failed (rc={rc}); skip", args.log_path)
                    summary.append(dict(pick, status="cem_failed"))
                    continue
            foundational = cem_dir

        # --- (2) Eval the base policy ---
        log(f"evaluating base policy on candidate {cid}", args.log_path)
        metrics = eval_policy(args.base_policy_ckpt, foundational, eval_args, args.log_path)
        if metrics is None:
            log(f"eval failed; skip finetune", args.log_path)
            summary.append(dict(pick, status="eval_failed"))
            continue
        log(f"  lift_success_6cm={metrics['lift_success_6cm']:.3f}  "
            f"contact_min_hold={metrics['contact_min_hold']:.3f}", args.log_path)

        entry = dict(pick, base_metrics=metrics, status="eval_only")

        # --- (3) Conditional finetune ---
        if metrics["lift_success_6cm"] < args.finetune_threshold and cid != 0:
            tag = f"multimorph_cand{cid:04d}_{label}_ft"
            log(f"  -> below {args.finetune_threshold:.2f}: finetuning ({args.finetune_iters} iters)",
                args.log_path)
            ft_dir = finetune_policy(args.base_policy_ckpt, foundational, tag,
                                     eval_args, args.finetune_iters, args.log_path)
            if ft_dir is not None:
                ft_ckpt = find_latest_ckpt(ft_dir)
                if ft_ckpt is not None:
                    ft_metrics = eval_policy(ft_ckpt, foundational, eval_args, args.log_path)
                    entry.update(status="finetuned", finetune_dir=str(ft_dir),
                                 finetune_ckpt=str(ft_ckpt), finetune_metrics=ft_metrics)
                    if ft_metrics:
                        log(f"  finetuned lift_success_6cm={ft_metrics['lift_success_6cm']:.3f}",
                            args.log_path)
        summary.append(entry)

    (args.out_dir / "multimorph_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"multimorph pipeline DONE; summary -> {args.out_dir / 'multimorph_summary.json'}",
        args.log_path)


if __name__ == "__main__":
    main()
