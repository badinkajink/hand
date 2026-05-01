from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run6 analysis: embeddings and feature-metric plots for sampled morphologies."
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Run directories produced by run6_screwdriver_multikey_sampling.py",
    )
    parser.add_argument(
        "--keyframes",
        nargs="+",
        default=["open_flat", "open_vertical", "open_90vertical"],
    )
    parser.add_argument("--embedding", choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["cube_xy_drift", "cube_yaw_drift", "cube_axis_tilt", "cube_ang_drift", "finger_flex_drift"],
    )
    parser.add_argument("--output-subdir", default="analysis")
    return parser


def _read_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get(key, 0.0)))
        except (TypeError, ValueError):
            vals.append(0.0)
    return np.asarray(vals, dtype=np.float64)


def _embedding(features: np.ndarray, method: str, perplexity: float, seed: int) -> np.ndarray:
    if len(features) < 3:
        return np.column_stack([features[:, 0], np.zeros(len(features))])

    if method == "umap":
        try:
            import umap  # type: ignore

            model = umap.UMAP(n_components=2, random_state=seed)
            return np.asarray(model.fit_transform(features), dtype=np.float64)
        except Exception:
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE

        p = max(2.0, min(perplexity, float(max(2, len(features) - 1))))
        model = TSNE(n_components=2, perplexity=p, random_state=seed, init="pca")
        return np.asarray(model.fit_transform(features), dtype=np.float64)

    raise ValueError(f"Unsupported method: {method}")


def _plot_embedding(coords: np.ndarray, color: np.ndarray, out_path: Path, title: str, cbar_label: str) -> None:
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=color, cmap="viridis", s=24, alpha=0.9)
    plt.title(title)
    plt.xlabel("Embedding-1")
    plt.ylabel("Embedding-2")
    plt.grid(alpha=0.25)
    plt.colorbar(sc, label=cbar_label)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_thumb_heat(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    out_path_2d: Path,
    out_path_3d: Path,
    title_base: str,
    metric_name: str,
) -> None:
    plt.figure(figsize=(7, 5))
    hb = plt.hexbin(x, y, C=z, gridsize=25, reduce_C_function=np.mean, cmap="plasma")
    plt.xlabel("thumb_x")
    plt.ylabel("thumb_y")
    plt.title(f"{title_base} - {metric_name} (2D feature heat)")
    plt.colorbar(hb, label=metric_name)
    plt.tight_layout()
    out_path_2d.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_2d, dpi=180)
    plt.close()

    if len(np.unique(x)) < 3 or len(np.unique(y)) < 3:
        return

    tri = Triangulation(x, y)
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(tri, z, cmap="plasma", linewidth=0.0, antialiased=True, alpha=0.95)
    ax.set_xlabel("thumb_x")
    ax.set_ylabel("thumb_y")
    ax.set_zlabel(metric_name)
    ax.set_title(f"{title_base} - {metric_name} (3D surface)")
    fig.colorbar(surf, ax=ax, shrink=0.7, aspect=12, label=metric_name)
    plt.tight_layout()
    out_path_3d.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_3d, dpi=180)
    plt.close()


def _mode_name(run_dir: Path) -> str:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        fp_mode = payload.get("fp_adaptation")
        if fp_mode:
            return str(fp_mode)
    return run_dir.name


def _load_keyframe_rows(run_dir: Path, keyframe: str) -> list[dict[str, Any]]:
    # Legacy layout: run_dir/<keyframe>/all_candidates.csv
    legacy_csv = run_dir / keyframe / "all_candidates.csv"
    legacy_rows = _read_rows(legacy_csv)
    if legacy_rows:
        return legacy_rows

    # Combined layout:
    # - run_dir/all_candidates_multitask.csv: morphology + aggregate stats
    # - run_dir/all_task_results.csv: per-keyframe task metrics per candidate
    morph_rows = _read_rows(run_dir / "all_candidates_multitask.csv")
    task_rows = _read_rows(run_dir / "all_task_results.csv")
    if not morph_rows or not task_rows:
        return []

    morph_by_id: dict[int, dict[str, Any]] = {}
    for r in morph_rows:
        try:
            cid = int(r.get("candidate_id", -1))
        except (TypeError, ValueError):
            continue
        morph_by_id[cid] = r

    merged: list[dict[str, Any]] = []
    for r in task_rows:
        if str(r.get("keyframe", "")) != keyframe:
            continue
        try:
            cid = int(r.get("candidate_id", -1))
        except (TypeError, ValueError):
            continue
        m = morph_by_id.get(cid)
        if m is None:
            continue
        merged.append(
            {
                "candidate_id": cid,
                "thumb_x": m.get("thumb_x", 0.0),
                "thumb_y": m.get("thumb_y", 0.0),
                "thumb_len": m.get("thumb_len", 0.0),
                "index_x": m.get("index_x", 0.0),
                "index_y": m.get("index_y", 0.0),
                "index_len": m.get("index_len", 0.0),
                "middle_x": m.get("middle_x", 0.0),
                "middle_y": m.get("middle_y", 0.0),
                "middle_len": m.get("middle_len", 0.0),
                "score": r.get("score", 0.0),
                "feasible": r.get("feasible", False),
                "cube_xy_drift": r.get("cube_xy_drift", 0.0),
                "cube_yaw_drift": r.get("cube_yaw_drift", 0.0),
                "cube_axis_tilt": r.get("cube_axis_tilt", 0.0),
                "cube_ang_drift": r.get("cube_ang_drift", 0.0),
                "finger_flex_drift": r.get("finger_flex_drift", 0.0),
            }
        )

    return merged


def main() -> None:
    args = build_parser().parse_args()

    mode_stats: list[dict[str, Any]] = []

    for run_dir in args.run_dirs:
        mode = _mode_name(run_dir)
        out_root = run_dir / args.output_subdir

        for keyframe in args.keyframes:
            rows = _load_keyframe_rows(run_dir, keyframe)
            if not rows:
                continue

            features = np.column_stack(
                [
                    _to_float(rows, "thumb_x"),
                    _to_float(rows, "thumb_y"),
                    _to_float(rows, "thumb_len"),
                    _to_float(rows, "index_x"),
                    _to_float(rows, "index_y"),
                    _to_float(rows, "index_len"),
                    _to_float(rows, "middle_x"),
                    _to_float(rows, "middle_y"),
                    _to_float(rows, "middle_len"),
                ]
            )

            score = _to_float(rows, "score")
            coords = _embedding(features, args.embedding, args.perplexity, args.seed)
            _plot_embedding(
                coords=coords,
                color=score,
                out_path=out_root / keyframe / f"embedding_{args.embedding}_score.png",
                title=f"{mode} | {keyframe} morphology embedding",
                cbar_label="score",
            )

            thumb_x = _to_float(rows, "thumb_x")
            thumb_y = _to_float(rows, "thumb_y")
            for metric in args.metrics:
                metric_values = _to_float(rows, metric)
                _plot_thumb_heat(
                    x=thumb_x,
                    y=thumb_y,
                    z=metric_values,
                    out_path_2d=out_root / keyframe / f"thumbxy_{metric}_2d.png",
                    out_path_3d=out_root / keyframe / f"thumbxy_{metric}_3d.png",
                    title_base=f"{mode} | {keyframe}",
                    metric_name=metric,
                )

            feasible = sum(1 for r in rows if str(r.get("feasible", "")).lower() == "true")
            mode_stats.append(
                {
                    "mode": mode,
                    "keyframe": keyframe,
                    "total": len(rows),
                    "feasible": feasible,
                    "feasible_rate": float(feasible) / max(1, len(rows)),
                    "mean_score": float(np.mean(score)) if len(score) else 0.0,
                    "max_score": float(np.max(score)) if len(score) else 0.0,
                }
            )

    if mode_stats:
        summary_md = [
            "# Run6 Analysis Summary",
            "",
            "| mode | keyframe | total | feasible | feasible_rate | mean_score | max_score |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in mode_stats:
            summary_md.append(
                "| {mode} | {keyframe} | {total} | {feasible} | {feasible_rate:.3f} | {mean_score:.4f} | {max_score:.4f} |".format(
                    **r
                )
            )

        # Write to the first run dir for convenience.
        target = args.run_dirs[0] / args.output_subdir / "run6_analysis_summary.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(summary_md) + "\n", encoding="utf-8")
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
