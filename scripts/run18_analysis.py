"""Analysis for Run18 multi-object morphology sweep.

Inputs:
  <run_dir>/all_candidates_multi.csv        (one row per morphology, score per task + mean)
  <run_dir>/<task>/all_candidates.csv       (per-task full diagnostics)

Outputs under <run_dir>/analysis/:

  per_task/<task>_tsne.png                  TSNE of morphology dims, coloured by task score
  per_task/<task>_heatmap_<dim_pair>.png    Hexbin of score vs (dim_pair) for each task
  landscape/<task>_diff_<dim_pair>.png      score_task - score_mean heatmap per task
  similarity/spearman_matrix.png             Spearman rank correlation across tasks
  similarity/spearman_matrix.csv             same data as CSV
  top_k/<scope>.csv                          top-K morphologies per task + cross-set
  summary.json                               numeric summary
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

MORPH_KEYS = ["thumb_x", "thumb_y", "thumb_len",
              "index_x", "index_y", "index_len",
              "middle_x", "middle_y", "middle_len"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float_col(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(r.get(key, "0")) for r in rows], dtype=np.float64)


def _morph_matrix(rows: list[dict]) -> np.ndarray:
    return np.column_stack([_to_float_col(rows, k) for k in MORPH_KEYS])


def _embedding_tsne(features: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    if len(features) < 4:
        return np.column_stack([features[:, 0], np.zeros(len(features))])
    from sklearn.manifold import TSNE
    p = max(2.0, min(perplexity, float(len(features) - 1)))
    model = TSNE(n_components=2, perplexity=p, random_state=seed, init="pca")
    return np.asarray(model.fit_transform(features), dtype=np.float64)


# -------- labeled-overlay TSNE ----------------------------------------------------

SUBSETS: dict[str, list[str]] = {
    "cube+prism": ["cube", "prism"],
    "screwdrivers": ["screwdriver_medium_flat", "screwdriver_medium_vertical",
                     "screwdriver_medium_90vert", "screwdriver_small_flat"],
    "screwdrivers+drill": ["screwdriver_medium_flat", "screwdriver_medium_vertical",
                           "screwdriver_medium_90vert", "screwdriver_small_flat",
                           "power_drill"],
}


def _aggregate_subset_score(rows: list[dict], task_keys: list[str]) -> np.ndarray:
    cols = []
    for k in task_keys:
        col = _to_float_col(rows, f"score_{k}")
        cols.append(col)
    return np.mean(np.column_stack(cols), axis=1) if cols else np.zeros(len(rows))


def plot_labeled_tsne(rows: list[dict], emb: np.ndarray, score_mean: np.ndarray,
                      scores_by_task: dict[str, np.ndarray], out_path: Path) -> None:
    """Background = all candidates by score_mean; overlay best per task / subset /
    cross-set as larger labeled markers."""
    fig, ax = plt.subplots(figsize=(11, 9))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=score_mean, cmap="viridis", s=7, alpha=0.45)
    fig.colorbar(sc, ax=ax, label="score_mean")

    overlays: list[tuple[str, int, str, str]] = []  # (label, row_idx, color, marker)

    # cross-set top
    best_cross_idx = int(np.argmax(score_mean))
    overlays.append((f"cross-set top (mean={score_mean[best_cross_idx]:+.1f})",
                     best_cross_idx, "white", "*"))

    # per-task top
    per_task_colors = {
        "cube": "#ff5252", "prism": "#ff9800", "power_drill": "#ffeb3b",
        "screwdriver_medium_flat": "#4caf50", "screwdriver_medium_vertical": "#26c6da",
        "screwdriver_medium_90vert": "#42a5f5", "screwdriver_small_flat": "#ab47bc",
    }
    for task, score_arr in scores_by_task.items():
        i = int(np.argmax(score_arr))
        overlays.append((f"{task} top ({score_arr[i]:+.1f})", i,
                         per_task_colors.get(task, "white"), "o"))

    # subset tops
    subset_markers = {"cube+prism": "P", "screwdrivers": "D", "screwdrivers+drill": "X"}
    subset_colors = {"cube+prism": "#e91e63", "screwdrivers": "#00bcd4", "screwdrivers+drill": "#673ab7"}
    for sub_name, task_keys in SUBSETS.items():
        agg = _aggregate_subset_score(rows, task_keys)
        i = int(np.argmax(agg))
        overlays.append((f"{sub_name} top ({agg[i]:+.1f})", i,
                         subset_colors[sub_name], subset_markers[sub_name]))

    # Render overlay markers — larger, with a thin black edge for visibility
    for label, idx, color, marker in overlays:
        ax.scatter([emb[idx, 0]], [emb[idx, 1]],
                   s=260, c=color, marker=marker,
                   edgecolors="black", linewidths=1.2, zorder=5,
                   label=label)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_title("Morphology TSNE — score_mean heatmap + best-hand overlays")
    ax.set_xlabel("tsne-1"); ax.set_ylabel("tsne-2")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# -------- per-task TSNE -----------------------------------------------------------

def plot_tsne(features: np.ndarray, score: np.ndarray, title: str, out_path: Path,
              perplexity: float = 30.0, seed: int = 0) -> None:
    emb = _embedding_tsne(features, perplexity, seed)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=score, cmap="viridis", s=10, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("tsne-1"); ax.set_ylabel("tsne-2")
    fig.colorbar(sc, ax=ax, label="score")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# -------- per-task hexbin (MAP-Elites style) -------------------------------------

DIM_PAIRS = [("thumb_x", "thumb_y"), ("index_x", "index_y"), ("middle_x", "middle_y"),
             ("thumb_len", "index_len")]


def plot_hexbin(rows: list[dict], score: np.ndarray, dx: str, dy: str, title: str, out_path: Path) -> None:
    xs = _to_float_col(rows, dx)
    ys = _to_float_col(rows, dy)
    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(xs, ys, C=score, gridsize=18, cmap="viridis", reduce_C_function=np.max)
    ax.set_title(title)
    ax.set_xlabel(dx); ax.set_ylabel(dy)
    fig.colorbar(hb, ax=ax, label="max(score)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# -------- per-task vs full-set landscape diff ------------------------------------

def plot_diff_heatmap(rows: list[dict], score_task: np.ndarray, score_mean: np.ndarray,
                      dx: str, dy: str, title: str, out_path: Path) -> None:
    xs = _to_float_col(rows, dx)
    ys = _to_float_col(rows, dy)
    diff = score_task - score_mean
    # symmetric colormap centred at 0
    vmax = float(np.max(np.abs(diff))) if len(diff) else 1.0
    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(xs, ys, C=diff, gridsize=18, cmap="RdBu_r", reduce_C_function=np.mean,
                   vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(dx); ax.set_ylabel(dy)
    fig.colorbar(hb, ax=ax, label="score_task - score_mean")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def specialization_score(score_task: np.ndarray, score_mean: np.ndarray) -> float:
    """Variance of (score_task - score_mean) over the morphology grid."""
    diff = score_task - score_mean
    return float(np.var(diff))


# -------- object similarity ------------------------------------------------------

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    from scipy.stats import spearmanr
    r, _ = spearmanr(x, y)
    return float(r) if not np.isnan(r) else 0.0


def plot_similarity(scores_by_task: dict[str, np.ndarray], out_path: Path) -> None:
    labels = list(scores_by_task.keys())
    n = len(labels)
    mat = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            mat[i, j] = spearman(scores_by_task[a], scores_by_task[b])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    ax.set_title("Spearman rank correlation of task scores")
    fig.colorbar(im, ax=ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    # also write CSV
    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for i, lab in enumerate(labels):
            w.writerow([lab] + [f"{mat[i, j]:.4f}" for j in range(n)])


# -------- top-K --------------------------------------------------------------------

def write_top_k(rows: list[dict], score_key: str, top_k: int, out_path: Path) -> list[dict]:
    """Sort rows by score_key desc, take top_k, write CSV with morph + score."""
    sorted_rows = sorted(rows, key=lambda r: -float(r.get(score_key, "0")))
    top = sorted_rows[:top_k]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        keys = ["candidate_id", score_key] + MORPH_KEYS
        w.writerow(keys)
        for r in top:
            w.writerow([r["candidate_id"], r[score_key]] + [r[k] for k in MORPH_KEYS])
    return top


# -------- main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--cross-csv", default="all_candidates_multi.csv",
                    help="Filename under --run-dir for the cross-task CSV (e.g. _filtered)")
    ap.add_argument("--per-task-csv", default="all_candidates.csv",
                    help="Filename under <run-dir>/<task>/ for per-task CSV")
    ap.add_argument("--output-subdir", default="analysis")
    ap.add_argument("--tasks", nargs="+", default=[
        "cube", "prism", "power_drill",
        "screwdriver_medium_flat", "screwdriver_medium_vertical",
        "screwdriver_medium_90vert", "screwdriver_small_flat",
    ])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cross_path = args.run_dir / args.cross_csv
    cross_rows = _read_csv(cross_path)
    print(f"loaded {len(cross_rows)} cross-task rows from {cross_path.name}")

    out_root = args.run_dir / args.output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    score_mean = _to_float_col(cross_rows, "score_mean")
    scores_by_task: dict[str, np.ndarray] = {}
    specialization: dict[str, float] = {}

    for task in args.tasks:
        task_rows = _read_csv(args.run_dir / task / args.per_task_csv)
        if not task_rows:
            print(f"[skip] {task}: no all_candidates.csv")
            continue
        score_task = _to_float_col(cross_rows, f"score_{task}")
        scores_by_task[task] = score_task
        morph_x = _morph_matrix(cross_rows)

        # TSNE
        plot_tsne(
            morph_x, score_task,
            title=f"{task}: morphology TSNE colored by score",
            out_path=out_root / "per_task" / f"{task}_tsne.png",
            perplexity=args.perplexity, seed=args.seed,
        )

        # MAP-Elites style hexbins
        for dx, dy in DIM_PAIRS:
            plot_hexbin(
                cross_rows, score_task,
                dx, dy,
                title=f"{task}: max-score over ({dx},{dy})",
                out_path=out_root / "per_task" / f"{task}_hex_{dx}_{dy}.png",
            )

        # diff vs mean
        spec = specialization_score(score_task, score_mean)
        specialization[task] = spec
        for dx, dy in DIM_PAIRS:
            plot_diff_heatmap(
                cross_rows, score_task, score_mean,
                dx, dy,
                title=f"{task} vs full-set mean ({dx},{dy})  (spec_var={spec:.2f})",
                out_path=out_root / "landscape" / f"{task}_diff_{dx}_{dy}.png",
            )

        # top-K
        write_top_k(
            cross_rows, score_key=f"score_{task}", top_k=args.top_k,
            out_path=out_root / "top_k" / f"{task}.csv",
        )

    # cross-set top-K
    write_top_k(cross_rows, score_key="score_mean", top_k=args.top_k,
                out_path=out_root / "top_k" / "_cross_set.csv")

    # similarity matrix
    plot_similarity(scores_by_task, out_root / "similarity" / "spearman_matrix.png")

    # labeled TSNE with subset overlays (single canonical TSNE for all overlays)
    morph_x_all = _morph_matrix(cross_rows)
    emb_all = _embedding_tsne(morph_x_all, args.perplexity, args.seed)
    plot_labeled_tsne(
        cross_rows, emb_all, score_mean,
        scores_by_task,
        out_root / "labeled_tsne.png",
    )

    # summary
    summary = {
        "n_candidates": len(cross_rows),
        "score_mean_distribution": {
            "min": float(np.min(score_mean)),
            "max": float(np.max(score_mean)),
            "mean": float(np.mean(score_mean)),
            "median": float(np.median(score_mean)),
        },
        "specialization_score_per_task": specialization,
        "best_morphology_cross_set": cross_rows[int(np.argmax(score_mean))]
                                      if len(cross_rows) else None,
        "best_morphology_per_task": {
            t: cross_rows[int(np.argmax(s))] for t, s in scores_by_task.items()
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote analysis under {out_root}")


if __name__ == "__main__":
    main()
