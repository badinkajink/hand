from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from morphohand.tools.morphology_xml import MorphologyValues


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def morph_row_fields(m: MorphologyValues) -> dict[str, float]:
    """Standard morphology columns for CSV rows."""
    return {
        "thumb_x": float(m.thumb_x),
        "thumb_y": float(m.thumb_y),
        "thumb_len": float(m.thumb_len),
        "index_x": float(m.index_x),
        "index_y": float(m.index_y),
        "index_len": float(m.index_len),
        "middle_x": float(m.middle_x),
        "middle_y": float(m.middle_y),
        "middle_len": float(m.middle_len),
    }


def plot_feasible_scatter(
    feasible_rows: list[dict[str, object]],
    pareto_indices: list[int],
    out_dir: Path,
    title: str = "Feasible Morphologies",
) -> None:
    """Two standard plots: lift vs tip-distance colored by score (w/ pareto ring),
    and lift vs contacts (w/ pareto ring)."""
    if not feasible_rows:
        return

    lift = np.asarray([float(r["cube_lift"]) for r in feasible_rows], dtype=np.float64)
    dist = np.asarray([float(r["mean_tip_distance"]) for r in feasible_rows], dtype=np.float64)
    contacts = np.asarray([float(r["cube_tip_contacts"]) for r in feasible_rows], dtype=np.float64)
    score = np.asarray([float(r["score"]) for r in feasible_rows], dtype=np.float64)

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.8))
    sc = plt.scatter(dist, lift, c=score, s=28, alpha=0.85, cmap="viridis")
    if pareto_indices:
        plt.scatter(
            dist[pareto_indices],
            lift[pareto_indices],
            facecolors="none",
            edgecolors="red",
            s=80,
            linewidths=1.2,
            label="Pareto",
        )
        plt.legend()
    plt.xlabel("Mean tip distance (lower is better)")
    plt.ylabel("Lift")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.colorbar(sc, label="Score")
    plt.tight_layout()
    plt.savefig(out_dir / "feasible_scatter_lift_vs_distance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.scatter(contacts, lift, c=score, s=28, alpha=0.85, cmap="plasma")
    if pareto_indices:
        plt.scatter(
            contacts[pareto_indices],
            lift[pareto_indices],
            facecolors="none",
            edgecolors="black",
            s=80,
            linewidths=1.2,
        )
    plt.xlabel("Cube/object contacts")
    plt.ylabel("Lift")
    plt.title(f"{title}: Lift vs Contacts")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "feasible_scatter_lift_vs_contacts.png", dpi=180)
    plt.close()
