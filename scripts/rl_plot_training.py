"""Read PPO tensorboard logs for one or more runs and render diagnostic plots.

Usage:
  uv run --extra rl python scripts/rl_plot_training.py \
    --run results/rl/20260527-1825-cube_dr_curriculum \
    [--run results/rl/20260528-1448-prism_stable ...] \
    [--out plots/]

Generates one plot file per run plus an `_overlay.png` comparing runs on the
main reward / stability / std curves. Annotates the iter at which the DR
curriculum reaches full strength (Curriculum/dr_anneal == 1.0).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


# TB tag -> (subplot key, display label).
# Subplot keys are mapped to a layout in `LAYOUT` below.
TAG_MAP = {
    "Train/mean_reward":                       ("reward", "mean reward"),
    "Train/mean_episode_length":               ("episode", "mean episode length"),
    "Curriculum/dr_anneal":                    ("dr",     "DR anneal progress"),
    "Policy/mean_std":                         ("std",    "action std"),
    "Loss/entropy":                            ("std",    "entropy"),
    "Metrics/lift_height/object_height":       ("lift",   "object_height (m)"),
    "Metrics/lift_height/episode_success":     ("lift",   "episode success"),
    "Episode_Reward/contact_min":              ("contact","Σ contact_min (reward)"),
    "Episode_Reward/contact_mean":             ("contact","Σ contact_mean (reward)"),
    "Episode_Reward/lift_height":              ("lift",   "Σ lift_height (reward)"),
    "Episode_Reward/object_xy_drift":          ("drift",  "Σ xy_drift"),
    "Episode_Reward/object_orientation_drift": ("drift",  "Σ orientation_drift"),
    "Episode_Reward/finger_drift_from_grip":   ("drift",  "Σ finger_drift"),
    "Episode_Reward/object_drop":              ("drift",  "Σ drop"),
    "Episode_Reward/track_finger_qpos":        ("track",  "Σ track_finger_qpos"),
    "Episode_Reward/track_object_pos":         ("track",  "Σ track_object_pos"),
    "Episode_Reward/track_object_quat":        ("track",  "Σ track_object_quat"),
    "Episode_Reward/track_finger_ctrl_anchor": ("track",  "Σ track_finger_ctrl_anchor"),
}

LAYOUT = [
    ("reward",  "Mean reward"),
    ("lift",    "Lift / object height"),
    ("contact", "Contact rewards"),
    ("drift",   "Stability penalties (signed)"),
    ("track",   "Tracking rewards"),
    ("std",     "Exploration (std / entropy)"),
    ("dr",      "DR anneal progress"),
    ("episode", "Episode length / success"),
]


def load_run(run_dir: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int | None]:
    """Return {tag: (steps, values)} for one run + the iter at which
    DR anneal first hit 1.0 (None if no DR curriculum)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb_dir = run_dir / "tensorboard"
    if not tb_dir.exists():
        raise FileNotFoundError(f"no tensorboard/ under {run_dir}")
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag in TAG_MAP:
        if tag not in tags:
            continue
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=float)
        vals = np.array([e.value for e in events], dtype=float)
        out[tag] = (steps, vals)

    dr_full_iter = None
    if "Curriculum/dr_anneal" in out:
        s, v = out["Curriculum/dr_anneal"]
        hit = np.where(v >= 0.999)[0]
        if len(hit):
            dr_full_iter = int(s[hit[0]])
    return out, dr_full_iter


def render_one_run(run_dir: Path, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series, dr_full = load_run(run_dir)

    n = len(LAYOUT)
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    axes_flat = axes.flatten()
    subplot_by_key = {k: ax for (k, _), ax in zip(LAYOUT, axes_flat)}
    for (_, title), ax in zip(LAYOUT, axes_flat):
        ax.set_title(title)
        ax.grid(alpha=0.3)

    for tag, (steps, vals) in series.items():
        key, label = TAG_MAP[tag]
        ax = subplot_by_key[key]
        ax.plot(steps, vals, label=label, linewidth=1.2)
        ax.legend(fontsize=7, loc="best")

    if dr_full is not None:
        for ax in axes_flat:
            ax.axvline(dr_full, color="purple", linestyle=":", linewidth=1,
                       label=f"DR full @ iter {dr_full}")

    fig.suptitle(f"{run_dir.name}    (DR full = {dr_full})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  -> {out_path}")


def render_overlay(runs: list[tuple[Path, dict, int | None]], out_path: Path) -> None:
    """Side-by-side comparison: mean reward + lift_height + std + drift, one
    line per run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Train/mean_reward",                "Mean reward"),
        ("Metrics/lift_height/object_height","Object height (m)"),
        ("Policy/mean_std",                  "Action std"),
        ("Episode_Reward/object_xy_drift",   "Σ xy_drift (reward, signed)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for (tag, title), ax in zip(panels, axes.flatten()):
        ax.set_title(title)
        ax.grid(alpha=0.3)
        for run_dir, series, dr_full in runs:
            if tag in series:
                steps, vals = series[tag]
                ax.plot(steps, vals, label=run_dir.name, linewidth=1.2)
            if dr_full is not None:
                ax.axvline(dr_full, linestyle=":", alpha=0.4)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, action="append", required=True,
                   help="One or more results/rl/<tag>/ dirs. Repeat for overlay.")
    p.add_argument("--out", type=Path, default=Path("plots/rl"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[plot] rendering {len(args.run)} run(s) -> {args.out}/")
    loaded = []
    for run_dir in args.run:
        run_dir = run_dir.resolve()
        print(f"[plot] {run_dir.name}")
        series, dr_full = load_run(run_dir)
        loaded.append((run_dir, series, dr_full))
        render_one_run(run_dir, args.out / f"{run_dir.name}.png")

    if len(loaded) > 1:
        render_overlay(loaded, args.out / "_overlay.png")
    print(f"[plot] DONE")


if __name__ == "__main__":
    main()
