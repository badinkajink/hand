"""Render side-by-side comparison plots for the in-hand reorientation runs.

One-off script for docs/rl/reorientation.md. Reads tensorboard event files
from v2/v3/v4/v5/policyB runs and produces a multi-panel PNG.

Usage:
  uv run python scripts/rl_plot_reorient.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / "results" / "rl"

RUNS = [
    ("v2",      RL / "20260529-1929-inhand_reorient_v2_fingeronly", "tab:gray"),
    ("v3",      RL / "20260529-2311-inhand_reorient_v3",            "tab:olive"),
    ("v4",      RL / "20260530-2159-inhand_reorient_v4",            "tab:orange"),
    ("v5",      RL / "20260531-1739-inhand_reorient_v5",            "tab:red"),
    ("PolicyB", RL / "20260601-1033-policyB_v1",                    "tab:green"),
]

TAGS = [
    ("Train/mean_reward",                            "mean episode reward",          "reward"),
    ("Train/mean_episode_length",                    "mean episode length (steps)",  "ep_len"),
    ("Episode_Reward/target_axis_alignment",         "Σ target_axis_alignment",      "align"),
    ("Episode_Reward/target_axis_progress",          "Σ target_axis_progress (Δ)",   "progress"),
    ("Metrics/lift_height/object_height",            "object_height (m)",            "obj_z"),
    ("Episode_Reward/contact_min",                   "Σ contact_min",                "contact"),
    ("Episode_Reward/action_rate_l2",                "Σ action_rate_l2 (penalty)",   "action_rate"),
    ("Episode_Reward/object_ang_acc_l2",             "Σ object_ang_acc_l2 (penalty)","ang_acc"),
    ("Episode_Termination/tip_lost",                 "tip_lost terminations/iter",   "tip_lost"),
    ("Episode_Termination/object_floor_proximity",   "floor_proximity terms/iter",   "floor"),
    ("Policy/mean_std",                              "action std",                    "std"),
    ("Loss/value_function",                          "value loss",                    "vloss"),
]


def load_run(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    tb = run_dir / "tensorboard"
    if not tb.exists():
        return {}
    ea = EventAccumulator(str(tb), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags()["scalars"])
    out = {}
    for tag, _, _ in TAGS:
        if tag in available:
            ev = ea.Scalars(tag)
            steps = np.array([e.step for e in ev], dtype=float)
            vals = np.array([e.value for e in ev], dtype=float)
            out[tag] = (steps, vals)
    return out


def smooth(y: np.ndarray, w: int = 11) -> np.ndarray:
    if len(y) < w:
        return y
    kernel = np.ones(w) / w
    pad = w // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(padded, kernel, mode="valid")[:len(y)]


def main():
    data = {label: load_run(d) for label, d, _ in RUNS}
    ncols = 3
    nrows = (len(TAGS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, (tag, ylabel, _slug) in enumerate(TAGS):
        ax = axes[i]
        any_drawn = False
        for label, _d, color in RUNS:
            if tag not in data[label]:
                continue
            x, y = data[label][tag]
            ax.plot(x, smooth(y, 11), label=label, color=color, alpha=0.9, linewidth=1.6)
            any_drawn = True
        ax.set_title(ylabel, fontsize=10)
        ax.set_xlabel("PPO iter")
        ax.grid(True, alpha=0.3)
        if any_drawn:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes, ha="center", va="center", color="gray")

    for j in range(len(TAGS), len(axes)):
        axes[j].axis("off")

    fig.suptitle("In-hand reorientation runs: v2 → v3 → v4 → v5 → Policy B", fontsize=14, y=1.00)
    fig.tight_layout()

    out_dir = ROOT / "docs" / "rl" / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reorient_comparison.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"wrote {out_path}")

    # Also print which tags each run has
    print("\nAvailable tags per run:")
    for label, _, _ in RUNS:
        present = sorted(data[label].keys())
        print(f"  {label}: {len(present)} tags")


if __name__ == "__main__":
    main()
