"""Render side-by-side comparison plots for the in-hand reorientation runs.

Produces TWO multi-panel PNGs for docs/rl/reorientation.md:

  reorient_comparison.png      — the historical journey: v2 → v5 → Policy B v1
  reorient_comparison_v2.png   — the v2-era zoom: Policy B v1 → v2 variants (and beyond)

Splitting keeps each figure legible: the historical one shows how the task got
solved; the v2 one zooms in on the smooth-&-quick finetune sweep where all the
curves are bunched near the Policy B baseline. New v2+ runs should be appended to
V2_RUNS only (the historical figure is frozen).

Usage:
  uv run python scripts/rl_plot_reorient.py            # both figures
  uv run python scripts/rl_plot_reorient.py --only v2  # just reorient_comparison_v2.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / "results" / "rl"

# Historical journey (frozen — this is the original reorient_comparison.png set).
HISTORICAL_RUNS = [
    ("v2",         RL / "20260529-1929-inhand_reorient_v2_fingeronly", "tab:gray"),
    ("v3",         RL / "20260529-2311-inhand_reorient_v3",            "tab:olive"),
    ("v4",         RL / "20260530-2159-inhand_reorient_v4",            "tab:orange"),
    ("v5",         RL / "20260531-1739-inhand_reorient_v5",            "tab:red"),
    ("PolicyB v1", RL / "20260601-1033-policyB_v1",                    "tab:green"),
]

# v2 era: start at Policy B v1, then the v2 variants (and beyond — append here).
V2_RUNS = [
    ("PolicyB v1",      RL / "20260601-1033-policyB_v1",                  "tab:green"),
    ("v2 s1-5x",        RL / "20260601-2310-policyB_v2_smooth5x",         "tab:blue"),
    ("v2 s1-10x",       RL / "20260601-2311-policyB_v2_smooth10x",        "tab:cyan"),
    ("v2 s2-5x-quick",  RL / "20260602-0024-policyB_v2_smooth5x_quick",   "tab:purple"),
    ("v2 s2-10x-quick", RL / "20260602-0024-policyB_v2_smooth10x_quick",  "magenta"),
]

# Panel sets. The historical figure tracks floor_proximity (the v4/v5 story);
# the v2 figure swaps in alignment_success (the Stage-2 "quick" mechanism).
_BASE_TAGS = [
    ("Train/mean_reward",                    "mean episode reward",           "reward"),
    ("Train/mean_episode_length",            "mean episode length (steps)",   "ep_len"),
    ("Episode_Reward/target_axis_alignment", "Σ target_axis_alignment",       "align"),
    ("Episode_Reward/target_axis_progress",  "Σ target_axis_progress (Δ)",    "progress"),
    ("Metrics/lift_height/object_height",    "object_height (m)",             "obj_z"),
    ("Episode_Reward/contact_min",           "Σ contact_min",                 "contact"),
    ("Episode_Reward/action_rate_l2",        "Σ action_rate_l2 (penalty)",    "action_rate"),
    ("Episode_Reward/object_ang_acc_l2",     "Σ object_ang_acc_l2 (penalty)", "ang_acc"),
    ("Episode_Termination/tip_lost",         "tip_lost terminations/iter",    "tip_lost"),
]
_TAIL_TAGS = [
    ("Policy/mean_std",        "action std", "std"),
    ("Loss/value_function",    "value loss", "vloss"),
]
HIST_TAGS = _BASE_TAGS + [
    ("Episode_Termination/object_floor_proximity", "floor_proximity terms/iter", "floor"),
] + _TAIL_TAGS
V2_TAGS = _BASE_TAGS + [
    ("Episode_Termination/alignment_success", "alignment_success terms/iter (quick)", "align_succ"),
] + _TAIL_TAGS


def load_run(run_dir: Path, tags) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    tb = run_dir / "tensorboard"
    if not tb.exists():
        return {}
    ea = EventAccumulator(str(tb), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags()["scalars"])
    out = {}
    for tag, _, _ in tags:
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


def make_figure(runs, tags, title, out_path):
    data = {label: load_run(d, tags) for label, d, _ in runs}
    ncols = 3
    nrows = (len(tags) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, (tag, ylabel, _slug) in enumerate(tags):
        ax = axes[i]
        any_drawn = False
        for label, _d, color in runs:
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

    for j in range(len(tags), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14, y=1.00)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["historical", "v2"], default=None,
                    help="generate just one figure (default: both)")
    args = ap.parse_args()
    img = ROOT / "docs" / "rl" / "img"
    if args.only in (None, "historical"):
        make_figure(HISTORICAL_RUNS, HIST_TAGS,
                    "In-hand reorientation — the journey: v2 → v3 → v4 → v5 → Policy B v1",
                    img / "reorient_comparison.png")
    if args.only in (None, "v2"):
        make_figure(V2_RUNS, V2_TAGS,
                    "Policy B v2 finetune: v1 → Stage-1 (5x/10x smooth) → Stage-2 (quick)",
                    img / "reorient_comparison_v2.png")


if __name__ == "__main__":
    main()
