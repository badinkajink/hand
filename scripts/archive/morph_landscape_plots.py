"""Morphology LANDSCAPE — analysis plots (2026-06-30).

Two deliverables from the 2026-06-25 landscape sweep (scripts/morph_landscape_sweep.py):

  1. TRAINING DYNAMICS across morphologies — parse every per-morphology skip-lift
     reorienter trainer log (landscapeB_landscape_B_*.trainer.log; all warmstart B4,
     15 M ts each) into per-iteration curves and overlay them. Answers: do the designs
     train *differently*, or just converge to different endpoints?
  2. LANDSCAPE SUMMARY — from MORPH_LANDSCAPE.json: reorient-cos ranking, the
     grasp-balance-does-NOT-predict-reorient scatter, and the per-finger force balance
     that separates the genuine in-hand winner (m05) from the idle-finger failures.

Outputs PNGs to docs/rl/img/. Pure CPU, no GPU / sim needed.

Run: uv run python scripts/morph_landscape_plots.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs/rl/img"
IMG.mkdir(parents=True, exist_ok=True)
JSON = ROOT / "MORPH_LANDSCAPE.json"

# the per-morphology trainer logs (skip-lift reorienter, warmstart B4, 15 M ts)
LOG_GLOB = "landscapeB_landscape_B_*.trainer.log"

ITER_RE = re.compile(r"Learning iteration\s+(\d+)/")
SCALARS = {
    "reward": re.compile(r"Mean reward:\s+([-\d.]+)"),
    "ep_len": re.compile(r"Mean episode length:\s+([-\d.]+)"),
    "action_std": re.compile(r"Mean action std:\s+([-\d.]+)"),
    "value_loss": re.compile(r"Mean value loss:\s+([-\d.]+)"),
    "align": re.compile(r"target_axis_alignment:\s+([-\d.]+)"),
    "progress": re.compile(r"target_axis_progress:\s+([-\d.]+)"),
    "obj_z": re.compile(r"lift_height/object_height:\s+([-\d.]+)"),
    "tip_lost": re.compile(r"Termination/tip_lost:\s+([-\d.]+)"),
    "floor_prox": re.compile(r"Termination/object_floor_proximity:\s+([-\d.]+)"),
    "drop": re.compile(r"Episode_Termination/object_drop:\s+([-\d.]+)"),
}


def parse_log(path: Path) -> dict[str, np.ndarray]:
    """One block per learning iteration -> arrays keyed by SCALARS + 'iter'."""
    text = path.read_text(errors="ignore")
    # split on the iteration header so each chunk holds exactly one iteration's scalars
    chunks = re.split(r"Learning iteration\s+\d+/", text)
    iters, cols = [], {k: [] for k in SCALARS}
    it = 0
    for m in ITER_RE.finditer(text):
        iters.append(int(m.group(1)))
    for chunk, it in zip(chunks[1:], iters):
        for k, rx in SCALARS.items():
            mm = rx.search(chunk)
            cols[k].append(float(mm.group(1)) if mm else np.nan)
    out = {"iter": np.array(iters)}
    for k in SCALARS:
        out[k] = np.array(cols[k])
    return out


def morph_id(path: Path) -> str:
    # landscapeB_landscape_B_m05_lhs.trainer.log -> m05_lhs
    return path.name.replace("landscapeB_landscape_B_", "").replace(".trainer.log", "")


# verdict labels + colors from the landscape result (MORPH_LANDSCAPE.txt)
VERDICT = {
    "m05_lhs":       ("m05 — genuine in-hand winner (cos 0.93, balanced)", "#1b9e77", 2.6),
    "m03_lhs":       ("m03 — floor-braced (cos 0.90, ~0 contact)",          "#d95f02", 1.8),
    "m11_lhs":       ("m11 — partial (cos 0.36)",                           "#7570b3", 1.4),
    "m07_lhs":       ("m07 — partial (cos 0.24, 1-finger)",                 "#e7298a", 1.4),
    "m00_baseline":  ("m00 — baseline (cos -0.30)",                         "#999999", 1.4),
    "m06_lhs":       ("m06 — idle fingers (cos -0.15)",                     "#66a61e", 1.2),
    "m08_lhs":       ("m08 — dropped early (cos -0.01)",                    "#a6761d", 1.2),
    "m01_thumbWinner": ("m01 — thumb-opposition grasp, reorient FAILS (cos -0.68)", "#e6194b", 1.8),
}


def plot_training_dynamics():
    logs = sorted(ROOT.glob(LOG_GLOB))
    runs = {}
    for p in logs:
        mid = morph_id(p)
        d = parse_log(p)
        if len(d["iter"]) < 5:  # m10 aborted at iter 0
            continue
        runs[mid] = d

    panels = [
        ("reward", "Mean reward", False),
        ("align", "target_axis_alignment (Σ ep)", False),
        ("progress", "target_axis_progress (Σ ep)", False),
        ("obj_z", "object height (m)", False),
        ("ep_len", "mean episode length", False),
        ("action_std", "mean action std", False),
        ("tip_lost", "tip_lost / iter", False),
        ("floor_prox", "object_floor_proximity / iter", False),
        ("value_loss", "value loss (log)", True),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    axes = axes.ravel()
    # consistent ordering: winner-ish first
    order = [m for m in VERDICT if m in runs] + [m for m in runs if m not in VERDICT]
    for key, title, logy in panels:
        ax = axes[panels.index((key, title, logy))]
        for mid in order:
            d = runs[mid]
            label, color, lw = VERDICT.get(mid, (mid, None, 1.2))
            y = d[key]
            if logy:
                y = np.clip(y, 1e-3, None)
            ax.plot(d["iter"], y, label=label, color=color, lw=lw, alpha=0.9)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25)
    # one shared legend
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-morphology reorienter training dynamics — skip-lift Policy B, "
                 "warmstart B4, 15 M ts each\n(8 graspable designs; identical recipe, "
                 "morphology is the only variable)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = IMG / "morph_landscape_training.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[wrote] {out}")
    plt.close(fig)
    return runs


def plot_landscape_summary():
    data = json.loads(JSON.read_text())
    rec = {d["id"]: d for d in data}

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    # ---- panel 1: reorient-cos ranking (bar) ----
    ax = axes[0]
    rows = []
    for d in data:
        r = d.get("reorient") or {}
        cos = r.get("held_cos")
        if cos is None:
            cos = np.nan if d.get("note") else None
        rows.append((d["id"], d.get("lift", np.nan), cos, d.get("note", "")))
    rows = [r for r in rows]
    # sort by cos (nan/ungraspable last)
    def cos_key(r):
        return -1e9 if (r[2] is None or np.isnan(r[2])) else r[2]
    rows.sort(key=cos_key, reverse=True)
    ids = [r[0].replace("_lhs", "").replace("_baseline", "") for r in rows]
    coss = [(r[2] if (r[2] is not None and not np.isnan(r[2])) else 0.0) for r in rows]
    colors = []
    for r in rows:
        c = r[2]
        if c is None or np.isnan(c):
            colors.append("#cccccc")           # ungraspable
        elif c >= 0.85:
            colors.append("#1b9e77")           # genuine reorient
        elif c >= 0.2:
            colors.append("#d9b310")           # partial
        else:
            colors.append("#e6194b")           # failed
    ax.barh(range(len(ids)), coss, color=colors)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(0.85, color="#1b9e77", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("reorient held-cos (probe; ungraspable=grey 0)")
    ax.set_title("Reorient quality VARIES strongly across the landscape\n"
                 "(−0.68 … +0.93 over 12 designs)", fontsize=11)
    ax.grid(axis="x", alpha=0.25)

    # ---- panel 2: grasp-balance does NOT predict reorient ----
    ax = axes[1]
    for d in data:
        r = d.get("reorient") or {}
        cos = r.get("held_cos")
        lift = d.get("lift", np.nan)
        imbal = d.get("g_imbal", np.nan)
        if cos is None or np.isnan(lift) or lift < 0.02:
            continue  # ungraspable
        # grasp persistence balance: lower g_imbal = more balanced grasp
        ax.scatter(imbal, cos, s=90, zorder=3,
                   color="#1b9e77" if cos >= 0.85 else ("#d9b310" if cos >= 0.2 else "#e6194b"),
                   edgecolor="k", lw=0.6)
        tag = d["id"].replace("_lhs", "").replace("_baseline", "")
        ax.annotate(tag, (imbal, cos), fontsize=8.5, xytext=(5, 3),
                    textcoords="offset points")
    ax.set_xlabel("grasp persistence imbalance  (0 = perfectly balanced grasp)")
    ax.set_ylabel("reorient held-cos")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("GRASP balance does NOT predict REORIENT quality\n"
                 "(m01: perfect grasp balance → reorient fails; "
                 "m05: poor grasp persist → reorient great)", fontsize=11)
    ax.grid(alpha=0.25)

    # ---- panel 3: per-finger reorient force balance ----
    ax = axes[2]
    show = [d for d in data if (d.get("reorient") or {}).get("held_cos") is not None]
    show.sort(key=lambda d: -(d["reorient"]["held_cos"]))
    labels, T, I, M = [], [], [], []
    for d in show:
        r = d["reorient"]
        labels.append(d["id"].replace("_lhs", "").replace("_baseline", "")
                      + f"\ncos {r['held_cos']:.2f}")
        T.append(r.get("thumbF", 0)); I.append(r.get("indexF", 0)); M.append(r.get("midF", 0))
    x = np.arange(len(labels)); w = 0.27
    ax.bar(x - w, T, w, label="thumb", color="#1f77b4")
    ax.bar(x,     I, w, label="index", color="#ff7f0e")
    ax.bar(x + w, M, w, label="middle", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("per-finger contact force (N)")
    ax.legend(fontsize=9)
    ax.set_title("Per-finger reorient force: only m05 loads ALL THREE\n"
                 "(others idle ≥1 finger → degenerate pinch / floor-brace)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Morphology landscape sweep (2026-06-25) — 12 designs, "
                 "CEM grasp + skip-lift reorienter per design", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = IMG / "morph_landscape_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[wrote] {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_training_dynamics()
    plot_landscape_summary()
