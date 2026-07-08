"""Full A->B pipeline sweep — analysis plots + markdown summary (2026-07-03).

Reads MORPH_PIPELINE_<tag>.json (scripts/morph_pipeline_sweep.py) and the per-design trainer
logs, and produces the "in depth analysis" deliverables:

  1. SUMMARY figure (docs/rl/img/morph_pipeline_<tag>_summary.png): per-design held-cos ranking
     (colored by health verdict), post-handoff min-z (hold), object jitter, per-finger force
     BALANCE, de-centering drift, and the grasp-vs-reorient scatter (does the cheap grasp screen
     predict the full-pipeline reorient?).
  2. TRAINING-DYNAMICS figure (docs/rl/img/morph_pipeline_<tag>_training.png): overlay every
     design's Policy A and Policy B learning curves (reward / alignment / progress / tip_lost).
  3. Markdown SUMMARY TABLE (MORPH_PIPELINE_<tag>_TABLE.md) — paste-ready for the docs.

Pure CPU (matplotlib), no GPU / sim. Run: uv run python scripts/morph_pipeline_plots.py --tag initial8
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs/rl/img"; IMG.mkdir(parents=True, exist_ok=True)
VERDICT_COLOR = {"PASS": "#2ca02c", "WARN": "#ff7f0e", "FAIL": "#d62728", None: "#999999"}

ITER_RE = re.compile(r"Learning iteration\s+(\d+)/")
SCALARS = {
    "reward": re.compile(r"Mean reward:\s+([-\d.]+)"),
    "align": re.compile(r"target_axis_alignment:\s+([-\d.]+)"),
    "progress": re.compile(r"target_axis_progress:\s+([-\d.]+)"),
    "obj_z": re.compile(r"lift_height/object_height:\s+([-\d.]+)"),
    "tip_lost": re.compile(r"tip_lost:\s+([-\d.]+)"),
}


def parse_log(path: Path):
    text = path.read_text(errors="ignore")
    chunks = re.split(r"Learning iteration\s+\d+/", text)
    iters = [int(m.group(1)) for m in ITER_RE.finditer(text)]
    cols = {k: [] for k in SCALARS}
    for chunk in chunks[1:len(iters) + 1]:
        for k, rx in SCALARS.items():
            mm = rx.search(chunk)
            cols[k].append(float(mm.group(1)) if mm else np.nan)
    out = {"iter": np.array(iters)}
    for k in SCALARS:
        out[k] = np.array(cols[k])
    return out


def _hm(rec):  # handoff health metrics (or {})
    return (((rec.get("handoff") or {}).get("health") or {}).get("metrics") or {})


def _verdict(rec):
    return ((rec.get("handoff") or {}).get("health") or {}).get("verdict")


def summary_fig(data, tag):
    # designs that produced a handoff, sorted by held-cos desc
    d = [r for r in data if _hm(r)]
    d.sort(key=lambda r: _hm(r).get("held_cos_tail", -1), reverse=True)
    ids = [r["id"] for r in d]
    cols = [VERDICT_COLOR[_verdict(r)] for r in d]
    x = np.arange(len(d))
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    def bar(a, vals, title, ylab, hlines=()):
        a.bar(x, vals, color=cols)
        a.set_xticks(x); a.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
        a.set_title(title, fontsize=11); a.set_ylabel(ylab)
        for y, lab in hlines:
            a.axhline(y, ls="--", c="k", lw=0.8, alpha=0.6); a.text(0, y, " " + lab, fontsize=7, va="bottom")

    bar(ax[0, 0], [_hm(r).get("held_cos_tail", 0) for r in d],
        "Reorient held-cos (last-50)  [color=health verdict]", "cos", [(0.9, "vertical~0.9")])
    bar(ax[0, 1], [(r.get("handoff") or {}).get("min_z_post") or 0 for r in d],
        "Post-handoff min-z (hold)", "m", [(0.05, "floor 0.05")])
    bar(ax[0, 2], [_hm(r).get("ang_jerk", 0) for r in d],
        "Object jitter (ang-jerk)", "1/s^2", [(20, "WARN"), (40, "FAIL")])
    bar(ax[1, 0], [_hm(r).get("net_drift_cm", 0) for r in d],
        "De-centering (net lateral drift)", "cm", [(3.0, "FAIL 3cm")])
    # per-finger force balance (grouped)
    a = ax[1, 1]; w = 0.26
    fm = np.array([_hm(r).get("force_mean", [0, 0, 0]) for r in d])
    if len(fm):
        for i, (lab, c) in enumerate(zip(("thumb", "index", "middle"), ("#1f77b4", "#9467bd", "#8c564b"))):
            a.bar(x + (i - 1) * w, fm[:, i], w, label=lab, color=c)
    a.axhline(5.0, ls="--", c="k", lw=0.8, alpha=0.6); a.text(0, 5.0, " over-clamp 5N", fontsize=7)
    a.set_xticks(x); a.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    a.set_title("Per-finger force BALANCE (recruit the thumb?)"); a.set_ylabel("N"); a.legend(fontsize=8)
    # grasp screen (CEM lift) vs reorient cos — does the cheap screen predict quality?
    a = ax[1, 2]
    gl = [r.get("lift", np.nan) for r in d]; rc = [_hm(r).get("held_cos_tail", np.nan) for r in d]
    a.scatter(gl, rc, c=cols, s=60)
    for r, gx, ry in zip(d, gl, rc):
        a.annotate(r["id"].split("_")[0], (gx, ry), fontsize=7)
    a.set_xlabel("CEM cube_lift (grasp screen)"); a.set_ylabel("reorient held-cos")
    a.set_title("Grasp screen vs reorient quality")
    fig.suptitle(f"Morphology pipeline sweep '{tag}' — {len(data)} designs "
                 f"(green=PASS amber=WARN red=FAIL)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = IMG / f"morph_pipeline_{tag}_summary.png"; fig.savefig(out, dpi=110); plt.close(fig)
    return out


def training_fig(data, tag):
    fig, ax = plt.subplots(2, 4, figsize=(22, 9))
    metrics = [("reward", "A: Mean reward"), ("align", "A: alignment"),
               ("progress", "A: progress"), ("obj_z", "A: object height"),
               ("reward", "B: Mean reward"), ("align", "B: alignment"),
               ("progress", "B: progress"), ("tip_lost", "B: tip_lost")]
    cmap = plt.get_cmap("tab10")
    for r in data:
        mid = r["id"]
        for phase, cols in (("A", range(4)), ("B", range(4, 8))):
            log = ROOT / f"sweep_{phase}_{mid}.trainer.log"
            if not log.exists():
                continue
            c = parse_log(log)
            if not len(c["iter"]):
                continue
            color = cmap(hash(mid) % 10)
            for j, ci in enumerate(cols):
                key = metrics[ci][0]
                ax.flat[ci].plot(c["iter"], c[key], lw=1.2, color=color, alpha=0.8, label=mid)
    for ci, (_, title) in enumerate(metrics):
        ax.flat[ci].set_title(title, fontsize=10); ax.flat[ci].set_xlabel("iter")
    ax.flat[0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"Morphology pipeline sweep '{tag}' — training dynamics (A lift, B reorient)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = IMG / f"morph_pipeline_{tag}_training.png"; fig.savefig(out, dpi=110); plt.close(fig)
    return out


def table_md(data, tag):
    rows = ["| design | Δm05 (nonzero) | grasp lift/persist | verdict | held-cos | min-z | jerk | force t/i/m | drift | note |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    ds = sorted(data, key=lambda r: _hm(r).get("held_cos_tail", -1) if _hm(r) else -9, reverse=True)
    for r in ds:
        hm = _hm(r); ho = r.get("handoff") or {}
        AX = ("thumb_x thumb_y thumb_len index_x index_y index_len middle_x middle_y middle_len").split()
        nz = ", ".join(f"{AX[i]}{d:+.3f}" for i, d in enumerate(r.get("delta_m05", []) or []) if abs(d) > 1e-9) or "—"
        fm = hm.get("force_mean", ["—"] * 3)
        rows.append("| `{}` | {} | {}/{:.2f}·{:.2f}·{:.2f} | **{}** | {} | {} | {} | {} | {} | {} |".format(
            r["id"], nz,
            f"{r.get('lift', float('nan')):.3f}", r.get("pt", 0), r.get("pi", 0), r.get("pm", 0),
            _verdict(r) or "—",
            hm.get("held_cos_tail", "—"), ho.get("min_z_post", "—"), hm.get("ang_jerk", "—"),
            "/".join(str(v) for v in fm), hm.get("net_drift_cm", "—"),
            r.get("note") or r.get("error") or ""))
    out = ROOT / f"docs/experiments/MORPH_PIPELINE_{tag}_TABLE.md"
    out.write_text("\n".join(rows) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="initial8")
    args = ap.parse_args()
    data = json.loads((ROOT / f"docs/experiments/MORPH_PIPELINE_{args.tag}.json").read_text())
    print("summary  ->", summary_fig(data, args.tag))
    print("training ->", training_fig(data, args.tag))
    print("table    ->", table_md(data, args.tag))
    # pick the best PASS/WARN design and stash its vector for a `--center best` follow-on sweep
    ranked = [r for r in data if _hm(r) and _verdict(r) in ("PASS", "WARN")]
    ranked.sort(key=lambda r: _hm(r).get("held_cos_tail", -1), reverse=True)
    if ranked:
        best = ranked[0]
        (ROOT / "docs/experiments/MORPH_PIPELINE_best_center.json").write_text(json.dumps(best["morph"]))
        print(f"best (non-FAIL) design = {best['id']}  held-cos {_hm(best).get('held_cos_tail')}  "
              f"-> MORPH_PIPELINE_best_center.json")


if __name__ == "__main__":
    main()
