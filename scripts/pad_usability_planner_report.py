#!/usr/bin/env python3
"""Scores, table and figure of experiment U4 of the 2026-10-14 pad-usability study: pinned iCEM planning on the 1 mm
pad plant with its own pad model (icem_pin), with point contact (icem_pin_pt) and with condim 4 (icem_pin_pt4).

Terms
  pair          one cell (layout and cylinder) and one seed (placement jitter and the planner's random stream); runs
                are deterministic, so the planners are paired on the same 16 pairs (4 cells x seeds 0-3)
  held turn     the end turn if the cylinder ends less than 20 mm below its post-grip height with two fingers each
                pressing at least its weight, else 0 (hand_object_scale_mpc.mpc_turn)
  paired change held turn of a planner minus the pads planner's on the same pair, deg; median with a cell-cluster
                bootstrap 95 % interval (cells resampled with their seeds, 10,000 resamples, generator seed 0) and an
                exact two-sided Wilcoxon signed-rank test
  three-finger ending  three_frac >= 0.99: all three fingertips loaded over the last second; exact McNemar test against
                the pads planner
  replanning time  planner seconds per replanning, plan_s / 275, at the run's thread count

    .venv/bin/python scripts/pad_usability_planner_report.py
Reads docs/experiments/20261014-pad_usability/planner/turn.jsonl and the icem_pin rows of
docs/experiments/20261013-scale_turn_controllers/planners/turn.jsonl (seeds 0 and 1, 6 threads); writes
planner/summary_U4.json and planner/20261014-u4_planner_models.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pad_usability_agree as AG  # noqa: E402

PL = ROOT / "docs/experiments/20261014-pad_usability/planner"
OLD = ROOT / "docs/experiments/20261013-scale_turn_controllers/planners/turn.jsonl"
CELLS = [("cylinder", 25.0, 0.125), ("cylinder", 16.0, 0.625), ("cylinder", 10.0, 0.75), ("cylinder", 16.0, -0.375)]
CONFIGS = [("icem_pin", "Pads planner", "#2a78d6", "o"), ("icem_pin_pt4", "Condim-4 planner", "#eda100", "D"),
           ("icem_pin_pt", "Point-contact planner", "#1baf7a", "^")]
SEEDS = (0, 1, 2, 3)
N_REPLAN = 275


def rows():
    new = [r for r in AG.jl(PL / "turn.jsonl") if r.get("status") == "ok" and not r.get("variant")]
    old = [r for r in AG.jl(OLD) if r.get("status") == "ok" and r.get("config") == "icem_pin" and not r.get("variant")]
    by = {}
    for r in old:
        by[("icem_pin", r["shape"], r["d_mm"], r["t"], r["seed"])] = dict(r, source="20261013")
    repro = None
    for r in new:
        k = (r["config"], r["shape"], r["d_mm"], r["t"], r["seed"])
        if k in by and r["config"] == "icem_pin":
            repro = dict(old=by[k]["turn_end_deg"], new=r["turn_end_deg"], cell=k[1:4], seed=k[4],
                         plan_s_new=r["plan_s"], nthread_new=r.get("prm_nthread"))
            by[k] = dict(by[k], plan_s_3thread=r["plan_s"])      # keep the 2026-10-13 row; record the 3-thread time
            continue
        by[k] = dict(r, source="20261014")
    return by, repro


def boot_cluster(d_by_cell, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    cells = list(d_by_cell)
    meds = []
    for _ in range(n):
        pick = rng.integers(0, len(cells), len(cells))
        meds.append(np.median(np.concatenate([d_by_cell[cells[i]] for i in pick])))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main() -> int:
    from scipy.stats import wilcoxon
    by, repro = rows()
    out = dict(id="U4", reproduction=repro, configs=[])
    pads = {(c, s): by.get(("icem_pin",) + c + (s,)) for c in CELLS for s in SEEDS}
    for cfg, lab, _, _ in CONFIGS:
        got = {(c, s): by.get((cfg,) + c + (s,)) for c in CELLS for s in SEEDS}
        have = [k for k in got if got[k] is not None and pads[k] is not None]
        three = [bool(got[k]["three_frac"] >= 0.99) for k in have]
        held = [bool(got[k]["held"]) for k in have]
        rep = dict(config=cfg, label=lab, n_pairs=len(have), held=sum(held), three_finger=sum(three),
                   held_turn_median=float(np.median([got[k]["held_turn_deg"] for k in have])) if have else None,
                   contact_breaks_median=float(np.median([got[k]["contact_breaks"] for k in have])) if have else None,
                   cmd_rate_rms_median=float(np.median([got[k]["cmd_rate_rms"] for k in have])) if have else None,
                   omega_jitter_median=float(np.median([got[k]["omega_jitter"] for k in have])) if have else None,
                   peak_rate_median=float(np.median([got[k]["peak_rate"] for k in have])) if have else None,
                   pairs=[dict(cell=k[0], seed=k[1], held_turn=got[k]["held_turn_deg"], turn_end=got[k]["turn_end_deg"],
                               three_frac=got[k]["three_frac"], breaks=got[k]["contact_breaks"],
                               pads_held_turn=pads[k]["held_turn_deg"]) for k in have])
        ps = [(got[k].get("plan_s_3thread") or got[k]["plan_s"]) / N_REPLAN for k in have
              if got[k].get("source") == "20261014" or got[k].get("plan_s_3thread")]
        rep["replan_s_3threads"] = float(np.median(ps)) if ps else None
        if cfg != "icem_pin" and have:
            d = np.array([got[k]["held_turn_deg"] - pads[k]["held_turn_deg"] for k in have])
            dc = {}
            for k, v in zip(have, d):
                dc.setdefault(k[0], []).append(v)
            dc = {c: np.array(v) for c, v in dc.items()}
            rep["paired_change_median"] = float(np.median(d))
            rep["paired_change_ci95"] = list(boot_cluster(dc)) if len(dc) > 1 else None
            rep["wilcoxon_p"] = (float(wilcoxon(d, zero_method="wilcox", method="exact").pvalue)
                                 if np.any(d != 0) and len(d) >= 6 else None)
            p3 = [bool(pads[k]["three_frac"] >= 0.99) for k in have]
            rep["mcnemar_three_vs_pads"] = AG.mcnemar_exact(p3, three)
            rep["paired_change_by_cell"] = {f"{c[1]:g} mm, t {c[2]:g}": [round(float(x), 2) for x in v] for c, v in dc.items()}
        out["configs"].append(rep)
    out["pads_replan_s_3threads"] = next(r["replan_s_3threads"] for r in out["configs"] if r["config"] == "icem_pin")
    (PL / "summary_U4.json").write_text(json.dumps(out, indent=1) + "\n")
    fig(out, PL / "20261014-u4_planner_models.png")
    print("reproduction", repro)
    for r in out["configs"]:
        print({k: v for k, v in r.items() if k not in ("pairs",)})
    print("pads replanning at 3 threads", out["pads_replan_s_3threads"])
    return 0


def fig(S, path):
    plt = AG._style()
    f, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.9), dpi=110, gridspec_kw=dict(width_ratios=[2.3, 1.0], wspace=0.3))
    for j, (cfg, lab, col, mk) in enumerate(CONFIGS):
        rep = next((r for r in S["configs"] if r["config"] == cfg), None)
        if not rep:
            continue
        xs, ys = [], []
        for p in rep["pairs"]:
            i = CELLS.index(tuple(p["cell"]))
            xs.append(i + (j - 1) * 0.22 + (p["seed"] - 1.5) * 0.04)
            ys.append(p["held_turn"])
        a.plot(xs, ys, mk, color=col, ms=7, markeredgecolor="white", markeredgewidth=0.8, label=lab, linestyle="none")
        if rep.get("replan_s_3threads") or cfg == "icem_pin":
            v = rep.get("replan_s_3threads") if cfg != "icem_pin" else S.get("pads_replan_s_3threads")
            if v:
                b.bar(j, v, color=col, width=0.6)
                b.text(j, v * 1.02, f"{v:.2f}", ha="center", va="bottom", fontsize=13)
    a.set_xticks(range(len(CELLS)))
    a.set_xticklabels([f"{c[1]:g} mm, t {c[2]:g}" for c in CELLS], fontsize=13)
    a.set_ylabel("Held turn (°)")
    a.set_ylim(0, 100)
    a.legend(loc="lower left", frameon=False, fontsize=13)
    b.set_xticks(range(len(CONFIGS)))
    b.set_xticklabels(["Pads", "Condim 4", "Point\ncontact"], fontsize=13)
    b.set_ylabel("Replanning time, 3 threads (s)")
    for ax in (a, b):
        ax.grid(axis="y", color="#ddd", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    f.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.14)
    f.savefig(path)
    plt.close(f)


if __name__ == "__main__":
    sys.exit(main())
