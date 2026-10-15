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


def med_ok(v):
    """Median over the values present (a run that drops the cylinder early has no turn-rate jitter)."""
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


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
                   contact_breaks_median=med_ok([got[k]["contact_breaks"] for k in have]),
                   cmd_rate_rms_median=med_ok([got[k]["cmd_rate_rms"] for k in have]),
                   omega_jitter_median=med_ok([got[k]["omega_jitter"] for k in have]),
                   peak_rate_median=med_ok([got[k]["peak_rate"] for k in have]),
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
    # the plant's sliding friction lowered while the planner's model keeps 1 (variants mu0.5, mu0.7; seeds 0 and 1)
    var = [r for r in AG.jl(PL / "turn.jsonl") if r.get("status") == "ok" and r.get("variant")]
    out["friction"] = []
    for vname in sorted({r["variant"] for r in var}):
        vv = {(r["shape"], r["d_mm"], r["t"], r["seed"]): r for r in var if r["variant"] == vname and r["config"] == "icem_pin"}
        pairs = [(k, vv[k], pads.get((k[:3], k[3]))) for k in vv if pads.get((k[:3], k[3]))]
        if not pairs:
            continue
        three = sum(r["three_frac"] >= 0.99 for _, r, _ in pairs)
        med_v = float(np.median([r["held_turn_deg"] for _, r, _ in pairs]))
        med_n = float(np.median([b["held_turn_deg"] for _, _, b in pairs]))
        d = float(np.median([r["held_turn_deg"] - b["held_turn_deg"] for _, r, b in pairs]))
        mu = vname.replace("mu", "")
        out["friction"].append(dict(variant=vname, n=len(pairs), three_finger=int(three), held=sum(bool(r["held"]) for _, r, _ in pairs),
                                    held_turn_median=med_v, nominal_median=med_n, paired_change_median=d,
                                    sentence=(f"With the plant&#8217;s sliding friction at {mu} while the planner&#8217;s model keeps "
                                              f"1, the pads planner ends {three} of {len(pairs)} runs with three fingertips loaded and "
                                              f"turns a median {med_v:.1f}&#176; against {med_n:.1f}&#176; on the same pairs at "
                                              f"nominal friction (median change {d:+.1f}&#176;).")))
    (PL / "summary_U4.json").write_text(json.dumps(out, indent=1) + "\n")
    findings(out)
    fig(out, PL / "20261014-u4_planner_models.png")
    print("reproduction", repro)
    for r in out["configs"]:
        print({k: v for k, v in r.items() if k not in ("pairs",)})
    print("pads replanning at 3 threads", out["pads_replan_s_3threads"])
    return 0


def findings(S):
    """planner/findings_U4.json from the summary; its sentences take the numbers of the rows, so a rerun after more rows
    rewrites it."""
    R = {r["config"]: r for r in S["configs"]}
    p, c4, pt = R["icem_pin"], R["icem_pin_pt4"], R["icem_pin_pt"]
    fr = {v["variant"]: v for v in S.get("friction", [])}
    rel = "docs/experiments/20261014-pad_usability/planner/"
    gate = lambda r: (r["paired_change_median"] >= -10 and r["three_finger"] >= 15)  # noqa: E731
    fsent = " ".join(f"At plant friction {v['variant'][2:]} (the planner's model keeping 1) the pads planner ends "
                     f"{v['three_finger']} of {v['n']} runs with three fingertips loaded and keeps the cylinder in {v['held']}; "
                     f"median held turn {v['held_turn_median']:.1f} deg against {v['nominal_median']:.1f} deg at nominal friction."
                     for v in fr.values())
    pend = "" if len(fr) >= 2 and all(v["n"] >= 8 for v in fr.values()) else " The friction arm is incomplete."
    summ = (f"On the 1 mm pad plant (servo-limited real_v1, four cells of the scale study, seeds 0-3) pinned iCEM with the pads as "
            f"its rollout model ends {p['three_finger']} of {p['n_pairs']} runs with three fingertips loaded, median held turn "
            f"{p['held_turn_median']:.1f} deg, at {p['replan_s_3threads']:.2f} s per replanning on three threads; its seed-0 run on "
            f"the 25 mm cell reproduces the 13 October row ({S['reproduction']['new']:.2f} against {S['reproduction']['old']:.2f} "
            f"deg). Rolling out condim 4 on the TPU block replans in {c4['replan_s_3threads']:.2f} s and changes the held turn by a "
            f"median {c4['paired_change_median']:+.1f} deg (cell-cluster bootstrap 95 % interval {c4['paired_change_ci95'][0]:+.1f} "
            f"to {c4['paired_change_ci95'][1]:+.1f}, exact Wilcoxon p {c4['wilcoxon_p']:.1e}), ending {c4['three_finger']} of 16 "
            f"runs with three fingertips loaded; point contact replans in {pt['replan_s_3threads']:.2f} s and changes it by "
            f"{pt['paired_change_median']:+.1f} deg ({pt['paired_change_ci95'][0]:+.1f} to {pt['paired_change_ci95'][1]:+.1f}, p "
            f"{pt['wilcoxon_p']:.1e}), {pt['three_finger']} of 16 three-finger endings. " + fsent + pend)
    passing = [c for c in (c4, pt) if gate(c)]
    dec = (("(1) The CPU planner keeps the 1 mm pads as its rollout model: " if not passing else
            f"(1) {passing[0]['label']} passes the gate and may replace the pads in the CPU planner: ")
           + f"condim 4 and point contact cut the replanning to {c4['replan_s_3threads'] / p['replan_s_3threads']:.2f} and "
           f"{pt['replan_s_3threads'] / p['replan_s_3threads']:.2f} of the pads planner's time but lose a median "
           f"{-c4['paired_change_median']:.1f} and {-pt['paired_change_median']:.1f} deg of held turn and end "
           f"{16 - c4['three_finger']} and {16 - pt['three_finger']} of 16 runs without three loaded fingertips. (2) No 5 ms "
           "configuration ran: U3 found the pads and point contact below 36 of 40 placements within 3 deg of 1 ms at 4-5 ms on "
           "the servo-limited plant. (3) The CAD-hand arm ran no configuration, since none passed. "
           + ("(4) MPC rollouts need the plant's friction: planning with friction 1 on a plant at 0.5 ends "
              f"{fr['mu0.5']['three_finger']} of {fr['mu0.5']['n']} runs with three fingertips loaded, so the planner's "
              "friction must come from the bench pull test or be randomised in its rollouts before a bench run."
              if "mu0.5" in fr and fr["mu0.5"]["three_finger"] < 7 else ""))
    K = [dict(name=f"{r['label']}: three-finger endings, median held turn, replanning time", value=
              f"{r['three_finger']} of {r['n_pairs']}, {r['held_turn_median']:.1f} deg, {r['replan_s_3threads']:.2f} s",
              unit="runs, deg, s at 3 threads", n=f"{r['n_pairs']} pairs", source=rel + "turn.jsonl" +
              (" and docs/experiments/20261013-scale_turn_controllers/planners/turn.jsonl (seeds 0-1)" if r["config"] == "icem_pin" else ""),
              how="three_frac >= 0.99; held_turn_deg median; plan_s / 275 median over this study's 3-thread runs")
         for r in (p, c4, pt)]
    K += [dict(name=f"{r['label']}: paired change of held turn against the pads planner", value=
               f"{r['paired_change_median']:+.1f} [{r['paired_change_ci95'][0]:+.1f}, {r['paired_change_ci95'][1]:+.1f}]",
               unit="deg, median [cell-cluster bootstrap 95 % interval]", n="16 pairs",
               source=rel + "summary_U4.json", how=f"exact Wilcoxon p {r['wilcoxon_p']:.2e}; McNemar on three-finger endings "
               f"{r['mcnemar_three_vs_pads'][0]}:{r['mcnemar_three_vs_pads'][1]}, p {r['mcnemar_three_vs_pads'][2]:.3f}")
          for r in (c4, pt)]
    K += [dict(name=f"Pads planner at plant friction {v['variant'][2:]}", value=f"{v['three_finger']} of {v['n']} three-finger, "
               f"{v['held']} held, median held turn {v['held_turn_median']:.1f} deg", unit="runs", n=f"{v['n']} pairs (seeds 0-1)",
               source=rel + "turn.jsonl", how="variant rows; the planner's model compiles from the XML at friction 1")
          for v in fr.values()]
    films = []
    meta = PL / "media/films.jsonl"
    for mp in sorted((PL / "media").glob("planner_models_*.mp4")) if (PL / "media").exists() else []:
        poster = mp.with_suffix(".jpg")
        films.append(dict(path=rel + "media/" + mp.name, poster=rel + "media/" + poster.name,
                          caption=("Pinned iCEM on the 1 mm pad plant with three rollout models, left to right: the pads, condim 4 "
                                   "and point contact on the TPU block; seed 0, " + mp.stem.replace("planner_models_", "") +
                                   "; panels from fresh runs with the rows' seeds.")))
    F = dict(id="U4", status="complete" if not pend else "partial", summary=summ,
             hypothesis_verdict="falsified" if not passing else "supported",
             verdict_reason=("The condim-4 planner falls below both thresholds of the hypothesis (a median change of at least -10 "
                             f"deg: {c4['paired_change_median']:+.1f}; at least 15 of 16 three-finger endings: {c4['three_finger']}); "
                             "the point-contact planner loses more than 10 deg, as predicted. The friction arm "
                             + ("fails its falsifier (fewer than 7 of 8 three-finger endings at friction 0.5)."
                                if "mu0.5" in fr and fr["mu0.5"]["three_finger"] < 7 else "passes at friction 0.5.")),
             decision=dec, key_numbers=K,
             rows=[rel + "turn.jsonl", rel + "summary_U4.json", "docs/experiments/20261013-scale_turn_controllers/planners/turn.jsonl"],
             scripts=["scripts/pad_usability_planner_model.py", "scripts/pad_usability_planner_report.py",
                      "scripts/hand_object_scale_planners.py", "scripts/hand_object_scale_mpc.py"],
             deviations=("(1) The planners ran at three threads (two processes on cores 12-14 and 15, 16, 19); the pads planner's "
                         "seeds 0 and 1 are the 13 October rows at six threads, so its replanning time is taken from this study's "
                         "3-thread runs. (2) The configurations fs, fs5, pt5 and pads2 did not run: the GPU solver settings reset "
                         "CPU rollouts and 2 mm pads missed the one-third replanning gate (U1), and no model kept 36 of 40 placements "
                         "at 4-5 ms (U3). (3) The friction arm ran with the pads planner, since no cheaper planner passed."),
             issues=("Four cells and four seeds; cell effects (34-91 deg for the pads planner) exceed seed effects, and the bootstrap "
                     "resamples cells. The planners differ only in the model they roll out; the cost weights were tuned with the "
                     "pads planner on 13 October."),
             figures=[dict(path=rel + "20261014-u4_planner_models.png", caption=(
                 "Left: held turn of each pair (cell, seed 0-3) under the pads, condim-4 and point-contact planners on the 1 mm pad "
                 "plant. Right: planner time per replanning on three threads (64 rollouts of 0.5 s)."))],
             films=films)
    (PL / "findings_U4.json").write_text(json.dumps(F, indent=1, ensure_ascii=False) + "\n")


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
