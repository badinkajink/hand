#!/usr/bin/env python3
"""The pre-registered analysis of the morphology-ranking transfer study.

    scripts/real_v1_transfer_study.py --runs logs/hardware --out docs/experiments/...

Protocol: `docs/experiments/20260831-real_v1-transfer-protocol/README.md`. That document fixes
what is measured and what is computed; this file is the executable half of it, written BEFORE
any trial exists so the statistics cannot be chosen after the numbers arrive. It runs on partial
data on purpose -- point it at ten Stage-1 trials and it will report what ten trials support and
say the rest is missing.

The question is whether the simulator's ORDERING of hands survives transfer. Absolute agreement
is not expected and is not tested: the hand arrives at 0.44-0.90 of its commanded yaw, so the
bench alignment is systematically below the simulated one by construction.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs/experiments/20260829-real_v1_deploy/deploy/catalog.json"
BANDS = (ROOT / "docs/experiments/20260830-real_v1-budget-rescreen/deploy_plan_bands.json",
         ROOT / "docs/experiments/20260831-real_v1-sobol8192/plan_bands_shipped.json")

#: Plans excluded before the study by the finger-finger clearance gate, not by their scores.
#: g24 is the one that matters rhetorically: the simulator ranks it second and it cannot be
#: built, which is the failure a reconfigurable platform exists to catch.
CLEARANCE_EXCLUDED = {"g23": 0.8, "rv04_mid": -2.6, "g24": -5.2}

#: Families where several exported plans are the SAME morphology at different residual clips.
#: These are the study's within-hand control: a bench difference inside a family is control, not
#: morphology, and a cross-family difference no larger than it means nothing.
FAMILIES = {"g12": ["g12", "g12w08", "g12_b095", "g12w11"],
            "sv1_u0060": ["sv1_u0060_b75", "sv1_u0060_b100"],
            "sv1_u0100": ["sv1_u0100_b70", "sv1_u0100_b100"]}


# ---- inputs -----------------------------------------------------------------------------

def plan_key(summary: dict) -> str | None:
    """Which exported plan produced this run.

    `design` is the morphology tag and several plans share one -- g12 ships at four clips under
    the single design "g12" -- so the file name is the identity. Runs recorded before the web
    service started stamping `plan_file` fall back to (design, budget), which is unique across
    everything currently exported.
    """
    meta = summary.get("plan_meta") or {}
    name = meta.get("plan_file")
    if name:
        return name[:-len("_plan.json")] if name.endswith("_plan.json") else name
    design, budget = summary.get("design"), meta.get("budget_rad")
    if not design or budget is None:
        return None
    return f"{design}@b{float(budget):.2f}"


def load_sim() -> dict:
    """Simulated prediction per plan: held_cos at the plan's OWN clip, and its replicate sd."""
    cat = json.loads(CATALOG.read_text())["designs"]
    rows = []
    for path in BANDS:
        if path.exists():
            rows += json.loads(path.read_text())["rows"]
    reps = defaultdict(list)
    for r in rows:
        reps[(r["plan"], round(float(r["budget"]), 2))].append(r)
    out = {}
    for plan, v in cat.items():
        budget = round(float(v["budget_rad"]), 2)
        kept = [r["final_cos"] for r in reps.get((plan, budget), []) if r["ok"]]
        n = len(reps.get((plan, budget), []))
        out[plan] = {"held_cos": v.get("held_cos"), "budget_rad": budget,
                     "clearance_mm": v.get("clearance_mm"),
                     "sim_rank": v.get("sim_rank"), "n_reps": n, "n_kept": len(kept),
                     "sd": st.stdev(kept) if len(kept) > 1 else 0.0,
                     "excluded": plan in CLEARANCE_EXCLUDED}
    return out


def load_bench(runs_dir: Path) -> tuple[dict, list]:
    """Every completed reorientation run that carries an instrument trace, grouped by plan."""
    by_plan, skipped = defaultdict(list), []
    for path in sorted(runs_dir.glob("*_SUMMARY.json")):
        s = json.loads(path.read_text())
        if s.get("operation") != "reorientation":
            continue
        key = plan_key(s)
        track = s.get("object_track")
        if key is None:
            skipped.append((path.name, "no plan identity", None))
            continue
        if s.get("status") != "complete":
            skipped.append((path.name, f"status {s.get('status')}", key))
            continue
        if not track:
            skipped.append((path.name, "no instrument trace", key))
            continue
        if track.get("cos_hold") is None:
            # The pose the hand ENDED holding was never observed -- the tag went dark before
            # the hold window. This is NOT interchangeable with a missing trace: a tool that
            # is flung out of the camera's view produces one, so silently dropping these
            # deletes failures from the numerator of exactly the plans that fail. Excluded
            # from the correlation, but counted and attributed to the plan below.
            skipped.append((path.name, f"ENDING UNOBSERVED after "
                                       f"{track.get('duration_s', float('nan')):.1f}s", key))
            continue
        # Voids declared in the protocol, applied here so they cannot be applied selectively.
        if (track.get("visibility") or 0) < 0.9:
            skipped.append((path.name, f"VOID visibility {track['visibility']:.2f}", key))
            continue
        manual = s.get("manual_score") or {}
        by_plan[key].append({
            "run_id": s["run_id"], "design": s.get("design"),
            # Parity with the simulator: a released tool scores 0, exactly as an ok=False
            # simulated replicate contributes 0 to held_cos. Score them differently and the
            # correlation is between two different quantities.
            "cos_hold": 0.0 if track.get("dropped") else float(track["cos_hold"]),
            "cos_raw": float(track["cos_hold"]), "dropped": bool(track.get("dropped")),
            "deg_turned": track.get("deg_turned"), "slip_mm": track.get("slip_mm"),
            "z_drop_mm": track.get("z_drop_mm"), "wrong_pole": bool(track.get("wrong_pole")),
            "visibility": track.get("visibility"),
            "operator_deg": manual.get("turn_deg"), "operator_success": manual.get("success"),
        })
    return dict(by_plan), skipped


# ---- statistics -------------------------------------------------------------------------

def spearman(x, y) -> tuple[float, float]:
    from scipy import stats
    r = stats.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def bootstrap_rho(x, y, n=10000, seed=0) -> tuple[float, float]:
    """Percentile interval, resampling PLANS (not trials): the plan is the unit of the claim."""
    rng = np.random.default_rng(seed)
    x, y, k = np.asarray(x, float), np.asarray(y, float), len(x)
    out = []
    for _ in range(n):
        i = rng.integers(0, k, k)
        if len(set(x[i])) < 3 or len(set(y[i])) < 3:
            continue
        out.append(spearman(x[i], y[i])[0])
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def kappa(a, b) -> float:
    """Cohen's kappa on two binary labels."""
    a, b = list(a), list(b)
    n = len(a)
    if not n:
        return float("nan")
    obs = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    exp = pa * pb + (1 - pa) * (1 - pb)
    return (obs - exp) / (1 - exp) if exp < 1 else float("nan")


# ---- the six pre-registered analyses -----------------------------------------------------

def analyse(sim: dict, bench: dict) -> dict:
    eligible = {p: v for p, v in sim.items() if not v["excluded"]}
    paired = [(p, eligible[p], bench[p]) for p in sorted(bench) if p in eligible]
    report = {"n_plans_eligible": len(eligible), "n_plans_measured": len(paired),
              "n_trials": sum(len(t) for _, _, t in paired)}

    if len(paired) < 3:
        report["primary"] = {"status": f"only {len(paired)} plans measured; need 3 to correlate"}
        return report

    x = [s["held_cos"] for _, s, _ in paired]
    y = [st.mean([t["cos_hold"] for t in trials]) for _, _, trials in paired]
    rho, p = spearman(x, y)
    lo, hi = bootstrap_rho(x, y)
    report["primary"] = {
        "n": len(paired), "spearman_rho": round(rho, 3), "p": round(p, 4),
        "ci95": [round(lo, 3), round(hi, 3)],
        # n=16 two-sided 5% critical value; stated here so an underpowered null is read as
        # underpowered rather than as evidence of no effect.
        "critical_rho_at_n": round(_critical_rho(len(paired)), 3),
        "plans": {p: {"sim": s["held_cos"], "bench_mean": round(m, 3),
                      "bench_sd": round(st.stdev([t["cos_hold"] for t in tr]), 3)
                      if len(tr) > 1 else None, "n": len(tr)}
                  for (p, s, tr), m in zip(paired, y)}}

    # 2. distinct morphologies only -- each family's best plan, so the answer is not carried by
    #    the clip duplicates.
    of_family = {}
    for p, s, tr in paired:
        fam = next((f for f, members in FAMILIES.items() if p in members), p)
        cand = (s["held_cos"], p, st.mean([t["cos_hold"] for t in tr]))
        if fam not in of_family or cand[0] > of_family[fam][0]:
            of_family[fam] = cand
    if len(of_family) >= 3:
        fx = [v[0] for v in of_family.values()]
        fy = [v[2] for v in of_family.values()]
        frho, fp = spearman(fx, fy)
        report["morphologies_only"] = {"n": len(of_family), "spearman_rho": round(frho, 3),
                                       "p": round(fp, 4),
                                       "plans": {v[1]: round(v[2], 3) for v in of_family.values()}}

    # 3. within-family clip contrasts: same steel, so any difference here is CONTROL.
    fams = {}
    for fam, members in FAMILIES.items():
        got = [(m, sim[m]["budget_rad"], sim[m]["held_cos"],
                st.mean([t["cos_hold"] for t in bench[m]]), len(bench[m]))
               for m in members if m in bench and m in sim]
        if len(got) >= 2:
            fams[fam] = {
                "clips": [{"plan": g[0], "clip": g[1], "sim": g[2],
                           "bench": round(g[3], 3), "n": g[4]} for g in sorted(got, key=lambda g: g[1])],
                "sim_order": [g[0] for g in sorted(got, key=lambda g: -g[2])],
                "bench_order": [g[0] for g in sorted(got, key=lambda g: -g[3])]}
            fams[fam]["orders_agree"] = fams[fam]["sim_order"] == fams[fam]["bench_order"]
    report["within_family"] = fams

    # 4. retention agreement
    sim_keep = [1 if (s["held_cos"] or 0) > 0 else 0 for _, s, _ in paired]
    ben_keep = [1 if st.mean([0 if t["dropped"] else 1 for t in tr]) >= 0.5 else 0
                for _, _, tr in paired]
    report["retention"] = {
        "kappa": round(kappa(sim_keep, ben_keep), 3),
        "table": {"both_keep": sum(a and b for a, b in zip(sim_keep, ben_keep)),
                  "sim_only": sum(a and not b for a, b in zip(sim_keep, ben_keep)),
                  "bench_only": sum(b and not a for a, b in zip(sim_keep, ben_keep)),
                  "both_drop": sum((not a) and (not b) for a, b in zip(sim_keep, ben_keep))}}

    # 6. instrument vs operator, on the trials where the operator recorded a number
    pairs = [(t["operator_deg"], 90.0 - math.degrees(math.acos(max(-1, min(1, t["cos_raw"])))))
             for _, _, tr in paired for t in tr if t.get("operator_deg") is not None]
    if len(pairs) >= 3:
        d = [a - b for a, b in pairs]
        report["operator_vs_instrument"] = {
            "n": len(d), "bias_deg": round(st.mean(d), 2),
            "loa95_deg": [round(st.mean(d) - 1.96 * st.stdev(d), 2),
                          round(st.mean(d) + 1.96 * st.stdev(d), 2)] if len(d) > 1 else None}

    # Stage 1: whichever plan has the most trials is the repeatability estimate.
    best = max(paired, key=lambda t: len(t[2]))
    if len(best[2]) >= 5:
        v = [t["cos_hold"] for t in best[2]]
        report["repeatability"] = {"plan": best[0], "n": len(v), "mean": round(st.mean(v), 3),
                                   "sd": round(st.stdev(v), 3),
                                   "sim_sd": round(best[1]["sd"], 3),
                                   "drops": sum(t["dropped"] for t in best[2])}
    return report


def _critical_rho(n: int) -> float:
    """Two-sided 5% critical Spearman rho, via the t approximation. Honest power, stated up front."""
    from scipy import stats
    if n < 4:
        return float("nan")
    t = stats.t.ppf(0.975, n - 2)
    return float(t / math.sqrt(t * t + n - 2))


# ---- figures ----------------------------------------------------------------------------

def figure_transfer(sim, bench, report, out: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plans = report.get("primary", {}).get("plans")
    if not plans:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    fam_of = {m: f for f, members in FAMILIES.items() for m in members}
    colours = {f: c for f, c in zip(FAMILIES, ("#c1440e", "#1f6f8b", "#5b7c1f"))}
    for p, v in plans.items():
        tr = bench[p]
        dropped = sum(t["dropped"] for t in tr) > len(tr) / 2
        c = colours.get(fam_of.get(p), "#333333")
        ax.errorbar(v["sim"], v["bench_mean"], yerr=v["bench_sd"] or 0,
                    xerr=sim[p]["sd"], fmt="o" if not dropped else "o",
                    mfc="white" if dropped else c, mec=c, ecolor=c, ms=6, lw=1, capsize=2)
        ax.annotate(p.replace("sv1_", ""), (v["sim"], v["bench_mean"]), fontsize=5.5,
                    xytext=(4, 3), textcoords="offset points", color=c)
    for fam, members in FAMILIES.items():
        pts = sorted(((plans[m]["sim"], plans[m]["bench_mean"]) for m in members if m in plans))
        if len(pts) > 1:
            ax.plot(*zip(*pts), lw=0.7, alpha=0.5, color=colours[fam], zorder=0)
    lim = [-0.05, 1.0]
    ax.plot(lim, lim, ls=":", lw=0.8, color="#999999", label="identity (not expected)")
    pr = report["primary"]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("simulated axis alignment, held (cos)")
    ax.set_ylabel("measured axis alignment, hold window (cos)")
    ax.set_title(f"Spearman $\\rho$ = {pr['spearman_rho']:.2f} "
                 f"[{pr['ci95'][0]:.2f}, {pr['ci95'][1]:.2f}], n = {pr['n']} plans", fontsize=9)
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    ax.grid(alpha=0.15, lw=0.5)
    fig.tight_layout()
    path = out / "fig_transfer.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---- output -----------------------------------------------------------------------------

def markdown(sim, bench, report, skipped) -> str:
    L = ["# Transfer study — pre-registered analysis", "",
         f"{report['n_trials']} trials over {report['n_plans_measured']} of "
         f"{report['n_plans_eligible']} eligible plans.", ""]
    pr = report.get("primary", {})
    if "spearman_rho" in pr:
        verdict = ("above" if abs(pr["spearman_rho"]) >= pr["critical_rho_at_n"] else "BELOW")
        L += [f"**Primary.** Spearman rho = {pr['spearman_rho']:+.3f} "
              f"(95% CI {pr['ci95'][0]:+.3f} to {pr['ci95'][1]:+.3f}, p = {pr['p']:.4f}), "
              f"{verdict} the n = {pr['n']} critical value of {pr['critical_rho_at_n']:.3f}.", "",
              "| plan | sim cos | bench cos | bench sd | n |", "|---|---|---|---|---|"]
        for p, v in sorted(pr["plans"].items(), key=lambda kv: -kv[1]["sim"]):
            sd = "" if v["bench_sd"] is None else f"{v['bench_sd']:.3f}"
            L.append(f"| `{p}` | {v['sim']:.3f} | {v['bench_mean']:.3f} | {sd} | {v['n']} |")
    else:
        L.append(f"**Primary.** {pr.get('status', 'not computable yet')}")
    L += [""]
    if report.get("repeatability"):
        r = report["repeatability"]
        L += [f"**Repeatability.** `{r['plan']}`, n = {r['n']}: mean {r['mean']:.3f}, "
              f"sd {r['sd']:.3f} (simulated replicate sd {r['sim_sd']:.3f}), "
              f"{r['drops']} drop(s).", ""]
    for fam, v in (report.get("within_family") or {}).items():
        L.append(f"**Within `{fam}`** (same hand, clip varies): sim order "
                 f"{' > '.join(v['sim_order'])}; bench order {' > '.join(v['bench_order'])}"
                 f" — {'AGREE' if v['orders_agree'] else 'DISAGREE'}.")
    if report.get("retention"):
        t = report["retention"]["table"]
        L += ["", f"**Retention.** kappa = {report['retention']['kappa']:.3f}; both keep "
              f"{t['both_keep']}, sim only {t['sim_only']}, bench only {t['bench_only']}, "
              f"both drop {t['both_drop']}."]
    if report.get("morphologies_only"):
        m = report["morphologies_only"]
        L += ["", f"**Distinct morphologies only.** n = {m['n']}, rho = {m['spearman_rho']:+.3f} "
              f"(p = {m['p']:.4f}) — the answer is not carried by the clip duplicates."]
    if report.get("operator_vs_instrument"):
        o = report["operator_vs_instrument"]
        loa = "" if not o["loa95_deg"] else f", 95% limits {o['loa95_deg'][0]:+.1f} to {o['loa95_deg'][1]:+.1f} deg"
        L += ["", f"**Operator vs instrument.** n = {o['n']}, the eye reads "
              f"{o['bias_deg']:+.1f} deg relative to the tags{loa}."]
    if skipped:
        L += ["", f"**Excluded from the analysis ({len(skipped)}):**", ""]
        # Attributed to the plan, because an exclusion rate that CLUSTERS on one hand is a
        # result about that hand, and an exclusion rate spread evenly is an instrument note.
        per_plan = defaultdict(int)
        for _, _, key in skipped:
            per_plan[key] += 1
        for key, n in sorted(per_plan.items(), key=lambda kv: -kv[1]):
            kept = len(bench.get(key, ()))
            L.append(f"- **{key or 'unidentified'}** — {n} excluded, {kept} kept"
                     + ("  ← more excluded than kept" if n > kept else ""))
        L += [""] + [f"  - `{n}` — {why}" for n, why, _ in skipped[:30]]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True,
                    help="directory of *_SUMMARY.json pulled from the CB1")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs/experiments/20260831-real_v1-transfer-protocol")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()

    if not a.runs.is_dir():
        print(f"no such directory: {a.runs}", file=sys.stderr)
        return 2
    sim = load_sim()
    bench, skipped = load_bench(a.runs)
    report = analyse(sim, bench)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    md = markdown(sim, bench, report, skipped)
    (a.out / "ANALYSIS.md").write_text(md)
    print(md)
    if not a.no_figures:
        p = figure_transfer(sim, bench, report, a.out)
        print(f"wrote {p}" if p else "no figure: not enough measured plans yet")
    print(f"wrote {a.out / 'analysis.json'} and {a.out / 'ANALYSIS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
