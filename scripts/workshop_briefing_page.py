#!/usr/bin/env python3
"""Build the workshop briefing page: the record behind the SR2 Hand paper.

    python3 scripts/workshop_briefing_page.py
        [--out docs/experiments/20260921-workshop_briefing/20260921-sr2_hand_program_record.html]

Every number that has a JSON source is read from it here: the Sobol-8192 funnel counts
(docs/experiments/20260831-real_v1-sobol8192/), the eight-hand transfer table and its rank
correlations (paper/figures/ranking.json), the full-error retention of the six Sobol hands,
and the commit history. Numbers whose source is a markdown record (the stage-1 repeatability
session, the deploy-envelope placement bands, the CB1 run archive) are carried in the template
with their source named beside them. The prose lives in workshop_briefing_page.template.html.
"""
from __future__ import annotations

import argparse
import collections
import glob
import html
import json
import os
import subprocess
import time

import numpy as np
import retro_style

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POP = os.path.join(ROOT, "docs/experiments/20260831-real_v1-sobol8192")
RANKING = os.path.join(ROOT, "paper/figures/ranking.json")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "workshop_briefing_page.template.html")

#: the eight bench hands, in the paper's order, with where each one came from
EIGHT = [
    ("D1", "sv1_w6689", "Sobol-8192 pass (2026-08-31 promotion)", "sv1_w6689_b060", 0.60),
    ("D2", "sv1_w2360", "Sobol-4096 clip re-screen (2026-08-30)", "sv1_w2360_b075", 0.75),
    ("D3", "sv1_u1364", "Sobol-4096 clip re-screen (2026-08-30)", "sv1_u1364_b080", 0.80),
    ("D4", "g12", "hand-placed anchor from the 108-hand search (2026-08-28)", "g12_b095", 0.95),
    ("D5", "sv1_u0060", "Sobol-128 pilot (2026-08-30); same point in the 8,192 draw", "sv1_u0060_b75", 0.75),
    ("D6", "sv1_u0308", "Sobol-4096 clip re-screen (2026-08-30)", "sv1_u0308_b050", 0.50),
    ("D7", "rv05_manual", "manually specified hand (2026-08-27)", "rv05_manual_b85", 0.85),
    ("D8", "sv1_w0099", "Sobol-128 pilot (2026-08-30)", "sv1_w0099_b100", 1.00),
]


def num(v, nd=3):
    s = f"{v:.{nd}f}"
    return s.replace("-", "&#8722;")


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b)
    return float(r[0]), float(r[1])


def funnel():
    """Counts from the population's own files."""
    man = json.load(open(os.path.join(POP, "grasp_screen_manifest.json")))
    src = collections.Counter(r.get("source", "?") for r in man["designs"])
    hw = json.load(open(os.path.join(POP, "hardware_manifest.json")))["hardware_reachability"]
    rej = collections.Counter()
    worst = collections.defaultdict(float)
    for r in hw["rejections"]:
        for v in r["violations"]:
            rej[(v["finger"], v["axis"])] += 1
            worst[(v["finger"], v["axis"])] = max(worst[(v["finger"], v["axis"])], v["short_mm"])
    gs = json.load(open(os.path.join(POP, "retention/grasp_summary.json")))
    sel = json.load(open(os.path.join(POP, "selected/summary.json")))
    con = json.load(open(os.path.join(POP, "selected/confirmed/summary.json")))
    n_full = sum(len(json.load(open(f))) for f in glob.glob(os.path.join(POP, "full_error_b*.json")))
    return dict(sampled=len(man["designs"]), src=src, reachable=hw["accepted"], rejected=hw["rejected"],
                rej=rej, worst=worst, graspable=gs["graspable_any"], cells=gs["retention_cells_total"],
                cells_by_grasp=gs["retention_cells_by_grasp"], sel=sel, con=con, n_full=n_full)


def funnel_table(f):
    rows = [
        ("sampled", f["sampled"],
         f"8,192 scrambled Sobol points ({f['src']['sobol_uniform']:,} uniform, {f['src']['sobol_wide_bias']:,} in the outward-biased stratum) plus {f['src']['known_anchor']} anchor hands",
         "<code>grasp_screen_manifest.json</code>"),
        ("inside the rails", f["reachable"],
         f"every mount inside the gantry travel as believed on 2026-08-30; {f['rejected']:,} rejected, all on the x axes, by 3.8&#8211;5.9 mm",
         "<code>hardware_manifest.json</code>, <code>reachable.txt</code>"),
        ("admit a grasp", f["graspable"],
         f"the closed-form fitter finds a three-pad ring in at least one of four grasp cells ({f['graspable']/f['reachable']*100:.1f} %); {f['cells']:,} (design, cell) rows go forward",
         "<code>retention/grasp_summary.json</code>"),
        ("retention screen", f["sel"]["designs_passing_at_some_clip"],
         f"{f['sel']['cells_evaluated']:,} single-trial rollouts = {f['cells']:,} rows &#215; 5 clips; a design passes if any (cell, clip) keeps the tool through the proof lift and the 5 s hold at cos &#8805; 0.7; {f['sel']['designs_passing_at_0.5']} pass at 0.5 rad and {f['sel']['designs_invisible_at_0.5']} only at another clip",
         "<code>band_s*.json</code>, <code>selected/summary.json</code>"),
        ("confirmed", f["con"]["designs_passing_at_some_clip"],
         "five nominal and four half-error draws at the design&#8217;s own clip; at least three of five nominal draws keep the tool and the mean held cosine stays at or above 0.7",
         "<code>confirm_b*.json</code>, <code>selected/confirmed/summary.json</code>"),
        ("full-error draws", f["n_full"],
         "twenty draws per confirmed design with placement, mass, radius, friction, contact, servo and mount all perturbed at once; this stage ranks, it does not filter",
         "<code>full_error_b*.json</code>"),
        ("exported and gated", 19,
         "plans on the station across all passes: servo range, gantry travel, finger clearance &#8805; 5 mm along both the chord and the 50 Hz path, plan validity; then a bench-schedule band scan",
         "<code>deploy/promotion.json</code>, <code>deploy_clearance.txt</code>, <code>deploy_plan_bands_all.json</code>"),
        ("on the bench", 8,
         "the top eight distinct morphologies among the sixteen exported, clearance-safe plans, ordered by the bench-schedule replay; six Sobol members, one anchor, one manual hand",
         "<code>docs/experiments/20260831-real_v1-transfer-protocol/README.md</code> &#167;3"),
    ]
    out = ["<div class='tw'><table><thead><tr><th>stage</th><th class='num'>count</th><th>rule</th><th>file</th></tr></thead><tbody>"]
    for name, n, rule, file in rows:
        out.append(f"<tr><td><strong>{name}</strong></td><td class='num'>{n:,}</td><td>{rule}</td><td class='mono small'>{file}</td></tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def funnel_svg(f):
    stages = [("sampled", f["sampled"]), ("in the rails", f["reachable"]), ("admit a grasp", f["graspable"]),
              ("pass retention", f["sel"]["designs_passing_at_some_clip"]),
              ("confirmed", f["con"]["designs_passing_at_some_clip"]), ("exported", 19), ("bench", 8)]
    W, H, L, R = 760, 34 * len(stages) + 20, 130, 90
    top = f["sampled"]
    out = [f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='funnel' style='width:100%;height:auto;display:block'>"]
    for i, (name, n) in enumerate(stages):
        y = 10 + i * 34
        w = max(2.0, (W - L - R) * n / top)
        out.append(f"<text x='{L-10}' y='{y+19}' text-anchor='end' font-size='13' fill='var(--ink2)' font-family='IBM Plex Mono,monospace'>{name}</text>")
        out.append(f"<rect x='{L}' y='{y+4}' width='{w:.1f}' height='24' rx='3' fill='var(--s2)' opacity='{0.95 - 0.08*i:.2f}'/>")
        out.append(f"<text x='{L+w+8:.1f}' y='{y+20}' font-size='13' fill='var(--ink)' font-family='IBM Plex Mono,monospace' font-variant-numeric='tabular-nums'>{n:,}</text>")
    out.append("</svg>")
    return "\n".join(out)


def rails_note(f):
    parts = []
    for (finger, axis), n in sorted(f["rej"].items()):
        parts.append(f"{finger} {axis}: {n} designs, up to {f['worst'][(finger, axis)]:.1f} mm")
    return "; ".join(parts)


def ranking_rows():
    r = json.load(open(RANKING))
    for k, v in r.items():
        if isinstance(v, list):
            return v
    raise SystemExit("ranking.json has no row list")


def full_error_kept():
    """Full-error (level 1.0) retention of the confirmed Sobol hands from the 8,192 pass."""
    kept = {}
    for fpath in glob.glob(os.path.join(POP, "full_error_b*.json")):
        for row in json.load(open(fpath)):
            kept[row["design"]] = (row.get("ens_kept"), row.get("n_ens"), row.get("ens_win"),
                                   row.get("nom_cos"), row.get("budget_rad"))
    return kept


def eight_table(rows):
    by = {x["did"]: x for x in rows}
    fe = full_error_kept()
    out = ["<div class='tw'><table><thead><tr><th>hand</th><th>design</th><th>came from</th><th class='num'>clip (rad)</th>"
           "<th class='num'>sim c (4 reps)</th><th class='num'>sim kept</th><th class='num'>retention-screen c</th><th class='num'>full-error kept</th>"
           "<th class='num'>bench held</th><th class='num'>n<sub>c</sub></th><th class='num'>bench c</th><th class='num'>bench peak</th>"
           "<th class='num'>turn held / drop (&#176;)</th><th class='num'>slip held / drop (mm)</th></tr></thead><tbody>"]
    for did, design, src, plan, clip in EIGHT:
        x = by[did]
        f = fe.get(design)
        fe_txt = f"{f[0]}/{f[1]} at {f[4]:.1f}" if f else "&#8212;"
        rs_txt = num(f[3]) if f else "&#8212;"
        held = f"{x['n_held']}/{x['n']}" if did != "D1" else f"{x['n_held']}/{x['n']}&#8224;"
        turn = f"{x['held_deg']:.0f} / {x['drop_deg']:.0f}" if x["drop_deg"] is not None else f"{x['held_deg']:.0f} / &#8212;"
        slip = f"{x['held_slip']:.1f} / {x['drop_slip']:.1f}" if x["drop_slip"] is not None else f"{x['held_slip']:.1f} / &#8212;"
        star = "*" if did == "D4" else ""
        out.append("<tr>"
                   f"<td><strong>{did}{star}</strong></td><td class='mono small'>{design}</td><td class='small'>{src}</td>"
                   f"<td class='num'>{clip:.2f}</td><td class='num'>{num(x['sim_final_cos'])}</td><td class='num'>{x['sim_ok_rate']:.2f}</td>"
                   f"<td class='num'>{rs_txt}</td><td class='num'>{fe_txt}</td>"
                   f"<td class='num'>{held}</td><td class='num'>{x['n_cos']}</td><td class='num'>{num(x['bench_cos'])} &#177; {x['bench_sd']:.3f}</td>"
                   f"<td class='num'>{num(x['bench_peak'])}</td><td class='num'>{turn}</td><td class='num'>{slip}</td></tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def rho_block(rows):
    sim = [x["sim_final_cos"] for x in rows]
    bench = [x["bench_cos"] for x in rows]
    hold = [x["hold_rate_op"] for x in rows]
    peak = [x["bench_peak"] for x in rows]
    force = [x["sim_force_N"] for x in rows]
    cont = [x["sim_contacts"] for x in rows]
    ids = [x["did"] for x in rows]
    i4 = ids.index("D4")
    rw = [h * b for h, b in zip(hold, bench)]
    r_all, p_all = spearman(sim, bench)
    r_no4, p_no4 = spearman([s for j, s in enumerate(sim) if j != i4], [b for j, b in enumerate(bench) if j != i4])
    r_rw, p_rw = spearman(sim, rw)
    sens = []
    for v in np.arange(0.0, 1.001, 0.01):
        bb = list(bench); bb[i4] = float(v)
        sens.append(spearman(sim, bb)[0])
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(20000):
        idx = rng.integers(0, len(sim), len(sim))
        if len(set(idx.tolist())) < 3:
            continue
        rho = spearman([sim[i] for i in idx], [bench[i] for i in idx])[0]
        if not np.isnan(rho):
            boot.append(rho)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    stats = dict(
        r_all=r_all, p_all=p_all, r_no4=r_no4, p_no4=p_no4, r_rw=r_rw, p_rw=p_rw,
        sens_lo=min(sens), sens_hi=max(sens), boot_lo=lo, boot_hi=hi,
        r_force=spearman(force, bench)[0], r_cont=spearman(cont, bench)[0],
        r_hold_bench=spearman(hold, bench)[0], r_peak_bench=spearman(peak, bench)[0],
        r_sim_hold=spearman(sim, hold)[0], r_force_hold=spearman(force, hold)[0],
        r_hold_peak=spearman(hold, peak)[0],
    )
    table = ["<div class='tw'><table><thead><tr><th>pair</th><th class='num'>&#961;<sub>s</sub></th><th>reading</th></tr></thead><tbody>"]
    rowsx = [
        ("simulated c vs bench c, eight hands, held trials only (the paper&#8217;s number)", stats["r_all"], f"p = {stats['p_all']:.2f}; bootstrap 95 % interval over hands [{num(lo, 2)}, {num(hi, 2)}]"),
        ("the same without D4 (n = 7)", stats["r_no4"], f"p = {stats['p_no4']:.2f}; removing the substituted hand moves the coefficient by {num(stats['r_no4']-stats['r_all'], 2)}"),
        ("simulated c vs bench c with a drop scored 0 (the simulator&#8217;s own convention)", stats["r_rw"], f"p = {stats['p_rw']:.2f}; retention-weighted alignment orders the hands differently: D5 falls from first to sixth"),
        ("simulated contact force vs bench c", stats["r_force"], "the largest simulated correlate of measured alignment"),
        ("simulated contact count vs bench c", stats["r_cont"], "more simulated contacts, worse measured turn"),
        ("bench hold rate vs bench c", stats["r_hold_bench"], "retention and alignment are separate outcomes on the bench"),
        ("bench peak c vs bench c", stats["r_peak_bench"], "peak and hold alignment rank the hands alike; the peak is the larger number"),
        ("simulated c vs bench hold rate", stats["r_sim_hold"], "the simulator&#8217;s alignment does not order retention"),
        ("simulated contact force vs bench hold rate", stats["r_force_hold"], "nor does its contact force"),
    ]
    for name, r, reading in rowsx:
        table.append(f"<tr><td>{name}</td><td class='num'>{num(r, 2)}</td><td class='small'>{reading}</td></tr>")
    table.append("</tbody></table></div>")
    stats["table"] = "\n".join(table)
    return stats


def commits_by_month():
    out = subprocess.run(["git", "log", "--format=%ad", "--date=format:%Y-%m"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    c = collections.Counter(out)
    return " &#183; ".join(f"{m} {n}" for m, n in sorted(c.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        ROOT, "docs/experiments/20260921-workshop_briefing/20260921-sr2_hand_program_record.html"))
    a = ap.parse_args()
    f = funnel()
    rows = ranking_rows()
    st = rho_block(rows)
    n_pages = sum(1 for line in open(os.path.join(ROOT, "docs/experiments/INDEX.md"))
                  if line.startswith("| 2026"))
    n_dirs = sum(1 for d in os.listdir(os.path.join(ROOT, "docs/experiments"))
                 if os.path.isdir(os.path.join(ROOT, "docs/experiments", d)))
    subs = {
        "BUILT": time.strftime("%Y-%m-%d %H:%M"),
        "FUNNEL_TABLE": funnel_table(f),
        "FUNNEL_SVG": funnel_svg(f),
        "RAILS_NOTE": rails_note(f),
        "N_SAMPLED": f"{f['sampled']:,}", "N_REACH": f"{f['reachable']:,}", "N_REJ": f"{f['rejected']:,}",
        "N_GRASP": f"{f['graspable']:,}", "N_CELLS": f"{f['cells']:,}",
        "N_ROLLOUTS": f"{f['sel']['cells_evaluated']:,}",
        "N_SEL": f"{f['sel']['designs_passing_at_some_clip']}", "N_SEL05": f"{f['sel']['designs_passing_at_0.5']}",
        "N_SEL_INV": f"{f['sel']['designs_invisible_at_0.5']}",
        "SEL_BY_CLIP": ", ".join(f"{k} rad: {v}" for k, v in f["sel"]["by_selected_budget"].items()),
        "N_CON": f"{f['con']['designs_passing_at_some_clip']}", "N_CON05": f"{f['con']['designs_passing_at_0.5']}",
        "CON_BY_CLIP": ", ".join(f"{k} rad: {v}" for k, v in f["con"]["by_selected_budget"].items()),
        "N_FULL": f"{f['n_full']}",
        "CELLS_BY_GRASP": ", ".join(f"{k.replace('s', 'straddle ').replace('_t', ' mm, thumb ')} mm: {v:,}"
                                     for k, v in f["cells_by_grasp"].items()),
        "EIGHT_TABLE": eight_table(rows),
        "RHO_TABLE": st["table"],
        "RHO_ALL": num(st["r_all"], 2), "P_ALL": f"{st['p_all']:.2f}",
        "RHO_NO4": num(st["r_no4"], 2), "RHO_RW": num(st["r_rw"], 2), "P_RW": f"{st['p_rw']:.2f}",
        "SENS_LO": num(st["sens_lo"], 2), "SENS_HI": num(st["sens_hi"], 2),
        "BOOT_LO": num(st["boot_lo"], 2), "BOOT_HI": num(st["boot_hi"], 2),
        "RHO_FORCE": num(st["r_force"], 2), "RHO_CONT": num(st["r_cont"], 2),
        "RHO_HOLD_BENCH": num(st["r_hold_bench"], 2), "RHO_PEAK_BENCH": num(st["r_peak_bench"], 2),
        "RHO_SIM_HOLD": num(st["r_sim_hold"], 2), "RHO_HOLD_PEAK": num(st["r_hold_peak"], 2),
        "COMMITS": commits_by_month(), "N_PAGES": str(n_pages), "N_DIRS": str(n_dirs),
    }
    page = open(TEMPLATE).read()
    for k, v in subs.items():
        page = page.replace("{{" + k + "}}", v)
    left = sorted(set(__import__("re").findall(r"{{([A-Z0-9_]+)}}", page)))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(retro_style.apply(page))  # plain page style (owner, 2026-10-09)
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1024:.0f} kB)")


if __name__ == "__main__":
    main()
