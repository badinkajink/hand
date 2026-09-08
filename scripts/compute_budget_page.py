#!/usr/bin/env python3
"""Build the compute-budget page from compute_budget.json.

    python3 scripts/compute_budget_data.py     # collect the trainer logs
    python3 scripts/compute_budget_page.py     # build the page

No media: every figure is an SVG drawn here from the JSON, so the output file is the
whole artifact.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "docs/experiments/20260908-compute_budget"
DATA = DIR / "compute_budget.json"
TPL = ROOT / "scripts/compute_budget_page.template.html"
OUT = DIR / "20260908-compute_budget.html"

W = 1020
PAD_L, PAD_R = 190, 24
NSPE = 24                                     # num_steps_per_env


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def note(x, y, text, width_chars=112, dy=15):
    """Footnote text wrapped to the drawing width. SVG does not wrap, so do it here."""
    import textwrap
    return "\n".join(
        f'<text class="note" x="{x}" y="{y + i * dy}">{ln}</text>'
        for i, ln in enumerate(textwrap.wrap(text, width_chars,
                                             break_long_words=False)))


def gh_rows(d):
    """GH200 sweep with steps/s and per-env-step cost derived from s_per_iter."""
    out = []
    for r in d["deltaai"]["sweep"]:
        e, t = r["envs"], r["s_per_iter"]
        out.append({**r, "sps": e * NSPE / t, "us": 1e6 * t / (e * NSPE)})
    return out


# ---------------------------------------------------------------- charts

def chart_throughput(d) -> str:
    """The local population against the GH200 points, on one steps/s axis."""
    loc, ghs = d["local"]["by_env"], gh_rows(d)
    ref = d["published_local_reference_sps"]
    lo, hi = 6000, 11000
    bh, gap = 26, 12
    rows = [("3072", loc["3072"], "local"), ("2048", loc["2048"], "local")]
    rows += [(str(g["envs"]), g, "gh") for g in ghs]
    h = 62 + len(rows) * (bh + gap) + 62
    x0, x1 = PAD_L, W - PAD_R - 108

    def X(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    o = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="Training '
         f'throughput, workstation population versus GH200">']
    o.append(f'<text class="axlab" x="{PAD_L}" y="20">ENV-STEPS PER SECOND, STEADY STATE '
             f'(ITERATION 0 EXCLUDED)</text>')
    for v in range(6000, 11001, 1000):
        o.append(f'<line class="grid" x1="{X(v):.1f}" y1="32" x2="{X(v):.1f}" '
                 f'y2="{h - 46}"/>')
        o.append(f'<text class="tick" x="{X(v):.1f}" y="{h - 30}" text-anchor="middle">'
                 f'{v // 1000}k</text>')
    for i, (lab, r, kind) in enumerate(rows):
        y = 44 + i * (bh + gap)
        col = "var(--s2)" if kind == "local" else "var(--s1)"
        name = ("workstation" if kind == "local" else "GH200") + f" &#183; {lab} envs"
        n = f'n = {r["n"]}' if kind == "local" else "n = 1"
        o.append(f'<text class="ser" x="{PAD_L - 12}" y="{y + bh * .72}" text-anchor="end" '
                 f'fill="var(--ink)">{name}</text>')
        o.append(f'<text class="tick" x="{PAD_L - 12}" y="{y + bh * .72 + 13}" '
                 f'text-anchor="end">{n}</text>')
        if kind == "local":
            o.append(f'<line x1="{X(r["min"]):.1f}" y1="{y + bh / 2:.1f}" '
                     f'x2="{X(r["max"]):.1f}" y2="{y + bh / 2:.1f}" stroke="{col}" '
                     f'stroke-width="1.5" opacity=".55"/>')
            for e in ("min", "max"):
                o.append(f'<line x1="{X(r[e]):.1f}" y1="{y + 5}" x2="{X(r[e]):.1f}" '
                         f'y2="{y + bh - 5}" stroke="{col}" stroke-width="1.5" '
                         f'opacity=".55"/>')
            o.append(f'<rect x="{X(r["q1"]):.1f}" y="{y}" '
                     f'width="{X(r["q3"]) - X(r["q1"]):.1f}" height="{bh}" rx="4" '
                     f'fill="{col}" opacity=".30"/>')
            o.append(f'<line x1="{X(r["med"]):.1f}" y1="{y}" x2="{X(r["med"]):.1f}" '
                     f'y2="{y + bh}" stroke="{col}" stroke-width="3"/>')
            o.append(f'<text class="val" x="{x1 + 12}" y="{y + bh * .74}">'
                     f'{r["med"]:,.0f}</text>')
        else:
            o.append(f'<circle cx="{X(r["sps"]):.1f}" cy="{y + bh / 2:.1f}" r="6" '
                     f'fill="{col}"/>')
            o.append(f'<text class="val" x="{x1 + 12}" y="{y + bh * .74}">'
                     f'{r["sps"]:,.0f}</text>')
    o.append(f'<line x1="{X(ref):.1f}" y1="32" x2="{X(ref):.1f}" y2="{h - 46}" '
             f'stroke="var(--bad)" stroke-width="1.5" stroke-dasharray="4 3"/>')
    o.append(f'<text class="mark" x="{X(ref) - 8:.1f}" y="{h - 52}" text-anchor="end" '
             f'fill="var(--bad)">{ref:,} &#8212; THE REFERENCE THE 0.66&#215; RATIO USED</text>')
    o.append(note(PAD_L, h - 24,
                  f'Box is the interquartile range, bar the median, whiskers the min and max '
                  f'of the per-run medians. No run in the {loc["3072"]["n"]}-run population at '
                  f'3072 envs reached {ref:,}; the highest was {loc["3072"]["max"]:,.0f}.'))
    return "\n".join(o) + "</svg>"


def chart_cost(d) -> str:
    """Collection cost per simulated env-step, against batch width."""
    loc, ghs = d["local"]["by_env"], gh_rows(d)
    widths = [1024, 2048, 3072, 8192, 16384]
    h, top, bot = 332, 44, 214
    x0, x1 = PAD_L + 56, W - PAD_R - 118
    lo, hi = 0, 160

    def Y(v):
        return bot - (v - lo) / (hi - lo) * (bot - top)

    def X(i):
        return x0 + i * (x1 - x0) / (len(widths) - 1)

    o = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="Per-env-step '
         f'collection cost against batch width">']
    o.append(f'<text class="axlab" x="{PAD_L}" y="20">MICROSECONDS OF COLLECTION PER '
             f'SIMULATED ENV-STEP &#183; LOWER IS FASTER</text>')
    for v in (0, 40, 80, 120, 160):
        o.append(f'<line class="grid" x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}"/>')
        o.append(f'<text class="tick" x="{x0 - 22}" y="{Y(v) + 4:.1f}" text-anchor="end">'
                 f'{v}</text>')
    for i, wv in enumerate(widths):
        o.append(f'<text class="tick" x="{X(i):.1f}" y="{bot + 20}" text-anchor="middle">'
                 f'{wv:,}</text>')
    o.append(f'<text class="axlab" x="{(x0 + x1) / 2:.1f}" y="{bot + 42}" '
             f'text-anchor="middle">PARALLEL ENVIRONMENTS</text>')
    series = [
        ("workstation", "var(--s2)", +22,
         [(widths.index(int(k)), v["us_per_env_step"]) for k, v in loc.items()]),
        ("GH200", "var(--s1)", -13,
         [(widths.index(g["envs"]), g["us"]) for g in ghs]),
    ]
    for name, col, lab_dy, pts in series:
        pts = sorted(pts)
        dpath = " ".join(f'{"M" if j == 0 else "L"}{X(i):.1f},{Y(v):.1f}'
                         for j, (i, v) in enumerate(pts))
        o.append(f'<path d="{dpath}" fill="none" stroke="{col}" stroke-width="2"/>')
        for i, v in pts:
            o.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="5" fill="{col}"/>')
            o.append(f'<text class="val" x="{X(i):.1f}" y="{Y(v) + lab_dy:.1f}" '
                     f'text-anchor="middle">{v:.1f}</text>')
        li, lv = pts[-1]
        o.append(f'<text class="ser" x="{X(li) + 14:.1f}" y="{Y(lv) + 4:.1f}" fill="{col}">'
                 f'{name}</text>')
    o.append(note(PAD_L, h - 40,
                  'Both machines are flat: tripling the batch moves the workstation\u2019s '
                  'cost by 0.01%. The vertical gap between the two lines is the whole '
                  'result, and it closes at no width either machine can hold.'))
    return "\n".join(o) + "</svg>"


def chart_duty(d) -> str:
    """Training hours per calendar day over the span the logs cover."""
    import datetime as dt
    per = {r["day"]: r["hours"] for r in d["local"]["per_day"]}
    d0 = dt.date.fromisoformat(d["local"]["first_day"])
    d1 = dt.date.fromisoformat(d["local"]["last_day"])
    days = [(d0 + dt.timedelta(days=i)) for i in range((d1 - d0).days + 1)]
    h, top, bot = 300, 44, 200
    x0, x1 = PAD_L, W - PAD_R - 40
    bw = (x1 - x0) / len(days)

    def Y(v):
        return bot - v / 24.0 * (bot - top)

    o = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="Training hours '
         f'per calendar day">']
    o.append(f'<text class="axlab" x="{PAD_L}" y="20">GPU-HOURS OF TRAINING PER CALENDAR '
             f'DAY &#183; {d0.isoformat()} TO {d1.isoformat()}</text>')
    for v in (0, 8, 16, 24):
        o.append(f'<line class="grid" x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}"/>')
        o.append(f'<text class="tick" x="{x0 - 10}" y="{Y(v) + 4:.1f}" text-anchor="end">'
                 f'{v} h</text>')
    for i, day in enumerate(days):
        v = per.get(day.isoformat(), 0.0)
        x = x0 + i * bw
        if v <= 0:
            o.append(f'<rect x="{x:.1f}" y="{bot - 2:.1f}" width="{bw - 1.6:.1f}" '
                     f'height="2" fill="var(--rule)"/>')
            continue
        o.append(f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{bw - 1.6:.1f}" '
                 f'height="{bot - Y(v):.1f}" rx="2" fill="var(--s2)"/>')
    for lab, i in (("Jul 10", 0), ("Aug 1", (dt.date(2026, 8, 1) - d0).days),
                   ("Aug 28", len(days) - 1)):
        o.append(f'<text class="tick" x="{x0 + i * bw + bw / 2:.1f}" y="{bot + 20}" '
                 f'text-anchor="middle">{lab}</text>')
    dc = d["local"]["duty_cycle"]
    o.append(note(PAD_L, h - 56,
                  f'{d["local"]["train_hours"]} hours of training spread over '
                  f'{d["local"]["span_days"]} days, a duty cycle of {dc}%. '
                  f'{d["local"]["active_days"]} days carried a run; the other '
                  f'{d["local"]["span_days"] - d["local"]["active_days"]} are the grey '
                  f'baseline. On the days it ran at all the GPU averaged '
                  f'{d["local"]["train_hours"] / d["local"]["active_days"]:.1f} hours, so the '
                  f'idle time is not runs queueing for the card.'))
    return "\n".join(o) + "</svg>"


def chart_budget(d) -> str:
    """What a run asks for against where it converged."""
    c = d["convergence"]
    h = 190
    x0, x1 = PAD_L, W - PAD_R - 150
    bh = 30

    def X(i):
        return x0 + i / c["budget_iters"] * (x1 - x0)

    o = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="Converged '
         f'iteration against requested budget">']
    o.append(f'<text class="axlab" x="{PAD_L}" y="20">PPO ITERATIONS &#183; b33&#8217;S OWN '
             f'20M-TIMESTEP RUN</text>')
    y = 46
    o.append(f'<text class="ser" x="{PAD_L - 12}" y="{y + bh * .7}" text-anchor="end" '
             f'fill="var(--ink)">requested</text>')
    o.append(f'<rect x="{x0}" y="{y}" width="{x1 - x0:.1f}" height="{bh}" rx="4" '
             f'fill="var(--sunk)" stroke="var(--rule)"/>')
    o.append(f'<rect x="{x0}" y="{y}" width="{X(c["converged_iter"]) - x0:.1f}" '
             f'height="{bh}" rx="4" fill="var(--good)"/>')
    o.append(f'<text class="val" x="{x1 + 12}" y="{y + bh * .7}">'
             f'{c["budget_iters"]} iters</text>')
    o.append(f'<text class="val" x="{X(c["converged_iter"]) + 10:.1f}" '
             f'y="{y + bh * .7}" fill="var(--good)">converged at iteration '
             f'{c["converged_iter"]}</text>')
    yy = y + bh + 30
    for lab, it, v in (("iteration 26", 26, c["align_it26"]),
                       ("iteration 134", 134, c["align_it134"]),
                       ("iteration 270", 270, c["align_it270"])):
        o.append(f'<line x1="{X(it):.1f}" y1="{y + bh}" x2="{X(it):.1f}" y2="{yy - 6:.1f}" '
                 f'stroke="var(--rule)" stroke-width="1"/>')
        o.append(f'<text class="tick" x="{X(it):.1f}" y="{yy + 6}" text-anchor="middle">'
                 f'{lab}</text>')
        o.append(f'<text class="val" x="{X(it):.1f}" y="{yy + 24}" text-anchor="middle">'
                 f'{v}</text>')
    o.append(f'<text class="mark" x="{PAD_L - 12}" y="{yy + 24}" text-anchor="end">'
             f'TARGET-AXIS ALIGNMENT</text>')
    o.append(note(PAD_L, h - 26,
                  f'Target-axis alignment moves {c["align_it26"]} \u2192 {c["align_it134"]} '
                  f'\u2192 {c["align_it270"]} over the remaining '
                  f'{c["budget_iters"] - c["converged_iter"]} iterations, which cost '
                  f'{(c["budget_iters"] - c["converged_iter"]) / c["budget_iters"] * 20:.1f}M '
                  f'of the 20M timesteps.'))
    return "\n".join(o) + "</svg>"


# ---------------------------------------------------------------- tables

def table_seeds(d) -> str:
    """n = 15.7 sigma^2 / Delta^2, priced at both sigmas."""
    s = d["seed_band"]
    loc_h = 20e6 / d["local"]["by_env"]["3072"]["med"] / 3600
    gh_h = 20e6 / d["deltaai"]["headline_sps"] / 3600
    o = ['<div class="tw"><table><thead><tr>'
         '<th>Resolvable difference in mean held-cos</th>'
         '<th class="num">Seeds at &#963; = 0.35</th>'
         '<th class="num">20 designs, GH200-h</th>'
         '<th class="num">Seeds at &#963; = 0.032</th>'
         '<th class="num">20 designs, GH200-h</th>'
         '<th class="num">&#8230; on the workstation</th>'
         '</tr></thead><tbody>']
    for delta in (0.5, 0.3, 0.2, 0.1):
        na = max(1, math.ceil(15.7 * s["assumed_sd"] ** 2 / delta ** 2))
        nm = max(1, math.ceil(15.7 * s["measured_sd_core"] ** 2 / delta ** 2))
        hi = ' class="hi"' if delta == 0.2 else ""
        o.append(f'<tr{hi}><td>&#916; = {delta}</td>'
                 f'<td class="num">{na}</td><td class="num">{20 * na * gh_h:,.0f}</td>'
                 f'<td class="num">{nm}</td><td class="num">{20 * nm * gh_h:,.0f}</td>'
                 f'<td class="num">{20 * nm * loc_h:,.1f} h</td></tr>')
    o.append('</tbody></table></div>')
    return "\n".join(o)


def table_machines(d) -> str:
    loc, ghs = d["local"]["by_env"], gh_rows(d)
    g3 = next(g for g in ghs if g["envs"] == 3072)
    med = loc["3072"]["med"]
    rows = [
        ("GPU", d["local"]["gpu"], d["deltaai"]["gpu"]),
        ("SM clock under load", f'{d["local"]["sm_clock_mhz_under_load"]:,} MHz (measured)',
         f'{d["deltaai"]["sm_clock_mhz_spec"]:,} MHz (spec, not measured on the node)'),
        ("Steps/s at 3072 envs", f'{med:,.0f} median of {loc["3072"]["n"]} runs',
         f'{g3["sps"]:,.0f} single job'),
        ("Collection per env-step", f'{loc["3072"]["us_per_env_step"]:.1f} &#181;s',
         f'{g3["us"]:.1f} &#181;s'),
        ("Best batch width found", "3072 (nothing above helps)",
         "3072 (nothing above helps, to 95 GiB)"),
        ("Concurrent runs", "1", "as many as the balance and the queue allow"),
        ("Charged", "electricity",
         f'1 SU per reserved GPU-hour, rounded up &#183; '
         f'{d["deltaai"]["credits_per_gpu_hour"]} ACCESS credits per hour'),
    ]
    o = ['<div class="tw"><table><thead><tr><th></th><th>Workstation</th>'
         '<th>DeltaAI <span class="mono">ghx4</span></th></tr></thead><tbody>']
    for a, b, c in rows:
        o.append(f'<tr><td class="wrap"><strong>{a}</strong></td><td class="wrap">{b}</td>'
                 f'<td class="wrap">{c}</td></tr>')
    o.append('</tbody></table></div>')
    return "\n".join(o)


def main() -> None:
    d = json.loads(DATA.read_text())
    loc = d["local"]
    ghs = gh_rows(d)
    g3 = next(g for g in ghs if g["envs"] == 3072)
    med3 = loc["by_env"]["3072"]["med"]
    loc_h = 20e6 / med3 / 3600
    gh_h = 20e6 / d["deltaai"]["headline_sps"] / 3600
    sb = d["seed_band"]

    sub = {
        "N_LOGS": f'{loc["n_logs"]:,}',
        "N_ITERS": f'{loc["n_iterations"]:,}',
        "N_GSTEPS": f'{loc["total_env_steps"] / 1e9:.2f}',
        "TRAIN_H": f'{loc["train_hours"]}',
        "SPAN_DAYS": f'{loc["span_days"]}',
        "ACTIVE_DAYS": f'{loc["active_days"]}',
        "IDLE_DAYS": f'{loc["span_days"] - loc["active_days"]}',
        "DUTY": f'{loc["duty_cycle"]}',
        "H_PER_ACTIVE": f'{loc["train_hours"] / loc["active_days"]:.1f}',
        "FIRST_DAY": loc["first_day"], "LAST_DAY": loc["last_day"],
        "MED3": f'{med3:,.0f}', "N3": str(loc["by_env"]["3072"]["n"]),
        "Q1_3": f'{loc["by_env"]["3072"]["q1"]:,.0f}',
        "Q3_3": f'{loc["by_env"]["3072"]["q3"]:,.0f}',
        "MAX3": f'{loc["by_env"]["3072"]["max"]:,.0f}',
        "MIN3": f'{loc["by_env"]["3072"]["min"]:,.0f}',
        "MED2": f'{loc["by_env"]["2048"]["med"]:,.0f}',
        "N2": str(loc["by_env"]["2048"]["n"]),
        "US1": f'{loc["by_env"]["1024"]["us_per_env_step"]:.2f}',
        "US2": f'{loc["by_env"]["2048"]["us_per_env_step"]:.2f}',
        "US3": f'{loc["by_env"]["3072"]["us_per_env_step"]:.2f}',
        "GH_SPS": f'{d["deltaai"]["headline_sps"]:,}',
        "GH_SPS3": f'{g3["sps"]:,.0f}', "GH_US3": f'{g3["us"]:.1f}',
        "GH_US_ASYM": f'{ghs[-1]["us"]:.1f}',
        "RATIO_MED": f'{d["local"]["gh200_ratio_vs_median"]:.2f}',
        "RATIO_PUB": f'{d["local"]["gh200_ratio_vs_published"]:.2f}',
        "PUB_REF": f'{d["published_local_reference_sps"]:,}',
        "LEARN_PCT": f'{loc["learn_share_pct"]}',
        "LEARN_MAX": f'{loc["learn_share_max_pct"]}',
        "COLLECT_PCT": f'{100 - loc["learn_share_pct"]:.2f}',
        "UNACC": f'{loc["unaccounted_pct"]}',
        "CLK_LOC": f'{loc["sm_clock_mhz_under_load"]:,}',
        "CLK_GH": f'{d["deltaai"]["sm_clock_mhz_spec"]:,}',
        "CLK_RATIO": f'{loc["sm_clock_mhz_under_load"] / d["deltaai"]["sm_clock_mhz_spec"]:.2f}',
        "COST_RATIO": f'{ghs[-1]["us"] / loc["by_env"]["3072"]["us_per_env_step"]:.2f}',
        "CLK_GAP": f'{abs(1 - (ghs[-1]["us"] / loc["by_env"]["3072"]["us_per_env_step"]) / (loc["sm_clock_mhz_under_load"] / d["deltaai"]["sm_clock_mhz_spec"])) * 100:.0f}',
        "SB_N": str(sb["n_draws"]), "SB_GPUH": f'{sb["gpu_hours"]}',
        "SB_SU": str(sb["su_billed"]),
        "SB_SD_ASSUMED": f'{sb["assumed_sd"]}',
        "SB_SD_CORE": f'{sb["measured_sd_core"]}',
        "SB_SD_ALL": f'{sb["measured_sd_all"]}',
        "SB_WITHIN": f'{sb["within_draw_sd"]}',
        "SB_REF": f'{sb["reference_cos"]}', "SB_RANK": sb["rank"],
        "SB_BILL_OVER": f'{sb["su_billed"] / sb["gpu_hours"]:.2f}',
        "SEQ_LOCAL_H": f'{sb["n_draws"] * loc_h:.1f}',
        "IDEAL_GH_H": f'{gh_h:.1f}',
        "IDEAL_SPEEDUP": f'{sb["n_draws"] * loc_h / gh_h:.0f}',
        "EIGHT_WAY_H": f'{2 * gh_h:.1f}',
        "EIGHT_WAY_SPEEDUP": f'{sb["n_draws"] * loc_h / (2 * gh_h):.1f}',
        "BALANCE": str(d["deltaai"]["balance_su"]),
        "DEPOSITED": str(d["deltaai"]["deposited_su"]),
        "HISTORY_GH_H": f'{loc["total_env_steps"] / d["deltaai"]["headline_sps"] / 3600:.0f}',
        "HISTORY_LOC_H": f'{loc["total_env_steps"] / med3 / 3600:.0f}',
        "CONV_IT": str(d["convergence"]["converged_iter"]),
        "CONV_BUDGET": str(d["convergence"]["budget_iters"]),
        "BUDGET_30M": str(next(b["logs"] for b in loc["budgets"] if b["timesteps_m"] == 30.0)),
        "BUDGET_20M": str(next(b["logs"] for b in loc["budgets"] if b["timesteps_m"] == 20.0)),
        "CHART_THROUGHPUT": chart_throughput(d),
        "CHART_COST": chart_cost(d),
        "CHART_DUTY": chart_duty(d),
        "CHART_BUDGET": chart_budget(d),
        "TABLE_MACHINES": table_machines(d),
        "TABLE_SEEDS": table_seeds(d),
    }

    html = TPL.read_text()
    for k, v in sub.items():
        html = html.replace("{{" + k + "}}", str(v))
    missing = set(__import__("re").findall(r"\{\{([A-Z0-9_]+)\}\}", html))
    if missing:
        raise SystemExit(f"unsubstituted placeholders: {sorted(missing)}")
    OUT.write_text(html)
    print(f"{OUT}  ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
