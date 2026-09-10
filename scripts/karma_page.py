#!/usr/bin/env python3
"""Build the KaRMA evaluation page from the sweep's own JSON.

    python3 scripts/karma_collect.py  --sample ... --work ... --out karma_table.json
    ~/miniconda3/bin/python scripts/karma_analysis.py --table ... --out karma_analysis.json
    python3 scripts/karma_page.py

Everything on the page is read back out of karma_table.json / karma_analysis.json /
seed_geometry.json, so re-running the sweep and re-running this is the whole update path.
Charts are hand-emitted theme-aware SVG; media under media/ are inlined as data URIs, so
the output file is the whole artifact.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import statistics as st

ROOT = "docs/experiments/20260910-karma_metric"
MEDIA = f"{ROOT}/media"
TPL = "scripts/karma_page.template.html"
OUT = f"{ROOT}/20260910-karma_metric_evaluation.html"

W = 1020
PAD_L, PAD_R = 132, 24

# Two categorical series at most, per the palette note carried by every page in this
# programme: amber = KaRMA, blue = this programme's own predictors, grey = reference.
A, B, REF = "var(--s1)", "var(--s2)", "var(--ref)"


def uri(name: str) -> str:
    p = os.path.join(MEDIA, name)
    mt = mimetypes.guess_type(p)[0] or "application/octet-stream"
    return f"data:{mt};base64," + base64.b64encode(open(p, "rb").read()).decode()


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ------------------------------------------------------------------------- charts

def chart_spread(tab, pub) -> str:
    """KaRMA-T for one hand family against the sixteen published hands, log axis."""
    ours = sorted(r["karma_t"] for r in tab
                  if r["variant"] == "scaled" and r["pair"] == "thumb-index"
                  and r["karma_t"] > 0)
    theirs = sorted((r["KaRMA_T"], r["robot"]) for r in pub)
    import math
    lo = math.log10(min(ours[0], theirs[0][0])) - 0.08
    hi = math.log10(max(ours[-1], theirs[-1][0])) + 0.08
    x0, x1 = PAD_L, W - PAD_R - 10

    def X(v):
        return x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)

    h = 250
    o = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="KaRMA-T for '
         f'{len(ours)} real_v1 designs against the sixteen published hands">']
    o.append(f'<text class="axlab" x="{x0}" y="18">KaRMA-T &#183; LOG SCALE &#183; '
             f'DIMENSIONLESS REACHABLE OBJECT-CENTRE VOLUME</text>')
    for e in range(int(math.floor(lo)), int(math.ceil(hi)) + 1):
        v = 10.0 ** e
        if not (lo <= math.log10(v) <= hi):
            continue
        o.append(f'<line class="grid" x1="{X(v):.1f}" y1="34" x2="{X(v):.1f}" y2="{h-56}"/>')
        o.append(f'<text class="tick" x="{X(v):.1f}" y="{h-40}" text-anchor="middle">'
                 f'1e{e}</text>')
    o.append(f'<text class="ser" x="{x0-10}" y="66" text-anchor="end" fill="var(--ink)">'
             f'real_v1</text>')
    o.append(f'<text class="tick" x="{x0-10}" y="82" text-anchor="end">n={len(ours)}</text>')
    for n, v in enumerate(ours):
        # Deterministic jitter: Python salts str hashes per process, so hash() here would
        # make the page fail to regenerate identically.
        o.append(f'<circle cx="{X(v):.1f}" cy="{62 + (n * 7) % 26 - 13}" r="2.6" '
                 f'fill="{A}" fill-opacity="0.5"/>')
    o.append(f'<text class="ser" x="{x0-10}" y="146" text-anchor="end" fill="var(--ink)">'
             f'published</text>')
    o.append(f'<text class="tick" x="{x0-10}" y="162" text-anchor="end">n=16</text>')
    for i, (v, name) in enumerate(theirs):
        y = 132 + (i % 3) * 13
        o.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="3.4" fill="{B}"/>')
        if name in ("leap", "allegro", "inspire", "ability_hand_right", "dclaw", "shadowhand"):
            o.append(f'<text class="mark" x="{X(v):.1f}" y="{y-7}" text-anchor="middle">'
                     f'{esc(name[:9])}</text>')
    med_o, med_p = st.median(ours), st.median([t[0] for t in theirs])
    for v, lab, col in ((med_o, "real_v1 median", A), (med_p, "published median", B)):
        o.append(f'<line x1="{X(v):.1f}" y1="34" x2="{X(v):.1f}" y2="{h-58}" stroke="{col}" '
                 f'stroke-width="1.4" stroke-dasharray="4 3"/>')
    o.append(f'<text class="note" x="{x0}" y="{h-20}">One topology, one set of finger links, '
             f'six mount coordinates: {ours[-1]/ours[0]:.0f}&#215; between the best and worst '
             f'of them, against {theirs[-1][0]/theirs[0][0]:.0f}&#215; across sixteen '
             f'different commercial hands.</text>')
    o.append(f'<text class="note" x="{x0}" y="{h-4}">Dashed lines are the two medians. '
             f'Vertical jitter on the real_v1 points is cosmetic.</text>')
    return "\n".join(o) + "</svg>"


def chart_auc(an, arm_key: str) -> str:
    """AUC for retention: KaRMA's scores against this programme's own predictors."""
    arm = an["arms"][arm_key]
    a = arm["auc_retained"]
    rows = []
    for k, lab, col in (
            ("karma_r", "KaRMA-R  rotational coverage", A),
            ("karma_t", "KaRMA-T  translational volume", A),
            ("n_voxels", "voxels reached", A),
            ("karma_s", "KaRMA-S  seed sensitivity", A),
            ("karma_trs_logistic_cv", "KaRMA T+R+S, logistic (5-fold CV)", A),
            ("seed_depth_mm", "depth of the pinch KaRMA chose", REF),
            ("ruler_closeness_of_the_scored_pair",
             "ruler: how close the two scored mounts are", B),
            ("ruler_small_x_sep", "ruler: thumb-to-pair span, small is high", B),
            ("ruler_small_y_sep", "ruler: pair opening, small is high", B),
            ("six_mounts_logistic_cv", "the six mount coordinates, logistic (5-fold CV)", B),
            ("mounts_plus_karma_logistic_cv", "six mounts + KaRMA T/R/S, logistic (CV)", B)):
        if k in a:
            rows.append((lab, a[k]["auc"], a[k].get("ci"), col))
    bh, gap = 24, 11
    h = 62 + len(rows) * (bh + gap) + 54
    x0, x1 = PAD_L + 186, W - PAD_R - 66

    def X(v):
        return x0 + (v - 0.5) / 0.45 * (x1 - x0)

    o = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="AUC for '
         f'predicting retention">']
    o.append(f'<text class="axlab" x="{PAD_L}" y="18">AUC FOR PREDICTING WHICH DESIGNS '
             f'SURVIVE THE RETENTION SCREEN &#183; n = {arm["n"]}, '
             f'{arm["n_retained"]} RETAINED</text>')
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        o.append(f'<line class="grid" x1="{X(t):.1f}" y1="34" x2="{X(t):.1f}" y2="{h-52}"/>')
        o.append(f'<text class="tick" x="{X(t):.1f}" y="{h-36}" text-anchor="middle">'
                 f'{t:.1f}</text>')
    for i, (lab, v, ci, col) in enumerate(rows):
        y = 42 + i * (bh + gap)
        o.append(f'<text class="tick" x="{PAD_L}" y="{y+bh*0.72:.0f}">{esc(lab)}</text>')
        w = max(X(v) - X(0.5), 1.5)
        o.append(f'<rect x="{X(0.5):.1f}" y="{y}" width="{w:.1f}" height="{bh}" rx="4" '
                 f'fill="{col}"/>')
        if ci:
            o.append(f'<line x1="{X(ci[0]):.1f}" y1="{y+bh/2}" x2="{X(ci[1]):.1f}" '
                     f'y2="{y+bh/2}" stroke="var(--ink2)" stroke-width="1.3"/>')
            for e in ci:
                o.append(f'<line x1="{X(e):.1f}" y1="{y+4}" x2="{X(e):.1f}" y2="{y+bh-4}" '
                         f'stroke="var(--ink2)" stroke-width="1.3"/>')
        o.append(f'<text class="val" x="{x1+9}" y="{y+bh*0.74:.0f}">{v:.3f}</text>')
    o.append(f'<line x1="{X(0.5):.1f}" y1="34" x2="{X(0.5):.1f}" y2="{h-52}" '
             f'stroke="var(--ink2)" stroke-width="1.4"/>')
    o.append(f'<text class="note" x="{PAD_L}" y="{h-16}">0.5 is a coin flip. Bars are the '
             f'AUC, whiskers a 2,000-sample bootstrap 95% interval. Amber = KaRMA, blue = '
             f'this programme&#8217;s own predictor on the same designs.</text>')
    return "\n".join(o) + "</svg>"


def chart_depth(tab, ops) -> str:
    """Where KaRMA pinches, against the depth band the deployed plans actually grasp at."""
    d = sorted(r["seed_depth_mm"] for r in tab
               if r["variant"] == "scaled" and r["pair"] == "thumb-index")
    grips = sorted(v["plan"]["grip_depth_mm"] for v in ops["hands"].values())
    x0, x1 = PAD_L, W - PAD_R - 10
    hi = max(d[-1], grips[-1]) * 1.06

    def X(v):
        return x0 + v / hi * (x1 - x0)

    h = 226
    nb = 34
    bins = [0] * nb
    for v in d:
        bins[min(nb - 1, int(v / hi * nb))] += 1
    mx = max(bins)
    o = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="Depth below the '
         f'palm at which KaRMA chose to pinch">']
    o.append(f'<text class="axlab" x="{x0}" y="18">DEPTH BELOW THE MOUNTING PLANE, mm</text>')
    o.append(f'<rect x="{X(grips[0]):.1f}" y="32" width="{X(grips[-1])-X(grips[0]):.1f}" '
             f'height="{h-92}" fill="{B}" fill-opacity="0.13"/>')
    o.append(f'<text class="mark" x="{X(grips[0])+7:.1f}" y="46">GRIP DEPTH OF THE EIGHT '
             f'DEPLOYED PLANS &#183; {grips[0]:.1f}&#8211;{grips[-1]:.1f} mm</text>')
    for i, c in enumerate(bins):
        if not c:
            continue
        bw = (x1 - x0) / nb
        bhh = (c / mx) * (h - 108)
        o.append(f'<rect x="{x0+i*bw+0.7:.1f}" y="{h-72-bhh:.1f}" width="{bw-1.4:.1f}" '
                 f'height="{bhh:.1f}" rx="2" fill="{A}"/>')
    for t in range(0, int(hi) + 1, 10):
        o.append(f'<text class="tick" x="{X(t):.1f}" y="{h-54}" text-anchor="middle">'
                 f'{t}</text>')
    med = st.median(d)
    o.append(f'<line x1="{X(med):.1f}" y1="32" x2="{X(med):.1f}" y2="{h-72}" stroke="{A}" '
             f'stroke-width="1.6" stroke-dasharray="4 3"/>')
    o.append(f'<text class="note" x="{x0}" y="{h-30}">KaRMA picks its own pinch. Over '
             f'{len(d)} designs it chose a median depth of {med:.1f} mm &#8212; dashed '
             f'line &#8212; where the deployed grasps sit at '
             f'{grips[0]:.1f}&#8211;{grips[-1]:.1f} mm.</text>')
    o.append(f'<text class="note" x="{x0}" y="{h-13}">'
             f'{100*sum(1 for v in d if v < grips[0])/len(d):.0f}% of the pinches it scored '
             f'are shallower than every grasp this hand has been deployed with.</text>')
    return "\n".join(o) + "</svg>"


def chart_clip(tab, an) -> str:
    """KaRMA-T against the highest residual clip a design still passes retention at."""
    sub = [r for r in tab if r["variant"] == "scaled" and r["pair"] == "thumb-index"
           and r.get("max_clip_rad")]
    if len(sub) < 30:
        return ""
    clips = sorted({r["max_clip_rad"] for r in sub})
    cw, gap = 150, 18
    h = 300
    y0, y1 = 44, h - 74
    vals = [r["karma_t"] for r in sub]
    import math
    lo, hi = math.log10(min(vals)), math.log10(max(vals))

    def Y(v):
        return y1 - (math.log10(v) - lo) / (hi - lo) * (y1 - y0)

    o = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="KaRMA-T against '
         f'the highest residual clip a design still passes at">']
    o.append(f'<text class="axlab" x="{PAD_L}" y="18">KaRMA-T (log) BY THE HIGHEST RESIDUAL '
             f'CLIP THE DESIGN STILL PASSES RETENTION AT &#183; n = {len(sub)}</text>')
    for i, c in enumerate(clips):
        x = PAD_L + i * (cw + gap)
        grp = [r["karma_t"] for r in sub if r["max_clip_rad"] == c]
        o.append(f'<text class="tick" x="{x + cw / 2:.0f}" y="{h - 52}" '
                 f'text-anchor="middle">{c:.2f} rad</text>')
        o.append(f'<text class="mark" x="{x + cw / 2:.0f}" y="{h - 36}" '
                 f'text-anchor="middle">n = {len(grp)}</text>')
        for n, v in enumerate(grp):
            o.append(f'<circle cx="{x + 14 + (n * 11) % (cw - 28):.1f}" cy="{Y(v):.1f}" '
                     f'r="2.7" fill="{A}" fill-opacity="0.55"/>')
        med = sorted(grp)[len(grp) // 2]
        o.append(f'<line x1="{x}" y1="{Y(med):.1f}" x2="{x + cw}" y2="{Y(med):.1f}" '
                 f'stroke="var(--ink)" stroke-width="1.8"/>')
        o.append(f'<text class="val" x="{x + cw + 4:.0f}" y="{Y(med) + 4:.1f}">'
                 f'{med:.4f}</text>')
    rho = an["arms"]["scaled|thumb-index"]["rho_max_clip"]["karma_t"]
    o.append(f'<text class="note" x="{PAD_L}" y="{h - 14}">A bigger clip is a HARDER test '
             f'under this gate, not an easier one. Spearman {rho["rho"]:+.3f} '
             f'(p = {rho["p"]:.1e}); heavy rules are group medians.</text>')
    return "\n".join(o) + "</svg>"


def chart_pairs(tab) -> str:
    """Thumb-index against thumb-middle: does the choice of two fingers matter?"""
    import math
    by = {}
    for r in tab:
        if r["variant"] == "scaled":
            by.setdefault(r["design"], {})[r["pair"]] = r
    pts = [(v["thumb-index"]["karma_t"], v["thumb-middle"]["karma_t"], v["thumb-index"])
           for v in by.values() if "thumb-index" in v and "thumb-middle" in v]
    pts = [p for p in pts if p[0] > 0 and p[1] > 0]
    if len(pts) < 10:
        return ""
    vals = [p[0] for p in pts] + [p[1] for p in pts]
    lo, hi = math.log10(min(vals)) - 0.1, math.log10(max(vals)) + 0.1
    S, pad = 300, 52
    o = [f'<svg class="chart" viewBox="0 0 {W} {S+96}" role="img" aria-label="KaRMA-T for '
         f'the thumb-index pair against the thumb-middle pair">']

    def X(v):
        return PAD_L + (math.log10(v) - lo) / (hi - lo) * S

    def Y(v):
        return 34 + S - (math.log10(v) - lo) / (hi - lo) * S

    o.append(f'<text class="axlab" x="{PAD_L}" y="18">KaRMA-T, THUMB&#8211;INDEX (x) '
             f'AGAINST THUMB&#8211;MIDDLE (y) &#183; LOG&#8211;LOG &#183; '
             f'n = {len(pts)}</text>')
    o.append(f'<line x1="{PAD_L}" y1="{34+S}" x2="{PAD_L+S}" y2="34" stroke="{REF}" '
             f'stroke-width="1.2" stroke-dasharray="4 3"/>')
    for e in range(int(math.floor(lo)), int(math.ceil(hi)) + 1):
        v = 10.0 ** e
        if not (lo <= e <= hi):
            continue
        o.append(f'<line class="grid" x1="{X(v):.1f}" y1="34" x2="{X(v):.1f}" y2="{34+S}"/>')
        o.append(f'<line class="grid" x1="{PAD_L}" y1="{Y(v):.1f}" x2="{PAD_L+S}" '
                 f'y2="{Y(v):.1f}"/>')
        o.append(f'<text class="tick" x="{X(v):.1f}" y="{34+S+16}" text-anchor="middle">'
                 f'1e{e}</text>')
        o.append(f'<text class="tick" x="{PAD_L-8}" y="{Y(v)+4:.1f}" text-anchor="end">'
                 f'1e{e}</text>')
    for xv, yv, meta in pts:
        col = B if meta["retained"] else REF
        o.append(f'<circle cx="{X(xv):.1f}" cy="{Y(yv):.1f}" r="3" fill="{col}" '
                 f'fill-opacity="{0.85 if meta["retained"] else 0.4}"/>')
    o.append(f'<text class="note" x="{PAD_L+S+34}" y="70">The dashed line is agreement.</text>')
    o.append(f'<text class="note" x="{PAD_L+S+34}" y="88">Blue = retained by this '
             f'programme&#8217;s screen,</text>')
    o.append(f'<text class="note" x="{PAD_L+S+34}" y="104">grey = not retained.</text>')
    return "\n".join(o) + "</svg>"


def table_deployed(an) -> str:
    if "deployed" not in an:
        return ""
    rows = an["deployed"]["rows"]
    o = ['<div class="tw"><table><thead><tr><th>hand</th><th>design</th>'
         '<th class="num">KaRMA-T</th><th class="num">KaRMA-R</th><th class="num">KaRMA-S</th>'
         '<th class="num">voxels</th><th class="num">L_ref mm</th>'
         '<th class="num">screened turn cos</th></tr></thead><tbody>']
    for r in rows:
        c = r.get("nom_cos")
        o.append(f'<tr><td class="mono">{esc(r["id"])}</td><td class="mono">'
                 f'{esc(r["design"])}</td>'
                 f'<td class="num">{r["karma_t"]:.5f}</td>'
                 f'<td class="num">{r["karma_r"]:.3f}</td>'
                 f'<td class="num">{r["karma_s"]:.3f}</td>'
                 f'<td class="num">{r["n_voxels"]}</td>'
                 f'<td class="num">{r["l_ref_mm"]:.1f}</td>'
                 f'<td class="num">{"&#183;" if c is None else f"{c:+.3f}"}</td></tr>')
    return "\n".join(o) + "</tbody></table></div>"


def fig_replay() -> str:
    f = f"{MEDIA}/20260910-karma_replay.mp4"
    if not os.path.exists(f):
        return ""
    r = json.load(open(f"{ROOT}/replay.json"))["summary"]
    return (f'<figure><video autoplay loop muted playsinline src="{uri(os.path.basename(f))}">'
            f'</video><figcaption>KaRMA&#8217;s own rolling path for the highest-scoring hand '
            f'in the sweep, driven joint for joint through MuJoCo contact physics on '
            f'KaRMA&#8217;s own capsule geometry, gravity off, with {r["squeeze_deg"]:.0f}'
            f'&#176; of commanded squeeze added because the metric&#8217;s pinch carries no '
            f'preload. The sphere travels {r["actual_travel_mm"]:.0f} mm against the '
            f'{r["predicted_travel_mm"]:.0f} mm predicted, ends {r["final_error_mm"]:.0f} mm '
            f'from the predicted pose, and is never held at two contacts.</figcaption>'
            f'</figure>')


def fig_reach(tab) -> str:
    hi = f"{MEDIA}/20260910-reach_best.png"
    lo = f"{MEDIA}/20260910-reach_worst.png"
    if not (os.path.exists(hi) and os.path.exists(lo)):
        return ""
    sub = sorted([r for r in tab if r["variant"] == "scaled" and r["pair"] == "thumb-index"],
                 key=lambda r: r["n_voxels"])
    a, b = sub[-1], sub[0]
    return (f'<div class="duo">'
            f'<figure><img src="{uri(os.path.basename(hi))}">'
            f'<figcaption>{esc(a["design"])} &#183; {a["n_voxels"]} voxels, KaRMA-R '
            f'{a["karma_r"]:.3f}</figcaption></figure>'
            f'<figure><img src="{uri(os.path.basename(lo))}">'
            f'<figcaption>{esc(b["design"])} &#183; {b["n_voxels"]} voxels, KaRMA-R '
            f'{b["karma_r"]:.3f}</figcaption></figure></div>')


def main() -> None:
    tab = json.load(open(f"{ROOT}/karma_table.json"))
    an = json.load(open(f"{ROOT}/karma_analysis.json"))
    geo = json.load(open(f"{ROOT}/seed_geometry.json"))
    ops = json.load(open("docs/experiments/OPERATING_POINTS.json"))
    import yaml
    pub = yaml.safe_load(open(f"{ROOT}/published_16.yaml"))["rows"]

    prim = "scaled|thumb-index"
    sub = [r for r in tab if r["variant"] == "scaled" and r["pair"] == "thumb-index"]

    near, pad, onpad = [], [], []
    for r in geo:
        if "error" in r:
            continue
        for f in r["contact_fingers"]:
            if f in r["fingers"]:
                near.append(r["fingers"][f]["gap_nearest_mm"])
                onpad.append(r["fingers"][f]["on_pad"])
                if r["fingers"][f]["gap_pad_mm"] is not None:
                    pad.append(r["fingers"][f]["gap_pad_mm"])

    vals = {
        "N_DESIGNS": f"{len(sub):,}",
        "N_RUNS": f"{an['n_runs']:,}",
        "N_RETAINED": str(sum(r["retained"] for r in sub)),
        "CHART_SPREAD": chart_spread(tab, pub),
        "CHART_AUC": chart_auc(an, prim) if prim in an["arms"] else "",
        "CHART_DEPTH": chart_depth(tab, ops),
        "CHART_CLIP": chart_clip(tab, an),
        "CHART_PAIRS": chart_pairs(tab),
        "TABLE_DEPLOYED": table_deployed(an),
        "SPREAD_RATIO": f"{an['vs_published_16']['real_v1_T']['ratio_max_min']:.0f}",
        "PUB_RATIO": f"{an['vs_published_16']['published_T']['ratio_max_min']:.0f}",
        "PCTILE": f"{an['vs_published_16']['real_v1_median_percentile_in_published']:.0f}",
        "GAP_MED": f"{st.median(near):+.2f}" if near else "n/a",
        "PAD_PCT": f"{100*sum(onpad)/len(onpad):.0f}" if onpad else "n/a",
        "DEPTH_MED": f"{st.median([r['seed_depth_mm'] for r in sub]):.1f}",
        "FIG_REPLAY": fig_replay(),
        "FIG_REACH": fig_reach(tab),
    }
    for k, v in (("AUC_R", "karma_r"), ("AUC_T", "karma_t"), ("AUC_S", "karma_s"),
                 ("AUC_VOX", "n_voxels"), ("AUC_MOUNTS", "six_mounts_logistic_cv"),
                 ("AUC_KARMA3", "karma_trs_logistic_cv"),
                 ("AUC_BOTH", "mounts_plus_karma_logistic_cv"),
                 ("AUC_DEPTH", "seed_depth_mm")):
        a = an["arms"].get(prim, {}).get("auc_retained", {}).get(v)
        vals[k] = f"{a['auc']:.3f}" if a else "n/a"
        if a and a.get("ci"):
            vals[k + "_CI"] = f"{a['ci'][0]:.2f}–{a['ci'][1]:.2f}"
        else:
            vals[k + "_CI"] = "n/a"
    # paired contrast and the geometry interpretation
    pd = an["arms"].get(prim, {}).get("paired_delta_auc", {})
    d = pd.get("karma_t_minus_ruler_closeness_of_the_scored_pair")
    vals["DELTA"] = f"{d['delta']:+.3f} AUC" if d else "n/a"
    vals["DELTA_CI"] = (f"95% {d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}") if d else "n/a"
    ru = an["arms"].get(prim, {}).get("auc_retained", {}).get(
        "ruler_closeness_of_the_scored_pair")
    vals["AUC_RULER"] = f"{ru['auc']:.3f}" if ru else "n/a"
    wm = an.get("what_it_measures", {})
    vals["RHO_GEOM"] = (f"{wm['thumb_to_index_mm']['karma_t']['rho']:+.3f}"
                        if "thumb_to_index_mm" in wm else "n/a")
    vals["RHO_LREF"] = (f"{wm['l_ref_mm']['karma_t']['rho']:+.3f}"
                        if "l_ref_mm" in wm else "n/a")
    cf = an["arms"].get(prim, {}).get("auc_confirmed_among_retained", {})
    vals["AUC_CONF_T"] = f"{cf['karma_t']['auc']:.3f}" if "karma_t" in cf else "n/a"
    vals["AUC_CONF_R"] = f"{cf['karma_r']['auc']:.3f}" if "karma_r" in cf else "n/a"
    vals["N_CONF"] = str(an["arms"].get(prim, {}).get("n_confirmed_among_retained", 0))

    mc = an["arms"].get(prim, {}).get("rho_max_clip", {})
    for k, v in (("RHO_CLIP_T", "karma_t"), ("RHO_CLIP_R", "karma_r")):
        vals[k] = f"{mc[v]['rho']:+.3f}" if v in mc else "n/a"
        vals[k + "_P"] = f"{mc[v]['p']:.1e}" if v in mc else "n/a"
    vals["N_CLIP"] = str(an["arms"].get(prim, {}).get("n_with_clip", 0))

    rho = an["arms"].get(prim, {}).get("rho_nom_cos", {})
    for k, v in (("RHO_R", "karma_r"), ("RHO_T", "karma_t")):
        vals[k] = f"{rho[v]['rho']:+.3f}" if v in rho else "n/a"
        vals[k + "_P"] = f"{rho[v]['p']:.2g}" if v in rho else "n/a"
    vals["N_TURN"] = str(an["arms"].get(prim, {}).get("n_with_turn", 0))
    pm = an.get("pair_matched", {})
    au = pm.get("auc", {})
    vals["PM_TI"] = f"{au['thumb-index|karma_t']['auc']:.3f}" if au else "n/a"
    vals["PM_TM"] = f"{au['thumb-middle|karma_t']['auc']:.3f}" if au else "n/a"
    vals["PM_BEST"] = f"{au['best-of-pairs|karma_t']['auc']:.3f}" if au else "n/a"
    pdl = pm.get("paired_delta", {}).get("karma_t")
    vals["PM_DELTA"] = (f"{pdl['delta']:+.3f} AUC, 95% {pdl['ci'][0]:+.3f} to "
                        f"{pdl['ci'][1]:+.3f}") if pdl else "n/a"

    pa = an.get("pair_agreement", {})
    vals["PAIR_RHO"] = f"{pa.get('spearman_T_thumb_index_vs_thumb_middle', float('nan')):+.3f}" \
        if pa else "n/a"
    vals["PAIR_N"] = str(pa.get("n", 0))
    vals["PAIR_RATIO"] = f"{pa.get('median_abs_log2_ratio', float('nan')):.2f}" if pa else "n/a"

    html = open(TPL).read()
    for k, v in vals.items():
        html = html.replace("{{" + k + "}}", str(v))
    left = [t for t in html.split("{{")[1:] if "}}" in t]
    if left:
        print("WARNING unfilled placeholders:", sorted({t.split("}}")[0] for t in left}))
    open(OUT, "w").write(html)
    print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
