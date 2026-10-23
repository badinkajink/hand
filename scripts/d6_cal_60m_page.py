#!/usr/bin/env python3
"""Build the D6-on-cal 60 M-step policy page from the eval JSONs and training curves.

    python3 scripts/d6_cal_60m_page.py
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os

import numpy as np
import interactive_figure as IF
import retro_style
from policy_eval_suite import load_traces

ROOT = "docs/experiments/20260917-d6_cal_60M"
TRACES = f"{ROOT}/traces"                          # re-evaluation of 2026-10-11 (scripts/policy_reeval_traces.py)
TIP_TRACES = "docs/experiments/20260916-tip_gait/traces"
RUN20, RUN60 = "20260916-2142-d6_cal_reorient_s0", "20260917-1141-d6_cal_reorient_gp025_60M_s0"
RUN60M, RUNFT = "20260917-1653-d6_cal_reorient_gpm025_60M_s0", "20260917-1844-d6_cal_reorient_ft_gpm5_20M_s0"
COL = {"60": IF.PALETTE[0], "20": IF.PALETTE[1], "60m": IF.PALETTE[2], "ft": IF.PALETTE[5]}
TPL = "scripts/d6_cal_60m_page.template.html"
OUT = f"{ROOT}/20260917-d6_reorient_policy_60M.html"


def uri(path):
    p = os.path.join(ROOT, path)
    return f"data:{mimetypes.guess_type(p)[0]};base64," + base64.b64encode(open(p, "rb").read()).decode()


def load(name):
    return json.load(open(os.path.join(ROOT, name)))


def reeval(run, key, folder=TRACES):
    """Summary of the 2026-10-11 re-evaluation of one run under one condition (traces/<run>_<key>.json)."""
    return json.load(open(os.path.join(folder, f"{run}_{key}.json")))


def figures(T, e):
    """The four evaluation figures, drawn from the stored traces."""
    pol = lambda on: [{"key": "policy", "label": "Policy", "values": [("60", "60 M, +0.25, model 812"), ("20", "20 M, model 270"),
                                                                    ("60m", "60 M, &#8722;0.25, model 812"), ("ft", "Grip finetune, model 270")],
                       "on": on}]
    item = lambda pid, label, tr, dash="", ev=None: dict(id=f"{pid}{ev or ''}", label=label, trace=tr, color=COL[pid], dash=dash,
                                                       facets={"policy": pid, **({"ev": ev} if ev else {})})
    f60 = IF.eval_figure("eval-60m", [item("60", "60 M, +0.25, model 812", T["60"]["nominal"]),
                                      item("20", "20 M, model 270", T["20"]["nominal"])],
                         facets=[{**pol(["60"])[0], "values": pol(0)[0]["values"][:2]}],
                         caption=(f"Model 812 over 64 deterministic rollouts with the residual from step 58: median and 10&#8211;90 % band "
                                  f"of the signed cosine, the tool height and the pad force sum, and the share of rollouts held. The "
                                  f"policy box adds the 20 M checkpoint, which stops at cos {e['20']['final_cos_mean']:.2f}."))
    ev = [("t", "As trained, residual from step 58", ""), ("r0", "Residual from step 0", "7 4"),
          ("r0s", "Residual from step 0, sampled actions", "2 3")]
    items = [item("60", "60 M, model 812, as trained", T["60"]["nominal"], "", "t"),
             item("60", "60 M, model 812, residual from step 0", T["60"]["nominal_r0"], "7 4", "r0"),
             item("60", "60 M, model 812, residual from step 0, sampled", T["60"]["nominal_r0_stochastic"], "2 3", "r0s"),
             item("20", "20 M, model 270, as trained", T["20"]["nominal"], "", "t"),
             item("20", "20 M, model 270, residual from step 0", T["20old"]["nominal_r0"], "7 4", "r0")]
    legend = [("60 M, model 812", COL["60"], "", ["60t", "60r0", "60r0s"]), ("20 M, model 270", COL["20"], "", ["20t", "20r0"]),
              ("As trained (solid)", "#000", "", ["60t", "20t"]), ("Residual from step 0", "#000", "7 4", ["60r0", "20r0"]),
              ("Residual from step 0, sampled actions", "#000", "2 3", ["60r0s"])]
    fr0 = IF.eval_figure("eval-residual", items, legend=legend, show_band=False,
                         facets=[{**pol(["60"])[0], "values": pol(0)[0]["values"][:2]},
                                 {"key": "ev", "label": "Evaluator", "values": [(k, lab) for k, lab, _ in ev], "on": ["t", "r0"]}],
                         caption=(f"The evaluator settings of the table, 64 rollouts each, medians. With the residual from step 0, model 812 "
                                  f"drops the tool in {e['60r0']['n_lost']} of 64 rollouts. Tools standing on the floor read a cosine "
                                  f"near 1 at a height under 60&#8202;mm. The policy box adds the 20 M checkpoint, which holds the "
                                  f"tool in {int(round(e['20r0']['hold_rate'] * 64))} of 64 rollouts with the residual from step 0 "
                                  f"(cos {e['20r0']['final_cos_mean']:.2f}) and {int(round(e['20']['hold_rate'] * 64))} of 64 as trained "
                                  f"(cos {e['20']['final_cos_mean']:.2f}); its residual was small enough that the lift survived it."))
    fm = IF.eval_figure("eval-gpm", [item("60m", "60 M, &#8722;0.25, model 812", T["60m"]["nominal"]),
                                     item("60", "60 M, +0.25, model 812", T["60"]["nominal"])],
                        facets=[{**pol(["60m", "60"])[0], "values": [pol(0)[0]["values"][2], pol(0)[0]["values"][0]]}],
                        caption=(f"The two 60 M runs of the recipe, 64 rollouts each: the same roll to step 110, after which the "
                                 f"&#8722;0.25 run stops at cos {e['60m']['final_cos_mean']:.3f} and the +0.25 run continues to "
                                 f"{e['60']['final_cos_mean']:.3f}."))
    fft = IF.eval_figure("eval-ft", [item("ft", "Grip finetune, model 270", T["ft"]["nominal"]),
                                     item("60", "Its parent, 60 M, +0.25, model 812", T["60"]["nominal"])],
                         facets=[{**pol(["ft", "60"])[0], "values": [pol(0)[0]["values"][3], pol(0)[0]["values"][0]]}],
                         caption=(f"The grip finetune against its parent, 64 rollouts each: the finetune crosses cos 0.9 at step "
                                  f"{e['ft']['t_align_mean']:.0f} against {e['60']['t_align_mean']:.0f} and holds the tool at a lower "
                                  f"pad force sum."))
    return f60, fr0, fm, fft


def table(head, rows):
    out = ['<div class="tw"><table><thead><tr>']
    for h in head:
        out.append(f'<th{" class=num" if h.startswith("#") else ""}>{h.lstrip("#")}</th>')
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>")
        for h, c in zip(head, r):
            if isinstance(c, tuple):
                out.append(f'<td class="{c[1]}">{c[0]}</td>')
            else:
                out.append(f'<td{" class=num" if h.startswith("#") else ""}>{c}</td>')
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def cellc(frac, text):
    cls = "c4" if frac >= 0.99 else "c3" if frac >= 0.6 else "c2" if frac >= 0.4 else "c1" if frac > 0 else "c0"
    return (text, f"cell {cls}")


def deg(c):
    return f"{np.degrees(np.arccos(np.clip(c, -1, 1))):.0f}"


def row(label, e, timing):
    n = e["n"]
    held = int(round(e["hold_rate"] * n))
    aligned = int(round(e["align_rate"] * n))
    t_al = "&#8211;" if aligned == 0 else f"{e['t_align_mean']:.0f}"
    forces = f"{e['mean_force_thumb']:.1f} / {e['mean_force_index']:.1f} / {e['mean_force_middle']:.1f}"
    total = e["mean_force_thumb"] + e["mean_force_index"] + e["mean_force_middle"]
    return [label, timing, cellc(held / n, f"{held}/{n}"), cellc(aligned / n, f"{aligned}/{n}"),
            f"{e['final_cos_mean']:+.3f} &#177; {e['final_cos_sd']:.3f}", deg(e["final_cos_mean"]), t_al,
            f"{e['final_z_mean'] * 1000:.0f}", forces, f"{total:.1f}"]


def forces(csv_name, lo=58, hi=250):
    a = np.loadtxt(os.path.join(ROOT, csv_name), delimiter=",", skiprows=1)
    m = (a[:, 0] >= lo) & (a[:, 0] < hi)
    f = a[m, 3:6].mean(axis=0)
    return f, f.sum()


def table_runs(e20, e60, e60m, eft):
    rows = []
    for label, run, w, e, csv in [
        ("20 M, model 270", "20260916-2142-d6_cal_reorient_s0", "off", e20, "20260917-d6_cal_20M_m270_trace.csv"),
        ("60 M, model 812", "20260917-1141-d6_cal_reorient_gp025_60M_s0", "+0.25 (wrong sign)", e60, "20260917-d6_cal_60M_gp025_m812_trace.csv"),
        ("60 M, model 812", "20260917-1653-d6_cal_reorient_gpm025_60M_s0", "&#8722;0.25", e60m, "20260917-d6_cal_60M_gpm025_m812_trace.csv"),
        ("finetune of the +0.25 model 812, 20 M, model 270", "20260917-1844-d6_cal_reorient_ft_gpm5_20M_s0", "&#8722;5", eft, "20260917-d6_cal_ft_gpm5_m270_trace.csv"),
    ]:
        n = e["n"]
        held = int(round(e["hold_rate"] * n)); aligned = int(round(e["align_rate"] * n))
        fa, ta = forces(csv); fe, te = forces(csv, 200)
        rows.append([label, f"<code>{run}</code>", w, cellc(held / n, f"{held}/{n}"), cellc(aligned / n, f"{aligned}/{n}"),
                     f"{e['final_cos_mean']:+.3f} &#177; {e['final_cos_sd']:.3f}", deg(e["final_cos_mean"]),
                     f"{fa[0]:.1f} / {fa[1]:.1f} / {fa[2]:.1f} = {ta:.1f}", f"{fe[0]:.1f} / {fe[1]:.1f} / {fe[2]:.1f} = {te:.1f}"])
    return table(["checkpoint", "run", "grip term weight", "held at end", "cos &#8805; 0.9 reached", "final cos",
                  "#from vertical (deg)", "pad N th / ix / md, steps 58&#8211;249", "pad N, steps 200&#8211;249"], rows)


def main():
    # every evaluation number is the 2026-10-11 re-evaluation; the originals of 2026-09-16/17 stay beside the page
    e60, e60m, eft, e20 = reeval(RUN60, "nominal"), reeval(RUN60M, "nominal"), reeval(RUNFT, "nominal"), reeval(RUN20, "nominal")
    e60_r0, e60_r0s = reeval(RUN60, "nominal_r0"), reeval(RUN60, "nominal_r0_stochastic")
    e20_old = reeval(RUN20, "nominal_r0", TIP_TRACES)
    T = {"60": load_traces(f"{TRACES}/{RUN60}.npz"), "20": load_traces(f"{TRACES}/{RUN20}.npz"),
         "60m": load_traces(f"{TRACES}/{RUN60M}.npz"), "ft": load_traces(f"{TRACES}/{RUNFT}.npz"),
         "20old": load_traces(f"{TIP_TRACES}/{RUN20}.npz")}
    f60, fr0, fm, fft = figures(T, {"60": e60, "60m": e60m, "ft": eft, "20": e20, "60r0": e60_r0, "20r0": e20_old})
    c_end = T["60"]["nominal"]["cos"][-1]
    r0 = T["60"]["nominal_r0"]
    r0_stand = int(((r0["cos"][-1] > 0.9) & (r0["z"][-1] < r0["meta"]["floor_z"])).sum())
    when = sorted({e["evaluated"][:10] for e in (e60, e60m, eft, e20, e60_r0, e60_r0s, e20_old)})
    curves = load("20260917-training_curves.json")
    head = ["checkpoint", "residual active from", "held at end", "cos &#8805; 0.9 reached", "final cos", "#from vertical (deg)",
            "#steps to 0.9", "#tool z (mm)", "pad N thumb / index / middle", "#total N"]
    table_main = table(head, [
        row("20 M, model 270", e20, "step 58 (as trained)"),
        row("60 M, model 812", e60, "step 58 (as trained)"),
    ])
    table_eval = table(head, [
        row("20 M, model 270", e20_old, "step 0 (evaluator default)"),
        row("20 M, model 270", e20, "step 58 (as trained)"),
        row("60 M, model 812", e60_r0, "step 0 (evaluator default)"),
        row("60 M, model 812, sampled actions", e60_r0s, "step 0 (evaluator default)"),
        row("60 M, model 812", e60, "step 58 (as trained)"),
    ])
    al = dict(curves["60M (s0, grip term +0.25)"]["Episode_Reward/target_axis_alignment"])
    al20 = dict(curves["20M (s0)"]["Episode_Reward/target_axis_alignment"])

    def at(d, it):
        ks = sorted(d)
        k = min(ks, key=lambda x: abs(x - it))
        return d[k]
    FT = "finetune of 60M +0.25 model 812, weight -5 (20M)"
    alm = dict(curves["60M (s0, grip term -0.25)"]["Episode_Reward/target_axis_alignment"])
    gm = dict(curves["60M (s0, grip term -0.25)"]["Episode_Reward/grip_force_excess"])
    sub = {
        "TABLE_MAIN": table_main, "TABLE_EVAL": table_eval, "TABLE_RUNS": table_runs(e20, e60, e60m, eft),
        "COS60M": f"{e60m['final_cos_mean']:.3f}", "SD60M": f"{e60m['final_cos_sd']:.3f}", "DEG60M": deg(e60m["final_cos_mean"]),
        "ALM812": f"{at(alm, 812):.1f}", "GM812": f"{at(gm, 812):.2f}",
        "COSFT": f"{eft['final_cos_mean']:.3f}", "SDFT": f"{eft['final_cos_sd']:.3f}", "DEGFT": deg(eft["final_cos_mean"]),
        "TALFT": f"{eft['t_align_mean']:.0f}", "HOLDFT": f"{eft['hold_steps_mean']:.0f}", "ZFT": f"{eft['final_z_mean'] * 1000:.0f}",
        "FFT": f"{eft['mean_force_thumb']:.1f} / {eft['mean_force_index']:.1f} / {eft['mean_force_middle']:.1f}",
        "FT_TOT": f"{forces('20260917-d6_cal_ft_gpm5_m270_trace.csv')[1]:.1f}",
        "FT_TOT_END": f"{forces('20260917-d6_cal_ft_gpm5_m270_trace.csv', 200)[1]:.1f}",
        "FT_THUMB_MAX": f"{np.loadtxt(os.path.join(ROOT, '20260917-d6_cal_ft_gpm5_m270_trace.csv'), delimiter=',', skiprows=1)[:, 3].max():.1f}",
        "GFT10": f"{at(dict(curves[FT]['Episode_Reward/grip_force_excess']), 10):.2f}",
        "GFT270": f"{at(dict(curves[FT]['Episode_Reward/grip_force_excess']), 270):.2f}",
        "AFT270": f"{at(dict(curves[FT]['Episode_Reward/target_axis_alignment']), 270):.1f}",
        "I_STRIP_FT": uri("20260917-d6_cal_ft_gpm5_m270_strip.png"),
        "V_RL_FT": uri("videos/d6_cal_ft_gpm5_m270.mp4"),
        "F_EVAL_FT": fft.html(),
        "I_STRIP_M": uri("20260917-d6_cal_60M_gpm025_m812_strip.png"),
        "V_RL_M": uri("videos/d6_cal_60M_gpm025_m812.mp4"),
        "F_EVAL_M": fm.html(),
        "COS60": f"{e60['final_cos_mean']:.3f}", "SD60": f"{e60['final_cos_sd']:.3f}", "DEG60": deg(e60["final_cos_mean"]),
        "TAL60": f"{e60['t_align_mean']:.0f}", "HOLD60": f"{e60['hold_steps_mean']:.0f}",
        "Z60": f"{e60['final_z_mean'] * 1000:.0f}",
        "F60": f"{e60['mean_force_thumb']:.1f} / {e60['mean_force_index']:.1f} / {e60['mean_force_middle']:.1f}",
        "FT60": f"{e60['mean_force_thumb'] + e60['mean_force_index'] + e60['mean_force_middle']:.1f}",
        "COS20": f"{e20['final_cos_mean']:.3f}", "DEG20": deg(e20["final_cos_mean"]),
        "R0_ALIGNED": f"{int(round(e60_r0['align_rate'] * 64))}", "R0_HELD": f"{int(round(e60_r0['hold_rate'] * 64))}",
        "R0_LOST": f"{e60_r0['n_lost']}", "R0_DROP": f"{e60_r0['drop_step_mean']:.0f} &#177; {e60_r0['drop_step_sd']:.0f}",
        "R0_Z": f"{e60_r0['final_z_mean'] * 1000:.0f}",
        "R0S_HELD": f"{int(round(e60_r0s['hold_rate'] * 64))}", "R0S_ALIGNED": f"{int(round(e60_r0s['align_rate'] * 64))}",
        "AL270": f"{at(al20, 270):.1f}", "AL60_270": f"{at(al, 270):.1f}", "AL405": f"{at(al, 405):.1f}", "AL567": f"{at(al, 567):.1f}",
        "AL648": f"{at(al, 648):.1f}", "AL729": f"{at(al, 729):.1f}", "AL812": f"{at(al, 812):.1f}",
        "AL_LAST160": f"{100 * (at(al, 812) / at(al, 648) - 1):.0f}",
        "I_STRIP": uri("20260917-d6_cal_60M_m812_strip.png"),
        "V_RL": uri("videos/d6_cal_60M_m812.mp4"),
        "F_EVAL": f60.html(),
        "I_CURVES": uri("20260917-training_curves.png"),
        "I_STRIP_R0": uri("20260917-d6_cal_60M_m812_residual0_strip.png"),
        "V_R0": uri("videos/d6_cal_60M_m812_residual0.mp4"),
        "F_EVAL_R0": fr0.html(),
        "REEVAL": f"evaluation numbers and figures from the re-evaluation of {', '.join(when)} (<code>traces/</code>)",
        "BAND60": f"{np.percentile(c_end, 90) - np.percentile(c_end, 10):.3f}",
        "R0_STAND": str(r0_stand), "R0_COS": f"{e60_r0['final_cos_mean']:.2f} &#177; {e60_r0['final_cos_sd']:.2f}",
        "THREEFT": f"{100 * eft['three_finger_share']:.1f}",
    }
    tpl = open(TPL).read()
    for k, v in sub.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    left = [ln for ln in tpl.splitlines() if "{{" in ln]
    assert not left, left[:3]
    page = IF.add_assets(retro_style.apply(tpl))  # plain page style (owner, 2026-10-09); interactive figures (2026-10-11)
    open(OUT, "w").write(page)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB, figure data {IF.data_bytes(page) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
