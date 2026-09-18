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

ROOT = "docs/experiments/20260917-d6_cal_60M"
TPL = "scripts/d6_cal_60m_page.template.html"
OUT = f"{ROOT}/20260917-d6_reorient_policy_60M.html"


def uri(path):
    p = os.path.join(ROOT, path)
    return f"data:{mimetypes.guess_type(p)[0]};base64," + base64.b64encode(open(p, "rb").read()).decode()


def load(name):
    return json.load(open(os.path.join(ROOT, name)))


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


def table_runs(e20, e60, e60m):
    rows = []
    for label, run, w, e, csv in [
        ("20 M, model 270", "20260916-2142-d6_cal_reorient_s0", "off", e20, "20260917-d6_cal_20M_m270_trace.csv"),
        ("60 M, model 812", "20260917-1141-d6_cal_reorient_gp025_60M_s0", "+0.25 (wrong sign)", e60, "20260917-d6_cal_60M_gp025_m812_trace.csv"),
        ("60 M, model 812", "20260917-1653-d6_cal_reorient_gpm025_60M_s0", "&#8722;0.25", e60m, "20260917-d6_cal_60M_gpm025_m812_trace.csv"),
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
    e60 = load("20260917-d6_cal_60M_eval.json")
    e60m = load("20260917-d6_cal_60M_gpm025_eval.json")
    e20 = load("20260917-d6_cal_20M_m270_eval_fixed.json")
    r0 = load("20260917-d6_cal_60M_eval_residual0.json")
    e60_r0 = r0["deterministic"]
    e60_r0s = load(r0["stochastic_file"])
    e20_old = json.load(open("docs/experiments/20260916-tip_gait/d6_cal_rl_eval.json"))
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
    alm = dict(curves["60M (s0, grip term -0.25)"]["Episode_Reward/target_axis_alignment"])
    gm = dict(curves["60M (s0, grip term -0.25)"]["Episode_Reward/grip_force_excess"])
    sub = {
        "TABLE_MAIN": table_main, "TABLE_EVAL": table_eval, "TABLE_RUNS": table_runs(e20, e60, e60m),
        "COS60M": f"{e60m['final_cos_mean']:.3f}", "SD60M": f"{e60m['final_cos_sd']:.3f}", "DEG60M": deg(e60m["final_cos_mean"]),
        "ALM812": f"{at(alm, 812):.1f}", "GM812": f"{at(gm, 812):.2f}",
        "FT_STATUS": os.environ.get("FT_STATUS", "Running at the time of writing; its evaluation is added here when it finishes."),
        "I_STRIP_M": uri("20260917-d6_cal_60M_gpm025_m812_strip.png"),
        "V_RL_M": uri("videos/d6_cal_60M_gpm025_m812.mp4"),
        "I_EVAL_M": uri("20260917-d6_cal_60M_gpm025_eval.png"),
        "COS60": f"{e60['final_cos_mean']:.3f}", "SD60": f"{e60['final_cos_sd']:.3f}", "DEG60": deg(e60["final_cos_mean"]),
        "TAL60": f"{e60['t_align_mean']:.0f}", "HOLD60": f"{e60['hold_steps_mean']:.0f}",
        "Z60": f"{e60['final_z_mean'] * 1000:.0f}",
        "F60": f"{e60['mean_force_thumb']:.1f} / {e60['mean_force_index']:.1f} / {e60['mean_force_middle']:.1f}",
        "FT60": f"{e60['mean_force_thumb'] + e60['mean_force_index'] + e60['mean_force_middle']:.1f}",
        "COS20": f"{e20['final_cos_mean']:.3f}", "DEG20": deg(e20["final_cos_mean"]),
        "COS20OLD": f"{e20_old['final_cos_mean']:.2f}",
        "R0_ALIGNED": f"{int(round(e60_r0['align_rate'] * 64))}", "R0_HELD": f"{int(round(e60_r0['hold_rate'] * 64))}",
        "R0_LOST": f"{e60_r0['n_lost']}", "R0_DROP": f"{e60_r0['drop_step_mean']:.0f} &#177; {e60_r0['drop_step_sd']:.0f}",
        "R0_Z": f"{e60_r0['final_z_mean'] * 1000:.0f}",
        "R0S_HELD": f"{int(round(e60_r0s['hold_rate'] * 64))}", "R0S_ALIGNED": f"{int(round(e60_r0s['align_rate'] * 64))}",
        "AL270": f"{at(al20, 270):.1f}", "AL60_270": f"{at(al, 270):.1f}", "AL405": f"{at(al, 405):.1f}", "AL567": f"{at(al, 567):.1f}",
        "AL648": f"{at(al, 648):.1f}", "AL729": f"{at(al, 729):.1f}", "AL812": f"{at(al, 812):.1f}",
        "AL_LAST160": f"{100 * (at(al, 812) / at(al, 648) - 1):.0f}",
        "I_STRIP": uri("20260917-d6_cal_60M_m812_strip.png"),
        "V_RL": uri("videos/d6_cal_60M_m812.mp4"),
        "I_EVAL": uri("20260917-d6_cal_60M_eval.png"),
        "I_CURVES": uri("20260917-training_curves.png"),
        "I_STRIP_R0": uri("20260917-d6_cal_60M_m812_residual0_strip.png"),
        "V_R0": uri("videos/d6_cal_60M_m812_residual0.mp4"),
        "I_EVAL20": uri("20260917-d6_cal_20M_m270_eval_fixed.png"),
    }
    tpl = open(TPL).read()
    for k, v in sub.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    left = [ln for ln in tpl.splitlines() if "{{" in ln]
    assert not left, left[:3]
    open(OUT, "w").write(tpl)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
