#!/usr/bin/env python3
r"""Build docs/experiments/20261011-diffmjx_audit/20261011-diffmjx_gradient_probe_audit.html.

    .venv/bin/python scripts/diffmjx_audit_page.py            # the page; fails while a section's data is missing
    .venv/bin/python scripts/diffmjx_audit_page.py --draft    # local preview with placeholders for missing data

The audit of the DiffMJX and contact-gait studies of 2026-10-07. Every number in the prose is computed here from the
studies' own JSON (docs/experiments/20261007-diffmjx/, docs/experiments/20261007-contact-gait/) and from the audit's
JSON in docs/experiments/20261011-diffmjx_audit/, written by scripts/diffmjx_audit_probes.py (CPU checks and the
corrected searches) and scripts/diffmjx_turn_onset.py (the fork at the turn onset). Mathematics is LaTeX (\( \)
inline, \[ \] display) typeset to inline SVG by scripts/texsvg.py, cached beside the page.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D07 = os.path.join(ROOT, "docs/experiments/20261007-diffmjx")
G07 = os.path.join(ROOT, "docs/experiments/20261007-contact-gait")
A = os.path.join(ROOT, "docs/experiments/20261011-diffmjx_audit")
REL = "docs/experiments/20261011-diffmjx_audit"
OUT = os.path.join(A, "20261011-diffmjx_gradient_probe_audit.html")
TPL = os.path.join(ROOT, "scripts/diffmjx_audit_page.template.html")
TEX_CACHE = os.path.join(A, "texsvg_cache.json")
FORK = "martius-lab/mujoco@ced2cff"          # the fork's pinned commit (docs/experiments/20261007-diffmjx/README.md)

EXTRA_CSS = """
table.gloss th{text-align:left;vertical-align:top;white-space:nowrap;padding-right:12px}
table.gloss td, td.txt{white-space:normal;text-align:left}
figure.diagram svg{max-width:100%;height:auto}
""" + texsvg.BLOCK_CSS + "\n"

FIG, TAB = [0], [0]
DRAFT = [False]


# ------------------------------------------------------------------------------------------ helpers

def esc(s):
    return html.escape(str(s), quote=True)


def im(tex):
    return "\\(" + tex + "\\)"


def eq(tex, n):
    return f'<div class="eq">\\[{tex}\\]<span class="eqno">({n})</span></div>'


def code(s):
    return f"<code>{esc(s)}</code>"


def js(path):
    with open(path) as fh:
        return json.load(fh)


def opt(path):
    """JSON of a later step, or None in a draft build; a missing file fails the page build."""
    if os.path.exists(path):
        return js(path)
    if DRAFT[0]:
        return None
    raise SystemExit(f"missing {os.path.relpath(path, ROOT)}; run its step or build with --draft")


def pending(step):
    return f"<p><i>Written at step {step} of the audit.</i></p>"


def g(x, n=3):
    """n significant figures, plain decimal for moderate magnitudes, a times-ten form otherwise."""
    if x is None or not math.isfinite(float(x)):
        return "&#8211;"
    x = float(x)
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    if -3 <= e < 5:
        s = f"{x:.{max(0, n - 1 - e)}f}"
        return s.replace("-", "&#8722;")
    m = x / 10 ** e
    return f"{m:.{n - 1}f}&#215;10<sup>{e}</sup>".replace("-", "&#8722;")


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}&#8202;%"


def grp(x):
    return f"{x:,.0f}"


def table(head, rows, cls="", text=False):
    out = [f"<div class='tw'><table{' class=' + repr(cls) if cls else ''}><thead><tr>"
           + "".join(f"<th{' class=num' if i and not text else ''}>{h}</th>" for i, h in enumerate(head))
           + "</tr></thead><tbody>"]
    for r in rows:
        if isinstance(r, str):
            out.append(f"<tr class='grp'><td colspan='{len(head)}'>{r}</td></tr>")
            continue
        cells = []
        for i, c in enumerate(r):
            if isinstance(c, tuple):
                cells.append(f"<td class='{c[1]}'>{c[0]}</td>")
            else:
                cells.append(f"<td{' class=num' if i else ''}>{c}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(out) + "</tbody></table></div>"


def tcap(text):
    TAB[0] += 1
    return f"<p class='tnote'><b>Table&#160;{TAB[0]}.</b> {text}</p>"


def figure(svg, caption):
    FIG[0] += 1
    return f'<figure class="diagram">{svg}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def render_tex(t):
    maths = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(maths, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    return P.TEX_RE.sub(lambda m: next(svgs), t), len(maths)


def tilt_deg(c0, c1):
    return math.degrees(math.acos(c0) - math.acos(c1))


def rel(a, b):
    return abs(a - b) / abs(b)


# ------------------------------------------------------------------------------------------ data

def load():
    S = {}
    S["ls"] = js(os.path.join(D07, "20261007-pad_local_search_20_mj331.json"))
    S["jac"] = js(os.path.join(D07, "20261007-official_padsT1_cylinder_jacobian_20.json"))
    S["cfd"] = js(os.path.join(D07, "20261007-official_padsT1_cylinder_cfd_jacobian_20.json"))
    S["sweep"] = js(os.path.join(D07, "20261007-official_padsT1_cylinder_fd_sweep_20.json"))
    S["jvp1"] = js(os.path.join(D07, "20261007-official_padsT1_cylinder_jvp_1.json"))
    S["fd1"] = js(os.path.join(D07, "20261007-official_padsT1_cylinder_fd_1.json"))
    S["jvp2"] = js(os.path.join(D07, "20261007-official_padsT2_cylinder_jvp_1.json"))
    S["pad1"] = js(os.path.join(D07, "20261007-official_padsT1_cylinder_1.json"))
    S["pad2"] = js(os.path.join(D07, "20261007-official_padsT2_cylinder_1.json"))
    S["sph"] = js(os.path.join(D07, "20261007-official_cylinder_plain_1.json"))
    S["mask"] = js(os.path.join(D07, "20261007-pad_mask_cpu.json"))
    S["rj"] = js(os.path.join(G07, "20261007-fork-padsT2-recontact-jvp.json"))
    S["rc"] = js(os.path.join(G07, "20261007-fork-padsT2-recontact-cfd-jvp.json"))
    S["attempt"] = js(os.path.join(G07, "20261007-mjx-exact1mm-attempt.json"))
    S["gpu2"] = js(os.path.join(G07, "20261007-mjx-gpu-padsT2-20-warm.json"))
    for k, f in (("pads", "20261007-padsT-seed1-256.json"), ("pads2", "20261007-padsT2-seed1-256.json"),
                 ("mesh", "20261007-mesh-seed1-256.json"), ("sphere", "20261007-sphere-seed1-256.json"),
                 ("pads_rel", "20261007-padsT-release-seed1-256.json"),
                 ("mesh_rel", "20261007-mesh-release-seed1-256.json"),
                 ("sphere_rel", "20261007-sphere-release-seed1-256.json")):
        S[k] = js(os.path.join(G07, f))
    S["cpu360"] = js(os.path.join(G07, "20261007-transfer-mj360.json"))
    S["warp"] = [js(os.path.join(G07, f)) for f in ("20261007-warp-padsT-free-8.json",
                                                    "20261007-warp-padsT-release-8.json",
                                                    "20261007-warp-padsT-free-seed-8.json",
                                                    "20261007-warp-padsT-release-seed-8.json")]
    S["interfinger"] = js(os.path.join(A, "interfinger.json"))
    S["fdstep"] = js(os.path.join(A, "fdstep.json"))
    S["reach"] = js(os.path.join(A, "cfd_reach.json"))
    S["frames"] = js(os.path.join(A, "frames.json"))
    S["contacts20"] = js(os.path.join(A, "contacts20.json"))
    S["corr"] = opt(os.path.join(A, "corrections.json"))
    S["module"] = opt(os.path.join(A, "module_check.json"))
    S["onset"] = opt(os.path.join(A, "turn_onset.json"))
    return S


def warp_agreement(S):
    """Largest |Warp - CPU| orientation gain over the held packed-pad rollouts of the 2026-10-07 replays."""
    label = {False: "tpu6_padsT", True: "tpu6_padsT_release"}
    worst, n = 0.0, 0
    for w, release in zip(S["warp"], (False, True, False, True)):
        for row in w["rows"]:
            if row["plan"] == "hold" or not row["held_final"]:
                continue
            cpu = [r for r in S["cpu360"]["rows"] if r["destination"] == "tpu6_padsT" and r["seed"] == row["seed"]
                   and r["plan"] == f"{label[release]}_{row['plan']}"]
            if cpu:
                worst, n = max(worst, abs(cpu[0]["cos_gain"] - row["cos_gain"])), n + 1
    return worst, n


# ------------------------------------------------------------------------------------------ sections

def glossary():
    G = [
        ("MJX, native MJX", "MuJoCo&#8217;s JAX simulation backend (MuJoCo XLA). Native MJX is the release in the "
         "project environment, MuJoCo 3.6."),
        ("DiffMJX fork", f"the fork of MJX at MuJoCo 3.3.1 by Paulus et al. ({code(FORK)}), which adds contact from "
         "distance, a scan-loop solver and smooth collision; it runs in the isolated Python 3.13 environment of "
         f"{code('docs/experiments/20261007-diffmjx/README.md')} with JAX 0.11.2 in 64-bit arithmetic."),
        ("AD, JVP", "automatic differentiation through the JAX simulation. A Jacobian-vector product (JVP) is one "
         "forward-mode directional derivative; a gradient over " + im("n") + " controls takes " + im("n") + " JVPs."),
        ("FD", "finite difference. Central: " + im(r"\big(f(u+h e_i)-f(u-h e_i)\big)/2h") + "; two-point: "
         + im(r"\big(f(u+h e_i)-f(u)\big)/h") + ", with step " + im("h") + " in the control&#8217;s units."),
        ("Vertical cosine " + im("c"), "The cosine between the shaft (the tool&#8217;s body " + im("z") + " axis) and "
         "world vertical, " + im(r"c = 1-2(q_x^2+q_y^2)") + " for the orientation quaternion "
         + im(r"(q_w,q_x,q_y,q_z)") + ". 1 is upright; the held state has 0.112, 83.6&#176; from vertical. Orientation "
         "gain " + im(r"\Delta c") + " is the final minus the held value."),
        ("Body-x angular speed, body-x torque", "the components about the tool&#8217;s body " + im("x") + " axis of "
         "the free joint&#8217;s angular velocity (rad/s) and of its constraint force (N&#8201;m). MuJoCo expresses "
         "a free joint&#8217;s rotational coordinates in the body frame; the pages of 7 October call these "
         "quantities angular speed and contact torque about world " + im("x") + "."),
        ("Contact gap", "smallest signed distance between a finger&#8217;s spheres and the tool, in &#181;m or mm; "
         "negative is penetration."),
        ("Held state", "the deployed D1 grip pose held 0.8&#8201;s with the tool on its 100&#8201;mm post, then the "
         "post removed; tool placement jitter seed 1."),
        ("Packed pads", "spheres of radius 0.75&#8201;mm on the printed TPU fingertip block at 1&#8201;mm (796 per "
         "finger) or 2&#8201;mm (227 per finger) spacing. In these scenes they touch only the tool and the block is "
         "visual."),
        ("Plan", "the 13 values of the 400&#8201;ms continuation: yaw-target offsets of the three fingers in the "
         "phases 0&#8211;120, 120&#8211;240 and 240&#8211;400&#8201;ms (0.35&#8201;rad per unit), index and middle "
         "opening amplitudes until 180&#8201;ms, and their closing offsets afterwards (0.30&#8201;rad per unit)."),
        ("Score " + im("S") + ", held", "equation (1). Held: tool descent at most 8&#8201;mm and every finger&#8217;s "
         "final normal force at least 0.5&#8201;N."),
        ("CEM, CEM128", "cross-entropy method: 64 plans per generation drawn from a Gaussian refitted to the best 8, "
         "256 evaluations; CEM128 is its first 128 evaluations."),
        ("L-BFGS-B, hybrid", "SciPy&#8217;s quasi-Newton method with bounds, its gradient from two-point differences; "
         "the hybrid runs 128 evaluations of L-BFGS-B from the best CEM128 plan."),
        ("Contact from distance (CFD)", "the fork&#8217;s backward rule: a second constraint solve in which every "
         "candidate pair within the width " + im(r"w = \texttt{cfd\_solimp}[2]") + " of touching is active, with the "
         "impedance " + im(r"\texttt{cfd\_solimp}") + "; its derivative replaces that of the ordinary solve, whose "
         "forward values are kept."),
        ("Scan loop", "the fork&#8217;s fixed-length solver loop: all " + im(r"\texttt{iterations}") + " Newton steps, "
         "each with up to " + im(r"\texttt{ls\_iterations}") + " line-search steps, evaluated every time so that "
         "reverse mode can pass through."),
        ("Cost ratio " + im(r"\rho"), im(r"\rho = (t_c + t_{AD})/t_{FD}") + ": wall time to compile and evaluate one AD "
         "derivative over the wall time of the CPU finite-difference evaluations of the same derivative; "
         + im(r"\rho_{\mathrm{warm}} = t_{AD}/t_{FD}") + " leaves out compilation."),
        ("Forward parity", "agreement between the fork&#8217;s and CPU MuJoCo&#8217;s trajectories from the same "
         "state under the same controls."),
    ]
    rows = "".join(f"<tr><th>{t}</th><td>{d}</td></tr>" for t, d in G)
    return f"<div class='tw'><table class='gloss'><tbody>{rows}</tbody></table></div>" + eq(
        r"S = \Delta c - 70\,\max\!\big(0,\; z_0 - z_{\min} - 0.003\big) - 0.7 \sum_{f} \max\!\big(0,\; 0.5 - F_f\big)",
        1) + ("<p>with the tool&#8217;s held height " + im("z_0") + " and lowest height " + im(r"z_{\min}")
              + " in metres and each finger&#8217;s final normal force " + im("F_f") + " in newtons, summed over the "
              "finger&#8217;s contacts with the tool.</p>")


def scripts_section(S):
    wa, wn = warp_agreement(S)
    I = S["interfinger"]
    rows = [
        (code("real_v1_diffmjx_gate.py"),
         "the continuation scene (palm plate off, fingertip block visual for packed pads, capsule and no-post "
         "variants), the held state, native MJX 3.6 rollouts against CPU MuJoCo, an optional native gradient",
         "The parity numbers of the contact-gradients page hold. Its third and fourth outputs are about the "
         f"tool&#8217;s body " + im("x") + " axis (lines 280&#8211;282, 290&#8211;292); both pages say world "
         + im("x") + "."),
        (code("real_v1_diffmjx_official_gate.py"),
         "the same parity check in the fork, compilation and warm timing, a Jacobian or one JVP with fork and CPU "
         "differences at 0.01&#8201;rad, the CFD, scan-loop and smooth-collision switches",
         "CFD width 0.05&#8201;m (line 75) puts every pad in the soft solve; CFD always runs with the scan loop, so "
         "the two switches are never separated."),
        (code("real_v1_diffmjx_pad_cpu.py"),
         "masked against full pad collision over 20&#8201;ms of the held grasp",
         "Holds (difference 0). The mask also leaves the fingertip blocks without collision; the blocks stay "
         f"{I['overall_min_block_block_mm']:.1f}&#8201;mm apart and {I['overall_min_block_link_mm']:.1f}&#8201;mm from "
         "the other fingers&#8217; links or more in all ten 400&#8201;ms plans, so it has no effect here."),
        (code("real_v1_diffmjx_pad_jvp.py"),
         "fork JVPs and three-yaw Jacobians on the packed pads, fork and CPU step sweeps, the 0.1&#8201;rad action "
         "along the AD gradient scored in CPU MuJoCo",
         f"Correct. Its timing key {code('cpu_fd_3_evals_s')} holds 12 evaluations with {code('--all-directions')} "
         "(lines 133&#8211;158, 185); the pages do not use those two values."),
        (code("real_v1_diffmjx_local_search.py"),
         "the CPU difference direction and 128 random 0.1&#8201;rad directions over 20&#8201;ms",
         "Correct; the 20&#8201;ms response is linear in the three controls, so every direction estimate scores "
         "alike (<a href='#horizon'>Horizon and action of the 20&#8201;ms test</a>)."),
        (code("real_v1_diffmjx_recontact_jvp.py"),
         "a CPU replay to 280&#8201;ms, then a fork JVP over 10&#8201;ms with respect to the index MCP target and CPU "
         "differences at three steps",
         "The plan is the 1&#8201;mm search&#8217;s CEM release plan replayed on 2&#8201;mm pads (lines "
         "56&#8211;57); the contact-release page calls it the sampled 2&#8201;mm pad plan."),
        (code("real_v1_diffmjx_page.py"), "the contact-gradients page",
         "The lede sets the warm gradient, 35.7&#8201;s, against the CPU baseline and leaves out compilation (line "
         "242); the column &#8220;AD error&#8221; (line 308) is the difference between the fork and CPU MuJoCo."),
        (code("real_v1_contact_gait_search.py"),
         "the 400&#8201;ms plan search: L-BFGS-B, CEM and the hybrid, 256 evaluations each",
         "The two-point step is SciPy&#8217;s default, 1.49&#215;10<sup>&#8722;8</sup> (lines 158&#8211;161), "
         "which measures collision noise on the convex mesh; all searches share the CEM seed 20261007 (lines "
         "215&#8211;216)."),
        (code("real_v1_contact_gait_transfer.py"),
         "replays of the fixed plans over eight grip jitters and four fingertip models",
         "Correct; jitter 1 is the search&#8217;s own grasp, so seven of eight are held out."),
        (code("real_v1_contact_gait_warp.py"), "MuJoCo-Warp replays of the plans in 32 worlds",
         f"Reproduces the CPU orientation gain within {g(wa, 2)} in all {wn} held rollouts."),
        (code("real_v1_contact_gait_page.py"), "the contact-release page",
         "The recontact paragraph&#8217;s plan label (line 332); the convex-mesh rows &#8220;Local derivative&#8221; "
         "and &#8220;CEM + derivative&#8221; of its Table 1 measure finite-difference noise."),
        (code("optimization/phase1_strategy_diffmjx.py"),
         "April&#8217;s gradient ascent through native MJX on a sigmoid distance reward for a cube grasp",
         "Neither the authors&#8217; DiffMJX nor part of the 7 October work; the contact-gradients page says so."),
        (code("backends/factory.py") + " (" + code("diffmjx") + ")",
         "Returns " + code("DiffMJXLiteBackend") + ", the native MJX backend with two metadata flags",
         "The CFD and smooth-collision flags do nothing (" + code("backends/diffmjx.py") + " lines 11&#8211;19); "
         "selecting &#8220;diffmjx&#8221; runs native MJX."),
    ]
    head = ["Script", "What it computes", "Finding"]
    body = table(head, [((r[0], "txt"), (r[1], "txt"), (r[2], "txt")) for r in rows], text=True)
    return ("<p>Table&#160;1 lists each script of the two studies with what it computes and what the audit found; "
            "line numbers refer to commit 10eb3f1e, before the corrections.</p>" + body
            + tcap("Scripts of the two studies of 7 October. Paths under <code>scripts/</code> and "
                   "<code>src/morphohand/</code>."))


def horizon_section(S):
    ls, jac = S["ls"], S["jac"]
    c0 = ls["candidates"]["zero"]["vertical_cos"]
    fd = ls["candidates"]["cpu_fd"]["vertical_cos"] - c0
    ad = jac["local_search"]["cpu_score_ad_direction"] - c0
    best = ls["random_best"]["vertical_cos"] - c0
    pads = S["pads"]
    gain400 = pads["candidates"]["direct_gradient"]["cos_gain"]
    turn400 = tilt_deg(pads["initial_vertical_cos"], pads["initial_vertical_cos"] + gain400)
    base_diff = abs(jac["fork_base_metrics"][0] - jac["cpu_base_metrics"][0])
    c20 = S["contacts20"]["rows"]
    cpu_sw = [v[0] for v in S["sweep"]["cpu_fd_sweep"].values()]
    return (f"<p>The 20&#8201;ms test adds one constant offset to the three yaw targets, runs 20 steps from the held "
            f"state and scores the vertical cosine. The best 0.1&#8201;rad offset raises it by {g(fd, 3)}, a tilt of "
            f"{tilt_deg(c0, c0 + fd):.3f}&#176;. The 400&#8201;ms local search turned the tool by {turn400:.1f}&#176; "
            f"and a reorientation to vertical takes {math.degrees(math.acos(c0)):.1f}&#176;. In these 20 steps the "
            f"tool shifts after the post is removed: {c20['no action']['opened']} of the {c20['no action']['at_start']} "
            f"pad contacts open and {c20['no action']['closed']} closes without an action, "
            f"{c20['CPU finite-difference action']['opened']} and {c20['CPU finite-difference action']['closed']} with the "
            "best offset (<code>contacts20.json</code>).</p>"
            f"<p>The response to the offsets is linear over these 20 steps (the CPU derivative changes by "
            f"{pct(max(cpu_sw) / min(cpu_sw) - 1, 1)} between steps of 10<sup>&#8722;4</sup> and "
            f"10<sup>&#8722;2</sup>&#8201;rad), so every estimate of its gradient selects "
            f"the same action. The CPU finite-difference direction gained {g(fd, 4)}, the fork&#8217;s AD gradient "
            f"{g(ad, 4)} and the best of 128 random directions {g(best, 4)} ({pct(best / fd)} of the CPU direction). "
            "The test measures whether the fork reproduces CPU MuJoCo&#8217;s local response, and every method "
            f"passes it. Without an action the fork&#8217;s vertical cosine after the same 20&#8201;ms differs from "
            f"CPU MuJoCo&#8217;s by {g(base_diff, 2)}, {pct(base_diff / fd, 0)} of the action&#8217;s effect.</p>")


def cost_rows(S):
    """(label, first-call s, warm s, CPU FD s, CPU evaluations, n controls) of each 2026-10-07 derivative."""
    j1, jac, cfd, ls = S["jvp1"], S["jac"], S["cfd"], S["ls"]
    corr = S["corr"]
    rows = [("1&#8201;ms, index yaw, 1&#8201;mm pads, CPU", j1["gradient_compile_s"], j1["gradient_warm_s"],
             j1["cpu_fd_3_evals_s"], 3, 1),
            ("20&#8201;ms, three yaws, 1&#8201;mm pads, CPU", jac["gradient_compile_s"], jac["gradient_warm_s"],
             ls["cpu_fd_6_evals_s"], 6, 3),
            ("20&#8201;ms, three yaws, 1&#8201;mm pads, CPU, CFD and scan loop", cfd["gradient_compile_s"],
             cfd["gradient_warm_s"], ls["cpu_fd_6_evals_s"], 6, 3)]
    if corr and "recontact_cpu_fd_s" in corr:
        rj = S["rj"]
        rows.append(("10&#8201;ms, index MCP, 2&#8201;mm pads, GPU", rj["derivative_compile_s"],
                     rj["derivative_warm_s"], corr["recontact_cpu_fd_s"], 2, 1))
    return rows


def cost_section(S):
    rows = cost_rows(S)
    # first = compilation and the first JVP; warm = all n JVPs. A first gradient is first + (n-1)/n of warm.
    first = [(r[1] + r[2] * (r[5] - 1) / r[5]) / r[3] for r in rows]
    warm = [r[2] / r[3] for r in rows]
    out = [[r[0], g(r[1], 3), g(r[2], 3), f"{g(1e3 * r[3], 3)} ({r[4]})", grp(f), grp(w)]
           for r, f, w in zip(rows, first, warm)]
    gpu, att = S["gpu2"], S["attempt"]
    text = ("<p>The lede of the contact-gradients page set a warm three-control gradient, 35.7&#8201;s, against "
            "six CPU evaluations, 23.5&#8201;ms, and left out the 83.1&#8201;s of its first call. The AD runs used "
            "JAX on the workstation&#8217;s CPU with up to four cores (the resource guard&#8217;s cap), the CPU "
            "differences MuJoCo on one core. A planner pays compilation once per horizon and scene and the warm "
            f"cost at every iteration. A first gradient cost {grp(min(first))} to {grp(max(first))} times its CPU "
            f"finite differences, a warm one {grp(min(warm))} to {grp(max(warm))} times (Table&#160;2).</p>")
    text += table(["Derivative", "First call s", "Warm s", "CPU FD ms (evaluations)", "&#961;", "&#961; warm"], out)
    text += tcap("Cost of one AD derivative in the fork against CPU finite differences of the same derivative. "
                 "First call: compilation and one JVP; warm: all JVPs of the derivative. "
                 + im(r"\rho") + " counts compilation and every JVP. JSON: <code>20261007-official_padsT1_cylinder_"
                 "jvp_1.json</code>, <code>_jacobian_20.json</code>, <code>_cfd_jacobian_20.json</code>, "
                 "<code>20261007-pad_local_search_20_mj331.json</code>, <code>20261007-fork-padsT2-recontact-jvp.json"
                 "</code>.")
    text += (f"<p>The GPU does not change the comparison for one world. On the RTX 4070 Ti SUPER the fork&#8217;s "
             f"20&#8201;ms forward rollout of the 2&#8201;mm pads, without a derivative, took "
             f"{g(gpu['rollout_warm_median_s'], 3)}&#8201;s warm against {g(1e3 * gpu['cpu_rollout_s'], 3)}&#8201;ms "
             f"in CPU MuJoCo, and the 1&#8201;mm model had not finished compiling that rollout after "
             f"{att['elapsed_before_stop_s'] / 60:.1f}&#8201;min at {att['peak_resident_kib'] / 1048576:.1f}&#8201;GiB."
             "</p>")
    return text


def fd_figure(S):
    F = S["fdstep"]
    steps = F["steps"]
    series = [("1 mm packed", "var(--s1)", "circle"), ("convex mesh", "var(--s2)", "square"),
              ("single sphere", "var(--ink3)", "triangle")]
    w, h, left, right, top, bottom = 760, 380, 78, 150, 30, 58
    xs = [math.log10(s) for s in steps]
    x0, x1 = -8.3, -1.7
    y0, y1 = -1.25, 1.5       # log10 of the gradient norm
    X = lambda v: left + (math.log10(v) - x0) / (x1 - x0) * (w - left - right)  # noqa: E731
    Y = lambda v: top + (1 - (math.log10(v) - y0) / (y1 - y0)) * (h - top - bottom)  # noqa: E731
    p = [f'<svg viewBox="0 0 {w} {h}" width="{w}" role="img" aria-label="Finite-difference gradient norm against step">']
    for e in range(-8, -1):
        xx = X(10 ** e)
        p.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{h - bottom}" stroke="var(--rule2)" stroke-width="1"/>')
        p.append(f'<text x="{xx:.1f}" y="{h - bottom + 20}" text-anchor="middle" font-size="15">10<tspan dy="-6" font-size="11">{e}</tspan></text>')
    for v in (0.1, 1, 10):
        yy = Y(v)
        p.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{w - right}" y2="{yy:.1f}" stroke="var(--rule2)" stroke-width="1"/>')
        p.append(f'<text x="{left - 8}" y="{yy + 5:.1f}" text-anchor="end" font-size="15">{v:g}</text>')
    ends = []
    xd = X(1.49e-8)
    p.append(f'<line x1="{xd:.1f}" y1="{top}" x2="{xd:.1f}" y2="{h - bottom}" stroke="#000" stroke-width="1" stroke-dasharray="4 3"/>')
    p.append(f'<text x="{xd + 6:.1f}" y="{top + 14}" font-size="14">SciPy default step</text>')
    for name, col, shape in series:
        norms = F["tips"][name]["plans"]["cem128"]["norm"]
        pts = [(X(s), Y(max(n, 0.1))) for s, n in zip(steps, norms)]
        p.append('<polyline fill="none" stroke="' + col + '" stroke-width="2" stroke-linejoin="round" points="'
                 + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
        for (a, b), s, n in zip(pts, steps, norms):
            tip = f"{name}: step {s:g}, gradient norm {n:.3g}"
            if shape == "circle":
                mark = f'<circle cx="{a:.1f}" cy="{b:.1f}" r="5" fill="{col}" stroke="#fff" stroke-width="2"/>'
            elif shape == "square":
                mark = f'<rect x="{a - 5:.1f}" y="{b - 5:.1f}" width="10" height="10" fill="{col}" stroke="#fff" stroke-width="2"/>'
            else:
                mark = (f'<polygon points="{a:.1f},{b - 6:.1f} {a + 6:.1f},{b + 5:.1f} {a - 6:.1f},{b + 5:.1f}" '
                        f'fill="{col}" stroke="#fff" stroke-width="2"/>')
            p.append(f'<g><title>{esc(tip)}</title>{mark}</g>')
        ends.append((pts[-1][1], pts[-1][0], name))
    last = -1e9
    for b, a, name in sorted(ends):               # end labels at least 18 px apart
        b = max(b, last + 18)
        p.append(f'<text x="{a + 12:.1f}" y="{b + 5:.1f}" font-size="15">{esc(name)}</text>')
        last = b
    p.append(f'<text x="{(left + w - right) / 2:.1f}" y="{h - 12}" text-anchor="middle" font-size="15">Step in plan units</text>')
    p.append(f'<text x="18" y="{(top + h - bottom) / 2:.1f}" text-anchor="middle" font-size="15" '
             f'transform="rotate(-90 18 {(top + h - bottom) / 2:.1f})">Gradient norm of the score</text>')
    p.append("</svg>")
    return "".join(p)


def fd_section(S):
    sw, F = S["sweep"], S["fdstep"]
    fork = [v[0] for v in sw["fork_fd_sweep"].values()]
    cpu = [v[0] for v in sw["cpu_fd_sweep"].values()]
    tf = [v[3] for v in sw["fork_fd_sweep"].values()]
    pads, mesh, sph = (F["tips"][k]["plans"] for k in ("1 mm packed", "convex mesh", "single sphere"))
    msrc, mrel = S["mesh"], S["mesh_rel"]
    step = F["tips"]["1 mm packed"]["probe_step_per_coordinate"][0]
    ccd = F["tips"]["convex mesh"]["ccd_tolerance"]
    text = (f"<p>The 20&#8201;ms differences use 0.01&#8201;rad. Between steps of 10<sup>&#8722;4</sup> and "
            f"10<sup>&#8722;2</sup>&#8201;rad the fork&#8217;s derivative of the vertical cosine changes by "
            f"{pct(max(fork) / min(fork) - 1, 0)}, against {pct(max(cpu) / min(cpu) - 1, 1)} in CPU MuJoCo, and its "
            f"torque derivative by a factor of {max(tf) / min(tf):.1f}: the fork&#8217;s response bends between "
            "10<sup>&#8722;3</sup> and 3&#215;10<sup>&#8722;3</sup>&#8201;rad.</p>")
    text += (f"<p>The 400&#8201;ms search takes L-BFGS-B&#8217;s gradient from SciPy&#8217;s two-point differences at "
             f"the default step, {g(step, 3)} in plan units ({g(step * 0.35, 2)}&#8201;rad of yaw target). On the "
             f"packed pads this step gives the gradient of the larger steps. The cosine to the 10<sup>&#8722;3</sup> gradient is "
             f"{pads['hold']['cosine_to_1e-3'][0]:.3f} at the hold plan and "
             f"{pads['cem128']['cosine_to_1e-3'][0]:.2f} at the CEM128 plan that the hybrid refines, where a "
             f"10<sup>&#8722;4</sup> step crosses a small jump in the score. On the convex mesh the default step measures "
             f"collision-solver noise, and the score is rough at every step below 10<sup>&#8722;2</sup>. At the CEM128 "
             f"plan the gradient&#8217;s norm is {mesh['cem128']['norm'][0]:.1f} at the default step, "
             f"{mesh['cem128']['norm'][2]:.1f} at 10<sup>&#8722;4</sup>, {mesh['cem128']['norm'][3]:.2f} at "
             f"10<sup>&#8722;3</sup> and {mesh['cem128']['norm'][-1]:.2f} at 10<sup>&#8722;2</sup> (Figure&#160;1). "
             f"MuJoCo&#8217;s convex collision, which the mesh uses, converges to {g(ccd, 1)} "
             f"({code('ccd_tolerance')}), and its contact point moves between the mesh&#8217;s faces as the "
             "fingers move. The mesh hybrid ended where it started, "
             f"{msrc['methods']['hybrid_gradient']['score']:.4f} after "
             f"{msrc['methods']['hybrid_gradient']['evals']} evaluations, and so did the mesh release hybrid "
             f"({g(mrel['methods']['hybrid_gradient']['score'], 4)}); the convex-mesh rows of Table&#160;1 of the "
             "contact-release page for the local derivative and the hybrid record this noise. The single "
             f"sphere&#8217;s small-step gradient is consistent (norm {sph['cem128']['norm'][0]:.3f} at both small "
             "steps) but changes with the step above 10<sup>&#8722;4</sup>. <a href='#bugs'>Corrections</a> gives "
             "the searches rerun with a 10<sup>&#8722;2</sup> step.</p>")
    text += figure(fd_figure(S),
                   "Norm of the finite-difference gradient of the score (1) at the CEM128 plan of each fingertip "
                   "model&#8217;s search, against the step, central differences in CPU MuJoCo 3.6. A noise-free "
                   "score gives the same norm at every step. JSON: <code>fdstep.json</code>.")
    text += ("<p>Both penalties of the score (1) are zero for every held plan, so the score ranks held plans by their "
             "orientation gain. Every search used the same CEM sampling seed, 20261007; the "
             f"{code('--seed')} option selects the grip jitter. The CEM128 start of each hybrid is the first half "
             "of the same search&#8217;s CEM run. Each comparison between CEM, local search and hybrid is therefore "
             "one optimizer draw.</p>")
    corr = S["corr"]
    if corr and "cem_seeds" in corr:
        text += corr_seed_paragraph(corr)
    return text


def corr_seed_paragraph(corr):
    rows = corr["cem_seeds"]
    span = lambda r, k: f"{min(x[k] for x in r):.3f}&#8211;{max(x[k] for x in r):.3f}"  # noqa: E731
    wins = {tip: sum(x["hybrid_gain"] > x["cem_gain"] for x in r) for tip, r in rows.items()}
    P_, M_, S_ = rows["1 mm packed"], rows["convex mesh"], rows["single sphere"]
    n = len(P_)
    return (f"<p>Over {['', 'one', 'two', 'three', 'four', 'five', 'six'][n]} CEM seeds with the corrected step, the "
            f"gains from the seed-1 grasp span {span(P_, 'cem_gain')} for CEM and {span(P_, 'hybrid_gain')} for the "
            f"hybrid on the packed pads, {span(M_, 'cem_gain')} and {span(M_, 'hybrid_gain')} on the convex mesh, and "
            f"{span(S_, 'cem_gain')} and {span(S_, 'hybrid_gain')} on the single sphere. The hybrid beats CEM for "
            f"{wins['1 mm packed']}, {wins['convex mesh']} and {wins['single sphere']} of the {n} seeds. The local "
            f"search from the hold plan draws no sample and gains {P_[0]['direct_gain']:.3f}, {M_[0]['direct_gain']:.3f} "
            f"and {S_[0]['direct_gain']:.3f} (<code>corrections.json</code>).</p>")


def cfd_section(S):
    R = S["reach"]
    r1, r2 = R["rows"]["1 mm"], R["rows"]["2 mm"]
    rj, rc = S["rj"], S["rc"]
    cpu0 = rj["cpu_fd"]["0.01"]["derivative"]
    return (f"<p>The fork&#8217;s CFD runs a second, soft constraint solve in which every candidate pair within the "
            f"width " + im("w") + " of touching is active, and passes that solve&#8217;s derivative through the "
            f"ordinary solve&#8217;s forward values ({code('forward.py')} lines 432&#8211;439, {code('constraint.py')} "
            f"lines 570&#8211;571 of {code(FORK)}). The fork&#8217;s README gives its purpose as a derivative toward "
            "contact for bodies that do not yet touch. The probes of 7 October set " + im("w = 0.05") + "&#8201;m, "
            "the value in the fork&#8217;s object-toss configurations, which leave CFD off; its billiard example, "
            "which uses CFD, sets 0.1&#8201;m. Every pad lies within "
            f"{max(r1['max_pad_tool_distance_mm'], r2['max_pad_tool_distance_mm']):.0f}&#8201;mm of the tool in the "
            f"held state, so all {r1['pads']:,} pads at 1&#8201;mm spacing and all {r2['pads']} at 2&#8201;mm enter "
            f"the soft solve, against {r1['in_contact']} and {r2['in_contact']} in contact "
            "(<code>cfd_reach.json</code>).</p>"
            "<p>The probes evaluated CFD in the held grasp and 10&#8201;ms before the index recontacts, at a gap of "
            f"{rj['initial_fork_index_min_gap_m'] * 1e6:.0f}&#8201;&#181;m. In both states the ordinary derivative "
            "exists, and CFD replaced it as designed: at recontact the orientation derivative became "
            f"{rc['fork_ad'][0] / cpu0[0]:.1f} times the CPU value and the height derivative "
            f"{rc['fork_ad'][1] / cpu0[1]:.1f} times. The state CFD is meant for, a finger millimetres from the tool "
            "with no ordinary derivative, was not measured. CFD also always ran with the scan loop, which evaluates "
            "all 100 Newton iterations of the bed numerics, with up to 50 line-search steps each, in every simulation "
            f"step ({code('solver.py')} line 577). The fork&#8217;s examples use 2 to 4 Newton iterations and 10 "
            "line-search steps.</p>")


def parity_section(S):
    pad1, pad2, sph = S["pad1"], S["pad2"], S["sph"]
    gap = lambda d: max(abs(d["fork_initial_finger_dist_m"][f] - d["cpu_initial_contact"]["finger_min_dist_m"][f])  # noqa: E731
                        for f in ("thumb", "index", "middle"))
    jac, j1, f1, j2 = S["jac"], S["jvp1"], S["fd1"], S["jvp2"]
    sw = S["sweep"]
    ratio = [a / b for a, b in zip(jac["fork_base_metrics"], jac["cpu_base_metrics"])]
    ad_fd1 = max(rel(a, b) for a, b in zip(j1["ad_directional_derivative"], f1["fork_fd_directional_derivative"]))
    ad_fd20 = [rel(a, b) for a, b in zip(jac["ad_directional_derivative"], sw["fork_fd_sweep"]["0.0001"])]
    fork_cpu1 = [rel(a, b) for a, b in zip(f1["fork_fd_directional_derivative"], f1["cpu_fd_directional_derivative"])]
    ad2, cpu2 = j2["ad_directional_derivative"], j2["cpu_fd_directional_derivative"]
    vel = [d["max_qvel_abs_error"] for d in (pad1, pad2)]
    return (f"<p>The fork computes sphere-to-cylinder contact by ten steps of gradient descent on the sum of the two "
            f"signed distance functions ({code('collision_sdf.py')} lines 183&#8211;217 of {code(FORK)}), the same "
            "routine for the 0.75&#8201;mm pads and the 10.55&#8201;mm single sphere. Its contact gap differs from "
            f"CPU MuJoCo&#8217;s by {gap(pad1) * 1e6:.1f}&#8201;&#181;m on the 1&#8201;mm pads, "
            f"{gap(pad2) * 1e6:.1f}&#8201;&#181;m on the 2&#8201;mm pads and {gap(sph) * 1e3:.2f}&#8201;mm on the "
            "single sphere.</p>"
            f"<p>The forces differ more than the gaps. After one step from the held state the generalized velocities "
            f"differ by up to {g(min(vel), 2)}&#8211;{g(max(vel), 2)} (rad/s and m/s). After 20 steps the "
            f"tool&#8217;s body-x angular speed is {ratio[2]:.2f} times CPU MuJoCo&#8217;s and its body-x torque "
            f"{ratio[3]:.2f} times. At 1&#8201;ms on the 2&#8201;mm pads the derivatives of both have the opposite "
            f"sign to CPU MuJoCo&#8217;s: {g(ad2[2], 3)} against {g(cpu2[2], 3)} rad/s per rad, and {g(ad2[3], 3)} "
            f"against {g(cpu2[3], 3)}&#8201;N&#8201;m/rad. The fork&#8217;s AD agrees with its own central "
            f"differences to {pct(ad_fd1, 2)} at 1&#8201;ms and to {pct(min(ad_fd20), 2)}&#8211;{pct(max(ad_fd20), 1)} "
            "at 20&#8201;ms with a 10<sup>&#8722;4</sup>&#8201;rad step, so the disagreement with CPU MuJoCo comes from "
            "the forward model. The column &#8220;AD error&#8221; of the contact-gradients page, "
            f"{pct(min(fork_cpu1), 0)}&#8211;{pct(max(fork_cpu1), 0)}, is this forward difference.</p>")


def uses_section(S):
    pads, pads2 = S["pads"], S["pads2"]
    t1 = pads["methods"]["cem"]["wall_s"] / 256
    t2 = pads2["methods"]["cem"]["wall_s"] / 256
    att = S["attempt"]
    return ("<p>The reorientation policies are trained with PPO in MuJoCo-Warp through mjlab. PPO estimates its "
            "gradient from sampled returns and uses no simulator derivative, so DiffMJX would change only the "
            "simulator. Training through simulator derivatives (analytic policy gradients) needs reverse-mode "
            "derivatives over hundreds of steps. The fork&#8217;s ordinary solver loop supports forward mode only. "
            "Under these numerics the scan loop costs 100 Newton iterations with up to 50 line-search steps per simulation step, "
            "and the 1&#8201;mm model did not finish compiling a 20&#8201;ms forward rollout in "
            f"{att['elapsed_before_stop_s'] / 60:.1f}&#8201;min on the GPU.</p>"
            f"<p>For plan search, a finite-difference gradient of the 13 plan values needs 14 CPU rollouts. They take "
            f"{14 * t1:.2f}&#8201;s on the 1&#8201;mm pads and {14 * t2:.2f}&#8201;s on the 2&#8201;mm pads, against "
            f"{pads['methods']['cem']['wall_s']:.1f} and {pads2['methods']['cem']['wall_s']:.1f}&#8201;s for the "
            "256-evaluation CEM. Local derivatives describe one contact mode: the packed-pad score jumps within "
            "10<sup>&#8722;4</sup> of the CEM128 plan, and from a fully opened index the local search dropped the "
            "tool while CEM found recontact. CFD supplies a derivative toward contact, which concerns first touch "
            "and regrasp, the state that was not measured.</p>"
            "<p>Finite differences cost one rollout per parameter, so a reverse-mode gradient pays when a plan has "
            "hundreds of values, such as per-step controls or a fingertip shape. Reverse mode through the fork needs "
            "the scan loop, which under these numerics evaluates up to 5,000 line-search steps per simulation "
            "step.</p>")


def bugs_section(S):
    corr = S["corr"]
    if corr is None:
        return pending(2)
    fr, jac, ls = S["frames"], S["jac"], S["ls"]
    j1, f1, j2, F = S["jvp1"], S["fd1"], S["jvp2"], S["fdstep"]
    rr = {r["label"]: r for r in corr["rerun"]}
    mesh, pads = rr["convex mesh, free"], rr["1 mm packed, free"]
    mcem = F["tips"]["convex mesh"]["plans"]["cem128"]["norm"]
    ad_fd = [rel(a, b) for a, b in zip(j1["ad_directional_derivative"], f1["fork_fd_directional_derivative"])]
    fk_cpu = [rel(a, b) for a, b in zip(f1["fork_fd_directional_derivative"], f1["cpu_fd_directional_derivative"])]
    ad_cpu = [rel(a, b) for a, b in zip(j1["ad_directional_derivative"], f1["cpu_fd_directional_derivative"])]
    first = jac["gradient_compile_s"] + jac["gradient_warm_s"] * 2 / 3
    items = [
        "Angular speed and torque were labelled about world " + im("x") + ". They are the free joint&#8217;s components "
        "about the tool&#8217;s body " + im("x") + " axis, which 20&#8201;ms after the held state lies "
        f"{fr['angle_between_tool_x_and_world_x_deg']:.1f}&#176; from world " + im("x") + ": the angular speed is "
        f"{g(fr['omega_body_rad_s'][0], 2)}&#8201;rad/s about body " + im("x") + f" and {g(fr['omega_world_rad_s'][0], 2)}"
        "&#8201;rad/s about world " + im("x") + f", the torque {g(fr['torque_body_Nm'][0], 2)} and "
        f"{g(fr['torque_world_Nm'][0], 2)}&#8201;N&#8201;m ({code('diffmjx_audit_probes.py frames')}). Both pages now "
        "say &#8220;about the tool&#8217;s body x axis&#8221;; the values are unchanged.",
        f"The column &#8220;AD error&#8221; measured the fork against CPU MuJoCo. In the 1&#8201;ms index-yaw "
        f"derivatives it read {pct(min(ad_cpu), 1)}&#8211;{pct(max(ad_cpu), 1)}. The fork&#8217;s AD differs from its "
        f"own finite differences by {pct(min(ad_fd), 3)}&#8211;{pct(max(ad_fd), 3)}, and the fork&#8217;s finite "
        f"differences differ from CPU MuJoCo&#8217;s by {pct(min(fk_cpu), 1)}&#8211;{pct(max(fk_cpu), 1)}, the two columns "
        "that replace it.",
        f"The lede left compilation out of the gradient cost: {jac['gradient_warm_s']:.1f}&#8201;s warm against "
        f"{1e3 * ls['cpu_fd_6_evals_s']:.1f}&#8201;ms, {grp(jac['gradient_warm_s'] / ls['cpu_fd_6_evals_s'])} times. "
        f"With the {jac['gradient_compile_s']:.1f}&#8201;s first call a first gradient takes {first:.1f}&#8201;s, "
        f"{grp(first / ls['cpu_fd_6_evals_s'])} times; Table&#160;4 of the page gains that column.",
        "The recontact derivative&#8217;s plan was called the sampled 2&#8201;mm pad plan. Its JSON names the "
        "1&#8201;mm release search and 2&#8201;mm pads; the page now says the 1&#8201;mm packed-pad CEM release plan "
        "replayed on 2&#8201;mm pads.",
        "Of the two sign changes at 1&#8201;ms on the 2&#8201;mm pads the page reported the torque&#8217;s. The "
        f"angular-speed derivative changes sign too ({g(j2['ad_directional_derivative'][2], 3)} against "
        f"{g(j2['cpu_fd_directional_derivative'][2], 3)}&#8201;rad/s per rad), and both are now stated.",
        f"The CPU timing key {code('cpu_fd_3_evals_s')} of {code('real_v1_diffmjx_pad_jvp.py')} held 12 evaluations "
        f"with {code('--all-directions')} ({1e3 * jac['cpu_fd_3_evals_s']:.1f}&#8201;ms, against "
        f"{1e3 * ls['cpu_fd_6_evals_s']:.1f}&#8201;ms for six evaluations of the local search). The script now records "
        f"{code('cpu_fd_s')} with {code('cpu_fd_evals')} and the Jacobian&#8217;s time apart; no page used the two "
        "affected values.",
        "The 400&#8201;ms search took its two-point derivatives at SciPy&#8217;s default step, which measures "
        f"collision noise on the convex mesh (Figure&#160;1); {code('--fd-step scipy')} reproduces the stored mesh "
        "search exactly. With a 10<sup>&#8722;2</sup> step, now the default, the mesh local search gains "
        f"{mesh['after']['direct']:+.3f} instead of {mesh['before']['direct']:+.3f} and the mesh hybrid "
        f"{mesh['after']['hybrid']:+.3f} instead of {mesh['before']['hybrid']:+.3f}; the packed-pad hybrid gains "
        f"{pads['after']['hybrid']:+.3f} instead of {pads['before']['hybrid']:+.3f} (Table&#160;3).",
    ]
    text = ("<p>Each defect was reproduced from the stored data or by the command named in its item before the fix. "
            "The text corrections are in the two pages of 7 October, rebuilt from their builders with a dated note "
            "that links this page; their measured numbers are unchanged. The corrected searches are new runs with the "
            "same plans, budget, grasp and CEM seed, in <code>searches/</code> and <code>corrections.json</code>.</p>")
    text += "<ol>" + "".join(f"<li>{x}</li>" for x in items) + "</ol>"
    text += rerun_table(corr)
    t = corr["transfer"]
    hy, ml = t["tpu6_padsT_hybrid"], t["tpu6_pt_direct_gradient"]
    text += (f"<p>Over the eight grip jitters the corrected packed-pad hybrid holds {hy['tpu6_padsT']['after']['held']} of 8 "
             f"on the packed pads at a gain of {hy['tpu6_padsT']['after']['gain_if_held']:+.3f} "
             f"({hy['tpu6_padsT']['before']['held']} of 8 at {hy['tpu6_padsT']['before']['gain_if_held']:+.3f} before). It holds "
             f"{hy['tpu6_pt']['after']['held']} of 8 on the convex mesh ({hy['tpu6_pt']['before']['held']} of 8 before). "
             f"The corrected mesh local plan holds {ml['tpu6_pt']['after']['held']} of 8 on the mesh at "
             f"{ml['tpu6_pt']['after']['gain_if_held']:+.3f} ({ml['tpu6_pt']['before']['held']} of 8 at "
             f"{ml['tpu6_pt']['before']['gain_if_held']:+.3f} before). JSON: <code>transfer-fd1e-2-mj360.json</code>.</p>")
    return text


def rerun_table(corr):
    rows = []
    for r in corr["rerun"]:
        b, a = r["before"], r["after"]
        sg = lambda x: f"{x:+.3f}".replace("-", "&#8722;")  # noqa: E731
        rows.append([r["label"], sg(b["direct"]), sg(a["direct"]), sg(b["cem"]), sg(b["hybrid"]), sg(a["hybrid"]),
                     a["held"]])
    t = table(["Fingertip, plan", "Local, SciPy step", "Local, 10<sup>&#8722;2</sup>", "CEM",
               "Hybrid, SciPy step", "Hybrid, 10<sup>&#8722;2</sup>", "Held after"], rows, text=True)
    return t + tcap("Orientation gain " + im(r"\Delta c") + " of the 400&#8201;ms searches from the seed-1 grasp with "
                    "SciPy&#8217;s default step (the runs of 7 October) and with a 10<sup>&#8722;2</sup> step; 256 "
                    "evaluations per method, CEM seed 20261007, so the CEM column is common to both. The last column "
                    "gives the corrected local and hybrid plans&#8217; held status (yes/no).")


def module_section(S):
    mod = S["module"]
    if mod is None:
        return pending(3)
    before = sum(x["before"] for x in mod["scripts"])
    after = sum(x["after"] for x in mod["scripts"])
    t = mod["tests"]
    exact = [c for c in mod["checks"] if c["max_abs_diff"] == 0]
    other = [c for c in mod["checks"] if c["max_abs_diff"] != 0]
    text = (f"<p>The nine scripts of 7 October import {code(mod['module'])} for the scene and held state "
            f"({code('make_scene')}, {code('held_scene')}), the 13-value plan ({code('plan_ctrl')}), the score and the "
            f"held criterion ({code('score_and_held')}, {code('release_stats')}), the four tool outputs "
            f"({code('metric_values')}, {code('cpu_metrics')}, {code('central_difference')}) and the MJX set-up that "
            f"native MJX and the fork share ({code('mjx_setup')}, {code('mjx_rollout')}, {code('mjx_finger_gaps')}). "
            f"The scripts shrink from {before:,} to {after:,} lines and keep their command lines and output keys; the "
            f"module has {mod['module_lines']} lines. {code(t['file'])} pins the plan schedule, the score, the release "
            f"statistics, the finite differences and the scene&#8217;s collision masks; its {t['passed']} tests pass in "
            f"{t['wall_s']:.1f}&#8201;s.</p>")
    text += (f"<p>Each refactored script reran a case of 7 October (Table&#160;4). {len(exact)} of the "
             f"{len(mod['checks'])} outputs equal the stored data exactly"
             + (", and the MuJoCo-Warp replay, in 32-bit arithmetic on the GPU, agrees to "
                f"{g(other[0]['max_abs_diff'], 2)}" if other else "") + ". Both pages of 7 October rebuild from their "
             "builders. Before the corrections the rebuild reproduced the committed files byte for byte; the numbers "
             "that differ now are those of the corrections above, and no measured value changed.</p>")
    rows = [((c["name"], "txt"), (c["interpreter"], "txt"), (code(c["compared_with"].split("/")[-1]), "txt"),
             "0" if c["max_abs_diff"] == 0 else g(c["max_abs_diff"], 2)) for c in mod["checks"]]
    text += table(["Check", "Interpreter", "Compared with", "Largest difference"], rows, text=True)
    text += tcap("Outputs of the refactored scripts against the stored data of 7 October, compared field by field "
                 "(scores, plans, traces, derivatives, poses, contact summaries; wall times excluded). Outputs in "
                 "<code>regression/</code>; JSON: <code>module_check.json</code>.")
    return text


def onset_figure(on):
    c0 = on["initial_vertical_cos"]
    series = [("turn", "cpu", "var(--s1)", "", "CPU MuJoCo, turn plan"), ("turn", "fork", "var(--s2)", "", "fork, turn plan"),
              ("hold", "cpu", "var(--s1)", "6 4", "CPU MuJoCo, hold plan"), ("hold", "fork", "var(--s2)", "6 4",
                                                                           "fork, hold plan")]
    vals = [r["vertical_cos"] - c0 for p_ in ("turn", "hold") for b in ("cpu", "fork") for r in on["plans"][p_][b]]
    lo, hi = min(vals + [0.0]), max(vals)
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    w, h, left, right, top, bottom = 760, 360, 92, 190, 24, 56
    X = lambda t: left + t / on["horizon_ms"] * (w - left - right)  # noqa: E731
    Y = lambda v: top + (1 - (v - lo) / (hi - lo)) * (h - top - bottom)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" role="img" aria-label="Orientation gain in the fork and CPU MuJoCo">']
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / 4))
    for m in (1, 2, 5):
        if span / (step * m) <= 6:
            step *= m
            break
    v = math.ceil(lo / step) * step
    while v <= hi:
        yy = Y(v)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{w - right}" y2="{yy:.1f}" stroke="var(--rule2)" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{yy + 5:.1f}" text-anchor="end" font-size="15">{g(v, 2) if v else "0"}</text>')
        v += step
    for t in range(0, on["horizon_ms"] + 1, 20):
        xx = X(t)
        out.append(f'<text x="{xx:.1f}" y="{h - bottom + 20}" text-anchor="middle" font-size="15">{t}</text>')
    labels = []
    for plan, backend, col, dash, name in series:
        rows = on["plans"][plan][backend]
        pts = [(X(0), Y(0.0))] + [(X(r["t_ms"]), Y(r["vertical_cos"] - c0)) for r in rows]
        out.append(f'<polyline fill="none" stroke="{col}" stroke-width="2"' + (f' stroke-dasharray="{dash}"' if dash else "")
                   + ' points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
        for r, (a, b) in zip(rows, pts[1:]):
            out.append(f'<g><title>{esc(name)}: {r["t_ms"]} ms, gain {r["vertical_cos"] - c0:.5f}</title>'
                       f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="{col}" stroke="#fff" stroke-width="2"/></g>')
        labels.append((pts[-1][1], name))
    labels.sort()
    last = -1e9
    for yy, name in labels:                       # spread end labels at least 17 px apart
        yy = max(yy, last + 17)
        out.append(f'<text x="{w - right + 10}" y="{yy + 5:.1f}" font-size="14">{esc(name)}</text>')
        last = yy
    out.append(f'<text x="{(left + w - right) / 2:.1f}" y="{h - 12}" text-anchor="middle" font-size="15">'
               'Time after the held state (ms)</text>')
    out.append(f'<text x="20" y="{(top + h - bottom) / 2:.1f}" text-anchor="middle" font-size="15" '
               f'transform="rotate(-90 20 {(top + h - bottom) / 2:.1f})">Orientation gain</text></svg>')
    return "".join(out)


def onset_section(S):
    on = S["onset"]
    if on is None or "decision" not in on:
        return pending(4)
    par, crit = on["parity"], on["criterion"]
    t, hd = on["plans"]["turn"], on["plans"]["hold"]
    force = lambda r: r["finger_normal_force_N"]  # noqa: E731
    fmax = max(abs(force(a)[f] - force(b)[f]) for a, b in zip(t["fork"], t["cpu"]) for f in force(a))
    text = (f"<p>The fork and CPU MuJoCo {on['mujoco_version']} ran {on['horizon_ms']}&#8201;ms from the held state of the "
            f"2&#8201;mm packed pads ({sum(on['n_pads'].values())} spheres) under the controls that the 2&#8201;mm free "
            f"hybrid plan of 7 October applies in its first 120&#8201;ms, the turn onset, and under the hold plan. The "
            f"criterion was fixed "
            f"before the run: the fork&#8217;s orientation gain at {on['horizon_ms']}&#8201;ms within "
            f"{pct(crit['gain_rel'], 0)} of CPU MuJoCo&#8217;s and its tool position within "
            f"{crit['position_m'] * 1e3:.1f}&#8201;mm. The turn plan gains {g(par['gain_cpu'], 3)} in CPU MuJoCo and "
            f"{g(par['gain_fork'], 3)} in the fork, a difference of {pct(par['gain_rel_error'], 1)}; the tool positions "
            f"differ by {par['tool_position_error_m'] * 1e3:.3f}&#8201;mm and the fingers&#8217; normal forces by up to "
            f"{fmax:.2f}&#8201;N (Figure&#160;2). ")
    if not par["holds"]:
        text += ("Forward parity fails, so by the stop rule no gradient or search was run.</p>")
    else:
        gr = on["gradient"]
        text += "Forward parity holds.</p>"
        text += (f"<p>One forward-mode gradient of the {on['horizon_ms']}&#8201;ms vertical cosine with respect to the five "
                 f"plan values that act before 120&#8201;ms cost {g(gr['first_call_s'], 3)}&#8201;s with compilation and "
                 f"{g(gr['warm_s'], 3)}&#8201;s warm. Central differences in CPU MuJoCo at a "
                 f"{gr['fd_step']:g} step cost {g(1e3 * gr['cpu_fd_s'], 3)}&#8201;ms for {gr['cpu_fd_evals']} rollouts, so "
                 f"the gradient costs {grp(gr['ratio_first'])} times the finite differences with compilation and "
                 f"{grp(gr['ratio_warm'])} times warm. The two directions agree (cosine {gr['cosine']:.3f}); a step of "
                 f"{gr['probe_step']:g} along each raises the CPU vertical cosine by {g(gr['cpu_gain_along_ad'], 3)} "
                 f"(AD) and {g(gr['cpu_gain_along_fd'], 3)} (finite differences). ")
        if on["decision"] == "stop":
            text += (f"The gradient costs more than {crit['stop_ratio']:.0f} times the CPU baseline, so by the stop rule "
                     "the action search was not run.</p>")
        else:
            text += "The action search follows.</p>"
    text += figure(onset_figure(on),
                   f"Orientation gain over the first {on['horizon_ms']}&#8201;ms of the 2&#8201;mm packed-pad continuation "
                   "in the fork and in CPU MuJoCo, under the turn plan (solid) and the hold plan (dashed), sampled every "
                   f"{on['sample_ms']}&#8201;ms. JSON: <code>turn_onset.json</code>.")
    text += (f"<p>The fork&#8217;s forward pass costs {g(on['fork_warm_100ms_s'], 3)}&#8201;s warm for "
             f"{on['horizon_ms']}&#8201;ms against {g(1e3 * hd['cpu_rollout_s'], 3)}&#8201;ms in CPU MuJoCo with contact "
             f"summaries every 10&#8201;ms. Its first 10&#8201;ms chunk, with compilation, took "
             f"{g(on['fork_first_chunk_s'], 3)}&#8201;s.</p>")
    return text


def lede(S):
    on, corr = S["onset"], S["corr"]
    mesh = {r["label"]: r for r in corr["rerun"]}["convex mesh, free"] if corr else None
    if on is None or "decision" not in on:
        onset = "<i>[turn-onset result, step 4]</i> "
    elif not on["parity"]["holds"]:
        onset = (f"At the turn onset on 2&#8201;mm pads the fork&#8217;s 100&#8201;ms orientation gain differs from CPU "
                 f"MuJoCo&#8217;s by {pct(on['parity']['gain_rel_error'], 1)}, so forward parity fails and no gradient "
                 "search was run. ")
    else:
        gr = on["gradient"]
        onset = (f"The fork&#8217;s derivatives are correct for its own model, and at the turn onset on 2&#8201;mm pads its "
                 f"100&#8201;ms trajectory stays within {pct(on['parity']['gain_rel_error'], 1)} of CPU MuJoCo&#8217;s, but "
                 f"one gradient costs {grp(gr['ratio_first'])} times the CPU finite differences "
                 f"({grp(gr['ratio_warm'])} times warm)"
                 + (", so the stop rule ended the comparison before any gradient search. " if on["decision"] == "stop"
                    else ". "))
    tail = (f"On the convex mesh the plan search&#8217;s finite-difference step measured collision noise; corrected, the "
            f"mesh local search gains {mesh['after']['direct']:.3f} instead of {mesh['before']['direct']:.3f}."
            if mesh else "")
    return ("No policy was ever trained with DiffMJX; on 7 October the released fork&#8217;s derivatives were measured "
            "at the held D1 grasp over 1&#8211;20&#8201;ms, and 400&#8201;ms plans were searched in CPU MuJoCo. "
            + onset + "Contact from distance, meant for bodies that do not yet touch, was never tested in that state. "
            + tail)


def limits_section(S):
    return ("<p>Every measurement uses the D1 hand, its seed-1 grasp and the bed numerics (1&#8201;ms step, Newton "
            "with 100 iterations, tolerance 10<sup>&#8722;10</sup>). The fork is the commit pinned on 7 October, run "
            "in 64-bit arithmetic on the CPU; its GPU timings are those of 7 October. The CPU references use MuJoCo "
            "3.3.1 beside the fork and MuJoCo 3.6 in the project environment.</p>")


def next_section(S):
    items = [
        ("First touch with contact from distance", "Start the D1 continuation with the index retracted to tool gaps of "
         "0.5, 1 and 3&#8201;mm, and compare the fork&#8217;s CFD gradient of the index force after 50&#8201;ms with "
         "CPU finite differences of a smoothed contact objective and with CEM at equal wall time; extend "
         f"{code('scripts/diffmjx_turn_onset.py')} with a gap start. The CFD gradient is useful if a step along it "
         "closes the gap in CPU MuJoCo where the ordinary gradient is zero; it fails if the step leaves the gap "
         "open."),
        ("MuJoCo&#8217;s own derivatives in a local planner", "Linearize the CPU model with "
         f"{code('mjd_transitionFD')} along the 400&#8201;ms continuation and run iLQR from the hold plan, scored "
         f"like the search; new script {code('scripts/d1_ilqr.py')} on {code('real_v1_d1_continuation.py')}. "
         "Keep it if it beats CEM&#8217;s gain of 0.114 within the 10&#8201;s that CEM takes."),
        ("Fork cost at the fork&#8217;s own solver settings", "Rerun the gradient stage of "
         f"{code('scripts/diffmjx_turn_onset.py')} with the Newton solver at 4 iterations and 10 line-search steps (the "
         "fork&#8217;s examples) and in 32-bit arithmetic, after checking forward parity at those settings. The stop "
         "decision stands if one gradient still costs more than 100 times the CPU finite differences at the same "
         "settings."),
    ]
    return "<ol>" + "".join(f"<li><b>{t}.</b> {d}</li>" for t, d in items) + "</ol>"


def footer(S):
    return ("<p>Rebuild: <code>.venv/bin/python scripts/diffmjx_audit_page.py</code>. The commands of every check, "
            f"search and regression run are in the runbook <code>{REL}/README.md</code>. Data: "
            f"<code>{REL}/</code>, <code>docs/experiments/20261007-diffmjx/</code>, "
            "<code>docs/experiments/20261007-contact-gait/</code>. The fork&#8217;s source is the pinned clone under "
            "<code>logs/20261007-diffmjx/mujoco/</code> (git-ignored; commits in the runbook of 7 October).</p>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", action="store_true")
    DRAFT[0] = ap.parse_args().draft
    os.makedirs(A, exist_ok=True)
    S = load()
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": "2026-10-11"}
    v["GLOSSARY"] = glossary()
    v["SCRIPTS"] = scripts_section(S)
    v["HORIZON"] = horizon_section(S)
    v["COST"] = cost_section(S)
    v["FD"] = fd_section(S)
    v["CFD"] = cfd_section(S)
    v["PARITY"] = parity_section(S)
    v["USES"] = uses_section(S)
    v["BUGS"] = bugs_section(S)
    v["MODULE"] = module_section(S)
    v["ONSET"] = onset_section(S)
    v["LIMITS"] = limits_section(S)
    v["NEXT"] = next_section(S)
    v["FOOTER"] = footer(S)
    v["LEDE"] = lede(S)
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    t, n_math = render_tex(t)
    open(OUT, "w").write(retro_style.apply(t))
    print(f"formulas {n_math}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)" + (" [draft]" if DRAFT[0] else ""))


if __name__ == "__main__":
    main()
