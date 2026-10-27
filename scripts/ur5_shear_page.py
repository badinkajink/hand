#!/usr/bin/env python3
"""Page on the shear, the friction coefficient and the friction torque of the printed fingertip measured on the UR5, and on
impratio under the policy trainer's solver (hand-off 20261026-shear_friction_impratio, owner 2026-10-26: "cant we just set
impratio to 100 ...", "can we use the existing ur5 data + our sprung skin ...", "can't we measure the friction coefficient
from that + the fingertip geometry?", and the friction torque about the normal). Reads the rows and numbers of
scripts/ur5_shear_slides.py, ur5_shear_impratio.py, ur5_shear_skin.py, ur5_shear_torsion.py and ur5_shear_figures.py; writes
docs/experiments/20261026-shear_friction_impratio/20261026-shear_friction_and_impratio_of_the_printed_fingertip.html.

    .venv/bin/python scripts/ur5_shear_page.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import ur5_tip_calibration_page as P  # noqa: E402   (esc, code, fig, film, table, TEX_RE)

D = ROOT / "docs/experiments/20261026-shear_friction_impratio"
MEDIA, DATA = D / "media", D / "data"
PAGE = D / "20261026-shear_friction_and_impratio_of_the_printed_fingertip.html"
TEMPLATE = ROOT / "scripts/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
EXP = ROOT / "docs/experiments"
L = dict(
    cal=EXP / "20261026-fingertip_calibration/20261026-fingertip_calibration_on_the_ur5.html",
    gt=EXP / "20261022-ur5_tip_calibration/20261022-ur5_tip_ground_truth_and_predictors.html",
    skin=EXP / "20261012-sprung_skin/20261012-sprung_skin_presliding.html",
    contact=ROOT / "docs/overviews/contact_model.html",
    overview=ROOT / "docs/overviews/printed_fingertips.html",
    servo=EXP / "20261011-servo_plant_policies/20261011-servo_plant_contact_model_policies.html",
    fec=EXP / "20261018-fe_contact_calibration",
)


def link(p, text):
    return f'<a href="{P.esc(os.path.relpath(p, D))}">{text}</a>'


def f0(x):
    return f"{x:.0f}"


WORDS = ("none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def sig2(x):
    """Two significant figures with thousands separators: 493578 -> 490,000."""
    from math import floor, log10
    if x == 0:
        return "0"
    d = 10 ** (floor(log10(abs(x))) - 1)
    return f"{round(x / d) * d:,.0f}"


def sections(N):
    S1 = N["slides"]["cb2t"]
    T1 = N["slides"]["cb1r"]
    b1, b2 = S1["by_level"]["1"], S1["by_level"]["2"]
    c1, c2 = T1["by_level"]["1"], T1["by_level"]["2"]
    C = N["creep"]
    R = N["replays"]
    G = N["grasp"]
    K = N["skin_slides"]
    T = N["torsion"]
    tq = {F: (min(v["tau_on_Nmm"]), max(v["tau_on_Nmm"]), v["hertz_Nmm"], v["measured"]) for F, v in T.items()}
    cr = lambda key, ir: C[f"{key}_rl_ir{ir}"]["median_um_s"]  # noqa: E731
    sph, rod = "sphere2_E0.9008", "rod1_E1.9974"
    r10, r100, r300 = (R[f"d1d8_E10_ir{x}"] for x in (10, 100, 300))
    q10, q100, q300 = (R[f"d1d8_E2_ir{x}"] for x in (10, 100, 300))
    g = lambda k: G[k]  # noqa: E731
    V = N["vector"]
    v1, v2 = V["1"], V["2"]
    SW = (10, 20, 30, 50, 70, 100, 300)
    div10 = [R[f"d1d8_E10_ir{x}"]["diverged"] for x in SW[1:]]
    div2 = [R[f"d1d8_E2_ir{x}"]["diverged"] for x in SW[1:-1]]
    held2 = [R[f"d1d8_E2_ir{x}"]["held"] for x in SW[:-1]]
    d2d = [x for k in SW[1:] for x in (R[f"d1d8_E10_ir{k}"]["drop_to_div_s"] or [])]
    ns10, ns3 = R["d1d8_E10_ir10_ns10"], R["d1d8_E10_ir10_ns3"]
    ns10q, ns3q = R["d1d8_E2_ir10_ns10"], R["d1d8_E2_ir10_ns3"]
    mjw = [G[f"mjw_pads_E10_ir{x}"] for x in SW[:-1]]
    DV = N["divergence"]
    gp = [DV[f"grasp_p{k}"] for k in (1, 4, 5, 8, 11, 14)]
    rp = [DV["d1d8_D2_s0"], DV["d1d8_D6_s0"]]
    held_rr = lambda st: [c for c in gp if c["settings"][st]["held"]]  # noqa: E731

    lede = (
        f"On 87 UR5 slides of one printed fingertip, the friction coefficient of the 85A pad on a printed PLA sphere is "
        f"{v1['mu_const']['median']:.2f} at 1&#8202;N and {v2['mu_const']['median']:.2f} at 2&#8202;N. Before slip the pad moves {f0(b1['give_um_N']['median'])}&#8202;&#181;m per "
        f"newton of shear at 1&#8202;N and {f0(b2['give_um_N']['median'])}&#8202;&#181;m/N at 2&#8202;N. In the finite-element model "
        "it moves 48&#8211;177&#8202;&#181;m/N under the 12.5&#8202;mm rod. Under the policy trainer&#8217;s solver every impratio from "
        f"20 to 100 makes {min(div10)}&#8211;{max(div10)} of the 24 D1&#8211;D8 replays diverge after they drop the tool. In MuJoCo-Warp "
        f"the scripted grasp keeps {min(x['held_placements'] for x in mjw)}&#8211;{max(x['held_placements'] for x in mjw)} of 64 "
        f"placements up to impratio 100, and the tool&#8217;s slip falls from {f0(mjw[0]['slip_um_s'])} to "
        f"{f0(mjw[-1]['slip_um_s'])}&#8202;&#181;m/s. The divergence is the flung tool&#8217;s spin growing in free flight under the "
        f"implicitfast integrator. Noslip at impratio 10 lowers the replays&#8217; creep from {f0(r10['creep_um_s'])} to "
        f"{f0(ns10['creep_um_s'])}&#8202;&#181;m/s without divergence. MuJoCo-Warp has no noslip. Pads on a skin "
        f"sized from the slides match the measured shear within {K['skin_ir1000_1N']['rms_N']:.2f}&#8202;N RMS at impratio 1000. "
        "At impratio 100 creep flattens their loops. The pads&#8217; friction torque, "
        f"{tq['1'][0]:.1f}&#8211;{tq['5'][1]:.1f}&#8202;N&#8202;mm at 1&#8211;5&#8202;N, is below the HEX-E&#8217;s resolution.")
    summary = (
        f"UR5 slides of the printed fingertip: friction coefficient {v1['mu_const']['median']:.2f} and {v2['mu_const']['median']:.2f} "
        "(PLA sphere on 85A, 1 and 2 N, from the magnitude of the shear at slip), "
        f"give before slip {f0(b1['give_um_N']['median'])} and {f0(b2['give_um_N']['median'])} um/N at 1 and 2 N; under the "
        f"trainer's solver every impratio from 20 to 100 makes {min(div10)}-{max(div10)} of 24 D1-D8 replays diverge after the "
        f"drop (CPU), the CPU grasp diverges from 50 and the MuJoCo-Warp grasp never (slip {f0(mjw[0]['slip_um_s'])} to "
        f"{f0(mjw[-1]['slip_um_s'])} um/s at 10 to 100); the divergence is the flung tool's spin running away under "
        f"implicitfast; noslip 10 at impratio 10 cuts the replays' creep {f0(r10['creep_um_s'])} to {f0(ns10['creep_um_s'])} "
        "um/s (CPU only); a slide-sized skin matches the loops only at impratio 1000; the friction torque is below the HEX-E's "
        "resolution.")

    S = []
    terms = [
        ("Slide", "A turn of the pressed pad about the 5&#8202;mm sphere&#8217;s centre. The contact stays at one place on the pad "
         "while the sphere&#8217;s surface slides over it. Each slide of session cb2t turns in eight directions 45&#176; apart. "
         "In each direction it turns to 1.146 and 2.292&#176;, 0.1 and 0.2&#8202;mm at the surface, and back to 0&#176;. The "
         "shear-* slides also turn to 0.3&#8202;mm (" + link(L["cal"], "calibration page") + ", Section 4)."),
        ("Travel $d$", "The motion of the pad&#8217;s contact point along the direction of the turn, from the robot&#8217;s measured "
         "poses less the HEX-E&#8217;s deflection under the measured wrench, in &#181;m."),
        ("Shear $s$", "The component of the force on the pad against the travel, in N. The HEX-E is read at 100&#8202;Hz, in steps of "
         "0.1&#8202;N, and averaged over bins of 0.1&#8202;s."),
        ("Spring-and-slider fit", "The shear of one direction fitted by $s=s_0+z$, where $z$ follows the travel with stiffness "
         "$k_t$ until it reaches $\\pm\\mu f_n$ and then slides. $f_n$ is the measured normal force, and $s_0$ the HEX-E&#8217;s "
         "offset along the direction."),
        ("Give before slip", "The fitted compliance $1/k_t$, in &#181;m/N: the travel per newton of shear while the contact "
         "sticks. In the finite-element (FE) model the pad moves 48&#8211;177&#8202;&#181;m/N under the 12.5&#8202;mm rod."),
        ("Vector fit", "The same model in the tangent plane, fitted to the 2D shear of a slide&#8217;s eight directions in "
         "sequence: a spring of stiffness $k_t$ and an isotropic slider whose shear vector is returned to the circle of radius "
         "$\\mu f_n$, plus one constant in-plane offset per slide."),
        ("Friction coefficient $\\mu$", "The slider&#8217;s radius over the normal force in the vector fit: the magnitude of the "
         "shear at slip. The scenes are run with 1."),
        ("In-plane offset", "The fitted constant part of a slide&#8217;s shear vector, given over the mean normal force."),
        ("Shear left at the return", "$z$ at the end of a direction, when the turn is back at 0&#176;, in N. Negative values oppose "
         "the outward travel."),
        ("Impratio $\\kappa$", "MuJoCo&#8217;s ratio of the friction rows&#8217; impedance to the normal rows&#8217;. Below the friction "
         "limit the rows act as dampers, and the creep falls as $1/\\kappa$ (contact overview, step&#160;8)."),
        ("Creep", "The sliding speed under a held load below slip, in &#181;m/s. In the UR5 rig: the sphere&#8217;s tangential speed "
         "over the second half of 1&#8202;s under a press hold&#8217;s measured force vector. In the D1&#8211;D8 replays: the "
         "tool&#8217;s drift from the end of the 1.5&#8202;s hold to the end of the run, 4&#8202;s later. In the scripted grasp: the "
         "speed of the tool&#8217;s centre in the fingertips&#8217; frames from 1 to 2&#8202;s, the median over the three fingertips."),
        ("Trainer&#8217;s numerics", "The step and solver of the reinforcement-learning trainer: 2&#8202;ms, implicitfast, elliptic "
         "cone, Newton with 10 iterations, 20 line-search iterations and tolerance $10^{-8}$."),
        ("Iteration cap", "The share of steps on which all 10 solver iterations were used."),
        ("Held", "D1&#8211;D8: at the end the tool is less than 20&#8202;mm below its grip height and touched by at least two "
         "fingertips. Readiness commands: the same, with two fingertips each pressing with the tool&#8217;s weight. Scripted "
         "grasp: the tool rose with the palm to within 20&#8202;mm and two fingertips touch it."),
        ("Diverged", "MuJoCo found a non-finite acceleration or one above $10^{10}$ and reset the state (warning BADQACC), or a "
         "MuJoCo-Warp world became non-finite."),
        ("Noslip", "MuJoCo&#8217;s second pass over the friction rows after the main solve: a projected Gauss&#8211;Seidel sweep "
         "without their regulariser, repeated <code>noslip_iterations</code> times. The trainer runs without it."),
        ("Cone state", "A contact whose force lies on the surface of its friction cone, so that it slides (MuJoCo&#8217;s "
         "constraint state CONE)."),
        ("Last improvement", "The fall of the solver&#8217;s cost in its last iteration of a step, as MuJoCo reports it "
         "(<code>mjSolverStat.improvement</code>). The solver stops when it falls below the tolerance, or at the cap."),
        ("Kick", "Of the steps with a contact on the tool, the one that adds the most kinetic energy to the tool."),
        ("Free flight", "Steps on which nothing touches the tool."),
        ("Placement", "The tool&#8217;s start offset of a run: none for seed 0, else 2&#8202;mm and 2&#176; standard deviation."),
        ("Sprung skin", "The pads on a 2&#8202;g body joined to the fingertip by two sideways slides and a twist hinge, each with "
         "a spring and critical damping (" + link(L["skin"], "skin page") + ")."),
        ("Spin-onset torque", "The torque about the contact normal at which the sphere turns at 30&#176;/s, under a ramp of "
         "2&#8202;N&#8202;mm/s, in N&#8202;mm. Its arm is the torque over $\\mu f_n$."),
        ("Torsion", "The HEX-E&#8217;s moment about the normal through the contact in a twist, half the difference between the "
         "way out and the way back at the same angle, in N&#8202;mm."),
        ("&#181;s per step", "The wall time of one physics step on one CPU core, or of one world-step in a MuJoCo-Warp batch of "
         "256 worlds."),
    ]
    S.append(("terms", "Terms and metrics", '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms)
              + "</dl>"))

    # 2. slides
    tab1 = P.table(
        ["Tip, press level", "Directions", "Normal force (N)", "Give before slip (&#181;m/N)", "First 50&#8202;&#181;m (&#181;m/N)",
         "Shear left at the return (N)"],
        [[f"Tip {t}, {F}&#8202;N", v["n"], f"{v['f_n_N']['median']:.2f}",
          f"{v['give_um_N']['median']:.0f} ({v['give_um_N']['p25']:.0f}&#8211;{v['give_um_N']['p75']:.0f})",
          f"{v['give_load50_um_N']['median']:.0f}", f"{v['z_back_N']['median']:+.2f}"]
         for t, F, v in ((2, "0.5", S1["by_level"]["0.5"]), (2, "1", b1), (2, "1.5", S1["by_level"]["1.5"]), (2, "2", b2),
                         (2, "3", S1["by_level"]["3"]), (1, "1", c1), (1, "2", c2))],
        num_cols=(1, 2, 3, 4, 5), n=1,
        caption="Give before slip on the slides of tip 2 (cb2t, 2026-10-25) and tip 1 (cb1r), by press level: medians and "
                "quartiles over the directions of the spring-and-slider fit along the travel. The normal force is the mean "
                "during the slide; the press stopped at the level.")
    tabF = P.table(
        ["Press level", "Slides", "Friction coefficient, vector fit", "Without offset", "Offset / normal force",
         "Largest |shear| / normal force", "Along the travel only"],
        [[f"{F}&#8202;N", v["n"], f"{v['mu_const']['median']:.2f} ({v['mu_const']['p25']:.2f}&#8211;{v['mu_const']['p75']:.2f})",
          f"{v['mu_none']['median']:.2f}", f"{v['b_over_fn']:.2f}",
          f"{v['max_ft_over_fn']['median']:.2f} ({v['max_ft_over_fn']['p10']:.2f}&#8211;{v['max_ft_over_fn']['p90']:.2f})",
          f"{bb['mu']['median']:.2f}"] for F, v, bb in (("1", v1, b1), ("2", v2, b2))],
        num_cols=(1, 2, 3, 4, 5, 6), n=2,
        caption="Friction coefficient of tip 2 on the PLA sphere. Vector fit: an isotropic slider and spring with one in-plane "
                "offset per slide, median and quartiles over slides. Largest |shear| / normal force: the largest ratio over the "
                "holds of one direction, median with the 10th and 90th percentiles over directions. Along the travel only: the "
                "slider&#8217;s limit in the fit of each direction&#8217;s component along its travel.")
    slides = (
        f"<p>The pad&#8217;s friction coefficient on the printed PLA sphere is {v1['mu_const']['median']:.2f} at 1&#8202;N and "
        f"{v2['mu_const']['median']:.2f} at 2&#8202;N on tip 2 (Table&#160;2). It is the magnitude of the shear at slip over the normal "
        "force, from a vector fit of each slide&#8217;s eight directions. In that fit an isotropic slider sits in series with a "
        "sideways spring, and one in-plane offset is fitted per slide. The offset is "
        f"{v1['b_over_fn']:.2f} of the normal force at 1&#8202;N and {v2['b_over_fn']:.2f} at 2&#8202;N. The presses show the same "
        f"part before any turn, {V['press_ft_over_fn']:.2f} of the normal force. Between the 1 and 2&#8202;N slides at one target "
        f"it grows {V['offset_pairs']['ratio_median']:.1f} times, a median {V['offset_pairs']['angle_median_deg']:.0f}&#176; apart. "
        "It is therefore read as the tangential part of the normal force along a normal tilted by about 8&#176;, and not as "
        f"friction. Without the offset the fit gives {v1['mu_none']['median']:.2f} and {v2['mu_none']['median']:.2f}.</p>"
        "<p>The largest ratio of the shear vector&#8217;s magnitude to the normal force in a direction has a median of "
        f"{v1['max_ft_over_fn']['median']:.2f} at 1&#8202;N and {v2['max_ft_over_fn']['median']:.2f} at 2&#8202;N. It bounds the "
        "coefficient from below only if all of the shear is friction. Here it is close to the coefficient plus the offset. "
        "The component of the shear along the travel alone gives "
        f"{b1['mu']['median']:.2f} and {b2['mu']['median']:.2f}, because shear stored across the travel is left out of it. The "
        f"scenes are run with 1. In training the friction is randomised over 0.55&#8211;1.15 of it (Figure&#160;1c).</p>"
        f"<p>Before slip the pad moves {f0(b1['give_um_N']['median'])}&#8202;&#181;m per newton of shear at 1&#8202;N and "
        f"{f0(b2['give_um_N']['median'])}&#8202;&#181;m/N at 2&#8202;N (Table&#160;1). The give falls as "
        f"$f_n^{{{S1['give_power_law']['exponent']:.2f}}}$ with the normal force. Mindlin&#8217;s compliance of an elastic "
        "half-space under the same sphere falls as $f_n^{-1/3}$. At the modulus fitted to the rod presses, 2.0&#8202;MPa, it is "
        "246 and 195&#8202;&#181;m/N at 1 and 2&#8202;N (Figure&#160;1b). The FE pad moves 48&#8211;177&#8202;&#181;m/N sideways "
        "under the 12.5&#8202;mm rod, a larger contact. A straight line through the first 50&#8202;&#181;m of travel gives "
        f"{f0(b1['give_load50_um_N']['median'])} and {f0(b2['give_load50_um_N']['median'])}&#8202;&#181;m/N, since the fitted "
        "slider slips at once where the real contact slips gradually.</p>"
        f"<p>In the fit along the travel the slider reaches its limit both ways in {sum(v['slip_both'] for v in S1['by_level'].values())} "
        f"of {S1['mu_all']['n']} directions. In the vector fit the shear is at the slip limit for {100 * v1['slip_share']:.0f}&#8202;% of "
        f"the time at 1&#8202;N and {100 * v2['slip_share']:.0f}&#8202;% at 2&#8202;N. At "
        f"the return to 0&#176; a shear of {abs(b1['z_back_N']['median']):.2f}&#8202;N is left at 1&#8202;N and "
        f"{abs(b2['z_back_N']['median']):.2f}&#8202;N at 2&#8202;N. It is about {abs(b1['z_back_over_fn']['median']):.2f} of the "
        "normal force and opposes the outward travel. The next direction starts from that shear. At 1.146&#176; the robot "
        "travelled a median 80&#8211;94&#8202;&#181;m, against 100&#8202;&#181;m planned. It also moved up to 50&#8202;&#181;m along the "
        "normal, so the measured travel is fitted.</p>")
    S.append(("slides", "Friction coefficient and give before slip from the UR5 slides",
              slides + P.fig(1, MEDIA / "20261026-ur5_slides.png",
                             "Shear against travel on the UR5 slides. (a) One face slide at 1&#8202;N (slide-000-1N), its eight "
                             "directions, shear less the fitted offset. The direction across the finger is blue, with its fitted "
                             "spring and slider dashed. (b) Give before slip of every direction against the mean normal force "
                             "during the slide, with medians and quartiles per press level. Lines: Mindlin&#8217;s compliance under "
                             "the 5&#8202;mm sphere at 2.0 and 0.9&#8202;MPa; grey: the FE pad&#8217;s range under the 12.5&#8202;mm rod. "
                             "(c) Friction coefficient of each slide from the vector fit (blue; medians and quartiles per level) "
                             "and the largest shear over normal force of each slide&#8217;s directions, as a median (grey squares), "
                             "with the scenes&#8217; 1 and the trainer&#8217;s randomisation range. "
                             "Script: <code>scripts/ur5_shear_slides.py</code>.") + tabF + tab1))

    # 3. impratio (2026-10-27: the sweep 10-300, noslip and the divergence)
    def cell(d, k, fmt="{:.0f}"):
        v = (d or {}).get(k)
        return "&#8211;" if v is None else fmt.format(v)

    def trow(E, x, ns=0):
        sfx = f"_ns{ns}" if ns else ""
        r = R.get(f"d1d8_E{E}_ir{x}{sfx}")
        ic = R.get(f"icem_E{E}_ir{x}{sfx}")
        gc = G.get(f"cpu_pads_E{E}{'' if E == 10 else '_tc20'}_ir{x}{sfx}")
        gm = G.get(f"mjw_pads_E{E}_ir{x}") if E == 10 and not ns else None
        rig = C.get(f"{'sphere2_E10' if E == 10 else 'rod1_E1.9974'}_rl_ir{x}") if not ns else None
        lab = f"{'10' if E == 10 else '2.0'}&#8202;MPa, {x}" + (f", noslip {ns}" if ns else "")
        return [lab, cell(rig, "median_um_s", "{:.1f}") + (" (rod)" if rig and E != 10 else ""),
                f"{r['held']} / {r['diverged']}", cell(r, "creep_um_s"), f"{ic['held']} of 6" if ic else "&#8211;",
                f"{gc['held']} / {gc['nonfinite']}" if gc else "&#8211;", f"{gm['held_placements']} of 64" if gm else "&#8211;",
                cell(gc, "slip_um_s"), cell(gm, "slip_um_s"), cell(r, "us_held")]
    tab2 = P.table(
        ["Pads, impratio", "UR5 rig creep, sphere (&#181;m/s)", "D1&#8211;D8 held / diverged", "Creep of held (&#181;m/s)",
         "Readiness held", "Grasp CPU held / diverged", "Grasp MuJoCo-Warp held", "Slip CPU (&#181;m/s)",
         "Slip MuJoCo-Warp (&#181;m/s)", "&#181;s per step"],
        [trow(E, x) for E in (10, 2) for x in SW] + [trow(E, 10, ns) for E in (10, 2) for ns in (3, 10)],
        num_cols=(1, 2, 3, 4, 5, 6, 7, 8, 9), n=3,
        caption="Impratio and noslip under the trainer&#8217;s numerics. UR5 rig: the 812 pads under the sphere presses&#8217; "
                "forces at 10&#8202;MPa (10&#8202;MPa rows) and under the rod presses&#8217; at 2.0&#8202;MPa. D1&#8211;D8: 24 replays per "
                "setting on the UR5 pad&#8217;s shape, at a 20&#8202;ms time constant for 2.0&#8202;MPa. Readiness: the two logged "
                "pinned-iCEM runs from three placements. Grasp: the scripted close and lift on the servo plant&#8217;s pad scene "
                "of the RL runs, 16 placements on the CPU (2.0&#8202;MPa at 20&#8202;ms) and 64 in MuJoCo-Warp. Slip: median over "
                "the held grasps. &#181;s per step: the held D1&#8211;D8 replays on one core. Noslip rows: impratio 10 with "
                "<code>noslip_iterations</code> 3 or 10, on the CPU only.")
    imp = (
        f"<p>Under the trainer&#8217;s numerics the pads&#8217; creep falls as $1/\\kappa$ in the UR5 rig. At the sphere presses&#8217; "
        f"0.90&#8202;MPa it is {cr(sph, 10):.0f}, {cr(sph, 100):.0f} and {cr(sph, 300):.0f}&#8202;&#181;m/s at impratio 10, 100 and 300. "
        f"Under the rod presses at 2.0&#8202;MPa it is {cr(rod, 10):.0f}, {cr(rod, 100):.0f} and {cr(rod, 300):.1f}&#8202;&#181;m/s. The "
        "bed&#8217;s numerics, 1&#8202;ms and 100 iterations, give the same speeds, and no step reaches the iteration cap "
        "(Figure&#160;2a).</p>"
        "<p>On the hand the replays decide. The D1&#8211;D8 plans were replayed open loop from three placements on 1&#8202;mm pads of "
        "the UR5 pad&#8217;s shape, at impratio 10, 20, 30, 50, 70, 100 and 300 (Table&#160;3). At 10&#8202;MPa 8 or 9 of the 24 "
        f"replays keep the tool at every impratio, on hands {', '.join(r10['hands_held'][:-1])} and {r10['hands_held'][-1]}. The "
        "others drop it. At impratio 10 no "
        f"replay diverges. From 20 on, the replays that drop the tool diverge {min(d2d):.2f}&#8211;{max(d2d):.1f}&#8202;s after the "
        f"drop: {div10[0]} of 24 at 20, {div10[1]} at 30 and {min(div10[2:])}&#8211;{max(div10[2:])} from 50 to 300 (Figure&#160;2b). "
        f"The held tool creeps at {f0(r10['creep_um_s'])}&#8202;&#181;m/s at impratio 10 and {f0(R['d1d8_E10_ir20']['creep_um_s'])}"
        f"&#8202;&#181;m/s at 20. Its creep falls to {f0(r100['creep_um_s'])}&#8202;&#181;m/s at 100 and {f0(r300['creep_um_s'])}"
        "&#8202;&#181;m/s at 300.</p>"
        "<p>At 2.0&#8202;MPa the replays run at a 20&#8202;ms time constant. A pad this soft needs it: at 10&#8202;ms its impedance "
        f"$d_0$ falls to 0.05, the floor of the pad builder. Here {min(held2)}&#8211;{max(held2)} of the 24 replays are held from "
        f"impratio 10 to 100. {'One replay' if q10['diverged'] == 1 else str(q10['diverged']) + ' replays'} diverge"
        f"{'s' if q10['diverged'] == 1 else ''} at impratio 10, and {min(div2)}&#8211;{max(div2)} from 20 to 100. The held "
        f"tool&#8217;s creep falls from {f0(q10['creep_um_s'])}&#8202;&#181;m/s at impratio 10 to {f0(q100['creep_um_s'])}&#8202;&#181;m/s "
        "at 100. The readiness commands, two logged pinned-iCEM runs from three placements, keep the tool in 6 of 6 up to "
        f"impratio 50 at both moduli. At 100 they keep it in {R['icem_E10_ir100']['held']} of 6 at 10&#8202;MPa and "
        f"{R['icem_E2_ir100']['held']} of 6 at 2.0&#8202;MPa.</p>"
        "<p>The trainer steps the scripted grasp in MuJoCo-Warp, on the servo plant&#8217;s pad scene of the RL runs. There "
        f"{mjw[0]['held_placements']} of 64 placements are held at impratio 10 to 70, and {mjw[-1]['held_placements']} at 100. No "
        f"world diverges at any setting. The tool slips on the fingertips at {f0(mjw[0]['slip_um_s'])}&#8202;&#181;m/s at impratio 10 "
        f"and {f0(mjw[-1]['slip_um_s'])}&#8202;&#181;m/s at 100. A world-step costs {min(x['us'] for x in mjw):.0f}&#8211;"
        f"{max(x['us'] for x in mjw):.0f}&#8202;&#181;s at 256 worlds. On the CPU all 16 grasps are held at impratio 10, 20 and 30. "
        f"At 50, {g('cpu_pads_E10_ir50')['nonfinite']} of 16 diverge, and at 100, {g('cpu_pads_E10_ir100')['nonfinite']}. Pads rescaled "
        "to 2.0&#8202;MPa at 10&#8202;ms ($d_0=0.055$) diverged in every CPU grasp and every MuJoCo-Warp world. At 20&#8202;ms no CPU "
        f"grasp diverges up to impratio 30, and {g('cpu_pads_E2_tc20_ir50')['nonfinite']}&#8211;{g('cpu_pads_E2_tc20_ir100')['nonfinite']} "
        f"diverge from 50 to 100. At 15&#8202;ms MuJoCo-Warp keeps all 64 placements at impratio 10 and 100.</p>"
        "<p>Noslip lowers the creep at the trainer&#8217;s impratio. With 10 noslip iterations at impratio 10 the held D1&#8211;D8 "
        f"replays creep at {f0(ns10['creep_um_s'])}&#8202;&#181;m/s at 10&#8202;MPa and {f0(ns10q['creep_um_s'])}&#8202;&#181;m/s at "
        f"2.0&#8202;MPa. With 3 iterations they creep at {f0(ns3['creep_um_s'])} and {f0(ns3q['creep_um_s'])}&#8202;&#181;m/s. With 10 "
        f"iterations no replay diverges at either modulus, and {ns10['held']} of 24 are held at 10&#8202;MPa. A step of a held "
        f"replay then costs {f0(ns10['us_held'])}&#8202;&#181;s, against {f0(r10['us_held'])}&#8202;&#181;s without noslip. The "
        f"scripted grasp on the CPU slips at {f0(g('cpu_pads_E10_ir10_ns10')['slip_um_s'])}&#8202;&#181;m/s with 10 iterations, "
        f"against {f0(g('cpu_pads_E10_ir10')['slip_um_s'])}&#8202;&#181;m/s without, at {f0(g('cpu_pads_E10_ir10_ns10')['us'])} "
        f"against {f0(g('cpu_pads_E10_ir10')['us'])}&#8202;&#181;s per step. MuJoCo-Warp has no noslip: its <code>put_model</code> "
        "raises <code>NotImplementedError</code> for any <code>noslip_iterations</code> above zero.</p>"
        "<p>The largest setting with no divergence therefore depends on the test. For the D1&#8211;D8 replays on the CPU it is "
        f"impratio 10, where the held tool creeps at {f0(r10['creep_um_s'])}&#8202;&#181;m/s. For the scripted grasp on the CPU it is "
        f"impratio 30, with a slip of {f0(g('cpu_pads_E10_ir30')['slip_um_s'])}&#8202;&#181;m/s. For the grasp in MuJoCo-Warp it is "
        f"impratio 100, the largest tried, with a slip of {f0(mjw[-1]['slip_um_s'])}&#8202;&#181;m/s. Noslip with 10 iterations at "
        f"impratio 10 diverges in neither CPU test, and the held replays creep at {f0(ns10['creep_um_s'])}&#8202;&#181;m/s.</p>")
    dvt = (
        "<p>The divergence is a runaway of the tool&#8217;s spin (Figure&#160;3). The six CPU grasps that diverge at impratio 100, and "
        "the replays of D2 and D6 from placement 0, were run again with every step recorded. While the fingers close, "
        f"{100 * min(c['cone_share_grip'] for c in gp):.0f}&#8211;{100 * max(c['cone_share_grip'] for c in gp):.0f}&#8202;% of the "
        "pad contacts on the tool are in the cone state. The solver uses all 10 iterations on "
        f"{100 * min(c['cap_grip'] for c in gp):.0f}&#8211;{100 * max(c['cap_grip'] for c in gp):.0f}&#8202;% of these steps. Its last "
        f"iteration still lowers the cost by up to {sig2(min(c['improvement_max_grip'] for c in gp))}&#8211;"
        f"{sig2(max(c['improvement_max_grip'] for c in gp))}. The tool rattles between the fingertips at up to "
        f"{min(c['v_max_grip_mm_s'] for c in gp) / 1e3:.1f}&#8211;{max(c['v_max_grip_mm_s'] for c in gp) / 1e3:.1f}&#8202;m/s and "
        f"{sig2(min(c['w_max_grip'] for c in gp))}&#8211;{sig2(max(c['w_max_grip'] for c in gp))}&#8202;rad/s. It leaves the pads "
        f"{min(c['t_leave_s'] for c in gp):.2f}&#8211;{max(c['t_leave_s'] for c in gp):.2f}&#8202;s into the run and falls to the "
        "floor. In the two replays the dropped tool lies between the closing fingertips and the floor. There a single floor "
        "contact in the cone state, pushing with 43&#8211;74&#8202;N, kicks it to "
        f"{rp[0]['w_free']:,.0f} and {rp[1]['w_free']:,.0f}&#8202;rad/s. The friction rows of that contact have a regulariser of "
        "0.002, a hundredth of its normal row&#8217;s 0.2, as impratio 100 sets.</p>"
        "<p>In free flight the spin then grows exponentially. In grasp 1 it grows from "
        f"{gp[0]['w_free']:.0f}&#8202;rad/s at {gp[0]['t_free_s']:.2f}&#8202;s by a factor $e$ every {gp[0]['efold_ms']:.0f}&#8202;ms. In "
        f"the replays it grows by $e$ every {rp[0]['efold_ms']:.0f} and {rp[1]['efold_ms']:.0f}&#8202;ms. At 17&#8211;30&#8202;rad/s "
        "the free tool&#8217;s spin stays steady (grasp 4, 0.60&#8211;0.84&#8202;s). In the last 0.2&#8202;s before the warning the tool "
        "is in the air on 61&#8211;100&#8202;% of the steps. In five grasps it touches the floor while it spins above "
        "1,000&#8202;rad/s, and those contacts push with 3.6&#8202;kN to 9.1&#8202;MN. The warning fires when an acceleration of the "
        f"tool&#8217;s free joint passes $10^{{10}}$, {min(c['t_fail_s'] for c in gp):.2f}&#8211;{max(c['t_fail_s'] for c in gp):.2f}"
        f"&#8202;s into the grasps and {rp[0]['t_fail_s'] - rp[0]['t_drop_s']:.2f} and {rp[1]['t_fail_s'] - rp[1]['t_drop_s']:.2f}"
        "&#8202;s after the drops.</p>"
        "<p>The integrator decides the last stage. The implicitfast integrator of the trainer leaves the derivative of the "
        "Coriolis and centrifugal forces out of its implicit step. With MuJoCo&#8217;s full implicit integrator none of the eight "
        f"runs reaches the warning. {WORDS[6 - len(held_rr('implicit'))].capitalize()} of the six grasps still fling the tool, and "
        "the two replays end "
        f"without a warning. More iterations act on the grip. With 20 iterations {len(held_rr('it20'))} of the 6 grasps are held, "
        "no step reaches the cap, and 50 iterations give the same runs. Grasp 1 still diverges, at "
        f"{gp[0]['settings']['it20']['t_fail_s']:.2f}&#8202;s. A step of the held grasps costs "
        f"{min(c['settings']['it20']['us'] for c in held_rr('it20')):.0f}&#8211;{max(c['settings']['it20']['us'] for c in held_rr('it20')):.0f}"
        f"&#8202;&#181;s at 20 iterations, against {f0(g('cpu_pads_E10_ir10')['us'])}&#8202;&#181;s at impratio 10. At a 1&#8202;ms "
        f"step all {WORDS[len(held_rr('dt1ms'))]} grasps are held, at {min(c['settings']['dt1ms']['us'] for c in gp):.0f}&#8211;"
        f"{max(c['settings']['dt1ms']['us'] for c in gp):.0f}&#8202;&#181;s per step and twice the steps per second. The two replays "
        "diverge at 20 and 50 iterations and at 1&#8202;ms as well. MuJoCo-Warp offers Euler, RK4 and implicitfast, and no full "
        "implicit integrator.</p>")
    S.append(("impratio", "Impratio under the trainer's solver",
              imp + P.fig(2, MEDIA / "20261026-impratio.png",
                          "Impratio under the trainer&#8217;s numerics. (a) Creep against impratio. Orange and purple: the UR5 rig "
                          "under the measured forces of the 5&#8202;mm sphere at 0.90&#8202;MPa and the 12.5&#8202;mm rod at 2.0&#8202;MPa. "
                          "Green: the held D1&#8211;D8 replays at 10&#8202;MPa (open) and 2.0&#8202;MPa (filled). Black: the scripted grasp "
                          "at 10&#8202;MPa in MuJoCo-Warp (crosses) and on the CPU (circles). Blue: on the CPU at 2.0&#8202;MPa. Marks "
                          "left of 10: noslip with 3 (light) and 10 (dark) iterations at impratio 10. Grey: the slope of $1/\\kappa$. "
                          "(b) The share of runs that diverged. Scripts: <code>scripts/ur5_shear_impratio.py</code>, "
                          "<code>scripts/ur5_shear_figures.py</code>.")
              + tab2 + dvt
              + P.fig(3, MEDIA / "20261027-divergence.png",
                      "The divergence at impratio 100 on 10&#8202;MPa pads, every step recorded. (a) The scripted grasp from "
                      "placement 1. (b) The D2 replay from placement 0, from 2.3&#8202;s. Top: the tool&#8217;s spin and speed. "
                      "Middle: the pad contacts on the tool (left axis) and the tool&#8217;s contacts on the floor (right axis). "
                      "Bottom: the solver&#8217;s iterations per step, at the cap in black, and its last improvement (right axis). "
                      "Lines mark the drop, the kick, the start of free flight and MuJoCo&#8217;s warning. Script: "
                      "<code>scripts/ur5_shear_impratio.py divergence</code>.")
              + P.film(4, MEDIA / "20261026-impratio_replays.mp4", MEDIA / "20261026-impratio_replays.png",
                       "D1 (top) and D2 (bottom) replayed open loop at the trainer&#8217;s numerics on 10&#8202;MPa pads, impratio 10 "
                       "(left) and 100 (right), placement 0, at real speed. D1 keeps the tool at both. D2 drops it at both, and at "
                       "100 the simulation diverges 0.5&#8202;s after the drop.")))

    # 4. skin
    sz = N["skin_sizing"]
    tab3 = P.table(
        ["Model, impratio", "Give (&#181;m/N), 1 / 2&#8202;N", "Slip limit along the travel, 1 / 2&#8202;N", "RMS to measured (N), 1 / 2&#8202;N",
         "Shear left at the return (N), 1 / 2&#8202;N", "&#181;s per step"],
        [[f"{'Pads' if m_ == 'pads' else 'Sized skin'}, {x}",
          f"{K[f'{m_}_ir{x}_1N']['give_um_N']:.0f} / {K[f'{m_}_ir{x}_2N']['give_um_N']:.0f}",
          f"{K[f'{m_}_ir{x}_1N']['mu']:.2f} / {K[f'{m_}_ir{x}_2N']['mu']:.2f}",
          f"{K[f'{m_}_ir{x}_1N']['rms_N']:.2f} / {K[f'{m_}_ir{x}_2N']['rms_N']:.2f}",
          f"{K[f'{m_}_ir{x}_1N']['s_back_N']:+.2f} / {K[f'{m_}_ir{x}_2N']['s_back_N']:+.2f}",
          f"{K[f'{m_}_ir{x}_1N']['us']:.0f}"] for m_ in ("pads", "skin") for x in (10, 100, 1000)]
        + [["Measured", f"{K['skin_ir1000_1N']['give_measured_um_N']:.0f} / {K['skin_ir1000_2N']['give_measured_um_N']:.0f}",
            f"{K['skin_ir1000_1N']['mu_measured']:.2f} / {K['skin_ir1000_2N']['mu_measured']:.2f}", "&#8211;",
            f"{K['skin_ir1000_1N']['z_back_measured_N']:+.2f} / {K['skin_ir1000_2N']['z_back_measured_N']:+.2f}", "&#8211;"]],
        num_cols=(1, 2, 3, 4, 5), n=4,
        caption="The 38 face slides at 1 and 2&#8202;N (304 directions) replayed on the UR5 rig&#8217;s pads, plain and on the sized "
                "skin, at the slides&#8217; friction coefficients (0.37 at 1&#8202;N, 0.44 at 2&#8202;N) and the bed&#8217;s numerics; each simulated direction fitted like the measured one. RMS: "
                "the simulated shear against the measured shear less its fitted offset. Medians over directions.")
    skin = (
        f"<p>The skin&#8217;s sideways springs were sized from the slides: {sz['kt_N_m'] / 1e3:.2f}&#8202;kN/m, the measured "
        f"{f0(sz['give_um_N'])}&#8202;&#181;m/N at 1&#8202;N. The slides do not measure twist, so the hinge takes Lubkin&#8217;s "
        "ratio to Mindlin&#8217;s sideways stiffness, $k_\\theta/k_t=\\tfrac23(2-\\nu)a^2$, with the Hertz radius "
        f"$a$={sz['a_mm']:.2f}&#8202;mm at 0.90&#8202;MPa: {sz['kth_N_m_rad'] * 1e3:.2f}&#8202;mN&#8202;m/rad. The 812 pads of the UR5 rig "
        "ride on this skin, and the rig replays the 38 face slides at 1 and 2&#8202;N. The pads&#8217; base follows the measured "
        "travel, and the sphere is pushed with the measured normal force, both smoothed over 0.7&#8202;s.</p>"
        f"<p>At impratio 1000 the skin gives {f0(K['skin_ir1000_1N']['give_um_N'])}&#8202;&#181;m/N at 1&#8202;N, against the measured "
        f"{f0(K['skin_ir1000_1N']['give_measured_um_N'])}. It keeps {abs(K['skin_ir1000_1N']['s_back_N']):.2f}&#8202;N at the return, "
        f"as measured, and its shear is within {K['skin_ir1000_1N']['rms_N']:.2f}&#8202;N RMS of the measurement (Figure&#160;5c). "
        "Plain pads stick rigidly at this impratio, with no give, and the sign of their shear flips at every small reversal of "
        f"the travel ({K['pads_ir1000_1N']['rms_N']:.2f}&#8202;N RMS). At 2&#8202;N the springs sized at 1&#8202;N give "
        f"{f0(K['skin_ir1000_2N']['give_um_N'])} against {f0(K['skin_ir1000_2N']['give_measured_um_N'])}&#8202;&#181;m/N. A linear "
        "spring follows one normal force, and the measured give falls as $f_n^{-0.53}$.</p>"
        "<p>At impratio 100 the friction rows creep through the loops, and the shear stays below "
        f"{K['skin_ir100_1N']['mu']:.2f} of the normal force. At 10 it stays below 0.02 of it, with or without the skin "
        f"(Figure&#160;5a&#8211;b). On the CPU the skin costs {K['skin_ir1000_1N']['us']:.0f} against "
        f"{K['pads_ir1000_1N']['us']:.0f}&#8202;&#181;s per step of the rig.</p>"
        "<p>The servo plant&#8217;s RL skin scene has springs of 52.3&#8202;kN/m, from Mindlin at 5&#8202;N, and 0.26&#8202;kg of "
        "armature on each slide. Under the trainer&#8217;s numerics at impratio 100, "
        f"{g('cpu_skin_E10_ir100')['held']} of its 16 CPU grasps are held and {g('cpu_skin_E10_ir100')['nonfinite']} diverge. In "
        f"MuJoCo-Warp {g('mjw_skin_E10_ir100')['held_placements']} of 64 placements are held. With the springs re-sized from the "
        f"slides at 5&#8202;N, 9.8&#8202;kN/m, {g('cpu_skin_slides_E10_ir100')['held']} of 16 CPU grasps and "
        f"{g('mjw_skin_slides_E10_ir100')['held_placements']} of 64 MuJoCo-Warp placements are held. A world-step of the skin scene costs {g('mjw_skin_E10_ir100')['us']:.1f}&#8202;&#181;s against "
        f"{g('mjw_pads_E10_ir100')['us']:.1f} for the pads at 256 worlds. The tool slips on the skin at "
        f"{f0(g('mjw_skin_E10_ir100')['slip_um_s'])}&#8202;&#181;m/s at impratio 100, against "
        f"{f0(g('mjw_pads_E10_ir100')['slip_um_s'])}&#8202;&#181;m/s on the pads.</p>")
    S.append(("skin", "The sprung skin on the UR5 slides",
              skin + P.fig(5, MEDIA / "20261026-skin_slides.png",
                           "One direction of slide-000-1N, across the finger, replayed on the UR5 rig at impratio 10, 100 and 1000. "
                           "Grey: the measured shear less its offset. Orange: plain pads. Blue: pads on the sized skin. "
                           "Script: <code>scripts/ur5_shear_skin.py</code>.") + tab3))

    # 5. torsion
    tor = (
        "<p>The pads were twisted at the slides&#8217; friction coefficient, 0.37 at 1&#8202;N and 0.44 at 2.5 and 5&#8202;N, and "
        "at 0.90&#8202;MPa, at three face targets of cb2t&#8217;s twists. Under the 5&#8202;mm sphere they start to spin at "
        f"{tq['1'][0]:.2f}&#8211;{tq['1'][1]:.2f}&#8202;N&#8202;mm at 1&#8202;N and {tq['2.5'][0]:.2f}&#8211;{tq['2.5'][1]:.2f}&#8202;N&#8202;mm at "
        f"2.5&#8202;N. At 5&#8202;N they spin at {tq['5'][0]:.1f}&#8211;{tq['5'][1]:.1f}&#8202;N&#8202;mm (Figure&#160;6). "
        "Hertz&#8217;s patch under the same sphere gives $\\tau=\\mu f_n\\,\\tfrac{3\\pi}{16}a$: "
        f"{tq['1'][2]:.2f}, {tq['2.5'][2]:.2f} and {tq['5'][2]:.2f}&#8202;N&#8202;mm. On cb2t the HEX-E read a median torsion of "
        f"{tq['1'][3]['median_Nmm']:.2f}, {tq['2.5'][3]['median_Nmm']:.2f} and {tq['5'][3]['median_Nmm']:.2f}&#8202;N&#8202;mm, with single "
        f"values from {min(v[3]['min_Nmm'] for v in tq.values()):.1f} to {max(v[3]['max_Nmm'] for v in tq.values()):.1f}&#8202;N&#8202;mm.</p>"
        "<p>The HEX-E cannot resolve these torques. It reports torque in steps of 1&#8202;N&#8202;mm with 1&#8211;2&#8202;N&#8202;mm of noise. On "
        "the UR5 the face&#8217;s normal lies across the sensor&#8217;s axis, so a 2.5&#8202;N push bends the sensor with about "
        "230&#8202;N&#8202;mm. Its crosstalk of up to 5&#8202;% is then up to 11&#8202;N&#8202;mm, five times the torsion at 5&#8202;N. The torsion "
        "of the twists is half the difference between the way out and the way back, which removes the push&#8217;s moment but "
        "not the crosstalk&#8217;s change with pose. A clean measurement clamps the finger with the face&#8217;s normal along the "
        "sensor&#8217;s axis (Future work).</p>")
    S.append(("torsion", "Friction torque about the contact normal",
              tor + P.fig(6, MEDIA / "20261026-torsion.png",
                          "Friction torque about the normal under the 5&#8202;mm sphere. Blue: the 1&#8202;mm pads at spin onset, three "
                          "targets. Line: Hertz&#8217;s patch at the same friction and modulus. Orange: the torsion of cb2t&#8217;s twists "
                          "read by the HEX-E, median and range. Grey: the HEX-E&#8217;s torque noise. Script: "
                          "<code>scripts/ur5_shear_torsion.py</code>.", width=900)))

    # 6. limitations
    S.append(("limits", "Limitations", (
        "<p>The sphere is printed PLA, and the friction coefficient is that of PLA on the 85A pad, from slides at "
        "0.07&#8211;0.7&#176;/s. The robot&#8217;s pose is logged in steps of 10&#8202;&#181;m and the HEX-E&#8217;s force in steps of "
        "0.1&#8202;N. The HEX-E&#8217;s channel along the tool axis has 0.25&#8202;N of noise per sample. The shear includes an "
        f"in-plane part of {v1['b_over_fn']:.2f} of the normal force that the friction coefficient leaves out. The vector fit "
        "assumes isotropic friction and one spring, and its slider is at the limit for a small share of each slide. The fit removes an offset per "
        "direction, up to 0.4&#8202;N, so the shear left at the return is read from the fitted loop. The arm, the mount and the "
        "indenter&#8217;s stand are taken as rigid sideways. The spring and slider has one stiffness; the real loop slips "
        "gradually, so its first 50&#8202;&#181;m are 13&#8211;19&#8202;% stiffer than the fit. The D1&#8211;D8 replays put the UR5 "
        "pad&#8217;s shape on the bench hands. At 2.0&#8202;MPa they run at a 20&#8202;ms time constant. The divergence was traced "
        "on the CPU, in six grasps and two replays; the D1&#8211;D8 drops were not run in MuJoCo-Warp.</p>")))

    # 7. future work
    fw = [
        ("Friction torque on the sensor&#8217;s axis.", "Clamp the finger on the UR5 with the face&#8217;s normal along the force "
         "sensor&#8217;s axis. Press the 5&#8202;mm sphere at the face centre to 1, 2.5 and 5&#8202;N. At each force twist to 5, 10 "
         "and 15&#176; each way and back at 0.25&#176;/s, three times. The pads predict "
         f"{tq['1'][1]:.2f}, {tq['2.5'][1]:.2f} and {tq['5'][1]:.1f}&#8202;N&#8202;mm. A torque outside half to twice these falsifies the "
         "pads&#8217; friction torque at the measured friction coefficient."),
        ("Friction on the tool&#8217;s surface.", "Repeat four slides at 1 and 2&#8202;N on an indenter of the screwdriver&#8217;s "
         "handle material, or pull the bench cylinder between two printed tips (<code>scripts/tpu_tip_calibrate.py fit "
         "--pull</code>). A coefficient below 0.55 puts the tool outside the trainer&#8217;s friction range."),
        ("Impratio above 10 in the trainer.", "Run the trainer&#8217;s environment for one iteration of 2,048 worlds at impratio "
         "30 and 100 with its drop termination, and count the worlds that become non-finite before they terminate. Replay the "
         "D1&#8211;D8 drops in MuJoCo-Warp at the same settings, with the spin of the dropped tool logged. More non-finite worlds "
         "than at impratio 10, or any during the grip, keeps the trainer at 10. The decision waits on the owner."),
        ("Task outcomes against creep.", "Replay the D1&#8211;D8 plans and the readiness commands at impratio 10 with and without "
         "10 noslip iterations on the CPU, and compare the tool&#8217;s turn and the held count. A change of more than 5&#176; in the "
         "median turn or of more than two held replays shows that the trainer&#8217;s creep changes the task."),
        ("Creep against the bench holds.", "Read the tool&#8217;s drift over the hold after the turn from the AprilTag tracks of "
         "the 177 tracked bench runs (<code>logs/tracker/*_track.csv</code>, about 28 frames per second for 3&#8211;6&#8202;s), "
         "with the tracker&#8217;s noise from its still frames. Compare it with the held D1&#8211;D8 replays&#8217; "
         f"{f0(r10['creep_um_s'])}&#8202;&#181;m/s at impratio 10. A measured drift of a tenth of that or less, above the noise, "
         "shows a creep that the print does not have."),
        ("A give that falls with load.", "Replay the 2&#8202;N slides with springs sized at 2&#8202;N, 5.5&#8202;kN/m. If the "
         "measured loops at both loads match within 0.15&#8202;N RMS only with a stiffness that grows as $f_n^{0.53}$, a linear "
         "skin cannot represent the contact. A stiffness scheduled on the normal force is then needed."),
        ("The skin under the trainer at impratio 100.", "Repeat the scripted grasp of the RL skin scene on MuJoCo-Warp with "
         "2,048 worlds and the trainer&#8217;s placement noise. Log the springs&#8217; deflection and the tool&#8217;s slip over "
         "the hold. Train the skin arm only if all 64 placements are held and the deflection exceeds the slip. Otherwise the "
         "skin is trained without its give, as in the arms of 2026-10-08 and 10-11."),
    ]
    S.append(("future", "Future work", "<ul class=\"open\">" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in fw) + "</ul>"))
    return S, lede, summary


def build():
    N = json.loads((DATA / "20261026-numbers.json").read_text())
    import ur5_shear_skin as SK
    N["skin_sizing"] = SK.sizing(1.0)
    S, lede, summary = sections(N)
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    byline = ("UR5 sessions cb2t (tip 2) and cb1r (tip 1) of 2026-10-25 with the OptoForce HEX-E, sphere2 and rod1 of 10-22/23; "
              "MuJoCo 3.6 on one CPU core and MuJoCo-Warp on the GPU, 2026-10-26. Scripts <code>scripts/ur5_shear_slides.py</code>, "
              "<code>ur5_shear_impratio.py</code>, <code>ur5_shear_skin.py</code>, <code>ur5_shear_torsion.py</code>, "
              "<code>ur5_shear_figures.py</code>; page by <code>scripts/ur5_shear_page.py</code>; rows in "
              + link(DATA, "data/") + ". The sessions are described on " + link(L["cal"], "the calibration page")
              + "; the contact model on " + link(L["contact"], "the contact overview") + ".")
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "Shear, friction and impratio of the printed fingertip",
                 "H1": "Shear, friction coefficient and impratio of the printed fingertip against the UR5 slides",
                 "EYEBROW": "Fingertip contact model &#183; 2026-10-26", "LEDE": lede, "BYLINE": byline, "FILM": "",
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = P.TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="contact", summary=summary, date="2026-10-26")
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    build()
