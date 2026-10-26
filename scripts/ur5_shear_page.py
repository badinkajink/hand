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

    lede = (
        f"On 87 UR5 slides of one printed fingertip, the friction coefficient of the 85A pad on a printed PLA sphere is "
        f"{b1['mu']['median']:.2f} at 1 and 2&#8202;N. Before slip the pad moves {f0(b1['give_um_N']['median'])}&#8202;&#181;m per "
        f"newton of shear at 1&#8202;N and {f0(b2['give_um_N']['median'])}&#8202;&#181;m/N at 2&#8202;N. In the finite-element model "
        "it moves 48&#8211;177&#8202;&#181;m/N under the 12.5&#8202;mm rod. Under the policy trainer&#8217;s solver, impratio 100 lowers "
        f"the 1&#8202;mm pads&#8217; creep from {f0(r10['creep_um_s'])} to {f0(r100['creep_um_s'])}&#8202;&#181;m/s in the D1&#8211;D8 "
        f"replays. Every replay that drops the tool then diverges, as do {g('cpu_pads_E10_ir100')['nonfinite']} of 16 CPU grasps. "
        f"In MuJoCo-Warp {g('mjw_pads_E10_ir100')['held_placements']} of 64 placements are held at the same cost. Pads on a skin "
        f"sized from the slides match the measured shear within {K['skin_ir1000_1N']['rms_N']:.2f}&#8202;N RMS at impratio 1000. "
        "At impratio 100 creep flattens their loops. The pads&#8217; friction torque, "
        f"{tq['1'][0]:.1f}&#8211;{tq['5'][1]:.1f}&#8202;N&#8202;mm at 1&#8211;5&#8202;N, is below the HEX-E&#8217;s resolution.")
    summary = (
        f"UR5 slides of the printed fingertip: friction coefficient {b1['mu']['median']:.2f} (PLA sphere on 85A, 1 and 2 N), "
        f"give before slip {f0(b1['give_um_N']['median'])} and {f0(b2['give_um_N']['median'])} um/N at 1 and 2 N; under the "
        f"trainer's solver impratio 100 cuts the pads' creep {r10['creep_um_s'] / r100['creep_um_s']:.1f}x but replays that "
        f"drop the tool diverge and 7 of 16 CPU grasps hold (MuJoCo-Warp 63/64); a slide-sized skin matches the loops only at "
        "impratio 1000; the friction torque is below the HEX-E's resolution.")

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
        ("Friction coefficient $\\mu$", "The slider&#8217;s limit over the normal force, from the same fit. The scenes are run "
         "with 1."),
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
        ("Diverged", "MuJoCo found a non-finite or huge acceleration and reset the state (warning BADQACC), or a MuJoCo-Warp "
         "world became non-finite."),
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
         "Friction coefficient", "Shear left at the return (N)"],
        [[f"Tip {t}, {F}&#8202;N", v["n"], f"{v['f_n_N']['median']:.2f}",
          f"{v['give_um_N']['median']:.0f} ({v['give_um_N']['p25']:.0f}&#8211;{v['give_um_N']['p75']:.0f})",
          f"{v['give_load50_um_N']['median']:.0f}",
          f"{v['mu']['median']:.2f} ({v['mu']['p25']:.2f}&#8211;{v['mu']['p75']:.2f})", f"{v['z_back_N']['median']:+.2f}"]
         for t, F, v in ((2, "0.5", S1["by_level"]["0.5"]), (2, "1", b1), (2, "1.5", S1["by_level"]["1.5"]), (2, "2", b2),
                         (2, "3", S1["by_level"]["3"]), (1, "1", c1), (1, "2", c2))],
        num_cols=(1, 2, 3, 4, 5, 6), n=1,
        caption="The slides of tip 2 (cb2t, 2026-10-25) and tip 1 (cb1r), by press level: medians and quartiles over the "
                "directions. The normal force is the mean during the slide; the press stopped at the level.")
    slides = (
        f"<p>The pad&#8217;s friction coefficient on the printed PLA sphere is {b1['mu']['median']:.2f} at 1&#8202;N and "
        f"{b2['mu']['median']:.2f} at 2&#8202;N on tip 2. Over all {S1['mu_all']['n']} directions it grows only as "
        f"$f_n^{{{S1['mu_power_law']['exponent']:.2f}}}$, and it lies between "
        f"{min(v['mu']['median'] for v in S1['by_direction'].values()):.2f} and "
        f"{max(v['mu']['median'] for v in S1['by_direction'].values()):.2f} in each of the eight directions. Tip 1 gives "
        f"{c1['mu']['median']:.2f} and {c2['mu']['median']:.2f}. The scenes are run with 1. In training the friction is "
        "randomised over 0.55&#8211;1.15 of it (Figure&#160;1c).</p>"
        f"<p>Before slip the pad moves {f0(b1['give_um_N']['median'])}&#8202;&#181;m per newton of shear at 1&#8202;N and "
        f"{f0(b2['give_um_N']['median'])}&#8202;&#181;m/N at 2&#8202;N (Table&#160;1). The give falls as "
        f"$f_n^{{{S1['give_power_law']['exponent']:.2f}}}$ with the normal force. Mindlin&#8217;s compliance of an elastic "
        "half-space under the same sphere falls as $f_n^{-1/3}$. At the modulus fitted to the rod presses, 2.0&#8202;MPa, it is "
        "246 and 195&#8202;&#181;m/N at 1 and 2&#8202;N (Figure&#160;1b). The FE pad moves 48&#8211;177&#8202;&#181;m/N sideways "
        "under the 12.5&#8202;mm rod, a larger contact. A straight line through the first 50&#8202;&#181;m of travel gives "
        f"{f0(b1['give_load50_um_N']['median'])} and {f0(b2['give_load50_um_N']['median'])}&#8202;&#181;m/N, since the fitted "
        "slider slips at once where the real contact slips gradually.</p>"
        f"<p>The contact slips both ways in {sum(v['slip_both'] for v in S1['by_level'].values())} of "
        f"{S1['mu_all']['n']} directions. Each turn of 0.2&#8202;mm reaches the friction limit, and each return slides back. At "
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
                             "(c) Friction coefficient, with the scenes&#8217; 1 and the trainer&#8217;s randomisation range. "
                             "Script: <code>scripts/ur5_shear_slides.py</code>.") + tab1))

    # 3. impratio
    tab2 = P.table(
        ["Pads, impratio", "UR5 rig creep, sphere (&#181;m/s)", "D1&#8211;D8 held / diverged", "Creep of held (&#181;m/s)",
         "Readiness held", "Grasp CPU held / diverged", "Grasp MuJoCo-Warp held", "Slip (&#181;m/s)", "Cap (%)", "&#181;s per step"],
        [[f"10&#8202;MPa, {x}", f"{C[f'sphere2_E10_rl_ir{x}']['median_um_s']:.1f}",
          f"{R[f'd1d8_E10_ir{x}']['held']} / {R[f'd1d8_E10_ir{x}']['diverged']}", f"{R[f'd1d8_E10_ir{x}']['creep_um_s']:.0f}",
          f"{R[f'icem_E10_ir{x}']['held']} of 6",
          (f"{G[f'cpu_pads_E10_ir{x}']['held']} / {G[f'cpu_pads_E10_ir{x}']['nonfinite']}" if f'cpu_pads_E10_ir{x}' in G else "&#8211;"),
          (f"{G[f'mjw_pads_E10_ir{x}']['held_placements']} of 64" if f'mjw_pads_E10_ir{x}' in G else "&#8211;"),
          (f"{G[f'mjw_pads_E10_ir{x}']['slip_um_s']:.0f}" if f'mjw_pads_E10_ir{x}' in G else "&#8211;"),
          f"{100 * R[f'd1d8_E10_ir{x}']['cap_share_max']:.1f}", f"{R[f'd1d8_E10_ir{x}']['us_held']:.0f}"]
         for x in (10, 100, 300)]
        + [[f"2.0&#8202;MPa, {x}", f"{C[f'rod1_E1.9974_rl_ir{x}']['median_um_s']:.1f} (rod)",
            f"{R[f'd1d8_E2_ir{x}']['held']} / {R[f'd1d8_E2_ir{x}']['diverged']}", f"{R[f'd1d8_E2_ir{x}']['creep_um_s']:.0f}",
            f"{R[f'icem_E2_ir{x}']['held']} of 6", "&#8211;", "&#8211;", "&#8211;",
            f"{100 * R[f'd1d8_E2_ir{x}']['cap_share_max']:.1f}", f"{R[f'd1d8_E2_ir{x}']['us_held']:.0f}"]
           for x in (10, 100, 300)],
        num_cols=(1, 2, 3, 4, 5, 6, 7, 8, 9), n=2,
        caption="Impratio under the trainer&#8217;s numerics. UR5 rig: the 812 pads under the sphere presses&#8217; forces at 10&#8202;MPa "
                "(10&#8202;MPa rows) and under the rod presses&#8217; at 2.0&#8202;MPa. D1&#8211;D8: 24 replays per setting on the UR5 "
                "pad&#8217;s shape. Readiness: the two logged pinned-iCEM runs from three placements. Grasp: the scripted close and "
                "lift on the servo plant&#8217;s pad scene of the RL runs, 16 placements on the CPU and 64 in MuJoCo-Warp. Cap: the "
                "largest share of steps at the iteration cap over the D1&#8211;D8 replays. &#181;s per step: the held D1&#8211;D8 replays "
                "on one core.")
    imp = (
        f"<p>Under the trainer&#8217;s numerics the pads&#8217; creep falls as $1/\\kappa$ in the UR5 rig. At the sphere presses&#8217; "
        f"0.90&#8202;MPa it is {cr(sph, 10):.0f}, {cr(sph, 100):.0f} and {cr(sph, 300):.0f}&#8202;&#181;m/s at impratio 10, 100 and 300. "
        f"Under the rod presses at 2.0&#8202;MPa it is {cr(rod, 10):.0f}, {cr(rod, 100):.0f} and {cr(rod, 300):.1f}&#8202;&#181;m/s. The "
        "bed&#8217;s numerics, 1&#8202;ms and 100 iterations, give the same speeds, and no step reaches the iteration cap "
        "(Figure&#160;2a).</p>"
        "<p>On the hand the replays decide. The D1&#8211;D8 plans were replayed open loop from three placements on 1&#8202;mm pads of "
        f"the UR5 pad&#8217;s shape. At 10&#8202;MPa they hold the tool in {r10['held']}, {r100['held']} and {r300['held']} of 24 "
        f"at impratio 10, 100 and 300, on hands {', '.join(r10['hands_held'])}. The creep of the held tool falls from "
        f"{f0(r10['creep_um_s'])} to {f0(r100['creep_um_s'])} and {f0(r300['creep_um_s'])}&#8202;&#181;m/s, three and nine times less. "
        "The thumb presses with 1.8&#8211;1.9&#8202;N and the middle fingertip with 1.6&#8211;1.9&#8202;N at all three. At impratio 10 the other "
        f"{r10['dropped']} replays drop the tool. At 100 and 300 those replays diverge 0.3&#8211;1&#8202;s after the drop, and "
        f"{r300['diverged_before_2s']} at 300 diverge during the grip (Figure&#160;2b). MuJoCo then resets the state, and a replay loop "
        "that watches only the simulated time never ends. The loops of these scripts stop at the reset.</p>"
        f"<p>At 2.0&#8202;MPa {q10['held']}, {q100['held']} and {q300['held']} of the 24 replays are held. The creep is "
        f"{f0(q10['creep_um_s'])}, {f0(q100['creep_um_s'])} and {f0(q300['creep_um_s'])}&#8202;&#181;m/s. A pad this soft needs a "
        "longer time constant: at 10&#8202;ms its impedance $d_0$ falls to 0.05, the floor of the pad builder. The D1&#8211;D8 "
        "replays at 2.0&#8202;MPa therefore run at 20&#8202;ms. On the RL scene, rescaled to 2.0&#8202;MPa at 10&#8202;ms "
        "($d_0=0.055$), every CPU grasp and every MuJoCo-Warp world diverged. At 15&#8202;ms "
        f"{g('cpu_pads_E2_ir10')['held']} and {g('cpu_pads_E2_ir100')['held']} of 16 CPU grasps are held at impratio 10 and 100. "
        "In MuJoCo-Warp all 64 placements are held at both. The readiness commands, two logged pinned-iCEM runs from three placements, hold 6, 3 and 1 of 6 at "
        "10&#8202;MPa, and 6, 6 and 1 at 2.0&#8202;MPa.</p>"
        "<p>The trainer steps the scripted grasp in MuJoCo-Warp. On the servo plant&#8217;s pad scene of the RL runs, "
        f"{g('cpu_pads_E10_ir10')['held']} of 16 CPU grasps are held at impratio 10 and {g('cpu_pads_E10_ir100')['held']} at 100. "
        f"At 100, {g('cpu_pads_E10_ir100')['nonfinite']} diverge, and {100 * g('cpu_pads_E10_ir100')['cap']:.1f}&#8202;% of steps reach "
        f"the iteration cap. In MuJoCo-Warp, in float32, {g('mjw_pads_E10_ir10')['held_placements']} and "
        f"{g('mjw_pads_E10_ir100')['held_placements']} of 64 placements are held. The tool slips on the fingertips at "
        f"{f0(g('mjw_pads_E10_ir10')['slip_um_s'])} and {f0(g('mjw_pads_E10_ir100')['slip_um_s'])}&#8202;&#181;m/s. A world-step "
        f"costs {g('mjw_pads_E10_ir10')['us']:.1f} and {g('mjw_pads_E10_ir100')['us']:.1f}&#8202;&#181;s at 256 worlds (Table&#160;2). "
        "The film shows D1, which keeps the tool at both impratios, and D2, whose replay at 100 diverges after the drop.</p>")
    S.append(("impratio", "Impratio under the trainer's solver",
              imp + P.fig(2, MEDIA / "20261026-impratio.png",
                          "Impratio under the trainer&#8217;s numerics. (a) Creep against impratio. Orange and blue: the UR5 rig "
                          "under the measured forces of the 5&#8202;mm sphere and the 12.5&#8202;mm rod, open marks at 10&#8202;MPa. Green: "
                          "the held D1&#8211;D8 replays. Black: the scripted grasp in MuJoCo-Warp. Grey: the slope of $1/\\kappa$. (b) The 24 D1&#8211;D8 replays per pad modulus and impratio: held, dropped and "
                          "diverged. Scripts: <code>scripts/ur5_shear_impratio.py</code>, <code>scripts/ur5_shear_figures.py</code>.")
              + P.film(3, MEDIA / "20261026-impratio_replays.mp4", MEDIA / "20261026-impratio_replays.png",
                       "D1 (top) and D2 (bottom) replayed open loop at the trainer&#8217;s numerics on 10&#8202;MPa pads, impratio 10 "
                       "(left) and 100 (right), placement 0, at real speed. D1 keeps the tool at both. D2 drops it at both, and at "
                       "100 the simulation diverges 0.5&#8202;s after the drop.") + tab2))

    # 4. skin
    sz = N["skin_sizing"]
    tab3 = P.table(
        ["Model, impratio", "Give (&#181;m/N), 1 / 2&#8202;N", "Friction coefficient, 1 / 2&#8202;N", "RMS to measured (N), 1 / 2&#8202;N",
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
        num_cols=(1, 2, 3, 4, 5), n=3,
        caption="The 38 face slides at 1 and 2&#8202;N (304 directions) replayed on the UR5 rig&#8217;s pads, plain and on the sized "
                "skin, at friction 0.35 and the bed&#8217;s numerics; each simulated direction fitted like the measured one. RMS: "
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
        f"as measured, and its shear is within {K['skin_ir1000_1N']['rms_N']:.2f}&#8202;N RMS of the measurement (Figure&#160;4c). "
        "Plain pads stick rigidly at this impratio, with no give, and the sign of their shear flips at every small reversal of "
        f"the travel ({K['pads_ir1000_1N']['rms_N']:.2f}&#8202;N RMS). At 2&#8202;N the springs sized at 1&#8202;N give "
        f"{f0(K['skin_ir1000_2N']['give_um_N'])} against {f0(K['skin_ir1000_2N']['give_measured_um_N'])}&#8202;&#181;m/N. A linear "
        "spring follows one normal force, and the measured give falls as $f_n^{-0.53}$.</p>"
        "<p>At impratio 100 the friction rows creep through the loops, and the shear stays below "
        f"{K['skin_ir100_1N']['mu']:.2f} of the normal force. At 10 it stays below 0.02 of it, with or without the skin "
        f"(Figure&#160;4a&#8211;b). On the CPU the skin costs {K['skin_ir1000_1N']['us']:.0f} against "
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
              skin + P.fig(4, MEDIA / "20261026-skin_slides.png",
                           "One direction of slide-000-1N, across the finger, replayed on the UR5 rig at impratio 10, 100 and 1000. "
                           "Grey: the measured shear less its offset. Orange: plain pads. Blue: pads on the sized skin. "
                           "Script: <code>scripts/ur5_shear_skin.py</code>.") + tab3))

    # 5. torsion
    tor = (
        "<p>The pads were twisted at the slides&#8217; friction coefficient and 0.90&#8202;MPa, at three face targets of "
        "cb2t&#8217;s twists. Under the 5&#8202;mm sphere they start to spin at "
        f"{tq['1'][0]:.2f}&#8211;{tq['1'][1]:.2f}&#8202;N&#8202;mm at 1&#8202;N and {tq['2.5'][0]:.2f}&#8211;{tq['2.5'][1]:.2f}&#8202;N&#8202;mm at "
        f"2.5&#8202;N. At 5&#8202;N they spin at {tq['5'][0]:.1f}&#8211;{tq['5'][1]:.1f}&#8202;N&#8202;mm (Figure&#160;5). "
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
              tor + P.fig(5, MEDIA / "20261026-torsion.png",
                          "Friction torque about the normal under the 5&#8202;mm sphere. Blue: the 1&#8202;mm pads at spin onset, three "
                          "targets. Line: Hertz&#8217;s patch at the same friction and modulus. Orange: the torsion of cb2t&#8217;s twists "
                          "read by the HEX-E, median and range. Grey: the HEX-E&#8217;s torque noise. Script: "
                          "<code>scripts/ur5_shear_torsion.py</code>.", width=900)))

    # 6. limitations
    S.append(("limits", "Limitations", (
        "<p>The sphere is printed PLA, and the friction coefficient is that of PLA on the 85A pad, from slides at "
        "0.07&#8211;0.7&#176;/s. The robot&#8217;s pose is logged in steps of 10&#8202;&#181;m and the HEX-E&#8217;s force in steps of "
        "0.1&#8202;N. The HEX-E&#8217;s channel along the tool axis has 0.25&#8202;N of noise per sample. The fit removes an offset per "
        "direction, up to 0.4&#8202;N, so the shear left at the return is read from the fitted loop. The arm, the mount and the "
        "indenter&#8217;s stand are taken as rigid sideways. The spring and slider has one stiffness; the real loop slips "
        "gradually, so its first 50&#8202;&#181;m are 13&#8211;19&#8202;% stiffer than the fit. The D1&#8211;D8 replays put the UR5 "
        "pad&#8217;s shape on the bench hands. At 2.0&#8202;MPa they run at a 20&#8202;ms time constant.</p>")))

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
        ("Impratio 100 in the trainer.", "Run the trainer&#8217;s environment for one iteration of 2,048 worlds at impratio 100 "
         "with its drop termination, and count the worlds that become non-finite before they terminate. More than the "
         "impratio-10 run&#8217;s count, or any during the grip, keeps the trainer at 10. The decision waits on the owner."),
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
