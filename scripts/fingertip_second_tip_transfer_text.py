"""The prose of the second-fingertip page (scripts/fingertip_second_tip_transfer_page.py), from the two rounds' results."""
from __future__ import annotations

import numpy as np

ROOT_PAGES = {
    "results": "docs/experiments/20261022-ur5_tip_calibration/20261022-ur5_tip_ground_truth_and_predictors.html",
    "protocol": "docs/experiments/20261022-ur5_tip_calibration/20261022-ur5_tip_calibration_protocol.html",
    "estimation": "docs/experiments/20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html",
    "drift": "docs/experiments/20261022-ur5_tip_calibration/20261023-ur5_tip_board_drift.html",
    "status": "docs/experiments/20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html",
}


def num(v, d=2):
    """A signed number with a true minus sign."""
    s = f"{v:.{d}f}"
    return s.replace("-", "&#8722;")


def pt(p, d=2, keep=(0, 1, 2)):
    return ", ".join(num(p[i], d) for i in keep)


def poles(ps):
    """The pattern as mount end (-y, +y), centre, tip end (-y, +y)."""
    s = ["+" if v > 0 else "&#8722;" for v in ps]
    return f"{s[0]} {s[1]} / {s[2]} / {s[3]} {s[4]}"


def sections(R1, R2, robot, F, P, link, NAME):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    pg = {k: root / v for k, v in ROOT_PAGES.items()}
    esc, fig, table = P.esc, P.fig, P.table
    own, rp, ra = R2["rest"]["own"], R2["rest"]["robot_poles_refit"], R2["rest"]["robot_as_is"]
    pv = R2["preview"]
    A, Dd = pv["sets"]["A"]["scores"], pv["sets"]["D"]["scores"]
    H, Sm = R2["holds"], R2["summary"]
    inp = Sm["inputs"]
    s2 = R2["sphere2"]
    ph, pc, gr, mr, rc = (Sm[k] for k in ("physical", "physical+gp", "gp raw", "mlp raw", "robot cal"))
    shift = np.array([np.subtract(h["est"]["physical+gp"]["p"], h["est"]["physical"]["p"]) for h in H])
    ms = shift.mean(0)
    dF = np.array([h["est"]["physical+gp"]["f_n"] - h["est"]["physical"]["f_n"] for h in H])
    ratio = np.array([h["est"]["robot cal"]["f_n"] / h["est"]["physical"]["f_n"] for h in H])
    gaps = np.array([H[i + 1]["t0_s"] - H[i]["t1_s"] for i in range(len(H) - 1)])
    lens = np.array([h["s"] for h in H])
    st, h1 = R1["step"], R1["holds"][0]
    turn = 2 * np.degrees(np.arcsin(st["uniform"]["g_norm_uT"] / (2 * 44.3)))
    robot_rest_mT = float(np.linalg.norm(np.array(robot.b_rest_uT)) / 1e3)

    lede = (
        "The calibration learned on the UR5 fingertip in session sphere2 was applied to a second fingertip on the "
        "workstation. On the second pad the two magnets at the mount end point the other way. Its rest reading alone gave "
        f"that pole pattern, fitted at {own['rms_mT']:.2f} mT RMS against {own['rms_next_mT']:.2f} mT for the next "
        "pattern. The physical model rebuilt from this fit placed six presses at the face centre within "
        f"{ph['spread_max_mm']:.2f} mm of their mean. That mean lies {ph['mean_from_ref_mm']:.1f} mm from the reference "
        f"point, at normal forces of {ph['f_min']:.1f} to {ph['f_max']:.1f} N. Sphere2's Gaussian process (GP) and "
        "multilayer perceptron (MLP) on the raw board change did not transfer. The GP gave one answer for all six "
        f"presses, and the MLP placed them {mr['mean_from_ref_mm']:.1f} mm away, at the mount end. Sphere2's GP "
        f"correction of the physical estimate moved the points {np.linalg.norm(ms):.1f} mm toward the mount end, to "
        f"{pc['mean_from_ref_mm']:.1f} mm from the reference point. No force was measured, and the presses were aimed "
        "by eye.")
    summary = (
        f"The physical model rebuilt from the second fingertip's rest fit (two magnets reversed) placed six centre "
        f"presses within {ph['spread_max_mm']:.2f} mm of their mean, {ph['mean_from_ref_mm']:.1f} mm from the reference "
        f"point, at {ph['f_min']:.1f} to {ph['f_max']:.1f} N. The UR5 fingertip's raw GP and MLP did not transfer (one "
        f"answer for all presses; {mr['mean_from_ref_mm']:.1f} mm), and its GP correction moved the points "
        f"{np.linalg.norm(ms):.1f} mm toward the mount end. No force reference.")

    film_html = P.film(
        1, F["film"], F["poster"],
        "The contact viewer replays the second round, from 12:27:38 to 12:28:15 of the recording, in real time. It was "
        "calibrated on the recording's first 2 s at rest, as it is live, and found the pad's poles itself. Arrow: the "
        "estimated force; colour: the sphere pads' pressure; deformation from the FE model, drawn 20 times. The panel's "
        "clock shows the time of the replay. Made by " + P.code("scripts/fingertip_second_tip_film.py") + ".")

    S = []
    S.append(("terms", "Terms and metrics", f"""<dl class="terms">
<dt>Pad frame</dt><dd>$x$ along the normal of the palmar face, toward the palm; $y$ across the finger; $z$ from the mount
end toward the tip end; in mm, as on the {link(pg['results'], 'UR5 results page')}.</dd>
<dt>Pole pattern</dt><dd>The sign of each of the five magnets, + with its north pole toward the palm. It is written as
the mount end's pair ($-y$, $+y$), the centre magnet, then the tip end's pair.</dd>
<dt>Strength</dt><dd>The magnets' moment as a fraction of a 3 &#215; 2 mm N52 magnet's.</dd>
<dt>Rest fit</dt><dd>The field model fitted to 2 s of the board's reading with the pad untouched. It searches all 32 pole
patterns and fits the strength and the pad's offset on the board ({link(pg['estimation'], 'contact estimation')}).
Its residual is the RMS over the 15 field components, in mT; the next pattern's is the best fit with any other pole
pattern.</dd>
<dt>Board change</dt><dd>The mean reading over a hold less the rest just before its press: 15 values in &#181;T, after
each chip's temperature correction.</dd>
<dt>Hold</dt><dd>The middle half of a stretch in which the board change exceeds 30 &#181;T for at least 1 s.</dd>
<dt>Physical model</dt><dd>The viewer's estimator on a pad's rest fit, untrained. It maps a board change to a contact
point $\\mathbf p_{{\\mathrm{{ph}}}}$ and a normal force $f_{{\\mathrm{{ph}}}}$ through the FE model's magnet motions.</dd>
<dt>GP correction</dt><dd>A Gaussian process (GP) trained on sphere2's {s2['holds']} press holds ({s2['presses']}
presses). Its input is the physical estimate; its output is the HEX-E's ground truth less that estimate. The corrected
estimate is $$\\hat{{\\mathbf p}} = \\Pi\\big(\\mathbf p_{{\\mathrm{{ph}}}} + \\Delta\\mathbf p(\\mathbf p_{{\\mathrm{{ph}}}},
f_{{\\mathrm{{ph}}}})\\big),\\qquad \\hat f_n = f_{{\\mathrm{{ph}}}} + \\Delta f(\\mathbf p_{{\\mathrm{{ph}}}},
f_{{\\mathrm{{ph}}}}),$$ where $\\Pi$ moves a point onto the pad's surface. Its inputs and outputs are in the pad's frame,
so the correction can be applied to any pad whose physical model has been rebuilt.</dd>
<dt>GP and MLP on the board change</dt><dd>Sphere2's predictors from the 15 components of the board change to the
contact point and normal force: a GP and an ensemble of five multilayer perceptrons (MLP), trained on the same
holds.</dd>
<dt>Robot calibration as it stands</dt><dd>The viewer's estimator on sphere2's rest fit, applied to the second pad
unchanged.</dd>
<dt>Reference point</dt><dd>The UR5 protocol's reference press, pad point 788 at ({pt([8.475, 0.502, 10.36])}) mm, at
the centre of the palmar face ({link(pg['protocol'], 'protocol page')}).</dd>
<dt>Distance from the reference point, spread</dt><dd>The distance in mm from the reference point to the mean of a
predictor's six contact points; the spread is the largest distance of one of the six from that mean.</dd>
<dt>Location and force errors</dt><dd>In the simulation: the distance in mm from the estimated to the true contact
point, tangential to the pad, and $|\\hat f_n - f_n|$ in N; medians over the held-out holds.</dd>
</dl>"""))

    t1 = table(
        ["Quantity", "Robot fingertip (sphere2)", "Second fingertip"],
        [["Board", "9c:d0", "d9:94"],
         ["Where and when", "Lab UR5, 2026-10-22", "Workstation's desk, 2026-10-23"],
         ["Pole pattern", poles(robot.poles), poles(own["poles"])],
         ["Strength (of N52)", f"{robot.k:.3f}", f"{own['k']:.3f}"],
         ["Pad offset on the board, x, y, z (mm)", pt([robot.dx, robot.dy, robot.dz]), pt(own["offset_mm"])],
         ["Rest fit (mT RMS)", f"{robot.rms_mT:.2f}", f"{own['rms_mT']:.2f}"],
         ["Next pattern (mT RMS)", f"{robot.rms_next_mT:.2f}", f"{own['rms_next_mT']:.2f}"],
         ["Field at rest, norm of the 15 components (mT)", f"{robot_rest_mT:.1f}", f"{own['rest_norm_mT']:.1f}"]],
        num_cols=(1, 2), caption="The two fingertips' rest fits.", n=1)
    t2 = table(
        ["Calibration applied to the second pad's rest", "Pole pattern", "Strength (of N52)", "Residual (mT RMS)"],
        [["Its own rest fit", poles(own["poles"]), f"{own['k']:.3f}", f"{own['rms_mT']:.2f}"],
         ["The robot pad's poles; strength and offsets refitted", poles(rp["poles"]), f"{rp['k']:.3f}", f"{rp['rms_mT']:.2f}"],
         ["The robot calibration as it stands", poles(ra["poles"]), f"{ra['k']:.3f}", f"{ra['rms_mT']:.2f}"]],
        num_cols=(2, 3), caption="The second pad's field at rest against three calibrations.", n=2)
    S.append(("pads", "The two fingertips and their rest fields", (
        "<p>Both fingertips are concept b's pad on DIP link b, over a ReSkin 5X board read by a QT Py. Each pad has five "
        f"3 &#215; 2 mm magnets of the same batch, at {robot.k:.2f} to {own['k']:.2f} of an N52 magnet's moment. The "
        f"robot fingertip, board 9c:d0, was calibrated in the UR5 session sphere2 on 2026-10-22 "
        f"({link(pg['results'], 'UR5 ground truth and predictors')}). The second fingertip, board d9:94, has stayed on "
        "the workstation's desk under the live viewer. Its rest fit puts the two magnets at the mount end with their "
        "south pole toward the palm, and the other three with their north pole (Figure 2).</p>"
        f"<p>The second pad's rest fits this pattern at {own['rms_mT']:.2f} mT RMS, against {own['rms_next_mT']:.2f} mT "
        f"for the next pattern (Table 1). Its strength is {own['k']:.3f}, against {robot.k:.3f} for the robot pad. "
        f"The robot calibration as it stands misses the second pad's rest by {ra['rms_mT']:.2f} mT RMS. With the robot "
        f"pad's poles kept and the strength and offsets refitted, the miss is {rp['rms_mT']:.2f} mT, at a strength of "
        f"{rp['k']:.2f} (Table 2).</p>"
        + t1 + t2
        + fig(2, F["poles"], "The two pads seen from the palm: magnets (red, north toward the palm; blue, south), the "
              "chips U1 to U5 (squares) and the reference point (star). Grey: the pad's surface points that face the "
              "palm. The chips' positions are the board's nominal ones.", width=1000)
    )))

    rows3 = [["Physical model, rebuilt for each pad, untrained", "physical", "physical"],
             ["GP on the board change, trained on set A", "gp raw (trained on A)", "gp raw (trained on A)"],
             ["Physical model with a GP correction trained on set A", "physical+gp (trained on A)", "physical+gp (trained on A)"],
             ["GP on the board change, retrained on each set", "gp raw (retrained on A)", "gp raw (retrained on D)"]]
    t3 = table(["Predictor", "Set A, robot pad (mm, N)", "Set D, second pad (mm, N)"],
               [[r[0], f"{A[r[1]]['loc_mm']:.2f}, {A[r[1]]['force_N']:.3f}", f"{Dd[r[2]]['loc_mm']:.2f}, "
                 f"{Dd[r[2]]['force_N']:.3f}"] for r in rows3],
               num_cols=(1, 2), n=3, caption="Median location and force errors on the held-out holds, simulated board "
                                             "changes on sphere2's mechanics.")
    S.append(("sim", "Transfer in simulation", (
        "<p>Before the presses, the transfer was simulated on sphere2's mechanics, as on the UR5 results page (its "
        f"Table 8). Each of sphere2's {s2['holds']} press holds keeps its true contact point and force. Its board change "
        "is synthesized for each pad from the FE model's magnet motions and that pad's field model. Both pads get the "
        f"same two errors of the mechanics. Their compliance is {pv['c_true']:.2f} times the FE's, and each contact is "
        f"shifted by ({pt(pv['delta_mm'])}) mm, sphere2's mean location error at 3 N and over. Noise of "
        f"{pv['noise_uT']:.0f} &#181;T is added to each component. Set A is the robot pad, set D the second pad. The "
        f"predictors are trained on set A's {pv['n_train']} training holds and scored on the {pv['n_test']} held-out "
        "holds of each set (Table 3).</p>"
        f"<p>The rebuilt physical model places the second pad's contacts within {Dd['physical']['loc_mm']:.2f} mm, as it "
        f"places the robot pad's within {A['physical']['loc_mm']:.2f} mm. The GP on the raw board change goes from "
        f"{A['gp raw (trained on A)']['loc_mm']:.2f} mm on the robot pad to {Dd['gp raw (trained on A)']['loc_mm']:.2f} mm "
        f"on the second pad. The GP correction of the physical estimate goes from "
        f"{A['physical+gp (trained on A)']['loc_mm']:.2f} to {Dd['physical+gp (trained on A)']['loc_mm']:.2f} mm. A GP "
        f"retrained on the second pad's own synthesized presses gives {Dd['gp raw (retrained on D)']['loc_mm']:.2f} "
        "mm.</p>" + t3
    )))

    S.append(("round1", "Presses with the fingertip loose on the desk", (
        "<p>In the first round, at 12:15, the owner pressed the face centre about three times with the fingertip loose "
        f"on the desk. Afterwards the rest stayed {st['step_norm_uT']:.1f} &#181;T from the rest before, "
        f"{min(st['per_chip_uT']):.1f} to {max(st['per_chip_uT']):.1f} &#181;T on each chip. Two models of this step "
        "were fitted: a uniform field $\\mathbf g$ read by each chip $k$ in its own axes $R_k$, and a rigid offset "
        "$\\mathbf o$ of the pad on the board, through the rest field's Jacobian $J$:</p>"
        "$$\\Delta\\mathbf b_k = R_k\\,\\mathbf g,\\qquad \\Delta\\mathbf b = J\\,\\mathbf o.$$"
        f"<p>A uniform field of {st['uniform']['g_norm_uT']:.1f} &#181;T reproduces the step to "
        f"{st['uniform']['residual_uT']:.1f} &#181;T (Figure 3). The best offset of the pad, "
        f"({pt(st['offset']['o_um'], 1)}) &#181;m, leaves {st['offset']['residual_uT']:.1f} &#181;T. The board therefore "
        f"turned by about {turn:.0f}&#176; in Earth's field (44 &#181;T in the lab's fit of 2026-10-22), or a magnetic "
        "object stayed near it. The presses never came back to one rest, so they merge into a single hold of "
        f"{h1['s']:.1f} s. This round is not scored.</p>"
        + fig(3, F["step"], "The change of the rest over the first round, per chip and axis (bars, grey), with the fit of "
              "a uniform field (blue) and of a rigid offset of the pad (red).", width=1000)
    )))

    t4 = table(
        ["Start", "Length (s)", "Change (&#181;T)", "Physical model", "With GP correction", "Robot calibration"],
        [[h["t0"], f"{h['s']:.1f}", f"{h['change_uT']:.1f}"]
         + [f"{pt(h['est'][k]['p'], keep=(1, 2))}; {h['est'][k]['f_n']:.2f}" for k in ("physical", "physical+gp", "robot cal")]
         for h in H],
        num_cols=(1, 2, 3, 4, 5), n=4, caption="The six holds of the second round and three predictors' estimates. Each estimate gives y and z of the "
                "contact point (mm), then the normal force (N).")
    t5 = table(
        ["Predictor", "Mean point, x, y, z (mm)", "From the reference point (mm)", "Spread (mm)", "Normal force (N)",
         "Distinct answers"],
        [[NAME[k], pt(Sm[k]["mean_p"]), f"{Sm[k]['mean_from_ref_mm']:.2f}", f"{Sm[k]['spread_max_mm']:.2f}",
          f"{Sm[k]['f_min']:.2f} to {Sm[k]['f_max']:.2f}", f"{Sm[k]['distinct']}"]
         for k in ("physical", "physical+gp", "robot cal", "gp raw", "mlp raw")],
        num_cols=(2, 3, 4, 5), n=5, caption="Each predictor over the six holds.")
    S.append(("round2", "Six presses with the fingertip fixed", (
        "<p>For the second round the fingertip was fixed to the desk. The owner pressed the face centre six times with "
        f"the 5 mm sphere indenter, from {H[0]['t0']} to 12:28:14. Each board change stayed above 30 &#181;T for "
        f"{lens.min():.1f} to {lens.max():.1f} s, with {gaps.min():.1f} to {gaps.max():.1f} s of rest between them. "
        "Figure 1 shows the viewer replaying the round on the pad's own rest fit. Figure 4 traces the change and the "
        "estimates, and Table 4 gives each hold.</p>"
        + fig(4, F["trace"], "The second round. (a) Board change: inside a run over the 30 &#181;T threshold (dotted) "
              "against the rest before that run, elsewhere against the last rest that stayed still for 4 s. Shaded: the "
              "six holds; the touch at 18 s lasted under 1 s and is not one. (b) Normal force of three predictors in "
              "0.2 s bins. (c) Contact point of the physical model (circles) and with the GP correction (triangles): y "
              "open, z filled, with the reference point's y and z.", width=1000)
        + f"<p>The physical model places the six contacts within {ph['spread_max_mm']:.2f} mm of their mean. That mean "
        f"lies {ph['mean_from_ref_mm']:.2f} mm from the reference point, {abs(ph['mean_minus_ref'][1]):.2f} mm toward "
        f"$-y$ and {abs(ph['mean_minus_ref'][2]):.2f} mm toward the mount end. Its normal forces are {ph['f_min']:.2f} "
        f"to {ph['f_max']:.2f} N (Figure 5, Table 5).</p>"
        f"<p>Sphere2's GP correction moves every point, by {np.linalg.norm(ms):.1f} mm on average. The mean move is "
        f"{abs(ms[2]):.1f} mm toward the mount end and {abs(ms[1]):.1f} mm toward $+y$. The correction raises the force "
        f"by {dF.min():.1f} to {dF.max():.1f} N. The corrected points lie {pc['mean_from_ref_mm']:.2f} mm from the "
        f"reference point, with a spread of {pc['spread_max_mm']:.2f} mm.</p>"
        "<p>The GP on the raw board change gives the same answer for all six presses: "
        f"({pt(gr['mean_p'])}) mm and {gr['f_min']:.2f} N. Far from its training inputs a GP returns the mean of its "
        f"training outputs. Here that is {s2['train_mean'][3]:.2f} N, at the training holds' mean point moved onto the "
        "pad's surface. The presses' board "
        f"changes lie {min(inp['nearest_training_uT']):.0f} to {max(inp['nearest_training_uT']):.0f} &#181;T from the "
        f"nearest of sphere2's {s2['holds']} training changes. Those training changes lie a median "
        f"{inp['training_nearest_median_uT']:.1f} &#181;T from their own nearest neighbours. The MLP on the raw board "
        f"change places all six presses at the mount end's edge, z = {mr['mean_p'][2]:.1f} mm, "
        f"{mr['mean_from_ref_mm']:.2f} mm from the reference point.</p>"
        f"<p>The robot calibration as it stands places the presses {rc['mean_from_ref_mm']:.2f} mm from the reference "
        f"point, {abs(rc['mean_minus_ref'][2]):.2f} mm toward the tip end. Its forces are {rc['f_min']:.2f} to "
        f"{rc['f_max']:.2f} N, {ratio.min():.1f} to {ratio.max():.1f} times the physical model's.</p>"
        + t4 + t5
        + fig(5, F["points"], "The six presses' contact points on the pad, seen from the palm, by predictor. Thin lines "
              "join each physical estimate to its corrected one. The GP on the board change gives one point for all six "
              "presses. Dashed squares: the chips; grey: the pad's surface points that face the palm.", width=640)
    )))

    S.append(("limits", "Limitations", (
        "<p>The presses were aimed at the face centre by eye, and the aim was not measured. A distance from the "
        "reference point therefore adds the aim's error to the estimator's. No force was measured, so the forces compare "
        "the predictors with each other only. The six presses test one point of the pad. Both pads have magnets of the "
        "same weak batch, so N52 magnets and the checkerboard pattern are tested in simulation only "
        f"({link(pg['results'], 'UR5 results page')}, Table 8).</p>"
    )))

    S.append(("future", "Future work", (
        "<ul>"
        "<li>Known points and forces on the desk. Clamp the fingertip in a printed jig with sphere seats at the reference "
        "point and 4 mm from it toward $\\pm y$ and $\\pm z$. Press through a kitchen scale or a load cell at about 1, 2 "
        "and 4 N, and run " + P.code("scripts/fingertip_second_tip_transfer.py --landmarks") + " with the seats' points "
        "and the scale's forces. If the GP correction's median location error exceeds the physical model's by more than "
        f"{ph['spread_max_mm']:.1f} mm, this round's spread, the correction does not transfer across pole patterns.</li>"
        "<li>A UR5 session on the second fingertip, with the protocol's registration and sphere2's 15 held-out grid "
        f"presses ({link(pg['protocol'], 'protocol page')}). On the robot pad the untrained estimator placed those "
        "presses within 3.79 mm. A physical model within 20 % of that on the second pad transfers with the rest fit "
        "alone.</li>"
        "<li>Retraining on the second pad: about 40 presses retrain the GP and the MLP on its own board changes (UR5 "
        f"results page, Figure 4). The simulation predicts {Dd['gp raw (retrained on D)']['loc_mm']:.2f} mm for the "
        "retrained GP.</li>"
        "<li>N52 magnets: after the swap, refit at rest (viewer key c) and repeat the presses in the jig. The simulation "
        "predicts 1.52 mm for the rebuilt physical model and 0.73 mm with the GP correction (UR5 results page, Table "
        "8).</li>"
        "</ul>"
        f"<p>Recordings: {P.code('data/20261023-121357_round1_records.npz')} and "
        f"{P.code('data/20261023-122717_round2_records.npz')} (raw board records; "
        + P.code("wrench_viewer.py --replay") + " plays them). Results: "
        f"{P.code('data/20261023-round1_hand_held.json')} and {P.code('data/20261023-round2_fixed_tip.json')}. Requests "
        f"and their state: {link(pg['status'], 'status page')}.</p>"
    )))
    return S, lede, summary, film_html
