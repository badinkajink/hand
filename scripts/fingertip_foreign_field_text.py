"""The prose of the page on fields from outside the pad (scripts/fingertip_foreign_field_page.py), from the study's data."""
from __future__ import annotations

import numpy as np

ROOT_PAGES = {
    "results": "docs/experiments/20261022-ur5_tip_calibration/20261022-ur5_tip_ground_truth_and_predictors.html",
    "estimation": "docs/experiments/20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html",
    "transfer": "docs/experiments/20261023-second_fingertip_transfer/20261023-second_fingertip_transfer.html",
    "overview": "docs/overviews/printed_fingertips.html",
}


def num(v, d=2):
    """A signed number with a true minus sign."""
    return f"{v:.{d}f}".replace("-", "&#8722;")


def reg(r):
    """A region's name with a true minus sign."""
    return r.replace("-y", "&#8722;y")


def rng(a, d=1, unit=""):
    a = np.asarray(a, float)
    u = f" {unit}" if unit else ""
    return f"{num(a.min(), d)} to {num(a.max(), d)}{u}"


def sections(J, F, P, link):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    pg = {k: root / v for k, v in ROOT_PAGES.items()}
    fig, table, code = P.fig, P.table, P.code
    R, H, A = J["touches"], J["holds"], J["approach"]["rows"]
    six = [p for p in R["plateaus"] if p["group"] == "six"]
    other = [p for p in R["plateaus"] if p["group"] != "six"]
    pr = R["presses"]
    geo = R["geometry"]
    ft, f1, f2, fs, fr = (J[k] for k in ("frames_touches", "frames_round1", "frames_round2", "frames_sphere2", "frames_rod1"))
    ep = J["episodes_sphere2"]["runs"]
    n_holds = sum(H[k]["n"] for k in H)
    n_turns = H["sphere2_turn"]["n"] + H["rod1_turn"]["n"]
    lam_max = max(H[k]["lam_max"] for k in H)
    d6 = np.array([p["change_uT"] for p in six])
    f6 = np.array([p["f_n"] for p in six])
    l6 = np.array([p["lam"] for p in six])
    dp = np.array([p["change_uT"] for p in pr])
    fp = np.array([p["f_n"] for p in pr])
    lp = np.array([p["lam"] for p in pr])
    shown_ur5 = fs["shown_off"] + fr["shown_off"]
    ep_dF = np.array([v for e in ep for v in e["hexe_from_rest_after_N"]])
    ep_f = np.array([v for e in ep for v in (e["f_shown_min"], e["f_shown_max"])])
    Mimg = np.array(geo["image_face_chip_uT"])
    foreign = [p for p in other if p["lam"] > 25]
    neg = [p for p in other if p["lam"] < 0]
    weak = [p for p in other if 0 <= p["lam"] <= 25]
    mid = [r for r in A if 40 <= r["change_uT"] <= 90]
    big = [r for r in A if r["change_uT"] >= 100]
    cos = J["cos"]

    lf = J["left"]
    lede = (
        "Steel touched without force to the second fingertip&#8217;s face changed the board&#8217;s reading by "
        f"{rng(d6, 0)} &#181;T. Each touch was estimated as a press of {rng(f6, 1)} N on a side wall. Each contact "
        "is now compared with a field from outside the pad, modelled as a uniform field and a linear gradient across the "
        f"board. On the six touches that field&#8217;s chi-square was lower than the contact&#8217;s by {rng(l6, 0)}. "
        f"On {n_holds} press and turn holds of the UR5 it was lower by at most {lam_max:.1f}. With a threshold of 25, "
        f"the frames shown as contacts during the touches fell from {ft['six']['shown_off']} of {ft['six']['frames']} "
        f"to {ft['six']['shown_on']}, at the onsets and ends of three touches. Of the UR5 streams&#8217; {shown_ur5:,} "
        f"contact frames, {fs['rejected'] + fr['rejected']} were rejected, all between holds, where the HEX-E read under "
        f"{ep_dF.max():.2f} N. The test is off by default; key f in the viewer switches it.")
    summary = (
        f"Steel touched to the fingertip without force changed the reading by {rng(d6, 0)} uT and was estimated as "
        f"{rng(f6, 1)} N presses. A test against a uniform field and a linear gradient across the board (threshold 25 "
        f"in chi-square) cut the contact frames of six touches from {ft['six']['shown_off']} to {ft['six']['shown_on']} "
        f"and rejected none of {n_holds} UR5 holds. Toggle: key f, --foreign-gate.")

    film_html = P.film(
        1, F["film"], F["poster"],
        "The contact viewer replays the six touches, from 15:43:16 to 15:43:46 of the recording, twice in real time: "
        "first with the test off, then with it on. Each run was calibrated on the clip&#8217;s first 2 s at rest, as the "
        "viewer is live. The poster shows the fourth touch in both runs. Arrow: the estimated force; the panel&#8217;s "
        "force trace covers the last 10 s, and its status line&#8217;s clock gives the time of the replay. Made by " + code("scripts/fingertip_foreign_field_film.py") + ".")

    S = []
    S.append(("terms", "Terms and metrics", f"""<dl class="terms">
<dt>Tip frame</dt><dd>$x$ along the normal of the palmar face, toward the palm; $y$ across the finger; $z$ from the mount
end toward the tip end; in mm, as on the {link(pg['estimation'], 'contact estimation page')}.</dd>
<dt>Board change $\\mathbf d$</dt><dd>The reading less the viewer&#8217;s baseline: 15 values in &#181;T, three axes of each of
five chips in the chip&#8217;s own frame. Each chip&#8217;s reading is divided by its gain at its die temperature.</dd>
<dt>Plateau</dt><dd>A steady stretch in which the board change exceeds 50 &#181;T for at least 0.8 s (30 &#181;T and 1 s for
presses). Its change is the mean of its middle 40 % less the mean of the 0.6 s before it.</dd>
<dt>Frame</dt><dd>One estimate of the viewer: every 33 ms, from the mean of the last 50 ms against the baseline. The baseline
is taken again once the reading has been still for 3 s.</dd>
<dt>Shown as a contact</dt><dd>The viewer&#8217;s display rule: the estimate explains more than 25 in chi-square, and its
pressing force is at least 0.25 N (0.15 N while a contact is shown).</dd>
<dt>Best contact</dt><dd>The estimator&#8217;s contact point $\\hat{{\\mathbf p}}$ and force $\\hat{{\\mathbf f}}$ for a board change,
anywhere on the pad&#8217;s free surface.</dd>
<dt>Far field</dt><dd>The field of sources outside the pad to first order across the board: a uniform field and a linear
gradient, eight parameters. Section 4 defines it.</dd>
<dt>Chi-square $\\chi^2$</dt><dd>The sum over the 15 components of the squared residual over its noise $\\sigma$. The
noise is the chips&#8217; noise for the window (1.65, 1.65 and 2.57 &#181;T a sample along $x$, $y$, $z$) and a 2
&#181;T floor, added in quadrature.</dd>
<dt>$\\Lambda$</dt><dd>The chi-square of the best contact less that of the far field (Section 4). It is positive when the far
field fits better.</dd>
<dt>Share explained</dt><dd>The fraction of $|\\mathbf d|^2$ fitted by least squares with a uniform field, with the far field,
or by the best contact (its explained energy over the reading&#8217;s, as the viewer computes it).</dd>
<dt>Images</dt><dd>The mirror images of the magnets in a steel plane. For a plane parallel to the face, each image sits at
its magnet&#8217;s mirror point with the same moment, since the moments lie along the plane&#8217;s normal.</dd>
<dt>Hold</dt><dd>On the UR5, a stretch in which the robot keeps a force level, a slide, a roll or a twist. Its ground truth
comes from the HEX-E, the six-axis force sensor at the robot&#8217;s flange ({link(pg['results'], 'UR5 results page')}).</dd>
</dl>"""))

    t1 = table(
        ["Event", "Start", "Length (s)", "Change (&#181;T)", "Estimate", "Force (N)", "Uniform share", "Far-field share",
         "Contact share", "&#923;"],
        [[f"Touch {k + 1}", p["t0"], f"{p['s']:.1f}", f"{p['change_uT']:.0f}", reg(p["region"]), f"{p['f_n']:.1f}",
          f"{p['share_uniform']:.2f}", f"{p['share_far']:.2f}", f"{p['share_contact']:.2f}", num(p["lam"], 0)]
         for k, p in enumerate(six)]
        + [[f"Press {k + 1}", p["t0"], f"{p['s']:.1f}", f"{p['change_uT']:.0f}", reg(p["region"]), f"{p['f_n']:.1f}",
            f"{p['share_uniform']:.2f}", f"{p['share_far']:.2f}", f"{p['share_contact']:.2f}", num(p["lam"], 0)]
           for k, p in enumerate(pr)],
        num_cols=(2, 3, 5, 6, 7, 8, 9), n=1,
        caption="The six touches of steel (15:43) and the six presses with the printed 5 mm sphere (12:27 to 12:28) on "
                "the second fingertip, each as the estimator saw its plateau on a 50 ms window&#8217;s noise.")
    S.append(("touches", "Six touches of steel on the second fingertip", (
        "<p>The owner touched two steel objects, a 100 g mass and a steel ruler, to the face centre of the second "
        "fingertip without pressing. The fingertip sat on the workstation&#8217;s board (d9:94), whose two magnets at the "
        f"mount end point the other way ({link(pg['transfer'], 'second-fingertip page')}). The viewer&#8217;s stream "
        "log recorded the board. Its records from 15:38:00 to 15:44:15 are kept in "
        + code("data/" + R["window"]["records"]) + ".</p>"
        f"<p>Six plateaus fell between 15:43:17 and 15:43:42 (Table 1). The first three changed the reading by "
        f"{rng(d6[:3], 0)} &#181;T and the last three by {rng(d6[3:], 0)} &#181;T. The record does not say which object "
        f"made which three. Every touch was estimated as a press low on a side wall, at $|y| = 7.4$ mm, of "
        f"{rng(f6, 1)} N. The six presses of 12:27 changed the reading by {rng(dp, 0)} &#181;T and were placed at the "
        f"face centre with {rng(fp, 1)} N.</p>" + t1)))

    S.append(("images", "Field of steel at the face", (
        "<p>Steel near the pad is magnetized by the magnets. Outside a plane of high permeability, the field of that "
        "magnetization equals the field of the magnets&#8217; mirror images (the method of images; Jackson, "
        "<i>Classical Electrodynamics</i>, 3rd ed., 1999). The magnets&#8217; centres lie 8.5 mm below the top of the "
        "face and 5.65 mm above the chips. A plane touching the face therefore puts the images 22.6 mm from the "
        "chips.</p>"
        f"<p>Computed with the viewer&#8217;s own field model, the images&#8217; field at the chips is "
        f"{geo['image_face_uT']:.0f} &#181;T, the norm of the 15 components. At single chips it is {rng(Mimg, 0)} "
        f"&#181;T (Figure 2c). No parameter was fitted. The touches changed the reading by {rng(d6, 0)} &#181;T. The "
        f"cosine between their pattern and the images&#8217; is {cos['t13']:.2f} for touches 1 to 3 and "
        f"{cos['t46']:.2f} for touches 4 to 6. For the presses it is {cos['press']:.2f}.</p>"
        "<p>The images lie four times as far from the chips as the magnets, so their field varies little across the "
        f"14 mm board. A uniform field and a linear gradient fit {rng([p['share_far'] * 100 for p in six], 0)} % of "
        f"each touch&#8217;s change and {rng([p['share_far'] * 100 for p in pr], 0)} % of each press&#8217;s. A press "
        "changes the field mostly at the chip under the contact: 78 &#181;T at the centre chip U1, against 7 to 33 "
        "&#181;T at the other four (Figure 2d). A uniform field change, from the finger turning in the Earth&#8217;s "
        "field, is part of the estimator&#8217;s model, with a prior of 30 &#181;T. The remaining 200 to 300 &#181;T of "
        f"a touch is fitted by a contact on a side wall, which explains {rng([p['share_contact'] * 100 for p in six], 0)} "
        f"% of the change. For a press the best contact explains {rng([p['share_contact'] * 100 for p in pr], 0)} %.</p>"
        + fig(2, F["patterns"],
              "Field change at each chip in the tip frame. (a, b) The means of the first and last three touches. (c) "
              "The images of the five magnets in a perfect steel plane touching the face, computed with the "
              "pad&#8217;s rest fit and no fitted parameter. (d) The mean of the six presses with the printed sphere. "
              "Chips from the mount end to the tip end, then the two sides.", width=900))))

    S.append(("test", "Test against a field from outside the pad", (
        "<p>Each contact the viewer would show is compared with the far field. The two chi-squares are "
        "$$\\chi^2_{\\mathrm c} = \\min_{\\mathbf g,\\,\\epsilon}\\big\\|\\Sigma^{-1/2}\\big(\\mathbf d - S(\\hat{\\mathbf p})\\,"
        "\\hat{\\mathbf f} - U\\mathbf g - \\mathbf b_0\\,\\epsilon\\big)\\big\\|^2,$$ "
        "$$\\chi^2_{\\mathrm f} = \\min_{\\mathbf g,\\,\\boldsymbol\\gamma,\\,\\epsilon}\\big\\|\\Sigma^{-1/2}\\big(\\mathbf d - "
        "U\\mathbf g - \\Gamma\\boldsymbol\\gamma - \\mathbf b_0\\,\\epsilon\\big)\\big\\|^2,$$ "
        "and $$\\Lambda = \\chi^2_{\\mathrm c} - \\chi^2_{\\mathrm f}.$$ "
        "Here $S(\\hat{\\mathbf p})$ ($15 \\times 3$) is the change per newton at the best contact point. $U$ ($15 \\times 3$) "
        "reads a uniform field $\\mathbf g$ through each chip&#8217;s rotation $R_c$. $\\mathbf b_0$ is the reading at rest, "
        "so $\\epsilon$ is a relative change of the magnets&#8217; strength. $\\Sigma$ is the diagonal noise covariance "
        "of the chi-square. Column $k$ of $\\Gamma$ ($15 \\times 5$) is the reading of the gradient field "
        "$G_k(\\mathbf r - \\bar{\\mathbf r})$ at the chips, with $\\bar{\\mathbf r}$ their centroid. The $G_k$ are "
        "five symmetric traceless matrices, so the field is free of curl and divergence.</p>"
        "<p>Each explanation has nine parameters. The contact has its force (3), point (2), uniform field (3) and "
        "strength change (1). The far field has its uniform field (3), gradient (5) and strength change (1). A contact "
        f"is not shown when $\\Lambda > 25$. The largest $\\Lambda$ of the {n_holds} holds of sphere2 and rod1 on the "
        f"UR5 is {lam_max:.1f}, and {n_turns} of those holds are slides, rolls and twists (Figure 3). The six touches "
        f"gave {rng(l6, 0)}, and the presses with the printed sphere {rng(lp, 0)}. The test is the class "
        + code("fingertip_wrench_model.ForeignField") + ".</p>"
        + fig(3, F["lam"],
              "&#923; of every hold of the UR5 sessions sphere2 and rod1 and of every plateau on the second fingertip, "
              "with the threshold of 25 (dashed). The axis is linear within &#177;10 and logarithmic beyond.",
              width=900))))

    t2 = table(
        ["Recording", "Frames", "Shown, test off", "Shown, test on", "Rejected in holds", "Largest &#923;"],
        [["Six touches of steel (&#177;0.3 s)", f"{ft['six']['frames']:,}", f"{ft['six']['shown_off']:,}",
          f"{ft['six']['shown_on']:,}", "&#8212;", "&#8212;"],
         ["Plateaus before the touches (&#177;0.3 s)", f"{ft['early']['frames']:,}", f"{ft['early']['shown_off']:,}",
          f"{ft['early']['shown_on']:,}", "&#8212;", "&#8212;"],
         ["Desk, 15:38:00 to 15:44:15", f"{ft['frames']:,}", f"{ft['shown_off']:,}", f"{ft['shown_on']:,}", "&#8212;",
          "&#8212;"],
         ["Desk, round 2: presses, fingertip fixed", f"{f2['frames']:,}", f"{f2['shown_off']:,}", f"{f2['shown_on']:,}",
          "&#8212;", num(f2["lam_max"], 1)],
         ["Desk, round 1: presses, fingertip in the hand", f"{f1['frames']:,}", f"{f1['shown_off']:,}",
          f"{f1['shown_on']:,}", "&#8212;", num(f1["lam_max"], 1)],
         ["UR5 sphere2 stream, 2.3 h", f"{fs['frames']:,}", f"{fs['shown_off']:,}", f"{fs['shown_on']:,}",
          f"{fs['hold_rejected']} of {fs['hold_shown_off']:,}", num(fs["lam_max"], 1)],
         ["UR5 rod1 stream, 1.0 h", f"{fr['frames']:,}", f"{fr['shown_off']:,}", f"{fr['shown_on']:,}",
          f"{fr['hold_rejected']} of {fr['hold_shown_off']:,}", num(fr["lam_max"], 1)]],
        num_cols=(1, 2, 3, 4, 5), n=2,
        caption="The viewer&#8217;s frames replayed from each recording with the test off and on: 30 frames a second on "
                "the desk, 15 on the UR5 streams. &#923; is computed for frames the viewer would show.")
    rows3 = {}
    for e in ep:
        r = rows3.setdefault(e["action"], dict(frames=0, f=[], h=[], lam=[], t0=e["t0"], kind=e["kind"]))
        r["frames"] += e["frames"]
        r["f"] += [e["f_shown_min"], e["f_shown_max"]]
        r["h"] += e["hexe_from_rest_after_N"]
        r["lam"].append(e["lam_max"])
    t3 = table(
        ["Action", "Start", "Frames", "Force shown (N)", "HEX-E from its rest (N)", "Largest &#923;"],
        [[k, r["t0"], f"{r['frames']}", rng(r["f"], 1), rng(r["h"], 2), f"{max(r['lam']):.0f}"] for k, r in rows3.items()],
        num_cols=(2, 3, 4, 5), n=3,
        caption="The frames of sphere2&#8217;s stream rejected by the test, by the robot&#8217;s action. The HEX-E&#8217;s "
                "force is taken against its rest after the action&#8217;s press.")
    S.append(("frames", "Frames of the viewer on the desk and the UR5", (
        "<p>Each recording was replayed through the viewer&#8217;s frame loop: the 50 ms window, the baseline taken "
        "again at rest, and the display rule (Table 2). During the six touches the contact frames fell from "
        f"{ft['six']['shown_off']} to {ft['six']['shown_on']}, and the largest force shown from "
        f"{ft['six']['f_max_off']:.1f} to {ft['six']['f_max_on']:.1f} N. Of the {lf['n']} frames left, {lf['onset']} fall "
        f"in the first {lf['onset_s'][0]:.2f} to {lf['onset_s'][1]:.2f} s of touches 4 to 6 and {lf['end']} at their "
        f"ends. They show {rng(lf['f'], 1)} N, with $\\Lambda$ from {rng(lf['lam'], 0)} (Figure 4b). The replay samples "
        "each onset at fixed times; in the film&#8217;s run with the test on, the force trace reaches 8.8 N at the "
        "fourth onset (Figure 1).</p>"
        f"<p>Before the six touches, {len(other)} more plateaus were recorded from 15:38:42 to 15:43:06. Of these, "
        f"{len(foreign)} gave $\\Lambda$ of {rng([p['lam'] for p in foreign], 0)} and were rejected (Table 4). For "
        f"{len(neg)} others $\\Lambda$ was negative, and the best contact explained "
        f"{rng([p['share_contact'] * 100 for p in neg], 0)} % of the change, as it does for a press. The remaining "
        f"{len(weak)}, at {' and '.join(p['t0'] for p in weak)}, were fitted better by the far field, by "
        f"{rng([p['lam'] for p in weak], 1)}. The record does not say what touched the pad then.</p>"
        f"<p>No frame of the presses on the desk was rejected, with the fingertip fixed or in the hand. On the UR5 "
        f"streams no frame inside a hold was rejected. In sphere2&#8217;s stream {fs['rejected']} frames were rejected "
        "between holds (Table 3). They fell in the retreats after the 3 N levels of two pivot presses and at the end of "
        f"a twist with the move to the next. The viewer showed {rng(ep_f, 1)} N in those frames, while the HEX-E read "
        f"{rng(ep_dF, 2)} N from its rest. No frame of rod1&#8217;s stream was rejected.</p>"
        + t2 + t3
        + fig(4, F["traces"],
              "Replay of the viewer&#8217;s frames over (a) 15:38:30 to 15:40:30, with 19 of the plateaus of Table 4, and (b) the "
              "six touches. "
              "Top: the board change against the baseline. Middle: &#923; of each frame the viewer would show, with "
              "the threshold of 25 (dashed); linear within &#177;30. Bottom: the pressing force shown with the test off "
              "(grey) and on (black)."))))

    S.append(("approach", "Object coming nearer", (
        "<p>Scaled between 2 and 100 % of their size, with the chips&#8217; noise added, the six touches stand for an "
        "object at decreasing distances (Figure 5). Below 30 &#181;T no contact is shown, with the test or without it, "
        "since the explained chi-square stays under 25. From 40 to 90 &#181;T, "
        f"{rng([r['shown_off'] * 100 for r in mid], 0)} % of the frames show a contact of "
        f"{rng([r['f_mean_shown_off'] for r in mid], 1)} N with the test off, and "
        f"{rng([r['shown_on'] * 100 for r in mid], 0)} % with it on. From 100 &#181;T, "
        f"{rng([r['shown_off'] * 100 for r in big], 0)} % of the frames show "
        f"{rng([r['f_mean_shown_off'] for r in big], 1)} N with the test off, and none with it on. The real onsets "
        f"and ends scored lower than a scaled touch of the same size: the frames left in Section 5, at {rng(lf['d'], 0)} "
        f"&#181;T, gave $\\Lambda$ from {rng(lf['lam'], 0)}.</p>"
        + fig(5, F["approach"],
              f"The six touches scaled from 2 to 100 %, {J['approach']['n_per_scale']} noisy frames at each scale: (a) "
              "the share of frames shown as a contact, (b) the mean force of those frames.", width=900))))

    S.append(("viewer", "Toggle in the viewer", (
        "<p>" + code("wrench_viewer.py") + " takes " + code("--foreign-gate") + " (off by default) and "
        + code("--foreign-lambda") + " (25). Key f and a checkbox switch the test while the viewer runs. When a contact "
        "is rejected the panel reads &#8220;No contact: a field from outside the pad&#8221;, and the status line gives "
        "the change and $\\Lambda$. The settings line counts the rejected frames. The test adds two least-squares fits "
        "of nine columns to a frame.</p>"
        "<p>A press made with a steel object is rejected where the object&#8217;s field dominates the reading. A steel "
        "object left "
        "still on the pad for 3 s is taken into the baseline by the re-zero, like any steady reading.</p>"
        "<pre>~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/wrench_viewer.py --foreign-gate</pre>")))

    S.append(("limits", "Limitations", (
        f"<p>The onsets and ends of three touches still showed contacts: {rng(lf['f'], 1)} N on the replay and 8.8 N "
        "in the film. Fields of 40 to 90 &#181;T are not separated from presses. Two steel objects were touched to one "
        "fingertip, by hand, without a force reference. The threshold rests on the robot fingertip&#8217;s two UR5 "
        "sessions and on the second fingertip&#8217;s presses of the morning.</p>")))

    S.append(("future", "Future work", (
        "<ol>"
        "<li>Steel at known distances: touch the mass and the ruler through printed spacers of 0.5, 1, 2, 5 and 10 mm "
        "and run " + code("scripts/fingertip_foreign_field.py touches") + " on the new window. The spacers give the "
        "distances at which the reading lies between 40 and 90 &#181;T. Section 6 is wrong if a spacer reading over 100 "
        "&#181;T scores $\\Lambda$ under 25.</li>"
        "<li>A known force with steel: set the 100 g mass on the face (0.98 N) and fit a contact with the far field as a "
        "nuisance. The joint fit is useful if its force falls within 0.3 N of 0.98 N.</li>"
        "<li>The UR5 live: run the viewer with " + code("--foreign-gate") + " in the next lab session and count the "
        "rejected frames inside holds. If a frame inside a hold is rejected while the HEX-E reads above 0.5 N, the "
        "threshold is too low.</li>"
        "<li>Small steel parts: touch a 5 mm steel ball and an M3 screw to the face centre and score them as in "
        "Table 1.</li>"
        "<li>Onsets: delay a new contact by 0.1 s before it is shown, and score the delay on the six touches and on the "
        "UR5 streams&#8217; onsets.</li>"
        "</ol>")))

    S.append(("files", "Files", (
        "<ul>"
        "<li>" + code("scripts/fingertip_wrench_model.py") + ": " + code("far_field_basis") + " and "
        + code("ForeignField") + ".</li>"
        "<li>" + code("src/morphohand/driver/reskin5x/host/wrench_viewer.py") + " and " + code("wrench_viewer.html") + ": "
        "the toggle.</li>"
        "<li>" + code("scripts/fingertip_foreign_field.py") + " (" + code("touches") + ", " + code("holds") + ", "
        + code("frames") + ", " + code("episodes") + ", " + code("approach") + "): the analysis, into "
        + code("data/") + ".</li>"
        "<li>" + code("scripts/fingertip_foreign_field_film.py") + ": Figure 1. " + code("scripts/fingertip_foreign_field_page.py")
        + " and " + code("scripts/fingertip_foreign_field_text.py") + ": this page.</li>"
        "<li>" + code("data/" + R["window"]["records"]) + ": the raw records, readable by "
        + code("wrench_viewer.py --replay") + "; " + code("data/" + R["calib"]["file"]) + ": the viewer&#8217;s calibration "
        "of the second fingertip.</li>"
        "</ul>")))

    t4 = table(
        ["Start", "Length (s)", "Change (&#181;T)", "Estimate", "Force (N)", "Far-field share", "Contact share", "&#923;"],
        [[p["t0"], f"{p['s']:.1f}", f"{p['change_uT']:.0f}", reg(p.get("region", "&#8212;")), f"{p.get('f_n', 0):.1f}",
          f"{p['share_far']:.2f}", f"{p['share_contact']:.2f}", num(p["lam"], 0)] for p in other],
        num_cols=(1, 2, 4, 5, 6, 7), n=4, caption="The plateaus recorded before the six touches.")
    S[4] = (S[4][0], S[4][1], S[4][2] + t4)
    return S, lede, summary, film_html
