"""Prose of the page on the fingertip's calibration on the UR5 (scripts/fingertip_calibration_page.py), kept apart from the
plotting. Numbers come from the page's numbers.json and from the sessions' analysis outputs named in the text."""
from __future__ import annotations


def sections(N, F, P, link, L):
    """(sections, lede, summary, film_html)."""
    bl = N["by_level"]
    s2, c1, c2 = (bl[k] for k in ("20261022-210852-sphere2", "20261023-203938-cb1t", "20261025-152920-cb2t"))
    gp_hi = [s2["gp_over_1p5N"], c1["gp_over_1p5N"], c2["gp_over_1p5N"]]
    fn_err = [s2["gp_fn_err_N"], c1["gp_fn_err_N"], c2["gp_fn_err_N"]]
    cov = N["coverage_cb2t"]
    ps = N["pooling_shear"]
    tr = N["transfer"]
    import json
    from pathlib import Path
    # v1 tips' presses scaled by the magnets' strength fitted at rest, applied to v2 ID 1 (2026-10-27, commit 2bbedb0b)
    st = json.loads((Path(__file__).resolve().parents[1] / "docs/experiments/20261022-ur5_tip_calibration/data/"
                     "20261026-192158-v2id1/scaled_transfer.json").read_text())

    lede = (
        "Three printed fingertips were pressed onto a 5 mm sphere by the lab's UR5, with an OptoForce HEX-E force-torque "
        "sensor between the flange and the fingertip as the force reference. In four sessions they gave 4,579 holds. The "
        "registered robot pose gives each hold's contact point, and the HEX-E its force. Trained on a tip's own presses, "
        "a Gaussian process from the 15 magnetometer readings locates presses at new points within "
        f"{min(gp_hi):.2f}–{max(gp_hi):.2f} mm at 1.5 N and over. It gives their normal force within "
        f"{min(fn_err):.2f}–{max(fn_err):.2f} N. The untrained physical model gives 0.9–2.4 mm. The error grows with "
        "the distance to the nearest training press, from 0.32 mm within 1 mm to 6 mm beyond 4 mm. With slides in "
        "training, the shear is within 0.12 N and its direction within 6°. Torsion is not recovered. A model learned on "
        "one tip misses on another, and pooling two tips' presses helps where one tip has few.")
    summary = (
        f"UR5 + HEX-E calibration of three printed magnetic fingertips (4,579 holds): a GP from the 15 readings locates new "
        f"points within {min(gp_hi):.2f}-{max(gp_hi):.2f} mm at 1.5 N and over and the normal force within "
        f"{min(fn_err):.2f}-{max(fn_err):.2f} N (physical model untrained 0.9-2.4 mm); error grows with distance to the "
        "nearest training press (0.32 mm within 1 mm, 6 mm beyond 4 mm); shear 0.12 N and 6 deg with slides in training; "
        "torsion not recovered; a GP trained on another tip gives 1.42 mm against 0.39 mm.")
    film = P.film(0, F["film"], F["film_poster"],
                  "The first UR5 session, sphere2, from the overhead camera: one frame every 2 s of its first touch-offs, "
                  "played at 12 frames a second. The pad touches the sphere on the right-hand fixture; the left-hand fixture "
                  "holds the rod. Film from " + link(L["lab"], "the lab runs page") + ".")
    film = film.replace("Figure 0.", "Film.").replace('id="fig0"', 'id="film"')

    S = []
    terms = [
        ("Pad frame P", "The fingertip's frame in mm: $x$ along the face's normal toward the palm, $y$ across the finger, "
         "$z$ along it from the mount toward the tip end. The other frames are the robot's base B, its flange E, the "
         "HEX-E S, the calibration mount M and the indenter I (Table 1)."),
        ("Touch-off", "A slow approach of the pad onto the indenter until the push begins. Its onset is the flange pose "
         "where the push starts, found by a fit at zero force."),
        ("Registration", "The least-squares fit of the mount's pose on the flange, the pad's pose on the mount and the "
         "sphere's centre to the touch-offs."),
        ("Press, hold, level", "A press moves the pad along the surface normal at a target point. It stops at force "
         "levels from 0.25 to 5 N, and each level is kept about 1.5 s: a hold. A hold is one sample of the ground truth "
         "and of the board."),
        ("Slide, roll, twist", "Turns of the pressed pad: about the sphere's centre (the contact slides at constant "
         "indentation), about the contact point (the contact rolls), and about the normal through the contact."),
        ("Ground truth", "A hold's contact point $\\mathbf p$, normal $\\mathbf n$ and force $\\mathbf f$ on the pad, from "
         "the robot's pose, the registration and the HEX-E (Section 6)."),
        ("Normal force, shear, torsion", "The force's component along $-\\mathbf n$ in N, its component in the tangent "
         "plane in N, and the moment about the normal through the contact in N mm."),
        ("Board change", "The 15 readings of the five magnetometer chips over a hold, in µT, less the readings at rest "
         "before and after the press, interpolated to the hold's time."),
        ("Location error", "Distance in mm between an estimated contact point and the ground truth, in the tangent "
         "plane; a median over holds."),
        ("Physical model", "The estimator of " + link(L["phys"], "the physical-model page") + ": finite elements and a "
         "dipole model inverted, fitted only to the pad's readings at rest."),
        ("Gaussian process (GP), MLP", "Regressions from the board change to the contact point and the force, trained on "
         "holds (Section 8). MLP: a multilayer perceptron, here an ensemble of five."),
        ("Pooling", "Training one model on the holds of more than one tip."),
        ("Tips 0, 1, 2", "Tip 0 has its magnets all one way (session sphere2); tips 1 and 2 have the checkerboard of "
         "poles (sessions cb1t, cb1r and cb2t). All have the same magnets, at 0.34–0.57 of an N52's strength."),
    ]
    S.append(("terms", "Terms and metrics", '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms)
              + "</dl>"))

    body = (
        "<p>Six numbers describe the contact of a soft fingertip with an object. They are two coordinates of the point on "
        "its surface, the normal force, two components of shear and the torsion about the normal. The normal force is the "
        "grip. The ratio of shear to normal force, against the friction coefficient, is the margin to slip. The torsion "
        "resists a tool turning between two fingertips, as derived in " + link(L["contact"], "the contact overview")
        + ". The point says where the object lies in the hand.</p>"
        "<p>The board measures none of these. It measures the field of five magnets that a load moves. The physical "
        "model predicts the readings from finite elements and a dipole model. Untrained, it places presses 0.9–2.4 mm "
        "off at 2 N and over, and it fails below 1 N. The calibration therefore records the readings together with a "
        "reference for all six numbers, over the whole surface and the range of forces a grasp uses. A robot gives the "
        "point, and a six-axis sensor the force and torque.</p>"
    )
    S.append(("goal", "What the calibration must measure", body))

    body = (
        "<p>The fingertip, its DIP link and the calibration mount sit on the HEX-E, which sits on the UR5's flange. The "
        "finger points along the tool axis, so the face's normal lies across it. The indenter is a printed sphere of "
        "radius 5 mm on a post clamped to the table; a rod of radius 12.5 mm stands beside it (Figure 1). The pad's face "
        "centre lies 128.4 mm from the flange.</p>"
        + P.fig(1, F["cell"], "The cell in the simulated session: a touch-off on the face (top) and on the flank (bottom). "
                "Grey: the UR5 and the HEX-E; blue: the mount; cream: the DIP link; dark grey: its cover and cap; orange: "
                "the pad. From " + link(L["protocol"], "the protocol page") + ".")
        + P.table(["Frame", "Origin and axes"], [
            ["B", "The robot's base, as the controller reports poses."],
            ["E", "The robot's flange."],
            ["S", "The HEX-E: its moment origin on its tool face."],
            ["M", "The mount: origin on the HEX-E's tool face, $z$ along the tool axis toward the tip, $x$ toward the "
                  "palmar side."],
            ["P", "The pad frame of the estimator."],
            ["I", "The indenter: $z$ up from its base, the sphere's centre at (0, 0, 41) mm."]],
            caption="Frames. $T_{XY}$ is the pose of frame Y in frame X.", n=1)
    )
    S.append(("cell", "Cell, tool and frames", body))

    body = (
        "<p>Each kind of action in a session excites one part of the contact (Figure 2, Table 2). Touch-offs fix where "
        "the pad is. Presses map the readings against the point and the normal force. Slides add shear, rolls new tool "
        "orientations, and twists torsion. Reference presses at the face centre, repeated every 15 presses, show drift. "
        "The session cb2t of 2026-10-25 ran the full suite in six passes on tip 2 (Table 3).</p>"
        + P.fig(2, F["motions"], "The six motions, drawn in a section through the pad, face down, and the 5 mm sphere. "
                "In (b) the dotted line is the tilted contact normal. In (d) and (e) the red dot is the centre of the "
                "turn, and in (f) the dashed line is the twist's axis, the contact normal.")
        + P.table(["Action", "Motion", "What it measures", "On cb2t"], [
            ["Homing", "The mount's socket seated on the indenter's key by hand", "The indenter's pose, to about 0.2 mm",
             "1"],
            ["Gravity sweep", "Twelve tool orientations in free air", "The HEX-E's axes and gains, the payload's weight and "
             "centre", "Taken from the session first2"],
            ["Touch-off", "Along the normal at 0.15 mm/s to 0.5 N", "The flange pose at first contact", "19"],
            ["Offset touch-off", "As a touch-off, 3 mm beside the sphere's top", "The pad's place along the finger", "12"],
            ["Grid press", "Along the normal at a target, to 0.25–5 N", "Readings against point and normal force",
             f"{cov['grid']} + {cov['fast_grid']} faster"],
            ["Twin press", "A grid target mirrored about $y=0$", "The side with fewer targets", str(cov["twin"])],
            ["New target", "Farthest from earlier targets, at four speeds", "Readings between targets; speed",
             str(cov["dense"])],
            ["Rate press", "Six points to 2 N at 0.045, 0.6 and 2 mm/s", "The effect of press speed", str(cov["rate"])],
            ["Slide", "Turn about the sphere's centre at 1 N", "Shear in stick and slip", "87"],
            ["Roll", "Turn about the contact point", "New tool orientations", "5"],
            ["Twist", "Turn about the normal, ±5–20°, at 1, 2.5 and 5 N", "Torsion", "17"],
            ["Reference", "Face centre to 0.5, 1 and 2 N", "Drift of the seat and the board", str(cov["ref"])]],
            caption="Actions of a session and what each measures. The counts are actions on cb2t.", n=2)
        + "<p>Three choices in the suite came from the data. The first sessions' touch-offs all met the sphere at its top, "
        "with a vertical normal, and none of them constrains a horizontal shift of the sphere. Twelve touch-offs 3 mm "
        "beside the top meet tilted parts of the pad. On tip 1 they took the spread of the contact points along the "
        "finger from 0.348 to 0.105 mm. The grid of the first plans had 30 targets at $y<-1.5$ mm and 19 at "
        "$y>1.5$ mm, and the errors differed by side. Eighteen twins therefore mirror grid targets about $y=0$. Pass 6 placed "
        "37 new targets as far as possible from all earlier ones, to measure how errors grow between targets.</p>"
        + P.table(["Pass", "Actions", "Contents"], [
            ["1", "28", "Touch-offs, registration"],
            ["2a", "31", "Offset touch-offs and a joint registration"],
            ["2b", "67", "58 grid presses, rolls, twists"],
            ["3", "55", "Twins, rate presses at four speeds, slides, twists at 5 N, rolls"],
            ["5", "101", "52 grid presses at 0.6 mm/s, 48 slides"],
            ["6", "73", "37 new targets, 33 slides, 2 twists"]],
            caption="The passes of cb2t (tip 2 on mount v7, 2026-10-25 15:29 to 23:42). Pass 4 was a rate test on tip 1.",
            n=3)
    )
    S.append(("motions", "Motions of a session and what each identifies", body))

    body = (
        "<p><b>Homing.</b> In freedrive the operator seats the mount's socket on a key of the indenter's post. The key's "
        "faces are drafted 20°, so they centre the socket and fix its turn. With $T_{IM}^{h}$ the mount's pose at homing "
        "from the CAD, the indenter's pose follows from the mean flange pose of three seatings:</p>"
        "$$T_{BI}=\\bar T_{BE}^{h}\\,T_{EM}\\,\\bigl(T_{IM}^{h}\\bigr)^{-1}.$$"
        "<p><b>Touch-offs.</b> The push $y$ against the travel $s$ is fitted by</p>"
        "$$y=a\\,(s-s_0)_+^{\\,p}+b,\\qquad p\\in\\{1,\\,1.25,\\,1.5,\\,1.75,\\,2\\}.$$"
        "<p>At the onset $s_0$ the push is zero, so neither the pad's compliance nor the HEX-E's enters it. The flange "
        "pose at the onset, $T_{BE}^{(k)}$, is the touch-off's datum.</p>"
        "<p><b>Registration.</b> One Levenberg–Marquardt fit of 15 parameters moves the mount on the flange and the pad on "
        "the mount (a rotation and a translation each) and the sphere's centre $\\mathbf c_B$. At every touch-off the "
        "sphere's centre must lie one radius from the pad's surface:</p>"
        "$$\\min\\ \\sum_k\\frac{\\bigl(d_P\\bigl(T_{PB}^{(k)}\\mathbf c_B\\bigr)-R\\bigr)^2}{\\sigma_p^2}+\\text{priors},"
        "\\qquad T_{PB}^{(k)}=\\bigl(T_{BE}^{(k)}T_{EM}T_{MP}\\bigr)^{-1}.$$"
        "<p>$d_P$ is the signed distance to the pad's surface, $R=5$ mm and $\\sigma_p=0.03$ mm. The priors keep each "
        "parameter near its CAD value within its tolerance. The fits leave 0.045–0.057 mm RMS on the four sessions. "
        "They moved the tool centre point by up to 2.0 mm and 2.8° from its CAD and homing values. In a simulated "
        "session with injected errors, the registration placed the contact within 0.07 mm, against 0.39 mm from CAD "
        "and homing alone (" + link(L["protocol"], "protocol page") + ", Section 7).</p>"
    )
    S.append(("registration", "Registration of the tool and the sphere", body))

    body = (
        "<p>Each hold gives one sample. The HEX-E deflects under the wrench by $\\delta_S$, from its data sheet. The pad's "
        "pose and the sphere's centre in the pad frame are</p>"
        "$$T_{BP}=T_{BE}\\,T_{ES}\\,\\exp(\\delta_S)\\,T_{ES}^{-1}\\,T_{EP},\\qquad "
        "\\mathbf c_P=T_{BP}^{-1}\\,\\mathbf c_B .$$"
        "<p>The contact point $\\mathbf p$ is the surface point whose normal passes through $\\mathbf c_P$, and "
        "$\\mathbf n$ is that normal. The indentation is $R-d_P(\\mathbf c_P)$. The force is the HEX-E's change from the "
        "rest before the press, less the change of the payload's weight between the two orientations:</p>"
        "$$\\mathbf f_P=R_{PE}\\,K^{-1}\\bigl(\\mathbf f_S-\\mathbf f_S^{\\mathrm{rest}}-A\\,(\\hat{\\mathbf g}_E-"
        "\\hat{\\mathbf g}_E^{\\mathrm{rest}})\\bigr),\\qquad f_n=-\\mathbf f_P\\cdot\\mathbf n,\\qquad "
        "\\mathbf f_t=\\mathbf f_P+f_n\\,\\mathbf n .$$"
        "<p>$K$ maps a force in the flange frame to the HEX-E's reading, and $A$ is the payload's weight through $K$. Both "
        "come from the gravity sweep. $\\hat{\\mathbf g}_E$ is gravity's direction in the flange frame. The torque "
        "about the normal through the contact is $\\tau_n=\\mathbf n^{\\top}\\bigl(\\boldsymbol\\tau_E-(\\mathbf p_E-"
        "\\mathbf o_E)\\times\\mathbf f_E\\bigr)$. On the UR5 the face's normal lies across the tool axis, so a 2.5 N "
        "push puts about 230 N mm of moment on the HEX-E, a hundred times the torsion. A twist therefore turns the pad "
        "out to an angle and back. Friction reverses with the direction of turning and the push's moment does not, so the "
        "torsion is half the difference:</p>"
        "$$\\tau_{\\mathrm{fr}}=\\tfrac12\\bigl(\\tau_n^{\\mathrm{out}}-\\tau_n^{\\mathrm{back}}\\bigr).$$"
        "<p>The board change of a hold is its mean reading less the rests before and after the press, interpolated to the "
        "hold's time. Each chip's reading is first divided by its gain at its die temperature, "
        "$1+\\kappa_c(T_c-25\\,^{\\circ}\\mathrm C)$. Over a 2.3 h session the board's zero moved by up to 71 µT, 98 % of it "
        "from Earth's field turning with the tool. A press left a step of about 7 µT, which no learned zero removed "
        "better than this re-zero from the press's own rests (" + link(L["drift"], "drift page") + ").</p>"
        "<p>On a simulated session these equations recovered the contact point within 0.064 mm, the normal force within "
        "0.4 % above 0.4 N, and the shear within 0.016 N (medians). On the robot, the reference press at the face centre "
        "kept its onset within 0.07 mm over 65 minutes of sphere2.</p>"
    )
    S.append(("truth", "Ground truth of a hold", body))

    rows = [
        ["sphere2", "0, magnets one way", "v3", "2026-10-22 night", "357 (74)", "78 (10)"],
        ["rod1", "0, rod of radius 12.5 mm", "v3", "2026-10-23", "279", "–"],
        ["cb1t", "1, checkerboard", "v3", "2026-10-23 evening", "302 (60)", "64 (8)"],
        ["cb1r", "1, checkerboard", "v3", "2026-10-25", "170 (38)", "254 (16)"],
        ["cb2t", "2, checkerboard", "v7", "2026-10-25", "921 (187)", "2,433 (109)"],
        ["cb3t", "3, checkerboard", "v7", "2026-10-26", "running", "running"],
    ]
    body = (
        "<p>Table 4 lists the sessions. Tip 0 and tip 1's first session ran early plans with few slides. Tip 2's session "
        "cb2t has most of the data: 104 press points in 921 holds, and 87 slides. Its coverage is in Figure 3. The face "
        "and edges are covered within about 1.5 mm. The flanks are reached only above the cover's walls, and 74–79 % of "
        "their FE surface points lie more than 1.5 mm from any press.</p>"
        + P.table(["Session", "Tip, indenter", "Mount", "Date", "Press holds (presses)", "Turn holds (turns)"], rows,
                  caption="Sessions on the UR5. Turn holds are those of slides, rolls and twists.", n=4)
        + P.fig(3, F["coverage"], "Press targets of cb2t on the pad, by pass. Light grey: FE surface points within 1.5 mm of "
                "a press; dark grey: farther. The flanks' lower part lies behind the cover's walls, and the tip end under "
                "the cap.")
    )
    S.append(("data", "Sessions and coverage of the pad", body))

    body = (
        "<p>A learned model maps the board change to the contact (Figure 4). Its input is the 15-component board change. "
        "Its output is the contact point in the pad frame, moved onto the nearest surface point, and the force vector, "
        "from which the normal force and the shear follow. The temperature change and the time since the rest were tried "
        "as inputs and did not help on sphere2. The tool's orientation was left out: each target is pressed at its own "
        "orientation, so the orientation names the target.</p>"
        "<p><b>Gaussian process.</b> The inputs $\\mathbf x$ are standardized to $\\mathbf z$, with a floor of 1 µT per "
        "component, and the outputs to zero mean and unit variance. The kernel is squared-exponential with one length "
        "$\\ell$ for all inputs:</p>"
        "$$k(\\mathbf z,\\mathbf z')=\\exp\\Bigl(-\\frac{\\lVert\\mathbf z-\\mathbf z'\\rVert^2}{2\\ell^2}\\Bigr),\\qquad "
        "\\hat{\\mathbf y}(\\mathbf z_*)=\\mathbf k_*^{\\top}\\bigl(K+\\sigma^2I\\bigr)^{-1}Y .$$"
        "<p>$\\ell$ and the noise $\\sigma$ are chosen from grids of five and four values by the summed log marginal "
        "likelihood of the outputs. A GP interpolates between training holds, and it predicts the training mean far "
        "from all of them.</p>"
        "<p><b>MLP ensemble.</b> Five networks, each of 15 → 64 → 64 → 4 units with SiLU activations, are trained by AdamW "
        "with weight decay on standardized data. The number of steps is chosen on the latest fifth of the training "
        "presses. "
        "The prediction is the five networks' mean.</p>"
        "<p><b>Physical model with a correction.</b> A GP whose input is the physical model's estimate, the point and the "
        "normal force, learns that estimate's error. It is trained on the holds that the physical model places within "
        "2 mm. Because its input is the physical estimate, it can be applied to a tip whose physical model is rebuilt "
        "from its own rest calibration.</p>"
        "<p><b>Splits.</b> All holds of one press point are always on the same side of a split, because the holds of one "
        "press differ only in force. Press points within 0.75 mm of each other, such as a grid target and its faster "
        "copy, form one group. Scores are five-fold over groups, or on held-out groups. Pooling adds the holds of "
        "another tip to the training set.</p>"
        + P.fig(4, F["learning"], "The learned models: input, two regressors, output, and the correction of the physical "
                "model.")
    )
    S.append(("learning", "Learning the map from readings to contact", body))

    body = (
        "<p><b>Contact point and normal force.</b> Figure 5 gives the location error by force level for the GP on new "
        "points and for the untrained physical model. At 1.5 N and over the GP places new points within "
        f"{s2['gp_over_1p5N']:.2f}, {c1['gp_over_1p5N']:.2f} and {c2['gp_over_1p5N']:.2f} mm on tips 0, 1 and 2. Below "
        f"1.5 N it gives {s2['gp_under_1p5N']:.2f}, {c1['gp_under_1p5N']:.2f} and {c2['gp_under_1p5N']:.2f} mm, where "
        "the physical model fails. Its normal force is within "
        f"{s2['gp_fn_err_N']:.3f}, {c1['gp_fn_err_N']:.3f} and {c2['gp_fn_err_N']:.3f} N. On sphere2 the MLP ensemble "
        "gave 0.79 mm over four folds, against the GP's 1.00 mm. On the rolls and twists, at tool orientations that no "
        "press had, it gave 2.93 mm against the GP's 2.05 mm (" + link(L["gt"], "ground-truth page") + ", Section 6).</p>"
        + P.fig(5, F["by_level"], "Median location error by force level. Solid: a GP trained on the tip's other press "
                "points, five folds over groups of points. Dashed: the physical model, fitted only at rest. Holds at "
                "3 N are edge presses; at 3.5 and 5 N face presses only, so the levels also differ in region.")
        + "<p><b>Coverage.</b> The error depends on how far a press lies from the nearest training press (Figure 6). On "
        "cb2t, with a fifth of the points held out and eight seeds, the error at 1.5 N and over is 0.32 mm within 1 mm of "
        "a training hold. It is 0.98 mm at 1.5–2 mm and 6.1 mm beyond 4 mm. At the same point pressed at another speed "
        "the error is 0.22 mm, the floor of the method. With all 84 training points, the median distance is 1.0 mm and "
        "the error 0.36 mm. Pass 6 measured this directly: with the grid alone in training its 37 new targets were "
        "placed within 2.13 mm, and with the other new targets added within 0.47 mm. More points help where the "
        "nearest press is more than about 1 mm away, which today is most of the flanks.</p>"
        + P.fig(6, F["density"], "Location error on cb2t against coverage. (a) Error against the distance from a held-out "
                "hold to the nearest training hold, over eight seeds and six training fractions. The dotted line is the "
                "same point pressed at another speed. (b) Error against the number of press points in training, with "
                "the seeds' range at 1.5 N and over.")
        + "<p><b>Shear.</b> On cb2t's 87 slides the HEX-E measured a median shear of 0.35 N and at most 1.61 N (Figure 7). "
        "The physical model gives 1.30 N of error and 38.7° in direction. A GP trained on presses alone gives 0.18 N and "
        "9.6°, and with the other slides in training 0.12 N and 5.9°. Presses contain little shear, so the slides teach "
        "the model what shear does to the readings.</p>"
        + P.fig(7, F["shear"], "Shear on cb2t's slides: each component of the estimated shear in the tangent plane against "
                "the HEX-E's, for the physical model and two GPs. The numbers are medians of the error's length and of "
                "the angle between the estimated and the measured force.")
        + "<p><b>Torsion.</b> Twists out to 20° and back give torsions of 1.6–2.8 N mm (medians at 1–5 N, cb2t) and up "
        "to 8.6 N mm. A GP from the board change follows them with a correlation of 0.39 on cb2t and 0.09 on tip 1's "
        "cb1r. The torsion is read from the HEX-E under a push moment a hundred times larger, and its value scatters "
        "from −3 to 9 N mm at one force. The torsion is therefore not calibrated.</p>"
        "<p><b>Speed and time.</b> Faster presses overshoot the force level: by 0.16 N at 0.6 mm/s and 0.65 N at 2 mm/s. "
        "The board change at the same target and level grows by 1.24 and 1.79 times; at 2 mm/s the larger force explains "
        "about half of that growth. "
        "The GP trained at 0.15 mm/s still places those presses within 0.41–0.68 mm. A day after sphere2, 27 of its "
        "presses were repeated; the GP trained on the night's presses placed them within 0.80 mm, against 0.84 mm on "
        "the night's own held-out presses.</p>"
        "<p><b>Transfer and pooling.</b> A GP trained on tip 1 and applied to tip 2 misses by "
        f"{tr['gp_id1']['over_1p5N']:.2f} mm at 1.5 N and over (Figure 8a). Tip 2's own GP gives "
        f"{tr['gp_own_5fold']['over_1p5N']:.2f} mm, and tip 2's physical model from its rest fit "
        f"{tr['physical_own_rest']['over_1p5N']:.2f} mm. Two tips with the same magnets and pole pattern still differ in the magnets' places and seating. "
        "Pooling helps where one tip has few presses, once the tips' board changes are put on one scale. Each change is "
        "divided by its magnets' strength as fitted at rest: 0.98 of an N52's for v2 ID 1, with five N52 magnets, and "
        f"{min(st['k'][k] for k in st['k'] if 'v2id1' not in k):.2f}–{max(st['k'][k] for k in st['k'] if 'v2id1' not in k):.2f} "
        f"for the v1 tips. A GP of the v1 tips' {st['n_v1_holds']:,} holds then places v2 ID 1's contacts within "
        f"{st['v1_scaled']['over_1p5N']:.2f} mm at 1.5 N and over, against {st['v1_raw']['over_1p5N']:.2f} mm unscaled. "
        f"With 12 or 24 of v2 ID 1's {st['n_actions']} press actions added, the error is "
        f"{st['thinned']['12']['own_and_v1_scaled']['over_1p5N']:.2f} and {st['thinned']['24']['own_and_v1_scaled']['over_1p5N']:.2f} mm, "
        f"against {st['thinned']['12']['own']['over_1p5N']:.2f} and {st['thinned']['24']['own']['over_1p5N']:.2f} mm from those "
        f"presses alone; all {st['n_actions']} give {st['thinned']['all']['own']['over_1p5N']:.2f} mm "
        "(<code>scripts/ur5_tip_calibration_scaled_transfer.py</code>). Between tips 1 and 2, whose magnets match, tip 1's "
        "own GP missed its twins by 1.64 and 2.09 mm on the two sides, and with tip 2's presses added by 0.75 and 0.40 mm "
        "(Figure 8b). Tip 2's slides took the direction error of tip 1's shear "
        f"from {ps['id1_presses']['angle_deg']:.1f}° to {ps['id1_presses_and_id2_slides']['angle_deg']:.1f}°. The GP that "
        "corrects the physical estimate does not transfer either. Trained on tip 0, it raised tip 1's error from 2.06 to "
        "2.48 mm. Trained on 47 of tip 1's own presses, it gave 0.55 mm at 2 N and over ("
        + link(L["cb"], "checkerboard page") + ").</p>"
        + P.fig(8, F["transfer"], "(a) Location error on tip 2's grid at 1.5 N and over, for tip 2's physical model, a GP "
                "trained on tip 1, and tip 2's own GP over five folds. (b) Location error on tip 1's 18 twin presses at "
                "1.5 N and over, by side, for GPs trained on four sets of presses.")
    )
    S.append(("results", "Results on three tips", body))

    body = (
        "<p>The HEX-E is a weak reference. Its $F_z$ channel reads 0.78–0.87 of its other axes' gain in two gravity fits. "
        "Its link dropped out in three of the first night's runs and once on 2026-10-26. The arm, the mount and the "
        "indenter's stand are taken as rigid; the rod presses bound their compliance at 0.10 mm/N. The registration "
        "fixes the place along the finger less well than across it. The tips' magnets have 0.34–0.57 of an N52's "
        "strength. A press below 1 N therefore changes the readings by less than the step of about 7 µT that a press "
        "leaves at rest. Every model here is trained and scored on one cell, one indenter shape and one night's seating of the "
        "tip.</p>"
    )
    S.append(("limits", "Limitations", body))
    items = [
        "ATI Mini45 as the reference: calibrate it on the UR5 with the 100 g weight in its printed holder (39 stations, "
        "about 27 min; " + link(L["ati"], "ATI plan") + "), then repeat a grid of presses. A board force outside "
        "0.78–1.05 of the ATI's means the HEX-E's gain biased the comparisons.",
        "Torsion on the sensor's axis: clamp the finger with the face's normal along the force sensor's axis and twist at "
        "5 N, so the torsion is read without the push's moment. A correlation above 0.8 between the GP's torsion and "
        "the sensor's would make torsion a calibrated output.",
        "Flank coverage: add press targets on the flanks above the cover's walls until no reachable FE point lies more "
        "than 1 mm from a press (" + P.code("scripts/ur5_tip_calibration_density_curve.py") + " reports the share). "
        "The density curve predicts 0.3–0.6 mm there.",
        "N52 magnets: repeat cb2t's suite with N52 magnets in the checkerboard. The physical model should reach about "
        "1.4 mm untrained (transfer simulation), and the light presses should then move above the rest step.",
        "Random order: run one grid in random order, so that clock time and absolute temperature stop following the "
        "target, and score the temperature inputs again.",
    ]
    S.append(("future", "Future work", "<ul>" + "".join(f"<li>{t}</li>" for t in items) + "</ul>"))
    return S, lede, summary, film
