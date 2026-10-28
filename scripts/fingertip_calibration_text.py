"""Prose of the page on the fingertip's calibration on the UR5 (scripts/fingertip_calibration_page.py), kept apart from the
plotting. Numbers come from the page's numbers.json and from the sessions' analysis outputs named in the text; the UR5
coordinator's analysis files of v2 ID 1 and v3 ID 1 are read as committed at HEAD (at_head)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UR5D = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data"
V3D = ROOT / "docs/experiments/20261027-v3_tip_calibration_and_crosstalk/data"


def at_head(path):
    """A file as committed at HEAD, LFS pointers resolved: the coordinator may be rewriting its working copy mid-analysis."""
    rel = str(Path(path).resolve().relative_to(ROOT))
    blob = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"], capture_output=True, check=True).stdout
    if blob.startswith(b"version https://git-lfs.github.com/spec/v1"):
        blob = subprocess.run(["git", "-C", str(ROOT), "lfs", "smudge", "--", rel], input=blob, capture_output=True,
                              check=True).stdout
    return blob


def sections(N, F, P, link, L):
    """(sections, lede, summary, film_html)."""
    bl = N["by_level"]
    s2, c1, c2 = (bl[k] for k in ("20261022-210852-sphere2", "20261023-203938-cb1t", "20261025-152920-cb2t"))
    gp_hi = [s2["gp_over_1p5N"], c1["gp_over_1p5N"], c2["gp_over_1p5N"]]
    fn_err = [s2["gp_fn_err_N"], c1["gp_fn_err_N"], c2["gp_fn_err_N"]]
    cov = N["coverage_cb2t"]
    ps = N["pooling_shear"]
    tr = N["transfer"]
    # v1 tips' presses scaled by the magnets' strength fitted at rest, applied to v2 ID 1 (2026-10-27, commit 2bbedb0b)
    st = json.loads((UR5D / "20261026-192158-v2id1/scaled_transfer.json").read_text())
    # v2 ID 1 and v3 ID 1: the UR5 coordinator's analysis files, as committed at HEAD
    jl = lambda p: json.loads(at_head(p))  # noqa: E731
    TP = jl(UR5D / "20261027-v1_v2_tips.json")["tips"]
    HX = jl(UR5D / "20261027-hard_extrapolation_v2id1.json")
    SV = jl(UR5D / "20261027-suite_value_v2id1.json")
    TL = jl(UR5D / "20261027-143529-tallv2id1/r29_tall_vs_sphere.json")
    RD = jl(UR5D / "20261027-095303-rodv2id1/rod_tips.json")
    G = jl(V3D / "20261028-four_vs_five_grid.json")
    hx = HX["splits"]["2N"]["estimators"]
    HB = ("2-3.5", "3.5-5", "5-8")
    strict = lambda est, key, k=1: [k * hx[est]["test_strict"][b][key] for b in HB]  # noqa: E731
    rg = lambda xs, f="{:.2f}": f"{f.format(min(xs))}–{f.format(max(xs))}"  # noqa: E731
    g4, g5 = G["four_v3id1"], G["five_v2id1"]
    e4, e5 = g4["gp_direction"], g5["gp_direction"]
    acc = gp_hi + [TP["ID 3"]["own"]["over_1p5N"], TP["v2 ID 1"]["own"]["over_1p5N"]]
    n_hold = sum(v["press_holds"] + v["turn_holds"] for v in N["sessions"].values()) + G["holds"]["v3"]
    dloc, dfe = strict("gp_direction", "loc_mm"), strict("gp_direction", "force_rel_err", 100)
    bloc, bfe = strict("gp", "loc_mm"), strict("gp", "force_rel_err", 100)

    lede = (
        "Six printed fingertips were pressed onto 5 mm spheres and a rod by the lab's UR5. The reference force was "
        "measured by an OptoForce HEX-E force-torque sensor between the flange and the fingertip. In eleven sessions "
        f"they gave {n_hold:,} holds. Trained on a tip's own presses, a Gaussian process (GP) from the 15 "
        f"magnetometer readings locates presses at new points within {rg(acc)} mm at 1.5 N and over. On tips 0 to 2 it "
        f"gives their normal force within {min(fn_err):.2f}–{max(fn_err):.2f} N. On v2 ID 1, with five N52 magnets, a GP "
        "from the direction of the readings' change was trained on holds of 2 N and under. It places presses of 2 to 8 N "
        f"within {rg(dloc)} mm and {rg(dfe, '{:.0f}')} % of their force. With four N52 magnets (v3 ID 1) the signal per "
        f"newton is half that of five, {g4['gain_uT_per_N']['1-2']:.1f} against {g5['gain_uT_per_N']['1-2']:.1f} µT/N at 1–2 N. "
        f"The same GP places the two tips' contacts within {e4['all']['loc_mm']:.2f} and {e5['all']['loc_mm']:.2f} mm. Torsion is "
        "not recovered, and a model learned on one tip misses on another.")
    summary = (
        f"UR5 + HEX-E calibration of six printed magnetic fingertips ({n_hold:,} holds, 2026-10-22 to 10-28): a GP from the "
        f"15 readings locates new points within {rg(acc).replace('–', '-')} mm at 1.5 N and over on a tip's own presses "
        f"(normal force within {min(fn_err):.2f}-{max(fn_err):.2f} N on tips 0-2); on v2 ID 1 (five N52) a GP from the "
        f"direction of the change, trained at 2 N and under, places 2-8 N presses within {rg(dloc).replace('–', '-')} mm and "
        f"{rg(dfe, '{:.0f}').replace('–', '-')} % of the force (board GP {rg(bloc, '{:.1f}').replace('–', '-')} mm, "
        f"{rg(bfe, '{:.0f}').replace('–', '-')} %); four N52 magnets (v3 ID 1) give half the signal per newton "
        f"({g4['gain_uT_per_N']['1-2']:.1f} against {g5['gain_uT_per_N']['1-2']:.1f} uT/N) and the same calibrated "
        f"errors ({e4['all']['loc_mm']:.2f} and {e5['all']['loc_mm']:.2f} mm); torsion not recovered; a GP trained on "
        "another tip gives 1.42 mm against 0.39 mm.")
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
        ("Board GP, direction GP", "The board GP is the GP of Section 8, from the board change $\\mathbf d$ to the point "
         "and the force. In the direction GP the input is the change's direction $\\mathbf d/\\lVert\\mathbf d\\rVert$. "
         "It predicts the point and the force per unit change, and its force is that times $\\lVert\\mathbf d\\rVert$."),
        ("Signal per newton, noise-equivalent force", "The signal per newton is the median of "
         "$\\lVert\\mathbf d\\rVert/f_n$ over a tip's press holds, in µT/N. The record noise is the standard deviation of "
         "one board record during a hold, after a linear trend is removed, as a median over the 15 components. The "
         "noise-equivalent force is the record noise divided by the signal per newton, in mN."),
        ("Force error, force angle, normal angle", "The force error is the median of "
         "$\\bigl|\\lVert\\hat{\\mathbf f}\\rVert-\\lVert\\mathbf f\\rVert\\bigr|$ as a share of "
         "$\\lVert\\mathbf f\\rVert$, in %. The force angle is the median angle between the estimated and the measured force. "
         "The normal angle is the median angle between the pad's normals at the estimated and the registered point."),
        ("Compliance scale $c$", "A factor on every displacement of the FE basis, fitted to the board's change. The FE "
         "is linear in one TPU, so the fitted modulus is the FE's 12 MPa divided by $c$."),
        ("Pooling", "Training one model on the holds of more than one tip."),
        ("Tips 0 to 3, v2 ID 1, v3 ID 1", "Tip 0 has its magnets all one way (session sphere2). Tips 1 to 3 are the v1 "
         "tips ID 1 to ID 3, with the checkerboard of poles. The magnets of tips 0 to 3 have 0.20–0.57 of an N52's "
         "strength. v2 ID 1 has five N52 magnets in the checkerboard, at 0.98 of an N52's strength. v3 ID 1 has four N52 "
         "magnets in an alternating 5.0 mm square, under pad 05R4."),
    ]
    S.append(("terms", "Terms and metrics", '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms)
              + "</dl>"))

    body = (
        "<p>Six numbers describe the contact of a soft fingertip with an object. They are two coordinates of the point on "
        "its surface, the normal force, two components of shear and the torsion about the normal. The normal force is the "
        "grip. The ratio of shear to normal force, against the friction coefficient, is the margin to slip. The torsion "
        "resists a tool turning between two fingertips, as derived in " + link(L["contact"], "the contact overview")
        + ". The point says where the object lies in the hand.</p>"
        "<p>The board measures none of these. It measures the field of the pad's magnets, which a load moves. The physical "
        "model predicts the readings from finite elements and a dipole model. Untrained, it places presses 0.9–2.4 mm "
        "off at 2 N and over, and it fails below 1 N. The calibration therefore records the readings together with a "
        "reference for all six numbers, over the whole surface and the range of forces a grasp uses. A robot gives the "
        "point, and a six-axis sensor the force and torque.</p>"
    )
    S.append(("goal", "What the calibration must measure", body))

    body = (
        "<p>The fingertip, its DIP link and the calibration mount sit on the HEX-E, which sits on the UR5's flange. The "
        "finger points along the tool axis, so the face's normal lies across it. The indenter is a printed sphere of "
        "radius 5 mm on a post, on a 2x4 clamped to the table. A second fixture beside it has a rod of radius 12.5 mm "
        "(Figure 1). The pad's face centre lies 128.4 mm from the flange. By 2026-10-27 the cell had two more fixtures. "
        "On " + link(L["tall"], "the tall indenter") + ", a sphere of the same radius 160 mm above its 2x4, the robot "
        "reaches more of the pad's edges and flanks. Two more fingertips stand on the fingertip holder for "
        + link(L["crosstalk"], "the crosstalk map") + " of the v3 tips.</p>"
        + P.fig(1, F["cell"], "The cell as it stood on 2026-10-27, rendered in MuJoCo from the fixtures' registered poses. "
                "(a) The UR5 with the HEX-E, the mount and the fingertip, at a face touch-off on the low sphere. (b, c) The "
                "pad at the contact onset of a face and a flank touch-off of v2 ID 1's sphere session; the flange poses are "
                "those recorded at the onsets. Purple: the printed fixtures and the mount; black: the fingertips; green: "
                "their boards. Scene by " + P.code("scripts/ur5_cell_render.py") + ", stills by "
                + P.code("scripts/fingertip_calibration_cell_render.py") + ".")
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

    ses = N["sessions"]

    def srow(name, key, tip, mount, date):
        s = ses[key]
        return [name, tip, mount, date, f"{s['press_holds']:,} ({s['presses']})",
                f"{s['turn_holds']:,} ({s['turns']})" if s["turns"] else "–"]
    rows = [
        srow("sphere2", "20261022-210852-sphere2", "0, magnets one way", "v3", "2026-10-22 night"),
        srow("rod1", "20261023-104526-rod1-joint", "0, rod of radius 12.5 mm", "v3", "2026-10-23"),
        srow("cb1t", "20261023-203938-cb1t", "1, checkerboard", "v3", "2026-10-23 evening"),
        srow("cb1r", "20261025-114653-cb1r", "1, checkerboard", "v3", "2026-10-25"),
        srow("cb2t", "20261025-152920-cb2t", "2, checkerboard", "v7", "2026-10-25"),
        srow("cb3t", "20261026-134522-cb3t", "3, checkerboard", "v7", "2026-10-26"),
        srow("rod3t", "20261026-180938-rod3t", "3, rod", "v7", "2026-10-26"),
        srow("v2id1", "20261026-192158-v2id1", "v2 ID 1, five N52", "v7", "2026-10-26 night"),
        srow("rodv2id1", "20261027-095303-rodv2id1", "v2 ID 1, rod", "v7", "2026-10-27"),
        srow("tallv2id1", "20261027-143529-tallv2id1", "v2 ID 1, tall indenter", "v7", "2026-10-27"),
        ["spherev3id1", "v3 ID 1, four N52", "v7", "2026-10-27 night", f"{G['holds']['v3']:,} ({G['actions']}), grid stage",
         "running"],
    ]
    body = (
        "<p>Table 4 lists the sessions. Tip 0 and tip 1's first session ran early plans with few slides. From cb2t on, "
        "each tip ran the same suite of six passes on the low sphere, and from cb3t on presses to 8 N (R26). Tip 3 ran "
        "passes 1 and 2a, the actions past joint 6's cap and R26, and was then swapped for v2 ID 1. v2 ID 1 ran the whole "
        "suite, the rod and part of a pass on the tall indenter (Section 10). v3 ID 1's suite was running on 2026-10-28 "
        "(Section 11). Tip 2's session "
        "cb2t has 104 press points in 921 holds, and 87 slides. Its coverage is in Figure 3. The face "
        "and edges are covered within about 1.5 mm. The flanks are reached only above the cover's walls, and 74–79 % of "
        "their FE surface points lie more than 1.5 mm from any press.</p>"
        + P.table(["Session", "Tip, indenter", "Mount", "Date", "Press holds (presses)", "Turn holds (turns)"], rows,
                  caption="Sessions on the UR5. Turn holds are those of slides, rolls and twists. Counts from each session's "
                          "<code>holds.npz</code> in <code>docs/experiments/20261022-ur5_tip_calibration/data/</code>; v3 ID 1's "
                          "grid stage from the comparison of Section 11. The tall indenter's 26 slides of 2026-10-27 are not "
                          "analysed yet.", n=4)
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
    S.append(("results", "Results on tips 0, 1 and 2", body))

    # 10. v2 ID 1 (five N52): the coordinator's analyses of 2026-10-27
    cp = lambda rel: f"<code>docs/experiments/20261022-ur5_tip_calibration/data/{rel}</code>"  # noqa: E731
    REG = ("face", "edge", "flank")
    v2 = TP["v2 ID 1"]
    phys = [TP[k]["physical"]["under_1p5N"] for k in ("ID 1", "ID 2", "ID 3")]
    mloc, mfe = strict("mlp", "loc_mm"), strict("mlp", "force_rel_err", 100)
    ploc, pfe = strict("physical", "loc_mm"), strict("physical", "force_rel_err", 100)
    sens = [b["sensitivity_over_own_1_2N"] for b in HX["linearity"]["by_bin"].values()]
    rod = {k: RD["presses"][k]["all"] for k in ("rod", "sphere", "sphere_rod")}
    tt, lo, sa, cv = TL["transfer"], TL["loto"], TL["shear"], TL["coverage_1p5mm"]
    bd = lambda k: next(r["loc_mm"] for r in TL["by_distance"][k] if r["d_lo"] == 1.0 and r["band"] == "over_1p5N")  # noqa: E731
    sz = lambda k: [sa[k][r][2] for r in REG]  # noqa: E731
    sgn = lambda x: f"{x:.2f}".replace("-", "−")  # noqa: E731
    pl, tsp = SV["plan"], list(SV["times"]["turn_speed_deg_s"].values())
    th = pl["total_h"]
    t5 = P.table(["Estimator", "Tip 1", "Tip 2", "Tip 3", "v2 ID 1"], [
        [name] + [f"{TP[k][est]['over_1p5N']:.2f} / {TP[k][est]['under_1p5N']:.2f}" for k in ("ID 1", "ID 2", "ID 3", "v2 ID 1")]
        for est, name in (("own", "Board GP, own presses"), ("own_thinned", "Board GP, own presses thinned to 48 actions"),
                          ("others", "Board GP, the other tips' presses"), ("pooled", "Board GP, own and the other tips' presses"),
                          ("physical", "Physical model, untrained"))],
        caption="Median location error in mm at 1.5 N and over / under 1.5 N, on each tip's sphere presses, under five "
                "folds grouped by action. Thinned: the tip's own training set cut to 48 press actions, as many as tip 3 "
                "has. The other tips: a GP of the v1 tips other than the one scored, all three for v2 ID 1. The tips have "
                "98, 187, 60 and 198 press actions. Rows in " + cp("20261027-v1_v2_tips.json") + " ("
                + P.code("scripts/ur5_tip_calibration_v1_tips.py --v2") + ").", n=5)
    t6 = P.table(["Region", "Tall press actions (holds)", "Sphere GP (mm)", "With the other tall presses (mm)",
                  "Shear along the finger, tall / low sphere (N/N)", "FE points within 1.5 mm, sphere / with tall (%)"],
                 [[nm, f"{tt[r]['n_actions']} ({tt[r]['n_holds']})", f"{tt[r]['loc_mm_over_1p5N']:.2f}",
                   f"{lo[r]['loc_mm_over_1p5N']:.2f}", f"{sgn(sa['tall'][r][2])} / {sgn(sa['sphere'][r][2])}",
                   f"{100 * cv[r]['sphere']:.0f} / {100 * cv[r]['sphere_and_tall']:.0f}"]
                  for r, nm in zip(REG, ("Face", "Edges", "Flanks"))],
                 caption="v2 ID 1's presses on the tall indenter at 1.5 N and over, by region. Location error of a board GP "
                         "of the sphere presses, and of one with the other tall presses added, one tall press action left "
                         "out at a time. Shear along the finger: the median shear component along $z$ of the pad frame per "
                         "newton of normal force. FE points: the share of the region's FE surface points within 1.5 mm of a "
                         "press. Rows in " + cp("20261027-143529-tallv2id1/r29_tall_vs_sphere.json") + " ("
                         + P.code("scripts/ur5_tip_calibration_tall_r29.py") + ").", n=6)
    t7 = P.table(["Stage", "At v2 ID 1's settings (h)", "Cut suite (h)"],
                 [[nm, f"{pl['by_stage'][k]['h']:.2f}", f"{pl['by_stage'][k]['h_reduced']:.2f}"]
                  for k, nm in (("1 sphere pass 1", "Low sphere, pass 1"), ("2 sphere", "Low sphere, the other passes"),
                                ("3 rod", "Rod"), ("4 tall", "Tall indenter"))]
                 + [["Total", f"{th['as_run']:.1f}", f"{th['reduced']:.1f}"]],
                 caption="Robot time of each stage of v2 ID 1's suite at its program settings, from the timed actions of its "
                         "sessions, and after the cut. The tall stage is R29 as planned; v2 ID 1 stopped 37 of its actions "
                         "short of the end. Rows in " + cp("20261027-suite_value_v2id1.json") + ".", n=7)
    body = (
        f"<p>v2 ID 1 has five N52 magnets in the checkerboard; fitted at rest, they have {RD['k']['v2 ID 1']:.2f} of an "
        "N52's strength. On 2026-10-26 and 10-27 it ran the whole suite on the low sphere, two passes on the rod and part "
        "of a pass on the tall indenter (Table 4). Calibrated on its own presses, a board GP places new points within "
        f"{v2['own']['over_1p5N']:.2f} mm at 1.5 N and over and {v2['own']['under_1p5N']:.2f} mm under (Table 5). Tip 2, "
        f"the best v1 tip, gives {TP['ID 2']['own']['over_1p5N']:.2f} and {TP['ID 2']['own']['under_1p5N']:.2f} mm. v2 ID "
        f"1's force errs by {v2['own']['force_err_N_over_1p5N']:.2f} N at 1.5 N and over. Untrained, the physical model "
        f"places its presses under 1.5 N within {v2['physical']['under_1p5N']:.2f} mm, against {rg(phys, '{:.1f}')} mm on "
        f"the v1 tips. A GP of the v1 tips' presses misses v2 ID 1's contacts by {v2['others']['over_1p5N']:.2f} mm, so a "
        "v2 tip is calibrated on its own presses.</p>"
        + t5
        + "<p><b>Presses beyond the training force.</b> The estimators were trained on v2 ID 1's holds of 2 N and under. "
        "They were scored on its presses of 2 to 8 N, at actions outside training (Figure 9). The board GP misses those "
        f"presses by {rg(bloc)} mm and gives {rg(bfe, '{:.0f}')} % too little force. Far from its training holds a GP "
        f"returns toward the training mean (Section 8). The direction GP places the same presses within {rg(dloc)} mm and "
        f"their force within {rg(dfe, '{:.1f}')} %. The MLP ensemble gives the force within {rg(mfe, '{:.1f}')} %, and it "
        f"misses the point by {rg(mloc)} mm. The physical model misses by {rg(ploc)} mm and {rg(pfe, '{:.0f}')} %. Within "
        "the training range, at 1.5 N and over, the direction GP's location error is 0.17, 0.23 and 0.23 mm on the face, "
        "edges and flanks. The board GP's is 0.23, 0.32 and 0.49 mm there (" + link(L["v3"], "v3 plan") + "). From 0.25 to 8 N, v2 ID 1's "
        f"signal per newton stays within {min(sens):.2f}–{max(sens):.2f} of its value at 1–2 N.</p>"
        + P.fig(9, F["hard"], "Location and force error of v2 ID 1's estimators, trained on its press holds of 2 N and "
                "under. White band: five folds over actions within that range. Grey band: presses of 2 to 8 N at actions "
                "outside training. The number under each force bin is its count of holds. Rows in "
                + cp("20261027-hard_extrapolation_v2id1.json") + " ("
                + P.code("scripts/ur5_tip_calibration_hard_extrapolation.py") + ").")
        + "<p><b>Rod.</b> On the rod of radius 12.5 mm, a board GP was trained on v2 ID 1's own rod presses. It places new "
        f"points within {rod['rod']['loc_over_1p5N']:.2f} mm at 1.5 N and over and {rod['rod']['loc_under_1p5N']:.2f} mm under. Its "
        f"force errs by {rod['rod']['force_err_over_1p5N']:.2f} N and {rod['rod']['angle_over_1p5N']:.1f}°. A GP of the "
        f"sphere presses alone misses the rod's points by {rod['sphere']['loc_over_1p5N']:.2f} and "
        f"{rod['sphere']['loc_under_1p5N']:.2f} mm, and their force by {rod['sphere']['force_err_over_1p5N']:.2f} N. With "
        f"both indenters in training the error is {rod['sphere_rod']['loc_over_1p5N']:.2f} and "
        f"{rod['sphere_rod']['loc_under_1p5N']:.2f} mm (" + cp("20261027-095303-rodv2id1/rod_tips.json") + ").</p>"
        "<p><b>Tall indenter.</b> The tall indenter is a 5 mm sphere 160 mm above its 2x4, where the robot reaches more of "
        "the edges and flanks. Its pass 1 registered the tool at 0.054 mm rms over 21 touch-offs. R29 then pressed 45 "
        "points that had no sphere press within 1 mm, until the HEX-E dropped out at joint 6 +18.5°. Its resume added 12 "
        "presses and 26 of 50 slides before the tip was swapped. A board GP of the sphere presses misses the tall "
        f"indenter's points by {rg([tt[r]['loc_mm_over_1p5N'] for r in REG])} mm at 1.5 N and over (Table 6). With the "
        f"other tall presses in training it gives {rg([lo[r]['loc_mm_over_1p5N'] for r in REG])} mm. At 1.0–1.25 mm from "
        f"the nearest training press, the tall presses err by {bd('tall'):.2f} mm and the sphere's by {bd('sphere'):.2f} mm. "
        f"The tall indenter's forces have a shear along the finger of {rg(sz('tall'))} N per newton. On the low sphere it "
        f"is {sgn(min(sz('sphere')))} to {sgn(max(sz('sphere')))} N per newton.</p>"
        + t6
        + "<p><b>Twists.</b> In v2 ID 1's twists, half the board's out-and-back difference is 12 to 21 µT. Once the change "
        "from the force's own difference is removed, 5 to 8 µT remain. Their correlation with the HEX-E's torsion is 0.84, "
        "0.47 and 0.00 at the three points twisted most (" + cp("20261026-192158-v2id1/twist_board.json")
        + ", commit d92e0e03).</p>"
        f"<p><b>Suite time.</b> v2 ID 1's suite took {th['as_run']:.1f} h of robot time (Table 7). Four program changes "
        f"save {th['as_run'] - th['all_program']:.1f} h without changing the data. They are faster travel, turn waypoints "
        "every 1.4° in place of 0.5°, slide holds of 0.2 s and a 30 mm travel lift. Dropping the 23 turn-speed variants and the "
        "rod's 24 rate presses, and thinning the sphere's rate set and the tall twists, saves "
        f"{th['all_program'] - th['reduced']:.1f} h more. The robot turned at "
        f"{rg([v['mean_deg_s'] for v in tsp])}°/s for planned speeds of {min(v['planned_deg_s'] for v in tsp):.2f}–"
        f"{max(v['planned_deg_s'] for v in tsp):.1f}°/s, so the "
        "turn-speed variants differed little. Every press target, force level, hard press, slide and touch-off stays in "
        f"the cut suite of {th['reduced']:.1f} h. Without the slides the board GP places slide contacts within 0.96 mm and "
        "their shear's direction within 15.4°; with them, within 0.43 mm and 6.6°. Without the hard presses its error at "
        "3 N and over rises 48 % (" + link(L["v3"], "v3 plan") + ", "
        + P.code("scripts/ur5_tip_calibration_suite_value.py") + ").</p>"
        + t7
    )
    S.append(("v2id1", "Calibration of the five-magnet tip v2 ID 1", body))

    # 11. v3 ID 1 (four N52) against v2 ID 1 (five), the grid stage of 2026-10-27/28
    nef = [1000 * g["nef_N"]["1-2"] for g in (g4, g5)]
    gn = lambda k: (g4["gain_uT_per_N"][k], g5["gain_uT_per_N"][k])  # noqa: E731
    FE = jl(V3D / "20261028-fe_calibration.json")
    f4, f5 = FE["four_v3id1"], FE["five_v2id1"]
    dep = lambda f: [f["depth"][r]["slope_um_per_N"] for r in REG]  # noqa: E731
    fe_give = [f["depth"][r]["fe_give_um_per_N"] for f in (f4, f5) for r in REG]
    body = (
        "<p>v3 ID 1 has four N52 magnets in an alternating 5.0 mm square under pad 05R4. In the readout model of "
        + link(L["overview"], "the fingertip overview") + ", four magnets give 1.8 times the noise-equivalent force of "
        "five, 0.051 against 0.029 N. On 2026-10-27 from 22:52, 18 touch-offs on the low sphere registered v3 ID 1 at "
        "0.049 mm rms. Against v2 ID 1's registration, the sphere's frame moved 0.51 mm and 0.68°, and the pad's frame in "
        f"the mount 0.47 mm and 1.55°. Stage 2 of its suite began at 23:14. Its grid stage of {G['actions']} press actions "
        "ended at 00:56 on 2026-10-28, and the stage's slides, rolls and twists came next.</p>"
        f"<p>Both tips were scored on the same {G['actions']} press actions of the grid stage, paired by action, with "
        f"{G['holds']['v3']} holds each (Figure 10). With four magnets the signal per newton is about half: "
        f"{gn('1-2')[0]:.1f} against {gn('1-2')[1]:.1f} µT/N at 1 to 2 N. On the face it is {gn('face')[0]:.1f} against "
        f"{gn('face')[1]:.1f} µT/N, and on the flanks {gn('flank')[0]:.1f} against {gn('flank')[1]:.1f}. The record noise "
        f"is {g4['noise_uT']:.2f} and {g5['noise_uT']:.2f} µT. The noise-equivalent force at 1 to 2 N is therefore "
        f"{nef[0]:.0f} against {nef[1]:.0f} mN, the ratio of {nef[0] / nef[1]:.1f} predicted by the readout model.</p>"
        "<p>The calibrated errors stay alike. Under five folds grouped by action, the direction GP places contacts within "
        f"{e4['all']['loc_mm']:.2f} mm with four magnets and {e5['all']['loc_mm']:.2f} mm with five. Its force errs by "
        f"{100 * e4['all']['force_rel_err']:.1f} and {100 * e5['all']['force_rel_err']:.1f} %. Under 1 N the location errs "
        f"by {e4['0-1']['loc_mm']:.2f} and {e5['0-1']['loc_mm']:.2f} mm, and between 2 and 3.5 N by "
        f"{e4['2-3.5']['loc_mm']:.2f} and {e5['2-3.5']['loc_mm']:.2f} mm. On the flanks it errs by "
        f"{e4['flank']['loc_mm']:.2f} and {e5['flank']['loc_mm']:.2f} mm. The force angle is {e4['all']['angle_deg']:.1f} "
        f"and {e5['all']['angle_deg']:.1f}°, and the normal angle {e4['all']['normal_deg']:.1f} and "
        f"{e5['all']['normal_deg']:.1f}° (" + P.code(str((V3D / "20261028-four_vs_five_grid.json").relative_to(ROOT)))
        + ", " + P.code("scripts/ur5_tip_calibration_four_vs_five.py") + ").</p>"
        + P.fig(10, F["four_five"], "Four N52 magnets (v3 ID 1, orange) against five (v2 ID 1, blue) on the same press "
                f"actions of the low sphere's grid stage, {G['holds']['v3']} holds each. (a, b) Signal per newton by normal "
                "force and by region. (c, d) Location and force error of the direction GP, five folds grouped by action.")
        + "<p><b>Physical model of pad 05R4.</b> Pad 05R4's FE case and surface basis were built on 2026-10-28 on b_v2's "
        f"grid and depth, with the four magnets. On the same {FE['actions']} grid actions, the magnets' strength, the "
        f"board's offsets and the compliance scale $c$ were fitted to the holds up to {f4['fit']['max_n']:g} N (Table 8). "
        f"The fit gives $c$ = {f4['fit']['c']:.3f} with four magnets and {f5['fit']['c']:.3f} with five. The fitted "
        f"modulus is therefore {f4['fit']['E_fit_MPa']:.1f} and {f5['fit']['E_fit_MPa']:.1f} MPa, "
        f"against the FE's {FE['E_FE_MPa']:g} MPa. The registered indentation grows by {rg(dep(f4), '{:.0f}')} µm/N with "
        f"four magnets and {rg(dep(f5), '{:.0f}')} µm/N with five, more than twice the FE's "
        f"{rg(fe_give, '{:.0f}')} µm/N. The fitted scale does not improve the contact location. Under five folds grouped by "
        f"action, the physical model errs by {f5['cv']['table']['all']['loc_mm']:.2f} mm with five magnets "
        f"({f5['untrained']['all']['loc_mm']:.2f} mm untrained). With four it errs by "
        f"{f4['cv']['table']['all']['loc_mm']:.2f} mm ({f4['untrained']['all']['loc_mm']:.2f} mm untrained). With four "
        "magnets, the rest field leaves the magnets' strength and the board's offset along $x$ nearly interchangeable. A "
        "free fit puts the strength at its bound of 2.0 and the offset at 2.69 mm. With the offset held near v2 ID 1's "
        f"0.73 mm, it goes to its bound of {f4['rest']['dx']:.2f} mm and the strength to {f4['rest']['k']:.2f}.</p>"
        + P.table(["Tip, FE case", "Compliance scale $c$", "Fitted modulus (MPa)",
                   "Indentation, face / edges / flanks (µm/N)", "Location error, untrained (mm)",
                   "Location error, $c$ fitted (mm)", "Under 1 N, $c$ fitted (mm)"],
                  [[nm, f"{f['fit']['c']:.3f}", f"{f['fit']['E_fit_MPa']:.1f}",
                    " / ".join(f"{f['depth'][r]['slope_um_per_N']:.0f}" for r in REG),
                    f"{f['untrained']['all']['loc_mm']:.2f}", f"{f['cv']['table']['all']['loc_mm']:.2f}",
                    f"{f['cv']['table']['0-1']['loc_mm']:.2f}"]
                   for f, nm in ((f4, "v3 ID 1, b_05R4"), (f5, "v2 ID 1, b_v2"))],
                  caption=f"The physical model of each pad on the same {FE['actions']} grid actions, {f4['holds']} holds "
                          "each. The fitted modulus is the FE's 12 MPa divided by $c$. The indentation per newton is the "
                          f"slope of the registered indentation against the normal force; the FE gives {rg(fe_give, '{:.0f}')} "
                          "µm/N. Location errors are medians under five folds grouped by action. Rows in "
                          + P.code(str((V3D / "20261028-fe_calibration.json").relative_to(ROOT))) + " ("
                          + P.code("scripts/ur5_tip_calibration_fe_calibration.py") + ").", n=8)
    )
    S.append(("v3id1", "Four N52 magnets against five on v3 ID 1", body))

    body = (
        "<p>The HEX-E is a weak reference. Its $F_z$ channel reads 0.78–0.87 of its other axes' gain in two gravity fits. "
        "Its link dropped out in three of the first night's runs, once on 2026-10-26 and once in R29 on 10-27. The arm, the "
        "mount and the indenter's stand are taken as rigid; the rod presses bound their compliance at 0.10 mm/N. The "
        "registration fixes the place along the finger less well than across it. The magnets of tips 0 to 3 have "
        "0.20–0.57 of an N52's strength. A press below 1 N therefore changes their readings by less than the step of about "
        "7 µT that a press leaves at rest. The tall indenter's forces have a shear along the finger that the low sphere's "
        "lack. It comes from the finger or from the indenter's neck; the sphere check that would separate the two did not "
        "run before v2 ID 1 came off. v3 ID 1's results cover its grid stage alone. Every model here is trained and scored "
        "on one cell and one seating of each tip.</p>"
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
        "The density curve predicts 0.3–0.6 mm there. On v2 ID 1 the tall indenter took the flanks' FE points within "
        f"1.5 mm of a press from {100 * cv['flank']['sphere']:.0f} to {100 * cv['flank']['sphere_and_tall']:.0f} %.",
        "Six shared points: v3 ID 1's stage 4 presses six points on the low sphere and on the tall indenter in one "
        "session (" + link(L["v3"], "v3 plan") + "). If their errors agree within their repeat spread, the tall indenter "
        "alone covers the face, edges and flanks, and the sphere stage is dropped; a larger difference keeps both.",
        "Four-magnet rest fit: measure the board's offset in v3 ID 1's DIP link, hold it, and refit the magnets' strength "
        "at rest (" + P.code("scripts/ur5_tip_calibration_fe_calibration.py") + "). A physical-model error on the grid "
        f"still above v2 ID 1's {f5['cv']['table']['all']['loc_mm']:.2f} mm would place the fault in the four-magnet "
        "field model.",
        "Random order: run one grid in random order, so that clock time and absolute temperature stop following the "
        "target, and score the temperature inputs again.",
    ]
    S.append(("future", "Future work", "<ul>" + "".join(f"<li>{t}</li>" for t in items) + "</ul>"))
    return S, lede, summary, film
