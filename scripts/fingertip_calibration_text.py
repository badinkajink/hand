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
    # the first 40 slides of v3 ID 1's stage 2, before the HEX-E fell off its DAQ at 02:54 (feed line of 03:13)
    SL = jl(V3D / "20261028-four_vs_five_slides40.json")["slides"]
    # v3 ID 1's 12 twists of 2026-10-28, 09:41 to 10:25, against v2 ID 1's of the same ids (feed line of 12:25)
    TW = jl(V3D / "20261028-four_vs_five_twists.json")["twists"]
    tw4, tw5 = TW["four_v3id1"], TW["five_v2id1"]
    # the face centre's error with the press targets in folds, four magnets against five (owner, 10-28 13:4x; feed line
    # of 13:48)
    CC = jl(V3D / "20261028-centre_coverage_by_target.json")
    # the rod: v3 ID 1's stage 2 and six morning presses of 10-28 against v2 ID 1's rod passes of 10-27 (feed line of 15:14)
    RO = jl(V3D / "20261028-four_vs_five_rod.json")
    # press coverage of the pad by region (owner, 10-28, on the density of coverage; scripts/ur5_tip_calibration_coverage.py)
    CV = jl(V3D / "20261028-press_coverage_by_region.json")
    # v3 ID 2, the second four-magnet tip, against v3 ID 1 on the press actions both ran on each indenter, and v3 ID 2's
    # top-edge pass on the low sphere (feed lines of 2026-10-29 09:46 and 10:00)
    FX = ("sphere", "rod", "tall")
    TT = {k: jl(V3D / f"20261029-tip_vs_tip_{k}_v3id2_v3id1.json") for k in FX}
    GR = {k: jl(V3D / f"20261029-gain_by_region_{k}.json") for k in FX}
    TE = jl(V3D / "20261029-top_edge_gp_v3id2.json")
    # four magnets against five, v3 ID 2 against v2 ID 1 on each indenter's shared actions (feed line of 10-29 10:28), and
    # v3 ID 1 against v2 ID 1 on the tall indenter (feed line of 10:29)
    G2 = jl(V3D / "20261029-four_vs_five_grid_v3id2.json")
    F2 = dict(sphere=G2, rod=jl(V3D / "20261029-four_vs_five_rod_v3id2.json"),
              tall=jl(V3D / "20261029-four_vs_five_tall_v3id2.json"))
    TA = jl(V3D / "20261029-four_vs_five_tall_v3id1.json")
    # what each motion adds on v3 ID 1 and ID 2: direction GPs with one kind of hold left out of training, on both tips'
    # sessions on the three indenters (feed line of 10-29 12:45); v3 ID 2's twists are in its four-against-five file
    MV = jl(V3D / "20261029-motion_value.json")
    T3 = ("v3 ID 1", "v3 ID 2")
    tw2 = G2["twists"]["four_v3id2"]
    C2, V2S, V2T, V2R, V3S = ("cb2t (10-25)", "v2 ID 1 sphere", "v2 ID 1 sphere + tall", "v2 ID 1 sphere + tall + rod",
                              "v3 ID 1 sphere so far")
    # rounded half up, as the coordinator quotes them (0.885 is 89 %)
    cvp = lambda k, r, d=10: str(int(100 * CV[k][r]["within_1mm" if d == 10 else "within_1p5mm"] + 0.5 + 1e-9))  # noqa: E731
    # sphere presses held out with the presses near them left out of training, at 1.5 N and over, by distance bin
    bdb = lambda lo: next(r["loc_mm"] for r in TL["by_distance"]["sphere"]  # noqa: E731
                          if r["d_lo"] == lo and r["band"] == "over_1p5N")
    hx = HX["splits"]["2N"]["estimators"]
    HB = ("2-3.5", "3.5-5", "5-8")
    strict = lambda est, key, k=1: [k * hx[est]["test_strict"][b][key] for b in HB]  # noqa: E731
    rg = lambda xs, f="{:.2f}": (f.format(min(xs)) if f.format(min(xs)) == f.format(max(xs))  # noqa: E731
                                 else f"{f.format(min(xs))}–{f.format(max(xs))}")
    g4, g5 = G["four_v3id1"], G["five_v2id1"]
    e4, e5 = g4["gp_direction"], g5["gp_direction"]
    ro4, ro5 = RO["four_v3id1"]["gp_direction"], RO["five_v2id1"]["gp_direction"]
    ta4, ta5 = TA["four_v3id1"]["gp_direction"], TA["five_v2id1"]["gp_direction"]
    RT = ((RO, "four_v3id1"), (RO, "five_v2id1"), (TA, "four_v3id1"), (TA, "five_v2id1"))
    acc = gp_hi + [TP["ID 3"]["own"]["over_1p5N"], TP["v2 ID 1"]["own"]["over_1p5N"]]
    # the v3 tips' sessions as Section 13's analysis takes them: presses, slides, twists and rolls on each indenter
    mvh = lambda t, k: sum(MV["holds"][t][k][a] for a in ("press", "slide", "twist", "roll"))  # noqa: E731
    n_hold = (sum(v["press_holds"] + v["turn_holds"] for v in N["sessions"].values())
              + sum(mvh(t, k) for t in T3 for k in FX))
    t2 = {k: TT[k]["tips"]["v3 ID 2"]["gp_direction"] for k in FX}
    t1 = {k: TT[k]["tips"]["v3 ID 1"]["gp_direction"] for k in FX}
    tg = lambda k, t: TT[k]["tips"][t]["gain_uT_per_N"]["1-2"]  # noqa: E731
    T5 = ("four_v3id2", "five_v2id1")
    rt2 = lambda k: F2[k]["five_v2id1"]["gain_uT_per_N"]["1-2"] / F2[k]["four_v3id2"]["gain_uT_per_N"]["1-2"]  # noqa: E731
    f2 = lambda k, t: F2[k][t]["gp_direction"]["all"]["loc_mm"]  # noqa: E731
    mf = lambda t: MV["tips"][t]["fixtures"]  # noqa: E731
    wo = [mf(t)["without_it"][k]["loc_mm"] for t in T3 for k in FX]
    own = [mf(t)["own_cv"][k]["loc_mm"] for t in T3 for k in FX]
    xf = lambda a, b: MV["transfer"][f"{a} -> {b}"]["presses"]["all"]  # noqa: E731
    pall = lambda t: mf(t)["pooled_cv"]["all"]["loc_mm"]  # noqa: E731
    dloc, dfe = strict("gp_direction", "loc_mm"), strict("gp_direction", "force_rel_err", 100)
    bloc, bfe = strict("gp", "loc_mm"), strict("gp", "force_rel_err", 100)

    lede = (
        "Seven printed fingertips were pressed onto 5 mm spheres and a rod by the lab's UR5. The reference force was "
        "measured by an OptoForce HEX-E force-torque sensor between the flange and the fingertip. In sixteen sessions "
        f"they gave {n_hold:,} holds. Trained on a tip's own presses, a Gaussian process (GP) from the 15 "
        f"magnetometer readings locates presses at new points within {rg(acc)} mm at 1.5 N and over. On tips 0 to 2 it "
        f"gives their normal force within {min(fn_err):.2f}–{max(fn_err):.2f} N. On v2 ID 1, with five N52 magnets, a GP "
        "from the direction of the readings' change was trained on holds of 2 N and under. It places presses of 2 to 8 N "
        f"within {rg(dloc)} mm and {rg(dfe, '{:.0f}')} % of their force. With four N52 magnets (v3 ID 1) the signal per "
        f"newton is half that of five, {g4['gain_uT_per_N']['1-2']:.1f} against {g5['gain_uT_per_N']['1-2']:.1f} µT/N at 1–2 N. "
        f"The same GP places the two tips' contacts within {e4['all']['loc_mm']:.2f} and {e5['all']['loc_mm']:.2f} mm on the "
        f"low sphere, and within {ro4['all']['loc_mm']:.2f} and {ro5['all']['loc_mm']:.2f} mm on the rod. A second "
        f"four-magnet tip, v3 ID 2, matches v3 ID 1 on the presses both ran. On the sphere its signal per newton is "
        f"{tg('sphere', 'v3 ID 2'):.1f} against {tg('sphere', 'v3 ID 1'):.1f} µT/N, and on the rod both tips' contacts "
        f"are placed within {max(t2['rod']['all']['loc_mm'], t1['rod']['all']['loc_mm']):.2f} mm. With five magnets the "
        f"signal per newton is {rg([rt2(k) for k in FX])} times its own on all three indenters. The tip end was first "
        f"pressed on v3 ID 2, in a pass of {TE['pass_presses']} presses. With the pass's other presses in training, "
        f"they are placed within {TE['with_the_pass']['all']['loc_mm']:.2f} mm at the median, against "
        f"{TE['grid_only']['all']['loc_mm']:.2f} mm from the grid alone. Each indenter's presses, the tip end and the "
        "slides must all be in a tip's training data. Without its own presses, an indenter's contacts are placed "
        f"{min(wo):.2f}–{max(wo):.2f} mm off, against {min(own):.2f}–{max(own):.2f} mm with them. Torsion is not "
        "recovered. On the twists of three tips, a GP's estimate correlates with the measured torsion at "
        f"{rg([tw['gp_twists']['r'] for tw in (tw2, tw4, tw5)])}. A model learned on "
        "one tip misses on another. Between the two four-magnet tips, presses are placed "
        f"{xf(*T3)['loc_mm']:.2f} and {xf(*T3[::-1])['loc_mm']:.2f} mm off by the other tip's GP, against "
        f"{pall(T3[1]):.2f} and {pall(T3[0]):.2f} mm by the tip's own.")
    summary = (
        f"UR5 + HEX-E calibration of seven printed magnetic fingertips ({n_hold:,} holds, 2026-10-22 to 10-29): a GP from the "
        f"15 readings locates new points within {rg(acc).replace('–', '-')} mm at 1.5 N and over on a tip's own presses "
        f"(normal force within {min(fn_err):.2f}-{max(fn_err):.2f} N on tips 0-2); on v2 ID 1 (five N52) a GP from the "
        f"direction of the change, trained at 2 N and under, places 2-8 N presses within {rg(dloc).replace('–', '-')} mm and "
        f"{rg(dfe, '{:.0f}').replace('–', '-')} % of the force (board GP {rg(bloc, '{:.1f}').replace('–', '-')} mm, "
        f"{rg(bfe, '{:.0f}').replace('–', '-')} %); four N52 magnets (v3 ID 1) give half the signal per newton "
        f"({g4['gain_uT_per_N']['1-2']:.1f} against {g5['gain_uT_per_N']['1-2']:.1f} uT/N) and the same calibrated "
        f"errors ({e4['all']['loc_mm']:.2f} and {e5['all']['loc_mm']:.2f} mm on the sphere, {ro4['all']['loc_mm']:.2f} and "
        f"{ro5['all']['loc_mm']:.2f} mm on the rod, {ta4['all']['loc_mm']:.2f} and {ta5['all']['loc_mm']:.2f} mm on the tall "
        "indenter); a second four-magnet tip, v3 ID 2, matches v3 ID 1 on the presses "
        f"both ran ({tg('sphere', 'v3 ID 2'):.1f} against {tg('sphere', 'v3 ID 1'):.1f} uT/N; the five-magnet signal "
        f"{rg([rt2(k) for k in FX]).replace('–', '-')} times its own on the three indenters; "
        f"{t2['sphere']['all']['loc_mm']:.2f} and {t1['sphere']['all']['loc_mm']:.2f} mm on the sphere, "
        f"{t2['rod']['all']['loc_mm']:.2f} and {t1['rod']['all']['loc_mm']:.2f} mm on the rod), and its tip end is placed "
        f"within {TE['with_the_pass']['all']['loc_mm']:.2f} mm with the pass's other presses in training "
        f"({TE['grid_only']['all']['loc_mm']:.2f} mm from the grid alone); each indenter's presses, the tip end and "
        f"slides are needed in a tip's training (an indenter left out: {min(wo):.2f}-{max(wo):.2f} mm against "
        f"{min(own):.2f}-{max(own):.2f} mm with its own); torsion not recovered (GP r {tw4['gp_twists']['r']:.2f}, "
        f"{tw5['gp_twists']['r']:.2f} and {tw2['gp_twists']['r']:.2f} on the twists); a GP trained on another tip gives "
        f"1.42 mm against 0.39 mm on the v1 tips and {xf(*T3)['loc_mm']:.2f} and {xf(*T3[::-1])['loc_mm']:.2f} mm against "
        f"{pall(T3[1]):.2f} and {pall(T3[0]):.2f} mm between the four-magnet tips.")
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
        ("Region, press target, coverage", "The regions split the 812 points of the FE pad's surface basis by their "
         "normal: the tip end where $n_z>0.7$, the flanks where $|n_y|>0.7$, the face where $n_x>0.9$ and the edges where "
         "$0.5<n_x\\le 0.9$. A press target is the mean contact point of one press action's holds. The coverage of a "
         "region is the share of its points within 1 mm, or 1.5 mm, of a target."),
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
        "The session cb2t of 2026-10-25 ran the full suite in six passes on tip 2 (Table 3). What each motion adds to "
        "the calibration of the four-magnet tips is measured in Section 13.</p>"
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

    def v3row(name, t, k, tip, date):
        h, f = MV["holds"][t][k], MV["tips"][t]
        turns = ((f["slides"][k]["slide_actions"], "slides"), (f["time"][k].get("twist", {}).get("actions", 0), "twists"),
                 (f["time"][k].get("roll", {}).get("actions", 0), "rolls"))
        return [name, tip, "v7", date, f"{h['press']:,} ({f['fixtures']['press_actions'][k]})",
                f"{h['slide'] + h['twist'] + h['roll']:,} (" + ", ".join(f"{n} {w}" for n, w in turns if n) + ")"]
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
        v3row("spherev3id1", "v3 ID 1", "sphere", "v3 ID 1, four N52", "2026-10-27 night and 10-28"),
        v3row("rodv3id1", "v3 ID 1", "rod", "v3 ID 1, rod", "2026-10-28"),
        v3row("tallv3id1", "v3 ID 1", "tall", "v3 ID 1, tall indenter", "2026-10-28"),
        v3row("spherev3id2", "v3 ID 2", "sphere", "v3 ID 2, four N52", "2026-10-28 night"),
        v3row("tallv3id2", "v3 ID 2", "tall", "v3 ID 2, tall indenter", "2026-10-28 and 10-29"),
        v3row("rodv3id2", "v3 ID 2", "rod", "v3 ID 2, rod", "2026-10-29"),
    ]
    assert len(rows) == 16  # the lede's "sixteen sessions"
    t5 = P.table(["Sessions and indenters", "Press actions"]
                 + [f"{nm} ({CV[C2][r]['n']} points)" for r, nm in (("face", "Face"), ("edge", "Edges"), ("flank", "Flanks"))],
                 [[nm, str(CV[k]["targets"])] + [f"{cvp(k, r)} / {cvp(k, r, 15)}" for r in ("face", "edge", "flank")]
                  for k, nm in ((C2, "cb2t (tip 2), low sphere"), (V2S, "v2 ID 1, low sphere"),
                                (V2T, "v2 ID 1, low sphere and tall indenter"),
                                (V2R, "v2 ID 1, low sphere, tall indenter and rod"),
                                (V3S, "v3 ID 1, the low sphere's grid stage"))],
                 caption="Coverage of the pad's regions: the share of a region's points within 1 mm and within 1.5 mm of a "
                         f"press target, in %. No target lies within 1.5 mm of the tip end's {CV[C2]['tip']['n']} points. "
                         "Rows in " + P.code("docs/experiments/20261027-v3_tip_calibration_and_crosstalk/data/"
                                             "20261028-press_coverage_by_region.json")
                         + " (" + P.code("scripts/ur5_tip_calibration_coverage.py") + ").", n=5)
    body = (
        "<p>Table 4 lists the sessions. Tip 0 and tip 1's first session ran early plans with few slides. From cb2t on, "
        "each tip ran the same suite of six passes on the low sphere, and from cb3t on presses to 8 N (R26). Tip 3 ran "
        "passes 1 and 2a, the actions past joint 6's cap and R26, and was then swapped for v2 ID 1. v2 ID 1 ran the whole "
        "suite, the rod and part of a pass on the tall indenter (Section 10). v3 ID 1's stage 2 stopped at 02:54 on "
        "2026-10-28, after 40 of its 84 slides, when the HEX-E fell off its acquisition. The stage's rolls and twists ran "
        "from 09:41, once the HEX-E was back, and the rod's session from 10:27 (Section 11). v3 ID 2 ran its suite of "
        "488 actions from 18:44 on 2026-10-28 to 08:13 on 10-29 (Section 12). Tip 2's session "
        "cb2t has 104 press points in 921 holds, and 87 slides.</p>"
        + P.table(["Session", "Tip, indenter", "Mount", "Date", "Press holds (presses)", "Turn holds (turns)"], rows,
                  caption="Sessions on the UR5. Turn holds are those of slides, rolls and twists. Counts from each session's "
                          "<code>holds.npz</code> in <code>docs/experiments/20261022-ur5_tip_calibration/data/</code>. The "
                          "v3 tips' counts are those of Section 13's analysis ("
                          + P.code("data/20261029-motion_value.json") + " in the v3 folder). v3 ID 2's presses on the "
                          "low sphere include its top-edge pass. Sections 11 and 12 compare the press actions that two "
                          "tips both ran. v2 ID 1's 26 slides on the tall indenter, of 2026-10-27, are not analysed "
                          "yet.", n=4)
        + f"<p><b>Coverage of the pad.</b> With the tall indenter and the rod, {cvp(V2R, 'face')} % of v2 ID 1's face "
        "lies within 1 mm of a press target (Figure 3). On the low sphere alone the share is "
        f"{cvp(V2S, 'face')} %, as on cb2t ({cvp(C2, 'face')} %, Table 5). With the tall indenter's "
        f"presses added it is {cvp(V2T, 'face')} %. Within 1.5 mm, the face's share rose from {cvp(V2S, 'face', 15)} to "
        f"{cvp(V2T, 'face', 15)} % with the tall indenter. On the edges, the share within 1 mm rose from "
        f"{cvp(V2S, 'edge')} to {cvp(V2T, 'edge')} % with the tall indenter and {cvp(V2R, 'edge')} % with the rod. The "
        "flanks are pressed only above "
        f"the cover's walls. Of their points, {cvp(V2S, 'flank', 15)} % lie within 1.5 mm of a target on the low sphere, "
        f"and {cvp(V2R, 'flank', 15)} % with all three indenters. No target lies within 1.5 mm of the tip end, whose "
        f"median point is {CV[V2R]['tip']['median_mm']:.2f} mm from the nearest target. After v3 ID 1's grid stage, "
        f"{cvp(V3S, 'face')} % of the face lies within 1 mm of a target.</p>"
        "<p>The location error grows with the distance to the nearest training press. For 60 of v2 ID 1's sphere press "
        "actions, a board GP was trained without the sphere presses within 1 to 3 mm of the action. At 1.5 N and over, those "
        f"actions' contacts are placed within {bdb(1.0):.2f} mm at 1.0–1.25 mm from the nearest training press. The error "
        f"is {rg([bdb(1.25), bdb(1.5)])} mm at 1.25–2 mm. It is {bdb(2.0):.2f} mm at 2–3 mm. Beyond 3 mm, where the "
        f"median flank point lies, it is {bdb(3.0):.2f} mm (rows in "
        + P.code("docs/experiments/20261022-ur5_tip_calibration/data/20261027-143529-tallv2id1/r29_tall_vs_sphere.json")
        + "). cb2t's curve is in Figure 6.</p>"
        + P.fig(3, F["coverage"], "Press targets of v2 ID 1 on the pad, by indenter. Light grey: FE surface points within "
                "1 mm of a target; mid grey: 1 to 1.5 mm; dark grey: farther. The flanks' lower part lies behind the "
                "cover's walls. The tip end lies under the cap, and its rim forms the top rows of (a). Drawn from each "
                "session's <code>holds.npz</code>.")
        + t5
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
    tt, lo, sa = TL["transfer"], TL["loto"], TL["shear"]
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
                + P.code("scripts/ur5_tip_calibration_v1_tips.py --v2") + ").", n=6)
    t6 = P.table(["Region", "Tall press actions (holds)", "Sphere GP (mm)", "With the other tall presses (mm)",
                  "Shear along the finger, tall / low sphere (N/N)"],
                 [[nm, f"{tt[r]['n_actions']} ({tt[r]['n_holds']})", f"{tt[r]['loc_mm_over_1p5N']:.2f}",
                   f"{lo[r]['loc_mm_over_1p5N']:.2f}", f"{sgn(sa['tall'][r][2])} / {sgn(sa['sphere'][r][2])}"]
                  for r, nm in zip(REG, ("Face", "Edges", "Flanks"))],
                 caption="v2 ID 1's presses on the tall indenter at 1.5 N and over, by region. Location error of a board GP "
                         "of the sphere presses, and of one with the other tall presses added, one tall press action left "
                         "out at a time. Shear along the finger: the median shear component along $z$ of the pad frame per "
                         "newton of normal force. Rows in " + cp("20261027-143529-tallv2id1/r29_tall_vs_sphere.json") + " ("
                         + P.code("scripts/ur5_tip_calibration_tall_r29.py") + ").", n=7)
    t7 = P.table(["Stage", "At v2 ID 1's settings (h)", "Cut suite (h)"],
                 [[nm, f"{pl['by_stage'][k]['h']:.2f}", f"{pl['by_stage'][k]['h_reduced']:.2f}"]
                  for k, nm in (("1 sphere pass 1", "Low sphere, pass 1"), ("2 sphere", "Low sphere, the other passes"),
                                ("3 rod", "Rod"), ("4 tall", "Tall indenter"))]
                 + [["Total", f"{th['as_run']:.1f}", f"{th['reduced']:.1f}"]],
                 caption="Robot time of each stage of v2 ID 1's suite at its program settings, from the timed actions of its "
                         "sessions, and after the cut. The tall stage is R29 as planned; v2 ID 1 stopped 37 of its actions "
                         "short of the end. Rows in " + cp("20261027-suite_value_v2id1.json") + ".", n=8)
    body = (
        f"<p>v2 ID 1 has five N52 magnets in the checkerboard; fitted at rest, they have {RD['k']['v2 ID 1']:.2f} of an "
        "N52's strength. On 2026-10-26 and 10-27 it ran the whole suite on the low sphere, two passes on the rod and part "
        "of a pass on the tall indenter (Table 4). Calibrated on its own presses, a board GP places new points within "
        f"{v2['own']['over_1p5N']:.2f} mm at 1.5 N and over and {v2['own']['under_1p5N']:.2f} mm under (Table 6). Tip 2, "
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
        f"indenter's points by {rg([tt[r]['loc_mm_over_1p5N'] for r in REG])} mm at 1.5 N and over (Table 7). With the "
        f"other tall presses in training it gives {rg([lo[r]['loc_mm_over_1p5N'] for r in REG])} mm. At 1.0–1.25 mm from "
        f"the nearest training press, the tall presses err by {bd('tall'):.2f} mm and the sphere's by {bd('sphere'):.2f} mm. "
        f"The tall indenter's forces have a shear along the finger of {rg(sz('tall'))} N per newton. On the low sphere it "
        f"is {sgn(min(sz('sphere')))} to {sgn(max(sz('sphere')))} N per newton.</p>"
        + t6
        + "<p><b>Twists.</b> In v2 ID 1's twists, half the board's out-and-back difference is 12 to 21 µT. Once the change "
        "from the force's own difference is removed, 5 to 8 µT remain. Their correlation with the HEX-E's torsion is 0.84, "
        "0.47 and 0.00 at the three points twisted most (" + cp("20261026-192158-v2id1/twist_board.json")
        + ", commit d92e0e03).</p>"
        f"<p><b>Suite time.</b> v2 ID 1's suite took {th['as_run']:.1f} h of robot time (Table 8). Four program changes "
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
    sl4, sl5 = (lambda k: SL["four_v3id1"][k]), (lambda k: SL["five_v2id1"][k])
    f4, f5 = FE["four_v3id1"], FE["five_v2id1"]
    dep = lambda f: [f["depth"][r]["slope_um_per_N"] for r in REG]  # noqa: E731
    # the refits to the presses from each rest fit (feed line of 2026-10-28 02:44)
    ps4 = lambda k: [v[k] for n, v in FE["prior_scan"].items() if n.startswith("four")]  # noqa: E731
    ps5 = lambda k: [v[k] for n, v in FE["prior_scan"].items() if n.startswith("five")]  # noqa: E731
    fe_give = [f["depth"][r]["fe_give_um_per_N"] for f in (f4, f5) for r in REG]
    # the direction GP by region and force, and on the rest of the face by the distance to the nearest training target
    c4, c5 = CC["four_v3id1"]["gp_direction"], CC["five_v2id1"]["gp_direction"]
    cb = lambda c, reg, b, key="loc_mm": c["by_force"][f"{reg} {b} N"][key]  # noqa: E731
    cn = lambda c, b: c["by_nearest_training_press"][f"rest_of_face nearest {b} mm"]  # noqa: E731
    CB = ("0-1", "1-2", "2-8")
    n_cen = sum(cb(c4, "centre", b, "n") for b in CB)
    cen_near = [k for k in c4["by_nearest_training_press"] if k.startswith("centre")]
    assert cen_near == ["centre nearest 1-1.5 mm"] and c4["by_nearest_training_press"][cen_near[0]]["n"] == n_cen
    n_cc = sum(r["n"] for r in c4["by_force"].values())
    near5 = [cn(c5, b)["loc_mm"] for b in ("0.5-1", "1-1.5", "1.5-2.5")]
    assert n_cc == sum(r["n"] for r in c5["by_force"].values()) and n_cen == sum(cb(c5, "centre", b, "n") for b in CB)
    body = (
        "<p>v3 ID 1 has four N52 magnets in an alternating 5.0 mm square under pad 05R4. In the readout model of "
        + link(L["overview"], "the fingertip overview") + ", four magnets give 1.8 times the noise-equivalent force of "
        "five, 0.051 against 0.029 N. On 2026-10-27 from 22:52, 18 touch-offs on the low sphere registered v3 ID 1 at "
        "0.049 mm rms. Against v2 ID 1's registration, the sphere's frame moved 0.51 mm and 0.68°, and the pad's frame in "
        f"the mount 0.47 mm and 1.55°. Stage 2 of its suite began at 23:14. Its grid stage of {G['actions']} press actions "
        "ended at 00:56 on 2026-10-28, and the stage's slides came next. At 02:54 the HEX-E fell off its acquisition after 40 "
        "of the 84 slides, and the stage stopped. The owner reseated the sensor's connector, and it read again from 09:33. "
        "From 09:41 to 10:25 the stage's 2 rolls, 12 twists and 4 final reference presses ran. The touch-offs on the rod "
        "and on the tall indenter followed, from 10:27 to 12:21. Stage 2 on the rod then ran all 94 of its actions, and "
        "six more presses ended at 13:52. Stage 2 on the tall indenter came next. The HEX-E stopped again at 14:32, after "
        "44 of its 102 actions. The other 58 resumed at 14:56. The sphere's remaining slides are still to run.</p>"
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
        + "<p><b>Error at the face centre.</b> The face centre is the apex over the board's centre chip U1, at "
        f"$(y, z) = ({CC['apex_yz_mm'][0]:g}, {CC['apex_yz_mm'][1]:g})$ mm in the pad frame. Face holds whose contact lies "
        f"within 2 mm of it count as central. Both tips ran {CC['actions']} press actions of the same ids on the low sphere, "
        f"{n_cc:,} holds each, and {n_cen} of each tip's holds are central. For Table 9 the press targets were split into "
        "five folds, and each fold was left out of training in turn. Each estimate is therefore of a point that no "
        "training hold pressed. The folds by action of Figure 10 keep the other actions at the same point in training. "
        f"Under 1 N central holds are placed by the direction GP within {cb(c4, 'centre', '0-1'):.2f} mm with four magnets "
        f"and {cb(c5, 'centre', '0-1'):.2f} mm with five. From 1 N up, four magnets give "
        f"{rg([cb(c4, 'centre', b) for b in CB[1:]])} mm. With five the error is "
        f"{rg([cb(c5, 'centre', b) for b in CB[1:]])} mm. On the rest of the face, under 1 N, four magnets give "
        f"{cb(c4, 'rest_of_face', '0-1'):.2f} mm against {cb(c5, 'rest_of_face', '0-1'):.2f} mm with five. At the centre "
        f"the board changes by {rg([cb(c4, 'centre', b, 'gain_uT_per_N') for b in CB], '{:.1f}')} µT/N with four magnets, "
        f"against {rg([cb(c5, 'centre', b, 'gain_uT_per_N') for b in CB], '{:.1f}')} with five. Untrained, the physical "
        "model of Table 13 places the four-magnet tip's holds of 1 N and over within "
        f"{rg([f4['untrained'][b]['loc_mm'] for b in ('1-2', '2-3.5', '3.5-8')], '{:.1f}')} mm.</p>"
        "<p>Every central hold has a training target 1 to 1.5 mm from its own. On the rest of the face, with four "
        f"magnets, holds whose nearest training target is 0.5 to 1 mm away err by {cn(c4, '0.5-1')['loc_mm']:.2f} mm, "
        f"over {cn(c4, '0.5-1')['n']} holds. Those at 1 to 1.5 mm err by {cn(c4, '1-1.5')['loc_mm']:.2f} mm as well, "
        f"over {cn(c4, '1-1.5')['n']} holds. At 1.5 to 2.5 mm the error is {cn(c4, '1.5-2.5')['loc_mm']:.2f} mm, over "
        f"{cn(c4, '1.5-2.5')['n']} holds. With five magnets the same three groups give {near5[0]:.2f}, {near5[1]:.2f} "
        f"and {near5[2]:.2f} mm. The record noise was then added once to each held-out board change, as "
        f"Gaussian noise of {g4['noise_uT']:.2f} and {g5['noise_uT']:.2f} µT. The centre's error under 1 N doubles, to "
        f"{cb(c4, 'centre', '0-1', 'loc_one_record_mm'):.2f} mm with four magnets and "
        f"{cb(c5, 'centre', '0-1', 'loc_one_record_mm'):.2f} mm with five ("
        + P.code(str((V3D / "20261028-centre_coverage_by_target.json").relative_to(ROOT))) + ", "
        + P.code("scripts/ur5_tip_calibration_centre_coverage.py") + ").</p>"
        + P.table(["Region", "Normal force", "Holds, four / five", "Location error, four magnets (mm)",
                   "Location error, five magnets (mm)", "With record noise, four / five (mm)",
                   "Signal per newton, four / five (µT/N)"],
                  [[rn if b == CB[0] else "", bn, f"{cb(c4, rk, b, 'n')} / {cb(c5, rk, b, 'n')}", f"{cb(c4, rk, b):.2f}",
                    f"{cb(c5, rk, b):.2f}",
                    f"{cb(c4, rk, b, 'loc_one_record_mm'):.2f} / {cb(c5, rk, b, 'loc_one_record_mm'):.2f}",
                    f"{cb(c4, rk, b, 'gain_uT_per_N'):.1f} / {cb(c5, rk, b, 'gain_uT_per_N'):.1f}"]
                   for rk, rn in (("centre", "Centre"), ("rest_of_face", "Rest of the face"),
                                  ("edge_and_flank", "Edges and flanks"))
                   for b, bn in zip(CB, ("Under 1 N", "1 to 2 N", "2 to 8 N"))],
                  caption=f"Location error of the direction GP by region and normal force on the {CC['actions']} press "
                          f"actions of the low sphere, {n_cc:,} holds on each tip: medians over the holds. The press "
                          "targets were split into five folds, each left out of training in turn. Central holds lie "
                          "within 2 mm of the apex over chip U1. With record noise: Gaussian noise with the record noise as its standard deviation, "
                          f"{g4['noise_uT']:.2f} and {g5['noise_uT']:.2f} µT, added to each held-out board change. Rows "
                          "in " + P.code(str((V3D / "20261028-centre_coverage_by_target.json").relative_to(ROOT))) + " ("
                          + P.code("scripts/ur5_tip_calibration_centre_coverage.py") + ").", n=9)
        + "<p><b>Shear on the slides.</b> Before the stop, v3 ID 1 ran 40 slides at 1 and 2 N, and v2 ID 1 ran the same "
        f"slides, with {SL['four_v3id1']['n_holds']:,} turn holds each. The HEX-E measured a median shear of "
        f"{rg([SL[t]['hexe_shear_N'] for t in SL])} N on both tips (Table 10). A board GP trained on each tip's presses "
        f"alone estimates the shear within {sl4('gp_presses')['shear_err_N']:.3f} N with four magnets and "
        f"{sl5('gp_presses')['shear_err_N']:.3f} N with five. With the other slides in training, one slide left out at a "
        f"time, it errs by {sl4('gp_presses_and_other_slides')['shear_err_N']:.3f} and "
        f"{sl5('gp_presses_and_other_slides')['shear_err_N']:.3f} N. The direction GP gives "
        f"{sl4('gp_direction_presses')['shear_err_N']:.3f} and {sl5('gp_direction_presses')['shear_err_N']:.3f} N from "
        f"presses alone, and {sl4('gp_direction_presses_and_other_slides')['shear_err_N']:.3f} and "
        f"{sl5('gp_direction_presses_and_other_slides')['shear_err_N']:.3f} N with the slides.</p>"
        + P.table(["Estimator, training", "Four magnets: shear error (N)", "Four magnets: force angle (°)",
                   "Five magnets: shear error (N)", "Five magnets: force angle (°)"],
                  [[nm, f"{sl4(k)['shear_err_N']:.3f}", f"{sl4(k)['angle_deg']:.1f}", f"{sl5(k)['shear_err_N']:.3f}",
                    f"{sl5(k)['angle_deg']:.1f}"]
                   for k, nm in (("gp_presses", "Board GP, presses"),
                                 ("gp_presses_and_other_slides", "Board GP, presses and the other slides"),
                                 ("gp_direction_presses", "Direction GP, presses"),
                                 ("gp_direction_presses_and_other_slides", "Direction GP, presses and the other slides"))],
                  caption=f"Shear on the first {SL['four_v3id1']['n_slides']} slides of v3 ID 1's stage 2 and the same "
                          "slides of v2 ID 1: the median length of the error of the estimated shear, and the force angle. "
                          "With the other slides, one slide is left out of training at a time. Rows in "
                          + P.code(str((V3D / "20261028-four_vs_five_slides40.json").relative_to(ROOT))) + " ("
                          + P.code("scripts/ur5_tip_calibration_four_vs_five.py") + ").", n=10)
        + "<p><b>Torsion on the twists.</b> On the 12 twists, the torsion estimated by a board GP correlates with the "
        f"HEX-E's at $r$ = {tw4['gp_twists']['r']:.2f} with four magnets and {tw5['gp_twists']['r']:.2f} with five. v2 ID 1's "
        f"twists of the same ids are the comparison, with {tw4['n_holds']} holds on each tip. The twists reached "
        f"{tw4['n_pairs']} angles both on the way out and on the way back. At each of them the torsion "
        "$\\tau_{\\mathrm{fr}}$ is half the difference of the two torques about the normal (Section 6). The GP maps the "
        "15 readings of a hold to its torque about the normal. It was trained on the holds of the tip's other "
        f"{tw4['n_twists'] - 1} twists, one twist left out at a time. Its RMS error is "
        f"{tw4['gp_twists']['rms_Nmm']:.2f} and {tw5['gp_twists']['rms_Nmm']:.2f} N mm, "
        f"{100 * tw4['gp_twists']['rms_Nmm'] / tw4['gp_twists']['hexe_rms_Nmm']:.0f} and "
        f"{100 * tw5['gp_twists']['rms_Nmm'] / tw5['gp_twists']['hexe_rms_Nmm']:.0f} % of the RMS of the torsion itself. "
        "With the presses in training as well, the correlation falls to "
        f"{tw4['gp_presses_and_twists']['r']:.2f} and {tw5['gp_presses_and_twists']['r']:.2f}. On v3 ID 1 the torsion's "
        f"median magnitude is {tw4['hexe_1N']['median_Nmm']:.2f} N mm at 1 N and {tw4['hexe_5N']['median_Nmm']:.2f} N mm "
        f"at 5 N. On v2 ID 1 it is {tw5['hexe_1N']['median_Nmm']:.2f} and {tw5['hexe_5N']['median_Nmm']:.2f} N mm "
        "(Table 11).</p>"
        + P.table(["Quantity", "Four magnets, v3 ID 1", "Five magnets, v2 ID 1"],
                  [[f"Median torsion at {f} N (N mm)", f"{tw4['hexe_' + k]['median_Nmm']:.2f}",
                    f"{tw5['hexe_' + k]['median_Nmm']:.2f}"] for f, k in (("1", "1N"), ("2.5", "2.5N"), ("5", "5N"))]
                  + [["RMS of the torsion (N mm)", f"{tw4['gp_twists']['hexe_rms_Nmm']:.2f}",
                      f"{tw5['gp_twists']['hexe_rms_Nmm']:.2f}"]]
                  + [[f"Board GP, {nm}: {q}", fmt.format(tw4[k][key]), fmt.format(tw5[k][key])]
                     for k, nm in (("gp_twists", "other twists"), ("gp_presses_and_twists", "presses and other twists"))
                     for q, key, fmt in (("correlation $r$", "r", "{:.2f}"), ("RMS error (N mm)", "rms_Nmm", "{:.2f}"))],
                  caption=f"Torsion on v3 ID 1's {tw4['n_twists']} twists of 2026-10-28 and v2 ID 1's twists of the same "
                          f"ids, {tw4['n_holds']} holds each. Medians of the torsion's magnitude over "
                          f"{tw4['hexe_1N']['n_pairs']}, {tw4['hexe_2.5N']['n_pairs']} and {tw4['hexe_5N']['n_pairs']} "
                          "angles; the 5 N row includes the hard twist at 6 N. The RMS of the torsion is the error of an "
                          "estimate of zero. $r$ is the correlation between the half difference of the GP's estimates and "
                          "the HEX-E's torsion, with one twist left out of training at a time. Rows in "
                          + P.code(str((V3D / "20261028-four_vs_five_twists.json").relative_to(ROOT))) + " ("
                          + P.code("scripts/ur5_tip_calibration_four_vs_five.py") + ").", n=11)
        + "<p><b>Four against five on the rod.</b> Both tips ran "
        f"{RO['actions']} press actions on the rod of radius 12.5 mm, with {RO['holds']['v3']} holds each. On v3 ID 1 they "
        "come from its stage 2 and six more presses, run on 2026-10-28 until 13:52. On v2 ID 1 they come from its rod passes of "
        "2026-10-27. Each tip's direction GP was trained on its own rod presses, in five folds grouped by action. The "
        f"contact is placed within {ro4['all']['loc_mm']:.2f} mm with four magnets and {ro5['all']['loc_mm']:.2f} mm with "
        f"five (Table 12). Under 1 N four magnets give {ro4['0-1']['loc_mm']:.2f} mm and five "
        f"{ro5['0-1']['loc_mm']:.2f} mm. At 1 to 2 N they give {ro4['1-2']['loc_mm']:.2f} and "
        f"{ro5['1-2']['loc_mm']:.2f} mm. The force errs by {100 * ro4['all']['force_rel_err']:.1f} and "
        f"{100 * ro5['all']['force_rel_err']:.1f} %. On the rod the board changes by "
        f"{RO['four_v3id1']['gain_uT_per_N']['1-2']:.1f} µT/N at 1 to 2 N with four magnets, against "
        f"{RO['five_v2id1']['gain_uT_per_N']['1-2']:.1f} with five. On the sphere's grid it changed by {gn('1-2')[0]:.1f} "
        f"and {gn('1-2')[1]:.1f} µT/N. The noise-equivalent force at 1 to 2 N is therefore "
        f"{1000 * RO['four_v3id1']['nef_N']['1-2']:.0f} mN with four magnets and "
        f"{1000 * RO['five_v2id1']['nef_N']['1-2']:.0f} mN with five. The rod's "
        f"{RO['slides']['four_v3id1']['n_slides']} slides at 1 N were scored as the sphere's were. With the presses and "
        "the other slides in training, the shear is estimated by the direction GP within "
        f"{RO['slides']['four_v3id1']['gp_direction_presses_and_other_slides']['shear_err_N']:.3f} N with four magnets "
        f"and {RO['slides']['five_v2id1']['gp_direction_presses_and_other_slides']['shear_err_N']:.3f} N with five.</p>"
        "<p><b>Four against five on the tall indenter.</b> On the tall indenter v3 ID 1 and v2 ID 1 share "
        f"{TA['actions']} press actions, with {TA['holds']['v3']} holds each. v3 ID 1's session ran on 2026-10-28 from "
        "11:49 to 17:41, and v2 ID 1's in R29 on 10-27. Contacts are placed by the direction GP within "
        f"{ta4['all']['loc_mm']:.2f} mm with four magnets and {ta5['all']['loc_mm']:.2f} mm with five (Table 12). On the "
        f"low sphere the two tips gave {e4['all']['loc_mm']:.2f} and {e5['all']['loc_mm']:.2f} mm. The board changes by "
        f"{TA['four_v3id1']['gain_uT_per_N']['1-2']:.1f} against {TA['five_v2id1']['gain_uT_per_N']['1-2']:.1f} µT/N at 1 "
        "to 2 N. The noise-equivalent force is therefore "
        f"{1000 * TA['four_v3id1']['nef_N']['1-2']:.0f} mN with four magnets and "
        f"{1000 * TA['five_v2id1']['nef_N']['1-2']:.0f} mN with five.</p>"
        + P.table(["Quantity", "Rod, four magnets", "Rod, five magnets", "Tall indenter, four magnets",
                   "Tall indenter, five magnets"],
                  [["Press actions"] + [str(d["actions"]) for d, _ in RT],
                   ["Holds"] + [str(d["holds"]["v3" if t.startswith("four") else "v2"]) for d, t in RT]]
                  + [[f"Location error, {nm} (mm)"] + [f"{d[t]['gp_direction'][k]['loc_mm']:.2f}" for d, t in RT]
                     for k, nm in (("all", "all holds"), ("0-1", "under 1 N"), ("1-2", "1 to 2 N"), ("2-3.5", "2 to 3.5 N"),
                                   ("3.5-8", "3.5 to 8 N"))]
                  + [["Force error (%)"] + [f"{100 * d[t]['gp_direction']['all']['force_rel_err']:.1f}" for d, t in RT]]
                  + [[f"Signal per newton {nm} (µT/N)"] + [f"{d[t]['gain_uT_per_N'][k]:.1f}" for d, t in RT]
                     for k, nm in (("1-2", "at 1 to 2 N"), ("face", "on the face"))]
                  + [["Record noise (µT)"] + [f"{d[t]['noise_uT']:.2f}" for d, t in RT],
                     ["Noise-equivalent force at 1 to 2 N (mN)"] + [f"{1000 * d[t]['nef_N']['1-2']:.0f}" for d, t in RT],
                     ["Shear error on the slides (N)",
                      f"{RO['slides']['four_v3id1']['gp_direction_presses_and_other_slides']['shear_err_N']:.3f}",
                      f"{RO['slides']['five_v2id1']['gp_direction_presses_and_other_slides']['shear_err_N']:.3f}", "–", "–"]],
                  caption="Four N52 magnets (v3 ID 1) against five (v2 ID 1) on the rod and the tall indenter, on the "
                          "press actions both tips ran there. Location and force errors are those of the direction GP "
                          "under five folds grouped by action. The shear error is the median length of the error of the "
                          f"estimated shear on the rod's {RO['slides']['four_v3id1']['n_slides']} slides, with the presses "
                          "and the other slides in training, one slide left out at a time. No slides were compared on the "
                          "tall indenter. Rows in "
                          + P.code(str((V3D / "20261028-four_vs_five_rod.json").relative_to(ROOT))) + " and "
                          + P.code("20261029-four_vs_five_tall_v3id1.json") + " beside it ("
                          + P.code("scripts/ur5_tip_calibration_four_vs_five.py") + ").", n=12)
        + "<p><b>Physical model of pad 05R4.</b> Pad 05R4's FE case and surface basis were built on 2026-10-28 on b_v2's "
        f"grid and depth, with the four magnets. On the same {FE['actions']} grid actions, the magnets' strength, the "
        f"board's offsets and the compliance scale $c$ were fitted to the holds up to {f4['fit']['max_n']:g} N (Table 13). "
        f"The fit gives $c$ = {f4['fit']['c']:.3f} with four magnets and {f5['fit']['c']:.3f} with five. The fitted "
        f"modulus is therefore {f4['fit']['E_fit_MPa']:.1f} and {f5['fit']['E_fit_MPa']:.1f} MPa, "
        f"against the FE's {FE['E_FE_MPa']:g} MPa. The registered indentation grows by {rg(dep(f4), '{:.0f}')} µm/N with "
        f"four magnets and {rg(dep(f5), '{:.0f}')} µm/N with five, more than twice the FE's "
        f"{rg(fe_give, '{:.0f}')} µm/N. The fitted scale does not improve the contact location. Under five folds grouped by "
        f"action, the physical model errs by {f5['cv']['table']['all']['loc_mm']:.2f} mm with five magnets "
        f"({f5['untrained']['all']['loc_mm']:.2f} mm untrained). With four it errs by "
        f"{f4['cv']['table']['all']['loc_mm']:.2f} mm ({f4['untrained']['all']['loc_mm']:.2f} mm untrained).</p>"
        "<p><b>Where the magnets sit.</b> With four magnets, the magnets' strength and the board's offset along $x$ are "
        "nearly interchangeable in the rest field: a free rest fit puts the strength at its bound of 2.0 and the offset at "
        "2.69 mm. The offset is fixed by the presses. Refitted to them from either rest fit, with the offset's prior loosened to "
        f"2.0 mm, it goes to {rg(ps4('dx'))} mm on the four-magnet tip. On the five-magnet tip it goes to "
        f"{rg(ps5('dx'))} mm, against {FE['prior_scan']['five/tight']['start']['dx']:.2f} mm in its rest fit. Only "
        f"the product of the strength and the compliance scale is fixed by them: {rg(ps4('k_times_c'), '{:.3f}')} with four magnets "
        f"and {rg(ps5('k_times_c'), '{:.3f}')} with five. In every variant the location error stays at "
        f"{rg(ps4('loc_mm'))} mm with four magnets and {rg(ps5('loc_mm'))} mm with five. At the registered contact, "
        f"{f4['fit']['rms_uT']:.2f} and {f5['fit']['rms_uT']:.2f} µT per hold are left after the field model.</p>"
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
                          + P.code("scripts/ur5_tip_calibration_fe_calibration.py") + ").", n=13)
    )
    S.append(("v3id1", "Four N52 magnets against five on v3 ID 1", body))

    # 12. v3 ID 2 (four N52) against v3 ID 1 on the presses both ran, and v3 ID 2's top-edge pass (feed lines of 10-29)
    T2 = ("v3 ID 2", "v3 ID 1")
    gr = lambda k, t, g: GR[k]["sessions"][t]["groups"][g]["gain_0.5-3.5"]  # noqa: E731
    tt = lambda k, t: TT[k]["tips"][t]  # noqa: E731
    HI = ("1-2", "2-3.5", "3.5-8")
    tef = TE["with_the_pass"]
    fe_diff = max(100 * abs(gr(k, T2[0], g) / gr(k, T2[1], g) - 1) for k in FX for g in ("grid, face", "grid, edge"))
    body = (
        "<p>v3 ID 2 is the second tip printed with pad 05R4's four N52 magnets. At about 18:40 on 2026-10-28 the owner "
        "put it on the robot in place of v3 ID 1. Its suite ran on the low sphere, the tall indenter and the rod from "
        "18:44 to 08:13 on 10-29. All 488 of its actions ran without a HEX-E time-out. From its touch-offs it was registered at 0.044 mm "
        "rms on the sphere, 0.073 mm on the tall indenter and 0.023 mm on the rod ("
        + P.code("docs/experiments/20261022-ur5_tip_calibration/lab/STATE.md") + "). Its sessions were analysed on the "
        "lab computer, corvus, while the robot was idle.</p>"
        f"<p>On the low sphere both tips ran {TT['sphere']['actions']} press actions, with {tt('sphere', T2[0])['holds']} "
        f"holds each (Table 14). The board changes by {tg('sphere', T2[0]):.1f} µT/N at 1 to 2 N on v3 ID 2 and "
        f"{tg('sphere', T2[1]):.1f} µT/N on v3 ID 1. The record noise is {G2['four_v3id2']['noise_uT']:.2f} and "
        f"{g4['noise_uT']:.2f} µT. Each tip's direction GP was trained on its own holds and scored in five folds grouped "
        f"by action. v3 ID 2's contacts are placed within {t2['sphere']['all']['loc_mm']:.2f} mm and v3 ID 1's within "
        f"{t1['sphere']['all']['loc_mm']:.2f} mm. The force errs by {100 * t2['sphere']['all']['force_rel_err']:.1f} and "
        f"{100 * t1['sphere']['all']['force_rel_err']:.1f} %. From 1 N up the location errs by "
        f"{rg([t2['sphere'][b]['loc_mm'] for b in HI])} mm on v3 ID 2 and {rg([t1['sphere'][b]['loc_mm'] for b in HI])} mm "
        "on v3 ID 1. With five magnets, v2 ID 1's errors on the same actions are "
        f"{G2['five_v2id1']['gp_direction']['all']['loc_mm']:.2f} mm and "
        f"{100 * G2['five_v2id1']['gp_direction']['all']['force_rel_err']:.1f} %. v3 ID 1's grid stage had 47 speed "
        "variants and one dense press more, which v3 ID 2's suite left out. With them, on all "
        f"{G['actions']} actions, v3 ID 1's error is {e4['all']['loc_mm']:.2f} mm (Section 11).</p>"
        f"<p>On the rod the shared actions are {TT['rod']['actions']}, with {tt('rod', T2[0])['holds']} holds each. Both "
        f"tips' contacts are placed within {max(t2['rod']['all']['loc_mm'], t1['rod']['all']['loc_mm']):.2f} mm, and the "
        f"force errs by {100 * t2['rod']['all']['force_rel_err']:.1f} and {100 * t1['rod']['all']['force_rel_err']:.1f} %. "
        f"On the tall indenter the shared actions are {TT['tall']['actions']}, with {tt('tall', T2[0])['holds']} holds "
        f"each. There the location errs by {t2['tall']['all']['loc_mm']:.2f} mm on v3 ID 2 and "
        f"{t1['tall']['all']['loc_mm']:.2f} mm on v3 ID 1, about three times the sphere's. The force errs by "
        f"{100 * t2['tall']['all']['force_rel_err']:.1f} and {100 * t1['tall']['all']['force_rel_err']:.1f} %. At 3 N and "
        "over, the contact read from the two tips' boards lies 0.75 and 0.98 mm along −y of the registration on the tall "
        "indenter. On the sphere it lies 0.39 and 0.12 mm along +y (" + link(L["v3"], "v3 plan") + ", section "
        "&#8220;v3 ID 2 against v3 ID 1&#8221;). On every indenter the two tips' signals per newton on the face and the "
        f"edges differ by {fe_diff:.0f} % or less. Presses on the tall indenter's flanks change the board by "
        f"{gr('tall', T2[0], 'grid, flank'):.1f} µT/N on v3 ID 2 and {gr('tall', T2[1], 'grid, flank'):.1f} µT/N on v3 ID 1. "
        f"On the sphere's flanks the change is {gr('sphere', T2[0], 'grid, flank'):.1f} and "
        f"{gr('sphere', T2[1], 'grid, flank'):.1f} µT/N.</p>"
        + P.table(["Quantity", "Sphere, v3 ID 2", "Sphere, v3 ID 1", "Rod, v3 ID 2", "Rod, v3 ID 1",
                   "Tall indenter, v3 ID 2", "Tall indenter, v3 ID 1"],
                  [["Press actions"] + [str(TT[k]["actions"]) for k in FX for t in T2],
                   ["Holds"] + [str(tt(k, t)["holds"]) for k in FX for t in T2]]
                  + [[f"Location error, {nm} (mm)"] + [f"{tt(k, t)['gp_direction'][b]['loc_mm']:.2f}" for k in FX for t in T2]
                     for b, nm in (("all", "all holds"), ("0-1", "under 1 N"), ("1-2", "1 to 2 N"), ("2-3.5", "2 to 3.5 N"),
                                   ("3.5-8", "3.5 to 8 N"))]
                  + [["Force error (%)"] + [f"{100 * tt(k, t)['gp_direction']['all']['force_rel_err']:.1f}"
                                            for k in FX for t in T2],
                     ["Signal per newton at 1 to 2 N (µT/N)"] + [f"{tg(k, t):.1f}" for k in FX for t in T2]]
                  + [[f"Signal per newton, {nm} (µT/N)"]
                     + [(f"{gr(k, t, g):.1f}" if g in GR[k]["sessions"][t]["groups"] else "–") for k in FX for t in T2]
                     for g, nm in (("grid, face", "face"), ("grid, edge", "edges"), ("grid, flank", "flanks"),
                                   ("face centre, other plans", "face centre"))],
                  caption="v3 ID 2 against v3 ID 1 on the press actions that both tips ran on each indenter. Location "
                          "and force errors are those of each tip's direction GP under five folds grouped by action: "
                          "medians over the holds. The signal per newton by region is the median over 0.5 to 3.5 N, on "
                          f"the grid actions of both sessions ({GR['sphere']['common_actions']}, "
                          f"{GR['rod']['common_actions']} and {GR['tall']['common_actions']} actions). The face centre's "
                          "row is the reference presses at the apex over chip U1. The rod's "
                          "presses include no flank. Rows in "
                          + P.code(str((V3D / "20261029-tip_vs_tip_sphere_v3id2_v3id1.json").relative_to(ROOT)))
                          + ", its rod and tall files, and " + P.code("data/20261029-gain_by_region_*.json") + " beside them ("
                          + P.code("scripts/ur5_tip_calibration_tip_vs_tip.py") + ", "
                          + P.code("scripts/ur5_tip_calibration_gain_by_region.py") + ").", n=14)
        + "<p><b>Four against five on v3 ID 2.</b> v3 ID 2's presses were also compared with v2 ID 1's of the same "
        f"actions, on each indenter (Table 15). The shared actions are {F2['sphere']['actions']} on the sphere, "
        f"{F2['rod']['actions']} on the rod and {F2['tall']['actions']} on the tall indenter. At 1 to 2 N the signal per "
        f"newton with five magnets is {rt2('sphere'):.2f} times v3 ID 2's on the sphere. It is {rt2('rod'):.2f} times "
        f"v3 ID 2's on the rod and {rt2('tall'):.2f} times on the tall indenter. "
        f"On v3 ID 1 the ratio on the sphere was {gn('1-2')[1] / gn('1-2')[0]:.2f}, and the readout model predicts 1.8. "
        f"The record noise is {rg([F2[k]['four_v3id2']['noise_uT'] for k in FX])} µT with four magnets and "
        f"{rg([F2[k]['five_v2id1']['noise_uT'] for k in FX])} µT with five. The noise-equivalent force at 1 to 2 N is "
        f"therefore {rg([1000 * F2[k]['four_v3id2']['nef_N']['1-2'] for k in FX], '{:.0f}')} mN with four magnets and "
        f"{rg([1000 * F2[k]['five_v2id1']['nef_N']['1-2'] for k in FX], '{:.0f}')} mN with five. The direction GP errs "
        f"by {f2('sphere', 'four_v3id2'):.2f} and {f2('sphere', 'five_v2id1'):.2f} mm on the sphere. On the rod it errs by "
        f"{f2('rod', 'four_v3id2'):.2f} and {f2('rod', 'five_v2id1'):.2f} mm, as on v3 ID 1, and on the tall indenter by "
        f"{f2('tall', 'four_v3id2'):.2f} and {f2('tall', 'five_v2id1'):.2f} mm. The analysis can also move the sphere "
        "along the approach by the drift of the reference presses' onset, interpolated in time ("
        + P.code("--ref-drift") + " of " + P.code("scripts/ur5_tip_calibration_analysis.py") + "). On v3 ID 2's sphere "
        "session this moves the contact points by 0.014 mm at the median and the depths by up to 0.11 mm. The direction "
        "GP's error stays at 0.29 mm ("
        + P.code("docs/experiments/20261027-v3_tip_calibration_and_crosstalk/feed.jsonl") + ", 10:28).</p>"
        + P.table(["Quantity", "Sphere, v3 ID 2", "Sphere, v2 ID 1", "Rod, v3 ID 2", "Rod, v2 ID 1",
                   "Tall indenter, v3 ID 2", "Tall indenter, v2 ID 1"],
                  [["Press actions"] + [str(F2[k]["actions"]) for k in FX for t in T5],
                   ["Holds"] + [str(F2[k]["holds"]["v3" if t.startswith("four") else "v2"]) for k in FX for t in T5]]
                  + [[f"Location error, {nm} (mm)"] + [f"{F2[k][t]['gp_direction'][b]['loc_mm']:.2f}" for k in FX for t in T5]
                     for b, nm in (("all", "all holds"), ("0-1", "under 1 N"), ("1-2", "1 to 2 N"), ("2-3.5", "2 to 3.5 N"),
                                   ("3.5-8", "3.5 to 8 N"))]
                  + [["Force error (%)"] + [f"{100 * F2[k][t]['gp_direction']['all']['force_rel_err']:.1f}"
                                            for k in FX for t in T5],
                     ["Signal per newton at 1 to 2 N (µT/N)"] + [f"{F2[k][t]['gain_uT_per_N']['1-2']:.1f}"
                                                                 for k in FX for t in T5],
                     ["Record noise (µT)"] + [f"{F2[k][t]['noise_uT']:.2f}" for k in FX for t in T5],
                     ["Noise-equivalent force at 1 to 2 N (mN)"] + [f"{1000 * F2[k][t]['nef_N']['1-2']:.0f}"
                                                                    for k in FX for t in T5]],
                  caption="Four N52 magnets (v3 ID 2) against five (v2 ID 1) on the press actions that both tips ran "
                          "on each indenter. Location and force errors are those of the direction GP under five folds "
                          "grouped by action: medians over the holds. The noise-equivalent force is the record noise "
                          "divided by the signal per newton. Rows in "
                          + P.code(str((V3D / "20261029-four_vs_five_grid_v3id2.json").relative_to(ROOT)))
                          + " and its rod and tall files (" + P.code("scripts/ur5_tip_calibration_four_vs_five.py")
                          + ").", n=15)
        + "<p><b>Tip end.</b> v3 ID 2 ran the top-edge pass on the low sphere from 19:21 to 19:30 on 2026-10-28, with the "
        "owner watching. The pass pressed the tip end nine times at 0.25 to 2 N, and an edge once to 3 N. A reference "
        "press at the face centre came before the pass and another after it. On the earlier tips no press target lay within 1.5 mm of the "
        "tip end (Table 5). Pressing the tip end changes the board by "
        f"{gr('sphere', T2[0], 'top-edge pass, tip'):.1f} µT/N at 0.5 to 3.5 N. That lies between the face's "
        f"{gr('sphere', T2[0], 'grid, face'):.1f} and the edges' {gr('sphere', T2[0], 'grid, edge'):.1f} µT/N. With a "
        f"direction GP trained on the sphere's grid alone, the pass's {TE['pass_holds']} holds are placed "
        f"{TE['grid_only']['all']['loc_mm']:.2f} mm off at the median. With the pass's other presses in training as well, "
        f"they are placed within {tef['all']['loc_mm']:.2f} mm. At 1 to 2 N and at 2 to 3.5 N the error is "
        f"{tef['1-2']['loc_mm']:.2f} and {tef['2-3.5']['loc_mm']:.2f} mm. The grid's own holds give "
        f"{TE['grid_five_folds']['1-2']['loc_mm']:.2f} and {TE['grid_five_folds']['2-3.5']['loc_mm']:.2f} mm under five "
        "folds (" + P.code(str((V3D / "20261029-top_edge_gp_v3id2.json").relative_to(ROOT))) + ", "
        + P.code("scripts/ur5_tip_calibration_top_edge_gp.py") + ").</p>"
    )
    S.append(("v3id2", "Calibration of the second four-magnet tip v3 ID 2", body))

    # 13. what each motion adds on the four-magnet tips (feed line of 10-29 12:45)
    FXN = dict(sphere="Low sphere", tall="Tall indenter", rod="Rod")
    FX3 = ("sphere", "tall", "rod")
    lf = lambda d: f"{d['loc_mm']:.2f} / {100 * d['force_rel_err']:.0f}"  # noqa: E731
    pc = lambda k, t: mf(t)["pooled_cv"][k]["loc_mm"] - mf(t)["own_cv"][k]["loc_mm"]  # noqa: E731
    lv = lambda t, b, s: MV["tips"][t]["levels"][b][s]["loc_mm"]  # noqa: E731
    cnt = lambda t: MV["tips"][t]["count"]["0.5"]  # noqa: E731
    sl = lambda t, k: MV["tips"][t]["slides"][k]  # noqa: E731
    sh = lambda t, q: MV["tips"][t]["shared"]["at_1N_and_over"][q]  # noqa: E731
    k100 = lambda a, b: MV["transfer"][f"{a} -> {b}"]["with_k"]["100"]["all"]["loc_mm"]  # noqa: E731
    te1, te2 = (MV["tips"][t]["tip_end"] for t in T3)
    tm2 = MV["tips"]["v3 ID 2"]["time"]
    hrs = lambda *acts: sum(tm2[k].get(a, {}).get("hours", 0.0) for k in FX3 for a in acts)  # noqa: E731
    suite = sum(v["hours"] for k in FX3 for a, v in tm2[k].items() if a != "press families")
    hard = tm2["sphere"]["press families"]["hard"]
    two = lambda f: f"{f(T3[0]):.2f} and {f(T3[1]):.2f}"  # noqa: E731
    rod_low = [-100 * mf(t)["without_it"]["rod"]["force_rel_bias"] for t in T3]
    light = [lv(t, "0.5 N and under", s) for t in T3 for s in ("left_out", "pooled_cv")]
    mvp = P.code(str((V3D / "20261029-motion_value.json").relative_to(ROOT)))
    body = (
        "<p>A four-magnet tip's calibration needs presses on each indenter, presses on the tip end and slides. Presses "
        "over 3.5 N can be left out of training, and no torsion is recovered from the twists. A GP trained on one tip places "
        f"the other tip's presses {xf(*T3)['loc_mm']:.2f} and {xf(*T3[::-1])['loc_mm']:.2f} mm off, so each tip runs its "
        "own suite. Direction GPs were trained on v3 ID 1's and ID 2's sessions on the three indenters. One kind of hold "
        "at a time was left out of training (" + link(L["motions"], "calibration motions page") + "; rows in " + mvp
        + ", " + P.code("scripts/ur5_tip_calibration_motion_value.py") + ").</p>"
        f"<p><b>Presses on each indenter.</b> Trained on the other two indenters, the GP places an indenter's presses "
        f"{min(wo):.2f} to {max(wo):.2f} mm off (Table 16). With its own presses in five folds by action, they are "
        f"placed {min(own):.2f} to {max(own):.2f} mm off. Without its own presses, the rod's forces read "
        f"{rg(rod_low, '{:.0f}')} % low on both tips. On the hand a contact's shape is unknown, so one GP must cover all "
        "three indenters. Trained on all three, the low sphere's error grows by "
        f"{pc('sphere', T3[0]):.2f} and {pc('sphere', T3[1]):.2f} mm, and the rod's by {pc('rod', T3[0]):.2f} and "
        f"{pc('rod', T3[1]):.2f} mm.</p>"
        + P.table(["Tip", "Indenter", "Press actions", "Own presses", "All three indenters", "Other two indenters",
                   "Low sphere only"],
                  [[t, FXN[k], str(mf(t)["press_actions"][k]), lf(mf(t)["own_cv"][k]), lf(mf(t)["pooled_cv"][k]),
                    lf(mf(t)["without_it"][k]), lf(mf(t)["sphere_only"][k]) if k in mf(t)["sphere_only"] else "–"]
                   for t in T3 for k in FX3],
                  caption="Location error (mm) / force error (%) of the direction GP on each indenter's presses, by "
                          "training set: the indenter's own presses, all three indenters' presses, the other two "
                          "indenters' presses and the low sphere's presses. Sets that hold the indenter's presses are "
                          "scored in five folds grouped by action. Medians over the holds; rows in " + mvp + ".", n=16)
        + "<p><b>Force levels and press count.</b> Presses over 3.5 N add nothing to training. Left out, they are placed "
        f"{two(lambda t: lv(t, 'over 3.5 N', 'left_out'))} mm off on v3 ID 1 and ID 2. In training, in five folds, they "
        f"are placed {two(lambda t: lv(t, 'over 3.5 N', 'pooled_cv'))} mm off. Holds at 0.5 N and under are placed "
        f"{rg(light)} mm off, whether or not they are in training. With half the press actions in training, the error on "
        f"all three indenters rises from {pall(T3[0]):.2f} to {cnt(T3[0])['all']['loc_mm']:.2f} mm on v3 ID 1. On v3 ID 2 it rises from "
        f"{pall(T3[1]):.2f} to {cnt(T3[1])['all']['loc_mm']:.2f} mm.</p>"
        f"<p><b>Tip end.</b> v3 ID 2's {len(te2['actions'])} presses on the tip end were left out of a GP trained on all "
        f"three indenters. It places them {te2['without_end_presses']['loc_mm']:.2f} mm off at the median. With the "
        "other end presses in training, one action left out at a time, they are placed "
        f"{te2['with_other_end_presses']['loc_mm']:.2f} mm off. v3 ID 1 has {len(te1['actions'])} end presses, from the "
        f"tall indenter's top edge. They are placed {te1['without_end_presses']['loc_mm']:.2f} mm off without end presses "
        f"in training and {te1['with_other_end_presses']['loc_mm']:.2f} mm with the other. The low sphere's GP on the "
        "whole top-edge pass is in Section 12.</p>"
        "<p><b>Slides.</b> Slides must be in training too, each indenter's own among them (Table 17). The measured "
        f"shear on the low sphere's slides is {two(lambda t: sl(t, 'sphere')['hexe_shear_N'])} N. A GP trained on "
        f"presses alone errs in shear by {two(lambda t: sl(t, 'sphere')['presses']['shear_err_N'])} N there. With the "
        "other indenters' slides in training the error is "
        f"{two(lambda t: sl(t, 'sphere')['presses_other_slides']['shear_err_N'])} N. With the sphere's own slides in "
        f"five folds it is {two(lambda t: sl(t, 'sphere')['all_cv']['shear_err_N'])} N. With its own slides the "
        f"sliding contact is placed within {two(lambda t: sl(t, 'sphere')['all_cv']['loc_mm'])} mm. With only the other "
        f"indenters' slides it is placed {two(lambda t: sl(t, 'sphere')['presses_other_slides']['loc_mm'])} mm off.</p>"
        + P.table(["Tip", "Indenter", "Slides", "Measured shear (N)", "Presses only", "With the other indenters' slides",
                   "With its own slides"],
                  [[t, FXN[k], str(sl(t, k)["slide_actions"]), f"{sl(t, k)['hexe_shear_N']:.2f}"]
                   + [f"{sl(t, k)[s]['shear_err_N']:.2f} / {sl(t, k)[s]['loc_mm']:.2f}"
                      for s in ("presses", "presses_other_slides", "all_cv")]
                   for t in T3 for k in FX3],
                  caption="Shear error (N) / location error (mm) of the direction GP on each indenter's slides, by "
                          "training set. The shear error is the size of the difference between the estimated and the "
                          "measured shear; medians over the slides' holds. The set with the indenter's own slides is "
                          "scored in five folds grouped by action. Rows in " + mvp + ".", n=17)
        + f"<p><b>Twists.</b> No torsion is recovered from the twists. On v3 ID 2's {tw2['n_twists']} twists a board "
        f"GP's torsion correlates with the HEX-E's at $r$ = {tw2['gp_twists']['r']:.2f}. On v3 ID 1's it was "
        f"{tw4['gp_twists']['r']:.2f} (Section 11). The GP's RMS error is {tw2['gp_twists']['rms_Nmm']:.1f} N mm, and "
        f"the torsion itself is {tw2['gp_twists']['hexe_rms_Nmm']:.1f} N mm RMS ("
        + P.code(str((V3D / "20261029-four_vs_five_grid_v3id2.json").relative_to(ROOT))) + ").</p>"
        "<p><b>One tip's calibration on the other.</b> Trained on v3 ID 1's presses and slides, the GP places v3 ID 2's "
        f"presses {xf(*T3)['loc_mm']:.2f} mm off, with {100 * xf(*T3)['force_rel_err']:.0f} % force error. Trained on "
        f"ID 2's own holds, in five folds, it places them {pall(T3[1]):.2f} mm off. The other way round the error is "
        f"{xf(*T3[::-1])['loc_mm']:.2f} mm against {pall(T3[0]):.2f} mm. A hundred of a tip's own press actions were also "
        f"added to the other tip's data. Its presses are then placed {k100(*T3):.2f} mm off on v3 ID 2 and "
        f"{k100(*T3[::-1]):.2f} mm on v3 ID 1. From half of each tip's own actions alone, "
        f"{cnt(T3[1])['training_actions']} and {cnt(T3[0])['training_actions']}, they are placed "
        f"{cnt(T3[1])['all']['loc_mm']:.2f} and {cnt(T3[0])['all']['loc_mm']:.2f} mm off.</p>"
        "<p><b>Tall indenter against the low sphere.</b> At the same registered contact the two indenters change the "
        "board in different directions. Six points, three on the face and three on the edges, were pressed on both "
        f"indenters at the same force levels. At 1 N and over their registered contact points lie "
        f"{two(lambda t: sh(t, 'point_mm'))} mm apart on v3 ID 1 and ID 2. The tool's orientation differs by "
        f"{sh(T3[0], 'tool_angle_deg'):.1f} and {sh(T3[1], 'tool_angle_deg'):.1f}°. The board's changes point "
        f"{sh(T3[0], 'board_angle_deg'):.1f} and {sh(T3[1], 'board_angle_deg'):.1f}° apart. On the tall indenter the "
        f"change per newton is {two(lambda t: sh(t, 'gain_ratio'))} times the low sphere's. Its forces have a shear of "
        f"{two(lambda t: sh(t, 'shear_per_N_tall'))} N per newton, against {two(lambda t: sh(t, 'shear_per_N_sphere'))} "
        "on the low sphere. A GP trained on the low sphere's other presses places the six points "
        f"{two(lambda t: sh(t, 'gp_loc_sphere_mm'))} mm off on the low sphere. On the tall indenter they are placed "
        f"{two(lambda t: sh(t, 'gp_loc_tall_mm'))} mm off. The cause of the difference is open. Indenter plate v2e, "
        "ready to print since 10-29, follows plate v2d. It has one 5 mm sphere on a column for the whole suite of the finger on the "
        "robot (" + link(L["rig"], "inverted rig page") + ", Section 3). The comparison is therefore to be repeated "
        "on it before the low sphere is taken off.</p>"
        f"<p><b>Robot time.</b> v3 ID 2's suite took {suite:.1f} h of robot time. Its slides took "
        f"{hrs('slide'):.1f} h and its touch-offs {hrs('touch-off'):.1f} h. The twists and rolls took "
        f"{hrs('twist', 'roll'):.1f} h. A suite of 10.7 h is proposed on the calibration motions page, in its Section "
        f"11. In it the hard presses to 8 N are cut from {hard['actions']} to 5, as a test. The twists and rolls are "
        "cut to 4 twists on the low sphere, and the other motions run as before.</p>"
    )
    S.append(("motion_value", "Contribution of each calibration motion on the four-magnet tips", body))

    body = (
        "<p>The HEX-E is a weak reference. Its $F_z$ channel reads 0.78–0.87 of its other axes' gain in two gravity fits. "
        "Its link dropped out in three of the first night's runs, once on 2026-10-26 and once in R29 on 10-27. On 10-28 it "
        "stopped twice, at 02:54 and at 14:32, and read again each time after its connector was reseated. The arm, the "
        "mount and the indenter's stand are taken as rigid; the rod presses bound their compliance at 0.10 mm/N. The "
        "registration fixes the place along the finger less well than across it. The magnets of tips 0 to 3 have "
        "0.20–0.57 of an N52's strength. A press below 1 N therefore changes their readings by less than the step of about "
        "7 µT that a press leaves at rest. At the six points pressed on both indenters, the tall indenter's forces have "
        "twice the low sphere's shear per newton. The board's changes there point "
        f"{sh(T3[0], 'board_angle_deg'):.1f} and {sh(T3[1], 'board_angle_deg'):.1f}° apart (Section 13). The cause is "
        "open. v3 ID 1 ran 40 of its 84 slides on the low sphere. In Section 13 the twists' and rolls' holds are left "
        "out of every training set. Every model here is trained and scored on one cell and one seating of each tip.</p>"
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
        "than 1 mm from a target (" + P.code("scripts/ur5_tip_calibration_coverage.py") + " reports the share by "
        f"region). With v2 ID 1's three indenters, {cvp(V2R, 'flank', 15)} % of the flanks' points lie within 1.5 mm of "
        "a target (Table 5). The density curve predicts 0.3–0.6 mm there.",
        "Tall indenter against the low sphere: press the six shared points of Section 13 on both indenters again in one "
        "session, with the HEX-E's raw wrench logged at 1 kHz. Then compare the force's direction during the approach ("
        + link(L["motions"], "calibration motions page") + "). If the board's changes agree within 3° and the gains "
        "within 5 %, one 5 mm sphere on a column can stand for both. Plate v2e's single sphere can then replace the low "
        "sphere.",
        "Inverted rig: a probe on the HEX-E presses tips held pad up on plate v2e's three seats ("
        + link(L["rig"], "inverted rig page") + "). In its reach search a press of the 5 or 2 mm sphere reaches 452 "
        "of the pad's 808 points, against 286 for the finger on the robot. On the tip end it reaches 40 of 88 points, "
        "against 14. The probe also pushes 20 and 35° off the normal at every face and edge point. By a beam "
        "estimate, a face press deflects a seat by 0.2 µm/N at the pad's apex. The parts are ready to print, on two P1S "
        "plates of 6 h 13 min and 15 h 37 min. The lab program still needs a probe mode. About 330 lines and a day "
        "of work are estimated for it, then 2 to 3 h of simulations ("
        + P.code("docs/experiments/20261022-ur5_tip_calibration/lab/NOTES.md") + "). Then run one seat's plan of 271 "
        "to 279 actions with v3 ID 3 held, and score the direction GP on the face's held-out presses. Within 0.3 mm "
        "and 6 % of their force, the rig calibrates a tip as well as the sphere suite of the finger on the robot.",
        "Twists and slides in the suite: score the presses and slides again with the twists' holds in training, and "
        "with half the low sphere's slides (" + P.code("scripts/ur5_tip_calibration_motion_value.py") + "). If the "
        "errors move by under 2 % with the twists, cutting them to 4 stands. A shear error that rises by under 10 % "
        f"with half the slides would halve their {tm2['sphere']['slide']['hours']:.1f} h "
        "on the low sphere.",
        "Random order: run one grid in random order, so that clock time and absolute temperature stop following the "
        "target, and score the temperature inputs again.",
    ]
    S.append(("future", "Future work", "<ul>" + "".join(f"<li>{t}</li>" for t in items) + "</ul>"))
    return S, lede, summary, film
