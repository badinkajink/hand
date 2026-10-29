"""Prose of the page on the magnetic fingertip's physical model (scripts/fingertip_physical_model_page.py), kept apart from
the plotting. Every number comes from the page's numbers.json, the UR5 sessions' summaries or the linked pages.

The page leads with the four-magnet pad 05R4 (owner, 2026-10-28: "i think we will stick with the 4 magnet config"). Each
five-magnet figure follows its four-magnet one in a collapsed block (owner: "we dont have to delete the 5 magnet figures,
maybe just add them to a collapsed section of like 'view with 5 magnets'")."""
from __future__ import annotations

import math


def sections(N, F, P, link, L):
    """(sections, lede, summary, film_html). N: numbers.json; F: media paths; P: page helpers; link(path, text); L: the
    linked pages by key."""
    fe4, fe5 = N["fe_four"], N["fe"]
    p1, p3, sz, sy = (fe4[k] for k in ("press1", "press3", "shear_z", "shear_y"))
    pr = N["face_centre_press_uT_per_N"]
    fc4 = N["field_at_chips_mT_N52_four"]
    pat4, pat5 = N["patterns_N52_four"], N["patterns_N52_checkerboard"]
    lk4, lk5 = N["likelihood_v3id1"], N["likelihood_tip2"]
    u4 = N["ur5_four"]
    a4, a5 = u4["four_v3id1"], u4["five_v2id1"]
    hi = ("1-2", "2-3.5", "3.5-8")
    loc4, loc5 = [a4[b]["loc_mm"] for b in hi], [a5[b]["loc_mm"] for b in hi]
    fn4 = [a4[b]["fn_rel"] for b in hi]
    rng = lambda v: f"{min(v):.1f}–{max(v):.1f}"   # noqa: E731
    cal = link(L["cal"], "calibration page")
    # the three N52 tips on the press actions they all ran (scripts/fingertip_physical_model_v3_check.py, 2026-10-29)
    vc = N["v3_check"]
    ft, rs, gn = vc["fit"], vc["rest"], vc["gain"]
    FC_, FACE = "face centre, other plans", "grid, face"
    fit_ok = [abs(r["fitted"] / r["measured"] - 1) for t in gn.values() for g, r in t.items()
              if g != FC_ and r["measured"] is not None]
    within = math.ceil(100 * max(fit_ok))
    cen = {k: gn[k][FC_] for k in gn}
    TIP3 = (("v3id1", "v3 ID 1, four"), ("v3id2", "v3 ID 2, four"), ("v2id1", "v2 ID 1, five"))
    BINS4 = ("0-1", "1-2", "2-3.5", "3.5-8")
    un = lambda k, b: ft[k]["untrained"][b]                       # noqa: E731
    loc_by = lambda k: [un(k, b)["loc_mm"] for b in hi]           # noqa: E731
    fn_by = lambda k: [100 * un(k, b)["fn_rel"] for b in hi]      # noqa: E731
    pct = lambda v: f"{min(v):.0f}–{max(v):.0f}"                  # noqa: E731
    ratio = lambda k, w: gn[k][FC_][w] / gn[k][FACE][w]           # noqa: E731
    E_fit = [ft[k]["E_fit_MPa"] for k, _ in TIP3]
    give = [v["slope_um_per_N"] / v["fe_give_um_per_N"] for t in vc["depth"].values() for v in t.values()]
    k1, k2 = (f"{rs[k]['at_press_board']['k']:.2f}" for k in ("v3id1", "v3id2"))
    k_at = f"{k1} on both tips" if k1 == k2 else f"{k1} and {k2}"

    def five(n, png, caption):
        """Figure n with five magnets, in a block that opens on a click."""
        f = P.fig(n, png, caption).replace(f'id="fig{n}"', f'id="fig{n}-5"', 1)
        f = f.replace(f"Figure {n}.", f"Figure {n} with five magnets.", 1)
        return '<details class="five"><summary>View with 5 magnets</summary>' + f + "</details>"

    lede = (
        "A press on the printed fingertip moves the four magnets inside its pad by a few micrometres. Five magnetometers "
        "under the pad read the change of the magnets' field. The physical model computes that sequence from finite "
        f"elements and a dipole model. Under a 5 mm sphere at 1 N the face sinks {p1['approach_um']:.0f} µm and the "
        f"magnets move {rng(p1['magnet_travel_um'])} µm. With four N52 magnets in a checkerboard, the layout chosen on "
        f"2026-10-28, the readings change by {pr['checker_N52_four']:.0f} µT at the face centre "
        f"({pr['checker_N52']:.0f} µT with five). Inverted without training, the model placed the four-magnet tip's UR5 "
        f"presses of 1 N and over within {rng(loc4)} mm, and those under 1 N {a4['0-1']['loc_mm']:.1f} mm off. Fitted to "
        f"the presses of three N52 tips, it gives the signal per newton of the face, edges, flanks and tip end within "
        f"{within} %. At the face centre the two four-magnet tips gave {cen['v3id1']['measured']:.1f} and "
        f"{cen['v3id2']['measured']:.1f} µT/N, against the model's {cen['v3id1']['fitted']:.1f} and "
        f"{cen['v3id2']['fitted']:.1f}.")
    summary = (
        "FE of the printed pad plus a dipole field model give the 15 magnetometer readings for a load anywhere on the "
        "surface. With four N52 magnets in a 5.0 mm checkerboard (pad 05R4, chosen 2026-10-28), at 1 N under a 5 mm sphere "
        f"the face sinks {p1['approach_um']:.0f} µm, the magnets move {rng(p1['magnet_travel_um'])} µm and the readings "
        f"change by {pr['checker_N52_four']:.0f} µT at the face centre (five magnets: {pr['checker_N52']:.0f} µT). Inverted "
        f"without training it places v3 ID 1's UR5 presses of 1 N and over within {rng(loc4)} mm and their normal force "
        f"within {100 * min(fn4):.0f}–{100 * max(fn4):.0f} % (medians); under 1 N it fails ({a4['0-1']['loc_mm']:.1f} mm). "
        f"Fitted to the presses of v3 ID 1, v3 ID 2 and v2 ID 1, it gives each region's signal per newton within {within} % "
        f"except the face centre, where the four-magnet tips gave {cen['v3id1']['measured']:.1f} and "
        f"{cen['v3id2']['measured']:.1f} µT/N against the model's {cen['v3id1']['fitted']:.1f} and "
        f"{cen['v3id2']['fitted']:.1f}.")

    film = P.film(0, F["film"], F["film_poster"],
                  "The contact viewer replays five presses that the model made with the board's measured noise, on the "
                  "five-magnet pad B05R. The pad "
                  "is drawn deformed by the finite elements, 20 times magnified. The arrow is the estimated force, the ring "
                  "the 1-sigma ellipse of the contact point, and the colour the pressure of the simulation's sphere pads. "
                  "The five presses are on the face, the +y flank, the face's −y edge and the tip end, and on the face with a "
                  "shear along the finger. The −y edge is behind the pad in this view. Film from " + link(L["est"], "the estimation page") + ".")
    film = film.replace("Figure 0.", "Film.").replace('id="fig0"', 'id="film"')

    S = []
    # ------------------------------------------------------------------------------------------------ terms
    terms = [
        ("Tip frame", "Coordinates of the fingertip in mm. $x$ points along the face's normal toward the palm, $y$ across the "
         "finger and $z$ along it, from the mount end toward the tip end. The apex of the face is at "
         "$(8.5,\\,0.5,\\,10.4)$ mm, the centre of the four magnets at $(0,\\,0,\\,10.9)$ over chip U1, and the chips' "
         "sensing plane at $x=-5.37$ mm."),
        ("Pad", "The printed part of the fingertip: the front 13.45 mm of the 11/22 half-ellipsoid tip, cut flat at "
         "$x=-4.95$ mm, where it rests on the board. In concept b of the DIP-link study a cover on the DIP link clamps it "
         "over the board."),
        ("Pads 05R4 and B05R", "Pad 05R4 has four magnets of 3 × 2 mm at the corners of a 5.0 mm square centred over "
         "chip U1. The v3 tips have this pad, and v3 ID 1 was the first on the UR5. Pad B05R has five: a centre magnet and four corners, "
         "5.0 mm apart across the finger and 9.8 mm along it. Tips 0 to 2 have B05R with magnets weaker than N52, and "
         "v2 ID 1 has it with N52."),
        ("Face, edges, flanks", "Regions of the pad's surface by the direction of the outward normal $\\mathbf n$. The "
         "face looks toward the palm ($n_x>0.9$) and the flanks across the finger; the edges lie between them."),
        ("Approach", "Travel of a rigid indenter into the pad from first touch at a given normal force, in µm."),
        ("Magnet motion", "The rigid translation of a magnet, in µm, and its small rotation, in mrad, fitted to the "
         "finite-element nodes inside it. Per newton of load it is the magnet's travel in µm/N."),
        ("Reading, reading change", "The 15 field components that the five chips report, three per chip, in µT. The "
         "reading change $\\Delta\\mathbf b$ is the reading minus the reading at rest."),
        ("Pole pattern", "Which pole of each magnet faces the palm. All north: every magnet's north pole toward the palm. "
         "Checkerboard: neighbouring magnets opposite. With four magnets each diagonal pair is one way and the two pairs "
         "are opposite; with five, the centre magnet and one diagonal pair are one way and the other pair the other way."),
        ("Strength $k$", "A magnet's moment as a fraction of a 3 × 2 mm N52 cylinder's (remanence 1.45 T)."),
        ("Sensitivity $S(\\mathbf p)$", "The reading change per newton of a load at surface point $\\mathbf p$, for loads "
         "along $x$, $y$ and $z$: a 15 × 3 matrix, in µT/N."),
        ("Rest calibration", "The fit of the field model to 2 s of readings with the pad untouched. It finds which chip "
         "address sits at which place, every magnet's pole, the strength $k$ and the chips' offset from the pad."),
        ("Board offset", "How much farther the chips lie from the magnets than in the design, along $x$, in mm. The "
         "rest calibration fits it, and so does the fit to the presses."),
        ("Untrained model", "The estimator of this page on a pad's rest calibration, fitted to no press."),
        ("Fitted model", "The model with the magnets' strength, the board's offsets and a compliance scale $c$ fitted to a "
         "tip's UR5 presses up to 3.5 N. $c$ multiplies every magnet motion of the finite elements, so the fitted "
         "modulus is the FE's 12 MPa divided by $c$. Its location error is scored under five folds grouped by action."),
        ("Signal per newton", "The length of the 15-component reading change over the normal force, "
         "$|\\Delta\\mathbf b|/f_n$, in µT/N. On the UR5 $f_n$ is the HEX-E's, and the signal is a median over a region's holds "
         "of 0.5 to 3.5 N."),
        ("HEX-E, hold", "The OptoForce HEX-E six-axis force-torque sensor between the UR5's flange and the fingertip's "
         "mount, the force reference of the calibration sessions. A hold is one force level of a UR5 press, kept for "
         "about 1.5 s."),
        ("Location error", "Distance in mm from an estimated contact point to the registered one, along the surface; a "
         "median over holds."),
        ("Normal force error", "The model's normal force less the HEX-E's, in magnitude, over the HEX-E's; a median over "
         "holds, in %."),
        ("Force ratio", "Least-squares slope of the model's normal force against the HEX-E's over a region's presses; "
         "1 is exact."),
    ]
    S.append(("terms", "Terms and metrics", '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms)
              + "</dl>"))

    # ------------------------------------------------------------------------------------------------ the device
    body = (
        "<p>The pad is printed in Siraya Tech 85A thermoplastic polyurethane (TPU). Each layer has two 0.45 mm perimeters, "
        "the walls. Under the top and bottom surfaces lie four solid 0.2 mm layers, the skins. The inside, the core, is a "
        "gyroid infill at the slicer's 30 % setting. The pad is printed upright, in layers across the finger's axis. "
        "The print pauses twice so that four 3 × 2 mm N52 magnets can be pressed into pockets. Their tops lie 7.5 mm under "
        "the apex, at the corners of a 5.0 mm square centred over the board's centre chip (Figure 1). Each diagonal pair "
        "has one pole toward the palm, and the two pairs are opposite. This is pad 05R4 of the v3 tips. It was chosen on "
        "2026-10-28, after the UR5 comparison of four and five magnets (" + cal + ", Section 11). The earlier pad B05R "
        "has five magnets: a centre magnet and four corners 5.0 mm apart across the finger and 9.8 mm along it.</p>"
        "<p>The ReSkin 5X board lies under the pad's cut face. Its five MLX90393 magnetometer chips sit in a plus, 7.0 mm "
        "apart. U1 lies under the centre of the magnets' square, U2 toward the tip end, U4 toward the mount, and U3 and "
        "U5 to either side. Each chip senses the field 0.42 mm under its package top, 5.37 mm under the centre of the "
        "magnets.</p>"
        + P.fig(1, F["anatomy4"], "The fingertip and its sensor, pad 05R4. (a) Seen from the palm: the pad's outline, the "
                "five chips under the pad (dashed) and the four magnets. Each magnet's colour is the pole that faces the "
                "palm, red north and blue south. The dotted line marks the section (b). (b) Section along the finger at "
                "$y=2.5$ mm through two magnets, drawn from the slicer's material classes at 0.05 mm, the input of "
                "the finite elements. The core's lines are the gyroid's zero set at the 30 % setting's period of 3.5 mm. The "
                "chips U4, U1 and U2 (grey) lie 2.5 mm behind the section. The sphere, the UR5 calibration's indenter at "
                "first touch, is cut by the same plane.")
        + five(1, F["anatomy"], "The fingertip and its sensor, pad B05R. (a) Seen from the palm: the pad's outline, the five "
               "chips under the pad (dashed) and the five magnets. Each magnet's colour is the pole that faces the palm, "
               "red north and blue south, in the checkerboard of tips 1 and 2. (b) Section along the finger through the "
               "centre magnet, drawn from the slicer's material classes at 0.05 mm, the input of the finite elements. The "
               "core's lines are the gyroid's zero set at the 30 % setting's period of 3.5 mm. The sphere is the UR5 "
               "calibration's indenter at first touch.")
    )
    S.append(("device", "The printed pad, its magnets and the magnetometer board", body))

    # ------------------------------------------------------------------------------------------------ the chain
    body = (
        "<p>The model predicts the readings for a load of known place and size. It is a chain of four linear steps "
        "(Figure 2). The print recipe fixes the material in every voxel. The finite elements turn a load at a surface "
        "point $\\mathbf p$ into a rigid motion of each magnet. The field model turns those motions into the change of "
        "the 15 readings. A pad's rest calibration supplies the field model's unknowns: the poles, the strength and the "
        "chips' places. The whole chain is one matrix per surface point, the sensitivity:</p>"
        "$$\\Delta\\mathbf b \\;=\\; \\underbrace{J}_{15\\times24}\\;\\underbrace{M(\\mathbf p)}_{24\\times3}\\;\\mathbf f "
        "\\;=\\; S(\\mathbf p)\\,\\mathbf f .$$"
        "<p>Here $\\mathbf f$ is the force on the pad in newtons. $M(\\mathbf p)$ contains the translation and rotation of "
        "the four magnets per newton of load along $x$, $y$ and $z$, from the finite elements. $J$ contains the change of "
        "each reading per micrometre and per milliradian of each magnet's motion, from the field model. With five "
        "magnets $J$ is 15 × 30 and $M(\\mathbf p)$ 30 × 3. Both maps are linear for the micrometre motions of a press. "
        "The chain is inverted at every frame of the board: the estimator searches for the point and the force whose "
        "predicted change fits the measured one best.</p>"
        + P.fig(2, F["chain4"], "The physical model of pad 05R4 as a chain. The top row is computed once per pad design and "
                "printed recipe. The bottom row runs on the board's stream: the rest calibration fixes the field model's "
                "unknowns, and the estimator compares each reading change with $S(\\mathbf p)\\,\\mathbf f$ at every "
                "candidate point.")
        + five(2, F["chain"], "The physical model of pad B05R as a chain, as in Figure 2 with 30 numbers of magnet motion "
               "per load.")
    )
    S.append(("chain", "The model as a chain of linear maps", body))

    # ------------------------------------------------------------------------------------------------ FE
    body = (
        "<p>A printed pad is not a uniform solid. Its walls and skins are solid TPU, and its core is a lattice with a third "
        "of the material. Under a contact the walls and skins spread the load over the core. Hertz's law and a uniform "
        "layer therefore miss both the surface's sinking and the magnets' motion. The finite-element (FE) model "
        "resolves the three structures as the slicer prints them. Its scripts are our own Python with SciPy and "
        "CHOLMOD (" + P.code("scripts/tpu_tip_fem_core.py") + ", " + P.code("scripts/tpu_tip_fem_tip.py") + ").</p>"
        "<p><b>Material classes.</b> The tip's shape is voxelized at 0.05 mm. In each layer, TPU within 0.9 mm of the "
        "layer's outline is wall. TPU within 0.8 mm of a surface exposed along the build axis is skin. The rest is core, "
        "and walls and skins also line every magnet pocket. Walls and skins are isotropic solid TPU with a modulus "
        "$E_s=12$ MPa and Poisson's ratio 0.45, from the NinjaFlex 85A data sheet; Siraya's 85A has not been measured. "
        "The magnets are bonded NdFeB at 160 GPa.</p>"
        "<p><b>Homogenized core.</b> The gyroid core is replaced by an equivalent solid with the same average stiffness. "
        "A periodic cell of the printed gyroid, beads of 0.45 mm on the slicer's toolpath, is strained by each of the six "
        "unit strains $\\boldsymbol\\varepsilon_i$. The cell's periodic displacement $\\mathbf u_i$ and its mean stress "
        "give the effective 6 × 6 tensor:</p>"
        "$$K\\,\\mathbf u_i=-\\mathbf f_i,\\qquad C^{*}\\boldsymbol\\varepsilon_i=\\frac{1}{V}\\int_V "
        "D\\,\\bigl(\\boldsymbol\\varepsilon_i+\\boldsymbol\\varepsilon(\\mathbf u_i)\\bigr)\\,dV .$$"
        "<p>Here $K$ is the cell's stiffness matrix, $\\mathbf f_i$ the load that the unit strain puts on its nodes and $D$ "
        "the solid TPU's elasticity matrix. The 30 % setting prints 32.6 % solid. Its modulus is $0.087\\,E_s$ across the "
        "layers' plane and $0.135\\,E_s$ along the build axis, about 1.0 and 1.6 MPa (Figure 3). The core is therefore "
        "anisotropic, and the solver rotates its tensor into the tip frame.</p>"
        + P.fig(3, F["moduli"], "Moduli of the printed gyroid against the core's solid fraction, in the layer plane "
                "($E_x$) and along the build axis ($E_z$), from homogenized cells; the 30 % setting prints 32.6 % solid. "
                "Gibson and Ashby's bending and stretching limits bracket the fits. From " + link(L["fe"], "the finite "
                "elements of the printed tip") + ".")
        + "<p><b>Tip model and contact.</b> The pad is meshed with eight-node hexahedra on a graded grid, 0.25 mm under "
        "the face and about 1 mm deep inside: 38,144 elements and 119,061 unknowns. The elements have incompatible modes, "
        "so the thin walls bend without locking. The cut face and the mount face are fixed. One sparse Cholesky "
        "factorization of $K$ is reused for every load. A rigid frictionless indenter meets the face nodes. With $W$ the "
        "compliance between those nodes and $\\mathbf g_0$ their gaps at first touch, the nodal contact forces "
        "$\\mathbf p$ at approach $\\delta$ solve a linear complementarity problem:</p>"
        "$$\\mathbf p\\ge0,\\qquad \\mathbf g_0-\\delta\\,\\mathbf 1+W\\mathbf p\\ge0,\\qquad "
        "\\mathbf p^{\\top}\\bigl(\\mathbf g_0-\\delta\\,\\mathbf 1+W\\mathbf p\\bigr)=0 .$$"
        "<p>The approach $\\delta$ is bisected until the forces sum to the load. Each magnet's motion is the least-squares "
        "rigid motion of its nodes. The model is linear and small-strain. A neo-Hookean check on the softest tip lost 11 % "
        "of the force at 350 µm of approach.</p>"
        "<p><b>What a load does.</b> Figure 4 shows pad 05R4 under three loads of 1 N at the apex. Under the 5 mm sphere "
        f"the face sinks {p1['approach_um']:.0f} µm, and {p3['approach_um']:.0f} µm at 3 N. Most of that travel is local: "
        f"the magnets, 7.5 mm down, move {rng(p1['magnet_travel_um'])} µm, almost all of it toward the board. In pad B05R "
        f"they move {rng(fe5['press1']['magnet_travel_um'])} µm, the mount-end pair least. A shear along the finger moves the four magnets {rng(sz['magnet_travel_um'])} µm and tilts "
        "the magnet set about $y$: the tip-end magnets sink, the mount-end ones rise. A shear across the finger moves "
        f"them {rng(sy['magnet_travel_um'])} µm sideways, three to four times their travel under a press: the narrow "
        "half-ellipsoid bends sideways on its fixed base.</p>"
        + P.fig(4, F["fe4"], "The finite elements of pad 05R4 under 1 N at the apex. Left: displacement in the section "
                "$y=2.5$ mm through two magnets (outlined in blue), with the model's cells in white. The black line is "
                "the face deformed, with the largest displacement drawn as 1.2 mm; the grey line is the face at rest. The "
                "blue arrows are the two magnets' motions, 300 times magnified. Right: the four magnets seen from the "
                "palm, with the load as a star. The arrow is each magnet's motion in the face's plane at 0.25 mm per µm. "
                "The number is its motion along $x$ in µm, negative toward the board.")
        + five(4, F["fe"], "The finite elements of pad B05R under 1 N at the apex. Left: displacement in the section "
               "through the centre magnet (outlined in blue), with the model's cells in white. The black line is the "
               "face deformed, with the largest displacement drawn as 1.2 mm; the grey line is the face at rest. The blue "
               "arrow is the centre magnet's motion, 300 times magnified. Right: all five magnets seen from the palm, "
               "with the load as a star. The arrow is each magnet's motion in the face's plane at 0.25 mm per µm. The "
               "number is its motion along $x$ in µm, negative toward the board.")
        + "<p><b>Checks.</b> On the first study's 95A block the model is 1.9 % off Hertz's solution under a 40 mm sphere. "
        "The 85A pads were also solved with every gyroid bead resolved at 0.1 mm, with 5–8 million unknowns on the GPU. "
        "They are then 1.6–4.8 % stiffer under the rods than the homogenized model, and up to 7.8 % stiffer under a "
        "2 mm sphere. A pad modelled "
        "from its G-code, with the TPU volume set to the weighed parts, agrees with the homogenized pad within 1 % "
        "under the rod. Details are on " + link(L["fe"], "the finite-element page") + " and "
        + link(L["v2"], "the v2 fingertip checks") + ".</p>"
    )
    S.append(("fe", "Elasticity of the printed pad from finite elements", body))

    # ------------------------------------------------------------------------------------------------ magnets
    body = (
        "<p>Each magnet is a uniformly magnetized cylinder. Outside a magnet, its field equals that of point dipoles "
        "spread through its volume. The model uses 72 dipoles per magnet; a single dipole reads 5.5 % high at the chips' "
        "distance. With $\\mathbf r=\\mathbf x-\\mathbf c_j-\\boldsymbol\\rho_q$ the vector from dipole $q$ of magnet "
        "$j$ to the field point,</p>"
        "$$\\mathbf B(\\mathbf x)=\\frac{\\mu_0}{4\\pi}\\sum_{j=1}^{4}\\sum_{q=1}^{72} w_q\\,"
        "\\frac{3\\,\\hat{\\mathbf r}\\,(\\hat{\\mathbf r}\\cdot\\mathbf m_j)-\\mathbf m_j}{r^{3}},\\qquad "
        "\\mathbf m_j=s_j\\,k\\,\\frac{B_r V}{\\mu_0}\\,\\mathbf e_x .$$"
        "<p>Here $s_j=\\pm1$ is the magnet's pole, $k$ its strength, $B_r=1.45$ T the N52 remanence and $V$ the magnet's "
        "volume; the sum runs to 5 for pad B05R. A magnet that moves by $\\mathbf t_j$ and turns by "
        "$\\boldsymbol\\omega_j$ shifts its dipoles and turns its moment. Chip $c$ at $\\mathbf x_c$ reports the field "
        "in its own axes, $R_c$, together with Earth's field $\\mathbf g$:</p>"
        "$$\\mathbf b_c=R_c\\bigl(\\mathbf B(\\mathbf x_c)+\\mathbf g\\bigr),\\qquad "
        "J=\\frac{\\partial\\mathbf b}{\\partial(\\mathbf t_1,\\boldsymbol\\omega_1,\\dots,\\mathbf t_4,\\boldsymbol\\omega_4)} .$$"
        "<p>The field falls as the cube of distance, so the nearest magnets dominate each chip. With four N52 magnets all "
        f"north the centre chip U1 sits in {fc4['all_N']['U1']:.1f} mT and the other four chips in "
        f"{fc4['all_N']['U2']:.1f} mT (Figure 5c). In the checkerboard the four fields cancel at U1, and the other chips "
        f"sit in {fc4['checkerboard']['U2']:.1f} mT (Figure 5d). A press changes these fields by tens of µT.</p>"
        "<p>The pole pattern shapes the field. With all magnets one way their fields add, and far from the fingertip "
        "the pad looks like one dipole of four magnets' moment. In the four-magnet checkerboard the moments cancel, so "
        "the field falls off faster with distance, and between neighbouring magnets it changes sign. A neighbouring "
        "fingertip therefore adds less field to the board. In the field model a touching neighbour that faces the pad "
        "adds 38 µT at the board with four magnets and 513 µT with five. Between two v3 tips on the UR5, 39 to 60 µT "
        "were measured (" + link(L["xt"], "crosstalk page") + "). At the face centre the checkerboard gives less signal. "
        f"A 1 N press there changes the readings by {pr['checker_N52_four']:.0f} µT with the checkerboard and "
        f"{pr['all_N_N52_four']:.0f} µT with all north. With five magnets the same press gives {pr['checker_N52']:.0f} "
        f"and {pr['all_N_N52']:.0f} µT.</p>"
        + P.fig(5, F["field4"], "Field of four N52 magnets, pad 05R4. (a, b) Field lines and field magnitude in the "
                "section $y=2.5$ mm, which cuts two magnets; darker grey is stronger. The tan lines are the pad's face and "
                "cut face. The chips U4, U1 and U2 (grey) lie 2.5 mm behind the section. (c, d) The field at the chips' "
                "sensing plane: colour is the component normal to the board, arrows the component in its plane, and the "
                "numbers each chip's field magnitude in mT. The thin line is the pad's outline.")
        + five(5, F["field"], "Field of five N52 magnets, pad B05R. (a, b) Field lines and field magnitude in the section "
               "$y=0$, which cuts only the centre magnet; darker grey is stronger. The tan lines are the pad's face and cut "
               "face, and the dark rectangles the chips U4, U1 and U2. (c, d) The field at the chips' sensing plane: "
               "colour is the component normal to the board, arrows the component in its plane, and the numbers each "
               "chip's field magnitude in mT. The thin line is the pad's outline.")
    )
    S.append(("magnets", "Field of the embedded magnets", body))

    # ------------------------------------------------------------------------------------------------ chips
    body = (
        "<p>An MLX90393 measures the magnetic field with Hall elements. A current through a semiconductor plate is "
        "deflected by the field. The voltage across the plate is then proportional to the field's component normal to "
        "the plate. Flux concentrators on the die turn the field's in-plane components onto further Hall elements, so one "
        "chip reports all three components. The firmware sets gain GAIN_SEL 1 and resolution RES 1: 1.20 µT a count on "
        "$x$ and $y$, 1.94 µT on $z$, and a range of 38.4 mT. The range decides the gain. With N52 magnets all north the "
        "centre chip sits in 27 mT, and the most sensitive gain covers only 4.8 mT. With four magnets in the checkerboard "
        "the chips sit in at most 5.0 mT in the design, and in 0.6–4.4 mT on v3 ID 1 at rest.</p>"
        "<p>Each chip converts every 7.0–7.1 ms (141 Hz), and the board streams all five over I²C to a QT Py ESP32-S3. "
        "At rest the noise of one conversion is 1.5–1.8 µT on $x$ and $y$ and 2.3–2.6 µT on $z$. Three disturbances are "
        "larger than the noise:</p>"
        "<ul><li>Temperature. The chips run without their on-chip compensation, and each chip's gain falls by "
        "0.65–0.70 % per °C. In a field of 7.6 mT one degree changes the reading by 53 µT, the change of a 2.5 N press "
        "on tip 2. The estimator divides each reading by $1+\\kappa_c(T_c-25\\,^{\\circ}\\mathrm C)$, with $T_c$ "
        "the chip's die temperature (" + link(L["est"], "estimation page") + ", Section 9).</li>"
        "<li>Earth's field, about 44 µT, turns with the finger. It is the same at all five chips, so it is estimated with "
        "the contact and removed.</li>"
        "<li>Nearby magnets and steel. Another fingertip's magnets or a steel tool add a field that varies slowly across "
        "the 14 mm board. A test against such fields rejects most steel touches "
        "(" + link(L["steel"], "page on fields from outside the pad") + ").</li></ul>"
    )
    S.append(("chips", "The magnetometers and their disturbances", body))

    # ------------------------------------------------------------------------------------------------ patterns
    a, b_, c_, d_ = (pat4[k] for k in ("(a) Press at the face centre", "(b) Press 5 mm toward the tip end",
                                      "(c) Press on the edge, −y side", "(d) Press on the +y flank"))
    e_, f_ = pat4["(e) Shear along the finger"], pat4["(f) Shear across the finger"]
    side = (c_["norm_uT"], d_["norm_uT"])
    body = (
        "<p>A reading change is a vector of 15 numbers. Its length grows in proportion to the force, and its direction "
        "depends on where the load acts and in which direction. Figure 6 draws the change at each chip for six loads of "
        "1 N. A press at the face centre moves the four magnets toward the board by nearly equal amounts. Each chip lies "
        "between magnets of both poles, so the changes from the four magnets partly cancel, and the readings change by "
        f"{a['norm_uT']:.0f} µT. A press 5 mm toward the tip end moves the tip-end pair more than the mount-end pair. "
        f"U1, between the pairs, then changes most: {b_['per_chip_uT'][0]:.0f} µT of the {b_['norm_uT']:.0f} µT. A press "
        "on an edge or a flank pushes the pad sideways, the direction in which its magnets move most (Figure 4c). It "
        f"changes the readings by {min(side):.0f}–{max(side):.0f} µT, "
        f"{min(side) / a['norm_uT']:.0f} to {max(side) / a['norm_uT']:.0f} times a face press, and U2 and U4 by up to "
        f"{max(d_['per_chip_uT'][1], d_['per_chip_uT'][3]):.0f} µT with opposite signs. A shear along the finger tilts the "
        f"magnet set (Figure 4b) and changes the readings by {e_['norm_uT']:.0f} µT. A shear across the finger shifts "
        f"all four magnets sideways and changes them by {f_['norm_uT']:.0f} µT. With five magnets the face-centre press "
        f"gives {pat5['(a) Press at the face centre']['norm_uT']:.0f} µT, because the centre magnet moves toward U1, and "
        "an edge or a flank press about twice that.</p>"
        "<p>The estimator separates the contact point from the force by the direction of $\\Delta\\mathbf b$ and the force's "
        "size by its length. The unknowns are three force components, two coordinates on the surface, a uniform field "
        "change of three components and a common change of the magnets' strength: nine numbers against 15 readings. "
        "Two points are told apart when their patterns differ by more than the noise. Neighbouring points differ little, "
        "so the noise sets how finely a point can be placed. The weaker a press, the larger the region whose patterns "
        "fit its reading.</p>"
        + P.fig(6, F["patterns4"], "Reading change at the five chips for 1 N loads, four N52 magnets in the checkerboard "
                "(dotted circles, red north toward the palm), seen from the palm. Each chip's change is drawn in the tip "
                "frame. The filled circle's area is the change normal to the board, red toward the palm and blue toward "
                "the board. The arrow is the change in the board's plane. The star marks the load, and the orange arrow "
                "the shear's direction. The number is the length of the 15-component change. Keys in (a): an arrow and a "
                "circle of 50 µT.")
        + five(6, F["patterns"], "Reading change at the five chips for 1 N loads, five N52 magnets in the checkerboard, "
               "seen from the palm. Each chip's change is drawn in the tip frame. The circle's area is the change normal "
               "to the board, red toward the palm and blue toward the board. The arrow is the change in the board's "
               "plane. The star marks the load, and the orange arrow the shear's direction. The number is the length of "
               "the 15-component change. Keys in (a): an arrow and a circle of 50 µT.")
        + "<p>On the built tips the change is smaller than in the design. With v3 ID 1's rest calibration, a strength of "
        "1.07 and the board 1.03 mm farther from the magnets, a press at the face centre changes the readings by "
        f"{pr['as_built_v3id1']:.0f} µT per newton. On the UR5 the presses at the face centre changed v3 ID 1's readings "
        f"by {cen['v3id1']['measured']:.1f} µT per newton, and its other presses on the face by "
        f"{gn['v3id1'][FACE]['measured']:.1f} (" + '<a href="#ur5">Section 9</a>' + "). Tips 0 to 2 have five magnets "
        "weaker than N52. Their strengths from the rest "
        "calibrations are 0.34 on tip 0 and 0.47–0.57 on tips 1 and 2. A press at the face centre of tip 2 changes the "
        f"readings by {pr['as_built_tip2']:.0f} µT per newton. v2 ID 1's five N52 magnets fit at 0.98 of an N52's "
        "strength.</p>"
    )
    S.append(("patterns", "How a reading encodes the contact point and the force", body))

    # ------------------------------------------------------------------------------------------------ estimator
    d4, d5 = lk4["draws100"], lk5["draws100"]
    body = (
        "<p>The estimator inverts the chain for one reading change. Its model of the change adds two nuisance terms and "
        "the noise:</p>"
        "$$\\Delta\\mathbf b=S(\\mathbf p)\\,\\mathbf f+U\\,\\Delta\\mathbf g+\\mathbf b^{\\mathrm{rest}}\\varepsilon+\\mathbf n .$$"
        "<p>$U\\,\\Delta\\mathbf g$ is a uniform field change seen through the five chips' axes, for Earth's field as the "
        "finger turns. $\\mathbf b^{\\mathrm{rest}}\\varepsilon$ is a common change of the magnets' strength, for "
        "temperature. Both have Gaussian priors, and together with the noise they give the covariance $C$. For a "
        "candidate point the force follows in closed form, and the point is chosen by the log marginal likelihood "
        "$\\ell(\\mathbf p)$ of the reading:</p>"
        "$$\\hat{\\mathbf f}(\\mathbf p)=\\bigl(G+\\sigma_f^{-2}I\\bigr)^{-1}\\mathbf t,\\qquad "
        "\\ell(\\mathbf p)=\\mathbf t^{\\top}\\hat{\\mathbf f}-\\ln\\det\\bigl(I+\\sigma_f^{2}G\\bigr),\\qquad "
        "G=S^{\\top}C^{-1}S,\\quad \\mathbf t=S^{\\top}C^{-1}\\Delta\\mathbf b .$$"
        "<p>The force prior $\\sigma_f=3$ N and the determinant penalize points where the reading barely responds; "
        "without them a flank point could explain any change with a force of 100 N. The force must press, lie in a "
        "friction cone of 2 and stay under 25 N. The search runs over 12,581 points about 0.2 mm apart; $S(\\mathbf p)$ "
        "between the 812 FE points is interpolated by a thin-plate spline. One estimate takes 1.2 ms.</p>"
        "<p>Figure 7 shows $\\ell(\\mathbf p)$ over the pad for one simulated press at the face centre of v3 ID 1, with "
        "the noise of the viewer's 50 ms window. The model has v3 ID 1's magnets as fitted at rest. At 5 N the marginal "
        f"likelihood has one sharp peak, {lk4['5N']['error_mm']:.1f} mm from the true point. At 2 N it is high from the "
        "face centre to the pad's mount end, and in this draw the estimate lies "
        f"{lk4['2N']['error_mm']:.1f} mm off at the mount end. At 0.5 N no point near the truth explains the reading with "
        "a pressing force, and the estimate goes to the edge of the pad. Over 100 draws of the noise the median error is "
        f"{d4['5N']['median_mm']:.2f} mm at 5 N, {d4['2N']['median_mm']:.2f} mm at 2 N and {d4['1N']['median_mm']:.1f} mm "
        f"at 1 N. At 2 N, {100 * d4['2N']['within_1mm']:.0f} % of the draws lie within 1 mm, and 10 % lie "
        f"{d4['2N']['p90_mm']:.0f} mm or more away. On tip 2, with five weaker magnets, the medians are "
        f"{d5['5N']['median_mm']:.2f}, {d5['2N']['median_mm']:.2f} and {d5['1N']['median_mm']:.1f} mm.</p>"
        + P.fig(7, F["likelihood4"], "Log marginal likelihood of a contact at each surface point, for one simulated press "
                "at the face centre (star) of v3 ID 1. The model has v3 ID 1's four magnets as fitted at rest and the "
                "viewer's noise: a 50 ms window and a 2 µT floor. Grey points cannot explain the reading with a force that "
                "presses inside the friction cone. The cross is the estimate.")
        + five(7, F["likelihood"], "Log marginal likelihood of a contact at each surface point, for one simulated press at "
               "the face centre (star) of tip 2. The model has tip 2's five magnets as built and the viewer's noise: a "
               "50 ms window and a 2 µT floor. Grey points cannot explain the reading with a force that presses inside the "
               "friction cone. The cross is the estimate.")
        + "<p><b>Rest calibration.</b> Before any press, the estimator needs the field model of the pad as built. With "
        "the pad untouched the 15 readings are the magnets' field and Earth's. The fit searches the pole patterns, 8 with "
        "four magnets and 16 with five, the 24 assignments of chip addresses to places and the chips' common turn. For "
        "each it fits the strength, the offset and Earth's field by least squares, and the best reproduces the reading at "
        f"rest. On v3 ID 1 the planned checkerboard fits at {rs['v3id1']['free']['rms_mT']:.2f} mT RMS, against "
        f"{rs['v3id1']['free']['rms_next_mT']:.2f} mT for the next pattern. On v3 ID 2 it fits at "
        f"{rs['v3id2']['free']['rms_mT']:.2f} against {rs['v3id2']['free']['rms_next_mT']:.2f} mT. The rest field of four "
        "alternating magnets fixes their strength and the board's offset only together. A free fit gives a strength at "
        f"its bound of 2.0, with the board {rs['v3id1']['free']['dx']:.2f} mm farther away on v3 ID 1 and "
        f"{rs['v3id2']['free']['dx']:.2f} mm on v3 ID 2. With the offset held near v2 ID 1's, on the same board mount, "
        f"the strengths are {rs['v3id1']['bounded']['k']:.2f} and {rs['v3id2']['bounded']['k']:.2f} at "
        f"{rs['v3id1']['bounded']['dx']:.2f} mm. Figures 7 and 8 and Table 1 have this calibration. The presses put the "
        f"boards {ft['v3id1']['board_dx']:.2f} and {ft['v3id2']['board_dx']:.2f} mm farther away "
        "(" + '<a href="#ur5">Section 9</a>' + "). With the offset held there, the rest fit gives a strength of "
        f"{k_at}, and its residual "
        f"grows by {rs['v3id1']['at_press_board']['rms_mT'] - rs['v3id1']['free']['rms_mT']:.3f} and "
        f"{rs['v3id2']['at_press_board']['rms_mT'] - rs['v3id2']['free']['rms_mT']:.3f} mT. On tip 1, with five magnets, the planned checkerboard fits at 0.24 mT and the next "
        "pattern at 0.87 mT. Its strength and the magnets' depth are also found only together. A strength of 0.57 with "
        "the chips 1.9 mm farther away and one of 0.34 at 0.7 mm fit 0.09 mT apart.</p>"
        "<p>On simulated presses with five N52 magnets all north, 2 % of presses are placed more than 1.5 mm off. The "
        "others have median errors of 0.18 mm and 0.03 N, and the errors follow the model's 1-sigma within 10–16 % "
        "(" + link(L["est"], "estimation page") + ", Section 6). These figures assume a perfect model.</p>"
    )
    S.append(("estimator", "Inverting the model for the contact point and the force", body))

    # ------------------------------------------------------------------------------------------------ UR5
    reg4 = [100 * a4[r]["fn_rel"] for r in ("face", "edge")]
    reg5 = [100 * a5[r]["fn_rel"] for r in ("face", "edge")]
    body = (
        "<p>On the UR5 the pad was pressed onto a 5 mm sphere at registered points with the HEX-E as reference "
        "(" + cal + "). v3 ID 1, with four N52 magnets, ran the grid stage of the suite on the night of 2026-10-27: 160 "
        "press actions with 818 holds. v2 ID 1, with five, ran the same actions. Figure 8 scores the untrained model on "
        f"both tips' holds. From 1 N up it places v3 ID 1's contacts within {rng(loc4)} mm (medians by force), against "
        f"{rng(loc5)} mm with five magnets. Under 1 N it places them {a4['0-1']['loc_mm']:.1f} mm away, against "
        f"{a5['0-1']['loc_mm']:.1f} mm. Its normal force errs by a median of {min(reg4):.0f}–{max(reg4):.0f} % on the face "
        f"and edges with four magnets, against {min(reg5):.0f}–{max(reg5):.0f} % with five. On the flanks it errs by "
        f"{100 * a4['flank']['fn_rel']:.0f} and {100 * a5['flank']['fn_rel']:.0f} %.</p>"
        + P.fig(8, F["ur54"], "The untrained model on the same 160 grid actions of the low sphere, with 818 press holds "
                "on each tip. Orange: four N52 magnets (v3 ID 1); blue: five (v2 ID 1). (a) Median location error by the "
                "hold's normal force. (b) Median normal force error by region. The model is the FE and each tip's rest "
                "calibration, with no press in training. Rows in "
                + P.code("docs/experiments/20261027-v3_tip_calibration_and_crosstalk/data/20261028-fe_calibration.json")
                + ", from the UR5 coordinator's " + P.code("scripts/ur5_tip_calibration_fe_calibration.py") + ".")
        + "<p>The first tips, 0 to 2, have five magnets weaker than N52. Four sessions on them (Figure 8 with five "
        "magnets) place contacts within 0.9–2.4 mm from 2 N up, by force level. The model's normal force is 0.78–1.31 of "
        "the HEX-E's by region. At 0.25 and 0.5 N it places contacts a median 8.7–11.7 mm away. On tip 0 each press left "
        "the reading at rest about 7 µT off, the change of a 0.3 N press. Below 1 N that step is as large as the press "
        "itself.</p>"
        + five(8, F["ur5"], "The untrained model on the UR5 sessions of tips 0 to 2. (a) Median location error by force "
               "level. (b) The model's normal force over the HEX-E's, by region. Tip 0 (session sphere2) has its magnets "
               "all one way; tips 1 and 2 have the checkerboard. Tip 1 ran twice, on 2026-10-23 and 2026-10-25.")
        + "<p>The second four-magnet tip, v3 ID 2, ran the suite on the night of 2026-10-28 (" + cal + ", Section 12). "
        f"On the low sphere it shares {vc['shared_actions']} press actions with v3 ID 1 and v2 ID 1, "
        f"{ft['v3id2']['holds']} holds on each tip. Figure 8's 160 actions include 47 speed variants and a dense press "
        "that v3 ID 2 did not run. Table 1 scores the model on the shared holds. From 1 N up the untrained model places "
        f"v3 ID 2's contacts within {rng(loc_by('v3id2'))} mm, against {rng(loc_by('v3id1'))} mm on v3 ID 1. Under 1 N "
        f"it places them {un('v3id2', '0-1')['loc_mm']:.1f} and {un('v3id1', '0-1')['loc_mm']:.1f} mm away. Its normal "
        f"force errs by {pct(fn_by('v3id2'))} % on v3 ID 2 and {pct(fn_by('v3id1'))} % on v3 ID 1. The two tips' readings "
        f"change by {vc['gain_1_2_tip_vs_tip']['v3id2']:.1f} and {vc['gain_1_2_tip_vs_tip']['v3id1']:.1f} µT per newton "
        "at 1 to 2 N (" + cal + ", Section 12). Fitted to the presses, v3 ID 2 has a compliance scale of "
        f"{ft['v3id2']['c']:.2f} against v3 ID 1's {ft['v3id1']['c']:.2f}, a modulus of {ft['v3id2']['E_fit_MPa']:.1f} "
        f"against {ft['v3id1']['E_fit_MPa']:.1f} MPa. Its board sits {ft['v3id1']['board_dx'] - ft['v3id2']['board_dx']:.2f} "
        f"mm closer to the magnets, and its strength is {100 * (ft['v3id2']['k'] / ft['v3id1']['k'] - 1):.0f} % higher. "
        "Under five folds grouped by action, the fitted model's location error over all holds is "
        f"{ft['v3id1']['cv']['all']['loc_mm']:.2f} and {ft['v3id2']['cv']['all']['loc_mm']:.2f} mm on v3 ID 1 and ID 2. "
        f"Untrained it is {ft['v3id1']['untrained']['all']['loc_mm']:.2f} and "
        f"{ft['v3id2']['untrained']['all']['loc_mm']:.2f} mm.</p>"
        + P.table(["Tip, magnets", "Board offset (mm)", "Strength at rest, board held there", "Compliance scale $c$",
                   "Location error, untrained, 0–1 / 1–2 / 2–3.5 / 3.5–8 N (mm)",
                   "Normal force error, untrained, same bins (%)", "Location error, all holds, untrained / fitted (mm)"],
                  [[nm, f"{ft[k]['board_dx']:.2f}", f"{rs[k]['at_press_board']['k']:.2f}", f"{ft[k]['c']:.2f}",
                    " / ".join(f"{un(k, b)['loc_mm']:.1f}" for b in BINS4),
                    " / ".join(f"{100 * un(k, b)['fn_rel']:.0f}" for b in BINS4),
                    f"{ft[k]['untrained']['all']['loc_mm']:.2f} / {ft[k]['cv']['all']['loc_mm']:.2f}"] for k, nm in TIP3],
                  num_cols=(1, 2, 3, 4, 5, 6), n=1,
                  caption=f"The model on the {vc['shared_actions']} press actions of the low sphere that the three N52 "
                  f"tips ran, {ft['v3id1']['holds']} holds on each tip. Board offset and $c$: the fitted model's. Strength "
                  "at rest: the rest fit with the board held within 0.02 mm of that offset. Errors are medians, by the "
                  "hold's normal force. Rows in "
                  + P.code("docs/experiments/20261026-fingertip_physical_model/data/20261029-v3_tips_model_check.json")
                  + ", from " + P.code("scripts/fingertip_physical_model_v3_check.py") + ", which runs the UR5 "
                  "coordinator's fit (" + P.code("scripts/ur5_tip_calibration_fe_calibration.py") + ") on each tip.")
        .replace("<th class=n>", '<th class=n style="white-space:normal">')
        + "<p>Figure 9 compares the signal per newton by region of the pad. For each hold the model predicts the reading "
        "change from the registered contact point and the HEX-E's force. Fitted to the presses, it gives the face, the "
        f"edges, the flanks and the tip end within {within} % of the measured medians on all three tips. At the face "
        f"centre the four-magnet tips gave {cen['v3id1']['measured']:.1f} and {cen['v3id2']['measured']:.1f} µT per "
        f"newton, and the fitted model gives {cen['v3id1']['fitted']:.1f} and {cen['v3id2']['fitted']:.1f}. Their other "
        f"presses on the face gave {gn['v3id1'][FACE]['measured']:.1f} and {gn['v3id2'][FACE]['measured']:.1f} µT per "
        f"newton. So the measured centre gives {ratio('v3id1', 'measured'):.2f} and {ratio('v3id2', 'measured'):.2f} of "
        f"the face's signal, and the model {ratio('v3id1', 'fitted'):.2f} and {ratio('v3id2', 'fitted'):.2f}. The "
        "cancellation at U1 is deeper in the printed tips than in the model. With five magnets the face centre gave "
        f"{cen['v2id1']['measured']:.1f} µT per newton, {ratio('v2id1', 'measured'):.2f} times the face's, and the "
        f"fitted model gives {cen['v2id1']['fitted']:.1f}.</p>"
        + P.fig(9, F["gain"], "Signal per newton by region of the pad on the low sphere: the median of "
                "$|\\Delta\\mathbf b|/f_n$ over the holds of 0.5 to 3.5 N. Bars are measured. The black line is the "
                "fitted model and the dashed line the untrained model; each predicts a hold's change from its "
                "registered contact point and the HEX-E's force. Face centre: the reference presses at the face's "
                f"centre. Face, edges and flanks: the grid stage's presses on the {vc['shared_grid_actions']} grid "
                "actions that all three tips ran. Tip end: v3 ID 2's top-edge pass. Groups as in "
                + P.code("scripts/ur5_tip_calibration_gain_by_region.py") + "; rows in "
                + P.code("data/20261029-v3_tips_model_check.json") + ".")
        + "<p>The registered indentation is larger than the FE pad's at the measured force. On tip 0, at the registered "
        "depth, the FE pad would push back with 5.1 times the measured force. A pad that much softer would move its "
        "magnets five times as far per newton, and the board would read five times the HEX-E's force. It reads "
        "0.78–1.31 of it, so the pad's bulk follows the FE model within about 30 %. The extra indentation, 0.33 mm plus "
        "0.20 mm per newton under the sphere, lies where the magnets do not move. Under a rod of radius 12.5 mm it is "
        "0.07 mm plus 0.10 mm per newton. A spring in series with the pad, in the mount, the arm or the indenter's stand, "
        "would add the same slope under both indenters. So at most 0.10 mm/N lies in series. At least 0.10 mm/N arises "
        "at the sphere's small contact, where the print is softer than the FE pad ("
        + link(L["gt"], "UR5 ground truth") + ", Sections 3, 4 and 8). On the three N52 tips the fitted modulus is "
        f"{min(E_fit):.1f}–{max(E_fit):.1f} MPa, against the FE's 12 MPa (Table 1). The registered indentation of the "
        f"same presses grows {min(give):.1f} to {max(give):.1f} times as fast as the FE pad's.</p>"
        "<p>At 2 N and over, most of the untrained model's location error is geometric. On tip 1 an affine map of the "
        "estimated points takes the error from 1.16 to 0.60 mm, with a scale of 0.68 and a turn of 14.7°. The magnets' "
        "places and depth, which the rest calibration fixes only jointly with their strength, are the likely source. "
        "Trained on 47 presses of the same tip, a Gaussian process places held-out contacts within 0.71 mm and an "
        "ensemble of small neural networks within 0.42 mm. On v3 ID 1 a Gaussian process of the change's direction "
        "places contacts within 0.26 mm, and 0.28 mm on v2 ID 1. The learned models, their training data and their "
        "transfer between tips are on " + link(L["cal"], "the calibration page") + ".</p>"
    )
    S.append(("ur5", "The model against the UR5 presses", body))

    # ------------------------------------------------------------------------------------------------ limitations
    body = (
        "<p>The FE gives Siraya 85A the 12 MPa of a NinjaFlex data sheet; fitted to the N52 tips' presses, the modulus "
        f"is {min(E_fit):.1f}–{max(E_fit):.1f} MPa. It is linear and frictionless at the indenter. It fixes the pad's cut face, so a pad that shifts in its "
        "seat is outside the model; the owner's first presses by hand shifted it by 17–23 µm. The contact patch under a "
        "small sphere is resolved by about 15 face nodes at 1 N. The field model assumes the design's magnet places. The "
        "rest fit trades their strength against their depth, and with four alternating magnets against the board's "
        "offset as well. The temperature correction uses each chip's die temperature, and the magnets' own coefficient, "
        "about 0.12 % per °C, follows the pad's lagging temperature. The estimator assumes one contact.</p>"
    )
    S.append(("limits", "Limitations", body))

    # ------------------------------------------------------------------------------------------------ future work
    items = [
        "Board offset on the four-magnet tips: measure the gap between the pad's cut face and the board's chips on "
        "v3 ID 1 and ID 2 with feeler gauges. Convert the gap to the board offset. The presses put the offsets at "
        f"{ft['v3id1']['board_dx']:.2f} and {ft['v3id2']['board_dx']:.2f} mm, where the rest fit gives a strength of "
        f"{k_at}. Gauge offsets within "
        "0.15 mm of these confirm the field model of four magnets. Offsets near the free rest fits' "
        f"{rs['v3id1']['free']['dx']:.2f} and {rs['v3id2']['free']['dx']:.2f} mm put the error in the magnets' depth.",
        "Depth of the cancellation at the face centre: in the fit to the presses, let each of the four magnets move by "
        "up to 0.5 mm in the plane of the square (the fit of Table 1 moves only the board). A fit that brings the face "
        f"centre within 10 % of the measured {cen['v3id1']['measured']:.1f} and {cen['v3id2']['measured']:.1f} µT/N, "
        f"with the other regions still within {within} %, places the printed magnets off the design's square. A fit "
        "that cannot puts the error in the FE's magnet motions near the centre.",
        "Siraya 85A's modulus: press a solid 85A coupon under a dial indicator at 0.5–5 N "
        "(" + P.code("scripts/tpu_tip_calibrate.py fit --indent") + "). A modulus outside 9–15 MPa rescales every FE "
        f"compliance by the same factor, as the N52 tips' {min(E_fit):.1f}–{max(E_fit):.1f} MPa would.",
        "Softness at a small contact: clamp the pad's printed twin on a rigid block and press it with the 5 mm sphere "
        "under a dial indicator at 0.5–5 N. Travel near the FE pad's 214 µm at 1 N places the UR5's extra "
        "indentation in the robot's tool. Travel near the registered 0.73 mm confirms that the FE pad is too stiff "
        "under a sphere.",
        "Large presses: compare the FE's approach at 3 and 5 N with a neo-Hookean solve of the same pad. A difference "
        "over 10 % at 5 N bounds the linear model's range.",
    ]
    S.append(("future", "Future work", "<ul>" + "".join(f"<li>{t}</li>" for t in items) + "</ul>"))
    return S, lede, summary, film
