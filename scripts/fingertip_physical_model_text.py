"""Prose of the page on the magnetic fingertip's physical model (scripts/fingertip_physical_model_page.py), kept apart from
the plotting. Every number comes from the page's numbers.json, the UR5 sessions' summaries or the linked pages."""
from __future__ import annotations


def sections(N, F, P, link, L):
    """(sections, lede, summary, film_html). N: numbers.json; F: media paths; P: page helpers; link(path, text); L: the
    linked pages by key."""
    fe = N["fe"]
    p1, p3, sz, sy = fe["press1"], fe["press3"], fe["shear_z"], fe["shear_y"]
    pr = N["face_centre_press_uT_per_N"]
    sh = N["face_centre_shear_uT_per_N"]
    fc = N["field_at_chips_mT_N52"]
    pat = N["patterns_N52_checkerboard"]
    lk = N["likelihood_tip2"]
    rng = lambda v: f"{min(v):.1f}–{max(v):.1f}"   # noqa: E731

    lede = (
        "A press on the printed fingertip moves the five magnets inside its pad by a few micrometres. Five magnetometers "
        "under the pad read the change of the magnets' field. The physical model computes that sequence: finite elements "
        "give each magnet's motion for a load at any of 812 surface points, and a dipole model gives the 15 readings. "
        f"Under a 5 mm sphere at 1 N the face sinks {p1['approach_um']:.0f} µm and the magnets move "
        f"{rng(p1['magnet_travel_um'])} µm. The readings then change by {pr['checker_N52']:.0f} µT with N52 magnets in a "
        f"checkerboard and by {pr['as_built_tip2']:.0f} µT with the weaker magnets in the tips today. Inverted without "
        "training, the model located UR5 presses of 2 N and more within 0.9–2.4 mm and gave 0.78–1.31 of the reference "
        "force. Below 1 N it fails.")
    summary = (
        f"FE of the printed pad plus a dipole field model give the 15 magnetometer readings for a load anywhere on the "
        f"surface: at 1 N under a 5 mm sphere the face sinks {p1['approach_um']:.0f} µm, the magnets move "
        f"{rng(p1['magnet_travel_um'])} µm and the readings change by {pr['as_built_tip2']:.0f} µT (today's magnets) to "
        f"{pr['checker_N52']:.0f} µT (N52 checkerboard). Inverted without training it locates UR5 presses of 2 N and over "
        "within 0.9–2.4 mm and gives 0.78–1.31 of the HEX-E force; below 1 N it fails.")

    film = P.film(0, F["film"], F["film_poster"],
                  "The contact viewer replays five presses that the model made with the board's measured noise. The pad "
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
         "$(8.5,\\,0.5,\\,10.4)$ mm, the centre magnet at $(0,\\,0,\\,10.4)$ and the chips' sensing plane at "
         "$x=-5.37$ mm."),
        ("Pad", "The printed part of the fingertip: the front 13.45 mm of the 11/22 half-ellipsoid tip, cut flat at "
         "$x=-4.95$ mm, where it rests on the board. In concept b of the DIP-link study a cover on the DIP link clamps it "
         "over the board."),
        ("Face, edges, flanks", "Regions of the pad's surface by the direction of the outward normal $\\mathbf n$. The "
         "face looks toward the palm ($n_x>0.9$) and the flanks across the finger; the edges lie between them."),
        ("Approach", "Travel of a rigid indenter into the pad from first touch at a given normal force, in µm."),
        ("Magnet motion", "The rigid translation of a magnet, in µm, and its small rotation, in mrad, fitted to the "
         "finite-element nodes inside it. Per newton of load it is the magnet's travel in µm/N."),
        ("Reading, reading change", "The 15 field components that the five chips report, three per chip, in µT. The "
         "reading change $\\Delta\\mathbf b$ is the reading minus the reading at rest."),
        ("Pole pattern", "Which pole of each magnet faces the palm. All north: every magnet's north pole toward the palm. "
         "Checkerboard: the centre magnet and one diagonal pair one way, the other diagonal pair the other way."),
        ("Strength $k$", "A magnet's moment as a fraction of a 3 × 2 mm N52 cylinder's (remanence 1.45 T)."),
        ("Sensitivity $S(\\mathbf p)$", "The reading change per newton of a load at surface point $\\mathbf p$, for loads "
         "along $x$, $y$ and $z$: a 15 × 3 matrix, in µT/N."),
        ("Rest calibration", "The fit of the field model to 2 s of readings with the pad untouched. It finds which chip "
         "address sits at which place, every magnet's pole, the strength $k$ and the chips' offset from the pad."),
        ("Untrained model", "The estimator of this page on a pad's rest calibration, fitted to no press."),
        ("HEX-E, hold", "The OptoForce HEX-E six-axis force-torque sensor between the UR5's flange and the fingertip's "
         "mount, the force reference of the calibration sessions. A hold is one force level of a UR5 press, kept for "
         "about 1.5 s."),
        ("Location error", "Distance in mm from an estimated contact point to the registered one, along the surface; a "
         "median over holds."),
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
        "The print pauses twice so that five 3 × 2 mm magnets can be pressed into pockets. Their tops lie 7.5 mm under the "
        "apex: a centre magnet and four corners 5.0 mm apart across the finger and 9.8 mm along it (Figure 1). Tips 1 "
        "and 2, the checkerboard tips of the UR5 sessions, have the pattern of Figure 1a. Tip 0, the first UR5 tip, has "
        "all five one way.</p>"
        "<p>The ReSkin 5X board lies under the pad's cut face. Its five MLX90393 magnetometer chips sit in a plus, 7.0 mm "
        "apart. U1 lies under the centre magnet, U2 toward the tip end, U4 toward the mount, and U3 and U5 to either "
        "side. Each "
        "chip senses the field 0.42 mm under its package top, 5.37 mm under the centre of the magnets.</p>"
        + P.fig(1, F["anatomy"], "The fingertip and its sensor. (a) Seen from the palm: the pad's outline, the five chips "
                "under the pad (dashed) and the five magnets. Each magnet's colour is the pole that faces the palm, red "
                "north and blue south, in the checkerboard of tips 1 and 2. (b) Section along the finger through the centre magnet, drawn from the "
                "slicer's material classes at 0.05 mm that the finite elements use. The core's lines are the gyroid's "
                "zero set at the 30 % setting's period of 3.5 mm. The sphere is the UR5 calibration's indenter at first "
                "touch.")
    )
    S.append(("device", "The printed pad, its magnets and the magnetometer board", body))

    # ------------------------------------------------------------------------------------------------ the chain
    body = (
        "<p>The model predicts the readings for a load of known place and size. It is a chain of four linear steps "
        "(Figure 2). The print recipe fixes the material in every voxel. The finite elements turn a load at a surface "
        "point $\\mathbf p$ into a rigid motion of each magnet. The field model turns those motions into the change of "
        "the 15 readings. A pad's rest calibration supplies the field model's unknowns: the poles, the strength and the "
        "chips' places. The whole chain is one matrix per surface point, the sensitivity:</p>"
        "$$\\Delta\\mathbf b \\;=\\; \\underbrace{J}_{15\\times30}\\;\\underbrace{M(\\mathbf p)}_{30\\times3}\\;\\mathbf f "
        "\\;=\\; S(\\mathbf p)\\,\\mathbf f .$$"
        "<p>Here $\\mathbf f$ is the force on the pad in newtons. $M(\\mathbf p)$ contains the translation and rotation of "
        "the five magnets per newton of load along $x$, $y$ and $z$, from the finite elements. $J$ contains the change of "
        "each reading per micrometre and per milliradian of each magnet's motion, from the field model. Both are linear "
        "for the micrometre motions of a press. The estimator inverts the chain at every frame of the board: it finds "
        "the point and the force whose predicted change best explains the measured one.</p>"
        + P.fig(2, F["chain"], "The physical model as a chain. The top row is computed once per pad design and printed "
                "recipe. The bottom row runs on the board's stream: the rest calibration fixes the field model's unknowns, "
                "and the estimator compares each reading change with $S(\\mathbf p)\\,\\mathbf f$ at every candidate "
                "point.")
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
        "<p><b>What a load does.</b> Figure 4 shows the pad under three loads of 1 N at the apex. Under the 5 mm sphere "
        f"the face sinks {p1['approach_um']:.0f} µm, and {p3['approach_um']:.0f} µm at 3 N. Most of that travel is local: "
        f"the magnets, 7.5 mm down, move {rng(p1['magnet_travel_um'])} µm, almost all of it toward the board. A shear "
        f"along the finger moves them {rng(sz['magnet_travel_um'])} µm and tilts the magnet set about $y$: the tip-end "
        "magnets sink, the mount-end ones rise. A shear across the finger moves them "
        f"{rng(sy['magnet_travel_um'])} µm sideways, three to four times their travel under a press: the narrow "
        "half-ellipsoid bends sideways on its fixed base.</p>"
        + P.fig(4, F["fe"], "The finite elements under 1 N at the apex. Left: displacement in the section through the "
                "centre magnet (outlined in blue), with the model's cells in white. The black line is the face deformed, "
                "with the largest displacement drawn as 1.2 mm; the grey line is the face at rest. The blue arrow is the "
                "centre magnet's motion, 300 times magnified. Right: all five magnets seen from the palm, with the load as a "
                "star. The arrow is each magnet's motion in the face's plane at 0.25 mm per µm. The number is its motion "
                "along $x$ in µm, negative toward the board.")
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
        "$$\\mathbf B(\\mathbf x)=\\frac{\\mu_0}{4\\pi}\\sum_{j=1}^{5}\\sum_{q=1}^{72} w_q\\,"
        "\\frac{3\\,\\hat{\\mathbf r}\\,(\\hat{\\mathbf r}\\cdot\\mathbf m_j)-\\mathbf m_j}{r^{3}},\\qquad "
        "\\mathbf m_j=s_j\\,k\\,\\frac{B_r V}{\\mu_0}\\,\\mathbf e_x .$$"
        "<p>Here $s_j=\\pm1$ is the magnet's pole, $k$ its strength, $B_r=1.45$ T the N52 remanence and $V$ the magnet's "
        "volume. A magnet that moves by $\\mathbf t_j$ and turns by $\\boldsymbol\\omega_j$ shifts its dipoles and turns "
        "its moment. Chip $c$ at $\\mathbf x_c$ reports the field in its own axes, $R_c$, together with Earth's field "
        "$\\mathbf g$:</p>"
        "$$\\mathbf b_c=R_c\\bigl(\\mathbf B(\\mathbf x_c)+\\mathbf g\\bigr),\\qquad "
        "J=\\frac{\\partial\\mathbf b}{\\partial(\\mathbf t_1,\\boldsymbol\\omega_1,\\dots,\\mathbf t_5,\\boldsymbol\\omega_5)} .$$"
        "<p>The field falls as the cube of distance, so the nearest magnet dominates each chip. With N52 magnets all "
        f"north the centre chip U1 sits in {fc['all_N']['U1']:.1f} mT and the side chips in about "
        f"{fc['all_N']['U3']:.0f} mT (Figure 5c). A press changes these fields by tens of µT, a few thousandths of "
        "them.</p>"
        "<p>The pole pattern shapes the field. With all magnets one way their fields add, and far from the fingertip "
        "the pad looks like one strong dipole of five magnets' moment. The checkerboard leaves a net moment of one magnet. "
        "Its fields cancel at a distance, and the field between neighbouring magnets changes sign (Figure 5d). Two "
        "consequences were computed on " + link(L["est"], "the estimation page") + ". A second fingertip 26 mm away side "
        "by side puts 151 µT on the board with checkerboards, against 658 µT with all-north magnets. A 1 N press on the "
        f"face changes the readings by {pr['checker_N52']:.0f} µT with the checkerboard, against "
        f"{pr['all_N_N52']:.0f} µT with all north, because the field's gradient is steeper between opposite poles.</p>"
        + P.fig(5, F["field"], "Field of five N52 magnets. (a, b) Field lines and field magnitude in the section $y=0$, "
                "which cuts only the centre magnet; darker grey is stronger. The tan lines are the pad's face and cut "
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
        "$x$ and $y$, 1.94 µT on $z$, and a range of 38.4 mT. The range decides the gain. The nearest chip sits in up to "
        "27 mT with N52 magnets, and the most sensitive gain covers only 4.8 mT.</p>"
        "<p>Each chip converts every 7.0–7.1 ms (141 Hz), and the board streams all five over I²C to a QT Py ESP32-S3. "
        "At rest the noise of one conversion is 1.5–1.8 µT on $x$ and $y$ and 2.3–2.6 µT on $z$. Three disturbances are "
        "larger than the noise:</p>"
        "<ul><li>Temperature. The chips run without their on-chip compensation, and each chip's gain falls by "
        "0.65–0.70 % per °C. In a field of 7.6 mT one degree changes the reading by 53 µT, the change of a 2.5 N press "
        "with today's magnets. The estimator divides each reading by $1+\\kappa_c(T_c-25\\,^{\\circ}\\mathrm C)$, with $T_c$ "
        "the chip's die temperature (" + link(L["est"], "estimation page") + ", Section 9).</li>"
        "<li>Earth's field, about 44 µT, turns with the finger. It is the same at all five chips, so it is estimated with "
        "the contact and removed.</li>"
        "<li>Nearby magnets and steel. Another fingertip's magnets or a steel tool add a field that varies slowly across "
        "the 14 mm board. A test against such fields rejects most steel touches "
        "(" + link(L["steel"], "page on fields from outside the pad") + ").</li></ul>"
    )
    S.append(("chips", "The magnetometers and their disturbances", body))

    # ------------------------------------------------------------------------------------------------ patterns
    a, b_, c_, d_ = (pat[k] for k in ("(a) Press at the face centre", "(b) Press 5 mm toward the tip end",
                                     "(c) Press on the edge, −y side", "(d) Press on the +y flank"))
    body = (
        "<p>A reading change is a vector of 15 numbers. Its length grows in proportion to the force, and its direction "
        "depends on where the load acts and in which direction. Figure 6 draws the change at each chip for six loads of "
        "1 N. A press at the face centre moves the centre magnet toward U1 and strengthens the field normal to the board "
        "there. A press 5 mm toward the tip end gives a similar pattern with a larger share at U2: the ratio of U2's "
        f"change to U1's rises from {a['per_chip_uT'][1] / a['per_chip_uT'][0]:.2f} to "
        f"{b_['per_chip_uT'][1] / b_['per_chip_uT'][0]:.2f}. A press on an edge or a flank pushes the pad sideways, the "
        "direction in which its magnets move most (Figure 4c). It changes U2 and U4 by up to "
        f"{max(d_['per_chip_uT'][1], d_['per_chip_uT'][3]):.0f} µT with opposite signs, about twice a face press. A "
        "shear along the finger tilts the magnet set (Figure 4b) and changes U2 most. A shear across the finger shifts "
        f"all five magnets sideways and changes the readings by {pat['(f) Shear across the finger']['norm_uT']:.0f} µT, "
        "more than a press.</p>"
        "<p>The estimator separates the contact point from the force by the direction of $\\Delta\\mathbf b$ and the force's "
        "size by its length. The unknowns are three force components, two coordinates on the surface, a uniform field "
        "change of three components and a common change of the magnets' strength: nine numbers against 15 readings. "
        "Two points are told apart when their patterns differ by more than the noise. Neighbouring points differ little, "
        "so the noise sets how finely a point can be placed. The weaker a press, the larger the region whose patterns "
        "fit its reading.</p>"
        + P.fig(6, F["patterns"], "Reading change at the five chips for 1 N loads, N52 magnets in the checkerboard, seen "
                "from the palm. Each chip's change is drawn in the tip frame. The circle's area is the change normal to the "
                "board, red toward the palm and blue toward the board. The arrow is the change in the board's plane. "
                "The star marks the load, and the orange arrow the shear's direction. The number is the length of the "
                "15-component change. Keys in (a): an arrow and a circle of 50 µT.")
        + "<p>The magnets in the tips today are weaker than N52. Their rest calibrations give strengths of 0.34 on tip 0 "
        "and 0.47–0.57 on tips 1 and 2. On the checkerboard tips they also put the chips 1.1–1.9 mm farther from the "
        "magnets than the design. A press at the face centre of tip 2 changes the readings by "
        f"{pr['as_built_tip2']:.0f} µT per newton, a quarter of the N52 checkerboard's {pr['checker_N52']:.0f} µT. "
        "N52 magnets are on order.</p>"
    )
    S.append(("patterns", "How a reading encodes the contact point and the force", body))

    # ------------------------------------------------------------------------------------------------ estimator
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
        "<p>Figure 7 shows $\\ell(\\mathbf p)$ over the pad for one simulated press at the face centre of tip 2, with "
        "the noise of the viewer's 50 ms window. At 5 N the marginal likelihood has one sharp peak, "
        f"{lk['5N']['error_mm']:.1f} mm from the true point. At 2 N the peak is broad and the estimate is "
        f"{lk['2N']['error_mm']:.1f} mm off. At 0.5 N no point near the truth explains the reading with a pressing force, "
        "and the estimate goes to the edge of the pad.</p>"
        + P.fig(7, F["likelihood"], "Log marginal likelihood of a contact at each surface point, for one simulated press at "
                "the face centre (star) of tip 2. The model has tip 2's magnets as built and the viewer's noise: a 50 ms "
                "window and a 2 µT floor. Grey points cannot explain the reading with a force that presses inside the friction cone. "
                "The cross is the estimate.")
        + "<p><b>Rest calibration.</b> Before any press, the estimator needs the field model of the pad as built. With "
        "the pad untouched the 15 readings are the magnets' field and Earth's. The fit searches the 16 pole patterns, "
        "the 24 assignments of chip addresses to places and the chips' common turn. For each it fits the strength, the "
        "offset and Earth's field by least squares, and the best reproduces the reading at rest. On tip 1 the planned checkerboard fits at 0.24 mT "
        "RMS and the next pattern at 0.87 mT. The fit finds the strength and the magnets' depth only together: 0.57 of "
        "N52 with the chips 1.9 mm farther away, or 0.34 at 0.7 mm, fit 0.09 mT apart.</p>"
        "<p>On simulated presses with N52 magnets all north, 2 % of presses are placed more than 1.5 mm off. The others "
        "have median errors of 0.18 mm and 0.03 N, and the errors follow the model's 1-sigma within 10–16 % "
        "(" + link(L["est"], "estimation page") + ", Section 6). These figures assume a perfect model.</p>"
    )
    S.append(("estimator", "Inverting the model for the contact point and the force", body))

    # ------------------------------------------------------------------------------------------------ UR5
    u = N["ur5"]
    body = (
        "<p>On the UR5 the pad was pressed onto a 5 mm sphere at registered points with the HEX-E as reference "
        "(" + link(L["cal"], "calibration page") + "). Four sessions on three tips test the untrained model "
        "(Figure 8). From 2 N up it places contacts within 0.9–2.4 mm (medians by force level). Its normal force is "
        "0.78–1.31 of the HEX-E's by region. At 0.25 and 0.5 N it places contacts a median 8.7–11.7 mm away. On tip 0 "
        "each press left the reading at rest about 7 µT off, the change of a 0.3 N press. Below 1 N that step is as "
        "large as the press itself.</p>"
        + P.fig(8, F["ur5"], "The untrained model on the UR5 sessions. (a) Median location error by force level. (b) The "
                "model's normal force over the HEX-E's, by region. Tip 0 (session sphere2) has its magnets all one way; tips "
                "1 and 2 have the checkerboard. Tip 1 ran twice, on 2026-10-23 and 2026-10-25.")
        + "<p>The registered indentation is larger than the FE pad's at the measured force. At the registered depth the "
        "FE pad would push back with 5.1 times the measured force. A pad that much softer would move its magnets five "
        "times as far per newton, and the board would read five times the HEX-E's force. It reads 0.78–1.31 of it, so "
        "the pad's bulk follows the FE model within about 30 %. The extra indentation, 0.33 mm plus 0.20 mm per "
        "newton under the sphere, lies where the magnets do not move. Under a rod of radius 12.5 mm it is 0.07 mm plus "
        "0.10 mm per newton. A spring in series with the pad, in the mount, the arm or the indenter's stand, would add "
        "the same slope under both indenters. So at most 0.10 mm/N lies in series. At least 0.10 mm/N arises at the "
        "sphere's small contact, where the print is softer than the FE pad ("
        + link(L["gt"], "UR5 ground truth") + ", Sections 3, 4 and 8).</p>"
        "<p>At 2 N and over, most of the untrained model's location error is geometric. On tip 1 an affine map of the "
        "estimated points takes the error from 1.16 to 0.60 mm, with a scale of 0.68 and a turn of 14.7°. The magnets' "
        "places and depth, which the rest calibration fixes only jointly with their strength, are the likely source. "
        "Trained on 47 presses of the same tip, a Gaussian process places held-out contacts within 0.71 mm and an "
        "ensemble of small neural networks within 0.42 mm. The learned models, their training data and their transfer "
        "between tips are on " + link(L["cal"], "the calibration page") + ".</p>"
    )
    S.append(("ur5", "The model against the UR5 presses", body))

    # ------------------------------------------------------------------------------------------------ limitations
    body = (
        "<p>The FE gives Siraya 85A the 12 MPa of a NinjaFlex data sheet. It is linear and frictionless at the indenter. "
        "It fixes the pad's cut face, so a pad that shifts in its seat is outside the model; the owner's first presses "
        "by hand shifted it by 17–23 µm. The contact patch under a small sphere is resolved by about 15 face nodes at "
        "1 N. The field model assumes the design's magnet places; the rest fit trades their strength against their "
        "depth. The temperature correction uses each chip's die temperature, and the magnets' own coefficient, about "
        "0.12 % per °C, follows the pad's lagging temperature. The estimator assumes one contact.</p>"
    )
    S.append(("limits", "Limitations", body))

    # ------------------------------------------------------------------------------------------------ future work
    items = [
        "N52 magnets: recalibrate at rest in the viewer (key c). A strength of 0.8–1.2, the poles as inserted and a rest "
        "residual under 0.3 mT confirm the field model. A strength near 0.5 with a large offset repeats the depth trade "
        "of today's tips.",
        "Siraya 85A's modulus: press a solid 85A coupon under a dial indicator at 0.5–5 N "
        "(" + P.code("scripts/tpu_tip_calibrate.py fit --indent") + "). A modulus outside 9–15 MPa rescales every FE "
        "compliance by the same factor.",
        "Softness at a small contact: clamp the pad's printed twin on a rigid block and press it with the 5 mm sphere "
        "under a dial indicator at 0.5–5 N. Travel near the FE pad's 214 µm at 1 N places the UR5's extra "
        "indentation in the robot's tool. Travel near the registered 0.73 mm confirms that the FE pad is too stiff "
        "under a sphere.",
        "Magnet places: fit each magnet's place and the pad's pose in its seat to tip 2's presses at 3 N and over, "
        "without the priors on the offsets. A mean location error under 0.5 mm, from 1.4 mm, puts the geometric error "
        "in the magnets; an error that stays puts it in the registration.",
        "Large presses: compare the FE's approach at 3 and 5 N with a neo-Hookean solve of the same pad. A difference "
        "over 10 % at 5 N bounds the linear model's range.",
    ]
    S.append(("future", "Future work", "<ul>" + "".join(f"<li>{t}</li>" for t in items) + "</ul>"))
    return S, lede, summary, film
