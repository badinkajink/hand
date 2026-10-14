"""Text of the topic overview docs/overviews/printed_fingertips.html (builder: scripts/topic_overview_page.py).

Reader: a collaborator new to the project. Takeaways: (1) how stiff the printed tips are, and how far the sphere pads
are from them; (2) what eFlesh lattices cost in printability and stiffness; (3) what force resolution the embedded
magnets give on the five-chip board, and which disturbances dominate. Budget: 900 words of prose. Sources: the pages in
PAGES (owner, 2026-10-13: the material analysis gets its own topic). The finite-element films of the whole tip
(2026-10-14, scripts/tpu_tip_fem_film.py --whole / --printed) are placed once their files exist.
"""
import os

KEY = "fingertip"
SHORT_TITLE = "Printed fingertips and force sensing"
TITLE = "Stiffness, printability and magnetic force sensing of the printed TPU fingertips"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILM_G20 = "docs/experiments/20261009-tpu_tip_fem/media/20261014-film_whole_g20x.mp4"
FILM_EF = "docs/experiments/20261009-tpu_tip_fem/media/20261014-film_whole_eflesh_open.mp4"
FILMS = all(os.path.exists(os.path.join(_ROOT, p)) for p in (FILM_G20, FILM_EF))

LEDE = (
    "The SR&#178; fingertip is a 17 by 14.8 by 22&#8202;mm block of 95A TPU printed with two walls and gyroid infill. "
    "Finite elements on the tip&#8217;s own G-code give an approach of 63&#8202;&#181;m under 1&#8202;N at 20&#8202;% infill; "
    "eFlesh lattices of 8&#8202;mm cells sink 381&#8211;613&#8202;&#181;m, and 1&#8202;mm sphere pads at one modulus miss the "
    "modelled approach by 20&#8211;28&#8202;% over 0.5&#8211;3&#8202;N. A 4.76 by 1.59&#8202;mm magnet 5&#8202;mm under the face, "
    "read by ReSkin&#8217;s five-chip MLX90393 board, resolves 0.044&#8202;N per sample at a known contact point; with the "
    "contact point unknown, four magnets hold 0.14&#8211;0.16&#8202;N over the face. No tip or board has been measured yet.")

TERMS = [
    ("Approach", "Travel of the 12.5&#8202;mm screwdriver shaft into the tip from first touch under a given normal force, "
     "in &#181;m."),
    ("Contact modulus", "\\(E^*\\), the modulus of the pads&#8217; pressure law that best reproduces the finite-element "
     "approach at 0.5, 1 and 3&#8202;N, in MPa; the RL scenes use 10&#8202;MPa."),
    ("Twist arm", "Torsional friction torque over the friction coefficient times the normal force, in mm: the radius at "
     "which the contact patch resists spin."),
    ("Gyroid infill", "PrusaSlicer&#8217;s triply periodic infill; the percentage is the volume fraction of the core."),
    ("eFlesh", "An open-source printed sensor: cells of a microstructure pattern cut to the part&#8217;s shape, with magnets "
     "in printed pouches over a magnetometer board."),
    ("Five-chip board", "ReSkin&#8217;s 5X board: five Melexis MLX90393 three-axis magnetometers, U1 under the contact "
     "point and U2&#8211;U5 7&#8202;mm from it in a plus."),
    ("Force noise", "Standard deviation of the normal force estimated from one 141&#8202;Hz sample of the five chips, "
     "computed from finite-element magnet travel, a dipole field and the datasheet&#8217;s chip noise, in N."),
    ("GAIN_SEL", "The MLX90393&#8217;s gain register; at 7, the most sensitive setting, its guaranteed range is "
     "4.8&#8202;mT."),
]

_FILM_G20 = "\n{{FIG film_g20}}" if FILMS else ""
_FILM_EF = "\n{{FIG film_ef}}" if FILMS else ""

SECTIONS = [
    ("gyroid", "Stiffness of the printed gyroid tip",
     """<p>The finite-element models resolve the tip&#8217;s two walls, its four top and bottom layers and the gyroid beads as
voxels of 0.1&#8202;mm, pressed by the 12.5&#8202;mm shaft. On those models the pads&#8217; contact modulus is 13.0&#8202;MPa at
20&#8202;% infill, 6.6&#8202;MPa at 10&#8202;% and 3.4&#8202;MPa at 5&#8202;%. One modulus misses the approach by
20&#8211;28&#8202;% over 0.5&#8211;3&#8202;N, because the tip&#8217;s force grows as the approach to the power 1.07 and the
pads&#8217; to the power 1.5; the walls carry the edges of the face, and the pads&#8217; twist arm falls 7&#8211;23&#8202;% short
of the finite elements&#8217; at 3&#8202;N, depending on the print
([finite elements of the printed tip](exp:20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html)).</p>
<p>Solved from its own G-code, the 20&#8202;% tip with its magnet is softer than these models: 63&#8202;&#181;m at 1&#8202;N
against 41&#8202;&#181;m, a contact modulus of 7.3&#8202;MPa against 14.6&#8202;MPa, because the stacked beads overlap only
partly ([eFlesh printability](exp:20261013-eflesh_printability/20261013-eflesh_lattice_printability.html)). A magnet
5&#8202;mm under the face stiffens the tip by 15&#8202;%, one 3&#8202;mm under it by 31&#8202;% and one 1.5&#8202;mm under it by
99&#8202;%.</p>""" + _FILM_G20),
    ("eflesh", "Printability and stiffness of eFlesh lattices",
     """<p>eFlesh&#8217;s cut-cell tool crashes in OpenVDB on every input, and its notebook&#8217;s fallback tiles whole cells,
which gives only boxes. <code>scripts/eflesh_build.py</code> builds a lattice of any shape, with the pouches and the pause,
in about 9&#8202;s by stitching cells and cutting them with mesh booleans. In a 24&#8202;mm cube at the modulus parameter of
the owner&#8217;s print, 3, 28 and 71&#8202;% of the islands of a layer are less than half supported at 8, 6 and 4&#8202;mm
cells on a 0.4&#8202;mm nozzle, where the owner&#8217;s prints delaminated; the gyroid tip prints each layer as one island
({{FIGREF support}}).</p>
<p>From their G-code, an eFlesh tip of 8&#8202;mm cells sinks 613&#8202;&#181;m at 1&#8202;N when open on every face, as
eFlesh prints it, 381&#8202;&#181;m with a 0.9&#8202;mm palmar skin and 81&#8202;&#181;m with a skin on every face
({{FIGREF sections}}). At 3&#8202;N the open lattice sinks about 1.5&#8202;mm, deeper than 1&#8202;mm sphere pads
represent. The skin closes the open cells that catch on a tool and costs sensitivity: the magnet travels 250&#8202;&#181;m/N
in the open tip and 26&#8202;&#181;m/N with a skin on every face
([eFlesh printability](exp:20261013-eflesh_printability/20261013-eflesh_lattice_printability.html)).</p>
{{FIG support}}
{{FIG sections}}""" + _FILM_EF),
    ("readout", "Embedded magnets and the five-chip readout",
     """<p>With the chips against the tip&#8217;s back face and the Earth&#8217;s field fitted as an unknown, one magnet
5&#8202;mm under the face gives 0.044&#8202;N of force noise per sample on the gyroid tip (0.060&#8202;N from U1 alone) and
0.005&#8202;N on the open eFlesh tip. The owner&#8217;s v17 mount sets the board under 2.5&#8202;mm of solid TPU and doubles the
noise to 0.090&#8202;N (Figures&#160;{{FIGN board}} and&#160;{{FIGN noise}}). The disturbances are larger than the noise: a
10&#8202;&#176;C change drifts the gyroid tip&#8217;s estimate by 0.47&#8202;N, and the facing finger&#8217;s magnet moving from
25 to 30&#8202;mm shifts it by 5.2&#8202;N, which a fitted field gradient reduces to 2.0&#8202;N at 0.079&#8202;N of noise; the
hand&#8217;s kinematics give that magnet&#8217;s position, so its field can be subtracted
([magnetometer readout](exp:20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html)).</p>
<p>With the contact point unknown, one magnet gives 0.08&#8211;1.48&#8202;N over 15 contact points on the face. Four magnets of
the same size, 6&#8202;mm apart across the face and 7&#8202;mm along it, give 0.14&#8211;0.16&#8202;N everywhere, stiffen the face
by 12&#8202;% and raise the field at the chips to 10.6&#8202;mT, which needs GAIN_SEL&#8202;3
([magnetometer readout](exp:20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html#layouts)).</p>
{{FIG board}}
{{FIG noise}}"""),
    ("print", "Print plan and magnet pockets",
     """<p>The printed tip is whole, without the insert&#8217;s mounting slots; the owner designs the attachment separately.
Each pocket is 90&#8202;% of the magnet&#8217;s diameter, for a press fit, and one layer deeper than the magnet, so the
nozzle never prints onto it. At 0.2&#8202;mm layers the 4.76&#8202;mm magnet 5&#8202;mm under the face sits in nine layers from
10.40 to 12.20&#8202;mm of print height under 0.21&#8202;mm of air, and the print pauses before layer 62. The tip page holds
the STL and pause sheet of each print and a calculator for magnet depth, chip distance, TPU modulus, magnet grade and
output rate
([finite elements of the printed tip](exp:20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html)).</p>"""),
    ("budget", "Compliance of the tip and of the servo",
     """<p>At 1&#8202;N the gyroid tip sinks 63&#8202;&#181;m and the eFlesh tips 0.38&#8211;0.61&#8202;mm. The servo plant used in
training (gain 0.72&#8202;N&#183;m/rad, inferred from the load register and never measured) lets the fingertip yield
3.4&#8211;4.7&#8202;mm per newton toward the tool, so on that plant the servos set the finger&#8217;s compliance and the tip
adds less than a fifth of it. The servo identification protocol measures the gain on a pendulum and with finger presses
on a load cell before the tip&#8217;s deformation is modelled further
([servo identification protocol](exp:20261013-servo_identification/20261013-servo_identification_protocol.html)).</p>"""),
]

FIGURES = {
    "support": dict(page="20261013-eflesh_printability/20261013-eflesh_lattice_printability.html", n=2, source_n=4,
                    source="eFlesh printability", alt="Islands less than half supported against cell size for two nozzles.",
                    caption="Share of the islands of a layer less than half supported by the layer below, against eFlesh&#8217;s "
                            "cell size, in a 24&#8202;mm cube for a 0.4&#8202;mm and a 0.25&#8202;mm nozzle."),
    "sections": dict(image="docs/experiments/20261013-eflesh_printability/media/20261013-tip_sections.png", width=760,
                     alt="Sections of the printed tips through the contact point.",
                     caption="Sections of the printed tips through the contact point, palmar face up, as the solver sees them "
                             "from their G-code: TPU black, the bonded magnet red (<a href='../experiments/20261013-eflesh_printability/"
                             "20261013-eflesh_lattice_printability.html'>eFlesh printability</a>, Figure&#160;5)."),
    "board": dict(page="20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html", n=2, source="magnetometer readout",
                  alt="The five-chip board under the SR2 tip, from the back face and in section.",
                  caption="The five-chip board under the SR2 tip: the chips U1&#8211;U5 under the back face, and the sensing gap "
                          "with the chips against the face and on the v17 mount."),
    "noise": dict(page="20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html", n=3, source="magnetometer readout",
                  alt="Normal-force noise and three disturbances for four tips on the five-chip board.",
                  caption="Normal-force noise of one 141&#8202;Hz sample and three force-equivalent disturbances (Earth&#8217;s "
                          "field, 10&#8202;&#176;C of drift, the facing magnet) for four tips on the five-chip board."),
    "film_g20": dict(film=[[(FILM_G20, "95A TPU, gyroid 20 %, whole tip")]], page="20261009-tpu_tip_fem/"
                     "20261009-printed_tpu_fingertip_finite_elements.html", source="finite elements of the printed tip",
                     caption="The whole tip with 20&#8202;% gyroid infill under the 12.5&#8202;mm shaft: the section through the "
                             "contact point, displacement magnified 15 times and coloured by its size, beside the pressure on "
                             "the palmar face, as the approach grows."),
    "film_ef": dict(film=[[(FILM_EF, "eFlesh lattice, 8 mm cells, open, from its G-code")]],
                    page="20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html",
                    source="finite elements of the printed tip",
                    caption="The open eFlesh tip of 8&#8202;mm cells under the same shaft, solved from its G-code, displacement "
                            "magnified twice; the magnet is outlined in red."),
}

FUTURE = [
    "Press a printed 95A 20&#8202;% tip with the 12.5&#8202;mm shaft on a load cell at 0.5, 1 and 3&#8202;N and fit it with "
    "<code>scripts/tpu_tip_calibrate.py</code>; an approach outside 50&#8211;76&#8202;&#181;m at 1&#8202;N contradicts the "
    "as-printed finite elements (63&#8202;&#181;m).",
    "Log the five chips at GAIN_SEL&#8202;7 for 10&#8202;min with the tip unloaded, then over a 10&#8202;&#176;C change; a "
    "per-sample force noise above 0.066&#8202;N (1.5 times 0.044) or a drift above 0.7&#8202;N per 10&#8202;&#176;C contradicts "
    "<code>scripts/mlx90393_model.py</code>.",
    "Fit sphere pads with a lumped normal spring to the eFlesh tip&#8217;s approach and twist arm in "
    "<code>docs/experiments/20261013-eflesh_printability/fem.jsonl</code>; if no spring brings both within 15&#8202;% over "
    "0.5&#8211;3&#8202;N, eFlesh needs a distributed model.",
    "Rerun the magnet-depth finite elements with the grid refined to 0.3&#8202;mm from &#8722;2 to 7&#8202;mm in depth "
    "(<code>scripts/tpu_tip_magnet_layouts.py</code>), to test the noise trend of 0.137, 0.121 and 0.098&#8202;N at 1.5, 3 "
    "and 5&#8202;mm.",
    "Print the four-magnet tip <code>print/20261013-sr2_tip_four_d4p76x1p59_depth5.stl</code> and check that the recessed "
    "pockets hold the magnets through the rest of the print without glue.",
]

PAGES = [
    ("Printed fingertips and force sensing", "20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html",
     "Finite elements, the pads&#8217; modulus, magnet depth, print plan, calculator"),
    ("Printed fingertips and force sensing", "20261013-eflesh_printability/20261013-eflesh_lattice_printability.html",
     "Audit and build of eFlesh lattices, as-printed stiffness"),
    ("Printed fingertips and force sensing", "20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html",
     "Five-chip board, force noise, one or four magnets"),
    ("Related pages", "20261013-servo_identification/20261013-servo_identification_protocol.html",
     "The servo gain that sets the finger&#8217;s compliance"),
    ("Related pages", "20261012-sprung_skin/20261012-sprung_skin_presliding.html", "Presliding compliance in simulation"),
]
