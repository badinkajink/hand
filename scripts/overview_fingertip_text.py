"""Text of the topic overview docs/overviews/printed_fingertips.html (builder: scripts/topic_overview_page.py).

Reader: a collaborator, or the owner preparing a meeting, who needs the method, the reason for each choice and the open
questions of the printed fingertip, without reading the fifteen result pages (owner, 2026-10-20: "understand what tools,
what's the underlying math behind: FEA over the gyroid infill, the impacts our various modifications and magnet
embedding does, how we're able to map the magnetometer readings to the surface via FEA"; "go into my meetings tomorrow
and have a clear understanding of what i need to communicate ... the methodology and rationale, and the remaining
questions"); owner, 2026-10-23: "make sure our main overview doc for the printed tip + sensing is up to date and ready
to read". Takeaways: (1) one chain of models runs from the print recipe to the chip readings and back to a contact
estimate, and on the UR5 against the HEX-E the board's force agrees within 0.78-1.05 while the untrained estimate is
1.6-2.6 mm off from 1 N up (a GP trained on 59 presses: 0.84 mm); (2) what each design choice does to stiffness and force
resolution, and why concept b's pad with five magnets; (3) what is still open: the print's softness under a small
indenter (rod presses put at most 0.10 of sphere2's 0.20 mm/N in series), presses below 1 N, the force reference, four
magnets or five, the transfer of learned corrections. Owner, 2026-10-28: "i think we shouldve updated at least the 10/27
results and preliminary v3 id1 4 n52 magnet calibration" (v2 ID 1's hard presses and tall indenter, v3 ID 1 against v2
ID 1, mount v7; step 13 of docs/handoff/20261027-crosstalk_feed_poller.md). Budget: about 3,600 words
of prose. Sources: the pages in PAGES and the fact sheets of 2026-10-20 (FE and printing; readout and hardware), each
number as its page gives it.
"""
import os

KEY = "fingertip"
SHORT_TITLE = "Printed fingertips and force sensing"
TITLE = "Finite elements, magnetic readout and contact estimation of the printed TPU fingertips"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILM_G20 = "docs/experiments/20261009-tpu_tip_fem/media/20261014-film_whole_g20x.mp4"
FILM_VIEW = "docs/experiments/20261020-fingertip_contact_estimation/media/20261020-viewer_surface_presses.mp4"
FILM_UR5 = "docs/experiments/20261022-ur5_tip_calibration/media/20261022-lab_sphere2_overhead.mp4"
FILM_REAL = "docs/experiments/20261023-second_fingertip_transfer/media/20261023-viewer_centre_presses.mp4"

CHAIN_SVG = """<svg class="chain" viewBox="0 0 828 330" width="828" height="330" role="img" aria-label="The chain of models of the printed fingertip, from the print recipe to the contact estimate" style="display:block;margin:12px auto;max-width:100%;height:auto;font-family:'Times New Roman',Times,serif">
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#000"/></marker></defs>
<rect x="6" y="14" width="150" height="70" fill="#fff" stroke="#000"/>
<text x="81" y="36" text-anchor="middle" font-size="15" font-weight="bold">Shape and recipe</text>
<text x="81" y="54" text-anchor="middle" font-size="13.5">11/22 tip, cut pad, 85A,</text>
<text x="81" y="72" text-anchor="middle" font-size="13.5">2 walls, gyroid, 4 skins</text>
<rect x="176" y="14" width="150" height="70" fill="#fff" stroke="#000"/>
<text x="251" y="36" text-anchor="middle" font-size="15" font-weight="bold">Material per voxel</text>
<text x="251" y="54" text-anchor="middle" font-size="13.5">wall, skin, gyroid core,</text>
<text x="251" y="72" text-anchor="middle" font-size="13.5">magnet (0.05&#8202;mm grid)</text>
<rect x="346" y="14" width="150" height="70" fill="#fff" stroke="#000"/>
<text x="421" y="36" text-anchor="middle" font-size="15" font-weight="bold">Gyroid stiffness</text>
<text x="421" y="54" text-anchor="middle" font-size="13.5">one repeating cell &#8594;</text>
<text x="421" y="72" text-anchor="middle" font-size="13.5">bulk stiffness</text>
<rect x="516" y="14" width="150" height="70" fill="#fff" stroke="#000"/>
<text x="591" y="36" text-anchor="middle" font-size="15" font-weight="bold">Finite elements</text>
<text x="591" y="54" text-anchor="middle" font-size="13.5">small 3D cells,</text>
<text x="591" y="72" text-anchor="middle" font-size="13.5">force &#8594; motion</text>
<rect x="686" y="14" width="136" height="70" fill="#fff" stroke="#000"/>
<text x="754" y="36" text-anchor="middle" font-size="15" font-weight="bold">Magnet motions</text>
<text x="754" y="54" text-anchor="middle" font-size="13.5">per newton at 812</text>
<text x="754" y="72" text-anchor="middle" font-size="13.5">surface points</text>
<rect x="6" y="128" width="150" height="70" fill="#fff" stroke="#000"/>
<text x="81" y="150" text-anchor="middle" font-size="15" font-weight="bold">Reading at rest</text>
<text x="81" y="168" text-anchor="middle" font-size="13.5">15 components,</text>
<text x="81" y="186" text-anchor="middle" font-size="13.5">pad untouched</text>
<rect x="176" y="128" width="150" height="70" fill="#fff" stroke="#000"/>
<text x="251" y="150" text-anchor="middle" font-size="15" font-weight="bold">Calibration</text>
<text x="251" y="168" text-anchor="middle" font-size="13.5">chip positions and axes,</text>
<text x="251" y="186" text-anchor="middle" font-size="13.5">poles, strength, offset</text>
<rect x="346" y="128" width="150" height="70" fill="#fff" stroke="#000"/>
<text x="421" y="150" text-anchor="middle" font-size="15" font-weight="bold">Field model</text>
<text x="421" y="168" text-anchor="middle" font-size="13.5">5 magnets &#215; 72 dipoles</text>
<text x="421" y="186" text-anchor="middle" font-size="13.5">at the 5 chips</text>
<rect x="516" y="128" width="150" height="70" fill="#fff" stroke="#000"/>
<text x="591" y="150" text-anchor="middle" font-size="15" font-weight="bold">Signals per newton</text>
<text x="591" y="168" text-anchor="middle" font-size="13.5">15 reading changes</text>
<text x="591" y="186" text-anchor="middle" font-size="13.5">at each surface point</text>
<rect x="686" y="128" width="136" height="70" fill="#fff" stroke="#000"/>
<text x="754" y="150" text-anchor="middle" font-size="15" font-weight="bold">Design check</text>
<text x="754" y="168" text-anchor="middle" font-size="13.5">force resolution</text>
<text x="754" y="186" text-anchor="middle" font-size="13.5">from signal and noise</text>
<rect x="176" y="244" width="150" height="70" fill="#fff" stroke="#000"/>
<text x="251" y="266" text-anchor="middle" font-size="15" font-weight="bold">Readings, live</text>
<text x="251" y="284" text-anchor="middle" font-size="13.5">15 components,</text>
<text x="251" y="302" text-anchor="middle" font-size="13.5">140&#8202;Hz, 50&#8202;ms window</text>
<rect x="516" y="244" width="150" height="70" fill="#fff" stroke="#000"/>
<text x="591" y="266" text-anchor="middle" font-size="15" font-weight="bold">Estimator</text>
<text x="591" y="284" text-anchor="middle" font-size="13.5">point, force, 1-sigma</text>
<text x="591" y="302" text-anchor="middle" font-size="13.5">best fit on the surface</text>
<rect x="686" y="244" width="136" height="70" fill="#fff" stroke="#000"/>
<text x="754" y="266" text-anchor="middle" font-size="15" font-weight="bold">Browser viewer</text>
<text x="754" y="284" text-anchor="middle" font-size="13.5">deformation, pad</text>
<text x="754" y="302" text-anchor="middle" font-size="13.5">pressure, traces</text>
<line x1="156" y1="49" x2="174" y2="49" stroke="#000" marker-end="url(#ah)"/>
<line x1="326" y1="49" x2="344" y2="49" stroke="#000" marker-end="url(#ah)"/>
<line x1="496" y1="49" x2="514" y2="49" stroke="#000" marker-end="url(#ah)"/>
<line x1="666" y1="49" x2="684" y2="49" stroke="#000" marker-end="url(#ah)"/>
<line x1="156" y1="163" x2="174" y2="163" stroke="#000" marker-end="url(#ah)"/>
<line x1="326" y1="163" x2="344" y2="163" stroke="#000" marker-end="url(#ah)"/>
<line x1="496" y1="163" x2="514" y2="163" stroke="#000" marker-end="url(#ah)"/>
<line x1="666" y1="163" x2="684" y2="163" stroke="#000" marker-end="url(#ah)"/>
<line x1="326" y1="279" x2="514" y2="279" stroke="#000" marker-end="url(#ah)"/>
<line x1="666" y1="279" x2="684" y2="279" stroke="#000" marker-end="url(#ah)"/>
<path d="M754,84 L754,106 L621,106 L621,126" fill="none" stroke="#000" marker-end="url(#ah)"/>
<line x1="591" y1="198" x2="591" y2="242" stroke="#000" marker-end="url(#ah)"/>
<text x="597" y="224" font-size="13">inverted</text>
</svg>"""

LEDE = (
    "The printed fingertip has four or five magnets in a soft thermoplastic polyurethane (TPU) pad and five magnetic sensors "
    "behind it. A press deforms the pad, moves the magnets and changes the sensors&#8217; readings. We model that "
    "sequence to estimate where the tool presses and with how much force. On a UR5 arm, with a six-axis force sensor as "
    "the reference, the untrained model placed presses of 2&#8202;N and more within 0.9&#8211;2.4&#8202;mm. Its force was "
    "0.78&#8211;1.31 of the reference. Below 1.5&#8202;N it placed presses 8&#8211;10&#8202;mm off with the first tips&#8217; weak "
    "magnets, and within 2.0&#8202;mm with five N52 magnets. Trained on a tip&#8217;s own presses, a Gaussian process placed "
    "presses at new points within 0.25&#8211;0.53&#8202;mm at 1.5&#8202;N and over. It gave the force&#8217;s direction within "
    "2.1&#8211;2.5&#176; on the face. A second Gaussian process is given only the direction of the board&#8217;s change. "
    "Trained up to 2&#8202;N, it placed harder presses, up to 8&#8202;N, within 0.25&#8211;0.31&#8202;mm. It gave their force "
    "within 5.1&#8211;6.2&#8202;%. The board&#8217;s change per newton with four N52 magnets is half that with five. The "
    "calibrated location errors are 0.26 and 0.28&#8202;mm. A model learned on one tip misses on another by 1.4&#8211;2.5&#8202;mm, and by 0.64&#8202;mm "
    "once each board change is scaled by its magnets&#8217; strength. A neighbouring fingertip&#8217;s magnets change a board by "
    "up to 4,162&#8202;&#181;T, against 23&#8202;&#181;T for a 0.5&#8202;N press. Under a small sphere the printed pad sinks "
    "further than its finite-element model predicts. Three pages explain this in order: the "
    "<a href='../experiments/20261026-fingertip_physical_model/20261026-fingertip_physical_model.html'>physical model</a> "
    "of the pad and its readout, the "
    "<a href='../experiments/20261026-fingertip_calibration/20261026-fingertip_calibration_on_the_ur5.html'>calibration "
    "on the UR5</a>, and the first part of the <a href='contact_model.html#chain'>contact overview</a>, which compares the "
    "print with the simulations&#8217; sphere pads.")

INTRO_TITLE = "One press through the model"
INTRO = (
    "<p>At rest, the five chips measure the field from the magnets inside the pad. Each chip measures the field in "
    "three directions, giving 15 numbers. A press moves the magnets by a few micrometres and changes those numbers. "
    "The change depends on both the contact point and the force, so one chip alone cannot generally separate them.</p>"
    "<p>Finite-element analysis divides the printed walls, solid skins and gyroid infill into small pieces. It predicts "
    "how a force at each candidate point moves the magnets. A magnetic-field model then predicts the 15 readings. "
    "For a live press, the estimator searches for the point and force whose predicted change fits the measured one. "
    "The <a href='../experiments/20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html'>print model</a>, "
    "<a href='../experiments/20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html'>board readout</a>, "
    "<a href='../experiments/20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html'>estimator</a> "
    "and <a href='../experiments/20261022-ur5_tip_calibration/20261022-ur5_tip_ground_truth_and_predictors.html'>UR5 presses</a> "
    "give the calculations and measurements.</p>")

TERMS = [
    ("Approach", "Travel of a rigid indenter into the tip from first touch under a given normal force, in &#181;m; the "
     "reference indenter is the 12.5&#8202;mm screwdriver shaft across the finger."),
    ("Wall, skin, core", "The three printed structures. Walls are the two 0.45&#8202;mm perimeters of every layer "
     "(0.9&#8202;mm), also around each magnet pocket. Skins are the four solid 0.2&#8202;mm layers under every top and "
     "bottom surface (0.8&#8202;mm). The core is the gyroid infill."),
    ("Infill setting, solid fraction", "The slicer&#8217;s gyroid density setting \\(\\rho\\) (30 or 50&#8202;%) and the "
     "volume fraction of TPU it prints, which differ because successive layers&#8217; beads overlap only in part."),
    ("Homogenized tensor", "The 6&#215;6 elasticity tensor of a periodic cell of printed gyroid, per unit modulus of solid "
     "TPU \\(E_s\\): the stiffness of the core seen at the scale of the tip."),
    ("Contact modulus \\(E^*\\)", "The modulus of the sphere pads&#8217; pressure law that best reproduces the FE approach "
     "at 0.5, 1 and 3&#8202;N, in MPa; the RL scenes use 10&#8202;MPa."),
    ("Magnet travel", "Rigid motion of a magnet per newton of contact force, from the FE, in &#181;m/N."),
    ("Sensing plane", "The plane of the five chips&#8217; Hall elements; on concept b it lies 0.42&#8202;mm below the "
     "pad&#8217;s back face."),
    ("Force resolution", "The Cram&#233;r&#8211;Rao bound on the standard deviation of the pressing force, in N. It is "
     "computed from one conversion of the five chips (141&#8202;Hz) at a known contact point, with Earth&#8217;s field "
     "unknown. It is the spread of the best unbiased estimate when the models are exact."),
    ("Sphere pads", "The simulations&#8217; contact model of the tip: 0.75&#8202;mm spheres on a 1&#8202;mm lattice over its "
     "surface. Each has stiffness \\(K_i=E\\,a_i/h_i\\), with \\(h_i\\) its depth below the tip&#8217;s volume "
     "centroid (builder v2)."),
    ("Rest calibration", "The fit of the field model to the reading with the pad untouched. It gives which chip address sits "
     "at which position, the chips&#8217; axes, every magnet&#8217;s pole, the magnets&#8217; strength and the pad&#8217;s "
     "offset on the board."),
    ("Sensitivity \\(S(\\mathbf p)\\)", "The change of the 15 readings per newton of a patch load at surface point "
     "\\(\\mathbf p\\), along each axis of the tip frame (15&#215;3)."),
    ("Placed away", "A simulated press whose estimated point lies more than 1.5&#8202;mm from the true one."),
    ("HEX-E", "The OptoForce HEX-E six-axis force-torque sensor between the UR5&#8217;s flange and the calibration mount: "
     "the force reference of the UR5 sessions."),
    ("Touch-off, registration", "A slow approach of the pad onto an indenter until the push begins. The registration fits "
     "the tool&#8217;s pose and the indenter&#8217;s centre to many touch-offs."),
    ("Hold", "One force level of a press on the UR5, kept for about 1.5&#8202;s: one sample of the ground truth and of the "
     "board."),
    ("Location error", "The distance in mm from an estimated contact point to the true one, tangential to the pad; a "
     "median over holds."),
    ("Physical model", "The estimator of this page on a pad&#8217;s rest calibration, with no training."),
    ("Board GP, direction GP", "Gaussian processes (GP) trained on a tip&#8217;s own holds. The board GP maps the "
     "board&#8217;s change of 15 components to the contact point and the force. The input of the direction GP is only the "
     "change&#8217;s direction. It returns the point and the force per unit change, and that force is scaled by the "
     "change&#8217;s size."),
    ("Noise-equivalent force", "The force whose change at the board equals the noise of one record, one reading of the "
     "five chips, in mN. It is the record noise divided by the median change per newton of the presses at 1 to 2&#8202;N."),
    ("Tip frame", "\\(x\\) along the pad&#8217;s normal toward the palmar face, \\(y\\) across the finger, \\(z\\) from the "
     "mount toward the tip end, in mm."),
]

SECTIONS = [
    ("chain", "Components of the fingertip and its models",
     r"""<p>The fingertip is concept b of the DIP-link study, which the owner chose after the first prints of 2026-10-20. The pad
is the front 13.45 of the 17.0&#8202;mm of the 11/22 half-ellipsoid tip of the simulations. It slides into a PLA
cover on the DIP link, over the ReSkin 5X board. The board&#8217;s chips lie 0.42&#8202;mm below the pad&#8217;s back face; the current link is b v4
([DIP link b v4](exp:20261021-dip_link_b_v4/20261021-dip_link_b_v4_and_wall_heights.html)). Five
N52 magnets, a centre one and four corners 7.4&#8202;mm apart, sit with their tops 7.5&#8202;mm under the apex. That is the
shallowest depth that keeps 1.8&#8202;mm of TPU around every pocket. A press moves the magnets by a few micrometres; the
chips read the change of their field.</p>
""" + CHAIN_SVG + r"""
<p>Each box is one model with its own check. The finite elements resolve what the printer makes: walls, skins and
gyroid. The walls take the load at the face&#8217;s edges, and so they set the patch that the pads must reproduce. The field model
represents each magnet as a cylinder of 72 dipoles, because a single dipole is 5.5&#8202;% high at the chips&#8217;
distance. The rest calibration makes the chain usable on a pad as built. From 2&#8202;s of untouched reading it finds the
chips&#8217; layout and the magnets&#8217; poles and strength. Forward, the chain ranks designs by their force
resolution before anything is printed. Inverted, it turns readings into a contact point and a force without a robot.</p>"""),
    ("print", "The printed pad",
     r"""<p>The pads print in Siraya Tech 85A on the Prusa XL&#8217;s tool 2 or the P1S&#8217;s external spool. The nozzle is at
240&#8202;&#176;C and the bed at 35&#8202;&#176;C. The pad prints upright in 0.1&#8202;mm layers, with two perimeters
outside and around each pocket, four top and four bottom layers, and gyroid at 30 or 50&#8202;%. The print pauses at 9.2
and 16.2&#8202;mm of height so that the magnets can be placed, north toward the palmar face. Each magnet sits in a pocket
with a 2.9&#8202;mm seat and is held by press fit (the owner&#8217;s choice from the seat coupon, 2026-10-18). The recipe stays the same on every printer, so one FE describes every pad
([print procedure](exp:20261018-fingertip_print_procedure/20261018-siraya85a_fingertip_print_procedure.html)). The
pads weigh 0.6&#8202;% more TPU than the FE assumes, and concept a&#8217;s tips 3.0&#8202;% less, about 7&#8202;% softer
([DIP link b v3 and mount v2](exp:20261020-ur5_calibration_mount_v2/20261020-dip_link_b_v3_and_ur5_mount_v2.html)).</p>
{{FIG section}}
<p>85A at 30&#8202;% gyroid was chosen for the magnets&#8217; travel: the softer the pad, the more field change per
newton, until the approach outgrows what 1&#8202;mm pads represent. The rule from the co-design study is at least
12&#8202;&#181;m/N of travel for 0.1&#8202;N
([fingertip co-design](exp:20261014-tip_codesign/20261014-sphere_pad_accuracy_and_fingertip_codesign.html)); the
95A tips of the first studies gave 11.7&#8202;&#181;m/N at best. eFlesh lattices, printed open as eFlesh prints them, give
250&#8202;&#181;m/N of travel, but they sink 613&#8202;&#181;m at 1&#8202;N and print with unsupported islands. With a skin closing
every face they sink 81&#8202;&#181;m and give 0.075&#8202;N, which is the gyroid&#8217;s resolution
([eFlesh printability](exp:20261013-eflesh_printability/20261013-eflesh_lattice_printability.html)).</p>"""),
    ("fe", "Finite elements of the printed structure",
     r"""<p><b>Materials.</b> The tip&#8217;s shape is voxelized at 0.05&#8202;mm. In each layer plane, TPU within 0.9&#8202;mm
of the outline is wall. TPU within 0.8&#8202;mm of an exposed surface along the build axis is skin. The rest is core. The
magnets are a fourth class, and walls and skins line their pockets as the slicer prints them. The classes are
sampled onto a graded hexahedral grid, 0.2&#8211;0.25&#8202;mm under the face and 0.8&#8211;1.1&#8202;mm deep inside
(about 120&#8211;220 thousand degrees of freedom). Walls and skins are solid TPU, isotropic with Poisson 0.45. Their modulus is
30&#8202;MPa for 95A and 12&#8202;MPa for 85A, from datasheets; NinjaFlex stands in for Siraya&#8217;s unmeasured 85A. The magnets
are NdFeB at 160&#8202;GPa, bonded
([finite elements of the printed tip](exp:20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html)).</p>
<p><b>Gyroid core.</b> The core is homogenized from the gyroid as the slicer prints it. In each 0.2&#8202;mm layer the
toolpath follows the zero set of the gyroid function at mid-layer, with a 0.45&#8202;mm bead. Successive layers bond
only where their beads overlap:</p>
\[g(\mathbf x)=\sin X\cos Y+\sin Y\cos Z+\sin Z\cos X,\qquad \mathbf X=\frac{2.44\,\rho}{s}\,\mathbf x,\quad s=0.407\ \mathrm{mm}\]
<p>A periodic cell of these voxels takes six unit macroscopic strains \(\boldsymbol\varepsilon_i\); each load case solves
for the periodic fluctuation \(\mathbf u_i\), and the effective tensor is the cell&#8217;s mean stress:</p>
\[K\,\mathbf u_i=-\mathbf f_i,\qquad \mathbf f_i=\sum_e B_e^{\top}D\,\boldsymbol\varepsilon_i\,V_e,\qquad
C^{*}_{ij}=\frac{V_s\,D_{ij}+\mathbf f_i\cdot\mathbf u_j}{\lvert Y\rvert}\]
<p>The result is orthotropic, and it is stiffer along the build axis. At the settings used, 30&#8202;% prints 32.6&#8202;% solid.
Its modulus is \(E_x=0.087\,E_s\) in the layer plane and \(E_z=0.135\,E_s\) along the build axis. The setting 50&#8202;%
prints 55.0&#8202;% solid, with 0.228 and 0.311. Over 11&#8211;42&#8202;% solid the moduli follow
\(E_x/E_s=0.43\,\phi^{1.40}\) and \(E_z/E_s=0.60\,\phi^{1.29}\) in the solid fraction \(\phi\). These fits lie between
the bending and stretching limits of Gibson and Ashby.</p>
{{FIG moduli}}
<p><b>Tip model and contact.</b> The elements are eight-node hexahedra with Wilson&#8211;Taylor incompatible modes, so the
thin walls bend without shear locking. A 40-voxel cantilever comes out 2.4&#8202;% below beam theory, against 9.8&#8202;%
for plain hexahedra. Each element takes the tensor of its class, and the core&#8217;s tensor is rotated into the tip frame.
Concept b&#8217;s pad is fixed at its cut face and the mount face. One sparse Cholesky factorization of \(K\) is reused
for every load. A rigid frictionless indenter (the shaft, a plate or a sphere) meets the face nodes. Their normal compliance
\(W\) comes from columns of \(K^{-1}\). At approach \(\delta\) the nodal forces \(\mathbf p\) solve</p>
\[\mathbf p\ge0,\qquad \mathbf g_0-\delta\,\mathbf 1+W\mathbf p\ge0,\qquad \mathbf p^{\top}\big(\mathbf g_0-\delta\,\mathbf 1+W\mathbf p\big)=0,\]
<p>and \(\delta\) is bisected until \(\sum p_i\) equals the load. A magnet&#8217;s motion is the rigid fit of its nodes&#8217;
displacements. Each gap \(g_0\) is measured from the tip&#8217;s analytic surface; on the curved tips the grid&#8217;s face
nodes lie a median 0.10&#8211;0.13&#8202;mm from that surface (95th percentile 0.43&#8211;0.49&#8202;mm).</p>
<p><b>Solver.</b> The solver is our own Python (<code>scripts/tpu_tip_fem_core.py</code>; SciPy, CHOLMOD and PyAMG on
the CPU, PyTorch on the GPU). Its input is voxels: the slicer&#8217;s material classes at 0.05&#8202;mm, or every gyroid
bead at 0.1&#8202;mm. On a voxel grid all elements of one size share one element matrix. The model resolving every
bead, 1.7&#8211;2.2 million elements and 5&#8211;8 million degrees of freedom, is then solved matrix-free on the GPU in
2&#8211;3&#8202;GB and 60&#8211;140&#8202;s per approach. The off-the-shelf solvers installed for the study run on the CPU
from tetrahedral meshes. FEniCSx took 3.3&#8211;5.7&#8202;GB for 0.45&#8211;0.68 million degrees of freedom of quadratic
tetrahedra. At that rate the resolved tip needs above 40&#8202;GB, and the workstation has 30&#8202;GB. PolyFEM&#8217;s IPC
contact took 7&#8211;14&#8202;min for 57 thousand degrees of freedom. One element code solves the periodic cells, the
homogenized tip and the resolved tip, so their comparison tests the homogenization alone. FEniCSx and PolyFEM remain as checks
([finite elements of the printed tip](exp:20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html),
Tables 7 and 8).</p>
<p><b>Checks.</b> On the 95A block of the first study the model is 1.9&#8202;% off Hertz under a 40&#8202;mm sphere. With
every bead resolved at 0.1&#8202;mm, the approach is within 2.6&#8202;% of the homogenized tip&#8217;s (44.8 against
46.0&#8202;&#181;m at 1&#8202;N). The same check was run on the 85A 11/22 tip and b&#8217;s pad, with 6&#8211;8 million
degrees of freedom on the GPU. It puts the resolved beads 1.6&#8211;4.8&#8202;% stiffer than the homogenized core under the rods, and up
to 7.8&#8202;% stiffer under a 2&#8202;mm sphere. About 1&#8202;% of that comes from the cell&#8217;s voxel size. A finer
contact grid changes the approach by at most 3.6&#8202;% (v2 fingertip checks, Section 6;
[calibration to the finite elements](exp:20261018-fe_contact_calibration/20261018-fe_contact_calibration.html)).
Pad B00 was solved from its G-code. Under the rod at 1&#8202;N it sinks 18&#8202;% less than the design pad when its beads are
drawn at their nominal widths. Those widths put 11&#8202;% more TPU in the model than the printed parts weigh. With its TPU
set to the weighed volume the pad sinks 3.9&#8202;% less. Its 0.1&#8202;mm layers make the gyroid 8&#8202;% stiffer in the
layer plane. With that cell the homogenized pad agrees with the G-code model within 1&#8202;% under the rod
([v2 fingertip checks](exp:20261019-v2_tip_fe_checks/20261019-v2_fingertip_stiffness_checks.html)). The G-code solve of
the 95A tip of 10-13 read none of the G-code&#8217;s arc moves, which are 11&#8202;% of its filament. With them the tip sinks
33&#8202;&#181;m at 1&#8202;N against the design model&#8217;s 41, where the earlier reading gave 63
([eFlesh printability](exp:20261013-eflesh_printability/20261013-eflesh_lattice_printability.html)).</p>
{{FIG film_g20}}"""),
    ("changes", "Effects of the design changes",
     r"""<p>Each change below was solved on the same models. Approaches are under the 12.5&#8202;mm shaft at 1&#8202;N, and
resolutions are per sample at the contact point
([finite elements of the printed tip](exp:20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html),
[v2 fingertip checks](exp:20261019-v2_tip_fe_checks/20261019-v2_fingertip_stiffness_checks.html)).</p>
<div class="tw"><table class="index"><thead><tr><th>Change</th><th>Effect</th></tr></thead><tbody>
<tr><td>Infill, 85A</td><td>Solid 20.2&#8202;&#181;m (\(E^*\) 47.5&#8202;MPa); 50&#8202;% gyroid 43.3&#8202;&#181;m (14.4); 30&#8202;% 74.7&#8202;&#181;m (6.3) on the block; 95A at 20&#8202;% 46.0&#8202;&#181;m</td></tr>
<tr><td>Walls</td><td>Take the load at the face&#8217;s edges: 53&#8202;% of the load on the outer 1.2&#8202;mm of the 20&#8202;% tip, against 30&#8202;% for the pads&#8217; law; twist arm 3.35&#8202;mm against the pads&#8217; 2.72</td></tr>
<tr><td>Print deviations</td><td>Perimeters 10&#8202;% thinner +8.6&#8202;% approach; gyroid beads 10&#8202;% narrower +9.3&#8202;%; under-extrusion 5 or 10&#8202;% +8.7 or +19.6&#8202;%; skins 0.7 for 0.8&#8202;mm +0.05&#8202;%</td></tr>
<tr><td>Magnets as inclusions</td><td>Five 3&#8202;&#215;&#8202;2&#8202;mm at 7.5&#8202;mm stiffen the 11/22 tip by 3.4&#8202;%; one 4.76&#8202;mm magnet 1.5, 3 or 5&#8202;mm deep in the 95A block by 99, 31 or 15&#8202;%</td></tr>
<tr><td>Five or four magnets</td><td>In b&#8217;s pad at 7.5&#8202;mm: 0.0285&#8202;N with five, 0.0513 with four; five at 7.0 or 6.5&#8202;mm 0.0299 or 0.0308 (more travel, farther from the chips); at 8.5&#8202;mm the 41&#8202;mT field exceeds every gain</td></tr>
<tr><td>Magnet shape</td><td>A 3&#8202;mm N52 sphere 0.0285&#8202;N on pad B00, the 3&#8202;&#215;&#8202;2&#8202;mm disc of equal volume 0.0304&#8202;N; spheres of 4&#8202;mm and more exceed the chips&#8217; range at the five-magnet centres ([spherical magnets](exp:20261021-spherical_magnets/20261021-spherical_and_disc_magnets.html))</td></tr>
<tr><td>One perimeter per pocket</td><td>Five magnets 0.0287&#8202;N, four 0.0460; travel 5.23 to 6.04&#8202;&#181;m/N</td></tr>
<tr><td>Concept b against a</td><td>The cut pad of b: 2.5&#8202;% stiffer, 0.028 against 0.065&#8202;N, contact located to 0.34&#8202;&#215;&#8202;0.42 against 1.40&#8202;&#215;&#8202;1.31&#8202;mm, chips 3.5&#8202;mm closer</td></tr>
<tr><td>Cover walls on b&#8217;s flanks</td><td>Held both ways in the model (an upper bound): 3.0&#8202;mm walls &#8722;0.16&#8202;% approach and 0.030&#8202;N; v3&#8217;s 6&#8202;mm walls 0.0313&#8202;N; with the proposed tip cap 0.0317&#8202;N, with its side walls 0.0332&#8202;N</td></tr>
<tr><td>Shape</td><td>Of 24 printable fillet shapes none turns the tool better than the as-built block; curved tips cut the twist-onset arm to a third; in 85A at 30&#8202;% all shapes sink 251&#8211;262&#8202;&#181;m under a 2&#8202;mm sphere</td></tr>
</tbody></table></div>
<p>Two choices follow from the table. Five magnets are used, because the centre magnet over U1 halves the resolution of four.
Concept b is used, because the cut pad puts the magnets 3.5&#8202;mm closer to the chips. The cover walls and the cap cost
at most 16&#8202;% of the resolution, in the bound that fixes the flanks both ways.</p>"""),
    ("readout", "Magnetic readout and force resolution",
     r"""<p>The ReSkin 5X board has five MLX90393 three-axis Hall magnetometers in a plus, U1 at the centre and U2&#8211;U5
7.00&#8202;mm from it. Each chip senses 0.42&#8202;mm under its package top. The chips are read over the STEMMA QT port of
a QT Py ESP32-S3 in 2.0&#8202;ms, 375 records a second. Each chip converts every 7.0&#8211;7.1&#8202;ms (140&#8211;143&#8202;Hz).
The firmware sets the gain at GAIN_SEL&#8202;1, RES&#8202;1. That gives 1.20&#8202;&#181;T a count on \(x\) and \(y\), 1.94 on
\(z\), and a range of 38.4&#8202;mT. This gain is used because b&#8217;s five magnets put 27.5&#8202;mT on the nearest chip,
and the most sensitive gain covers only 4.8&#8202;mT. At rest the measured noise per sample is 1.54&#8211;1.76&#8202;&#181;T on \(x\) and \(y\) and 2.32&#8211;2.59&#8202;&#181;T on
\(z\), 1.4 times the datasheet model for this gain. Three fingers&#8217; boards answer at the same addresses and need an
I&#178;C multiplexer. Three fingers on one multiplexer were read together on 2026-10-25 at 152 records a second, so every
conversion of the 15 chips is read. Each chip&#8217;s noise matched that of the same finger read alone to within 2&#8202;%, and
the boards&#8217; readings did not correlate (at most 0.031)
([magnetometer readout](exp:20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html),
<code>src/morphohand/driver/reskin5x/README.md</code>). Three v1 fingertips left at rest on one multiplexer for
8.4&#8202;h on 2026-10-26/27 were read without a gap, and the viewer showed no contact at its 2&#8202;&#181;T floor
([three fingertips at rest](exp:20261027-three_v1_fingers_overnight/20261027-three_v1_fingertips_at_rest_overnight.html)).</p>
{{FIG board}}
<p>The design studies rank layouts by force resolution. The FE gives each magnet&#8217;s translation and rotation per newton
of a patch load. The field model turns that into the change of the 15 readings, which is the sensitivity \(S\). The noise
\(\Sigma\) then bounds how well a force can be read. With a uniform background field \(\mathbf b\) (Earth&#8217;s) unknown, and
the contact point too when it is unknown,</p>
\[\mathbf r=S\,\mathbf f+U\mathbf b+\mathbf n,\qquad J=\big[\,S\ \ U\ \ \partial_yS\,\mathbf f\ \ \partial_zS\,\mathbf f\,\big],\qquad
\sigma_{f_n}=\sqrt{\big[(J^{\top}\Sigma^{-1}J)^{-1}\big]_{nn}},\]
<p>the Cram&#233;r&#8211;Rao bound on the pressing force from one sample. It is a lower bound: it assumes the models are exact.
On the first 95A tip one 4.76&#8202;mm magnet 5&#8202;mm deep resolved 0.044&#8202;N at a known point. When the point
was unknown it resolved 0.08&#8211;1.48&#8202;N over the face. For that reason the layouts went to four magnets
(0.14&#8202;N everywhere) and then five. On the 11/22 tip concept a resolves 0.065&#8202;N and b resolves 0.028&#8202;N. The
contact is located to 0.34&#8202;&#215;&#8202;0.42&#8202;mm with b and 1.40&#8202;&#215;&#8202;1.31&#8202;mm with a. At the measured
noise b&#8217;s 0.028&#8202;N becomes about 0.039&#8202;N ([v2 fingertip checks](exp:20261019-v2_tip_fe_checks/20261019-v2_fingertip_stiffness_checks.html)).</p>
<p>The disturbances are larger than the noise. The chips run without their on-chip temperature compensation. Each
chip&#8217;s sensitivity falls by 0.65&#8211;0.70&#8202;% per &#176;C of its die temperature. On the centre chip, in
7.6&#8202;mT, one degree is 53&#8202;&#181;T, which is twice a 1&#8202;N press on the face. The board warms by about
1&#8202;&#176;C in the minutes after it is plugged in. Dividing each reading by its chip&#8217;s gain at the chip&#8217;s own temperature leaves 9 of those 70&#8202;&#181;T
([contact estimation](exp:20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html), Section 9). On the first tip, a facing
fingertip&#8217;s magnet moving from 25 to 30&#8202;mm read as 5.16&#8202;N, and as 2.01&#8202;N with the field&#8217;s gradient
estimated. The other magnets&#8217; positions follow from the hand&#8217;s joint angles, so their field can be subtracted. Earth&#8217;s field, 1.71&#8202;N on U1 alone,
drops out of the five-chip estimate. The board must not move against the pad. On concept a with four magnets,
1&#8202;&#181;m of board motion reads as 0.160&#8202;N, and for that reason the board is clamped
([DIP link board mount](exp:20261018-dip_link_board_mount/20261018-dip_link_magnetometer_board_mount.html)).</p>"""),
    ("estimate", "Contact point and force from the readings",
     r"""<p>The estimator runs the chain backwards
([contact estimation](exp:20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html)). At rest
the 15 components are the magnets&#8217; field and Earth&#8217;s. The rotation of each chip is taken from the board&#8217;s
pick-and-place file, and one common offset is fitted. The rest calibration then finds which address sits at which
position. It also finds every magnet&#8217;s pole, the magnets&#8217; strength and the pad&#8217;s offset on the board. On pad B00 as first assembled it fits at
0.43&#8202;mT RMS against fields of 1.9&#8211;7.0&#8202;mT. It finds magnets at 35&#8202;% of an N52&#8217;s moment, with two
diagonal corners reversed. The owner confirmed that these magnets were not N52 and went in at random. A press then changes the reading
by</p>
\[\Delta\mathbf b=S(\mathbf p)\,\mathbf f+N\mathbf q+\mathbf n,\qquad N=\big[\,U\ \ \mathbf b^{\mathrm{rest}}\,\big],\]
<p>where \(S(\mathbf p)\) comes from an FE basis of 1&#8202;N loads along \(x\), \(y\) and \(z\) at the 812 pad points.
The points cover the whole free surface (2,436 load cases, 101&#8202;s, 2.0&#8202;GB). A thin-plate spline interpolates between the points.
The vector \(\mathbf q\) contains a uniform field change and a strength change, each with a Gaussian prior. With \(C\) the covariance of the
noise and the nuisance terms and a force prior \(\sigma_f=3\)&#8202;N, the force at each candidate point and the point
itself follow from</p>
\[\hat{\mathbf f}(\mathbf p)=\big(S^{\top}C^{-1}S+\sigma_f^{-2}I\big)^{-1}S^{\top}C^{-1}\Delta\mathbf b,\qquad
\hat{\mathbf p}=\arg\max_{\mathbf p}\ \Big[\Delta\mathbf b^{\top}C^{-1}S\,\hat{\mathbf f}-\ln\det\big(I+\sigma_f^{2}S^{\top}C^{-1}S\big)\Big],\]
<p>the log marginal likelihood of a contact at \(\mathbf p\). It is maximized over 12,581 candidate points about
0.2&#8202;mm apart, with the force pressing, inside a friction cone of 2 and below 25&#8202;N. The determinant charges each point for the patterns it
could explain. Without it the best fit often went to a flank point where the reading barely responds, with forces of
100&#8202;N and more. The Fisher information gives each estimate&#8217;s 1-sigma; one estimate takes 1.2&#8202;ms.</p>
<p>Of 600 simulated presses with N52 magnets and the 50&#8202;ms window, 2&#8202;% are placed more than 1.5&#8202;mm away. The
rest have median errors of 0.20&#8202;mm and 0.036&#8202;N on the face, and 0.17&#8202;mm and 0.050&#8202;N on the flanks. The
errors follow the predicted 1-sigma (ratio 1.10&#8211;1.16). With the magnets in the pad today, 15&#8202;% are placed away
and the rest have 0.39&#8202;mm and 0.08&#8202;N. Over 30&#8202;s without a press, one frame of 880 shows a contact.</p>
<p>Steel near the pad is magnetized by the magnets, and its field at the chips is that of the magnets&#8217; mirror images.
A 100&#8202;g mass and a steel ruler, touched to the second fingertip&#8217;s face without force, changed the reading by
217 to 360&#8202;&#181;T. Each touch was estimated as a press of 9.0 to 17.5&#8202;N on a side wall. The images lie four times
as far from the chips as the magnets, so their field varies little across the board. A uniform field and a linear gradient
fit 94 to 98&#8202;% of each touch&#8217;s change and 28 to 30&#8202;% of a press&#8217;s. With the viewer&#8217;s key f, each
contact is compared with such a field and not shown when the field fits better by more than 25 in chi-square
([fields from outside the pad](exp:20261023-foreign_field_rejection/20261023-foreign_field_rejection.html)). The frames
shown as contacts during six touches fell from 432 to 29. Of 20,851 contact frames in the UR5 streams, 117 were
rejected, none inside a hold. A press made with a steel object is rejected where the object&#8217;s field dominates,
so the test is off by default.</p>
{{FIG surface}}
{{FIG viewer}}"""),
    ("calibration", "Calibration against the HEX-E on the UR5",
     r"""<p>The calibration presses the pad on indenters fixed in the lab&#8217;s UR5 cell, with the HEX-E between the flange
and the mount ([calibration protocol](exp:20261022-ur5_tip_calibration/20261022-ur5_tip_calibration_protocol.html)). The
HEX-E is 35&#8202;mm tall, so the face centre lies 128.4&#8202;mm from the flange. A session homes the tool on the
indenter&#8217;s key and fits the HEX-E&#8217;s offsets in a gravity sweep. Touch-offs of the pad on a sphere of radius
5&#8202;mm then register the tool and the sphere&#8217;s centre to the pad. Presses follow the surface normal at points over
the face, the edges and the flanks, at force levels of 0.25 to 5&#8202;N. Rolls and twists turn the pad on the sphere. In
a simulated session the registration gives the contact point within 0.07&#8202;mm and the force within 0.4&#8202;%
(medians).</p>
<p>The program runs on the lab&#8217;s computer, which pushes each run for this workstation to analyse
([lab runs](exp:20261022-ur5_tip_calibration/20261022-ur5_tip_calibration_lab_runs.html)). In the session sphere2 of
2026-10-22, 29 touch-offs registered the tool with a residual of 57&#8202;&#181;m RMS on the pad. They moved the tool centre
point by 2.01&#8202;mm and 2.81&#176; from its CAD and homing values. Then 74 presses gave 357 holds, and 10 rolls and twists
gave 78 more. Over 65&#8202;min the reference press kept its onset within 0.07&#8202;mm and its depth within 0.05&#8202;mm,
so the pad stayed in its seat.</p>
{{FIG ur5_film}}
<p>Under the sphere the board&#8217;s force agrees with the HEX-E&#8217;s. Its slope against the HEX-E&#8217;s normal force is 0.83 on the
face, 0.78 on the edges and 1.05 on the flanks. The registered indentation is larger than the FE pad&#8217;s at the
measured force: at that depth the FE pad would push back with 5.1 times the measured force. If the pad itself were that
much softer than its model, the magnets would move about five times as far per newton. The board would then read about
five times the HEX-E&#8217;s force. The pad&#8217;s bulk therefore lies within about 22&#8202;% of the FE model. The extra
indentation fits 0.33&#8202;mm plus 0.20&#8202;mm per newton, in parts whose motion the magnets do not see
([UR5 ground truth and predictors](exp:20261022-ur5_tip_calibration/20261022-ur5_tip_ground_truth_and_predictors.html),
Sections 3 and 4). Presses under a rod of radius 12.5&#8202;mm, in the sessions of 2026-10-23, travel 0.07&#8202;mm plus
0.10&#8202;mm per newton beyond the FE pad. A spring in series with the pad, in the mount, the arm or the indenter&#8217;s
stand, would add the same travel per newton under both indenters. At most 0.10&#8202;mm/N therefore lies in series. At
least 0.10&#8202;mm/N arises at the sphere&#8217;s contact, where the FE pad is stiffer than the print (Section 8 of the
same page).</p>
<p>From 1&#8202;N up, the untrained estimator places contacts 1.6 to 2.6&#8202;mm from the registered points (medians by
force level). Its forces are within 10 to 26&#8202;% of the HEX-E&#8217;s. At 3&#8202;N and over its mean error is a
systematic offset of 1.4&#8202;mm. Under the rod its force is 0.63 of the HEX-E&#8217;s on the face and 0.71 on the
edges. Below 1&#8202;N it places contacts a median 9.8&#8202;mm away. Each press leaves the rest
about 7&#8202;&#181;T off, the change of 0.3&#8202;N, and below 1&#8202;N that step is as large as the press itself.</p>
{{FIG board_hexe}}
<p>Learned models do better on the same pad. A Gaussian process (GP) from the board change to the contact point and normal
force was trained on 59 presses. It places 15 held-out presses within 0.84&#8202;mm and 0.086&#8202;N. On the same holds the
untrained estimator gives 3.79&#8202;mm and 0.382&#8202;N. An ensemble of five small multilayer perceptrons (MLP) gives
0.74&#8202;mm and 0.061&#8202;N. Over four folds the MLP gives 0.79&#8202;mm and the GP 1.00&#8202;mm. On the rolls and twists,
at orientations that no press had, the GP gives 2.05&#8202;mm, the MLP 2.93&#8202;mm and the untrained estimator
2.44&#8202;mm. Trained on 45 rod presses, the MLP places held-out contacts within 0.57&#8202;mm and the GP within
1.05&#8202;mm.</p>
<p>The learned models held a day later. The lab repeated 27 of sphere2&#8217;s grid presses about 18&#8202;h later. Trained on
all of the night&#8217;s presses, the GP placed their contacts within 0.80&#8202;mm and the MLP within 0.71&#8202;mm. At 1 and
2&#8202;N the board change at the same target differed from the night&#8217;s by 38 and 20&#8202;%. The night&#8217;s own
repeats of one press differed by 35 and 23&#8202;%. The pad met the sphere about 0.15&#8202;mm earlier at the same force
([UR5 ground truth and predictors](exp:20261022-ur5_tip_calibration/20261022-ur5_tip_ground_truth_and_predictors.html),
Section 9).</p>
<p>Over the 2.3&#8202;h session the board&#8217;s zero moved by up to 71&#8202;&#181;T. Earth&#8217;s field, turning with the
tool, made 98&#8202;% of that change, and the estimator absorbs it as a uniform field. A press leaves a step of 7&#8202;&#181;T
along its own change. Of that step 0.29 lasts, and the rest fades with a time constant of 19&#8202;s. No learned zero beat a
re-zero at rest before each press ([board drift](exp:20261022-ur5_tip_calibration/20261023-ur5_tip_board_drift.html)).</p>
<p>The HEX-E is a weak reference. Its Fz channel reads 0.78 and 0.87 of the other axes&#8217; gain in two gravity fits, and
its link failed in three of the first night&#8217;s eight runs. An ATI Mini45 of the lab is to replace it. It will first be
calibrated on the UR5 with a weighed mass hung at five points (39 stations, about 27&#8202;min). In simulated sessions
that calibration recovers the sensor to 0.021&#8202;N with the lab&#8217;s 100&#8202;g weight in a printed holder, and to
0.005&#8202;N with a 0.5&#8202;kg mass (medians)
([ATI calibration plan](exp:20261023-ati_ft_calibration/20261023-ati_ft_calibration_plan.html)).</p>"""),
    ("transfer", "Transfer to other magnets and a second fingertip",
     r"""<p>A calibration learned on one pad must also work on a pad with other magnets. This was simulated on sphere2&#8217;s
mechanics: each hold keeps its true point and force, and its board change is synthesized for other magnets. The physical
model, rebuilt from each set&#8217;s rest calibration, gives 2.02&#8202;mm with today&#8217;s magnets. It gives 1.52&#8202;mm
with N52 magnets all one way. It gives 1.36&#8202;mm with N52 magnets in the checkerboard now printed, with the centre magnet
and one diagonal one way and the other diagonal the other way. Trained on today&#8217;s magnets, a GP on the raw board change
is off by 1.64 and 4.09&#8202;mm on the two N52 sets. A GP that corrects the physical estimate, with that estimate as its
input, gives 0.73 and 0.83&#8202;mm. Retrained on the checkerboard, the GP gives 0.61&#8202;mm. On sphere2&#8217;s real presses the corrected estimate is off by 1.50&#8202;mm
over four folds, and the MLP by 0.79&#8202;mm
([UR5 ground truth and predictors](exp:20261022-ur5_tip_calibration/20261022-ur5_tip_ground_truth_and_predictors.html),
Tables 8 and 9).</p>
<p>The second fingertip on the workstation&#8217;s desk has its two magnets at the mount end reversed. Its rest reading alone
gave that pattern, at 0.25&#8202;mT RMS against 1.06&#8202;mT for the next pattern. With the fingertip fixed to the desk,
the owner pressed the face centre six times with the 5&#8202;mm sphere indenter. The physical model rebuilt from the rest
fit placed the six within 0.93&#8202;mm of their mean. That mean lies 3.07&#8202;mm from the reference point, which was
aimed at by eye, at 2.9 to 4.4&#8202;N. For all six presses sphere2&#8217;s GP on the raw board change returned one and the same
point. Its MLP put them 9.9&#8202;mm away. Sphere2&#8217;s correction moved the points 2.3&#8202;mm toward the mount end. No
force was measured, so the correction is not yet scored on this pad
([second fingertip](exp:20261023-second_fingertip_transfer/20261023-second_fingertip_transfer.html)).</p>
{{FIG real_presses}}
<p>The first checkerboard tip was then pressed on the UR5 against the HEX-E (session cb1t, on the evening of 2026-10-23:
302 press holds and 64 turn holds). One pole pattern fits its field at rest, at 0.24&#8202;mT against 0.87&#8202;mT for the
next. It is the planned checkerboard: the owner had named its poles from the board&#8217;s side. Built from that fit alone, the physical model placed held-out contacts within 2.06&#8202;mm
and 0.28&#8202;N, against 3.79&#8202;mm and 0.38&#8202;N on sphere2&#8217;s tip. At 2&#8202;N and over its error was 1.1 to
1.2&#8202;mm. Trained on sphere2&#8217;s presses, no learned model came closer. The GP on the raw board change missed by
5.3&#8202;mm, and sphere2&#8217;s correction of the physical estimate raised the error to 2.5&#8202;mm, where the simulation
had lowered it. Sphere2&#8217;s offset between the board&#8217;s point and the registered one, 1.4&#8202;mm, is 0.7&#8202;mm in
another direction on this tip. Trained on 47 of cb1t&#8217;s own presses, the GP reached 0.71&#8202;mm and the MLP
0.42&#8202;mm ([checkerboard fingertip](exp:20261022-ur5_tip_calibration/20261024-checkerboard_fingertip_on_the_ur5.html),
Sections 3, 6, 7 and 8). Corrected by a GP whose inputs are its own estimate, trained on the same presses, the physical
model missed by 0.55&#8202;mm at 2&#8202;N and over. On turn holds at orientations that no press had it missed by
0.71&#8202;mm, where the GP on the board missed by 1.03&#8202;mm. Below 1&#8202;N the model and its correction both
fail.</p>
{{FIG checkerboard}}
<p>The tip v2 ID 1 has five N52 magnets, fitted at rest at 0.98 of an N52&#8217;s strength (UR5 sessions of 2026-10-26
and 27). Trained on its own 198 press actions, the GP places held-out contacts within 0.25&#8202;mm at 1.5&#8202;N and over
and 0.66&#8202;mm below, as on tip 2. The physical model places presses under 1.5&#8202;N within 2.0&#8202;mm, where the v1
tips&#8217; weaker magnets gave 8&#8211;10&#8202;mm (rows <code>20261027-v1_v2_tips.json</code>). A GP of the v1 tips misses
v2 ID 1&#8217;s contacts by 2.52&#8202;mm. It misses by 0.64&#8202;mm once each board change is scaled by its magnets&#8217;
strength ([calibration on the UR5](exp:20261026-fingertip_calibration/20261026-fingertip_calibration_on_the_ur5.html),
Sections 9 and 10). On the four checkerboard tips the own-press GP gives the force&#8217;s direction within 2.1&#8211;2.5&#176; on the
face at 1.5&#8202;N and over. On the edges it is within 3.1&#8211;4.6&#176;, and on the flanks, where fewer presses were made,
within 4.9&#8211;8.5&#176; (rows <code>20261027-direction_by_region.json</code>). A GP trained on v2 ID 1&#8217;s presses on the
12.5&#8202;mm rod places their contacts within 0.29 and 0.73&#8202;mm over and under 1.5&#8202;N. Its forces are within
0.22&#8202;N and 2.7&#176; (rows <code>rod_tips.json</code>).</p>
<p>v2 ID 1 was also pressed at 2 to 8&#8202;N, at actions outside its training. Trained on its holds of 2&#8202;N and under,
the board GP places those presses 1.04&#8211;3.84&#8202;mm off. Its force is 45&#8211;87&#8202;% too small there. With the
direction GP, which is given only the direction of the board&#8217;s change, the same presses are placed within
0.25&#8211;0.31&#8202;mm and their force within 5.1&#8211;6.2&#8202;%. The tall indenter is a 5&#8202;mm sphere 160&#8202;mm
above its base, where the robot reaches more of the edges and flanks. The GP of the low sphere&#8217;s presses places its
points 0.98&#8211;1.21&#8202;mm off at 1.5&#8202;N and over. The tall indenter&#8217;s forces have a shear along the finger
of 0.08&#8211;0.13&#8202;N per newton. On the low sphere it is &#8722;0.02 to 0.02&#8202;N per newton
([calibration on the UR5](exp:20261026-fingertip_calibration/20261026-fingertip_calibration_on_the_ur5.html#v2id1),
Section 10).</p>
<p>In the tip v3 ID 1, four N52 magnets lie in a 5.0&#8202;mm square under pad 05R4. On the night of 2026-10-27 the tip ran
the grid stage of v2 ID 1&#8217;s suite on the low sphere. That stage has 160 press actions with 818 holds. The HEX-E fell
off its acquisition at 02:54, after 40 of the next stage&#8217;s 84 slides, and the session stopped. The board&#8217;s
change is 28.3&#8202;&#181;T per newton at 1 to 2&#8202;N with four magnets. With five, on v2 ID 1&#8217;s presses of the same actions,
it is 52.3&#8202;&#181;T. The record noise is alike, 1.77 and 1.83&#8202;&#181;T. The noise-equivalent force is
therefore 63&#8202;mN with four magnets and 35&#8202;mN with five. The same ratio, 1.8, was predicted by the readout model. The
calibrated errors stay alike. With the direction GP, contacts are placed within 0.26 and 0.28&#8202;mm, and the force errs
by 5.0 and 5.1&#8202;%. On the 40 slides, a board GP trained with the other slides gives the shear within 0.123&#8202;N
with four magnets and 0.140&#8202;N with five. The physical model on pad 05R4&#8217;s FE basis places contacts within
2.64&#8202;mm with four magnets and 1.61&#8202;mm with five
([calibration on the UR5](exp:20261026-fingertip_calibration/20261026-fingertip_calibration_on_the_ur5.html#v3id1),
Section 11).</p>
<p>On the hand the other fingertips&#8217; magnets add their field to each board. In v2 ID 1&#8217;s field model a
five-magnet neighbour adds 513&#8202;&#181;T with the pads facing and touching, and 4,162&#8202;&#181;T side by side. A
0.5&#8202;N press changes the board by 23&#8202;&#181;T. Subtracting the field predicted from a pose 0.3&#8202;mm and 0.3&#176;
off leaves 23 and 457&#8202;&#181;T. The neighbour&#8217;s pose can be fitted from both boards, with the joint angles as a
1&#8202;mm, 1&#176; prior. The bound on a press at the face&#8217;s centre is then 0.147&#8211;0.149&#8202;N facing and
0.158&#8211;0.176&#8202;N side by side. With the pose known it is 0.147&#8202;N
([magnetic crosstalk](exp:20261027-fingertip_crosstalk/20261027-magnetic_crosstalk_between_neighbouring_fingertips.html)).
Between two four-magnet v3 tips on the UR5 the field was measured. Touching, they changed each other&#8217;s boards by 39
to 60&#8202;&#181;T. Of the change within 10&#8202;mm, 37 to 63&#8202;% is explained by the dipole model fixed before the
data. With a net moment of 7 to 16&#8202;% of one magnet fitted for each tip, the share rises to 85 to 95&#8202;%.
Fitted on those touch-offs alone, the moments explain 97&#8202;% of the change at either held board in 137 hovers without
contact ([measured on the holder](exp:20261027-fingertip_crosstalk/20261027-magnetic_crosstalk_between_neighbouring_fingertips.html#holder)).</p>
{{FIG crosstalk}}"""),
    ("pads", "The fingertip in the simulations",
     r"""<p>The simulations represent the tip by sphere pads, which are 0.75&#8202;mm spheres on a 1&#8202;mm lattice over its surface.
Each pad has stiffness \(K_i=E\,a_i/h_i\). In builder v2, the default since 2026-10-15, \(h_i\) is the pad&#8217;s depth
below the tip&#8217;s volume centroid. A stiff local modulus behind a series spring of 9.7&#8202;N/mm fits the FE model of
the 11/22 tip within 5.6&#8202;% RMS in approach. With a spring in series, pads fitted to the FE pad follow the indentation
measured on the UR5 within 135&#8202;&#181;m RMS, against 133&#8202;&#181;m for the FE pad. From the board&#8217;s force, the
printed pad&#8217;s bulk is within about 22&#8202;% of the FE model. The estimator reports its forces at the same pads the
simulations use, so a measured contact and a simulated one can be compared pad by pad. The fits, the turn they change in
two simulators, the rod presses and the pads&#8217; creep under the measured shear are in the contact overview
(<a href="contact_model.html#pad-calibration">calibration of the pads</a>).</p>"""),
    ("hardware", "Hardware built and measured",
     r"""<p>The DIP link holds the board against the pad. Three concepts were drawn: the window plate with a clamp (a), a
cover that clamps the board behind the pad on rails (b) and a board printed over (c). On 2026-10-20 the owner chose b
(&#8220;concept B is the clear winner&#8221;), which also resolves 0.028 against 0.065&#8202;N. In version 2 the
board keeps its own Qwiic connector, and the cable lies in a groove past the slotted arm; both links are 9.75&#8202;mm longer
([DIP link board mount](exp:20261018-dip_link_board_mount/20261018-dip_link_magnetometer_board_mount.html)). Version 3 adds
6&#8202;mm cover walls along the pad&#8217;s flanks, a column under the pad and a cable leg open to the top
([DIP link b v3](exp:20261020-ur5_calibration_mount_v2/20261020-dip_link_b_v3_and_ur5_mount_v2.html)).
Version 4 recesses the bridge&#8217;s palmar edge and shrinks the column. A flat object then meets the pad 4.0&#8202;mm
before any rigid part, against 0.0&#8202;mm on v3. In a height test the owner chose 3&#8202;mm retaining walls. The
snug covers 3S and 2S fit pad B05R at zero interference on the rails
([DIP link b v4](exp:20261021-dip_link_b_v4/20261021-dip_link_b_v4_and_wall_heights.html)).</p>
<p>In the owner&#8217;s first presses by hand, on 2026-10-20, the pad rocked in its seat. After the fourth and fifth
presses the reading stayed 62 and 104&#8202;&#181;T from rest. A rigid shift of the whole pad by 17&#8211;23&#8202;&#181;m,
with a tilt of about 4&#8202;mrad, reproduces this. The estimator read the shift as standing contacts of
11&#8211;12&#8202;N. The same evening a cap went over the tip end, with a 2&#8202;mm pin into the pad
([DIP link board mount](exp:20261018-dip_link_board_mount/20261018-dip_link_magnetometer_board_mount.html)). The viewer
now adds a noise floor of 2&#8202;&#181;T, shows a contact only above 0.25&#8202;N and takes a new baseline after 3&#8202;s
at rest. No contact then shows at rest over a night
([contact estimation](exp:20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html), Section 9).
On the UR5 the reference press kept its onset within 0.07&#8202;mm over 65&#8202;min.</p>
{{FIG hand}}
<p>The UR5 calibration mount clamps the DIP link to the HEX-E with the finger along the tool axis
([UR5e mount v1](exp:20261018-ur5_calibration_mount/20261018-ur5_calibration_mount.html)). The pad&#8217;s normal force then
falls on the sensor&#8217;s lateral axes. Version 3 takes the lab&#8217;s
M6&#8202;&#215;&#8202;20 screws, captive in the fork&#8217;s foot, with M3 heat-set inserts in the adapter; sphere2 ran on
it with DIP link b v4 ([mount v3](exp:20261021-ur5_calibration_mount_v3/20261021-ur5_calibration_mount_v3.html)). Pressed
hard, the DIP link turns a little in the fork. Mount v4 adds a stop plate behind the link&#8217;s tongue with three set
screws. Mount v5 replaces the screws by a PLA wedge, since the lab has no brass set screws. In the model either cuts the
link&#8217;s turn under a face press from 1.0&#176; to 0.05&#176;
([mount v4](exp:20261023-ur5_calibration_mount_v4/20261023-ur5_calibration_mount_v4.html),
[mount v5](exp:20261023-ur5_calibration_mount_v5/20261023-ur5_calibration_mount_v5.html)). On 2026-10-23 the owner found
v5&#8217;s wedge unusable, and cb1t ran on v3&#8217;s fork. In mount v6 a 2.5&#8202;mm plate pressed onto the tongue&#8217;s
back, 9.3&#8202;N/mm at the contact by a beam estimate. The owner judged that it still flexed (2026-10-25: &#8220;flex is
still a big problem&#8221;). Mount v7 is v3&#8217;s fork, leg walls included, with a solid back whose 14&#8202;mm face
presses on the tongue. By the same estimate it is 141.5&#8202;N/mm. v7 has been on the robot since 2026-10-25
(<code>docs/experiments/20261025-ur5_calibration_mount_v7/</code>).</p>"""),
    ("open", "Open questions",
     r"""<ol>
<li><b>The softness at a small contact.</b> Under the sphere the presses travel 0.33&#8202;mm plus 0.20&#8202;mm/N
beyond the FE pad&#8217;s indentation; under the rod, 0.07&#8202;mm plus 0.10&#8202;mm/N. At most 0.10&#8202;mm/N lies in
series with the pad: in the DIP link&#8217;s clamp, the mount, the HEX-E, the arm or the indenter&#8217;s stand. At least
0.10&#8202;mm/N arises at the sphere&#8217;s contact, where the FE pad is stiffer than the print. The cause in the print
has not been found.</li>
<li><b>The registration along the finger.</b> Sphere2&#8217;s own registration and a joint one with the rod&#8217;s and
sph3t&#8217;s touch-offs fit them equally well. They place sphere2&#8217;s contacts a median 1.09&#8202;mm apart, 0.94&#8202;mm
of it along the finger. Every touch-off met the indenter at its top, with a vertical contact normal, so none of them sees
a horizontal shift of the indenter; that place rests on the homing seating. Twelve touch-offs aimed 3&#8202;mm beside the
sphere&#8217;s top meet curved parts of the pad off the top. In the simulator, with the seating 0.6&#8202;mm off, they cut
the contact error at the presses from 0.23 to 0.17&#8202;mm and from 1.36 to 0.04&#8202;mm under two sets of tool errors.
On the checkerboard tip they moved the fitted sphere centre 1.96&#8202;mm and turned the pad&#8217;s frame 1.93&#176;. With
that registration the grid presses&#8217; indentation beyond the FE pad fits within 108&#8202;&#181;m RMS, against
142&#8202;&#181;m before, and the board&#8217;s point lies 0.72&#8202;mm from the registered one, against 1.54&#8202;mm.
Sphere2 and sph3t were registered with plain touch-offs only.</li>
<li><b>Light presses.</b> With the v1 magnets the untrained estimator places contacts below 1&#8202;N about 10&#8202;mm away.
With five N52 magnets it places those under 1.5&#8202;N within 2.0&#8202;mm. The step of about
7&#8202;&#181;T that each press leaves on the rest is the size of a 0.3&#8202;N press. On the v1 tips learned models give
1.1 to 2.5&#8202;mm there. Under 1&#8202;N, contacts placed by the direction GP are off by 0.59&#8202;mm with four N52
magnets and 0.75&#8202;mm with five. On the checkerboard tip the estimator&#8217;s own position sigma flags most of these misses: 18&#8202;mm at
0.25&#8202;N, 3&#8202;mm at 1&#8202;N. A gate at 2&#8202;mm keeps 95&#8211;100&#8202;% of the holds at 2&#8202;N and over and
2&#8211;13&#8202;% below, where a few misses still pass.</li>
<li><b>The force reference.</b> The HEX-E&#8217;s Fz gain is 0.78&#8211;0.87 of its other axes&#8217;, and its link drops
out; an ATI Mini45, once calibrated, is to replace it. The FE takes Siraya 85A at 12&#8202;MPa from a NinjaFlex datasheet. Through the
board&#8217;s force, sphere2&#8217;s session puts the printed pad within about 22&#8202;% of it. Fitted on pad 05R4&#8217;s FE
basis to the presses with four N52 magnets and with five, the modulus is 16.9 and 16.3&#8202;MPa. The registered
indentation of those presses grows by 291&#8211;316 and 282&#8211;336&#8202;&#181;m/N. In the FE pad the slope is
135&#8211;137&#8202;&#181;m/N, so the printed pads sink 2.1&#8211;2.5 times as far per newton.</li>
<li><b>The magnets.</b> The v1 tips&#8217; magnets have 0.20&#8211;0.57 of an N52 magnet&#8217;s moment as fitted at rest, and
v2 ID 1&#8217;s five N52 magnets 0.98. The models predict 0.028&#8211;0.031&#8202;N with N52, about 0.039&#8202;N at the measured
noise. On the UR5 the noise-equivalent force was 63&#8202;mN with four magnets and 35&#8202;mN with five. That is the
ratio of 1.8 between the readout model&#8217;s 0.051 and 0.029&#8202;N.
With the direction GP, the two tips&#8217; contacts were placed within 0.26 and 0.28&#8202;mm. The choice between four and five magnets waits on
the owner. In the field model a touching neighbour
that faces the pad adds 38&#8202;&#181;T at the board with four magnets, against 513&#8202;&#181;T with five. Between two
four-magnet v3 tips on the UR5, 39 to 60&#8202;&#181;T were measured at contact.</li>
<li><b>Transfer of learned models.</b> The physical model transfers to a new pad through its rest calibration: on the
checkerboard tip it placed contacts within 2.06&#8202;mm without a press. The GP correction learned on sphere2 does not, and
raised that error to 2.48&#8202;mm. Fine-tuned with 20 of the new tip&#8217;s presses it gives 1.59&#8202;mm; the tip&#8217;s
own 47 presses give 0.42&#8211;0.71&#8202;mm. Scaled by the magnets&#8217; strength fitted at rest, the v1 tips&#8217; presses
place v2 ID 1&#8217;s contacts within 0.64&#8202;mm, against 2.52&#8202;mm unscaled.</li>
<li><b>Disturbances in the hand.</b> Between two facing v3 tips on the UR5 the held boards changed by 55 to
60&#8202;&#181;T at contact. The dipole model with each tip&#8217;s net moment leaves about 10&#8202;&#181;T of that
([magnetic crosstalk](exp:20261027-fingertip_crosstalk/20261027-magnetic_crosstalk_between_neighbouring_fingertips.html#holder)). Side by side and touching, a five-magnet neighbour adds 4,162&#8202;&#181;T in the
model, and its pose must be fitted from the boards. Steel touched to the face reads as a press of 9 to 17.5&#8202;N. The test against
fields from outside the pad removes most of those frames, but the first half second of a touch and fields of 40 to
90&#8202;&#181;T still show 1 to 9&#8202;N. The chips&#8217; temperature drift is measured,
0.65&#8211;0.70&#8202;% per &#176;C, and the correction from each chip&#8217;s temperature leaves 9 of 70&#8202;&#181;T over a
warm-up. The magnets&#8217; own coefficient, which follows the pad&#8217;s lagging temperature, is not separated yet.</li>
<li><b>Large presses, shear and torsion.</b> The FE is linear and frictionless; a neo-Hookean check lost 11&#8202;% of the
force at 350&#8202;&#181;m on the softest tip. The tip&#8217;s tangential compliance, 48&#8211;177&#8202;&#181;m/N, is in
neither simulator. On the checkerboard tip the estimator reports 1.08&#8202;N of shear where the HEX-E measures
0.25&#8202;N. A GP trained on the 3D force gets the shear within 0.12&#8202;N on held-out presses, but misses by 0.31&#8202;N at
the turns&#8217; new orientations. The twists&#8217; torsion, half the difference between outbound and return, measures a median
2.9&#8202;N&#8202;mm and up to 9.9&#8202;N&#8202;mm at 2.5&#8202;N; a point torque in the physical model does not recover it
from the board. On the UR5 the face&#8217;s normal lies across the
tool axis. The torsion then lies under a push moment of
about 230&#8202;N&#8202;mm on the HEX-E ([calibration protocol](exp:20261022-ur5_tip_calibration/20261022-ur5_tip_calibration_protocol.html)).</li>
<li><b>The simulators.</b> Calibrated to the FE, the pads and Drake agree on approach within 6&#8202;%. They end a turn
6.3&#176; apart, with contact forces 0.37&#8202;N apart. Whether builder v2 stays the default waits on the owner.</li>
</ol>"""),
]

FIGURES = {
    "section": dict(page="20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html", n=1,
                    source="finite elements of the printed tip",
                    alt="Section of the printed tip: walls, skins, gyroid core, magnet pocket and the pause.",
                    caption="Section through the first study&#8217;s tip as printed, back face on the bed. Two perimeter walls "
                            "and four solid layers surround the gyroid core and the magnet&#8217;s pocket. The pause before "
                            "the magnet&#8217;s layer and the magnetometer under the back face are marked. The 11/22 pad prints the same "
                            "structures upright."),
    "moduli": dict(image="docs/overviews/media/fingertip_moduli_plot.png", width=760,
                   alt="Homogenized moduli of printed gyroid against solid fraction.",
                   caption="Moduli of the printed gyroid against the core&#8217;s solid fraction, in the layer plane and along the "
                           "build axis. They come from the homogenized cells of the "
                           "<a href='../experiments/20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html'>"
                           "finite-element study</a> (filled: 0.1&#8202;mm voxels; open: 0.05&#8202;mm, the cells the tips "
                           "use). The plot also shows the fits \\(0.43\\,\\phi^{1.40}\\) and "
                           "\\(0.60\\,\\phi^{1.29}\\), the ideal sheet gyroid, eFlesh&#8217;s cut cells and the "
                           "Gibson&#8211;Ashby limits. Drawn by <code>scripts/overview_fingertip_figures.py</code>."),
    # film labels go to ffmpeg drawtext, which reads a bare % as an expansion and then draws no label: escape it as \%
    "film_g20": dict(film=[[(FILM_G20, "95A TPU, gyroid 20 \\%, whole tip")]], page="20261009-tpu_tip_fem/"
                     "20261009-printed_tpu_fingertip_finite_elements.html", source="finite elements of the printed tip",
                     caption="The first study&#8217;s whole tip at 20&#8202;% gyroid under the 12.5&#8202;mm shaft: the section "
                             "through the contact point, displacement magnified 15 times and coloured by its size, beside the "
                             "pressure on the palmar face, as the approach grows. The walls take the edges of the patch."),
    "crosstalk": dict(image="docs/experiments/20261027-fingertip_crosstalk/media/20261027-crosstalk_budget.png", width=900,
                      page="20261027-fingertip_crosstalk/20261027-magnetic_crosstalk_between_neighbouring_fingertips.html", n=1,
                      source="magnetic crosstalk",
                      alt="A neighbouring fingertip's field at the board against the gap, facing and side by side, before and "
                          "after subtraction, with a 0.5 N press for scale.",
                      caption="Change at v2 ID 1&#8217;s board caused by a neighbouring fingertip, in its field model: (a) pads "
                              "facing across a gap W, (b) side by side with a gap g between the flanks. Solid: the "
                              "neighbour&#8217;s field. Dashed: what is left after subtracting the field predicted at a pose "
                              "0.3&#8202;mm and 0.3&#176; off. Blue: five N52 magnets; orange: four. Dashed black: a 0.5&#8202;N "
                              "face press; grey band: presses of 0.2&#8211;3&#8202;N."),
    "board": dict(page="20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html", n=2, source="magnetometer readout",
                  alt="The five-chip board under the tip, from the back face and in section.",
                  caption="The five-chip board under a tip: the chips U1&#8211;U5 under the back face in a plus 7&#8202;mm "
                          "apart, and the sensing gap in section."),
    "surface": dict(image="docs/experiments/20261020-fingertip_contact_estimation/media/20261020-surface_uncertainty.png",
                    width=828, alt="Press 1-sigma over the pad's surface, two magnet sets, three views.",
                    caption="Model 1-sigma of the pressing force for 1&#8202;N along the normal at each pad point, from 50&#8202;ms "
                            "of samples. The views are from the palm and from both flanks. Two magnet sets are shown: the magnets "
                            "in the pad today and N52 magnets all one way (log scale). Red circles: magnets; white squares: chips. The mount end "
                            "and the flanks&#8217; lowest millimetres respond least. From the "
                            "<a href='../experiments/20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html'>"
                            "contact estimation</a>, Figure&#160;3."),
    "viewer": dict(film=[[(FILM_VIEW, "Browser viewer, five synthetic presses, magnets as found")]],
                   page="20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html",
                   source="contact estimation",
                   caption="The browser viewer replays five presses that the model made with the measured noise. Three presses are on the palmar "
                           "face, the +y flank and the face&#8217;s edge on the &#8722;y side, behind the pad in this view. "
                           "The last two press the tip end and push with shear along the finger. Arrow: the estimated force; ring: the 1-sigma "
                           "position ellipse; colour: the pads&#8217; pressure; deformation from the FE, drawn 20 times."),
    "ur5_film": dict(film=[[(FILM_UR5, "Overhead camera over sphere2, 21:13 to 21:19")]],
                     page="20261022-ur5_tip_calibration/20261022-ur5_tip_calibration_lab_runs.html", source="lab runs", crf=32,
                     caption="The UR5 session sphere2 from the overhead camera, one frame every 2&#8202;s of its first "
                             "touch-offs, played at 12 frames a second. The pad touches the sphere on the right-hand "
                             "fixture; the left-hand fixture holds the rod."),
    "board_hexe": dict(image="docs/experiments/20261022-ur5_tip_calibration/media/20261022-sphere2_board_against_hexe.png",
                       width=760, page="20261022-ur5_tip_calibration/20261022-ur5_tip_ground_truth_and_predictors.html", n=2,
                       source="UR5 ground truth and predictors",
                       alt="The board's force against the HEX-E's, and the estimator's location error by force level.",
                       caption="The board against the HEX-E in sphere2. (a) Normal force of the untrained estimator against "
                               "the HEX-E&#8217;s, by region; the line is equality. (b) The estimator&#8217;s location error "
                               "by force level; the bars are medians and the dashed line marks 2&#8202;mm."),
    "real_presses": dict(film=[[(FILM_REAL, "Second fingertip, six presses at the face centre", "1100:622:0:38")]],
                         page="20261023-second_fingertip_transfer/20261023-second_fingertip_transfer.html",
                         source="second fingertip", crf=32, poster_t=0.09,
                         caption="The browser viewer replays the second fingertip&#8217;s six presses at the face centre, "
                                 "recorded with the fingertip fixed to the desk. The viewer calibrated on the "
                                 "recording&#8217;s first 2&#8202;s at rest and found the pad&#8217;s reversed magnets "
                                 "itself. Arrow: the estimated force; colour: the sphere pads&#8217; pressure."),
    "checkerboard": dict(image="docs/experiments/20261022-ur5_tip_calibration/media/20261024-cb1t_fine_tuning.png",
                         width=760, page="20261022-ur5_tip_calibration/20261024-checkerboard_fingertip_on_the_ur5.html", n=3,
                         source="checkerboard fingertip on the UR5",
                         alt="Location error on the checkerboard tip against its training presses, with and without sphere2's.",
                         caption="The checkerboard tip&#8217;s held-out location error against the number of its own "
                                 "training presses: trained on sphere2&#8217;s presses and the tip&#8217;s (solid) and on "
                                 "the tip&#8217;s alone (dashed). The dotted line is the physical model rebuilt from the "
                                 "rest fit, with no press."),
    "hand": dict(image="docs/experiments/20261020-fingertip_contact_estimation/media/20261020-presses_by_hand.png",
                 width=760, alt="The owner's first presses by hand: reading change, estimate, and the offset left by the pad.",
                 caption="The owner&#8217;s first presses by hand (2026-10-20 19:39). The top panel shows the reading change "
                         "from the first rest, with the rests in grey, and the middle one the estimator&#8217;s force. The "
                         "bottom panel compares the last rest&#8217;s offset per chip axis with a rigid motion of the whole "
                         "pad fitted to it. From the "
                         "<a href='../experiments/20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html'>"
                         "contact estimation</a>, Figure&#160;5."),
}

FUTURE = [
    'Compliance in series (QUEUE.md R16): put a dial indicator on the sphere&#8217;s block. Press a rigid part of the tool, the HEX-E&#8217;s adapter or the fork, on the sphere at 1 to 5&#8202;N. Travel of the block puts the compliance on the indenter&#8217;s side. Travel beyond the HEX-E&#8217;s data sheet with a still block puts it in the arm or the mount. Travel near the data sheet&#8217;s leaves the DIP link&#8217;s clamp for the series part, at most 0.10&#8202;mm/N.',
    'Softness at the sphere: press the FE pad&#8217;s printed twin with a 5&#8202;mm sphere on a bench under a dial indicator, the pad clamped on a rigid block, at 0.5 to 5&#8202;N. Travel near the sphere&#8217;s local part (UR5 results page, Section 8) confirms that the FE pad is too stiff under a sphere. Travel near the FE pad&#8217;s own places that part in the robot&#8217;s tool.',
    'Series compliance on mount v7: fit the registered indentation under the rod against the force, by region, as <code>depth</code> in <code>scripts/ur5_tip_calibration_fe_calibration.py</code> fits it under the sphere. v2 ID 1 ran its rod presses on v7, and v3 ID 1 is queued for the same. On mount v3 rod1 travelled 0.07&#8202;mm plus 0.10&#8202;mm/N beyond the FE pad. A slope that stays at 0.10&#8202;mm/N on v7 rules out the link&#8217;s turn in v3&#8217;s fork as the series part.',
    'Checkerboard tip&#8217;s poles: check each magnet&#8217;s pole at the palm face with a compass. The rest fit predicts north toward the palm at the bottom-left and top-right magnets, seen from the palm with the tip end up, and south at the other three. Any other result means wrong chip frames in the board&#8217;s calibration.',
    'ATI Mini45 (the ATI plan&#8217;s QUEUE.md, A1): run the box check, print the stack and run the 39 stations with the 100&#8202;g weight in its printed holder, or with 0.5&#8202;kg. Then repeat a grid of presses against the ATI. A board force outside 0.78&#8211;1.05 of the ATI&#8217;s means that the HEX-E&#8217;s gain biased sphere2&#8217;s comparison.',
    'Untrained estimator under the rod: score the viewer&#8217;s estimator on rod1&#8217;s 279 holds, as Table 3 of the UR5 results page scores it on sphere2&#8217;s. A location error well above the sphere&#8217;s 1.6&#8211;2.6&#8202;mm means that the estimator&#8217;s point-load basis misses line contacts.',
    'Second fingertip with known points and forces: clamp it in a printed jig with sphere seats at the reference point and 4&#8202;mm from it toward &#177;y and &#177;z. Press through a kitchen scale at about 1, 2 and 4&#8202;N. Run <code>scripts/fingertip_second_tip_transfer.py --landmarks</code> on the stretch. A GP correction whose median location error exceeds the physical model&#8217;s by more than 0.9&#8202;mm does not transfer across pole patterns.',
    'N52 magnets: recalibrate in the viewer (key c). A strength of 0.8&#8211;1.2, the poles as inserted and a rest residual below 0.43&#8202;mT confirm the field model. Then repeat the jig presses or a short UR5 grid; the simulation predicts 1.5&#8202;mm for the rebuilt physical model and 0.7&#8202;mm with the GP correction.',
    'Torsion on the HEX-E&#8217;s axis: twist the pad at 5&#8202;N on a fixture that puts the face&#8217;s normal along the tool axis. The torsion is half the difference between the outbound and return holds. A torque outside the sphere pads&#8217; friction torque at the measured friction coefficient falsifies the pads&#8217; torsion.',
    'Sensing rows on the probes&#8217; gap rule: <code>run_sense</code> in <code>scripts/v2_tip_fe_checks.py</code> takes its gaps from <code>tpu_tip_magnet.gaps</code>. That function moves a face node onto the analytic surface only within half a grid cell of it. It gives 150&#8202;&#181;m against the probe rows&#8217; 185 for the same pad at 1&#8202;N. Rerun <code>--sense b_v2</code> with every face node on the surface; a force resolution that moves by more than 5&#8202;% means the DIP page&#8217;s resolutions need the same rerun.',
    'Facing finger: record the board with a second tip&#8217;s magnets at 20&#8211;40&#8202;mm and compare with the dipole prediction before subtracting it from the joint angles.',
    'Steel at known distances and forces: touch the 100&#8202;g mass and the ruler through printed spacers of 0.5 to 10&#8202;mm, then set the mass on the face (0.98&#8202;N). Run <code>scripts/fingertip_foreign_field.py touches</code> on each stretch. A spacer reading over 100&#8202;&#181;T with &#923; under 25 means the test misses steel. A joint fit of a contact and the far field within 0.3&#8202;N of 0.98&#8202;N would measure presses made with steel.',
    'Temperature: with the pad at rest and the viewer logging, hold a warm hand around the finger for 60&#8202;s and take it away. Run <code>scripts/fingertip_contact_estimation_study.py thermal</code> on that stretch. No contact should show while the chips warm by 2&#8211;3&#8202;&#176;C. A drift that grows with the temperature after the correction means the magnets need a coefficient of their own. Firmware with TCMP_EN 1 is the alternative, to be tested the same way.',
    'Fingertips beside each other on the hand (owner, 2026-10-25): once every fingertip has its UR5 calibration, mount them on the SR2 hand. Record the boards through a wide range of finger motions, with the joint angles. Fit each fingertip&#8217;s field at its neighbours&#8217; chips as a function of those angles. Laid against each other on the desk, three fingertips moved each other&#8217;s rest by 210 to 991&#8202;&#181;T. With the re-zero, no contact was shown at rest in 39 minutes (<code>src/morphohand/driver/reskin5x/README.md</code>). The fitted model fails if, during motion, it leaves a contact shown on a fingertip that nothing touches.',
]

PAGES = [
    ("Method", "20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html",
     "Finite elements of the printed structure: classes, homogenized gyroid, contact, magnet depth, print plan"),
    ("Method", "20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html",
     "Five-chip board, force noise and resolution, disturbances, one or four magnets"),
    ("Method", "20261018-fe_contact_calibration/20261018-fe_contact_calibration.html",
     "Drake&#8217;s field and the pads calibrated to the FE of the four 85A prints"),
    ("Method", "20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html",
     "Rest calibration, FE surface basis, whole-surface estimator, the first presses by hand, the browser viewer"),
    ("Method", "20261023-foreign_field_rejection/20261023-foreign_field_rejection.html",
     "Steel near the pad: the magnets&#8217; images, a test against fields from outside the pad, the viewer&#8217;s toggle"),
    ("Design", "20261019-v2_tip_fe_checks/20261019-v2_fingertip_stiffness_checks.html",
     "The v2 tips and b&#8217;s pad: print deviations, magnet count and depth, cover walls, resolution"),
    ("Design", "20261014-tip_codesign/20261014-sphere_pad_accuracy_and_fingertip_codesign.html",
     "Sphere-pad accuracy across tip shapes; fillets and recipes screened on the hand&#8217;s tasks"),
    ("Design", "20261015-curved_tips/20261015-capsule_and_ellipsoid_fingertips.html",
     "Capsule and ellipsoid tips on the tasks, stiffness maps against the FE"),
    ("Design", "20261013-eflesh_printability/20261013-eflesh_lattice_printability.html",
     "Printability, as-printed stiffness and resolution of eFlesh lattices"),
    ("Design", "20261021-spherical_magnets/20261021-spherical_and_disc_magnets.html",
     "Spherical against disc N52 magnets in pad B00"),
    ("Calibration", "20261022-ur5_tip_calibration/20261022-ur5_tip_calibration_protocol.html",
     "Protocol and program of the UR5 calibration with the HEX-E: tool stack, registration, presses, a simulated session"),
    ("Calibration", "20261022-ur5_tip_calibration/20261022-ur5_tip_calibration_lab_runs.html",
     "The lab&#8217;s runs of 2026-10-22: faults, checks, touch-offs and presses of sphere2"),
    ("Calibration", "20261022-ur5_tip_calibration/20261022-ur5_tip_ground_truth_and_predictors.html",
     "Sphere2 against the HEX-E: pad compliance, board force, estimator, GP and MLP, transfer in simulation"),
    ("Calibration", "20261022-ur5_tip_calibration/20261023-ur5_tip_board_drift.html",
     "The board&#8217;s zero over sphere2: Earth&#8217;s field, the steps a press leaves, learned zeros"),
    ("Calibration", "20261023-second_fingertip_transfer/20261023-second_fingertip_transfer.html",
     "Sphere2&#8217;s calibration on a second fingertip with two magnets reversed"),
    ("Calibration", "20261022-ur5_tip_calibration/20261024-checkerboard_fingertip_on_the_ur5.html",
     "The first checkerboard fingertip on the UR5: pole pattern, offset touch-offs, physical model, sphere2&#8217;s predictors"),
    ("Calibration", "20261027-fingertip_crosstalk/20261027-magnetic_crosstalk_between_neighbouring_fingertips.html",
     "A neighbouring fingertip&#8217;s field at the board, measured between v3 tips on the UR5, and its subtraction from the joint angles and from both boards"),
    ("Calibration", "20261023-ati_ft_calibration/20261023-ati_ft_calibration_plan.html",
     "Calibration of the lab&#8217;s ATI Mini45 with a weighed mass, to replace the HEX-E"),
    ("Hardware", "20261018-dip_link_board_mount/20261018-dip_link_magnetometer_board_mount.html",
     "Board mount in the DIP link: concepts a, b and c, cable route, print projects"),
    ("Hardware", "20261020-ur5_calibration_mount_v2/20261020-dip_link_b_v3_and_ur5_mount_v2.html",
     "DIP link b v3 and the UR5e calibration mount v2"),
    ("Hardware", "20261021-dip_link_b_v4/20261021-dip_link_b_v4_and_wall_heights.html",
     "DIP link b v4, retaining-wall heights and the snug covers"),
    ("Hardware", "20261018-ur5_calibration_mount/20261018-ur5_calibration_mount.html",
     "The first UR5e calibration mount: adapter, fork and clamp plate"),
    ("Hardware", "20261021-ur5_calibration_mount_v3/20261021-ur5_calibration_mount_v3.html",
     "Mount v3 with captive M6 &#215; 20 screws and heat-set inserts"),
    ("Hardware", "20261023-ur5_calibration_mount_v4/20261023-ur5_calibration_mount_v4.html",
     "Mount v4: a stop plate with set screws behind the DIP link&#8217;s tongue"),
    ("Hardware", "20261023-ur5_calibration_mount_v5/20261023-ur5_calibration_mount_v5.html",
     "Mount v5: a PLA wedge in place of v4&#8217;s set screws"),
    ("Hardware", "20261018-fingertip_print_procedure/20261018-siraya85a_fingertip_print_procedure.html",
     "Print order, pauses, seat coupon and records for the 85A tips"),
    ("Hardware", "20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html",
     "Requests, jobs, prints and decisions waiting on the owner"),
]
