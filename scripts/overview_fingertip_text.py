"""Text of the topic overview docs/overviews/printed_fingertips.html (builder: scripts/topic_overview_page.py).

Reader: a collaborator, or the owner preparing a meeting, who needs the method, the reason for each choice and the open
questions of the printed fingertip, without reading the fifteen result pages (owner, 2026-10-20: "understand what tools,
what's the underlying math behind: FEA over the gyroid infill, the impacts our various modifications and magnet
embedding does, how we're able to map the magnetometer readings to the surface via FEA"; "go into my meetings tomorrow
and have a clear understanding of what i need to communicate ... the methodology and rationale, and the remaining
questions"). Takeaways: (1) one chain of models runs from the print recipe to the chip readings and back to a contact
estimate, and it has been checked only against other models; (2) what each design choice does to stiffness and force
resolution, and why concept b's pad with five magnets; (3) what is still open: a force reference, the pad's seat, the
N52 magnets. Budget: about 2,600 words of prose. Sources: the pages in PAGES and the fact sheets of 2026-10-20 (FE and
printing; readout and hardware), each number as its page gives it.
"""
import os

KEY = "fingertip"
SHORT_TITLE = "Printed fingertips and force sensing"
TITLE = "Finite elements, magnetic readout and contact estimation of the printed TPU fingertips"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILM_G20 = "docs/experiments/20261009-tpu_tip_fem/media/20261014-film_whole_g20x.mp4"
FILM_VIEW = "docs/experiments/20261020-fingertip_contact_estimation/media/20261020-viewer_surface_presses.mp4"

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
<text x="421" y="36" text-anchor="middle" font-size="15" font-weight="bold">Gyroid homogenized</text>
<text x="421" y="54" text-anchor="middle" font-size="13.5">periodic slicer cell &#8594;</text>
<text x="421" y="72" text-anchor="middle" font-size="13.5">6&#215;6 stiffness tensor</text>
<rect x="516" y="14" width="150" height="70" fill="#fff" stroke="#000"/>
<text x="591" y="36" text-anchor="middle" font-size="15" font-weight="bold">Finite elements</text>
<text x="591" y="54" text-anchor="middle" font-size="13.5">hexahedra, K u = f,</text>
<text x="591" y="72" text-anchor="middle" font-size="13.5">cut face fixed</text>
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
<text x="591" y="150" text-anchor="middle" font-size="15" font-weight="bold">Sensitivity S(p)</text>
<text x="591" y="168" text-anchor="middle" font-size="13.5">readings per newton</text>
<text x="591" y="186" text-anchor="middle" font-size="13.5">at each surface point</text>
<rect x="686" y="128" width="136" height="70" fill="#fff" stroke="#000"/>
<text x="754" y="150" text-anchor="middle" font-size="15" font-weight="bold">Design check</text>
<text x="754" y="168" text-anchor="middle" font-size="13.5">force resolution</text>
<text x="754" y="186" text-anchor="middle" font-size="13.5">(Cram&#233;r&#8211;Rao)</text>
<rect x="176" y="244" width="150" height="70" fill="#fff" stroke="#000"/>
<text x="251" y="266" text-anchor="middle" font-size="15" font-weight="bold">Readings, live</text>
<text x="251" y="284" text-anchor="middle" font-size="13.5">15 components,</text>
<text x="251" y="302" text-anchor="middle" font-size="13.5">140&#8202;Hz, 50&#8202;ms window</text>
<rect x="516" y="244" width="150" height="70" fill="#fff" stroke="#000"/>
<text x="591" y="266" text-anchor="middle" font-size="15" font-weight="bold">Estimator</text>
<text x="591" y="284" text-anchor="middle" font-size="13.5">point, force, 1-sigma</text>
<text x="591" y="302" text-anchor="middle" font-size="13.5">(marginal likelihood)</text>
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
    "The printed fingertip is a pad of Siraya Tech 85A TPU, two walls and four solid skins around a gyroid core, on the "
    "DIP link over ReSkin&#8217;s five-chip MLX90393 board, with five 3&#8202;&#215;&#8202;2&#8202;mm magnets 7.5&#8202;mm "
    "under its apex. One chain of models follows it from the slicer&#8217;s rules to the chip readings: finite elements of "
    "the printed structure with the gyroid homogenized, the magnets&#8217; motion under any surface load, and a field "
    "model of the five chips. Forward, the chain chose the design: concept b&#8217;s pad resolves 0.028&#8211;0.031&#8202;N "
    "per sample at its contact point with N52 magnets. Inverted, it estimates contact point and force anywhere on the pad "
    "from the 15 readings, 0.18&#8202;mm and 0.03&#8202;N median on simulated presses. No printed pad has yet been pressed "
    "against a force reference, and in the first presses by hand the pad moved in its seat.")

TERMS = [
    ("Approach", "Travel of a rigid indenter into the tip from first touch under a given normal force, in &#181;m; the "
     "reference indenter is the 12.5&#8202;mm screwdriver shaft across the finger."),
    ("Wall, skin, core", "The three printed structures: walls are the two 0.45&#8202;mm perimeters of every layer (0.9&#8202;mm, "
     "also around each magnet pocket), skins the four solid 0.2&#8202;mm layers under every top and bottom surface "
     "(0.8&#8202;mm), the core the gyroid infill."),
    ("Infill setting, solid fraction", "The slicer&#8217;s gyroid density setting \\(\\rho\\) (30 or 50&#8202;%) and the "
     "volume fraction of TPU it prints, which differ because successive layers&#8217; beads overlap only in part."),
    ("Homogenized tensor", "The 6&#215;6 elasticity tensor of a periodic cell of printed gyroid, per unit modulus of solid "
     "TPU \\(E_s\\): the stiffness of the core seen at the scale of the tip."),
    ("Contact modulus \\(E^*\\)", "The modulus of the sphere pads&#8217; pressure law that best reproduces the FE approach "
     "at 0.5, 1 and 3&#8202;N, in MPa; the RL scenes use 10&#8202;MPa."),
    ("Magnet travel", "Rigid motion of a magnet per newton of contact force, from the FE, in &#181;m/N."),
    ("Sensing plane", "The plane of the five chips&#8217; Hall elements; on concept b it lies 0.42&#8202;mm below the "
     "pad&#8217;s back face."),
    ("Force resolution", "The Cram&#233;r&#8211;Rao bound on the standard deviation of the pressing force from one "
     "conversion of the five chips (141&#8202;Hz) at a known contact point, with Earth&#8217;s field unknown, in N: the "
     "spread of the best unbiased estimate when the models are exact."),
    ("Sphere pads", "The simulations&#8217; contact model of the tip: 0.75&#8202;mm spheres on a 1&#8202;mm lattice over its "
     "surface, each with stiffness \\(K_i=E\\,a_i/h_i\\) (builder v2: \\(h_i\\) its depth below the tip&#8217;s volume "
     "centroid)."),
    ("Rest calibration", "The fit of the field model to the reading with the pad untouched: which chip address sits at "
     "which position, the chips&#8217; axes, every magnet&#8217;s pole, the magnets&#8217; strength and the pad&#8217;s "
     "offset on the board."),
    ("Sensitivity \\(S(\\mathbf p)\\)", "The change of the 15 readings per newton of a patch load at surface point "
     "\\(\\mathbf p\\), along each axis of the tip frame (15&#215;3)."),
    ("Placed away", "A simulated press whose estimated point lies more than 1.5&#8202;mm from the true one."),
    ("Tip frame", "\\(x\\) along the pad&#8217;s normal toward the palmar face, \\(y\\) across the finger, \\(z\\) from the "
     "mount toward the tip end, in mm."),
]

SECTIONS = [
    ("chain", "Components of the fingertip and its models",
     r"""<p>The fingertip is concept b of the DIP-link study, chosen by the owner after the first prints of 2026-10-20: the
front 13.45 of the 17.0&#8202;mm of the 11/22 half-ellipsoid tip of the simulations, printed as a pad that slides into a PLA
cover on the DIP link over the ReSkin 5X board, whose chips lie 0.42&#8202;mm below the pad&#8217;s back face
([DIP link b v3 and mount v2](exp:20261020-ur5_calibration_mount_v2/20261020-dip_link_b_v3_and_ur5_mount_v2.html)). Five
N52 magnets, a centre one and four corners 7.4&#8202;mm apart, sit with their tops 7.5&#8202;mm under the apex, the
shallowest depth that keeps 1.8&#8202;mm of TPU around every pocket. A press moves the magnets by a few micrometres; the
chips read the change of their field.</p>
""" + CHAIN_SVG + r"""
<p>Each box is one model with its own check. The finite elements resolve what the printer makes, walls, skins and
gyroid, because the walls carry the face&#8217;s edges and set the patch the pads must reproduce. The field model
represents each magnet as a cylinder of 72 dipoles, because one dipole is 5.5&#8202;% high at the chips&#8217;
distance. The rest calibration makes the chain usable on a pad as built: it finds the chips&#8217; layout and the
magnets&#8217; poles and strength from 2&#8202;s of untouched reading. Forward, the chain ranks designs by their force
resolution before anything is printed; inverted, it turns readings into a contact point and a force without a robot.</p>"""),
    ("print", "The printed pad",
     r"""<p>The pads print on the Prusa XL&#8217;s tool 2 or the P1S&#8217;s external spool in Siraya Tech 85A at 240&#8202;&#176;C on
a 35&#8202;&#176;C bed, upright in 0.1&#8202;mm layers: two perimeters outside and around each pocket, four top and four
bottom layers, gyroid at 30 or 50&#8202;%. The print pauses at 9.2 and 16.2&#8202;mm of height to take the magnets, north
toward the palmar face, into pockets whose 2.9&#8202;mm seats hold them by press fit (the owner&#8217;s choice from the
seat coupon, 2026-10-18). The recipe stays the same on every printer, so one FE describes every pad
([print procedure](exp:20261018-fingertip_print_procedure/20261018-siraya85a_fingertip_print_procedure.html)). The
pads weigh 0.6&#8202;% more TPU than the FE assumes, and concept a&#8217;s tips 3.0&#8202;% less, about 7&#8202;% softer
([DIP link b v3 and mount v2](exp:20261020-ur5_calibration_mount_v2/20261020-dip_link_b_v3_and_ur5_mount_v2.html)).</p>
{{FIG section}}
<p>85A at 30&#8202;% gyroid was chosen for the magnets&#8217; travel: the softer the pad, the more field change per
newton, until the approach outgrows what 1&#8202;mm pads represent. The rule from the co-design study is at least
12&#8202;&#181;m/N of travel for 0.1&#8202;N
([fingertip co-design](exp:20261014-tip_codesign/20261014-sphere_pad_accuracy_and_fingertip_codesign.html)); the
95A tips of the first studies gave 11.7&#8202;&#181;m/N at best. eFlesh lattices travel 250&#8202;&#181;m/N open, as
eFlesh prints them, but sink 613&#8202;&#181;m at 1&#8202;N and print with unsupported islands; closed by a skin on every
face they sink 81&#8202;&#181;m and give 0.075&#8202;N, the gyroid&#8217;s resolution
([eFlesh printability](exp:20261013-eflesh_printability/20261013-eflesh_lattice_printability.html)).</p>"""),
    ("fe", "Finite elements of the printed structure",
     r"""<p><b>Materials.</b> The tip&#8217;s shape is voxelized at 0.05&#8202;mm. In each layer plane, TPU within 0.9&#8202;mm
of the outline is wall; TPU within 0.8&#8202;mm of an exposed surface along the build axis is skin; the rest is core, and
the magnets are a fourth class, with walls and skins lining their pockets as the slicer prints them. The classes are
sampled onto a graded hexahedral grid, 0.2&#8211;0.25&#8202;mm under the face and 0.8&#8211;1.1&#8202;mm deep inside
(about 120&#8211;220 thousand degrees of freedom). Walls and skins are solid TPU, isotropic with Poisson 0.45: 30&#8202;MPa
for 95A and 12&#8202;MPa for 85A, from datasheets (NinjaFlex standing in for Siraya&#8217;s unmeasured 85A). The magnets
are NdFeB at 160&#8202;GPa, bonded
([finite elements of the printed tip](exp:20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html)).</p>
<p><b>Gyroid core.</b> The core is homogenized from the gyroid as the slicer prints it. In each 0.2&#8202;mm layer the
toolpath follows the zero set of the gyroid function at mid-layer with a 0.45&#8202;mm bead, and successive layers bond
only where their beads overlap:</p>
\[g(\mathbf x)=\sin X\cos Y+\sin Y\cos Z+\sin Z\cos X,\qquad \mathbf X=\frac{2.44\,\rho}{s}\,\mathbf x,\quad s=0.407\ \mathrm{mm}\]
<p>A periodic cell of these voxels takes six unit macroscopic strains \(\boldsymbol\varepsilon_i\); each load case solves
for the periodic fluctuation \(\mathbf u_i\), and the effective tensor is the cell&#8217;s mean stress:</p>
\[K\,\mathbf u_i=-\mathbf f_i,\qquad \mathbf f_i=\sum_e B_e^{\top}D\,\boldsymbol\varepsilon_i\,V_e,\qquad
C^{*}_{ij}=\frac{V_s\,D_{ij}+\mathbf f_i\cdot\mathbf u_j}{\lvert Y\rvert}\]
<p>The result is orthotropic, stiffer along the build axis. At the settings used, 30&#8202;% prints 32.6&#8202;% solid with
\(E_x=0.087\,E_s\) in the layer plane and \(E_z=0.135\,E_s\) along the build axis, and 50&#8202;% prints 55.0&#8202;% with
0.228 and 0.311; over 11&#8211;42&#8202;% solid \(E_x/E_s=0.43\,\phi^{1.40}\) and \(E_z/E_s=0.60\,\phi^{1.29}\) in the solid
fraction \(\phi\), between the bending and stretching limits of Gibson and Ashby.</p>
{{FIG moduli}}
<p><b>Tip model and contact.</b> The elements are eight-node hexahedra with Wilson&#8211;Taylor incompatible modes, so the
thin walls bend without shear locking (a 40-voxel cantilever comes out 2.4&#8202;% below beam theory, against 9.8&#8202;%
for plain hexahedra); each takes the tensor of its class, the core&#8217;s rotated into the tip frame.
Concept b&#8217;s pad is fixed at its cut face and the mount face. One sparse Cholesky factorization of \(K\) is reused
for every load. A rigid frictionless indenter (the shaft, a plate or a sphere) meets the face nodes, whose normal compliance
\(W\) comes from columns of \(K^{-1}\); at approach \(\delta\) the nodal forces \(\mathbf p\) solve</p>
\[\mathbf p\ge0,\qquad \mathbf g_0-\delta\,\mathbf 1+W\mathbf p\ge0,\qquad \mathbf p^{\top}\big(\mathbf g_0-\delta\,\mathbf 1+W\mathbf p\big)=0,\]
<p>and \(\delta\) is bisected until \(\sum p_i\) equals the load. A magnet&#8217;s motion is the rigid fit of its nodes&#8217;
displacements. Each gap \(g_0\) is measured from the tip&#8217;s analytic surface; on the curved tips the grid&#8217;s face
nodes lie a median 0.10&#8211;0.13&#8202;mm from that surface (95th percentile 0.43&#8211;0.49&#8202;mm).</p>
<p><b>Solver.</b> The solver is our own Python (<code>scripts/tpu_tip_fem_core.py</code>; SciPy, CHOLMOD and PyAMG on
the CPU, PyTorch on the GPU). Its input is voxels, the slicer&#8217;s material classes at 0.05&#8202;mm or every gyroid
bead at 0.1&#8202;mm, and on a voxel grid all elements of one size share one element matrix. The model resolving every
bead, 1.7&#8211;2.2 million elements and 5&#8211;8 million degrees of freedom, is then solved matrix-free on the GPU in
2&#8211;3&#8202;GB and 60&#8211;140&#8202;s per approach. The off-the-shelf solvers installed for the study run on the CPU
from tetrahedral meshes: FEniCSx took 3.3&#8211;5.7&#8202;GB for 0.45&#8211;0.68 million degrees of freedom of quadratic
tetrahedra, a rate that puts the resolved tip above 40&#8202;GB on a 30&#8202;GB workstation, and PolyFEM&#8217;s IPC
contact took 7&#8211;14&#8202;min for 57 thousand degrees of freedom. One element code also solves the periodic cells, the homogenized tip and
the resolved tip, so their comparison tests the homogenization alone. FEniCSx and PolyFEM remain as checks
([finite elements of the printed tip](exp:20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html),
Tables 7 and 8).</p>
<p><b>Checks.</b> On the 95A block of the first study the model is 1.9&#8202;% off Hertz under a 40&#8202;mm sphere, and a
model resolving every bead at 0.1&#8202;mm gives the homogenized tip&#8217;s approach within 2.6&#8202;% (44.8 against
46.0&#8202;&#181;m at 1&#8202;N). On the curved 85A tips the one check is grid refinement: a 0.2&#8202;mm contact grid
changes the capsule&#8217;s approach by at most 3.6&#8202;%, and the 11/22 tip&#8217;s by 0.1&#8202;% under the rod and
1.7&#8202;% RMS over the 18 calibration probes
([calibration to the finite elements](exp:20261018-fe_contact_calibration/20261018-fe_contact_calibration.html),
[v2 fingertip checks](exp:20261019-v2_tip_fe_checks/20261019-v2_fingertip_stiffness_checks.html)). No model resolving
the beads, no tetrahedral model and no G-code model of a curved or 85A tip has been solved. Solved from its own G-code,
the 95A tip of the first study was softer than the design model, 63 against 41&#8202;&#181;m at 1&#8202;N, because the
stacked beads overlap only in part; only an indentation of a printed tip decides between the two
([eFlesh printability](exp:20261013-eflesh_printability/20261013-eflesh_lattice_printability.html)).</p>
{{FIG film_g20}}"""),
    ("changes", "Effects of the design changes",
     r"""<p>Each change below was solved on the same models; approaches are under the 12.5&#8202;mm shaft at 1&#8202;N and
resolutions per sample at the contact point
([finite elements of the printed tip](exp:20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html),
[v2 fingertip checks](exp:20261019-v2_tip_fe_checks/20261019-v2_fingertip_stiffness_checks.html)).</p>
<div class="tw"><table class="index"><thead><tr><th>Change</th><th>Effect</th></tr></thead><tbody>
<tr><td>Infill, 85A</td><td>Solid 20.2&#8202;&#181;m (\(E^*\) 47.5&#8202;MPa); 50&#8202;% gyroid 43.3&#8202;&#181;m (14.4); 30&#8202;% 74.7&#8202;&#181;m (6.3) on the block; 95A at 20&#8202;% 46.0&#8202;&#181;m</td></tr>
<tr><td>Walls</td><td>Carry the face&#8217;s edges: 53&#8202;% of the load on the outer 1.2&#8202;mm of the 20&#8202;% tip, against 30&#8202;% for the pads&#8217; law; twist arm 3.35&#8202;mm against the pads&#8217; 2.72</td></tr>
<tr><td>Print deviations</td><td>Perimeters 10&#8202;% thinner +8.6&#8202;% approach; gyroid beads 10&#8202;% narrower +9.3&#8202;%; under-extrusion 5 or 10&#8202;% +8.7 or +19.6&#8202;%; skins 0.7 for 0.8&#8202;mm +0.05&#8202;%</td></tr>
<tr><td>Magnets as inclusions</td><td>Five 3&#8202;&#215;&#8202;2&#8202;mm at 7.5&#8202;mm stiffen the 11/22 tip by 3.4&#8202;%; one 4.76&#8202;mm magnet 1.5, 3 or 5&#8202;mm deep in the 95A block by 99, 31 or 15&#8202;%</td></tr>
<tr><td>Five or four magnets</td><td>In b&#8217;s pad at 7.5&#8202;mm: 0.0285&#8202;N with five, 0.0513 with four; five at 7.0 or 6.5&#8202;mm 0.0299 or 0.0308 (more travel, farther from the chips); at 8.5&#8202;mm the 41&#8202;mT field exceeds every gain</td></tr>
<tr><td>One perimeter per pocket</td><td>Five magnets 0.0287&#8202;N, four 0.0460; travel 5.23 to 6.04&#8202;&#181;m/N</td></tr>
<tr><td>Concept b against a</td><td>The cut pad of b: 2.5&#8202;% stiffer, 0.028 against 0.065&#8202;N, contact located to 0.34&#8202;&#215;&#8202;0.42 against 1.40&#8202;&#215;&#8202;1.31&#8202;mm, chips 3.5&#8202;mm closer</td></tr>
<tr><td>Cover walls on b&#8217;s flanks</td><td>Held both ways in the model (an upper bound): 3.0&#8202;mm walls &#8722;0.16&#8202;% approach and 0.030&#8202;N; v3&#8217;s 6&#8202;mm walls 0.0313&#8202;N; with the proposed tip cap 0.0317&#8202;N, with its side walls 0.0332&#8202;N</td></tr>
<tr><td>Shape</td><td>Of 24 printable fillet shapes none turns the tool better than the as-built block; curved tips cut the twist-onset arm to a third; in 85A at 30&#8202;% all shapes sink 251&#8211;262&#8202;&#181;m under a 2&#8202;mm sphere</td></tr>
</tbody></table></div>
<p>Two choices follow from the table: five magnets, because the centre magnet over U1 halves the resolution of four, and
concept b, because the cut pad puts the magnets 3.5&#8202;mm closer to the chips. The cover walls and the cap cost at most
16&#8202;% of the resolution, in the bound that holds the flanks both ways.</p>"""),
    ("readout", "Magnetic readout and force resolution",
     r"""<p>The ReSkin 5X board has five MLX90393 three-axis Hall magnetometers in a plus, U1 at the centre and U2&#8211;U5
7.00&#8202;mm from it; each chip senses 0.42&#8202;mm under its package top. A QT Py ESP32-S3 reads them over its STEMMA QT
port in 2.0&#8202;ms, 375 records a second, while each chip converts every 7.0&#8211;7.1&#8202;ms (140&#8211;143&#8202;Hz). The
firmware sets the gain at GAIN_SEL&#8202;1, RES&#8202;1 (1.20&#8202;&#181;T a count on \(x\) and \(y\), 1.94 on \(z\), a range of
38.4&#8202;mT), because b&#8217;s five magnets put 27.5&#8202;mT on the nearest chip and the most sensitive gain covers
4.8&#8202;mT. At rest the measured noise per sample is 1.54&#8211;1.76&#8202;&#181;T on \(x\) and \(y\) and 2.32&#8211;2.59&#8202;&#181;T on
\(z\), 1.4 times the datasheet model for this gain. Three fingers&#8217; boards answer at the same addresses and need an
I&#178;C multiplexer; a sweep of 15 chips takes 6.74&#8202;ms, inside one conversion
([magnetometer readout](exp:20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html),
<code>src/morphohand/driver/reskin5x/README.md</code>).</p>
{{FIG board}}
<p>The design studies rank layouts by force resolution. The FE gives each magnet&#8217;s translation and rotation per newton
of a patch load; the field model turns that into the change of the 15 readings, the sensitivity \(S\); and the noise
\(\Sigma\) bounds how well a force can be read. With a uniform background field \(\mathbf b\) (Earth&#8217;s) unknown, and
the contact point too when it is unknown,</p>
\[\mathbf r=S\,\mathbf f+U\mathbf b+\mathbf n,\qquad J=\big[\,S\ \ U\ \ \partial_yS\,\mathbf f\ \ \partial_zS\,\mathbf f\,\big],\qquad
\sigma_{f_n}=\sqrt{\big[(J^{\top}\Sigma^{-1}J)^{-1}\big]_{nn}},\]
<p>the Cram&#233;r&#8211;Rao bound on the pressing force from one sample. It is a lower bound: it assumes the models are exact.
On the first 95A tip one 4.76&#8202;mm magnet 5&#8202;mm deep resolved 0.044&#8202;N at a known point and 0.08&#8211;1.48&#8202;N over
the face when the point was unknown, which is why the layouts went to four magnets (0.14&#8202;N everywhere) and then
five. On the 11/22 tip concept a resolves 0.065&#8202;N and b 0.028&#8202;N, with the contact located to
0.34&#8202;&#215;&#8202;0.42&#8202;mm against a&#8217;s 1.40&#8202;&#215;&#8202;1.31&#8202;mm; at the measured noise b&#8217;s 0.028&#8202;N
becomes about 0.039&#8202;N ([v2 fingertip checks](exp:20261019-v2_tip_fe_checks/20261019-v2_fingertip_stiffness_checks.html)).</p>
<p>The disturbances are larger than the noise. On the first tip, 10&#8202;&#176;C of chip drift read as 0.47&#8202;N, and a facing
fingertip&#8217;s magnet moving from 25 to 30&#8202;mm as 5.16&#8202;N, 2.01&#8202;N with the field&#8217;s gradient estimated; the hand&#8217;s
joint angles give the other magnets&#8217; positions, so their field can be subtracted. Earth&#8217;s field, 1.71&#8202;N on U1 alone,
drops out of the five-chip estimate. The board must not move against the pad: on concept a with four magnets,
1&#8202;&#181;m of board motion reads as 0.160&#8202;N, which is why the board is clamped
([DIP link board mount](exp:20261018-dip_link_board_mount/20261018-dip_link_magnetometer_board_mount.html)).</p>"""),
    ("estimate", "Contact point and force from the readings",
     r"""<p>The estimator runs the chain backwards
([contact estimation](exp:20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html)). At rest
the 15 components are the magnets&#8217; field and Earth&#8217;s. With each chip&#8217;s rotation taken from the board&#8217;s
pick-and-place file and one common offset fitted, the rest calibration finds which address sits at which position, every
magnet&#8217;s pole, the magnets&#8217; strength and the pad&#8217;s offset on the board. On pad B00 as first assembled it fits at
0.43&#8202;mT RMS against fields of 1.9&#8211;7.0&#8202;mT, and finds magnets at 35&#8202;% of an N52&#8217;s moment with two
diagonal corners reversed, which the owner confirmed were not N52 and inserted at random. A press then changes the reading
by</p>
\[\Delta\mathbf b=S(\mathbf p)\,\mathbf f+N\mathbf q+\mathbf n,\qquad N=\big[\,U\ \ \mathbf b^{\mathrm{rest}}\,\big],\]
<p>where \(S(\mathbf p)\) comes from an FE basis of 1&#8202;N loads along \(x\), \(y\) and \(z\) at the 812 pad points over the
whole free surface (2,436 load cases, 101&#8202;s, 2.0&#8202;GB), interpolated between them by a thin-plate spline, and
\(\mathbf q\) holds a uniform field change and a strength change with Gaussian priors. With \(C\) the covariance of the
noise and the nuisance terms and a force prior \(\sigma_f=3\)&#8202;N, the force at each candidate point and the point
itself follow from</p>
\[\hat{\mathbf f}(\mathbf p)=\big(S^{\top}C^{-1}S+\sigma_f^{-2}I\big)^{-1}S^{\top}C^{-1}\Delta\mathbf b,\qquad
\hat{\mathbf p}=\arg\max_{\mathbf p}\ \Big[\Delta\mathbf b^{\top}C^{-1}S\,\hat{\mathbf f}-\ln\det\big(I+\sigma_f^{2}S^{\top}C^{-1}S\big)\Big],\]
<p>the log marginal likelihood of a contact at \(\mathbf p\), over 12,581 candidate points about 0.2&#8202;mm apart, with the
force pressing, inside a friction cone of 2 and below 25&#8202;N. The determinant charges each point for the patterns it
could explain: without it the best fit often went to a flank point where the reading barely responds, with forces of
100&#8202;N and more. The Fisher information gives each estimate&#8217;s 1-sigma; one estimate takes 1.2&#8202;ms.</p>
<p>On 600 simulated presses with N52 magnets and the 50&#8202;ms window, 2&#8202;% are placed more than 1.5&#8202;mm away; the rest
have median errors of 0.20&#8202;mm and 0.036&#8202;N on the face and 0.17&#8202;mm and 0.050&#8202;N on the flanks, and the errors
follow the predicted 1-sigma (ratio 1.10&#8211;1.16). With the magnets in the pad today, 15&#8202;% are placed away and the rest
have 0.39&#8202;mm and 0.08&#8202;N. Over 30&#8202;s without a press, one frame of 880 shows a contact.</p>
{{FIG surface}}
{{FIG viewer}}"""),
    ("pads", "The fingertip in the simulations",
     r"""<p>The simulations represent the tip by sphere pads, 0.75&#8202;mm spheres on a 1&#8202;mm lattice over its surface,
each with stiffness \(K_i=E\,a_i/h_i\); builder v2, the default since 2026-10-15, takes \(h_i\) as each pad&#8217;s depth
below the tip&#8217;s volume centroid. One modulus misses the FE&#8217;s approach by 20&#8211;28&#8202;%, because the FE&#8217;s
force grows as \(\delta^{1.07}\) and the law&#8217;s as \(\delta^{1.5}\): the walls and skins spread the load over the core.
A stiff local modulus behind a series spring, \(a(F)=F/k_s+a_{\mathrm{loc}}(F;E_{\mathrm{loc}})\) with
\(k_s=9.7\)&#8202;N/mm, fits the 11/22 tip within 5.2&#8202;% RMS approach in Drake&#8217;s hydroelastic field and
5.6&#8202;% in 0.5&#8202;mm pads, though no single spring fits the capsule
([calibration to the finite elements](exp:20261018-fe_contact_calibration/20261018-fe_contact_calibration.html)). With the
spring in both simulators, the same servo commands end the tool&#8217;s turn 6.3&#176; apart in place of 18.6&#176;, while
their contact forces still differ by 0.37&#8202;N against a 0.25&#8202;N bound
([v2 fingertip checks](exp:20261019-v2_tip_fe_checks/20261019-v2_fingertip_stiffness_checks.html)). The estimator
reports its forces at the same pads the simulations use, so a measured contact and a simulated one can be compared pad
by pad.</p>"""),
    ("hardware", "Hardware built and measured",
     r"""<p>The DIP link holds the board against the pad. Of three concepts, the window plate with a clamp (a), a cover that
clamps the board behind the pad on rails (b) and a board printed over (c), the design judge chose a on 2026-10-18 and the
owner took a and b to print; on 2026-10-20 the owner chose b (&#8220;concept B is the clear winner&#8221;), which also resolves
0.028 against 0.065&#8202;N. Version 2 keeps the board&#8217;s own Qwiic connector and lays the cable in a groove past the
slotted arm, with both links 9.75&#8202;mm longer; v3 adds 6&#8202;mm cover walls along the pad&#8217;s flanks, a column under the
pad and a cable leg open to the top
([DIP link board mount](exp:20261018-dip_link_board_mount/20261018-dip_link_magnetometer_board_mount.html),
[DIP link b v3 and mount v2](exp:20261020-ur5_calibration_mount_v2/20261020-dip_link_b_v3_and_ur5_mount_v2.html)).</p>
<p>In the prints of 2026-10-20 the rails at groove fit B00 slot together but pop off under a sideways push, and cover W00
keeps pad B00 in place. In the owner&#8217;s first presses by hand the pad rocked in its seat: after the fourth and fifth
presses the reading stayed 62 and 104&#8202;&#181;T from rest, which a rigid shift of the whole pad by 17&#8211;23&#8202;&#181;m with a
tilt of about 4&#8202;mrad reproduces (7&#8211;14&#8202;% residual, 62&#8202;% for random patterns), and which the estimator reads as
standing contacts of 11&#8211;12&#8202;N. The owner accepted a cap over the tip end the same evening: an end plate on the
pad&#8217;s flat end, ears under the two distal board screws and a 2&#8202;mm pin into the pad (pad B00P has its hole), with
6&#8202;mm side walls on the distal flanks in variant TW; in the FE bound the plate and pin cost 1&#8202;% of the resolution
and the side walls 6&#8202;%, and both caps print in 17&#8202;min
([DIP link board mount](exp:20261018-dip_link_board_mount/20261018-dip_link_magnetometer_board_mount.html)).</p>
{{FIG hand}}
<p>The UR5e calibration mount clamps the DIP link to the OnRobot HEX-E with the finger along the tool axis, so the normal
force falls on the sensor&#8217;s lateral axes (noise 0.035&#8202;N against 0.15&#8202;N axially); under 10&#8202;N the HEX-E deflects
0.49&#8202;mm, the link 0.010&#8211;0.026&#8202;mm and the mount 0.004&#8202;mm. The first version printed with a spigot the HEX-E
has no bore for and a key that snapped; v2 clocks the link by walls around its legs and homes on a drafted key on each
indenter, and is sliced but not all printed
([UR5e mount v1](exp:20261018-ur5_calibration_mount/20261018-ur5_calibration_mount.html)). The UR5e and the HEX-E are
connected to the lab&#8217;s computer, which this workstation cannot reach.</p>"""),
    ("open", "Open questions",
     r"""<ol>
<li><b>The force scale.</b> No printed pad has been pressed against a force reference. The FE takes Siraya 85A at
12&#8202;MPa from a NinjaFlex datasheet (95A datasheets span 26&#8211;67&#8202;MPa), every force scales with that modulus, and on
the first tip the G-code solve was half as stiff as the design model. A scale or load cell under a printed indenter
settles the scale; an indentation with the HEX-E separates the pad&#8217;s stiffness from the magnets&#8217; strength.</li>
<li><b>The seat.</b> The FE fixes the pad&#8217;s cut face, and the pad as assembled moved 17&#8211;23&#8202;&#181;m and stayed there. The
tip cap is meant to hold it; until the pad returns to its seat within a few &#181;T after a firm press, neither the model
nor a UR5e calibration describes it.</li>
<li><b>The magnets.</b> The N52 magnets are on order. Today&#8217;s resolve 0.10&#8211;0.12&#8202;N; with N52 all one way the models
predict 0.028&#8211;0.031&#8202;N, about 0.039&#8202;N at the measured noise. Four magnets (0.051&#8202;N) or five (0.029&#8202;N) waits on
the owner; a test of pads B4 and B00 measures the ratio, predicted 1.8.</li>
<li><b>Disturbances in the hand.</b> The other fingers&#8217; magnets (5.16&#8202;N equivalent at 25&#8211;30&#8202;mm on the first tip),
temperature (0.47&#8202;N per 10&#8202;&#176;C) and steel near the pad are modelled but not measured.</li>
<li><b>The FE on the 11/22 pad.</b> Hertz and the model resolving the beads checked the FE on the 95A block; on the
85A pad that is printed now, grid refinement is the one check, so the homogenized 30&#8202;% core on a curved surface has
not been compared with its beads.</li>
<li><b>Large presses and shear.</b> The FE is linear and frictionless; a neo-Hookean check lost 11&#8202;% of the force at
350&#8202;&#181;m on the softest tip. The tip&#8217;s tangential compliance, 48&#8211;177&#8202;&#181;m/N, is in neither simulator, and the
estimator&#8217;s shear has not been tested.</li>
<li><b>The simulators.</b> Calibrated to the FE, the pads and Drake agree on approach within 6&#8202;% but end a turn
6.3&#176; apart with contact forces 0.37&#8202;N apart; whether builder v2 stays the default waits on the owner.</li>
</ol>"""),
]

FIGURES = {
    "section": dict(page="20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html", n=1,
                    source="finite elements of the printed tip",
                    alt="Section of the printed tip: walls, skins, gyroid core, magnet pocket and the pause.",
                    caption="Section through the first study&#8217;s tip as printed, back face on the bed: two perimeter walls "
                            "and four solid layers around the gyroid core and the magnet&#8217;s pocket, the pause before the "
                            "magnet&#8217;s layer, and the magnetometer under the back face. The 11/22 pad prints the same "
                            "structures upright."),
    "moduli": dict(image="docs/overviews/media/fingertip_moduli_plot.png", width=760,
                   alt="Homogenized moduli of printed gyroid against solid fraction.",
                   caption="Moduli of the printed gyroid against the core&#8217;s solid fraction, from the homogenized cells "
                           "of the <a href='../experiments/20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html'>"
                           "finite-element study</a> (filled: 0.1&#8202;mm voxels; open: 0.05&#8202;mm, the cells the tips "
                           "use), in the layer plane and along the build axis, with the fits \\(0.43\\,\\phi^{1.40}\\) and "
                           "\\(0.60\\,\\phi^{1.29}\\), the ideal sheet gyroid, eFlesh&#8217;s cut cells and the "
                           "Gibson&#8211;Ashby limits. Drawn by <code>scripts/overview_fingertip_figures.py</code>."),
    # film labels go to ffmpeg drawtext, which reads a bare % as an expansion and then draws no label: escape it as \%
    "film_g20": dict(film=[[(FILM_G20, "95A TPU, gyroid 20 \\%, whole tip")]], page="20261009-tpu_tip_fem/"
                     "20261009-printed_tpu_fingertip_finite_elements.html", source="finite elements of the printed tip",
                     caption="The first study&#8217;s whole tip at 20&#8202;% gyroid under the 12.5&#8202;mm shaft: the section "
                             "through the contact point, displacement magnified 15 times and coloured by its size, beside the "
                             "pressure on the palmar face, as the approach grows. The walls take the edges of the patch."),
    "board": dict(page="20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html", n=2, source="magnetometer readout",
                  alt="The five-chip board under the tip, from the back face and in section.",
                  caption="The five-chip board under a tip: the chips U1&#8211;U5 under the back face in a plus 7&#8202;mm "
                          "apart, and the sensing gap in section."),
    "surface": dict(image="docs/experiments/20261020-fingertip_contact_estimation/media/20261020-surface_uncertainty.png",
                    width=828, alt="Press 1-sigma over the pad's surface, two magnet sets, three views.",
                    caption="Model 1-sigma of the pressing force for 1&#8202;N along the normal at each pad point, 50&#8202;ms "
                            "of samples, seen from the palm and from both flanks, for the magnets in the pad today and for "
                            "N52 magnets all one way (log scale). Red circles: magnets; white squares: chips. The mount end "
                            "and the flanks&#8217; lowest millimetres respond least. From the "
                            "<a href='../experiments/20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html'>"
                            "contact estimation</a>, Figure&#160;3."),
    "viewer": dict(film=[[(FILM_VIEW, "Browser viewer, five synthetic presses, magnets as found")]],
                   page="20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html",
                   source="contact estimation",
                   caption="The browser viewer replaying five presses the model made with the measured noise: the palmar "
                           "face, the +y flank, the face&#8217;s edge on the &#8722;y side (behind the pad in this view), the "
                           "tip end, and a press with shear along the finger. Arrow: the estimated force; ring: the 1-sigma "
                           "position ellipse; colour: the pads&#8217; pressure; deformation from the FE, drawn 20 times."),
    "hand": dict(image="docs/experiments/20261020-fingertip_contact_estimation/media/20261020-presses_by_hand.png",
                 width=760, alt="The owner's first presses by hand: reading change, estimate, and the offset left by the pad.",
                 caption="The owner&#8217;s first presses by hand (2026-10-20 19:39): the reading change from the first rest "
                         "with the rests in grey, the estimator&#8217;s force, and the last rest&#8217;s offset per chip axis "
                         "against a rigid motion of the whole pad fitted to it. From the "
                         "<a href='../experiments/20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html'>"
                         "contact estimation</a>, Figure&#160;5."),
}

FUTURE = [
    'Seat: with the pad fastened, record five firm presses with 2&#8202;s rests and run <code>src/morphohand/driver/reskin5x/host/pad_shift_check.py</code>; every rest within 5&#8202;&#181;T of the first passes (0.2&#8202;N on the face), offsets of tens of &#181;T that a rigid motion fits mean the pad still moves.',
    'N52 magnets: recalibrate in the viewer (key c); a strength of 0.8&#8211;1.2, all poles one way and a rest residual below 0.43&#8202;mT confirm the field model; a lower strength or larger residual means the magnets&#8217; depth or the pad&#8217;s seat differs from the FE.',
    'Force reference without the UR5e: a kitchen scale or load cell under a printed indenter, holds at about 1, 2 and 4&#8202;N at the face centre, a flank and the tip end; an estimated press more than three model sigmas from the reference falsifies the force scale at that point, and the ratio refits \\(E^*\\).',
    'FE of the 11/22 pad with its beads resolved: voxelize b&#8217;s pad at 0.1&#8202;mm with the 30&#8202;% gyroid (<code>scripts/tpu_tip_fem_tip.py voxels --stl &lt;pad STL&gt; --rho 0.3</code>), solve it under the 12.5&#8202;mm rod at 0.5, 1 and 3&#8202;N with <code>--E 12</code> (<code>scripts/tpu_tip_fem_gpu.py --indenter cyl</code>, 2&#8211;3&#8202;GB of GPU memory, minutes per load) and compare with the homogenized pad without magnets. The GPU solver first needs the gaps from the 11/22 surface of <code>tip_codesign_geometry</code> (its <code>--surface</code> reads the first study&#8217;s insert) and the pad&#8217;s cut face fixed. A difference above 5&#8202;% falsifies the homogenized core on the curved pad; the same pad voxelized from its G-code (a new entry in <code>TIPS</code> of <code>scripts/eflesh_tip_fem.py</code>) gives the as-printed stiffness that the indentation tests.',
    'Indentation with the HEX-E: approach against force at 0.5, 1 and 3&#8202;N under the 12.5&#8202;mm rod; outside &#177;25&#8202;% of the FE&#8217;s 150&#8202;&#181;m at 1&#8202;N falsifies the pad&#8217;s modelled stiffness, independently of the magnets.',
    'UR5e session on the lab&#8217;s computer: probe points from the basis&#8217;s pad points over the face and both flanks, three normal forces and two shear directions each, a return to one reference point every ten presses, 30&#8202;s rests; refit strength, offsets, \\(\\theta_0\\), \\(E^*\\) and per-magnet strengths on half the points and compare the other half&#8217;s errors with the model 1-sigma.',
    'Facing finger: record the board with a second tip&#8217;s magnets at 20&#8211;40&#8202;mm and compare with the dipole prediction before subtracting it from the joint angles.',
    'Temperature: warm the pad at rest with a hand around it; the strength term should take up the drift without a contact appearing.',
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
    ("Design", "20261019-v2_tip_fe_checks/20261019-v2_fingertip_stiffness_checks.html",
     "The v2 tips and b&#8217;s pad: print deviations, magnet count and depth, cover walls, resolution"),
    ("Design", "20261014-tip_codesign/20261014-sphere_pad_accuracy_and_fingertip_codesign.html",
     "Sphere-pad accuracy across tip shapes; fillets and recipes screened on the hand&#8217;s tasks"),
    ("Design", "20261015-curved_tips/20261015-capsule_and_ellipsoid_fingertips.html",
     "Capsule and ellipsoid tips on the tasks, stiffness maps against the FE"),
    ("Design", "20261013-eflesh_printability/20261013-eflesh_lattice_printability.html",
     "Printability, as-printed stiffness and resolution of eFlesh lattices"),
    ("Hardware", "20261018-dip_link_board_mount/20261018-dip_link_magnetometer_board_mount.html",
     "Board mount in the DIP link: concepts a, b and c, cable route, print projects"),
    ("Hardware", "20261020-ur5_calibration_mount_v2/20261020-dip_link_b_v3_and_ur5_mount_v2.html",
     "DIP link b v3 and the UR5e calibration mount v2"),
    ("Hardware", "20261018-fingertip_print_procedure/20261018-siraya85a_fingertip_print_procedure.html",
     "Print order, pauses, seat coupon and records for the 85A tips"),
    ("Hardware", "20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html",
     "Requests, jobs, prints and decisions waiting on the owner"),
]
