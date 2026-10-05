# Fingertip contact literature: CSLC and SDF hydroelastic contact

Notes for the fingertip-contact overview page, 2026-10-05. Sources are the two workshop papers in
`docs/experiments/20261004-codex/` and Codex's asides page in the same folder. No simulation was run.
Two checks supplement the papers. The first is a 225-unknown linear solve of CSLC Eqs. (3)-(4), which
reproduces the paper's Fig. 2 (Appendix A). The second is a read of the Newton source our probes run
(Section 3.2). Page numbers are PDF pages; both papers have five pages, the fifth being references.

## 0. Findings

- The CSLC lattice is a discrete Pasternak foundation with Winkler modulus $k = k_a/s^2$ and shear
  parameter $G = k_\ell$, where $s$ is the lattice spacing (the paper's $h$). The paper's decay length
  $\ell = s\sqrt{k_\ell/k_a} = 4.24$ mm follows. A 4-neighbour Laplacian reproduces Fig. 2: 0.092 mm
  centre deflection under 58 mN, against about 0.09 mm in the figure. An 8-neighbour stencil gives 0.047 mm.
- The CSLC normal law per sphere is the anchor $k_a$ in series with a Hertz-like element
  $k_c A\,\phi^{3/2}$. With $k_\ell = 0$ it is a nonlinear Winkler model. Our pads match it only where the
  anchor dominates, and they lack its tangential anchor $\rho k_a$ and its stick stiffness $k_\mathrm{stick}$.
- CSLC never tests torsion about the contact normal. Its rotational test (Fig. 1, p. 2; Sec. IV-C, p. 4)
  is a pitch disturbance resisted by the couple of distributed normal forces. It is compared only with a
  rigid point contact and has no $k_\ell = 0$ ablation.
- The CSLC friction law (Eq. 8) acts on $\delta_t$, the skin shear relative to the robot body. Neither the
  workshop PDF nor arXiv v1 states how sliding of the object relative to the lattice enters $\delta_t$.
- Newton's hydroelastic contact is a Winkler foundation evaluated on the equal-pressure surface. Per
  shape, $p = k_h \cdot \text{depth}$ with $k_h$ in N/m³ and no normalization by body size. Per triangle,
  the series modulus $k_A k_B/(k_A + k_B)$ applies. It equals Drake's sphere field when $k_h = E/R$, as in
  our runs ($9.479\times10^{8}$ N/m³).
- Newton's contact reduction matches each normal bin's force exactly and its torsional friction capacity
  approximately. The paper gives no error figure. Codex's static study measured the friction envelope
  17.6 % low with default reduction and 5.9 % / 14.2 % low with moment matching at 0.2 / 0.4 mm indentation.
  In the Newton build our probes use, moment matching is off by default and off in our probe scripts.
- Neither paper reports a timestep, a steps/s figure or a measured comparison with Drake. Newton reports
  accumulated realtime factors on an RTX 5090 (Table I, p. 3). CSLC reports only "roughly three times" the
  per-step cost of point contact (p. 4).

## 1. Citations

### 1.1 Newton SDF hydroelastic contact

L. Röstel, Y. Liu, J. Yin, M. Zamora, M. Macklin, P. Reist, T. Widmer, "GPU-Accelerated Hydroelastic
Contact via Signed Distance Fields," Workshop on the Path Towards Generalizable Contact-Rich Robotics:
Control and Representation (CR2) at ICRA 2026, Vienna, 1 June 2026. Spotlight talk and poster;
OpenReview id `ogndqznZyY`.

- The PDF prints the title, the authors and their affiliations (NVIDIA; TUM, Germany; EPFL,
  Switzerland), a note that the first author's work was done during an NVIDIA internship, and the code URL
  `github.com/newton-physics/newton`. It prints no venue, year or arXiv id. PDF created 2026-05-29.
- Venue and OpenReview id come from the workshop site, cr2-icra.github.io, which lists the fourth author
  as Miguel Angel Zamora Mora. A search on 2026-10-05 found no arXiv version.
- Local file: `docs/experiments/20261004-codex/11_GPU_Accelerated_Hydroelasti.pdf`.

### 1.2 Compliant Sphere Lattice Contact (CSLC)

N. Nechyporenko, A. Abderezaei, A. Roncone, "Compliant Sphere Lattice Contact: Distributed Contact
Modeling for Sphere-Based Robot Representations," same workshop (CR2 at ICRA 2026). Spotlight talk and
poster; OpenReview id `46BqR1wKqA`; arXiv:2608.00263 (v1 submitted 31 July 2026, comment "ICRA 2026
Workshop on Contact-Rich Control and Representation").

- The PDF prints the title and the authors only, with no affiliations, venue, year or URL. PDF created
  2026-05-29. The arXiv v1 HTML has the same sections, equations and parameter values, including the
  $k_c$ unit typo (Section 2.6). It has no code link.
- Local file: `docs/experiments/20261004-codex/22_Compliant_Sphere_Lattice_Co.pdf`.

## 2. CSLC

### 2.1 Model in equations (pp. 2-3)

**Lattice geometry.** A rigid body carries $n$ surface spheres. Sphere $i$ has a rest centre $p_i$ fixed
to the body, a contact radius $r_i$ ("the thickness of the compliant skin") and a precomputed outward
normal $\hat n_i$. The skin displacement $\delta_i \in \mathbb R^3$ moves the centre while $r_i$ stays fixed:

$$ q_i = p_i + \delta_i, \qquad \delta_{n,i} = \delta_i\cdot\hat n_i, \qquad \delta_{t,i} = \delta_i - \delta_{n,i}\,\hat n_i \qquad (1,2) $$

$\delta_{n,i} < 0$ is compression. $\delta = [\delta_1;\dots;\delta_n]\in\mathbb R^{3n}$ is the state of the
contact problem, and the rigid body follows its own dynamics independently of $\delta$. Sphere placement on
a pad is not described. The book test takes fingertip spheres from Morphit [28] (p. 4); Fig. 2 uses a flat
15 × 15 square grid at 3 mm spacing.

**Anchor spring (Eq. 3).**

$$ f_i^{\rm anchor} = -k_a\,\delta_{n,i}\,\hat n_i - \rho\,k_a\,\delta_{t,i}, \qquad k_a > 0,\ \rho\in(0,1] $$

$\rho = 1$ is isotropic. $\rho = 1/3$ is said to "match the shear modulus of a nearly-incompressible
elastomer", i.e. $G/E = 1/(2(1+\nu)) = 1/3$ at $\nu = 0.5$.

**Lateral springs (Eq. 4).**

$$ f_i^{\rm lateral} = -k_\ell \sum_{j\in\mathcal N(i)} (\delta_i - \delta_j) = -k_\ell\,(L\delta)_i, \qquad k_\ell \ge 0 $$

$L$ is the graph Laplacian of the lattice. The coupling acts component-wise on the 3-vector, so it carries
both normal compression and tangential shear to neighbours. The paper calls it the analogue of the
continuous pressure field of hydroelastic contact. The neighbourhood $\mathcal N(i)$ is not defined;
Appendix A shows that Fig. 2 requires the 4-neighbour stencil.

**Contact force (Eqs. 5-7).** The target is a set of elements $t_j$: object spheres, or points sampled on a
mesh. Each carries an outward normal $\hat n_{{\rm face},j}$ and an area element $A_j$ (packing area for
spheres, Voronoi area for mesh samples), and the $A_j$ sum to the target's surface area. For a lattice
sphere $i$ that reaches $t_j$:

$$ \hat n_{ij} = \hat n_{{\rm face},j}, \qquad \phi_{ij} = -\hat n_{{\rm face},j}\cdot(q_i - t_j) $$

$$ f^{\rm contact}_{ij} = k_c\,A_j\,s_{ij}\,\phi^{\rm eff}_{ij}\,\hat n_{ij}, \qquad \phi^{\rm eff}_{ij} = \max(\phi_{ij},0)^{3/2}, \qquad s_{ij} = \frac{w_{t,ij}}{\sum_k w_{t,kj}} $$

$\phi_{ij} = 0$ when the sphere centre reaches the target's tangent plane, so $r_i$ enters only through the
locality kernel. $w_{t,ij}$ "marks" lattice spheres that reach $t_j$ within their tangent-plane radius
$r_i$. The share $s_{ij}$ splits $A_j$ among them, so each target element contributes its area once. The
body force $\sum_{ij} f^{\rm contact}_{ij}$ approximates $\int_\Omega k_c\,\phi^{\rm eff}\,\hat n\,dA$, a
pressure $p = k_c\,\phi^{3/2}$ with a $C^1$ onset. Units of $k_c$ are N/m^{7/2}.

**Pre-sliding friction (Eq. 8).** With $f_{n,i} = |F_i^{\rm contact}\cdot\hat n_i|$ and
$F_i^{\rm contact} = \sum_j f^{\rm contact}_{ij}$:

$$ f_i^{\rm friction} = -\,\frac{k_{\rm stick}\,\mu f_{n,i}}{k_{\rm stick}\lVert\delta_{t,i}\rVert + \mu f_{n,i}}\;\delta_{t,i} $$

This is $-k_{\rm stick}\delta_{t,i}$ at small shear and saturates at magnitude $\mu f_{n,i}$. There is no
separate bristle state; $\delta_t$ comes out of the equilibrium solve. The authors call the law
"quasistatic and tangentially stateless". It represents pre-slip elasticity and the Coulomb limit and
excludes kinetic sliding dynamics (p. 3).

**Stiffness calibration (Eq. 9).** An engaged sphere is a series chain of the anchor $k_a$, the contact
element and a target stiffness $k_e^{\rm target}$. Linearizing Eq. (5) at $\phi_0$ gives
$\kappa_c = \tfrac32 k_c A_c\sqrt{\phi_0}$. Matching $N_{\rm contact}$ such chains in parallel to an
aggregate $k_e^{\rm bulk}$ gives

$$ \frac{1}{\kappa_c} = \frac{N_{\rm contact}}{k_e^{\rm bulk}} - \frac{1}{k_a} - \frac{1}{k_e^{\rm target}}, \qquad k_c = \frac{2\kappa_c}{3A_c\sqrt{\phi_0}} $$

The paper calls $k_e^{\rm bulk}$ and $k_e^{\rm target}$ moduli, but they enter as stiffnesses (N/m).
"To simplify the analysis in this work we set $k_c$ directly" (p. 3), so no experiment uses Eq. (9).

**Quasistatic equilibrium (Eq. 10).** At each timestep,

$$ f_i^{\rm anchor} + f_i^{\rm lateral} + F_i^{\rm contact} + f_i^{\rm friction} = 0 \qquad \forall i $$

is solved "with a damped Jacobi sweep, warm-started from the $\delta$ of the previous timestep". Updates
are independent across spheres, which suits the GPU. The update formula, damping factor, iteration count
and tolerance are not given; arXiv v1 also omits them.

**Coupling to the rigid bodies.** "After the lattice solve converges, CSLC hands the deformed sphere
centres $q_i$ and the per-sphere contact and friction forces to the host rigid-body simulator, which
advances the body state according to its own integration scheme" (p. 3). Two hosts are named: a MuJoCo
rigid-body pipeline with mesh-sampled targets and the PBD-R position-based solver [27] with sphere targets.
The introduction says CSLC is implemented "in the Newton simulator" (p. 1). The paper does not say whether
the host applies the returned forces explicitly or re-solves contact at $q_i$.

**Parameters.** Values appear only for Fig. 2: $s = 3$ mm, $k_a = 100$ N/m, $k_\ell = 200$ N/m,
$k_c = 10^{9}$ N/m^{7/2}. The MuJoCo grasp uses $\mu = 0.5$ and the book test $\mu = 1.0$. Unreported:
$\rho$, $k_{\rm stick}$, $r_i$, sphere counts, pad geometry and host timesteps. Nothing is identified
from data; Eq. (9) is offered for future offline tuning.

### 2.2 What the equations imply (derived here, not stated in the paper)

1. **Pasternak mapping.** For normal displacement $w$ on a flat 4-neighbour lattice,
   $\sum_j(\delta_j - \delta_i) \approx s^2\nabla^2 w$, so the per-area balance is
   $$ p = \frac{k_a}{s^2}\,w - k_\ell\,\nabla^2 w, $$
   a Pasternak foundation with $k = k_a/s^2$ (N/m³), $G = k_\ell$ (N/m) and $\ell = \sqrt{G/k} = s\sqrt{k_\ell/k_a}$.
   Fig. 2's parameters give $k = 1.11\times10^{7}$ N/m³ and $\ell = 4.24$ mm. The continuum point-load
   solution is $w(r) = (P/2\pi G)\,K_0(r/\ell)$, whose large-$r$ asymptote is
   $\propto r^{-1/2}e^{-r/\ell}$. The $e^{-r/\ell}$ line in Fig. 2 is therefore an upper envelope. The
   lattice solution matches both the figure and the $K_0$ form to within about 10 % for $r \ge 3$ mm
   (Appendix A). For comparison, our pads' per-area stiffness $E/R = 9.48\times10^{8}$ N/m³ (E = 10 MPa,
   R = 10.55 mm) is 85 times that of the Fig. 2 lattice.
2. **Point stiffness.** The 58 mN load deflects the loaded sphere 0.092 mm, a point stiffness of 632 N/m
   or $6.3\,k_a$. Without coupling the same load would deflect it 0.58 mm.
3. **Tangential spreading.** In-plane displacement sees the anchor $\rho k_a$ and the same $k_\ell$, so
   $\ell_t = s\sqrt{k_\ell/(\rho k_a)} = \ell/\sqrt\rho$. At $\rho = 1/3$, $\ell_t = 1.73\,\ell$
   (7.35 mm for Fig. 2). On contacting spheres $k_{\rm stick}$ adds to $\rho k_a$ and shortens it.
4. **Curvature.** $L$ acts on 3-vectors in a common frame. On a curved pad, the compression of one sphere
   therefore enters the tangential components of a neighbour whose normal differs. Coupling is tested only
   on the flat lattice.
5. **Jacobi convergence.** For $k_aI + k_\ell L$ on a 4-neighbour grid, the undamped Jacobi contraction
   factor is about $4k_\ell/(k_a + 4k_\ell) = 1 - 1/(1 + 4(\ell/s)^2)$. That is 0.89 for Fig. 2, about 20
   sweeps per decade of residual, and 0.97 at $\ell = 3s$, about 84 sweeps per decade. Damping slows it
   further. This fits the paper's attribution of part of its 3× cost to the Jacobi solve (p. 4).
6. **Friction reference frame.** Eq. (8) is written in $\delta_{t,i}$, the sphere's shear relative to its
   rest position on the robot body, and acts along $-\delta_{t,i}$, the same sense as the tangential
   anchor. Eq. (5) acts along the target's face normal. For a flat face aligned with $\hat n_i$, no term in
   Eqs. (3)-(10) carries the object's tangential motion into $\delta_t$. The MuJoCo grasp still holds
   against gravity (Fig. 4), so the implementation must couple object sliding to the lattice somewhere.
   The paper does not describe it.
7. **Bonded layer.** $\rho = 1/3$ is the $G/E$ ratio of an unconfined column. A thin, nearly incompressible
   layer bonded to a rigid core is much stiffer in compression than $E/h$ when the loaded region is wide
   compared with $h$, so a bonded TPU skin would sit well below $\rho = 1/3$. This is standard elasticity;
   neither paper tests it.

### 2.3 Validation

**A. Lattice under a point load** (Sec. IV-A, Fig. 2, p. 3). A flat 15 × 15 lattice with $s = 3$ mm,
$k_a = 100$ N/m, $k_\ell = 200$ N/m and $k_c = 10^9$ N/m^{7/2} has one contact sample pressed into the
centre sphere. Only the centre sphere receives contact force (58 mN), so the rest of the deformation comes
from $L$ alone. The metric is $|\delta_n|$ against distance on a log scale, with the envelope $e^{-r/\ell}$
at $\ell = 4.24$ mm. Fig. 2 reads about 0.09 mm at the centre, falling to about $5\times10^{-5}$ mm at the
lattice corner (30 mm). Nothing compares this with an elastic solution or a measurement.

**B. MuJoCo squeeze, lift and hold** (Sec. IV-B, Figs. 3-4, p. 4). Three two-finger grasps are run: a
flat pad on a sphere, a dome pad on a sphere and a dome pad on a box. The squeeze is 1 mm in $\phi$, with
$\mu = 0.5$ and mesh-sampled targets (markers in Fig. 3). The metric is per-pad normal force against the
slip floor of Eq. (11), $2\mu F_n \ge mg$, which is 0.57 N for the ball and 1.09 N for the box. These floors
imply masses of 58 g and 111 g.

- Reported: the dome grasps hold at 0.7 N (ball) and 1.5 N (box) and the flat pad at 2.9 N (ball). The
  objects rise about 20 mm.
- Fig. 4 at 0.5 s: squeeze peaks of 12.7 N (flat·sphere), 8.6 N (dome·box) and 4.0 N (dome·sphere).
- Fig. 4, lift and hold: all three objects reach 23.5 mm at 1.8 s and settle at 20.0-20.5 mm by 2.0 s.
  From 2 s to 5 s the flat pad holds at 20.5 mm, while dome·box falls to about 19.2 mm and dome·sphere to
  about 18.8 mm, a 1.0-1.2 mm drop over 3 s.
- Fig. 4, during the lift (0.75-1.9 s): the dome·sphere force is about 0.5 N, at or below the 0.57 N floor.
- No point-contact or native MuJoCo baseline is run in this test.

**C. Rotational grasp stability in PBD-R** (Sec. IV-C, p. 4; traces in Fig. 1 right, p. 2). A book of
2 × 10 × 20 cm and 0.3 kg is pinched by two curved fingertips approximated with spheres, then lifted. A
3 N, 50 ms pulse is applied at the book's lower end, normal to its face (Fig. 1 schematic). The disturbance
pitches the book about an axis perpendicular to the pinch axis.

- Baseline: the same fingertip treated as a rigid convex shape (point contact), with the same scene,
  $\mu = 1.0$, grip kinematics and squeeze force.
- Load: friction capacity is "roughly two orders of magnitude above the book's weight" (2.94 N), which
  implies a squeeze of order 150 N per finger at $\mu = 1$ (derived).
- Reported: point contact tilts more than 6° and slides 18 mm down, while CSLC stays within ±0.7° with
  negligible height loss and damps back.
- Fig. 1, rigid contact (log time axis 1-1500 ms): $\Delta$tilt peaks at about 6.5° near 650 ms and ends
  at about 4.9°; $\Delta z$ reaches about −18.2 mm by 600 ms.
- Fig. 1, CSLC: $\Delta$tilt peaks at about +0.7° near 60 ms; $\Delta z$ is about −0.6 mm at 1.5 s.
- Mechanism given: "restoring torque through its off-axis compliant springs", which is the couple of the
  distributed normal forces. No CSLC run with $k_\ell = 0$ is shown.

Nothing in the paper is compared with hardware, FEM, Hertz theory or a hydroelastic model.

### 2.4 Reported performance (p. 4)

The only figure is "roughly three times as much per step" as point contact "on the same grasp scene".
Neither the scene nor the host is named. No hardware, timestep, sphere count, iteration count or absolute
time is given. Part of the gap is called intrinsic: CSLC resolves a distributed patch instead of a few
points and solves a lattice equilibrium every step.

### 2.5 Stated limitations (p. 4, and p. 3 for friction and $k_c$)

- Quasistatic lattice relaxation: the solver "breaks down during high-speed impacts where the skin cannot
  equilibrate within a timestep". A dynamic extension is named as the remedy.
- About 3× the per-step cost of point contact. A better-conditioned solve with fewer Jacobi iterations is
  named as the remedy.
- The hard $\max(\cdot,0)$ clamp and hard active-set culls in the forward path block differentiability.
  Smooth surrogates are future work.
- The friction law captures pre-slip elasticity and the Coulomb limit but not kinetic sliding dynamics.
- $k_c$ is set directly, not identified from a bulk modulus.

### 2.6 Inconsistencies and omissions

- Units of $k_c$: "N/m^{2/7}" beside Eq. (5) and "N/m^{7/2}" in Sec. IV-A, both on p. 3. Eq. (5) requires
  N/m^{7/2}. arXiv v1 repeats the typo.
- "Roughly 1.4 times the limit" (p. 4) fits the box (1.5/1.09 = 1.38) but not the ball (0.7/0.57 = 1.23).
- "Holds without slipping" (p. 4): the dome traces in Fig. 4 drop 1.0-1.2 mm between 2 s and 5 s.
- Eq. (11) assumes horizontal contact normals. A dome contacting a ball below its equator carries part of
  the weight through the normal force, which may explain the sub-floor force during the lift. The paper does
  not comment.
- The abstract's "two independent solvers" are the MuJoCo pipeline and PBD-R, while the introduction names
  Newton as the implementation.
- Not specified: $\mathcal N(i)$, the form of $w_{t,ij}$, sphere placement on curved pads, $\rho$,
  $k_{\rm stick}$, and how the host applies the lattice forces.

## 3. Newton SDF hydroelastic contact

### 3.1 Model in equations (pp. 2-3)

**SDF construction.** Each object is a signed distance field $\phi(x)$ sampled on a regular grid of
axis-aligned cubic voxels in the object frame, with $\phi < 0$ inside. The implementation uses sparse
narrow-band SDFs (p. 2). Generation from a mesh is not described beyond the claim that it works for
arbitrary geometry.

**Pressure field per shape.** Pressure rises linearly with penetration depth from each surface:

$$ p_A(x) = k_A\,(-\phi_A(x)), \qquad p_B(x) = k_B\,(-\phi_B(x)) $$

$k_A$ and $k_B$ are "hydroelastic moduli". Since $\phi$ is a length, $k$ has units of pressure per length
(N/m³), so it is a Winkler modulus with no normalization by body size. Our runs use pad $k_h = E/R =
9.479\times10^{8}$ N/m³. The paper writes $p(v_j) = k_A\phi_A(v_j) = k_B\phi_B(v_j)$ (p. 2), which is
negative under its own sign convention; magnitudes are meant.

**Equal-pressure surface (Eqs. 1-2).**

$$ g(x) = k_A\,\phi_A(x) - k_B\,\phi_B(x), \qquad \mathcal S = \{x\in\mathbb R^3 \mid g(x) = 0,\ \phi_A(x) < 0,\ \phi_B(x) < 0\} $$

**Marching cubes and pruning.** Marching cubes runs on the finer of the two SDF grids, with $g$ sampled
at voxel corners and the coarser SDF read by trilinear interpolation. The broad and midphase first keep
pairs whose SDF bounding boxes overlap. They then evaluate $g$ at the centre $x_T$ of each 8×8×8-voxel
block and prune the block if $|g(x_T)| > R_T = r(k_A + k_B)$, with $r$ the block half-diagonal, recursing
through 4³, 2³ and 1³ sub-blocks. The bound holds because $|\nabla\phi| \le 1$ makes $g$ Lipschitz with
constant $k_A + k_B$; the paper calls $R_T$ a "stiffness-corrected bounding radius". Triangles are kept
when all their vertices satisfy $\phi_A < 0$ and $\phi_B < 0$, and each vertex carries $p(v_j)$.

**Contact points, normals, areas and stiffness (Eqs. 3-4).** Following Masterjohn et al. (2022), each
triangle yields one contact $(x_c, \hat n, \phi_0, k_{\rm eff})$ at its centroid $x_c$, with the face normal
$\hat n$ and area $a$:

$$ \phi_0 = \phi_A(x_c) + \phi_B(x_c), \qquad k_{\rm eff} = \frac{k_A k_B}{k_A + k_B}\,a, \qquad f_{n,i} = k_{\rm eff}\,|\phi_{0,i}| $$

On $\mathcal S$, $p = k_A d_A = k_B d_B$ with $d = -\phi$, so $d_A + d_B = p\,(1/k_A + 1/k_B)$ and
$f_n = p\,a$ exactly. Centroid evaluation integrates the linear field exactly over the triangle. Triangles
appear and disappear at zero area, so forces vary continuously with configuration. Eq. (4) is the head-on
case of Masterjohn's series form, which projects each pressure gradient on $\hat n$
($g_A = k_A\nabla\phi_A\cdot\hat n$). The two differ for oblique or conforming patches (Section 3.2).

**Speculative contacts.** Non-penetrating faces with $\phi_A < d_{\rm gap}$ and $\phi_B < d_{\rm gap}$ are
also kept (p. 2).

**Contact reduction (Sec. II-C, Eqs. 5-8, p. 3).** Reduction exists because MuJoCo's solver cost grows
superlinearly with constraint count. The Jacobi-style XPBD solver can take the unreduced set.

1. Normal binning. $N_b$ bin normals come from icosahedral subdivision; contact $i$ goes to
   $b^* = \arg\max_b\langle\hat n_i,\hat n_b\rangle$.
2. Per-bin aggregation:
   $$ F_b = \sum_{i\in b} f_{n,i}\,\hat n_i, \qquad \bar x_b = \frac{\sum_{i\in b} p_i x_i}{\sum_{i\in b} p_i}, \qquad M_b = \sum_{i\in b}\mu f_{n,i}\,\lvert r_i\times\hat n_i\rvert,\quad r_i = x_i - \bar x_b $$
   "Second-moment statistics" of the bin are also computed; their use is not described. $M_b$ is the
   pure-spin friction torque about the centroid.
3. Spatial selection. $N_u$ directions $u_j$ are spaced in the tangent plane of $\hat n_b$, and the step
   keeps $i^*_j = \arg\max_i\,\langle x_i - \bar x_b, u_j\rangle\,|\phi_{0,i}|$ (Eq. 7). It also keeps the
   deepest contact in each voxel of a coarse grid on the object's bounding box. Neither convexity nor
   connectivity of the patch is assumed.
4. Wrench matching. The selected contacts get a uniform stiffness
   $k^b_{\rm eff} = \lVert F_b\rVert / \sum_{\rm sel}|\phi_{0,j}|$. Their normals are then rotated by
   $R_b = {\rm rot}(F_{\rm sel}/\lVert F_{\rm sel}\rVert \to F_b/\lVert F_b\rVert)$, with
   $F_{\rm sel} = \sum_{\rm sel} k^b_{\rm eff}\,\phi_{0,j}\,\hat n_j$ (Eq. 8).
5. Moment matching. The selected contacts' friction coefficients $\mu_j$ are scaled "non-uniformly based
   on their distance to" $\bar x_b$, so that the maximum friction moment about $\hat n_b$ "approximately
   matches $M_b$" while the bin's net tangential friction capacity is kept. The paper gives no formula.

The reduction keeps the bin force $F_b$ exactly at the current configuration, and the torsional and
tangential friction capacities approximately. No stated equation constrains the moment of the normal forces
about $\bar x_b$, so the centre of pressure, and with it the rocking stiffness, is not matched. Nor does any
constrain the force's derivative: $k^b_{\rm eff}$ is a secant fitted at one configuration, so the patch
stiffness is not preserved.

**Solver coupling.** Reduced contacts go to MuJoCo-Warp (the nut-and-bolt and Panda scenes) and to
Newton's XPBD (the bunny pile). The paper does not say how $k_{\rm eff}$ enters MuJoCo-Warp's constraint,
for example through the solref mapping or the damping.

**Friction.** Each contact point has a Coulomb $\mu$, and torsion comes from the spread of the points:
"the hydroelastic model instead resolves contact area from the SDF, so torsional capacity emerges from the
spatial extent of the reconstructed isopressure surface" (p. 4). This is set against MuJoCo's per-geom
torsional coefficient. Neither condim nor cone type is stated. No tangential compliance or tangential state
is described.

### 3.2 Paper against the code our probes run

The code is Newton 1.7.0.dev0 in `logs/20261004-contact-transfer/newton-src`, commit 009158e
(2026-10-04), installed editable in that folder's venv. Paths below are under `newton/_src/geometry/`.

- **Per-face separation and stiffness.** The separation is $\phi_A + \phi_B$, averaged over the
  triangle's vertices (`sdf_hydroelastic.py` lines 340 and 358). The stiffness is
  $c = a\,p/|\phi_A + \phi_B|$ with $p$ from shape B's field (line 1911). For the linear law this equals
  Eq. (4) at any modulus ratio.
- **Issue #3503** (newton-physics/newton, open since 2026-07-13). It describes an older single-body secant
  ($k = a\,k_{hB}/2$, $\phi_0 = 2d_B$) and asks for Masterjohn's projected-gradient series form. Against
  this build, the outstanding difference concerns oblique or conforming patches only. The force $a\,p$ is
  the same in all three forms.
- **Bin centroid.** The code weights the centroid by $a_i p_i$, which makes it the centre of pressure
  (lines 2158-2160). The paper's text weights by $p_i$ alone.
- **Moment matching** (`contact_reduction_hydroelastic.py` lines 1219-1305). Over the selected contacts
  let $S_0 = \sum d_j$, $S_1 = \sum d_j L_j$ and $S_2 = \sum d_j L_j^2$, with $d_j$ the depth and $L_j$
  the lever about the centre of pressure, and set the target $m^* = M_{\rm unreduced}\,S_0/\lVert F_b\rVert$.
  If $m^* < S_1$, every $\mu_j$ is scaled by $m^*/S_1$. Otherwise
  $$ \mu_j \leftarrow \mu_j\Big[1 + \alpha\,\frac{L_j - \bar L}{\bar L}\Big], \qquad \bar L = S_1/S_0, \qquad \alpha = {\rm clamp}\Big(\frac{(m^* - S_1)\,S_1}{S_2 S_0 - S_1^2},\,0,\,1\Big). $$
  This keeps $\sum d_j\mu_j$ fixed and matches the moment while $\alpha \le 1$. The clamp leaves a residual
  when the selected levers are too uniform, which is consistent with the 5.9 % and 14.2 % shortfalls Codex
  measured.
- **Anchor contact.** An optional synthetic contact at the centre of pressure (`anchor_contact`) has no
  counterpart in the paper.
- **Defaults.** `moment_matching=False` and `anchor_contact=False` (`sdf_hydroelastic.py` lines 465, 468).
  Our scripts set `anchor_contact=True` and leave moment matching off: `scripts/distributed_contact_newton.py`
  (line 160, default), `scripts/newton_scaling.py` (line 206), `scripts/newton_pinch_probe.py` (line 156)
  and `scripts/contact_bed_newton.py` (line 137). Codex's static study ran both settings
  (`docs/experiments/20261004-codex/data/articulated/newton_pressure.json`).
- **Zero per-contact stiffness.** A contact that arrives with zero per-contact stiffness falls back to the
  geom solref in `SolverMuJoCo`. The MJCF importer stored a one-value solref with damping ratio 0, which
  blew up the 10-05 probe (commit cf3b7ea8). The paper does not cover this path.

### 3.3 Validation (pp. 3-4)

**A. Throughput scenes** (Sec. III, Fig. 2 and Table I, p. 3): an M20 nut-and-bolt (from Factory) in
MuJoCo-Warp with 128 instances, a pile of 91 non-convex bunnies in XPBD, and a Franka Panda placing a pen
in a cup in MuJoCo-Warp with 512 instances. The metrics are contact count and accumulated realtime factor.
No accuracy metric is reported.

**B. Visuotactile sensing** (Sec. IV, Fig. 3, p. 4). On the Sharpa Wave hand grasping a bottle, rays cast
inward from each taxel along its local normal against the unreduced contact surface give per-pixel
elastomer displacement maps (0-6 mm colour scale). The result is qualitative, with no comparison to the
real sensor.

**C. Area-resolved release dynamics** (Sec. V, Fig. 4, p. 4). A parallel-jaw gripper holds a bar off its
centre of mass and releases gradually. Gravity torque then pivots the bar about the pinch axis, so the test
exercises torsional friction about the contact normal.

- Sweep: $\mu$ and contact width, following Acosta et al. [21]. Width changes by shifting the bar in the
  gripper while the CoM offset from the patch centre stays fixed.
- Metric: $|\omega_y|$ at detachment.
- Fig. 4: over $\mu$ = 0.4-1.3 and width 22-40 mm, $|\omega_y|$ runs from about 0.2 rad/s
  ($\mu \approx 1.2$, 38-40 mm) to about 1.6 rad/s ($\mu \approx 0.4$-0.55, 22-27 mm).
- Claim: the trends agree with the ILS model of Liu and Billard [18], and the map is "smooth and monotonic".
- The ILS comparison exists in the text only. The PDF's text layer holds the labels of a second, clipped
  panel (title beginning "I", axis "Effective Contact Radius (mm)" from 2 to 11 mm, a second friction axis)
  that does not render on the page.
- The paper reports no real-robot data and does not state whether reduction was on in this test.

**D. Not reported:** any check of the reduction error (force, centre of pressure or torsion), any
comparison with Drake or point contact, and any hardware data.

### 3.4 Reported performance (Table I, p. 3; NVIDIA RTX 5090)

| Scene | Solver, instances | Contacts | Acc. realtime factor | Per instance, if totals |
|---|---|---|---|---|
| nut-and-bolt, reduced | MuJoCo-Warp, 128 | 18k | 141× | 141 contacts, 1.10× |
| nut-and-bolt, unreduced | MuJoCo-Warp, 128 | 4M | 18× | 31k contacts, 0.14× |
| bunny pile, reduced | XPBD, not stated | 14k | 5.0× | not computable |
| Panda-pen, reduced | MuJoCo-Warp, 512 | 57k | 104× | 111 contacts, 0.20× |

On the nut-and-bolt, reduction cuts the contact count 222× (the paper says "more than 200×") and raises
speed 7.8×. Fig. 1 (p. 2) shows 274 contacts reduced to 17 for one pair. The paper gives no timestep, so
neither steps/s nor contacts/s can be recovered. It also does not say whether the counts and factors are
totals over instances; the last column assumes they are. There is no Drake or point-contact timing.

For scale only, on a different scene and GPU: our 1 mm sphere pads on MuJoCo-Warp reach 923k
world-steps/s at 4096 worlds and a 1 ms step, about 920 simulated seconds per wall second, on the
workstation's RTX 4070 Ti SUPER (commit 51358fff).

### 3.5 Stated limitations and future work (pp. 1, 4)

- Motivation on the Drake side (p. 1): Drake generates tetrahedral meshes only for convex primitives,
  while non-convex objects need user-supplied VTK files. Tet-tet intersection also grows expensive with
  resolution. None of this is measured.
- Sec. V: none of the remedies for Zeno-like release oscillations, such as viscous smoothing with implicit
  integration, is built into the formulation tested.
- Future work (p. 4): sim-to-real validation of policies, including tactile ones; system-identification
  guidelines for $k$; hardware texture sampling for SDF queries; stress tests of contact reduction and its
  interplay with solvers "under more challenging configurations and edge-cases"; differentiability, with
  preliminary gradients with respect to $k$; and a comprehensive benchmark against Drake, "acknowledging
  that direct comparison is challenging due to differences in underlying formulations".

## 4. Relation to Winkler, Pasternak, Drake and our sphere pads

### 4.1 Normal laws side by side

| Model | Element law | Per-area normal stiffness | Lateral coupling | Tangential elasticity | Torsion |
|---|---|---|---|---|---|
| Winkler | $p = (E/h)\,\delta$ | $E/h$ | none | none | n/a |
| Pasternak | $p = k\delta - G\nabla^2\delta$ | $k$ | $G$, length $\ell = \sqrt{G/k}$ | none in the normal model | n/a |
| Drake hydroelastic, sphere (Elandt 2019) | $p = E(R - \lvert x\rvert)/R$ | $E/R$ | none | none | Coulomb per polygon |
| Newton SDF hydroelastic | $p = k_h\,$depth per shape; series per triangle | $k_Ak_B/(k_A + k_B)$ | none | none described | Coulomb per point; reduction rescales $\mu_j$ |
| Our 1 mm pads, MuJoCo | $k_s = (E/R)\,A_s((R - r_s)/R)^2$, realized as $k_s = 1/(t_c^2(1 - d_0)\Lambda)$ | $E/R$ | none | none; first-order friction creeps below the cone | condim-3 cone per sphere |
| CSLC | anchor $k_a$ in series with $p = k_c\phi^{3/2}$; lattice $p = (k_a/s^2)w - k_\ell\nabla^2 w$ | $k_a/s^2$ where the anchor dominates | $k_\ell$, $\ell = s\sqrt{k_\ell/k_a}$ | $\rho k_a$ anchor plus $k_{\rm stick}$, saturating at $\mu f_n$ | per-sphere saturation; not tested |

The pad row uses the mapping of 2026-10-01 (`docs/experiments/20261001-hom_contact_patch/`), with
$\Lambda$ taken once per pad-object pair at load (commit e8d3e950).

1. **Newton and Drake.** Both are Winkler foundations evaluated on the equal-pressure surface. Newton's
   $k_h$ equals Drake's $E/R$ for a sphere when chosen so. Drake's field for each shape is normalized to
   that shape's extent ($\epsilon \in [0,1]$), while Newton's grows with Euclidean depth, so at equal $E$
   the two differ on thin or elongated bodies. Newton reduces contacts for MuJoCo-Warp; Drake passes every
   polygon to SAP.
2. **Pads against Newton.** Our pads use the same Winkler law ($E/R$), discretized at fixed sites on the
   pad. Newton discretizes the moving isosurface into triangles. Both hand linear-stiffness point contacts
   with their own Coulomb cones to the solver, and both get torsion from the spread of the points.
   Newton's elements migrate with the isosurface into the softer body, while the pad sites are rigidly
   attached and overlap the object by the penetration.
3. **Pads against CSLC.** CSLC with $k_\ell = 0$ is a nonlinear Winkler model. The pads correspond to it
   only when the anchor dominates ($k_a \ll \kappa_c$) and with $\rho k_a$ and $k_{\rm stick}$ removed.

### 4.2 Pasternak foundation under a spherical tip (derived here)

Take a rigid sphere of radius $R$ pressed $\delta$ into a Pasternak foundation ($k$, $G$,
$\ell = \sqrt{G/k}$), with slope continuity at the contact edge $\rho = a$:

$$ p(\rho) = k\Big(\delta - \frac{\rho^2}{2R}\Big) + \frac{2G}{R}\quad(\rho < a), \qquad w(\rho) = \frac{a\ell}{R}\,\frac{K_0(\rho/\ell)}{K_1(a/\ell)}\quad(\rho > a), $$

$$ \delta = \frac{a^2}{2R} + \frac{a\ell}{R}\,\frac{K_0(a/\ell)}{K_1(a/\ell)}, \qquad F = \pi k a^2\Big(\delta - \frac{a^2}{4R}\Big) + \frac{2\pi G a^2}{R}. $$

Inside the contact, the shear layer adds a uniform $2G/R$ to the Winkler paraboloid. The friction arm
therefore lies between $8a/15$ (Winkler) and $2a/3$ (uniform pressure). Setting $G \to 0$ recovers
$F = \pi k a^4/(4R)$, and $a \ll \ell$ gives $F \approx 2\pi G a^2/R$.

For a skin of thickness $h$ on a rigid core, a Vlasov-type estimate (linear decay through the layer,
$\nu \approx 0.5$) gives $\ell \approx 0.29$-$0.33\,h$. A 3 mm TPU skin would then have $\ell \approx 1$ mm,
against our contact radius $a$ = 1.6-2.5 mm at 0.5-3 N. Codex's trigger, $\ell > 0.5a$, is 0.8-1.2 mm.

### 4.3 Friction torque against normal force (derived here)

Take a sphere of radius $R$ on a flat with a local law $p = c\,d^m$ and no lateral coupling. The contact
is the overlap disc, $a^2 = 2R\delta$, and

$$ F \propto R\,\delta^{m+1} \propto a^{2m+2} \;\Rightarrow\; a \propto F^{1/(2m+2)}, \qquad \bar r = \frac{\int p\,\rho\,dA}{\int p\,dA} \propto a, \qquad M = \mu F \bar r. $$

| Law | Where it applies | $a \propto F^{\,\cdot}$ | $\bar r/a$ | $\bar r$(3 N)/$\bar r$(0.5 N) |
|---|---|---|---|---|
| $m = 1$ | pads, Drake sphere, Newton, CSLC with the anchor dominant | 1/4 | 8/15 = 0.533 | 1.57 |
| $m = 3/2$ | CSLC contact element with a stiff anchor, $k_\ell = 0$ | 1/5 | 5π/32 = 0.491 | 1.43 |
| $m = 1/2$ | softening local law | 1/3 | 3π/16 = 0.589 | 1.82 |
| Hertz half-space | fully coupled elastic tip | 1/3 | 3π/16 = 0.589 | 1.82 |
| Pasternak | CSLC lattice | 1/2 at $a \ll \ell$, 1/4 at $a \gg \ell$ | 2/3 to 8/15 | depends on $a/\ell$ |

The $m = 1/2$ law reproduces the Hertz pressure shape $p \propto \sqrt{1 - \rho^2/a^2}$. With
$c = 2\sqrt2\,E^*/(\pi\sqrt R)$ it also reproduces Hertz's $a(F)$ exactly, and hence its $\bar r(F)$. A
torsion sweep therefore cannot separate lateral coupling from a nonlinear local law. The indentation
influence function, i.e. the neighbour displacement under a point load as in CSLC Fig. 2, can.

The measured anchors for this plot are as follows. Drake's law on the real_v1 tip is
$\bar r = 0.996$ mm $(F/1\,{\rm N})^{1/4}$ (2026-10-01). Our pads give $\bar r$ = 0.802 and 1.246 mm at
0.5 and 3 N, against 0.838 and 1.311 mm from the law (commit e8d3e950).

### 4.4 Which torque each paper exercises

- **Newton, Sec. V:** torsional friction about the contact normal, from a bar pivoting in a parallel-jaw
  grasp under gravity. This is the mechanism of our pinch-brake swing,
  $\cos\varphi = 2\mu N\,\bar r(N)/(m g d)$, so Fig. 4's $|\omega_y|(\mu, \text{width})$ map has a direct
  analogue on our rig.
- **CSLC, Sec. IV-C:** the rocking couple of distributed normal forces about an axis perpendicular to the
  pinch axis. Our pads also produce such a couple, since independent Winkler elements sit at different
  heights on the pad. Rerunning the book test with pads therefore separates the effect of a distributed
  normal law from that of $k_\ell$.
- **Our goal** (task-level, 2026-10-05): friction torque about the pinch axis, patch area and tangential
  friction under rolling or sliding. Only Newton's test addresses the first. Neither paper measures
  tangential friction under sliding.

## 5. Check of Codex's notes

The page checked is `docs/experiments/20261004-codex/20261004-contact_research_asides.html` (builder
`scripts/distributed_contact_page.py`), searched for CSLC, Newton, Pasternak and Winkler. The citation
lines in `20261004-distributed_contact_spec.html` were also checked.

| # | Claim (asides page) | Verdict | Evidence |
|---|---|---|---|
| 1 | CSLC "already solves a quasistatic equilibrium using warm-started damped Jacobi iterations" | Supported | CSLC p. 3, Sec. III-D, Eq. (10) |
| 2 | CSLC "has three-dimensional displacement, anchor springs and lateral springs" | Supported | Eqs. (1)-(4), p. 2 |
| 3 | "presliding friction saturates smoothly; the authors explicitly describe it as tangentially stateless and exclude kinetic sliding dynamics" | Supported | Eq. (8) and text, p. 3. The paper does not say how object sliding enters $\delta_t$ (Section 2.2, item 6) |
| 4 | CSLC "does not establish the proposed rolling shear-history model"; an explicit stiff lattice "is ... not an inherent CSLC requirement" | Supported | Quasistatic solve, no history state, no rolling test |
| 5 | "Drake's hydroelastic guide explicitly excludes true deformation and tangential compliance" | Not checkable here | Drake documentation; neither paper covers it |
| 6 | Newton: "contact reduction affected the torsional envelope" | Our measurement, not the paper's | The paper claims approximate preservation through moment matching (p. 3) and gives no error. Codex's static study at 0.2 mm indentation: 1.702 mN·m unreduced, 1.402 with default reduction, 1.601 with moment matching; at 0.4 mm, 8.289 against 9.657 mN·m. Moment matching is off by default and in our probe scripts (Section 3.2) |
| 7 | "u(r) ∼ exp(−r/ℓ) is a fitting hypothesis, not a universal elastic Green's function" | Supported, refined | In CSLC the exponential is the large-$r$ envelope of the lattice's $K_0$-type Green's function (Fig. 2, p. 3); an elastic half-space decays as $1/r$ |
| 8 | $K = K_nI + K_\ell L$ "can be factorized once, relaxed with a few Jacobi iterations, run as a GPU stencil" | Partly supported | CSLC uses this operator plus state-dependent contact and friction terms. It names the Jacobi solve as a cost driver (p. 4) and gives no iteration count. The contraction is about 0.89 per sweep at $k_\ell = 2k_a$ (about 20 sweeps per decade) and grows with $(\ell/s)^2$, so "a few" is unsupported |
| 9 | Pad spheres are Winkler elements, $K = 1/(t_c^2(1 - d_0)\,{\rm diagApprox})$ | Local model; consistent with MuJoCo's soft-constraint algebra at damping ratio 1 | Not in either paper |
| 10 | "Hydroelastic contact is also Winkler-type: the pressure ... depends only on the local field value" | Supported for Newton | $p = k_A\phi_A = k_B\phi_B$, linear in depth (p. 2) |
| 11 | Pads and the hydroelastic patch are "one class of model with the same per-area stiffness E/R" | Partly supported | The class is shared. $E/R$ holds for Drake's sphere and for the pads by construction. Newton's $k_h$ (N/m³) is a free modulus, equal to $E/R$ only by choice (our runs: $9.479\times10^8$ N/m³), and two compliant bodies take the series value |
| 12 | CSLC has "no tangential state by its authors' account" | Supported | p. 3 |
| 13 | "the pads and the hydroelastic patch correspond to CSLC's anchor springs alone, and CSLC's lateral springs are the $K_\ell L$ term" | Second half supported; first half partly | The lateral springs are $K_\ell L$ with the 4-neighbour Laplacian (Appendix A). The CSLC normal law is the anchor in series with a $k_c\phi^{3/2}$ element, and its anchor has a tangential part $\rho k_a$. The pads match only the normal anchor, where it dominates |
| 14 | Table row: Drake / Newton torsion "Distributed slip; Newton's contact reduction alters it" | Partly supported | As row 6 |
| 15 | Table row: CSLC normal law "Anchor springs", torsion "Distributed slip" | Not supported as written | The normal law is anchor plus Hertz-like contact in series. Torsion about the contact normal is neither derived nor tested; the rotational test is a pitch couple (Fig. 1; Sec. IV-C) |
| 16 | $\bar r$ grows by $6^{1/4} = 1.57$ under a Winkler law and $6^{1/3} = 1.82$ under a Hertz half-space; an exponent above 1/4 "counts as such a miss" | Arithmetic correct; inference needs a qualifier | An uncoupled law $p \propto d^{1/2}$ also gives 1.82, and CSLC's own local law gives 1.43 (Section 4.3). An exponent above 1/4 shows that linear Winkler pads miss, but not that coupling is needed |
| 17 | Plate-on-springs coupling length $(D/k)^{1/4}$ | Standard (Hetényi) | Not in either paper |
| 18 | Spec citations: CSLC "ICRA workshop paper/abstract, 2026", OpenReview `46BqR1wKqA`; Newton "Workshop paper, 2026", OpenReview `ogndqznZyY` | Supported by the workshop site | The asides page's arXiv link (2608.00263) is the CSLC paper. The spec spells the first Newton author "Rostel"; the PDF has Röstel |

## 6. Figures and tables an overview page can cite

| Paper | Item | Page | Content | Use |
|---|---|---|---|---|
| Newton | Fig. 1 | 2 | SDF grids; marching cubes on $g$; per-triangle forces; reduction 274 → 17 | pipeline strip, beside the pad discretization |
| Newton | Eqs. (1)-(4) | 2 | $g$, $\mathcal S$, $\phi_0$, $k_{\rm eff}$ | model box |
| Newton | Eqs. (5)-(8) | 3 | binning, aggregation, selection, wrench matching | reduction box |
| Newton | Table I | 3 | contacts and accumulated realtime factors, RTX 5090 | cost table, with the no-timestep caveat |
| Newton | Fig. 2 | 3 | the three benchmark scenes | none needed |
| Newton | Fig. 3 | 4 | visuotactile displacement maps on the Sharpa Wave hand | tactile use of the unreduced surface |
| Newton | Fig. 4 | 4 | $\lvert\omega_y\rvert$ at detachment against $\mu$ and contact width | torsional-friction task; analogue of our brake swing |
| CSLC | Fig. 1 | 2 | left: sphere decomposition; centre: spring network; right: book pitch and $\Delta z$ traces | centre panel for the model hierarchy; right panel for the rocking test |
| CSLC | Eqs. (3)-(8), (10) | 2-3 | anchor, lateral, contact, friction, equilibrium | model box |
| CSLC | Eq. (9) | 3 | stiffness calibration (unused) | identification route |
| CSLC | Fig. 2 | 3 | point-load decay, $\ell$ = 4.24 mm | Pasternak coupling length |
| CSLC | Figs. 3-4, Eq. (11) | 4 | MuJoCo squeeze-lift-hold near the Coulomb floor | grasp at low force |

**Figures worth redrawing for a comparison:**

1. **Model hierarchy** (new; borrow CSLC Fig. 1 centre and Newton Fig. 1). The levels are point contact
   (MuJoCo condim 3); per-geom torsion (condim 4, where a per-step $\mu_t = \mu\bar r(N)$ matched Drake
   within 1 % on 2026-10-01); local distributed Winkler (our pads, Drake and Newton hydroelastic); coupled
   distributed (Pasternak: the CSLC lattice); coupled with tangential history (the brush/Pasternak
   proposal); and continuum FEM. Annotate each level with its normal law, coupling length and tangential
   state, and with the torque its paper tested.
2. **Torque against force** (new; neither paper has one). Plot $\bar r = M/(\mu F)$ against $F$ from 0.5
   to 3 N on log-log axes, with the slopes of Section 4.3 (1/5, 1/4, 1/3) and the Pasternak curve for
   $\ell$ = 0.5, 1 and 2 mm. Overlay the Drake law, our pad points and Newton probe points.
3. **CSLC Fig. 2 redrawn** with the 4-neighbour lattice solution, the continuum $K_0$ curve and the Winkler
   limit (one loaded site), to show what lateral coupling adds and what our pads lack.
4. **Newton Fig. 4 redrawn** as the same $|\omega_y|$ map for pads, condim 4 and Drake on our brake-swing
   rig, if that run is made.
5. **CSLC Fig. 1 traces** (pitch and $\Delta z$ after a 3 N, 50 ms pulse), with a pads curve added, and a
   CSLC $k_\ell = 0$ curve if the code becomes available.

## 7. What these notes do not settle

1. **Whether CSLC's lateral springs change a grasp outcome.** Rerun the book pulse (0.3 kg, $\mu = 1$, 3 N
   for 50 ms) with our 1 mm pads on a two-finger pinch at matched squeeze. Pads held within ±0.7° and
   under 1 mm of drop would attribute the CSLC result to the distributed normal law alone.
2. **How CSLC couples object sliding into $\delta_t$.** This needs the authors' code or an answer from the
   authors. Until then CSLC cannot serve as a comparator for tangential friction or torsion.
3. **Newton torsion error with moment matching on our rig.** Add a `moment_matching` flag to
   `scripts/newton_pinch_probe.py` (line 156). Measure torsion capacity and spin onset at pad forces of
   0.5, 1, 2 and 3 N, with reduction off, on, and on with moment matching. A residual above 5 % at task
   forces means torsion-sensitive tasks should run unreduced (686 contacts on the pinch).
4. **Newton stiffness on oblique patches** (issue #3503). For the rolling task, log $c$ and $\phi_0$ per
   face and compare them with the projected-gradient form. A difference above 10 % over the patch means
   Newton's creep and stability margin are not comparable with the pads at equal $dt$, although forces
   agree.
5. **Coupling on the printed tip.** Use Codex's influence-function measurement: a 0.5 mm probe at 0.5, 1
   and 3 N, with neighbour displacement read by DIC. The torsion sweep alone cannot decide between coupling
   and a nonlinear local law. If the influence decays within one pad spacing ($\ell < 1$ mm) while
   $\bar r$ grows faster than $F^{1/4}$, a per-sphere nonlinear law is the cheaper fix, for example through
   solimp's penetration-dependent impedance (untested).
6. **Newton throughput on our scene.** Table I has no timestep. The comparison needs Newton's world-steps/s
   at 1k-8k worlds on our fixture, with `scripts/newton_scaling.py` and moment matching stated, which is
   the next item on the contact bed.

## Appendix A. Check of CSLC Fig. 2

The solve takes the normal components of Eqs. (3)-(4) on a 15 × 15 grid with free edges, $s$ = 3 mm,
$k_a$ = 100 N/m and $k_\ell$ = 200 N/m, with 58 mN on the centre sphere. A load normal to a flat lattice
produces no in-plane displacement, so the normal components decouple.

```python
import numpy as np
n, ka, kl, F = 15, 100.0, 200.0, 0.058
K = np.zeros((n*n, n*n)); idx = lambda i, j: i*n + j
for i in range(n):
    for j in range(n):
        a = idx(i, j); K[a, a] += ka
        for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:      # add the diagonals for the 8-neighbour case
            if 0 <= i+di < n and 0 <= j+dj < n:
                K[a, a] += kl; K[a, idx(i+di, j+dj)] -= kl
f = np.zeros(n*n); f[idx(7, 7)] = F
d = np.linalg.solve(K, f).reshape(n, n)                    # metres
```

| r (mm) | 4-neighbour $\lvert\delta_n\rvert$ (mm) | Fig. 2 reading (mm) | $(P/2\pi G)K_0(r/\ell)$ (mm) |
|---|---|---|---|
| 0 | 0.0917 | ≈ 0.09 | singular |
| 3 | 0.0307 | ≈ 0.03 | 0.0301 |
| 6 | 0.0116 | ≈ 0.012 | 0.0110 |
| 9 | 0.0048 | ≈ 0.005 | 0.0045 |
| 12 | 0.0021 | ≈ 0.0025 | 0.0020 |

With 8 neighbours at equal $k_\ell$ the centre deflection is 0.047 mm and the continuum length is
$s\sqrt{3k_\ell/k_a}$ = 7.35 mm, neither of which matches the figure.
