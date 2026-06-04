#import "template.typ": conf, media, fig, callout, det, refbox
#show: conf.with(title: "Morphology & Grasp Optimization — MorphoHand", current: "morphology")

= Morphology & Grasp Optimization

This pillar answers two coupled questions at once: *what hand shape* grasps a target
object set best, and *what grasp* each shape should use. That coupling is the defining
structure of the problem — it is a *bi-level optimization* — and most of the
engineering here is about making the inner solve cheap and well-defined enough that the
outer search is tractable.

== The bi-level structure

Let θ ∈ ℝ^d parameterize the hand morphology (finger placements, link lengths)
and q ∈ ℝ^9 the finger control (three joints per finger). The grasp quality of a
morphology on an object is the value of its *best* grasp, so the problem nests an inner
maximization inside an outer one:

$ theta^* = arg max_theta EE_(o ~ cal(O)) [ underbrace(max_q Q(theta, q, o), "inner: best grasp for this hand") ]\, $

where 𝒪 is the target object distribution and Q is a grasp-quality measure.
The two levels are *coupled*: arg max\_q Q(θ\_1, ·) ≠ arg max\_q Q(θ\_2, ·), so every outer proposal θ requires (re)solving the inner grasp problem.
How exactly to pay that cost is the central design choice.

#det([The space of co-design strategies (and what we chose)], kind: "related work")[
  *Nested bi-level* runs a full inner solve per outer sample — robust to contact
  discontinuities (the inner solver may jump to a different contact set when the
  morphology changes) but expensive. *Single-level gradient* co-design treats (θ, q) as one variable and steps both with gradients — efficient *within* a contact mode,
  but the gradient is undefined when the optimal contact set changes (three-finger
  → two-finger). *Evolutionary* methods are black-box and robust but
  sample-inefficient (a 100-morphology population × 100 inner calls × 10
  objects = 100k sim calls per generation). *Contact-map decoupling* precomputes
  object contact regions so the inner solve becomes a fast IK-like reach.

  MorphoHand uses *nested bi-level with contact-map decoupling and warm-starting*: the
  inner CEM is cheap (∼2 min), warm-started across neighboring morphologies, and
  optionally guided by precomputed contact targets.
]

== The grasp-quality objective

The morphology-quality signal is the *Grasp Wrench Space ε-metric*, but the
inner-loop search optimizes a richer *physics-rollout* objective that captures what a
static metric cannot — whether a grasp survives motion.

#det([Grasp Wrench Space and the ε-metric], kind: "background")[
  Each contact applies a force inside its friction cone; the *grasp wrench* is the 6-D
  force+torque it exerts about the object's center of mass. The *GWS* is the convex hull
  of all jointly achievable wrenches. The *Ferrari–Canny ε-metric* is the radius
  of the largest origin-centered ball inscribed in the GWS:
  $ epsilon = min_(norm(w) = 1) max_(f in "FC") w^top W(f)\, $
  i.e. the worst-case unit disturbance direction's maximum resistible magnitude. ε > 0 is the *force-closure* condition — the hand can resist disturbances in every
  direction. It is millisecond-fast given contacts, but assumes known contact locations
  and idealized point-contact Coulomb friction; differentiable approximations (GWB) make
  it usable as a gradient-providing inner oracle.
]

The inner physics objective J(θ, q) is a weighted sum accumulated over a
*settle → lift → (pivot) → hold* rollout — rewarding lift and contact,
penalizing every mode of instability:

$ J(theta, q) = w_"lift" L + w_"contact" C + w_"persist" P_"all" + w_"minp" P_"min"
  - w_"dist" D - w_"drop" Z_(arrow.b) - w_"xy" Delta_(x y) - dots.c $

#det([What every term in J means], kind: "derivation")[
  - L — peak lift of the object center of mass, w\_lift(z\_peak - z\_before),
    the core reward (w\_lift = 35).
  - C — number of tip–object contact pairs at end-of-settle; D = ‖x\_f - x\_obj^closest‖̄ — mean closest-point fingertip-to-object distance (attract tips
    before contact resolves).
  - P\_all, P\_min — all-three-fingers and *minimum* per-finger contact
    persistence over the dynamic phase. The minimum-persistence weight is raised to 2.4
    on the screwdriver sweep to *suppress two-finger grasps* that lift well but drop
    during hold.
  - Stability penalties — object XY drift Δ\_xy, yaw drift Δ ψ\_obj,
    axis tilt Δ θ\_axis, end-of-hold velocity norm(ẋ\_obj), and
    post-peak drop Z\_↓ — each gates how far an unstable grasp may move the
    object off-plan.
  - Finger-drift penalties — persistence imbalance imb(P\_f) and mean joint/yaw drift
    |Δ q\_f|̄ — keep the grip from creeping.

  The three opt-in grasp-specification methods (below) add their own terms with weights
  that default to zero, so they compose *additively* on top of this baseline.
]

#callout(tag: "Why a rollout, not just ε", kind: "note")[
  A snapshot ε at settle can be high for a grasp that *falls apart in motion*.
  The persistence and drift terms distinguish "tips touching at settle" from a grasp
  that actually holds through the lift — which is what hardware cares about.
]

== The frozen-scene protocol

A subtle failure mode: a candidate morphology is encoded as joint values on *morph
joints* in the scene, and those joints can *drift* during a dynamic rollout, so the hand
you score is not the hand you proposed. The *frozen-scene protocol* patches the
keyframe's qpos with θ and then *removes the morph joints* from the evaluation
scene entirely — the morphology is baked into fixed geometry. This makes J(θ, ·)
well-defined and reproducible. (It is load-bearing enough that the RL pillar reuses the
same `freeze_scene_for_eval` discipline.)

== Inner loop: cross-entropy grasp synthesis

For a fixed scene, keyframe, and morphology, the 9-D finger control q is optimized by a
textbook *cross-entropy method (CEM)*.

#det([The CEM loop, and why CEM here], kind: "method")[
  Initialize μ ← q\_init (the keyframe ctrl), σ ← σ\_init 𝟏; for
  t = 1, …, T:
  + sample q\_i ∼ 𝒩(μ, diag(σ^2)), i = 1, …, N;
  + clip each q\_i to MJCF actuator bounds;
  + evaluate s\_i = J(q\_i) via a settle→lift→pivot→hold rollout;
  + select elites ℰ = top ρ N by s\_i;
  + refit μ ← mean(ℰ), σ ← std(ℰ) + σ\_floor;
  + track best-so-far (q^∗, s^∗).

  Production settings: N = 24–160, ρ = 0.2–0.25, σ\_init = 0.1–0.2,
  T = 12–120; variance floor σ\_floor = 10^-3 stops a degenerate early
  elite set from zeroing a coordinate.

  *Why CEM.* (1) The contact landscape is *non-smooth* — a small finger-angle change can
  flip a finger in/out of contact, jumping the reward; CEM needs no differentiability.
  (2) It scales *linearly* — each iteration is N independent rollouts (∼50 eval/s on 16
  cores; a 120 × 52 run finishes in ∼2 min). (3) It is *cold-start robust* — with
  σ\_init = 0.2 over actuator units in [-1.5, 1.5] the first iteration spans the
  local control box, so it can start from a keyframe only *near* a viable grasp.

  *Trajectory variant.* The same loop generalizes to a piecewise-linear control sequence
  over P phases (default P = 4), a 9P-D search — used when the finger pose must
  change mid-trajectory (lift-and-reorient, pre-grasp → pinch).
]

=== Three grasp-specification methods

On top of the baseline objective, three opt-in methods steer *where and how* the hand
grasps: *contact-target patches* (reward fingertips landing inside precomputed object
contact regions), *synergy / eigengrasp* (search a low-D synergy subspace), and an
explicit *trajectory force-closure* term. They share the additive-objective design so any
subset can be combined.

#media("assets/screwdriver_medium_flat__baseline.mp4", label: [Baseline 9-D CEM.],
  caption: [Flat screwdriver, baseline objective — for comparison with the contact-map run below.])
#media("assets/screwdriver_medium_flat__contact_map.mp4", label: [Contact-map patches.],
  caption: [Same object and morphology, with contact-target patches steering the fingertips into precomputed high-GWS regions.])

== Outer loop: quality-diversity morphology search

The outer search is *not* single-objective. We want to *illuminate* the design space —
map how grasp quality varies across morphology styles — both because the landscape is
multimodal and because the hardware oracle should see a *diverse* set of designs.

#det([MAP-Elites, SAIL, and CMA-ME], kind: "method")[
  *MAP-Elites* maintains an archive over a grid of behavioral descriptors (e.g. mean
  fingertip spread, palm-normalized reach), storing the best solution per cell:
  + initialize the archive with random morphologies (evaluate quality q + descriptors
    b);
  + repeat: select an elite, mutate it, evaluate the offspring, and place it in cell
    b, replacing the occupant iff q is higher.

  The result is an *illumination*: even cells a single-objective search would never visit
  are filled with the best morphology *for that descriptor region*. That archive is
  directly the hardware-selection criterion (one elite per occupied cell) and the test
  instrument for object-set specificity (compare archives computed over 𝒪\_A vs.
  𝒪\_B).

  *SAIL* wraps MAP-Elites in a GP surrogate trained on (morphology, mean GWS ε)
  pairs, screening offspring before committing to a full inner-loop solve — 50–80% fewer
  expensive evaluations. *CMA-ME* replaces Gaussian mutation with CMA-ES for better
  coverage and peak quality; where the contact set is stable, inner-loop gradients
  (through the differentiable GWS and forward kinematics) augment the offspring, falling
  back to evolution at contact boundaries.
]

=== Morphology sampling

Candidates are drawn by uniform perturbation around a base hand: propose θ' = θ\_base + ε with ε\_i ∼ 𝒰(-δ\_i, δ\_i), clip to bounds,
reject rounded duplicates, repeat until N unique samples (the base is always first).
The production sweep uses bounds x, y ∈ [-0.03, 0.03] m, ℓ ∈ [0, 0.035] m and
half-widths (δ\_x, δ\_y, δ\_ℓ) = (0.012, 0.012, 0.012) m. Each sampled
θ is materialized as a *frozen* scene before evaluation.

== Results

Five preliminary result blocks. The headline: the pipeline runs end-to-end on CPU
MuJoCo at a usable cadence (1000 morphologies in 762 s), keyframe choice *alone* reshapes
the quality landscape before any morphology variation, per-morphology pre-grasp
adaptation buys +22% feasibility, and contact-target patches turn a near-failure (the
long prism) into a reliable grasp while leaving already-easy benchmarks unchanged.

=== Pre-grasp baselines (fixed base morphology)

CEM (population 48, 16 iterations, two-seed best) on each medium-screwdriver keyframe.
Even with the morphology fixed, the three keyframes produce distinct quality
profiles — `open_90vertical` favors deeper lifts but fewer contact pairs.

#table(
  columns: (auto, auto, auto, auto),
  table.header([*Keyframe*], [*Best score*], [*Best lift (m)*], [*Best contacts*]),
  [`open_flat`], [5.70], [0.0496], [3],
  [`open_vertical`], [6.91], [0.0501], [6],
  [`open_90vertical`], [7.09], [0.0696], [5],
)

=== 1000-morphology combined sweep

All three keyframes evaluated on every sampled morphology, per-morphology pre-grasp
adaptation on, 1000 samples in *762 s* (CPU MuJoCo). Feasibility under the production
gate (mean tip-distance ≤ 0.022 m, ≥ 2 contacts):

#table(
  columns: (auto, auto, auto, auto, auto),
  table.header([*Keyframe*], [*Total*], [*Feasible*], [*Rate*], [*Mean score*]),
  [`open_flat`], [1000], [994], [0.994], [5.48],
  [`open_vertical`], [1000], [884], [0.884], [3.94],
  [`open_90vertical`], [1000], [696], [0.696], [2.46],
)

Two stable findings: the *same* morphology distribution is far easier to satisfy in
`open_flat` (99.4%) than `open_90vertical` (69.6%); and 60.3% of morphologies are
*simultaneously* feasible across all three keyframes (top-5 multitask candidates score
≥ 6.36 on their worst keyframe). So the design space is rich enough to support
pose-specific specialization while still leaving a broad robust family.

=== Foundational-pose adaptation ablation

500-sample cube + 3-prism sweep under all five adaptation modes (identical seed, order,
gates). Gain is feasible-count delta vs. the fixed-pose baseline (out of 2000):

#table(
  columns: (auto, auto, auto, auto, auto),
  table.header([*Mode*], [*Feasible*], [*Gain*], [*Extra s*], [*Gain/s*]),
  [`none` (baseline)], [1442], [—], [—], [—],
  [`interval-open`], [1465], [+23], [165], [0.14],
  [`interval-initial-fp` (i=50)], [1753], [+311], [243], [1.28],
  [`sparse-per-morph` (5)], [1758], [+316], [223], [*1.42*],
  [`local-perturbation`], [1800], [*+358*], [739], [0.48],
)

Adaptation matters (+22% feasibility); *warm-starting* matters (re-running CEM from the
current best pose recovers +311, restarting from a uniform sample only +23); and
`sparse-per-morph` is the cost-effectiveness winner (+316 at +223 s), while
`local-perturbation` buys the absolute maximum at 3× the cost.

=== Eval suite: contact-target patches vs. baseline 9-D CEM

Matched budget (960 evaluations/seed, 3 seeds, frozen scenes). Each best grasp is
re-scored under the *baseline* objective to give an *oracle score* comparable across
methods. Δ is contact-map minus baseline.

#table(
  columns: (auto, auto, auto, auto),
  table.header([*Benchmark*], [*baseline*], [*contact-map*], [*Δ*]),
  [`prism`], [2.31 ± 1.10], [*5.57 ± 0.08*], [*+3.26*],
  [`screwdriver_medium_flat`], [5.52 ± 0.22], [5.85 ± 0.31], [+0.33],
  [`cube`], [6.70 ± 0.17], [6.98 ± 0.19], [+0.28],
  [`screwdriver_medium_90vertical`], [6.20 ± 0.52], [6.27 ± 0.57], [+0.07],
  [`screwdriver_small_flat`], [−0.06 ± 0.00], [−0.07 ± 0.01], [−0.01],
  [`power_drill_short_proximal`], [7.89 ± 0.47], [7.84 ± 0.42], [−0.05],
  [`screwdriver_medium_vertical`], [5.83 ± 0.17], [5.72 ± 0.01], [−0.11],
  [`power_drill`], [7.37 ± 0.62], [6.20 ± 0.48], [−1.17],
  [*Mean Δ*], [], [], [*+0.32*],
)

Mean Δ +0.32, median +0.03, wins 4/8.

#fig("assets/scores.png", label: [Per-benchmark oracle scores.],
  caption: [Baseline vs. contact-map (error bars min/max over 3 seeds). Contact-target
    patches retain baseline quality on six benchmarks and convert the long-prism
    near-failure into a stable grasp.])
#fig("assets/deltas.png", label: [Contact-target Δ over baseline.],
  caption: [The eval-suite headline average is almost entirely carried by the prism.])

#det([Reading the eval suite: the prism win is the whole story], kind: "extended results")[
  *The prism win.* On the long prism the baseline gets only L = 0.031 m lift,
  all-three-fingers persistence 0.26, high variance (σ = 1.10). Contact-map jumps to
  L = 0.049 m, persistence 1.00, σ = 0.08. The mechanism is geometric: the prism is long
  along y with 22 mm extent along x, so the baseline's closest-point distance term is
  *flat in y* and CEM has no gradient toward an opposed-finger pinch; three target
  patches (thumb −x, index +x-forward, middle +x-back) install a direction-aware
  attractor and the optimizer converges to the pinch reliably.

  *Most benchmarks are ties.* Six of eight land within ±0.35 — the AABB-distance signal
  is already informative there, so the prior neither helps nor hurts.

  *Two slightly lose.* `power_drill` (−1.17) and `screwdriver_medium_flat` (−0.28 unfrozen,
  *+0.33 once frozen*) lose where the authored patches disagree with the unconstrained
  optimum — the cost of specifying intent when the patches are not quite right.

  *Both fail the small screwdriver.* An 8 mm shaft plus a 110 mm hand-to-target
  displacement at the open keyframe leaves both methods at zero contacts/lift — an
  included known floor.
]

=== Quantifying the frozen-scene protocol

Running the suite with and without frozen-scene enforcement (else identical) shows the
protocol is not cosmetic — several per-benchmark numbers shift *sign*:

#table(
  columns: (auto, auto, auto, auto, auto),
  table.header([*Benchmark*], [*Δ unfrozen*], [*Δ frozen*], [*base unfr.*], [*base fr.*]),
  [`prism`], [+3.16], [*+3.26*], [1.00], [2.31],
  [`screwdriver_medium_flat`], [−0.28], [*+0.33*], [5.67], [5.52],
  [`power_drill_short_proximal`], [+0.30], [−0.05], [8.46], [7.89],
  [`power_drill`], [−0.33], [−1.17], [7.39], [7.37],
  [`cube`], [+0.18], [+0.28], [6.64], [6.70],
)

`screwdriver_medium_flat` flips −0.28 → +0.33 (the apparent contact-map *loss was a drift
artifact*), and `power_drill_short_proximal`'s baseline falls 8.46 → 7.89 once the morph
DOFs are pinned — drift had been quietly rearranging the fingertips into a tighter grip.
*All results in this work post-date the protocol fix*; pre-fix numbers (e.g. the
single-scene drill runs at 8.5+) are suspect for cross-method ranking.

=== Case study: power-drill pivot-to-down (the hardest scene)

#det([Three failure modes, and why the keyframe is the blocker], kind: "case study")[
  The drill must be grasped at `open_flat` and rotated 90° to down-pointing while held —
  the only benchmark where baseline CEM consistently fails. Three diagnosed modes:

  + *Wrong pivot deltas* — early runs drove only one of three wrist axes; setting all of
    `pivot-delta-rx/ry/rz` to rotate the wrist to (0, 0, π/2) recovered the intended
    3-axis motion.
  + *Contact persistence at zero* — corrected pivot still gave all-finger persistence 0:
    tips touched at settle but lost contact at dynamic step 1. CEM had chosen a pose
    *more open* than the keyframe; the baseline drift penalties measure motion *during*
    the rollout, not deviation from the keyframe at start. An anchor penalty
    (w = 2.0) pulled the pose back to within 0.06–0.16 of the keyframe — but persistence
    stayed 0.
  + *The keyframe is not a grip pose* — the trajectory force-closure metric (K = 8 samples)
    read max Q₁ = 2.0 (no-contact cap) with min F = 0 at 5/6 seeds: no q in the σ = 0.15
    basin around `open_flat` has a closing grasp, because `open_flat` places the
    fingertips *tangent* to the barrel (index PIP straight, pointing past the drill).

  *Cross-method check:* Lightning Grasp on the same scene (31 candidates in 18.4 s) also
  produces *every* grasp with persistence 0 (best 3.8 cm lift, 19° partial pivot) — a
  geometrically different pipeline converging to the same non-gripping family confirms the
  keyframe, not the search method, is the blocker.

  *Resolution (open):* author an `open_flat_gripping` keyframe with inward-flexed PIPs, or
  add a CEM start-control override decoupled from the keyframe. The run-15 objective shape
  (anchor + trajectory-FC + boosted min-persistence) was correct, just not sufficient on
  this keyframe.
]

#refbox[
  *Sources.* Ferrari & Canny, _Planning optimal grasps_, ICRA 1992. Mouret & Clune,
  _Illuminating search spaces (MAP-Elites)_, 2015. Gaier et al., _SAIL_, 2017. Fontaine
  et al., _CMA-ME_, 2020. Full method + experiments: `paper/main.tex`, `hand_paper/main.tex`.
]
