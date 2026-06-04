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

== Results (to expand)

#callout(tag: "Stub", kind: "warn")[
  Results media and numbers to port from `paper/main.tex` §Experiments: the
  pre-grasp baselines, the 1000-morphology combined sweep, the foundational-pose
  adaptation ablation, the eval-suite (contact-target patches vs. baseline 9-D CEM
  across the object set), the frozen-scene quantification, and the power-drill
  pivot-to-down case study. The eval-suite comparison videos above are representative;
  the full per-object grid lives in `docs/overview_gifs.md`.
]

#refbox[
  *Sources.* Ferrari & Canny, \_Planning optimal grasps\_, ICRA 1992. Mouret & Clune,
  \_Illuminating search spaces (MAP-Elites)\_, 2015. Gaier et al., \_SAIL\_, 2017. Fontaine
  et al., \_CMA-ME\_, 2020. Full method + experiments: `paper/main.tex`, `hand_paper/main.tex`.
]
