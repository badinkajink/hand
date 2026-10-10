#import "template.typ": conf, callout, det, refbox
#show: conf.with(title: "Hardware Validation — MorphoHand", current: "hardware")

= Hardware Validation

Simulation-optimized morphologies make a claim only hardware can test. The hardware
pillar is the reconfigurable physical hand that closes the loop — and the experimental
protocol that asks whether the simulation's *ranking* of morphologies survives contact
with reality, and whether the optimum is *object-set specific*.

== The object-set-specificity hypothesis

The central scientific conjecture: *the optimal hand morphology depends on the
distribution of objects to be grasped*, strongly enough to matter in practice. Nobody
would use the same hand for picking strawberries as for assembling electronics — but this
has not been systematically verified with hardware. Simulation suggests different objects
prefer different contact triangulations, finger-spread angles, and reach distances; this
work tests it *directly*:

#callout(tag: "The experiment in one line", kind: "note")[
  Run quality-diversity optimization *separately* over different object distributions,
  extract the best morphology for each, then *cross-evaluate every morphology on every
  distribution* — in simulation *and* on hardware. If the specialization effect is real,
  narrow-distribution applications (industrial, agricultural) should invest in
  task-specific hand design over general-purpose anthropomorphic hands.
]

== The MorphoHand platform

MorphoHand is a 3-finger, *18-DOF* reconfigurable platform built as a *morphology
evaluation substrate*, not an operational end-effector: motorized gantry finger bases,
telescoping proximal phalanges, and hot-swap fingertips let one physical device
instantiate a whole family of searched morphologies.

=== Degrees of freedom: a clean morphology/control split

The 18 DOFs (6 per finger, fingers i ∈ {T, I, M}) divide into two roles — and that
split is exactly what makes the bi-level optimization tractable, and what ties all three
pillars together.

#table(
  columns: (1fr, auto, auto, auto),
  table.header([*DOF*], [*Symbol*], [*Range*], [*Role*]),
  [Base X (lateral)], [x\_i], [± 60 mm], [Morphology],
  [Base Y (fore–aft)], [y\_i], [± 40 mm], [Morphology],
  [Proximal length], [l\_i], [40–90 mm], [Morphology],
  [Yaw (abd/add)], [ψ\_i], [± 45 °], [Control],
  [MCP flex/ext], [q\_i^p], [0–110°], [Control],
  [PIP flex/ext], [q\_i^d], [0–100°], [Control],
)

- *Morphological DOFs* (9: x\_i, y\_i, l\_i) define the hand's *shape* — the outer
  optimizer's variable θ. They are lead-screw driven (high holding torque, slow), so
  they are *set per design*, not actuated on the fly.
- *Control DOFs* (9: ψ\_i, q\_i^p, q\_i^d) define the joint angles during a grasp — the
  inner optimizer's variable q, *and the very 9 DOF the RL pillar's policies control*.
  Fast (sub-second), adjusted per grasp.

#callout(tag: "The through-line", kind: "note")[
  θ (9 morphology DOF) is what the *quality-diversity outer loop* searches; q (9
  control DOF) is what the *CEM inner loop* optimizes *and* what the *RL policies* actuate.
  The same physical 9+9 split is the optimization's bi-level structure, the hardware's
  reconfigurability, and the RL action space — one design choice running through every
  pillar.
]

#det([Mechanism detail (to port)], kind: "stub")[
  Port from `hand_paper/main.tex` §"Detailed Mechanism Descriptions": the 2-DOF Cartesian
  gantry (palm plane), the telescoping proximal phalanx, the servo-driven yaw/MCP/PIP
  joints, and the palm structure / wrist / sensing (§"Palm Structure, Wrist, and
  Sensing"), plus the full design-space parameterization (§"Design Space
  Parameterization").
]

== Experimental protocol

Five experiments, to port in full from `hand_paper/main.tex` §"Experimental Methodology":

#det([The five experiments], kind: "method")[
  + *Sim-to-real rank correlation* — does simulation's ordering of morphologies match the
    physically measured ordering? (The core sim-first validity check.)
  + *Object-set specificity* — cross-evaluate per-distribution optima on all
    distributions; measure whether specialization is real and significant.
  + *Archive coverage and surrogate efficiency* — how well MAP-Elites / SAIL illuminate
    the design space, and the compute saved by the GP surrogate.
  + *Morphological DOF contribution* — ablate which morphology DOFs (x, y, l) matter
    most for grasp quality.
  + *Hardware repeatability* — measurement noise floor of the physical platform.
]

== Results

#callout(tag: "Stub", kind: "warn")[
  CAD renders, platform photos, and bench results to port from `hand_paper/main.tex`
  (§Hardware Design, §Discussion: Expected Results) as hardware data lands. The
  formulation and protocol above are complete; the empirical section is pending bench
  measurements.
]

#refbox[
  *Source.* `hand_paper/main.tex` — _A Reconfigurable Hand for Validation of
  Simulation-Optimized Morphologies_. Related background (GWS, co-design, quality-diversity)
  is shared with the #link("morphology.html")[Morphology & Grasp Optimization] pillar.
]
