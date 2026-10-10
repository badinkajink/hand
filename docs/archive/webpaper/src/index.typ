#import "template.typ": conf, callout, det
#show: conf.with(title: "MorphoHand — Project Overview", current: "index")

#html.elem("div", attrs: (class: "hero"), {
  [= MorphoHand]
  html.elem("p", attrs: (class: "sub"),
    "Co-designing a 3-finger hand, its grasps, and its in-hand control entirely in simulation — and the hardware to validate it.")
})

== Abstract

MorphoHand is a simulation-first stack for *dexterous co-design*: it searches the
space of hand *morphologies*, synthesizes high-quality *grasps* on each candidate,
and learns *in-hand manipulation* policies — all in simulation, with a reconfigurable
physical hand to validate the predictions. The work spans three coupled pillars. The
*morphology + grasp* pillar formulates grasp synthesis as a non-differentiable
cross-entropy search against a physics-based objective, and morphology search as a
quality-diversity illumination of the design space, structured as a *bi-level*
optimization. The *RL* pillar learns to lift a flat-laying cylinder and reorient it to
vertical in-hand on the optimized 9-DOF morphology, and confronts the *policy-switching*
problem that arises when manipulation is decomposed into chained skills. The *hardware*
pillar tests the central scientific claim — the *object-set-specificity hypothesis* —
that an optimal morphology is specific to the object set it was optimized for.

#callout(tag: "The thesis", kind: "note")[
  You can co-design a hand and its control in simulation, validate the *ranking* of
  morphologies on hardware, and show that the optimum is *object-set specific* — a
  hand optimized for one object distribution is measurably better on that
  distribution than a hand optimized for another.
]

== Contributions

- A *frozen-scene protocol* that removes morph joints from the evaluation scene so a
  candidate morphology cannot drift during grasp scoring — making the inner objective
  well-defined and reproducible.
- A *contact-map decoupling* of the bi-level coupling: precomputed object contact
  regions turn the inner grasp solve into a fast IK-like reach, avoiding a full
  from-scratch grasp synthesis per morphology.
- A *quality-diversity* morphology search (MAP-Elites / CMA-ME, optionally GP-surrogate
  assisted) that *illuminates* the design landscape rather than returning a single
  optimum — directly producing the diverse morphology set a hardware oracle needs.
- A *two-policy RL decomposition* (lift → reorient) for in-hand reorientation on the
  optimized morphology, with a precise diagnosis of the *seam* between chained policies
  as an initiation-set / terminal-set mismatch, mapped onto the skill-chaining
  literature.
- A *reconfigurable hardware platform* and an experimental protocol to test sim-to-real
  rank correlation and the object-set-specificity hypothesis.

== The three pillars

#html.elem("div", attrs: (class: "cards"), {
  html.elem("a", attrs: (href: "morphology.html", class: "card"), {
    [=== Morphology & Grasp Optimization]
    html.elem("p", attrs: (:),
      "Bi-level co-design: CEM grasp synthesis (inner) under a physics objective, MAP-Elites / CMA-ME morphology search (outer), the GWS ε-metric, and contact-map decoupling.")
  })
  html.elem("a", attrs: (href: "rl.html", class: "card"), {
    [=== RL Manipulation]
    html.elem("p", attrs: (:),
      "Lift-and-reorient in-hand on the 9-DOF morphology: the MDP, reward design, the two-policy split, and the policy-switching problem with the full branch A-F analysis.")
  })
  html.elem("a", attrs: (href: "hardware.html", class: "card"), {
    [=== Hardware Validation]
    html.elem("p", attrs: (:),
      "The reconfigurable physical hand: the object-set-specificity hypothesis, the platform and design-space parameterization, and the sim-to-real validation protocol.")
  })
})

== Background: three literatures this work sits between

#det([Grasp quality and the Grasp Wrench Space], kind: "related work")[
  A grasp is a set of contact points each applying a force inside its friction cone.
  The *Grasp Wrench Space (GWS)* is the convex hull of all object-frame wrenches (6-D
  force+torque) the contacts can jointly produce. The Ferrari–Canny *ε-metric*
  is the radius of the largest origin-centered ball inside the GWS — the worst-case
  disturbance the grasp can resist. ε > 0 is exactly the *force-closure*
  condition. Analytic metrics are millisecond-fast but assume known contacts and
  idealized point-contact Coulomb friction; differentiable approximations (GWB) enable
  gradient-based synthesis; simulation-based metrics (Grasp'D) correlate better with
  real success at higher cost. MorphoHand uses a physics-rollout objective for the
  inner loop and the GWS ε-metric as the morphology-quality signal.
]

#det([Morphology co-design is inherently bi-level], kind: "related work")[
  The *outer* problem asks which hand shape θ maximizes performance; the *inner*
  problem asks, for a fixed θ, what the best grasp q is. They are coupled — the
  best grasp for θ\_1 is generally not best for θ\_2 — so every outer step
  must (re)solve the inner grasp problem. Approaches trade off how exactly they solve
  this: *nested bi-level* (robust to contact discontinuities, but a full inner solve per
  sample), *single-level gradient* co-design (efficient within a contact mode, but the
  gradient is undefined when the contact set changes), *evolutionary* (black-box,
  robust, sample-inefficient), and *contact-map decoupling* (precompute object contact
  regions, reduce the inner solve to IK). MorphoHand takes the contact-map route and
  warm-starts the inner CEM across neighboring morphologies.
]

#det([Quality-diversity, not single-objective optimization], kind: "related work")[
  Standard optimization returns one best solution; morphology design wants more. The
  landscape is *multimodal* (structurally different hands can be near-equally good), the
  optimum is *object-set dependent* (we want to map that dependence), and a hardware
  oracle should be shown a *diverse* set of design points. *MAP-Elites* maintains an
  archive over a behavioral-descriptor grid, keeping the best solution per cell — an
  *illumination* of the design space rather than a single optimum. *SAIL* wraps it in a
  GP surrogate to cut expensive inner-loop calls 50–80%; *CMA-ME* replaces Gaussian
  mutation with CMA-ES for better coverage, and gradient-augmented variants use inner-loop
  gradients where the contact set is stable. The archive doubles as the hardware-selection
  criterion: one representative elite per occupied cell.
]

#det([Chaining learned skills: the policy-switching problem], kind: "related work")[
  Decomposing manipulation into sub-skills introduces a *seam*: the terminal-state
  distribution of one policy may fall outside the *initiation set* of the next, so the
  successor is out-of-distribution the moment it takes over. The literature offers
  *terminal-state regularization* (an adversarial discriminator pulls a policy's terminal
  states into the next policy's initiation set), *transition feasibility functions*
  (Sequential Dexterity: a value-like net both fine-tunes the predecessor and times the
  switch), *transition policies* (a learned bridge), and *action-smoothness* regularizers
  (CAPS) for the deployment-time discontinuity. The RL pillar diagnoses MorphoHand's
  lift→reorient seam in exactly these terms — and argues a friction-held object makes
  the seam far less forgiving than a legged robot's.
]

#callout(tag: "Provenance", kind: "note")[
  Synthesized from the project's working documents: the two LaTeX papers (`paper/`,
  `hand_paper/`), the `docs/` MkDocs corpus, and the RL research logs (`docs/rl/`). The
  raw logs remain the source of truth; this site is the readable, media-rich synthesis.
  Tip: the *"Full paper view"* button (bottom-right) expands every collapsible at once.
]
