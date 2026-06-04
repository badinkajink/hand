#import "template.typ": conf, media, fig, callout, det, refbox
#show: conf.with(title: "RL Manipulation — MorphoHand", current: "rl")

= RL Manipulation: lift-and-reorient in-hand

The RL pillar learns one concrete task on the optimized 3-finger MorphoHand: take a
screwdriver-shaped cylinder lying *flat* on the ground, pick it up, and reorient it to
*vertical* — entirely in-hand, using only the 9 finger DOF, with no floor contact during
the rotation. It is a deliberately hard test of dexterous control on an under-actuated,
sampling-optimized morphology.

#media("assets/handoff_demo.mp4",
  label: [The end-to-end demo.],
  caption: [Policy A picks up the flat-laying cylinder and holds it stable (3 s); Policy
    B then reorients it toward vertical (4 s). Both policies run live in simulation,
    concatenated in one rollout.])

== The control problem

We pose finger control as a discrete-time MDP (𝒮, 𝒜, P, r, γ) solved with
PPO. The action a ∈ ℝ^9 is a *residual* on a scripted finger-closing setpoint
(`finger_residual_scale = 0.5`), so the policy shapes — rather than replaces — an
open-loop grip schedule. The palm (6 DOF) is scripted. Policies are trained at 50 Hz on a
500 Hz sim (`decimation = 10`) across thousands of parallel environments.

#det([PPO objective and the actor–critic split], kind: "method")[
  PPO maximizes the clipped surrogate
  $ cal(L)^"CLIP"(phi) = EE_t [min(rho_t(phi) hat(A)_t,\
    "clip"(rho_t(phi), 1 - epsilon, 1 + epsilon) hat(A)_t)]\, quad
    rho_t(phi) = (pi_phi(a_t | s_t)) / (pi_(phi_"old")(a_t | s_t))\, $
  with Â\_t a GAE advantage from a learned value function (the *critic*) V\_ψ(s).
  The actor and critic are separate MLPs reading the same observation. *Two facts the
  project paid for in wasted runs:* (1) when fine-tuning a converged policy, the critic
  must be warm-started too — loading only the actor and re-initializing the critic knocks
  the actor off its optimum (`--warmstart-critic`, default on); (2) the critic's value
  V(s) is only meaningful on the state distribution it was trained on — a fact that
  returns to bite us in branch D below.
]

== The observation space (why it is 66-dimensional)

The policy observes a 66-d vector (65 without the reorient task). It is *not* large by
dexterous-RL standards; what inflates it past a naive "joints + object" count is that this
is a *trajectory-tracking* policy — it sees both the *actual* state and a *reference*
trajectory to track — plus the full state of the scripted palm.

#table(
  columns: (auto, auto, 1fr),
  table.header([*Term*], [*Dim*], [*Meaning*]),
  [`joint_pos`], [15], [all hand DOF (9 fingers + 6 palm), range-normalized],
  [`joint_vel`], [15], [same 15 DOF, velocities],
  [`object_pos`], [3], [object position relative to the palm site],
  [`object_pose_actual`], [7], [*actual* object pose in palm frame: pos (3) + quat (4)],
  [`ref_finger_qpos`], [9], [*reference* finger-config lookahead (the CEM plan)],
  [`ref_object_pose`], [7], [*reference* object-pose lookahead: pos (3) + quat (4)],
  [`actions`], [9], [previous action (for smoothness)],
  [`target_axis_misalign`], [1], [scalar "how far from vertical" (reorient only)],
  [*Total*], [*66*], [drop the last scalar → *65*, the hold-only variant],
)

$ "obs" = [ underbrace(q\, dot(q), "state, 30")\,
            underbrace(p\, p_"palm"\, g, "actual obj, 10")\,
            underbrace(hat(q)\, hat(p)\, hat(g), "reference, 16")\,
            underbrace(a, "9")\, underbrace(m, "1") ] in RR^66 . $

The 16 reference dimensions are load-bearing: they are how Policy B knows the *target*
trajectory without a goal image. Strip the references and the scripted palm and the
genuinely-novel state is only ∼35-d.

== Reward design

The reward is *contact-shaped* rather than palm-based — for this morphology the palm is
static and the *fingers* do the gripping, so a palm-distance reward is the wrong signal.
It composes three groups: *tracking* (follow the CEM reference), *contact + lift*
(establish and keep force closure), and *reorient* (Policy B only).

#det([The three reward groups, with the gotchas baked in], kind: "derivation")[
  *Tracking* — exponential potentials on finger config, object position, object
  orientation, e.g. r\_track = exp(-α norm(p - p̂)^2).

  *Contact + lift* — fingertip-contact mean/min, a lift-height term, a drop penalty. A
  subtlety paid for: with `lift_height` weighted 80, PPO would *cage* the cylinder
  without fingertip contact, so the contact terms were boosted to be a primary signal.

  *Reorient* (B only) — with n̂ the object long axis in world and n^∗ vertical:
  $ r_"align" = exp(-alpha (1 - hat(n) dot.c n^*))\, wide
    r_"prog" = max(0, Delta(hat(n) dot.c n^*)) . $

  *Lift-task terminations are hostile to rotation* — finger-slip, orientation-slip, and
  xy-slip must all be disabled for the reorient task; only a 10-step-grace `tip_lost`
  remains. *The reorient reward gate must fire before mean episode death* (an early
  version gated rotation at step 50 while episodes died at step 40 — zero reward for 2000
  iterations). *RL exploits floor contact* whenever it is geometrically available, so true
  in-hand behavior needs both a higher lift target and a floor-proximity termination.
]

#callout(tag: "Hard-won", kind: "warn")[
  *Reward-based jerk penalties backfire here.* The corrective high-frequency finger
  motion *is* the stabilization mechanism for an object held by friction; penalizing it in
  the reward makes the hold slip. Smoothness must come from a non-reward lever
  (deploy-time action low-pass, or motor-delay / observation-noise DR), not a reward term.
  Judge policies on *deterministic* held-cosine, never on training reward sums.
]

== Why two policies

A single policy trained for "pick up *and* reorient" suffers tangled credit assignment:
the lift and rotation rewards fight, and a warm-start from a "hold still after lift" prior
bakes in a bias PPO cannot easily override. The fix that worked is an explicit split —
*Policy A (lift)*: flat-on-floor → lifted and held horizontal; *Policy B (reorient)*:
from a pre-lifted held state, rotate to vertical without dropping or bracing on the floor.
Each then sees one objective and a clean reward landscape. The cost is a new problem — the
*handoff* — treated in its own section.

#media("assets/policyA_lift.mp4", label: [Policy A.],
  caption: [The lift half: flat-laying cylinder lifted and held horizontal.])
#media("assets/policyB_final.mp4", label: [Policy B (B1).],
  caption: [True in-hand reorientation from a pre-lifted spawn. Visibly jittery (a sim-only
    exploit) but the cylinder genuinely rotates without floor contact.])

== The B-registry (canonical naming)

Policy B has many variants; historically they accreted three colliding numbering schemes.
This is the single canonical registry — *roles* are A (lift) / B (reorient); every B
policy gets one chronological ID. Methods/levers are named in words or by branch letter
(below), never reusing a policy number. Run directories are unchanged; this table maps to
them.

#table(
  columns: (auto, 1fr, auto),
  table.header([*ID*], [*What it is*], [*Status / headline*]),
  [B1], [first working reorienter (was "Policy B v1")], [unlocks in-hand rotation],
  [B2], [smoothness finetune (`v2_smooth10x_quick`)], [smoothest of the v2 sweep],
  [B3], [`signed+critic` baseline (`abl_signed`, model 405)], [warm-start base for B4–B6; held-cos 0.979],
  [B4], [*lateral-only* (was P2 / `p2_lateral`)], [*best standalone reorienter, held-cos 0.988*],
  [B5], [handoff-DR alone (was P1)], [worse — destabilizes the grip; discard],
  [B6], [handoff state-bank (was P3)], [best de-centerer (3.0 cm), weakest reorient (0.933)],
  [B7], [first normal-lift finetune (`v2_fromP2`)], [collapsed (held-cos 0.029, 100% drop)],
  [B8], [normal-lift *grace window* (`v3_grace`)], [stable but NaN-crashed at iter 60/750],
  [B9], [grace window to completion (`v3b` repro/soft)], [collapsed to a degenerate plateau],
)

== The reorientation journey

Getting B to rotate at all took several dead ends, each a reusable lesson.

#det([The journey in full: v2 → v3 → v4 → v5 → B1], kind: "extended results")[
  #table(
    columns: (auto, 1fr),
    table.header([*Attempt*], [*Outcome*]),
    [v2], [`orientation_drift = -20` penalized the very rotation we wanted -> no motion.],
    [v3], [reorient reward gated at step 50, episodes *died* at step 40 -> reward never fired for 2034 iters.],
    [v4], [*floor-bracing emerges*: RL pivots the cylinder against the ground. Creative, not in-hand.],
    [v5], [floor contact forbidden -> collapses to "hold the lift, don't rotate".],
    [B1], [the two-policy split unlocks it: genuine in-hand rotation, 3.2 × v4 alignment, no floor contact.],
  )
  The throughline: lift-task terminations are hostile to rotation; the reorient gate must
  fire before mean episode death; and RL exploits floor contact whenever available.
]

#media("assets/v4_peak_floorbracing.mp4", label: [v4: floor-bracing.],
  caption: [The cylinder is rolled while its distal end braces against the floor — RL found
    the ground reaction helps when fingers alone cannot. Creative, but not in-hand: this is
    why the lift target was raised and a floor-proximity termination added.])
#media("assets/reorient_comparison_grid.mp4", label: [2×2 overview.],
  caption: [Top-left v3 (reward never fires), top-right v4 floor-bracing, bottom-left v5
    (no floor = no rotation), bottom-right the A→B two-policy solution.])

The current best *standalone* reorienter is *B4* (`lateral-only`): held-vertical cosine
*0.988*, object-jerk *halved* vs B3. Surprisingly the lateral-drift penalty that produced
it acts as a *smoothing regularizer*, not a de-centerer — a *position* penalty smooths
where velocity/acceleration penalties (which destabilize the hold) failed.

= The policy-switching problem

This is the crux, and it generalizes far beyond this hand. When Policy B takes over from
Policy A in a continuous, no-reset rollout, the object *drops within 3–5 steps* —
identically across every B variant. It is not a grip-strength problem.

== Diagnosis: an observation-discontinuity, out-of-distribution shock

Instrumenting object height every step shows the drop is *instantaneous at the seam*:

```
step 40  z=0.111 (A holding)  ->  45  z=0.094 (handoff)
     46  z=0.073  ->  48  z=0.022  ->  50  z=0.010 (floor)
```

In skill-chaining terms, A's *terminal-state distribution* β\_A (where its lift
rollout ends) does not lie inside B's *initiation set* ℐ\_B. The instant B takes
control it is *out of distribution* and emits large corrective actions — exactly the "huge
noise and action magnitudes" failure mode. Every skip-lift-trained B had never seen A's
true normal-lift delivery state.

== Why a hand is worse than a whole-body robot

A legged or whole-body robot at a policy switch has a *static stability margin* — a
support polygon, ground reaction, momentum to bleed off — and can take several
out-of-distribution steps and recover. A hand holding an object by *friction alone* has
essentially *zero margin*: the object is in unstable equilibrium maintained by an active
force-closure the fingers continuously regulate. One wrong action breaks contact and
gravity wins in 3–5 steps. Worse, the relevant state includes *contact forces*, which are
not in the 66-d observation and are themselves discontinuous at the seam. The tolerable
out-of-distribution budget at the seam is therefore ∼0 steps for a hand versus tens for
locomotion. The fix must make the seam in-distribution *by construction*.

== How the literature frames it — and where we sit

The seam is a textbook *initiation-set / terminal-set mismatch*, with a family of fixes.
We had been pushing only one.

#table(
  columns: (auto, 1fr, auto),
  table.header([*Branch*], [*Idea*], [*Us*]),
  [A. Adapt B→A], [train B from A's real terminal states (forward init), or on a broad grasp pool, so ℐ\_B ⊇ β\_A], [the *only* branch pushed (B7–B9); grace window lives here],
  [B. Adapt A→B], [pull A's terminal state *into* B's initiation set: a terminal-state-regularization discriminator, or fine-tune A with B's value as reward], [*never tried — A is frozen. Biggest unexplored lever.*],
  [C. Transition policy], [a third short policy bridging β\_A → ℐ\_B], [adds two seams; lower priority],
  [D. Switch on a signal], [switch when B's critic / feasibility crosses threshold, not at a fixed step], [*tried — see below*],
  [E. Action continuity], [residual on A's last action; blend window; CAPS low-pass], [partial; reward jerk penalties proven harmful],
  [F. Don't switch], [one goal/phase-conditioned policy], [parked — the split exists for credit-assignment reasons],
)

#det([Terminal-state regularization (the untried branch B)], kind: "derivation")[
  Train an initiation-set discriminator D\_ω^B to tell B's good start states from A's
  terminal states, then add to *A's* reward a term that pays A for finishing where B
  succeeds:
  $ R_"TSR"(s; omega) = bb(1)_(s in beta_A) dot.c D_omega^B (s)\, wide
    cal(L)(omega) = EE_(s ~ cal(I)_B)[D_omega^B (s) - 1]^2
                  + EE_(s_T ~ beta_A)[D_omega^B (s_T)]^2 . $
  Equivalently (Sequential Dexterity), fine-tune A with B's *transition-feasibility*
  value F\_B as an auxiliary reward, R'\_A = λ\_1 R\_A + λ\_2 F\_B.
]

== What branch A actually showed (B7–B9)

Branch A — make B robust to A's delivery by training it in the normal-lift env — was
pushed hard and *did not* close the seam. A naive normal-lift finetune (*B7*) *collapses*
(held-cos 0.029, 100% drop): at takeover, B's residual, the lift/floor terminations, and
the full reorient reward all fire at once, the OOD warm-start fumbles, and terminations
kill episodes faster than B can learn. A *grace window* (B8 / B9 — take over early but
only *hold* until the reorient pressure engages at a later step) stops the *training*
collapse, but at 40M steps converges to a *degenerate plateau*: reward sits flat (∼9)
while deterministic held-cosine goes *negative* and the object drops 100% of the time. B
learns to hold during grace and drop the instant the reorient phase engages — it never
crosses from "hold" into a working post-seam reorient from the OOD skip-lift prior.

== Branch D, tried: a critic-gated switch — and why it exposed the real problem

Rather than a human picking the switch step, branch D lets *B's value function choose the
moment to take over*: roll A while evaluating V\_B(obs) each step, and switch when it
peaks (B most confident it can succeed from here). Implemented in
`rl_demo_handoff_continuous.py --switch-on-critic`, paired with the best reorienter *B4*.

The result is a clean negative finding: *B4's critic is itself out-of-distribution on the
normal-lift delivery.* Its value reads +4.75 at step 0 (before A even lifts), then
collapses monotonically to ≈ -5000. The "peak" is trivially step 0; the gate fires
at step 29; the object still drops (min-z 0.006).

#media("assets/handoff_branchD_p2.mp4", label: [Branch D (critic-gated, B4).],
  caption: [The critic gate switches at step 29 — but B4's value landscape on the
    normal-lift delivery is degenerate (peaks at step 0, then collapses), so there is no
    good moment to hand B4 the object. It drops.])

#callout(tag: "The real lesson", kind: "note")[
  *Branch D is sound, but it cannot rescue an out-of-distribution policy* — a critic is
  only meaningful on the distribution it was trained on, and a skip-lift B's critic is
  garbage on a normal-lift delivery. So the two genuinely high-value moves both point the
  same way: get B (and its critic) *in*-distribution on A's delivery, *and* un-freeze
  Policy A so it delivers into B's basin (branch B). Widening B's catch alone (branch A)
  has now hit its ceiling; meeting in the middle is what a ∼0-margin hand needs.
]

== Next experiment

Stay in branch A but fix the *initialization*: warm-start from the *hold-only control* —
which already proved it survives A's delivery (`tip_lost` 44 → 1–4) — instead of the
OOD skip-lift reorienter, so the grace→reorient transition begins *in*-distribution.
Then open branch B by un-freezing Policy A. The best *standalone* reorienter is unchanged:
*B4*, held-cosine 0.988.

#refbox[
  *Sources.* Lee et al., _Adversarial Skill Chaining via Terminal State Regularization_,
  CoRL 2021 (arXiv:2111.07999). Chen et al., _Sequential Dexterity_, CoRL 2023
  (arXiv:2309.00987). Röstel et al., _Composing Dexterous Grasping and In-hand
  Manipulation via Scoring with an RL Critic_, 2025 (arXiv:2505.13253). Mysore et al.,
  _CAPS_, ICRA 2021. Full analysis: `docs/rl/policy_switching_literature.md`; research
  log: `RESEARCH_STATE.md`, `docs/rl/reorientation.md`.
]
