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
  [B10], [*hold-only warm-start*, hard onset], [reorients (held-cos 0.977) but violent (jerk 108); drops in continuous handoff],
  [B11], [hold-only warm-start, soft (residual 0.4, 150-it α)], [holds but never tilts (held-cos −0.137; too dilute)],
  [B12], [smoothness finetune *of B10*], [smoothing destroyed the tilt (held-cos 0.086, jerk 550)],
  [B13], [soft-but-committing onset (residual 0.5, 40-it α)], [smooth seam (jerk 21) but half-tilt (held-cos 0.462)],
  [B14], [adapt-B skip-lift bank (`adaptToA`)], [trains clean, still drops (min-z 0.0028)],
  [B15], [`Badapt` — static onset-inject], [new best at the time (min-z 0.0081), still drops],
  [B19/B22], [co-adaptation pairings (B→Atol delivery)], [B22 = best relative min-z 0.0114],
  [*B24*], [*live-A reset, warm-start B10*], [*🎉 first to HOLD the continuous seam (post-handoff min-z 0.110) + reorient (cos 0.751)*],
  [B25], [live-A canonical 0.5 retrain (deploy parity)], [lineage-comparable deployable version],
  [B26], [live-A, warm-start B4], [collapses — the B4 catch-22 (reorients from step 0, drops first)],
  [B27], [live-A B24 continuation, +40M], [reorient PLATEAUED (cos 0.742) — the B10 warm-start is the ceiling],
)

The lift/deliver side has its own registry: *A1* is the frozen deliverer
(`...stable_v1/model_500`); *A2–A9* are the un-freeze-A grip-migration runs (the `unfreezeA`
and `branchB` series, of which *A7* = `Atol20` is the canonical migrated A). Run directories
now carry their ID as a prefix (e.g. `b04_…p2_lateral_only`); intermediate Policy-B explorations
that were never canonized are prefixed `bx_`. The full directory↔ID map is the single source of
truth in `scripts/rename_results_bids.sh`.

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

== The hold-only warm-start — the seam, broken (B10)

The fix that branch A had been missing: warm-start B not from the OOD skip-lift reorienter
but from the *hold-only control* — which already proved it survives A's delivery
(`tip_lost` 44 → 1–4) — so the grace→reorient transition begins *in-distribution*. The
hold-only policy is 65-d (no reorient obs); the run is 66-d, so it is a partial warm-start
that zero-inits the new column for *both* actor and critic.

*It worked.* `B10` (hold-only warm-start, hard reorient onset) is the *first policy to both
survive the A→B handoff and reorient* — standalone held-cos *0.977*. Two tells confirmed the
in-distribution init landed in a better basin: reward started at ∼12 and climbed (vs the v3b
plateau's flat ∼9), and B10's *critic is no longer OOD* on the delivery.

But B10 reorients *violently* — `obj_jerk` *108*, four times the skip-lift reference B4's 27.
And `B11` (the soft variant: residual 0.4, basin curriculum α 0.5→4 over 150 iters) holds the
object but *never tilts* (held-cos −0.137): the 150-iter curriculum was so dilute the policy
settled into a hold-only optimum before the reorient signal sharpened.

#table(
  columns: (auto, auto, auto, 1fr),
  table.header([*ID*], [*held-cos*], [*obj-jerk*], [*Handoff behavior*]),
  [B4 (skip-lift ref)], [0.990], [27], [great standalone, but drops at the seam],
  [*B10* hold-only, hard], [*0.977*], [*108*], [*survives seam + reorients — but violent*],
  [B11 hold-only, soft], [−0.137], [10.8], [survives, holds, never tilts (too dilute)],
)

#callout(tag: "Where this leaves us", kind: "note")[
  The hold-only warm-start *closed the seam* — survival + reorientation through a continuous
  handoff, for the first time. The remaining problem narrowed from "does it work at all" to
  "make the transition *smooth*" — find the point between B10 (commits, but violent) and B11
  (smooth, but never commits). The standalone skip-lift metric env is OOD for these
  normal-lift Bs (its `drop=1.0` is an artifact); survival is judged on the *continuous-handoff*
  `min-z` and the rendered video, quality on held-cos, violence on `obj_jerk`.
]

== Iteration 2: smooth the seam, keep the tilt (B12 / B13)

#det([The two candidates], kind: "method")[
  - *B12 — smoothness finetune of B10.* Warm-start B10 (which already reorients), then ramp the
    action-rate and object-angular-acceleration penalties in *late* (curriculum starts iter 40)
    so it learns to tilt first, then smooths the seam — the "learn it, then make it smooth"
    recipe that worked for the v2 smoothness finetunes (B2). Targets the `obj_jerk` 108 directly;
    risk (the sim-only-jitter gotcha) is that smoothing re-breaks the tilt, so the ramp is gentle.
  - *B13 — soft-but-committing onset.* Same hold-only warm-start, but residual 0.5 (full
    authority, unlike B11's 0.4) and a basin curriculum α 0.5→4 over just *40* iters (commits
    fast, unlike B11's 150). Eases the first few iterations of the reorient onset without
    diluting it into a hold-only optimum.
]

Both are normal-lift grace-window runs warm-started in-distribution, 40M / 3072, NaN-resilient.
Target: continuous-handoff `min-z > 0.05` (survives) *and* held-cos near B4's 0.988 (reaches
vertical) *and* `obj_jerk` well below 108 (smooth seam).

== Iteration 2 outcome: smoothing broke the tilt, and the seam is still open

Neither candidate met the bar, and the honest deterministic numbers also *retire the earlier
optimism about B10*. Two findings:

*Smoothing B10 was catastrophic (B12).* Ramping the action-rate / angular-acceleration penalties
into B10 did not gently smooth a working reorient — it *destroyed* it: held-cos collapsed to
*0.086* (no tilt) while `obj_jerk` *tripled to 550* (worse than B10's own 108). This is the
sim-only-jitter gotcha at full strength — the corrective finger jerk *is* the stabilization, so
penalizing it makes the policy thrash. "Learn it, then smooth it" worked for the standalone
reorienter (B2) but not across the seam.

*Soft-commit is smooth but timid (B13).* The residual-0.5 / fast-α-curriculum onset gave a
genuinely *smooth* seam — `obj_jerk` *21*, below even B4's 27 — but it under-commits: held-cos only
*0.462*, a half-tilt that never reaches vertical. The soft onset trades the tilt away.

#table(
  columns: (auto, auto, auto, auto, 1fr),
  table.header([*ID*], [*held-cos*], [*obj-jerk*], [*cont. min-z*], [*Verdict*]),
  [B4 (skip-lift ref)], [0.988], [27.7], [0.117\*], [great standalone; drops at the seam],
  [B10 hold-only, hard], [0.975], [117], [0.0029], [reorients, but violent — and drops in the continuous handoff],
  [B12 smooth-of-B10], [0.086], [550], [0.0026], [smoothing destroyed the tilt *and* the jerk],
  [B13 soft-commit], [0.462], [21.3], [−0.0007], [smooth seam, but only half-tilts],
)

#det([Why the continuous `min-z` is the decisive number], kind: "note")[
  The held-cos / obj-jerk columns come from the standalone metric env, which resets to a clean
  held state — so they measure *reorient quality from a good start*. The `cont. min-z` column is
  the full A-lift → B-takeover rollout with no reset (handoff at step 40): the lowest object-center
  height over 240 steps. `min-z > 0.05` means B *kept the object off the floor* through the seam.
  *Every normal-lift B fails it* — B10 0.0029, B12 0.0026, B13 −0.0007 — i.e. the object reaches the
  floor in the continuous handoff even when standalone held-cos looks good. (\*B4's 0.117 is in its
  *own* skip-lift env; in the continuous normal-lift handoff B4 also drops.) The earlier "B10 survives
  the seam" read was from looser evidence; under this strict continuous bar, *no variant clears it.*
]

*Deploy-time levers (branch E) are exhausted.* With no retraining we tried, on B10, a deploy-time
action-blend window (ramp B's action in over 8–12 steps) and a critic-gated switch (let B10's
now-in-distribution critic pick the switch step). Neither rescued survival: blend-12 `min-z`
*0.0063*, critic-gate (fired at step 29) *0.0044*, B12/B13 blend-8 *0.0026 / −0.0007*. Softening
*when* or *how fast* B takes over does not fix *what* B does once it has the object.

#media("assets/handoff_B12_smooth.mp4",
  label: [B12 — smoothing finetune of B10.],
  caption: [The smoothness penalties destroyed the reorient: the object barely tilts and the grip
    thrashes (held-cos 0.086, jerk 550). The object reaches the floor.])

#media("assets/handoff_B13_softcommit.mp4",
  label: [B13 — soft-but-committing onset.],
  caption: [A smooth seam (jerk 21) but a timid half-tilt (held-cos 0.462); the object still settles
    to the floor in the continuous handoff (min-z −0.0007).])

#callout(tag: "Verdict", kind: "warn")[
  Iteration 2 did not close the seam, and it sharpened the diagnosis. *Moving B alone is not
  enough.* B10 (commits hard), B13 (commits soft), and B12 (over-smoothed) span the entire
  commit/smoothness axis, and *all three drop the object in the continuous handoff* — as do the
  deploy-time blend and critic-gate levers. The failure is not B's onset schedule or its
  smoothness; it is that B, handed A's *actual* delivered state, cannot maintain the grip while it
  reorients. The remaining untried lever is the one that moves *A*.
]

== The single best next experiment — branch B: co-adapt Policy A

Un-freeze Policy A and fine-tune *its* delivery toward a state B can reorient from, rather than
forcing B to absorb a fixed delivery. Concretely: keep B10 frozen as the reorienter and fine-tune
A in the continuous env with a *terminal-state-regularization* term (Lee et al. 2021) — penalize A
for ending its lift in states outside B10's initiation set, using B10's critic value at the seam as
the reward signal (Röstel et al. 2025). This directly attacks the one thing every B-side experiment
left untouched: the seam state itself. It is preferred over more B-side tuning because the data now
shows the commit/smoothness axis is fully explored and saturated on the wrong side of `min-z`, and
over branch E because the deploy-time blend/gate levers already failed to rescue survival. A cheaper
fallback worth pairing in: record A's *real* delivered seam states with `rl_record_handoff_states.py`
and reset B's normal-lift training from that bank (`--handoff-state-bank`), closing the residual gap
between B's training reset and A's actual hand-off.

== What the seam sweep showed: both sides saturate alone, co-adaptation is the lever

That program was carried out. Both one-sided moves were pushed to their limit, and the result is a
clean structural finding.

*Adapt B → A (branch A) saturates.* Training B on A's delivery via a state bank (`--handoff-state-bank`)
trains cleanly but still drops (min-z 0.0028). The diagnosis sharpened: the bank fires only in the
*skip-lift* env, so B still trains under the skip-lift *observation schedule* — different from the
normal-lift deploy even when the physical state matches. The binding constraint is the *obs schedule*,
not the state. Closing that — train B in the normal-lift env *and* inject A's real delivered state at
the seam onset (`inject_handoff_bank_at_onset`) — reached a new best of *0.0081*, but still a drop.

*Making the teleport Markov-complete does not help.* The residual gap looked like the *teleport*:
the injected snapshot was static (a recorder bug had silently zeroed the bank velocities). Following
an external analysis we made the injection physically complete — A's real object + finger velocities,
and an override of the one history-dependent observation (`last_action`, which A delivers at a
substantial 0.23 rad while B had always trained expecting ≈ 0). Since the observation space has no
differenced or stacked-history terms and the position actuators carry no activation state, this makes
the injected seam *Markov-indistinguishable* from organic arrival. *It did not close the seam* —
min-z *0.0027*, no better than the static inject. The conclusion is load-bearing: the seam is *not a
missing-state-information problem*, and the entire *inject-A's-state-into-B* family is saturated
(0.0028 / 0.0081 / 0.0027, all far below the 0.05 bar).

#table(
  columns: (auto, auto, 1fr),
  table.header([*Pairing (handoff\@40)*], [*min-z*], [*Reading*]),
  [baseline frozen-A × B10], [\~0.0029], [either piece frozen — drops],
  [A-migrated × B10 (A alone)], [−0.0001], [moving A alone, B frozen — no better],
  [frozen-A × B-adapted (B alone)], [0.0075], [moving B alone — modest],
  [*A-migrated × B-adapted (BOTH)*], [*0.0114*], [*co-adaptation — new best*],
)

*Co-adaptation is the lever.* Pairing the independently-migrated A (fine-tuned toward B's grip) with
the independently-adapted B (trained on A's delivery) — neither aware of the other's move — gives the
best min-z yet, *0.0114*, beating either side alone. Moving *both* terminal and initiation
distributions toward each other is exactly the adversarial-skill-chaining prescription (Lee 2021 /
Röstel 2025), and here it is the first thing to clear the prior best by a clear margin.

#callout(tag: "Honest status", kind: "warn")[
  *The seam is still open.* 0.0114 is the best *relative* min-z, but it is still a drop, not a hold
  (the bar is 0.05). And the adapted B has no stable post-seam holding grip — it drops by \~step 48 —
  so the weak link is *B catching*, not A delivering. Co-adaptation is a direction, not a solution.
]

#callout(tag: "The untried mechanism", kind: "note")[
  Every adaptation so far trains B on a *teleport* into the seam (a bank or an injection); the deploy
  seam is *organic* (A runs live). Even the Markov-complete injection did not close it — the remaining
  suspect is the contact-solver warm-start / one-step contact-force ramp that no instantaneous
  teleport reproduces. The next mechanism removes the teleport entirely: run *frozen Policy A live*
  for the first \~40 steps of every B training episode (real physics, real contacts, real
  `last_action`), then begin B's PPO rollout at the seam. It is a training-loop change, not a reward
  knob — A drives the pre-onset steps and those steps are masked from the PPO update — which is why it
  is the deliberate next build rather than another swept run.
]

== The live-A reset closes the seam

The build worked. The live-A reset is the first mechanism to make a single policy *both
survive the continuous A→B handoff and reorient* — the seam is solved in principle. Frozen
Policy A drives B's training env live for steps 0..40 of every episode (real physics, real
contacts, real `last_action`); B's PPO rollout then begins at the *organic* seam, and the
A-driven pre-onset steps are masked from the PPO update (advantages zeroed and renormalized,
returns kept — masking *advantages*, not log-probs, avoids the log-prob trap).

Warm-started from B10, 20M / 3072 (`policyB_liveAreset_fromB10`, model 270): the continuous
handoff now holds at *post-handoff min-z 0.110 m* (`≫` the 0.05 bar) at held-cos *0.751* —
where every prior teleport approach dropped within 3–5 steps (min-z 0.003–0.011). The training
signature confirmed it: masked-frac fell 0.95→0.20 (episodes lengthened `~5×`), alignment
climbed 0.45→58.9, `tip_lost` fell 51→8, and episodes ran to time-out.

#media("assets/handoff_liveAreset_scale02.mp4", label: [The live-A reset — seam held.],
  caption: [Policy A delivers live, B takes over at the organic seam with no teleport and holds
    the object at full height through the whole post-seam rollout while reorienting. First policy
    to both survive the continuous handoff and reorient.])

#callout(tag: "Config-parity gotcha", kind: "warn")[
  That run trained at `finger_residual_scale = 0.2` (the trainer default) while B10 and the deploy
  demo use *0.5*. Evaluating it at 0.5 applies its residuals 2.5× too large and the seam collapses
  instantly — an *artifact, not a failure*. The training env must match the deploy env on
  scale / easing / contact-gate. And the decisive number is *post-handoff* min-z: whole-rollout
  min-z is dominated by the pre-lift floor phase (`z ~ 0.012`), so the 0.05 bar is unreachable by it.
]

== What is still open: reorientation *quality*

The hold is solved; the *quality* is not. Asked to judge the result, the verdict was "still
jittery, and it doesn't reorient well" — and both flaws trace to the same root, the *B10
warm-start* (the "violent survivor": held-cos 0.977 but `obj_jerk` 108, four times B4's 27).
Three tests pinned it down:

#table(
  columns: (auto, auto, 1fr),
  table.header([*Attempt*], [*Result*], [*Takeaway*]),
  [+40M more training (from B10-live-A)], [held-cos 0.742, no climb],
    [the 0.74 reorient ceiling is the *warm-start*, not undertraining],
  [warm-start *B4* (smooth, 0.988) instead], [collapses at the seam (align 0)],
    [B4's reorient-from-step-0 actions shock A's grip → it drops before the reorient gate fires → zero gradient],
  [deploy action low-pass], [drops the object (min-z 0.0027)],
    [B10's corrective jerk *is* the stabilization — it cannot be filtered out],
)

Both complaints share one fix: get B4's smooth, full-vertical reorientation to *survive the
seam*. The blocker is the *B4 catch-22* — a policy must survive the seam before it can be
taught to reorient there, but B4 reorients from step 0 and drops first (it is exactly the
catch-22 that forced the hold-only B10 warm-start in the first place). The next experiment —
a build, not a knob — is a *training-time seam action-ramp-in*: in `live_a_runner.py`, ease B
into control over the first `~8–12` steps after onset (`α·B + (1−α)·A`, α ramping 0→1, those
blend steps masked from PPO), warm-starting B4. This is the training analog of the demo's
action-blend, applied where it can actually teach.

== Outcome: the B4 path is dead; the win is lifting B10's quality (B24→B29)

The ramp-in was built and tested. A layered diagnosis showed it is *necessary but not sufficient*:
the ramp keeps the object aloft *during* the blend, and delaying the terminations to B's takeover does
let trainable B-steps accumulate — but the instant B4 has authority it drops A's flat delivery to the
floor and never recovers, because *B4 has no hold prior* (precisely why B10/hold-only was the warm-start
that worked). All three B4-warm-start runs failed (floor-drop + transient NaN). *B4 cannot be made to
survive the seam.*

The productive move is the *inverse*: keep the already-surviving B10-live-A policy (B24) and lift its
*quality* with a *non-terminating commit + speed bonus* (the no-terminate success bonus the v2 sweep had
recommended, so no threshold-gaming) plus a sharper near-vertical basin re-anneal.

#table(
  columns: (auto, 1fr, auto, auto, auto),
  table.header([*ID*], [*recipe*], [*held-cos*], [*peak*], [*min-z*]),
  [B24], [B10-live-A baseline], [0.751], [0.816], [0.110],
  [B27], [+40M continuation], [0.742], [0.817], [0.105],
  [B28], [+ commit-bonus 30], [0.759], [*0.891*], [0.110],
  [*B29*], [+ commit 60 + basin re-anneal (α 1→6)], [*0.784*], [0.866], [0.108],
)

The commit bonus + sharper basin *moved the quality ceiling 0.75 → 0.78* (real but modest, `~5°` closer to
vertical; the hold stays solid `~0.11`) — but did *not* reach B4's standalone 0.988. *The B10-warm-start
basin is a stubborn quality ceiling that reward-shaping only nudges.* B29 is the new best handoff policy.
Breaking past `~0.8` likely needs a *new mechanism* — distilling B4's full reorientation onto the held
post-seam states B29 visits (behavior cloning / teacher-student) — not more reward tuning.

== The force chase, and the phantom it was chasing (B30→B34)

From B29 the work turned to the grip. B32 (`gripSmooth_w4`) became the *firmest and smoothest handoff
yet* — post-handoff min-z 0.1085, held-cos *0.891*, and 4–5× smoother than a rejected jittery candidate
that the new eval diagnostics had exposed (a near-vertical policy can still be *frantically shaking* in
place; cos and min-z are blind to it, so the eval now reports lin/ang *jerk*, wander, and contact force).
But B32 held with an `~11 N` fingertip clamp — believed to be the source of both the thumb penetration
and the residual jitter. So we added an over-grip *penalty* (`grip_force_excess`, a quadratic cost on
fingertip force above a threshold) and swept the threshold down (B33→B34) to find the floor.

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  table.header([*thresh*], [*force*], [*held-cos*], [*min-z*], [*ang-jerk*], [*palm force*]),
  [4.0 (B33)], [7.5 N], [0.845], [0.111], [49], [*0.0 N*],
  [2.5],       [6.7 N], [0.771], [0.112], [69], [*0.0 N*],
  [2.0 (t20)], [6.6 N], [0.782], [0.112], [76], [*0.0 N*],
)

Halving the penalty knee moved the force `< 1 N`: a fingertip hold of this rod floors at `~6.6 N`, palm
force is *0.0 N in every run* (the object never seats into the palm), and held-cos stays flat. It looked
like gentleness was a *seating* problem — until we verified the premise the whole arc rested on.

#callout(tag: "The premise was a phantom", kind: "warn")[
  "B3 is gentle because it holds a `~3 N` seated grip" is *false*. Direct measurement
  (`probe_grip_force.py`, each policy's own standalone rollout) gives *B3 = 7.04 N*, *B4 = 8.77 N* —
  as hard or *harder* than our handoff (6.6 N). Nobody seats (palm force 0.00 N everywhere). Force does
  *not* cause jitter — B4 is the *smoothest* policy at the *highest* force. Penetration is universal (the
  same soft solver). The "`~3 N`" figure was a misread of `grip_force_max = 3.0`, the grip-force
  *reward's saturation cap* — not B3's real force. *B30→B34 optimised a non-problem.* The sweep was still
  informative: it proved force is *decoupled* from both the hold and the smoothness — the real remaining
  gaps are *verticality* and *smoothness* from the seam, ceilinged by the B10 warm-start basin and the
  marginal fingertip grip, not the grip force.
]

A 3-stage composition (A → B32 → B4) was also ruled out: B32 handed B4 a rock-stable cos-0.90 pose and
*B4 still dropped it* — the catch-22 is the *grip basin*, not the object pose, so naive action-distillation
would hit the same wall. B32 is therefore reframed as *already a good handoff reorienter* (holds + cos 0.895).

== Where it stands, and the 2026-06-22 pivot

The seam is solved (the policy holds the continuous handoff at min-z `~0.11`); B32 reorients to cos 0.895.
Reaching B4's 0.988 *and* its smoothness from a live-A handoff is blocked by the grip-basin catch-22 and
the B10 jitter ceiling, which reward-tuning only nudges. At this point the user reprioritised:

#callout(tag: "The user's call", kind: "note")[
  "I don't care about a close-to-brace pullup or a close-to-vertical reorientation — I just want a
  *smooth, low-force* grasp and reorient. Then we can think about morphology optimization."
]

The honest constraint: genuinely *low* force (below the `~6.6 N` fingertip floor) needs the object to
*seat into the palm* so the palm bears the load and the fingers relax — which *this morphology cannot do*
(the object sits 7–8 cm below the palm). That is the *morphology* step. But one lever is untried *within*
the current morphology: every run so far *maximised verticality* (alignment +100), which is exactly what
forces the tense corrective clamp and the jitter. A *gentle partial reorient* may relax the grip and
smooth out for free. Two runs test this: (1) a *gentle low-force reorienter* (warm-start the gentlest
survivor B34-t20, relax verticality, lower the grip reward, raise the lateral smoother), and (2) a
*re-opened Policy A* fine-tuned with an over-grip penalty to deliver with less force. Morphology
optimization — the real low-force lever — follows, and deliberately breaks the A/B lineage, so it comes last.

#callout(tag: "Known artifact — thumb penetration", kind: "note")[
  In the live-A rollouts and many prior runs the *thumb visibly phases into the screwdriver*
  during the grip. This is a consequence of the deliberately soft contact solver this task uses
  (`impratio = 10`, elliptic friction cone, and a soft `solref = 0.006` / `solimp = 0.97 0.995`
  geom default), which trades some interpenetration for a stable, non-explosive grip. It is
  cosmetically and physically imperfect, but the contact parameters are *deliberately left
  unchanged* until reorientation quality is solved — retuning them would perturb every policy's
  grip force and invalidate the A/B lineage and the seam comparisons above. It is deferred to a
  sim-to-real hardening pass.
]

= Morphology co-design: can the hand *shape* help?

The RL side reached a *structural* ceiling: no reward lever removes the excess grip force, because
the cause is the hand's geometry, not the policy (the fingertip force floors at `~6.6 N`; the grip
is a lopsided pinch the policy cannot rebalance; genuinely low force needs the object to *seat into
the palm*, which this morphology cannot do). So the remaining lever is the *9-param finger
geometry* (per-finger base x, y, and length). The question this section answers: *does changing the
hand shape measurably improve the lift-and-reorient — and can we even tell?*

== The honest per-design pipeline

An earlier landscape sweep hinted that morphology matters (one design, `m05`, reoriented best), but
its scoring had two artifacts: it transferred the grasp keyframe in *joint* space (mis-placing
fingertips on repositioned fingers) and scored reorientation on a *teleport* grip with no real
Policy A. The honest evaluator fixes both and runs the *full* chain per design:

#det([The per-design A→B pipeline, step by step], kind: "method")[
  For a 9-param design vector: (1) `generate_morphology_xml` bakes it into fixed geometry; (2)
  `retarget_keyframe_ik` writes an `open_ik` keyframe by IK-ing the *world* fingertip positions of
  the known-good baseline onto the new fingers; (3) CEM grasp synthesis from `open_ik` (a
  graspability gate skips hopeless designs); (4) a *native Policy A* trained *from scratch* (a
  warmstarted A ejects the re-CEM'd object); (5) *Policy B* via the *live-A reset* (frozen A
  drives the real lift, B learns reorient from the organic seam) — and it *must* pass
  `--open-finger-from-keyframe`, or B resets to the wrong open pose and drops the object; (6) a
  continuous A→B handoff scored by the trajectory-health scorecard. Orchestrated, resumable, in
  `scripts/morph_pipeline_sweep.py`. This is the reusable unit every design is scored by.
]

On the co-designed `m05` hand this yields the first *health-gated* genuine pickup→reorient on a
co-designed morphology: an instant balanced 3-finger grasp, held aloft the whole rollout (min-z
`0.12`), reoriented to cos `~0.90`, smooth. Reference policies *a10* (its lift) → *b33* (its
reorient).

#media("assets/handoff_m05_FIXED.mp4", label: [a10 → b33 on the co-designed m05 hand.],
  caption: [The health-gated pickup→reorient. Instant 3-finger grasp, held aloft, reoriented toward
  vertical. This is the "blessed" reference policy the sweep explores around.])

== Trajectory-health monitoring — so degeneracy cannot hide

The deeper lesson: our headline metrics (reward, tip-lost, object-held, even held-cos) *masked*
real defects visible only on video — a 2-finger grasp with a *late* third finger, high-frequency
jitter, de-centering slides, idle-finger pinches. A single logged rollout now becomes an explicit
PASS/WARN/FAIL scorecard, baked into every eval by default.

#det([The six health checks (`src/morphohand/rl/trajectory_health.py`)], kind: "detail")[
  *late_finger* — spread in per-finger first-contact step; *idle_finger* — a finger touching too
  little or carrying `< 1 N` (degenerate pinch); *drop* — object-center below the floor threshold in
  the hold window; *jitter* — object angular jerk; *de_centering* — net lateral drift + slide ratio;
  *over_clamp* — mean fingertip force above `~5 N`. Verdict = worst check. It runs inside the handoff
  eval (writes a `.health.json`), standalone as `scripts/policy_healthcheck.py`, and as an
  acceptance gate at the end of A-training. It earns its keep below: it flagged a "cos 0.94" handoff
  we had called a win as degenerate (late finger + jitter 156 + 2-finger), and it correctly rejected
  a sweep design whose `2 N` "low force" was achieved by *idling a finger*.
]

#fig("assets/m05_health_characterization.png", label: [Health scorecard.],
  caption: [The monitor cleanly separates a degenerate lift (idle/late finger, jitter) from the
  fixed, balanced one — a separation the aggregate reward could not see.])

== The sweep — and three bugs it surfaced before any science

An initial 8-design sweep around `m05` looked like a disaster (6/8 aborted). Reading the *logs* (not
the reward) showed it was not eight bad designs — it was three bugs in the evaluator, each a
concrete lesson:

#det([The three pipeline bugs (all fixed)], kind: "analysis")[
  (1) *Policy B was missing `--open-finger-from-keyframe`* → it reset to the baseline flung-out-thumb
  open pose, closed from the wrong grip, and dropped the object (sank four designs; verified
  `open_finger_from_keyframe: false` in their configs vs `true` in `b33`). (2) *Checkpoint
  selection*: on a clean run, use the *final* A checkpoint — an early one lifts marginally higher
  but has an under-refined grip → idle finger; salvage an earlier checkpoint *only* when training
  actually collapsed. (3) *Warmstarting A ejects the object* (a grip-specific residual on a
  re-CEM'd grasp) — A must train from scratch. With the fixes, the `m05` anchor reproduced a clean,
  balanced, held reorientation (all three fingers loaded).
]

The corrected pipeline then ran a *16-design* local search around `m05`. Fourteen of sixteen held
the object and reoriented — no drops. But ranking them surfaced the real result.

#det([The large16 ranking (held-cos / force / jerk)], kind: "results")[
  Best cos `L01_06` 0.90 (but geometrically ≈ `m05`); anchor `m05` 0.78; `L01_13` (thumb moved 9 mm
  outward) 0.76 at *lower* force (7.4 vs 10.9 N) and *half* the jerk (6.0 vs 12.5) — the apparent
  "design lead"; `L01_02` posted the lowest force (2.0 N) but *by idling a finger* → the scorecard
  correctly FAILed it. Full table `MORPH_PIPELINE_large16_TABLE.md`; figure
  `morph_pipeline_large16_summary.png`.
]

== The result: seed variance swamps the design effect

The "design lead" `L01_13` was tested against `m05` with three fresh seeds each. It *did not
replicate*. Pooling every run of each exact design:

#callout(tag: "The finding", kind: "warn")[
  `m05`: held-cos *0.32 ± 0.38* (range −0.29 … 0.78, n = 5). `L01_13`: *0.38 ± 0.44* (range
  −0.36 … 0.76, n = 4). The between-design gap is *0.07 against a pooled seed noise of 0.41* —
  indistinguishable, on cos *and* force *and* jerk. The *same* hand reorients anywhere from
  *backwards* to *near-vertical* depending only on the training seed.
]

#fig("assets/morph_confirm_seedbands.png", label: [Multi-seed confirmation.],
  caption: [`L01_13` vs `m05`, several seeds each. The bands overlap almost completely — the design
  difference is far smaller than the seed-to-seed spread. No design was promoted.])

#det([Why so wide? It is *training-convergence* variance, not rollout noise], kind: "analysis")[
  The tell is *peak* cos, not just the held tail: across seeds the peak reorientation itself ranges
  0.02 → 0.81. Different seeds converge to *qualitatively different policies* — some discover the
  "roll the cylinder up to vertical" strategy, many get stuck at "hold, barely rotate," a few
  collapse in training. Reorientation is a *hard-exploration* target (the reward is only reachable
  once the policy stumbles into the rolling motion), and from-scratch PPO on a non-convex,
  contact-rich landscape finds it or not depending on early exploration. `m05`'s original 0.90 was
  a *seed that found it*; it is a lottery, not a property of the geometry.
]

== What it means for co-design

The 9-param designs here are *grasp-equivalent* (IK-retarget gives every one a persistent tripod),
and their reorientation quality is dominated by RL-training seed luck. So a single (even 3-seed) run
per design cannot resolve morphology differences. *To search designs meaningfully we must first
reduce the evaluator's variance*, by one of:

#det([Three ways to cut the evaluator variance (and what "shared warm-start" means)], kind: "method")[
  (a) *Many seeds averaged* — brute force; the `~0.4` cos spread needs `≥ 5–10` seeds to pin a `0.1`
  difference. (b) *A shared reorient warm-start* — warm-start *every design's Policy B from the one
  proven reorienter (`b33`)*, which already knows the rolling motion, instead of from each design's
  freshly-trained *holder* A. This removes the per-seed re-discovery lottery: every design starts
  knowing *how* to reorient and only adapts that behavior to its grip, so what varies is the design,
  not the seed's luck in re-inventing the skill. (Distinct from warm-starting *Policy A*, which
  ejects the object — the shared prior belongs on the *reorienter*, the skill that is hard to
  discover.) (c) *A cheaper, lower-variance proxy* than a full from-scratch A→B rollout. Until one of
  these is in place, `m05` (a10 → b33) remains the reference design, and the sweep's real deliverable
  is the *health-gated pipeline* + this *variance characterization*, not a new winning hand.
]

#refbox[
  *Sources.* Lee et al., _Adversarial Skill Chaining via Terminal State Regularization_,
  CoRL 2021 (arXiv:2111.07999). Chen et al., _Sequential Dexterity_, CoRL 2023
  (arXiv:2309.00987). Röstel et al., _Composing Dexterous Grasping and In-hand
  Manipulation via Scoring with an RL Critic_, 2025 (arXiv:2505.13253). Mysore et al.,
  _CAPS_, ICRA 2021. Full analysis: `docs/rl/policy_switching_literature.md`; research
  log: `RESEARCH_STATE.md`, `docs/rl/reorientation.md`.
]
