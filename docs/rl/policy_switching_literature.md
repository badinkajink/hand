# Policy switching / handoff: how the literature frames it (2026-06-04)

Stepping back from the furious iteration. We have been hammering on **one side** of a
problem the literature has a clean taxonomy for. This doc maps our A→B handoff onto that
taxonomy, names what we have *not* tried, and ranks concrete next moves. Sources at the end.

## 1. Our problem in the literature's vocabulary

What we call "the seam drop" is the canonical **skill-chaining mismatch**: the **terminal
state distribution** of policy A (`β_A`, where A's lift rollout ends) does not lie inside the
**initiation set** of policy B (`I_B`, the states from which B can succeed). The moment B
takes over, it is **out-of-distribution** and emits large corrective actions → grip fails.
This is exactly our diagnosis (instantaneous collapse within 3–5 steps, identical across
P2/P3/baseline because all were trained in the skip-lift env and none ever saw A's true
delivery state). The literature name for our failure is **initiation-set / terminal-set
mismatch**, and there is a whole family of fixes — we have been using only one of them.

## 2. Why a HAND is worse than a body (the user's intuition, made precise)

A legged/whole-body robot at a policy switch has a **static stability margin**: a support
polygon, ground reaction, momentum it can bleed off over several steps. It can take a few
OOD steps and recover. A hand holding an object by **friction alone has essentially zero
margin** — the object is in unstable equilibrium maintained by an active force-closure the
fingers are continuously regulating. An OOD observation → one wrong action → contact breaks →
gravity wins in 3–5 steps, irrecoverably. Consequences:

- **The tolerable OOD budget at the seam is ~0 steps for a hand, ~tens for a body.** So
  action-blending windows and "recover after a stumble" tricks that work for locomotion are
  far weaker here.
- **The relevant state includes contact forces**, which are (a) high-dimensional, (b) not in
  our 66-dim obs, and (c) discontinuous at the seam. The initiation-set mismatch is partly a
  *contact-configuration* mismatch the policy can't even observe.
- ⇒ For a hand the fix must make the seam **in-distribution by construction** (both sides) and
  preferably preserve **contact continuity**, not merely "survive a few bad steps."

## 3. The taxonomy of fixes — and where we sit

### A. Adapt B to A: widen / shift B's initiation set toward A's delivery  ← WHERE WE LIVE
Train B starting from A's actual terminal states so `I_B ⊇ β_A`.
- **Forward initialization** (Sequential Dexterity): after A trains, its successful terminal
  states become B's *initial-state distribution*. This is precisely what normal-lift B
  training approximates, and what our **v3b grace-window** run is doing (let B see the post-lift
  state, hold first, reorient later).
- **Train on a broad grasp distribution** (Röstel et al., "Composing… via Scoring"): they
  sidestep the seam by training the in-hand policy from a *wide pool of grasps* so whatever
  arrives is in-distribution — initiation-set widening by brute coverage. Our P1 handoff-DR
  and P3 statebank were weaker versions of this (but trained in the wrong env, so they missed).
- **Status for us:** this is the only branch we've pushed. v3b is the right move *within* it.

### B. Adapt A to B: pull A's terminal state INTO B's initiation set  ← WE HAVE NOT DONE THIS
Don't just move B toward A — move **A toward B**. Two concrete mechanisms:
- **Terminal State Regularization (Adversarial Skill Chaining, Lee et al. 2021).** Train an
  *initiation-set discriminator* `D_B(s)` (1 on B's good init states, 0 on A's terminal
  states), then add `R_TSR = 1[s∈β_A]·D_B(s)` to **A's** reward so A learns to *finish in a
  state B likes*. They weight it heavily (λ≈1e4) and fine-tune the chain jointly.
- **Backward fine-tuning with B's value as A's reward (Sequential Dexterity).** Train a
  **transition-feasibility function** `F_B(s_{t-10:t}) ≈ E_B[Σr]` (a small attention net over
  the last 10 obs predicting B's expected return from a handoff state), then fine-tune **A**
  with `R'_A = λ1·R_A + λ2·F_B(...)`. A is rewarded for delivering into B's high-value basin.
- **Why this matters for us:** a hand's seam tolerance is ~0, so meeting in the middle (A
  tightens its delivery *and* B widens its catch) is much stronger than B alone chasing a
  fixed, possibly-awkward `β_A`. **This is the biggest unexplored lever.** Policy A is frozen
  in our pipeline; the literature says it shouldn't be.

### C. Bridge with a dedicated transition policy
Learn a third short policy whose only job is to drive `β_A → I_B` (transition policies via
distribution matching, Lee et al. 2021b). Cleaner credit assignment, but it's a *third* thing
to train and adds another seam (A→T and T→B). Lower priority for us than B.

### D. Switch at the RIGHT TIME, not a fixed step  ← CHEAP WIN WE'RE MISSING
We hard-switch at **step 45**. But A's terminal state at *exactly* step 45 is not guaranteed
to be B-friendly — we're sampling `β_A` at an arbitrary clock tick. Both Sequential Dexterity
and the Composing-via-Scoring paper switch on a **learned signal**, not a clock:
- Sequential Dexterity: compute feasibility `c = F_B(s_{t-10:t})/h`; **switch when c > 1**
  (B predicts it can succeed from here). Enables skipping/recovery too.
- Composing-via-Scoring: use the **RL critic / value function** to *score states* and pick the
  handoff (they score grasps; same idea applies to handoff timing).
- **For us:** even without retraining, gating the switch on "B's critic value (or a held-cos /
  contact-stability proxy) exceeds threshold" instead of step==45 should cut seam variance.
  We already have B's critic (we now always warmstart it) — it can double as the switch oracle.

### E. Action-level continuity (attacks the "huge action magnitudes" directly)
The noise/magnitude spike at the seam is an *action-space* discontinuity (A's last action vs
B's first absolute action). Independent of state mismatch:
- **Residual formulation at the seam.** Have B output a *residual on A's last action* (or on a
  held setpoint) rather than a fresh absolute target, so the command can't jump. Residual
  policy learning is built for exactly "abrupt transition around control-authority switching →
  distributional discontinuity." Our finger-residual gating is in this spirit but B still
  produces absolute targets once active.
- **Blend/overlap window.** Linearly ramp authority `a = (1-α)a_A + α·a_B` over N steps. We
  have `--blend-steps` and it already extends the hold 6→18 steps — but a hand's ~0 margin
  means blending alone only delays the drop; pair it with B (state fix), don't rely on it.
- **Smoothness regularization (CAPS / Grad-CAPS).** Penalize ‖a_t − a_{t-1}‖ and local
  gradient to kill high-frequency components — proven for sim-to-real (≈80% power drop on a
  quadrotor). Matches gotcha #10 (our sim-only finger jitter). **Caveat we already learned:**
  reward-based jerk penalties *destabilize our hold* (the corrective jerk IS the
  stabilization). CAPS-style smoothing is best applied as a **deploy-time action low-pass or a
  DR term**, not a training reward — consistent with RESEARCH_STATE's "use a non-reward lever."

### F. Don't switch at all: one policy, phase/goal-conditioned
The cleanest way to have no seam is to have no switch. Several dexterous groups train a
**single goal-conditioned policy** for grasp→reorient, or a wide-coverage in-hand policy fed
hand-picked grasps (Composing-via-Scoring; unified/goal-conditioned manipulation lines). The
seam becomes an internal phase the network smooths itself. Cost: harder credit assignment (the
exact reason we split A/B in the first place — see cross-phase lesson #6). Worth a scoping
note, not an immediate pivot, given our split is finally working.

## 4. Recommendation — ranked

1. **Add the B-side switch oracle now (branch D), ~free.** Replace the step==45 hard switch in
   `rl_demo_handoff_continuous.py` with "switch when B's critic value (or a contact/held-cos
   proxy) crosses a threshold." Reuses the critic we already warmstart. Removes the
   arbitrary-clock contribution to seam variance without any retraining. Validate on the same
   min-z metric.
2. **Finish v3b (branch A) — already running, leave it.** It is the correct move within "adapt
   B to A." Let the eval trigger render the seamless video.
3. **THEN open branch B (biggest unexplored lever): stop freezing A.** Train B's
   transition-feasibility function `F_B` (or reuse B's critic), then **fine-tune A** with
   `R'_A = R_A + λ·F_B(handoff_state)` so A delivers into B's basin — the meet-in-the-middle the
   ~0-margin hand needs. Equivalent cheaper proxy: terminal-state regularization with a small
   `I_B` discriminator. Do this only after v3b converges so we fine-tune A toward a *real* B.
4. **Deploy-time action low-pass (branch E), if the seamless video still shows jitter/spike.**
   Non-reward smoothing only (we proved reward jerk penalties hurt the hold).
5. **Park branch F (single goal-conditioned policy)** as a research note; revisit only if the
   A↔B chain plateaus.

Bottom line: the seam is a textbook **initiation-set mismatch**, and a hand's near-zero
stability margin is *why* it's so unforgiving. We've only been widening B's catch (branch A).
The literature's strongest chained-manipulation results **also tighten A's delivery** (branch
B) and **switch on a learned feasibility signal, not a clock** (branch D). Those two are our
highest-leverage untried moves.

## Sources
- Lee, Yamada, Lim, Lim — *Adversarial Skill Chaining for Long-Horizon Robot Manipulation via
  Terminal State Regularization*, CoRL 2021. https://arxiv.org/abs/2111.07999
- Chen, Ci, et al. — *Sequential Dexterity: Chaining Dexterous Policies for Long-Horizon
  Manipulation*, CoRL 2023. https://arxiv.org/abs/2309.00987
- Röstel, et al. — *Composing Dexterous Grasping and In-hand Manipulation via Scoring with a
  Reinforcement Learning Critic*, 2025. https://arxiv.org/abs/2505.13253
- Lee, et al. — *Training Transition Policies via Distribution Matching for Complex Tasks*,
  ICLR 2021. https://arxiv.org/abs/2110.04357
- Silver, Allen, Tenenbaum, Kaelbling — *Residual Policy Learning*, 2018.
  https://arxiv.org/abs/1812.06298
- Mysore, Mabsout, et al. — *Regularizing Action Policies for Smooth Control with RL (CAPS)*,
  ICRA 2021. https://ai.bu.edu/caps/ ; *Grad-CAPS*: https://arxiv.org/abs/2407.04315
