# In-hand reorientation — research state & handoff (updated 2026-06-09 eve)

Living handoff doc for a FRESH session **and a self-contained brief for an external analyst.**
Full chronological log: `docs/rl/reorientation.md`. Published narrative: `webpaper/src/rl.typ`.

## Problem & goal (read first — no repo context assumed)
**System.** A 3-finger robot hand, **9 DOF, finger-only** (the palm is fixed — no arm, no wrist
translation). Simulated in MuJoCo-Warp; policies are PPO (rsl_rl). **Task:** a screwdriver-shaped
**cylinder lying flat** on a surface must be brought **upright (vertical)**, entirely **in-hand**.

**Why two policies.** A single end-to-end policy never learned grasp→lift→reorient together. We
decomposed it into two PPO policies that run in sequence at deploy:
- **Policy A — lift + deliver.** Grasps the flat cylinder, lifts it clear, and holds it at a
  delivery pose. Works well; **kept FROZEN** (`...20260529-1219...stable_v1/.../model_500.pt`).
- **Policy B — reorient.** Takes the held cylinder and rotates it to vertical. Best standalone
  reorienter B4 reaches held-vertical cosine **0.988** — but only when its *training* env resets it
  into a good, centered, already-held state at the start of every episode.

**The handoff "seam" = THE open problem.** At deploy we run A for ~40 steps, then switch control to
B with no reset ("continuous handoff"). **B drops the object within 3–5 steps of taking over.** The
two pieces are each excellent in isolation; the *seam between them* is not solved.

**Why it's hard (the crux).** B is excellent on its training distribution (a clean held start) and
useless just off it (A's *actual* delivered state: a slightly off-center grip, and — decisively — a
different **observation schedule**, because B trains in a "skip-lift" env whose lift-command /
reference-pose observations differ from the normal-lift continuous deploy). It is the textbook
off-distribution collapse.

**Success metric.** Continuous A→B rollout (no reset); take the **minimum object-center height**
over the whole rollout. **`min-z > 0.05 m` = the object stayed in the hand = seam closed.** Best
achieved so far is **0.0073** (branch-B); everything else is ~0.003 (object on the floor). The bar
is ~7× above the best result — this is not a tuning gap, it is an unsolved problem.

**What's been tried (one line each; details below).** (a) Move **A → B's grip** (branch-B, un-freeze
A): trains clean, A migrates its grip only partway, seam stays open. (b) Move **B → A's delivery**
(adapt-B, state bank): trains clean, still drops — which *proved* the binding constraint is the
**observation schedule**, not the state. (c) Deploy-time blends / critic-gated switching: exhausted,
no effect. (d) **Onset-grip injection** — train B in the *normal-lift* env (obs schedule matches
deploy) **and** inject A's delivered state at the onset (state matches deploy): the combination
adapt-B/branch-B couldn't reach. RAN 2026-06-09 → **min-z 0.0081, a new best but seam still open.**
The residual gap is now narrowed to the *teleport* (we inject a STATIC snapshot — the bank has zero
velocities — while deploy hands off a moving state); the next step injects A's REAL velocities. See
START HERE.

Task object/scene: flat-laying `screwdriver_medium` cylinder; morphology run
`results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259`.

## ⚡ FRESH SESSION — START HERE (updated 2026-06-09 eve)
**The open problem is the A→B handoff seam (the reorienter B drops the object when A hands it
over). Standalone pieces both work; the seam between them does not.** Bar = continuous-handoff
**min-z > 0.05**; everything so far drops to the floor.

**TWO NEW RESULTS TONIGHT (2026-06-09 eve):**

**(1) Complete-state onset injection → min-z 0.0027 (WORSE than static 0.0081). The B-side
injection paradigm is SATURATED.** Following the Opus-4.8 analysis, I made the onset teleport
*Markov-complete*: fixed the recorder's silent velocity bug (it read a nonexistent attribute
`root_link_velocity_w` → zeros; correct is `root_link_vel_w`), added `robot_qvel`, captured A's
last action `a_last`, and made the inject write A's REAL obj_vel + finger/palm qvel AND override
the seam `last_action` obs. Measured: A's delivery velocity is TINY (1.6 cm/s, settled) but
`a_last` is SUBSTANTIAL (0.23 rad) — so the `last_action` mismatch (Opus's new finding) was the
bigger unaddressed OOD. **Yet fixing both gave 0.0027, not better.** (Run `20260609-1313-
policyB_onsetInjectFull_bankA_s40`, NaN-crashed @iter221 after holding healthily — object_height
0.067, align 38 — so model_200 is a fair late ckpt.) Since the env has no differenced/history obs
and position actuators carry no `act` state, the injected seam IS Markov-equivalent to organic
arrival — so **making the teleport more faithful does not close the seam → the seam is NOT a
missing-state-info problem.** This kills the "inject A's state into B" family (skip-lift bank
0.0028, static onset 0.0081, complete onset 0.0027 — all saturated ≪ 0.05).

**(2) CO-ADAPTATION is the new best — 0.0114 (free eval, existing ckpts).** Pairing the
independently-**migrated A** (`Atol20` = frozen-A finetuned toward B10's grip) with the
independently-**adapted B** (`Badapt` = B10 adapted to frozen-A's delivery) gives continuous
min-z **0.0114** at handoff@40 — a new best, beating EITHER SIDE ALONE: A-moved×B10 **−0.0001**,
frozenA×Badapt **0.0075**, baseline ~0.0029. **Moving BOTH toward each other is the lever**
(Lee 2021 / Röstel 2025 co-adaptation, confirmed empirically for free). ⚠️ HONEST CAVEAT: 0.0114
is still a DROP (≪ 0.05), not a hold — it's the best *relative* min-z, the object falls slightly
less far. And `Badapt` has NO stable post-seam holding grip (it drops by ~step 48), so the **weak
link is B CATCHING**, not A delivering (A's migrated grip holds fine through 0-40). Results in
`MEET_IN_MIDDLE_EVALS.txt`; videos `docs/rl/videos/reorient/meet_*.mp4`.

**OVERNIGHT BATCH IS RUNNING (launched 2026-06-09 18:13).** `scripts/overnight_batch.sh` (detached;
`overnight_batch.run.log`) runs a co-adaptation wave 1 ONE-AT-A-TIME, auto-evals each on continuous
min-z, appends to **`BATCH_RESULTS.md`**: (1) coadapt_B_toAtol20 [B warmstart Badapt + MIGRATED A's
delivery]; (2) B_complete_fromBadapt; (3,4) branchB w6/w4 A-migration pushes (eval vs Badapt);
(5,6) complete-state 2×2 ablation (velocity-only / last_action-only). **CHECK `BATCH_RESULTS.md`
FIRST next session.**

> **EARLY READ (run 1):** `coadapt_B_toAtol20` **stuck** — object_height FLAT at ~0.02 for all 53
> iters (not climbing toward a hold, not floor-collapsed), align stayed 0 → a drop-post-inject local
> optimum; the watchdog culled it at iter 50 (correctly — flat, not learning). Undertrained ckpt
> still evaled **0.0076**. **Hypothesis (the ablations test it): warmstarting the fragile static-adapted
> `Badapt` + injecting the 0.31-rad `last_action` override shocks B into dropping.** So runs (5,6) —
> does removing the `last_action` override recover holding? — are now the **most informative** runs.
> Likely **wave 2**: retry coadapt with `INJECT_LASTACT=0` and/or warmstart B10 (not Badapt).

**TOMORROW — TWO PRIORITIES:**
- **A. Design wave 2 from `BATCH_RESULTS.md`.** If `coadapt_B_toAtol20` beats 0.0114, iterate the
  co-adaptation loop (record the new B's catch, re-migrate A to it, repeat — alternating A/B).
- **B. BUILD THE LIVE-A RESET (the one untried mechanism that removes ALL teleport artifacts).**
  Every adaptation so far trains B on a *teleport* into the seam (bank/inject); the deploy seam is
  *organic* (A runs live). Even Markov-complete injection (result 1) didn't close it — the remaining
  suspect is the contact-solver warmstart / one-step contact-force ramp that NO instantaneous
  teleport reproduces. The fix: B's training env runs **frozen Policy A LIVE** for steps 0..40
  (real physics, real contacts, real `last_action` — zero teleport), then B's PPO rollout begins at
  the seam. IMPLEMENTATION (needs rsl_rl integration, NOT a flag — that's why it's a build, not a
  queued run): load A's actor in the training process; per-env, while `episode_length_buf < onset`,
  apply A's action instead of B's; **MASK those pre-onset steps from PPO** (zero their advantages/
  returns in the rollout storage, else PPO trains B toward A's lift actions). Sanity check first:
  eval B10 with its episode STARTING from a live-A delivery vs from skip-lift spawn — if it holds
  longer from live-A, that's direct proof the seam is the train/deploy distribution. (Opus's
  central recommendation; deferred tonight because unattended rsl_rl surgery is too bug-prone.)

**(adapt-B / onset-inject history retained underneath for the record.)**

**adapt-B-to-A HAS NOW RUN (2026-06-08 eve) — RESULT: SEAM STILL OPEN.** Trained clean to iter 270
(`results/rl/20260608-1738-policyB_adaptToA_bankA_s40/tensorboard/model_270.pt`, object held +0.012
throughout its own skip-lift env, no collapse). But the decisive continuous-handoff eval gave
**min-z = 0.0028 m** (object z=0.0999 at handoff step 40, then falls to floor) — essentially tied
with B10-alone (0.0029), WORSE than branch-B tol20 (0.0073), ≪ 0.05 bar. Video:
`docs/rl/videos/reorient/handoff_adaptB.mp4`. (Infra note: it had failed twice at startup on a
wedged-`nvidia_uvm` CUDA-context error — NOT a too-fast-relaunch transient; needed a module reload,
see gotcha #12. Don't waste relaunches on it.)

**WHAT THIS PROVES (the diagnosis is now sharp):** putting A's real delivery STATE into B's training
distribution (via the bank) is NOT enough. So the binding constraint is NOT the state — it's the
**skip-lift OBSERVATION SCHEDULE** (the known B6 obs-OOD). adapt-B trained in skip-lift sees a
different lift-command / ref_object_pose obs trajectory than the normal-lift continuous deploy, even
when the underlying physical state matches. That obs discontinuity at the seam is what drops it.

**onset-grip injection RAN 2026-06-09 — min-z 0.0081 = NEW BEST, but SEAM STILL OPEN.** Mechanism
built + committed (56b8361): train B in the *normal-lift* env (obs schedule == deploy) AND inject A's
delivered state at the onset (state == deploy) — `mjlab_terms.inject_handoff_bank_at_onset`
(step-mode event), `env_cfg.handoff_onset_bank/_step`, `scripts/train_handoff_onset_inject.sh`
(warmstart B10). Ran clean to iter 270 (`results/rl/20260609-1113-policyB_onsetInject_bankA_s40/
tensorboard/model_270.pt`; in training B HELD + reoriented from the injected state — object_height
0.068, align 18.3). Continuous-handoff eval (frozen A → this B, handoff@40): z@40 0.113,
**min-z 0.0081** — beats branch-B tol20 (0.0073), adapt-B (0.0028), B10-alone (0.0029), but still
≪ 0.05; object reaches the floor (~8 mm). Video `docs/rl/videos/reorient/handoff_onset.mp4`.

**THE SHARPENED GAP (next step lives here).** B holds+reorients fine from the INJECTED state in
training, yet drops at the ORGANIC deploy handoff. The only remaining train/deploy difference is the
teleport: **we inject a fully STATIC snapshot** — and worse, the bank itself is static by accident.
`results/rl/handoff_state_bank_A_s40.npz` has `obj_vel` **exactly 0.0** across all 2048 states
(recorder `rl_record_handoff_states.py:76` reads `obj.data.root_link_velocity_w`, an attribute that
misses → silent `torch.zeros` fallback) and **no `robot_qvel` at all** (finger velocities never
recorded); the injection then also zeroes finger vel. So in training B always sees a motionless
seam, while at deploy A hands off a state still in motion with a continuous contact history.

**~~IMMEDIATE NEXT ACTION: inject A's REAL delivery velocities~~ — DONE 2026-06-09 eve (see the
two new results at the top).** Both halves ran: (1) recorder fixed (`root_link_vel_w` + `robot_qvel`
+ `a_last`), inject writes real velocities AND overrides `last_action`; result min-z **0.0027**
(worse than static 0.0081 — A's velocities were tiny/settled, the `last_action` mismatch was real
but fixing it didn't help → injection paradigm saturated). The flagged fallback (trajectory replay /
organic arrival) generalizes to the **LIVE-A RESET**, now tomorrow's priority B. Tooling left in
place: `--handoff-inject-velocity` / `--handoff-inject-last-action` toggles ablate the two variables
(queued in the batch).

Schedule used (tunable): onset/inject @40 (matches s40 bank + deploy handoff@40), residual from 40,
reorient reward from 45 (5-step grace).

**Reference numbers to beat (continuous-handoff min-z, handoff@40, bar = 0.05):**
baseline frozenA×B10 0.0029; adapt-B 0.0028; static onset 0.0081; complete-state onset 0.0027;
A-migration alone (Atol20×B10) −0.0001; **CO-ADAPT cross-pairing (Atol20×Badapt) 0.0114 = BEST**.

**Pieces (all paths real, verified):**
- Frozen Policy A (lift+deliver, GOOD — keep frozen): `results/rl/20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt`
- B4 = best standalone reorienter (0.988): `results/rl/20260603-1746-policyB_p2_lateral_only/tensorboard/model_541.pt`
- B10 = first to survive delivery + reorient but violent: `results/rl/20260604-1642-policyB_holdonlyws_repro/tensorboard/model_541.pt`
- A's real-delivery state bank (grip+pose, 2048 states, z=0.111): `results/rl/handoff_state_bank_A_s40.npz`
  (OLD, static/zero-vel) — **use the COMPLETE bank now: `results/rl/handoff_state_bank_A_s40_full.npz`**
  (real obj_vel + robot_qvel + a_last, 2048 states).
- adapt-B (skip-lift bank) result that just closed off this branch: `results/rl/20260608-1738-policyB_adaptToA_bankA_s40/tensorboard/model_270.pt` (min-z 0.0028)
- **Atol20** = migrated A (frozen-A → B10's grip, branchB tol20): `results/rl/20260605-1609-policyA_unfreezeA_v2_w2_tol20/tensorboard/model_270.pt`
- **Badapt** = adapted B (B10 → frozen-A delivery, static onset): `results/rl/20260609-1113-policyB_onsetInject_bankA_s40/tensorboard/model_270.pt`
- **Atol20's delivery bank** (complete, for co-adapting B to the migrated A): `results/rl/handoff_state_bank_Atol20_s40_full.npz`
- ⚠️ `results/rl/badapt_initiation_s48.npz` is GARBAGE (Badapt drops by step 48 → recorded a floor grip; do not use as a branchB target).

## TL;DR — what's true now
- **Best reorientation policy = `p2_lateral`** (held-vertical cos **0.988**, peak **0.999**,
  obj_jerk **25.8** = HALF the prior best, no drop):
  `results/rl/20260603-1746-policyB_p2_lateral_only/tensorboard/model_541.pt`.
  Recipe = signed+critic + `--lateral-drift-weight=-8` ALONE (one constraint at a time).
  Surprise: the lateral penalty did NOT reduce de-centering (it actually drifted ~1 cm more);
  what it DID do was act as a **smoothing regularizer** — halving object jerk while pushing
  verticality up. (Prior best was `signed+critic` model_405 at 0.978/jerk 51.6.)
- **THE bug that wasted v2:** the warmstart loaded only the actor and **discarded the critic**;
  a fresh value function knocks the converged actor off its optimum. Fixed: `--warmstart-critic`
  (default ON). Always warmstart the critic.
- **Judge on deterministic behavior, not training reward sums.** Reward sums conflate
  how-long×how-aligned and hid a verticality regression. Use
  `scripts/rl_eval_reorient_metrics.py` (held_cos / peak / obj_jerk / min_z / drop).
- **Open problems (each unsolved, all documented):**
  - *Smoothness:* jerk-penalties are counterproductive — the corrective finger jerk IS the
    stabilization; penalizing it makes the hold slip/wobble. Use a non-reward lever
    (action low-pass at deploy, or motor-delay/obs-noise DR), not a reward.
  - *De-centering (REAL):* still open. P2's `--lateral-drift-weight=-8` did NOT curb it
    (5.6/4.8 cm world vs baseline 3.8/5.2). **The best de-centering lever found is the
    statebank** (P3: 3.0/3.1 cm ≈ 4.3 cm 3D) — training on A's real centered grips keeps it
    centered — but P3 reorients worse (0.930). Still: stacking diverges; add ONE at a time.
  - *Bracing:* unchanged; reaches cos 0.99 ~3 cm below palm (vertically), no contact.
  - *Seamless A→B handoff:* **STILL OPEN. Latest (2026-06-09 eve): injection paradigm SATURATED
    (complete-state onset 0.0027); CO-ADAPTATION is the new lever (0.0114 best). See START HERE +
    `BATCH_RESULTS.md`. The history below (smoothness framing, 2026-06-04) is superseded.**
    Diagnosis
    held: seam drop was an observation-discontinuity OOD shock (skip-lift B OOD on A's normal-lift
    delivery). The grace-window-from-P2 attempts (v2/v3/v3b = B7/B8/B9) all collapsed because they
    warmstarted the OOD skip-lift reorienter. **THE FIX: warmstart the HOLD-ONLY control** (proven
    to survive the takeover), partial-load 65→66-d (zero-init reorient col, actor+critic).
    **B10** (hold-only ws, hard onset) = **first policy to SURVIVE the handoff AND reorient**
    (held-cos **0.977**) — but **VIOLENT** (obj_jerk **108** vs B4's 27). **B11** (soft, residual
    0.4, α-curric 0.5→4/150it) holds but **never tilts** (held-cos −0.137: too dilute). Now a
    smoothness problem: find the point between B10 (commits, violent) and B11 (smooth, won't
    commit). **ITER2 RAN (B12/B13) — BOTH FAILED, and it retired the "B10 survives" optimism.**
    - **B12** (smoothness finetune OF B10, learn-then-smooth like B2): **catastrophic** — held-cos
      **0.086** (tilt destroyed), obj_jerk **550** (3× B10, *worse*). The sim-jitter gotcha at full
      force: the corrective finger jerk IS the stabilization; penalizing it makes B thrash.
    - **B13** (soft-but-committing, residual 0.5, α 0.5→4/40it): **smooth but timid** — obj_jerk
      **21.3** (below B4's 27, great!) but held-cos **0.462** (half-tilt, never reaches vertical).
    - **THE decisive new number — continuous-handoff min-z (>0.05 = survives): EVERYTHING FAILS.**
      B10 hard **0.0029**, B10 blend-12 **0.0063**, B10 critic-gate@29 **0.0044**, B12 blend-8
      **0.0026**, B13 blend-8 **−0.0007**. So **B10 does NOT survive the continuous handoff** either
      — the earlier "B10 survives" read (held-cos 0.977) was from the *standalone* env (resets to a
      good held state); under the strict continuous A→B rollout B10 drops the object to the floor.
    - **Branch E (deploy-time levers) is EXHAUSTED:** action-blend window (8–12 step ramp) and
      critic-gated switch on B10 both leave min-z ≪ 0.05. Softening *when/how-fast* B takes over does
      not fix *what* B does once holding.
    - **DIAGNOSIS sharpened:** B10 (hard commit), B13 (soft commit), B12 (over-smoothed) span the
      whole commit/smoothness axis and ALL drop in the continuous handoff → moving B alone is not
      enough. The seam state itself (A's actual delivery) is the unfixed variable.
    - **THE single best next experiment = branch B (un-freeze Policy A).** Keep B10 frozen as the
      reorienter; fine-tune A in the continuous env with terminal-state regularization — penalize A
      for ending its lift outside B10's initiation set, using B10's critic value at the seam as the
      reward (Lee 2021 / Röstel 2025). Cheaper pairing: record A's REAL delivered seam states
      (`rl_record_handoff_states.py`) and reset B's normal-lift training from that bank
      (`--handoff-state-bank`) to close the train-reset-vs-real-delivery gap. Budget spent; NOT run.
    NOTE: standalone skip-lift eval env is OOD for normal-lift Bs (drop=1.0 is an artifact); judge
    survival on continuous-handoff min-z, not standalone drop. Full write-up: `webpaper/src/rl.typ`;
    eval: `STATE_HANDOFF_RESULTS.txt`.

## BRANCH B v1 (un-freeze A) — RAN 2026-06-04 eve, FAILED (collapsed A's grasp); v2 corrected, NOT YET RUN
Branch B was implemented + launched (commits f6abb13/14a22db/c920706): keep B10 frozen, warmstart A,
add a seam-gated (steps 33-37) dense reward (`handoff_target_proximity`) pulling A's finger qpos onto
B10's recorded step-35 grip (`results/rl/b10_initiation_bank_s35.npz`). Sweep grip weight 4/8/16
(`scripts/{train_handoff_branchB_unfreezeA,sweep_branchB_unfreezeA}.sh`).
**Hypothesis (still UNTESTED):** the seam OOD is the GRIP (finger qpos), not object pose — at handoff
A's grip is ~0.16 rad/joint off B10's holding grip (A scores 0.347 vs B10-self 1.0 at tol 0.05).

**v1 OUTCOME — all 3 runs (`20260604-22{05,06,10}-policyA_unfreezeA_gripw{4,8,16}`) FAILED two ways:**
1. **FATAL — destabilized A's grasp; collapse to floor.** Identical curve across weights: A lifts
   fine early (object_height ~0.09, contact_min 6.6 at iter ~15) then **collapses at iter ~25**
   (object_height → 0.012-0.018, contact_min → 0.008) and **never recovers**. Root cause: the launch
   script **stripped ALL of A's lift-phase terminations** (never passed `--enable-lift-terminations`)
   on the theory lift_height+track_object rewards hold the grasp — they DON'T. With no drop
   termination (only `time_out` ever fired) and no drop penalty (`object_drop` reward 0.0),
   **object-on-floor is a stable non-terminating attractor**; PPO + the finger-perturbing proximity
   reward slid A into it in 25 iters. (= gotcha #4 + new lesson #7 below.)
2. **OOM kill (rc=137).** All 3 SIGKILLed at ~8-16 min (ETA 30-44 min left), never reached 20M ts.
   Three parallel 3072-env runs over-subscribed memory → **run ONE at a time.**
**Crucially the hypothesis was NEVER tested:** `handoff_target_proximity` stayed at ~0.014-0.05 the
whole time (once the object is on the floor the fingers aren't near B10's grip), so v1 only proved
*this env destabilizes A*, nothing about whether grip-matching closes the seam.

**v2 FIX (`scripts/train_handoff_branchB_v2.sh`, written, NOT launched) — change ONE thing vs A's
own training env:** A was TRAINED with `enable_lift_terminations:true, term_object_drop 0.02,
term_finger_slip 0.3, lift_phase_start_step 40`. v2 **restores all of it** (floor can't be an
attractor) and relaxes **only** `term_finger_slip` 0.3→2.0 (so A may migrate its grip ~0.48 rad L2
toward B10's without the slip-term killing it — the one real tension v1 spotted but "fixed" by
deleting every guardrail). Worst case now = "A keeps its grip, proximity stays low" (safe), never "A
learns to drop". Weight MODEST (2.0, not 4/16); SINGLE run (OOM fix); baked-in **collapse watchdog**
kills the run if object_height < 0.045 at iter ≥ 24. Bank verified sane (512 samples, 9 finger
joints, target grip deterministic std≈1e-4). After it holds: eval seam with frozen B10 via
`rl_demo_handoff_continuous.py --policy-a <A_v2> --policy-b <B10>` → min-z > 0.05 = SURVIVES.
**Fallback if A is too fragile to nudge:** don't touch A — record A's REAL delivered grip and
fine-tune B with its onset grip domain-randomized over A's delivery distribution.

**v2 RAN TO COMPLETION 2026-06-05 (`20260605-1608/1609-policyA_unfreezeA_v2_w2_tol{15,20}`,
20M/3072) — TRAINS CLEAN, BUT SEAM STILL OPEN.** Took THREE fixes to get a trainable run:
(1) restore A's drop/tip-loss terminations (v1 stripped them → floor attractor); (2) relax ONLY the
object-slip guards 0.015→0.05 m (enabling terminations re-introduced A's tight 1.5 cm slip term,
which fired 100+/iter on the grip migration → killed every episode @iter1); (3) widen the proximity
basin qpos_tol 0.05→0.15/0.20 (at 0.05 the reward was ~3σ-flat at the 0.16-rad grip gap → no
gradient, proximity stuck at 0.002). With all three: **both runs held the object the whole run**
(object_height ~0.10) and migrated their grip PARTWAY (proximity 0.002→0.026 tol15 / 0.037 tol20,
then plateaued). **Continuous-handoff eval vs FROZEN B10 (model_541, handoff@40):**
| A policy | z@handoff | min-z | survives(>0.05) |
|---|---|---|---|
| B10 alone (baseline) | 0.112 | 0.0029 | ✗ |
| tol15 | 0.109 | 0.0049 | ✗ |
| tol20 | 0.108 | 0.0073 | ✗ |
Survival rose monotonically with migration (0.0029→0.0049→0.0073) but only ~2.5× on a number that
needs ~7× — object still hits the floor (~7 mm). **VERDICT: grip-match is REAL but INSUFFICIENT.**
A resists fully adopting B10's grip (proximity plateaued despite 271 iters — moving further off A's
own grasp drops the object; A's lift/contact rewards outweigh a modest grip nudge), and pushing the
weight harder re-invites collapse. Videos `docs/rl/videos/reorient/handoff_branchB_tol{15,20}.mp4`.
**RECOMMENDED next (the symmetric move, leave A's good grasp alone): adapt B to A's delivery** —
record A's REAL delivered seam states+grip (`rl_record_handoff_states.py`) and fine-tune B10 with
resets / onset-grip DR drawn from that bank, so B becomes robust to A's actual grip instead of
forcing A onto B's. Data now favors this over more A-side pushing (A won't migrate further).

## ADAPT B TO A — RAN TO COMPLETION 2026-06-09, SEAM STILL OPEN (this branch is now CLOSED)
The chosen direction: leave A frozen, make B robust to A's real delivered grip. Built, validated,
and now run. **Result: it does NOT close the seam — and that is the informative finding.**
- **Setup:** recorded A's real delivered states from FROZEN A model_500 in the normal-lift env
  @step40 → `results/rl/handoff_state_bank_A_s40.npz` (2048/2048 kept, object z med 0.111,
  `robot_qpos` (2048,15) incl. grip; verified). Launch `scripts/train_handoff_adaptB_to_A.sh`:
  warmstart B4 (0.988) + B4's exact skip-lift knobs (target-axis 100 / progress 300, lateral −8) +
  `--handoff-state-bank` (activates bank reset in skip-lift) = "B6 done right".
- **Result (run dir `20260608-1738-policyB_adaptToA_bankA_s40`, iter 270):** trains perfectly clean
  — object held +0.012 in its own skip-lift env the whole run, no collapse. But continuous-handoff
  eval (frozen A → this B, handoff@40) gives **min-z = 0.0028** (z=0.0999 at the seam, then floor):
  tied with B10-alone (0.0029), worse than branch-B tol20 (0.0073), ≪ 0.05. Video
  `docs/rl/videos/reorient/handoff_adaptB.mp4`.
- **WHAT IT PROVES → the next experiment.** Putting A's real delivery *state* into B's training
  distribution (the bank) is NOT enough, because the bank only fires in **skip-lift** (env_cfg.py
  l.1140 gates `reset_from_handoff_bank` on `skip_lift_phase and handoff_state_bank`). So B still
  trained under the skip-lift OBSERVATION schedule, which differs from the normal-lift deploy even
  when the physical state matches. **The binding constraint is the obs schedule, not the state.**
  The one untried combination — **state in-distribution AND obs in-distribution** — is the
  normal-lift onset-grip injection (see START HERE). adapt-B + branch-B together rule out moving
  either policy under the OLD mechanisms; the next step changes the mechanism.
- **Infra footnote:** the run failed twice at startup before this on a **wedged `nvidia_uvm`**
  (`CUDA unknown error`, `torch.cuda.is_available()` False while nvidia-smi works) — NOT the
  context-after-kill transient (#8); needs a module reload (gotcha #12), which fixed it.

## P1/P2/P3 — DONE (2026-06-03, 40M ts / 3072 envs each). Authoritative deterministic eval:
| policy | held_cos | peak | obj_jerk | min_z | drop | world Δlat |
|---|---|---|---|---|---|---|
| baseline signed+critic (405) | 0.979 | 0.988 | 51.6 | 0.109 | 0 | 3.8/5.2 |
| P1 handoff-DR alone (541) | 0.954 | 0.994 | 59.1 | 0.115 | 0 | — |
| **P2 lateral-only (541)** | **0.988** | **0.999** | **25.8** | 0.117 | 0 | 5.6/4.8 |
| P3 statebank (541) | 0.930 | 0.944 | **8.4** | 0.114 | 0 | **3.0/3.1** |
- **P1 (handoff-DR alone): worse, discard.** DR-alone destabilized the grip (training term
  stats: drop 16.4, floor 11.25) — confirms gotcha #4.
- **P2: new best reorienter** (verticality + smoothness). De-centering unchanged.
- **P3: best de-centering + smoothest, but weakest reorienter.**

## HANDOFF DIAGNOSIS (the one remaining open problem) + the fix in training
Instrumented the continuous A→B rollout (z every step). The drop is **instantaneous at the seam**,
identical for P2/P3/baseline (all skip-lift trained):
```
step 40 z=0.111 (A holding) → 45 z=0.094 (handoff) → 46 z=0.073 → 48 z=0.022 → 50 z=0.010 (floor)
```
B collapses the grip within **3–5 steps** of taking over → it's an OOD shock, not grip weakness.
That's why neither DR (P1) nor statebank (P3) helped: both still trained B in the **skip-lift**
env, whose lift-command phase / `ref_object_pose` schedule differs from the **normal-lift** env
used at deploy. B never saw the seam obs in training. **PROOF:** the (undertrained, 15M/1024)
`20260603-1315-policyB_normallift` B — trained in the normal-lift env with residual gated to
activate at step 35 — held **z≈0.09 for ~10–15 steps past the seam** (vs instant collapse), then
dropped (undertrained). So normal-lift training removes the shock; it just needs convergence.

### Normal-lift B history: v2 collapsed → v3 grace (NaN'd) → **v3b ran to completion, BOTH COLLAPSED**
- **v2_fromP2 (normal-lift, warmstart P2): COLLAPSED** — held-cos 0.029, 100% drop, handoff
  min-z 0.005. Step-35 fired residual+terminations+full-reorient at once; OOD warmstart fumbles,
  terminations kill episodes → reward 12→3 → never learns.
- **v3 grace window: looked promising, NaN'd before it could be judged** — B takes over (residual)
  at step 35 but only HOLDS until step 50, when terminations+reorient engage. Reward stayed
  flat (~10) for 60 iters (no v2 *training* collapse), then **NaN-crashed at iter ~60/750**.
  Only model_50 (undertrained, drops).
- **v3 hold-only control (reorient OFF): completed, PROVES B survives the handoff** — tip_lost
  humped to ~44 then recovered to ~1–4. (65-dim, not deployable; isolation control only.)
  **→ this is the warmstart to use next.**
- **v3b (`policyB_normallift_v3b_{repro,soft}`, 40M/3072, completed 2026-06-04): BOTH COLLAPSED.**
  NaN-resilience worked (both ran the full 542 iters, no crash), but that *revealed* the v3 "flat
  ~10 reward" was a degenerate plateau, not learning: reward stayed flat ~9.3 (R)/~9.0 (S) the
  whole run. Deterministic held-cos **−0.035 (R) / −0.078 (S)**, **100% drop**; handoff min-z
  **0.0037 / 0.0069** (≪ 0.05). Training at convergence: floor_proximity term 95.9/58.9,
  object_height 0.012/0.015 m, success 0. **Mechanism:** grace window stopped the v2 collapse but
  locked B in a *hold-during-grace, drop-at-reorient-onset* local optimum — it can't cross from
  "hold" into a working post-seam reorient from the OOD skip-lift P2 prior. Soft onset (S) bought
  nothing. Videos `handoff_v3b_{repro,soft}.mp4` (show the drop). Full writeup: reorientation.md
  "v3b OUTCOME".
- **Best untried next experiment (NOT run, budget spent):** finetune the **v3 hold-only**
  checkpoint toward reorient (it already holds A's delivery → grace→reorient transition is
  in-distribution), instead of the OOD skip-lift P2. Backup: add P3's handoff **state-bank** to
  the normal-lift env (train on A's real seam states). Longer training is least promising (reward
  was *flat*, i.e. a local optimum, not undertraining).

## How to EVALUATE (the honest metrics)
```
# held-cos / jerk / min_z / drop  (deterministic) — the authoritative comparison:
WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python \
  scripts/rl_eval_reorient_metrics.py "name=<run_dir>:model_<N>.pt" ...
# seamless A→B handoff (no reset); reports object-z at handoff + min-z (hold = min-z>0.05):
WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python \
  scripts/rl_demo_handoff_continuous.py --policy-b <ckpt> --output <mp4> --handoff-step 45 --total-steps 240
# de-centering (palm-frame lateral excursion): see /tmp/diag2.py pattern in reorientation.md.
# single-policy reorient video: scripts/rl_render_reorient.py --run <dir> --checkpoint model_N.pt --output <mp4>
```
Reference numbers: v1 held 0.96/jerk 41; signed+critic 0.978/52; de-center w40 cut drift but on a
degraded base. **min_z<0.05 ⇒ floor contact/drop.**

## Tooling / knobs (all in `scripts/rl_train_cube.py`, env in `src/morphohand/rl/env_cfg.py`)
- `--warmstart-critic` (default ON), `--init-actor-checkpoint`.
- reorient: `--enable-target-axis-reward --target-axis-weight --target-axis-progress-weight`
  (signed by default), `--skip-lift-phase`, `--reorient-start-step`.
- de-centering: `--lateral-drift-weight/-deadband/-power` (palm-frame, quadratic past deadband).
- handoff DR: `--skip-lift-spawn-tilt-jitter --skip-lift-spawn-z-jitter --handoff-dr-curriculum-iters`.
- train-the-handoff: `--handoff-state-bank <npz>` (record via `scripts/rl_record_handoff_states.py`).
- bracing (built, geometry-limited): `--brace-force-weight --brace-distance-weight --grip-force-weight`.
- smoothness (don't rely on it): `--action-rate-weight --object-ang-acc-weight --*-final --smoothness-curriculum-*`.
- finger-residual gating: `--finger-residual-active-from-step` (zero residual during scripted lift).

## GOTCHAS (do not relearn these)
1. **Always `--warmstart-critic`** (default ON) — actor-only warmstart wrecks finetunes.
2. **Parallel training:** give each run its OWN Warp cache `WARP_CACHE_PATH=$(mktemp -d)` — a shared
   cache races and NaNs. VRAM is cheap: 3×3072-env runs ≈ 11 GB / 50% on the 16 GB GPU; push more
   (4096 envs and/or 4 parallel). Stagger launches ~60-90 s so kernel compiles don't pile up.
3. **Judge on `rl_eval_reorient_metrics.py` (deterministic held-cos), never reward sums.**
4. **Stacking objectives diverges** — the finger-only reorient is fragile; add ONE new
   constraint (DR, or lateral, or brace) at a time; warmstart from a stable base.
5. **Launch training DETACHED** (`nohup setsid bash … >log 2>&1 </dev/null & disown`) — SSH/laptop
   sleep otherwise kills it (and the harness-tracked jobs).
6. **The "revert gremlin":** SSH reconnects have reverted tracked files (env_cfg.py / rl_train_cube.py)
   to stale versions mid-session. **Commit after every change**, and if a file looks short/old,
   `git checkout HEAD -- <file>` to restore. HEAD is the source of truth.
7. **NEVER strip the grasp guardrails (drop/tip-loss terminations) when adding a finger-perturbing
   reward** (branch-B v1, 2026-06-04). Without a drop termination AND with episodes not ending on
   floor-contact, "object on the floor" is a stable non-terminating attractor and PPO collapses into
   it within ~25 iters. If a new reward conflicts with a *specific* guardrail (here `term_finger_slip`
   vs grip migration), relax ONLY that one — keep `--enable-lift-terminations` + `term_object_drop`.
   Warmstart finetunes can also collapse silently: watch `Metrics/lift_height/object_height` live.
   NB `--enable-lift-terminations` also re-arms the TIGHT object-slip guards (`term_object_slip_xy`
   0.015 m / `_yaw` 0.5 rad) — these fire on any reward that moves the object; relax them too
   (e.g. 0.05 m / 1.0 rad) if your reward perturbs the grasp.

## INFRA / LOGISTICS GOTCHAS (run-starting — cost real time this session)
8. **CUDA-context not released after killing a Warp run → next launch fails with a bogus alloc error.**
   `RuntimeError: Failed to allocate <small> bytes on device 'cuda:0'` while `nvidia-smi` shows GB
   free is NOT real OOM — the just-killed Warp process still holds the context. After killing a run,
   **wait until `nvidia-smi` memory drops back to baseline (~1 GB) AND no python/warp procs remain**
   before relaunching (a `sleep 4` is too short; give 15–30 s and verify). This is the bug that made
   "nothing run" on 2026-06-05.
9. **Do NOT `pkill -f "<pattern>"` when `<pattern>` also appears in the killing command's own line** —
   `pkill -f` matches the parent shell running your command (its cmdline contains the pattern) and
   kills YOUR shell (seen as exit 144). Kill by explicit PID, or by the process *group* of a known
   PID (`kill -TERM -<pgid>`), never by a self-matching `-f` pattern.
10. **Watchdog/collapse thresholds are LIFT-MODE-SPECIFIC.** Normal-lift: object spawns on the floor
    and lifts, so `Metrics/lift_height/object_height` (height ABOVE init) is ~0.09 when held → drop =
    `< 0.045`. Skip-lift (`lift_target_z_above_init=0`): object spawns already lifted, so the same
    metric is ~0.00 when held and goes NEGATIVE (~−0.10) on a drop → drop = `< −0.06`. A normal-lift
    threshold on a skip-lift run false-fires and kills a healthy run. Also: when grepping the value,
    keep the sign (`grep -oE "\-?[0-9.]+$"`), or you can't see negatives.
11. **Parallelism: ≤2 concurrent 3072-env runs.** THREE killed each other via OOM (rc=137, branch-B
    v1). Two fit (~8 GB / 16 GB). Each needs its OWN `WARP_CACHE_PATH=$(mktemp -d)` and a staggered
    launch (gotcha #2). The baked-in object_height watchdog (in the train scripts) auto-kills a
    collapsing run in ~3 min so you don't burn 40 — keep using it.
12. **`CUDA unknown error` after a hard-killed Warp run = wedged `nvidia_uvm`, NOT a wait-it-out
    transient.** Distinct from #8 (#8 clears with time; this does NOT). Symptom: `nvidia-smi` works
    and `torch.cuda.device_count()` returns 1, but `torch.cuda.is_available()` is **False** with
    `UserWarning: CUDA initialization: CUDA unknown error` — persists even with
    `CUDA_VISIBLE_DEVICES=0`, no Xid/ECC errors (GPU hardware is fine). The `nvidia_uvm` kernel
    module is wedged; only a RELOAD (or reboot) fixes it — waiting never does. This killed the
    adapt-B run twice at startup (`RuntimeError: CUDA not available`) on 2026-06-08 with the GPU idle.
    Fix (needs sudo; `gnome-remote-desktop` is a USER service holding `/dev/nvidia-uvm` open, so stop
    it first or `modprobe -r` fails "in use"):
    `systemctl --user stop gnome-remote-desktop && sudo modprobe -r nvidia_uvm && sudo modprobe
    nvidia_uvm && systemctl --user start gnome-remote-desktop`. Probe before relaunching:
    `uv run --extra gpu python -c "import torch; print(torch.cuda.is_available())"`.

## Reproduce the in-progress launches
The exact P1/P2/P3 commands are in `scripts/queue_reorient_handoff_dr.sh` / the run dirs'
`config.yaml`. All three: `--num-envs 3072 --total-timesteps 40000000 --init-actor-checkpoint
results/rl/20260602-1636-policyB_abl_signed/tensorboard/model_405.pt` + the per-path knobs above.
