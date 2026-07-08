# In-hand reorientation — research state & handoff (updated 2026-06-22)

Living handoff doc for a FRESH session **and a self-contained brief for an external analyst.**
Full chronological log: `docs/rl/reorientation.md`. Published narrative: `webpaper/src/rl.typ`.

> **⚠️ STATUS BANNER (read before the problem statement below).** The A→B handoff "seam"
> that the *Problem & goal* section frames as "THE open problem" was **SOLVED on 2026-06-10**
> (the live-A reset — the policy now holds the handoff at post-handoff min-z ~0.11 m). The
> later force/grip-quality work (b32→b34, 2026-06-12/13) chased a **phantom** (B3/B4 are NOT
> gentle — see the correction below). As of **2026-06-22** the project has **pivoted** to the
> user's actual goal — a *smooth, LOW-FORCE* grasp+reorient (verticality de-prioritised) — with
> two runs in flight, then morphology optimisation. **The authoritative current state is the
> `2026-06-22` section immediately below;** everything under the older `START HERE` dates is
> retained as the historical record.

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
*post-handoff* (the honest hold metric — whole-rollout min-z is dominated by the pre-lift floor
phase; gotcha #13). **`min-z > 0.05 m` = the object stayed in the hand = seam closed.**
*(Historical note: pre-2026-06-10 the best was 0.0073 and every teleport approach dropped — this
paragraph framed it as unsolved. The **live-A reset closed it on 2026-06-10**: post-handoff
min-z ~0.11 m, held continuously. The open problem is no longer the hold; see the 2026-06-22
section.)*

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

## ⚡ FRESH SESSION — START HERE (2026-06-22) — PIVOT TO SMOOTH / LOW-FORCE

### Where the project actually is (one screen, honest)
1. **The handoff seam is SOLVED** (2026-06-10, live-A reset). Policy B holds A's organic delivery
   at full height through the continuous handoff and reorients — post-handoff min-z ~**0.11 m**
   (≫ 0.05 bar). Every prior teleport approach dropped in 3–5 steps. *This is no longer open.*
2. **Best handoff reorienters (all hold; the trade is verticality ↔ smoothness ↔ force):**
   - **b32** (`b32_…gripSmooth_w4/model_405`): held-cos **0.895** (~25° off vertical) — but **jittery**
     (ang-jerk ~112) and clamps ~**11 N**.
   - **b29** (`b29_…commit60/model_405`): held-cos 0.78, gentler.
   - **b34_t20** (`20260613-0909-…forcefloor_t20/model_405`): the **gentlest** — fingertip **6.6 N**,
     ang-jerk 76, held-cos 0.78. This is the warmstart for the new gentle-B run.
3. **THE GRIP-QUALITY PREMISE WAS A PHANTOM (2026-06-13 correction — load-bearing).** We spent
   b32→b34 trying to make the grip "gentle like B3." **Direct measurement (`scripts/probe_grip_force.py`)
   demolished it:** B3 grips **7.04 N**, B4 **8.77 N** — *harder* than our handoff (6.6 N). Nobody
   "seats" (palm force = 0.00 N in ALL policies). Force does **not** cause jitter (B4 is the smoothest
   AND highest-force). Penetration is universal (the deliberately-soft contact solver, frozen —
   [[feedback_thumb_penetration_soft_contact]]). The "B3 ≈ 3 N" benchmark was a **misread of
   `grip_force_max=3.0`** (the reward saturation cap), not B3's real force. **So the entire force arc
   optimised a non-problem.** It was still informative: it proved force is **decoupled** from both hold
   and smoothness, and that for a **fingertip grip of this rod the force floor is ~6.6 N** — you cannot
   reward your way below it.
4. **The genuine remaining gaps** (vs B4 standalone 0.988 / smooth) are **verticality** (handoff ~0.78–0.90)
   and **smoothness** (handoff ang-jerk ~76–112 vs B4 ~26), both ceilinged by the **B10-warmstart basin**
   ("the violent survivor" — every seam-survivor descends from B10 because B10 is the only thing that
   survived the seam; B10 is inherently twitchy) plus the **marginal fingertip grip** of a 10 cm rod on
   3 smooth tips. Reward-shaping plateaus here (b29 nudged 0.75→0.78; jerk-penalty backfires; deploy
   low-pass drops it; force-penalty trades verticality).

### THE 2026-06-22 PIVOT (user directive — supersedes the verticality/brace push)
> *"i still don't love our best grasp and am still very frustrated that we can't train a seamless
> handoff that doesn't use excess force. at this point i don't care about a super close-to-brace
> pullup or a close-to-vertical reorientation, i just want a SMOOTH, LOW-FORCE grasp and reorient.
> … then we can start thinking about morphology optimization."*

**Reprioritisation:** drop the verticality and brace ambitions; optimise **smooth + low-force** for
both the grasp (A) and the reorient (B). **Honest framing the user accepted:** genuinely *low* force
(below the ~6.6 N fingertip floor) needs the object to **seat into the palm** so the palm bears load
→ fingers relax — which **this morphology can't do** (object sits 7–8 cm below the palm; `palm_brace_force`
has fired in 0 of all runs). **That is the morphology step, queued next.** What IS untried within the
current morphology: **nobody has ever relaxed verticality** — every run held alignment at +100,
maximising vertical, which is exactly what forces the tense corrective clamp + jitter. A *gentle
partial* reorient may relax the grip and smooth out for free. That is what this round tests.

### TWO RUNS IN FLIGHT (launched 2026-06-22) — both detached, watchdog'd, ≤2 concurrent (gotcha #11)
**Run 1 — gentle low-force REORIENT (Policy B).** `scripts/train_gentle_lowforce_B.sh`
(TAG `policyB_gentleLowforce`, log `gentleB_policyB_gentleLowforce.trainer.log`). Warmstart **b34_t20**;
live-A reset @ scale 0.2 (gotcha #13). Single coherent change-set vs b34: **relax verticality**
(target-axis-weight 100→40, alpha 4.0→**1.5** = wide basin so partial tilt already rewards, progress
300→120); **lower force** (grip-force REWARD 6→2, keep the over-grip PENALTY at thresh **2.5**); **smoother**
(lateral-drift −8→**−12**, the proven smoothing regulariser). Smoke = healthy (object held 0.111, penalty
fires −9, no NaN). **Eval:** `rl_demo_handoff_continuous.py --policy-b results/rl/<dir>/tensorboard/model_<N>.pt
--finger-residual-scale 0.2 --finger-close-easing linear --no-contact-gate` → post-handoff min-z + lin/ang
JERK + fingertip FORCE + held-cos. **WIN = holds (min-z>0.05) at materially lower force AND lower ang-jerk
than b34_t20, even at a lower held-cos** (that's the point — buying smooth+gentle WITH a lower cos).

**Run 2 — lower-force GRASP (re-open Policy A).** `scripts/train_lowforce_A.sh` (TAG `policyA_lowforce`,
log `lowforceA_policyA_lowforce.trainer.log`). User chose to re-open A (branch-B territory). Objective =
**can A lift+deliver with less fingertip force?** (A was trained force-unaware.) Warmstart a01; A's full
lift recipe; **add ONLY the over-grip penalty** (thresh **6.0** — milder; the smoke at thresh 5.0 showed
tip_lost volatility, so we shave the top first and sweep lower later). **Lesson #7 honoured:** keep ALL
grasp guardrails (drop/tip-loss terms), relax ONLY the precision-slip guards (finger-slip 0.3→2.0,
slip-xy/yaw) so the grip may re-shape; collapse watchdog kills it if object_height < 0.045 at iter ≥ 24.
**Eval:** `scripts/probe_grip_force.py` on the new A vs the a01 baseline (did force drop?); then seam-eval
the gentler A with a frozen reorienter. **Risk:** A keeps its grip (force floors — informative, the b34
finding on the A side) or collapses (watchdog catches, ~12 min lost).

**Round 2 (natural follow-up, NOT launched):** co-adapt — once a lower-force A lands, retrain gentle-B
against the *new* A's delivery (live-A reset uses whatever A checkpoint we pass). Run 1 trains against the
CURRENT frozen A, so its B is not yet matched to Run 2's A.

### HOW TO CHECK THE RUNS
`tail -f gentleB_policyB_gentleLowforce.trainer.log` / `lowforceA_policyA_lowforce.trainer.log`. Healthy =
`Metrics/lift_height/object_height` stays ~0.09–0.11 (held), `grip_force_excess` reward becomes *less*
negative over iters (force dropping), `tip_lost` settles low. A `*.COLLAPSED` sidecar file ⇒ the watchdog
aborted (grasp dropped). Run dirs land under `results/rl/<TAG>/tensorboard/`. **Eval at each policy's OWN
config (gotcha #13); judge on the full diagnostics, never cos/min-z alone (b31 lesson).**

### RESULTS + PIVOT TO MORPHOLOGY (2026-06-22 later) — the grip defect is STRUCTURAL
Both runs landed; then a per-finger probe + a spread-penalty run + a contact-hardening probe
together showed the excess force is a **geometry** problem, not a reward one → **morphology is the
active direction.** Full plan: **`docs/rl/morphology_optimization_plan.md`**.
- **gentleB** (relaxed verticality) WORKED as intended: vs b34_t20, ang-jerk **74→57** (smoother),
  fingertip force **6.8→5.3 N** (lower), at lower held-cos (0.77→0.64) — the smooth+low-force trade
  the user asked for. Ckpt `results/rl/bx_20260622-1738-policyB_gentleLowforce/tensorboard/model_405.pt`.
- **lowforceA** ran to completion (A held the grasp, no collapse); the force-probe eval vs a01 is
  still pending (`results/rl/bx_20260622-1740-policyA_lowforce`).
- **PER-FINGER FINDING (the key one).** Added per-finger instrumentation (`probe_grip_balance.py` +
  per-finger output in `rl_demo_handoff_continuous.py`; slot order verified [thumb,index,middle]).
  The grip is a **degenerate pinch**: **thumb idle ~1.6 N** while **index+middle clamp ~8 N each**
  (all three touch). B4 (the good reorienter) is a **balanced** tripod (7/10/10 N) — balance tracks
  reorient QUALITY, and our total force (~20 N) is already *below* B4's (~27 N). So "excessive" =
  **lopsided**, not over-clamped.
- **SPREAD PENALTY FAILED (built `grip_force_spread`, ran `policyB_spreadBalance`):** thumb 1.6→**1.8 N**
  (still idle), index 8.0→7.1. The policy **cannot recruit the thumb** — its placement can't oppose
  the other two against this object → **structural, not reward-fixable.** (New term + `--frozen-scene-xml`
  override committed; reusable.)
- **CONTACT-HARDENING FAILED informatively:** stiffening the contact even mildly (solimp 0.97/0.995→
  0.985/0.999) broke **frozen Policy A's grasp** (object never left the floor). The soft contact is
  **functionally load-bearing**; penetration is a symptom of the marginal grip. ⇒ a hard-contact run
  needs retraining A+B from scratch (the deferred sim-to-real pass), not a warmstart.
- **→ MORPHOLOGY.** The two structural defects map onto the **existing 9-param design space**
  (`src/morphohand/sampling/morphology.py`: per-finger x/y/length): (i) reposition the **thumb** for
  true opposition (recruit it → balanced, lower-peak grip); (ii) **seat** the object into the palm
  (lengths/palm → palm bears load → fingers relax to ~3 N). The "Shape Your Body"/VGDS value-gradient
  approach is the wrong tool here (needs a morphology-conditioned universal critic; our task is too
  brittle for the value to generalize across designs) — evaluate designs by **rollout**, not value
  gradient. **RECOMMENDED FIRST EXPERIMENT:** Stage 1(a) — reposition the thumb, retrain B, measure
  per-finger balance/force (~1 h). See the plan doc for the staged details.

---

## ⚡ FRESH SESSION — START HERE (updated 2026-06-10)

### 🎉 2026-06-10: THE LIVE-A RESET CLOSES THE SEAM. (first time — held + reoriented)
The handoff seam is **SOLVED in principle.** Built the live-A reset (the one untried mechanism,
scoped 06-09): frozen Policy A drives B's training env LIVE for steps 0..40 of every episode
(real physics, zero teleport), then B's PPO rollout begins at the organic seam; the A-driven
pre-onset steps are **masked** from the PPO update (advantages zeroed + renormalized; returns
kept). Code: `src/morphohand/rl/live_a_runner.py` (`LiveAOnPolicyRunner`),
`scripts/rl_train_cube.py --live-a-checkpoint/--live-a-onset`, `scripts/train_handoff_liveA_reset.sh`.

**Result (run `20260610-1046-policyB_liveAreset_fromB10`, model_270, 20M/3072, warmstart B10):**
continuous handoff@40, **post-handoff min-z 0.110 m (HELD ≫ 0.05)**, held-vertical **cos 0.751
(peak 0.816)**. B holds A's organic delivery at full height for the ENTIRE post-seam rollout and
reorients — the FIRST policy to do both. Every prior teleport approach dropped in 3-5 steps
(min-z 0.003-0.011). Training signature confirmed it: masked frac fell 0.95→0.20 (episodes
lengthened ~5×), align 0.45→58.9, tip_lost 51→8, episodes ran to time_out. Video
`docs/rl/videos/reorient/handoff_liveAreset_scale02.mp4`.

> **⚠️ CONFIG-PARITY GOTCHA (cost the first eval — now gotcha #13).** That run trained at
> `finger_residual_scale=0.2` (the rl_train_cube DEFAULT) while B10 (warmstart) AND the deploy
> demo use **0.5**. B relearned to hold at 0.2; the 0.5 eval applied its residuals **2.5× too
> large** → instant seam collapse — an **artifact, not a failure**. Re-eval at the matched scale
> 0.2 (`rl_demo_handoff_continuous.py --finger-residual-scale 0.2 --finger-close-easing linear
> --no-contact-gate`) gave the 0.110/0.75 hold. **The TRAINING env must match the DEPLOY env on
> scale/easing/contact-gate.** NB the *every prior B-finetune* (onset-inject, adapt-B, branch-B,
> co-adapt) ALSO trained at 0.2 and was eval'd at 0.5 — so the "injection capped at 0.011"
> verdict was drawn under the same mismatch and may be partly confounded; the live-A win is real
> regardless (held at its own matched scale). **Also: whole-rollout min-z is the WRONG metric**
> (dominated by the pre-lift floor phase z~0.012; the bar 0.05 is unreachable by it). Use the new
> **POST-HANDOFF min-z** + held-cos that `rl_demo_handoff_continuous.py` now prints.

### 2026-06-10 eve — REORIENTATION QUALITY is the open problem (hold is solved). User feedback: "still jittery + doesn't reorient well."
The live-A reset HOLDS the seam (min-z 0.105) but the reorientation is **mediocre (held-cos ~0.74,
≈42° off vertical) AND jittery** — both inherited from the **B10 warmstart** ("the violent
survivor"). Three things tried tonight, all confirming the diagnosis:
- **Continuation (40M, warmstart B10-live-A model_270, scale 0.2):** `20260610-1355-
  policyB_liveAreset_cont40M/model_541`. Reorientation **PLATEAUED** — held-cos **0.742** (peak
  0.817) ≈ identical to the original 0.751; align reward bounced 47-59 and ended ~55, no climb.
  **So the 0.74 ceiling is the B10 warmstart, NOT undertraining.** Hold still solid (min-z 0.105).
- **Warmstart B4 instead (the smooth 0.988 reorienter) → COLLAPSES.** `policyB_liveAreset_fromB4`
  (killed): B4 applies *reorienting* actions the instant it takes over → destabilizes A's
  delivery grip → object drops ~3 steps post-seam, BEFORE the reorient reward (gate@45) fires →
  zero reorient gradient → no escape (ep-len stuck 43, tip_lost 3072/iter, align 0). **The
  catch-22 that forced the hold-only B10 warmstart in the first place: a policy must SURVIVE the
  seam before it can be taught to reorient there, and B4's reorient-from-step-0 behavior can't.**
- **Deploy action low-pass (the documented non-reward jitter lever) → DROPS the object.**
  `--action-lowpass 0.5` on cont40M: post-handoff min-z **0.0027** (dropped), cos ~0. Confirms the
  smoothness gotcha at DEPLOY too: B10's high-freq corrective jerk IS the stabilization; filtering
  it breaks the hold. So jitter is NOT removable by a deploy filter on a B10-derived policy either.

**→ BOTH user complaints (jitter + poor reorient) have the SAME root (B10) and SAME fix: get B4's
SMOOTH, FULL-reorient quality to SURVIVE the seam.** The blocker is the B4 catch-22. **THE next
experiment (untried, needs supervised runner surgery): a TRAINING-TIME seam action-ramp-in** — in
`live_a_runner.py`, for the first ~8-12 steps after onset, step the env with `alpha*B + (1-alpha)*A`
ramping alpha 0→1 (the training analog of the demo's `--blend-steps`), and mask those blend steps
from PPO too. This eases B4 into A's grip gently instead of shocking it, so B4 survives long enough
to get reorient gradient while KEEPING its smooth full reorientation. Warmstart B4. (Alt levers:
handoff-DR curriculum that starts B's takeover state near B4's skip-lift comfort zone and anneals
toward A's real delivery; or a brief hold-grace where B4's residual scale ramps up post-seam.)
Eval everything at the policy's matched scale + POST-HANDOFF metric (gotcha #13).

**INFRA NOTE (2026-06-10 eve):** another user's `legged_gym h1_2_rma_magpie` run (4096 envs, ~8 GB)
is on cuda:0 — leave room / don't kill it. Also: a `pgrep -f "<pattern>"` watcher whose OWN cmdline
contains `<pattern>` deadlocks (it matches itself) — `liveA_cont_eval_trigger.sh` hung this way;
kill stuck waiters by PID, or grep a pattern that can't appear in the watcher's own command line.

**KNOWN VISUAL ARTIFACT — thumb penetrates the screwdriver (do NOT fix yet, 2026-06-10).** In
the `cont40M` rollout (`20260610-1355`) and many prior runs the **thumb visibly phases into the
screwdriver geometry** during the grip. Believed to be the deliberately *soft contact solver*
this task uses: `impratio=10`, `cone="elliptic"` (`env_cfg.py:1399-1405`) + soft scene geom
`solref="0.006 1" solimp="0.97 0.995 0.0005"` (`scene_screwdriver_medium_flat_short_proximal.xml:11`),
which trade some interpenetration for a stable non-explosive grip. It's somewhat bad
(cosmetics + sim-to-real fidelity) but **DO NOT retune impratio/solimp/solref until a better
policy lands** — changing them perturbs every policy's grip force and invalidates the A/B
lineage + all the seam comparisons. Revisit as a sim-to-real hardening pass after quality is solved.

**NEXT (in priority order):**
0. **(2026-06-10 — DONE/superseded)** ~~CANONICAL 0.5 RETRAIN~~ killed (B10 path is the quality
   ceiling; the scale-0.2 lineage from model_270 is the working one). See the eve section above:
   the live priority is now BREAKING THE B4 CATCH-22 (seam action-ramp-in), not more B10 training.
1. **CANONICAL 0.5 RETRAIN** (`policyB_liveAreset_fromB10_s05`, launched 2026-06-10,
   scale 0.5 / ease_out_quad / contact-gate ON = deploy parity; warmstart B10 at its native
   scale). When done: eval with `rl_demo_handoff_continuous.py --policy-b <ckpt>` (defaults now
   0.5) → post-handoff min-z + cos. This is the deployable, lineage-comparable version. **CHECK
   the trainer log / `BATCH_RESULTS.md` FIRST.**
2. **Push the reorientation FURTHER** (cos 0.75 < B4 standalone 0.988). UPDATE 2026-06-11 (see the
   06-11 section below for the full verdict): warmstart-B4 is DEAD (won't hold A's delivery, NaN-prone).
   The working lever is **commit-bonus + basin re-anneal on the B10-live-A lineage** → best is now **b29**
   (held-cos 0.784, up from 0.75). Iterate FROM b29; to break past ~0.8 needs a NEW mechanism (distill
   B4's reorientation into a seam-surviving policy), not more reward tuning.
3. **Re-examine the "injection capped" runs at matched scale** — cheap re-evals (no retrain) of
   Badapt/co-adapt at scale 0.2 may show they were better than recorded. Lower priority now that
   live-A works.

### 2026-06-12 — b32 registered (firmest+smoothest handoff yet); diagnostics now measure slip/jitter; the real gap is GRIP QUALITY not verticality
Iterated from b30 (the schedtrackB3 run, renamed b30_; its continuous-handoff eval DROPS —
held-cos 0.95 was **gamed** by a precarious near-vertical hold, post-handoff min-z 0.007). Two
follow-ups, both warmstart b30/model_405, live-A reset @ scale 0.2:

- **b31 cand** (`brace_d12f4_schedw8`, UNREGISTERED — rejected): schedule w20→8 + `brace_distance 12`
  / `brace_force 4`. Eval read held-cos 0.95 / min-z 0.078 (HELD) — looked like a win, but the
  **new diagnostics exposed it**: ang-jerk **449** (B4 ref ~27), 99cm horizontal path while netting
  0.2cm (violent vibration in place), 18 rad/s peak. User confirmed visually: "slips/jitters the
  entire time." Lesson: cos+min-z are blind to jitter; a frantically-shaking object still averages
  near-vertical.
- **b32** (`policyB_b30iter_gripSmooth_w4`, REGISTERED = new best handoff): `grip_force 6` +
  `action_rate −0.05` + `object_ang_acc −0.05` (gated step 45) + `brace_distance_scale 0.025`,
  schedule **w4**, `target_axis_alpha 4` (ease off vertical reach → firm-first). Eval: **min-z 0.1085**
  (firmest hold of ANY run, ties b29), **held-cos 0.891**, 4–5× smoother than b31 — ang-jerk
  **113**, lin-jerk **3.6**, wander 99→**23cm**. Video `docs/rl/videos/reorient/b32_gripSmooth_w4_cont.mp4`.

**New eval diagnostics** (`rl_demo_handoff_continuous.py`, no reward cost): per-20-step heartbeat
(`[diag] step z lat_drift cos`) + end summary (lateral drift, horizontal path/wander, z sink-rate,
lin/ang speed, **lin/ang jerk**, auto VERDICT flagging SLIP/SINKING/JITTER). The honest judge now —
use it, not cos/min-z alone.

**Two findings that redirect the effort:**
1. **Deploy-time `--action-lowpass` is DEAD.** lp 0.5/0.3 smooth perfectly (ang-jerk →16/10) but the
   object FALLS OFF (lateral 83–99cm, min-z 0.003). The jitter is **load-bearing corrective action** —
   the high-freq finger corrections ARE the stabilization of a marginal grip. Can't filter our way to B3.
2. **Penetration = grip-force readout; the gap is grip QUALITY not verticality.** b32 and B3
   (`b03_…_abl_signed`) use the **same frozen scene / contact model** — so the thumb-into-screwdriver
   penetration the user sees in b32 (and NOT in B3) is purely **higher normal force**: b32 over-clamps a
   tense fingertip grip (we even rewarded `grip_force` → harder press → more penetration + jitter),
   while B3 holds a gentle **seated** grip (low force → stays on surface → smooth). b32 is already
   near-vertical (cos 0.89 ≈ B3); what's missing is B3's relaxed seated grip. **`palm_brace_force`
   never fired in any run — the object is held at the fingertips ~8cm BELOW the palm, never seats.**

**NEXT (proposed, not launched):** stop pushing verticality; target grip quality — get the object to
**seat up into the palm** so the fingers can relax (gentle grip → no penetration → low corrective
jitter). Candidate diagnostic to add: object↔finger contact-force (penetration proxy) in the eval.
Open question the user raised: how to move b32→B3 grip without breaking the handoff (every
verticality-push attempt has broken the hold).

### 2026-06-12 (eve) — B3 path RULED OUT by render; force-regularize b32 (b33) LAUNCHED
**B3/B4 render dead-end (documented).** Rendered the gentle standalone reorienters B3
(`b03_…_abl_signed`) and B4 (lateral-only) on the CONTINUOUS handoff
(`docs/rl/videos/reorient/B3_signed_critic.mp4`, `B4_lateral_only.mp4`): both **drop the real
terminus** — B3/B4's relaxed seated grip survives only their own clean training reset, not A's
organic delivery. So "just deploy B3's gentle grip" is dead: the gentleness is inseparable from
the precarious start it was trained on. This closes the "move b32→B3 by swapping in B3" idea —
the only remaining grip-quality lever is to make b32 ITSELF gentler in place.

**b33 = force-regularize b32 (LAUNCHED 2026-06-12, `policyB_b32iter_forcereg_w6`).** The one
untested, high-information lever. b32 holds with an ~11 N fingertip **death-grip** (the source of
BOTH the thumb-into-screwdriver penetration AND the residual jitter — high-freq corrections of a
tense grip; verticality is already cos 0.89 ≈ B3). Question: is 11 N **necessary, or learned
laziness?** New code (commit pending): `mjlab_terms.grip_force_excess` + cfg
`grip_force_penalty_{weight,thresh,scale,reduce}` + CLI — a **quadratic penalty on fingertip force
ABOVE thresh** (`((force-thresh)/scale)**2`). The existing `grip_force` REWARD saturates at 3 N, so
the two only overlap above thresh: below it grip is still rewarded, above it extra force now COSTS.
Run `scripts/train_handoff_b33_forcereg.sh`: **single variable vs b32** — warmstart b32/model_405
(actor+critic), keep EVERY b32 term (grip_force +6, brace, smoothness −0.05, live-A reset @ scale
0.2, contact-gate OFF), ADD only the penalty (w −6, thresh 4 N, scale 4). Smoke confirmed the term
fires hard on warmstart (`grip_force_excess` −0.39 at iter 1). 20M ts / 2048 envs, ~40 min.
**Decision rule** — judge on FULL diagnostics (`rl_demo_handoff_continuous.py`: hold min-z + lin/ang
jerk + contact force), NOT cos/min-z (b31 lesson):
  - **still holds at 2–3 N** ⇒ jitter + penetration fall out together; B3-like gentleness on a
    policy that SURVIVES the handoff. WIN.
  - **drops** ⇒ the fingertip grip is fundamentally marginal; the real lever is **A's
    delivery/centering**, not the grip weights. Clean negative result, redirects effort.
(Prior turn's "seat up into the palm" proposal is deferred — `palm_brace_force` still never fired;
revisit only if b33 is inconclusive.)

**b33 RESULT (2026-06-12 eve) — REGISTERED. Verdict: the third case — neither clean WIN nor DROP.
It HOLDS, gets measurably gentler+smoother, but stalls well short of B3.** The 11 N was *partly*
learned laziness — it compressed for free — but the floor is well above the 2–3 N B3 target, and the
residual force still penetrates.

| metric | b32 | **b33 (force-reg)** | B3/B4 ref |
|---|---|---|---|
| post-handoff min-z | 0.1085 | **0.1114** (HELD) | — |
| held-cos | 0.891 | **0.845** | ~0.98 |
| ang-jerk | 113 | **49** | ~27 (B4) |
| lin-jerk | 3.6 | **1.17** | — |
| fingertip force (mean) | ~11 N | **7.5 N** (settles ~5–6 N) | ~3 N (B3) |
| wander | 23 cm | **8.9 cm** | — |

- **Grip was NOT fatally marginal** (rules out the "drops ⇒ A's delivery is the lever" branch): the
  penalty took force 11→7.5 N (steady-state ~5–6 N; trace decays 19 N at the catch → ~5 N by end),
  ang-jerk 113→49 (2.3×), lin-jerk 3.6→1.17 (3×), wander 23→8.9 cm (2.6×) — all for free, still holding.
- **But it did NOT reach the WIN branch either:** 7.5 N ≫ 3 N so it **still over-clamps and still
  penetrates**; still jitters (49 vs B4's 27); and it cost a little verticality (cos 0.891→0.845).
- **It's a partial, diminishing-returns move in the right direction, not a B3-level transformation.**
  Visually confirmed: gentler and smoother, but the thumb still phases in. Video
  `docs/rl/videos/reorient/b33_forcereg_w6_cont.mp4`.

**NEXT (open, not launched):** the obvious untried lever is to **push the penalty harder** (lower
thresh 4→2–3 N and/or raise weight) to find where the grip actually breaks — i.e. locate the true
force floor this morphology+contact-model can hold A's delivery at. If it holds gentler, win; if it
drops, that pins the floor and redirects to A's delivery/centering. (Deferred alts unchanged: seat
into palm; B3 distillation for the verticality gap.)

**b34 FORCE-FLOOR SWEEP RESULT (2026-06-13) — GRIP-PENALTY LEVER IS DEAD. The floor is flat at
~6.6–7.5 N; gentleness is a SEATING problem, not a force-reward problem.** Ran 3 runs continuing
from b33/model_405, weight −6 fixed, only the penalty threshold varied (`scripts/sweep_b34_thresh.sh`
→ `scripts/b34_eval_on_done.sh` → `b34_EVAL_RESULTS.txt`; all eval'd at matched parity scale 0.2 /
linear / contact-gate OFF). **Halving the penalty knee 4→2 N moved fingertip force <1 N:**

| thresh | fingertip force | held-cos | post-min-z | ang-jerk | palm force |
|---|---|---|---|---|---|
| 4.0 (b33) | 7.5 N | 0.845 | 0.111 | 49 | **0.0 N** |
| 3.0 (t30) | 7.5 N | 0.747 | 0.110 | 85 | **0.0 N** |
| 2.5 (t25) | 6.7 N | 0.771 | 0.112 | 69 | **0.0 N** |
| 2.0 (t20) | 6.6 N | 0.782 | 0.112 | 76 | **0.0 N** |

All HOLD (~0.11), all over-clamp (~7 N), all jitter (ang-jerk 69–85), held-cos flat ~0.75–0.78,
and **palm force is 0.0 N in EVERY run** — the object is held at a fingertip pinch ~8 cm below the
palm and NEVER seats. **Mechanistic verdict:** B3's gentleness IS the seated grip (palm bears load →
fingers relax to ~3 N); a fingertip-only hold of this object at this pose has a physical minimum
force ≈7 N (fingers alone resist gravity+torque by friction) — **you cannot reward your way below
it.** The lever was never grip force; it's the **contact configuration** (seated vs fingertip),
set upstream by A's fingertip delivery + a morphology that (geometry note: object sits 7–8 cm below
palm) likely **can't lift the object to the palm to seat it at all.** ⇒ **B3-gentle from a live-A
fingertip delivery is likely physically UNAVAILABLE, not under-tuned.** Caveats: (1) ~6.6 N may be
the honest optimum for the fingertip-catch REGIME (≠ B3's seated-reset regime — we've been measuring
against an unreachable benchmark); (2) visible penetration is partly the deliberately-soft contact
solver ([[feedback_thumb_penetration_soft_contact]]) — at 6.6 N a firmer solimp would penetrate less,
so "force" and "penetration" are partly decoupled. **STOP the grip-penalty line.** Remaining levers
are all bigger than reward tuning (in rough order of odds×effort): (A) accept the floor, register
t20/t25 as the gentlest live-A handoff, move on; (B) re-seat into palm — quick feasibility probe
first (can ANY scripted finger motion bring object→palm contact? `palm_brace_force` has fired in 0
runs, geometry note says no); (C) change A's DELIVERY to hand off a higher/seated pose (touches A,
branch-B territory, may hit the same morphology wall); (D) morphology change (longer/non-smooth
fingertips) — the true root if seating is geometrically impossible, but breaks the A/B lineage.

### 2026-06-13 (later) — ⚠️ CORRECTION: THE GRIP-QUALITY PREMISE WAS A PHANTOM. B3/B4 are NOT gentle; grip force was never the lever. The whole b32→b34 arc optimized a non-problem.
Before committing to a big swing (seat/change-A/morphology), VERIFIED the premise it rests on —
"B3 is gentle because it holds a ~3 N seated grip." It is FALSE. Direct measurement
(`scripts/probe_grip_force.py`, fingertip+palm contact force in each policy's own standalone
held+reorient rollout, steady-state after settle):

| policy | regime | held-cos | fingertip force | palm force |
|---|---|---|---|---|
| **B3** (b03 signed) | standalone | 0.978 | **7.04 N** | **0.00 N** |
| **B4** (b04 lateral) | standalone | 0.988 | **8.77 N** | **0.00 N** |
| **b34_t20** (ours)   | live-A seam | 0.782 | **6.6 N** | **0.00 N** |

**FOUR facts that demolish the grip-quality framing:**
1. **B3/B4 are NOT gentle — they grip 7–9 N, AS HARD OR HARDER than our handoff policy (6.6 N).**
   "Make the grip gentle like B3" was chasing a behavior that does not exist.
2. **Nobody seats. Palm force = 0.00 in ALL policies incl. the two "good" ones.** Seating was never
   the differentiator. (The 2026-06-13 "gentleness is a SEATING problem" verdict above is WRONG.)
3. **Force does NOT cause jitter.** B4 is the SMOOTHEST policy we have (obj-jerk ~26) at the HIGHEST
   force (8.77 N). Squeezing force could never have fixed smoothness.
4. **Penetration is universal.** B3/B4 hold at the same ~7 N on the same soft contact model ⇒ they
   penetrate the same. Any visual "B3 looks cleaner" is a contact ANGLE/pose effect, not lower force;
   it's the deliberately-soft `solimp` we froze ([[feedback_thumb_penetration_soft_contact]]),
   affecting every policy equally — orthogonal to the policy, deferred sim2real hardening.

**ROOT OF THE ERROR:** the "B3 ≈ 3 N" benchmark = `grip_force_max=3.0` (rl_train_cube.py:327 /
env_cfg.py:571), the grip-force REWARD's saturation cap, misread as B3's actual force. (Plausibly
compounded by reading B3's force on the continuous handoff where B3 DROPS → holds nothing → low force.)
So b32 (`grip_force` reward), b33 (force-penalty), b34 (force-floor sweep) all optimized toward a
phantom target. NB: the b34 sweep was still INFORMATIVE — it proved force is decoupled from hold
(6.6–8.8 N all hold) and from smoothness — it just wasn't the problem we thought.

**THE REAL, UNCHANGED GAP = reorientation QUALITY from the seam, NOT grip.** Two coherent deficits,
both about operating from A's delivered configuration under the B10 warmstart, NOT force:
- **Verticality:** handoff 0.78 vs B3/B4 standalone 0.98 (~38° off vertical).
- **Smoothness:** handoff ang-jerk ~76 vs B4 standalone smooth — jitter from the seam start-state.
These are exactly the B10-warmstart CEILING + B4-CATCH-22 documented pre-b32 (06-10/06-11 sections).
The lever is to get B4's smooth 0.988 reorientation to SURVIVE the seam, which reward tuning on the
B10 lineage plateaus at ~0.78–0.80. **NEXT (real mechanism, not reward tuning):**
1. **DISTILL B4 → a seam-surviving student** (teacher-student/DAgger): roll out the live-A handoff
   with a seam-surviving base (b29/b32 — they HOLD post-seam); at the post-seam held states query
   B4 (the smooth 0.988 teacher) for its reorienting action; train the student by BC(+RL finetune)
   to imitate B4's reorientation while keeping the hold. The only path with headroom past ~0.80.
2. Cheap first: RENDER B3/B4 standalone vs the handoff side-by-side to SEE whether penetration/jitter
   are actually worse in the handoff or universal (settles the last visual question for ~0 cost).
**STOP the grip-force line entirely.** `scripts/probe_grip_force.py` is the reusable grip-config probe.

### 2026-06-13 (later still) — 3-STAGE A→b32→B4 IS DEAD: the catch-22 is about GRIP BASIN, not object pose. + b32 reframed as a GOOD reorienter.
Tested the cheap "use the survivor to manufacture B4's start" idea: A lifts → b32 catches+stabilizes
→ hand off a 2nd time to B4 to finish the reorientation. Built it into `rl_demo_handoff_continuous.py`
(`--policy-c`, `--handoff-step-2`, `--policy-c-residual-scale`, `--blend-steps-2`; C's action is
rescaled ×2.5 so B4's native 0.5 residual matches the env's 0.2 — gotcha #13). RESULTS:
- **b32 baseline (hard switch, my harness):** HELD, cos **0.895** (peak 0.923), min-z 0.108 —
  reproduces b32's registered 0.891. (First 3-stage attempt's drop was an artifact of a `--blend-steps 8`
  at the A→b32 seam; b32 was trained with a HARD live-A switch, so the ramp put it OOD. Lesson: eval
  b32-lineage with blend 0.)
- **3-stage, HARD seam2:** b32 reorients to cos **0.909 @step60 / 0.903 @step80, rock-stable** (z 0.111,
  lat 0.8cm) — an almost-ideal near-vertical held start. B4 takes over @step90 → **dropped by step100**
  (z 0.007, lat 31cm).
- **3-stage, seam2 BLENDED (ease B4 in over 15 steps):** same — b32 holds 0.898, B4 eases in, **drops
  by step120** (z 0.012). The blend only slowed the fall.

**VERDICT — composition is DEAD, and the WHY is the keeper:** b32 handed B4 a clean, stable, cos-0.90
object pose — exactly the "clean start" the idea assumed B4 needs — and **B4 still dropped it within
10–30 steps, gently or hard.** So the B4 catch-22 is **NOT about the object pose; it's about the GRIP
configuration.** B4's competence is inseparable from B4's own finger placement / hand pose; you cannot
drop it into b32's grip any more than into A's raw delivery. **Policies that don't share a grip basin
cannot be composed.** COROLLARY: naive **action-distillation (DAgger with B4 as the action teacher)
will hit the SAME wall** — B4's actions are wrong for any other grip. Distillation, if attempted, must
transfer the OUTCOME/trajectory, not B4's actions (≈ what `target_axis` reward already does → likely
marginal over current reward tuning).

**THE REFRAME (drop the phantom and look again): b32 is already a GOOD handoff reorienter.** It HOLDS
the seam AND reaches cos **0.895** (~25° off vertical) with lat-drift <1cm. The grip-force chase
obscured this. The true remaining gaps vs B4 (0.988) are: (1) **JITTER** (b32 ang-jerk ~112 vs B4 ~26),
(2) a **modest verticality** gap (0.90 vs 0.99). Both trace to the **B10 warmstart basin** ("the
violent survivor"): every seam-survivor is warmstarted from B10 because B10 is the only thing that
survives the seam (catch-22), and B10 is inherently twitchy. Within the lineage there's a
force/jitter/verticality TRADE: b32 (11N / jerk 112 / cos 0.90) vs b34_t20 (6.6N / jerk 76 / cos 0.78)
— neither dominates. Reward jerk-penalty backfires (jitter is load-bearing for a marginal grip),
deploy low-pass drops it, force-penalty trades verticality. **The jitter is the B10 basin + the
fundamentally marginal fingertip grip of a 10cm rod on 3 smooth tips.**

**HONEST STATE OF THE HANDOFF (2026-06-13):** the seam is SOLVED (holds) and b32 reorients to cos 0.90.
Reaching B4's 0.988+smoothness from a live-A handoff is blocked by the grip-basin catch-22 (composition
dead, naive distillation predicted dead) and the B10 jitter ceiling (reward tuning plateaued). **PATHS:**
1. **ACCEPT b32 as the deliverable handoff policy** (holds + cos 0.90; jitter is high-freq finger
   correction, object stays put). Defensible, honest stopping point. **[recommended]**
2. **From-scratch (no-B10) live-A training** — the ONLY untried way to escape B10's jittery basin: train
   a fresh policy with the live-A reset + strong lateral/smoothing regularizer, no B10 warmstart. HIGH
   RISK (B10 warmstart exists precisely because holding the seam from scratch is hard) but it's the one
   lever that could find a SMOOTH seam-surviving basin. The real research bet if we keep pushing.
3. **Accept the jitter as morphology-limited** (marginal smooth-fingertip grip on a long rod) and, if
   sim2real fidelity matters, revisit `solimp` hardening + possibly fingertip geometry — a separate
   morphology pass that breaks the A/B lineage.

### 2026-06-11 — SEAM ACTION-RAMP-IN built to break the B4 catch-22; layered diagnosis + 3 runs in flight
Built the documented next experiment: a **training-time seam action-ramp-in** in `live_a_runner.py`
(`--live-a-blend-steps N`): for N steps after onset, step the env with `alpha*B+(1-alpha)*A`
(alpha 0->1), masked from PPO like the pre-onset steps (B trains only on fully-B steps >=
onset+N). Also exposed `--term-tip-lost-steps` and auto `LIFT_TERM_START=onset+BLEND` +
`NUM_ENVS`/`EXTRA_ARGS` in `scripts/train_handoff_liveA_reset.sh`. **Layered smoke diagnosis
(1M ts each), each peeling one blocker:**
1. ramp=10/20 alone → object stays UP during ramp (obj_h 0.10) but `tip_lost` kills @~48
   (terminations engage at onset=40, grace 3) BEFORE keep_from → **mask frac stuck 1.0 = zero
   trainable steps.** Ramp length is NOT the constraint.
2. + delay terminations to B-takeover (`LIFT_TERM_START=keep_from`) → frac 1.0→0.92, ep-len→62
   ✓ trainable steps! BUT B4 drops at takeover (obj_h 0.10→0.055, tip_lost ALL envs, align 0).
3. + survival window (`--term-tip-lost-steps 20`) → ep-len→69 but object falls to FLOOR (0.012),
   align ~0. **B4, un-finetuned, drops A's flat delivery the instant it has authority — it has
   NO hold fallback (exactly why B10/hold-only was the warmstart that worked).**
**Conclusion: the ramp-in is NECESSARY but NOT SUFFICIENT.** It keeps the object up during the
ramp and now yields trainable steps, but whether PPO can climb from "drop→floor" to "hold+reorient"
in 20M is the open gamble (every prior B4-warmstart live-A attempt collapsed).

**3 RUNS LAUNCHED 2026-06-11 (20M ts, 2048 envs each, parallel, per-process Warp cache):**
- **Run A** `policyB_liveAreset_fromB4_survwin` — B4 + ramp(20) + tip-lost-steps 20, reorient@65,
  lift-term@60. Watchdog OFF (starts on floor by design). Tests exp-1 directly. Log
  `liveA_b4_survwin.trainer.log`.
- **Run B** `policyB_liveAreset_fromB4_easein` — B4 + GENTLER ease-in: ramp(30) + tip-lost-steps 30,
  reorient gated LATE (@95), lift-term@70 → big hold window before reorient pressure. Watchdog OFF.
  Log `liveA_b4_easein.trainer.log`.
- **Run C** `policyB_liveAreset_B10qual_commitbonus` — the INVERSE strategy: continue B10-live-A
  (warmstart **b24** model_270) at its NATIVE config (scale 0.2 / linear / contact-gate OFF —
  gotcha #13!) + a NON-terminating commit+speed bonus (`--success-bonus-weight 30 --speed-bonus-weight
  15`, thresh 0.85; NO success-termination → avoids the documented threshold-gaming) to push the
  0.74 held-cos ceiling toward B4's 0.99. Hard onset (BLEND=0, like b24). Watchdog ON. Log
  `liveA_b10_commitbonus.trainer.log`.
**WATCH:** Run A/B success = mask frac keeps falling + align reward CLIMBS + obj_h recovers off the
floor (the gamble paying off). Run C success = held-cos > 0.74 at eval. Eval each with
`rl_demo_handoff_continuous.py` (post-handoff min-z + cos, gotcha #13 — match each policy's scale).

**INTERIM RESULTS (2026-06-11, ~iter 30):**
- **Runs A & B (B4 warmstart) are FAILING — the B4 hypothesis looks dead.** Both NaN-crashed early
  (transient Warp NaN, check_nan fatal — not systemic) AND were not learning: Run B (gentler ease-in)
  reached ep-len 95+ but the object sat **on the FLOOR (obj_h 0.012)** with align reward ~0.04 — i.e.
  B4 drops A's flat delivery and the long episodes are just "sitting on the floor," not holding. Run A's
  frac stayed ~0.99 (≈no trainable steps). **Confirms (again): B4 has no hold prior, can't learn one
  from a dropping start.** A relaunched once (`_survwin_r2`) and **NaN'd AGAIN at iter 13 → B4 path ABANDONED** (3 crashes +
floor-drop non-learning; B won't hold A's flat delivery). B NOT relaunched.
- **Run C (B10-live-A + non-terminating commit bonus) is WORKING — the promising lever.** frac
  1.0→0.417→**0.242**, episodes run to **time_out (ep-len 209)**, object **HELD (obj_h 0.110)**, align
  reward **57**, and the **commit bonus fires non-terminating** (`alignment_success_bonus` 0.30,
  `alignment_speed_bonus` 0.037 — proves the bonus computes WITHOUT enabling the terminating success,
  so no threshold-gaming). Whether it lifts held-cos > 0.74 at eval is TBD (run completing).
- **Run C2** (`_B10qual_commit60`) launched in B's freed slot: same b24 warmstart, STRONGER commit
  (success 60 / speed 30) + sharper basin re-anneal (alpha 1→6 over 150 it) — hedge on the working lever.
- **FINAL VERDICT (all 3 ran to completion; eval = `rl_demo_handoff_continuous.py` @ scale 0.2, matched):**

  | policy | held-cos | peak | post-handoff min-z |
  |---|---|---|---|
  | b24 baseline (B10-live-A) | 0.751 | 0.816 | 0.110 |
  | b27 cont40M | 0.742 | 0.817 | 0.105 |
  | **b28** Run C (commit-bonus 30) | 0.759 | **0.891** | 0.110 |
  | **b29** Run C2 (commit 60 + α re-anneal 1→6) | **0.784** | 0.866 | 0.108 |

  The non-terminating commit bonus + sharper near-vertical basin **moved the quality ceiling 0.75→0.78**
  (real but modest, ~5° closer to vertical) and the HOLD stays solid (~0.11). It did **NOT** break through
  to B4's standalone 0.988. **Conclusion: the B10-warmstart basin is a stubborn quality ceiling that
  reward-shaping only nudges; the B4 path is dead (won't hold A's flat delivery).** Best handoff policy now
  = **b29** (`results/rl/b29_20260611-1152-policyB_liveAreset_B10qual_commit60/tensorboard/model_405.pt`),
  video `docs/rl/videos/reorient/handoff_B10qual_commit60.mp4`.
- **NEXT to break past ~0.8 (needs a NEW mechanism, not reward tuning):** distill B4's full reorientation
  into a seam-surviving policy — e.g. behavior-clone B4's actions onto the held post-seam states b29 visits,
  or a teacher-student where B10 holds and B4 advises; or push the b29 recipe harder (even sharper basin /
  goal-tilt curriculum / longer training) for incremental gains. The commit+re-anneal direction (b29) is the
  current best lever; iterate from b29, not b24.

### 2026-06-11 (later) — ROOT CAUSE of the quality ceiling = the reorienter's BRITTLENESS, not orientation
User asked WHY full reorientation from live-A is hard, hypothesizing the object starts more horizontal
than B3/B4 trained on. **Measured it — the hypothesis is wrong on orientation but right that the start
condition differs:** at the handoff A delivers the object essentially identical to B3/B4's skip-lift
training spawn on every axis — held-cos (tilt) 0.00 vs −0.009 (both FLAT), object z ~0.11 both, palm_pz
0.067 both (the skip-lift spawn raises the palm by lift_delta_z, env_cfg ~L669-675). The ONLY deltas:
**object xy off-center ~0.8 cm** (A delivers (0.005,−0.007) vs B's centered (0,0)) + **finger grip differs
~2–8°/joint** + **live contact velocities** vs a settled drop.
**Sensitivity probe (NEW eval path — `rl_eval_reorient_metrics.py name=run:ckpt:handoff_state_bank=…,skip_lift_phase=True`
spawns the reorienter from A's REAL delivery bank under its OWN matched obs schedule):**

  | policy | held-cos | jerk | (clean baseline) |
  |---|---|---|---|
  | B3 from A's delivery | **0.261** | 354 | (clean 0.980, jerk 53) |
  | B4 from A's delivery | **0.514** | 35 | (clean 0.989, jerk 27) |

  **That sub-cm off-center + few-° grip CRATERS B3 0.980→0.261 (7× jerk), B4 0.989→0.514 — with matched
  obs, flat orientation, and the object not even dropping (min-z 0.076/0.095).** So the ceiling is NOT
  orientation, NOT obs-schedule alone (matched here), NOT a drop — it's that the reorienter memorized a
  RAZOR-THIN basin around its one pristine centered spawn. (B4 is notably MORE robust than B3, opposite the
  visual preference.) This also explains the distillation discontinuity the user flagged: raw-B3 imitation
  onto b29 is DOUBLY OOD (B3 is garbage on A's-grip states AND b29's normal-lift obs ≠ B3's skip-lift obs),
  so distillation needs a HARDENED B3 teacher first.

**2 HARDENING RUNS LAUNCHED (20M, skip-lift, spawn from A's FULL delivery bank w/ real velocities, warmstart
the reorienter, its full reorient recipe — `scripts/train_handoff_adaptB_to_A.sh`):**
- `policyB_adaptB3_fromAbank` (warmstart B3, user's preferred) — log `adaptB3_fromAbank.trainer.log`.
- `policyB_adaptB4_fromAbank_full` (warmstart B4, more-robust base = hedge) — log `adaptB4_fromAbank.trainer.log`.
**WATCH:** held-cos recovering toward ~0.9 from the bank = brittleness is trainable away. Then (a) eval the
hardened policy via continuous live-A deploy (does fixing the grip alone close the seam, or does the
skip-lift→normal-lift obs gap still bite?), and (b) THEN it's a valid teacher for the on-policy
distillation onto b29 (the user's idea, now correctly sequenced). NOTE: the bank reset overrides
`reset_robot_joints`, so synthetic grip-jitter DR can't stack with the bank (would need clean-spawn + jitter).

### History below: the teleport/injection era (2026-06-09 eve) — SUPERSEDED by the live-A win above.

**TWO RESULTS 2026-06-09 eve:**

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

> **WAVE 1 COMPLETE (2026-06-09 ~20:49). Key finding: the VELOCITY injection is harmful, NOT
> last_action (my earlier hypothesis was BACKWARDS).** Results (continuous min-z, handoff@40):
>
> | run | warmstart | inject | min-z | reading |
> |---|---|---|---|---|
> | coadapt_B_toAtol20 | Badapt | complete, Atol20 deliv | 0.0076 | stuck (flat 0.02), culled — Badapt can't catch *migrated* delivery |
> | B_complete_fromBadapt | Badapt | complete, frozenA deliv | 0.0022 | held in train (0.073) but evals WORST — converged complete-state confirms it hurts |
> | branchB_w6_tol20 | frozenA | (A-side, w6) | 0.0042 | collapsed A's grasp (z@handoff 0.012) — weight too high |
> | branchB_w4_tol15 | frozenA | (A-side, w4) | 0.0073 | held; ×Badapt 0.0073 < w2's 0.0114 — migration non-monotonic |
> | **inject_velOnly** | B10 | vel ON, lastact OFF | **0.0023** | velocity ON → BAD |
> | **inject_lastactOnly** | B10 | vel OFF, lastact ON | **0.0073** | velocity OFF → good (≈ static) |
>
> **The 2×2 (velocity / last_action):** velocity-OFF holds ~0.007–0.008 (static 0.0081, lastactOnly
> 0.0073); velocity-ON drops ~0.002 (velOnly 0.0023, complete 0.0027). **Injecting A's real finger
> qvel at the teleport gives the fingers momentum the position setpoint doesn't expect → grip-
> perturbing transient.** So *static (zero-vel) injection is the best B-side variant*; last_action is
> ~neutral. (This is why the "Markov-complete" 0.0027 < static 0.0081 — the velocity made it worse.)
> **Nothing trained beats the free co-adapt pairing 0.0114.**
>
> **WAVE 2 COMPLETE (2026-06-09 ~23:00) — VERDICT: the injection paradigm is CAPPED at ~0.011.**
> The proper co-adapt B (B10 ws + STATIC inject of Atol20's migrated delivery, lenient watchdog so it
> had room to learn): `coadapt_B10_Atol20_static` **0.0113**, `coadapt_Badapt_Atol20_static` **0.0118**
> — both essentially TIED with the free pairing 0.0114, i.e. **training B on the migrated delivery did
> NOT improve over the untrained cross-pairing.** Run-1's object_height stayed **FLAT at 0.031 the whole
> run** (never climbed → B does NOT learn to *hold* the teleported delivery; it sags ~3 cm and the eval
> reads a drop). **CONCLUSION (definitive): every teleport-based B-side approach — skip-lift bank /
> normal-lift static / complete / co-adapt, any warmstart, any A — converges to "B sags off the
> injected state, min-z 0.002–0.012." The TELEPORT ITSELF is the ceiling.** Co-adaptation is the best
> *direction* (0.0114) but cannot be pushed past the cap by more injection training.
>
> **➡️ THE PATH IS NOW THE LIVE-A RESET (no more injection variants — that family is exhausted).**
> Run frozen Policy A LIVE for steps 0..40 of every B training episode (real physics / real contacts /
> real `last_action`, zero teleport), then B's PPO rollout begins at the organic seam. **Implementation
> — RECOMMENDED = advantage-masking (I scoped it tonight; the obvious "wrapper" has a trap). The
> rollout loop is `actions = alg.act(obs); env.step(actions); alg.process_env_step(...)`
> (`.venv/.../rsl_rl/runners/on_policy_runner.py:85`). Tempting fix — wrap the policy to *emit A's
> action* pre-onset — is WRONG: `alg.act` also stores the action's **log-prob under B**, so swapping in
> A's action leaves a stale log-prob → corrupt PPO ratio (you'd have to recompute log π_B(a_A),
> PPO-internal surgery). **Cleaner: (1) ENV applies A's action while `episode_length_buf < onset`** —
> hold frozen A's actor on the env, in a pre-physics hook compute A's action from `env.obs_buf[:, :65]`
> and write it to the finger actuators, ignoring B's passed action; **(2) MASK pre-onset steps from the
> PPO update** — zero their advantages+returns in the rollout storage before `alg.update()` (A drove
> those steps, so B mustn't be trained on them; their stored action/log-prob then don't matter). B
> trains ONLY on post-onset steps from A's organically-arrived state. SMOKE-TEST FIRST (1M ts, ~3 min):
> A lifts in the reset window (object_height → ~0.10 by step 40), B trains without NaN, post-onset
> reorient reward fires, pre-onset advantages ≈ 0. **DO THIS CHEAP CHECK BEFORE BUILDING** (no training,
> ~5 min): eval B10/B4 with its episode STARTING from a live-A delivery vs its skip-lift spawn — if it
> holds materially longer from live-A, that's direct proof the seam is the train/deploy gap and masking
> is the only work left. (Deferred tonight: PPO-internal surgery must be built + validated SUPERVISED;
> a silent log-prob/masking bug would waste an unattended run — not worth the risk while idle.)

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
(`results/rl/b14_20260608-1738-policyB_adaptToA_bankA_s40/tensorboard/model_270.pt`, object held +0.012
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
(warmstart B10). Ran clean to iter 270 (`results/rl/b15_20260609-1113-policyB_onsetInject_bankA_s40/
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
- Frozen Policy A (lift+deliver, GOOD — keep frozen): `results/rl/a01_20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt`
- B4 = best standalone reorienter (0.988): `results/rl/b04_20260603-1746-policyB_p2_lateral_only/tensorboard/model_541.pt`
- B10 = first to survive delivery + reorient but violent: `results/rl/b10_20260604-1642-policyB_holdonlyws_repro/tensorboard/model_541.pt`
- A's real-delivery state bank (grip+pose, 2048 states, z=0.111): `results/rl/handoff_state_bank_A_s40.npz`
  (OLD, static/zero-vel) — **use the COMPLETE bank now: `results/rl/handoff_state_bank_A_s40_full.npz`**
  (real obj_vel + robot_qvel + a_last, 2048 states).
- adapt-B (skip-lift bank) result that just closed off this branch: `results/rl/b14_20260608-1738-policyB_adaptToA_bankA_s40/tensorboard/model_270.pt` (min-z 0.0028)
- **Atol20** = migrated A (frozen-A → B10's grip, branchB tol20): `results/rl/a07_20260605-1609-policyA_unfreezeA_v2_w2_tol20/tensorboard/model_270.pt`
- **Badapt** = adapted B (B10 → frozen-A delivery, static onset): `results/rl/b15_20260609-1113-policyB_onsetInject_bankA_s40/tensorboard/model_270.pt`
- **Atol20's delivery bank** (complete, for co-adapting B to the migrated A): `results/rl/handoff_state_bank_Atol20_s40_full.npz`
- ⚠️ `results/rl/badapt_initiation_s48.npz` is GARBAGE (Badapt drops by step 48 → recorded a floor grip; do not use as a branchB target).

## TL;DR — what's true now
- **Best reorientation policy = `p2_lateral`** (held-vertical cos **0.988**, peak **0.999**,
  obj_jerk **25.8** = HALF the prior best, no drop):
  `results/rl/b04_20260603-1746-policyB_p2_lateral_only/tensorboard/model_541.pt`.
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
13. **TRAIN/DEPLOY CONFIG PARITY — a B finetuned at one `finger_residual_scale` and eval'd at
    another is OOD (silently).** `rl_train_cube.py` defaults `finger_residual_scale=0.2`,
    `finger_close_easing=linear`, `contact_gate_stability_rewards=off`; but B10/B4 AND
    `rl_demo_handoff_continuous.py` (deploy) use **0.5 / ease_out_quad / on**. A B trained at 0.2
    relearns its grip for 0.2; eval at 0.5 applies its residuals **2.5× too large** → instant
    grip blowup that LOOKS like a seam failure but is an artifact (cost the first live-A eval,
    2026-06-10). ALWAYS pass `--finger-residual-scale 0.5 --finger-close-easing ease_out_quad
    --contact-gate-stability-rewards` when finetuning a B for the continuous-handoff deploy, OR
    eval at the policy's own training config (the demo now takes `--finger-residual-scale` etc.).
    Corollary: **continuous-handoff `min-z` must be measured POST-HANDOFF** — whole-rollout min-z
    is dominated by the pre-lift floor phase (z~0.012), so the 0.05 bar is unreachable by it;
    the demo now prints the honest post-handoff min-z + held-cos.

## Reproduce the in-progress launches
The exact P1/P2/P3 commands are in `scripts/queue_reorient_handoff_dr.sh` / the run dirs'
`config.yaml`. All three: `--num-envs 3072 --total-timesteps 40000000 --init-actor-checkpoint
results/rl/b03_20260602-1636-policyB_abl_signed/tensorboard/model_405.pt` + the per-path knobs above.
