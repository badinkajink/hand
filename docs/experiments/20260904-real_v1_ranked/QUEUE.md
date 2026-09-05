# Queued work — closed-loop grasp on the chain, across both hand sets (2026-09-04)

Written because usage ran out mid-session. Everything below is set up but NOT started.

## The two hand sets, and why both

**Set A — the eight the user bench-tested for reorientation** (`data.json` rows in
`docs/experiments/20260901-real_v1-transfer-firstpass/`, plans in
`docs/experiments/20260829-real_v1_deploy/deploy/`). These have HARDWARE data, n=7-10 each:

| design | plan | clip |
|---|---|---|
| sv1_w6689 | sv1_w6689_b060 | 0.60 |
| sv1_w2360 | sv1_w2360_b075 | 0.75 |
| sv1_u1364 | sv1_u1364_b080 | 0.80 |
| g12 | g12_b095 | 0.95 |
| sv1_u0060 | sv1_u0060_b75 | 0.75 |
| sv1_u0308 | sv1_u0308_b050 | 0.50 |
| rv05_manual | rv05_manual_b85 | 0.85 |
| sv1_w0099 | sv1_w0099_b100 | 1.00 |

**Set B — the eight promoted from the Sobol-8192 screen** (`20260831-real_v1-sobol8192/deploy/
promotion.json`), already run: `docs/experiments/20260904-real_v1_ranked/ranked_slip.json`,
videos in `videos/`. Only `sv1_w6689` is in both. **Do not abandon set A** — it is the set with
bench measurements, so it is the only one where a sim claim can be checked against hardware.

## The task, in order

1. **Closed-loop grasp control through the whole chain.** The regulator already exists:
   `real_v1_deploy_envelope._force_step` / `_load_step`, and `execute(force_target=,
   force_phase=)` already accepts `"all"` vs `"hold"`. In the deployed maneuver it runs in the
   HOLD phase only, because regulating through the turn "holds the shaft better and turns it
   worse" (the screen's own comment). The user's point is that this trade is worth taking: kill
   the slip-to-vertical, get a grasp that does not fail, and then a harder reorient->gait pose
   trajectory becomes explorable. So: run `force_phase="all"` and a load target through the
   CHAIN (`probe_real_v1_chain.chain`), which currently has no regulator at all — port
   `_load_step` into it, gated on the same servo-load units.
2. **Get it working AT ALL on a selection of hands, with videos every time.** Both sets. Expect
   iteration; the chain's grasp phase is fitted on rv05_manual and set A/B hands need their own
   `open_ik` (use `_grip_from_fit` + the anchor-by-joint-name path already added to
   `chain(anchor_ctrl=)`).
3. **Then the relay -> gait pose trajectory**, which is the thing a reliable grasp unlocks.
4. **Then widen the population** — the 227 confirmed hands, not 8. ~3 s a rollout.

## Gotchas already paid for

- **Replay through the SCREEN's own code** (`make_plan` + `execute`), not a reconstruction:
  rebuilding the maneuver out of `probe_real_v1_carry` on a table with a palm lift put every
  ranked hand at 78-89 deg of residual tilt. Validate against `confirm_b*.json` `nom_cos`,
  keyed on design AND `budget_rad`.
- **Gate every tilt on contact** (`held_turn`): a dropped shaft standing in a countersink reads
  vertical, and `rv04_mid_sp40` scores better than the hand that actually turned it.
- Set A's plans are for the BENCH scene (tool on a 100 mm platform, palm fixed). The chain is a
  table pickup with an arm. These are different maneuvers; say which one a number came from.

## New instrumentation added this session

`turn_tilt_deg` / `settle_deg` at the last commanded turn step, in both
`probe_real_v1_carry.carry` and `real_v1_deploy_envelope.execute`.
`probe_real_v1_chain.chain(anchor_ctrl=...)` takes a fitted grasp by joint name.
`scripts/real_v1_ranked_slip_study.py`, `scripts/real_v1_ranked_videos.py`.


---

## 2026-09-04 17:0x — first pass on the closed loop. It does not reach the shaft.

**Done:** `probe_real_v1_chain.chain(load_target=, load_gain=, reg_band=, reg_every=)` regulates
per-finger servo load through EVERY phase (ported from `real_v1_deploy_envelope._load_step`; the
trim is removed before each phase's `before(k)` and re-added after, so a phase that commands only
some fingers — the relay walks one pad at a time — neither loses it nor double-applies it).
`scripts/real_v1_chain_hands.py` prepares both sets: base scene -> flat pads -> `_grip_from_fit`
baked into `open_ik` -> countersink -> UR5e arm scene, then runs the chain at each hand's own cell.

**Set A through the chain, loop off: 0/8 hold the tool at the last commanded turn step, 0/8
complete.** Turn tilts 68-90 deg. (Set A's own maneuver is the fixed-palm bench carry, where these
hands score cos 0.75-0.95 — the chain is new to them, so this is not a failure of the ranking.)

**The loop does not fix it, and the reason is not the loop.** On `rv05_manual_b85`, load target
0/60/120/250/400 units: pad force at the turn stays 0.06-0.11 N and the trim SATURATES at 25.78
deg (= the 0.45 rad band) at every non-zero target. The regulator is pushing the fingers inward at
full authority and still not making contact.

**Ablation says the cell is not the cause either.** Same hand, same scene, four cells (its own
k0.05/-60/b0.85, the chain's k0.25/-90/b0.50, and two crosses): turn 39-69 deg, pad force
0.11-0.33 N, 1-2 pads in all four. What is common is the GRASP: 2 pads and 1.9 N at the lift,
decaying to 1 pad and 0.08 N through the turn.

**So the grasp geometry is wrong for this maneuver before the loop ever runs.** The fit
(`_grip_from_fit`, 10 mm squeeze) is solved for the tool where the design scene puts it and is
correct on the bench, where the tool is held at working height on a platform. In the chain the
tool is picked up off a table and then rotated, and the pads end up further from the shaft than
0.45 rad of joint travel can close. This is the self-extinguishing grip the carry probe already
documents ("the grip force IS the commanded-minus-actual error; as the shaft creeps down through
the pads the error shrinks and the force decays to zero", measured 16.1 N -> 0 over 0.8 s).

**NEXT LEVER — a geometric re-anchor, not a load trim.** The bench's own answer to exactly this
decay is `hold_squeeze`: re-solve each pad's IK target to sit INSIDE the rotated shaft's surface
and ramp to it. That is a new set-point, not a trim on the old one, which is why the load
regulator cannot substitute for it. The chain has no equivalent — `turn_squeeze` (added earlier)
holds a closure THROUGH the turn and is known bad. Port `execute`'s `squeeze_delta` /
`plan["squeeze_delta"]` path into the chain as a re-grip at the top of the turn, THEN re-run this
sweep. Widening `reg_band` past 0.45 rad is the cheap thing to try alongside it, but the
saturation says the gap is geometric.

Artifacts: `docs/experiments/20260904-real_v1_chain_hands/20260904-rv05_manual_b85_chain_load250.mp4`
and `.png` (filmstrip). Data for the loop-off pass was a smoke run, not kept.

---

## Session 2 (2026-09-04, evening) — what got done and where it stalled

**The regulator is ported and it works.** `probe_real_v1_chain.chain(load_target=, load_gain=,
reg_band=, reg_every=)` runs `real_v1_deploy_envelope._load_step` through EVERY phase, hooked
into `_run` — the one step loop every phase goes through. The trim is stripped before a phase
writes its set-point and re-added after, so a phase that commands only some fingers (the relay
walks one pad at a time) neither loses the trim nor double-applies it. Measured on the chain:
trim reaches its 0.45 rad authority cap (25.8 deg) on `rv05_manual_b85` and lift force goes
0.9 -> 1.92 N; on `sv1_u0308_b050` turn-seam force goes 0.33 -> 3.7 N. Videos in
`docs/experiments/20260904-real_v1_chain_bothsets/videos/`.

**But the chain still does not work on set A: 0/8 complete.** Two failure modes:
1. `sv1_w6689_b060` and `sv1_w2360_b075` never close on the tool at all (0 pads from the first
   seam; the two shallowest fitted depths, 52.5 and 58.5 mm).
2. The other six lift (100-113 mm rise, 2-3 pads) but **the turn does essentially nothing**:
   86.8 -> 87.6 deg on u0060, 68.8 -> 68.3 on rv05 with the loop on. Tilt at the turn seam sits
   at 68-88 deg for every hand, loop or no loop.

**So the blocker is the TURN, not only the grip.** The loop raises the grip and the turn still
does not happen, which means the next thing to fix is upstream of both.

**Dead ends, measured, do not repeat:**
- **Gain does not matter.** `_load_step` clips each step to +-0.0006 rad, so gain above ~0.003
  changes nothing (0.020 and 0.080 give bit-identical results). The loop is rate-limited; over a
  750-step window it only reaches 3-4 deg of trim, which is why a short probe reads it as broken.
  Over a full chain (~10k steps) it reaches the authority cap.
- **Commanded squeeze is not the fix.** Sweeping 10/14/18/22/26 mm: force is non-monotone and
  mostly falls (g12 0.38 -> 0.30 -> 0.26 -> DROP), every hand drops the tool by 22 mm, and only
  `rv05_manual_b85` gains (0.35 -> 3.97 -> 6.25 N at 18-22 mm). This is the flat-pad shortfall
  the screen already documented; sweeping it per hand is not a general answer.
- **The fitted grasp makes only 0.33-1.56 N** on a tool LYING ON A TABLE, against 12.1 N for the
  same hand on the bench scene where the tool STANDS. The grasp geometry, not the hand, is the
  difference.

**Next, in order:**
1. Find why the turn stalls at ~68 deg. The chain's turn is an IK sweep of the finger anchor
   clipped to `budget`; these hands' plans carry their own `axis_k`/`angle_deg`/`budget` and
   those are being passed. Check the turn's commanded joint delta against `budget` per hand
   (the plan's own `delta` field is the reference) — a clip that binds everywhere would explain
   a uniform stall. `scripts/probe_action_budget.py` is the existing tool for exactly this.
2. Fix the table grasp before judging any hand: the 0.33 N grip is the real defect, and the
   bench scene shows the same hands can make 12 N. Consider fitting the grasp with the tool in
   the pose the chain actually presents it in.
3. Only then re-run `scripts/real_v1_chain_hands.py --sets AB --loads 0,250` (the runner is
   written, cached per hand, and renders a video per hand per load).

`scripts/real_v1_chain_hands.py` is the runner for both sets; its cache lives in
`assets/mjcf/experimental/20260904-chain_bothsets/` (own directory — `20260904-chain_hands` is
written by another study and a shared path returned half-populated records).

---

## Session 3 — the turn: what moves it, and the three things that do not

**THE COMMANDED ANGLE IS THE LEVER, AND THE DEPLOYED ANGLES ARE TOO BIG FOR A MID-AIR TURN.**
The chain's turn is an IK sweep of the finger anchor about a raised pivot, and the IK residual
scales with the commanded angle: 14-18 mm at -30 deg, 20-28 at -50, 27-43 at -90. At the plans'
own angles (-60 to -90) the targets are 27-43 mm out of reach, so the command is not a rotation
and the achieved turn is under a degree. Halving the ask roughly triples what is achieved:

| hand | plan angle -> achieved | best cell found | achieved |
|---|---|---|---|
| sv1_u0308_b050 | -80 deg -> +15.7 | **-50 deg, k 0.25** | **+36.1** (87.3 -> 51.1, held) |
| rv05_manual_b85 | -60 deg -> +0.5 | -50 deg, k 0.05 | +25.4 (84.0 -> 58.7, held) |
| g12_b095 | -70 deg -> -0.1 | -40 deg, k 0.05 | +27.4 (87.5 -> 60.0, held) |

`make_plan` never checks this residual (it calls `ik_finger` and discards the return), so the
deployed plans carry it too. They work on the bench because the POST supports the tool through
the turn -- consistent with the program's own finding that 46-69% of the alignment gain happens
on the floor. Mid-air there is nothing to take up the shortfall.

**THE GRIP IS 41 N AT CLOSURE AND 0.33 N AFTER THE LIFT.** Measured directly: `F@close` 41.0 /
40.5 / 43.2 N on u0308 / g12 / rv05, `F@hold` 0.33 / 0.38 / 0.33 N after a 100 mm lift and
settle. Pad station is -3 to -7 mm, so this is NOT an above-the-equator wedge; it is the
position-servo decay the screen already documented (grip force IS commanded-minus-actual, the
shaft creeps down through the pads, the error shrinks). This is the root cause of everything
downstream, and it is upstream of the turn.

**THREE THINGS THAT DO NOT FIX IT — all measured, do not repeat:**
1. **The servo-load loop** (`--load-target`, `_load_step`). It runs, it reaches its 0.45 rad
   authority cap over a full chain, and it roughly doubles the lift force (0.9 -> 1.92 N on
   rv05). It does not change the turn: 68.8 -> 68.3 deg.
2. **The contact-force loop** (`--force-target`, `_force_step`, added this session). Targets of
   0 / 1.5 / 3 / 6 N give results identical to three significant figures. Both loops steer with
   `_squeeze_dirs`, which moves the pad radially toward the pinch axis; when the shaft has slid
   AXIALLY through the pads, radial squeeze does not restore the lost position error.
3. **The re-squeeze** (`--turn-squeeze`). Worse on every hand at every setting: u0308 +36.1 ->
   +23.9 -> +19.0 -> +15.5 -> +6.8 over 0/2/4/6/10 mm, and the tool drops out of a 2-pad hold
   into a 1-pad one. This now replicates in both regimes -- on rv05_manual_stored with a healthy
   12.9 N grip and here with a 0.33 N one -- so it is settled: closure through the turn is not a
   control anywhere.

**So the user's hypothesis is answered and it is negative:** closed-loop grasp control does not
rescue the chain. The loop is real and it works, but it regulates the wrong degree of freedom
for this failure. Videos: `20260904-sv1_u0308_b050_turn50_k0.25.mp4` (best, +36.1) against
`..._turn80_k0.15.mp4` (the deployed cell, +15.7).

**NEXT — the axial slip, not the radial squeeze.** The pads need to be put back where they were
along the shaft, not pressed harder into it. Two candidates, in order:
1. Re-solve the grasp against the shaft's CURRENT pose partway through the lift and command the
   corrected joint targets (a mid-lift re-grasp, one extra set-point, still open-loop).
2. Regulate on pad STATION (`pad_s_mm`, already in every seam) rather than on force or load:
   drive each pad back to its station at closure. `_squeeze_dirs` needs an axial sibling.
Only after the grip survives the lift is it worth re-running the angle sweep or judging any
hand, because every turn number above is measured through a grip that has already collapsed.

---

## Session 4 — why nothing worked: you cannot preload a light free object with position servos

**THE 0.33 N GRIP IS NOT A DECAYED GRIP. IT IS STATICS.** The transient at closure is 80-99 N;
by step 900 it is whatever the object's weight demands, and it scales with mass on the same
command and the same hand:

| object mass | 24.5 g | 98 g | 393 g | 1571 g |
|---|---|---|---|---|
| sv1_u0308_b050 | 0.33 N | 1.32 N | 9.84 N | 83.6 N |
| rv05_manual_b85 | 0.34 N | 1.54 N | 8.87 N | 28.0 N |

Position servos converge until forces balance. In mid-air nothing opposes the squeeze except the
object's own weight and inertia, so a 24 g tool is held at a third of a newton however deep the
command goes. On the bench the POST supplies the reaction, which is why the same hands make 12 N
there and 0.33 N here.

**This explains every negative in sessions 2-3 at once.** Commanded squeeze, re-squeeze, the
servo-load loop, the contact-force loop and the station-referenced re-grasp were all trying to
move a quantity that statics fixes. The re-grasp is the clearest case: it put the pads back where
they sat at closure and force went 0.33 -> 0.32, 0.59 -> 0.30, 0.90 -> 0.39, 2.19 -> 0.34 N,
because the pads had not slid -- there was never a grip to restore.

**So the mid-air chain is grip-limited at ~0.33 N and the turn gets what friction at 0.33 N
allows** -- 20-38 deg of the ~87 needed, best `sv1_u0308_b050` 87.3 -> 51.1 at -50 deg / k 0.25.
No control knob reaches this. What does:
1. **Opposition.** Pads that squeeze against each other THROUGH the object, so the interference
   has nowhere to go. That is grasp geometry and morphology, not a controller -- and it is a
   co-design variable the program has a topology for (the opposed pair,
   `project_perp_topology`). Screen for it: the achievable mid-air grip at fixed command, which
   is one 900-step rollout per hand with no schedule at all.
2. **Keep a reaction surface in contact.** Every maneuver that works in this program -- the
   bench carry, the ground-supported gait, the countersink seat -- has one. A mid-air reorient
   may simply be the wrong primitive for a 24 g tool and a 3-finger position-controlled hand.

**ON RL.** It cannot beat statics either: a policy trained on this chain would be optimising
against a 0.33 N ceiling and would rediscover the same ~30 deg. It IS the right tool for the one
thing no scalar schedule can express -- trading grip against roll DURING the turn, where every
knob swept here is monotone in the wrong direction on one of the two objectives -- but only once
the grasp can generate force at all. Fix opposition first, then RL on the modulation.

---

## Session 5 — give up the roll: stand the tool up ON the table, then hand it over

**THE REORIENTATION IS SOLVED AND THE HANDOVER IS NOT.** The fingers are never asked to turn the
tool (`angle_deg` 0) and the tool never leaves a surface: it is picked up, put back down lying,
and stood up by pivoting about its own foot with the table carrying its weight. Ten of sixteen
hands stand it 4/4 at 0.00-3.3 deg off vertical. Then the palm has to re-index to the gait's
pose and take the ring, and that is where 34 of 36 successful stands are lost.

16 hands x 4 seeds, `scripts/real_v1_chain_hands.py --stand table`
(`docs/experiments/20260904-real_v1_held/chain_hands.json`, one video per hand in `videos/`):

| | count |
|---|---|
| hold the lift | 41/64 (5 hands drop the tool at the grasp: w6689_b060, w6689_b050, u5860_b070, w7583_b130, and w2360_b075 at 2/4) |
| stand it | 36/64, ten hands at 4/4 |
| upright tilt, on the stands | 0.00-3.3 deg on nine hands, 17.4 on `sv1_u0060_b75` |
| complete the chain | **2/64** with the single-leg re-index, **8/64** with the three-leg one below |

The best run, `sv1_u1364_b080` seed 3: stood at 0.00 deg, ring taken on three pads at 32.2 N,
6/6 gait cycles at -40.6 deg/cycle (80% of the 1.684-gear ceiling), tool at 0.09 deg when the
chain ends, 23.3 deg of roll in the grasp over the whole run, `ok` True.
`docs/experiments/20260904-real_v1_held/20260904-table_stand_u1364.mp4`.

**WHY THE TABLE AND NOT THE ARM.** A 90 deg mid-air rigid reorientation drops the tool on every
variant tried: fingers turning it, arm turning it, tip-up, tip-down, UR5e, floating palm. That
is session 4's statics again -- 0.4 N on a 24 g tool -- and it does not care who commands the
rotation. It is also why "give up the roll" is not by itself the fix: the roll was not what was
failing, the hold was, and only a reaction surface changes that.

**IT IS NOT A HELD REORIENT.** On the plane the tool rotates 70.6 deg inside the grasp while the
table pivots it (`roll_max_deg`, new, measured in the palm's frame). The hand guides; the floor
does the work; eight rigid-transfer corrections against the measured pose land it at 0.09 deg.
On the flipped heading below the same move costs only 10.8 deg of roll, so a nearly-held version
of it exists.

**FOUR DEFECTS IN THE CHAIN PROBE, ALL FOUND BY TAKING THE FINGER TURN OUT** (commit 2e0bd26d):

1. **The tool axis was folded, so a screwdriver standing on its HANDLE read as a perfect
   0.00 deg stand.** Every tilt in the probe was arccos|cos|, right for a plain cylinder and
   wrong for a tool whose tip is one specific end. With the rotation left to the arm, `_stage`
   picks the end to raise from the sign of R[2,2], which on a horizontal tool is numerical
   noise -- and the run then reported `tilt_deg` 0.00, z 0.0500, `stood_ok` True for a tool
   upside down on the annulus beside the socket. Fixed: signed axis whenever `tip_len > 0`.
2. **Rigid transfers were interpolated in joint space.** For the small corrections the published
   chain makes that is the same path; for a 90 deg reorientation the arm swings the hand through
   an arc the command never named, with an 0.008 mm IK residual saying it went where it was
   told. Fixed: interpolate the pose, re-solve each control step.
3. **The gait's ring was solved on fixed world bearings** (thumb pi, index pi/3), which is the
   tripod a palm looking DOWN makes. Stand the tool up by pivoting and the palm is left on edge
   beside it; the canonical ring is then 14.0 mm out of reach, so the pads either hover at
   0.02 N or arrive with that residual as interference at 57-91 N and shove the tool 17-26 deg
   over. `--ring-az pads` builds the same table on the azimuths the pads already occupy, and
   `ring_z` is scanned for the height whose ring the fingers can close on.
4. **The gait's palm height was the fit's grip depth.** The fit reports the height that pinches
   a tool LYING DOWN (64.5 mm on g12); the gait needs the height from which three fingers close
   on a ring around a STANDING one (45 mm). At 64.5 the ring solves 13.0 mm out of reach and the
   close knocks the tool over. Now scanned per hand: residual 0.43 mm.

**THE PRESS IS A CLIFF, NOT A KNOB.** Same hand, same seed, `--press` 2 / 6 / 10 mm: thrown
(-1243 deg/cycle, dropped) / 6 cycles at -37.7 and 15.6 deg / 6 cycles at -37.9 and 12.5 deg.
The gait study's own window was -6..+6 mm for a shaft already standing; a shaft the hand has
just stood up needs the top of that range.

**TWO HANDOVER VARIANTS THAT DO NOT HELP -- do not repeat.**
* **Closing one finger at a time after the palm move** (`--reindex regrip`, added this session):
  0/4 against `full`'s 1/4 on g12. Each finger's Cartesian legs pass close to a tool that is now
  free-standing, and three passes are three chances to knock it over.
* **Reducing the ring's commanded interference.** `--squeeze` 0.5 / 1.0 / 2.0 mm over five hands
  x four seeds: g12 2/4, 1/4, 1/4 and every other hand flat at its own value. The 41-45 N at the
  close is not what the knob controls.

**THE COUNTERSINK: the tool's heading on the bench is a spec.** Standing a screwdriver TIP DOWN
cannot happen on a flat table, so the seated version has to arrive at the socket. `--stand-order
pivot` (descend lying, pivot up about the foot, walk that foot to the socket) stalls at 34 deg on
the heading the scenes ship with -- the same tip-down rotation the mid-air stage could not carry.
Lay the tool down the other way round (`prepare(h, yaw_deg=180)`, one quaternion on the object
body) and the same move stands it TIP DOWN at 8.79 deg with 10.8 deg of roll. The tool then sits
10 mm proud of its seat and the handover topples it, so it is not yet a chain, but the stand-up
is no longer the obstacle.

**THE HANDOVER LOSES IT DURING THE PALM MOVE, WITH THE HAND OPEN.** Of the 36 runs that stood
the tool, the first seam past `pressed` where the tilt exceeds 14 deg is `reindexed` in 29,
`gait_grip` in 2, `gaited` in 3, and never in 2. Mean tilt across the stands: 1.15 deg at
`released`, **41.4 at `reindexed`**, 71.0 at `gait_grip`. The close was never the problem.

The cause is the shape of the move. The re-index is a ~90 deg wrist rotation from on-edge-beside
-the-tool to looking-down-at-it, and one Cartesian leg rotates and translates at once, which
walks the open fingers through the space the standing tool occupies. `--clear` (default 80 mm)
splits it into three legs: straight up until the fingertips clear the tool's top, across and
around at that height, then straight down onto the gait pose. Same endpoint, nothing passes
through the tool. On `g12_b095` over four seeds that is **1/4 -> 3/4 complete chains**, and the
surviving runs end 5.4 / 9.1 / 10.8 deg off vertical against 11.7 before (commit e2a47be0).

**THE 16-HAND RE-RUN WITH THE THREE-LEG RE-INDEX: 8/64, and the loss at `reindexed` falls from
29 to 20 of 36.** `sv1_u1364_b080` 3/4, `sv1_u7952_b050` and `_b065` 2/4 each, `sv1_w0099_b100`
1/4. `g12_b095` runs all six cycles on all four seeds and never drops the tool (end tilt
8.4-16.3 deg) but fails `grip_ok` on three of them, and its -67 to -86 deg/cycle is past the
50.5 deg gear ceiling, which is the shaft being spun rather than gaited. At `repose_steps` 900
instead of 800 the same hand is 3/4, so the leg length matters and has not been swept.

**NEXT, in order.**
1. **Sweep the re-index leg length and clearance** (`--clear`, and `repose_steps`, which sets how
   long each of the three legs takes). g12 flips 0/4 to 3/4 between 800 and 900 steps, which is
   too sensitive to leave at a default. 20 of 36 stands are still lost at `reindexed`.
2. **`gain_mean_deg` past the gear ceiling is not a gait.** g12 at -67 to -86 deg/cycle and
   u7952 at -1164 and +582 on its failing seeds are the tool being spun or thrown, not turned by
   the pads. The gait probe already flags this for the brake (`brake_pumped`); the chain needs
   the same guard, or those cells will be read as fast gaits.
3. **Seat the tool the last 10 mm** on the flipped heading, then re-run the handover there. A
   tool in a 45 deg cone is not free-standing, and the program has already measured that
   continuous contact buys everything in a seat and nothing on a plane.
3. **The five hands that drop the tool at the grasp** (roll 147-167 deg, `free_frac` 1.00) fail
   upstream of any of this. Their fitted grasp does not hold a tool lying on a table at all;
   check the fit before reading anything else about them.

---

## Session 6 — the grasp was wrong, and it was wrong because the squeeze is a bench number

**THE USER SAW THE TOOL POPPING OUT OF THE HAND AND WAS RIGHT.** Measured at closure with the
palm stationary and the tool lying on the table, at the deployed plans' own `squeeze_mm` of 10.0:

| squeeze | 2 mm | 4 mm | 6 mm | 10 mm |
|---|---|---|---|---|
| `g12_b095` rise at closure | +3.8 mm | +8.9 | +12.8 | **+18.6** |
| `g12_b095` pad force | 10.4 N | 5.4 | 1.8 | **0.5** |

Nine of sixteen hands rise +14 to +20 mm -- more than the shaft's own 12.5 mm radius -- ride up
onto `thumb_pip_frame` and `index_pip_frame`, and lose pad force as the squeeze rises, because
the shaft has escaped over the top of the pads. Three hands go the other way and crush it into
the floor at 28-67 N.

**WHY.** `tip_targets` puts the pad CENTRES at `r_obj + r_pad + gap - squeeze` = 12.5 + 10.55 +
1 - 10 = **14.05 mm** from the shaft axis when contact needs 23.05. The command is 9 mm inside
the surface. The plans were fitted on `...__medium__bench100py-35__flat15w21T.xml`, where the
shaft floats at 100 mm on a post that takes that reaction; a shaft lying on a table has nothing
behind it. It is the wedge-sign failure `tip_targets` documents for positive elevation, arriving
through excess squeeze instead of elevation.

**THE FITTER'S HOLD PROBE CANNOT CATCH IT, AND IT WAS SWITCHED OFF ANYWAY.** `held` is measured
against the object's height BEFORE the close, so the pop counts as retained lift:
`held ~= lift_probe + dz` is an identity. Every grasp that "passed" the 20 mm gate at 65-75 mm on
a 50 mm probe passed by the amount it threw the tool, and the tell -- `held > lift_probe` -- was
never checked. On top of that `probe_real_v1_carry._grip_from_fit` calls `fp.fit(..., hold_min=
-1.0)`, so six of sixteen hands ran the whole chain on grasps the fitter itself scores at
-0.1 mm. Those six are exactly the six that dropped the tool at the grasp in every session-5 run.

**THE FIX AND WHAT IT BUYS.** `SQUEEZE_MM = 2.0` is the chain's own value now, `--squeeze-mm`
sweeps it, and `prepare()` records `close_dz_mm` / `close_pads` / `close_nonpad` / `close_pad_N`.
At 2 mm every one of the 16 hands closes on three pads with the tool where it started (rise -0.4
to +5.0 mm, 8.6-38.7 N) and the pip-frame contacts are gone on ten of them.

| 16 hands x 4 seeds | 10 mm (the plans') | 2 mm |
|---|---|---|
| hold the lift | 41/64 | **64/64** |
| stand the tool | 36/64 | 35/64 |
| complete the chain | 8/64 | **23/64** |

`rv05_manual_b85` goes 0/4 to 4/4, and so do `sv1_u1364_b080`, `sv1_u7952_b050` and `_b065`;
`g12_b095` 0/4 to 3/4. Of the 35 stands, 23 never exceed 14 deg after `pressed`, against 8.

**THE FAILURE MOVED TO THE SET-DOWN.** 24 of the 29 non-stands lose every pad at `set_down` --
the phase that puts the tool back down lying before the pivot. A 2 mm grasp has less margin there
than a 10 mm one, so the squeeze is a trade with an optimum between the two ends and not a knob
to minimise. `--squeeze-mm 2,3,4,6` is running.

**NEXT, in order.**
1. **Read the squeeze sweep** (`docs/experiments/20260904-real_v1_held/squeeze_sweep/`). Report
   `close_dz_mm` beside the chain rate: the right value is the largest squeeze whose closure
   still leaves the tool where it was.
2. **Re-run everything in session 5 on the corrected grasp.** The re-index leg sweep, the
   three-leg clearance, the countersink heading -- all of them were measured through a grasp that
   had thrown the tool 15-20 mm before the maneuver started, so none of those numbers stand.
3. **Gate the fit.** `_grip_from_fit` should pass the fitter's real `hold_min` and reject
   `held > lift_probe + 2 mm` as a pop rather than a pass. `sv1_w6689_*` fails at every squeeze
   and is a genuinely bad grasp; it should be excluded by the gate, not by hand.

---

## Session 7 (2026-09-05) -- the reorientation is not the loss, and the handover is one seam

**The squeeze sweep, 16 hands x 4 seeds x {2, 3, 4, 6} mm.** Splitting the chain's AND-gate into
its four stages puts the 90 deg turn on one side of the ledger and everything else on the other.

| squeeze (mm) | lost in carry | lost standing | lost at handover | lost gaiting | complete | reorient (deg) |
|---|---|---|---|---|---|---|
| 2 | 24 | 5 | 11 | 1 | **23** | 89.1 +- 0.9 |
| 3 | 16 | 2 | 25 | 0 | 21 | 88.3 +- 4.5 |
| 4 | 13 | 4 | 22 | 4 | 21 | 89.3 +- 0.8 |
| 6 | 13 | 2 | 32 | 8 | 9 | 89.4 +- 0.9 |

**When a hand stands the tool it stands it at 89.1-89.4 deg of the 90, sd 0.9.** The
reorientation does not vary across hands, seeds or squeeze. What varies is the carry that
delivers the tool and the handover that gives it back. The two pull opposite ways -- 2 mm loses
24 in the carry and 11 at the handover, 6 mm loses 13 and 32 -- so the squeeze is a per-hand
trade. Per-hand best totals **28/64** against 23 for a single population value, on n=4 per cell.

**Three new fields** (`probe_real_v1_chain.py`): `reorient_deg` (90 minus the tilt at `upright`),
`drop_stage` (the first seam the tool is loose on the floor at -- referenced to the shaft RADIUS,
because a tool standing on its end sits at the half-LENGTH and is not dropped), `reorient_ok`.

**The handover is one seam, and the films name it.** Of the 58 handover failures at 2-4 mm, 53
are already past 14 deg at `reindexed`, having stood at 1.47 deg through `released`; mean tilt
goes 1.5 -> 75.4 deg and z 50.4 -> 16.8 mm across that one move. The completed runs read 0.64 deg
at the same seam. `sv1_w2360_b075` stands 4/4 and completes 0/4, and its filmstrip shows the tool
flung clear of the table disc, not toppled in place.

**`--gait-scan` A/B: NEGATIVE, and it inverts the fix.** The gait palm height is scanned over
30-72 mm against the CLOSED ring's IK residual, but the palm descends on the OPEN ring, so
scanning the open ring looked like the obvious correction. Over 128 runs per arm (16 hands x
{2, 4} mm x 4 seeds):

| scan | carry | stands | handover | chain |
|---|---|---|---|---|
| grip (shipped) | 91 | 82 | **49** | **44**/128 |
| open | 91 | 82 | 29 | 21/128 |
| both | 91 | 82 | 46 | 35/128 |

Minimising the open-ring residual is the wrong objective: it puts the open pads ON a ring 6 mm
from the tool instead of clear of it. The descent needs the fingers WIDER than the tool, which is
`release_mm` (6.0), not the palm height. `--releases 6,11,16,22` x {2, 4} mm is running.

**Films.** `scripts/real_v1_chain_films.py --sweep <chain_hands.json> --out <dir>` re-runs each
hand's furthest-reaching cell with the renderer on: one mp4 and one seam filmstrip per hand plus
a labelled grid. Output `docs/experiments/20260904-real_v1_held/20260905-films/`.

**Two gotchas that cost a sweep each.**
- `tpad=stop=-1` on every xstack input makes an ENDLESS stream; the grid encode ran until killed
  and left a 23 MB file with no moov atom. Cut the output with `-t <longest input>`.
- A sweep arm whose name is already a key in the base cell dict raises "got multiple values for
  keyword argument" INSIDE the per-cell try, so all 512 cells fail silently and the summary
  prints zeros that look like a finding. `_cell` now pops every kw key from the base cell.

**NEXT, in order.**
1. **Read the release sweep** (`docs/experiments/20260904-real_v1_held/release_sweep/`). It is the
   only untried lever on the dominant loss. If a wider release helps, the ceiling is the open
   ring's own IK residual (`ring_ik_open_mm`, 13-16 mm today) -- past some width the fingers
   cannot reach the ring at all.
2. **Apply the close-probe gate in `prepare()`.** `close_nonpad >= 2` selects exactly the hands
   that eject the tool: `sv1_w6689_b050` and `_b060` at every squeeze (0/16 each) and
   `sv1_u5855_b050` at 2 mm (0/4). Removing them on evidence takes the denominator from 64 to 56.
3. **Per-hand squeeze, then eight seeds.** 28/64 vs 23/64 on n=4 does not separate a 5-run
   difference from noise. Fix the squeeze per hand from the close probe, then re-run at n=8 on the
   hands that survive gate 2.
4. **Guard `gain_mean_deg` against the 50.5 deg/cycle gear ceiling.** g12 reads -67 to -86 and
   u5860 +188; those are the shaft being spun or thrown and they currently read as fast gaits.
5. **The countersink heading is still unmeasured on the corrected grasp.** `--stand-order pivot`
   stalls at 34 deg on the shipped heading and reaches 8.79 deg at `yaw_deg=180`, both measured
   through the 10 mm grasp.

---

## Session 7, RETRACTION (2026-09-05) -- the table stand is not a reorientation

**Everything in the session 7 entry above that quotes a reorientation angle is wrong.** The
metric did not test whether the hand was holding the tool.

`stood_ok` is `ground_contacts >= 1 and tilt_deg < 14 and |z - rest_z| < 0.010`. It never asked
about the grasp, and `reorient_deg` (90 minus the tilt at `upright`) inherited that. Counting pad
contacts at the `upright` seam over the 177 runs that passed the gate:

| pads on the tool at `upright` | runs |
|---|---|
| 0 | **116** |
| 1 | 57 |
| 2 | 4 |

**Pad force at that seam has a median of 0.000 N over all 177 stands and exceeds the tool's own
0.24 N weight in 2 of them.** With the load test applied the maneuver holds the tool through the
turn in **2 of 256 runs**. The table is carrying the tool; the hand sets it down, lets go, and it
settles vertical. The "89.1 +- 0.9 deg of the 90 deg turn" was the tilt of a released tool.

This is the same defect already recorded twice in this program -- peak_cos without final_z scores
a dropped shaft as perfect, and rv04 reads 8 deg with ZERO pads because a dropped shaft in a
countersink is vertical. It was rebuilt from scratch here and published before anyone checked
contacts.

**Three further gaps, all visible in the run records and none of them measured:**
- `tip_len_mm = 0.0` on every run. The object is a plain cylinder, the tool axis is folded, and
  standing on EITHER end scores `tilt_deg = 0`. Tip-down was never tested.
- `place_xy = None` on every run. No countersink. 79 of the 177 stands end below the table top.
- `reindex = "full"` sets the tool down and releases it. There is no relay from the reorient
  grasp to the gait grasp. `reindex = "relay"` exists and was not run.

**The gate is now in the code.** `chain()` returns `upright_pads`, `upright_force_N` and
`held_reorient` (2 pads and >1 N at `upright`), and `reorient_ok` requires it. No tilt from this
family should be quoted without it.

**Both handover sweeps are also negative**, and they are the only part of session 7 that stands:
scanning the gait palm height against the open ring gives 21 chains against 44 (128 runs per
arm), and opening the pads wider before the descent changes the handover count by exactly zero at
6, 11, 16 and 22 mm -- `ring_ik_open_mm` grows one-for-one with the target (14.3 -> 26.9 mm), so
the fingers are already as open as they go and the pad radius is not a lever.

**PATH ABORTED.** Do not resume the table-stand chain. What a real attempt needs, and none of it
was in these runs: a tool with a tip, a socket to seat it in, a grasp that carries load through
the turn, and the relay instead of a release.
