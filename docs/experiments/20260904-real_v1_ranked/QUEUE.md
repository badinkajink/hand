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
