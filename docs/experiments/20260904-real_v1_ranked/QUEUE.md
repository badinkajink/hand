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
