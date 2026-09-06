# Skills 3 and 4: runbook and state, 2026-09-06

Written before compaction. This is the runbook, not a result page — results go to
`docs/experiments/<YYYYMMDD-topic>/<YYYYMMDD-name>.html` via a builder (see CLAUDE.md).

## The skill decomposition (user, 2026-09-06)

1. grasp and lift · 2. reorient · 3. move the palm pose so the held reorientation aligns with
vertical at the screwdriver target · 4. relay from the held-reorient grasp to the gait grasp ·
5. gait. Screen candidates per skill; the chain is too long to screen with. RL, if it returns,
returns as a per-skill residual, not over the whole horizon.

## Acceptance test per skill — NOT a cos threshold

Fields are `object_track.*` / `manual_score.*` on hardware, the chain row's own fields in sim.

| skill | accept if | score on |
|---|---|---|
| 1 | `held_lift`, >= 2 pads, force >= 0.240 N (the tool's weight) | pad force |
| 2 | retained (`manual_score.success` on hardware; >= 2 pads at >= 0.240 N through the terminal hold window in sim) AND `slip_mm` < 20 mm | `deg_turned` |
| 3 | tool still held at the target pose; tip inside the 6 mm capture radius | residual angle closed |
| 4 | continuous contact through the transition, no pad sweeping through the tool | `grip_ok` |
| 5 | — | turns/cycle against the 50.5 deg/cycle gear ceiling |

**Never gate on `cos_hold`.** 0 of 8 hands reach mean 0.95; retention ranks against alignment at
Spearman -0.12 (paper filter) / +0.12 (per-trial median over retained), both noise at n=8.
**Never read `held_reorient` in table mode** — the maneuver stands the tool and releases it by
construction, so that field is the released-tool measurement. Read `held_turn`.

## Which hands to use

Hands that lift and hold the turn 16/16 in the chain, ready for the skill-3/4 test:

    g12_b095 (D4)   rv05_manual_b85 (D7)   sv1_u0060_b75 (D5)
    sv1_u0308_b050 (D6)   sv1_u1364_b080 (D3)   sv1_w2360_b075 (D2)

Hardware retention, for prioritising: D2 10/10, D7 10/10, D4 9/10 (D1 6/7 but blocked below),
D3 4/10, D8 4/10, D5 3/10, D6 2/10. **User prefers hands validated high-retention in real** —
that is D2, D7, D4 first, then the rest.

Blocked: `sv1_w6689_b060` (D1) — best on hardware, 0/16 carried. `sv1_w0099_b100` (D8) — 6/16.

## Residual angle skills 3+4 must absorb

90 deg minus the median retained bench turn, per hand:

    D5 24.5   D1 35.3   D4 37.6   D2 38.4   D8 45.0   D3 49.3   D6 53.8   D7 56.5

## The measurement state

- The chain funnel, set A, 128 rollouts: `held_lift` 122, `carry_ok` 102, `held_turn` 102,
  `stood_ok` 91, **`grip_ok` 43**, `ok` 41. Skills 3+4 are the 91 -> 43 loss.
- The chain runs table mode with `angle_deg` 0 — the fingers never turn the tool. **The screened
  turn has not been run inside the chain on any hand.**
- The relay handover exists (`probe_real_v1_chain.py --reindex relay --relay-gait`): 6/6, 0.4%
  of the task uncontacted vs 49% for release-and-regrip, at half the turn rate. It only
  re-seats; it has never been asked to also advance the angle.

## Next measurements, in order

1. **Run the screened turn inside the chain** on the six ready hands. Until that exists, skills
   3 and 4 are being tested downstream of a maneuver that is not the screened one.
2. **Skill 3 standalone**: from a held, partially-reoriented tool, close the residual angle by
   palm re-pose. No script exists; this is the gap. Accept on held + residual closed.
3. **Rotating relay**: make each relay cycle re-seat AND advance the angle, so the tool passes
   through the residual in increments that each stay inside the retention interval. This is the
   one structure the bench data says survives.
4. **Unblock D1**: 2x2 on `sv1_w6689` alone, straddle {32, 40} x depth {52.5, 58.5}, n >= 6,
   scored on pad count and pad force at the `turned` seam.

## Standing traps (all cost a session already)

- `rv03_narrowy_sp40` and most `results/phase1/real_v1/*` runs have **no deploy plan and were
  never built**. Check `docs/experiments/*/deploy/*_plan.json` before choosing a design.
- The bench record is `docs/experiments/20260902-cb1-log-archive/logs/` (251 runs, 11 designs),
  not the `*bench*` folders.
- `--linear-anchor` (`max_ik_residual_mm` 0.0) is required for a carry that keeps its grip; the
  per-step IK carry re-centres on the achieved pad pose and grip decays to 0 in 0.8 s.
- Plans' `squeeze_mm` 10.0 is a bench number: on a table it raises the tool 14.5-19.6 mm at
  closure. The chain's 2.0 mm override is correct.
