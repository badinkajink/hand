# Sobol-8192 — the population doubles, and one new hand takes the top of the bench

Ran overnight 2026-08-30 → 2026-08-31 (`scripts/real_v1_sobol8192.sh`, log
`logs/20260831-sobol8192.log`), then confirmed and promoted the same morning.

`sobol_designs` is prefix-stable, so this is a strict **superset** of the 4,096-hand population:
the first half keeps its tags, its vectors and its generated scenes, and only the new half cost
anything. 8,198 sampled, **6,629 inside the measured rails**, of which 3,311 were genuinely new.

## The funnel

| stage | what it does | result |
|---|---|---|
| A / A2 | sample, then keep only what the rails reach | 6,629 of 8,198 |
| B | grasp screen, 16 shards | — |
| C | one table per grasp cell | `retention/` |
| D | retention screen **swept across five clips** (0.50–1.30), servo gate inside | 57,220 cells over 3,298 designs |
| E | one (cell, clip) operating point per hand | **535 pass at some clip** |
| 3 | 5 nominal + 4 half-error draws at each hand's own clip | **227 confirmed** |
| 4 | 20 full-error draws for the survivors | 227 cells |

Against the 4,096 population's 269 → 119, this is roughly a doubling in both places, which is
what a doubled sample should do and is worth stating precisely because it means the search is not
yet saturating.

**75 of the 535 are invisible at 0.5 rad** — they pass at no clip the pre-2026-08-30 screens ever
tried. The single-clip screen was leaving 14% of the passing population unseen.

### The reproduction that matters

`sv1_u1939` came out of the 4,096 pass as the full-error leader at 0.35 win rate, clip 0.70. It
came out of this pass at **0.35 at clip 0.70** — same design, same clip, same number, from an
independently launched run. `sv1_u2699` reproduces at 0.25 as well. The pipeline is repeatable;
the draws are not so noisy that the ranking is decoration.

## Promotion, and the non-transfer result again

Six new leaders were exported at their retention clips. Five passed all four deployment gates
(`sv1_u5860` failed clearance at +3.9 mm). Then they were band-scanned **on the bench schedule** —
fixed palm, tool on a post at 100 mm, 9.6 s hold — which is a different maneuver from the
retention gate that selected them:

| design | retention clip | on the bench |
|---|---:|---|
| `sv1_w6689` | 0.50 | band **0.40–0.65**, holds 4/4 |
| `sv1_u7952` | 0.50 | band **0.50–0.70**, holds 4/4 |
| `sv1_w5120` | 0.70 | no band — 3/4 at best, never all four |
| `sv1_w7583` | 1.30 | no band — 1/4 |
| `sv1_u5855` | 0.50 | no band — drops every rollout |

**Two of five.** On 2026-08-30 the equivalent figure was three of sixteen. The retention gate and
the bench schedule keep disagreeing about which hands work, and the disagreement is not small.
Anything selected under one has to be re-scanned under the other before it is shipped.

## What went on the station

Re-exported at the clip the bench band recommends (band interior, ≥0.10 rad above the lower edge
to absorb the measured yaw droop, best alignment among what the servos can be told):

| | plan | clip | held cos | clearance |
|---:|---|---:|---:|---:|
| **1** | `sv1_w6689_b060` | 0.60 | **0.827** 4/4 | +8.5 mm |
| 10 | `sv1_u7952_b065` | 0.65 | 0.416 4/4 | +5.4 mm |

`sv1_w6689_b060` is the **best-aligned plan this program has put on the bench** — the previous
leader, `sv1_w2360_b075`, holds at 0.726. Nineteen plans now on the station, all nineteen load.

One caution on `sv1_u7952`: its alignment across the band is jagged (0.479 at 0.50, 0.281 at
0.55, **0.093** at 0.60, 0.416 at 0.65). It is shipped at 0.65 by the margin rule, but a plan
whose neighbouring clip reads 0.093 has no margin in the quantity that matters, and the hand
arrives 4–6° short under load. Treat rank 10 as a data point, not a candidate.

## Files

`selected/` — stage E, one operating point per hand. `selected/confirmed/` — what survived stage
3. `confirm_b*.json`, `full_error_b*.json` — the draws. `deploy/` — the eight exported plans and
`promotion.json` with every gate's verdict, pass or fail. `plan_bands.json` /
`plan_bands_shipped.json` — the bench band scans. `deploy_plan_bands_all.json` — those merged with
the 2026-08-30 seventeen, which is what `catalog.json` is generated from.
`deploy_clearance.txt` — all nineteen, regenerated.

The generated scene XMLs live in `assets/mjcf/experimental/20260830-real_v1-sobol4096/` alongside
the 4,096 run's, because the tags and vectors are shared.
