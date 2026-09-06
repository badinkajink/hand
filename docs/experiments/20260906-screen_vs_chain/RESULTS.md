# Why plans that reorient 4/4 in the deploy screen lose the tool in the chain

2026-09-06. Data in `../20260906-rv05_band/`, films in `../20260906-rv05_band/20260906-films/`.

## 1. The deploy screen reproduces, and it matches hardware

`scripts/real_v1_plan_band.py` rebuilds each plan's trajectory from its own `meta` and runs it
in its own scene. Re-run today at each plan's shipped clip, 4 repeats:

| plan | D | sim cos | kept | bench cos | bench retention |
|---|---|---|---|---|---|
| `sv1_w6689_b060` | D1 | 0.827 | 4/4 | 0.834 | 24/24 |
| `sv1_w2360_b075` | D2 | 0.726 | 4/4 | 0.797 | 10/10 |
| `rv05_manual_b85` | D7 | 0.568 | 4/4 | 0.553 | 10/10 |

The simulation of the screened maneuver is sound. The disagreement is on the chain side.

## 2. The chain is not broadly failing, and my earlier report of it was wrong

`docs/experiments/20260904-real_v1_held/squeeze_sweep/chain_hands.json`, set A, 128 runs:

| gate | count |
|---|---|
| `held_lift` | 122/128 |
| `carry_ok` | 102/128 |
| `held_turn` | 102/128 |
| `stood_ok` | 91/128 |
| `grip_ok` | **43/128** |
| `ok` | 41/128 |

Six of the eight hands lift and hold the turn 16/16. The earlier "2 of 256 held reorientations"
was `held_reorient`, read at the `upright` seam in **table** mode, where the tool is deliberately
stood on the table and let go. It is the released-tool measurement by construction, and reporting
it as the chain's reorientation rate was an error. `held_turn`, measured while the fingers carry,
is 102/128.

## 3. Defect one: the plan's squeeze ejects a tool that is lying on a table

Every deployed plan carries `squeeze_mm` 10.0, which drives the pad centres to
`r_obj + r_pad + gap - squeeze` = 14.05 mm from the shaft axis when contact needs 23.05 mm. On
the screen's 100 mm post the post takes the reaction. On a table there is nothing behind the
shaft. Measured at closure with the palm stationary (`scripts/real_v1_grasp_refit.py`):

| tag | plan sq 10 mm: dz, pads, N | chain sq 2 mm: dz, pads, N |
|---|---|---|
| `sv1_w6689_b060` | −2.4, 3, 46.39 | −0.4, 3, 28.20 |
| `sv1_w2360_b075` | +0.6, 3, 81.75 | +0.2, 3, 22.81 |
| `sv1_u1364_b080` | **+16.9**, 3, 15.21 | +4.0, 3, 17.62 |
| `g12_b095` | **+18.6**, 3, 0.56 | +3.6, 3, 10.81 |
| `sv1_u0060_b75` | **+19.1**, 3, 28.06 | +2.3, 3, 30.79 |
| `sv1_u0308_b050` | **+19.6**, 3, 13.47 | +2.7, 3, 27.29 |
| `rv05_manual_b85` | **+15.8**, 3, 26.99 | +3.1, 3, 8.59 |
| `sv1_w0099_b100` | **+14.5**, 2, 15.81 | +0.7, 3, 23.34 |

`close_dz_mm` is the tool's rise during closure with the palm held still. Six of eight rise
14.5-19.6 mm, more than the shaft's own 12.5 mm radius: the pinch throws the tool out. The
chain's substitution to 2.0 mm holds every hand to 4.0 mm or less. This defect was already
found and fixed; it is recorded here because it is the only mechanism by which a screened plan
fails to pick the tool up at all.

## 4. Refuted: the depth was not stale

`_grip_from_fit` clamps its palm-height search to `[depth - 8 mm, depth]`, so overriding squeeze
without re-fitting depth looked like the cause. It is not. Re-fitting with the depth search free
at squeeze 2 mm returns the plan's own depth on six of eight hands and moves `sv1_w6689` by
6.0 mm, `sv1_w2360` by 1.0 and `g12` by 1.0, with pad force changing by at most 6.3 N and no
change in contact count. The transplanted depth is not what is breaking the chain.

## 5. Defect two, open: D1 lifts the tool and lets go at the next seam

`sv1_w6689_b060` is the best hand on hardware (24/24 retained, bench cos 0.834) and the worst in
the chain (`carry_ok` 0/16). Its seam trace:

| seam | z (m) | tilt | pads | pad force |
|---|---|---|---|---|
| `lifted` | 0.1008 | 87.3 | 3 | **16.71 N** |
| `turned` | 0.0125 | 90.0 | **0** | **0.00 N** |

Every later seam repeats the second row: the tool is on the table top for the rest of the run.
The film (`20260905-sv1_w6689_b060_sq2_s0_seams.png`) shows the tool in the fingers in the first
frame and lying on the table in every frame after. `rv05_manual_b85` on the same schedule reads
4 pads at 5.22 N lifted and 4 pads at 1.04 N turned, and completes.

So the failure is not the grasp and not the pickup: D1 closes on the tool with 3 pads and
28.20 N and lifts it to 100.8 mm. It releases during the phase that follows.

What distinguishes D1 from the six that work, from the data at hand: it is the only set-A hand
at `straddle_mm` 32.0 (the rest are 40.0), the shallowest at `grip_depth_mm` 52.5 (the rest
56.5-66.5), and the only one with 2 non-pad contacts at closure in the chain arm.

## 6. What this does not settle

1. **Which of the three differences causes D1's release.** Straddle, depth and the non-pad
   contacts are confounded in one hand. The test is a 2x2 on `sv1_w6689` alone: straddle
   {32, 40} x depth {52.5, 58.5}, n >= 6, scored on pad count and force at the `turned` seam.
2. **`sv1_w0099_b100` (D8) at 6/16** has not been looked at at all.
3. **Whether the chain's turn is the screened turn.** The chain runs in table mode with
   `angle_deg` 0, so the fingers do not turn the tool; the screened maneuver's turn has not
   been run inside the chain on any hand.
