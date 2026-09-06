# The held turn on rv05_manual: the deployed control setting drops the tool, another one holds it at cos 0.971

2026-09-06. `scripts/rv05_carry_band.sh`, data in this folder, film
`20260906-rv05_k025.mp4` + `20260906-rv05_k025_seams.png`.

## 1. Provenance correction

`rv03_narrowy_sp40`, the design the 2026-09-05 reproduction used, **has no deploy plan and has
never been built**. The designs with an exported plan number 33; the designs physically run on
the CB1 are `g12`, `g23`, `sv1_u0060`, `sv1_u0100` and `rv05_manual`. Every carry number before
today is therefore a simulation study on an unbuilt hand.

This run repeats the same measurement on `rv05_manual_stored` and on `rv04_mid_sp40`, both of
which have plans.

## 2. Result

`--morph-run results/phase1/real_v1/rv05_manual_stored --straddle 0.040 --lift 0.10
--turn-steps 250 --angle-deg -90 --linear-anchor --repeats 6`. Table top at z = 0.0124 m.

| axis h (mm) | budget | final cos | sd | contacts | force (N) | z (m) | verdict |
|---|---|---|---|---|---|---|---|
| 1.6 | **0.85** | 0.000 | 0.000 | 0.0 | 0.00 | 0.0124 | dropped 6/6 |
| 3.3 | 0.85 | 0.000 | 0.000 | 0.0 | 0.00 | 0.0124 | dropped 6/6 |
| 4.9 | 0.85 | 0.000 | 0.000 | 0.0 | 0.00 | 0.0124 | dropped 6/6 |
| 6.6 | 0.85 | 0.000 | 0.000 | 0.0 | 0.00 | 0.0124 | dropped 6/6 |
| 8.2 | 0.85 | 0.000 | 0.000 | 0.0 | 0.00 | 0.0124 | dropped 6/6 |
| 9.9 | 0.85 | 0.148 | 0.363 | 0.2 | 0.01 | 0.0187 | dropped 5/6 |
| 1.6 | 0.5 | −0.915 | 0.007 | 1.8 | 0.15 | 0.0915 | held, opposite pole |
| 3.3 | 0.5 | −0.890 | 0.014 | 2.0 | 0.19 | 0.0932 | held, opposite pole |
| 4.9 | 0.5 | −0.936 | 0.059 | 2.0 | 0.20 | 0.0939 | held, opposite pole |
| 6.6 | 0.5 | 0.000 | 0.000 | 0.0 | 0.00 | 0.0124 | dropped 6/6 |
| **8.2** | **0.5** | **+0.971** | **0.003** | **3.0** | **6.28** | **0.1084** | **held, floor-free** |
| 9.9 | 0.5 | +0.834 | 0.007 | 3.0 | 19.27 | 0.1101 | held, floor-free |

At h = 8.2 mm and budget 0.5 the tilt is **8.6 deg at the end of the turn** and **14.6 deg after
the settle**, on three fingers carrying 6.28 N, with the tool 96 mm above the table. Six of six.

The deployed `rv05_manual_b85` plan runs `axis_k` 0.05 (h = 1.6 mm) at budget 0.85 — the first
row. The morphology, the tool, the scene and the grasp are identical across the whole table; the
only variables are the pivot height and the residual budget.

`rv04_mid_sp40` reaches 0.797 (2.2 contacts, 2.53 N) at h = 1.9 mm / budget 0.85 and is at or
near zero everywhere else, so the working cell is per-design and does not transfer between hands.

## 3. The August candidate set, load-tested

`docs/experiments/20260828-real_v1_search/table.json` records `contacts` and `force_N` per
design, so the 49-of-108 claim can be checked against the load it was carrying.

| filter | count of the 49 |
|---|---|
| contacts >= 2 and force >= 0.10 N | 48 |
| contacts >= 2 and force >= 0.24 N (the tool's weight) | 26 |
| contacts >= 2 and force >= 1.00 N | 2 |
| `final_z` < 30 mm (resting on the table) | 0 |
| `cos_turn_end` >= 0.9 | 2 |
| `cos_turn_end` >= 0.9 **and** contacts >= 2 and force >= 0.24 N | **0** |

Those 49 were genuine holds — none is the released-tool artifact that invalidated the table-stand
chain. What they were not is 90 deg turns: `cos_turn_end` spans 0.6-0.8 for almost all of them.
The 24 deg shortfall reproduced on 2026-09-05 is the same shortfall the design search reported.

## 4. Why the paper's reorientation did not carry over

- The paper's reorienter is **b33 on m05**, a legacy simulation-only topology that `CLAUDE.md`
  gate 0 lists as a historical control. `REORIENT_PRIMITIVE` attributes 46-69% of its alignment
  gain to the floor, and on `m05` that share belongs to a10, not b33.
- On `real_v1` b33 reads `peak_cos` 0.019 and `held_cos_tail` -0.093 (`docs/rl/reorientation.md`
  2026-08-27). `real_v1` is a different topology with coincident yaw/MCP axes; b33 does not
  survive a proximal-length change even on its own topology.
- **No policy has been trained since 2026-08-30.** The last RL run of any kind is
  `results/rl/20260830-S1_sighted_jitter_s43`. Every `real_v1` reorientation figure since
  2026-08-28 comes from a hand-written open-loop set-point schedule.

## 5. What this does not settle

1. **The 6 deg the settle costs.** The turn ends at 8.6 deg and relaxes to 14.6 deg. Not measured:
   whether an over-rotation (`--angle-deg` past -90, at h = 8.2 mm, n >= 6) recovers it, or whether
   it is the servo spring at 0.0186 deg per load unit and therefore not recoverable open-loop.
2. **The other four built hands.** `g12`, `g23`, `sv1_u0060`, `sv1_u0100` have not been run
   through this band. Their plans use `axis_k` 0.05, so the presumption is that they sit in the
   dropped row too.
3. **Why the pole flips between h = 4.9 and h = 8.2 mm** with a drop in between. The sign of the
   turn is not supposed to depend on the pivot height.
4. **Whether it composes.** This is measured standing alone at a fixed lift, not after a carry.
