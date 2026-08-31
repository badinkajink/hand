# The residual clip is a screening axis, and nobody had ever varied it

**2026-08-30, evening.** Every morphology screen this program has run — the 108-hand search, the
128-hand pilot, the 4,096-hand retention screen — planned its turn with `budget = 0.5` rad,
hardcoded. That number is **Policy B's residual action budget**: a cap on how far a learned
residual may push a joint away from its anchor. It has no business constraining an open-loop
trajectory, and it arrived here by inheritance rather than by choice.

This is that decision re-opened on two fronts: the plans already on the bench, and the population
they were drawn from.

**A digestible version of this day, with the band grid drawn:**
<https://claude.ai/code/artifact/865c4572-ce31-40b9-a616-da5ca28a9f57>
(source kept here as [`clip-cap-stall.html`](clip-cap-stall.html)).

---

## Part 1 — the plans on the bench

`scripts/real_v1_plan_band.py` rebuilds each deployed plan from its own metadata, checks the
rebuild against the shipped set-points (all 14 match to 0.000°), and then re-runs it at 33 clips
from 0.40 to 2.00 rad, four repeats each, with the 9.6 s hold. 1,848 rollouts.

Every plan has a **contiguous band** of clips inside which it keeps the tool, and drops it on both
sides. Cells are rollouts kept out of 4; `.` means every repeat dropped.

```
plan               own   0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30
g12 / w08 / w11    var      4    .    .    .    .    .    4    4    4    4    4    4    4    4    1    1    .    .    .
g23               0.50      4    .    .    .    .    .    .    4    4    4    4    4    4    4    4    4    4    4    4
g24               0.50      .    .    .    .    4    4    4    4    4    4    4    4    4    4    4    1    2    2    .
rv04_mid          0.50      .    .    .    .    .    2    4    4    4    4    4    4    4    4    4    4    4    4    4
rv05_manual_b85   0.85      4    .    .    .    .    .    .    4    4    4    4    4    4    4    4    4    4    4    3
sv1_u0060_b75     0.75      .    .    .    .    3    4    4    4    4    4    4    4    4    4    4    4    4    4    4
sv1_u0100_b70     0.70      .    .    .    .    4    4    4    4    4    4    4    .    .    .    .    .    .    .    .
sv1_w0099_b100    1.00      4    1    .    .    .    .    .    .    .    2    2    4    3    2    .    2    .    1    1
sv1_w0116_b100    1.00      .    .    .    .    .    .    .    .    .    .    .    .    2    .    .    .    .    .    .
```

**Three of the plans the bench has been running are outside their own band.** `g12`, `g23` and
`rv04_mid` at 0.50 rad are all *below* theirs and drop 4/4. `g12w11` at 1.10 is *above* g12's
upper edge of 1.05 and holds 1/4.

**The isolated 4/4 at 0.40 is not an operating point.** It has 0/4 immediately above it on every
design that shows it: a turn small enough not to disturb the grip is not a turn. The band is always
the longest contiguous run, which is what the script reports.

### Alignment is not monotonic across the band

The pilot's six plans, measured on the pilot's own cells, had alignment falling monotonically from
the lower edge — so the ship rule became "lower edge plus a margin". Measured on the cells the
bench actually runs, that does not hold:

```
g12         clip  0.70  0.75  0.80  0.85  0.90  0.95  1.00  1.05  1.10
            cos   0.61  0.52  0.48  0.49  0.60  0.63  0.59  0.52  0.27
rv04_mid    clip  0.60  0.65  0.70  0.75  0.80  0.85  0.90
            cos   0.25  0.94  0.74  0.49  0.26  0.09  0.00
```

g12 dips in the middle of its band and recovers; rv04_mid peaks hard at 0.65 and is worthless by
0.90. **Read the shape.** What survives from the edge rule is the *margin* — stay clear of the
lower edge, because the joints do not arrive (Part 3).

### The clip has a ceiling, and it is the servos

`anchor` and `delta` do not depend on the clip, so the largest commandable clip is closed-form
rather than sampled:

| design | largest commandable clip (rad) | first joint over |
|---|---:|---|
| most of the family | 1.345 | `turn_end : middle_fe2` (+77.05° cap) |
| sv1_w0099 | 1.212 | `turn_end : middle_aa` |
| sv1_u0060 | 0.932 | `turn_end : middle_aa` |
| sv1_w0116 | 0.891 | `turn_end : middle_aa` |
| g24 | 0.650 | `turn_end : thumb_fe1` |

`sv1_w0116`'s hold band is 1.45–2.00 and lies **entirely above** the largest clip it can be
commanded to. There is no clip at which that hand both holds the tool and can be told to. It was
the 128-hand pilot's nominal winner.

### What changed on the station

- **`g12_b095` is new** and supersedes `g12w08`/`g12w11`: 4/4 held at cos 0.627, the best
  simulated alignment of any plan in the directory, against 0.484 and 0.267.
- **The four budget-1.0 audit exports were added** at the user's request. Two of them
  (`sv1_u0060_b100`, `sv1_w0116_b100`) are 3.88° and 6.27° outside the ±70° yaw cap, and
  `HandRuntime.load_plan` *raises* on any validation failure — so they will appear in the dropdown
  and refuse to load. Nothing is silently clamped.
- **`catalog.json` now carries a prediction per plan** — band, held cosine at its own clip,
  clearance, largest commandable clip, a recommended clip, and a sentence of what to expect —
  generated by `scripts/real_v1_plan_catalog.py`. The station shows it. It exists to be falsified
  by the next bench session.

Expected ranking, simulation only, plans that hold at their own clip:

| # | plan | clip | band | held cos | clearance |
|---:|---|---:|---|---:|---:|
| 1 | `g12_b095` | 0.95 | 0.70–1.05 | 0.627 | +8.7 mm |
| 2 | `sv1_u0060_b75` | 0.75 | 0.65–2.00 | 0.597 | +9.9 mm |
| 3 | `rv05_manual_b85` | 0.85 | 0.75–1.25 | 0.568 | +10.3 mm |
| 4 | `g12w08` | 0.80 | 0.70–1.05 | 0.484 | +8.7 mm |
| 5 | `sv1_u0100_b70` | 0.70 | 0.60–0.90 | 0.231 | +7.5 mm |

`g23`, `g24` and `rv04_mid` are excluded on clearance (+0.8, −5.3, −2.7 mm) as well as on band.

### Reproduce

```bash
python3 scripts/real_v1_plan_band.py \
  --deploy-dir docs/experiments/20260829-real_v1_deploy/deploy \
  --budgets 0.40,2.00,0.05 --reps 4 --workers 6 --out deploy_plan_bands.json
python3 scripts/real_v1_trajectory_clearance.py \
  --deploy-dir docs/experiments/20260829-real_v1_deploy/deploy --all --substeps 8 \
  > deploy_clearance.txt
python3 scripts/real_v1_plan_catalog.py --bands deploy_plan_bands.json \
  --clearance deploy_clearance.txt --deploy-dir docs/experiments/20260829-real_v1_deploy/deploy
```

---

## Part 2 — the population

`scripts/real_v1_budget_rescreen.sh` re-runs every fitted grasp cell in the 4,096-hand population
at five clips under the full retention gate: a 60 mm proof lift the object has to follow, five free
seconds with at most 10 mm of slip, under the SCS0009 torque caps. 28,480 rollouts.

It also carries a gate the screen never had — `servo_short_deg`, the `HandPlan.validate` violation
computed from the carry plan. A trajectory the driver would refuse is not a candidate, however well
it scores.

| clip (rad) | cells | pass the retention gate | …and are commandable | blocked by servo range |
|---:|---:|---:|---:|---:|
| 0.50 | 5,696 | 315 | 295 | 2,031 |
| 0.70 | 5,696 | 264 | 230 | 2,856 |
| 0.90 | 5,696 | 221 | 110 | 4,042 |
| 1.10 | 5,696 | 217 | 73 | 4,927 |
| 1.30 | 5,696 | 212 | 53 | 5,227 |

### Raising the clip does not help this maneuver

Fewer cells pass at every step above 0.50, and the servo cap blocks a steeply growing share. That
is the opposite of what the *deployed* plans wanted, and the two results are not in conflict:

- the **population screen** ends with a proof lift and five free seconds of hold. A bigger turn
  excursion costs grip before the lift, so the clip that maximises retention is small.
- the **bench plans** hold the tool on a post with a fixed palm. A bigger turn is what produces
  the rotation, so three of them were below their own band and dropping every rollout.

The mistake was never that 0.5 is the wrong number. It is that one number was used for two
different maneuvers, and neither was measured. **State which maneuver a clip belongs to.**

### What the sweep recovers

| clip a hand does its best work at | hands |
|---:|---:|
| 0.5 | 144 |
| 0.7 | 70 |
| 0.9 | 19 |
| 1.1 | 15 |
| 1.3 | 21 |

Of **269** morphologies that pass the gate at some clip,
**39** pass at no value the old screen ever tried, and 125 do
their best work away from 0.50.

### The servo gate catches hands the old screen promoted

**18 of the 248 hands that pass at 0.50 rad cannot be commanded.** They
would have failed at export, four stages later, after being confirmed, ranked and rendered.

### Why the stage-1 numbers are not the answer

Stage 1 is *one* rollout per cell per clip. Each hand gets twenty chances (four grasp cells × five
clips) to produce its best number, so taking the maximum selects hard on noise — single-trial
cosines run to 0.998 and will not survive. The point of making stage 1 cheap is to afford stage 2:
five nominal repeats plus a four-draw wrong-hand ensemble at each survivor's chosen clip, then
twenty full-error draws for what remains (`scripts/real_v1_budget_confirm.sh`).

The population screen's cosine is measured after a proof lift and a free hold; Part 1's is measured
after a fixed-palm hold on a post. **They are different maneuvers and the numbers are not
interchangeable.**

### Reproduce

```bash
bash scripts/real_v1_budget_rescreen.sh      # stage 1, ~50 min on 18 workers
bash scripts/real_v1_budget_confirm.sh       # stages 2-4
python3 scripts/real_v1_promote_designs.py \
  --table  docs/experiments/20260830-real_v1-budget-rescreen/selected/confirmed/selected_table.json \
  --manifest docs/experiments/20260830-real_v1-sobol4096/hardware_manifest.json \
  --generated-dir assets/mjcf/experimental/20260830-real_v1-sobol4096 \
  --designs-file <the shortlist> --out-dir <dir> --ship <the station's plans dir>
```

---

## Part 3 — promotion, and what did not transfer

The confirmation pass leaves **119 of 269** morphologies (five nominal repeats, ≥3 kept, cos ≥ 0.7,
≥5 mm rollout clearance, commandable), and **50 of those 119 confirm at no clip the old screen ever
tried**. Twenty full-error draws each put two new hands ahead of anything the single-clip screen
found:

| design | clip | grasp cell | nominal cos | full-error win | rollout clearance |
|---|---:|---:|---:|---:|---:|
| `sv1_u1939` | 0.70 | 32/10 | 0.821 (5/5) | **0.35** | 13.2 mm |
| `sv1_u2745` | 1.10 | 32/10 | 0.799 (5/5) | **0.30** | 8.9 mm |
| `sv1_u1364` | 0.50 | 40/20 | 0.913 (5/5) | 0.25 | 19.1 mm |
| `sv1_u2699` | 0.50 | 32/10 | 0.867 (5/5) | 0.25 | 12.2 mm |
| `sv1_u0308` | 0.70 | 40/10 | 0.799 (5/5) | 0.25 | 16.3 mm |
| `sv1_w3408` | 0.70 | 40/20 | 0.707 (5/5) | 0.25 | 17.3 mm |

The previous best on this population was `sv1_u2699` at 0.25, and it is now fourth. Both new
leaders are at a clip the old screen never ran.

### Sixteen went to the gates; thirteen passed; five hold the tool on the bench

Thirteen of the top sixteen pass export, servo range, gantry travel and trajectory clearance — and
**no design failed on servo range**, which is the screen's new gate doing its job upstream. The
three that failed, failed on clearance, and they failed it the way the roadmap warned about:
`sv1_w3408` reads 17.3 mm in the rollout and **+4.6 mm** along the commanded chord, `sv1_u0720`
reads 5.7 mm and **−4.4 mm**. The screen's clearance figure is not the export's.

Then the harder question. These hands were selected under the **retention** maneuver — proof lift,
free hold. The bench runs the **fixed-palm** schedule with the tool on a post. Band-scanning all
sixteen exported plans on that schedule:

| plan | bench band (rad) | best cos | recommended clip | largest commandable |
|---|---|---:|---:|---:|
| `sv1_w2360` | 0.50–0.80 | 0.739 | **0.75** | 0.85 |
| `sv1_u1364` | 0.40–0.95 | 0.712 | **0.80** | 0.80 |
| `sv1_w3408` | 0.40–1.00 | 0.708 | 0.50 | 0.80 |
| `sv1_u0308` | 0.40–0.85 | 0.712 | **0.50** | 0.80 |
| `sv1_u1676` | 1.05–2.00 | 0.515 | — band is uncommandable | 0.80 |
| `sv1_u2175` | 0.60–0.75 | 0.369 | 0.70 | 0.85 |
| the other ten | none | — | — | — |

**Eleven of the sixteen keep the tool at no clip at all on the bench schedule** — including both new
leaders, `sv1_u1939` and `sv1_u2745`. A hand chosen for proof-lift retention is not a hand that
rolls a tool resting on a post. `sv1_u1676` repeats `sv1_w0116`'s failure exactly: a band that lies
entirely above the largest clip its servos can be commanded to.

### Three new hands on the station

Re-exported at their **bench** clip, re-gated, and copied to the plans directory:

| plan | clip | bench band | held cos | clearance | full-error win (retention) |
|---|---:|---|---:|---:|---:|
| `sv1_w2360_b075` | 0.75 | 0.50–0.80 | 0.726 | +10.4 mm | 0.20 |
| `sv1_u1364_b080` | 0.80 | 0.40–0.95 | 0.711 | +5.0 mm | 0.25 |
| `sv1_u0308_b050` | 0.50 | 0.40–0.85 | 0.585 | +9.2 mm | 0.25 |

`sv1_w3408` is not among them: its export clearance is +4.6 mm at every clip, because the minimum
sits at the grip pose, which the clip does not touch.

**These are the first hands to reach the bench from a search that varied the clip**, and the first
to have passed a servo-command gate before they were ranked rather than after.
