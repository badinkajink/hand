# The held floor-free turn reproduces at 65.7 deg, and the table-stand chain does not reorient

2026-09-05. Data in this folder. Supersedes today's earlier claim of a solved reorientation.

## 1. The table-stand chain: RETRACTED

`stood_ok` is `ground_contacts >= 1 and tilt_deg < 14 and |z - rest_z| < 0.010`. It never tested
whether the tool was still in the grasp, and the `reorient_deg` built on it inherited that. Over
the 177 runs (of 256) that passed the gate, at the `upright` seam:

| pads on the tool | runs |
|---|---|
| 0 | **116** |
| 1 | 57 |
| 2 | 4 |

Median pad force 0.000 N against the tool's own 0.240 N weight; 2 runs exceed it. Held
reorientation is **2 of 256** at a >1 N threshold, **11 of 256** at 0.8x the tool's weight
(the correct, mass-referenced threshold -- steady-state grip on a free object is bounded by its
weight, so 1 N was never achievable as a carrying force). Either way the maneuver sets the tool
down, releases it, and it settles vertical.

Also, on every run: `tip_len_mm = 0.0` (plain cylinder, axis folds, either end down scores
tilt 0), `place_xy = None` (no countersink), `reindex = "full"` (release, not relay). 79 of the
177 stands end below the table top.

`chain()` now returns `upright_pads`, `upright_force_N`, `held_reorient`, and `reorient_ok`
requires them. **Path aborted.**

## 2. Two handover levers, both negative

Measured before the retraction and unaffected by it (128 runs per arm, 16 hands x {2,4} mm x 4 seeds):

| gait palm-height scan | handover | chain |
|---|---|---|
| grip (shipped) | 49 | **44**/128 |
| open | 29 | 21/128 |
| both | 46 | 35/128 |

Opening the pads wider before the descent changes the handover count by **exactly zero** at
release 6, 11, 16 and 22 mm. `ring_ik_open_mm` grows one-for-one with the target
(14.3 -> 26.9 mm), so the fingers are already as open as they go; pad radius is not a lever.

## 3. The held floor-free turn DOES reproduce -- 65.7 deg, 30/30

The 2026-08-28 result was chased through three mis-invocations before the settings were read out
of `docs/experiments/20260827-real_v1/carry_pospole.json` field by field rather than from prose:

| field | value | what a naive rerun uses |
|---|---|---|
| `max_ik_residual_mm` | **0.0** -> `--linear-anchor` | per-step IK, which RELEASES the shaft (grip decays 16.1 -> 0 N in 0.8 s) |
| `budget_rad` | **0.5** | 0.85 (the plan's) |
| `turn_steps` | **250** | 400 |
| `angle_deg` | **-90** | +90 gives a negative cosine |
| run | `rv03_narrowy_sp40` | rv05_manual's deployed plan params (40 mm half-straddle vs 24.7) |

`probe_real_v1_carry.py --morph-run results/phase1/real_v1/rv03_narrowy_sp40 --straddle 0.040
--axis-k 0.1,0.15,0.2,0.25,0.3 --angle-deg -90 --budget 0.5 --turn-steps 250 --repeats 6
--linear-anchor --lift 0.10` -> `carry_repro_exact.json`:

| axis h | final cos, all 6 repeats | contacts | force, all 6 (N) |
|---|---|---|---|
| 3.8 mm | 0.897 0.909 0.874 0.876 0.884 0.856 | 2,2,2,2,2,2 | 21.4 21.2 21.3 17.7 21.2 20.5 |
| **5.8** | **0.912 0.916 0.905 0.917 0.911 0.908** | 2,2,2,2,2,2 | **22.4 22.8 22.5 21.9 22.6 22.9** |
| 7.7 | 0.909 0.909 0.887 0.903 0.885 0.884 | 2,2,3,2,3,2 | 12.9 13.9 16.9 14.4 17.8 10.8 |
| 9.6 | 0.859 0.853 0.863 0.857 0.864 0.866 | 2,2,2,2,2,2 | 4.2 5.4 4.1 6.3 4.2 4.1 |
| 11.5 | 0.836 0.844 0.829 0.800 0.827 0.807 | 2,2,2,2,2,2 | 0.9 1.2 0.9 1.9 0.9 1.0 |

**0 drops in 30. Minimum 2 contacts, minimum 0.92 N.** At h = 5.8 mm the turn ends at
cos 0.912 +- 0.004 holding 22.4 +- 0.4 N, tool at z = 0.111 with the floor 60 mm below.

**It is 65.7 deg of the 90, not 90.** It stops 24.3 deg short, consistently. Grip force falls off
a cliff with axis height -- 22 N at 5.8 mm, 0.9 N at 11.5 mm -- so the band is narrow, and the
deployed plans' `axis_k = 0.05` sits at h ~ 2 mm, below its bottom.

## 4. What the 2026-08-28 note claims vs what its own data holds

`MECHANISM.md` quotes rv05_manual at 0.996/0.996, 3 contacts, 12.0 N. In the source JSONs the
same cells at n=3 give 22.3 / 22.9 / **5.1 N**, and at `axis_k` 0.25 and 0.30 one repeat in three
**drops the shaft**. `carry_fine.json` contains a row reading **cos 0.993, 0 contacts, 0.0 N** --
the released-tool artifact was already in the dataset that established the mechanism. The
headline was a best-of-three. The n=6 rerun above is the number to use.

## What this does not settle

1. **The last 24.3 deg.** Nothing here measures what closing it costs. The sweep to run is
   `--angle-deg` past -90 (over-rotate) and `--axis-shift` at h = 5.8 mm, n >= 6, scored on
   final cos AND contacts AND force.
2. **rv05_manual is unmeasured under the correct settings.** Only `rv03_narrowy_sp40` has been
   rerun. The user's own hand needs the same n=6 pass before any bench claim.
3. **Whether this composes with a carry and a handover.** The turn is measured standing alone at
   a fixed lift. Nothing has taken it into a grasp-carry-turn-release sequence.
4. **The n=3 drops did not recur at n=6.** Either seeding or something in the code changed since
   August; unexplained, and worth one bisect before the number is leaned on.
