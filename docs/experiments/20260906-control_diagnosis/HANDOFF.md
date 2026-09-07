# Control diagnosis, 2026-09-06 — state for the next session

Page: `20260906-control_diagnosis.html` ·
https://claude.ai/code/artifact/f10f07dd-6812-4c1b-a55c-9bd58c125924

## The one thing that works, and how to check it in 2 seconds

    PYTHONPATH= uv run --extra rl --extra arm --extra dev python -m pytest \
      tests/test_chain_reference.py -q

`rv05_manual_stored`, spheres, CEM anchor from `best_rollout.npz`, `axis_k` 0.25, clip 0.50,
angle −90, `turn_steps` 550, floating palm. Reorients to **cos +0.994 on 3 pads at 10.47 N**,
tip down, gaits 8/8. 169/169 on 2026-09-03. **Run this before and after touching
`probe_real_v1_chain`, `palm_driver`, the scene generators or the fitters.** `PYTHONPATH=` is
required — the ROS `launch` package on the system path breaks pytest collection.

## The two defects

1. **`axis_k` decides which end goes down.** 0.25 → tip down; the plans' 0.05–0.15 → the turn
   carries the tool to cos −0.988, handle down, and drops it. The lift is bit-identical either
   way, because the pivot only enters after it.
2. **`_grip_from_fit` spends the finger travel the turn is paid for in.** Closed-grip
   mount-to-pad distance against the 68.11 mm chain: CEM 62.6/63.7/65.4 mm (**2.74 mm of
   headroom**), fitter 66.0–67.7 mm at every squeeze from 2 to 12 mm (**0.44–1.26 mm**). No
   squeeze value chains. The 4 mm cell lifts on 3 pads at 8.79 N — *more* than the reference —
   and is on the floor at the next seam, so grip force is not the missing quantity.

Not defects: pad shape (flat pads *with a grasp fitted for them* lift 3/4 on 3 pads at 3.37 N;
the two arms that looked like pad failures were mismatched grasp/geometry and both died at the
**lift**), the arm (the reference runs on the UR5e in `20260904-real_v1_bench`), the turn angle
(−60 under-rotates and still chains 3/4).

## The next measurement, named

**Add an extension-headroom constraint to `_grip_from_fit`** — reject a pose leaving under
~2.5 mm of the 68.11 mm chain — and widen its palm-height search past the current
`[depth − 8 mm, depth]` clamp, because the constraint is unsatisfiable inside it: asked for 62,
58, 54, 50, 46 mm on `rv05_manual_stored`, the fitter returns **no pose at every one**. Its only
feasible depth on that hand is the deepest, which is the problem.

Accept if a headroom-constrained fit on `rv05_manual_stored` reproduces the reference chain
(the regression test above, with `--squeezes`/`--depths` off). Then re-run the pivot × clip
sweep on the eight deployed hands — not before; the 432-rollout sweep already run
(`20260906-chain_band`, clip 0.40–1.15 × pivot 0.15/0.25/0.35) gives **0 chains**, and there is
no reason to expect a pivot value to help a grasp with no travel left.

## Traps that cost this session

- **Score `cos`, never `tilt_deg`.** `_av()` folds the axis when `tip_len == 0`, and
  `real_v1_chain_hands._cell` passes `tip_len=0.0` for every table-stand cell. **952 of 967**
  reported table-mode stands were the tool standing on its **handle**. `--stand table` is
  sign-blind, full stop. Pair the cosine with ≥2 pads at ≥0.240 N.
- **`axis_k`, clip and angle are per-hand AND per-maneuver.** `docs/experiments/OPERATING_POINTS.json`
  (built by `scripts/real_v1_operating_points.py`) is the record; the chain column is
  `measured: false` for all eight. Never pass one clip for all hands — 0.50 is below the bench
  band for four of seven.
- **Read the memory FILE, not the MEMORY.md hook.** The pole flip was already written in
  `project_rv05_carry_band.md` ("h 1.6-4.9 mm / 0.5 → held but the opposite pole") and
  re-derived from scratch anyway.

## Scripts added 2026-09-06

| script | what |
|---|---|
| `real_v1_chain_ablate.py` | one factor at a time off the reference; `--squeezes`, `--depths` |
| `real_v1_grasp_closure.py` | pad contact height on the shaft, wedge sign, `Fz_pad` |
| `real_v1_operating_points.py` | writes `OPERATING_POINTS.json` from the source studies |
| `tests/test_chain_reference.py` | pins the 169/169 cell on the signed cosine |

`real_v1_chain_hands.py` gained `--hands --straddles --depths --elevations --turn-steps
--budgets --axis-ks --angles`. `palm_driver.GantryPalm` gained the three IK counters that had
killed the whole floating-palm path.


## 2026-09-06, later: the turn IS solved on three deployed hands

Scanning every chain sweep on the **signed** cosine with a load test at `reoriented`
(cos > +0.85, >= 2 pads, >= 0.240 N, z > 0.08): 1,550 rollouts, **zero** completed chains on any
hand but the reference. But `docs/experiments/20260906-pivot` (96 rollouts, clip 0.50, angle -90,
pivot 0.15/0.25/0.35) holds the tip-down turn on three:

| id | hand | pivot | cos | pads | force | z | lost at |
|---|---|---|---|---|---|---|---|
| D5 | `sv1_u0060_b75`  | 0.15 | +0.894 | 2 | 14.13 N | 123 mm | `set_down` |
| D3 | `sv1_u1364_b080` | 0.35 | +0.987 | 2 |  3.36 N | 121 mm | `set_down` |
| D1 | `sv1_w6689_b060` | 0.35 | +0.913 | 1 | 10.10 N |  55 mm | `staged`   |
| D6 | `sv1_u0308_b050` | 0.15 | +0.568 | 3 |  4.20 N | 121 mm | chains 3/4, 35 deg short |

D5 holds 8/8 seeds across pivots 0.15 and 0.25. D3 is seed-fragile (2 of 4 seeds go to the wrong
pole at the same pivot).

**The pivot is per-hand.** 0.15 for D5, 0.35 for D3/D1, 0.25 for the reference.

**Why the band sweep missed it.** `20260906-chain_band` stepped clip 0.40 / 0.55 / 0.70 / 0.85 /
1.00 / 1.15 and never tried 0.50 — the clip all three of these turns run at. Its "0 chains in 432
rollouts" is true of that grid only.

**The named next measurement (revised).** Instrument the **set-down seam** on those three cells
before touching `_grip_from_fit`. The films separate two mechanisms that the `drop_stage` field
does not: D5's tool topples out of the pads onto the disc; D3's UR5e wrist descends into the
table. Neither is a turn failure, and the fitter is not implicated in either until they are told
apart. Re-run with the descent logged per step and per contact.

Structural note: the reference chain has **no arm and no table disc** — floating gantry palm,
tool stood on the floor. Every deployed-hand chain adds the UR5e and the countersink at exactly
the seam that fails.

## Films

Both grids and per-hand mp4s + seam filmstrips:

- `docs/experiments/20260906-held_turn_films/` — eight hands, each at its own best pivot,
  ranked by the signed held turn. Built with
  `scripts/real_v1_chain_films.py --sweep docs/experiments/20260906-pivot/chain_hands.json
  --rank held --mode air --date 20260906`.
- `docs/experiments/20260906-ablate/20260906-films/` — the eight ablation arms on the reference
  hand. Built with `scripts/real_v1_chain_ablate.py --films <dir> --reps 1`.

`real_v1_chain_films.py` gained `--rank held|chain`, `--mode air|table|sweep` and `--date`, and
now parses the sweep's own arm tag back into kwargs so a film re-runs the picked cell rather than
the base config. Verified: all eight films reproduce their sweep row's `reoriented` seam exactly.
