# rv05_manual, three runs on a correctly configured tracker — 2026-09-02

Three consecutive operator-driven runs of `rv05_manual_b85` at speed_ratio 1.0, rate 50 Hz,
servo_speed 80, after the tracker gained the flags in `BENCH_DEFAULTS` (commit `03bd7927`).
Logs: `../20260902-cb1-log-archive/logs/`, traces `logs/tracker/*_track.csv`.

## 1. Result

| run | visibility | cos hold | turn, tag | turn, operator | settle |
|---|---|---|---|---|---|
| `d58cf0` | **1.00** (136/136) | **0.530** | 31.9° | 32° | 2.40 s |
| `dc9b4e` | 0.39 (44/112) | — lost | 32.5° partial | 33° | 1.85 s |
| `546fe7` | **1.00** (145/145) | **0.656** | 36.0° | 36° | 2.00 s |

All three scored `success=True` by the operator. The two fully tracked runs give cos hold
**0.530 and 0.656**, bracketing the transfer study's published rv05 figure of **+0.553
(sd 0.049, n=10)** — the instrument now agrees with the record.

`dc9b4e`'s hold is not reported: its tag dies at 1.85 s, so any trailing-window average covers
the rise rather than the hold. Its *peak* (0.572) is real; its hold is unmeasured, not low.

The tag angle and the operator's independent reading agree to **within 0.5° on all three**
(31.9/32, 32.5/33, 36.0/36). Worth confirming the score dialog does not display the tracked
angle before treating that as two independent instruments.

## 2. The spring predicts runs it never saw

`docs/experiments/20260902-servo-compliance` fitted deflection = 0.0186 deg per load unit from
236 archived runs of *other* designs and clips. Applied to these three, at the terminal hold:

| run | joint | deficit | 0.0186 x load | error |
|---|---|---|---|---|
| `d58cf0` | `middle_yaw` | 13.63° | 13.67° | **0.04°** |
| `dc9b4e` | `middle_yaw` | 10.70° | 11.16° | 0.46° |
| `546fe7` | `middle_yaw` | 10.70° | 10.88° | 0.18° |
| `dc9b4e` | `thumb_mcp` | 7.86° | 7.81° | **0.05°** |
| `546fe7` | `thumb_mcp` | 8.44° | 8.65° | 0.21° |
| `d58cf0` | `thumb_mcp` | 6.68° | 8.09° | 1.41° |

One constant, fitted elsewhere, predicting the two largest deficits on unseen runs to a few
tenths of a degree. `middle_yaw` gives up 10.7-13.6° of a commanded 32.68° — a third of the
turn joint's travel, lost to compliance alone.

## 3. `protective_torque` is not the limiter on this hand

`middle_yaw` terminal load reads **735, 600, 585** across the three. The trip is 200. Nothing
tripped, every joint stayed on the linear branch, and the shortfall is entirely the spring. The
20 -> 40 register experiment proposed in `../20260830-real_v1_bench_suite/README_open_loop.md`
would change nothing here; it is an experiment for a hand and clip that actually trips, which
2026-08-30's `sv1_u0100` (42% trip rate) does and `rv05_manual` at b85 does not.

## 4. Tag dropout is now rare, and it is not exposure

Fixed exposure lifted decode margin from a mean of 28 on auto to **48-54** across these runs.
Two of three tracked every frame.

`dc9b4e` still lost its tag, and the manner rules out the obvious causes:

- **not margin decay** — it died at margin 49.7, its own minimum but a healthy one
- **not out of frame** — last detection at `u,v = 809,560` of 1280x800, centre
- **not height** — it died at tag_z 17.1 mm, between `d58cf0`'s 19.8 and `546fe7`'s 11.5, both
  of which tracked to the end

An abrupt loss at healthy margin in the middle of the frame is occlusion or a pass through
edge-on, not a threshold being grazed. **This retires the vane-length theory** in
`../20260901-real_v1-transfer-firstpass` section 4, which inferred "out of frame" from world
height without checking image coordinates. A second tag on a perpendicular face is the fix that
addresses the actual mechanism.

## 5. What this session got wrong

Recorded because the errors were in reading the instrument, and the instrument is the thing the
residual policy will be trained against.

- Two earlier runs today reported cos **-0.684** and were nearly written up as no-rotation
  failures. The cause was a missing `--symmetric-object`: the 100x25 cylinder is symmetric
  end-for-end, so which pole lands on top is a property of seating, and the flag folds cos to
  [0,1]. The turn was 38°, held four seconds, centre up 6 mm.
- The tag loss was diagnosed as field of view and "fixed" by moving to the D435's native
  1280x800. That change is real (identical intrinsics, 80 more rows) but was not the problem.
- `--shaft-axis=-x` was suspected of being inverted. With `--tag-end top` the probe checks
  rather than narrates, and reports `-x is CORRECT`.
