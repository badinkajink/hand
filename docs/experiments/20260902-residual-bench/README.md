# Bench session setup, 2026-09-02

Staging for the residual-policy work. Four hands, chosen as good baselines rather than as the
hardest cases: `sv1_w6689` and `sv1_w2360` (bench ranks 2 and 3), `rv05_manual`, and
`sv1_u1364`, whose sd of 0.221 is four times any other design's and is a mixture of grips that
held and grips that slipped rather than a distribution around a mean.

## Plans

`deploy/` holds the four plans from the transfer study, re-checked before staging:

```bash
python3 scripts/real_v1_trajectory_clearance.py \
  --plan sv1_w6689_b060 --plan sv1_w2360_b075 --plan rv05_manual_b85 --plan sv1_u1364_b080 \
  --deploy-dir docs/experiments/20260829-real_v1_deploy/deploy --substeps 8
```

| plan | chord | csv | verdict |
|---|---|---|---|
| `sv1_w6689_b060` | +8.5 mm | +8.5 mm | chord, csv |
| `sv1_w2360_b075` | +10.4 mm | +10.4 mm | chord, csv |
| `rv05_manual_b85` | +10.3 mm | +10.3 mm | chord, csv |
| `sv1_u1364_b080` | +5.0 mm | +5.2 mm | **csv only** |

`sv1_u1364_b080`'s chord sits exactly on the 5.0 mm threshold, so it must be run from its
trajectory CSV. The chord and the CSV are different paths, not two encodings of one path.

**The old plans directory should not be served.** `20260829-real_v1_deploy/deploy` contains
`g23` (+0.8 mm), `rv04_mid` (−2.6 mm) and `g24` (−5.2 mm), all excluded by the clearance gate for
interpenetrating their own fingers, and the CB1's pruned copy of it held three of those four.
The control station now points here instead.

## Services

Tracker companion, on the workstation, at the D435's full IR sensor size:

```bash
~/miniconda3/bin/python scripts/real_v1_tracker_service.py --host 10.99.99.50 --port 8770 \
  --tracker-arg=--width --tracker-arg=1280 --tracker-arg=--height --tracker-arg=800 \
  --tracker-arg=--fps --tracker-arg=30 --tracker-arg=--video-hz --tracker-arg=8
```

1280x800 is the sensor's native IR size and 1280x720 is a vertical crop of it. Probed at both:
`fx = 642.7`, HFOV 89.8 deg, and id6 measures 80.1 px edge at 327 mm either way — identical
optics and identical accuracy, with 80 more rows. At working range that is about 20 mm of extra
view below the tag, on the axis where trials are being lost: the first-pass analysis found 25 %
of trials unresolved, with the last detection a mean 9.5 mm above the floor and 24 of 24 below
60 mm. It does not replace the shorter vane, it buys margin for free.

`BENCH_DEFAULTS` in the service supplies `--shaft-axis=-x`, `--video-scale 1.0` and
`--video-quality 92`; `--tracker-arg` is appended after them, so the overrides above win and the
rig defaults survive. A session was lost on 2026-09-01 to launching this service bare.

Control station, on the CB1, via `~/run_control_station.sh start` (original backed up to
`run_control_station.sh.bak-20260902`). Three changes from the version that was there:

- `--plans-dir` moved to this directory, for the clearance reason above.
- `--tracker-url http://10.99.99.50:8770` added. It was absent, which silently disables the
  fail-closed tracking interlock — `/reorient` will otherwise run untracked trials.
- `--servo-fields load,temperature`. `load` was already on; the 08-31 and 09-01 sessions
  nonetheless produced no `servo_load`, and without it the tripped and untripped branches of
  `docs/experiments/20260902-servo-compliance` cannot be separated after the fact.
  `temperature` is new and is what makes the `protective_torque` experiment measurable: the
  SCS0009 cutoff is 70 C and the hand idles at 19-20 C.

Verified after launch: `servo_load`, `servo_torque` and `servo_temperature` all present in
`/api/v1/state`, four plans listed, tracker block live and awaiting a run.
