# HOM trials on the SR2 hand: runbook

How to run a HOM trial on the SR2 hand on the UR5, record it, and use it to calibrate the simulation (owner,
2026-11-15: "make sure we are ready to start collecting data and calibrating from real trials of HOM"). The trial
sequence and the series are those of `docs/notes/20261105-object_seating_and_tracking/20261105-object_seating_and_tracking_plan.md`.

## State on 2026-11-20 17:5x: the blind pick-up

- **17:30-17:41.** A4feas_f76_x105y160g0_t2 stopped on the drift guard (30.3 mm), and the sphere toppled against the
  cable bundle in front of the arm's base. The series' pick-up refused the 62.4 deg wrist turn (the limit was 60; it is
  90 since 7969e33cb). The pick-ups run by hand at 17:38-17:41 stopped in the descent on the tracker. Behind the turned
  hand the lying block showed 1-2 tags, and its pose jumped 11.1, 13.3 and 5.3 mm, the last with a 110 deg flip. The
  HEX-E read at most 1.4 N.
- **17:51-17:55: the user's go, "lets just fly blind here and go for the pinch".** `pinch_pickup.py grasp --blind`
  measures the sphere once, with the hand high and unturned. The turn, the descent and the close then run without the
  tracker, watched by the HEX-E (3 N) and the finger joints (4 deg). The grasp went 124 mm down with at most 1.6 N; the
  tracker gave no pose at the grasp's height. `lift --blind --lift-mm 65.2` (standing height 73.9 + 12 - 20.7 mm lying)
  lifted it, the slip took it from 59 to 10.7 deg, and the putdown left it 0.7 deg from vertical at the table's centre.
  `unturn` turned the wrist back.
- **The series** restarted from entry 90 (A4feas_f17_x95y140g0_t1) through `wait_upright.py` at about 17:58. Its own
  pick-up still runs the watched grasp.

## State on 2026-11-20 10:5x: the morning's two stops, the owner's order

- **The login.** The Claude login of the session watching the bench reached its fixed time limit before 08:04 ("this
  login reached its time limit and cannot be renewed"). A claude.ai login lasts about four weeks, and using Claude Code
  does not extend it. Nothing could act on the bench until the owner logged in again at 10:33. `lab/claude_login.py`
  now prints the renew-by date when `hom_series.py` and `wait_for_lights.py` start, with WARNING within 2 days of it.
  The current login: renew by 2026-12-15, ends 12-18 19:10. Before arming an unattended run, read that line. The
  one-year token of `claude setup-token` makes model requests only: no Remote Control, no claude.ai connectors.
- **09:17-09:21.** The lights were on from 09:07, and the watcher started the morning series at 09:17.
  A1_r15_x90y135g3 t1 and t2 twisted -11..53 and -6..55 deg, with a rise of 10.2 mm each. t2's putdown stopped in
  the levelling turn: the sphere moved 2.18 mm against the hand (limit 2). The sphere hung 49 mm above the table until
  10:38, servo 0 reached 52 C and servos 1, 3 and 6 flagged overload. `putdown --level-max 0` set it down and let go
  at 10:39. t2's row was added to the morning jsonl by hand (field `note`).
- **hom_series:** when a putdown's levelling stops, the series now runs the putdown again without levelling.
- **The owner's order (08:0x, `docs/notes/20261120-twist_ranking_decisions.md`):** "We don't have to run the full
  infeasible set, we should get more coverage among the probably good and feasible set of hands first."
  `lab/series_entries_20261120_day.txt` (`lab/series_20261120_day_order.json`) has 170 trials: the rest of A1 (51),
  A4feas (40: the 20 predicted-good feasible layouts of 0ec04cd54, 2 trials each, ports 47162-47181), the (70, 110)
  anchors (9), b3E (18), b4cem (33), then b2touch, A2bad and A3drop at t1 only (19). Started at 10:43, log
  `logs/20261116-hom_trials/20261120-day_series.log`, rows to the same `20261120-series_morning_seed20261120.jsonl`.
- **10:45, stopped again.** A1_r15_x90y135g3_t3 twisted -4..58 deg, but the sphere rolled in the grasp: drift 22 mm,
  49.6 deg off P's axis. The 35 deg levelling turn ran out of time after 96 mm. The putdown without levelling then set
  the sphere down, seated, and released it. But both ways of straightening were refused: the index tip end starts
  3.3 mm into the block in the model, and the paths take it to 4.0 and 4.9 mm. The hand holds its open pose over the
  standing sphere, the sphere 18 mm off the grasp's axis. No coded step backs out of this (`--rise-open` refuses past
  its lateral limit), so it waited for the owner.
- **10:5x, the owner reset the arm** (level, the palm 107 mm beside the upright sphere, 118 mm from the table's centre).
  `letgo` straightened the fingers and lifted the palm at 10:55 (the path to zero kept 28.6 mm from the ball). The
  approach's lateral limit (40 mm) refuses a move that long, and at the letgo height the straight thumb hung 22.6 mm
  below the sphere's top. So: `pose_sweep.py clear --go` (up 49.6 mm, the fingers 27 mm above the top), then the lateral
  move alone, `hom_setup.py approach --plan NEXT_PLAN --fine --fine-max 130 --go` (120 mm, no descent). The cylinder
  stood at x_B -44 mm, outside the fingers' path. t3's row was added by hand (field `note`), and the series restarted
  at 10:59 from entry 2 of the day file (`$(cut -d" " -f2- series_entries_20261120_day.txt)`), appending to the same log.
- **11:02-11:05, A1_p35_x105y100g15_t2.** The 32 deg levelling ran out of time again (89 mm in 12 s). The putdown
  without levelling then stopped on a 1.5 mm slip in the grasp (the lateral touch limit), the block still 41 mm up. The
  series' "not seated" branch ran `letgo --tilt-ok`, which refused because of the height. So the row's putdown field
  ("tilted; letgo --tilt-ok") is wrong. By hand: `putdown --touch-lat 3` stopped 4.5 mm up, on a 0.82 mm touch along
  z_P with 2 tags seen. `letgo --seat-mm 6` then failed: the twist had left the middle PIP at -17.3 deg, and the
  station refused the release ramp's first step (-16.25, the driver's limit -16.11). `letgo --seat-mm 6 --ramp-s 0` let
  go at 11:05. Servo 0 had reached 51 C.
- **Trials that started tilted.** The tilt at `ready` (the object's axis against z_P) was 1.2, 12.2 and 13.0 deg for
  A1_r15_x90y135g3 t1-t3, and 0.6 and 25.5 deg for A1_p35_x105y100g15 t1-t2. The putdown stands the sphere up by tilting
  the hand, and `letgo` levels the hand again only within its `--max-level`, so the next trial started tilted: three of
  today's five were not the planned grasp. The 11-19 trials started within 0.3-1.5 deg.
- **Fixes (11:1x):** levelling turns over 15 deg go in steps, each with its own force reference (`putdown`; `place`,
  which also stops now when a step stops). The release ramp skips steps the driver refuses. `hom_series` levels the hand
  before every recentre when z_P is over 2 deg from vertical: `pose_sweep.py clear`, `place --max-level 45`, then
  `approach --fine --fine-max 150`. The rows carry `ready_tilt_deg`, and the putdown args add `--touch-lat 3`. Done by
  hand at 11:07-11:10: the 36 deg levelling went in three 12 deg steps, the HEX-E at 4.8-5.4 N of 12. The series
  restarted at 11:12 from entry 4 (`cut -d" " -f4-`, A1_anchor_x70y110g0_t1).
- **11:19-11:30, A1_r3_x115y160g8_t1 and the reversed twist.** The anchor's three trials started at 1.1, 0.6 and
  0.5 deg, the series levelling the hand itself before t2 and t3 (15.5 and 16.1 deg, in two 7.8 deg steps at 3.3 N at
  most). On (115, 160) g8, t1 twisted to 74 deg and ended on the trial's drift limit: the sphere was 26.7 mm up and
  30.5 mm off in the grasp. The putdown refused the 100.6 mm descent (`--max-down 80`). The owner, at the bench, asked
  to play the twist backwards ("just for fun"). `lab/reverse_twist.py` streams a record's targets last-first at half
  speed, clipped to the safe range, the arm still. By 10 s the sphere's block had swung back near its start. In the
  last second the fingers opened to the trial's first target, and the sphere dropped about 100 mm and fell over. The
  owner stood it up: "ok yea that didn't work". t1's row was added by hand.
- **The owner's release rule (11:3x):** "it's fine to lower there, slowly release the fingers, and ok if not released
  fully to zero, just while it is collision free, and pray for a good reset". The series' putdown args add
  `--max-down 150`. When neither the path to zero nor the flex-back is clear, `letgo` now takes the fingers along
  the path to zero as far as it stays clear (`partial_zero`, in tenths), slowly. If no tenth is clear, the fingers stay
  open; either way the watched lift follows. Restarted at 11:31 from entry 8 (A1_r3_x115y160g8_t2).
- **11:3x-11:5x, jitter and 60 fps.** A1_r3_x115y160g8_t2 twisted to 65 deg; the sphere rose 20.6 mm and drifted
  26.0 mm. The putdown's levelling stopped, and the owner pried the sphere out and reset it. The owner: "the controls
  are way more jittery and cause a ton of drift". `analysis/20261120-command_jitter.py` finds the chatter in HOM's
  commanded targets (01e8e60fc); the workstation is diagnosing it offline, and HOM stays unchanged all day.
  - The owner's go (11:4x, by relay): "prioritize (within reason) the 60fps tracking". With the owner gone, the
    moving-block test of 11-13 (the block turned by hand) could not run. Instead: a 15 s static check on the standing
    sphere (`logs/20261120-tracker_modes/`). 1080p30 at 16.6 ms: 30.0 Hz, tracked 1.000, frame age 39 ms, sd 0.020-0.026
    mm and 0.036 deg. 540p60 at 5 ms: 59.5 Hz, tracked 1.000, 22 ms, sd 0.040-0.057 mm and 0.089 deg. The twists are
    the moving test.
  - `hom_series --cam-mode 960x540@60 --cam-exposure 50` sets HOM_CAM_MODE and HOM_CAM_EXPOSURE for every step
    (`hom_trial.cam_mode`; the recorder writes 960 x 540 at the tracker's rate), and every row carries `camera`.
  - When `ready` refuses an object put back too far from the hand, the series lifts clear and moves over it
    (`approach --fine`), then readies again.
  - Restarted at 11:5x from entry 9 (A1_r3_x115y160g8_t3) at 540p60.
- **12:0x-12:31, four more stops, each closed in code.** Stops took 78 % of the logged time (the workstation's
  `analysis/20261120-series_time_table.py`; a clean cycle is 142 s).
  - A released sphere standing at 1.7 deg was taken for toppled: the series read its tilt in a hand still tilted
    35 deg. `letgo` now prints the block's axis from vertical, and the series reads that (36adb1cfd).
  - Layout refused fingers left partly straight by the partial release. The series now straightens them (`zero`,
    checked) and lays out again (33ea534d0).
  - A descent stopped 65 mm up on 3.5 mm of slip in the grasp. A stop on the object's motion more than `--touch-window`
    above the expected touchdown now counts as slip, and the leg goes on, at most `--max-slips` 3 times (310276959).
  - letgo's lift levels z_P about the palm when the tilt is within `--max-level` (10). A 9.3 deg turn swung the thumb
    into the ball, which moved 3 mm, and the lift stopped. On the model, a straight lift took the thumb away (1.1 mm
    clear at the start, 9.4 mm at 20 mm up). The series' putdown args add `--max-level 0`, so the lift goes straight up
    and the hand is levelled before the next recentre, after the clear. Restarted at 12:31 at A1_p42_x55y75g10_t2.

## State on 2026-11-19 21:1x: the ranked layouts' series, unattended

The owner's go (relayed 18:1x): "if you finish analysis early, i give you the go ahead to just start running the
experiments", "we can do the top 10 x N trials". The plans are the twist ranking's
(`docs/experiments/20261119-twist_layout_ranking`, its `data/20261119-trial_plans.json`, commit 6e925ca45): ten of the
179 deployable layouts and the anchor (70, 110) g0, each with its prediction in `meta.ranking`, written before any
trial. Three sets, run in this order: the 11 plans with HOM's default forces, then the 10 with stage E's lower forces
(`_f.._w..` in the name; the anchor's first), then stage F's CEM plans (about 23:15).

- `lab/series_plan.py` keeps the list's rows by role and forces (`--forces default|refined|both`), shuffles the plans
  with a printed seed, repeats each three times in a row, writes `lab/series_entries.txt` and prints the hom_series
  command.
- `lab/hom_series.py --serve 'twist 40 28.8 0.56'` starts each plan's HOM server on the workstation
  (`hom_serve_budget.sh`) before its first trial and stops it after its last: one server at a time, 410 MB each.
- Before each trial it reads the station's servo diagnostics: torque on all nine, and with a servo at 55 C or more it
  waits, the hand idle after letgo, until all are below 40 C.
- The sweep tracks the thumb's DIP tag (24) through both cameras (`--no-thumb-tag` to skip). Each pose's line has the
  tag's place and normal over the hold against the hand model at the measured joints; `d_mm_2cam` uses the samples
  both cameras saw. One camera's depth scatters 2.5 mm (sd) along its view, both cameras' 0.3 mm. At the zero pose
  (21:00, (105, 70)) both cameras put the tag 2.6 mm and 2.1 deg from the model.
- The first pass, started 21:08 (seed 210748, ports 47102-47112): (65,110) g0, (70,90) g1, (115,70) g7, (105,60) g9,
  (60,135) g8, anchor (70,110) g0, (85,60) g14, (115,160) g8, (85,120) g15, (135,65) g0, (110,155) g3. The sweep's
  dry run skips "mixed B" at (105, 60) and "yaw +15" and "mixed A" at (85, 60) (fingers under 8 mm apart). Log
  `logs/20261116-hom_trials/20261119-series_rank_seed210748.log`, one line per trial in `...seed210748.jsonl`, the
  session's films in `logs/20261116-hom_trials/20261119-2108_session/`.
- **The lights went off between 21:09:36 and 21:10:32** (frames' mean grey 20 and 29 lit, 16 dark), and the first
  `ready` at (65, 110) got 0 poses of the sphere: the series stopped before any trial. They go off at 20:10-21:40 on
  every night recorded; schedule tracked work before about 21:00.
- **In the dark, the free-air sweeps at all 11 layouts** (`hom_series.py --sweep-only`, 21:13-21:32; log
  `...20261119-sweeps_dark_seed210748.log`). At (70, 90) the pose "curl 40/40" stalled the thumb's and index's MCPs
  6-7.5 deg short and the sweep stopped and took the fingers back: their DIP links met. The sweep's check then took
  each finger as three balls (tip pad, tip end, PIP joint) and had them 27 mm apart; the cad_v3 scene's collision
  spheres put them 0.6 mm into each other, and 8.6 mm apart at (65, 110), where the pose ran. The rest ran at a 30 mm
  margin on the balls (`--sweep-args='--ff-mm 30'`), skipping the converging poses. Since 21:3x `pose_sweep.py`'s
  finger gap uses the CAD spheres (s3_approach.hand_points) with the 8 mm margin; hom_setup's other checks still use
  the balls.
- **The own-pose term does not depend on the layout** (fingertip sensing page, Table 4): the 11-04/05 GP left a median
  of 5.9, 7.4 and 8.0 uT (thumb, index, middle) over the 11 layouts, with no trend; the (70, 110) refit was no better
  anywhere. The per-layout sweep is dropped from the series.
- **The morning start:** `lab/wait_for_lights.py` (pid in `pgrep -af "[w]ait_for_lights"`) checks the cameras from
  07:00 every 5 min and, after three lit checks with the sphere seen, runs hom_series on `lab/series_entries_morning.txt`
  without `--sweep`. The file is read when it fires, so a new order can replace it until then. Log
  `logs/20261116-hom_trials/20261120-morning_series_seed20261120.log`, one line per trial in
  `...20261120-series_morning_seed20261120.jsonl`; checks in `.../wait_for_lights.jsonl`. To cancel, kill the watcher's
  pid.
- **The morning's order** (the twist ranking's 86bc736b6, 21:5x; `lab/series_morning_order.json`, seed 20261120, 66
  trials on ports 47102-47123). Along the simulated twists, on the CAD spheres, the index's MCP link and the middle's
  PIP link touch at (135, 65) g0, (115, 70) g7, (105, 60) g9 and (85, 60) g14 and their stage E plans; on the hand's
  joints of 11-17 the same check read -2 to -5 mm in all six g13 trials at (105, 70), where the two met, so the hand
  runs 8-13 mm closer than the simulation for this pair. Block 1, the ranking test: 11 layouts (the seven others and
  four replacements, (90, 135) g3, (80, 130) g15, (95, 90) g14 and (75, 140) g5) in seeded random order, then the
  anchor's stage E plan, then the anchor's two CEM plans (stage F, 463cc988f: `cem_knots`, a four-knot force schedule,
  and `cem_full`, 1.40 N without the spin rows). Block 2: the four touching layouts, labelled `b2touch`. Block 3: the
  stage E plans of the non-touching layouts. Block 4: the other eleven CEM plans that keep the fingers apart on the CAD
  spheres (the six at the touching layouts left out), shuffled. 105 trials on ports 47102-47136; at about 4.5 min a
  trial the list runs past midday, and blocks 3 and 4 can be cut.
- **The morning's order, rebuilt 00:2x on 11-20 for the denser spread** (owner, relayed 23:2x: "we should shoot for higher
  density of evaluation (which is just more time, but we have to accept it) to see if we can first ascertain a rough
  correspondence, including simulation-predicted bad morphologies"; the twist ranking's 361206fa4). Block A first: A1,
  block 1's 11 layouts (3 trials) shuffled with 10 new layouts over the predicted range 0.25-0.50 (2 trials, all at
  least 10 mm apart on the CAD spheres); A2, 12 predicted-bad layouts (0.00-0.24), most touching on the CAD spheres
  (-5.3 to 8.2 mm), 2 trials each; A3, 3 layouts that dropped the sphere in simulation, 2 trials each. Then the anchor's
  stage E and CEM plans, block 2, block 3, block 4. 155 trials on 60 plans, ports 47102-47161; about 12 h at 4.5 min a
  trial, so cut from the end. A2 is expected to stall fingers against each other and may trip a servo's overload guard:
  the putdown then refuses (torque off) and the series stops with the object in the hand, for a person to recover
  (`hom_setup.py torque-on --go` with every MCP under 60 deg, then `putdown`). After a drop the series goes on when the
  sphere lands upright or lies near the table's centre (the pick-up), and stops when it rolls away.
- **The owner's call on the layouts without a feasible twist** (owner, relayed 23:5x: "we should not be filtering bad
  hands because of collisions from infeasible trajectories, we should be judging them by their feasible trajectories").
  The twist ranking searched all 16 ground-clear grasps of the 19 touching layouts (block 2, A2, A3; cf510eb97): none
  keeps the fingers 10 mm apart on the CAD spheres through its twist, so their feasible prediction is 0, and their
  trials test that. They start after A1, near 11:00. **To skip or pause them while the series runs**, write
  `lab/series_control.txt` (read before every trial; no file runs everything): a line `skip A2bad` (or `skip A3drop`,
  `skip b2touch`) skips those labels, a line `hold` pauses before the next trial until the line is removed.

## State on 2026-11-19 19:1x: series across layouts, with a lift and a free-air sweep at each

The owner's go (18:1x, relayed; 18:4x direct): run the ranked layouts' twist trials unattended, and interim trials.
Both objects stand in view on their stems (the sphere seen by camera 1918 only, tags 14-15; 5 tags in all). Servers on
the workstation run on its CPU (`hom_serve.sh`, `CUDA_VISIBLE_DEVICES=` empty), so they do not compete with GPU work.

- `lab/pose_sweep.py clear` lifts the palm (HEX-E guarded, at most 70 mm) until the fingers, straight and in the sweep's
  poses, clear the object's top by 25 mm. Letgo leaves the palm only 30 mm over the grasp's height with the straight
  fingers beside the object, so a gantry move needs it. `pose_sweep.py sweep` holds the same 12 free-air poses (the plan's
  open pose first) for 3 s each at a new layout, each checked on the hand model first, about 50 s; logs in
  `logs/20261119-pose_sweeps/`, scored by the study's `sim/20261119-pose_sweep_own_pose.py`.
- `lab/hom_series.py` now runs clear before a layout change and, with `--sweep`, the sweep after it; it skips the layout
  step when the station holds the plan at its mounts; it passes `--plan` to the trial (the fingertip estimator needs it).
- The first sweep (19:05, (70, 110), the arm parked high): 12 of 12 poses, joints within 1.5 deg except the middle MCP
  at negative angles (4.5 deg short at the open pose). The deployed own-pose GP leaves 7.2, 10.5 and 11.4 uT (thumb,
  index, middle), the (70, 110) refit 7.0, 7.2 and 6.9.
- Start of the evening (19:07-19:08): `place --clear 201.7` levelled the parked palm (3.8 deg) at its height; `approach
  --max-lat 400 --max-dz 400` moved 140 mm across and 200 mm down onto the sphere: READY at 0.12 mm.
- The layout step refuses when the fingers are more than 5 deg from both the zero and the open pose; the middle's MCP
  stops about 5 deg short of the open pose's -15 deg. After a stopped series, run `letgo` (not `zero`, which refuses
  round the object) before starting again.
- **Servers need `--torque-budget 0.56` at 7.5 V.** `hom_serve.sh` passes no budget, and the server's default, 0.84 in its
  5 V units, reset all nine servos 2 s into the first rehearsal trial (19:12): the sphere drifted 30 mm and stayed upright
  on the table, torque off. `logs/20261116-hom_setup/hom_serve_budget.sh` on the workstation takes the same arguments
  plus the budget. Recovery: `torque-on --go` (every MCP under 0), `zero` (the object 40 mm below the grasp's height
  after the trial's lift), then the series again.
- **Topples (19:2x).** g13 at (105, 70) over-rotated at budget 0.56 (72 deg, 9.4 mm rise, tilted 42 deg in the hand);
  the putdown's default levelling takes out at most 8 deg, so the block was not seated, and `letgo --tilt-ok` let the
  sphere topple (94 deg). The controlled-slip pick-up stood it up again in 5 min (19:28-19:33; plan at the station's
  layout, contacts 5 deg above the equator: joint margin 4.9 deg, 8.5-9.3 mm off the table; 85 -> 35 deg in the lift,
  4.3 deg after the slip; set down upright at 1.2 deg and the wrist turned back). For the series: `--putdown-args="--level-frac
  1 --level-max 35"` (the 11-17 recovery's levelling) and `--pickup` (hom_series runs that pick-up after a topple and
  goes on; any failed step stops it).

## State on 2026-11-17 19:30: the evening's commands, and the night's runs

The evening's results are on the session page
`docs/experiments/20261117-sr2_bench_session/20261117-sr2_hand_bench_session_7v5.html`
(https://claude.ai/artifact/JuQrKr5dzHEtGkzLv8XHSe). How to run what it added:

- **The pick-up of a lying sphere as one command** (owner, 18:1x: "hopefully this task can be automated for next runs,
  without losing this interpretability"). From this folder, with the sphere lying near the table's centre:

      lab/pickup_cycle.sh /home/will/workspace/hand/logs/20261116-hom_trials/<YYYYmmdd-HHMM>_pickup        (plan only)
      lab/pickup_cycle.sh /home/will/workspace/hand/logs/20261116-hom_trials/<YYYYmmdd-HHMM>_pickup go     (the run)

  It runs diag, plan (with the wrist turn), the plan's checks (table at touch >= 7.5 mm, joint margin >= 3.5 deg),
  grasp, lift, slip, putdown and unturn, prints each step's key lines and keeps each step's whole output in the
  folder. It stops at the first step that fails. The defaults are the 7.5 V settings (squeeze 5.3 deg, slip steps 0.35
  deg); at 5 V use `SQUEEZE=8 STEP=0.5 RESQUEEZE=1`. A turn over 45 deg refuses: lay the sphere with its stem across
  the hand's x axis.
- **Twist repeats as one command:** `lab/repeat_cycle.sh V N` with the HOM servers up (env `P`, `ARMS` as
  `port:tips:name` list, `DRIFT`). Each cycle: diag and the 50/40 C thermal rule, place twice, ready (`--max-lat 70`),
  the trial, the putdown levelled fully, a second putdown when the seat check refuses, the fingers straightened when a
  putdown fails and the path is clear. It stops at the first failure.
- **The release when the straight path to zero is refused** (owner, 18:5x): `letgo` and `putdown` now fall back to
  `flex_back_release`: MCPs and PIPs to `--flex-deg` with the yaws kept, ramped over `--flex-s` 3 s, then a rise, then
  the yaws to zero, each step checked on the hand's model. `--no-flex-back` keeps the old refusal.
- **Layouts.** g13 at (105, 70) drives the index and middle together (least gap on the model 1.9 mm at 7.5 V); run
  twists on g0 at (70, 110), `plans/20261117-film_x70_y110_twist_gr_mcpstop_plan.json` (servers 47051 tips off, 47052
  tips on). The press: `plans/20261117-press_x70_y110_lat0_replan_plan.json` (the workstation's open pose inside the MCP
  stops; the HOM server still loads `20261116-press_x70_y110_lat0_plan.json`).
- **The night of 11-17** (`docs/experiments/20261117-sr2_system_identification/lab/night_rest.sh`, its log
  `logs/20261117-night_rest.out`): S2 at the roll layout (85, 170), S1(b), twelve g0 twist repeats over four arms (the
  paper-rate filter's A/B), then the boards' random-pose sweep. `touch logs/20261117-night_rest/STOP` stops it between
  steps; `touch V/STOP` stops the repeats between cycles.

## State on 2026-11-17 17:22: the servos at 7.5 V

The owner, 16:1x: "i have approximately zeroed the servos (all). you can go ahead with the restart. i agree on the HOM
target clipping. please fix the estimator subprocess yes. i am now powering the servos on 7.5V ... please experiment
with this carefully, but see if we can improve stiffness/speed/force with these 50% higher voltage". The SCS0009 is
listed for 4-7.4 V; a coworker had held one at stall for 45 s at 7.5 V, and the servos' own input limit register reads
9.0 V. The owner watched the trials from 16:5x and left at about 17:15.

In short: at the same servo current as at 5 V, one trial without the fingertip contacts tracked 0.439 against 0.322-0.378
at 5 V, a hint and no more. At the 5 V offsets the 7.5 V supply browned out mid-squeeze. Free-air steps reach 90 % in 0.1 s.
The plan's open pose drives two MCPs into a mechanical stop, which puts a pad against the sphere.

- **The morning's fixes.** The restart at 16:2x left every joint within 0.29 deg; the index PIP read +0.3 deg again.
  Commit 0edf083d5: HOM's targets stay 5 deg from the ends of the pots' track (`sr2hand.control.servo_limits`: index PIP
  +1.78 deg or more, index MCP 60.04 or less, middle MCP 65.02, middle PIP 72.34; the server clips every target and
  `hom_trial.py` clips again before the station), and the fingertip estimator runs in a process of its own
  (`tip_contacts.TipProcess`; `--tips-thread` keeps the thread). In the live trials of 17:09 and 17:11 an estimate took
  16.8-18.8 ms (median) against 68-73 ms in the thread at 09:15-09:21. The loop's longest period fell from 37-44 ms to
  21-39 ms, and both trials passed the default 60 ms guard. In the 17:11 trial the clip held the index PIP at +1.8 deg.
- **How the servos protect themselves** (their registers, read on 11-06 at 5 V:
  `docs/experiments/20261106-sr2_servo_sysid/data/registers.txt`). The overload trips on PWM duty: over 80 % for the
  protection time (register value 100), then the output is cut to 20 %, and the unloading byte (0x20) turns torque off.
  Maximum torque 100 %, P 15, D 15. A stalled motor's torque at a given duty scales with the supply, so at 7.5 V the 80
  % trip sits near 1.5 times the 5 V torque. The same force needs about two thirds of the duty, the same current and the
  same heat. The same offsets give 1.5 times the force and 2.25 times the heat.
- **Free-air steps** (`docs/experiments/20261117-sr2_servo_7v5/lab/step_test.py`, one joint at a time, 20 deg, read back
  at 50 Hz; `data/20261117-step_responses.json` there). A step reached 90 % in 0.09 s at speed 80 and 0.10 s at speed 40.
  Peak slews ran 230-510 deg/s, and every joint ended within 1 deg of its target. The speed setting does not cap these
  moves. The 5 V yaw holds of 11-06 reached 90 % in 0.17-0.27 s, but their samples are 0.1 s apart and timed from a
  manual command, so they only bound the comparison.
- **The open pose pushes the sphere.** The g13 plan sends the index and middle MCPs to -15.0 and -13.7 deg. They stop
  against the mechanism at -11.4 and -10.0 deg, in free air too, the servos pushing at 21-22 % duty. There the index
  pad sits 2.6 mm and the middle's 4.7 mm from the taped sphere on the hand's model; on 11-16 the tags put these tips 3-7
  mm off the model. `ready` found the sphere 0.05-0.08 mm from the grasp's pose before the fingers opened and 1.8-2.1 mm
  after, and the owner saw a fingertip tilt it. The morning's 5 V trials started the same way (MCPs at -10.5 and -9.1
  deg). A pose that keeps both pads 7 mm off (`plans/20261117-..._g13_open9_plan.json`: MCPs at -9, PIPs at 37.2 and
  41.4) changed the grasp itself: HOM closes from wherever the fingers start, and the index closed 16 deg straighter.
  The fix is to re-plan g13's open pose in simulation within the MCPs' real range.
- **Brownout at the 5 V offsets.** At 16:55 (budget 0.84, the open9 start) all nine servos reset at 1.45 s, mid-squeeze,
  before the lift. The station's read-backs stopped for 0.5 s, and the pads fell from 3.9 and 3.1 N to 1.2 and 1.0 N.
  The weakened pinch slid off at the lift. At 7.5 V the same offsets draw 1.5 times the current: the gr_g4 failure of
  11-16 at 5 V again.
- **The comparison at budget 0.56** (the owner's choice at 17:0x: the summed servo current kept at the 5 V level), from
  the morning's open pose with the two MCPs sent to their stops, not past them (`plans/20261117-..._g13_mcpstop_plan.json`;
  the start pushed the sphere about 2 mm, so the trials ran from `ready`'s pose without its 1 mm gate):

  | Trial | Supply, budget (N m) | Tips | Ended | Tracking | Twist at the end (deg) | Drift (mm) |
  |---|---|---|---|---|---|---|
  | 09:15 | 5 V, 0.84 | off | schedule | 0.322 | 33.6 of 90.6 | 9.0 |
  | 09:21 | 5 V, 0.84 | on | schedule | 0.394 | 35.0 | 19.8 |
  | 17:09 | 7.5 V, 0.56 | off | schedule | 0.439 | 41.4 | 10.5 |
  | 17:11 | 7.5 V, 0.56 | on | drift guard at 5.28 s | 0.482 (2.0-5.28 s) | 74.2 (59.4 commanded) | 20.4 |

  The budget is in the server's 5 V units (kp 0.55 N m/rad); at 7.5 V, 0.56 there is about 0.84 N m of real torque.
  The same grasp at 5 V on 11-16 tracked 0.378 (39.6 deg). Each row is one trial, so the gain is a hint, not a result.
  At 17:11 the sphere turned ahead of the command, rose 9.1 mm and tilted about 40 deg in the grasp. HOM's targets stayed
  13 deg (median) from the thumb MCP, index MCP and middle yaw, against 19 deg at 5 V. That difference is the budget's
  scale (0.56/0.84), not stiffness; at 7.5 V, 13 deg is about the torque 19 deg gave at 5 V. No servo reset at 0.56. The
  rail sagged to 6.3-7.4 V while the hand held the sphere, and the overload flag showed on the thumb MCP, index MCP and
  middle yaw.
- **Heat.** Holding the tilted sphere through the long putdown of 17:12 warmed the thumb yaw servo to 50 C and the
  middle MCP to 48 C (limit 70). They were back to 30-32 C at 17:2x.
- **The 17:11 putdown.** The levelling turn (32.4 deg) stopped at its 12 s limit; the touchdown seated the block at 4.7
  deg. The release left the thumb's tip end under the ball with the hand tilted 35 deg near the table, and `letgo`
  refused to straighten the fingers. The way out was checked on the hand's model first: the thumb's PIP from 44.8 to 20
  deg, a 35 mm rise along z_P (1.19 N at most at the HEX-E), `letgo`, then the rise and the levelling in two steps.
- **No reply at 16:49.** The first trial received no target: the server answered all 106 states, and none reached
  corvus. The hand lifted empty at 1.6 s. Four dry runs at 16:51-16:52 received every target; their server records
  were deleted. Cause not found. A trial now stops when no target comes within 0.5 s, and its lift is cancelled
  (`hom_trial.py --first-target-s`).
- **The station closes idle connections.** The step test's first stream start, on a connection idle since its state
  read, failed with "remote end closed connection". Its client now reconnects once.
- **The tracker and the printed blocks.** At 17:06 the tracker fitted the sphere's map (tags 10-15) to extra printed
  blocks 460 mm away. With them out of view it found the sphere at once. The cylinder's tags (16-23) do not overlap the
  sphere's, so the cylinder may stay in view.

The rig at 17:22: the hand level (z_P 0.01 deg) with its palm at z 234 mm in B, over the upright sphere near the
table's centre. Fingers straight, torque on, 7.4-7.7 V. No HOM server runs (47047-47050 mine, 47045-47046 hand-4a's), and
the board splitter is stopped. Records: `runs/` and `runs_corvus/` from 20261117-1649 to -1711. Films, fingertip
records and the splitter's log are in corvus's `logs/20261116-hom_trials/20261117-1650_7v5/` and
`logs/20261117-three_finger_mux/20261117-163428-mux.bin`.

Next, in the owner's words where given: a map of the data paths, rates and clocks with each past failure on it, on
`docs/index.html` ("we can queue this up after the 7.5v experiments"); re-plan g13's open pose within the MCPs' range;
a supply strong enough for budget 0.84 at 7.5 V, or keep 0.56; repeat the trials (one per row so far); the cylinder in reach
("ideally you could experiment on both"); "picking up the cylinder from dropped via a HOM-style roll" (a plan from
simulation first); faster tag detection ("surely at this point we can accelerate the detections"); and the records'
contact forces and points for the pad and point contact models.

## State on 2026-11-17 10:06: the fingertip comparison stopped after three trials, the index PIP's reading wrapped

The owner, 09:0x from away: "are the lights back on? can we finish running some tests". At the tracker's fixed
exposure both cameras saw a lit lab (mean luma 138-140 of 255). The comparison of the next section started at 09:12 on
ground-clear grasp 13 at (105, 70). It used HOM servers 47041 (`--tips off`) and 47042 (`--tips on`) with the torque
budget at 0.84 N m and the drift guard at 20 mm; 47043 and 47044 served g12 for later. It stopped after three trials,
when the index PIP's position reading wrapped. The owner, 10:0x: stop for today.

| Trial | Tips | Ended | Tracking | Twist (deg) | Rise (mm) | Drift (mm) | Tilt (deg) |
|---|---|---|---|---|---|---|---|
| 09:15:23 | off | schedule | 0.322 | 33.6 | 1.9 | 9.0 | 13.1 |
| 09:17:52 | on | guard, 3.54 s | – | – | 1.8 | 2.5 | 6.1 |
| 09:21:12 | on | schedule | 0.394 | 35.0 | 4.0 | 19.8 | 35.4 |

Tracking and the twist at the task's end come from `scripts/hardware/hom_trial_tracking.py`, run on a scratch copy of
the records. Rise, drift and tilt are the largest over each record. The same grasp tracked 0.378 under the same budget
at 19:58 on 11-16. One pair cannot tell the two settings apart from the spread between trials.

- **The loop stalls.** Inside the trial's process the estimator took 57-73 ms per estimate (each trial's median;
  125-127 ms at the 95th percentile, 218 ms at most). The offline replays took 7.4 ms. The same Python process runs the
  tracker on two 1080p cameras and two video writers, and its 50 Hz loop stalled for 40-70 ms. The first tips-on trial
  stopped at 3.54 s on the guard for three missing targets (60 ms), while HOM's server stepped in 3-7 ms. The third
  trial ran with `--miss 6` (120 ms). The estimates reached the server 43 ms old (median, first trial). Next: run the
  estimator, or the films, in a process of its own.
- **The index PIP's reading wrapped.** In the tips-on trial of 09:21, HOM drove the index PIP to -2.9 deg, against the
  station's lower limit of -2.93 deg. During the putdown that followed, its reading jumped to +184 to +189 deg and
  stayed there. The driver's table (`manta_hand/servos.py`, FINGER_JOINTS) puts that limit at the servo's -149.7 deg,
  0.3 deg inside the end of its 300 deg range, so a small overshoot leaves the potentiometer's track. The index and middle
  MCPs' flexion limits and the middle PIP's flexion limit sit at +149.7 deg the same way, which fits the MCPs' known jump
  past 62-68 deg. The station refuses any frame or pose with the index PIP outside [-2.93, 92] deg, so the
  interpolated release failed. The servo cannot see its position and pushes against the stop at its overload protection
  (status 32). Next: free the PIP with the station stopped (a raw-bus nudge into flexion), then the motionless restart.
  Clip HOM's targets a few degrees inside these four limits, and just inside every other limit: the press of 11-16
  lost frames at the middle PIP's -16.11 deg, a range check, not a wrap (servo +56.6 deg).
- **The putdown of 09:21.** The sphere had tilted 39 deg in the grasp. The touchdown found the block 30.7 deg from
  vertical, not seated. After a 25 mm lift, `putdown --level-frac 1 --level-max 35` turned the hand 31 deg, stopped by
  its 12 s move limit near the end, and seated the block at 3.4 deg. The release had to go around the index PIP: the
  plan's open pose with that PIP's goal held at -2.9 deg, as one goal at speed 40. The sphere toppled as the fingers
  opened and rolled about 190 mm. While the hand held it, the thumb's MCP and the middle's yaw also showed the overload
  flag, and the index yaw's servo warmed to 39 deg C; both eased after the release.
- **The first ready refused.** The sphere sat 59.6 mm from the hand's nominal pose: the 20:10 putdown had left it at
  the table's centre. `approach --fine --fine-max 60 --max-lat 60 --fingers-straight` moved the hand over it at its
  height.

The rig at 09:29: the hand level (z_P 0.02 deg), its palm at z 254 mm in B over the table's centre. The fingers are at
the open pose with torque on, and the index PIP is blind against its stop (29 deg C). The sphere lies on its side about
190 mm from the hand. HOM servers 47041-47044 and the board splitter are stopped. Records: `runs_corvus/20261117-*` and
`runs/20261117-*`. Films and fingertip records are in corvus's `logs/20261116-hom_trials/20261117-0912_tips_eval/`.

## Trials with the fingertips (from 2026-11-17)

Every HOM trial now records the three fingertip boards and sends their contact estimates to the HOM server. Whether HOM
uses them is set by the server's `--tips` switch. On the pick-up of 2026-11-16 the estimates placed the pinch's contacts
0.3 and 1.6 mm (medians) outside the tracked sphere's surface. On the press of 13:19 they found the thumb at 2.8 N and
the index at 2.4 N. Fed the press's recorded states, HOM with `--tips on` moved its targets by 0.86 deg (median) and 18
deg at most. With `--tips off` the targets were those of 11-16 to 0.017 deg. The owner asked for this at 23:4x on 11-16:
"lets make sure the next time we run trials we have the fingertip sensing going. ideally with pose tracking, closed loop
control too. that can be churned away on overnight to be ready for evaluation".

Terms and metrics:

- A board is one fingertip's ReSkin 5X with five magnetometer chips, read through one I2C mux:

  | Finger | Tip | Mux channel |
  |---|---|---|
  | thumb | v3 ID 1 | 3 |
  | index | v3 ID 3 | 1 |
  | middle | v3 ID 2 | 0 |

- The splitter is `src/sr2hand/driver/reskin5x/host/reskin5x_mux.py`. It reads the mux at 160.6 records/s and logs every
  record to `logs/<date>-three_finger_mux/<start>-mux.bin`. Each board's records also go to a pseudo-terminal at
  `/tmp/reskin5x_mux/ch<channel>`, from which the trial reads them.
- The rest is a board's mean reading over 1 s with the fingers still at the trial's open pose, before t = 0.
- A contact estimate is a point on the pad (mm, in the palm frame P) and the normal force there (N). A finger counts as
  touching from 0.25 N, the boards' floor.
- The gap is the distance from an estimated contact point to the tracked object's surface in P, negative inside the
  object. The apex gap is the same for the pad's apex, where the hand's model puts the contact without the boards.
- The data age is the time from a board's newest record to the state that sends its estimate.

The estimator, `src/sr2hand/driver/reskin5x/host/tip_contacts.py`, runs the contact viewer's estimate on each board
(`wrench_viewer.py --board-gp`). The chips' readings are corrected for their temperature, and the change from the rest
is averaged over 50 ms. Whether there is a contact is decided by the physical model of the tip's magnets. That model is
the tip's calibration of 11-02, kept when it fits the rest at least as well as a new fit. The point and the force then
come from the UR5 board GP of the v3 tips. One term is new. A board's reading changes with its own finger's pose by up
to 166 uT with nothing touching, the reading of a 5.8 N press (crosstalk sweep of 11-04/05). A GP on the finger's three
joint angles, fitted to that sweep, predicts the change between the rest's pose and the present pose. That change is
taken off first (`data/20261117-tip_contacts/20261117-own_pose_gp.npz`; `--no-own-pose` leaves it out). The point goes
from the pad frame to the hand's tip frame (x_tip = s (x + 2.05), y_tip = -s y, z_tip = 10.86 - z, with s the finger's
face sign). The tool chain then places it in P at the measured joints and the plan's mounts.

On corvus, from the repository, the first command shows the splitter's state; a trial starts the splitter when it is
not running. The second, at the open pose after `ready`, takes the rests and prints a few seconds of estimates. Nothing
moves, and no contact should show:

    /usr/bin/python3 src/sr2hand/driver/reskin5x/host/tip_contacts.py logger --start
    /usr/bin/python3 src/sr2hand/driver/reskin5x/host/tip_contacts.py rest --plan PLAN --station http://10.99.99.2:8765

The HOM server on the workstation, with `--tips off` (the default) or `--tips on`:

    .venv/bin/python -m sr2hand.control.hom_server serve --task twist --d 40 --plant measured --torque-budget 0.84 \
        --plan PLAN --listen PORT --reply CORVUS_TAILSCALE_IP:47001 \
        --log-dir docs/experiments/20261115-hom_bench_trials/runs --tips on

The trial, with the fingertips unless `--no-tips`; `--plan` places their contacts in P. The cycle passes `--plan` and
`TIP_FLAGS`:

    /usr/bin/python3 lab/hom_trial.py trial --task twist --controller 100.65.73.59:PORT --object sphere --map MAP \
        --plan PLAN --station http://10.99.99.2:8765 --live --arm --ft --out-dir OUT
    lab/trial_cycle.sh PLAN PORT LABEL [MAX_DRIFT_MM]                  # TIP_FLAGS=--no-tips: without the boards

A trial attaches to the splitter or starts it. It refuses to start unless all three boards stream at 100 records/s or
more. It then prints "fingertip boards: ..." and `trial_cycle.sh` stops with exit 4, the hand at the open pose. Every
record from 2 s before t = 0 goes to `OUT/<stamp>_<task>_tip_<finger>.bin`: host time (f8, the trial's clock), then 5 x
(t x y z) (f4), read by `tip_contacts.read_stream`. After the rests the estimator runs in a thread of its own, every 20
ms at most. Each state sends the newest estimate in StateMsg.cop, placed with the joints that state sends. The trial's
record has, per period, "tips" (the point, normal and force in P, the pad-frame point, the GP's weight) and
"tips_age_ms". Its header has the rests, its end line the estimator's timing. The server's record keeps what arrived
under "contacts" and each finger's contact frame as HOM took it under "gcf" (source and distance). `pinch_pickup.py`'s
grasp, lift and slip read the boards the same way (`--no-tips` without them). They print the field change of 11-16 and
the contacts beside it, and write the records beside each step's telemetry. Only one program may read the
pseudo-terminals at a time: the viewer must not run on them during a trial.

The server's switch:

- `--tips off` (default): the contacts are recorded, and HOM takes every contact frame from witness points on its model,
  as on 11-16.
- `--tips on`: a finger pressing with 0.25 N or more takes its contact frame from its centre of pressure
  (hom.Mirror.gcf_from_cop), as hom_tasks' tip variant does in simulation.
- `--tips raw`: the plan's own floor. `sil` runs its server so, since its simulated hand sends the plant's contacts. On
  the press plan the UDP run's targets equalled the in-process loop's (0.0 deg) and the float64 run's to 0.0008 deg.

The checks without hardware are `lab/tips_offline_check.py` (press, pickup, hom) and `lab/tips_live_check.py`, with
their summaries in `data/20261117-tip_contacts/`. The press with the pinch at the axis (13:19:27) through the
estimator, over 301 periods (10th, 50th and 90th percentiles):

| Finger | Periods with a contact | First (s) | Force (N) | Gap (mm) | Apex gap (mm) |
|---|---|---|---|---|---|
| thumb | 280 | 0.44 | 2.48, 2.78, 2.94 | -5.1, -4.0, -2.9 | 0.3, 1.3, 2.8 |
| index | 259 | 0.86 | 2.23, 2.38, 2.49 | -9.8, -8.7, -7.8 | -9.2, -8.2, -7.3 |
| middle | 137 | 1.68 | 0.43, 0.53, 0.64 | 2.4, 3.3, 5.0 | 4.6, 5.5, 6.4 |

The first pick-up (15:43-15:46: grasp, lift and slip), over 1,174 periods at 20 Hz, times from the close's start:

| Finger | Periods with a contact | First (s) | Force (N) | Gap (mm) | Apex gap (mm) |
|---|---|---|---|---|---|
| thumb | 888 | 4.06 | 0.91, 1.29, 1.46 | 0.0, 0.3, 1.0 | 1.6, 1.9, 2.5 |
| index | 0 | | | | |
| middle | 996 | 4.16 | 0.67, 1.26, 1.40 | 1.1, 1.6, 2.7 | 3.2, 3.7, 4.5 |

In the pick-up the contacts lay nearer the tracked sphere than the apex did, on the pads' distal halves. The index,
curled away, showed none. The pinch's median forces were 1.37 and 1.34 N in the lift (thumb, middle). They fell to 1.11
and 0.98 N as the slip lowered the squeeze. In the press the index's contact lay 8.7 mm inside the tracked cylinder, and
its apex 8.2 mm. There the hand's model and the tracked pose disagree by about 8 mm. The thumb's contact lay 5 mm deeper
than its apex. The own-pose term changed the middle finger. During the approach, from 0.5 to 1.3 s, its apex stayed 4-10
mm from the cylinder. Its board changed by 30 uT, and the term's prediction was 35 uT. The explained chi-square fell
from 18 to 0.8 (medians). Without the term the middle counted as touching from 1.20 s, at 1.34 N (median); with it, from
1.68 s, at 0.53 N. The thumb's and the index's median forces changed by 0.14 and 0.06 N.

The timing was measured on one core, under corvus's Python 3.10 with numpy 1.26.4 and scipy 1.12.0. The estimate for
the three boards took 7.4 ms (median) and 8.5 ms (95th percentile) in the replays. `tips_live_check.py` ran
`hom_trial.py trial` on a played log with the estimator's thread, a stand-in station and a local server. There:

| Quantity (ms) | Median | 95th percentile | Most |
|---|---|---|---|
| Link period | 20.0 | 20.3 | 22.9 |
| Estimate | 7.5 | 8.8 | 9.6 |
| Estimate's age at its state | 7.8 | 17.4 | 20.2 |
| Data age | 41 | 65 | |

Most of the data age is the splitter's 50 ms batches. HOM's step time was 4.0 ms on corvus on 11-13 and 3.1 ms here
on 11-16. An estimate there should therefore take about 10 ms.

HOM was then fed the press's recorded states again (`hom_trial.py replay`, three servers in turn):

| Server | Contacts sent | Compared with | Most (deg) | Median (deg) |
|---|---|---|---|---|
| `--tips off` | none | the server record of 11-16 | 0.017 | 0.0001 |
| `--tips off` | the estimates | `--tips off` without them | 0.000 | 0.000 |
| `--tips on` | the estimates | `--tips off` with them | 18.0 | 0.86 |

With `--tips on` the contact frame came from the centre of pressure in 280, 259 and 137 of the 301 periods (thumb,
index, middle). The middle finger's MCP and PIP targets moved most, by 3.8 and 4.7 deg RMS. In the press, the hold
and the release they moved up to 10.6, 14.3 and 18.0 deg. The thumb's moved 1.1 deg RMS at most. HOM's step time was
2.7 ms either way.

The evaluation for the next session compares HOM with and without the fingertips on the same grasps:

1. Grasps: the twists g13 and g12 at (105, 70), and the press with the pinch at the axis. Both twists ran their full
   8.3 s under the torque budget on 11-16. The press waits until HOM's targets are clipped to the station's range
   (Future work of 23:45).
2. Two trials per grasp and setting, in the order off, on, on, off, from the same `ready` and the same server plan.
3. Metrics: the tracking score (`scripts/hardware/hom_trial_tracking.py`) and the object's drift. Beside them, the
   contact traces: each finger's force and point in P over time, from the trial's "tips". They are compared with HOM's
   commanded force and the server's "gcf" sources.
4. Each grasp's change in tracking score between the settings is compared with the gap between its two trials at one
   setting. A change within that gap shows no gain from the contacts as HOM now uses them.

Future work:

- The index's 8 mm: measure the index's and middle's tips with their DIP tags at the press's touch pose
  (`lab/fingertip_tag_check.py`). A tag within 2 mm of the model puts the error in the cylinder's tracked pose.
- Forces on the hand: press each tip against the HEX-E at the open pose and compare the estimate's normal force with
  the HEX-E's. Errors over 20 % at 1-3 N falsify the UR5 GP's transfer to the hand.

## State on 2026-11-16 23:45: the press against simulation

With HOM in the loop, no simulated contact model reproduced both press trials of 13:12 and 13:19. On the trial with the
pinch at the axis, the station had refused 153 of 301 frames, so the commands that reached the hand's servos were
different from HOM's. Replayed open loop, those commands pressed the simulated cylinder as the hand did when the
tangential contact was soft. Point contact at friction 0.5 stayed within 2.6 and 3.7 deg RMS of the two press-angle
traces. In the generalist's replay it let the cylinder fall from 5.14 s; the hand's fell from 5.05 s. The owner asked
for this comparison at 23:0x ("sure, send off a background autopilot run to churn on the press task calibration").

The press angle is the turn of the cylinder's axis about x_P from its first tracked pose, in degrees, positive in the
sense of HOM's command theta_ref. The command rises from 0 to 20 deg between 2 and 3 s and stays there for 1 s. It
returns to 0 at 5 s and stays there to 5.5 s. The tracking score is max(0, 1 - err/ref) over 2.0-5.5 s, as
`scripts/hardware/hom_trial_tracking.py` computes it. Here err is the RMS angle between the axis and its 2.0 s pose
turned by -theta_ref about x_P, and ref is the RMS of theta_ref, 13.8 deg. The trace RMS is the RMS difference between a
simulated press angle and the hand's, taken at the hand's capture times from 0 to 5.5 s. Soft and stiff tangential
contact are MuJoCo's impratio 1 (the twists' calibration) and 100 (the scenes'). Noslip is MuJoCo's noslip solver at 30
iterations, which removes the soft contact's creep.

The simulated object is the one on the hand (`logs/20261116-sim_real_reversal/press_stem.py`): the 25 mm cylinder with
the sphere's stem and tag block glued under its middle, 30.1 g in all as weighed. Its centre of mass lies 10.3 mm below
the axis and 38.2 mm from the pinch line toward the middle finger. Gravity therefore turns it about that line with
11.3 mN m. It stands on its block on a table, which is removed at 1.6 s, when the arm lifted the hand. The servo model
is the twists' (`rev_clutch2.py --stiff`): gears that the load cannot backdrive, kp 0.55, the index's and middle's yaw
at 0.35. Each of eight contact variants (pads or point contact, impratio 1 or 100, noslip or not) ran in two ways
(`press_predict.py`). In the closed loop HOM drove the simulated hand as the bench server drove the real one, seeds 0-2.
In the replay the targets that reached the hand's servos were applied open loop from the hand's start pose. The design
pipeline's model (HOM and servo at 0.72, stiff pads, the bare cylinder on its post) ran for reference. Rows, scripts and
the comparison are in `data/20261116-press/` (`scripts/hardware/hom_press_compare.py`), the figure is
`media/20261116-press_sim_vs_hand.png`.

| Model | Contact | Track, pinch at the axis (hand 0.452) | Track, generalist (hand 0.469) | Trace RMS, axis / generalist (deg) |
|---|---|---|---|---|
| Design pipeline | pads, stiff | 0.235 | 0.905 | 3.6 / 10.3 |
| Closed loop | pads, soft | 0, dropped at the lift | 0, fell at 4.5-5.1 s | 65.9 / 25.9 |
| Closed loop | point, soft | 0, dropped at the lift | 0.421, fell at 5.0 and 5.4 s in 2 of 3 | 63.8 / 12.3 |
| Closed loop | stiff or noslip, 6 variants | 0.09-0.18 | 0.73-0.77 | 4.0-5.5 / 10.4-11.4 |
| Replay | pads, soft | 0.247 | 0.593 | 2.3 / 8.6 |
| Replay | point, soft | 0.303 | 0.645 | 3.3 / 6.6 |
| Replay | stiff or noslip, 6 variants | 0.06-0.12 | 0.74-0.83 | 4.1-5.3 / 8.1-8.7 |
| Replay, friction 0.5 | pads, soft | 0.244 | 0.443 | 2.2 / 9.2 |
| Replay, friction 0.5 | point, soft | 0.300 | 0.238, fell at 5.14 s | 2.6 / 3.7 |

The station's refusals shaped the trial with the pinch at the axis. HOM sets a joint at its limit in the model's units,
-16.110026 deg for the middle PIP. The station's range ends at -16.11 deg, and it refuses the whole frame. In that trial
153 of 301 targets lay at the limit, from 0.94 s on. Every one failed: 102 were refused and 51 were lost with a dropped
connection. Of the other 148, 147 arrived. HOM then ran on its own reference joints in 69 of the task's 175 periods, and
its middle PIP target cycled: three periods at the limit, two near 0 deg. The servos received only the second kind. The
hand's middle PIP therefore rose from -15 deg to about 0 deg between 1.2 and 1.6 s. The cylinder was then pressed
9.4 deg at most. In simulation HOM's target for that PIP stayed at -16 to -14 deg, just short of the limit. With the
station's rule applied, 0 to 3 frames were refused per run (78 in one run of 48). The closed loop gave the same scores
with and without it. The generalist's trial reached the limit only at 5.10 s, after its press.

On the generalist the closed loop commanded the hand's squeeze. HOM's target exceeded the measured joint by
13.5-13.9 deg at the thumb's MCP, against 11.6 deg on the hand. At the index's MCP the excess was 10.8-11.5 deg, against
10.9 deg. Soft tangential contact reproduced the fall after the release: the pads in all three seeds at 4.5-5.1 s, point
contact in two seeds at 5.0 and 5.4 s. The hand's cylinder was 5 mm low at 5.05 s. With stiff or noslip contact the
cylinder was held and pressed 19-20 deg, where the hand reached 15.8 deg. Its tracking score was 0.73-0.77. On the pinch
at the axis the cylinder was lost with soft contact once the table went, in all six runs. On the pads it swung 76 deg
down about the pinch line; under point contact it fell. With stiff contact it was held and pressed 1-4 deg, where the
hand reached 9.4 deg. The middle MCP's target exceeded its angle by 0-8 deg in simulation and by 20.8 deg on the hand.

In the replay the simulated joints stayed within 6 deg RMS of the hand's at every finger, the middle's farthest at
1.7-5.9 deg. With soft tangential contact the simulated press on the pinch at the axis stayed within 2.2-3.3 deg RMS of
the hand's. At friction 1.0 the stiff and noslip variants turned it at most 1.3 deg from its 2.0 s pose and missed the
hand's trace by 4.1-5.3 deg RMS. At friction 0.5 the generalist's cylinder fell under point contact after the release,
from 5.14 s. Its press angle at 5.5 s was -34.8 deg, the hand's -32.9 deg. No soft replay brought the pinch at the axis
back with the command during the release: it stayed at 8-11 deg at 5.5 s, the hand's at 3.6 deg. At friction 0.5 the
closed loop with soft contact dropped both cylinders at the lift, which the hand did not. The pick-up hang of 21:5x
favoured the opposite setting: the hanging sphere was held only by pads with creep-free stiff contact.

The close differed before any pressing. On the hand the closing fingers slid the cylinder 8.4 and 9.5 mm toward the
thumb and turned its axis 17 and 22 deg about the vertical before the lift. In the replays the same targets slid it
1.3-4.5 mm and turned it 4-8 deg. Lowering the friction of the table, stem and block from 1.0 to 0.35 changed the slide
by at most 0.7 mm.

Future work:

-  Clip HOM's targets exactly to the station's servo range, in `HomBench.step` (it clips only beyond 1e-3 deg) or in
  `lab/hom_trial.py` before posting. Then repeat the press with the pinch at the axis. The calibrated closed loop with
  soft contact predicts that the cylinder swings down at the lift (six of six runs, `press_predict.py run`). A cylinder
  that stays level on the hand falsifies that contact model for the pinch's hold.
-  Look at the two trials' films on corvus (`logs/20261116-hom_trials/20261116-1258_session/20261116-131239_*` and
  `20261116-131927_*`) for the 9 mm slide and 20 deg turn during the close. If they are real, measure the index's and
  middle's fingertips at the touch pose; their DIP tags sat 3-7 mm from the model on 11-16.
-  Measure the pads' friction on the cylinder's grip tape (UR5 and HEX-E drag). Friction 0.5 fits the replays best but
  drops both cylinders at the lift in the closed loop, where at friction 1.0 the generalist's was kept.

## State on 2026-11-16 20:10: blind twists under a torque budget, the sphere carried to the table's centre

The owner, 19:5x: "i think we can try again with the capped targets. we should also re-center the sphere on the table.
let's try to do this generally in each trial reset"; "im not at the bench, but you've run autonomously without me there
before". hand-4a ran four trials at (105, 70) from the workstation through corvus; the owner was not at the bench. The
rig at the end: the sphere upright on its block at the table's centre, the hand level and empty over it, station homed,
torque on; HOM servers 47035-47038 (`--torque-budget 0.84`) still up on the workstation.

- **Why the blind gr_g4 trials lost power.** The calibration was in the simulator only: the three servers of 13:03,
  16:12 and 16:47 ran the same plant (`kp0.55_kv0_fr0.18_fl0_dp0.011`) and the same HOM forces. gr_g4 loads more joints
  at once. In the lift its estimated servo torques (0.55 N m/rad past a 0.59 deg dead band, 0.18 N m at most) summed to
  a median 0.96 N m, against 0.84 N m for the 13:03 grasp, which held. The servos read 5.0-5.3 V at rest and 4.7-4.9 V
  holding a grasp (the new diagnostic below), the bottom of the SCS0009's 4-7.4 V range.
- **The budget.** `hom_bench.torque_budget`, served with `hom_server serve --torque-budget 0.84`: the nine targets'
  offsets past the dead band are scaled by one factor whenever their estimated torques sum past 0.84 N m. The
  predictions were redone with it and committed before any trial (`data/20261116-blind_twists/20261116-predictions_budget084.json`,
  19:51). No trial lost power.

| grasp | predicted track | hand track (from 2.0 s) | ended | budget active |
|---|---|---|---|---|
| gr_g13 | 0.30 | 0.378 | full run, 39.6 deg of 90.6 | 207 of 416 periods, scale >= 0.85 |
| gr_g12 | 0.33 | 0.326 | full run, 31.3 deg | 80 of 416, >= 0.92 |
| gr_g4 | 0.43 | 0.379 to 5.98 s | drift guard, 20.0 mm at 6.0 s | 224 of 300, >= 0.76 |
| gr_g8 | 0.10 | 0.00 to 5.82 s | drift guard, 21.2 mm at 5.8 s | 173 of 291, >= 0.76 |

  Records: `runs/20261116-1958*`, `20261116-2000*`, `20261116-2003*`, `20261116-2006*` and `runs_corvus/`;
  `data/20261116-trial_tracking.json`; films in corvus's `logs/20261116-hom_trials/20261116-budget_twists/`. gr_g4 and
  gr_g8 slid sideways in the hand about 1 s into the twist ramp; the simulation's summary has the rise (8.1 and 0.7 mm),
  not the lateral slip.
- **Servo diagnostics on the CB1** (owner's go, installed 19:45, backups in `~/backups/20261116-servo_diagnostics/`;
  `cb1/20261116-servo_diagnostics.patch`): `POST /api/v1/servos/diagnostics` reads each servo's voltage, temperature,
  torque_enable and status once. `lab/hom_setup.py diag` calls it, and `hom_trial.py` calls it once after a station
  error (gr_g8's dropped HTTP connection: no servo rebooted, 4.7-4.9 V). The station was restarted motionless at 19:45
  by hand (corvus's key does not reach the CB1 non-interactively, so `hom_setup.py torque-on --go` failed at its scp).
- **The sphere at the table's centre.** `putdown` and `pinch_pickup.py setdown` carry the held object over
  `TABLE_CENTRE_B` (-150, -375) mm in B before the descent: up 30 mm, then x and y at 20 mm/s, stopped by the HEX-E or the
  object moving in the hand. The 19:58 trial picked the sphere up from (52.4, -316.9), where the cameras saw 2 tags and
  `ready`'s descent stopped three times on the tracker's jumps (passed with `--watch-mm 6 --watch-deg 15`); its putdown
  carried it 207 mm, to 1.0 mm from the centre, where both cameras see 4 tags. With 4 tags the guards kept their
  defaults, except `putdown --touch-lat 3`: at 1.5 mm a 1.79 mm jump stopped one fast descent 55 mm over the table.
  gr_g8's putdown first refused to release the sphere at 23.8 deg from vertical; `place --clear 25 --no-level` lifted it
  and `putdown --level-frac 1 --level-max 26` set it down at 0.4 deg.

## State on 2026-11-16 17:01: blind twists stopped, the sphere upright

The owner, 17:0x: "we can stop after that. i think we have enough data." The rig: the sphere upright on its block
(1.6 deg), the hand level at its usual heading over it, fingers straight, torque on; the fingertip-board recorder
stopped (logs/20261116-three_finger_mux/20261116-154014-mux.bin, 193 MB, not in git); HOM servers 47014-47017 and
47025-47027 still up on the workstation.

- **The blind twist tests produced no result.** hand-4a predicted four grasps at (105, 70) before any ran
  (`data/20261116-blind_twists/20261116-predictions.json`). gr_g4 ran twice and dropped the sphere both times
  (`runs_corvus/20261116-161252_*` and `-164726_*`). Mid-lift, a sync write of the goal positions timed out on the
  servo bus (the CB1's web.log: `RuntimeError: Timeout error` in `stream_frame`), and all nine servos then read torque
  0 with nothing commanding it. It looks like a supply dip resetting them together. About six servos sat at or near
  their 0.18 N m limit through the lift, with larger force offsets than in the 13:03 ground-clear run. The station's
  telemetry has no supply voltage. gr_g12, gr_g8 and g7 did not run.
- **The open poses of gr_g4, gr_g12 and gr_g8** put a flexion MCP at its -15 deg limit. The middle's stopped at -9.7
  deg, and its pad pushed the sphere 4.5 mm (16:08). The corvus copies `plans/..._open9_plan.json` hold those MCPs at
  -9, with every pad at least 6.5 mm from the sphere. hand-4a served them on 47025-47027.
- **Two more pick-ups by controlled slip.** 16:31-16:38: translation only with the stem 33 deg off the pinch axis.
  The hang kept 26 deg locked along the axis, and the putdown's levelling turned the hand 26 deg to stand it up (film
  `media/20261116-pinch_pickup_reorient_tilted.mp4`). The owner then: "next time we can just rotate the wrist first."
  16:49-17:01: `plan --turn` turned the wrist -32 deg first. Lift to 23.9 deg, slip to 5.8 deg. With the wrist turned
  the cameras saw only 2 of the block's tags, so the new `setdown` put it down on the tracker and the HEX-E backstop
  (block seated at +0.03 mm). Then `letgo` (upright at 1.6 deg) and `unturn` (up 35 mm, turned back +32 deg).

## State on 2026-11-16 15:51: pick-up by controlled slip

The owner, about 15:20: "carefully try to pick up the sphere and run a controlled slip to have it rotate to more level, and
try placing it? would be a very interesting skill to build. controlled friction is the name of the game after all."
`lab/pinch_pickup.py` does it for the sphere lying on its side. The film is
`media/20261116-pinch_pickup_controlled_slip.mp4` (27.7 s, both cameras, people blurred).

- **The pinch.** The thumb against the middle (or the index) along x_P, the pads' normals along x_P, the contacts
  `--lat-deg` (3) above the ball's equator. A finger's yaw turns it about x_P, so every flexion plane contains x_P. The
  object lying across x_P therefore needs no wrist turn. In the pinch the object can turn only about the line through
  the two contacts. The third finger curls away on the side opposite the block. `plan` solves the joints on the hand's
  model with the v4b pads as half-ellipsoids. It checks the table (8.4 mm at touch on the conservative tip-end ball),
  the stem and block (22 mm), and the ball on the way down (7.8 mm). It writes the plan file that putdown and letgo
  read.
- **The run, 15:43-15:47.** `grasp`: open pose, 106 mm sideways, 70 mm down in two legs, pinch closed. Each arm move
  stops on 3 N at the HEX-E, the object moving 1.5 mm in B, or a finger joint pushed 4 deg. The sphere moved 1.3 mm
  over the close. `lift`: 64 mm up at 3 mm/s. The block dragged on the table and the object stood up from 85 to 42
  deg, then hung there on the pads' torsional friction. `slip`: the PIPs' squeeze lowered from 8 deg in 0.5 deg steps.
  At 4 deg the object turned from 42 to 13 deg, and at 0 deg it reached 6.9 deg. `hom_setup.py putdown --plan
  plans/20261116-pinch_pickup_thumb_middle_plan.json --level-max 0` gave a tracked touchdown with the block seated at
  6.7 deg. The object settled upright at 0.4 deg on the release.
- **The gears do not backdrive.** From the slip's log (`data/20261116-pinch_pickup/`): while the squeeze fell from 8 to
  4.5 deg, the PIPs' loads fell in proportion to their error, about 60 units a degree (300 is about 30 % PWM). The
  middle's load reached 0 inside the servo's dead band. The pads' fields stayed at 35-37 and 42-44 uT, and the joints
  did not move. The gear held the grip with no motor torque. Each yield was a servo reversal: once the command lay
  about 0.5 deg below the measured angle, that finger stepped open by 0.9-1.8 deg. The contact never backdrove a joint.
  This fits a gear train about 40 % efficient forward and self-locking backward. The 2.5 times larger free-air yaw sag
  points the same way. So does the sphere's failure to turn the index and middle yaws in the twists.
- **Logging.** The fingertip boards come from `reskin5x_mux.py` under `/usr/bin/python3` (miniconda's Python lacks
  numpy on corvus). In its log the boards run ch0 middle, ch1 index, ch3 thumb. `pinch_pickup.py` reads that log
  live: each board's field change from a rest taken just before the close. Its streams, at 20 Hz, hold the joints,
  servo loads, tracked object, HEX-E force and the pads.

    /usr/bin/python3 lab/pinch_pickup.py plan  --map maps/20261116-sphere_fitted_map_O.json --gap 8
    /usr/bin/python3 lab/pinch_pickup.py grasp --plan PLAN --map MAP --go --video-dir DIR
    /usr/bin/python3 lab/pinch_pickup.py lift  --plan PLAN --map MAP --go --video-dir DIR
    /usr/bin/python3 lab/pinch_pickup.py slip  --plan PLAN --map MAP --go --video-dir DIR
    /usr/bin/python3 lab/hom_setup.py putdown --plan PLAN --object sphere --map MAP --level-max 0 --go --video-dir DIR

## State on 2026-11-16 13:21

**The ground-clear grasps transfer.** The grid2dgr twist grasps put the contacts 2-5 deg below the sphere's equator,
with the palm about 9 mm higher. On the real hand both twisted the sphere more than 40 deg over the full 8.3 s. The
sphere drifted about 10 mm and was not pushed up into the hand. The film grasps (18-20 deg below the equator) pushed it
up 13-30 mm. Simulation had scored the ground-clear grasps 0.15-0.17, against 0.64-0.76 for the film grasps. The film
is `media/20261116-real_hand_twist_ground_clear.mp4`: 27.0 s from one continuous recording, the controls in real time
and the rest at 10x with still stretches cut (`lab/film_session.py --trim-idle --blur`); a person at the right edge is
blurred in its last three segments.

No ground-clear press grasp exists: the toy's press pose puts the middle finger 30-45 deg below the cylinder's axis.
`plans/20261116-press_x70_y110_lat0_plan.json` is the generalist press with the thumb-index pinch raised to the axis's
height; simulation scores it 0.17. On the real hand it held the cylinder through the full 6 s, where the generalist
press dropped it. That last trial recorded the fingertip boards (mux channels 0, 1 and 3, 160.6 records/s) with
`reskin5x_mux.py`, into `logs/20261116-three_finger_mux/20261116-131831-mux.bin`. No other trial recorded them.

`putdown` now ends with `letgo`: release, straighten, lift. The fingers straighten when every finger part either moves
away from the object or keeps 3 mm. `letgo` also runs on its own.

| record | plan | outcome |
|---|---|---|
| 125922 | twist, ground-clear grasp 0, (70, 110) | twist -0.4 to +42.5 deg; full 8.3 s; drift about 10 mm |
| 130318 | twist, ground-clear grasp 13, (105, 70) | twist -12.3 to +32.2 deg; full 8.3 s; drift about 11 mm |
| 131239 | press, generalist, (70, 110) | drift guard at about 6 s (30.6 mm); the cylinder dropped |
| 131927 | press, pinch at the axis (lat0), (70, 110) | full 6 s, held; the cylinder ended turned 22.6 deg about the vertical |

The trials recorded joints, HOM targets, object pose, arm pose and the HEX-E's wrench at 50 Hz. That is enough for a
first sim-to-real comparison of HOM and a first replay fit of the contact model, but with one or two runs per plan
there is no spread. Calibrating the magnetic fingertips needs those boards recorded in every trial and a force
reference, such as presses against a force sensor.

## State on 2026-11-16 12:57

The afternoon session (owner at the UR5) added these:

- **Putdown, zero and recording.** `putdown` now needs `--object` and `--map`. It measures the object and predicts the
  touchdown from the block's height above the table. It descends at 10 mm/s, then at 2 mm/s, and the tracker stops it
  when the object moves 0.8 mm against the hand. At 12:13 the HEX-E alone had let the hand go 66 mm down with the
  sphere sliding up into it. Then it checks the seat and, on the hand's model, the release path. It releases slowly,
  rises with the fingers open while the tracker watches the object, straightens the fingers and rises again. `zero`
  refuses to straighten the fingers while they are round the object, unless given `--unchecked`. Every moving command
  takes `--video-dir`, and `hom_trial.py trial` takes `--video-dir --label`. `lab/film_session.py` joins a session
  folder into one film, with the trials at 1x and the rest at 6x.
- **Fingertip tags** (`lab/fingertip_tag_check.py`, at the (70,110) layout). The index and middle DIP tags (25, 26)
  agree with the hand model to 3.2-4.3 mm and 7.0-7.1 mm at the zero, open and grip poses. Their normals agree to
  1.9-3.4 deg. The thumb agrees to 3.2 mm. Across the camera's view the error is 2-3 mm; the rest lies along the one
  camera's depth. Placement does not explain the 7-30 mm the sphere is pushed up in the hand. These tags face away from
  both cameras; the check turned the wrist -135 deg, which the owner does not want repeated.
- **MCP readings past about 62-68 deg flexion are wrong.** There the servo's potentiometer is out of range and the
  reading jumps by about -110 deg. Never turn torque on with a finger curled that far. The flexion signs are right.
- **Ground-clear grasps** (grid2dgr, contacts 2-5 deg below the equator, `plans/20261116-film_*_gr_plan.json`). In
  simulation the sphere turns, late and with the index off: 77 and 73 deg of the commanded 90 in hand-4d's CPU rollouts
  (g0 and g13, sv25 plant, 2.5 MPa pads, seed 0; rows
  `docs/experiments/20261110-codesign_formulation/rows/20261115-abstract_twist_{70_110,105_70}_gr_pads.json`). Their J of
  0.15 and 0.17 (0.756 and 0.643 for the film grasps) comes from the contact factor: the index pad carries load for only
  20-39 % of the task. Their simulated tracking is 0.62-0.74.
- **The continuous film's trials.** The film is `media/20261116-real_hand_twist_continuous.mp4`, 48.7 s, from camera
  832412070344, which has no people in view. Its segments are in `logs/20261116-hom_trials/20261116-1248_session/`.

| record | layout | twist | ended |
|---|---|---|---|
| 124945 | (105, 70) | up to +22.7 deg | drift guard at 3.5 s: 33.6 mm, the sphere 30 mm up in the hand |
| 125501 | (70, 110) | up to +17.9 deg | full length, 8.3 s; drift about 21 mm |

## State on 2026-11-16 12:20

The first live trials ran on 2026-11-16 with the owner supervising at the UR5. The set-up is one command
(`lab/hom_setup.py`), each trial is filmed by both cameras, and the station's stream holds 50 Hz with read-back.

A trial cycle on corvus (the HOM server for the plan running on the workstation):

    /usr/bin/python3 lab/hom_setup.py torque-on --go                      # motionless restart: goals = present, torque on
    /usr/bin/python3 lab/hom_setup.py zero --plan PLAN --go               # fingers straight down
    /usr/bin/python3 lab/hom_setup.py layout --plan PLAN --go [--video]   # plan in, gantries to the layout, filmed
    /usr/bin/python3 lab/hom_setup.py place --plan PLAN --map MAP --go     # hand 80 mm above the grasp height, levelled
    /usr/bin/python3 lab/hom_setup.py ready --plan PLAN --object sphere --map MAP --go
    /usr/bin/python3 lab/hom_trial.py trial --task twist --controller 100.65.73.59:PORT --object sphere --map MAP \
        --station http://10.99.99.2:8765 --live --arm --ft --out-dir logs/20261116-hom_trials
    /usr/bin/python3 lab/hom_setup.py putdown --plan PLAN --go            # down to a 3 N touchdown, slow release

`ready` straightens the fingers, measures the object, rises first when the hand is below the grasp's height, moves
sideways and then down to the nominal pose (the descent stops if the tracked object moves 1.5 mm or turns 3 deg), and
opens the fingers in small steps. It ended within 0.04 to 0.24 mm of the nominal pose every time. `trial` records
each camera at 960 x 540 (`<stamp>_twist_video_<serial>.mp4`, frame times in `_frames.csv`), stops when the object
drifts more than `--max-drift-mm` (20) from its start, and opens the arm's RTDE connection before t = 0. Without
`--go` every command only prints.

Changes of 2026-11-16:

- Objects. The owner glued the tag blocks to each other's objects: the cylinder carries ids 16 to 21 on the sphere's
  shorter stem, the sphere ids 10 to 15 on the cylinder's longer one (the 11-13 test block also has ids 10 to 15, so it
  stays out of view). `object_tracking_mjcf.assembled_map` gives the CAD maps; the fitted maps are
  `maps/20261116-*_fitted_map_O.json` (`block_views.py --ids --extra-ids`, `block_map_fit.py`, `object_tag_map.py
  --frame ends`). The cylinder's end tags put its block 0.25 mm lower than the as-assembled CAD and turned 1.4 deg.
  Measured masses: block 6.5 g, cylinder 23.6 g, sphere 21.59 g (taped).
- Network. The HEX-E's compute box answers at 192.168.1.1 and serves DHCP on the lab LAN. corvus's profile
  "ASUS Router lab" refuses its offers and holds 192.168.1.100 beside 10.99.99.77 and 192.168.123.100.
- CB1 driver (owner's go). `web.py` sets `disable_nagle_algorithm`: Nagle's algorithm and the client's delayed ACK had
  cost about 40 ms per request, so a frame with read-back went from 55 to 14 ms. `servos.py` and `runtime.py` read the
  stream's positions with one attempt per servo: a READ that timed out under the grip's load had been retried for
  1.5 s and the stream's lease lapsed. Backups are in `~/backups/20261116-*` on the CB1; the repository's copies carry the
  same edits.

The trials (records in `logs/20261116-hom_trials/`, the server's in `runs/` on the workstation):

| Time | Plan | Outcome |
|---|---|---|
| 11:37 | generalist twist | corvus froze 1.5 s at the lift (RTDE opened then); stream lost |
| 11:43 | generalist twist | full schedule at 18 Hz frames; the grasp scooped the sphere 24 mm up into the hand |
| 11:57 | generalist twist | stream lease lapsed after 33 frames; servos held the grip goals until overload |
| 12:00 | generalist twist | a servo READ timed out (1.5 s); loose grip, the sphere dropped at the lift |
| 12:09 | film (105, 70), grasp 10 | 191 frames without error; drift guard at 3.9 s (20.4 mm) |
| 12:11 | reconfiguration to (70, 110) | filmed, 57 s |
| 12:12 | film (70, 110), grasp 11 | 282 frames without error; twist about z_P 10-14 deg; drift guard at 5.7 s (30 mm) |

The film for the owner's talk, the counterpart of the abstract-hand film, is
`media/20261116-real_hand_twist_reconfiguration.mp4` (23.8 s: the (105, 70) twist, the reconfiguration at 8x, the
(70, 110) twist; camera 832412070344 re-timed to real time; its three parts beside it).

Findings:

- The grasp is deep and pushes the sphere up. HOM makes its force through the servo: at 0.8 s its targets jump 20 to
  36 deg past the measured joints (0.55 N m/rad for about 4 N), the fingers stall at the touch pose and keep curling
  (the thumb's PIP from 46 to 74 deg over the twist), and the pads roll off the contact. Next: a lower f_tripod and a
  cap on target minus measured.
- The thumb sits where the model puts it at the open pose: its DIP tag (joint two-camera solve) is 0.2 mm off the model
  along the approach to the sphere and 5 mm sideways.
- After a lapsed stream the station holds the grip's goals and the SCS0009's overload protection turns servos off.
  Torque comes on only through `torque-on` (a plain torque-on drives each servo to a stale goal).
- The index PIP servo read +186 deg after a touchdown with the finger in place; the owner re-zeroed it by hand.

## State on 2026-11-15 17:42

Built and checked without hardware:

- HOM controller server `src/sr2hand/control/hom_server.py` (`serve`). It answers corvus's state messages with
  HomBench's targets at 50 Hz and records every period. Over UDP against a simulated hand (`sil`), its targets equal an
  in-process loop through the same messages on all four tasks.
- Trial script for corvus `lab/hom_trial.py`: `check`, `trial` (link loop, the arm's lift at 1.6 s, a per-period record
  with the arm pose and the HEX-E), `return` (guarded move down). Its defaults move nothing. A dry run against a local
  server ran 302 periods and stopped at the schedule's end.
- The tracker places each frame with the arm's pose at the frame's capture time (`sr2hand.tracking.live.ArmPose.at`).
  Before, a frame 42 ms old was placed with the newest pose, up to about 8 mm off during the lift.
- The station's servo readings are stamped with their own age (`servo_age_s`), so stale readings during a stream show
  as stale.
- Plans of the first series (`plans/`, 14 files; `scripts/hardware/hom_bench_plans.py`): the five layouts of the
  task-conflict study with each task's S3 grasp, forces and contact-mode weights, mounts in the driver's coordinates
  (`sr2hand.robot.sr2_layouts`: driver = P - o, o = (-6.18, -2.40) mm for the thumb, (6.18, 3.40) for the index and
  (6.18, -3.40) for the middle). The sphere tasks run only at the generalist and pin-push layouts (the seating plan's
  decision 2). HomBench builds each plan's GPU world on the CPU (`HomBench.from_plan`, `scripts/hom/hom_bench_world.py`).
- `hom_server nominal --plan PLAN --task T --d D --out NOMINAL.json` gives the object's start pose in P for that
  grasp, and `hom_trial.py check --nominal NOMINAL.json` the tracked pose's distance from it.
- `scripts/hardware/object_tag_map.py` moves a block's fitted tag map into the object's frame (the 11-07 test block:
  0.33 mm RMS against the CAD).

Measured on corvus on 2026-11-13 (`docs/experiments/20261113-tag_tracking_throughput/`): tracker 29.9 Hz with 42 ms frame
age; link round trip 3.4 ms median to the workstation; HOM 4.0 ms per step on corvus's CPU.

Done since 14:15:

- The CB1 runs the read-back patch (owner's go, 2026-11-15: "go ahead on the CB1 patch and restart"). It was applied with
  `git apply -p0` (the CB1 has no `patch`), the originals kept in `~/backups/20261115-stream_read_back/` on the CB1, and
  the station restarted at about 16:12. On the CB1's own patched files with fake hardware
  (`scripts/hardware/station_read_back_check.py`, now without numpy): 50 frames, read-back equal to the written pose,
  2.96 ms per frame. The restarted station reads the nine servos (servo age 2 ms); it is unhomed with no plan loaded.
- The pin-push specialist runs at (75, 56.8), the owner's choice ("sure we can use that layout"): its four plans were
  rewritten there at about 16:18 with the study's pick (`hom_bench_plans.py write`; meta.hom keeps the study's layout).
  Its index and middle gantries sit at the end of their 60 mm travel (margin 0.00 mm). The twist grasp fails the pose
  toy's constraints there (meta.hom.toy_feasible false) and scores 0.619 against 0.641 at the study's layout (CPU, seeds
  20-24).
- From 14:15 to 16:22 `hom_bench_world.make_bench` ran without the plant's hold of the object (HomBench's own hold, 0,
  replaced the world's). The server and the SIL runs build HomBench from the plan and were not affected; the plans'
  check runs on make_bench again from 16:22 (the generalist pin-push plan gives 0.7273 again).
- Stickers for the backs of the DIP links (`assets/tracking/20261115-dip_link_tags/`) and the gantry travel measurement
  (`docs/experiments/20261115-gantry_travel/`): the stage scale, axis direction, backlash and travel ends from the tags.

Blocking a real trial:

0. The gantries (resolved at 17:38; gantry page `docs/experiments/20261115-gantry_travel/
   20261115-gantry_travel_dip_tags.html`). In the owner's DIP-tag session (16:53-17:13) the middle's palm-y stage J4
   moved nothing: its motor was unplugged at the CB1's stepper board, and a home that ends on its timeout (result 3)
   reports homed whether the axis moved or not. Plugged back in at 17:38, J4 moved 4.93 mm toward +y_P for a 5 mm
   command, the station's direction, and homed to its -85 end (tag at y_P -86.93, 0.04 mm from its earlier home). All
   six stages are homed at their stops (counters 0, checked against the tags). The station runs again since 17:42 (the
   owner's go): homed by adoption, every mount at its home (thumb -80,-55; index 80,85; middle 80,-85), servo torque
   off. By the owner's hand check, the index's palm-y
   stage J2 reaches about 1.2 mm past the station's limit at y 25 (to about 23.8 mm), and the other stages end at their
   commanded limits. With both cameras the stages travelled 0.99 to 1.00 times their command (index x 59.50 of 60 mm,
   thumb y 109.92 and 109.85 of 110, thumb x 60.07 of 60), and every jointly solved axis turned 1.3 to 2.3 deg
   clockwise about z_P: the hand sits about 2 deg in yaw off the tool chain's P, or the hand-eye does; the hand's tag 6
   can tell which.
1. Joint angles during a stream (resolved at 16:12: the CB1 is patched). The station does not read the servos while a policy stream runs
   (`manta_hand/runtime.py`, `_telemetry_loop`). Without measured angles HOM fails: in the simulated loop on the cad_v1
   hand it dropped the sphere on the twist and the roll (J 0 against 0.352 and 0.315 with the angles), whether its
   reference or its targets stood in for them. Change on the CB1 (needs the owner's go; the CB1's working copy runs
   uncommitted edits, so the change is made on that file alone): `stream_frame(..., read_back=True)` reads the nine
   positions after each write (`servos.sync_read_joint_positions`, nine reads in about 9 ms at a 1 ms FTDI latency
   timer) and returns them; `/api/v1/stream/frame` passes `read_back` and answers with them; `hom_trial.py` posts frames
   from a worker thread and takes them as the next state's joints. Made in the repository's copy and checked on the
   station's fake hardware (`scripts/hardware/station_read_back_check.py`: 50 frames, read-back equal to the written
   servo pose, 2.1 ms per frame without the real bus). The same change against the CB1's own files, whose `runtime.py`
   differs from the repository's only in comments and whose `web.py` keeps its own defaults, is
   `cb1/20261115-stream_read_back.patch`. To apply, with the owner's go and the station stopped:

       scp -o ProxyJump=corvus docs/experiments/20261115-hom_bench_trials/cb1/20261115-stream_read_back.patch irlab@10.99.99.2:/tmp/
       ssh -J corvus irlab@10.99.99.2 'cd ~/hand && D=src/morphohand/driver/manta/host/manta_hand && \
           cp $D/runtime.py /tmp/runtime.py.bak-20261115 && cp $D/web.py /tmp/web.py.bak-20261115 && \
           patch -p0 --dry-run < /tmp/20261115-stream_read_back.patch && patch -p0 < /tmp/20261115-stream_read_back.patch && \
           python3 -m py_compile $D/runtime.py $D/web.py'

   then `~/run_control_station.sh start` on the CB1. An unpatched station ignores `read_back` and answers without the
   positions.
2. The pin-push specialist layout (resolved: (75, 56.8)). (75, 55) does not fit the hand: its index and middle sit
   0.90 mm past their travel boxes (60.90 mm of the steppers' 60 mm). At (75, 56.8) the S3 picks score 0.916, 0.875,
   0.583 and 0.619 (pin-push, press, roll, twist; CPU, seeds 20-24) against 0.917, 0.874, 0.590 and 0.641; its twist
   grasp fails the pose toy's constraints.
3. Tag maps of the printed objects' blocks. Each block needs about 40 two-camera views and a bundle adjustment
   (`sr2hand.tracking.pose.bundle_adjust_multi`, as for the test block on 2026-11-13), then the map moved into the
   object's frame against the CAD map (`object_tracking_mjcf.object_map`).
4. The seat: the object rests on its block on a flat surface, and the arm brings the hand to it (`check --nominal`).

The plans in the simulated loop (`hom_server sil --plan`, the cad_v2 hand, the measured SCS0009 at 0.55 N m/rad, seed
20; UDP run equal to the in-process loop on each). J is the score of `hom_landscape.track_score`:

| Layout (x_sep, y_sep mm) | Pin-push | Press | Roll | Twist |
|---|---|---|---|---|
| Generalist (70, 110) | 0.277 | 0.935 | 0.404 | 0.439 |
| Pin-push (75, 56.8) | 0.373 | 0.764 | 0.364 | 0.421 |
| Press (70, 150) | 0.008 | 0.566 | | |
| Twist (100, 70) | 0.286 | 0.487 | | |
| Roll (85, 170) | failed | 0 (dropped) | | |

The pin-push row is at (75, 56.8) since 16:29 (at the study's (75, 55): 0.372, 0.764, 0.393 and 0.408;
`logs/20261115-hom_server_sil/sil_v3_pinpush_*.log`). At the roll layout HOM's Eq. (2) least-squares solve failed on the cylinder's diverged state (pin-push at 6.34 s); the
server then holds its last targets and stops the trial.

## Before a session

- The station turns the servos on when it starts, and each servo then drives to its goal register. To make a restart
  motionless, stop the station and, with torque off, set every servo's goal to its present position and its goal speed
  to 40 (`cb1/20261115-servo_hold.py`, the lab session's script of 11-15: copy it to the CB1, then
  `cd ~/hand/src/morphohand/driver/manta/host && .venv/bin/python /tmp/20261115-servo_hold.py`). On 11-15 at 17:42 the
  start that followed turned torque on with no motion and zero load on all nine servos, although the index's MCP servo
  sat at 149.7 deg against its 143.6 deg goal limit.
- Weigh each object with its stem and block; the simulation takes the measured mass.
- Grip tape on the surfaces HOM grips (the HOM results use friction 1.0; bare PLA gives 0.37-0.44, under which the
  twist fell from 0.91 to 0.08 in simulation).
- Sync corvus (its clone cannot fetch from GitHub): on the workstation
  `git push --no-verify corvus:workspace/hand main:refs/remotes/origin/main`, then on corvus
  `GIT_LFS_SKIP_SMUDGE=1 git merge --ff-only origin/main`.

## A trial

1. Workstation, the HOM server for the task and layout (plant `measured`: the SCS0009 at 0.55 N m/rad; `servo`: 0.72):

       .venv/bin/python -m sr2hand.control.hom_server serve --task press --d 25 --plant measured \
           --plan docs/experiments/20261115-hom_bench_trials/plans/20261115-generalist_press_plan.json \
           --listen 47002 --reply CORVUS_TAILSCALE_IP:47001 \
           --log-dir docs/experiments/20261115-hom_bench_trials/runs

   The station loads the same plan file (its mounts are the driver's), so the gantries and the controller agree.

2. corvus, the object on the seat and the hand over it:

       .venv/bin/python -m sr2hand.control.hom_server nominal --task press --d 25 --plant measured \
           --plan .../20261115-generalist_press_plan.json --out NOMINAL.json        # on the workstation, once per plan
       /usr/bin/python3 docs/experiments/20261115-hom_bench_trials/lab/hom_trial.py check --object cylinder \
           --map MAP.json --nominal NOMINAL.json

   The object's pose in P should be within 1 mm and 2 deg of the grasp's nominal pose.
3. corvus, a dry run through the real tracker (nothing moves; the server answers, the record is written):

       /usr/bin/python3 .../hom_trial.py trial --task press --controller WORKSTATION_IP:47002 --object cylinder \
           --map MAP.json --station http://10.99.99.2:8765 --out-dir OUT

4. With the owner present, the station homed with torque on and the layout's plan loaded: add `--live` (the fingers
   move), then `--arm --ft` (the lift at 1.6 s; the HEX-E's force change past 5 N or a protective stop ends it).
5. After the trial the hand holds the object in the air. The operator looks, then `hom_trial.py return` lowers the
   hand 40 mm at 10 mm/s (stopped by a 3 N force change).

## Records

Each trial gives two JSON-lines files with one line per 20 ms period, joined by the state's sequence number:

- corvus (`OUT/<stamp>_<task>_corvus.jsonl`): the state sent (joints and their time, the object's pose in P with its
  capture time, tracker flags and tag count), the arm's TCP pose, the HEX-E's wrench, the target applied with its age
  and round trip, the lift's phase.
- workstation (`runs/<stamp>_<task>_d<d>_trial<n>.jsonl`): the inputs HOM used (and where its joint angles came from),
  its targets in servo and sim degrees, the phase, the schedule's references, the compute time, any error.

Raw records stay out of git when over 2 MB; they move to the workstation by rsync.

## Scoring and calibration

    .venv/bin/python -m sr2hand.control.trial_score merge --server runs/SERVER.jsonl --corvus OUT/CORVUS.jsonl \
        --mass-g 25 --out TRIAL.jsonl                       # one trial file: header, one line per period
    .venv/bin/python -m sr2hand.control.trial_score score TRIAL.jsonl          # J and its inputs
    .venv/bin/python scripts/hardware/sr2_hom_replay.py replay --truth TRIAL.jsonl   # open loop at the header's plant
    .venv/bin/python scripts/hardware/sr2_hom_replay.py fit TRIAL_*.jsonl            # the plant from the trials

The merge joins the two records by the state's sequence number. Gravity in P comes from the arm's pose and the tool
chain, the target delay from corvus's round trips, and the lift's start from corvus's end line. The score follows
`hom_tasks.metrics` from the tracked quaternions; `J_025` counts a pad in contact from 0.25 N, the fingertip boards'
floor. The replay applies the recorded targets open loop from the first sample. The fit minimizes the joint, position
and orientation errors of the replays over kp, damping, torque limit, dead band, joint friction, contact friction and
pad modulus, with a held-out split. Fitting is to trajectories, never to task outcomes (owner, 2026-10-06).

Checked in simulation: a SIL trial on the generalist twist plan, both records written (`hom_server sil --corvus-log`),
merged and scored at J 0.4397 against the simulation's 0.4386, replayed at its own plant to 0.0013 deg RMS in the
joints, 0.0024 mm in the object's position and 0.0079 deg in its orientation. On six synthetic trials at kp 0.55 and
friction 0.8 the fit recovered kp 0.552, friction 0.809, damping 0.0146 and pad modulus 2.58 MPa (truth 0.0144 and 2.5),
in 2.5 to 13 min on 7 CPU workers (`logs/20261115-sr2_hom_replay/`). The torque limit was not reached in those trials
and stayed undetermined.

Not yet in the replay's model: the stem and tag block (the scene's object is the bare 25 g sphere or cylinder) and the
arm's lift (the replay removes the post at the lift's start). A record without fingertip boards has no contacts, which
the score cannot tell from no contact; with the boards, the press's pusher (median 0.18 N in simulation) lies below their
0.25 N floor. The fit's error weights (1 deg, 1 mm, 1 deg) are to be set from the measured noise of the servos and the
tracker.
