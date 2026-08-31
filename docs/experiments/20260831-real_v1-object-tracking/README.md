# The bench gets an object-pose sensor — 2026-08-31

*Digestible version:* <https://claude.ai/code/artifact/56cfae1b-3922-48ae-8e20-94ea2a7f66ca>
(source kept here as `tape-measure.html`).

Every hardware result this program has produced was scored by an operator looking at the shaft
and typing a number. `scripts/probe_obs_ablation.py` called the object-derived observation
blocks *hidden*; `real_v1_bench_session.py`'s docstring said "there is no object-pose sensor on
this hand"; `HandRuntime.state()` reported `capabilities.object_pose: False` as a hard-coded
literal. All three are now false, and the change cost two printed tags and four tape-measure
readings.

Two things happened today. They are unrelated in mechanism and closely related in consequence.

---

## 1. The ±70° yaw cap is gone, and the two audited plans load

`sv1_u0060_b100` and `sv1_w0116_b100` were shipped to the station on 2026-08-30 and would not
load: `HandRuntime.load_plan` raises on any `HandPlan.validate` violation, and both overran the
aa/yaw command range — by 3.88° and 6.27°, both at `turn_end`, both on **middle yaw**.

That range was not a hardware limit. `servos.FINGER_JOINTS` had it at ±70°, a conservative cap
the user set on 2026-08-29 before any plan had been driven; the declared contract, the one
`assets/mjcf/real_v1/real_hand.xml` still encodes and `test_manta_frame_map` pins, is ±85°. So
the cap was not protecting the servo, it was **deciding how far the hand may turn** — a screening
variable that nobody chose, exactly like the ±0.5 rad residual clip was
([the re-screen](../20260830-real_v1-budget-rescreen/README.md)).

Restored to ±85° at the user's request. What the manual torque-free sweep actually demonstrated,
in this module's own zero-relative degrees:

| servo | joint | swept range |
|---|---|---|
| 0 | thumb aa | −70.02 … +74.71 |
| 3 | index aa | −79.69 … +74.41 |
| 6 | **middle aa** | **−162.60 … +136.82** |

Servo 6 — the one both plans need — was swept well past 85° in both directions. Thumb aa has
never been shown past −70.02, and that is a gap in the *evidence*, not a measured hardstop: the
sweep records where a person stopped moving a freed joint. No deployed plan commands thumb aa
past −23.1°, so nothing rides on it today; if one ever does, sweep servo 0 again first. The note
is in `servos.py` above `AA_LIMIT_DEG`.

### What the cap was costing

`real_v1_plan_band.py --servo-only` recomputes the shortfall column without re-running the 1,716
rollouts behind the band scan (the shortfall is kinematics against a table, and only the table
moved):

| plan | clip ceiling under ±70 | under ±85 |
|---|---:|---:|
| sv1_u0060_b75 / _b100 | 0.90 | **unbounded to 2.00** |
| sv1_u0308_b050 | 0.80 | 2.00 |
| sv1_u1364_b080 | 0.80 | 2.00 |
| sv1_w2360_b075 | 0.85 | 1.10 |
| sv1_w0099_b100 | 1.20 | 1.30 |
| sv1_w0116_b100 | 0.85 | 1.30 |

All 17 plans now validate — verified on the CB1 itself, not just here. `sv1_u0060_b100` enters
the ranking at **rank 8** (holds 4/4, cos 0.44). `sv1_w0116_b100` loads but stays unranked: it
holds 2 of 4 at its own clip of 1.00, and the band scan puts its hold at **1.45–2.00**, which is
still above its 1.30 ceiling. It is a plan running 45 centirad below the band it wants. The
catalog says so in as many words now — until today `expectation()` described any plan outside
its band's lower edge as being *at* that edge, which reads like a tuned plan.

---

## 2. Two AprilTags, and where the shaft actually is

`id 6` (40 mm) is static and vertical; `id 0` (30 mm) rides a vane on the cylinder, its plane
containing the shaft axis. The camera is a RealSense D435 on the **workstation** — the CB1 has
neither the USB bandwidth nor the dependencies — reading IR stream 1 with the projector off at
1280×720/30.

### The measurements that make it absolute

Recorded by the user, 2026-08-31, all in mm:

| what | value |
|---|---|
| reference tag centre, above the floor | 175.0 |
| …from the palm centre along x (past the index gantry) | 133.5 |
| …from the palm centre along y (toward the thumb side) | 15.0 |
| cylinder tag centre, past the flat end face | 21.0 → **71.0 from the cylinder centre**, on the axis |
| distal link's top flat edge above the floor, link vertical | 65.0 (link is 37.2 long) |

**The fingertip datum is the load-bearing one.** It is the only measurement that ties the bench
floor to the simulator's floor, and it does it through a part both sides model. In
`real_hand_morphology_actuated.xml` at the zero pose, with the palm at 62.5 mm, every
`<finger>_pip_frame` sits at z = 21.0 mm and the distal link runs 37.2 mm down from there to
16.2 mm *below* the floor — which is `build_real_v1_scenes.py`'s own "straight fingers put every
pad 16 mm below the floor" note, arrived at from the other direction. So

```
z_bench = z_sim + 44.0 mm
```

and the physical palm plane stands 106.5 mm over its table where the sim's stands at 62.5 over
its own. The bench scenes' 100 mm post is **144 mm** on the bench floor. `SIM_TO_BENCH_Z_MM` is
the only place that 44.0 is written down, and `tests/test_bench_tags.py` derives it rather than
asserting it.

The 71 mm matters more than it looks. The tag is on a vane *past the end face*, so through a 90°
turn the tag sweeps 71 mm while the cylinder's centre barely moves. Reporting the tag's position
as the object's would put a 70 mm phantom translation in every trace.

### What is calibrated, and what is withheld

A single static tag supplies world up (observed) and, through the tape measure, a height datum.
In general it cannot supply its own heading about vertical — but **on this rig it can**, and an
earlier version of this note wrongly treated that as a standing limitation.

The reference tag is bolted facing normal to the gantry x-axis and stays that way, with the
camera facing it. So its in-plane horizontal axis is ±y_B: the heading is **±90°, and only the
sign is unknown**. The two candidates mirror the shaft about the tag's own x, landing it at
133.5 − d and 133.5 + d, and the hand at x = 0 decides between them.

That discriminator only works when **the shaft is staged in the hand**, and the first live arm
showed why the test has to be able to fail. With the shaft parked on its post beside the
reference tag the two candidates came out at bench x +114 and +153 mm — both far outside the
hand, 19 mm either side of the tag — and the first version of this method simply took the
smaller and reported it as "the hand's side". It would have written x/y into the trace off a
coin flip. The rule is now **exactly one candidate inside `OBJECT_X_ENVELOPE_MM` (±90 mm)**;
staging beside the tag is refused with both candidate positions named, rather than guessed.
With the shaft in the hand the candidates are ≈0 and ≈+267 mm and the choice is unambiguous.
`BenchFrame.heading_from_mounting()` resolves it on the first frame the cylinder is seen, in both
`--probe` and a recording, and prints which it chose and why; a recording keeps retrying for
20 s rather than giving up on the first frame, since the tracker can be armed a moment before
the shaft is finally seated.

What that cannot check is the *premise*: if the tag is ever re-aimed by hand it is no longer
normal to the gantry axis and the true heading is not ±90 at all. `--calibrate-heading X,Y`
remains for that case, and `--no-mounting-heading` turns the inference off. When neither is
available `heading_deg` stays `None` and `locate()` returns height and radius with `xy = None` —
blank rather than a plausible-looking zero.

### The instrument, measured

179 frames on a static target at 320 mm, both tags at decision margin ~65:

| | rms | peak-to-peak |
|---|---:|---:|
| angle from vertical | **0.017°** | 0.092° |
| cylinder centre height | **0.030 mm** | 0.200 mm |
| frame rate | 30.0 fps | 100% visibility |

Against an operator's ±5° eyeball, and against the ±0.33° the vane-angle method managed on
synthetic footage. This is the *floor* — a static target at short range with no motion smear —
not what a swinging shaft will give. But it settles whether the instrument can resolve the
differences the sim predicts between plans (cos 0.73 vs 0.59 is 12° apart).

---

## Running it

```bash
# 1. before trusting anything. Checks exposure, both tags, and the SIGN of --shaft-axis
#    (get that backwards and the centre lands 142 mm the wrong way down the shaft)
~/miniconda3/bin/python scripts/real_v1_tag_tracker.py --probe

# 2. only if the fixed tag was re-aimed by hand. Stage the cylinder somewhere its bench
#    (x, y) is known, then:
~/miniconda3/bin/python scripts/real_v1_tag_tracker.py --calibrate-heading 0,0

# 3a. normal web-UI operation. Start this companion once per workstation boot; every
#     reorientation button press then arms and finalizes its own run-id trace.
MANTA_TOKEN="$MANTA_TOKEN" ~/miniconda3/bin/python scripts/real_v1_tracker_service.py \
  --host 10.99.99.50 --port 8770

# 3b. use the bench-session script only when its extra experimental arms/manifest are
#     wanted. Its own tracking remains automatic.
python3 scripts/real_v1_bench_session.py --design sv1_w2360_b075 --arms grip,loaded
```

`pyrealsense2` and `pupil_apriltags` are **not** in this repo's uv environment; on this
workstation they live in `~/miniconda3`. `real_v1_tag_tracker.py --print-interpreter` finds one,
and the bench session calls that automatically.

### What lands where

Each loaded repeat writes `loaded_N_track.csv` (per-frame: cos, degrees from up, bench height,
simulator height, radial distance, x/y when calibrated, detector margin, range),
`loaded_N_track_SUMMARY.json`, and `loaded_N_track.stdout.txt` into the session directory, and
the summary goes into `MANIFEST.json` under `runs[].tracking`.

`observe()` now offers the measured value as the default instead of asking for a recollection —
and **keeps asking**. The operator's reading stays a separate field, because the tracker cannot
see a shaft that turned by slipping through the fingers rather than with them, reports a dropout
as missing data where the operator sees a dropped shaft, and has no opinion about whether two
fingers touched. When the two differ by more than 10° the session says so at the prompt and
records `operator_minus_tag_deg`.

### On the station

`POST /api/v1/tracker/sample` takes one reading; `--push` sends them at 10 Hz. With the CB1's
`--tracker-url` configured, `/reorient` first asks the persistent workstation service to arm the
exact new run id and waits for id6 to latch and the first id0 sample to reach the CB1. It stops
and finalizes that trace after motion. An
arm failure refuses the motion, so a camera fault cannot silently create an untracked trial.
The web app grows
a **Shaft tracking** card: a signed dial (left is the wrong pole, and it is red), degrees from
up, height in both frames, peak and lowest cos, and how far it has fallen from the start. A
stale sample says stale and greys its numbers; a lost tag says LOST rather than freezing on the
last pose. Samples arriving during a run are appended to that run's JSONL as `kind: "object"`,
so the trace and the joint commands share one timeline, and the tracker's end-of-run summary
lands in the run's `_SUMMARY.json` under `object_track` — next to `manual_score`, never
replacing it. The scoring dialog seeds its angle from the measurement and says where the number
came from.

`capabilities.object_pose` is now derived from whether samples are actually arriving, not from a
literal. The camera can be unplugged, aimed at a wall, or looking at a shadowed tag, and in each
of those the station must not claim an object sensor.

### The bench session runs all seventeen plans now

`SAFE_U` was a hand-transcribed table of six designs. The deploy directory had grown to
seventeen on 2026-08-30, so the top-ranked hand on the station could not be run by the driver
meant to run it. It is now read from `deploy_clearance.txt`, keeping the hand-entered
truncations for the three designs whose fingers cross (`g23`, `g24`, `rv04_mid`) — that number
is a judgement about how much of a colliding path is worth running, which no report can supply.
`sv1_u1364_b080` correctly comes out `csv`-only.

---

## 3. So can we run the RL policy now?

`scripts/real_v1_obs_sources.py` maps all 66 columns and checks the total against the actor's
own input width (66, from `model_200.pt` — it matches):

| term | columns | width | status |
|---|---|---:|---|
| `joint_pos` | 0–14 | 15 | measured — servo present-position |
| `joint_vel` | 15–29 | 15 | measured — differenced, and noisy |
| `object_pos` | 30–32 | 3 | **measured — new today** |
| `object_pose_actual` | 33–39 | 7 | **measured — new today** |
| `ref_finger_qpos` | 40–48 | 9 | replayable from the frozen reference |
| `ref_object_pose` | 49–55 | 7 | replayable |
| `actions` | 56–64 | 9 | the controller's own last output |
| `target_axis_misalign` | 65 | 1 | **measured — new today** |

**Eleven of sixty-six went from absent to measured, and nothing is absent any more.** So the
observation problem is solved — and it turns out to be the smallest of four:

1. **Rate.** The tags run at 30 Hz and the policy at 50. The servo bus sustains ~111 Hz of sync
   *writes*, but sync *reads* do not work on the SCS0009 at all, so `joint_pos` costs individual
   reads.
2. **Plant.** The yaw joints arrive at 0.44–0.90 of what they are told
   ([20 bench runs](../20260830-real_v1_bench_sobol/README.md)), and it is torque, not speed. A
   policy trained where commanded equals achieved is driving a different hand, feedback or no
   feedback.
3. **The policy.** b33 was shown to **ignore its observations** — replaying the whole 66-dim
   input from a different rollout cost nothing. Handing a policy that ignores its inputs a real
   sensor changes nothing about what it does. *The tags make a sighted policy trainable; they do
   not make an open-loop one closed.*
4. **Safety.** Nothing gates a streamed policy against finger-finger collision, and three of the
   four originally deployed designs interpenetrate along their own planned path.

The honest answer: **not yet, and the tags are not what was stopping it — but they are what
unblocks the version that would work.** The path they open is not "stream b33 at 50 Hz". It is
that the object-derived columns are now a *training* signal that survives deployment, so a
policy trained to use them is no longer trained on information the bench cannot supply. That,
plus system identification of the yaw shortfall from traces the tracker can now produce
automatically, is the shortest route to a closed loop that means anything.

What the tags *do* deliver today, immediately: every bench run gets a real turn angle, a real
drop time, a real slip, and a height in the simulator's own coordinates — so the ranking in
`catalog.json` becomes falsifiable rather than merely written down.

---

## Files

| path | what |
|---|---|
| `src/morphohand/bench/tags.py` | the geometry. numpy only — no camera, no mujoco, no torch |
| `tests/test_bench_tags.py` | 16 tests, synthetic poses whose answer is known by construction |
| `scripts/real_v1_tag_tracker.py` | the maintained tracker: probe, calibrate, record, push |
| `scripts/real_v1_tracker_service.py` | persistent workstation camera owner for automatic web runs |
| `scripts/real_v1_obs_sources.py` | the 66-column map, checked against a checkpoint |
| `scripts/real_v1_bench_session.py` | tracking wired into every loaded repeat |
| `manta_hand/runtime.py`, `web.py`, `static/*` | the station side |
| `docs/experiments/20260830-apriltag-tracking/` | the original probe and the printable sheet |


## The reference tag's detectability is not stable (2026-08-31)

Recorded because it will happen again and because the first diagnosis was wrong.

Three probe frames, same tag, same camera, same afternoon:

| time | raw | after CLAHE | after equalization | Michelson contrast at the tag | sharpness |
|---|---|---|---|---|---|
| 14:11 | **62.4** | 61.6 | 36.8 | 0.678 | 777 |
| 15:20 | — | — | **35.4** | 0.716 | 593 |
| 16:57 | — | — | — | 0.711 | 683 |
| 17:06 | **62.6** | — | — | — | — |

The 16:57 frame decodes under nothing: not raw, not either normalization, and not under any of
24 detector configurations swept over `quad_decimate`, `quad_sigma`, `refine_edges` and
`decode_sharpening`. It was first read as a local underexposure, but the contrast and sharpness
columns rule that out — they are the same in the frame that works and the frame that does not.
The tag had shifted a few pixels in the image. Something bumps it, and it comes back on its own.

Two consequences. `detect()` now tries raw, then CLAHE, then global equalization, because the
two normalizations rescue different frames (CLAHE reads 14:11 at 62 where equalization manages
37; equalization is the only one that reads 15:20) — but the ladder is for the marginal middle
case and is **not** a substitute for staging. And `--probe` before every sitting is a gate, not
paperwork: require id6 at raw margin ≥ 30. With `--tracker-url` configured the station is
already fail-closed on this — an unlatched reference refuses the motion rather than producing an
untracked trial — so a bumped tag costs a refused run, not a wasted one.
