# real_v1 bench suite — data collection protocol and guide

**Status 2026-08-30 evening.**  The instrument works.  Two designs are measured.
The measurement itself is now the bottleneck, and the next section of this file is
a hardware spec for fixing that, because the fix is a printed part and a camera
bracket rather than more code.

Everything runs from the workstation against the CB1 service.  One command per
session; hand back the whole directory.

```bash
python3 scripts/real_v1_bench_session.py --design g12 --note "what this is testing"
python3 scripts/real_v1_bench_report.py            # read every session back
```

---

## 0. What we are measuring, and what a result looks like

**The dependent variable is the angle the cylinder turns about the pinch axis.**
Everything else in the log is an instrument reading that explains that number.

The plan commands a −70° rotation about **palm X** (the thumb↔pair direction);
the shaft lies along **palm Y**.  The hand is not asked to move the shaft anywhere,
only to roll it in place.  A result is a rotation angle with a repeat count and a
stated measurement method — nothing else counts, because there is no object-pose
sensor on this hand.

Three secondary readings make a rotation number interpretable, and a rotation
number without them is not usable:

| reading | where it comes from | what it rules out |
|---|---|---|
| **driver-finger yaw at the last step** | the JSONL | the finger did not move ⇒ the shaft could not have turned |
| **free-air yaw on the same day** | the `freeair` arm | separates grip load from servo droop and from the design |
| **`slipped`** | the operator | a shaft that rotated *because the grasp released* is the opposite result |

That last one is not pedantry.  It has already happened: `20260830-1437-g12` repeat
2 read **90°**, by far the largest rotation on record, because the index finger came
off and the cylinder fell through the grasp.  Pooled with the real runs it turned
g12 into "26 ± 17°".  Excluded, g12 is **10 ± 0° over two clean runs**.  The report
now segregates slipped runs automatically.

### Where the numbers actually stand

```
design    rep path   maxu drv      free   load   defc  peakL   plateau  spd  held    rot  slip touch
g12         1 chord  1.00 middle  +24.0  +13.2  +10.8    585  225x   3    -  True     10 False False
g12         2 chord  1.00 middle  +24.0  +23.4   +0.6    600   60x   2    -  True     90  True  True
g12         3 chord  1.00 middle  +24.0  +11.4  +12.6    795  200x  17    -  True     10 False False
g23         1 chord  0.92 middle      -   -3.2      -    810  200x  17    -  True      5 False False
g23         2 chord  0.92 middle      -   -3.2      -    705  180x   2    -  True      2 False False
g23         3 chord  0.92 middle      -   +1.2      -    795  200x   8    -  True      2 False False
```

- **g12 turns ~10°, holds 3/3, and gives up ~12° of driver yaw to grip load.**
- **g23 turns ~3° and its driver finger does not move at all** (−3.2, −3.2, +1.2°
  against a +24° command) at loads 705–810.  Its index also loads to 765 — two
  fingers in overload.  g23 is a negative result and does not need re-running.
- The two designs are separated by **7°, which is larger than the ±5° eye-estimate
  floor.**  So the current instrument can rank designs this far apart and no closer.

## 1. Yes, `real_v1_bench_session.py` stays the entry point

It is the thing that makes a session interpretable a week later, and that is the
whole reason it exists.  Nothing about the AprilTag work replaces it — the tag
changes `rotation_source` from `eye` to `apriltag` and adds two fields.  It does
not change the arms, the gating, or the directory layout.

**Division of labour with the web UI, which you should keep open the whole time:**

| the web app | the session driver |
|---|---|
| bring-up: backend real, servo torque on, home the gantries | refuses to start if any of that is wrong |
| **E-stop** — `Stop`, `Motors disable` | has no E-stop; it is a script |
| re-staging between repeats: `Load open keyframe`, `Load grasp keyframe` | `ramp_to()` walks the fingers back automatically, but the *shaft* is yours |
| live load view while you watch a run | records one measured servo sample per step |
| fiddling: nudging a finger, checking a pose | never touches the hand outside an arm |

The one hard rule: **only one writer on the bus at a time.**  Servo telemetry
suspends while a writer owns it, so a stale reading *during* a run is expected and
a stale reading *between* runs means the web app is mid-operation.  Don't drive a
pose from the UI while a session arm is running.

### What changed today

- `ramp_to()` between repeats (bf79fe0) — before it, repeat 2 opened with middle
  yaw at +18.7° against a commanded 0.0 and a yaw load of −930.  Repeats that do
  not return to the staged pose are not repeats; `20260830-1359-g12` is retired
  for exactly this and carries an `EXCLUDED.txt` saying so.
- `--speed-sweep 40,80,150` — one loaded repeat per speed, `servo_speed` recorded
  per run.  This is the "spam it at different speeds" knob.
- `homed == False` is now a blocking preflight problem (`/stream/start` 409s
  otherwise, which is what killed the 1359 free-air arm).
- The session prompts for `--note` if you didn't pass one.  Every session recorded
  so far has an empty note.
- `started_unix` in each run's meta, sub-second — so a video frame can be aligned
  to a **step** rather than to the nearest second: `frame_t = started_unix + step["t_s"]`.
- `EXCLUDED.txt` in a session directory retires it without deleting it.

## 2. The experiment that is actually worth the bench time

The 2026-08-29 stall was **not mechanical**.  Middle-yaw load rose 615 → 630 → 705
→ 690 and then dropped in one control step to exactly **200** — 20 % of the 0–1000
scale, the servos' `protective_torque` — and sat there for 32 consecutive steps
without varying by one unit.  200 is the only plateau in the whole dataset that is
not a multiple of 15, so it is not a load measurement at all.

Separately, the shipped plans were clipped at Policy B's ±0.5 rad residual budget.
Re-exported (`--budget`, commit 6842519), g12's middle yaw wants **62.65°**; the
plan you have been benching commands **28.65°**, i.e. **46 % of its own turn**.
Clearance is unchanged at every budget (+8.7 mm), because the minimum sits at the
grip pose, which the budget does not touch.

So there are two candidate explanations for the ~12° ceiling and they are
**separable by a 2×2**:

| | `protective_torque` = 20 (stock) | `protective_torque` = 40 |
|---|---|---|
| **g12** (28.6° commanded) | done: ~12° yaw, 10° turn, plateau at 200 | does raising the trip point recover the missing yaw? |
| **g12w11** (62.7° commanded) | is the ceiling command-limited or load-limited? | best case; if this doesn't beat 13°, neither knob is the answer |

Read it like this:

- **g12w11 stock reaches more yaw than g12 stock** ⇒ the plan was under-asking; the
  clip was the ceiling.
- **g12w11 stock trips *earlier* in `u` and reaches the same or less yaw** ⇒ load is
  the ceiling and the extra command buys nothing.  This is my expectation.
- **`protective_torque = 40` moves the plateau 200 → 400 and the yaw past 13°** ⇒
  the ceiling is a *configuration* ceiling and it is recoverable for free.
- **Neither cell beats 13°** ⇒ the ceiling is the grip itself, and the fix is
  mechanical: less `squeeze`, or a compliant pad.

Three repeats per cell.  Free-air once per design per day.

### Running the protection cell

It needs the raw bus, so the control station must be stopped and it runs on the
CB1.  `--arms protection` prints the exact procedure and records your readings.
The number to read is the **plateau value** in the report's `plateau` column:
200 before, 400 after, if the mechanism is what we think.

**Put `protective_torque` back to 20 afterwards.**  That protection exists for a
thermal reason and we have no temperature margin data.  Also expect a *wider*
command to trip *sooner*, which is why the two knobs are best moved together.

### Priority order

| # | command | what it settles |
|---|---|---|
| 1 | `--design g12 --repeats 3` | a third clean g12 run; the baseline is currently n=2 |
| 2 | `--design g12w11 --repeats 3` | is the ceiling command or load |
| 3 | `--arms protection` then repeat 1 and 2 | the configuration ceiling |
| 4 | `--design g12 --arms loaded --speed-sweep 40,80,150` | does a slower turn beat the trip |
| 5 | `--design rv04_mid --repeats 2` | the search's best design, one holder, truncated to u=0.65.  Expect drops; that is the point |
| 6 | `--design g24 --repeats 2` | one holder, 5/9 joints clipped, u=0.55.  Least likely to work, cheapest to falsify |

g23 is done.  Rows 1–3 are the ones worth having.

## 3. The measurement upgrade — cylinder and camera spec

This is the part to read before you open CAD.

Every rotation number in this suite is an operator eyeball.  That is a **±5°
instrument at best**, it cannot see axial slip at all, and it makes `slipped` a
judgement rather than a measurement.  The fix is not a better eyeball, and it is
not a 3D pose estimate either.

**The trick: make the turn an *in-plane* image rotation.**  The turn is about palm
X.  A flat tag whose face normal is parallel to palm X stays face-on to a camera
looking along X for the entire rotation, and the turn appears as pure in-plane
rotation of the tag in the image.  In-plane angle is the single most accurate thing
a tag detector gives you — sub-degree, no pose ambiguity, no camera calibration
beyond lens distortion.  A tag on the cylinder's *end face* would have to be read
out-of-plane and is exactly the ambiguous case; don't do that.

### The cylinder

Current object, which must not change: **Ø25.0 mm × 100 mm long, 24 g**
(`<geom type="cylinder" size="0.0125 0.05" density="500">`).  Diameter and surface
friction are both documented fragility cliffs — print it in the same material, the
same orientation, and the same diameter, or the new part is a second variable.

Contacts sit at **y = ±40 mm** (the index/middle pair) and **y = +20 mm** (the
thumb), so the whole barrel out to ±50 mm is grasp surface and must stay clean.

Add a **flat vane** on one end:

- **Which end:** the one that *rises*.  `_rotx(−70°)` sends the +y end down and the
  −y end up, so the vane goes on **−y**, which also puts it clear of the support
  post at y = −35 mm.  It is worth confirming this with one free-air run with the
  vane taped on before you commit to a print.
- **Geometry:** the vane is a flat plate lying in the **Y–Z plane** — its face
  normal points along palm X, the same direction the camera looks.  Spanning
  roughly y ∈ [−52, −97] gives a ~45 × 45 mm face, which carries a 35 mm tag with
  border.  It must be rigid and it must be indexed to the shaft's roll, so
  **print the shaft and the vane as one part** rather than gluing.
- **Do not mirror it on the other end.**  A matching +y vane would swing *down*
  ~89 mm during the turn and end up ~11 mm off the bench.
- **Balance it.**  A 5 g flag 80 mm out is a 2.5× increase in the gravity moment
  about the pinch axis, which is a real confound and not a small one.  Put a
  counterweight pocket (a couple of nuts, or a brass slug) at about y = +55 mm and
  balance the finished assembly on a knife edge within a few mm of mid-length.  The
  commanded pivot is the three-tip centroid, ~7 mm off-centre toward the thumb, so
  exact balance is not needed — a 2.5× moment change is.
- **Mass budget:** 24 g now, and 65 g is proven to work on g12.  Shaft + vane +
  counterweight should land ~35–40 g, comfortably inside the band, but weigh it and
  record the number in the session note.
- **Finish:** matte.  Glue a **paper-printed tag** on flat; tags printed as plastic
  geometry read badly under bench lighting.

### The reference tag — the part that makes the camera mount easy

Mount a **second tag, different ID, on something rigid that never moves** (the
bench, or the palm bracket — not the hand, not the post), facing the camera, at
roughly the same X station as the vane so the perspective matches.

Then the reported angle is **`vane_angle − reference_angle`**, and camera roll,
camera drift, and re-mounting between sessions all cancel.  This is what turns the
camera bracket from a precision part into a clamp.  Without it, every session needs
the camera in the same place to a degree; with it, "roughly there" is fine.

### The camera

- **Optical axis parallel to palm X**, within ~±3°.  It does *not* need to be *on*
  the axis — a rotation about X reads the same in-plane angle from anywhere with an
  X-parallel view, up to a second-order perspective term.
- **Side:** the −X side (the thumb).  One finger in the way instead of two, and the
  thumb barely moves during the turn (11° of yaw, 4° of MCP), so whatever it
  occludes it occludes constantly.
- **Distance 250–400 mm.**  At 300 mm a 1080p / 60°-HFOV webcam gives ~5.5 px/mm,
  so a 35 mm tag is ~190 px — far above the ~30 px a detector needs.
- **30 fps is plenty.**  The commanded turn is 1.1 s; the stepped replay takes
  ~6 s with gating and dwell.
- Aim it at the vane's *sweep*, not at its start pose: the vane travels a ~90 mm
  arc.  Frame ~150 × 150 mm of free space around it.
- **Rigid to the bench**, and it must not be bumped mid-session.  If it is, the
  reference tag absorbs it — say so in the notes anyway.

### What the camera buys beyond the angle

Three things that are currently unmeasurable:

1. **Axial slip.**  If the vane's centroid translates along Y in the image, the
   shaft slid through the fingers.  This is the open defect no reward term has ever
   measured.
2. **`slipped` becomes a measurement.**  A grasp release and a driven turn look
   identical in the servo log and completely different in the tag trace — the
   release is fast, monotonic, and continues after the fingers stop.
3. **Drop detection**, for free: the tag leaves the frame.

### Software

`scripts/real_v1_vane_angle.py` reads a video and writes per-frame
`(t, vane_deg, ref_deg, angle_deg, cx, cy)`.  It uses OpenCV's ArUco detector with
`cv2.aruco.DICT_APRILTAG_36h11` for the four corner pixels — never a pose — and
`imageio`, already a project dependency, for frames.  The only extra install is the
detector:

```bash
uv pip install opencv-python-headless
python3 scripts/real_v1_vane_angle.py run.mp4 --vane-id 0 --ref-id 1 \
    --t0 <started_unix from the run's meta> --out vane.csv
```

**Validated on synthetic footage only.**  A 70° sweep filmed by a camera rolled 7°
reads back to **0.33° rms, 0.64° max**, with the reference tag absorbing the roll
exactly; a sweep crossing the ±180° wrap reads to 0.51° max.  So the method is
worth ~0.5° against the eyeball's ±5°.  It has not met real lighting, motion blur,
or a tag seen at an angle — expect one pass over it after the first real video.

## 4. What to write down at the bench

The driver asks after every loaded run.  Answer even when the answer is boring — a
blank is indistinguishable from "did not happen".

| field | why it exists |
|---|---|
| `held` | the log ends at the trajectory; whether the shaft is still there is not in it |
| `rotation_deg` | **the dependent variable** |
| `rotation_source` | `eye` / `protractor` / `apriltag` / `video`.  An eyeball and a tag read are not the same measurement and the report says so |
| `vane_deg_start`, `vane_deg_end` | recorded when `rotation_source` is `apriltag`; both readings, not just the difference |
| `fingers_touched` | ground truth for the clearance table |
| `slipped` | **the run is excluded from the rotation mean if this is true** |
| `media` | filename of any photo/video, so it can be found later |

Until the vane exists, the honest fallback is a fixed phone on a tripod
perpendicular to the pinch axis with a paper protractor behind the shaft, and a
still at the grip pose and at the end pose.  That is ±3–5°, enough to separate
g12 from g23 and not enough to rank anything closer.

Useful proxy in the meantime: **object rotation tracked driver-finger yaw close to
1:1** on g12 (15° of cylinder for 13.2° of middle yaw), and driver yaw *is* in the
log.  It is a hypothesis from one run; do not report it as a measurement.

## 5. Truncation table — do not raise these without re-running the scan

| design | drivers (relieve) | holders (keep firm) | joints clipped at ±0.5 rad | safe `max_u`, chord | csv | min clearance |
|---|---|---|---|---|---|---|
| **g12** | middle | thumb, index | 2/9 | **1.00** | 1.00 | +8.7 mm |
| **g12w08 / g12w11** | middle | thumb, index | 0/9 (re-exported) | **1.00** | 1.00 | +8.7 mm |
| **g23** | middle | thumb, index | 2/9 | **0.92** | 0.84 | +0.8 mm |
| **g24** | thumb, middle | index | **5/9** | **0.55** | 0.10 | −5.2 mm |
| **rv04_mid** | thumb, middle | index | 3/9 | **0.65** | 0.70 | −2.6 mm |

The driver picks `max_u` from this table automatically.  The sim's links are
thinner than the printed parts, so these are already optimistic.
`scripts/real_v1_trajectory_clearance.py --all` regenerates it.

**g24 and rv04_mid have only one holder** — two of three fingers travel more than
60 % of what the busiest one does, so one finger has to anchor the shaft while the
other two turn it.  They are also exactly the two designs that interpenetrate.  If
they drop the shaft twice, that is the result, not bad staging.

## 6. Things not to do

- **Do not command the plan's grip pose directly.**  `<design>_build.txt` asks for
  10 mm of "squeeze" (pads driven inside the object surface) — soft-contact
  compliance in MuJoCo, pure clamping force on a printed shaft.  Always go through
  `grip` → `regrip`.
- **Do not use `--load-delta` without `--stall-deg`.**  Load rising *is* the turn
  working; only load-plus-no-motion is a fault.  A bare load abort killed a healthy
  run at step 10.
- **Do not `--preload-start` below ~9.0.**  Starting the walk-down at 5.0 capped the
  thumb at load 270 and it never reached its 450 target.
- **Do not read servo telemetry while a writer owns the bus** — you get one stale
  sample repeated.  Every helper here asserts `servo_polling_suspended == False`
  and a fresh `servo_age_s`.
- **Do not pool a slipped run into a rotation mean.**  The report won't, but a
  hand-written table might.
- **Do not run a design above its `max_u`** without re-running the clearance scan.
- **Do not change the cylinder's diameter, material, or surface** while adding the
  vane.  Both are documented fragility cliffs.

## 7. Handing the data back

```
docs/experiments/20260830-real_v1_bench_suite/20260830-1530-g12/
  MANIFEST.json          preflight, plan facts, note, commands, observations
  freeair.jsonl          per-step commanded/achieved/load, no object
  grip_seat.stdout.txt
  regrip.stdout.txt
  regrip_pose.json       the relieved grip actually used
  loaded_1.jsonl  loaded_2.jsonl  loaded_3.jsonl
  loaded_*.stdout.txt
  EXCLUDED.txt           only if the session should not be pooled — say why
  <media>.mp4            if you filmed it
```

Commit the directory (they are small) and say which sessions are new.  Then
`python3 scripts/real_v1_bench_report.py` gives the whole table.  The first thing
worth extracting from a loaded log is the driver finger's **yaw at the last step**
against the free-air value from the same day — that difference is grip load and
nothing else.
