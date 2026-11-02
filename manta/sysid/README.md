# Servo system identification — working notes

## 2026-10-26: read-only health check (`servo_health_check.py`)

Ran on the CB1 (`irlab@10.99.99.2`) through a shared SSH connection, using the hand repo's driver
(`~/hand/.../manta_hand/servos.py`, byte-identical to hand-main's). The check only reads: no torque,
no register writes, no motion. Raw output is in `health_2026-10-26.json`.

**Bus:** 252 reads in 5.1 s, **0 timeouts**.
- FTDI (USB adapter) latency was **16 ms**. Setting it to 1 ms needs `sudo`, so it was left unchanged:
  `echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer`

**All nine servos are healthy:**
- IDs 0–8 respond, all model 1029 (SCS0009).
- Status 0 (no latched alarms), torque off, not moving, load 0.
- 5.0–5.3 V, 18–21 °C.

**Controller and protection settings are identical on all nine** (factory defaults, presumably):

| Register | Value | Meaning / use |
|---|---|---|
| P / I / D coefficient | 15 / 0 / 15 | PD position controller, no integral. Seeds the BAM controller model. |
| max_torque_limit | 100 | |
| overload_torque | 80 | |
| protection_time | 100 | Time above the overload threshold before protection trips |
| protective_torque | 20 | Torque the servo drops to after an overload trips |
| unloading_condition | 32 | Bit 5: unload on overload |
| led_alarm_condition | 37 | Bits 0, 2, 5: voltage, overheat, overload |
| max_temperature_limit | 70 | °C |
| voltage limits | 45 / 90 | 4.5–9.0 V |
| minimum_startup_force | 45 | |
| dead zones (cw / ccw) | 1 / 1 | |

Units for the torque and time registers aren't confirmed against the SCS0009 manual yet.

**Angle limits (EEPROM): don't trust the converted values, and don't rely on them for safety.**
- rustypot's *converted* `min/max_angle_limit` are inconsistent with position. Servo 7 reads a
  "limit" range of −230…−4° relative to its zero while it sits at +24°.
- The *raw* registers read min=0 / max=1023, i.e. **no restriction**, on **servos 0, 1, 5, 6, 7, 8**.
  Only **servos 2, 3, 4** have non-default limits.
- The driver docstring's claim that "each servo's own EEPROM limit is set to its real mechanical
  range" dates from the old 2-servo fingers and **doesn't hold for the current 3-servo hardware**.
- Conclusion: for six of nine servos, the software limits in `FINGER_JOINTS` are the only range
  protection. System-ID motions must clamp to them in software.
- A second discrepancy: hand-main's `FINGER_JOINTS` uses `AA_LIMIT_DEG` = ±85° for all three
  adduction joints, while manta-hand's older table had narrower measured adduction ranges
  (−70…+75° etc.). Check which is current before commanding large adduction moves.

## 2026-10-26: first step test (`single_joint_step_test.py`)
Index fe1, +10° and back, speed 80. Safe (0 timeouts, 22 °C). But goals went through the individual
write path, whose 0.3 s post-write quiet period hid the whole move, and load read 0 in every sample
(only holds were captured). Superseded by the test below.

## 2026-10-27: step + press test (`step_press_test.py`)
Same joint and step; goals via `sync_set_joints` (2 ms read gap). 2,799 samples at 332 Hz
(position + load + raw load), 0 timeouts / 8,405, 20 °C throughout.
- **present_load works.** The 0s last time were real: the servo outputs 0 when it is close enough
  to target. Raw register = byte-swapped 11-bit sign-magnitude (bit 10 = direction); rustypot's
  converted value matches that decode in 99% of samples (mismatches are single-sample tears).
- Values seen are all multiples of 15 (= P coefficient), consistent with load ≈ P × error counts,
  not yet confirmed.
- Move: 10° in ~0.25 s (~40°/s at speed 80), peak load −555 at start, ~−225 mid-move.
- Position resolution 0.293° (300°/1024 counts).
- Hold at +10°: settled 0.3–0.6° short with load 0 (within the servo's deadband).
- Return: overshot to −6.15° (target −4.69, 5 counts past) and stayed there, load −105:
  the servo is pushing but static friction holds the joint. First direct friction evidence.
- **Press (operator: ~6 s, pushing the finger open):** load stepped 0 → −60 at t = 6.1 s and held
  there to the end of the window, with at most a one-sample 0.3° dip in position. So load registers
  a press that position barely sees. The earlier +0.9° drift at t = 3.4–4.7 s was not a press
  (load stayed 0; it is within the deadband around the target).
- **Sign convention (consistent across move, press and return):** negative load = servo pushing
  toward + (closing); positive = pushing toward − (opening). Pushing the finger open → negative.
- Position was the same (4.69°) with load 0 before the press and −60 during it, so load is not
  simply P × (readable) position error; the multiples-of-15 pattern remains unexplained.
- Next: calibrate load against a known force (fingertip on a kitchen scale, or hanging masses).

### Run 2 (same script, operator pressed the finger open again)
`step_press_index_fe1_2026-10-27_run2.{csv,png}`. 332 Hz, 0 timeouts / 8,255, 20 → 21 °C.
- Step out stopped **1.2° short** (9.08 vs 10.29) with load −90: the servo is pushing but static
  friction holds the joint (run 1 stopped 0.3–0.6° short with load 0 — the stopping error varies).
- Press: the finger ratcheted open in single-count steps 9.08 → 8.50 → 8.20 → 7.91° while load went
  −90 → −120 → −135 → −150. **Each 0.293° count of deflection added exactly 15 load units** (= P gain):
  load behaves like the PD controller's output, i.e. a stiff spring on position error.
- Operator started pushing ~2 s after the sequence started (matches the first step at t = 2.25 s)
  and **kept pushing until torque was disabled**. The flat 7.91° / −150 from t = 4.5–6.9 s is a
  sustained press held against friction.
- **Load is non-zero with no contact**, shown by two no-contact holds: run 2 t = 0.75–2.0 s (before
  the press) −90, stopped 1.2° short; run 1 final hold −105 after overshooting. So load reflects
  controller effort / position error, not contact force: "stuck 1.2° from target" and "pressing on
  an object" can read the same. Give a policy load together with position error (goal − measured).
- During run 2's final hold the operator was still pushing but load read 0 (the finger had moved
  open, likely with the push, so little force was transmitted). Unexplained in detail.

## 2026-10-27: kitchen-scale calibration (`load_scale_calibration.py`)
Setup: hand points down, so fe1 was first bent to 49° (`move_joint_and_release.py`; friction held
it there with torque off), scale flat underneath, ~4 mm gap. fe1 then OPENED 0.6° per 4 s step
(`DIRECTION=-1 MAX_STEPS=25 NONINTERACTIVE=1`), pressing the **back** of the finger into the scale,
until load hit 300, then stepped back. fe2 held with torque on. Scale read off the operator's video
(IMG_7631.MOV, 1 fps, `load_scale_cal2_video_readings.csv`); clocks aligned via file timestamps and
confirmed by force jumps coinciding with position slips to within ~1 s.
Plot: `load_scale_cal2_2026-10-27.png`. (Run 1, `load_scale_cal_2026-10-27.csv`, curled sideways
above the scale and never touched it: free-motion stick-slip data only.)

**Result: scale force follows joint POSITION, not present_load.**
- The force rose in jumps (0 → 18 → 81 → 97 → 148 g) exactly when the joint slipped by one or two
  counts (0.3–0.6°). Between slips the joint was stuck and force stayed flat while load kept rising.
- At ~143 g, load ranged **225 → 300 → 210** with no change in force. At ~96 g, load 165–225.
  So one force reading maps to a load band at least ~90 units wide; the gear train's static
  friction absorbs the difference.
- Force crept down 148 → 141.7 g over ~26 s at fixed position (plastic/scale creep).
- fe2 load stayed exactly 0 throughout, though the press acts through the distal link.
- Unloading (camera rotated ~120° from ~110 s; frames de-rotated, `load_scale_cal2_paired.csv`):
  force went UP to ~161 g during the camera move (~110–115 s) with no logged position change —
  unexplained, possibly a bump — then held ~159 g while load fell 180 → 75 → **0**.
- **159 g on the scale with load = 0** (t ≈ 132–140 s), and **~72 g with load −60 to −90**
  (servo pulling away, t ≈ 144–152 s). Force only dropped when the joint slipped (→ 69 g at 141 s,
  → 0 g at 153 s, lift-off).
- So the joint is effectively **non-backdrivable** at these forces: friction alone holds up to
  ~160 g at the contact with zero or reversed motor effort.
- Lever arm (fe1 axis → contact, horizontal) not yet measured, so no torque numbers yet.

Implication: present_load cannot be turned into contact force with a single gain. Static
friction inside the servo means effort and output torque differ by up to ~±100 load units.
Upper-envelope points at slip onset (load just before slip → force after): 120 → ~81 g,
150–165 → ~97 g, 225 → ~146 g.

### Run 3 (camera fixed the whole time) — `load_scale_cal3_*`, IMG_7632.MOV
Same protocol, started at fe1 47.17°. Scale read at 1 fps (`load_scale_cal3_video_readings.csv`),
paired with the log (`load_scale_cal3_paired.csv`); every force jump lines up with a logged
position slip to within ~1 s. Plot: `load_scale_cal3_2026-10-27.png`. Confirms run 2 cleanly:
- Force rises only at slips: 0 → 2 → 31 → 35 → 120 → 135 → 180 g. Peak plateau ~167–172 g.
- **Force held at ~167–172 g for ~50 s while load went 270 → 300 → 0**, including ~10 s at load 0.
- After the first unloading slip: ~76–80 g at load 0 → −60. After the second: ~10–18 g at load
  −60 to −120 (servo pulling away). Lift-off only at the third slip.
- Force is a near-deterministic function of joint position (contact stiffness ≈ 80 g/deg here),
  with small loading/unloading hysteresis: 43.36° → 100 g loading / 80 g unloading;
  43.95° → 34 g / 17 g; 42.19° → 172 g / 167 g.
- Peak force repeatable with run 2 (≈160 g vs ≈170 g at the same load cap).
- Viscoelastic creep: after each slip into the scale the force overshoots and relaxes
  (e.g. 120 → 100 g over 12 s, 180 → 166 g over 50 s); after each slip away it recovers
  slowly (76 → 80 g, 10 → 18 g). Likely the finger plastic and/or the scale's load cell.

### Lever arm and torque (from the v3 CAD link meshes)
`hand-main/assets/mjcf/real_v3/`: the MorphIt source meshes `morphit/links/{mid,dist}.stl` are in the
same frames as bodies `index_mcp_frame` / `index_pip_frame` (checked: `correspondence.npz` surface
points sit ≤0.7 mm from the mesh vertices). Posed index_mcp = hardware fe1, index_pip = fe2 (−2.34°)
and took the lowest mesh vertex as the contact with the flat scale. It is on the **back of the
distal link (`dist.stl`)**, ~58 mm below the knuckle axis; the TPU pad (spheres only, no mesh) sits
~4 mm higher, consistent with the operator seeing back-of-finger contact.
Horizontal lever arm from the fe1 axis: **35.9 mm at 42.19°**, 36.5 at 42.77°, 37.1 at 43.36°,
37.7 at 43.95°, 40.9 at 47.17° (sphere fit gave 36.3 / – / 37.4 / 38.0 / 40.9). Assumes model mcp
sign = hardware fe1 sign (matching −15° limits support this).

Torque at each loading slip (run 3; force read just after the slip, load just before):
| load before slip | force after | lever | fe1 torque |
|---|---|---|---|
| 150 | 102 g | 37.1 mm | 37 mN·m (0.38 kgf·cm) |
| 210 | 132 g | 36.5 mm | 47 mN·m (0.48 kgf·cm) |
| 270 | 172 g | 35.9 mm | 61 mN·m (0.62 kgf·cm) |
**Breakaway torque ≈ 0.23–0.25 mN·m per load unit.** Extrapolated to load 1000 that is
~2.4 kgf·cm, close to the SCS0009's rated stall torque (2.3 kgf·cm at 6 V; we run at ~5 V),
consistent with load being duty cycle in 0.1 % units. This is an upper envelope only: between
slips, torque at the contact can sit anywhere from this value down to ~0 (or below) at the same load.

## 2026-10-29: free-motion friction sweep, all 9 joints (`friction_sweep.py`)
Each joint in turn, only that servo's torque on: triangle wave ±4° around its start pose at 1°/s,
speed 80. No contact. No aborts, torque off after every joint, 0 timeouts / 13,424, 19–22 °C.
Hand in the same pose as the scale runs (pointing down). Raw: `friction_sweep_raw_2026-10-29.txt`;
samples `friction_sweep_2026-10-29.csv`; per-leg stats `friction_sweep_2026-10-29_summary.csv`
(`analyze_friction_sweep.py`); plot `friction_sweep_2026-10-29.png`.
**Sampling was only 31 Hz** (FTDI latency still 16 ms; each sample is two reads), which is enough
for breakaway load (stick periods last 0.5–2 s) but does not resolve the slips themselves.

Every joint behaves like index fe1: load ramps up in steps of 15 while the joint is stuck, the joint
slips 1–2 counts (0.29–0.59°), load drops, and the cycle repeats. Nothing tracks the ramp smoothly.

| joint | breakaway load, + / − (median) | lag while moving, median / max (deg) | reversal dead band, + / − (deg) |
|---|---|---|---|
| thumb aa | −90 / 75 | 0.8–1.4 / 2.4 | 1.8 / 2.9 |
| thumb fe1 | −68 / 75 | 0.6–0.9 / 1.8 | 1.5 / 2.5 |
| thumb fe2 | −75 / 82 | 0.8–1.0 / 1.8 | 1.0 / 2.4 |
| index aa | **−135 / 135** (up to 210) | **1.6 / 3.3** | 2.1 / **4.3** |
| index fe1 | −60 / 60 | 0.5–0.8 / 1.8 | 0.9 / 1.5 |
| index fe2 | −75 / 90 | 0.7–1.2 / 1.8 | 0.6 / 3.0 |
| middle aa | −105 / 120 | 1.3 / 2.5 | 1.8 / 2.9 |
| middle fe1 | −60 / 60 | 0.4–0.7 / 1.5 | 0.9 / 1.2 |
| middle fe2 | −75 / 68 | 0.6–1.0 / 2.4 | 0.9 / 0.2 |

(+ = the "up" leg, center → +4°; − = the "down" leg, +4° → −4°. Load sign as in the step tests.)
- **Flexion joints (fe1, fe2) are consistent:** breakaway load 60–90, lag ~0.5–1°, max ~1.8°.
  At ~0.24 mN·m per load unit (index fe1 scale result, upper envelope) that is roughly
  **15–22 mN·m of static friction** per joint.
- **Adduction joints (aa) have more friction:** breakaway 90–135, index aa up to 210, lag up to
  3.3°, reversal dead band up to 4.3°. Roughly 22–32 mN·m (index aa up to ~50). The aa joint
  likely carries extra drivetrain or bearing load; worth checking the mechanism.
- **Lag is about breakaway / 15 counts**, e.g. 90 load ≈ 6 counts ≈ 1.8°, matching load ≈ P × error.
- **Reversal dead band 0.2–4.3°:** after the target reverses, the joint stays put until the error
  builds past breakaway the other way. A policy sees this as backlash.
- Several fe1/fe2 slips start with load at 0 (the joint drifts one count with no effort), and the
  index/middle fe1 "down" legs have median load 0. Gravity in this pose, or the controller's
  deadband, are candidates; not resolved.
- Breakaway values are multiples of 15 because load is quantized at P = 15, so ±15 is the
  resolution of every number above.

Next: compare with the BAM (Better Actuator Models) fit of the bare SCS0009 from the pendulum rig.
Friction in the hand minus bench friction = what the finger mechanism adds per joint.

## 2026-10-29: full-range friction sweep (`full_range_sweep.py`)
Each joint alone, torque on only for it: ramp at 2°/s from 3° inside the low software limit to 3°
inside the high one and back. **Adduction (aa) joints were held to ±10°** (limits still
unconfirmed). FTDI latency now **1 ms → 500 Hz** position + load. Index fe1 ran first as a trial
(`full_range_raw_2026-10-29_index_fe1.txt`), the other 8 after (`full_range_raw_2026-10-29_rest.txt`).
0 timeouts / 540,225. Analysis: `analyze_full_range.py` → `full_range_2026-10-29_bins.csv` and
`full_range_2026-10-29.png` (10° bins).

Method: at each angle, friction = (mean load going down − mean load going up) / 2, and
bias = their average (gravity, cable or spring preload, controller offset). Means rather than
medians, because load is quantized in steps of 15.

| joint | swept (deg) | friction load across range | trend with flexion | bias (gravity etc.) |
|---|---|---|---|---|
| thumb aa | −10…+10 | 86 | — | +11 |
| thumb fe1 | −9…+86 | 57–70 | flat | +15 → **−46** at 77° |
| thumb fe2 | −15…+84 | **78 → 151** | **doubles from 0° to 40°**, then 115–140 | −12…+18 |
| index aa | −10…+10 | **150** | — | +10 |
| index fe1 | −12…+62 | 48–62 | flat, slightly up | +4 → −29 at 45° |
| index fe2 | 0…+89 | 85 → 103 | slight rise | ~0 |
| middle aa | −10…+10 | 100 | — | +33 |
| middle fe1 | −12…+67 (rerun) | 40–60 below 18°; **172–174 at 20–24°**; 85–130 above 26° | **tight spot at 20–24°**, ~2× higher above it | +27 → −56 |
| middle fe2 | −13…+74 | 69 → 93 | slight rise | −12…+6 |

- **Friction is roughly position-independent for most joints:** within ±15% across the range for
  thumb fe1, index fe1 and index fe2, with a mild rise (~20–30%) toward flexion for the fe2 joints.
  A constant per-joint friction is a reasonable first model.
- **Exception: thumb fe2.** Friction doubles between 0° and 40° (78 → 151) and stays high. Possibly
  a tight spot in the linkage or cable path; worth a look by hand.
- **Gravity shows up only on fe1** (as expected with the hand pointing down: fe1 carries the whole
  finger). Bias goes from ~0 near straight to −30…−46 at 45–77°. On fe2 it stays within ±15.
  Hand orientation matters: these numbers are for this pose only.
- **Adduction friction is high** even within ±10° (86–150 load), matching the small sweep;
  index aa is the stickiest joint on the hand.
- Compared with the ±4° sweep (breakaway 60–90 on fe joints): mean-load friction here is similar,
  so the small sweep was representative near zero.

**Middle fe1: friction, not contact.** The first pass aborted on the jam guard at ~22° (target 27.8°;
load to −300, peak −435). The operator found nothing touching the finger, and the rerun
(`full_range_raw_2026-10-29_middle_fe1.txt`; 2° bins in `full_range_raw_2026-10-29_middle_fe1_bins.csv`)
completed the full range. It shows a **tight spot at 20–24°**: friction rises from ~50 to ~174 load
units, **in both directions** (contact would show up as one-sided bias, not as friction), with lag up
to 5.9° going up. Above 26° friction stays at ~85–130, about double the value below 18°. The 5° /
0.6 s jam guard sits close to this spot's lag, so future middle fe1 runs may trip it again; a
position-dependent friction term is needed for this joint.

**Index fe2** shows a load spike to ~−200 right at the top of its sweep (~85–89°). Not yet checked
whether that is contact near the upper limit or another tight spot.

### Second full-range pass: repeatability and sticking spots
Same protocol, all 9 joints (`full_range_pass2_raw_2026-10-29_{a,b}.txt`; middle fe1 run with
`JAM_DEG=7` so its tight spot doesn't trip the lag guard). No aborts; 1 corrupted reply (CRC) in
~690,000 reads. Two earlier pass-2 attempts were cut short by cable swaps (USB adapter, now
`/dev/ttyUSB1`, latency re-set to 1 ms); their partial logs are in `aborted/`.
Comparison: `compare_full_range_passes.py` → `full_range_passes_2026-10-29.{csv,png}` (2° bins).

| joint | median friction, pass 1 / 2 | profile correlation r | repeatable features |
|---|---|---|---|
| thumb aa | 93 / 93 | 0.89 | (±10° only) |
| thumb fe1 | 67 / 64 | 0.57 | flat (end-bin rise at ~81° did not repeat in pass 3) |
| thumb fe2 | 116 / 117 | 0.87 | **banded: ~80–100 below 20°, ~140–170 at 22–48°, dip to ~90 at 50–56°, ~130–170 at 60–80°** |
| index aa | 158 / 147 | 0.16 | flat (±10° only) |
| index fe1 | 57 / 55 | 0.56 | flat, mild rise above 30° |
| index fe2 | 94 / 95 | 0.79 | slow rise 80 → ~100 (the ~118 top bin is an end-of-sweep artifact) |
| middle aa | 124 / 119 | 0.51 | flat (±10° only) |
| middle fe1 | 88 / 88 | **0.94** | **tight spot 21–27° (~170, 3.5× the region below), decays to ~90 by 45°** |
| middle fe2 | 75 / 75 | 0.64 | slow rise 65 → 100 |

- **Friction is repeatable.** Medians agree within ~7% between passes, and joints with structure
  in their profile (thumb fe2, middle fe1, index fe2) reproduce it closely (r 0.79–0.94). The flat
  joints have low r only because there is little variation to correlate.
- **Sticking spots are fixed in angle.** The operator attributes them to print errors. Middle fe1
  (21–27°) is the only spot above 1.5× the joint's median in both passes; thumb fe2 has a repeatable
  banded profile, and thumb fe1 and index fe2 stiffen near their upper limits.
- The index fe2 −200 load spike seen at 85–89° on pass 1 lies outside the bins both passes cover;
  in the shared range it shows only as the ~118 rise at the top.
- **Model implication:** a constant friction per joint is fine for thumb fe1, index fe1/fe2 and
  middle fe2 (±20%). Thumb fe2 and middle fe1 need friction as a function of angle; the averaged
  2° bins in `full_range_passes_2026-10-29.csv` can serve as a lookup table. These are this hand's
  print-specific values, so a reprinted part would need re-measuring.

### Third full-range pass (sanity check)
Same protocol (`full_range_pass3_raw_2026-10-29_{a,b}.txt`), all 9 joints, no aborts, 0 timeouts
/ 683,718. Compared with pass 1 and pass 2 (`full_range_passes_{1v3,2v3}_2026-10-29.{csv,png}`):
- **Medians match:** pass 3 is within 5% of passes 1 and 2 on every flexion joint (e.g. thumb fe2
  116 / 117 / 112, middle fe1 88 / 88 / 89, index fe2 94 / 95 / 93). Index aa varies most
  (158 / 147 / 144), still under 10%.
- **Middle fe1's tight spot repeats a third time** at 21–29° (r = 0.95 against both passes).
  Thumb fe2's banded profile repeats too (r = 0.82–0.85).
- **Correction to the pass-2 notes:** the "stiffening near the upper limit" of thumb fe1 (~81°) and
  index fe2 (~84°) does **not** repeat. Pass 3 reads thumb fe1's last bin at 40 vs 95 / 100, and index
  fe2 is flat at ~100 through its top bins in all three passes. The last bin of each sweep includes
  the reversal, so treat single end-bin values as artifacts. Neither joint has a sticking spot.
- Low r on flat joints (thumb fe1 ≈ 0, index fe1 0.3–0.5) is expected: there is no structure to
  correlate, only bin-to-bin noise of about ±10 load units.

Final picture: two print-related friction features (middle fe1 21–29°, thumb fe2 bands at
22–48° and 60–80°); every other joint is a constant friction value within about ±20%.

## 2026-10-29: full-hand control-loop rate (`loop_rate_test.py`, test 7)
All nine servos torque on, each following a ±1° 0.5 Hz sine around its pose. Per cycle: one
`sync_set_joints` write of all nine goals, then per-servo reads. FTDI latency 1 ms (`/dev/ttyUSB1`).
10 s per variant, 0 timeouts / 30,627, 22 °C max. Raw: `loop_rate_raw_2026-10-29.txt`.

| variant | rate | cycle time (median / max) | breakdown |
|---|---|---|---|
| position only, free-running | **87 Hz** | 11.2 / 12.2 ms | write 0.25 ms, 9 position reads 10.8 ms (incl. the 2 ms post-write gap) |
| position + load, free-running | **48 Hz** | 21.0 / 22.0 ms | + 9 load reads 9.0 ms |
| position + load, paced at 50 Hz | 47.8 Hz | 21.0 ms | **misses nearly every deadline**: 50 Hz is not reachable this way |
| position + load, paced at 30 Hz | 30.1 Hz | 33.3 / 34.0 ms | 0 misses, ~12 ms slack |

Each per-servo read costs ~1 ms (the FTDI latency floor), so the loop cost is 1 ms × servos × fields.
Sync read doesn't work on this hardware (driver notes), so the only way to go faster is fewer reads.

### One read per servo for position + speed + load (`combined_read_check.py`)
On the SCS0009, present position (0x38), speed (0x3A) and load (0x3C) are contiguous, and rustypot
exposes `read_raw_data(id, addr, len)`. One 6-byte read at 0x38 returns all three:
- position counts = b0·256 + b1 (big-endian); deg = (counts − 511) × 300/1024, identical to the driver
  on all 9 servos;
- speed = b2·256 + b3, sign bit 15; load = b4·256 + b5, sign bit 10, magnitude in the low 10 bits.
- Checked under torque on index fe1 (±3° sine, 1,150 samples): position agrees in 100% of samples;
  load equal in 94%, and every mismatch is a timing difference (the separate reads are 1–2 ms later
  while load is changing), never a sign disagreement. 0 timeouts / 4,605.
- Read-only timing: 450 combined reads in 450 ms = **111 full-hand cycles/s** for position + speed +
  load. With the sync write and its 2 ms gap, expect ~85 Hz, so **50 Hz with ~8 ms of slack**.

**Implication for training:** with the current driver (separate position and load reads), the real
loop tops out at 48 Hz, so train at **≤ 30–40 Hz** or switch the driver to the combined read, which
makes 50 Hz comfortable and adds speed for free. Observation delay is roughly one read pass: ~11 ms
(combined) or ~21 ms (separate) from the goal write to the last servo's feedback.

## 2026-10-29: servo gain sweep, index fe1 (`gain_sweep.py`)
Why: small, slow motions (±2° sine at 0.5 Hz, goals at 50 Hz) looked jittery on video (IMG_7664).
The goal is not a fixed angle; it is a smooth sine, but it reaches the servo quantized to whole
counts (0.29°), a staircase of one count every few updates. The bigger effect is friction: the
servo's P controller has to build 1–2° of error before it breaks away, then jumps.

Gain registers: P 0x15 = 15, D 0x16 = 15, I 0x17 = 0 on every servo (health check). The lock flag
(0x30) reads 1; it was left untouched. Every write was read back, and the script restores the
original gains in `finally` (confirmed 15/0/15 afterwards). Whether a write with lock = 1
survives a power cycle is NOT confirmed on the SCS0009.

Protocol per setting: ±2° 0.5 Hz sine for 10 s, then 4 s hold at center. One 6-byte feedback read
per sample, ~890 Hz. Raw `gain_sweep_raw_2026-10-29_index_fe1.txt`; summary `..._summary.csv`
(`analyze_gain_sweep.py`); plot `gain_sweep_raw_2026-10-29_index_fe1.png`. 0 timeouts / 87,171,
22–23 °C, no oscillation at any setting.

| P / I / D | tracking RMS / max | moves/s | hold error / hold load |
|---|---|---|---|
| 15 / 0 / 15 (factory, run first) | 0.89° / 2.32° | 11.5 | 0.88° / 75 |
| 30 / 0 / 15 | **0.58° / 1.28°** | 13.5 | **0.29° / 0** |
| 45 / 0 / 15 | **0.51° / 1.71°** | 16.6 | **0.00° / 0** |
| 15 / 2 / 15 | 0.94° / 1.89° | 10.8 | 0.29° / 46 |
| 15 / 5 / 15 | 0.84° / 2.49° | 15.1 | 0.59° / 63 |
| 30 / 2 / 15 | 0.49° / 1.40° | 17.1 | 0.29° / 48 |
| 15 / 0 / 15 (factory, run last) | 0.87° / 2.32° | 10.8 | 0.59° / 0 |

- **Raising P is what helps.** P 30 cuts tracking error ~35% and nearly halves the worst lag; the
  joint makes more frequent one-count steps instead of stalling for up to ~0.6 s and lurching.
  It also settles to within one count when holding, with zero holding load.
- **I does not help** at P 15 (I 2 ≈ baseline; I 5 worsens the worst lag) and adds holding load.
- Steps stay one count (0.29°) at every setting: that is the position resolution, so motion can't
  get smoother than one-count steps. Higher P makes those steps regular rather than bursty.
- Single-sample position spikes (e.g. −3.8° for one sample) appear in every setting: read tears,
  not motion (same as the 2026-10-27 notes).
- Factory runs first and last match, so no drift or heating effect over the ~2 min test.

Not yet known: whether P 30–45 holds up on the stickier joints (middle fe1, thumb fe2, the aa
joints), under larger and faster moves, and over long holds (heating). The simulator must use the
same gains as the hand; the BAM pendulum data includes kp 15 and kp 4 runs, so a fit that captures
the kp dependence could be checked at the new value.

**Decision (2026-10-29): keep the factory gains (P 15 / I 0 / D 15).** The hand worked at these
gains, and every characterization so far (friction sweeps, loop rate, the BAM pendulum data) is at
P 15; changing them would mean re-identifying everything. The jitter seen at ±2° is the
friction we measured, and the simulator should reproduce it rather than the hand be tuned around
it. The gain sweep on the sticky joints was stopped part-way; stopping it killed the script before
its restore step, so servo 7 (middle fe1) was left at P 30 and was set back to 15 by hand.
All nine servos confirmed at 15/0/15 afterwards. Partial logs: `gain_sweep_raw_2026-10-29_middle_fe1.txt`.

### Double pendulum at P 30 / 45, D 0 (bam_pendulum/RESULTS.md)
On the pendulum rig, at D = 0 (the BAM fitting setting), raising P made joint 1 limit-cycle by
~±5° where it hangs unloaded near plumb (backlash), and heat at P 45 (21 → 37 °C, full PWM);
joint 2 tracked better. The BAM fits don't reproduce the oscillation (tip error 1.8 mm at P 15 →
3.5 / 8.0 mm at P 30 / 45). The hand runs D = 15, which damps this; deciding on higher P needs
D = 15 pendulum runs, which the BAM model also needs.

### 2026-10-29: ±15° demo, index fe1, factory vs P 30 (`gain_demo_raw_2026-10-29_index_fe1_amp15.txt`)
`JOINT=1:fe1 AMP=15 CENTER=20 SETTINGS=15/0/15,30/0/15 python3 gain_sweep.py`; 0 timeouts, 23 °C, gains restored.
The user, watching, said the factory run was "way smoother". The data agrees:

| P / I / D | tracking RMS / max | stalled while moving | velocity roughness | velocity sign flips |
|---|---|---|---|---|
| 15 / 0 / 15 | 1.80° / 3.60° | 20% | 21.6°/s | 7.9 /s |
| 30 / 0 / 15 | 1.21° / 3.01° | 24% | 28.5°/s | 16.9 /s |

P 30 tracks closer but chatters: twice the velocity reversals, the same effect as the pendulum's P 30
limit cycle. Better tracking is not smoother motion. Factory gains stay (consistent with the decision above).

### 2026-10-29: ±15° demo, factory vs P 15 / D 30 (`gain_demo_raw_2026-10-29_index_fe1_amp15_d30.txt`)
Factory first, then the candidate (the user's rule for every gain test). Scored with `smoothness.py`.

| P / I / D | tracking RMS / max | stalled while moving | velocity roughness | velocity sign flips |
|---|---|---|---|---|
| 15 / 0 / 15 | 1.80° / 3.56° | 21% | 20.3°/s | 6.8 /s |
| 15 / 0 / 30 | 1.97° / 3.76° | 20% | 20.9°/s | 10.9 /s |

D 30 is not smoother and buzzed during the hold: the oscillation guard tripped (7 reversals in ~1 s),
torque off, gains restored to 15/0/15. Likely the D term acting on a 1-count (0.29°) velocity signal.
Neither P 30 nor D 30 beats factory; smoother motion has to come from the commands, not the gains.

### 2026-10-29: direction asymmetry and friction feed-forward, index fe1 ±15°, factory gains
The user saw flexion smooth and **extension jitter at every setting**. Index fe1 has no cable or spring
(direct servo drive), so it is the servo. Split by direction (angle increasing = flexion):

| run (factory P15/I0/D15) | flexion: roughness / stalled | extension: roughness / stalled | tracking RMS |
|---|---|---|---|
| no lead | 16.9°/s / 15% | 25.7°/s / 24% | 1.91° |
| lead 1° (`LEAD=1`) | **14.7°/s / 7%** | 28.9°/s / 26% | **1.13°** |

Lead = goal + 1°·tanh(target velocity / 10°/s) (`gain_sweep.py` env `LEAD`, `LEAD_V`; scored against
the un-led target). Raw: `gain_demo_raw_2026-10-29_index_fe1_amp15_lead0.txt`, `..._lead1.txt`.
- Extension is ~55% rougher and stalls ~60% more, in every run (P15, P30, D30, lead).
- Same on BAM pendulum servo 20 (no linkage): moving + against gravity 19°/s / 9% stalled; every
  other case (helping load, or moving −) 28–29°/s / 16–33%. So: one direction of the SCS0009 is
  intrinsically rougher, and a load that helps the motion makes either direction rough.
- The 1° lead halves flexion stalls and cuts tracking error 40%, but does **not** fix extension.
- Paused here (user). Open ideas: a bigger or extension-only lead; faster goals (200 Hz); BAM's
  directional model (M5) fitted with hand data so the simulator reproduces the asymmetry.

## 2026-10-29: gain optimization, pendulum rig + hand
Goal: servo P/D that tracks best without oscillating or heating, and that BAM can still model.
I stays 0 throughout (it didn't help on index fe1). Factory = P 15 / D 15.

### Double pendulum (bam_pendulum, `compare_gains.py`; 4 paths per setting, ~270 Hz)
"jitter" = RMS of position minus its own 0.25 s moving average: oscillation that doesn't follow
the goal (square-wave path excluded from it). Joint 1 = servo 20 (carries link 2), joint 2 = servo 21.

| P / D | RMS error j1 / j2 | jitter j1 / j2 | j1 at full PWM |
|---|---|---|---|
| 15 / 15 (factory) | 2.05 / 2.85° | 0.66 / 0.37° | 0% |
| 22 / 15 | 1.86 / 2.12° | 0.78 / 0.39° | 0% |
| 22 / 45 | 1.88 / 2.27° | 0.82 / 0.36° | 0% |
| 26 / 90 | 1.81 / 2.20° | 0.90 / 0.39° | 0% |
| 30 / 15 | 1.88 / 1.66° | 1.93 / 0.72° | 0% |
| 30 / 45 | 1.70 / 1.76° | 1.23 / 0.45° | 0% |
| 30 / 90 | 1.74 / 1.98° | 0.93 / 0.40° | 0% |
| 30 / 150 | 1.98 / 2.43° | 1.06 / 0.52° | 0.2% |
| 45 / 15 | 3.21 / 1.53° | 3.80 / 1.38° | 2.1% |

- Joint 1 limit-cycles near plumb (unloaded gear backlash) once P ≥ 30; D damps it, best at
  D ≈ 90; D 150 is worse again. P 45 is out (oscillation, full PWM, heating to 37 °C).
- All rig recordings ended with both servos restored to 15/0/15 (checked).

### Hand (`gain_sweep.py` → `gain_confirm_raw_2026-10-29_*.txt`; `analyze_gain_sweep.py`)
Each joint alone: ±2° (the jittery case) and ±8° 0.5 Hz sines (±4° for index aa, ±10° rule), then
a 6 s hold. Gains restored to 15/0/15 after every run (confirmed in each log).
RMS tracking error (jitter on the hand stayed 0.2° at ±2° and 0.4° at ±8° for every setting):

| joint, amplitude | 15/15 | 22/15 | 30/90 | 30/45 |
|---|---|---|---|---|
| index fe1, ±2° | 0.96° | 0.69° | 0.57° | 0.59° |
| index fe1, ±8° | 1.16° | 0.97° | 0.92° | 0.88° |
| middle fe1, ±2° | 0.85° | 0.65° | 0.54° | 0.55° |
| middle fe1, ±8° | 1.32° | 1.04° | 0.99° | 0.93° |
| thumb fe2, ±2° | 1.11° | 0.89° | 0.71° | 0.72° |
| thumb fe2, ±8° | 1.55° | 1.21° | 1.09° | 1.06° |
| index aa, ±2° | 1.45° | 1.22° | **oscillated in the hold (aborted)** | not run |
| index aa, ±4° | 1.94° | 1.48° | 1.17° | 1.17° |

20–23 °C throughout; no heating at any setting over these short runs.

### Direction reversals (the smoothness the eye sees)
RMS error and the 0.25 s jitter measure miss one-count back-and-forth flicks. Counting direction
reversals of position per second during the sines (same runs):

| joint, amplitude | 15/15 | 22/15 | 30/90 | 30/45 | 15/15 again |
|---|---|---|---|---|---|
| index fe1, ±2° | 1.9 | 2.8 | 3.6 | 3.6 | 2.3 |
| index fe1, ±8° | 1.9 | 3.2 | 4.6 | 3.6 | 2.2 |
| middle fe1, ±2° | 2.5 | 3.2 | 1.8 | 2.2 | 3.3 |
| middle fe1, ±8° | 2.0 | 3.2 | 3.2 | 2.2 | 2.6 |
| thumb fe2, ±2° | 1.9 | 3.3 | 3.8 | 3.1 | 2.5 |
| thumb fe2, ±8° | 2.2 | 3.4 | 4.4 | 3.2 | 2.4 |
| index aa, ±4° | 3.2 | 3.8 | 2.6 | 3.0 | 3.4 |

Higher P trades long stalls for frequent small flicks: tracking error drops, reversals rise
~1.5× at P 22 and ~2× at P 30 on the flexion joints. This matches the operator's visual judgment
in the other session (index fe1, ±15°, factory first): factory "way smoother" than P 30 / D 15,
P 15 / D 30 similar to factory but buzzed in the hold (see the sections from session 29b373 above).

### Conclusion: keep the factory gains (P 15 / I 0 / D 15)
- No P/D setting found is smoother than factory. Every setting that tracks better does it by
  flicking more, and P ≥ 30 also oscillates on index aa and the rig's joint 1 unless D is raised.
- Tracking can instead be improved without touching the servo: the friction feed-forward lead
  tested by the other session (factory gains + 1° lead) halved flexion stalls and cut tracking
  error ~40%. Extension stays rough at every setting; it is the SCS0009 itself (same asymmetry on
  the BAM rig's servo 20).
- So the gains stay at factory, the BAM fit stays valid where it was fitted, and the simulator
  should reproduce factory-gain stick-slip, including the flexion/extension asymmetry.
- Rule from the user (via the other session): every gain/tuning test runs factory first, then
  the tuned setting, with the same motion. All runs above did.

## 2026-10-29: hand replay gate for policy C (`hand_replay.py`)
Replays the full-range sweeps (one joint at a time, 2°/s, P 15 / D 15) in MuJoCo: hand-main real_v3
(CAD masses, hand pointing down, contacts off, the other joints pinned at their rest angles), each
joint driven by the BAM SCS0009 actuator fitted on the pendulum (servo 20), with the D = 15 term and
**encoder quantization** (the firmware sees whole 0.29° counts, which reproduces the real load
flicker to 0 after each count step). Per-joint friction scales calibrated on pass 3
(`--calibrate` → `hand_friction_scales.json`), checked on pass 2 (held out). Plot `hand_replay_pass2_scaled.png`.

Held-out pass 2, M6, quantized, with scales:

| joint | position MAE | load corr | load MAE | friction real / sim | gravity+bias real / sim |
|---|---|---|---|---|---|
| thumb yaw | 0.42° | 0.92 | 26 | 83 / 87 | −8…+28 / −3…+12 |
| thumb mcp | 0.47° | 0.83 | 32 | 63 / 56 | −4…+59 / −3…+43 |
| thumb pip | 0.79° | 0.93 | 41 | 119 / 91 | −13…+19 / −1…+4 |
| index yaw | 1.28° | 0.85 | 71 | 120 / 93 | −30…+73 / −7…+9 |
| index mcp | 0.58° | 0.76 | 38 | 55 / 50 | +4…+38 / −7…+36 |
| index pip | 0.41° | 0.92 | 22 | 94 / 91 | −3…+4 / 0…+4 |
| middle yaw | 0.65° | 0.91 | 36 | 105 / 93 | −15…+39 / −7…+9 |
| middle pip | 0.51° | 0.86 | 35 | 77 / 74 | −3…+13 / 0…+12 |
| middle mcp | 0.64° | 0.83 | 42 | 83 / 83 | −4…+71 / −10…+30 |

Load units = PWM × 1000; typical |load| while moving ~100. Without the scales, load MAE was worse
on 6 of 9 joints (index yaw 86, middle yaw 52, thumb pip 50).
- **Passes for the flexion joints:** position ≤ 0.8°, load correlation 0.76–0.93, friction within ~10%.
- **Yaw (aa) joints are under-modelled:** stickier than servo 20 and the friction scale saturates
  (index yaw 120 real vs 93 sim at scale 1.74). Load-dependent friction terms don't grow with
  a constant scale; a per-joint fit of the friction parameters would.
- **Gravity on thumb / middle mcp is low in sim** (real up to +59 / +71, sim +43 / +30; index matches).
  Likely the uncalibrated v3 zero / ROM (the middle finger's CAD pose has MCP at about −29°) or link masses.
- **Not yet tested: contact.** These sweeps are free air. Policy C's load signal matters most in
  contact; the next check replays the scale-press logs (`load_scale_calibration.py`).

## 2026-10-30: joint zero check + handoff snapshot (`zero_check.py`, `rs_grab.py`)
Before handing the hand to a collaborator. Camera: RealSense D435 on the CB1 (`/dev/video5` = color;
`rs_grab.py` grabs raw YUYV with the standard library only, convert on the Mac with ffmpeg), low and
looking along the cylinder axis. Frames: session scratchpad `zero/p1.jpg` (zero), `b_{thumb,index,middle}.jpg`.
- **Move 1, all joints to 0** (ramped 10°/s; the firmware jumps straight to a new goal, so a raw 10°
  step reads load ~520 and tripped the 500 abort on the first try). All three fingers straight and
  perpendicular to the palm, within ~±3° by eye.
- **Moves 2–4, one finger at MCP 30 / PIP 30** (encoder 27.5–29.3 / 28.4–29.6 after torque off).
  First link ~30°, second ~60° from vertical, ±5° by eye (camera looks up at the fingers). Flexion
  signs match the sim: the thumb curls toward the pair and the pair toward the thumb.
- **Conclusion:** no large zero error. The CAD-pose offset suspected from the replay gate (middle MCP
  ~−29°) is ruled out, so the low sim gravity on thumb / middle MCP points at link masses. Residual
  zero uncertainty ~±3–5° → randomize joint offsets ±3° in sim; the UR5 z correction covers
  fingertip height.
- **Two driver copies on the CB1 disagree on the middle yaw zero:** `~/hand/src/morphohand/driver/manta/host`
  (19.3359, yaw limits ±85, locally modified, uncommitted) vs `~/manta-hand/host` (12.89, ±75).
  A script run as a file imports the `~/hand` one; code run from inside `~/manta-hand/host` gets the
  other, and reads the middle yaw 6.45° off. All other joints match.
- Index fingers can be pushed far by hand with torque off (index PIP read 186.6° rel after handling,
  back to 8.5° once straightened), so re-read positions after anyone touches the hand.
- **Handoff health check** (read-only, `handoff_health_check_2026-10-30.txt`): all 9 servos status 0,
  5.0–5.3 V, 18–22 °C, P15/I0/D15, overload torque 80 / protection time 100, latency timer 1 ms,
  0 bus timeouts. All joints at 0 ±2°, torque off.
