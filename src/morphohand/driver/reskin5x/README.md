# reskin5x

Readout of the ReSkin 5X magnetometer board (five MLX90393; U1 at the centre, U2 to U5 7.00 mm from it in a plus) through
an Adafruit QT Py ESP32-S3, with a Qwiic cable from the board's J1 to the QT Py's STEMMA QT connector. The firmware is the
eFlesh sketch ported to this board. Its record format is eFlesh's and AnySkin's, so the eFlesh visualizer in the
`external/eFlesh` submodule (the owner's fork, github.com/badinkajink/eFlesh) reads it.

The workstation's QT Py has 4 MB of flash and 2 MB of PSRAM (esptool: "Embedded Flash 4MB (XMC), Embedded PSRAM 2MB"),
and `flash.sh` builds for that board, `esp32:esp32:adafruit_qtpy_esp32s3_n4r2`. The 8 MB no-PSRAM QT Py has the same pins
and builds with `RESKIN5X_FQBN=esp32:esp32:adafruit_qtpy_esp32s3_nopsram`; an image for one does not flash on the other.

## Layout

- `firmware/reskin5x_stream/reskin5x_stream.ino`: the port of `external/eFlesh/arduino/5X_eflesh_stream/5X_eflesh_stream.ino`.
  The STEMMA QT connector is the board's `Wire1` (SDA1 GPIO41, SCL1 GPIO40), `Serial` is the TinyUSB CDC port, and the
  register settings are those below; the header comment lists every change.
- `firmware/reskin5x_mux_stream/reskin5x_mux_stream.ino`: several boards behind an I2C mux on the STEMMA QT port, one
  record of 80 bytes a board plus `\r\n` per sweep (section on three boards).
- `host/reskin5x_mux.py`: splits that stream into one 82-byte stream a board, each on a pseudo-terminal linked at
  `/tmp/reskin5x_mux/ch<channel>` that `host/wrench_viewer.py --port` reads, and logs all boards; `host/reskin5x_mux_check.py`
  gives a log's record rate, each chip's noise and the boards' separation, `host/reskin5x_mux_rest.py` each board's rest
  over time (split into a uniform field and the rest of the change), and `host/reskin5x_mux_soak.py` the viewer's frames
  replayed over fingers left at rest.
- `firmware/bench_three_board_sweep/`, `firmware/bench_conversion_period/`: timing sketches behind the numbers in the
  sections on rates and on three boards; they print `STAT`, `CONV` and `NACK` lines. Build and upload one with
  `RESKIN5X_SKETCH=<name> host/flash.sh`, and put `reskin5x_stream` back afterwards.
- `host/flash.sh`: compile and upload.
- `host/reskin5x_check.py`: per-chip field, noise, drift and temperature, the record rate, and the fraction of the range in
  use, from a few seconds of stream; `--out` writes the records as text.
- `host/reskin5x_plot.py`: live plot of each chip's change from a baseline (|ΔB| of all chips, then ΔBx, ΔBy, ΔBz per
  chip, µT, last 10 s); key `b` takes a new baseline.
- `scripts/reskin5x_pad_fit.py`: the magnets' polarities, position and strength fitted to a capture of the board with a
  pad on, and the sensing model's force resolution for the fit.

## Setup

    git submodule update --init external/eFlesh
    git -C external/eFlesh submodule update --init arduino/arduino-MLX90393

The fork's other submodule, visuoskin, has an SSH URL and is not needed. The build uses the esp32:esp32 core (3.3.2 on the
workstation, in `~/.arduino15`) and the arduino-cli inside the Arduino IDE 2 flatpak, which `flash.sh` finds. The host
scripts run with `~/miniconda3/bin/python` (anyskin 1.0.0, pygame 2.6, pyserial 3.5); elsewhere
`pip install anyskin pygame pyserial`.

## Flashing

    src/morphohand/driver/reskin5x/host/flash.sh            # upload to the first /dev/ttyACM*
    src/morphohand/driver/reskin5x/host/flash.sh build      # compile only

A board running this firmware resets into its bootloader over USB (the 1200-baud touch), takes the upload and restarts
with no button pressed. Adafruit's factory sketch ignores the touch. To flash such a board, hold BOOT, press and release
RESET, release BOOT, and run `flash.sh`. In the ROM bootloader the board runs nothing and its LED stays dark; it
enumerates as USB 303a:1001, "USB JTAG/serial debug unit". The RTS reset after the upload re-enters the bootloader because
BOOT's strap stays latched, so `flash.sh` follows the upload with esptool's watchdog reset. The board then enumerates as
239a:8143.

## Checking the stream

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_check.py --seconds 5

Noise is the standard deviation of each axis inside 0.5 s windows, averaged over the windows; drift is the largest range
of the 0.5 s means. Measured on 2026-10-20 at the workstation, board at rest and no magnet near it: 375 records/s; noise
1.54 to 1.76 µT on x and y and 2.32 to 2.59 µT on z; drift 0.8 to 1.7 µT; |B| 41 to 82 µT, Earth's field plus the chips'
offsets, which changes as the board turns; 28 to 31 °C. The noise model of `scripts/mlx90393_model.py` gives 1.20 and
1.84 µT at this setting, so the measured noise is 1.4 times the model's; a force resolution computed with the model scales
by the same factor (concept b's pad: 0.028 N at the contact point, about 0.039 N with the measured noise). A drift of tens
of µT means the board turned or a magnet moved near it during the capture: one capture taken while the board was handled
gave 40 µT on every chip.

## Sample rate on one board

The firmware reads the five chips in 2.0 ms (397 µs a chip at 400 kHz, the MLX90393's I2C limit), writes the 82-byte
record and waits 0.5 ms: 375 records/s. A chip converts T, X, Y and Z every 7.01 to 7.13 ms, 140 to 143 Hz (from the
stream, 2026-10-20; the datasheet's formula gives 7.0 ms), so each chip's values repeat in about 2.6 consecutive records
and its three axes change at different records. The conversion sets the rate of new data. DIG_FILT 4 converts in 4.0 ms
(250 Hz) and DIG_FILT 3 in 2.5 ms (400 Hz), at 1.4 and 2.0 times the noise by the datasheet's figures; without the wait,
one board's read limits the stream to about 460 records/s. Reading one chip back to back, every 0.40 ms, lengthens its
conversion to 7.55 ms (`firmware/bench_conversion_period`).

## Three boards on one QT Py

Every board of this design answers at the same five addresses: U1 is an MLX90393ELW-ABA-011 at 0x0C and U2 to U5 are
-012 parts at 0x10 to 0x13 (ReSkin's BOM). Three boards on this QT Py therefore need an I2C mux on the STEMMA QT port with
one channel per board, such as Adafruit's PCA9548 8-channel STEMMA QT board (address 0x70, 10 kΩ pull-ups on its
controller side). Each board has its own pull-ups, R1 and R2 beside J1 (10 kΩ in the pick-and-place file, 4.7 kΩ in the
BOM), so the mux channels need none. Timed on this QT Py with a sweep that emulates three boards
(`firmware/bench_three_board_sweep`): one chip's read 397 µs, a channel write about 190 µs, a 242-byte record over USB
0.28 ms, and a sweep of 15 chips 6.74 ms (at most 6.97 ms), 148 sweeps/s. The sweep is shorter than a chip's 7.0 to
7.1 ms conversion, so every conversion of all 15 chips is read, at 140 Hz a chip. The 15 chips draw about 37 mA in burst
mode (49 mA at the datasheet's maximum) from the QT Py's 600 mA regulator.

A transfer to an absent address takes 19.1 ms at Wire's 50 ms timeout and 20.0 ms at 1 ms, so a sweep that kept reading
an unplugged board would lose about 95 ms; the firmware reads only the chips it finds at start-up. The three-board
firmware writes the mux channel before each board's reads and sends 15 chips a record (242 bytes, which anyskin reads with
`num_mags=15`); `viz_eflesh.py` draws one board, so three boards need a view with three board images. The QT Py's other
I2C controller, on the SDA and SCL pads (GPIO7, GPIO6), can take a second board without a mux, and a third would still
need a mux or a third pin pair.

Measured on 2026-10-25 with three fingers on the channels 0, 1 and 3 of a mux at 0x70 on the workstation's QT Py
(`firmware/reskin5x_mux_stream`, `host/reskin5x_mux.py`, `host/reskin5x_mux_check.py`): 152 records/s, so each 7.0 ms
conversion of all 15 chips is read, and no zero records. Each chip's noise from one conversion to the next is 1.88 µT on
x and y and 2.89 µT on z (median of the 15 chips). The channel-3 finger gave 1.88 and 2.95 µT on its own board on
2026-10-23, chip by chip within 2 %, so the mux adds no noise. Between boards the readings' changes correlate by 0.007
(median over component pairs) and at most 0.031, under the 0.036 that independent noise reaches over 12,338 records, and
no board repeated another's values. The rest fit of the channel-3 finger through the mux matches its fit on its own board
to 0.01 mm and 0.003 in strength. A faster processor or FreeRTOS tasks would not raise the rate: 6.0 of the 6.6 ms sweep
is bus time at the MLX90393's 400 kHz limit, and the conversion sets the rate of new data. A higher rate needs a shorter
conversion (DIG_FILT 4: 4.0 ms at 1.4 times the noise) and the second I2C controller for part of the boards.

The same day the three fingers were stacked against each other and left for 39 minutes (11:39:30 to 12:18). The stack
moved the rests by 991, 293 and 210 µT on channels 1, 0 and 3; without a re-zero each would read as an 8 to 11 N contact.
Replayed through the viewer's logic (`host/reskin5x_mux_soak.py`), no contact was shown at rest in 23,264 frames a finger.
The change the viewer reads stayed at a median of 4.0 to 4.7 µT and at most 12.3 µT, while the dies cooled 0.8 to 1.4 °C
and the re-zeros took up 7.7 to 13.0 µT of drift.

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_mux.py &
    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/wrench_viewer.py --port /tmp/reskin5x_mux/ch0 --http 8790
    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_mux_check.py logs/<date>-three_finger_mux/<start>-mux.bin

Start the viewers at least a second apart: each names its stream log by the second it starts.

Since 2026-10-26 one viewer shows every board on the mux, with no splitter: each board gets a row of the page with its
pad and its readout, and its own stream log and calibrations named after it. `--board-gp` adds the GP from the board's
change to the contact point and force, pooled over ID 1's and ID 2's UR5 holds (`scripts/ur5_tip_calibration_board_gp.py`,
3,813 presses and slides of cb1t, cb1r and cb2t; the fit equals `ur5_tip_calibration_predict.GP`'s to 3e-13 on a
300-hold subset). Each panel's menu, and key g for every board, picks the estimate shown.

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/wrench_viewer.py --mux /dev/serial/by-id/<board>-if00 \
        --names "ID 1" "ID 2" --calib <ID 1 calib_rest.json> <ID 2 calib_rest.json> --correction <cb1t folder> <cb2t folder> \
        --board-gp docs/experiments/20261022-ur5_tip_calibration/data/20261026-board_gp_id1_id2.npz

The same firmware runs on a SparkFun Thing Plus ESP32-C6, whose Qwiic connector is `Wire` on GPIO6 and GPIO7, a bus it
shares with the board's LiPo fuel gauge at 0x36 (2026-10-25). Its mux answered at 0x71. With one finger on channel 0 it
streams 537 records/s in the one-board format, so `host/wrench_viewer.py --port` reads its port with no splitter. The
noise per conversion is 1.88 µT on x and y and 2.89 µT on z, as on the QT Py. Its USB serial unit enumerates as
303a:1001 in the app as well, and `host/flash.sh` uploads to it without the 1200-baud touch:

    RESKIN5X_FQBN=esp32:esp32:sparkfun_esp32c6_thing_plus RESKIN5X_SKETCH=reskin5x_mux_stream \
        src/morphohand/driver/reskin5x/host/flash.sh /dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_<MAC>-if00

The sketch that was on the C6 before, an older port of the one-board stream, took the fuel gauge and the mux for
magnetometers and crashed on the null bus that `firmware/reskin5x_stream` fixes.

## Pad on the board

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_plot.py
    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_check.py --out pad.csv   # pad at rest
    ~/miniconda3/bin/python scripts/reskin5x_pad_fit.py pad.csv

With pad B00 and its first magnets (3 × 2 mm, not N52) on the board, 2026-10-20: the static field was 1.9 to 7.0 mT per
chip, the largest component 6.8 mT (18 % of the range), and the noise unchanged at 1.6 and 2.6 µT. The fit prefers two
diagonally opposite corner magnets reversed relative to the other three, with a moment 35 % of a 3 × 2 mm N52's at the
design spacing (0.38 mT RMS); full-strength magnets 2.8 mm farther away fit worse (0.71 mT), and five poles the same way
fit worst (1.91 mT). At the measured noise both readings give a normal-force resolution of 0.10 to 0.12 N at the contact
point, against 0.039 N for the design; the reversed magnets change it by under 20 %, in their favour. A press gave up to
64 µT on one chip and 8 µT on another, about 30 times the noise. N52 magnets should raise the fields about 2.9 times,
to 19 to 28 mT on the strongest chip, inside the range. Stacked into one column, magnets align with each other; marking
the column's top face and inserting every magnet marked face toward the palm puts all poles the same way, as the sensing
model assumes and as a calibration shared by three fingers needs.

## Visualizer

    ~/miniconda3/bin/python external/eFlesh/visualizer/viz_eflesh.py -p /dev/ttyACM0 --board white [-v magnitude] [-s 10]

With a pad on, start it with the pad at rest or press `b` before touching: the pad's static field is thousands of µT,
which at the default scale draws circles hundreds of pixels across. A press changes the field by tens of µT, so
`reskin5x_plot.py` shows it more clearly.

The window shows the board image with one mark per chip, drawn from the field minus a baseline (the mean of five samples at
start; key `b` takes it again). In the default mode each chip has a red circle of radius |ΔBz|/s pixels (filled for
ΔBz ≥ 0, a ring for ΔBz < 0) and a green line of length |(ΔBx, ΔBy)|/s pixels, rotated by the chip's orientation on the
image; `-v magnitude` draws one filled circle of radius |ΔB|/s. "No Contact" is shown while the norm of the 15
baseline-subtracted components is under 200 µT; at rest it reads about 8 µT. `--board` selects the chip rotations: white
for a board at 0x0C and 0x10 to 0x13, as the workstation's board is, black (the default) for one at 0x0C to 0x10. `-r`
saves the 15 baseline-subtracted components of each displayed frame (60 Hz) to `data/data_<time>.txt` under the working
directory.

## Contact estimation viewer

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/wrench_viewer.py   # then open http://localhost:8790

The viewer calibrates the measurement model from 2 s at rest (keep the pad untouched; press c to repeat), then shows the
estimated contact point and force anywhere on the pad's surface, sides included, the FE deformation and the pad pressure
in the browser; keys b (new baseline), c (recalibrate at rest), r (write the stretch between two presses of r as a CSV
for `--replay`). Every record of the stream also goes to `logs/20261020-contact_estimation/stream/<start>-stream.bin`
(88 bytes a record: host time float64, 5 x [t x y z] float32; `wrench_viewer.read_stream` reads it), so a session is
kept without pressing r. The model, the estimator and their errors are on the page
`docs/experiments/20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html`.

The chips run without their on-chip temperature compensation, and each one's sensitivity falls by 0.65 to 0.70 % per C of
its die temperature: the board warms by about 1 C in the minutes after it is plugged in, which moves the rest reading by
70 uT, about 2 N on the face. The viewer divides every reading by its chip's gain at the chip's temperature (coefficients
in `docs/experiments/20261020-fingertip_contact_estimation/data/20261021-thermal_gain.json`, fitted on a night at rest by
`scripts/fingertip_contact_estimation_study.py thermal`), adds a 2 uT noise floor to the estimator's noise and shows a
contact only above 0.25 N, and takes a new baseline once the reading has been as still as at rest for 3 s, which also
takes in a press held that still. The page has sliders for the threshold and the floor and a switch for the re-zero (key
z); `--floor`, `--min-force`, `--auto-zero 0` and `--no-thermal` set them at the start. When the board is unplugged the
viewer waits for it, opens it again (on its `/dev/serial/by-id` name, which stays the same when the board comes back as
another ttyACM) and calibrates again 2 s later, since the pad may have been changed meanwhile; nothing else needs to run
first.

A pad change needs no key (2026-10-22). With no pad on the board the reading's magnitude is 0.16 mT (9 to 13 mT with a
pad), and below 0.5 mT the viewer shows no contact; a pad put on is calibrated once it has been still for 3 s. A re-zero
that would move the rest by more than 10 % of the calibrated field calibrates instead, since the pad was set down in
another place or pose. All 32 pole patterns are searched, so each magnet may go in either way. The status line gives
the poles row by row, tip end first, and adds a warning when the next pattern fits within 1.5 times the residual. Every
calibration is saved under `logs/20261020-contact_estimation/calib/`.

A pad that moves in its seat leaves an offset after each press, which the estimator reads as a standing contact. To check
the seat, record about 2 s untouched, then five firm presses at different spots, each under a second and followed by at
least 2 s untouched, and run

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/pad_shift_check.py <recording>.csv
    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/pad_shift_check.py <stream>.bin --t0 120 --t1 180

It lists every rest's offset from the first rest and the rigid motion of the whole pad that fits it, with the readings
corrected for the chips' temperature (`--no-thermal` leaves them raw; uncorrected, a warming board reads as offsets).
Offsets of a few uT pass; the loose pad of 2026-10-20 left 62 and 104 uT, fit by shifts of 17 and 23 um and tilts of
about 4 mrad, and the touches of 2026-10-21 left 33-69 uT (shifts of 8-19 um). The model gives about 26 uT for 1 N on
the face.

## Calibration session on the UR5

`host/ur5_tip_calibration.py` runs the calibration on the lab's UR5 (CB3, RTDE at 192.168.0.4) against the OptoForce
HEX-E (its Ethernet DAQ at 192.168.0.5, UDP port 49152, polled at 500 Hz): homing on the indenter's key in freedrive, a
gravity sweep, touch-offs on the mount's column and on the pad, a registration fit, pivot presses, then calibration
presses, slides, rolls and twists from a plan written on the workstation
(`scripts/ur5_tip_calibration_plan.py` -> `docs/experiments/20261022-ur5_tip_calibration/plan/20261022-plan_sphere.json`).
The robot's computer runs it from a clone of the repository: the runbook
`docs/experiments/20261022-ur5_tip_calibration/README.md` and its setup check `scripts/ur5_tip_calibration_lab_setup.sh`
(git-lfs files, a venv with numpy, scipy, pyserial and ur_rtde, old copies of magpie_control's F/T reader, the robot,
the DAQ and the board, three simulated stages) cover it. The program stops the DAQ and empties its socket before it
starts, and empties it again before each polled request, so a late answer cannot pass for a newer one.

    python ur5_tip_calibration.py --plan 20261022-plan_sphere.json --stages preflight      # streams, payload
    python ur5_tip_calibration.py --plan 20261022-plan_sphere.json --dry 15                 # the session 15 mm off
    python ur5_tip_calibration.py --plan 20261022-plan_sphere.json --name sphere1           # the session
    ... --resume logs/<date>-tip_calibration/<session>                                      # after an interruption
    .venv/bin/python src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py --plan <plan> --sim   # the simulator

The lab's HEX-E is the 2017 OptoForce model, 35 mm from the flange to its tool face and 260 g (data sheet V1.0), so the
pad's face centre lies 128.36 mm from the flange along the tool axis (mount v3's pages took the 50 mm OnRobot HEX-E QC).
The session's files, the ground truth it gives and a simulated session are on the page
`docs/experiments/20261022-ur5_tip_calibration/20261022-ur5_tip_calibration_protocol.html`.

## Register settings

GAIN_SEL 1, RES 1 on all three axes, OSR 0, DIG_FILT 5 and HALLCONF 0xC: one count is 1.20 µT on x and y and 1.94 µT on z,
the range is 38.4 mT (the rule of `scripts/v2_tip_fe_checks.py`: 4.8 mT at GAIN_SEL 7 and RES 0, scaled by the count),
and one conversion of T, X, Y and Z takes 7.0 to 7.1 ms. The range covers the five-magnet concept-b pad B00, whose
largest field component at a chip is 27.5 mT (`docs/experiments/20261019-v2_tip_fe_checks/sensing.json`, case b_v2);
concept a's tip, at 11.4 mT, takes GAIN_SEL 2 and RES 0. To change the settings, edit the constants at the top of the
sketch, run `flash.sh`, and pass the new range to the check's `--range-mT`. The firmware converts counts to µT with the
settings it wrote, so the host reads µT at any setting.

## Record format and chip order

Each record is 82 bytes: for each chip in streaming order, t (°C) and x, y, z (µT) as little-endian float32, then CR LF.
A chip's four values are zeros when it did not answer at start-up, when its transfer failed, and until its first
conversion (5 to 222 ms after start-up in four boots); a read that returns no conversion (raw 0xFFFF, once in about
18,700 reads) keeps the chip's last values. The chip's status byte is not used: in burst mode its error bit was set on
93 % of reads of valid data.

The workstation's board answers at 0x0C, 0x10, 0x11, 0x12 and 0x13 and streams 0x0C, 0x11, 0x12, 0x13, 0x10; a board at
0x0C to 0x10 streams them in address order. The visualizer draws the five chips at the centre, left, right, top and
bottom of its board image in that order; 0x0C, the only -011 part, is U1 at the centre. The rest fit of 2026-10-20 places 0x11 at U3, 0x12 at
U5, 0x13 at U2 and 0x10 at U4 (contact estimation page, Table 1); a magnet held over one chip at a time would confirm it.
