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
- `host/flash.sh`: compile and upload.
- `host/reskin5x_check.py`: per-chip field, noise, drift and temperature, the record rate, and the fraction of the range in
  use, from a few seconds of stream; `--out` writes the records as text.

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

## Visualizer

    ~/miniconda3/bin/python external/eFlesh/visualizer/viz_eflesh.py -p /dev/ttyACM0 --board white [-v magnitude] [-s 10]

The window shows the board image with one mark per chip, drawn from the field minus a baseline (the mean of five samples at
start; key `b` takes it again). In the default mode each chip has a red circle of radius |ΔBz|/s pixels (filled for
ΔBz ≥ 0, a ring for ΔBz < 0) and a green line of length |(ΔBx, ΔBy)|/s pixels, rotated by the chip's orientation on the
image; `-v magnitude` draws one filled circle of radius |ΔB|/s. "No Contact" is shown while the norm of the 15
baseline-subtracted components is under 200 µT; at rest it reads about 8 µT. `--board` selects the chip rotations: white
for a board at 0x0C and 0x10 to 0x13, as the workstation's board is, black (the default) for one at 0x0C to 0x10. `-r`
saves the 15 baseline-subtracted components of each displayed frame (60 Hz) to `data/data_<time>.txt` under the working
directory.

## Register settings

GAIN_SEL 1, RES 1 on all three axes, OSR 0, DIG_FILT 5 and HALLCONF 0xC: one count is 1.20 µT on x and y and 1.94 µT on z,
the range is 38.4 mT (the rule of `scripts/v2_tip_fe_checks.py`: 4.8 mT at GAIN_SEL 7 and RES 0, scaled by the count),
and one conversion of T, X, Y and Z takes 7.0 ms (143 Hz). The range covers the five-magnet concept-b pad B00, whose
largest field component at a chip is 27.5 mT (`docs/experiments/20261019-v2_tip_fe_checks/sensing.json`, case b_v2);
concept a's tip, at 11.4 mT, takes GAIN_SEL 2 and RES 0. To change the settings, edit the constants at the top of the
sketch, run `flash.sh`, and pass the new range to the check's `--range-mT`. The firmware converts counts to µT with the
settings it wrote, so the host reads µT at any setting.

## Record format and chip order

Each record is 82 bytes: for each chip in streaming order, t (°C) and x, y, z (µT) as little-endian float32, then CR LF.
The firmware reads the five chips about 375 times a second, faster than they convert, so consecutive records repeat
values, and the three axes of one chip change at different records. The workstation's board answers at 0x0C, 0x10, 0x11,
0x12 and 0x13 and streams 0x0C, 0x11, 0x12, 0x13, 0x10; a board at 0x0C to 0x10 streams them in address order. The
visualizer draws the five chips at the centre, left, right, top and bottom of its board image in that order. Which
address is U1 to U5 on this board has not been measured; a magnet held over one chip at a time shows it.
