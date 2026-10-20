# reskin5x

Readout of the ReSkin 5X magnetometer board (five MLX90393; U1 at the centre, U2 to U5 7.00 mm from it in a plus) through
an Adafruit QT Py ESP32-S3 (8 MB flash, no PSRAM), with a Qwiic cable from the board's J1 to the QT Py's STEMMA QT
connector. The firmware is the eFlesh sketch ported to this board. Its record format is eFlesh's and AnySkin's, so the
eFlesh visualizer in the `external/eFlesh` submodule (the owner's fork, github.com/badinkajink/eFlesh) runs unchanged.

## Layout

- `firmware/reskin5x_stream/reskin5x_stream.ino`: the port of `external/eFlesh/arduino/5X_eflesh_stream/5X_eflesh_stream.ino`.
  The STEMMA QT connector is the board's `Wire1` (SDA1 GPIO41, SCL1 GPIO40), `Serial` is the TinyUSB CDC port, and the
  register settings are those below; the header comment lists every change.
- `host/flash.sh`: compile and upload.
- `host/reskin5x_check.py`: per-chip field, noise and temperature, the record and conversion rates, and the fraction of
  the range in use, from a few seconds of stream; `--out` writes the records as text.

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

A board running this firmware resets into its bootloader over USB (the 1200-baud touch). A board running other firmware,
such as Adafruit's factory sketch, may not: hold BOOT, press and release RESET, release BOOT, then run `flash.sh`. The ROM
bootloader enumerates as USB 303a:1001; after the upload the board enumerates as 239a:8119, "QT Py ESP32-S3 No PSRAM".

## Checking the stream

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_check.py --seconds 5

## Visualizer

    ~/miniconda3/bin/python external/eFlesh/visualizer/viz_eflesh.py -p /dev/ttyACM0 [-v magnitude] [-s 10]

The window shows the board image with one mark per chip, drawn from the field minus a baseline (the mean of five samples at
start; key `b` takes it again). In the default mode each chip has a red circle of radius |ΔBz|/s pixels (filled for
ΔBz ≥ 0, a ring for ΔBz < 0) and a green line of length |(ΔBx, ΔBy)|/s pixels, rotated by the chip's orientation on the
image; `-v magnitude` draws one filled circle of radius |ΔB|/s. "No Contact" is shown while the norm of the 15
baseline-subtracted components is under 200 µT. `-r` saves the samples to `data/` under the working directory, which must
exist.

## Register settings

GAIN_SEL 1, RES 1 on all three axes, OSR 0, DIG_FILT 5 and HALLCONF 0xC: one count is 1.20 µT on x and y and 1.94 µT on z,
the range is 38.4 mT (the rule of `scripts/v2_tip_fe_checks.py`: 4.8 mT at GAIN_SEL 7 and RES 0, scaled by the count),
and one conversion of T, X, Y and Z takes 7.0 ms (143 Hz). The range covers the five-magnet concept-b pad B00, whose
largest field component at a chip is 27.5 mT (`docs/experiments/20261019-v2_tip_fe_checks/sensing.json`, case b_v2);
concept a's tip, at 11.4 mT, takes GAIN_SEL 2 and RES 0. To change the settings, edit the constants at the top of the
sketch and run `flash.sh`. The firmware converts counts to µT with the settings it wrote, so the host reads µT at any
setting.

## Record format and chip order

Each record is 82 bytes: for each chip in streaming order, t (°C) and x, y, z (µT) as little-endian float32, then CR LF.
The streaming order is 0x0C, 0x0D, 0x0E, 0x0F, 0x10 when those five addresses answer; eFlesh's white board, at 0x0C and
0x10 to 0x13, streams 0x0C, 0x11, 0x12, 0x13, 0x10. The visualizer draws the five chips at the centre, left, right, top
and bottom of its board image in that order, with the chip rotations of the black board. Which address is U1 to U5 on
this board has not been measured; a magnet held over one chip at a time shows it.
