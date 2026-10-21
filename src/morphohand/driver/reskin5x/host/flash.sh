#!/usr/bin/env bash
# Compile firmware/reskin5x_stream for the Adafruit QT Py ESP32-S3 (4 MB flash, 2 MB PSRAM) and upload it; the 8 MB
# no-PSRAM QT Py takes RESKIN5X_FQBN=esp32:esp32:adafruit_qtpy_esp32s3_nopsram.
#   host/flash.sh            compile, then upload to the first /dev/ttyACM*
#   host/flash.sh PORT       compile, then upload to PORT
#   host/flash.sh build      compile only
# Uses arduino-cli (on PATH, $ARDUINO_CLI, or the copy inside the Arduino IDE 2 flatpak), the esp32:esp32 core in
# ~/.arduino15, and the MLX90393 library of the external/eFlesh submodule. A board in the ROM bootloader (BOOT held while
# RESET is pressed; USB 303a:1001) is flashed without the 1200-baud touch, which would reset it out of the bootloader.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
FQBN=${RESKIN5X_FQBN:-esp32:esp32:adafruit_qtpy_esp32s3_n4r2}
LIB=$REPO/external/eFlesh/arduino/arduino-MLX90393
BUILD=$REPO/logs/reskin5x_build
CLI=${ARDUINO_CLI:-$(command -v arduino-cli || ls /var/lib/flatpak/app/cc.arduino.IDE2/x86_64/stable/*/files/arduino-ide/resources/app/lib/backend/resources/arduino-cli 2>/dev/null | head -1)}
[ -x "$CLI" ] || { echo "arduino-cli not found; set ARDUINO_CLI" >&2; exit 1; }

if [ ! -f "$LIB/MLX90393.h" ]; then
  git -C "$REPO" submodule update --init external/eFlesh
  git -C "$REPO/external/eFlesh" submodule update --init arduino/arduino-MLX90393
fi

"$CLI" compile --fqbn "$FQBN" --library "$LIB" --jobs 4 --build-path "$BUILD" "$HERE/firmware/reskin5x_stream"
[ "${1:-}" = build ] && exit 0

PORT=${1:-$(ls /dev/ttyACM* 2>/dev/null | head -1)}
[ -n "$PORT" ] || { echo "no /dev/ttyACM* port" >&2; exit 1; }
ROM=0
[ "$(udevadm info -q property -n "$PORT" | sed -n 's/^ID_VENDOR_ID=//p')" = 303a ] && ROM=1
if [ $ROM = 0 ]; then
  "$CLI" upload -p "$PORT" --fqbn "$FQBN" --input-dir "$BUILD"
  exit 0
fi
"$CLI" upload -p "$PORT" --fqbn "$FQBN" --input-dir "$BUILD" \
  --upload-property upload.use_1200bps_touch=false --upload-property upload.wait_for_upload_port=false
# The RTS reset after the upload re-enters the ROM bootloader, whose BOOT strap stays latched; a watchdog reset boots the app.
ET=$(find ~/.arduino15/packages/esp32/tools/esptool_py -maxdepth 2 -type f -name esptool -perm -u+x | sort -V | tail -1)
sleep 1
"$ET" --chip esp32s3 --port "$PORT" --before no-reset --after watchdog-reset read-mac
