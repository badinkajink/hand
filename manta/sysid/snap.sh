#!/bin/bash
# Grab one RealSense colour frame on the CB1 and convert it on the Mac.
#   sysid/snap.sh <name>   -> $SNAP_DIR/<name>.png
set -euo pipefail
NAME=$1
OUT=${SNAP_DIR:?set SNAP_DIR}
SOCK=(-o ControlPath=$HOME/.ssh/cb1.sock)
F=$(ssh -S ~/.ssh/cb1.sock irlab@10.99.99.2 'rm -rf /tmp/snap && mkdir -p /tmp/snap && timeout 10 python3 /tmp/v4l2_grab.py ${SNAP_DEV:-/dev/video4} /tmp/snap --count 2 --period 0.2 >/dev/null 2>&1; ls /tmp/snap/*.yuyv | tail -1')
TMP=$(mktemp -d)
scp -q "${SOCK[@]}" "irlab@10.99.99.2:$F" "$TMP/$NAME.yuyv"
"${PYTHON:-/opt/homebrew/bin/python3.11}" "$(dirname "$0")/yuyv_to_png.py" "$TMP" 640 480 "$OUT" >/dev/null
rm -rf "$TMP"
echo "$OUT/$NAME.png"
