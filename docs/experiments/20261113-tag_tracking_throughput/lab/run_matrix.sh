#!/bin/bash
# The tracker's throughput at each camera mode and exposure, 60 s each, the hex block still in both views.
#   bash run_matrix.sh OUTDIR
set -u
OUT=${1:?OUTDIR}
B=$(dirname "$0")/tracker_bench.py
mkdir -p "$OUT"
run() { name=$1; shift; echo "$(date +%H:%M:%S) $name $*"; /usr/bin/python3 -u "$B" "$OUT/$name" --seconds 60 --arm "$@" 2>&1 | grep -v "^   ts domain"; echo "$(date +%H:%M:%S) $name rc ${PIPESTATUS[0]}"; sleep 3; }
run 1080p30_e166 --width 1920 --height 1080 --fps 30 --exposure 166 --gain 64
run 1080p30_e50 --width 1920 --height 1080 --fps 30 --exposure 50 --gain 128
run 720p30_e166 --width 1280 --height 720 --fps 30 --exposure 166 --gain 64
run 540p60_e100 --width 960 --height 540 --fps 60 --exposure 100 --gain 96
run 1080p15_e166 --width 1920 --height 1080 --fps 15 --exposure 166 --gain 64
echo "$(date +%H:%M:%S) matrix done"
