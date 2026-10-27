#!/bin/bash
# The analysis of the three v1 fingertips' night at rest (hand-off 20261026-three_v1_fingers_overnight, steps 2-7):
# waits until the splitter's log has passed 04:00 on 2026-10-27, then runs every tool on 19:35-04:00 under resguard and
# writes the rows the page builder reads (scripts/three_finger_soak_page.py) to data/. Progress goes to
# logs/20261026-three_finger_mux/analysis.log; a job that resguard refuses or kills leaves its file missing, and the
# script goes on. The viewer soak runs two jobs at a time (2.6 GB each on 7.3 h of log).
#
#     nohup setsid docs/experiments/20261027-three_v1_fingers_overnight/run_analysis.sh >> logs/20261026-three_finger_mux/analysis.log 2>&1 &
cd /home/humanoid/Programs/hand || exit 1
LOG=logs/20261026-three_finger_mux/20261026-193314-mux.bin
OUT=docs/experiments/20261027-three_v1_fingers_overnight/data
H=src/morphohand/driver/reskin5x/host
ST=logs/20261020-contact_estimation/stream
PY=$HOME/miniconda3/bin/python
VPY=.venv/bin/python
RG="$HOME/.claude/bin/resguard.sh run --cpu 100 --mem"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1
CAL="ch0=logs/20261020-contact_estimation/calib/20261026-134217-id_1-calib.json ch1=docs/experiments/20261022-ur5_tip_calibration/data/20261025-152920-cb2t/calib_rest.json ch2=logs/20261020-contact_estimation/calib/20261025-112147-calib.json"
NAMES=("ID 1" "ID 2" "ID 3")
say() { echo "=== $(date '+%F %T') $*"; }

until [ "$(date +%s)" -ge "$(date -d '2026-10-27 04:00:40' +%s)" ]; do sleep 20; done
say "start; $(~/.claude/bin/resguard.sh status | head -1)"
mkdir -p $OUT
$RG 3G -- $PY $H/reskin5x_mux_check.py $LOG --since 03:50 --until 04:00 --out $OUT/check_0350.json
$RG 3G -- $PY $H/reskin5x_mux_check.py $LOG --since 19:35 --until 04:00 --scan --out $OUT/check_night.json
say "rest"
$RG 4G -- $PY $H/reskin5x_mux_rest.py $LOG --ref 19:35:00 19:40:00 --since 19:35 --until 04:00 --names "${NAMES[@]}" \
    --bin-s 60 --tempfit --rows 20 --out $OUT/rest.json --series $OUT/rest_60s.npz
$RG 4G -- $PY $H/reskin5x_mux_rest.py $LOG --ref 19:35:00 19:40:00 --since 19:35 --until 04:00 --names "${NAMES[@]}" \
    --bin-s 60 --tempfit --rows 20 --no-thermal --out $OUT/rest_raw.json --series $OUT/rest_60s_raw.npz
$RG 4G -- $PY $H/reskin5x_mux_rest.py $LOG --ref 19:35:00 19:40:00 --since 19:35 --until 04:00 --names "${NAMES[@]}" \
    --bin-s 1 --rows 1 --series $OUT/rest_1s.npz
$PY $H/reskin5x_mux_steps.py $OUT/rest_1s.npz --out $OUT/steps.json
$PY $H/reskin5x_mux_steps.py $OUT/rest_1s.npz --min-uT 1.5 --out $OUT/steps_1p5.json
say "allan"
$RG 4G -- $VPY $H/reskin5x_mux_allan.py $LOG --since 19:35 --until 04:00 --names "${NAMES[@]}" --out $OUT/allan.json
$RG 4G -- $VPY $H/reskin5x_mux_allan.py $LOG --since 19:35 --until 04:00 --names "${NAMES[@]}" --no-thermal --out $OUT/allan_raw.json
say "recovery"
$RG 5G -- $PY $H/reskin5x_mux_recovery.py --names "ID 1" "ID 2" \
    --streams $ST/20261026-134212-id_1-stream.bin,$ST/20261026-141207-id_1-stream.bin \
              $ST/20261026-134212-id_2-stream.bin,$ST/20261026-141207-id_2-stream.bin \
    --mux $LOG --mux-index 0 1 --mux-since 19:35 --mux-until 04:00 --ref 13:42:30 13:43:30 --presses 13:49 14:00 \
    --rows 40 --out $OUT/recovery.json
say "soak"
for fl in 2 1; do
    for pair in "0 1" "2"; do
        for ch in $pair; do
            $RG 4G -- $VPY $H/reskin5x_mux_soak.py $LOG --since 19:35 --until 04:00 --channels 0 1 2 --calib $CAL \
                --floor $fl --only $ch --out $OUT/soak_floor${fl}_ch${ch}.json &
            sleep 5
        done
        wait
        say "soak floor $fl channels $pair done; $(~/.claude/bin/resguard.sh status | head -1)"
    done
    $PY - $OUT $fl <<'EOF'
import json, sys
from pathlib import Path
out, fl = Path(sys.argv[1]), sys.argv[2]
parts = [json.loads((out / f"soak_floor{fl}_ch{c}.json").read_text()) for c in range(3) if (out / f"soak_floor{fl}_ch{c}.json").exists()]
m = dict(parts[0], boards={k: v for p in parts for k, v in p["boards"].items()})
(out / f"soak_floor{fl}.json").write_text(json.dumps(m, indent=1) + "\n")
print(f"soak_floor{fl}.json: {sorted(m['boards'])}")
EOF
done
say "done"
