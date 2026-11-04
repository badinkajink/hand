#!/bin/bash
# Compute and throughput after the HOM score queue (step 8 of hand-20261104-hom_score_refine_page, 11-04): GPU batches
# of 64, 256, 1024 and the queue's size per task, the CPU runner on one core (variants plant and filter; never matched
# or solver, scripts/hom_compute.py docstring) and on four cores, the STS3032 plants on one core, then ledger and report.
cd /home/humanoid/Programs/hand || exit 1
RG=$HOME/.claude/bin/resguard.sh
PY=.venv/bin/python
log() { echo "$(date +%T) $*"; }
run() {   # one guarded job; retries while resguard refuses it (exit 3)
  while true; do
    "$RG" run "$@"; rc=$?
    [ $rc -ne 3 ] && return $rc
    log "refused by resguard; retry in 120 s"; sleep 120
  done
}
log "start"; "$RG" status | head -1
log "gpu-bench"
# --force: the queue ended at 01:20:11; gpu-bench's pgrep for queue.sh also matches the autopilot session's own command
# line, which quotes that path
run --mem 5G --cpu 400 -- /usr/bin/time -f "peak %M kB, %e s" $PY scripts/hom_compute.py gpu-bench --force || log "gpu-bench failed"
"$RG" status | head -1
log "cpu-bench sv25, one core"
run --mem 2G --cpu 200 -- $PY scripts/hom_compute.py cpu-bench --n 6 --variants plant filter --procs 1 || log "failed"
log "cpu-bench sv25, four cores"
run --mem 6G --cpu 500 -- $PY scripts/hom_compute.py cpu-bench --n 4 --variants plant --procs 4 || log "failed"
log "cpu-bench k21, one core"
run --mem 2G --cpu 200 -- $PY scripts/hom_compute.py cpu-bench --plant k21 --n 4 --variants plant filter --procs 1 || log "failed"
log "cpu-bench k21mu0p4, one core"
run --mem 2G --cpu 200 -- $PY scripts/hom_compute.py cpu-bench --plant k21mu0p4 --n 4 --variants plant filter --procs 1 || log "failed"
"$RG" status | head -1
$PY scripts/hom_compute.py ledger
$PY scripts/hom_compute.py report
log "compute bench done"
