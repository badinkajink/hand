#!/usr/bin/env bash
# Workstation companion launcher: the counterpart to ~/run_control_station.sh on the CB1.
#
# This exists because the service was being launched by hand, and the hand-typed form has one
# silent failure. The service reads --station-token from MANTA_TOKEN; unset, it starts happily,
# reports itself healthy, and every sample it pushes to the station is answered 401. The run
# then fails closed on `automatic-required` twenty seconds in, with the object already grasped.
# That was 2026-09-09. The repo path is resolved from this script's own location, so no cd is
# required -- the runbook's `cd ~/hand` is the CB1's path and does not exist here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TRACKER_PYTHON:-$HOME/miniconda3/bin/python}"   # the camera stack is not in the uv env
LOG="$ROOT/logs/tracker/service.log"
HOST="${TRACKER_HOST:-10.99.99.50}"
PORT="${TRACKER_PORT:-8770}"
PATTERN="real_v1_tracker_servic[e]"

# --width/--height are the only rig flags left to pass; BENCH_DEFAULTS in the service carries
# the rest and --tracker-arg is appended after them, so these override without dropping any.
ARGS=(--host "$HOST" --port "$PORT"
      --tracker-arg=--width --tracker-arg=1280
      --tracker-arg=--height --tracker-arg=800
      --tracker-arg=--fps --tracker-arg=30)

start() {
  if pgrep -f "$PATTERN" >/dev/null; then
    echo "already running:"; pgrep -af "$PATTERN"; return 0
  fi
  if [[ -f "$ROOT/.dotenv" ]]; then set -a; . "$ROOT/.dotenv"; set +a; fi
  if [[ -z "${MANTA_TOKEN:-}" ]]; then
    echo "MANTA_TOKEN is unset and $ROOT/.dotenv did not supply it." >&2
    echo "Without it every pushed sample is 401 and /reorient fails closed mid-trial." >&2
    return 2
  fi
  mkdir -p "$ROOT/logs/tracker"
  # The service verifies the token against the station before it serves, so a bad token is a
  # launch-time error here rather than a failed trial later.
  cd "$ROOT"
  nohup setsid "$PY" scripts/real_v1_tracker_service.py "${ARGS[@]}" \
    >>"$LOG" 2>&1 </dev/null &
  sleep 3
  if ! pgrep -f "$PATTERN" >/dev/null; then
    echo "service exited during startup; tail of $LOG:" >&2; tail -n 15 "$LOG" >&2; return 1
  fi
  tail -n 3 "$LOG"
}

stop()   { pkill -f "$PATTERN" && echo stopped || echo "not running"; }
status() {
  pgrep -af "$PATTERN" || { echo "not running"; return 1; }
  curl -s -m 5 "http://$HOST:$PORT/status" && echo
}
log()    { tail -n "${2:-40}" "$LOG"; }

case "${1:-start}" in
  start) start ;; stop) stop ;; status) status ;; log) log "$@" ;;
  restart) stop || true; sleep 2; start ;;
  *) echo "usage: $0 {start|stop|restart|status|log}" >&2; exit 64 ;;
esac
