#!/usr/bin/env bash
# Wait until the mechanical conditions of the scripts switch hold (RUNBOOK.md, "The switch"), then exit 0; exit 1
# after MAX_MIN minutes. The autopilot runs it as a background command and then reads the lab's STATE.md itself.
#   bash docs/notes/20261103-scripts_cleanup/wait_switch_conditions.sh [MAX_MIN] [EARLIEST_HHMM]
# Conditions: local time after EARLIEST_HHMM (default 1900) or before 08:00; no commit on origin/main under the lab folder or under
# scripts/ in the last 45 min; no process runs a file under scripts/ other than crosstalk_feed_watch.py; the HOM score
# queue does not run; no uncommitted change under scripts/, src/, tests/, CLAUDE.md, AGENTS.md, README.md or the lab's
# STATE.md. EARLIEST_HHMM 0700 lifts the clock from 07:00: the HOM page job stops at 07:15 for the switch, so its 45 minutes
# end at 08:00, and the switch may land until 09:00 on 11-04.
set -u
MAX_MIN=${1:-240}
EARLIEST=${2:-1900}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
LAB=docs/experiments/20261022-ur5_tip_calibration/lab
cd "$ROOT"
end=$(( $(date +%s) + MAX_MIN * 60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  why=""
  hm=$(date +%H%M)
  [ "$hm" -ge "$EARLIEST" ] || [ "$hm" -lt 0800 ] || why="between 08:00 and $EARLIEST"
  git fetch -q origin 2>/dev/null
  now=$(date +%s)
  for p in "$LAB" scripts; do
    t=$(git log -1 --format=%ct origin/main -- "$p" 2>/dev/null || echo 0)
    [ $(( now - ${t:-0} )) -ge 2700 ] || why="$why; commit under $p $(( (now - t) / 60 )) min ago"
  done
  # An autopilot's claude and tmux processes name scripts in their prompt; they run nothing by being there (`ap status` says
  # whether a session is busy, and the autopilot reads it before it switches).
  procs=$(pgrep -af "scripts/[A-Za-z0-9_]+\.(py|sh)" | grep -v -e crosstalk_feed_watch -e wait_switch_conditions -e pgrep \
    | grep -v -E '^[0-9]+ (/usr/bin/tmux|tmux|/home/humanoid/\.local/bin/claude|claude) ')
  [ -z "$procs" ] || why="$why; running: $(echo "$procs" | head -2 | cut -c1-120 | tr '\n' ' ')"
  # The HOM queue pauses between its steps, and a step that starts after the move finds no script (window 220623e7, 18:27).
  queue=$(pgrep -af "[h]om_score_refine/queue.sh" | grep -v -E '^[0-9]+ (/usr/bin/tmux|tmux|/home/humanoid/\.local/bin/claude|claude) ')
  [ -z "$queue" ] || why="$why; the HOM score queue runs"
  dirty=$(git status --porcelain -- scripts src tests CLAUDE.md AGENTS.md README.md "$LAB/STATE.md" | grep -v '^??')
  [ -z "$dirty" ] || why="$why; uncommitted: $(echo "$dirty" | head -3 | tr '\n' ' ')"
  if [ -z "$why" ]; then
    echo "$(date '+%F %H:%M') conditions hold"
    exit 0
  fi
  echo "$(date '+%F %H:%M') waiting: ${why#; }"
  sleep 600
done
echo "$(date '+%F %H:%M') gave up after $MAX_MIN min"
exit 1
