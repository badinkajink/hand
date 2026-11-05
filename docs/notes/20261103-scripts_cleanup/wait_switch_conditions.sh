#!/usr/bin/env bash
# Wait until the mechanical conditions of the scripts switch hold (RUNBOOK.md, "The switch"), then exit 0; exit 1
# after MAX_MIN minutes. The autopilot runs it as a background command and then reads the lab's STATE.md itself.
#   bash docs/notes/20261103-scripts_cleanup/wait_switch_conditions.sh [MAX_MIN] [EARLIEST_HHMM]
# Conditions: the fingertip-shape rerun is done or stopped (owner, 19:3x on 11-03) and no other autopilot job but the
# crosstalk poller runs; local time after EARLIEST_HHMM (default 1900) or before 08:00; no commit on origin/main under the lab
# folder, the SR2 hand's crosstalk folder or scripts/ in the last 45 min; no process runs a file under scripts/ other than crosstalk_feed_watch.py; no
# queue under logs/ runs; no uncommitted change under scripts/, src/, tests/, CLAUDE.md, AGENTS.md, README.md or the lab's
# STATE.md.
set -u
MAX_MIN=${1:-240}
EARLIEST=${2:-1900}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
LAB=docs/experiments/20261022-ur5_tip_calibration/lab
AP=$HOME/.claude/autopilot/ap
SELF=hand-20261103-scripts_topic_folders
POLLER=hand-20261027-crosstalk_feed_poller
RERUN=hand-20261103-fingertip_shape_rerun
NOT_PROG='^[0-9]+ (/usr/bin/tmux|tmux|/home/humanoid/\.local/bin/claude|claude) '
cd "$ROOT"
end=$(( $(date +%s) + MAX_MIN * 60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  why=""
  hm=$(date +%H%M)
  [ "$hm" -ge "$EARLIEST" ] || [ "$hm" -lt 0800 ] || why="between 08:00 and $EARLIEST"
  status=$("$AP" status 2>/dev/null | sed -n '/^== jobs/,/^== log/p')
  if echo "$status" | grep -E "^  $RERUN: (armed|running|waiting)" >/dev/null; then
    why="$why; the fingertip-shape rerun is not done"
  fi
  others=$(echo "$status" | grep -E '^  hand-[^:]+: running' | grep -v -e "^  $SELF:" -e "^  $POLLER:" | sed -E 's/^  ([^:]+):.*/\1/' | tr '\n' ' ')
  [ -z "$others" ] || why="$why; autopilot running: $others"
  git fetch -q origin 2>/dev/null
  now=$(date +%s)
  for p in "$LAB" docs/experiments/20261104-sr2_hand_crosstalk scripts; do
    t=$(git log -1 --format=%ct origin/main -- "$p" 2>/dev/null || echo 0)
    [ $(( now - ${t:-0} )) -ge 2700 ] || why="$why; commit under $p $(( (now - t) / 60 )) min ago"
  done
  # An autopilot's claude and tmux processes name scripts in their prompt; they run nothing by being there (`ap status` says
  # whether a session is busy, and the autopilot reads it before it switches).
  procs=$(pgrep -af "scripts/[A-Za-z0-9_]+\.(py|sh)" | grep -v -e crosstalk_feed_watch -e wait_switch_conditions -e pgrep | grep -v -E "$NOT_PROG")
  [ -z "$procs" ] || why="$why; running: $(echo "$procs" | head -2 | cut -c1-120 | tr '\n' ' ')"
  # A queue pauses between its steps, and a step that starts after the move finds no script (window 220623e7, 18:27 on 11-03).
  queue=$(pgrep -af "logs/[^ ]*queue[^ ]*\.sh" | grep -v -e wait_switch_conditions -e pgrep | grep -v -E "$NOT_PROG")
  [ -z "$queue" ] || why="$why; a queue runs: $(echo "$queue" | head -1 | cut -c1-100)"
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
