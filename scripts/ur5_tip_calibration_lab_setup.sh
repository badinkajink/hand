#!/usr/bin/env bash
# Setup check of the UR5's computer for the fingertip calibration session (runbook:
# docs/experiments/20261022-ur5_tip_calibration/README.md). Run it from anywhere in a fresh clone:
#
#   bash scripts/ur5_tip_calibration_lab_setup.sh [--python PATH | --venv DIR] [--robot IP] [--ft IP] [--no-sim]
#
# 1. pulls the plan, the protocol page and the simulator's data through git-lfs (a fresh clone has only pointers);
# 2. makes a venv with numpy, scipy, pyserial and ur_rtde (default ~/.venvs/tipcal), or checks the interpreter --python;
# 3. lists every copy of magpie_control's OptoForce reader in $HOME and marks the old one (2024-11 to 2026-03: a 20 ms
#    sleep per sample, stale samples piling up in the socket; the reader since commit 806b94a has VALID_RATES);
# 4. checks the robot's RTDE ports, asks each candidate address of the HEX-E's DAQ for one sample, looks for the board;
# 5. runs preflight, homing and gravity in the simulator.
# Nothing moves on the robot. Every check prints "ok" or "FIX"; fix those and run it again.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=$HOME/.venvs/tipcal; PY=""; ROBOT=192.168.0.4; FT=192.168.0.5; SIM=1
while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV=$2; shift ;;
    --python) PY=$2; shift ;;
    --robot) ROBOT=$2; shift ;;
    --ft) FT=$2; shift ;;
    --no-sim) SIM=0 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
cd "$ROOT" || exit 1
FAIL=0
ok() { printf '  ok   %s\n' "$*"; }
fix() { printf '  FIX  %s\n' "$*"; FAIL=1; }

CAL=docs/experiments/20261022-ur5_tip_calibration
EST=docs/experiments/20261020-fingertip_contact_estimation/data
PLAN=$CAL/plan/20261022-plan_sphere.json
LFS_FILES="$PLAN,$CAL/20261022-ur5_tip_calibration_protocol.html,$EST/20261020-b_v2_surface_basis.npz,$EST/20261020-calib_pad_as_found.json,$EST/20261021-thermal_gain.json,docs/experiments/20261019-v2_tip_fe_checks/sense_fe.jsonl"

echo "1. files kept in git-lfs"
if git lfs version >/dev/null 2>&1; then
  # .lfsconfig sets lfs.fetchexclude = * so that clones skip the media; cleared here for these six files only
  if git -c lfs.fetchexclude= lfs pull --include="$LFS_FILES"; then ok "git lfs pull"; else fix "git lfs pull failed: GitHub access (SSH key or token) or the network"; fi
else
  fix "git-lfs is missing: sudo apt install git-lfs (or the binary from github.com/git-lfs/git-lfs/releases), then run this again"
fi
for f in ${LFS_FILES//,/ }; do
  if [ ! -s "$f" ]; then fix "$f is missing"
  elif head -c 64 "$f" | grep -q 'git-lfs.github.com/spec'; then fix "$f is still an LFS pointer"
  else ok "$f ($(du -h "$f" | cut -f1))"; fi
done

echo "2. Python"
if [ -z "$PY" ]; then
  if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV" || fix "python3 -m venv failed: sudo apt install python3-venv, or pass --python PATH"
  fi
  PY=$VENV/bin/python
  if [ -x "$PY" ]; then
    "$PY" -m pip install -q --disable-pip-version-check numpy scipy pyserial ur_rtde \
      || fix "pip could not install numpy, scipy, pyserial and ur_rtde into $VENV (ur_rtde has wheels for CPython 3.8 to 3.12)"
  fi
fi
if "$PY" -c "import numpy, scipy, serial" 2>/dev/null; then ok "$PY: $("$PY" --version 2>&1), numpy, scipy, pyserial"
else fix "$PY lacks numpy, scipy or pyserial"; fi
if "$PY" -c "import rtde_control, rtde_receive" 2>/dev/null; then ok "ur_rtde imports"
else fix "$PY lacks ur_rtde (pip install ur_rtde)"; fi

echo "3. magpie_control's OptoForce readers on this computer"
n_old=0
while IFS= read -r f; do
  grep -q 'class OptoForce' "$f" 2>/dev/null || continue
  if grep -q VALID_RATES "$f"; then ok "current reader: $f"; continue; fi
  n_old=$((n_old + 1))
  if repo=$(git -C "$(dirname "$f")" rev-parse --show-toplevel 2>/dev/null); then
    at=$(git -C "$repo" log -1 --format='%h %ad' --date=short 2>/dev/null)
    printf '  OLD  %s\n       checkout %s at %s: update it to magpie_control main (see the runbook)\n' "$f" "$repo" "$at"
  else
    printf '  OLD  %s\n       a built or installed copy: rebuild or reinstall it from a current checkout\n' "$f"
  fi
done < <(timeout 120 find "$HOME" -xdev -name ft_sensor.py -not -path '*/.git/*' 2>/dev/null)
[ "$n_old" -eq 0 ] && ok "no old reader found"

echo "4. robot, HEX-E and board"
[ -x "$PY" ] || PY=$(command -v python3)
for port in 30002 30004; do
  if timeout 2 bash -c "</dev/tcp/$ROBOT/$port" 2>/dev/null; then ok "robot $ROBOT answers on port $port"
  else fix "robot $ROBOT does not answer on port $port: network (this computer needs an address on 192.168.0.x), or the controller is off"; fi
done
found=$("$PY" - "$FT" <<'EOF'
import socket, struct, sys
seen = []
for ip in dict.fromkeys([sys.argv[1], "192.168.0.5", "192.168.0.3"]):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.3)
    try:
        s.connect((ip, 49152))
        s.send(struct.pack("!HHI", 0x1234, 0x0002, 1))
        d = s.recv(64)
        if len(d) == 36:
            u = struct.unpack("!3I6i", d)
            print(f"{ip} sample {u[1]}, Fz {u[5] / 1e4:+.2f} N")
            seen.append(ip)
    except OSError:
        pass
    finally:
        s.close()
EOF
)
if [ -n "$found" ]; then
  while IFS= read -r line; do ok "HEX-E DAQ $line"; done <<< "$found"
  first=$(echo "$found" | head -1 | cut -d' ' -f1)
  [ "$first" != "$FT" ] && echo "       pass --ft $first to the program"
else
  fix "no HEX-E DAQ answers on $(printf '%s\n' "$FT" 192.168.0.5 192.168.0.3 | awk '!s[$0]++' | paste -sd' '), UDP 49152: its power, its Ethernet cable, its address"
fi
board=$(ls /dev/serial/by-id/*QT_Py* 2>/dev/null | head -1)
if [ -n "$board" ]; then
  if [ -r "$board" ] && [ -w "$board" ]; then ok "board $board"
  else fix "board $board is not readable: sudo usermod -aG dialout $USER, then log in again"; fi
else
  fix "no board under /dev/serial/by-id/*QT_Py*: plug it in, or pass --board /dev/ttyACM0 to the program"
fi

if [ "$SIM" -eq 1 ]; then
  echo "5. simulator: preflight, homing and gravity"
  tmp=$(mktemp -d)
  out=$("$PY" src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py --plan "$PLAN" --sim \
        --stages preflight,homing,gravity --logs "$tmp" --name setup_check 2>&1)
  if echo "$out" | grep -q '^gravity: weight'; then ok "$(echo "$out" | grep '^gravity: weight' | cut -c1-110)"
  else fix "the simulated stages failed:"; echo "$out" | tail -15 | sed 's/^/       /'; fi
  rm -rf "$tmp"
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "ready. In tmux (tmux new -s tipcal), from $ROOT:"
else
  echo "fix the FIX lines and run this again. The session's commands, once ready (in tmux new -s tipcal, from $ROOT):"
fi
echo "  PY=$PY"
echo "  PLAN=$PLAN"
echo "  \$PY src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py --plan \$PLAN --stages preflight --name preflight"
exit $FAIL
