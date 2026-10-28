#!/bin/bash
# Copy a UR5 session folder from the lab computer corvus, read-only, for analysis before the lab pushes it.
#   scripts/corvus_pull.sh 143529-tallv2id1 [20261027]      -> logs/corvus/<session>/
# Logs in as will over Tailscale with the password in the repository's gitignored .dotenv (CORVUS_WILL_PW), through
# ~/.local/bin/corvus-askpass. It never writes on corvus; plans go there by git.
set -euo pipefail
S=${1:?session, e.g. 143529-tallv2id1}; D=${2:-$(date +%Y%m%d)}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export SSH_ASKPASS="$HOME/.local/bin/corvus-askpass" SSH_ASKPASS_REQUIRE=force
SSH="ssh -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive -o ConnectTimeout=10"
mkdir -p "$ROOT/logs/corvus"
rsync -a --partial -e "$SSH" "will@100.93.255.76:workspace/hand/logs/${D}-tip_calibration/${S}/" "$ROOT/logs/corvus/${S}/"
echo "-> logs/corvus/${S}"
