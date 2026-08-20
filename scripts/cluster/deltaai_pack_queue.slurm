#!/usr/bin/env bash
# PACKED WORK QUEUE — the right job shape once you know how DeltaAI charges.
#
# An SU is one GH200 for one hour, billed on what the job RESERVES rather than
# what it uses, and the reservation is rounded up. So a 40-minute training run
# submitted as its own job burns a whole hour and throws away 20 minutes; 48 of
# them throw away 16 hours, which is 17% of the current balance.
#
# The fix is not smaller reservations (the queue punishes churn and every job
# re-pays startup). It is to reserve a LONG block and keep it full: this script
# drains a shared manifest, running task after task back to back, and stops
# starting new ones only when the remaining wall time can no longer fit the next
# one. Nothing is wasted except the tail.
#
# It is also a self-balancing work queue rather than an array. Tasks are claimed
# atomically via mkdir, so you can submit K of these against the SAME manifest
# and get K-way concurrency with no index bookkeeping — and a slow task strands
# nobody, because the other workers just keep taking from the front.
#
# Resumption is the same mechanism: state lives in the queue dir, so if a block
# expires mid-queue the next submission picks up exactly where it stopped. That
# is what makes "measure where we left off" free.
#
# Manifest format (TAB-separated), one task per line:
#     EST_HOURS <TAB> LABEL <TAB> COMMAND
# COMMAND is run verbatim through bash from the repo root, so any experiment
# shape fits — A-lift, live-A reorient, an eval sweep — without this script
# having to know their flags.
#
# Usage:
#   scripts/cluster/deltaai_queue.py new <name> < tasks.tsv     # create the queue
#   sbatch --time=08:00:00 scripts/cluster/deltaai_pack_queue.slurm <name>
#   sbatch --time=08:00:00 scripts/cluster/deltaai_pack_queue.slurm <name>   # 2nd worker
#   scripts/cluster/deltaai_queue.py status <name>              # where we left off
#
# Sizing the reservation: 8 h of one GH200 = 8 SU and fits ~7 perp runs. Do NOT
# raise --mem past 110G or --cpus-per-task past 72 — charging is by the number of
# GH200s needed to satisfy memory OR cores, whichever is larger, so 220G would
# silently bill 2 SU per hour for the same one GPU.
#
#SBATCH --job-name=mh_pack
#SBATCH --account=bgcd-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pack_%j.log
set -uo pipefail          # NOT -e: a failed task must not kill the whole block

QNAME="${1:?usage: sbatch ... deltaai_pack_queue.slurm <queue-name>}"
cd "$SLURM_SUBMIT_DIR"
module purge
module load cuda
source .venv/bin/activate

QDIR="queue/$QNAME"
[ -d "$QDIR" ] || { echo "no such queue: $QDIR (create it with deltaai_queue.py new)"; exit 1; }
mkdir -p "$QDIR/claimed" "$QDIR/done" "$QDIR/failed" "$QDIR/logs"

# Warp cache: primed once, private per worker, node-local.
export WARP_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/warp_pack_$SLURM_JOB_ID"
mkdir -p "$WARP_CACHE_PATH"
[ -f "$HOME/warp_cache.tar" ] && tar -C "$WARP_CACHE_PATH" -xf "$HOME/warp_cache.tar"
trap 'rm -rf "$WARP_CACHE_PATH"' EXIT

# When does this reservation end? Slurm exports SLURM_JOB_END_TIME on recent
# versions; fall back to scontrol, and to "start + TimeLimit" if even that fails.
job_end_epoch() {
  if [ -n "${SLURM_JOB_END_TIME:-}" ]; then echo "$SLURM_JOB_END_TIME"; return; fi
  local et
  et=$(scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | grep '^EndTime=' | cut -d= -f2)
  if [ -n "$et" ]; then date -d "$et" +%s; else echo $(( $(date +%s) + 8*3600 )); fi
}
END_EPOCH=$(job_end_epoch)
# Leave a margin so the tail task is not SIGKILLed mid-checkpoint.
MARGIN_S=${MARGIN_S:-300}

echo "== worker $SLURM_JOB_ID on $(hostname), queue '$QNAME'"
echo "== block ends $(date -d "@$END_EPOCH"), margin ${MARGIN_S}s"

TOTAL=$(wc -l < "$QDIR/manifest.tsv")
RAN=0

while :; do
  # --- claim the first unclaimed task, atomically -----------------------------
  IDX=""
  for i in $(seq 1 "$TOTAL"); do
    [ -e "$QDIR/done/$i" ] && continue
    [ -e "$QDIR/failed/$i" ] && continue
    if mkdir "$QDIR/claimed/$i" 2>/dev/null; then IDX="$i"; break; fi
  done
  if [ -z "$IDX" ]; then echo "== queue drained (or all remaining tasks claimed)"; break; fi

  LINE=$(sed -n "${IDX}p" "$QDIR/manifest.tsv")
  IFS=$'\t' read -r EST LABEL CMD <<<"$LINE"
  EST_S=$(python3 -c "print(int(float('${EST:-1.0}')*3600))")

  # --- will it fit in what is left of the block? ------------------------------
  REMAIN=$(( END_EPOCH - $(date +%s) - MARGIN_S ))
  if [ "$REMAIN" -lt "$EST_S" ]; then
    rmdir "$QDIR/claimed/$IDX" 2>/dev/null   # release it for the next block
    echo "== ${REMAIN}s left, next task needs ${EST_S}s — stopping cleanly with the queue intact"
    break
  fi

  echo
  echo "======== [$IDX/$TOTAL] $LABEL  (est ${EST}h, ${REMAIN}s left in block)"
  date -Is
  START=$SECONDS
  bash -c "$CMD" > "$QDIR/logs/${IDX}_${LABEL}.log" 2>&1
  RC=$?
  ELAPSED=$((SECONDS - START))

  if [ "$RC" -eq 0 ]; then
    date -Is > "$QDIR/done/$IDX"
    echo "   OK in ${ELAPSED}s (est $((EST_S))s)" | tee -a "$QDIR/done/$IDX"
    RAN=$((RAN + 1))
  else
    # Record it as failed rather than leaving it claimed, so the queue does not
    # deadlock on one bad task — and so `status` reports it instead of hiding it.
    { date -Is; echo "rc=$RC after ${ELAPSED}s"; } > "$QDIR/failed/$IDX"
    echo "   FAILED rc=$RC after ${ELAPSED}s — see $QDIR/logs/${IDX}_${LABEL}.log"
  fi
  rmdir "$QDIR/claimed/$IDX" 2>/dev/null
done

echo
echo "== worker $SLURM_JOB_ID finished: ran $RAN task(s)"
echo "== done $(ls "$QDIR/done" 2>/dev/null | wc -l)/$TOTAL, failed $(ls "$QDIR/failed" 2>/dev/null | wc -l)"
