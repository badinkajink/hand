#!/usr/bin/env bash
# After scripts/rl_contact_train_queue.sh: evaluate every checkpoint (scripts/rl_contact_eval.py, one GPU job) and
# make the phase-aligned filmstrip of each run's newest eval video (scripts/policy_filmstrip.py). Waits for the
# training queue to exit first, so two GPU jobs never overlap.
#
#   nohup setsid bash scripts/rl_contact_post_queue.sh > logs/20261006-rl_contact/post_queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
until ! pgrep -f "[r]l_contact_train_queue.sh" >/dev/null; do sleep 60; done
sleep 90
echo "$(date +%H:%M) training queue finished; evaluating checkpoints"
until ~/.claude/bin/resguard.sh status | grep -q "launch gate: OPEN"; do sleep 60; done
WARP_CACHE_PATH=$(mktemp -d) ~/.claude/bin/resguard.sh run --mem 8G --cpu 400 -- \
  uv run --extra rl --extra gpu python scripts/rl_contact_eval.py
echo "$(date +%H:%M) eval exit $?"
mkdir -p docs/experiments/20261006-rl_contact/media
for run in results/rl/20261006-d6_work_*_20M_s*; do
  tag=$(basename "$run")
  uv run python scripts/policy_filmstrip.py --run "$run" --out "docs/experiments/20261006-rl_contact/media/${tag}_strip.png" \
    || echo "filmstrip failed for $tag"
done
echo "$(date +%H:%M) post queue done"
