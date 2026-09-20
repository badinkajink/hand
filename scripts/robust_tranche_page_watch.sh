#!/bin/bash
cd /home/humanoid/Programs/hand
while pgrep -f "[t]ranche_queue.py --queue docs/experiments/20260920-robust_tranche" > /dev/null; do
  python3 scripts/robust_tranche_page.py > /dev/null 2>&1; echo "$(date +%F\ %T) rebuilt"; sleep 600
done
python3 scripts/robust_tranche_page.py > /dev/null 2>&1; echo "$(date +%F\ %T) final rebuild; driver gone"
git add docs/experiments/20260920-robust_tranche/*.json docs/experiments/20260920-robust_tranche/*.csv docs/experiments/20260920-robust_tranche/*.png docs/experiments/20260920-robust_tranche/*.html docs/experiments/20260920-robust_tranche/web 2>/dev/null
git commit -q -m "robust tranche: results as the queue finished (driver + page watcher)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" && echo "$(date +%F\ %T) committed"
