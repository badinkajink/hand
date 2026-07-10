#!/bin/bash
# probe_queue.sh — POLICY-BOTTLENECK probe queue (2026-07-10): P1 rescue -> P2 avar.
# =============================================================================
# Validates the user's intuition that the morphology-landscape bottleneck is the
# POLICY OPTIMIZER, not the designs. Plan + decision tree:
# docs/rl/morph_sweep_STATUS.md "POLICY-BOTTLENECK PROBES (2026-07-10)".
#
# Sequential (one GPU). Each stage is morph_pipeline_sweep.py (RESUMABLE — finished
# designs skip), so re-running this script after any crash/kill continues where it
# stopped. Per-stage sentinel logs/MORPH_PIPELINE_<tag>.DONE; queue sentinel
# logs/PROBE_QUEUE.DONE.
#
# Launch (detached):
#   nohup setsid bash scripts/probe_queue.sh > logs/PROBE_QUEUE.log 2>&1 </dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
mkdir -p logs

stage () {  # stage <tag> <sweep args...>
  local tag=$1; shift
  if [ -e "logs/MORPH_PIPELINE_${tag}.DONE" ]; then
    echo "[queue] $(date '+%F %T') stage $tag already DONE — skip"; return 0
  fi
  echo "[queue] $(date '+%F %T') stage $tag START: $*"
  MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/morph_pipeline_sweep.py \
    --tag "$tag" "$@" > "logs/sweep_${tag}.run.log" 2>&1
  local rc=$?
  local done=no; [ -e "logs/MORPH_PIPELINE_${tag}.DONE" ] && done=yes
  echo "[queue] $(date '+%F %T') stage $tag END rc=$rc sentinel=$done"
}

# P1 RESCUE (~12-16 h): the 5 large16 failures under the STRONG evaluator — A best-of-2
# (retry on collapse/health-FAIL/never-lift; every attempt recorded = raw draw data) +
# PAIRED B recipes on the same kept A (imit AND self). H1 = failure flip-rate;
# H3 = imitation-prior fairness off-m05 (per-design cos delta imit vs self).
stage rescue --morph-set rescue --b-recipe both --a-attempts 2

# P2 AVAR (~8 h): raw A-draw distribution (NO retry — draws must stay uncensored):
# m05 x3 (control) + L01_05 x2 (pool with its P1 attempts), imit-B on each viable A.
# H2 = per-design P(A collapse)/P(health-FAIL) + cos spread across A draws -> is
# single-A(+retry) a sound landscape evaluator, or does P4 need >=2 A draws/design?
stage avar --morph-set avar --b-recipe imit --a-attempts 1

date '+%F %T' > logs/PROBE_QUEUE.DONE
echo "[queue] $(date '+%F %T') ALL STAGES DONE -> logs/PROBE_QUEUE.DONE"
