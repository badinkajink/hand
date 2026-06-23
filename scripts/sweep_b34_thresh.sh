#!/bin/bash
# B34 — FORCE-FLOOR SWEEP: how low can the grip go before it breaks?
# =============================================================================
# b33 (thresh 4 N, w -6) HELD but stalled: force 11->7.5 N (steady ~5-6 N), gentler
# + smoother (ang-jerk 113->49, wander 23->8.9 cm) but cost verticality (cos
# 0.891->0.845) and STILL penetrates. This sweep continues FROM b33 and only lowers
# the penalty threshold to find the floor — where holding A's delivery actually breaks.
#
#   thresh 3.0  -> B3's actual gentle target; the mildest push past b33
#   thresh 2.5  -> moderate
#   thresh 2.0  -> aggressive; most likely to drop / wreck verticality
#
# All: warmstart b33/model_405 (actor+critic), weight -6 fixed, EVERYTHING else == b33.
# Per-run WARP_CACHE (already in train_handoff_b33_forcereg.sh) + 75s stagger so the
# Warp kernel compiles don't thundering-herd the GPU. 2048 envs each; 3x fits 16 GB.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
B33="$ROOT/results/rl/20260612-1750-policyB_b32iter_forcereg_w6/tensorboard/model_405.pt"
[ -e "$B33" ] || { echo "FATAL missing b33 ckpt $B33"; exit 1; }

launch () { # $1=thresh  $2=tagsuffix
  local th="$1" tag="$2"
  echo "[sweep] launching thresh=$th tag=$tag $(date +%H:%M:%S)"
  B_CKPT="$B33" PEN_THRESH="$th" PEN_WEIGHT=-6.0 PEN_SCALE=4.0 \
    TAG="policyB_b33iter_forcefloor_${tag}" \
    LOG="$ROOT/b34_${tag}.trainer.log" \
    nohup setsid bash "$ROOT/scripts/train_handoff_b33_forcereg.sh" \
    > "$ROOT/b34_${tag}.run.log" 2>&1 </dev/null &
  disown
}

launch 3.0 t30; sleep 75
launch 2.5 t25; sleep 75
launch 2.0 t20
echo "[sweep] all 3 launched. logs: b34_t{30,25,20}.trainer.log"
