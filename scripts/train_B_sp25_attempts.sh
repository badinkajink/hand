#!/bin/bash
# Policy B on the 25 mm-proximal hand (sp25), FROM SCRATCH, best-of-N attempts.
# =============================================================================
# WHY FROM SCRATCH. b33's own config has init_actor_checkpoint=None -- it was trained from
# scratch at 20M ts, with a10 driving the live-A reset rather than seeding the actor. So the
# apples-to-apples length comparison is a from-scratch B on the sp25 hand under b33's recipe.
# (The separate b33-WARMSTARTED run, 20260819-1357-b_sp25_fromb33, answers a different and
# also useful question -- does the shipped reorienter adapt to the short hand -- and the answer
# is partially: zero-shot b33 drops 2/3 of rollouts on this hand, the finetune recovers hold to
# 1.00 and reorient to ~0.5.)
#
# WHY ATTEMPTS. From-scratch reorient is the hard-exploration target in this program: seeds
# converge to qualitatively different policies, peak reorient-cos anywhere in 0.0-0.9 (gotcha
# #7), and per-design draw sd is 0.3-0.5. Attempt 1 oscillated 0.02-0.12 without ever settling
# and tripped the collapse watchdog at iter 55. One failed draw is not a result either way.
#
# Launch: nohup setsid bash scripts/train_B_sp25_attempts.sh > logs/sp25/B_attempts.run.log 2>&1 </dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
ATTEMPTS=${ATTEMPTS:-3}
START=${START:-2}                     # attempt 1 already ran (collapsed)
A_CKPT="$ROOT/results/rl/20260819-1259-policyA_sp25_t2/tensorboard/model_609.pt"

for t in $(seq "$START" "$ATTEMPTS"); do
  TAG="b_sp25_scratch_t${t}"
  LOG="$ROOT/logs/sp25/B_sp25_scratch_t${t}.trainer.log"
  rm -f "${LOG}.COLLAPSED"
  echo "[sp25-B] === attempt $t/$ATTEMPTS -> $TAG"
  MORPH=results/phase1/shortprox25/20260819-sp25_ik_cem \
  A_CKPT="$A_CKPT" B_CKPT=none \
  LIFT_DELTA=0.1 ONSET_STEP=40 BLEND=0 LIFT_TERM_START=58 REORIENT_START=58 TIP_LOST_STEPS=10 \
  NUM_ENVS=3072 TOTAL_TS=20000000 RESID_SCALE=0.5 EASING=ease_out_quad CONTACT_GATE=1 \
  EXTRA_ARGS="--open-finger-from-keyframe" \
  TAG="$TAG" LOG="$LOG" bash scripts/train_handoff_liveA_reset.sh

  if [ -e "${LOG}.COLLAPSED" ]; then
    echo "[sp25-B] attempt $t COLLAPSED — retrying"
    for _ in $(seq 1 40); do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ "$u" -lt 2000 ] && break; sleep 15
    done
    continue
  fi
  echo "[sp25-B] attempt $t SURVIVED — keeping $TAG"
  echo "$TAG" > "$ROOT/logs/sp25/B_KEPT"
  break
done
echo "[sp25-B] done. kept=$(cat "$ROOT/logs/sp25/B_KEPT" 2>/dev/null || echo NONE)"
