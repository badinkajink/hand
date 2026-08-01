#!/usr/bin/env bash
# Init-NaN RATE on the GPU, with vs without the palm<->finger excludes.
#
# Why this exists: CPU MuJoCo cannot see the effect. With noise on the FINGER ctrl DOF (what
# the policy actually perturbs) neither scene diverges in 500 seeds; with noise on all 15 ctrl
# DOF both diverge at 35.7% vs 34.7% — indistinguishable, and the blow-up is the screwdriver's
# free joint in both. Yet the GPU trainer NaN'd at iteration 0 on 4/4 attempts with the
# excludes and 0/2 without. Whatever the asymmetry is, it lives in MJWarp, so it has to be
# measured there. One trainer launch = one Bernoulli sample; this just repeats it.
#
# TWO failure modes, one experiment. The excludes arm shows both — NaN at iteration 0 (shipped
# design, 3/3 in the r6 queue) and, when it does start, a completely flat object height pinned
# at spawn 0.0123 that trips the watchdog at iteration 60 (2/2 among r6 runs that got that far).
# r4 — the same shipped morphology WITHOUT excludes — did neither, but n=1. So each sample runs
# to iteration 65: past the watchdog point, and it classifies as nan / collapsed / lifting.
# ~10 min per sample (8 min training + ~1.5 min Warp compile).
#
#   bash scripts/probe_init_nan_rate.sh [N] [ITERS]     # default N=3, ITERS=65
set -uo pipefail
cd "$(dirname "$0")/.."
N="${1:-3}"
ITERS="${2:-65}"
TS=$(( ITERS * 3072 * 24 ))
OUT="logs/init_nan_rate_$(date +%Y%m%d-%H%M).log"
NOEXCL="results/phase1/perp/perp_v1/frozen_scene.xml"
EXCL="results/rl/perp_compact_queue/t0.00_x0.00_y0.00/frozen_scene.xml"

# Both are the SHIPPED morphology (all nine params zero); they differ only in the exclude
# block. Verify that before trusting any rate difference.
echo "scene diff (expect only <exclude> lines, model name, keyframe whitespace):" | tee -a "$OUT"
diff <(sed 's/^[ \t]*//' "$NOEXCL") <(sed 's/^[ \t]*//' "$EXCL") | grep -E "^[<>]" | grep -vcE "exclude|<contact>|</contact>|mujoco model|key name" \
  | xargs -I{} echo "  non-exclude differing lines: {}" | tee -a "$OUT"

for arm in noexcl excl; do
  scene=$NOEXCL; [[ $arm == excl ]] && scene=$EXCL
  nan=0
  for i in $(seq 1 "$N"); do
    log="logs/nanrate_${arm}_$i.log"
    export WARP_CACHE_PATH="$(mktemp -d)"
    MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/rl_train_cube.py \
        --recipe perp_single --morphology-run results/phase1/perp/perp_v1 \
        --frozen-scene-xml "$scene" --tag "nanrate_${arm}_$i" \
        --num-envs 3072 --total-timesteps "$TS" --no-wandb \
        >"$log" 2>&1
    rc=$?
    rm -rf "$WARP_CACHE_PATH"
    z=$(grep -oE "lift_height/object_height: [0-9.]+" "$log" | tail -1 | awk '{print $2}')
    if grep -q "contains NaN values" "$log"; then
      nan=$((nan+1)); res="NaN@iter$(grep -oE 'Learning iteration [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+')"
    elif [[ $rc -ne 0 ]]; then res="rc=$rc (obj_z=$z)"
    # 0.04 is the queue's watchdog threshold: r4 never drops below 0.0556 after iter 60,
    # every dead r6 run sat at 0.0123.
    elif awk "BEGIN{exit !($z < 0.04)}"; then res="COLLAPSED obj_z=$z"
    else res="LIFTING obj_z=$z"
    fi
    echo "  $arm run $i: $res" | tee -a "$OUT"
    # let the previous run's VRAM actually free before the next launch
    for _ in $(seq 1 30); do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [[ "$u" -lt 2500 ]] && break; sleep 5
    done
  done
  echo "== $arm: $nan/$N NaN ==" | tee -a "$OUT"
done
echo "wrote $OUT"
