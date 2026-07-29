#!/usr/bin/env bash
# Run 16: Try Lightning Grasp on the short-proximal drill scene.
#
# Pipeline (mirrors the cube MVP):
#   1. Build a URDF matching the drill's frozen morphology (MCP_LEN=0.025,
#      mounts from scene_power_drill_short_proximal.xml@open_flat).
#   2. Swap the default morphohand.urdf in lightning-grasp/assets to that
#      URDF (back up the cube one).
#   3. Run lightning_grasp_runner in lightning-grasp/.venv against the
#      drill OBJ (external/035_power_drill/poisson/textured.obj).
#   4. Score every Lightning grasp through Phase1GraspEvaluator on the
#      drill scene with the corrected 3-axis pivot deltas from run13/14/15.
#   5. Restore the original (cube) URDF.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
LIGHTNING_VENV_PY="${LIGHTNING_VENV_PY:-$ROOT_DIR/external/lightning-grasp/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: $PYTHON_BIN not executable"; exit 1
fi
if [[ ! -x "$LIGHTNING_VENV_PY" ]]; then
  echo "ERROR: $LIGHTNING_VENV_PY not executable -- need external/lightning-grasp/.venv"; exit 1
fi

SCENE_XML="${SCENE_XML:-assets/mjcf/baseline/scenes/scene_power_drill_short_proximal.xml}"
KEYFRAME="${KEYFRAME:-open_flat}"
DRILL_OBJ="${DRILL_OBJ:-$ROOT_DIR/external/035_power_drill/poisson/textured.obj}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/lightning_grasp/power_drill_short_proximal}"
BATCH_OUTER="${BATCH_OUTER:-256}"
BATCH_INNER="${BATCH_INNER:-128}"
N_SEEDS="${N_SEEDS:-4}"
N_CONTACT="${N_CONTACT:-3}"

URDF_DIR="external/lightning-grasp/assets/hand/morphohand"
URDF_DEFAULT="$URDF_DIR/morphohand.urdf"
URDF_DRILL="$URDF_DIR/morphohand_drill_short_proximal.urdf"
URDF_BACKUP="$URDF_DIR/morphohand.cube_backup.urdf"

mkdir -p "$OUTPUT_ROOT"

# ---- 1) build drill URDF -----------------------------------------------------
echo "[run16] === build drill URDF ==="
"$PYTHON_BIN" scripts/build_morphohand_urdf.py \
  --scene-xml "$SCENE_XML" --keyframe "$KEYFRAME" \
  --out "$URDF_DRILL"

# ---- 2) swap default URDF, with restore trap --------------------------------
if [[ ! -f "$URDF_BACKUP" && -f "$URDF_DEFAULT" ]]; then
  cp "$URDF_DEFAULT" "$URDF_BACKUP"
  echo "[run16] backed up cube URDF -> $URDF_BACKUP"
fi
cp "$URDF_DRILL" "$URDF_DEFAULT"
trap 'echo "[run16] restoring cube URDF"; [[ -f "$URDF_BACKUP" ]] && cp "$URDF_BACKUP" "$URDF_DEFAULT" || true' EXIT

# ---- 3) run lightning runner (multiple seeds, pool grasps) ------------------
echo "[run16] === run lightning ($N_SEEDS seeds, outer=$BATCH_OUTER inner=$BATCH_INNER) ==="
grasps_jsons=()
for seed in $(seq 1 $N_SEEDS); do
  out_json="$OUTPUT_ROOT/grasps_seed_${seed}.json"
  if [[ -f "$out_json" ]]; then
    echo "[run16] seed=$seed already present, skipping"
  else
    echo "[run16] --- seed=$seed ---"
    (cd external/lightning-grasp && "$LIGHTNING_VENV_PY" "$ROOT_DIR/scripts/lightning_grasp_runner.py" \
      --robot morphohand \
      --object_mesh_path "$DRILL_OBJ" \
      --output_json "$ROOT_DIR/$out_json" \
      --batch_size_outer "$BATCH_OUTER" \
      --batch_size_inner "$BATCH_INNER" \
      --n_contact "$N_CONTACT" \
      --object_pose_sampling_strategy canonical)
  fi
  grasps_jsons+=("$out_json")
done

# ---- 4) merge JSONs ---------------------------------------------------------
merged_json="$OUTPUT_ROOT/grasps_merged.json"
echo "[run16] === merge ${#grasps_jsons[@]} JSONs -> $merged_json ==="
"$PYTHON_BIN" - "$merged_json" "${grasps_jsons[@]}" <<'PY'
import json, sys
out_path, *paths = sys.argv[1:]
merged = []
robot = None; obj = None; n_contact = None; wall = 0.0
for p in paths:
    d = json.load(open(p))
    merged.extend(d["grasps"])
    robot = d.get("robot", robot)
    obj = d.get("object_mesh_path", obj)
    n_contact = d.get("n_contact", n_contact)
    wall += float(d.get("wall_time_s", 0.0))
out = {
    "robot": robot,
    "object_mesh_path": obj,
    "n_contact": n_contact,
    "n_grasps": len(merged),
    "wall_time_s": wall,
    "grasps": merged,
}
with open(out_path, "w") as f:
    json.dump(out, f)
print(f"merged {len(merged)} grasps across {len(paths)} JSONs (total Lightning wall {wall:.1f}s)")
PY

# ---- 5) score through Phase1GraspEvaluator ----------------------------------
# Use the same dynamics protocol as runs 13-15 (lift+pivot+hold with the
# correct 3-axis pivot deltas) so scores compare apples-to-apples.
eval_json="$OUTPUT_ROOT/eval.json"
echo "[run16] === score ${#grasps_jsons[@]} pooled grasps through MuJoCo ==="
"$PYTHON_BIN" scripts/lightning_grasp_eval.py \
  --grasps-json "$merged_json" \
  --scene-xml "$SCENE_XML" \
  --keyframe "$KEYFRAME" \
  --output-json "$eval_json" \
  --settle-steps 240 \
  --lift-steps 220 \
  --hold-steps 140 \
  --lift-delta-z 0.110 \
  --lift-ramp-steps 180 \
  --pivot-steps 180 \
  --pivot-ramp-steps 120 \
  --pivot-delta-rx -1.6 \
  --pivot-delta-ry -1.4 \
  --pivot-delta-rz 1.6018 \
  --mode ctrl_only

# ---- 6) summary ------------------------------------------------------------
echo "[run16] === summary ==="
"$PYTHON_BIN" - "$eval_json" <<'PY'
import json, sys
import numpy as np
e = json.load(open(sys.argv[1]))
results = e.get("results") or e.get("evaluations") or e
if isinstance(results, dict) and "results" in results:
    results = results["results"]
scores = [r["score"] for r in results if isinstance(r, dict) and "score" in r]
if not scores:
    print(f"(no scores parsed; eval json keys = {list(e)[:8]})")
    raise SystemExit
arr = np.array(scores, dtype=np.float64)
print(f"n_grasps = {len(arr)}")
print(f"best     = {arr.max():+.3f}")
print(f"median   = {float(np.median(arr)):+.3f}")
print(f"mean     = {arr.mean():+.3f} +/- {arr.std():.3f}")
print(f"frac >0  = {float((arr > 0).mean()):.2%}")
PY
