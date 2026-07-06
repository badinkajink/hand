"""Score GraspGenX grasps on the real morphohand via Phase1GraspEvaluator.

GraspGenX emits gripper poses in the object frame. This converts each to an
object-pose-in-palm-frame, places the object there, sets the fingers to the
gripper's close config, and runs the codebase's own grasp evaluator
(settle -> lift via palm_pz) on a FROZEN scene. Reports per-grasp cube lift.

Frame: GraspGenX pose T is the (canonical) gripper pose in the object frame.
palm-in-object = T @ base_rotation; object-in-palm = inv(palm-in-object). We
place the object at the ACTUAL palm world transform (read after reset) so the
nonzero palm_px in the keyframe is handled correctly.

    uv run python scripts/graspgenx_eval_phase1.py \
        --grasps /tmp/mh_cube.yml \
        --gripper-dir external/GraspGenX/assets/x_grippers/morphohand \
        --topk 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh.transformations as tra
import yaml

from morphohand.optimization.phase1_common import Phase1EvalConfig, Phase1GraspEvaluator
from morphohand.sampling.scene import freeze_scene_for_eval
from morphohand.tools.morphology_xml import (
    create_rigid_morphology_xml,
    extract_morphology_from_qpos,
)

ROOT = Path(__file__).resolve().parents[1]


def _frozen_with_close_len(scene_xml: Path, keyframe: str, close_len: dict, out: Path) -> Path:
    """Freeze the scene but bake `len` at the grasp (close) value, since
    Phase1GraspEvaluator can't actuate the len Z-joint. x/y stay at design."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    qpos = list(m.key_qpos[kid])
    morph = extract_morphology_from_qpos(qpos, has_scene_prefix=True)
    morph = morph.__class__(
        thumb_x=morph.thumb_x, thumb_y=morph.thumb_y, thumb_len=close_len["thumb"],
        index_x=morph.index_x, index_y=morph.index_y, index_len=close_len["index"],
        middle_x=morph.middle_x, middle_y=morph.middle_y, middle_len=close_len["middle"],
    )
    create_rigid_morphology_xml(scene_xml, morph, out, model_name=out.stem)
    return out


def load_poses(yml: Path, topk: int):
    d = yaml.safe_load(yml.read_text())
    gs = sorted(d["grasps"].values(), key=lambda g: -g["confidence"])[:topk]
    out = []
    for g in gs:
        T = tra.quaternion_matrix([g["orientation"]["w"], *g["orientation"]["xyz"]])
        T[:3, 3] = g["position"]
        out.append((T, float(g["confidence"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", type=Path, required=True)
    ap.add_argument("--gripper-dir", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--success-lift", type=float, default=0.04,
                    help="cube_lift (m) above which a grasp counts as held (lift_delta_z=0.05).")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--bake-close-len", action="store_true",
                    help="Bake len at the grasp (close) value into the frozen scene "
                         "(len is the Z-actuation joint; Phase1GraspEvaluator can't actuate it).")
    args = ap.parse_args()

    side = json.loads((args.gripper_dir / "morphohand_grasp.json").read_text())
    config = json.loads((args.gripper_dir / "config.json").read_text())
    scene_xml = Path(side["scene_xml"])
    keyframe = side["keyframe"]
    close_ctrl = np.asarray(side["close_finger_ctrl"], dtype=np.float64)
    base_rot = np.asarray(side["base_rotation"], dtype=np.float64)  # 4x4 palm<-canonical

    work = ROOT / "results" / "graspgenx_eval"
    work.mkdir(parents=True, exist_ok=True)
    if args.bake_close_len:
        close_len = {f: config["close"][f"{f}_len"] for f in ("thumb", "index", "middle")}
        frozen = work / f"{scene_xml.stem}__{keyframe}.frozen_len.xml"
        _frozen_with_close_len(scene_xml, keyframe, close_len, frozen)
    else:
        frozen = work / f"{scene_xml.stem}__{keyframe}.frozen.xml"
        freeze_scene_for_eval(scene_xml, keyframe, frozen)

    ev = Phase1GraspEvaluator(scene_xml=frozen, keyframe=keyframe, cfg=Phase1EvalConfig())

    # object free-joint qpos address + palm body id
    obj_jadr = int(ev.model.jnt_qposadr[int(ev.model.body_jntadr[ev.cube_body_id])])
    palm_bid = mujoco.mj_name2id(ev.model, mujoco.mjtObj.mjOBJ_BODY, "palm_pose")
    if palm_bid < 0:
        palm_bid = mujoco.mj_name2id(ev.model, mujoco.mjtObj.mjOBJ_BODY, "palm")

    original_reset = ev._reset_to_keyframe

    def patched_reset():
        original_reset()
        obj_in_palm = getattr(ev, "_obj_in_palm", None)
        if obj_in_palm is None:
            return
        # actual palm world transform after reset (includes palm_px etc.)
        Tpw = np.eye(4)
        Tpw[:3, :3] = ev.data.xmat[palm_bid].reshape(3, 3)
        Tpw[:3, 3] = ev.data.xpos[palm_bid]
        Tow = Tpw @ obj_in_palm
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, Tow[:3, :3].flatten())
        ev.data.qpos[obj_jadr:obj_jadr + 3] = Tow[:3, 3]
        ev.data.qpos[obj_jadr + 3:obj_jadr + 7] = q
        ev.data.qpos[ev.finger_joint_qpos_ids] = close_ctrl
        mujoco.mj_forward(ev.model, ev.data)

    ev._reset_to_keyframe = patched_reset

    grasps = load_poses(args.grasps, args.topk)
    print(f"object_scene={scene_xml.name}  grasps={len(grasps)}")
    results, n_ok = [], 0
    for i, (T, conf) in enumerate(grasps):
        palm_in_obj = T @ base_rot
        ev._obj_in_palm = np.linalg.inv(palm_in_obj)
        score, diag = ev.evaluate(close_ctrl)
        lift = float(diag.get("cube_lift", float("nan")))
        ok = lift > args.success_lift
        n_ok += ok
        results.append({"idx": i, "conf": conf, "score": float(score), "cube_lift": lift, "held": bool(ok)})
        print(f"  grasp {i:2d} conf={conf:.3f} score={score:+.3f} cube_lift={lift:+.4f} "
              f"{'HOLD' if ok else 'drop'}")
    print(f"=> {n_ok}/{len(grasps)} held (cube_lift > {args.success_lift} m)")

    if args.output:
        args.output.write_text(json.dumps(
            {"scene": str(scene_xml), "n_held": n_ok, "n": len(grasps), "per_grasp": results}, indent=2))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
