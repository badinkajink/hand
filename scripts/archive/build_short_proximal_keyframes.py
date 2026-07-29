"""Run Lightning Grasp on each non-drill eval object and write a closing
keyframe into its `_short_proximal` scene.

Pipeline per object:
  1. Build URDF from the object's `_short_proximal` scene at its existing
     keyframe (morphology baked in so Lightning sees the correct hand).
  2. Swap that URDF in for Lightning's default `morphohand.urdf`.
  3. Run `scripts/lightning_grasp_runner.py` inside `external/lightning-grasp/.venv`
     against the object's mesh, producing a JSON of candidate grasps.
  4. Restore the original morphohand.urdf.
  5. Score every candidate via `scripts/lightning_grasp_eval.py` in
     `init_pose` mode against the same `_short_proximal` scene.
  6. Pick the best grasp by score.
  7. Load the scene, replay the best grasp's init pose (palm + finger
     joints + object), step a short settle, then capture full qpos + ctrl
     and emit a new `<key>` element into the scene XML.

Existing keyframes in each scene are preserved. The new keyframe name is
``open_short`` per object (consistent across the eval set).

Designed to be run end-to-end in one shot; objects can be filtered with
`--objects label1,label2`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

URDF_DIR = ROOT / "external" / "lightning-grasp" / "assets" / "hand" / "morphohand"
URDF_DEFAULT = URDF_DIR / "morphohand.urdf"
URDF_BACKUP = URDF_DIR / "morphohand.cube_backup.urdf"
LIGHTNING_VENV_PY = ROOT / "external" / "lightning-grasp" / ".venv" / "bin" / "python"
MAIN_VENV_PY = ROOT / ".venv" / "bin" / "python"

OBJECT_ASSETS_DIR = ROOT / "external" / "lightning-grasp" / "assets" / "object" / "morphohand"


@dataclass(frozen=True)
class ObjectSpec:
    label: str
    short_scene: Path                  # base scene with short proximal + morph joints
    original_keyframe: str             # keyframe to base the URDF on (and pose the palm)
    object_body: str
    mesh_obj: Path                     # OBJ for Lightning
    new_keyframe_name: str = "open_short"


SPECS: list[ObjectSpec] = [
    ObjectSpec(
        label="cube",
        short_scene=ROOT / "assets/mjcf/baseline/scenes/scene_cube_short_proximal.xml",
        original_keyframe="open",
        object_body="cube",
        mesh_obj=OBJECT_ASSETS_DIR / "cube_40mm.obj",
    ),
    ObjectSpec(
        label="prism",
        short_scene=ROOT / "assets/mjcf/baseline/scenes/scene_prism_short_proximal.xml",
        original_keyframe="open",
        object_body="prism",
        mesh_obj=OBJECT_ASSETS_DIR / "prism_22x68x18mm.obj",
    ),
    ObjectSpec(
        label="screwdriver_medium_flat",
        short_scene=ROOT / "assets/mjcf/baseline/scenes/scene_screwdriver_medium_flat_short_proximal.xml",
        original_keyframe="open",
        object_body="screwdriver_medium",
        mesh_obj=OBJECT_ASSETS_DIR / "screwdriver_medium_25x100mm.obj",
    ),
    ObjectSpec(
        label="screwdriver_medium_vertical",
        short_scene=ROOT / "assets/mjcf/baseline/scenes/scene_screwdriver_medium_vertical_short_proximal.xml",
        original_keyframe="open",
        object_body="screwdriver_medium",
        mesh_obj=OBJECT_ASSETS_DIR / "screwdriver_medium_25x100mm.obj",
    ),
    ObjectSpec(
        label="screwdriver_small_flat",
        short_scene=ROOT / "assets/mjcf/baseline/scenes/scene_screwdriver_small_flat_short_proximal.xml",
        original_keyframe="open",
        object_body="screwdriver_small",
        mesh_obj=OBJECT_ASSETS_DIR / "screwdriver_small_8x80mm.obj",
    ),
]


def _run(cmd: list[str], *, env: dict | None = None, cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def build_urdf_for_spec(spec: ObjectSpec, urdf_out: Path) -> None:
    _run([
        str(MAIN_VENV_PY), str(ROOT / "scripts/build_morphohand_urdf.py"),
        "--scene-xml", str(spec.short_scene),
        "--keyframe", spec.original_keyframe,
        "--out", str(urdf_out),
    ])


def run_lightning(spec: ObjectSpec, output_json: Path,
                  batch_outer: int = 128, batch_inner: int = 64, n_contact: int = 3) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(LIGHTNING_VENV_PY),
        str(ROOT / "scripts/lightning_grasp_runner.py"),
        "--robot", "morphohand",
        "--object_mesh_path", str(spec.mesh_obj),
        "--output_json", str(output_json),
        "--batch_size_outer", str(batch_outer),
        "--batch_size_inner", str(batch_inner),
        "--n_contact", str(n_contact),
    ]
    _run(cmd)


def score_grasps(spec: ObjectSpec, grasps_json: Path, eval_json: Path,
                 settle_steps: int = 200, lift_steps: int = 200, hold_steps: int = 60,
                 lift_delta_z: float = 0.05, lift_ramp_steps: int = 100) -> dict:
    eval_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(MAIN_VENV_PY), str(ROOT / "scripts/lightning_grasp_eval.py"),
        "--grasps-json", str(grasps_json),
        "--scene-xml", str(spec.short_scene),
        "--keyframe", spec.original_keyframe,
        "--output-json", str(eval_json),
        "--settle-steps", str(settle_steps),
        "--lift-steps", str(lift_steps),
        "--hold-steps", str(hold_steps),
        "--lift-delta-z", str(lift_delta_z),
        "--lift-ramp-steps", str(lift_ramp_steps),
        "--mode", "init_pose",
    ]
    _run(cmd)
    return json.loads(eval_json.read_text())


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> mujoco-style (qw, qx, qy, qz)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def capture_keyframe_from_best(spec: ObjectSpec, best_grasp_q: list[float],
                               best_object_pose: list[list[float]], settle_steps: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Replay Lightning's init pose on the short scene and capture qpos+ctrl."""
    model = mujoco.MjModel.from_xml_path(str(spec.short_scene))
    data = mujoco.MjData(model)

    # Start from original keyframe so morph qpos + palm pose are sane.
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, spec.original_keyframe)
    if kid < 0:
        raise RuntimeError(f"keyframe {spec.original_keyframe} not found in {spec.short_scene}")
    mujoco.mj_resetDataKeyframe(model, data, kid)

    # Apply Lightning init: object pose (palm-frame) + finger joints.
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.object_body)
    palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm_pose")
    palm_pos = model.body_pos[palm_bid].copy()

    obj_pose_palm = np.asarray(best_object_pose, dtype=np.float64)  # 4x4 row-major
    world_pos = palm_pos + obj_pose_palm[:3, 3]
    world_quat = _rot_to_quat(obj_pose_palm[:3, :3])

    obj_jid = int(model.body_jntadr[obj_body_id])
    obj_qposadr = int(model.jnt_qposadr[obj_jid])
    data.qpos[obj_qposadr:obj_qposadr + 3] = world_pos
    data.qpos[obj_qposadr + 3:obj_qposadr + 7] = world_quat

    # Set finger joints to q. Joint name order is the same as the
    # `FINGER_ACTUATOR_NAMES` used elsewhere.
    finger_joint_names = [
        "thumb_yaw", "thumb_mcp", "thumb_pip",
        "index_yaw", "index_mcp", "index_pip",
        "middle_yaw", "middle_mcp", "middle_pip",
    ]
    for jname, qv in zip(finger_joint_names, best_grasp_q):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        data.qpos[model.jnt_qposadr[jid]] = float(qv)
    # Set ctrl to match.
    for jname, qv in zip(finger_joint_names, best_grasp_q):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{jname}")
        if aid >= 0:
            data.ctrl[aid] = float(qv)

    mujoco.mj_forward(model, data)

    # Brief settle to let small interpenetrations resolve.
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    return data.qpos.copy(), data.ctrl.copy()


def write_keyframe(scene_xml: Path, key_name: str, qpos: np.ndarray, ctrl: np.ndarray) -> None:
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")

    # Drop any existing key with the same name.
    for k in list(keyframe.findall("key")):
        if k.get("name") == key_name:
            keyframe.remove(k)

    qpos_str = "\n        " + " ".join(f"{v:.6f}" for v in qpos) + "\n      "
    ctrl_str = "\n        " + " ".join(f"{v:.6f}" for v in ctrl) + "\n      "
    elem = ET.SubElement(keyframe, "key")
    elem.set("name", key_name)
    elem.set("qpos", qpos_str)
    elem.set("ctrl", ctrl_str)

    ET.indent(root, space="  ")
    tree.write(scene_xml, encoding="utf-8", xml_declaration=False)


def process_one(spec: ObjectSpec, workdir: Path, batch_outer: int, batch_inner: int) -> dict:
    print(f"\n{'=' * 80}\n[{spec.label}] starting\n{'=' * 80}")
    t0 = time.time()
    obj_workdir = workdir / spec.label
    obj_workdir.mkdir(parents=True, exist_ok=True)

    # 1) URDF
    urdf_path = URDF_DIR / f"morphohand_{spec.label}_short_proximal.urdf"
    build_urdf_for_spec(spec, urdf_path)

    # 2) swap URDF in
    try:
        if URDF_DEFAULT.exists() and not URDF_BACKUP.exists():
            shutil.copy(URDF_DEFAULT, URDF_BACKUP)
        shutil.copy(urdf_path, URDF_DEFAULT)

        # 3) Lightning
        grasps_json = obj_workdir / "grasps.json"
        run_lightning(spec, grasps_json, batch_outer=batch_outer, batch_inner=batch_inner)
    finally:
        # 4) restore URDF — always, even on failure
        if URDF_BACKUP.exists():
            shutil.copy(URDF_BACKUP, URDF_DEFAULT)

    # 5) score
    eval_json = obj_workdir / "eval.json"
    summary = score_grasps(spec, grasps_json, eval_json)

    best_idx = summary["best_idx"]
    best_q = summary["best_q"]
    best_grasp_full = summary["per_grasp"][best_idx]
    # The eval JSON only stores q; recover object_pose from the grasps JSON.
    grasps_data = json.loads(grasps_json.read_text())
    best_object_pose = grasps_data["grasps"][best_idx]["object_pose"]

    # 6) capture qpos+ctrl and write keyframe
    qpos, ctrl = capture_keyframe_from_best(spec, best_q, best_object_pose)
    write_keyframe(spec.short_scene, spec.new_keyframe_name, qpos, ctrl)

    elapsed = time.time() - t0
    print(f"\n[{spec.label}] DONE in {elapsed:.1f}s — best={summary['best_score']:+.3f} "
          f"median={summary['median_score']:+.3f}; wrote keyframe '{spec.new_keyframe_name}' "
          f"to {spec.short_scene.name}")
    return {
        "label": spec.label,
        "best_score": summary["best_score"],
        "median_score": summary["median_score"],
        "best_idx": best_idx,
        "wall_time_s": elapsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=Path, default=ROOT / "results/short_proximal_keyframes")
    ap.add_argument("--objects", type=str, default="",
                    help="Comma-separated subset of object labels; empty means all")
    ap.add_argument("--batch-outer", type=int, default=128)
    ap.add_argument("--batch-inner", type=int, default=64)
    args = ap.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    chosen = [s.label for s in SPECS]
    if args.objects:
        wanted = {x.strip() for x in args.objects.split(",") if x.strip()}
        chosen = [lab for lab in chosen if lab in wanted]

    summary = []
    for spec in SPECS:
        if spec.label not in chosen:
            continue
        try:
            summary.append(process_one(spec, args.workdir,
                                        batch_outer=args.batch_outer,
                                        batch_inner=args.batch_inner))
        except subprocess.CalledProcessError as e:
            print(f"\n[{spec.label}] FAILED: {e}")
            summary.append({"label": spec.label, "error": str(e)})

    print("\n=== END-TO-END SUMMARY ===")
    for r in summary:
        if "error" in r:
            print(f"  {r['label']:30s} ERROR: {r['error']}")
        else:
            print(f"  {r['label']:30s} best={r['best_score']:+.3f}  median={r['median_score']:+.3f}  "
                  f"({r['wall_time_s']:.1f}s)")
    (args.workdir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
