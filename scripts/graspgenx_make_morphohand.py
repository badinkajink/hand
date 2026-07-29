"""Non-interactive GraspGenX gripper exporter for the morphohand (MuJoCo-based).

GraspGenX onboards a gripper from a URDF via an interactive Viser wizard that
writes config.json (sweep-volume boxes + open/close joint states + gripper
type); at inference `get_gripper_info` reads ONLY config.json (+ meshes), never
the URDF, and the released model conditions on the sweep_volume_v2 boxes.

This reproduces the wizard head-less straight from our real MuJoCo hand, so the
gripper GraspGenX sees is the ACTUAL actuated morphohand, not a flat URDF:

  * morphology (thumb/index/middle x/y mounts) is whatever the source scene's
    keyframe encodes and is FROZEN here (we never mutate it),
  * the gripper "open" config is the scene's ready keyframe posture (fingers
    pre-flexed), NOT all-zeros (that produced degenerate, unactuated grasps),
  * "close" flexes yaw/mcp/pip further AND extends `len` (the per-finger
    Z-actuation joint) so the swept volume reflects a real grasp,
  * base_rotation = diag(-1,1,-1) maps the palm frame into GraspGenX's canonical
    convention: palm -z -> +Z approach (the hand cups downward), palm -x -> +X
    closing (thumb opposes index+middle along x).

Outputs into <out>/<name>/: config.json, coll_mesh.obj, vis_mesh.obj, and a
sidecar morphohand_grasp.json (close finger ctrl + base_rotation + scene) that
graspgenx_eval_phase1.py uses to score the generated poses on the real hand.

Run in the project uv env (mujoco 3.6):
    uv run python scripts/graspgenx_make_morphohand.py \
        --scene-xml assets/mjcf/baseline/scenes/scene.xml --keyframe open --name morphohand
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
GRASPGENX_ROOT = ROOT / "external" / "GraspGenX"

# palm-frame -> GraspGenX canonical (+Z approach, +X closing). 180 deg about Y.
BASE_ROTATION = np.diag([-1.0, 1.0, -1.0])

FINGERS = ("thumb", "index", "middle")
# Finger DOF read/written per finger. yaw/mcp/pip are the actuated grasp joints;
# len is the Z-actuation morphology slide (extends during close).
DOF = ("yaw", "mcp", "pip", "len")

# The scene `open` keyframe IS the closed grasp: the fingertips already
# surround the object (thumb opposes index+middle, each tip ~4 cm from the
# cube centre). The grasp CLOSES by the index/middle tips swinging down onto
# the object while the thumb holds the opposing wall. So the GraspGenX
# open->close motion is keyframe-with-mcp-reduced (spread) -> keyframe (cup);
# flexing PAST the keyframe drives every tip ~10 cm away from the object.
OPEN_MCP_DELTA = -0.8  # subtract from each finger's keyframe mcp to spread for the pre-grasp pose


def jadr(model, name):
    return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]


def read_open_posture(model, data, key_id) -> dict:
    """Finger DOF values at the ready keyframe."""
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    out = {}
    for f in FINGERS:
        out[f] = {d: float(data.qpos[jadr(model, f"{f}_{d}")]) for d in DOF}
    return out


def set_posture(model, data, key_id, posture):
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    for f in FINGERS:
        for d in DOF:
            data.qpos[jadr(model, f"{f}_{d}")] = posture[f][d]
    mujoco.mj_forward(model, data)


def palm_frame(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm_pose")
    if bid < 0:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    return bid, data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def tip_positions_canonical(model, data):
    _, pp, pR = palm_frame(model, data)
    pts = []
    for f in FINGERS:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{f}_tip")
        pts.append(BASE_ROTATION @ (pR.T @ (data.xpos[b] - pp)))
    return np.array(pts)


def box_from_points(point_sets, margin=0.012, floor=(0.04, 0.04, 0.04)):
    pts = np.vstack(point_sets)
    lo, hi = pts.min(0), pts.max(0)
    extents = np.maximum((hi - lo) + 2 * margin, np.array(floor))
    offset = (lo + hi) / 2.0
    return extents.tolist(), offset.tolist()


def export_hand_mesh(model, data, out_path: Path):
    """Build a trimesh of the palm+finger geoms at the current posture, expressed
    in the GraspGenX canonical frame (palm-relative, base_rotation applied)."""
    _, pp, pR = palm_frame(model, data)
    Rc = BASE_ROTATION
    meshes = []
    for gi in range(model.ngeom):
        bid = model.geom_bodyid[gi]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not (bname.startswith(FINGERS) or "palm" in bname):
            continue
        gtype = model.geom_type[gi]
        size = model.geom_size[gi]
        gpos = pR.T @ (data.geom_xpos[gi] - pp)            # palm frame
        gR = pR.T @ data.geom_xmat[gi].reshape(3, 3)
        if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            m = trimesh.creation.capsule(height=2 * size[1], radius=size[0])
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            m = trimesh.creation.icosphere(radius=size[0])
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            m = trimesh.creation.box(extents=2 * size[:3])
        else:
            continue
        T = np.eye(4); T[:3, :3] = gR; T[:3, 3] = gpos     # geom -> palm
        Tc = np.eye(4); Tc[:3, :3] = Rc                    # palm -> canonical
        m.apply_transform(Tc @ T)
        meshes.append(m)
    merged = trimesh.util.concatenate(meshes)
    merged.export(str(out_path))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-xml", type=Path, default=ROOT / "assets" / "mjcf" / "baseline" / "scenes" / "scene.xml")
    ap.add_argument("--keyframe", default="open")
    ap.add_argument("--name", default="morphohand")
    ap.add_argument("--out", type=Path, default=GRASPGENX_ROOT / "assets" / "x_grippers")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene_xml))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if key_id < 0:
        raise SystemExit(f"keyframe '{args.keyframe}' not in {args.scene_xml}")

    # close (grasp) = keyframe posture; open (pre-grasp) = keyframe with mcp
    # reduced so the index/middle tips swing up/out (fingers spread).
    close_p = read_open_posture(model, data, key_id)
    open_p = {f: dict(close_p[f]) for f in FINGERS}
    for f in FINGERS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_mcp")
        lo, hi = model.jnt_range[jid]
        open_p[f]["mcp"] = float(np.clip(close_p[f]["mcp"] + OPEN_MCP_DELTA, lo, hi))
    half_p = {f: {d: 0.5 * (open_p[f][d] + close_p[f][d]) for d in DOF} for f in FINGERS}

    set_posture(model, data, key_id, open_p);  tips_open = tip_positions_canonical(model, data)
    set_posture(model, data, key_id, half_p);  tips_half = tip_positions_canonical(model, data)
    set_posture(model, data, key_id, close_p); tips_close = tip_positions_canonical(model, data)

    # The object is held in the CLOSE (cup) region. Centre the sweep boxes
    # there. open box = spread->cup region (max aperture), mid box = half->cup.
    # Depth toward the palm comes from including the palm origin.
    palm0 = np.zeros((1, 3))
    sv_extents, sv_offset = box_from_points([tips_open, tips_close, palm0])
    sv2_extents, sv2_offset = box_from_points([tips_half, tips_close, palm0])

    # bbox of the whole hand in canonical frame at open posture
    dest = args.out / args.name
    dest.mkdir(parents=True, exist_ok=True)
    set_posture(model, data, key_id, open_p)
    mesh = export_hand_mesh(model, data, dest / "coll_mesh.obj")
    mesh.export(str(dest / "vis_mesh.obj"))
    bbox_min, bbox_max = mesh.bounds.tolist()

    print(f"tips_open  centroid={np.round(tips_open.mean(0),4)}")
    print(f"tips_close centroid={np.round(tips_close.mean(0),4)}")
    print(f"open sweep  extents={np.round(sv_extents,4)} offset={np.round(sv_offset,4)}")
    print(f"half sweep  extents={np.round(sv2_extents,4)} offset={np.round(sv2_offset,4)}")
    print(f"bbox min={np.round(bbox_min,4)} max={np.round(bbox_max,4)}")

    # config.json — joints keyed by name so they FK-match the URDF if ever used;
    # the released model only consumes the sweep boxes + type though.
    def jdict(p):
        return {f"{f}_{d}": p[f][d] for f in FINGERS for d in DOF}

    config = {
        "open": jdict(open_p),
        "close": jdict(close_p),
        "fingertip": list(sv_offset),
        "sweep_volume": {"extents": sv_extents, "offset": sv_offset,
                         "extents2": sv2_extents, "offset2": sv2_offset},
        "links": [], "standoff": [0.0, sv_extents[2] / 2],
        "symmetric": False, "type": "revolute_3f",
        "bbox": [bbox_min, bbox_max],
        "base_rotation": np.eye(4).tolist(),  # mesh already baked in canonical frame
    }
    (dest / "config.json").write_text(json.dumps(config, indent=4))

    # sidecar for Phase1 eval: close finger ctrl (9, in finger_joint order) + the
    # palm<-canonical base_rotation used to convert GraspGenX poses.
    finger_order = ["thumb_yaw", "thumb_mcp", "thumb_pip",
                    "index_yaw", "index_mcp", "index_pip",
                    "middle_yaw", "middle_mcp", "middle_pip"]
    close_ctrl = [close_p[j.split("_")[0]][j.split("_")[1]] for j in finger_order]
    R4 = np.eye(4); R4[:3, :3] = BASE_ROTATION
    sidecar = {
        "scene_xml": str(args.scene_xml), "keyframe": args.keyframe,
        "finger_joint_names": finger_order, "close_finger_ctrl": close_ctrl,
        "base_rotation": R4.tolist(),
    }
    (dest / "morphohand_grasp.json").write_text(json.dumps(sidecar, indent=4))
    print(f"Wrote {dest}/config.json, coll_mesh.obj, vis_mesh.obj, morphohand_grasp.json")


if __name__ == "__main__":
    main()
