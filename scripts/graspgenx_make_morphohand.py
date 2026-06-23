"""Non-interactive GraspGenX gripper exporter for the morphohand.

GraspGenX normally onboards a new gripper through an interactive Viser GUI
(scripts/gripper_config_wizard.py). That wizard only produces config.json +
vis_mesh.obj; the point-cloud / tsdf / vae caches fall back to dummy values
at inference and the model (gripper=sweep_volume_v2) conditions on the
sweep-volume boxes + gripper type + bbox in config.json. get_gripper_info()
never reads the URDF at inference.

This script reproduces the wizard's save step head-less for our parametric
3-finger hand:
  1. builds the morphohand URDF via our existing builder (scripts/build_morphohand_urdf.py),
  2. mirrors each <collision> as a <visual> so yourdfpy can load a scene,
  3. picks open/close joint poses + a base_rotation that aligns the hand to
     GraspGenX's canonical frame (+Z = approach, +X = closing),
  4. computes the open/half sweep-volume boxes and the gripper bbox with the
     wizard's own geometry helpers,
  5. writes config.json + gripper.urdf + vis_mesh.obj + coll_mesh.obj into
     <out>/<name>/ (an x_grippers-style directory GraspGenX can resolve).

Run it inside the GraspGenX uv env so yourdfpy/trimesh are importable:
    cd external/GraspGenX
    uv run python /abs/path/scripts/graspgenx_make_morphohand.py \
        --out assets/x_grippers --name morphohand
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy

HAND_REPO_ROOT = Path(__file__).resolve().parents[1]
GRASPGENX_ROOT = HAND_REPO_ROOT / "external" / "GraspGenX"

# Our URDF builder (stdlib-only) + the wizard geometry helpers.
sys.path.insert(0, str(HAND_REPO_ROOT / "scripts"))
sys.path.insert(0, str(GRASPGENX_ROOT / "scripts"))

import build_morphohand_urdf as bmu  # noqa: E402
from gripper_config_wizard import (  # noqa: E402
    compute_gripper_bbox,
    export_merged_mesh,
)


def add_visual_geoms(urdf_text: str) -> str:
    """yourdfpy builds its scene from <visual> geometry; our builder only
    emits <collision>. Mirror each collision element as a visual one."""
    root = ET.fromstring(urdf_text)
    for link in root.findall("link"):
        for coll in list(link.findall("collision")):
            vis = ET.fromstring(ET.tostring(coll))
            vis.tag = "visual"
            link.append(vis)
    return ET.tostring(root, encoding="unicode")


# base_rotation maps the morphohand's native URDF frame into GraspGenX's
# canonical convention. Native: fingers extend along +X, splay along Y, palm
# normal +Z. Canonical: +Z = approach (along finger length), +X = closing
# (the inter-finger spread axis). R sends old +X->+Z, old +Y->+X, old +Z->+Y.
BASE_ROTATION = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def build_open_close(joint_names: list[str]) -> tuple[dict, dict]:
    """Open = fingers extended (all joints ~0). Close = MCP/PIP flexed so the
    fingertips converge toward the palm centerline."""
    open_js = {j: 0.0 for j in joint_names}
    close_js = {}
    for j in joint_names:
        if j.endswith("_mcp"):
            close_js[j] = 1.3
        elif j.endswith("_pip"):
            close_js[j] = 0.9
        else:  # yaw
            close_js[j] = 0.0
    return open_js, close_js


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=GRASPGENX_ROOT / "assets" / "x_grippers")
    ap.add_argument("--name", default="morphohand")
    ap.add_argument(
        "--scene-xml",
        type=Path,
        default=None,
        help="Optional MJCF scene to derive the baked morphology from; "
        "defaults to the canonical cube/prism open morphology.",
    )
    ap.add_argument("--keyframe", default="open_flat")
    args = ap.parse_args()

    # 1. morphology -> URDF text
    if args.scene_xml is not None:
        bmu.BAKED_MOUNT, bmu.BAKED_LEN, bmu.MCP_LEN = bmu._derive_morphology_from_scene(
            args.scene_xml, args.keyframe
        )
        print(f"Derived morphology from {args.scene_xml}@{args.keyframe}")
    urdf_text = add_visual_geoms(bmu.build_urdf())

    dest = args.out / args.name
    dest.mkdir(parents=True, exist_ok=True)
    urdf_path = dest / "gripper.urdf"
    urdf_path.write_text(urdf_text)
    print(f"Wrote URDF: {urdf_path}")

    # 2. load + actuated joints
    robot = yourdfpy.URDF.load(
        str(urdf_path),
        build_scene_graph=True,
        load_meshes=True,
        force_mesh=False,
    )
    joint_names = [j.name for j in robot.robot.joints if j.type != "fixed"]
    link_names = [l.name for l in robot.robot.links]
    print(f"Actuated joints ({len(joint_names)}): {joint_names}")

    open_js, close_js = build_open_close(joint_names)
    half_js = {k: open_js[k] + 0.5 * (close_js[k] - open_js[k]) for k in open_js}

    # 3. sweep volumes from the actual fingertip arc, in the canonical frame.
    #
    # The wizard's auto-estimator assumes an opposed gripper and a single
    # closing axis; our 3 fingers curl together (non-opposed), so its boxes
    # came out spanning the whole finger length. Instead we bound the region
    # the fingertips actually enclose as they sweep open->close, centred on
    # the grasp "pocket" (where the flexed fingers converge), so the model
    # places objects where the hand can wrap them rather than at the
    # fully-extended fingertips.
    R = BASE_ROTATION[:3, :3]

    def tip_positions(js: dict) -> np.ndarray:
        robot.update_cfg(js)
        pts = []
        for f in ("thumb", "index", "middle"):
            T = robot.get_transform(frame_to=f + "_tip_link", frame_from="base_link")
            pts.append(R @ T[:3, 3])
        return np.array(pts)

    tips_open = tip_positions(open_js)
    tips_half = tip_positions(half_js)
    tips_close = tip_positions(close_js)

    MARGIN = 0.012  # ~finger radius, so the box wraps finger thickness
    FLOOR = np.array([0.04, 0.04, 0.04])  # min graspable box

    def box(point_sets: list[np.ndarray]) -> tuple[list, list]:
        pts = np.vstack(point_sets)
        lo, hi = pts.min(0), pts.max(0)
        extents = np.maximum((hi - lo) + 2 * MARGIN, FLOOR)
        offset = (lo + hi) / 2.0
        return extents.tolist(), offset.tolist()

    # pocket = where the half/closed fingertips converge (object rests here)
    pocket = 0.5 * (tips_half.mean(0) + tips_close.mean(0))
    # open box: from the open fingertips down into the pocket (max aperture)
    sv_extents, sv_offset = box([tips_open, pocket[None, :]])
    # half box: the tighter region between half and closed fingertips
    sv2_extents, sv2_offset = box([tips_half, tips_close])

    bbox_min, bbox_max = compute_gripper_bbox(robot, open_js, base_T=BASE_ROTATION)
    print(f"tips_open  centroid={np.round(tips_open.mean(0),4)}")
    print(f"tips_half  centroid={np.round(tips_half.mean(0),4)}")
    print(f"tips_close centroid={np.round(tips_close.mean(0),4)}")
    print(f"pocket={np.round(pocket,4)}")
    print(f"open  sweep extents={np.round(sv_extents,4)} offset={np.round(sv_offset,4)}")
    print(f"half  sweep extents={np.round(sv2_extents,4)} offset={np.round(sv2_offset,4)}")
    print(f"bbox  min={np.round(bbox_min,4)} max={np.round(bbox_max,4)}")

    config = {
        "open": open_js,
        "close": close_js,
        "fingertip": list(sv_offset),
        "sweep_volume": {
            "extents": sv_extents,
            "offset": sv_offset,
            "extents2": sv2_extents,
            "offset2": sv2_offset,
        },
        "links": link_names,
        "standoff": [0.0, sv_extents[2] / 2],
        "symmetric": False,
        "type": "revolute_3f",
        "bbox": [bbox_min, bbox_max],
        "base_rotation": BASE_ROTATION.tolist(),
    }
    (dest / "config.json").write_text(json.dumps(config, indent=4))
    print(f"Wrote config: {dest/'config.json'}")

    # 4. merged meshes (canonical frame) for collision filtering + viz
    export_merged_mesh(robot, open_js, str(dest / "vis_mesh.obj"), base_T=BASE_ROTATION)
    export_merged_mesh(robot, open_js, str(dest / "coll_mesh.obj"), base_T=BASE_ROTATION)
    print(f"Wrote vis_mesh.obj + coll_mesh.obj")
    print("\nDone. Run inference with:")
    print(
        f"  uv run python scripts/demo_object_mesh.py --gripper_name {args.name} "
        f"--mesh_file assets/sample_data/object_mesh/banana.obj --mesh_scale 1.0 "
        f"--grasp_threshold -1.0 --return_topk --topk_num_grasps 50 "
        f"--no-visualization --output_file /tmp/morphohand_grasps.yml"
    )


if __name__ == "__main__":
    main()
