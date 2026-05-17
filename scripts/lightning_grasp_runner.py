"""Run Lightning Grasp's pipeline once and dump the resulting grasps to JSON.

This script runs inside the Lightning Grasp Python 3.10 environment under
external/lightning-grasp/.venv. It's the bridge between Lightning's GPU
pipeline and our MuJoCo eval, which lives in the main Python 3.12 env.

Each output entry is one grasp:
    {
      "q":              [9 floats],      # active joint angles (Lightning's hand pose)
      "object_pose":    [[16 floats]],   # 4x4 row-major, in palm frame
      "active_joints":  [name, ...]      # joint name order matching q
    }

The 9 active joints are in our standard MJCF order so q can be used directly
as `finger_ctrl` for `Phase1GraspEvaluator.evaluate(...)`.

Usage (from external/lightning-grasp/):

    python ../../scripts/lightning_grasp_runner.py \
        --robot morphohand \
        --object_mesh_path ./assets/object/morphohand/cube_40mm.obj \
        --output_json ../../results/lightning_grasp/cube_grasps.json \
        --batch_size_outer 256 --batch_size_inner 128 --n_contact 3
"""
# pyright: reportMissingImports=false

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure the lygra package (which lives under external/lightning-grasp/) is
# importable regardless of how this script is invoked. We expect to be run
# from within external/lightning-grasp/ with its .venv activated.
_HERE = Path(__file__).resolve()
_LYGRA_ROOT = _HERE.parents[1] / "external" / "lightning-grasp"
if str(_LYGRA_ROOT) not in sys.path:
    sys.path.insert(0, str(_LYGRA_ROOT))
os.chdir(_LYGRA_ROOT)  # urdf default paths are resolved relative to this dir

import numpy as np
import torch

from lygra.robot import build_robot
from lygra.contact_set import get_link_dependency_matrix
from lygra.kinematics import build_kinematics_tree
from lygra.mesh import (
    get_urdf_mesh,
    get_urdf_mesh_decomposed,
    get_urdf_mesh_for_projection,
)
from lygra.mesh_analyzer import get_support_point_mask
from lygra.utils.geom_utils import MeshObject
from lygra.memory import IKGPUBufferPool

from lygra.pipeline.module.object_placement import (
    sample_object_pose,
    get_object_pose_sampling_args,
)
from lygra.pipeline.module.contact_query import (
    batch_object_all_contact_fields_interaction,
)
from lygra.pipeline.module.contact_collection import (
    sample_pose_and_contact_from_interaction,
)
from lygra.pipeline.module.contact_optimization import search_contact_point
from lygra.pipeline.module.kinematics import batch_ik, batch_contact_adjustment
from lygra.pipeline.module.postprocess import batch_assign_free_finger_and_filter


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="morphohand")
    ap.add_argument("--object_mesh_path", required=True, type=str)
    ap.add_argument("--output_json", required=True, type=Path)
    ap.add_argument("--n_contact", type=int, default=3)
    ap.add_argument("--n_sample_point", type=int, default=2048)
    ap.add_argument("--batch_size_outer", type=int, default=256)
    ap.add_argument("--batch_size_inner", type=int, default=128)
    ap.add_argument("--ik_finetune_iter", type=int, default=5)
    ap.add_argument("--zo_lr_sigma", type=float, default=5.0)
    ap.add_argument("--object_pose_sampling_strategy", default="canonical")
    ap.add_argument("--max_grasps", type=int, default=0, help="0 = no cap")
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    robot = build_robot(args.robot)
    tree = build_kinematics_tree(robot.urdf_path, active_joint_names=robot.get_active_joints())
    mesh_data = get_urdf_mesh(robot.urdf_path, tree=tree, mesh_scale=robot.get_mesh_scale())
    mesh_data_for_ik = get_urdf_mesh_for_projection(
        robot.urdf_path, tree=tree,
        config=robot.get_contact_field_config(),
        mesh_scale=robot.get_mesh_scale(),
    )
    decomposed_static_mesh = get_urdf_mesh_decomposed(
        robot.urdf_path, tree=tree,
        override_link_names=robot.get_static_links(),
        mesh_scale=robot.get_mesh_scale(),
    )
    decomposed_mesh = get_urdf_mesh_decomposed(
        robot.urdf_path, tree=tree, mesh_scale=robot.get_mesh_scale()
    )
    self_pairs = tree.get_self_collision_check_link_pairs(
        link_body_id=decomposed_mesh["link_body_id"],
        whitelist_link=[],
        whitelist_pairs=robot.get_white_list_pairs(),
    )
    self_pairs = torch.from_numpy(self_pairs).cuda().int()

    contact_field = robot.get_contact_field()
    dep_sets = tree.get_dependency_sets([robot.get_base_link()])
    parent_links = contact_field.get_all_parent_link_names()
    parent_ids = torch.tensor([tree.get_link_id(l) for l in parent_links]).cuda()
    dep_matrix = get_link_dependency_matrix(contact_field, dep_sets).cuda()
    accel = contact_field.generate_acceleration_structure(method="lbvhs2")

    obj = MeshObject(args.object_mesh_path)
    zo_lr = ((obj.get_area() / args.n_sample_point) ** 0.5) * args.zo_lr_sigma
    pts, nrms = obj.sample_point_and_normal(count=args.n_sample_point)
    pts_all = torch.from_numpy(pts).cuda().float()
    nrms_all = torch.from_numpy(nrms).cuda().float()
    sup = get_support_point_mask(pts_all, nrms_all, [0.01])[0]
    points = pts_all[torch.where(sup)]
    normals = nrms_all[torch.where(sup)]

    pool = IKGPUBufferPool(
        n_dof=tree.n_dof(), n_link=tree.n_link(),
        n_actuated_dof=tree.n_actuated_dof(),
        max_batch=min([args.batch_size_outer * args.batch_size_inner, 65536]),
        retry=10,
    )

    print(f"[lightning_runner] Inference starting "
          f"(outer={args.batch_size_outer}, inner={args.batch_size_inner})",
          file=sys.stderr)

    with torch.no_grad():
        object_poses, condition = sample_object_pose(
            n=args.batch_size_outer, points=points, normals=normals,
            contact_field=contact_field, tree=tree,
            mesh_data=decomposed_static_mesh,
            sampling_args=get_object_pose_sampling_args(args.object_pose_sampling_strategy, robot),
        )
        interaction = batch_object_all_contact_fields_interaction(
            object_pos=points, object_normal=normals,
            object_pose=object_poses, accel_structure=accel,
        )
        link_interaction = contact_field.reduce_link_interaction((interaction >= 0).int())

        (cdp, cdn, cdi, object_poses, contact_link_ids, condition, valid_outer_idx) = (
            sample_pose_and_contact_from_interaction(
                n_contact=args.n_contact,
                interaction_matrix=link_interaction, dependency_matrix=dep_matrix,
                object_points=points, object_normals=normals,
                object_poses=object_poses, condition=condition,
            )
        )
        (tcp, tcn, tci, object_poses, target_link_ids, target_outer_ids) = search_contact_point(
            contact_domain_pos=cdp, contact_domain_normal=cdn, contact_domain_point_idx=cdi,
            object_poses=object_poses, contact_ids=contact_link_ids,
            batch_size=args.batch_size_inner, return_hand_frame=True,
            condition=condition, zo_lr=zo_lr,
        )
        contact_ids, local_ids = contact_field.sample_contact_ids(
            interaction_matrix=(interaction >= 0).int()[valid_outer_idx],
            interaction_matrix_hand_point_idx=interaction[valid_outer_idx],
            target_batch_outer_ids=target_outer_ids,
            target_contact_link_ids=target_link_ids,
            target_contact_point_idx=tci,
        )
        cpos_l, cnrm_l = contact_field.sample_contact_geometry(contact_ids, local_ids)

        result = batch_ik(
            tree=tree, contact_ids=contact_ids, contact_parent_ids=parent_ids,
            contact_pos_in_linkf=cpos_l.float(), contact_normal_in_linkf=cnrm_l.float(),
            target_contact_pos=tcp.float(), target_contact_normal=tcn.float(),
            object_pose=object_poses.float(), gpu_memory_pool=pool,
        )
        result = batch_contact_adjustment(
            tree=tree, mesh=mesh_data_for_ik,
            q_init=result["q"], q_mask=result["q_mask"],
            contact_ids=contact_ids, contact_link_ids=result["contact_link_id"],
            contact_pos_in_linkf=result["contact_pos"],
            contact_normal_in_linkf=result["contact_normal"],
            target_contact_pos=result["target_pos"],
            target_contact_normal=result["target_normal"],
            object_pose=result["object_pose"],
            n_iter=args.ik_finetune_iter, gpu_memory_pool=pool, ret_mesh_buffer=True,
        )
        result = batch_assign_free_finger_and_filter(
            tree=tree, result=result, object_point=pts_all,
            self_collision_link_pairs=self_pairs, decomposed_mesh_data=decomposed_mesh,
        )

    q = result["q"].detach().cpu().numpy()                    # [N, n_actuated]
    obj_pose = result["object_pose"].detach().cpu().numpy()   # [N, 4, 4]
    n = q.shape[0]
    t1 = time.time()
    print(f"[lightning_runner] Got {n} grasps in {t1 - t0:.2f}s", file=sys.stderr)

    active = robot.get_active_joints()
    grasps = []
    for i in range(n):
        if args.max_grasps and i >= args.max_grasps:
            break
        grasps.append({
            "q": [float(x) for x in q[i].tolist()],
            "object_pose": obj_pose[i].tolist(),
            "active_joints": active,
        })

    out = {
        "robot": args.robot,
        "object_mesh_path": str(args.object_mesh_path),
        "n_contact": args.n_contact,
        "n_grasps": len(grasps),
        "wall_time_s": float(t1 - t0),
        "grasps": grasps,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out))
    print(f"[lightning_runner] Wrote {len(grasps)} grasps to {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
