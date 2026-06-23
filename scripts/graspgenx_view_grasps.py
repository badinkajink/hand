"""Headless visual validation for GraspGenX morphohand grasps.

Loads an object mesh + the gripper collision mesh, places the gripper at the
top-K predicted grasp poses (Isaac-grasp YAML, object frame), and writes a
.glb scene (always) plus a .png render (best-effort, offscreen) so grasp
quality can be eyeballed instead of trusting confidence alone.

Run in the GraspGenX uv env:
    cd external/GraspGenX
    uv run python /abs/path/scripts/graspgenx_view_grasps.py \
        --object assets/sample_data/object_mesh/banana.obj \
        --gripper-dir assets/x_grippers/morphohand \
        --grasps /tmp/morphohand_banana.yml --topk 5 --out /tmp/morpho_banana
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
import trimesh.transformations as tra
import yaml


def load_poses(yml: Path, topk: int) -> list[np.ndarray]:
    data = yaml.safe_load(yml.read_text())
    grasps = list(data["grasps"].values())
    grasps.sort(key=lambda g: -g["confidence"])
    out = []
    for g in grasps[:topk]:
        T = tra.quaternion_matrix([g["orientation"]["w"], *g["orientation"]["xyz"]])
        T[:3, 3] = g["position"]
        out.append((T, g["confidence"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", type=Path, required=True)
    ap.add_argument("--object-scale", type=float, default=1.0)
    ap.add_argument("--gripper-dir", type=Path, required=True)
    ap.add_argument("--grasps", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    obj = trimesh.load(str(args.object), force="mesh")
    obj.apply_scale(args.object_scale)
    obj.visual.face_colors = [160, 160, 170, 255]

    grip = trimesh.load(str(args.gripper_dir / "coll_mesh.obj"), force="mesh")

    scene = trimesh.Scene()
    scene.add_geometry(obj, node_name="object")
    poses = load_poses(args.grasps, args.topk)
    # color best -> green, worst-of-topk -> red
    for i, (T, conf) in enumerate(poses):
        g = grip.copy()
        frac = i / max(1, len(poses) - 1)
        g.visual.face_colors = [int(255 * frac), int(200 * (1 - frac)), 60, 230]
        g.apply_transform(T)
        scene.add_geometry(g, node_name=f"grasp_{i}_{conf:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    glb = args.out.with_suffix(".glb")
    scene.export(str(glb))
    print(f"Wrote {glb}  (object + top-{len(poses)} grasps; green=best)")
    print(f"  best conf {poses[0][1]:.3f}  poses span conf "
          f"{poses[-1][1]:.3f}..{poses[0][1]:.3f}")

    # best-effort offscreen PNG
    try:
        png = args.out.with_suffix(".png")
        png.write_bytes(scene.save_image(resolution=(1280, 960)))
        print(f"Wrote {png}")
    except Exception as e:
        print(f"(PNG render skipped: {type(e).__name__}: {e}) — open the .glb instead")


if __name__ == "__main__":
    main()
