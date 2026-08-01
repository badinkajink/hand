#!/usr/bin/env python3
"""Export a MorphoHand MJCF to a UHAS-ready URDF folder, then print the process_urdf command.

    uv run --extra rl --extra gpu python scripts/uhas_export_hand.py \
        --mjcf assets/mjcf/baseline/hand.xml --out results/uhas/hands/baseline

The output folder is laid out the way UHAS_sim/grippers/<hand> is, so
``process_urdf.py --robot_path <out>/<name>.urdf`` works directly on it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morphohand.uhas import export_hand_to_urdf  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mjcf", required=True, help="MorphoHand MJCF (morphology already baked)")
    p.add_argument("--out", required=True, help="output gripper folder")
    p.add_argument("--name", default=None, help="robot name (default: output folder name)")
    p.add_argument("--palm-body", default="palm")
    p.add_argument("--tip-suffix", default="_tip")
    p.add_argument("--open-mcp", type=float, default=0.0,
                   help="MCP angle for the open-hand pose; non-zero breaks the yaw "
                        "singularity that a fully extended finger sits at")
    p.add_argument("--open-pip", type=float, default=0.0)
    p.add_argument("--open-yaw", type=float, default=0.0)
    args = p.parse_args()

    out = Path(args.out)
    name = args.name or out.name
    exp = export_hand_to_urdf(args.mjcf, out, robot_name=name, palm_body=args.palm_body,
                             tip_suffix=args.tip_suffix, open_mcp=args.open_mcp,
                             open_pip=args.open_pip, open_yaw=args.open_yaw)

    print(f"URDF     : {exp.urdf_path}")
    print(f"config   : {exp.config_path}")
    print(f"meshes   : {exp.mesh_dir} ({len(list(exp.mesh_dir.glob('*.stl')))} stl)")
    print(f"base_link: {exp.base_link}")
    print(f"fingers  : {exp.finger_names}")
    print(f"hinges   : {exp.hinge_joints}")
    print("tip distances from palm centre:")
    for f, d in exp.tip_distances.items():
        print(f"    {f:8s} {d:.4f} m")
    print(f"=> sphere radius estimate 2l/pi = {exp.sphere_radius_estimate:.4f} m "
          f"(LEAP reference 0.0912)")
    print()
    print("next:")
    print(f"  cd docs/uhas/UHAS_sim/process_urdf && MPLBACKEND=Agg \\")
    print(f"    ../../../../.venv-uhas/bin/python process_urdf.py \\")
    print(f"      --robot_path {exp.urdf_path.resolve()} \\")
    print(f"      --base_link {exp.base_link} --correct_axes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
