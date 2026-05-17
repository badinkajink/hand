"""Generate a URDF of the morphohand for Lightning Grasp.

Lightning Grasp needs a URDF; our hand lives in MJCF. The morphology (mount
positions and segment lengths) is frozen at the keyframe morphology of the
target scene -- in our MVP, the cube scene's `open` keyframe. Capsules are
expressed as cylinder + 2 spheres (URDF has no capsule primitive but
Lightning supports cylinder/sphere via trimesh.creation primitives).

Outputs:
  external/lightning-grasp/assets/hand/morphohand/morphohand.urdf
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# Baked morphology = cube/prism `open` keyframe values
# (matches results/.../prism.frozen.xml; same morph for cube scene)
BAKED_MOUNT = {
    "thumb":  (-0.03,   -0.008,  0.0),   # base (-0.03, -0.028, 0) + y=0.020
    "index":  ( 0.01,    0.0177, 0.0),   # base (0.0,   0.030,  0) + (x=0.01, y=-0.0123)
    "middle": ( 0.01,   -0.0147, 0.0),   # base (0.0,  -0.030,  0) + (x=0.01, y= 0.0153)
}
BAKED_LEN = 0.0  # all `*_len` morph slides are zero in this keyframe

# Joint limits mirror MJCF ranges
JOINT_LIMITS = {
    "thumb_yaw":  (-1.1, 1.1),
    "thumb_mcp":  (0.0, 3.14),
    "thumb_pip":  (-1.2, 0.5),
    "index_yaw":  (-1.1, 1.1),
    "index_mcp":  (0.0, 2.5),
    "index_pip":  (-1.2, 1.57),
    "middle_yaw": (-1.1, 1.1),
    "middle_mcp": (0.0, 2.5),
    "middle_pip": (-1.2, 1.57),
}

# Capsule segment dimensions from MJCF (fromto length × size = radius)
MCP_LEN, MCP_R = 0.05,  0.010
PRX_LEN, PRX_R = 0.04,  0.0085
PIP_LEN, PIP_R = 0.03,  0.0075
TIP_R          = 0.010
TIP_AUX_R      = 0.005
TIP_AUX_DY     = 0.004

PALM_HALF = (0.06, 0.045, 0.001)


def _inertial(mass: float = 0.01) -> str:
    # Tiny diagonal inertia is fine; Lightning doesn't use it for kinematics.
    return (
        f'    <inertial>\n'
        f'      <mass value="{mass}"/>\n'
        f'      <origin rpy="0 0 0" xyz="0 0 0"/>\n'
        f'      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>\n'
        f'    </inertial>\n'
    )


def _capsule_collision_x(length: float, radius: float) -> str:
    """Capsule from (0,0,0) to (length,0,0) as cylinder + 2 spheres.

    URDF cylinder is along +Z by default; rpy="0 pi/2 0" rotates it to +X.
    """
    half = length / 2.0
    rpy_y90 = f"0 {math.pi/2:.6f} 0"
    return (
        f'    <collision>\n'
        f'      <origin rpy="{rpy_y90}" xyz="{half:.6f} 0 0"/>\n'
        f'      <geometry><cylinder radius="{radius}" length="{length}"/></geometry>\n'
        f'    </collision>\n'
        f'    <collision>\n'
        f'      <origin rpy="0 0 0" xyz="0 0 0"/>\n'
        f'      <geometry><sphere radius="{radius}"/></geometry>\n'
        f'    </collision>\n'
        f'    <collision>\n'
        f'      <origin rpy="0 0 0" xyz="{length:.6f} 0 0"/>\n'
        f'      <geometry><sphere radius="{radius}"/></geometry>\n'
        f'    </collision>\n'
    )


def _tip_collision() -> str:
    return (
        f'    <collision>\n'
        f'      <origin rpy="0 0 0" xyz="0 0 0"/>\n'
        f'      <geometry><sphere radius="{TIP_R}"/></geometry>\n'
        f'    </collision>\n'
        f'    <collision>\n'
        f'      <origin rpy="0 0 0" xyz="0 {TIP_AUX_DY} 0"/>\n'
        f'      <geometry><sphere radius="{TIP_AUX_R}"/></geometry>\n'
        f'    </collision>\n'
        f'    <collision>\n'
        f'      <origin rpy="0 0 0" xyz="0 {-TIP_AUX_DY} 0"/>\n'
        f'      <geometry><sphere radius="{TIP_AUX_R}"/></geometry>\n'
        f'    </collision>\n'
    )


def _revolute(name: str, parent: str, child: str, origin_xyz: tuple[float, float, float], axis: tuple[float, float, float]) -> str:
    lo, hi = JOINT_LIMITS[name]
    ox, oy, oz = origin_xyz
    ax, ay, az = axis
    return (
        f'  <joint name="{name}" type="revolute">\n'
        f'    <parent link="{parent}"/>\n'
        f'    <child link="{child}"/>\n'
        f'    <origin rpy="0 0 0" xyz="{ox:.6f} {oy:.6f} {oz:.6f}"/>\n'
        f'    <axis xyz="{ax} {ay} {az}"/>\n'
        f'    <limit effort="10" velocity="10" lower="{lo}" upper="{hi}"/>\n'
        f'  </joint>\n'
    )


def _fixed(name: str, parent: str, child: str, origin_xyz: tuple[float, float, float]) -> str:
    ox, oy, oz = origin_xyz
    return (
        f'  <joint name="{name}" type="fixed">\n'
        f'    <parent link="{parent}"/>\n'
        f'    <child link="{child}"/>\n'
        f'    <origin rpy="0 0 0" xyz="{ox:.6f} {oy:.6f} {oz:.6f}"/>\n'
        f'  </joint>\n'
    )


def _finger(finger: str) -> str:
    """Emit one finger's links + joints. Active joints: yaw, mcp, pip."""
    mount_xyz = BAKED_MOUNT[finger]
    chunks: list[str] = []

    # Yaw link (revolute), virtual frame
    chunks.append(f'  <link name="{finger}_yaw_link">\n{_inertial()}  </link>\n')
    chunks.append(_revolute(f"{finger}_yaw", "base_link", f"{finger}_yaw_link", mount_xyz, (1, 0, 0)))

    # MCP link (revolute): hosts the first capsule (50mm) AND the frozen-len capsule (40mm at +0.05)
    mcp_geoms = _capsule_collision_x(MCP_LEN, MCP_R)
    # The 'len' segment in MJCF sits at +MCP_LEN along x; with BAKED_LEN=0 it begins exactly at MCP_LEN.
    # Express by translating its capsule origin by +MCP_LEN.
    prx_offset = MCP_LEN + BAKED_LEN  # 0.05
    # We can't put a child capsule at offset using a single collision tag; emit a third capsule manually.
    rpy_y90 = f"0 {math.pi/2:.6f} 0"
    prx_half = PRX_LEN / 2.0
    prx_geoms = (
        f'    <collision>\n'
        f'      <origin rpy="{rpy_y90}" xyz="{prx_offset + prx_half:.6f} 0 0"/>\n'
        f'      <geometry><cylinder radius="{PRX_R}" length="{PRX_LEN}"/></geometry>\n'
        f'    </collision>\n'
        f'    <collision>\n'
        f'      <origin rpy="0 0 0" xyz="{prx_offset:.6f} 0 0"/>\n'
        f'      <geometry><sphere radius="{PRX_R}"/></geometry>\n'
        f'    </collision>\n'
        f'    <collision>\n'
        f'      <origin rpy="0 0 0" xyz="{prx_offset + PRX_LEN:.6f} 0 0"/>\n'
        f'      <geometry><sphere radius="{PRX_R}"/></geometry>\n'
        f'    </collision>\n'
    )
    chunks.append(f'  <link name="{finger}_mcp_link">\n{_inertial()}{mcp_geoms}{prx_geoms}  </link>\n')
    chunks.append(_revolute(f"{finger}_mcp", f"{finger}_yaw_link", f"{finger}_mcp_link", (0, 0, 0), (0, 1, 0)))

    # PIP link (revolute), origin at end of frozen-len segment
    pip_origin = (MCP_LEN + BAKED_LEN + PRX_LEN, 0.0, 0.0)
    chunks.append(f'  <link name="{finger}_pip_link">\n{_inertial()}{_capsule_collision_x(PIP_LEN, PIP_R)}  </link>\n')
    chunks.append(_revolute(f"{finger}_pip", f"{finger}_mcp_link", f"{finger}_pip_link", pip_origin, (0, 1, 0)))

    # Tip link (fixed), origin at end of pip capsule
    tip_origin = (PIP_LEN, 0.0, 0.0)
    chunks.append(f'  <link name="{finger}_tip_link">\n{_inertial()}{_tip_collision()}  </link>\n')
    chunks.append(_fixed(f"{finger}_tip_fixed", f"{finger}_pip_link", f"{finger}_tip_link", tip_origin))

    return "".join(chunks)


def build_urdf() -> str:
    palm_geom = (
        f'    <collision>\n'
        f'      <origin rpy="0 0 0" xyz="0 0 0"/>\n'
        f'      <geometry><box size="{2*PALM_HALF[0]} {2*PALM_HALF[1]} {2*PALM_HALF[2]}"/></geometry>\n'
        f'    </collision>\n'
    )
    base_link = f'  <link name="base_link">\n{_inertial(mass=0.1)}{palm_geom}  </link>\n'

    fingers = "".join(_finger(f) for f in ("thumb", "index", "middle"))

    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<robot name="morphohand">\n'
        + base_link
        + fingers
        + '</robot>\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("external/lightning-grasp/assets/hand/morphohand/morphohand.urdf"),
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_urdf())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
