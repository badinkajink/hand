"""Export a MorphoHand MJCF to the URDF flavour UHAS's ``process_urdf.py`` expects.

UHAS builds its canonical sphere from a URDF, so a morphology has to leave MJCF to get a
UHAS representation. The conversion is not generic URDF export -- ``process_urdf.py``
reads three conventions out of the file, and all three are silent if you get them wrong:

1. **A fixed joint literally named ``palm_normal``.** Its +Z must point the way the
   fingers close; the sphere centre is placed along it. Ours is a pi-rotation about X of
   the palm frame, because our MCP joints (axis +Y) curl the fingers toward palm -Z.
2. **One fixed joint per finger named ``<finger>_ft``**, +Z along the fingertip normal.
   These terminate the kinematic chains that UHAS discovers.
3. **Mesh geometry only.** ``load_link_meshes`` handles ``urdf_parser_py.urdf.Mesh`` and
   silently ignores primitive tags, so our capsules/spheres are baked to STL.

Two further choices worth stating, because they are easy to get backwards:

* The URDF is written with every hinge at **zero**, since a URDF joint origin is by
  definition the parent->child transform at joint value zero. The open-hand pose lives in
  ``config.json`` (``opened_dofs``) instead, which is what UHAS actually reads.
* The morphology slide joints are **baked** into link origins at their design values.
  Leaving them articulated would hand UHAS a hand whose geometry drifts during rollout --
  the frozen-scene rule that governs every other evaluation in this repo.

At the zero pose each finger's local -Z coincides with the palm normal, so the fingertip
frames satisfy UHAS's "fingertip normals align with the palm normal in the open
configuration" requirement exactly rather than approximately.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:  # mujoco is the project's sim dependency; trimesh comes from the UHAS toolchain.
    import mujoco
except ImportError as exc:  # pragma: no cover - surfaced to the CLI
    raise ImportError("mjcf_to_urdf needs `mujoco` (project .venv)") from exc

import trimesh


# MuJoCo geom type ids we know how to bake into meshes.
_GEOM_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_GEOM_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
_GEOM_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
_GEOM_CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)

_JNT_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_JNT_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)


def _rpy_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """URDF fixed-axis roll-pitch-yaw (equivalently Z-Y-X Euler) from a rotation matrix."""
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(sy) > 1.0 - 1e-9:  # gimbal lock: fold roll into yaw
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return float(roll), float(pitch), float(yaw)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """MuJoCo stores quaternions wxyz."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=np.float64))
    return R.reshape(3, 3)


@dataclass
class _Link:
    """A URDF link: a frame plus the geometry that rides on it."""

    name: str
    pos: np.ndarray  # world position of the link frame
    rot: np.ndarray  # world rotation of the link frame
    meshes: list[tuple[str, trimesh.Trimesh]] = field(default_factory=list)
    mass: float = 0.0


@dataclass
class _Joint:
    name: str
    jtype: str  # "revolute" | "fixed"
    parent: str
    child: str
    xyz: np.ndarray
    rpy: tuple[float, float, float]
    axis: np.ndarray | None = None
    lower: float = 0.0
    upper: float = 0.0
    effort: float = 10.0
    velocity: float = 5.0


@dataclass
class HandUrdfExport:
    """Where the export landed, plus the facts a caller needs to drive process_urdf."""

    urdf_path: Path
    config_path: Path
    mesh_dir: Path
    base_link: str
    finger_names: list[str]
    hinge_joints: list[str]
    opened_dofs: dict[str, float]
    sphere_radius_estimate: float
    tip_distances: dict[str, float]


def _collect_geoms(model, data, body_id: int, frame_pos, frame_rot, link: _Link,
                   mesh_dir: Path, prefix: str) -> None:
    """Bake body `body_id`'s geoms into meshes expressed in `link`'s frame."""
    bpos = data.xpos[body_id]
    brot = data.xmat[body_id].reshape(3, 3)

    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id:
            continue
        gtype = int(model.geom_type[g])
        size = model.geom_size[g]

        if gtype == _GEOM_CAPSULE:
            mesh = trimesh.creation.capsule(height=2.0 * float(size[1]),
                                            radius=float(size[0]), count=[16, 16])
        elif gtype == _GEOM_SPHERE:
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
        elif gtype == _GEOM_BOX:
            mesh = trimesh.creation.box(extents=2.0 * np.asarray(size[:3], dtype=float))
        elif gtype == _GEOM_CYLINDER:
            mesh = trimesh.creation.cylinder(radius=float(size[0]),
                                             height=2.0 * float(size[1]), sections=24)
        else:
            continue  # planes and everything exotic are not hand surface

        # geom -> world -> link frame
        g_world_pos = bpos + brot @ model.geom_pos[g]
        g_world_rot = brot @ _quat_to_matrix(model.geom_quat[g])
        rel_pos = frame_rot.T @ (g_world_pos - frame_pos)
        rel_rot = frame_rot.T @ g_world_rot

        T = np.eye(4)
        T[:3, :3] = rel_rot
        T[:3, 3] = rel_pos
        mesh.apply_transform(T)

        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        fname = f"{prefix}_{gname}_{g}.stl"
        mesh.export(mesh_dir / fname)
        link.meshes.append((fname, mesh))

    link.mass += float(model.body_mass[body_id])


def _body_children(model, body_id: int) -> list[int]:
    return [b for b in range(model.nbody)
            if model.body_parentid[b] == body_id and b != body_id]


def _body_joints(model, body_id: int) -> list[int]:
    start = model.body_jntadr[body_id]
    num = model.body_jntnum[body_id]
    return list(range(start, start + num)) if start >= 0 else []


def export_hand_to_urdf(
    mjcf_path: str | Path,
    out_dir: str | Path,
    robot_name: str = "morphohand",
    palm_body: str = "palm",
    tip_suffix: str = "_tip",
    open_mcp: float = 0.0,
    open_pip: float = 0.0,
    open_yaw: float = 0.0,
    morph_qpos: dict[str, float] | None = None,
) -> HandUrdfExport:
    """Convert a MorphoHand MJCF into a UHAS-ready URDF directory.

    `open_mcp` / `open_pip` / `open_yaw` set the ``opened_dofs`` pose recorded in
    config.json. Zero gives the fully extended hand, which is what UHAS's sphere
    construction assumes; a small non-zero `open_mcp` is the escape hatch if the
    fully-extended pose puts a lateral joint at a kinematic singularity (our yaw axis runs
    along the finger, so at full extension it does not move the fingertip at all).

    `morph_qpos` sets the morphology slide joints before they are baked; omit it to use
    whatever the MJCF already encodes (generated morphology files bake them already).
    """
    mjcf_path = Path(mjcf_path)
    out_dir = Path(out_dir)
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Hinges at zero (URDF origins are defined there); slides at their design values.
    for j in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = model.jnt_qposadr[j]
        if int(model.jnt_type[j]) == _JNT_HINGE:
            data.qpos[adr] = 0.0
        elif int(model.jnt_type[j]) == _JNT_SLIDE and morph_qpos and jname in morph_qpos:
            data.qpos[adr] = float(morph_qpos[jname])
    mujoco.mj_forward(model, data)

    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, palm_body)
    if palm_id < 0:
        raise ValueError(f"no body named {palm_body!r} in {mjcf_path}")

    links: dict[str, _Link] = {}
    joints: list[_Joint] = []
    hinge_joints: list[str] = []
    tip_frames: list[tuple[str, np.ndarray, np.ndarray]] = []  # finger, world pos, link name
    finger_roots: list[np.ndarray] = []

    base = _Link(name=palm_body, pos=data.xpos[palm_id].copy(),
                 rot=data.xmat[palm_id].reshape(3, 3).copy())
    links[palm_body] = base
    _collect_geoms(model, data, palm_id, base.pos, base.rot, base, mesh_dir, palm_body)

    def walk(body_id: int, parent_link: _Link) -> None:
        cur = parent_link
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body{body_id}"

        hinges = [j for j in _body_joints(model, body_id)
                  if int(model.jnt_type[j]) == _JNT_HINGE]
        if len(hinges) > 1:
            raise ValueError(f"body {bname!r} carries {len(hinges)} hinges; "
                             "UHAS chains need one joint per link")

        if hinges:
            j = hinges[0]
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            brot = data.xmat[body_id].reshape(3, 3)
            # The link frame sits at the joint anchor, aligned with the body (hinge is at 0).
            frame_pos = data.xpos[body_id] + brot @ model.jnt_pos[j]
            frame_rot = brot.copy()

            link = _Link(name=f"{jname}_link", pos=frame_pos, rot=frame_rot)
            links[link.name] = link

            rel_pos = cur.rot.T @ (frame_pos - cur.pos)
            rel_rot = cur.rot.T @ frame_rot
            lo, hi = (float(model.jnt_range[j][0]), float(model.jnt_range[j][1])) \
                if model.jnt_limited[j] else (-np.pi, np.pi)
            joints.append(_Joint(name=jname, jtype="revolute", parent=cur.name,
                                 child=link.name, xyz=rel_pos, rpy=_rpy_from_matrix(rel_rot),
                                 axis=np.asarray(model.jnt_axis[j], dtype=float).copy(),
                                 lower=lo, upper=hi))
            hinge_joints.append(jname)
            if model.body_parentid[body_id] == palm_id or cur.name == palm_body:
                finger_roots.append(frame_pos.copy())
            cur = link

        _collect_geoms(model, data, body_id, cur.pos, cur.rot, cur, mesh_dir, bname)

        if bname.endswith(tip_suffix):
            finger = bname[: -len(tip_suffix)]
            tip_frames.append((finger, data.xpos[body_id].copy(), cur.name))

        for child in _body_children(model, body_id):
            walk(child, cur)

    for child in _body_children(model, palm_id):
        walk(child, base)

    if not tip_frames:
        raise ValueError(f"no bodies ending in {tip_suffix!r}; cannot build fingertip frames")

    # --- palm_normal: +Z along the closing direction (palm -Z) --------------------------
    flip_x = np.diag([1.0, -1.0, -1.0])  # pi about X
    if finger_roots:
        roots_local = np.array([base.rot.T @ (p - base.pos) for p in finger_roots])
        palm_centre_local = roots_local.mean(axis=0)
        palm_centre_local[2] = 0.0
    else:
        palm_centre_local = np.zeros(3)

    joints.append(_Joint(name="palm_normal", jtype="fixed", parent=palm_body,
                         child="palm_dummy", xyz=palm_centre_local,
                         rpy=_rpy_from_matrix(flip_x)))
    links["palm_dummy"] = _Link(name="palm_dummy", pos=base.pos, rot=base.rot @ flip_x)

    # --- fingertip frames: +Z along the fingertip normal (also palm -Z at q=0) ----------
    finger_names: list[str] = []
    tip_distances: dict[str, float] = {}
    for finger, tip_world, parent_name in tip_frames:
        parent = links[parent_name]
        rel_pos = parent.rot.T @ (tip_world - parent.pos)
        joints.append(_Joint(name=f"{finger}_ft", jtype="fixed", parent=parent_name,
                             child=f"{finger}_ft_link", xyz=rel_pos,
                             rpy=_rpy_from_matrix(flip_x)))
        links[f"{finger}_ft_link"] = _Link(name=f"{finger}_ft_link",
                                           pos=tip_world, rot=parent.rot @ flip_x)
        finger_names.append(finger)
        tip_local = base.rot.T @ (tip_world - base.pos)
        tip_distances[finger] = float(np.linalg.norm(tip_local - palm_centre_local))

    l_mean = float(np.mean(list(tip_distances.values())))
    sphere_radius = 2.0 * l_mean / np.pi

    # --- emit URDF ----------------------------------------------------------------------
    robot = ET.Element("robot", name=robot_name)
    for name, link in links.items():
        el = ET.SubElement(robot, "link", name=name)
        total = max(link.mass, 1e-6)
        combined = trimesh.util.concatenate([m for _, m in link.meshes]) if link.meshes else None
        inertial = ET.SubElement(el, "inertial")
        if combined is not None and combined.volume > 1e-12:
            combined.density = total / combined.volume
            com = combined.center_mass
            it = combined.moment_inertia
        else:
            com = np.zeros(3)
            it = np.eye(3) * 1e-6
        ET.SubElement(inertial, "origin", xyz=" ".join(f"{v:.8f}" for v in com), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{total:.8f}")
        ET.SubElement(inertial, "inertia",
                      ixx=f"{it[0,0]:.10f}", ixy=f"{it[0,1]:.10f}", ixz=f"{it[0,2]:.10f}",
                      iyy=f"{it[1,1]:.10f}", iyz=f"{it[1,2]:.10f}", izz=f"{it[2,2]:.10f}")
        for fname, _ in link.meshes:
            for tag in ("visual", "collision"):
                node = ET.SubElement(el, tag)
                ET.SubElement(node, "origin", xyz="0 0 0", rpy="0 0 0")
                geom = ET.SubElement(node, "geometry")
                ET.SubElement(geom, "mesh", filename=f"meshes/{fname}")

    for j in joints:
        el = ET.SubElement(robot, "joint", name=j.name, type=j.jtype)
        ET.SubElement(el, "origin",
                      xyz=" ".join(f"{v:.8f}" for v in j.xyz),
                      rpy=" ".join(f"{v:.8f}" for v in j.rpy))
        ET.SubElement(el, "parent", link=j.parent)
        ET.SubElement(el, "child", link=j.child)
        if j.jtype == "revolute":
            ET.SubElement(el, "axis", xyz=" ".join(f"{v:.8f}" for v in j.axis))
            ET.SubElement(el, "limit", lower=f"{j.lower:.6f}", upper=f"{j.upper:.6f}",
                          effort=f"{j.effort}", velocity=f"{j.velocity}")

    ET.indent(robot, space="  ")
    urdf_path = out_dir / f"{robot_name}.urdf"
    urdf_path.write_bytes(ET.tostring(robot, encoding="utf-8", xml_declaration=True))

    # --- config.json: the open-hand pose UHAS measures the sphere from ------------------
    opened: dict[str, float] = {}
    for jname in hinge_joints:
        if jname.endswith("_yaw"):
            opened[jname] = float(open_yaw)
        elif jname.endswith("_mcp"):
            opened[jname] = float(open_mcp)
        elif jname.endswith("_pip"):
            opened[jname] = float(open_pip)
        else:
            opened[jname] = 0.0
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps({"opened_dofs": opened}, indent=4))

    return HandUrdfExport(urdf_path=urdf_path, config_path=config_path, mesh_dir=mesh_dir,
                          base_link=palm_body, finger_names=finger_names,
                          hinge_joints=hinge_joints, opened_dofs=opened,
                          sphere_radius_estimate=sphere_radius, tip_distances=tip_distances)
