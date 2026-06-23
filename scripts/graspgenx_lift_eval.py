"""Physics lift-test for GraspGenX grasps on the real MuJoCo morphohand.

GraspGenX scores grasps with a learned discriminator; this script checks
whether those grasps actually HOLD when executed on our hand in MuJoCo:

  1. build an eval scene = morphohand (baseline morphology, mocap-welded palm)
     + the object as a free body, floor non-colliding,
  2. for each predicted grasp pose, place the palm at  T_grasp @ base_rotation
     with the fingers fully extended (GraspGenX's "open" config),
  3. settle, then close the fingers to the config's "close" targets (gravity
     off so the floating object stays put while contacts form),
  4. turn gravity on and raise the palm 0.2 m,
  5. success = the object came up with the hand (z gain > threshold).

Only the baseline morphology is evaluated: it matches the MJCF "open" keyframe
exactly, and the MJCF morph joints can't express the z-raised thumb / out-of-
range splay/length of the other swept morphologies.

Run in the project uv env (mujoco 3.6):
    uv run python scripts/graspgenx_lift_eval.py \
        --object external/lightning-grasp/assets/object/morphohand/cube_40mm.obj \
        --grasps /tmp/mh_cube.yml --config external/GraspGenX/assets/x_grippers/morphohand/config.json \
        --topk 10
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh.transformations as tra
import yaml

ROOT = Path(__file__).resolve().parents[1]
HAND_XML = ROOT / "assets" / "mjcf" / "hand.xml"

# Baseline morphology = MJCF "open" keyframe morph qpos (matches the URDF the
# GraspGenX baseline gripper was built from).
MORPH_QPOS = {
    "thumb_x": 0.0, "thumb_y": 0.02, "thumb_len": 0.0,
    "index_x": 0.01, "index_y": -0.0123, "index_len": 0.0,
    "middle_x": 0.01, "middle_y": 0.0153, "middle_len": 0.0,
}
CTRL_JOINTS = ["thumb_yaw", "thumb_mcp", "thumb_pip",
               "index_yaw", "index_mcp", "index_pip",
               "middle_yaw", "middle_mcp", "middle_pip"]


def build_model_xml(object_obj: Path, density: float, friction: str) -> str:
    tree = ET.parse(HAND_XML)
    root = tree.getroot()

    palm = next(b for b in root.iter("body") if b.get("name") == "palm")
    palm.set("pos", "0 0 0")
    # palm gets a free joint but is pinned kinematically each step (the "robot"
    # holds a commanded pose; it does not react to contact). High-damping armature
    # via the freejoint isn't needed because we overwrite qpos/qvel every step.
    palm.insert(0, ET.Element("freejoint", {"name": "palm_free"}))

    worldbody = root.find("worldbody")
    # camera that keeps the (moving) object centred
    ET.SubElement(worldbody, "camera", {
        "name": "track", "pos": "0.3 0.3 0.18",
        "mode": "targetbody", "target": "object"})

    # object as a free body, mesh geom, at the world origin (= mesh frame)
    asset = root.find("asset")
    ET.SubElement(asset, "mesh", {"name": "obj", "file": str(object_obj.resolve())})
    obj_body = ET.SubElement(worldbody, "body", {"name": "object", "pos": "0 0 0"})
    ET.SubElement(obj_body, "freejoint", {"name": "obj_free"})
    ET.SubElement(obj_body, "geom", {
        "type": "mesh", "mesh": "obj", "density": str(density),
        "friction": friction, "rgba": "0.18 0.5 0.9 1",
        "condim": "4", "solref": "0.01 1", "solimp": "0.95 0.99 0.001",
    })

    return ET.tostring(root, encoding="unicode")


def set_joint_qpos(model, data, name, value):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = value


def set_free_pose(model, data, joint_name, pos, quat_wxyz):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr + 3] = pos
    data.qpos[adr + 3:adr + 7] = quat_wxyz
    data.qvel[model.jnt_dofadr[jid]:model.jnt_dofadr[jid] + 6] = 0.0


def load_grasps(yml: Path, topk: int):
    data = yaml.safe_load(yml.read_text())
    gs = sorted(data["grasps"].values(), key=lambda g: -g["confidence"])[:topk]
    out = []
    for g in gs:
        T = tra.quaternion_matrix([g["orientation"]["w"], *g["orientation"]["xyz"]])
        T[:3, 3] = g["position"]
        out.append((T, g["confidence"]))
    return out


def run_one(model, data, T_grasp, base_rot, close_targets, lift=0.2, renderer=None, frames=None):
    def snap():
        if renderer is not None:
            renderer.update_scene(data, camera="track")
            frames.append(renderer.render().copy())

    mujoco.mj_resetData(model, data)
    for n, v in MORPH_QPOS.items():
        set_joint_qpos(model, data, n, v)
    for j in CTRL_JOINTS:
        set_joint_qpos(model, data, j, 0.0)  # extended (GraspGenX open)

    M = T_grasp @ base_rot
    palm_pos = M[:3, 3].copy()
    palm_quat = tra.quaternion_from_matrix(M)  # w,x,y,z

    palm_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "palm_free")
    palm_adr = model.jnt_qposadr[palm_jid]
    palm_dof = model.jnt_dofadr[palm_jid]

    def pin_palm(z_extra=0.0, vz=0.0):
        data.qpos[palm_adr:palm_adr + 3] = palm_pos + [0, 0, z_extra]
        data.qpos[palm_adr + 3:palm_adr + 7] = palm_quat
        data.qvel[palm_dof:palm_dof + 6] = 0.0
        data.qvel[palm_dof + 2] = vz

    set_free_pose(model, data, "obj_free", [0, 0, 0], [1, 0, 0, 0])
    data.ctrl[:] = 0.0
    pin_palm()
    mujoco.mj_forward(model, data)

    obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")
    obj_adr = model.jnt_qposadr[obj_jid]
    z0 = float(data.qpos[obj_adr + 2])

    model.opt.gravity[:] = [0, 0, 0]
    snap()
    for _ in range(100):           # settle, fingers open
        pin_palm(); mujoco.mj_step(model, data)
    snap()
    for s in range(250):           # ramp fingers closed
        data.ctrl[:] = ((s + 1) / 250.0) * close_targets
        pin_palm(); mujoco.mj_step(model, data)
        if renderer is not None and s % 50 == 0:
            snap()
    for _ in range(100):           # hold closed
        pin_palm(); mujoco.mj_step(model, data)
    snap()

    model.opt.gravity[:] = [0, 0, -9.81]
    nlift, vz = 400, lift / (400 * model.opt.timestep)
    for s in range(nlift):         # lift at constant velocity
        pin_palm(z_extra=lift * (s + 1) / nlift, vz=vz)
        mujoco.mj_step(model, data)
        if renderer is not None and s % 80 == 0:
            snap()
    for _ in range(150):           # settle at top
        pin_palm(z_extra=lift); mujoco.mj_step(model, data)
    snap()

    z1 = float(data.qpos[obj_adr + 2])
    obj_pos = data.qpos[obj_adr:obj_adr + 3].copy()
    palm_now = palm_pos + [0, 0, lift]
    dist_to_palm = float(np.linalg.norm(obj_pos - palm_now))
    return float(z1 - z0), dist_to_palm, float(palm_now[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", type=Path, required=True)
    ap.add_argument("--grasps", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--density", type=float, default=500.0)
    ap.add_argument("--friction", default="2.0 0.1 0.01")
    ap.add_argument("--close-scale", type=float, default=1.0,
                    help="Multiply the config close targets (>1 closes harder).")
    ap.add_argument("--success-z", type=float, default=0.10)
    ap.add_argument("--render-grasp", type=int, default=None,
                    help="If set, render this grasp index's execution to --render-out (PNG strip).")
    ap.add_argument("--render-out", type=Path, default=None)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    base_rot = np.array(cfg["base_rotation"])
    close_js = cfg["close"]

    xml = build_model_xml(args.object, args.density, args.friction)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # close targets in actuator order
    act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                 for i in range(model.nu)]
    close_targets = np.zeros(model.nu)
    for i, an in enumerate(act_names):
        jn = an[2:]  # a_thumb_yaw -> thumb_yaw
        close_targets[i] = close_js.get(jn, 0.0) * args.close_scale

    grasps = load_grasps(args.grasps, args.topk)

    if args.render_grasp is not None:
        import imageio
        renderer = mujoco.Renderer(model, height=480, width=640)
        frames: list = []
        T, conf = grasps[args.render_grasp]
        dz, dist, _ = run_one(model, data, T, base_rot, close_targets,
                              renderer=renderer, frames=frames)
        strip = np.concatenate(frames, axis=1)
        imageio.imwrite(str(args.render_out), strip)
        print(f"grasp {args.render_grasp} conf={conf:.3f} dz={dz:+.3f} dist={dist:.3f} "
              f"-> {args.render_out} ({len(frames)} frames)")
        return

    print(f"object={args.object.name}  grasps={len(grasps)}  close_scale={args.close_scale}")
    n_ok = 0
    for i, (T, conf) in enumerate(grasps):
        dz, dist, palm_z = run_one(model, data, T, base_rot, close_targets)
        ok = dz > args.success_z
        n_ok += ok
        print(f"  grasp {i:2d} conf={conf:.3f}  obj_dz={dz:+.3f}  "
              f"dist_to_palm={dist:.3f}  {'HOLD' if ok else 'drop'}")
    print(f"=> {n_ok}/{len(grasps)} held (dz>{args.success_z}m)")


if __name__ == "__main__":
    main()
