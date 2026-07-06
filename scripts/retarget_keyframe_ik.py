"""Retarget the 'about-to-contact' open keyframe across morphologies via fingertip IK.

The bug this fixes (2026-07-01): the `open_short_manual` keyframe is defined in JOINT
space (fixed finger yaw/mcp/pip angles), hand-tuned so the BASELINE hand's fingertips sit
just off the screwdriver. `generate_morphology_xml.py` bakes a new morphology's finger
attachment (x/y) + link length into the geometry but keeps those SAME joint angles — so on
a repositioned/lengthened finger the fingertip lands at a different WORLD position, no
longer about-to-contact. CEM then seeds from a bad open pose and (e.g. on m05) never seats
the thumb → a spurious "2-finger design" conclusion.

Fix (per the user): transfer the keyframe in WORLD space, not joint space. Read the 3
fingertip world XYZ from the known-good baseline keyframe, then damped-least-squares IK each
finger of the TARGET morphology (3 joints: yaw/mcp/pip → 3D tip target) to the same world
positions, keeping the palm/object pose identical. Optionally close a little (flex mcp/pip)
so the tips seat into contact. Writes an `open_ik` keyframe into the target scene, usable as
the CEM seed / LerpFinger open pose.

Run:
  uv run --extra rl python scripts/retarget_keyframe_ik.py \
    --base-scene assets/mjcf/scene_screwdriver_medium_flat_short_proximal.xml \
    --keyframe open_short_manual --target-scene <generated_morph_scene.xml> \
    [--close-rad 0.15] [--validate]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import mujoco

FINGERS = {"thumb": ("thumb_yaw", "thumb_mcp", "thumb_pip"),
           "index": ("index_yaw", "index_mcp", "index_pip"),
           "middle": ("middle_yaw", "middle_mcp", "middle_pip")}
TIPS = {"thumb": "thumb_tip", "index": "index_tip", "middle": "middle_tip"}
PALM_JOINTS = ["palm_px", "palm_py", "palm_pz", "palm_rx", "palm_ry", "palm_rz"]


def tip_targets(base_scene: str, keyframe: str):
    m = mujoco.MjModel.from_xml_path(base_scene)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(keyframe).id)
    mujoco.mj_forward(m, d)
    tips = {f: d.body(TIPS[f]).xpos.copy() for f in FINGERS}
    palm = {j: float(d.qpos[m.jnt_qposadr[m.joint(j).id]]) for j in PALM_JOINTS
            if _has_joint(m, j)}
    obj_qpos = d.qpos[:7].copy()  # object free joint
    return tips, palm, obj_qpos


def _has_joint(m, name):
    try:
        m.joint(name); return True
    except Exception:
        return False


def ik_finger(m, d, finger, target, iters=300, lam=1e-3, step=0.5):
    """Damped-least-squares IK: drive `finger`'s tip to `target` world xyz."""
    jids = [m.joint(j).id for j in FINGERS[finger]]
    qadr = [m.jnt_qposadr[j] for j in jids]
    dadr = [m.jnt_dofadr[j] for j in jids]
    rng = np.array([m.jnt_range[j] for j in jids])
    bid = m.body(TIPS[finger]).id
    jacp = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_forward(m, d)
        err = target - d.body(bid).xpos
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacBody(m, d, jacp, None, bid)
        J = jacp[:, dadr]  # 3x3
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
        for k, a in enumerate(qadr):
            d.qpos[a] = np.clip(d.qpos[a] + step * dq[k], rng[k, 0], rng[k, 1])
    mujoco.mj_forward(m, d)
    return float(np.linalg.norm(target - d.body(bid).xpos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-scene", required=True)
    ap.add_argument("--keyframe", default="open_short_manual")
    ap.add_argument("--target-scene", required=True)
    ap.add_argument("--out-keyframe", default="open_ik")
    ap.add_argument("--close-rad", type=float, default=0.0,
                    help="extra mcp+pip flexion (rad) after IK, to seat tips into contact")
    ap.add_argument("--validate", action="store_true",
                    help="report per-finger tip errors + tip-object distances")
    ap.add_argument("--write-keyframe", action="store_true",
                    help="inject <key name=OUT_KEYFRAME> into the target scene XML (CEM seed)")
    args = ap.parse_args()

    tips, palm, obj_qpos = tip_targets(args.base_scene, args.keyframe)
    print(f"[targets] tip world xyz (from {args.base_scene} @ {args.keyframe}):")
    for f in FINGERS:
        print(f"    {f:7s} {np.round(tips[f], 4)}")

    m = mujoco.MjModel.from_xml_path(args.target_scene)
    d = mujoco.MjData(m)
    # seed from the target scene's own open_short_manual (baseline finger angles) if present
    try:
        mujoco.mj_resetDataKeyframe(m, d, m.key(args.keyframe).id)
    except Exception:
        pass
    # pin palm + object to the baseline keyframe pose (only the hand geometry differs)
    for j, v in palm.items():
        if _has_joint(m, j):
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
    d.qpos[:7] = obj_qpos
    mujoco.mj_forward(m, d)

    print("[ik] retargeting fingers ...")
    errs = {}
    for f in FINGERS:
        errs[f] = ik_finger(m, d, f, tips[f])
        print(f"    {f:7s} residual {errs[f]*1000:.2f} mm")

    if args.close_rad > 0:
        for f in FINGERS:
            for jn in (FINGERS[f][1], FINGERS[f][2]):  # mcp, pip
                jid = m.joint(jn).id
                a = m.jnt_qposadr[jid]
                lo, hi = m.jnt_range[jid]
                d.qpos[a] = np.clip(d.qpos[a] + args.close_rad, lo, hi)
        mujoco.mj_forward(m, d)

    # build the finger ctrl vector (position actuators, in actuator order)
    finger_joint_names = [jn for f in FINGERS for jn in FINGERS[f]]
    ctrl = {}
    for jn in finger_joint_names:
        ctrl[jn] = float(d.qpos[m.jnt_qposadr[m.joint(jn).id]])

    print("[ik] retargeted finger joint angles:")
    for f in FINGERS:
        print(f"    {f:7s} yaw/mcp/pip = "
              + " ".join(f"{ctrl[jn]:+.4f}" for jn in FINGERS[f]))

    if args.validate:
        # object surface: tip-to-object-body distance (proxy for contact readiness)
        mujoco.mj_forward(m, d)
        print("[validate] tip -> object-center distance (target obj center "
              f"{np.round(obj_qpos[:3],4)}):")
        for f in FINGERS:
            tp = d.body(TIPS[f]).xpos
            print(f"    {f:7s} tip {np.round(tp,4)}  dist {np.linalg.norm(tp-obj_qpos[:3])*1000:.1f} mm")

    # emit the full qpos + finger ctrl so it can be written as an `open_ik` keyframe
    np.set_printoptions(suppress=True, precision=6)
    print("\n[out] full qpos:\n" + " ".join(f"{v:.6g}" for v in d.qpos))
    print("[out] finger ctrl (yaw/mcp/pip x thumb,index,middle):\n"
          + " ".join(f"{ctrl[jn]:.6g}" for jn in finger_joint_names))

    if args.write_keyframe:
        # full ctrl vector in actuator order: hold each actuator's transmission joint
        # at the IK-open pose (position actuators). CEM anchors to this ctrl.
        ctrl_vec = []
        for a in range(m.nu):
            jid = m.actuator_trnid[a, 0]
            ctrl_vec.append(float(d.qpos[m.jnt_qposadr[jid]]) if jid >= 0 else 0.0)
        qpos_s = " ".join(f"{v:.6g}" for v in d.qpos)
        ctrl_s = " ".join(f"{v:.6g}" for v in ctrl_vec)
        _inject_keyframe(Path(args.target_scene), args.out_keyframe, qpos_s, ctrl_s)
        print(f"[write] injected <key name=\"{args.out_keyframe}\"> into {args.target_scene}")
    return errs


def _inject_keyframe(scene: Path, name: str, qpos: str, ctrl: str):
    import xml.etree.ElementTree as ET
    tree = ET.parse(scene)
    root = tree.getroot()
    kf = root.find("keyframe")
    if kf is None:
        kf = ET.SubElement(root, "keyframe")
    for k in kf.findall("key"):
        if k.get("name") == name:
            kf.remove(k)
    ET.SubElement(kf, "key", {"name": name, "qpos": qpos, "ctrl": ctrl})
    ET.indent(tree, space="  ")
    tree.write(scene, encoding="unicode")


if __name__ == "__main__":
    main()
