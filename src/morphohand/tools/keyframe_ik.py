"""Fingertip-IK keyframe retargeting across morphologies.

The bug this fixes (2026-07-01): the `open_short_manual` keyframe is defined in JOINT space
(fixed finger yaw/mcp/pip angles), hand-tuned so the BASELINE hand's fingertips sit just off
the screwdriver. `generate_morphology_xml.py` bakes a new morphology's finger attachment
(x/y) + link length into the geometry but keeps those SAME joint angles — so on a
repositioned/lengthened finger the fingertip lands at a different WORLD position, no longer
about-to-contact. CEM then seeds from a bad open pose and (e.g. on m05) never seats the
thumb → a spurious "2-finger design" conclusion.

Fix: transfer the keyframe in WORLD space, not joint space. Read the 3 fingertip world XYZ
from the known-good baseline keyframe, then damped-least-squares IK each finger of the
TARGET morphology (3 joints: yaw/mcp/pip → 3D tip target) to the same world positions,
keeping the palm/object pose identical. `retarget_scene()` does the whole thing and injects
an `open_ik` keyframe into the target scene (the CEM seed / LerpFinger open pose).

CLI: scripts/retarget_keyframe_ik.py. Extracted here (CODEBASE_AUDIT.md step 2) so
morph_pipeline_sweep / ik_recem_landscape import a library, not another script.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

FINGERS = {"thumb": ("thumb_yaw", "thumb_mcp", "thumb_pip"),
           "index": ("index_yaw", "index_mcp", "index_pip"),
           "middle": ("middle_yaw", "middle_mcp", "middle_pip")}
TIPS = {"thumb": "thumb_tip", "index": "index_tip", "middle": "middle_tip"}
PALM_JOINTS = ["palm_px", "palm_py", "palm_pz", "palm_rx", "palm_ry", "palm_rz"]


def has_joint(m, name) -> bool:
    try:
        m.joint(name); return True
    except Exception:
        return False


def tip_targets(base_scene: str, keyframe: str):
    """(fingertip world xyz per finger, palm joint values, object free-joint qpos) at the
    given keyframe of the known-good baseline scene."""
    m = mujoco.MjModel.from_xml_path(base_scene)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(keyframe).id)
    mujoco.mj_forward(m, d)
    tips = {f: d.body(TIPS[f]).xpos.copy() for f in FINGERS}
    palm = {j: float(d.qpos[m.jnt_qposadr[m.joint(j).id]]) for j in PALM_JOINTS
            if has_joint(m, j)}
    obj_qpos = d.qpos[:7].copy()  # object free joint
    return tips, palm, obj_qpos


def ik_finger(m, d, finger, target, iters=300, lam=1e-3, step=0.5) -> float:
    """Damped-least-squares IK: drive `finger`'s tip to `target` world xyz. Returns the
    residual distance (m)."""
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


def actuator_ctrl_from_qpos(m, d) -> list[float]:
    """Full ctrl vector in actuator order: hold each actuator's transmission joint at the
    current qpos (position actuators). CEM anchors to this ctrl."""
    ctrl = []
    for a in range(m.nu):
        jid = m.actuator_trnid[a, 0]
        ctrl.append(float(d.qpos[m.jnt_qposadr[jid]]) if jid >= 0 else 0.0)
    return ctrl


def inject_keyframe(scene: Path, name: str, qpos: str, ctrl: str) -> None:
    """Insert/replace <key name=...> in the scene XML.

    Parses with comments retained — ElementTree drops them by default, which silently ate the
    design rationale out of hand-authored scenes every time a keyframe was rewritten.
    """
    tree = ET.parse(scene, ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
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


def retarget_scene(scene: Path, base_scene: Path, keyframe: str = "open_short_manual",
                   out_keyframe: str = "open_ik") -> dict[str, float]:
    """IK-retarget `base_scene`'s fingertip world positions at `keyframe` onto `scene`'s
    morphology and inject the result as an `out_keyframe` keyframe. Returns per-finger IK
    residuals in mm (rounded to 0.01)."""
    tips, palm, obj_qpos = tip_targets(str(base_scene), keyframe)
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    # seed from the target scene's own baseline keyframe (baseline finger angles) if present
    try:
        mujoco.mj_resetDataKeyframe(m, d, m.key(keyframe).id)
    except Exception:
        pass
    # pin palm + object to the baseline keyframe pose (only the hand geometry differs)
    for j, v in palm.items():
        if has_joint(m, j):
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
    d.qpos[:7] = obj_qpos
    mujoco.mj_forward(m, d)
    errs = {f: ik_finger(m, d, f, tips[f]) for f in FINGERS}
    ctrl = actuator_ctrl_from_qpos(m, d)
    inject_keyframe(Path(scene), out_keyframe,
                    " ".join(f"{v:.6g}" for v in d.qpos),
                    " ".join(f"{v:.6g}" for v in ctrl))
    return {f: round(errs[f] * 1000, 2) for f in FINGERS}
