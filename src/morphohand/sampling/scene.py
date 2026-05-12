from __future__ import annotations
# pyright: reportMissingImports=false

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from morphohand.tools.morphology_xml import (
    MorphologyValues,
    apply_morphology_to_qpos,
    create_rigid_morphology_xml,
    extract_morphology_from_qpos,
    rebase_asset_file_paths,
)

from .morphology import FINGER_ACTUATOR_NAMES


def write_scene_with_morphology(
    base_scene_xml: Path,
    output_scene_xml: Path,
    morphology: MorphologyValues,
) -> Path:
    """Clone `base_scene_xml` into `output_scene_xml` with every keyframe's qpos
    patched to carry the given morphology. Morph joints remain (non-rigid emission)."""
    root = ET.parse(base_scene_xml).getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        raise ValueError(f"No <keyframe> section in {base_scene_xml}")

    for key in keyframe.findall("key"):
        qpos_raw = key.get("qpos")
        if not qpos_raw:
            continue
        qpos = [float(v) for v in qpos_raw.replace("\n", " ").split()]
        if len(qpos) < 31:
            continue
        apply_morphology_to_qpos(qpos=qpos, morphology=morphology, has_scene_prefix=True)
        key.set("qpos", "\n        " + " ".join(f"{v:.10g}" for v in qpos) + "\n      ")

    rebase_asset_file_paths(root=root, source_xml=base_scene_xml, output_xml=output_scene_xml)
    root.set("model", output_scene_xml.stem)
    ET.indent(root, space="  ")
    output_scene_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_scene_xml, encoding="utf-8", xml_declaration=False)
    return output_scene_xml


def write_rigid_scene_with_object_size(
    base_scene_xml: Path,
    output_scene_xml: Path,
    morphology: MorphologyValues,
    object_body_name: str = "cube",
    object_geom_type: str = "box",
    size_xyz: tuple[float, float, float] | None = None,
) -> Path:
    """Emit a rigid scene (morphology baked in, morph joints removed). If `size_xyz`
    is provided, resize the primary geom of the named object body to those half-extents
    and re-seat it at z=size_z."""
    create_rigid_morphology_xml(
        base_xml_path=base_scene_xml,
        morphology=morphology,
        output_xml_path=output_scene_xml,
        model_name=output_scene_xml.stem,
    )

    if size_xyz is None:
        return output_scene_xml

    root = ET.parse(output_scene_xml).getroot()
    target_body = None
    for body in root.iter("body"):
        if body.get("name") == object_body_name:
            target_body = body
            break
    if target_body is None:
        raise ValueError(f"Could not find body '{object_body_name}' in {output_scene_xml}")

    target_geom = None
    for geom in target_body.findall("geom"):
        if geom.get("type") == object_geom_type:
            target_geom = geom
            break
    if target_geom is None:
        raise ValueError(
            f"Could not find {object_geom_type} geom in body '{object_body_name}' of {output_scene_xml}"
        )

    sx, sy, sz = size_xyz
    target_geom.set("size", f"{sx:.6f} {sy:.6f} {sz:.6f}")
    target_body.set("pos", f"0.000000 0.000000 {sz:.6f}")

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output_scene_xml, encoding="utf-8", xml_declaration=False)
    return output_scene_xml


def freeze_scene_for_eval(
    scene_xml: Path,
    keyframe: str,
    frozen_scene_xml: Path,
) -> Path:
    """Produce a frozen scene XML suitable for grasp/morphology evaluation.

    REQUIRED before constructing any `Phase1GraspEvaluator` against a scene
    that still has morph joints (e.g. anything under ``assets/mjcf/``
    except the pre-generated rigid ones). The morph joints
    (``thumb_x``, ``thumb_y``, ``thumb_len``, ...) are not actuated but they
    will drift during the rollout, so the "fixed" morphology silently
    changes and ruins the experiment.

    Behaviour:
    - If the source scene already has no morph joints, the file is copied
      verbatim to ``frozen_scene_xml`` (the contract is "frozen path, ready
      to evaluate", regardless of starting point).
    - Otherwise, the morphology is extracted from the qpos of the requested
      keyframe and baked into body transforms via
      ``create_rigid_morphology_xml``.
    """
    frozen_scene_xml = Path(frozen_scene_xml)
    frozen_scene_xml.parent.mkdir(parents=True, exist_ok=True)

    root = ET.parse(scene_xml).getroot()
    has_morph_joints = any(j.get("name") == "thumb_x" for j in root.iter("joint"))
    if not has_morph_joints:
        shutil.copyfile(scene_xml, frozen_scene_xml)
        return frozen_scene_xml

    keyframe_elem = root.find("keyframe")
    if keyframe_elem is None:
        raise ValueError(f"No <keyframe> section in {scene_xml}")

    selected_key = None
    for key in keyframe_elem.findall("key"):
        if key.get("name") == keyframe:
            selected_key = key
            break
    if selected_key is None or not selected_key.get("qpos"):
        raise ValueError(
            f"Keyframe '{keyframe}' not found or missing qpos in {scene_xml}"
        )

    qpos = [float(v) for v in selected_key.get("qpos", "").replace("\n", " ").split()]
    morphology = extract_morphology_from_qpos(qpos, has_scene_prefix=True)
    create_rigid_morphology_xml(
        base_xml_path=Path(scene_xml),
        morphology=morphology,
        output_xml_path=frozen_scene_xml,
        model_name=frozen_scene_xml.stem,
    )
    return frozen_scene_xml


def simulate_settle_qpos(
    scene_xml: Path,
    keyframe_name: str,
    finger_ctrl: np.ndarray,
    settle_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reset to keyframe, hold `finger_ctrl` on the 9 finger actuators for N steps.
    Returns (settled_qpos, full_ctrl_vector)."""
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if key_id < 0:
        raise ValueError(f"Keyframe '{keyframe_name}' not found in {scene_xml}")

    actuator_ids: list[int] = []
    for name in FINGER_ACTUATOR_NAMES:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if idx < 0:
            raise ValueError(f"Actuator '{name}' not found in {scene_xml}")
        actuator_ids.append(int(idx))

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    ctrl = data.ctrl.copy()
    ctrl[np.asarray(actuator_ids, dtype=np.int32)] = finger_ctrl

    for _ in range(settle_steps):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

    return data.qpos.copy(), ctrl.copy()


def add_foundational_keyframe(
    scene_xml: Path,
    key_name: str,
    qpos: np.ndarray,
    ctrl: np.ndarray,
) -> None:
    """Add (or overwrite) a `<key>` of the given name carrying qpos/ctrl."""
    root = ET.parse(scene_xml).getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")

    existing = None
    for key in keyframe.findall("key"):
        if key.get("name") == key_name:
            existing = key
            break

    qpos_text = "\n        " + " ".join(f"{v:.10g}" for v in qpos.tolist()) + "\n      "
    ctrl_text = "\n        " + " ".join(f"{v:.10g}" for v in ctrl.tolist()) + "\n      "

    if existing is None:
        ET.SubElement(keyframe, "key", name=key_name, qpos=qpos_text, ctrl=ctrl_text)
    else:
        existing.set("qpos", qpos_text)
        existing.set("ctrl", ctrl_text)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(scene_xml, encoding="utf-8", xml_declaration=False)
