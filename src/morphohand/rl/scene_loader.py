"""Scene loading + index resolution for the RL env.

Mirrors `Phase1GraspEvaluator.__init__` id resolution (see
`src/morphohand/optimization/phase1_common.py`) so RL training and CEM
evaluation see identical actuator/joint ordering on the same frozen scene.

The frozen scene path is produced by `freeze_scene_for_eval` (rebased,
self-contained MJCF — see `src/morphohand/sampling/scene.py`).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from morphohand.sampling.scene import freeze_scene_for_eval

# Mirror the actuator order used by Phase1GraspEvaluator (phase1_common.py:142-151).
FINGER_ACTUATOR_NAMES: tuple[str, ...] = (
    "a_thumb_yaw", "a_thumb_mcp", "a_thumb_pip",
    "a_index_yaw", "a_index_mcp", "a_index_pip",
    "a_middle_yaw", "a_middle_mcp", "a_middle_pip",
)
PALM_ACTUATOR_NAMES: tuple[str, ...] = (
    "a_palm_px", "a_palm_py", "a_palm_pz",
    "a_palm_rx", "a_palm_ry", "a_palm_rz",
)
# Fingertip body names (used for contact resolution).
FINGERTIP_BODY_NAMES: tuple[str, ...] = (
    "thumb_tip", "index_tip", "middle_tip",
)


@dataclass
class SceneAssets:
    """Everything an RL env needs to know about a frozen morphology scene."""
    frozen_scene_xml: Path
    keyframe_name: str
    n_finger_actuators: int  # 9
    n_palm_actuators: int    # 6
    finger_actuator_ids: tuple[int, ...]
    palm_actuator_ids: tuple[int, ...]
    finger_ctrl_lo: np.ndarray   # (9,)
    finger_ctrl_hi: np.ndarray   # (9,)
    palm_ctrl_lo: np.ndarray     # (6,)
    palm_ctrl_hi: np.ndarray     # (6,)
    object_body_id: int
    fingertip_body_ids: tuple[int, ...]
    keyframe_qpos: np.ndarray
    keyframe_ctrl: np.ndarray


def prepare_scene(
    base_scene_xml: Path | str,
    keyframe: str,
    output_dir: Path | str,
    object_body_name: str = "cube",
) -> SceneAssets:
    """Freeze the scene, load the model, and resolve all ids/ranges.

    Idempotent: skips freezing if the frozen file already exists.
    """
    import mujoco

    base_scene_xml = Path(base_scene_xml)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_xml = output_dir / f"frozen_{base_scene_xml.stem}.xml"

    if not frozen_xml.exists():
        freeze_scene_for_eval(base_scene_xml, keyframe, frozen_xml)
    else:
        # Sanity-check that the cached frozen file is up-to-date with the
        # base scene mtime; rebuild if stale.
        if base_scene_xml.stat().st_mtime > frozen_xml.stat().st_mtime:
            freeze_scene_for_eval(base_scene_xml, keyframe, frozen_xml)

    model = mujoco.MjModel.from_xml_path(str(frozen_xml))

    def aid(name: str) -> int:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise KeyError(f"actuator '{name}' not found in {frozen_xml}")
        return int(i)

    def bid(name: str) -> int:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise KeyError(f"body '{name}' not found in {frozen_xml}")
        return int(i)

    finger_ids = tuple(aid(n) for n in FINGER_ACTUATOR_NAMES)
    palm_ids = tuple(aid(n) for n in PALM_ACTUATOR_NAMES)
    tip_ids = tuple(bid(n) for n in FINGERTIP_BODY_NAMES)
    obj_id = bid(object_body_name)

    # ctrl ranges from MJCF actuator_ctrlrange (n_actuator, 2)
    crange = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    f_lo = np.array([crange[i, 0] for i in finger_ids])
    f_hi = np.array([crange[i, 1] for i in finger_ids])
    p_lo = np.array([crange[i, 0] for i in palm_ids])
    p_hi = np.array([crange[i, 1] for i in palm_ids])

    # Keyframe qpos + ctrl (used for warm starts / reference trajectories).
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if keyframe_id < 0:
        raise KeyError(f"keyframe '{keyframe}' not in frozen scene")
    kf_qpos = np.asarray(model.key_qpos[keyframe_id], dtype=np.float64).copy()
    kf_ctrl = np.asarray(model.key_ctrl[keyframe_id], dtype=np.float64).copy()

    return SceneAssets(
        frozen_scene_xml=frozen_xml,
        keyframe_name=keyframe,
        n_finger_actuators=len(finger_ids),
        n_palm_actuators=len(palm_ids),
        finger_actuator_ids=finger_ids,
        palm_actuator_ids=palm_ids,
        finger_ctrl_lo=f_lo, finger_ctrl_hi=f_hi,
        palm_ctrl_lo=p_lo, palm_ctrl_hi=p_hi,
        object_body_id=obj_id,
        fingertip_body_ids=tip_ids,
        keyframe_qpos=kf_qpos,
        keyframe_ctrl=kf_ctrl,
    )
