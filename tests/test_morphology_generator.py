# pyright: reportMissingImports=false

from pathlib import Path
import xml.etree.ElementTree as ET

from morphohand.tools.morphology_xml import (
    MorphologyValues,
    apply_morphology_to_qpos,
    create_rigid_hand_and_scene_xmls,
    create_rigid_morphology_xml,
)


def _joint_names(path: Path) -> set[str]:
    root = ET.parse(path).getroot()
    return {j.get("name", "") for j in root.iter("joint")}


def _actuator_joint_names(path: Path) -> set[str]:
    root = ET.parse(path).getroot()
    actuator = root.find("actuator")
    if actuator is None:
        return set()
    return {a.get("joint", "") for a in actuator}


def test_create_rigid_hand_xml_removes_morph_joints(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = root / "assets" / "mjcf" / "hand.xml"
    out = tmp_path / "rigid_hand.xml"

    morphology = MorphologyValues(
        thumb_x=0.01,
        thumb_y=-0.005,
        thumb_len=0.01,
        index_x=0.0,
        index_y=0.004,
        index_len=0.007,
        middle_x=-0.008,
        middle_y=0.002,
        middle_len=0.0,
    )

    create_rigid_morphology_xml(base_xml_path=base, morphology=morphology, output_xml_path=out)

    joints = _joint_names(out)
    for name in (
        "thumb_x",
        "thumb_y",
        "thumb_len",
        "index_x",
        "index_y",
        "index_len",
        "middle_x",
        "middle_y",
        "middle_len",
    ):
        assert name not in joints

    actuator_joints = _actuator_joint_names(out)
    assert actuator_joints == {
        "thumb_yaw",
        "thumb_mcp",
        "thumb_pip",
        "index_yaw",
        "index_mcp",
        "index_pip",
        "middle_yaw",
        "middle_mcp",
        "middle_pip",
    }


def test_create_rigid_hand_and_scene_xmls(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    hand_base = root / "assets" / "mjcf" / "hand.xml"
    scene_base = root / "assets" / "mjcf" / "scene.xml"

    morphology = MorphologyValues(
        thumb_x=0.002,
        thumb_y=-0.003,
        thumb_len=0.004,
        index_x=0.005,
        index_y=0.001,
        index_len=0.006,
        middle_x=-0.007,
        middle_y=0.002,
        middle_len=0.003,
    )

    hand_out, scene_out = create_rigid_hand_and_scene_xmls(
        base_hand_xml_path=hand_base,
        base_scene_xml_path=scene_base,
        morphology=morphology,
        output_dir=tmp_path,
    )

    assert hand_out.exists()
    assert scene_out.exists()
    assert hand_out.name.startswith("hand_")
    assert scene_out.name.startswith("scene_")


def test_apply_morphology_to_qpos_in_place() -> None:
    morphology = MorphologyValues(
        thumb_x=0.01,
        thumb_y=0.02,
        thumb_len=0.03,
        index_x=-0.01,
        index_y=-0.02,
        index_len=0.015,
        middle_x=0.004,
        middle_y=-0.006,
        middle_len=0.01,
    )

    qpos = [0.0] * 31
    apply_morphology_to_qpos(qpos, morphology, has_scene_prefix=True)

    assert qpos[13] == morphology.thumb_x
    assert qpos[14] == morphology.thumb_y
    assert qpos[17] == morphology.thumb_len
    assert qpos[19] == morphology.index_x
    assert qpos[20] == morphology.index_y
    assert qpos[23] == morphology.index_len
    assert qpos[25] == morphology.middle_x
    assert qpos[26] == morphology.middle_y
    assert qpos[29] == morphology.middle_len
