from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import copy
import xml.etree.ElementTree as ET

FINGER_ORDER = ("thumb", "index", "middle")
MORPH_KEYS = ("x", "y", "len")


@dataclass(frozen=True)
class MorphologyValues:
    """Per-finger morphology values in meters."""

    thumb_x: float
    thumb_y: float
    thumb_len: float
    index_x: float
    index_y: float
    index_len: float
    middle_x: float
    middle_y: float
    middle_len: float

    def by_finger(self) -> Dict[str, Dict[str, float]]:
        return {
            "thumb": {"x": self.thumb_x, "y": self.thumb_y, "len": self.thumb_len},
            "index": {"x": self.index_x, "y": self.index_y, "len": self.index_len},
            "middle": {
                "x": self.middle_x,
                "y": self.middle_y,
                "len": self.middle_len,
            },
        }


def _format_token(value: float) -> str:
    """Filename-safe encoding: +0.012 -> p0d0120 and -0.012 -> n0d0120."""
    return f"{value:+0.4f}".replace("+", "p").replace("-", "n").replace(".", "d")


def build_rigid_model_name(prefix: str, morphology: MorphologyValues) -> str:
    """Build a deterministic filename stem with grouped per-finger morphology."""
    t = morphology.by_finger()["thumb"]
    i = morphology.by_finger()["index"]
    m = morphology.by_finger()["middle"]
    return (
        f"{prefix}_"
        f"t{_format_token(t['x'])}{_format_token(t['y'])}{_format_token(t['len'])}_"
        f"i{_format_token(i['x'])}{_format_token(i['y'])}{_format_token(i['len'])}_"
        f"m{_format_token(m['x'])}{_format_token(m['y'])}{_format_token(m['len'])}"
    )


def build_morphology_suffix(morphology: MorphologyValues) -> str:
    """Build only the morphology token suffix (no leading model prefix)."""
    return build_rigid_model_name(prefix="morph", morphology=morphology).split("_", 1)[1]


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise ValueError(f"Could not find body '{name}' in XML")


def _find_joint(root: ET.Element, name: str) -> ET.Element:
    for joint in root.iter("joint"):
        if joint.get("name") == name:
            return joint
    raise ValueError(f"Could not find joint '{name}' in XML")


def _parse_xyz(text: str) -> list[float]:
    return [float(v) for v in text.split()]


def _fmt_xyz(values: list[float]) -> str:
    return " ".join(f"{v:.6f}".rstrip("0").rstrip(".") if abs(v) >= 1e-12 else "0" for v in values)


def _remove_joint_by_name(root: ET.Element, joint_name: str) -> None:
    for body in root.iter("body"):
        for child in list(body):
            if child.tag == "joint" and child.get("name") == joint_name:
                body.remove(child)
                return
    raise ValueError(f"Could not remove joint '{joint_name}'")


def _remove_actuators_for_joints(root: ET.Element, joint_names: set[str]) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        return

    for child in list(actuator):
        joint_name = child.get("joint")
        if joint_name in joint_names:
            actuator.remove(child)


def _remove_defaults_by_class(root: ET.Element, class_names: set[str]) -> None:
    # Keep defaults in the base file; only remove if explicitly class-scoped and obsolete.
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "default" and child.get("class") in class_names:
                has_non_default_children = any(grand.tag != "joint" for grand in child)
                if not has_non_default_children:
                    parent.remove(child)


def rebase_asset_file_paths(root: ET.Element, source_xml: Path, output_xml: Path) -> None:
    """Rewrite `file=` attributes (mesh/texture) so they resolve from the new XML path."""
    src_dir = source_xml.resolve().parent
    out_dir = output_xml.resolve().parent
    for elem in root.iter():
        file_attr = elem.get("file")
        if not file_attr:
            continue
        src_path = (src_dir / file_attr).resolve() if not Path(file_attr).is_absolute() else Path(file_attr)
        if not src_path.exists():
            continue
        try:
            elem.set("file", str(src_path.relative_to(out_dir)))
        except ValueError:
            elem.set("file", str(src_path))


def _is_scene_model(root: ET.Element) -> bool:
    return any(j.get("name") == "palm_px" for j in root.iter("joint"))


def _default_scene_prefix_qpos() -> list[float]:
    # cube freejoint + palm 6DOF
    return [0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]


def _scene_prefix_qpos_values(root: ET.Element) -> list[float]:
    """Return qpos for free joints before the finger chain (cube + palm pose in scene.xml)."""
    keyframe = root.find("keyframe")
    if keyframe is None:
        return _default_scene_prefix_qpos()

    open_key = None
    for key in keyframe.findall("key"):
        if key.get("name") == "open":
            open_key = key
            break

    if open_key is None or open_key.get("qpos") is None:
        return _default_scene_prefix_qpos()

    all_vals = [float(v) for v in open_key.get("qpos", "").split()]
    if len(all_vals) < 13:
        return _default_scene_prefix_qpos()

    # Scene layout: first 13 qpos entries are cube freejoint (7) + palm pose (6).
    return all_vals[:13]


def _default_finger_ctrl_values() -> list[float]:
    vals = []
    for finger in FINGER_ORDER:
        if finger == "thumb":
            vals.extend([0.0, 2.0, -0.707])
        else:
            vals.extend([0.0, 1.25, 1.0])
    return vals


def _source_finger_ctrl_values(open_key: ET.Element | None, is_scene: bool) -> list[float] | None:
    """The 9 finger ctrl-joint angles already in the source `open` key, or None.

    Called before the morph qpos entries are stripped, so the layout is still the full
    [x, y, yaw, mcp, len, pip] x 3 block (offset by the 13-entry object+palm prefix on a
    scene model). For every baseline scene these values equal `_default_finger_ctrl_values()`,
    so preserving them is a no-op there — but a hand whose open pose is NOT the baseline's
    (e.g. the perpendicular/opposed-pair topology, whose fingers face each other and whose
    open pose comes from fingertip IK) would otherwise have it silently overwritten.
    """
    if open_key is None or not open_key.get("qpos"):
        return None
    values = [float(v) for v in (open_key.get("qpos") or "").replace("\n", " ").split()]
    start = 13 if is_scene else 0
    if len(values) < start + 18:
        return None
    block = values[start : start + 18]
    return [block[i] for i in (2, 3, 5, 8, 9, 11, 14, 15, 17)]


def _write_open_keyframe(root: ET.Element) -> None:
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")

    open_key = None
    for key in keyframe.findall("key"):
        if key.get("name") == "open":
            open_key = key
            break

    is_scene = _is_scene_model(root)
    source_ctrl_joints = _source_finger_ctrl_values(open_key, is_scene)
    source_ctrl = None
    if open_key is not None and open_key.get("ctrl"):
        source_ctrl = [float(v) for v in (open_key.get("ctrl") or "").replace("\n", " ").split()]

    if open_key is None:
        open_key = ET.SubElement(keyframe, "key", name="open")

    ctrl_joint_values = source_ctrl_joints or _default_finger_ctrl_values()
    prefix_vals = _scene_prefix_qpos_values(root) if is_scene else []
    qpos_vals = prefix_vals + ctrl_joint_values
    open_key.set("qpos", "\n        " + " ".join(f"{v:g}" for v in qpos_vals) + "\n      ")

    actuator = root.find("actuator")
    if actuator is not None:
        ctrl_lookup = {
            "palm_px": 0.02,
            "palm_py": 0.0,
            "palm_pz": 0.0,
            "palm_rx": 0.0,
            "palm_ry": 0.0,
            "palm_rz": 0.0,
            "thumb_yaw": 0.0,
            "thumb_mcp": 2.0,
            "thumb_pip": -0.707,
            "index_yaw": 0.0,
            "index_mcp": 1.25,
            "index_pip": 1.0,
            "middle_yaw": 0.0,
            "middle_mcp": 1.25,
            "middle_pip": 1.0,
        }
        if source_ctrl is not None and len(source_ctrl) == len(list(actuator)):
            ctrl_vals = source_ctrl
        else:
            ctrl_vals = [ctrl_lookup.get(a.get("joint", ""), 0.0) for a in list(actuator)]
        open_key.set("ctrl", "\n        " + " ".join(f"{v:g}" for v in ctrl_vals) + "\n      ")


def _strip_scene_morph_qpos(qpos_values: list[float]) -> list[float]:
    """Remove morphology qpos entries from a scene qpos vector."""
    if len(qpos_values) < 31:
        return qpos_values

    rigid_prefix = qpos_values[:13]
    rigid_fingers = [
        qpos_values[15],
        qpos_values[16],
        qpos_values[18],
        qpos_values[21],
        qpos_values[22],
        qpos_values[24],
        qpos_values[27],
        qpos_values[28],
        qpos_values[30],
    ]
    return rigid_prefix + rigid_fingers


def create_rigid_morphology_xml(
    base_xml_path: Path,
    morphology: MorphologyValues,
    output_xml_path: Path,
    model_name: str | None = None,
) -> Path:
    """Create a rigid-morphology XML by baking morphology into body transforms.

    The resulting XML has no morphology joints (x/y/len), so only control joints remain.
    """
    root = ET.parse(base_xml_path).getroot()
    root = copy.deepcopy(root)

    if model_name:
        root.set("model", model_name)

    morph = morphology.by_finger()
    removed_joint_names: set[str] = set()

    for finger in FINGER_ORDER:
        mount_body = _find_body(root, f"{finger}_mount")
        len_frame = _find_body(root, f"{finger}_len_frame")

        mount_pos = _parse_xyz(mount_body.get("pos", "0 0 0"))
        mount_pos[0] += morph[finger]["x"]
        mount_pos[1] += morph[finger]["y"]
        mount_body.set("pos", _fmt_xyz(mount_pos))

        len_pos = _parse_xyz(len_frame.get("pos", "0 0 0"))
        len_pos[0] += morph[finger]["len"]
        len_frame.set("pos", _fmt_xyz(len_pos))

        for key in MORPH_KEYS:
            removed_joint_names.add(f"{finger}_{key}")

    for joint_name in sorted(removed_joint_names):
        _remove_joint_by_name(root, joint_name)

    _remove_actuators_for_joints(root, removed_joint_names)
    _remove_defaults_by_class(root, {"morph"})
    _write_open_keyframe(root)

    keyframe = root.find("keyframe")
    if keyframe is not None:
        for key in keyframe.findall("key"):
            qpos_raw = key.get("qpos")
            if not qpos_raw:
                continue
            qpos_values = [float(v) for v in qpos_raw.replace("\n", " ").split()]
            rigid_qpos = _strip_scene_morph_qpos(qpos_values)
            key.set("qpos", "\n        " + " ".join(f"{v:.10g}" for v in rigid_qpos) + "\n      ")

    rebase_asset_file_paths(root, base_xml_path, output_xml_path)

    ET.indent(root, space="  ")
    output_xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_xml_path, encoding="utf-8", xml_declaration=False)
    return output_xml_path


def create_rigid_hand_and_scene_xmls(
    base_hand_xml_path: Path,
    base_scene_xml_path: Path,
    morphology: MorphologyValues,
    output_dir: Path,
    hand_prefix: str = "hand",
    scene_prefix: str = "scene",
) -> tuple[Path, Path]:
    """Generate paired rigid hand+scene XML files for the same morphology."""
    suffix = build_morphology_suffix(morphology)
    hand_stem = f"{hand_prefix}_{suffix}"
    scene_stem = f"{scene_prefix}_{suffix}"

    output_dir.mkdir(parents=True, exist_ok=True)
    hand_path = output_dir / f"{hand_stem}.xml"
    scene_path = output_dir / f"{scene_stem}.xml"

    create_rigid_morphology_xml(
        base_xml_path=base_hand_xml_path,
        morphology=morphology,
        output_xml_path=hand_path,
        model_name=hand_stem,
    )
    create_rigid_morphology_xml(
        base_xml_path=base_scene_xml_path,
        morphology=morphology,
        output_xml_path=scene_path,
        model_name=scene_stem,
    )
    return hand_path, scene_path


def extract_morphology_from_qpos(qpos: list[float], has_scene_prefix: bool = False) -> MorphologyValues:
    """Extract (x,y,len) values from a full qpos vector from hand.xml or scene.xml.

    hand.xml qpos layout (18): [tx,ty, yaw,mcp,tl,pip, ix,iy, yaw,mcp,il,pip, mx,my, yaw,mcp,ml,pip]
    scene.xml qpos layout (31): [cube7, palm6] + the same 18 finger entries.
    """
    start = 13 if has_scene_prefix else 0
    if len(qpos) < start + 18:
        raise ValueError(
            f"qpos length {len(qpos)} is too short; expected at least {start + 18} values"
        )

    block = qpos[start : start + 18]
    return MorphologyValues(
        thumb_x=block[0],
        thumb_y=block[1],
        thumb_len=block[4],
        index_x=block[6],
        index_y=block[7],
        index_len=block[10],
        middle_x=block[12],
        middle_y=block[13],
        middle_len=block[16],
    )


def apply_morphology_to_qpos(
    qpos: list[float],
    morphology: MorphologyValues,
    has_scene_prefix: bool = False,
) -> None:
    """Write morphology values into qpos in-place for hand or scene layouts."""
    start = 13 if has_scene_prefix else 0
    if len(qpos) < start + 18:
        raise ValueError(f"qpos length {len(qpos)} is too short; expected at least {start + 18}")

    qpos[start + 0] = morphology.thumb_x
    qpos[start + 1] = morphology.thumb_y
    qpos[start + 4] = morphology.thumb_len

    qpos[start + 6] = morphology.index_x
    qpos[start + 7] = morphology.index_y
    qpos[start + 10] = morphology.index_len

    qpos[start + 12] = morphology.middle_x
    qpos[start + 13] = morphology.middle_y
    qpos[start + 16] = morphology.middle_len
