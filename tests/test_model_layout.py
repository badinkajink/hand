from pathlib import Path
import xml.etree.ElementTree as ET


def _count_actuators(path: Path) -> int:
    root = ET.parse(path).getroot()
    actuator = root.find("actuator")
    return 0 if actuator is None else len(list(actuator))


def test_hand_actuator_count() -> None:
    root = Path(__file__).resolve().parents[1]
    hand = root / "assets" / "mjcf" / "hand.xml"
    assert _count_actuators(hand) == 9


def test_scene_actuator_count() -> None:
    root = Path(__file__).resolve().parents[1]
    scene = root / "assets" / "mjcf" / "scene.xml"
    assert _count_actuators(scene) == 15
