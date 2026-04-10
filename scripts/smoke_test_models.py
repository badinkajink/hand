from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


def actuator_count(xml_path: Path) -> int:
    root = ET.parse(xml_path).getroot()
    actuator = root.find("actuator")
    if actuator is None:
        return 0
    return len(list(actuator))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    hand = root / "assets" / "mjcf" / "hand.xml"
    scene = root / "assets" / "mjcf" / "scene.xml"

    hand_count = actuator_count(hand)
    scene_count = actuator_count(scene)

    print(f"hand.xml actuators: {hand_count}")
    print(f"scene.xml actuators: {scene_count}")

    if hand_count != 9:
        raise SystemExit(f"Expected 9 actuators in hand.xml, got {hand_count}")
    if scene_count != 15:
        raise SystemExit(f"Expected 15 actuators in scene.xml, got {scene_count}")

    print("Model smoke test passed.")


if __name__ == "__main__":
    main()
