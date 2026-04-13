from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a scene XML variant with a prism object by editing the cube geom size."
    )
    parser.add_argument(
        "--base-scene-xml",
        type=Path,
        default=PROJECT_ROOT / "assets" / "mjcf" / "scene.xml",
        help="Input scene MJCF path.",
    )
    parser.add_argument(
        "--size-x",
        type=float,
        default=0.02,
        help="Prism half-size in X (meters).",
    )
    parser.add_argument(
        "--size-y",
        type=float,
        required=True,
        help="Prism half-size in Y (meters).",
    )
    parser.add_argument(
        "--size-z",
        type=float,
        default=0.02,
        help="Prism half-size in Z (meters).",
    )
    parser.add_argument(
        "--output-scene-xml",
        type=Path,
        default=None,
        help="Output scene XML path. If omitted, auto-generates under assets/mjcf/generated.",
    )
    return parser


def _find_cube_geom(root: ET.Element) -> ET.Element:
    cube_body = None
    for body in root.iter("body"):
        if body.get("name") == "cube":
            cube_body = body
            break
    if cube_body is None:
        raise ValueError("Could not find body named 'cube'")

    for geom in cube_body.findall("geom"):
        if geom.get("type") == "box":
            return geom

    raise ValueError("Cube body does not contain a box geom")


def _format_xyz(x: float, y: float, z: float) -> str:
    return f"{x:.6f} {y:.6f} {z:.6f}"


def _default_output_path(size_x: float, size_y: float, size_z: float) -> Path:
    return (
        PROJECT_ROOT
        / "assets"
        / "mjcf"
        / "generated"
        / f"scene_prism_x{size_x:.4f}_y{size_y:.4f}_z{size_z:.4f}.xml"
    )


def main() -> None:
    args = build_parser().parse_args()

    root = ET.parse(args.base_scene_xml).getroot()
    cube_geom = _find_cube_geom(root)
    cube_geom.set("size", _format_xyz(args.size_x, args.size_y, args.size_z))

    cube_body = None
    for body in root.iter("body"):
        if body.get("name") == "cube":
            cube_body = body
            break
    assert cube_body is not None
    cube_body.set("pos", _format_xyz(0.0, 0.0, args.size_z))

    out_path = args.output_scene_xml or _default_output_path(args.size_x, args.size_y, args.size_z)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=False)

    print(f"Wrote prism scene XML: {out_path}")


if __name__ == "__main__":
    main()
