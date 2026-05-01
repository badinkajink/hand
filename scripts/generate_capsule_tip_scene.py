from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a fingertip-capsule variant of a MuJoCo scene by replacing tip spheres."
    )
    parser.add_argument("--input-scene", type=Path, required=True)
    parser.add_argument("--output-scene", type=Path, required=True)
    parser.add_argument("--tip-radius", type=float, default=0.005)
    parser.add_argument("--tip-half-length", type=float, default=0.006)
    return parser


def _replace_tip_geoms(root: ET.Element, tip_radius: float, tip_half_length: float) -> int:
    replaced = 0
    for body in root.iter("body"):
        name = body.get("name", "")
        if not name.endswith("_tip"):
            continue

        geoms = [child for child in list(body) if child.tag == "geom"]
        sphere_geoms = [g for g in geoms if g.get("type", "sphere") == "sphere"]
        if not sphere_geoms:
            continue

        base = sphere_geoms[0]
        material = base.get("material")
        friction = base.get("friction")
        condim = base.get("condim")
        solref = base.get("solref")
        solimp = base.get("solimp")

        for g in sphere_geoms:
            body.remove(g)

        cap = ET.Element("geom")
        cap.set("type", "capsule")
        cap.set("fromto", f"{-tip_half_length:.6f} 0 0 {tip_half_length:.6f} 0 0")
        cap.set("size", f"{tip_radius:.6f}")
        if material is not None:
            cap.set("material", material)
        if friction is not None:
            cap.set("friction", friction)
        if condim is not None:
            cap.set("condim", condim)
        if solref is not None:
            cap.set("solref", solref)
        if solimp is not None:
            cap.set("solimp", solimp)

        body.append(cap)
        replaced += 1

    return replaced


def main() -> None:
    args = build_parser().parse_args()

    tree = ET.parse(args.input_scene)
    root = tree.getroot()

    replaced = _replace_tip_geoms(
        root,
        tip_radius=float(args.tip_radius),
        tip_half_length=float(args.tip_half_length),
    )
    if replaced == 0:
        raise RuntimeError("No fingertip sphere groups found to replace")

    ET.indent(tree, space="  ")
    args.output_scene.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output_scene, encoding="utf-8", xml_declaration=False)
    print(f"Wrote {args.output_scene} with {replaced} fingertip capsule geoms")


if __name__ == "__main__":
    main()
