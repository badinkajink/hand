from __future__ import annotations
# pyright: reportMissingImports=false

import argparse
from pathlib import Path
import sys

# Allow direct script execution without editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from morphohand.tools.morphology_xml import (  # noqa: E402
    MorphologyValues,
    create_rigid_hand_and_scene_xmls,
    extract_morphology_from_qpos,
)


def _parse_qpos(raw: str) -> list[float]:
    return [float(v) for v in raw.replace(",", " ").split()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired rigid hand+scene XMLs by baking morphology values into transforms "
            "and removing morph joints (x/y/len)."
        )
    )
    parser.add_argument(
        "--base-hand-xml",
        type=Path,
        default=PROJECT_ROOT / "assets" / "mjcf" / "hand.xml",
        help="Source hand MJCF with morph joints.",
    )
    parser.add_argument(
        "--base-scene-xml",
        type=Path,
        default=PROJECT_ROOT / "assets" / "mjcf" / "scene.xml",
        help="Source scene MJCF with morph joints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "assets" / "mjcf" / "generated",
        help="Directory for paired auto-named outputs.",
    )
    parser.add_argument(
        "--hand-prefix",
        default="hand",
        help="Filename prefix for generated hand XML.",
    )
    parser.add_argument(
        "--scene-prefix",
        default="scene",
        help="Filename prefix for generated scene XML.",
    )

    parser.add_argument("--thumb", nargs=3, type=float, metavar=("X", "Y", "LEN"))
    parser.add_argument("--index", nargs=3, type=float, metavar=("X", "Y", "LEN"))
    parser.add_argument("--middle", nargs=3, type=float, metavar=("X", "Y", "LEN"))

    parser.add_argument(
        "--qpos",
        type=str,
        default=None,
        help=(
            "Optional full qpos vector copied from simulation. When provided, morphology "
            "is extracted from qpos and overrides --thumb/--index/--middle."
        ),
    )
    parser.add_argument(
        "--scene-qpos",
        action="store_true",
        help="Interpret --qpos as scene layout with cube+palm prefix.",
    )
    return parser


def _morphology_from_args(args: argparse.Namespace) -> MorphologyValues:
    if args.qpos is not None:
        qpos = _parse_qpos(args.qpos)
        return extract_morphology_from_qpos(qpos=qpos, has_scene_prefix=args.scene_qpos)

    if not (args.thumb and args.index and args.middle):
        raise SystemExit(
            "Provide either --qpos or all of --thumb, --index, --middle morphology triples."
        )

    return MorphologyValues(
        thumb_x=args.thumb[0],
        thumb_y=args.thumb[1],
        thumb_len=args.thumb[2],
        index_x=args.index[0],
        index_y=args.index[1],
        index_len=args.index[2],
        middle_x=args.middle[0],
        middle_y=args.middle[1],
        middle_len=args.middle[2],
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    morphology = _morphology_from_args(args)

    hand_out, scene_out = create_rigid_hand_and_scene_xmls(
        base_hand_xml_path=args.base_hand_xml,
        base_scene_xml_path=args.base_scene_xml,
        morphology=morphology,
        output_dir=args.output_dir,
        hand_prefix=args.hand_prefix,
        scene_prefix=args.scene_prefix,
    )

    print(f"Wrote rigid hand XML:  {hand_out}")
    print(f"Wrote rigid scene XML: {scene_out}")


if __name__ == "__main__":
    main()
