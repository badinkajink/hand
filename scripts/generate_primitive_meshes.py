"""Generate OBJ meshes for the primitive eval objects so Lightning Grasp can ingest them.

Outputs land in `external/lightning-grasp/assets/object/morphohand/`:
  - prism_22x68x18mm.obj                (box, half-extents from scene_prism.xml)
  - screwdriver_medium_25x100mm.obj     (cylinder, sized from scene_screwdriver_medium*.xml)
  - screwdriver_small_8x80mm.obj        (cylinder, sized from scene_screwdriver_small_flat.xml)

Cube already has cube_40mm.obj available; not regenerated here.

Uses trimesh so face winding (and therefore vertex normals) is correct —
Lightning's `get_support_point_mask` requires outward-pointing normals or
its support-point filter returns 0 and the pipeline crashes.
"""
from __future__ import annotations

from pathlib import Path

import trimesh

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "external" / "lightning-grasp" / "assets" / "object" / "morphohand"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # prism: scene_prism.xml geom size=0.01125 0.03375 0.009 -> full extents 22.5x67.5x18mm
    prism = trimesh.creation.box(extents=(0.0225, 0.0675, 0.018))
    prism.export(OUT_DIR / "prism_22x68x18mm.obj")
    print(f"wrote prism: vertices={len(prism.vertices)} faces={len(prism.faces)}")

    # screwdriver_medium: cylinder r=0.0125 height=0.1 (full length); 64-segment ring
    medium = trimesh.creation.cylinder(radius=0.0125, height=0.1, sections=64)
    medium.export(OUT_DIR / "screwdriver_medium_25x100mm.obj")
    print(f"wrote medium cyl: vertices={len(medium.vertices)} faces={len(medium.faces)}")

    # screwdriver_small_flat: cylinder r=0.004 height=0.08; 64-segment ring
    small = trimesh.creation.cylinder(radius=0.004, height=0.08, sections=64)
    small.export(OUT_DIR / "screwdriver_small_8x80mm.obj")
    print(f"wrote small cyl: vertices={len(small.vertices)} faces={len(small.faces)}")


if __name__ == "__main__":
    main()
