"""Generate `_short_proximal` base-scene variants for all eval objects.

For each source scene, produces a copy where the three `*_len_frame` body
positions are changed from "0.05 0 0" to "0.025 0 0" (the same convention
as the existing `scene_power_drill_short_proximal.xml`). Morph joints and
keyframes are otherwise preserved.

The output scenes are the *base* scenes for the morphology sweep — they
keep morph joints so the sweep can perturb thumb/index/middle x, y, len.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "assets" / "mjcf"


# (source, output) pairs. medium has 3 keyframes incl. open_90vertical so we
# preserve them all.
SCENES = [
    ("scene.xml",                          "scene_cube_short_proximal.xml"),
    ("scene_prism.xml",                    "scene_prism_short_proximal.xml"),
    ("scene_screwdriver_medium_flat.xml",  "scene_screwdriver_medium_flat_short_proximal.xml"),
    ("scene_screwdriver_medium_vertical.xml","scene_screwdriver_medium_vertical_short_proximal.xml"),
    ("scene_screwdriver_small_flat.xml",   "scene_screwdriver_small_flat_short_proximal.xml"),
    ("scene_screwdriver_medium.xml",       "scene_screwdriver_medium_short_proximal.xml"),
]


def shorten_proximal(src_xml: Path, dst_xml: Path) -> None:
    text = src_xml.read_text()
    replaced = 0
    for finger in ("thumb", "index", "middle"):
        marker = f'<body name="{finger}_len_frame" pos="0.05 0 0"'
        if marker in text:
            text = text.replace(
                marker,
                f'<body name="{finger}_len_frame" pos="0.025 0 0"',
                1,
            )
            replaced += 1
    if replaced != 3:
        raise RuntimeError(
            f"Expected to shorten 3 len_frame body positions in {src_xml}, only matched {replaced}"
        )
    # Also rewrite the model name embedded in the root element so the new
    # scene reports its own identity.
    text = text.replace(
        f'<mujoco model="{src_xml.stem}"',
        f'<mujoco model="{dst_xml.stem}"',
    )
    dst_xml.write_text(text)


def main() -> None:
    for src_name, dst_name in SCENES:
        src = MJCF / src_name
        dst = MJCF / dst_name
        shorten_proximal(src, dst)
        print(f"wrote {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
