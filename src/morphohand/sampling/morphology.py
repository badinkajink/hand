from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from morphohand.tools.morphology_xml import (
    MorphologyValues,
    extract_morphology_from_qpos,
)


FINGER_ACTUATOR_NAMES: tuple[str, ...] = (
    "a_thumb_yaw",
    "a_thumb_mcp",
    "a_thumb_pip",
    "a_index_yaw",
    "a_index_mcp",
    "a_index_pip",
    "a_middle_yaw",
    "a_middle_mcp",
    "a_middle_pip",
)


@dataclass(frozen=True)
class MorphologyBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    len_min: float
    len_max: float


def clip_morphology(values: MorphologyValues, bounds: MorphologyBounds) -> MorphologyValues:
    return MorphologyValues(
        thumb_x=float(np.clip(values.thumb_x, bounds.x_min, bounds.x_max)),
        thumb_y=float(np.clip(values.thumb_y, bounds.y_min, bounds.y_max)),
        thumb_len=float(np.clip(values.thumb_len, bounds.len_min, bounds.len_max)),
        index_x=float(np.clip(values.index_x, bounds.x_min, bounds.x_max)),
        index_y=float(np.clip(values.index_y, bounds.y_min, bounds.y_max)),
        index_len=float(np.clip(values.index_len, bounds.len_min, bounds.len_max)),
        middle_x=float(np.clip(values.middle_x, bounds.x_min, bounds.x_max)),
        middle_y=float(np.clip(values.middle_y, bounds.y_min, bounds.y_max)),
        middle_len=float(np.clip(values.middle_len, bounds.len_min, bounds.len_max)),
    )


def morph_to_array(m: MorphologyValues) -> np.ndarray:
    return np.array(
        [
            m.thumb_x, m.thumb_y, m.thumb_len,
            m.index_x, m.index_y, m.index_len,
            m.middle_x, m.middle_y, m.middle_len,
        ],
        dtype=np.float64,
    )


def morph_distance(a: MorphologyValues, b: MorphologyValues) -> float:
    return float(np.linalg.norm(morph_to_array(a) - morph_to_array(b)))


def _morph_key(m: MorphologyValues) -> tuple[float, ...]:
    return (
        round(m.thumb_x, 6), round(m.thumb_y, 6), round(m.thumb_len, 6),
        round(m.index_x, 6), round(m.index_y, 6), round(m.index_len, 6),
        round(m.middle_x, 6), round(m.middle_y, 6), round(m.middle_len, 6),
    )


def sample_morphologies(
    base: MorphologyValues,
    sample_count: int,
    rng: np.random.Generator,
    bounds: MorphologyBounds,
    x_perturb: float,
    y_perturb: float,
    len_perturb: float,
) -> list[MorphologyValues]:
    """Uniform-perturbation sampler around `base`, deduped + bounds-clipped.
    Always includes `base` as the first candidate."""
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")

    candidates: list[MorphologyValues] = [base]
    seen: set[tuple[float, ...]] = {_morph_key(base)}

    while len(candidates) < sample_count:
        proposal = MorphologyValues(
            thumb_x=base.thumb_x + float(rng.uniform(-x_perturb, x_perturb)),
            thumb_y=base.thumb_y + float(rng.uniform(-y_perturb, y_perturb)),
            thumb_len=base.thumb_len + float(rng.uniform(-len_perturb, len_perturb)),
            index_x=base.index_x + float(rng.uniform(-x_perturb, x_perturb)),
            index_y=base.index_y + float(rng.uniform(-y_perturb, y_perturb)),
            index_len=base.index_len + float(rng.uniform(-len_perturb, len_perturb)),
            middle_x=base.middle_x + float(rng.uniform(-x_perturb, x_perturb)),
            middle_y=base.middle_y + float(rng.uniform(-y_perturb, y_perturb)),
            middle_len=base.middle_len + float(rng.uniform(-len_perturb, len_perturb)),
        )
        proposal = clip_morphology(proposal, bounds)
        key = _morph_key(proposal)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(proposal)

    return candidates


def morph_suffix(m: MorphologyValues) -> str:
    """Filename-safe encoding of a morphology, matching the `tXXXX_iXXXX_mXXXX` convention."""
    return (
        f"t{m.thumb_x:+0.4f}_{m.thumb_y:+0.4f}_{m.thumb_len:+0.4f}_"
        f"i{m.index_x:+0.4f}_{m.index_y:+0.4f}_{m.index_len:+0.4f}_"
        f"m{m.middle_x:+0.4f}_{m.middle_y:+0.4f}_{m.middle_len:+0.4f}"
    ).replace("+", "p").replace("-", "n").replace(".", "d")


def _decode_token(token: str) -> float:
    if token.startswith("p"):
        sign, body = 1.0, token[1:]
    elif token.startswith("n"):
        sign, body = -1.0, token[1:]
    else:
        raise ValueError(f"Invalid morphology token sign: {token}")
    if "d" not in body:
        raise ValueError(f"Invalid morphology token format: {token}")
    return sign * float(body.replace("d", ".", 1))


def parse_morphology_from_generated_filename(scene_xml: Path) -> MorphologyValues:
    """Reverse `morph_suffix`: pulls morphology from a `scene_..._tXXXX_iXXXX_mXXXX.xml` name."""
    parts = scene_xml.stem.split("_")
    t_part = next((p for p in parts if p.startswith("t")), None)
    i_part = next((p for p in parts if p.startswith("i")), None)
    m_part = next((p for p in parts if p.startswith("m")), None)
    if t_part is None or i_part is None or m_part is None:
        raise ValueError(f"Scene filename missing t/i/m morphology groups: {scene_xml.name}")

    def split_triplet(group: str) -> tuple[float, float, float]:
        payload = group[1:]
        chunks = [payload[k : k + 7] for k in range(0, 21, 7)]
        return (_decode_token(chunks[0]), _decode_token(chunks[1]), _decode_token(chunks[2]))

    tx, ty, tl = split_triplet(t_part)
    ix, iy, il = split_triplet(i_part)
    mx, my, ml = split_triplet(m_part)
    return MorphologyValues(tx, ty, tl, ix, iy, il, mx, my, ml)


def read_keyframe_qpos(
    scene_xml: Path,
    keyframe_name: str,
    min_qpos_len: int = 0,
) -> list[float]:
    root = ET.parse(scene_xml).getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        raise ValueError(f"No <keyframe> section in {scene_xml}")
    for key in keyframe.findall("key"):
        if key.get("name") == keyframe_name:
            qpos_raw = key.get("qpos") or ""
            values = [float(v) for v in qpos_raw.replace("\n", " ").split()]
            if min_qpos_len and len(values) < min_qpos_len:
                raise ValueError(
                    f"Keyframe '{keyframe_name}' in {scene_xml} has short qpos ({len(values)})"
                )
            return values
    raise ValueError(f"Keyframe '{keyframe_name}' not found in {scene_xml}")


def parse_morphology_from_keyframe(scene_xml: Path, keyframe_name: str) -> MorphologyValues:
    qpos = read_keyframe_qpos(scene_xml=scene_xml, keyframe_name=keyframe_name, min_qpos_len=31)
    return extract_morphology_from_qpos(qpos=qpos, has_scene_prefix=True)


def load_base_morphology(scene_xml: Path, keyframe_name: str = "open") -> MorphologyValues:
    """Best-effort extraction: prefer filename encoding, fall back to keyframe qpos."""
    try:
        return parse_morphology_from_generated_filename(scene_xml)
    except Exception:
        return parse_morphology_from_keyframe(scene_xml, keyframe_name)
