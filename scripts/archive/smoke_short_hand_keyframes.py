"""Pre-flight smoke test: do existing keyframes still close on the short-proximal hand?

For each (scene, keyframe, object_body) in the eval set, this script:
  1. Reads the base scene XML.
  2. Shortens the proximal phalanx body for thumb/index/middle: changes
     `<body name="*_len_frame" pos="0.05 0 0">` -> `pos="0.025 0 0"`.
  3. Bakes the keyframe morph values into body transforms and removes morph
     joints via `create_rigid_morphology_xml`.
  4. Loads the resulting rigid scene, resets to the requested keyframe.
  5. Settles for N steps holding the keyframe ctrl, then briefly lifts.
  6. Reports closure metrics: # contacts between fingertips and object,
     min fingertip-to-object distance, object lift, and a verdict.

This is NOT a full Phase 1 evaluation. It only checks that the keyframe pose
on a hand with 0.025 m proximal phalanges still produces enough finger-object
contact to serve as a foundational grasp seed for a morphology sweep.

Usage:
  uv run python scripts/smoke_short_hand_keyframes.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from morphohand.tools.morphology_xml import (  # noqa: E402
    create_rigid_morphology_xml,
    extract_morphology_from_qpos,
)


SHORT_LEN_FRAME_POS = "0.025 0 0"
LONG_LEN_FRAME_POS = "0.05 0 0"


@dataclass(frozen=True)
class EvalCase:
    label: str
    scene: Path
    keyframe: str
    object_body: str
    # Per-scene flag: scene already has the short proximal baked in. If True,
    # we skip the shortening edit and only run the morphology freeze.
    already_short: bool = False


# The exact (scene, keyframe, object_body) combos the user named.
EVAL_CASES: list[EvalCase] = [
    EvalCase("cube",                      ROOT_DIR / "assets/mjcf/scene.xml",                          "open",         "cube"),
    EvalCase("prism",                     ROOT_DIR / "assets/mjcf/scene_prism.xml",                    "open",         "prism"),
    EvalCase("screwdriver_medium_flat",   ROOT_DIR / "assets/mjcf/scene_screwdriver_medium_flat.xml",  "open",         "screwdriver_medium"),
    EvalCase("screwdriver_medium_vert",   ROOT_DIR / "assets/mjcf/scene_screwdriver_medium_vertical.xml","open",       "screwdriver_medium"),
    EvalCase("screwdriver_medium_90vert", ROOT_DIR / "assets/mjcf/scene_screwdriver_medium.xml",       "open_90vertical","cube"),
    EvalCase(
        "power_drill_short_proximal",
        ROOT_DIR / "assets/mjcf/scene_power_drill_short_proximal_rigid_capsuletips.xml",
        "open_flat_gripping",
        "power_drill",
        already_short=True,
    ),
]


def shorten_proximal_body_chain(src_xml: Path, dst_xml: Path) -> None:
    """Copy src_xml to dst_xml with each *_len_frame pos="0.05 0 0" rewritten to 0.025."""
    text = src_xml.read_text()
    new_text = text
    replaced = 0
    for finger in ("thumb", "index", "middle"):
        # Robustly find the exact line for this finger's len_frame.
        marker = f'<body name="{finger}_len_frame" pos="{LONG_LEN_FRAME_POS}"'
        if marker in new_text:
            new_text = new_text.replace(
                marker,
                f'<body name="{finger}_len_frame" pos="{SHORT_LEN_FRAME_POS}"',
                1,
            )
            replaced += 1
    if replaced != 3:
        raise RuntimeError(
            f"Expected to shorten 3 len_frame body positions in {src_xml}, only matched {replaced}"
        )
    dst_xml.write_text(new_text)


def extract_keyframe_morphology(scene_xml: Path, keyframe_name: str) -> tuple:
    """Read morphology values (thumb_x/y/len + index/middle) from a keyframe qpos."""
    root = ET.parse(scene_xml).getroot()
    keyframe_elem = root.find("keyframe")
    if keyframe_elem is None:
        return None
    for key in keyframe_elem.findall("key"):
        if key.get("name") != keyframe_name:
            continue
        qpos_raw = key.get("qpos") or ""
        qpos = [float(v) for v in qpos_raw.replace("\n", " ").split()]
        if len(qpos) < 31:
            return None
        return extract_morphology_from_qpos(qpos, has_scene_prefix=True)
    return None


def build_short_rigid_scene(case: EvalCase, workdir: Path) -> Path:
    """Produce a rigid scene with short proximal phalanges and morph values baked."""
    rigid_path = workdir / f"{case.label}_short.xml"

    if case.already_short:
        # Scene is already morph-baked + short. Just copy.
        shutil.copyfile(case.scene, rigid_path)
        return rigid_path

    # Step 1: shorten body chain by editing XML text.
    shortened = workdir / f"{case.label}_shortened_src.xml"
    shorten_proximal_body_chain(case.scene, shortened)

    # Step 2: extract morphology from the requested keyframe and bake.
    morph = extract_keyframe_morphology(shortened, case.keyframe)
    if morph is None:
        # Some scenes (cube/prism/medium-flat) may have keyframe qpos without
        # the morph block; fall back to zero morphology.
        from morphohand.tools.morphology_xml import MorphologyValues
        morph = MorphologyValues(
            thumb_x=0.0, thumb_y=0.0, thumb_len=0.0,
            index_x=0.0, index_y=0.0, index_len=0.0,
            middle_x=0.0, middle_y=0.0, middle_len=0.0,
        )

    create_rigid_morphology_xml(
        base_xml_path=shortened,
        morphology=morph,
        output_xml_path=rigid_path,
        model_name=rigid_path.stem,
    )
    return rigid_path


def _body_geom_ids(model: mujoco.MjModel, body_id: int) -> set[int]:
    geom_ids = set()
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == body_id:
            geom_ids.add(g)
    return geom_ids


def _finger_chain_geom_ids(model: mujoco.MjModel) -> dict[str, set[int]]:
    """Collect ALL hand geoms for each finger chain (mcp, len, pip, tip frames)."""
    chains: dict[str, set[int]] = {f: set() for f in ("thumb", "index", "middle")}
    for finger in chains:
        for sub in ("mcp_frame", "len_frame", "pip_frame", "tip"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{finger}_{sub}")
            if bid >= 0:
                chains[finger] |= _body_geom_ids(model, bid)
    return chains


def _fingertip_body_ids(model: mujoco.MjModel) -> dict[str, int]:
    return {
        finger: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{finger}_tip")
        for finger in ("thumb", "index", "middle")
    }


def _fingertip_geom_ids(model: mujoco.MjModel, finger_body_ids: dict[str, int]) -> dict[str, set[int]]:
    return {f: _body_geom_ids(model, bid) for f, bid in finger_body_ids.items()}


def evaluate_case(case: EvalCase, rigid_xml: Path, settle_steps: int = 200, lift_z: float = 0.05, lift_steps: int = 200) -> dict:
    """Load rigid scene at keyframe, settle, lift, report closure metrics."""
    model = mujoco.MjModel.from_xml_path(str(rigid_xml))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, case.keyframe)
    if key_id < 0:
        return {"label": case.label, "error": f"keyframe '{case.keyframe}' not found"}

    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, case.object_body)
    if object_body_id < 0:
        return {"label": case.label, "error": f"object body '{case.object_body}' not found"}

    object_geom_ids = _body_geom_ids(model, object_body_id)
    # Count contact from the entire finger chain (mcp / mid / pip / tip),
    # not only the tip cap — drill grasp closes via the middle phalanx.
    finger_geom_ids = _finger_chain_geom_ids(model)
    fingertip_body_ids = _fingertip_body_ids(model)
    fingertip_geom_ids = _fingertip_geom_ids(model, fingertip_body_ids)

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # Determine palm pz actuator (for lifting).
    palm_pz_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_palm_pz")
    palm_pz_qid = -1
    if palm_pz_act >= 0:
        palm_pz_jid = model.actuator_trnid[palm_pz_act, 0]
        palm_pz_qid = model.jnt_qposadr[palm_pz_jid]

    settle_pz_target = data.ctrl[palm_pz_act] if palm_pz_act >= 0 else 0.0

    # SETTLE: hold ctrl at keyframe values.
    initial_ctrl = data.ctrl.copy()
    for _ in range(settle_steps):
        data.ctrl[:] = initial_ctrl
        mujoco.mj_step(model, data)

    # Counts at settle.
    fingertip_contact_per_finger = {f: 0 for f in finger_geom_ids}
    fingertip_total_contacts = 0
    for ci in range(data.ncon):
        c = data.contact[ci]
        for finger, gids in finger_geom_ids.items():
            if (c.geom1 in gids and c.geom2 in object_geom_ids) or \
               (c.geom2 in gids and c.geom1 in object_geom_ids):
                fingertip_contact_per_finger[finger] += 1
                fingertip_total_contacts += 1

    # Min finger-to-object surface distance via mj_geomDistance. Use the
    # whole chain since contact may be on a non-tip phalanx.
    min_dists = {}
    for finger, gids in finger_geom_ids.items():
        best = float("inf")
        for fg in gids:
            for og in object_geom_ids:
                d = mujoco.mj_geomDistance(model, data, fg, og, 0.5, np.zeros(6, dtype=np.float64))
                if d < best:
                    best = d
        min_dists[finger] = best

    obj_z_after_settle = float(data.xpos[object_body_id, 2])

    # LIFT: ramp palm pz target up by lift_z over lift_steps.
    obj_z_after_lift = obj_z_after_settle
    fingers_in_contact_during_lift = {f: 0 for f in finger_geom_ids}
    if palm_pz_act >= 0:
        for step in range(lift_steps):
            frac = (step + 1) / lift_steps
            data.ctrl[:] = initial_ctrl
            data.ctrl[palm_pz_act] = settle_pz_target + lift_z * frac
            mujoco.mj_step(model, data)
            # tally per-step fingertip contact
            for ci in range(data.ncon):
                c = data.contact[ci]
                for finger, gids in finger_geom_ids.items():
                    if (c.geom1 in gids and c.geom2 in object_geom_ids) or \
                       (c.geom2 in gids and c.geom1 in object_geom_ids):
                        fingers_in_contact_during_lift[finger] += 1
        obj_z_after_lift = float(data.xpos[object_body_id, 2])

    lift_actual = obj_z_after_lift - obj_z_after_settle

    fingers_touching_at_settle = sum(1 for v in fingertip_contact_per_finger.values() if v > 0)
    persistence_during_lift = {
        f: fingers_in_contact_during_lift[f] / max(lift_steps, 1) for f in finger_geom_ids
    }

    verdict_parts = []
    if fingers_touching_at_settle >= 2 and lift_actual >= 0.5 * lift_z:
        verdict = "OK"
    elif fingers_touching_at_settle >= 2:
        verdict = "WEAK"
        verdict_parts.append(f"lift_actual={lift_actual:.3f} target={lift_z:.3f}")
    else:
        verdict = "FAIL"
        verdict_parts.append(f"only {fingers_touching_at_settle} fingers touching")

    return {
        "label": case.label,
        "scene": str(case.scene.relative_to(ROOT_DIR)),
        "keyframe": case.keyframe,
        "object_body": case.object_body,
        "fingertip_contacts_at_settle": fingertip_contact_per_finger,
        "fingers_touching_at_settle": fingers_touching_at_settle,
        "min_dist": min_dists,
        "obj_z_after_settle": obj_z_after_settle,
        "obj_z_after_lift": obj_z_after_lift,
        "lift_actual": lift_actual,
        "persistence_during_lift": persistence_during_lift,
        "verdict": verdict,
        "verdict_notes": verdict_parts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=ROOT_DIR / "results/smoke_short_hand_keyframes")
    parser.add_argument("--settle-steps", type=int, default=200)
    parser.add_argument("--lift-z", type=float, default=0.05)
    parser.add_argument("--lift-steps", type=int, default=200)
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)

    print(f"{'label':30s} {'verdict':6s} {'fingers':7s} {'lift':>7s} "
          f"{'t.dist':>7s} {'i.dist':>7s} {'m.dist':>7s}  notes")
    print("-" * 100)
    results = []
    for case in EVAL_CASES:
        try:
            rigid = build_short_rigid_scene(case, args.workdir)
            r = evaluate_case(case, rigid,
                              settle_steps=args.settle_steps,
                              lift_z=args.lift_z,
                              lift_steps=args.lift_steps)
        except Exception as e:  # pragma: no cover
            print(f"{case.label:30s} ERROR  {type(e).__name__}: {e}")
            results.append({"label": case.label, "error": str(e)})
            continue
        if "error" in r:
            print(f"{case.label:30s} ERROR  {r['error']}")
            results.append(r)
            continue
        d = r["min_dist"]
        notes = "; ".join(r["verdict_notes"])
        print(
            f"{r['label']:30s} {r['verdict']:6s} "
            f"{r['fingers_touching_at_settle']}/3     "
            f"{r['lift_actual']:+7.3f} "
            f"{d['thumb']:+7.3f} {d['index']:+7.3f} {d['middle']:+7.3f}  {notes}"
        )
        results.append(r)

    import json
    out_json = args.workdir / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
