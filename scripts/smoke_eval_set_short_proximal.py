"""Pre-flight closure check on the 7-object short-proximal eval set.

Verifies that, for each (scene, keyframe) combo planned for the run18
morphology sweep, the keyframe actually produces a closing grasp on the
short-proximal hand. Reports: contacts at settle, fingers touching object,
lift achieved over a 5 cm palm lift.

Run after `build_short_proximal_keyframes.py` has authored the
`open_short` keyframes on the new scenes.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvalCase:
    label: str
    scene: Path
    keyframe: str
    object_body: str


EVAL_SET: list[EvalCase] = [
    EvalCase("cube",                       ROOT / "assets/mjcf/scene_cube_short_proximal.xml",                       "open_short_manual",          "cube"),
    EvalCase("prism",                      ROOT / "assets/mjcf/scene_prism_short_proximal.xml",                      "open_short_manual",          "prism"),
    EvalCase("power_drill",                ROOT / "assets/mjcf/scene_power_drill_short_proximal.xml",                "open_flat_gripping",         "power_drill"),
    EvalCase("screwdriver_medium_flat",    ROOT / "assets/mjcf/scene_screwdriver_medium_flat_short_proximal.xml",    "open_short_manual",          "screwdriver_medium"),
    EvalCase("screwdriver_medium_vertical",ROOT / "assets/mjcf/scene_screwdriver_medium_vertical_short_proximal.xml","open_short_manual",          "screwdriver_medium"),
    EvalCase("screwdriver_medium_90vert",  ROOT / "assets/mjcf/scene_screwdriver_medium_short_proximal.xml",         "open_90vertical_manual",     "cube"),
    EvalCase("screwdriver_small_flat",     ROOT / "assets/mjcf/scene_screwdriver_small_flat_short_proximal.xml",     "open_short_manual",          "screwdriver_small"),
]


def _body_geom_ids(model: mujoco.MjModel, body_id: int) -> set[int]:
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id}


def _finger_chain_geom_ids(model: mujoco.MjModel) -> dict[str, set[int]]:
    chains: dict[str, set[int]] = {f: set() for f in ("thumb", "index", "middle")}
    for finger in chains:
        for sub in ("mcp_frame", "len_frame", "pip_frame", "tip"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{finger}_{sub}")
            if bid >= 0:
                chains[finger] |= _body_geom_ids(model, bid)
    return chains


def evaluate(case: EvalCase, settle_steps: int = 200, lift_z: float = 0.05, lift_steps: int = 200) -> dict:
    try:
        model = mujoco.MjModel.from_xml_path(str(case.scene))
    except Exception as e:
        return {"label": case.label, "error": f"load failed: {e}"}
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, case.keyframe)
    if kid < 0:
        return {"label": case.label, "error": f"keyframe '{case.keyframe}' not found"}
    object_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, case.object_body)
    if object_bid < 0:
        return {"label": case.label, "error": f"object body '{case.object_body}' not found"}

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)
    ncon_reset = int(data.ncon)

    object_gids = _body_geom_ids(model, object_bid)
    finger_gids = _finger_chain_geom_ids(model)
    palm_pz = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_palm_pz")

    initial_ctrl = data.ctrl.copy()
    for _ in range(settle_steps):
        data.ctrl[:] = initial_ctrl
        mujoco.mj_step(model, data)

    touch = {f: 0 for f in finger_gids}
    for ci in range(data.ncon):
        c = data.contact[ci]
        for f, gs in finger_gids.items():
            if (c.geom1 in gs and c.geom2 in object_gids) or \
               (c.geom2 in gs and c.geom1 in object_gids):
                touch[f] += 1
    fingers_touching = sum(1 for v in touch.values() if v > 0)
    obj_z0 = float(data.xpos[object_bid, 2])

    for s in range(lift_steps):
        data.ctrl[:] = initial_ctrl
        if palm_pz >= 0:
            data.ctrl[palm_pz] = initial_ctrl[palm_pz] + lift_z * (s + 1) / lift_steps
        mujoco.mj_step(model, data)
    obj_z1 = float(data.xpos[object_bid, 2])
    lift_actual = obj_z1 - obj_z0

    verdict = "OK" if fingers_touching >= 2 and lift_actual >= 0.5 * lift_z else \
              "WEAK" if fingers_touching >= 1 else "FAIL"
    return {
        "label": case.label,
        "ncon_reset": ncon_reset,
        "fingers_touching": fingers_touching,
        "lift_actual": lift_actual,
        "obj_z0": obj_z0,
        "obj_z1": obj_z1,
        "verdict": verdict,
    }


def main() -> None:
    print(f"{'label':30s} {'verdict':6s} {'ncon0':>5s} {'fingers':>7s} {'lift':>7s}")
    print("-" * 70)
    results = []
    for case in EVAL_SET:
        r = evaluate(case)
        results.append(r)
        if "error" in r:
            print(f"{r['label']:30s} ERROR  {r['error']}")
            continue
        print(f"{r['label']:30s} {r['verdict']:6s} {r['ncon_reset']:>5d} "
              f"{r['fingers_touching']}/3      {r['lift_actual']:+7.4f}")
    out = ROOT / "results/smoke_short_hand_keyframes/eval_set.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
