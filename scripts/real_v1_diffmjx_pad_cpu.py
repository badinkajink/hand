#!/usr/bin/env python3
"""Compare full and tool-only 1 mm sphere-pad contacts in the D1 held state."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import real_v1_diffmjx_gate as gate  # noqa: E402


def main() -> None:
    import mujoco

    rows = {}
    for contact_model in ("pads", "padsT"):
        scene, meta = gate.make_scene("D1", "bed", 100.0, tip="tpu6",
                                      contact_model=contact_model)
        model = mujoco.MjModel.from_xml_path(str(scene))
        t0 = time.perf_counter()
        held_post = gate.grip_state(model, meta, "D1", 1)
        grip_s = time.perf_counter() - t0
        no_post_scene, _ = gate.make_scene("D1", "bed", 100.0, no_post=True,
                                            tip="tpu6", contact_model=contact_model)
        no_post = mujoco.MjModel.from_xml_path(str(no_post_scene))
        held = gate.clone_data(no_post, held_post)
        obj = no_post.body(gate.rb.OBJ).id
        qadr = int(no_post.jnt_qposadr[no_post.body_jntadr[obj]])
        initial = gate.contact_summary(no_post, held)
        t0 = time.perf_counter()
        for _ in range(20):
            mujoco.mj_step(no_post, held)
        rollout_s = time.perf_counter() - t0
        rows[contact_model] = {
            "scene": str(no_post_scene.relative_to(ROOT)),
            "n_pads": meta["n_pads"],
            "initial_contact": initial,
            "final_contact": gate.contact_summary(no_post, held),
            "final_tool_qpos": held.qpos[qadr:qadr + 7].tolist(),
            "grip_cpu_s": grip_s,
            "rollout_20_cpu_s": rollout_s,
        }
    a, b = rows["pads"], rows["padsT"]
    out = {
        "mujoco_version": mujoco.__version__,
        "steps": 20, "timestep_s": 0.001,
        "rows": rows,
        "final_tool_qpos_max_abs_error": float(np.max(np.abs(
            np.asarray(a["final_tool_qpos"]) - np.asarray(b["final_tool_qpos"])))),
    }
    path = ROOT / "docs/experiments/20261007-diffmjx/20261007-pad_mask_cpu.json"
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
