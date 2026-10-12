#!/usr/bin/env python3
"""Compare full and tool-only 1 mm sphere-pad contacts in the D1 held state.

'pads' lets every pad collide with everything; 'padsT' (real_v1_d1_continuation.py) lets the pads touch only the
tool and leaves the fingertip blocks visual. Both grip on the post for 0.8 s, lose the post and run 20 ms in CPU MuJoCo.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import real_v1_d1_continuation as d1  # noqa: E402


def main() -> None:
    import mujoco

    rows = {}
    for contact_model in ("pads", "padsT"):
        scene, meta = d1.make_scene("D1", tip="tpu6", contact_model=contact_model)
        model = mujoco.MjModel.from_xml_path(str(scene))
        t0 = time.perf_counter()
        held_post = d1.grip_state(model, meta, "D1", 1)
        grip_s = time.perf_counter() - t0
        no_post_scene, _ = d1.make_scene("D1", no_post=True, tip="tpu6", contact_model=contact_model)
        no_post = mujoco.MjModel.from_xml_path(str(no_post_scene))
        held = d1.clone_data(no_post, held_post)
        qadr, _ = d1.tool_addresses(no_post)
        initial = d1.contact_summary(no_post, held)
        t0 = time.perf_counter()
        for _ in range(20):
            mujoco.mj_step(no_post, held)
        rollout_s = time.perf_counter() - t0
        rows[contact_model] = {
            "scene": str(no_post_scene.relative_to(ROOT)),
            "n_pads": meta["n_pads"],
            "initial_contact": initial,
            "final_contact": d1.contact_summary(no_post, held),
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
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
