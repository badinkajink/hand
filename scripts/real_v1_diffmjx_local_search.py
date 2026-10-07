#!/usr/bin/env python3
"""Compare local yaw-control search on the real_v1 1 mm sphere-pad D1 grasp."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs/experiments/20261007-diffmjx"
sys.path.insert(0, str(ROOT / "scripts"))
import real_v1_diffmjx_gate as gate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=D / "20261007-pad_local_search_20.json")
    parser.add_argument("--gradient-actions", action="store_true")
    args = parser.parse_args()
    import mujoco

    scene, meta = gate.make_scene("D1", "bed", 100.0, no_post=True,
                                  tip="tpu6", contact_model="padsT")
    model = mujoco.MjModel.from_xml_path(str(scene))
    supported, _ = gate.make_scene("D1", "bed", 100.0,
                                    tip="tpu6", contact_model="padsT")
    held = gate.clone_data(model, gate.grip_state(mujoco.MjModel.from_xml_path(str(supported)),
                                                  meta, "D1", 1))
    obj = model.body(gate.rb.OBJ).id
    qadr = int(model.jnt_qposadr[model.body_jntadr[obj]])
    yaw_ids = np.array([model.actuator(f"a_{finger}_yaw").id for finger in gate.rb.FINGERS])

    def evaluate(delta):
        state = gate.clone_data(model, held)
        state.ctrl[yaw_ids] += delta
        for _ in range(20):
            mujoco.mj_step(model, state)
        quat = state.qpos[qadr + 3:qadr + 7]
        summary = gate.contact_summary(model, state)
        return {
            "vertical_cos": float(1 - 2 * (quat[1] ** 2 + quat[2] ** 2)),
            "tool_z_m": float(state.qpos[qadr + 2]),
            "finger_normal_force_N": summary["finger_normal_force_N"],
        }

    radius = 0.1  # Euclidean radius across three yaw targets, radians.
    eps = 0.01
    basis = np.eye(3)
    t0 = time.perf_counter()
    fd = np.array([(evaluate(eps * b)["vertical_cos"] -
                    evaluate(-eps * b)["vertical_cos"]) / (2 * eps) for b in basis])
    fd_s = time.perf_counter() - t0
    fd_direction = radius * fd / np.linalg.norm(fd)
    candidates = {
        "zero": {"delta_rad": [0.0] * 3, **evaluate(np.zeros(3))},
        "cpu_fd": {"delta_rad": fd_direction.tolist(), **evaluate(fd_direction)},
    }
    if args.gradient_actions:
        for name, filename in (
            ("plain_ad", "20261007-official_padsT1_cylinder_jacobian_20.json"),
            ("cfd_ad", "20261007-official_padsT1_cylinder_cfd_jacobian_20.json"),
        ):
            row = json.loads((D / filename).read_text())
            direction = np.asarray(row["local_search"]["ad_direction_rad"])
            candidates[name] = {"delta_rad": direction.tolist(), **evaluate(direction)}
    rng = np.random.default_rng(20261007)
    random = rng.normal(size=(128, 3))
    random *= radius / np.linalg.norm(random, axis=1)[:, None]
    t0 = time.perf_counter()
    random_rows = [evaluate(vector) for vector in random]
    random_s = time.perf_counter() - t0
    random_scores = np.array([row["vertical_cos"] for row in random_rows])
    best = int(np.argmax(random_scores))
    out = {
        "scene": str(scene.relative_to(ROOT)),
        "mujoco_version": mujoco.__version__,
        "n_pads": meta["n_pads"],
        "steps": 20, "timestep_s": float(model.opt.timestep),
        "yaw_target_radius_rad": radius,
        "fd_step_rad": eps,
        "cpu_fd_gradient_per_rad": fd.tolist(),
        "cpu_fd_6_evals_s": fd_s,
        "candidates": candidates,
        "random_count": len(random),
        "random_best": {"delta_rad": random[best].tolist(), **random_rows[best]},
        "random_vertical_cos": random_scores.tolist(),
        "random_median_vertical_cos": float(np.median(random_scores)),
        "random_128_evals_s": random_s,
    }
    path = args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
