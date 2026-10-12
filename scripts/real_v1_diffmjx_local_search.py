#!/usr/bin/env python3
"""Compare local yaw-control search on the real_v1 1 mm sphere-pad D1 grasp.

Over 20 ms from the held state (real_v1_d1_continuation.py), the CPU central-difference direction, 128 random
directions and, with --gradient-actions, the fork's AD directions are each applied as a 0.1 rad offset of the three
yaw targets and scored by the tool's vertical cosine in CPU MuJoCo.
"""
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
import real_v1_d1_continuation as d1  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=D / "20261007-pad_local_search_20.json")
    parser.add_argument("--gradient-actions", action="store_true")
    args = parser.parse_args()
    import mujoco

    scene, meta, model, held = d1.held_scene("tpu6", "padsT", 1)
    qadr, _ = d1.tool_addresses(model)
    yaw_ids = np.array([model.actuator(f"a_{finger}_yaw").id for finger in d1.FINGERS])

    def evaluate(delta):
        state = d1.clone_data(model, held)
        state.ctrl[yaw_ids] += delta
        for _ in range(20):
            mujoco.mj_step(model, state)
        return {
            "vertical_cos": float(d1.vertical_cos(state.qpos[qadr + 3:qadr + 7])),
            "tool_z_m": float(state.qpos[qadr + 2]),
            "finger_normal_force_N": d1.contact_summary(model, state)["finger_normal_force_N"],
        }

    radius = 0.1  # Euclidean radius across three yaw targets, radians.
    eps = 0.01
    t0 = time.perf_counter()
    fd = d1.central_difference(lambda u: [evaluate(u)["vertical_cos"]], np.zeros(3), np.eye(3), eps)[0]
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
