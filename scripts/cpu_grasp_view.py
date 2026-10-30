"""Watch the real_v3 / D1 open-loop grasp plan (P0) live in the MuJoCo viewer.

    .venv-cpu/bin/python scripts/cpu_grasp_view.py --diameter 35 --mass 50

The plan replays on a loop (reset every --period seconds). Toggle geom group 2 for the CAD
meshes, group 0 for the MorphIt spheres / pads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from morphohand.cpu_grasp import kin, plan, scene  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diameter", type=float, default=35.0, help="mm")
    ap.add_argument("--mass", type=float, default=50.0, help="g")
    ap.add_argument("--period", type=float, default=plan.T_GRASP_END + 1.0)
    args = ap.parse_args()
    r = args.diameter / 2000.0

    m = scene.build()
    P = plan.make_plan(m, r, K=kin.Kinematics(m))
    scene.set_object(m, r, args.mass / 1000.0)
    d = mujoco.MjData(m)
    ix = scene.indices(m)

    def reset():
        mujoco.mj_resetData(m, d)
        d.qpos[ix["obj_qpos"]:ix["obj_qpos"] + 7] = list(P.obj_pos) + [0.70710678, 0.70710678, 0, 0]
        d.qpos[ix["joint_qpos"]] = P.q(0)
        d.qpos[ix["palm_qpos"]] = P.palm(0)
        mujoco.mj_forward(m, d)

    def control(model, data):
        if data.time > args.period:
            reset()
        data.ctrl[ix["joint_act"]] = P.q(data.time)
        data.ctrl[ix["palm_act"]] = P.palm(data.time)

    reset()
    print(f"D1 hand, {args.diameter:.0f} mm / {args.mass:.0f} g cylinder, plan feasible={P.feasible}")
    mujoco.set_mjcb_control(control)
    mujoco.viewer.launch(m, d)


if __name__ == "__main__":
    main()
