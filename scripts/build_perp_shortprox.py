"""Build the 25 mm-proximal ("short phalanx") version of the opposed-pair hand.

Two things have to happen together, and doing only the first produces a hand that cannot touch
the object:

  1. Shorten the proximal phalanx — capsule and kinematics both (`Scene.set_proximal_length`).
     Every finger is then ~25 mm shorter and its whole reach shell moves in with it.
  2. Re-seat and re-pose. The long hand's grasp keyframe is stated in JOINT space; on a shorter
     finger those same angles put the tip 25 mm short of the shaft. So the palm drops by the
     length that was removed, and each fingertip is IK'd to the WORLD position it occupies in
     the long hand's keyframe (the standing rule: retarget keyframes in world space, never in
     joint space).

The result is written as a sibling scene with the same keyframe name, so every probe and
launcher that takes `--scene` works on it unchanged.

Run:
  MUJOCO_GL=egl uv run python scripts/build_perp_shortprox.py \
    --scene assets/mjcf/perp/scenes/scene_screwdriver_medium_perp.xml --keyframe open_manual
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphohand.studies.scene_mutate import Scene  # noqa: E402
from morphohand.tools.keyframe_ik import (  # noqa: E402
    FINGERS, TIPS, actuator_ctrl_from_qpos, ik_finger, inject_keyframe,
)

MOUNTS = {f: f"{f}_mount" for f in FINGERS}


def reach_shell(model, data, finger: str) -> tuple[float, float]:
    saved = data.qpos.copy()
    mount = data.body(MOUNTS[finger]).xpos.copy()
    jm, jp = model.joint(FINGERS[finger][1]), model.joint(FINGERS[finger][2])
    dists = []
    for a in np.linspace(*model.jnt_range[jm.id], 20):
        for b in np.linspace(*model.jnt_range[jp.id], 20):
            data.qpos[model.jnt_qposadr[jm.id]] = a
            data.qpos[model.jnt_qposadr[jp.id]] = b
            mujoco.mj_forward(model, data)
            dists.append(float(np.linalg.norm(data.body(TIPS[finger]).xpos - mount)))
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    return min(dists), max(dists)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--keyframe", default="open_manual")
    ap.add_argument("--proximal", type=float, default=0.025)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--settle-steps", type=int, default=400,
                    help="steps to settle the SOURCE keyframe before reading its tip targets — "
                         "a hand-typed qpos is not the pose the ctrl actually holds")
    args = ap.parse_args()

    out = args.out or args.scene.with_name(args.scene.stem + "_sp25.xml")

    # --- the long hand's settled fingertip targets -------------------------------------
    src = mujoco.MjModel.from_xml_path(str(args.scene))
    sd = mujoco.MjData(src)
    mujoco.mj_resetDataKeyframe(src, sd, src.key(args.keyframe).id)
    for _ in range(args.settle_steps):
        mujoco.mj_step(src, sd)
    targets = {f: sd.body(TIPS[f]).xpos.copy() for f in FINGERS}
    obj_qpos = sd.qpos[:7].copy()
    old_prox = float(src.body("index_len_frame").pos[0])
    drop = old_prox - args.proximal
    print(f"[src] {args.scene.name} keyframe={args.keyframe} settled {args.settle_steps} steps")
    for f in FINGERS:
        print(f"  {f:7s} tip {np.round(targets[f], 4)}")
    print(f"[src] proximal {old_prox*1000:.1f} mm -> {args.proximal*1000:.1f} mm "
          f"(palm drops {drop*1000:.1f} mm to keep the same tip targets in reach)")

    Scene(args.scene).set_proximal_length(args.proximal).write(out)

    # --- re-pose the short hand onto those same world targets ---------------------------
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(args.keyframe).id)
    data.qpos[:7] = obj_qpos
    data.qpos[model.jnt_qposadr[model.joint("palm_pz").id]] = -drop
    mujoco.mj_forward(model, data)

    ok = True
    for f in FINGERS:
        lo, hi = reach_shell(model, data, f)
        need = float(np.linalg.norm(targets[f] - data.body(MOUNTS[f]).xpos))
        flag = "OK " if lo - 1e-4 <= need <= hi + 1e-4 else "OUT"
        ok &= flag == "OK "
        print(f"  {f:7s} reach shell [{lo:.4f}, {hi:.4f}]  target D {need:.4f}  [{flag}]")

    for f in FINGERS:
        err = ik_finger(model, data, f, targets[f])
        print(f"[ik] {f:7s} -> {np.round(data.body(TIPS[f]).xpos, 4)}  "
              f"residual {err*1000:6.2f} mm")

    inject_keyframe(out, args.keyframe,
                    " ".join(f"{v:.6g}" for v in data.qpos),
                    " ".join(f"{v:.6g}" for v in actuator_ctrl_from_qpos(model, data)))
    print(f"[write] {out}  <key name=\"{args.keyframe}\">"
          f"{'' if ok else '   (WARNING: a target was outside the reach shell)'}")


if __name__ == "__main__":
    main()
