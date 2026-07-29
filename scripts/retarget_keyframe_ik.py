"""Retarget the 'about-to-contact' open keyframe across morphologies via fingertip IK.

Thin CLI over morphohand.tools.keyframe_ik (see its docstring for the joint-space-transfer
bug this fixes and the world-space IK method). Reads the baseline keyframe's fingertip
world positions, IKs the target morphology's fingers onto them, and optionally injects the
result as an `open_ik` keyframe (the CEM seed / LerpFinger open pose).

Run:
  uv run --extra rl python scripts/retarget_keyframe_ik.py \
    --base-scene assets/mjcf/baseline/scenes/scene_screwdriver_medium_flat_short_proximal.xml \
    --keyframe open_short_manual --target-scene <generated_morph_scene.xml> \
    [--close-rad 0.15] [--validate]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import mujoco

from morphohand.tools.keyframe_ik import (
    FINGERS, TIPS, actuator_ctrl_from_qpos, has_joint, ik_finger, inject_keyframe, tip_targets,
)

# Back-compat aliases (older scripts imported the underscore names from this module).
_has_joint = has_joint
_inject_keyframe = inject_keyframe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-scene", required=True)
    ap.add_argument("--keyframe", default="open_short_manual")
    ap.add_argument("--target-scene", required=True)
    ap.add_argument("--out-keyframe", default="open_ik")
    ap.add_argument("--close-rad", type=float, default=0.0,
                    help="extra mcp+pip flexion (rad) after IK, to seat tips into contact")
    ap.add_argument("--validate", action="store_true",
                    help="report per-finger tip errors + tip-object distances")
    ap.add_argument("--write-keyframe", action="store_true",
                    help="inject <key name=OUT_KEYFRAME> into the target scene XML (CEM seed)")
    args = ap.parse_args()

    tips, palm, obj_qpos = tip_targets(args.base_scene, args.keyframe)
    print(f"[targets] tip world xyz (from {args.base_scene} @ {args.keyframe}):")
    for f in FINGERS:
        print(f"    {f:7s} {np.round(tips[f], 4)}")

    m = mujoco.MjModel.from_xml_path(args.target_scene)
    d = mujoco.MjData(m)
    # seed from the target scene's own open_short_manual (baseline finger angles) if present
    try:
        mujoco.mj_resetDataKeyframe(m, d, m.key(args.keyframe).id)
    except Exception:
        pass
    # pin palm + object to the baseline keyframe pose (only the hand geometry differs)
    for j, v in palm.items():
        if has_joint(m, j):
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
    d.qpos[:7] = obj_qpos
    mujoco.mj_forward(m, d)

    print("[ik] retargeting fingers ...")
    errs = {}
    for f in FINGERS:
        errs[f] = ik_finger(m, d, f, tips[f])
        print(f"    {f:7s} residual {errs[f]*1000:.2f} mm")

    if args.close_rad > 0:
        for f in FINGERS:
            for jn in (FINGERS[f][1], FINGERS[f][2]):  # mcp, pip
                jid = m.joint(jn).id
                a = m.jnt_qposadr[jid]
                lo, hi = m.jnt_range[jid]
                d.qpos[a] = np.clip(d.qpos[a] + args.close_rad, lo, hi)
        mujoco.mj_forward(m, d)

    # build the finger ctrl vector (position actuators, in actuator order)
    finger_joint_names = [jn for f in FINGERS for jn in FINGERS[f]]
    ctrl = {}
    for jn in finger_joint_names:
        ctrl[jn] = float(d.qpos[m.jnt_qposadr[m.joint(jn).id]])

    print("[ik] retargeted finger joint angles:")
    for f in FINGERS:
        print(f"    {f:7s} yaw/mcp/pip = "
              + " ".join(f"{ctrl[jn]:+.4f}" for jn in FINGERS[f]))

    if args.validate:
        # object surface: tip-to-object-body distance (proxy for contact readiness)
        mujoco.mj_forward(m, d)
        print("[validate] tip -> object-center distance (target obj center "
              f"{np.round(obj_qpos[:3],4)}):")
        for f in FINGERS:
            tp = d.body(TIPS[f]).xpos
            print(f"    {f:7s} tip {np.round(tp,4)}  dist {np.linalg.norm(tp-obj_qpos[:3])*1000:.1f} mm")

    # emit the full qpos + finger ctrl so it can be written as an `open_ik` keyframe
    np.set_printoptions(suppress=True, precision=6)
    print("\n[out] full qpos:\n" + " ".join(f"{v:.6g}" for v in d.qpos))
    print("[out] finger ctrl (yaw/mcp/pip x thumb,index,middle):\n"
          + " ".join(f"{ctrl[jn]:.6g}" for jn in finger_joint_names))

    if args.write_keyframe:
        ctrl_vec = actuator_ctrl_from_qpos(m, d)
        inject_keyframe(Path(args.target_scene), args.out_keyframe,
                        " ".join(f"{v:.6g}" for v in d.qpos),
                        " ".join(f"{v:.6g}" for v in ctrl_vec))
        print(f"[write] injected <key name=\"{args.out_keyframe}\"> into {args.target_scene}")
    return errs


if __name__ == "__main__":
    main()
