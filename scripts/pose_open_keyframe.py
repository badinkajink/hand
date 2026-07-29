"""Author an `open` keyframe by IK-ing each fingertip to an explicit WORLD target.

`retarget_keyframe_ik.py` transfers an EXISTING good keyframe from a baseline scene onto a
new morphology. That assumes the two hands share a grasp geometry. When the topology itself
changes (e.g. the perpendicular/opposed-pair hand) there is no baseline pose to transfer, so
the targets have to be stated directly: "put the index tip here, the middle tip there".

This is that tool. It also reports each finger's reachable radius shell [D_min, D_max] from
its mount, which is the constraint that actually bites — a fingertip target closer than D_min
(pip at its flexion limit) is silently unreachable and the IK just parks the finger nearby.

Run:
  MUJOCO_GL=egl uv run python scripts/pose_open_keyframe.py \
    --scene assets/mjcf/scene_screwdriver_medium_perp.xml \
    --index-tip "0 0.0201 0.0225" --middle-tip "0 -0.0201 0.0225" \
    --thumb-tip "-0.030 0 0.045" --write
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
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from morphohand.tools.keyframe_ik import (  # noqa: E402
    FINGERS, PALM_JOINTS, TIPS, actuator_ctrl_from_qpos, has_joint, ik_finger, inject_keyframe,
)

MOUNTS = {f: f"{f}_mount" for f in FINGERS}


def _vec(raw: str) -> np.ndarray:
    return np.array([float(v) for v in raw.replace(",", " ").split()], dtype=float)


def reach_shell(model: mujoco.MjModel, data: mujoco.MjData, finger: str) -> tuple[float, float]:
    """[min, max] tip distance from the mount, scanning the pip range at fixed mcp.

    Link lengths are read off the model by brute force: sweep pip across its range with mcp
    held, and measure |tip - mount|. mcp only rotates the whole 2-link chain, so the shell is
    pip-determined.
    """
    saved = data.qpos.copy()
    pip = model.joint(FINGERS[finger][2])
    adr = model.jnt_qposadr[pip.id]
    lo, hi = model.jnt_range[pip.id]
    mount = data.body(MOUNTS[finger]).xpos.copy()
    dists = []
    for value in np.linspace(lo, hi, 60):
        data.qpos[adr] = value
        mujoco.mj_forward(model, data)
        dists.append(float(np.linalg.norm(data.body(TIPS[finger]).xpos - mount)))
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    return min(dists), max(dists)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--keyframe", default="open", help="keyframe to seed from and rewrite")
    ap.add_argument("--out-keyframe", default=None, help="defaults to --keyframe")
    ap.add_argument("--thumb-tip", default=None)
    ap.add_argument("--index-tip", default=None)
    ap.add_argument("--middle-tip", default=None)
    ap.add_argument("--yaw", default=None, help="thumb,index,middle yaw to pin before IK")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    out_keyframe = args.out_keyframe or args.keyframe
    targets = {
        "thumb": _vec(args.thumb_tip) if args.thumb_tip else None,
        "index": _vec(args.index_tip) if args.index_tip else None,
        "middle": _vec(args.middle_tip) if args.middle_tip else None,
    }

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(args.keyframe).id)
    mujoco.mj_forward(model, data)

    if args.yaw:
        for finger, value in zip(("thumb", "index", "middle"), _vec(args.yaw)):
            data.qpos[model.jnt_qposadr[model.joint(FINGERS[finger][0]).id]] = value
        mujoco.mj_forward(model, data)

    print(f"[scene] {args.scene}  keyframe={args.keyframe}")
    for finger in FINGERS:
        lo, hi = reach_shell(model, data, finger)
        mount = data.body(MOUNTS[finger]).xpos
        line = f"  {finger:7s} mount {np.round(mount, 4)}  reach shell [{lo:.4f}, {hi:.4f}]"
        target = targets[finger]
        if target is not None:
            need = float(np.linalg.norm(target - mount))
            flag = "OK " if lo - 1e-4 <= need <= hi + 1e-4 else "OUT"
            line += f"  target D {need:.4f} [{flag}]"
        print(line)

    for finger, target in targets.items():
        if target is None:
            continue
        err = ik_finger(model, data, finger, target)
        got = data.body(TIPS[finger]).xpos
        print(f"[ik] {finger:7s} -> {np.round(got, 4)}  residual {err * 1000:6.2f} mm")

    print("[pose] joint angles (yaw/mcp/pip):")
    for finger in FINGERS:
        vals = [float(data.qpos[model.jnt_qposadr[model.joint(j).id]]) for j in FINGERS[finger]]
        print(f"  {finger:7s} " + " ".join(f"{v:+.4f}" for v in vals))

    palm = {j: float(data.qpos[model.jnt_qposadr[model.joint(j).id]])
            for j in PALM_JOINTS if has_joint(model, j)}
    print(f"[pose] palm {palm}")

    if args.write:
        ctrl = actuator_ctrl_from_qpos(model, data)
        inject_keyframe(args.scene, out_keyframe,
                        " ".join(f"{v:.6g}" for v in data.qpos),
                        " ".join(f"{v:.6g}" for v in ctrl))
        print(f"[write] injected <key name=\"{out_keyframe}\"> into {args.scene}")


if __name__ == "__main__":
    main()
