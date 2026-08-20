"""Can the policy COMMAND the behaviour you are about to pay it for?

A residual policy's action is a bounded offset around a fixed set-point: the finger targets are
`closed_ctrl + finger_residual_scale * a` with `a` in [-1, 1]. So any behaviour further than
`finger_residual_scale` from the set-point on any single joint is not merely unexplored — it
cannot be expressed, and no reward weight reaches it.

This has now cost three runs. r7 paid a thumb-brace reward for 339 iterations and the term read
0.0000 throughout; r8 repeated it on a hand where the same maneuver demonstrably holds, and read
0.0000 again. Both were read as "PPO will not explore this". The scripted hold they were paying
for sits 1.296 rad from the set-point at `thumb_pip` against a +-0.5 rad budget. The reward table
cannot show you this: an unreachable target and an unattractive one are the same flat zero.

So measure it before launching. Give it the demonstration you want imitated and the set-point the
policy will be centred on, and it reports the per-joint excursion against the budget.

  MUJOCO_GL=egl uv run python scripts/probe_action_budget.py \
    --scene results/phase1/perp_thumb_engage/sp25_manual/frozen_scene.xml \
    --closed-keyframe closed_manual --demo-npz /tmp/chuck_full.npz --last 1200 \
    --residual-scale 0.5

Exit 0 = the demonstration is inside the budget, 1 = it is not (with the joints that overflow).
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

from morphohand.rl.deploy import finger_ctrl_from_keyframe  # noqa: E402

FINGER_JOINTS = (("thumb", "thumb_yaw", "thumb_mcp", "thumb_pip"),
                 ("index", "index_yaw", "index_mcp", "index_pip"),
                 ("middle", "middle_yaw", "middle_mcp", "middle_pip"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path, help="the FROZEN scene the run uses")
    ap.add_argument("--closed-keyframe", default="closed",
                    help="keyframe the policy's residual is centred on (--closed-ctrl-from-keyframe)")
    ap.add_argument("--demo-npz", type=Path, default=None,
                    help="rollout with a `qpos` array (scripts/probe_perp_thumb_engage.py --save-npz)")
    ap.add_argument("--demo-keyframe", default=None,
                    help="alternative to --demo-npz: a keyframe holding the target pose")
    ap.add_argument("--last", type=int, default=0,
                    help="use only the last N steps of the demo (the sustained hold, not the "
                         "approach — the first contact is typically far closer than the hold)")
    ap.add_argument("--residual-scale", type=float, default=0.5,
                    help="finger_residual_scale the run will train with")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    adr = {j: model.jnt_qposadr[model.joint(j).id]
           for f in FINGER_JOINTS for j in f[1:]}
    centre = np.asarray(finger_ctrl_from_keyframe(args.scene, args.closed_keyframe)).reshape(3, 3)

    if args.demo_npz is not None:
        q = np.load(args.demo_npz)["qpos"]
        q = q[-args.last:] if args.last else q
        target = {j: float(q[:, adr[j]].mean()) for j in adr}
        source = f"{args.demo_npz.name} (last {args.last or len(q)} steps)"
    elif args.demo_keyframe:
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key(args.demo_keyframe).id)
        mujoco.mj_forward(model, data)
        target = {j: float(data.qpos[adr[j]]) for j in adr}
        source = f"keyframe {args.demo_keyframe}"
    else:
        print("[budget] need --demo-npz or --demo-keyframe", file=sys.stderr)
        return 2

    print(f"[budget] scene    {args.scene.name}")
    print(f"[budget] centred  on '{args.closed_keyframe}'   target from {source}")
    print(f"[budget] residual +-{args.residual_scale:.2f} rad per joint\n")
    print(f"{'finger':8s} {'joint':13s} {'set-point':>10s} {'target':>9s} {'excursion':>10s}  headroom")

    over = []
    worst = 0.0
    for i, (finger, *joints) in enumerate(FINGER_JOINTS):
        for k, j in enumerate(joints):
            delta = target[j] - centre[i, k]
            worst = max(worst, abs(delta))
            head = args.residual_scale - abs(delta)
            flag = "OK" if head >= 0 else f"OVER by {-head:.2f}"
            if head < 0:
                over.append((j, abs(delta)))
            print(f"{finger:8s} {j:13s} {centre[i, k]:10.3f} {target[j]:9.3f} "
                  f"{delta:+10.3f}  {head:+6.3f} {flag}")

    print()
    if not over:
        print(f"[budget] REACHABLE — worst excursion {worst:.3f} rad inside "
              f"+-{args.residual_scale:.2f}")
        return 0
    names = ", ".join(f"{j} ({d:.3f} rad)" for j, d in sorted(over, key=lambda t: -t[1]))
    print(f"[budget] UNREACHABLE — {len(over)} joint(s) outside the budget: {names}")
    print(f"[budget] the smallest residual scale that covers this demonstration is "
          f"{worst:.2f} rad. Below it, any reward for this behaviour reads a flat 0.0000 and "
          f"that zero says nothing about whether the behaviour is worth learning.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
