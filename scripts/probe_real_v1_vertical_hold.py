"""Can this hand hold the shaft VERTICAL, and can the policy command that pose?

Two different questions, and the reorient reward reads as a flat zero either way.

  REACHABLE?     Can the three pads ring a vertical shaft at all, with the palm where the
                 handoff holds it? A kinematic question about the morphology.
  COMMANDABLE?   Is that pose within `finger_residual_scale` of the CEM grip the policy is
                 centred on? A residual policy's finger target is `closed_ctrl + scale * a`
                 with a in [-1, 1], so anything further than `scale` on any single joint cannot
                 be expressed and no reward weight reaches it (see
                 scripts/probe_action_budget.py, which cost this program three runs).

WHY IT EXISTS (2026-08-28). All three trained real_v1 designs HOLD the shaft cleanly through the
A->B handoff -- min-z 0.105-0.116 against a 0.05 bar, three fingers in contact 100% of the time --
and NONE of them rotates it: peak alignment cosine 0.015-0.069, i.e. under 4 degrees. Read from
the reward table alone that looks like a morphology verdict. It is not:

    design         index/middle residual   max excursion from the CEM grip
    rv00_wide          15.8 mm             0.928 rad   OVER (thumb/index/middle PIP)
    rv03_narrowy        8.8 mm             0.889 rad   OVER (both PIPs, both MCPs)
    rv04_mid            2.5 mm             0.465 rad   inside the +-0.5 budget

rv00_wide cannot reach the vertical hold at all. rv03_narrowy can reach it but the policy cannot
command it -- 0.89 rad of PIP against a 0.5 rad budget. Only rv04_mid is both reachable and
commandable, which makes it the design to push on and makes `finger_residual_scale` the first
knob to move, not the reward weights.

    MUJOCO_GL=egl uv run python scripts/probe_real_v1_vertical_hold.py \
        --morph-run results/phase1/real_v1/rv04_mid_sp30 --lift 0.10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from morphohand.tools.keyframe_ik import FINGERS, ik_finger  # noqa: E402

PAD_RADIUS = 0.010550
# Azimuths around the standing shaft: thumb opposite, pair at +/-60 deg. A tripod on a circle,
# which is what a vertical cylinder affords -- unlike the horizontal case, where index and middle
# straddle ALONG the shaft and the azimuths are identical.
AZIMUTH = {"thumb": np.pi, "index": np.pi / 3, "middle": -np.pi / 3}


def probe(morph_run: Path, lift: float, residual_scale: float, grip_depth: float,
          object_body: str = "screwdriver_medium") -> dict:
    m = mujoco.MjModel.from_xml_path(str(morph_run / "frozen_scene.xml"))
    d = mujoco.MjData(m)
    closed = np.load(morph_run / "best_rollout.npz")["best_finger_ctrl"]

    mujoco.mj_resetDataKeyframe(m, d, m.key("open_ik").id)
    d.qpos[m.jnt_qposadr[m.joint("palm_pz").id]] += lift    # where the handoff holds it
    mujoco.mj_forward(m, d)
    palm = d.body("palm_pose").xpos.copy()

    radius = float(m.geom_size[[g for g in range(m.ngeom)
                                if m.geom_bodyid[g] == m.body(object_body).id][0], 0])
    r = radius + PAD_RADIUS
    centre_z = palm[2] - grip_depth
    targets = {f: np.array([palm[0] + r * np.cos(AZIMUTH[f]),
                            palm[1] + r * np.sin(AZIMUTH[f]), centre_z]) for f in FINGERS}

    res = {f: ik_finger(m, d, f, targets[f], iters=800) for f in FINGERS}
    excursion, over = {}, {}
    for i, (f, joints) in enumerate(FINGERS.items()):
        for k, j in enumerate(joints):
            q = float(d.qpos[m.jnt_qposadr[m.joint(j).id]])
            e = abs(q - float(closed[i * 3 + k]))
            excursion[j] = round(e, 4)
            if e > residual_scale:
                over[j] = round(e, 4)
    return {
        "run": morph_run.name,
        "tip_residual_mm": {f: round(v * 1000, 2) for f, v in res.items()},
        "max_excursion_rad": round(max(excursion.values()), 4),
        "residual_scale": residual_scale,
        "over_budget": over,
        "reachable": max(res.values()) < 0.005,
        "commandable": not over,
        "excursion_rad": excursion,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--morph-run", type=Path, required=True, action="append",
                    help="repeatable")
    ap.add_argument("--lift", type=float, default=0.10)
    ap.add_argument("--residual-scale", type=float, default=0.5,
                    help="the trainer's finger_residual_scale (a_lift/b_liveA pin it at 0.5)")
    ap.add_argument("--grip-depth", type=float, default=0.0615,
                    help="metres below the mounting plane the shaft's centre is held at")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = {}
    print(f"{'run':22} {'thumb':>7} {'index':>7} {'middle':>7} {'max exc':>9}  verdict")
    for run in args.morph_run:
        r = probe(run, args.lift, args.residual_scale, args.grip_depth)
        rows[run.name] = r
        t = r["tip_residual_mm"]
        verdict = ("reachable + commandable" if r["reachable"] and r["commandable"]
                   else "NOT reachable" if not r["reachable"]
                   else f"OVER BUDGET {sorted(r['over_budget'])}")
        print(f"{run.name:22} {t['thumb']:6.1f}m {t['index']:6.1f}m {t['middle']:6.1f}m "
              f"{r['max_excursion_rad']:8.3f}r  {verdict}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
