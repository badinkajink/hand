"""Gantry versus UR5e, and how stiff the wrist has to be.

Two arms, and the second is the one with a number in it.

  wrist     the same chain on the floating palm and on the arm    does a real arm cost anything?
  stiffness gravity compensation 0..1 on the arm's own links      how much droop is too much?

The stiffness arm is not a study of gravity compensation for its own sake. The menagerie UR5e
has no gravity feedforward, so sweeping how much of it is applied is a way of sweeping PALM
DROOP UNDER LOAD continuously, from 12.7 mm down to 0.35 mm, on an otherwise identical robot.
The answer comes out as millimetres, which is a spec any arm can be checked against.

    uv run --extra rl --extra arm python scripts/real_v1_arm_study.py \
        --out docs/experiments/20260903-real_v1_chain
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RUN = ROOT / "results/phase1/real_v1/rv05_manual_stored"
GRAVCOMP = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)


def _scene(g: float):
    return ((RUN / "arm_scene.xml", RUN / "arm_ik.xml") if g == 1.0
            else (RUN / f"arm_scene_g{g}.xml", RUN / f"arm_ik_g{g}.xml"))


def _build(g: float) -> None:
    sc, ik = _scene(g)
    if sc.exists() and ik.exists():
        return
    subprocess.run([sys.executable, str(ROOT / "scripts/build_real_v1_arm_scene.py"),
                    "--arm-gravcomp", str(g), "--out", str(sc), "--ik-out", str(ik)],
                   check=True, capture_output=True)


def droop(g: float) -> dict:
    """Palm position and orientation error at the top of the lift, holding the hand + tool."""
    import mujoco
    import palm_driver as pd
    sc, ik = _scene(g)
    m = mujoco.MjModel.from_xml_path(str(sc))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("open_ik").id)
    d.ctrl[:] = m.key_ctrl[m.key("open_ik").id]
    palm = pd.make(m, d, ik)
    mujoco.mj_forward(m, d)
    Rc, pc = palm.cmd_pose()
    u0 = palm.read()
    u1 = palm.solve(Rc, pc + np.array([0.0, 0.0, 0.10]))[0]
    for k in range(200):
        palm.write(u0 + (u1 - u0) * (k + 1) / 200)
        mujoco.mj_step(m, d)
    for _ in range(800):
        mujoco.mj_step(m, d)
    Rc, pc = palm.cmd_pose()
    Ra = d.body("palm_pose").xmat.reshape(3, 3)
    pa = d.body("palm_pose").xpos
    return {"gravcomp": g, "droop_mm": round(float(np.linalg.norm(pa - pc)) * 1000, 3),
            "droop_deg": round(float(np.degrees(np.arccos(
                np.clip((np.trace(Ra.T @ Rc) - 1) / 2, -1, 1)))), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--arms", default="wrist,stiffness")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arms = set(args.arms.split(","))
    import probe_real_v1_chain as C
    rows = []

    if "wrist" in arms:
        for tag, ik in (("gantry", None), ("ur5e", RUN / "arm_ik.xml")):
            if ik is not None:
                _build(1.0)
            for s in range(args.reps):
                r = C.chain(RUN, cycles=args.cycles, jitter=0.0005, seed=s, arm_ik=ik)
                r.update(arm="wrist", wrist_tag=tag)
                rows.append(r)
            r = C.chain(RUN, cycles=40, jitter=0.0, seed=0, arm_ik=ik)
            r.update(arm="endurance", wrist_tag=tag)
            rows.append(r)
            print(f"{tag} done")

    if "stiffness" in arms:
        for g in GRAVCOMP:
            _build(g)
            sc, ik = _scene(g)
            dr = droop(g)
            for s in range(args.reps):
                r = C.chain(RUN, cycles=args.cycles, jitter=0.0005, seed=s,
                            arm_ik=ik, scene_path=sc)
                r.update(arm="stiffness", **dr)
                rows.append(r)
            ok = sum(1 for r in rows if r.get("arm") == "stiffness"
                     and r.get("gravcomp") == g and r["ok"])
            print(f"gravcomp {g}: droop {dr['droop_mm']} mm, ok {ok}/{args.reps}")

    (args.out / "arm_study.json").write_text(json.dumps(rows, indent=2))
    print(f"-> {args.out / 'arm_study.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
