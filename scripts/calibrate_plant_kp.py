#!/usr/bin/env python3
"""Find the actuator gain and torque ceiling that reproduce the bench's measured shortfall.

    python3 scripts/calibrate_plant_kp.py --plan <plan.json> --out <report.json>

The bench measures, at the terminal hold of rv05_manual_b85 and averaged over three clean runs,
a deficit of +11.68 deg on `middle_yaw` and +7.66 on `thumb_mcp` -- the joint arriving short of
where it was told to go, by an amount proportional to its load.  The simulator ships the finger
actuators at `kp=30` with `forcerange=+-10`, and the question this answers is what it produces
under the same grasp and what gain would make it agree.

Deficit is read as `ctrl - qpos` at the settled hold, which is the same quantity the bench reads
as `commanded - achieved`, so the two are directly comparable without any load-to-N*m conversion
-- that conversion is unknown (the servo's `load` is a PWM-duty proxy) and this method does not
need it.

The object is in the scene and the grip is the plan's own, so the load is generated the way the
bench generates it, by contact, rather than by an assumed torque.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINGERS = ("thumb", "index", "middle")
JOINTS = ("yaw", "mcp", "pip")

# mean over the three clean rv05_manual_b85 runs of 2026-09-02 (d58cf0, dc9b4e, 546fe7)
MEASURED = {"thumb_yaw": -5.73, "thumb_mcp": 7.66, "thumb_pip": 1.38,
            "index_yaw": -2.86, "index_mcp": 0.37, "index_pip": 0.59,
            "middle_yaw": 11.68, "middle_mcp": -1.04, "middle_pip": -1.89}
# the joints that carry real load and so actually constrain the fit
SCORED = ("middle_yaw", "thumb_mcp", "thumb_yaw", "index_yaw")


def simulate(scene: Path, plan: dict, settle_s: float = 2.0) -> dict[str, float]:
    """Run open -> grip -> turn_end and return ctrl - qpos per finger joint at the hold."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    poses = {p["name"]: p["joints"] for p in plan["poses"]}
    aid, qadr = {}, {}
    for f in FINGERS:
        for j in JOINTS:
            aid[(f, j)] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{f}_{j}")
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_{j}")
            qadr[(f, j)] = m.jnt_qposadr[jid]
    mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey else mujoco.mj_resetData(m, d)

    def hold(pose, seconds):
        for f in FINGERS:
            for j in JOINTS:
                d.ctrl[aid[(f, j)]] = np.deg2rad(pose[f][j])
        for _ in range(int(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)

    hold(poses["open"], 0.6)
    hold(poses["grip"], 1.2)
    # ramp to turn_end the way the plan does, then settle
    a, b = poses["grip"], poses["turn_end"]
    steps = 60
    for i in range(steps + 1):
        u = i / steps
        hold({f: {j: a[f][j] + (b[f][j] - a[f][j]) * u for j in JOINTS} for f in FINGERS},
             1.6 / steps)
    hold(b, settle_s)
    return {f"{f}_{j}": float(np.rad2deg(d.ctrl[aid[(f, j)]] - d.qpos[qadr[(f, j)]]))
            for f in FINGERS for j in JOINTS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path,
                    default=ROOT / "docs/experiments/20260902-residual-bench/deploy/rv05_manual_b85_plan.json")
    ap.add_argument("--kps", default="30,16,8,4,2,1,0.5,0.25")
    ap.add_argument("--forcerange", type=float, default=10.0)
    ap.add_argument("--frictionloss", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    plan = json.loads(a.plan.read_text())
    base = Path(plan["meta"]["scene"])
    rows = []
    print(f"plan {a.plan.name}   base scene {base.name}")
    print(f"target deficits (bench, n=3): " +
          ", ".join(f"{k} {MEASURED[k]:+.2f}" for k in SCORED))
    print(f"\n{'kp':>7} " + " ".join(f"{k:>13}" for k in SCORED) + f"{'  MAE':>8}")
    with tempfile.TemporaryDirectory() as td:
        for kp in [float(x) for x in a.kps.split(",")]:
            out = Path(td) / f"kp{kp:g}.xml"
            subprocess.run([sys.executable, str(ROOT / "scripts/apply_measured_plant.py"),
                            "--scene", str(base), "--out", str(out), "--kp", str(kp),
                            "--forcerange", str(a.forcerange),
                            "--frictionloss", str(a.frictionloss)],
                           check=True, capture_output=True)
            try:
                sim = simulate(out, plan)
            except Exception as e:
                print(f"{kp:>7g}  FAILED: {type(e).__name__}: {e}")
                continue
            mae = float(np.mean([abs(sim[k] - MEASURED[k]) for k in SCORED]))
            rows.append({"kp": kp, "sim": sim, "mae": mae})
            print(f"{kp:>7g} " + " ".join(f"{sim[k]:>+13.2f}" for k in SCORED) + f"{mae:>8.2f}")
    if rows:
        best = min(rows, key=lambda r: r["mae"])
        print(f"\nbest kp = {best['kp']:g}  (MAE {best['mae']:.2f} deg over {len(SCORED)} loaded joints)")
        print("  full sim deficits: " +
              ", ".join(f"{k} {v:+.1f}" for k, v in best["sim"].items()))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"measured": MEASURED, "scored": list(SCORED),
                                     "rows": rows}, indent=1))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
