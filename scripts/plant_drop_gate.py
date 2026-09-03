#!/usr/bin/env python3
"""Does the corrected plant make the simulator FAIL the way the bench fails?

    python3 scripts/plant_drop_gate.py --trials 24 --out <report.json>

THE GATE.  A residual policy can only learn a disturbance its simulator produces.  Before the
plant correction the simulator retained the shaft on 1.00 of trials for seven of eight transfer
designs, while the bench drops 6 of 77 and shows a further design whose trials are a MIXTURE of
grips that held and grips that slipped.  Training a residual against a simulator that cannot
fail teaches it nothing, and every previous reorienter's indifference to its own observations is
the honest consequence of exactly that.

So this compares, on the same plan and the same spawn perturbations:

  shipped     kp=30, forcerange +-10, no frictionloss, masses at MuJoCo's default density
  corrected   kp calibrated to the bench's measured deficits, a real torque ceiling, the
              measured friction cone, and the builder's masses and centres of mass

and reports retention, final alignment and object height for each.  The corrected plant PASSES
the gate if it drops the shaft on some trials and holds it on others -- a plant that never drops
is as useless for this purpose as one that always does.
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
OBJECT = "screwdriver_medium"


def run_trial(scene: Path, plan: dict, jitter_xy: float, jitter_yaw: float,
              seed: int, traj: Path | None = None) -> dict:
    """`traj` is the plan's own <design>_traj.csv, the per-step schedule the carry produced.

    PREFER IT.  Interpolating the plan JSON's three set-points gives a DIFFERENT path -- the
    same distinction `real_v1_trajectory_clearance.py` scores separately as `chord` and `csv`,
    where a design can clear one and not the other.  The chord happens to work on
    rv05_manual_b85 and drops the shaft on 10 of 10 trials for sv1_w6689_b060 in BOTH plants,
    which reads as a plant verdict and is a path artifact: on the CSV the same hand and the
    same corrected plant retain 4 of 4 at cos 0.928."""

    import mujoco
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    poses = {p["name"]: p["joints"] for p in plan["poses"]}
    aid = {(f, j): mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{f}_{j}")
           for f in FINGERS for j in JOINTS}
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJECT)
    qadr = m.jnt_qposadr[m.body_jntadr[bid]]

    # The plan carries the state it was exported against, and it is NOT the scene keyframe.
    # `replay_base_ctrl` holds palm_pz = 0.1035, the 103.5 mm lift that puts the hand at an
    # object standing on a 100 mm bench post; the keyframe's palm_pz is 0.01. Reset from the
    # keyframe instead and the fingertips sit 83 mm below the shaft, the grip closes on air,
    # and every trial reports "never retained" -- a harness bug that looks exactly like a
    # plant result. Same convention as replay_real_v1_hardware_log.py.
    mujoco.mj_resetData(m, d)
    meta = plan["meta"]
    d.qpos[:] = np.asarray(meta["replay_initial_qpos"], dtype=float)
    d.ctrl[:] = np.asarray(meta["replay_base_ctrl"], dtype=float)
    base_ctrl = d.ctrl.copy()
    mujoco.mj_forward(m, d)
    rng = np.random.default_rng(seed)
    d.qpos[qadr + 0] += rng.normal(0, jitter_xy)
    d.qpos[qadr + 1] += rng.normal(0, jitter_xy)
    if jitter_yaw:
        a = rng.normal(0, jitter_yaw) / 2.0
        q = d.qpos[qadr + 3:qadr + 7].copy()
        dq = np.array([np.cos(a), 0.0, 0.0, np.sin(a)])
        d.qpos[qadr + 3:qadr + 7] = np.array([
            q[0]*dq[0]-q[1]*dq[1]-q[2]*dq[2]-q[3]*dq[3],
            q[0]*dq[1]+q[1]*dq[0]+q[2]*dq[3]-q[3]*dq[2],
            q[0]*dq[2]-q[1]*dq[3]+q[2]*dq[0]+q[3]*dq[1],
            q[0]*dq[3]+q[1]*dq[2]-q[2]*dq[1]+q[3]*dq[0]])

    def hold(pose, seconds):
        d.ctrl[:] = base_ctrl          # keeps the six palm-pose channels where the plan put them
        for f in FINGERS:
            for j in JOINTS:
                d.ctrl[aid[(f, j)]] = np.deg2rad(pose[f][j])
        for _ in range(max(1, int(seconds / m.opt.timestep))):
            mujoco.mj_step(m, d)

    # NO open phase. `replay_base_ctrl`'s finger channels decode to exactly the plan's `grip`
    # pose, so the exported replay state already has the shaft held; commanding `open` first
    # releases it, and a released shaft falls off the 100 mm post and lands UPRIGHT reading
    # cos 1.000 -- the failure probe_real_v1_carry.py's docstring warns about by name.
    hold(poses["grip"], 0.8)
    z_pre = float(d.xpos[bid][2])
    zs = []
    if traj is not None:
        import csv as _csv
        rows = list(_csv.DictReader(open(traj)))
        span = float(rows[-1]["t_s"]) or 1.6
        for r in rows:
            hold({f: {j: float(r[f"{f}_{j}_deg"]) for j in JOINTS} for f in FINGERS},
                 span / len(rows))
            zs.append(float(d.xpos[bid][2]))
        hold({f: {j: float(rows[-1][f"{f}_{j}_deg"]) for j in JOINTS} for f in FINGERS}, 1.5)
    else:
        a_, b_ = poses["grip"], poses["turn_end"]
        for i in range(61):
            u = i / 60
            hold({f: {j: a_[f][j] + (b_[f][j] - a_[f][j]) * u for j in JOINTS} for f in FINGERS},
                 1.6 / 60)
            zs.append(float(d.xpos[bid][2]))
        hold(b_, 1.5)
    zs.append(float(d.xpos[bid][2]))

    R = d.xmat[bid].reshape(3, 3)
    cos = float(abs(R[:, 2] @ np.array([0.0, 0.0, 1.0])))
    z_end = float(d.xpos[bid][2])
    return {"cos": cos, "z_end": z_end, "z_min_turn": float(min(zs)), "z_pre": z_pre,
            "dropped": bool(z_end < z_pre - 0.020)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path,
                    default=ROOT / "docs/experiments/20260902-residual-bench/deploy/rv05_manual_b85_plan.json")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--jitter-xy", type=float, default=0.004, help="m, 1 sigma")
    ap.add_argument("--jitter-yaw", type=float, default=0.09, help="rad, 1 sigma")
    ap.add_argument("--kp-corrected", type=float, default=0.5)
    ap.add_argument("--forcerange-corrected", type=float, default=0.35)
    ap.add_argument("--frictionloss-corrected", type=float, default=0.0035)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    plan = json.loads(a.plan.read_text())
    base = Path(plan["meta"]["scene"])
    results = {}
    with tempfile.TemporaryDirectory() as td:
        variants = {
            "shipped": ["--kp", "30", "--forcerange", "10", "--frictionloss", "0", "--no-mass"],
            "corrected": ["--kp", str(a.kp_corrected),
                          "--forcerange", str(a.forcerange_corrected),
                          "--frictionloss", str(a.frictionloss_corrected)],
        }
        for name, flags in variants.items():
            scene = Path(td) / f"{name}.xml"
            subprocess.run([sys.executable, str(ROOT / "scripts/apply_measured_plant.py"),
                            "--scene", str(base), "--out", str(scene)] + flags,
                           check=True, capture_output=True)
            trials = []
            for s in range(a.trials):
                try:
                    trials.append(run_trial(scene, plan, a.jitter_xy, a.jitter_yaw, 1000 + s))
                except Exception as e:
                    print(f"  {name} seed {s}: {type(e).__name__}: {e}")
            held = [t for t in trials if not t["dropped"]]
            results[name] = {"n": len(trials), "held": len(held),
                             "retention": len(held) / max(1, len(trials)),
                             "cos_held_mean": float(np.mean([t["cos"] for t in held])) if held else None,
                             "cos_all_mean": float(np.mean([t["cos"] for t in trials])),
                             "trials": trials}
            r = results[name]
            print(f"{name:<10} retention {r['retention']:.2f} ({r['held']}/{r['n']})   "
                  f"cos|held {r['cos_held_mean'] if r['cos_held_mean'] is None else round(r['cos_held_mean'],3)}   "
                  f"cos|all {r['cos_all_mean']:.3f}")

    s, c = results.get("shipped", {}), results.get("corrected", {})
    print()
    if c.get("retention") is None:
        print("GATE: could not evaluate")
    elif c["retention"] in (0.0, 1.0):
        print(f"GATE FAILED: corrected plant retention is {c['retention']:.2f} -- it is "
              f"{'never' if c['retention']==0 else 'always'} failing, so a residual has "
              f"{'nothing to hold' if c['retention']==0 else 'nothing to learn'}.")
    else:
        print(f"GATE PASSED: corrected plant drops on {1-c['retention']:.0%} of trials and holds "
              f"on {c['retention']:.0%} (shipped: {s.get('retention', float('nan')):.0%} held). "
              f"There is a disturbance to learn.")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"args": vars(a), "results": results}, indent=1, default=str))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
