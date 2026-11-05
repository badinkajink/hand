#!/usr/bin/env python3
"""Staircase a free-hanging finger joint up and back down, and see whether it retraces.

    # ON THE CB1, with the control station STOPPED (the servo bus is exclusive)
    python3 scripts/servo_hysteresis_sweep.py --out /tmp/hysteresis.json

WHAT THIS SEPARATES.  `docs/experiments/20260902-servo-compliance` fitted a steady-state
position error proportional to load, 0.0186 deg per load unit at R2 0.94-0.99, and read it as
finite servo stiffness -- MuJoCo `kp`.  Reading the SCS0009's registers gives that a mechanism:
`i_coefficient = 0`, so the position loop is a PD with no integral and any standing load torque
leaves a proportional error forever.

But a load-proportional error is ALSO what BAM's load-dependent friction model predicts
(Duclusaud et al. 2024, arXiv:2410.08650, M3: tau_f = Kv*qd + Kc + Kl*|tau_m - tau_e|), and
terminal holds cannot tell the two apart because both are linear in load with no constant term.

They differ in PATH DEPENDENCE, which is what this measures:

  proportional droop  the error is a function of the CURRENT load only, so the up-sweep and the
                      down-sweep lie on ONE curve
  friction cone       the joint stops anywhere inside the cone, so the two sweeps separate into
                      a LOOP whose width is twice the friction torque

Gravity supplies the load: the finger hangs, and the flexion joints (fe1/fe2) lift its own
distal chain, so commanded angle and load torque move together over the sweep.

WHY ONLY THE FLEXION JOINTS.  `aa` is a ROLL about the hanging finger's long axis, so gravity
exerts no torque about it in this pose and the sweep would be unloaded and uninformative.  The
turn joint the program actually cares about is a yaw, which means THIS BENCH CANNOT IDENTIFY IT
without re-orienting the finger or hanging an offset mass.  Stated here because it is the first
real difference between BAM's single-joint pendulum and this hand.

The distal joints are left torque-enabled and holding, so the segment below the swept joint is
approximately one rigid link -- BAM's pendulum has a genuinely rigid link and a point mass, and
this is the closest that a serial chain of servos gets to it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HOST = Path(__file__).resolve().parents[1] / "src/morphohand/driver/manta/host"
sys.path.insert(0, str(HOST))

from manta_hand.servos import FINGER_JOINTS, ServoBus  # noqa: E402

FINGER_NAME = {0: "thumb", 1: "index", 2: "middle"}
JOINT_NAME = {"aa": "yaw", "fe1": "mcp", "fe2": "pip"}


def sweep_joint(bus: ServoBus, fid: int, joint: str, lo: float, hi: float,
                step: float, dwell: float, speed: int) -> list[dict]:
    servo_id, zero_deg, (jmin, jmax) = FINGER_JOINTS[fid][joint]
    lo, hi = max(lo, jmin + 1.0), min(hi, jmax - 1.0)
    finger, servo = bus.finger(fid), bus.servo(FINGER_JOINTS[fid][joint][0])

    ups = [lo + i * step for i in range(int((hi - lo) / step) + 1)]
    downs = list(reversed(ups))
    out = []
    for direction, seq in (("up", ups), ("down", downs)):
        for target in seq:
            finger.set_joint(joint, target, speed=speed)
            time.sleep(dwell)
            raw = servo.status.position_deg
            achieved = raw - zero_deg
            load = None
            try:
                lf = bus.read_field("load")
                load = None if lf is None else lf.get(servo_id)
            except Exception:
                pass
            out.append({"finger": FINGER_NAME[fid], "joint": JOINT_NAME[joint],
                        "servo_id": servo_id, "direction": direction,
                        "commanded_deg": target, "achieved_deg": achieved,
                        "error_deg": target - achieved, "load": load,
                        "t": time.time()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lo", type=float, default=5.0)
    ap.add_argument("--hi", type=float, default=61.0)
    ap.add_argument("--step", type=float, default=4.0)
    ap.add_argument("--dwell", type=float, default=0.45,
                    help="settle time per set-point. Must be long enough that what is read is "
                         "an equilibrium and not tracking lag")
    ap.add_argument("--speed", type=int, default=80)
    ap.add_argument("--joints", default="fe1,fe2",
                    help="aa is unloaded in the hanging pose; see the module docstring")
    ap.add_argument("--fingers", default="0,1,2")
    a = ap.parse_args()

    fingers = [int(x) for x in a.fingers.split(",") if x.strip()]
    joints = [j.strip() for j in a.joints.split(",") if j.strip()]

    records, meta = [], {"lo": a.lo, "hi": a.hi, "step": a.step, "dwell": a.dwell,
                         "speed": a.speed, "started": time.time()}
    with ServoBus() as bus:
        bus.enable_all()
        time.sleep(0.5)
        # neutral: every joint slightly flexed so nothing rests on a hard stop
        for fid in (0, 1, 2):
            for j in ("fe1", "fe2"):
                try:
                    bus.finger(fid).set_joint(j, a.lo, speed=a.speed)
                except Exception as e:
                    print(f"  neutral {fid}/{j}: {e}")
        time.sleep(1.5)

        for fid in fingers:
            for j in joints:
                print(f"sweeping {FINGER_NAME[fid]}_{JOINT_NAME[j]} "
                      f"({a.lo:.0f} -> {a.hi:.0f} -> {a.lo:.0f} deg) ...", flush=True)
                try:
                    recs = sweep_joint(bus, fid, j, a.lo, a.hi, a.step, a.dwell, a.speed)
                    records += recs
                    up = [r for r in recs if r["direction"] == "up"]
                    dn = [r for r in recs if r["direction"] == "down"]
                    ov = sorted({r["commanded_deg"] for r in up} &
                                {r["commanded_deg"] for r in dn})
                    gaps = []
                    for c in ov:
                        u = next(r["achieved_deg"] for r in up if r["commanded_deg"] == c)
                        d = next(r["achieved_deg"] for r in dn if r["commanded_deg"] == c)
                        gaps.append(d - u)
                    print(f"   mean error {sum(r['error_deg'] for r in recs)/len(recs):+.2f} deg"
                          f" | hysteresis (down-up) mean {sum(gaps)/len(gaps):+.2f}"
                          f" max {max(gaps, key=abs):+.2f} deg", flush=True)
                except Exception as e:
                    print(f"   FAILED: {type(e).__name__}: {e}", flush=True)
                # park back at lo so the next joint starts from the same place
                try:
                    bus.finger(fid).set_joint(j, a.lo, speed=a.speed)
                    time.sleep(0.8)
                except Exception:
                    pass

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"meta": meta, "records": records}, indent=1))
    print(f"\nwrote {a.out}  ({len(records)} set-points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
