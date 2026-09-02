#!/usr/bin/env python3
"""Sweep a flexion joint at several distal-joint angles, so the load varies while the joint does not.

    # ON THE CB1, station stopped
    python3 scripts/servo_load_dependence.py --out /tmp/loaddep.json

WHY.  `servo_hysteresis_sweep.py` answers "is there a friction cone" but a free-hanging finger
is barely loaded -- the pilot on middle_pip saw a mean error of 0.24 deg, against the 10-13 deg
`middle_yaw` gives up during an actual grasp.  A friction term fitted at ~0 load extrapolates to
the grasp regime on faith.

BAM (arXiv:2410.08650) varies the load by bolting different masses to the pendulum: 0.5/1/1.5 kg
over three link lengths.  This hand has no such fixture and nothing to bolt a mass to.  It does
have a distal joint, and folding it changes where the distal mass sits relative to the swept
joint -- so the MOMENT ARM becomes the free variable in place of the mass.  With `fe2` extended
the distal chain reaches away from `fe1` and loads it; with `fe2` folded the same mass sits close
in and loads it much less.  Same joint, same trajectory, different load.

Both quantities are then read against the servo's own `load` register:

  slope    error vs load, pooled over distal angles.  If ONE line fits every distal angle, the
           error is a function of load alone -- a proportional droop, which is MuJoCo `kp`.
  loop     hysteresis width at each distal angle.  If it GROWS with load, the friction is
           load-dependent (BAM M3) and MuJoCo cannot express it, since `frictionloss` is a
           constant.  If it is flat, a constant `frictionloss` is enough.

The two are independent: a hand can have both, and the split decides how much of the shortfall a
feedforward correction can remove and how much only feedback can.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from servo_hysteresis_sweep import FINGER_NAME, sweep_joint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--finger", type=int, default=2)
    ap.add_argument("--joint", default="fe1", help="the swept joint")
    ap.add_argument("--distal", default="fe2", help="the joint used as the variable moment arm")
    ap.add_argument("--distal-angles", default="0,25,50,75")
    ap.add_argument("--lo", type=float, default=5.0)
    ap.add_argument("--hi", type=float, default=57.0)
    ap.add_argument("--step", type=float, default=6.5)
    ap.add_argument("--dwell", type=float, default=0.4)
    ap.add_argument("--speed", type=int, default=80)
    a = ap.parse_args()

    angles = [float(x) for x in a.distal_angles.split(",") if x.strip()]
    _, _, (dmin, dmax) = FINGER_JOINTS[a.finger][a.distal]
    angles = [x for x in angles if dmin + 1 <= x <= dmax - 1]
    print(f"finger {FINGER_NAME[a.finger]}  sweeping {a.joint} at {a.distal} = {angles}")

    records = []
    with ServoBus() as bus:
        bus.enable_all()
        time.sleep(0.5)
        finger = bus.finger(a.finger)
        for da in angles:
            finger.set_joint(a.distal, da, speed=a.speed)
            time.sleep(1.2)
            print(f"  {a.distal} = {da:.0f} deg ...", flush=True)
            recs = sweep_joint(bus, a.finger, a.joint, a.lo, a.hi, a.step, a.dwell, a.speed)
            for r in recs:
                r["distal_joint"] = a.distal
                r["distal_deg"] = da
            records += recs
            up = [r for r in recs if r["direction"] == "up"]
            dn = [r for r in recs if r["direction"] == "down"]
            ov = sorted({r["commanded_deg"] for r in up} & {r["commanded_deg"] for r in dn})
            gaps = [next(r["achieved_deg"] for r in dn if r["commanded_deg"] == c)
                    - next(r["achieved_deg"] for r in up if r["commanded_deg"] == c) for c in ov]
            loads = [abs(r["load"]) for r in recs if r["load"] is not None]
            print(f"     mean err {sum(r['error_deg'] for r in recs)/len(recs):+.2f} deg"
                  f" | loop {sum(gaps)/len(gaps):+.2f} deg"
                  f" | mean |load| {sum(loads)/len(loads) if loads else float('nan'):.0f}",
                  flush=True)
        finger.set_joint(a.distal, a.lo, speed=a.speed)
        time.sleep(0.8)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"meta": vars(a) | {"angles": angles}, "records": records},
                                indent=1, default=str))
    print(f"\nwrote {a.out}  ({len(records)} set-points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
