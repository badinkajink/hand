"""A guarded retreat of the hand from the S3 block: --away mm level along the block's tag normal made level (+1: toward
the tag's side, the cameras; -1: toward the taped plate's side), then --up mm straight up (hom_setup.Arm.move, the HEX-E
guarding at --stop-n). The normal comes from the newest approach plan in --plans (the block is clamped; it has not moved
more than 0.02 mm all afternoon). For backing off a press pose before re-planning: a press leaves the finger inside the
planner's 5 mm edge margin.

    /usr/bin/python3 s3_retreat.py --side -1 [--away 10] [--up 80] [--go]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s3_press as S3  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--side", type=int, choices=(-1, 1), required=True,
                    help="+1: the hand is on the tag's side and moves further toward the cameras; -1: the taped side")
    ap.add_argument("--away", type=float, default=10.0)
    ap.add_argument("--up", type=float, default=80.0)
    ap.add_argument("--stop-n", type=float, default=5.0)
    ap.add_argument("--plans", default=str(S3.ROOT / "logs" / time.strftime("%Y%m%d-sr2_s3_press")))
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    plan = sorted(Path(a.plans).glob("*_approach_*.json"), key=lambda p: p.stat().st_mtime)[-1]
    nh = np.array(json.loads(plan.read_text())["nh_B"])
    v = a.side * nh
    print(f"away {a.away:g} mm along {np.round(v, 3).tolist()} (B; from {plan.name}), then up {a.up:g} mm")
    if not a.go:
        return 0
    FT, HS, SL = S3.imports()
    arm = HS.Arm(False, 0.4)
    try:
        arm.connect(True)
        if a.away > 0:
            T = arm.T_BE()
            T[:3, 3] += a.away * v
            why, moved = arm.move(T, 0.004, 0.02, a.stop_n, a.away / 4.0 + 6.0, f"retreat {a.away:g} mm")
            if why:
                raise SystemExit(f"the retreat stopped: {why}")
        if a.up > 0:
            T = arm.T_BE()
            T[2, 3] += a.up
            why, moved = arm.move(T, 0.012, 0.05, a.stop_n, a.up / 12.0 + 6.0, f"up {a.up:g} mm")
            if why:
                raise SystemExit(f"the rise stopped: {why}")
    finally:
        arm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
