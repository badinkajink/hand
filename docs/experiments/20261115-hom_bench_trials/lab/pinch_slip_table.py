"""The pick-up by controlled slip of 2026-11-16 15:43-15:47, step by step (pinch_pickup.py's telemetry streams): each
step's commanded and measured PIPs of the thumb and the middle, their difference, the servos' loads (2 thumb PIP, 8
middle PIP, 1 and 7 their MCPs), the pads' field change (uT) and the object's tilt, medians over the step's last 0.4 s.
The slip shows the gears holding the grip with the motor's load at zero, and every yield a servo reversal (README,
"State on 2026-11-16 15:51").

    /usr/bin/python3 pinch_slip_table.py [OUT.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data/20261116-pinch_pickup"
PLAN = HERE.parent / "plans/20261116-pinch_pickup_thumb_middle_plan.json"
STREAMS = {"grasp": "20261116-154328_pinch_grasp_telemetry.jsonl", "slip": "20261116-154534_pinch_slip_telemetry.jsonl"}
GRIP_SQUEEZE = 8.0


def rows(path):
    return [json.loads(x) for x in open(path) if x.strip()]


def table(recs, touch, label):
    segs, cur = {}, None
    for r in recs:
        for m in r.get("mark", []):
            cur = m
        if cur is not None:
            segs.setdefault(cur, []).append(r)
    print(f"== {label}")
    print(f"{'step':>14} {'cmd_th':>7} {'pip_th':>7} {'e_th':>6} {'cmd_mi':>7} {'pip_mi':>7} {'e_mi':>6} "
          f"{'ld_th2':>7} {'ld_mi8':>7} {'ld_th1':>7} {'ld_mi7':>7} {'dB_th':>6} {'dB_mi':>6} {'tilt':>6}")
    out = []
    for name, rs in segs.items():
        tail = rs[-8:]

        def med(key):
            v = [key(r) for r in tail if key(r) is not None]
            return float(np.median(v)) if v else float("nan")
        sq = (float(name.split()[1]) if " " in name else GRIP_SQUEEZE) if name.startswith("squeeze") else None
        cth = touch["thumb"]["pip"] + (sq or 0.0)
        cmi = touch["middle"]["pip"] + (sq or 0.0)
        pth = med(lambda r: (r.get("joints") or {}).get("thumb", {}).get("pip"))
        pmi = med(lambda r: (r.get("joints") or {}).get("middle", {}).get("pip"))

        def ld(i):
            return med(lambda r: (r.get("load") or {}).get(str(i)))
        dth = med(lambda r: (r.get("pads_dB") or {}).get("thumb"))
        dmi = med(lambda r: (r.get("pads_dB") or {}).get("middle"))
        tilt = med(lambda r: (r.get("obj") or {}).get("tilt"))
        print(f"{name:>14} {cth:7.2f} {pth:7.2f} {cth - pth:6.2f} {cmi:7.2f} {pmi:7.2f} {cmi - pmi:6.2f} "
              f"{ld(2):7.0f} {ld(8):7.0f} {ld(1):7.0f} {ld(7):7.0f} {dth:6.1f} {dmi:6.1f} {tilt:6.1f}")
        out.append(dict(step=name, squeeze=sq, cmd_th=cth, pip_th=pth, cmd_mi=cmi, pip_mi=pmi, load_th_pip=ld(2),
                        load_mi_pip=ld(8), load_th_mcp=ld(1), load_mi_mcp=ld(7), dB_th=dth, dB_mi=dmi, tilt=tilt))
    return out


def main():
    touch = {p["name"]: p["joints"] for p in json.loads(PLAN.read_text())["poses"]}["touch"]
    res = {k: table(rows(DATA / f), touch, k) for k, f in STREAMS.items()}
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
