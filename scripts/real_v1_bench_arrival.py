#!/usr/bin/env python3
"""How much of the commanded turn the hand actually performed, from the CB1's own run logs.

    python3 scripts/real_v1_bench_arrival.py --logs docs/experiments/<session>/logs \
        --out docs/experiments/<session>/commanded_vs_achieved.json

The exported plan is open loop, which means it assumes commanded == achieved.  The station logs
both: the command stream it sent (`kind: "command"`, sim degrees) and the servo telemetry around
it (`phase: "before"` / `"after"`, servo degrees).  Per joint this reports

    achieved fraction = (measured after - measured before) / (commanded end - commanded start)

and the terminal `servo_load`, because the two answer each other: a joint that performs half its
commanded travel while its load sits near the overload trip did not mis-calibrate, it stalled.
Joints asked to move less than --min-travel-deg are skipped -- a ratio over a 1 degree command is
noise.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/morphohand/driver/manta/host"))

from manta_hand.plan import FINGER_ID, SIM_JOINT_TO_SERVO, servo_deg  # noqa: E402

JOINTS = [(f, j) for f in ("thumb", "index", "middle") for j in ("yaw", "mcp", "pip")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, required=True, help="directory of *.jsonl run logs")
    ap.add_argument("--min-travel-deg", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    out = []
    for p in sorted(a.logs.glob("*.jsonl")):
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        cmds = [r for r in rows if r["kind"] == "command"]
        tel = [r for r in rows if r["kind"] == "telemetry" and r["data"].get("servos")]
        before = next((r for r in tel if r["phase"] == "before"), None)
        after = next((r for r in reversed(tel) if r["phase"] == "after"), None)
        summary_path = p.with_name(p.stem + "_SUMMARY.json")
        if not cmds or before is None or after is None or not summary_path.exists():
            print(f"{p.stem[:34]:34s} incomplete log -- skipped")
            continue
        summary = json.loads(summary_path.read_text())
        rec = {"run": p.stem, "design": summary["design"],
               "speed_ratio": summary["settings"]["speed_ratio"],
               "achieved_fraction": {}, "commanded_deg": {},
               "servo_load_end": after["data"].get("servo_load", {})}
        for f, j in JOINTS:
            want = (servo_deg(f, j, cmds[-1]["sim_joint_deg"][f][j])
                    - servo_deg(f, j, cmds[0]["sim_joint_deg"][f][j]))
            if abs(want) < a.min_travel_deg:
                continue
            got = (after["data"]["servos"][str(FINGER_ID[f])][SIM_JOINT_TO_SERVO[j]]
                   - before["data"]["servos"][str(FINGER_ID[f])][SIM_JOINT_TO_SERVO[j]])
            rec["achieved_fraction"][f"{f}_{j}"] = round(got / want, 3)
            rec["commanded_deg"][f"{f}_{j}"] = round(want, 2)
        out.append(rec)

    print(f"{'run':24s} {'design':13s} {'spd':>4s}  "
          + "".join(f"{f[:2]}_{j:<5s}" for f, j in JOINTS))
    print(f"{'':24s} {'':13s} {'':>4s}  fraction of commanded travel measured; "
          f"'.' = commanded under {a.min_travel_deg:g} deg")
    for rec in out:
        line = f"{rec['run'][:24]:24s} {rec['design']:13s} {rec['speed_ratio']:4.2f}  "
        for f, j in JOINTS:
            v = rec["achieved_fraction"].get(f"{f}_{j}")
            line += f"{'.':>8s}" if v is None else f"{v:8.2f}"
        print(line)

    per_design: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in out:
        for k, v in rec["achieved_fraction"].items():
            per_design[rec["design"]][k].append(v)
    print(f"\n{'design':13s} {'runs':>5s}  " + "".join(f"{f[:2]}_{j:<5s}" for f, j in JOINTS))
    for design, joints in sorted(per_design.items()):
        n = max(len(v) for v in joints.values())
        line = f"{design:13s} {n:5d}  "
        for f, j in JOINTS:
            vals = joints.get(f"{f}_{j}")
            line += f"{'.':>8s}" if not vals else f"{statistics.median(vals):8.2f}"
        print(line)

    if a.out:
        a.out.write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {a.out} ({len(out)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
