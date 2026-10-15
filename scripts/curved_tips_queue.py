#!/usr/bin/env python3
"""Job queues of the curved-tip task screen (docs/handoff/20261015-curved_tips.md, step 5), the 2026-10-14 stages
(docs/notes/20261014-tip_codesign_plan.md, steps 7-9) for the curved candidates, the as-built block and the block
finalist (3, 6), every tip with builder-v2 pads:

  screen   stages 1 and 2 together (8 tips): per condition (mu 1.0, 0.7 and 0.5 on the servo-limited plant; at mu 0.7
           also gain 0.5 and torque limit 0.12 N m) the scale study's four CAD-hand cells (HOM seeds 0-2; the hold test
           at mu 0.7 and 0.5) and HOM on D1-D8 (seeds 0-4); the mu 0.7 servo jobs first, which give J1
  stage3   fresh placements for the finalists (screen/summary.json): cells seeds 3-5 and D1-D8 seeds 5-9 at mu 0.7 and
           0.5 and the two servo conditions

Job lines (logs/20261015-curved_tips/screen/worker.sh):
  scale <tip> <mu> <plant> <what> <seeds, comma-separated> <out dir>
  hom <tip> <mu> <plant> <seeds, comma-separated> <out file>

    python3 scripts/curved_tips_queue.py screen
"""
import argparse
import json
from pathlib import Path

import curved_tips_common as C

SERVO = C.SERVO
CONDS = [("0.7", SERVO), ("1.0", SERVO), ("0.5", SERVO), ("0.7", "kp0.5_kv0_fr0.18_fl0_dp0.0144"),
         ("0.7", "kp0.72_kv0_fr0.12_fl0_dp0.0144")]
TIPS = C.CURVED + C.BLOCKS
Q = C.LOGS / "screen" / "queue.txt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["screen", "stage3"])
    a = ap.parse_args()
    S = C.D / "screen"
    jobs = []
    if a.what == "screen":
        for mu, pl in CONDS:                         # condition by condition: J1's (the first) completes first
            for k in TIPS:
                what = "hold,turn" if (mu, pl) in (("0.7", SERVO), ("0.5", SERVO)) else "turn"
                jobs.append(f"scale {k} {mu} {pl} {what} 0,1,2 {S}/scale/{k}_mu{mu}_{pl}")
                jobs.append(f"hom {k} {mu} {pl} 0,1,2,3,4 {S}/hom.jsonl")
    else:
        fin = json.loads((S / "summary.json").read_text())["stage3_tips"]
        F = C.D / "final"
        for k in fin:
            for mu, pl in CONDS:
                if (mu, pl) == ("1.0", SERVO):
                    continue
                jobs.append(f"scale {k} {mu} {pl} turn 3,4,5 {F}/scale/{k}_mu{mu}_{pl}")
                jobs.append(f"hom {k} {mu} {pl} 5,6,7,8,9 {F}/hom.jsonl")
    Q.parent.mkdir(parents=True, exist_ok=True)
    Q.write_text("\n".join(jobs) + "\n")
    print(len(jobs), "jobs ->", Q)


if __name__ == "__main__":
    main()
