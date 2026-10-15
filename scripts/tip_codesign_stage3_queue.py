#!/usr/bin/env python3
"""Write the stage-3 job queue (logs/20261014-tip_codesign/stage3/queue.txt) from the stage-2 finalists
(docs/experiments/20261014-tip_codesign/screen/summary_stage2.json): fresh placements (scale cells seeds 3-5, HOM on
D1-D8 seeds 5-9) at mu 0.7 and 0.5 on the servo plant and at kp 0.5 and torque limit 0.12 N m at mu 0.7 (plan step 9),
turns only: the hold test of hand_object_scale_sim always runs seed 0, so the mu 0.5 three-finger hold of stage 3 is read
from the stage-2 hold rows; and the builder-v2 check of ledger step 27 (replays D1-D8 seeds 0-4 and HOM seeds 0-2 at mu 1.0, builder v1 and v2).
Job lines are those of logs/20261014-tip_codesign/stage3/worker.sh.

    python3 scripts/tip_codesign_stage3_queue.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"
CONDS = [("0.7", SERVO), ("0.5", SERVO), ("0.7", "kp0.5_kv0_fr0.18_fl0_dp0.0144"), ("0.7", "kp0.72_kv0_fr0.12_fl0_dp0.0144")]


def main():
    S = json.loads((ROOT / "docs/experiments/20261014-tip_codesign/screen/summary_stage2.json").read_text())
    fin = S["finalists"]
    jobs = []
    for k in fin:
        for mu, pl in CONDS:
            jobs.append(f"scale {k} {mu} {pl} turn")
            jobs.append(f"hom {k} {mu} {pl}")
    for k in fin:
        if k.startswith("tcd_"):
            for b in ("v1", "v2"):
                jobs.append(f"v2replay {k} {b}")
                jobs.append(f"v2hom {k} {b}")
    q = ROOT / "logs/20261014-tip_codesign/stage3/queue.txt"
    q.write_text("\n".join(jobs) + "\n")
    (ROOT / "docs/experiments/20261014-tip_codesign/final/stage3_tips.json").write_text(json.dumps(
        {"finalists": fin, "why": S["finalist_rule"], "conditions": CONDS}, indent=1))
    print(len(fin), "finalists,", len(jobs), "jobs ->", q)


if __name__ == "__main__":
    main()
