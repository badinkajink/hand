#!/usr/bin/env python3
"""Write the stage-2 job queue (logs/20261014-tip_codesign/stage2/queue.txt) from the stage-1 ranking
(docs/experiments/20261014-tip_codesign/screen/summary_stage1.json): the 8 highest J1, the as-built tip as printed
(tcd_rs3_re3_p0.2) and the simulation default (tpu2.7, the comparison baseline), each at the five conditions of plan
step 8.

    python3 scripts/tip_codesign_stage2_queue.py [--n 8]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"
CONDS = [("1.0", SERVO), ("0.7", SERVO), ("0.5", SERVO), ("0.7", "kp0.5_kv0_fr0.18_fl0_dp0.0144"),
         ("0.7", "kp0.72_kv0_fr0.12_fl0_dp0.0144")]
BASE = "tcd_rs3_re3_p0.2"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    S = json.loads((ROOT / "docs/experiments/20261014-tip_codesign/screen/summary_stage1.json").read_text())
    rank = sorted(((v["J1"] or -1, k) for k, v in S["candidates"].items()), reverse=True)
    keys = [k for _, k in rank[:a.n]]
    for ref in (BASE, "tpu2.7"):
        if ref not in keys:
            keys.append(ref)
    jobs = []
    for k in keys:                                   # the slow jobs first
        jobs.append(f"drake {k}")
    for k in keys:
        for mu, pl in CONDS:
            if (mu, pl) != ("0.7", SERVO):           # stage 1 has the scale cells at mu 0.7
                jobs.append(f"scale {k} {mu} {pl} {'hold,turn' if mu == '0.5' else 'turn'}")
            jobs.append(f"hom {k} {mu} {pl}")
        jobs.append(f"replay {k}")
    q = ROOT / "logs/20261014-tip_codesign/stage2/queue.txt"
    q.write_text("\n".join(jobs) + "\n")
    (ROOT / "docs/experiments/20261014-tip_codesign/screen/stage2_candidates.json").write_text(
        json.dumps({"candidates": keys, "J1": {k: S["candidates"][k]["J1"] for k in keys}, "conditions": CONDS}, indent=1))
    print(len(keys), "candidates,", len(jobs), "jobs ->", q)


if __name__ == "__main__":
    main()
