"""Summary of a hand_link.py log: the period's steadiness, the round trip from a state to the target that answers it
(controller time and network included), the object's age when its state was sent, and the stop after the
controller went quiet. Writes LOG's summary beside it (_summary.json).

    python3 link_summary.py LOG.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np

log = Path(sys.argv[1])
R = [json.loads(l) for l in open(log)]
t = np.array([r["t"] for r in R])
per = np.diff(t)
rtt = np.array([r["rtt"] for r in R if r.get("rtt") is not None])
age = np.array([r["obj_age"] for r in R if r.get("obj_age") is not None])
applied = [r for r in R if "target" in r]
stops = [(r["seq"], r["stop"]) for r in R if "stop" in r]


def q(x, ps=(50, 95, 99)):
    return {f"p{p}": round(float(np.percentile(x, p)), 3) for p in ps} | {"max": round(float(np.max(x)), 3)} if len(x) else None


out = dict(periods=len(R), period_ms=q(per), targets_applied=len(applied), round_trip_ms=q(rtt),
           obj_age_ms=q(age), stops=stops)
if stops and applied:
    last = max(r["seq"] for r in applied if r["seq"] < stops[0][0])
    out["stop_after_last_target_ms"] = round(R[stops[0][0]]["t"] - R[last]["t"], 2)
Path(str(log).replace(".jsonl", "_summary.json")).write_text(json.dumps(out, indent=1))
print(json.dumps(out))
