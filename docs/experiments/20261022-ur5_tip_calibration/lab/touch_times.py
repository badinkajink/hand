#!/usr/bin/env python3
"""Where a session's touch-off time goes: per touch-off, travel (action start to the tare), the tare, the fast approach,
the slow approach (to the stop) and the return; means over a folder's ok touch-offs, and the session's span.
   touch_times.py SESSION [SESSION ...]"""
import json, sys
import numpy as np
for D in sys.argv[1:]:
    ev = [json.loads(l) for l in open(D + "/events.jsonl")]
    tj = {t["id"]: t for t in json.load(open(D + "/touches.json"))}
    starts, rows = {}, []
    for e in ev:
        if e["event"] == "start":
            starts[e["id"]] = e["t"]
        if e["event"] == "end" and e["id"] in tj and e.get("status") == "ok":
            r = e["rec"]; t0 = starts[e["id"]]
            fast = (r["fast"]["t"][1] - r["fast"]["t"][0]) if r.get("fast") and r["fast"]["t"][1] else 0.0
            rows.append([e["t"] - t0, r["t_rest0"][0] - t0, r["t_rest0"][1] - r["t_rest0"][0], fast,
                         r["t_stop"] - r["t_move0"], e["t"] - r["t_stop"]])
    a = np.array(rows)
    m = a.mean(0)
    pad = [e for e in ev if e["event"] in ("start", "end") and e["id"] in tj]
    span = (max(e["t"] for e in pad) - min(e["t"] for e in pad)) / 60 if pad else float("nan")
    print(f"{D}: {len(a)} touch-offs over {span:.1f} min; each {m[0]:.0f} s = travel {m[1]:.0f} + tare {m[2]:.1f} + fast "
          f"{m[3]:.1f} + slow {m[4]:.0f} + return {m[5]:.0f}")
