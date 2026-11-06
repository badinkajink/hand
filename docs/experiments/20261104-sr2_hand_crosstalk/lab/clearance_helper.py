"""The least finger-to-finger clearance of the SR2 hand in the CAD (handgeo's cad_v2 spheres), one answer a request, for
run_plan.py's --min-clearance. Each line in is {"mounts": {finger: {"x", "y"}}, the palm frame of the station's
mount_positions, "pose": {finger: {"yaw", "mcp", "pip"}}, sim degrees}; each line out is {"least": mm, "pair": "a-b"} or
{"error": ...}. The first line out is {"ready": true}.

    uv run --no-project --python 3.10 --with mujoco --with numpy --with scipy python clearance_helper.py
"""
import json
import sys

from handgeo import F, ORIGIN, Hand

h = Hand()
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    try:
        q = json.loads(line)
        h.set({f: (q["mounts"][f]["x"] - ORIGIN[f][0], q["mounts"][f]["y"] - ORIGIN[f][1]) for f in F}, q["pose"])
        c = h.clearances()
        k = min(c, key=c.get)
        print(json.dumps({"least": round(float(c[k]), 2), "pair": k}), flush=True)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)
