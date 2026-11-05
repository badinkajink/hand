"""Yaw-axis centres of the three CAD fingers in frame A (mm): the foot, on the carriage's 10.2 mm yaw bore axis, of the
common perpendicular to the yaw link's MCP bore axis. Reads parts.json only (no meshes)."""
import json
import numpy as np

P = json.load(open("/home/humanoid/Programs/hand/logs/20261104-cad_v3/extract/parts.json"))["parts"]
fingers = {}
for p in P:
    if p["file"] != "hand" or not p["chain"][0].startswith("Finger"):
        continue
    for c in p["cylinders"]:
        if c["concave"] and abs(c["r"] - 5.1) < 0.06:
            fingers.setdefault(p["chain"][0], []).append((round(p["volume"]), np.array(c["point"]), np.array(c["axis"]), p["name"]))
out = {}
for f, rows in sorted(fingers.items()):
    yaw = [r for r in rows if r[0] == 23913][0]
    mcp = [r for r in rows if r[0] == 10683][0]
    pip = [r for r in rows if r[0] == 21293][0]
    p1, a1 = yaw[1], yaw[2] / np.linalg.norm(yaw[2])
    p2, a2 = mcp[1], mcp[2] / np.linalg.norm(mcp[2])
    n = np.cross(a1, a2); n /= np.linalg.norm(n)
    t = np.linalg.solve(np.column_stack([a1, -a2, n]), p2 - p1)
    c1 = p1 + t[0] * a1; c2 = p2 + t[1] * a2
    # PIP axis: parallel to MCP axis; distance between MCP and PIP lines
    p3, a3 = pip[1], pip[2] / np.linalg.norm(pip[2])
    d_mp = np.linalg.norm(np.cross(p3 - c2, a2))
    yaw_deg = np.degrees(np.arctan2(a2[2], a2[0]))
    out[f] = dict(yaw_centre_A=np.round(c1, 2).tolist(), yaw_to_mcp_mm=round(float(np.linalg.norm(c2 - c1)), 3),
                  mcp_to_pip_mm=round(float(d_mp), 3), yaw_axis_A=np.round(a1, 4).tolist(),
                  mcp_axis_A=np.round(a2, 4).tolist(), mcp_axis_tilt_deg=round(float(yaw_deg), 2))
    print(f, out[f])
c = {k: np.array(v["yaw_centre_A"]) for k, v in out.items()}
ks = sorted(c)
for i in range(3):
    for j in range(i + 1, 3):
        print(ks[i], ks[j], "dxy", np.round(c[ks[j]][:2] - c[ks[i]][:2], 2), "dist", round(float(np.linalg.norm(c[ks[j]][:2] - c[ks[i]][:2])), 2))
json.dump(out, open("/home/humanoid/Programs/hand/logs/20261104-cad_v3/current/yaw_centres.json", "w"), indent=1)
