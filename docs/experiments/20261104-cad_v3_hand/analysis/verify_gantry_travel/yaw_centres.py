"""Yaw centre of each finger from B-rep cylinders in parts.json: the foot, on the carriage's 10.2 mm yaw bore axis,
of the common perpendicular to the yaw link's 10.2 mm MCP bore axis. No meshes are loaded."""
import json, numpy as np
from pj import part
fing = {'index': ('094', '088'), 'middle': ('028', '029'), 'thumb': ('016', '015')}
out = {}
def bore(p, r=5.1):
    cs = [c for c in p['cylinders'] if abs(c['r'] - r) < 1e-3 and c['concave'] and c['span_deg'] > 359]
    assert len(cs) == 1, len(cs)
    return np.array(cs[0]['point']), np.array(cs[0]['axis']) / np.linalg.norm(cs[0]['axis'])
for f, (carr, yawl) in fing.items():
    p1, d1 = bore(part('hand', carr))      # yaw axis
    p2, d2 = bore(part('hand', yawl))      # MCP axis
    w0 = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w0, d2 @ w0
    den = a * c - b * b
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    q1, q2 = p1 + s * d1, p2 + t * d2
    ang = np.degrees(np.arccos(abs(b)))
    out[f] = dict(yaw_centre_A=q1.round(4).tolist(), mcp_foot_A=q2.round(4).tolist(), yaw_to_mcp_mm=round(float(np.linalg.norm(q1 - q2)), 4),
                  angle_yaw_mcp_deg=round(float(ang), 3), yaw_axis=d1.round(5).tolist(),
                  tilt_about_x_deg=round(float(np.degrees(np.arctan2(abs(d1[2]), abs(d1[1])))), 4))
    print(f, out[f])
json.dump(out, open('yaw_centres.json', 'w'), indent=1)
