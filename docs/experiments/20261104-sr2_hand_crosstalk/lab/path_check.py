import json, sys
import numpy as np
from handgeo import Hand, HOME_LOCAL, F
h = Hand()
poses = json.loads(sys.argv[1])
full = lambda p: {f: {j: p.get(f, {}).get(j, 0.0) for j in ("yaw", "mcp", "pip")} for f in F}
for a, b in zip(poses[:-1], poses[1:]):
    a, b = full(a), full(b)
    cs = []
    for s in np.linspace(0, 1, 21):
        h.set(HOME_LOCAL, {f: {j: a[f][j] + (b[f][j] - a[f][j]) * s for j in a[f]} for f in F})
        cs.append(min(h.clearances().values()))
    print(f"path min {min(cs):.1f} mm, end {cs[-1]:.1f} mm")
