"""SR2 hand: a crosstalk plan checked as lab/run_plan.py --ramp-deg 3 walks it (docs/experiments/20261104-sr2_hand_crosstalk).
The lab's layout plan replayed as run_plan.py walks it: every move, and every pose's safety return to zero, in
waypoints at most 3 deg apart on every joint. Within a segment between two waypoints a servo may run ahead of or behind
the others, so each joint lies anywhere between its two waypoint values: the segment's box. The clearance between the
closest pair of fingers (cad_v2 collision spheres) is taken at every waypoint, and where it is under 20 mm at the 64
corners of the box over that pair's six joints (the third finger at the segment's middle). Usage: .venv/bin/python scripts/sr2_plan_ramp_replay.py PLAN (about 15 min for 672 steps; run it under resguard)"""
import itertools
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/humanoid/Programs/hand/docs/experiments/20261104-sr2_hand_crosstalk/lab")
from handgeo import F, Hand, ORIGIN  # noqa: E402

J = ("yaw", "mcp", "pip")
h = Hand()
RMAX = {f: float(h.r[f].max()) for f in F}
CAP = 40.0
PAIRS = list(itertools.combinations(F, 2))


def pose(v):
    return {f: {j: float(v[3 * i + k]) for k, j in enumerate(J)} for i, f in enumerate(F)}


def clear(local, v):
    h.set(local, pose(v))
    C = {f: h.centres(f) for f in F}
    best = (CAP, "")
    for a, b in PAIRS:
        ta, tb = cKDTree(C[a]), cKDTree(C[b])
        M = ta.sparse_distance_matrix(tb, CAP + RMAX[a] + RMAX[b], output_type="coo_matrix")
        if M.nnz:
            d = M.data - h.r[a][M.row] - h.r[b][M.col]
            best = min(best, (float(d.min()), f"{a}-{b}"))
    return best


def segment(local, w0, w1, c0, c1):
    """Worst clearance in the box between two waypoints."""
    worst = min(c0, c1)
    if worst[0] >= 20.0 or not worst[1]:
        return worst
    a, b = worst[1].split("-")
    idx = [3 * F.index(a) + k for k in range(3)] + [3 * F.index(b) + k for k in range(3)]
    mid = 0.5 * (w0 + w1)
    for bits in itertools.product((0, 1), repeat=6):
        v = mid.copy()
        for i, bit in zip(idx, bits):
            v[i] = w1[i] if bit else w0[i]
        worst = min(worst, clear(local, v))
    return worst


def walk(local, start, target):
    d = np.max(np.abs(target - start))
    n = int(d // 3.0) + 1
    W = [start + (target - start) * i / n for i in range(n + 1)]
    C = [clear(local, w) for w in W]
    line = min(C)
    box = min(segment(local, W[i], W[i + 1], C[i], C[i + 1]) for i in range(n))
    return line, box


P = json.load(open(sys.argv[1]))
local = None
cur = np.zeros(9)
res = {}
for k, st in enumerate(P["steps"]):
    if st["op"] == "mounts":
        if local is not None:
            r = res[lay]
            print(lay, json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items() if k != "worst"}),
                  sorted(r["worst"])[:3], flush=True)
        local = {f: (st["mounts"][f]["x"] - ORIGIN[f][0], st["mounts"][f]["y"] - ORIGIN[f][1]) for f in F}
        lay = st["family"].replace("layout_", "")
        res.setdefault(lay, dict(moves=0, line=(CAP, ""), box=(CAP, ""), ret_line=(CAP, ""), ret_box=(CAP, ""), worst=[]))
        continue
    tgt = np.array([st["joints"][f][j] for f in F for j in J], float)
    if local is None:
        cur = tgt
        continue
    r = res[lay]
    line, box = walk(local, cur, tgt)
    r["moves"] += 1
    r["line"], r["box"] = min(r["line"], line), min(r["box"], box)
    r["worst"].append((round(box[0], 1), round(line[0], 1), k, st["family"]))
    if np.any(tgt != 0):                              # the safety return from this pose to zero
        rl, rb = walk(local, tgt, np.zeros(9))
        r["ret_line"], r["ret_box"] = min(r["ret_line"], rl), min(r["ret_box"], rb)
    cur = tgt
for lay, r in res.items():
    r["worst"] = sorted(r["worst"])[:3]
    print(lay, json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}), flush=True)
