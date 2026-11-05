"""SR2 hand (docs/experiments/20261104-sr2_hand_crosstalk): plan_layouts_s1 with one stepper axis left behind at a layout move: for each layout and each axis that moves there,
that axis stays at its previous value. Clearance of every pose and every move (21 samples on the joint-space line),
cad_v2 collision spheres.

    .venv/bin/python scripts/fingertip/sr2_plan_stepper_skip_check.py PLAN [FIRST_STEP]"""
import itertools, json, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/humanoid/Programs/hand/docs/experiments/20261104-sr2_hand_crosstalk/lab")
from handgeo import F, Hand, ORIGIN, HOME_LOCAL
J = ("yaw", "mcp", "pip")
h = Hand()
RMAX = {f: float(h.r[f].max()) for f in F}
CAP = 40.0
def clear(local, v):
    h.set(local, {f: {j: float(v[3 * i + k]) for k, j in enumerate(J)} for i, f in enumerate(F)})
    C = {f: h.centres(f) for f in F}
    best = (CAP, "")
    for a, b in itertools.combinations(F, 2):
        M = cKDTree(C[a]).sparse_distance_matrix(cKDTree(C[b]), CAP + RMAX[a] + RMAX[b], output_type="coo_matrix")
        if M.nnz:
            best = min(best, (float((M.data - h.r[a][M.row] - h.r[b][M.col]).min()), f"{a}-{b}"))
    return best
P = json.load(open(sys.argv[1]))
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
home = {f: (ORIGIN[f][0] + HOME_LOCAL[f][0], ORIGIN[f][1] + HOME_LOCAL[f][1]) for f in F}
segs, cur, prev_m, m = [], None, home, home
for k, st in enumerate(P["steps"]):
    if st["op"] == "mounts":
        prev_m, m = m, {f: (st["mounts"][f]["x"], st["mounts"][f]["y"]) for f in F}
        cur = dict(name=st["family"], prev=prev_m, mounts=m, steps=[])
        segs.append(cur)
    elif cur is not None and k >= start:
        cur["steps"].append((k, np.array([st["joints"][f][j] for f in F for j in J], float)))
for s in segs:
    if not s["steps"]:
        continue
    for f in F:
        for ax in (0, 1):
            if abs(s["mounts"][f][ax] - s["prev"][f][ax]) < 1e-6:
                continue
            mm = {g: list(s["mounts"][g]) for g in F}
            mm[f][ax] = s["prev"][f][ax]
            local = {g: (mm[g][0] - ORIGIN[g][0], mm[g][1] - ORIGIN[g][1]) for g in F}
            last, worst_pose, worst_move = np.zeros(9), (CAP, ""), (CAP, "")
            for k, q in s["steps"]:
                worst_pose = min(worst_pose, clear(local, q))
                for t in np.linspace(0, 1, 11):
                    worst_move = min(worst_move, clear(local, last + (q - last) * t))
                last = q
            print(f"{s['name']}: {f} {'xy'[ax]} left at {s['prev'][f][ax]:.1f} (target {s['mounts'][f][ax]:.1f}): "
                  f"poses min {worst_pose[0]:.1f} mm ({worst_pose[1]}), moves min {worst_move[0]:.1f} mm ({worst_move[1]})", flush=True)
