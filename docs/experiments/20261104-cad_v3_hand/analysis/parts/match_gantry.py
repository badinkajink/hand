"""Match each solid of gantry_v3.step / gantry_v3_thumb.step to the instances in the hand file.
The instance transform T (file frame -> A) comes from the frame part's placement; every other part is then compared by
volume and by the offset of its centre of mass, com_A - T(com_file): zero for parts fixed to the frame, a constant
vector along the x axis for the x stage, and so on."""
import json, numpy as np
from scipy.spatial.transform import Rotation as Rot
P = json.load(open('/home/humanoid/Programs/hand/logs/20261104-cad_v3/extract/parts.json'))['parts']
def T_of(p):
    T = np.eye(4); T[:3, :3] = Rot.from_quat(p['placement']['quat_xyzw']).as_matrix(); T[:3, 3] = p['placement']['pos']; return T
def vol(p):
    return p['volume']
hand = [p for p in P if p['file'] == 'hand']
res = {}
for inst, fname, fvol in [('gantry_v3 <1>', 'gantry', 53883), ('gantry_v3 <2>', 'gantry', 53883), ('gantry_v3_thumb <1>', 'gantry_thumb', 77381)]:
    fparts = [p for p in P if p['file'] == fname]
    iparts = [p for p in hand if p['chain'][0] == inst]
    ff = [p for p in fparts if abs(vol(p) - fvol) < 5][0]
    fi = [p for p in iparts if abs(vol(p) - fvol) < 5][0]
    T = T_of(fi) @ np.linalg.inv(T_of(ff))
    # check the frame's own COM and bbox map
    print(f'\n== {inst} <- {fname}: T rotation (deg about z) {np.degrees(np.arctan2(T[1,0], T[0,0])):.3f}, t = {np.round(T[:3,3], 3)}, det {np.linalg.det(T[:3,:3]):.3f}')
    print('   R =', np.round(T[:3, :3], 4).tolist())
    used = set()
    rows = []
    for p in fparts:
        c = T[:3, :3] @ np.array(p['com']) + T[:3, 3]
        cands = [q for q in iparts if abs(vol(q) - vol(p)) < 0.002 * vol(p) + 1 and q['name'] not in used]
        # also allow parts of the same volume in the hand file outside the instance (top level)
        best = None
        for q in cands:
            dlt = np.array(q['com']) - c
            # rotation check: principal inertia orientation, compare inertia tensors after rotating
            I_f = T[:3, :3] @ np.array(p['inertia_mm5']) @ T[:3, :3].T
            I_q = np.array(q['inertia_mm5'])
            dI = np.linalg.norm(I_f - I_q) / max(np.linalg.norm(I_q), 1e-9)
            score = (dI > 0.01, np.linalg.norm(dlt))
            if best is None or score < best[0]:
                best = (score, q, dlt, dI)
        rows.append((p, best))
    # resolve: greedy by distance with matching inertia orientation
    for p, best in rows:
        if best is None:
            print(f'   {fname}/{p["name"]} vol {vol(p):8.0f}  -> MISSING in instance')
            continue
        (flag, dist), q, dlt, dI = best
        print(f'   {fname}/{p["name"]} vol {vol(p):8.0f}  -> hand/{q["name"]}  offset {np.round(dlt, 2)} |d| {dist:6.2f}  inertia-rel-diff {dI:.4f}')
