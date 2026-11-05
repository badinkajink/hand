import sys, numpy as np, json, time
sys.path.insert(0, '.')
from stepread import Step
from collections import Counter
np.set_printoptions(suppress=True, precision=6, linewidth=160)
t0 = time.time()
S = Step('/home/humanoid/Programs/hand/assets/cad_v2/full_cad/assemblies/hand v3 (anthro config).step')
print('parsed', len(S.e), 'entities in %.1f s' % (time.time() - t0), 'scale', S.s)
W = S.world_transforms()
print('solids', len(W))
rows = []
for k, (s, T, ch) in enumerate(W):
    fs = S.faces(s)
    kinds = Counter(S.surface(f)['kind'] for f in fs)
    radii = sorted(round(S.surface(f)['r'], 3) for f in fs if S.surface(f)['kind'] == 'cyl')
    # vertices in A
    V = np.vstack([S.face_vertices(f) for f in fs])
    VA = (T[:3, :3] @ V.T).T + T[:3, 3]
    rows.append(dict(k=k, solid=s, chain=[c[0] for c in ch], nfaces=len(fs), kinds=dict(kinds), radii=radii,
                     vmin=VA.min(0).tolist(), vmax=VA.max(0).tolist(), T=T.tolist()))
json.dump(rows, open('hand_solids_step.json', 'w'), indent=0)
for r in rows:
    print(r['k'], r['solid'], r['chain'][-1], r['nfaces'], r['kinds'], 'zmin %.3f zmax %.3f' % (r['vmin'][2], r['vmax'][2]))
