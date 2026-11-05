"""Standalone robot_mount.step and base_plate.step against their local frames inside the hand file; the adaptor's
axis-angle; the lowest point of the hand from the STEP surfaces; the translation-only alternative."""
import sys, json, numpy as np
sys.path.insert(0, '.')
from stepread import Step
from scipy.spatial.transform import Rotation
np.set_printoptions(suppress=True, precision=6, linewidth=160)
ROOT = '/home/humanoid/Programs/hand/'
hand = Step(ROOT + 'assets/cad_v2/full_cad/assemblies/hand v3 (anthro config).step')
W = hand.world_transforms()
out = {}

def feats(S, solid, T=np.eye(4)):
    cs, ps = [], []
    for f in S.faces(solid):
        su = S.surface(f)
        V = S.face_vertices(f)
        V = (T[:3, :3] @ V.T).T + T[:3, 3]
        if su['kind'] == 'cyl':
            a = T[:3, :3] @ su['axis']; o = T[:3, :3] @ su['origin'] + T[:3, 3]
            tt = (V - o) @ a
            p0, p1 = o + a * tt.min(), o + a * tt.max()
            cs.append((round(su['r'], 4), tuple(np.round(np.sort([p0, p1], axis=0).ravel(), 3))))
        elif su['kind'] == 'plane':
            n = T[:3, :3] @ su['normal'] * (1 if su['same_sense'] == '.T.' else -1)
            ps.append((tuple(np.round(n, 5)), tuple(np.round(V.mean(0), 3))))
    return sorted(cs), sorted(ps)

for name, fname, k in (('robot_mount', 'robot_mount.step', 1), ('base_plate', 'base_plate.step', 2), ('adaptor', 'UR5 Adaptor.step', 4)):
    S = Step(ROOT + 'assets/cad_v2/full_cad/' + fname)
    sol = S.world_transforms()[0][0]
    s, T, ch = W[k]
    a = feats(S, sol)          # standalone, own frame
    b = feats(hand, s)         # inside the hand file, local frame
    c = feats(S, sol, T)       # standalone mapped by the hand's instance transform
    d = feats(hand, s, T)      # inside the hand file, in A
    same_local = (a == b)
    same_A = (c == d)
    rv = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    ang = np.degrees(np.linalg.norm(rv)); ax = rv / np.linalg.norm(rv)
    print('%-12s standalone == hand-local: %s (cyl %d, planes %d); mapped standalone == hand in A: %s' % (name, same_local, len(a[0]), len(a[1]), same_A))
    print('    T_A_from_file =\n', T, '\n    rotation %.4f deg about %s, translation %s' % (ang, np.round(ax, 4), np.round(T[:3, 3], 6)))
    out[name] = dict(same_local=same_local, same_A=same_A, T=T.tolist(), angle_deg=ang, axis=ax.tolist())
    # translation-only alternative: displacement 2|p_xy - com_xy| for the plane centres and cylinder ends
    if name != 'adaptor':
        pts = np.array([p[1] for p in a[1]] + [q[1][:3] for q in a[0]] + [q[1][3:] for q in a[0]])
        V = np.vstack([S.face_vertices(f) for f in S.faces(sol)])
        com = None
        pj = [x for x in json.load(open(ROOT + 'logs/20261104-cad_v3/extract/parts.json'))['parts'] if x['file'] == name][0]
        com = np.array(pj['com'])
        Tt = np.eye(4); Tt[:3, 3] = (T[:3, :3] @ com + T[:3, 3]) - com
        dev_pts = np.linalg.norm(((T[:3, :3] - Tt[:3, :3]) @ (pts - com).T).T, axis=1)
        dev_v = np.linalg.norm(((T[:3, :3] - Tt[:3, :3]) @ (V - com).T).T, axis=1)
        print('    translation-only alternative: largest displacement of a face centre / cylinder end %.2f mm, of a vertex %.2f mm' % (dev_pts.max(), dev_v.max()))
        out[name]['translation_only_max_face_mm'] = float(dev_pts.max())
        out[name]['translation_only_max_vertex_mm'] = float(dev_v.max())

# lowest point of the hand: spheres, tori, cylinders evaluated exactly where possible
zmin = []
for k, (s, T, ch) in enumerate(W):
    for f in hand.faces(s):
        su = hand.surface(f)
        V = hand.face_vertices(f)
        VA = (T[:3, :3] @ V.T).T + T[:3, 3]
        zmin.append((VA[:, 2].min(), k, 'vertex'))
        t, p = hand.e[f]
        st, sp = hand.e[p[2]]
        if st == 'SPHERICAL_SURFACE':
            Tp = hand.placement(sp[1]); r = sp[2] * hand.s
            cA = T[:3, :3] @ Tp[:3, 3] + T[:3, 3]
            low = cA - np.array([0, 0, r])
            # is the lowest point of the sphere inside this face? check that it is below every vertex of the face's loops
            zmin.append((low[2], k, 'sphere bottom (if on face) r=%.3f centre %s, face vertex min z %.3f' % (r, np.round(cA, 3), VA[:, 2].min())))
zmin.sort()
for z in zmin[:6]:
    print('low:', round(z[0], 4), 'solid', z[1], z[2])
json.dump(out, open('standalone.json', 'w'), indent=1)
