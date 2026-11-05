import sys, numpy as np
sys.path.insert(0, '.')
from stepread import Step
np.set_printoptions(suppress=True, precision=5, linewidth=150)
S = Step('/home/humanoid/Programs/hand/assets/cad_v2/full_cad/UR5 Adaptor.step')
print('scale to mm', S.s)
W = S.world_transforms()
print('solids', len(W))
for s, T, ch in W:
    fs = S.faces(s)
    print('solid', s, 'faces', len(fs), 'T=\n', T)
    from collections import Counter
    kinds = Counter(S.surface(f)['kind'] for f in fs)
    print(kinds)
    for f in fs:
        su = S.surface(f)
        v = S.face_vertices(f)
        if su['kind'] in ('cyl', 'cone'):
            a = su['axis']; o = su['origin']
            t = (v - o) @ a
            print('%-5s r %.4f axis %s origin %s t [%.4f, %.4f] same %s nverts %d' % (su['kind'], su['r'], a, o, t.min(), t.max(), su['same_sense'], len(v)), ('semi %.4f' % su['semi_angle'] if su['kind']=='cone' else ''))
        elif su['kind'] == 'plane':
            n = su['normal'] * (1 if su['same_sense'] == '.T.' else -1)
            print('plane n_out %s d %.4f  verts-bbox %s %s' % (n, np.dot(n, su['origin']), v.min(0), v.max(0)))
