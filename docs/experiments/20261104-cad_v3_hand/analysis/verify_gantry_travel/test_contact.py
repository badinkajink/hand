"""Check contact.py on boxes and a tilted plate with known first contacts."""
import numpy as np
import trimesh
from contact import samples, Caster, pair_gap

def box(lo, hi):
    lo, hi = np.array(lo, float), np.array(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    return m

cases = []
# 1. block ahead along +x, overlapping in y,z
A = box([0, 0, 0], [1, 1, 1]); B = box([5, 0.5, 0], [6, 1.5, 1]); cases.append((A, B, 0, 4.0, -np.inf))
# 2. flush in z: B sits exactly on top of A's top plane -> no contact either way
A = box([0, 0, 0], [1, 1, 1]); B = box([5, 0, 1], [6, 1, 2]); cases.append((A, B, 0, np.inf, -np.inf))
# 3. obstacle behind, -x
A = box([0, 0, 0], [1, 1, 1]); B = box([-7, 0.2, 0.2], [-3, 0.8, 0.8]); cases.append((A, B, 0, np.inf, -3.0))
# 4. edge-edge: A rotated 45 deg about x, B rotated 45 deg about z, crossing edges
A = trimesh.creation.box(extents=[1, 1, 1]); A.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [1, 0, 0]))
B = trimesh.creation.box(extents=[1, 1, 1]); B.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [0, 1, 0])); B.apply_translation([3, 0, 0])
# A's +x face at x=0.5; B's leading edge (rotated about y) at x = 3 - sqrt(2)/2 along the line z=0, so gap = 2.5 - 0.7071
cases.append((A, B, 0, 2.5 - np.sqrt(0.5), -np.inf))
# 5. tilted plate 1 deg: moving cube, static plate with normal tilted about y by 1 deg, along x
A = box([0, -1, -1], [1, 1, 1]); B = box([4, -3, -3], [5, 3, 3]); B.apply_transform(trimesh.transformations.rotation_matrix(np.radians(1.0), [0, 1, 0], point=[4.5, 0, 0]))
# B's -x face: plane through (4,0,0) rotated 1 deg about y about point (4.5,0,0): x = 4.5 - 0.5cos - ... compute exactly
c, s_ = np.cos(np.radians(1.0)), np.sin(np.radians(1.0))
# face points (4, y, z) -> x' = 4.5 + (-0.5)c + z s_ ; min over A's corners z in [-1,1] of x' - 1
exp = 4.5 - 0.5 * c - 1 * s_ - 1.0
cases.append((A, B, 0, exp, -np.inf))
for n, (A, B, ax, ep, em) in enumerate(cases, 1):
    tp, tm, _, _ = pair_gap(samples(A), Caster(A, ax), samples(B), Caster(B, ax), ax)
    print(n, 'tp %.4f (exp %.4f)  tm %.4f (exp %.4f)' % (tp, ep, tm, em))
