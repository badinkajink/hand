"""Cross-coupling check (thumb on the base plate holes): Y stops with the X stage at its home and far stops, and X stops
with the Y stage at its home and far stops, all other stages at their CAD pose. Stop displacements from travel_check."""
import json, time
import numpy as np
from contact import load, samples, Caster, pair_gap
from travel_check import ROOT, ALL, G, THUMB_ALL, THUMB_STAY

STOPS = {  # stage: (home, far) displacement from the CAD pose, mm (travel_check.json, thumb_on_holes, minus the 0.006 bias)
    'index': {'X': (54.132, -10.668), 'Y': (-47.884, 14.698)},
    'middle': {'X': (-45.786, 19.013), 'Y': (-44.827, 17.756)},
    'thumb': {'X': (-41.712, 73.088), 'Y': (45.495, -17.088)},
}
M0 = {n: load(ROOT + n + '.stl') for n in ALL}
for n in THUMB_ALL:
    if n not in THUMB_STAY:
        M0[n].apply_translation([0, 2.5, 0])
out = {}
t0 = time.time()
for f, g in G.items():
    slide = set(g['slide']) | {(b, a) for a, b in g['slide']}
    for stage, ax, other, oax in (('Y', 1, 'X', 0), ('X', 0, 'Y', 1)):
        for which, disp in zip(('home', 'far'), STOPS[f][other]):
            M = {n: m.copy() for n, m in M0.items()}
            shift_group = (g['X'] + g['Y']) if other == 'X' else g['Y']
            v = np.zeros(3); v[oax] = disp
            # back off 0.05 mm from the stop so the group is not in contact
            v[oax] -= 0.05 * np.sign(disp)
            for n in shift_group:
                M[n].apply_translation(v)
            S = {n: samples(M[n]) for n in ALL}
            bb = {n: M[n].bounds for n in ALL}
            mov = g['Y'] if stage == 'Y' else g['X'] + g['Y']
            sta = [n for n in ALL if n not in mov]
            tr = [a for a in range(3) if a != ax]
            C, rows = {}, []
            for m in mov:
                for s in sta:
                    if (m, s) in slide:
                        continue
                    lo = np.maximum(bb[m][0], bb[s][0]); hi = np.minimum(bb[m][1], bb[s][1])
                    if (lo[tr] > hi[tr]).any():
                        continue
                    for n in (m, s):
                        if n not in C:
                            C[n] = Caster(M[n], ax)
                    tp, tm, wp, wm = pair_gap(S[m], C[m], S[s], C[s], ax)
                    rows.append((m, s, float(tp), float(tm)))
            up = sorted([r for r in rows if np.isfinite(r[2])], key=lambda r: r[2])[:4]
            dn = sorted([r for r in rows if np.isfinite(r[3])], key=lambda r: -r[3])[:4]
            key = '%s_%s_at_%s_%s' % (f, stage, other, which)
            out[key] = dict(up=[(a, b, round(c, 3)) for a, b, c, d in up], down=[(a, b, round(d, 3)) for a, b, c, d in dn])
            print(key, out[key], '%.0fs' % (time.time() - t0), flush=True)
json.dump(out, open('/home/humanoid/Programs/hand/logs/20261104-cad_v3/verify_gantry_travel/travel_corners.json', 'w'), indent=1)
