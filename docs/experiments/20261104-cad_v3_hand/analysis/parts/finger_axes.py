"""Joint axes of each Finger v3 instance from the cylinders of its shafts (214 mm^3, r 2.5), bearings (236, r 2.5/5.0),
disc horns (429, r 9.0), arm horn (191, r 3.0) and servos (6673, r 6.05 boss)."""
import json, numpy as np
P = json.load(open('/home/humanoid/Programs/hand/logs/20261104-cad_v3/extract/parts.json'))['parts']
def main_cyl(p, r):
    cs = [c for c in p['cylinders'] if abs(c['r'] - r) < 0.02]
    if not cs:
        return None
    c = max(cs, key=lambda c: c['area'])
    a = np.array(c['axis']); a = a / np.linalg.norm(a)
    # canonical sign
    i = np.argmax(np.abs(a)); a = a * np.sign(a[i])
    pt = np.array(c['point']); com = np.array(p['com'])
    # point on axis closest to the part's COM
    q = pt + a * np.dot(com - pt, a)
    return a, q
def line_dist(a1, q1, a2, q2):
    ang = np.degrees(np.arccos(min(1, abs(np.dot(a1, a2)))))
    d = q2 - q1; d = d - a1 * np.dot(d, a1)
    return ang, np.linalg.norm(d)
for inst in ['Finger v3 <1>', 'Finger v3 <2>', 'Finger v3 <3>']:
    ps = [p for p in P if p['file'] == 'hand' and p['chain'][0] == inst]
    print('\n==', inst)
    shafts = [p for p in ps if abs(p['volume'] - 214) < 3]
    axes = {}
    for s in shafts:
        a, q = main_cyl(s, 2.5)
        axes[s['name']] = (a, q)
        print(f"  shaft {s['name']} axis {np.round(a,3)} through {np.round(q,2)}")
    for p in ps:
        v = p['volume']
        r = {236: 5.0, 429: 9.0, 191: 3.0, 6673: 6.05, 1077: 10.0}.get(min([236, 429, 191, 6673, 1077], key=lambda x: abs(x - v)) if min(abs(v - x) for x in [236, 429, 191, 6673, 1077]) < 5 else None)
        if r is None:
            continue
        mc = main_cyl(p, r)
        if mc is None:
            print('  no cyl', p['name'], v); continue
        a, q = mc
        best = min(((line_dist(a, q, *axes[k]), k) for k in axes), key=lambda t: (t[0][0] > 2, t[0][1]))
        print(f"  {p['name']} vol {v:6.0f} r {r}: axis {np.round(a,3)} closest shaft {best[1]}: angle {best[0][0]:.2f} deg, offset {best[0][1]:.2f} mm")
