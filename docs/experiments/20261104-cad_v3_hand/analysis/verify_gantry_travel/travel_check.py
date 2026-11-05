"""Stage travel of the three SR2 gantries from the hand STLs (file 'hand', frame A, mm), with contact.py.

Groups are assigned here from the bounding boxes and volumes printed in notes (my own classification):
  Y stage = the Finger v3 sub-assembly (carriage, Y nut, servos, links, tip) + the Y rail block; moves along y_A.
  X stage = beam, X rail block, X nut, Y rail, Y screw, Y bearings and collar, Y coupler, Y motor and shaft, + Y stage;
            moves along x_A.
  Everything else is static for that motion (other gantries and fingers at their CAD poses included).
Sliding pairs excluded: X block/X rail, X nut/X screw, Y block/Y rail, Y nut/Y screw.
Thumb: run as exported and with the thumb gantry and finger moved +2.5 mm in y_A, except its X motor, mount and shaft.
"""
import json, sys, time
import numpy as np
from contact import load, samples, Caster, pair_gap

ROOT = '/home/humanoid/Programs/hand/logs/20261104-cad_v3/extract/hand/'
ALL = ['%03d' % i for i in range(121)]
G = {
    'index': dict(Y=['082'] + ['%03d' % i for i in range(85, 102)],
                  X=['081', '080', '078', '084', '073', '066', '068', '069', '063', '072', '067'],
                  slide=[('080', '064'), ('078', '079'), ('082', '084'), ('085', '073')]),
    'middle': dict(Y=['117'] + ['%03d' % i for i in range(23, 40)],
                   X=['120', '113', '105', '116', '111', '112', '104', '108', '115', '103', '106'],
                   slide=[('113', '109'), ('105', '110'), ('117', '116'), ('036', '111')]),
    'thumb': dict(Y=['049'] + ['%03d' % i for i in range(6, 23)],
                  X=['047', '052', '053', '046', '058', '043', '056', '055', '040', '044', '057'],
                  slide=[('052', '059'), ('053', '050'), ('049', '046'), ('017', '058')]),
}
THUMB_ALL = ['%03d' % i for i in range(6, 23)] + ['%03d' % i for i in range(40, 62)]
THUMB_STAY = ['060', '051', '054']          # X motor mount, X motor, X motor shaft stay on the base plate


LOG = open('/home/humanoid/Programs/hand/logs/20261104-cad_v3/verify_gantry_travel/travel_check.log', 'a')


def run(shift_thumb):
    t0 = time.time()
    LOG.write('run shift_thumb=%s\n' % shift_thumb); LOG.flush()
    M = {n: load(ROOT + n + '.stl') for n in ALL}
    if shift_thumb:
        for n in THUMB_ALL:
            if n not in THUMB_STAY:
                M[n].apply_translation([0, 2.5, 0])
    S = {n: samples(M[n]) for n in ALL}
    bb = {n: M[n].bounds for n in ALL}
    res = {}
    for f, g in G.items():
        slide = set(g['slide']) | {(b, a) for a, b in g['slide']}
        for stage, ax in (('Y', 1), ('X', 0)):
            mov = g['Y'] if stage == 'Y' else g['X'] + g['Y']
            sta = [n for n in ALL if n not in mov]
            tr = [a for a in range(3) if a != ax]
            C = {}
            rows = []
            for m in mov:
                for s in sta:
                    if (m, s) in slide:
                        continue
                    lo = np.maximum(bb[m][0], bb[s][0]); hi = np.minimum(bb[m][1], bb[s][1])
                    if (lo[tr] > hi[tr]).any():
                        continue                        # no transverse overlap: never meet along ax
                    for n in (m, s):
                        if (n, ax) not in C:
                            C[(n, ax)] = Caster(M[n], ax)
                    npair = locals().get('npair', 0) + 1
                    tp, tm, wp, wm = pair_gap(S[m], C[(m, ax)], S[s], C[(s, ax)], ax)
                    if np.isfinite(tp) or np.isfinite(tm):
                        rows.append(dict(m=m, s=s, tp=float(tp), tm=float(tm),
                                         wp=None if wp is None else np.round(wp, 3).tolist(),
                                         wm=None if wm is None else np.round(wm, 3).tolist()))
            up = sorted([r for r in rows if np.isfinite(r['tp'])], key=lambda r: r['tp'])[:8]
            dn = sorted([r for r in rows if np.isfinite(r['tm'])], key=lambda r: -r['tm'])[:8]
            zero = [r for r in rows if (np.isfinite(r['tp']) and r['tp'] < 0.02) or (np.isfinite(r['tm']) and r['tm'] > -0.02)]
            res[f + '_' + stage] = dict(axis='xyz'[ax], up=[(r['m'], r['s'], round(r['tp'], 3), r['wp']) for r in up],
                                         down=[(r['m'], r['s'], round(r['tm'], 3), r['wm']) for r in dn],
                                         touching_at_cad=[(r['m'], r['s'], r['tp'], r['tm']) for r in zero])
            LOG.write('%s %s up %s down %s touch %d %.0fs\n' % (f, stage, res[f + '_' + stage]['up'][:3], res[f + '_' + stage]['down'][:3], len(zero), time.time() - t0)); LOG.flush()
            C.clear()
            print(f, stage, 'up', res[f + '_' + stage]['up'][:3], 'down', res[f + '_' + stage]['down'][:3], 'touch', len(zero), '%.0fs' % (time.time() - t0), flush=True)
    return res


if __name__ == '__main__':
    out = {'as_exported': run(False), 'thumb_on_holes': run(True)}
    json.dump(out, open('/home/humanoid/Programs/hand/logs/20261104-cad_v3/verify_gantry_travel/travel_check.json', 'w'), indent=1)
