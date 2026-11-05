"""Helpers to read parts.json (B-rep features) for the travel check."""
import json, numpy as np
PJ = '/home/humanoid/Programs/hand/logs/20261104-cad_v3/extract/parts.json'
D = json.load(open(PJ))
def part(file, name):
    for p in D['parts']:
        if p['file'] == file and p['name'] == name:
            return p
    raise KeyError((file, name))
def planes(p, axis, sign=None, amin=0.0, tol=1e-6):
    """planar faces whose normal is along +-axis (0,1,2)"""
    out = []
    for f in p['planes']:
        n = np.array(f['normal'])
        if abs(abs(n[axis]) - 1) < tol and f['area'] >= amin:
            if sign is None or np.sign(n[axis]) == sign:
                out.append((round(float(np.sign(n[axis]))), float(f['centre'][axis]), float(f['area']), [round(c, 3) for c in f['centre']], f['face']))
    return sorted(out, key=lambda r: r[1])
def cyls(p, rmin=0, rmax=1e9):
    out = []
    for c in p['cylinders']:
        if rmin <= c['r'] <= rmax:
            out.append(c)
    return out
