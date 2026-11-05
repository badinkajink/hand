"""Mesh volumes of the extracted STLs against the B-rep volumes, and the volumes of assets/cad_v1 parts."""
import json, numpy as np, trimesh, sys
from pathlib import Path
R = Path('/home/humanoid/Programs/hand')
d = json.load(open(R / 'logs/20261104-cad_v3/extract/parts.json'))
out = {}
for x in d['parts']:
    m = trimesh.load(R / x['stl'], force='mesh')
    out[f"{x['file']}/{x['name']}"] = dict(mesh_vol=float(m.volume), watertight=bool(m.is_watertight),
                                           brep_vol=x['volume'], nf=len(m.faces))
bad = {k: v for k, v in out.items() if abs(v['mesh_vol'] - v['brep_vol']) > 0.02 * v['brep_vol'] or not v['watertight']}
print('parts whose mesh volume differs >2% from the B-rep or is not watertight:')
for k, v in bad.items():
    print(k, v)
cad = {}
for f in sorted((R / 'assets/cad_v1').glob('*.stl')):
    m = trimesh.load(f, force='mesh')
    e = m.bounding_box_oriented.primitive.extents
    cad[f.name] = dict(vol=float(m.volume), watertight=bool(m.is_watertight), extents_obb=sorted(map(float, e)))
    print(f.name, cad[f.name])
for f in sorted((R / 'assets/mjcf/experimental/20261013-cad_v1/meshes').glob('*.stl')):
    m = trimesh.load(f, force='mesh')
    print('cad_v1 mjcf', f.name, round(float(m.volume), 1), m.is_watertight, sorted(np.round(m.bounding_box_oriented.primitive.extents, 2)))
json.dump(dict(mesh=out, cad_v1=cad), open(R / 'logs/20261104-cad_v3/parts/measure.json', 'w'), indent=1)
