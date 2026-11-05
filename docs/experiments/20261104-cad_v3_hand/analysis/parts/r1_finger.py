import sys
sys.path.insert(0, '/home/humanoid/Programs/hand/logs/20261104-cad_v3/parts')
import numpy as np
from render_lib import PARTS, BY_KEY, render_views, centre_mm
import matplotlib.cm as cm
OUT = '/home/humanoid/Programs/hand/logs/20261104-cad_v3/parts/'
inst = sys.argv[1] if len(sys.argv) > 1 else 'Finger v3 <3>'
keys = [f"hand/{p['name']}" for p in PARTS if p['file'] == 'hand' and p['chain'][0] == inst]
cmap = cm.get_cmap('tab20')
items = []
for i, k in enumerate(keys):
    v = BY_KEY[k]['volume']
    c = cmap(i % 20)
    a = 0.35 if v > 9000 else 1.0
    items.append(dict(key=k, rgba=(c[0], c[1], c[2], a)))
ctr, ext = centre_mm(keys)
L = ctr / 1000
dist = float(np.linalg.norm(ext)) / 1000 * 1.6
views = [dict(lookat=L, distance=dist, azimuth=az, elevation=el, title=f'az {az} el {el}') for az, el in [(0, -15), (90, -15), (180, -15), (270, -15), (45, 40), (225, -50)]]
labels = [(f"{BY_KEY[k]['name']} v{BY_KEY[k]['volume']:.0f}", BY_KEY[k]['com']) for k in keys]
render_views(items, views, OUT + f"r1_{inst.replace(' ', '_').replace('<','').replace('>','')}.png", title=inst + ' (large parts translucent)', labels=labels, W=900, H=900, ncols=3)
print('ok')
