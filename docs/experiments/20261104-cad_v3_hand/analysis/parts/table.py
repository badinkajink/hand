import json, numpy as np, sys
d = json.load(open('/home/humanoid/Programs/hand/logs/20261104-cad_v3/extract/parts.json'))
p = d['parts']
for x in p:
    b = np.array(x['bbox']); dims = b[3:] - b[:3]; c = (b[3:] + b[:3]) / 2
    sub = x['chain'][0] if len(x['chain']) > 1 else '-'
    radii = sorted(set(round(cc['r'], 2) for cc in x['cylinders']))
    print(f"{x['file'][:6]:6s} {x['name']} {sub:20s} vol={x['volume']:8.0f} dims={dims[0]:6.1f}x{dims[1]:6.1f}x{dims[2]:6.1f} ctr=({c[0]:7.1f},{c[1]:7.1f},{c[2]:7.1f}) com=({x['com'][0]:7.1f},{x['com'][1]:7.1f},{x['com'][2]:7.1f}) nf={x['nfaces']:3d}")
