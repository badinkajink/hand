import json, numpy as np
P = json.load(open('/home/humanoid/Programs/hand/logs/20261104-cad_v3/extract/parts.json'))
parts = [p for p in P['parts'] if p['file'] == 'hand']
for p in parts:
    b = np.array(p['bbox']).reshape(2, 3) if not isinstance(p['bbox'][0], list) else np.array(p['bbox'])
    print(f"{p['name']} {p['label']:>9} {' / '.join(p['chain'][1:]) or '-':>28} vol {p['volume']:9.0f} "
          f"com ({p['com'][0]:7.1f},{p['com'][1]:7.1f},{p['com'][2]:7.1f}) "
          f"bb x[{b[0,0]:7.1f},{b[1,0]:7.1f}] y[{b[0,1]:6.1f},{b[1,1]:6.1f}] z[{b[0,2]:6.1f},{b[1,2]:6.1f}] cyl {len(p['cylinders'])}")
