"""One finger (Finger v3 <2>, on gantry_v3 <1>) coloured by link, every solid labelled."""
import sys, json
sys.path.insert(0, '/home/humanoid/Programs/hand/logs/20261104-cad_v3/parts')
import numpy as np
from render_lib import BY_KEY, render_views
OUT = '/home/humanoid/Programs/hand/logs/20261104-cad_v3/parts/'
inst = sys.argv[1] if len(sys.argv) > 1 else 'Finger v3 <2>'
rows = json.load(open(OUT + 'inventory.json'))
LINK_COL = {'carriage': (0.60, 0.45, 0.75), 'yaw_link': (0.45, 0.75, 0.40), 'mcp_link': (0.35, 0.55, 0.85),
            'dip_link': (0.95, 0.70, 0.35), 'tip': (0.85, 0.20, 0.25)}
SHORT = {'finger_carriage': 'carriage', 'yaw_link': 'yaw link', 'mcp_link': 'MCP link', 'dip_link': 'DIP link',
         'fingertip_tpu_block': 'TPU tip', 'scs0009_servo': 'SCS0009', 'horn_disc': 'disc horn', 'horn_arm': 'arm horn',
         'joint_shaft': 'keyed shaft', 'mr105_bearing': 'MR105', 'lead_nut': 'y nut'}
items, labels = [], []
for r in rows:
    if r['file'] != 'hand' or r['sub_assembly'] != inst:
        continue
    link = r['body'].split()[-1]
    c = LINK_COL[link]
    big = r['kind'] in ('finger_carriage', 'yaw_link', 'mcp_link')
    items.append(dict(key='hand/' + r['name'], rgba=(c[0], c[1], c[2], 0.45 if big else 1.0)))
    if r['kind'] in ('scs0009_servo',):
        items[-1]['rgba'] = (0.12, 0.20, 0.45, 1)
    j = f" ({r['joint']})" if r['joint'] else ''
    labels.append((f"{r['name']} {SHORT[r['kind']]}{j}", r['com_mm']))
b = np.array([BY_KEY[i['key']]['bbox'] for i in items]); ctr = (b[:, :3].min(0) + b[:, 3:].max(0)) / 2000
views = [dict(lookat=ctr, distance=0.26, azimuth=a, elevation=e, title=f'az {a}, el {e}') for a, e in [(0, -10), (90, -10), (180, -10), (270, -10)]]
legend = [('carriage', LINK_COL['carriage']), ('yaw link', LINK_COL['yaw_link']), ('MCP link', LINK_COL['mcp_link']), ('DIP link', LINK_COL['dip_link']),
          ('tip', LINK_COL['tip']), ('SCS0009 (in the parent link)', (0.12, 0.20, 0.45))]
tag = inst.replace(' ', '_').replace('<', '').replace('>', '')
render_views(items, views, OUT + f'finger_links_{tag}.png', title=f'{inst}: solids by link (large links translucent; label: solid, component, joint)', labels=labels, W=900, H=900, ncols=2, legend=legend, label_size=7)
print('ok')
