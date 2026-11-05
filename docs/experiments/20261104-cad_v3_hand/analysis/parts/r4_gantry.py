"""gantry_v3 <1> in the hand file coloured by moving body, and gantry_v3.step with the solids absent from each instance
of the hand file outlined in red (translucent)."""
import sys, json
sys.path.insert(0, '/home/humanoid/Programs/hand/logs/20261104-cad_v3/parts')
import numpy as np
from render_lib import BY_KEY, render_views
OUT = '/home/humanoid/Programs/hand/logs/20261104-cad_v3/parts/'
rows = json.load(open(OUT + 'inventory.json'))
summ = json.load(open(OUT + 'summary.json'))
BODY_COL = {'palm (fixed)': (0.70, 0.72, 0.78, 1), 'x_stage': (0.25, 0.65, 0.55, 1), 'y_stage': (0.85, 0.45, 0.20, 1),
            'finger': (0.60, 0.45, 0.75, 0.35)}
SHORT = {'gantry_frame': 'frame', 'x_stage_beam': 'beam', 'x_stepper_mount': 'x mount', 'nema8_stepper': 'NEMA 8',
         'stepper_shaft': 'shaft', 'shaft_coupler': 'coupler', 'lead_screw_100': 'screw 100', 'lead_nut': 'nut',
         'mgn7_rail_100': 'MGN7 rail', 'mgn7_carriage': 'MGN7 car.', 'mr105_bearing': 'MR105', 'ring_5x10x5': 'ring 5x10x5'}
# 1) gantry_v3 <1> with Finger v3 <2>'s carriage and nut
items, labels = [], []
for r in rows:
    if r['file'] != 'hand':
        continue
    if r['sub_assembly'] == 'gantry_v3 <1>':
        b = 'palm (fixed)' if r['body'] == 'palm (fixed)' else r['body'].split()[-1]
        items.append(dict(key='hand/' + r['name'], rgba=BODY_COL[b]))
        labels.append((f"{r['name']} {SHORT.get(r['kind'], r['kind'])}", r['com_mm']))
    elif r['sub_assembly'] == 'Finger v3 <2>' and r['kind'] in ('finger_carriage', 'lead_nut'):
        items.append(dict(key='hand/' + r['name'], rgba=BODY_COL['finger'] if r['kind'] == 'finger_carriage' else (0.85, 0.45, 0.20, 1)))
        labels.append((f"{r['name']} {'finger carriage' if r['kind'] == 'finger_carriage' else 'y nut (Finger v3 <2>)'}", r['com_mm']))
b = np.array([BY_KEY[i['key']]['bbox'] for i in items]); ctr = (b[:, :3].min(0) + b[:, 3:].max(0)) / 2000
views = [dict(lookat=ctr, distance=0.42, azimuth=-55, elevation=-35, title='gantry_v3 <1> from below, az -55'),
         dict(lookat=ctr, distance=0.42, azimuth=125, elevation=-35, title='gantry_v3 <1> from below, az 125'),
         dict(lookat=ctr, distance=0.42, azimuth=-90, elevation=-89.9, title='from below (x_A right, y_A down)'),
         dict(lookat=ctr, distance=0.42, azimuth=0, elevation=0, title='side view along +x_A (y_A left, z_A up)')]
legend = [('palm (fixed to the frame)', BODY_COL['palm (fixed)']), ('x stage', BODY_COL['x_stage']), ('y stage (MGN7 carriage, finger nut)', BODY_COL['y_stage']),
          ('finger carriage (translucent)', BODY_COL['finger'])]
render_views(items, views, OUT + 'gantry_v3_1_bodies.png', title='gantry_v3 <1> in the hand file: solids by moving body (labels: solid name and component)', labels=labels, W=1000, H=800, ncols=2, legend=legend, label_size=7)
# 2) gantry_v3.step: the solids absent from gantry_v3 <1> and <2>
miss1 = {m['part'] for m in summ['missing']['gantry_v3 <1>']}
miss2 = {m['part'] for m in summ['missing']['gantry_v3 <2>']}
items, labels = [], []
for r in rows:
    if r['file'] != 'gantry':
        continue
    k = 'gantry/' + r['name']
    a, b2 = k in miss1, k in miss2
    col = (0.85, 0.15, 0.15, 1) if (a and b2) else (0.95, 0.60, 0.10, 1) if a else (0.20, 0.45, 0.90, 1) if b2 else (0.75, 0.75, 0.78, 0.35)
    items.append(dict(key=k, rgba=col))
    if a or b2:
        labels.append((f"{r['name']} {SHORT.get(r['kind'], r['kind'])}", r['com_mm']))
b = np.array([BY_KEY[i['key']]['bbox'] for i in items]); ctr = (b[:, :3].min(0) + b[:, 3:].max(0)) / 2000
views = [dict(lookat=ctr, distance=0.50, azimuth=-60, elevation=-35, title='gantry_v3.step, from below'),
         dict(lookat=ctr, distance=0.50, azimuth=-90, elevation=89.9, title='gantry_v3.step, from above (x_G right, y_G up)')]
legend = [('absent from both instances', (0.85, 0.15, 0.15, 1)), ('absent from gantry_v3 <1> only', (0.95, 0.60, 0.10, 1)),
          ('absent from gantry_v3 <2> only', (0.20, 0.45, 0.90, 1)), ('present in both (translucent)', (0.75, 0.75, 0.78, 1))]
render_views(items, views, OUT + 'gantry_v3_step_absent_parts.png', title='gantry_v3.step: solids the hand file leaves out of each instance', labels=labels, W=1000, H=800, ncols=2, legend=legend, label_size=8)
print('ok')
