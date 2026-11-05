"""The two top-level sensor mounts (hand/000, hand/005) in context, and the top-level x drive of gantry_v3 <2>."""
import sys, json
sys.path.insert(0, '/home/humanoid/Programs/hand/logs/20261104-cad_v3/parts')
import numpy as np
import trimesh
from render_lib import BY_KEY, render_views, R
OUT = '/home/humanoid/Programs/hand/logs/20261104-cad_v3/parts/'
ctx = {'hand/002': (0.82, 0.82, 0.85, 0.45), 'hand/001': (0.62, 0.62, 0.66, 0.45), 'hand/074': (0.55, 0.75, 0.95, 0.6),
       'hand/045': (0.55, 0.75, 0.95, 0.6), 'hand/071': (0.45, 0.55, 0.85, 1), 'hand/077': (0.15, 0.15, 0.18, 1),
       'hand/064': (0.55, 0.55, 0.55, 1)}
items = [dict(key=k, rgba=c) for k, c in ctx.items()] + [dict(key='hand/000', rgba=(0.85, 0.25, 0.75, 1)), dict(key='hand/005', rgba=(0.95, 0.55, 0.10, 1))]
labels = [('000 sensor arm', BY_KEY['hand/000']['com']), ('005 bracket, 45 x 50 mm face', [22.6, 103.3, -32]),
          ('002 base plate', [40, 60, 1.5]), ('001 robot mount', [-20, 200, 15]), ('077 NEMA 8 (gantry_v3 <1> x)', BY_KEY['hand/077']['com']),
          ('000: 5.4 mm hole, y_A 281.5', [15.8, 281.5, -2])]
ctr = np.array([10.0, 200.0, -15.0]) / 1000
views = [dict(lookat=ctr, distance=0.45, azimuth=-30, elevation=-25, title='from below, az -30'),
         dict(lookat=ctr, distance=0.45, azimuth=60, elevation=-20, title='from below, az 60'),
         dict(lookat=ctr, distance=0.45, azimuth=150, elevation=25, title='from above, az 150'),
         dict(lookat=ctr, distance=0.45, azimuth=-90, elevation=-89.9, title='from below (x_A right, y_A down)')]
legend = [('000 sensor arm', (0.85, 0.25, 0.75, 1)), ('005 bracket', (0.95, 0.55, 0.10, 1)), ('base plate / robot mount (translucent)', (0.7, 0.7, 0.75, 1)),
          ('gantry frames', (0.55, 0.75, 0.95, 1))]
render_views(items, views, OUT + 'sensor_mounts_context.png', title='Top-level solids 000 and 005 at the +x_A edge of the hand (frame A, mm)', labels=labels, W=1000, H=800, ncols=2, legend=legend, label_size=8)
# interference of 005 and 000 with the base plate and robot mount, by sampled surface points (a few thousand)
res = {}
for a in ['hand/005', 'hand/000']:
    ma = trimesh.load(R / BY_KEY[a]['stl'], force='mesh')
    pts, _ = trimesh.sample.sample_surface_even(ma, 3000, seed=1)
    for b in ['hand/002', 'hand/001', 'hand/074']:
        mb = trimesh.load(R / BY_KEY[b]['stl'], force='mesh')
        sd = trimesh.proximity.signed_distance(mb, pts)      # positive inside mb
        res[f'{a} in {b}'] = dict(points=len(pts), inside_over_0p05mm=int((sd > 0.05).sum()), max_depth_mm=round(float(sd.max()), 3))
print(json.dumps(res, indent=1))
json.dump(res, open(OUT + 'sensor_interference.json', 'w'), indent=1)
