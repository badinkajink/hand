"""Overview renders of the hand file coloured by component, with the instances and the top-level parts labelled."""
import sys, json
sys.path.insert(0, '/home/humanoid/Programs/hand/logs/20261104-cad_v3/parts')
import numpy as np
from render_lib import BY_KEY, render_views
OUT = '/home/humanoid/Programs/hand/logs/20261104-cad_v3/parts/'
rows = json.load(open(OUT + 'inventory.json'))
COL = {
    'robot_mount': (0.62, 0.62, 0.66, 1), 'base_plate': (0.80, 0.80, 0.82, 1), 'ur5_adaptor': (0.95, 0.55, 0.15, 1),
    'gantry_frame': (0.55, 0.75, 0.95, 1), 'gantry_frame_thumb': (0.55, 0.75, 0.95, 1),
    'x_stage_beam': (0.30, 0.65, 0.60, 1), 'x_stage_beam_thumb': (0.30, 0.65, 0.60, 1),
    'x_stepper_mount': (0.45, 0.55, 0.85, 1), 'x_stepper_mount_thumb': (0.45, 0.55, 0.85, 1),
    'nema8_stepper': (0.15, 0.15, 0.18, 1), 'stepper_shaft': (0.7, 0.7, 0.7, 1),
    'shaft_coupler': (0.85, 0.75, 0.30, 1), 'lead_screw_100': (0.75, 0.70, 0.55, 1), 'lead_screw_150': (0.75, 0.70, 0.55, 1),
    'lead_nut': (0.80, 0.60, 0.20, 1), 'mgn7_rail_100': (0.55, 0.55, 0.55, 1), 'mgn7_rail_150': (0.55, 0.55, 0.55, 1),
    'mgn7_carriage': (0.35, 0.35, 0.40, 1), 'mr105_bearing': (0.9, 0.2, 0.2, 1), 'ring_5x10x5': (0.9, 0.2, 0.2, 1),
    'scs0009_servo': (0.12, 0.20, 0.45, 1), 'horn_disc': (0.98, 0.98, 0.98, 1), 'horn_arm': (0.98, 0.98, 0.98, 1),
    'joint_shaft': (0.98, 0.98, 0.98, 1), 'finger_carriage': (0.60, 0.45, 0.75, 1), 'yaw_link': (0.45, 0.75, 0.40, 1),
    'mcp_link': (0.35, 0.55, 0.85, 1), 'dip_link': (0.95, 0.70, 0.35, 1), 'fingertip_tpu_block': (0.85, 0.20, 0.25, 1),
    'sensor_arm': (0.85, 0.25, 0.75, 1), 'sensor_bracket': (0.85, 0.25, 0.75, 1),
}
hand = [r for r in rows if r['file'] == 'hand']
items = [dict(key=f"hand/{r['name']}", rgba=COL[r['kind']]) for r in hand]
def com(name):
    return next(r for r in hand if r['name'] == name)['com_mm']
labels = [('001 robot mount', com('001')), ('002 base plate', [57.0 - 30, 230, 1.5]), ('004 UR5 adaptor', com('004')),
          ('000 sensor arm', com('000')), ('005 sensor bracket', com('005')), ('062 NEMA 8 (top level)', com('062')),
          ('003 coupler (top level)', com('003'))]
for inst, frame in [('gantry_v3 <1>', '074'), ('gantry_v3 <2>', '102'), ('gantry_v3_thumb <1>', '045')]:
    labels.append((inst, com(frame)))
for f, car in [('Finger v3 <1>', '028'), ('Finger v3 <2>', '094'), ('Finger v3 <3>', '016')]:
    c = com(car); labels.append((f, [c[0], c[1], c[2] - 40]))
b = np.array([r['bbox_mm'] for r in hand]); lo, hi = b[:, :3].min(0), b[:, 3:].max(0)
ctr = (lo + hi) / 2000
views = [dict(lookat=ctr, distance=0.75, azimuth=-60, elevation=-30, title='from below, az -60'),
         dict(lookat=ctr, distance=0.75, azimuth=120, elevation=-30, title='from below, az 120'),
         dict(lookat=ctr, distance=0.75, azimuth=-90, elevation=-89.9, title='from below (looking up +z_A); x_A right, y_A down'),
         dict(lookat=ctr, distance=0.75, azimuth=-90, elevation=89.9, title='from above (looking down -z_A); x_A right, y_A up')]
legend = [('robot mount / base plate', COL['robot_mount']), ('UR5 adaptor', COL['ur5_adaptor']), ('gantry frame', COL['gantry_frame']),
          ('x-stage beam', COL['x_stage_beam']), ('x stepper mount', COL['x_stepper_mount']), ('NEMA 8', COL['nema8_stepper']),
          ('MGN7 rail / carriage', COL['mgn7_rail_100']), ('lead screw / nut / coupler', COL['lead_nut']),
          ('bearing / ring', COL['mr105_bearing']), ('SCS0009', COL['scs0009_servo']), ('finger carriage', COL['finger_carriage']),
          ('yaw link', COL['yaw_link']), ('MCP link', COL['mcp_link']), ('DIP link', COL['dip_link']), ('TPU tip', COL['fingertip_tpu_block']),
          ('sensor mounts', COL['sensor_arm'])]
render_views(items, views, OUT + 'overview_hand_by_component.png', title='hand v3 (anthro config): solids coloured by component, frame A (mm)', labels=labels, W=1000, H=800, ncols=2, legend=legend, label_size=8)
print('ok')
