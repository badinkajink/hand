"""The data block of this task's report, from inventory.json and summary.json."""
import json
OUT = '/home/humanoid/Programs/hand/logs/20261104-cad_v3/parts/'
rows = json.load(open(OUT + 'inventory.json'))
S = json.load(open(OUT + 'summary.json'))
counts = {}
for f in ['hand', 'gantry', 'gantry_thumb', 'base_plate', 'robot_mount', 'adaptor']:
    c = {}
    for r in rows:
        if r['file'] == f:
            c[r['kind']] = c.get(r['kind'], 0) + 1
    counts[f] = dict(sorted(c.items()))
counts['hand_by_body'] = {}
for r in rows:
    if r['file'] == 'hand':
        counts['hand_by_body'][r['body']] = counts['hand_by_body'].get(r['body'], 0) + 1
counts['hand_by_body'] = dict(sorted(counts['hand_by_body'].items()))
fg = {}
NAME = {'gantry_v3 <1>': 'index', 'gantry_v3 <2>': 'middle', 'gantry_v3_thumb <1>': 'thumb'}
for inst, v in S['finger_gantry'].items():
    fg[inst] = dict(sub_assembly=inst, gantry=v['gantry'], finger_inferred=NAME[v['gantry']], y_lead_screw=v['y_screw'],
                    y_nut_in_finger=v['nut'], nut_bore_offset_from_screw_axis_mm=0.574, nut_tilt_deg=0.926)
NOTE = {'gantry/007': 'x-axis stepper mount on the -x_G side', 'gantry/008': 'x-axis stepper mount on the +x_G side',
        'gantry/014': 'x-axis coupler (+x_G side)', 'gantry/015': 'x-axis NEMA 8 (+x_G side)',
        'gantry/020': 'x-axis motor shaft (+x_G side)', 'gantry/018': 'y lead-screw nut',
        'gantry_thumb/011': 'y lead-screw nut'}
REPL = {('gantry_v3 <1>', 'gantry/018'): 'replaced by Finger v3 <2> hand/085 on the same screw',
        ('gantry_v3 <2>', 'gantry/018'): 'replaced by Finger v3 <1> hand/036 on the same screw',
        ('gantry_v3_thumb <1>', 'gantry_thumb/011'): 'replaced by Finger v3 <3> hand/017 on the same screw',
        ('gantry_v3 <2>', 'gantry/014'): 'the -x_A side coupler hand/003 sits at the top level of the hand file',
        ('gantry_v3 <2>', 'gantry/015'): 'the -x_A side NEMA 8 hand/062 sits at the top level of the hand file',
        ('gantry_v3 <2>', 'gantry/020'): 'no motor shaft solid on the -x_A side either'}
miss = {}
for inst, lst in S['missing'].items():
    miss[inst] = [dict(part=m['part'], kind=m['kind'], what=NOTE[m['part']],
                       **({'replacement': REPL[(inst, m['part'])]} if (inst, m['part']) in REPL else {})) for m in lst]
side = {
    'gantry_v3 <1>': dict(mount_used='gantry/008 (+x_G side) = hand/071', side_in_A='+x_A (outer side, away from gantry_v3 <2>)',
                          mount_x_A_mm=[0.0, 56.8], stepper='hand/077 at x_A 19.2..47.8', coupler='hand/070', shaft='hand/075',
                          absent='gantry/007 (-x_G side)'),
    'gantry_v3 <2>': dict(mount_used='gantry/007 (-x_G side) = hand/118', side_in_A='-x_A (outer side, away from gantry_v3 <1>)',
                          mount_x_A_mm=[-278.8, -222.0], stepper='hand/062 at x_A -269.8..-241.2 (top level)',
                          coupler='hand/003 (top level)', shaft='none in the hand file',
                          absent='gantry/008 (+x_G side) with its stepper, coupler and shaft'),
    'gantry_v3_thumb <1>': dict(mount_used='gantry_thumb/001 (the file has one, +x_G side) = hand/060',
                                side_in_A='-x_A (the instance is rotated 180 deg about z_A)', mount_x_A_mm=[-247.8, -191.0],
                                stepper='hand/051 at x_A -238.8..-210.2', coupler='hand/061', shaft='hand/054'),
}
FEAT = {
    '000': ('4 mm plate at z_A -4..0 under the base plate; 4.4 mm holes at (21.3, 230.38), (-1.5, 173.8) and (6.5, 117.0) '
            'mm in A coincide with base-plate holes (the last two also pass the robot mount); it reaches y_A 295.5, 52.6 mm '
            'past the base plate edge (242.9), where two ridges to z_A -10 leave a 20.4 mm slot (y_A 271.1..291.5) with a '
            '5.4 mm hole at (15.8, 281.5). The D435 or its bracket mounts here (inferred; no camera solid in the export)'),
    '005': ('2 mm plate whose 45 x 50 mm face (x_A 0.1..45.1, z_A -57..-7) faces -y_A at y_A 103.3, with no holes in the '
            'face; hangs from 000 by a 3.4 mm hole at (6.5, 117.0) and rises to z_A 18 along the +x_A faces of the base '
            'plate and robot mount (x_A 11.0; its wall starts at 11.2). AprilTag '
            'plate (inferred from the plain face)'),
    '001': 'robot mount, z_A 3..28 on the base plate (z_A 0..3), with the adaptor pocket',
    '002': 'base plate, z_A 0..3; the gantry_v3 <1> and <2> frames bolt under it (4 shared 4.4 mm holes each)',
    '004': 'UR5 adaptor in the robot mount pocket, flange face z_A 28; four M6 on a 50 mm circle centred (-111.0, 116.0)',
    '003': 'x-axis coupler of gantry_v3 <2> on its -x_A side (outside the gantry sub-assembly)',
    '062': 'x-axis NEMA 8 of gantry_v3 <2> on its -x_A side (outside the gantry sub-assembly)',
}
top = []
for r in rows:
    if r['file'] == 'hand' and r['sub_assembly'] is None:
        top.append(dict(part='hand/' + r['name'], label=r['label'], kind=r['kind'], component=r['component'],
                        mass_g=r['mass_g'], bbox_mm=r['bbox_mm'], what=FEAT[r['name']]))
data = dict(counts_by_component=counts, gantry_instance_of_finger=fg, missing_from_hand_instances=miss,
            real_stepper_mount_side=side, top_level_parts=top,
            gantry_instance_transforms_A_from_file=S['instance_transforms'],
            x_stage_offset_hand_vs_gantry_file_mm={k: v[0] for k, v in S['stage_offsets_mm'].items()},
            y_stage_offset_hand_vs_gantry_file_mm={'gantry_v3 <1>': [-29.06, 15.82], 'gantry_v3 <2>': [6.06, 12.77],
                                                   'gantry_v3_thumb <1>': [-11.33, 1.53]},
            joint_axes_A=S['joint_axes_A'], body_mass_g=S['body_mass_g'],
            hand_mass_total_g=round(sum(r['mass_g'] for r in rows if r['file'] == 'hand'), 1))
json.dump(data, open(OUT + 'data.json', 'w'), indent=1)
print(json.dumps(data)[:3000])
