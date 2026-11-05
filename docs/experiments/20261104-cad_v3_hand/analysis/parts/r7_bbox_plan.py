"""Plan and side schematics of the hand file from the solids' bounding boxes (parts.json only, no meshes):
every solid as its axis-aligned box in frame A, coloured by component, the instances and top-level solids labelled."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
OUT = '/home/humanoid/Programs/hand/logs/20261104-cad_v3/parts/'
rows = [r for r in json.load(open(OUT + 'inventory.json')) if r['file'] == 'hand']
COL = {
    'robot_mount': '#9e9ea8', 'base_plate': '#cfcfd4', 'ur5_adaptor': '#f28c26', 'gantry_frame': '#8cbff2',
    'gantry_frame_thumb': '#8cbff2', 'x_stage_beam': '#4da68c', 'x_stage_beam_thumb': '#4da68c',
    'x_stepper_mount': '#738cd9', 'x_stepper_mount_thumb': '#738cd9', 'nema8_stepper': '#26262e', 'stepper_shaft': '#26262e',
    'shaft_coupler': '#d9bf4d', 'lead_screw_100': '#bfb38c', 'lead_screw_150': '#bfb38c', 'lead_nut': '#cc9933',
    'mgn7_rail_100': '#8c8c8c', 'mgn7_rail_150': '#8c8c8c', 'mgn7_carriage': '#595966', 'mr105_bearing': '#e63333',
    'ring_5x10x5': '#e63333', 'scs0009_servo': '#1f3373', 'horn_disc': '#bbbbbb', 'horn_arm': '#bbbbbb', 'joint_shaft': '#bbbbbb',
    'finger_carriage': '#9973bf', 'yaw_link': '#73bf66', 'mcp_link': '#598cd9', 'dip_link': '#f2b359', 'fingertip_tpu_block': '#d93340',
    'sensor_arm': '#d940bf', 'sensor_bracket': '#d940bf'}
ORDER = ['base_plate', 'robot_mount', 'ur5_adaptor', 'gantry_frame', 'gantry_frame_thumb', 'mgn7_rail_150', 'mgn7_rail_100',
         'lead_screw_150', 'lead_screw_100', 'x_stage_beam', 'x_stage_beam_thumb', 'x_stepper_mount', 'x_stepper_mount_thumb',
         'mgn7_carriage', 'lead_nut', 'shaft_coupler', 'nema8_stepper', 'stepper_shaft', 'mr105_bearing', 'ring_5x10x5',
         'finger_carriage', 'yaw_link', 'mcp_link', 'scs0009_servo', 'dip_link', 'fingertip_tpu_block', 'horn_disc', 'horn_arm',
         'joint_shaft', 'sensor_arm', 'sensor_bracket']
LBL = {'001': 'robot mount 001', '004': 'UR5 adaptor 004', '000': 'sensor arm 000', '005': 'bracket 005',
       '062': 'NEMA 8 062\n(top level)', '003': 'coupler 003\n(top level)', '074': 'gantry_v3 <1>\nframe 074',
       '102': 'gantry_v3 <2>\nframe 102', '045': 'gantry_v3_thumb <1>\nframe 045', '028': 'Finger v3 <1>', '094': 'Finger v3 <2>',
       '016': 'Finger v3 <3>', '077': 'NEMA 8 077', '051': 'NEMA 8 051', '072': 'y NEMA 8 072', '103': 'y NEMA 8 103',
       '044': 'y NEMA 8 044', '071': 'mount 071', '118': 'mount 118', '060': 'mount 060', '081': 'beam 081', '120': 'beam 120', '047': 'beam 047'}
OFF = {'062': (0, -22), '118': (0, 22), '003': (8, 22), '051': (-5, 22), '060': (-10, -22), '077': (5, -22), '071': (0, 22)}
fig, axs = plt.subplots(1, 2, figsize=(17, 9.6), dpi=130, gridspec_kw=dict(width_ratios=[1.25, 1]))
views = [(0, 1, 'Plan view from above (+z_A toward the reader): x_A right, y_A up, mm', axs[0]),
         (0, 2, 'Side view from -y_A: x_A right, z_A up, mm', axs[1])]
for i, j, title, ax in views:
    rs = sorted(rows, key=lambda r: (ORDER.index(r['kind']), r['bbox_mm'][2 if j == 1 else 1]))
    if j == 2:
        rs = sorted(rows, key=lambda r: (-r['bbox_mm'][4], ORDER.index(r['kind'])))   # far (large y) first
    for r in rs:
        b = r['bbox_mm']
        lw = 0.4
        fill = r['kind'] not in ('base_plate', 'robot_mount')
        ax.add_patch(Rectangle((b[i], b[j]), b[3 + i] - b[i], b[3 + j] - b[j], fc=COL[r['kind']], ec='k', lw=lw,
                               alpha=0.9 if fill else 0.35))
    for r in rows:
        if r['name'] in LBL:
            b = r['bbox_mm']
            if j == 2 and r['name'] in ('074', '102', '045', '081', '120', '047', '071', '118', '060', '072', '103', '044'):
                continue
            dx, dy = OFF.get(r['name'], (0, 0)) if j == 1 else (0, 0)
            ax.text((b[i] + b[3 + i]) / 2 + dx, (b[j] + b[3 + j]) / 2 + dy, LBL[r['name']], fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='0.4', lw=0.4, alpha=0.85))
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.set_title(title, fontsize=11)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.tick_params(labelsize=9)
axs[0].set_xlabel('x_A (mm)'); axs[0].set_ylabel('y_A (mm)')
axs[1].set_xlabel('x_A (mm)'); axs[1].set_ylabel('z_A (mm)')
seen = []
for k in ORDER:
    name = {'robot_mount': 'robot mount', 'base_plate': 'base plate', 'ur5_adaptor': 'UR5 adaptor', 'gantry_frame': 'gantry frame',
            'x_stage_beam': 'x-stage beam', 'x_stepper_mount': 'x stepper mount', 'nema8_stepper': 'NEMA 8 stepper',
            'shaft_coupler': 'coupler', 'lead_screw_100': 'lead screw', 'lead_nut': 'lead nut', 'mgn7_rail_100': 'MGN7 rail',
            'mgn7_carriage': 'MGN7 carriage', 'mr105_bearing': 'bearing / ring', 'scs0009_servo': 'SCS0009 servo',
            'finger_carriage': 'finger carriage', 'yaw_link': 'yaw link', 'mcp_link': 'MCP link', 'dip_link': 'DIP link',
            'fingertip_tpu_block': 'TPU tip', 'horn_disc': 'horns, keyed shafts', 'sensor_arm': 'sensor mounts'}.get(k)
    if name and name not in seen:
        seen.append(name)
handles = []
for nm in seen:
    k = next(k for k in ORDER if {'robot mount': 'robot_mount', 'base plate': 'base_plate', 'UR5 adaptor': 'ur5_adaptor',
                                  'gantry frame': 'gantry_frame', 'x-stage beam': 'x_stage_beam', 'x stepper mount': 'x_stepper_mount',
                                  'NEMA 8 stepper': 'nema8_stepper', 'coupler': 'shaft_coupler', 'lead screw': 'lead_screw_100',
                                  'lead nut': 'lead_nut', 'MGN7 rail': 'mgn7_rail_100', 'MGN7 carriage': 'mgn7_carriage',
                                  'bearing / ring': 'mr105_bearing', 'SCS0009 servo': 'scs0009_servo', 'finger carriage': 'finger_carriage',
                                  'yaw link': 'yaw_link', 'MCP link': 'mcp_link', 'DIP link': 'dip_link', 'TPU tip': 'fingertip_tpu_block',
                                  'horns, keyed shafts': 'horn_disc', 'sensor mounts': 'sensor_arm'}[nm] == k)
    handles.append(plt.Line2D([], [], ls='', marker='s', ms=9, mfc=COL[k], mec='0.3', label=nm))
fig.legend(handles=handles, loc='lower center', ncol=7, fontsize=9, frameon=False)
fig.suptitle('hand v3 (anthro config): every solid as its bounding box in frame A', fontsize=13)
fig.tight_layout(rect=(0, 0.09, 1, 0.96))
fig.savefig(OUT + 'plan_side_bboxes.png')
print('ok')
