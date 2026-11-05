"""Plan and side schematics of the hand file coloured by rigid body (inventory.json only, no meshes)."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
OUT = '/home/humanoid/Programs/hand/logs/20261104-cad_v3/parts/'
rows = [r for r in json.load(open(OUT + 'inventory.json')) if r['file'] == 'hand']
def group(rb):
    if rb == 'palm':
        return 'palm (fixed)'
    if rb.endswith('x stage'):
        return 'x stage'
    if 'y stage' in rb:
        return 'y stage + finger carriage'
    return rb.split()[-1]
COL = {'palm (fixed)': '#b8bcc6', 'x stage': '#3a9e86', 'y stage + finger carriage': '#e08a2e', 'yaw_link': '#73bf66',
       'mcp_link': '#598cd9', 'dip_link': '#f2c14e', 'tip': '#d93340'}
ORDER = list(COL)
fig, axs = plt.subplots(1, 2, figsize=(17, 9.0), dpi=130, gridspec_kw=dict(width_ratios=[1.25, 1]))
for i, j, title, ax in [(0, 1, 'Plan view from above: x_A right, y_A up (mm)', axs[0]), (0, 2, 'Side view from -y_A: x_A right, z_A up (mm)', axs[1])]:
    if j == 1:
        rs = sorted(rows, key=lambda r: (ORDER.index(group(r['rigid_body'])), r['bbox_mm'][5]))
    else:
        rs = sorted(rows, key=lambda r: -r['bbox_mm'][4])
    for r in rs:
        b = r['bbox_mm']
        big = r['kind'] in ('base_plate', 'robot_mount')
        ax.add_patch(Rectangle((b[i], b[j]), b[3 + i] - b[i], b[3 + j] - b[j], fc=COL[group(r['rigid_body'])], ec='k', lw=0.35,
                               alpha=0.3 if big else 0.9))
    ax.set_aspect('equal'); ax.autoscale_view(); ax.grid(True, lw=0.3, alpha=0.5); ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=9)
# labels: one per rigid body at the centre of its solids' boxes (plan view)
from collections import defaultdict
acc = defaultdict(list)
for r in rows:
    acc[r['rigid_body']].append(r['bbox_mm'])
NAME = {'gantry_v3 <1>': 'gantry_v3 <1>', 'gantry_v3 <2>': 'gantry_v3 <2>', 'gantry_v3_thumb <1>': 'thumb gantry'}
for rb, bs in acc.items():
    if rb == 'palm':
        continue
    import numpy as np
    bs = np.array(bs)
    lo, hi = bs[:, :3].min(0), bs[:, 3:].max(0)
    if rb.endswith('x stage'):
        g = rb.replace(' x stage', '')
        txt, xy = f'{g}\nx stage', ((lo[0] + hi[0]) / 2, lo[1] + 8)
    elif 'y stage' in rb:
        txt, xy = rb.split(' + ')[1].replace(' carriage', '') + '\ny stage + carriage', ((lo[0] + hi[0]) / 2, hi[1] + 9)
    else:
        continue
    axs[0].text(xy[0], xy[1], txt, fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='0.4', lw=0.4, alpha=0.85))
axs[0].set_xlabel('x_A (mm)'); axs[0].set_ylabel('y_A (mm)'); axs[1].set_xlabel('x_A (mm)'); axs[1].set_ylabel('z_A (mm)')
h = [plt.Line2D([], [], ls='', marker='s', ms=10, mfc=COL[k], mec='0.3', label={'mcp_link': 'MCP link', 'dip_link': 'DIP link', 'yaw_link': 'yaw link'}.get(k, k)) for k in ORDER]
fig.legend(handles=h, loc='lower center', ncol=7, fontsize=10, frameon=False)
fig.suptitle('hand v3 (anthro config): the 19 rigid bodies of the hand file, each solid drawn as its bounding box in frame A', fontsize=13)
fig.tight_layout(rect=(0, 0.06, 1, 0.96))
fig.savefig(OUT + 'plan_side_rigid_bodies.png')
print('ok')
