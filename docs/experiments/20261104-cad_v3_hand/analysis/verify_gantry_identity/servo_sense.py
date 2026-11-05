"""Predict the aa (yaw) sign of each finger from the CAD servo orientations.

For each finger: find the three 6673 mm^3 servo bodies, their output spline (convex r=2.0 cylinder)
and its outward direction n_out (from the r=6.05 boss toward the spline tip). Decide which link each
servo body is fixed to by comparing its mounting-hole direction with the links' frames. Calibrate the
servo's intrinsic sense c from fe1/fe2 (positive command = flexion, JOINT_SIGN +1, no per-servo sign in
servos.py), then predict the direction a positive aa command swings each fingertip, in A.
"""
import json, numpy as np
P = json.load(open('/home/humanoid/Programs/hand/logs/20261104-cad_v3/extract/parts.json'))
H = [p for p in P['parts'] if p['file'] == 'hand']
fingers = {}
for p in H:
    if len(p['chain']) > 1 and p['chain'][0].startswith('Finger'):
        fingers.setdefault(p['chain'][0], []).append(p)
def unit(v): v = np.asarray(v, float); return v / np.linalg.norm(v)
out = {}
for fname in sorted(fingers):
    parts = fingers[fname]
    byvol = lambda v: [p for p in parts if abs(p['volume'] - v) < 2]
    carriage = byvol(23913)[0]; yawlink = byvol(10683)[0]; mcplink = byvol(21293)[0]
    tip = byvol(4822)[0]
    servos = byvol(6673)
    # yaw bore of the carriage: concave r 5.1 cylinder
    bores = [c for c in carriage['cylinders'] if abs(c['r'] - 5.1) < 0.05 and c['concave']]
    yaw_axis = unit(bores[0]['axis'])
    rec = {'tip_com': tip['com'], 'servos': []}
    for s in servos:
        cyl = {round(c['r'], 2): c for c in s['cylinders'] if not c['concave']}
        boss, spline = cyl[6.05], cyl[2.0]
        a = unit(spline['axis'])
        # spline segment and boss segment along a
        sp0 = np.array(spline['point']) + a * spline['t'][0]; sp1 = np.array(spline['point']) + a * spline['t'][1]
        bo0 = np.array(boss['point']) + a * boss['t'][0]; bo1 = np.array(boss['point']) + a * boss['t'][1]
        sp_mid = (sp0 + sp1) / 2; bo_mid = (bo0 + bo1) / 2
        n_out = unit(sp_mid - bo_mid)           # boss -> spline tip = out of the servo
        holes = [np.array(c['point']) for c in s['cylinders'] if c['concave'] and abs(c['r'] - 1.15) < 0.01]
        long_axis = unit(holes[1] - holes[0])
        # role: shaft parallel to yaw axis -> aa servo; otherwise fe servo (mcp if higher)
        role = 'aa' if abs(np.dot(a, yaw_axis)) > 0.99 else 'fe'
        rec['servos'].append(dict(name=s['name'], com=np.round(s['com'], 2).tolist(), role=role,
                                  n_out=np.round(n_out, 4).tolist(), long_axis=np.round(long_axis, 4).tolist(),
                                  spline_tip=np.round(sp1 if np.dot(sp1 - sp0, n_out) > 0 else sp0, 2).tolist()))
    # sort fe servos by height: higher = mcp
    fe = sorted([r for r in rec['servos'] if r['role'] == 'fe'], key=lambda r: -r['com'][2])
    fe[0]['role'] = 'mcp'; fe[1]['role'] = 'pip'
    rec['yaw_axis_bore'] = np.round(yaw_axis, 5).tolist()
    out[fname] = rec
    print(f"== {fname}: carriage {carriage['name']} com {np.round(carriage['com'],1)} yaw bore axis {np.round(yaw_axis,4)}")
    for r in rec['servos']:
        print(f"   servo {r['name']} role {r['role']:>4} com {r['com']} n_out {r['n_out']} long_axis(holes) {r['long_axis']}")
json.dump(out, open('/home/humanoid/Programs/hand/logs/20261104-cad_v3/verify_gantry_identity/servo_sense.json', 'w'), indent=1)
