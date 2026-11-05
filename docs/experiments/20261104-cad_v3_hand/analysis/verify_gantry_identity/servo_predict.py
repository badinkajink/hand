import json, numpy as np
P = json.load(open('/home/humanoid/Programs/hand/logs/20261104-cad_v3/extract/parts.json'))
H = [p for p in P['parts'] if p['file'] == 'hand']
fingers = {}
for p in H:
    if len(p['chain']) > 1 and p['chain'][0].startswith('Finger'):
        fingers.setdefault(p['chain'][0], []).append(p)
def unit(v): v = np.asarray(v, float); return v / np.linalg.norm(v)
def closest_point_on_line(p0, d, q):
    return p0 + np.dot(q - p0, d) * d
measured = {'thumb': +1, 'index': -1, 'middle': -1}   # plan.JOINT_SIGN yaw, "toward +y_P (index side)"
res = {}
for fname in sorted(fingers):
    parts = fingers[fname]
    byvol = lambda v: [p for p in parts if abs(p['volume'] - v) < 2]
    tip = byvol(4822)[0]; carriage = byvol(23913)[0]
    servos = []
    for s in byvol(6673):
        cyl = {round(c['r'], 2): c for c in s['cylinders'] if not c['concave']}
        a = unit(cyl[2.0]['axis'])
        sp = np.array(cyl[2.0]['point']) + a * np.mean(cyl[2.0]['t'])
        bo = np.array(cyl[6.05]['point']) + a * np.mean(cyl[6.05]['t'])
        n_out = a * np.sign(np.dot(sp - bo, a))      # out of the body through the spline
        servos.append(dict(name=s['name'], com=np.array(s['com']), axis_pt=sp, n_out=n_out))
    bore = [c for c in carriage['cylinders'] if abs(c['r'] - 5.1) < 0.05 and c['concave']][0]
    yaw_dir = unit(bore['axis'])
    aa = [s for s in servos if abs(np.dot(s['n_out'], yaw_dir)) > 0.99][0]
    fe = sorted([s for s in servos if s is not aa], key=lambda s: -s['com'][2])
    mcp, pip = fe
    tipc = np.array(tip['com'])
    # palmar direction: the pair flexes toward +y_A (thumb side), the thumb toward -y_A
    is_thumb = tipc[1] > 120
    palmar = np.array([0, -1.0, 0]) if is_thumb else np.array([0, 1.0, 0])
    # c from MCP: positive command = flexion. Horn on the child (MCP link), body on the parent (yaw link).
    # child angular velocity = c * n_out ; tip velocity = w x (tip - joint point)
    mcp_pt = closest_point_on_line(mcp['axis_pt'], mcp['n_out'], tipc)
    v_mcp = np.cross(mcp['n_out'], tipc - mcp_pt)
    c_mcp = np.sign(np.dot(v_mcp, palmar))          # c such that c*n_out flexes
    pip_pt = closest_point_on_line(pip['axis_pt'], pip['n_out'], tipc)
    v_pip = np.cross(pip['n_out'], tipc - pip_pt)
    c_pip = np.sign(np.dot(v_pip, palmar))
    # aa: body on carriage, horn on yaw link; positive command -> w = c * n_out
    yaw_pt = closest_point_on_line(aa['axis_pt'], aa['n_out'], tipc)
    v_aa = np.cross(c_mcp * aa['n_out'], tipc - yaw_pt)
    res[fname] = dict(is_thumb=bool(is_thumb), c_mcp=float(c_mcp), c_pip=float(c_pip),
                      aa_n_out=np.round(aa['n_out'], 4).tolist(), mcp_n_out=np.round(mcp['n_out'], 4).tolist(),
                      pip_n_out=np.round(pip['n_out'], 4).tolist(),
                      tip_velocity_dir_positive_aa=np.round(unit(v_aa), 3).tolist())
    print(f"{fname}: thumb={is_thumb} aa n_out {np.round(aa['n_out'],4)} mcp n_out {np.round(mcp['n_out'],4)} "
          f"pip n_out {np.round(pip['n_out'],4)}\n    c from MCP {c_mcp:+.0f}, from PIP {c_pip:+.0f}; "
          f"positive aa swings the tip along {np.round(unit(v_aa),3)} (x_A component {unit(v_aa)[0]:+.3f})")
# hypotheses
print()
for hyp, idx_side, names in [('H1 (claim): index = Finger v3 <2> on gantry_v3 <1>, index side = +x_A', +1,
                              {'Finger v3 <2>': 'index', 'Finger v3 <1>': 'middle', 'Finger v3 <3>': 'thumb'}),
                             ('H2 (mirror): index = Finger v3 <1> on gantry_v3 <2>, index side = -x_A', -1,
                              {'Finger v3 <1>': 'index', 'Finger v3 <2>': 'middle', 'Finger v3 <3>': 'thumb'})]:
    print(hyp)
    ok = 0
    for fname, r in res.items():
        pred = int(np.sign(r['tip_velocity_dir_positive_aa'][0] * idx_side))
        role = names[fname]
        ok += pred == measured[role]
        print(f"   {role:6s} ({fname}): predicted sign {pred:+d}, measured {measured[role]:+d}")
    print(f"   agreement {ok}/3")
json.dump(res, open('/home/humanoid/Programs/hand/logs/20261104-cad_v3/verify_gantry_identity/servo_predict.json', 'w'), indent=1)
