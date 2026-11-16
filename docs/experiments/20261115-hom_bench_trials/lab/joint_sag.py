"""Free-air sag of each finger's yaw (both ways) and MCP (flexion 0-55 deg): one joint at a time, the others straight
(a neighbour swung clear when the moving finger turns toward it), each step checked on the hand model (>= 8 mm between
fingers) before it is sent; commanded and measured in servo degrees. --go to move."""
import sys, time, json
from pathlib import Path
import numpy as np
GO = "--go" in sys.argv
sys.argv = ['x']
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hom_setup as HS
from pathlib import Path
plan = HS.load_plan('/home/will/workspace/hand/docs/experiments/20261115-hom_bench_trials/plans/20261116-film_x105_y70_twist_g10_lat12_plan.json')
st = HS.Station("http://10.99.99.2:8765", str(Path.home() / ".sr2_station_token"))
P = HS._driver_plan()
F = ('thumb', 'index', 'middle')
Z = {f: {k: 0.0 for k in ('yaw', 'mcp', 'pip')} for f in F}
def gap(J):
    return min(np.linalg.norm(p - q) - ra - rb for a, b in (('middle', 'index'), ('middle', 'thumb'), ('index', 'thumb'))
               for (p, ra) in HS.finger_pts(plan, a, J[a]) for (q, rb) in HS.finger_pts(plan, b, J[b]))
def tipy(f, y):
    return HS.finger_pts(plan, f, {'yaw': y, 'mcp': 0.0, 'pip': 0.0})[0][0][1]
seqs = []
for f in F:
    for sgn in (+1, -1):
        seqs.append((f, 'yaw', [sgn * a for a in (10, 20, 30, 40)]))
    seqs.append((f, 'mcp', [15, 30, 45, 55]))
plans = []
for f, jn, steps in seqs:
    base = {g: dict(v) for g, v in Z.items()}
    if jn == 'yaw':
        # the neighbour this finger turns toward, swung away by 30 deg (sim)
        d = np.sign(tipy(f, steps[-1]) - tipy(f, 0.0))
        for g in F:
            if g == f:
                continue
            gy = HS.finger_pts(plan, g, Z[g])[0][0][1]
            fy = HS.finger_pts(plan, f, Z[f])[0][0][1]
            if np.sign(gy - fy) == d and g != 'thumb':
                away = 30.0 if np.sign(tipy(g, 30.0) - tipy(g, 0.0)) == d else -30.0
                base[g]['yaw'] = away
    seq = []
    for v in steps:
        J = {g: dict(x) for g, x in base.items()}
        J[f][jn] = float(v)
        seq.append((v, J, gap(J)))
    bad = [round(g, 1) for v, J, g in seq if g < 8]
    print(f"{f:6s} {jn} {steps}: base {[(g, base[g]['yaw']) for g in F if base[g]['yaw']]} model gaps {[round(g, 1) for v, J, g in seq]}"
          + ("  REFUSED" if bad else ""), flush=True)
    if not bad:
        plans.append((f, jn, base, seq))
if not GO:
    sys.exit(0)
if st.state().get("servo_torque") != 1:
    sys.exit("torque off")
out = []
for f, jn, base, seq in plans:
    st.call("POST", "/api/v1/manual/joints", {"joints": base, "servo_speed": 30}); time.sleep(1.5)
    for v, J, g in seq:
        st.call("POST", "/api/v1/manual/joints", {"joints": J, "servo_speed": 30}); time.sleep(2.0)
        m = HS.sim_joints(st.state())
        sgn = P.JOINT_SIGN[(f, jn)]
        out.append({"finger": f, "joint": jn, "cmd_servo": v * sgn, "meas_servo": m[f][jn] * sgn, "cmd_sim": v, "meas_sim": m[f][jn]})
        print(f"{f:6s} {jn}: commanded {v * sgn:+6.1f}, measured {m[f][jn] * sgn:+6.1f} (servo deg), short {abs(v - m[f][jn]):.1f}", flush=True)
    st.call("POST", "/api/v1/manual/joints", {"joints": Z, "servo_speed": 30}); time.sleep(2.0)
print("back to zero:", HS.show_pose("measured", HS.sim_joints(st.state())), flush=True)
Path('/home/will/workspace/hand/logs/20261116-hom_setup/20261116_joint_sag.json').write_text(json.dumps(out, indent=1))
