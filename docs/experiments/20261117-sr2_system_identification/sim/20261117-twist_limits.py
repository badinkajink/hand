"""Which joints of g0 at (70, 110) reach their limits as the simulated twist stalls (point contact soft, 7.5 V plant, no
budget, the safe ranges lifted): per joint the angle at 2, 4, 6, 8 s against the model's range, and the object's twist."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import replan as R
R.RG.gear_class = R.RC2.clutch_class; R.RC2.STIFF, R.RC2.BREAKAWAY = True, 0.0; R.RG.RF.RL.rows_file(); R.RG.init(str(R.RG.RF.SCENE))
R.SRC, R.CALL, R.K = "grid2dgr", 162, 0
pf = json.loads(Path(sys.argv[1]).read_text()); R.PLAN_D6 = pf["meta"]["hom"]["d6_P_mm"]
from sr2hand.control import servo_limits as SLm
SLm.safe_range = lambda f, sv: (-1e9, 1e9)
import mujoco
from sr2hand.control import hom_bench as HB
import hand_object_scale_sim as S
import hom_tasks as T
import hand_object_scale_kin as KIN
RP = R.RG.RF._G["RP"]; RP.TOY_ROWS = R.RG.RF._G["toy_rows"]
R.RC2.KP_YAW_IM = 0.35 * 1.5
setup, mxy, _ = RP.toy_setup("twist", 40.0, 162, 0, "grid2dgr")
p_pads, p_pt = RP.ref_scene("sphere", 40.0)
m_pt = mujoco.MjModel.from_xml_path(str(p_pt)); RP.move_mounts(m_pt, mxy, RP.mount_base(m_pt))
q_open = {f: np.radians([pf["poses"][0]["joints"][f][j] for j in ("yaw", "mcp", "pip")]) for f in T.FINGERS}
S.set_palm(m_pt, setup["palm"])
bench = HB.HomBench("twist", 40.0, setup, m_pt, q_open, dt=HB.DT_BENCH, p_pads=p_pads, torque_budget=None)
GP = R.RG._G["GP"]
bench.T = RP._Tasks(lambda path, st: GP(str(R.D / "scenes" / R.SCENES["point"]), st, mxy, {**R.PLANT_75, "mu": 1.0}, 0.0, c=0.6))
stream, res = HB.simulate(bench, 0, contacts=False)
lim = KIN.limits(False)
from scipy.spatial.transform import Rotation as Rot
Q = np.array([s["obj_quat"] for s in stream]); R0 = Rot.from_quat(Q[0][[1,2,3,0]])
for tt in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
    i = min(range(len(stream)), key=lambda k: abs(stream[k]["t"] - tt)); s = stream[i]
    q = HB.sim_from_servo(s["joints"]); qt = HB.sim_from_servo(s["targets"]) if s.get("targets") else q
    tw = math.degrees((R0.inv() * Rot.from_quat(Q[i][[1,2,3,0]])).as_rotvec()[2])
    cells = []
    for f in T.FINGERS:
        for k, j in enumerate(("yaw", "mcp", "pip")):
            lo, hi = np.degrees(lim[f][k])
            v, vt = math.degrees(q[f][k]), math.degrees(qt[f][k])
            mark = "*" if min(v - lo, hi - v) < 3 else " "
            cells.append(f"{f[0]}{j[0]} {v:6.1f}{mark}(tgt {vt:6.1f})")
    print(f"t {tt:.0f} twist {tw:6.1f} | " + " ".join(cells))
print("limits (deg):", {f: np.degrees(lim[f]).round(1).tolist() for f in T.FINGERS})
print("track", res.get("track_err_rms_deg"), res.get("track_ref_rms_deg"))
