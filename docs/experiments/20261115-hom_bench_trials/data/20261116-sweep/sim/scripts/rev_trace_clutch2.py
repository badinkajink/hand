"""rev_trace_one.py's joint trace on the load-holding plant of rev_clutch2.py (c_f 0.6): gr g13 at (105, 70), soft
contact at friction 1.0, server plant, seed 0, the plant's contacts and current pose given to HOM as in rev_clutch2's runs
(hand-4a, 2026-11-16): do the index and middle yaws hold still during the twist, as on the hand?
    OMP_NUM_THREADS=1 .venv/bin/python rev_trace_clutch2.py
"""
import sys
from pathlib import Path
import numpy as np
D = Path.home() / "Programs/hand/logs/20261116-sim_real_reversal"
sys.path.insert(0, str(D))
import rev_clutch2 as RC2  # noqa: E402
import rev_fric2 as RF  # noqa: E402
import rev_rescore as RR  # noqa: E402
RF.init(str(RF.SCENE))
RP = RF._G["RP"]
RP.TOY_ROWS = RF._G["toy_rows"]
from sr2hand.control import hom_bench as HB  # noqa: E402
import mujoco  # noqa: E402
import hand_object_scale_sim as S  # noqa: E402
import hom_tasks as T  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402
RC2.STIFF = "--stiff" in sys.argv
CP = RC2.clutch_class()
setup, mxy, _ = RP.toy_setup("twist", 40.0, 329, 13, "grid2dgr")
p_pads, p_pt = RP.ref_scene("sphere", 40.0)
m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
RP.move_mounts(m_pt, mxy, RP.mount_base(m_pt))
q_open, _ = T.open_pose(setup, m_pt)
S.set_palm(m_pt, setup["palm"])
bench = HB.HomBench("twist", 40.0, setup, m_pt, q_open, dt=HB.DT_BENCH, p_pads=p_pads)
params = {**RR.SERVER, "mu": 1.0}
bench.T = RP._Tasks(lambda path, st: CP(path, st, mxy, params, 0.0, c=0.6))
stream, res = HB.simulate(bench, 0)
Q = np.array([s["obj_quat"] for s in stream], float)
tw = np.degrees((Rot.from_quat(Q[:, [1, 2, 3, 0]]) * Rot.from_quat(Q[0, [1, 2, 3, 0]]).inv()).as_rotvec()[:, 2])
print("sim gr g13, load-holding servos (servo deg): t | middle aa meas/tgt | index aa meas/tgt | thumb aa meas/tgt | thumb fe1 meas/tgt | twist | phase")
for x in (1.0, 1.6, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0):
    k = int(np.argmin([abs(float(s["t"]) - x) for s in stream]))
    s = stream[k]
    j, t = s["joints"], s["targets"]
    print(f"  {float(s['t']):4.2f} | {j['middle']['aa']:6.1f}/{t['middle']['aa']:6.1f} | {j['index']['aa']:6.1f}/{t['index']['aa']:6.1f} | "
          f"{j['thumb']['aa']:6.1f}/{t['thumb']['aa']:6.1f} | {j['thumb']['fe1']:6.1f}/{t['thumb']['fe1']:6.1f} | {tw[k]:6.1f} | {s['phase']}")
