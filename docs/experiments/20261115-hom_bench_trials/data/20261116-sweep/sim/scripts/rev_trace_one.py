"""One closed-loop run's joint trace at fixed times (hand-4a, 2026-11-16): gr g13 at (105, 70), the stem scene with the
soft contact (impratio 1, friction 1.0), the server's plant, seed 0: the yaw joints, the thumb's MCP and the object's
twist about z_P, beside the hand's trial of 13:03.
    OMP_NUM_THREADS=1 .venv/bin/python rev_trace_one.py
"""
import sys
from pathlib import Path
import numpy as np
D = Path.home() / "Programs/hand/logs/20261116-sim_real_reversal"
sys.path.insert(0, str(D))
import rev_inputs as RI  # noqa: E402
import rev_rescore as RR  # noqa: E402
RI.init(str(RI.SCENE))
RP = RI._G["RP"]
RP.TOY_ROWS = RI._G["toy_rows"]
from sr2hand.control import hom_bench as HB  # noqa: E402
import mujoco  # noqa: E402
import hand_object_scale_sim as S  # noqa: E402
import hom_tasks as T  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402
setup, mxy, _ = RP.toy_setup("twist", 40.0, 329, 13, "grid2dgr")
p_pads, p_pt = RP.ref_scene("sphere", 40.0)
m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
RP.move_mounts(m_pt, mxy, RP.mount_base(m_pt))
q_open, _ = T.open_pose(setup, m_pt)
S.set_palm(m_pt, setup["palm"])
bench = HB.HomBench("twist", 40.0, setup, m_pt, q_open, dt=HB.DT_BENCH, p_pads=p_pads)
params = {**RR.SERVER, "mu": 1.0}
bench.T = RP._Tasks(lambda path, st: RP.SR2Plant(path, st, mxy, params, 0.0))
stream, res = RI.simulate(HB, bench, 0, contacts=False, late=True)
Q = np.array([s["obj_quat"] for s in stream], float)
tw = np.degrees((Rot.from_quat(Q[:, [1, 2, 3, 0]]) * Rot.from_quat(Q[0, [1, 2, 3, 0]]).inv()).as_rotvec()[:, 2])
print("sim gr g13, bench inputs (servo deg): t | middle aa meas/tgt | index aa meas/tgt | thumb aa meas/tgt | thumb fe1 meas/tgt | twist | phase")
for x in (1.0, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0):
    k = int(np.argmin([abs(float(s["t"]) - x) for s in stream]))
    s = stream[k]
    j, t = s["joints"], s["targets"]
    print(f"  {float(s['t']):4.2f} | {j['middle']['aa']:6.1f}/{t['middle']['aa']:6.1f} | {j['index']['aa']:6.1f}/{t['index']['aa']:6.1f} | "
          f"{j['thumb']['aa']:6.1f}/{t['thumb']['aa']:6.1f} | {j['thumb']['fe1']:6.1f}/{t['thumb']['fe1']:6.1f} | {tw[k]:6.1f} | {s['phase']}")
