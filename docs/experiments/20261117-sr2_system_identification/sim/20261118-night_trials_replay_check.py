"""A check of night_replay.py (hand-4a, 2026-11-18): HOM's closed-loop twist in simulation, on the replay's own plant
and from the start of the hand's trial 20261117-194243, written as a server record; then night_replay.run1 replays that
record's targets with no delay on the same plant. The replay should give the record's own twist, rise and joints.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 SCALE_HAND=cad_v2 OMP_NUM_THREADS=1 \\
        .venv/bin/python logs/20261117-g13_open_pose/night_replay_check.py [CONTACT VOLTS]
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import night_replay as NR  # noqa: E402

HAND = NR.ROOT / "docs/experiments/20261115-hom_bench_trials/runs/20261117-194243_twist_d40_trial001.jsonl"


def main():
    import mujoco
    import hom_trial_tracking as TT
    from sr2hand.control import hom_bench as HB
    import hand_object_scale_sim as S
    import hom_tasks as T
    contact, volts = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("pads", "7v5")
    NR.RPL.RG.gear_class = NR.RPL.RC2.clutch_class
    NR.RPL.RC2.STIFF, NR.RPL.RC2.BREAKAWAY = True, 0.0
    NR.RPL.RG.RF.RL.rows_file()
    NR.RPL.RG.init(str(NR.RPL.RG.RF.SCENE))
    head, rows, _ = TT.load_server(HAND)
    RP = NR.RPL.RG.RF._G["RP"]
    RP.TOY_ROWS = NR.RPL.RG.RF._G["toy_rows"]
    setup, mxy, _ = RP.toy_setup("twist", 40.0, 162, 0, "grid2dgr")
    p_pads, p_pt = RP.ref_scene("sphere", 40.0)
    m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
    RP.move_mounts(m_pt, mxy, RP.mount_base(m_pt))
    q_open = HB.sim_from_servo(rows[0]["joints"])
    S.set_palm(m_pt, setup["palm"])
    bench = HB.HomBench("twist", 40.0, setup, m_pt, q_open, dt=HB.DT_BENCH, p_pads=p_pads, torque_budget=0.56)
    bench.T = RP._Tasks(lambda path, st: NR.make_plant(contact, volts)[0])
    stream, res = HB.simulate(bench, 0, contacts=False)
    h = {"kind": "header", "task": "twist", "t0_ms": 0.0, "t_end": 7.8, "dt": 0.02, "palm_m": list(map(float, setup["palm"])),
         "prm": head.get("prm"), "started": "2026-11-18T00:00", "tips": "off"}
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, dir=NR.ROOT / "logs/20261118-night_replay") as fh:
        fh.write(json.dumps(h) + "\n")
        for s in stream:
            fh.write(json.dumps({"t": s["t"], "joints": s["joints"], "targets": s["targets"], "phase": s["phase"],
                                 "obj_pos_mm": s["obj_pos_mm"], "obj_quat": s["obj_quat"],
                                 "t_joints_ms": 1e3 * float(s["t"])}) + "\n")
        path = Path(fh.name)
    NR.DELAY_S = 0.0
    r = NR.run1({"stamp": "check", "file": str(path.relative_to(NR.ROOT)), "lift_s": T.T_LIFT, "contact": contact,
                 "volts": volts})
    print(f"{contact} {volts}: closed loop track {res.get('track_err_rms_deg')}/{res.get('track_ref_rms_deg')}")
    print("record ", r["hand"]["track"], r["hand"]["motion"])
    print("replay ", r["replay"]["track"], r["replay"]["motion"])
    print("twist trace RMS", r["twist_trace_rms_deg"], "| joints RMS", {w: v["all"] for w, v in r["joint_rms_deg"].items()},
          "| object", r["object_task"])
    path.unlink()


if __name__ == "__main__":
    main()
