"""Leave the tall indenter's homing key after a seating by hand, without crossing its sphere (2026-10-27). move_guarded.py's
home goes in a straight line and does not know the tall indenter: from the key that line swept the mount's fork onto the
sphere's +y_I side (13:52, stopped at 3.2 N by the HEX-E guard). This route goes up off the key, straight back toward
the robot along -x_I, up again, then home.

    UR5_MOUNT=v7 /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/leave_tall_key.py plan|go [--seat LABEL]

Waypoints (flange poses): up 10 mm at 3 mm/s (the key grips the socket: the lift may see up to 25 N, as move_guarded's
lift off a key), up 15 mm more at 10 mm/s, 80 mm along -x_I at 10 mm/s, up 40 mm at 10 mm/s, home at 25 mm/s. The tall
indenter is placed by the seating LABEL (default tall_indenter_seated) with the plan's homing geometry, the sphere
fixture and the rod as the other indenters; every segment is sampled against ur5_cb3_cell with the tall indenter, and
the run is refused when the tool or a link comes within 5 mm of anything after the lift. Each move runs as move_guarded
runs its moves (the HEX-E's change from the move's start over its limit, a stale HEX-E or a protective stop: stopL) and
goes to moves.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "scripts"), str(ROOT / "src/morphohand/driver/reskin5x/host")]
import move_guarded as MG  # noqa: E402
import ur5_cb3_cell as CELL  # noqa: E402

C = CELL.C
TALL_PLAN = ROOT / "docs/experiments/20261026-tall_indenter/plan/20261026-plan_tall_H160_check.json"
BASE_PLAN = HERE / "20261026-plan_sphere_lab_r25_pass1_cb3t.json"      # the cell: board, rod; the sphere fixture below


def slerp(Ta, Tb, s):
    from scipy.spatial.transform import Rotation as Rot, Slerp
    T = np.eye(4)
    T[:3, :3] = Slerp([0, 1], Rot.from_matrix([Ta[:3, :3], Tb[:3, :3]]))([s]).as_matrix()[0]
    T[:3, 3] = Ta[:3, 3] + s * (Tb[:3, 3] - Ta[:3, 3])
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("plan", "go"))
    ap.add_argument("--seat", default="tall_indenter_seated")
    a = ap.parse_args()
    g = json.loads(TALL_PLAN.read_text())["geometry"]
    T_EM, T_IMh = np.array(g["T_EM"]), np.array(g["T_IM_home"])
    ck = json.loads(BASE_PLAN.read_text())["checks"]
    CELL.BOARD.update(ck["board"])
    CELL.OTHER[:] = [(np.array(o["T_BI"], float), o["kind"]) for o in ck.get("other", [])]
    sph = ck["placement_mm_deg"]                           # the sphere fixture's base, from cb3t's pass 1 build
    T_sph = np.eye(4)
    T_sph[:3, :3] = C.Rz(np.radians(sph[3])) if hasattr(C, "Rz") else np.array(
        [[np.cos(np.radians(sph[3])), -np.sin(np.radians(sph[3])), 0], [np.sin(np.radians(sph[3])), np.cos(np.radians(sph[3])), 0], [0, 0, 1]])
    T_sph[:3, 3] = sph[:3]
    CELL.OTHER.append((T_sph, "sphere"))
    seat = [r for r in map(json.loads, open(HERE / "poses.jsonl")) if r["label"] == a.seat][-1]
    T_BI = np.array(seat["T_flange_fk_mm"]) @ T_EM @ np.linalg.inv(T_IMh)
    xI = T_BI[:3, 0] / np.linalg.norm(T_BI[:3, 0])
    import rtde_receive
    r = rtde_receive.RTDEReceiveInterface(MG.ROBOT, 125.0)
    cur_T, cur_q = C.ur_to_T(r.getActualTCPPose()), np.array(r.getActualQ())
    seat_T = np.array(seat["T_flange_fk_mm"])
    off = C.pose_error(cur_T, seat_T)
    print(f"from flange {np.round(cur_T[:3, 3], 1)} mm: {off[0]:.1f} mm, {off[1]:.1f} deg from the seating {a.seat}")
    if off[0] > 5.0:
        sys.exit("REFUSED: the arm is not at the seating; this route starts on the tall key")
    T1 = cur_T.copy(); T1[2, 3] += 10.0
    T2 = T1.copy(); T2[2, 3] += 15.0
    T3 = T2.copy(); T3[:3, 3] -= 80.0 * xI
    T4 = T3.copy(); T4[2, 3] += 40.0
    plan_mg, recs, T_EM_mg = MG.load()
    T5 = MG.waypoints("home", 2.0, T4, cur_q, plan_mg, recs, T_EM_mg, "plan")[-1]["T"]
    segs = [("lift off the tall key 10 mm", T1, 0.003, 25.0), ("up 15 mm more", T2, 0.010, 3.0),
            ("80 mm toward the robot", T3, 0.010, 3.0), ("up 40 mm", T4, 0.010, 3.0), ("home", T5, 0.025, 3.0)]
    bad, q, Tp = [], cur_q, cur_T
    for i, (name, Tt, v, lim) in enumerate(segs):
        worst = {}
        for s in np.linspace(0, 1, 25):
            T = slerp(Tp, Tt, s)
            q = CELL.SIM.ik(T, q)
            if q is None:
                bad.append(f"{name}: no IK"); break
            for k, val in CELL.clearance(q, T, T_BI, T_EM, "tall").items():
                worst[k] = min(worst.get(k, 1e9), val)
        least = sorted(worst.items(), key=lambda kv: kv[1])[:4]
        print(f"  {name}: to {np.round(Tt[:3, 3], 1).tolist()} at {v * 1e3:.0f} mm/s; least clearance "
              + ", ".join(f"{k} {val:.1f}" for k, val in least) + " mm")
        if i > 0 and least and least[0][1] < 5.0:
            bad.append(f"{name}: {least[0][0]} {least[0][1]:.1f} mm")
        Tp = Tt
    r.disconnect()
    if bad:
        sys.exit("REFUSED: " + "; ".join(bad))
    if a.mode == "plan":
        return
    import rtde_control
    r = rtde_receive.RTDEReceiveInterface(MG.ROBOT, 125.0)
    ft = C.OptoForceStream(MG.FT, rate=100, bias=False, filt=3)
    ft.configure()
    ft.start()
    c = rtde_control.RTDEControlInterface(MG.ROBOT)
    try:
        c.setTcp([0.0] * 6)
        for name, Tt, v, lim in segs:
            time.sleep(0.6)
            _, v0 = ft.ring.since(time.time() - 0.4)
            F0 = v0[:, :3].mean(axis=0)
            t0, fmax, why = time.time(), 0.0, None
            c.moveL(C.T_to_ur(Tt), v, 0.1, True)
            time.sleep(0.05)
            while True:
                t, val = ft.ring.latest()
                if t is None or time.time() - t > 0.1:
                    why = "no HEX-E sample for 100 ms"
                else:
                    fmax = max(fmax, float(np.linalg.norm(val[:3] - F0)))
                    if fmax > lim:
                        why = f"HEX-E force changed {fmax:.2f} N (limit {lim:g})"
                if why is None and (r.isProtectiveStopped() or r.isEmergencyStopped()):
                    why = "protective or emergency stop"
                if why:
                    c.stopL(1.0)
                    break
                if c.getAsyncOperationProgress() < 0 and time.time() - t0 > 0.3:
                    break
                if time.time() - t0 > 90.0:
                    why = "move over 90 s"
                    c.stopL(1.0)
                    break
                time.sleep(0.01)
            got = C.ur_to_T(r.getActualTCPPose())
            err = C.pose_error(got, Tt)
            _, v1 = ft.ring.since(time.time() - 0.2)
            rec = dict(time=time.strftime("%H:%M:%S"), what="leave_tall_key", name=name, target_flange=Tt.round(3).tolist(),
                       reached_flange=got.round(3).tolist(), error_mm_deg=[round(err[0], 3), round(err[1], 3)],
                       max_dF_N=round(fmax, 3), stop_N=lim, F0_N=F0.round(3).tolist(),
                       F_end_N=v1[:, :3].mean(axis=0).round(3).tolist() if len(v1) else None, stopped=why)
            with open(HERE / "moves.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  {name}: {'STOPPED, ' + why if why else 'reached'}; {err[0]:.2f} mm, {err[1]:.2f} deg from the target; "
                  f"HEX-E force changed at most {fmax:.2f} N", flush=True)
            if why:
                break
    finally:
        try:
            c.stopScript()
            c.disconnect()
        except Exception:                                   # noqa: BLE001
            pass
        ft.stop = True


if __name__ == "__main__":
    main()
