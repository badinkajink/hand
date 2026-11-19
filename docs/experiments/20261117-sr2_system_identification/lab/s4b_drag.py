"""S4(b) of the identification queue on the owner's S3 block (owner's notes of 2026-11-18 09:0x: "Drag 5 mm at 1 mm/s
each way at 0.5, 1 and 2 N, twice each"; go at 18:3x: "ok go ahead on these plans"). One finger's MCP steps its pad onto
the taped plate until the HEX-E's force (x_P, y_P) has changed --normal N, and holds that target. The arm then drags the
hand --dist mm along --axis (z: straight up and down, along the finger, which loads its joints least; w: level along the
plate) at --speed mm/s: +dist, back, -dist, back, --cycles times, the HEX-E guarding each leg at --stop-n. Then the MCP
steps back. The arm must already hold the pad a few mm off the plate (s3_approach.py plan --finger F, go).

Recorded: OUT/<stamp>_<finger>_drag<N>.jsonl (a header with F0 and R_PH, the hold's steps, one line per leg with the
force in P over the leg, an end line) and _raw.jsonl (the HEX-E at 100 Hz, the TCP at 50 Hz, the station's joints and
loads at 20 Hz). The friction coefficient is the tangential force (along the drag) over the normal (x_P) while
sliding; the tangential stiffness, the tangential force's slope against the TCP's travel before the slip.

    /usr/bin/python3 s4b_drag.py --finger index --normal 1.0 [--axis z] [--dist 5] [--speed 1] [--cycles 2] [--go]
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s3_press as S3  # noqa: E402

JOINTS = S3.JOINTS


class TcpSampler(threading.Thread):
    """The TCP pose (UR vector, m and rotation vector) at about hz from its own RTDE receive connection."""

    def __init__(self, robot, hz=50.0):
        super().__init__(daemon=True)
        import rtde_receive
        self.r = rtde_receive.RTDEReceiveInterface(robot, 125.0)
        self.dt, self.rows, self.lock, self.halt = 1.0 / hz, [], threading.Lock(), False

    def run(self):
        while not self.halt:
            try:
                p = self.r.getActualTCPPose()
                with self.lock:
                    self.rows.append((time.time(), [round(v, 6) for v in p]))
            except Exception:                             # noqa: BLE001
                pass
            time.sleep(self.dt)
        self.r.disconnect()

    def since(self, t0):
        with self.lock:
            return [r for r in self.rows if r[0] >= t0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--finger", required=True, choices=S3.FINGERS)
    ap.add_argument("--normal", type=float, required=True, help="N: the MCP's hold force on the plate")
    ap.add_argument("--axis", choices=("z", "w"), default="z")
    ap.add_argument("--dist", type=float, default=5.0)
    ap.add_argument("--speed", type=float, default=1.0, help="mm/s")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--step", type=float, default=0.5, help="deg: the MCP's step in the hold and the step back")
    ap.add_argument("--dwell", type=float, default=1.0)
    ap.add_argument("--max-offset", type=float, default=18.0)
    ap.add_argument("--stop-n", type=float, default=4.0, help="N of force change that stops a drag leg")
    ap.add_argument("--station", default="http://10.99.99.2:8765")
    ap.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
    ap.add_argument("--out", default=str(S3.ROOT / "logs" / time.strftime("%Y%m%d-sr2_s3_press")))
    ap.add_argument("--note", default="")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    FT, HS, SL = S3.imports()
    P, SV = SL._driver()
    f = a.finger
    sids = {k: str(SV.FINGER_JOINTS[P.FINGER_ID[f]][P.SIM_JOINT_TO_SERVO[k]][0]) for k in JOINTS}
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    if s.get("servo_torque") != 1:
        raise SystemExit("servo torque is not on")
    start = dict(HS.sim_joints(s)[f])
    q0 = float(start["mcp"])
    R_PH = S3.hexe_in_P()
    plan = sorted(Path(a.out).glob("*_approach_*.json"), key=lambda p: p.stat().st_mtime)[-1]
    w_B = np.array(json.loads(plan.read_text())["w_B"])
    axis_B = np.array([0.0, 0.0, 1.0]) if a.axis == "z" else w_B
    print(f"{f}: MCP from {q0:+.2f} deg steps {a.step:g} deg every {a.dwell:g} s until |dF| {a.normal:g} N (x_P, y_P), "
          f"holds; the arm drags {a.dist:g} mm along {a.axis} ({np.round(axis_B, 3).tolist()} in B) at {a.speed:g} mm/s, "
          f"+, back, -, back, x{a.cycles}; guard {a.stop_n:g} N; then the MCP steps back")
    if not a.go:
        return 0
    import tip_contacts as TC
    pids, how = TC.ensure_splitter(start=True)
    arm = HS.Arm(False, 0.4)                     # one HEX-E stream: the arm's guard's, also recorded here
    arm.connect(True)
    ft = arm.ft
    time.sleep(1.5)
    F0, sd0, n0 = S3.hexe_mean(ft, 1.0)
    if F0 is None:
        arm.close()
        ft.stop = True
        raise SystemExit("the HEX-E is not streaming")
    samp = S3.Sampler(HS, st)
    samp.start()
    tcp = TcpSampler(FT.HT.ROBOT)
    tcp.start()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{time.strftime('%Y%m%d-%H%M%S')}_{f}_drag{a.normal:g}N_{a.axis}"
    fh = open(f"{stem}.jsonl", "w")
    t_start = time.time() - 1.0
    hdr = {"what": "S4(b) drag", "finger": f, "normal_N": a.normal, "axis": a.axis, "axis_B": axis_B.tolist(),
           "dist_mm": a.dist, "speed_mm_s": a.speed, "cycles": a.cycles, "F0_N": F0.tolist(), "F0_sd_N": sd0.tolist(),
           "R_PH": R_PH.tolist(), "start_joints": start, "splitter": how, "plan": plan.name, "note": a.note,
           "t": time.time()}
    fh.write(json.dumps(hdr) + "\n")

    def dFP(seconds=0.3):
        Fm, _, _ = S3.hexe_mean(ft, seconds)
        return None if Fm is None else R_PH @ (Fm - F0)

    def send(target):
        cmd = dict(start)
        cmd["mcp"] = float(target)
        st.call("POST", "/api/v1/manual/joints", {"joints": {f: cmd}, "servo_speed": 20})

    why, k = None, 0
    try:
        for k in range(1, int(25 / a.step) + 1):           # the hold
            tgt = q0 + a.step * k
            send(tgt)
            t0 = time.time()
            while time.time() - t0 < a.dwell:
                time.sleep(0.1)
                d = dFP(0.2)
                if d is not None and np.linalg.norm(d[:2]) >= a.normal:
                    why = "held"
                    break
            rs = samp.since(time.time() - 0.5)
            rd = float(np.median([r[1][f]["mcp"] for r in rs if r[1]])) if rs else None
            d = dFP(0.3)
            fh.write(json.dumps({"phase": "hold", "k": k, "target": round(tgt, 3), "reading": rd,
                                 "dF_P": None if d is None else np.round(d, 4).tolist(), "t": time.time()}) + "\n")
            print(f"{time.strftime('%H:%M:%S')} hold {k:2d}: mcp {tgt:+.2f} reading {S3.num(rd, '+.2f')} dF_P "
                  f"{np.round(d, 2).tolist() if d is not None else '-'}", flush=True)
            if why == "held":
                break
            if rd is not None and tgt - rd >= a.max_offset:
                why = "the hold's offset reached --max-offset"
                break
        if why != "held":
            raise SystemExit(f"the hold stopped short: {why}")
        time.sleep(1.0)
        T0 = arm.T_BE()
        legs = []
        for c in range(a.cycles):
            for name, d_mm in (("plus", a.dist), ("back", 0.0), ("minus", -a.dist), ("back", 0.0)):
                T = T0.copy()
                T[:3, 3] = T0[:3, 3] + d_mm * axis_B
                tl0 = time.time()
                lw, moved = arm.move(T, a.speed / 1e3, 0.02, a.stop_n, a.dist / a.speed + 6.0,
                                     f"drag {c + 1} {name} to {d_mm:+g} mm")
                time.sleep(0.5)
                tl1 = time.time()
                tt, vv = ft.ring.since(tl0, tl1)
                FP = (np.asarray(vv)[:, :3] - F0) @ R_PH.T if len(tt) else np.zeros((0, 3))
                ax_P = np.linalg.inv(arm.T_BP())[:3, :3] @ axis_B
                ft_ = FP @ ax_P if len(FP) else np.array([])
                rec = {"phase": "drag", "cycle": c + 1, "leg": name, "to_mm": d_mm, "t0": tl0, "t1": tl1,
                       "moved_mm": moved, "stopped": lw, "axis_P": np.round(ax_P, 4).tolist(),
                       "F_normal_x_P": None if not len(FP) else [round(float(FP[:, 0].min()), 3),
                                                                  round(float(np.median(FP[:, 0])), 3),
                                                                  round(float(FP[:, 0].max()), 3)],
                       "F_along_drag": None if not len(ft_) else [round(float(ft_.min()), 3),
                                                                  round(float(np.median(ft_)), 3),
                                                                  round(float(ft_.max()), 3)]}
                legs.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"{time.strftime('%H:%M:%S')} drag {c + 1} {name:5s} -> {d_mm:+g} mm: moved {moved:.2f}, normal "
                      f"{rec['F_normal_x_P']}, along {rec['F_along_drag']}" + (f" -- STOPPED: {lw}" if lw else ""),
                      flush=True)
                if lw:
                    why = f"drag leg stopped: {lw}"
                    break
            if why != "held":
                break
        if why == "held":
            why = "done"
        if why != "done":                                    # back to the leg's start before the MCP lets go
            T = arm.T_BE()
            if np.linalg.norm(T[:3, 3] - T0[:3, 3]) > 0.2:
                arm.move(T0, a.speed / 1e3, 0.02, a.stop_n + 2.0, 12.0, "drag: back to the start")
    finally:
        for kb in range(k - 1, -1, -1):                    # the MCP steps back
            send(q0 + a.step * kb)
            time.sleep(a.dwell)
        d = dFP(0.3)
        print(f"{time.strftime('%H:%M:%S')} MCP back at {q0:+.2f}; dF_P {np.round(d, 2).tolist() if d is not None else '-'}",
              flush=True)
        t1 = time.time()
        tt, vv = ft.ring.since(t_start, t1)
        with open(f"{stem}_raw.jsonl", "w") as raw:
            raw.write(json.dumps({"hexe_t": np.asarray(tt).tolist(), "hexe": np.round(np.asarray(vv), 4).tolist(),
                                  "tcp": tcp.since(t_start),
                                  "station": [(r[0], r[1][f], {kk: r[2].get(sids[kk]) for kk in JOINTS})
                                              for r in samp.since(t_start) if r[1]]}) + "\n")
        fh.write(json.dumps({"end": time.time(), "reason": why}) + "\n")
        fh.close()
        samp.halt = True
        tcp.halt = True
        arm.close()
        ft.stop = True
    print(f"S4(b) {f} {a.normal:g} N {a.axis}: {why}; -> {stem}.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
