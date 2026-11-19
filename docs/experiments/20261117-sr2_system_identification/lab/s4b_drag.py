"""S4(b) of the identification queue on the owner's S3 block: the pad's friction on a plate. One finger's MCP steps its
pad onto the plate until the HEX-E's force (x_P, y_P) has changed by the normal force N, and holds that target. The arm
then drags the hand along --axis while the finger holds: z, straight up and down along the finger, which loads its
joints least (a level drag loads the yaw, whose chain gives about 2 deg per N). The MCP steps back after each N.

Two modes:
- cycles (2026-11-18 18:4x; owner's notes 09:0x: "Drag 5 mm at 1 mm/s each way at 0.5, 1 and 2 N, twice each"):
  +dist, back, -dist, back, --cycles times at --speed.
- protocol (owner's go for the night of 2026-11-18, relayed 19:1x: "yea you have the go ahead to run tonight's
  session"; hand-4a's plan: normal 0.5-3 N, drag speeds 0.2 and 2 mm/s, rests of 1, 10 and 60 s with the pad loaded
  before each drag). For each N in --normals: hold; then for each speed in --speeds and each rest in --rests: rest,
  up --dist, rest, down --dist. The first leg after each rest shows the breakaway against the sliding force.

The queue's 3 N rule holds as an absolute guard: a leg stops when the force's x_P, y_P part passes --max-n (3 N) from
the force before the hold. The drag runs along the HEX-E's z, which drifts about 0.4 N a minute; each rest records the
drift for the leg after it. Any servo flag or torque loss on the finger ends the run. A stopped leg returns the hand to
the hold's position and ends that N; the end line and NOTES list every early stop. Rows are fsynced. The HEX-E (100 Hz),
the TCP (50 Hz) and the station (20 Hz) go to _raw.jsonl after every rest and leg (the HEX-E ring keeps 180 s).

    /usr/bin/python3 s4b_drag.py --finger index --normal 1.0 [--axis z] [--dist 5] [--speed 1] [--cycles 2] [--go]
    /usr/bin/python3 s4b_drag.py --finger index --protocol --normals 0.5,1,2 --speeds 0.2,2 --rests 1,10,60 [--go]
"""
import argparse
import json
import os
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

    def since(self, t0, t1=None):
        with self.lock:
            return [r for r in self.rows if r[0] >= t0 and (t1 is None or r[0] < t1)]


def floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--finger", required=True, choices=S3.FINGERS)
    ap.add_argument("--normal", type=float, help="N: the MCP's hold force (cycles mode)")
    ap.add_argument("--protocol", action="store_true")
    ap.add_argument("--normals", type=floats, default=[0.5, 1.0, 2.0])
    ap.add_argument("--speeds", type=floats, default=[0.2, 2.0], help="mm/s (protocol)")
    ap.add_argument("--rests", type=floats, default=[1.0, 10.0, 60.0], help="s (protocol)")
    ap.add_argument("--axis", choices=("z", "w"), default="z")
    ap.add_argument("--dist", type=float, default=5.0)
    ap.add_argument("--speed", type=float, default=1.0, help="mm/s (cycles)")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--step", type=float, default=0.5, help="deg: the MCP's step in the hold and the step back")
    ap.add_argument("--dwell", type=float, default=1.0)
    ap.add_argument("--max-offset", type=float, default=18.0)
    ap.add_argument("--max-n", type=float, default=3.0, help="N: the absolute guard on the force's x_P, y_P part")
    ap.add_argument("--stop-n", type=float, default=4.0, help="N of force change from a leg's start that stops it")
    ap.add_argument("--max-temp", type=float, default=50.0, help="C: wait before an N until the finger is under 40")
    ap.add_argument("--station", default="http://10.99.99.2:8765")
    ap.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
    ap.add_argument("--out", default=str(S3.ROOT / "logs" / time.strftime("%Y%m%d-sr2_s3_press")))
    ap.add_argument("--note", default="")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    if not a.protocol and a.normal is None:
        raise SystemExit("--normal N, or --protocol")
    normals = a.normals if a.protocol else [a.normal]
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
    if a.protocol:
        legs_per_n = [(v, r, d) for v in a.speeds for r in a.rests for d in (a.dist, 0.0)]
        est = sum(r + a.dist / v + 2.0 for v, r, d in legs_per_n) + 40.0
        print(f"{f} protocol: N {normals}; per N: hold, then for speed {a.speeds} mm/s and rest {a.rests} s: rest, up "
              f"{a.dist:g} mm, rest, down; about {est / 60:.1f} min per N, {len(normals) * est / 60:.0f} min in all")
    else:
        print(f"{f}: hold {a.normal:g} N; drag {a.dist:g} mm along {a.axis} at {a.speed:g} mm/s, +, back, -, back, "
              f"x{a.cycles}")
    print(f"guards: |F| x_P, y_P {a.max_n:g} N absolute, {a.stop_n:g} N change per leg, any flag, {a.max_temp:g} C; "
          f"axis {np.round(axis_B, 3).tolist()} in B (plan {plan.name})")
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
    tag = "protocol" if a.protocol else f"drag{a.normal:g}N_{a.axis}"
    stem = out / f"{time.strftime('%Y%m%d-%H%M%S')}_{f}_{tag}"
    fh = open(f"{stem}.jsonl", "w")
    raw = open(f"{stem}_raw.jsonl", "w")
    t_raw = [time.time() - 1.0]

    def write(rec):
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    def dump(label):
        t1 = time.time()
        tt, vv = ft.ring.since(t_raw[0], t1)
        raw.write(json.dumps({"label": label, "t0": t_raw[0], "t1": t1, "hexe_t": np.asarray(tt).tolist(),
                              "hexe": np.round(np.asarray(vv), 4).tolist(), "tcp": tcp.since(t_raw[0], t1),
                              "station": [(r[0], r[1][f], {kk: r[2].get(sids[kk]) for kk in JOINTS})
                                          for r in samp.since(t_raw[0]) if r[1] and r[0] < t1]}) + "\n")
        raw.flush()
        os.fsync(raw.fileno())
        t_raw[0] = t1

    write({"what": "S4(b) drag", "mode": "protocol" if a.protocol else "cycles", "finger": f, "normals": normals,
           "speeds": a.speeds if a.protocol else [a.speed], "rests": a.rests if a.protocol else None, "axis": a.axis,
           "axis_B": axis_B.tolist(), "dist_mm": a.dist, "cycles": a.cycles, "F0_N": F0.tolist(), "F0_sd_N": sd0.tolist(),
           "R_PH": R_PH.tolist(), "start_joints": start, "guards": {"max_n": a.max_n, "stop_n": a.stop_n,
                                                                    "max_temp": a.max_temp},
           "splitter": how, "plan": plan.name, "note": a.note, "t": time.time()})

    def dFP(seconds=0.3):
        Fm, _, _ = S3.hexe_mean(ft, seconds)
        return None if Fm is None else R_PH @ (Fm - F0)

    def send(target):
        cmd = dict(start)
        cmd["mcp"] = float(target)
        st.call("POST", "/api/v1/manual/joints", {"joints": {f: cmd}, "servo_speed": 20})

    def flags():
        d = S3.diag(st)
        bad = {k: (d.get("status") or {}).get(sids[k]) for k in JOINTS if (d.get("status") or {}).get(sids[k]) not in
               (None, 0, 0.0)}
        off = {k: (d.get("torque_enable") or {}).get(sids[k]) for k in JOINTS
               if (d.get("torque_enable") or {}).get(sids[k]) not in (None, 1, 1.0)}
        temps = {k: (d.get("temperature") or {}).get(sids[k]) for k in JOINTS}
        return bad, off, temps

    def guard():
        d = dFP(0.1)
        if d is not None and np.linalg.norm(d[:2]) > a.max_n:
            return f"|F| x_P, y_P {np.linalg.norm(d[:2]):.2f} N over {a.max_n:g} N"
        return None

    def wait_cool():
        while True:
            bad, off, temps = flags()
            hot = max(float(v) for v in temps.values() if v is not None)
            if hot < a.max_temp:
                return temps
            print(f"{time.strftime('%H:%M:%S')} {f} at {hot:g} C: waiting under 40", flush=True)
            while hot >= 40.0:
                time.sleep(30.0)
                bad, off, temps = flags()
                hot = max(float(v) for v in temps.values() if v is not None)

    early, k_hold, ended = [], 0, None
    try:
        for N in normals:
            temps = wait_cool()
            k_hold, why = 0, None
            for k_hold in range(1, int(25 / a.step) + 1):    # the hold
                tgt = q0 + a.step * k_hold
                send(tgt)
                t0 = time.time()
                while time.time() - t0 < a.dwell:
                    time.sleep(0.1)
                    d = dFP(0.2)
                    if d is not None and np.linalg.norm(d[:2]) >= N:
                        why = "held"
                        break
                rs = samp.since(time.time() - 0.5)
                rd = float(np.median([r[1][f]["mcp"] for r in rs if r[1]])) if rs else None
                d = dFP(0.3)
                write({"phase": "hold", "N": N, "k": k_hold, "target": round(tgt, 3), "reading": rd,
                       "dF_P": None if d is None else np.round(d, 4).tolist(), "t": time.time()})
                if why == "held":
                    break
                if rd is not None and tgt - rd >= a.max_offset:
                    why = "the hold's offset reached --max-offset"
                    break
            dump(f"hold {N:g}")
            print(f"{time.strftime('%H:%M:%S')} hold {N:g} N: {why} at mcp {tgt:+.2f} (reading {S3.num(rd, '+.2f')}), "
                  f"dF_P {np.round(dFP(0.3), 2).tolist()}; temps {temps}", flush=True)
            if why != "held":
                early.append({"N": N, "where": "hold", "why": why})
            else:
                time.sleep(1.0)
                T0 = arm.T_BE()
                if a.protocol:
                    seq = [(v, r, d) for v in a.speeds for r in a.rests for d in (a.dist, 0.0)]
                else:
                    seq = [(a.speed, 0.0, d) for _ in range(a.cycles) for d in (a.dist, 0.0, -a.dist, 0.0)]
                for i, (v, rest, d_mm) in enumerate(seq):
                    tr0, g = time.time(), None
                    while time.time() - tr0 < rest:                 # the rest, the pad loaded
                        time.sleep(0.2)
                        g = guard()
                        if g:
                            break
                    dr = dFP(0.5)
                    write({"phase": "rest", "N": N, "i": i, "rest_s": rest, "t0": tr0, "t1": time.time(),
                           "dF_P_end": None if dr is None else np.round(dr, 4).tolist(), "stopped": g})
                    dump(f"rest {N:g} {i}")
                    if g:
                        early.append({"N": N, "where": f"rest {i}", "why": g})
                        break
                    T = T0.copy()
                    T[:3, 3] = T0[:3, 3] + d_mm * axis_B
                    tl0 = time.time()
                    lw, moved = arm.move(T, v / 1e3, 0.02, a.stop_n, a.dist / v + 6.0,
                                         f"drag {N:g} N {v:g} mm/s rest {rest:g} s to {d_mm:+g} mm", check=guard)
                    time.sleep(0.3)
                    tl1 = time.time()
                    tt, vv = ft.ring.since(tl0, tl1)
                    FP = (np.asarray(vv)[:, :3] - F0) @ R_PH.T if len(tt) else np.zeros((0, 3))
                    ax_P = np.linalg.inv(arm.T_BP())[:3, :3] @ axis_B
                    fa = FP @ ax_P if len(FP) else np.array([])
                    bad, off, temps = flags()
                    rec = {"phase": "leg", "N": N, "i": i, "speed_mm_s": v, "rest_s": rest, "to_mm": d_mm,
                           "t0": tl0, "t1": tl1, "moved_mm": moved, "stopped": lw, "axis_P": np.round(ax_P, 4).tolist(),
                           "F_normal_x_P": None if not len(FP) else [round(float(FP[:, 0].min()), 3),
                                                                      round(float(np.median(FP[:, 0])), 3),
                                                                      round(float(FP[:, 0].max()), 3)],
                           "F_along": None if not len(fa) else [round(float(fa.min()), 3), round(float(np.median(fa)), 3),
                                                                round(float(fa.max()), 3)],
                           "flags": bad, "torque_off": off, "temps": temps}
                    write(rec)
                    dump(f"leg {N:g} {i}")
                    print(f"{time.strftime('%H:%M:%S')} {N:g} N {v:g} mm/s rest {rest:g} s -> {d_mm:+g} mm: moved "
                          f"{moved:.2f}, normal {rec['F_normal_x_P']}, along {rec['F_along']}"
                          + (f" -- STOPPED: {lw}" if lw else ""), flush=True)
                    if bad or off:
                        ended = f"servo flags {bad} or torque off {off}"
                        early.append({"N": N, "where": f"leg {i}", "why": ended})
                        break
                    if lw:
                        early.append({"N": N, "where": f"leg {i}", "why": lw})
                        break
                if np.linalg.norm(arm.T_BE()[:3, 3] - T0[:3, 3]) > 0.2:   # back to the hold's position
                    arm.move(T0, 0.002, 0.02, a.stop_n + 2.0, 12.0, "drag: back to the hold's position")
            if ended and "torque off" in ended and off:
                break
            for kb in range(k_hold - 1, -1, -1):                     # the MCP steps back
                send(q0 + a.step * kb)
                time.sleep(a.dwell)
            dump(f"unload {N:g}")
            write({"phase": "unload", "N": N, "dF_P": np.round(dFP(0.3), 4).tolist(), "t": time.time()})
            if ended:
                break
    finally:
        dump("end")
        write({"end": time.time(), "early": early, "ended": ended})
        fh.close()
        raw.close()
        samp.halt = True
        tcp.halt = True
        arm.close()
        ft.stop = True
    print(f"S4(b) {f} {tag}: early stops {early or 'none'}{'; ENDED: ' + ended if ended else ''}; -> {stem}.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
