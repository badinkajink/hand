"""S3 of the identification queue (QUEUE.md; the owner's notes of 2026-11-18 09:0x and 09:55): the torque scale at
7.5 V. One joint of one finger steps its target past its reading, --step deg every --dwell s, toward a rigid face that
the finger's pad rests 1-2 mm from (a flat land of the owner's 80/20 block, never across a T-slot). The owner or the arm
places the face; this tool never moves the arm. It records the HEX-E (100 Hz), the station's joints and loads (20 Hz),
each step's servo flags, torque and temperatures (the station's diagnostics) and, with --tag, the finger's DIP tag in
both cameras; the fingertip boards record in the splitter's file. The steps go on past contact until the HEX-E's force
has changed --max-n from its mean before the first step, a servo flags or loses torque, the pressing servo's load
passes --max-load, a servo reaches --max-temp, or the target is --max-offset past the reading. Then the target steps
back the same way to where it started.

    /usr/bin/python3 s3_press.py check
        the HEX-E's mean and noise over 2 s, the station's torque, joints, loads and temperatures; no motion
    /usr/bin/python3 s3_press.py run --finger index --joint yaw --sign -1 --out OUT [--tag] [--go]
        without --go, the plan and the same checks; --sign is the direction (simulator degrees) that moves the pad
        toward the face

OUT/<stamp>_<finger>_<joint>.jsonl: a header, one line per step (target, reading, the pressing servo's load, the
HEX-E's force change from before the first step, mean over the dwell's last second, and with --tag the tag's position in
P), an end line with the reason. OUT/<stamp>_<finger>_<joint>_raw.jsonl holds the HEX-E's and the station's samples. The
output is force against target offset; the workstation fits the gain, the give and the dead band (W2, W3). Prediction
for the index's and middle's yaws (W2, 11-18): the reading advances about 4 deg past first contact by the time the
target is 14-15 deg past it, at about 1.0 N if the gear passes 0.4 of the servo's torque and 2.4 N if it passes all.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LAB = ROOT / "docs/experiments/20261115-hom_bench_trials/lab"
FINGERS = ("thumb", "index", "middle")
JOINTS = ("yaw", "mcp", "pip")


def imports():
    sys.path[:0] = [str(ROOT / "src"), str(LAB), str(ROOT / "src/sr2hand/driver/reskin5x/host")]
    import fingertip_tag_check as FT
    import hom_setup as HS
    from sr2hand.control import servo_limits as SL
    return FT, HS, SL


class Sampler(threading.Thread):
    """The station's joints (simulator degrees) and loads at about hz, kept as [(t, joints, loads)]."""

    def __init__(self, HS, st, hz=20.0):
        super().__init__(daemon=True)
        self.HS, self.st, self.dt = HS, st, 1.0 / hz
        self.rows, self.lock, self.halt, self.error = [], threading.Lock(), False, None

    def run(self):
        while not self.halt:
            try:
                s = self.st.state()
                row = (time.time(), self.HS.sim_joints(s), (s.get("telemetry") or {}).get("servo_load") or {})
                with self.lock:
                    self.rows.append(row)
            except Exception as e:                       # noqa: BLE001
                self.error = repr(e)
            time.sleep(self.dt)

    def since(self, t0):
        with self.lock:
            return [r for r in self.rows if r[0] >= t0]


def num(x, spec):
    return "-" if x is None else format(x, spec)


def hexe_mean(ft, seconds):
    t, v = ft.ring.since(time.time() - seconds)
    return (None, None, len(t)) if len(t) < 5 else (v[:, :3].mean(axis=0), v[:, :3].std(axis=0), len(t))


def diag(st):
    d = st.call("POST", "/api/v1/servos/diagnostics")
    return {k: d.get(k) for k in ("status", "temperature", "torque_enable", "voltage")}


def cmd_check(a):
    FT, HS, SL = imports()
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    ft = FT.HT.open_ft(True)
    time.sleep(2.5)
    F, sd, n = hexe_mean(ft, 2.0)
    ft.stop = True
    print("HEX-E: " + ("no samples: is it on?" if F is None else
                       f"{n} samples in 2 s, mean {np.round(F, 2).tolist()} N, sd {np.round(sd, 3).tolist()} N"))
    tel = s.get("telemetry") or {}
    j = HS.sim_joints(s)
    print(f"station: torque {s.get('servo_torque')}, mounts {s.get('mount_positions')}")
    print("joints " + "  ".join(f"{f} " + " ".join(f"{k} {j[f][k]:+.1f}" for k in JOINTS) for f in FINGERS))
    print(f"loads {tel.get('servo_load')}; temperatures {tel.get('servo_temperature')}")
    print(f"diagnostics {diag(st)}")
    return 0


def cmd_run(a):
    FT, HS, SL = imports()
    P, SV = SL._driver()
    f, jn = a.finger, a.joint
    sids = {k: str(SV.FINGER_JOINTS[P.FINGER_ID[f]][P.SIM_JOINT_TO_SERVO[k]][0]) for k in JOINTS}
    sid = sids[jn]
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    if s.get("servo_torque") != 1:
        raise SystemExit("servo torque is not on")
    start = dict(HS.sim_joints(s)[f])
    q0 = float(start[jn])
    lo, hi = st.call("GET", "/api/v1/manual/limits")["joint_deg"][f][jn]
    n_max = int(a.max_travel / a.step)
    while n_max > 0 and not lo + 1.0 <= q0 + a.sign * a.step * n_max <= hi - 1.0:
        n_max -= 1
    if n_max < 4:
        raise SystemExit(f"{f} {jn} at {q0:+.1f} deg has under 2 deg of the station's range [{lo:g}, {hi:g}] toward "
                         f"sign {a.sign:+d}")
    ft = FT.HT.open_ft(True)
    time.sleep(1.5)
    F0, sd0, n0 = hexe_mean(ft, 1.0)
    if F0 is None:
        ft.stop = True
        raise SystemExit(f"the HEX-E gave {n0} samples in 1 s: is it on?")
    dg = diag(st)
    print(f"{f} {jn} (servo {sid}) from {q0:+.2f} deg, sign {a.sign:+d}: up to {n_max} steps of {a.step:g} deg every "
          f"{a.dwell:g} s (target to {q0 + a.sign * a.step * n_max:+.1f}, the station's range [{lo:g}, {hi:g}]); "
          f"the finger's other joints held at " + ", ".join(f"{k} {start[k]:+.1f}" for k in JOINTS if k != jn))
    print(f"stops: |dF| {a.max_n:g} N on the HEX-E, the target {a.max_offset:g} deg past the reading, load "
          f"{a.max_load:g}, a flag or torque off on the finger's servos, {a.max_temp:g} C; then back the same way")
    print(f"HEX-E before: {np.round(F0, 2).tolist()} N (sd {np.round(sd0, 3).tolist()}, {n0} samples in 1 s); "
          f"diagnostics {dg}")
    if not a.go:
        ft.stop = True
        return 0
    import tip_contacts as TC
    try:
        pids, how = TC.ensure_splitter(start=True)
    except RuntimeError as e:
        ft.stop = True
        raise SystemExit(f"fingertip boards: {e}")
    mux = sorted((ROOT / "logs").glob("*-three_finger_mux/*-mux.bin"), key=lambda p: p.stat().st_mtime)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{time.strftime('%Y%m%d-%H%M%S')}_{f}_{jn}"
    samp = Sampler(HS, st)
    samp.start()
    tags, tlock, arm, tracker = [], threading.Lock(), None, None
    if a.tag:
        def on(smp):
            m = smp.msg
            if m is not None and m.n_tags > 0 and np.all(np.isfinite(m.p_PO)):
                with tlock:
                    tags.append((time.time(), np.array(m.p_PO, float), int(m.n_cams)))

        class _A:
            map = FT.tag_map_file(FT.TAG[f])
        arm, tracker = FT.HT.start_tracker(_A, on)
    hdr = {"what": "S3 press", "finger": f, "joint": jn, "servo": sid, "sign": a.sign, "step_deg": a.step,
           "dwell_s": a.dwell, "speed": a.speed, "start_joints": start, "range": [lo, hi], "F0_N": F0.tolist(),
           "F0_sd_N": sd0.tolist(), "stops": {"max_n": a.max_n, "max_offset": a.max_offset, "max_load": a.max_load,
                                              "max_temp": a.max_temp}, "diag": dg, "splitter": how,
           "mux": str(mux[-1].relative_to(ROOT)) if mux else None, "mounts": s.get("mount_positions"),
           "tag": a.tag, "t": time.time(), "note": a.note}
    fh, raw = open(f"{stem}.jsonl", "w"), open(f"{stem}_raw.jsonl", "w")
    fh.write(json.dumps(hdr) + "\n")
    t_raw = [time.time() - 1.0]

    def dump():
        t1 = time.time()
        tt, vv = ft.ring.since(t_raw[0], t1)
        raw.write(json.dumps({"hexe_t": tt.tolist(), "hexe": np.round(vv, 4).tolist(),
                              "station": [(r[0], r[1][f], {k: r[2].get(sids[k]) for k in JOINTS})
                                          for r in samp.since(t_raw[0]) if r[1] and r[0] <= t1]}) + "\n")
        raw.flush()
        t_raw[0] = t1

    def send(target):
        cmd = dict(start)
        cmd[jn] = float(target)
        st.call("POST", "/api/v1/manual/joints", {"joints": {f: cmd}, "servo_speed": a.speed})

    def dwell_and_row(k, target, phase):
        """Waits --dwell s watching the HEX-E and the load; returns (row, stop reason or None)."""
        t0, why = time.time(), None
        while time.time() - t0 < a.dwell:
            time.sleep(0.1)
            t_l, _ = ft.ring.latest()
            if t_l is None or time.time() - t_l > 0.5:
                why = "the HEX-E stopped streaming"
                break
            Fn, _, _ = hexe_mean(ft, 0.2)
            if Fn is not None and np.linalg.norm(Fn - F0) >= a.max_n and phase == "press":
                why = f"|dF| {np.linalg.norm(Fn - F0):.2f} N reached {a.max_n:g} N"
                break
            rs = samp.since(time.time() - 0.2)
            if rs and abs(float(rs[-1][2].get(sid) or 0.0)) >= a.max_load and phase == "press":
                why = f"load {rs[-1][2].get(sid)} reached {a.max_load:g}"
                break
        tw = time.time() - min(1.0, a.dwell / 2)
        Fm, Fsd, nF = hexe_mean(ft, min(1.0, a.dwell / 2))
        rs = samp.since(tw)
        reading = float(np.median([r[1][f][jn] for r in rs if r[1]])) if rs else None
        load = float(np.median([float(r[2].get(sid) or 0.0) for r in rs])) if rs else None
        d = diag(st)
        row = {"k": k, "phase": phase, "t": time.time(), "target": round(target, 3), "reading": reading,
               "offset": None if reading is None else round(a.sign * (target - reading), 3), "load": load,
               "dF_N": None if Fm is None else np.round(Fm - F0, 4).tolist(),
               "dF_abs_N": None if Fm is None else round(float(np.linalg.norm(Fm - F0)), 4),
               "F_sd_N": None if Fsd is None else np.round(Fsd, 4).tolist(), "hexe_n": nF,
               "temp": {k2: (d.get("temperature") or {}).get(sids[k2]) for k2 in JOINTS},
               "status": {k2: (d.get("status") or {}).get(sids[k2]) for k2 in JOINTS},
               "torque": {k2: (d.get("torque_enable") or {}).get(sids[k2]) for k2 in JOINTS}}
        if a.tag:
            with tlock:
                tg = [x for x in tags if x[0] >= tw]
            row["tag_P"] = np.round(np.median([x[1] for x in tg], axis=0), 2).tolist() if len(tg) >= 3 else None
            row["tag_n"] = len(tg)
        if why is None and phase == "press":
            if any(v not in (None, 0, 0.0) for v in row["status"].values()):
                why = f"a servo flag {row['status']}"
            elif any(v not in (None, 1, 1.0) for v in row["torque"].values()):
                why = f"torque off {row['torque']}"
            elif any(v is not None and float(v) >= a.max_temp for v in row["temp"].values()):
                why = f"temperature {row['temp']}"
            elif row["offset"] is not None and row["offset"] >= a.max_offset:
                why = f"the target {row['offset']:.1f} deg past the reading"
        return row, why

    why, k, rows = None, 0, []
    try:
        for k in range(1, n_max + 1):
            target = q0 + a.sign * a.step * k
            send(target)
            row, why = dwell_and_row(k, target, "press")
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            dump()
            print(f"{time.strftime('%H:%M:%S')} step {k:2d}: target {target:+6.2f}, reading {num(row['reading'], '+6.2f')} "
                  f"(offset {num(row['offset'], '+5.2f')}), load {num(row['load'], '5.0f')}, |dF| "
                  f"{num(row['dF_abs_N'], '.3f')} N"
                  + (f", tag {row.get('tag_P')}" if a.tag else "") + (f" -- STOP: {why}" if why else ""), flush=True)
            if why:
                break
        else:
            why = f"the target's travel ({n_max} steps) ran out"
        torque_on = all(v in (None, 1, 1.0) for v in rows[-1]["torque"].values()) if rows else True
        if torque_on:
            for kb in range(k - 1, -1, -1):
                target = q0 + a.sign * a.step * kb
                send(target)
                row, _ = dwell_and_row(kb, target, "back")
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                dump()
                print(f"{time.strftime('%H:%M:%S')} back {kb:2d}: target {target:+6.2f}, reading "
                      f"{num(row['reading'], '+6.2f')}, |dF| {num(row['dF_abs_N'], '.3f')} N", flush=True)
        else:
            print("a servo of the finger lost torque: no step back. Torque on only through hom_setup.py torque-on "
                  "--go, with every MCP under 60 deg", flush=True)
    finally:
        fh.write(json.dumps({"end": time.time(), "reason": why, "steps": k}) + "\n")
        fh.close()
        dump()
        raw.close()
        samp.halt = True
        ft.stop = True
        if tracker is not None:
            tracker.stop()
            arm.stop()
    print(f"S3 {f} {jn}: {why}; -> {stem}.jsonl")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run"):
        p = sub.add_parser(name)
        p.add_argument("--station", default="http://10.99.99.2:8765")
        p.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
        if name == "run":
            p.add_argument("--finger", required=True, choices=FINGERS)
            p.add_argument("--joint", required=True, choices=JOINTS)
            p.add_argument("--sign", type=int, required=True, choices=(-1, 1))
            p.add_argument("--out", default=str(ROOT / "logs" / time.strftime("%Y%m%d-sr2_s3_press")))
            p.add_argument("--step", type=float, default=0.5)
            p.add_argument("--dwell", type=float, default=2.0)
            p.add_argument("--speed", type=int, default=20)
            p.add_argument("--max-travel", type=float, default=25.0, help="deg of target travel at most")
            p.add_argument("--max-n", type=float, default=3.0)
            p.add_argument("--max-offset", type=float, default=18.0)
            p.add_argument("--max-load", type=float, default=900.0)
            p.add_argument("--max-temp", type=float, default=55.0)
            p.add_argument("--tag", action="store_true", help="record the finger's DIP tag with the tracker")
            p.add_argument("--note", default="")
            p.add_argument("--go", action="store_true")
    a = ap.parse_args()
    return {"check": cmd_check, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
