"""Free-air step responses of the SR2 hand's nine servos, through the station's policy stream with read-back (50 Hz),
one joint at a time from the hand's hold pose (owner, 2026-11-17 16:1x: "i am now powering the servos on 7.5V ... please
experiment with this carefully, but see if we can improve stiffness/speed/force with these 50% higher voltage").

Each step: --hold-s at the start pose, the joint stepped by its delta for --step-s, back to the start for --step-s. The
yaws step +-YAW_DEG (the thumb's +-20 deg, as the yaw holds of 2026-11-06 at 5 V; the index's and the middle's +-15 deg,
one finger at a time, so a tip moves at most ~30 mm toward a neighbour 48 mm away), the MCPs and PIPs flex +20 deg.
Every frame is clipped to the servos' safe range (sr2hand.control.servo_limits); the index PIP starts from its +1.78 deg
floor. The sequence stops when a joint not being stepped moves more than --guard-deg from its start, or the stepped
joint ends a hold more than --guard-deg short of its target. The station holds the last goal (the start pose) after.

Writes logs/20261117-sr2_servo_7v5/<stamp>_steps_speed<S>.jsonl: a header (the start pose, the speed, the sequence),
then per frame t (corvus's clock, s), the step's index, its time from the step's command, the target (sim degrees) and
the nine read-back joints (sim degrees).

    /usr/bin/python3 docs/experiments/20261117-sr2_servo_7v5/lab/step_test.py --speed 80 [--go]
"""
import argparse
import http.client
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/sr2hand/driver/manta/host")]
from manta_hand import plan as P  # noqa: E402
from sr2hand.control import servo_limits as SL  # noqa: E402

FINGERS = ("thumb", "index", "middle")
J = ("yaw", "mcp", "pip")
COL = {"yaw": "aa", "mcp": "fe1", "pip": "fe2"}
LOG = ROOT / "logs/20261117-sr2_servo_7v5"


def sequence():
    seq = []
    for f, a in (("thumb", 20.0), ("index", 15.0), ("middle", 15.0)):
        seq += [(f, "yaw", +a), (f, "yaw", -a)]
    for f in FINGERS:
        for j in ("mcp", "pip"):
            seq.append((f, j, 20.0))
    return seq


class Stream:
    def __init__(self, url, token_file):
        u = url.split("//", 1)[1]
        self.host, self.port = u.split(":")[0], int(u.split(":")[1].split("/")[0])
        self.c = http.client.HTTPConnection(self.host, self.port, timeout=2.0)
        self.token = Path(token_file).expanduser().read_text().strip()
        self.st = None

    def call(self, method, path, body=None):
        hdr = {"X-Manta-Token": self.token}
        if body is not None:
            hdr["Content-Type"] = "application/json"
        for attempt in (0, 1):                  # the station closes an idle keep-alive connection: reconnect once
            try:
                self.c.request(method, path, None if body is None else json.dumps(body), hdr)
                r = self.c.getresponse()
                break
            except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError):
                self.c.close()
                self.c = http.client.HTTPConnection(self.host, self.port, timeout=2.0)
                if attempt:
                    raise
        out = r.read()
        if r.status != 200:
            raise RuntimeError(f"{path}: {r.status} {out[:300]!r}")
        return json.loads(out or b"{}")

    def joints_sim(self, servos):
        return {f: {j: float(servos[str(k)][COL[j]]) / P.JOINT_SIGN[(f, j)] for j in J} for k, f in enumerate(FINGERS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--station", default="http://10.99.99.2:8765")
    ap.add_argument("--token-file", default="~/.sr2_station_token")
    ap.add_argument("--speed", type=int, default=80)
    ap.add_argument("--hold-s", type=float, default=0.5)
    ap.add_argument("--step-s", type=float, default=1.2)
    ap.add_argument("--guard-deg", type=float, default=6.0)
    ap.add_argument("--only", default=None, help="comma-separated finger.joint to step (default: all nine)")
    ap.add_argument("--go", action="store_true", help="move; without it the sequence is only printed")
    a = ap.parse_args()
    s = Stream(a.station, a.token_file)
    st = s.call("GET", "/api/v1/state")
    if st.get("servo_torque") != 1:
        raise SystemExit(f"servo torque is {st.get('servo_torque')}, not on")
    start = s.joints_sim(st["telemetry"]["servos"])
    clipped = SL.clip_sim(start)
    seq = [x for x in sequence() if a.only is None or f"{x[0]}.{x[1]}" in a.only.split(",")]
    print("start (sim deg): " + "  ".join(f"{f} " + " ".join(f"{start[f][j]:+.1f}" for j in J) for f in FINGERS)
          + (f"  (clipped to the safe range: {clipped})" if clipped else ""))
    for f, j, d in seq:
        lo, hi = SL.safe_range_sim(f, j)
        tgt = start[f][j] + d
        print(f"   {f}.{j}: {start[f][j]:+.1f} -> {tgt:+.1f} -> back  (safe range {lo:+.1f} .. {hi:+.1f})"
              + ("  OUTSIDE" if not lo <= tgt <= hi else ""))
        if not lo <= tgt <= hi:
            raise SystemExit("a step leaves the safe range")
    if not a.go:
        return 0
    LOG.mkdir(parents=True, exist_ok=True)
    path = LOG / f"{time.strftime('%Y%m%d-%H%M%S')}_steps_speed{a.speed}.jsonl"
    fh = open(path, "w")
    fh.write(json.dumps({"kind": "header", "speed": a.speed, "start": start, "sequence": seq, "hold_s": a.hold_s,
                         "step_s": a.step_s, "station": a.station}) + "\n")
    tok = s.call("POST", "/api/v1/stream/start", {"timeout_s": 1.0})["token"]
    stop, first = None, True
    try:
        for i, (f, j, d) in enumerate(seq):
            phases = [(a.hold_s, 0.0), (a.step_s, d), (a.step_s, 0.0)]
            for ph, (dur, delta) in enumerate(phases):
                target = {g: dict(start[g]) for g in FINGERS}
                target[f][j] = start[f][j] + delta
                SL.clip_sim(target)
                t_cmd = time.time()
                nxt = time.perf_counter()
                last = None
                while time.time() - t_cmd < dur:
                    body = {"token": tok, "joints": target, "read_back": True}
                    if first:
                        body["servo_speed"] = a.speed
                        first = False
                    r = s.call("POST", "/api/v1/stream/frame", body)
                    t = time.time()
                    sv = r.get("servos")
                    q = s.joints_sim(sv) if sv else None
                    fh.write(json.dumps({"t": round(t, 4), "step": i, "phase": ph, "t_cmd": round(t - t_cmd, 4),
                                         "target": target, "q": q}) + "\n")
                    if q is not None:
                        last = q
                        moved = [f"{g}.{k}" for g in FINGERS for k in J
                                 if (g, k) != (f, j) and abs(q[g][k] - start[g][k]) > a.guard_deg]
                        if moved:
                            stop = f"{moved} moved more than {a.guard_deg} deg from the start during {f}.{j}"
                            break
                    nxt += 0.02
                    dt = nxt - time.perf_counter()
                    if dt > 0:
                        time.sleep(dt)
                    else:
                        nxt = time.perf_counter()
                if stop:
                    break
                if last is not None and abs(last[f][j] - target[f][j]) > a.guard_deg:
                    stop = f"{f}.{j} ended its hold {abs(last[f][j] - target[f][j]):.1f} deg from its target"
                    break
            print(f"{time.strftime('%H:%M:%S')} step {i} {f}.{j} {d:+.0f}: "
                  + ("done" if not stop else f"STOPPED: {stop}"), flush=True)
            if stop:
                break
    finally:
        try:
            s.call("POST", "/api/v1/stream/end", {"token": tok})
        except Exception as e:  # noqa: BLE001
            print(f"stream end: {e!r}")
        fh.write(json.dumps({"kind": "end", "stop": stop}) + "\n")
        fh.close()
    print(f"-> {path}")
    return 1 if stop else 0


if __name__ == "__main__":
    sys.exit(main())
