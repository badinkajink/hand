"""Run a sweep plan of make_plan.py on the SR2 hand through the control station (manta_hand.web on the CB1), from
corvus. Every step's command, settle and end, and the station's telemetry (nine servo positions, loads, temperatures,
torque, steppers) are written on corvus's clock, the clock of the fingertip boards' log; the CB1's clock is hours off.

    /usr/bin/python3 run_plan.py PLAN OUTDIR --token-file F [--passes N] [--until HH:MM] [--resume]

OUTDIR gets events.jsonl (one line a step and a safety stop), telemetry.jsonl (one line a new station sample, with
t_recv and t_sample = t_recv - the sample's age) and progress.json (pass and step done; --resume skips them).
touch OUTDIR/STOP ends the run after the current step. Safety stops, each followed by the zero pose and the end:
a station error or lost link; a servo at 60 C or more; a joint 6 deg or more from its command 6 s after the command
(a finger held off its pose); a servo alarm. At the end the hand goes to the zero pose and the motors are disabled
(steppers off, servo torque off), which also gives up the home.
"""
import argparse
import datetime as dt
import json
import threading
import time
import urllib.request
from pathlib import Path

F = ("thumb", "index", "middle")
FID = {"thumb": "0", "index": "1", "middle": "2"}
YAW_SIGN = {"thumb": 1.0, "index": -1.0, "middle": -1.0}     # manta_hand.plan.JOINT_SIGN, aa
SERVO_IDS = {"thumb": (0, 1, 2), "index": (3, 4, 5), "middle": (6, 7, 8)}
ap = argparse.ArgumentParser()
ap.add_argument("plan")
ap.add_argument("out")
ap.add_argument("--token-file", required=True)
ap.add_argument("--station", default="http://10.99.99.2:8765")
ap.add_argument("--passes", type=int, default=1)
ap.add_argument("--until", default=None, help="HH:MM: no pass starts after this time (next day if earlier than now)")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--no-disable", action="store_true",
                help="leave the motors on when the plan ends normally (the next plan of a chain runs on this home)")
ap.add_argument("--max-temp", type=float, default=60.0)
ap.add_argument("--max-err", type=float, default=6.0)
a = ap.parse_args()
plan = json.load(open(a.plan))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
TOKEN = Path(a.token_file).read_text().strip()
EV = open(out / "events.jsonl", "a", buffering=1)
TL = open(out / "telemetry.jsonl", "a", buffering=1)
STOP = out / "STOP"
latest = {"state": None, "t": 0.0}
lock = threading.Lock()
quit_flag = threading.Event()


class Abort(Exception):
    pass


def http(method, path, body=None, timeout=10.0):
    req = urllib.request.Request(a.station + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Manta-Token": TOKEN})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def event(kind, **kw):
    EV.write(json.dumps({"t": time.time(), "kind": kind, **kw}) + "\n")


def poller():
    last_ts = None
    fails = 0
    while not quit_flag.is_set():
        try:
            s = http("GET", "/api/v1/state", timeout=5.0)
            t = time.time()
            fails = 0
            with lock:
                latest["state"], latest["t"] = s, t
            tel = s["telemetry"]
            if tel.get("timestamp") != last_ts:
                last_ts = tel.get("timestamp")
                TL.write(json.dumps({"t_recv": t, "t_sample": t - (tel.get("age_s") or 0.0),
                                     "servos": tel.get("servos"), "load": tel.get("servo_load"),
                                     "temperature": tel.get("servo_temperature"), "torque": tel.get("servo_torque"),
                                     "steppers": tel.get("steppers"), "operation": s.get("operation"),
                                     "last_command": s.get("last_command"), "error": tel.get("error"),
                                     "last_error": s.get("last_error")}) + "\n")
        except Exception as e:
            fails += 1
            if fails in (1, 10, 100):
                event("poll_error", error=str(e), fails=fails)
        time.sleep(0.1)


def state():
    with lock:
        return latest["state"], latest["t"]


def check(ctx=""):
    s, t = state()
    if s is None:
        return
    if time.time() - t > 15:
        raise Abort(f"no station state for {time.time() - t:.0f} s {ctx}")
    if s.get("link_down"):
        raise Abort(f"link down: {s['link_down']} {ctx}")
    tel = s["telemetry"]
    temps = tel.get("servo_temperature") or {}
    hot = {k: v for k, v in temps.items() if v is not None and v >= a.max_temp}
    if hot:
        raise Abort(f"servo temperature {hot} {ctx}")


def servo_target(joints):
    """sim degrees -> the telemetry's servo degrees (aa, fe1, fe2) per finger id"""
    return {FID[f]: {"aa": YAW_SIGN[f] * joints[f]["yaw"], "fe1": joints[f]["mcp"], "fe2": joints[f]["pip"]}
            for f in F}


def joint_error(joints):
    s, _ = state()
    sv = s["telemetry"].get("servos") or {}
    tgt = servo_target(joints)
    worst = (0.0, None)
    for fid, js in tgt.items():
        for j, v in js.items():
            have = (sv.get(fid) or {}).get(j)
            if have is None:
                continue
            e = abs(have - v)
            if e > worst[0]:
                worst = (e, f"{fid}.{j}")
    return worst


def wait_idle(timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        check("waiting for idle")
        s, _ = state()
        if s and s.get("operation") == "idle":
            return time.time() - t0
        time.sleep(0.2)
    raise Abort(f"station still busy after {timeout:.0f} s")


def sleep_checked(sec, ctx):
    t0 = time.time()
    while time.time() - t0 < sec:
        check(ctx)
        time.sleep(min(0.25, max(0.0, sec - (time.time() - t0))))


def servo_vector():
    s, t = state()
    sv = s["telemetry"].get("servos") or {}
    return t, [(sv.get(fid) or {}).get(j) for fid in ("0", "1", "2") for j in ("aa", "fe1", "fe2")]


def do_joints(st, ctx):
    t_cmd = time.time()
    http("POST", "/api/v1/manual/joints", {"joints": st["joints"], "servo_speed": st.get("speed", 80)})
    # settled: a fresh sample within 2 deg of the command, or two fresh samples 1.2 s or more after the command that
    # differ by under 0.4 deg on every joint (stopped, e.g. drooping under gravity); at most 6 s. A joint 6 deg or more
    # from its command then stops the run.
    settle, prev = None, None
    while time.time() - t_cmd < 6.0:
        check(ctx)
        t, v = servo_vector()
        if t > t_cmd + 0.3 and (prev is None or t > prev[0]):
            e, _ = joint_error(st["joints"])
            if e < 2.0:
                settle = time.time() - t_cmd
                break
            if (prev is not None and t > t_cmd + 1.2 and None not in v and None not in prev[1]
                    and max(abs(x - y) for x, y in zip(v, prev[1])) < 0.4):
                settle = time.time() - t_cmd
                break
            prev = (t, v)
        time.sleep(0.05)
    e, which = joint_error(st["joints"])
    if e >= a.max_err:
        sleep_checked(1.0, ctx)
        e, which = joint_error(st["joints"])
        if e >= a.max_err:
            raise Abort(f"joint {which} {e:.1f} deg from its command after {time.time() - t_cmd:.1f} s {ctx}")
    t_settled = time.time()
    sleep_checked(st.get("dwell_s", 3.0), ctx)
    return {"t_cmd": t_cmd, "t_settled": t_settled, "t_end": time.time(), "settle_s": settle,
            "err_deg": round(e, 2), "err_joint": which}


def do_torque(st, ctx):
    t_cmd = time.time()
    rb = http("POST", "/api/v1/servos/torque", {"state": st["state"]}, timeout=30.0)
    t_settled = time.time()
    sleep_checked(st.get("dwell_s", 10.0), ctx)
    return {"t_cmd": t_cmd, "t_settled": t_settled, "t_end": time.time(), "readback": rb}


def do_mounts(st, ctx):
    t_cmd = time.time()
    http("POST", "/api/v1/manual/mounts", {"mounts": st["mounts"]})
    time.sleep(0.5)
    wait_idle(120.0)
    s, _ = state()
    t_settled = time.time()
    sleep_checked(st.get("dwell_s", 10.0), ctx)
    return {"t_cmd": t_cmd, "t_settled": t_settled, "t_end": time.time(), "mount_positions": s.get("mount_positions")}


ZERO = {f: {"yaw": 0.0, "mcp": 0.0, "pip": 0.0} for f in F}


def to_zero(why):
    try:
        s, _ = state()
        if s and s["telemetry"].get("servo_torque") and any(v != 1 for v in s["telemetry"]["servo_torque"].values()):
            http("POST", "/api/v1/servos/torque", {"state": "on"}, timeout=30.0)
        wait_idle(120.0)
        http("POST", "/api/v1/manual/joints", {"joints": ZERO, "servo_speed": 60})
        time.sleep(3.0)
        event("zero_pose", why=why, err=joint_error(ZERO))
    except Exception as e:
        event("zero_pose_failed", why=why, error=str(e))


def deadline():
    if not a.until:
        return None
    hh, mm = map(int, a.until.split(":"))
    now = dt.datetime.now()
    d = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if d < now - dt.timedelta(hours=1):
        d += dt.timedelta(days=1)
    return d.timestamp()


prog_path = out / "progress.json"
done = json.loads(prog_path.read_text()) if a.resume and prog_path.exists() else {"pass": 0, "step": -1}
th = threading.Thread(target=poller, daemon=True)
th.start()
time.sleep(1.5)
until = deadline()
event("start", plan=a.plan, passes=a.passes, until=a.until, resume=done, n_steps=len(plan["steps"]))
reason = "plan done"
ok = True
try:
    s, _ = state()
    if s is None:
        raise Abort("no station state at start")
    if not s.get("homed"):
        raise Abort("station not homed")
    wait_idle(30.0)
    for p in range(done["pass"], a.passes):
        if until and time.time() > until and p > done["pass"]:
            reason = f"deadline {a.until} before pass {p}"
            break
        for i, st in enumerate(plan["steps"]):
            if p == done["pass"] and i <= done["step"]:
                continue
            if STOP.exists():
                raise Abort("STOP file")
            ctx = f"(pass {p} step {i} {st['family']})"
            fn = {"joints": do_joints, "torque": do_torque, "mounts": do_mounts}[st["op"]]
            r = fn(st, ctx)
            event("step", **{"pass": p, "step": i, "op": st["op"], "family": st["family"],
                             "joints": st.get("joints"), "mounts": st.get("mounts"), "state": st.get("state"),
                             "clearance_mm": st.get("clearance_mm")}, **r)
            prog_path.write_text(json.dumps({"pass": p, "step": i}))
        prog_path.write_text(json.dumps({"pass": p + 1, "step": -1}))
        done = {"pass": p + 1, "step": -1}
except Abort as e:
    reason, ok = f"safety stop: {e}", False
except BaseException as e:
    reason, ok = f"error: {type(e).__name__}: {e}", False
finally:
    event("ending", reason=reason)
    to_zero(reason)
    if not (ok and a.no_disable):
        try:
            time.sleep(2.0)
            r = http("POST", "/api/v1/motors/disable", {}, timeout=60.0)
            event("motors_disabled", result=r)
        except Exception as e:
            event("disable_failed", error=str(e))
    time.sleep(2.0)
    quit_flag.set()
    event("end", reason=reason)
    print(reason, flush=True)
raise SystemExit(0 if ok else 3)
