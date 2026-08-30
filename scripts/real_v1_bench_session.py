#!/usr/bin/env python3
"""Operator-driven bench session: one design, one sitting, one directory to hand back.

Why this exists.  The 2026-08-29 bench produced four good runs and three orphan
JSONLs whose meaning had to be reconstructed a day later from timestamps.  The
runs were fine; the SESSION was not recorded.  This driver runs the same three
scripts in the same order, but it also captures what the log cannot: which arm
this run was, what the operator SAW (there is no object-pose sensor on this
hand), and the preflight state that makes a run interpretable at all.

It writes ONE directory per session containing every raw log, every stdout, and
a MANIFEST.json -- self-describing, so the whole thing can be handed to an
analyst (or a fresh session) with no other context.

  python3 scripts/real_v1_bench_session.py --design g12
  python3 scripts/real_v1_bench_session.py --design g24 --arms freeair
  python3 scripts/real_v1_bench_session.py --design g12 --arms loaded --repeats 3

Arms, in the order they must be run:
  freeair   the turn with NOTHING in the hand.  No clamp, no drop, no risk.
            Establishes the design's own tracking/droop floor -- without it a
            loaded shortfall cannot be attributed to grip (this is exactly the
            control the 2026-08-29 runs nearly went without).
  grip      seat the grip, then walk it down to a per-finger load band.
            Records the grip window itself, which is a result, not a setup step.
  loaded    the instrumented turn with the shaft in hand, repeated.
  protection  prints the servo overload-protection procedure (needs the CB1 bus,
            so it cannot run from here) and records the operator's readings.
"""
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import mh

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "docs/experiments/20260829-real_v1_deploy/deploy"
SUITE = REPO / "docs/experiments/20260830-real_v1_bench_suite"
FINGERS = ("thumb", "index", "middle")
FID = {"thumb": "0", "index": "1", "middle": "2"}
KEY = {"yaw": "aa", "mcp": "fe1", "pip": "fe2"}
SIGN = {"thumb": 1.0, "index": -1.0, "middle": -1.0}

# Largest fraction of the turn that still clears finger<->finger by >=3mm in the
# design's OWN sim geometry (scripts/real_v1_trajectory_clearance.py --all, and
# the sim links are THINNER than the printed parts, so these are optimistic).
# Only g12 clears the whole path; the rest are truncated rather than skipped,
# which is what makes "try them all" safe.
SAFE_U = {
    "g12":      {"chord": 1.00, "csv": 1.00},
    "g23":      {"chord": 0.92, "csv": 0.84},
    "g24":      {"chord": 0.55, "csv": 0.10},
    "rv04_mid": {"chord": 0.65, "csv": 0.70},
}
# margin already applied: g24/rv04_mid backed off ~1 step from the 3mm crossing.

GRIP_FIRM, GRIP_FREE = 440.0, 0.0   # |fe1| load targets for holder vs driver


def say(*a):
    print(*a, flush=True)


def ask(prompt, default=None, cast=str, allow_blank=False):
    """Operator prompt.  Non-interactive stdin returns the default."""
    if not sys.stdin.isatty():
        return default
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  ?? {prompt}{suffix}: ").strip()
        if not raw:
            if default is not None or allow_blank:
                return default
            continue
        try:
            return cast(raw)
        except ValueError:
            say("     (not a number)")


def ask_bool(prompt, default=None):
    d = None if default is None else ("y" if default else "n")
    v = ask(prompt + " (y/n)", d)
    return None if v is None else str(v).lower().startswith("y")


def read_fresh(max_age=0.4, tries=40):
    for _ in range(tries):
        t = mh.get("/state")["telemetry"]
        if not t.get("servo_polling_suspended") and (t.get("servo_age_s") or 9) < max_age:
            return t
        time.sleep(0.05)
    raise RuntimeError("no fresh, unsuspended servo sample -- is another writer on the bus?")


def preflight(design):
    """Everything that makes a run interpretable later, captured before it runs."""
    s = mh.get("/state")
    t = read_fresh()
    pf = {
        "backend": s.get("backend"), "homed": s.get("homed"),
        "busy": s.get("busy"), "operation": s.get("operation"),
        "servo_torque": s.get("servo_torque"),
        "loaded_plan": (s.get("plan") or {}).get("name") if isinstance(s.get("plan"), dict) else s.get("plan"),
        "servo_age_s": t.get("servo_age_s"),
        "servo_polling_suspended": t.get("servo_polling_suspended"),
        "servo_load": t.get("servo_load"),
        "servos": t.get("servos"),
    }
    problems = []
    if pf["backend"] != "real":
        problems.append(f"backend is {pf['backend']!r}, not 'real'")
    if pf["busy"]:
        problems.append(f"service is busy with {pf['operation']!r}")
    if not pf["servo_torque"]:
        problems.append("servo torque is OFF -- enable it in the web app first")
    if pf["servo_polling_suspended"]:
        problems.append("telemetry is suspended: another writer owns the bus")
    if not pf["homed"]:
        # /stream/start 409s with "home the gantries once in this daemon session
        # first", and it does so on the FIRST FRAME -- i.e. after the shaft is
        # staged and the operator is committed.  Catch it in preflight instead.
        problems.append("gantries are not homed in this daemon session -- every "
                        "/stream/start will 409.  Home in the web app first")
    say(f"  preflight: backend={pf['backend']} torque={pf['servo_torque']} "
        f"homed={pf['homed']} servo_age={pf['servo_age_s']}s")
    for p in problems:
        say(f"  !! {p}")
    pf["problems"] = problems
    return pf


def plan_facts(design):
    """Which finger DRIVES this design's turn, and by how much.

    The grip policy is not one number: the holders need force (their whole
    contribution is torque at a moment arm) and the driver needs to move, and
    its own clamp is what stalls it.  Which finger is which is a property of
    the design, so read it off the plan instead of hard-coding g12's answer.
    """
    plan = json.loads((DEPLOY / f"{design}_plan.json").read_text())
    poses = {p["name"]: p["joints"] for p in plan["poses"]}
    g, e = poses["grip"], poses["turn_end"]
    exc = {f: {j: e[f][j] - g[f][j] for j in ("yaw", "mcp", "pip")} for f in FINGERS}
    tot = {f: sum(abs(v) for v in exc[f].values()) for f in FINGERS}
    peak = max(tot.values())
    # A finger that travels most of what the busiest one does is DRIVING the turn,
    # and its own clamp is what stalls it (2026-08-29 run 4).  A finger that barely
    # moves is HOLDING, and its whole contribution is torque at a moment arm, so it
    # needs force.  One number for all three is what locked every earlier run.
    drivers = [f for f in FINGERS if tot[f] > 0.6 * peak]
    holders = [f for f in FINGERS if f not in drivers]
    driver = max(FINGERS, key=lambda f: tot[f])
    return {"driver": driver, "drivers": drivers, "holders": holders,
            "no_clean_holder": len(holders) < 2,
            "excursion_deg": exc, "total_deg": {f: round(tot[f], 1) for f in FINGERS},
            "clip_saturated": [f"{f}_{j}" for f in FINGERS for j in ("yaw", "mcp", "pip")
                               if abs(abs(exc[f][j]) - 28.648) < 0.05],
            "meta": plan.get("meta", {}), "grip_pose": g}


def ramp_to(target, secs=1.2, speed=50, steps=30):
    """Walk the fingers to `target` instead of commanding it in one frame.

    Repeat 2 of the 2026-08-30 g12 session opened at middle-yaw +18.7 against a
    commanded 0.0 with a yaw load of -930: the run started from wherever repeat 1
    had ENDED, and the first frame tried to close a 19 degree gap in one step.
    Repeats have to begin from the same pose or they are not repeats, and the pose
    has to be reached gently or the first frame is a slam.
    """
    cur = read_fresh()
    a = {f: {j: cur["servos"][FID[f]][KEY[j]] / (SIGN[f] if j == "yaw" else 1.0)
             for j in ("yaw", "mcp", "pip")} for f in FINGERS}
    tok = mh.post("/stream/start", {"timeout_s": 2.0})["token"]
    try:
        for i in range(steps + 1):
            u = i / steps
            mh.post("/stream/frame", {"token": tok, "servo_speed": speed,
                                      "joints": {f: {j: a[f][j] + (target[f][j] - a[f][j]) * u
                                                     for j in ("yaw", "mcp", "pip")}
                                                 for f in FINGERS}})
            time.sleep(secs / steps)
    finally:
        mh.post("/stream/end", {"token": tok})
    time.sleep(0.5)
    t = read_fresh()
    err = {f: round(t["servos"][FID[f]][KEY["yaw"]] / SIGN[f] - target[f]["yaw"], 2)
           for f in FINGERS}
    say("  returned to start pose, yaw err: " + "  ".join(f"{f}={err[f]:+.1f}" for f in FINGERS))
    return err


def run(cmd, sess, tag):
    """Run one bench script, tee stdout into the session, return (rc, text)."""
    say(f"\n  $ {' '.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in p.stdout:
        sys.stdout.write(line)
        lines.append(line)
    p.wait()
    text = "".join(lines)
    (sess / f"{tag}.stdout.txt").write_text(text)
    if p.returncode:
        say(f"  !! exited {p.returncode}")
    return p.returncode, text


def harvest(text, sess, tag):
    """Pull the JSONL the runner wrote into the session directory."""
    m = re.search(r"^\s*log:\s*(\S+)\s*$", text, re.M)
    if not m:
        return None
    src = REPO / m.group(1)
    if not src.exists():
        return None
    dst = sess / f"{tag}{src.suffix}"
    shutil.copy2(src, dst)
    return dst.name


def observe(kind):
    """The measurement no log contains.  Keep the field names stable."""
    say("\n  --- operator observation (this is the only object-pose sensor we have) ---")
    o = {"kind": kind}
    o["held"] = ask_bool("was the shaft still in the hand at the end", True)
    o["rotation_deg"] = ask("estimated cylinder rotation, degrees (protractor/tag if you have it)",
                            None, float)
    o["rotation_source"] = ask("how measured (eye/protractor/apriltag/video)", "eye")
    o["fingers_touched"] = ask_bool("did any two fingers touch each other", False)
    o["slipped"] = ask_bool("did the shaft slide or roll rather than turn with the fingers", False)
    o["media"] = ask("photo/video filename (blank if none)", "", allow_blank=True)
    o["notes"] = ask("anything else worth writing down", "", allow_blank=True)
    return o


# ---------------------------------------------------------------- arms

def arm_freeair(a, sess, facts, rec):
    say("\n== ARM freeair ==  NOTHING in the hand.  Take the shaft out now.")
    if sys.stdin.isatty():
        ask("press enter when the hand is empty", "")
    cmd = ["python3", "scripts/real_v1_bench_stepped_run.py", "--plan", a.design,
           "--steps", str(a.steps), "--gate", "5.0", "--gate-timeout", "0.8",
           "--dwell", str(a.dwell), "--speed", str(a.speed),
           "--max-u", f"{a.max_u:.3f}", "--hold", "2"]
    if a.path == "csv":
        cmd.append("--csv")
    rc, txt = run(cmd, sess, "freeair")
    rec["runs"].append({"arm": "freeair", "rc": rc, "log": harvest(txt, sess, "freeair"),
                        "cmd": cmd})


def arm_grip(a, sess, facts, rec):
    say(f"\n== ARM grip ==  shaft IN the hand, hand open."
        f"  relieve {','.join(facts['drivers'])}, firm {','.join(facts['holders']) or 'nothing'}")
    if sys.stdin.isatty():
        ask("press enter with the shaft staged in the open hand", "")
    rc, txt = run(["python3", "scripts/real_v1_bench_grip.py", a.design], sess, "grip_seat")
    rec["runs"].append({"arm": "grip_seat", "rc": rc})

    tgt = {f: (GRIP_FREE if f in facts["drivers"] else GRIP_FIRM) for f in FINGERS}
    cmd = ["python3", "scripts/real_v1_bench_regrip.py", "--plan", a.design,
           "--preload-start", str(a.preload_start)]
    for f in FINGERS:
        cmd += [f"--target-{f}", str(tgt[f])]
    rc, txt = run(cmd, sess, "regrip")
    pose = REPO / "logs/regrip_pose.json"
    if pose.exists():
        shutil.copy2(pose, sess / "regrip_pose.json")
    rec["runs"].append({"arm": "regrip", "rc": rc, "targets": tgt, "cmd": cmd})
    rec["observations"].append({"kind": "grip",
                                "seated": ask_bool("is the shaft held steady with no visible squash", True),
                                "notes": ask("grip notes", "", allow_blank=True)})


def arm_loaded(a, sess, facts, rec):
    say(f"\n== ARM loaded ==  {a.repeats} repeat(s), max_u={a.max_u:.2f} ({a.path} path)")
    regrip = sess / "regrip_pose.json"
    if not regrip.exists():
        regrip = REPO / "logs/regrip_pose.json"
        if not regrip.exists():
            say("  !! no regrip pose -- run the grip arm first, or the plan grip will over-clamp")
            return
    start_pose = json.loads(Path(regrip).read_text())
    # A speed sweep is a repeat sweep with one thing varied: the servos are
    # position-controlled with a goal-speed, so the same trajectory at a different
    # speed is a different LOAD profile, and load is what trips the overload
    # protection.  Cheapest lever we have that does not touch the plan.
    speeds = [int(x) for x in a.speed_sweep.split(",")] if a.speed_sweep else [a.speed] * a.repeats
    for k, spd in enumerate(speeds):
        say(f"\n  -- repeat {k + 1} of {len(speeds)}"
            + (f", servo_speed {spd} --" if a.speed_sweep else " --"))
        if k:
            # the previous repeat left the fingers at turn_end; go back before the
            # shaft is re-staged, so it is placed into the same hand every time
            try:
                rec["runs"].append({"arm": "return", "repeat": k + 1,
                                    "yaw_err": ramp_to(start_pose)})
            except Exception as exc:
                say(f"  !! could not return to the start pose: {exc}")
            if sys.stdin.isatty():
                ask("re-stage the shaft and press enter", "")
        cmd = ["python3", "scripts/real_v1_bench_stepped_run.py", "--plan", a.design,
               "--steps", str(a.steps), "--gate", "5.0", "--gate-timeout", "0.8",
               "--dwell", str(a.dwell), "--speed", str(spd),
               "--max-u", f"{a.max_u:.3f}", "--regrip", str(regrip),
               "--load-delta", "400", "--stall-deg", "0.5", "--stall-window", "5",
               "--hold", str(a.hold)]
        if a.path == "csv":
            cmd.append("--csv")
        if a.mcp_scale != 1.0:
            cmd += ["--mcp-scale", str(a.mcp_scale)]
        rc, txt = run(cmd, sess, f"loaded_{k + 1}")
        rec["runs"].append({"arm": "loaded", "repeat": k + 1, "servo_speed": spd, "rc": rc,
                            "log": harvest(txt, sess, f"loaded_{k + 1}"), "cmd": cmd})
        o = observe(f"loaded_{k + 1}")
        rec["observations"].append(o)
        write(sess, rec)


PROTECTION_DOC = """\
  The servo unloads ITSELF: sustained load above overload_torque (80%) for
  protection_time drops the output to protective_torque (20% == the load 200
  plateau seen on 2026-08-29) and holds it there.  The registers are writable,
  so the reorient ceiling may be configuration rather than hardware.

  This needs the raw servo bus, so it runs ON the CB1 with the control station
  STOPPED.  From the workstation:

    ssh irlab@10.99.99.2                       # password in .dotenv (MANTA_IRLAB_PW)
    ~/run_control_station.sh stop
    python3 - <<'PY'
    from rustypot import Scs0009PyController as C
    c = C(serial_port="/dev/ttyUSB0", baudrate=1000000, timeout=0.05)
    MID_YAW = 6
    for r in ("max_torque_limit","overload_torque","protective_torque",
              "protection_time","unloading_condition","minimum_startup_force"):
        print(r, getattr(c, "read_" + r)(MID_YAW))
    c.write_protective_torque(MID_YAW, 40)     # 20% -> 40%
    print("readback", c.read_protective_torque(MID_YAW))
    PY
    ~/run_control_station.sh start

  Then re-run the loaded arm.  The result to read is the PLATEAU VALUE in the
  middle-yaw load trace: 200 before, 400 after if the mechanism is what we think.
  Put it back to 20 afterwards -- that protection exists for a thermal reason.
"""


def arm_protection(a, sess, facts, rec):
    say("\n== ARM protection ==")
    say(PROTECTION_DOC)
    (sess / "protection_procedure.txt").write_text(PROTECTION_DOC)
    rec["observations"].append({
        "kind": "protection",
        "ran": ask_bool("did you run the register write", False),
        "protective_torque_set": ask("protective_torque written (blank if not run)", None, float),
        "plateau_before": ask("load plateau BEFORE (200 expected)", None, float),
        "plateau_after": ask("load plateau AFTER", None, float),
        "driver_yaw_reached_deg": ask("driver-finger yaw reached, degrees", None, float),
        "notes": ask("notes", "", allow_blank=True)})


ARMS = {"freeair": arm_freeair, "grip": arm_grip,
        "loaded": arm_loaded, "protection": arm_protection}


def write(sess, rec):
    rec["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (sess / "MANIFEST.json").write_text(json.dumps(rec, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="g12", choices=sorted(SAFE_U))
    ap.add_argument("--arms", default="freeair,grip,loaded",
                    help="comma-separated: " + ",".join(ARMS))
    ap.add_argument("--repeats", type=int, default=3,
                    help="loaded repeats.  n=1 cannot separate a design from a staging accident")
    ap.add_argument("--path", default="chord", choices=("chord", "csv"),
                    help="chord = the plan's 3 set-points interpolated (what the service "
                         "drives); csv = the exported dense trajectory.  They are DIFFERENT "
                         "paths and clear by different margins")
    ap.add_argument("--max-u", type=float, default=None,
                    help="override the clearance-derived truncation.  Do not raise it without "
                         "re-running real_v1_trajectory_clearance.py")
    ap.add_argument("--steps", type=int, default=55)
    ap.add_argument("--dwell", type=float, default=0.1)
    ap.add_argument("--speed", type=int, default=80)
    ap.add_argument("--speed-sweep", default="",
                    help="comma-separated servo speeds, one loaded repeat each, e.g. 40,80,120. "
                         "Overrides --repeats.  Same trajectory, different load profile")
    ap.add_argument("--hold", type=float, default=4.0)
    ap.add_argument("--mcp-scale", type=float, default=1.0)
    ap.add_argument("--preload-start", type=float, default=9.0)
    ap.add_argument("--operator", default=os.environ.get("USER", "?"))
    ap.add_argument("--note", default="", help="what this session is testing")
    a = ap.parse_args()

    if a.max_u is None:
        a.max_u = SAFE_U[a.design][a.path]

    facts = plan_facts(a.design)
    stamp = time.strftime("%Y%m%d-%H%M")
    sess = SUITE / f"{stamp}-{a.design}"
    sess.mkdir(parents=True, exist_ok=True)

    say(f"\n=== real_v1 bench session: {a.design} ===")
    say(f"  session dir : {sess.relative_to(REPO)}")
    say(f"  drivers (relieve): {','.join(facts['drivers'])}"
        f"    holders (keep firm): {','.join(facts['holders']) or 'NONE'}")
    say("  excursion deg: " + "  ".join(
        f"{f} yaw{facts['excursion_deg'][f]['yaw']:+6.1f} mcp{facts['excursion_deg'][f]['mcp']:+6.1f}"
        f" pip{facts['excursion_deg'][f]['pip']:+6.1f}" for f in FINGERS))
    if facts["clip_saturated"]:
        say("  clipped at the +-0.5 rad residual budget: "
            + ", ".join(facts["clip_saturated"]))
    if facts["no_clean_holder"]:
        say("  !! ONLY ONE HOLDER: two of three fingers travel, so a single finger has to\n"
            "     anchor the shaft while the other two turn it.  Expect grip and turn to\n"
            "     fight; if the loaded arm drops the shaft twice, that IS the finding, not\n"
            "     bad staging.  (Both designs in this class also interpenetrate.)")
    say(f"  truncation  : max_u {a.max_u:.2f} on the {a.path} path"
        + ("  (FULL path clears)" if a.max_u >= 1.0 else "  (finger-finger clearance)"))

    rec = {"design": a.design, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "operator": a.operator, "note": a.note, "args": vars(a),
           "plan_facts": facts, "safe_u_table": SAFE_U[a.design],
           "runs": [], "observations": []}
    rec["preflight"] = preflight(a.design)
    write(sess, rec)
    if rec["preflight"]["problems"]:
        if not ask_bool("preflight found problems -- continue anyway", False):
            say("  stopped."); return 1

    for name in [x.strip() for x in a.arms.split(",") if x.strip()]:
        if name not in ARMS:
            say(f"  !! unknown arm {name!r}"); continue
        try:
            ARMS[name](a, sess, facts, rec)
        except KeyboardInterrupt:
            say("\n  interrupted -- manifest saved"); break
        except Exception as exc:
            say(f"  !! arm {name} failed: {exc}")
            rec["runs"].append({"arm": name, "error": str(exc)})
        write(sess, rec)

    rec["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write(sess, rec)
    say(f"\n  session written: {sess.relative_to(REPO)}")
    say("  hand back the whole directory -- it is self-describing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
