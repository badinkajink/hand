"""The set-up of a HOM trial on the SR2 hand on the UR5, run on corvus before hom_trial.py's trial (owner, 2026-11-16:
"be ready, to the extent that i can place the assembled object down, ..., the system localizes, moves (marginally
translates to the object without the fingers colliding), and tries to grasp and in-hand manipulate").

Every command prints what it would do; only with --go does anything move.

    status     the station (homed, torque, plan, mounts), the arm (mode, the palm's height) and the plan's nominal pose
    torque-on  the motionless restart of the runbook's "Before a session": on the CB1 the station stopped, every servo's
               goal set to its present position at goal speed 40 (cb1/20261115-servo_hold.py), the station started
               (which turns the servos on) and its home adopted; the fingers stay where they are
    open       the fingers to the plan's open pose, the gantries where they are (manual/joints at --speed); run at the
               home mounts, where the fingers are far apart
    layout     the plan into the station (inline), the gantries to its mounts (morphology), and the open pose
               commanded again: plans/load resets the station's ramp start to the zero pose, and this re-seeds it
    place      the arm straight up or down (base z) to the placement height: the hand --clear mm above the grasp's
               height for the object resting on its block on the table (--table-z, base frame), so the object can be
               put under the open fingers
    approach   the object on the table under the hand: its pose in P from the tracker (median over --measure-s), and
               the move that puts it at the plan's nominal pose (position, and on the cylinder its axis' yaw modulo
               180 deg; the sphere's yaw is free). The lateral part runs at the current height, the tracker measures
               again and a fine lateral correction follows, then the descent along -z_P; every arm move is limited by
               --max-lat, --max-yaw and --max-dz and stopped by the HEX-E's force change past --stop-n; at the end the
               residual is measured. --fine: only the lateral correction, at the current height (at most 3 mm)
    release    after a trial and hom_trial.py return: the open pose (the fingers let go of the object on the table),
               then the arm up by --clear for the next placement

    /usr/bin/python3 hom_setup.py status --plan ../plans/20261115-generalist_press_plan.json
    /usr/bin/python3 hom_setup.py approach --plan PLAN --object cylinder --map ../maps/20261116-cylinder_fitted_map_O.json --go

The nominal pose is the simulation's start (hom_server nominal; hom_tasks.Plant.reset at seed 0): the object at
(0, 0, Z_OBJ) in the model's world and the palm at meta.hom.setup.palm_m, so p_PO = (0, 0, Z_OBJ) - palm_m, turned by
meta.hom.setup.quat. The object's height on the table comes from its tag map: the block's tag centres lie at the
block's mid-height, half the block above the table. On 2026-11-16 the objects carry each other's tag blocks (the
cylinder ids 16-21 on the sphere's shorter stem, the sphere ids 10-15 on the cylinder's longer one;
object_tracking_mjcf.assembled_map), which the maps of maps/ include.
"""
import argparse
import json
import math
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/sr2hand/driver/reskin5x/host"), str(HERE)]
import hom_trial as HT  # noqa: E402  open_ft, start_tracker, ur_pose, ROBOT
from sr2hand.geometry.rotations import (quat_conj, quat_from_rotvec, quat_mul, quat_to_mat, quat_to_rotvec,  # noqa: E402
                                        rot_between, rot_z, tf, tf_from_ur, tf_inv, tf_to_ur)
from sr2hand.robot.tool_chain import SR2Hand  # noqa: E402

Z_OBJ = 0.120                  # m: the object's centre above the model's table at the start (hand_object_scale_sim.Z_OBJ)
BLOCK_H = 17.0                 # mm: the tag block's height (object_tracking_mjcf.BLOCK)
END_IDS = (22, 23)             # the cylinder's end tags, not on the block
CB1 = "irlab@10.99.99.2"
CB1_HOST_DIR = "~/hand/src/morphohand/driver/manta/host"
LOG = ROOT / "logs/20261116-hom_setup"
T_EP = SR2Hand.load().T_E_P()


def note(rec):
    """Append a record to today's set-up log."""
    LOG.mkdir(parents=True, exist_ok=True)
    with open(LOG / f"{time.strftime('%Y%m%d')}_hom_setup.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}, default=float) + "\n")


# ------------------------------------------------------------------------------------------------ plan and object

def load_plan(path):
    return json.loads(Path(path).read_text())


def nominal(plan):
    """(p_PO mm, q_PO wxyz) of the plan's grasp at the simulation's start."""
    s = plan["meta"]["hom"]["setup"]
    return 1e3 * (np.array([0.0, 0.0, Z_OBJ]) - np.asarray(s["palm_m"], float)), np.asarray(s["quat"], float)


def open_pose(plan):
    for p in plan["poses"]:
        if p["name"] == "open":
            return {f: {j: float(v) for j, v in js.items()} for f, js in p["joints"].items()}
    raise SystemExit("the plan has no open pose")


def map_height(map_file):
    """The object's centre above the table (mm) when it rests on its block, from its tag map in O: the block's tag
    centres are at the block's mid-height, y_O is up."""
    m = json.loads(Path(map_file).read_text())
    cen = [np.mean(np.asarray(c, float), axis=0) for i, c in m["corners_mm"].items() if int(i) not in END_IDS]
    return -float(np.mean([c[1] for c in cen])) + BLOCK_H / 2


# ------------------------------------------------------------------------------------------------ station

class Station:
    def __init__(self, url, token_file):
        self.url, self.token = url.rstrip("/"), Path(token_file).read_text().strip()

    def call(self, method, path, body=None, timeout=15.0):
        req = urllib.request.Request(self.url + path, method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "X-Manta-Token": self.token})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise SystemExit(f"station {method} {path}: {e.code} {e.read()[:400]!r}")

    def state(self):
        return self.call("GET", "/api/v1/state")

    def wait_idle(self, what, timeout=120.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self.state()
            if not s.get("busy") and s.get("operation") in (None, "", "idle"):
                return s
            time.sleep(0.5)
        raise SystemExit(f"station still busy after {timeout:.0f} s ({what})")


def _driver_plan():
    host = str(ROOT / "src/sr2hand/driver/manta/host")
    if host not in sys.path:
        sys.path.insert(0, host)
    from manta_hand import plan as P
    return P


def sim_joints(state):
    """The station's measured joints, simulator degrees {finger: {yaw, mcp, pip}}: the telemetry's servo degrees
    {id: {aa, fe1, fe2}} over the driver's JOINT_SIGN (manta_hand.plan; the index's and the middle's yaw are -1)."""
    P = _driver_plan()
    tel = state.get("telemetry", {})
    sv = tel.get("servos") or {}
    out = {}
    for k, f in enumerate(("thumb", "index", "middle")):
        s = sv.get(str(k))
        if s is None:
            return None
        out[f] = {j: s[P.SIM_JOINT_TO_SERVO[j]] / P.JOINT_SIGN[(f, j)] for j in ("yaw", "mcp", "pip")}
    return out


def show_pose(name, pose):
    return f"{name}: " + "  ".join(f"{f} " + " ".join(f"{j} {pose[f][j]:+6.1f}" for j in ("yaw", "mcp", "pip"))
                                   for f in ("thumb", "index", "middle"))


# ------------------------------------------------------------------------------------------------ arm

class Arm:
    """rtde_receive always; rtde_control and the force guard only for a move (--go). The guard is the HEX-E, or with
    ur_ft the UR's own TCP force (getActualTCPForce; 0.25-0.48 N sd at rest on 2026-11-16, when the HEX-E did not
    answer)."""

    def __init__(self, ur_ft=False, blank_s=0.4):
        import rtde_receive
        self.r = rtde_receive.RTDEReceiveInterface(HT.ROBOT, 125.0)
        self.c = self.ft = None
        self.ur_ft, self.blank_s = ur_ft, blank_s

    def force(self):
        """(time s, force N in 3) of the guard's newest sample, or (None, None)."""
        if self.ur_ft:
            return time.time(), np.asarray(self.r.getActualTCPForce(), float)[:3]
        t, v = self.ft.ring.latest()
        return (None, None) if t is None else (t, np.asarray(v, float)[:3])

    def force_mean(self, seconds=0.4):
        if self.ur_ft:
            F = []
            t0 = time.time()
            while time.time() - t0 < seconds:
                F.append(np.asarray(self.r.getActualTCPForce(), float)[:3])
                time.sleep(0.008)
            return np.mean(F, axis=0)
        _, v0 = self.ft.ring.since(time.time() - seconds)
        if len(v0) < 10:
            raise SystemExit(f"the HEX-E gave {len(v0)} samples in {seconds:.1f} s: is it on? (--ur-ft guards with the "
                             "UR's own force instead)")
        return np.asarray(v0, float)[:, :3].mean(axis=0)

    def T_BE(self):
        return tf_from_ur(self.r.getActualTCPPose())

    def T_BP(self):
        return self.T_BE() @ T_EP

    def ready(self):
        mode, prot = self.r.getRobotMode(), self.r.isProtectiveStopped()
        if mode != 7 or prot:
            raise SystemExit(f"the arm is not running (robot mode {mode}, protective stop {prot}): release the brakes on "
                             "the pendant (ON, START) and clear any stop")

    def connect(self, guard=True):
        import rtde_control
        self.ready()
        self.ft = None if (self.ur_ft or not guard) else HT.open_ft(True)
        self.c = rtde_control.RTDEControlInterface(HT.ROBOT)
        self.c.setTcp([0.0] * 6)
        time.sleep(0.6)

    def move(self, T_BE_goal, v, acc, stop_n, max_s, label, check=None):
        """moveL of the flange to T_BE_goal (mm), stopped by the guard's force change past stop_n from its mean over the
        0.4 s before (stop_n None: no force guard, the owner watching), a stale guard, check() returning a reason, a
        protective stop or max_s. Returns (stopped reason or None, moved mm)."""
        guard = stop_n is not None
        F0 = self.force_mean(0.4) if guard else None
        src = ("UR TCP force" if self.ur_ft else "HEX-E") if guard else "no force guard"
        p0 = self.T_BE()[:3, 3]
        self.c.moveL(tf_to_ur(T_BE_goal).tolist(), v, acc, True)
        t0, fmax, why = time.time(), 0.0, None
        time.sleep(0.02)
        while True:
            if guard:
                t, F = self.force()
                if time.time() - t0 < self.blank_s:
                    pass                                   # the start's transient (4.4 N at 0.1 m/s^2 on 2026-11-16)
                elif t is None or time.time() - t > 0.1:
                    why = f"no {src} sample for 100 ms"
                else:
                    fmax = max(fmax, float(np.linalg.norm(F - F0)))
                    if fmax > stop_n:
                        why = f"{src} changed {fmax:.2f} N (limit {stop_n:g})"
            if why is None and check is not None:
                why = check()
            if why is None and (self.r.isProtectiveStopped() or self.r.isEmergencyStopped()):
                why = "protective or emergency stop"
            if why is None and time.time() - t0 > max_s:
                why = f"move over {max_s:.0f} s"
            if why:
                self.c.stopL(1.0)
                break
            if self.c.getAsyncOperationProgress() < 0 and time.time() - t0 > 0.2:
                break
            time.sleep(0.01)
        moved = float(np.linalg.norm(self.T_BE()[:3, 3] - p0))
        print(f"   {label}: {'STOPPED, ' + why if why else 'done'}; moved {moved:.2f} mm, {src} change up to {fmax:.2f} N",
              flush=True)
        note({"cmd": "move", "label": label, "goal": tf_to_ur(T_BE_goal).tolist(), "stopped": why, "moved_mm": moved,
              "max_dF_N": fmax, "guard": src})
        return why, moved

    def close(self):
        try:
            if self.c is not None:
                self.c.stopScript()
                self.c.disconnect()
        finally:
            self.r.disconnect()


# ------------------------------------------------------------------------------------------------ tracker

class Measure:
    """The tracker of hom_trial.py on --map: object poses in P with the arm's pose at each frame's capture."""

    def __init__(self, a):
        self.samples, self.lock = [], threading.Lock()
        self.arm_pose, self.tracker = HT.start_tracker(a, self._on)
        time.sleep(1.5)                                    # the cameras' first frames

    def _on(self, s):
        m = s.msg
        if m is not None and m.n_tags > 0 and np.all(np.isfinite(m.p_PO)):
            with self.lock:
                self.samples.append((time.time(), np.array(m.p_PO, float), np.array(m.q_PO, float), int(m.n_tags)))

    def pose(self, seconds, settle=0.5):
        """(p mm, q wxyz, info) from the samples of the next `seconds` after `settle` s: the median position and the
        mean rotation about the first sample's."""
        time.sleep(settle)
        t0 = time.time()
        time.sleep(seconds)
        with self.lock:
            got = [s for s in self.samples if s[0] >= t0]
        if len(got) < 3:
            raise SystemExit(f"the tracker gave {len(got)} poses of the object in {seconds:.1f} s: is it in view of both "
                             "cameras, and is --map its map?")
        P = np.array([g[1] for g in got])
        q0 = got[0][2]
        w = np.mean([quat_to_rotvec(quat_mul(quat_conj(q0), g[2])) for g in got], axis=0)
        q = quat_mul(q0, quat_from_rotvec(w))
        return np.median(P, axis=0), q, {"n": len(got), "sd_mm": P.std(axis=0).round(3).tolist(),
                                         "tags": int(np.median([g[3] for g in got]))}

    def watch(self, p_ref, q_ref, lat_mm=1.5, turn_deg=3.0):
        """check() for Arm.move during a move along z_P: a reason when the newest tracked pose (at most 0.3 s old) lies
        more than lat_mm off p_ref in P's xy or turned turn_deg from q_ref (a finger pushing the object), or when the
        tracker has gone quiet."""
        def check():
            with self.lock:
                s = self.samples[-1] if self.samples else None
            if s is None or time.time() - s[0] > 0.3:
                return "the tracker lost the object for 0.3 s"
            lat = float(np.hypot(*(s[1][:2] - p_ref[:2])))
            turn = math.degrees(float(np.linalg.norm(quat_to_rotvec(quat_mul(quat_conj(q_ref), s[2])))))
            if lat > lat_mm or turn > turn_deg:
                return f"the object moved {lat:.2f} mm in P's xy and turned {turn:.2f} deg (limits {lat_mm:g}, {turn_deg:g})"
            return None
        return check

    def stop(self):
        self.tracker.stop()
        self.arm_pose.stop()


def yaw_error(q_m, q_n, shape):
    """Signed turn (rad) about z_P from the nominal pose to the measured one: of the cylinder's axis (z_O) projected on
    the palm plane, modulo 180 deg (the cylinder and its centred block are symmetric); 0 for the sphere."""
    if shape != "cylinder":
        return 0.0
    a, b = quat_to_mat(q_n)[:2, 2], quat_to_mat(q_m)[:2, 2]
    psi = math.atan2(a[0] * b[1] - a[1] * b[0], a @ b)
    return (psi + math.pi / 2) % math.pi - math.pi / 2


def tilt_deg(q):
    """Angle (deg) of the object's up axis y_O from z_P."""
    return math.degrees(math.acos(float(np.clip(quat_to_mat(q)[2, 1], -1.0, 1.0))))


def correction(p_m, p_n, psi):
    """(C_lat, dz): in P, the hand's lateral move C_lat (a turn psi about the vertical through the object and the shift
    that brings the measured point over the nominal one) and its move dz along z_P. T_BP C_lat Trans(0, 0, dz) puts the
    measured object at the nominal position and turns it by -psi; the two parts commute."""
    C = tf(None, [p_m[0], p_m[1], 0.0]) @ tf(rot_z(psi), None) @ tf(None, [-p_n[0], -p_n[1], 0.0])
    return C, float(p_m[2] - p_n[2])


def report(tag, p_m, q_m, p_n, q_n, shape, info):
    d = p_m - p_n
    psi = yaw_error(q_m, q_n, shape)
    print(f"{tag}: object in P {np.round(p_m, 2).tolist()} mm ({info['n']} poses, sd {info['sd_mm']} mm, "
          f"{info['tags']} tags); from nominal dx {d[0]:+.2f} dy {d[1]:+.2f} dz {d[2]:+.2f} mm, lateral "
          f"{np.hypot(d[0], d[1]):.2f} mm, yaw {math.degrees(psi):+.2f} deg, tilt {tilt_deg(q_m):.2f} deg "
          f"(nominal {tilt_deg(q_n):.2f})", flush=True)
    note({"cmd": "measure", "tag": tag, "p_PO": p_m.tolist(), "q_PO": q_m.tolist(), "d_mm": d.tolist(),
          "yaw_deg": math.degrees(psi), "tilt_deg": tilt_deg(q_m), **info})
    return d, psi


# ------------------------------------------------------------------------------------------------ commands

def cmd_status(a):
    st = Station(a.station, a.token_file)
    s = st.state()
    pl = s.get("plan") or {}
    print(f"station: homed {s.get('homed')}, torque {s.get('servo_torque')} (1 on, 2 off), operation "
          f"{s.get('operation')}, plan {pl.get('design')}, mounts {s.get('mount_positions')}, applied "
          f"{s.get('mounts_applied')}")
    j = sim_joints(s)
    if j:
        print("   " + show_pose("measured", j))
    arm = Arm()
    try:
        T = arm.T_BP()
        print(f"arm: robot mode {arm.r.getRobotMode()} (7 running), protective stop {arm.r.isProtectiveStopped()}; "
              f"palm P at {np.round(T[:3, 3], 1).tolist()} mm in the base, z_P tilted "
              f"{math.degrees(math.acos(min(1.0, T[2, 2]))):.2f} deg from base z")
    finally:
        arm.close()
    if a.plan:
        plan = load_plan(a.plan)
        p, q = nominal(plan)
        print(f"plan {plan['design']}: nominal p_PO {np.round(p, 2).tolist()} mm, q_PO {np.round(q, 4).tolist()}, "
              f"mounts {plan['mounts_palm_mm']}")
        print("   " + show_pose("open", open_pose(plan)))
        if j:
            dj = max(abs(open_pose(plan)[f][k] - j[f][k]) for f in j for k in j[f])
            print(f"   largest joint change from the measured pose to the open pose: {dj:.1f} deg")
    return 0


def ssh(cmd, timeout=60):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", CB1, cmd], capture_output=True,
                       text=True, timeout=timeout)
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "token" not in l.lower())
    return r.returncode, out


def cmd_torque_on(a):
    st = Station(a.station, a.token_file)
    s = st.state()
    j0 = sim_joints(s)
    print(f"station: homed {s.get('homed')}, torque {s.get('servo_torque')}, mounts {s.get('mount_positions')}")
    if j0:
        print("   " + show_pose("measured", j0))
    steps = ["~/run_control_station.sh stop", "copy cb1/20261115-servo_hold.py and run it (goal = present, speed 40)",
             "~/run_control_station.sh start (the servos turn on and hold)", "POST /api/v1/home/adopt"]
    print("would run on the CB1: " + "; ".join(steps))
    if not a.go:
        return 0
    rc, out = ssh("~/run_control_station.sh stop")
    print(f"stop: rc {rc}\n{out[-600:]}")
    time.sleep(2.0)
    hold = HERE.parent / "cb1/20261115-servo_hold.py"
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(hold), f"{CB1}:/tmp/20261115-servo_hold.py"], check=True)
    rc, out = ssh(f"cd {CB1_HOST_DIR} && .venv/bin/python /tmp/20261115-servo_hold.py", timeout=120)
    print(f"servo hold: rc {rc}\n{out}")
    if rc != 0:
        raise SystemExit("the servo hold failed: the station stays stopped; look before starting it")
    rc, out = ssh("~/run_control_station.sh start", timeout=120)
    print(f"start: rc {rc}\n{out[-400:]}")
    t0 = time.time()
    while time.time() - t0 < 60:
        try:
            s = st.state()
            break
        except (OSError, SystemExit):
            time.sleep(1.0)
    else:
        raise SystemExit("the station did not answer within 60 s of its start")
    time.sleep(2.0)
    res = st.call("POST", "/api/v1/home/adopt")
    s = st.state()
    j1 = sim_joints(s)
    print(f"adopted: homed {s.get('homed')}, torque {s.get('servo_torque')}, mounts {s.get('mount_positions')} "
          f"({json.dumps(res)[:200]})")
    if j0 and j1:
        print("   " + show_pose("measured", j1))
        print(f"   largest joint change over the restart: "
              f"{max(abs(j1[f][k] - j0[f][k]) for f in j0 for k in j0[f]):.2f} deg")
    note({"cmd": "torque-on", "before": j0, "after": j1, "torque": s.get("servo_torque"), "homed": s.get("homed")})
    return 0


def cmd_open(a):
    st = Station(a.station, a.token_file)
    plan = load_plan(a.plan)
    s = st.state()
    j = sim_joints(s)
    tgt = open_pose(plan)
    print(f"station: torque {s.get('servo_torque')}, mounts {s.get('mount_positions')}")
    if j:
        print("   " + show_pose("measured", j))
        print(f"   largest joint change: {max(abs(tgt[f][k] - j[f][k]) for f in j for k in j[f]):.1f} deg")
    print("   " + show_pose("open (plan)", tgt))
    if not a.go:
        return 0
    if s.get("servo_torque") != 1:
        raise SystemExit("servo torque is not on: run torque-on first")
    if a.ramp_s > 0 and j:
        # small steps from the measured pose (owner, 2026-11-16: "if we slowly release the servos the block should
        # come to a flat rest")
        n = max(1, int(round(10 * a.ramp_s)))
        for i in range(1, n + 1):
            u = i / n
            step = {f: {k: j[f][k] + (tgt[f][k] - j[f][k]) * u for k in tgt[f]} for f in tgt}
            res = st.call("POST", "/api/v1/manual/joints", {"joints": step, "servo_speed": a.speed})
            time.sleep(0.1)
    else:
        res = st.call("POST", "/api/v1/manual/joints", {"joints": tgt, "servo_speed": a.speed})
    time.sleep(a.wait_s)
    j2 = sim_joints(st.state())
    if j2:
        print("   " + show_pose("measured", j2))
        print(f"   largest error to the open pose: {max(abs(tgt[f][k] - j2[f][k]) for f in j2 for k in j2[f]):.1f} deg")
    note({"cmd": "open", "plan": a.plan, "target": tgt, "after": j2, "station": res, "ramp_s": a.ramp_s})
    return 0


def cmd_zero(a):
    """The fingers to the zero pose (straight down), the gantries where they are (owner, 2026-11-16: "if the fingers
    are at 0 (hanging straight down), at the specified morphology, then actuate to a pre-grasp keyframe")."""
    st = Station(a.station, a.token_file)
    s = st.state()
    j = sim_joints(s)
    tgt = {f: {k: 0.0 for k in ("yaw", "mcp", "pip")} for f in ("thumb", "index", "middle")}
    if j:
        print("   " + show_pose("measured", j))
        print(f"   largest joint change to zero: {max(abs(j[f][k]) for f in j for k in j[f]):.1f} deg")
    if not a.go:
        return 0
    if s.get("servo_torque") != 1:
        raise SystemExit("servo torque is not on: run torque-on first")
    res = st.call("POST", "/api/v1/manual/joints", {"joints": tgt, "servo_speed": a.speed})
    time.sleep(a.wait_s)
    j2 = sim_joints(st.state())
    if j2:
        print("   " + show_pose("measured", j2))
    note({"cmd": "zero", "after": j2, "station": res})
    return 0


def cmd_layout(a):
    st = Station(a.station, a.token_file)
    plan = load_plan(a.plan)
    s = st.state()
    print(f"station: mounts {s.get('mount_positions')} -> plan {plan['design']} mounts {plan['mounts_palm_mm']}")
    j = sim_joints(s)
    zero = {f: {k: 0.0 for k in ("yaw", "mcp", "pip")} for f in ("thumb", "index", "middle")}
    tgt = None
    if j:
        for name, pose in (("zero", zero), ("open", open_pose(plan))):
            dj = max(abs(pose[f][k] - j[f][k]) for f in j for k in j[f])
            print(f"   fingers {dj:.1f} deg at most from the {name} pose")
            if dj < 5.0 and tgt is None:
                tgt = pose
        if tgt is None and a.go:
            raise SystemExit("the fingers are at neither the zero nor the open pose: run zero (or open) first")
    if not a.go:
        return 0
    raw = dict(plan)
    raw.setdefault("meta", {})["plan_file"] = Path(a.plan).name
    st.call("POST", "/api/v1/plans/load", {"plan": {k: raw[k] for k in ("design", "mounts_palm_mm", "poses", "meta")}},
            timeout=30.0)
    rec = None
    if a.video:                                           # the reconfiguration filmed (owner, 2026-11-16)
        out = ROOT / "logs/20261116-hom_trials"
        out.mkdir(parents=True, exist_ok=True)
        rec = HT.VideoRecorder(str(out / f"{time.strftime('%Y%m%d-%H%M%S')}_reconfig_{Path(a.plan).stem}"))
        class _M:
            map = a.video_map
        arm_pose, tracker = HT.start_tracker(_M, lambda s_: None, on_frame=rec.on_frame)
        time.sleep(2.0)
    st.call("POST", "/api/v1/morphology")
    try:
        s = st.wait_idle("morphology", timeout=180.0)
    finally:
        if rec is not None:
            time.sleep(1.0)
            tracker.stop()
            arm_pose.stop()
            print(f"   video: {rec.close()}", flush=True)
    print(f"morphology: mounts {s.get('mount_positions')}, applied {s.get('mounts_applied')}, error "
          f"{s.get('last_error')}")
    st.call("POST", "/api/v1/manual/joints", {"joints": tgt, "servo_speed": a.speed})   # the ramp start, no motion
    time.sleep(1.0)
    note({"cmd": "layout", "plan": a.plan, "mounts": s.get("mount_positions"), "applied": s.get("mounts_applied")})
    return 0


def cmd_place(a):
    plan = load_plan(a.plan)
    p_n, _ = nominal(plan)
    h = map_height(a.map)
    arm = Arm(a.ur_ft, a.blank_s)
    try:
        T_BP = arm.T_BP()
        z_goal = a.table_z + h - p_n[2] + a.clear
        dz = z_goal - T_BP[2, 3]
        tilt = math.degrees(math.acos(min(1.0, float(T_BP[2, 2]))))
        print(f"palm at z {T_BP[2, 3]:.1f} mm in the base; the object's centre rests {h:.1f} mm above the table "
              f"(z {a.table_z:.1f}), nominal {-p_n[2]:.1f} mm below P, clearance {a.clear:.0f} mm -> palm to z "
              f"{z_goal:.1f}: {dz:+.1f} mm" + (f"; z_P levelled from {tilt:.2f} deg (the least turn about the palm's "
                                               "origin, as the simulation's gravity along -z_P)" if a.level else ""))
        if abs(dz) > a.max_dz:
            raise SystemExit(f"{dz:+.1f} mm is more than --max-dz {a.max_dz:g}")
        if a.level and tilt > a.max_level:
            raise SystemExit(f"z_P is {tilt:.2f} deg from vertical, more than --max-level {a.max_level:g}")
        if not a.go:
            return 0
        arm.connect(a.stop_n is not None)
        T_BP = arm.T_BP()
        R = rot_between(T_BP[:3, 2], [0.0, 0.0, 1.0]) @ T_BP[:3, :3] if a.level else T_BP[:3, :3]
        T_goal = tf(R, T_BP[:3, 3] + [0.0, 0.0, dz]) @ tf_inv(T_EP)
        arm.move(T_goal, a.v, a.acc, a.stop_n, abs(dz) / (1e3 * a.v) + 6.0,
                 f"place {dz:+.1f} mm in z" + (f", levelled {tilt:.2f} deg" if a.level else ""))
        T_BP = arm.T_BP()
        print(f"   palm at {np.round(T_BP[:3, 3], 1).tolist()} mm, z_P "
              f"{math.degrees(math.acos(min(1.0, float(T_BP[2, 2])))):.2f} deg from vertical", flush=True)
    finally:
        arm.close()
    return 0


def cmd_approach(a):
    plan = load_plan(a.plan)
    shape = plan["meta"]["hom"]["shape"]
    if shape != a.object:
        raise SystemExit(f"the plan's object is the {shape}, not the {a.object}")
    p_n, q_n = nominal(plan)
    print(f"plan {plan['design']}: nominal p_PO {np.round(p_n, 2).tolist()} mm, yaw of the axis "
          f"{math.degrees(math.atan2(quat_to_mat(q_n)[1, 2], quat_to_mat(q_n)[0, 2])):.1f} deg in P"
          if shape == "cylinder" else f"plan {plan['design']}: nominal p_PO {np.round(p_n, 2).tolist()} mm")
    arm = Arm(a.ur_ft, a.blank_s)
    meas = Measure(a)
    try:
        p_m, q_m, info = meas.pose(a.measure_s, settle=0.0)
        d, psi = report("seen", p_m, q_m, p_n, q_n, shape, info)
        lat = float(np.hypot(d[0], d[1]))
        dz = float(d[2])
        lim = a.fine_max if a.fine else a.max_lat
        if lat > lim or abs(math.degrees(psi)) > a.max_yaw or (not a.fine and abs(dz) > a.max_dz):
            raise SystemExit(f"too far to move: lateral {lat:.1f} mm (limit {lim:g}), yaw {math.degrees(psi):+.1f} deg "
                             f"(limit {a.max_yaw:g}), dz {dz:+.1f} mm (limit {a.max_dz:g}): place the object closer "
                             "to the nominal pose under the hand")
        rise = (not a.fine) and dz > a.up_ok
        if rise and not a.fingers_straight:
            raise SystemExit(f"the object is {dz:.1f} mm higher than the grasp's height: the hand would have to rise "
                             "with the fingers round it; straighten them (zero) and pass --fingers-straight, or use "
                             "ready")
        if not a.go:
            print("would move: " + (f"up {dz:.2f} mm along z_P (fingers straight), then " if rise else "")
                  + f"lateral {lat:.2f} mm and yaw {math.degrees(psi):+.2f} deg"
                  + ("" if a.fine or rise else f", then {dz:+.2f} mm along z_P"))
            return 0
        arm.connect(a.stop_n is not None)
        if rise:                                          # the hand below the grasp's height: up first, fingers straight
            arm.move(arm.T_BP() @ tf(None, [0.0, 0.0, dz]) @ tf_inv(T_EP), a.v_down, a.acc, a.stop_n,
                     dz / (1e3 * a.v_down) + 5.0, f"rise {dz:+.2f} mm along z_P (fingers straight)")
            p_m, q_m, info = meas.pose(a.measure_s)
            d, psi = report("after the rise", p_m, q_m, p_n, q_n, shape, info)
            lat = float(np.hypot(d[0], d[1]))
        C, _ = correction(p_m, p_n, psi)
        T_goal = arm.T_BP() @ C @ tf_inv(T_EP)
        arm.move(T_goal, a.v_lat, a.acc, a.stop_n, lat / (1e3 * a.v_lat) + 6.0,
                 f"lateral {lat:.2f} mm, yaw {math.degrees(psi):+.2f} deg")
        p_m, q_m, info = meas.pose(a.measure_s)
        d, psi = report("after the lateral move", p_m, q_m, p_n, q_n, shape, info)
        lat = float(np.hypot(d[0], d[1]))
        if lat > 0.4 or abs(math.degrees(psi)) > 0.5:
            if lat > a.fine_max:
                raise SystemExit(f"still {lat:.2f} mm off after the lateral move: stopping here")
            C, _ = correction(p_m, p_n, psi)
            arm.move(arm.T_BP() @ C @ tf_inv(T_EP), a.v_lat, a.acc, a.stop_n, 8.0,
                     f"fine lateral {lat:.2f} mm, yaw {math.degrees(psi):+.2f} deg")
            p_m, q_m, info = meas.pose(a.measure_s)
            d, psi = report("after the fine move", p_m, q_m, p_n, q_n, shape, info)
        if a.fine:
            return 0
        dz = float(d[2])
        if abs(dz) > 0.3:
            why, _ = arm.move(arm.T_BP() @ tf(None, [0.0, 0.0, dz]) @ tf_inv(T_EP), a.v_down, a.acc, a.stop_n,
                              abs(dz) / (1e3 * a.v_down) + 5.0, f"descent {dz:+.2f} mm along z_P",
                              check=meas.watch(p_m, q_m, a.watch_mm, a.watch_deg))
            if why:
                print("the descent stopped: look at the fingers and the object before anything else", flush=True)
        p_m, q_m, info = meas.pose(a.measure_s)
        d, psi = report("final", p_m, q_m, p_n, q_n, shape, info)
        ok = float(np.linalg.norm(d)) <= 1.0 and abs(math.degrees(psi)) <= 2.0
        print(f"{'READY' if ok else 'NOT within 1 mm and 2 deg'}: {np.linalg.norm(d):.2f} mm and "
              f"{math.degrees(psi):+.2f} deg from the grasp's nominal pose"
              + ("" if ok else "; approach --fine corrects up to 3 mm laterally at this height"), flush=True)
    finally:
        meas.stop()
        arm.close()
    return 0


def cmd_ready(a):
    """The whole recentre before a trial (owner, 2026-11-16: "skill-ify / make quicker the process of translation to
    recenter the grasp"): the fingers straight (zero), the tracked approach (up first when the hand is below the
    grasp's height, then sideways, then down), the open pose in small steps, and a last measurement."""
    if not a.go:
        print("ready (dry): zero, approach, open; each prints what it would do")
    cmd_zero(a)
    a.fingers_straight = True
    cmd_approach(a)
    if not a.go:
        return 0
    cmd_open(a)
    meas = Measure(a)
    try:
        plan = load_plan(a.plan)
        p_n, q_n = nominal(plan)
        p_m, q_m, info = meas.pose(a.measure_s)
        d, psi = report("ready", p_m, q_m, p_n, q_n, plan["meta"]["hom"]["shape"], info)
        ok = float(np.linalg.norm(d)) <= 1.0 and abs(math.degrees(psi)) <= 2.0
        print(("READY for the trial" if ok else "NOT READY") + f": {np.linalg.norm(d):.2f} mm, "
              f"{math.degrees(psi):+.2f} deg from the grasp's nominal pose, fingers at the open pose", flush=True)
    finally:
        meas.stop()
    return 0


def cmd_putdown(a):
    """After a trial, the hand holding the object up: down along base z at 10 mm/s until the block touches the table
    (the HEX-E's force change past --stop-n; at most --max-down mm), then the fingers to the open pose in small steps
    (owner, 2026-11-16: "if we slowly release the servos the block should come to a flat rest")."""
    print(f"putdown: down at most {a.max_down:.0f} mm until a {a.stop_n} N force change, then the open pose over "
          f"{a.ramp_s:g} s")
    if not a.go:
        return 0
    arm = Arm(a.ur_ft, a.blank_s)
    try:
        arm.connect(a.stop_n is not None)
        T_goal = arm.T_BE()
        T_goal[2, 3] -= a.max_down
        why, moved = arm.move(T_goal, a.v, a.acc, a.stop_n, a.max_down / (1e3 * a.v) + 5.0, "down to the table")
        if why is None:
            print(f"   no touchdown within {a.max_down:.0f} mm: the object may hang above the table", flush=True)
    finally:
        arm.close()
    cmd_open(a)
    return 0


def cmd_release(a):
    st = Station(a.station, a.token_file)
    plan = load_plan(a.plan)
    tgt = open_pose(plan)
    print("   " + show_pose("open (plan)", tgt) + f"; then the arm up {a.clear:.0f} mm")
    if not a.go:
        return 0
    st.call("POST", "/api/v1/manual/joints", {"joints": tgt, "servo_speed": a.speed})
    time.sleep(a.wait_s)
    arm = Arm(a.ur_ft, a.blank_s)
    try:
        arm.connect(a.stop_n is not None)
        T_goal = arm.T_BE()
        T_goal[2, 3] += a.clear
        arm.move(T_goal, a.v, a.acc, a.stop_n, a.clear / (1e3 * a.v) + 5.0, f"up {a.clear:.0f} mm")
    finally:
        arm.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, plan=True, station=True):
        if plan:
            p.add_argument("--plan", required=True)
        if station:
            p.add_argument("--station", default="http://10.99.99.2:8765")
            p.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
        p.add_argument("--go", action="store_true", help="move; without it the command only prints")
        p.add_argument("--stop-n", type=float, default=3.0, help="N of force change that stops an arm move")
        p.add_argument("--no-guard", dest="stop_n", action="store_const", const=None,
                       help="no force guard on arm moves (the owner watching; 2026-11-16: the HEX-E's stream is down and "
                            "the UR's TCP force jumped 16 N on a free move)")
        p.add_argument("--ur-ft", action="store_true", help="guard arm moves with the UR's TCP force, not the HEX-E's stream")
        p.add_argument("--blank-s", type=float, default=0.4, help="s at a move's start without the force check")
        p.add_argument("--acc", type=float, default=0.05, help="m/s^2 of the arm's moves")

    p = sub.add_parser("status")
    common(p, plan=False)
    p.add_argument("--plan", default=None)
    p = sub.add_parser("torque-on")
    common(p, plan=False)
    p = sub.add_parser("layout")
    common(p)
    p.add_argument("--speed", type=int, default=40)
    p.add_argument("--video", action="store_true", help="film the reconfiguration with both cameras")
    p.add_argument("--video-map", default=str(HERE.parent / "maps/20261116-sphere_fitted_map_O.json"))
    for name in ("open", "zero", "release"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--speed", type=int, default=40, help="the servos' goal speed (the station's default is 80)")
        p.add_argument("--wait-s", type=float, default=4.0)
        p.add_argument("--ramp-s", type=float, default=0.0, help="open: reach the pose in small steps over this long")
        p.add_argument("--clear", type=float, default=80.0)
        p.add_argument("--v", type=float, default=0.01, help="m/s of the arm")
    p = sub.add_parser("place")
    common(p, station=False)
    p.add_argument("--map", required=True, help="the object's tag map in O (maps/), for its height on the table")
    p.add_argument("--table-z", type=float, default=3.0,
                   help="the table's height in the base frame (mm; 3.0 from the blocks resting there on 2026-11-16)")
    p.add_argument("--clear", type=float, default=80.0, help="mm above the grasp's height")
    p.add_argument("--max-dz", type=float, default=160.0)
    p.add_argument("--v", type=float, default=0.01, help="m/s")
    p.add_argument("--no-level", dest="level", action="store_false", help="keep z_P's tilt")
    p.add_argument("--max-level", type=float, default=5.0, help="deg of tilt the levelling may take out")
    for name in ("ready", "putdown"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--speed", type=int, default=40)
        p.add_argument("--wait-s", type=float, default=2.0)
        p.add_argument("--ramp-s", type=float, default=2.0 if name == "ready" else 4.0)
        p.add_argument("--v", type=float, default=0.01, help="m/s")
        p.add_argument("--max-down", type=float, default=80.0)
        if name == "ready":
            _approach_args(p)
    p = sub.add_parser("approach")
    common(p, station=False)
    _approach_args(p)
    a = ap.parse_args()
    return {"status": cmd_status, "torque-on": cmd_torque_on, "open": cmd_open, "zero": cmd_zero, "layout": cmd_layout,
            "place": cmd_place, "approach": cmd_approach, "release": cmd_release, "ready": cmd_ready,
            "putdown": cmd_putdown}[a.cmd](a)


def _approach_args(p):
    p.add_argument("--object", choices=("cylinder", "sphere"), required=True)
    p.add_argument("--map", required=True)
    p.add_argument("--measure-s", type=float, default=1.0)
    p.add_argument("--max-lat", type=float, default=40.0)
    p.add_argument("--max-yaw", type=float, default=20.0)
    p.add_argument("--max-dz", type=float, default=100.0)
    p.add_argument("--up-ok", type=float, default=3.0, help="mm the object may sit above the grasp's height")
    p.add_argument("--fine", action="store_true", help="only the lateral correction, at this height")
    p.add_argument("--fine-max", type=float, default=3.0)
    p.add_argument("--v-lat", type=float, default=0.015, help="m/s")
    p.add_argument("--v-down", type=float, default=0.010, help="m/s")
    p.add_argument("--watch-mm", type=float, default=1.5, help="the object's lateral motion that stops the descent")
    p.add_argument("--watch-deg", type=float, default=3.0, help="the object's turn that stops the descent")
    p.add_argument("--fingers-straight", action="store_true",
                   help="the fingers are at zero: the hand may rise first when it is below the grasp's height")


if __name__ == "__main__":
    sys.exit(main())
