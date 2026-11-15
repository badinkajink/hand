"""One HOM trial on the SR2 hand on the UR5, run on corvus (owner, 2026-11-15: "make sure we are ready to start
collecting data and calibrating from real trials of HOM").

corvus's end of the controller link (the loop of docs/experiments/20261113-tag_tracking_throughput/lab/hand_link.py)
with what a trial adds: the arm's lift at --lift-at s of the link's clock, a record of every period, and the HEX-E. The
HOM controller (python -m sr2hand.control.hom_server serve, on the workstation or here) answers each state; its first
state is the trial's t = 0, the time the controller's schedule and this script's lift both count from.

    check    the arm's pose, the tracker's object pose and the station, without motion: what a trial would start from
    trial    the link loop; with --arm the lift (moveL up --lift-mm in the base frame at --lift-v and --lift-a, the
             HEX-E's force change past --stop-N or a protective stop ends it with stopL); with --live the targets go to
             the station's stream (the owner present, a homed station with torque on and the layout's plan loaded).
             The hand holds the object at the end: the station holds the last goal when the stream ends.
    return   the guarded move back down by --lift-mm at 10 mm/s with the same HEX-E stop, after the operator has
             looked at the object (it must be over the seat)

Each trial writes OUT_DIR/<stamp>_<task>_corvus.jsonl: a header, then per period the state sent (joints and their
time, the object's pose in P with its capture time, flags and tag count), the arm's TCP pose (mm, rotation vector) and
the HEX-E's wrench at that time, the target applied (sim degrees, its sequence, age and round trip), the station's
status and the lift's phase. The controller's own record (hom_server) holds the same periods by sequence number.

    /usr/bin/python3 hom_trial.py check --object cylinder --map MAP.json
    /usr/bin/python3 hom_trial.py trial --task press --controller 100.x.y.z:47002 --object cylinder --map MAP.json
        [--station http://10.99.99.2:8765 --live] [--arm] [--ft] --out-dir OUT
    /usr/bin/python3 hom_trial.py return [--lift-mm 40]

Defaults are the cautious ones: no --live, no --arm, no --ft, so a bare `trial` moves nothing and logs what it would
have sent. The lift's default, 60 mm/s at 0.3 m/s^2 (40 mm in about 0.9 s), is gentler than the simulation's 0.4 s; the
object leaves the seat as the lift starts, and the hand has risen about 18 mm by the task's start at 2.0 s.
"""
import argparse
import http.client
import json
import math
import os
import select
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/sr2hand/driver/reskin5x/host")]
from sr2hand.control.link import DRY_RUN, STOP, STOPPED, STREAMING, StateMsg, TargetMsg  # noqa: E402
from sr2hand.geometry.rotations import log_so3  # noqa: E402

CAL = ROOT / "docs/experiments/20261113-camera_arm_calibration/data/fits"
ROBOT, FT = "192.168.0.4", "192.168.0.6"


class Station:
    """Keep-alive HTTP to the CB1's station: state polling (read-only) and the policy stream (--live only); the
    hand_link.py class."""

    def __init__(self, url, token_file, live):
        u = urlparse(url)
        self.host, self.port, self.live = u.hostname, u.port or 80, live
        self.token = Path(token_file).read_text().strip() if live else None
        self.stream_token = None
        self.joints, self.t_joints, self.error, self.state, self.suspended = None, float("nan"), None, {}, None
        self.stop_evt = threading.Event()

    def _conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=1.0)

    def poll(self, hz=20.0):
        c = self._conn()
        while not self.stop_evt.is_set():
            try:
                t_req = time.time() * 1000.0
                c.request("GET", "/api/v1/state")
                s = json.loads(c.getresponse().read())
                self.state = s
                tel = s.get("telemetry", {})
                sv = tel.get("servos", {})
                if sv:
                    self.joints = np.array([[sv[str(k)][j] for j in ("aa", "fe1", "fe2")] for k in range(3)], float)
                    # the reading's own age on the CB1 (its clock is hours off corvus's, so its timestamps are not
                    # used): while a policy stream runs the station stops reading the servos and this age grows,
                    # where the poll's time made stale angles look fresh (hand_link.py)
                    age = tel.get("servo_age_s")
                    self.t_joints = t_req - 1000.0 * float(age) if age is not None else float("nan")
                    self.suspended = bool(tel.get("servo_polling_suspended"))
            except (OSError, http.client.HTTPException, ValueError, KeyError) as e:
                self.error = repr(e)
                c.close()
                c = self._conn()
                time.sleep(0.5)
            time.sleep(1.0 / hz)

    def _post(self, c, path, body):
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json", "X-Manta-Token": self.token})
        r = c.getresponse()
        out = r.read()
        if r.status != 200:
            raise RuntimeError(f"{path}: {r.status} {out[:200]!r}")
        return json.loads(out or b"{}")

    def start_stream(self):
        self.c = self._conn()
        self.stream_token = self._post(self.c, "/api/v1/stream/start", {"timeout_s": 0.25})["token"]

    def frame(self, joints, speed=None):
        body = {"token": self.stream_token, "joints": joints}
        if speed is not None:
            body["servo_speed"] = int(speed)
        self._post(self.c, "/api/v1/stream/frame", body)

    def end_stream(self):
        if self.stream_token:
            try:
                self._post(self.c, "/api/v1/stream/end", {"token": self.stream_token})
            finally:
                self.stream_token = None


def ur_pose(T):
    """A transform (mm) -> [x, y, z mm, rotation vector rad] for the record."""
    return [*np.round(T[:3, 3], 3).tolist(), *np.round(log_so3(T[:3, :3]), 6).tolist()]


def open_ft(enabled):
    """The HEX-E stream of the calibration program (100 Hz, its 50 Hz filter), or None."""
    if not enabled:
        return None
    import ur5_tip_calibration as C
    ft = C.OptoForceStream(FT, rate=100, bias=False, filt=3)
    ft.configure()
    ft.start()
    return ft


def ft_now(ft):
    if ft is None:
        return None, None
    t, v = ft.ring.latest()
    return t, (None if v is None else np.round(np.asarray(v, float)[:6], 4).tolist())


class Lift(threading.Thread):
    """The arm's lift at t_lift (host seconds): moveL up dz mm in the base frame, asynchronous, watched every 10 ms for the
    HEX-E's force change past stop_N from its reading before the move, a stale HEX-E, or a protective stop."""

    def __init__(self, t_lift, dz, v, acc, stop_n, ft, arm_enabled):
        super().__init__(daemon=True, name="lift")
        self.t_lift, self.dz, self.v, self.acc, self.stop_n, self.ft, self.arm = t_lift, dz, v, acc, stop_n, ft, arm_enabled
        self.phase, self.rec = "waiting", {}

    def run(self):
        while time.time() < self.t_lift:
            time.sleep(0.001)
        self.rec["t_start_ms"] = time.time() * 1000.0
        if not self.arm:
            self.phase, self.rec["note"] = "logged only (no --arm)", "dry run: no motion"
            return
        import rtde_control
        import rtde_receive
        r = rtde_receive.RTDEReceiveInterface(ROBOT, 125.0)
        c = rtde_control.RTDEControlInterface(ROBOT)
        try:
            c.setTcp([0.0] * 6)
            p0 = np.array(r.getActualTCPPose())
            tgt = p0.copy()
            tgt[2] += self.dz / 1000.0
            F0 = None
            if self.ft is not None:
                _, v0 = self.ft.ring.since(time.time() - 0.2)
                F0 = np.asarray(v0, float)[:, :3].mean(axis=0) if len(v0) else None
            self.phase = "lifting"
            c.moveL(tgt.tolist(), self.v, self.acc, True)
            t0, fmax, why = time.time(), 0.0, None
            time.sleep(0.02)
            while True:
                if self.ft is not None and F0 is not None:
                    t, v = self.ft.ring.latest()
                    if t is None or time.time() - t > 0.1:
                        why = "no HEX-E sample for 100 ms"
                    else:
                        fmax = max(fmax, float(np.linalg.norm(np.asarray(v, float)[:3] - F0)))
                        if fmax > self.stop_n:
                            why = f"HEX-E force changed {fmax:.2f} N (limit {self.stop_n:g})"
                if why is None and (r.isProtectiveStopped() or r.isEmergencyStopped()):
                    why = "protective or emergency stop"
                if why:
                    c.stopL(1.0)
                    break
                if c.getAsyncOperationProgress() < 0 and time.time() - t0 > 0.1:
                    break
                if time.time() - t0 > 5.0:
                    why = "lift over 5 s"
                    c.stopL(1.0)
                    break
                time.sleep(0.01)
            p1 = np.array(r.getActualTCPPose())
            self.rec.update(t_end_ms=time.time() * 1000.0, from_m=p0.round(6).tolist(), to_m=p1.round(6).tolist(),
                            dz_mm=round(1000.0 * float(p1[2] - p0[2]), 3), max_dF_N=round(fmax, 3), stopped=why)
            self.phase = f"stopped: {why}" if why else "up"
        finally:
            try:
                c.stopScript()
                c.disconnect()
            finally:
                r.disconnect()


def start_tracker(a, on_sample):
    from sr2hand.robot.tool_chain import SR2Hand
    from sr2hand.tracking import live
    arm = live.ArmPose()
    arm.start()
    tracker = live.Tracker(live.load_cameras(CAL / "20261113-lens_board_k3_arm.json",
                                             CAL / "20261113-handeye_tcp_fitted_lens_moved836.json"),
                           live.load_map(a.map), arm=arm, T_EP=SR2Hand.load().T_E_P(), on_sample=on_sample).start()
    return arm, tracker


def cmd_check(a):
    """The arm's pose, the tracked object in P and the station's state for a few seconds; no motion."""
    latest = {"msg": None}
    arm, tracker = start_tracker(a, lambda s: latest.__setitem__("msg", s.msg))
    station = None
    if a.station:
        station = Station(a.station, a.token_file, False)
        threading.Thread(target=station.poll, daemon=True).start()
    try:
        t0 = time.time()
        while time.time() - t0 < a.seconds:
            time.sleep(0.5)
            m = latest["msg"]
            T = arm.T_BE
            print(f"{time.time() - t0:5.1f} s  arm {ur_pose(T) if T is not None else arm.error}  object "
                  + ("none" if m is None else f"p_PO {np.round(m.p_PO, 2).tolist()} mm q_PO {np.round(m.q_PO, 4).tolist()} "
                     f"tags {m.n_tags} flags {m.flags} age {time.time() * 1000.0 - m.t_capture_ms:.0f} ms")
                  + ("" if station is None else f"  station joints {None if station.joints is None else np.round(station.joints, 1).tolist()} "
                     f"error {station.error}"), flush=True)
    finally:
        tracker.stop()
        arm.stop()
    return 0


def cmd_trial(a):
    P = 1.0 / a.hz
    ctrl = (a.controller.rsplit(":", 1)[0], int(a.controller.rsplit(":", 1)[1]))
    station = None
    if a.station:
        station = Station(a.station, a.token_file, a.live)
        threading.Thread(target=station.poll, daemon=True).start()
    elif a.live:
        sys.exit("--live needs --station")
    latest = {"msg": None}
    arm = tracker = None
    if not a.no_tracker:
        arm, tracker = start_tracker(a, lambda s: latest.__setitem__("msg", s.msg))
        time.sleep(a.settle_s)                                   # the cameras' first frames
    ft = open_ft(a.ft)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}_{a.task}_corvus.jsonl"
    log = open(path, "w")
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", a.listen))
    rx.setblocking(False)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent, status = {}, (DRY_RUN if not a.live else 0)
    last_seq, t_last_new, armed = -1, None, False
    seq = 0
    t0 = time.time()
    lift = Lift(t0 + a.lift_at, a.lift_mm, a.lift_v, a.lift_a, a.stop_n, ft, a.arm) if a.lift_mm > 0 else None
    log.write(json.dumps({"kind": "header", "task": a.task, "object": a.object, "map": a.map, "controller": a.controller,
                          "live": a.live, "arm": a.arm, "ft": a.ft, "hz": a.hz, "t0_ms": t0 * 1000.0,
                          "lift": {"at_s": a.lift_at, "mm": a.lift_mm, "v_mps": a.lift_v, "a_mps2": a.lift_a,
                                   "stop_N": a.stop_n}, "host": socket.gethostname(),
                          "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
    if lift is not None:
        lift.start()
    next_t = time.perf_counter()
    print(f"{time.strftime('%H:%M:%S')} trial {a.task}: {1000 * P:.0f} ms period, controller {ctrl}, "
          f"{'LIVE' if a.live else 'dry run'}, arm {'on' if a.arm else 'off'}, HEX-E {'on' if ft else 'off'} -> {path}",
          flush=True)
    try:
        while time.time() - t0 < a.seconds:
            now = time.time() * 1000.0
            st = StateMsg(seq=seq, t_ms=now, last_target_seq=max(last_seq, 0), status=status)
            if station is not None and station.joints is not None:
                st.joints, st.t_joints_ms = station.joints.ravel(), station.t_joints
            m = latest["msg"]
            if m is not None:
                st.t_capture_ms, st.p_PO, st.q_PO, st.obj_flags, st.n_tags = (m.t_capture_ms, m.p_PO, m.q_PO, m.flags,
                                                                              m.n_tags)
            tx.sendto(st.pack(), ctrl)
            sent[seq] = now
            sent.pop(seq - 500, None)
            newest, deadline = None, time.perf_counter() + a.wait_ms / 1000.0
            while True:
                r_, _, _ = select.select([rx], [], [], max(0.0, deadline - time.perf_counter()))
                if not r_:
                    break
                try:
                    tg = TargetMsg.unpack(rx.recv(256))
                except BlockingIOError:
                    continue
                if tg is not None and tg.seq > last_seq and (newest is None or tg.seq > newest.seq):
                    newest = tg
                if newest is not None and newest.state_seq == seq:
                    break
            T_BE = arm.at(now) if arm is not None else None
            t_ft, w = ft_now(ft)
            row = {"seq": seq, "t_ms": round(now, 3), "t": round((now - 1000.0 * t0) / 1000.0, 4),
                   "joints": None if st.joints is None or not np.all(np.isfinite(st.joints)) else np.round(st.joints, 3).tolist(),
                   "t_joints_ms": st.t_joints_ms if math.isfinite(st.t_joints_ms) else None,
                   "obj": None if m is None else {"p_PO": np.round(m.p_PO, 4).tolist(), "q_PO": np.round(m.q_PO, 6).tolist(),
                                                  "t_capture_ms": m.t_capture_ms, "flags": m.flags, "n_tags": m.n_tags},
                   "arm": None if T_BE is None else ur_pose(T_BE), "ft": w, "t_ft_ms": t_ft,
                   "servo_polling_suspended": None if station is None else station.suspended,
                   "lift": None if lift is None else lift.phase, "status": status}
            now2 = time.time() * 1000.0
            if newest is not None and not status & STOPPED:
                if newest.flags & STOP:
                    status |= STOPPED
                    row["stop"] = "controller"
                else:
                    if a.live:
                        if station.stream_token is None:
                            station.start_stream()
                            status |= STREAMING
                        station.frame(newest.joints_frame(), None if np.isnan(newest.servo_speed) else newest.servo_speed)
                    last_seq, t_last_new, armed = newest.seq, now2, True
                    row.update(target=newest.seq, target_sim_deg=np.round(newest.targets, 4).tolist(),
                               target_age=round(now2 - newest.t_ms, 3),
                               rtt=round(now2 - sent[newest.state_seq], 3) if newest.state_seq in sent else None)
            if armed and not status & STOPPED and now2 - t_last_new > a.miss * 1000.0 * P:
                status |= STOPPED
                row["stop"] = f"no target for {a.miss} periods"
            if status & STOPPED and status & STREAMING:
                station.end_stream()
                status &= ~STREAMING
            log.write(json.dumps(row) + "\n")
            seq += 1
            if status & STOPPED and time.time() - t0 > a.lift_at + 0.5:
                break
            next_t += P
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                next_t = time.perf_counter()
    finally:
        if station is not None and station.stream_token:
            station.end_stream()
        if lift is not None:
            lift.join(timeout=6.0)
        log.write(json.dumps({"kind": "end", "states": seq, "last_target": last_seq, "status": status,
                              "lift": None if lift is None else {"phase": lift.phase, **lift.rec},
                              "station_error": None if station is None else station.error}) + "\n")
        log.close()
        if tracker is not None:
            tracker.stop()
        if arm is not None:
            arm.stop()
        print(f"{time.strftime('%H:%M:%S')} trial done: {seq} states, last target {last_seq}, status {status}, lift "
              f"{None if lift is None else lift.phase} -> {path}", flush=True)
        os._exit(0)


def cmd_return(a):
    """Down by --lift-mm at 10 mm/s, stopped by the HEX-E's force change past --stop-N or a protective stop."""
    import rtde_control
    import rtde_receive
    ft = open_ft(True)
    r = rtde_receive.RTDEReceiveInterface(ROBOT, 125.0)
    c = rtde_control.RTDEControlInterface(ROBOT)
    try:
        c.setTcp([0.0] * 6)
        time.sleep(0.6)
        _, v0 = ft.ring.since(time.time() - 0.4)
        F0 = np.asarray(v0, float)[:, :3].mean(axis=0)
        p0 = np.array(r.getActualTCPPose())
        tgt = p0.copy()
        tgt[2] -= a.lift_mm / 1000.0
        c.moveL(tgt.tolist(), 0.010, 0.1, True)
        t0, fmax, why = time.time(), 0.0, None
        while True:
            t, v = ft.ring.latest()
            if t is None or time.time() - t > 0.1:
                why = "no HEX-E sample for 100 ms"
            else:
                fmax = max(fmax, float(np.linalg.norm(np.asarray(v, float)[:3] - F0)))
                if fmax > a.stop_n:
                    why = f"HEX-E force changed {fmax:.2f} N (limit {a.stop_n:g})"
            if why is None and (r.isProtectiveStopped() or r.isEmergencyStopped()):
                why = "protective or emergency stop"
            if why:
                c.stopL(1.0)
                break
            if c.getAsyncOperationProgress() < 0 and time.time() - t0 > 0.3:
                break
            if time.time() - t0 > 30.0:
                why = "return over 30 s"
                c.stopL(1.0)
                break
            time.sleep(0.01)
        p1 = np.array(r.getActualTCPPose())
        print(f"return: {'STOPPED, ' + why if why else 'down'}; moved {1000 * (p1[2] - p0[2]):.2f} mm in z, HEX-E force "
              f"changed at most {fmax:.2f} N", flush=True)
    finally:
        try:
            c.stopScript()
            c.disconnect()
        finally:
            r.disconnect()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("check")
    pc.add_argument("--object", choices=("cylinder", "sphere"), required=True)
    pc.add_argument("--map", required=True, help="the tag map of the object's block in the object's frame")
    pc.add_argument("--station", default=None)
    pc.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
    pc.add_argument("--seconds", type=float, default=5.0)
    pt = sub.add_parser("trial")
    pt.add_argument("--task", required=True, choices=("pinpush", "press", "roll", "twist"))
    pt.add_argument("--controller", required=True, help="the HOM server's HOST:PORT")
    pt.add_argument("--object", choices=("cylinder", "sphere"), required=True)
    pt.add_argument("--map", default=None, help="the tag map of the object's block in the object's frame")
    pt.add_argument("--listen", type=int, default=47001)
    pt.add_argument("--hz", type=float, default=50.0)
    pt.add_argument("--miss", type=int, default=3)
    pt.add_argument("--wait-ms", type=float, default=8.0)
    pt.add_argument("--seconds", type=float, default=9.0, help="the link's length; the controller's STOP ends it first")
    pt.add_argument("--settle-s", type=float, default=2.0, help="the cameras' start before t = 0")
    pt.add_argument("--station", default=None)
    pt.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
    pt.add_argument("--live", action="store_true")
    pt.add_argument("--no-tracker", action="store_true")
    pt.add_argument("--arm", action="store_true", help="move the arm (the lift); without it the lift is only logged")
    pt.add_argument("--ft", action="store_true", help="stream the HEX-E (needed with --arm for the force stop)")
    pt.add_argument("--lift-at", type=float, default=1.6)
    pt.add_argument("--lift-mm", type=float, default=40.0)
    pt.add_argument("--lift-v", type=float, default=0.06, help="m/s")
    pt.add_argument("--lift-a", type=float, default=0.3, help="m/s^2")
    pt.add_argument("--stop-n", type=float, default=5.0, help="N of HEX-E force change that stops the lift")
    pt.add_argument("--out-dir", required=True)
    pr = sub.add_parser("return")
    pr.add_argument("--lift-mm", type=float, default=40.0)
    pr.add_argument("--stop-n", type=float, default=3.0)
    a = ap.parse_args()
    if a.cmd == "trial":
        if a.arm and not a.ft:
            sys.exit("--arm needs --ft: the lift is stopped by the HEX-E's force change")
        if not a.no_tracker and not a.map:
            sys.exit("--map is needed with the tracker")
        return cmd_trial(a)
    return cmd_check(a) if a.cmd == "check" else cmd_return(a)


if __name__ == "__main__":
    sys.exit(main())
