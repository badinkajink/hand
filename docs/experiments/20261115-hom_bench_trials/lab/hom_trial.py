"""One HOM trial on the SR2 hand on the UR5, run on corvus (owner, 2026-11-15: "make sure we are ready to start
collecting data and calibrating from real trials of HOM").

corvus's end of the controller link (the loop of docs/experiments/20261113-tag_tracking_throughput/lab/hand_link.py)
with what a trial adds: the arm's lift at --lift-at s of the link's clock, a record of every period, and the HEX-E. The
HOM controller (python -m sr2hand.control.hom_server serve, on the workstation or here) answers each state; its first
state is the trial's t = 0, the time the controller's schedule and this script's lift both count from.

    plan     load a plan of plans/ into the station from this clone (POST /api/v1/plans/load with the plan inline, so the
             plan need not be copied to the CB1); nothing moves until the station's morphology is applied
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
Every frame is clipped to the servos' safe range before it goes to the station (sr2hand.control.servo_limits, from
2026-11-17); a period whose target needed it records the joints under "clipped_corvus".

The fingertips (from 2026-11-17; owner, 2026-11-16: "lets make sure the next time we run trials we have the fingertip
sensing going"). A trial attaches to the boards' splitter (reskin5x_mux.py), or starts it, and refuses to start unless
all three boards stream at 100 records/s or more; --no-tips runs without them. Every board record from 2 s before t = 0
goes into OUT_DIR/<stamp>_<task>_tip_<finger>.bin (host time f8 on the trial's clock, 5 x (t x y z) f4;
tip_contacts.read_stream). Before t = 0, with the fingers still at the open pose, each board's rest is taken over
--rest-s; from then on tip_contacts.TipContacts estimates the contacts in a process of its own (tip_contacts.TipProcess,
from 2026-11-17; --tips-thread keeps the earlier thread of this process), and every state sends
the newest in StateMsg.cop (per finger the point in P, mm, and the normal force, N; NaN without a contact), placed with
the joints that state sends and the mounts of --plan. Each period's record has them under "tips". Whether HOM uses them
is the server's choice (hom_server serve --tips).

    /usr/bin/python3 hom_trial.py plan --plan ../plans/20261115-generalist_press_plan.json
    /usr/bin/python3 hom_trial.py check --object cylinder --map MAP.json
    /usr/bin/python3 hom_trial.py trial --task press --controller 100.x.y.z:47002 --object cylinder --map MAP.json
        --plan PLAN [--station http://10.99.99.2:8765 --live] [--arm] [--ft] --out-dir OUT [--no-tips]
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
import queue
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
from sr2hand.control import servo_limits as SL  # noqa: E402
from sr2hand.geometry.rotations import log_so3  # noqa: E402

CAL = ROOT / "docs/experiments/20261113-camera_arm_calibration/data/fits"
ROBOT = "192.168.0.4"
FT = "192.168.1.1"     # the HEX-E's compute box since its recabling on 2026-11-16 (it was 192.168.0.6 by DHCP)


class Station:
    """Keep-alive HTTP to the CB1's station: state polling (read-only) and the policy stream (--live only); the
    hand_link.py class."""

    def __init__(self, url, token_file, live, read_back=True):
        u = urlparse(url)
        self.host, self.port, self.live, self.read_back = u.hostname, u.port or 80, live, read_back
        self.token = Path(token_file).read_text().strip() if live else None
        self.stream_token, self.n_read, self.n_frames = None, 0, 0
        self.lease_s, self.n_restarts, self.errors = 1.0, 0, {}
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
        # a 1 s lease (2026-11-16: at 0.25 s a frame held up behind the station's lock let the lease lapse after 33
        # frames, and the station then held the grip's goals until the servos' overload protection let go)
        self.stream_token = self._post(self.c, "/api/v1/stream/start", {"timeout_s": self.lease_s})["token"]
        self.q = queue.Queue(maxsize=1)
        threading.Thread(target=self._frames, daemon=True, name="frames").start()

    def frame(self, joints, speed=None):
        """Hand the newest frame to the worker; a frame still waiting is dropped. The loop never waits on HTTP: with the
        read-back a frame takes the station about 9 ms more."""
        try:
            self.q.get_nowait()
        except queue.Empty:
            pass
        self.q.put_nowait((joints, speed))

    def _frames(self):
        """POST each frame on a connection of its own; with read_back the station reads the nine servos after writing
        and answers with them (a station without the change ignores the flag and answers without them)."""
        c = self._conn()
        while self.stream_token is not None:
            try:
                joints, speed = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            body = {"token": self.stream_token, "joints": joints, "read_back": self.read_back}
            if speed is not None:
                body["servo_speed"] = int(speed)
            try:
                r = self._post(c, "/api/v1/stream/frame", body)
                self.n_frames += 1
                sv = r.get("servos")
                if sv:
                    self.joints = np.array([[sv[str(k)][j] for j in ("aa", "fe1", "fe2")] for k in range(3)], float)
                    self.t_joints = time.time() * 1000.0 - 1.0      # read at the end of the station's handler
                    self.n_read += 1
            except (OSError, http.client.HTTPException, RuntimeError, ValueError, KeyError) as e:
                self.error = repr(e)
                key = type(e).__name__ + (": lease" if "absent or expired" in str(e) else "")
                self.errors[key] = self.errors.get(key, 0) + 1
                c.close()
                c = self._conn()
                # no restart after a lapsed lease: the first frame of a new stream jumped the fingers to the
                # controller's targets of that moment (owner, 2026-11-16: "on the lift they spasmed quickly")

    def diagnostics(self):
        """The station's one read of each servo (POST /api/v1/servos/diagnostics, 2026-11-16): supply voltage (0.1 V),
        temperature, torque_enable, status byte and the servos that look rebooted; None from a station without it."""
        c = self._conn()
        c.timeout = 60.0
        try:
            return self._post(c, "/api/v1/servos/diagnostics", {})
        except (OSError, http.client.HTTPException, RuntimeError, ValueError) as e:
            return {"error": repr(e)[:200]}
        finally:
            c.close()

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

    def __init__(self, t_lift, dz, v, acc, stop_n, ft, arm_enabled, rtde=None):
        super().__init__(daemon=True, name="lift")
        self.t_lift, self.dz, self.v, self.acc, self.stop_n, self.ft, self.arm = t_lift, dz, v, acc, stop_n, ft, arm_enabled
        self.rtde = rtde
        self.phase, self.rec = "waiting", {}
        self.cancel = threading.Event()               # set before t_lift: no lift (a trial without a target)

    @staticmethod
    def connect():
        """(rtde_receive, rtde_control) opened before the trial's t = 0: opening the control interface holds the
        interpreter for about 1.5 s (2026-11-16: opened at the lift, it froze the link loop and the frames, the
        controller ended the trial after 1 s without a state and the station's stream lease expired)."""
        import rtde_control
        import rtde_receive
        r = rtde_receive.RTDEReceiveInterface(ROBOT, 125.0)
        c = rtde_control.RTDEControlInterface(ROBOT)
        c.setTcp([0.0] * 6)
        return r, c

    def run(self):
        while time.time() < self.t_lift:
            if self.cancel.is_set():
                self.phase, self.rec["note"] = "cancelled", "no lift: the trial stopped before it"
                return
            time.sleep(0.001)
        self.rec["t_start_ms"] = time.time() * 1000.0
        if not self.arm:
            self.phase, self.rec["note"] = "logged only (no --arm)", "dry run: no motion"
            return
        r, c = self.rtde
        try:
            p0 = np.array(r.getActualTCPPose())
            tgt = p0.copy()
            tgt[2] += self.dz / 1000.0
            F0 = None
            if self.ft == "ur":
                F0 = np.mean([np.asarray(r.getActualTCPForce(), float)[:3] for _ in range(25) if time.sleep(0.008) is None],
                             axis=0)
            elif self.ft is not None:
                _, v0 = self.ft.ring.since(time.time() - 0.2)
                F0 = np.asarray(v0, float)[:, :3].mean(axis=0) if len(v0) else None
            self.phase = "lifting"
            c.moveL(tgt.tolist(), self.v, self.acc, True)
            t0, fmax, why = time.time(), 0.0, None
            time.sleep(0.02)
            while True:
                if self.ft is not None and F0 is not None:
                    t, v = ((time.time(), r.getActualTCPForce()) if self.ft == "ur" else self.ft.ring.latest())
                    src = "UR TCP force" if self.ft == "ur" else "HEX-E"
                    if t is None or time.time() - t > 0.1:
                        why = f"no {src} sample for 100 ms"
                    else:
                        fmax = max(fmax, float(np.linalg.norm(np.asarray(v, float)[:3] - F0)))
                        if fmax > self.stop_n:
                            why = f"{src} force changed {fmax:.2f} N (limit {self.stop_n:g})"
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
        except Exception as e:                            # noqa: BLE001  the record says so; close() still runs
            self.phase = f"error: {e!r}"

    def close(self):
        """The RTDE interfaces closed after the trial's loop (stopScript and disconnect hold the interpreter too)."""
        if self.rtde is None:
            return
        r, c = self.rtde
        try:
            c.stopScript()
            c.disconnect()
        finally:
            r.disconnect()


def cam_mode():
    """(width, height, fps, exposure or None) of the bench's tracker: HOM_CAM_MODE "WxH@FPS" and HOM_CAM_EXPOSURE (0.1 ms,
    the RealSense unit), set by hom_series --cam-mode/--cam-exposure (2026-11-20, owner: "id like to prioritize (within
    reason) the 60fps tracking"); 1920x1080@30 at the calibration's exposure (live.SESSION, 166) without them."""
    wh, fps = os.environ.get("HOM_CAM_MODE", "1920x1080@30").split("@")
    w, h = (int(x) for x in wh.split("x"))
    e = os.environ.get("HOM_CAM_EXPOSURE")
    return w, h, int(fps), (float(e) if e else None)


class VideoRecorder:
    """Each camera's frames during a trial (owner, 2026-11-16: "make sure to record video from the realsense's"): the
    tracker's luma frames, at 960 x 540 (halved from 1080p), into OUT/<stem>_<serial>.mp4 at the tracker's rate by a
    writer thread, with each frame's camera time and number in OUT/<stem>_<serial>_frames.csv, so the film lines up with
    the trial's record."""

    def __init__(self, stem, scale=None, fps=None):
        import cv2
        w, _h, f, _e = cam_mode()
        scale = 960.0 / w if scale is None else scale
        fps = float(f) if fps is None else fps
        self.cv2, self.stem, self.scale, self.fps = cv2, stem, scale, fps
        self.q = queue.Queue(maxsize=240)
        self.writers, self.stamps, self.dropped, self.written = {}, {}, 0, 0
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="video")
        self.thread.start()

    def on_frame(self, sn, y, det):
        try:
            self.q.put_nowait((sn, self.cv2.resize(y, None, fx=self.scale, fy=self.scale,
                                                   interpolation=self.cv2.INTER_AREA), det.t_frame_ms, det.frame))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while not (self.stop_evt.is_set() and self.q.empty()):
            try:
                sn, img, t_ms, n = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if sn not in self.writers:
                h, w = img.shape[:2]
                self.writers[sn] = self.cv2.VideoWriter(f"{self.stem}_{sn}.mp4", self.cv2.VideoWriter_fourcc(*"mp4v"),
                                                        self.fps, (w, h), isColor=False)
                self.stamps[sn] = open(f"{self.stem}_{sn}_frames.csv", "w")
                self.stamps[sn].write("frame,t_camera_ms,t_host_ms\n")
            self.writers[sn].write(img)
            self.stamps[sn].write(f"{n},{t_ms:.3f},{time.time() * 1000.0:.3f}\n")
            self.written += 1

    def close(self):
        self.stop_evt.set()
        self.thread.join(timeout=10)
        for w in self.writers.values():
            w.release()
        for f in self.stamps.values():
            f.close()
        return {"files": [f"{self.stem}_{sn}.mp4" for sn in self.writers], "frames": self.written,
                "dropped": self.dropped}


def start_tracker(a, on_sample, on_frame=None):
    from sr2hand.robot.tool_chain import SR2Hand
    from sr2hand.tracking import live
    arm = live.ArmPose()
    arm.start()
    w, h, fps, exposure = cam_mode()
    tracker = live.Tracker(live.load_cameras(CAL / "20261113-lens_board_k3_arm.json",
                                             CAL / "20261113-handeye_tcp_fitted_lens_moved836.json", width=w, height=h),
                           live.load_map(a.map), fps=fps, arm=arm, T_EP=SR2Hand.load().T_E_P(), on_sample=on_sample,
                           settings=None if exposure is None else dict(live.SESSION, exposure=exposure))
    if on_frame is not None:
        for w in tracker.workers:
            w.on_frame = on_frame
    return arm, tracker.start()


def open_tips(a, stem):
    """The fingertip boards for a trial: the splitter attached to or started, its three streams checked, every record
    from 2 s back written to <stem>_<finger>.bin, and the estimator built on the plan's mounts. Exits with the reason
    when a board does not stream (--no-tips runs the trial without them). Returns (TipContacts, Outlets, files)."""
    import tip_contacts as TC
    if not a.plan:
        sys.exit("the fingertips need --plan (the mounts in P that place their contacts), or --no-tips")
    outlets, why = None, None
    try:
        if a.tips_links:                                  # a test feed (tip_contacts.py play), not the splitter
            pids, how = [], f"not checked (test links {a.tips_links})"
        else:
            pids, how = TC.ensure_splitter(start=True)
        if not a.tips_thread:
            # the feed, the records and the estimator in a process of their own (owner, 2026-11-17: "please fix the
            # estimator subprocess yes"): in this process its thread stalled the 50 Hz loop for 40-70 ms
            tips = TC.TipProcess(a.plan, str(stem), links=a.tips_links, gp=not a.no_tips_gp,
                                 own_pose=not a.no_own_pose)
            print(f"{time.strftime('%H:%M:%S')} fingertips: splitter {how} (pids {pids}), boards at "
                  + ", ".join(f"{f} {v:.0f}/s" for f, v in tips.rates().items())
                  + f"; estimator in process {tips.pid}; records -> {stem}_<finger>.bin", flush=True)
            return tips, tips, tips.files
        outlets = TC.Outlets(a.tips_links) if a.tips_links else TC.Outlets()
        outlets.start()
        why = TC.check_streams(outlets, 1.0)
    except (RuntimeError, OSError) as e:
        why = f"{type(e).__name__}: {e}"
    if why:
        if outlets is not None:
            outlets.close()
        sys.exit(f"fingertip boards: {why}; --no-tips runs the trial without them")
    files = outlets.record(str(stem), since=time.time() - 2.0)
    tips = TC.TipContacts(outlets, TC.load_plan_d6(a.plan), gp=not a.no_tips_gp, own_pose=not a.no_own_pose)
    print(f"{time.strftime('%H:%M:%S')} fingertips: splitter {how} (pids {pids}), boards at "
          + ", ".join(f"{f} {v:.0f}/s" for f, v in outlets.rates(1.0).items()) + f"; estimator in this process's "
          f"thread (--tips-thread); records -> {stem}_<finger>.bin", flush=True)
    return tips, outlets, files


def pose_error(p_mm, q, nominal):
    """(distance mm, angle deg) of the tracked pose in P from the grasp's nominal one (hom_server nominal)."""
    from sr2hand.geometry.rotations import quat_angle
    dp = float(np.linalg.norm(np.asarray(p_mm, float) - np.asarray(nominal["p_PO_mm"], float)))
    return dp, float(np.degrees(quat_angle(np.asarray(q, float), np.asarray(nominal["q_PO"], float))))


def cmd_check(a):
    """The arm's pose, the tracked object in P and the station's state for a few seconds; no motion. With --nominal
    (hom_server nominal --out) the tracked pose's distance and angle from the grasp's start pose: a trial starts within
    1 mm and 2 deg."""
    nominal = json.loads(Path(a.nominal).read_text()) if a.nominal else None
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
                     f"tags {m.n_tags} flags {m.flags} age {time.time() * 1000.0 - m.t_capture_ms:.0f} ms"
                     + ("" if nominal is None or not np.all(np.isfinite(m.p_PO)) else
                        "  from nominal %.2f mm %.2f deg" % pose_error(m.p_PO, m.q_PO, nominal)))
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
    arm = tracker = video = None
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    path = out / f"{stamp}_{a.task}_corvus.jsonl"
    tips = outlets = tip_files = tips_rest = None
    if a.tips:
        tips, outlets, tip_files = open_tips(a, out / f"{stamp}_{a.task}_tip")
    if not a.no_tracker:
        if a.video:
            # with --video-dir, the session's folder of hom_setup.py's recordings (film_session.py joins them)
            stem = (Path(a.video_dir) / f"{stamp}_trial_{a.label or a.task}" if a.video_dir
                    else out / f"{stamp}_{a.task}_video")
            stem.parent.mkdir(parents=True, exist_ok=True)
            video = VideoRecorder(str(stem))
        arm, tracker = start_tracker(a, lambda s: latest.__setitem__("msg", s.msg),
                                     on_frame=None if video is None else video.on_frame)
        time.sleep(a.settle_s)                                   # the cameras' first frames
    ft = "ur" if a.ur_ft else open_ft(a.ft)
    rtde = Lift.connect() if (a.arm and a.lift_mm > 0) else None

    def joints_now():
        return None if station is None or station.joints is None else np.array(station.joints, float)
    if tips is not None:
        # each board's rest with the fingers still at the open pose, before the close; then the estimator's thread
        tips_rest = tips.rest(time.time(), a.rest_s, refit=not a.no_tips_refit, joints_servo=joints_now())
        tips.start(1.0 / a.hz, joints=joints_now)
        print(f"{time.strftime('%H:%M:%S')} fingertip rests: " + "; ".join(
            f"{f} {r['model']} ({r['saved_fit_mT']:.2f} mT" + (f", new fit {r['new_fit_mT']:.2f}" if "new_fit_mT" in r else "")
            + ")" for f, r in tips_rest.items()), flush=True)
    log = open(path, "w")
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", a.listen))
    rx.setblocking(False)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent, status = {}, (DRY_RUN if not a.live else 0)
    last_seq, t_last_new, armed = -1, None, False
    seq = 0
    p_obj0 = None                                         # the object's first tracked position, for --max-drift-mm
    t0 = time.time()
    lift = Lift(t0 + a.lift_at, a.lift_mm, a.lift_v, a.lift_a, a.stop_n, ft, a.arm, rtde) if a.lift_mm > 0 else None
    log.write(json.dumps({"kind": "header", "task": a.task, "object": a.object, "map": a.map, "controller": a.controller,
                          "live": a.live, "arm": a.arm, "ft": "ur" if a.ur_ft else a.ft, "hz": a.hz, "t0_ms": t0 * 1000.0,
                          "lift": {"at_s": a.lift_at, "mm": a.lift_mm, "v_mps": a.lift_v, "a_mps2": a.lift_a,
                                   "stop_N": a.stop_n}, "host": socket.gethostname(), "plan": a.plan,
                          "tips": None if tips is None else {
                              "files": tip_files, "rest_s": a.rest_s, "rest": tips_rest, "gp": not a.no_tips_gp,
                              "estimator": "thread" if a.tips_thread else f"process {tips.pid}",
                              "own_pose": not a.no_own_pose, "refit": not a.no_tips_refit},
                          "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
    if lift is not None:
        lift.start()
    next_t = time.perf_counter()
    print(f"{time.strftime('%H:%M:%S')} trial {a.task}: {1000 * P:.0f} ms period, controller {ctrl}, "
          f"{'LIVE' if a.live else 'dry run'}, arm {'on' if a.arm else 'off'}, guard {'UR TCP force' if ft == 'ur' else 'HEX-E' if ft else 'off'} -> {path}",
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
            trec = t_est = None
            if tips is not None:
                # the newest estimate, placed in P with the joints this state sends
                t_est, est = tips.get()
                jj = np.asarray(st.joints, float)
                st.cop, trec = tips.to_P(est, jj.reshape(3, 3) if np.all(np.isfinite(jj)) else None)
                if "error" in est:
                    trec = {"error": est["error"]}
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
            t_ft, w = ft_now(None if ft == "ur" else ft)
            row = {"seq": seq, "t_ms": round(now, 3), "t": round((now - 1000.0 * t0) / 1000.0, 4),
                   "joints": None if st.joints is None or not np.all(np.isfinite(st.joints)) else np.round(st.joints, 3).tolist(),
                   "t_joints_ms": st.t_joints_ms if math.isfinite(st.t_joints_ms) else None,
                   "obj": None if m is None else {"p_PO": np.round(m.p_PO, 4).tolist(), "q_PO": np.round(m.q_PO, 6).tolist(),
                                                  "t_capture_ms": m.t_capture_ms, "flags": m.flags, "n_tags": m.n_tags},
                   "arm": None if T_BE is None else ur_pose(T_BE), "ft": w, "t_ft_ms": t_ft,
                   "servo_polling_suspended": None if station is None else station.suspended,
                   "lift": None if lift is None else lift.phase, "status": status}
            if tips is not None:
                row["tips"] = trec
                row["tips_age_ms"] = round(now - 1e3 * t_est, 1) if t_est is not None and math.isfinite(t_est) else None
            now2 = time.time() * 1000.0
            if m is not None and np.all(np.isfinite(m.p_PO)):
                if p_obj0 is None:
                    p_obj0 = np.array(m.p_PO, float)
                drift = float(np.linalg.norm(np.asarray(m.p_PO, float) - p_obj0))
                # the lift raises the hand and the object together, so the drift in P stays the object's own motion
                if a.max_drift_mm > 0 and drift > a.max_drift_mm and not status & STOPPED:
                    status |= STOPPED
                    row["stop"] = f"object drifted {drift:.1f} mm from its start in P (limit {a.max_drift_mm:g})"
                    print(f"{time.strftime('%H:%M:%S')} STOP: {row['stop']}", flush=True)
            if newest is not None and not status & STOPPED:
                if newest.flags & STOP:
                    status |= STOPPED
                    row["stop"] = "controller"
                else:
                    # the backstop of the server's clip (servo_limits, owner 2026-11-17: "i agree on the HOM target
                    # clipping"): nothing past the safe range reaches the station, whatever the server runs
                    frame = newest.joints_frame()
                    clipped = SL.clip_sim(frame)
                    if clipped:
                        row["clipped_corvus"] = clipped
                    if a.live:
                        if station.stream_token is None:
                            station.start_stream()
                            status |= STREAMING
                        station.frame(frame, None if np.isnan(newest.servo_speed) else newest.servo_speed)
                    last_seq, t_last_new, armed = newest.seq, now2, True
                    row.update(target=newest.seq, target_sim_deg=np.round(newest.targets, 4).tolist(),
                               target_age=round(now2 - newest.t_ms, 3),
                               rtt=round(now2 - sent[newest.state_seq], 3) if newest.state_seq in sent else None)
            if armed and not status & STOPPED and now2 - t_last_new > a.miss * 1000.0 * P:
                status |= STOPPED
                row["stop"] = f"no target for {a.miss} periods"
            if (not armed and not status & STOPPED and a.first_target_s > 0
                    and now2 - 1000.0 * t0 > 1000.0 * a.first_target_s):
                # 2026-11-17 16:49: no reply reached corvus, and the lift raised the open hand at 1.6 s anyway
                status |= STOPPED
                row["stop"] = f"no target from the controller in {a.first_target_s:g} s"
                if lift is not None and lift.phase == "waiting":
                    lift.cancel.set()
                print(f"{time.strftime('%H:%M:%S')} STOP: {row['stop']} (the lift cancelled)", flush=True)
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
            lift.close()
        if video is not None:
            time.sleep(1.0)                                      # a second of the hold after the end
        vid = None
        if tracker is not None:
            tracker.stop()
            tracker = None
        if video is not None:
            vid = video.close()
        tips_end = None
        if tips is not None:
            tips.stop()
            cm = np.array(tips.compute_ms, float)
            tips_end = {"files": tip_files, "records": dict(outlets.count), "outlet_error": outlets.error,
                        "estimates": int(len(cm)), "estimator": "thread" if a.tips_thread else "process",
                        "dropped": None if a.tips_thread else (tips.stats or {}).get("dropped"),
                        "compute_ms": np.round(np.percentile(cm, [50, 95, 100]), 2).tolist() if len(cm) else None}
            outlets.close()
            print(f"fingertips: {tips_end['estimates']} estimates, {tips_end['compute_ms']} ms (median, p95, max); "
                  f"records {tips_end['records']}", flush=True)
        diag = None
        if station is not None and station.live and (station.error or station.errors):
            # after a station error, once (owner, 2026-11-16: "we dont need to poll this all the time, it's just a
            # useful diagnostic"): did the servos reboot (torque_enable 0), and what voltage do they read now
            time.sleep(1.5)                                      # the stream's lease lapses first
            diag = station.diagnostics()
            print(f"servo diagnostics after the station error: volts "
                  f"{ {k: v / 10.0 for k, v in (diag.get('voltage') or {}).items() if v is not None} }, rebooted "
                  f"{diag.get('suspected_resets')}, no answer {diag.get('no_answer')}", flush=True)
        log.write(json.dumps({"kind": "end", "states": seq, "last_target": last_seq, "status": status, "video": vid,
                              "servo_diagnostics": diag, "tips": tips_end,
                              "frames_posted": None if station is None else station.n_frames,
                              "frames_with_joints": None if station is None else station.n_read,
                              "frame_errors": None if station is None else station.errors,
                              "stream_restarts": None if station is None else station.n_restarts,
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


def cmd_replay(a):
    """A recorded trial (this script's corvus record) sent again to a controller state by state, offline: the same
    sequence numbers, times, joints and object poses, and StateMsg.cop from the fingertips' estimates on the splitter's
    log of that trial (tip_contacts.MuxLog; the rest over --rest-s before t = 0, the record's first joints as its pose),
    or NaN with --no-tips. Each state waits for its answer. Writes OUT_DIR/<stamp>_<task>_replay_corvus.jsonl in the
    record's format. Nothing moves; no station, cameras or arm."""
    import tip_contacts as TC
    lines = [json.loads(x) for x in Path(a.corvus).read_text().splitlines() if x.strip()]
    head, rows = lines[0], [x for x in lines[1:] if "seq" in x]
    t0 = head["t0_ms"] / 1000.0
    tips = rest = None
    if a.tips:
        if not (a.mux and a.plan):
            sys.exit("replay with the fingertips needs --mux and --plan (or --no-tips)")
        feed = TC.MuxLog(a.mux, t0 - a.rest_s - TC.KEEP_S - 1.0, rows[-1]["t_ms"] / 1000.0 + 1.0)
        tips = TC.TipContacts(feed, TC.load_plan_d6(a.plan), gp=not a.no_tips_gp, own_pose=not a.no_own_pose)
        j0 = next((np.asarray(r["joints"], float).reshape(3, 3) for r in rows if r.get("joints")), None)
        rest = tips.rest(t0 + a.rest_end, a.rest_s, refit=not a.no_tips_refit, joints_servo=j0)
    ctrl = (a.controller.rsplit(":", 1)[0], int(a.controller.rsplit(":", 1)[1]))
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", a.listen))
    rx.settimeout(a.wait_s)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}_{head['task']}_replay_corvus.jsonl"
    est_ms = []
    with open(path, "w") as log:
        log.write(json.dumps({**head, "kind": "header", "replay_of": a.corvus, "mux": a.mux, "plan": a.plan,
                              "tips": None if tips is None else {"rest": rest, "rest_s": a.rest_s, "rest_end_s": a.rest_end,
                                                                 "gp": not a.no_tips_gp, "own_pose": not a.no_own_pose,
                                                                 "refit": not a.no_tips_refit}}) + "\n")
        last, n_ans = -1, 0
        for r in rows:
            st = StateMsg(seq=r["seq"], t_ms=r["t_ms"], last_target_seq=max(last, 0), status=r.get("status", 0))
            j = None
            if r.get("joints"):
                j = np.asarray(r["joints"], float)
                st.joints, st.t_joints_ms = j, r["t_joints_ms"] if r.get("t_joints_ms") is not None else float("nan")
            o = r.get("obj")
            if o:
                st.t_capture_ms, st.p_PO, st.q_PO = o["t_capture_ms"], np.asarray(o["p_PO"]), np.asarray(o["q_PO"])
                st.obj_flags, st.n_tags = o.get("flags", 0), o.get("n_tags", 0)
            trec = None
            if tips is not None:
                tc = time.perf_counter()
                est = tips.estimate(r["t_ms"] / 1000.0, None if j is None else j.reshape(3, 3))
                st.cop, trec = tips.to_P(est, None if j is None else j.reshape(3, 3))
                est_ms.append(1e3 * (time.perf_counter() - tc))
            tg = None
            for _try in range(3):                     # resent until answered (the server may still be building)
                tx.sendto(st.pack(), ctrl)
                try:
                    while True:
                        m = TargetMsg.unpack(rx.recv(256))
                        if m is not None and m.state_seq == r["seq"]:
                            tg = m
                            break
                except socket.timeout:
                    continue
                break
            row = {"seq": r["seq"], "t_ms": r["t_ms"], "t": r.get("t"), "joints": r.get("joints"), "obj": o,
                   "tips": trec, "status": r.get("status")}
            if tg is not None:
                n_ans += 1
                last = tg.seq
                row.update(target=tg.seq, target_sim_deg=np.round(tg.targets, 4).tolist(), stop=bool(tg.flags & STOP))
            log.write(json.dumps(row) + "\n")
            if tg is None:
                print(f"no answer to state {r['seq']}: stopping the replay", flush=True)
                break
            if tg.flags & STOP:
                break
        log.write(json.dumps({"kind": "end", "states": len(rows), "answered": n_ans,
                              "estimate_ms": None if not est_ms else np.round(np.percentile(est_ms, [50, 95, 100]), 2).tolist()})
                  + "\n")
    print(f"replay: {n_ans} of {len(rows)} states answered -> {path}", flush=True)
    return 0


def cmd_return(a):
    """Down by --lift-mm at 10 mm/s, stopped by the HEX-E's force change past --stop-N or a protective stop."""
    import rtde_control
    import rtde_receive
    ft = None if a.ur_ft else open_ft(True)
    r = rtde_receive.RTDEReceiveInterface(ROBOT, 125.0)
    c = rtde_control.RTDEControlInterface(ROBOT)
    src = "UR TCP force" if a.ur_ft else "HEX-E"

    def latest():
        return (time.time(), r.getActualTCPForce()) if a.ur_ft else ft.ring.latest()
    try:
        c.setTcp([0.0] * 6)
        time.sleep(0.6)
        if a.ur_ft:
            F0 = np.mean([np.asarray(r.getActualTCPForce(), float)[:3] for _ in range(50) if time.sleep(0.008) is None],
                         axis=0)
        else:
            _, v0 = ft.ring.since(time.time() - 0.4)
            F0 = np.asarray(v0, float)[:, :3].mean(axis=0)
        p0 = np.array(r.getActualTCPPose())
        tgt = p0.copy()
        tgt[2] -= a.lift_mm / 1000.0
        c.moveL(tgt.tolist(), 0.010, 0.1, True)
        t0, fmax, why = time.time(), 0.0, None
        while True:
            t, v = latest()
            if t is None or time.time() - t > 0.1:
                why = f"no {src} sample for 100 ms"
            else:
                fmax = max(fmax, float(np.linalg.norm(np.asarray(v, float)[:3] - F0)))
                if fmax > a.stop_n:
                    why = f"{src} force changed {fmax:.2f} N (limit {a.stop_n:g})"
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
        print(f"return: {'STOPPED, ' + why if why else 'down'}; moved {1000 * (p1[2] - p0[2]):.2f} mm in z, {src} "
              f"changed at most {fmax:.2f} N", flush=True)
    finally:
        try:
            c.stopScript()
            c.disconnect()
        finally:
            r.disconnect()
    return 0


def cmd_plan(a):
    """POST the plan inline to the station's plans/load; print what the station loaded."""
    raw = json.loads(Path(a.plan).read_text())
    raw.setdefault("meta", {})["plan_file"] = Path(a.plan).name
    st = Station(a.station, a.token_file, True)
    c = st._conn()
    c.timeout = 10.0
    res = st._post(c, "/api/v1/plans/load", {"plan": {k: raw[k] for k in ("design", "mounts_palm_mm", "poses", "meta")}})
    s = json.loads((c.request("GET", "/api/v1/state"), c.getresponse().read())[1])
    pl = s.get("plan") or {}
    print(json.dumps({"loaded": res if len(json.dumps(res)) < 2000 else sorted(res),
                      "station_plan": {k: pl.get(k) for k in ("design", "mounts_palm_mm", "violations") if k in pl},
                      "homed": s.get("homed")}, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("plan")
    pp.add_argument("--plan", required=True)
    pp.add_argument("--station", default="http://10.99.99.2:8765")
    pp.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
    pc = sub.add_parser("check")
    pc.add_argument("--object", choices=("cylinder", "sphere"), required=True)
    pc.add_argument("--map", required=True, help="the tag map of the object's block in the object's frame")
    pc.add_argument("--station", default=None)
    pc.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
    pc.add_argument("--seconds", type=float, default=5.0)
    pc.add_argument("--nominal", default=None, help="the grasp's start pose (hom_server nominal --out)")
    pt = sub.add_parser("trial")
    pt.add_argument("--task", required=True, choices=("pinpush", "press", "roll", "twist"))
    pt.add_argument("--controller", required=True, help="the HOM server's HOST:PORT")
    pt.add_argument("--object", choices=("cylinder", "sphere"), required=True)
    pt.add_argument("--map", default=None, help="the tag map of the object's block in the object's frame")
    pt.add_argument("--listen", type=int, default=47001)
    pt.add_argument("--hz", type=float, default=50.0)
    pt.add_argument("--miss", type=int, default=3)
    pt.add_argument("--first-target-s", type=float, default=0.5,
                    help="s without any target from the controller after which the trial stops and its lift is "
                         "cancelled (0: never)")
    pt.add_argument("--wait-ms", type=float, default=8.0)
    pt.add_argument("--seconds", type=float, default=9.0, help="the link's length; the controller's STOP ends it first")
    pt.add_argument("--settle-s", type=float, default=2.0, help="the cameras' start before t = 0")
    pt.add_argument("--station", default=None)
    pt.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
    pt.add_argument("--live", action="store_true")
    pt.add_argument("--no-tracker", action="store_true")
    pt.add_argument("--no-video", dest="video", action="store_false",
                    help="do not record the cameras (by default each camera's frames go to OUT, 960 x 540)")
    pt.add_argument("--video-dir", default=None,
                    help="the videos into DIR/<stamp>_trial_<label>_<serial>.mp4 in place of OUT/<stamp>_<task>_video_..., "
                         "beside hom_setup.py's --video-dir recordings of the same session (2026-11-16)")
    pt.add_argument("--label", default=None, help="the videos' name with --video-dir (the plan's stem; default the task)")
    pt.add_argument("--arm", action="store_true", help="move the arm (the lift); without it the lift is only logged")
    pt.add_argument("--ft", action="store_true", help="stream the HEX-E (needed with --arm for the force stop)")
    pt.add_argument("--ur-ft", action="store_true",
                    help="the UR's own TCP force as the lift's force stop, in place of the HEX-E (2026-11-16)")
    pt.add_argument("--lift-at", type=float, default=1.6)
    pt.add_argument("--lift-mm", type=float, default=40.0)
    pt.add_argument("--lift-v", type=float, default=0.06, help="m/s")
    pt.add_argument("--lift-a", type=float, default=0.3, help="m/s^2")
    pt.add_argument("--stop-n", type=float, default=5.0, help="N of HEX-E force change that stops the lift")
    pt.add_argument("--out-dir", required=True)
    pt.add_argument("--max-drift-mm", type=float, default=20.0,
                    help="stop the trial (the station holds the last goals) when the tracked object moves this far from "
                         "its first pose in P; 0 turns it off (owner, 2026-11-16: keep the sphere in the usable "
                         "workspace)")
    pt.add_argument("--plan", default=None, help="the trial's plan: its mounts (meta.hom.d6_P_mm) place the fingertips' "
                                                 "contacts in P (needed with the fingertips)")
    pt.add_argument("--no-tips", dest="tips", action="store_false",
                    help="without the fingertip boards (by default a trial refuses to start unless all three stream)")
    pt.add_argument("--rest-s", type=float, default=1.0, help="s of the boards' rest before t = 0, at the open pose")
    pt.add_argument("--no-tips-gp", action="store_true", help="the boards' physical model alone, without the UR5 board GP")
    pt.add_argument("--no-own-pose", action="store_true", help="without the boards' own-pose term")
    pt.add_argument("--no-tips-refit", action="store_true",
                    help="keep each tip's saved model, re-zeroed at the rest (no new fit competing with it)")
    pt.add_argument("--tips-thread", action="store_true",
                    help="the estimator in a thread of this process, as in the trials of 2026-11-17 09:12-09:29 (the "
                         "default is a process of its own, tip_contacts.TipProcess)")
    pt.add_argument("--tips-links", default=None,
                    help="test only: the boards' pseudo-terminals in this folder (tip_contacts.py play), without the "
                         "splitter's check")
    pr = sub.add_parser("return")
    pr.add_argument("--lift-mm", type=float, default=40.0)
    pr.add_argument("--stop-n", type=float, default=3.0)
    pr.add_argument("--ur-ft", action="store_true", help="the UR's own TCP force as the force stop")
    pl = sub.add_parser("replay", help="a recorded trial sent again to a controller, offline")
    pl.add_argument("--corvus", required=True, help="the trial's record (<stamp>_<task>_corvus.jsonl)")
    pl.add_argument("--controller", required=True, help="the HOM server's HOST:PORT")
    pl.add_argument("--listen", type=int, default=47001)
    pl.add_argument("--out-dir", required=True)
    pl.add_argument("--mux", default=None, help="the splitter's .bin over the trial (reskin5x_mux.py)")
    pl.add_argument("--plan", default=None, help="the trial's plan (the mounts in P)")
    pl.add_argument("--no-tips", dest="tips", action="store_false", help="StateMsg.cop NaN, as on 2026-11-16")
    pl.add_argument("--rest-s", type=float, default=1.0)
    pl.add_argument("--rest-end", type=float, default=-0.2, help="s of the trial's clock at which the rest ends")
    pl.add_argument("--no-tips-gp", action="store_true")
    pl.add_argument("--no-own-pose", action="store_true")
    pl.add_argument("--no-tips-refit", action="store_true")
    pl.add_argument("--wait-s", type=float, default=60.0, help="longest wait for an answer (the server's build)")
    a = ap.parse_args()
    if a.cmd == "replay":
        return cmd_replay(a)
    if a.cmd == "trial":
        if a.arm and not (a.ft or a.ur_ft):
            sys.exit("--arm needs --ft (or --ur-ft): the lift is stopped by a force change")
        if not a.no_tracker and not a.map:
            sys.exit("--map is needed with the tracker")
        return cmd_trial(a)
    if a.cmd == "plan":
        return cmd_plan(a)
    return cmd_check(a) if a.cmd == "check" else cmd_return(a)


if __name__ == "__main__":
    sys.exit(main())
