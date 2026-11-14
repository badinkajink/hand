"""corvus's half of the controller link (sr2hand.control.link): state up, targets down, hold and stop.

Every control period (--hz, 50 by default) corvus sends a StateMsg to the controller (--controller HOST:PORT),
waits up to --wait-ms (8) for the answer to it, and applies the newest TargetMsg that has arrived on --listen. The state carries:
  joints   the station's servo telemetry (GET /api/v1/state, polled in a thread; read-only), when --station is given
           and answers; else NaN. While a policy stream runs the station does not read the servos (runtime.py,
           write-only by design), so these go stale: the controller must use its own targets or a station change
           must read between frames;
  object   the tracker's newest T_PO, run in this process (sr2hand.tracking.live with the arm's TCP pose);
  cop      NaN for now (the fingertip boards' wrench_viewer serves them over a websocket; not wired yet).
A target with a newer sequence number is applied once: logged in dry run (the default), or in --live mode sent to the
station as POST /api/v1/stream/frame inside a stream lease (POST /stream/start with timeout_s 0.25 at the first
target). When no new target has come for --miss periods (3), the link stops: no more frames, /stream/end, status
STOPPED; the station holds the last goal. A target with the STOP flag does the same.

--live moves the hand. It needs the owner present and their go, a homed station with torque on, and a plan loaded
(the station refuses a stream otherwise). Without --live nothing is sent to the station.

Writes one JSON line per period to --log: the state's times and sequence, the target applied (its sequence, its age
when applied, the round trip from the state it answers), and the object's age.

    /usr/bin/python3 hand_link.py --controller 127.0.0.1:47002 [--listen 47001] [--hz 50] [--seconds 30]
        [--station http://10.99.99.2:8765] [--live] [--no-tracker] --log OUT.jsonl
"""
import argparse
import http.client
import json
import os
import select
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from sr2hand.control.link import DRY_RUN, STOP, STOPPED, STREAMING, StateMsg, TargetMsg  # noqa: E402

CAL = ROOT / "docs/experiments/20261113-camera_arm_calibration/data/fits"
ap = argparse.ArgumentParser()
ap.add_argument("--controller", required=True)
ap.add_argument("--listen", type=int, default=47001)
ap.add_argument("--hz", type=float, default=50.0)
ap.add_argument("--miss", type=int, default=3)
ap.add_argument("--wait-ms", type=float, default=8.0, help="wait this long for the answer to the state just sent")
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--station", default=None)
ap.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
ap.add_argument("--live", action="store_true")
ap.add_argument("--no-tracker", action="store_true")
ap.add_argument("--log", required=True)
a = ap.parse_args()
P = 1.0 / a.hz
ctrl = (a.controller.rsplit(":", 1)[0], int(a.controller.rsplit(":", 1)[1]))


class Station:
    """Keep-alive HTTP to the CB1's station: state polling (read-only) and the policy stream (--live only)."""

    def __init__(self, url, token_file, live):
        u = urlparse(url)
        self.host, self.port, self.live = u.hostname, u.port or 80, live
        self.token = Path(token_file).read_text().strip() if live else None
        self.stream_token = None
        self.joints, self.t_joints, self.error = None, float("nan"), None
        self.stop_evt = threading.Event()

    def _conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=1.0)

    def poll(self, hz=20.0):
        c = self._conn()
        while not self.stop_evt.is_set():
            try:
                c.request("GET", "/api/v1/state")
                s = json.loads(c.getresponse().read())
                sv = s.get("telemetry", {}).get("servos", {})
                if sv:
                    self.joints = np.array([[sv[str(k)][j] for j in ("aa", "fe1", "fe2")] for k in range(3)], float)
                    self.t_joints = time.time() * 1000.0
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


station = None
if a.station:
    station = Station(a.station, a.token_file, a.live)
    threading.Thread(target=station.poll, daemon=True).start()
elif a.live:
    sys.exit("--live needs --station")

tracker, latest_obj = None, {"msg": None}
if not a.no_tracker:
    from sr2hand.robot.tool_chain import SR2Hand
    from sr2hand.tracking import live
    arm = live.ArmPose()
    arm.start()
    tracker = live.Tracker(live.load_cameras(CAL / "20261113-lens_board_k3_arm.json",
                                             CAL / "20261113-handeye_tcp_fitted_lens_moved836.json"),
                           live.load_map(CAL / "20261113-hex_block_map.json"), arm=arm, T_EP=SR2Hand.load().T_E_P(),
                           on_sample=lambda s: latest_obj.__setitem__("msg", s.msg)).start()

rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rx.bind(("0.0.0.0", a.listen))
rx.setblocking(False)
tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
log = open(a.log, "w")
sent = {}                                              # state seq -> send time (ms), for the round trip
status = DRY_RUN if not a.live else 0
last_seq, t_last_new, armed = -1, None, False
seq, t0 = 0, time.time()
next_t = time.perf_counter()
print(f"{time.strftime('%H:%M:%S')} link: {1000 * P:.0f} ms period, controller {ctrl}, listen {a.listen}, "
      f"{'LIVE' if a.live else 'dry run'}, station {a.station}, tracker {'on' if tracker else 'off'}", flush=True)
try:
    while time.time() - t0 < a.seconds:
        now = time.time() * 1000.0
        st = StateMsg(seq=seq, t_ms=now, last_target_seq=max(last_seq, 0), status=status)
        if station is not None and station.joints is not None:
            st.joints, st.t_joints_ms = station.joints.ravel(), station.t_joints
        m = latest_obj["msg"]
        if m is not None:
            st.t_capture_ms, st.p_PO, st.q_PO, st.obj_flags, st.n_tags = m.t_capture_ms, m.p_PO, m.q_PO, m.flags, m.n_tags
        tx.sendto(st.pack(), ctrl)
        sent[seq] = now
        sent.pop(seq - 500, None)
        newest, deadline = None, time.perf_counter() + a.wait_ms / 1000.0
        while True:                                    # the newest target; wait for the answer to this state
            r_, _, _ = select.select([rx], [], [], max(0.0, deadline - time.perf_counter()))
            if not r_:
                break
            try:
                t = TargetMsg.unpack(rx.recv(256))
            except BlockingIOError:
                continue
            if t is not None and t.seq > last_seq and (newest is None or t.seq > newest.seq):
                newest = t
            if newest is not None and newest.state_seq == seq:
                break
        row = dict(seq=seq, t=round(now, 3), obj_age=None if m is None else round(now - m.t_capture_ms, 2))
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
                row.update(target=newest.seq, target_age=round(now2 - newest.t_ms, 3),
                           rtt=round(now2 - sent[newest.state_seq], 3) if newest.state_seq in sent else None)
        if armed and not status & STOPPED and now2 - t_last_new > a.miss * 1000.0 * P:
            status |= STOPPED
            row["stop"] = f"no target for {a.miss} periods"
        if status & STOPPED and status & STREAMING:
            station.end_stream()
            status &= ~STREAMING
        log.write(json.dumps(row) + "\n")
        seq += 1
        next_t += P
        dt = next_t - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        else:
            next_t = time.perf_counter()                # overran: restart the schedule
finally:
    if station is not None and station.stream_token:
        station.end_stream()
    if tracker is not None:
        tracker.stop()
    log.close()
    print(f"{time.strftime('%H:%M:%S')} link done: {seq} states, last target {last_seq}, status {status}", flush=True)
    os._exit(0)
