#!/usr/bin/env python3
"""Persistent workstation companion for automatic web-UI AprilTag capture.

The RealSense is attached to the workstation while ``manta-hand-web`` runs on the CB1.
This small service bridges that physical split: the CB1 asks it to arm a tracker for an
exact run id, waits for the static reference latch, performs the motion, and asks it to
stop. One service can cover hundreds of trials; no per-trial terminal command is needed.

Typical workstation launch (use the same token as the CB1 control service)::

    MANTA_TOKEN=... ~/miniconda3/bin/python scripts/real_v1_tracker_service.py

The CB1 control service then uses ``--tracker-url http://10.99.99.50:8770``. Bind this only
on the isolated hand network. ``--token``/``--tracker-token`` add a shared control secret.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import real_v1_tag_tracker as tagtool  # noqa: E402


class TrackerProcess:
    """Own at most one camera subprocess and expose its latch/finalization boundary."""

    def __init__(self, *, station_url: str, station_token: str, logs_dir: Path,
                 seconds: float, arm_timeout: float, extra: list[str],
                 post_roll: float = 0.0):
        self.station_url = station_url.rstrip("/")
        self.station_token = station_token
        self.logs_dir = logs_dir
        self.seconds = seconds
        self.arm_timeout = arm_timeout
        self.extra = extra
        self.post_roll = float(post_roll)
        self.lock = threading.RLock()
        self.proc: subprocess.Popen | None = None
        self.run_id = ""
        self.csv: Path | None = None
        self.lines: list[str] = []
        self.output: queue.Queue[str] = queue.Queue()
        self.reader: threading.Thread | None = None
        self.armed = False
        self.error: str | None = None
        self.started_at: float | None = None

    def _read_output(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            with self.lock:
                self.lines.append(line)
            self.output.put(line)

    def _write_stdout(self) -> None:
        if self.csv is not None:
            path = self.csv.with_name(self.csv.stem + "_stdout.txt")
            path.write_text("".join(self.lines), encoding="utf-8")

    def _reap_if_done(self) -> None:
        if self.proc is not None and self.proc.poll() is not None:
            self._write_stdout()
            if self.proc.returncode and self.error is None:
                self.error = f"tracker exited with status {self.proc.returncode}"
            self.proc = None
            self.armed = False

    def status(self) -> dict:
        with self.lock:
            self._reap_if_done()
            summary = None
            if self.csv is not None:
                sidecar = self.csv.with_name(self.csv.stem + "_SUMMARY.json")
                if sidecar.exists():
                    try:
                        summary = json.loads(sidecar.read_text(encoding="utf-8")).get("summary")
                    except (OSError, json.JSONDecodeError):
                        pass
            return {"running": self.proc is not None, "armed": self.armed,
                    "run_id": self.run_id, "trace": str(self.csv) if self.csv else None,
                    "started_at": self.started_at, "error": self.error,
                    "summary": summary}

    def start(self, run_id: str) -> dict:
        if not run_id or "/" in run_id or "\\" in run_id:
            raise ValueError("run_id must be a non-empty filename-safe station run id")
        with self.lock:
            self._reap_if_done()
            if self.proc is not None:
                if self.run_id == run_id and self.armed:
                    return self.status()
                raise RuntimeError(f"camera is already recording run {self.run_id}")
            exe = tagtool.find_interpreter()
            if not exe:
                raise RuntimeError("no interpreter on this workstation has the RealSense/tag stack")
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self.run_id = run_id
            self.csv = self.logs_dir / f"{run_id}_track.csv"
            self.lines = []
            self.output = queue.Queue()
            self.armed = False
            self.error = None
            self.started_at = time.time()
            cmd = [exe, str(ROOT / "scripts/real_v1_tag_tracker.py"),
                   "--seconds", str(self.seconds), "--out", str(self.csv), "--quiet",
                   "--push", "--push-url", self.station_url, "--run-id", run_id]
            if self.station_token:
                cmd += ["--push-token", self.station_token]
            cmd += self.extra
            self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
            self.reader = threading.Thread(target=self._read_output,
                                           name="tag-tracker-output", daemon=True)
            self.reader.start()

        deadline = time.monotonic() + self.arm_timeout
        recording = pushed = False
        while time.monotonic() < deadline:
            with self.lock:
                if self.proc is None or self.proc.poll() is not None:
                    detail = "".join(self.lines[-8:]).strip()
                    self._reap_if_done()
                    raise RuntimeError(f"tracker exited before arming: {detail or self.error}")
            try:
                line = self.output.get(timeout=max(0.01, min(0.25,
                                                            deadline - time.monotonic())))
            except queue.Empty:
                continue
            if "recording" in line:
                recording = True
            if "station push confirmed" in line:
                pushed = True
            if recording and pushed:
                with self.lock:
                    self.armed = True
                    return self.status()
        self.stop(run_id)
        missing = "id6/cylinder sample" if not recording else "station sample delivery"
        raise TimeoutError(f"tracker did not confirm {missing} within {self.arm_timeout:.0f}s")

    def stop(self, run_id: str = "") -> dict:
        """End the recording -- after a dwell, so the trace contains a pose being HELD.

        The station stops the tracker the instant its last ramp finishes, which made every
        trace end at the end of the motion. `cos_hold` then averaged the last second of the
        TURN, and a hand that reached vertical and let go a moment later scored exactly like
        one that reached vertical and kept it -- the failure the hold window exists to catch.
        The servos hold their last commanded position after the plan runs, so the dwell costs
        nothing but the seconds and observes the only thing the metric is about.
        """
        with self.lock:
            self._reap_if_done()
            if self.proc is None:
                return self.status()
            if run_id and run_id != self.run_id:
                raise ValueError(f"active tracker belongs to {self.run_id}, not {run_id}")
            proc, armed = self.proc, self.armed
        # Only after a run that actually armed: an arm-timeout abort must fail fast, not sit
        # there taping a bench nobody is watching.
        if armed and self.post_roll > 0:
            time.sleep(self.post_roll)
        with self.lock:
            if self.proc is not proc:
                return self.status()
            proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            with self.lock:
                self.error = "tracker did not finalize within 30s and was killed"
        if self.reader is not None:
            self.reader.join(timeout=2)
        with self.lock:
            self._reap_if_done()
            return self.status()

    def close(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


class TrackerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, tracker: TrackerProcess, token: str):
        self.tracker = tracker
        self.token = token
        super().__init__(address, TrackerHandler)


class TrackerHandler(BaseHTTPRequestHandler):
    server: TrackerServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"tracker-service: {fmt % args}", flush=True)

    def do_GET(self):
        if urlparse(self.path).path.rstrip("/") == "/status":
            return self._json(HTTPStatus.OK, self.server.tracker.status())
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self):
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}"
            body = json.loads(raw)
            if self.server.token and self.headers.get("X-Tracker-Token", "") != self.server.token:
                return self._json(HTTPStatus.UNAUTHORIZED, {"error": "invalid tracker token"})
            path = urlparse(self.path).path.rstrip("/")
            if path == "/start":
                return self._json(HTTPStatus.OK,
                                  self.server.tracker.start(str(body.get("run_id", ""))))
            if path == "/stop":
                return self._json(HTTPStatus.OK,
                                  self.server.tracker.stop(str(body.get("run_id", ""))))
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (ValueError, RuntimeError, TimeoutError) as exc:
            self._json(HTTPStatus.CONFLICT,
                       {"error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                       {"error": f"{type(exc).__name__}: {exc}"})

    def _json(self, status: HTTPStatus, value: dict) -> None:
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--station-url", default="http://10.99.99.2:8765/api/v1")
    ap.add_argument("--station-token", default=os.environ.get("MANTA_TOKEN", ""))
    ap.add_argument("--token", default=os.environ.get("MANTA_TRACKER_TOKEN", ""),
                    help="optional X-Tracker-Token required from the CB1")
    ap.add_argument("--logs-dir", type=Path, default=ROOT / "logs/tracker")
    ap.add_argument("--seconds", type=float, default=300.0,
                    help="safety cap; the CB1 normally stops each recording earlier")
    ap.add_argument("--arm-timeout", type=float, default=20.0)
    ap.add_argument("--post-roll", type=float, default=2.0,
                    help="keep recording this long after the station says the trajectory is "
                         "done, so the trace contains the pose being HELD rather than ending "
                         "on the last instant of the motion (0 restores the old behaviour)")
    ap.add_argument("--tracker-arg", action="append", default=[],
                    help="repeatable argument passed to real_v1_tag_tracker.py")
    args = ap.parse_args()
    tracker = TrackerProcess(station_url=args.station_url, station_token=args.station_token,
                             logs_dir=args.logs_dir, seconds=args.seconds,
                             arm_timeout=args.arm_timeout, extra=args.tracker_arg,
                             post_roll=args.post_roll)
    server = TrackerServer((args.host, args.port), tracker, args.token)
    print(f"AprilTag tracker service: http://{args.host}:{args.port} -> {args.station_url}",
          flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        tracker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
