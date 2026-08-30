"""Safety-oriented runtime shared by the CB1 HTTP service and offline tests.

The runtime deliberately owns all hardware access.  Web request threads never touch a
serial controller directly: they enqueue one bounded operation, while telemetry reads use
the backend's bus lock.  A cached status document is therefore cheap to serve at any UI
refresh rate without turning browser refreshes into servo packets.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .kinematics import STEPS_PER_MM
from .plan import FINGER_ID, JOINT_SIGN, SIM_JOINT_TO_SERVO, HandPlan, Pose, servo_deg
from .servos import DEFAULT_JOINT_SPEED, FINGER_JOINTS

FINGER_ORDER = ("thumb", "index", "middle")
JOINT_ORDER = ("yaw", "mcp", "pip")
HOME_CONFIRMATION = "HOME ALL AXES"


class RuntimeErrorState(RuntimeError):
    """A command is unsafe or invalid in the runtime's current state."""


class HardwareBackend(Protocol):
    kind: str

    def home_all(self) -> None: ...
    def apply_mounts(self, plan: HandPlan) -> None: ...
    def write_joints(self, pose: dict[int, dict[str, float]], speed: int | None = None) -> None: ...
    def read_telemetry(self, include_servos: bool = True) -> dict: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class RealHardwareBackend:
    """The two physical serial links on the CB1, opened only on construction."""

    kind = "real"

    def __init__(self, stepper_port: str = "/dev/ttyACM0", servo_port: str = "/dev/ttyUSB0"):
        from .driver import MantaHandDriver
        from .hand import Hand
        from .servos import ServoBus

        self.driver = MantaHandDriver(stepper_port)
        try:
            self.servos = ServoBus(servo_port)
            self.hand = Hand(self.driver, self.servos)
        except Exception:
            self.driver.close()
            raise

    def home_all(self) -> None:
        self.hand.home_all()

    def apply_mounts(self, plan: HandPlan) -> None:
        plan.apply_mounts(self.hand, home=False)
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if not any(status.moving for status in self.driver.get_all_status()[:6]):
                return
            time.sleep(0.1)
        self.driver.stop_all()
        raise TimeoutError("gantry morphology move did not finish within 60 seconds")

    def write_joints(self, pose: dict[int, dict[str, float]], speed: int | None = None) -> None:
        self.hand.set_joints_fast(pose, speed=speed)

    def read_telemetry(self, include_servos: bool = True) -> dict:
        step = self.driver.get_all_status()[:6]
        out = {
            "steppers": [
                {
                    "id": i,
                    "position_mm": s.position / STEPS_PER_MM[i],
                    "target_mm": s.target / STEPS_PER_MM[i],
                    "moving": s.moving,
                    "enabled": s.enabled,
                    "homing_result": s.homing_result,
                }
                for i, s in enumerate(step)
            ]
        }
        if include_servos:
            out["servos"] = self.servos.sync_read_joint_positions()
        return out

    def stop(self) -> None:
        # STOPALL decelerates the gantries.  There is no equivalent broadcast
        # "stop" for position servos; cancelling the writer holds the last goal.
        self.driver.stop_all()

    def close(self) -> None:
        self.driver.close()


class MockHardwareBackend:
    """Deterministic no-hardware backend used by the UI demo and tests."""

    kind = "mock"

    def __init__(self):
        self.homed = False
        self.mounts: dict[str, tuple[float, float]] = {}
        self.joints = {fid: {name: 0.0 for name in ("aa", "fe1", "fe2")}
                       for fid in range(3)}
        self.writes = 0

    def home_all(self) -> None:
        time.sleep(0.02)
        self.homed = True
        self.joints = {fid: {name: 0.0 for name in ("aa", "fe1", "fe2")}
                       for fid in range(3)}

    def apply_mounts(self, plan: HandPlan) -> None:
        self.mounts = dict(plan.mounts_palm_mm)

    def write_joints(self, pose: dict[int, dict[str, float]], speed: int | None = None) -> None:
        for fid, values in pose.items():
            self.joints[fid].update(values)
        self.writes += 1

    def read_telemetry(self, include_servos: bool = True) -> dict:
        steppers = []
        for i in range(6):
            steppers.append({"id": i, "position_mm": 0.0, "target_mm": 0.0,
                             "moving": False, "enabled": True, "homing_result": 2 if self.homed else 0})
        out = {"steppers": steppers}
        if include_servos:
            out["servos"] = json.loads(json.dumps(self.joints))
        return out

    def stop(self) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass
class Event:
    seq: int
    timestamp: str
    level: str
    message: str
    data: dict = field(default_factory=dict)


class RunLog:
    """Append-only JSONL with a small sidecar summary and manual score."""

    def __init__(self, root: Path, plan: HandPlan, operation: str, settings: dict):
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}-{plan.design}-{uuid.uuid4().hex[:6]}"
        self.path = root / f"{self.run_id}.jsonl"
        self.summary_path = root / f"{self.run_id}_SUMMARY.json"
        self._lock = threading.Lock()
        self.summary = {
            "schema_version": 1,
            "run_id": self.run_id,
            "design": plan.design,
            "operation": operation,
            "started_at": _utc_now(),
            "finished_at": None,
            "status": "running",
            "settings": settings,
            "plan_meta": plan.meta,
            "samples": {"command": 0, "telemetry": 0, "event": 0},
            "manual_score": None,
        }
        self._write_summary()

    def append(self, kind: str, payload: dict) -> None:
        row = {"schema_version": 1, "kind": kind, "wall_time": _utc_now(),
               "monotonic_s": time.monotonic(), **payload}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
            if kind in self.summary["samples"]:
                self.summary["samples"][kind] += 1

    def finish(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self.summary["finished_at"] = _utc_now()
            self.summary["status"] = status
            if error:
                self.summary["error"] = error
            self._write_summary_unlocked()

    def score(self, success: bool, reorientation_deg: float | None, notes: str) -> None:
        with self._lock:
            self.summary["manual_score"] = {
                "success": bool(success), "reorientation_deg": reorientation_deg,
                "notes": notes, "scored_at": _utc_now(),
            }
            self._write_summary_unlocked()

    def _write_summary(self) -> None:
        with self._lock:
            self._write_summary_unlocked()

    def _write_summary_unlocked(self) -> None:
        self.summary_path.write_text(json.dumps(self.summary, indent=2) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _zero_sim_pose() -> dict[str, dict[str, float]]:
    return {f: {j: 0.0 for j in JOINT_ORDER} for f in FINGER_ORDER}


class HandRuntime:
    """One state machine around the hand, safe to call from HTTP request threads."""

    def __init__(self, backend: HardwareBackend, *, logs_dir: str | Path = "logs/hardware",
                 telemetry_hz: float = 0.0, signs_checked: bool = False):
        if not 0.0 <= telemetry_hz <= 100.0:
            raise ValueError("telemetry_hz must be in [0, 100]")
        self.backend = backend
        self.logs_dir = Path(logs_dir)
        self.telemetry_hz = float(telemetry_hz)
        self.signs_checked = bool(signs_checked)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._shutdown = threading.Event()
        self._operation_thread: threading.Thread | None = None
        self._events: deque[Event] = deque(maxlen=500)
        self._event_seq = 0
        self._plan: HandPlan | None = None
        self._homed = False
        self._mounts_applied = False
        self._operation = "idle"
        self._current_pose: str | None = None
        self._last_command = _zero_sim_pose()
        self._last_error: str | None = None
        self._telemetry: dict = {"timestamp": None, "age_s": None, "servos": None,
                                "servo_timestamp": None, "servo_age_s": None,
                                "steppers": None, "error": None, "samples": 0,
                                "measured_hz": None, "servo_polling_suspended": False}
        self._active_log: RunLog | None = None
        self._logs: dict[str, RunLog] = {}
        self._stream_token: str | None = None
        self._stream_deadline = 0.0
        self._stream_timeout_s = 0.25
        self._event("info", f"runtime ready ({backend.kind})")
        self._telemetry_thread = threading.Thread(target=self._telemetry_loop,
                                                  name="manta-telemetry", daemon=True)
        self._telemetry_thread.start()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop,
                                                 name="manta-stream-watchdog", daemon=True)
        self._watchdog_thread.start()

    # ---- state and catalog --------------------------------------------------------------
    def state(self) -> dict:
        with self._lock:
            telem = dict(self._telemetry)
            if telem["timestamp"] is not None:
                telem["age_s"] = max(0.0, time.time() - telem["timestamp"])
            if telem["servo_timestamp"] is not None:
                telem["servo_age_s"] = max(0.0, time.time() - telem["servo_timestamp"])
            plan = None
            if self._plan is not None:
                plan = {
                    "design": self._plan.design,
                    "meta": self._plan.meta,
                    "mounts_palm_mm": self._plan.mounts_palm_mm,
                    "poses": [p.name for p in self._plan.poses],
                    "violations": [str(v) for v in self._plan.validate()],
                }
            return {
                "schema_version": 1,
                "backend": self.backend.kind,
                "homed": self._homed,
                "mounts_applied": self._mounts_applied,
                "operation": self._operation,
                "busy": self._operation != "idle",
                "current_pose": self._current_pose,
                "plan": plan,
                "signs_checked": self.signs_checked,
                "telemetry": telem,
                "last_error": self._last_error,
                "streaming": self._stream_token is not None,
                "capabilities": {
                    "servo_position": "sync-read when supported by installed rustypot",
                    "servo_load": "not enabled; uncalibrated proxy, not force/current",
                    "servo_current": False,
                    "object_pose": False,
                    "contact": False,
                    "closed_loop_rl": False,
                    "open_loop_buffered": True,
                    "policy_streaming": "experimental write-only",
                },
            }

    def events(self, after: int = 0) -> list[dict]:
        with self._lock:
            return [asdict(e) for e in self._events if e.seq > after]

    def load_plan(self, plan: HandPlan) -> dict:
        schema = plan.schema_errors()
        if schema:
            raise RuntimeErrorState("invalid plan schema:\n" + "\n".join(schema))
        bad = plan.validate()
        if bad:
            raise RuntimeErrorState("plan does not fit the measured hand:\n" +
                                    "\n".join(str(v) for v in bad))
        with self._lock:
            self._require_idle()
            self._plan = plan
            self._mounts_applied = False
            self._current_pose = None
            self._last_command = _zero_sim_pose()
            self._event("info", f"loaded plan {plan.design}", {"meta": plan.meta})
        return self.state()

    # ---- bounded operations -------------------------------------------------------------
    def home(self, confirmation: str, *, force: bool = False) -> None:
        if confirmation != HOME_CONFIRMATION:
            raise RuntimeErrorState(f"confirmation must be exactly {HOME_CONFIRMATION!r}")
        with self._lock:
            if self._homed and not force:
                raise RuntimeErrorState("already homed in this daemon session; pass force=true to re-home")
            self._start_operation("homing", self._do_home)

    def _do_home(self) -> None:
        self.backend.home_all()
        with self._lock:
            self._homed = True
            self._mounts_applied = False
            self._current_pose = "zero"
            self._last_command = _zero_sim_pose()

    def apply_morphology(self) -> None:
        with self._lock:
            self._require_homed_plan()
            self._start_operation("applying morphology", self._do_apply_morphology)

    def _do_apply_morphology(self) -> None:
        assert self._plan is not None
        self.backend.apply_mounts(self._plan)
        with self._lock:
            if not self._stop.is_set():
                self._mounts_applied = True
                self._current_pose = "zero"

    def move_to_pose(self, name: str, *, speed_ratio: float = 1.0,
                     rate_hz: float = 50.0, servo_speed: int = DEFAULT_JOINT_SPEED) -> None:
        self._validate_timing(speed_ratio, rate_hz)
        with self._lock:
            self._require_motion_ready()
            pose = self._pose(name)
            duration = pose.ramp_s / speed_ratio
            if duration <= 0:
                duration = 1.0 / speed_ratio  # open pose has no incoming ramp in exported plans
            self._start_operation(f"moving to {name}", self._do_move_pose,
                                  pose, duration, rate_hz, servo_speed)

    def _do_move_pose(self, pose: Pose, duration: float, rate_hz: float, speed: int) -> None:
        if not self._ramp(self._last_command, pose.joints, duration, rate_hz, speed, pose.name):
            return
        if pose.hold_s > 0 and not self._stop.wait(pose.hold_s):
            pass
        with self._lock:
            if not self._stop.is_set():
                self._current_pose = pose.name

    def run_reorientation(self, *, speed_ratio: float = 1.0, rate_hz: float = 50.0,
                          servo_speed: int = DEFAULT_JOINT_SPEED) -> str:
        self._validate_timing(speed_ratio, rate_hz)
        with self._lock:
            self._require_motion_ready()
            if self._current_pose != "grip":
                raise RuntimeErrorState("move to the morphology-specific 'grip' pose first")
            assert self._plan is not None
            log = RunLog(self.logs_dir, self._plan, "reorientation",
                         {"speed_ratio": speed_ratio, "rate_hz": rate_hz,
                          "servo_speed": servo_speed, "telemetry_hz": self.telemetry_hz,
                          "joint_signs": {
                              finger: {joint: JOINT_SIGN[(finger, joint)]
                                       for joint in JOINT_ORDER}
                              for finger in FINGER_ORDER
                          }})
            self._logs[log.run_id] = log
            self._active_log = log
            self._start_operation("reorienting", self._do_run_reorientation,
                                  speed_ratio, rate_hz, servo_speed, log)
            return log.run_id

    def _do_run_reorientation(self, speed_ratio: float, rate_hz: float,
                              speed: int, log: RunLog) -> None:
        assert self._plan is not None
        self._capture_run_telemetry(log, "before")
        grip_index = next(i for i, p in enumerate(self._plan.poses) if p.name == "grip")
        previous = self._last_command
        for pose in self._plan.poses[grip_index + 1:]:
            if self._stop.is_set():
                break
            complete = self._ramp(previous, pose.joints, pose.ramp_s / speed_ratio,
                                  rate_hz, speed, pose.name, log)
            if not complete:
                break
            previous = pose.joints
            if pose.hold_s > 0 and self._stop.wait(pose.hold_s / speed_ratio):
                break
            with self._lock:
                self._current_pose = pose.name
        self._capture_run_telemetry(log, "after")

    # ---- experimental local-policy write stream ----------------------------------------
    def begin_stream(self, *, timeout_s: float = 0.25) -> str:
        if not 0.1 <= timeout_s <= 2.0:
            raise ValueError("stream timeout_s must be in [0.1, 2.0]")
        with self._lock:
            self._require_motion_ready()
            self._require_idle()
            self._operation = "policy stream"
            self._stream_token = uuid.uuid4().hex
            self._stream_timeout_s = timeout_s
            self._stream_deadline = time.monotonic() + timeout_s
            self._event("warning", "experimental write-only policy stream armed",
                        {"timeout_s": timeout_s})
            return self._stream_token

    def stream_frame(self, token: str, joints: dict[str, dict[str, float]],
                     *, servo_speed: int | None = None) -> None:
        with self._lock:
            if token != self._stream_token or self._operation != "policy stream":
                raise RuntimeErrorState("policy stream token is absent or expired")
            normalized = self._validate_sim_pose(joints)
            self._stream_deadline = time.monotonic() + self._stream_timeout_s
        self.backend.write_joints(self._servo_pose(normalized), speed=servo_speed)
        with self._lock:
            self._last_command = normalized

    def end_stream(self, token: str) -> None:
        with self._lock:
            if token != self._stream_token:
                raise RuntimeErrorState("wrong policy stream token")
            self._end_stream("policy stream ended")

    # ---- stop, logs, shutdown ------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self._stream_token is not None:
                self._end_stream("policy stream stopped")
        try:
            self.backend.stop()
        finally:
            self._event("warning", "stop requested; servos hold their last position goal")

    def score_run(self, run_id: str, *, success: bool,
                  reorientation_deg: float | None, notes: str = "") -> dict:
        log = self._logs.get(run_id)
        if log is None:
            summary = self.logs_dir / f"{run_id}_SUMMARY.json"
            if not summary.exists():
                raise KeyError(run_id)
            raw = json.loads(summary.read_text(encoding="utf-8"))
            raw["manual_score"] = {"success": bool(success),
                                   "reorientation_deg": reorientation_deg,
                                   "notes": notes, "scored_at": _utc_now()}
            summary.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
            return raw
        log.score(success, reorientation_deg, notes)
        return dict(log.summary)

    def list_logs(self) -> list[dict]:
        out = []
        if self.logs_dir.exists():
            for path in sorted(self.logs_dir.glob("*_SUMMARY.json"), reverse=True):
                try:
                    out.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
        return out

    def close(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
        self._shutdown.set()
        self._telemetry_thread.join(timeout=2.0)
        self._watchdog_thread.join(timeout=2.0)
        self.backend.close()

    # ---- internals ----------------------------------------------------------------------
    def _start_operation(self, name: str, fn, *args) -> None:
        self._require_idle()
        self._operation = name
        self._stop.clear()
        self._last_error = None
        self._event("info", name)

        def run():
            error = None
            try:
                fn(*args)
            except Exception as exc:  # surfaced in status/events; request already returned 202
                error = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._last_error = error
                self._event("error", f"{name} failed", {"error": error})
            finally:
                with self._lock:
                    stopped = self._stop.is_set()
                    if self._active_log is not None:
                        self._active_log.finish("failed" if error else "stopped" if stopped else "complete",
                                                error)
                        self._active_log = None
                    self._operation = "idle"
                    self._stop.clear()
                if error is None:
                    self._event("info", f"{name} {'stopped' if stopped else 'complete'}")

        self._operation_thread = threading.Thread(target=run, name=f"manta-{name}", daemon=True)
        self._operation_thread.start()

    def _ramp(self, start: dict[str, dict[str, float]], end: dict[str, dict[str, float]],
              duration: float, rate_hz: float, speed: int, label: str,
              log: RunLog | None = None) -> bool:
        n = max(1, int(round(max(0.0, duration) * rate_hz)))
        dt = 1.0 / rate_hz
        t0 = time.monotonic()
        for i in range(1, n + 1):
            if self._stop.is_set():
                return False
            u = i / n
            pose = {f: {j: float(start[f][j] + (end[f][j] - start[f][j]) * u)
                        for j in JOINT_ORDER} for f in FINGER_ORDER}
            self.backend.write_joints(self._servo_pose(pose), speed=speed if i == 1 else None)
            with self._lock:
                self._last_command = pose
            if log is not None:
                log.append("command", {"elapsed_s": time.monotonic() - t0,
                                       "phase": label, "frame": i, "frames": n,
                                       "sim_joint_deg": pose})
            delay = t0 + i * dt - time.monotonic()
            if delay > 0 and self._stop.wait(delay):
                return False
        return True

    def _servo_pose(self, sim_pose: dict[str, dict[str, float]]) -> dict[int, dict[str, float]]:
        if not self.signs_checked:
            raise RuntimeErrorState(
                "aa signs are unmeasured; start the server with explicit --aa-signs after the hardware check"
            )
        return {FINGER_ID[f]: {SIM_JOINT_TO_SERVO[j]: servo_deg(f, j, deg)
                               for j, deg in joints.items()}
                for f, joints in sim_pose.items()}

    def _validate_sim_pose(self, joints: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        if set(joints) != set(FINGER_ORDER):
            raise ValueError(f"pose fingers must be exactly {FINGER_ORDER}")
        out = {}
        for finger in FINGER_ORDER:
            if set(joints[finger]) != set(JOINT_ORDER):
                raise ValueError(f"{finger} joints must be exactly {JOINT_ORDER}")
            out[finger] = {}
            for sim_joint in JOINT_ORDER:
                value = float(joints[finger][sim_joint])
                servo_name = SIM_JOINT_TO_SERVO[sim_joint]
                _sid, _zero, (lo, hi) = FINGER_JOINTS[FINGER_ID[finger]][servo_name]
                actual = servo_deg(finger, sim_joint, value)
                if not math.isfinite(value) or not lo <= actual <= hi:
                    raise ValueError(f"{finger}_{sim_joint}={value} maps outside [{lo}, {hi}]")
                out[finger][sim_joint] = value
        return out

    def _pose(self, name: str) -> Pose:
        assert self._plan is not None
        try:
            return next(p for p in self._plan.poses if p.name == name)
        except StopIteration:
            raise KeyError(f"plan {self._plan.design} has no pose {name!r}") from None

    @staticmethod
    def _validate_timing(speed_ratio: float, rate_hz: float) -> None:
        if not 0.1 <= speed_ratio <= 2.0:
            raise ValueError("speed_ratio must be in [0.1, 2.0]")
        if not 1.0 <= rate_hz <= 100.0:
            raise ValueError("rate_hz must be in [1, 100]")

    def _require_idle(self) -> None:
        if self._operation != "idle":
            raise RuntimeErrorState(f"hand is busy: {self._operation}")

    def _require_homed_plan(self) -> None:
        self._require_idle()
        if not self._homed:
            raise RuntimeErrorState("home the gantries once in this daemon session first")
        if self._plan is None:
            raise RuntimeErrorState("load a validated hand plan first")

    def _require_motion_ready(self) -> None:
        self._require_homed_plan()
        if not self._mounts_applied:
            raise RuntimeErrorState("apply the selected morphology before moving fingers")
        if not self.signs_checked:
            raise RuntimeErrorState("aa signs have not been hardware-verified")

    def _event(self, level: str, message: str, data: dict | None = None) -> None:
        with self._lock:
            self._event_seq += 1
            event = Event(self._event_seq, _utc_now(), level, message, data or {})
            self._events.append(event)
            if self._active_log is not None:
                self._active_log.append("event", asdict(event))

    def _telemetry_loop(self) -> None:
        if self.telemetry_hz <= 0:
            return
        period = 1.0 / self.telemetry_hz
        previous = None
        while not self._shutdown.is_set():
            start = time.monotonic()
            try:
                with self._lock:
                    # Servo feedback is never allowed to take the half-duplex bus
                    # away from a pose/trajectory writer.  Gantry status is on the
                    # independent USB-CDC link and may continue.
                    include_servos = self._operation == "idle"
                sample = self.backend.read_telemetry(include_servos=include_servos)
                now = time.time()
                with self._lock:
                    count = int(self._telemetry["samples"]) + 1
                    measured = None if previous is None else 1.0 / max(1e-9, start - previous)
                    updated = dict(self._telemetry)
                    updated.update({"timestamp": now, "age_s": 0.0, "error": None,
                                    "samples": count, "measured_hz": measured,
                                    "servo_polling_suspended": not include_servos, **sample})
                    if "servos" in sample:
                        updated["servo_timestamp"] = now
                        updated["servo_age_s"] = 0.0
                    self._telemetry = updated
                    if self._active_log is not None:
                        self._active_log.append("telemetry", {"phase": "during",
                                                              "data": sample})
                previous = start
            except Exception as exc:
                with self._lock:
                    self._telemetry["error"] = f"{type(exc).__name__}: {exc}"
                # Do not hammer a sick packet bus.  This also turns a missing
                # sync-read feature into one clear, low-rate error.
                self._shutdown.wait(max(period, 1.0))
            delay = start + period - time.monotonic()
            if delay > 0:
                self._shutdown.wait(delay)

    def _capture_run_telemetry(self, log: RunLog, phase: str) -> None:
        """One deliberate servo snapshot outside the high-rate command ramp."""
        try:
            sample = self.backend.read_telemetry(include_servos=True)
            log.append("telemetry", {"phase": phase, "data": sample})
        except Exception as exc:
            log.append("event", {"level": "warning", "message": "telemetry snapshot failed",
                                 "phase": phase, "error": f"{type(exc).__name__}: {exc}"})

    def _watchdog_loop(self) -> None:
        while not self._shutdown.wait(0.05):
            with self._lock:
                if self._stream_token is not None and time.monotonic() > self._stream_deadline:
                    self._end_stream("policy stream lease expired; holding last servo goals")

    def _end_stream(self, message: str) -> None:
        self._stream_token = None
        self._stream_deadline = 0.0
        self._operation = "idle"
        self._event("warning", message)
