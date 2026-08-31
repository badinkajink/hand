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

from .kinematics import (FULL_EXTENSION_MM, STEPPER_JOINTS, STEPS_PER_MM,
                          HomingAborted, _home_timeout_ms)
from .plan import (FINGER_ID, FINGER_NAME, JOINT_SIGN, SIM_JOINT_TO_SERVO, HandPlan,
                    Pose, local_from_palm, servo_deg, stepper_mm)
from .manual import (ManualCommandError, check_mount as manual_check_mount,
                     limits as manual_limits_payload, parse as manual_parse,
                     validate as manual_validate)
from .protocol import MantaHandError
from .servos import (DEFAULT_JOINT_SPEED, FINGER_JOINTS, TORQUE_FREE, TORQUE_OFF,
                      TORQUE_ON, TORQUE_UNSET)

FINGER_ORDER = ("thumb", "index", "middle")
JOINT_ORDER = ("yaw", "mcp", "pip")
HOME_CONFIRMATION = "HOME ALL AXES"

# Consecutive telemetry failures before the runtime says so loudly rather than
# quietly retrying, and the ceiling on the geometric backoff between retries.
TELEMETRY_FAILURE_LIMIT = 3
TELEMETRY_BACKOFF_CAP_S = 10.0

# Static per-axis facts the operator needs in order to judge what they are watching.
# The home timeout is the big one: an axis whose StallGuard2 does not fire runs its
# full computed window pressed against the hardstop -- 46s on J0 alone -- and without
# this number on screen that is indistinguishable from a hang. See
# kinematics._home_timeout_ms for why the window cannot simply be shortened.
AXIS_INFO = [
    {"joint": j,
     "finger": FINGER_NAME[fid],
     "axis": "x" if j == pair[0] else "y",
     "travel_mm": FULL_EXTENSION_MM[j],
     "home_timeout_s": round(_home_timeout_ms(j) / 1000.0, 1)}
    for fid, pair in sorted(STEPPER_JOINTS.items()) for j in pair
]
HOME_WORST_CASE_S = round(sum(a["home_timeout_s"] for a in AXIS_INFO), 1)


class RuntimeErrorState(RuntimeError):
    """A command is unsafe or invalid in the runtime's current state."""


class LinkDown(RuntimeError):
    """A serial link stopped answering. Distinct from a command being refused: every
    piece of session state that depends on the hardware (homed, mounts applied, the
    current pose) becomes unknown when this is raised, so the runtime latches it and
    demands an explicit reconnect rather than letting the next command act on stale
    assumptions."""


def _is_link_failure(exc: BaseException) -> bool:
    """Does this exception mean 'the device is gone' rather than 'the command failed'?

    pyserial raises SerialException for a vanished /dev node, and OSError/termios errors
    for a port that re-enumerated underneath us. MantaHandError covers protocol-level
    trouble, and only its timeout/desync forms indicate a link problem -- an ERR reply
    is the firmware refusing a command, which is not the link's fault."""
    import serial  # local: keeps this module importable without pyserial

    if isinstance(exc, (serial.SerialException, OSError)):
        return True
    text = str(exc).lower()
    return isinstance(exc, MantaHandError) and (
        "timeout" in text or "unexpected reply" in text or "got fewer" in text)


class HardwareBackend(Protocol):
    kind: str

    def home_all(self, cancel=None, report=None) -> list[dict]: ...
    def apply_mounts(self, plan: HandPlan, cancel=None, report=None) -> None: ...
    def move_mounts(self, targets: dict[int, tuple[float, float]],
                    cancel=None, report=None) -> None: ...
    def write_joints(self, pose: dict[int, dict[str, float]], speed: int | None = None) -> None: ...
    def read_telemetry(self, include_servos: bool = True) -> dict: ...
    def adoptable_home(self) -> dict: ...
    def set_servo_torque(self, state: int) -> dict: ...
    def disable_motors(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class RealHardwareBackend:
    """The two physical serial links on the CB1, opened only on construction.

    Servo torque is enabled here, at connect, exactly as examples/hand_control.py does
    before it calls home_all(). This is not a nicety: a torque-OFF SCS0009 accepts a
    goal-position write and reads it back correctly without moving, so a service that
    skips this step reports a successful home having zeroed nothing. See
    ServoBus.set_torque_all."""

    kind = "real"

    def __init__(self, stepper_port: str = "/dev/ttyACM0", servo_port: str = "/dev/ttyUSB0",
                 *, enable_servos: bool = True, servo_fields: tuple[str, ...] = ()):
        from .driver import MantaHandDriver
        from .hand import Hand
        from .servos import ServoBus

        self.stepper_port = stepper_port
        self.servo_port = servo_port
        # Extra per-servo feedback fields to sample alongside position. Each one costs
        # another nine transactions (position alone measured 111Hz, position+load
        # 55.6Hz), so this is opt-in rather than "read everything".
        self.servo_fields = tuple(servo_fields)
        self.link_error: str | None = None
        self.driver = MantaHandDriver(stepper_port)
        try:
            self.servos = ServoBus(servo_port)
            self.hand = Hand(self.driver, self.servos)
            if enable_servos:
                self.servos.enable_all()
        except Exception:
            self.driver.close()
            raise

    # -- link supervision ---------------------------------------------------------------
    def guard(self, fn, *args, **kwargs):
        """Run a hardware call, latching link failures so callers can tell 'the command
        was refused' from 'the device is gone'."""
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            if _is_link_failure(exc):
                self.link_error = f"{type(exc).__name__}: {exc}"
                raise LinkDown(
                    f"{self.stepper_port} stopped answering ({exc}). The M8P needs the "
                    f"BOOT0/RESET cycle from the README before it will re-enumerate; the "
                    f"home reference is gone with it."
                ) from exc
            raise
        self.link_error = None
        return result

    def adoptable_home(self) -> dict:
        """Does the BOARD still vouch for a home the host has forgotten?

        The M8P keeps its step counters, its SETSCALE calibration and its per-axis
        `homing_result` across a HOST restart -- only a board reset or an explicit DIS
        clears them. A daemon restart therefore does not invalidate the reference, even
        though the session state is rebuilt from nothing. Re-homing in that situation
        costs about two minutes of the rails grinding into their hardstops for no new
        information, on a hand that may be holding something.

        `homing_result` 2 (stalled) or 3 (timed out having covered the full measured
        travel) is the firmware's own record that a home completed. An axis that is
        disabled, still moving, or reporting 0/1 has no trustworthy reference and
        disqualifies the whole set -- the reference is only as good as its worst axis."""
        status = self.guard(self.driver.get_all_status)[:6]
        axes = []
        for i, st in enumerate(status):
            axes.append({"joint": i, "homing_result": st.homing_result,
                         "enabled": st.enabled, "moving": st.moving,
                         "position_mm": st.position / STEPS_PER_MM[i],
                         "stalled": st.homing_result == 2, "elapsed_s": 0.0,
                         "ok": bool(st.enabled and not st.moving
                                    and st.homing_result in (2, 3))})
        return {"adoptable": all(a["ok"] for a in axes), "axes": axes}

    def ping(self) -> bool:
        try:
            self.guard(self.driver.get_all_status)
            return True
        except Exception:
            return False

    # -- operations ---------------------------------------------------------------------
    def home_all(self, cancel=None, report=None) -> list[dict]:
        return self.guard(self.hand.home_all, cancel=cancel, report=report,
                          require_torque=True)

    def apply_mounts(self, plan: HandPlan, cancel=None, report=None) -> None:
        # sequential=True: one axis at a time, each waited on. See
        # Gantry.move_sequential for why six simultaneous MOVEMMs are not a profile
        # this hardware has been validated at.
        self.guard(plan.apply_mounts, self.hand, home=False, sequential=True,
                   cancel=cancel, report=report)

    def move_mounts(self, targets: dict[int, tuple[float, float]],
                    cancel=None, report=None) -> None:
        # Same one-axis-at-a-time profile as apply_mounts; a manual move is not a
        # different kind of motion, only a different source of the numbers.
        self.guard(self.hand.move_mounts_sequential, targets, frame="global",
                   cancel=cancel, report=report)

    def write_joints(self, pose: dict[int, dict[str, float]], speed: int | None = None) -> None:
        self.hand.set_joints_fast(pose, speed=speed)

    def set_servo_torque(self, state: int) -> dict:
        return self.servos.set_torque_all(state)

    def read_telemetry(self, include_servos: bool = True) -> dict:
        step = self.guard(self.driver.get_all_status)[:6]
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
            out["servo_torque"] = self.servos.read_torque_all()
            for field in self.servo_fields:
                values = self.servos.read_field(field)
                if values is not None:
                    # Keyed by servo id, not joint name: present_load in particular is
                    # an uncalibrated per-servo number and giving it a joint name would
                    # invite reading it as a joint torque, which it is not.
                    out[f"servo_{field}"] = values
        out["servo_bus"] = self.servos.health()
        return out

    def stop(self) -> None:
        # STOPALL decelerates the gantries.  There is no equivalent broadcast
        # "stop" for position servos; cancelling the writer holds the last goal.
        self.driver.stop_all()

    def disable_motors(self) -> None:
        """The real kill switch: stop, then drop torque on all six steppers and all
        nine servos. STOPALL alone leaves every axis energised and holding, which is
        the wrong state for an axis that is grinding into a hardstop -- and STOPALL
        decelerates at HOME_ACCEL, so from 12000 sps that is ~6 seconds of continued
        motion before anything stops. Disabling also clears the firmware's
        homing_result, which is what actually cancels an in-flight home."""
        errors = []
        try:
            self.driver.stop_all()
        except Exception as exc:
            errors.append(f"STOPALL: {exc}")
        for x_joint, y_joint in STEPPER_JOINTS.values():
            for joint_index in (x_joint, y_joint):
                try:
                    self.driver.joints[joint_index].disable()
                except Exception as exc:
                    errors.append(f"DIS J{joint_index}: {exc}")
        try:
            self.servos.set_torque_all(TORQUE_OFF)
        except Exception as exc:
            errors.append(f"servo torque off: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def close(self) -> None:
        try:
            self.servos.set_torque_all(TORQUE_OFF)
        except Exception:
            pass  # closing must not fail on a link that is already gone
        self.driver.close()


class MockHardwareBackend:
    """Deterministic no-hardware backend for tests and UI work.

    Note the ceiling on what this can prove: it replaces the BACKEND, so no line of the
    real driver stack runs behind it. To exercise MantaHandDriver/Joint/Gantry/ServoBus
    without hardware, use fake_hardware.FakeM8P + FakeScs0009Controller behind a real
    RealHardwareBackend instead (`python -m manta_hand.web --fake`)."""

    kind = "mock"

    def __init__(self):
        self.homed = False
        self.mounts: dict[str, tuple[float, float]] = {}
        self.joints = {fid: {name: 0.0 for name in ("aa", "fe1", "fe2")}
                       for fid in range(3)}
        self.writes = 0
        self.torque = TORQUE_ON
        self.motors_disabled = False
        self.link_error: str | None = None

    def home_all(self, cancel=None, report=None) -> list[dict]:
        outcomes = []
        for joint in range(6):
            if cancel and cancel():
                raise HomingAborted(f"J{joint}: homing cancelled")
            if report:
                report("home_axis_start", {"joint": joint, "timeout_s": 0.01})
            time.sleep(0.003)
            outcomes.append({"joint": joint, "homing_result": 2, "stalled": True,
                             "elapsed_s": 0.003})
            if report:
                report("home_axis_done", {"joint": joint, "homing_result": 2,
                                          "stalled": True, "elapsed_s": 0.003})
        self.homed = True
        self.joints = {fid: {name: 0.0 for name in ("aa", "fe1", "fe2")}
                       for fid in range(3)}
        return outcomes

    def apply_mounts(self, plan: HandPlan, cancel=None, report=None) -> None:
        for finger, xy in plan.mounts_palm_mm.items():
            if cancel and cancel():
                raise HomingAborted("gantry move cancelled")
            if report:
                report("gantry_axis_start", {"joint": FINGER_ID[finger] * 2,
                                             "target_mm": xy[0]})
        self.mounts = dict(plan.mounts_palm_mm)

    def move_mounts(self, targets: dict[int, tuple[float, float]],
                    cancel=None, report=None) -> None:
        for fid, xy in targets.items():
            if cancel and cancel():
                raise HomingAborted("gantry move cancelled")
            if report:
                report("gantry_axis_start", {"joint": fid * 2, "target_mm": xy[0]})
            self.mounts[FINGER_NAME[fid]] = (float(xy[0]), float(xy[1]))

    def write_joints(self, pose: dict[int, dict[str, float]], speed: int | None = None) -> None:
        for fid, values in pose.items():
            self.joints[fid].update(values)
        self.writes += 1

    def set_servo_torque(self, state: int) -> dict:
        self.torque = state
        return {sid: state for sid in range(9)}

    def disable_motors(self) -> None:
        self.motors_disabled = True
        self.torque = TORQUE_OFF

    def ping(self) -> bool:
        return True

    def adoptable_home(self) -> dict:
        axes = [{"joint": i, "homing_result": 2 if self.homed else 0,
                 "enabled": not self.motors_disabled, "moving": False,
                 "position_mm": 0.0, "stalled": True, "elapsed_s": 0.0,
                 "ok": self.homed and not self.motors_disabled} for i in range(6)]
        return {"adoptable": all(a["ok"] for a in axes), "axes": axes}

    def read_telemetry(self, include_servos: bool = True) -> dict:
        steppers = []
        for i in range(6):
            steppers.append({"id": i, "position_mm": 0.0, "target_mm": 0.0,
                             "moving": False, "enabled": not self.motors_disabled,
                             "homing_result": 2 if self.homed else 0})
        out = {"steppers": steppers,
               "servo_bus": {"transactions": 0, "timeouts": 0,
                             "consecutive_timeouts": 0, "timeout_rate": 0.0}}
        if include_servos:
            out["servos"] = json.loads(json.dumps(self.joints))
            out["servo_torque"] = {sid: self.torque for sid in range(9)}
        return out

    def stop(self) -> None:
        return None

    def close(self) -> None:
        return None


#: A tracker sample older than this is not an object pose, it is a memory of one. The
#: camera runs at 30 Hz and pushes at 10, so anything past a second means the tracker
#: process died, the tag went out of view, or the link to the workstation dropped.
TRACKER_STALE_S = 1.0


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
            "samples": {"command": 0, "telemetry": 0, "event": 0, "object": 0},
            "manual_score": None,
            # What the AprilTag tracker measured, kept SEPARATE from manual_score on
            # purpose: the operator's estimate and the instrument's are two readings of
            # the same run and the interesting case is the one where they disagree.
            "object_track": None,
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

    def track(self, block: dict) -> None:
        with self._lock:
            self.summary["object_track"] = block
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
        # Where the gantries were last COMMANDED to, palm-frame mm, whether by a plan's
        # morphology or by hand. Kept apart from _mounts_applied: that flag means "the
        # loaded plan's morphology is on the hand", and a manual move makes it false
        # while still leaving the gantries at a known, motion-legal place.
        self._mount_positions: dict[str, dict[str, float]] | None = None
        self._manual_mounts = False
        self._operation = "idle"
        self._current_pose: str | None = None
        self._last_command = _zero_sim_pose()
        self._last_error: str | None = None
        self.servo_fields = tuple(getattr(backend, "servo_fields", ()))
        self._telemetry: dict = {"timestamp": None, "age_s": None, "servos": None,
                                "servo_timestamp": None, "servo_age_s": None,
                                "steppers": None, "error": None, "samples": 0,
                                "measured_hz": None, "servo_polling_suspended": False}
        self._active_log: RunLog | None = None
        self._logs: dict[str, RunLog] = {}
        # Pushed by scripts/real_v1_tag_tracker.py, which owns the camera on the
        # workstation. The station never opens a camera itself: the RealSense is on the
        # other side of the subnet and the CB1 has neither the USB bandwidth nor the
        # dependencies. What arrives here is already-reduced geometry, in millimetres.
        self._tracker: dict = {"last": None, "received": 0, "timestamp": None,
                               "run_id": "", "start_cos": None, "peak_cos": None,
                               "min_cos": None, "start_z_mm": None, "min_z_mm": None,
                               "summary": None, "source": None}
        self._home_outcomes: list[dict] = []
        self._home_progress: dict | None = None
        self._link_down: str | None = None
        self._unhomed_reason: str | None = None
        self._servo_torque: int | None = None
        self._telemetry_failures = 0
        self._stream_token: str | None = None
        self._stream_deadline = 0.0
        self._stream_timeout_s = 0.25
        self._startup_check()
        self._event("info", f"runtime ready ({backend.kind})")
        self._telemetry_thread = threading.Thread(target=self._telemetry_loop,
                                                  name="manta-telemetry", daemon=True)
        self._telemetry_thread.start()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop,
                                                 name="manta-stream-watchdog", daemon=True)
        self._watchdog_thread.start()

    def _startup_check(self) -> None:
        """One telemetry read at construction, before anything can be commanded.

        Two jobs. It populates the servo torque state, which the backend set at connect
        and which nothing else would report until the first home -- the UI would
        otherwise show "unknown" for a hand that is in fact energised. And it puts a
        broken link or a silent servo bus in front of the operator at startup, rather
        than at the first command, which is when it used to surface."""
        try:
            sample = self.backend.read_telemetry(include_servos=True)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._telemetry["error"] = message
                if isinstance(exc, LinkDown):
                    self._link_down = str(exc)
            self._event("error", "hardware did not answer at startup", {"error": message})
            return
        now = time.time()
        with self._lock:
            self._telemetry.update({"timestamp": now, "age_s": 0.0, "error": None,
                                    "servo_timestamp": now, "servo_age_s": 0.0,
                                    "samples": 1, **sample})
            torque = set((sample.get("servo_torque") or {}).values())
            if len(torque) == 1:
                value = torque.pop()
                self._servo_torque = TORQUE_OFF if value == TORQUE_UNSET else value
        self._seed_last_command(sample)
        bus = sample.get("servo_bus") or {}
        latency = bus.get("ftdi_latency") or {}
        if latency.get("after_ms") and latency["after_ms"] > 1:
            # 16x on every servo read, and nothing errors -- say so loudly once.
            self._event("warning", f"servo link latency timer is "
                                    f"{latency['after_ms']}ms: {latency.get('note', '')}",
                        latency)


    def _seed_last_command(self, sample: dict) -> None:
        """Adopt the hand's MEASURED pose as the ramp start.

        `_last_command` is what every ramp interpolates FROM, and a fresh runtime used to
        initialise it to the zero pose on the assumption that a session begins with a home.
        It does not: adopting an existing home, or simply restarting this daemon while the
        servos hold their last goal, leaves the hand somewhere else entirely -- after the
        2026-08-29 run it was still holding `turn_end`. The first frame of the next ramp
        then jumps the fingers to a pose nobody asked for before ramping anywhere.

        Sign-only mapping (see plan.servo_deg), so the inverse is the same multiply."""
        servos = sample.get("servos")
        if not servos:
            return
        pose: dict[str, dict[str, float]] = {}
        try:
            for finger in FINGER_ORDER:
                fid = FINGER_ID[finger]
                # int keys straight off the backend, str keys after a JSON round trip.
                measured = servos.get(fid, servos.get(str(fid)))
                pose[finger] = {
                    sim_joint: measured[SIM_JOINT_TO_SERVO[sim_joint]]
                               / JOINT_SIGN[(finger, sim_joint)]
                    for sim_joint in JOINT_ORDER
                }
        except (KeyError, TypeError, ZeroDivisionError) as exc:
            # Not fatal -- but say so, because falling back to the zero assumption is
            # exactly the silent wrong-start this method exists to prevent.
            self._event("warning", "could not seed the ramp start from telemetry; "
                                   "ramps will assume the zero pose",
                        {"error": f"{type(exc).__name__}: {exc}"})
            return
        with self._lock:
            self._last_command = pose
        self._event("info", "ramp start seeded from the servos' measured pose",
                    {finger: {j: round(v, 2) for j, v in joints.items()}
                     for finger, joints in pose.items()})

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
            track = dict(self._tracker)
            track["age_s"] = (None if track["timestamp"] is None
                              else max(0.0, time.time() - track["timestamp"]))
            track["fresh"] = track["age_s"] is not None and track["age_s"] < TRACKER_STALE_S
            return {
                "schema_version": 1,
                "backend": self.backend.kind,
                "homed": self._homed,
                "link_down": self._link_down,
                "unhomed_reason": self._unhomed_reason,
                "home_outcomes": list(self._home_outcomes),
                "home_progress": self._home_progress,
                "home_worst_case_s": HOME_WORST_CASE_S,
                "servo_torque": self._servo_torque,
                "mounts_applied": self._mounts_applied,
                "manual_mounts": self._manual_mounts,
                "mount_positions": self._mount_positions,
                "operation": self._operation,
                "busy": self._operation != "idle",
                "current_pose": self._current_pose,
                # What the joints were last TOLD to do, sim frame. Distinct from
                # telemetry["servos"], which is what they report having done -- the
                # manual controls seed from this so a slider shows the target, not the
                # droop.
                "last_command": {f: dict(v) for f, v in self._last_command.items()},
                "plan": plan,
                "tracker": track,
                "signs_checked": self.signs_checked,
                "axes": AXIS_INFO,
                "telemetry": telem,
                "last_error": self._last_error,
                "streaming": self._stream_token is not None,
                "servo_fields": list(self.servo_fields),
                "capabilities": {
                    "servo_position": "sync-read when supported by installed rustypot",
                    "servo_load": "uncalibrated duty-cycle proxy; NOT current, NOT force",
                    # The SCS0009 has no present-current register at all -- this is not a
                    # question of read cost, the data does not exist on the part. Checked
                    # against the installed rustypot register table 2026-08-29.
                    "servo_current": False,
                    # True only while samples are actually arriving. The capability is a
                    # running tracker process, not a line of code -- the camera can be
                    # unplugged, aimed at a wall, or looking at a shadowed tag, and in
                    # every one of those cases this must not claim an object sensor.
                    "object_pose": ("AprilTag id0 on the shaft against static id6, pushed "
                                    "by scripts/real_v1_tag_tracker.py"
                                    if track["fresh"] else False),
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
            self._require_link()
            if self._homed and not force:
                raise RuntimeErrorState("already homed in this daemon session; pass force=true to re-home")
            # The reference is unknown from the first HOME command until the last axis
            # is zeroed. Clearing it up front means an abort, a crash or a link failure
            # part-way through cannot leave the session believing in a reference that
            # no longer exists.
            self._homed = False
            self._mounts_applied = False
            self._manual_mounts = False
            self._mount_positions = None
            self._home_outcomes = []
            self._home_progress = None
            self._unhomed_reason = "homing in progress"
            self._start_operation("homing", self._do_home)

    def _do_home(self) -> None:
        outcomes = self.backend.home_all(cancel=self._stop.is_set, report=self._home_report)
        with self._lock:
            self._homed = True
            self._unhomed_reason = None
            self._home_outcomes = list(outcomes)
            self._home_progress = None
            self._mounts_applied = False
            self._manual_mounts = False
            self._mount_positions = None
            self._current_pose = "zero"
            self._last_command = _zero_sim_pose()
            self._servo_torque = TORQUE_ON
        timed_out = [o["joint"] for o in outcomes if not o.get("stalled", True)]
        if timed_out:
            # Not a failure -- the timeout path is a deliberate, documented outcome
            # (see kinematics.HOME_COOLCONF) -- but it means those axes ground against
            # their hardstop for the axis's full computed timeout, 22-46s each. Anyone
            # watching needs to be told that was expected.
            self._event("warning",
                        f"StallGuard2 did not fire on J{', J'.join(str(j) for j in timed_out)}; "
                        f"homed via the timeout guarantee instead (expected on J3/J5 with the "
                        f"current SGT tuning)",
                        {"joints": timed_out,
                         "outcomes": outcomes})

    def _home_report(self, event: str, payload: dict) -> None:
        """Per-axis progress from kinematics._home_one_axis, surfaced live."""
        with self._lock:
            self._home_progress = {"event": event, **payload}
        if event == "home_axis_start":
            self._event("info", f"homing J{payload['joint']} "
                                f"(up to {payload.get('timeout_s', 0):.0f}s of travel)", payload)
        elif event == "home_axis_done":
            how = "StallGuard2" if payload.get("stalled") else "timeout guarantee"
            self._event("info", f"J{payload['joint']} homed via {how} "
                                f"in {payload.get('elapsed_s', 0):.1f}s", payload)
        elif event == "gantry_axis_start":
            self._event("info", f"J{payload['joint']} -> {payload.get('target_mm', 0):.2f}mm",
                        payload)

    def adopt_home(self, *, tolerance_mm: float = 0.5) -> dict:
        """Take over a home the board still vouches for, without re-homing.

        Also adopts `mounts_applied` when the gantries are already sitting at the loaded
        plan's stepper targets within `tolerance_mm` -- which after a daemon restart they
        usually are, because nothing moved them. Both adoptions are logged as adopted
        rather than performed, so a run summary never claims a home this process watched
        when it did not."""
        with self._lock:
            self._require_link()
            self._require_idle()
            if self._homed:
                raise RuntimeErrorState("this session is already homed")
        report = self.backend.adoptable_home()
        if not report["adoptable"]:
            bad = [str(a["joint"]) for a in report["axes"] if not a["ok"]]
            raise RuntimeErrorState(
                f"the board does not vouch for a completed home on J{', J'.join(bad)} "
                f"(needs enabled, stopped, homing_result 2 or 3) -- home properly instead")
        with self._lock:
            self._homed = True
            self._unhomed_reason = None
            self._home_outcomes = [{k: a[k] for k in
                                    ("joint", "homing_result", "stalled", "elapsed_s")}
                                   for a in report["axes"]]
            self._current_pose = None
            adopted_mounts = False
            if self._plan is not None:
                targets: dict[int, float] = {}
                for finger, (x, y) in self._plan.mounts_palm_mm.items():
                    lx, ly = local_from_palm(finger, x, y)
                    targets.update(stepper_mm(finger, lx, ly))
                here = {a["joint"]: a["position_mm"] for a in report["axes"]}
                deltas = [abs(here[j] - mm) for j, mm in targets.items() if j in here]
                if deltas and max(deltas) <= tolerance_mm:
                    self._mounts_applied = True
                    self._manual_mounts = False
                    self._mount_positions = {f: {"x": xy[0], "y": xy[1]}
                                             for f, xy in self._plan.mounts_palm_mm.items()}
                    adopted_mounts = True
        self._event("warning",
                    "adopted the board's existing home without re-homing"
                    + (" (and its morphology position)" if adopted_mounts else ""),
                    {"axes": report["axes"], "mounts_adopted": adopted_mounts})
        return self.state()

    def apply_morphology(self) -> None:
        with self._lock:
            self._require_link()
            self._require_homed_plan()
            self._start_operation("applying morphology", self._do_apply_morphology)

    def _do_apply_morphology(self) -> None:
        assert self._plan is not None
        self.backend.apply_mounts(self._plan, cancel=self._stop.is_set,
                                  report=self._home_report)
        with self._lock:
            if not self._stop.is_set():
                self._mounts_applied = True
                self._manual_mounts = False
                self._mount_positions = {f: {"x": xy[0], "y": xy[1]}
                                         for f, xy in self._plan.mounts_palm_mm.items()}
                # NOT "zero": this operation moves steppers and never commands a servo.
                # Labelling it zero was true only on the home-then-morphology path, where
                # home_all had just zeroed the servos; on adopt-home, or on a second
                # morphology change, it claimed a pose the hand was not in.
                self._current_pose = None

    # ------------------------------------------------------------------ manual control
    # `examples/hand_control.py` over the station's own link. It exists because that
    # script needs the two USB ports for itself, so using it means stopping the service
    # -- and the thing you most often want to do by hand (nudge one finger, walk a
    # gantry clear, check a sign) is exactly the thing you want to do WITHOUT losing the
    # home and the telemetry.

    def manual_limits(self) -> dict:
        """Bounds and current targets, i.e. everything a UI needs to build the controls."""
        payload = manual_limits_payload()
        with self._lock:
            payload["mount_positions"] = self._mount_positions
            payload["manual_mounts"] = self._manual_mounts
            payload["mounts_applied"] = self._mounts_applied
            payload["last_command"] = {f: dict(v) for f, v in self._last_command.items()}
        return payload

    def manual_resolve(self, line: str) -> dict:
        """Parse and bounds-check a line WITHOUT moving anything.

        The UI needs this because palm mm and firmware mm are different numbers for the
        same place and only this process knows the transform -- duplicating it in
        JavaScript is exactly the drift `kinematics` exists to prevent."""
        with self._lock:
            return manual_validate(manual_parse(line), current_mounts=self._mount_positions)

    def manual_joints(self, joints: dict[str, dict[str, float]],
                      *, servo_speed: int | None = None) -> dict:
        """Move only the joints named; every other joint is re-commanded to where it
        already was, which for a position servo is what holding still means."""
        if not joints:
            raise ValueError("no joints given")
        with self._lock:
            self._require_motion_ready()
            self._require_idle()
            merged = {f: dict(self._last_command[f]) for f in FINGER_ORDER}
            for finger, values in joints.items():
                if finger not in merged:
                    raise ValueError(f"no finger {finger!r} (have {FINGER_ORDER})")
                merged[finger].update({k: float(v) for k, v in values.items()})
            normalized = self._validate_sim_pose(merged)
        self.backend.write_joints(self._servo_pose(normalized),
                                  speed=servo_speed or DEFAULT_JOINT_SPEED)
        with self._lock:
            self._last_command = normalized
            self._current_pose = None      # no longer at any named pose
        self._event("info", "manual joint write",
                    {"joints": {f: dict(v) for f, v in joints.items()},
                     "servo_speed": servo_speed or DEFAULT_JOINT_SPEED})
        return normalized

    def manual_mounts(self, targets: dict[str, tuple[float, float]]) -> dict:
        """Drive one or more gantries to palm-frame (x, y). Async, like a morphology
        apply, because steppers take seconds and move one axis at a time."""
        if not targets:
            raise ValueError("no mounts given")
        with self._lock:
            self._require_link()
            self._require_idle()
            if not self._homed:
                detail = f" ({self._unhomed_reason})" if self._unhomed_reason else ""
                raise RuntimeErrorState(
                    f"home the gantries once in this daemon session first{detail}")
            resolved = {}
            for finger, (x, y) in targets.items():
                steppers = manual_check_mount(finger, float(x), float(y))
                resolved[finger] = {"x": float(x), "y": float(y),
                                    "steppers": {str(k): round(v, 3) for k, v in steppers.items()}}
            self._start_operation("manual gantry move", self._do_manual_mounts,
                                  {f: (v["x"], v["y"]) for f, v in resolved.items()})
        return resolved

    def _do_manual_mounts(self, targets: dict[str, tuple[float, float]]) -> None:
        self.backend.move_mounts({FINGER_ID[f]: xy for f, xy in targets.items()},
                                 cancel=self._stop.is_set, report=self._home_report)
        with self._lock:
            if self._stop.is_set():
                return
            known = dict(self._mount_positions or {})
            known.update({f: {"x": xy[0], "y": xy[1]} for f, xy in targets.items()})
            self._mount_positions = known
            self._manual_mounts = True
            # The loaded plan's morphology is no longer what is on the hand -- but the
            # gantries ARE at a known, bounds-checked place, which is what the finger
            # interlock actually needs.
            self._mounts_applied = False
            self._current_pose = None

    def manual_command(self, line: str, *, servo_speed: int | None = None) -> dict:
        """One `hand_control.py` line. Parsed and bounds-checked whole before anything
        moves, so a typo in the third segment does not leave the first two applied."""
        request = manual_parse(line)
        if not request:
            raise ManualCommandError("nothing to do")
        with self._lock:
            checked = manual_validate(request, current_mounts=self._mount_positions)
        if checked["mounts"] and checked["joints"]:
            # A gantry move is an async operation and a joint write requires an idle
            # runtime, so one line cannot be both without racing itself. hand_control
            # could, because it was synchronous and owned the ports.
            raise ManualCommandError(
                "one line cannot move gantries and joints at once here -- the gantry move "
                "is asynchronous and the joint write needs an idle hand. Send them as two "
                "lines, gantries first.")
        out = {"line": line, "mounts": {}, "joints": {}}
        if checked["mounts"]:
            out["mounts"] = self.manual_mounts(
                {f: (v["x"], v["y"]) for f, v in checked["mounts"].items()})
        if checked["joints"]:
            out["joints"] = self.manual_joints(checked["joints"], servo_speed=servo_speed)
        return out

    def set_servo_torque(self, state: int) -> dict:
        if state not in (TORQUE_ON, TORQUE_OFF, TORQUE_FREE):
            raise ValueError(f"torque state must be one of {TORQUE_ON}/{TORQUE_OFF}/{TORQUE_FREE}")
        with self._lock:
            self._require_link()
            self._require_idle()
        result = self.backend.set_servo_torque(state)
        with self._lock:
            self._servo_torque = state
            if state != TORQUE_ON:
                # A servo that is not holding has moved, or will; the commanded pose is
                # no longer where the hand is.
                self._current_pose = None
        self._event("warning", f"servo torque set to {state}", {"readback": result})
        return result

    def disable_motors(self) -> dict:
        """Stop and de-energise everything. Invalidates the home: a disabled stepper is
        free to be moved by hand or by the mechanism, so its step counter no longer
        describes where the axis physically is."""
        self._stop.set()
        try:
            self.backend.disable_motors()
        finally:
            with self._lock:
                self._homed = False
                self._mounts_applied = False
                self._manual_mounts = False
                self._mount_positions = None
                self._current_pose = None
                self._unhomed_reason = "motors were disabled; the step reference is gone"
                self._servo_torque = TORQUE_OFF
                if self._stream_token is not None:
                    self._end_stream("policy stream stopped by motor disable")
            self._event("warning", "motors disabled -- re-home before moving anything")
        return self.state()

    def reconnect(self) -> dict:
        """Re-check a latched-down link. Clears the latch only if the board answers."""
        with self._lock:
            self._require_idle()
        ok = bool(getattr(self.backend, "ping", lambda: True)())
        with self._lock:
            if ok:
                self._link_down = None
                self._event("info", "link is answering again; re-home before moving")
            else:
                self._event("error", "link still not answering -- do the BOOT0/RESET cycle")
        return self.state()

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
        """Cancel the running operation and decelerate the gantries.

        Two things this does NOT do, both of which have surprised an operator:
        servos are position devices and hold their last goal (use disable_motors for
        torque off), and STOPALL decelerates at the axis's configured accel -- from
        12000 sps at 2000 sps^2 that is about six seconds of continued travel. The
        cancel flag is what actually ends the operation; the deceleration is what ends
        the motion."""
        self._stop.set()
        with self._lock:
            operation = self._operation
            if self._stream_token is not None:
                self._end_stream("policy stream stopped")
        try:
            self.backend.stop()
        finally:
            self._event("warning",
                        f"stop requested during {operation!r}; gantries decelerate over "
                        f"~6s and servos hold their last goal")

    def tracker_sample(self, sample: dict) -> dict:
        """Take one reading from the workstation's tag tracker.

        Two jobs, and only two. It keeps the newest reading so the web app can show where
        the shaft actually is, and -- when a run is in flight -- it appends the reading to
        that run's JSONL so the trace and the joint commands share one timeline. It does
        NOT do the geometry (that happened in `morphohand.bench.tags`, on the machine with
        the camera) and it does NOT summarise the trace: the tracker pushes its own summary
        at the end of a run, computed by the same code the offline analysis uses, so there
        is one implementation of "it turned 42 degrees" rather than two that can drift.
        """
        now = time.time()
        with self._lock:
            t = self._tracker
            run_id = str(sample.get("run_id") or "")
            if run_id != t.get("run_id"):
                # a new run resets the per-run extremes; a run_id-less push does not
                t.update({"run_id": run_id, "start_cos": None, "peak_cos": None,
                          "min_cos": None, "start_z_mm": None, "min_z_mm": None,
                          "summary": None})
            t["received"] += 1
            t["timestamp"] = now
            t["source"] = sample.get("source") or t.get("source")
            if sample.get("summary") is not None:
                t["summary"] = sample["summary"]
            seen = bool(sample.get("seen"))
            t["last"] = sample if seen else {**(t.get("last") or {}), "seen": False,
                                             "t": sample.get("t")}
            if seen and sample.get("cos") is not None:
                cos, z = float(sample["cos"]), sample.get("z_bench_mm")
                if t["start_cos"] is None:
                    t["start_cos"] = cos
                t["peak_cos"] = cos if t["peak_cos"] is None else max(t["peak_cos"], cos)
                t["min_cos"] = cos if t["min_cos"] is None else min(t["min_cos"], cos)
                if z is not None:
                    z = float(z)
                    if t["start_z_mm"] is None:
                        t["start_z_mm"] = z
                    t["min_z_mm"] = z if t["min_z_mm"] is None else min(t["min_z_mm"], z)
            log = self._active_log
            summary = sample.get("summary")
        if log is not None:
            log.append("object", {"object": sample})
            if summary is not None:
                log.track(summary)
        elif summary is not None and run_id:
            # the run finished before the tracker did, which is the normal ordering:
            # the trace is written after the motion stops
            known = self._logs.get(run_id)
            if known is not None:
                known.track(summary)
        return self.state()["tracker"]

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
            except HomingAborted as exc:
                # A cancelled home/gantry move is a normal outcome of pressing stop, not
                # a fault -- but the axis was NOT zeroed, so the reference is gone.
                with self._lock:
                    self._homed = False
                    self._mounts_applied = False
                    self._manual_mounts = False
                    self._mount_positions = None
                    self._current_pose = None
                    self._unhomed_reason = f"cancelled part-way: {exc}"
                self._event("warning", f"{name} cancelled -- re-home before moving",
                            {"detail": str(exc)})
            except LinkDown as exc:
                error = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._last_error = error
                    self._link_down = str(exc)
                    self._homed = False
                    self._mounts_applied = False
                    self._manual_mounts = False
                    self._mount_positions = None
                    self._current_pose = None
                    self._unhomed_reason = "the serial link dropped"
                self._event("error", "serial link lost", {"error": error})
            except Exception as exc:  # surfaced in status/events; request already returned 202
                error = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._last_error = error
                    if name == "homing":
                        # Leaving _unhomed_reason at "homing in progress" made every
                        # later refusal quote a home that had already failed.
                        self._unhomed_reason = f"the last home failed: {error}"
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

    def _require_link(self) -> None:
        if self._link_down:
            raise RuntimeErrorState(
                f"serial link is down: {self._link_down} -- POST /api/v1/reconnect once the "
                f"board is back")

    def _require_homed_plan(self) -> None:
        self._require_idle()
        if not self._homed:
            detail = f" ({self._unhomed_reason})" if self._unhomed_reason else ""
            raise RuntimeErrorState(
                f"home the gantries once in this daemon session first{detail}")
        if self._plan is None:
            raise RuntimeErrorState("load a validated hand plan first")

    def _require_motion_ready(self) -> None:
        self._require_link()
        self._require_homed_plan()
        if self._servo_torque is not None and self._servo_torque != TORQUE_ON:
            raise RuntimeErrorState(
                "servo torque is not ON -- position writes would be accepted and read back "
                "correctly while nothing moves; enable torque first")
        if not (self._mounts_applied or self._manual_mounts):
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
        """Single reader for both links, with backoff.

        Three rules this enforces, each learned the hard way somewhere in this project:

        1. Servo reads never run while a writer owns the half-duplex bus. A trajectory
           frame that loses its slot to a telemetry packet is a timing defect in the
           motion, not just a slow read.
        2. Stepper reads never run while a gantry operation is in flight. STATALL is
           nine USB-CDC packets sent from the firmware's main loop, and the firmware is
           busy servicing step ISRs at a higher interrupt priority than USB at exactly
           that moment. The moving operation publishes its own per-axis STAT polls
           instead, so the UI still sees live positions.
        3. Consecutive failures back off and then suspend. A chain that has begun
           dropping packets gets worse when you poll it harder; this is the behaviour
           an operator coming from Dynamixel expects and the SCS0009 does not provide
           for itself.
        """
        if self.telemetry_hz <= 0:
            return
        period = 1.0 / self.telemetry_hz
        previous = None
        while not self._shutdown.is_set():
            start = time.monotonic()
            try:
                with self._lock:
                    if self._link_down:
                        self._shutdown.wait(2.0)
                        continue
                    idle = self._operation == "idle"
                    include_servos = idle
                    include_steppers = idle or self._operation not in (
                        "homing", "applying morphology")
                if not include_steppers and not include_servos:
                    self._shutdown.wait(period)
                    continue
                sample = self.backend.read_telemetry(include_servos=include_servos)
                now = time.time()
                with self._lock:
                    self._telemetry_failures = 0
                    count = int(self._telemetry["samples"]) + 1
                    measured = None if previous is None else 1.0 / max(1e-9, start - previous)
                    updated = dict(self._telemetry)
                    updated.update({"timestamp": now, "age_s": 0.0, "error": None,
                                    "samples": count, "measured_hz": measured,
                                    "consecutive_failures": 0,
                                    "servo_polling_suspended": not include_servos, **sample})
                    if "servos" in sample:
                        updated["servo_timestamp"] = now
                        updated["servo_age_s"] = 0.0
                    if isinstance(sample.get("servo_torque"), dict) and sample["servo_torque"]:
                        states = set(sample["servo_torque"].values())
                        torque = states.pop() if len(states) == 1 else None
                        # 0 is the servo's power-on default (servos.TORQUE_UNSET), not
                        # an explicit disable; either way it is not holding.
                        self._servo_torque = (TORQUE_OFF if torque == TORQUE_UNSET
                                               else torque)
                    self._telemetry = updated
                    if self._active_log is not None:
                        self._active_log.append("telemetry", {"phase": "during",
                                                              "data": sample})
                previous = start
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._telemetry_failures += 1
                    failures = self._telemetry_failures
                    self._telemetry["error"] = message
                    self._telemetry["consecutive_failures"] = failures
                    if isinstance(exc, LinkDown):
                        self._link_down = str(exc)
                        self._homed = False
                        self._mounts_applied = False
                        self._manual_mounts = False
                        self._mount_positions = None
                        self._unhomed_reason = "the serial link dropped"
                if isinstance(exc, LinkDown):
                    self._event("error", "serial link lost (telemetry)", {"error": message})
                elif failures == TELEMETRY_FAILURE_LIMIT:
                    self._event("warning",
                                f"telemetry has failed {failures} times in a row; backing off. "
                                f"Last error: {message}")
                # Do not hammer a sick packet bus: back off geometrically, capped.
                self._shutdown.wait(min(TELEMETRY_BACKOFF_CAP_S,
                                        max(period, 1.0) * min(failures, 8)))
                continue
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
