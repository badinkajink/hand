"""Regression tests for the control-station faults found on 2026-08-29.

Each test here corresponds to something that actually happened at the bench, or to a
device behaviour that made one of those things invisible. They run the REAL driver
stack -- MantaHandDriver, Joint, Gantry, Hand, ServoBus, RealHardwareBackend,
HandRuntime -- against manta_hand.fake_hardware, so they exercise the code that talks
to the hardware rather than a stand-in for it.

The incident, for context: an operator homed the hand from the web UI, became
suspicious of how long an axis was grinding, pressed "Stop motion", watched the home
continue anyway, then pressed "Move gantries to morphology" and lost the service. Live
inspection of the board afterwards showed it had never crashed.
"""

from __future__ import annotations

import http.client
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "src/morphohand/driver/manta/host"
sys.path.insert(0, str(HOST))

# ruff: noqa: E402
from manta_hand import servos as servos_module
from manta_hand.fake_hardware import TORQUE_OFF, TORQUE_ON, FakeM8P, FakeScs0009Controller
from manta_hand.kinematics import HomingAborted
from manta_hand.plan import HandPlan, Pose
from manta_hand.runtime import (HOME_CONFIRMATION, HandRuntime, LinkDown,
                                RealHardwareBackend, RuntimeErrorState)
from manta_hand.web import ControlHTTPServer


@pytest.fixture(autouse=True)
def fast_servo_bus(monkeypatch):
    """The real bus enforces a 0.3s gap after every individual call and a 1s settle on
    open. Both are empirical workarounds for real timeouts (see servos.py); neither is
    needed against a fake, and paying them would make this file take minutes."""
    monkeypatch.setattr(servos_module, "_controller_cls", lambda: FakeScs0009Controller)
    monkeypatch.setattr(servos_module, "INTER_CMD_DELAY_S", 0.0)
    monkeypatch.setattr(servos_module, "PORT_SETTLE_S", 0.0)


@pytest.fixture
def rig(tmp_path):
    """A RealHardwareBackend + HandRuntime on a fake M8P, torn down cleanly."""
    made: list = []

    def build(*, stall_axes=None, drop_lines=0, enable_servos=True, telemetry_hz=0.0,
              time_scale=0.004):
        device = FakeM8P(stall_axes=stall_axes, drop_lines=drop_lines,
                          time_scale=time_scale)
        backend = RealHardwareBackend(device.port, "fake://servos",
                                       enable_servos=enable_servos)
        runtime = HandRuntime(backend, logs_dir=tmp_path, telemetry_hz=telemetry_hz,
                              signs_checked=True)
        made.append((device, runtime))
        return device, backend, runtime

    yield build
    for device, runtime in made:
        try:
            runtime.close()
        finally:
            device.close()


def tiny_plan() -> HandPlan:
    zero = {f: {j: 0.0 for j in ("yaw", "mcp", "pip")}
            for f in ("thumb", "index", "middle")}
    grip = json.loads(json.dumps(zero))
    for f in grip:
        grip[f]["mcp"] = 5.0
    return HandPlan(design="test",
                    mounts_palm_mm={"thumb": (-42.5, 0.0), "index": (42.5, 40.0),
                                    "middle": (42.5, -40.0)},
                    poses=[Pose("open", 0.01, 0.0, zero), Pose("grip", 0.01, 0.0, grip)],
                    meta={})


def start_home_then_stop(device: FakeM8P, runtime: HandRuntime) -> None:
    """Begin a home and press stop while the FIRST axis is genuinely mid-travel.

    The rigs that use this run with a time_scale making one axis's homing window several
    seconds long, against the host's 0.3s HOME_POLL_PERIOD_S. Cancelling reliably in the
    middle of an axis rather than in the gap between two is the whole point: an axis
    interrupted mid-home is the one that must never be zeroed."""
    runtime.load_plan(tiny_plan())
    runtime.home(HOME_CONFIRMATION)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if any(c.startswith("HOME J") for c in device.commands) and device.axes[0].moving:
            break
        time.sleep(0.005)
    else:
        raise AssertionError("the first axis never started homing")
    runtime.stop()


def wait_idle(runtime: HandRuntime, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = runtime.state()
        if not state["busy"]:
            return state
        time.sleep(0.01)
    raise AssertionError(f"still busy: {runtime.state()['operation']}")


# ----------------------------------------------------------------------------------
# 1. The silent home: torque off means the goal register is written and nothing moves
# ----------------------------------------------------------------------------------
def test_torque_off_servo_accepts_a_write_and_does_not_move(rig):
    """The device behaviour that made the bug invisible. Asserted here so the fake
    cannot quietly stop modelling it."""
    _device, backend, _runtime = rig(enable_servos=False)
    controller = backend.servos.controller
    assert all(v == TORQUE_OFF for v in controller.torque_enable.values())

    backend.servos.finger(0).set_joint("fe1", 20.0)     # succeeds, verifies, no error
    assert controller.goal_position[1] != 0.0            # the register took the value
    assert controller.present_position[1] == 0.0         # the horn never moved


def test_backend_enables_servo_torque_on_connect(rig):
    """hand_control.py enables all nine before homing; the service must too."""
    _device, backend, _runtime = rig()
    assert backend.servos.controller.all_torque_on()


def test_home_refuses_to_run_with_torque_off_rather_than_silently_zeroing_nothing(rig):
    _device, backend, runtime = rig(enable_servos=False)
    runtime.load_plan(tiny_plan())
    runtime.home(HOME_CONFIRMATION)
    state = wait_idle(runtime)
    assert state["homed"] is False
    assert "torque" in state["last_error"].lower()


def test_home_actually_moves_the_servos_to_zero(rig):
    device, backend, runtime = rig()
    controller = backend.servos.controller
    backend.servos.finger(0).set_joint("fe1", 20.0)
    assert controller.present_position[1] != 0.0
    runtime.load_plan(tiny_plan())
    runtime.home(HOME_CONFIRMATION)
    state = wait_idle(runtime)
    assert state["homed"] is True
    # zero_joints drives every servo to its own calibrated zero, i.e. zero-relative 0.
    for finger_id, joints in servos_module.FINGER_JOINTS.items():
        for name, (sid, zero_deg, _limits) in joints.items():
            import math
            assert abs(math.degrees(controller.present_position[sid]) - zero_deg) < 0.5, (
                f"finger {finger_id} {name} did not reach its zero")
    assert device.commands.count("ZERO J0") == 1


# ----------------------------------------------------------------------------------
# 2. Stop must cancel a home, and a cancelled axis must never be zeroed
# ----------------------------------------------------------------------------------
def test_stop_cancels_homing_instead_of_running_the_remaining_axes(rig):
    device, _backend, runtime = rig(stall_axes=set(), time_scale=0.06)
    start_home_then_stop(device, runtime)
    state = wait_idle(runtime)

    homed_axes = [c for c in device.commands if c.startswith("HOME J")]
    assert len(homed_axes) < 6, f"stop did not stop the sequence: {homed_axes}"
    assert state["homed"] is False
    assert "cancel" in (state["unhomed_reason"] or "").lower()


def test_a_cancelled_axis_is_never_zeroed(rig):
    """The dangerous half. The firmware's STOP leaves homing_result at 1, the supervisor
    later flips it to 3, and the old poll loop read 3 as 'timeout guarantee -- safe to
    zero'. Zeroing an axis that never reached its hardstop puts step 0 in the middle of
    the rail, and the next MOVEMM drives away from that bogus origin with no stall
    protection at all."""
    device, _backend, runtime = rig(stall_axes=set(), time_scale=0.06)
    start_home_then_stop(device, runtime)
    wait_idle(runtime)
    assert not [c for c in device.commands if c.startswith("ZERO ")], (
        "the interrupted axis was zeroed anyway -- step 0 is now mid-rail")


def test_cancelled_axis_is_left_disabled_which_clears_the_firmware_homing_state(rig):
    device, _backend, runtime = rig(stall_axes=set(), time_scale=0.06)
    start_home_then_stop(device, runtime)
    wait_idle(runtime)
    assert any(c.startswith("DIS J") for c in device.commands), (
        "STOP alone leaves homing_result at 1; only DIS clears it")
    assert device.axes[0].homing_result == 0


def test_the_session_can_re_home_after_a_cancel(rig):
    """Aborting disables the axis, and Stepper_Home refuses ERR NODIAG on a disabled
    axis -- so without re-enabling, one cancelled home would end the session."""
    device, _backend, runtime = rig(stall_axes=set(), time_scale=0.06)
    start_home_then_stop(device, runtime)
    wait_idle(runtime)

    device.stall_axes = {0, 1, 2, 3, 4, 5}
    device.time_scale = 0.004
    runtime.home(HOME_CONFIRMATION, force=True)
    state = wait_idle(runtime)
    assert state["homed"] is True, state["last_error"]


def test_morphology_is_refused_after_a_cancelled_home_and_says_why(rig):
    device, _backend, runtime = rig(stall_axes=set(), time_scale=0.06)
    start_home_then_stop(device, runtime)
    wait_idle(runtime)
    with pytest.raises(RuntimeErrorState, match="cancel"):
        runtime.apply_morphology()


def test_a_daemon_restart_can_adopt_the_boards_home_without_re_homing(rig):
    """The M8P keeps step counters, SETSCALE and homing_result across a HOST restart --
    only a board reset or a DIS clears them. Re-homing then costs two minutes of the
    rails grinding for no information, on a hand that may be holding something."""
    device, backend, runtime = rig()
    runtime.load_plan(tiny_plan())
    runtime.home(HOME_CONFIRMATION)
    assert wait_idle(runtime)["homed"] is True
    runtime.apply_morphology()
    assert wait_idle(runtime)["mounts_applied"] is True

    # A fresh runtime on the SAME board, as after a daemon restart.
    fresh = HandRuntime(backend, logs_dir=runtime.logs_dir, signs_checked=True)
    try:
        assert fresh.state()["homed"] is False
        fresh.load_plan(tiny_plan())
        state = fresh.adopt_home()
        assert state["homed"] is True
        assert state["mounts_applied"] is True, "gantries were already on target"
        assert [o["joint"] for o in state["home_outcomes"]] == list(range(6))
    finally:
        fresh.close()


def test_adopting_is_refused_when_the_board_does_not_vouch(rig):
    device, backend, runtime = rig()
    runtime.load_plan(tiny_plan())
    with pytest.raises(RuntimeErrorState, match="does not vouch"):
        runtime.adopt_home()          # never homed: homing_result is 0 everywhere

    runtime.home(HOME_CONFIRMATION)
    wait_idle(runtime)
    runtime.disable_motors()          # DIS clears the firmware's homing_result
    fresh = HandRuntime(backend, logs_dir=runtime.logs_dir, signs_checked=True)
    try:
        with pytest.raises(RuntimeErrorState, match="does not vouch"):
            fresh.adopt_home()
    finally:
        fresh.close()


# ----------------------------------------------------------------------------------
# 3. The gantry move must be sequential
# ----------------------------------------------------------------------------------
def test_morphology_move_is_one_axis_at_a_time(rig):
    """Six MOVEMMs fired back to back means six simultaneous TMC5160 motion-start
    current kicks (IRUN 1 -> 7 for 500ms each) on the rail that also backfeeds the CB1,
    and six axes of step ISRs preempting the USB interrupt while the host polls that
    same link. Nothing validated on this hand has ever moved more than one axis at a
    time."""
    device, _backend, runtime = rig()
    runtime.load_plan(tiny_plan())
    runtime.home(HOME_CONFIRMATION)
    wait_idle(runtime)
    before = len(device.commands)
    runtime.apply_morphology()
    state = wait_idle(runtime)
    assert state["mounts_applied"] is True

    # Between consecutive MOVEMMs there must be at least one STAT for the axis just
    # commanded: that is the settle poll, and it is what proves nothing overlapped.
    after = device.commands[before:]
    moves = [i for i, c in enumerate(after) if c.startswith("MOVEMM J")]
    assert len(moves) == 6, after
    for first, second in zip(moves, moves[1:]):
        joint = after[first].split()[1]
        between = after[first + 1:second]
        assert any(c == f"STAT {joint}" for c in between), (
            f"no settle poll for {joint} between its move and the next: {between}")


def test_a_gantry_move_out_of_envelope_commands_nothing_at_all(rig):
    device, _backend, runtime = rig()
    runtime.load_plan(tiny_plan())
    runtime.home(HOME_CONFIRMATION)
    wait_idle(runtime)
    bad = tiny_plan()
    bad.mounts_palm_mm["thumb"] = (-42.5, 400.0)
    # load_plan validates, so reach past it the way a stale in-memory plan would.
    runtime._plan = bad
    before = len(device.commands)
    runtime.apply_morphology()
    wait_idle(runtime)
    after = device.commands[before:]
    assert not [c for c in after if c.startswith("MOVEMM J")], after
    assert runtime.state()["mounts_applied"] is False


# ----------------------------------------------------------------------------------
# 4. Link failure must latch, not be swallowed
# ----------------------------------------------------------------------------------
def test_a_board_that_stops_answering_latches_link_down_and_forces_a_re_home(rig):
    """The failure that ends with the BOOT0/RESET cycle. Previously this surfaced as a
    bare MantaHandError in last_error while the session still claimed to be homed with
    its mounts applied, so the next command acted on a reference that no longer
    existed."""
    device, _backend, runtime = rig()
    runtime.load_plan(tiny_plan())
    runtime.home(HOME_CONFIRMATION)
    assert wait_idle(runtime)["homed"] is True

    device.answering = False
    runtime.apply_morphology()
    state = wait_idle(runtime)
    assert state["link_down"], state
    assert state["homed"] is False
    with pytest.raises(RuntimeErrorState, match="link is down"):
        runtime.apply_morphology()

    device.answering = True
    assert runtime.reconnect()["link_down"] is None


def test_a_short_statall_reply_is_a_link_failure_not_a_silent_retry(rig):
    """cdc_send_blocking gives up on a busy IN endpoint and the reply arrives short --
    eight status lines promised, seven delivered."""
    _device, backend, _runtime = rig(drop_lines=1)
    with pytest.raises(LinkDown):
        backend.read_telemetry(include_servos=False)
    assert backend.link_error


def test_servo_bus_health_counts_timeouts_the_retry_would_otherwise_hide(rig):
    _device, backend, _runtime = rig()
    backend.servos.controller.dead_ids = {4}
    with pytest.raises(RuntimeError):
        backend.servos.finger(1).set_joint("fe1", 10.0)
    health = backend.servos.health()
    assert health["timeouts"] > 0
    assert health["consecutive_timeouts"] > 0


# ----------------------------------------------------------------------------------
# 5. Torque interlock and the motor kill switch
# ----------------------------------------------------------------------------------
def test_pose_is_refused_while_servo_torque_is_off(rig):
    _device, _backend, runtime = rig()
    runtime.load_plan(tiny_plan())
    runtime.home(HOME_CONFIRMATION)
    wait_idle(runtime)
    runtime.apply_morphology()
    wait_idle(runtime)
    runtime.set_servo_torque(TORQUE_OFF)
    with pytest.raises(RuntimeErrorState, match="torque"):
        runtime.move_to_pose("open")


def test_disable_motors_de_energises_everything_and_invalidates_the_home(rig):
    device, backend, runtime = rig()
    runtime.load_plan(tiny_plan())
    runtime.home(HOME_CONFIRMATION)
    assert wait_idle(runtime)["homed"] is True

    runtime.disable_motors()
    state = runtime.state()
    assert state["homed"] is False
    assert "disabled" in state["unhomed_reason"]
    assert all(not axis.enabled for axis in device.axes[:6])
    assert all(v == TORQUE_OFF for v in backend.servos.controller.torque_enable.values())


def test_the_trajectory_write_path_is_not_rate_limited(rig):
    """sync_set_joints is the only real-time path on the servo bus and must not wait
    out the inter-command gap.

    The gap exists because a transaction issued too soon AFTER a write fails (15% in
    measurement); it is not a rate limit on writes themselves, and consecutive sync
    writes are one bus transaction each. Applying it here silently capped a 50Hz
    trajectory at 3.33Hz -- the moves still happen, just 15x too slowly, which on an
    open-loop reorientation is a wrong experiment rather than an obvious failure."""
    _device, backend, _runtime = rig()
    pose = {0: {"fe1": 0.0}, 1: {"fe1": 0.0}, 2: {"fe1": 0.0}}
    backend.servos.sync_set_joints(pose, speed=80)      # first call may set speed
    start = time.monotonic()
    frames = 60
    for _ in range(frames):
        backend.servos.sync_set_joints(pose)
    elapsed = time.monotonic() - start
    rate = frames / elapsed
    assert rate > 200, (
        f"trajectory writes ran at {rate:.1f} Hz; the 50Hz replay path needs headroom "
        f"well above its own rate")


def test_a_read_still_waits_out_the_gap_after_a_write(rig):
    """The other half of the same rule: reads are free after reads, but a read that
    follows a write waits out POST_SYNC_WRITE_READ_GAP_S -- 2ms, measured, which leaves
    ~90Hz for a full write+read closed loop instead of the 3.3Hz the full
    INTER_CMD_DELAY_S would have allowed."""
    _device, backend, _runtime = rig()
    bus = backend.servos
    import manta_hand.servos as servos_real
    original = servos_real.POST_SYNC_WRITE_READ_GAP_S
    try:
        servos_real.POST_SYNC_WRITE_READ_GAP_S = 0.05
        bus.sync_set_joints({0: {"fe1": 0.0}})
        start = time.monotonic()
        bus.sync_read_joint_positions()
        assert time.monotonic() - start >= 0.04, "read did not wait out the write gap"
    finally:
        servos_real.POST_SYNC_WRITE_READ_GAP_S = original


# ----------------------------------------------------------------------------------
# 6. HTTP transport: the ways a browser sees "Failed to fetch" with no server error
# ----------------------------------------------------------------------------------
@pytest.fixture
def http_service(rig, tmp_path):
    _device, _backend, runtime = rig()
    server = ControlHTTPServer(("127.0.0.1", 0), runtime,
                               ROOT / "docs/experiments/20260829-real_v1_deploy/deploy",
                               control_token="secret")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()
    yield server.server_address
    server.shutdown()
    server.server_close()


def test_a_rejected_token_does_not_desync_the_keep_alive_connection(http_service):
    """A 401 that returns without reading the request body leaves those bytes in the
    socket; the next request line on that connection is then parsed out of the middle
    of the abandoned JSON. Every later request on the connection fails, which in a
    browser reads as "Failed to fetch" with nothing wrong on the server."""
    host, port = http_service
    conn = http.client.HTTPConnection(host, port, timeout=10)
    body = json.dumps({"confirmation": "HOME ALL AXES", "padding": "x" * 4000})
    conn.request("POST", "/api/v1/home", body=body,
                 headers={"Content-Type": "application/json", "X-Manta-Token": "wrong"})
    assert conn.getresponse().read() and True
    # Same connection, immediately afterwards: must be a clean, correct response.
    conn.request("GET", "/api/v1/state")
    response = conn.getresponse()
    assert response.status == 200
    assert json.loads(response.read())["schema_version"] == 1
    conn.close()


def test_state_and_events_stay_available_while_an_operation_runs(http_service, rig):
    """The UI polls twice a second throughout a two-minute home. If those polls block
    or fail, the operator loses sight of the hand at the exact moment they need it."""
    host, port = http_service
    conn = http.client.HTTPConnection(host, port, timeout=10)

    def get(path):
        conn.request("GET", path)
        response = conn.getresponse()
        payload = response.read()
        return response.status, payload

    for _ in range(20):
        status, payload = get("/api/v1/state")
        assert status == 200
        status, _ = get("/api/v1/events?after=0")
        assert status == 200
    conn.close()


def test_a_command_error_is_json_not_a_dropped_connection(http_service):
    host, port = http_service
    conn = http.client.HTTPConnection(host, port, timeout=10)
    conn.request("POST", "/api/v1/morphology", body="{}",
                 headers={"Content-Type": "application/json", "X-Manta-Token": "secret"})
    response = conn.getresponse()
    assert response.status == 409
    assert "home the gantries" in json.loads(response.read())["error"]
    conn.request("GET", "/api/v1/state")          # connection still usable
    assert conn.getresponse().status == 200
    conn.close()
