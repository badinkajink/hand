from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# The CB1 package is intentionally independent of the root project install.
# ruff: noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "src/morphohand/driver/manta/host"
sys.path.insert(0, str(HOST))

from manta_hand.plan import HandPlan, Pose
from manta_hand.runtime import (HOME_CONFIRMATION, HandRuntime, MockHardwareBackend,
                                RuntimeErrorState)
from manta_hand.servos import FINGER_JOINTS
from manta_hand.web import ControlHTTPServer


def tiny_plan() -> HandPlan:
    zero = {f: {j: 0.0 for j in ("yaw", "mcp", "pip")}
            for f in ("thumb", "index", "middle")}
    grip = json.loads(json.dumps(zero))
    end = json.loads(json.dumps(zero))
    for f in grip:
        grip[f]["mcp"] = 5.0
        end[f]["mcp"] = 6.0
    return HandPlan(
        design="test",
        mounts_palm_mm={"thumb": (-50.0, 0.0), "index": (50.0, 55.0),
                       "middle": (50.0, -55.0)},
        poses=[Pose("open", 0.01, 0.0, zero), Pose("grip", 0.01, 0.0, grip),
               Pose("turn_end", 0.02, 0.0, end)],
        meta={"predicted": "test-only"},
    )


def wait_idle(runtime: HandRuntime, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not runtime.state()["busy"]:
            return
        time.sleep(0.005)
    raise AssertionError(runtime.state())


def test_safe_sequence_and_run_log(tmp_path):
    backend = MockHardwareBackend()
    rt = HandRuntime(backend, logs_dir=tmp_path, signs_checked=True)
    rt.load_plan(tiny_plan())
    with pytest.raises(RuntimeErrorState, match="confirmation"):
        rt.home("yes")
    rt.home(HOME_CONFIRMATION)
    wait_idle(rt)
    with pytest.raises(RuntimeErrorState, match="already homed"):
        rt.home(HOME_CONFIRMATION)

    rt.apply_morphology()
    wait_idle(rt)
    assert backend.mounts["thumb"] == (-50.0, 0.0)
    rt.move_to_pose("open", rate_hz=100)
    wait_idle(rt)
    rt.move_to_pose("grip", rate_hz=100)
    wait_idle(rt)
    run_id = rt.run_reorientation(rate_hz=100)
    wait_idle(rt)

    summary = json.loads((tmp_path / f"{run_id}_SUMMARY.json").read_text())
    assert summary["status"] == "complete"
    assert summary["samples"]["command"] >= 1
    assert summary["settings"]["joint_signs"] == {
        "thumb": {"yaw": 1.0, "mcp": 1.0, "pip": 1.0},
        "index": {"yaw": -1.0, "mcp": 1.0, "pip": 1.0},
        "middle": {"yaw": -1.0, "mcp": 1.0, "pip": 1.0},
    }
    rows = [json.loads(x) for x in (tmp_path / f"{run_id}.jsonl").read_text().splitlines()]
    assert {x["kind"] for x in rows} >= {"command"}
    scored = rt.score_run(run_id, success=True, reorientation_deg=74, notes="held")
    assert scored["manual_score"]["success"] is True
    rt.close()


def test_yaw_sign_interlock_blocks_finger_motion(tmp_path):
    rt = HandRuntime(MockHardwareBackend(), logs_dir=tmp_path, signs_checked=False)
    rt.load_plan(tiny_plan())
    rt.home(HOME_CONFIRMATION)
    wait_idle(rt)
    rt.apply_morphology()
    wait_idle(rt)
    with pytest.raises(RuntimeErrorState, match="aa signs"):
        rt.move_to_pose("open")
    rt.close()


def test_policy_stream_has_limits_and_expires(tmp_path):
    backend = MockHardwareBackend()
    rt = HandRuntime(backend, logs_dir=tmp_path, signs_checked=True)
    rt.load_plan(tiny_plan())
    rt.home(HOME_CONFIRMATION)
    wait_idle(rt)
    rt.apply_morphology()
    wait_idle(rt)

    token = rt.begin_stream(timeout_s=0.1)
    rt.stream_frame(token, tiny_plan().poses[0].joints, servo_speed=80)
    bad = json.loads(json.dumps(tiny_plan().poses[0].joints))
    # Read the bound rather than restating it: this line said 70.1 until the aa cap was
    # restored to the declared +-85 on 2026-08-31, at which point it silently stopped
    # testing anything -- the value it pushed had become legal.
    bad["thumb"]["yaw"] = FINGER_JOINTS[0]["aa"][2][1] + 0.1
    with pytest.raises(ValueError, match="outside"):
        rt.stream_frame(token, bad)
    time.sleep(0.18)
    assert rt.state()["streaming"] is False
    assert rt.state()["operation"] == "idle"
    with pytest.raises(RuntimeErrorState, match="expired"):
        rt.stream_frame(token, tiny_plan().poses[0].joints)
    rt.close()


def test_stop_does_not_claim_an_interrupted_pose(tmp_path):
    plan = tiny_plan()
    plan.poses[0].ramp_s = 0.5
    rt = HandRuntime(MockHardwareBackend(), logs_dir=tmp_path, signs_checked=True)
    rt.load_plan(plan)
    rt.home(HOME_CONFIRMATION)
    wait_idle(rt)
    rt.apply_morphology()
    wait_idle(rt)
    # Applying a morphology moves steppers and commands no servo, so it does NOT leave
    # the hand at the zero pose -- see _do_apply_morphology. The pose is unknown here.
    assert rt.state()["current_pose"] is None
    rt.move_to_pose("open", rate_hz=100)
    time.sleep(0.04)
    rt.stop()
    wait_idle(rt)
    # and an interrupted pose must not claim to have arrived
    assert rt.state()["current_pose"] is None
    rt.close()


def test_mock_telemetry_is_cached(tmp_path):
    rt = HandRuntime(MockHardwareBackend(), logs_dir=tmp_path,
                     telemetry_hz=20.0, signs_checked=True)
    time.sleep(0.12)
    telemetry = rt.state()["telemetry"]
    assert telemetry["samples"] >= 1
    assert telemetry["servos"] is not None
    assert telemetry["age_s"] < 0.2
    rt.close()


def test_http_service_serves_ui_and_protects_commands(tmp_path):
    rt = HandRuntime(MockHardwareBackend(), logs_dir=tmp_path, signs_checked=True)
    plans = ROOT / "docs/experiments/20260829-real_v1_deploy/deploy"
    server = ControlHTTPServer(("127.0.0.1", 0), rt, plans, "secret")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/") as response:
            assert b"MorphoHand control" in response.read()
        with urllib.request.urlopen(base + "/api/v1/state") as response:
            assert json.load(response)["backend"] == "mock"

        request = urllib.request.Request(
            base + "/api/v1/plans/load",
            data=json.dumps({"file": "g12_plan.json"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(request)
        assert denied.value.code == 401
        request.add_header("X-Manta-Token", "secret")
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["plan"]["design"] == "g12"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        rt.close()


# ---- object tracking ---------------------------------------------------------------------
# Until 2026-08-31 this station had no object-pose sensor at all and every run was scored by
# an operator's eye. These cover the ingest path for the AprilTag tracker, whose one hard
# requirement is that a reading it never received can never be reported as one.
def test_object_pose_capability_follows_the_data_not_the_code(tmp_path):
    rt = HandRuntime(MockHardwareBackend(), logs_dir=tmp_path, signs_checked=True)
    try:
        assert rt.state()["capabilities"]["object_pose"] is False
        assert rt.state()["tracker"]["fresh"] is False
        rt.tracker_sample({"seen": True, "t": 0.0, "cos": 0.3, "z_bench_mm": 144.0})
        st = rt.state()
        assert st["capabilities"]["object_pose"]          # a string describing the sensor
        assert st["tracker"]["fresh"] is True
        assert st["tracker"]["age_s"] < 1.0
    finally:
        rt.close()


def test_tracker_extremes_are_per_run_and_signed(tmp_path):
    rt = HandRuntime(MockHardwareBackend(), logs_dir=tmp_path, signs_checked=True)
    try:
        for cos, z in ((0.02, 144.0), (0.55, 150.0), (0.41, 148.0)):
            t = rt.tracker_sample({"seen": True, "t": 0.0, "cos": cos,
                                   "z_bench_mm": z, "run_id": "a"})
        assert t["start_cos"] == 0.02 and t["peak_cos"] == 0.55 and t["min_cos"] == 0.02
        assert t["start_z_mm"] == 144.0 and t["min_z_mm"] == 144.0
        # a wrong-pole swing must lower min_cos, not raise the peak
        t = rt.tracker_sample({"seen": True, "t": 1.0, "cos": -0.7, "run_id": "a"})
        assert t["peak_cos"] == 0.55 and t["min_cos"] == -0.7
        # a different run starts over rather than inheriting the last one's peak
        t = rt.tracker_sample({"seen": True, "t": 0.0, "cos": 0.1, "run_id": "b"})
        assert t["start_cos"] == 0.1 and t["peak_cos"] == 0.1
    finally:
        rt.close()


def test_a_lost_tag_does_not_overwrite_the_last_good_reading(tmp_path):
    """The live display must not freeze on a stale pose and call it current: `seen` goes
    false while the numbers stay, so the operator sees the last position AND that it is no
    longer being measured."""
    rt = HandRuntime(MockHardwareBackend(), logs_dir=tmp_path, signs_checked=True)
    try:
        rt.tracker_sample({"seen": True, "t": 0.0, "cos": 0.61, "z_bench_mm": 150.0})
        t = rt.tracker_sample({"seen": False, "t": 0.1})
        assert t["last"]["seen"] is False
        assert t["last"]["cos"] == 0.61          # kept, but flagged as not current
        assert t["peak_cos"] == 0.61             # a lost frame is not a new extreme
    finally:
        rt.close()


def test_tracker_samples_land_in_the_running_log_and_the_summary(tmp_path):
    rt = HandRuntime(MockHardwareBackend(), logs_dir=tmp_path, signs_checked=True)
    try:
        rt.load_plan(tiny_plan())
        rt.home(HOME_CONFIRMATION)
        wait_idle(rt)
        rt.apply_morphology()
        wait_idle(rt)
        rt.move_to_pose("open")
        wait_idle(rt)
        rt.move_to_pose("grip")
        wait_idle(rt)
        run_id = rt.run_reorientation(rate_hz=100.0)
        for i in range(5):
            rt.tracker_sample({"seen": True, "t": i * 0.1, "cos": 0.1 * i,
                               "z_bench_mm": 144.0, "run_id": run_id})
        wait_idle(rt)
        rows = [json.loads(line) for line in
                (tmp_path / f"{run_id}.jsonl").read_text().splitlines()]
        obj = [r for r in rows if r["kind"] == "object"]
        assert len(obj) == 5
        assert obj[0]["object"]["cos"] == 0.0
        # the tracker finishes after the motion does; its summary still reaches the run
        rt.tracker_sample({"seen": False, "final": True, "run_id": run_id,
                           "summary": {"cos_peak": 0.4, "dropped": False}})
        summary = json.loads((tmp_path / f"{run_id}_SUMMARY.json").read_text())
        assert summary["object_track"] == {"cos_peak": 0.4, "dropped": False}
        assert summary["samples"]["object"] == 5
        # measured and operator-estimated stay separate readings of the same run
        assert summary["manual_score"] is None
    finally:
        rt.close()


def test_tracker_sample_is_reachable_over_http(tmp_path):
    rt = HandRuntime(MockHardwareBackend(), logs_dir=tmp_path, signs_checked=True)
    server = ControlHTTPServer(("127.0.0.1", 0), rt, tmp_path, control_token="tok")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/api/v1"
    try:
        req = urllib.request.Request(
            base + "/tracker/sample",
            data=json.dumps({"seen": True, "t": 0.0, "cos": 0.5}).encode(),
            headers={"Content-Type": "application/json", "X-Manta-Token": "tok"},
            method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            assert json.load(r)["received"] == 1
        with urllib.request.urlopen(base + "/state", timeout=5) as r:
            assert json.load(r)["tracker"]["last"]["cos"] == 0.5
    finally:
        server.shutdown()
        rt.close()
