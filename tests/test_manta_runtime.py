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
    bad["thumb"]["yaw"] = 70.1
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
    assert rt.state()["current_pose"] == "zero"
    rt.move_to_pose("open", rate_hz=100)
    time.sleep(0.04)
    rt.stop()
    wait_idle(rt)
    assert rt.state()["current_pose"] == "zero"
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
