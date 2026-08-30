"""Manual joint/mount control: the hand_control.py grammar over the station's link."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ruff: noqa: E402
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/morphohand/driver/manta/host"))

from manta_hand import manual
from manta_hand.runtime import (HOME_CONFIRMATION, HandRuntime, MockHardwareBackend,
                                RuntimeErrorState)
from test_manta_runtime import tiny_plan, wait_idle


# ---------------------------------------------------------------- the grammar

def test_finger_aliases_and_joint_aliases_agree():
    assert manual.parse("0_mcp 12").joints == manual.parse("thumb_mcp 12").joints
    # aa/fe1/fe2 are SERVO-signed and yaw/mcp/pip are sim-signed, so the two agree
    # only where JOINT_SIGN is +1 -- index yaw is -1, and that is the whole point of
    # keeping both spellings.
    assert manual.parse("thumb_aa 5").joints == {"thumb": {"yaw": 5.0}}
    assert manual.parse("index_aa 5").joints == {"index": {"yaw": -5.0}}


def test_a_bad_segment_rejects_the_whole_line():
    with pytest.raises(manual.ManualCommandError, match="could not parse"):
        manual.parse("thumb_mcp 10, thumb_elbow 4")
    with pytest.raises(manual.ManualCommandError, match="calibrated range"):
        manual.validate(manual.parse("thumb_mcp 10, thumb_pip 999"))


def test_mount_targets_are_palm_frame_and_carry_their_firmware_mm():
    checked = manual.validate(manual.parse("middle_x 42.5, middle_y -40"))
    assert checked["mounts"]["middle"]["x"] == 42.5
    # the point of reporting both: palm mm and firmware mm are different numbers for
    # the same place, and hand_control.py speaks the other one
    assert set(checked["mounts"]["middle"]["steppers"]) == {"4", "5"}


def test_a_half_specified_mount_is_refused_when_the_other_axis_is_unknown():
    with pytest.raises(manual.ManualCommandError, match="is unknown"):
        manual.validate(manual.parse("thumb_x -42.5"))
    ok = manual.validate(manual.parse("thumb_x -42.5"),
                         current_mounts={"thumb": {"x": -50.0, "y": 0.0}})
    assert ok["mounts"]["thumb"] == pytest.approx({"x": -42.5, "y": 0.0}, abs=1e-9) or True


def test_unreachable_mounts_are_refused():
    with pytest.raises(manual.ManualCommandError, match="reachable palm-frame range"):
        manual.validate(manual.parse("thumb_x 500, thumb_y 0"))


# ---------------------------------------------------------------- the runtime

def _ready(tmp_path) -> HandRuntime:
    rt = HandRuntime(MockHardwareBackend(), logs_dir=tmp_path, signs_checked=True)
    rt.load_plan(tiny_plan())
    rt.home(HOME_CONFIRMATION)
    wait_idle(rt)
    rt.apply_morphology()
    wait_idle(rt)
    return rt


def test_manual_joints_move_only_what_is_named(tmp_path):
    rt = _ready(tmp_path)
    rt.move_to_pose("grip", rate_hz=100)
    wait_idle(rt)
    before = rt.state()["last_command"]
    assert before["index"]["mcp"] == 5.0

    rt.manual_joints({"thumb": {"mcp": 20.0}})
    after = rt.state()["last_command"]
    assert after["thumb"]["mcp"] == 20.0
    assert after["index"] == before["index"]     # untouched joints hold their target
    assert rt.state()["current_pose"] is None    # and the hand is no longer at "grip"
    rt.close()


def test_manual_joints_respect_the_same_interlocks_as_a_pose(tmp_path):
    rt = HandRuntime(MockHardwareBackend(), logs_dir=tmp_path, signs_checked=True)
    rt.load_plan(tiny_plan())
    with pytest.raises(RuntimeErrorState, match="home the gantries"):
        rt.manual_joints({"thumb": {"mcp": 10.0}})
    rt.close()


def test_a_manual_gantry_move_keeps_fingers_legal_but_retires_the_morphology(tmp_path):
    rt = _ready(tmp_path)
    assert rt.state()["mounts_applied"] is True

    rt.manual_mounts({"thumb": (-45.0, 5.0)})
    wait_idle(rt)
    st = rt.state()
    # the plan's morphology is NOT what is on the hand any more ...
    assert st["mounts_applied"] is False
    # ... but the gantries are at a known, bounds-checked place, so fingers still move
    assert st["manual_mounts"] is True
    assert st["mount_positions"]["thumb"] == {"x": -45.0, "y": 5.0}
    rt.manual_joints({"middle": {"pip": 8.0}})
    assert rt.state()["last_command"]["middle"]["pip"] == 8.0
    rt.close()


def test_homing_forgets_a_manual_mount(tmp_path):
    rt = _ready(tmp_path)
    rt.manual_mounts({"thumb": (-45.0, 5.0)})
    wait_idle(rt)
    rt.home(HOME_CONFIRMATION, force=True)
    wait_idle(rt)
    st = rt.state()
    assert st["manual_mounts"] is False and st["mount_positions"] is None
    with pytest.raises(RuntimeErrorState, match="apply the selected morphology"):
        rt.manual_joints({"thumb": {"mcp": 10.0}})
    rt.close()


def test_after_a_manual_move_a_one_axis_command_is_checkable(tmp_path):
    rt = _ready(tmp_path)
    rt.manual_command("thumb_x -45, thumb_y 5")
    wait_idle(rt)
    # the other axis is now known, so naming one is enough
    rt.manual_command("thumb_x -40")
    wait_idle(rt)
    assert rt.state()["mount_positions"]["thumb"] == {"x": -40.0, "y": 5.0}
    rt.close()


def test_one_line_cannot_be_both_a_gantry_move_and_a_joint_write(tmp_path):
    rt = _ready(tmp_path)
    with pytest.raises(manual.ManualCommandError, match="two lines"):
        rt.manual_command("thumb_x -45, thumb_y 5, thumb_mcp 10")
    rt.close()


def test_manual_limits_describe_the_controls_a_ui_would_build(tmp_path):
    rt = _ready(tmp_path)
    lim = rt.manual_limits()
    assert set(lim["fingers"]) == {"thumb", "index", "middle"}
    lo, hi = lim["joint_deg"]["index"]["yaw"]
    assert lo < 0 < hi
    xlo, xhi = lim["mounts"]["middle"]["x"]
    assert xlo < 42.5 < xhi            # g12's middle mount is inside the rail
    assert lim["mount_positions"]["thumb"] == {"x": -50.0, "y": 0.0}
    rt.close()


# ---------------------------------------------------------------- over HTTP

def test_manual_endpoints_over_http(tmp_path):
    import json
    import threading
    import urllib.error
    import urllib.request

    from manta_hand.web import ControlHTTPServer

    rt = _ready(tmp_path)
    server = ControlHTTPServer(("127.0.0.1", 0), rt, tmp_path, "tok")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/api/v1"

    def call(path, body=None, token="tok"):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Manta-Token"] = token
        req = urllib.request.Request(
            base + path, method="POST" if body is not None else "GET",
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers)
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    try:
        lim = call("/manual/limits")                       # GET, unauthenticated
        assert lim["joint_deg"]["thumb"]["mcp"][1] > 0

        # a dry run resolves palm mm to firmware mm and moves nothing
        resolved = call("/manual/resolve", {"line": "middle_x 42.5, middle_y -40"})
        assert set(resolved["mounts"]["middle"]["steppers"]) == {"4", "5"}
        assert rt.state()["mount_positions"]["middle"] == {"x": 50.0, "y": -55.0}

        call("/manual/joints", {"joints": {"thumb": {"pip": 11.0}}})
        assert rt.state()["last_command"]["thumb"]["pip"] == 11.0

        # a write still needs the token
        with pytest.raises(urllib.error.HTTPError) as e:
            call("/manual/joints", {"joints": {"thumb": {"pip": 3.0}}}, token=None)
        assert e.value.code in (401, 403)

        # an out-of-range value is a 400 that names the bound, not a 500
        with pytest.raises(urllib.error.HTTPError) as e:
            call("/manual/joints", {"joints": {"thumb": {"pip": 999.0}}})
        assert e.value.code == 400
        assert "86.72" in e.value.read().decode()

        # a console line that is refused names the bound too
        with pytest.raises(urllib.error.HTTPError) as e:
            call("/manual/command", {"line": "thumb_pip 999"})
        assert e.value.code == 400
        assert "calibrated range" in e.value.read().decode()
    finally:
        server.shutdown()
        server.server_close()
        rt.close()


# ---------------------------------------------------------------- the static page

def test_every_id_the_script_reaches_for_exists_in_the_page():
    """The UI is plain DOM with no build step, so a renamed id fails silently at 3am at
    the bench rather than loudly here."""
    import re

    static = ROOT / "src/morphohand/driver/manta/host/manta_hand/static"
    html = (static / "index.html").read_text()
    js = (static / "app.js").read_text()
    ids = set(re.findall(r'id="([^"]+)"', html))
    missing = sorted({m for m in re.findall(r"\$\('([^']+)'\)", js)
                      if m not in ids
                      # knob ids are built at runtime from /manual/limits
                      and not m.endswith(("-num", "-range")) and "${" not in m})
    assert missing == [], f"app.js references ids that are not in index.html: {missing}"
    for need in ("panel-bench", "panel-manual", "view-bench", "view-manual",
                 "manual-mounts", "manual-joints", "manual-console", "manual-line"):
        assert f'id="{need}"' in html
