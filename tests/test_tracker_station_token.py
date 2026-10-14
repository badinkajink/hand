"""The tracker service must reject a bad station token at launch, not at trial time.

On 2026-09-09 the service ran with an empty ``--station-token``. It started, reported itself
healthy, and the station answered **401** to all 153 samples it pushed. The station therefore
saw no object samples, the fail-closed ``automatic-required`` interlock refused the turn, and
the run failed 20 s in with the cylinder already grasped. The probe these tests cover turns
that into a one-second launch-time error.
"""
from __future__ import annotations

import json
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("real_v1_tag_tracker", reason="camera stack absent")
from real_v1_tracker_service import probe_station_token  # noqa: E402


class _Station(BaseHTTPRequestHandler):
    """The station's auth shape: every POST is authorised BEFORE the path is dispatched."""

    token = "sekrit"

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        if self.token and self.headers.get("X-Manta-Token", "") != self.token:
            body, status = {"error": "missing or invalid control token"}, HTTPStatus.UNAUTHORIZED
        else:
            body, status = {"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):  # keep pytest output clean
        pass


@pytest.fixture()
def station():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Station)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/api/v1"
    srv.shutdown()
    srv.server_close()


def test_correct_token_is_accepted(station):
    # 404 means the token passed the gate and only the probe path was unknown.
    assert probe_station_token(station, "sekrit") == ("ok", "station answered 404")


def test_empty_token_is_caught(station):
    # The exact 2026-09-09 configuration: MANTA_TOKEN unset.
    verdict, _ = probe_station_token(station, "")
    assert verdict == "unauthorized"


def test_wrong_token_is_caught(station):
    verdict, _ = probe_station_token(station, "not-the-token")
    assert verdict == "unauthorized"


def test_unreachable_station_is_not_a_token_verdict():
    # A down station must not be reported as a bad token; the service warns and serves.
    verdict, _ = probe_station_token("http://127.0.0.1:9/api/v1", "sekrit", timeout=1.0)
    assert verdict == "unreachable"


def test_probe_writes_nothing(station):
    """The probe is safe to run on every arm: it must not reach a real endpoint."""
    seen: list[str] = []
    original = _Station.do_POST

    def spy(self):
        seen.append(self.path)
        original(self)

    _Station.do_POST = spy
    try:
        probe_station_token(station, "sekrit")
    finally:
        _Station.do_POST = original
    assert seen == ["/api/v1/__token_check"]
    assert not any(p.endswith(("/tracker/sample", "/reorient", "/plans/load")) for p in seen)
