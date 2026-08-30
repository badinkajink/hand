"""Dependency-free HTTP/JSON control service and static operator UI.

This intentionally uses the Python standard library so the CB1 does not need a web
framework toolchain.  The API is versioned; long hardware operations return 202 and are
observed through the cached ``/state`` endpoint.  CORS allows the same static UI to be
served on the workstation while targeting ``http://10.99.99.2:8765``.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import signal
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import plan as plan_module
from .plan import HandPlan
from .runtime import HandRuntime, MockHardwareBackend, RealHardwareBackend, RuntimeErrorState

STATIC_DIR = Path(__file__).with_name("static")


class ControlHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, runtime: HandRuntime, plans_dir: Path, control_token: str = ""):
        self.runtime = runtime
        self.plans_dir = plans_dir.resolve()
        self.control_token = control_token
        super().__init__(address, ControlRequestHandler)

    def plan_catalog(self) -> list[dict]:
        rows = []
        if not self.plans_dir.exists():
            return rows
        catalog = {}
        catalog_path = self.plans_dir / "catalog.json"
        if catalog_path.is_file():
            try:
                catalog = json.loads(catalog_path.read_text(encoding="utf-8")).get("designs", {})
            except (OSError, json.JSONDecodeError):
                catalog = {}
        for path in sorted(self.plans_dir.glob("*_plan.json")):
            try:
                plan = HandPlan.from_json(path)
                schema = plan.schema_errors()
                rows.append({"file": path.name, "design": plan.design, "meta": plan.meta,
                             "metrics": catalog.get(plan.design, {}),
                             "mounts_palm_mm": plan.mounts_palm_mm,
                             "poses": [p.name for p in plan.poses],
                             "violations": schema + [str(v) for v in plan.validate()]})
            except Exception as exc:
                rows.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
        return rows

    def resolve_plan(self, name: str) -> Path:
        path = (self.plans_dir / name).resolve()
        if path.parent != self.plans_dir or not path.name.endswith("_plan.json"):
            raise ValueError("plan must name one *_plan.json file in the configured plans directory")
        if not path.is_file():
            raise FileNotFoundError(path.name)
        return path


class ControlRequestHandler(BaseHTTPRequestHandler):
    server: ControlHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("manta-web: " + fmt % args + "\n")

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Manta-Token")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            query = parse_qs(parsed.query)
            if path == "/api/v1/state":
                return self._json(HTTPStatus.OK, self.server.runtime.state())
            if path == "/api/v1/events":
                after = int(query.get("after", ["0"])[0])
                return self._json(HTTPStatus.OK, {"events": self.server.runtime.events(after)})
            if path == "/api/v1/plans":
                return self._json(HTTPStatus.OK, {"plans": self.server.plan_catalog()})
            if path == "/api/v1/logs":
                return self._json(HTTPStatus.OK, {"logs": self.server.runtime.list_logs()})
            if path.startswith("/api/v1/logs/"):
                return self._serve_log(unquote(path.removeprefix("/api/v1/logs/")))
            return self._serve_static(path)
        except Exception as exc:
            self._error(exc)

    def do_POST(self):
        try:
            if (self.server.control_token and
                    self.headers.get("X-Manta-Token", "") != self.server.control_token):
                return self._json(HTTPStatus.UNAUTHORIZED, {"error": "missing or invalid control token"})
            path = urlparse(self.path).path.rstrip("/")
            body = self._body()
            rt = self.server.runtime
            if path == "/api/v1/plans/load":
                if "plan" in body:
                    raw = body["plan"]
                    plan = HandPlan(
                        design=raw["design"],
                        mounts_palm_mm={f: tuple(v) for f, v in raw["mounts_palm_mm"].items()},
                        poses=[plan_module.Pose(p["name"], float(p["ramp_s"]),
                                                float(p["hold_s"]), p["joints"])
                               for p in raw["poses"]],
                        meta=raw.get("meta", {}),
                    )
                else:
                    plan = HandPlan.from_json(self.server.resolve_plan(str(body["file"])))
                return self._json(HTTPStatus.OK, rt.load_plan(plan))
            if path == "/api/v1/home":
                rt.home(str(body.get("confirmation", "")), force=bool(body.get("force", False)))
                return self._json(HTTPStatus.ACCEPTED, {"accepted": True, "operation": "homing"})
            if path == "/api/v1/morphology":
                rt.apply_morphology()
                return self._json(HTTPStatus.ACCEPTED,
                                  {"accepted": True, "operation": "applying morphology"})
            if path == "/api/v1/pose":
                rt.move_to_pose(str(body["name"]), speed_ratio=float(body.get("speed_ratio", 1.0)),
                                rate_hz=float(body.get("rate_hz", 50.0)),
                                servo_speed=int(body.get("servo_speed", 80)))
                return self._json(HTTPStatus.ACCEPTED, {"accepted": True})
            if path == "/api/v1/reorient":
                run_id = rt.run_reorientation(speed_ratio=float(body.get("speed_ratio", 1.0)),
                                               rate_hz=float(body.get("rate_hz", 50.0)),
                                               servo_speed=int(body.get("servo_speed", 80)))
                return self._json(HTTPStatus.ACCEPTED, {"accepted": True, "run_id": run_id})
            if path == "/api/v1/stop":
                rt.stop()
                return self._json(HTTPStatus.OK, {"stopped": True})
            if path == "/api/v1/stream/start":
                token = rt.begin_stream(timeout_s=float(body.get("timeout_s", 0.25)))
                return self._json(HTTPStatus.OK, {"token": token})
            if path == "/api/v1/stream/frame":
                rt.stream_frame(str(body["token"]), body["joints"],
                                servo_speed=(int(body["servo_speed"])
                                             if body.get("servo_speed") is not None else None))
                return self._json(HTTPStatus.OK, {"accepted": True})
            if path == "/api/v1/stream/end":
                rt.end_stream(str(body["token"]))
                return self._json(HTTPStatus.OK, {"ended": True})
            if path.startswith("/api/v1/logs/") and path.endswith("/score"):
                run_id = unquote(path.removeprefix("/api/v1/logs/").removesuffix("/score"))
                if not isinstance(body.get("success"), bool):
                    raise ValueError("success must be a JSON boolean")
                summary = rt.score_run(run_id, success=body["success"],
                                       reorientation_deg=(float(body["reorientation_deg"])
                                                          if body.get("reorientation_deg") is not None
                                                          else None),
                                       notes=str(body.get("notes", "")))
                return self._json(HTTPStatus.OK, summary)
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
        except Exception as exc:
            self._error(exc)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 2_000_000:
            raise ValueError("request body exceeds 2 MB")
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("JSON request body must be an object")
        return value

    def _serve_static(self, url_path: str):
        name = "index.html" if url_path == "/" else unquote(url_path.lstrip("/"))
        path = (STATIC_DIR / name).resolve()
        if path.parent != STATIC_DIR.resolve() or not path.is_file():
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        data = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_log(self, name: str):
        if "/" in name or "\\" in name or not (name.endswith(".jsonl") or name.endswith("_SUMMARY.json")):
            raise ValueError("invalid log name")
        path = (self.server.runtime.logs_dir / name).resolve()
        if path.parent != self.server.runtime.logs_dir.resolve() or not path.is_file():
            raise FileNotFoundError(name)
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson" if name.endswith(".jsonl") else "application/json")
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: HTTPStatus, value: dict):
        data = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, exc: Exception):
        if isinstance(exc, (ValueError, KeyError, RuntimeErrorState)):
            status = HTTPStatus.CONFLICT if isinstance(exc, RuntimeErrorState) else HTTPStatus.BAD_REQUEST
        elif isinstance(exc, FileNotFoundError):
            status = HTTPStatus.NOT_FOUND
        else:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        self._json(status, {"error": f"{type(exc).__name__}: {exc}"})


def serve(runtime: HandRuntime, *, host: str = "0.0.0.0", port: int = 8765,
          plans_dir: str | Path = "docs/experiments/20260829-real_v1_deploy/deploy",
          control_token: str = "") -> None:
    server = ControlHTTPServer((host, port), runtime, Path(plans_dir), control_token)

    def stop_server(_signum=None, _frame=None):
        # shutdown() must run outside serve_forever's own thread.
        import threading
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    print(f"MorphoHand control: http://{host}:{port}  backend={runtime.backend.kind}")
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        runtime.close()


def _parse_signs(spec: str) -> dict[str, float]:
    out = {}
    if spec:
        for item in spec.split(","):
            name, value = item.split(":", 1)
            if name not in FINGER_NAMES or value not in ("+1", "1", "-1"):
                raise ValueError("--aa-signs format: thumb:+1,index:-1,middle:+1")
            out[name] = float(value)
    if out and set(out) != set(FINGER_NAMES):
        raise ValueError("--aa-signs must specify thumb, index, and middle")
    return out


FINGER_NAMES = ("thumb", "index", "middle")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mock", action="store_true", help="no hardware; exercise the full UI safely")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--stepper-port", default="/dev/ttyACM0")
    ap.add_argument("--servo-port", default="/dev/ttyUSB0")
    ap.add_argument("--plans-dir", type=Path,
                    default=Path("docs/experiments/20260829-real_v1_deploy/deploy"))
    ap.add_argument("--logs-dir", type=Path, default=Path("logs/hardware"))
    ap.add_argument("--telemetry-hz", type=float, default=0.0,
                    help="0 disables polling; benchmark before selecting a nonzero rate")
    ap.add_argument("--token", default="",
                    help="required X-Manta-Token for commands (mandatory with real hardware)")
    ap.add_argument("--aa-signs", default="",
                    help="override recorded sim->servo yaw signs after a new hardware check")
    args = ap.parse_args(argv)
    if not args.mock and not args.token:
        ap.error("--token is required with real hardware; generate one with: openssl rand -hex 16")

    signs = _parse_signs(args.aa_signs)
    if signs:
        for finger, sign in signs.items():
            plan_module.JOINT_SIGN[(finger, "yaw")] = sign
        plan_module.SIGNS_MEASURED = True
    backend = (MockHardwareBackend() if args.mock else
               RealHardwareBackend(args.stepper_port, args.servo_port))
    runtime = HandRuntime(backend, logs_dir=args.logs_dir,
                          telemetry_hz=args.telemetry_hz,
                          signs_checked=plan_module.SIGNS_MEASURED or bool(signs) or args.mock)
    serve(runtime, host=args.host, port=args.port, plans_dir=args.plans_dir,
          control_token=args.token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
