"""Dependency-free HTTP/JSON control service and static operator UI.

This intentionally uses the Python standard library so the CB1 does not need a web
framework toolchain.  The API is versioned; long hardware operations return 202 and are
observed through the cached ``/state`` endpoint.  CORS allows the same static UI to be
served on the workstation while targeting ``http://10.99.99.2:8765``.
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import mimetypes
import signal
import sys
import threading
import traceback
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import plan as plan_module
from .plan import HandPlan
from .runtime import (HandRuntime, MockHardwareBackend, RealHardwareBackend,
                       RuntimeErrorState)
from .servos import TORQUE_FREE, TORQUE_OFF, TORQUE_ON

STATIC_DIR = Path(__file__).with_name("static")
LOG = logging.getLogger("manta_hand.web")


class HTTPTrackerController:
    """Synchronous control of the workstation camera service.

    Calls happen inside the reorientation worker, never a request handler: ``start`` may
    legitimately take a few seconds while IR auto-exposure settles and id6 is latched.
    Keeping only cached state here preserves the runtime's cheap, non-blocking /state reads.
    """

    def __init__(self, url: str, token: str = "", timeout_s: float = 30.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._state = {"configured": True, "url": self.url, "running": False,
                       "armed": False, "run_id": "", "error": None}

    def _post(self, path: str, run_id: str, timeout: float) -> dict:
        req = urllib.request.Request(
            self.url + path, json.dumps({"run_id": run_id}).encode(), method="POST",
            headers={"Content-Type": "application/json", "X-Tracker-Token": self.token})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("error")
            except Exception:
                detail = str(exc)
            raise RuntimeError(f"workstation tracker {path} failed: {detail}") from None
        except OSError as exc:
            raise RuntimeError(self._unreachable(exc)) from None

    def _unreachable(self, exc: OSError) -> str:
        """Say which of the two failures this is, and what to type.

        An operator reads this at the bench with the hand in the grip pose, so the message
        has to name the fix rather than the errno. Connection REFUSED means the workstation
        answered and nothing is listening -- the service is simply not running, which is
        where it ends up after a workstation reboot or a closed terminal. Anything else
        (no route, timed out) is the hand network itself, and the interface is the suspect."""
        refused = isinstance(exc, ConnectionRefusedError) or getattr(exc, "errno", None) == 111
        if refused:
            return (f"the workstation tracker at {self.url} is not running (connection "
                    f"refused). Start it on the workstation and retry:  PYTHONPATH= "
                    f"MANTA_TOKEN=... ~/miniconda3/bin/python "
                    f"scripts/real_v1_tracker_service.py")
        return (f"cannot reach the workstation tracker at {self.url}: {exc}. The service may "
                f"be up but unreachable -- check the hand-network interface on the "
                f"workstation (ip addr add 10.99.99.50/24 dev <iface>)")

    def start(self, run_id: str) -> dict:
        try:
            result = self._post("/start", run_id, self.timeout_s)
        except Exception as exc:
            with self._lock:
                self._state.update({"running": False, "armed": False, "run_id": run_id,
                                    "error": str(exc)})
            raise
        with self._lock:
            self._state.update(result)
            self._state["error"] = None
            return dict(self._state)

    def stop(self, run_id: str) -> dict:
        try:
            result = self._post("/stop", run_id, max(self.timeout_s, 40.0))
        except Exception as exc:
            with self._lock:
                self._state["error"] = str(exc)
            raise
        with self._lock:
            self._state.update(result)
            return dict(self._state)

    def state(self) -> dict:
        with self._lock:
            return dict(self._state)


def configure_logging(log_file: Path | None, verbose: bool = False) -> None:
    """Console plus, when asked, a rotating file.

    The file matters more than it looks. This service is normally started in an
    interactive SSH shell on the CB1, so its only record of what happened lives in that
    terminal's scrollback -- which is gone the moment the session drops, exactly when
    something has gone wrong and the traceback is what you need. Point --log-file at the
    repo's logs/ directory and the next incident leaves evidence."""
    root = logging.getLogger("manta_hand")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    console.setLevel(logging.DEBUG if verbose else logging.WARNING)
    root.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=8_000_000,
                                                        backupCount=3, encoding="utf-8")
        handler.setFormatter(fmt)
        handler.setLevel(logging.DEBUG)
        root.addHandler(handler)


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
                             # keyed by FILE STEM first: the same design now ships several
                             # plans that differ only in residual clip (sv1_u0060_b75 and
                             # sv1_u0060_b100 are one hand at two clips, and they behave
                             # nothing alike), so metrics keyed by design alone would show
                             # the same numbers under both.
                             "metrics": catalog.get(path.name.removesuffix("_plan.json"),
                                                    catalog.get(plan.design, {})),
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
    # Without this a keep-alive connection that the browser abandons (page reload, a
    # network blip, a laptop lid) holds its handler thread forever. Over a long bench
    # session on a CB1 those accumulate with nothing to reap them.
    timeout = 30.0

    def setup(self):
        super().setup()
        self._responded = False

    def log_message(self, fmt, *args):
        # Goes through logging, not straight to stderr: a foreground service on the CB1
        # writes to whatever pty started it, and a blocked or closed pty then blocks or
        # kills the service. The file handler installed in serve() is what survives the
        # SSH session that launched it.
        LOG.info("%s %s", self.address_string(), fmt % args)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up mid-response. Normal with a polling UI; not an error,
            # and definitely not something to print a traceback about on every refresh.
            self.close_connection = True

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
            if path == "/api/v1/manual/limits":
                return self._json(HTTPStatus.OK, self.server.runtime.manual_limits())
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
            path = urlparse(self.path).path.rstrip("/")
            # Read the body BEFORE any early return. On a keep-alive connection an
            # unread request body is left in the socket and the next request line is
            # parsed out of the middle of it -- the connection then desyncs and every
            # later request on it fails, which in a browser reads as "Failed to fetch"
            # with no server-side error at all.
            body = self._body()
            if (self.server.control_token and
                    self.headers.get("X-Manta-Token", "") != self.server.control_token):
                return self._json(HTTPStatus.UNAUTHORIZED,
                                  {"error": "missing or invalid control token"})
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
                    plan_path = self.server.resolve_plan(str(body["file"]))
                    plan = HandPlan.from_json(plan_path)
                    # Stamp which FILE was read. `design` is the morphology tag, and several
                    # exported plans share one -- g12 ships at four residual clips under the
                    # single design "g12" -- so without this a run log cannot say which of
                    # them ran, and the run id (stamp-design-uuid) cannot either.
                    plan.meta["plan_file"] = plan_path.name
                return self._json(HTTPStatus.OK, rt.load_plan(plan))
            if path == "/api/v1/home":
                rt.home(str(body.get("confirmation", "")), force=bool(body.get("force", False)))
                return self._json(HTTPStatus.ACCEPTED, {"accepted": True, "operation": "homing"})
            if path == "/api/v1/home/adopt":
                return self._json(HTTPStatus.OK, rt.adopt_home())
            if path == "/api/v1/morphology":
                rt.apply_morphology()
                return self._json(HTTPStatus.ACCEPTED,
                                  {"accepted": True, "operation": "applying morphology"})
            if path == "/api/v1/manual/joints":
                joints = body.get("joints")
                if not isinstance(joints, dict) or not joints:
                    raise ValueError("joints must be a non-empty object of "
                                     "{finger: {yaw|mcp|pip: degrees}}")
                speed = body.get("servo_speed")
                return self._json(HTTPStatus.OK, {"commanded": rt.manual_joints(
                    joints, servo_speed=(int(speed) if speed is not None else None))})
            if path == "/api/v1/manual/mounts":
                mounts = body.get("mounts")
                if not isinstance(mounts, dict) or not mounts:
                    raise ValueError("mounts must be a non-empty object of "
                                     "{finger: {x: mm, y: mm}} in the palm frame")
                targets = {str(f): (float(v["x"]), float(v["y"])) for f, v in mounts.items()}
                return self._json(HTTPStatus.ACCEPTED,
                                  {"accepted": True, "targets": rt.manual_mounts(targets)})
            if path == "/api/v1/manual/resolve":
                return self._json(HTTPStatus.OK, rt.manual_resolve(str(body.get("line", ""))))
            if path == "/api/v1/manual/command":
                speed = body.get("servo_speed")
                return self._json(HTTPStatus.OK, rt.manual_command(
                    str(body.get("line", "")),
                    servo_speed=(int(speed) if speed is not None else None)))
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
            if path == "/api/v1/motors/disable":
                return self._json(HTTPStatus.OK, rt.disable_motors())
            if path == "/api/v1/servos/torque":
                names = {"on": TORQUE_ON, "off": TORQUE_OFF, "free": TORQUE_FREE}
                requested = str(body.get("state", "")).lower()
                if requested not in names:
                    raise ValueError(f"state must be one of {sorted(names)}")
                return self._json(HTTPStatus.OK,
                                  {"readback": rt.set_servo_torque(names[requested])})
            if path == "/api/v1/reconnect":
                return self._json(HTTPStatus.OK, rt.reconnect())
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
            if path == "/api/v1/tracker/sample":
                return self._json(HTTPStatus.OK, rt.tracker_sample(body))
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
        self._responded = True
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
        self._responded = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson" if name.endswith(".jsonl") else "application/json")
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: HTTPStatus, value: dict):
        try:
            data = json.dumps(value, allow_nan=False).encode("utf-8")
        except (ValueError, TypeError) as exc:
            # A non-finite or unserialisable value anywhere in the status document used
            # to raise HERE, after the caller had already decided to respond -- the
            # handler then unwound with nothing written and the browser saw a dead
            # connection. Degrade to a describable error instead of losing the response.
            LOG.exception("response for %s was not serialisable", self.path)
            data = json.dumps({"error": f"response not serialisable: {exc}"}).encode("utf-8")
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        self._responded = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, exc: Exception):
        if self._responded:
            # Something failed after the response was already on the wire. Writing a
            # second one would desync a keep-alive connection and break every request
            # that follows it, so close instead and leave the evidence in the log.
            LOG.error("error after response was sent for %s: %s", self.path, exc)
            self.close_connection = True
            return
        if isinstance(exc, (ValueError, KeyError, RuntimeErrorState)):
            status = HTTPStatus.CONFLICT if isinstance(exc, RuntimeErrorState) else HTTPStatus.BAD_REQUEST
        elif isinstance(exc, FileNotFoundError):
            status = HTTPStatus.NOT_FOUND
        else:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            LOG.error("unhandled error serving %s\n%s", self.path, traceback.format_exc())
        try:
            self._json(status, {"error": f"{type(exc).__name__}: {exc}"})
        except OSError:
            self.close_connection = True


def serve(runtime: HandRuntime, *, host: str = "0.0.0.0", port: int = 8765,
          plans_dir: str | Path = "docs/experiments/20260829-real_v1_deploy/deploy",
          control_token: str = "") -> None:
    try:
        server = ControlHTTPServer((host, port), runtime, Path(plans_dir), control_token)
    except OSError as exc:
        # Starting a second instance is the most common way to get here, and the two
        # serial links are exclusive, so the running one is the one that owns the hand.
        runtime.close()
        raise SystemExit(
            f"cannot bind {host}:{port}: {exc}\n"
            f"Another control service is probably already running and holding the "
            f"serial ports. Find it with:  pgrep -af manta_hand"
        ) from exc

    def stop_server(_signum=None, _frame=None):
        # shutdown() must run outside serve_forever's own thread.
        import threading
        threading.Thread(target=server.shutdown, daemon=True).start()

    # SIGHUP alongside INT/TERM: this is normally started in a foreground SSH shell, and
    # the shell dying is the single most common way this service has ended. Handling it
    # means the hand is stopped and de-energised on the way out rather than left holding
    # its last goal.
    #
    # signal.signal() only works on the main thread, and serve() is worth calling from a
    # worker in tests and embedding harnesses -- so a failure here costs the clean
    # shutdown, not the service.
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, stop_server)
        except ValueError:
            LOG.debug("not on the main thread; %s will not be handled", name)
    LOG.info("serving on %s:%s backend=%s", host, port, runtime.backend.kind)
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
    ap.add_argument("--mock", action="store_true",
                    help="no hardware, no driver code: swaps the whole backend for a stub")
    ap.add_argument("--fake", action="store_true",
                    help="no hardware, but the REAL driver stack against a simulated M8P on a "
                         "pty and a simulated SCS0009 bus. Use this, not --mock, to reproduce "
                         "and debug anything involving homing, gantry motion or the serial link")
    ap.add_argument("--fake-stall-axes", default="0,1,2,4",
                    help="--fake only: which axes' StallGuard2 fires. The default matches the "
                         "real hand as measured 2026-08-29 (J3 and J5 home by timeout)")
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
    ap.add_argument("--tracker-url", default="",
                    help="workstation tracker service (for example http://10.99.99.50:8770). "
                         "When set, every reorientation must arm tracking before motion")
    ap.add_argument("--tracker-token", default="",
                    help="optional X-Tracker-Token shared with the workstation service")
    ap.add_argument("--tracker-timeout", type=float, default=30.0,
                    help="seconds allowed for the workstation to settle exposure and latch id6")
    ap.add_argument("--aa-signs", default="",
                    help="override recorded sim->servo yaw signs after a new hardware check")
    ap.add_argument("--log-file", type=Path, default=None,
                    help="rotating log of every request, error and traceback. Strongly "
                         "recommended on the CB1: without it the only record is the SSH "
                         "session's scrollback, which is gone exactly when you need it")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--servo-fields", default="",
                    help="extra per-servo feedback to sample with each telemetry frame, "
                         "comma separated (load, voltage, temperature, speed). Each costs "
                         "another nine bus transactions; 'load' is the only force-ish proxy "
                         "this part has and it is UNCALIBRATED -- not amps, not newtons")
    args = ap.parse_args(argv)
    if args.mock and args.fake:
        ap.error("--mock and --fake are different things; pick one (see their help text)")
    simulated = args.mock or args.fake
    if not simulated and not args.token:
        ap.error("--token is required with real hardware; generate one with: openssl rand -hex 16")

    configure_logging(args.log_file, args.verbose)
    servo_fields = tuple(f.strip() for f in args.servo_fields.split(",") if f.strip())
    unknown = sorted(set(servo_fields) - {"load", "voltage", "temperature", "speed"})
    if unknown:
        ap.error(f"unknown --servo-fields: {', '.join(unknown)}")
    signs = _parse_signs(args.aa_signs)
    if signs:
        for finger, sign in signs.items():
            plan_module.JOINT_SIGN[(finger, "yaw")] = sign
        plan_module.SIGNS_MEASURED = True

    fake_device = None
    if args.mock:
        backend = MockHardwareBackend()
    elif args.fake:
        from . import servos as servos_module
        from .fake_hardware import FakeM8P, FakeScs0009Controller

        servos_module._controller_cls = lambda: FakeScs0009Controller
        servos_module.INTER_CMD_DELAY_S = 0.0
        servos_module.PORT_SETTLE_S = 0.0
        stall = {int(x) for x in args.fake_stall_axes.split(",") if x.strip() != ""}
        fake_device = FakeM8P(stall_axes=stall)
        print(f"FAKE hardware: simulated M8P on {fake_device.port}, "
              f"StallGuard2 fires on axes {sorted(stall)}")
        backend = RealHardwareBackend(fake_device.port, "fake://servos",
                                       servo_fields=servo_fields)
    else:
        backend = RealHardwareBackend(args.stepper_port, args.servo_port,
                                       servo_fields=servo_fields)

    tracker_controller = (HTTPTrackerController(args.tracker_url, args.tracker_token,
                                                args.tracker_timeout)
                          if args.tracker_url else None)
    runtime = HandRuntime(backend, logs_dir=args.logs_dir,
                          telemetry_hz=args.telemetry_hz,
                          signs_checked=plan_module.SIGNS_MEASURED or bool(signs) or simulated,
                          tracker_controller=tracker_controller)
    try:
        serve(runtime, host=args.host, port=args.port, plans_dir=args.plans_dir,
              control_token=args.token)
    finally:
        if fake_device is not None:
            fake_device.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
