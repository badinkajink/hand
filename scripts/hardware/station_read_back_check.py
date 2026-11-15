#!/usr/bin/env python3
"""The station's stream with read-back on the fake hardware of `manta_hand.web --fake`: plan, homing, morphology, torque,
then a policy stream whose frames ask for the nine servo positions after each write (runtime.stream_frame read_back,
2026-11-15; HOM on the bench needs the joints every frame, and the telemetry loop does not read the servos during a
stream). Prints each frame's handling time and the read-back against the servo pose that was written.

    .venv/bin/python scripts/hardware/station_read_back_check.py [--plan docs/experiments/20260829-real_v1_deploy/deploy/g12_plan.json]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/sr2hand/driver/manta/host"))

import numpy as np  # noqa: E402

from manta_hand import plan as plan_module  # noqa: E402
from manta_hand import runtime as RT  # noqa: E402
from manta_hand import servos as servos_module  # noqa: E402
from manta_hand.fake_hardware import FakeM8P, FakeScs0009Controller  # noqa: E402


def wait_idle(rt, what, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = rt.state()
        if s.get("operation") in (None, "idle") and not s.get("busy"):
            return s
        time.sleep(0.05)
    raise SystemExit(f"{what}: not idle after {timeout:.0f} s: {rt.state().get('operation')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", default=str(ROOT / "docs/experiments/20260829-real_v1_deploy/deploy/g12_plan.json"))
    ap.add_argument("--frames", type=int, default=50)
    a = ap.parse_args()
    servos_module._controller_cls = lambda: FakeScs0009Controller
    servos_module.INTER_CMD_DELAY_S = 0.0
    servos_module.PORT_SETTLE_S = 0.0
    dev = FakeM8P(stall_axes={0, 1, 2, 4})
    backend = RT.RealHardwareBackend(dev.port, "fake://servos", servo_fields=())
    rt = RT.HandRuntime(backend, logs_dir=Path(tempfile.mkdtemp()), telemetry_hz=10.0, signs_checked=True)
    try:
        rt.load_plan(plan_module.HandPlan.from_json(Path(a.plan)))
        rt.home(RT.HOME_CONFIRMATION)
        wait_idle(rt, "homing")
        rt.apply_morphology()
        wait_idle(rt, "morphology")
        rt.set_servo_torque(RT.TORQUE_ON)
        token = rt.begin_stream(timeout_s=0.5)
        base = {"thumb": {"yaw": 0.0, "mcp": 20.0, "pip": 20.0}, "index": {"yaw": 0.0, "mcp": 20.0, "pip": 20.0},
                "middle": {"yaw": 0.0, "mcp": 20.0, "pip": 20.0}}
        ms, err = [], []
        for k in range(a.frames):
            pose = {f: {j: v + (5.0 * np.sin(2 * np.pi * k / 25.0) if j == "mcp" else 0.0) for j, v in d.items()}
                    for f, d in base.items()}
            t0 = time.perf_counter()
            res = rt.stream_frame(token, pose, read_back=True)
            ms.append(1e3 * (time.perf_counter() - t0))
            want = rt._servo_pose(rt._validate_sim_pose(pose))
            got = {int(fid): v for fid, v in res["servos"].items()}
            err.append(max(abs(got[fid][j] - want[fid][j]) for fid in want for j in want[fid]))
            time.sleep(0.02)
        plain = []
        for _ in range(10):
            t0 = time.perf_counter()
            assert rt.stream_frame(token, base) is None          # without read_back: the old answer
            plain.append(1e3 * (time.perf_counter() - t0))
        rt.end_stream(token)
        tel = rt.state().get("telemetry", {})
        print(f"{a.frames} frames with read-back: handling {np.median(ms):.2f} ms median, {max(ms):.2f} ms max; "
              f"read-back minus written servo pose at most {max(err):.4f} deg; without read-back "
              f"{np.median(plain):.2f} ms; telemetry servo_age_s after the stream {tel.get('servo_age_s')}")
    finally:
        rt.close() if hasattr(rt, "close") else None
        dev.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
