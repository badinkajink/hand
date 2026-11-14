"""Time per step of the HOM controller on corvus's CPU, through the bench adapter (sr2hand.control.hom_bench).

For each task a short closed-loop simulation on the pads plant (hom_bench.simulate) gives realistic inputs; a fresh
controller then steps over those samples (as hom_bench.replay does) and each step is timed. Prints and writes the
median, 95th percentile and largest step time per task, against the bench's 20 ms period.

    uv run --no-project --python 3.10 --with mujoco --with numpy --with scipy python hom_timing.py OUT.json \
        [--tasks twist,roll,press,pinpush] [--d 40] [--seconds 3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  scripts/hom (hom_tasks, hand_object_scale_sim): hom_bench._paths adds only scripts/
from sr2hand.control import hom_bench as HB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--tasks", default="twist,roll,press,pinpush")
ap.add_argument("--d", type=float, default=40.0)
ap.add_argument("--seconds", type=float, default=3.0)
a = ap.parse_args()
res = {}
for task in a.tasks.split(","):
    try:
        t0 = time.perf_counter()
        bench = HB.HomBench.from_design(task, a.d)
        t_build = time.perf_counter() - t0
        stream, _ = HB.simulate(bench, 0, a.seconds)
        bench.reset()
        dts, phases = [], []
        for smp in stream:
            t1 = time.perf_counter()
            _tg, info = bench.step(float(smp["t"]), smp["joints"], smp["obj_pos_mm"], smp["obj_quat"], smp.get("contacts"))
            dts.append((time.perf_counter() - t1) * 1000.0)
            phases.append(info["phase"])
        d = np.array(dts)
        res[task] = dict(steps=len(d), build_s=round(t_build, 2), median_ms=round(float(np.median(d)), 3),
                         p95_ms=round(float(np.percentile(d, 95)), 3), max_ms=round(float(d.max()), 3),
                         phases=sorted(set(map(str, phases))))
        print(f"{task}: {len(d)} steps, median {np.median(d):.2f} ms, p95 {np.percentile(d, 95):.2f}, max {d.max():.2f} "
              f"(build {t_build:.1f} s)", flush=True)
    except Exception as e:                               # noqa: BLE001 - report and go on with the next task
        res[task] = {"error": repr(e)}
        print(f"{task}: {e!r}", flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))
