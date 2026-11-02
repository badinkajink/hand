"""Whether the HEX-E streams: reads it for SECONDS with the program's own reader (100 Hz, filter 3, no bias command) and
exits 0 when at least 80 % of the expected samples came, 1 otherwise. For a chain that waits out a link loss before it
resumes a run; never while the program runs (one reader on the DAQ). Moves nothing.

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/hexe_probe.py [--seconds 3] [--ft 192.168.0.6]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration as U  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=3.0)
ap.add_argument("--ft", default="192.168.0.6")
a = ap.parse_args()
ft = U.OptoForceStream(a.ft, rate=100, appender=None, bias=False, filt=3)
try:
    ft.configure()
    ft.start()
    time.sleep(a.seconds)
    n = ft.n
finally:
    ft.stop = True
    time.sleep(0.3)
ok = n >= 0.8 * 100 * a.seconds
print(time.strftime("%H:%M:%S"), f"HEX-E: {n} samples in {a.seconds:g} s, {ft.n_timeout} timed out", "ok" if ok else "NOT STREAMING",
      flush=True)
sys.exit(0 if ok else 1)
