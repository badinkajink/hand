"""Records the HEX-E and the fingertip board with the program's own readers, for drift and dropouts while nothing runs.

Writes ft.bin and board.bin in the program's formats (ft: float64 rows t, sequence, status, Fx..Tz; board: t <f8 and
20 x <f4) into OUT, and prints a line every PRINT_S: the HEX-E's mean load (untared: no bias command is sent), its
reads, gaps over 0.5 s and restarts of its sample counter, and each chip's mean field and temperature. The DAQ gets the
program's rate and filter. Stop it before the program runs (one reader on the DAQ and on the board's port):
touch logs/ur5_lab_stop_watch, or wait SECONDS. Moves nothing.

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/sensor_watch.py OUT [--seconds 3600] [--print-s 10]
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration as U  # noqa: E402

BOARD = "/dev/serial/by-id/usb-Adafruit_QT_Py_ESP32-S3__4MB_Flash_2MB_PSRAM__b4:3a:45:b0:9c:d0-if00"
STOP = ROOT / "logs/ur5_lab_stop_watch"

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--seconds", type=float, default=3600.0)
ap.add_argument("--print-s", type=float, default=10.0)
ap.add_argument("--ft", default="192.168.0.6")
ap.add_argument("--no-board", action="store_true")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
STOP.unlink(missing_ok=True)

ft = U.OptoForceStream(a.ft, rate=100, appender=U.Appender(out / "ft.bin"), bias=False, filt=3)
ft.configure()
ft.start()
board = None
if not a.no_board:
    board = U.BoardStream(BOARD, appender=U.Appender(out / "board.bin", U.BoardStream.DTYPE))
    board.start()
print(f"recording into {out} for {a.seconds:.0f} s (stop: touch {STOP})", flush=True)

t_start, last_print, last_t, last_seq, n_gap, n_restart = time.time(), time.time(), None, None, 0, 0
try:
    while time.time() - t_start < a.seconds and not STOP.exists():
        time.sleep(0.5)
        if time.time() - last_print < a.print_s:
            continue
        t0, last_print = last_print, time.time()
        tf, vf = ft.ring.since(t0)
        line = datetime.datetime.now().strftime("%H:%M:%S")
        if len(tf):
            vf = vf.reshape(-1, 6)
            m = vf.mean(axis=0)
            line += f"  HEX-E {len(tf) / a.print_s:5.1f} Hz F ({m[0]:+7.2f}, {m[1]:+7.2f}, {m[2]:+8.2f}) N"
            line += f" T ({m[3]:+.3f}, {m[4]:+.3f}, {m[5]:+.4f}) N m"
            gaps = np.diff(np.r_[last_t if last_t is not None else tf[0], tf])
            n_gap += int((gaps > 0.5).sum())
            last_t = tf[-1]
        else:
            line += "  HEX-E no samples"
        line += f"  timeouts {ft.n_timeout} late {ft.n_late} gaps>0.5s {n_gap}"
        if board is not None:
            tb, vb = board.ring.since(t0)
            if len(tb):
                r = vb.reshape(-1, 5, 4)
                B = np.linalg.norm(r[:, :, 1:], axis=2).mean(axis=0)
                T = r[:, :, 0].mean(axis=0)
                line += f"  board {len(tb) / a.print_s:4.0f}/s |B| " + " ".join(f"{b:7.1f}" for b in B)
                line += "  T " + " ".join(f"{x:5.2f}" for x in T)
            else:
                line += "  board no records"
        print(line, flush=True)
finally:
    ft.stop = True
    if board is not None:
        board.stop = True
    time.sleep(0.5)
    print(f"stopped; HEX-E {ft.n} samples, {ft.n_timeout} timed out, {ft.n_late} late", flush=True)
