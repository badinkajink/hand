"""Log every record of the fingertip mux board (firmware/reskin5x_mux_stream) on corvus's clock, in the format of
host/reskin5x_mux.py's log (host time float64, then t x y z float32 per chip, board by board), with the records of one
read spaced evenly between that read and the one before (reskin5x_mux.py spaces them at 1/150 s, the three-board rate).
A sidecar <log>.json names the mux channels. When the board goes away the logger waits and starts a new log.

    /usr/bin/python3 board_log.py PORT OUTDIR
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import serial

port, outdir = sys.argv[1], Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
CHIP_BYTES, CHIPS = 16, 5


def info(s, wait_s=3.0):
    s.reset_input_buffer()
    s.write(b"i")
    buf, t0 = bytearray(), time.time()
    while time.time() - t0 < wait_s:
        buf += s.read(4096)
        m = re.search(rb"Boards on mux channels: ([0-9 ]*)\r?\nRecord bytes: (\d+)\r?\n", buf)
        if m:
            return [int(c) for c in m.group(1).split()], int(m.group(2)), bytearray(buf[m.end():])
    raise RuntimeError("no answer to 'i'")


def run():
    s = serial.Serial(port, 115200, timeout=0.05)
    chans, frame, buf = info(s)
    n = len(chans)
    dt = np.dtype([("t", "<f8"), ("r", "<f4", (n * CHIPS * 4,))])
    path = outdir / time.strftime("%Y%m%d-%H%M%S-mux.bin")
    path.with_suffix(".json").write_text(json.dumps({"port": port, "channels": chans, "record_bytes": frame,
                                                     "dtype": "t <f8, r <f4 x 20 per board", "start": time.time()}))
    log = open(path, "ab")
    print(f"{time.strftime('%H:%M:%S')} channels {chans}, {frame}-byte records -> {path}", flush=True)
    synced, t_last, n_rec, t_stat = False, time.time(), 0, time.time()
    try:
        while True:
            buf += s.read(16384)
            now = time.time()
            recs = []
            while len(buf) >= 2 * frame:
                if not synced:
                    if buf[frame - 2:frame] == b"\r\n" and buf[2 * frame - 2:2 * frame] == b"\r\n":
                        synced = True
                    else:
                        del buf[:1]
                        continue
                if buf[frame - 2:frame] != b"\r\n":
                    synced = False
                    continue
                recs.append(bytes(buf[:frame - 2]))
                del buf[:frame]
            if not recs:
                if now - t_last > 3.0:
                    raise RuntimeError("no records for 3 s (board reset?)")
                continue
            rows = np.zeros(len(recs), dt)
            k = np.arange(1, len(recs) + 1)
            rows["t"] = t_last + (now - t_last) * k / len(recs)
            for i, r in enumerate(recs):
                rows["r"][i] = np.frombuffer(r, "<f4")
            t_last = now
            log.write(rows.tobytes())
            n_rec += len(recs)
            if now - t_stat >= 300:
                log.flush()
                print(f"{time.strftime('%H:%M:%S')} {n_rec / (now - t_stat):.1f} records/s", flush=True)
                n_rec, t_stat = 0, now
    finally:
        log.close()
        s.close()


while True:
    try:
        run()
    except (serial.SerialException, OSError, RuntimeError) as e:
        print(f"{time.strftime('%H:%M:%S')} {type(e).__name__}: {e}; waiting for the board", flush=True)
        time.sleep(2.0)
    except KeyboardInterrupt:
        break
