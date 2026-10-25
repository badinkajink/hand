#!/usr/bin/env python3
"""Split the stream of firmware/reskin5x_mux_stream (several ReSkin 5X boards behind an I2C mux on one QT Py) into one
stream a board, each on a pseudo-terminal that wrench_viewer.py --port and reskin5x_plot.py read as a single board's
port (owner, 2026-10-25: three fingers on mux channels 0, 1 and 3).

The splitter sends 'i' to learn the mux channels and the record length, then writes each board's five chips as an 82-byte
record (the format of firmware/reskin5x_stream) to its pseudo-terminal, linked at <links>/ch<channel>, and appends every
record of all boards to logs/<date>-three_finger_mux/<start>-mux.bin (host time float64, then t x y z float32 per chip
in streaming order, board by board; 8 + 80 n bytes a record). A pseudo-terminal that no program reads drops records
instead of blocking the others. When the board is unplugged the splitter waits for it and starts again.

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_mux.py [--port P] [--links /tmp/reskin5x_mux]
    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/wrench_viewer.py --port /tmp/reskin5x_mux/ch0 --http 8790
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
import tty
from pathlib import Path

import numpy as np
import serial

ROOT = Path(__file__).resolve().parents[5]
PORT_GLOBS = ("/dev/serial/by-id/*QT_Py*", "/dev/ttyACM*")
CHIP_BYTES, CHIPS = 16, 5


def find_port():
    for g in PORT_GLOBS:
        hits = sorted(glob.glob(g))
        if hits:
            return hits[0]
    return None


def info(s, wait_s=3.0):
    """Ask the firmware for its channels and record length; returns (channels, record bytes) and the bytes read after
    the info text."""
    s.reset_input_buffer()
    s.write(b"i")
    buf, t0 = bytearray(), time.time()
    while time.time() - t0 < wait_s:
        buf += s.read(4096)
        m = re.search(rb"Boards on mux channels: ([0-9 ]*)\r?\nRecord bytes: (\d+)\r?\n", buf)
        if m:
            chans = [int(c) for c in m.group(1).split()]
            return chans, int(m.group(2)), bytearray(buf[m.end():])
    raise RuntimeError("no answer to 'i': is firmware/reskin5x_mux_stream on the board?")


class Outlet:
    """A pseudo-terminal in raw mode with its slave linked at `link`; writes never block."""

    def __init__(self, link):
        self.master, self.slave = os.openpty()
        tty.setraw(self.slave)
        os.set_blocking(self.master, False)
        self.path = os.ttyname(self.slave)
        self.link = Path(link)
        self.link.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.link.with_suffix(".new")
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(self.path)
        tmp.replace(self.link)
        self.dropped = 0

    def write(self, b):
        try:
            os.write(self.master, b)
        except (BlockingIOError, OSError):
            self.dropped += 1

    def close(self):
        for fd in (self.master, self.slave):
            try:
                os.close(fd)
            except OSError:
                pass


def run(a):
    port = a.port or find_port()
    if port is None:
        raise SystemExit("no serial port of the QT Py")
    s = serial.Serial(port, 115200, timeout=0.05)
    chans, frame, buf = info(s)
    n = len(chans)
    assert frame == n * CHIPS * CHIP_BYTES + 2, (chans, frame)
    outs = [Outlet(Path(a.links) / f"ch{c}") for c in chans]
    log_dir = ROOT / "logs" / time.strftime("%Y%m%d-three_finger_mux")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / time.strftime("%Y%m%d-%H%M%S-mux.bin")
    dt = np.dtype([("t", "<f8"), ("r", "<f4", (n * CHIPS * 4,))])
    log = open(log_path, "ab")
    print(f"{time.strftime('%H:%M:%S')} {port}: boards on mux channels {chans}, {frame}-byte records; "
          + ", ".join(f"ch{c} -> {o.link} ({o.path})" for c, o in zip(chans, outs)) + f"; log {log_path}", flush=True)
    n_rec, t_stat, synced = 0, time.time(), False
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
                continue
            rows = np.zeros(len(recs), dt)
            for k, r in enumerate(recs):
                rows["t"][k] = now - (len(recs) - 1 - k) / 150.0
                rows["r"][k] = np.frombuffer(r, "<f4")
                for b, o in enumerate(outs):
                    o.write(r[b * CHIPS * CHIP_BYTES:(b + 1) * CHIPS * CHIP_BYTES] + b"\r\n")
            log.write(rows.tobytes())
            n_rec += len(recs)
            if now - t_stat >= a.stat_s:
                log.flush()
                print(f"{time.strftime('%H:%M:%S')} {n_rec / (now - t_stat):.1f} records/s; dropped by outlet "
                      + ", ".join(f"ch{c} {o.dropped}" for c, o in zip(chans, outs)), flush=True)
                n_rec, t_stat = 0, now
    finally:
        log.close()
        for o in outs:
            o.close()
        s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None)
    ap.add_argument("--links", default="/tmp/reskin5x_mux", help="folder of the links ch<channel> to the pseudo-terminals")
    ap.add_argument("--stat-s", type=float, default=60.0, help="print the record rate this often (s)")
    a = ap.parse_args()
    while True:
        try:
            run(a)
        except (serial.SerialException, OSError, RuntimeError) as e:
            print(f"{time.strftime('%H:%M:%S')} {type(e).__name__}: {e}; waiting for the board", flush=True)
            time.sleep(2.0)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    sys.exit(main())
