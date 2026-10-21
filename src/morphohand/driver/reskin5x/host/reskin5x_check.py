"""Stream check for the ReSkin 5X board on the QT Py ESP32-S3 (firmware/reskin5x_stream).

Reads the 82-byte records (five chips x t, x, y, z as float32, then CR LF) for a few seconds and prints, per chip in
streaming order, the mean field and its magnitude (uT); the noise, as the standard deviation of each axis inside 0.5 s
windows averaged over the windows (uT); the drift, as the largest range of the 0.5 s means over the three axes (uT; tens of
uT when the board turns in Earth's field or a magnet moves near it); and the temperature (degC). Then the largest field
component as a fraction of the range (38.4 mT at GAIN_SEL 1, RES 1; --range-mT). The loop reads faster than a chip
converts, so consecutive records repeat values, and the three axes of a chip update at different records. The firmware
prints its I2C scan and register read-back once, to the first program that opens the port after a reset; this script shows
those lines when it is that program.

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_check.py [--port P] [--seconds 5] [--out f.csv]
"""
import argparse
import glob
import time

import numpy as np
import serial

N_CHIPS = 5
FRAME = 4 * 4 * N_CHIPS + 2
WINDOW_S = 0.5


def frames(buf):
    """Fixed-length records found in buf, and the printable lines before the first one."""
    p, text = 0, []
    while (i := buf.find(b"\r\n", p)) >= 0 and all(32 <= c < 127 for c in buf[p:i]):
        text += [buf[p:i].decode()] if i > p else []
        p = i + 2
    sync = lambda p: all(buf[p + k * FRAME - 2:p + k * FRAME] == b"\r\n" for k in (1, 2, 3))
    p = next((q for q in range(p, len(buf) - 3 * FRAME) if sync(q)), None)
    if p is None:
        return np.zeros((0, 4 * N_CHIPS)), text
    recs = []
    while p + FRAME <= len(buf):
        if buf[p + FRAME - 2:p + FRAME] == b"\r\n":
            recs.append(np.frombuffer(buf[p:p + FRAME - 2], "<f4"))
            p += FRAME
        else:
            p = next((q for q in range(p + 1, len(buf) - 3 * FRAME) if sync(q)), len(buf))
    return np.array(recs, dtype=float), text


def report(r, dt, range_mT):
    """Per-chip table for records r (n, chip, t x y z) spanning dt seconds."""
    w = max(2, int(len(r) / dt * WINDOW_S))
    print(f"chip   Bx      By      Bz     |B| (uT)   sd x    sd y    sd z (uT, in {WINDOW_S} s)   drift (uT)   T (degC)")
    for c in range(N_CHIPS):
        b = r[:, c, 1:]
        win = b[:len(b) // w * w].reshape(-1, w, 3) if len(b) >= 2 * w else b[None]
        m, sd = b.mean(axis=0), win.std(axis=1).mean(axis=0)
        drift = np.ptp(win.mean(axis=1), axis=0).max()
        flag = "  (all zero: chip missing)" if not np.any(r[:, c]) else ""
        print(f"{c:3d} {m[0]:7.1f} {m[1]:7.1f} {m[2]:7.1f} {np.linalg.norm(m):7.1f}   "
              f"{sd[0]:6.2f}  {sd[1]:6.2f}  {sd[2]:6.2f}              {drift:6.1f}      {r[:, c, 0].mean():6.1f}{flag}")
    peak = np.abs(r[:, :, 1:]).max()
    print(f"largest component {peak:.0f} uT = {100 * peak / (1000 * range_mT):.1f} % of the {range_mT:.1f} mT range")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--out", default=None, help="CSV of the records: t0 x0 y0 z0 ... t4 x4 y4 z4")
    ap.add_argument("--range-mT", type=float, default=38.4, help="range of the firmware's GAIN_SEL and RES")
    a = ap.parse_args()
    port = a.port or sorted(glob.glob("/dev/ttyACM*"))[0]

    s = serial.Serial(port, 115200, timeout=0.1)
    buf, t0 = b"", time.time()
    while time.time() - t0 < a.seconds:
        buf += s.read(65536)
    dt = time.time() - t0
    s.close()

    r, text = frames(buf)
    for ln in text:
        print("  |", ln)
    if not len(r):
        print(f"{port}: {len(buf)} bytes in {dt:.1f} s, no records")
        return
    r = r.reshape(len(r), N_CHIPS, 4)
    live = np.any(r != 0, axis=(0, 2))
    ok = np.all(np.any(r[:, live] != 0, axis=2), axis=1)
    print(f"{port}: {len(r)} records in {dt:.1f} s ({len(r) / dt:.0f} Hz); {np.sum(~ok)} without data on a live chip "
          "(before its first conversion or after a failed transfer) left out")
    report(r[ok], dt * ok.mean(), a.range_mT)
    if a.out:
        np.savetxt(a.out, r.reshape(len(r), -1), fmt="%.3f",
                   header=" ".join(f"{q}{c}" for c in range(N_CHIPS) for q in "txyz"))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
