"""Stream check for the ReSkin 5X board on the QT Py ESP32-S3 (firmware/reskin5x_stream).

Reads the 82-byte records (five chips x t, x, y, z as float32, then CR LF) for a few seconds and prints, per chip in
streaming order, the mean field and its magnitude (uT), the standard deviation of each axis over distinct conversions
(uT) and the temperature (degC); then the record rate, the conversion rate (distinct values on chip 0) and the largest
field component as a fraction of the range (38.4 mT at GAIN_SEL 1, RES 1; --range-mT). The firmware prints its I2C scan
and register read-back once, to the first program that opens the port after a reset; this script shows those lines when
it is that program.

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_check.py [--port P] [--seconds 5] [--out f.csv]
"""
import argparse
import glob
import time

import numpy as np
import serial

N_CHIPS = 5
FRAME = 4 * 4 * N_CHIPS + 2


def frames(buf):
    """Fixed-length records found in buf, and the printable lines before the first one."""
    sync = lambda p: all(buf[p + k * FRAME - 2:p + k * FRAME] == b"\r\n" for k in (1, 2, 3))
    p = next((p for p in range(len(buf) - 3 * FRAME) if sync(p)), None)
    if p is None:
        return np.zeros((0, 4 * N_CHIPS)), buf.decode("ascii", "replace").splitlines()
    text = [ln for ln in buf[:p].decode("ascii", "replace").splitlines() if ln.strip()]
    recs = []
    while p + FRAME <= len(buf):
        if buf[p + FRAME - 2:p + FRAME] == b"\r\n":
            recs.append(np.frombuffer(buf[p:p + FRAME - 2], "<f4"))
            p += FRAME
        else:
            p = next((q for q in range(p + 1, len(buf) - 3 * FRAME) if sync(q)), len(buf))
    return np.array(recs, dtype=float), text


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--out", default=None, help="CSV of the records: t0 x0 y0 z0 ... t4 x4 y4 z4")
    ap.add_argument("--range-mT", type=float, default=38.4, help="range of the firmware's GAIN_SEL and RES")
    a = ap.parse_args()
    range_ut = 1000 * a.range_mT
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
    print(f"{port}: {len(r)} records in {dt:.1f} s ({len(r) / dt:.0f} Hz)")
    print("chip   Bx      By      Bz     |B| (uT)   sd x    sd y    sd z (uT)   T (degC)")
    for c in range(N_CHIPS):
        b = r[:, c, 1:]
        new = np.r_[True, np.any(np.diff(b, axis=0) != 0, axis=1)]
        sd = b[new].std(axis=0) if new.sum() > 2 else np.full(3, np.nan)
        m = b.mean(axis=0)
        flag = "  (all zero: chip missing)" if not np.any(r[:, c]) else ""
        print(f"{c:3d} {m[0]:7.1f} {m[1]:7.1f} {m[2]:7.1f} {np.linalg.norm(m):7.1f}   "
              f"{sd[0]:6.2f}  {sd[1]:6.2f}  {sd[2]:6.2f}      {r[:, c, 0].mean():6.1f}{flag}")
    new0 = np.r_[True, np.any(np.diff(r[:, 0, 1:], axis=0) != 0, axis=1)]
    peak = np.abs(r[:, :, 1:]).max()
    print(f"conversions on chip 0: {new0.sum() / dt:.0f} Hz; largest component {peak:.0f} uT "
          f"= {100 * peak / range_ut:.1f} % of the {a.range_mT:.1f} mT range")
    if a.out:
        np.savetxt(a.out, r.reshape(len(r), -1), fmt="%.3f",
                   header=" ".join(f"{q}{c}" for c in range(N_CHIPS) for q in "txyz"))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
