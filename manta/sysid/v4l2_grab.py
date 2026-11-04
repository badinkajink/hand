#!/usr/bin/env python3
"""Minimal V4L2 YUYV frame grabber, standard library only (the CB1 has no
OpenCV/numpy and no internet). Used to watch the gantries with the
RealSense D435's colour stream while gantry_rom_sweep.py runs.

    python3 v4l2_grab.py <device> <out_dir> [--width W --height H --period S --count N]

Writes <out_dir>/<unix_time_ms>.yuyv raw frames (W*H*2 bytes each).
Convert on a machine with numpy: see yuyv_to_png.py. --count 0 = run until killed.
"""

import argparse
import fcntl
import mmap
import os
import struct
import time


def _ioc(d, nr, size):
    return (d << 30) | (size << 16) | (ord("V") << 8) | nr


VIDIOC_ENUM_FMT = _ioc(3, 2, 64)
VIDIOC_S_FMT = _ioc(3, 5, 208)
VIDIOC_REQBUFS = _ioc(3, 8, 20)
VIDIOC_QUERYBUF = _ioc(3, 9, 88)
VIDIOC_QBUF = _ioc(3, 15, 88)
VIDIOC_DQBUF = _ioc(3, 17, 88)
VIDIOC_STREAMON = _ioc(1, 18, 4)
VIDIOC_STREAMOFF = _ioc(1, 19, 4)
CAPTURE, MMAP = 1, 1
YUYV = struct.unpack("<I", b"YUYV")[0]


def formats(fd):
    out, i = [], 0
    while True:
        buf = bytearray(struct.pack("<II", i, CAPTURE) + bytes(56))
        try:
            fcntl.ioctl(fd, VIDIOC_ENUM_FMT, buf)
        except OSError:
            return out
        out.append(struct.unpack_from("<I", buf, 44)[0])
        i += 1


def buffer_struct(index=0):
    b = bytearray(88)
    struct.pack_into("<II", b, 0, index, CAPTURE)
    struct.pack_into("<I", b, 60, MMAP)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("device")
    ap.add_argument("out_dir")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--count", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    fd = os.open(a.device, os.O_RDWR)
    if YUYV not in formats(fd):
        raise SystemExit(f"{a.device}: no YUYV format")
    fmt = bytearray(208)
    struct.pack_into("<I", fmt, 0, CAPTURE)
    struct.pack_into("<IIII", fmt, 8, a.width, a.height, YUYV, 0)
    fcntl.ioctl(fd, VIDIOC_S_FMT, fmt)
    w, h = struct.unpack_from("<II", fmt, 8)
    print(f"{a.device}: {w}x{h} YUYV", flush=True)

    req = bytearray(struct.pack("<III", 4, CAPTURE, MMAP) + bytes(8))
    fcntl.ioctl(fd, VIDIOC_REQBUFS, req)
    n = struct.unpack_from("<I", req, 0)[0]
    maps = []
    for i in range(n):
        b = buffer_struct(i)
        fcntl.ioctl(fd, VIDIOC_QUERYBUF, b)
        offset = struct.unpack_from("<I", b, 64)[0]
        length = struct.unpack_from("<I", b, 72)[0]
        maps.append(mmap.mmap(fd, length, mmap.MAP_SHARED, mmap.PROT_READ, offset=offset))
        fcntl.ioctl(fd, VIDIOC_QBUF, b)
    fcntl.ioctl(fd, VIDIOC_STREAMON, struct.pack("<I", CAPTURE))
    try:
        saved, next_save = 0, time.time() + 0.5  # let auto-exposure settle briefly
        while a.count == 0 or saved < a.count:
            b = buffer_struct()
            fcntl.ioctl(fd, VIDIOC_DQBUF, b)
            i = struct.unpack_from("<I", b, 0)[0]
            used = struct.unpack_from("<I", b, 8)[0]
            now = time.time()
            if now >= next_save:
                with open(os.path.join(a.out_dir, f"{int(now * 1000)}.yuyv"), "wb") as f:
                    f.write(maps[i][:used])
                saved += 1
                next_save = now + a.period
            fcntl.ioctl(fd, VIDIOC_QBUF, b)
    finally:
        fcntl.ioctl(fd, VIDIOC_STREAMOFF, struct.pack("<I", CAPTURE))
        os.close(fd)


if __name__ == "__main__":
    main()
