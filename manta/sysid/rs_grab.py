#!/usr/bin/env python3
"""Grab one color frame from the RealSense D435 on the CB1 with the Python standard library only
(raw V4L2 (Video4Linux) mmap streaming; the CB1 has no ffmpeg / OpenCV / numpy).

    python3 rs_grab.py OUT.yuyv [--dev /dev/video5] [--w 1920 --h 1080] [--skip 30]

Writes raw YUYV 4:2:2. Convert on the Mac:
    ffmpeg -f rawvideo -pix_fmt yuyv422 -s 1920x1080 -i OUT.yuyv OUT.jpg
"""
from __future__ import annotations

import argparse
import ctypes as C
import fcntl
import mmap
import os
import select


def _ioc(d, nr, size):
    return (d << 30) | (size << 16) | (ord("V") << 8) | nr


class PixFormat(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ("width", "height", "pixelformat", "field", "bytesperline",
                                            "sizeimage", "colorspace", "priv", "flags", "ycbcr_enc",
                                            "quantization", "xfer_func")]


class FmtUnion(C.Union):
    _fields_ = [("pix", PixFormat), ("raw", C.c_uint8 * 200), ("_align", C.c_void_p)]


class Format(C.Structure):
    _fields_ = [("type", C.c_uint32), ("fmt", FmtUnion)]


class ReqBufs(C.Structure):
    _fields_ = [("count", C.c_uint32), ("type", C.c_uint32), ("memory", C.c_uint32),
                ("capabilities", C.c_uint32), ("flags", C.c_uint8), ("reserved", C.c_uint8 * 3)]


class Timeval(C.Structure):
    _fields_ = [("sec", C.c_long), ("usec", C.c_long)]


class Timecode(C.Structure):
    _fields_ = [("type", C.c_uint32), ("flags", C.c_uint32), ("frames", C.c_uint8), ("seconds", C.c_uint8),
                ("minutes", C.c_uint8), ("hours", C.c_uint8), ("userbits", C.c_uint8 * 4)]


class MUnion(C.Union):
    _fields_ = [("offset", C.c_uint32), ("userptr", C.c_ulong), ("planes", C.c_void_p), ("fd", C.c_int32)]


class Buffer(C.Structure):
    _fields_ = [("index", C.c_uint32), ("type", C.c_uint32), ("bytesused", C.c_uint32), ("flags", C.c_uint32),
                ("field", C.c_uint32), ("timestamp", Timeval), ("timecode", Timecode), ("sequence", C.c_uint32),
                ("memory", C.c_uint32), ("m", MUnion), ("length", C.c_uint32), ("reserved2", C.c_uint32),
                ("request_fd", C.c_int32)]


RW = 3
S_FMT = _ioc(RW, 5, C.sizeof(Format))
REQBUFS = _ioc(RW, 8, C.sizeof(ReqBufs))
QUERYBUF = _ioc(RW, 9, C.sizeof(Buffer))
QBUF = _ioc(RW, 15, C.sizeof(Buffer))
DQBUF = _ioc(RW, 17, C.sizeof(Buffer))
STREAMON = _ioc(1, 18, 4)
STREAMOFF = _ioc(1, 19, 4)
CAPTURE, MMAP, NBUF = 1, 1, 4
YUYV = int.from_bytes(b"YUYV", "little")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dev", default="/dev/video5")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--skip", type=int, default=30, help="frames to drop while auto-exposure settles")
    ap.add_argument("--seconds", type=float, default=0.0, help="stream mode: write every --keep-th frame to OUT ('-' = stdout) for this long")
    ap.add_argument("--keep", type=int, default=3)
    a = ap.parse_args()
    fd = os.open(a.dev, os.O_RDWR)
    try:
        f = Format(type=CAPTURE)
        f.fmt.pix.width, f.fmt.pix.height, f.fmt.pix.pixelformat = a.w, a.h, YUYV
        fcntl.ioctl(fd, S_FMT, f)
        w, h = f.fmt.pix.width, f.fmt.pix.height
        rb = ReqBufs(count=NBUF, type=CAPTURE, memory=MMAP)
        fcntl.ioctl(fd, REQBUFS, rb)
        maps = []
        for i in range(rb.count):
            b = Buffer(index=i, type=CAPTURE, memory=MMAP)
            fcntl.ioctl(fd, QUERYBUF, b)
            maps.append(mmap.mmap(fd, b.length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                                  offset=b.m.offset))
            fcntl.ioctl(fd, QBUF, b)
        fcntl.ioctl(fd, STREAMON, C.c_int(CAPTURE))
        data = b""
        if a.seconds > 0:
            import sys
            import time
            out = sys.stdout.buffer if a.out == "-" else open(a.out, "wb")
            t_end, k, n = time.monotonic() + a.seconds, 0, 0
            while time.monotonic() < t_end:
                if not select.select([fd], [], [], 5.0)[0]:
                    raise SystemExit("no frame within 5 s")
                b = Buffer(type=CAPTURE, memory=MMAP)
                fcntl.ioctl(fd, DQBUF, b)
                if k % a.keep == 0:
                    out.write(maps[b.index][:b.bytesused]); n += 1
                fcntl.ioctl(fd, QBUF, b)
                k += 1
            out.flush()
            fcntl.ioctl(fd, STREAMOFF, C.c_int(CAPTURE))
            print(f"{w}x{h} {n} frames in {a.seconds:.0f}s", file=sys.stderr)
            return 0
        for k in range(a.skip + 1):
            if not select.select([fd], [], [], 5.0)[0]:
                raise SystemExit("no frame within 5 s")
            b = Buffer(type=CAPTURE, memory=MMAP)
            fcntl.ioctl(fd, DQBUF, b)
            if k == a.skip:
                data = maps[b.index][:b.bytesused]
            fcntl.ioctl(fd, QBUF, b)
        fcntl.ioctl(fd, STREAMOFF, C.c_int(CAPTURE))
        for m in maps:
            m.close()
    finally:
        os.close(fd)
    open(a.out, "wb").write(data)
    print(f"{w}x{h} {len(data)} bytes -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
