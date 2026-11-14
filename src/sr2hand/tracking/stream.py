"""UDP messages of the live tracker and the controller link (plan: docs/notes/20261105-object_seating_and_tracking/).

All messages are little-endian structs with a 4-byte magic, a version and a sequence number, so a receiver can drop
stale or foreign datagrams without parsing JSON. Times are host milliseconds since the epoch (time.time() * 1000);
on corvus the cameras' frame times are librealsense's global time in the same clock. Lengths are mm, quaternions wxyz.

PoseMsg   (tracker -> listeners): one tracked body.
    seq, t_capture_ms (the mean of the cameras' frame times), t_publish_ms, skew_ms (between the cameras' frames),
    T_BO (body in the arm's base: position, quaternion), T_PO (body in the palm frame P through the arm's TCP pose;
    NaN without the arm), n_cams, n_tags, flags (bit 0 lost, bit 1 single camera, bit 2 no arm pose), rms_px.
"""
from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass, field

import numpy as np

POSE_MAGIC = b"SR2T"
POSE_VERSION = 1
_POSE = struct.Struct("<4sHHIddd3d4d3d4dBBHf")
LOST, SINGLE, NO_ARM = 1, 2, 4


@dataclass
class PoseMsg:
    seq: int
    t_capture_ms: float
    t_publish_ms: float
    skew_ms: float
    p_BO: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    q_BO: np.ndarray = field(default_factory=lambda: np.full(4, np.nan))
    p_PO: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    q_PO: np.ndarray = field(default_factory=lambda: np.full(4, np.nan))
    n_cams: int = 0
    n_tags: int = 0
    flags: int = 0
    rms_px: float = float("nan")

    def pack(self) -> bytes:
        return _POSE.pack(POSE_MAGIC, POSE_VERSION, 0, self.seq & 0xFFFFFFFF, self.t_capture_ms, self.t_publish_ms,
                          self.skew_ms, *np.asarray(self.p_BO, float), *np.asarray(self.q_BO, float),
                          *np.asarray(self.p_PO, float), *np.asarray(self.q_PO, float), self.n_cams, self.n_tags,
                          self.flags, self.rms_px)

    @classmethod
    def unpack(cls, b: bytes) -> "PoseMsg | None":
        if len(b) != _POSE.size or b[:4] != POSE_MAGIC:
            return None
        v = _POSE.unpack(b)
        if v[1] != POSE_VERSION:
            return None
        return cls(seq=v[3], t_capture_ms=v[4], t_publish_ms=v[5], skew_ms=v[6], p_BO=np.array(v[7:10]),
                   q_BO=np.array(v[10:14]), p_PO=np.array(v[14:17]), q_PO=np.array(v[17:21]), n_cams=v[21],
                   n_tags=v[22], flags=v[23], rms_px=v[24])

    @property
    def age_ms(self):
        return time.time() * 1000.0 - self.t_capture_ms


class UdpSender:
    """Sends bytes to every (host, port) in `to`, never blocking."""

    def __init__(self, to):
        self.to = [(h, int(p)) for h, p in to]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def send(self, b: bytes):
        for addr in self.to:
            try:
                self.sock.sendto(b, addr)
            except (BlockingIOError, OSError):
                pass


class UdpReceiver:
    """Receives datagrams on (host, port); latest() drains the socket and returns the newest message that parses."""

    def __init__(self, port, host="0.0.0.0", parse=PoseMsg.unpack):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, int(port)))
        self.parse = parse

    def recv(self, timeout=None):
        """The next message (blocking up to timeout s) with its arrival time, or (None, None)."""
        self.sock.settimeout(timeout)
        try:
            b = self.sock.recv(65536)
        except (socket.timeout, BlockingIOError):
            return None, None
        return self.parse(b), time.time() * 1000.0

    def latest(self):
        self.sock.setblocking(False)
        out = None
        while True:
            try:
                m = self.parse(self.sock.recv(65536))
            except (BlockingIOError, OSError):
                return out
            if m is not None:
                out = m
