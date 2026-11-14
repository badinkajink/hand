"""UDP link between corvus (the hand, its cameras and boards) and a controller on another host, or on corvus itself
(owner 2026-11-13: "if we need to setup policy streaming over ethernet, so be it"; loop-benchmark hand-off, step 15).

corvus sends a StateMsg every control period and applies the newest TargetMsg it has. When no new target has come for
`miss_periods` periods (3), it stops: it sends no more frames, ends the station's stream, and the station holds the last
goal (the station's own lease of 0.25 s does the same if corvus dies). Restarting needs a new arm().

Little-endian structs, host milliseconds since the epoch, joints per finger thumb, index, middle.

StateMsg  (corvus -> controller), magic "SR2S":
    seq, t_ms (sent), t_joints_ms, joints (9 x servo degrees: aa, fe1, fe2 per finger, NaN when unknown),
    t_capture_ms, p_PO (mm), q_PO (wxyz) of the tracked body with its tracker flags and tag count,
    cop (per finger: point in P, mm, and normal force, N; NaN when unknown), last_target_seq, status bits
    (DRY_RUN, STOPPED, STREAMING).
TargetMsg (controller -> corvus), magic "SR2C":
    seq, state_seq (the state it answers), t_ms (sent), targets (9 x sim degrees: yaw, mcp, pip per finger, the
    station's /stream/frame convention), servo_speed (NaN: unchanged), flags (bit 0: stop).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

FINGERS = ("thumb", "index", "middle")
SIM_JOINTS = ("yaw", "mcp", "pip")
SERVO_JOINTS = ("aa", "fe1", "fe2")
DRY_RUN, STOPPED, STREAMING = 1, 2, 4
STOP = 1

_STATE = struct.Struct("<4sHHIdd9fd3d4dHB" + "x" + "12f" + "IH")
_TARGET = struct.Struct("<4sHHIId9ffH")
STATE_MAGIC, TARGET_MAGIC, VERSION = b"SR2S", b"SR2C", 1


def _nan(n):
    return np.full(n, np.nan)


@dataclass
class StateMsg:
    seq: int
    t_ms: float
    t_joints_ms: float = float("nan")
    joints: np.ndarray = field(default_factory=lambda: _nan(9))
    t_capture_ms: float = float("nan")
    p_PO: np.ndarray = field(default_factory=lambda: _nan(3))
    q_PO: np.ndarray = field(default_factory=lambda: _nan(4))
    obj_flags: int = 0
    n_tags: int = 0
    cop: np.ndarray = field(default_factory=lambda: _nan(12))
    last_target_seq: int = 0
    status: int = 0

    def pack(self) -> bytes:
        return _STATE.pack(STATE_MAGIC, VERSION, 0, self.seq & 0xFFFFFFFF, self.t_ms, self.t_joints_ms,
                           *np.asarray(self.joints, np.float32), self.t_capture_ms, *np.asarray(self.p_PO, float),
                           *np.asarray(self.q_PO, float), self.obj_flags, self.n_tags,
                           *np.asarray(self.cop, np.float32), self.last_target_seq & 0xFFFFFFFF, self.status)

    @classmethod
    def unpack(cls, b: bytes) -> "StateMsg | None":
        if len(b) != _STATE.size or b[:4] != STATE_MAGIC:
            return None
        v = _STATE.unpack(b)
        if v[1] != VERSION:
            return None
        return cls(seq=v[3], t_ms=v[4], t_joints_ms=v[5], joints=np.array(v[6:15]), t_capture_ms=v[15],
                   p_PO=np.array(v[16:19]), q_PO=np.array(v[19:23]), obj_flags=v[23], n_tags=v[24],
                   cop=np.array(v[25:37]), last_target_seq=v[37], status=v[38])

    def joints_dict(self):
        """{finger: {aa, fe1, fe2}} in servo degrees (HomBench's input)."""
        J = np.asarray(self.joints).reshape(3, 3)
        return {f: dict(zip(SERVO_JOINTS, map(float, J[k]))) for k, f in enumerate(FINGERS)}


@dataclass
class TargetMsg:
    seq: int
    state_seq: int
    t_ms: float
    targets: np.ndarray = field(default_factory=lambda: _nan(9))
    servo_speed: float = float("nan")
    flags: int = 0

    def pack(self) -> bytes:
        return _TARGET.pack(TARGET_MAGIC, VERSION, 0, self.seq & 0xFFFFFFFF, self.state_seq & 0xFFFFFFFF, self.t_ms,
                            *np.asarray(self.targets, np.float32), self.servo_speed, self.flags)

    @classmethod
    def unpack(cls, b: bytes) -> "TargetMsg | None":
        if len(b) != _TARGET.size or b[:4] != TARGET_MAGIC:
            return None
        v = _TARGET.unpack(b)
        if v[1] != VERSION:
            return None
        return cls(seq=v[3], state_seq=v[4], t_ms=v[5], targets=np.array(v[6:15]), servo_speed=v[15], flags=v[16])

    def joints_frame(self):
        """{finger: {yaw, mcp, pip}} in sim degrees: the body of the station's POST /api/v1/stream/frame."""
        T = np.asarray(self.targets, float).reshape(3, 3)
        return {f: dict(zip(SIM_JOINTS, map(float, T[k]))) for k, f in enumerate(FINGERS)}
