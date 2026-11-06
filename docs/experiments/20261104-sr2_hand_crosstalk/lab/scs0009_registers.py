"""Read-only dump of the SR2 hand's nine SCS0009 servos' registers. Runs on the CB1 with the control station stopped,
since the servo bus is exclusive: the script refuses to start while a manta_hand.web process runs, and opens the port
exclusively.

It sends nothing but READ instructions (0x02). send_read() builds every packet; no other packet is ever written, and an
assert checks the instruction byte before each write. Each servo's table is read in blocks of at most 16 bytes, over
addresses 0-70 and 78-82, and printed as hex. The named fields are then decoded from those bytes. Every field's raw
bytes stand beside its value.

Addresses come from Feetech's SCSCL map (FTServo_Arduino src/SCSCL.h and FTServo_Python scservo_sdk/scscl.py) wherever
that map names them: version, ID, baud rate, angle limits, dead bands 26/27, torque enable, goal position, goal time,
goal speed, lock, and present position, speed, load, voltage, temperature, moving and current. The fields that map
leaves out come from rustypot's SCS0009 table (src/servo/feetech/scs0009.rs, the library under the CB1's driver): the
gains P, D and I at 21-23, the minimum startup force (punch) at 24, the maximum torque at 16, the limits of temperature
and voltage, the unloading and LED conditions, the protective torque, protection time and overload torque at 37-39,
the acceleration, the status byte and the factory block at 78-82. Words are big-endian, as in Feetech's SCSCL class
(End = 1: the lower address holds the high byte). Speeds and current carry their sign on bit 15 and the load on bit
10, as in SCSCL.cpp. Positions are steps; the degrees beside them are rustypot's (steps - 511) x 300 / 1024, the
scale of the driver's zero_deg.

    ~/hand/src/morphohand/driver/manta/host/.venv/bin/python scs0009_registers.py [--port /dev/ttyUSB0] [--out F.txt]
        [--dry]      (--dry prints the packets it would send and opens nothing)
"""
import argparse
import json
import os
import time
from pathlib import Path

INST_READ = 0x02
SERVOS = {0: "thumb yaw", 1: "thumb mcp", 2: "thumb pip", 3: "index yaw", 4: "index mcp", 5: "index pip",
          6: "middle yaw", 7: "middle mcp", 8: "middle pip"}
BLOCKS = [(0, 16), (16, 16), (32, 16), (48, 16), (64, 7), (78, 5)]
F, R = "Feetech SCSCL.h", "rustypot scs0009.rs"
# (address, bytes, name, decoding, source)
FIELDS = [
    (0, 1, "firmware major version", "u8", R), (1, 1, "firmware minor version", "u8", R),
    (3, 2, "version (model)", "u16", F), (5, 1, "ID", "u8", F), (6, 1, "baud rate code (0 = 1 Mbps)", "u8", F),
    (7, 1, "return delay time", "u8", R), (8, 1, "response status level", "u8", R),
    (9, 2, "min angle limit", "pos", F), (11, 2, "max angle limit", "pos", F),
    (13, 1, "max temperature limit, C", "u8", R), (14, 1, "max input voltage, 0.1 V", "u8", R),
    (15, 1, "min input voltage, 0.1 V", "u8", R), (16, 2, "max torque, 0.1 %", "u16", R),
    (18, 1, "phase (special function byte)", "hex", R), (19, 1, "unloading condition (bits)", "hex", R),
    (20, 1, "LED alarm condition (bits)", "hex", R), (21, 1, "P coefficient", "u8", R),
    (22, 1, "D coefficient", "u8", R), (23, 1, "I coefficient", "u8", R),
    (24, 2, "minimum startup force (punch)", "u16", R), (26, 1, "CW dead band", "u8", F),
    (27, 1, "CCW dead band", "u8", F), (37, 1, "protective torque, %", "u8", R),
    (38, 1, "protection time", "u8", R), (39, 1, "overload torque, %", "u8", R),
    (40, 1, "torque enable", "u8", F), (41, 1, "acceleration", "u8", R), (42, 2, "goal position", "pos", F),
    (44, 2, "goal time", "u16", F), (46, 2, "goal speed", "sm15", F), (48, 1, "lock", "u8", F),
    (56, 2, "present position", "pos", F), (58, 2, "present speed", "sm15", F), (60, 2, "present load", "sm10", F),
    (62, 1, "present voltage, 0.1 V", "u8", F), (63, 1, "present temperature, C", "u8", F),
    (64, 1, "sync write flag", "u8", R), (65, 1, "status (error bits)", "hex", R), (66, 1, "moving", "u8", F),
    (69, 2, "present current", "sm15", F), (78, 1, "PWM maximum step", "u8", R),
    (79, 1, "moving velocity threshold, 50 steps/s", "u8", R), (80, 1, "DTs", "u8", R),
    (81, 1, "minimum velocity limit, 50 steps/s", "u8", R), (82, 1, "maximum velocity limit, 50 steps/s", "u8", R),
]


def read_packet(sid, addr, n):
    body = [sid, 4, INST_READ, addr, n]
    return bytes([0xFF, 0xFF] + body + [~sum(body) & 0xFF])


def send_read(port, sid, addr, n):
    """One READ of n bytes at addr; the n data bytes, or None. Skips an echo of the packet the adapter may return."""
    pkt = read_packet(sid, addr, n)
    assert pkt[4] == INST_READ and len(pkt) == 8
    port.reset_input_buffer()
    port.write(pkt)
    port.flush()
    buf, t0 = b"", time.time()
    while time.time() - t0 < 0.1:
        buf += port.read(64)
        i = 0
        while True:
            i = buf.find(b"\xff\xff", i)
            if i < 0 or len(buf) < i + 4:
                break
            ln = buf[i + 3]
            if len(buf) < i + 4 + ln:
                break
            p = buf[i:i + 4 + ln]
            if p == pkt:                                        # the adapter's echo of the request
                i += len(p)
                continue
            if p[2] == sid and ln == n + 2 and (~sum(p[2:-1]) & 0xFF) == p[-1]:
                return p[5:5 + n], p[4]
            i += 1
    return None, None


def decode(kind, b):
    v = b[0] if len(b) == 1 else (b[0] << 8) | b[1]
    if kind == "pos":
        return f"{v} steps ({(v - 511) * 300 / 1024:+.2f} deg)"
    if kind == "sm15":
        return f"{-(v & 0x7FFF) if v & 0x8000 else v}"
    if kind == "sm10":
        return f"{-(v & 0x3FF) if v & 0x400 else v & 0x3FF}"
    if kind == "hex":
        return f"0x{v:02x}"
    return f"{v}"


def station_running():
    me = os.getpid()
    for d in Path("/proc").iterdir():
        if d.name.isdigit() and int(d.name) != me:
            try:
                if b"manta_hand.web" in (d / "cmdline").read_bytes():
                    return True
            except OSError:
                pass
    return False


ap = argparse.ArgumentParser()
ap.add_argument("--port", default="/dev/ttyUSB0")
ap.add_argument("--baud", type=int, default=1_000_000)
ap.add_argument("--out", default=None, help="also write the report here; the raw table goes to the same name .json")
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
if a.dry:
    for sid in SERVOS:
        print(sid, " ".join(read_packet(sid, ad, n).hex(" ") for ad, n in BLOCKS))
    raise SystemExit(0)
if station_running():
    raise SystemExit("the control station is running (manta_hand.web): stop it first; nothing was read")
import serial  # noqa: E402

port = serial.Serial(a.port, a.baud, timeout=0.01, exclusive=True)
time.sleep(1.0)                                                 # the driver's PORT_SETTLE_S
lines = [f"SCS0009 registers, {time.strftime('%Y-%m-%d %H:%M:%S')}, {a.port} at {a.baud} baud; READ (0x02) only"]
table = {}
n_sent = 0
for sid, label in SERVOS.items():
    mem, errs = {}, []
    for addr, n in BLOCKS:
        for attempt in range(3):
            data, err = send_read(port, sid, addr, n)
            n_sent += 1
            if data is not None:
                break
            time.sleep(0.01)
        if data is None:
            errs.append(f"no reply at {addr}+{n}")
            continue
        if err:
            errs.append(f"status 0x{err:02x} at {addr}")
        mem.update({addr + k: data[k] for k in range(n)})
        time.sleep(0.005)
    table[sid] = {"label": label, "bytes": {str(k): v for k, v in sorted(mem.items())}, "errors": errs}
    lines.append(f"\nservo {sid} ({label}){'; ' + ', '.join(errs) if errs else ''}")
    for addr, size, name, kind, src in FIELDS:
        b = [mem.get(addr + k) for k in range(size)]
        if None in b:
            lines.append(f"  {addr:3d}  {name:40s} (not read)")
            continue
        lines.append(f"  {addr:3d}  {name:40s} {' '.join(f'{x:02x}' for x in b):6s}  {decode(kind, b):>26s}   {src}")
    for lo, hi in ((0, 71), (78, 83)):
        lines.append(f"  hex {lo:2d}-{hi - 1:2d}: " + " ".join(f"{mem[k]:02x}" if k in mem else ".." for k in range(lo, hi)))
port.close()
lines.append(f"\n{n_sent} READ packets sent, no other instruction")
text = "\n".join(lines) + "\n"
print(text, end="")
if a.out:
    Path(a.out).write_text(text)
    Path(a.out).with_suffix(".json").write_text(json.dumps({"read_at": time.time(), "port": a.port, "servos": table},
                                                           indent=1))
