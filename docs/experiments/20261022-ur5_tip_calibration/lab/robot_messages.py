"""The controller's own messages (errors, popups, program stops, runtime exceptions) from the UR's primary interface,
read-only: it connects to port 30001 and never sends a byte, since anything sent there runs as URScript. Prints each
robot message (package type 20) with its subtype and text, and every change of the robot mode, the program's run state
and the emergency and protective stops from the robot-mode data, with host times. For finding what stops a running
script, as freedrive's twice on 2026-11-02.

    nohup setsid /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/robot_messages.py >> logs/ur5_lab_robot_messages.log 2>&1 &
"""
import re
import socket
import struct
import time

HOST, PORT = "192.168.0.4", 30001
SUBTYPES = {0: "text", 1: "program label", 2: "popup", 3: "version", 5: "safety mode", 6: "error code", 7: "key message",
            9: "request value", 10: "runtime exception"}


def stamp():
    return time.strftime("%H:%M:%S") + f".{int(time.time() * 10) % 10}"


def texts(b):
    return [t.decode("ascii", "replace") for t in re.findall(rb"[ -~]{4,}", b)]


def main():
    s = socket.create_connection((HOST, PORT), timeout=10.0)
    print(stamp(), "connected to", HOST, PORT, flush=True)
    buf, last = b"", None
    while True:
        chunk = s.recv(65536)
        if not chunk:
            print(stamp(), "closed by the controller", flush=True)
            return
        buf += chunk
        while len(buf) >= 5:
            n, kind = struct.unpack(">iB", buf[:5])
            if n < 5 or n > 1 << 20:
                print(stamp(), f"lost sync (size {n}); reconnecting", flush=True)
                return
            if len(buf) < n:
                break
            pkg, buf = buf[:n], buf[n:]
            if kind == 20:
                ts, src, sub = struct.unpack(">QbB", pkg[5:15])
                body = pkg[15:]
                extra = ""
                if sub == 6 and len(body) >= 12:
                    code, arg, level = struct.unpack(">iii", body[:12])
                    extra = f" code C{code}A{arg} level {level}"
                print(stamp(), f"message {SUBTYPES.get(sub, sub)} (source {src}){extra}:", " | ".join(texts(body)), flush=True)
            elif kind == 16:                                     # robot state: find its robot-mode sub-package (type 0)
                i = 5
                while i + 5 <= n:
                    sn, st = struct.unpack(">iB", pkg[i:i + 5])
                    if sn < 5:
                        break
                    if st == 0 and sn >= 5 + 8 + 7 + 1:
                        b = pkg[i + 5:i + sn]
                        fields = struct.unpack(">Q???????B", b[:16])
                        connected, enabled, power, estop, pstop, running, paused, mode = fields[1:9]
                        now = (mode, estop, pstop, running, paused, power, enabled)
                        if now != last:
                            print(stamp(), f"robot mode {mode} power {power} enabled {enabled} estop {estop} "
                                  f"protective stop {pstop} program running {running} paused {paused}", flush=True)
                            last = now
                        break
                    i += sn


if __name__ == "__main__":
    while True:
        try:
            main()
        except OSError as e:
            print(stamp(), "socket:", e, flush=True)
        time.sleep(2.0)
