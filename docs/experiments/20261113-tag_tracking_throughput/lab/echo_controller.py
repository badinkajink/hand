"""A stand-in controller for testing the link (sr2hand.control.link): it answers every StateMsg with a TargetMsg that
holds a fixed pose (sim degrees, all zero by default) after an optional compute delay, and can go silent after
--quiet-after seconds to test corvus's stop. Runs on corvus or on the workstation; nothing here touches hardware.

    python3 echo_controller.py --listen 47002 --reply HOST:47001 [--delay-ms 2] [--quiet-after S] [--seconds 40]
"""
import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from sr2hand.control.link import StateMsg, TargetMsg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--listen", type=int, default=47002)
ap.add_argument("--reply", required=True)
ap.add_argument("--delay-ms", type=float, default=0.0)
ap.add_argument("--quiet-after", type=float, default=None)
ap.add_argument("--seconds", type=float, default=40.0)
a = ap.parse_args()
rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rx.bind(("0.0.0.0", a.listen))
rx.settimeout(0.5)
tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
to = (a.reply.rsplit(":", 1)[0], int(a.reply.rsplit(":", 1)[1]))
t0, n, seq = time.time(), 0, 0
while time.time() - t0 < a.seconds:
    try:
        s = StateMsg.unpack(rx.recv(512))
    except socket.timeout:
        continue
    if s is None:
        continue
    n += 1
    if a.quiet_after is not None and time.time() - t0 > a.quiet_after:
        continue
    if a.delay_ms:
        time.sleep(a.delay_ms / 1000.0)
    tx.sendto(TargetMsg(seq=seq, state_seq=s.seq, t_ms=time.time() * 1000.0, targets=np.zeros(9)).pack(), to)
    seq += 1
print(f"echo: {n} states, {seq} targets", flush=True)
