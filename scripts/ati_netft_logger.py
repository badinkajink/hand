#!/usr/bin/env python3
"""Logger for an ATI Net F/T box (NETBA, serial FT35796 in the lab) during a UR5 calibration session: streams the box's six
channels over RDT (UDP 49152) and writes them with host time stamps beside the session's own streams, so that the
calibration fit (scripts/ati_calibration_fit.py) can pair them with the HEX-E's and the robot's.

RDT, as in ATI's Net F/T manual (document 9620-05-Net-FT): a request is 8 bytes in network order, uint16 0x1234, uint16
command, uint32 sample count (0: until stopped); command 0x0002 starts real-time streaming and 0x0000 stops it. Each
record is 36 bytes: uint32 rdt_sequence, uint32 ft_sequence, uint32 status, then six int32 counts. In force/torque mode
the counts divide by the box's counts per force and per torque (cfgcpf, cfgcpt in http://<box>/netftapi2.xml). In the
factory raw-gage mode (calibration #16, FT00000, ATI Quick Start 9610-05-1022, section 3.4.2) the six values are the
strain gages' raw values, which this calibration needs: no borrowed matrix is applied anywhere.

Before a session (the handoff of 2026-10-23, docs/experiments/20261023-ati_ft_calibration/): save the box's web pages and
netftapi2.xml as they are, then select the raw-gage calibration by hand in the web page and record which one it is. This
script reads the configuration and never writes it.

    python3 scripts/ati_netft_logger.py --ip 192.168.1.1 --out <session folder> [--seconds 0] [--fake]

Writes <out>/ati.bin (float64 rows: host time, rdt_sequence, ft_sequence, status, six counts) and <out>/ati.json (the box's
netftapi2.xml as read, its parsed fields, start time, packets, losses). --fake serves a simulated box on 127.0.0.1 for a
test without hardware.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PORT = 49152
REQ = struct.Struct("!HHI")
REC = struct.Struct("!IIIiiiiii")


def read_config(ip, timeout=2.0):
    """The box's netftapi2.xml (raw text) and its fields, or (None, {}) if the page does not answer."""
    try:
        with urllib.request.urlopen(f"http://{ip}/netftapi2.xml", timeout=timeout) as r:
            txt = r.read().decode("latin-1")
    except Exception as e:                       # noqa: BLE001 (the box may serve no page; streaming still works)
        return None, dict(error=str(e))
    try:
        root = ET.fromstring(txt)
        return txt, {c.tag: (c.text or "").strip() for c in root}
    except ET.ParseError as e:
        return txt, dict(error=f"parse: {e}")


def fake_box(stop, ip="127.0.0.1", rate_hz=1000.0):
    """A simulated box: answers a start request by streaming records of slowly varying counts until a stop request."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((ip, PORT))
    s.settimeout(0.05)
    peer, seq, streaming, t0 = None, 0, False, time.time()
    while not stop.is_set():
        try:
            data, addr = s.recvfrom(64)
            hdr, cmd, n = REQ.unpack(data[:8])
            if hdr == 0x1234:
                streaming, peer = cmd == 0x0002, addr
        except socket.timeout:
            pass
        if streaming and peer:
            for _ in range(int(rate_hz * 0.05)):
                seq += 1
                t = time.time() - t0
                c = [int(1000 * np.sin(t * (k + 1))) for k in range(6)]
                s.sendto(REC.pack(seq, seq * 2, 0, *c), peer)
    s.close()


def log(ip, out, seconds=0.0, flush_every=500):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    xml, cfg = read_config(ip)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)
    sock.sendto(REQ.pack(0x1234, 0x0002, 0), (ip, PORT))
    t_start = time.time()
    rows, n, lost, last = [], 0, 0, None
    f = open(out / "ati.bin", "ab")
    meta = dict(ip=ip, start=t_start, start_local=time.strftime("%Y-%m-%d %H:%M:%S"), config_fields=cfg, netftapi2_xml=xml,
                columns=["t_host", "rdt_sequence", "ft_sequence", "status", "g0", "g1", "g2", "g3", "g4", "g5"],
                note="counts as the box sends them; in force/torque mode divide by cfgcpf / cfgcpt, in raw-gage mode they "
                     "are gage values")
    try:
        while not seconds or time.time() - t_start < seconds:
            try:
                data = sock.recv(64)
            except socket.timeout:
                print("no record for 1 s", flush=True)
                continue
            t = time.time()
            if len(data) < REC.size:
                continue
            r = REC.unpack(data[:REC.size])
            if last is not None and r[0] != last + 1:
                lost += max(r[0] - last - 1, 0)
            last = r[0]
            rows.append((t,) + r)
            n += 1
            if len(rows) >= flush_every:
                f.write(np.asarray(rows, float).tobytes())
                f.flush()
                os.fsync(f.fileno())
                rows = []
    except KeyboardInterrupt:
        pass
    finally:
        sock.sendto(REQ.pack(0x1234, 0x0000, 0), (ip, PORT))
        if rows:
            f.write(np.asarray(rows, float).tobytes())
        f.flush()
        os.fsync(f.fileno())
        f.close()
        meta.update(stop=time.time(), records=n, lost_by_sequence=lost)
        (out / "ati.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"{n} records, {lost} lost by sequence, {n / max(time.time() - t_start, 1e-9):.0f} per s -> {out / 'ati.bin'}", flush=True)
    return meta


def load(folder):
    """ati.bin of a session as an (n, 10) array."""
    a = np.fromfile(Path(folder) / "ati.bin", "<f8")
    return a[: len(a) // 10 * 10].reshape(-1, 10)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ip", default="192.168.1.1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=0.0, help="0: until Ctrl-C")
    ap.add_argument("--fake", action="store_true", help="test against a simulated box on 127.0.0.1")
    a = ap.parse_args()
    if a.fake:
        stop = threading.Event()
        th = threading.Thread(target=fake_box, args=(stop,), daemon=True)
        th.start()
        time.sleep(0.2)
        m = log("127.0.0.1", a.out, a.seconds or 2.0)
        stop.set()
        th.join(timeout=1.0)
        d = load(a.out)
        print("rows", d.shape, "first counts", d[0, 4:].astype(int).tolist(), "host-time span", round(d[-1, 0] - d[0, 0], 3), "s")
        sys.exit(0 if m["records"] > 0 else 1)
    log(a.ip, a.out, a.seconds)
