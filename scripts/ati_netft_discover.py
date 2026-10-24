#!/usr/bin/env python3
"""Finds ATI Net F/T boxes on the local network by ATI's RTA discovery, whatever their IP address and subnet: one UDP
broadcast to port 51000, and the boxes answer on the multicast group 224.0.5.128, port 28250. The socket joins that
group; without the join nothing is received (badinkajink/ati_ft_sensor demos/ati_gui/netft_gui.py, whose request this
copies, found 0 devices on 2026-10-23 until the workstation added it). Sends nothing else and changes nothing.

On 2026-10-23 the lab's Net box (NETBA, MAC 00:16:BD:00:2D:E4) answered from a static 192.168.123.50/24, outside
every subnet a sweep had tried. corvus then reaches it through a live second address on its LAN port, which keeps the
DHCP address and the internet (no sudo; gone after a reboot):

    nmcli device modify enp3s0 +ipv4.addresses 192.168.123.100/24     # undo: -ipv4.addresses

    /usr/bin/python3 scripts/ati_netft_discover.py [--iface-ip 192.168.0.5] [--wait 4]
"""
import argparse
import socket
import time

SEND_PORT, RECV_PORT, GROUP = 51000, 28250, "224.0.5.128"
REQ = b"RTA Device DiscoveryRTAD"
TAG_MAC, TAG_IP, TAG_MASK, TAG_APP, TAG_DELAY_MULT = 0x01, 0x02, 0x03, 0x0D, 0xF2


def request():
    return REQ + bytes([TAG_IP, 4, *map(int, GROUP.split("."))]) + bytes([TAG_DELAY_MULT, 1, 10])


def parse(payload):
    """The tagged fields of a reply ('RTAD', then tag, length, value ...) as a dict, or None."""
    if not payload.startswith(b"RTAD"):
        return None
    out, pos = {}, 4
    while pos + 2 <= len(payload):
        tag, n = payload[pos], payload[pos + 1]
        v = payload[pos + 2:pos + 2 + n]
        if len(v) < n:
            break
        if tag == TAG_IP and n == 4:
            out["ip"] = ".".join(map(str, v))
        elif tag == TAG_MASK and n == 4:
            out["mask"] = ".".join(map(str, v))
        elif tag == TAG_MAC:
            out["mac"] = ":".join(f"{b:02x}" for b in v)
        elif tag == TAG_APP:
            out["app"] = v.decode("latin-1", errors="ignore").strip("\x00 ")
        pos += 2 + n
    return out if "ip" in out else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iface-ip", default="0.0.0.0", help="this computer's address on the LAN port (default: any)")
    ap.add_argument("--wait", type=float, default=4.0, help="seconds to listen for replies")
    a = ap.parse_args()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.bind(("", RECV_PORT))
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(GROUP) + socket.inet_aton(a.iface_ip))
    if a.iface_ip != "0.0.0.0":
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(a.iface_ip))
    s.settimeout(0.2)
    s.sendto(request(), ("255.255.255.255", SEND_PORT))
    found, end = {}, time.monotonic() + a.wait
    while time.monotonic() < end:
        try:
            payload, src = s.recvfrom(2048)
        except socket.timeout:
            continue
        d = parse(payload)
        if d and d["ip"] not in found:
            found[d["ip"]] = d
            print(f"{d['ip']}  mask {d.get('mask', '?')}  MAC {d.get('mac', '?')}  {d.get('app', '')}  (from {src[0]})")
    s.close()
    print(f"{len(found)} Net F/T box(es) answered in {a.wait:.0f} s")


if __name__ == "__main__":
    main()
