#!/usr/bin/env python3
"""Calibration session of the lab's ATI Mini45 on the UR5 (CB3) with a weighed mass, for the lab computer: the robot
visits the stations of a plan written by scripts/ati_calibration_plan.py while the Net F/T box streams the six raw gage
values (RDT, UDP 49152), the robot its joints (RTDE, 125 Hz) and, if asked, the HEX-E its wrench. The operator hangs the
mass on one eye of the hook cross at the plan's loading station and takes it off there again.

Files, in logs/<date>-ati_calibration/<time>-<name>/: ati.bin (float64 rows: host time, RDT sequence, F/T sequence,
status, six gage values), robot.bin (host time, controller time, q0..q5 rad, flange pose m and rad, safety mode), ft.bin
(the HEX-E, as the tip calibration's runner writes it), events.jsonl (every step's start and end, each rest's window and
mean gage values), plan.json, session.json (arguments, the weighed mass, the box's netftapi2.xml as read).
scripts/ati_calibration_analysis.py turns the folder into the calibration.

Before the session (docs/experiments/20261023-ati_ft_calibration/, the plan page's procedure): the box's state saved as
found; the raw-gage calibration #16 (FT00000) selected by hand with lbf and lbf-in, counts per force and per torque 1;
RDT enabled. This program reads the box's configuration and never writes it; it never sends the box a bias command.

    python3 scripts/ati_calibration_session.py --plan <plan.json> --ati 192.168.1.1 --mass-g 500.3 [--hexe 192.168.0.6]
        --dry       every block without the mass and without prompts: the moves in the cell, the operator watching
        --step      ask before each move
        --resume D  continue the session in folder D (steps that ended ok are skipped; asks for the mass's state)
        --sim       the same session against a simulated robot and box in virtual time (writes the truth to sim_truth.json)
Stops the robot (stopJ) on Ctrl-C, on an error bit in the box's status, on a gap of over 0.5 s in its stream and on a
protective stop. The robot must stand at the plan's homing pose (within 3 deg per joint) when the session starts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import struct
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "src/morphohand/driver/reskin5x/host"
sys.path.insert(0, str(HOST))
import ur5_tip_calibration as TC  # noqa: E402  (Ring, Appender, SessionLog, URRobot, OptoForceStream, jsonable)

RDT_PORT = 49152
REQ = struct.Struct("!HHI")
REC = struct.Struct("!IIIiiiiii")
STATUS_OK = (0x00000000, 0x80010000)       # healthy; a monitor condition latched (not an error), Net F/T manual 17.1
SAT = 0x00020000                           # transducer saturation or A/D error


# ------------------------------------------------------------------------------------------- the Net F/T box
def read_config(ip, timeout=2.0):
    """netftapi2.xml as text and its fields, or (None, {error}) (Net F/T manual section 8.1)."""
    import xml.etree.ElementTree as ET
    try:
        with urllib.request.urlopen(f"http://{ip}/netftapi2.xml", timeout=timeout) as r:
            txt = r.read().decode("latin-1")
    except Exception as e:                                       # noqa: BLE001
        return None, dict(error=str(e))
    try:
        return txt, {c.tag: (c.text or "").strip() for c in ET.fromstring(txt)}
    except ET.ParseError as e:
        return txt, dict(error=f"parse: {e}")


def raw_mode_problems(cfg):
    """What in the box's configuration says it is not sending raw gage values (Quick Start 3.4.2)."""
    out = []
    if "error" in cfg:
        return [f"netftapi2.xml not read: {cfg['error']}"]
    if cfg.get("cfgcalsn", "").upper() != "FT00000":
        out.append(f"active calibration serial {cfg.get('cfgcalsn')!r}, not FT00000 (#16)")
    if cfg.get("cfgcpf") not in ("1", None) or cfg.get("cfgcpt") not in ("1", None):
        out.append(f"counts per force {cfg.get('cfgcpf')}, per torque {cfg.get('cfgcpt')}: not 1")
    if cfg.get("comrdte", "1") not in ("1", "Enabled", "enabled", "true"):
        out.append(f"RDT interface {cfg.get('comrdte')!r}")
    return out


class AtiStream(threading.Thread):
    """RDT from the box: start real-time streaming (0x0002, count 0), read every 36-byte record of each datagram (the
    box may buffer several), keep a ring of the last minutes and append rows to ati.bin. A gap of over 0.5 s and every
    status other than healthy are kept for the session to act on. Never sends a bias (0x0042)."""

    def __init__(self, ip, appender=None):
        super().__init__(daemon=True)
        self.addr, self.app = (ip, RDT_PORT), appender
        self.ring = TC.Ring(keep_s=120.0)
        self.stop, self.n, self.lost, self.last_seq, self.last_t = False, 0, 0, None, None
        self.gaps, self.bad_status = [], []
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.5)

    def run(self):
        self.sock.sendto(REQ.pack(0x1234, 0x0002, 0), self.addr)
        rows = []
        while not self.stop:
            try:
                data = self.sock.recv(65536)
            except socket.timeout:
                if self.last_t is not None and time.time() - self.last_t > 0.5:
                    self.gaps.append((self.last_t, time.time()))
                continue
            t = time.time()
            for k in range(len(data) // REC.size):
                r = REC.unpack_from(data, k * REC.size)
                if self.last_seq is not None and r[0] != self.last_seq + 1:
                    self.lost += max(r[0] - self.last_seq - 1, 0)
                if self.last_t is not None and t - self.last_t > 0.5:
                    self.gaps.append((self.last_t, t))
                self.last_seq, self.last_t = r[0], t
                if r[2] not in STATUS_OK and (not self.bad_status or self.bad_status[-1][1] != r[2]):
                    self.bad_status.append((t, r[2]))
                self.ring.add(t, np.array(r[3:], float))
                rows.append((t,) + r)
                self.n += 1
            if self.app is not None and len(rows) >= 250:
                self.app.write(rows)
                rows = []
        if self.app is not None and rows:
            self.app.write(rows)
        try:
            self.sock.sendto(REQ.pack(0x1234, 0x0000, 0), self.addr)
        except OSError:
            pass


# ------------------------------------------------------------------------------------------- backends
class RealBackend:
    def __init__(self, a, log):
        self.ati = AtiStream(a.ati, log.appender("ati.bin"))
        self.ati.start()
        self.hexe = None
        if a.hexe:
            self.hexe = TC.OptoForceStream(a.hexe, rate=100, appender=log.appender("ft.bin"), filt=3)
            self.hexe.configure()
            self.hexe.start()
        self.robot = TC.URRobot(a.robot, log.appender("robot.bin"))

    @staticmethod
    def now():
        return time.time()

    @staticmethod
    def sleep(dt):
        time.sleep(dt)

    @staticmethod
    def operator(prompt):
        return input(f"\n>>> {prompt}; then step back and press Enter (or type stop): ")

    def close(self):
        self.ati.stop = True
        if self.hexe is not None:
            self.hexe.stop = True
        self.robot.close()


# ------------------------------------------------------------------------------------------- the session
class Abort(Exception):
    pass


class Session:
    def __init__(self, plan, be, log, a):
        self.plan, self.be, self.log, self.a = plan, be, log, a
        self.st = {s["id"]: s for s in plan["stations"]}
        self.home = np.radians(plan["home_q_deg"])
        self.v, self.acc = plan["speed"]["v"] * a.speed, plan["speed"]["a"] * a.speed
        self.done = log.done()
        self.mass_on = False
        self.noise = None

    def q_of(self, sid):
        return self.home if sid == "home" else np.radians(self.st[sid]["q_deg"])

    def check_streams(self):
        ati = self.be.ati
        if ati.bad_status:
            t, s = ati.bad_status[-1]
            raise Abort(f"the box's status 0x{s:08x}" + (" (gage saturation or A/D error)" if s & SAT else ""))
        if ati.gaps:
            g = ati.gaps[-1]
            raise Abort(f"no record from the box for {g[1] - g[0]:.1f} s")
        if self.be.robot.stopped():
            raise Abort("the robot is in a protective or emergency stop")

    def gages(self, t0, t1=None):
        return self.be.ati.ring.since(t0, t1)

    def preflight(self):
        self.log.event(event="start", id="preflight")
        self.be.sleep(3.0)
        t1 = self.be.now()
        t, g = self.gages(t1 - 3.0)
        rate = len(t) / 3.0
        if rate < 100:
            raise SystemExit(f"the box sends {rate:.0f} records/s: check its IP ({self.a.ati}), RDT on comm.htm, the cable")
        self.noise = g.std(axis=0)
        q = self.be.robot.q()
        off = np.degrees(np.abs(q - self.home)).max()
        print(f"preflight: box {rate:.0f} records/s, gage noise {np.round(self.noise, 1)} counts, mean "
              f"{np.round(g.mean(axis=0), 0)}; robot {off:.1f} deg from the plan's homing pose")
        self.check_streams()
        if off > 3.0:
            raise SystemExit("jog the robot to the plan's homing pose first (home_q_deg in the plan): the first move "
                             "was checked from there")
        self.log.event(event="end", id="preflight", status="ok", rec=dict(rate=rate, noise=self.noise, mean=g.mean(axis=0),
                                                                           q_deg=np.degrees(q)))

    def payload(self, on):
        """The plan's payload; with the mass on, the mass as weighed (--mass-g) at the plan's place for it (the plan's
        with_mass record is for its nominal mass)."""
        p = self.plan["payload"]
        kg, cog = p["kg"], p["cog_mm"]
        if on:
            w = p["with_mass"]
            m_nom = w["kg"] - kg
            z_m = (w["kg"] * w["cog_mm"][2] - kg * cog[2]) / m_nom
            m = (self.a.mass_g or 1e3 * m_nom) * 1e-3
            cog = [0.0, 0.0, round((kg * cog[2] + m * z_m) / (kg + m), 1)]
            kg = round(kg + m, 4)
        self.be.robot.set_payload(kg, cog)

    def move(self, sid):
        q = self.q_of(sid)
        if self.a.step and self.be.operator(f"next: move to {sid}") .strip().lower() == "stop":
            raise Abort("stopped by the operator")
        self.be.robot.moveJ(q, self.v, self.acc, asynchronous=True)
        t0 = self.be.now()
        self.be.sleep(0.1)
        while not self.be.robot.steady():
            self.check_streams()
            self.be.sleep(0.02)
            if self.be.now() - t0 > 90.0:
                raise Abort("the move took over 90 s")
        err = np.degrees(np.abs(self.be.robot.q() - q)).max()
        if err > 0.5:
            raise Abort(f"the robot stopped {err:.2f} deg from {sid}")

    def rest(self):
        """Wait for the gages to settle (std over 0.5 s within 1.5 times the rest noise, at least settle_s, at most
        settle_max_s), then average rest_s seconds (the plan's, or --rest-s)."""
        T = dict(self.plan["time"], **({"rest_s": self.a.rest_s} if self.a.rest_s else {}))
        t0 = self.be.now()
        self.be.sleep(T["settle_s"])
        while self.be.now() - t0 < T["settle_max_s"]:
            _, g = self.gages(self.be.now() - 0.5)
            if len(g) > 10 and np.all(g.std(axis=0) <= 1.5 * self.noise + 0.5):
                break
            self.be.sleep(0.25)
        settle = self.be.now() - t0
        r0 = self.be.now()
        self.be.sleep(T["rest_s"])
        r1 = self.be.now()
        self.check_streams()
        t, g = self.gages(r0, r1)
        tr, vr = self.be.robot.ring.since(r0, r1)
        out = dict(t=[r0, r1], settle_s=settle, n=len(t), g_mean=g.mean(axis=0), g_std=g.std(axis=0),
                   q=np.mean(vr[:, 1:7], axis=0) if len(vr) else self.be.robot.q())
        if self.be.hexe is not None:
            th, vh = self.be.hexe.ring.since(r0, r1)
            out["hexe"] = vh.mean(axis=0) if len(vh) else None
        return out

    def run(self):
        if "preflight" not in self.done:
            self.preflight()
        else:
            _, g = self.gages(self.be.now() - 3.0)
            self.noise = g.std(axis=0) if len(g) > 10 else np.full(6, 3.0)
            if not self.a.sim:
                ans = self.be.operator("resuming: is the mass hanging on an eye now? type yes or no")
                self.mass_on = ans.strip().lower().startswith("y")
        self.payload(False)
        for bi, b in enumerate(self.plan["blocks"]):
            for si, s in enumerate(b["steps"]):
                sid = f"b{bi}-s{si:02d}"
                if sid in self.done:
                    self.mass_on = s["mass"] == "on" and s.get("role") != "unload"
                    continue
                if self.a.dry and s.get("role") == "via" and s["mass"] == "on":
                    pass
                self.log.event(event="start", id=sid, block=bi, eye=b["eye"], station=s["station"], mass=s["mass"],
                               role=s.get("role"))
                self.move(s["station"])
                if s.get("role") == "via":
                    self.log.event(event="end", id=sid, status="ok")
                    continue
                if s.get("prompt") and not self.a.dry:
                    ans = self.be.operator(f"{s['prompt']} (the robot holds still)")
                    if ans.strip().lower() == "stop":
                        raise Abort("stopped by the operator")
                    self.mass_on = s.get("role") == "load"
                    self.payload(self.mass_on)
                    self.log.event(event="mass", on=self.mass_on, eye=b["eye"], t_mass=self.be.now())
                rec = self.rest()
                rec["mass_on"] = bool(self.mass_on) and not self.a.dry
                if s.get("role") == "unload" and not self.a.dry:      # the rest above had the mass on; now off
                    self.mass_on = False
                print(f"{sid} {s['station']} eye {b['eye']} mass {'on ' if rec['mass_on'] else 'off'} "
                      f"settled {rec['settle_s']:.1f} s, gage std {np.round(rec['g_std'], 1)}", flush=True)
                self.log.event(event="end", id=sid, status="ok", rec=rec)
        if np.degrees(np.abs(self.be.robot.q() - self.home)).max() > 0.5 and self.plan.get("end_home"):
            self.move("home")
        print("session complete")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--robot", default="192.168.0.4")
    ap.add_argument("--ati", default="192.168.1.1", help="the Net F/T box (its documented default)")
    ap.add_argument("--hexe", default=None, help="the HEX-E's DAQ, e.g. 192.168.0.6 (optional: its gains, R5)")
    ap.add_argument("--mass-g", type=float, default=None, help="the mass with its holder, weighed to 0.1 g")
    ap.add_argument("--speed", type=float, default=1.0, help="factor on the plan's joint speed and acceleration")
    ap.add_argument("--rest-s", type=float, default=None, help="seconds averaged at each rest (the plan's 3 s by default)")
    ap.add_argument("--line-mm", type=float, default=None, help="the line's diameter, measured (recorded for the analysis)")
    ap.add_argument("--eye-hole-mm", type=float, default=None, help="the eyes' holes as drilled, if not the plan's 1.2 mm")
    ap.add_argument("--sim-line-mm", type=float, default=None, help="--sim only: twice the line's offset off the edge in the "
                    "simulated truth (a stiff line), in place of --line-mm")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--step", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--logs", default=str(Path.cwd() / "logs"))
    ap.add_argument("--name", default="ati")
    ap.add_argument("--no-config-check", action="store_true", help="run although the box's configuration was not read")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--sim-seed", type=int, default=0)
    a = ap.parse_args(argv)
    plan = json.loads(Path(a.plan).read_text())
    if plan.get("kind") != "ati_mass_calibration":
        raise SystemExit("not an ATI calibration plan")
    if a.mass_g is None and not a.dry:
        raise SystemExit("--mass-g: weigh the mass with its holder to 0.1 g")
    folder = Path(a.resume) if a.resume else Path(a.logs) / f"{time.strftime('%Y%m%d')}-ati_calibration" / \
        f"{time.strftime('%H%M%S')}-{a.name}"
    if a.sim:
        import ati_calibration_sim_session as SS
        be = SS.SimBackend(plan, a, seed=a.sim_seed)
        log = TC.SessionLog(None, None, resume=str(folder), clock=be.now)
        be.attach(log)
        cfg_txt, cfg = be.config()
    else:
        cfg_txt, cfg = read_config(a.ati)
        probs = raw_mode_problems(cfg)
        if probs:
            print("the box's configuration:", "; ".join(probs))
            if not a.no_config_check:
                raise SystemExit("select #16 (FT00000), lbf and lbf-in on the box's Configuration page, enable RDT on "
                                 "comm.htm, then start again (or --no-config-check after checking it by hand)")
        log = TC.SessionLog(None, None, resume=str(folder))
        be = RealBackend(a, log)
    if not a.resume:
        log.save("plan.json", plan)
    log.save("session.json" if not a.resume else f"session_resume_{time.strftime('%H%M%S')}.json",
             dict(args=vars(a), started=time.strftime("%Y-%m-%d %H:%M:%S"), mass_g=a.mass_g, box_config=cfg,
                  box_xml=cfg_txt, ati_columns="t_host rdt_seq ft_seq status g0..g5 (raw gage values)",
                  robot_columns="t_host t_ctrl q0..q5 (rad) x y z (m) rx ry rz safety_mode"))
    ses = Session(plan, be, log, a)
    try:
        ses.run()
    except (Abort, KeyboardInterrupt) as e:
        print(f"\nstopping the robot: {e}")
        try:
            be.robot.stopJ(2.0) if hasattr(be.robot, "stopJ") else be.robot.stopL(2.0)
        except Exception:                                        # noqa: BLE001
            pass
        log.event(event="abort", why=str(e))
    finally:
        log.event(event="ati_stream", n=be.ati.n, lost=be.ati.lost, gaps=be.ati.gaps, bad_status=be.ati.bad_status)
        be.close()
        log.close()
        print(f"session in {log.dir}")


if __name__ == "__main__":
    main()
