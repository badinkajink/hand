"""HOM on the hand: the controller end of the bench link (sr2hand.control.link), a simulated hand that plays corvus's end,
and the trial record (owner, 2026-11-15: "make sure we are ready to start collecting data and calibrating from real
trials of HOM").

    serve     answer corvus's StateMsg with HomBench's targets and write one record per control period
    sim-hand  play corvus's end against the pads plant of hom_tasks.py in lockstep (no hardware): a StateMsg per period
              from the plant, the answer's targets applied before the plant advances
    sil       both on this host over UDP, then hom_bench.simulate in process on the same layout and seed; prints both
              runs' J and the largest difference of their targets
    nominal   the object's start pose in P (mm, wxyz) for the task's grasp, as the plant starts it: the pose the operator
              aligns the hand to before a trial (hom_trial.py check --nominal)

The hand model and the plant that HOM inverts are set before hom_tasks is imported. The hand is cad_v2, the v4b fingertip
calibrated on the UR5 (SCALE_HAND), with pads at 2.5 MPa from the scene cache of 2026-11-02 (HOS_SCENES). The plant
(HOS_PLANT) is "measured", the SCS0009 as identified on 2026-11-06 (0.55 N m/rad at the measured 5.0-5.2 V supply,
0.18 N m), or "servo", the 0.72 N m/rad the simulations assumed. HOM divides its contact and gravity torques by this
gain to get angle offsets, so the gain sets the grip it commands. Without these variables hom_tasks falls back to cad_v1
and the 4 N m/rad work plant, which on the hand would command about a seventh of the grip.

The trial clock is corvus's: each state advances t by the periods elapsed on corvus's clock since the last one (one in a
steady stream), summed as hom_tasks sums its time, so the schedule (approach from 0.1 s, forces at 0.8 s, the arm's lift
at 1.6 s, the task from 2.0 s) follows the host that also times the lift, and each segment starts in the period it starts
in simulation. (From the raw millisecond stamps 3.3 s arrived as 3.2999998 s, and the twist's ramp started a period
late.) A resent state gets the same answer again. While a policy stream
runs, the station does not read the servos (manta_hand/runtime.py) and the joints arrive as NaN or stale, stamped with
the poll's time. Without fresh angles the controller takes its own last reference (HOM's theta_ref, which in contact
sits at the object's surface) as the joint angles: --joints reference, or auto (the bus's angles when finite and at most
--max-joint-age-ms older than the state, both of corvus's clock, else the reference). Its last targets (--joints targets)
are a worse stand-in: they hold the force and gravity offsets, which a re-anchor to them would add to the reference
again. The record says which source each period used.

Record (JSON lines, one file per trial in --log-dir): a header line {"kind": "header", ...}, one line per period in the
stream format of hom_bench.simulate (t, joints, obj_pos_mm, obj_quat, contacts, targets in servo degrees, phase,
clipped) with the link's sequence numbers and times, the object's tracker flags, the sim-degree targets sent, the
controller's compute time and the schedule's references, each finger's generalized contact frame as HOM took it (gcf:
its source, cop or witness, and the bodies' distance in mm), and an end line {"kind": "end", ...}.

The fingertips (--tips, 2026-11-17). corvus fills StateMsg.cop from the three boards (hom_trial.py, tip_contacts.py),
and every period's record keeps what arrived under "contacts". With --tips off (the default) HOM gets none of them and
takes every finger's contact frame from witness points on its model, as in the trials of 2026-11-16. With --tips on HOM
gets them as hom_tasks' tip variant does in simulation: a finger with 0.25 N or more (hom_tasks.TIP_F_MIN, the boards'
floor) takes its contact frame from its centre of pressure (hom.Mirror.gcf_from_cop). With --tips raw HOM gets them at
the plan's own floor; sil runs its server so, because the simulated hand sends the plant's contacts, as before.

    .venv/bin/python -m sr2hand.control.hom_server serve --task press --d 25 [--plan PLAN.json] [--plant measured]
        --listen 47002 --reply CORVUS:47001 --log-dir docs/experiments/20261115-hom_bench_trials/runs
    .venv/bin/python -m sr2hand.control.hom_server sil --task press --d 25 --seed 0 --log-dir /tmp/sil
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import servo_limits as SL

ROOT = Path(__file__).resolve().parents[3]
PLANTS = {"measured": "kp0.55_kv0_fr0.18_fl0_dp0.011", "servo": "servo"}
SCENES_2P5 = "logs/20261102-hom_servo_plant/scenes"      # cell scenes with pads at 2.5 MPa (reorient_backends.E_TPU)


def configure(plant: str = "measured", hand: str = "cad_v2"):
    """The hand and the plant for hom_tasks, before its first import in this process."""
    if "hom_tasks" in sys.modules or "hand_object_scale_sim" in sys.modules:
        raise RuntimeError("hom_tasks is already imported: configure() must run first")
    os.environ["SCALE_HAND"] = hand
    os.environ["HOS_PLANT"] = PLANTS.get(plant, plant)
    if hand == "cad_v2":                       # the cad_v2 cache at 2.5 MPa; other hands keep their own caches
        os.environ.setdefault("HOS_SCENES", SCENES_2P5)
    os.environ.setdefault("HOM_ROWS_TAG", "_bench")


def _bench(a):
    configure(a.plant, a.hand)
    from dataclasses import replace
    from . import hom_bench as HB
    kw = {"torque_budget": getattr(a, "torque_budget", None), "pose_tau": getattr(a, "pose_tau", None)}
    b = (HB.HomBench.from_plan(a.plan, a.task, a.d, a.dt, **kw) if a.plan
         else HB.HomBench.from_design(a.task, a.d, dt=a.dt, **kw))
    if getattr(a, "tips", "off") == "on":                 # the tip variant's floor (hom_tasks.VARIANTS["tip"])
        b.prm = replace(b.prm, contact_force_min=b.T.TIP_F_MIN)
        b.reset()
    over = {}                                             # HOM's options of 2026-11-17 (hom.py), off by default
    if getattr(a, "filter_step_s", None) or getattr(a, "filter_exact", False):
        over["filter"] = replace(b.prm.filter, step_s=getattr(a, "filter_step_s", None) or None,
                                 exact=bool(getattr(a, "filter_exact", False)))
    if getattr(a, "contact_off", None) is not None:
        over["contact_force_off"] = a.contact_off
    if getattr(a, "eq1_frames", "measured") != "measured":
        over["eq1_frames"] = a.eq1_frames
    if over:
        b.prm = replace(b.prm, **over)
        b.reset()
    return HB, b


def _git():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return str(v)


def _contacts(cop):
    """The StateMsg's 12 values (per finger x, y, z in P mm and the normal force N) -> {finger: ([x, y, z], F)}."""
    from .link import FINGERS
    c = np.asarray(cop, float).reshape(3, 4)
    return {f: (c[k, :3].tolist(), float(c[k, 3])) for k, f in enumerate(FINGERS) if np.all(np.isfinite(c[k]))}


def _sim_deg(HB, out):
    """HomBench's servo-degree targets -> the link's 9 sim degrees (yaw, mcp, pip per finger)."""
    q = HB.sim_from_servo(out)
    return np.array([math.degrees(float(v)) for f in HB.FINGERS for v in q[f]])


# ------------------------------------------------------------------------------------------------ serve
class Server:
    """One HomBench answering StateMsgs; a new trial starts at a state with seq 0 or after --gap-s without states."""

    def __init__(self, a, HB, bench):
        from . import link as L
        self.a, self.HB, self.b, self.L = a, HB, bench, L
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("0.0.0.0", a.listen))
        self.rx.settimeout(0.5)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host, port = a.reply.rsplit(":", 1)
        self.to = (host, int(port))
        self.log_dir = Path(a.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.fh = None
        self.trials = 0

    def _open(self, s):
        self.b.reset()
        self.t0_ms, self.seq, self.n, self.last_out, self.last_obj, self.last_ref = s.t_ms, 0, 0, None, None, None
        self.n_src = {"bus": 0, "reference": 0, "targets": 0, "open": 0}
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = self.log_dir / f"{stamp}_{self.a.task}_d{self.a.d:g}_trial{self.trials:03d}.jsonl"
        self.fh = open(self.path, "w")
        b = self.b
        head = {"kind": "header", "task": self.a.task, "d_mm": self.a.d, "plan": self.a.plan, "plant": self.a.plant,
                "plant_spec": os.environ.get("HOS_PLANT"), "hand": os.environ.get("SCALE_HAND"),
                "scenes": os.environ.get("HOS_SCENES"), "dt": b.prm.dt, "t_end": b.t_end,
                "design": b.setup.get("design"), "mounts_mm": {f: (1e3 * np.asarray(b.setup["mounts"][f][:2])).tolist()
                                                               for f in self.HB.FINGERS},
                "q_open_rad": b.q_open, "palm_m": b.palm, "prm": b.task_prm, "hold_s": b.hold_s,
                "torque_budget": b.torque_budget,
                "object_mass_kg": None if self.a.object_mass_g is None else self.a.object_mass_g / 1000.0,
                "git": _git(), "host": socket.gethostname(),
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "t0_ms": self.t0_ms,
                "joints_mode": self.a.joints, "max_joint_age_ms": self.a.max_joint_age_ms,
                "tips": self.a.tips, "contact_force_min_N": b.prm.contact_force_min,
                "contact_force_off_N": b.prm.contact_force_off, "filter_step_s": b.prm.filter.step_s,
                "filter_exact": b.prm.filter.exact,
                "eq1_frames": b.prm.eq1_frames}
        self.fh.write(json.dumps(head, default=_jsonable) + "\n")
        self.trials += 1
        print(f"trial {self.trials}: {self.path}", flush=True)

    def _close(self, reason):
        if self.fh:
            self.fh.write(json.dumps({"kind": "end", "reason": reason, "periods": self.n, "joints_src": self.n_src,
                                      "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
            self.fh.close()
            self.fh = None
            print(f"  ended ({reason}): {self.n} periods, joints from {self.n_src}", flush=True)

    def _joints(self, s):
        """The bus's joints when they are fresh (--joints auto or bus), else the controller's own last targets, else the
        open pose. The age is the state's time less the joints' time, both of corvus's clock."""
        j = np.asarray(s.joints, float)
        age = s.t_ms - s.t_joints_ms if math.isfinite(s.t_joints_ms) else math.inf
        if self.a.joints == "bus" or (self.a.joints == "auto" and np.all(np.isfinite(j))
                                      and age <= self.a.max_joint_age_ms):
            return s.joints_dict(), "bus", age
        if self.a.joints == "targets" and self.last_out is not None:
            return self.last_out, "targets", age
        if self.last_ref is not None:
            return self.last_ref, "reference", age
        return self.HB.servo_from_sim(self.b.q_open), "open", age

    def handle(self, s):
        L, b = self.L, self.b
        if self.fh is None or (s.seq == 0 and self.n > 0):
            if self.fh:
                self._close("new trial")
            self._open(s)
        if self.n > 0 and s.seq == self.last_state_seq:       # a resent state: the same answer again
            self.tx.sendto(self.last_msg, self.to)
            return False
        dt = self.b.prm.dt
        if self.n == 0:
            self.t = 0.0
        else:                                  # the periods elapsed on corvus's clock, summed as hom_tasks sums its time
            for _ in range(max(1, round((s.t_ms - self.last_t_ms) / 1000.0 / dt))):
                self.t += dt
        t, self.last_t_ms, self.last_state_seq = self.t, s.t_ms, s.seq
        joints, src, age = self._joints(s)
        self.n_src[src] += 1
        if np.all(np.isfinite(s.p_PO)) and np.all(np.isfinite(s.q_PO)):
            self.last_obj = (np.asarray(s.p_PO, float), np.asarray(s.q_PO, float))
        contacts = _contacts(s.cop)
        if self.last_obj is None:                                       # no pose yet: hold the open pose
            out, info = self.HB.servo_from_sim(b.q_open), {"phase": "no object", "clipped": [], "schedule": {}}
        else:
            tc = time.perf_counter()
            try:
                age_f = (s.t_ms - s.t_capture_ms) / 1000.0 if math.isfinite(s.t_capture_ms) else None
                out, info = b.step(t, joints, self.last_obj[0], self.last_obj[1],
                                   contacts if self.a.tips != "off" else None, frame_age_s=age_f,
                                   frame=s.t_capture_ms if math.isfinite(s.t_capture_ms) else None)
            except Exception as e:             # noqa: BLE001  e.g. a least-squares solve on a diverged state
                out = self.last_out if self.last_out is not None else self.HB.servo_from_sim(b.q_open)
                info = {"phase": "controller error", "clipped": [], "schedule": {}, "error": repr(e)}
                print(f"  controller error at t = {t:.2f} s: {e!r}; holding the last targets and stopping", flush=True)
            info["compute_ms"] = 1e3 * (time.perf_counter() - tc)
        stop = t > b.t_end + self.a.tail_s or "error" in info
        # every path's targets (the open pose before the first pose, the controller's, the held ones after an error)
        # inside the safe range (servo_limits; step() has clipped the controller's already)
        extra = SL.clip_servo(out)
        if extra:
            info["clipped"] = sorted(set(info.get("clipped") or []) | set(extra))
        sim_deg = _sim_deg(self.HB, out)
        self.last_msg = L.TargetMsg(seq=self.seq, state_seq=s.seq, t_ms=time.time() * 1000.0, targets=sim_deg,
                                    flags=L.STOP if stop else 0).pack()
        self.tx.sendto(self.last_msg, self.to)
        self.seq += 1
        self.last_out = out
        hom = info.get("hom") or {}
        if all(f in hom and "theta_ref" in hom[f] for f in self.HB.FINGERS):
            self.last_ref = self.HB.servo_from_sim({f: hom[f]["theta_ref"] for f in self.HB.FINGERS})
        sch = info.get("schedule", {})
        rec = {"t": round(t, 4), "seq": s.seq, "t_ms": s.t_ms, "t_joints_ms": s.t_joints_ms, "joints": joints,
               "joints_src": src, "joint_age_ms": round(age, 2) if math.isfinite(age) else None,
               "obj_pos_mm": None if self.last_obj is None else self.last_obj[0],
               "obj_quat": None if self.last_obj is None else self.last_obj[1], "t_capture_ms": s.t_capture_ms,
               "obj_flags": s.obj_flags, "n_tags": s.n_tags, "contacts": contacts, "status": s.status,
               "gcf": {f: [hom[f]["source"], round(1e3 * float(hom[f]["dist"]), 3)] for f in self.HB.FINGERS
                       if f in hom and "source" in hom[f]},
               "targets": out, "targets_sim_deg": sim_deg, "phase": info["phase"], "clipped": info["clipped"],
               "budget_scale": info.get("budget_scale"), "obj_filtered": info.get("pose_filtered"),
               "compute_ms": round(info.get("compute_ms", float("nan")), 3),
               "ref": {k: sch[k] for k in ("w_ref", "v_ref", "theta_ref") if k in sch}, "error": info.get("error")}
        self.fh.write(json.dumps(rec, default=_jsonable) + "\n")
        self.n += 1
        if stop:
            self._close("controller error: " + info["error"] if "error" in info else "end of schedule")
        return stop

    def run(self):
        last = time.time()
        while True:
            try:
                data = self.rx.recv(512)
            except socket.timeout:
                if self.fh and time.time() - last > self.a.gap_s:
                    self._close("no state for %.1f s" % self.a.gap_s)
                if self.a.once and self.trials and self.fh is None:
                    return 0
                continue
            s = self.L.StateMsg.unpack(data)
            if s is None:
                continue
            last = time.time()
            if self.handle(s) and self.a.once:
                return 0


def cmd_serve(a):
    HB, bench = _bench(a)
    print(f"HOM server: {a.task} d {a.d:g} mm, plant {os.environ['HOS_PLANT']}, hand {os.environ['SCALE_HAND']}, "
          f"listen {a.listen}, reply {a.reply}, t_end {bench.t_end:.2f} s", flush=True)
    return Server(a, HB, bench).run()


# ------------------------------------------------------------------------------------------------ sim hand
def _state_msg(HB, L, bench, s, seq, tm):
    """The StateMsg corvus would send for the plant state s at its time tm (ms)."""
    smp = HB.bench_sample(bench, s)
    cop = np.full(12, np.nan)
    for k, f in enumerate(HB.FINGERS):
        if f in smp["contacts"]:
            p, F = smp["contacts"][f]
            cop[4 * k:4 * k + 3], cop[4 * k + 3] = p, F
    return L.StateMsg(seq=seq, t_ms=tm, t_joints_ms=tm,
                      joints=np.array([smp["joints"][f][j] for f in HB.FINGERS for j in L.SERVO_JOINTS]),
                      t_capture_ms=tm, p_PO=np.asarray(smp["obj_pos_mm"]), q_PO=np.asarray(smp["obj_quat"]),
                      n_tags=6, cop=cop, status=L.DRY_RUN | L.STREAMING)


def loopback(HB, bench, seed: int):
    """hom_bench.simulate with every sample and target passed through the link's pack and unpack (float32), in
    process: what the UDP run must reproduce exactly."""
    from . import link as L
    T = bench.T
    plant = T.Plant(bench.p_pads, bench.setup)
    plant.hold_s = bench.hold_s
    plant.reset(bench.q_open, seed)
    bench.reset()
    dt, t_end = bench.prm.dt, bench.t_end
    t, lifted, s0, seq, rows, sent = 0.0, False, None, 0, [], []
    while t < t_end - 1e-9:
        s = plant.state()
        if t >= T.T_LIFT and not lifted:
            plant.post_off()
            lifted = True
        if s0 is None and t >= T.T_LIFT + 0.4 - 1e-9:
            s0 = s
        m = L.StateMsg.unpack(_state_msg(HB, L, bench, s, seq, 1000.0 * t).pack())
        out, info = bench.step(t, m.joints_dict(), np.asarray(m.p_PO, float), np.asarray(m.q_PO, float),
                               _contacts(m.cop))
        tg = L.TargetMsg.unpack(L.TargetMsg(seq=seq, state_seq=seq, t_ms=0.0, targets=_sim_deg(HB, out)).pack())
        plant.set_targets({f: np.radians(np.asarray(tg.targets, float).reshape(3, 3)[k])
                           for k, f in enumerate(HB.FINGERS)})
        sent.append(np.asarray(tg.targets, float))
        rows.append(HB._metric_row(bench.task, s, info["schedule"], t))
        seq += 1
        t += dt
        plant.advance(t)
    return T.metrics(bench.task, bench.setup, rows, s0, plant.state()), np.array(sent)


def sim_hand(a, HB, bench, port_out: int, port_in: int):
    """corvus's end against the pads plant, in lockstep: state, wait for its answer, apply, advance. Returns the run's
    hom_tasks.metrics (as hom_bench.simulate) and the targets applied per period (sim degrees)."""
    from . import link as L
    T = bench.T
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", port_in))
    rx.settimeout(1.0)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    plant = T.Plant(bench.p_pads, bench.setup)
    plant.hold_s = bench.hold_s
    plant.reset(bench.q_open, a.seed)
    sch = T.Schedule(bench.task, bench.setup, T.PRM)                   # the references for the metrics' rows
    dt, t_end = bench.prm.dt, bench.t_end
    t, lifted, s0, seq = 0.0, False, None, 0
    t0_ms = time.time() * 1000.0
    rows, sent = [], []
    clog = open(a.corvus_log, "w") if getattr(a, "corvus_log", None) else None    # corvus's record, as hom_trial.py
    if clog:
        clog.write(json.dumps({"kind": "header", "task": bench.task, "simulated": True, "t0_ms": t0_ms}) + "\n")
    while t < t_end - 1e-9:
        s = plant.state()
        if t >= T.T_LIFT and not lifted:
            plant.post_off()
            lifted = True
        if s0 is None and t >= T.T_LIFT + 0.4 - 1e-9:
            s0 = s
        msg = _state_msg(HB, L, bench, s, seq, t0_ms + 1000.0 * t)
        tg, waited = None, 0.0
        while tg is None:                     # resend until answered: the server may still be building its model
            tx.sendto(msg.pack(), ("127.0.0.1", port_out))
            t_send = time.time()
            while time.time() - t_send < 1.0:
                try:
                    r = L.TargetMsg.unpack(rx.recv(512))
                except socket.timeout:
                    break
                if r is not None and r.state_seq == seq:
                    tg = r
                    break
            waited += time.time() - t_send
            if tg is None and waited > a.wait_s:
                raise RuntimeError(f"no answer to state {seq} in {a.wait_s:.0f} s")
        if clog:
            clog.write(json.dumps({"seq": seq, "t_ms": msg.t_ms, "t": round(t, 4), "joints": msg.joints,
                                   "t_joints_ms": msg.t_joints_ms, "obj": {"p_PO": msg.p_PO, "q_PO": msg.q_PO,
                                                                            "t_capture_ms": msg.t_capture_ms},
                                   "arm": None, "ft": None, "rtt": 0.0, "target": tg.seq,
                                   "target_sim_deg": tg.targets}, default=_jsonable) + "\n")
        if tg.flags & L.STOP:                  # the controller stopped (its schedule's end or an error): so does corvus
            break
        q = {f: np.radians(np.asarray(tg.targets, float).reshape(3, 3)[k]) for k, f in enumerate(HB.FINGERS)}
        plant.set_targets(q)
        sent.append(np.asarray(tg.targets, float))
        _cmd, info = sch(t, {"quat": np.asarray(s["quat"], float)})
        rows.append(HB._metric_row(bench.task, s, info, t))
        seq += 1
        t += dt
        plant.advance(t)
    if clog:
        clog.write(json.dumps({"kind": "end", "states": seq, "lift": {"phase": "simulated (the post's removal)",
                                                                      "t_start_ms": t0_ms + 1000.0 * T.T_LIFT}}) + "\n")
        clog.close()
    return T.metrics(bench.task, bench.setup, rows, s0, plant.state()), np.array(sent)


def cmd_nominal(a):
    """The object's pose in P at the plant's start (seed 0) and the open pose, as JSON for hom_trial.py check."""
    HB, bench = _bench(a)
    plant = bench.T.Plant(bench.p_pads, bench.setup)
    plant.hold_s = bench.hold_s
    plant.reset(bench.q_open, 0)
    s = plant.state()
    out = {"task": a.task, "d_mm": a.d, "plan": a.plan, "hand": os.environ["SCALE_HAND"],
           "p_PO_mm": (1e3 * (np.asarray(s["pos"], float) - bench.palm)).round(4).tolist(),
           "q_PO": np.asarray(s["quat"], float).round(6).tolist(),
           "q_open_servo_deg": HB.servo_from_sim(bench.q_open), "design": bench.setup.get("design")}
    text = json.dumps(out, indent=1, default=_jsonable)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


def cmd_sim_hand(a):
    HB, bench = _bench(a)
    res, _sent = sim_hand(a, HB, bench, a.port_out, a.port_in)
    print(json.dumps(res, default=_jsonable))
    return 0


def cmd_sil(a):
    """The server in a child process, the simulated hand here, then hom_bench.simulate on the same bench and seed."""
    HB, bench = _bench(a)                     # builds the scenes once, before the server looks for them
    import hom_landscape as L_
    srv = [sys.executable, "-m", "sr2hand.control.hom_server", "serve", "--task", a.task, "--d", str(a.d),
           "--plant", a.plant, "--hand", a.hand, "--dt", str(a.dt), "--listen", str(a.port_out),
           "--reply", f"127.0.0.1:{a.port_in}", "--log-dir", a.log_dir, "--once", "--joints", a.server_joints,
           "--tail-s", "0", "--tips", "raw"] + (["--plan", a.plan] if a.plan else [])
    srv += (["--filter-step-s", str(a.filter_step_s)] if a.filter_step_s else []) + \
        (["--filter-exact"] if a.filter_exact else []) + \
        (["--contact-off", str(a.contact_off)] if a.contact_off is not None else []) + ["--eq1-frames", a.eq1_frames]
    proc = subprocess.Popen(srv, cwd=str(ROOT), env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    try:
        t0 = time.time()
        res_link, sent = sim_hand(a, HB, bench, a.port_out, a.port_in)
        wall = time.time() - t0
    finally:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    try:
        res_lb, sent_lb, err_lb = *loopback(HB, bench, a.seed), None
    except Exception as e:                     # noqa: BLE001  the same failure as the server's, in process
        res_lb, sent_lb, err_lb = {"held": None}, np.zeros((0, 9)), repr(e)
    try:
        stream, res_in = HB.simulate(bench, a.seed)
        tgt_in = np.array([[math.degrees(float(v)) for f in HB.FINGERS for v in HB.sim_from_servo(x["targets"])[f]]
                           for x in stream])
        err_in = None
    except Exception as e:                     # noqa: BLE001  a diverged float64 run (roll layout, cylinder tasks)
        res_in, tgt_in, err_in = {"held": None}, np.zeros((0, 9)), repr(e)

    def cmp(x, y):
        n = min(len(x), len(y))
        d = np.abs(x[:n] - y[:n]).max(1) if n else np.zeros(0)
        return {"max_deg": float(d.max()) if n else None,
                "first_period_over_1e-3": int(np.argmax(d > 1e-3)) if n and d.max() > 1e-3 else None}
    out = {"task": a.task, "seed": a.seed, "plant": os.environ["HOS_PLANT"], "hand": os.environ["SCALE_HAND"],
           "periods": [len(sent), len(sent_lb), len(tgt_in)],
           "J": {"udp": L_.track_score(a.task, res_link) if res_link.get("held") is not None else None,
                 "loopback": None if err_lb else L_.track_score(a.task, res_lb), "loopback_error": err_lb,
                 "float64": None if err_in else L_.track_score(a.task, res_in)}, "float64_error": err_in,
           "held": [res_link.get("held"), res_lb.get("held"), res_in.get("held")],
           "targets_udp_vs_loopback": cmp(sent, sent_lb), "targets_udp_vs_float64": cmp(sent, tgt_in),
           "wall_s": round(wall, 1)}
    print(json.dumps(out, default=_jsonable), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "sim-hand", "sil", "nominal"):
        p = sub.add_parser(name)
        p.add_argument("--task", required=True, choices=["pinpush", "press", "roll", "twist"])
        p.add_argument("--d", type=float, required=True, help="object diameter, mm")
        p.add_argument("--plan", default=None, help="deploy plan JSON whose mounts must be the design's layout")
        p.add_argument("--plant", default="measured", help="measured (0.55 N m/rad), servo (0.72) or a HOS_PLANT spec")
        p.add_argument("--hand", default="cad_v2")
        p.add_argument("--dt", type=float, default=0.02)
        p.add_argument("--torque-budget", type=float, default=None,
                       help="N m: the servos' estimated torques summed over the nine kept at or under this "
                            "(hom_bench.torque_budget; 0.84 after the blind twists of 2026-11-16)")
        p.add_argument("--filter-step-s", type=float, default=None,
                       help="s: the complementary filter (Eqs. 4-6) in sub-steps of at most this inside each period; "
                            "0.002 is the paper's 500 Hz (FilterParams.step_s); off by default: one step per period")
        p.add_argument("--filter-exact", action="store_true",
                       help="the filter's correction (Eq. 5) integrated exactly over the period, k and the measurement "
                            "held (FilterParams.exact): the continuous filter at any rate, at the cost of one step")
        p.add_argument("--contact-off", type=float, default=None,
                       help="N: a finger in contact counts as apart only below this sensed force (hysteresis below "
                            "the contact floor; HomParams.contact_force_off); off by default")
        p.add_argument("--eq1-frames", default="measured", choices=("measured", "filtered"),
                       help="Eq. (1)'s frames in contact: measured (the GCF from the centre of pressure, the paper's "
                            "text) or filtered (the filter's {C0}, {C1}; a test)")
        if name in ("serve", "sil"):
            p.add_argument("--log-dir", required=True)
        if name == "nominal":
            p.add_argument("--out", default=None)
        if name == "serve":
            p.add_argument("--listen", type=int, default=47002)
            p.add_argument("--reply", required=True, help="corvus's HOST:PORT (hand_link --listen)")
            p.add_argument("--joints", default="auto", choices=("auto", "bus", "reference", "targets"),
                           help="joint angles from the bus when fresh, else HOM's last reference (auto); always the bus; "
                                "always the reference; or the last targets")
            p.add_argument("--max-joint-age-ms", type=float, default=60.0)
            p.add_argument("--gap-s", type=float, default=1.0, help="a trial ends after this long without a state")
            p.add_argument("--tail-s", type=float, default=0.5, help="periods past the schedule's end before STOP")
            p.add_argument("--once", action="store_true", help="exit after one trial")
            p.add_argument("--object-mass-g", type=float, default=None,
                           help="the object's measured mass with stem and block, for the record")
            p.add_argument("--tips", default="off", choices=("off", "on", "raw"),
                           help="the fingertips' contacts (StateMsg.cop): off, recorded and not given to HOM (as on "
                                "2026-11-16); on, given to HOM as hom_tasks' tip variant does (a finger counts from "
                                "0.25 N); raw, given at the plan's own floor (sil's simulated hand)")
            p.add_argument("--pose-tau", type=float, default=None,
                           help="the tracked pose's filter time constant, s (hom_bench.HomBench._pose_filter, as "
                                "hom_gpu's POSE_FILTER; 0.2 in the twist's sensing study); off by default")
        elif name != "nominal":
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--port-out", type=int, default=47102)
            p.add_argument("--port-in", type=int, default=47101)
            p.add_argument("--wait-s", type=float, default=180.0, help="longest wait for an answer (the server's build)")
            p.add_argument("--server-joints", default="auto", help="sil: the server's --joints")
            p.add_argument("--corvus-log", default=None, help="also write corvus's record (hom_trial.py's format)")
    a = ap.parse_args()
    if a.cmd == "serve":
        return cmd_serve(a)
    if a.cmd == "nominal":
        return cmd_nominal(a)
    return cmd_sim_hand(a) if a.cmd == "sim-hand" else cmd_sil(a)


if __name__ == "__main__":
    sys.exit(main())
