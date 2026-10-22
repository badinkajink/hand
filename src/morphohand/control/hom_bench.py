"""The HOM controller (hom.py) on the bench: a thin adapter between the bench's signals and HomController.

Inputs, per control period (50 Hz on the bench):
  joints     servo-bus positions per finger in the servo's own zero-relative degrees, keyed by the firmware joint
             names aa, fe1, fe2 (manta_hand.plan: sim yaw/mcp/pip = aa/fe1/fe2, sim degrees = JOINT_SIGN x servo
             degrees, no offset term)
  object     the object body's pose in the palm frame {P} from the tag tracker: position in mm, quaternion (w, x, y,
             z); {P} is the sim's palm frame (manta_hand.plan, item 1) and the body frame is the scene's (cylinder
             axis = body z)
  contacts   optional fingertip contact estimates per finger: centre of pressure in {P} in mm and normal force in N;
             without them every finger's generalized contact frame comes from witness points on the kinematic model
Output: servo targets per finger in the servo's degrees (aa, fe1, fe2), clipped to the servo's calibrated range
(manta_hand.servos.FINGER_JOINTS), and the controller's info.

The controller, its kinematic model (the Mirror) and the task schedule are those of scripts/hom_tasks.py for one
task and object size on the CAD-model hand: HomBench.from_design takes an entry of the study's design.json (the
layout, palm height and grasp), HomBench.from_plan a deploy plan whose mounts must be that layout. The two dry-run
modes, at 50 Hz:
  simulate   closed loop on the pads plant of hom_tasks.py; every sample goes through the bench units both ways and
             is recorded as a stream (JSON lines: t, joints, obj_pos_mm, obj_quat, contacts, targets)
  replay     open loop over a recorded or simulated stream: the targets a fresh controller would send

    .venv/bin/python -m morphohand.control.hom_bench simulate --task roll --d 40 --seconds 3 --out /tmp/s.jsonl
    .venv/bin/python -m morphohand.control.hom_bench replay --task roll --d 40 --stream /tmp/s.jsonl --out /tmp/r.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import hom as H

ROOT = Path(__file__).resolve().parents[3]
FINGERS = ("thumb", "index", "middle")
SIM_JOINTS = ("yaw", "mcp", "pip")
DT_BENCH = 0.02                                   # s: the servo bus's closed-loop rate


def _paths():
    """scripts/ (hom_tasks and the scene builders) and the manta host package (plan, servos) on sys.path."""
    for p in (ROOT / "scripts", ROOT / "src/morphohand/driver/manta/host"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _driver():
    _paths()
    from manta_hand import plan as P
    from manta_hand import servos as SV
    return P, SV


def servo_from_sim(q: dict) -> dict:
    """Sim joint angles (rad) per finger -> servo degrees {finger: {aa, fe1, fe2}}."""
    P, _ = _driver()
    return {f: {P.SIM_JOINT_TO_SERVO[j]: P.servo_deg(f, j, math.degrees(float(q[f][k])))
                for k, j in enumerate(SIM_JOINTS)} for f in FINGERS}


def sim_from_servo(joints: dict) -> dict:
    """Servo degrees {finger: {aa, fe1, fe2}} -> sim joint angles (rad) per finger."""
    P, _ = _driver()
    return {f: np.array([math.radians(float(joints[f][P.SIM_JOINT_TO_SERVO[j]]) / P.JOINT_SIGN[(f, j)])
                         for j in SIM_JOINTS]) for f in FINGERS}


def servo_range(finger: str, servo: str) -> tuple[float, float]:
    P, SV = _driver()
    return tuple(SV.FINGER_JOINTS[P.FINGER_ID[finger]][servo][2])


class HomBench:
    """HomController, its Mirror and the task schedule of hom_tasks.py for one task and object on one layout."""

    def __init__(self, task: str, d_mm: float, setup: dict, m_pt, q_open: dict, dt: float = DT_BENCH,
                 prm: H.HomParams | None = None, p_pads: Path | None = None):
        _paths()
        import hom_tasks as T
        self.T = T
        self.task, self.d_mm, self.setup, self.p_pads = task, float(d_mm), setup, p_pads
        self.palm = np.asarray(setup["palm"], float)          # {P} origin in the model's world frame, no rotation
        fs, os_ = T.sr2_specs(setup)
        self.mirror = H.Mirror(m_pt, fs, os_)
        self.prm = replace(prm or H.HomParams(), dt=dt)
        self.q_open = {f: np.asarray(q_open[f], float) for f in FINGERS}
        self.reset()

    # -------------------------------------------------------------------------------------------- builders
    @staticmethod
    def _cell(task: str, d_mm: float, entry: dict | None):
        _paths()
        import mujoco
        import hand_object_scale_sim as S
        import hom_tasks as T
        if entry is not None:                                 # a design entry from elsewhere: stage it for build()
            dj = json.loads((T.OUT / "design.json").read_text())
            key = f"{T.SHAPE[task]}{float(d_mm):g}"
            if dj.get(key) != entry:
                raise ValueError(f"the design entry differs from design.json[{key!r}]; add it there first")
        p_pads, p_pt, setup = T.build(task, float(d_mm))
        m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
        S.set_palm(m_pt, setup["palm"])
        q_open, _ = T.open_pose(setup, mujoco.MjModel.from_xml_path(str(p_pt)))
        return setup, m_pt, q_open, p_pads

    @classmethod
    def from_design(cls, task: str, d_mm: float, entry: dict | None = None, dt: float = DT_BENCH, **kw):
        """The layout, palm height and grasp of the study's design.json entry for this object."""
        setup, m_pt, q_open, p_pads = cls._cell(task, d_mm, entry)
        return cls(task, d_mm, setup, m_pt, q_open, dt, p_pads=p_pads, **kw)

    @classmethod
    def from_plan(cls, plan: str | Path | dict, task: str, d_mm: float, dt: float = DT_BENCH, tol_mm: float = 0.5,
                  **kw):
        """A deploy plan (manta_hand.plan.HandPlan JSON): its mounts must be the CAD-hand layout of the design entry
        within tol_mm; the plan's "open" pose, when it has one, replaces the computed open pose."""
        raw = json.loads(Path(plan).read_text()) if not isinstance(plan, dict) else plan
        setup, m_pt, q_open, p_pads = cls._cell(task, d_mm, None)
        for f in FINGERS:
            want = 1e3 * np.asarray(setup["mounts"][f][:2], float)
            got = np.asarray(raw["mounts_palm_mm"][f], float)
            if np.linalg.norm(want - got) > tol_mm:
                raise ValueError(f"{f} mount {got.tolist()} mm is not the layout's {want.round(2).tolist()} mm "
                                 f"({setup['design']['tag']})")
        for pose in raw.get("poses", []):
            if pose["name"] == "open":
                q_open = {f: np.radians([pose["joints"][f][j] for j in SIM_JOINTS]) for f in FINGERS}
        return cls(task, d_mm, setup, m_pt, q_open, dt, p_pads=p_pads, **kw)

    # -------------------------------------------------------------------------------------------- control
    def reset(self):
        self.ctl = H.HomController(self.mirror, self.prm, healthy=self.q_open)
        self.sch = self.T.Schedule(self.task, self.setup, self.T.PRM)

    def measurement(self, joints: dict, obj_pos_mm, obj_quat, contacts: dict | None = None) -> H.Measurement:
        """Bench units -> the controller's Measurement (model world frame, m, rad)."""
        cop = {}
        for f, (p_mm, F) in (contacts or {}).items():
            cop[f] = (self.palm + 1e-3 * np.asarray(p_mm, float), float(F))
        return H.Measurement(q=sim_from_servo(joints), obj_pos=self.palm + 1e-3 * np.asarray(obj_pos_mm, float),
                             obj_quat=np.asarray(obj_quat, float), cop=cop)

    def step(self, t: float, joints: dict, obj_pos_mm, obj_quat, contacts: dict | None = None):
        """One control period at time t (s since the run started): servo targets {finger: {aa, fe1, fe2}} in servo
        degrees and the info of this period (phase, clipped joints, the controller's per-finger info)."""
        meas = self.measurement(joints, obj_pos_mm, obj_quat, contacts)
        s = {"quat": meas.obj_quat}
        cmd, info = self.sch(t, s)
        for f in info.get("reanchor", ()):
            self.ctl.reanchor(f, meas.q[f])
        q_cmd, ci = self.ctl.step(meas, cmd)
        out = servo_from_sim(q_cmd)
        clipped = []
        for f in FINGERS:
            for sv, v in out[f].items():
                lo, hi = servo_range(f, sv)
                if not lo - 1e-3 <= v <= hi + 1e-3:           # the scene's ranges are the servo's to 1e-6 rad
                    clipped.append(f"{f}.{sv}")
                    out[f][sv] = min(max(v, lo), hi)
        return out, {"phase": info["phase"], "clipped": clipped, "q_cmd": q_cmd, "hom": ci, "schedule": info}

    @property
    def t_end(self) -> float:
        return self.sch.t_end


# ------------------------------------------------------------------------------------------------ dry runs

def bench_sample(bench: HomBench, s: dict) -> dict:
    """A hom_tasks.Plant state in bench units (the stream's sample)."""
    return {"joints": servo_from_sim(s["q"]), "obj_pos_mm": (1e3 * (s["pos"] - bench.palm)).tolist(),
            "obj_quat": np.asarray(s["quat"], float).tolist(),
            "contacts": {f: ((1e3 * (p - bench.palm)).tolist(), float(F)) for f, (p, F) in s["cop"].items()}}


def simulate(bench: HomBench, seed: int = 0, seconds: float | None = None, contacts: bool = True):
    """Closed loop on the pads plant of hom_tasks.py through the bench units at 1/dt Hz. Returns the stream (one
    sample per period with the targets sent) and the hom_tasks.metrics of the run."""
    T = bench.T
    plant = T.Plant(bench.p_pads, bench.setup)
    plant.reset(bench.q_open, seed)
    bench.reset()
    dt = bench.prm.dt
    t_end = bench.t_end if seconds is None else min(seconds, bench.t_end)
    t, lifted, s0 = 0.0, False, None
    stream, rows = [], []
    while t < t_end - 1e-9:
        s = plant.state()
        if t >= T.T_LIFT and not lifted:
            plant.post_off()
            lifted = True
        if s0 is None and t >= T.T_LIFT + 0.4 - 1e-9:
            s0 = s
        smp = bench_sample(bench, s)
        out, info = bench.step(t, smp["joints"], smp["obj_pos_mm"], smp["obj_quat"],
                               smp["contacts"] if contacts else None)
        plant.set_targets(sim_from_servo(out))
        stream.append({"t": round(t, 4), **smp, "targets": out, "phase": info["phase"], "clipped": info["clipped"]})
        rows.append(_metric_row(bench.task, s, info["schedule"], t))
        t += dt
        plant.advance(t)
    res = T.metrics(bench.task, bench.setup, rows, s0, plant.state()) if seconds is None else {}
    return stream, res


def _metric_row(task, s, info, t):
    """The fields of a hom_tasks.rollout row that hom_tasks.metrics reads, rounded as there."""
    R = H.quat_to_mat(s["quat"])
    row = {"t": round(t, 3), "phase": info["phase"], "pos_mm": (1e3 * s["pos"]).round(3).tolist(),
           "w_dps": np.degrees(s["w"]).round(2).tolist(), "axis": R[:, 2].round(5).tolist(),
           "F": {f: round(s["cop"].get(f, (None, 0.0))[1], 3) for f in FINGERS},
           "F_link": {f: round(max(0.0, s["Ftot"][f] - s["cop"].get(f, (None, 0.0))[1]), 3) for f in FINGERS}}
    row["cop_obj_mm"] = {f: (1e3 * (R.T @ (p - s["pos"]))).round(3).tolist() for f, (p, _F) in s["cop"].items()}
    if "w_ref" in info:
        row["w_ref_dps"] = np.degrees(info["w_ref"]).round(2).tolist()
        row["v_ref_mms"] = (1e3 * info["v_ref"]).round(3).tolist()
    if "theta_ref" in info:
        row["theta_ref_deg"] = round(info["theta_ref"], 3)
    return row


def replay(bench: HomBench, stream) -> list[dict]:
    """Open loop over recorded samples: the targets a fresh controller sends for each."""
    bench.reset()
    out = []
    for smp in stream:
        tg, info = bench.step(float(smp["t"]), smp["joints"], smp["obj_pos_mm"], smp["obj_quat"], smp.get("contacts"))
        out.append({"t": smp["t"], "targets": tg, "phase": info["phase"], "clipped": info["clipped"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("simulate", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--task", required=True, choices=["pinpush", "press", "roll", "twist"])
        p.add_argument("--d", type=float, required=True, help="object diameter, mm")
        p.add_argument("--plan", default=None, help="deploy plan JSON whose mounts must be the design's layout")
        p.add_argument("--dt", type=float, default=DT_BENCH)
        p.add_argument("--out", required=True)
        if name == "simulate":
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--seconds", type=float, default=None)
            p.add_argument("--no-contacts", action="store_true", help="witness points only (no contact estimates)")
        else:
            p.add_argument("--stream", required=True)
    a = ap.parse_args()
    bench = HomBench.from_plan(a.plan, a.task, a.d, a.dt) if a.plan else HomBench.from_design(a.task, a.d, dt=a.dt)
    if a.cmd == "simulate":
        stream, res = simulate(bench, a.seed, a.seconds, not a.no_contacts)
        lines = stream
        print(json.dumps(res))
    else:
        lines = replay(bench, [json.loads(x) for x in Path(a.stream).read_text().splitlines() if x.strip()])
    Path(a.out).write_text("".join(json.dumps(x, default=_jsonable) + "\n" for x in lines))
    print(f"{len(lines)} samples -> {a.out}")
    return 0


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v))


if __name__ == "__main__":
    sys.exit(main())
