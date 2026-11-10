#!/usr/bin/env python3
"""Inverse-dynamics torque control of the SR2 hand's fingers under HOM, against the position servos (owner, 2026-11-09:
"implement, test and analyse inverse-dynamics torque control of the fingers in place of MuJoCo's position actuators";
"that might help things?").

HOM's planner is unchanged. Each control period (10 ms) solves Eq. (2) for each finger's joint velocities u and
integrates them into the joint reference theta_ref; the impedance step moves the reference's action point onto the
measured surface (the offset off_gap) and names the commanded pad force F along the action direction n. The position
plant (hom_tasks.Plant, the SCS0009 model: MuJoCo position actuators at kp 0.72 N m/rad, 0.18 N m) receives the servo
targets theta_ref + off_gap + J^T F n / kp + tau_g / kp. The torque plant (TorquePlant) turns the finger actuators into
motors and sets, at every 1 ms physics step,

    tau = M(q) qdd_ref + c(q, qd) + g(q) - qfrc_passive + J(p)^T F n,
    qdd_ref = wn^2 (theta_t - q) + 2 zeta wn (theta_t' - qd),

clipped at the SCS0009's 0.18 N m. theta_t runs from the period's previous reference to its new one at Eq. (2)'s rate,
plus off_gap; M is the fingers' joint-space inertia with the servos' armature, c + g is MuJoCo's qfrc_bias, the passive
term is the joints' damping, and J is the pad body's Jacobian at the action point p. F n is the force the pad applies to
the object; the owner's - J^T f_cmd is the same term with f_cmd the force on the finger. With wn = sqrt(0.72 / 1.1e-3)
= 25.6 rad/s the loop has the servo's stiffness on the fingers' inertia, critically damped (the position plant's
damping ratio is 0.26); 63 rad/s is a 10 Hz loop, 2.5 times as fast.

The servo of the built hand takes positions only. A torque tau reaches it as a position offset tau / kp through its
spring, 0.55 to 0.72 N m/rad (docs/experiments/20261106-sr2_servo_sysid), with no drive inside its 0.59 deg dead band:
torques under about 6 to 7 mN m are lost, and the first drive outside the band is 13.7 mN m.

hom_tasks.rollout calls plant.set_hom(ci) with the controller's terms of each period when the plant has that method.

  .venv/bin/python scripts/hom/hom_torque.py smoke               # one twist layout, both plants, seed 0
  .venv/bin/python scripts/hom/hom_torque.py compare [--part k --parts n]   # rows/20261110-torque_compare.jsonl
  .venv/bin/python scripts/hom/hom_torque.py report              # tables of the comparison
Plant and scenes: HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25 (the films' plant).
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_gpu as HG  # noqa: E402
import hom_landscape as L  # noqa: E402
import hom_tasks as T  # noqa: E402

S, H = T.S, T.H
FINGERS = T.FINGERS
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/experiments/20261110-hom_torque_control"
ROWS = OUT / "rows" / "20261110-torque_compare.jsonl"
PICKS = HG.ROWS / "20261109-track_picks_sv25.json"      # the tracking page's per-size layouts (Table 3)
SIZES = {"twist": (25.0, 30.0, 63.0, 80.0), "roll": (25.0, 30.0, 63.0, 80.0), "press": (10.0, 13.0, 40.0, 50.0)}
GENERALIST = (70.0, 110.0)
SEEDS = (0, 1, 2)
TAU_MAX = 0.18                                            # N m: the SCS0009 plant's force range
WN = {"servo": 25.6, "fast": 63.0}                        # rad/s: the servo's stiffness on the fingers' inertia; 10 Hz
F_ON = 0.05                                               # N: a pad in contact (the twist's contact threshold)
AXIS = {"twist": 2, "roll": 1, "press": 0}                # the commanded rotation's axis in the palm (world) frame


class PositionPlant(T.Plant):
    """hom_tasks.Plant counting the physics steps on which a finger actuator is at its force limit."""

    def __init__(self, path, setup):
        super().__init__(path, setup)
        self.acts = np.concatenate([self.act[f] for f in FINGERS])
        self.lim = np.abs(self.m.actuator_forcerange[self.acts, 1])
        self.n_steps = self.n_sat = 0

    def advance(self, t1):
        m, d = self.m, self.d
        while d.time < t1 - 1e-9:
            self.mj.mj_step(m, d)
            self.n_steps += 1
            self.n_sat += int(np.any(np.abs(d.actuator_force[self.acts]) >= 0.999 * self.lim))
        self._hold()

    def _hold(self):
        if self.hold_s and self.d.time <= self.hold_s + 1e-9:          # as hom_tasks.Plant.advance
            self.d.qpos[self.qa:self.qa + 7] = self.obj0
            self.d.qvel[self.va:self.va + 6] = 0.0
            self.mj.mj_forward(self.m, self.d)


class TorquePlant(PositionPlant):
    """The fingers' actuators as motors under inverse dynamics (module docstring)."""

    def __init__(self, path, setup, wn=WN["servo"], zeta=1.0, tau_max=TAU_MAX, dt_c=T.DT_C):
        super().__init__(path, setup)
        m, mj = self.m, self.mj
        for a in self.acts:
            m.actuator_gaintype[a] = mj.mjtGain.mjGAIN_FIXED
            m.actuator_gainprm[a, :] = 0.0
            m.actuator_gainprm[a, 0] = 1.0
            m.actuator_biastype[a] = mj.mjtBias.mjBIAS_NONE
            m.actuator_biasprm[a, :] = 0.0
            m.actuator_ctrllimited[a] = 1
            m.actuator_ctrlrange[a] = (-tau_max, tau_max)
        self.wn, self.zeta, self.tau_max, self.dt_c = wn, zeta, tau_max, dt_c
        self.tip = {f: m.body(f"{f}_tip").id for f in FINGERS}
        self.jacp, self.jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.Mfull = np.zeros((m.nv, m.nv))
        self.ref, self.t_ref = None, 0.0

    def reset(self, q0, seed):
        super().reset(q0, seed)
        self.d.ctrl[self.acts] = 0.0
        self.ref = {f: (np.asarray(q0[f], float), np.asarray(q0[f], float), np.zeros(3)) for f in FINGERS}
        self.t_ref = 0.0

    def set_targets(self, q_cmd):
        """The position plant's servo targets: not used by the motors."""

    def set_hom(self, ci):
        """The controller's terms of this period: the reference before and after (with off_gap) and J^T F n."""
        m, d = self.m, self.d
        for f in FINGERS:
            c = ci[f]
            tau_f = np.zeros(3)
            if c["force"] > 0:
                self.mj.mj_jac(m, d, self.jacp, self.jacr, np.asarray(c["p_act"], float), self.tip[f])
                tau_f = self.jacp[:, self.vadr[f]].T @ (c["force"] * np.asarray(c["n_act"], float))
            self.ref[f] = (np.asarray(c["theta_ref_prev"], float) + c["off_gap"],
                           np.asarray(c["theta_ref"], float) + c["off_gap"], tau_f)
        self.t_ref = float(d.time)

    def advance(self, t1):
        mj, m, d = self.mj, self.m, self.d
        while d.time < t1 - 1e-9:
            mj.mj_step1(m, d)
            mj.mj_fullM(m, self.Mfull, d.qM)
            s = min(1.0, max(0.0, (d.time - self.t_ref) / self.dt_c))
            sat = False
            for f in FINGERS:
                v = self.vadr[f]
                th0, th1, tau_f = self.ref[f]
                qdd = (self.wn ** 2 * (th0 + s * (th1 - th0) - d.qpos[self.qadr[f]])
                       + 2.0 * self.zeta * self.wn * ((th1 - th0) / self.dt_c - d.qvel[v]))
                tau = self.Mfull[np.ix_(v, v)] @ qdd + d.qfrc_bias[v] - d.qfrc_passive[v] + tau_f
                sat = sat or bool(np.any(np.abs(tau) >= self.tau_max))
                d.ctrl[self.act[f]] = np.clip(tau, -self.tau_max, self.tau_max)
            mj.mj_step2(m, d)
            self.n_steps += 1
            self.n_sat += int(sat)
        self._hold()


# ------------------------------------------------------------------------------------------------ rollouts

_found = {}


def find(task, size, call, grasp):
    """(key, mounts, setup) of one layout and grasp of the 2-D lattice (hom_gpu.layout_list, cached per task and size)."""
    k = (task, float(size))
    if k not in _found:
        _found[k] = {(int(key["call"]), int(key.get("grasp", 0))): (key, mxy, setup)
                     for key, mxy, setup in HG.layout_list(task, "grid2dff_size", 0, size)}
    return copy.deepcopy(_found[k][(int(call), int(grasp))])


def call_at(task, size, x, y):
    return next(c for c, (a, b, _r) in L.grid_poses(task, size, "grid2dff").items() if a == x and b == y)


def task_metrics(task, rows):
    """Per finger the share of the task's periods with its pad loaded (F > F_ON) and its median pad force over those
    periods (N); the RMS angular velocity of the object off the commanded axis (deg/s)."""
    tr = [r for r in rows if r["t"] >= T.T_LIFT + 0.4 - 1e-9]
    if not tr:
        return {}
    w = np.array([r["w_dps"] for r in tr], float)
    off = np.delete(w, AXIS[task], axis=1)
    on = {f: [r["F"][f] for r in tr if r["F"][f] > F_ON] for f in FINGERS}
    return {"contact_frac": {f: round(len(on[f]) / len(tr), 3) for f in FINGERS},
            "force_med_N": {f: round(float(np.median(on[f])), 3) if on[f] else None for f in FINGERS},
            "off_axis_rms_dps": round(float(np.sqrt(np.mean(np.sum(off ** 2, axis=1)))), 2)}


def run(task, size, call, grasp, seed, plant="position", wn=WN["servo"], render=False):
    """One CPU rollout with every contact on the position or the torque plant: the rollout's metrics, both scores,
    the per-finger contact shares, the off-axis angular velocity and the share of physics steps at the torque limit.
    render=True: (row, frames of hom_films.Film every fourth period)."""
    import mujoco
    _key, mxy, setup = find(task, size, call, grasp)
    scene = HG.Scene(T.SHAPE[task], size)
    bm, q_open = scene.prepare(mxy, setup)
    pl = TorquePlant(scene.pads_path, setup, wn=wn) if plant == "torque" else PositionPlant(scene.pads_path, setup)
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for m in (pl.m, m_pt):
        for f in FINGERS:
            m.body_pos[m.body(f"{f}_mount").id] = bm[f]
    S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = H.Mirror(m_pt, fs, os_)
    var = "hold" if T.SHAPE[task] == "cylinder" else "base"
    over, noise = T.VARIANTS[var]
    film = None
    if render:
        import hom_films as HF
        film = HF.Film(pl, setup)
    t0 = time.time()
    try:
        res, rows, frames = T.rollout(task, setup, pl, mirror, q_open, seed, T.PRM,
                                      replace(H.HomParams(dt=T.DT_C), **over), film, noise,
                                      T.PLANT_VARIANTS.get(var, {}).get("hold_s", 0.0))
    finally:
        if film is not None:
            film.close()
    keep = ("held", "dropped", "fingers_end", "track_err_rms_deg", "track_ref_rms_deg", "contact_all_frac",
            "ramp_three_pads_max_deg", "ramp_end_deg", "w_rmse_ratio", "theta_rmse_deg")
    x, y, _r = L.grid_poses(task, size, "grid2dff")[call]
    row = dict(task=task, size=float(size), call=int(call), grasp=int(grasp), layout=[x, y], seed=int(seed),
               plant=plant, wn=float(wn) if plant == "torque" else None, **{k: res.get(k) for k in keep},
               track=round(L.track_score(task, res), 3), old=round(L.rollout_score(task, res), 3),
               **task_metrics(task, rows), sat_share=round(pl.n_sat / max(1, pl.n_steps), 4),
               wall_s=round(time.time() - t0, 1))
    return (row, frames) if render else row


def configs():
    """(task, size, call, grasp, which) of the comparison: each size's layout picked by the tracking score (the
    tracking page's Table 3) and layout (70, 110) mm with its first grasp, the foundational pose, as filmed."""
    picks = json.loads(PICKS.read_text())
    out = []
    for task, sizes in SIZES.items():
        for size in sizes:
            p = picks[task][f"{size:g}"]
            out.append((task, size, int(p["call"]), int(p["grasp"]), "pick"))
            try:
                c = call_at(task, size, *GENERALIST)
            except StopIteration:
                continue
            grasps = sorted(int(k.get("grasp", 0)) for k, _m, _s in HG.layout_list(task, "grid2dff_size", 0, size)
                            if int(k["call"]) == c)
            if grasps:
                out.append((task, size, c, grasps[0], "generalist"))
    return out


def _done():
    if not ROWS.exists():
        return set()
    return {(r["task"], r["size"], r["call"], r["grasp"], r["seed"], r["plant"], r.get("wn"))
            for r in (json.loads(x) for x in ROWS.read_text().splitlines() if x.strip())}


def _write(row):
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    with ROWS.open("a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def cmd_compare(a):
    jobs = []
    for task, size, call, grasp, which in configs():
        for seed in SEEDS:
            jobs.append((task, size, call, grasp, seed, "position", None, which))
            for wn in a.wn:
                jobs.append((task, size, call, grasp, seed, "torque", float(wn), which))
    jobs = jobs[a.part::a.parts]
    done = _done()
    for task, size, call, grasp, seed, plant, wn, which in jobs:
        if (task, float(size), call, grasp, seed, plant, wn) in done:
            continue
        try:
            row = run(task, size, call, grasp, seed, plant, wn or WN["servo"])
        except np.linalg.LinAlgError as e:               # the state diverged and HOM's least squares failed
            x, y, _r = L.grid_poses(task, size, "grid2dff")[call]
            row = dict(task=task, size=float(size), call=call, grasp=grasp, layout=[x, y], seed=seed, plant=plant,
                       wn=wn, diverged=str(e), track=0.0, old=0.0)
        row["which"] = which
        _write(row)
        print(f"{task:5s} {size:4.0f} {which:10s} {str(tuple(int(v) for v in row['layout'])):>11s} s{seed} "
              f"{plant:8s} {wn or 0:5.1f}  track {row['track']:.2f} old {row['old']:.2f} 3-pad "
              f"{row.get('contact_all_frac') or 0:.2f} off {row.get('off_axis_rms_dps') or 0:6.1f} sat "
              f"{row.get('sat_share') or 0:.2f} {row.get('wall_s') or 0:.0f} s", flush=True)
    return 0


def load():
    return [json.loads(x) for x in ROWS.read_text().splitlines() if x.strip()] if ROWS.exists() else []


def summary():
    """Per (task, size, which, arm) the means over seeds; arm is 'position' or 'torque <wn>'."""
    groups = {}
    for r in load():
        arm = "position" if r["plant"] == "position" else f"torque {r['wn']:g}"
        groups.setdefault((r["task"], r["size"], r["which"], arm), []).append(r)
    out = {}
    for k, rs in groups.items():
        held = [r for r in rs if r.get("held")]
        foll = [max(0.0, 1.0 - r["track_err_rms_deg"] / r["track_ref_rms_deg"]) if r.get("held") else 0.0 for r in rs]
        out[k] = {"layout": rs[0]["layout"], "n": len(rs), "held": len(held),
                  "track": float(np.mean([r["track"] for r in rs])), "old": float(np.mean([r["old"] for r in rs])),
                  "followed": float(np.mean(foll)),
                  "three_pad": float(np.mean([r.get("contact_all_frac") or 0.0 for r in rs])),
                  "contact": {f: float(np.mean([(r.get("contact_frac") or {}).get(f, 0.0) for r in rs])) for f in FINGERS},
                  "off_axis": float(np.mean([r["off_axis_rms_dps"] for r in held])) if held else None,
                  "sat": float(np.mean([r.get("sat_share") or 0.0 for r in rs])),
                  "seeds": [r["track"] for r in sorted(rs, key=lambda r: r["seed"])]}
    return out


def cmd_report(_a):
    sm = summary()
    arms = sorted({k[3] for k in sm}, key=lambda a: (a != "position", a))
    print(f"{'task':6s} {'mm':>4s} {'which':10s} {'layout':>11s} " + "  ".join(f"{a:>28s}" for a in arms))
    for key in sorted({k[:3] for k in sm}, key=lambda k: (list(SIZES).index(k[0]), k[1], k[2])):
        cells = []
        for a in arms:
            v = sm.get(key + (a,))
            cells.append("-" if v is None else f"{v['track']:.2f} f{v['followed']:.2f} 3p{v['three_pad']:.2f} "
                         f"off{v['off_axis'] or 0:5.1f} s{v['sat']:.2f}")
        lay = next(v["layout"] for k, v in sm.items() if k[:3] == key)
        print(f"{key[0]:6s} {key[1]:4.0f} {key[2]:10s} {str(tuple(int(x) for x in lay)):>11s} " + "  ".join(f"{c:>28s}" for c in cells))
    for task in SIZES:
        for a in arms:
            vs = [v for k, v in sm.items() if k[0] == task and k[3] == a]
            if vs:
                print(f"{task:6s} {a:12s} mean track {np.mean([v['track'] for v in vs]):.3f}  followed "
                      f"{np.mean([v['followed'] for v in vs]):.3f}  three-pad {np.mean([v['three_pad'] for v in vs]):.3f}  "
                      f"off-axis {np.mean([v['off_axis'] for v in vs if v['off_axis'] is not None]):.1f} deg/s  sat "
                      f"{np.mean([v['sat'] for v in vs]):.2f}  held {sum(v['held'] for v in vs)}/{sum(v['n'] for v in vs)}")
    return 0


FILM_MEDIA = OUT / "media"
ARM_NAME = {"position": "position servos, kp 0.72 N m/rad", "servo": "torque, wn 25.6 rad/s", "fast": "torque, wn 63 rad/s"}


def cmd_film(a):
    """One task, size and layout on the three arms side by side, each at the seed whose tracking score is closest to
    the arm's mean over seeds 0-2 in the comparison rows (MUJOCO_GL=egl)."""
    import hom_contact_rig as HR
    import hom_films as HF
    rows = [r for r in load() if r["task"] == a.task and r["size"] == a.size and r["which"] == a.which]
    if not rows:
        raise SystemExit(f"no comparison rows for {a.task} {a.size:g} mm {a.which}")
    call, grasp = rows[0]["call"], rows[0]["grasp"]
    clips = []
    for arm, plant, wn in (("position", "position", None), ("servo", "torque", WN["servo"]), ("fast", "torque", WN["fast"])):
        rs = [r for r in rows if r["plant"] == plant and (wn is None or r["wn"] == wn)]
        mean = float(np.mean([r["track"] for r in rs]))
        seed = min(rs, key=lambda r: (abs(r["track"] - mean), r["seed"]))["seed"]
        row, frames = run(a.task, a.size, call, grasp, seed, plant, wn or WN["servo"], render=True)
        label = (f"{ARM_NAME[arm]}: tracking {row['track']:.2f} (mean {mean:.2f}), off-axis "
                 f"{row.get('off_axis_rms_dps') or 0:.0f} deg/s, seed {seed}")
        clips.append([HR.annotate(f, label, size=12) for f in frames])
    out = FILM_MEDIA / f"20261110-film_torque_{a.task}_d{a.size:g}_{a.which}.mp4"
    HF.compose(clips, out)
    return 0


def cmd_smoke(a):
    task, size = a.task, a.size
    p = json.loads(PICKS.read_text())[task][f"{size:g}"]
    for plant, wn in (("position", None), ("torque", WN["servo"]), ("torque", WN["fast"])):
        r = run(task, size, int(p["call"]), int(p["grasp"]), 0, plant, wn or WN["servo"])
        print(json.dumps(r, default=float), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--task", default="twist")
    s.add_argument("--size", type=float, default=63.0)
    c = sub.add_parser("compare")
    c.add_argument("--wn", type=float, nargs="+", default=[WN["servo"], WN["fast"]])
    c.add_argument("--part", type=int, default=0)
    c.add_argument("--parts", type=int, default=1)
    sub.add_parser("report")
    f = sub.add_parser("film")
    f.add_argument("--task", required=True)
    f.add_argument("--size", type=float, required=True)
    f.add_argument("--which", default="pick", choices=("pick", "generalist"))
    a = ap.parse_args()
    return {"smoke": cmd_smoke, "compare": cmd_compare, "report": cmd_report, "film": cmd_film}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
