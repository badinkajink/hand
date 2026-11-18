"""The night's g0 twists replayed with a give in the index's and middle's yaw chains (W3 of the identification queue,
2026-11-18): does the hand's give under load, which the rigid replay plant lacks, account for the replays' twist deficit?

At the night's grasps the hand's index and middle yaw readings ran about 4 deg past the angle where the model's pads met
the tracked sphere, while the real pads stayed on it (sim/20261118-grasp_pad_gap.py). The replay plant (rev_clutch2's
clutch plant: the load cannot backdrive a servo, forward efficiency 0.4) holds a stalled joint rigid. Here each of the
two yaws is a servo shaft and a link joined by a spring of stiffness K_C (N m/rad):
  - the link is the model's yaw joint; its actuator is the spring, K_C (shaft - link), with no frictionloss on the link;
  - the shaft follows the servo's target with the plant's lag (damping / kp, 13 ms) up to 10 rad/s. Driving against the
    spring it stops where the gear's forward output, 0.4 min(kp |target - shaft|, tau_max), meets the spring's torque,
    and the spring never drives it back (the clutch plant's rule, moved from the link to the shaft);
  - the shaft is the yaw's reading: it is what the replay compares with the hand's read-backs.
On the 7.5 V plant (the index's and middle's yaw kp 0.525 N m/rad) K_C 0.76 N m/rad gives the 4 deg at the night's
14-15 deg of target offset (0.4 x 0.525 x 0.253 rad / 0.76); K_C 1.5 gives 2 deg. Everything else is night_replay.py's (logs/20261117-g13_open_pose): the hand's recorded targets open
loop from its measured start on the 7.5 V plant, the table removed at the lift.

--backdrive replays the owner's alternative (11-18): no spring, and the load may push the index's and middle's yaw
servos back. When the load's torque about such a yaw exceeds the servo's and opposes it, the joint loses its gear
friction and yields as a plain position servo (kp 0.525 N m/rad, at most 0.18 N m); otherwise it is the clutch plant.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 SCALE_HAND=cad_v2 OMP_NUM_THREADS=1 \\
        .venv/bin/python sim/20261118-yaw_give_replay.py OUT.jsonl [--kc 0.76 1.5] [--contacts pads pads_stiff] \\
        [--tips off] [--trials STAMP ...] [--workers 6] [--backdrive]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT / "logs/20261117-g13_open_pose"))
import night_replay as NR  # noqa: E402

K_C = None                  # N m/rad, the yaw chain's spring; None: the rigid clutch plant
BACKDRIVE = False           # --backdrive: the load may push the index's and middle's yaw servos back
ETA = 0.4                   # the gear's forward efficiency (1 - c of the clutch plant)
TAU_S = 0.013               # s, the shaft's lag (the 7.5 V plant's damping / kp)
W_MAX = 10.0                # rad/s, the shaft's top speed (the SCS0009 at 7.5 V, about 0.1 s per 60 deg)
FL_LINK = 0.0               # N m, the link's own friction about the yaw axis
GIVE = ("index", "middle")
JUMP_DEG = 8.0


def give_class():
    base = NR.RPL.RC2.clutch_class()
    SR2Plant = base.__mro__[1]
    RC2 = NR.RPL.RC2

    class GivePlant(base):
        """The clutch plant with a spring between the index's and middle's yaw shafts and links (module docstring)."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.kc = K_C
            self.give = []                                    # (actuator, dof, qpos address, kp, tau_max)
            if self.kc:
                m = self.m
                for f in GIVE:
                    a_ = int(self.act[f][0])
                    self.give.append((a_, int(self.vadr[f][0]), int(self.qadr[f][0]),
                                      float(m.actuator_gainprm[a_, 0]), float(m.actuator_forcerange[a_, 1])))
                    m.actuator_gainprm[a_, 0] = self.kc
                    m.actuator_biasprm[a_, 1] = -self.kc
                    m.actuator_forcerange[a_] = (-5.0, 5.0)
            self.theta, self.T = {}, {}
            v9 = np.asarray(self.vadr9)
            self.bd = [int(np.flatnonzero(v9 == int(self.vadr[f][0]))[0]) for f in GIVE]

        def reset_to(self, q0, pos, quat):
            super().reset_to(q0, pos, quat)
            for a_, _dof, qa, _kp, _tau in self.give:
                self.theta[a_] = float(self.d.qpos[qa])
                self.T[a_] = float(self.d.ctrl[a_])
                self.d.ctrl[a_] = self.theta[a_]
            if self.give:
                self.mj.mj_forward(self.m, self.d)

        def set_targets(self, q_cmd):
            super().set_targets(q_cmd)
            for a_, *_ in self.give:
                self.T[a_] = float(self.d.ctrl[a_])
                self.d.ctrl[a_] = self.theta[a_]

        def shaft(self, h):
            lag = 1.0 - math.exp(-h / TAU_S)
            for a_, _dof, qa, kp, tau in self.give:
                th, T, q = self.theta[a_], self.T[a_], float(self.d.qpos[qa])
                s = 1.0 if T >= th else -1.0
                new = th + max(-W_MAX * h, min(W_MAX * h, (T - th) * lag))
                new = max(th, new) if s > 0 else min(th, new)          # the spring cannot backdrive the gear
                if s * (th - q) > 0.0:                                 # the spring opposes the motor: stop at balance
                    bal = (ETA * kp * T + self.kc * q) / (ETA * kp + self.kc)
                    if kp * abs(T - bal) > tau:
                        bal = q + s * ETA * tau / self.kc
                    new = min(new, max(th, bal)) if s > 0 else max(new, min(th, bal))
                self.theta[a_] = new
                self.d.ctrl[a_] = new

        def advance(self, t1: float):
            if not self.give and not BACKDRIVE:
                return super().advance(t1)
            import mujoco
            m, d = self.m, self.d
            h, nv = float(m.opt.timestep), m.nv
            gd = np.array([g[1] for g in self.give], int)
            while d.time < t1 - 1e-9:
                tau = d.actuator_force[self.acts].copy()
                n = int(d.nefc)
                qc = np.zeros(nv)
                if n:
                    J = np.asarray(d.efc_J)
                    J = J[:n] if J.ndim == 2 else J[:n * nv].reshape(n, nv)
                    rows = np.asarray(d.efc_type[:n]) >= int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
                    qc = J[rows].T @ np.asarray(d.efc_force[:n])[rows]
                ext = qc[self.vadr9] - d.qfrc_bias[self.vadr9]
                hold = (ext * tau < 0.0) & (np.abs(ext) > np.abs(tau))
                fl = np.maximum(self.c * np.abs(tau), RC2.BREAKAWAY)
                fl = np.where(hold, np.maximum(fl, np.abs(ext)), fl)
                if BACKDRIVE:                                          # these two yield to a load that beats them
                    fl[self.bd] = np.where(hold[self.bd], 0.0, fl[self.bd])
                    hold[self.bd] = False
                m.dof_frictionloss[self.vadr9] = fl
                if self.give:
                    m.dof_frictionloss[gd] = FL_LINK
                    self.shaft(h)
                self.n_locked += int(hold.sum())
                SR2Plant.advance(self, min(t1, d.time + h))

        def state(self):
            st = super().state()
            for f, (a_, *_r) in zip(GIVE, self.give):
                st["q"][f] = np.asarray(st["q"][f], float).copy()
                st["q"][f][0] = self.theta[a_]
            return st

    return GivePlant


def run_give(job):
    global K_C, BACKDRIVE
    K_C, BACKDRIVE = job["k_c"], bool(job.get("backdrive"))
    return NR.run(job)


def jumps(t, tw):
    t, tw = np.asarray(t, float), np.asarray(tw, float)
    ok = np.isfinite(tw) & (t >= 2.0)
    d = tw[ok][2:] - tw[ok][:-2]
    return (round(float(d.max()), 1), int((d > JUMP_DEG).sum())) if len(d) else (None, 0)


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--kc", type=float, nargs="+", default=[0.76, 1.5])
    ap.add_argument("--contacts", nargs="+", default=["pads", "pads_stiff"], choices=list(NR.RPL.SCENES))
    ap.add_argument("--tips", default="off", choices=["off", "on", "all"])
    ap.add_argument("--trials", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--backdrive", action="store_true", help="no spring; the load may push the two yaws back")
    a = ap.parse_args()
    NR.RPL.RG.gear_class = give_class
    NR.RPL.RC2.STIFF, NR.RPL.RC2.BREAKAWAY = True, 0.0
    NR.RPL.RG.RF.RL.rows_file()
    rep = json.loads((STUDY / "data/20261118-night_trials_replay.json").read_text())
    tips = {t["stamp"]: t["tips"] for t in rep["trials"]}
    tl = [t for t in NR.trials() if (a.tips == "all" or tips.get(t["stamp"]) == a.tips)
          and (not a.trials or t["stamp"] in a.trials)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for x in out.read_text().splitlines():
            r = json.loads(x) if x.strip() else None
            if r and "error" not in r:
                done.add((r["stamp"], r["contact"], r["k_c"], bool(r.get("backdrive"))))
    kcs = [None] if a.backdrive else a.kc
    jobs = [{**t, "contact": c, "volts": "7v5", "k_c": k, "backdrive": a.backdrive} for k in kcs for c in a.contacts
            for t in tl if (t["stamp"], c, k, a.backdrive) not in done]
    print(f"{len(jobs)} replays on {a.workers} workers ({len(done)} done)", flush=True)
    with open(out, "a") as fo, mp.get_context("fork").Pool(a.workers, initializer=NR.RPL.RG.init,
                                                           initargs=(str(NR.RPL.RG.RF.SCENE),), maxtasksperchild=4) as pool:
        for r in pool.imap_unordered(run_give, jobs):
            if "trace" in r:
                r["replay_jumps"] = jumps(r["trace"]["t"], r["trace"]["twist_replay"])
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            os.fsync(fo.fileno())
            if "error" in r or r.get("diverged"):
                print(r["stamp"], r["contact"], r["k_c"], "ERROR" if "error" in r else "DIVERGED", r.get("error", ""),
                      flush=True)
                continue
            jr = r["joint_rms_deg"]
            print(f"{r['stamp']} {r['contact']:10s} kc {r['k_c']}: track hand {r['hand']['track'].get('score')} replay "
                  f"{r['replay']['track'].get('score')} | twist {r['hand']['motion'].get('twist_end_deg')} / "
                  f"{r['replay']['motion'].get('twist_end_deg')} | rise {r['replay']['motion'].get('rise_max_mm')} | "
                  f"yaw bias grasp {jr['grasp']['bias']['index.aa']:+.1f}/{jr['grasp']['bias']['middle.aa']:+.1f} task "
                  f"{jr['task']['bias']['index.aa']:+.1f}/{jr['task']['bias']['middle.aa']:+.1f} | jumps "
                  f"{r['replay_jumps']} | {r['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
