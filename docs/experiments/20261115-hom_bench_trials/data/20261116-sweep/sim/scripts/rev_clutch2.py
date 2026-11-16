"""Servos that the load cannot backdrive (hand-4a, 2026-11-16 15:5x, for the corvus session). The corvus session's powered
backdrive test (15:55, a thumb-middle pinch of the sphere with the PIP commands lowered in 0.5 deg steps) found the gears
holding the full grip with no motor torque, and every yield a servo reversal: "the contact never backdrove a joint". Its
model: forward efficiency about 0.4 (frictionloss = c_f |motor torque|, c_f about 0.6) and no backdrive.

rev_clutch2 (the first version, a kinematic reset of every backdriven joint, broke the grasps): before every physics
step each finger joint's external torque is taken from the previous step, the contacts' (J^T f over the contact rows of
the constraint Jacobian) less the bias force (gravity). Where that load opposes the joint's servo torque and exceeds it,
so that it would drive the joint backward, the joint's frictionloss is raised to the load's magnitude and holds it;
elsewhere it is c_f |tau|, the forward gear friction.
Two checks per setting, as in rev_gear.py: the free-air sag steps (rev_joint_sag.SEQS, server plant) and the closed-loop
series (grasp 10 at 18.2/12/8/4 deg and the two gr grasps, seeds 0-2, stem scene, soft contact at friction 1.0).

    OMP_NUM_THREADS=1 .venv/bin/python rev_clutch2.py clutch2_soft.json --c 0.0 0.6 [--workers 8]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.home() / "Programs/hand"
D = ROOT / "logs/20261116-sim_real_reversal"
sys.path.insert(0, str(D))
import rev_gear as RG  # noqa: E402

STIFF = False                                      # rev_trace_clutch2 --stiff and rev_clutch2 --stiff set it


def clutch_class():
    import sr2_hom_replay as RP
    import mujoco

    class ClutchPlant(RP.SR2Plant):
        """SR2Plant whose finger joints hold against any load that would backdrive them (module docstring)."""

        def __init__(self, *a, c=0.6, **k):
            super().__init__(*a, **k)
            self.c = c
            self.n_locked = 0
            if STIFF:                              # MuJoCo's dry friction creeps under a steady load; stiffen it
                self.m.dof_solref[self.vadr9] = (2.0 * float(self.m.opt.timestep), 1.0)
                self.m.dof_solimp[self.vadr9, :2] = (0.99, 0.999)

        def advance(self, t1: float):
            m, d = self.m, self.d
            h, nv = float(m.opt.timestep), m.nv
            while d.time < t1 - 1e-9:
                tau = d.actuator_force[self.acts].copy()
                n = int(d.nefc)
                qc = np.zeros(nv)
                if n:
                    J = np.asarray(d.efc_J)
                    J = J[:n] if J.ndim == 2 else J[:n * nv].reshape(n, nv)
                    rows = np.asarray(d.efc_type[:n]) >= int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
                    qc = J[rows].T @ np.asarray(d.efc_force[:n])[rows]
                ext = qc[self.vadr9] - d.qfrc_bias[self.vadr9]           # contacts and gravity on each joint
                hold = (ext * tau < 0.0) & (np.abs(ext) > np.abs(tau))  # the load would drive it against its servo
                fl = self.c * np.abs(tau)
                m.dof_frictionloss[self.vadr9] = np.where(hold, np.maximum(fl, np.abs(ext)), fl)
                self.n_locked += int(hold.sum())
                super().advance(min(t1, d.time + h))
    return ClutchPlant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--c", type=float, nargs="+", default=[0.0, 0.6])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stiff", action="store_true", help="joint friction with solref 2 ms and solimp 0.99-0.999")
    a = ap.parse_args()
    global STIFF
    STIFF = a.stiff
    RG.gear_class = clutch_class                        # rev_gear's checks, run on the clutch plant
    out = {"free_air": {}, "series": []}
    for c in a.c:
        fa = RG.free_air(c)
        out["free_air"][str(c)] = fa
        for r in fa:
            print(f"free air c {c}: {r['finger']:6s} {r['joint']:3s} {'+' if r['sign'] > 0 else '-'}: sim "
                  + "/".join(f"{v:.2f}" for v in r["sim"]) + (" | hand " + "/".join(f"{v:.1f}" for v in r["hand"])
                                                              if r["hand"] else ""), flush=True)
    RG.RF.RL.rows_file()
    import multiprocessing as mp
    jobs = [(n, s, cl, k, seed, c) for c in a.c for (n, s, cl, k) in RG.RF.GRASPS for seed in (0, 1, 2)]
    with mp.get_context("fork").Pool(a.workers, initializer=RG.init, initargs=(str(RG.RF.SCENE),)) as pool:
        rows = pool.map(RG.run, jobs)
    out["series"] = rows
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for c in a.c:
        print(f"series c {c}, load-holding friction", flush=True)
        for n, *_ in RG.RF.GRASPS:
            rs = [r for r in rows if r["c"] == c and r["grasp"] == n and "error" not in r]
            bad = sum(1 for r in rows if r["c"] == c and r["grasp"] == n and "error" in r)
            if not rs:
                print(f"   {n:16s} all failed: {[r.get('error') for r in rows if r['grasp'] == n][:1]}", flush=True)
                continue
            mm = lambda key: np.mean([r[key] for r in rs])                              # noqa: E731
            print(f"   {n:16s} held {sum(r['held'] for r in rs)}/{len(rs) + bad} track {np.mean([r['track'] or 0 for r in rs]):.2f}"
                  f" twist {mm('twist_end'):5.1f} rise {mm('rise_max'):5.1f} | gap middle.aa {mm('gap_middle_aa'):+6.1f} "
                  f"index.aa {mm('gap_index_aa'):+5.1f} thumb.fe1 {mm('gap_thumb_fe1'):+5.1f} | thumb PIP med "
                  f"{mm('thumb_pip_median'):+6.1f}" + (f" | {bad} failed" if bad else ""), flush=True)


if __name__ == "__main__":
    main()
