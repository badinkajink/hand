"""Which parts of each finger touch the sphere in the replays of the night's g0 twists (W3 of the identification queue,
2026-11-18): the pads or the fingers' rigid parts, and with how much normal force.

On the hand the middle's board read nothing in 18 of the 20 scored trials, while a press of its pad face on the S3
plate reads 0.36 of the force (sim/20261118-s3_board_vs_hexe.py): there its pad face took no load, as a contact on the
tip's side or the DIP link gives. Here the replay of sim/20261118-s3_plant_replay.py runs with the given plant,
friction, give and contact model, and at the end of every 20 ms period each contact between a finger and the object is
sorted by the finger's geom: a pad (the sphere pads of the scene), or any other geom of the finger. Per finger and
window (grasp 0.6-2.0 s, task from 2.0 s): the share of periods with a pad contact and with another contact, and the
median summed normal force of each.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 SCALE_HAND=cad_v2 OMP_NUM_THREADS=1 \\
        .venv/bin/python sim/20261118-replay_contacts.py [--contact pads_stiff] [--plant s3tc] [--mu 0.4] [--kc 2.0] \\
        [--trials STAMP ...] [--out data/20261118-replay_contacts.json]
"""
import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
_spec = importlib.util.spec_from_file_location("s3_plant_replay", HERE / "20261118-s3_plant_replay.py")
S3R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S3R)
NR = S3R.NR
FINGERS = ("thumb", "index", "middle")
WINDOWS = {"grasp": (0.6, 2.0), "task": (2.0, 99.0)}


def census_class(base):
    class CensusPlant(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            import mujoco
            m = self.m
            self.census = []
            self.obj_geoms = set(np.flatnonzero(m.geom_bodyid == self.tb).tolist())
            pads = set(int(g) for g in self.pads)
            self.kind = {}
            for g in range(m.ngeom):
                b = int(m.geom_bodyid[g])
                name = ""
                while b > 0 and not name:
                    bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                    hit = next((f for f in FINGERS if bn.startswith(f)), None)
                    name = hit or ""
                    b = int(m.body_parentid[b])
                if name:
                    self.kind[g] = (name, "pad" if g in pads else "rigid")

        def advance(self, t1: float):
            super().advance(t1)
            import mujoco
            m, d = self.m, self.d
            row = {f: {"pad": 0.0, "rigid": 0.0} for f in FINGERS}
            f6 = np.zeros(6)
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                other = g2 if g1 in self.obj_geoms else (g1 if g2 in self.obj_geoms else None)
                if other is None or other not in self.kind:
                    continue
                mujoco.mj_contactForce(m, d, i, f6)
                f, part = self.kind[other]
                row[f][part] += abs(float(f6[0]))
            self.census.append((float(d.time), row))

    return CensusPlant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact", default="pads_stiff", choices=list(NR.RPL.SCENES))
    ap.add_argument("--plant", default="s3tc", choices=["s3", "s3tc"])
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--kc", type=float, default=2.0)
    ap.add_argument("--trials", nargs="*", default=None)
    ap.add_argument("--out", default=str(STUDY / "data/20261118-replay_contacts.json"))
    a = ap.parse_args()
    S3R.PLANT, S3R.MU = a.plant, a.mu
    S3R.YG.K_C, S3R.YG.ETA, S3R.YG.BACKDRIVE = a.kc or None, 1.0, False
    S3R.patch_kinematics()
    NR.RPL.RG.gear_class = lambda: census_class(S3R.YG.give_class(S3R.s3_class()))
    NR.RPL.RC2.STIFF, NR.RPL.RC2.BREAKAWAY = True, 0.0
    NR.RPL.RG.RF.RL.rows_file()
    NR.RPL.RG.init(str(NR.RPL.RG.RF.SCENE))
    rep = json.loads((STUDY / "data/20261118-night_trials_replay.json").read_text())
    tips = {t["stamp"]: t["tips"] for t in rep["trials"]}
    tl = [t for t in NR.trials() if tips.get(t["stamp"]) == "off" and (not a.trials or t["stamp"] in a.trials)]
    box, make = {}, NR.make_plant

    def make_keep(contact, volts):
        p, setup = make(contact, volts)
        box["p"] = p
        return p, setup

    NR.make_plant = make_keep
    per = []
    for t in tl:
        job = {**t, "contact": a.contact, "volts": "7v5", "plant": a.plant, "mu": a.mu, "k_c": a.kc or None}
        out = NR.run1(job)
        cen = box["p"].census
        res = {"stamp": t["stamp"], "twist_end_deg": out["replay"]["motion"].get("twist_end_deg"),
               "hand_twist_end_deg": out["hand"]["motion"].get("twist_end_deg"), "fingers": {}}
        for f in FINGERS:
            res["fingers"][f] = {}
            for w, (t0, t1) in WINDOWS.items():
                rows = [r[f] for tt, r in cen if t0 <= tt < t1]
                pad = np.array([r["pad"] for r in rows])
                rig = np.array([r["rigid"] for r in rows])
                res["fingers"][f][w] = {
                    "pad_share": round(float((pad > 0.01).mean()), 3), "rigid_share": round(float((rig > 0.01).mean()), 3),
                    "pad_N_median": round(float(np.median(pad[pad > 0.01])), 3) if (pad > 0.01).any() else 0.0,
                    "rigid_N_median": round(float(np.median(rig[rig > 0.01])), 3) if (rig > 0.01).any() else 0.0}
        per.append(res)
        print(t["stamp"], "twist", res["twist_end_deg"], "hand", res["hand_twist_end_deg"],
              {f: (res["fingers"][f]["task"]["pad_share"], res["fingers"][f]["task"]["rigid_share"],
                   res["fingers"][f]["task"]["pad_N_median"], res["fingers"][f]["task"]["rigid_N_median"]) for f in FINGERS},
              flush=True)
    summ = {f: {w: {k: round(float(np.median([r["fingers"][f][w][k] for r in per])), 3)
                    for k in ("pad_share", "rigid_share", "pad_N_median", "rigid_N_median")} for w in WINDOWS}
            for f in FINGERS}
    print("medians", json.dumps(summ))
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "),
                                       "script": "docs/experiments/20261117-sr2_system_identification/sim/"
                                                 "20261118-replay_contacts.py",
                                       "plant": {"plant": a.plant, "mu": a.mu, "kc": a.kc, "contact": a.contact},
                                       "medians": summ, "trials": per}, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
