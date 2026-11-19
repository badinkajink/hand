"""The night's g0 twists replayed on a plant set to S3's presses (W3 of the identification queue, 2026-11-18): do servos
as strong as S3 measured them close the replays' twist deficit?

S3 on the owner's 80/20 post (data/s3/; sim/20261118-s3_yaw_head_on_fits.py) measured the static force at the pad
against the servo's error (target - reading):
  - yaws (eleven head-on presses of the index and middle): 0.82 N at 5 deg of error and 1.5 N at 10 deg, largest
    1.64 N (medians), at 88 mm: 0.83 N m/rad through zero, up to 0.144 N m;
  - the thumb's yaw (three presses): 1.16 N at 5 deg, largest 2.18 N: 1.17 N m/rad, up to 0.19 N m;
  - the index's MCP (one press): 1.73 N at 4 deg and 2.57 N at 6.2 deg, then flat to the overload flag, at 67 mm:
    1.6 N m/rad, up to 0.17 N m; the PIPs and the thumb's MCP and PIP take the same values;
  - no backdrive: on every step back the force held until the target came within about 1 deg of the reading.
The replay plant of data/20261118-night_trials_replay.json (7v5: kp 0.825, the index's and middle's yaws 0.525,
tau_max 0.27 N m, forward friction 0.6 |servo torque|) pushes 0.4 kp per rad, at most 0.108 N m. Here the clutch plant
keeps its hold rule (a load that would drive a joint backward is held) and loses its forward friction (c = 0); each
joint's servo is kp_j (target - q) up to tau_j. Two dampings: "s3" sets the free-air top speed, tau_j / damping, to the
7v5 plant's 9.8 rad/s (the SCS0009 at 7.5 V turns about 10 rad/s); "s3tc" keeps the 7v5y plant's free-air time constant,
damping / (0.4 kp) = 33 ms. The scenes' friction 1.0 (the grip tape) against S3's 0.5 for the pad on the tape.
Everything else is night_replay.py's (logs/20261117-g13_open_pose): the hand's recorded targets, open loop from its
measured start, the table removed at the lift.

--kc adds the give that S3's head-on yaw presses measured past the index's and middle's yaw potentiometers (median
2.04 N m/rad over eleven presses, sim/20261118-s3_yaw_head_on_fits.py): sim/20261118-yaw_give_replay.py's spring between
each yaw's shaft (its reading) and its link, on this plant, with the gear's forward efficiency 1.

--zshift MM moves the object's start up along z_P by MM (S1(b)'s tags sat 3.86 mm below the hand's model, shared by
the three fingers), and --zeros takes S1(b)'s shared-height model ("dz" in data/s1/20261118-s1b_tag_fit.json): its
joint zeros are added to every joint, so the hand's readings and targets become the model's angles plus the zeros and
the replayed angles are compared after taking them off again, and its mount offsets (mx, my) move each finger's mount
body in P.

--stick MU_S MU_K gives the friction two levels, as S4(b) measured the pad on the tape (sim/20261118-s4b_drag_friction.py:
sliding 0.44-0.49 at 0.7-1.0 N and 0.35 at 2.5 N; after a 20 s rest the slide began 1.2-1.4 times higher). Every geom's
friction is MU_S while the object turns slower than 0.2 rad/s, and MU_K from then until it turns slower than 0.05 rad/s
again; MuJoCo's friction otherwise has one coefficient for both.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 SCALE_HAND=cad_v2 OMP_NUM_THREADS=1 \\
        .venv/bin/python sim/20261118-s3_plant_replay.py OUT.jsonl [--plants s3 s3tc] [--mu 1.0 0.5] \\
        [--contacts pads pads_stiff point] [--kc 2.0] [--stick 0.55 0.40] [--zshift 3.86] [--zeros] [--tips off] \\
        [--trials STAMP ...] [--workers 6]
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT / "logs/20261117-g13_open_pose"))
import night_replay as NR  # noqa: E402

_spec = importlib.util.spec_from_file_location("yaw_give_replay", HERE / "20261118-yaw_give_replay.py")
YG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(YG)

JOINTS = {"yaw": 0, "mcp": 1, "pip": 2}               # the order of a finger's actuators and dofs in the plant
S3 = {("index", "yaw"): (0.83, 0.144), ("middle", "yaw"): (0.83, 0.144), ("thumb", "yaw"): (1.17, 0.19)}
S3_FLEX = (1.6, 0.17)                                # every MCP and PIP: kp (N m/rad), tau_max (N m)
W_TOP = 9.8                                          # rad/s, the 7v5 plant's free-air top speed (0.4 x 0.27 / 0.011)
T_FREE = 0.033                                       # s, the 7v5y plant's free-air time constant (0.011 / (0.4 x 0.825))
PLANT = None                                         # "s3" or "s3tc", set per job
MU = 1.0
STICK = None                                         # (friction at rest, friction while sliding), set per job
ZSHIFT = 0.0                                         # mm, the object's start moved up along z_P, set per job
ZEROS = None                                         # {finger: [yaw, mcp, pip] deg}, added to the readings, set per job
S1B = STUDY / "data/s1/20261118-s1b_tag_fit.json"
W_ON, W_OFF = 0.2, 0.05                              # rad/s: the object's turn rate that starts and ends a slide
JUMP_DEG = 8.0


def s3_class():
    base = NR.RPL.RC2.clutch_class()

    class S3Plant(base):
        """The clutch plant without forward friction and with each joint's servo as S3 measured it (module docstring)."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            m = self.m
            self.c = 0.0
            self.s3 = {}
            for f in ("thumb", "index", "middle"):
                for j, i in JOINTS.items():
                    kp, tau = S3.get((f, j), S3_FLEX)
                    damp = tau / W_TOP if PLANT == "s3" else T_FREE * kp
                    a_, v = int(self.act[f][i]), int(self.vadr[f][i])
                    m.actuator_gainprm[a_, 0] = kp
                    m.actuator_biasprm[a_, 1] = -kp
                    m.actuator_forcerange[a_] = (-tau, tau)
                    m.dof_damping[v] = damp
                    self.s3[f"{f}.{j}"] = (kp, tau, round(damp, 4))
            m.geom_friction[:, 0] = MU
            if ZEROS:
                pf = json.loads(S1B.read_text())["shared"]["dz"]["per_finger"]
                for f in ("thumb", "index", "middle"):
                    m.body_pos[m.body(f"{f}_mount").id][:2] += np.array([pf[f]["mx"], pf[f]["my"]]) / 1e3

    return S3Plant


def stick_class(base):
    class StickPlant(base):
        """Friction MU_S while the object rests in the grasp and MU_K while it slides (module docstring)."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.sliding = False
            self.n_slides = 0

        def advance(self, t1: float):
            if not STICK:
                return super().advance(t1)
            m, d = self.m, self.d
            h = float(m.opt.timestep)
            while d.time < t1 - 1e-9:
                w = float(np.linalg.norm(d.qvel[self.va + 3:self.va + 6]))
                was = self.sliding
                self.sliding = w > (W_OFF if self.sliding else W_ON)
                self.n_slides += int(self.sliding and not was)
                m.geom_friction[:, 0] = STICK[1] if self.sliding else STICK[0]
                super().advance(min(t1, d.time + h))

    return StickPlant


def s1b_zeros():
    pf = json.loads(S1B.read_text())["shared"]["dz"]["per_finger"]
    return {f: [pf[f]["yaw"], pf[f]["mcp"], pf[f]["pip"]] for f in pf}


def patch_kinematics():
    """Wrap the start pose and the servo-sim mapping of night_replay once per worker for ZSHIFT and ZEROS."""
    from sr2hand.control import hom_bench as HB
    if getattr(HB, "_s3_patched", False):
        return
    s2s, s2v, sp = HB.sim_from_servo, HB.servo_from_sim, NR.start_pose

    def sim_from_servo(joints):
        q = s2s(joints)
        if ZEROS:
            q = {f: np.asarray(q[f], float) + np.radians(ZEROS[f]) for f in q}
        return q

    def servo_from_sim(q):
        if ZEROS:
            q = {f: np.asarray(q[f], float) - np.radians(ZEROS[f]) for f in q}
        return s2v(q)

    def start_pose(setup, first):
        pos, quat, yaw, tilt = sp(setup, first)
        pos = np.asarray(pos, float).copy()
        pos[2] += ZSHIFT / 1e3
        return pos, quat, yaw, tilt

    HB.sim_from_servo, HB.servo_from_sim, NR.start_pose = sim_from_servo, servo_from_sim, start_pose
    HB._s3_patched = True


def run_s3(job):
    global PLANT, MU, STICK, ZSHIFT, ZEROS
    PLANT, MU = job["plant"], job["mu"]
    STICK = tuple(job["stick"]) if job.get("stick") else None
    ZSHIFT = float(job.get("zshift") or 0.0)
    ZEROS = s1b_zeros() if job.get("zeros") else None
    patch_kinematics()
    YG.K_C, YG.ETA, YG.BACKDRIVE = job["k_c"], 1.0, False
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
    ap.add_argument("--plants", nargs="+", default=["s3", "s3tc"], choices=["s3", "s3tc"])
    ap.add_argument("--mu", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--contacts", nargs="+", default=["pads", "pads_stiff", "point"], choices=list(NR.RPL.SCENES))
    ap.add_argument("--kc", type=float, nargs="+", default=[None], help="the yaw give's stiffness (N m/rad)")
    ap.add_argument("--stick", type=float, nargs=2, default=None, help="friction at rest and while sliding")
    ap.add_argument("--zshift", type=float, nargs="+", default=[0.0], help="mm, the object's start moved up along z_P")
    ap.add_argument("--zeros", action="store_true", help="S1(b)'s joint zeros (shared-height model)")
    ap.add_argument("--tips", default="off", choices=["off", "on", "all"])
    ap.add_argument("--trials", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    NR.RPL.RG.gear_class = lambda: stick_class(YG.give_class(s3_class()))
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
                done.add((r["stamp"], r["contact"], r["plant"], r["mu"], r["k_c"], tuple(r.get("stick") or ()),
                          float(r.get("zshift") or 0.0), bool(r.get("zeros"))))
    st = tuple(a.stick or ())
    mus = [a.stick[0]] if a.stick else a.mu
    lab = f" stick{a.stick[0]:g}/{a.stick[1]:g}" if a.stick else ""
    jobs = [{**t, "contact": c, "volts": "7v5", "plant": p, "mu": mu, "k_c": k, "stick": list(st) or None,
             "zshift": zs, "zeros": a.zeros,
             "variant": (f"{p} mu{mu:g}" if not a.stick else f"{p}") + (f" kc{k:g}" if k else "") + lab
             + (f" z{zs:+g}" if zs else "") + (" zeros" if a.zeros else "")}
            for k in a.kc for p in a.plants for mu in mus for zs in a.zshift for c in a.contacts for t in tl
            if (t["stamp"], c, p, mu, k, st, zs, a.zeros) not in done]
    print(f"{len(jobs)} replays on {a.workers} workers ({len(done)} done)", flush=True)
    with open(out, "a") as fo, mp.get_context("fork").Pool(a.workers, initializer=NR.RPL.RG.init,
                                                           initargs=(str(NR.RPL.RG.RF.SCENE),), maxtasksperchild=4) as pool:
        for r in pool.imap_unordered(run_s3, jobs):
            if "trace" in r:
                r["replay_jumps"] = jumps(r["trace"]["t"], r["trace"]["twist_replay"])
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            os.fsync(fo.fileno())
            if "error" in r or r.get("diverged"):
                print(r["stamp"], r["contact"], r["variant"], "ERROR" if "error" in r else "DIVERGED", r.get("error", ""),
                      flush=True)
                continue
            jr = r["joint_rms_deg"]
            print(f"{r['stamp']} {r['contact']:10s} {r['variant']:10s}: track hand {r['hand']['track'].get('score')} "
                  f"replay {r['replay']['track'].get('score')} | twist {r['hand']['motion'].get('twist_end_deg')} / "
                  f"{r['replay']['motion'].get('twist_end_deg')} | rise {r['replay']['motion'].get('rise_max_mm')} | "
                  f"joints approach {jr['approach']['all']} | yaw bias task {jr['task']['bias']['index.aa']:+.1f}/"
                  f"{jr['task']['bias']['middle.aa']:+.1f} | jumps {r['replay_jumps']} | {r['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
