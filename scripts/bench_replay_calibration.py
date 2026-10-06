#!/usr/bin/env python3
"""The finger servo plant fitted to the bench's own runs: every tracked bench run replayed in MuJoCo.

The archive (docs/experiments/20260902-cb1-log-archive/logs) holds 177 complete reorientation runs of the eight
deployed hands D1-D8 with AprilTag tracking: 11-43 per hand, one plan per hand. Each run logs the command frames sent
to the servos (50 Hz, sim joint degrees), three servo readbacks (before the turn, during it, at the settled hold
1.5-2 s after the last command) and the tool's tilt and height at 20-25 Hz.

Each run is replayed on the hand's bench scene (reorient_backends.build_scene: palm welded at the plan's pose, the
tool resting on the shortened post, the printed TPU tip with 2.7 mm fillets as 1 mm sphere pads, 1 ms step,
elliptic cone, impratio 100, the scene's friction 2.4): the grip is held 0.8 s, the logged command frames are applied
at their logged times, and the scene runs on to the run's last tracking sample. The plant is set on the compiled
model, so one scene per hand serves every candidate:

  kp     position gain, N m/rad          (actuator gainprm/biasprm)
  tau    finger time constant, s          joint damping = tau * kp, actuator kv 0
  fr     torque limit, N m                (actuator forcerange)
  fl     joint friction, N m              (dof frictionloss)

Compared at the bench's own sample times:

  joint error   sim minus bench achieved angle at the three servo readbacks, nine finger joints, deg
  turn error    sim minus bench turn, deg; turn = tilt at the first command minus tilt now, tilt = angle of the
                tool axis from vertical; RMS over the tag samples from the first command to the end
  drop          sim: tool 20 mm below its grip height or fewer than 2 fingers touching at the end;
                bench: the tracker's `dropped`

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/bench_replay_calibration.py runs
    $PY scripts/bench_replay_calibration.py fit --grid coarse --per-hand 2 --worker 0/4
    $PY scripts/bench_replay_calibration.py summary
Rows: docs/experiments/20261006-servo_recalibration/replays.jsonl, one fsynced line per run x plant.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reorient_backends as RB  # noqa: E402
from fit_servo_compliance import achieved_sim_deg, read_run  # noqa: E402

LOGS = ROOT / "docs/experiments/20260902-cb1-log-archive/logs"
OUT_DIR = ROOT / "docs/experiments/20261006-servo_recalibration"
ROWS = OUT_DIR / "replays.jsonl"
RUNS = OUT_DIR / "runs.json"
SCENES = ROOT / "logs/20261006-servo_recalibration/scenes"
DESIGN_HAND = {"sv1_w6689": "D1", "sv1_w2360": "D2", "sv1_u1364": "D3", "g12": "D4", "sv1_u0060": "D5",
               "sv1_u0308": "D6", "rv05_manual": "D7", "sv1_w0099": "D8"}
FINGERS, JOINTS = RB.FINGERS, RB.JOINTS
T_GRIP = 0.8
BASE_PLANT = "kp1_kv0_fr10_fl0_dp0.1"       # only builds the scene; every replay sets its own plant
GRIDS = {
    "coarse": dict(kp=[0.25, 0.5, 1, 2, 4, 8, 30], tau=[0.02, 0.06, 0.2, 0.6], fr=[0.2, 0.35, 1.0],
                   fl=[0.0, 0.005, 0.02]),
    "fine": dict(kp=[2, 4, 6, 10], tau=[0.01, 0.02, 0.04], fr=[0.5, 1.0, 2.0], fl=[0.0]),
}


# ---------------------------------------------------------------------------------- the bench runs

def load_runs() -> list[dict]:
    """The tracked, complete runs of D1-D8: commands, servo readbacks and tag samples on the CB1's clock."""
    out = []
    for p in sorted(LOGS.glob("*.jsonl")):
        s = p.with_name(p.stem + "_SUMMARY.json")
        if not s.exists():
            continue
        summ = json.loads(s.read_text())
        hand = DESIGN_HAND.get(summ.get("design"))
        ot = summ.get("object_track") or {}
        if hand is None or summ.get("status") != "complete" or not ot.get("frames"):
            continue
        run = read_run(p)
        if run is None:
            continue
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        cmds = [(r["monotonic_s"], {f"{f}_{j}": r["sim_joint_deg"][f][j] for f in FINGERS for j in JOINTS})
                for r in run["commands"]]
        t0 = cmds[0][0]
        servo = []
        for r in run["telemetry"]:
            q = {f"{f}_{j}": achieved_sim_deg(r["data"]["servos"], f, j) for f in FINGERS for j in JOINTS}
            servo.append({"t": r["monotonic_s"] - t0, "phase": r.get("phase"), "q_deg": q})
        tags = []
        for r in rows:
            o = r.get("object") if r.get("kind") == "object" else None
            if o and o.get("seen"):
                tags.append({"t": r["monotonic_s"] - t0, "deg": o["deg"], "cos": o["cos"], "z_mm": o["z_bench_mm"],
                             "x_mm": o["x_bench_mm"], "y_mm": o["y_bench_mm"]})
        if len(tags) < 5:
            continue
        out.append({"run": p.stem, "hand": hand, "design": summ["design"],
                    "speed_ratio": summ.get("settings", {}).get("speed_ratio"),
                    "cmds": [(t - t0, q) for t, q in cmds], "servo": servo, "tags": tags,
                    "bench": {k: ot.get(k) for k in ("deg_turned", "dropped", "z_drop_mm", "slip_mm", "cos_start",
                                                     "cos_final")}})
    return out


def runs_subset(runs, per_hand):
    """`per_hand` runs per hand at evenly spaced quantiles of the bench's turn (one run: the median)."""
    out = []
    for h in sorted({r["hand"] for r in runs}):
        rs = sorted([r for r in runs if r["hand"] == h], key=lambda r: r["bench"]["deg_turned"] or 0.0)
        if per_hand >= len(rs):
            out += rs
            continue
        q = (np.arange(per_hand) + 0.5) / per_hand
        out += [rs[int(round(x * (len(rs) - 1)))] for x in q]
    return out


# ---------------------------------------------------------------------------------- the plant on a model

def plants_of(grid: str) -> list[dict]:
    g = GRIDS[grid]
    return [dict(kp=kp, tau=tau, fr=fr, fl=fl) for kp, tau, fr, fl in itertools.product(g["kp"], g["tau"], g["fr"],
                                                                                         g["fl"])]


def plant_name(p: dict) -> str:
    return f"kp{p['kp']:g}_tau{p['tau']:g}_fr{p['fr']:g}_fl{p['fl']:g}"


def set_plant(m, p: dict):
    for f in FINGERS:
        for j in JOINTS:
            a = m.actuator(f"a_{f}_{j}").id
            m.actuator_gainprm[a, 0] = p["kp"]
            m.actuator_biasprm[a, 1] = -p["kp"]
            m.actuator_biasprm[a, 2] = 0.0
            m.actuator_forcelimited[a] = 1
            m.actuator_forcerange[a] = (-p["fr"], p["fr"])
            dof = m.jnt_dofadr[m.joint(f"{f}_{j}").id]
            m.dof_damping[dof] = p["tau"] * p["kp"]
            m.dof_frictionloss[dof] = p["fl"]


# ---------------------------------------------------------------------------------- one replay

class HandScene:
    def __init__(self, hand: str, tip: str = "tpu2.7", model: str = "pads", mu="scene", ir: float = 100.0):
        import mujoco
        self.mj = mujoco
        self.hand = hand
        self.plan, _ = RB.load_plan(hand)
        path, self.meta = RB.build_scene(hand, tip, model, BASE_PLANT, "bed", ir, mu, out_dir=SCENES)
        self.m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.tb = m.body(RB.OBJ).id
        self.qa = m.jnt_qposadr[m.body_jntadr[self.tb]]
        self.act = {f"{f}_{j}": m.actuator(f"a_{f}_{j}").id for f in FINGERS for j in JOINTS}
        self.qadr = {n: m.jnt_qposadr[m.joint(n).id] for n in self.act}
        self.tool_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == self.tb}
        self.finger_of = {}
        for i in range(m.ngeom):
            bn = m.body(m.geom_bodyid[i]).name
            for f in FINGERS:
                if bn.startswith(f + "_"):
                    self.finger_of[i] = f
        poses = {p["name"]: p["joints"] for p in self.plan["poses"]}
        self.grip = {f"{f}_{j}": math.radians(poses["grip"][f][j]) for f in FINGERS for j in JOINTS}

    def fingers_touching(self):
        d = self.d
        seen = set()
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if (g1 in self.tool_geoms) == (g2 in self.tool_geoms) or c.dist > 0:
                continue
            f = self.finger_of.get(g2 if g1 in self.tool_geoms else g1)
            if f:
                seen.add(f)
        return len(seen)

    def replay(self, run: dict, plant: dict, trace_dt: float = 0.01, dxy=(0.0, 0.0)):
        mj, m, d = self.mj, self.m, self.d
        set_plant(m, plant)
        mj.mj_resetData(m, d)
        tool7 = np.asarray(self.meta["tool7"], float)
        d.qpos[self.qa:self.qa + 7] = tool7
        d.qpos[self.qa] += dxy[0]
        d.qpos[self.qa + 1] += dxy[1]
        for n, a in self.act.items():
            d.qpos[self.qadr[n]] = self.meta["q0"][n]
            d.ctrl[a] = self.grip[n]
        mj.mj_forward(m, d)
        cmds = run["cmds"]
        t_end = T_GRIP + max(run["tags"][-1]["t"], run["servo"][-1]["t"], cmds[-1][0]) + 0.05
        dt = m.opt.timestep
        k_trace = max(1, int(round(trace_dt / dt)))
        T, Q, C, Z, N = [], [], [], [], []
        ci, step, w0 = 0, 0, time.perf_counter()
        while d.time < t_end:
            while ci < len(cmds) and T_GRIP + cmds[ci][0] <= d.time + 1e-9:
                for n, v in cmds[ci][1].items():
                    d.ctrl[self.act[n]] = math.radians(v)
                ci += 1
            mj.mj_step(m, d)
            step += 1
            if step % k_trace == 0:
                T.append(d.time)
                Q.append([d.qpos[self.qadr[n]] for n in self.act])
                C.append(float(d.xmat[self.tb][8]))
                Z.append(float(d.xpos[self.tb][2]))
                N.append(self.fingers_touching())
                if not np.isfinite(Z[-1]) or abs(Z[-1]) > 2.0:
                    break
        wall = time.perf_counter() - w0
        return dict(t=np.array(T), q=np.degrees(np.array(Q)), cos=np.array(C), z=np.array(Z), n=np.array(N),
                    wall_s=wall, steps=step, names=list(self.act))


def compare(run: dict, sim: dict) -> dict:
    """Sim against bench at the bench's sample times."""
    t = sim["t"]

    def at(arr, tb):
        return arr[int(np.clip(np.searchsorted(t, T_GRIP + tb), 0, len(t) - 1))]

    names = sim["names"]
    jerr = {}
    for s in run["servo"]:
        qs = at(sim["q"], max(s["t"], 0.0))
        for k, n in enumerate(names):
            b = s["q_deg"].get(n)
            if b is not None:
                jerr.setdefault(s["phase"], {})[n] = round(float(qs[k] - b), 2)
    tags = [g for g in run["tags"]]
    ref = [g for g in tags if g["t"] <= 0.0]
    ref = ref[-1] if ref else tags[0]
    i0 = int(np.clip(np.searchsorted(t, T_GRIP), 0, len(t) - 1))
    tilt0 = math.degrees(math.acos(max(-1.0, min(1.0, sim["cos"][i0]))))
    tb_, ts_, zb_, zs_ = [], [], [], []
    for g in tags:
        if g["t"] < 0.0:
            continue
        c = at(sim["cos"], g["t"])
        ts_.append(tilt0 - math.degrees(math.acos(max(-1.0, min(1.0, c)))))
        tb_.append(ref["deg"] - g["deg"])
        zs_.append(1e3 * (at(sim["z"], g["t"]) - sim["z"][i0]))
        zb_.append(g["z_mm"] - ref["z_mm"])
    ts_, tb_ = np.array(ts_), np.array(tb_)
    z_grip = sim["z"][i0]
    dropped = bool(sim["z"][-1] < z_grip - 0.020 or sim["n"][-1] < 2)
    allj = [abs(v) for ph in jerr.values() for v in ph.values()]
    return {"turn_sim_end": round(float(ts_[-1]), 2) if len(ts_) else None,
            "turn_bench_end": round(float(tb_[-1]), 2) if len(tb_) else None,
            "turn_rmse": round(float(np.sqrt(np.mean((ts_ - tb_) ** 2))), 2) if len(ts_) else None,
            "dz_sim_end_mm": round(float(zs_[-1]), 2) if zs_ else None,
            "dz_bench_end_mm": round(float(zb_[-1]), 2) if zb_ else None,
            "joint_mae": round(float(np.mean(allj)), 3) if allj else None,
            "joint_err": jerr, "dropped_sim": dropped, "dropped_bench": bool(run["bench"].get("dropped")),
            "fingers_end": int(sim["n"][-1]), "z_grip_mm": round(1e3 * float(z_grip), 2),
            "turn_trace_sim": [round(float(x), 2) for x in ts_[::2]],
            "turn_trace_bench": [round(float(x), 2) for x in tb_[::2]]}


# ---------------------------------------------------------------------------------- commands

def cmd_runs(a):
    runs = load_runs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS.write_text(json.dumps(runs))
    from collections import Counter
    c = Counter(r["hand"] for r in runs)
    print(f"{len(runs)} runs -> {RUNS}: " + ", ".join(f"{h} {c[h]}" for h in sorted(c)))
    return 0


def done_keys(path: Path) -> set:
    out = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                out.add(r["key"])
            except Exception:
                pass
    return out


def cmd_fit(a):
    runs = json.loads(RUNS.read_text())
    if a.hands:
        runs = [r for r in runs if r["hand"] in a.hands]
    runs = runs_subset(runs, a.per_hand) if a.per_hand else runs
    plants = plants_of(a.grid) if a.grid else [dict(zip(("kp", "tau", "fr", "fl"), map(float, s.split(","))))
                                               for s in a.plant]
    wi, wn = (int(x) for x in a.worker.split("/"))
    jobs = [(r, p, mu) for mu in a.mu for r in runs for p in plants]
    jobs = [j for k, j in enumerate(jobs) if k % wn == wi]
    out = Path(a.out) if a.out else ROWS
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out)
    scenes = {}
    t_start = time.time()
    n_new = 0
    jobs.sort(key=lambda j: (j[0]["hand"], j[2], j[0]["run"]))
    for r, p, mu in jobs:
        tag = f"{a.tip}_{a.model}_mu{mu}_ir{a.ir:g}"
        key = f"{r['run']}|{plant_name(p)}|{tag}|dxy{a.dxy_mm:g}"
        if key in done:
            continue
        h = r["hand"]
        if (h, mu) not in scenes:
            scenes.clear()
            scenes[(h, mu)] = HandScene(h, a.tip, a.model, mu if mu == "scene" else float(mu), a.ir)
        hs = scenes[(h, mu)]
        try:
            sim = hs.replay(r, p, dxy=(a.dxy_mm * 1e-3, 0.0))
            row = {"key": key, "run": r["run"], "hand": h, "plant": plant_name(p), **p, "tip": a.tip,
                   "model": a.model, "mu": mu, "ir": a.ir, "dxy_mm": a.dxy_mm, **compare(r, sim),
                   "bench_deg_turned": r["bench"]["deg_turned"], "wall_s": round(sim["wall_s"], 3),
                   "steps": sim["steps"], "when": time.strftime("%Y-%m-%d %H:%M")}
        except Exception as e:
            row = {"key": key, "run": r["run"], "hand": h, "plant": plant_name(p), **p, "status": "error",
                   "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        n_new += 1
        if n_new % 20 == 0:
            print(f"[{wi}/{wn}] {n_new} new rows, {time.time() - t_start:.0f} s, last {key} "
                  f"turn {row.get('turn_sim_end')} vs {row.get('turn_bench_end')}", flush=True)
    print(f"[{wi}/{wn}] done: {n_new} new rows in {time.time() - t_start:.0f} s", flush=True)
    return 0


def load_rows(path=ROWS):
    rows = {}
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        rows[r["key"]] = r
    return [r for r in rows.values() if r.get("status") != "error"]


def score_plants(rows, tag=None):
    """Per plant: hand-averaged turn RMSE, joint MAE, drop mismatch and end-turn error."""
    from collections import defaultdict
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if tag and not r["key"].endswith(tag):
            continue
        by[f"{r['plant']}_mu{r.get('mu', 'scene')}"][r["hand"]].append(r)
    out = []
    for pl, hands in by.items():
        def hm(f):
            return float(np.mean([np.mean([f(x) for x in rs]) for rs in hands.values()]))
        out.append({"plant": pl, "n_hands": len(hands), "n": sum(len(v) for v in hands.values()),
                    "turn_rmse": hm(lambda x: x["turn_rmse"]),
                    "turn_end_err": hm(lambda x: abs(x["turn_sim_end"] - x["turn_bench_end"])),
                    "joint_mae": hm(lambda x: x["joint_mae"]),
                    "drop_mismatch": hm(lambda x: float(x["dropped_sim"] != x["dropped_bench"])),
                    **{k: hands[next(iter(hands))][0][k] for k in ("kp", "tau", "fr", "fl")}})
    for o in out:
        o["cost"] = o["turn_rmse"] / 10.0 + o["joint_mae"] / 5.0 + o["drop_mismatch"]
    return sorted(out, key=lambda o: o["cost"])


def cmd_summary(a):
    rows = load_rows(Path(a.rows) if a.rows else ROWS)
    sc = score_plants(rows)
    print(f"{len(rows)} rows, {len(sc)} plants")
    print(f"{'plant':<32} {'n':>4} {'turnRMSE':>9} {'endErr':>7} {'jointMAE':>9} {'dropMis':>8} {'cost':>6}")
    for o in sc[:a.top]:
        print(f"{o['plant']:<32} {o['n']:>4} {o['turn_rmse']:>9.2f} {o['turn_end_err']:>7.2f} {o['joint_mae']:>9.2f} "
              f"{o['drop_mismatch']:>8.2f} {o['cost']:>6.2f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("runs")
    f = sub.add_parser("fit")
    f.add_argument("--grid", default=None, choices=sorted(GRIDS))
    f.add_argument("--plant", nargs="*", default=[], help="kp,tau,fr,fl")
    f.add_argument("--per-hand", type=int, default=0)
    f.add_argument("--hands", nargs="*", default=None)
    f.add_argument("--worker", default="0/1")
    f.add_argument("--tip", default="tpu2.7")
    f.add_argument("--model", default="pads")
    f.add_argument("--mu", nargs="+", default=["scene"])
    f.add_argument("--ir", type=float, default=100.0)
    f.add_argument("--dxy-mm", type=float, default=0.0)
    f.add_argument("--out", default=None)
    s = sub.add_parser("summary")
    s.add_argument("--rows", default=None)
    s.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    return {"runs": cmd_runs, "fit": cmd_fit, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
