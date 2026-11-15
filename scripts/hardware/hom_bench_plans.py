#!/usr/bin/env python3
"""Deploy plans of the first real-trial series (docs/notes/20261105-object_seating_and_tracking/
20261105-object_seating_and_tracking_plan.md, section "Reproduction of the simulated experiments"), one per layout and
task, in the format of manta_hand.plan.HandPlan, which the station reads, with the HOM world of the task-conflict study
that the bench controller (src/sr2hand/control/hom_bench.py) needs (2026-11-15).

The series runs the five layouts of the study on the SCS0009 plant (sr2hand.robot.sr2_layouts.FIRST_SERIES), each with
its S3 pick per task: the grasp, HOM's commanded forces and its contact modes refined together
(docs/experiments/20261103-hom_task_conflict/rows_refine/sv25). The cylinder tasks (pin-push, press) run at all five
layouts. The sphere tasks (roll, twist) run at the generalist (70,110) and the pin-push specialist (75,55) only,
because at the other three layouts the sphere's stem touches a finger (the plan's decision 2, 2026-11-12). That makes
14 plans; --all adds the other six.

Each plan has
  mounts_palm_mm  the layout in the driver's palm coordinates: P less each travel box's offset from the driver's origin,
                  (-6.18, -2.40) mm on the thumb, (6.18, 3.40) on the index, (6.18, -3.40) on the middle
                  (sr2_layouts.driver_from_P)
  poses           open: HOM's start pose, the pads 8 mm off the object (hom_tasks.open_pose); grip: the touch pose of the
                  same call, the pads on the object before the forces come on. Simulator degrees, as in every HandPlan.
  meta.hom        the GPU world: task, object, layout (d6 in P, x_sep and y_sep, lattice call, role), the S3 grasp v,
                  HOM's forces and contact modes prm, the variant and its hold, the plant, the study's scores, and the
                  world's numbers (setup, mount bodies, open and touch poses) that hom_bench_world.world_from_plan checks
  meta.mounts     every coordinate of the layout and its travel check (sr2_layouts.layout_report)
  meta.driver     HandPlan.schema_errors() and validate() of the plan: mounts outside the driver's envelope and poses
                  outside the servos' calibrated ranges

Commands
  write    the plans to docs/experiments/20261115-hom_bench_trials/plans/ and their index 20261115-plans_index.json
  verify   per plan: the driver's checks; hom_bench_world.world_from_plan (the plan's numbers rebuilt from its world);
           check_world against the GPU world; HomBench through the bench units (hom_bench.simulate with
           hom_bench_world.make_bench) over the whole task at the world's 0.01 s and --seed, against the study's GPU row
           and the CPU replica of 2026-11-05 at that seed; and a --seconds run at the bench's 0.02 s. For a layout
           outside the travel boxes, the same S3 pick at the nearest layout inside them (sr2_layouts.nearest_fit) at
           seeds 20-24, against the study's layout at the same seeds. Writes 20261115-plans_check.json.

    export SCALE_HAND=cad_v2 HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25r \\
           CUDA_VISIBLE_DEVICES= NUMPY_MADVISE_HUGEPAGE=0
    ~/.claude/bin/resguard.sh run --mem 3G -- .venv/bin/python scripts/hardware/hom_bench_plans.py write [--all]
    ~/.claude/bin/resguard.sh run --mem 3G -- .venv/bin/python scripts/hardware/hom_bench_plans.py verify [--seed 20]
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")       # CPU only: hom_gpu imports warp, nothing launches a kernel

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for _paths
import _paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/experiments/20261115-hom_bench_trials"
PLANS = OUT / "plans"
DATE = "20261115"
TASKS = ("pinpush", "press", "roll", "twist")
SPHERE_LAYOUTS = ("generalist", "pinpush")         # decision 2 of the plan (2026-11-12): the sphere's stem clears here
OTRACK = ROOT / "docs/experiments/20261105-object_tracking_sim/rows/rollouts.jsonl"   # the CPU replica of 2026-11-05
VAL_SEEDS = (20, 21, 22, 23, 24)
SIM_JOINTS = ("yaw", "mcp", "pip")
OPEN_RAMP_S, OPEN_HOLD_S, GRIP_RAMP_S = 1.0, 0.5, 1.0


def series(all_pairs=False):
    """[(role, (x_sep, y_sep), call, task)] of the series."""
    from sr2hand.robot import sr2_layouts as SL
    out = []
    for role, (xy, call) in SL.FIRST_SERIES.items():
        for task in TASKS:
            if all_pairs or task in ("pinpush", "press") or role in SPHERE_LAYOUTS:
                out.append((role, xy, call, task))
    return out


def plan_path(role, task):
    return PLANS / f"{DATE}-{role}_{task}_plan.json"


def _deg(q):
    return {f: {j: math.degrees(float(q[f][k])) for k, j in enumerate(SIM_JOINTS)} for f in q}


def _lst(a):
    return [float(x) for x in np.asarray(a, float).ravel()]


def _driver():
    host = ROOT / "src/sr2hand/driver/manta/host"
    if str(host) not in sys.path:
        sys.path.insert(0, str(host))
    from manta_hand import plan as PL
    return PL


def build_plan(role, xy, call, task):
    """The plan dict of one layout and task."""
    import hom_bench_world as W
    from sr2hand.robot import sr2_layouts as SL
    PL = _driver()
    HG, R, _E, _FP = W.modules("sv25")
    T = HG.T
    d6 = SL.sym6(*xy)
    p = W.study_pick(task, xy, "S3")
    if p["call"] != call or np.abs(np.asarray(p["d6"]) - d6).max() > 1e-9:
        raise RuntimeError(f"{role} {task}: the study's call {p['call']} d {p['d6']} is not {call} {d6.tolist()}")
    w = W.bench_world(task, d6, p["v"], p["prm"], variant=p["variant"])
    q_open, q_touch = T.open_pose(w["setup"], w["m_pt"])
    if max(float(np.abs(np.asarray(q_open[f]) - w["q_open"][f]).max()) for f in q_open) > 0.0:
        raise RuntimeError(f"{role} {task}: open_pose on the bench's model differs from the world's open pose")
    rep = SL.layout_report(d6)
    st = w["setup"]
    shape = st["shape"]
    obj = ("sphere, 40 mm" if shape == "sphere" else
           f"cylinder, {1e3 * 2 * st['r']:g} mm across and {1e3 * 2 * st['hl']:g} mm long, axis along y_P yawed by psi")
    hom = {"task": task, "d_mm": p["d_mm"], "shape": shape, "role": role, "xy_mm": list(map(float, xy)), "call": call,
           "d6_P_mm": _lst(d6), "pick": "S3", "v": p["v"], "prm": p["prm"], "task_prm": w["task_prm"],
           "variant": p["variant"], "hold_s": w["hold_s"], "dt_world_s": w["dt"], "homparams": "hom.HomParams() at dt",
           "plant": {"name": "sv25", "spec": p["plant_spec"], "SCALE_HAND": os.environ.get("SCALE_HAND"),
                     "HOS_PLANT": os.environ.get("HOS_PLANT"), "HOS_SCENES": os.environ.get("HOS_SCENES"),
                     "HOM_ROWS_TAG": os.environ.get("HOM_ROWS_TAG")},
           "scores": {"search_mean_seeds_0_2": p["search_mean"], "val_seeds": p["val_seeds"],
                      "val_scores": p["val_scores"], "val_mean": p["val_mean"], "score": "hom_landscape.rollout_score"},
           "rows": p["rows"],
           "setup": {"palm_m": _lst(st["palm"]), "quat": _lst(st["quat"]), "r_m": float(st["r"]), "hl_m": float(st["hl"]),
                     "psi_rad": float(st.get("psi", 0.0)),
                     "mounts_m": {f: _lst(st["mounts"][f]) for f in st["mounts"]},
                     "targets_m": {f: _lst(st["targets"][f]) for f in st["targets"]},
                     "n_out": {f: _lst(st["n_out"][f]) for f in st["n_out"]}},
           "body_mounts_m": {f: _lst(w["body_mounts"][f]) for f in w["body_mounts"]},
           "q_open_rad": {f: _lst(q_open[f]) for f in q_open}, "q_touch_rad": {f: _lst(q_touch[f]) for f in q_touch},
           "scenes": {"pads": str(w["scene_pads"].relative_to(ROOT)), "pt": str(w["scene_pt"].relative_to(ROOT)),
                      "pads_with_mounts": str(Path(w["p_pads"]).relative_to(ROOT))}}
    poses = [{"name": "open", "ramp_s": OPEN_RAMP_S, "hold_s": OPEN_HOLD_S, "joints": _deg(q_open)},
             {"name": "grip", "ramp_s": GRIP_RAMP_S, "hold_s": 0.0, "joints": _deg(q_touch)}]
    plan = {"design": f"hom_s3_{role}_x{xy[0]:g}_y{xy[1]:g}_{task}",
            "mounts_palm_mm": {f: _lst(SL.driver_from_P(f, xy_)) for f, xy_ in SL.mounts(d6).items()},
            "poses": poses,
            "meta": {"object": obj, "written": datetime.now().strftime("%Y-%m-%d %H:%M"),
                     "source": "scripts/hardware/hom_bench_plans.py write: the task-conflict study's S3 pick "
                               f"({p['rows']}, pick S3, call {call}) on the SCS0009 plant",
                     "frames": "mounts_palm_mm are the driver's palm coordinates (manta_hand.plan); meta.hom.d6_P_mm "
                               "and meta.mounts.fingers.<f>.P_mm are the simulator's palm frame P "
                               "(assets/cad_v3/frames.json); poses are simulator degrees (yaw, mcp, pip)",
                     "trial": "HOM runs the trial in closed loop from the open pose (HomBench); the grip pose is the "
                              "touch pose, for the driver's checks and an open-loop look",
                     "hom": hom, "mounts": rep}}
    hp = PL.HandPlan(plan["design"], {f: tuple(v) for f, v in plan["mounts_palm_mm"].items()},
                     [PL.Pose(x["name"], x["ramp_s"], x["hold_s"], x["joints"]) for x in poses], plan["meta"])
    plan["meta"]["driver"] = {"schema_errors": hp.schema_errors(), "violations": [str(v) for v in hp.validate()]}
    return plan


def cmd_write(a) -> int:
    PLANS.mkdir(parents=True, exist_ok=True)
    index = []
    for role, xy, call, task in series(a.all):
        t0 = time.time()
        plan = build_plan(role, xy, call, task)
        path = plan_path(role, task)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(plan, indent=1) + "\n")
        os.replace(tmp, path)
        h, m = plan["meta"]["hom"], plan["meta"]["mounts"]
        index.append({"plan": str(path.relative_to(ROOT)), "role": role, "xy_mm": list(xy), "call": call, "task": task,
                      "mounts_palm_mm": plan["mounts_palm_mm"], "fits_travel": m["fits"],
                      "min_margin_mm": m["min_margin_mm"], "driver_violations": plan["meta"]["driver"]["violations"],
                      "prm": h["prm"], "val_mean": h["scores"]["val_mean"]})
        print(f"{path.name}: fits {m['fits']} (margin {m['min_margin_mm']:+.2f} mm at {m['min_margin_at']}), "
              f"driver violations {len(plan['meta']['driver']['violations'])}, S3 val mean {h['scores']['val_mean']}, "
              f"{time.time() - t0:.1f} s", flush=True)
    (OUT / f"{DATE}-plans_index.json").write_text(json.dumps({"written": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                               "plans": index}, indent=1) + "\n")
    return 0


def _gpu_row(p, seed):
    import hom_landscape as L
    for line in (ROOT / p["rows"]).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if int(r["call"]) == p["call"] and r["pick"] == "S3" and int(r["seed"]) == seed:
                return round(float(L.rollout_score(p["task"], r)), 4)
    return None


def cmd_verify(a) -> int:
    import hom_bench_world as W
    import hom_landscape as L
    from sr2hand.control import hom_bench as HB
    from sr2hand.robot import sr2_layouts as SL
    PL = _driver()
    cpu = {}
    for line in OTRACK.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("config") == "base" and r.get("status") == "ok":
                cpu[(r["task"], int(r["call"]), int(r["seed"]))] = r["score"]
    out = []
    paths = sorted(PLANS.glob(a.plans))
    for path in paths:
        t0 = time.time()
        raw = json.loads(path.read_text())
        h = raw["meta"]["hom"]
        task, call = h["task"], int(h["call"])
        hp = PL.HandPlan.from_json(path)
        rec = {"plan": str(path.relative_to(ROOT)), "task": task, "role": h["role"], "call": call,
               "schema_errors": hp.schema_errors(), "driver_violations": [str(v) for v in hp.validate()]}
        w = W.world_from_plan(raw)
        rec["stored_numbers"] = "rebuilt within 1e-9"
        chk = W.check_world(w, deep=not a.quick)
        rec["check_world"] = chk
        b = W.make_bench(w)
        t1 = time.time()
        stream, res = HB.simulate(b, seed=a.seed)
        rec["full"] = {"dt_s": b.prm.dt, "seed": a.seed, "samples": len(stream), "wall_s": round(time.time() - t1, 2),
                       "score": round(float(L.rollout_score(task, res)), 4), "held": res.get("held"),
                       "gpu_score": _gpu_row({"rows": h["rows"], "call": call, "task": task}, a.seed),
                       "cpu_20261105_score": cpu.get((task, call, a.seed)),
                       "clipped_samples": sum(1 for s in stream if s["clipped"]),
                       "clipped": sorted({c for s in stream for c in s["clipped"]})}
        b2 = W.make_bench(w, dt=HB.DT_BENCH)
        s2, _ = HB.simulate(b2, seed=0, seconds=a.seconds)
        rec["dry"] = {"dt_s": b2.prm.dt, "seconds": a.seconds, "samples": len(s2),
                      "phases": sorted(set(s["phase"] for s in s2)),
                      "clipped": sorted({c for s in s2 for c in s["clipped"]})}
        if not raw["meta"]["mounts"]["fits"]:
            d6 = np.asarray(h["d6_P_mm"], float)
            fit = SL.nearest_fit(d6)
            import hom_grasp_refine as R
            wf = W.bench_world(task, fit, h["v"], h["prm"], variant=h["variant"])
            sc = {"study": [], "fit": []}
            for sd in VAL_SEEDS:
                for name, ww in (("study", w), ("fit", wf)):
                    _st, rs = HB.simulate(W.make_bench(ww), seed=sd)
                    sc[name].append(round(float(L.rollout_score(task, rs)), 4))
            rec["nearest_fit"] = {"d6_P_mm": _lst(fit), "xy_mm": list(SL.seps(fit)),
                                  "toy_feasible": bool(R.feasible(task, fit, np.asarray(h["v"], float))),
                                  "seeds": list(VAL_SEEDS), "cpu_scores_study_layout": sc["study"],
                                  "cpu_scores_fit_layout": sc["fit"], "mean_study": round(float(np.mean(sc["study"])), 4),
                                  "mean_fit": round(float(np.mean(sc["fit"])), 4), "check_world_ok": W.check_world(wf, deep=False)["ok"]}
        rec["wall_s"] = round(time.time() - t0, 1)
        out.append(rec)
        f = rec["full"]
        print(f"{path.name}: checks {'ok' if chk['ok'] else 'FAILED'}; full run score {f['score']} (GPU {f['gpu_score']}, "
              f"CPU 11-05 {f['cpu_20261105_score']}), clipped {f['clipped_samples']}; dry run {rec['dry']['samples']} "
              f"samples; driver violations {len(rec['driver_violations'])}"
              + (f"; nearest fit {rec['nearest_fit']['xy_mm']}: CPU mean {rec['nearest_fit']['mean_fit']} against "
                 f"{rec['nearest_fit']['mean_study']}" if "nearest_fit" in rec else "") + f" ({rec['wall_s']} s)", flush=True)
    path = OUT / f"{DATE}-plans_check.json"         # merged by plan, so the plans can be checked in several runs
    old = json.loads(path.read_text())["plans"] if path.exists() else []
    mine = {r["plan"] for r in out}
    plans = sorted([r for r in old if r["plan"] not in mine] + out, key=lambda r: r["plan"])
    summary = {"written": datetime.now().strftime("%Y-%m-%d %H:%M"), "seed": a.seed, "dry_seconds": a.seconds,
               "env": {k: os.environ.get(k) for k in ("SCALE_HAND", "HOS_PLANT", "HOS_SCENES", "HOM_ROWS_TAG")},
               "all_checks_ok": all(r["check_world"]["ok"] for r in plans), "plans": plans}
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    os.replace(tmp, path)
    return 0 if all(r["check_world"]["ok"] for r in out) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--all", action="store_true", help="also the sphere tasks at the twist, roll and press specialists")
    v = sub.add_parser("verify")
    v.add_argument("--plans", default="*_plan.json", help="glob of plan files in the plans folder")
    v.add_argument("--seed", type=int, default=20, help="seed of the full run (a validation seed of the study)")
    v.add_argument("--seconds", type=float, default=1.0, help="length of the run at the bench's 0.02 s")
    v.add_argument("--quick", action="store_true", help="check_world without the baked plant's field and stepping check")
    a = ap.parse_args()
    return {"write": cmd_write, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
