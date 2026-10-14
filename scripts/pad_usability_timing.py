#!/usr/bin/env python3
"""CPU cost per physics step of the fingertip contact models on the real_v1 hand at the scale study's three-finger grip.

The plant is the scale study's cell scene of one layout and cylinder (hand_object_scale_mpc.setup: welded palm at the
cell's best HOM candidate grasp, servo-limited plant, 1 ms steps, elliptic cone, impratio 100, Newton solver at
tolerance 1e-10 / 100 iterations / 50 line-search iterations, friction 1). Fingertip models:

  pads      the printed TPU block as 1 mm sphere pads (the cell's pad scene; every pad collides with every geom)
  pads_f    pads that collide with the cylinder only, set on the compiled model as the MPC Planner's pad_filter does
            (pads contype 2, conaffinity 0; cylinder conaffinity | 2; body-level filter fields left as compiled)
  pads_fx   pads_f with the bodies' contype/conaffinity recomputed from their geoms, as a compiled XML filter gives
  pads_fs   pads_f at the GPU planner's solver settings: tolerance 1e-6, 20 iterations, 10 line-search iterations
  pads_fs50 pads_f at tolerance 1e-6, 20 iterations and the cell scene's 50 line-search iterations
  pads2     the cell scene rebuilt with 2 mm pads of radius 1.5 mm (reorient_backends.replace_tips(s=0.002, rs=0.0015)),
            written as <tag>_pads2__<plant>.xml beside the cached cell scenes
  pads2_f   pads2 with the pad filter of pads_f
  pt        the TPU block as one convex mesh, MuJoCo point contact (condim 3)
  pt4       pt with condim 4, priority 1 and friction (1, 0.0028, 1e-4) on each {f}_tipgeom (torsional friction 2.8 mm,
            the pads' twist-onset arm at 3.5-7 N per finger)
  ptmc      pt with MuJoCo's multi-point convex collision (mjENBL_MULTICCD)

States (the fingertip models share qpos and qvel, so a state of one loads into every other):
  open      own plant after reset, finger commands held at the open pose
  grip      own plant 0.8 s after the grip command (the model closes the grasp itself)
  grip_ref  the pads plant's grip state (time, qpos, qvel, act, warm start, commands) loaded into the model: what a
            planner model that differs from the plant starts every rollout from
Per cell, model and state:
  step     hand_object_scale_mpc_timing.step_split, `--repeats` times over `--nstep` steps (state restored each time):
           mj_step time, mj_fwdPosition time, mj_fwdConstraint time (us per step), contacts and constraint rows
  timer    `--repeats` passes of `--nstep` mj_step with mujoco.set_mjcb_time(time.perf_counter): each d.timer stage in
           us per step under the callback and as a share of mjTIMER_STEP
  held     one pass of `--nstep` steps: solver iterations, bad-acceleration resets (mjWARN_BADQACC), finger forces and
           object displacement at its end; held = no reset, the object within 5 mm and two fingers each pressing with at
           least its weight
  rollout  (grip and grip_ref) hand_object_scale_mpc_timing.rollout_time: mujoco.rollout of 64 plans x 0.5 s with
           N(0, 0.08 rad) command offsets, `--rollout-repeats` times per thread count; plus one untimed rollout of the
           same plans that counts the plans MuJoCo reset after a bad acceleration (time going back)

  taskset -c 17-19 .venv/bin/python scripts/pad_usability_timing.py --plant kp0.72_kv0_fr0.18_fl0_dp0.0144 \\
      --cells cylinder:16:0.625 cylinder:25:0.125 --threads 1 3
Rows: docs/experiments/20261014-pad_usability/cost/cpu_timing.jsonl, one fsynced line per (cell, model, state, kind,
threads); resumable by those keys.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import hand_object_scale_mpc as M  # noqa: E402
import hand_object_scale_mpc_timing as TM  # noqa: E402
import hand_object_scale_sim as S  # noqa: E402
import hom_turn3 as H  # noqa: E402
import reorient_backends as RB  # noqa: E402

OUT = ROOT / "docs/experiments/20261014-pad_usability/cost/cpu_timing.jsonl"
MODELS = ["pads", "pads_f", "pads_fx", "pads_fs", "pads_fs50", "pads2", "pads2_f", "pt", "pt4", "ptmc"]
PADS2_S, PADS2_RS = 0.002, 0.0015
PT4_FRICTION = (1.0, 0.0028, 0.0001)
SOLVER = {"pads_fs": dict(tolerance=1e-6, iterations=20, ls_iterations=10),
          "pads_fs50": dict(tolerance=1e-6, iterations=20, ls_iterations=50)}
STAGES = ("STEP", "FORWARD", "POSITION", "POS_KINEMATICS", "POS_INERTIA", "POS_COLLISION", "COL_BROAD", "COL_NARROW",
          "POS_MAKE", "POS_PROJECT", "VELOCITY", "ACTUATION", "CONSTRAINT", "ADVANCE")
HOLD_MM = 5.0


def pads2_scene(cell, shape, dd):
    """The cell scene with 2 mm pads: hand_object_scale_sim.cell_scenes under the model name 'pads2', its
    replace_tips call redirected to spacing 2 mm and radius 1.5 mm."""
    orig = RB.replace_tips

    def rt(root, tip, model, MU, out_dir, tool=RB.OBJ, s=RB.PAD_S, **kw):
        if model == "pads2":
            return orig(root, tip, "pads", MU, out_dir, tool=tool, s=PADS2_S, rs=PADS2_RS, **kw)
        return orig(root, tip, model, MU, out_dir, tool=tool, s=s, **kw)
    RB.replace_tips = rt
    try:
        return S.cell_scenes(cell, shape, dd, models=("pads2",))["pads2"]
    finally:
        RB.replace_tips = orig


def pad_filter(m, tb, body_level=False):
    """Pads collide with the object only (hand_object_scale_mpc.Planner, prm pad_filter). body_level: also recompute
    each body's contype/conaffinity as the OR over its geoms, as the compiler does for a filter written in the XML."""
    pads = np.array([g for g in range(m.ngeom) if "_pad" in (m.geom(g).name or "")])
    tool = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == tb])
    m.geom_contype[pads], m.geom_conaffinity[pads] = 2, 0
    m.geom_conaffinity[tool] = m.geom_conaffinity[tool] | 2
    if body_level:
        for b in range(m.nbody):
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
            m.body_contype[b] = int(np.bitwise_or.reduce(m.geom_contype[gs])) if gs else 0
            m.body_conaffinity[b] = int(np.bitwise_or.reduce(m.geom_conaffinity[gs])) if gs else 0
    return int(len(pads))


def make_plant(mdl, paths, meta, gg):
    """The MjPlant of one fingertip model, with its edits applied to the compiled model."""
    import mujoco
    base = {"pads": "pads", "pads_f": "pads", "pads_fx": "pads", "pads_fs": "pads", "pads_fs50": "pads",
            "pads2": "pads2", "pads2_f": "pads2", "pt": "pt", "pt4": "pt", "ptmc": "pt"}[mdl]
    p = H.MjPlant(paths[base], meta)
    p.xml_path = str(paths[base])
    S.set_palm(p.m, gg["palm"])
    m = p.m
    edits = {"scene": Path(paths[base]).name}
    if mdl in ("pads_f", "pads_fx", "pads_fs", "pads_fs50", "pads2_f"):
        edits["pads_filtered"] = pad_filter(m, p.tb, body_level=(mdl == "pads_fx"))
    if mdl in SOLVER:
        for k, v in SOLVER[mdl].items():
            setattr(m.opt, k, v)
    if mdl == "pt4":
        for f in S.FINGERS:
            g = m.geom(f"{f}_tipgeom").id
            m.geom_condim[g] = 4
            m.geom_priority[g] = 1
            m.geom_friction[g] = PT4_FRICTION
    if mdl == "ptmc":
        m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    edits.update(tolerance=float(m.opt.tolerance), iterations=int(m.opt.iterations),
                 ls_iterations=int(m.opt.ls_iterations), multiccd=bool(m.opt.enableflags
                                                                     & int(mujoco.mjtEnableBit.mjENBL_MULTICCD)))
    return p, edits


def full_state(mj, m, d):
    """FULLPHYSICS state, commands and warm start of d."""
    spec = mj.mjtState.mjSTATE_FULLPHYSICS
    x = np.empty(mj.mj_stateSize(m, spec))
    mj.mj_getState(m, d, x, spec)
    return x, d.ctrl.copy(), d.qacc_warmstart.copy()


def load_state(mj, m, d, st):
    x, ctrl, warm = st
    mj.mj_setState(m, d, x, mj.mjtState.mjSTATE_FULLPHYSICS)
    d.ctrl[:] = ctrl
    d.qacc_warmstart[:] = warm
    mj.mj_forward(m, d)
    d.qacc_warmstart[:] = warm


def timer_pass(mj, m, d, n):
    """us per step of each d.timer stage over n mj_step with the perf_counter time callback (state restored)."""
    st = full_state(mj, m, d)
    ids = {s: int(getattr(mj.mjtTimer, f"mjTIMER_{s}")) for s in STAGES}
    before = {s: float(d.timer[i].duration) for s, i in ids.items()}
    mj.set_mjcb_time(time.perf_counter)
    try:
        for _ in range(n):
            mj.mj_step(m, d)
    finally:
        mj.set_mjcb_time(None)
    us = {s: 1e6 * (float(d.timer[i].duration) - before[s]) / n for s, i in ids.items()}
    load_state(mj, m, d, st)
    return us


def held_pass(mj, p, n, meta):
    """n steps from the current state (restored afterwards): solver iterations, bad-acceleration resets, and whether
    the object is still held at the end."""
    m, d = p.m, p.d
    st = full_state(mj, m, d)
    w = int(mj.mjtWarning.mjWARN_BADQACC)
    bad0 = int(d.warning[w].number)
    p0 = d.qpos[p.qa:p.qa + 3].copy()
    niter = np.empty(n)
    for k in range(n):
        mj.mj_step(m, d)
        niter[k] = d.solver_niter[0]
    s = p.state()
    bad = int(d.warning[w].number) - bad0
    disp = float(1e3 * np.linalg.norm(d.qpos[p.qa:p.qa + 3] - p0))
    weight = float(meta["mass_kg"]) * 9.81
    loaded = sum(s["F"][f] >= weight for f in S.FINGERS)
    out = {"solver_niter_mean": float(niter.mean()), "solver_niter_max": int(niter.max()), "badqacc_resets": bad,
           "F_end_N": {f: round(float(s["F"][f]), 3) for f in S.FINGERS}, "obj_disp_mm": round(disp, 3),
           "fingers_loaded_end": int(loaded), "held": bool(bad == 0 and disp < HOLD_MM and loaded >= 2)}
    load_state(mj, m, d, st)
    return out


def rollout_resets(mj, m, d, act, nthread=3, nsample=64, nstep=500, sigma=0.08):
    """The plans of hand_object_scale_mpc_timing.rollout_time rolled out once with their states returned: how many
    MuJoCo reset after a bad acceleration (the state's time goes back)."""
    from mujoco import rollout
    spec = mj.mjtState.mjSTATE_FULLPHYSICS
    x0 = np.empty(mj.mj_stateSize(m, spec))
    mj.mj_getState(m, d, x0, spec)
    rng = np.random.default_rng(0)
    ctrl = np.tile(d.ctrl, (nsample, nstep, 1))
    ctrl[:, :, act] += rng.normal(0, sigma, (nsample, 1, len(act)))
    with rollout.Rollout(nthread=nthread) as pool:
        datas = [mj.MjData(m) for _ in range(nthread)]
        states, _ = pool.rollout(m, datas, x0[None], ctrl, nstep=nstep)
    t = states[:, :, 0]
    return int(np.sum(np.any(np.diff(t, axis=1) < 0, axis=1)))


def contact_classes(p):
    """Contacts at the current state: with the object (by finger, and how many are pad geoms), and all others."""
    m, d = p.m, p.d
    out = {"tool_by_finger": {f: 0 for f in S.FINGERS}, "tool_pad_contacts": 0, "other": 0}
    for i in range(d.ncon):
        g1, g2 = int(d.contact[i].geom[0]), int(d.contact[i].geom[1])
        if (g1 in p.tool_geoms) == (g2 in p.tool_geoms):
            out["other"] += 1
            continue
        g = g2 if g1 in p.tool_geoms else g1
        f = p.finger_of.get(g)
        if f is None:
            out["other"] += 1
            continue
        out["tool_by_finger"][f] += 1
        out["tool_pad_contacts"] += int("_pad" in (m.geom(g).name or ""))
    out["dims"] = sorted({int(d.contact[i].dim) for i in range(d.ncon)})
    return out


def stats(v):
    v = np.asarray(v, float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "values": [float(x) for x in v]}


def done_keys(path: Path) -> set:
    out = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") == "ok":
                out.add((r["cell"], r["model"], r["state"], r["kind"], r.get("nthread", 0)))
    return out


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main() -> int:
    import mujoco as mj
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plant", default="kp0.72_kv0_fr0.18_fl0_dp0.0144")
    ap.add_argument("--cells", nargs="+", default=["cylinder:16:0.625", "cylinder:25:0.125"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rollout-repeats", type=int, default=3)
    ap.add_argument("--nstep", type=int, default=500)
    ap.add_argument("--hom-rows", default=str(M.OUT / "hom_postoff_servo/turn.jsonl"))
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    S.PLANT = a.plant
    done = done_keys(a.out)
    best = M.hom_best(Path(a.hom_rows))
    cpus = sorted(os.sched_getaffinity(0))
    states = ("open", "grip", "grip_ref")
    for spec in a.cells:
        shape, dd, t = spec.split(":")
        dd, t = float(dd), float(t)
        cell, _, _ = next(c for c in S.cell_list("diag") if c[1] == shape and c[2] == dd and c[0].get("t") == t)
        ci = best[(cell["tag"], shape, dd)]["cand"]
        plant0, meta, geo, sc, gg = M.setup(cell, shape, dd, ci)
        paths = {"pads": sc["pads"][0], "pt": sc["pt"][0]}
        if any(x.startswith("pads2") for x in a.models):
            paths["pads2"] = pads2_scene(cell, shape, dd)[0]
        grip = np.array([meta["grip"][f][k] for f in S.FINGERS for k in range(3)])
        q_open = np.array([meta["q0"][n] for n in S.NAMES])
        # the reference grip state: the pads plant (the cell scene as the scale study runs it) 0.8 s after the command
        plant0.reset(0)
        plant0.set_targets(grip)
        plant0.advance(0.8)
        ref = full_state(mj, plant0.m, plant0.d)
        del plant0
        for mdl in a.models:
            keys = [(spec, mdl, st, k, 0) for st in states for k in ("step", "timer", "held")] + \
                   [(spec, mdl, st, "rollout", n) for st in ("grip", "grip_ref") for n in a.threads]
            if all(k in done for k in keys):
                continue
            base = {"cell": spec, "tag": cell["tag"], "shape": shape, "d_mm": dd, "cand": ci, "plant": a.plant,
                    "model": mdl, "hand_model": S.HAND, "cpus": cpus, "mujoco": mj.__version__,
                    "when": time.strftime("%Y-%m-%d %H:%M")}
            try:
                p, edits = make_plant(mdl, paths, meta, gg)
            except Exception as e:
                append(a.out, {**base, "state": "build", "kind": "build", "status": "error",
                               "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]})
                print(f"{spec} {mdl}: build failed: {e}", flush=True)
                continue
            m, d = p.m, p.d
            base.update(ngeom=int(m.ngeom), nbody=int(m.nbody),
                        n_pad_geoms=sum("_pad" in (m.geom(g).name or "") for g in range(m.ngeom)), **edits)
            wq = int(mj.mjtWarning.mjWARN_BADQACC)
            for st in states:
                p.reset(0)
                p.set_targets(q_open)                 # the open state holds the open pose
                bad0 = int(d.warning[wq].number)
                if st == "grip":
                    p.set_targets(grip)
                    p.advance(0.8)
                elif st == "grip_ref":
                    load_state(mj, m, d, ref)
                s = p.state()
                info = {"state": st, "t_state_s": round(float(d.time), 4),
                        "F_fingers_N": {f: round(float(s["F"][f]), 3) for f in S.FINGERS},
                        "badqacc_resets_to_state": int(d.warning[wq].number) - bad0,
                        "contacts_at_state": contact_classes(p)}
                if (spec, mdl, st, "held", 0) not in done:
                    hp = held_pass(mj, p, a.nstep, meta)
                    append(a.out, {**base, **info, "kind": "held", "nstep": a.nstep, **hp, "status": "ok"})
                    print(f"{spec} {mdl:<9} {st:<8} held {hp['held']} resets {hp['badqacc_resets']} "
                          f"(to state {info['badqacc_resets_to_state']}) disp {hp['obj_disp_mm']} mm F {hp['F_end_N']} "
                          f"niter {hp['solver_niter_mean']:.2f}", flush=True)
                if (spec, mdl, st, "step", 0) not in done:
                    reps = [TM.step_split(mj, m, d, n=a.nstep) for _ in range(a.repeats)]
                    row = {**base, **info, "kind": "step", "nstep": a.nstep, "repeats": a.repeats,
                           **{k: stats([r[k] for r in reps]) for k in ("step_us", "fwd_position_us", "constraint_us")},
                           "ncon_mean": float(np.mean([r["ncon_mean"] for r in reps])),
                           "ncon_max": int(max(r["ncon_max"] for r in reps)),
                           "nefc_mean": float(np.mean([r["nefc_mean"] for r in reps])), "status": "ok"}
                    append(a.out, row)
                    print(f"{spec} {mdl:<9} {st:<8} step {row['step_us']['median']:7.1f} us "
                          f"[{row['step_us']['min']:.1f}-{row['step_us']['max']:.1f}]  pos "
                          f"{row['fwd_position_us']['median']:6.1f}  solve {row['constraint_us']['median']:6.1f}  "
                          f"ncon {row['ncon_mean']:.1f}  nefc {row['nefc_mean']:.1f}  F {info['F_fingers_N']}", flush=True)
                if (spec, mdl, st, "timer", 0) not in done:
                    reps = [timer_pass(mj, m, d, a.nstep) for _ in range(a.repeats)]
                    us = {k: stats([r[k] for r in reps]) for k in STAGES}
                    share = {k: stats([r[k] / r["STEP"] for r in reps]) for k in STAGES if k != "STEP"}
                    row = {**base, **info, "kind": "timer", "nstep": a.nstep, "repeats": a.repeats,
                           "stage_us": us, "stage_share": share, "status": "ok"}
                    append(a.out, row)
                    print(f"{spec} {mdl:<9} {st:<8} timer " + "  ".join(
                        f"{k} {share[k]['median']:.2f}" for k in ("POS_KINEMATICS", "COL_BROAD", "COL_NARROW",
                                                                   "POS_MAKE", "CONSTRAINT")), flush=True)
                if st in ("grip", "grip_ref"):
                    resets = None
                    for n in a.threads:
                        if (spec, mdl, st, "rollout", n) in done:
                            continue
                        if resets is None:
                            resets = rollout_resets(mj, m, d, p.act, nthread=max(a.threads))
                        reps = [TM.rollout_time(mj, m, d, n, act=p.act) for _ in range(a.rollout_repeats)]
                        row = {**base, **info, "kind": "rollout", "nthread": n, "nsample": 64, "nstep": 500,
                               "repeats": a.rollout_repeats, "plans_reset": resets,
                               "wall_s": stats([r["wall_s"] for r in reps]),
                               "us_per_step_thread": stats([r["us_per_step_thread"] for r in reps]), "status": "ok"}
                        append(a.out, row)
                        print(f"{spec} {mdl:<9} {st:<8} rollout {n:>2} threads: {row['wall_s']['median']:.3f} s "
                              f"[{row['wall_s']['min']:.3f}-{row['wall_s']['max']:.3f}]  plans reset {resets}/64",
                              flush=True)
            del p
    return 0


if __name__ == "__main__":
    sys.exit(main())
