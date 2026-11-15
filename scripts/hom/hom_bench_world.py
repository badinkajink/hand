#!/usr/bin/env python3
"""The HOM controller of the bench (src/sr2hand/control/hom_bench.py) on one world of the GPU studies: a layout, a grasp
and HOM's commanded forces and contact modes, built by the calls that hom_grasp_refine.evaluate makes before it hands
the world to hom_gpu.batch_child (2026-11-15).

bench_world(task, d6, v, prm) returns the arguments of HomBench.__init__ and two items it does not take yet:
  task, d_mm   the task and the object's diameter (hom_grasp_refine.TASKS: 40 mm sphere, 25 mm cylinder)
  setup        hom_grasp_refine.setup_of(hom_gpu, task, d6, v): the palm {P} in the world, the mounts, the pad-centre
               targets and outward normals of the grasp, the object's start orientation
  m_pt         the reference scene's pt model with the mount bodies at the layout and the palm set (HomBench's Mirror)
  q_open       hom_gpu.Scene.prepare: hom_tasks.open_pose on that model, the pads 8 mm off the object
  dt           the control period (s): the world's 0.01 unless given (the bench runs at 0.02)
  prm          hom.HomParams(dt=dt) with the variant's overrides (hom_tasks.VARIANTS; none for base and hold)
  p_pads       the pads plant with the mount bodies written at the layout: a copy of the reference scene beside it in
               the scene cache, <stem>__bench_<layout>.xml (hom_gpu moves the same bodies per world, and the CPU
               replicas move them on the compiled model)
  task_prm     {**hom_tasks.PRM, **prm}: the schedule's commanded forces and Eq. (2)'s contact-mode weights, the merge
               hom_gpu.finger_commands makes; HomBench.reset() passes hom_tasks.PRM alone
  hold_s       the plant holds the object at its start pose up to this time (hom_tasks.Plant.hold_s): 0.8 s on the
               cylinder tasks (variant hold of hom_ea.VARIANT), 0 on the sphere; HomBench.simulate() never sets it
  body_mounts  {finger: the mount body's position in the palm body (m)}
  world        the GPU world as evaluate() builds it: key, seed, setup, body_mounts, q_open, prm, var
make_bench(w) is HomBench on the world with its task_prm and hold_s. check_world(w) compares the
bench's objects with the world (mounts, palm, contact points, open pose, the schedule's forces and weights, HomParams,
the baked plant). study_pick(task, xy) reads a layout's S1, S2 or S3 pick of the task-conflict study's SCS0009 plant
(docs/experiments/20261103-hom_task_conflict/rows_refine/sv25), and world_from_plan(plan) rebuilds the world of a plan
written by scripts/hardware/hom_bench_plans.py and checks its stored numbers.

The plant is set before hand_object_scale_sim is imported: SCALE_HAND=cad_v2 (the v4b tip), HOS_PLANT and HOS_SCENES
of hom_grasp_refine.PLANTS (sv25: the SCS0009 with pads at 2.5 MPa, logs/20261102-hom_servo_plant/scenes), HOM_ROWS_TAG
when unset. A process that imported it with another plant gets a RuntimeError. hom_gpu imports warp, but nothing here
launches a kernel; the scripts that call this run with CUDA_VISIBLE_DEVICES empty.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")     # the v4b fingertip, as hom_gpu and hom_grasp_refine

import json  # noqa: E402
import sys  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from dataclasses import fields, replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

STUDY_ROWS = ROOT / "docs/experiments/20261103-hom_task_conflict/rows_refine"
LEVELS = {"S1": None, "S2": "grasp", "S3": "full"}            # pick -> hom_grasp_refine level (its search rows)
_SCENES = {}


# ------------------------------------------------------------------------------------------------ modules and plant

def _plant_env(plant):
    """(HOS_PLANT, HOS_SCENES, HOM_ROWS_TAG) of hom_grasp_refine.PLANTS[plant]: set when hand_object_scale_sim is not
    imported yet, checked when it is."""
    import hom_grasp_refine as R
    hp, scenes, tag = R.PLANTS[plant]
    if "hand_object_scale_sim" not in sys.modules:
        for k, v in (("HOS_PLANT", hp), ("HOS_SCENES", scenes)):
            if v is None:
                continue
            if os.environ.get(k, v) != v:
                raise RuntimeError(f"{k}={os.environ[k]} is set, but plant {plant} needs {v}")
            os.environ[k] = v
        os.environ.setdefault("HOM_ROWS_TAG", tag)
    return hp, scenes


def modules(plant="sv25"):
    """(hom_gpu, hom_grasp_refine, hom_ea, fp_sr2_toy) with the plant's environment; raises when the process runs another
    hand or plant. plant None: the plant the process was started with (hom_server.configure sets it), unchecked."""
    hp, scenes = _plant_env(plant) if plant is not None else (None, None)
    import fp_sr2_toy as FP
    import hom_ea as E
    import hom_gpu as HG
    import hom_grasp_refine as R
    S, K = HG.S, HG.K
    if K.HAND != "cad_v2":
        raise RuntimeError(f"SCALE_HAND is {K.HAND}: the study ran the v4b tip (cad_v2); set it before hom_tasks loads")
    if plant is None:
        return HG, R, E, FP
    want = {"servo": S.SERVO_PLANT, "work": S.WORK_PLANT}.get(hp, hp)
    if S.PLANT != want:
        raise RuntimeError(f"hand_object_scale_sim runs plant {S.PLANT}, plant {plant} is {want}")
    if scenes and Path(S.SCENES).resolve() != (ROOT / scenes).resolve():
        raise RuntimeError(f"hand_object_scale_sim's scene cache is {S.SCENES}, plant {plant} uses {scenes}")
    return HG, R, E, FP


def _scene(HG, shape, d_mm):
    """hom_gpu.Scene of the object, one per process (its pt model's mounts are moved by every prepare)."""
    key = (shape, float(d_mm))
    if key not in _SCENES:
        _SCENES[key] = HG.Scene(shape, d_mm)
    return _SCENES[key]


def layout_tag(d6):
    """File-name tag of six coordinates (mm): -35,0,35,55,35,-55 -> m35_0_35_55_35_m55."""
    return "_".join(f"{float(x):g}".replace("-", "m").replace(".", "p") for x in d6)


def bake_mounts(src, body_mounts, d6):
    """A copy of scene src with each <f>_mount body at body_mounts[f] (m, in its parent), written beside it (relative
    mesh paths) as <stem>__bench_<layout>.xml; rewritten only when its text would change."""
    src = Path(src)
    root = ET.parse(src).getroot()
    for f, p in body_mounts.items():
        b = root.find(f".//body[@name='{f}_mount']")
        if b is None:
            raise RuntimeError(f"{src.name}: no body {f}_mount")
        b.set("pos", " ".join(repr(float(x)) for x in p))
    dst = src.with_name(f"{src.stem}__bench_{layout_tag(d6)}.xml")
    text = ET.tostring(root, encoding="unicode")
    if not dst.exists() or dst.read_text() != text:
        tmp = dst.with_name(dst.name + f".{os.getpid()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, dst)
    return dst


# ------------------------------------------------------------------------------------------------ the world

def bench_world(task, d6, v, prm=None, dt=None, plant="sv25", seed=0, variant=None):
    """HomBench's arguments for the GPU world of grasp v on layout d6 (six coordinates in mm of the palm frame P) with
    HOM's forces and contact modes prm (hom_tasks.PRM keys; {} or None for HOM's defaults). dt: the control period (s),
    the world's 0.01 when None. plant: a hom_grasp_refine.PLANTS key whose environment is set or checked (sv25, the
    study's), or None for the plant the process runs (the bench server's measured SCS0009, say); the setup, mounts and
    open pose do not depend on it, the plant's servo gains enter the Mirror's gravity term and the plant. seed: the
    world's seed (its object-pose draw; the setup does not depend on it). variant: hom_tasks.VARIANTS key,
    hom_ea.VARIANT[task] when None (the study's: base on the sphere, hold on the cylinder). Returns the dict of the
    module docstring."""
    import mujoco
    HG, R, E, FP = modules(plant)
    T, H, S = HG.T, HG.H, HG.S
    if task not in R.TASKS:
        raise ValueError(f"task {task}: the GPU studies run {sorted(R.TASKS)}")
    d_mm = float(R.TASKS[task])
    shape = T.SHAPE[task]
    d6 = np.asarray(d6, float).reshape(6)
    v = np.asarray(v, float)
    prm = {k: float(x) for k, x in (prm or {}).items()}
    bad = set(prm) - set(T.PRM)
    if bad:
        raise ValueError(f"prm keys {sorted(bad)} are not hom_tasks.PRM keys")
    FP.set_sizes(**({"sph_d_mm": d_mm} if shape == "sphere" else {"cyl_d_mm": d_mm}))
    setup = R.setup_of(HG, task, d6, v)
    scene = _scene(HG, shape, d_mm)
    bm, q_open = scene.prepare(HG.mount_xy(d6), setup)
    var_name = variant or E.VARIANT[task]
    var = HG.variant_world(var_name, seed)
    over, _noise = T.VARIANTS[var_name]
    dt = var["dt"] if dt is None else float(dt)
    hom_prm = replace(replace(H.HomParams(dt=T.DT_C), **over), dt=dt)
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for f in HG.FINGERS:
        m_pt.body_pos[m_pt.body(f"{f}_mount").id] = bm[f]
    S.set_palm(m_pt, setup["palm"])
    p_pads = bake_mounts(scene.pads_path, bm, d6)
    world = {"key": {"d": d6.tolist(), "v": v.tolist()}, "seed": seed, "setup": setup, "body_mounts": bm,
             "q_open": q_open, "prm": prm, "var": var}
    return {"task": task, "d_mm": d_mm, "setup": setup, "m_pt": m_pt, "q_open": q_open, "dt": dt, "prm": hom_prm,
            "p_pads": p_pads, "task_prm": {**T.PRM, **prm}, "hold_s": float(var["hold"]), "body_mounts": bm,
            "world": world, "plant": plant, "plant_spec": S.PLANT, "variant": var_name,
            "scene_pads": Path(scene.pads_path), "scene_pt": Path(scene.pt_path)}


def make_bench(w, dt=None, **kw):
    """A HomBench on world w that runs the world's schedule parameters and plant hold (HomBench takes task_prm and hold_s
    since 2026-11-15 14:15). Until 16:22 this function wrapped HomBench to pass them, and from 14:15 simulate() set the
    plant's hold from the bench's own hold_s, 0, over the wrapper's: runs through it in that window had no hold."""
    from sr2hand.control import hom_bench as HB
    b = HB.HomBench(w["task"], w["d_mm"], w["setup"], w["m_pt"], w["q_open"], w["dt"] if dt is None else dt,
                    prm=w["prm"], p_pads=w["p_pads"], task_prm=w["task_prm"], hold_s=w["hold_s"], **kw)
    b.reset()
    return b


# ------------------------------------------------------------------------------------------------ checks

def _contact_points(w, q):
    """Per finger: the point of the object nearest the tip geom at joint angles q (world, m) and their distance (m), on
    the pt model with the object at its start pose."""
    import mujoco
    HG, _R, _E, _FP = modules(w["plant"])
    S = HG.S
    m = w["m_pt"]
    d = mujoco.MjData(m)
    tb = m.body(S.OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    d.qpos[qa:qa + 7] = [0.0, 0.0, S.Z_OBJ, *w["setup"]["quat"]]
    for f in HG.FINGERS:
        for k, j in enumerate(HG.JOINTS):
            d.qpos[m.jnt_qposadr[m.joint(f"{f}_{j}").id]] = q[f][k]
    mujoco.mj_forward(m, d)
    og = int(np.flatnonzero(m.geom_bodyid == tb)[0])
    out = {}
    for f in HG.FINGERS:
        ft = np.zeros(6)
        dist = mujoco.mj_geomDistance(m, d, m.geom(f"{f}_tipgeom").id, og, 0.05, ft)
        out[f] = (ft[3:], float(dist))
    return out


def check_world(w, deep=True):
    """The bench's objects against the GPU world, as maxima (m, rad or N) and flags:
      mounts     setup's mounts against d6; the world's mount bodies against the scene's base + d6; m_pt's and the
                 baked plant's mount bodies against the world's
      plant      deep: every other field of the baked plant against the reference plant with its mounts moved (the
                 CPU replicas' way), and 50 ms of stepping both from the open pose
      open_pose  a second hom_gpu.Scene.prepare against q_open
      contacts   at the touch pose of hom_tasks.open_pose, each tip's gap to the object (pt model) and the distance of
                 its nearest object point from the grasp's contact point (fp_sr2_toy.geometry pc, palm frame + palm)
      schedule   the forces and Eq. (2) weights of HomBench's schedule (hom_tasks.Schedule with task_prm, task phase)
                 against hom_gpu.finger_commands with the world's prm
      homparams  HomParams against hom.HomParams() and the variant's contact_force_min, force_loop_gain, reanchor
      hold       hold_s against the variant's hold"""
    import mujoco
    HG, _R, _E, FP = modules(w["plant"])
    T, H, S = HG.T, HG.H, HG.S
    task, setup, bm, world = w["task"], w["setup"], w["body_mounts"], w["world"]
    d6 = np.asarray(world["key"]["d"], float)
    mxy = HG.mount_xy(d6)
    scene = _scene(HG, T.SHAPE[task], w["d_mm"])
    out = {}
    e_set = max(float(np.abs(1e3 * np.asarray(setup["mounts"][f])[:2] - 1e3 * mxy[f]).max()) for f in HG.FINGERS)
    e_base = max(float(np.abs(bm[f] - scene.base[f] - np.array([*mxy[f], 0.0])).max()) for f in HG.FINGERS)
    m_pt = w["m_pt"]
    e_pt = max(float(np.abs(m_pt.body_pos[m_pt.body(f"{f}_mount").id] - bm[f]).max()) for f in HG.FINGERS)
    e_palm = float(np.abs(m_pt.body_pos[m_pt.body("palm_pose").id] - setup["palm"]).max())
    mb = mujoco.MjModel.from_xml_path(str(w["p_pads"]))
    e_pads = max(float(np.abs(mb.body_pos[mb.body(f"{f}_mount").id] - bm[f]).max()) for f in HG.FINGERS)
    out["mounts"] = {"setup_vs_d6_mm": e_set, "bodies_vs_base_m": e_base, "m_pt_vs_world_m": e_pt,
                     "m_pt_palm_vs_setup_m": e_palm, "baked_vs_world_m": e_pads,
                     "ok": max(e_set, 1e3 * e_base, 1e3 * e_pt, 1e3 * e_palm, 1e3 * e_pads) < 1e-9}
    if deep:
        mr = mujoco.MjModel.from_xml_path(str(w["scene_pads"]))
        for f in HG.FINGERS:
            mr.body_pos[mr.body(f"{f}_mount").id] = bm[f]
        names = ("body_pos", "body_quat", "body_mass", "body_inertia", "geom_type", "geom_size", "geom_pos",
                 "geom_quat", "geom_friction", "geom_solref", "geom_solimp", "geom_margin", "geom_gap", "geom_contype",
                 "geom_conaffinity", "jnt_range", "dof_damping", "dof_armature", "dof_frictionloss",
                 "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange")
        diff = {n: float(np.abs(np.asarray(getattr(mb, n), float) - np.asarray(getattr(mr, n), float)).max())
                for n in names if getattr(mb, n).shape == getattr(mr, n).shape}
        same_shape = all(getattr(mb, n).shape == getattr(mr, n).shape for n in names)
        opt = all(getattr(mb.opt, k) == getattr(mr.opt, k) for k in ("timestep", "impratio", "cone", "integrator"))
        S.set_palm(mb, setup["palm"])
        S.set_palm(mr, setup["palm"])
        qs = []
        for mm in (mb, mr):
            dd = mujoco.MjData(mm)
            tb = mm.body(S.OBJ).id
            qa = mm.jnt_qposadr[mm.body_jntadr[tb]]
            dd.qpos[qa:qa + 7] = [0.0, 0.0, S.Z_OBJ, *setup["quat"]]
            for f in HG.FINGERS:
                for k, j in enumerate(HG.JOINTS):
                    dd.qpos[mm.jnt_qposadr[mm.joint(f"{f}_{j}").id]] = w["q_open"][f][k]
                    dd.ctrl[mm.actuator(f"a_{f}_{j}").id] = w["q_open"][f][k]
            for _ in range(50):
                mujoco.mj_step(mm, dd)
            qs.append(dd.qpos.copy())
        out["plant"] = {"field_max_diff": max(diff.values()), "fields_same_shape": same_shape, "option_same": opt,
                        "nbody": int(mb.nbody), "ngeom": int(mb.ngeom), "qpos_50ms_max_diff": float(np.abs(qs[0] - qs[1]).max()),
                        "ok": same_shape and opt and max(diff.values()) == 0.0 and float(np.abs(qs[0] - qs[1]).max()) == 0.0}
    sc2 = HG.Scene(T.SHAPE[task], w["d_mm"])
    _bm2, q2 = sc2.prepare(mxy, setup)
    e_q = max(float(np.abs(np.asarray(q2[f]) - np.asarray(w["q_open"][f])).max()) for f in HG.FINGERS)
    out["open_pose"] = {"max_diff_rad": e_q, "ok": e_q == 0.0}
    _qo, q_touch = T.open_pose(setup, m_pt)
    fp = "tripod" if T.SHAPE[task] == "sphere" else "pinch"
    _obj, _tg, _n, pc, _ax = FP.geometry(fp, np.asarray(world["key"]["v"], float), FP.mounts6(d6))
    cp = _contact_points(w, q_touch)
    world_pc = {f: np.asarray(setup["palm"]) + np.asarray(pc[f]) for f in HG.FINGERS}
    out["contacts"] = {"gap_mm": {f: round(1e3 * cp[f][1], 4) for f in HG.FINGERS},
                       "from_grasp_point_mm": {f: round(1e3 * float(np.linalg.norm(cp[f][0] - world_pc[f])), 3)
                                               for f in HG.FINGERS}}
    out["contacts"]["ok"] = (max(abs(x) for x in out["contacts"]["gap_mm"].values()) < 0.1
                             and max(out["contacts"]["from_grasp_point_mm"].values()) < 3.0)
    sch = T.Schedule(task, setup, w["task_prm"])
    cmd, _info = sch(T.T_LIFT + 0.5, {"quat": np.asarray(setup["quat"], float)})
    wt, _fo, _pu, fg = HG.finger_commands(task, setup["n_out"], world["prm"])
    e_w = max(float(np.abs(np.asarray(cmd.fingers[f].weights) - wt[k]).max()) for k, f in enumerate(HG.FINGERS))
    e_f = max(abs(float(cmd.fingers[f].force) - float(fg[k])) for k, f in enumerate(HG.FINGERS))
    out["schedule"] = {"weights_max_diff": e_w, "force_max_diff_N": e_f, "ok": e_w < 1e-12 and e_f < 1e-12,
                       "forces_N": {f: round(float(fg[k]), 4) for k, f in enumerate(HG.FINGERS)}}
    hp, var = w["prm"], world["var"]
    ref = replace(H.HomParams(), dt=w["dt"])
    others = [f_.name for f_ in fields(H.HomParams) if f_.name not in ("dt", "filter")
              and getattr(hp, f_.name) != getattr(ref, f_.name)]
    out["homparams"] = {"dt": hp.dt, "differ_from_defaults": others,
                        "ok": (not others and hp.contact_force_min == var["cfmin"] and hp.force_loop_gain == var["flg"]
                               and float(hp.reanchor) == var["reanchor"])}
    out["hold"] = {"hold_s": w["hold_s"], "ok": w["hold_s"] == var["hold"]}
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict) and "ok" in v)
    return out


# ------------------------------------------------------------------------------------------------ the study's picks

def study_pick(task, xy, pick="S3", plant="sv25"):
    """One layout's pick of the task-conflict study (hom_grasp_refine's refinement on the ten shared layouts): d6, the
    grasp v, HOM's forces and modes prm, the variant, and its scores (hom_landscape.rollout_score) at the validation
    seeds 20-24 and, for S2 and S3, the search's mean at seeds 0-2 (hom_grasp_refine.picks, which must give the same v
    and prm). xy: (x_sep, y_sep) in mm or the lattice call."""
    import hom_landscape as L
    HG, R, E, FP = modules(plant)
    d_mm = R.TASKS[task]
    rows = [json.loads(x) for x in (STUDY_ROWS / plant / "val" / f"{task}_d{d_mm:g}.jsonl").read_text().splitlines()
            if x.strip()]
    lays = json.loads((STUDY_ROWS / plant / f"{task}_layouts.json").read_text())["layouts"]
    lay = next(L_ for L_ in lays if (int(L_["call"]) == xy if isinstance(xy, (int, np.integer))
                                     else tuple(map(float, L_["xy"])) == tuple(map(float, xy))))
    mine = [r for r in rows if int(r["call"]) == int(lay["call"]) and r["pick"] == pick]
    if not mine:
        raise LookupError(f"{task}: no {pick} rows for call {lay['call']}")
    r0 = mine[0]
    for r in mine[1:]:
        if r["v"] != r0["v"] or (r.get("prm") or {}) != (r0.get("prm") or {}) or r["d"] != r0["d"]:
            raise RuntimeError(f"{task} call {lay['call']} {pick}: the validation rows differ in v, prm or d")
    if [float(x) for x in r0["d"]] != [float(x) for x in lay["d"]]:
        raise RuntimeError(f"{task} call {lay['call']}: rows d {r0['d']} against the sample's {lay['d']}")
    out = {"task": task, "d_mm": float(d_mm), "call": int(lay["call"]), "xy": [float(x) for x in lay["xy"]],
           "d6": [float(x) for x in r0["d"]], "v": [float(x) for x in r0["v"]],
           "prm": {k: float(x) for k, x in (r0.get("prm") or {}).items()}, "pick": pick, "variant": r0["variant"],
           "plant_spec": r0["plant"], "scenes": r0.get("scenes"), "hand": r0.get("hand"),
           "val_seeds": [int(r["seed"]) for r in mine],
           "val_scores": [round(float(L.rollout_score(task, r)), 4) for r in mine],
           "val_mean": round(float(np.mean([L.rollout_score(task, r) for r in mine])), 4),
           "rows": str((STUDY_ROWS / plant / "val" / f"{task}_d{d_mm:g}.jsonl").relative_to(ROOT))}
    if LEVELS[pick]:
        old = R.ROWS_ROOT
        R.ROWS_ROOT = STUDY_ROWS
        try:
            p = R.picks(task, plant, LEVELS[pick])[int(lay["call"])]
        finally:
            R.ROWS_ROOT = old
        if [float(x) for x in p["v"]] != out["v"] or {k: float(x) for k, x in p["prm"].items()} != out["prm"]:
            raise RuntimeError(f"{task} call {lay['call']}: the {pick} search's pick differs from its validation rows")
        out.update(search_mean=round(float(p["score"]), 4), search_it=int(p["it"]), search_cand=int(p["cand"]))
    else:
        if [float(x) for x in lay["v1"]] != out["v"]:
            raise RuntimeError(f"{task} call {lay['call']}: S1's grasp differs from its validation rows")
        out["search_mean"] = float(lay["s1"])
    return out


def world_from_plan(plan, dt=None, check_stored=True, plant="plan"):
    """bench_world of a plan written by scripts/hardware/hom_bench_plans.py (path or dict): its meta.hom gives the
    task, layout, grasp, prm and variant; plant "plan" takes the plan's (sv25), None the process's (bench_world). The
    plan's mounts_palm_mm (the driver's coordinates) must map to the layout in P (sr2hand.robot.sr2_layouts.
    P_from_driver) within 1e-6 mm, and with check_stored the rebuilt setup, body mounts and open pose must equal the
    stored ones within 1e-9 (m, rad); RuntimeError otherwise."""
    from sr2hand.robot import sr2_layouts as SL
    raw = json.loads(Path(plan).read_text()) if not isinstance(plan, dict) else plan
    h = raw["meta"]["hom"]
    d6 = np.asarray(h["d6_P_mm"], float)
    for f, xy in SL.mounts(d6).items():
        got = SL.P_from_driver(f, raw["mounts_palm_mm"][f])
        if np.abs(got - xy).max() > 1e-6:
            raise RuntimeError(f"{f}: the plan's driver mount {raw['mounts_palm_mm'][f]} is {got.tolist()} mm in P, the "
                               f"layout's is {xy.tolist()}")
    w = bench_world(h["task"], d6, h["v"], h["prm"], dt=dt, plant=h["plant"]["name"] if plant == "plan" else plant,
                    variant=h["variant"])
    if check_stored:
        st = h["setup"]
        diffs = {"palm": np.abs(np.asarray(st["palm_m"]) - w["setup"]["palm"]).max(),
                 "quat": np.abs(np.asarray(st["quat"]) - w["setup"]["quat"]).max(),
                 "targets": max(np.abs(np.asarray(st["targets_m"][f]) - w["setup"]["targets"][f]).max()
                                for f in w["body_mounts"]),
                 "n_out": max(np.abs(np.asarray(st["n_out"][f]) - w["setup"]["n_out"][f]).max() for f in w["body_mounts"]),
                 "body_mounts": max(np.abs(np.asarray(h["body_mounts_m"][f]) - w["body_mounts"][f]).max()
                                    for f in w["body_mounts"]),
                 "q_open": max(np.abs(np.asarray(h["q_open_rad"][f]) - w["q_open"][f]).max() for f in w["body_mounts"])}
        bad = {k: float(v) for k, v in diffs.items() if v > 1e-9}
        if bad:
            raise RuntimeError(f"the rebuilt world differs from the plan's stored numbers: {bad}")
    return w
