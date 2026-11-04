#!/usr/bin/env python3
"""Refinement of each layout's grasp, and of HOM's commanded forces and contact modes, on the HOM task score (owner,
2026-11-03: "can't we score and refine a grasp based off how well the grasp (+morphology) executes the HOM control?
from what i understand we are doing pretty random sampling"; "is it possible to co-design HOM? are there
hyperparameters / parameters that we can change within it?"; then "let's send it on these tasks ... we can try out
multiple scores").

Three scores of one layout (a symmetric tripod of the 5 mm lattice of fp_sr2_toy.grid2d), each the task score of
hom_landscape.rollout_score averaged over seeds:
  S1  default HOM and the best grasp of the layout's grasp set (up to 16: the foundational pose and a random walk over
      the feasible set, fp_sr2_toy.grasp_set), the score of the grids (rows grid2dff_v4b<tag>);
  S2  default HOM and the grasp refined by the cross-entropy method (CEM) on the score, from S1's grasp (level grasp);
  S3  the grasp, the commanded normal forces and the contact modes refined together by CEM, from S2's grasp and HOM's
      defaults (level full). The modes are Eq. (2)'s row weights by finger role (hom_tasks.mode_weights): spin, roll
      and slide of the tripod's fingers, or of the pinching fingers and the pusher; a weight w = u^2, u in [0, 1].
      The forces are searched in log space: the tripod's per pad 1-5 N; the pinch 1.5-6 N, the pin-push drive
      0.5-3 N and the press 0.2-2 N.
A candidate grasp must pass the pose toy's constraints (fp_sr2_toy.Problem.slacks: reach inside the servo limits less
3 deg, the distal link facing the object, pad coverage, the plant's finger spheres 2 mm apart at the touch pose, the
pinch's 40 mm lever; then fp_sr2_toy.clearances and clear_ok); an infeasible draw is drawn again. Every candidate runs
at seeds 0-2 on the GPU evaluator of the grids (hom_gpu.batch_child), so the candidates of one iteration share their
object-pose draws, as S1's grasps did.

Layouts per task: the 30 best by S1 and 30 drawn at random (numpy seed 0) from the other layouts whose pose is reached.
CEM per layout: 24 candidates (grasp) or 32 (full) per iteration, the best quarter by mean score (ties by the fingers
in contact at the end) as elites, mean and spread moved 0.8 of the way to the elites', a floor on the spread, 8
iterations; the pick is the best candidate seen. From 2026-11-04 the start itself is candidate 0 of iteration 0, so
the pick never scores below the start at the search seeds. The starting spread is 0.25 (grasp) or 0.15 (full) of the
feasible width of each grasp variable that fp_sr2_toy.grasp_set measured around the foundational pose, and 0.3 for the log
forces and the mode variables u. `validate` runs each layout's three picks at seeds 20-24, which no search or grid
used; those are the scores to compare. The night's searches of 2026-11-03 scored draws around each start and never the
start itself, so `report` also gives the elitist picks (elitist()): S2e, the better of S1's grasp and S2's pick at
seeds 0-2, and S3e, the best of S1, S2 and S3.

    python3 scripts/hom_grasp_refine.py run --task twist --level grasp [--plant sv25] [--top 30 --rand 30] [--iters 8]
    python3 scripts/hom_grasp_refine.py run --task twist --level full
    python3 scripts/hom_grasp_refine.py run --task twist --level grasp --plant k21mu0p4 --layouts-from k21
    python3 scripts/hom_grasp_refine.py validate --task twist [--picks-from k21]   # another plant's picks run here
    python3 scripts/hom_grasp_refine.py report                # rows/<plant>/<date>-score_levels.json
    python3 scripts/hom_grasp_refine.py run ... --smoke       # rows under rows_smoke/
Plants (PLANTS): sv25 the SCS0009 with pads at 2.5 MPa (the hand), k21 the STS3032 at kp 2.1 N m/rad, k21mu0p4 and
k21mu0p7 the same at scene friction 0.4 (TPU on bare PLA) and 0.7, work10 the work plant at 10 MPa; each is a grid of
hom_plant_transfer.py, whose rows give S1. The overnight queue of 2026-11-03 is logs/20261103-hom_score_refine/queue.sh
(README.md there).

Worst-case score over a band of scene frictions (owner, 2026-11-03: "robustness to a band of physical parameters as a
scoring parameter in hand selection"; "worst-case sampling makes sense"). With --band (BANDS: band_k21mu, the STS3032
at friction 0.4, 0.7 and 1.0), every candidate runs at seeds 0-2 on each plant of the band, and its score is the lowest
of its three-seed means; the tie-break is the margin at that plant. S1 is the grasp of the layout's set with the best
worst-case score over the band's grids. The open pose is found once per candidate (hom_tasks.open_pose is kinematic,
and the band's pt scenes differ only in friction); each plant's batches are spawned with that plant's HOS_PLANT,
HOS_SCENES and HOM_ROWS_TAG, which hand_object_scale_sim and hom_gpu read at import, and every row's scene cache is
checked. Rows: rows/<band>/<level>/<plant>/<task>_d<d>.jsonl (and _timing.jsonl) per plant, the band's _gens.jsonl and
_iters.jsonl in rows/<band>/<level>/, the sample rows/<band>/<task>_layouts.json (another plant's layouts,
--layouts-from).
    python3 scripts/hom_grasp_refine.py run --task twist --level grasp --band k21mu0p4,k21mu0p7,k21 --layouts-from k21
    python3 scripts/hom_grasp_refine.py validate --task twist --band k21mu0p4,k21mu0p7,k21 [--picks-from k21]
    python3 scripts/hom_grasp_refine.py report --band k21mu0p4,k21mu0p7,k21   # rows/<band>/<date>-band_levels.json
`validate --band` runs the band's picks, or with --picks-from another plant's, on every plant of the band; rows of the
2026-11-03 queue (a plant's val/ and val_from_<plant>/) are read where they exist and never appended to.
Hom_gpu takes the forces and modes per world (a world's prm); check of 2026-11-03 18:2x (logs/20261103-hom_score_refine/
check_prm.py): default worlds reproduce the grid rows within the GPU's run-to-run spread, a world with prm scores the
same in a mixed batch and alone, and the CPU runner with the same prm agrees.
Rows: docs/experiments/20261103-hom_score_refine/rows/<plant>/<level>/<task>_d<d>.jsonl in hom_gpu.py's format, the
key with call, it, cand, the grasp v, prm and x (the CEM variables); <task>_d<d>_gens.jsonl, one line per layout and
iteration; <task>_d<d>_iters.jsonl (from 2026-11-04), one line per iteration with its draw, open-pose and GPU-batch
seconds, batches and rollouts; rows/<plant>/<task>_layouts.json, the layout sample with S1. A stopped run resumes: the
iterations are replayed from the rows with the same random draws.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "docs/experiments/20261103-hom_score_refine"
GRID_ROWS = ROOT / "docs/experiments/20261029-foundational_pose_review/rows"
LG = "logs/20261102-hom_servo_plant/"
# plant name -> (HOS_PLANT, HOS_SCENES or None, HOM_ROWS_TAG): the grids of hom_plant_transfer.py
PLANTS = {"sv25": ("servo", LG + "scenes", "_sv25"),                       # SCS0009, pads at 2.5 MPa: the hand
          "k21": ("kp2.1_kv0_fr0.515_fl0_dp0.042", LG + "scenes", "_k21"),  # STS3032 at kp 2.1, 2.5 MPa
          "k21mu0p4": ("kp2.1_kv0_fr0.515_fl0_dp0.042", LG + "scenes_mu0.4", "_k21mu0p4"),  # ... friction 0.4
          "k21mu0p7": ("kp2.1_kv0_fr0.515_fl0_dp0.042", LG + "scenes_mu0.7", "_k21mu0p7"),  # ... friction 0.7
          "work10": ("work", None, "")}                                     # the work plant, 10 MPa (10-30 grids)
# band name -> its plants, the nominal plant last (the parent process runs with it; worst-case score, --band)
BANDS = {"band_k21mu": ("k21mu0p4", "k21mu0p7", "k21")}
TASKS = {"twist": 40.0, "roll": 40.0, "pinpush": 25.0, "press": 25.0}
POSE = {"twist": "tripod", "roll": "tripod", "pinpush": "pinch", "press": "press"}
SEARCH_SEEDS, VAL_SEEDS = (0, 1, 2), (20, 21, 22, 23, 24)
N_TOP, N_RAND = 30, 30
POP = {"grasp": 24, "full": 32}
ITERS = 8
ELITE = 0.25
SMOOTH = 0.8
STD0 = {"grasp": 0.25, "full": 0.15}               # x the feasible width of each grasp variable
STD0_PRM, FLOOR_V, FLOOR_PRM = 0.3, 0.02, 0.03
TRIES = 40                                          # draws per candidate before it is left out
GEN_SEED = 100031                                   # per-period noise of the batches (unused by base and hold)
FORCES = {"twist": {"f_tripod": (1.0, 5.0)}, "roll": {"f_tripod": (1.0, 5.0)},
          "pinpush": {"f_pinch": (1.5, 6.0), "f_drive": (0.5, 3.0)},
          "press": {"f_pinch": (1.5, 6.0), "f_press": (0.2, 2.0)}}
ROLES = {"twist": ("tripod",), "roll": ("tripod",), "pinpush": ("pinch", "push"), "press": ("pinch", "push")}
COMPS = ("spin", "roll", "slide")
WORKERS = 4                                         # processes drawing feasible candidates


def setenv(plant):
    """The plant's environment, before hom_gpu imports hand_object_scale_sim."""
    hp, scenes, tag = PLANTS[plant]
    os.environ["HOS_PLANT"] = hp
    if scenes:
        os.environ["HOS_SCENES"] = scenes
    os.environ["HOM_ROWS_TAG"] = tag


ROWS_ROOT = OUT / "rows"                            # --smoke: OUT / "rows_smoke", so a test never joins a run's rows


def rows_dir(plant):
    return ROWS_ROOT / plant


def band_name(arg):
    """The BANDS name of --band: the name itself, or its plants comma-separated in BANDS's order."""
    if arg in BANDS:
        return arg
    want = tuple(x.strip() for x in arg.split(","))
    for k, v in BANDS.items():
        if v == want:
            return k
    raise SystemExit(f"--band {arg}: not a band of {BANDS}")


def use_plant(plant):
    """setenv, and HOS_SCENES cleared for a plant of the default scene cache: a band switches the plant of the GPU
    batches it spawns."""
    setenv(plant)
    if not PLANTS[plant][1]:
        os.environ.pop("HOS_SCENES", None)


def band_paths(path, plants):
    """{plant: rows path} of a band: path's file in a folder per plant beside it."""
    return {p: path.parent / p / path.name for p in plants}


# ------------------------------------------------------------------------------------------------ the search space

class Space:
    """The CEM variables of one task at one level: the toy's grasp variables v (fp_sr2_toy.bounds), then (level full)
    the log forces and the mode variables u (w = u^2)."""

    def __init__(self, task, level, widths):
        import fp_sr2_toy as FP
        import hom_tasks as T
        b = np.array(FP.bounds(POSE[task]), float)
        self.task, self.level, self.nv = task, level, len(b)
        self.forces = list(FORCES[task]) if level == "full" else []
        self.modes = [f"w_{c}_{r}" for r in ROLES[task] for c in COMPS] if level == "full" else []
        fb = np.log(np.array([FORCES[task][k] for k in self.forces], float).reshape(-1, 2))
        self.lo = np.r_[b[:, 0], fb[:, 0], np.zeros(len(self.modes))]
        self.hi = np.r_[b[:, 1], fb[:, 1], np.ones(len(self.modes))]
        self.width = np.maximum(np.asarray(widths, float), 1e-4 * (b[:, 1] - b[:, 0]))
        self.default_prm = np.r_[np.log([T.PRM[k] for k in self.forces]), np.sqrt([T.PRM[k] for k in self.modes])]

    def start(self, v0):
        """CEM's starting mean and spread around the grasp v0 and HOM's defaults."""
        mean = np.r_[np.asarray(v0, float), self.default_prm]
        std = np.r_[STD0[self.level] * self.width, np.full(len(self.default_prm), STD0_PRM)]
        return mean, std

    def floor(self):
        return np.r_[FLOOR_V * self.width, np.full(len(self.default_prm), FLOOR_PRM)]

    def prm(self, x):
        """hom_tasks.PRM changes of the variables x: the forces (N) and the mode weights."""
        out, k = {}, self.nv
        for name in self.forces:
            out[name] = round(float(np.exp(x[k])), 4)
            k += 1
        for name in self.modes:
            out[name] = round(float(np.clip(x[k], 0.0, 1.0) ** 2), 4)
            k += 1
        return out


def feasible(task, d6, v):
    """The pose toy's verdict on grasp v of layout d6 (mm): every constraint slack >= 0 and the clearances pass."""
    import fp_sr2_toy as FP
    fp = POSE[task]
    pr = FP.Problem(fp, d6)
    pr.pick_branches(v)
    if float(pr.slacks(v).min()) < 0.0:
        return False
    return bool(FP.clear_ok(FP.clearances(fp, v, pr.M, pr.branch)))


def draw_layout(job):
    """Candidates of one layout and iteration (a worker's job): n feasible draws around mean with spread std, numpy's
    generator seeded by (task, call, it), so a replayed iteration draws the same candidates. At iteration 0 the first
    candidate is the start itself (the mean, when feasible), so that the search's pick never scores below its start at
    the search seeds (2026-11-04; the searches of 2026-11-03 drew all n). Returns (call, [x or None], draws tried)."""
    import fp_sr2_toy as FP
    task, level, call, it, d6, mean, std, lo, hi, nv, n = job
    FP.set_sizes(**({"sph_d_mm": TASKS[task]} if POSE[task] == "tripod" else {"cyl_d_mm": TASKS[task]}))
    rng = np.random.default_rng([sum(map(ord, task + level)), int(call), int(it)])
    out, tried = [], 0
    for k in range(n):
        x = None
        if k == 0 and int(it) == 0:
            y = np.clip(np.asarray(mean, float), lo, hi)
            tried += 1
            if feasible(task, d6, y[:nv]):
                out.append(y)
                continue
        for t in range(TRIES):
            s = std if t < TRIES // 2 else 0.5 * std
            y = np.clip(mean + s * rng.standard_normal(len(mean)), lo, hi)
            tried += 1
            if feasible(task, d6, y[:nv]):
                x = y
                break
        out.append(x)
    return call, out, tried


# ------------------------------------------------------------------------------------------------ the layouts

def toy_rows(task):
    """{call: the pose toy's grid row} for the task's object (fp_sr2_toy.grid2d rows grid2dff_<pose><d>_v4b)."""
    import fp_sr2_toy as FP
    shape = "sphere" if POSE[task] == "tripod" else "cylinder"
    p = GRID_ROWS / f"{FP.size_name('grid2dff', shape, TASKS[task])}.jsonl"
    out = {}
    for x in p.read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            rec = r["fp"].get(POSE[task]) or {}
            if rec.get("ok") and rec.get("grasps"):
                out[int(r["call"])] = r
    return out


def s1_table(task, plant):
    """{call: (best grasp index, its mean score over seeds 0-2)} from the plant's grid rows."""
    import hom_landscape as L
    tag = PLANTS[plant][2]
    p = GRID_ROWS / f"grid2dff_v4b{tag}" / f"{task}_d{TASKS[task]:g}.jsonl"
    sc = {}
    for x in p.read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            if int(r["seed"]) in SEARCH_SEEDS:
                sc.setdefault(int(r["call"]), {}).setdefault(int(r.get("grasp", 0)), []).append(L.rollout_score(task, r))
    out = {}
    for c, g in sc.items():
        means = {j: float(np.mean(v)) for j, v in g.items() if len(v) >= len(SEARCH_SEEDS)}
        if means:
            j = max(sorted(means), key=lambda k: means[k])
            out[c] = (j, means[j])
    return out


def layouts(task, plant, n_top=N_TOP, n_rand=N_RAND, base=None):
    """The layout sample of a task (written once, then read): the n_top best by S1 and n_rand others at random; with
    base (another plant), base's layouts with S1 and its grasp taken from this plant's grid."""
    p = rows_dir(plant) / f"{task}_layouts.json"
    if p.exists():
        return json.loads(p.read_text())["layouts"]
    s1, toy = s1_table(task, plant), toy_rows(task)
    ranked = sorted((c for c in s1 if c in toy), key=lambda c: (-s1[c][1], c))
    if base:
        b = json.loads((rows_dir(base) / f"{task}_layouts.json").read_text())["layouts"]
        chosen = [(L_["call"], L_["group"]) for L_ in b if L_["call"] in s1]
    else:
        top, rest = ranked[:n_top], ranked[n_top:]
        rand = sorted(int(c) for c in np.random.default_rng(0).choice(rest, size=min(n_rand, len(rest)), replace=False))
        chosen = [(c, "top") for c in top] + [(c, "random") for c in rand]
    out = []
    for c, group in chosen:
        rec = toy[c]["fp"][POSE[task]]
        out.append({"call": c, "xy": toy[c]["xy"], "d": toy[c]["d"], "group": group,
                    "s1": round(s1[c][1], 4), "s1_grasp": s1[c][0], "v1": rec["grasps"][s1[c][0]]["v"],
                    "widths": rec["grasps"][0]["widths"]})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"task": task, "plant": plant, "n_reached": len(ranked), "base": base, "layouts": out},
                            indent=1))
    return out


def s1_band(task, plants):
    """{call: (grasp index, its lowest three-seed mean over the plants' grids, {plant: mean})}: the grasp of the layout's
    set with the best worst-case score (ties: the lowest index, as s1_table)."""
    import hom_landscape as L
    per = {}
    for p in plants:
        sc = {}
        for r in read_rows(GRID_ROWS / f"grid2dff_v4b{PLANTS[p][2]}" / f"{task}_d{TASKS[task]:g}.jsonl"):
            if int(r["seed"]) in SEARCH_SEEDS:
                sc.setdefault((int(r["call"]), int(r.get("grasp", 0))), []).append(L.rollout_score(task, r))
        per[p] = {k: float(np.mean(v)) for k, v in sc.items() if len(v) >= len(SEARCH_SEEDS)}
    keys = set.intersection(*(set(s) for s in per.values()))
    by_call = {}
    for c, j in keys:
        by_call.setdefault(c, []).append(j)
    out = {}
    for c, js in by_call.items():
        w = {j: min(per[p][(c, j)] for p in plants) for j in sorted(js)}
        j = max(w, key=lambda k: w[k])
        out[c] = (j, w[j], {p: round(per[p][(c, j)], 4) for p in plants})
    return out


def band_layouts(task, name, base):
    """A band's layout sample (written once, then read): base's sample (from the study's rows, also under --smoke), each
    layout with the grasp that S1 picks by the worst-case score (s1_band)."""
    p = rows_dir(name) / f"{task}_layouts.json"
    if p.exists():
        return json.loads(p.read_text())["layouts"]
    if not base:
        raise SystemExit(f"{name}: the band's layout sample is another plant's; give --layouts-from")
    s1, toy = s1_band(task, BANDS[name]), toy_rows(task)
    out = []
    for L_ in json.loads((OUT / "rows" / base / f"{task}_layouts.json").read_text())["layouts"]:
        c = L_["call"]
        if c not in s1:
            print(f"{task} call {c}: not on every grid of {name}; left out", flush=True)
            continue
        j, w, per = s1[c]
        rec = toy[c]["fp"][POSE[task]]
        out.append({"call": c, "xy": toy[c]["xy"], "d": toy[c]["d"], "group": L_["group"], "s1": round(w, 4),
                    "s1_plants": per, "s1_grasp": j, "v1": rec["grasps"][j]["v"], "widths": rec["grasps"][0]["widths"],
                    "base_s1_grasp": L_["s1_grasp"]})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"task": task, "band": name, "plants": BANDS[name], "base": base, "layouts": out},
                            indent=1))
    return out


# ------------------------------------------------------------------------------------------------ evaluation

def setup_of(HG, task, d6, v):
    """hom_tasks' setup dict of grasp v on layout d6 (mm), as hom_gpu.size_layouts builds it."""
    import fp_sr2_toy as FP
    if POSE[task] == "tripod":
        obj, _tg, n, _pc, _ax = FP.geometry("tripod", np.asarray(v, float), FP.mounts6(d6))
        return HG.tripod_setup(task, HG.mount_xy(d6), (np.asarray(obj["centre"], float),
                                                       {f: np.asarray(n[f], float) for f in HG.FINGERS}), TASKS[task])
    return HG.pinch_setup(task, v, d6)


def read_rows(path):
    return [json.loads(x) for x in (path.read_text().splitlines() if path.exists() else []) if x.strip()]


def evaluate(task, plant, path, items, seeds, scene, HG, E, timing=None):
    """Run every (key, d6, v, prm) of items at each seed not yet in the rows of path, as GPU batches in spawned
    processes. key must identify the candidate: rows are matched on (call, it, cand) or (call, pick). timing (a dict)
    receives the open poses' and the GPU batches' wall seconds and the number of batches."""
    import multiprocessing as mp
    t0 = time.time()
    if timing is not None:
        timing.update(open_s=0.0, gpu_s=0.0, batches=0)
    done = set()
    for r in read_rows(path):
        done.add((ident(r), int(r["seed"])))
    var = E.VARIANT[task]
    worlds = []
    for key, d6, v, prm in items:
        todo = [sd for sd in seeds if (ident(key), sd) not in done]
        if not todo:
            continue
        setup = setup_of(HG, task, d6, v)
        try:
            bm, q_open = scene.prepare(HG.mount_xy(d6), setup)
        except Exception as e:                          # the open pose not found: the candidate is left out
            print(f"{task} call {key['call']}: open pose failed ({type(e).__name__}: {e}); left out", flush=True)
            continue
        worlds += [{"key": key, "seed": sd, "setup": setup, "body_mounts": bm, "q_open": q_open, "prm": prm,
                    "var": HG.variant_world(var, sd)} for sd in todo]
    t_open = time.time() - t0
    if timing is not None:
        timing["open_s"] = round(t_open, 3)
    if not worlds:
        return 0
    ctx = mp.get_context("spawn")
    tpath = path.with_name(path.stem + "_timing.jsonl")
    b = E.BATCH[task]
    n0 = len(read_rows(tpath))
    t1 = time.time()
    for i, b0 in enumerate(range(0, len(worlds), b)):
        spec = {"shape": "sphere" if POSE[task] == "tripod" else "cylinder", "d_mm": TASKS[task], "task": task,
                "worlds": worlds[b0:b0 + b], "dt": worlds[0]["var"]["dt"], "gen_seed": GEN_SEED + b0,
                "batch": n0 + i, "path": str(path), "tpath": str(tpath), "layouts": "grid2dff_size"}
        pr = ctx.Process(target=HG.batch_child, args=(spec,))
        pr.start()
        pr.join()
        if pr.exitcode != 0:
            raise SystemExit(f"{task} {plant}: batch {n0 + i} failed (exit {pr.exitcode})")
    if timing is not None:
        timing.update(gpu_s=round(time.time() - t1, 2), batches=-(-len(worlds) // b))
    return len(worlds)


def ident(r):
    if "pick" in r:
        return int(r["call"]), str(r["pick"])
    return int(r["call"]), int(r["it"]), int(r["cand"])


def margin(task, r):
    """The tie-break of equal scores: the twist's three-pad ramp angle (deg, uncapped; its score saturates at 90 deg),
    else the fingers in contact at the end."""
    if task == "twist":
        return float(r.get("ramp_three_pads_max_deg") or 0.0) if r.get("held") else 0.0
    return float(r.get("fingers_end", 0))


def scores(task, rows):
    """{ident: (mean score, mean margin, seeds)} over the rows' seeds."""
    import hom_landscape as L
    acc = {}
    for r in rows:
        a = acc.setdefault(ident(r), [[], []])
        a[0].append(L.rollout_score(task, r))
        a[1].append(margin(task, r))
    return {k: (float(np.mean(s)), float(np.mean(f)), len(s)) for k, (s, f) in acc.items()}


def evaluate_band(task, plants, paths, items, seeds, scene, HG, E, timing=None):
    """evaluate() on several plants: each candidate's open pose once, then each plant's GPU batches, spawned with that
    plant's environment (use_plant), into paths[plant]. A row whose scene cache is not its plant's stops the run.
    timing receives the open poses' seconds and each plant's GPU seconds and batches."""
    import multiprocessing as mp
    t0 = time.time()
    done = {p: {(ident(r), int(r["seed"])) for r in read_rows(paths[p])} for p in plants}
    var = E.VARIANT[task]
    worlds = {p: [] for p in plants}
    for key, d6, v, prm in items:
        todo = {p: [sd for sd in seeds if (ident(key), sd) not in done[p]] for p in plants}
        if not any(todo.values()):
            continue
        setup = setup_of(HG, task, d6, v)
        try:
            bm, q_open = scene.prepare(HG.mount_xy(d6), setup)
        except Exception as e:                          # the open pose not found: the candidate is left out
            print(f"{task} call {key['call']}: open pose failed ({type(e).__name__}: {e}); left out", flush=True)
            continue
        for p in plants:
            worlds[p] += [{"key": key, "seed": sd, "setup": setup, "body_mounts": bm, "q_open": q_open, "prm": prm,
                           "var": HG.variant_world(var, sd)} for sd in todo[p]]
    if timing is not None:
        timing.update(open_s=round(time.time() - t0, 3), gpu_s=0.0, batches=0, gpu_s_plants={}, batches_plants={})
    ctx = mp.get_context("spawn")
    b = E.BATCH[task]
    home = {k: os.environ.get(k) for k in ("HOS_PLANT", "HOS_SCENES", "HOM_ROWS_TAG")}
    for p in plants:
        if not worlds[p]:
            continue
        path = paths[p]
        path.parent.mkdir(parents=True, exist_ok=True)
        tpath = path.with_name(path.stem + "_timing.jsonl")
        n0, nr0 = len(read_rows(tpath)), len(read_rows(path))
        use_plant(p)                                    # the spawned batch reads it at import
        t1 = time.time()
        for i, b0 in enumerate(range(0, len(worlds[p]), b)):
            spec = {"shape": "sphere" if POSE[task] == "tripod" else "cylinder", "d_mm": TASKS[task], "task": task,
                    "worlds": worlds[p][b0:b0 + b], "dt": worlds[p][0]["var"]["dt"], "gen_seed": GEN_SEED + b0,
                    "batch": n0 + i, "path": str(path), "tpath": str(tpath), "layouts": "grid2dff_size"}
            pr = ctx.Process(target=HG.batch_child, args=(spec,))
            pr.start()
            pr.join()
            if pr.exitcode != 0:
                raise SystemExit(f"{task} {p}: batch {n0 + i} failed (exit {pr.exitcode})")
        for k, v in home.items():                   # back to the parent's plant
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        want = os.path.normpath(PLANTS[p][1]) if PLANTS[p][1] else None
        bad = [r for r in read_rows(path)[nr0:] if want and os.path.normpath(r.get("scenes", "")) != want]
        if bad:
            raise SystemExit(f"{task} {p}: {len(bad)} rows ran on scene cache {bad[0].get('scenes')}, not {want}")
        if timing is not None:
            timing["gpu_s_plants"][p] = round(time.time() - t1, 2)
            timing["batches_plants"][p] = -(-len(worlds[p]) // b)
    if timing is not None:
        timing.update(gpu_s=round(sum(timing["gpu_s_plants"].values()), 2),
                      batches=sum(timing["batches_plants"].values()))
    return sum(len(w) for w in worlds.values())


def band_scores(task, paths, it=None):
    """{ident: (lowest mean score over the plants, the margin at that plant, the fewest seeds, {plant: mean score})} for
    the candidates run on every plant of paths; it: one search iteration's rows only."""
    per = {p: scores(task, [r for r in read_rows(q) if it is None or int(r["it"]) == it]) for p, q in paths.items()}
    out = {}
    for k in set.intersection(*(set(s) for s in per.values())):
        w = min(per, key=lambda p: per[p][k][:2])
        out[k] = (per[w][k][0], per[w][k][1], min(s[k][2] for s in per.values()),
                  {p: round(s[k][0], 4) for p, s in per.items()})
    return out


# ------------------------------------------------------------------------------------------------ the search

def cmd_run(a):
    name = band_name(a.band) if a.band else a.plant         # --band: the worst-case score over the band's plants
    band = BANDS.get(name)
    setenv(band[-1] if band else a.plant)
    import multiprocessing as mp
    import fp_sr2_toy as FP
    import hom_ea as E
    import hom_gpu as HG
    task, level = a.task, a.level
    FP.set_sizes(**({"sph_d_mm": TASKS[task]} if POSE[task] == "tripod" else {"cyl_d_mm": TASKS[task]}))
    lays = band_layouts(task, name, a.layouts_from) if band else layouts(task, a.plant, a.top, a.rand, a.layouts_from)
    lays = lays[:a.n] if a.n else lays
    d6 = {L_["call"]: L_["d"] for L_ in lays}
    path = rows_dir(name) / level / f"{task}_d{TASKS[task]:g}.jsonl"   # a band's rows: band_paths(path)
    paths = band_paths(path, band) if band else {a.plant: path}
    gpath = path.with_name(path.stem + "_gens.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    space = {c: Space(task, level, L_["widths"]) for c, L_ in ((L_["call"], L_) for L_ in lays)}
    if level == "grasp":
        start = {L_["call"]: L_["v1"] for L_ in lays}
    else:                                   # from S2's pick
        prev = band_picks(task, name, "grasp") if band else picks(task, a.plant, "grasp")
        if not prev:
            raise SystemExit("level full starts from level grasp's picks: run level grasp first")
        missing = [c for c in d6 if c not in prev]
        if missing:
            print(f"no level-grasp pick for calls {missing}: these start from S1's grasp", flush=True)
        start = {L_["call"]: (prev[L_["call"]]["v"] if L_["call"] in prev else L_["v1"]) for L_ in lays}
    state = {c: space[c].start(start[c]) for c in d6}
    best = {c: None for c in d6}
    gens_done = {(int(g["call"]), int(g["it"])) for g in read_rows(gpath)}
    scene = HG.Scene("sphere" if POSE[task] == "tripod" else "cylinder", TASKS[task])
    ctx = mp.get_context("spawn")
    iters = a.iters or ITERS
    with ctx.Pool(WORKERS) as pool:
        for it in range(iters):
            t0 = time.time()
            jobs = [(task, level, c, it, d6[c], state[c][0], state[c][1], space[c].lo, space[c].hi, space[c].nv,
                     a.pop or POP[level]) for c in d6]
            drawn = {c: (xs, tried) for c, xs, tried in pool.map(draw_layout, jobs)}
            t_draw = time.time() - t0
            stored = {ident(r): r["x"] for q in paths.values() for r in read_rows(q) if int(r["it"]) == it}
            for c, (xs, _tried) in drawn.items():          # a replayed iteration must draw what its rows hold
                for k, x in enumerate(xs):
                    if x is not None and (c, it, k) in stored and np.abs(np.asarray(stored[(c, it, k)]) - x).max() > 1e-5:
                        raise SystemExit(f"replay of {task} {level} it {it} call {c} cand {k} drew another candidate")
            items = []
            for c, (xs, _tried) in drawn.items():
                for k, x in enumerate(xs):
                    if x is None:
                        continue
                    sp = space[c]
                    key = {"d": d6[c], "call": c, "it": it, "cand": k, "level": level,
                           "v": [round(float(y), 7) for y in x[:sp.nv]], "prm": sp.prm(x),
                           "x": [round(float(y), 7) for y in x]}
                    items.append((key, d6[c], x[:sp.nv], key["prm"]))
            tm = {}
            if band:
                n_run = evaluate_band(task, band, paths, items, SEARCH_SEEDS, scene, HG, E, tm)
                sc = band_scores(task, paths, it)
            else:
                n_run = evaluate(task, a.plant, path, items, SEARCH_SEEDS, scene, HG, E, tm)
                sc = scores(task, [r for r in read_rows(path) if int(r["it"]) == it])
            for c, (xs, tried) in drawn.items():
                cand = [(sc[(c, it, k)], k, x) for k, x in enumerate(xs) if x is not None and (c, it, k) in sc]
                if not cand:
                    continue
                cand.sort(key=lambda z: (-z[0][0], -z[0][1], z[1]))
                if best[c] is None or cand[0][0][0] > best[c]["score"]:
                    best[c] = {"score": cand[0][0][0], "it": it, "cand": cand[0][1], "x": cand[0][2].tolist(),
                               "plants": cand[0][0][3] if band else None}
                ne = max(2, int(math.ceil(ELITE * len(cand))))
                X = np.array([z[2] for z in cand[:ne]])
                mean, std = state[c]
                mean = SMOOTH * X.mean(0) + (1 - SMOOTH) * mean
                std = np.maximum(SMOOTH * X.std(0) + (1 - SMOOTH) * std, space[c].floor())
                state[c] = (mean, std)
                if (c, it) not in gens_done:
                    with open(gpath, "a") as fh:
                        fh.write(json.dumps({"call": c, "it": it, "n": len(cand), "tried": tried,
                                             "scores": [round(z[0][0], 4) for z in cand],
                                             "elite_mean": round(float(np.mean([z[0][0] for z in cand[:ne]])), 4),
                                             "best": round(best[c]["score"], 4), "mean": mean.round(7).tolist(),
                                             "std": std.round(7).tolist(),
                                             **({"best_plants": best[c]["plants"]} if band else {})}) + "\n")
                        fh.flush()
                        os.fsync(fh.fileno())
            n_feas = sum(sum(x is not None for x in xs) for xs, _ in drawn.values())
            t_all = time.time() - t0
            print(f"{time.strftime('%H:%M:%S')} {task} {level} it {it}: {n_feas} candidates of {len(d6)} layouts, "
                  f"{n_run} rollouts run, draw {t_draw:.0f} s, all {t_all:.0f} s, median best "
                  f"{np.median([b['score'] for b in best.values() if b]):.3f}", flush=True)
            with open(path.with_name(path.stem + "_iters.jsonl"), "a") as fh:     # the iteration's compute record
                fh.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "task": task, "level": level,
                                     "plant": name, "it": it, "layouts": len(d6), "candidates": n_feas,
                                     "draws": sum(tr for _xs, tr in drawn.values()), "rollouts": n_run,
                                     "replay": n_run == 0, "draw_s": round(t_draw, 2), **tm,
                                     "total_s": round(t_all, 2)}) + "\n")
                fh.flush()
                os.fsync(fh.fileno())


def picks(task, plant, level):
    """{call: {"v", "prm", "score", "it", "cand"}}: the best candidate seen per layout at a level, from its rows."""
    path = rows_dir(plant) / level / f"{task}_d{TASKS[task]:g}.jsonl"
    rows = read_rows(path)
    sc = scores(task, rows)
    keyed = {}
    for r in rows:
        keyed.setdefault(ident(r), r)
    out = {}
    for k, (s, fe, n) in sc.items():
        if n < len(SEARCH_SEEDS):
            continue
        c = k[0]
        if c not in out or (s, fe, -k[1], -k[2]) > (out[c]["score"], out[c]["fe"], -out[c]["it"], -out[c]["cand"]):
            r = keyed[k]
            out[c] = {"v": r["v"], "prm": r.get("prm") or {}, "score": s, "fe": fe, "it": k[1], "cand": k[2]}
    return out


def band_picks(task, name, level):
    """picks() of a band: the best candidate seen per layout by the worst-case score, with its score on each plant."""
    paths = band_paths(rows_dir(name) / level / f"{task}_d{TASKS[task]:g}.jsonl", BANDS[name])
    keyed = {}
    for r in read_rows(paths[BANDS[name][-1]]):
        keyed.setdefault(ident(r), r)
    out = {}
    for k, (s, fe, n, per) in band_scores(task, paths).items():
        if n < len(SEARCH_SEEDS):
            continue
        c = k[0]
        if c not in out or (s, fe, -k[1], -k[2]) > (out[c]["score"], out[c]["fe"], -out[c]["it"], -out[c]["cand"]):
            r = keyed[k]
            out[c] = {"v": r["v"], "prm": r.get("prm") or {}, "score": s, "fe": fe, "it": k[1], "cand": k[2],
                      "plants": per}
    return out


def s1_start(task, plant, lays):
    """{call: (mean score, mean margin)} of S1's grasp at the search seeds, from the plant's grid rows."""
    import hom_landscape as L
    want = {int(L_["call"]): int(L_["s1_grasp"]) for L_ in lays}
    p = GRID_ROWS / f"grid2dff_v4b{PLANTS[plant][2]}" / f"{task}_d{TASKS[task]:g}.jsonl"
    acc = {}
    for x in p.read_text().splitlines():
        if not x.strip():
            continue
        r = json.loads(x)
        c = int(r["call"])
        if want.get(c) == int(r.get("grasp", 0)) and int(r["seed"]) in SEARCH_SEEDS:
            a = acc.setdefault(c, [[], []])
            a[0].append(L.rollout_score(task, r))
            a[1].append(margin(task, r))
    return {c: (float(np.mean(s)), float(np.mean(m))) for c, (s, m) in acc.items()}


def elitist(task, plant):
    """{call: {"S2e": pick, "S3e": pick}}, the elitist picks of a plant's searches (the night of 2026-11-03 scored draws
    around each start, never the start itself). S2e is the better of S1's grasp and S2's pick, S3e the best of S1, S2 and
    S3, by mean score at the search seeds and then the search's tie-break; a full tie keeps the earlier level."""
    lays = json.loads((rows_dir(plant) / f"{task}_layouts.json").read_text())["layouts"]
    p2, p3, s1 = picks(task, plant, "grasp"), picks(task, plant, "full"), s1_start(task, plant, lays)
    out = {}
    for L_ in lays:
        c = L_["call"]
        sc = {"S1": s1.get(c, (L_["s1"], 0.0))}
        if c in p2:
            sc["S2"] = (p2[c]["score"], p2[c]["fe"])
        if c in p3:
            sc["S3"] = (p3[c]["score"], p3[c]["fe"])
        out[c] = {"S2e": max((k for k in ("S1", "S2") if k in sc), key=lambda k: sc[k]),
                  "S3e": max((k for k in ("S1", "S2", "S3") if k in sc), key=lambda k: sc[k])}
    return out


# ------------------------------------------------------------------------------------------------ validation

def cmd_validate(a):
    """Each layout's S1, S2 and S3 picks at the validation seeds on a.plant; with --picks-from, another plant's picks
    (the same layouts), rows under val_from_<plant>/: the score a design tuned on that plant keeps on this one."""
    if a.band:
        return cmd_validate_band(a)
    setenv(a.plant)
    import fp_sr2_toy as FP
    import hom_ea as E
    import hom_gpu as HG
    task = a.task
    FP.set_sizes(**({"sph_d_mm": TASKS[task]} if POSE[task] == "tripod" else {"cyl_d_mm": TASKS[task]}))
    src = a.picks_from or a.plant
    lays = json.loads((rows_dir(src) / f"{task}_layouts.json").read_text())["layouts"]
    p2, p3 = picks(task, src, "grasp"), picks(task, src, "full")
    items = []
    for L_ in lays:
        c = L_["call"]
        sets = [("S1", L_["v1"], {})]
        if c in p2:
            sets.append(("S2", p2[c]["v"], {}))
        if c in p3:
            sets.append(("S3", p3[c]["v"], p3[c]["prm"]))
        for name, v, prm in sets:
            key = {"d": L_["d"], "call": c, "pick": name, "v": [round(float(y), 7) for y in v], "prm": prm}
            items.append((key, L_["d"], np.asarray(v, float), prm))
    path = rows_dir(a.plant) / ("val" if src == a.plant else f"val_from_{src}") / f"{task}_d{TASKS[task]:g}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = HG.Scene("sphere" if POSE[task] == "tripod" else "cylinder", TASKS[task])
    n = evaluate(task, a.plant, path, items, VAL_SEEDS, scene, HG, E)
    print(f"{task}: {len(items)} picks, {n} rollouts run", flush=True)


def val_path(task, name, plant, src):
    """The validation rows of src's picks (src: the band or a plant) on a plant of the band: the band's own under
    rows/<band>/val/<plant>/; another plant's from the 2026-11-03 queue (src's val/, or the plant's val_from_<src>/)
    where they exist, else under rows/<band>/val_from_<src>/<plant>/."""
    f = f"{task}_d{TASKS[task]:g}.jsonl"
    if src == name:
        return rows_dir(name) / "val" / plant / f
    old = rows_dir(src) / "val" / f if plant == src else rows_dir(plant) / f"val_from_{src}" / f
    return old if old.exists() else rows_dir(name) / f"val_from_{src}" / plant / f


def val_items(task, src, name):
    """The layouts of src and the (key, d6, v, prm) of their S1, S2 and S3 picks, as cmd_validate builds them."""
    band = src == name
    lays = json.loads((rows_dir(src) / f"{task}_layouts.json").read_text())["layouts"]
    p2, p3 = ((band_picks(task, src, "grasp"), band_picks(task, src, "full")) if band
              else (picks(task, src, "grasp"), picks(task, src, "full")))
    items = []
    for L_ in lays:
        c = L_["call"]
        sets = [("S1", L_["v1"], {})]
        if c in p2:
            sets.append(("S2", p2[c]["v"], {}))
        if c in p3:
            sets.append(("S3", p3[c]["v"], p3[c]["prm"]))
        for pick, v, prm in sets:
            key = {"d": L_["d"], "call": c, "pick": pick, "v": [round(float(y), 7) for y in v], "prm": prm}
            items.append((key, L_["d"], np.asarray(v, float), prm))
    return lays, items


def cmd_validate_band(a):
    """The band's picks, or with --picks-from a plant's picks (the same layouts), at the validation seeds on every plant
    of the band (val_path). Rows of the 2026-11-03 queue are read, never appended to: a plant whose old rows lack a
    pick or a seed stops the run."""
    name = band_name(a.band)
    band = BANDS[name]
    setenv(band[-1])
    import fp_sr2_toy as FP
    import hom_ea as E
    import hom_gpu as HG
    task = a.task
    FP.set_sizes(**({"sph_d_mm": TASKS[task]} if POSE[task] == "tripod" else {"cyl_d_mm": TASKS[task]}))
    src = a.picks_from or name
    lays, items = val_items(task, src, name)
    if a.n:
        keep = {L_["call"] for L_ in lays[:a.n]}
        items = [it for it in items if it[0]["call"] in keep]
    paths = {p: val_path(task, name, p, src) for p in band}
    run = []
    for p, q in paths.items():
        if q.is_relative_to(rows_dir(name)):
            run.append(p)
            continue
        have = {(ident(r), int(r["seed"])) for r in read_rows(q)}
        miss = [(ident(k), sd) for k, *_ in items for sd in VAL_SEEDS if (ident(k), sd) not in have]
        if miss:
            raise SystemExit(f"{task}: {q.relative_to(ROOT)} lacks {len(miss)} rollouts of {src}'s picks, e.g. {miss[0]}")
        print(f"{task} {p}: {src}'s picks read from {q.relative_to(ROOT)}", flush=True)
    scene = HG.Scene("sphere" if POSE[task] == "tripod" else "cylinder", TASKS[task])
    n = evaluate_band(task, run, paths, items, VAL_SEEDS, scene, HG, E) if run else 0
    print(f"{task}: {len(items)} picks of {src} on {', '.join(run) or 'no plant'}, {n} rollouts run", flush=True)


# ------------------------------------------------------------------------------------------------ report

def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return round(float(spearmanr(a, b).statistic), 3)


def cmd_report(a):
    if a.band:
        return cmd_report_band(a)
    setenv(a.plant)
    out = {"plant": a.plant, "hos_plant": PLANTS[a.plant][0], "scenes": PLANTS[a.plant][1], "search_seeds": SEARCH_SEEDS,
           "val_seeds": VAL_SEEDS, "pop": POP, "iters": ITERS, "elite": ELITE, "smooth": SMOOTH, "forces": FORCES,
           "roles": ROLES, "tasks": {}}
    for task in TASKS:
        lp = rows_dir(a.plant) / f"{task}_layouts.json"
        if not lp.exists():
            continue
        lays = json.loads(lp.read_text())["layouts"]
        p2, p3 = picks(task, a.plant, "grasp"), picks(task, a.plant, "full")
        val = scores(task, read_rows(rows_dir(a.plant) / "val" / f"{task}_d{TASKS[task]:g}.jsonl"))
        cross = {d.name.removeprefix("val_from_"): scores(task, read_rows(d / f"{task}_d{TASKS[task]:g}.jsonl"))
                 for d in sorted(rows_dir(a.plant).glob("val_from_*"))}
        eli = {p: elitist(task, p) for p in [a.plant, *cross]}     # a pick run elsewhere was chosen on its own plant
        s1 = s1_start(task, a.plant, lays)
        per = []
        for L_ in lays:
            c = L_["call"]
            e = {"call": c, "xy": L_["xy"], "group": L_["group"], "s1": L_["s1"],
                 "s2": round(p2[c]["score"], 4) if c in p2 else None,
                 "s3": round(p3[c]["score"], 4) if c in p3 else None,
                 "prm3": p3[c]["prm"] if c in p3 else None,
                 "s1_margin": round(s1[c][1], 3) if c in s1 else None,
                 "s2_margin": round(p2[c]["fe"], 3) if c in p2 else None,
                 "s3_margin": round(p3[c]["fe"], 3) if c in p3 else None,
                 "s2e_pick": eli[a.plant][c]["S2e"], "s3e_pick": eli[a.plant][c]["S3e"]}
            for k in ("S1", "S2", "S3"):
                v = val.get((c, k))
                e[f"{k.lower()}_val"] = round(v[0], 4) if v and v[2] >= len(VAL_SEEDS) else None
                if task == "twist" and v:
                    e[f"{k.lower()}_val_ramp_deg"] = round(v[1], 2)
                for src, vv in cross.items():
                    w = vv.get((c, k))
                    e[f"{k.lower()}_val_from_{src}"] = round(w[0], 4) if w and w[2] >= len(VAL_SEEDS) else None
            for k in ("S2e", "S3e"):
                e[f"{k.lower()}_val"] = e[f"{eli[a.plant][c][k].lower()}_val"]
                for src in cross:
                    e[f"{k.lower()}_pick_{src}"] = eli[src][c][k] if c in eli[src] else None
                    e[f"{k.lower()}_val_from_{src}"] = (e[f"{eli[src][c][k].lower()}_val_from_{src}"]
                                                        if c in eli[src] else None)
            per.append(e)
        t = {"layouts": per, "n": len(per)}
        for lo, hi in (("s1_val", "s2_val"), ("s1_val", "s3_val"), ("s2_val", "s3_val"), ("s1", "s1_val"),
                       ("s1_val", "s2e_val"), ("s1_val", "s3e_val"), ("s2e_val", "s3e_val")):
            pairs = [(e[lo], e[hi]) for e in per if e[lo] is not None and e[hi] is not None]
            t[f"rho_{lo}_{hi}"] = spearman(*zip(*pairs)) if len(pairs) >= 3 else None
            t[f"n_{lo}_{hi}"] = len(pairs)
        for k in ("s1", "s2", "s3", "s1_val", "s2_val", "s3_val", "s2e_val", "s3e_val"):
            vals = [e[k] for e in per if e[k] is not None]
            t[f"median_{k}"] = round(float(np.median(vals)), 4) if vals else None
            if vals:
                b = max((e for e in per if e[k] is not None), key=lambda e: e[k])
                t[f"best_{k}"] = {"call": b["call"], "xy": b["xy"], "score": b[k]}
        # a search that ended below its start: level grasp started from S1's grasp, level full from S2's raw pick
        for level, lo, hi in (("grasp", "s1", "s2"), ("full", "s2", "s3")):
            d = [e[hi] - e[lo] for e in per if e[lo] is not None and e[hi] is not None]
            t[f"below_start_{level}"] = {"n": int(sum(x < 0 for x in d)), "of": len(d),
                                         "worst": round(min(d), 4) if d and min(d) < 0 else None}
        out["tasks"][task] = t
        print(f"{task}: medians S1 {t['median_s1_val']} S2 {t['median_s2_val']} S3 {t['median_s3_val']}, elitist S2e "
              f"{t['median_s2e_val']} S3e {t['median_s3e_val']} (seeds 20-24); rho S1-S2 {t['rho_s1_val_s2_val']}, "
              f"S1-S3 {t['rho_s1_val_s3_val']}, S1-S3e {t['rho_s1_val_s3e_val']}; ended below the start: grasp "
              f"{t['below_start_grasp']}, full {t['below_start_full']}", flush=True)
    p = rows_dir(a.plant) / f"{time.strftime('%Y%m%d')}-score_levels.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p.relative_to(ROOT))


def stats(d):
    """Median, mean, n and the counts above +0.05 and below -0.05 of paired differences."""
    d = np.asarray(d, float)
    if not len(d):
        return {"n": 0, "median": None, "mean": None, "up": 0, "down": 0}
    return {"n": len(d), "median": round(float(np.median(d)), 4), "mean": round(float(d.mean()), 4),
            "up": int((d > 0.05).sum()), "down": int((d < -0.05).sum())}


def cmd_report_band(a):
    """rows/<band>/<date>-band_levels.json: per layout the band's S1, S2 and S3 (worst-case score at the search seeds,
    each plant's score, and each search's start re-scored as candidate 0), and the validated scores on every plant of
    the band and their lowest, of the band's picks and of the S1, S2e and S3e picks of each plant with searches of its
    own (the nominal friction 1.0 and friction 0.4); per task the medians and the paired differences between the
    band's picks and each plant's picks of the same level."""
    name = band_name(a.band)
    band = BANDS[name]
    out = {"band": name, "plants": list(band), "plant_specs": {p: PLANTS[p] for p in band}, "search_seeds": SEARCH_SEEDS,
           "val_seeds": VAL_SEEDS, "pop": POP, "elite": ELITE, "smooth": SMOOTH, "forces": FORCES, "roles": ROLES,
           "tasks": {}}
    for task in TASKS:
        lp = rows_dir(name) / f"{task}_layouts.json"
        if not lp.exists():
            continue
        lays = json.loads(lp.read_text())["layouts"]
        f = f"{task}_d{TASKS[task]:g}.jsonl"
        pk = {"s2": band_picks(task, name, "grasp"), "s3": band_picks(task, name, "full")}
        st = {"s2": band_scores(task, band_paths(rows_dir(name) / "grasp" / f, band)),
              "s3": band_scores(task, band_paths(rows_dir(name) / "full" / f, band))}
        srcs = [p for p in band if (rows_dir(p) / f"{task}_layouts.json").exists()]     # plants with searches
        eli = {p: elitist(task, p) for p in srcs}
        val = {(src, p): scores(task, read_rows(val_path(task, name, p, src))) for src in [name, *srcs] for p in band}
        per = []
        for L_ in lays:
            c = L_["call"]
            e = {"call": c, "xy": L_["xy"], "group": L_["group"], "s1": L_["s1"], "s1_plants": L_["s1_plants"],
                 "s1_grasp": L_["s1_grasp"], "base_s1_grasp": L_.get("base_s1_grasp")}
            for k in ("s2", "s3"):
                s0 = st[k].get((c, 0, 0))
                e[f"{k}_start"] = round(s0[0], 4) if s0 else None
                e[f"{k}_start_plants"] = s0[3] if s0 else None
                e[k] = round(pk[k][c]["score"], 4) if c in pk[k] else None
                e[f"{k}_plants"] = pk[k][c]["plants"] if c in pk[k] else None
                e[f"{k}_it_cand"] = [pk[k][c]["it"], pk[k][c]["cand"]] if c in pk[k] else None
            e["prm3"] = pk["s3"][c]["prm"] if c in pk["s3"] else None
            e["v"] = {"S1": L_["v1"], "S2": pk["s2"][c]["v"] if c in pk["s2"] else None,
                      "S3": pk["s3"][c]["v"] if c in pk["s3"] else None}
            e["val"] = {}
            for src in [name, *srcs]:
                d = {}
                for pick in ("S1", "S2", "S3"):
                    v = {}
                    for p in band:
                        w = val[(src, p)].get((c, pick))
                        v[p] = round(w[0], 4) if w and w[2] >= len(VAL_SEEDS) else None
                    v["min"] = min(v.values()) if all(x is not None for x in v.values()) else None
                    d[pick] = v
                if src != name:
                    for k in ("S2e", "S3e"):
                        d[k] = dict(d[eli[src][c][k]]) if c in eli[src] else {p: None for p in [*band, "min"]}
                        e[f"{k.lower()}_pick_{src}"] = eli[src][c][k] if c in eli[src] else None
                e["val"][src] = d
            per.append(e)
        t = {"n": len(per), "sources": [name, *srcs], "median_val": {}, "diff": {},
             "iters": {lvl: 1 + max((int(g["it"]) for g in read_rows(rows_dir(name) / lvl / f"{task}_d{TASKS[task]:g}"
                                                                     "_gens.jsonl")), default=-1)
                       for lvl in ("grasp", "full")}}
        for src in [name, *srcs]:
            for pick in (("S1", "S2", "S3") if src == name else ("S1", "S2e", "S3e")):
                for p in [*band, "min"]:
                    xs = [e["val"][src][pick][p] for e in per if e["val"][src][pick][p] is not None]
                    t["median_val"][f"{src}|{pick}|{p}"] = round(float(np.median(xs)), 4) if xs else None
        for src in srcs:                    # the band's pick against the plant's pick of the same level
            for wk, sk in (("S1", "S1"), ("S2", "S2e"), ("S3", "S3e")):
                for p in [*band, "min"]:
                    t["diff"][f"{wk}|{src}|{p}"] = stats([e["val"][name][wk][p] - e["val"][src][sk][p] for e in per
                                                          if e["val"][name][wk][p] is not None
                                                          and e["val"][src][sk][p] is not None])
        for k in ("s1", "s2_start", "s2", "s3_start", "s3"):
            xs = [e[k] for e in per if e[k] is not None]
            t[f"median_{k}"] = round(float(np.median(xs)), 4) if xs else None
        for k in ("s2", "s3"):              # the pick against its search's start, re-scored in the same batches
            t[f"{k}_minus_start"] = stats([e[k] - e[f"{k}_start"] for e in per
                                           if e[k] is not None and e[f"{k}_start"] is not None])
        t["layouts"] = per
        out["tasks"][task] = t
        mv = t["median_val"]
        line = lambda src, k: ", ".join(f"{p} {mv[src + '|' + k + '|' + p]}" for p in [*band, "min"])  # noqa: E731
        print(f"{task}: medians at seeds 20-24, {name} S3 {line(name, 'S3')}; "
              + "; ".join(f"{src} S3e {line(src, 'S3e')}" for src in srcs), flush=True)
    out["iters_run"] = max((n for t in out["tasks"].values() for n in t["iters"].values()), default=0)
    p = rows_dir(name) / f"{time.strftime('%Y%m%d')}-band_levels.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p.relative_to(ROOT))


def cmd_probe(a):
    """Time the feasibility check and scene.prepare on the first layouts of a task (no GPU)."""
    setenv(a.plant)
    import fp_sr2_toy as FP
    import hom_gpu as HG
    task = a.task
    FP.set_sizes(**({"sph_d_mm": TASKS[task]} if POSE[task] == "tripod" else {"cyl_d_mm": TASKS[task]}))
    lays = layouts(task, a.plant)[:a.n or 3]
    scene = HG.Scene("sphere" if POSE[task] == "tripod" else "cylinder", TASKS[task])
    for L_ in lays:
        sp = Space(task, a.level, L_["widths"])
        mean, std = sp.start(L_["v1"])
        t0 = time.time()
        _c, xs, tried = draw_layout((task, a.level, L_["call"], 99, L_["d"], mean, std, sp.lo, sp.hi, sp.nv, 8))
        t1 = time.time()
        ok = [x for x in xs if x is not None]
        for x in ok:
            setup = setup_of(HG, task, L_["d"], x[:sp.nv])
            scene.prepare(HG.mount_xy(L_["d"]), setup)
        t2 = time.time()
        print(f"call {L_['call']} xy {L_['xy']}: {len(ok)}/8 feasible in {tried} draws, {1000 * (t1 - t0) / tried:.0f} ms "
              f"per draw; prepare {1000 * (t2 - t1) / max(1, len(ok)):.0f} ms per candidate; prm {sp.prm(ok[0]) if ok else None}",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description="CEM refinement of grasps, forces and contact modes on the HOM score")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--task", required=True, choices=list(TASKS))
    pr.add_argument("--level", required=True, choices=("grasp", "full"))
    pr.add_argument("--plant", default="sv25", choices=list(PLANTS))
    pr.add_argument("--n", type=int, default=0, help="only the first n layouts of the sample (a smoke test)")
    pr.add_argument("--iters", type=int, default=0)
    pr.add_argument("--pop", type=int, default=0)
    pr.add_argument("--smoke", action="store_true", help="rows under rows_smoke/, never the study's")
    pr.add_argument("--top", type=int, default=N_TOP, help="layouts by S1 (when the sample is first written)")
    pr.add_argument("--rand", type=int, default=N_RAND, help="layouts at random (when the sample is first written)")
    pr.add_argument("--layouts-from", default=None, choices=list(PLANTS), help="another plant's layout sample")
    hb = "a band of plants (BANDS: its name, or its plants comma-separated): the worst-case score over them"
    pr.add_argument("--band", default=None, help=hb)
    pv = sub.add_parser("validate")
    pv.add_argument("--task", required=True, choices=list(TASKS))
    pv.add_argument("--plant", default="sv25", choices=list(PLANTS))
    pv.add_argument("--picks-from", default=None, choices=list(PLANTS), help="run another plant's picks here")
    pv.add_argument("--band", default=None, help=hb + "; the picks run on each plant of the band")
    pv.add_argument("--n", type=int, default=0, help="--band: only the first n layouts (a smoke test)")
    pv.add_argument("--smoke", action="store_true")
    pp = sub.add_parser("report")
    pp.add_argument("--plant", default="sv25", choices=list(PLANTS))
    pp.add_argument("--band", default=None, help=hb)
    pp.add_argument("--smoke", action="store_true")
    pb = sub.add_parser("probe")
    pb.add_argument("--task", required=True, choices=list(TASKS))
    pb.add_argument("--level", default="grasp", choices=("grasp", "full"))
    pb.add_argument("--plant", default="sv25", choices=list(PLANTS))
    pb.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    if getattr(a, "smoke", False):
        global ROWS_ROOT
        ROWS_ROOT = OUT / "rows_smoke"
    {"run": cmd_run, "validate": cmd_validate, "report": cmd_report, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    main()
