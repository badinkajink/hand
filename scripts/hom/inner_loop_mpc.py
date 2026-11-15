#!/usr/bin/env python3
"""Sampling MPC on the SR2 hand in MuJoCo-Warp: the training-free baselines of the inner-loop benchmark (owner,
2026-11-12: "let's try MPC. In addition to the typical cem, mppi, predictive sampling, we should see if dial-MPC
(annealed sampling) which we implemented on CPU finally yields better results now that we have warp, and maybe the
worst case sampling from spider (which uses annealed sampling) can be broadly used").

Plant and task are the benchmark's (uhas_sr2.UBatch, the twist of the 40 mm sphere). HOM drives the approach, the
closing, the squeeze and the lift. From the task phase (t = 2.4 s) a planner sets the nine servo targets. The planner's
model is the plant's: the same scene, pads, servo and 1 ms physics in MuJoCo-Warp. Every REPLAN periods the state of
each plant world (positions, velocities, servo targets, the post and the solver's warm start) is copied into the
planner's worlds of that plant world. A plan is the nine targets at KNOTS knots over H periods, linear between the
knots, with the first knot pinned to the target applied now. The plant applies the plan's first REPLAN periods.

Cost of a rollout, summed over its periods: the task's terms of J, then the limit terms that the UHAS rerun with
limits also has in its reward (uhas_sr2.REW_LIMITS):
    (rotation error / 5 deg)^2 + (distance from the start position / 3 mm)^2 + 2 per pad under 0.05 N
    + 100 per period after the sphere fell 15 mm
    + 0.5 sum_j clip((servo demand_j / torque limit - 0.8) / 0.2, 0, 3), the demand kp |target - q|
    + 0.25 sum_j clip((3 deg - distance to the joint's range limit) / 3 deg, 0, 1)
and, per gap between knots, sum_j (rate_j / 20 deg/s)^2.

Methods. Each spends iters x n rollouts per replanning (n = --n samples per plant world and iteration; spider_wc n / 3
samples in each of three friction groups); the update rules are those of hand_object_scale_mpc.Planner (2026-10-13)
and of SPIDER:
    ps        predictive sampling: the cheapest of the nominal and its noisy copies (Howell et al. 2022)
    mppi      an MPPI step per iteration: weights exp(-(c - c_min) / lam), lam = 0.03 std(c)
    cem       cross-entropy: the 15 % cheapest set the mean and std (momentum 0.1, floor 0.005 rad); the mean executed
    icem      CEM with AR(1) knot noise (0.7), the two best kept into the next replanning; the best sample executed
    anneal    DIAL-MPC's annealed sampling as in mujoco_mpc (the CPU planner of 2026-10-13): knot noise
              sigma exp(-i / (0.5 iters) - (K - 1 - k) / (0.5 K)) in iteration i at knot k, an MPPI step after each
    spider    SPIDER's optimizer (Pan et al. 2025; spider/optimizers/sampling.py): knot noise from 0.5 to 1.0 sigma
              along the horizon, times 0.1^(i / iters) in iteration i; the first sample noise-free, the last 1 % at
              0.01; a softmax at temperature 0.3 over the best 10 % of the normalized rewards (-cost per period); stop
              when no sample beats the nominal by 0.01
    spider_wc the same, each sample's reward the worst over the scene frictions 0.4, 0.7 and 1.0 (SPIDER's domain
              randomization groups), n / 3 samples per iteration
    grad      gradient descent with finite differences, MJPC's gradient planner (owner, 2026-11-14: "did we try gradient
              based methods? with everything with maintained contact, it should be possible?"; MuJoCo-Warp 3.6 has no
              derivatives): per pair of iterations, central differences of the cost at eps = 0.02 rad along each free
              knot coordinate and random unit directions, the gradient fitted by least squares, then a line search of n
              step sizes along it (0 and 0.002-0.3 rad, log-spaced), the cheapest kept
    ilqr      iLQR with finite-difference linearization (owner, 2026-11-15: "did we also try ilqr/ddp/feasibleddp on top
              of the gradient search?"): inner_loop_ilqr.Planner, targets held for --il-step periods, --iters
              iterations of linearization, Gauss-Newton backward pass and a parallel line search per replanning
The script also runs HOM through the task (method hom) and a trained UHAS policy (method uhas, --run), so that every
controller is scored on the same plant, pairs and seeds.

    .venv/bin/python scripts/hom/inner_loop_mpc.py pairs
    .venv/bin/python scripts/hom/inner_loop_mpc.py timing --chunk 16 --n 128
    .venv/bin/python scripts/hom/inner_loop_mpc.py run --method anneal [--seeds 0 1] [--plant-mu 1.0] [--chunk 16]
    .venv/bin/python scripts/hom/inner_loop_mpc.py pairs --task roll --rule rank      # rows/mpc_pairs_roll_rank.json
    .venv/bin/python scripts/hom/inner_loop_mpc.py run --task roll --pairs _rank --method hom
    .venv/bin/python scripts/hom/inner_loop_mpc.py pairs --task press --rule all      # every layout of the grid
    .venv/bin/python scripts/hom/inner_loop_mpc.py pairs --task press --rule top --src _all   # mpc_pairs_press_alltop
Pairs: rows/mpc_pairs_twist.json, the ten test layouts with HOM's best J and the ten with UHAS's best J, each with that
controller's best grasp. Rows: docs/experiments/20261112-inner_loop_benchmark/rows/mpc_<method>_twist[_mu<mu>].jsonl,
one per world, appended and fsynced per chunk; a rerun skips the worlds already written.

Tasks and the ranking test (owner, 2026-11-14: "is this objective score across multiple tasks? we should run them if
not"; "you can run the further test on running realtime DIAL"). --task runs the roll of the 40 mm sphere and the pin-push
and the press of the 25 mm cylinder as well (uhas_sr2.D_MM). The cost's rotation term is then the task's error of J:
the angle to the commanded orientation for the spheres, the angle between the cylinder's axis and its commanded axis
(the start axis turned by the commanded pinch angle about palm x) for the cylinders. Its position term follows the
task's reference point: the sphere's centre, which in the roll moves along the thumb's pad at r w (uhas_sr2.Tracker),
and for the cylinders the midpoint of the thumb's and the index's centres of pressure at the task's start, the pinch,
which the commanded rotation leaves in place. pairs --rule rank writes rows/mpc_pairs_<task>_rank.json, the 50 layouts
with HOM's best J on the task's grid and 50 drawn at random from the rest, each with HOM's best grasp, and
mpc_pairs_<task>_rank20.json, every fifth of each half; run --pairs _rank reads it and names its rows
mpc_<method>_<task>_rank....
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import inner_loop_plant  # noqa: E402,F401  the talk's plant (SCS0009, 2.5 MPa pads), set before hom_gpu reads it
import hom_gpu as HG  # noqa: E402
import inner_loop_usability as IU  # noqa: E402
import uhas_sr2 as UH  # noqa: E402
import uhas_sr2_cik as U  # noqa: E402
import warp as wp  # noqa: E402
from sr2hand.rl.math import quat_conj, quat_from_axis_angle, quat_mul, quat_rotate, rotate_axis  # noqa: E402

T = HG.T
TASK = "twist"                           # set_task; uhas_sr2.D_MM lists the four


def set_task(task):
    global TASK
    if task not in UH.D_MM:
        raise SystemExit(f"task {task}: one of {sorted(UH.D_MM)}")
    TASK = task


def pairs_path(name=""):
    return ROWS / f"mpc_pairs_{TASK}{name}.json"
ROWS = UH.OUT / "rows"
RUN = "20261112-1244_twist_s0"           # the UHAS run whose test split and usability rows choose the pairs
SHARED = ROOT / "docs/experiments/20261103-hom_score_refine/rows/sv25_track/shared"
REPLAN, H, KNOTS = 10, 30, 3             # periods between replannings, horizon (periods), knots (the first pinned)
COST = dict(rot_deg=5.0, pos_mm=3.0, pad=2.0, f_on=0.05, drop=100.0, drop_m=0.015, tau=0.5, tau_on=0.8, q=0.25,
            q_band_deg=3.0, rate=1.0, rate_dps=20.0, acc=0.0, rev=0.0, err_db_deg=0.0)
PLAN = dict(iters=4, sigma=0.08, lam_rel=0.03, elite_frac=0.15, momentum=0.1, sigma_min=0.005,
            ar_alpha=0.7, keep=2, beta1=0.5, beta2=0.5, sp_first=0.5, sp_last=1.0, sp_final=0.1, sp_exploit=0.01,
            sp_exploit_scale=0.01, sp_temp=0.3, sp_top=0.1, sp_improve=0.01, frictions=(0.4, 0.7, 1.0),
            grad_eps=0.02, grad_step_min=0.002, grad_step_max=0.3)
METHODS = ("ps", "mppi", "cem", "icem", "anneal", "spider", "spider_wc", "grad")
# HOM-space planners (owner, 2026-11-13: "HOM-space controls ... 18-dim control space ... keep fixed ... v_sep (x') and
# probably w_spin (ex')"): a plan sets, per finger and knot, the two rolling and the two sliding rates of the reference
# relative twist of HOM's Eq. (2) (rows HS_FREE of spin, roll y, roll z, sep, slide y, slide z), 12 numbers per knot in
# units of HS_UNIT (rad/s, m/s). Each period HOM solves Eq. (2) as usual, then again with those rows' references set to
# the relative twist of its own solution plus the sample's and their weights to 1 (hom_gpu.Batch.control, v_add): the
# zero plan is HOM, and HOM keeps the spin, the separation and its force loop. (A first version, 11:00, set those rows'
# weights to 1 around a zero reference: its zero plan held every contact sticking and lost the pads.)
HS_FREE = [1, 2, 4, 5]
HS_UNIT = (1.0, 1.0, 0.02, 0.02)
PLAN_HS = dict(sigma=0.3, bound=3.0, rate=2.0)    # the noise and the bound in units; the knot-rate cost's scale, units/s
HS_METHODS = tuple("hs_" + m for m in METHODS if m not in ("spider_wc", "grad"))
# Residual planners on HOM (owner, 2026-11-14: "can we fuse methods?"): HOM runs every period in the plant and in the
# planner's worlds, and a plan adds joint-space offsets to HOM's servo targets (uhas_sr2.UBatch q_add), 9 numbers per
# knot within +-PLAN_RES bound rad. The zero plan is HOM.
RES_METHODS = tuple("res_" + m for m in ("ps", "mppi", "anneal", "grad"))
PLAN_RES = dict(sigma=0.04, bound=0.3, rate_dps=20.0)
KNOT_T = np.linspace(0.0, H, KNOTS)
PLAN_DT = None                  # the planner's physics step, s (--plan-dt); None: the plant's 1 ms
WARM = "hold"                   # past its last knot the shifted plan holds the last knot, or (extrap) keeps its slope


def interp(times, knot_t=KNOT_T, extrap=False):
    """(len(times), K): the linear weights of the knots at each time, in periods. Beyond the last knot the last knot is
    held, or with extrap the last segment's slope continues."""
    K = len(knot_t)
    w = np.stack([np.interp(times, knot_t, np.eye(K)[k]) for k in range(K)], 1)
    if extrap and K > 1:
        a = np.clip((np.asarray(times, float) - knot_t[-1]) / (knot_t[-1] - knot_t[-2]), 0.0, None)
        w[:, -1] += a
        w[:, -2] -= a
    return torch.tensor(w, dtype=torch.float32, device="cuda")


W_ROLL = interp(np.arange(1, H + 1))          # the target of rollout period h (it ends at h + 1)
W_EXEC = interp(np.arange(1, REPLAN + 1))     # the plant's targets until the next replanning
W_SHIFT = interp(REPLAN + KNOT_T)             # the knots REPLAN periods later


def configure(replan, horizon, knots, iters):
    """Another replanning interval, horizon, knot count or iteration count (periods); the interpolation tables
    again."""
    global REPLAN, H, KNOTS, KNOT_T, W_ROLL, W_EXEC, W_SHIFT
    REPLAN, H, KNOTS = int(replan), int(horizon), int(knots)
    if REPLAN > H:
        raise SystemExit(f"replanning every {REPLAN} periods needs a horizon of at least that ({H})")
    PLAN["iters"] = int(iters)
    KNOT_T = np.linspace(0.0, H, KNOTS)
    W_ROLL = interp(np.arange(1, H + 1), KNOT_T)
    W_EXEC = interp(np.arange(1, REPLAN + 1), KNOT_T)
    W_SHIFT = interp(REPLAN + KNOT_T, KNOT_T, extrap=WARM == "extrap")


# ------------------------------------------------------------------------------------------------ worlds
def pairs(run=RUN, n_top=10):
    """The (layout, grasp) pairs: the n_top test layouts with HOM's best J, each with its best grasp by the grid's mean
    over seeds 0-2, and the n_top with UHAS's best J, each with its best grasp by UHAS's J at seed 0 (the usability
    rows); a pair chosen twice is kept once."""
    import inner_loop_benchmark_page as BP
    sp = json.loads((UH.RUNS / run / "split.json").read_text())
    test = set(map(tuple, sp["test"]))
    hom = {}
    for (x, y, g), v in BP.j_rows(TASK, UH.D_MM[TASK]).items():
        if (x, y) in test:
            j = float(np.mean([a for a, _h in v]))
            if (x, y) not in hom or j > hom[(x, y)][1]:
                hom[(x, y)] = (g, j)
    uh = {}
    for line in (ROWS / f"usability_{TASK}_{run}.jsonl").read_text().splitlines():
        r = json.loads(line)
        k = (r["x_sep"], r["y_sep"])
        if r["controller"] == "uhas" and (k not in uh or r["J"] > uh[k][1]):
            uh[k] = (r["grasp"], r["J"])
    out, seen = [], set()
    for by, d in (("hom", hom), ("uhas", uh)):
        for k in sorted(d, key=lambda k_: -d[k_][1])[:n_top]:
            if (k, d[k][0]) not in seen:
                seen.add((k, d[k][0]))
                out.append({"x": k[0], "y": k[1], "grasp": int(d[k][0]), "by": by, "J_by": round(d[k][1], 4)})
    return out


def rank_pairs(n_best=50, n_rand=50, seed=0):
    """The ranking test's pairs: the n_best layouts of the task's grid with HOM's best J (mean over seeds 0-2), each
    with that grasp, then n_rand layouts drawn at random from the rest, each with its best grasp by HOM's J."""
    import inner_loop_benchmark_page as BP
    best = {}
    for (x, y, g), v in BP.j_rows(TASK, UH.D_MM[TASK]).items():
        j = float(np.mean([a for a, _h in v]))
        if (x, y) not in best or j > best[(x, y)][1]:
            best[(x, y)] = (g, j)
    order = sorted(best, key=lambda k: (-best[k][1], k))
    rest = order[n_best:]
    pick = [rest[i] for i in sorted(np.random.default_rng(seed).choice(len(rest), n_rand, replace=False))]
    return [{"x": k[0], "y": k[1], "grasp": int(best[k][0]), "by": by, "J_by": round(best[k][1], 4)}
            for by, ks in (("hom_best", order[:n_best]), ("random", pick)) for k in ks]


def rank5_pairs(extra="grid2dff_v4b_sv25r_s34"):
    """The ranking test's 100 layouts (rows/mpc_pairs_<task>_rank.json), each with its best grasp by HOM's J over seeds
    0-4: the grid's seeds 0-2 and seeds 3 and 4 run into the grid rows folder `extra` (owner, 2026-11-15: grasps chosen
    at five placements in place of three). A grasp counts only with all five seeds; the old grasp and J are kept as
    grasp_s3 and J_s3."""
    import hom_landscape as L
    import hom_rows
    import inner_loop_benchmark_page as BP
    J = {k: [a for a, _h in v] for k, v in BP.j_rows(TASK, UH.D_MM[TASK]).items()}
    for x in (BP.GRID.parent / extra / f"{TASK}_d{UH.D_MM[TASK]:g}.jsonl").read_text().splitlines():
        r = hom_rows.rescale_metrics(json.loads(x))
        if r.get("status", "ok") != "ok":
            continue
        d = r["d"]
        J.setdefault((round(d[2] - d[0], 3), round(d[3] - d[5], 3), int(r.get("grasp", 0))), []).append(
            L.track_score(TASK, r))
    by_xy = {}
    for (x, y, g), v in J.items():
        if len(v) == 5:
            by_xy.setdefault((x, y), {})[g] = float(np.mean(v))
    out = []
    for q in json.loads(pairs_path("_rank").read_text())["pairs"]:
        cand = by_xy.get((q["x"], q["y"]), {})
        g = max(cand, key=lambda g_: (cand[g_], -g_)) if cand else q["grasp"]
        out.append({"x": q["x"], "y": q["y"], "grasp": int(g), "by": q["by"],
                    "J_by": round(cand.get(g, float("nan")), 4), "grasp_s3": q["grasp"], "J_s3": q["J_by"]})
    return out


def grid_best():
    """Each layout (x_sep, y_sep) of the task's grid: (its best grasp by HOM's J, that J), J the mean over seeds 0-2."""
    import inner_loop_benchmark_page as BP
    best = {}
    for (x, y, g), v in BP.j_rows(TASK, UH.D_MM[TASK]).items():
        j = float(np.mean([a for a, _h in v]))
        if (x, y) not in best or j > best[(x, y)][1]:
            best[(x, y)] = (g, j)
    return best


def all_pairs():
    """Every layout of the task's grid, each with its best grasp by HOM's J (mean over seeds 0-2): full DIAL's landscape
    over the whole grid (owner, 2026-11-15, the go for full DIAL on every press layout)."""
    best = grid_best()
    return [{"x": k[0], "y": k[1], "grasp": int(best[k][0]), "by": "all", "J_by": round(best[k][1], 4)}
            for k in sorted(best)]


def top_pairs(n=10, ref="anneal", ref_tag="_pdt2", src="_rank"):
    """The ranking test's confirmation pairs: the n best of the task's src pairs under HOM and the n best under the
    planner (rows mpc_hom_<task><src>.jsonl and mpc_<ref>_<task><src><ref_tag>.jsonl, J averaged over seeds 0 and 1),
    to run again at fresh seeds, so that neither controller's first choice is scored at the seeds it was chosen at. For
    src _all also the n best layouts of HOM's grid (J over seeds 0-2, the order of rank_pairs), the ten the search would
    hand to a planner; grid_top marks them."""
    def means(stem):
        d = {}
        for x in (ROWS / f"mpc_{stem}.jsonl").read_text().splitlines():
            r = json.loads(x)
            d.setdefault((r["x_sep"], r["y_sep"], r["grasp"]), []).append(r["J"])
        return {k: float(np.mean(v)) for k, v in d.items()}
    h, d = means(f"hom_{TASK}{src}"), means(f"{ref}_{TASK}{src}{ref_tag}")
    th = sorted(h, key=lambda k: -h[k])[:n]
    td = sorted(d, key=lambda k: -d[k])[:n]
    out = []
    for k in dict.fromkeys(th + td):
        by = "both" if k in th and k in td else "hom_top" if k in th else "planner_top"
        out.append({"x": k[0], "y": k[1], "grasp": int(k[2]), "by": by, "J_hom": round(h.get(k, float("nan")), 4),
                    "J_planner": round(d.get(k, float("nan")), 4)})
    if src == "_all":
        best = grid_best()
        tg = [(k[0], k[1], int(best[k][0])) for k in sorted(best, key=lambda k_: (-best[k_][1], k_))[:n]]
        have = {(o["x"], o["y"], o["grasp"]) for o in out}
        for o in out:
            o["grid_top"] = (o["x"], o["y"], o["grasp"]) in tg
        out += [{"x": k[0], "y": k[1], "grasp": k[2], "by": "grid_top", "grid_top": True,
                 "J_hom": round(h.get(k, float("nan")), 4), "J_planner": round(d.get(k, float("nan")), 4)}
                for k in tg if k not in have]
    return out


def patch_pairs(half=15.0):
    """The lattice neighbourhood of HOM's best layout of the task's grid: every reached layout within half (mm) of it in
    x_sep and in y_sep, each with its best grasp by HOM's J (mean over seeds 0-2), for both controllers' smoothness on
    the same layouts (owner, 2026-11-14: "stability across metrics, tasks, solutions? ... explore this")."""
    import inner_loop_benchmark_page as BP
    best = {}
    for (x, y, g), v in BP.j_rows(TASK, UH.D_MM[TASK]).items():
        j = float(np.mean([a for a, _h in v]))
        if (x, y) not in best or j > best[(x, y)][1]:
            best[(x, y)] = (g, j)
    x0, y0 = max(best, key=lambda k: (best[k][1], k))
    ks = sorted(k for k in best if abs(k[0] - x0) <= half and abs(k[1] - y0) <= half)
    return [{"x": k[0], "y": k[1], "grasp": int(best[k][0]), "by": "patch", "J_by": round(best[k][1], 4)} for k in ks]


def tune_pairs(n=6, j_min=0.05, seed=1):
    """Pairs for choosing a planner's configuration, apart from the ranking test's: n layouts outside rank_pairs with
    HOM's best J above j_min, drawn at random, each with its best grasp."""
    import inner_loop_benchmark_page as BP
    used = {(p["x"], p["y"]) for p in rank_pairs()}
    best = {}
    for (x, y, g), v in BP.j_rows(TASK, UH.D_MM[TASK]).items():
        j = float(np.mean([a for a, _h in v]))
        if (x, y) not in best or j > best[(x, y)][1]:
            best[(x, y)] = (g, j)
    cand = sorted(k for k in best if k not in used and best[k][1] > j_min)
    pick = [cand[i] for i in sorted(np.random.default_rng(seed).choice(len(cand), n, replace=False))]
    return [{"x": k[0], "y": k[1], "grasp": int(best[k][0]), "by": "tune", "J_by": round(best[k][1], 4)} for k in pick]


def worlds_of(prs, seeds, variant=None, prm=None):
    """The pairs' worlds at the seeds; variant: a sensing variant of hom_tasks.VARIANTS (HOM's measurement), default
    the benchmark's (exact, 100 Hz)."""
    scene = HG.Scene(T.SHAPE[TASK], UH.D_MM[TASK])
    want = {(p["x"], p["y"], p["grasp"]): p["by"] for p in prs}
    out = []
    for key, mxy, setup in UH.grid_entries(TASK):
        k = (*UH.xy_of(key), int(key.get("grasp", 0)))
        if k not in want:
            continue
        bm, q_open = scene.prepare(mxy, setup)
        for sd in seeds:
            out.append({"key": key, "seed": sd, "setup": setup, "body_mounts": bm, "q_open": q_open, "by": want[k],
                        "var": HG.variant_world(variant or UH.VARIANT[TASK], sd), **({"prm": prm} if prm else {})})
    return scene, out


def model_scene(scene, model):
    """The scene a planner rolls out in or the plant runs in: the sphere pads (pads), point contact on the v4b tip's
    convex mesh (pt, the Scene's second model; owner, 2026-11-13: "try MPC without the sphere packed model"), or that
    point contact with a torsional friction coefficient of x millimetres (pt_t<x>, torsion_scene)."""
    if model == "pads":
        return scene
    s2 = copy.copy(scene)
    s2.pads_path = scene.pt_path if model == "pt" else torsion_scene(scene.pt_path, float(model[4:]))
    return s2


def torsion_scene(pt_path, mm):
    """The point-contact scene with condim 4 on the three tip meshes: each tip contact resists a torque about its normal
    of up to mm/1000 times its normal force (MuJoCo's torsional friction coefficient, in metres). MuJoCo takes the larger
    of two geoms' coefficients, so every geom gets it; the other contacts keep condim 3, where it has no effect. The
    pads' pinch torque, measured on the rig of 2026-10-01 (an arm of 1.00 mm (N / 1 N)^0.25 at 10 MPa, scaling as
    E^-1/4), corresponds to about 1.4 mm at 1 N and 1.8 mm at 2.5 N for the 2.5 MPa pads. Written once beside pt_path."""
    out = pt_path.with_name(f"{pt_path.stem}__tor{mm:g}mm.xml")
    if out.exists():
        return out
    root = ET.parse(pt_path).getroot()
    for g in root.iter("geom"):                   # the default classes' geoms too
        fr = g.get("friction")
        if fr is not None:
            v = fr.split() + ["0.0001", "0.0001"]
            g.set("friction", f"{v[0]} {mm / 1000:g} {v[2]}")
        if (g.get("name") or "").endswith("_tipgeom"):
            g.set("condim", "4")
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    ET.ElementTree(root).write(tmp)
    os.replace(tmp, out)
    return out


def contact_model(s):
    """argparse type of --model and --plant-model: pads, pt or pt_t<x> (x > 0, mm)."""
    if s in ("pads", "pt") or (s.startswith("pt_t") and float(s[4:]) > 0):
        return s
    raise argparse.ArgumentTypeError(f"{s}: pads, pt or pt_t<mm>")


def model_label(model):
    return {"pads": "pad model", "pt": "point-contact model"}.get(model, f"point contact, {model[4:]} mm torsion")


LABELS = {"hom": "HOM", "ps": "Predictive sampling", "ps_rt": "Predictive sampling, real-time budget", "mppi": "MPPI",
          "cem": "CEM", "icem": "iCEM", "anneal": "DIAL annealing", "spider": "SPIDER",
          "spider_wc": "SPIDER, worst case", "uhas": "UHAS", "hs_ps": "Predictive sampling in HOM's space",
          "hs_mppi": "MPPI in HOM's space", "hs_cem": "CEM in HOM's space", "hs_icem": "iCEM in HOM's space",
          "hs_anneal": "DIAL annealing in HOM's space", "hs_spider": "SPIDER in HOM's space",
          "grad": "Gradient descent, finite differences", "res_ps": "HOM with a predictive-sampling residual",
          "res_mppi": "HOM with an MPPI residual", "res_anneal": "HOM with a DIAL residual",
          "res_grad": "HOM with a gradient-descent residual"}


SHORT = {"hom": "HOM", "ps": "PS", "ps_rt": "PS, real time", "mppi": "MPPI", "cem": "CEM", "icem": "iCEM",
         "anneal": "DIAL", "spider": "SPIDER", "spider_wc": "SPIDER worst case", "uhas": "UHAS", "hs_ps": "PS, HOM space",
         "hs_anneal": "DIAL, HOM space", "hs_icem": "iCEM, HOM space", "res_anneal": "HOM + DIAL residual",
         "grad": "Gradient", "res_ps": "HOM + PS residual"}      # film labels (480 px clips)


def set_friction(B, mu):
    """Every geom's sliding friction set to mu[w] in world w (MuJoCo-Warp's per-world geom_friction), and the batch's
    graphs captured again: a captured graph keeps the arrays it was captured with."""
    fr = np.repeat(B.m.geom_friction[None].astype(np.float32), B.W, 0)
    fr[:, :, 0] = np.asarray(mu, np.float32)[:, None]
    B.wm.geom_friction = wp.array(fr, dtype=wp.vec3)
    B._capture()


# ------------------------------------------------------------------------------------------------ rollouts
class Rollouts:
    """The planner's worlds, n samples times G friction groups per plant world, in the plant's scene and physics;
    world (p G + g) n + k is sample k of plant world p in group g."""

    def __init__(self, scene, chunk, n, frictions=None):
        self.P, self.n = len(chunk), n
        self.mu = tuple(frictions) if frictions else None
        self.G = len(self.mu) if self.mu else 1
        self.B = UH.UBatch(scene, TASK, self.rep(chunk), T.DT_C, gen_seed=0, phys_dt=PLAN_DT)
        if self.mu:
            set_friction(self.B, np.tile(np.repeat(np.asarray(self.mu, np.float32), n), self.P))
        B = self.B
        self.q9 = torch.tensor(np.asarray(B.qadr).reshape(-1), dtype=torch.long, device="cuda")
        self.kp = torch.tensor(B.kp.reshape(-1), dtype=torch.float32, device="cuda")
        self.lim = IU.Limits(B)

    def rep(self, chunk):
        return [w for w in chunk for _ in range(self.G * self.n)]

    def begin(self, chunk):
        self.B.begin(self.rep(chunk), hom=False)

    def load(self, st):
        S = self.B.state()
        r = self.G * self.n
        for k in ("qpos", "qvel", "ctrl", "mocap_pos", "warm"):
            if k in S and k in st:
                S[k].copy_(st[k].repeat_interleave(r, 0))

    def rollout(self, Uh, ctx):
        """The cost of each world's targets Uh (W, H, 9) from the loaded state (W,): the task's terms and the limit
        terms summed over the periods."""
        B, vw = self.B, self.B.vw
        c = torch.zeros(B.W, device="cuda")
        drop = torch.zeros(B.W, dtype=torch.bool, device="cuda")
        for h in range(H):
            vw["ctrl"][:, B.act_t] = Uh[:, h]
            wp.capture_launch(B.graph)
            c += self.period_cost(Uh[:, h], ctx, h, drop)
        return torch.nan_to_num(c, nan=1e6, posinf=1e6)

    def period_cost(self, u, ctx, h, drop):
        """The cost of period h after the physics, for the servo targets u (W, 9) that were applied; drop (W,) is
        updated in place."""
        B, lim = self.B, self.lim
        band = math.radians(COST["q_band_deg"])
        qp = B.vw["qpos"]
        p = qp[:, B.qa:B.qa + 3] - ctx["palm"]
        qo = qp[:, B.qa + 3:B.qa + 7]
        if "aref" in ctx:                         # a cylinder: its axis against the commanded axis (task_refs)
            err = torch.acos((rotate_axis(qo, (0.0, 0.0, 1.0)) * ctx["aref"][:, h]).sum(-1).clamp(-1.0, 1.0))
        else:
            dq = quat_mul(qo, quat_conj(ctx["goal"][:, h]))
            err = 2.0 * torch.asin(dq[:, 1:].norm(dim=-1).clamp(max=1.0))
        if COST["err_db_deg"] > 0:
            err = (err - math.radians(COST["err_db_deg"])).clamp(min=0.0)
        drop |= p[:, 2] < ctx["p0"][:, 2] - COST["drop_m"]
        pt = p + quat_rotate(qo, ctx["pb"]) if "pb" in ctx else p          # the task's reference point
        pref = ctx["pref"][:, h] if "pref" in ctx else ctx["p0"]
        q = qp[:, self.q9]
        dmd = self.kp * (u - q).abs() / lim.tau
        near = torch.minimum(q - lim.lo, lim.hi - q)
        return ((err / math.radians(COST["rot_deg"])) ** 2
                + ((pt - pref).norm(dim=-1) / (1e-3 * COST["pos_mm"])) ** 2
                + COST["pad"] * (B.ft_t < COST["f_on"]).float().sum(-1) + COST["drop"] * drop.float()
                + COST["tau"] * ((dmd - COST["tau_on"]) / (1.0 - COST["tau_on"])).clamp(0.0, 3.0).sum(-1)
                + COST["q"] * (((band - near) / band).clamp(0.0, 1.0) * lim.limited.float()).sum(-1))


def hs_add(u):
    """The added reference relative twists (W, 3, 6) of HOM-space samples u (W, 12): rows HS_FREE, in HS_UNIT."""
    add = torch.zeros((u.shape[0], 3, 6), dtype=torch.float64, device="cuda")
    add[..., HS_FREE] = (u.reshape(-1, 3, 4) * torch.tensor(HS_UNIT, device="cuda")).double()
    return add


class HomRollouts(Rollouts):
    """The planner's worlds under HOM for the HOM-space planners: every period HOM computes the servo targets from
    the measured state, with the sample's rolls and slides added to Eq. (2)'s reference (mode hs), or the sample's
    joint offsets added to HOM's targets (mode res). Before each replanning the plant's physics state and HOM's state
    (reference angles, filtered contact points, the witness's warm start, the orientation feedback's command, the
    measurement buffer, the forces of the last step) are copied in."""

    mode = "hs"

    def begin(self, chunk):
        super().begin(chunk)
        self.B.v_add_rows = HS_FREE
        dt = self.B.prm.dt
        self.B.clock = self.B.clock + [self.B.clock[-1] + (i + 1) * dt for i in range(64)]   # rollouts past the end

    def load(self, st):
        super().load(st)
        pl, B, r = st["plant"], self.B, self.G * self.n
        for c in ("theta_ref", "c1", "ap_x", "R_bm", "p_bm", "q_cmd"):
            v = getattr(pl, c, None)
            setattr(B, c, None if v is None else v.repeat_interleave(r, 0).clone())
        B.n_ra = pl.n_ra.repeat_interleave(r, 0).clone()
        for name in ("ft_t", "fo_t", "cop_t"):
            getattr(B, name).copy_(getattr(pl, name).repeat_interleave(r, 0))
        B.buf = {k_: v.repeat_interleave(r, 1).clone() for k_, v in pl.buf.items()}
        B.k, B.post_off = pl.k, pl.post_off

    def rollout(self, Uh, ctx):
        """The cost of each world's HOM-space plan Uh (W, H, 12) from the loaded state."""
        B = self.B
        c = torch.zeros(B.W, device="cuda")
        drop = torch.zeros(B.W, dtype=torch.bool, device="cuda")
        for h in range(H):
            if self.mode == "res":
                B.q_add = Uh[:, h]
            else:
                B.v_add = hs_add(Uh[:, h])
            B.tick(None)
            c += self.period_cost(B.vw["ctrl"][:, B.act_t], ctx, h, drop)
        B.v_add = B.q_add = None
        return torch.nan_to_num(c, nan=1e6, posinf=1e6)


# ------------------------------------------------------------------------------------------------ planners
class MPC:
    """One planner for P plant worlds: the nominal knots (P, K, 9) and the update rule of its method."""

    def __init__(self, method, P, lo, hi, n, G, seed, D=9, sigma=None, rate_scale=None):
        self.m, self.P, self.n, self.G, self.D = method, P, n, G, D
        self.sig = PLAN["sigma"] if sigma is None else sigma
        self.rate_scale = math.radians(COST["rate_dps"]) if rate_scale is None else rate_scale
        self.lo, self.hi = lo, hi
        self.gen = torch.Generator(device="cuda")
        self.gen.manual_seed(1000 + seed)
        self.nom, self.elites = None, None
        self.r_prev = None              # the executed targets' rate at the end of the last segment (P, D), per s
        self.ar = torch.arange(P, device="cuda")

    def noise(self, n, scale):
        e = torch.randn((self.P, n, KNOTS, self.D), generator=self.gen, device="cuda")
        if self.m == "icem":
            a = PLAN["ar_alpha"]
            for k in range(1, KNOTS):
                e[:, :, k] = a * e[:, :, k - 1] + math.sqrt(1.0 - a * a) * e[:, :, k]
        return e * scale

    def constrain(self, S, u):
        S = torch.maximum(torch.minimum(S, self.hi), self.lo)
        S[:, :, 0] = u[:, None]
        return S

    def rate_cost(self, S):
        """The knots' rates (rate), their changes from one knot gap to the next and from the executed rate to the
        first gap (acc), and the sign flips of each target's rate at the replanning (rev), in the rate scale."""
        gap = float(KNOT_T[1] - KNOT_T[0]) * T.DT_C
        r = torch.diff(S, dim=2) / gap / self.rate_scale
        c = COST["rate"] * (r ** 2).sum((-1, -2))
        if COST["acc"] > 0 and r.shape[2] > 1:
            c = c + COST["acc"] * (torch.diff(r, dim=2) ** 2).sum((-1, -2))
        if self.r_prev is not None and (COST["acc"] > 0 or COST["rev"] > 0):
            rp, r0 = (self.r_prev / self.rate_scale)[:, None], r[:, :, 0]
            c = c + COST["acc"] * ((r0 - rp) ** 2).sum(-1) + COST["rev"] * (-(r0 * rp)).clamp(min=0.0).sum(-1)
        return c

    def evaluate(self, R, st, S, ctx):
        """Costs (P, n) of the samples S (P, n, K, 9), each the worst over the friction groups."""
        P, n = S.shape[:2]
        Uh = torch.einsum("hk,pnkj->pnhj", W_ROLL, S)
        Uh = Uh[:, None].expand(P, self.G, n, H, self.D).reshape(P * self.G * n, H, self.D)
        R.load(st)
        c = R.rollout(Uh, ctx).reshape(P, self.G, n).max(1).values
        return c + self.rate_cost(S)

    def pick(self, S, idx):
        return S[self.ar, idx]

    def plan(self, R, st, u, ctx):
        """The knots (P, K, 9) to execute from the plant state st; u (P, 9) the targets applied now."""
        if self.nom is None:
            self.nom = u[:, None].expand(self.P, KNOTS, self.D).clone()
        else:
            self.nom = torch.einsum("ik,pkj->pij", W_SHIFT, self.nom)
            if self.elites is not None:
                self.elites = torch.einsum("ik,pekj->peij", W_SHIFT, self.elites)
        m, I, n, sig = self.m, PLAN["iters"], self.n, self.sig
        nom0 = self.constrain(self.nom[:, None], u)
        if m == "ps":
            best_c = torch.full((self.P,), float("inf"), device="cuda")
            best = nom0[:, 0].clone()
            for i in range(I):
                S = self.constrain(self.nom[:, None] + self.noise(n, sig), u)
                if i == 0:
                    S[:, 0] = nom0[:, 0]
                c = self.evaluate(R, st, S, ctx)
                cb, ib = c.min(1)
                best = torch.where((cb < best_c)[:, None, None], self.pick(S, ib), best)
                best_c = torch.minimum(best_c, cb)
            self.nom = best
            return best, {"iters": I, "rollouts": I * n, "cost": best_c}
        if m in ("mppi", "anneal"):
            kk = torch.arange(KNOTS, device="cuda", dtype=torch.float32)[:, None]
            for i in range(I):
                scale = sig if m == "mppi" else sig * torch.exp(-i / (PLAN["beta1"] * I) - (KNOTS - 1 - kk)
                                                                / (PLAN["beta2"] * KNOTS))
                S = self.constrain(self.nom[:, None] + self.noise(n, scale), u)
                S[:, 0] = self.constrain(self.nom[:, None], u)[:, 0]
                c = self.evaluate(R, st, S, ctx)
                lam = PLAN["lam_rel"] * (c.std(1, keepdim=True) + 1e-9)
                w = torch.softmax(-(c - c.min(1, keepdim=True).values) / lam, 1)
                self.nom = self.constrain(torch.einsum("pn,pnkj->pkj", w, S)[:, None], u)[:, 0]
            return self.nom, {"iters": I, "rollouts": I * n, "cost": c.min(1).values}
        if m in ("cem", "icem"):
            mean = nom0[:, 0].clone()
            std = torch.full_like(mean, sig)
            ne = max(2, int(round(PLAN["elite_frac"] * n)))
            best_c = torch.full((self.P,), float("inf"), device="cuda")
            best = mean.clone()
            for i in range(I):
                S = self.constrain(mean[:, None] + self.noise(n, std[:, None]), u)
                S[:, 0] = self.constrain(mean[:, None], u)[:, 0]
                if m == "icem" and i == 0 and self.elites is not None:
                    k_ = min(self.elites.shape[1], n - 1)
                    S[:, n - k_:] = self.constrain(self.elites[:, :k_], u)
                c = self.evaluate(R, st, S, ctx)
                idx = torch.argsort(c, 1)
                cb = c.gather(1, idx[:, :1])[:, 0]
                best = torch.where((cb < best_c)[:, None, None], self.pick(S, idx[:, 0]), best)
                best_c = torch.minimum(best_c, cb)
                el = S[self.ar[:, None], idx[:, :ne]]
                mom = PLAN["momentum"]
                mean = (1 - mom) * el.mean(1) + mom * mean
                std = ((1 - mom) * el.std(1) + mom * std).clamp(PLAN["sigma_min"], sig)
                if m == "icem":
                    self.elites = S[self.ar[:, None], idx[:, :PLAN["keep"]]].clone()
            self.nom = best if m == "icem" else self.constrain(mean[:, None], u)[:, 0]
            return self.nom, {"iters": I, "rollouts": I * n, "cost": best_c}
        if m == "grad":
            # gradient descent with finite differences (MJPC's gradient planner, Howell et al. 2022): iterations in pairs
            # of batches of n rollouts. The first batch perturbs the free knots by +-eps along each coordinate and, with
            # the rollouts left, along random unit directions; the gradient is the least-squares fit of the central
            # differences. The second batch is a line search along the negative gradient at n step sizes from 0 (the
            # plan itself) to PLAN grad_step_max, log-spaced; the cheapest becomes the plan.
            if I % 2 or n < 2 * (KNOTS - 1) * self.D:
                raise SystemExit(f"grad: an even iteration count, and n >= {2 * (KNOTS - 1) * self.D}")
            x, P, D, eps = nom0[:, 0].clone(), self.P, self.D, PLAN["grad_eps"]
            nf, npair = (KNOTS - 1) * D, n // 2
            dirs = torch.randn((P, npair, nf), generator=self.gen, device="cuda")
            dirs[:, :nf] = torch.eye(nf, device="cuda")
            dirs = dirs / dirs.norm(dim=-1, keepdim=True)
            pad = torch.zeros((P, npair, 1, D), device="cuda")
            Dk = torch.cat([pad, dirs.reshape(P, npair, KNOTS - 1, D)], 2)            # (P, npair, K, D), first knot 0
            steps = torch.cat([torch.zeros(1, device="cuda"),
                               torch.logspace(math.log10(PLAN["grad_step_min"]), math.log10(PLAN["grad_step_max"]), n - 1,
                                              device="cuda")])
            for i in range(I // 2):
                S = self.constrain(torch.cat([x[:, None] + eps * Dk, x[:, None] - eps * Dk], 1), u)
                if n % 2:
                    S = torch.cat([S, x[:, None]], 1)
                c = self.evaluate(R, st, S, ctx)
                dc = (c[:, :npair] - c[:, npair:2 * npair]) / (2.0 * eps)                # (P, npair) ~ dirs @ g
                g = torch.linalg.lstsq(dirs, dc[..., None]).solution[..., 0]             # (P, nf)
                d = -(g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)).reshape(P, 1, KNOTS - 1, D)
                C = x[:, None].expand(P, n, KNOTS, D).clone()
                C[:, :, 1:] = C[:, :, 1:] + steps[None, :, None, None] * d
                C = self.constrain(C, u)
                cl = self.evaluate(R, st, C, ctx)
                j = cl.argmin(1)
                x = self.pick(C, j)
                best_c = cl.gather(1, j[:, None])[:, 0]
            self.nom = x
            return x, {"iters": I, "rollouts": I * n, "cost": best_c}
        if m in ("spider", "spider_wc"):
            prof = torch.logspace(math.log10(PLAN["sp_first"]), math.log10(PLAN["sp_last"]), KNOTS,
                                  device="cuda")[:, None]
            ns = torch.ones(n, 1, 1, device="cuda")
            ns[0] = 0.0
            n_ex = max(1, int(n * PLAN["sp_exploit"]))
            ns[n - n_ex:] = PLAN["sp_exploit_scale"]
            beta = PLAN["sp_final"] ** (1.0 / I)
            top = max(1, int(PLAN["sp_top"] * n))
            active = torch.ones(self.P, dtype=torch.bool, device="cuda")
            it = 0
            for i in range(I):
                S = self.constrain(self.nom[:, None] + self.noise(n, sig * prof) * ns * beta ** i, u)
                c = self.evaluate(R, st, S, ctx)
                r = -c / H
                tv, ti = r.topk(top, 1)
                z = (tv - tv.mean(1, keepdim=True)) / (tv.std(1, keepdim=True) + 1e-2)
                w = torch.softmax(z / PLAN["sp_temp"], 1)
                new = torch.einsum("pt,ptkj->pkj", w, S[self.ar[:, None], ti])
                self.nom = torch.where(active[:, None, None], self.constrain(new[:, None], u)[:, 0], self.nom)
                it += 1
                active &= (r.max(1).values - r[:, 0]) >= PLAN["sp_improve"]
                if not bool(active.any()):
                    break
            return self.nom, {"iters": it, "rollouts": it * n * self.G, "cost": c.min(1).values}
        raise ValueError(f"unknown method {m}")


# ------------------------------------------------------------------------------------------------ episodes
def ref_table(task):
    """The commanded rotation from the task's start after each period (n + 1, 4): uhas_sr2.Tracker.advance's
    integration of the commanded angular velocity."""
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")
    out = [q]
    for i in range(task.n):
        w = task.w[i]
        ang = w.norm()
        dq = quat_from_axis_angle(w / ang.clamp_min(1e-12), ang * T.DT_C)
        q = quat_mul(dq, q)
        q = q / q.norm()
        out.append(q)
    return torch.stack(out)


def task_refs(B, task):
    """The task's references for the planners' cost from the plant's state at the task's start, per plant world and
    period (n + 1 rows: the commanded state after 0..n periods): the orientation goal (spheres; ref_table times the start
    orientation) or the cylinder's commanded axis aref (the start axis turned by -theta about palm x, as
    uhas_sr2.Tracker.rot_error), and pref, the reference point's position from the palm. pb is that point in the object's
    frame: the pinch, the midpoint of the thumb's and the index's centres of pressure, for the cylinders; the centre (0)
    for the spheres, whose reference moves along the thumb's pad at r w in the roll (uhas_sr2.Tracker.advance)."""
    vw, P = B.vw, B.W
    palm = vw["mocap_pos"][:, B.mo_palm].float()
    p0 = vw["qpos"][:, B.qa:B.qa + 3].float() - palm
    q0 = vw["qpos"][:, B.qa + 3:B.qa + 7].float()
    n1 = task.n + 1
    if task.pinch:
        F = B.ft_t.float()
        cop = B.cop_t.float() / F.clamp_min(1e-12)[..., None]
        mid = 0.5 * (cop[:, 0] + cop[:, 1]) - palm
        ok = (F[:, :2] > 0).all(-1)[:, None]
        pb = torch.where(ok, quat_rotate(quat_conj(q0), mid - p0), torch.zeros_like(p0))
        th = torch.cat([task.th, task.th[-1:]])                              # the commanded angle at each period's start
        c, s = torch.cos(-th)[:, None], torch.sin(-th)[:, None]
        a0 = rotate_axis(q0, (0.0, 0.0, 1.0))[None]
        aref = torch.stack([a0[..., 0].expand(n1, P), c * a0[..., 1] - s * a0[..., 2], s * a0[..., 1] + c * a0[..., 2]],
                           -1)
        return {"aref": aref, "pref": (p0 + quat_rotate(q0, pb))[None].expand(n1, P, 3)}, pb
    goal = quat_mul(ref_table(task)[:, None].expand(n1, P, 4), q0[None].expand(n1, P, 4))
    pref = p0[None].expand(n1, P, 3).clone()
    if task.roll:
        Wsum = torch.cat([torch.zeros((1, 3), device="cuda"), torch.cumsum(task.w * T.DT_C, 0)])        # (n1, 3)
        n0 = B.n_out[:, 0].float()
        pref = pref + torch.cross(Wsum[:, None].expand(n1, P, 3), (-B.scene.r * n0)[None].expand(n1, P, 3), dim=-1)
    return {"goal": goal, "pref": pref}, torch.zeros_like(p0)


# The planners' measured state (owner, 2026-11-13: "how well we can do with noisy in hand pose estimation"): with
# MEAS["delay"] > 0 a plan starts from the plant's state that many periods earlier (the tag tracker's 40 ms is 4), with
# MEAS["pos_mm"] and MEAS["rot_deg"] of Gaussian noise on the object's position and orientation (hom_tasks.VARIANTS
# "notip": 0.5 mm, 1 deg); the plan is applied from now. The joint-space planners only.
MEAS = {"delay": 0, "pos_mm": 0.0, "rot_deg": 0.0}


def measured(hist, B, gen):
    """The planner's start state: the oldest of the kept states, the object's pose noised."""
    st = {k_: v.clone() for k_, v in hist[0].items()}
    W, qa = B.W, B.qa
    if MEAS["pos_mm"] > 0:
        st["qpos"][:, qa:qa + 3] += torch.randn((W, 3), generator=gen, device="cuda") * (MEAS["pos_mm"] * 1e-3)
    if MEAS["rot_deg"] > 0:
        ax = torch.nn.functional.normalize(torch.randn((W, 3), generator=gen, device="cuda"), dim=-1)
        ang = torch.randn(W, generator=gen, device="cuda") * math.radians(MEAS["rot_deg"])
        q = quat_mul(quat_from_axis_angle(ax, ang), st["qpos"][:, qa + 3:qa + 7].float())
        st["qpos"][:, qa + 3:qa + 7] = q / q.norm(dim=-1, keepdim=True)
    return st


def episode(B, chunk, method, R=None, policy=None, F=None, seed=0, trace=None):
    """One batch of plant worlds through the episode: HOM to the task, then HOM, the policy or the planner. Returns
    the rows of hom_gpu.Batch.score, the usability metrics (inner_loop_usability.metrics) and the planner's record.
    trace (a list): world 0's positions, mocap positions, time and period index before each period, for films."""
    B.begin(chunk, record=True, hom=True)
    if method != "hom" and float(B.cal_q.abs().max()) > 0:
        raise RuntimeError("calibration offsets in the plant: the planner's and the policy's targets bypass them")
    P, vw, k0 = B.W, B.vw, B.k
    task = UH.Task(B)
    plan_rec = {"walls": [], "iters": [], "rollouts": []}
    hs = method in HS_METHODS
    res = method in RES_METHODS
    il = method == "ilqr"
    if method in METHODS or hs or res or il:
        R.begin(chunk)
        r = R.G * R.n
        palm = vw["mocap_pos"][:, B.mo_palm].float().clone()
        p0 = vw["qpos"][:, B.qa:B.qa + 3].float() - palm
        refs, pb = task_refs(B, task)
        ctx0 = {"palm": palm.repeat_interleave(r, 0), "p0": p0.repeat_interleave(r, 0), "pb": pb.repeat_interleave(r, 0)}
        if hs:
            B.v_add_rows = HS_FREE
            bd = torch.full((12,), PLAN_HS["bound"], device="cuda")
            mpc = MPC(method[3:], P, -bd, bd, R.n, R.G, seed, D=12, sigma=PLAN_HS["sigma"], rate_scale=PLAN_HS["rate"])
            u = torch.zeros((P, 12), device="cuda")
        elif res:
            bd = torch.full((9,), PLAN_RES["bound"], device="cuda")
            mpc = MPC(method[4:], P, -bd, bd, R.n, R.G, seed, D=9, sigma=PLAN_RES["sigma"],
                      rate_scale=math.radians(PLAN_RES["rate_dps"]))
            u = torch.zeros((P, 9), device="cuda")
        elif il:
            mpc = R
            u = vw["ctrl"][:, B.act_t].float().clone()
        else:
            mpc = MPC(method, P, R.lim.lo, R.lim.hi, R.n, R.G, seed)
            u = vw["ctrl"][:, B.act_t].float().clone()
    elif method == "uhas":
        from tensordict import TensorDict
        mounts = torch.tensor(np.array([[wd_["body_mounts"][f] for f in HG.FINGERS] for wd_ in chunk]),
                              dtype=torch.float32, device="cuda")
        Cc = U.SphereController(F, mounts)
        tr = UH.Tracker(P, "cuda")
        tr.reset(torch.arange(P, device="cuda"), B, Cc, F, mounts, B.n_out[:, 0].float())
    Q, Cm, A = [], [], []
    tau = 0
    hist = []
    gen_m = torch.Generator(device="cuda")
    gen_m.manual_seed(2000 + seed)
    with torch.inference_mode():
        while not B.done():
            if trace is not None:
                trace.append((vw["qpos"][0].cpu().numpy().copy(), vw["mocap_pos"][0].cpu().numpy().copy(),
                              B.clock[min(B.k, len(B.clock) - 1)], B.k))
            if method == "hom":
                B.tick(None)
            elif method == "uhas":
                pol, cri = tr.observe(task, B, Cc, F, mounts)
                a = policy(TensorDict({"policy": pol, "critic": cri}, batch_size=[P], device="cuda")).clamp(-1, 1)
                B.tick(Cc.solve(a))
                tr.last_a = a
                tr.tau += 1
                tr.advance(task, B.scene.r)
            else:
                if MEAS["delay"] > 0 or MEAS["pos_mm"] > 0 or MEAS["rot_deg"] > 0:
                    hist.append({k_: v.clone() for k_, v in B.state().items() if k_ != "body_pos"})
                    hist = hist[-(MEAS["delay"] + 1):]
                if tau % REPLAN == 0:
                    t0 = time.perf_counter()
                    i = torch.arange(tau + 1, tau + H + 1, device="cuda").clamp(max=task.n)
                    ctx = dict(ctx0, **{k_: v[i].transpose(0, 1).repeat_interleave(r, 0) for k_, v in refs.items()})
                    st = (dict(B.state(), plant=B) if hs or res else measured(hist, B, gen_m) if hist else B.state())
                    knots, inf = mpc.plan(R, st, u, ctx)
                    seg = knots if il else torch.einsum("hk,pkj->phj", W_EXEC, knots)
                    if REPLAN > 1:
                        mpc.r_prev = (seg[:, -1] - seg[:, -2]) / T.DT_C
                    torch.cuda.synchronize()
                    plan_rec["walls"].append(time.perf_counter() - t0)
                    plan_rec["iters"].append(inf["iters"])
                    plan_rec["rollouts"].append(inf["rollouts"])
                u = mpc.control(B, tau % REPLAN) if il else seg[:, tau % REPLAN]
                if hs:
                    B.v_add = hs_add(u)
                    B.tick(None)
                elif res:
                    B.q_add = u
                    B.tick(None)
                else:
                    B.tick(u.reshape(P, 3, 3))
            Q.append(vw["qpos"][:, B.qadr_t].reshape(P, 9).float().clone())
            Cm.append(IU.targets(B).reshape(P, 9).clone())
            A.append(wp.to_torch(B.wd.actuator_force)[:, B.act_t].float().clone())
            tau += 1
    B.v_add = B.q_add = None
    out = B.finish()
    M = IU.metrics(B, k0, torch.stack(Q), torch.stack(Cm), torch.stack(A), IU.Limits(B), B.prm.dt,
                   stride=R.S if method == "ilqr" else 1)       # iLQR holds its targets for R.S periods
    return out, M, plan_rec


def row_key(r):
    return (r["controller"], r["x_sep"], r["y_sep"], r["grasp"], r["seed"], r["plant_mu"])


def cmd_run(a):
    global PLAN_DT, WARM
    import hom_landscape as L
    set_task(a.task)
    a.variant = a.variant or UH.VARIANT[TASK]
    WARM = a.warm
    configure(a.replan, a.horizon, a.knots, a.iters)
    if (a.replan, a.horizon, a.knots, a.iters) != (10, 30, 3, 4):
        a.tag = f"_r{a.replan}h{a.horizon}k{a.knots}i{a.iters}" + a.tag
    if a.n != 64:
        a.tag = f"_n{a.n}" + a.tag
    if a.plan_dt is not None:
        PLAN_DT = a.plan_dt
        a.tag = f"_pdt{1e3 * a.plan_dt:g}" + a.tag
    sh = "" if WARM == "hold" else f"_{WARM}"          # the reversal options (owner, 2026-11-13)
    for flag, d, k, nm in ((a.lam_rel, PLAN, "lam_rel", "lam"), (a.sigma, PLAN, "sigma", "sig"),
                           (a.rate_w, COST, "rate", "rw"), (a.acc_w, COST, "acc", "acc"), (a.rev_w, COST, "rev", "rev"),
                           (a.err_db, COST, "err_db_deg", "db")):
        if flag is not None:
            d[k] = flag
            sh += f"_{nm}{flag:g}"
    a.tag = sh + a.tag
    if a.model != "pads":
        a.tag = f"_{a.model}" + a.tag
    ILQ = None
    if a.method == "ilqr":
        import inner_loop_ilqr as ILQ
        if a.il_step is not None:
            ILQ.IL["step"] = a.il_step
            a.tag = f"_st{a.il_step}" + a.tag
        if a.il_fb:
            ILQ.IL["fb"] = True
            a.tag = "_fb" + a.tag
        if a.il_pad:
            ILQ.IL["pad"] = True
            a.tag = "_pad" + a.tag
    UH.ATT_GAIN = a.att_gain if a.method == "hom" else 0.0
    UH.ATT_AXIS = bool(a.att_axis)
    if UH.ATT_GAIN > 0:
        a.tag = f"_att{a.att_gain:g}" + ("ax" if a.att_axis else "") + a.tag
    if a.hs_sigma is not None:
        PLAN_HS["sigma"] = a.hs_sigma
        a.tag = f"_hs{a.hs_sigma:g}" + a.tag
    if a.pose_tau is not None or a.meas_every > 1:
        if a.method != "hom":
            raise SystemExit("--pose-tau/--meas-every: HOM's measured pose (hom_gpu.POSE_FILTER)")
        HG.POSE_FILTER = {"tau": a.pose_tau, "every": a.meas_every}
        a.tag = (f"_pf{a.pose_tau:g}" if a.pose_tau is not None else "_hold") + (f"e{a.meas_every}" if a.meas_every > 1
                                                                                 else "") + a.tag
    if a.plan_delay or a.plan_noise:
        if a.method not in METHODS:
            raise SystemExit("--plan-delay/--plan-noise: the joint-space planners only")
        MEAS.update(delay=a.plan_delay, pos_mm=0.5 if a.plan_noise else 0.0, rot_deg=1.0 if a.plan_noise else 0.0)
        a.tag = f"_d{a.plan_delay}" + ("n" if a.plan_noise else "") + a.tag
    if a.variant != UH.VARIANT[TASK]:
        if a.method != "hom":
            raise SystemExit("--variant: only HOM reads the measured state; the planners and UHAS read the plant's")
        a.tag = f"_{a.variant}" + a.tag
    prs = json.loads(pairs_path(a.pairs).read_text())["pairs"]
    if a.limit_pairs:
        prs = prs[:a.limit_pairs]
    prm = {"f_tripod": a.f_tripod} if a.f_tripod is not None else None
    if prm:
        a.tag = f"_ft{a.f_tripod:g}" + a.tag
    if a.prm_shared:          # one set of HOM's forces and contact modes per task, CEM-searched under J on 11-11
        sh_ = SHARED / f"{TASK}_d{UH.D_MM[TASK]:g}_pick.json"
        prm = {**(prm or {}), **json.loads(sh_.read_text())["pick"]["prm"]}
        a.tag = "_prmsh" + a.tag
    if a.prm:
        prm = {**(prm or {}), **json.loads(a.prm)}
        a.tag = "_prm" + "".join(f"{k.replace('_', '')}{v:g}" for k, v in json.loads(a.prm).items()) + a.tag
    hp = dict(kv.split("=") for kv in a.hom_param)
    for k, v in hp.items():
        a.tag = f"_hp{k.replace('_', '')}{float(v):g}" + a.tag
    UH.PREVIEW_S = a.preview
    if a.preview:
        a.tag = f"_pv{1e3 * a.preview:g}" + a.tag
    if (hp or a.preview) and a.method in METHODS + ("uhas",):
        raise SystemExit("--hom-param/--preview: methods that run HOM in the task (hom, hs_*, res_*)")
    scene, worlds = worlds_of(prs, a.seeds, a.variant, prm)
    tag = ("" if a.plant_mu == 1.0 else f"_mu{a.plant_mu:g}") + ("" if a.plant_model == "pads"
                                                                 else f"_plant_{a.plant_model}")
    out = ROWS / f"mpc_{a.method}_{TASK}{a.pairs}{tag}{a.tag}.jsonl"
    done = set()
    if out.exists():
        done = {row_key(json.loads(x)) for x in out.read_text().splitlines() if x.strip()}
    todo = [w for w in worlds if (a.method, *UH.xy_of(w["key"]), int(w["key"].get("grasp", 0)), w["seed"],
                                  a.plant_mu) not in done]
    PL = inner_loop_plant.describe()
    print(f"{a.method}: {len(todo)} of {len(worlds)} worlds to run ({len(prs)} pairs, seeds {a.seeds}), plant friction "
          f"{a.plant_mu:g}; plant {PL['plant']}, scenes {PL['scenes']}; plant contact {a.plant_model}, planner model "
          f"{a.model}", flush=True)
    if not todo:
        return 0
    Wc = min(a.chunk, len(todo))
    policy, ckpt, F, R = None, None, None, None
    B = None
    for c0 in range(0, len(todo), Wc):
        chunk = todo[c0:c0 + Wc]
        pad = chunk + [chunk[-1]] * (Wc - len(chunk))
        if B is None:
            B = UH.UBatch(model_scene(scene, a.plant_model), TASK, pad, pad[0]["var"]["dt"], gen_seed=0)
            if a.plant_mu != 1.0:
                set_friction(B, np.full(Wc, a.plant_mu, np.float32))
            if a.method in METHODS:
                fr = PLAN["frictions"] if a.method == "spider_wc" else None
                n = a.n // len(fr) if fr else a.n
                R = Rollouts(model_scene(scene, a.model), pad, n, fr)
            elif a.method in HS_METHODS + RES_METHODS:
                R = HomRollouts(model_scene(scene, a.model), pad, a.n)
                R.mode = "res" if a.method in RES_METHODS else "hs"
            elif a.method == "ilqr":
                R = ILQ.Planner(sys.modules[__name__], model_scene(scene, a.model), pad, PLAN["iters"])
            elif a.method == "uhas":
                F = U.Fingers(B.m, "cuda")
                policy, ckpt = UH.load_policy(UH.RUNS / a.run)
        for B_ in (B, R.B if isinstance(R, HomRollouts) else None):
            for k, v in (hp.items() if B_ is not None else ()):
                if not hasattr(B_.prm, k):
                    raise SystemExit(f"--hom-param {k}: not a field of hom.HomParams")
                setattr(B_.prm, k, type(getattr(B_.prm, k))(float(v)))
        t0 = time.time()
        res, M, pr = episode(B, pad, a.method, R, policy, F, seed=int(chunk[0]["seed"]) + c0)
        wall = time.time() - t0
        rows = []
        for i, (wd_, r) in enumerate(zip(chunk, res)):
            x, y = UH.xy_of(wd_["key"])
            row = {"controller": a.method, "x_sep": x, "y_sep": y, "grasp": int(wd_["key"].get("grasp", 0)),
                   "seed": wd_["seed"], "by": wd_["by"], "plant_mu": a.plant_mu, "J": L.track_score(TASK, r),
                   "task": TASK, "pairs": a.pairs or "benchmark", **PL}
            row.update({k: r.get(k) for k in IU.KEEP if k != "J"})
            row.update({k: round(float(M[k][i]), 4) for k in IU.METRICS})
            row.update({"model": a.model, "plant_model": a.plant_model, "att_gain": UH.ATT_GAIN, "variant": a.variant,
                        "meas": dict(MEAS),
                        "pose_filter": HG.POSE_FILTER,
                        "space": "hom" if a.method in HS_METHODS else "residual" if a.method in RES_METHODS else "joint",
                        "preview_s": a.preview, "hom_param": hp, "prm": prm})
            if ILQ is not None:
                row["ilqr"] = dict(ILQ.IL)
                row["ilqr_stats"] = R.stats(i)
                row["metric_stride"] = ILQ.IL["step"]
            row.update({"chunk_wall_s": round(wall, 2), "chunk_worlds": Wc, "run": a.run if a.method == "uhas" else None,
                        "checkpoint": Path(ckpt).name if ckpt else None})
            if pr["walls"]:
                row.update({"replan_s_mean": round(float(np.mean(pr["walls"])), 4), "replans": len(pr["walls"]),
                            "rollouts_per_replan": round(float(np.mean(pr["rollouts"])), 1),
                            "iters_mean": round(float(np.mean(pr["iters"])), 2), "n": R.n, "groups": R.G,
                            "plan": {"iters": PLAN["iters"], "sigma": (PLAN_HS if a.method in HS_METHODS else PLAN_RES
                                                                       if a.method in RES_METHODS else PLAN)["sigma"]},
                            "replan": REPLAN, "H": H,
                            "knots": KNOTS, "plan_dt": PLAN_DT,
                            "shaping": {"warm": WARM, "lam_rel": PLAN["lam_rel"], "sigma": PLAN["sigma"],
                                        "rate_w": COST["rate"], "acc_w": COST["acc"], "rev_w": COST["rev"],
                                        "err_db_deg": COST["err_db_deg"]}})
            rows.append(row)
        with open(out, "a") as fh:
            for r_ in rows:
                fh.write(json.dumps(r_, default=float) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        js = [r_["J"] for r_ in rows]
        print(f"{a.method}: {c0 + len(chunk)} of {len(todo)} worlds; chunk {wall:.0f} s"
              + (f", replanning {np.mean(pr['walls']):.2f} s each ({len(pr['walls'])})" if pr["walls"] else "")
              + f"; J mean {np.mean(js):.3f}, held {np.mean([bool(r_['held']) for r_ in rows]):.2f}", flush=True)
    print(f"-> {out.relative_to(ROOT)}", flush=True)
    return 0


def cmd_pairs(a):
    set_task(a.task)
    if a.rule == "rank":
        prs = rank_pairs()
        rule = ("the 50 layouts of the task's grid with HOM's best J (mean over seeds 0-2), each with that grasp, then 50 "
                "layouts drawn at random from the rest (numpy seed 0), each with its best grasp by HOM's J")
        for name, sub in (("_rank", prs), ("_rank20", prs[0:50:5] + prs[50:100:5])):
            pairs_path(name).write_text(json.dumps({"task": TASK, "pairs": sub, "rule": rule if name == "_rank" else
                                                    "every fifth pair of each half of " + pairs_path("_rank").name},
                                                   indent=1))
            print(f"{len(sub)} pairs -> {pairs_path(name).relative_to(ROOT)}")
        return 0
    if a.rule == "patch":
        prs = patch_pairs()
        pairs_path("_patch").write_text(json.dumps({"task": TASK, "pairs": prs, "rule": patch_pairs.__doc__}, indent=1))
        print(f"{len(prs)} pairs -> {pairs_path('_patch').relative_to(ROOT)}")
        return 0
    if a.rule == "top":
        prs = top_pairs(src=a.src)
        name = "_top" if a.src == "_rank" else a.src + "top"
        pairs_path(name).write_text(json.dumps({"task": TASK, "pairs": prs, "src": a.src, "rule": top_pairs.__doc__},
                                               indent=1))
        print(f"{len(prs)} pairs -> {pairs_path(name).relative_to(ROOT)}")
        return 0
    if a.rule == "rank5":
        prs = rank5_pairs()
        pairs_path("_rank5").write_text(json.dumps({"task": TASK, "pairs": prs, "rule": rank5_pairs.__doc__}, indent=1))
        print(f"{len(prs)} pairs, {sum(p['grasp'] != p['grasp_s3'] for p in prs)} with another grasp -> "
              f"{pairs_path('_rank5').relative_to(ROOT)}")
        return 0
    if a.rule == "all":
        prs = all_pairs()
        pairs_path("_all").write_text(json.dumps({"task": TASK, "pairs": prs, "rule": all_pairs.__doc__}, indent=1))
        print(f"{len(prs)} pairs -> {pairs_path('_all').relative_to(ROOT)}")
        return 0
    if a.rule == "tune":
        prs = tune_pairs()
        pairs_path("_tune").write_text(json.dumps({"task": TASK, "pairs": prs, "rule": tune_pairs.__doc__}, indent=1))
        print(f"{len(prs)} pairs -> {pairs_path('_tune').relative_to(ROOT)}")
        return 0
    if TASK != "twist":
        raise SystemExit("the benchmark's pairs (HOM's and UHAS's ten best) exist for the twist only: --rule rank")
    prs = pairs(a.run, a.top)
    path = ROWS / "mpc_pairs_twist.json"
    path.write_text(json.dumps({"run": a.run, "top": a.top, "pairs": prs,
                                "rule": "the ten test layouts with HOM's best J (best grasp by the grid's mean over "
                                        "seeds 0-2) and the ten with UHAS's best J (best grasp by UHAS's J at seed 0)"},
                               indent=1))
    for p in prs:
        print(p)
    print(f"{len(prs)} pairs -> {path.relative_to(ROOT)}")
    return 0


def cmd_timing(a):
    """Wall time of a rollout period and of one replanning for a chunk of plant worlds: the configuration of the flags, or
    each of --configs, "iterations,horizon,knots,replanning interval,planner step in ms,samples"."""
    global PLAN_DT
    set_task(a.task)
    prs = json.loads(pairs_path(a.pairs).read_text())["pairs"]
    scene, worlds = worlds_of(prs, [0])
    chunk = (worlds * 4)[:a.chunk]
    B = UH.UBatch(scene, TASK, chunk, T.DT_C, gen_seed=0)
    if a.configs:
        cfgs = []
        for c in a.configs:
            i_, h_, k_, r_, dt_ms, n_ = c.split(",")
            cfgs.append((int(i_), int(h_), int(k_), int(r_), float(dt_ms), [int(n_)]))
    else:
        cfgs = [(a.iters, a.horizon, a.knots, a.replan, 1e3 * a.plan_dt if a.plan_dt else 1.0, a.n)]
    out = []
    for iters, h_, k_, r_, dt_ms, ns in cfgs:
        configure(r_, h_, k_, iters)
        PLAN_DT = None if dt_ms == 1.0 else 1e-3 * dt_ms
        for n in ns:
            R = Rollouts(model_scene(scene, a.model), chunk, n)
            B.begin(chunk, record=False, hom=True)
            R.begin(chunk)
            vw = B.vw
            palm = vw["mocap_pos"][:, B.mo_palm].float()
            p0 = vw["qpos"][:, B.qa:B.qa + 3].float() - palm
            q0 = vw["qpos"][:, B.qa + 3:B.qa + 7].float()
            ctx = {"palm": palm.repeat_interleave(n, 0), "p0": p0.repeat_interleave(n, 0),
                   "goal": q0[:, None].expand(len(chunk), H, 4).repeat_interleave(n, 0)}
            u = vw["ctrl"][:, B.act_t].float()
            mpc = MPC("mppi", len(chunk), R.lim.lo, R.lim.hi, n, 1, 0)
            with torch.inference_mode():
                mpc.plan(R, B.state(), u, ctx)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(3):
                    mpc.plan(R, B.state(), u, ctx)
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 3
            per = dt / (PLAN["iters"] * H)
            row = {"plant_worlds": len(chunk), "n": n, "worlds": R.B.W, "iters": PLAN["iters"], "H": H, "knots": KNOTS,
                   "replan": REPLAN, "plan_dt": PLAN_DT, "replan_s": round(dt, 3), "budget_s": REPLAN * T.DT_C,
                   "ms_per_period": round(1e3 * per, 2), "us_per_world_period": round(1e6 * per / R.B.W, 2),
                   "gpu": torch.cuda.get_device_name(0), "when": time.strftime("%Y-%m-%d %H:%M"), "model": a.model,
                   **inner_loop_plant.describe()}
            print(json.dumps(row), flush=True)
            out.append(row)
            del R
            torch.cuda.empty_cache()
    PLAN_DT = None
    configure(10, 30, 3, 4)
    with open(ROWS / f"mpc_timing_{TASK}.jsonl", "a") as fh:
        for r_ in out:
            fh.write(json.dumps(r_) + "\n")
    return 0


def cmd_film(a):
    """Films of several controllers on one pair and seed (the films of uhas_sr2_film.py: links see-through, the
    sphere's axes as rods, each finger's pad and rigid-part force per frame), side by side or two by two."""
    import hom_films as HF
    import hom_landscape as L
    import uhas_sr2_film as UF
    set_task(a.task)
    p = json.loads(pairs_path(a.pairs).read_text())["pairs"][a.pair]
    scene, worlds = worlds_of([p], [a.seed])
    world = worlds[0]
    plant_scene = model_scene(scene, a.plant_model)
    B = UH.UBatch(plant_scene, TASK, [world], T.DT_C, gen_seed=a.seed)
    if a.plant_mu != 1.0:
        set_friction(B, np.full(1, a.plant_mu, np.float32))
    clips, info = [], {}
    global PLAN_DT
    for spec in a.methods:
        spec_m, _, cfg = spec.partition("@")          # @iters,horizon,knots,replan,planner step ms,samples
        name, _, arg = spec_m.partition(":")
        method = "ps" if name == "ps_rt" else name
        n_s, PLAN_DT = a.n, None
        if cfg:
            i_, h_, k_, r_, dt_ms, n_ = cfg.split(",")
            configure(int(r_), int(h_), int(k_), int(i_))
            n_s, PLAN_DT = int(n_), (None if float(dt_ms) == 1.0 else 1e-3 * float(dt_ms))
        else:
            configure(*((10, 10, 2, 1) if name == "ps_rt" else (10, 30, 3, 4)))
        UH.ATT_GAIN = float(arg) if name == "hom" and arg else 0.0
        R = policy = F = None
        if method in METHODS:
            fr = PLAN["frictions"] if method == "spider_wc" else None
            R = Rollouts(model_scene(scene, arg or "pads"), [world], n_s // len(fr) if fr else n_s, fr)
        elif method in HS_METHODS + RES_METHODS:
            R = HomRollouts(model_scene(scene, arg or "pads"), [world], n_s)
            R.mode = "res" if method in RES_METHODS else "hs"
        elif method == "uhas":
            F = U.Fingers(B.m, "cuda")
            policy, _ckpt = UH.load_policy(UH.RUNS / (arg or RUN))
        trace = []
        res, M, pr = episode(B, [world], method, R, policy, F, seed=a.seed, trace=trace)
        r = dict(res[0])
        J = L.track_score(TASK, r)
        ft = B.rec["ft"][:, 0].cpu().numpy()
        fl = np.maximum(B.rec["fo"][:, 0].cpu().numpy() - ft, 0.0)
        lab = SHORT.get(name, name)
        if cfg:
            lab += f" {int(i_)}\u00d7{int(n_)}"
        if method in METHODS + HS_METHODS and (arg or a.plant_model != "pads"):
            lab += ", " + model_label(arg or "pads")
        if name == "hom" and UH.ATT_GAIN > 0:
            lab += f", feedback {UH.ATT_GAIN:g}/s"
        clips.append(UF.frames_of([(q, m, t, ft[k], fl[k]) for q, m, t, k in trace], plant_scene, world,
                                  f"{lab}: J {J:.2f}, {M['grip_mean_N'][0]:.1f} N, "
                                  f"{M['reversals_ps'][0]:.1f} rev/s"))
        info[spec] = {"J": J, "held": bool(r.get("held")), "track_err_rms_deg": r.get("track_err_rms_deg"),
                      "contact_all_frac": r.get("contact_all_frac"), **{k: float(M[k][0]) for k in IU.METRICS},
                      "replan_s_mean": float(np.mean(pr["walls"])) if pr["walls"] else None}
        print(spec, json.dumps(info[spec], default=float), flush=True)
        del R
        torch.cuda.empty_cache()
    UH.ATT_GAIN = 0.0
    PLAN_DT = None
    configure(10, 30, 3, 4)
    name = (f"film_mpc_{TASK}{a.pairs}_p{a.pair}_s{a.seed}" + ("" if a.plant_mu == 1.0 else f"_mu{a.plant_mu:g}")
            + ("" if a.plant_model == "pads" else f"_plant_{a.plant_model}") + a.tag)
    HF.compose(clips, UH.OUT / "media" / f"{a.date}-{name}.mp4", grid=(2, 2) if len(clips) == 4 else None)
    (ROWS / f"{name}.json").write_text(json.dumps({"pair": p, "seed": a.seed, "plant_mu": a.plant_mu,
                                                   "plant_model": a.plant_model,
                                                   "methods": a.methods, "clips": info,
                                                   **inner_loop_plant.describe()}, indent=1, default=float))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--run", default=RUN)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--rule", default="benchmark", choices=("benchmark", "rank", "rank5", "tune", "top", "patch", "all"),
                   help="benchmark: HOM's "
                   "and UHAS's ten best test layouts (the twist); rank: HOM's 50 best layouts and 50 at random "
                   "(rank_pairs); tune: six others for choosing a configuration (tune_pairs); top: the ten best of the "
                   "--src pairs under HOM and under full DIAL, to rerun at fresh seeds (top_pairs; _top, or <src>top); "
                   "patch: the layouts within 15 mm of HOM's best (patch_pairs); all: every layout of the grid "
                   "(all_pairs)")
    p.add_argument("--src", default="_rank", help="rule top: the pair set whose rows choose the ten best (_rank, _all)")
    p = sub.add_parser("timing")
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--n", type=int, nargs="+", default=[128])
    p = sub.add_parser("run")
    p.add_argument("--method", required=True, choices=("hom", "uhas", "ilqr") + METHODS + HS_METHODS + RES_METHODS)
    p.add_argument("--il-step", type=int, default=None, help="method ilqr: periods per step (inner_loop_ilqr.IL step, 5)")
    p.add_argument("--il-fb", action="store_true", help="method ilqr: the plant applies the gains too (IL fb)")
    p.add_argument("--il-pad", action="store_true", help="method ilqr: the smooth pad term in the model (IL pad)")
    p.add_argument("--preview", type=float, default=0.0, help="HOM's command read this many s ahead in the schedule "
                   "(uhas_sr2.PREVIEW_S; methods that run HOM)")
    p.add_argument("--prm", default=None, help="HOM's forces and contact-mode weights for every world, JSON "
                   "(hom_tasks.PRM keys)")
    p.add_argument("--prm-shared", action="store_true", help="HOM's forces and contact modes of the task's shared pick "
                   "under J (hom_grasp_refine level shared, rows/sv25_track/shared)")
    p.add_argument("--hom-param", nargs="+", default=[], help="key=value of hom.HomParams for HOM in the plant and the "
                   "planner's worlds, e.g. w_damp=0.01")
    p.add_argument("--run", default=RUN, help="the UHAS run (method uhas)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--plant-mu", type=float, default=1.0, help="the plant's sliding friction (the planners' model "
                   "keeps 1.0; spider_wc's spans 0.4-1.0)")
    p.add_argument("--chunk", type=int, default=16, help="plant worlds per batch")
    p.add_argument("--n", type=int, default=64, help="samples per plant world and iteration")
    p.add_argument("--limit-pairs", type=int, default=0, help="only the first pairs (smoke tests)")
    p.add_argument("--tag", default="", help="appended to the rows' file name")
    p.add_argument("--pose-tau", type=float, default=None, help="method hom: the pose filter's time constant, s (inf: "
                   "open loop from the task's start; hom_gpu.POSE_FILTER)")
    p.add_argument("--meas-every", type=int, default=1, help="method hom: a tracker frame every this many periods")
    p.add_argument("--plan-delay", type=int, default=0, help="planners: plan from the plant's state this many periods "
                   "earlier (4: the tag tracker's 40 ms)")
    p.add_argument("--plan-noise", action="store_true", help="planners: the tag tracker's noise on the object's pose "
                   "(0.5 mm, 1 deg)")
    p.add_argument("--f-tripod", type=float, default=None, help="HOM's commanded pad force per finger, N "
                   "(hom_tasks.PRM f_tripod, 2.5), for every world")
    p.add_argument("--hs-sigma", type=float, default=None, help="HOM-space planners: the sampling noise in units "
                   "(PLAN_HS sigma, 0.3)")
    p.add_argument("--variant", default=None, choices=sorted(T.VARIANTS), help="method hom: the sensing variant "
                   "(hom_tasks.VARIANTS; sense and tip run at 50 Hz); default the task's (uhas_sr2.VARIANT)")
    p.add_argument("--att-gain", type=float, default=0.0, help="method hom: feedback on the measured orientation, 1/s "
                   "(uhas_sr2.ATT_GAIN)")
    p.add_argument("--att-axis", action="store_true", help="method hom: that feedback only about the task's commanded "
                   "axis (uhas_sr2.ATT_AXIS)")
    for q in (sub.choices["timing"], sub.choices["run"]):
        q.add_argument("--model", default="pads", type=contact_model, help="the planner's contact model: pads, pt "
                       "(point contact) or pt_t<mm> (point contact with torsional friction; torsion_scene)")
        q.add_argument("--replan", type=int, default=10, help="periods between replannings")
        q.add_argument("--horizon", type=int, default=30, help="periods")
        q.add_argument("--knots", type=int, default=3)
        q.add_argument("--iters", type=int, default=4)
        q.add_argument("--plan-dt", type=float, default=None, help="the planner's physics step, s (the plant's: 0.001)")
    p.add_argument("--plant-model", default="pads", type=contact_model, help="the plant's contact model, as --model")
    p.add_argument("--warm", default="hold", choices=("hold", "extrap"), help="the shifted plan past its last knot: "
                   "hold it, or continue the last segment's slope")
    p.add_argument("--lam-rel", type=float, default=None, help="MPPI's and annealing's temperature over the costs' "
                   "standard deviation (PLAN lam_rel, 0.03)")
    p.add_argument("--sigma", type=float, default=None, help="the knots' sampling noise, rad (PLAN sigma, 0.08)")
    p.add_argument("--rate-w", type=float, default=None, help="weight of the knot-rate cost (COST rate, 1)")
    p.add_argument("--acc-w", type=float, default=None, help="weight of the rate changes between knot gaps and from "
                   "the executed rate, in the rate scale (COST acc, 0)")
    p.add_argument("--rev-w", type=float, default=None, help="weight of the sign flips of the targets' rates at the "
                   "replanning: max(0, -r_new r_executed) in the rate scale squared (COST rev, 0)")
    p.add_argument("--err-db", type=float, default=None, help="dead band on the rotation error in the cost, deg "
                   "(COST err_db_deg, 0)")
    sub.choices["timing"].add_argument("--configs", nargs="+", default=None, help="instead of the flags: "
                                       "iterations,horizon,knots,replanning interval,planner step in ms,samples")
    p = sub.add_parser("film")
    p.add_argument("--pair", type=int, default=0, help="index into rows/mpc_pairs_twist.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", nargs="+", default=["hom", "ps_rt", "anneal", "uhas"],
                   help="hom[:gain], ps_rt[:model], a planner[:model][@iters,horizon,knots,replan,step ms,samples], "
                   "uhas[:run]; four make a two-by-two film")
    p.add_argument("--plant-mu", type=float, default=1.0)
    p.add_argument("--plant-model", default="pads", type=contact_model, help="the plant's contact model (run --model)")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--tag", default="")
    p.add_argument("--date", default=time.strftime("%Y%m%d"), help="the film's file-name date")
    for q in sub.choices.values():
        q.add_argument("--task", default="twist", choices=sorted(UH.D_MM))
    for q in (sub.choices["timing"], sub.choices["run"], sub.choices["film"]):
        q.add_argument("--pairs", default="", help="the pair set: rows/mpc_pairs_<task><pairs>.json ('' the "
                       "benchmark's twist pairs, _rank or _rank20 those of pairs --rule rank)")
    a = ap.parse_args()
    return {"pairs": cmd_pairs, "timing": cmd_timing, "run": cmd_run, "film": cmd_film}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
