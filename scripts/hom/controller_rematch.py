#!/usr/bin/env python3
"""Controllers on the calibrated plant, each with its force knob tuned (hand-4a, 2026-11-20; owner, 08:0x: "Sure on all
three decisions", the controller rematch among them; 11-19 23:5x: "shouldn't we try to do the same for the other control
methods we benchmarked? Adding it to the action space or otherwise more intelligently controlling it might be helpful.
Either way, it seems we might be able to re order our methods and re-crown HOM").

The plant is the GPU port of the calibrated plant of the 11-17 blind predictions, hom_gpu.py with HOM_GPU_PLANT=cal75:
servos at 7.5 V, gears the load cannot backdrive, the hand's MCP stops, impratio 1 and friction 1.0, and the bench
server's torque budget of 0.56 N m on every controller's targets. The task is the sphere twist of the bench at the bench
server's control period of 20 ms. The layouts are the 45 of the morning series of 2026-11-20 whose simulated twist keeps
the fingers at least 2 mm apart (data/20261119-trial_plans.json of the twist ranking: default forces, no CEM, no
droppers), each at its plan's grasp.

    pairs   writes data/20261120-pairs.json
    hom     HOM on the GPU plant at --seeds, --model pads or pt, --prm (hom_tasks.PRM forces, JSON), rows to
            rows/hom_<model><tag>.jsonl
    check   the GPU HOM against the CPU plant's predictions (the twist ranking's rows at the same layout, grasp, seed and
            contact model): per pair, the mean difference, Spearman's rho; writes data/20261120-gpu_vs_cpu<tag>.json

    WARP_CACHE_PATH=$(mktemp -d) HOM_GPU_PLANT=cal75 .venv/bin/python scripts/hom/controller_rematch.py hom --model pads
Score: max(0, 1 - RMS error / RMS commanded rotation) over the task (hom_tasks.metrics), 0 when the sphere dropped or
the rollout diverged, as the twist ranking scores its runs.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")
os.environ.setdefault("HOS_PLANT", "kp0.55_kv0_fr0.18_fl0_dp0.011")
os.environ.setdefault("HOS_SCENES", "logs/20261102-hom_servo_plant/scenes")
os.environ.setdefault("HOM_ROWS_TAG", "_cal75")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401

STUDY = ROOT / "docs/experiments/20261120-controller_rematch"
DATA, ROWS = STUDY / "data", STUDY / "rows"
RANK = ROOT / "docs/experiments/20261119-twist_layout_ranking"
RANK_LOGS = ROOT / "logs/20261119-twist_layout_ranking"
DT = 0.02                                          # the bench server's control period (hom_bench.DT_BENCH)
GRID = "grid2dgr_size"                             # the twist ranking's lattice and its ground-clear grasps
# The CPU plant's scenes (20261117-g13_open_pose_replan.SCENES): the bench's sphere on its stem, impratio 1
STEM = ROOT / "logs/20261116-sim_real_reversal/scenes"
STEM_SCENES = {"pads": "sph040_stem_block_tape0_imp1.xml", "pt": "sph040_stem_block_pt_imp1.xml"}
VARIANT = "notip0"                                 # HOM without the fingertips, exact object pose, 50 Hz: the CPU runs' sensing


def score(r):
    e, ref = r.get("track_err_rms_deg"), r.get("track_ref_rms_deg")
    if r.get("dropped") or not e or not ref:
        return 0.0
    return max(0.0, 1.0 - e / ref)


def cmd_pairs(a):
    lst = json.loads((RANK / "data/20261119-trial_plans.json").read_text())["plans"]
    sel = [e for e in lst if e.get("role") in ("ranked", "anchor") and not e.get("cad_touch") and not e.get("prm")
           and "_cem_" not in e["plan"] and e.get("group") != "dropper"]
    out = [{"call": int(e["call"]), "xy": e["xy"], "k": int(e["k"]), "plan": Path(e["plan"]).name,
            "predicted": e.get("score"), "cad_least_mm": e.get("cad_least_mm")} for e in sel]
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "20261120-pairs.json").write_text(json.dumps({"what": "the rematch's layouts and grasps (module docstring)",
                                                          "written": time.strftime("%F %H:%M"), "pairs": out},
                                                         indent=1) + "\n")
    print(f"{len(out)} pairs -> {DATA / '20261120-pairs.json'}")
    return 0


def worlds(pairs, seeds, model, prms=(None,), variant=None):
    """The pairs' GPU worlds at each seed and each HOM parameter set of prms (None: hom_tasks.PRM's defaults)."""
    import copy
    import hom_gpu as HG
    scene = copy.copy(HG.Scene("sphere", 40.0))
    scene.pads_path = STEM / STEM_SCENES[model]               # the plant's scene; the open pose uses the cell's pt model
    want = {(p["call"], p["k"]) for p in pairs}
    out = []
    for key, mxy, setup in HG.layout_list("twist", GRID, 0, 40.0):
        if (int(key["call"]), int(key.get("grasp", 0))) not in want:
            continue
        bm, q_open = scene.prepare(mxy, setup)
        for prm in prms:
            for sd in seeds:
                out.append({"key": key, "seed": sd, "setup": setup, "body_mounts": bm, "q_open": q_open,
                            "var": HG.variant_world(variant or VARIANT, sd, dt_default=DT), **({"prm": prm} if prm else {})})
    return scene, out


# the twist ranking's stage E grid (sim/20261119-twist_force_refine.py): HOM's commanded normal force per finger (N) and
# the weight of the spin rows in Eq. (2)
F_TRIPOD = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
W_SPIN = (0.25, 0.5, 1.0, 2.0)
GRID_E = [{"f_tripod": f, "w_spin_tripod": w} for f in F_TRIPOD for w in W_SPIN]
# E2 (2026-11-20 08:4x): below stage E's edge, where its best setting lay (1.0 N, 0.25) on all 45 layouts
GRID_E2 = [{"f_tripod": f, "w_spin_tripod": w} for f in (0.5, 0.75, 1.0) for w in (0.1, 0.25, 0.5)]


def cmd_hom(a):
    import hom_gpu as HG
    if HG.PLANT_CAL != "cal75" and not a.base:
        sys.exit("set HOM_GPU_PLANT=cal75 (or --base for the scene's own plant)")
    pairs = json.loads((DATA / "20261120-pairs.json").read_text())["pairs"]
    if a.only:
        only = {tuple(int(v) for v in s_.split(":")) for s_ in a.only}
        pairs = [p for p in pairs if (p["call"], p["k"]) in only]
    if a.limit:
        pairs = pairs[:a.limit]
    prms = {"E": GRID_E, "E2": GRID_E2}.get(a.grid) or [json.loads(x) if x else None for x in (a.prm or [""])]
    scene, ws = worlds(pairs, a.seeds, a.model, prms, a.variant)
    xy = {(p["call"], p["k"]): p["xy"] for p in pairs}
    ROWS.mkdir(parents=True, exist_ok=True)
    path = ROWS / f"hom_{a.model}{a.tag}.jsonl"
    sc, nd = [], 0
    for i0 in range(0, len(ws), a.batch):
        chunk = ws[i0:i0 + a.batch]
        t0 = time.time()
        B = HG.Batch(scene, "twist", chunk, DT)
        t1 = time.time()
        rec, final, phases, wrefs, times, thref, wall = B.run()
        res = B.score(rec, final, phases, wrefs, times, thref)
        with path.open("a") as fh:
            for w, r in zip(chunk, res):
                ck = (int(w["key"]["call"]), int(w["key"].get("grasp", 0)))
                row = {"controller": "hom", "model": a.model, "call": ck[0], "k": ck[1], "xy": xy[ck],
                       "seed": w["seed"], "prm": w.get("prm"), "variant": a.variant or VARIANT, "plant": HG.PLANT_CAL or "scene", "dt": DT,
                       "track": round(score(r), 4), **{k: r.get(k) for k in ("held", "dropped", "track_err_rms_deg",
                                                                             "track_ref_rms_deg", "contact_all_frac",
                                                                             "link_force_max_N")}}
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        sc += [score(r) for r in res]
        nd += sum(bool(r.get("dropped")) for r in res)
        print(f"  {i0 + len(chunk)}/{len(ws)} worlds, build {t1 - t0:.0f} s, run {wall:.0f} s", flush=True)
        del B
    print(f"{len(ws)} worlds; mean {np.mean(sc):.3f}, dropped {nd} -> {path}")
    return 0


def cpu_rows(model):
    """{(call, k, seed): score} of the twist ranking's CPU runs (stages A, B, C, P, V and the feasible search)."""
    out = {}
    for st in ("A", "B", "C", "P", "V", "FG"):
        p = RANK_LOGS / f"rows_{st}.jsonl"
        if not p.exists():
            continue
        for x in p.read_text().splitlines():
            if not x.strip():
                continue
            r = json.loads(x)
            if r.get("model") != model:
                continue
            s = 0.0 if r.get("error") else float(r.get("track") or 0.0)
            out[(int(r["call"]), int(r["k"]), int(r["seed"]))] = s
    return out


def cmd_check(a):
    from scipy.stats import spearmanr
    pairs = json.loads((DATA / "20261120-pairs.json").read_text())["pairs"]
    out = {}
    for model in a.models:
        g = {}
        for x in (ROWS / f"hom_{model}{a.tag}.jsonl").read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                if r.get("prm") in (None, {}):
                    g[(r["call"], r["k"], r["seed"])] = r["track"]
        c = cpu_rows({"pt": "point"}.get(model, model))
        rows = []
        for p in pairs:
            ks = [(p["call"], p["k"], s) for s in a.seeds]
            if all(k in g for k in ks) and all(k in c for k in ks):
                rows.append({"call": p["call"], "xy": p["xy"], "k": p["k"],
                             "gpu": round(float(np.mean([g[k] for k in ks])), 4),
                             "cpu": round(float(np.mean([c[k] for k in ks])), 4)})
        G, C = np.array([r["gpu"] for r in rows]), np.array([r["cpu"] for r in rows])
        rho = float(spearmanr(G, C)[0]) if len(rows) > 2 else None
        out[model] = {"n": len(rows), "gpu_mean": round(float(G.mean()), 4), "cpu_mean": round(float(C.mean()), 4),
                      "mean_abs_diff": round(float(np.abs(G - C).mean()), 4), "rho": rho, "pairs": rows}
        print(f"{model}: {len(rows)} pairs, GPU {G.mean():.3f} vs CPU {C.mean():.3f}, |diff| {np.abs(G - C).mean():.3f}, "
              f"rho {rho:.2f}")
    path = DATA / f"20261120-gpu_vs_cpu{a.tag}.json"
    path.write_text(json.dumps({"what": "GPU HOM on the cal75 plant against the CPU plant's runs", "seeds": a.seeds,
                                "written": time.strftime("%F %H:%M"), "models": out}, indent=1) + "\n")
    print("wrote", path)
    return 0


REF_RMS_DEG = 59.4            # the twist's RMS commanded rotation over the task (every row of hom_tasks.metrics here)


def track(r):
    """The twist ranking's score of a row of either runner: 0 unless the sphere was held at the end."""
    e = r.get("track_err_rms_deg")
    if not r.get("held") or e is None:
        return 0.0
    return max(0.0, 1.0 - e / (r.get("track_ref_rms_deg") or REF_RMS_DEG))


def arm_rows():
    """{arm name: [rows]} of every run in rows/: HOM's grids by force setting (controller_rematch.py hom), and each
    inner_loop_mpc.py file (one arm per file)."""
    arms = {}
    for path in sorted(ROWS.glob("hom_*_grid*.jsonl")):
        for x in path.read_text().splitlines():
            if not x.strip():
                continue
            r = json.loads(x)
            p_ = r.get("prm") or {}
            name = (f"hom {r['model']} {r.get('variant', VARIANT)} f{p_.get('f_tripod', 2.5):g} "
                    f"w{p_.get('w_spin_tripod', 1.0):g}")
            arms.setdefault(name, {})[(r["call"], r["k"], r["seed"])] = r
    for path in sorted(ROWS.glob("mpc_*_twist_cal45*.jsonl")):
        if "_smoke" in path.name:
            continue
        name = path.stem.replace("mpc_", "").replace("_twist_cal45", " ")
        for x in path.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                arms.setdefault(name, {})[(r["x_sep"], r["y_sep"], r["grasp"], r["seed"])] = r
    return {k: list(v.values()) for k, v in arms.items()}


def cmd_summary(a):
    """Per arm: the mean score over the 45 layouts at seeds 0-2 and at 10-12, the drops, and the usability metrics of
    the planner rows; written to data/20261120-rematch_summary.json."""
    out = {}
    for name, rows in sorted(arm_rows().items()):
        rec = {"n": len(rows)}
        for lab, seeds in (("s012", (0, 1, 2)), ("s101112", (10, 11, 12))):
            rr = [r for r in rows if r["seed"] in seeds]
            if rr:
                rec[lab] = round(float(np.mean([track(r) for r in rr])), 4)
                rec[f"{lab}_n"] = len(rr)
                rec[f"{lab}_dropped"] = int(sum(not r.get("held") for r in rr))
        for k in ("reversals_ps", "cmd_rate_dps", "sat_frac", "link_mean_N", "pad_share", "replan_s_mean"):
            v = [r[k] for r in rows if r.get(k) is not None]
            if v:
                rec[k] = round(float(np.median(v)), 4)
        out[name] = rec
    path = DATA / "20261120-rematch_summary.json"
    path.write_text(json.dumps({"what": "per arm of the controller rematch (controller_rematch.py summary)",
                                "written": time.strftime("%F %H:%M"), "arms": out}, indent=1) + "\n")
    w = max(len(k) for k in out)
    for k, v in out.items():
        if a.filter and a.filter not in k:
            continue
        print(f"{k:<{w}}  s0-2 {v.get('s012', float('nan')):.3f} ({v.get('s012_dropped', 0)}/{v.get('s012_n', 0)} dropped)"
              f"  s10-12 {v.get('s101112', float('nan')):.3f} ({v.get('s101112_dropped', 0)}/{v.get('s101112_n', 0)})"
              + (f"  rev/s {v['reversals_ps']:.2f}  sat {v['sat_frac']:.3f}" if "reversals_ps" in v else ""))
    print("wrote", path)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pairs")
    p = sub.add_parser("hom")
    p.add_argument("--model", default="pads", choices=("pads", "pt"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--prm", nargs="*", default=None, help="HOM parameter sets, JSON each ('' the defaults)")
    p.add_argument("--variant", default=None, help="hom_tasks.VARIANTS key of HOM's sensing (default notip0)")
    p.add_argument("--tag", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", nargs="*", default=[], help="pairs call:k")
    p.add_argument("--grid", default="", choices=("", "E", "E2"), help="E: every world at each of stage E's 24 force "
                   "settings; E2: the 9 settings at and below its edge (0.5-1.0 N, spin weight 0.1-0.5)")
    p.add_argument("--batch", type=int, default=1200, help="worlds per GPU batch")
    p.add_argument("--base", action="store_true", help="the scene cache's own plant (no HOM_GPU_PLANT), for checks")
    p = sub.add_parser("check")
    p.add_argument("--models", nargs="+", default=["pads", "pt"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--tag", default="")
    p = sub.add_parser("summary")
    p.add_argument("--filter", default="", help="print only the arms whose name holds this text")
    a = ap.parse_args()
    return {"pairs": cmd_pairs, "hom": cmd_hom, "check": cmd_check, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
