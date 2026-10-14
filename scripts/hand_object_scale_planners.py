#!/usr/bin/env python3
"""Planner comparison on the in-hand turn of the hand-object scale study: HOM against sampling MPC variants that keep
three pads on the cylinder, on the same cells, grasps, placements and plant (2026-10-13; owner: "HOM looks way
smoother than MPC ... can we see if we can improve this smoothness/jitteriness? ... compare predictive sampling to
CEM, iCEM, and maybe GPU-based method like annealed sampling").

Every sampling configuration uses the three-pad cost of hand_object_scale_mpc.py (CONTROLLED: per-finger contact
force at least 0.5 N, turn rate below 60 deg/s) and the same budget: 64 rollouts of 0.5 s per replanning every 20 ms,
three knots, noise 0.08 rad. Each row carries the turn result, the smoothness metrics of
hand_object_scale_mpc.smooth_metrics and the 20 ms log.

  .venv/bin/python scripts/hand_object_scale_planners.py run --configs hom ps ps_pin mppi_pin cem_pin icem_pin anneal_pin \\
      --cells cylinder:25:0.125 cylinder:16:0.625 cylinder:10:0.75 cylinder:16:-0.375 --seeds 0 1 --shard 0/2
Rows: docs/experiments/20261013-scale_turn_controllers/planners/turn.jsonl (fsynced, resumable per config, cell, seed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hand_object_scale_mpc as M  # noqa: E402
import hand_object_scale_sim as S  # noqa: E402
import hom_turn3 as H  # noqa: E402

OUT = M.OUT / "planners"
SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"
BASE = dict(M.DEFAULTS, **M.PLANNER_DEFAULTS, **M.CONTROLLED, nsample=64, horizon=0.5)
CONFIGS = {
    "ps": dict(method="ps"),
    "ps_pin": dict(method="ps", pin0=1.0),
    "mppi_pin": dict(method="mppi", pin0=1.0),
    "cem_pin": dict(method="cem", iters=2, pin0=1.0),
    "icem_pin": dict(method="icem", iters=2, pin0=1.0),
    "anneal_pin": dict(method="anneal", iters=4, pin0=1.0),
    # MuJoCo-Warp: 4 stages of 256 plans (16 times the CPU budget); pads meet only the cylinder in the planner's model
    # and the GPU solver stops at tolerance 1e-6 or 20 iterations (hand_object_scale_mpc_gpu.py)
    "anneal_gpu": dict(method="anneal", iters=4, nsample=1024, pin0=1.0, pad_filter=1.0, device="gpu"),
}
CONFIGS["ilqr"] = dict(method="ilqr", pad_filter=1.0)    # hand_object_scale_ilqr.py: 25 intervals of 20 ms
STRING_KEYS = ("method", "device")


def prm_of(name, extra=()):
    prm = dict(BASE, **CONFIGS[name])
    for kv in extra:
        k, v = kv.split("=")
        prm[k] = v if k in STRING_KEYS else float(v)
    for k in ("nsample", "knots", "cost_every", "iters", "keep", "log_every"):
        prm[k] = int(prm[k])
    return prm


def done_keys(path):
    out = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok":
                out.add((r["config"], r["shape"], r["d_mm"], r["t"], r["seed"], r.get("variant", "")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--configs", nargs="+", default=["hom"] + list(CONFIGS))
    r.add_argument("--cells", nargs="+", required=True, help="shape:d_mm:t")
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    r.add_argument("--set", nargs="*", default=[], help="override parameters of every MPC config, k=v")
    r.add_argument("--variant", default="", help="label of a plant or parameter variant, stored in the rows")
    r.add_argument("--plant-set", nargs="*", default=[], help="plant changes for the variant: mu=<friction>")
    r.add_argument("--nthread", type=int, default=6)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--out", default=str(OUT / "turn.jsonl"))
    r.add_argument("--hom-rows", default=str(M.OUT / "hom_postoff_servo/turn.jsonl"))
    a = ap.parse_args()
    S.PLANT = SERVO
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out)
    best = M.hom_best(Path(a.hom_rows))
    jobs = []
    for spec in a.cells:
        shape, dd, t = spec.split(":")
        for seed in a.seeds:
            for cfg in a.configs:
                jobs.append((shape, float(dd), float(t), seed, cfg))
    k0, n0 = (int(v) for v in a.shard.split("/"))
    jobs = [j for i, j in enumerate(jobs) if i % n0 == k0]
    jobs = [j for j in jobs if (j[4], j[0], j[1], j[2], j[3], a.variant) not in done]
    print(f"{len(jobs)} runs in shard {a.shard}", flush=True)
    cells = {(c[1], c[2], c[0].get("t")): c[0] for c in S.cell_list("diag")}
    for shape, dd, t, seed, cfg in jobs:
        cell = cells[(shape, dd, t)]
        ci = best[(cell["tag"], shape, dd)]["cand"]
        plant, meta, geo, sc, gg = M.setup(cell, shape, dd, ci)
        for kv in a.plant_set:
            k, v = kv.split("=")
            if k == "mu":      # sliding friction of every geom of the plant (the planner keeps the model's)
                plant.m.geom_friction[:, 0] = float(v)
        row = {"config": cfg, "variant": a.variant, "plant_set": a.plant_set, "tag": cell["tag"], "t": t,
               "x_sep_mm": cell["x_sep_mm"], "shape": shape, "d_mm": dd, "plant": SERVO, "seed": seed, **geo,
               "when": time.strftime("%Y-%m-%d %H:%M")}
        try:
            if cfg == "hom":
                mirror = S.Mirror(Path(sc["pt"][0]))
                S.set_palm(mirror.m, gg["palm"])
                hom_prm = dict(H.DEFAULTS, post_off=1.0, kp_model=S.RB.plant_args(SERVO)["kp"])
                res = M.hom_turn_logged(plant, mirror, meta, seed, hom_prm)
                row.update({f"prm_{k}": v for k, v in hom_prm.items()})
            else:
                prm = prm_of(cfg, a.set)
                if a.plant_set:
                    prm["planner_model"] = "nominal"
                res = M.mpc_turn(plant, meta, seed, prm, nthread=a.nthread)
                row.update({f"prm_{k}": v for k, v in prm.items()})
            res.pop("trace", None)
            row.update(res, status="ok")
        except Exception as e:
            row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if row["status"] == "ok":
            print(f"{cfg:11s} {shape} {dd:4.0f} t{t:+.3f} s{seed}: held {row['held']} turn {row['turn_end_deg']:6.1f} "
                  f"fingers {row['fingers_loaded_end']} three {row.get('three_frac')} breaks {row.get('contact_breaks')} "
                  f"rate {row.get('cmd_rate_rms')} rev {row.get('cmd_reversals')} jit {row.get('omega_jitter')} "
                  f"peak {row.get('peak_rate')} {row['wall_s']} s", flush=True)
        else:
            print(f"{cfg} {shape} {dd} t{t} s{seed}: ERROR {row['error']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
