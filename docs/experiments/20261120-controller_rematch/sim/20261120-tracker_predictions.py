#!/usr/bin/env python3
"""Predictions of the morning plans of 2026-11-20 under the bench's 30 fps tag tracker, written before their trials
(hand-4a, 2026-11-20 09:xx; owner, 08:4x: "why don't we run tracking at 60hz?"; the stage E prediction of 0.61 assumed
an exact object pose).

For every plan of the twist ranking's plan list that runs HOM's twist at one force setting (the ranking's default plans
and the stage E plans; the CEM plans' force schedules are not run on the GPU), the mean tracking score at seeds 0-2 on the
GPU cal75 plant in the stem scenes (scripts/hom/controller_rematch.py's runs through inner_loop_mpc.py) is given with the
exact object pose (variant notip0) and with the 30 fps tracker (variant cam30, a frame every second period, 42 ms old),
beside the plan's CPU prediction (meta.ranking, exact pose). Score: the twist ranking's (controller_rematch.track).

    .venv/bin/python docs/experiments/20261120-controller_rematch/sim/20261120-tracker_predictions.py
Writes data/20261120-tracker_predictions.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
RANK = ROOT / "docs/experiments/20261119-twist_layout_ranking"
ROWS = STUDY / "rows"
sys.path.insert(0, str(ROOT / "scripts/hom"))
REF = 59.4


def track(r):
    e = r.get("track_err_rms_deg")
    if not r.get("held") or e is None:
        return 0.0
    return max(0.0, 1.0 - e / (r.get("track_ref_rms_deg") or REF))


def arm(stem):
    p = ROWS / f"mpc_hom_twist_cal45{stem}.jsonl"
    out = {}
    if p.exists():
        for x in p.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                out.setdefault((float(r["x_sep"]), float(r["y_sep"]), int(r["grasp"])), {})[int(r["seed"])] = track(r)
    return out


def prm_tag(prm):
    if not prm:
        return ""
    return "_prm" + "".join(f"{k.replace('_', '')}{v:g}" for k, v in prm.items())


def main():
    lst = json.loads((RANK / "data/20261119-trial_plans.json").read_text())["plans"]
    cache = {}
    out = []
    for e in lst:
        if "_cem_" in e["plan"] or e.get("role") not in ("ranked", "anchor"):
            continue
        prm = e.get("prm") or {}
        prm = {k: prm[k] for k in ("f_tripod", "w_spin_tripod") if k in prm}
        key = (float(e["xy"][0]), float(e["xy"][1]), int(e["k"]))
        pr = e.get("prediction") if isinstance(e.get("prediction"), dict) else {}
        cpu = pr.get("both_models_seeds012") if prm else None          # stage E plans: their refined prediction
        rec = {"plan": Path(e["plan"]).name, "xy": e["xy"], "k": e["k"], "prm": prm or None,
               "cpu_exact": cpu if cpu is not None else e.get("score"), "cad_least_mm": e.get("cad_least_mm"),
               "group": e.get("group")}
        for lab, var in (("gpu_exact", "_notip0"), ("gpu_cam30", "_cam30_holde2")):
            stem = prm_tag(prm) + var
            if stem not in cache:
                cache[stem] = arm(stem)
            v = cache[stem].get(key, {})
            s012 = [v[s] for s in (0, 1, 2) if s in v]
            rec[lab] = round(float(np.mean(s012)), 4) if len(s012) == 3 else None
        out.append(rec)
    have = [r for r in out if r["gpu_exact"] is not None and r["gpu_cam30"] is not None]
    path = STUDY / "data/20261120-tracker_predictions.json"
    path.write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "), "written": time.strftime("%F %H:%M"),
                                "plans": out}, indent=1) + "\n")
    print(f"{len(out)} plans, {len(have)} with both GPU predictions -> {path}")
    for r in sorted(have, key=lambda r: (r["prm"] is None, -(r["gpu_cam30"] or 0))):
        print(f"  {r['plan']:<52} cpu {r['cpu_exact'] if r['cpu_exact'] is not None else float('nan'):.3f}  "
              f"gpu exact {r['gpu_exact']:.3f}  gpu 30 fps {r['gpu_cam30']:.3f}")


if __name__ == "__main__":
    main()
