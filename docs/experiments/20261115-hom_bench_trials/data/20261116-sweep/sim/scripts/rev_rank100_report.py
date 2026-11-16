"""Summary of the 100-layout re-score (rev_rank100_run.sh; hand-4a, 2026-11-16): per layout of the twist ranking test
(rows/mpc_pairs_twist_rank.json) and per model, the best grasp by mean J over seeds 0 and 1 (a run that failed in HOM's
least squares counts as J 0), its tracking, rise and contact latitude; against the grid's own score of the layout
(grid2dff_v4b_sv25r, the bare ball at friction 1.0 on the SCS0009 plant, best of 16 grasps, mean of seeds 0-2,
hom_landscape.track_score). Prints Spearman's rho between the layouts' best J under each pair of scores, how many of
the grid's 10 and 50 best layouts stay among each model's, and each model's ten best layouts.

    .venv/bin/python rev_rank100_report.py [--json rank100_summary.json]
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path.home() / "Programs/hand"
D = ROOT / "logs/20261116-sim_real_reversal"
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401

FPR = ROOT / "docs/experiments/20261029-foundational_pose_review/rows"
MODELS = {"mu 0.5": D / "rescore_rank100_mu0.5.jsonl", "mu 1.0, impratio 1": D / "rescore_rank100_imp1.jsonl"}


def grid_scores():
    """{call: {grasp: mean J over seeds}} of the grid's rerun with the tracking fields, and the grasps' latitudes."""
    import hom_landscape as L
    import hom_rows
    sc = defaultdict(lambda: defaultdict(list))
    for x in (FPR / "grid2dff_v4b_sv25r" / "twist_d40.jsonl").read_text().splitlines():
        if x.strip():
            r = hom_rows.rescale_metrics(json.loads(x))
            if r.get("status", "ok") == "ok":
                sc[int(r["call"])][int(r.get("grasp", 0))].append(L.track_score("twist", r))
    lat, xy = {}, {}
    for x in (FPR / "grid2dff_tripod40_v4b.jsonl").read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            xy[int(r["call"])] = tuple(r["xy"])
            for g, gr in enumerate((r["fp"].get("tripod") or {}).get("grasps") or []):
                lat[(int(r["call"]), g)] = math.degrees(gr["v"][6])
    return {c: {g: float(np.mean(v)) for g, v in d.items()} for c, d in sc.items()}, lat, xy


def model_scores(path):
    by = defaultdict(lambda: defaultdict(list))
    extra = defaultdict(lambda: defaultdict(list))
    for x in Path(path).read_text().splitlines():
        if not x.strip():
            continue
        r = json.loads(x)
        c, g = (int(v) for v in r["grasp"].split("_c")[1].split("_g"))
        by[c][g].append(r["J"] if "error" not in r else 0.0)
        extra[c][g].append(r)
    return ({c: {g: float(np.mean(v)) for g, v in d.items()} for c, d in by.items()}, extra)


def best(d):
    g = max(d, key=d.get)
    return g, d[g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    grid, lat, xy = grid_scores()
    pairs = json.loads((ROOT / "docs/experiments/20261112-inner_loop_benchmark/rows/mpc_pairs_twist_rank.json")
                       .read_text())["pairs"]
    inv = {v: k for k, v in xy.items()}
    calls = [inv[(q["x"], q["y"])] for q in pairs]
    S = {"grid, mu 1.0 bare ball": {c: grid[c] for c in calls if c in grid}}
    E = {}
    for name, p in MODELS.items():
        if p.exists():
            S[name], E[name] = model_scores(p)
    names = list(S)
    common = [c for c in calls if all(c in S[n] and S[n][c] for n in names)]
    out = {"layouts": len(common), "models": names, "rho": {}, "top": {}, "layouts_best": {}}
    print(f"{len(common)} layouts scored under every model")
    vals = {n: np.array([best(S[n][c])[1] for c in common]) for n in names}
    for i, a_ in enumerate(names):
        for b_ in names[i + 1:]:
            rho = float(spearmanr(vals[a_], vals[b_]).statistic)
            out["rho"][f"{a_} | {b_}"] = round(rho, 3)
            print(f"rho of the layouts' best J, {a_} against {b_}: {rho:.2f}")
    order = {n: [common[i] for i in np.argsort(-vals[n])] for n in names}
    g0 = names[0]
    for n in names[1:]:
        for k in (10, 50):
            kept = len(set(order[g0][:k]) & set(order[n][:k]))
            out["top"][f"{n} keeps of the grid's {k} best"] = kept
            print(f"{n}: {kept} of the grid's {k} best layouts among its own {k} best")
    for n in names:
        print(f"\n{n}: ten best layouts (best grasp, its latitude below the equator, mean J"
              + (", track, rise mm)" if n in E else ")"))
        rows = []
        for c in order[n][:10]:
            g, j = best(S[n][c])
            extra = ""
            if n in E:
                rs = [r for r in E[n][c][g] if "error" not in r]
                tr = np.mean([r["track"] or 0 for r in rs]) if rs else 0.0
                ri = np.mean([r["rise_max"] for r in rs]) if rs else float("nan")
                extra = f", track {tr:.2f}, rise {ri:.1f}"
            gb, jb = best(S[g0][c])
            rows.append({"call": c, "xy": xy[c], "grasp": g, "lat_deg": round(lat.get((c, g), float("nan")), 1),
                         "J": round(j, 3), "grid_best_grasp": gb, "grid_J": round(jb, 3)})
            print(f"   ({xy[c][0]:5.1f}, {xy[c][1]:5.1f}) call {c:3d} g{g:<2d} lat {lat.get((c, g), float('nan')):5.1f} J {j:.3f}"
                  + extra + f" | grid: g{gb} J {jb:.3f}")
        out["layouts_best"][n] = rows
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
