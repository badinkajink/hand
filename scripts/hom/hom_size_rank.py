#!/usr/bin/env python3
"""HOM and full annealed sampling on the ranking test's 100 pairs at other object sizes (owner, 2026-11-16: "evaluate on a
wider range of sizes now too ... using/evaluating annealed sampling as the inner loop for search more comprehensively
(this is just rank correlation evaluation again)").

Per task and object diameter, the HOM grid at that size on the benchmark's plant (rows grid2dff_v4b_sv25r/<task>_d<d>.jsonl,
SCS0009 servos, 2.5 MPa pads) chose 100 pairs (inner_loop_mpc.py pairs --d <d> --rule rank: HOM's 50 best layouts and 50
drawn at random from the rest, each with HOM's best grasp), and both controllers ran them at seeds 0 and 1 (rows
mpc_hom_<task>_d<d>_rank.jsonl and mpc_anneal_<task>_d<d>_rank_pdt2.jsonl; the base sizes, 40 mm spheres and 25 mm
cylinders, have no _d<d>). Reported per (task, size), J of a pair the mean over its seeds:

  rho           Spearman correlation of the two controllers' J over the 100 pairs
  seed_rho      each controller's correlation between its seeds 0 and 1; ceiling sqrt(rho_H rho_A)
  cross1        the mean of the four single-seed correlations between the controllers
  mean          each controller's mean J over the pairs
  best          each controller's best pair, with its J, the other controller's J there and its index in the pair file
  best_dist_mm  the distance between the two controllers' best pairs in (x_sep, y_sep)
  top10_x       the x_sep (mm) of each controller's ten best pairs
  smooth        each controller's share of held runs, and over its held runs the median target reversals per
                joint per second, the median RMS target rate (deg/s), the median RMS tracking error (deg) and the mean
                three-pad share
  zero          the pairs at J = 0 under each controller, seeds averaged
  grid          the HOM grid at that size: layouts, mean J of each layout's best grasp, its best layout

    .venv/bin/python scripts/hom/hom_size_rank.py --json PATH [--fig PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

from sr2hand.numerics.stats import spearman  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "docs/experiments/20261112-inner_loop_benchmark/rows"
GRID = ROOT / "docs/experiments/20261029-foundational_pose_review/rows/grid2dff_v4b_sv25r"
BASE = {"twist": 40.0, "roll": 40.0, "pinpush": 25.0, "press": 25.0}
SIZES = {"twist": (28.0, 40.0, 67.0), "roll": (28.0, 40.0, 67.0), "pinpush": (18.0, 25.0, 45.0),
         "press": (18.0, 25.0, 45.0)}
N_PAIRS, SEEDS = 100, (0, 1)


def dtag(task, d):
    return "" if d == BASE[task] else f"_d{d:g}"


def load(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def by_pair(rows):
    """{(x_sep, y_sep, grasp): {seed: row}}, the last row of a world kept."""
    out = {}
    for r in rows:
        out.setdefault((r["x_sep"], r["y_sep"], r["grasp"]), {})[r["seed"]] = r
    return out


def smooth(D, keys):
    """Share of held runs over the pairs and seeds, and over the held runs the median target reversals per joint per
    second and the median RMS target rate."""
    runs = [D[k][s] for k in keys for s in SEEDS]
    h = [r for r in runs if r.get("held")]
    md = (lambda k: float(np.median([r[k] for r in h if r.get(k) is not None])) if h else None)
    return {"held": len(h) / len(runs), "reversals_ps": md("reversals_ps"), "cmd_rate_dps": md("cmd_rate_dps"),
            "err_med": md("track_err_rms_deg"), "contact": float(np.mean([r["contact_all_frac"] for r in h])) if h else None}


def grid_summary(task, d):
    """The HOM grid at that size: layouts, mean J of each layout's best grasp (seeds 0-2), its best layout."""
    import hom_landscape as L
    import hom_rows
    p = GRID / f"{task}_d{d:g}.jsonl"
    if not p.exists():
        return None
    J = {}
    for r in load(p):
        r = hom_rows.rescale_metrics(r)
        if r.get("status", "ok") != "ok":
            continue
        dd = r["d"]
        J.setdefault((round(dd[2] - dd[0], 3), round(dd[3] - dd[5], 3), int(r.get("grasp", 0))), []).append(
            L.track_score(task, r))
    best = {}
    for (x, y, g), v in J.items():
        j = float(np.mean(v))
        if (x, y) not in best or j > best[(x, y)][1]:
            best[(x, y)] = (g, j)
    if not best:
        return None
    k = max(best, key=lambda k_: (best[k_][1], -k_[0], -k_[1]))
    return {"layouts": len(best), "mean_J": float(np.mean([v[1] for v in best.values()])),
            "best": {"x": k[0], "y": k[1], "grasp": best[k][0], "J": best[k][1]}}


def cell(task, d):
    tg = dtag(task, d)
    H = by_pair(load(ROWS / f"mpc_hom_{task}{tg}_rank.jsonl"))
    A = by_pair(load(ROWS / f"mpc_anneal_{task}{tg}_rank_pdt2.jsonl"))
    keys = sorted(k for k in H if set(H[k]) >= set(SEEDS) and set(A.get(k, {})) >= set(SEEDS))
    pf = ROWS / f"mpc_pairs_{task}{tg}_rank.json"
    idx = {(q["x"], q["y"], q["grasp"]): i for i, q in enumerate(json.loads(pf.read_text())["pairs"])} if pf.exists() else {}
    out = {"task": task, "d": d, "base": d == BASE[task], "pairs": len(keys), "done": len(keys) >= N_PAIRS,
           "rows": {"H": sum(len(v) for v in H.values()), "A": sum(len(v) for v in A.values())},
           "grid": grid_summary(task, d)}
    if len(keys) < 10:
        return out
    jm = {nm: np.array([np.mean([float(D[k][s]["J"]) for s in SEEDS]) for k in keys]) for nm, D in (("H", H), ("A", A))}
    j1 = {nm: {s: np.array([float(D[k][s]["J"]) for k in keys]) for s in SEEDS} for nm, D in (("H", H), ("A", A))}
    seed_rho = {nm: spearman(j1[nm][0], j1[nm][1]) for nm in "HA"}
    xs = np.array([k[0] for k in keys])
    best, top = {}, {}
    for nm, other in (("H", "A"), ("A", "H")):
        o = np.argsort(-jm[nm], kind="stable")
        i = int(o[0])
        best[nm] = {"x": keys[i][0], "y": keys[i][1], "grasp": keys[i][2], "J": float(jm[nm][i]),
                    "J_other": float(jm[other][i]), "pair": idx.get(keys[i])}
        top[nm] = sorted(float(xs[j]) for j in o[:10])
    out.update({"rho": spearman(jm["H"], jm["A"]), "seed_rho": seed_rho,
                "ceiling": float(np.sqrt(max(seed_rho["H"] * seed_rho["A"], 0.0))),
                "cross1": float(np.mean([spearman(j1["H"][a], j1["A"][b]) for a in SEEDS for b in SEEDS])),
                "mean": {nm: float(jm[nm].mean()) for nm in "HA"}, "zero": {nm: int((jm[nm] == 0).sum()) for nm in "HA"},
                "smooth": {"H": smooth(H, keys), "A": smooth(A, keys)},
                "best": best, "top10_x": top,
                "top10_x_med": {nm: float(np.median(top[nm])) for nm in "HA"},
                "best_dist_mm": float(np.hypot(best["H"]["x"] - best["A"]["x"], best["H"]["y"] - best["A"]["y"])),
                "top10_overlap": int(len(set(np.argsort(-jm["H"], kind="stable")[:10])
                                         & set(np.argsort(-jm["A"], kind="stable")[:10])))})
    return out


def figure(R, out):
    """Per task, the best pair's x_sep against the object's diameter under each controller, and the median x_sep of
    each one's ten best pairs; dotted, a slope of 1 mm/mm through HOM's best at the base size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # shown at 828 px of 1,144, so 12.5 pt at 110 dpi is 13.8 px on the page
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13.5,
                         "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 13})
    TN = {"twist": "Twist", "roll": "Roll", "pinpush": "Pin-push", "press": "Press"}
    CH, CA = "#2563eb", "#f97316"
    fig, ax = plt.subplots(2, 2, figsize=(10.4, 7.4), layout="constrained")
    ax = ax.ravel()
    for g, t in zip(ax, ("twist", "roll", "pinpush", "press")):
        cs = [c for c in R[t] if "best" in c]
        for nm, col, lab, off in (("H", CH, "HOM", -0.5), ("A", CA, "Full annealed sampling", 0.5)):
            d = [c["d"] + off for c in cs]                  # half a millimetre apart, so that neither hides the other
            g.plot(d, [c["best"][nm]["x"] for c in cs], "o-", color=col, lw=2, ms=7, label=f"{lab}, best pair")
            g.plot(d, [c["top10_x_med"][nm] for c in cs], "s--", color=col, lw=1.4, ms=5, alpha=0.8,
                   label=f"{lab}, median of ten best")
        b = next((c for c in cs if c["base"]), None)
        if b:
            dd = np.array([min(SIZES[t]), max(SIZES[t])])
            g.plot(dd, b["best"]["H"]["x"] + (dd - b["d"]), ":", color="#6b7280", lw=1.2, label="Slope 1 mm/mm")
        g.set_title(TN[t])
        g.set_xlabel("Object diameter (mm)")
        g.set_xticks(SIZES[t])
        g.set_ylabel("$x_{\\mathrm{sep}}$ (mm)")
        for sp in ("top", "right"):
            g.spines[sp].set_visible(False)
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="outside lower center", ncol=2)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default=None)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    R = {t: [cell(t, d) for d in SIZES[t]] for t in SIZES}
    for t, cs in R.items():
        for c in cs:
            g = c["grid"]
            gs = (f"grid {g['layouts']} layouts, mean {g['mean_J']:.3f}, best ({g['best']['x']:g}, {g['best']['y']:g})"
                  f" {g['best']['J']:.2f}" if g else "no grid")
            if "rho" not in c:
                print(f"{t:8s} {c['d']:4g} mm: {c['pairs']} pairs with both controllers (rows H {c['rows']['H']}, "
                      f"A {c['rows']['A']}); {gs}")
                continue
            print(f"{t:8s} {c['d']:4g} mm: {c['pairs']} pairs; rho {c['rho']:+.2f}; seeds H {c['seed_rho']['H']:+.2f} "
                  f"A {c['seed_rho']['A']:+.2f}; single {c['cross1']:+.2f} (ceiling {c['ceiling']:.2f}); mean J H "
                  f"{c['mean']['H']:.3f} A {c['mean']['A']:.3f}; best H ({c['best']['H']['x']:g}, {c['best']['H']['y']:g})"
                  f" {c['best']['H']['J']:.2f}/{c['best']['H']['J_other']:.2f}, A ({c['best']['A']['x']:g}, "
                  f"{c['best']['A']['y']:g}) {c['best']['A']['J']:.2f}/{c['best']['A']['J_other']:.2f}; ten best x_sep "
                  f"median H {c['top10_x_med']['H']:g} A {c['top10_x_med']['A']:g}; overlap {c['top10_overlap']}; {gs}")
            sm = c["smooth"]
            print(f"{'':16s}held H {sm['H']['held']:.2f} A {sm['A']['held']:.2f}; reversals/s H {sm['H']['reversals_ps']:.2f} "
                  f"A {sm['A']['reversals_ps']:.2f}; target rate H {sm['H']['cmd_rate_dps']:.1f} A "
                  f"{sm['A']['cmd_rate_dps']:.1f} deg/s; J = 0 H {c['zero']['H']} A {c['zero']['A']}; tracking error H "
                  f"{sm['H']['err_med']:.1f} A {sm['A']['err_med']:.1f} deg; three pads H {sm['H']['contact']:.2f} A "
                  f"{sm['A']['contact']:.2f}")
    if a.json:
        Path(a.json).write_text(json.dumps(R, indent=1, default=float) + "\n")
    if a.fig:
        figure(R, a.fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
