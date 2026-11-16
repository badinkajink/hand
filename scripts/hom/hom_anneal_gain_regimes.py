#!/usr/bin/env python3
"""Where full annealed sampling gains on HOM over a task's grid (owner, 2026-11-16: "can we characterize in what regimes
annealed sampling recovers much better performance than hom, e.g. there is less rank correlation?").

Both controllers ran every layout of the four grids with HOM's best grasp of the layout, seeds 0 and 1 (rows
mpc_hom_<task>_all.jsonl and mpc_anneal_<task>_all_pdt2.jsonl in docs/experiments/20261112-inner_loop_benchmark/rows).
J is the tracking factor times the three-pad share (hom_inner_loop_page.metric_vectors). Per layout, seeds averaged:

  gain          J under full annealed sampling minus J under HOM
  tracking part Delta tf (c_H + c_A) / 2 per seed: the part of the gain from the tracking factor
  contact part  Delta c (tf_H + tf_A) / 2 per seed: the part from the three-pad share (the two parts sum to the gain)
  rank shift    the layout's percentile rank by J under the planner minus its rank under HOM

HOM's run on a layout falls in one of four classes by its outcome at both seeds: dropped (the object fell in a seed),
untracked (held, tracking factor 0: no closer than holding the object still), lost pad (held and tracking, three pads
in contact for less than 0.9 of the task) and kept (the rest). Reported per task and class: layouts, mean J of both
controllers, mean gain and its two parts, the Spearman correlation of the two controllers' J within the class, and the
planner's ten and 50 best layouts by class. Reported per task: the Spearman correlation of the gain and of the rank
shift with each of HOM's run measures and with the layout coordinates.

    .venv/bin/python scripts/hom/hom_anneal_gain_regimes.py --json PATH [--fig PATH]

The tracking error, the three-pad share and the seed agreement are reported beside the classes: per task, each
controller's seed-to-seed Spearman correlation, their single-seed correlation with the ceiling sqrt(rho_H rho_A) the
seeds allow, the layouts at J = 0, the correlation over the layouts where both score above 0 and, with J's floor
removed, of c (1 - e / e_ref) over the layouts both held; the Spearman correlation of each controller's error with
x_sep; and per x_sep value the means of the figure (--fig).
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
TASKS = ("twist", "roll", "pinpush", "press")
CLASSES = ("kept", "lost pad", "untracked", "dropped")
C_KEPT = 0.9
MEASURES = ("J", "tf", "c", "held", "limit_frac", "sat_frac", "link_mean_N", "link_frac_1N", "pad_share", "grip_mean_N",
            "pad_max_N", "reversals_ps", "cmd_rate_dps", "track_err_rms_deg")


def load(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def e_ref_of(rows):
    """The task's RMS commanded rotation (deg): e / (1 - J / c) of a held run with J > 0 (hom_inner_loop_page.e_ref_of)."""
    v = [r["track_err_rms_deg"] / (1.0 - r["J"] / r["contact_all_frac"]) for r in rows
         if r.get("held") and r["J"] > 0 and r.get("contact_all_frac") and r["J"] < r["contact_all_frac"]]
    return float(np.median(v))


def per_run(r, e_ref):
    held = bool(r.get("held"))
    tf = max(0.0, 1.0 - r["track_err_rms_deg"] / e_ref) if held and r.get("track_err_rms_deg") is not None else 0.0
    c = float(r.get("contact_all_frac") or 0.0) if held else 0.0
    out = {"J": float(r["J"]), "tf": tf, "c": c, "held": float(held)}
    for m in MEASURES[4:]:
        v = r.get(m)
        out[m] = float(v) if v is not None else np.nan
    return out


def layouts(task):
    """Per layout (x_sep, y_sep, grasp): each controller's runs by seed."""
    H, A = load(ROWS / f"mpc_hom_{task}_all.jsonl"), load(ROWS / f"mpc_anneal_{task}_all_pdt2.jsonl")
    e_ref = e_ref_of(H + A)
    by = {}
    for nm, rows in (("H", H), ("A", A)):
        for r in rows:
            by.setdefault((r["x_sep"], r["y_sep"], r["grasp"]), {}).setdefault(nm, {})[r["seed"]] = per_run(r, e_ref)
    keys = sorted(k for k, v in by.items() if set(v.get("H", {})) == {0, 1} and set(v.get("A", {})) == {0, 1})
    err = max(abs(s["J"] - s["tf"] * s["c"]) for k in keys for nm in "HA" for s in by[k][nm].values())
    return keys, by, e_ref, err


def classify(runs):
    """HOM's outcome on a layout over its two seeds."""
    if any(s["held"] < 1 for s in runs.values()):
        return "dropped"
    if any(s["tf"] <= 0 for s in runs.values()):
        return "untracked"
    if np.mean([s["c"] for s in runs.values()]) < C_KEPT:
        return "lost pad"
    return "kept"


def pct_rank(v):
    """Percentile rank (0-1, ties averaged)."""
    o = np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
    v = np.asarray(v)
    for u in np.unique(v):
        m = v == u
        o[m] = o[m].mean()
    return o / max(len(v) - 1, 1)


def analyse(task):
    keys, by, e_ref, err_id = layouts(task)
    n = len(keys)
    M = {nm: {m: np.array([np.nanmean([by[k][nm][s][m] for s in (0, 1)]) for k in keys]) for m in MEASURES}
         for nm in "HA"}
    gain = M["A"]["J"] - M["H"]["J"]
    tpart = np.array([np.mean([(by[k]["A"][s]["tf"] - by[k]["H"][s]["tf"]) * (by[k]["A"][s]["c"] + by[k]["H"][s]["c"]) / 2
                               for s in (0, 1)]) for k in keys])
    cpart = gain - tpart
    shift = pct_rank(M["A"]["J"]) - pct_rank(M["H"]["J"])
    cls = np.array([classify(by[k]["H"]) for k in keys])
    cls_a = np.array([classify(by[k]["A"]) for k in keys])
    top10, top50 = np.argsort(-M["A"]["J"], kind="stable")[:10], np.argsort(-M["A"]["J"], kind="stable")[:50]
    htop10 = np.argsort(-M["H"]["J"], kind="stable")[:10]
    xs, ys = np.array([k[0] for k in keys]), np.array([k[1] for k in keys])
    per_cls = {}
    for c in CLASSES:
        m = cls == c
        if not m.any():
            per_cls[c] = {"n": 0}
            continue
        per_cls[c] = {"n": int(m.sum()), "J_H": float(M["H"]["J"][m].mean()), "J_A": float(M["A"]["J"][m].mean()),
                      "gain": float(gain[m].mean()), "tpart": float(tpart[m].mean()), "cpart": float(cpart[m].mean()),
                      "rho": spearman(M["H"]["J"][m], M["A"]["J"][m]) if m.sum() >= 5 else None,
                      "top10": int(np.isin(np.nonzero(m)[0], top10).sum()),
                      "top50": int(np.isin(np.nonzero(m)[0], top50).sum()),
                      "c_H": float(M["H"]["c"][m].mean()), "c_A": float(M["A"]["c"][m].mean()),
                      "tf_H": float(M["H"]["tf"][m].mean()), "tf_A": float(M["A"]["tf"][m].mean()),
                      "planner_kept": int((cls_a[m] == "kept").sum())}
    corr = {}
    feats = {f"hom_{m}": M["H"][m] for m in MEASURES} | {"x_sep": xs, "y_sep": ys}
    for f, v in feats.items():
        ok = np.isfinite(v)
        if ok.sum() >= 10 and np.ptp(v[ok]) > 0:
            corr[f] = {"gain": spearman(v[ok], gain[ok]), "shift": spearman(v[ok], shift[ok])}
    # The planner's own measures beside HOM's on the layouts where it gained most (top quartile of the gain)
    q = gain >= np.quantile(gain, 0.75)
    contrast = {m: (float(np.nanmean(M["H"][m][q])), float(np.nanmean(M["A"][m][q])),
                    float(np.nanmean(M["H"][m][~q])), float(np.nanmean(M["A"][m][~q]))) for m in MEASURES}
    # Seed agreement: each controller's two seeds, the two controllers at single seeds, and the ceiling the seeds allow
    J1 = {nm: {s: np.array([by[k][nm][s]["J"] for k in keys]) for s in (0, 1)} for nm in "HA"}
    seed_rho = {nm: spearman(J1[nm][0], J1[nm][1]) for nm in "HA"}
    cross1 = float(np.mean([spearman(J1["H"][i], J1["A"][j]) for i in (0, 1) for j in (0, 1)]))
    # The tracking error (deg) of the held runs, and the three-pad share of the held runs, per layout
    def held_mean(nm, m):
        v = [[by[k][nm][s][m] for s in (0, 1) if by[k][nm][s]["held"]] for k in keys]
        return np.array([np.mean(x) if x else np.nan for x in v])
    err = {nm: held_mean(nm, "track_err_rms_deg") for nm in "HA"}
    c_held = {nm: held_mean(nm, "c") for nm in "HA"}
    drop = {nm: 1.0 - M[nm]["held"] for nm in "HA"}
    both_held = np.array([all(by[k][nm][s]["held"] for nm in "HA" for s in (0, 1)) for k in keys])
    unc = {nm: np.array([np.mean([by[k][nm][s]["c"] * (1.0 - by[k][nm][s]["track_err_rms_deg"] / e_ref) for s in (0, 1)])
                         if both_held[i] else np.nan for i, k in enumerate(keys)]) for nm in "HA"}
    pos = (M["H"]["J"] > 0) & (M["A"]["J"] > 0)
    okH, okA = np.isfinite(err["H"]), np.isfinite(err["A"])
    prof = {}
    for x in np.unique(xs):
        m = xs == x
        prof[f"{x:g}"] = {"n": int(m.sum()), "J_H": float(M["H"]["J"][m].mean()), "J_A": float(M["A"]["J"][m].mean()),
                          "e_H": float(np.nanmedian(err["H"][m])) if np.isfinite(err["H"][m]).any() else None,
                          "e_A": float(np.nanmedian(err["A"][m])) if np.isfinite(err["A"][m]).any() else None,
                          "c_H": float(np.nanmean(c_held["H"][m])) if np.isfinite(c_held["H"][m]).any() else None,
                          "c_A": float(np.nanmean(c_held["A"][m])) if np.isfinite(c_held["A"][m]).any() else None,
                          "drop_H": float(drop["H"][m].mean()), "drop_A": float(drop["A"][m].mean())}
    # Film candidates: the largest gain where HOM's held error exceeds the commanded rotation and the planner kept three
    # pads; and a layout where both kept three pads and the gain is near 0
    over = np.nonzero((err["H"] > e_ref) & (cls_a == "kept"))[0]
    even = np.nonzero((cls == "kept") & (cls_a == "kept") & (np.abs(gain) < 0.03))[0]
    film = {"gain": keys[int(over[np.argmax(gain[over])])] if len(over) else None,
            "even": keys[int(even[np.argmax(M["H"]["J"][even])])] if len(even) else None}
    return {"task": task, "n": n, "e_ref": e_ref, "J_identity_err": err_id,
            "seed_rho": seed_rho, "cross1": cross1, "ceiling": float(np.sqrt(max(seed_rho["H"] * seed_rho["A"], 0.0))),
            "zero": {"H": int((M["H"]["J"] == 0).sum()), "A": int((M["A"]["J"] == 0).sum())},
            "rho_pos": (int(pos.sum()), spearman(M["H"]["J"][pos], M["A"]["J"][pos])),
            "rho_unc": (int(both_held.sum()), spearman(unc["H"][both_held], unc["A"][both_held])),
            "err_med": {nm: float(np.nanmedian(err[nm])) for nm in "HA"},
            "err_over_ref_H": float(np.mean(err["H"][okH] > e_ref)),
            "err_x_rho": {"H": spearman(xs[okH], err["H"][okH]), "A": spearman(xs[okA], err["A"][okA])},
            "err_y_rho": {"H": spearman(ys[okH], err["H"][okH]), "A": spearman(ys[okA], err["A"][okA])},
            "gain_ge": {"0.2": float(np.mean(gain >= 0.2)), "0.3": float(np.mean(gain >= 0.3)),
                        "-0.05": float(np.mean(gain <= -0.05))},
            "top10_x": {"H": sorted(float(xs[i]) for i in htop10), "A": sorted(float(xs[i]) for i in top10)},
            "profile": prof, "film": film,
            "rho": spearman(M["H"]["J"], M["A"]["J"]), "mean_H": float(M["H"]["J"].mean()),
            "mean_A": float(M["A"]["J"].mean()), "gain": float(gain.mean()), "tpart": float(tpart.mean()),
            "cpart": float(cpart.mean()), "classes": per_cls, "corr": corr, "contrast_topq": contrast,
            "planner_classes": {c: int((cls_a == c).sum()) for c in CLASSES},
            "top10_overlap": int(np.isin(top10, htop10).sum()),
            "layouts": [{"x": k[0], "y": k[1], "grasp": k[2], "J_H": float(M["H"]["J"][i]), "J_A": float(M["A"]["J"][i]),
                         "tpart": float(tpart[i]), "cpart": float(cpart[i]), "cls": str(cls[i]), "cls_A": str(cls_a[i]),
                         "c_H": float(M["H"]["c"][i]), "c_A": float(M["A"]["c"][i]), "tf_H": float(M["H"]["tf"][i]),
                         "tf_A": float(M["A"]["tf"][i]), "limit_H": float(M["H"]["limit_frac"][i]),
                         "link_H": float(M["H"]["link_mean_N"][i])} for i, k in enumerate(keys)]}


def report(R):
    for t, a in R.items():
        print(f"\n== {t}: {a['n']} layouts, e_ref {a['e_ref']:.1f} deg (|J - tf c| max {a['J_identity_err']:.1e}); rho "
              f"{a['rho']:+.2f}; mean J HOM {a['mean_H']:.3f}, planner {a['mean_A']:.3f}; gain {a['gain']:+.3f} = "
              f"tracking {a['tpart']:+.3f} + contact {a['cpart']:+.3f}; planner's top 10 in HOM's top 10: "
              f"{a['top10_overlap']}; planner's classes {a['planner_classes']}")
        print(f"   seeds rho HOM {a['seed_rho']['H']:+.2f} planner {a['seed_rho']['A']:+.2f}; single-seed cross "
              f"{a['cross1']:+.2f} (ceiling {a['ceiling']:.2f}); J = 0 on {a['zero']['H']} (HOM) and {a['zero']['A']} "
              f"(planner); rho both > 0 {a['rho_pos'][1]:+.2f} ({a['rho_pos'][0]}), unclipped held by both "
              f"{a['rho_unc'][1]:+.2f} ({a['rho_unc'][0]}); error median {a['err_med']['H']:.1f} / {a['err_med']['A']:.1f} "
              f"deg, HOM over e_ref on {a['err_over_ref_H']:.2f}; error vs x_sep rho {a['err_x_rho']['H']:+.2f} / "
              f"{a['err_x_rho']['A']:+.2f}; gain >= 0.2 on {a['gain_ge']['0.2']:.2f}; top10 x_sep HOM "
              f"{a['top10_x']['H']} planner {a['top10_x']['A']}; film {a['film']}")
        print(f"   {'class':10s} {'n':>4s} {'J_H':>6s} {'J_A':>6s} {'gain':>7s} {'track':>7s} {'contact':>7s} {'rho':>6s} "
              f"{'top10':>5s} {'top50':>5s} {'c_H':>5s} {'c_A':>5s} {'tf_H':>5s} {'tf_A':>5s} {'A kept':>6s}")
        for c, s in a["classes"].items():
            if not s["n"]:
                print(f"   {c:10s} {0:4d}")
                continue
            rho = f"{s['rho']:+.2f}" if s["rho"] is not None else "   -"
            print(f"   {c:10s} {s['n']:4d} {s['J_H']:6.3f} {s['J_A']:6.3f} {s['gain']:+7.3f} {s['tpart']:+7.3f} "
                  f"{s['cpart']:+7.3f} {rho:>6s} {s['top10']:5d} {s['top50']:5d} {s['c_H']:5.2f} {s['c_A']:5.2f} "
                  f"{s['tf_H']:5.2f} {s['tf_A']:5.2f} {s['planner_kept']:6d}")
        print("   Spearman with the gain / the rank shift:")
        for f, v in sorted(a["corr"].items(), key=lambda kv: -abs(kv[1]["shift"])):
            print(f"     {f:22s} gain {v['gain']:+.2f}  shift {v['shift']:+.2f}")
        print("   top-quartile gain layouts vs the rest, HOM / planner:")
        for m, (h1, a1, h0, a0) in a["contrast_topq"].items():
            print(f"     {m:20s} top q: {h1:8.3f} / {a1:8.3f}   rest: {h0:8.3f} / {a0:8.3f}")


def figure(R, out):
    """Per task (columns), against the thumb-to-pair distance x_sep, each controller's mean over the layouts at that
    x_sep (y_sep and grasp vary): J; the median RMS tracking error of the held runs with the task's commanded rotation;
    the mean three-pad share of the held runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 12})
    TN = {"twist": "Twist", "roll": "Roll", "pinpush": "Pin-push", "press": "Press"}
    CH, CA = "#2563eb", "#f97316"
    fig, ax = plt.subplots(3, len(R), figsize=(10.4, 8.6), sharex="col", layout="constrained", squeeze=False)
    for j, (t, a) in enumerate(R.items()):
        P = a["profile"]
        x = np.array([float(k) for k in P])
        for i, (mh, ma, lab) in enumerate((("J_H", "J_A", "J"), ("e_H", "e_A", "Tracking error (deg)"),
                                           ("c_H", "c_A", "Three-pad share"))):
            g = ax[i, j]
            top = a["e_ref"]
            for m, col, nm in ((mh, CH, "HOM"), (ma, CA, "Full annealed sampling")):
                y = np.array([np.nan if P[k][m] is None else P[k][m] for k in P], float)
                g.plot(x, y, color=col, lw=2.2, label=nm)
                top = max(top, float(np.nanmax(y)))
            if i == 1:
                g.axhline(a["e_ref"], color="#6b7280", lw=1.2, ls=(0, (4, 3)), label="Commanded rotation, RMS")
                g.set_ylim(0, 1.08 * top)
            else:
                g.set_ylim(0, 1.0)
            if j == 0:
                g.set_ylabel(lab)
            if i == 0:
                g.set_title(f"{TN[t]}: \u03c1 {a['rho']:.2f}")
            if i == 2:
                g.set_xlabel("$x_{sep}$ (mm)")
            for sp in ("top", "right"):
                g.spines[sp].set_visible(False)
    h1, l1 = ax[1, 0].get_legend_handles_labels()
    fig.legend(h1, l1, frameon=False, loc="outside lower center", ncol=3)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default=None)
    ap.add_argument("--fig", default=None, help="PNG of J, tracking error and three-pad share against x_sep")
    a = ap.parse_args()
    R = {t: analyse(t) for t in TASKS}
    report(R)
    if a.json:
        Path(a.json).write_text(json.dumps(R, indent=1) + "\n")
    if a.fig:
        figure(R, a.fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
