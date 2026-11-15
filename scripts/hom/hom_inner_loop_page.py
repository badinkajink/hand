#!/usr/bin/env python3
"""Builds docs/experiments/20261114-hom_inner_loop/20261114-hom_inner_loop_stability_variants_fusion.html: HOM as the
inner-loop controller of the SR2 hand search, against sampling MPC (owner, 2026-11-14: "i would not like to stop using
HOM, it is quite nice to have such a fast controller. if theres anything else you can think of to improve it, feel free
to try. is there a better algorithm to use than DLS? can we fuse methods? ... did we try gradient based methods?"; "i
wonder if there are emergent trends in the hyperparameters such as impedance forces, twist weights"; "ideally we have
further evidence that HOM is the right choice ... stability across metrics, tasks, solutions? ... explore this").

Data:
  - the CEM refinements of HOM's forces and contact modes under J of 2026-11-11 (scripts/hom/hom_grasp_refine.py,
    levels hom and shared, docs/experiments/20261103-hom_score_refine/rows/sv25_track/), reduced to each layout's pick
    in data/20261111-hom_picks_track.json (refreshed with --picks);
  - the runs of scripts/hom/inner_loop_mpc.py on the benchmark's plant (rows of
    docs/experiments/20261112-inner_loop_benchmark/rows/): HOM, real-time and full DIAL on the four tasks' pair sets
    _tune, _rank and _rank20, HOM's variants (--preview, --prm-shared, --hom-param, --variant floop) and the planners
    fused with HOM (res_anneal, hs_anneal);
  - the grids grid2dff_v4b_sv25r of HOM on every layout (seeds 0-2).

    .venv/bin/python scripts/hom/hom_inner_loop_page.py [--picks]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import docs_index  # noqa: E402
import fp_review_page as P  # noqa: E402
import inner_loop_mpc_page as MP  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
from sr2hand.numerics.stats import spearman  # noqa: E402

D = ROOT / "docs/experiments/20261114-hom_inner_loop"
MEDIA, DATA = D / "media", D / "data"
OUT = D / "20261114-hom_inner_loop_stability_variants_fusion.html"
TEX_CACHE = D / "texsvg_cache.json"
MPC_PAGE = "../20261112-inner_loop_benchmark/20261113-sampling_mpc_vs_hom.html"
REFINE = ROOT / "docs/experiments/20261103-hom_score_refine"
REFINE_PAGE = "../20261103-hom_score_refine/20261103-hom_score_refine.html"
TRACK = REFINE / "rows/sv25_track"
TASKS = ("twist", "roll", "pinpush", "press")
D_MM = {"twist": 40, "roll": 40, "pinpush": 25, "press": 25}
TASK_NAME = {"twist": "twist", "roll": "roll", "pinpush": "pin-push", "press": "press"}
# HOM's defaults (hom_tasks.PRM, mode_weights) for the forces and contact modes that level hom searched
DEFAULT = {"f_tripod": 2.5, "f_pinch": 3.0, "f_drive": 1.5, "f_press": 0.5, "w_spin_tripod": 1.0, "w_roll_tripod": 0.0,
           "w_slide_tripod": 1.0, "w_spin_pinch": 0.0, "w_roll_pinch": 0.0, "w_slide_pinch": 1.0, "w_spin_push": 1.0,
           "w_roll_push": 1.0, "w_slide_push": 1.0}


def esc(s):
    return P.esc(s)


# ------------------------------------------------------------------------------------------------ the 11-11 picks
def picks_from_rows():
    """Each layout's pick of level hom (the candidate with the best mean J over seeds 0-2 of the search) with its
    forces and mode weights, the layout's coordinates and the default's J (candidate 0 of iteration 0, HOM's defaults),
    for the tasks that level hom ran; and each task's shared pick (level shared)."""
    import hom_landscape as L
    import hom_rows
    out = {"source": str(TRACK.relative_to(ROOT)), "tasks": {}}
    for task in TASKS:
        p = TRACK / "hom" / f"{task}_d{D_MM[task]}.jsonl"
        sh = TRACK / "shared" / f"{task}_d{D_MM[task]}_pick.json"
        rec = {"shared": json.loads(sh.read_text())["pick"]["prm"] if sh.exists() else None}
        if p.exists():
            cand, prm_of, d_of = collections.defaultdict(list), {}, {}
            for line in p.read_text().splitlines():
                r = hom_rows.rescale_metrics(json.loads(line))
                k = (r["call"], r["it"], r["cand"])
                cand[k].append(L.track_score(task, r) if r.get("status", "ok") == "ok" else 0.0)
                prm_of[k], d_of[r["call"]] = r["prm"], r["d"]
            best, base = {}, {}
            for (call, it, c), v in cand.items():
                m = float(np.mean(v))
                if call not in best or m > best[call][0]:
                    best[call] = (m, prm_of[(call, it, c)])
                if it == 0 and c == 0:
                    base[call] = m
            calls = sorted(best)
            rec.update({"calls": calls, "J_pick": [best[c][0] for c in calls], "J_default": [base[c] for c in calls],
                        "x_sep": [round(d_of[c][2] - d_of[c][0], 3) for c in calls],
                        "y_sep": [round(d_of[c][3] - d_of[c][5], 3) for c in calls],
                        "prm": {k: [best[c][1][k] for c in calls] for k in sorted(best[calls[0]][1])}})
        out["tasks"][task] = rec
    lv = json.loads((TRACK / "20261111-tuning_levels.json").read_text())["tasks"]
    for task in TASKS:
        t = lv.get(task, {})
        out["tasks"][task]["val"] = {k: {"median": t[k]["median"], "gain": t[k]["gain"]["median"],
                                         "up": t[k]["gain"]["up"], "down": t[k]["gain"]["down"],
                                         "rho_s1": t[k]["rho_s1_val"]}
                                     for k in ("hom_val", "shared_val", "both_val") if k in t}
        out["tasks"][task]["val"]["s1_median"] = t.get("median_s1_val")
    return out


def picks(refresh=False):
    path = DATA / "20261111-hom_picks_track.json"
    if refresh or not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(picks_from_rows(), indent=1))
    return json.loads(path.read_text())


FORCE_KEYS = (("twist", "f_tripod", "twist, pad force"), ("roll", "f_tripod", "roll, pad force"),
              ("press", "f_pinch", "press, pinch force"), ("press", "f_press", "press, pusher force"))
MODE_KEYS = (("twist", "w_spin_tripod", "twist, spin"), ("twist", "w_slide_tripod", "twist, slide"),
             ("roll", "w_spin_tripod", "roll, spin"), ("roll", "w_slide_tripod", "roll, slide"),
             ("press", "w_slide_pinch", "press, pinch slide"), ("press", "w_spin_push", "press, pusher spin"),
             ("press", "w_roll_push", "press, pusher roll"), ("press", "w_slide_push", "press, pusher slide"))


def fig_trends(PK, out):
    """(a) The picked forces of the 30 best layouts per task as a share of HOM's default, the shared pick marked; (b) the
    roll's picked pad force against x_sep, coloured by y_sep; (c) the picked mode weights against the defaults."""
    plt = MP.plt_setup()
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6), gridspec_kw={"width_ratios": [1.0, 1.0, 1.45]})
    rng = np.random.default_rng(0)
    for i, (task, k, lab) in enumerate(FORCE_KEYS):
        T = PK["tasks"][task]
        v = np.array(T["prm"][k]) / DEFAULT[k]
        ax[0].scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=22, color="#2563eb", alpha=0.75, lw=0)
        ax[0].plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="#1e3a8a", lw=2.5)
        if T.get("shared") and k in T["shared"]:
            ax[0].scatter([i + 0.32], [T["shared"][k] / DEFAULT[k]], marker="D", s=60, color="#d97706", zorder=3)
    ax[0].axhline(1.0, color="#6b7280", lw=1.0, ls=(0, (4, 3)))
    ax[0].set_xticks(range(len(FORCE_KEYS)))
    ax[0].set_xticklabels([lab.replace(", ", ",\n") for _t, _k, lab in FORCE_KEYS], fontsize=12, rotation=25, ha="right")
    ax[0].set_ylabel("Picked force / HOM's default")
    ax[0].set_title("(a) Picked forces", loc="left")
    T = PK["tasks"]["roll"]
    sc = ax[1].scatter(T["x_sep"], T["prm"]["f_tripod"], c=T["y_sep"], cmap="viridis", s=42, lw=0)
    ax[1].axhline(DEFAULT["f_tripod"], color="#6b7280", lw=1.0, ls=(0, (4, 3)))
    cb = fig.colorbar(sc, ax=ax[1], pad=0.02)
    cb.set_label("y_sep (mm)")
    ax[1].set_xlabel("x_sep (mm)")
    ax[1].set_ylabel("Picked pad force (N)")
    r1 = spearman(T["x_sep"], T["prm"]["f_tripod"])
    r2 = spearman(T["y_sep"], T["prm"]["f_tripod"])
    ax[1].set_title(f"(b) Roll: rho {r1:+.2f} with x_sep,\n{r2:+.2f} with y_sep", loc="left")
    for i, (task, k, lab) in enumerate(MODE_KEYS):
        v = np.array(PK["tasks"][task]["prm"][k])
        ax[2].scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=18, color="#2563eb", alpha=0.7, lw=0)
        ax[2].plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="#1e3a8a", lw=2.5)
        ax[2].scatter([i], [DEFAULT[k]], marker="_", s=380, color="#dc2626", lw=2.5, zorder=3)
    ax[2].set_xticks(range(len(MODE_KEYS)))
    ax[2].set_xticklabels([lab for _t, _k, lab in MODE_KEYS], fontsize=12, rotation=30, ha="right")
    ax[2].set_ylim(-0.05, 1.08)
    ax[2].set_ylabel("Picked weight of Eq. (2)'s row")
    ax[2].set_title("(c) Picked contact-mode weights", loc="left")
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def trend_numbers(PK):
    """Medians, quartiles and rank correlations of each picked parameter with the layout's coordinates and the default's
    J, per task."""
    out = {}
    for task, T in PK["tasks"].items():
        if "prm" not in T:
            continue
        d0, xs, ys = np.array(T["J_default"]), T["x_sep"], T["y_sep"]
        for k, v in T["prm"].items():
            v = np.array(v)
            out[(task, k)] = {"med": float(np.median(v)), "q1": float(np.quantile(v, 0.25)),
                              "q3": float(np.quantile(v, 0.75)), "rx": spearman(v, xs), "ry": spearman(v, ys),
                              "rj": spearman(v, d0), "rg": spearman(v, np.array(T["J_pick"]) - d0)}
    return out


# ------------------------------------------------------------------------------------------------ HOM's grids
def grid_stability_from_rows():
    """Per task, HOM's grid (grid2dff_v4b_sv25r, every layout with each grasp of its set at seeds 0-2): each layout's
    best grasp by the mean of the seeds; Spearman's rho between the layouts' J at seed 0 and at seed 1 and the mean
    |J0 - J1|; and the landscape's smoothness, the mean |dJ| between lattice neighbours 5 mm apart over the mean |dJ|
    between 20,000 random pairs of layouts."""
    import hom_landscape as L
    import hom_rows
    import inner_loop_benchmark_page as BP
    out = {}
    for task in TASKS:
        per = collections.defaultdict(dict)
        for line in (BP.GRID / f"{task}_d{D_MM[task]:g}.jsonl").read_text().splitlines():
            r = hom_rows.rescale_metrics(json.loads(line))
            if r.get("status", "ok") != "ok":
                continue
            dd = r["d"]
            per[(round(dd[2] - dd[0], 3), round(dd[3] - dd[5], 3), int(r.get("grasp", 0)))][r["seed"]] = \
                L.track_score(task, r)
        lay = {}
        for (x, y, _g), sd in per.items():
            m = float(np.mean(list(sd.values())))
            if (x, y) not in lay or m > lay[(x, y)][0]:
                lay[(x, y)] = (m, sd)
        ks = sorted(lay)
        J = np.array([lay[k][0] for k in ks])
        s0 = np.array([lay[k][1].get(0, np.nan) for k in ks])
        s1 = np.array([lay[k][1].get(1, np.nan) for k in ks])
        ok = np.isfinite(s0) & np.isfinite(s1)
        idx = {k: i for i, k in enumerate(ks)}
        nb = [abs(J[i] - J[idx[(x + dx, y + dy)]]) for (x, y), i in idx.items() for dx, dy in ((5, 0), (0, 5))
              if (x + dx, y + dy) in idx]
        rng = np.random.default_rng(0)
        a, b = rng.integers(0, len(J), 20000), rng.integers(0, len(J), 20000)
        out[task] = {"layouts": len(ks), "rho_seeds": spearman(s0[ok], s1[ok]),
                     "mean_abs_seed": float(np.mean(np.abs(s0[ok] - s1[ok]))), "neighbour": float(np.mean(nb)),
                     "random": float(np.mean(np.abs(J[a] - J[b]))), "pairs_neighbour": len(nb), "sd": float(J.std())}
    return out


def grid_stability(refresh=False):
    path = DATA / "20261114-hom_grid_stability.json"
    if refresh or not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(grid_stability_from_rows(), indent=1))
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--picks", action="store_true", help="reduce the 11-11 rows to data/20261111-hom_picks_track.json again")
    a = ap.parse_args()
    PK = picks(a.picks)
    GS = grid_stability(a.picks)
    for t, v in GS.items():
        print(t, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig_trends(PK, MEDIA / "20261114-hom_picks_trends.png")
    for (task, k), v in trend_numbers(PK).items():
        print(f"{task:8s} {k:16s} median {v['med']:.2f} ({v['q1']:.2f}-{v['q3']:.2f}) rho x {v['rx']:+.2f} y {v['ry']:+.2f} "
              f"J0 {v['rj']:+.2f} gain {v['rg']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
