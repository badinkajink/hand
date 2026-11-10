#!/usr/bin/env python3
"""Builds docs/experiments/20261029-foundational_pose_review/20261110-hom_servo_stiffness_mujoco4.html: the SR2 hand's HOM
landscapes at the default objects on the SCS0009 stiffness measured on 2026-11-06 (0.55 N m/rad, then with its 0.59 deg
dead band) against the model's 0.72 N m/rad, and on MuJoCo 4.0 against 3.6 (post-reset steps 1 and 2; owner, 11-09: "we
previously re-did sysid from 0.72 nmrad to 0.55nmrad, but i dont want to rerun everything, so we can stick to 0.72 for
now"; "part of me also wants to just send it on new mujoco").

Data: the grids of GRIDS (hom_gpu.py run, logs/20261110-hom_post_reset/gpu_queue_step*.sh), ranked by
hom_landscape.track_score with rollout_score beside it; the cached maps rows/20261110-stiffness_landscapes.json.

    .venv/bin/python scripts/hom/hom_stiffness_page.py maps    # the cache and the map figures
    .venv/bin/python scripts/hom/hom_stiffness_page.py         # the page
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SCALE_HAND", "cad_v2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import docs_index  # noqa: E402
import fp_review_figures as FR  # noqa: E402
import fp_review_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

table, img, film, esc = P.table, P.img, P.film, P.esc
D = ROOT / "docs/experiments/20261029-foundational_pose_review"
MEDIA, ROWS = D / "media", D / "rows"
OUT = D / "20261110-hom_servo_stiffness_mujoco4.html"
LAND = ROWS / "20261110-stiffness_landscapes.json"
PLANTPAGE = "20261102-hom_plant_modulus.html"
TRACK = "20261109-hom_tracking_score.html"
TRENDS = "20261030-hom_layout_trends.html"
SYSID = "../20261106-sr2_servo_sysid/20261106-sr2_servo_identification.html"
MJ4 = "../20261108-mujoco4_port/20261108-mujoco4_contact_and_stack.html"
TASKS = ("twist", "roll", "press")
DEFAULT = {"twist": 40, "roll": 40, "press": 25}
NAME = {"twist": "Twist", "roll": "Roll", "press": "Press"}
OBJ = {"twist": "40 mm sphere", "roll": "40 mm sphere", "press": "25 mm cylinder"}
# key: (label, rows directory, MuJoCo version, run date); every grid on the 2.5 MPa scenes of 11-02
GRIDS = {
    "sv72": ("0.72 N m/rad, MuJoCo 3.6, 11-02", "grid2dff_v4b_sv25", "3.6.0", "2026-11-02"),
    "sv72r": ("0.72 N m/rad, MuJoCo 3.6, 11-10", "grid2dff_v4b_sv25r", "3.6.0", "2026-11-10"),
    "sv55": ("0.55 N m/rad, MuJoCo 3.6", "grid2dff_v4b_sv55", "3.6.0", "2026-11-10"),
    "sv55db": ("0.55 N m/rad and dead band, MuJoCo 3.6", "grid2dff_v4b_sv55db", "3.6.0", "2026-11-10"),
}
SCORES = ("track", "rollout")


def load(name):
    p = ROWS / name
    return json.loads(p.read_text()) if p.exists() else None


def cmd_maps():
    """Each grid's best-of-16 per layout by both scores and its best grasp, cached; then the map figures."""
    import hom_plant_transfer as TR
    out = {}
    for k, (_label, src, _v, _d) in GRIDS.items():
        out[k] = {}
        for task in TASKS:
            p = ROWS / src / f"{task}_d{DEFAULT[task]}.jsonl"
            out[k][task] = {}
            for sc in SCORES:
                g = TR.grasp_means(task, p, sc)
                if g:
                    out[k][task][sc] = {"best": {str(c): round(max(v.values()), 4) for c, v in g.items()},
                                        "grasp": {str(c): max(v, key=v.get) for c, v in g.items()}}
        print(k, {t: {s: len(v["best"]) for s, v in out[k][t].items()} for t in TASKS}, flush=True)
    LAND.write_text(json.dumps(out))
    print("wrote", LAND.relative_to(ROOT))
    keys = [k for k in GRIDS if k != "sv72" and all("track" in out[k][t] for t in TASKS)]
    fig_maps(out, keys, "20261110-stiffness_maps.png")
    return 0


def fig_maps(Lc, keys, name):
    """Best-of-16 tracking-score maps over (x_sep, y_sep): a row per grid, a column per task; grey where no grasp was
    solved, the best layout starred."""
    plt = FR.plt_setup()
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15, "xtick.labelsize": 14,
                         "ytick.labelsize": 14})
    fig, axs = plt.subplots(len(keys), 3, figsize=(8.6, 2.6 * len(keys) + 0.7), constrained_layout=True, sharex=True,
                            sharey=True, squeeze=False)
    ext = [37.5, 162.5, 47.5, 172.5]
    im = None
    for i, k in enumerate(keys):
        for j, task in enumerate(TASKS):
            ax = axs[i, j]
            best = {int(c): v for c, v in Lc[k][task]["track"]["best"].items()}
            Z = np.full((25, 25), np.nan)
            for c, v in best.items():
                Z[c % 25, c // 25] = v
            ax.imshow(np.where(np.isfinite(Z), np.nan, 0.0), origin="lower", extent=ext, cmap="Greys", vmin=-1, vmax=1,
                      aspect="equal")
            im = ax.imshow(Z, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=0.8, aspect="equal")
            c = max(best, key=best.get)
            ax.plot(40 + 5 * (c // 25), 50 + 5 * (c % 25), "r*", ms=14, mec="k")
            ax.set_xticks([50, 100, 150])
            ax.set_yticks([60, 100, 140])
            if i == 0:
                ax.set_title(f"{NAME[task]}\n{OBJ[task]}")
            if i == len(keys) - 1:
                ax.set_xlabel("$x_{sep}$ (mm)")
            if j == 0:
                ax.set_ylabel(GRIDS[k][0].replace(", MuJoCo", ",\nMuJoCo") + "\n$y_{sep}$ (mm)", fontsize=12)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label="Best-of-16 tracking score")
    FR.save(fig, name)


def xy(c):
    return (40 + 5 * (int(c) // 25), 50 + 5 * (int(c) % 25))


def pair(Lc, a, b, task, score):
    """Spearman's rho of two grids' best-of-16 over the layouts both reached, the share of a's best tenth in b's, and
    each grid's best layout with its score on both (hom_plant_transfer.compare's measures on the cached maps)."""
    from sr2hand.numerics.stats import spearman
    A, B = Lc[a][task].get(score, {}).get("best"), Lc[b][task].get(score, {}).get("best")
    if not A or not B:
        return None
    calls = sorted(set(A) & set(B), key=int)
    x, y = np.array([A[c] for c in calls]), np.array([B[c] for c in calls])
    k = max(1, len(calls) // 10)
    ta = {calls[i] for i in np.argsort(-x)[:k]}
    tb = {calls[i] for i in np.argsort(-y)[:k]}
    ba, bb = calls[int(np.argmax(x))], calls[int(np.argmax(y))]
    return {"n": len(calls), "rho": float(spearman(x, y)), "top": len(ta & tb) / k,
            "best_a": (xy(ba), A[ba], B[ba]), "best_b": (xy(bb), B[bb], A[bb]),
            "median_a": float(np.median(x)), "median_b": float(np.median(y))}


def scores_rho(Lc, k, task):
    """Spearman's rho between a grid's best-of-16 by the tracking score and by the earlier score."""
    from sr2hand.numerics.stats import spearman
    T_, R_ = Lc[k][task].get("track", {}).get("best"), Lc[k][task].get("rollout", {}).get("best")
    if not T_ or not R_:
        return None
    calls = sorted(set(T_) & set(R_), key=int)
    return float(spearman(np.array([T_[c] for c in calls]), np.array([R_[c] for c in calls])))


def cmd_numbers():
    """The comparisons printed from the cache, for writing the page."""
    Lc = load(LAND.name)
    for a, b in (("sv72", "sv72r"), ("sv72", "sv55"), ("sv72r", "sv55"), ("sv55", "sv55db"), ("sv72r", "sv55db")):
        if a not in Lc or b not in Lc:
            continue
        for task in TASKS:
            for sc in SCORES:
                r = pair(Lc, a, b, task, sc)
                if r:
                    print(f"{a:6s} {b:6s} {task:6s} {sc:7s} n {r['n']:3d} rho {r['rho']:.3f} top {r['top']:.2f} best_a "
                          f"{r['best_a'][0]} {r['best_a'][1]:.2f}->{r['best_a'][2]:.2f} best_b {r['best_b'][0]} "
                          f"{r['best_b'][1]:.2f} (on a {r['best_b'][2]:.2f}) med {r['median_a']:.2f}/{r['median_b']:.2f}")
    for k in Lc:
        print(k, {t: (None if scores_rho(Lc, k, t) is None else round(scores_rho(Lc, k, t), 3)) for t in TASKS})
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["maps"]:
        sys.exit(cmd_maps())
    if sys.argv[1:2] == ["numbers"]:
        sys.exit(cmd_numbers())
    raise SystemExit("the page is not written yet: run maps, numbers")
