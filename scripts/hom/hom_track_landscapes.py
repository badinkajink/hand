#!/usr/bin/env python3
"""The four default landscapes of the hand-design talk under the tracking score (owner, 2026-11-10 night: "sure, let's
rerun the landscapes now"). The grids of rows grid2dff_v4b_sv25 (625 tripods, best of 16 grasps, seeds 0-2, SCS0009 at
0.72 N m/rad, 2.5 MPa pads, MuJoCo 3.6.0) predate the tracking fields; their reruns in rows grid2dff_v4b_sv25r carry them
(twist, roll and press from the post-reset queue's step 1 on 11-10, logs/20261110-hom_post_reset/gpu_queue_step1.sh; the
pin-push from gpu_pinpush_sv25r.sh the same night). Per task: the Spearman correlation between the earlier score and the
tracking score on the same rows (the test of the tracking-score page's future work: above 0.9, the earlier landscape
stands), the rerun's own spread on the earlier score, and the best layouts by each score with the two factors of the
tracking score. Writes rows/20261110-track_landscapes_sv25r.json and media/20261110-track_landscapes_sv25r.png (the
tracking-score page, Section 7).

    .venv/bin/python scripts/hom/hom_track_landscapes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_landscape as L  # noqa: E402
import hom_rows  # noqa: E402

ROWS = L.ROWS
MEDIA = ROWS.parent / "media"
STORED, RERUN = "grid2dff_v4b_sv25", "grid2dff_v4b_sv25r"
TASKS = [("twist", 40.0, "Twist, 40 mm sphere"), ("roll", 40.0, "Roll, 40 mm sphere"),
         ("pinpush", 25.0, "Pin-push, 25 mm cylinder"), ("press", 25.0, "Press, 25 mm cylinder")]


def factors(task, rs):
    """Mean over seeds of the tracking score's two factors: the rotation followed (0 if dropped) and the three-pad share."""
    f = [max(0.0, 1.0 - r["track_err_rms_deg"] / r["track_ref_rms_deg"]) if r.get("held") and r.get("track_err_rms_deg")
         is not None else 0.0 for r in rs]
    return float(np.mean(f)), float(np.mean([r.get("contact_all_frac", 0.0) for r in rs]))


def slip(rs):
    """Median over seeds of the largest slide of the pusher's pad on the cylinder (mm); None on the spheres."""
    v = [r["pusher_slip_max_mm"] for r in rs if r.get("pusher_slip_max_mm") is not None]
    return round(float(np.median(v)), 2) if v else None


def grid(task, size, rows_dir):
    """Per reached layout (x_sep, y_sep): the rows of each grasp."""
    L.GRID_SRC = "grid2dff"
    poses = L.grid_poses(task, size)
    by = {}
    for x in (ROWS / rows_dir / f"{task}_d{size:g}.jsonl").read_text().splitlines():
        if x.strip():
            r = hom_rows.rescale_metrics(json.loads(x))
            if r.get("status", "ok") == "ok":
                by.setdefault((int(r["call"]), int(r.get("grasp", 0))), []).append(r)
    out = {}
    for call, (xs, ys, rec) in poses.items():
        if rec and rec.get("ok"):
            g = {j: rs for (c, j), rs in by.items() if c == call}
            if g:
                out[(float(xs), float(ys))] = g
    return out


def best(task, g, fn):
    """The best grasp of one layout by a score: (score, grasp, its rows)."""
    j = max(g, key=lambda k: np.mean([fn(task, r) for r in g[k]]))
    return float(np.mean([fn(task, r) for r in g[j]])), j, g[j]


def tenth(m):
    return set(sorted(m, key=m.get, reverse=True)[:max(1, len(m) // 10)])


def rho(a, b):
    k = sorted(set(a) & set(b))
    return round(float(spearmanr([a[x] for x in k], [b[x] for x in k])[0]), 3), len(k)


def main():
    out, maps = {"tasks": {}}, {}
    for task, size, _ in TASKS:
        old_s = {xy: best(task, g, L.rollout_score)[0] for xy, g in grid(task, size, STORED).items()}
        G = grid(task, size, RERUN)
        old_r = {xy: best(task, g, L.rollout_score)[0] for xy, g in G.items()}
        J = {xy: best(task, g, L.track_score)[0] for xy, g in G.items()}
        maps[task] = (J, old_s)
        bo = max(old_s, key=old_s.get)
        bj = max(J, key=J.get)
        _, _, rs_o = best(task, G[bo], L.rollout_score)
        _, _, rs_j = best(task, G[bj], L.track_score)
        fo, co = factors(task, rs_o)
        fj, cj = factors(task, rs_j)
        out["tasks"][task] = {
            "size_mm": size, "layouts": rho(old_r, J)[1],
            "rho_earlier_vs_track": rho(old_r, J)[0],          # the same rerun rows, two scores
            "rho_rerun_earlier": rho(old_s, old_r)[0],         # the stored grid against its rerun, earlier score
            "rho_stored_vs_track": rho(old_s, J)[0],
            "tenth_shared": round(len(tenth(old_s) & tenth(J)) / len(tenth(J)), 2),
            "best_earlier": {"layout": list(bo), "earlier": round(old_s[bo], 3), "track": round(J.get(bo, np.nan), 3),
                             "followed": round(fo, 3), "three_pads": round(co, 3), "slip_mm": slip(rs_o)},
            "best_track": {"layout": list(bj), "track": round(J[bj], 3), "earlier": round(old_s.get(bj, np.nan), 3),
                           "followed": round(fj, 3), "three_pads": round(cj, 3), "slip_mm": slip(rs_j)},
            "track_median": round(float(np.median(list(J.values()))), 3)}
        print(task, out["tasks"][task], flush=True)
    Jp, Jr = maps["pinpush"][0], maps["press"][0]
    Op, Or = maps["pinpush"][1], maps["press"][1]
    shared = sorted(set(Jp) & set(Jr))
    out["pinpush_vs_press"] = {
        "layouts": len(shared), "rho_track": rho(Jp, Jr)[0], "rho_earlier": rho(Op, Or)[0],
        "tenth_shared_track": round(len(tenth({x: Jp[x] for x in shared}) & tenth({x: Jr[x] for x in shared}))
                                    / max(1, len(shared) // 10), 2),
        "press_best_on_pinpush": round(Jp.get(tuple(out["tasks"]["press"]["best_track"]["layout"]), np.nan), 3),
        "pinpush_best_on_press": round(Jr.get(tuple(out["tasks"]["pinpush"]["best_track"]["layout"]), np.nan), 3)}
    print("pin-push vs press", out["pinpush_vs_press"], flush=True)
    (ROWS / "20261110-track_landscapes_sv25r.json").write_text(json.dumps(out, indent=1))
    figure(maps, out)
    return 0


def figure(maps, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 9.4), sharex=True, sharey=True, constrained_layout=True)
    for ax, (task, size, title) in zip(axes.flat, TASKS):
        J = maps[task][0]
        xs = np.arange(40.0, 160.0 + 1e-6, 5.0)
        ys = np.arange(50.0, 170.0 + 1e-6, 5.0)
        Z = np.full((len(ys), len(xs)), np.nan)
        for (x, y), v in J.items():
            Z[int(round((y - ys[0]) / 5)), int(round((x - xs[0]) / 5))] = v
        im = ax.imshow(np.ma.masked_invalid(Z), origin="lower", extent=[xs[0] - 2.5, xs[-1] + 2.5, ys[0] - 2.5,
                       ys[-1] + 2.5], cmap="YlGnBu", vmin=0.0, vmax=0.8, aspect="equal")
        ax.set_facecolor("#e5e7eb")
        bo, bj = out["tasks"][task]["best_earlier"]["layout"], out["tasks"][task]["best_track"]["layout"]
        ax.scatter([bo[0]], [bo[1]], s=170, facecolors="white", edgecolors="black", linewidths=1.6, zorder=5)
        ax.scatter([bj[0]], [bj[1]], marker="*", s=330, c="#facc15", edgecolors="black", linewidths=1.2, zorder=6)
        ax.set_title(title)
        ax.set_xticks([50, 100, 150])
        ax.set_yticks([50, 100, 150])
    for ax in axes[1]:
        ax.set_xlabel(r"thumb to pair $s_1$ (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"index to middle $s_2$ (mm)")
    cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    cb.set_label("tracking score")
    h1 = axes[0, 0].scatter([], [], s=170, facecolors="white", edgecolors="black", linewidths=1.6,
                         label="best by the earlier score")
    h2 = axes[0, 0].scatter([], [], marker="*", s=330, c="#facc15", edgecolors="black", linewidths=1.2,
                         label="best by the tracking score")
    fig.legend(handles=[h1, h2], loc="lower center", bbox_to_anchor=(0.45, -0.06), ncol=2, frameon=False)
    path = MEDIA / "20261110-track_landscapes_sv25r.png"
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path.relative_to(L.ROWS.parents[3]), flush=True)


if __name__ == "__main__":
    sys.exit(main())
