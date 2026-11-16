#!/usr/bin/env python3
"""Builds docs/experiments/20261112-inner_loop_benchmark/20261112-inner_loop_benchmark.html: published inner-loop
evaluators against HOM on the SR2 layout lattice, each judged by the J of the hand it would pick and by its wall time per
layout on this workstation (owner, 2026-11-12: "evaluate some of the other inner loop controls we tried? Karma, uhas?";
"show these other methods ... are both worse and slower than what we have cooked up with gpu parallelized HOM. We can let
the other methods have the sphere packed model and warp"; 17:40: UHAS "is less usable for many reasons, to be codified
and quantified"; 21:1x: report smoothness beside J for every controller).

Data: the HOM grids under J (rows/grid2dff_v4b_sv25r of the foundational-pose folder, through hom_search_qd.lookup) and
their timing rows; KaRMA's rows (scripts/hom/karma_sr2.py: rows/karma_sph40.jsonl, karma_sph25.jsonl); UHAS on the
SCS0009 plant (scripts/hom/uhas_sr2.py): its evaluation rows rows/uhas_<task>_<run>_test.jsonl, 5 MB each and kept out
of git, summarized per layout and grasp into rows/uhas_twist_test_summary.json (refreshed whenever the rows are present),
the training progress copied to rows/uhas_progress_<run>.jsonl, the usability scorecard
rows/usability_<task>_<run>.jsonl (scripts/hom/inner_loop_usability.py), the action-space fit
rows/uhas_express_<task>_<n>.json and the films of scripts/hom/uhas_sr2_film.py. Rows ending in _work10 were run on the
work plant (4 N m/rad, 1 N m, 10 MPa pads) before 2026-11-13 10:30 and are not read.

    .venv/bin/python scripts/hom/inner_loop_benchmark_page.py
"""
from __future__ import annotations

import glob
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
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = ROOT / "docs/experiments/20261112-inner_loop_benchmark"
MEDIA, ROWS = D / "media", D / "rows"
OUT = D / "20261112-inner_loop_benchmark.html"
TEX_CACHE = D / "texsvg_cache.json"
GRID = P.ROWS / "grid2dff_v4b_sv25r"
SEARCH = "../20261029-foundational_pose_review/20261109-hom_search_methods.html"
TRACK = "../20261029-foundational_pose_review/20261109-hom_tracking_score.html"
MPC = "20261113-sampling_mpc_vs_hom.html"
fill, table, esc = P.fill, P.table, P.esc
TASKS = (("twist", 40.0, "sph40"), ("roll", 40.0, "sph40"), ("pinpush", 25.0, "sph25"), ("press", 25.0, "sph25"))
TNAME = {"twist": "twist", "roll": "roll", "pinpush": "pin-push", "press": "press"}
OBJ = {"twist": "40 mm sphere", "roll": "40 mm sphere", "pinpush": "25 mm cylinder", "press": "25 mm cylinder"}
SV_PLANT = "kp0.72_kv0_fr0.18_fl0_dp0.0144"          # scripts/hom/inner_loop_plant.py: the SCS0009 servo, 2.5 MPa pads
SUMMARY = ROWS / "uhas_twist_test_summary.json"
PERIOD_S = 0.01
REFS = [
    '[1] M. F. Peticco and P. Agrawal, "A Kinematic Metric for Fine Manipulation Ability in Robotic Hands" (KaRMA), '
    'IEEE/RSJ IROS 2026, arXiv 2605.15548; code github.com/mfpeticco/karma-hand-metric, commit c8d3698.',
    '[2] L. F. Casas, R. Teal, K. Shah, A. Tadepalli, W. Jin and Y. Xiang, "Cross-Embodiment Robot Manipulation via a '
    'Unified Hand Action Space" (UHAS), arXiv 2607.03570, 2026; code irvlutd.github.io/UHAS.',
    '[3] C. Schwarke, M. Mittal, N. Rudin, D. Hoeller and M. Hutter, "RSL-RL: A Learning Library for Robotics '
    'Research", arXiv 2509.10771, 2025.',
]


# ------------------------------------------------------------------------------------------------ data
def j_lookup(task, size):
    """{(x_sep, y_sep): (J of the best grasp, rollouts)} of the HOM grid (hom_search_qd.lookup)."""
    import hom_search_qd as Q
    return Q.lookup(task, size)[0]


def j_rows(task, size):
    """The HOM grid's J per (layout, grasp): mean over seeds 0-2."""
    import hom_landscape as L
    import hom_rows
    out = {}
    for x in (GRID / f"{task}_d{size:g}.jsonl").read_text().splitlines():
        r = hom_rows.rescale_metrics(json.loads(x))
        if r.get("status", "ok") != "ok":
            continue
        d = r["d"]
        k = (round(d[2] - d[0], 3), round(d[3] - d[5], 3), int(r.get("grasp", 0)))
        out.setdefault(k, []).append((L.track_score(task, r), bool(r.get("held"))))
    return out


def karma(variant):
    p = ROWS / f"karma_{variant}.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return {(r["x_sep"], r["y_sep"]): r for r in rows if "error" not in r}


def hom_timing():
    """GPU seconds per layout of each HOM grid (build, run and score of every batch over the layouts run)."""
    out = {}
    for task, size, _v in TASKS:
        T = [json.loads(x) for x in (GRID / f"{task}_d{size:g}_timing.jsonl").read_text().splitlines()]
        calls = {json.loads(x)["call"] for x in (GRID / f"{task}_d{size:g}.jsonl").read_text().splitlines()}
        tot = sum(t["build_s"] + t["run_s"] + t["score_s"] for t in T)
        out[task] = {"s_per_layout": tot / len(calls), "layouts": len(calls), "rollouts": sum(t["worlds"] for t in T),
                     "minutes": tot / 60.0}
    return out


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def karma_stats():
    """Per task: KaRMA-T and KaRMA-R against J over the layouts KaRMA ran, and the J of each score's best layout."""
    out = {}
    for task, size, var in TASKS:
        K = karma(var)
        if len(K) < 10:
            continue
        look = j_lookup(task, size)
        ks = [k for k in K if k in look and look[k][1] > 0]
        J = np.array([look[k][0] for k in ks])
        best = max((k for k in look if look[k][1] > 0), key=lambda k: look[k][0])
        rec = {"layouts": len(ks), "grid_best": float(look[best][0]), "grid_best_layout": list(best),
               "j_best_layout": list(ks[int(np.argmax(J))]), "j_best": float(J.max())}
        for name in ("karma_t", "karma_r"):
            v = np.array([K[k][name] for k in ks])
            rho, p = spearman(v, J)
            top = ks[int(np.argmax(v))]
            n10 = max(1, int(round(0.1 * len(ks))))
            topk = set(np.argsort(-v)[:n10])
            topj = set(np.argsort(-J)[:n10])
            rec[name] = {"rho": rho, "p": p, "best_layout": list(top), "best_J": float(look[top][0]),
                         "tenth_shared": len(topk & topj) / n10, "J_of_top_tenth": float(J[list(topk)].mean())}
        rec["secs_median"] = float(np.median([K[k]["secs"] for k in ks]))
        out[task] = rec
    return out


def eval_seconds(run):
    """(rollouts, seconds) of a run's evaluation on its split, from the line its log ends with."""
    m = None
    for log in sorted(glob.glob(str(ROOT / "logs/2026111*-uhas*/*.log")) + glob.glob(str(ROOT / "logs/20261112-mpc/sv_uhas_*.log"))):
        for line in Path(log).read_text(errors="replace").splitlines():
            if f"uhas_twist_{run}_test.jsonl" in line and "rollouts in" in line:
                m = re.search(r"(\d+) rollouts in (\d+) s", line)
    return [int(m.group(1)), int(m.group(2))] if m else None


def summarize():
    """rows/uhas_twist_test_summary.json from the evaluation rows present (the rows are 5 MB each and stay out of git):
    per run its split, plant and checkpoint, and per (x_sep, y_sep, grasp) the mean J over the seeds, the held share and
    the rollouts. Runs whose rows are absent keep their earlier entry."""
    S = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    for p in sorted(ROWS.glob("uhas_twist_*_test.jsonl")):
        run = p.stem[len("uhas_twist_"):-len("_test")]
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        if not rows or rows[0].get("plant") != SV_PLANT:
            continue
        sp_path = ROOT / "results/uhas_sr2" / run / "split.json"
        sp = json.loads(sp_path.read_text()) if sp_path.exists() else S.get(run, {}).get("split", {})
        per = {}
        for r in rows:
            d = r["d"]
            per.setdefault((round(d[2] - d[0], 3), round(d[3] - d[5], 3), int(r.get("grasp", 0))), []).append(
                (float(r["J"]), bool(r.get("held"))))
        S[run] = {"task": rows[0]["task"], "d_mm": float(rows[0]["d_mm"]), "plant": rows[0]["plant"],
                  "scenes": rows[0].get("scenes"), "checkpoint": rows[0].get("checkpoint"), "rollouts": len(rows),
                  "J_mean_all": round(float(np.mean([r["J"] for r in rows])), 4),
                  "held_all": round(float(np.mean([bool(r.get("held")) for r in rows])), 4),
                  "eval": eval_seconds(run) or S.get(run, {}).get("eval"),
                  "split": {k: (len(v) if k in ("train", "test") and len(v) > 2 else v) for k, v in sp.items()},
                  "per": [[x, y, g, round(float(np.mean([j for j, _h in v])), 4),
                           round(float(np.mean([h for _j, h in v])), 4), len(v)] for (x, y, g), v in sorted(per.items())]}
    if S:
        SUMMARY.write_text(json.dumps(S, indent=0, separators=(",", ":")))
    return S


def variant_of(E):
    sp = E.get("split", {})
    if sp.get("start") == "open":
        return "open"
    if isinstance(sp.get("train"), list) and len(sp["train"]) == 1:
        return "single"
    return "limits" if sp.get("limits") else "published"


VARIANTS = {"published": ("UHAS, published reward", "#0f766e"), "limits": ("UHAS with the limit terms", "#7c3aed"),
            "open": ("UHAS from the open hand", "#b45309"), "single": ("UHAS trained on one layout", "#be123c")}


def uhas_runs(S):
    """{variant: (run, summary entry)}, the newest run of each variant on the SCS0009 plant."""
    out = {}
    for run in sorted(S):
        E = S[run]
        if E.get("plant") == SV_PLANT and E.get("task") == "twist":
            out[variant_of(E)] = (run, E)
    return out


def uhas_stats(E):
    """UHAS against HOM on the evaluated layouts: per layout the best grasp's mean J over seeds 0-2 under each."""
    task, size = E["task"], E["d_mm"]
    hom = j_rows(task, size)
    lay, n_u, h_u, h_h = {}, 0, 0.0, []
    for x, y, g, j, held, n in E["per"]:
        hj = hom.get((x, y, g), [])
        L_ = lay.setdefault((x, y), {"uhas": [], "hom": []})
        L_["uhas"].append(j)
        L_["hom"].append(float(np.mean([a for a, _h in hj])) if hj else float("nan"))
        n_u += n
        h_u += held * n
        h_h += [h for _a, h in hj]
    best = {k: (max(v["uhas"]), np.nanmax(v["hom"])) for k, v in lay.items()}
    ks = sorted(best)
    U = np.array([best[k][0] for k in ks])
    H = np.array([best[k][1] for k in ks])
    rho = spearman(U, H) if np.std(U) > 0 else (float("nan"), float("nan"))
    iu, ih = int(np.argmax(U)), int(np.argmax(H))
    return {"task": task, "layouts": len(ks), "rollouts": n_u, "uhas_mean": float(U.mean()), "hom_mean": float(H.mean()),
            "uhas_best": float(U.max()), "hom_best": float(H.max()), "uhas_pick": list(ks[iu]),
            "hom_of_uhas_pick": float(H[iu]), "hom_pick": list(ks[ih]), "uhas_of_hom_pick": float(U[ih]),
            "held_uhas": h_u / max(1, n_u), "held_hom": float(np.mean(h_h)), "rho": rho[0], "U": U.tolist(),
            "H": H.tolist(), "keys": [list(k) for k in ks]}


SCORE = ("grip_mean_N", "link_mean_N", "pad_share", "sat_frac", "cmd_rate_dps", "reversals_ps", "drift_max_mm")


def usability(run):
    """The scorecard of HOM and of the run's policy on the same worlds (inner_loop_usability.py): per controller the
    rollouts, the held share, the mean J over every world and the medians over the held rollouts."""
    p = ROWS / f"usability_twist_{run}.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    out = {}
    for c in ("hom", "uhas"):
        R = [r for r in rows if r["controller"] == c]
        H = [r for r in R if r["held"]]
        if not R:
            continue
        out[c] = {"n": len(R), "held": len(H) / len(R), "J_all": float(np.mean([r["J"] for r in R]))}
        out[c].update({k: float(np.median([r[k] for r in H if r.get(k) is not None])) for k in SCORE})
    return out


def express():
    """How near UHAS's sphere actions come to HOM's handover grasps (uhas_sr2.py express) on the SCS0009 plant."""
    fits = {}
    for n in ("8x128", "16x256", "32x512"):
        q = ROWS / f"uhas_express_twist_{n}.json"
        if q.exists():
            fits[n] = json.loads(q.read_text())
    return fits


def uhas_progress(run):
    p = ROWS / f"uhas_progress_{run}.jsonl"
    src = ROOT / "results/uhas_sr2" / run / "progress.jsonl"
    if src.exists():                                   # results/ is outside git: keep a copy beside the page
        p.write_text(src.read_text())
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def film_info(tag, runs):
    """rows/film_hom_uhas_twist<tag>.json when its film exists and was made from a run of this page."""
    info, path = ROWS / f"film_hom_uhas_twist{tag}.json", MEDIA / f"20261112-film_hom_uhas_twist{tag}.mp4"
    if not (info.exists() and path.exists()):
        return None
    fi = json.loads(info.read_text())
    return (fi, path) if fi["run"] in {r for r, _E in runs.values()} else None


# ------------------------------------------------------------------------------------------------ figures
def plt_setup(size=15):
    """Times at size pt; a figure W inches wide shown 860 px wide has its text at size * 11.9 / W px."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": size,
                         "mathtext.fontset": "stix"})
    return plt


def fig_karma(St):
    """Per task: J and KaRMA-T over the layouts KaRMA ran (the 10 mm sub-lattice and J's ten best), with the best
    layout of each."""
    plt = plt_setup(14)
    tasks = [t for t, _s, _v in TASKS if t in St]
    fig, axs = plt.subplots(2, len(tasks), figsize=(3.1 * len(tasks), 6.3), squeeze=False, layout="constrained")
    for j, task in enumerate(tasks):
        size, var = [(s, v) for t, s, v in TASKS if t == task][0]
        K, look = karma(var), j_lookup(task, size)
        ks = [k for k in K if k in look and look[k][1] > 0]
        X = np.array([k[0] for k in ks])
        Y = np.array([k[1] for k in ks])
        for i, (vals, lab, cm) in enumerate(((np.array([look[k][0] for k in ks]), "J under HOM", "viridis"),
                                             (np.array([K[k]["karma_t"] for k in ks]), "KaRMA-T", "magma"))):
            ax = axs[i, j]
            sc = ax.scatter(X, Y, c=vals, cmap=cm, s=30, marker="s", edgecolors="none")
            b = int(np.argmax(vals))
            ax.plot(X[b], Y[b], "*", ms=16, mfc="white", mec="k", mew=1.0)
            ax.set_xlim(35, 165)
            ax.set_ylim(50, 175)
            ax.set_aspect("equal")
            ax.set_xlabel("$x_{sep}$ (mm)")
            if j == 0:
                ax.set_ylabel("$y_{sep}$ (mm)")
            ax.set_title(f"{TNAME[task].capitalize()}: {lab}", fontsize=14)
            fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    path = MEDIA / "20261112-karma_maps.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return path


def fig_uhas(runs):
    """UHAS's J proxy against training time, and UHAS's J against HOM's on the test layouts, for each variant."""
    plt = plt_setup()
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.2))
    lim = 0.05
    for v, (S, prog) in runs.items():
        lab, col = VARIANTS[v]
        if prog:
            axs[0].plot([r["minutes"] for r in prog], [r.get("J_proxy") or 0.0 for r in prog], "-", color=col, lw=2,
                        label=lab)
        if v != "single":
            axs[1].scatter(S["H"], S["U"], s=22, color=col, label=lab, alpha=0.85)
            lim = max(lim, max(S["H"] + S["U"]) * 1.05)
    axs[0].set_xlabel("Training time on the GPU (min)")
    axs[0].set_ylabel("J proxy of finished episodes")
    axs[0].set_title("Training on the twist", fontsize=15)
    axs[0].legend(loc="upper left", fontsize=14)
    axs[0].grid(color="#eeeeee")
    ax = axs[1]
    ax.plot([0, lim], [0, lim], ":", color="#999999")
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.set_xlabel("J under HOM")
    ax.set_ylabel("J under UHAS")
    ax.set_title("Test layouts of the twist, best grasp of each", fontsize=15)
    ax.legend(loc="lower right", fontsize=14)
    ax.grid(color="#eeeeee")
    fig.tight_layout()
    path = MEDIA / "20261112-uhas_vs_hom.png"
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------ sections
def sec_terms(runs):
    terms = [
        ("Layout", "the SR2 hand's symmetric tripod in two coordinates: $x_{sep}$, from the thumb's mount to the pair's, "
         "and $y_{sep}$, from the index's mount to the middle's, in steps of 5 mm. The HOM grids cover the layouts at "
         "which the pose solver found grasps: 527 for the twist and the roll, 437 for the pin-push and 369 for the "
         "press."),
        ("Task", "the twist and the roll of a 40 mm sphere, and the pin-push and the press of a 25 mm cylinder. Each "
         "commands a rotation of the object (<a href=\"" + TRACK + "\">tracking-score page</a>)."),
        ("J, the tracking score", "one minus the ratio of two RMS angles over the task: the angle between the object's "
         "orientation and the commanded one, and the commanded rotation. It is at least 0 and is multiplied by the share "
         "of the task with all three pads above 0.05 N. It is 0 when the object drops 15 mm "
         "(<code>hom_landscape.track_score</code>). A layout's J is its best grasp's mean over seeds 0-2."),
        ("Plant", "the model of the hand's servos. The SCS0009 plant, used on this page, is a position servo of "
         "0.72 N&#8201;m/rad with a torque limit of 0.18 N&#8201;m per joint, with 2.5 MPa pads "
         "(<code>scripts/hom/inner_loop_plant.py</code>). The work plant, 4 N&#8201;m/rad and 1 N&#8201;m, is 5.6 times "
         "as strong."),
        ("Inner loop", "what turns one layout into a score in co-design: a controller with its simulation (HOM, UHAS), or "
         "a metric of the layout alone (KaRMA). The layouts are chosen by the outer loop (<a href=\"" + SEARCH + "\">search "
         "methods page</a>)."),
        ("HOM", "the hand's model-based in-hand manipulation controller of the talk, run on every world of a MuJoCo-Warp "
         "batch; no training."),
        ("KaRMA", "a kinematic metric of a pinch on a sphere between the thumb and the index [1]. KaRMA-T is the volume "
         "of sphere-centre positions reached by rolling without letting go, normalized by the hand's size. KaRMA-R is "
         "the share of 228 orientation bins reached."),
        ("UHAS", "a learned controller for many hands [2]: one policy acting through a canonical sphere whose deformation is "
         "mapped to each hand's joints by cascade inverse kinematics, trained by PPO."),
        ("Handover", "the state at 2.0 s, when HOM has closed the fingers on the object, squeezed it and lifted it, and "
         "the commanded rotation starts. UHAS acts from the handover."),
        ("Held", "the object did not drop 15 mm below its start before the task ended."),
        ("Grip", "the total force of the fingers on the object, summed over the fingers and averaged over the 10 ms "
         "control periods of the task phase (N)."),
        ("Rigid-part force", "the part of the grip through the fingers' rigid parts instead of the pads (N)."),
        ("Pad share", "the pads' part of the fingers' force on the object, summed over the task phase."),
        ("Saturation", "the share of joint-periods with a servo's torque at 95 % of its limit or more."),
        ("RMS target rate", "the root mean square rate of change of the servo targets over the joints and periods "
         "(deg/s). At 8,000 deg/s a target moves 80 deg in one period."),
        ("Target reversals", "sign changes of a servo target's rate of change, counted where the rate exceeds 1 deg/s, "
         "per joint per second."),
        ("Drift", "the largest distance of the object's centre from its position at the start of the task phase (mm)."),
        ("Spearman's &#961;", "the rank correlation of two scores over the same layouts, from -1 to 1."),
        ("First choice", "the layout an evaluator ranks first. Its J under HOM is the J a designer would get by choosing "
         "with that evaluator."),
        ("Wall time per layout", "on this workstation: one CPU core of 20, and one RTX 4070 Ti SUPER of 16 GB."),
    ]
    terms.insert(8, ("Single-hand UHAS", "the same policy and training on one layout alone, the paper's single-hand "
                     "baseline; on the lattice it is a policy trained per layout."))
    return '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"


def sec_methods():
    return r"""<p>Each evaluator was judged by the J of the HOM grids on the layouts it scored. The HOM grids are those of
the talk: every layout with up to 16 grasps from the pose solver at seeds 0-2, 48 rollouts a layout.</p>
<p>KaRMA ran as published [1], its code unmodified, in its own conda environment
(<code>scripts/hom/karma_sr2.py</code>). The SR2 hand at each layout was exported as a URDF: each finger's three joints
with their axes and ranges, and a fixed fingertip link at the pad's distal end. Its test sphere was set to the tasks'
objects, 40 mm for the twist and the roll and 25 mm for the pin-push and the press. Its finger radius was set to 15.1 mm,
the pad's largest distance from the distal link's axis, and its scaling to the hand's size was switched off. KaRMA ran on
a 10 mm sub-lattice and on the ten best layouts by J of each task.</p>
<p>UHAS was ported to the HOM grids' MuJoCo-Warp scenes, with the same sphere-pad fingertips, plant, objects, grasps and
commanded rotations (<code>scripts/hom/uhas_sr2_cik.py</code>, <code>scripts/hom/uhas_sr2.py</code>). The sphere, its
deformation by two driving vectors per finger and the cascade inverse kinematics follow the paper and its code. The
torch kinematics of the SR2 finger agree with MuJoCo's to 0.0001 mm. HOM drives the approach, the closing, the squeeze
and the lift, the first 2.0 s, as in the grids. From the handover the policy drives the fingers at HOM's 10 ms period.
Its observations are UHAS's, without noise, plus the commanded angular velocity, because the goal moves. Its reward is
UHAS's, with the commanded rotation as the goal and J's three-pad factor added. PPO ran with UHAS's settings in RSL-RL
[3]. After a drop, a world restarts from a pool of HOM handovers in which the object was held, as UHAS's environment
restarts a fallen cube. The policy was trained on 418 layouts and their grasps and evaluated on the other 109, which
include the five best by J.</p>
<p>Two policies were trained on the SCS0009 plant at seed 0. One used UHAS's reward as published. The other added three limit terms to it, on the servos' torque demand, on the joints'
ranges and on the change of the action:
$$r_{\mathrm{lim}} = -0.02 \sum_j \max\!\left(0,\ \frac{d_j - 0.8}{0.2}\right) - 0.1 \sum_j \mathrm{clip}\!\left(\frac{3^\circ - m_j}{3^\circ},\ 0,\ 1\right) - 0.04\,\lVert a - a_{\mathrm{prev}} \rVert^2 .$$
Here $d_j = k_p\,|q^{*}_j - q_j| / \tau_{\max}$ is joint $j$'s torque demand as a share of its limit, $m_j$ is its
distance from the nearer end of its range, and $a$ is the action. The reward with $r_{\mathrm{lim}}$ added is clipped at
0 from below. HOM and each policy then drove the same worlds of the test split for the usability scorecard
(<code>scripts/hom/inner_loop_usability.py</code>).</p>
<p>The UHAS runs of 11-12 and of the morning of 11-13 were made on the work plant by mistake. Their rows end in
<code>_work10</code> and are reported in the work-plant section of the <a href=\"""" + MPC + r"""\">sampling MPC
page</a>.</p>"""


def sec_karma(St, HT):
    if not St:
        return ""
    rows = []
    for task, _s, _v in TASKS:
        if task not in St:
            continue
        r = St[task]
        rows.append([TNAME[task].capitalize(), r["layouts"],
                     f"({r['grid_best_layout'][0]:g}, {r['grid_best_layout'][1]:g})", f"{r['grid_best']:.2f}",
                     f"({r['karma_t']['best_layout'][0]:g}, {r['karma_t']['best_layout'][1]:g})",
                     f"{r['karma_t']['best_J']:.2f}", f"{r['karma_t']['rho']:.2f}",
                     f"{100 * r['karma_t']['tenth_shared']:.0f}", f"{r['karma_r']['best_J']:.2f}",
                     f"{r['karma_r']['rho']:.2f}"])
    tw = St["twist"]
    rest = [St[t] for t in ("roll", "pinpush", "press") if t in St]
    t = fill(r"""<p>KaRMA's first choice scored below the grid's best layout on every task (Table <<tab>>, Figure
<<fig>>). On the twist the ranking by KaRMA-T correlated with J at Spearman's &#961; = <<rt>> over <<n>> layouts. KaRMA-T's
first choice there, (<<bx>>, <<by>>) mm, scored J <<bj>>. The grid's best, (<<gx>>, <<gy>>) mm, scored <<gb>>. On the
other three tasks &#961; was <<r0>> to <<r1>>, and KaRMA-T's first choice scored <<j0>> to <<j1>>. KaRMA-R, the share of
orientations reached, correlated negatively with J on every task, at &#961; = <<q0>> to <<q1>>. Its first choice scored at
most <<qj>>.</p>""",
             tab=P.TAB[0] + 1, fig=P.FIG[0] + 1, rt=f"{tw['karma_t']['rho']:.2f}", n=tw["layouts"],
             bx=f"{tw['karma_t']['best_layout'][0]:g}", by=f"{tw['karma_t']['best_layout'][1]:g}",
             bj=f"{tw['karma_t']['best_J']:.2f}", gx=f"{tw['grid_best_layout'][0]:g}",
             gy=f"{tw['grid_best_layout'][1]:g}", gb=f"{tw['grid_best']:.2f}",
             r0=f"{min(r['karma_t']['rho'] for r in rest):.2f}", r1=f"{max(r['karma_t']['rho'] for r in rest):.2f}",
             j0=f"{min(r['karma_t']['best_J'] for r in rest):.2f}", j1=f"{max(r['karma_t']['best_J'] for r in rest):.2f}",
             q0=f"{min(r['karma_r']['rho'] for r in St.values()):.2f}",
             q1=f"{max(r['karma_r']['rho'] for r in St.values()):.2f}",
             qj=f"{max(r['karma_r']['best_J'] for r in St.values()):.2f}")
    t += table(["Task", "Layouts", "Grid's best (mm)", "Its J", "KaRMA-T's first choice (mm)", "Its J",
                "&#961;, KaRMA-T", "Top tenth shared (%)", "J of KaRMA-R's first choice", "&#961;, KaRMA-R"], rows,
               "KaRMA against J over the layouts KaRMA ran: a 10 mm sub-lattice and J's ten best of each task. &#961;: "
               "Spearman's rank correlation with J. Top tenth shared: the share of KaRMA-T's highest tenth that is also "
               "in J's highest tenth.", num_from=1)
    t += P.img(fig_karma(St), "J of each layout under HOM (top) and KaRMA-T (bottom), on the layouts KaRMA ran. The "
               "star marks each score's best layout. <code>scripts/hom/karma_sr2.py</code>.")
    secs = [St[k]["secs_median"] for k in St]
    t += fill(r"""<p>A KaRMA run lasted a median of <<s0>> to <<s1>> s per layout on one CPU core, with eight runs at a
time beside a training run on the GPU. HOM scored a layout in <<h0>> to <<h1>> s of the GPU: 48 rollouts with the
batch's build and scoring. Its grasps come from the pose solver beforehand, at 2.3 s per layout on one core for the
sphere and 4.5 s for the cylinder.</p>""",
              s0=f"{min(secs):.0f}", s1=f"{max(secs):.0f}",
              h0=f"{min(v['s_per_layout'] for v in HT.values()):.1f}",
              h1=f"{max(v['s_per_layout'] for v in HT.values()):.1f}")
    return t


def sec_express(fits):
    if "32x512" not in fits:
        return ""
    f = fits["32x512"]
    e, j = f["contact_err_mm"], f["joint_err_deg_median"]
    rows = [[n.replace("x", " &#215; "), f"{v['contact_err_mm']['median']:.1f}", f"{v['contact_err_mm']['p90']:.1f}",
             f"{100 * v['share_within_2mm']:.0f}", f"{100 * v['share_within_5mm']:.0f}"] for n, v in fits.items()]
    t = fill(r"""<p>UHAS's actions reach only part of the grasps that HOM hands over. For <<h>> handovers of the twist on
training layouts in which the object was held, the sphere action nearest each grasp was found by the cross-entropy
method on the pads' contact points. At the largest search the contact points lay a median <<m>> mm from HOM's, with a
90th percentile of <<p>> mm. Within 2 mm came <<s2>> % of the grasps, and within 5 mm <<s5>> % (Table <<tab>>). The pip
joint was off by a median <<pip>> deg and the mcp by <<mcp>> deg. In UHAS's rule each link is fixed against the
deformed sphere, so the pip follows the mcp.</p>""", h=f["held_at_handover"], m=f"{e['median']:.1f}",
             p=f"{e['p90']:.1f}", s2=f"{100 * f['share_within_2mm']:.0f}", s5=f"{100 * f['share_within_5mm']:.0f}",
             tab=P.TAB[0] + 1, pip=f"{j['pip']:.0f}", mcp=f"{j['mcp']:.0f}")
    t += table(["Search (iterations &#215; candidates)", "Median (mm)", "90th percentile (mm)", "Within 2 mm (%)",
                "Within 5 mm (%)"], rows,
               f"Distance from HOM's handover contact points to those of the nearest sphere action, over "
               f"{f['held_at_handover']} held handovers of the twist on the SCS0009 plant "
               "(<code>uhas_sr2.py express</code>).", num_from=1)
    return t


def sec_uhas(runs):
    if "published" not in runs:
        return ""
    st, rows = {}, []
    for v in ("published", "limits", "open"):
        if v not in runs:
            continue
        run, E = runs[v]
        S = uhas_stats(E)
        prog = uhas_progress(run)
        st[v] = (S, prog)
        last = prog[-1] if prog else {}
        rows.append([VARIANTS[v][0], f"{last.get('minutes', 0):.0f}", f"{100 * S['held_uhas']:.0f}",
                     f"{S['uhas_mean']:.3f}", f"{S['uhas_best']:.2f}",
                     f"({S['uhas_pick'][0]:g}, {S['uhas_pick'][1]:g})", f"{S['hom_of_uhas_pick']:.2f}",
                     f"{S['rho']:.2f}" if S["rho"] == S["rho"] else "&#8212;"])
    S0 = st["published"][0]
    rows.insert(0, ["HOM", "none", f"{100 * S0['held_hom']:.0f}", f"{S0['hom_mean']:.3f}", f"{S0['hom_best']:.2f}",
                    f"({S0['hom_pick'][0]:g}, {S0['hom_pick'][1]:g})", f"{S0['hom_best']:.2f}", "1"])
    S, prog = st["published"]
    last = prog[-1] if prog else {}
    t = fill(r"""<p>UHAS with the published reward reached a mean J of <<mu>> per test layout, twice HOM's <<mh>>
(Table <<tab>>, Figure <<fig>>). It held the sphere to the end in <<hu>> % of <<nr>> rollouts; HOM held it in <<hh>> %.
Its training took <<mins>> min on the GPU: <<its>> PPO iterations of 2,048 worlds and 16 steps. These <<tr>> M
transitions are <<pct>> % of the paper's 15,000 iterations of 8,192 environments.</p>
<p>Over the <<nl>> test layouts UHAS's J and HOM's correlated at Spearman's &#961; = <<rho>>. UHAS's first choice,
(<<ux>>, <<uy>>) mm, scored <<hp>> under HOM. HOM's best test layout, (<<hx>>, <<hy>>) mm, scored <<bh>> under HOM and
<<ub>> under UHAS.</p>""",
             mu=f"{S['uhas_mean']:.3f}", mh=f"{S['hom_mean']:.3f}", tab=P.TAB[0] + 1, fig=P.FIG[0] + 1,
             hu=f"{100 * S['held_uhas']:.0f}", nr=f"{S['rollouts']:,}", hh=f"{100 * S['held_hom']:.0f}",
             mins=f"{last.get('minutes', 0):.0f}", its=f"{last.get('iteration', 0) + 1:,}",
             tr=f"{last.get('transitions', 0) / 1e6:.0f}",
             pct=f"{100 * last.get('transitions', 0) / (15000 * 8192 * 16):.1f}", nl=S["layouts"], rho=f"{S['rho']:.2f}",
             ux=f"{S['uhas_pick'][0]:g}", uy=f"{S['uhas_pick'][1]:g}", hp=f"{S['hom_of_uhas_pick']:.2f}",
             hx=f"{S['hom_pick'][0]:g}", hy=f"{S['hom_pick'][1]:g}", bh=f"{S['hom_best']:.2f}",
             ub=f"{S['uhas_of_hom_pick']:.2f}")
    if "limits" in st:
        SL = st["limits"][0]
        t += fill(r"""<p>With the limit terms UHAS reached <<ml>> per test layout and held the sphere in <<hl>> % of the
rollouts.</p>""", ml=f"{SL['uhas_mean']:.3f}", hl=f"{100 * SL['held_uhas']:.0f}")
    if "open" in st:
        SO = st["open"][0]
        t += fill(r"""<p>Acting from the open hand, with its own grasp and lift, UHAS held the sphere in <<ho>> % of
<<no>> test rollouts, and its mean J per test layout was <<mo>>.</p>""", ho=f"{100 * SO['held_uhas']:.0f}",
                  no=f"{SO['rollouts']:,}", mo=f"{SO['uhas_mean']:.3f}")
    t += table(["Controller", "Training (min)", "Held (%)", "Mean J per layout", "Best layout's J", "First choice (mm)",
                "Its J under HOM", "&#961; with HOM's J"], rows,
               "The 109 test layouts of the twist on the SCS0009 plant, every grasp at seeds 0-2. Mean J per layout: "
               "each layout at its best grasp. Held: share of rollouts with the sphere held to the end.", num_from=1)
    t += P.img(fig_uhas(st), "Left: the J proxy of the episodes finished between checkpoints (1 - RMS error / RMS "
               "commanded rotation, times the three-pad share, 0 if dropped), against training time. Right: each test "
               "layout's J under UHAS against its J under HOM, each at its best grasp, mean of seeds 0-2.")
    return t


def sec_usability(runs):
    U = {v: usability(runs[v][0]) for v in ("published", "limits", "open") if v in runs}
    U = {v: u for v, u in U.items() if u.get("uhas")}
    if "published" not in U:
        return ""
    st = {v: uhas_stats(runs[v][1]) for v in U}
    h, u = U["published"]["hom"], U["published"]["uhas"]
    rows = []
    for lab, s, jm in [("HOM", h, st["published"]["hom_mean"])] + [(VARIANTS[v][0], U[v]["uhas"], st[v]["uhas_mean"])
                                                                   for v in U]:
        rows.append([lab, f"{jm:.3f}", f"{100 * s['held']:.0f}", f"{s['grip_mean_N']:.1f}", f"{s['link_mean_N']:.1f}",
                     f"{s['pad_share']:.2f}", f"{100 * s['sat_frac']:.0f}", f"{s['cmd_rate_dps']:,.0f}",
                     f"{s['reversals_ps']:.1f}", f"{s['drift_max_mm']:.0f}"])
    t = fill(r"""<p>UHAS's servo targets jumped about <<step>> deg in each 10 ms period, where HOM's moved about
<<steph>> deg (Table <<tab>>). The scorecard was taken on the <<n>> worlds of the test split at seed 0: every grasp of
every test layout. HOM and each policy drove the task phase of each world. UHAS's targets moved at <<ru>> deg/s RMS and
reversed <<vu>> times per joint per second. HOM's moved at <<rh>> deg/s and reversed <<vh>> times. UHAS's servos were at
their torque limit in <<su>> % of the joint-periods, HOM's in <<sh>> %. The sphere drifted up to <<du>> mm from its start
under UHAS and <<dh>> mm under HOM.""", tab=P.TAB[0] + 1, n=f"{h['n']:,}", ru=f"{u['cmd_rate_dps']:,.0f}",
             rh=f"{h['cmd_rate_dps']:,.0f}", step=f"{u['cmd_rate_dps'] * PERIOD_S:.0f}",
             steph=f"{h['cmd_rate_dps'] * PERIOD_S:.1f}", vu=f"{u['reversals_ps']:.1f}", vh=f"{h['reversals_ps']:.1f}",
             su=f"{100 * u['sat_frac']:.0f}", sh=f"{100 * h['sat_frac']:.0f}", du=f"{u['drift_max_mm']:.0f}",
             dh=f"{h['drift_max_mm']:.0f}")
    if "limits" in U:
        ul = U["limits"]["uhas"]
        t += fill(r""" With the limit terms the saturation fell to <<sl>> % and the targets' rate to <<rl>> deg/s. The
reversals rose to <<vl>> per joint per second, and the drift to <<dl>> mm.""", sl=f"{100 * ul['sat_frac']:.0f}",
                  rl=f"{ul['cmd_rate_dps']:,.0f}", vl=f"{ul['reversals_ps']:.1f}", dl=f"{ul['drift_max_mm']:.0f}")
    t += "</p>"
    t += fill(r"""<p>UHAS pressed with its pads more than HOM did. Of its finger force <<pu>> was on the pads, against
<<ph>> of HOM's. Its rigid parts pressed with <<lu>> N, HOM's with <<lh>> N. Its grip was <<gu>> N, HOM's <<gh>> N.</p>""",
              pu=f"{u['pad_share']:.2f}", ph=f"{h['pad_share']:.2f}", lu=f"{u['link_mean_N']:.1f}",
              lh=f"{h['link_mean_N']:.1f}", gu=f"{u['grip_mean_N']:.1f}", gh=f"{h['grip_mean_N']:.1f}")
    t += table(["Controller", "Mean J per layout", "Held (%)", "Grip (N)", "Rigid-part force (N)", "Pad share",
                "Saturation (%)", "RMS target rate (deg/s)", "Target reversals (1/s)", "Drift (mm)"], rows,
               f"The usability scorecard on the {h['n']:,} worlds of the test split at seed 0, SCS0009 plant: medians "
               "over the held rollouts (<code>rows/usability_twist_&lt;run&gt;.jsonl</code>). Mean J per layout as in the "
               f"previous table; held: share of the {h['n']:,} rollouts.", num_from=1)
    films = []
    for tag, what in (("", "HOM's best test layout"), ("_uhaspick", "UHAS's first choice"),
                      ("_limits", "HOM's best test layout"), ("_open", "HOM's best test layout")):
        f = film_info(tag, runs)
        if not f:
            continue
        fi, path = f
        v = next(k for k, (r, _E) in runs.items() if r == fi["run"])
        films.append(P.film(path, fill(
            "The twist at <<what>>, (<<x>>, <<y>>) mm, grasp <<g>>, seed <<sd>>, on the SCS0009 plant. Left: HOM, J <<jh>>, "
            "rigid-part force <<lh>> N. Right: <<lab>> (checkpoint <<ck>>), J <<ju>>, rigid-part force <<lu>> N. Each "
            "frame gives the pad and rigid-part force of each finger. <code>scripts/hom/uhas_sr2_film.py</code>.",
            what=what, x=f"{fi['layout'][0]:g}", y=f"{fi['layout'][1]:g}", g=fi["grasp"], sd=fi["seed"],
            jh=f"{fi['hom']['J']:.2f}", lh=f"{fi['hom']['link_force_mean_N']:.1f}", lab=VARIANTS[v][0],
            ck=fi["checkpoint"].replace(".pt", ""), ju=f"{fi['uhas']['J']:.2f}",
            lu=f"{fi['uhas']['link_force_mean_N']:.1f}")))
    return t + "".join(films)


def sec_single(runs):
    if "single" not in runs:
        return ""
    run, E = runs["single"]
    S = uhas_stats(E)
    x, y = S["keys"][0]
    return fill(r"""<p>Trained on the layout (<<x>>, <<y>>) mm alone for 30 min, UHAS reached J <<u>> there at its best
grasp and held the sphere in <<hu>> % of the 48 rollouts. HOM's J there was <<h>>, with the sphere held in <<hh>> %.
Trained this way on every layout, the twist's 527 layouts would take <<days>> days of the GPU.</p>""",
                x=f"{x:g}", y=f"{y:g}", u=f"{S['uhas_best']:.2f}", hu=f"{100 * S['held_uhas']:.0f}",
                h=f"{S['hom_best']:.2f}", hh=f"{100 * S['held_hom']:.0f}", days=f"{527 * 30 / 60 / 24:.0f}")


def sec_cost(St, HT, runs):
    """Wall time per layout and for one task's lattice, by evaluator."""
    tw = HT["twist"]
    rows = [["HOM", "none", "2.3 s, one core", f"{tw['s_per_layout']:.1f} s", f"{tw['minutes']:.0f} min"]]
    if "twist" in St:
        km = St["twist"]["secs_median"]
        rows.append(["KaRMA [1]", "none", "its own seed search", f"{km:.0f} s, one core",
                     f"{527 * km / 8 / 3600:.1f} h on 8 cores"])
    if "published" in runs:
        run, E = runs["published"]
        prog = uhas_progress(run)
        mins = prog[-1]["minutes"] if prog else float("nan")
        ev = E.get("eval")
        per = ev[1] / (ev[0] / 48.0) if ev else float("nan")
        rows.append(["UHAS [2]", f"{mins:.0f} min, once", "HOM's, 2.3 s, one core", f"{per:.1f} s",
                     f"{mins:.0f} min + {527 * per / 60:.0f} min"])
    rows.append(["Single-hand UHAS", "30 min per layout", "HOM's, 2.3 s, one core", "30 min",
                 f"{527 * 30 / 60 / 24:.0f} days"])
    return table(["Evaluator", "Training", "Grasps", "Score of one layout", "The twist's 527 layouts"], rows,
                 "Wall time on this workstation (one RTX 4070 Ti SUPER, 20 CPU cores) for the twist on the 40 mm sphere. "
                 "HOM: the grid's batches (build, run and scoring) over its layouts, 48 rollouts each. UHAS: its training, "
                 "and its evaluation's batches over the test layouts, 48 rollouts each. Single-hand UHAS: the paper's "
                 "baseline at 30 min of training per layout.", num_from=1)


def sec_limits(runs):
    items = ["UHAS trained once per setting, on the twist only, at seed 0, within a fixed budget of GPU time. Its paper "
             "trained 15,000 iterations of 8,192 environments, about 2 billion transitions, on another task and "
             "simulator.",
             "UHAS's reward was adapted to a moving goal: the paper's bonus per goal reached has no counterpart, and J's "
             "three-pad factor was added. Neither UHAS's observations nor HOM's measurements had noise.",
             "The scorecard is from simulation. The plant has no speed limit, so a target that moves 80 deg in a period "
             "is followed as far as the torque limit allows.",
             "KaRMA scores a pinch on a sphere between two fingers. The pin-push and the press move a cylinder, and the "
             "twist and the roll use all three fingers.",
             "KaRMA ran on 145 and 128 layouts: the 10 mm sub-lattice and J's ten best per task. The grids have 527, 437 "
             "and 369 layouts."]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def sec_future(runs):
    items = []
    if "open" not in runs:
        items.append("UHAS from the open hand on the SCS0009 plant (<code>uhas_sr2.py train --task twist --start open "
                     "--minutes 120</code>). On the work plant it held the sphere in none of 5,232 test rollouts. A held "
                     "share above 0 on the SCS0009 plant would show that UHAS grasps without HOM.")
    items += ["UHAS on the roll, the pin-push and the press, and a second training seed on the twist "
              "(<code>uhas_sr2.py train --task roll</code>). HOM's place above UHAS in usability fails on a task where "
              "UHAS's targets reverse fewer than HOM's 2.6 times per joint per second at a higher J.",
              "UHAS trained and evaluated as above with its servo targets' rate limited to the SCS0009's speed. The "
              "limit goes in the plant or in the sphere controller, an option of <code>uhas_sr2.py</code> still to be "
              "written. UHAS's lead in J fails if its mean test J falls to HOM's 0.25 or below under the limit.",
              "KaRMA on every layout of the lattice (<code>karma_sr2.py batch --variant sph40</code>, about 11 h on 8 "
              "cores), and its published scaled variant (<code>--variant scaled</code>). The ranking of KaRMA below "
              "HOM fails if KaRMA's first choice comes within 0.05 of the grid's best J on any task."]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def sec_files():
    return r"""<ul>
<li>KaRMA: <code>scripts/hom/karma_sr2.py</code>; rows <code>rows/karma_sph40.jsonl</code>,
<code>rows/karma_sph25.jsonl</code>; queue <code>logs/20261112-karma_sr2/queue.sh</code>; the code in
<code>external/karma</code> (commit c8d3698, environment <code>karma-hand-metric</code>).</li>
<li>UHAS: <code>scripts/hom/uhas_sr2_cik.py</code> (sphere controller), <code>scripts/hom/uhas_sr2.py</code> (environment,
training, evaluation), <code>scripts/hom/uhas_sr2_film.py</code>, <code>scripts/hom/inner_loop_usability.py</code>;
rows <code>rows/uhas_twist_test_summary.json</code> (the evaluation rows per layout and grasp; the rows themselves,
<code>rows/uhas_twist_&lt;run&gt;_test.jsonl</code>, stay on the workstation), <code>rows/uhas_progress_*.jsonl</code>,
<code>rows/usability_twist_*.jsonl</code>, <code>rows/uhas_express_twist_*.json</code>; runs
<code>results/uhas_sr2/</code> on the workstation; lane steps <code>logs/20261112-mpc/uhas_train_eval.sh</code>.</li>
<li>HOM: the grids <code>docs/experiments/20261029-foundational_pose_review/rows/grid2dff_v4b_sv25r/</code>.</li>
<li>This page: <code>scripts/hom/inner_loop_benchmark_page.py</code>.</li>
</ul>"""


def lede(St, HT, runs):
    """The lede, and the page's summary for the docs index: KaRMA's first choice, UHAS's J and its scorecard."""
    parts, summ = [], []
    hs = [v["s_per_layout"] for v in HT.values()]
    if St:
        secs = [St[k]["secs_median"] for k in St]
        gap = [St[k]["grid_best"] - St[k]["karma_t"]["best_J"] for k in St]
        kw = dict(h0=f"{min(hs):.1f}", h1=f"{max(hs):.1f}", k0=f"{min(secs):.0f}", k1=f"{max(secs):.0f}",
                  g0=f"{min(gap):.2f}", g1=f"{max(gap):.2f}")
        parts.append(fill("""HOM scored a layout of the SR2 lattice in <<h0>> to <<h1>> s of the GPU, 48 rollouts each.
A run of KaRMA, a kinematic metric, lasted a median <<k0>> to <<k1>> s per layout on one CPU core. Its first choice scored <<g0>> to
<<g1>> below each task's best J.""", **kw))
        summ.append(fill("KaRMA's first choice scored <<g0>> to <<g1>> below each task's best J under HOM, at <<k0>> to "
                         "<<k1>> s per layout against HOM's <<h0>> to <<h1>> s.", **kw))
    if "published" in runs:
        S = uhas_stats(runs["published"][1])
        prog = uhas_progress(runs["published"][0])
        kw = dict(m=f"{prog[-1]['minutes']:.0f}", u=f"{S['uhas_mean']:.2f}", h=f"{S['hom_mean']:.2f}",
                  hu=f"{100 * S['held_uhas']:.0f}", hh=f"{100 * S['held_hom']:.0f}")
        parts.append(fill("""UHAS, trained for <<m>> min on the twist, reached a mean J of <<u>> per test layout against
HOM's <<h>>. It held the sphere in <<hu>> % of the rollouts, HOM in <<hh>> %.""", **kw))
        summ.append(fill("UHAS, trained for <<m>> min, reached a mean J of <<u>> per test layout of the twist against "
                         "HOM's <<h>>.", **kw))
        U = usability(runs["published"][0])
        if U.get("uhas"):
            u, h = U["uhas"], U["hom"]
            kw = dict(vu=f"{u['reversals_ps']:.1f}", vh=f"{h['reversals_ps']:.1f}", su=f"{100 * u['sat_frac']:.0f}",
                      sh=f"{100 * h['sat_frac']:.0f}")
            parts.append(fill("""Its servo targets reversed <<vu>> times per joint per second against HOM's <<vh>>, and
its servos were at their torque limit in <<su>> % of the joint-periods against <<sh>> %.""", **kw))
            summ.append(fill("Its servo targets reversed <<vu>> times per joint per second against HOM's <<vh>>.", **kw))
    if "open" in runs:
        SO = uhas_stats(runs["open"][1])
        parts.append(fill("""From the open hand UHAS held the sphere in <<ho>> of <<no>> rollouts.""",
                          ho=f"{round(SO['held_uhas'] * SO['rollouts']):,}", no=f"{SO['rollouts']:,}"))
    text = " ".join(parts) or "Results pending."
    return text, " ".join(summ) or text


def render_tex(page):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    return P.TEX_RE.sub(lambda m: next(svgs), page)


def main():
    S = summarize()
    runs = uhas_runs(S)
    St, HT = karma_stats(), hom_timing()
    sections = [("terms", "Terms and metrics", sec_terms(runs)),
                ("methods", "Evaluators and their inputs", sec_methods()),
                ("karma", "KaRMA on the layout lattice", sec_karma(St, HT)),
                ("express", "UHAS's actions and HOM's grasps", sec_express(express())),
                ("uhas", "UHAS on the twist", sec_uhas(runs)),
                ("usability", "Usability of UHAS and HOM", sec_usability(runs)),
                ("single", "UHAS trained on one layout", sec_single(runs)),
                ("cost", "Wall time per layout", sec_cost(St, HT, runs)),
                ("limits", "Limitations", sec_limits(runs)),
                ("future", "Future work", sec_future(runs)),
                ("refs", "References and files", '<ol class="refs">' + "".join(
                    f"<li>{r[r.index(']') + 2:]}</li>" for r in REFS) + "</ol>" + sec_files())]
    sections = [x for x in sections if x[2]]
    lede_t, summary = lede(St, HT, runs)
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|N\b|s\b|min\b|h\b)", "\\1&nbsp;\\2", s_))
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{unit(b)}</section>'
                     for k, (sid, hd, b) in enumerate(sections, 1))
    byline = ("KaRMA and UHAS on the SR2 lattice, run 2026-11-12 to 11-15, against the HOM grids of the hand-design talk "
              "(SCS0009 plant, 2.5 MPa pads, MuJoCo-Warp 3.6; <code>rows/grid2dff_v4b_sv25r</code>). The outer-loop "
              f'methods are on the <a href="{SEARCH}">search methods page</a>, Section 12, and the sampling planners on '
              f'the <a href="{MPC}">sampling MPC page</a>. Builder: <code>scripts/hom/inner_loop_benchmark_page.py</code>.')
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": "SR2 inner-loop evaluators", "H1": "HOM, KaRMA and UHAS as inner-loop evaluators of SR2 hand "
                 "layouts", "EYEBROW": "Co-design benchmark, 2026-11-12", "LEDE": unit(lede_t), "BYLINE": byline,
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS + texsvg.BLOCK_CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    page = render_tex(page)
    page = docs_index.tag(page, topic="design", summary=summary, date="2026-11-12")
    page = retro_style.apply(page)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
