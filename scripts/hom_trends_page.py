#!/usr/bin/env python3
"""Builds docs/experiments/20261029-foundational_pose_review/20261030-hom_layout_trends.html: the SR2 layout study after
the finger-clearance fix, across object sizes and with a table under the object (2026-10-30, afternoon). It follows the
design page (20261029-foundational_pose_hand_design.html, built by scripts/fp_review_page.py, whose helpers and template it
uses) from its Section 9.

Data: rows/grid2dff_v4b (re-solved grids, toy_check.json, rerun.json, ground/), rows/ff_timing, rows/cpu_best
(top_*_ff*.jsonl, table_*.jsonl), rows/ea_ff (the offline EA on the re-solved grids), rows/ea_live and ea_live_gr (the
live EA, summary.json from scripts/hom_ea.py report), rows/grid2dgr_v4b (the grids clear of the table), rows/films_v4b.json.

    .venv/bin/python scripts/hom_trends_page.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import fp_review_figures as FF  # noqa: E402
import fp_review_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D, MEDIA, ROWS = P.D, P.MEDIA, P.ROWS
OUT = D / "20261030-hom_layout_trends.html"
DESIGN = "20261029-foundational_pose_hand_design.html"
fill, table, img, film, esc = P.fill, P.table, P.img, P.film, P.esc
NAMES = {"twist": "Twist", "roll": "Roll", "pinpush": "Pin-push", "press": "Press"}
OBJ = {"twist": "sphere", "roll": "sphere", "pinpush": "cylinder", "press": "cylinder"}
DEFAULT = {"twist": 40.0, "roll": 40.0, "pinpush": 25.0, "press": 25.0}
REFS = [P.REFS[0], P.REFS[1], P.REFS[5]]           # [1] foundational pose, [2] HOM, [3] Yang et al.


def load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


def v4b_film(name, caption_fn):
    """Film media/20261030-film_<name>.mp4 with its caption from rows/films_v4b.json, or nothing."""
    path = MEDIA / f"20261030-film_{name}.mp4"
    info = json.loads(P.FILMS_V4B.read_text()).get(name) if P.FILMS_V4B.exists() else None
    return film(path, caption_fn(info)) if (path.exists() and info) else ""


# ------------------------------------------------------------------------------------------------ numbers

def rerun_numbers():
    """The toy against the plant, the toy's solve times, the GPU's cost of finger-to-finger contact, the re-solved grids
    against the earlier ones, the CPU check of their best layouts and the offline EA on them."""
    G = ROWS / "grid2dff_v4b"
    out = {"check": json.loads((G / "toy_check.json").read_text()) if (G / "toy_check.json").exists() else {},
           "rerun": json.loads((G / "rerun.json").read_text()) if (G / "rerun.json").exists() else None}
    out["solve_s"] = {k: [float(np.median([r["ms"] for r in FF.load(n) if r["ok"]])) / 1e3 for n in names]
                      for k, names in (("tripod", ("grid2d_tripod40_v4b", "grid2dff_tripod40_v4b")),
                                       ("pinch", ("grid2d_pinch25_v4b", "grid2dff_pinch25_v4b")))}
    out["timing"] = {}
    for p_ in sorted((ROWS / "ff_timing").glob("t_*_timing.jsonl")):
        r = json.loads(p_.read_text().splitlines()[0])
        out["timing"][p_.name[2:-len("_timing.jsonl")]] = {"worlds": r["worlds"], "run_s": r["run_s"],
                                                           "ms": 1e3 * r["run_s"] / r["worlds"]}
    out["cpu"] = {}
    for task, d in DEFAULT.items():
        rows = load_jsonl(ROWS / "cpu_best" / f"top_{task}_d{d:g}_ff.jsonl")
        full = [r for r in rows if not r["gpu_filter"]]
        filt = [r for r in rows if r["gpu_filter"]]
        if full:
            out["cpu"][task] = {"n": len(full), "touch": sum(1 for r in full if r["finger_touch_periods"]),
                                "cpu_median": float(np.median([r["score"] for r in full])),
                                "gpu_median": float(np.median([r["gpu_best"] for r in full])),
                                "filter_median": float(np.median([r["score"] for r in filt])) if filt else None,
                                "low": sum(1 for r in full if r["score"] < 0.5 <= r["gpu_best"]),
                                "held": sum(1 for r in full if r["held"])}
    ep = ROWS / "ea_ff" / "summary.json"
    out["ea"] = json.loads(ep.read_text()) if ep.exists() else None
    return out


def trend_numbers(src="grid2dff"):
    """hom_ea.py report's summary (rows/ea_live[_gr]/summary.json) and the CPU check of each search's best layouts."""
    p_ = ROWS / ("ea_live" if src == "grid2dff" else "ea_live_gr") / "summary.json"
    if not p_.exists():
        return None
    S = json.loads(p_.read_text())
    out = {k: v for k, v in S.items() if isinstance(v, dict) and "task" in v}
    for k, v in out.items():
        rows = load_jsonl(ROWS / "cpu_best" / f"top_{v['task']}_d{v['d_mm']:g}_ff_ea.jsonl")
        full = [r for r in rows if not r["gpu_filter"]]
        if full:
            top = max(full, key=lambda r: r["gpu_best"])
            v["cpu"] = {"n": len(full), "touch": sum(1 for r in full if r["finger_touch_periods"]),
                        "low": sum(1 for r in full if r["score"] < 0.5 <= r["gpu_best"]),
                        "best_cpu": top["score"], "best_gpu": top["gpu_best"], "best_xy": [top["x_sep"], top["y_sep"]]}
    return out


def ground_numbers():
    """The table clearance of the re-solved grids' grasps (scripts/hom_ground.py): per pose family and size the share of
    grasps and of layouts within 1 mm of the table at the touch pose and at the touch and open poses."""
    G = ROWS / "grid2dff_v4b" / "ground"
    out = {}
    for p_ in sorted(G.glob("*.jsonl")):
        m = re.match(r"(tripod|pinch)([0-9.]+)\.jsonl", p_.name)
        if not m:
            continue
        rows = load_jsonl(p_)
        for pose in sorted({r["pose"] for r in rows}):
            rr = [r for r in rows if r["pose"] == pose]
            ok_t = [r["touch_mm"] >= -1.0 for r in rr]
            ok_b = [r["touch_mm"] >= -1.0 and r["open_mm"] >= -1.0 for r in rr]
            lay = {}
            for r, a in zip(rr, ok_b):
                lay[r["call"]] = lay.get(r["call"], False) or a
            out[(pose, float(m.group(2)))] = {
                "grasps": len(rr), "touch_ok": float(np.mean(ok_t)), "both_ok": float(np.mean(ok_b)),
                "layouts": len(lay), "layouts_ok": int(sum(lay.values())),
                "touch_median": float(np.median([r["touch_mm"] for r in rr]))}
    return out


def table_numbers():
    """The CPU plant with a physical table (scripts/hom_table_check.py) on the best layouts of both grids."""
    out = {}
    for task, d in DEFAULT.items():
        rows = load_jsonl(ROWS / "cpu_best" / f"table_{task}_d{d:g}.jsonl")
        if not rows:
            continue
        res = {}
        for src in ("grid2dff", "grid2dgr"):
            for tb in (False, True):
                rr = [r for r in rows if r["src"] == src and r["table"] == tb]
                if rr:
                    res[f"{src}_{'table' if tb else 'post'}"] = {
                        "n": len(rr), "held": sum(1 for r in rr if r["held"]),
                        "median": float(np.median([r["score"] for r in rr])),
                        "touch": sum(1 for r in rr if r["table_touch_periods"]),
                        "gpu_median": float(np.median([r["gpu_best"] for r in rr]))}
        out[task] = res
    return out


# ------------------------------------------------------------------------------------------------ sections

def sec_terms():
    terms = [
        ("Layout", "the six palm-frame mount coordinates (mm) of the SR2 hand's three gantries. This page uses the "
         "symmetric tripod only. The thumb's mount lies $x_{sep}$ from the pair's, and the index's $y_{sep}$ from the "
         "middle's. The lattice runs in steps of 5 mm from 40 to 160 mm and from 50 to 170 mm, 625 layouts."),
        ("HOM tasks", "the four tasks of the design page. The twist turns a sphere about the vertical, and the roll "
         "rolls it along the thumb's pad. In the pin-push and the press, the middle finger turns a cylinder about the "
         "pinch of thumb and index. Default objects: a 40 mm sphere and a 25 mm cylinder, 100 mm long."),
        ("Task score", "a rollout's score in [0, 1], 0 when the object is dropped; the design page's Terms give it per "
         "task. A grasp's score is its mean over seeds 0 to 2."),
        ("Grasp set, best of 16", "up to 16 grasps of one layout and object solved by the pose toy, the foundational "
         "pose first; a layout's best of 16 is its highest grasp score."),
        ("Touch pose, open pose", "the plant's finger joint angles with each pad's own surface on its contact point. "
         "The open pose lies 8 mm further out along each contact normal; the HOM pick starts there."),
        ("Finger clearance", "the smallest surface distance (mm) between the collision spheres of two fingers at the "
         "touch pose: the sphere packs of the links and the pad spheres, 1,465 per finger."),
        ("Table clearance", "the smallest height (mm) of the bottom of a finger's collision sphere above the plane "
         "through the object's lowest point. That plane is the table the object would rest on. A grasp clears the table "
         "when this height is at least -1 mm at the touch pose and at the open pose."),
        ("Collision filter", "on the GPU, the pads and the rigid spheres under 2 mm collide with the object only. Every "
         "contact is kept on the CPU plant."),
        ("EA", "the elitist, mutation-only evolutionary algorithm of Yang et al. [3]. In each generation the best half "
         "is kept and the best eighth are the parents. A child is a parent moved by up to 15 mm per coordinate, rounded "
         "to the 5 mm lattice. Population: the layouts per generation. Budget: the rollouts after which no generation "
         "starts."),
        ("Within 0.05", "a search succeeds when its best layout so far scores within 0.05 of the full grid's best."),
        ("Near set", "the layouts a search ran that score within 0.05 of the best one it found."),
        ("Reach", "the share of the 625 layouts on which the task's pose is found for the object."),
        ("Hold", "the cylinder tasks' plant variant. The object is held at its start pose until the grasp closes at "
         "0.8 s, so it cannot roll off its 3 mm post during the approach."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"


def sec_fix(N):
    R = N["rerun"]
    ck, tm, ss = R["check"], R["timing"], R["solve_s"]
    if not (ck and tm):
        return "<p>Not run yet.</p>"
    tw = ck["twist_d40"]
    return fill(r"""<p>In a third of the design page's sphere grasps, the index's and the middle's v4b caps overlapped at
the touch pose. The pose toy kept the fingers apart by a skeleton of capsules that omits the caps, and the GPU evaluator's
collision filter hid the overlap (design page, Section 9). The pose toy now keeps the fingers apart with the plant's own
collision spheres (<code>fp_sr2_toy.ff_clearance_v4b</code>). Each finger has 1,465 of them: the sphere packs of its links
and the 1,018 spheres of its pad. The toy places them by its closed-form kinematics, which reproduced MuJoCo's body poses
to within 1e-15 m at 50 random sets of joint angles. Each finger stands at the plant's touch pose. Every two fingers must
stay 2 mm apart, and that clearance is a constraint of the toy's program and of its grasp walk.</p>
<p>On <<n>> grasps of the earlier twist grid, the toy's clearance differed from the plant's by a median of <<md>> mm and
at most <<mx>> mm. Both put the same <<nb>> grasps under 2 mm, and <<po>> more fell under it on the plant alone. The
constraint slows the toy: a tripod layout's pose and grasp set took a median of <<t0>> s before and <<t1>> s after, a pinch
layout's <<p0>> and <<p1>> s.</p>
<p>Finger-to-finger contact in the GPU evaluator itself would cost 9 to 13 times the rollout time (rows
<code>ff_timing</code>, the first 42 layouts of the earlier twist grid). With the small rigid spheres colliding with every
finger part, as on the CPU, a batch of 2,016 twist worlds took <<r1>> s. With the filter it took <<r0>> s, <<m1>> against
<<m0>> ms per rollout. At 480 worlds the ratio was <<x4>>, and with the pads colliding as well it was <<xn>>. The filter
therefore stays, and the fingers are kept apart before a rollout starts. A coarser stand-in did not fit: 96 k-means spheres
per fingertip, each grown to contain its members, still reached 7.6 mm in radius on a tip 15 mm wide.</p>""",
                n=tw["grasps"], md=f"{tw['median_diff_mm']:+.2f}", mx=f"{tw['max_abs_diff_mm']:.2f}", nb=tw["both_under"],
                po=tw["plant_only"], t0=f"{ss['tripod'][0]:.1f}", t1=f"{ss['tripod'][1]:.1f}",
                p0=f"{ss['pinch'][0]:.1f}", p1=f"{ss['pinch'][1]:.1f}", r1=f"{tm['rigid_42']['run_s']:.0f}",
                r0=f"{tm['object_42']['run_s']:.0f}", m1=f"{tm['rigid_42']['ms']:.0f}", m0=f"{tm['object_42']['ms']:.0f}",
                x4=f"{tm['rigid_10']['run_s'] / tm['object_10']['run_s']:.1f}",
                xn=f"{tm['none_10']['run_s'] / tm['object_10']['run_s']:.1f}")


def sec_rerun(N):
    R = N["rerun"]
    if not R["rerun"]:
        return "<p>Not run yet.</p>"
    old, new = R["rerun"]["old"], R["rerun"]["new"]
    keys = [k for k in ("twist_d40", "roll_d40", "pinpush_d25", "press_d25") if k in new and k in old]
    if len(keys) < 4:
        return "<p>Not run yet.</p>"
    tw0, tw1 = old["twist_d40"], new["twist_d40"]
    c0, c1 = old["cross"], new["cross"]
    t = fill(r"""<p>Re-solved with the constraint, the grids at the default objects keep most of their layouts and
scores (Table <<tb>>, Figure <<fg>>). The twist's tripod is reached on <<r1>> layouts against <<r0>> before; the skeleton's
pad sphere is wider than the v4b tip across the finger and had rejected some clear poses. The twist's median best-of-16
score is <<b1>> against <<b0>>. Its best layout scores <<bb>> at (<<bx>>, <<by>>) mm, and its largest basin peaks at
(<<px>>, <<py>>) mm. On <<a1>> % of the layouts all four tasks score above 0.5 with their best grasps, against <<a0>> %
before. The best worst-case layout is (<<mx>>, <<my>>) mm, where the four tasks score <<ms>>. The cylinder tasks now
hold the object at its start until the grasp closes; before, it could roll off its post during the approach.</p>""",
             tb=P.TAB[0] + 1, fg=P.FIG[0] + 1, r1=tw1["reached"], r0=tw0["reached"], b1=f"{tw1['best_median']:.2f}",
             b0=f"{tw0['best_median']:.2f}", bb=f"{tw1['best']:.2f}", bx=f"{tw1['best_xy'][0]:.0f}",
             by=f"{tw1['best_xy'][1]:.0f}", px=f"{tw1['peak'][0]:.0f}", py=f"{tw1['peak'][1]:.0f}",
             a1=f"{100 * c1['share_all_above_0.5']:.0f}", a0=f"{100 * c0['share_all_above_0.5']:.0f}",
             mx=f"{c1['maximin_xy'][0]:.0f}", my=f"{c1['maximin_xy'][1]:.0f}",
             ms=", ".join(f"{x:.2f}" for x in c1["maximin_scores"]))
    rows = []
    for k in keys:
        a, b = old[k], new[k]
        task, d = k.split("_d")
        pk = (lambda q: f"{q[0]:.0f}, {q[1]:.0f}" if q else "&ndash;")
        rows.append([NAMES[task], f"{OBJ[task]} {d} mm", f"{a['reached']} / {b['reached']}",
                     f"{a['best_median']:.2f} / {b['best_median']:.2f}", f"{a['fp_median']:.2f} / {b['fp_median']:.2f}",
                     f"{100 * a['above_0.5']:.0f} / {100 * b['above_0.5']:.0f}", f"{b['best']:.2f} at {pk(b['best_xy'])}",
                     f"{pk(a['peak'])} / {pk(b['peak'])}"])
    t += table(["Task", "Object", "Layouts reaching the pose", "Best-of-16 median", "Foundational pose median",
                "Layouts above 0.5 (%)", "Best layout after (mm)", "Largest basin's peak (mm)"], rows,
               "The grids at the default objects before the fix (the skeleton's clearance, all grasps) and after it "
               "(the plant's spheres 2 mm apart at the touch pose), seeds 0 to 2, before / after in each cell. "
               "<code>hom_landscape.py rerun</code>.", num_from=2)
    t += img(MEDIA / "20261030-gridff_maps.png",
             "Best-of-16 score over the grid of symmetric tripods at the default objects, before the fix (top) and after "
             "it (bottom). Grey: the pose is not reached. Star: the best layout. Seeds 0 to 2.")
    cp = R["cpu"]
    if len(cp) == 4:
        t += fill(r"""<p>The CPU plant, with every contact, ran the best grasp of each new grid's <<n>> best layouts at
seed 0 (<code>hom_top_cpu_check.py --grid grid2dff</code>). The medians agree with the GPU's: <<med>>. The index and the
middle still touched during the task, though they started 2 mm apart: in <<tt>> of the <<n>> twist runs, <<tr>> roll
runs, <<tp>> pin-push runs and <<tq>> press runs. The CPU scored under 0.5 where the GPU scored 0.5 or more in <<lt>>
twist runs, <<lr>> roll runs, <<lp>> pin-push runs and <<lq>> press runs (Film <<fm>>). The best layouts of the searches
below therefore run on the CPU as well.</p>""",
                  n=cp["twist"]["n"],
                  med="; ".join(f"{NAMES[k].lower()} {cp[k]['cpu_median']:.2f} against {cp[k]['gpu_median']:.2f}"
                                for k in ("twist", "roll", "pinpush", "press")),
                  tt=cp["twist"]["touch"], tr=cp["roll"]["touch"], tp=cp["pinpush"]["touch"], tq=cp["press"]["touch"],
                  lt=cp["twist"]["low"], lr=cp["roll"]["low"], lp=cp["pinpush"]["low"], lq=cp["press"]["low"],
                  fm=P.FILM[0] + 1)

        def cap_c(i):
            a, b = i["all contacts"], i["pads and small rigid spheres meet the object only"]
            return (f"A re-solved twist layout whose index and middle start 2 mm apart and touch during the twist (x_sep "
                    f"{i['layout'][0]:.0f} mm, y_sep {i['layout'][1]:.0f} mm, GPU score {i['gpu']:.2f}), CPU plant, "
                    f"seed 0: all contacts (left, score {a['score']:.2f}) and the GPU's collision filter (right, "
                    f"{b['score']:.2f}). <code>scripts/hom_films.py contact_ff</code>.")
        t += v4b_film("contact_ff", cap_c)

    def cap_f(i):
        return (f"The four tasks on the re-solved grids' best worst-case layout (x_sep {i['layout'][0]:.0f} mm, y_sep "
                f"{i['layout'][1]:.0f} mm), each with its best grasp, CPU plant, seed 0. Scores: "
                + ", ".join(f"{NAMES[k].lower()} {i[k]['score']:.2f}" for k in ("twist", "roll", "pinpush", "press"))
                + ". <code>scripts/hom_films.py four_ff</code>.")
    t += v4b_film("four_ff", cap_f)
    return t


def sec_search(N):
    E = N["rerun"]["ea"]
    if not E:
        return "<p>Not run yet.</p>"
    T_ = {k: v for k, v in E.items() if isinstance(v, dict) and "runs" in v}
    if len(T_) < 4:
        return "<p>Not run yet.</p>"
    pop = E["pop"]

    def share(k, run, b):
        return T_[k]["runs"][run]["share_within_0.05_by"][str(b)]
    rows = []
    for k, v in T_.items():
        task = k.split("_d")[0]
        rows.append([NAMES[task], f"{v['grid_rollouts']:,}", str(v["layouts_within"]["0.05"]),
                     *[f"{100 * share(k, f'ea_best_pop{4 * pop}', b):.0f}" for b in (2000, 4000, 6000)],
                     *[f"{100 * share(k, f'ea_best_pop{pop}', b):.0f}" for b in (2000, 6000)],
                     *[f"{100 * share(k, 'random_best', b):.0f}" for b in (2000, 6000)]])
    by = [share(k, f"ea_best_pop{4 * pop}", 6000) for k in T_]
    gr = [6000 / T_[k]["grid_rollouts"] for k in T_]
    st16 = [T_[k]["runs"][f"ea_best_pop{pop}"]["to_within_0.05"]["reached"] for k in T_]
    rnd = [share(k, "random_best", 6000) for k in T_]
    t = fill(r"""<p>On the design page the EA came within 0.05 of the grid's best after a median of 5 to 8 % of the
grid's rollouts. The median hides how often a run fails to get there. On the re-solved grids, the EA with a population
of <<p2>> came within 0.05 of the grid's best within 6,000 rollouts in <<e0>> to <<e1>> % of its <<nr>> runs per task
(Table <<tb>>, Figure <<fg>>). That budget is <<g0>> to <<g1>> % of a full grid's rollouts. Random search over the
lattice got there within the same budget in <<q0>> to <<q1>> % of runs. With a population of <<p1>> the EA stalls: it
came within 0.05 in <<s0>> to <<s1>> % of its runs however long it ran, because its best eighth, two layouts, is all it
breeds from. The live searches of the next section therefore use a population of <<p2>> and 6,000 rollouts per task and
object.</p>""",
             p2=4 * pop, p1=pop, nr=E["runs"], e0=f"{100 * min(by):.0f}", e1=f"{100 * max(by):.0f}",
             g0=f"{100 * min(gr):.0f}", g1=f"{100 * max(gr):.0f}", q0=f"{100 * min(rnd):.0f}", q1=f"{100 * max(rnd):.0f}",
             s0=f"{100 * min(st16):.0f}", s1=f"{100 * max(st16):.0f}", tb=P.TAB[0] + 1, fg=P.FIG[0] + 1)
    t += table(["Task", "Grid rollouts", "Layouts within 0.05", f"EA {4 * pop}, 2,000", f"EA {4 * pop}, 4,000",
                f"EA {4 * pop}, 6,000", f"EA {pop}, 2,000", f"EA {pop}, 6,000", "Random, 2,000", "Random, 6,000"], rows,
               f"Share of {E['runs']} runs (%) whose best layout scores within 0.05 of the re-solved grid's best after "
               "2,000, 4,000 and 6,000 rollouts, for the EA with two populations and for random search; default objects, "
               "seeds 0 to 2. <code>hom_landscape.py ea --grid grid2dff</code>.", num_from=1)
    t += img(MEDIA / "20261030-ea_baseline_ff.png",
             "The median over runs of the best-of-16 score of the best layout found, against rollouts, on the re-solved "
             "grids. Dash-dotted: the grid's best; dotted: 0.05 below it.")
    return t


def sec_trends(N, src="grid2dff"):
    Tn = N["trends"] if src == "grid2dff" else N["trends_gr"]
    if not Tn:
        return "<p>Not run yet.</p>"
    rows = []
    for task in ("twist", "roll", "pinpush", "press"):
        for v in sorted((v for v in Tn.values() if v["task"] == task), key=lambda v: v["d_mm"]):
            src_ = v.get("grid") or v.get("ea")
            if src_ is None:
                rows.append([NAMES[task], f"{v['d_mm']:g}", f"{100 * (v['reach'] or 0):.0f}", "&ndash;", "&ndash;",
                             "&ndash;", "&ndash;", "&ndash;"])
                continue
            kind = "grid" if v.get("grid") else "EA"
            c = v.get("cpu")
            rows.append([NAMES[task], f"{v['d_mm']:g}", f"{100 * v['reach']:.0f}", kind,
                         f"{src_['best']:.2f}", f"{src_['best_xy'][0]:.0f}, {src_['best_xy'][1]:.0f}",
                         f"{src_['near_n']} ({src_['near_lo'][0]:.0f}&ndash;{src_['near_hi'][0]:.0f}, "
                         f"{src_['near_lo'][1]:.0f}&ndash;{src_['near_hi'][1]:.0f})",
                         f"{c['best_cpu']:.2f}" if c else "&ndash;"])
    t = table(["Task", "Diameter (mm)", "Reach (%)", "Search", "Best score", "Best layout (mm)",
               "Near set: layouts (x_sep, y_sep range, mm)", "CPU score of the best"], rows,
              "Per task and object size: the share of layouts that reach the pose, the best layout found by the live EA "
              "(population 64, 6,000 rollouts) or by the full grid at the default object, its best-of-16 score on seeds 0 "
              "to 2, the layouts within 0.05 of it, and the CPU plant's score of its best grasp with every contact at "
              "seed 0. <code>hom_ea.py report</code>.", num_from=1)
    suf = "" if src == "grid2dff" else "_gr"
    t += img(MEDIA / f"20261030-size_reach{suf}.png",
             "Layouts (of 625) on which the pose toy finds each pose, against the object's diameter.")
    t += img(MEDIA / f"20261030-size_trends{suf}.png",
             "Per task: the best layout's score against the object's diameter (top) and its x_sep and y_sep (middle, "
             "bottom), with bars over the layouts within 0.05 of it. Filled: the live EA; open: the full grid. Dotted: "
             "the gantries' travel.")
    return t


def sec_table(N):
    G = N["ground"]
    if not G:
        return "<p>Not run yet.</p>"
    rows = []
    for (pose, d), v in sorted(G.items(), key=lambda kv: (kv[0][0] != "tripod", kv[0][0], kv[0][1])):
        rows.append([pose, f"{d:g}", f"{v['grasps']:,}", f"{v['touch_median']:+.1f}", f"{100 * v['touch_ok']:.0f}",
                     f"{100 * v['both_ok']:.0f}", f"{v['layouts_ok']} of {v['layouts']}"])
    t = table(["Pose", "Diameter (mm)", "Grasps", "Table clearance at touch, median (mm)", "Clear at touch (%)",
               "Clear at touch and open (%)", "Layouts with a clear grasp"], rows,
              "The re-solved grids' grasps against a table under the object (<code>scripts/hom_ground.py</code>): the "
              "table clearance's median at the touch pose and the share of grasps within 1 mm of the table.", num_from=1)
    return t


def sec_limits(N):
    return r"""<ul>
<li>The GPU evaluator keeps its collision filter: the index and middle start 2 mm apart, but contact between them during
a task is missed there. The best layouts are therefore re-run on the CPU plant, at seed 0 only.</li>
<li>Each live search ran once per task and object, with one random start; on the full grids 3 % or fewer of such runs
missed the best by more than 0.05.</li>
<li>In the table study the GPU plant keeps its 3 mm post. The grasps clear a table, but a table stands under the object
only in the CPU check, on the default objects.</li>
<li>The lattice is the symmetric tripod in two coordinates; the six-coordinate layouts are the Sobol study's.</li>
</ul>"""


def sec_future(N):
    return r"""<ul>
<li>Finger-to-finger contact on the GPU at small cost. Fit capsules to each v4b part, or a sphere tree, with collisions
across fingers only. Time a 2,016-world twist batch against 31 ms per rollout. Rerun the 40 best twist layouts against
the CPU's scores. The k-means attempt is <code>scripts/hom_proxy_check.py</code>.</li>
<li>Repeat each live search from three random starts and report the spread of the best layout's position per size.</li>
<li>A table in the GPU plant (a box under the object until the lift) to score grasps from a table at grid scale.</li>
</ul>"""


def sec_files():
    return r"""<ul>
<li>Pose toy: <code>scripts/fp_sr2_toy.py</code> (<code>ff_clearance_v4b</code>, <code>grid2d</code>,
<code>--ground</code>); rows <code>grid2dff_*</code> and <code>grid2dgr_*</code>.</li>
<li>The plant's spheres: <code>assets/mjcf/experimental/20261030-cad_v2/finger_spheres.json</code>, from
<code>scripts/hom_grasp_overlap.py export</code>; the toy against the plant: <code>hom_grasp_overlap.py check</code>.</li>
<li>GPU evaluator: <code>scripts/hom_gpu.py</code> (<code>HOM_GPU_FILTER</code> for the timing); rows
<code>grid2dff_v4b/</code>, <code>grid2dgr_v4b/</code>, <code>ff_timing/</code>.</li>
<li>Live EA and the size report: <code>scripts/hom_ea.py</code>; rows <code>ea_live/</code>, <code>ea_live_gr/</code>.</li>
<li>Offline EA and the grid comparison: <code>scripts/hom_landscape.py ea --grid grid2dff</code>, <code>rerun</code>.</li>
<li>CPU checks: <code>scripts/hom_top_cpu_check.py</code>, <code>scripts/hom_table_check.py</code>; table clearance:
<code>scripts/hom_ground.py</code>; films: <code>scripts/hom_films.py</code>.</li>
<li>Queues: <code>logs/20261030-hom_ff/</code> (not tracked).</li>
</ul>"""


# ------------------------------------------------------------------------------------------------ assemble

def assemble(sections, title, h1, eyebrow, lede, byline, summary):
    num = {sid: k for k, (sid, _hd, _b) in enumerate(sections, 1)}
    sub = (lambda s: re.sub(r"SEC\[(\w+)\]", lambda m_: str(num[m_.group(1)]), s))
    sections = [(sid, hd, sub(body)) for sid, hd, body in sections]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>'
                     for k, (sid, hd, body) in enumerate(sections, 1))
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": title, "H1": h1, "EYEBROW": eyebrow, "LEDE": sub(lede), "BYLINE": byline, "TOC": toc,
                 "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    n = []
    page = P.render(page, n)
    page = docs_index.tag(page, topic="design", summary=summary, date="2026-10-30")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {n[0]} formulas", flush=True)


def main():
    N = {"rerun": rerun_numbers(), "trends": trend_numbers("grid2dff"), "trends_gr": trend_numbers("grid2dgr"),
         "ground": ground_numbers(), "table": table_numbers()}
    sections = [
        ("terms", "Terms and metrics", sec_terms()),
        ("fix", "Finger clearance in the pose toy and its cost on the GPU", sec_fix(N)),
        ("rerun", "Grids at the default objects re-solved with the finger clearance", sec_rerun(N)),
        ("search", "Reliability of the evolutionary search on the re-solved grids", sec_search(N)),
        ("trends", "Best layouts across object sizes from the evolutionary search", sec_trends(N)),
        ("table", "Grasps clear of a table under the object", sec_table(N)),
        ("limits", "Limitations", sec_limits(N)),
        ("future", "Future work", sec_future(N)),
        ("refs", "References and files", "<ol class=\"refs\">" + "".join(f"<li>{r[r.index(']') + 2:]}</li>"
                                                                         for r in REFS) + "</ol>" + sec_files()),
    ]
    lede = ("The SR2 layout study of the design page, after its pose toy learned the plant's finger clearance, across "
            "object sizes and with a table under the object.")
    byline = ("Simulation, 2026-10-30. Hand: the CAD-model SR2 hand with the v4b fingertip (<code>SCALE_HAND=cad_v2</code>); "
              f"HOM on MuJoCo-Warp and on the CPU plant. Follows <a href=\"{DESIGN}\">the design page</a>, Section 9. "
              "Builder: <code>scripts/hom_trends_page.py</code>.")
    assemble(sections, "SR2 layouts across object sizes", "SR2 layouts across object sizes under HOM control, with "
             "finger and table clearance", "HOM layout study, 2026-10-30", lede, byline, lede)
    return 0


if __name__ == "__main__":
    sys.exit(main())
