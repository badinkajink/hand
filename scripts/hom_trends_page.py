#!/usr/bin/env python3
"""Builds docs/experiments/20261029-foundational_pose_review/20261030-hom_layout_trends.html: the best layouts of the SR2
hand under HOM control across object sizes, with the fingers kept 2 mm apart and with a table under the object
(2026-10-30 and 31). It follows the design page (20261029-foundational_pose_hand_design.html, built by
scripts/fp_review_page.py, whose helpers and template it uses) from its Section 9.

Data: rows/ea_live (the live EA at 17 sizes, summary.json from scripts/hom_ea.py report), rows/cpu_best
(top_*_ff*.jsonl, table_*.jsonl), rows/ea_ff (the offline EA on the re-solved grids), rows/grid2dff_v4b (re-solved grids,
toy_check.json, rerun.json, ground/), rows/grid2dgr_v4b and ea_live_gr (the grids clear of the table), rows/ff_timing,
rows/films_v4b.json.

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
CONTACT = "../20261030-v3_tip_contact_model/20261030-v3_fingertip_contact_models.html"
fill, table, img, film, esc = P.fill, P.table, P.img, P.film, P.esc
TASKS = ("twist", "roll", "pinpush", "press")
NAMES = {"twist": "Twist", "roll": "Roll", "pinpush": "Pin-push", "press": "Press"}
OBJ = {"twist": "sphere", "roll": "sphere", "pinpush": "cylinder", "press": "cylinder"}
DEFAULT = {"twist": 40.0, "roll": 40.0, "pinpush": 25.0, "press": 25.0}
REFS = [P.REFS[0], P.REFS[1], P.REFS[5]]           # [1] foundational pose, [2] HOM, [3] Yang et al.
CPU_TOL = 0.04                                      # a best layout's CPU score agrees with its GPU score within this
L_SIZES = {"twist": (30.0, 40.0, 50.0, 63.0), "roll": (30.0, 40.0, 50.0, 63.0), "pinpush": (10.0, 16.0, 25.0, 40.0),
           "press": (10.0, 16.0, 25.0, 40.0)}        # the Sobol study's sizes, hom_landscape.SIZES


def load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


def v4b_film(name, caption_fn, date="20261030"):
    """Film media/<date>-film_<name>.mp4 with its caption from rows/films_v4b.json, or nothing."""
    path = MEDIA / f"{date}-film_{name}.mp4"
    info = json.loads(P.FILMS_V4B.read_text()).get(name) if P.FILMS_V4B.exists() else None
    return film(path, caption_fn(info)) if (path.exists() and info) else ""


def join_and(items):
    """'a', 'a and b', 'a, b and c'."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def span(v, fmt="{:.2f}"):
    """'a to b' over a list, or 'a' when both ends print alike."""
    lo, hi = fmt.format(min(v)), fmt.format(max(v))
    return lo if lo == hi else f"{lo} to {hi}"


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
    rc = G / "reach_check.json"                          # scripts/hom_toy_reach_check.py
    out["reach_check"] = json.loads(rc.read_text()) if rc.exists() else {}
    return out


def ea_top_calls(task, d, out="ea_live", n=10):
    """The n best layouts (lattice indices) of a live search by hom_ea.fitness: per layout, the best grasp's mean score
    over all its rows (a layout drawn twice in one generation has six)."""
    import hom_landscape as L
    sc = {}
    for r in load_jsonl(ROWS / out / f"{task}_d{d:g}_s0.jsonl"):
        sc.setdefault(int(r["call"]), {}).setdefault(int(r.get("grasp", 0)), []).append(L.rollout_score(task, r))
    fit = {c: max(float(np.mean(v)) for v in g.values()) for c, g in sc.items()}
    return sorted(fit, key=lambda c: -fit[c])[:n]


def trend_numbers(src="grid2dff"):
    """hom_ea.py report's summary (rows/ea_live[_gr]/summary.json) and the CPU check of each search's 10 best layouts
    (at the default objects, of the full grid's 40 best)."""
    p_ = ROWS / ("ea_live" if src == "grid2dff" else "ea_live_gr") / "summary.json"
    if not p_.exists():
        return None
    S = json.loads(p_.read_text())
    out = {k: v for k, v in S.items() if isinstance(v, dict) and "task" in v}
    for k, v in out.items():
        stem = f"top_{v['task']}_d{v['d_mm']:g}_" + ("ff" if src == "grid2dff" else "gr")
        if v.get("grid"):
            rows, calls = load_jsonl(ROWS / "cpu_best" / f"{stem}.jsonl"), None
        else:
            rows = load_jsonl(ROWS / "cpu_best" / f"{stem}_ea.jsonl")
            calls = ea_top_calls(v["task"], v["d_mm"], "ea_live" if src == "grid2dff" else "ea_live_gr")
        full = [r for r in rows if not r["gpu_filter"] and (calls is None or r["call"] in calls)]
        if full:
            top = (next((r for r in full if r["call"] == calls[0]), None) if calls else None) or \
                max(full, key=lambda r: r["gpu_best"])
            v["cpu"] = {"n": len(full), "touch": sum(1 for r in full if r["finger_touch_periods"]),
                        "top_touch": bool(top["finger_touch_periods"]),
                        "low": sum(1 for r in full if r["score"] < 0.5 <= r["gpu_best"]),
                        "best_cpu": top["score"], "best_gpu": top["gpu_best"], "best_xy": [top["x_sep"], top["y_sep"]]}
    return out


def ground_numbers():
    """The table clearance of the re-solved grids' grasps (scripts/hom_ground.py): per pose family and size the share of
    grasps and of layouts within 1 mm of the table at the touch pose and at the touch and open poses; the layouts that
    reach the pose when the toy keeps the table clear (rows grid2dgr_*) and their contacts' latitudes."""
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
    for (pose, d), v in out.items():                   # the grids solved with the table as a constraint
        rows = FF.load(f"grid2dgr_{'tripod' if pose == 'tripod' else 'pinch'}{d:g}_v4b")
        if rows:
            recs = [(r["fp"].get(pose) or {}) for r in rows]
            ok = [r for r in recs if r.get("ok")]
            lat = [np.degrees(g["v"][6]) for r in ok for g in (r.get("grasps") or [])]
            v["gr_reached"] = len(ok)
            v["gr_lat"] = [float(np.percentile(lat, 5)), float(np.percentile(lat, 95))] if lat else None
    return out


def gr_grid_numbers():
    """The GPU's best-of-16 scores on the grids clear of the table at the default objects, beside the re-solved grids'."""
    import hom_landscape as L
    out = {}
    for task, d in DEFAULT.items():
        if not (ROWS / "grid2dgr_v4b" / f"{task}_d{d:g}.jsonl").exists():
            continue
        res = {}
        for src in ("grid2dff", "grid2dgr", "grid2dgn", "grid2dgs"):
            if not (ROWS / f"{src}_v4b" / f"{task}_d{d:g}.jsonl").exists():
                continue
            keep = L.GRID_SRC, L.GRID_ROWS
            L.GRID_SRC, L.GRID_ROWS = src, ROWS / f"{src}_v4b"
            try:
                tab = [r for r in L.grid_table(task, d) if r.get("n_grasps")]
            finally:
                L.GRID_SRC, L.GRID_ROWS = keep
            top = max(tab, key=lambda r: r["best"])
            pose = FF.load(f"{src}_{'tripod' if OBJ[task] == 'sphere' else 'pinch'}{d:g}_v4b")
            key = {"twist": "tripod", "roll": "tripod", "pinpush": "pinch", "press": "press"}[task]
            res[src] = {"layouts": len(tab), "best": top["best"], "best_xy": [top["x_sep"], top["y_sep"]],
                        "reached": sum(1 for r in pose if (r["fp"].get(key) or {}).get("ok")),
                        "median": float(np.median([r["best"] for r in tab])),
                        "above": float(np.mean([r["best"] > 0.5 for r in tab]))}
        out[task] = res
    return out


def table_numbers():
    """The CPU plant with a physical table (scripts/hom_table_check.py) on the best layouts of both grids."""
    out = {}
    for task, d in DEFAULT.items():
        rows = load_jsonl(ROWS / "cpu_best" / f"table_{task}_d{d:g}.jsonl")
        if not rows:
            continue
        res = {}
        for src in ("grid2dff", "grid2dgr", "grid2dgn", "grid2dgs"):
            for tb in (False, True):
                rr = [r for r in rows if r["src"] == src and r["table"] == tb]
                if rr:
                    on = [r for r in rr if r["table_touch_periods"]]
                    off = [r for r in rr if not r["table_touch_periods"]]
                    res[f"{src}_{'table' if tb else 'post'}"] = {
                        "n": len(rr), "held": sum(1 for r in rr if r["held"]),
                        "median": float(np.median([r["score"] for r in rr])),
                        "touch": len(on), "touch_dropped": sum(1 for r in on if not r["held"]),
                        "touch_force": [min((r["table_force_max_N"] for r in on), default=0.0),
                                        max((r["table_force_max_N"] for r in on), default=0.0)],
                        "free": len(off), "free_held": sum(1 for r in off if r["held"]),
                        "free_median": float(np.median([r["score"] for r in off])) if off else None,
                        "gpu_median": float(np.median([r["gpu_best"] for r in rr]))}
        out[task] = res
    return out


def trend_stats(Tn, task):
    """Per task: sizes, reach, best scores and positions, the near sets' sizes and centroid slopes (mm per mm of
    diameter, least squares), and the CPU check of each size's best layout."""
    pts = sorted(((v["d_mm"], v.get("grid") or v.get("ea"), v) for v in Tn.values()
                  if v["task"] == task and (v.get("grid") or v.get("ea"))), key=lambda p_: p_[0])
    if len(pts) < 3:
        return None
    d = np.array([p_[0] for p_ in pts])
    cx = np.array([p_[1]["near_centroid"][0] for p_ in pts])
    cy = np.array([p_[1]["near_centroid"][1] for p_ in pts])
    return {"d": d, "reach": [p_[2]["reach"] for p_ in pts], "best": [p_[1]["best"] for p_ in pts],
            "best_x": [p_[1]["best_xy"][0] for p_ in pts], "best_y": [p_[1]["best_xy"][1] for p_ in pts],
            "near_n": [p_[1]["near_n"] for p_ in pts], "slope_x": float(np.polyfit(d, cx, 1)[0]),
            "slope_y": float(np.polyfit(d, cy, 1)[0]),
            "cpu": [(p_[0], p_[2]["cpu"]) for p_ in pts if p_[2].get("cpu")],
            "layouts": [p_[1]["layouts"] for p_ in pts if p_[2].get("ea") and not p_[2].get("grid")],
            "rollouts": [p_[1]["rollouts"] for p_ in pts if p_[2].get("ea") and not p_[2].get("grid")]}


def repeat_share(out="ea_live"):
    """Share of the live searches' rollouts that repeat a layout drawn twice in one generation (rows beyond a grasp's
    three seeds)."""
    from collections import Counter
    tot = rep = 0
    for p_ in sorted((ROWS / out).glob("*_s0.jsonl")):
        c = Counter((r["call"], r.get("grasp", 0)) for r in load_jsonl(p_))
        tot += sum(c.values())
        rep += sum(n - 3 for n in c.values() if n > 3)
    return rep / tot if tot else None


def table_costs(N):
    """The searches with and without the table constraint across sizes (rows/ea_live_gr and ea_live summaries): per task
    {d: (best without, best with, near-set x_sep without, with)} where both ran."""
    A, B = N.get("trends"), N.get("trends_gr")
    if not (A and B):
        return None
    out = {}
    for task in ("twist", "roll", "pinpush"):
        a = {v["d_mm"]: v.get("grid") or v.get("ea") for v in A.values() if v["task"] == task}
        b = {v["d_mm"]: v.get("grid") or v.get("ea") for v in B.values() if v["task"] == task}
        out[task] = {d: (a[d]["best"], b[d]["best"], a[d]["near_centroid"][0], b[d]["near_centroid"][0])
                     for d in sorted(a) if a.get(d) and b.get(d)}
    return out


def sobol_numbers():
    """The Sobol rerun with the finger clearance (rows/sizes_v4bff: rerun.json from hom_landscape.py sobol-rerun,
    landscape.json from stats --sobol sobol8kff) and the first run's statistics (rows/sizes_v4b/landscape.json)."""
    G, out = ROWS / "sizes_v4bff", {}
    for k, path in (("rerun", G / "rerun.json"), ("new", G / "landscape.json"),
                    ("old", ROWS / "sizes_v4b" / "landscape.json")):
        out[k] = json.loads(path.read_text()) if path.exists() else None
    return out


def cpu_agreement(St):
    """Over every task and size with a CPU check: how many best layouts agree within CPU_TOL, and the others."""
    n, ok, bad = 0, 0, []
    for task, s in St.items():
        for d, c in (s or {}).get("cpu", []):
            n += 1
            if abs(c["best_cpu"] - c["best_gpu"]) <= CPU_TOL:
                ok += 1
            else:
                bad.append((task, d, c))
    return n, ok, bad


# ------------------------------------------------------------------------------------------------ sections

def sec_terms():
    terms = [
        ("Layout", "the six palm-frame mount coordinates (mm) of the SR2 hand's three gantries. Only the symmetric "
         "tripod is used here, in two coordinates. The thumb's mount lies $x_{sep}$ from the pair's, and the index's "
         "lies $y_{sep}$ from the middle's. The lattice has 625 layouts in steps of 5 mm. Its $x_{sep}$ runs from 40 to "
         "160 mm and its $y_{sep}$ from 50 to 170 mm."),
        ("HOM tasks", "the four tasks of the design page under HOM control [2]. The twist turns a sphere about the "
         "vertical, and the roll rolls it along the thumb's pad. In the pin-push and the press, the middle finger turns a "
         "cylinder about the pinch of thumb and index. Default objects: a 40 mm sphere, and a cylinder 25 mm across and "
         "100 mm long."),
        ("Task score", "a rollout's score in [0, 1], 0 when the object is dropped. Twist: the largest turn with all "
         "three pads in contact, over 90 deg. Roll: $1/(1+e)$, where $e$ is the RMS error of the rolling rate over the "
         "commanded rate. Pin-push and press: one less the RMS error of the cylinder's angle over the stroke of 15 and "
         "20 deg. A grasp's score is its mean over seeds 0 to 2."),
        ("Grasp set, best of 16", "up to 16 grasps of one layout and object solved by the pose toy, the foundational "
         "pose [1] first. A layout's best of 16 is its highest grasp score."),
        ("GPU evaluator, CPU plant", "the GPU evaluator runs HOM in batches of up to 2,048 rollouts on MuJoCo-Warp. "
         "There the pads and the rigid spheres under 2 mm collide with the object only (the collision filter). The CPU "
         "plant runs the same scene on MuJoCo with every contact."),
        ("CPU check", "a layout's best grasp run once more on the CPU plant, seed 0. Its score is the CPU score."),
        ("Touch pose, open pose", "the plant's finger joint angles with each pad's own surface on its contact point. "
         "The HOM pick starts at the open pose, 8 mm further out along each contact normal."),
        ("Finger clearance", "the smallest surface distance (mm) between the collision spheres of two fingers at the "
         "touch pose. Each finger has 1,465 spheres: the sphere packs of its links and the spheres of its pad."),
        ("Table clearance", "the smallest height (mm) of the bottom of a finger's collision sphere above the plane "
         "through the object's lowest point. That plane is the table the object would rest on. A grasp clears the table "
         "when this height is at least -1 mm at the touch pose and at the open pose."),
        ("EA", "the elitist, mutation-only evolutionary algorithm of Yang et al. [3]. In each generation the best half "
         "is kept and the best eighth are the parents. A child is a parent moved by up to 15 mm per coordinate, rounded "
         "to the 5 mm lattice. Population: the layouts per generation. Budget: the rollouts after which no generation "
         "starts."),
        ("Within 0.05", "a search succeeds when its best layout so far scores within 0.05 of the full grid's best."),
        ("Near set, centroid", "the layouts a search ran that score within 0.05 of the best one it reached. The centroid "
         "is their mean $x_{sep}$ and $y_{sep}$."),
        ("Slope", "the least-squares slope of the near set's centroid against the object's diameter, in mm of mount "
         "spacing per mm of diameter."),
        ("Reach", "the share of the 625 layouts on which the pose toy finds the task's pose for the object."),
        ("Spearman, best tenth kept", "Spearman's rank correlation of two runs' scores over the layouts both ran, and "
         "the share of one run's best tenth of layouts that is in the other's best tenth."),
        ("Hold", "the cylinder tasks' plant variant. The object is held at its start pose until the grasp closes at "
         "0.8 s, so it cannot roll off its 3 mm post during the approach."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"


def sec_trends(N):
    Tn = N["trends"]
    if not Tn:
        return "<p>Not run yet.</p>"
    St = {task: trend_stats(Tn, task) for task in TASKS}
    if not all(St.values()):
        return "<p>Not run yet.</p>"
    tw, ro, pp, pr = (St[k] for k in TASKS)
    lay = [x for s in St.values() for x in s["layouts"]]
    tb, f1, fm = P.TAB[0] + 1, P.FIG[0] + 1, P.FILM[0] + 1
    t = fill(r"""<p>Each task was searched at 8 or 9 object sizes besides its default one. The twist and the roll used
spheres of 25 to 80 mm, and the pin-push and the press used cylinders 100 mm long and 8 to 50 mm across. At each size the
pose toy first solved the grasp sets of all 625 layouts. The EA of Section SEC[search] then ran on the GPU evaluator from
64 random layouts. It stopped after the generation that passed 6,000 rollouts (<code>scripts/hom_ea.py</code>). A search
ran <<l0>> to <<l1>> layouts in about 5 minutes. At the default objects the full grids of Section SEC[rerun] stand in.
The 10 best layouts of each search then ran on the CPU plant.</p>""", l0=min(lay), l1=max(lay))
    hi = [d_ for d_, b in zip(tw["d"], tw["best"]) if b >= 0.97]
    d_hi = next((tw["d"][k - 1] for k, b in enumerate(tw["best"]) if b < 0.97), tw["d"][-1]) if hi else None
    fi = json.loads(P.FILMS_V4B.read_text()) if P.FILMS_V4B.exists() else {}
    t80 = (fi.get("sizes_twist") or {}).get("80")
    e = [1.0 / b - 1.0 for b in ro["best"]]
    k_pr = int(np.argmax(pr["best"]))
    t += fill(r"""<p>The twist's best layout scores 0.97 or more up to the <<dh>> mm sphere and <<tw80>> at 80 mm (Table
<<tb>>, Figure <<f1>>).<<film80>> The roll's best layout scores <<ro>> at every size. Its rolling rate's RMS error stays at
<<e>> times the commanded rate. The pin-push's best layout scores <<pp>>. The press's rises from <<pr8>> on the 8 mm
cylinder to <<prm>> at <<prd>> mm.</p>""",
              dh=f"{d_hi:g}", tw80=f"{tw['best'][-1]:.2f}", tb=tb, f1=f1, ro=span(ro["best"]), e=span(e),
              pp=span(pp["best"]), pr8=f"{pr['best'][0]:.2f}", prm=f"{pr['best'][k_pr]:.2f}",
              prd=f"{pr['d'][k_pr]:g}",
              film80=(f" On the 80 mm sphere the turn reached {t80['ramp_three_pads_max_deg']:.0f} deg with three pads in "
                      f"contact (Film {fm}). By the hold, the thumb's pad had moved over the top of the sphere, and the "
                      "thumb and the index pressed on it with no force." if t80 else ""))
    edge = sum(1 for y in pr["best_y"] if y >= 155)
    far = [n for d_, n in zip(tw["d"], tw["near_n"]) if d_ >= 50]
    t += fill(r"""<p>The best layouts move with the object (Figures <<f1>> and <<f2>>). The near set's centroid moves
$x_{sep}$ out by <<sx_tw>> mm per millimetre of diameter in the twist, <<sx_ro>> in the roll, <<sx_pp>> in the pin-push
and <<sx_pr>> in the press. A slope of 1 keeps each mount at a fixed distance from the object's surface. On spheres
$y_{sep}$ narrows, by <<sy_tw>> mm per millimetre in the twist and <<sy_ro>> in the roll. On cylinders it widens, by
<<sy_pp>> in the pin-push and <<sy_pr>> in the press. At <<edge>> of its <<n_pr>> sizes the press's best $y_{sep}$ is 155
mm or more, within 15 mm of the lattice's edge.</p>
<p>From 50 mm on, <<far>> layouts lie within 0.05 of the twist's best. The roll has <<ro_n>> such layouts at every size,
as its scores differ little between layouts. The share of the lattice that reaches the pose depends on the size as well
(Figure <<f3>>). The tripod is reached on <<r25>> % of the layouts at 25 mm, <<rmax>> % at <<dmax>> mm and <<r80>> % at
80 mm. The pinch is reached on <<p8>> % at 8 mm and <<pmax>> % at <<pdmax>> mm. The press is reached on <<q>> % at
every size.</p>""",
              f1=f1, f2=f1 + 1, f3=f1 + 2, sx_tw=f"{tw['slope_x']:.2f}", sx_ro=f"{ro['slope_x']:.2f}",
              sx_pp=f"{pp['slope_x']:.2f}", sx_pr=f"{pr['slope_x']:.2f}", sy_tw=f"{-tw['slope_y']:.2f}",
              sy_ro=f"{-ro['slope_y']:.2f}", sy_pp=f"{pp['slope_y']:.2f}", sy_pr=f"{pr['slope_y']:.2f}", edge=edge,
              n_pr=len(pr["d"]), far=span(far, "{:d}"), ro_n=span(ro["near_n"], "{:d}"),
              r25=f"{100 * tw['reach'][0]:.0f}", rmax=f"{100 * max(tw['reach']):.0f}",
              dmax=f"{tw['d'][int(np.argmax(tw['reach']))]:g}", r80=f"{100 * tw['reach'][-1]:.0f}",
              p8=f"{100 * pp['reach'][0]:.0f}", pmax=f"{100 * max(pp['reach']):.0f}",
              pdmax=f"{pp['d'][int(np.argmax(pp['reach']))]:g}", q=span([100 * x for x in pr["reach"]], "{:.0f}"))
    n, ok, bad = cpu_agreement(St)
    exc = [f"the {NAMES[task].lower()} on the {d_:g} mm {OBJ[task]} ({c['best_cpu']:.2f} against {c['best_gpu']:.2f})"
           for task, d_, c in bad]
    tw25 = next((c for d_, c in tw["cpu"] if d_ == 25), None)
    cyl_touch = sum(c["touch"] for s in (pp, pr) for _d, c in s["cpu"])
    t += fill(r"""<p>On the CPU plant, the best layout of each search scored within <<tol>> of its GPU score at <<ok>> of
<<n>> sizes. <<exc>><<tw25>> <<cyl>></p>""",
              tol=CPU_TOL, ok=ok, n=n,
              exc=("The exceptions are " + " and ".join(exc) + "." if len(exc) > 1 else
                   f"The exception is {exc[0]}." if exc else ""),
              tw25=(f" On the 25 mm sphere the index and the middle touched in {tw25['touch']} of the {tw25['n']} twist "
                    "runs." if tw25 and tw25["touch"] else ""),
              cyl=("No two fingers touched during the pin-push or the press at any size." if cyl_touch == 0 else
                   f"Two fingers touched in {cyl_touch} of the cylinder runs."))
    rows = []
    for task in TASKS:
        for v in sorted((v for v in Tn.values() if v["task"] == task), key=lambda v: v["d_mm"]):
            src_ = v.get("grid") or v.get("ea")
            if src_ is None:
                continue
            c = v.get("cpu")
            rows.append([NAMES[task].replace("-", "&#8209;"), f"{v['d_mm']:g}", f"{100 * v['reach']:.0f}",
                         "grid" if v.get("grid") else "EA", f"{src_['best']:.2f}",
                         f"{c['best_cpu']:.2f}" if c else "&ndash;",
                         f"{src_['best_xy'][0]:.0f}, {src_['best_xy'][1]:.0f}", f"{src_['near_n']}"])
    t += table(["Task", "Diameter (mm)", "Reach (%)", "Search", "Best score", "CPU score", "Best layout (mm)",
                "Near set (layouts)"], rows,
               "Per task and object size: the share of layouts that reach the pose; the best layout found by the live EA "
               "(population 64, 6,000 rollouts) or by the full grid at the default object, as $x_{sep}$, $y_{sep}$; its "
               "best-of-16 score on seeds 0 to 2 and its CPU score; the number of layouts in its near set, whose ranges "
               f"are the bars of Figures {P.FIG[0] + 1} and {P.FIG[0] + 2}. <code>hom_ea.py report</code>.",
               num_from=1)
    t += img(MEDIA / "20261030-size_trends_sph.png",
             "Spheres. Top: the best layout's score against the diameter. Middle and bottom: its $x_{sep}$ and $y_{sep}$, "
             "with bars over the near set. Filled: the live EA; open: the full grid. Crosses: the near set's centroid; "
             "red line: its least-squares fit. Dotted: the lattice's edges.")
    t += img(MEDIA / "20261030-size_trends_cyl.png", f"Cylinders, drawn as Figure {f1}.")
    t += img(MEDIA / "20261030-size_reach.png",
             "Layouts of the 625 that reach each pose, against the object's diameter.")

    def cap(i, task):
        parts = [f"{k} mm: layout ({v['layout'][0]:.0f}, {v['layout'][1]:.0f}) mm, score {v['score']:.2f} here and "
                 f"{v['gpu']:.2f} on the GPU." for k, v in i.items() if isinstance(v, dict) and "layout" in v]
        return (f"The {NAMES[task].lower()} on the best layout found at three {OBJ[task]} diameters, each with its best "
                f"grasp, CPU plant, seed 0. " + " ".join(parts) + f" <code>scripts/hom_films.py sizes_{task}</code>.")
    for task in ("twist", "roll", "pinpush", "press"):
        t += v4b_film(f"sizes_{task}", lambda i, task=task: cap(i, task))
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
    pct = (lambda v: span([100 * x for x in v], "{:.0f}"))
    t = fill(r"""<p>The searches of Section SEC[trends] rest on the EA's reliability, which was measured on the full grids
at the default objects. Each search was repeated <<nr>> times per task from random starts (<code>hom_landscape.py ea
--grid grid2dff</code>). With a population of <<p2>>, the EA came within 0.05 of the grid's best within 6,000 rollouts in
<<e>> % of its runs (Table <<tb>>, Figure <<fg>>). That budget is <<g>> % of a full grid's rollouts. Random search over
the lattice got there within the same budget in <<q>> % of its runs. With a population of <<p1>>, the EA got there in
<<st>> % of its runs, however long it ran. On the design page's grids, with the overlapping grasps masked, 55 % of its
twist runs did. Its best eighth is two layouts, and with a population of <<p1>> the best basin can be lost.</p>""",
             p2=4 * pop, p1=pop, nr=E["runs"], e=pct(by), g=pct(gr), q=pct(rnd), st=pct(st16), tb=P.TAB[0] + 1,
             fg=P.FIG[0] + 1)
    t += table(["Task", "Grid rollouts", "Layouts within 0.05", f"EA {4 * pop}, 2,000", f"EA {4 * pop}, 4,000",
                f"EA {4 * pop}, 6,000", f"EA {pop}, 2,000", f"EA {pop}, 6,000", "Random, 2,000", "Random, 6,000"], rows,
               f"Share of {E['runs']} runs (%) whose best layout scores within 0.05 of the re-solved grid's best after "
               "2,000, 4,000 and 6,000 rollouts, for the EA with two populations and for random search; default objects, "
               "seeds 0 to 2.", num_from=1)
    t += img(MEDIA / "20261030-ea_baseline_ff.png",
             "The median over runs of the best-of-16 score of the best layout so far, against rollouts, on the re-solved "
             "grids at the default objects.")
    return t


def sec_fix(N):
    R = N["rerun"]
    ck, tm, ss = R["check"], R["timing"], R["solve_s"]
    if not (ck and tm):
        return "<p>Not run yet.</p>"
    tw = ck["twist_d40"]
    return fill(r"""<p>Finger-to-finger contact in the GPU evaluator would cost 9 to 13 times the rollout time (rows
<code>ff_timing</code>, the first 42 layouts of the earlier twist grid). With the small rigid spheres colliding with every
finger part, as on the CPU, a batch of 2,016 twist worlds took <<r1>> s. With the collision filter it took <<r0>> s,
<<m1>> against <<m0>> ms per rollout. At 480 worlds the ratio was <<x4>>, and with the pads colliding as well it was
<<xn>>. The filter therefore stays, and the fingers are kept apart before a rollout starts. A coarser stand-in did not fit:
96 k-means spheres per fingertip, each grown to contain its members, still reached 7.6 mm in radius on a tip 15 mm
wide.</p>
<p>The pose toy now keeps the fingers apart with the plant's own collision spheres
(<code>fp_sr2_toy.ff_clearance_v4b</code>). Before, a third of the design page's sphere grasps overlapped the index's and
the middle's caps at the touch pose (design page, Section 9). Each finger has 1,465 spheres: the sphere packs of its links
and the 1,018 spheres of its pad. The toy places them by its closed-form kinematics, which reproduced MuJoCo's body poses
to within 1e-15 m at 50 random sets of joint angles. Every two fingers must stay 2 mm apart at the touch pose. That
clearance is a constraint of the toy's program and of its grasp walk.</p>
<p>On <<n>> grasps of the earlier twist grid, the toy's clearance differed from the plant's by a median of <<md>> mm and
at most <<mx>> mm. Both put the same <<nb>> grasps under 2 mm, and <<po>> more fell under it on the plant alone. The
constraint slows the toy. A tripod layout's pose and grasp set took a median of <<t0>> s before and <<t1>> s after, and a
pinch layout's took <<p0>> and <<p1>> s.</p>""",
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
    t = fill(r"""<p>Re-solved with the finger clearance, the grids at the default objects keep most of their layouts and
scores (Table <<tb>>, Figure <<fg>>). The twist's tripod is reached on <<r1>> layouts against <<r0>> before.<<why>> The
twist's median best-of-16 score is <<b1>> against <<b0>>. Its best layout scores <<bb>> at (<<bx>>, <<by>>) mm, and its largest basin
peaks at (<<px>>, <<py>>) mm. On <<a1>> % of the layouts all four tasks score above 0.5 with their best grasps, against
<<a0>> % before. The best worst-case layout is (<<mx>>, <<my>>) mm, where the four tasks score <<ms>>. The cylinder tasks
now hold the object at its start until the grasp closes; before, it could roll off its post during the approach.</p>""",
             tb=P.TAB[0] + 1, fg=P.FIG[0] + 1, r1=tw1["reached"], r0=tw0["reached"], why=reach_why(R, "grid2dff"),
             b1=f"{tw1['best_median']:.2f}",
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
             "Best-of-16 score over the lattice at the default objects, before the fix (top) and after it (bottom). "
             "Grey: the pose is not reached. Star: the best layout. Seeds 0 to 2.")
    cp = R["cpu"]
    if len(cp) == 4:
        t += fill(r"""<p>The best grasp of each new grid's <<n>> best layouts then ran on the CPU plant, seed 0
(<code>hom_top_cpu_check.py --grid grid2dff</code>). On every task the CPU's median differs from the GPU's by <<med>> or
less. The index and the middle still touched during the task, though they started 2 mm apart. They touched in <<tt>> of
the <<n>> twist runs, <<tr>> of the roll runs, <<tp>> of the pin-push runs and <<tq>> of the press runs. The CPU scored
under 0.5 where the GPU scored 0.5 or more in <<lt>>, <<lr>>, <<lp>> and <<lq>> of the twist, roll, pin-push and press
runs (Film <<fm>>). The best layouts of every search are therefore checked on the CPU as well.</p>""",
                  n=cp["twist"]["n"],
                  med=f"{max(abs(cp[k]['cpu_median'] - cp[k]['gpu_median']) for k in cp):.2f}",
                  tt=cp["twist"]["touch"], tr=cp["roll"]["touch"], tp=cp["pinpush"]["touch"], tq=cp["press"]["touch"],
                  lt=cp["twist"]["low"], lr=cp["roll"]["low"], lp=cp["pinpush"]["low"], lq=cp["press"]["low"],
                  fm=P.FILM[0] + 1)

        def cap_c(i):
            a, b = i["all contacts"], i["pads and small rigid spheres meet the object only"]
            return (f"A re-solved twist layout, $x_{{sep}}$ {i['layout'][0]:.0f} mm and $y_{{sep}}$ {i['layout'][1]:.0f} "
                    f"mm, with a GPU score of {i['gpu']:.2f}. Its index and middle start 2 mm apart and touch during the "
                    f"twist. CPU plant, seed 0: all contacts on the left (score {a['score']:.2f}), the GPU's collision "
                    f"filter on the right ({b['score']:.2f}). <code>scripts/hom_films.py contact_ff</code>.")
        t += v4b_film("contact_ff", cap_c)

    def cap_f(i):
        return (f"The four tasks on the re-solved grids' best worst-case layout ($x_{{sep}}$ {i['layout'][0]:.0f} mm, "
                f"$y_{{sep}}$ {i['layout'][1]:.0f} mm), each with its best grasp, CPU plant, seed 0. Scores: "
                + ", ".join(f"{NAMES[k].lower()} {i[k]['score']:.2f}" for k in TASKS)
                + ". <code>scripts/hom_films.py four_ff</code>.")
    t += v4b_film("four_ff", cap_f)
    return t


def reach_why(R, stem):
    """Why the toy reaches more tripods with the clearance in its program (scripts/hom_toy_reach_check.py)."""
    c = (R.get("reach_check") or {}).get(f"{stem}_tripod40_v4b")
    if not c:
        return ""
    return fill(""" Before, the toy solved from one start without the clearance in its program, and it rejected the
result when two fingers' skeletons came within 1 mm. Solved that way again, <<rf>> of <<rt>> layouts that only the new toy
reaches were rejected for their fingers (<code>scripts/hom_toy_reach_check.py</code>). The new toy's poses on them keep
the skeletons a median of <<sk>> mm apart.""", rf=c["old_path"]["fingers"], rt=c["tested"],
                sk=f"{c['new_pose_skeleton_mm_median']:.1f}")


def sec_sobol(N):
    S = N["sobol"]
    if not (S["rerun"] and S["old"] and S["new"]):
        return "<p>Not run yet.</p>"
    R = S["rerun"]
    sph = [k for k in R if k.startswith("twist")]
    cyl = [k for k in R if k.startswith(("pinpush", "press"))]
    gain = [100 * (R[k]["new"]["reached"] / R[k]["old"]["reached"] - 1) for k in sph]
    cd = max(abs(R[k]["new"]["reached"] - R[k]["old"]["reached"]) for k in cyl)
    tw40, pr25 = R["twist_d40"], R["press_d25"]
    var = {t: S["new"]["tasks"][t]["variance"] for t in TASKS}
    ends = {t: S["new"]["tasks"][t]["pairs"] for t in TASKS}
    sp = {t: next(v["spearman"] for k, v in ends[t].items()
                  if k == f"{min(L_SIZES[t]):g}-{max(L_SIZES[t]):g}") for t in TASKS}
    t = fill(r"""<p>The design page's 8,192 Sobol layouts ran again with the finger clearance in the pose toy, at four
sizes per object and seeds 0 to 2 (rows <code>sizes_v4bff</code>, <code>hom_landscape.py sobol-rerun</code>). The
cylinder tasks ran with the hold, as everywhere on this page. On spheres the tripod is now reached on <<g0>> to <<g1>> %
more layouts, for the reason given in Section SEC[rerun]. On cylinders the reach changed by <<cd>> layouts or fewer. The
added sphere layouts mostly score low: the twist's share of layouts above 0.5 falls from <<a0>> to <<a1>> % at 40 mm.</p>
<p>On the layouts both runs reached, the scores rank alike on large objects and less so on small ones (Table <<tb>>).
Spearman's rank correlation is <<r63t>> for the twist and <<r63r>> for the roll on the 63 mm sphere, and <<r30t>> and
<<r30r>> on the 30 mm sphere. The hold raises the cylinder tasks' scores on small cylinders. On the 25 mm cylinder the
press's median rises from <<p0>> to <<p1>>, and its share above 0.5 from <<q0>> to <<q1>> %.</p>
<p>The design page's finding on object size stands. The interaction of layout and size makes up <<i0>> to <<i1>> % of the
score variance, against <<n1>> % or less from seed noise. The ranks at the smallest and the largest object correlate at
<<s0>> to <<s1>> for the twist, the roll and the pin-push, and at <<spr>> for the press.</p>""",
             g0=f"{min(gain):.0f}", g1=f"{max(gain):.0f}", cd=cd, a0=f"{100 * tw40['old']['above_0.5']:.0f}",
             a1=f"{100 * tw40['new']['above_0.5']:.0f}", tb=P.TAB[0] + 1,
             r63t=f"{R['twist_d63']['spearman']:.2f}", r63r=f"{R['roll_d63']['spearman']:.2f}",
             r30t=f"{R['twist_d30']['spearman']:.2f}", r30r=f"{R['roll_d30']['spearman']:.2f}",
             p0=f"{pr25['old']['median']:.2f}", p1=f"{pr25['new']['median']:.2f}",
             q0=f"{100 * pr25['old']['above_0.5']:.0f}", q1=f"{100 * pr25['new']['above_0.5']:.0f}",
             i0=f"{100 * min(v['interaction'] for v in var.values()):.0f}",
             i1=f"{100 * max(v['interaction'] for v in var.values()):.0f}",
             n1=f"{100 * max(v['interaction_noise'] for v in var.values()):.0f}",
             s0=f"{min(sp[k] for k in ('twist', 'roll', 'pinpush')):.2f}",
             s1=f"{max(sp[k] for k in ('twist', 'roll', 'pinpush')):.2f}", spr=f"{sp['press']:.2f}")
    rows = []
    for k, v in R.items():
        task, d = k.split("_d")
        o, n = v["old"], v["new"]
        rows.append([NAMES[task].replace("-", "&#8209;"), d, f"{o['reached']:,} / {n['reached']:,}",
                     f"{o['median']:.2f} / {n['median']:.2f}", f"{100 * o['above_0.5']:.0f} / {100 * n['above_0.5']:.0f}",
                     f"{o['best']:.2f} / {n['best']:.2f}", f"{v['spearman']:.2f}", f"{100 * v['top_kept']:.0f}"])
    t += table(["Task", "Diameter (mm)", "Layouts reached", "Median score", "Above 0.5 (%)", "Best score", "Spearman",
                "Best tenth kept (%)"], rows,
               "The 8,192 Sobol layouts, first run / rerun with the finger clearance and, on cylinders, the hold; seeds 0 "
               "to 2. Median and share over the layouts each run reached; Spearman and the best tenth over the layouts "
               "both reached.", num_from=1)
    return t


def lat_words(a):
    """A contact latitude (deg, positive below the equator) as words."""
    return f"{abs(a):.0f} deg {'below' if a > 0 else 'above'}"


def sec_table(N):
    G = N["ground"]
    if not G:
        return "<p>Not run yet.</p>"
    g = (lambda pose, d: G.get((pose, float(d)), {}))
    sph = sorted(d for (pose, d) in G if pose == "tripod")
    cyl = sorted(d for (pose, d) in G if pose == "pinch")
    first_all = next((d for d in sph if g("tripod", d)["both_ok"] >= 0.999), None)
    first_pinch = next((d for d in cyl if g("pinch", d)["both_ok"] > 0), None)
    t = fill(r"""<p>A real object starts on a table, and the grids above were solved without one. Few of their grasps
clear it (Table <<tb>>, <code>scripts/hom_ground.py</code>). Of the tripod grasps on the 40 mm sphere, <<s40>> % clear
it, and the median grasp reaches <<m40>> mm below it. Every tripod grasp clears it from the <<s100>> mm sphere on. No
pinch clears it below the <<fp>> mm cylinder, and no press clears it at any size.</p>""",
             tb=P.TAB[0] + 1, s40=f"{100 * g('tripod', 40)['both_ok']:.0f}", m40=f"{-g('tripod', 40)['touch_median']:.1f}",
             s100=f"{first_all:g}" if first_all else "&ndash;", fp=f"{first_pinch:g}" if first_pinch else "&ndash;")
    t40, p25 = g("tripod", 40), g("pinch", 25)
    if "gr_reached" in t40 and "gr_reached" in p25:
        p_first = next((d for d in cyl if g("pinch", d).get("gr_reached")), None)
        t += fill(r"""<p>The pose toy can keep the table clear as a constraint (<code>fp_sr2_toy.py grid2d
--ground</code>, rows <code>grid2dgr_*</code>). The touch and open poses must then stay within 1 mm of the table, and the
contacts may lie up to 40 deg above the equator. On the 40 mm sphere <<r40>> layouts then reach a tripod, against <<u40>>
without the table. Its contacts lie between <<a40>> and <<b40>> the equator. On cylinders a pinch is reached from
<<pf>> mm on: on <<rp1>> layouts at <<pf>> mm, <<r25>> at 25 mm and <<r50>> at 50 mm. No layout reaches a press, whose
middle finger presses 30 to 45 deg below the equator.</p>""",
                  r40=t40["gr_reached"], u40=t40["layouts"], a40=lat_words(t40["gr_lat"][0]),
                  b40=lat_words(t40["gr_lat"][1]), pf=f"{p_first:g}", rp1=g("pinch", p_first)["gr_reached"],
                  r25=p25["gr_reached"], r50=g("pinch", 50)["gr_reached"])
    rows = []
    for (pose, d), v in sorted(G.items(), key=lambda kv: (kv[0][0] != "tripod", kv[0][0], kv[0][1])):
        lat = v.get("gr_lat")
        rows.append([pose, f"{d:g}", f"{v['touch_median']:+.1f}", f"{100 * v['both_ok']:.0f}",
                     f"{v['layouts_ok']} of {v['layouts']}", str(v["gr_reached"]) if "gr_reached" in v else "&ndash;",
                     f"{lat[0]:+.0f} to {lat[1]:+.0f}" if lat else "&ndash;"])
    t += table(["Pose", "Diameter (mm)", "Table clearance at touch, median (mm)", "Grasps that clear the table (%)",
                "Layouts with a clear grasp", "Layouts reached with the table constraint",
                "Contact latitude with the constraint (deg)"], rows,
               "The re-solved grids' grasps against a table under the object, and the grids solved with the table as a "
               "constraint. A grasp clears the table when every finger sphere stays above it, less 1 mm, at the touch "
               "and open poses. Latitude: 5th to 95th percentile over the grasps, positive below the equator.",
               num_from=1)
    GG = N.get("gr_grid") or {}
    if GG.get("twist") and GG.get("roll") and GG.get("pinpush"):
        a = {k: GG[k]["grid2dff"] for k in GG}
        b = {k: GG[k]["grid2dgr"] for k in GG}
        t += fill(r"""<p>On the GPU, the best tripod layout clear of the table scores <<tw1>> in the twist and <<ro1>> in
the roll, against <<tw0>> and <<ro0>> without the table constraint. The best pinch layout clear of it scores <<pp1>> in the
pin-push, against <<pp0>>. Fewer layouts score well: <<s1>> % of the twist's layouts score above 0.5 with the constraint,
against <<s0>> % without it, and <<q1>> % of the pin-push's, against <<q0>> %.</p>""",
                  tw1=f"{b['twist']['best']:.2f}", ro1=f"{b['roll']['best']:.2f}", tw0=f"{a['twist']['best']:.2f}",
                  ro0=f"{a['roll']['best']:.2f}", pp1=f"{b['pinpush']['best']:.2f}", pp0=f"{a['pinpush']['best']:.2f}",
                  s1=f"{100 * b['twist']['above']:.0f}", s0=f"{100 * a['twist']['above']:.0f}",
                  q1=f"{100 * b['pinpush']['above']:.0f}", q0=f"{100 * a['pinpush']['above']:.0f}")
    C = table_costs(N)
    if C and all(C.values()):
        tw, ro, pp = C["twist"], C["roll"], C["pinpush"]
        small = [d for d in tw if d < 40]
        big = [d for d in tw if d >= 40]
        ro_off = [d for d in ro if abs(ro[d][0] - ro[d][1]) > 0.03]
        pl = [pp[d][0] - pp[d][1] for d in pp]
        t += fill(r"""<p>Searched across sizes with the table constraint (<code>hom_ea.py sweep --src grid2dgr</code>),
the constraint lowers the scores on small objects and not on large ones (Figure <<fg>>). The twist's best layout scores
<<tws>> on the <<ds>> mm
spheres, against <<twu>> without the constraint. From 40 mm on the two differ by <<twb>> or less. On the 25 mm sphere the
thumb's mount sits <<dx>> mm closer to the pair. The roll's best scores differ by 0.03 or less<<rx>>. The pin-push's best
layout scores <<p0>> on the <<d0>> mm cylinder and <<p1>> at <<d1>> mm, <<pl>> below its scores without the
constraint.</p>""",
                  fg=P.FIG[0] + 1, tws=join_and(f"{tw[d][1]:.2f}" for d in small),
                  ds=join_and(f"{d:g}" for d in small),
                  twu=span([tw[d][0] for d in small]), twb=f"{max(abs(tw[d][0] - tw[d][1]) for d in big):.2f}",
                  dx=f"{tw[25.0][2] - tw[25.0][3]:.0f}" if 25.0 in tw else "&ndash;",
                  rx=("" if not ro_off else ", except at " + join_and(
                      f"{d:g} mm ({ro[d][1]:.2f} against {ro[d][0]:.2f})" for d in ro_off)),
                  p0=f"{pp[min(pp)][1]:.2f}", d0=f"{min(pp):g}", p1=f"{pp[max(pp)][1]:.2f}", d1=f"{max(pp):g}",
                  pl=span(pl))
        t += img(MEDIA / "20261030-size_table_compare.png",
                 "The twist, the roll and the pin-push across object sizes, searched with grasps not checked against the "
                 "table (black) and with grasps clear of it (red). Top: the best layout's score. Bottom: the near set's "
                 "centroid in $x_{sep}$. Dotted: the lattice's edge.")
    TB = N.get("table") or {}
    if TB.get("twist") and all(k in TB["twist"] for k in ("grid2dff_post", "grid2dff_table", "grid2dgr_table")):
        rows = []
        for task in ("twist", "roll", "pinpush"):
            v = TB.get(task) or {}
            cell = (lambda k: f"{v[k]['held']} of {v[k]['n']}, {v[k]['median']:.2f}" if k in v else "&ndash;")
            rows.append([NAMES[task].replace("-", "&#8209;"), cell("grid2dff_post"), cell("grid2dff_table"),
                         cell("grid2dgr_post"),
                         cell("grid2dgr_table")])
        full = [k for k in ("twist", "roll", "pinpush") if all(
            x in (TB.get(k) or {}) for x in ("grid2dff_post", "grid2dff_table", "grid2dgr_post", "grid2dgr_table"))]
        if len(full) == 3:
            u = {k: TB[k]["grid2dff_table"] for k in full}
            c = {k: (TB[k]["grid2dgr_post"], TB[k]["grid2dgr_table"]) for k in full}
            fmax = max(u[k]["touch_force"][1] for k in full)
            pp = c["pinpush"][1]
            t += fill(r"""<p>With a physical table under the object until the lift, the 20 best layouts of each grid ran
on the CPU plant (Table <<tb>>, <code>scripts/hom_table_check.py</code>). In the grids without the table constraint, a
finger touched the table before the lift in <<ut>> of the 60 runs, with peak forces of <<f00>> N to <<fm>> MN. Most of
their grasps start inside the table (Table <<tb4>>). Their twist scored a median of <<u_tw>> with the table, against
<<p_tw>> on the post alone. In the film of the best such layout, the index and the middle never press on the sphere
together. No turn is therefore made with all three pads in contact (Film <<fl>>). With the constraint, the twist and the roll
scored the same with the table and without it, <<c_tw>> and <<c_ro>>.</p>
<p>The pin-push's layouts clear of the table scored a median of <<c_pp1>> with the table, against <<c_pp0>> on the post
alone. A finger touched the table before the lift in <<pt>> of the <<pn>> runs, with peak forces of <<f0>> to <<f1>> N.
Of those <<pt>> runs, <<pd>> dropped the cylinder. The <<pf>> runs without table contact all held it, with a median
score of <<pm>>. The constraint allows 1 mm of penetration at the touch and open poses and does not check the squeeze
that follows. After the lift, the cylinder of the best such layout swung down about the pinch (Film <<fl2>>).</p>""",
                      tb=P.TAB[0] + 1, tb4=P.TAB[0], ut=sum(u[k]["touch"] for k in full), fm=f"{fmax / 1e6:.1f}",
                      f00=f"{min(u[k]['touch_force'][0] for k in full):.0f}",
                      u_tw=f"{u['twist']['median']:.2f}", p_tw=f"{TB['twist']['grid2dff_post']['median']:.2f}",
                      c_tw=f"{c['twist'][1]['median']:.2f}", c_ro=f"{c['roll'][1]['median']:.2f}",
                      c_pp1=f"{pp['median']:.2f}", c_pp0=f"{c['pinpush'][0]['median']:.2f}", pt=pp["touch"],
                      pn=pp["n"], f0=f"{pp['touch_force'][0]:.0f}", f1=f"{pp['touch_force'][1]:,.0f}",
                      pd=pp["touch_dropped"], pf=pp["free"], pm=f"{pp['free_median']:.2f}" if pp["free_median"] else "&ndash;",
                      fl=P.FILM[0] + 1, fl2=P.FILM[0] + 2)
        t += table(["Task", "Unconstrained grid, post", "Unconstrained grid, table", "Table-constrained grid, post",
                    "Table-constrained grid, table"], rows,
                   "The 20 best layouts of each grid at the default object, best grasp, seed 0, CPU plant with every "
                   "contact: runs held and the median score, on the 3 mm post alone and with a table under the object "
                   "until the lift.", num_from=1)

    def cap_t(i, task):
        a, b = i.get("grid2dff", {}), i.get("grid2dgr", {})
        return (f"The {NAMES[task].lower()} with a table under the object until the lift, CPU plant, seed 0. Left: the "
                f"best layout of the grid without the table constraint, ({a['layout'][0]:.0f}, {a['layout'][1]:.0f}) "
                f"mm, score {a['score']:.2f} against {a['gpu']:.2f} on the GPU without a table. Right: the best layout "
                f"of the grid with it, ({b['layout'][0]:.0f}, {b['layout'][1]:.0f}) mm, {b['score']:.2f} against "
                f"{b['gpu']:.2f}. <code>scripts/hom_table_check.py film</code>.")
    for task in ("twist", "pinpush"):
        t += v4b_film(f"table_{task}", lambda i, task=task: cap_t(i, task))
    return t + sec_table_followup(N)


def sec_table_followup(N):
    """The pin-push's table failure traced, and the 25 mm pinch grid solved with no penetration of the table and with the
    squeezed pose (rows grid2dgn_*, grid2dgs_*; logs/20261030-hom_ff/table_study2.sh)."""
    tr = ROWS / "cpu_best" / "table_trace_pinpush_d25_grid2dgr.json"
    tp = ROWS / "cpu_best" / "table_trace_pinpush_d25_grid2dgr_post.json"
    PP = (N.get("table") or {}).get("pinpush") or {}
    GG = (N.get("gr_grid") or {}).get("pinpush") or {}
    srcs = ("grid2dgr", "grid2dgn", "grid2dgs")
    tc = ROWS / "cpu_best" / "table_trace_pinpush_d25_grid2dgs.json"
    if not (tr.exists() and tp.exists() and tc.exists() and all(f"{g}_table" in PP and g in GG for g in srcs)):
        return ""
    C_ = json.loads(tc.read_text())
    A, B = json.loads(tr.read_text()), json.loads(tp.read_text())
    fc = [v for v in A["table_force_N"]["close"].values()]
    t = fill(r"""<p>A trace of the best of those layouts shows the cause (<code>hom_table_check.py trace</code>). Its three
fingers' DIP links, cassettes and caps rested on the table from the first control step, up to <<dp>> mm into it, which the
1 mm tolerance allows. The approach then pressed them on the table with <<f0>> to <<f1>> N each, and no pad met the
cylinder before the lift at <<tl>> s. The pads met it at <<to>> s, and by <<td>> s the cylinder had fallen 10 mm. On the
post alone, the same grasp's pads met the cylinder at <<tp>> s and sank <<s0>> to <<s1>> mm into it before the lift.</p>""",
             dp=f"{-min(A['table_deepest_mm'].values()):.2f}", f0=f"{min(fc):.0f}", f1=f"{max(fc):.0f}",
             tl=f"{A['lift_s']:.1f}", to=f"{min(A['object_first_s'].values()):.2f}", td=f"{A['object_10mm_down_s']:.2f}",
             tp=f"{min(B['object_first_s'].values()):.2f}",
             s0=f"{-max(B['object_deepest_before_lift_mm'].values()):.2f}",
             s1=f"{-min(B['object_deepest_before_lift_mm'].values()):.2f}")
    r_, n_, s_ = (GG[g] for g in srcs)
    pr, pn, ps = (PP[f"{g}_table"] for g in srcs)
    t += fill(r"""<p>The 25 mm pinch grid was therefore solved again with no penetration of the table (rows
<code>grid2dgn_*</code>). It was solved once more with the squeezed pose clear of the table as well, the pads 0.6 mm into
the cylinder (<code>grid2dgs_*</code>). <<rn>> and <<rs>> layouts reach the pinch, against <<rr>> with the tolerance. Their best
layouts score <<gn>> and <<gs>> on the GPU, against <<gr>>. With the table under the cylinder, the 20 best layouts of each
grid held it in <<hn>> and <<hs>> of 20 runs. Their median scores were <<mn>> and <<ms>> (Table <<tb>>). No finger touched the
table before the lift in any of those runs, against <<cr>> of 20 with the tolerance. On the post alone the same layouts
held the cylinder in <<hpn>> and <<hps>> of 20 runs, so the table still costs the pin-push. The remaining drops come later.
In the best layout of the grid with the squeezed pose, the pads met the cylinder at <<ts>> s, before the lift. The
cylinder fell 10 mm at <<tds>> s, in the turn's down stroke (Film <<fm>>). The cause of these later drops is not known.</p>""",
              rn=n_["reached"], rs=s_["reached"], rr=r_["reached"], gn=f"{n_['best']:.2f}", gs=f"{s_['best']:.2f}",
              gr=f"{r_['best']:.2f}", hn=pn["held"], hs=ps["held"], mn=f"{pn['median']:.2f}", ms=f"{ps['median']:.2f}",
              tb=P.TAB[0] + 1, cr=pr["touch"], hpn=PP["grid2dgn_post"]["held"], hps=PP["grid2dgs_post"]["held"],
              ts=f"{min(C_['object_first_s'].values()):.2f}", tds=f"{C_['object_10mm_down_s']:.2f}",
              fm=P.FILM[0] + 1)
    names = {"grid2dgr": "1 mm allowed", "grid2dgn": "No penetration", "grid2dgs": "No penetration, squeeze"}
    rows = []
    for g in srcs:
        po, tb_ = PP[f"{g}_post"], PP[f"{g}_table"]
        rows.append([names[g], str(GG[g]["reached"]), f"{GG[g]['best']:.2f}",
                     f"{po['held']} of {po['n']}, {po['median']:.2f}", f"{tb_['held']} of {tb_['n']}, {tb_['median']:.2f}",
                     f"{tb_['touch']} ({tb_['touch_dropped']} dropped)"])
    t += table(["Constraint", "Layouts reached", "GPU best", "Post: held, median", "Table: held, median",
                "Table contact (dropped)"], rows,
               "The pin-push on the 25 mm cylinder with three versions of the pose toy's table constraint (rows "
               "<code>grid2dgr</code>, <code>grid2dgn</code>, <code>grid2dgs</code>). CPU plant with every contact, "
               "seed 0, the 20 best layouts of each grid by the GPU's score, each with its best grasp; runs held and "
               "median score on the post and with the table, and runs with a finger on the table before the lift.",
               num_from=1)

    def cap(i):
        parts = [f"{side}: {names[g].lower().replace('1 mm allowed', '1 mm of penetration allowed')}, layout "
                 f"({i[g]['layout'][0]:.0f}, {i[g]['layout'][1]:.0f}) mm, score "
                 f"{i[g]['score']:.2f} against {i[g]['gpu']:.2f} on the GPU." for side, g in
                 zip(("Left", "Middle", "Right"), srcs) if g in i]
        return ("The pin-push with a table under the cylinder until the lift, CPU plant, seed 0, on the best layout of "
                "each grid. " + " ".join(parts) + " <code>scripts/hom_table_check.py film</code>.")
    return t + v4b_film("table_pinpush_squeeze", cap, "20261031")


def sec_limits(N):
    return fill(r"""<ul>
<li>The GPU evaluator keeps its collision filter, so contact between two fingers during a task is missed there. The best
layouts are re-run on the CPU plant, at seed 0 only.</li>
<li>Each live search ran once per task and object, from one random start. On the full grids, 3 % or fewer of such runs
missed the best by more than 0.05.</li>
<li>Two children of one generation can land on the same layout, which is then run twice. Such repeats made up <<rep>> %
of the live searches' rollouts and count against the 6,000.</li>
<li>The lattice is the symmetric tripod in two coordinates, and its $y_{sep}$ ends at 170 mm. The press's best layouts lie
within 15 mm of that edge at most sizes. The six-coordinate layouts are the Sobol study's.</li>
<li>In the table study the GPU plant keeps its 3 mm post. The grasps clear a table, but a table stands under the object
only in the CPU check, at the default objects.</li>
<li>Every run uses the work plant (4 N&#8201;m/rad, 1 N&#8201;m), 4.4 times stronger than the hand's SCS0009 servos, and
the scenes cached on 2026-10-30, whose pads have a modulus of 10 MPa. The scenes have used 2.5 MPa, fitted to the v3
tips' presses, since that evening (<a href="<<contact>>">contact model page</a>). The study was rerun on the SCS0009
plant (0.72 N&#8201;m/rad, 0.18 N&#8201;m) at 2.5 MPa. There the near sets' slopes in $x_{sep}$ are 1.41, 0.57, 1.20
and 1.09 mm per mm of diameter for the twist, roll, pin-push and press, against 0.98, 0.59, 0.98 and 1.11 here. The
slopes in $y_{sep}$ do not persist: the press's changes sign, from 0.88 to &#8722;0.43, and the twist's falls from
&#8722;0.73 to &#8722;0.16. On the work plant at 2.5 MPa, 68 % of the 380 layouts checked on the CPU score within 0.05
of their 10 MPa scores. The servo, the pad modulus and the scene friction are compared on the
<a href="20261102-hom_plant_modulus.html">plant and modulus page</a>.</li>
</ul>""", contact=CONTACT, rep=f"{100 * (repeat_share() or 0):.1f}")


def sec_future(N):
    return r"""<ul>
<li>Finger-to-finger contact on the GPU at small cost: capsules or a sphere tree fitted to each v4b part, colliding
across fingers only. Time a 2,016-world twist batch against 31 ms per rollout, and rerun the 40 best twist layouts. The
claim that the filter changes the ranking only on the smallest spheres fails if their order differs from the CPU's. The
k-means attempt is <code>scripts/hom_proxy_check.py</code>.</li>
<li>Repeat each live search from three random starts and report the spread of the best layout's position per size. The
slopes of Section SEC[trends] hold if each lies within that spread of its fit.</li>
<li>Find why the pin-push's table-clear layouts drop the cylinder in 5 to 8 of 20 runs with the table, though no finger
touches it. On the post alone they drop it in 0 or 1 of 20. Compare the cylinder's angle and the pads' forces at the end of the squeeze with
and without the table (<code>hom_table_check.py trace pinpush 25 grid2dgs</code> on the dropping layouts). The account
holds if the dropping runs leave the squeeze with the cylinder turned or the pads' forces lower than on the post.</li>
<li>Re-solve the tripod grids and the table-clear searches across sizes with no penetration of the table
(<code>fp_sr2_toy.py grid2d --ground --ground-tol 0</code>), and repeat the table check of the twist and the roll.</li>
<li>A table in the GPU plant, a box under the object until the lift, to score grasps from a table at grid scale.</li>
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
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|MN\b|N\b|ms\b|s\b)", "\\1&nbsp;\\2", s_))  # a number keeps its unit
    sections = [(sid, hd, unit(sub(body))) for sid, hd, body in sections]
    lede = unit(lede)
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


def lede_and_summary(N):
    """The page's lede and its docs-summary from the numbers."""
    R, Tn = N["rerun"], N["trends"]
    plain = ("Best layouts of the SR2 hand under HOM control across object sizes, with the fingers kept 2 mm apart and "
             "with a table under the object.")
    if not (R["rerun"] and R["timing"] and R["ea"] and Tn):
        return plain, plain
    St = {task: trend_stats(Tn, task) for task in TASKS}
    if not all(St.values()):
        return plain, plain
    sx = [St[k]["slope_x"] for k in TASKS]
    n, ok, _bad = cpu_agreement(St)
    G = N["ground"] or {}
    cyl = sorted(d for (pose, d) in G if pose == "pinch")
    p_first = next((d for d in cyl if G[("pinch", d)].get("gr_reached")), None)
    table_s = (f" With a table under the object, no press and no pinch of a cylinder under {p_first:g} mm clears it."
               if p_first else "")
    C = table_costs(N)
    if C and all(C.values()):
        tw_loss = max(C["twist"][d][0] - C["twist"][d][1] for d in C["twist"] if d < 40)
        table_s += (f" Grasps kept clear of the table cost the twist up to {tw_loss:.2f} on spheres under 40 mm and the "
                    f"pin-push {span([a - b for a, b, _x, _y in C['pinpush'].values()])}.")
    TB = N.get("table") or {}
    tw_c, ro_c = (TB.get(k, {}).get("grid2dgr_table") for k in ("twist", "roll"))
    pp_c = TB.get("pinpush", {}).get("grid2dgr_table")
    if tw_c and ro_c and pp_c and pp_c["touch"]:
        table_s += (" On the CPU plant with a table, they kept the twist's and the roll's scores on the 40 mm sphere. The "
                    f"pin-push dropped the cylinder in {pp_c['touch_dropped']} of the {pp_c['touch']} runs in which a "
                    "finger pressed on the table.")
        nn = [TB["pinpush"].get(f"{g}_table") for g in ("grid2dgn", "grid2dgs")]
        if all(nn) and not any(x["touch"] for x in nn):
            table_s += (" With no penetration of the table allowed, no finger touched it, and the pin-push held the "
                        f"cylinder in {span([x['held'] for x in nn], '{:d}')} of 20 runs, against {pp_c['held']}.")
    lede = fill("""Over 19 object sizes, the best layout of the SR2 hand moves with the object. In all four HOM tasks the
thumb's mount moves away from the finger pair by <<sx>> mm per millimetre of diameter. The spacing of the index and the
middle narrows on spheres and widens on cylinders. Each size was searched with Yang et al.'s evolutionary algorithm in
6,000 rollouts, which came within 0.05 of a full grid's best in every test run. The CPU plant with every contact gave the
best layout's score within <<tol>> at <<ok>> of <<n>> sizes.<<tb>>""", sx=span(sx), tol=CPU_TOL, ok=ok, n=n, tb=table_s)
    summary = fill("""Across 19 object sizes the SR2 hand's best layout under HOM control moves with the object: the
thumb's mount moves out <<sx>> mm per mm of diameter in all four tasks, found by an evolutionary search in 6,000 rollouts
per size and confirmed on the CPU plant at <<ok>> of <<n>> sizes.<<tb>>""", sx=span(sx, "{:.1f}"), ok=ok, n=n,
                   tb=(f" With a table under the object, no press and no pinch of a cylinder under {p_first:g} mm "
                       "clears it." if p_first else ""))
    return lede, " ".join(summary.split())


def main():
    N = {"rerun": rerun_numbers(), "trends": trend_numbers("grid2dff"), "trends_gr": trend_numbers("grid2dgr"),
         "ground": ground_numbers(), "gr_grid": gr_grid_numbers(), "table": table_numbers(), "sobol": sobol_numbers()}
    sections = [
        ("terms", "Terms and metrics", sec_terms()),
        ("trends", "Best layouts across object sizes", sec_trends(N)),
        ("search", "Reliability of the evolutionary search", sec_search(N)),
        ("fix", "Finger clearance in the pose toy and its cost on the GPU", sec_fix(N)),
        ("rerun", "Grids at the default objects re-solved with the finger clearance", sec_rerun(N)),
        ("sobol", "The Sobol layouts re-run with the finger clearance", sec_sobol(N)),
        ("table", "Grasps clear of a table under the object", sec_table(N)),
        ("limits", "Limitations", sec_limits(N)),
        ("future", "Future work", sec_future(N)),
        ("refs", "References and files", "<ol class=\"refs\">" + "".join(f"<li>{r[r.index(']') + 2:]}</li>"
                                                                         for r in REFS) + "</ol>" + sec_files()),
    ]
    lede, summary = lede_and_summary(N)
    byline = ("Simulation, 2026-10-30 and 31. Hand: the CAD-model SR2 hand with the v4b fingertip "
              "(<code>SCALE_HAND=cad_v2</code>); HOM on MuJoCo-Warp and on the CPU plant. Follows "
              f"<a href=\"{DESIGN}\">the design page</a>, Section 9. Builder: <code>scripts/hom_trends_page.py</code>.")
    assemble(sections, "SR2 layouts across object sizes", "Best SR2 hand layouts across object sizes under HOM control",
             "HOM layout study, 2026-10-30", lede, byline, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
