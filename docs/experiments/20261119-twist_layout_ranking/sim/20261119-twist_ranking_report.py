#!/usr/bin/env python3
"""Summary of the twist layout ranking (sim/20261119-twist_ranking_search.py's rows), and the proposal of the layouts
for the hand (hand-4a, 2026-11-19).

Scores. A run's score is its tracking, 1 - RMS orientation error / RMS commanded rotation, at least 0, and 0 when the
sphere dropped or the run failed (the trials' score on the hand, sr2hand.control.trial_score). A layout's score under a
model is its best grasp's mean over seeds 0-2, among the grasps run at all three seeds (stage A ran every grasp at seed
0, stage B the three best at seeds 1-2); its validation score is the same grasp's mean over the fresh seeds 10-12 (stage C,
the 60 best layouts per model).

Comparisons (Spearman's rho over the layouts both scores have):
  point vs pads            the two contact models of the calibrated plant
  seed 0 vs seeds 0-2      what one seed per grasp ranks
  vs grid sv25r            the GPU grid of the landscapes (rows grid2dff_v4b_sv25r: the bare ball, the SCS0009 plant
                           at 0.72 N m/rad, impratio 100; best of its 16 film-grid grasps, hom_landscape.track_score)
  vs stage D's plants      stiff pads, friction 0.6, the servos at 5 V, on the 150 best layouts
and the drop from seeds 0-2 to seeds 10-12 on the 60 best (the winner's curse of picking on the same seeds).

Search. Yang et al.'s elitist EA (hom_landscape.ea_search, 20 runs) and uniform random sampling, replayed on stage A's
scores (a layout's fitness is its best grasp at seed 0, costing 16 runs): runs until a layout within 0.05 of the grid's
best is found, median and quartiles.

Proposal for the hand. Layouts that fit the gantries' travel boxes (sr2_layouts.layout_report), whose index and middle
stay at least GAP_MM apart in every simulated run of the chosen grasp (on 2026-11-17 the hand's fingers came about 10 mm
closer than the simulation's), and that held the sphere in all runs of that grasp under both models. Among them, ranked
by the mean of the two models' scores: the four best, then six at evenly spaced ranks down to the lowest scoring layout
that still held, so the test spans the predicted range; plus the layout already run on the hand 20 times on 2026-11-17,
(70, 110) grasp 0 ((105, 70) grasp 13 is left out: at 7.5 V its index and middle came within 1.9 mm). Each with its
grasp of the best mean over both models.

    .venv/bin/python docs/experiments/20261119-twist_layout_ranking/sim/20261119-twist_ranking_report.py
Writes data/20261119-twist_ranking_summary.json and data/20261119-trial_proposal.json.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
LOGS = ROOT / "logs/20261119-twist_layout_ranking"
FPR = ROOT / "docs/experiments/20261029-foundational_pose_review/rows"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import _paths  # noqa: E402,F401

MODELS = ("point", "pads")
PLANTS_D = ("pads_stiff", "point_mu0.6", "pads_mu0.6", "point_5v")
GAP_MM = 15.0
NEAR = 0.05
ANCHORS = ((162, 0),)                              # (70, 110) g0, 20 trials on the hand on 11-17 (g13 left out: its fingers came within 1.9 mm at 7.5 V)


def rows(stage):
    p = LOGS / f"rows_{stage}.jsonl"
    out = []
    if p.exists():
        for x in p.read_text().splitlines():
            if x.strip():
                try:
                    out.append(json.loads(x))
                except json.JSONDecodeError:
                    pass
    return out


def score(r):
    return 0.0 if "error" in r or not r.get("held") else float(r.get("track") or 0.0)


def table(rs, model, seeds):
    """{call: {k: {seed: row}}} of one model's runs at the given seeds."""
    t = defaultdict(lambda: defaultdict(dict))
    for r in rs:
        if r["model"] == model and int(r["seed"]) in seeds:
            t[int(r["call"])][int(r["k"])][int(r["seed"])] = r
    return t


def layout_scores(rs, model, seeds=(0, 1, 2)):
    """{call: (k, mean score)} the best grasp among those run at all the seeds."""
    out = {}
    for c, g in table(rs, model, seeds).items():
        full = {k: np.mean([score(v[s]) for s in seeds]) for k, v in g.items() if len(v) == len(seeds)}
        if full:
            k = max(full, key=full.get)
            out[c] = (k, float(full[k]))
    return out


def rho(a, b):
    ks = sorted(set(a) & set(b))
    if len(ks) < 5:
        return None, len(ks)
    r = spearmanr([a[k] for k in ks], [b[k] for k in ks]).correlation
    return round(float(r), 3), len(ks)


def grid_sv25r():
    """{call: best grasp's mean track_score over seeds 0-2} of the GPU grid of the landscapes."""
    import hom_landscape as L
    import hom_rows
    sc = defaultdict(lambda: defaultdict(list))
    p = FPR / "grid2dff_v4b_sv25r" / "twist_d40.jsonl"
    for x in p.read_text().splitlines():
        if x.strip():
            r = hom_rows.rescale_metrics(json.loads(x))
            if r.get("status", "ok") == "ok":
                sc[int(r["call"])][int(r.get("grasp", 0))].append(L.track_score("twist", r))
    return {c: max(float(np.mean(v)) for v in g.values()) for c, g in sc.items()}


def search_replay(a_scores, xy):
    """Rollouts the EA and random sampling need to find a layout within NEAR of the best, on stage A's seed-0 scores."""
    import hom_landscape as L
    look = {xy[c]: (v, 16) for c, v in a_scores.items()}
    best = max(v for v, _ in look.values())
    xs = sorted({k[0] for k in look})
    ys = sorted({k[1] for k in look})
    full = {(x, y): look.get((x, y), (0.0, 16)) for x in xs for y in ys}   # the lattice's layouts without grasps: 0
    out = {}
    for name, rnd in (("ea", False), ("random", True)):
        need = []
        for s in range(20):
            hist, seen = L.ea_search(full, np.random.default_rng(s), xs, ys, random=rnd, budget=len(full) * 16)
            hit = next((cost for cost, k in hist if seen[k] >= best - NEAR), None)
            need.append(hit if hit is not None else np.nan)
        need = np.array(need, float)
        out[name] = {"runs_to_within_0.05": {"median": float(np.nanmedian(need)), "q25": float(np.nanpercentile(need, 25)),
                                             "q75": float(np.nanpercentile(need, 75)), "found": int(np.isfinite(need).sum()),
                                             "of": len(need)},
                     "grid_runs": int(sum(n for _, n in full.values()))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUDY / "data/20261119-twist_ranking_summary.json"))
    ap.add_argument("--proposal", default=str(STUDY / "data/20261119-trial_proposal.json"))
    a = ap.parse_args()
    from sr2hand.robot import sr2_layouts as SL
    RA, RB, RC, RD = rows("A"), rows("B"), rows("C"), rows("D")
    R = RA + RB
    xy = {int(r["call"]): tuple(r["xy"]) for r in RA}
    S = {}
    for m in MODELS:
        s012 = layout_scores(R, m)
        s0 = layout_scores(RA, m, seeds=(0,))
        fresh = {}
        for c, (k, v) in s012.items():
            t = table(RC, m, (10, 11, 12)).get(c, {}).get(k, {})
            if len(t) == 3:
                fresh[c] = float(np.mean([score(t[s]) for s in (10, 11, 12)]))
        S[m] = {"s012": s012, "s0": s0, "fresh": fresh}
    out = {"what": __doc__.split("\n\n")[0].replace("\n", " "),
           "script": "docs/experiments/20261119-twist_layout_ranking/sim/20261119-twist_ranking_report.py",
           "runs": {"A": len(RA), "B": len(RB), "C": len(RC), "D": len(RD),
                    "errors": {st: sum("error" in r for r in rr) for st, rr in (("A", RA), ("B", RB), ("C", RC), ("D", RD))}},
           "layouts": len(xy)}
    # rank comparisons
    cmp = {}
    v = {m: {c: s for c, (k, s) in S[m]["s012"].items()} for m in MODELS}
    cmp["point vs pads (seeds 0-2)"] = rho(v["point"], v["pads"])
    for m in MODELS:
        cmp[f"{m}: seed 0 vs seeds 0-2"] = rho({c: s for c, (k, s) in S[m]["s0"].items()}, v[m])
    try:
        g = grid_sv25r()
        for m in MODELS:
            cmp[f"{m} vs grid sv25r"] = rho(v[m], g)
    except Exception as e:                                  # noqa: BLE001
        cmp["grid sv25r"] = f"not read: {e!r}"[:200]
    both = {c: (v["point"][c] + v["pads"][c]) / 2 for c in set(v["point"]) & set(v["pads"])}
    dc = {int(r["call"]) for r in RD}
    if dc:          # the two models over stage D's layouts alone, the plants' baseline over the same narrow range
        cmp["point vs pads (stage D layouts)"] = rho({c: v["point"][c] for c in dc if c in v["point"]}, v["pads"])
        out["stage_d_spread"] = {"layouts": len(dc), "mean": round(float(np.mean([both[c] for c in dc if c in both])), 4),
                                 "sd": round(float(np.std([both[c] for c in dc if c in both])), 4)}
    for p in PLANTS_D:
        sp = layout_scores(RD, p)
        if sp:
            cmp[f"{p} vs mean of point and pads (stage D layouts)"] = rho({c: s for c, (k, s) in sp.items()}, both)
    out["rho"] = {k: {"rho": x[0], "n": x[1]} if isinstance(x, tuple) else x for k, x in cmp.items()}
    # winner's curse on the validated layouts
    wc = {}
    for m in MODELS:
        ks = sorted(S[m]["fresh"])
        if ks:
            d = [S[m]["fresh"][c] - v[m][c] for c in ks]
            wc[m] = {"layouts": len(ks), "seeds012_mean": round(float(np.mean([v[m][c] for c in ks])), 4),
                     "seeds101112_mean": round(float(np.mean([S[m]["fresh"][c] for c in ks])), 4),
                     "change_mean": round(float(np.mean(d)), 4), "change_sd": round(float(np.std(d)), 4),
                     "rho_seeds012_vs_fresh": rho({c: v[m][c] for c in ks}, S[m]["fresh"])[0]}
    out["fresh_seeds"] = wc
    # per layout
    lay = []
    for c in sorted(xy):
        rec = {"call": c, "xy": list(xy[c])}
        rep = SL.layout_report(SL.sym6(*xy[c]))
        rec["fits"], rec["min_margin_mm"] = bool(rep["fits"]), rep["min_margin_mm"]
        for m in MODELS:
            if c in S[m]["s012"]:
                k, s = S[m]["s012"][c]
                runs = [r for r in R if r["model"] == m and int(r["call"]) == c and int(r["k"]) == k]
                rec[m] = {"k": k, "score": round(s, 4), "fresh": round(S[m]["fresh"][c], 4) if c in S[m]["fresh"] else None,
                          "twist_end": round(float(np.mean([r.get("twist_end") or 0 for r in runs])), 1),
                          "rise_max": round(float(np.mean([r.get("rise_max") or 0 for r in runs])), 2),
                          "held": sum(bool(r.get("held")) for r in runs), "runs": len(runs)}
        lay.append(rec)
    out["per_layout"] = lay
    # grasps by the mean over both models and every seed run, with the finger gaps of those runs
    pool = defaultdict(lambda: defaultdict(list))
    for r in R + RC:
        if r["model"] in MODELS:
            pool[int(r["call"])][int(r["k"])].append(r)
    cand = []
    for c, g in pool.items():
        best = None
        for k, rr in g.items():
            ms = {m for m in MODELS for r in rr if r["model"] == m and int(r["seed"]) in (0, 1, 2)}
            if ms != set(MODELS) or sum(int(r["seed"]) in (0, 1, 2) for r in rr) < 6:
                continue
            sc = float(np.mean([score(r) for r in rr if int(r["seed"]) in (0, 1, 2)]))
            gaps = [((r.get("gap_min_mm") or {}).get("joints") or {}).get("im", [None])[0] for r in rr]
            gaps = [x for x in gaps if x is not None]
            held = all(bool(r.get("held")) and "error" not in r for r in rr)
            if best is None or sc > best["score"]:
                best = {"call": c, "xy": list(xy[c]), "k": k, "score": round(sc, 4), "held_all": held,
                        "im_gap_min_mm": round(min(gaps), 1) if gaps else None, "runs": len(rr),
                        "twist_end": round(float(np.mean([r.get("twist_end") or 0 for r in rr])), 1)}
        if best:
            rep = SL.layout_report(SL.sym6(*xy[c]))
            best["fits"] = bool(rep["fits"])
            cand.append(best)
    out["candidates"] = sorted(cand, key=lambda b: -b["score"])   # each layout's best grasp over both models
    ok = [b for b in cand if b["fits"] and b["held_all"] and (b["im_gap_min_mm"] or 0) >= GAP_MM]
    ok.sort(key=lambda b: -b["score"])
    pick = ok[:4]
    rest = ok[4:]
    if rest:
        idx = np.unique(np.round(np.linspace(0, len(rest) - 1, 6)).astype(int))
        pick += [rest[i] for i in idx]
    anchors = []
    for c, k in ANCHORS:
        rr = pool.get(c, {}).get(k, [])
        anchors.append({"call": c, "xy": list(xy.get(c, (None, None))), "k": k,
                        "score": round(float(np.mean([score(r) for r in rr])), 4) if rr else None, "runs": len(rr)})
    rank = {id(b): i + 1 for i, b in enumerate(ok)}
    prop = {"what": "Layouts proposed for the hand on 2026-11-20, from the twist layout ranking: the four best and six at "
                    "evenly spaced ranks of the deployable layouts (fit the travel boxes, index-middle gap at least "
                    f"{GAP_MM:g} mm in every simulated run, held in every run under both models), and the two anchors run "
                    "on the hand before. score = mean tracking over both contact models and seeds 0-2.",
            "deployable": len(ok), "of": len(cand),
            "excluded": {"do not fit the travel boxes": sum(not b["fits"] for b in cand),
                         "sphere dropped or HOM failed in a run": sum(not b["held_all"] for b in cand),
                         f"index-middle gap under {GAP_MM:g} mm": sum((b["im_gap_min_mm"] or 0) < GAP_MM for b in cand)},
            "pick":[{**b, "rank_among_deployable": rank[id(b)]} for b in pick], "anchors": anchors}
    out["search"] = search_replay({c: s for c, (k, s) in S["point"]["s0"].items()}, xy) if S["point"]["s0"] else None
    out["top10"] = {m: [{"call": c, "xy": list(xy[c]), "k": k, "score": round(s, 4),
                         "fresh": round(S[m]["fresh"][c], 4) if c in S[m]["fresh"] else None}
                        for c, (k, s) in sorted(S[m]["s012"].items(), key=lambda t: -t[1][1])[:10]] for m in MODELS}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    Path(a.proposal).write_text(json.dumps(prop, indent=1) + "\n")
    print(json.dumps({"runs": out["runs"], "rho": out["rho"], "fresh_seeds": out["fresh_seeds"], "search": out["search"]},
                     indent=1))
    for b in prop["pick"]:
        print(f"  #{b['rank_among_deployable']:3d} call {b['call']:3d} {tuple(b['xy'])} g{b['k']:2d} score {b['score']:.3f} "
              f"twist {b['twist_end']:5.1f} gap {b['im_gap_min_mm']} mm")
    print("anchors", anchors)
    print("wrote", a.out, a.proposal)


if __name__ == "__main__":
    main()
