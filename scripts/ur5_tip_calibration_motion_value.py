#!/usr/bin/env python3
"""What each motion of the UR5 calibration suite adds on the four-magnet v3 tips (owner, 2026-10-29: the indenter plate
waits "until we have 100% finalized all the motions we need for effective calibration"), from v3 ID 1's and ID 2's
sessions on the low sphere, the tall indenter and the rod.

Every estimate is the direction GP of scripts/ur5_tip_calibration_hard_extrapolation.py (fit_gp_direction: from the
direction of the board's change, d/|d|, to the contact point p and to f/|d|), fitted on the holds in the output folders
of scripts/ur5_tip_calibration_analysis.py (holds.npz; f_n over 0.05 N and a finite board change, point and force). A
score is a median over the test holds (HX.stats): loc_mm, the location error across the contact normal (mm); force_rel_err,
| |f_est| - |f| | / |f|; force_rel_bias, the median of (|f_est| - |f|) / |f|; angle_deg, the angle between the estimated
and the measured force (deg); for slides also shear_err_N, the norm of the difference between the estimated and the
measured shear, the force's part across the contact normal (N). Cross-validation is five folds grouped by action
(HX.folds) over the set named; "A -> B" fits once on A and predicts B. Action ids carry their fixture's letter (S:, T:, R:).

  fixtures  presses: each fixture's own five folds; the three fixtures pooled in five folds, scored by fixture; each
            fixture predicted from the other two; the tall indenter and the rod predicted from the sphere alone.
  slides    each fixture's slides predicted from every press; from every press and the other fixtures' slides; and in
            five folds over its slides with every press and the other slides in training.
  levels    the pooled presses' holds at 0.5 N and under, or over 3.5 N, left out of training and predicted; against
            the pooled five folds on the same holds.
  count     the pooled presses' five folds with a random 25, 50 or 75 % of each fold's training actions (seeds 0 to 2).
  tip_end   the presses on the tip's end (region tip: ID 2's top-edge pass on the sphere, ID 1's first 8 of the tall top
            edge) predicted from every other press, and with the other end presses in training (one action left out).
  transfer  one tip's presses and slides predict the other tip's presses and slides; then with k of the other tip's
            press actions added (k 10, 25, 50, 100; seeds 0 to 2), scored on its remaining presses.
  shared    the six points that the tall stage presses at the levels of six sphere grid presses (QUEUE R29: tall-shared-728,
            -730, -719 on the face, -581, -576, -640 on the edges, against grid-029, -009, -013, -032, -051, -034): per
            pair and level, the distance between the registered contact points, the angle between the board's changes,
            the ratio of their gains (|d_board| / f_n, tall over sphere), the shear per newton of each (the force's part
            across the contact normal over f_n), the angle between the tool's orientations, and the direction GP fitted
            on the sphere's presses without the six grid presses, scored on both.
  time      robot time per action family from the lab's session folders (events.jsonl, each action's start to end);
            presses also by their id's family (the part before the first "-") with their largest level.

    ~/.claude/bin/resguard.sh run --mem 8G --cpu 500 -- env NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4 \\
        .venv/bin/python scripts/ur5_tip_calibration_motion_value.py [--parts fixtures,slides,...] [--out FILE]

Writes docs/experiments/20261027-v3_tip_calibration_and_crosstalk/data/20261029-motion_value.json; a run of some parts keeps
the others. Measured on 2026-10-29 (4 threads): fixtures, levels, count, tip_end and time 57 s and 0.8 GB; slides and
transfer 25 min and 6.3 GB at the peak (fits of up to 4,800 holds).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration_cb1r_checks as K  # noqa: E402
import ur5_tip_calibration_hard_extrapolation as HX  # noqa: E402

LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
OUT = ROOT / "docs/experiments/20261027-v3_tip_calibration_and_crosstalk/data/20261029-motion_value.json"
FIXTURES = ("sphere", "tall", "rod")
TIPS = {   # the analysis folders (scripts/ur5_tip_calibration_analysis.py) and the lab's session folders
    "v3 ID 1": dict(holds=dict(sphere="logs/20261028-spherev3id1_twists", tall="logs/20261029-tallv3id1",
                               rod="logs/20261028-rodv3id1"),
                    sessions=dict(sphere="20261027-225213-spherev3id1", tall="20261028-114908-tallv3id1",
                                  rod="20261028-102701-rodv3id1")),
    "v3 ID 2": dict(holds=dict(sphere="logs/20261029-spherev3id2", tall="logs/20261029-tallv3id2",
                               rod="logs/20261029-rodv3id2"),
                    sessions=dict(sphere="20261028-184415-spherev3id2", tall="20261028-200508-tallv3id2",
                                  rod="20261029-031559-rodv3id2")),
}
FRACTIONS = (0.25, 0.5, 0.75)
SEEDS = (0, 1, 2)
KS = (10, 25, 50, 100)
LOW, HIGH = 0.5, 3.5


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(folder, fixture):
    H = K.load(ROOT / folder)
    m = (np.isfinite(H["d_board"]).all(1) & np.isfinite(H["p"]).all(1) & np.isfinite(H["f"]).all(1)
         & np.isfinite(H["f_n"]) & (H["f_n"] > 0.05))
    H = K.sub(H, m)
    H["id"] = np.array([f"{fixture[0].upper()}:{i}" for i in H["id"].astype(str)])
    H["fixture"] = np.array([fixture] * len(H["id"]))
    return H


def fit(Htr):
    pred, _ = HX.fit_gp_direction(Htr, {})
    return pred


def predict(pred, H):
    P, F, _ = pred(H)
    return P, F


def cv(H, extra=None, frac=None, seed=0):
    """Five folds grouped by action over H, with extra always in training; frac keeps that share of each fold's
    training actions of H (random, seed)."""
    fo = HX.folds(H)
    P, F = np.full((len(fo), 3), np.nan), np.full((len(fo), 3), np.nan)
    rng = np.random.default_rng(seed)
    ids = H["id"].astype(str)
    for k in range(HX.FOLDS):
        te = fo == k
        tr = ~te
        if frac is not None:
            acts = np.unique(ids[tr])
            keep = rng.choice(acts, size=max(1, int(round(frac * len(acts)))), replace=False)
            tr = tr & np.isin(ids, keep)
        Htr = K.sub(H, tr) if extra is None else K.cat(K.sub(H, tr), extra)
        P[te], F[te] = predict(fit(Htr), K.sub(H, te))
    return P, F


def scores(H, P, F, by=("fixture",)):
    E = HX.errors(H, P, F)
    sh = np.linalg.norm(K.shear(F, H["n"]) - K.shear(H["f"], H["n"]), axis=1)
    out = {"all": dict(HX.stats(E, np.ones(len(H["id"]), bool)), shear_err_N=K.med(sh))}
    for key in by:
        for v in np.unique(H[key].astype(str)):
            m = H[key].astype(str) == v
            out[v] = dict(HX.stats(E, m), shear_err_N=K.med(sh[m]))
    return out


def short(s):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()
            if k in ("n", "loc_mm", "loc_p90_mm", "force_rel_err", "force_rel_bias", "angle_deg", "shear_err_N")}


def part_fixtures(D):
    out = {}
    pr = {f: K.sub(D[f], D[f]["kind"] == "press") for f in FIXTURES}
    pool = K.cat(*pr.values())
    out["own_cv"] = {f: short(scores(pr[f], *cv(pr[f]))["all"]) for f in FIXTURES}
    P, F = cv(pool)
    out["pooled_cv"] = {f: short(s) for f, s in scores(pool, P, F).items()}
    out["without_it"] = {}
    for f in FIXTURES:
        rest = K.cat(*[pr[g] for g in FIXTURES if g != f])
        out["without_it"][f] = short(scores(pr[f], *predict(fit(rest), pr[f]))["all"])
    ps = fit(pr["sphere"])
    out["sphere_only"] = {f: short(scores(pr[f], *predict(ps, pr[f]))["all"]) for f in ("tall", "rod")}
    out["press_holds"] = {f: int(len(pr[f]["id"])) for f in FIXTURES}
    out["press_actions"] = {f: int(len(np.unique(pr[f]["id"]))) for f in FIXTURES}
    return out


def part_slides(D):
    out = {}
    pr = K.cat(*[K.sub(D[f], D[f]["kind"] == "press") for f in FIXTURES])
    sl = {f: K.sub(D[f], D[f]["kind"] == "slide") for f in FIXTURES}
    for f in FIXTURES:
        if not len(sl[f]["id"]):
            continue
        others = [sl[g] for g in FIXTURES if g != f and len(sl[g]["id"])]
        r = dict(slide_holds=int(len(sl[f]["id"])), slide_actions=int(len(np.unique(sl[f]["id"]))),
                 hexe_shear_N=K.med(np.linalg.norm(K.shear(sl[f]["f"], sl[f]["n"]), axis=1)))
        r["presses"] = short(scores(sl[f], *predict(fit(pr), sl[f]))["all"])
        r["presses_other_slides"] = short(scores(sl[f], *predict(fit(K.cat(pr, *others)), sl[f]))["all"])
        r["all_cv"] = short(scores(sl[f], *cv(sl[f], extra=K.cat(pr, *others)))["all"])
        out[f] = r
    return out


def part_levels(D, pooled_cv=None):
    pool = K.cat(*[K.sub(D[f], D[f]["kind"] == "press") for f in FIXTURES])
    P0, F0 = pooled_cv if pooled_cv is not None else cv(pool)
    out = {}
    for name, test in (("0.5 N and under", pool["f_n"] <= LOW), ("over 3.5 N", pool["f_n"] > HIGH)):
        P, F = predict(fit(K.sub(pool, ~test)), K.sub(pool, test))
        Ht = K.sub(pool, test)
        out[name] = dict(left_out=short(scores(Ht, P, F)["all"]), pooled_cv=short(scores(Ht, P0[test], F0[test])["all"]),
                         left_out_by_fixture={k: short(v) for k, v in scores(Ht, P, F).items() if k != "all"})
    return out


def part_count(D):
    pool = K.cat(*[K.sub(D[f], D[f]["kind"] == "press") for f in FIXTURES])
    n_act = len(np.unique(pool["id"]))
    out = {"actions": int(n_act)}
    for fr in FRACTIONS:
        rows = [scores(pool, *cv(pool, frac=fr, seed=s)) for s in SEEDS]
        out[f"{fr:g}"] = {k: {m: round(float(np.median([r[k][m] for r in rows])), 3) for m in ("loc_mm", "force_rel_err",
                                                                                                 "angle_deg")}
                          for k in rows[0]}
        out[f"{fr:g}"]["training_actions"] = int(round(fr * n_act * (HX.FOLDS - 1) / HX.FOLDS))
    return out


def part_tip_end(D):
    pool = K.cat(*[K.sub(D[f], D[f]["kind"] == "press") for f in FIXTURES])
    end = pool["region"].astype(str) == "tip"
    if not end.any():
        return None
    Ht = K.sub(pool, end)
    out = dict(holds=int(end.sum()), actions=sorted(np.unique(Ht["id"]).tolist()))
    out["without_end_presses"] = short(scores(Ht, *predict(fit(K.sub(pool, ~end)), Ht))["all"])
    ids = pool["id"].astype(str)
    P, F = np.full((len(ids), 3), np.nan), np.full((len(ids), 3), np.nan)
    for i in np.unique(ids[end]):
        m = ids == i
        P[m], F[m] = predict(fit(K.sub(pool, ~m)), K.sub(pool, m))
    out["with_other_end_presses"] = short(scores(Ht, P[end], F[end])["all"])
    return out


def part_transfer(Ds):
    out = {}
    labels = list(Ds)
    for a, b in ((labels[0], labels[1]), (labels[1], labels[0])):
        A = K.cat(*[K.sub(Ds[a][f], np.isin(Ds[a][f]["kind"], ("press", "slide"))) for f in FIXTURES])
        Bp = K.cat(*[K.sub(Ds[b][f], Ds[b][f]["kind"] == "press") for f in FIXTURES])
        Bs = K.cat(*[K.sub(Ds[b][f], Ds[b][f]["kind"] == "slide") for f in FIXTURES])
        pa = fit(A)
        r = dict(presses={k: short(v) for k, v in scores(Bp, *predict(pa, Bp)).items()},
                 slides={k: short(v) for k, v in scores(Bs, *predict(pa, Bs)).items()})
        ids = Bp["id"].astype(str)
        acts = np.unique(ids)
        r["with_k"] = {}
        for k in KS:
            rows = []
            for s in SEEDS:
                rng = np.random.default_rng(s)
                add = rng.choice(acts, size=min(k, len(acts) - 1), replace=False)
                m = np.isin(ids, add)
                rows.append(scores(K.sub(Bp, ~m), *predict(fit(K.cat(A, K.sub(Bp, m))), K.sub(Bp, ~m))))
            r["with_k"][str(k)] = {f: {m_: round(float(np.median([x[f][m_] for x in rows])), 3)
                                       for m_ in ("loc_mm", "force_rel_err", "angle_deg")} for f in rows[0]}
        out[f"{a} -> {b}"] = r
    return out


def part_time(tip):
    out = {}
    for fx, ses in TIPS[tip]["sessions"].items():
        p = LAB / ses / "events.jsonl"
        if not p.exists():
            continue
        start, rows, fams = {}, collections.defaultdict(list), collections.defaultdict(list)
        for line in p.read_text().splitlines():
            e = json.loads(line)
            if e.get("event") == "start":
                start[e["id"]] = (e["t"], e.get("kind"), e.get("stage"))
            elif e.get("event") == "end" and e.get("id") in start and e.get("status") == "ok":
                t0, kind, stage = start.pop(e["id"])
                fam = "touch-off" if kind == "touch" else ("end press" if "top" in e["id"] else kind)
                rows[fam].append(e["t"] - t0)
                if kind == "press":
                    lv = [x.get("N", x.get("level_N")) if isinstance(x, dict) else x
                          for x in (e.get("rec") or {}).get("levels", [])]
                    lv = [x for x in lv if isinstance(x, (int, float))]
                    fams[e["id"].split("-")[0]].append((e["t"] - t0, max(lv) if lv else None))
        out[fx] = {k: dict(actions=len(v), median_s=round(float(np.median(v)), 1), hours=round(float(np.sum(v)) / 3600, 2))
                   for k, v in rows.items() if k not in ("check", None)}
        out[fx]["press families"] = {
            k: dict(actions=len(v), hours=round(sum(x for x, _ in v) / 3600, 2),
                    largest_level_N=max((y for _, y in v if y is not None), default=None)) for k, v in fams.items()}
    return out


SHARED = {"728": "029", "730": "009", "719": "013", "581": "032", "576": "051", "640": "034"}


def part_shared(D):
    S, T = K.sub(D["sphere"], D["sphere"]["kind"] == "press"), K.sub(D["tall"], D["tall"]["kind"] == "press")
    sid, tid = S["id"].astype(str), T["id"].astype(str)
    grid = [f"S:grid-{v}" for v in SHARED.values()]
    ps = fit(K.sub(S, ~np.isin(sid, grid)))
    rows = []
    for a, b in SHARED.items():
        for lv in np.unique(np.round(T["level_N"][tid == f"T:tall-shared-{a}"], 2)):
            it = np.where((tid == f"T:tall-shared-{a}") & (np.round(T["level_N"], 2) == lv))[0]
            js = np.where((sid == f"S:grid-{b}") & (np.round(S["level_N"], 2) == lv))[0]
            if not len(it) or not len(js):
                continue
            i, j = it[0], js[0]
            Ti, Sj = K.sub(T, np.arange(len(tid)) == i), K.sub(S, np.arange(len(sid)) == j)
            (PT, FT), (PS, FS) = predict(ps, Ti), predict(ps, Sj)
            eT, eS = HX.errors(Ti, PT, FT), HX.errors(Sj, PS, FS)
            dT, dS = T["d_board"][i], S["d_board"][j]
            dR = T["R_BE"][i].reshape(3, 3) @ S["R_BE"][j].reshape(3, 3).T
            rows.append(dict(
                tall=f"tall-shared-{a}", sphere=f"grid-{b}", region=str(S["region"][j]), level_N=float(lv),
                f_n=[round(float(T["f_n"][i]), 3), round(float(S["f_n"][j]), 3)],
                point_mm=round(float(np.linalg.norm(T["p"][i] - S["p"][j])), 3),
                board_angle_deg=round(float(np.degrees(np.arccos(np.clip(dT @ dS / np.linalg.norm(dT) / np.linalg.norm(dS),
                                                                             -1, 1)))), 2),
                gain_ratio=round(float((np.linalg.norm(dT) / T["f_n"][i]) / (np.linalg.norm(dS) / S["f_n"][j])), 3),
                shear_per_N=[round(float(np.linalg.norm(K.shear(T["f"][i:i + 1], T["n"][i:i + 1])) / T["f_n"][i]), 3),
                             round(float(np.linalg.norm(K.shear(S["f"][j:j + 1], S["n"][j:j + 1])) / S["f_n"][j]), 3)],
                tool_angle_deg=round(float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))), 2),
                gp_loc_mm=[round(float(eT["loc"][0]), 3), round(float(eS["loc"][0]), 3)],
                gp_force_rel=[round(float(eT["rel"][0]), 3), round(float(eS["rel"][0]), 3)],
                gp_points_apart_mm=round(float(np.linalg.norm(PT[0] - PS[0])), 3)))
    hi = [r for r in rows if r["level_N"] >= 1.0]
    med = lambda f: round(float(np.median([f(r) for r in hi])), 3)   # noqa: E731
    summary = dict(pairs=len(hi), point_mm=med(lambda r: r["point_mm"]), board_angle_deg=med(lambda r: r["board_angle_deg"]),
                   gain_ratio=med(lambda r: r["gain_ratio"]), shear_per_N_tall=med(lambda r: r["shear_per_N"][0]),
                   shear_per_N_sphere=med(lambda r: r["shear_per_N"][1]), tool_angle_deg=med(lambda r: r["tool_angle_deg"]),
                   gp_loc_tall_mm=med(lambda r: r["gp_loc_mm"][0]), gp_loc_sphere_mm=med(lambda r: r["gp_loc_mm"][1]),
                   gp_points_apart_mm=med(lambda r: r["gp_points_apart_mm"]))
    return dict(at_1N_and_over=summary, rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", default="fixtures,slides,levels,count,tip_end,shared,transfer,time")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    parts = a.parts.split(",")
    Ds = {t: {f: load(c["holds"][f], f) for f in FIXTURES} for t, c in TIPS.items()}
    old = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}   # a run of some parts keeps the others
    res = dict(script="scripts/ur5_tip_calibration_motion_value.py", model="direction GP (HX.fit_gp_direction)",
               holds={t: {f: dict(folder=c["holds"][f],
                                  **{k: int(np.sum(Ds[t][f]["kind"] == k)) for k in ("press", "slide", "twist", "roll")})
                          for f in FIXTURES} for t, c in TIPS.items()},
               tips=old.get("tips", {}), transfer=old.get("transfer"))
    for t, D in Ds.items():
        r = res["tips"].setdefault(t, {})
        if "fixtures" in parts:
            log(t, "fixtures")
            r["fixtures"] = part_fixtures(D)
        if "slides" in parts:
            log(t, "slides")
            r["slides"] = part_slides(D)
        if "levels" in parts:
            log(t, "levels")
            r["levels"] = part_levels(D)
        if "count" in parts:
            log(t, "count")
            r["count"] = part_count(D)
        if "tip_end" in parts:
            log(t, "tip_end")
            r["tip_end"] = part_tip_end(D)
        if "shared" in parts:
            log(t, "shared")
            r["shared"] = part_shared(D)
        if "time" in parts:
            r["time"] = part_time(t)
    if "transfer" in parts:
        log("transfer")
        res["transfer"] = part_transfer(Ds)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    log("->", Path(a.out).relative_to(ROOT) if Path(a.out).is_relative_to(ROOT) else a.out)


if __name__ == "__main__":
    main()
