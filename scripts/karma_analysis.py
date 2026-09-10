"""Score KaRMA against this program's own labels on the real_v1 design family.

The question is not whether KaRMA is a good metric -- it is whether it aids DISCOVERY
(ranking hands that have not been built) or VERIFICATION (agreeing with hands that have)
for a single-topology family whose design space is six mount coordinates.

Two things make that a fair test rather than a fishing trip.

First, there are published incumbents on exactly this label. Over the Sobol-8192
population, logistic regression on the six raw mount coordinates gives AUC 0.839 for
retention and the grasp fit's depth_fit gives 0.803, while a purely kinematic geometric
predictor -- the extension ceiling asin(extend/straddle) -- gave AUC 0.617 at n=80. KaRMA
is also purely kinematic and is, for this family, a three-number function of those same
six coordinates, so it cannot carry information the mounts do not. The question is
whether its compression is a USEFUL one.

Second, within-eight correlations are selection artifacts and are reported separately,
never pooled: the eight deployed hands were picked in sim-cos rank order, so any variable
correlated with the selection correlates with the outcome inside that set. depth_fit vs
sim cos reads -0.90 over the eight and +0.076 over the 535.

Requires sklearn, which lives in miniconda and not the repo venv:
    ~/miniconda3/bin/python scripts/karma_analysis.py --table ... --out ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

MOUNT_KEYS = ["thumb x", "thumb y", "index x", "index y", "middle x", "middle y"]
SCORES = ["karma_t", "karma_r", "karma_s", "n_voxels", "volume_mm3", "seed_depth_mm"]


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank AUC, ties averaged. Invariant to class balance, so a case-control sample
    estimates the population value."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_ci(score: np.ndarray, label: np.ndarray, n: int = 2000,
           seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(score))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if label[b].all() or (~label[b]).any() is False:
            continue
        if label[b].sum() in (0, len(b)):
            continue
        vals.append(auc(score[b], label[b]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def logistic_auc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Cross-validated AUC of a logistic fit -- the honest number for a MULTIVARIATE
    predictor, since an in-sample fit on six coordinates would flatter itself.

    KaRMA-S is undefined when only one seed is feasible, so rows carrying a non-finite
    feature are dropped rather than imputed; the caller records how many.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    keep = np.isfinite(X).all(axis=1)
    X, y = X[keep], y[keep]
    if len(np.unique(y)) < 2 or len(y) < 20:
        return float("nan")
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return auc(oof, y.astype(bool))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--karma-16", type=Path, default=None,
                    help="the paper's own results/16_hand_batch/summary.yaml, for the "
                         "spread comparison")
    args = ap.parse_args()

    rows = json.loads(args.table.read_text())
    out: dict = {"n_runs": len(rows)}

    # ── per (variant, pair) ────────────────────────────────────────────────────
    out["arms"] = {}
    for variant in sorted({r["variant"] for r in rows}):
        for pair in sorted({r["pair"] for r in rows if r["variant"] == variant}):
            sub = [r for r in rows if r["variant"] == variant and r["pair"] == pair]
            if len(sub) < 20:
                continue
            arm: dict = {"n": len(sub)}
            y = np.array([bool(r["retained"]) for r in sub])
            arm["n_retained"] = int(y.sum())

            arm["spread"] = {}
            for k in SCORES:
                v = np.array([r[k] for r in sub], float)
                arm["spread"][k] = {"min": float(np.nanmin(v)), "median": float(np.nanmedian(v)),
                                    "max": float(np.nanmax(v)), "sd": float(np.nanstd(v))}

            # Does KaRMA predict which hands survive the retention screen?
            arm["auc_retained"] = {}
            for k in SCORES:
                v = np.array([r[k] for r in sub], float)
                ok = np.isfinite(v)
                if ok.sum() < 20 or y[ok].sum() in (0, int(ok.sum())):
                    continue
                a = auc(v[ok], y[ok])
                lo, hi = auc_ci(v[ok], y[ok])
                arm["auc_retained"][k] = {"auc": round(a, 3), "ci": [round(lo, 3), round(hi, 3)]}

            # Single-number geometric baselines, measurable with a ruler and free. If a
            # 220-core-second kinematic search does not beat these on this family, its
            # ranking is a restatement of them.
            M = np.array([r["mounts_mm"] for r in sub], float)
            contact = {"thumb-index": (0, 2), "thumb-middle": (0, 4),
                       "index-middle": (2, 4)}[pair]
            i, j = contact
            rulers = {
                "pair_mount_distance_mm": -np.hypot(M[:, i] - M[:, j],
                                                    M[:, i + 1] - M[:, j + 1]),
                "x_sep_mm": -((M[:, 2] + M[:, 4]) / 2 - M[:, 0]),
                "y_sep_mm": -np.abs(M[:, 3] - M[:, 5]),
            }
            for k, v in rulers.items():
                if y.sum() in (0, len(y)):
                    continue
                lo, hi = auc_ci(v, y)
                arm["auc_retained"]["ruler_" + k] = {
                    "auc": round(auc(v, y), 3), "ci": [round(lo, 3), round(hi, 3)]}

            # The incumbent, computed on THIS sample so the comparison is like for like.
            X = np.array([r["mounts_mm"] for r in sub], float)
            XK = np.array([[r["karma_t"], r["karma_r"],
                            r["karma_s"] if r["karma_s"] is not None else np.nan]
                           for r in sub], float)
            arm["n_dropped_nonfinite_karma_s"] = int((~np.isfinite(XK).all(axis=1)).sum())
            try:
                arm["auc_retained"]["six_mounts_logistic_cv"] = {
                    "auc": round(logistic_auc(X, y.astype(int)), 3), "ci": None}
                arm["auc_retained"]["karma_trs_logistic_cv"] = {
                    "auc": round(logistic_auc(XK, y.astype(int)), 3), "ci": None}
                arm["auc_retained"]["mounts_plus_karma_logistic_cv"] = {
                    "auc": round(logistic_auc(np.hstack([X, XK]), y.astype(int)), 3), "ci": None}
            except ImportError:
                pass

            # The stricter screen, asked conditionally: among designs that were retained,
            # does KaRMA pick out the ones that went on to be confirmed? The sample is
            # case-control on retention, so this is the only unbiased way to ask about
            # confirmation with it.
            ret = [r for r in sub if r["retained"]]
            arm["auc_confirmed_among_retained"] = {}
            if len(ret) >= 20:
                yc = np.array([bool(r["confirmed"]) for r in ret])
                if 0 < yc.sum() < len(yc):
                    arm["n_confirmed_among_retained"] = int(yc.sum())
                    for k in SCORES:
                        v = np.array([r[k] for r in ret], float)
                        ok = np.isfinite(v)
                        if ok.sum() < 20 or yc[ok].sum() in (0, int(ok.sum())):
                            continue
                        arm["auc_confirmed_among_retained"][k] = {
                            "auc": round(auc(v[ok], yc[ok]), 3)}

            # Does KaRMA rank the screened turn among hands that got that far?
            conf = [r for r in sub if r.get("nom_cos") is not None]
            arm["n_with_turn"] = len(conf)
            arm["rho_nom_cos"] = {}
            if len(conf) >= 20:
                c = np.array([r["nom_cos"] for r in conf], float)
                for k in SCORES:
                    v = np.array([r[k] for r in conf], float)
                    ok = np.isfinite(v)
                    rho, p = stats.spearmanr(v[ok], c[ok])
                    arm["rho_nom_cos"][k] = {"rho": round(float(rho), 3), "p": float(p),
                                             "n": int(ok.sum())}
            out["arms"][f"{variant}|{pair}"] = arm

    # ── best-of-pairs: the fairest single number for a 3-finger hand ───────────
    for variant in sorted({r["variant"] for r in rows}):
        per: dict = {}
        for r in rows:
            if r["variant"] != variant:
                continue
            cur = per.get(r["design"])
            if cur is None or r["karma_t"] > cur["karma_t"]:
                per[r["design"]] = r
        pairs_seen = {r["pair"] for r in rows if r["variant"] == variant}
        if len(per) < 20 or len(pairs_seen) < 2:
            continue
        sub = list(per.values())
        y = np.array([bool(r["retained"]) for r in sub])
        arm = {"n": len(sub), "n_retained": int(y.sum()),
               "pairs_pooled": sorted(pairs_seen), "auc_retained": {}, "rho_nom_cos": {}}
        for k in SCORES:
            v = np.array([r[k] for r in sub], float)
            ok = np.isfinite(v)
            if ok.sum() < 20 or y[ok].sum() in (0, int(ok.sum())):
                continue
            lo, hi = auc_ci(v[ok], y[ok])
            arm["auc_retained"][k] = {"auc": round(auc(v[ok], y[ok]), 3),
                                      "ci": [round(lo, 3), round(hi, 3)]}
        conf = [r for r in sub if r.get("nom_cos") is not None]
        if len(conf) >= 20:
            c = np.array([r["nom_cos"] for r in conf], float)
            for k in SCORES:
                v = np.array([r[k] for r in conf], float)
                rho, p = stats.spearmanr(v, c)
                arm["rho_nom_cos"][k] = {"rho": round(float(rho), 3), "p": float(p),
                                         "n": len(conf)}
        out["arms"][f"{variant}|best-of-pairs"] = arm

    # ── how much do the pairs disagree? ───────────────────────────────────────
    byd: dict = {}
    for r in rows:
        if r["variant"] == "scaled":
            byd.setdefault(r["design"], {})[r["pair"]] = r
    both = [v for v in byd.values() if "thumb-index" in v and "thumb-middle" in v]
    if len(both) >= 20:
        ti = np.array([v["thumb-index"]["karma_t"] for v in both])
        tm = np.array([v["thumb-middle"]["karma_t"] for v in both])
        rho, p = stats.spearmanr(ti, tm)
        out["pair_agreement"] = {
            "n": len(both), "spearman_T_thumb_index_vs_thumb_middle": round(float(rho), 3),
            "p": float(p),
            "median_abs_log2_ratio": round(float(np.median(np.abs(
                np.log2(np.maximum(tm, 1e-9) / np.maximum(ti, 1e-9))))), 3)}

    # ── what is KaRMA measuring, in coordinates this programme already uses? ──
    # x_sep (thumb-to-pair span) and y_sep (pair opening) are linear in the six mounts and
    # are the two the retention screen demonstrably acts on, so they are the natural basis
    # for asking what the metric's compression keeps.
    sub = [r for r in rows if r["variant"] == "scaled" and r["pair"] == "thumb-index"]
    if len(sub) >= 30:
        M = np.array([r["mounts_mm"] for r in sub], float)
        geom = {
            "x_sep_mm": (M[:, 2] + M[:, 4]) / 2 - M[:, 0],
            "y_sep_mm": np.abs(M[:, 3] - M[:, 5]),
            "thumb_to_index_mm": np.hypot(M[:, 2] - M[:, 0], M[:, 3] - M[:, 1]),
            "l_ref_mm": np.array([r["l_ref_mm"] for r in sub], float),
        }
        out["what_it_measures"] = {}
        for gk, gv in geom.items():
            out["what_it_measures"][gk] = {}
            for k in ("karma_t", "karma_r", "n_voxels", "seed_depth_mm"):
                v = np.array([r[k] for r in sub], float)
                rho, p = stats.spearmanr(gv, v)
                out["what_it_measures"][gk][k] = {"rho": round(float(rho), 3), "p": float(p)}
        out["what_it_measures"]["n"] = len(sub)

    # ── the eight deployed hands, reported apart from the population ──────────
    dep = [r for r in rows if r.get("deployed") and r["variant"] == "scaled"
           and r["pair"] == "thumb-index"]
    if dep:
        dep.sort(key=lambda r: r["deployed"])
        out["deployed"] = {
            "note": "Reported separately and never pooled: the eight were selected in "
                    "sim-cos rank order, so any variable correlated with the selection "
                    "correlates with the outcome inside the set.",
            "rows": [{"id": r["deployed"], "design": r["design"], "karma_t": r["karma_t"],
                      "karma_r": r["karma_r"], "karma_s": r["karma_s"],
                      "n_voxels": r["n_voxels"], "l_ref_mm": r["l_ref_mm"],
                      "nom_cos": r.get("nom_cos")} for r in dep]}
        c = [r.get("nom_cos") for r in dep]
        if sum(x is not None for x in c) >= 5:
            m = [i for i, x in enumerate(c) if x is not None]
            for k in ("karma_t", "karma_r"):
                rho, p = stats.spearmanr([dep[i][k] for i in m], [c[i] for i in m])
                out["deployed"][f"rho_{k}_vs_nom_cos_WITHIN_EIGHT"] = {
                    "rho": round(float(rho), 3), "p": float(p), "n": len(m)}

    # ── spread against the paper's own 16 hands ──────────────────────────────
    if args.karma_16 and args.karma_16.exists():
        import yaml
        pub = yaml.safe_load(args.karma_16.read_text())["rows"]
        sub = [r for r in rows if r["variant"] == "scaled" and r["pair"] == "thumb-index"]
        if sub:
            pt = np.array([r["KaRMA_T"] for r in pub])
            ot = np.array([r["karma_t"] for r in sub])
            out["vs_published_16"] = {
                "published_T": {"min": float(pt.min()), "median": float(np.median(pt)),
                                "max": float(pt.max()), "ratio_max_min":
                                    round(float(pt.max() / max(pt.min(), 1e-12)), 1)},
                "real_v1_T": {"min": float(ot.min()), "median": float(np.median(ot)),
                              "max": float(ot.max()), "ratio_max_min":
                                  round(float(ot.max() / max(ot.min(), 1e-12)), 1)},
                "real_v1_median_percentile_in_published": round(float(
                    100.0 * (pt < np.median(ot)).mean()), 1)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:4000])
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
