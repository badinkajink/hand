#!/usr/bin/env python3
"""The hand-object scale landscape: per-cell task scores J_T(s, d), the ridges s*_T(d), the scaling-law fits and the
explanatory variables, from the study's rows. Writes docs/experiments/20261008-hand_object_scale/landscape.json.

Tasks (each maximised over the grasps a cell was evaluated at, so every cell is scored at its own best grasp):
  kin    fixed-contact precision-manipulation workspace: reachable object poses (3 mm position grid x 123
         orientations within 45 deg), Borras and Dollar's measure; best straddle         workspace.jsonl
  rotx   rotation range about the pinch axis with the contacts fixed (deg, the smaller of the two signs)
  hold   weakest of six force thresholds (N) an equator grip at 4 N per pad resists      hold.jsonl
  turn   held turn of the HOM controller (deg; median of 3 placements; 0 when dropped)   turn.jsonl
Ridge: for each object, the layout of largest score along a family, refined by a parabola through the maximum and its
two neighbours. Scaling law: palm radius rho* (spheres) or half thumb-pair span a* (cylinders) against the object
radius r, fitted as rho* = alpha r + beta.

  .venv/bin/python scripts/hand_object_scale_landscape.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hand_object_scale_kin as K  # noqa: E402

OUT_DIR = K.OUT_DIR
TASKS = ("kin", "rotx", "hold", "turn")
VARS = ("ext_min_mm", "sigma_min_mm", "margin_deg", "angle_max_deg", "f_trans_min", "ff_clear_mm")
F_CAP = 20.0


def rows(name):
    p = OUT_DIR / name
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def force_transmission(row_kin: dict, shape: str, x_sep: float, y_sep: float, d: float, h: float, sp: float) -> float:
    """Smallest over fingers of the pad force (N) its servos produce along the inward contact normal per N m of the
    most loaded joint: 1 / max_j |(J^T n)_j|. The servo torque margin of the grasp posture."""
    rel, c = K.contact_targets(shape, x_sep, y_sep, d / 2 - K.GAP, sp)
    centre = np.array([c[0], c[1], -h])
    out = []
    for f in K.FINGERS:
        q = np.radians(row_kin["q_deg"][f])
        n = rel[f].copy()
        if shape == "cylinder":
            n[1] = 0.0
        n = -n / np.linalg.norm(n)
        tau = K.jac_pad(f, q).T @ n
        out.append(1.0 / max(np.max(np.abs(tau)), 1e-9))
    return float(min(out))


def cell_vars(kin: dict, shape, x_sep, y_sep, d, h, sp) -> dict:
    return {"ext_min_mm": min(kin["ext_mm"].values()), "sigma_min_mm": min(kin["sigma_min_mm"].values()),
            "margin_deg": kin["min_margin_deg"], "angle_max_deg": max(kin["contact_angle_deg"].values()),
            "f_trans_min": force_transmission(kin, shape, x_sep, y_sep, d, h, sp), "ff_clear_mm": kin["ff_clear_mm"]}


def scores() -> dict:
    """(tag, shape, d) -> {task: score, task_grasp: {...}, vars per task}."""
    S = defaultdict(dict)
    for r in rows("workspace.jsonl"):
        k = (r["tag"], r["shape"], r["d_mm"])
        if r["poses"] > S[k].get("kin", -1):
            S[k]["kin"] = r["poses"]
            S[k]["kin_spread"] = r["spread_mm"]
        rx = min(r["rot_x_deg"])
        if rx > S[k].get("rotx", -1):
            S[k]["rotx"] = rx
    hold = defaultdict(list)
    for r in rows("hold.jsonl"):
        if r.get("status") != "ok" or r.get("model", "pads") != "pads":
            continue
        dist = r.get("disturb") or {}
        fmin = min([F_CAP if v is None else v for kk, v in dist.items() if kk.startswith("F")] or [0.0]) \
            if r["held"] else 0.0
        hold[(r["tag"], r["shape"], r["d_mm"])].append((fmin, r))
    for k, lst in hold.items():
        fmin, r = max(lst, key=lambda z: z[0])
        S[k]["hold"] = fmin
        S[k]["hold_grasp"] = {"h_mm": r["h_mm"], "spread_mm": r["spread_mm"], "cand": r["cand"]}
        S[k]["hold_vars"] = cell_vars(r["kin"], r["shape"], r["x_sep_mm"] / 1000, r["y_sep_mm"] / 1000,
                                      r["d_mm"] / 1000, r["h_mm"] / 1000, r["spread_mm"] / 1000)
        S[k]["hold_all"] = [round(z[0], 3) for z in sorted(lst, key=lambda z: z[1]["cand"])]
    turn = defaultdict(lambda: defaultdict(list))
    for r in rows("turn.jsonl"):
        if r.get("status") != "ok" or r.get("model", "pads") != "pads":
            continue
        turn[(r["tag"], r["shape"], r["d_mm"])][r["cand"]].append(r)
    for k, by in turn.items():
        best = None
        for ci, lst in by.items():
            med = float(np.median([x["held_turn_deg"] for x in lst]))
            if best is None or med > best[0]:
                best = (med, lst)
        med, lst = best
        r = lst[0]
        S[k]["turn"] = med
        S[k]["turn_seeds"] = [x["held_turn_deg"] for x in lst]
        S[k]["turn_grasp"] = {"h_mm": r["h_mm"], "spread_mm": r["spread_mm"], "cand": r["cand"],
                              "limit_joint": [x["limit_joint"] for x in lst]}
        S[k]["turn_vars"] = cell_vars(r["kin"], r["shape"], r["x_sep_mm"] / 1000, r["y_sep_mm"] / 1000,
                                      r["d_mm"] / 1000, r["h_mm"] / 1000, r["spread_mm"] / 1000)
        S[k]["turn_all"] = {int(ci): round(float(np.median([x["held_turn_deg"] for x in lst])), 2)
                            for ci, lst in by.items()}
    for r in rows("turn.jsonl") + rows("hold.jsonl"):
        if r.get("status") == "infeasible":
            k = (r["tag"], r["shape"], r["d_mm"])
            S[k].setdefault("infeasible", True)
    return S


def ridge(xs, ys):
    """Location of the maximum of ys over xs (sorted), refined by a parabola through it and its neighbours."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) == 0 or np.all(ys <= 0):
        return None
    i = int(np.argmax(ys))
    if 0 < i < len(xs) - 1:
        x3, y3 = xs[i - 1:i + 2], ys[i - 1:i + 2]
        a, b, _ = np.polyfit(x3, y3, 2)
        if a < 0:
            xv = -b / (2 * a)
            if x3[0] <= xv <= x3[2]:
                return float(xv), True
    return float(xs[i]), 0 < i < len(xs) - 1


def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.stack([x, np.ones_like(x)], 1)
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    n = len(x)
    yhat = A @ coef
    s2 = float(((y - yhat) ** 2).sum() / max(n - 2, 1))
    cov = s2 * np.linalg.inv(A.T @ A)
    return {"alpha": float(coef[0]), "beta_mm": float(coef[1]), "alpha_se": float(math.sqrt(cov[0, 0])),
            "beta_se": float(math.sqrt(cov[1, 1])), "rms_mm": float(math.sqrt(((y - yhat) ** 2).mean())), "n": n}


def spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def main() -> int:
    R_ = json.loads((OUT_DIR / "reach.json").read_text())
    fam = R_["families"]
    S = scores()
    out = {"meta": {"date": "2026-10-08", "f_cap_N": F_CAP, "tasks": TASKS, "vars": VARS}, "families": {}}
    for fname in ("diag", "xsep", "ysep"):
        lays = fam[fname]
        coord = {"diag": "t", "xsep": "x_sep", "ysep": "y_sep"}[fname]
        F = {"layouts": lays, "coord": coord, "objects": {}, "ridges": {}, "fits": {}, "explain": {}}
        for shape, ds in K.OBJECT_SET.items():
            for dmm in ds:
                key = f"{shape}:{dmm:g}"
                col = {}
                for lay in lays:
                    s = S.get((lay["tag"], shape, float(dmm)), {})
                    col[lay["tag"]] = {t: s.get(t) for t in TASKS}
                    for t in ("hold", "turn"):
                        if f"{t}_vars" in s:
                            col[lay["tag"]][f"{t}_vars"] = s[f"{t}_vars"]
                            col[lay["tag"]][f"{t}_grasp"] = s[f"{t}_grasp"]
                    if "turn_seeds" in s:
                        col[lay["tag"]]["turn_seeds"] = s["turn_seeds"]
                F["objects"][key] = col
                rr = {}
                for t in TASKS:
                    pts = [(lay[coord], col[lay["tag"]][t]) for lay in lays if col[lay["tag"]][t] is not None]
                    if fname == "diag":
                        pts = sorted(pts)
                    if len(pts) >= 3:
                        rg = ridge([p[0] for p in pts], [p[1] for p in pts])
                        if rg is not None:
                            x, interior = rg
                            rr[t] = {"at": x, "interior": interior, "max": float(max(p[1] for p in pts))}
                            if fname == "diag":
                                rr[t]["x_sep_mm"] = 100.0 - 60.0 * x
                                rr[t]["y_sep_mm"] = 110.0 - 60.0 * x
                                rr[t]["palm_radius_mm"] = K.palm_radius(rr[t]["x_sep_mm"] / 1000,
                                                                        rr[t]["y_sep_mm"] / 1000) * 1000
                F["ridges"][key] = rr
        if fname == "diag":
            for shape in K.OBJECT_SET:
                for t in TASKS:
                    xs, ys = [], []
                    for dmm in K.OBJECT_SET[shape]:
                        rg = F["ridges"].get(f"{shape}:{dmm:g}", {}).get(t)
                        if rg and rg["interior"]:
                            xs.append(dmm / 2)
                            ys.append(rg["palm_radius_mm"] if shape == "sphere" else rg["x_sep_mm"] / 2)
                    if len(xs) >= 3:
                        fit = linfit(xs, ys)
                        fit["r_mm"], fit["s_mm"] = xs, ys
                        fit["ratio"] = [y / x for x, y in zip(xs, ys)]
                        F["fits"][f"{shape}:{t}"] = fit
            # which variable's ridge sits where each task's ridge sits, and the within-object rank correlation
            for t in ("hold", "turn"):
                for shape in K.OBJECT_SET:
                    res = {}
                    for v in VARS:
                        cors, dist = [], []
                        for dmm in K.OBJECT_SET[shape]:
                            col = F["objects"][f"{shape}:{dmm:g}"]
                            pts = sorted((lay["t"], col[lay["tag"]][t], col[lay["tag"]].get(f"{t}_vars", {}).get(v))
                                         for lay in lays if col[lay["tag"]][t] is not None
                                         and col[lay["tag"]].get(f"{t}_vars"))
                            if len(pts) < 3:
                                continue
                            c = spearman([p[1] for p in pts], [p[2] for p in pts])
                            if c is not None:
                                cors.append(c)
                            # the contact angle's ridge is its minimum (the most face-on contacts)
                            vals = [180.0 - p[2] for p in pts] if v == "angle_max_deg" else [p[2] for p in pts]
                            rv = ridge([p[0] for p in pts], vals)
                            rt = ridge([p[0] for p in pts], [p[1] for p in pts])
                            if rv and rt:
                                dist.append(abs(rv[0] - rt[0]))
                        if cors:
                            res[v] = {"spearman_mean": float(np.mean(cors)), "n_objects": len(cors),
                                      "ridge_gap_t": float(np.mean(dist)) if dist else None}
                    F["explain"][f"{shape}:{t}"] = res
        out["families"][fname] = F
    (OUT_DIR / "landscape.json").write_text(json.dumps(out, indent=1, default=float))
    # console summary
    D = out["families"]["diag"]
    for key, rr in D["ridges"].items():
        print(f"{key:14s} " + "  ".join(f"{t} t*={rr[t]['at']:+.3f}{'' if rr[t]['interior'] else '(edge)'} "
                                        f"max {rr[t]['max']:.3g}" for t in TASKS if t in rr))
    for k, fit in D["fits"].items():
        print(f"fit {k:16s} s* = {fit['alpha']:.2f}({fit['alpha_se']:.2f}) r + {fit['beta_mm']:.1f}({fit['beta_se']:.1f}) mm"
              f"  rms {fit['rms_mm']:.1f}  n {fit['n']}  s*/r {min(fit['ratio']):.2f}..{max(fit['ratio']):.2f}")
    for k, res in D["explain"].items():
        print(f"explain {k:16s} " + "  ".join(f"{v} rho {x['spearman_mean']:+.2f} gap {x['ridge_gap_t']}"
                                            for v, x in res.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
