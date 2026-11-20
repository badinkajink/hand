#!/usr/bin/env python3
"""HOM with a clearance term in Eq. (2), on the CPU calibrated plant of the twist ranking (hand-4a, 2026-11-20; owner,
08:0x, "Sure on all three decisions": the collision-aware HOM of the decision log, "a clearance term in its least
squares could give these layouts a feasible twist").

For each finger's Eq. (2), the K closest pairs of link spheres between that finger and each other finger add rows
    sqrt(w) (n . J u - k_c (d_s - g)) / ell,    w = w_c ((d_s - g) / d_s)^2 for g < d_s, else 0,
where g is the spheres' surface gap in HOM's kinematic model (the Mirror, set to the measured joints), n the unit vector
from the other finger's sphere to this finger's, and J the Jacobian of this finger's sphere centre over its three
joints. A row asks the gap to open at k_c (d_s - g) while the other finger is held; a gap of d_s or more leaves HOM
unchanged. ell is HOM's length scale of the linear rows (0.02 m). The spheres: the Mirror's sphere geoms of 2 mm or more
on the yaw, MCP and PIP frames, every SUB-th of the PIP frame's small spheres (the DIP link), and an 8 mm ball at the
tip. The rows reach sr2hand.control.hom.solve_eq2 through a wrapper set in this process only (the bench server's code is
not changed).

Each run is the twist ranking's (sim/20261119-cad_gap_screen.py's job: the calibrated plant, seed 0, the CAD collision
spheres along the simulated twist), so the least CAD gap is the same test the feasible search applied (10 mm or more
under both models with the sphere held).

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 CUDA_VISIBLE_DEVICES= .venv/bin/python \\
        docs/experiments/20261120-controller_rematch/sim/20261120-clearance_hom.py run --workers 12
    ... summary     writes data/20261120-clearance_hom.json
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
RANK = ROOT / "docs/experiments/20261119-twist_layout_ranking"
spec = importlib.util.spec_from_file_location("cad_gap_screen", RANK / "sim/20261119-cad_gap_screen.py")
CS = importlib.util.module_from_spec(spec)
sys.modules["cad_gap_screen"] = CS
spec.loader.exec_module(CS)
SR = CS.SR
ROWS = ROOT / "logs/20261120-controller_rematch/rows_clear.jsonl"
SUB = 4
R_TIP = 0.008
CLEAR = {"d_s": 0.0, "k_c": 5.0, "w_c": 0.0, "K": 4}        # w_c 0: HOM as it is
SETTINGS = [{"d_s": 0.0, "k_c": 5.0, "w_c": 0.0, "K": 4}] + [
    {"d_s": d, "k_c": 5.0, "w_c": w, "K": 4} for d in (0.010, 0.015, 0.020) for w in (1.0, 10.0)]

_CTX = {"ctl": None, "i": 0}
_SPH = {}


def spheres(m):
    """{finger: (geom ids, radii (n,), tip geom id)} of the Mirror's model."""
    key = id(m)
    if key in _SPH:
        return _SPH[key]
    out = {}
    for f in ("thumb", "index", "middle"):
        ids, rad, small = [], [], []
        tip = None
        for g in range(m.ngeom):
            b = m.body(m.geom_bodyid[g]).name
            if not b.startswith(f + "_") or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue
            if b == f"{f}_tip" and int(m.geom_type[g]) == 7:
                tip = g
            if int(m.geom_type[g]) != 2 or b not in (f"{f}_yaw_frame", f"{f}_mcp_frame", f"{f}_pip_frame"):
                continue
            if m.geom_size[g][0] >= 0.002:
                ids.append(g)
                rad.append(float(m.geom_size[g][0]))
            elif b == f"{f}_pip_frame":
                small.append(g)
        for g in small[::SUB]:
            ids.append(g)
            rad.append(float(m.geom_size[g][0]))
        out[f] = (np.array(ids, int), np.array(rad), tip)
    _SPH[key] = out
    return out


def clearance_rows(mr, f):
    """[(a (3,), target gap rate (m/s), weight)] for finger f's Eq. (2) at the Mirror's state."""
    import mujoco
    m, d = mr.m, mr.d
    S = spheres(m)
    d_s, k_c, w_c, K = CLEAR["d_s"], CLEAR["k_c"], CLEAR["w_c"], int(CLEAR["K"])

    def centres(fi):
        ids, rad, tip = S[fi]
        C = d.geom_xpos[ids].copy()
        B = m.geom_bodyid[ids].copy()
        if tip is not None:
            C = np.vstack([C, d.geom_xpos[tip][None]])
            rad = np.concatenate([rad, [R_TIP]])
            B = np.concatenate([B, [m.geom_bodyid[tip]]])
        return C, rad, B
    Cf, rf, Bf = centres(f)
    jacp = np.zeros((3, m.nv))
    out = []
    for o in ("thumb", "index", "middle"):
        if o == f:
            continue
        Co, ro, _ = centres(o)
        D = np.linalg.norm(Cf[:, None] - Co[None], axis=2) - rf[:, None] - ro[None]
        flat = np.argsort(D, axis=None)[:K]
        for idx in flat:
            ia, ib = np.unravel_index(idx, D.shape)
            g = float(D[ia, ib])
            if g >= d_s:
                break
            v = Cf[ia] - Co[ib]
            n = v / max(np.linalg.norm(v), 1e-9)
            mujoco.mj_jac(m, d, jacp, None, Cf[ia], int(Bf[ia]))
            a = n @ jacp[:, mr.vadr[f]]
            act = d_s - g
            out.append((a, k_c * act, w_c * (act / d_s) ** 2))
    return out


def install():
    """The wrappers on sr2hand.control.hom (this process and its forks)."""
    from sr2hand.control import hom as H
    if getattr(H, "_clear_installed", False):
        return
    solve0, step0 = H.solve_eq2, H.HomController.step

    def solve_eq2(A, b, v_ref, w, theta, theta_ref, lo, hi, dt, *args, scale=None, **kw):
        ctl = _CTX["ctl"]
        if ctl is None or CLEAR["w_c"] <= 0 or CLEAR["d_s"] <= 0:
            return solve0(A, b, v_ref, w, theta, theta_ref, lo, hi, dt, *args, scale=scale, **kw)
        f = ctl.order[_CTX["i"]]
        _CTX["i"] += 1
        rows = clearance_rows(ctl.mr, f)
        if not rows:
            return solve0(A, b, v_ref, w, theta, theta_ref, lo, hi, dt, *args, scale=scale, **kw)
        sc = np.asarray(scale if scale is not None else H.SCALE, float)
        A2 = np.vstack([A, np.array([r[0] for r in rows])])
        b2 = np.concatenate([b, np.zeros(len(rows))])
        v2 = np.concatenate([np.asarray(v_ref, float), [r[1] for r in rows]])
        w2 = np.concatenate([np.asarray(w, float), [r[2] for r in rows]])
        s2 = np.concatenate([sc, np.full(len(rows), 1.0 / ctl.prm.ell)])
        return solve0(A2, b2, v2, w2, theta, theta_ref, lo, hi, dt, *args, scale=s2, **kw)

    def step(self, meas, cmd):
        _CTX["ctl"], _CTX["i"] = self, 0
        try:
            return step0(self, meas, cmd)
        finally:
            _CTX["ctl"] = None
    H.solve_eq2, H.HomController.step = solve_eq2, step
    H._clear_installed = True


def job(item):
    """cad_gap_screen.job with the clearance setting and HOM's forces (item prm) of the item."""
    from sr2hand.robot import sr2_layouts as SL
    CLEAR.update(item["clear"])
    install()
    e = {"call": item["call"], "xy": item["xy"], "k": item["k"], "prm": item.get("prm")}
    tr, r = CS.CG.simulated(e, item["model"])
    SA, FT, HS = CS.lab()
    plan = {"meta": {"hom": {"d6_P_mm": [float(v) for v in SL.sym6(*item["xy"])]}}}
    g = CS.CG.gaps_along(plan, tr, SA, FT, HS, "joints")
    return {**{k: item.get(k) for k in ("call", "xy", "k", "model", "group", "clear", "prm", "stage")},
            "track": r.get("track"), "held": r.get("held"), "least_mm": min(v["gap_mm"] for v in g.values()),
            "gaps": {p: v["gap_mm"] for p, v in g.items()}}


# the grasp search with the term (08:5x): every grasp of the near-touching layouts at seed 0 under both models, with the
# strongest setting of the first pass, with stage E's best forces, and with both
F_BEST = {"f_tripod": 1.0, "w_spin_tripod": 0.25}
GRASP_SETTINGS = [({"d_s": 0.020, "k_c": 5.0, "w_c": 10.0, "K": 4}, None),
                  ({"d_s": 0.0, "k_c": 5.0, "w_c": 0.0, "K": 4}, F_BEST),
                  ({"d_s": 0.020, "k_c": 5.0, "w_c": 10.0, "K": 4}, F_BEST)]


def layouts(n_clear):
    """The morning list's layouts whose simulated twist touches or comes within 10 mm (no droppers), and the first
    n_clear of its clear default-force layouts as a check that the term leaves them as they were."""
    lst = json.loads((RANK / "data/20261119-trial_plans.json").read_text())["plans"]
    base = [e for e in lst if e.get("role") in ("ranked", "anchor") and not e.get("prm") and "_cem_" not in e["plan"]
            and e.get("group") != "dropper"]
    near = [e for e in base if (e.get("cad_least_mm") if e.get("cad_least_mm") is not None else 99.0) < 10.0]
    clear = [e for e in base if (e.get("cad_least_mm") if e.get("cad_least_mm") is not None else 0.0) >= 15.0][:n_clear]
    return [(e, "near") for e in near] + [(e, "clear") for e in clear]


def rows():
    if not ROWS.exists():
        return []
    return [json.loads(x) for x in ROWS.read_text().splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "grasps", "summary"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-clear", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd in ("run", "grasps"):
        import multiprocessing as mp

        def ident(i):
            return (i["call"], i["k"], i["model"], json.dumps(i["clear"], sort_keys=True),
                    json.dumps(i.get("prm"), sort_keys=True))
        done = {ident(r) for r in rows()}
        if a.cmd == "run":
            items = [{"call": int(e["call"]), "xy": e["xy"], "k": int(e["k"]), "model": m, "group": grp, "clear": s,
                      "prm": None, "stage": "layouts"}
                     for e, grp in layouts(a.n_clear) for m in SR.MODELS for s in SETTINGS]
        else:
            ng = {c: n for c, xy, n in SR.layouts()}
            items = [{"call": int(e["call"]), "xy": e["xy"], "k": k, "model": m, "group": grp, "clear": s, "prm": prm,
                      "stage": "grasps"}
                     for e, grp in layouts(0) if grp == "near" for k in range(ng[int(e["call"])])
                     for s, prm in GRASP_SETTINGS for m in SR.MODELS]
        items = [i for i in items if ident(i) not in done]
        if a.limit:
            items = items[:a.limit]
        print(f"{len(items)} runs", flush=True)
        R = SR.load_replan()
        R.RG.gear_class = R.RC2.clutch_class
        R.RC2.STIFF, R.RC2.BREAKAWAY = True, 0.0
        R.RG.RF.RL.rows_file()
        SR._R["mod"] = R
        ROWS.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with mp.get_context("fork").Pool(a.workers, initializer=SR._init) as pool, ROWS.open("a") as fh:
            for r in pool.imap_unordered(job, items, chunksize=1):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                n += 1
                if n % 20 == 0 or n == len(items):
                    print(f"  {n}/{len(items)} {time.strftime('%H:%M:%S')}", flush=True)
    rr = rows()
    out = []
    for s in SETTINGS:
        ks = json.dumps(s, sort_keys=True)
        for grp in ("near", "clear"):
            sel = [r for r in rr if json.dumps(r["clear"], sort_keys=True) == ks and r["group"] == grp
                   and r.get("stage", "layouts") == "layouts" and r.get("prm") is None]
            by = {}
            for r in sel:
                by.setdefault((r["call"], r["k"]), {})[r["model"]] = r
            both = [v for v in by.values() if set(v) == set(SR.MODELS)]
            if not both:
                continue
            feas = [all(x["least_mm"] >= 10.0 and x["held"] for x in v.values()) for v in both]
            least = [min(x["least_mm"] for x in v.values()) for v in both]
            trk = [np.mean([SR.score(x) for x in v.values()]) for v in both]
            out.append({"clear": s, "group": grp, "layouts": len(both), "feasible": int(sum(feas)),
                        "least_mm_median": round(float(np.median(least)), 1),
                        "track_mean": round(float(np.mean(trk)), 4),
                        "track_feasible_mean": round(float(np.mean([t for t, f in zip(trk, feas) if f])), 4)
                        if any(feas) else None})
    # per layout: the first pass's plan grasp without and with the strongest term; the grasp search's best feasible
    # grasp (10 mm or more under both models, held) per setting, by the mean score of both models
    lay = []
    for e, grp in layouts(a.n_clear):
        c, k = int(e["call"]), int(e["k"])
        rec = {"call": c, "xy": e["xy"], "k": k, "group": grp}
        for lab, s_ in (("base", SETTINGS[0]), ("d20_w10", SETTINGS[-1])):
            v = {r["model"]: r for r in rr if r["call"] == c and r["k"] == k and r.get("stage", "layouts") == "layouts"
                 and r.get("prm") is None and json.dumps(r["clear"], sort_keys=True) == json.dumps(s_, sort_keys=True)}
            if set(v) == set(SR.MODELS):
                rec[lab] = {"least_mm": min(x["least_mm"] for x in v.values()),
                            "track": round(float(np.mean([SR.score(x) if x["held"] else 0.0 for x in v.values()])), 4),
                            "feasible": all(x["least_mm"] >= 10.0 and x["held"] for x in v.values())}
        for n_, (s_, prm) in zip(("grasps_clear", "grasps_force", "grasps_clear_force"), GRASP_SETTINGS):
            by = {}
            for r in rr:
                if (r["call"] == c and r.get("stage") == "grasps" and json.dumps(r["clear"], sort_keys=True)
                        == json.dumps(s_, sort_keys=True) and json.dumps(r.get("prm"), sort_keys=True)
                        == json.dumps(prm, sort_keys=True)):
                    by.setdefault(r["k"], {})[r["model"]] = r
            full = {kk: v for kk, v in by.items() if set(v) == set(SR.MODELS)}
            if not full:
                continue
            fz = {kk: float(np.mean([SR.score(x) if x["held"] else 0.0 for x in v.values()])) for kk, v in full.items()
                  if all(x["least_mm"] >= 10.0 and x["held"] for x in v.values())}
            best = max(fz, key=fz.get) if fz else None
            rec[n_] = {"grasps": len(full), "feasible": len(fz), "best_k": best,
                       "best_track": round(fz[best], 4) if best is not None else 0.0}
        lay.append(rec)
    path = STUDY / "data/20261120-clearance_hom.json"
    path.write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "), "written": time.strftime("%F %H:%M"),
                                "settings": SETTINGS, "grasp_settings": [{"clear": c_, "prm": p_} for c_, p_ in
                                                                         GRASP_SETTINGS],
                                "summary": out, "layouts": lay}, indent=1) + "\n")
    for o in out:
        c = o["clear"]
        print(f"d_s {1e3 * c['d_s']:4.0f} mm w_c {c['w_c']:5.1f}  {o['group']:5s}: {o['feasible']}/{o['layouts']} feasible, "
              f"least gap median {o['least_mm_median']} mm, tracking {o['track_mean']:.3f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
