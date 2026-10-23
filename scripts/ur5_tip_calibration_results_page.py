#!/usr/bin/env python3
"""Page of the fingertip's ground truth and contact predictors from a UR5 calibration session (2026-10-22): the pad's
compliance against its FE model, the board's response against the HEX-E, the pivot check, the viewer's estimator against
the ground truth, and predictors of the contact point and the normal force trained on the session's presses.

Reads the analysis's output of the session in docs/experiments/20261022-ur5_tip_calibration/data/<session>/ and the
session's own folder in lab/<session>/. The outputs come from

    .venv/bin/python scripts/ur5_tip_calibration_analysis.py lab/<session> --gravity-from lab/20261022-195420-first2
    .venv/bin/python scripts/ur5_tip_calibration_fe.py data/<session>/holds.npz --plan lab/<session>/plan.json --out data/<session>/fe.json
    .venv/bin/python scripts/ur5_tip_calibration_predict.py data/<session>
    .venv/bin/python scripts/ur5_tip_calibration_pads.py data/<session>/holds.npz --plan lab/<session>/plan.json \
        --fe-json data/<session>/fe.json --shear 1,10,100 --out data/<session>/pads.json

(paths under docs/experiments/20261022-ur5_tip_calibration/). Writes the figures to media/ and the page
docs/experiments/20261022-ur5_tip_calibration/20261022-ur5_tip_ground_truth_and_predictors.html.

    .venv/bin/python scripts/ur5_tip_calibration_results_page.py [--session 20261022-210852-sphere2]
        [--data 20261022-210852-sphere2-joint]

--data names the analysis folder under data/ (default: the session's own, with its own registration). The rod run's
analyses with the joint registration of three sessions (scripts/ur5_tip_calibration_rod_registration.py) and with
rod1t's own fit, data/20261023-104526-rod1-joint/ and -ownreg/, and the session analysed with the joint registration,
data/<session>-joint/, give Section 8 when they exist. The day repeat's analysis with the session's registration and
its comparison, data/20261023-124052-sph3t-s2reg/ with day_repeat.json from scripts/ur5_tip_calibration_day_repeat.py,
gives Section 9 (--day-data names another folder).
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import ur5_tip_calibration_page as P  # noqa: E402   (esc, link, code, fig, table, plt_setup)

D = ROOT / "docs/experiments/20261022-ur5_tip_calibration"
DATA, LAB, MEDIA = D / "data", D / "lab", D / "media"
PAGE = D / "20261022-ur5_tip_ground_truth_and_predictors.html"
TEMPLATE = ROOT / "scripts/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
LAB_PAGE = D / "20261022-ur5_tip_calibration_lab_runs.html"
PROTOCOL_PAGE = D / "20261022-ur5_tip_calibration_protocol.html"
EST_PAGE = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html"
STATUS_PAGE = ROOT / "docs/experiments/20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html"
QUEUE = D / "QUEUE.md"
E_TPU_DEFAULT = 10.0             # MPa, reorient_backends.E_TPU
ROD_DATA = D / "data/20261023-104526-rod1-joint"
SPH3T = D / "lab/20261023-124052-sph3t"
DAY_DATA = D / "data/20261023-124052-sph3t-s2reg"   # the day repeat (QUEUE.md R21), placed with sphere2's registration
REGION_STYLE = {"face": ("o", "#1f5fbf"), "edge": ("s", "#c0392b"), "flank": ("^", "#2e8b57")}
RNAME = {"face": "face", "edge": "edges", "flank": "flanks"}
MODELS = [("physical", "Physical model, compliance fitted", "#2e8b57", "-"),
          ("gp:board", "Gaussian process on the board", "#1f5fbf", "-"),
          ("ridge:board", "Ridge regression on the board", "#7f7f7f", "--"),
          ("physical+gp:board", "Physical model and a Gaussian process on its residual", "#c0392b", "-"),
          ("mlp:board", "MLP ensemble on the board", "#8e44ad", "-")]
STEPS = ["board", "+temperature change", "+time since rest", "+orientation (leaks the target)"]
STEP_TEXT = {"board": "Board", "+temperature change": "Board, temperature change",
             "+time since rest": "Board, temperature change, time since rest",
             "+orientation (leaks the target)": "Board, temperature change, time since rest, orientation"}


def code(p):
    return P.code(p)


def f1(x, n=2):
    return "&#8211;" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{n}f}"


def ands(parts):
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def hm(t):
    return datetime.datetime.fromtimestamp(t).strftime("%H:%M")


# ------------------------------------------------------------------------------------------------ data
def load(session, data=None, day_data=None):
    sd, ld = DATA / (data or session), LAB / session
    R = dict(session=session, sd=sd, ld=ld)
    R["S"] = json.loads((sd / "summary.json").read_text())
    Hz = np.load(sd / "holds.npz", allow_pickle=False)
    R["H"] = {k: Hz[k] for k in Hz.files}
    R["FE"] = json.loads((sd / "fe.json").read_text())
    R["PR"] = json.loads((sd / "predict/predict.json").read_text()) if (sd / "predict/predict.json").exists() else None
    R["older"] = []                                 # studies taken from the session's own analysis (its registration)
    for k, f in (("CV", "cv4_gp_mlp.json"), ("TR", "magnet_transfer_sim.json"), ("HY", "hybrid_cv.json")):
        q = sd / "predict" / f
        if not q.exists() and (DATA / session / "predict" / f).exists():
            q = DATA / session / "predict" / f
            R["older"].append(k)
        R[k] = json.loads(q.read_text()) if q.exists() else None
    R["PADS"] = json.loads((sd / "pads.json").read_text()) if (sd / "pads.json").exists() else None
    own = DATA / session / "fe.json"
    R["FE_OWN"] = json.loads(own.read_text()) if data and data != session and own.exists() else None
    if ROD_DATA.exists() and (ROD_DATA / "fe.json").exists():
        R["ROD"] = {k: json.loads((ROD_DATA / f).read_text()) for k, f in (("S", "summary.json"), ("FE", "fe.json"),
                    ("PR", "predict/predict.json"), ("PADS", "pads.json")) if (ROD_DATA / f).exists()}
        R["ROD"]["plan"] = json.loads((LAB / ROD_DATA.name.rsplit("-", 1)[0] / "plan.json").read_text())
    for key, folder in (("SPH_JOINT", DATA / f"{session}-joint"), ("ROD_OWN", ROD_DATA.with_name(ROD_DATA.name[:-len("joint")] + "ownreg"))):
        if folder != sd and (folder / "fe.json").exists():
            R[key] = {k: json.loads((folder / f).read_text()) for k, f in (("FE", "fe.json"), ("PR", "predict/predict.json"),
                      ("S", "summary.json")) if (folder / f).exists()}
            if key == "SPH_JOINT":
                a_, b_ = np.load(sd / "holds.npz"), np.load(folder / "holds.npz")
                m_ = a_["kind"] == "press"
                dp = b_["p"][m_] - a_["p"][m_]
                R[key]["moved"] = dict(median=float(np.median(np.linalg.norm(dp, axis=1))), mean=dp.mean(0).tolist())
    dd = Path(day_data) if day_data else DAY_DATA
    if (dd / "day_repeat.json").exists():
        R["DAY"] = {k: json.loads((dd / f).read_text()) for k, f in (("DR", "day_repeat.json"), ("FE", "fe.json"),
                    ("S", "summary.json")) if (dd / f).exists()}
        hz = np.load(dd / "holds.npz", allow_pickle=False)
        R["DAY"]["H"] = {k: hz[k] for k in ("id", "t", "kind", "level_N")}
        R["DAY"]["dir"] = dd
    plan = json.loads((ld / "plan.json").read_text())
    have = {a["id"] for a in plan["actions"]}
    for f in sorted(ld.glob("plan_resume_*.json")):
        for a in json.loads(f.read_text())["actions"]:
            if a["id"] not in have:
                plan["actions"].append(a)
                have.add(a["id"])
    R["plan"] = plan
    R["acts"] = {a["id"]: a for a in plan["actions"]}
    R["touches"] = json.loads((ld / "touches.json").read_text()) if (ld / "touches.json").exists() else []
    ev = [json.loads(l) for l in open(ld / "events.jsonl") if l.strip()]
    R["events"] = ev
    ends = [e for e in ev if e.get("event") == "end"]
    R["ok"] = [e["id"] for e in ends if e.get("status") == "ok"]
    R["abort"] = [e["id"] for e in ends if e.get("status") == "abort"]
    R["t0"], R["t1"] = ev[0]["t"], ev[-1]["t"]
    return R


def press_mask(H):
    return H["kind"] == "press"


def tangential_err(H, key="est_p"):
    d = H[key] - H["p"]
    n = H["n"]
    return np.linalg.norm(d - (d * n).sum(1)[:, None] * n, axis=1)


# ------------------------------------------------------------------------------------------------ figures
def fig_fe(R, out):
    plt = P.plt_setup()
    FE, H = R["FE"], R["FE"]["holds"]
    fn, s1, sb = np.array(H["f_n"]), np.array(H["fe_s1_N"]), np.array(H["fe_best_N"])
    dl = np.array([np.nan if x is None else x for x in H["delta_mm"]], float)
    reg = np.array(H["region"])
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 5.2), dpi=110)
    for r, (mk, col) in REGION_STYLE.items():
        m = reg == r
        ax[0].scatter(fn[m], s1[m], marker=mk, s=36, facecolors="none", edgecolors=col, linewidths=1.2,
                      label=f"{r.capitalize()}, FE as computed")
        ax[0].scatter(fn[m], sb[m], marker=mk, s=30, c=col, label=f"{r.capitalize()}, FE {FE['s_best']:g} times softer")
        ax[1].scatter(fn[m], dl[m], marker=mk, s=34, c=col)
    lim = [0.15, max(30.0, float(np.nanmax(s1)) * 1.3)]
    ax[0].plot(lim, lim, "k-", lw=1.0)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlim(0.15, 8)
    ax[0].set_ylim(*lim)
    ax[0].set_xlabel("Normal force from the HEX-E (N)")
    ax[0].set_ylabel("FE pad's force at the registered position (N)")
    ax[0].set_title("(a) FE pad against the HEX-E")
    for f_, t_ in ((ax[0].set_xticks, (0.2, 0.5, 1, 2, 5)), (ax[0].set_yticks, (0.2, 0.5, 1, 2, 5, 10, 20))):
        f_(t_)
    ax[0].set_xticklabels(["0.2", "0.5", "1", "2", "5"])
    ax[0].set_yticklabels(["0.2", "0.5", "1", "2", "5", "10", "20"])
    ax[0].minorticks_off()
    sr = FE["series"]
    x = np.linspace(0, max(5.0, float(np.nanmax(fn)) * 1.05), 50)
    ax[1].plot(x, sr["a_mm"] + sr["c_mm_N"] * x, "k-", lw=1.2,
               label=f"{sr['a_mm']:.2f} mm + {sr['c_mm_N']:.2f} mm/N")
    ax[1].set_xlim(0, x[-1])
    ax[1].set_ylim(0, max(1.0, float(np.nanmax(dl)) * 1.1))
    ax[1].set_xlabel("Normal force from the HEX-E (N)")
    ax[1].set_ylabel("Indentation beyond the FE pad (mm)")
    ax[1].set_title("(b) Indentation beyond the FE pad")
    ax[1].legend(fontsize=11, loc="upper left", frameon=False)
    for a in ax:
        a.grid(alpha=0.3, which="both")
    h, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_pads(R, out):
    plt = P.plt_setup()
    Q, FE = R["PADS"], R["FE"]
    lv = sorted(Q["by_level"], key=float)
    x = np.array([float(k) for k in lv])
    fh = FE["holds"]
    fl, fn, fb = np.array(fh["level_N"]), np.array(fh["f_n"]), np.array(fh["fe_best_N"])
    fe_b = [float(np.median(fn[(fl == L) & (fb > 0)] / fb[(fl == L) & (fb > 0)])) for L in x]
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.8), dpi=110)
    ax[0].plot(x, [Q["by_level"][k]["pads_over_fe"] for k in lv], "o-", color="#c0392b",
               label=f"Pads over FE pad, pads at {Q['E_FE_MPa']:.1f} MPa")
    ax[0].plot(x, [Q["by_level"][k]["meas_over_pads_E_meas"] for k in lv], "s-", color="#1f5fbf",
               label=f"Measured over pads at {Q['E_meas_MPa']:.2f} MPa")
    ax[0].plot(x, fe_b, "^--", color="#2e8b57", label=f"Measured over FE pad {FE['s_best']:g} times softer")
    ax[0].axhline(1.0, color="k", lw=1.0)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xticks([0.25, 0.5, 1, 2, 5])
    ax[0].set_xticklabels(["0.25", "0.5", "1", "2", "5"])
    ax[0].set_yticks([0.5, 0.7, 1, 1.5, 2])
    ax[0].set_yticklabels(["0.5", "0.7", "1", "1.5", "2"])
    ax[0].minorticks_off()
    ax[0].set_ylim(0.45, 2.2)
    ax[0].set_xlabel("Force level (N)")
    ax[0].set_ylabel("Force ratio (median over the holds)")
    ax[0].set_title("(a) Force ratios at the registered positions")
    ax[0].legend(fontsize=10.5, loc="upper left", frameon=False)
    runs = Q.get("shear", {}).get("runs", {})
    irs = sorted({v["impratio"] for v in runs.values()})
    for k_, (nm, col, mk) in enumerate((("E_FE", "#c0392b", "o"), ("E_meas", "#1f5fbf", "s"))):
        med, lo, hi = [], [], []
        for ir in irs:
            r = runs.get(f"impratio{ir:g}_{nm}")
            v = np.array([np.nan if y is None else y for y in r["per_hold"]], float)
            v = v[np.isfinite(v)]
            med.append(np.median(v))
            lo.append(np.percentile(v, 10))
            hi.append(np.percentile(v, 90))
        xs = np.array(irs) * (0.9 if k_ == 0 else 1.1)
        med, lo, hi = np.array(med), np.array(lo), np.array(hi)
        ax[1].errorbar(xs, med, yerr=[med - lo, hi - med], fmt=mk, color=col, capsize=4, ms=7,
                       label=f"Pads at {Q['delta_' + nm]['E_MPa']:.2f} MPa" if nm == "E_meas" else
                       f"Pads at {Q['delta_' + nm]['E_MPa']:.1f} MPa")
    hold = Q["shear"]["hold_s"]["median"]
    ax[1].axhline(0.1 / hold, color="k", lw=1.0, ls="--", label=f"0.1 mm over a {hold:.1f} s hold")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks(irs)
    ax[1].set_xticklabels([f"{v:g}" for v in irs])
    ax[1].minorticks_off()
    ax[1].set_xlabel("MuJoCo impratio")
    ax[1].set_ylabel("Sliding speed under the measured shear (mm/s)")
    ax[1].set_title("(b) Creep of the pads under the holds' shear")
    ax[1].legend(fontsize=10.5, loc="upper right", frameon=False)
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_rod(R, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.8), dpi=110)
    xm = 0.0
    for nm, fe, col, mk in (("Sphere, radius 5 mm", R["FE"], "#1f5fbf", "o"),
                            (f"Rod, radius {R['ROD']['plan']['geometry']['indenter']['radius']:g} mm", R["ROD"]["FE"],
                             "#c0392b", "s")):
        h, sr = fe["holds"], fe["series"]
        fn = np.array(h["f_n"])
        dl = np.array([np.nan if x is None else x for x in h["delta_mm"]], float)
        ax[0].scatter(fn, dl, s=10, marker=mk, color=col, alpha=0.35)
        x = np.linspace(0, float(np.nanmax(fn)) * 1.02, 50)
        ax[0].plot(x, sr["a_mm"] + sr["c_mm_N"] * x, "-", color=col, lw=2.0,
                   label=f"{nm}: {sr['a_mm']:.2f} mm + {sr['c_mm_N']:.2f} mm/N")
        xm = max(xm, x[-1])
        lv = sorted(fe["by_level"], key=float)
        ax[1].plot([float(k) for k in lv], [fe["by_level"][k]["ratio_median_s1"] for k in lv], marker=mk, color=col, label=nm)
    ax[0].set_xlim(0, xm)
    ax[0].set_ylim(-0.3, 1.6)
    ax[0].set_xlabel("Normal force from the HEX-E (N)")
    ax[0].set_ylabel("Indentation beyond the FE pad (mm)")
    ax[0].set_title("(a) Indentation beyond the FE pad")
    ax[0].legend(fontsize=10.5, loc="upper left", frameon=False)
    ax[1].set_xscale("log")
    ax[1].set_xticks([0.25, 0.5, 1, 2, 5])
    ax[1].set_xticklabels(["0.25", "0.5", "1", "2", "5"])
    ax[1].minorticks_off()
    ax[1].set_ylim(0, 0.6)
    ax[1].set_xlabel("Force level (N)")
    ax[1].set_ylabel("Measured over FE pad (median)")
    ax[1].set_title("(b) Measured force over the FE pad's")
    ax[1].legend(fontsize=10.5, loc="lower right", frameon=False)
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_day(R, out):
    plt = P.plt_setup()
    DR = R["DAY"]["DR"]
    st, snr = DR["same_target"], DR.get("same_night_refs")
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.8), dpi=110)
    srcs = [(st["pairs"], "A day later, same target", "#1f5fbf", "o", -0.15)]
    if snr and snr.get("pairs"):
        srcs.append((snr["pairs"], "Same night, reference press repeated", "#c0392b", "s", 0.15))
    levels = sorted({float(L) for src in srcs for L in src[0]["level_N"]})
    pos = {L: j for j, L in enumerate(levels)}
    rng = np.random.default_rng(0)
    top = 0.0
    for src, lab, col, mk, dx in srcs:
        lv, rel = np.array(src["level_N"], float), np.array(src["board_rel_change"], float) * 100.0
        x = np.array([pos[L] for L in lv]) + dx
        ax[0].scatter(x + rng.uniform(-0.06, 0.06, len(x)), rel, s=22, marker=mk, color=col, alpha=0.6, label=lab)
        for L in levels:
            m = lv == L
            if m.any():
                ax[0].plot([pos[L] + dx - 0.13, pos[L] + dx + 0.13], [np.median(rel[m])] * 2, "-", color=col, lw=2.5)
        top = max(top, float(np.percentile(rel, 98)))
    ax[0].set_xticks(range(len(levels)))
    ax[0].set_xticklabels([f"{L:g}" for L in levels])
    ax[0].set_ylim(0, max(10.0, top * 1.3))                  # room for the legend above the points
    ax[0].set_xlabel("Force level (N)")
    ax[0].set_ylabel("Difference of the board change (%)")
    ax[0].set_title("(a) Board change at the same target")
    ax[0].legend(fontsize=10.5, loc="upper right", frameon=False)
    dl, sn = DR["day_later"], DR.get("same_night", {})
    for k, lab, col in (("gp:board", "Gaussian process", "#1f5fbf"), ("mlp:board", "MLP ensemble", "#8e44ad"),
                        ("physical", "Physical model", "#2e8b57")):
        for src, ls, mk, when in ((dl, "-", "o", "a day later"), (sn, "--", "s", "same night")):
            if k not in src or "by_level" not in src[k]:
                continue
            bl = src[k]["by_level"]
            L = [x for x in sorted(bl, key=float) if bl[x]["loc_mm"]["median"] is not None]
            ax[1].plot([float(x) for x in L], [bl[x]["loc_mm"]["median"] for x in L], ls, marker=mk, ms=5, color=col,
                       lw=1.6, label=f"{lab}, {when}")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    xt = [L for L in (0.25, 0.5, 1, 2, 3, 5) if any(abs(L - v) < 1e-6 for v in levels)] or [0.5, 1, 2, 5]
    ax[1].set_xticks(xt)
    ax[1].set_xticklabels([f"{v:g}" for v in xt])
    ax[1].minorticks_off()
    lo, hi = ax[1].get_ylim()
    ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10) if lo <= t <= hi]
    ax[1].set_yticks(ticks)
    ax[1].set_yticklabels([f"{t:g}" for t in ticks])
    ax[1].set_xlabel("Force level (N)")
    ax[1].set_ylabel("Location error, median (mm)")
    ax[1].set_title("(b) Predictors' location error")
    for a in ax:
        a.grid(alpha=0.3, which="both")
    h, l_ = ax[1].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def own_fits_apart():
    """Median and largest distance (mm) over the pad's surface between sph3t's and sphere2's own registrations."""
    import ur5_tip_calibration as C
    regs = [json.loads((d / "registration.json").read_text())["notes"]["register"] for d in (SPH3T, LAB / "20261022-210852-sphere2")]
    g = C.Geometry(json.loads((LAB / "20261022-210852-sphere2/plan.json").read_text()))
    Ta, Tb = (np.array(r["T_EM"]) @ np.array(r["T_MP"]) for r in regs)
    d = np.linalg.norm(C.apply(Ta, g.pts) - C.apply(Tb, g.pts), axis=1)
    return float(np.median(d)), float(d.max())


def fig_board(R, out):
    plt = P.plt_setup()
    H = R["H"]
    m = press_mask(H) & H["est_contact"].astype(bool)
    fn, ef = H["f_n"][m], H["est_f_n"][m]
    reg, lv = H["region"][m], H["level_N"][m]
    err = tangential_err(H)[m]
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.6), dpi=110)
    for r, (mk, col) in REGION_STYLE.items():
        k = reg == r
        ax[0].scatter(fn[k], ef[k], marker=mk, s=34, c=col, label=r.capitalize())
    top = max(5.0, float(np.nanmax(np.r_[fn, ef])) * 1.05)
    ax[0].plot([0, top], [0, top], "k-", lw=1.0)
    ax[0].set_xlim(0, top)
    ax[0].set_ylim(0, top)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("Normal force from the HEX-E (N)")
    ax[0].set_ylabel("Normal force from the board (N)")
    ax[0].set_title("(a) Board's force against the HEX-E's")
    ax[0].legend(fontsize=11, loc="upper left", frameon=False)
    levels = sorted(set(lv.tolist()))
    rng = np.random.default_rng(0)
    for j, L in enumerate(levels):
        k = lv == L
        for r, (mk, col) in REGION_STYLE.items():
            kk = k & (reg == r)
            ax[1].scatter(j + rng.uniform(-0.18, 0.18, kk.sum()), err[kk], marker=mk, s=26, c=col, alpha=0.8)
        ax[1].plot([j - 0.3, j + 0.3], [np.median(err[k])] * 2, "k-", lw=2.0)
    ax[1].axhline(2.0, color="#999999", lw=1.0, ls="--")
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(levels)))
    ax[1].set_xticklabels([f"{L:g}" for L in levels])
    ax[1].set_xlabel("Force level of the press (N)")
    ax[1].set_ylabel("Location error (mm)")
    ax[1].set_title("(b) Estimator's location error")
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_map(R, out):
    """True contact points of the presses at 1 N and over, with a line to where the estimator puts each."""
    plt = P.plt_setup()
    H = R["H"]
    g = R["plan"]["geometry"]["surface"]
    Ps, Ns = np.array(g["pts"]), np.array(g["normals"])
    m = press_mask(H) & H["est_contact"].astype(bool) & (H["level_N"] >= 0.99)
    p, e, n = H["p"][m], H["est_p"][m], H["n"][m]
    reg = H["region"][m]
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.8), dpi=110)
    views = ((lambda v: v[:, 0] > 0.3, 1, "y across the finger (mm)", "(a) Palmar view"),
             (lambda v: v[:, 1] > 0.3, 0, "x along the face's normal (mm)", "(b) Side view"))
    for j, (sel, a1, lab, title) in enumerate(views):
        side = sel(Ns)
        ax[j].scatter(Ps[side, 2], Ps[side, a1], s=5, c="#d0d0d0")
        k = sel(n)
        for r, (mk, col) in REGION_STYLE.items():
            kk = k & (reg == r)
            for a_, b_ in zip(p[kk], e[kk]):
                ax[j].plot([a_[2], b_[2]], [a_[a1], b_[a1]], "-", color=col, lw=0.9, alpha=0.8)
            ax[j].scatter(p[kk, 2], p[kk, a1], marker=mk, s=30, c=col, label=f"{r.capitalize()}: ground truth" if j == 0 else None)
            ax[j].scatter(e[kk, 2], e[kk, a1], marker="x", s=22, c=col, label=f"{r.capitalize()}: estimator" if j == 0 else None)
        q = np.r_[p[k], e[k]]
        if len(q):
            ax[j].set_xlim(q[:, 2].min() - 1.0, q[:, 2].max() + 1.0)
            ax[j].set_ylim(q[:, a1].min() - 1.0, q[:, a1].max() + 1.0)
        ax[j].set_xlabel("z along the finger toward the tip end (mm)")
        ax[j].set_ylabel(lab)
        ax[j].set_aspect("equal")
        ax[j].grid(alpha=0.3)
        ax[j].set_title(title)
    h, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_predict(R, out):
    plt = P.plt_setup()
    PR = R["PR"]
    cv = PR["curve"]
    N = [r["N"] for r in cv]
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.6), dpi=110)
    for j, (key, ylab, title) in enumerate((("loc_mm", "Location error, median (mm)", "(a) Contact point"),
                                            ("force_N", "Normal force error, median (N)", "(b) Normal force"))):
        ax[j].axhline(PR["viewer"][key]["median"], color="#000000", ls=":", lw=1.5, label="Viewer's estimator, untrained")
        for mk, lab, col, ls in MODELS:
            y = [r[mk][key]["median"] if mk in r else np.nan for r in cv]
            ax[j].plot(N, y, ls, color=col, marker="o", ms=5, lw=1.6, label=lab)
        ax[j].set_xscale("log")
        ax[j].set_yscale("log")
        ax[j].set_xticks(N)
        ax[j].set_xticklabels([str(x) for x in N])
        ax[j].minorticks_off()
        lo, hi = ax[j].get_ylim()
        ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20) if lo <= t <= hi]
        ax[j].set_yticks(ticks)
        ax[j].set_yticklabels([f"{t:g}" for t in ticks])
        ax[j].set_xlabel("Training presses, in time order")
        ax[j].set_ylabel(ylab)
        ax[j].set_title(title)
        ax[j].grid(alpha=0.3, which="both")
    h, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ text
def session_facts(R):
    H, acts = R["H"], R["acts"]
    m = press_mask(H)
    ids = sorted(set(H["id"][m].tolist()))
    by_stage = {}
    for i in ids:
        st = acts[i]["stage"] if i in acts else "?"
        by_stage[st] = by_stage.get(st, 0) + 1
    turns = sorted(set(H["id"][~m].tolist()))
    return dict(n_holds=int(m.sum()), n_press=len(ids), by_stage=by_stage, n_turn_actions=len(turns),
                n_turn_holds=int((~m).sum()), refs=[i for i in ids if acts.get(i, {}).get("role") == "ref"],
                t_first=float(H["t"].min()), t_last=float(H["t"].max()))


def sections(R, F):
    S, FE, PR, H = R["S"], R["FE"], R["PR"], R["H"]
    sf = session_facts(R)
    reg = S["registration"]["register"]
    tcp_mm, tcp_deg = reg["tcp_change_mm_deg"]
    br = S["by_region"]
    fs = {r: br[r]["force_scale"] for r in br}
    fe_r = FE["by_region"]
    sr = FE["series"]
    srr = sr["by_region"]
    lvl = S["estimator_by_level"]
    hi_lv = [k for k in lvl if float(k) >= 1.0 and lvl[k]["n"] >= 3]
    lo_lv = [k for k in lvl if float(k) < 1.0]
    loc_hi = [lvl[k]["loc_median_mm"] for k in hi_lv]
    loc_lo = [lvl[k]["loc_median_mm"] for k in lo_lv]
    rf_hi = [lvl[k]["rel_force_median"] for k in hi_lv if lvl[k]["rel_force_median"] is not None]
    on = [o for o in S["ref_onsets"]["presses"] if o["id"].startswith("ref-")]
    last = PR["curve"][-1] if PR else None
    ref_txt = ""
    if len(on) > 1:
        ref_txt = (f" The grid's reference press was made {len(on)} times, from {hm(on[0]['t'])} to {hm(on[-1]['t'])}. Its "
                   f"onset, the travel at which the push begins, moved by {on[-1]['onset_mm'] - on[0]['onset_mm']:+.3f} mm "
                   f"from {on[0]['id']} to {on[-1]['id']}; the ground truth is not corrected for it.")

    # ---- lede
    k_fe = 1.0 / fe_r["face"]["ratio_median_s1"]
    lede = (f"At the sphere's registered positions in the UR5 run {R['session'].split('-', 2)[-1]}, the pad's "
            f"finite-element (FE) model gives {k_fe:.1f} times the force measured by the HEX-E, as a median on the "
            f"face. The force estimated from the board is {min(fs.values()):.2f} to {max(fs.values()):.2f} times the "
            f"HEX-E's, so the pad itself is not {FE['s_best']:g} times softer than its model. "
            + (f"Under the sphere the presses travel {sr['a_mm']:.2f} mm plus {sr['c_mm_N']:.2f} mm per newton beyond the "
               f"FE pad. Under a rod of radius {R['ROD']['plan']['geometry']['indenter']['radius']:g} mm they travel "
               f"{R['ROD']['FE']['series']['a_mm']:.2f} mm plus {R['ROD']['FE']['series']['c_mm_N']:.2f} mm per newton. So at most "
               f"{R['ROD']['FE']['series']['c_mm_N']:.2f} mm/N lies in series with the pad, and the rest arises at the "
               f"sphere's contact. " if R.get("ROD") else
               f"The extra indentation, {sr['a_mm']:.2f} mm plus {sr['c_mm_N']:.2f} mm per newton, occurs in parts whose "
               f"deformation is not sensed by the board. ")
            + f"From 1 N up, the viewer's estimator places contacts {min(loc_hi):.1f} to {max(loc_hi):.1f} mm "
            f"(median) from the registered points; below 1 N it fails.")
    if last:
        g = last["gp:board"]
        lede += (f" A Gaussian process on the board change, trained on {last['N']} presses, places held-out contacts "
                 f"within {g['loc_mm']['median']:.2f} mm and their force within {g['force_N']['median']:.2f} N. With the "
                 f"chips' temperature change and the time since the rest added, its location error is "
                 f"{last['gp:+time since rest']['loc_mm']['median']:.2f} mm.")
        if PR.get("turns"):
            t = PR["turns"]
            lede += (f" On the rolls and twists, at orientations that no press had, it places contacts within "
                     f"{t['gp:board']['loc_mm']['median']:.2f} mm, against {t['viewer']['loc_mm']['median']:.2f} mm for "
                     f"the untrained estimator.")
        if R.get("CV"):
            cv = R["CV"]
            lede += (f" An ensemble of small MLPs places held-out contacts within {cv['mlp']['loc_median']:.2f} mm over four "
                     f"folds, against {cv['gp']['loc_median']:.2f} mm for the Gaussian process"
                     + (f", but within {PR['turns']['mlp:board']['loc_mm']['median']:.2f} mm on the rolls and twists."
                        if PR.get("turns") and "mlp:board" in PR["turns"] else "."))
    DY_ = R.get("DAY")
    if DY_ and DY_.get("DR") and "gp:board" in DY_["DR"]["day_later"]:
        lede += (f" A day later, the Gaussian process trained on all of the night's presses places the contacts of "
                 f"repeated presses within {DY_['DR']['day_later']['gp:board']['loc_mm']['median']:.2f} mm.")
    Q = R.get("PADS")
    if Q:
        ru = Q.get("shear", {}).get("runs", {})
        lede += (f" The simulations' sphere pads on the same surface, with a spring in series, follow the measured "
                 f"indentation within {Q['delta_E_FE']['law']['rms_mm'] * 1e3:.0f} µm rms, against "
                 f"{sr['rms_mm'] * 1e3:.0f} µm for the FE pad."
                 + (f" Under the holds' shear, MuJoCo's soft friction lets the sphere slide along them at "
                    f"{ru['impratio1_E_FE']['creep_mm_s']['median']:.1f} mm/s at impratio 1 and "
                    f"{ru['impratio100_E_FE']['creep_mm_s']['median']:.3f} mm/s at impratio 100."
                    if "impratio1_E_FE" in ru and "impratio100_E_FE" in ru else ""))
    summary = (f"UR5 run {R['session'].split('-', 2)[-1]}: measured force is {fe_r['face']['ratio_median_s1']:.2f} of the FE "
               f"pad's at the registered positions while the board reads {min(fs.values()):.2f}-{max(fs.values()):.2f} of "
               f"the HEX-E's force; extra indentation {sr['a_mm']:.2f} mm + {sr['c_mm_N']:.2f} mm/N under the sphere"
               + (f", {R['ROD']['FE']['series']['a_mm']:.2f} mm + {R['ROD']['FE']['series']['c_mm_N']:.2f} mm/N under the rod"
                  if R.get("ROD") else "") + "; "
               f"estimator {min(loc_hi):.1f}-{max(loc_hi):.1f} mm from 1 N"
               + (f"; GP on {last['N']} presses {last['gp:board']['loc_mm']['median']:.2f} mm, "
                  f"{last['gp:board']['force_N']['median']:.2f} N on held-out presses" if last else "")
               + (f", {PR['turns']['gp:board']['loc_mm']['median']:.2f} mm on rolls and twists" if PR and PR.get("turns") else "")
               + (f"; trained on all its presses, {R['DAY']['DR']['day_later']['gp:board']['loc_mm']['median']:.2f} mm on "
                  f"presses a day later." if R.get("DAY") and "gp:board" in R["DAY"]["DR"]["day_later"] else "."))

    S_ = []
    # ---- 1 terms
    terms = [
        ("Press and hold", "A press moves the pad onto the indenter, a sphere of radius 5 mm, along the normal at a target "
         "point of the pad. It stops at force levels between 0.25 and 5 N and keeps each level for about 1.5 s. Each hold "
         "is one sample of the ground truth and of the board."),
        ("Region", "The pad's surface is divided by the direction of its normal: the face looks toward the palm, the "
         "flanks look across the finger, and the edges lie between them."),
        ("Ground truth", "The contact point $\\mathbf{p}$ and the normal $\\mathbf{n}$ on the pad follow from the flange's "
         "pose, the registration of the tool and of the sphere's centre, and the sphere's radius. The normal force $f_n$ "
         "is the HEX-E's force along $-\\mathbf{n}$, in N, with the tool's weight and the drift of the zero removed. "
         "The arm, the tool's parts and the indenter's mount are taken as rigid. The HEX-E's own deflection is removed with "
         "its data sheet's compliance."),
        ("Registered indentation $d$", "The depth of the sphere inside the pad's nominal surface at the registered pose, in mm."),
        ("FE pad", "The finite-element model of the printed pad, pressed by a frictionless sphere. Its surface compliance "
         "is that of DIP link b, case b_v2, from " + F["code"]("scripts/fingertip_wrench_fe.py --surface") + ". The "
         "contact forces are non-negative and solved by least squares. The scale $s$ multiplies the compliance; $s = 1$ is "
         "the model as computed."),
        ("Measured over FE", "$f_n$ divided by the FE pad's normal force with the sphere at its registered position, at "
         "$s = 1$."),
        ("Indentation beyond the FE pad $\\Delta$", "The distance in mm that the sphere must move back along "
         "$\\mathbf{n}$ before the force of the FE pad at $s = 1$ equals the measured $f_n$. It is fitted as "
         "$\\Delta = a + c\\,f_n$, with $a$ an offset at zero force (mm) and $c$ a compliance in series with the pad (mm/N)."),
        ("Board change", "The mean field of the board's five three-axis magnetometer chips over a hold, minus the field at "
         "rest: 15 values in µT. The field at rest is interpolated in time between the rests before and after the press, "
         "and the chips' temperature drift is corrected first."),
        ("Viewer's estimator", "The surface estimator of the contact viewer (" + F["link"](EST_PAGE, "contact estimation page")
         + "). The magnets' strength $k$ and the chips' offsets are fitted to the field at rest before the first reference "
         "press. A board change is then mapped to a contact point and a normal force through the FE model's magnet motions."),
        ("Location error", "The distance between an estimated and the true contact point, tangential to the pad, in mm."),
        ("Force error", "$|\\hat f_n - f_n|$ in N, where $\\hat f_n$ is the estimated normal force; the relative force "
         "error is $|\\hat f_n / f_n - 1|$."),
        ("Force scale", "The least-squares slope of the estimator's normal force against $f_n$ over a region's grid "
         "presses. At a value of 1 the estimated force equals the HEX-E's on average."),
        ("Sphere pads", "The contact model of the simulations: spheres of radius 0.75 mm, one per square millimetre of "
         "the tip's surface. Each is a spring of stiffness $K_i = E\\,a_i / h_i$ (builder v2), with $a_i$ its area and "
         "$h_i$ its depth below the centroid of the pad's solid. Here they sit on the 812 surface points of the FE basis."),
        ("Pad modulus $E$", "The modulus in $K_i$, in MPa. $E_\\mathrm{FE}$ fits the pads' force to the FE pad's at the "
         "registered positions, by least squares of the log ratio. $E_\\mathrm{meas}$ fits it to the HEX-E's force, by the "
         "median relative error. The simulations' default is 10 MPa."),
        ("Creep", "The sphere's speed along the pad's surface in MuJoCo, in mm/s, over the second half of 1 s. A constant "
         "force equal to a hold's measured force vector pushes the sphere, and the friction coefficient is 1."),
        ("Predictor, training presses $N$", "A model trained on the holds of the first $N$ presses in time order and "
         "scored on held-out presses (every fourth grid press, apart from the reference presses). Each score is the "
         "median over the held-out holds."),
    ]
    if R.get("DAY"):
        terms += [
            ("Difference of the board change", "For two holds at the same press target and force level, "
             "$|\\mathbf{b}_2 - \\mathbf{b}_1| / |\\mathbf{b}_1|$ in %, where $\\mathbf{b}$ is a hold's board change. "
             "A repeat of the reference press is paired with the first reference press of sphere2."),
            ("Indentation change", "$d_2 - d_1$ in µm, the registered indentation of the same pairs of holds.")]
    S_.append(("terms", "Terms and metrics",
               '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"))

    # ---- 2 session
    stages = ", ".join(f"{v} {k}" for k, v in sorted(sf["by_stage"].items()))
    ab_st = {}
    for i in R["abort"]:
        k = R["acts"].get(i, {}).get("stage", "other")
        ab_st[k] = ab_st.get(k, 0) + 1
    rows = [["Touch-offs fitted", f"{reg['n_pad']}"], ["Pad residual, rms (µm)", f"{reg['rms_pad_mm'] * 1e3:.0f}"],
            ["Tool centre point moved by (mm, °)", f"{tcp_mm:.2f}, {tcp_deg:.2f}"],
            ["Sphere's centre moved by (mm)", f"{reg['centre_change_mm']:.2f}"],
            ["Press holds analysed", f"{sf['n_holds']} from {sf['n_press']} presses ({stages})"],
            ["Turn holds (slides, rolls, twists)", f"{sf['n_turn_holds']} from {sf['n_turn_actions']} actions"],
            ["Aborted actions", f"{len(R['abort'])} (" + ", ".join(f"{k} {v}" for k, v in sorted(ab_st.items())) + ")"]]
    exc = S.get("excluded", {}).get("actions", [])
    if exc:
        rows.append(["Presses left out of the ground truth", f"{len(exc)}"])
    body = (f"<p>The run {P.esc(R['session'].split('-', 2)[-1])} started at {hm(R['t0'])}; its last event in this copy "
            f"is from {hm(R['t1'])}. "
            + (f"Its pad touch-offs were fitted together with those of two sessions of 2026-10-23, on the rod and on the "
               f"sphere, to one registration of the tool (Table 1, Section 8). "
               if reg.get("joint_sessions") else "Its pad touch-offs were fitted to a registration of the tool and the sphere (Table 1). ")
            + f"The registration moved the tool centre point by {tcp_mm:.2f} mm and {tcp_deg:.2f}° from the CAD and homing values. "
            + (f"With the session's own registration of 2026-10-22, the extra indentation of Section 3 was "
               f"{R['FE_OWN']['series']['a_mm']:.2f} mm + {R['FE_OWN']['series']['c_mm_N']:.2f} mm/N, with a residual of "
               f"{R['FE_OWN']['series']['rms_mm'] * 1e3:.0f} µm rms. " if R.get("FE_OWN") else "")
            + f"The run itself, its restarts and the HEX-E's link are described on the "
            + (F["link"](LAB_PAGE, "page of the lab runs") if LAB_PAGE.exists() else "page of the lab runs (not built yet)")
            + f"; the protocol is on the {F['link'](PROTOCOL_PAGE, 'protocol page')}.{ref_txt}</p>"
            + F["table"](["Quantity", "Value"], rows, num_cols=(1,), caption="Registration and data of the session.", n=1)
            + (f"<p>The presses {ands(exc)} are left out of the ground truth. In each of them the HEX-E measured 0.3 to "
               f"0.4 N before the pad met the sphere, rising with the travel. That load did not show on the board. They "
               f"are flank presses at the most turned wrist of the grid, with joint 6 at 70 to 72°. The lab traced the "
               f"load to the HEX-E's cable or a graze of the tool (its message stamped 23:50).</p>" if exc else "")
            + "<p>The HEX-E's gravity fit was taken from the run first2 (19:54), since this run had no gravity stage. "
            "The zero of each hold is interpolated in time between the rests before and after its press.</p>")
    S_.append(("session", "Session and ground truth", body))

    # ---- 3 FE
    lv_rows = []
    fe_lv = FE["by_level"]
    for k in sorted(fe_lv, key=float):
        mm = np.array(FE["holds"]["level_N"]) == float(k)
        lv_rows.append([f"{float(k):g}", f"{fe_lv[k]['n']}", f1(float(np.median(np.array(FE['holds']['f_n'])[mm]))),
                        f1(float(np.median(np.array(FE['holds']['depth_mm'])[mm]))),
                        f1(float(np.median(np.array(FE['holds']['fe_s1_N'])[mm]))), f1(fe_lv[k]["ratio_median_s1"])])
    hi_r = [v["ratio_median_s1"] for k, v in fe_lv.items() if float(k) >= 1.0 and v["n"] >= 3]
    lo_k = min(fe_lv, key=float)
    s1 = np.array(FE["holds"]["fe_s1_N"])
    miss = np.nonzero(s1 <= 0)[0]
    miss_txt = ""
    if len(miss):
        dm = np.array(FE["holds"]["depth_mm"])[miss]
        mr = sorted(set(np.array(FE["holds"]["region"])[miss].tolist()))
        ml = sorted(set(np.array(FE["holds"]["level_N"])[miss].tolist()))
        miss_txt = (f" At {len(miss)} holds ({ands([RNAME[r] for r in mr])}, "
                    f"{ands([f'{x:g}' for x in ml])} N) the registered sphere stands {-dm.max():.2f} to {-dm.min():.2f} mm "
                    f"clear of the pad's surface, although the HEX-E measured contact; their ratio is left out.")
    body = (f"<p>At the registered sphere positions, the force of the FE pad is {k_fe:.1f} times the measured force on "
            f"the face (Fig. 1a, Table 2). The median of measured over FE is {fe_r['face']['ratio_median_s1']:.2f} on the "
            f"face, {fe_r['edge']['ratio_median_s1']:.2f} on the edges and {fe_r['flank']['ratio_median_s1']:.2f} on the "
            f"flanks. It is {fe_lv[lo_k]['ratio_median_s1']:.2f} at {float(lo_k):g} N and {min(hi_r):.2f} to "
            f"{max(hi_r):.2f} from 1 N up.{miss_txt} "
            f"The compliance scale that fits best is $s = {FE['s_best']:g}$, with a median force error of "
            f"{FE['median_rel_err_best'] * 100:.0f} %, against {FE['median_rel_err_s1'] * 100:.0f} % at $s = 1$.</p>"
            f"<p>With the FE pad kept at $s = 1$, the sphere must be moved back along the normal by $\\Delta$ to give the "
            f"measured force. Over {sr['n']} holds, $\\Delta = {sr['a_mm']:.3f} + {sr['c_mm_N']:.3f}\\,f_n$ "
            f"(mm, with $f_n$ in N), with a residual of {sr['rms_mm'] * 1e3:.0f} µm rms (Fig. 1b). "
            f"Through zero the slope is {sr['c0_mm_N']:.3f} mm/N and the residual {sr['rms0_mm'] * 1e3:.0f} µm. "
            + " ".join(f"On the {RNAME[r]} the fit is {srr[r]['fit'][0]:.2f} mm + {srr[r]['fit'][1]:.3f} mm/N"
                       + (f", with a standard error of {srr[r]['se'][1]:.3f} mm/N on the slope." if "se" in srr[r] else ".")
                       for r in RNAME if r in srr)
            + "</p>"
            + P.table(["Level (N)", "Holds", "Measured force (N)", "Registered indentation (mm)", "FE force at s = 1 (N)",
                       "Measured over FE"], lv_rows, num_cols=(0, 1, 2, 3, 4, 5),
                      caption="Measured and FE forces by force level (medians over the holds).", n=2)
            + F["fig"](1, MEDIA / "20261022-sphere2_fe_against_hexe.png",
                       f"Pad compliance against the FE model. (a) Normal force of the FE pad with the sphere at its "
                       f"registered position, against the HEX-E's normal force. Open markers are the FE model as computed, "
                       f"filled markers the model with its compliance scaled by {FE['s_best']:g}; the line is equality. "
                       f"(b) Indentation beyond the FE pad against the HEX-E's normal force, with the fitted line."))
    S_.append(("fe", "Pad compliance against the finite-element model", body))

    # ---- 4 board and estimator
    bm = S.get("board_minus_registration_mm")
    rp = [x["change_uT"] for x in S.get("reference_press_1N", [])]
    est_rows = []
    for k in sorted(lvl, key=float):
        v = lvl[k]
        est_rows.append([f"{float(k):g}", f"{v['n']}", f1(v["loc_median_mm"]), f1(v["loc_p90_mm"]),
                         f"{v['far_2mm'] * 100:.0f}", f1(v["rel_force_median"])])
    body = (f"<p>Against the HEX-E's normal force, the estimator's normal force has a slope of {fs['face']:.2f} on the "
            f"face, {fs['edge']:.2f} on the edges and {fs['flank']:.2f} on the flanks (Fig. 2a). In a pad {FE['s_best']:g} "
            f"times softer than its FE model, the magnets would move {FE['s_best']:g} times as far per newton. The estimated "
            f"force would then be about {FE['s_best']:g} times the measured. The pad's bulk is therefore within about "
            f"{(1 - min(fs.values())) * 100:.0f} % of the FE model's compliance. The extra indentation of Section 3 occurs "
            f"where the magnets do not move against the chips.</p>"
            f"<p>On the tool's side, the pad is in series with the DIP link's clamp in the fork, the printed mount, the "
            f"HEX-E and the arm's joints. The arm's pose comes from its joint encoders; a deflection under load that the "
            f"encoders do not report adds to the registered depth. On the indenter's side are the sphere's block, the 2x4, its clamps and the "
            f"table. A compliance at the printed surface that the FE mesh of 1 mm does not resolve would also leave the "
            f"magnets still. Every press pushes the sphere down along the vertical, so the indenter's side alone would give "
            f"one slope in every region. The slopes are "
            + ands([f"{srr[r]['fit'][1]:.3f} mm/N on the {RNAME[r]}" for r in RNAME if r in srr])
            + ", with standard errors of "
            + ands([f"{srr[r]['se'][1]:.3f}" for r in RNAME if r in srr and "se" in srr[r]])
            + " mm/N. The direction of the load in the tool's frame changes with the region, so a compliance on the tool's "
            "side would make the slopes differ. A regional error of the FE model would make them differ as well.</p>"
            f"<p>From 1 N up, contacts are placed by the estimator within {min(loc_hi):.1f} to {max(loc_hi):.1f} mm "
            f"(median; Table 3, Fig. 2b). Its force is within {min(rf_hi) * 100:.0f} to {max(rf_hi) * 100:.0f} % of the "
            f"HEX-E's. Below 1 N the median location error is "
            + (f"{min(loc_lo):.1f} to {max(loc_lo):.1f} mm. " if round(min(loc_lo), 1) != round(max(loc_lo), 1)
               else f"{min(loc_lo):.1f} mm. ")
            + (f"At 3 N and over, the mean location error over {bm['n']} holds is ({bm['mean'][0]:.2f}, "
               f"{bm['mean'][1]:.2f}, {bm['mean'][2]:.2f}) mm in the pad's frame. Its standard errors are "
               f"{bm['se'][0]:.2f}, {bm['se'][1]:.2f} and {bm['se'][2]:.2f} mm. The offset of "
               f"{np.linalg.norm(bm['mean']):.1f} mm is therefore systematic. Either the registered contact points or the "
               f"model's magnets are displaced by about that much. " if bm else "")
            + "Fig. 3 shows where the estimates lie on the pad.</p>"
            + (f"<p>Between the rests before and after a press, the board's field at rest changes by "
               f"{S['rest_drift_uT']['median']:.1f} µT (median; 90th percentile {S['rest_drift_uT']['p90']:.1f} µT). "
               f"The six reference presses at 1 N changed it by {min(rp):.0f} to {max(rp):.0f} µT. Below 1 N the change "
               f"of a press is about as large as that of its rests, and there the estimator fails.</p>" if rp else "")
            + P.table(["Level (N)", "Holds", "Location error, median (mm)", "Location error, 90th percentile (mm)",
                       "Holds over 2 mm (%)", "Relative force error, median"], est_rows, num_cols=(0, 1, 2, 3, 4, 5),
                      caption="The viewer's estimator against the ground truth, by force level.", n=3)
            + F["fig"](2, MEDIA / "20261022-sphere2_board_against_hexe.png",
                       "The board against the HEX-E. (a) Normal force of the viewer's estimator against the HEX-E's, by "
                       "region; the line is equality. (b) Location error of the estimator by the press's force level; the "
                       "bars are medians and the dashed line marks 2 mm.")
            + F["fig"](3, MEDIA / "20261022-sphere2_estimates_on_the_pad.png",
                       "Contact points of the presses at 1 N and over on the pad. Filled markers are the ground truth, "
                       "crosses the estimator, and a line joins the two; grey points are the pad's surface. (a) Points "
                       "whose normal looks toward the palm; (b) points whose normal looks across the finger."))
    S_.append(("board", "Board response and the contact estimator", body))

    # ---- 5 pivot
    pc = S.get("pivot_check")
    if pc and pc.get("presses"):
        rows = [[p["id"], p["role"], f"{p['twist']:g}", f"{p['offset_P'][0]:+.1f}, {p['offset_P'][1]:+.1f}",
                 f"{p['dp_board'][0]:+.2f}, {p['dp_board'][1]:+.2f}", f"{p['resid_uT']:.1f}"] for p in pc["presses"]]
        offs = [p for p in pc["presses"] if p["role"] == "offset"]
        tw = [p for p in pc["presses"] if p["role"] == "turn"]
        body = ("<p>In the pivot stage the pad is pressed at the same point of the sphere with the tool offset by ±0.5 mm "
                "along two directions of the face. One more press is turned about the normal. A linear map from position "
                f"to board change is fitted to the four offset presses at {pc['level_N']:g} N, and each press is placed "
                f"with it. The offset presses are placed about two centres "
                f"{np.linalg.norm(np.array(offs[0]['board_minus_registration']) - np.array(offs[-1]['board_minus_registration'])):.2f} mm apart. "
                + (f"The press turned by {tw[0]['twist']:g}° is placed {np.linalg.norm(tw[0]['dp_board']):.1f} mm from "
                   f"the reference. Its fit residual is {tw[0]['resid_uT']:.1f} µT, against "
                   f"{min(p['resid_uT'] for p in offs):.1f} to {max(p['resid_uT'] for p in offs):.1f} µT for the offsets. "
                   "A turned press is not described by the map from four offsets, so the registration is not tested by "
                   "this turn (Table 4)." if tw else "")
                + "</p>"
                + P.table(["Press", "Role", "Turn (°)", "Offset (mm)", "Placed by the board (mm)", "Fit residual (µT)"],
                          rows, num_cols=(2, 3, 4, 5), caption="The pivot presses as placed by the board's linear map.", n=4))
        S_.append(("pivot", "Pivot check of the registration", body))

    # ---- 6 predictors
    if PR:
        rows = []
        v = PR["viewer"]
        rows.append(["Viewer's estimator, untrained", "Board", f1(v["loc_mm"]["median"]), f1(v["force_N"]["median"], 3)])
        rows.append(["Physical model, compliance fitted", "Board", f1(last["physical"]["loc_mm"]["median"]),
                     f1(last["physical"]["force_N"]["median"], 3)])
        for st in STEPS:
            for lab, key in (("Gaussian process", "gp"), ("MLP ensemble", "mlp"), ("Ridge regression", "ridge"),
                             ("Physical model and Gaussian process", "physical+gp")):
                e = last.get(f"{key}:{st}")
                if e:
                    rows.append([lab, STEP_TEXT[st],
                                 f1(e["loc_mm"]["median"]), f1(e["force_N"]["median"], 3)])
        mk = press_mask(H) & H["est_contact"].astype(bool) & (H["level_N"] >= 0.99)
        med_hi = float(np.median(tangential_err(H)[mk]))
        names = {"physical": "the physical model", "gp:board": "the GP", "ridge:board": "ridge regression",
                 "physical+gp:board": "the physical model with the GP"}
        best_f = min(((names[k], last[k]["force_N"]["median"]) for k in names), key=lambda x: x[1])
        cvals = [r["physical_params"]["c"] for r in PR["curve"]]
        body = (f"<p>The owner asked at 22:10 how much each additional input improves the prediction. "
                f"Of the grid presses, {PR['n_test_presses']} ({PR['n_test_holds']} holds) are held out. The models are "
                f"trained on the first $N$ of the other {PR['n_train_pool']} presses in time order (Fig. 4). "
                f"The inputs are added one at a time: the board change, the chips' temperature change since the press's "
                f"rest, and the time since that rest (Table 5).</p>"
                f"<p>Four models are compared. The physical model is the viewer's estimator with a compliance scale $c$ "
                f"fitted to the training holds. In presses, $k$ and $c$ trade against each other. The magnets' strength "
                f"and the chips' offsets are therefore kept near the rest's fit by priors of 5 % and 0.2 mm. A Gaussian "
                f"process (GP) with a squared-exponential kernel maps the inputs to the contact point and $f_n$; ridge "
                f"regression is its linear baseline. The last model adds a zero-mean GP to the physical model's residual. "
                f"That GP is trained on the holds that the physical model places within 2 mm.</p>"
                f"<p>On all {last['N']} training presses, the GP on the board change places the held-out contacts within "
                f"{last['gp:board']['loc_mm']['median']:.2f} mm and their force within "
                f"{last['gp:board']['force_N']['median']:.3f} N (medians). On the same holds, the errors of the untrained "
                f"estimator are {v['loc_mm']['median']:.2f} mm and {v['force_N']['median']:.3f} N. "
                f"The physical model's compliance scale is fitted between {min(cvals):.2f} and {max(cvals):.2f}, and its "
                f"location error is {last['physical']['loc_mm']['median']:.2f} mm. This scale is below the force scale of "
                f"Section 4, {min(fs.values()):.2f} to {max(fs.values()):.2f}. In the fit the contact points are the "
                f"registered ones, which lie a median of {med_hi:.1f} mm from the estimator's contacts at 1 N and over.</p>"
                f"<p>For the normal force, the smallest error with the board alone is that of "
                f"{best_f[0]}, {best_f[1]:.3f} N. With the temperature change added, the GP's location error is "
                f"{last['gp:+temperature change']['loc_mm']['median']:.2f} mm; with the time since the rest as well, it is "
                f"{last['gp:+time since rest']['loc_mm']['median']:.2f} mm. With the board alone it is "
                f"{last['gp:board']['loc_mm']['median']:.2f} mm.</p>"
                f"<p>The tool's orientation is tried apart. In this protocol each target point is pressed with its own tool "
                f"orientation, so the orientation identifies the target. A model given it reproduces the training targets: "
                f"with orientation, the GP's location error is "
                f"{last['gp:+orientation (leaks the target)']['loc_mm']['median']:.2f} mm. The absolute temperature and the "
                f"clock time follow the grid's order in the same way and are left out.</p>"
                + P.table(["Model", "Inputs", "Location error (mm)", "Normal force error (N)"], rows, num_cols=(2, 3),
                          caption=f"Predictors trained on {last['N']} presses ({last['holds']} holds), scored on the "
                                  f"{PR['n_test_presses']} held-out presses (medians).", n=5)
                + F["fig"](4, MEDIA / "20261022-sphere2_predictor_learning_curves.png",
                           "Median errors on the held-out presses against the number of training presses, for the "
                           "board change as the only input. (a) Location error; (b) normal force error. The dotted line "
                           "is the untrained estimator."))
        # by force level, on the held-out presses
        keys = (("viewer", "Viewer's estimator, untrained"), ("gp:board", "Gaussian process"), ("mlp:board", "MLP ensemble"),
                ("physical+gp:board", "Physical model and Gaussian process"))
        src = dict(viewer=v, **{k: last[k] for k, _ in keys if k != "viewer"})
        levels = sorted(v.get("by_level", {}), key=float)
        if levels:
            lrows = [[f"{float(L):g}"] + [f1(src[k]["by_level"].get(L, {}).get("loc_mm", {}).get("median")) for k, _ in keys]
                     for L in levels]
            gl = [src["gp:board"]["by_level"][L]["loc_mm"]["median"] for L in levels if float(L) < 1.0
                  and src["gp:board"]["by_level"].get(L, {}).get("loc_mm", {}).get("median") is not None]
            vl = [v["by_level"][L]["loc_mm"]["median"] for L in levels if float(L) < 1.0
                  and v["by_level"].get(L, {}).get("loc_mm", {}).get("median") is not None]
            body += ((f"<p>Below 1 N the untrained estimator's median location error is {min(vl):.1f} to {max(vl):.1f} mm "
                      f"on the held-out presses. The GP's is {min(gl):.1f} to {max(gl):.1f} mm there (Table 6).</p>"
                      if gl and vl else "")
                     + P.table(["Level (N)"] + [lab for _, lab in keys], lrows, num_cols=tuple(range(len(keys) + 1)),
                               caption="Location error (mm, median) on the held-out presses by force level, with the "
                                       "board change as the input.", n=6))
        tv = PR.get("turns")
        if tv:
            kinds = sorted(tv["viewer"].get("by_kind", {}))
            tkeys = (("viewer", "Viewer's estimator, untrained"), ("physical", "Physical model, compliance fitted"),
                     ("gp:board", "Gaussian process, board"), ("mlp:board", "MLP ensemble, board"),
                     ("ridge:board", "Ridge regression, board"),
                     ("physical+gp:board", "Physical model and Gaussian process, board"),
                     ("gp:+orientation (leaks the target)", "Gaussian process, board and orientation"))
            trows = [[lab, f1(tv[k]["loc_mm"]["median"]), f1(tv[k]["force_N"]["median"], 3)]
                     + [f1(tv[k].get("by_kind", {}).get(kd, {}).get("loc_mm", {}).get("median")) for kd in kinds]
                     for k, lab in tkeys if k in tv]
            kind_name = {"slide": "slides", "roll": "rolls", "twist": "twists"}
            body += (f"<p>The {ands([kind_name.get(k, k) for k in kinds])} test the predictors at tool orientations that "
                     f"no press has. Every press trains them here, and the {tv['n_holds']} holds of "
                     f"{tv['n_actions']} turn actions are scored (Table 7). The GP on the board places these contacts "
                     f"within {tv['gp:board']['loc_mm']['median']:.2f} mm, and the physical model within "
                     f"{tv['physical']['loc_mm']['median']:.2f} mm. With the orientation added, the GP's error is "
                     f"{tv['gp:+orientation (leaks the target)']['loc_mm']['median']:.2f} mm.</p>"
                     + P.table(["Model", "Location error (mm)", "Normal force error (N)"]
                               + [f"Location error, {kind_name.get(k, k)} (mm)" for k in kinds], trows,
                               num_cols=tuple(range(1, 3 + len(kinds))),
                               caption="Predictors trained on every press and scored on the turn holds (medians).", n=7))
        cv = R.get("CV")
        if cv and "mlp:board" in last:
            m, g = last["mlp:board"], last["gp:board"]
            d = cv["mlp_minus_gp"]
            body += (f"<p>An ensemble of five small multilayer perceptrons (MLP) was trained on the same inputs. Each has two "
                     f"hidden layers of 64 units and weight decay; its number of training steps is chosen on the latest fifth "
                     f"of the training presses. On the {PR['n_test_presses']} held-out presses it places contacts within "
                     f"{m['loc_mm']['median']:.2f} mm and their force within {m['force_N']['median']:.3f} N, against "
                     f"{g['loc_mm']['median']:.2f} mm and {g['force_N']['median']:.3f} N for the GP. "
                     + ("The four-fold comparison and the studies below it use the ground truth of the session's own "
                        "registration of 2026-10-22. " if R.get("older") else "")
                     + f"Over four folds, in which "
                     f"each of the {d['presses']} grid presses is held out once, the MLP gives {cv['mlp']['loc_median']:.2f} mm "
                     f"against {cv['gp']['loc_median']:.2f} mm. The difference is {d['loc_median_mm']:.2f} mm (95 % interval "
                     f"{d['loc_ci95'][0]:.2f} to {d['loc_ci95'][1]:.2f}, bootstrap over presses), and "
                     f"{d['force_median_N']:.3f} N in force ({d['force_ci95'][0]:.3f} to {d['force_ci95'][1]:.3f}). From 1 N up "
                     f"the MLP gives {cv['mlp']['loc_median_from_1N']:.2f} mm and the GP {cv['gp']['loc_median_from_1N']:.2f} mm; "
                     f"below 1 N they give {cv['mlp']['loc_median_below_1N']:.2f} and {cv['gp']['loc_median_below_1N']:.2f} mm. "
                     + (f"On the rolls and twists the MLP is worse: {tv['mlp:board']['loc_mm']['median']:.2f} mm against "
                        f"{tv['gp:board']['loc_mm']['median']:.2f} mm for the GP.</p>" if tv and "mlp:board" in tv else "</p>"))
        TR = R.get("TR")
        if TR:
            sc = TR["scores"]
            keys = (("physical", "Physical model, rebuilt for each set, untrained"),
                    ("gp raw (trained on A)", "GP on the board change, trained on set A"),
                    ("physical+gp (trained on A)", "Physical model with a GP correction trained on set A"))
            ks = [k_ for k_ in ("A", "B", "C", "D") if k_ in sc]
            new_ = [k_ for k_ in ks if k_ != "A"]
            trow = [[lab] + [f"{sc[k_][key]['loc_mm']:.2f}, {sc[k_][key]['force_N']:.3f}" for k_ in ks] for key, lab in keys]
            trow.append(["GP on the board change, retrained on each set"]
                        + [f"{sc[k_]['gp raw (retrained on ' + k_ + ')']['loc_mm']:.2f}, {sc[k_]['gp raw (retrained on ' + k_ + ')']['force_N']:.3f}"
                           for k_ in ks])
            pol = {k_: "".join("+" if q > 0 else "-" for q in TR["sets"][k_]["poles"]) for k_ in ks}
            setname = {"A": f"Set A has today's magnets (k {TR['sets']['A']['k']:.3f}).",
                       "B": "Set B has N52 magnets with the same poles (k 1.0).",
                       "C": f"Set C has N52 magnets with the centre one against the four around it ({pol.get('C', '')}).",
                       "D": f"Set D has N52 magnets in the checkerboard printed on 2026-10-23: the centre and one diagonal "
                            f"against the other diagonal ({pol.get('D', '')})."}
            vals = lambda key: ands([f"{sc[k_][key]['loc_mm']:.2f}" for k_ in new_]) + f" mm on sets {ands(new_)}"  # noqa: E731
            body += (f"<p>The owner asked whether these predictors transfer to stronger N52 magnets in a checkerboard of poles. "
                     f"This was tried in simulation on sphere2's mechanics. Each hold keeps its true contact point and force. "
                     f"The board's change is synthesized from the magnets' motions in the FE model and their dipole fields. "
                     f"The simulated pad differs from the FE model in two ways that do not depend on the magnets. Its compliance "
                     f"is {TR['c_true']:g} times the FE's, and it responds like a point {np.linalg.norm(TR['delta_mm']):.1f} mm away. "
                     + "In the poles, + is north toward the palm. " + " ".join(setname[k_] for k_ in ks) + " "
                     f"Every learned model was trained on set A only (Table 8).</p>"
                     f"<p>The physical model is rebuilt for each set from its magnets' strength and poles, which the rest fit "
                     f"finds; the stronger magnets alone take it from {sc['A']['physical']['loc_mm']:.1f} to "
                     f"{sc['B']['physical']['loc_mm']:.1f} mm. A GP on the raw board change does not transfer: "
                     f"{vals('gp raw (trained on A)')}. A GP that corrects the physical estimate, with the estimate itself as its "
                     f"input, does transfer: {vals('physical+gp (trained on A)')}. A GP retrained on each new set gives "
                     + ands([f"{sc[k_]['gp raw (retrained on ' + k_ + ')']['loc_mm']:.2f}" for k_ in new_]) + " mm. "
                     f"Errors that do depend on the magnets, such as a magnet seated off its place, are not in this simulation "
                     f"and would not transfer ({code('scripts/fingertip_magnet_transfer_sim.py')}).</p>"
                     + (f"<p>A third input was tried: the magnets' translations, solved from the board change through each "
                        f"set's own field model. A GP on them gives "
                        + ands([f"{sc[k_]['gp on magnet motions (trained on A)']['loc_mm']:.2f}" for k_ in ks])
                        + f" mm on sets {ands(ks)}. Solving for translations alone amplifies the noise of today's weak magnets "
                        f"and leaves out the magnets' turns.</p>"
                        if "gp on magnet motions (trained on A)" in sc["A"] else "")
                     + P.table(["Predictor"] + [{"A": "Set A: today's magnets (mm, N)", "B": "Set B: N52 (mm, N)",
                                                 "C": "Set C: N52, centre against four (mm, N)",
                                                 "D": "Set D: N52, checkerboard (mm, N)"}[k_] for k_ in ks],
                               trow, num_cols=tuple(range(1, len(ks) + 1)), caption="Median location and normal force errors "
                               "on the held-out presses, simulated board changes on sphere2's mechanics.", n=8))
            HY = R.get("HY")
            if HY:
                hrow = [[lab, f"{HY[k]['loc_median']:.2f}", f"{HY[k]['loc_median_from_1N']:.2f}", f"{HY[k]['loc_median_below_1N']:.2f}",
                         f"{HY[k]['force_median']:.3f}"]
                        for k, lab in (("physical", "Physical model, untrained"), ("gp raw", "GP on the board change"),
                                       ("mlp raw", "MLP on the board change"), ("physical+gp", "Physical model with a GP correction"),
                                       ("physical+mlp", "Physical model with an MLP correction"))]
                body += (f"<p>On the real presses the correction that carries over costs accuracy. Over the four folds, its "
                         f"median location error is {HY['physical+gp']['loc_median']:.2f} mm. The MLP on the board change gets "
                         f"{HY['mlp raw']['loc_median']:.2f} mm (Table 9). Below "
                         f"1 N it gives {HY['physical+gp']['loc_median_below_1N']:.2f} mm, because the physical estimate there "
                         f"is {HY['physical']['loc_median_below_1N']:.1f} mm off and the correction sees only that estimate. "
                         f"After a change of magnets, the corrected physical model works from a rest fit alone. A short session "
                         f"of presses that retrains the MLP brings back the better figure; the learning curve above puts that at "
                         f"about 40 presses ({code(ROOT / 'scripts/fingertip_hybrid_cv.py')}).</p>"
                         + P.table(["Predictor", "Location (mm)", "From 1 N (mm)", "Below 1 N (mm)", "Force (N)"], hrow,
                                   num_cols=(1, 2, 3, 4), caption="Real presses of sphere2, four folds (each grid press held "
                                   "out once), medians.", n=9))
        S_.append(("predict", "Predictors of the contact point and the normal force", body))

    # ---- sphere pads
    Q = R.get("PADS")
    if Q:
        dF, dM, dT = Q["delta_E_FE"], Q["delta_E_meas"], Q["delta_E_TPU"]
        rq, lq, loc, m3 = Q["ratio_meas"], Q["by_level"], Q["location"], Q.get("M3")
        k0, k1 = min(lq, key=float), max(lq, key=float)
        mj = max(Q[f"delta_{k}"]["mujoco"]["law_minus_mujoco_um"]["max"] for k in ("E_FE", "E_meas", "E_TPU"))
        fl, fn_, fb = (np.array(FE["holds"][k]) for k in ("level_N", "f_n", "fe_best_N"))
        rr = ("face", "edge", "flank")

        def mrow(name, E, ratios, d):
            return [name, "" if E is None else f"{E:.2f}", ", ".join(f1(ratios[r]) for r in rr), f"{d['fit'][0]:.2f}",
                    f"{d['fit'][1]:.3f}", f"{d['rms_mm'] * 1e3:.0f}"]
        mrows = [mrow("FE pad as computed", None, {r: fe_r[r]["ratio_median_s1"] for r in rr},
                      dict(fit=[sr["a_mm"], sr["c_mm_N"]], rms_mm=sr["rms_mm"])),
                 mrow("Pads at the simulations' default", dT["E_MPa"], rq["E_TPU"]["by_region"],
                      dict(fit=dT["law"]["fit"]["fit"], rms_mm=dT["law"]["rms_mm"])),
                 mrow("Pads fitted to the FE pad", dF["E_MPa"], rq["E_FE"]["by_region"],
                      dict(fit=dF["law"]["fit"]["fit"], rms_mm=dF["law"]["rms_mm"])),
                 mrow("Pads fitted to the HEX-E", dM["E_MPa"], rq["E_meas"]["by_region"],
                      dict(fit=dM["law"]["fit"]["fit"], rms_mm=dM["law"]["rms_mm"]))]
        lrows = []
        for k in sorted(lq, key=float):
            mm = (fl == float(k)) & (fb > 0)
            lrows.append([f"{float(k):g}", f"{lq[k]['n']}", f1(lq[k]["pads_over_fe"]), f1(lq[k]["meas_over_pads_E_meas"]),
                          f1(float(np.median(fn_[mm] / fb[mm])))])
        sh = Q.get("shear")
        sh_txt = ""
        if sh:
            ru = sh["runs"]
            irs = sorted({v["impratio"] for v in ru.values()})
            vF = {ir: ru[f"impratio{ir:g}_E_FE"]["creep_mm_s"]["median"] for ir in irs}
            vM = {ir: ru[f"impratio{ir:g}_E_meas"]["creep_mm_s"]["median"] for ir in irs}
            hold = sh["hold_s"]["median"]

            def left(key):                                  # the sphere slid off the pad: faster than 100 mm/s
                return int(sum(1 for y in ru[key]["per_hold"] if y is not None and y > 100.0))
            sh_txt = (f"<p>The pads do not keep the sphere still under the holds' shear. The measured tangential force is a "
                      f"median {sh['tangential_N']['median']:.2f} N, {sh['tangential_over_normal']['median']:.2f} of the normal "
                      f"force. In the rig the sphere takes the measured force vector, normal and tangential, and slides along the "
                      f"pad at a steady speed. With the pads at {dF['E_MPa']:.1f} MPa the median speed is "
                      + ands([f"{vF[ir]:.3g} mm/s at impratio {ir:g}" for ir in irs])
                      + f" (Fig. 5b). The pads at {dM['E_MPa']:.2f} MPa slide faster, "
                      + ands([f"{vM[ir]:.3g} mm/s" for ir in irs])
                      + f". Over the median hold of {hold:.1f} s, impratio {irs[0]:g} would move the sphere "
                      f"{vF[irs[0]] * hold:.1f} mm with the stiffer pads. At impratio {irs[0]:g} the sphere left the pad "
                      f"within the second in {left(f'impratio{irs[0]:g}_E_FE')} of {Q['n']} holds with the stiffer pads. "
                      f"With the softer pads it left in {left(f'impratio{irs[0]:g}_E_meas')}. At impratio {irs[-1]:g} it "
                      f"left in {left(f'impratio{irs[-1]:g}_E_FE')} and {left(f'impratio{irs[-1]:g}_E_meas')} holds. The robot fixed the "
                      f"sphere's position during the real holds, so there the shear built up with the contact in place.</p>")
        body = (f"<p>Sphere pads built on this pad's surface follow the measured indentation as closely as the FE pad does "
                f"when both take a spring in series. With the pads fitted to the FE pad, the residual of $\\Delta = a + "
                f"c\\,f_n$ is {dF['law']['rms_mm'] * 1e3:.0f} µm rms, against {sr['rms_mm'] * 1e3:.0f} µm for the FE pad "
                f"(Table 10). Here $\\Delta$ is defined as in Section 3, with the pads in place of the FE pad. With one "
                f"modulus of {dM['E_MPa']:.2f} MPa and no spring in series, the pads give the measured force within a median "
                f"{Q['median_rel_err_meas'] * 100:.0f} %. The FE pad scaled by {FE['s_best']:g} gives it within "
                f"{FE['median_rel_err_best'] * 100:.0f} %. That modulus is {E_TPU_DEFAULT / dM['E_MPa']:.0f} times below "
                f"the simulations' default of {E_TPU_DEFAULT:g} MPa, because it also takes up the extra indentation of "
                f"Section 3.</p>"
                f"<p>The {Q['n_pads']} pads stand on the surface points of the FE basis, so both models share one surface. "
                f"Their depths $h_i$ run from {Q['h_mm']['min']:.1f} to {Q['h_mm']['max']:.1f} mm. In the MuJoCo rig, made "
                f"by {F['code']('scripts/ur5_tip_calibration_pads.py')}, the pads are fixed and the sphere rides on a slide "
                f"along the contact normal. The measured normal force pushes it toward the pad for 0.4 s. At every hold its "
                f"final position agrees with the pads' spring law within {max(mj, 0.01):.2f} µm, so at rest MuJoCo adds "
                f"nothing to the law the pads were built with.</p>"
                f"<p>Against the FE pad at the same positions, the modulus that fits best, {Q['E_FE_MPa']:.1f} MPa, leaves "
                f"{Q['pads_vs_fe_log_rms_pct']:.0f} % rms in the logarithm of the force ratio. The ratio of pads over FE "
                f"rises with the load, from {lq[k0]['pads_over_fe']:.2f} at {float(k0):g} N to {lq[k1]['pads_over_fe']:.2f} "
                f"at {float(k1):g} N (Table 11, Fig. 5a). The pads' force grows faster with the sphere's depth than the FE "
                f"pad's."
                + (f" Stiffer pads, at {m3['E_loc_MPa']:.0f} MPa, in series with a spring of {m3['k_s_N_mm']:.2f} N/mm "
                   f"follow the FE pad's depth at the measured forces within {m3['rms_mm'] * 1e3:.0f} µm rms. The study of "
                   f"2026-10-18 that calibrated the simulators to FE models used this form as its recipe M3, with the pads' "
                   f"modulus set by the twist arm. Here both values are fitted to depth alone, so the size of the contact "
                   f"patch is not checked." if m3 and m3.get("k_s_N_mm") else "")
                + "</p>"
                f"<p>The pads place the centre of the contact a median {loc['pads_vs_fe']['median_mm'] * 1e3:.0f} µm from the "
                f"FE pad's, and {loc['pads_vs_fe']['p90_mm'] * 1e3:.0f} µm at the 90th percentile. The FE pad's centre lies "
                f"a median {loc['fe_vs_geo']['median_mm'] * 1e3:.0f} µm from the surface point nearest the sphere's "
                f"centre.</p>" + sh_txt
                + P.table(["Model", "Modulus (MPa)", "Measured over model: face, edges, flanks", "a (mm)", "c (mm/N)",
                           "Residual of Δ (µm rms)"], mrows, num_cols=(1, 3, 4, 5),
                          caption="FE pad and sphere pads against the HEX-E at the registered positions (medians by region).",
                          n=10)
                + P.table(["Level (N)", "Holds", "Pads over FE pad", "Measured over pads at E_meas",
                           "Measured over FE pad, scaled"], lrows, num_cols=(0, 1, 2, 3, 4),
                          caption="Force ratios by force level (medians over the holds).", n=11)
                + F["fig"](5, MEDIA / "20261022-sphere2_sphere_pads_against_hexe.png",
                           f"Sphere pads against the measured pad. (a) Force ratios at the registered positions by force level: "
                           f"the pads over the FE pad, and the measured force over the pads and over the scaled FE pad. (b) The sphere's sliding speed in MuJoCo "
                           f"under a constant force equal to each hold's measured force, by impratio: medians with the 10th "
                           f"and 90th percentiles. The dashed line is 0.1 mm over the median hold."))
        S_.append(("pads", "Sphere-pad model against the measured pad", body))

    # ---- rod
    RD = R.get("ROD")
    if RD:
        rfe, rsr = RD["FE"], RD["FE"]["series"]
        rfs = {r: v["force_scale"] for r, v in RD["S"]["by_region"].items()}
        rpr = RD["PR"]["curve"][-1] if RD.get("PR") else None
        spr = PR["curve"][-1] if PR else None
        R_rod = RD["plan"]["geometry"]["indenter"]["radius"]
        jr = reg.get("joint_rms_um", {})
        jn = {k.split("-")[-1]: v for k, v in jr.items()}
        own_med, own_max = own_fits_apart()
        loc_part = sr["c_mm_N"] - rsr["c_mm_N"]

        def row(name, fe, srs, pr):
            br = fe["by_region"]
            return [name, f"{srs['n']}", f"{fe['s_best']:g}",
                    ", ".join(f1(br[r]["ratio_median_s1"]) for r in ("face", "edge") if r in br),
                    f"{srs['a_mm']:.3f}", f"{srs['c_mm_N']:.3f}", f"{srs['rms_mm'] * 1e3:.0f}",
                    f"{pr['gp:board']['loc_mm']['median']:.2f}" if pr else "",
                    f"{pr['mlp:board']['loc_mm']['median']:.2f}" if pr and "mlp:board" in pr else ""]
        rows = [row("Sphere, radius 5 mm (sphere2), its own registration", FE, sr, spr)]
        SJ, RO = R.get("SPH_JOINT"), R.get("ROD_OWN")
        if SJ:
            rows.append(row("Sphere (sphere2), joint registration", SJ["FE"], SJ["FE"]["series"],
                            SJ["PR"]["curve"][-1] if SJ.get("PR") else None))
        rows.append(row(f"Rod, radius {R_rod:g} mm (rod1), joint registration", rfe, rsr, rpr))
        if RO:
            rows.append(row("Rod (rod1), rod1t's own fit", RO["FE"], RO["FE"]["series"],
                            RO["PR"]["curve"][-1] if RO.get("PR") else None))
        rq = RD.get("PADS")
        body = (f"<p>Under the rod the presses travel {rsr['a_mm']:.2f} mm + {rsr['c_mm_N']:.2f} mm/N beyond the FE pad, "
                f"with a residual of {rsr['rms_mm'] * 1e3:.0f} µm rms over {rsr['n']} holds. Under the sphere, with the same "
                f"registration of the tool, they travel {sr['a_mm']:.2f} mm + {sr['c_mm_N']:.2f} mm/N (Section 3; Fig. 6a, "
                f"Table 12). A spring in series with the pad, in the mount, the arm or the indenter's support, adds the same "
                f"travel per newton under both indenters. At most {rsr['c_mm_N']:.2f} mm/N is therefore in series. At least "
                f"{loc_part:.2f} mm/N arises at the sphere's contact, where the FE pad is stiffer than the print. The rod, of "
                f"radius {R_rod:g} mm, spreads its load along a line across the pad. Measured over FE at $s = 1$ is "
                f"{rfe['by_region']['face']['ratio_median_s1']:.2f} on the face under the rod, against "
                f"{fe_r['face']['ratio_median_s1']:.2f} under the sphere (Fig. 6b).</p>"
                f"<p>The rod run rod1 (10:45 on 2026-10-23) began its touch-offs with the pad already on the rod, since its "
                f"plan placed the rest poses with the nominal tool. The lab then made 22 touch-offs on the rod at a 5.0 mm "
                f"standoff (rod1t) and 19 on the sphere (sph3t). Most of sph3t's onsets repeat sphere2's within 0.06 mm, so the "
                f"tool did not move overnight. One registration of the tool, with each indenter's centre fitted, leaves "
                + (f"{jn.get('rod1t', float('nan')):.0f} µm rms on rod1t, {jn.get('sph3t', float('nan')):.0f} µm on sph3t "
                   f"and {jn.get('sphere2', float('nan')):.0f} µm on sphere2. " if jn else "residuals of 40 to 60 µm rms. ")
                + f"It leaves out rod1t's pad-edge-01, which met the rod 0.66 mm before the modelled pad, with joint 6 at "
                f"45.5°. The two sphere sessions' own fits put the pad {own_med:.2f} mm apart, as a median over its surface "
                f"(largest {own_max:.2f} mm). Touch-offs on one sphere fix the turn about the tool axis only weakly. In "
                f"two simulated sessions with realistic tool errors, a joint fit of rod and sphere touch-offs placed contacts "
                f"0.17 to 0.30 mm from the truth (median). One session's fit placed them 0.28 to 0.60 mm off (QUEUE.md R19).</p>"
                + (f"<p>On the real presses the two registrations disagree along the finger. With the joint one, sphere2's "
                   f"contact points move by a median {SJ['moved']['median']:.2f} mm: on average {abs(SJ['moved']['mean'][2]):.2f} mm "
                   f"along the finger, toward its {'mount' if SJ['moved']['mean'][2] < 0 else 'tip'}. Both fit sphere2's touch-offs equally well, and neither indenter's touch-offs fix "
                   f"that direction well; the rod's tilt, never measured, may also pull the joint fit. The other sections "
                   f"therefore keep sphere2's own registration. With the joint one, sphere2's extra indentation is "
                   f"{SJ['FE']['series']['a_mm']:.2f} mm + {SJ['FE']['series']['c_mm_N']:.2f} mm/N, with "
                   f"{SJ['FE']['series']['rms_mm'] * 1e3:.0f} µm rms (Table 12). The slope under each indenter is the same "
                   f"under either registration, so the split above holds; the offsets $a$ depend on the registration.</p>"
                   if SJ else "")
                + f"<p>Under the rod the board's force estimate is {rfs.get('face', float('nan')):.2f} of the HEX-E's on the "
                f"face and {rfs.get('edge', float('nan')):.2f} on the edges, against {fs['face']:.2f} and {fs['edge']:.2f} "
                f"under the sphere."
                + (f" Trained on {rpr['N']} rod presses, the Gaussian process places held-out contacts within "
                   f"{rpr['gp:board']['loc_mm']['median']:.2f} mm and the MLP ensemble within "
                   f"{rpr['mlp:board']['loc_mm']['median']:.2f} mm. Their force errors are "
                   f"{rpr['gp:board']['force_N']['median']:.3f} and {rpr['mlp:board']['force_N']['median']:.3f} N."
                   if rpr and "mlp:board" in rpr else "")
                + (f" Under the rod, the sphere pads fitted to the FE pad ({rq['E_FE_MPa']:.1f} MPa) leave "
                   f"{rq['delta_E_FE']['law']['rms_mm'] * 1e3:.0f} µm rms of $\\Delta$ with a spring in series. The FE pad "
                   f"leaves {rsr['rms_mm'] * 1e3:.0f} µm." if rq else "")
                + "</p>"
                + P.table(["Indenter", "Holds", "FE scale s", "Measured over FE at s = 1: face, edges", "a (mm)", "c (mm/N)",
                           "Residual of Δ (µm rms)", "GP location (mm)", "MLP location (mm)"], rows,
                          num_cols=(1, 2, 4, 5, 6, 7, 8),
                          caption="Sphere and rod presses against the FE pad under two registrations. The predictors are "
                                  "trained on all but the held-out presses of each run (medians).", n=12)
                + F["fig"](6, MEDIA / "20261023-rod1_against_sphere2.png",
                           "Sphere and rod against the FE pad. (a) Indentation beyond the FE pad against the HEX-E's normal "
                           "force, each hold a point, with the fitted lines. (b) Measured force over the FE pad's at $s = 1$, "
                           "by force level (medians)."))
        S_.append(("rod", "Rod presses and the split of the extra indentation", body))

    # ---- day repeat
    DY = R.get("DAY")
    if DY and DY.get("DR"):
        DR = DY["DR"]
        dl, sn = DR["day_later"], DR.get("same_night", {})
        st, snr = DR.get("same_target"), DR.get("same_night_refs")
        Hd = DY["H"]
        pm = Hd["kind"] == "press"
        ids_d = sorted(set(Hd["id"][pm].tolist()))
        grid_d = [i for i in ids_d if not i.startswith("ref-")]
        n_ref = len(ids_d) - len(grid_d)
        n_grid_s = sum(1 for i in set(H["id"][press_mask(H)].tolist()) if R["acts"].get(i, {}).get("stage") == "grid"
                       and not i.startswith("ref-"))
        td0, td1 = float(Hd["t"][pm].min()), float(Hd["t"][pm].max())
        same = np.isin(H["id"], grid_d)
        md = pm & np.isin(Hd["id"], grid_d)
        hours = (float(np.median(Hd["t"][md])) - float(np.median(H["t"][same]))) / 3600.0 if same.any() and md.any() else None
        names = [("viewer", "Viewer's estimator, untrained"), ("physical", "Physical model, compliance fitted"),
                 ("gp:board", "Gaussian process on the board"), ("mlp:board", "MLP ensemble on the board"),
                 ("physical+gp:board", "Physical model and a Gaussian process on its residual")]
        prow = []
        for k, nm in names:
            a_ = DR["viewer"] if k == "viewer" else dl.get(k)
            b_ = (PR["viewer"] if PR else None) if k == "viewer" else sn.get(k)
            prow.append([nm] + [f1(x[key]["median"], nd) if x else "&#8211;" for x in (a_, b_)
                                for key, nd in (("loc_mm", 2), ("force_N", 3))])
        sl = st["by_level"] if st else {}
        rl = snr["by_level"] if snr else {}
        lrow = []
        for L in sorted(set(sl) | set(rl), key=float):
            a_, b_ = sl.get(L), rl.get(L)
            lrow.append([f"{float(L):g}"] + [y for x in (a_, b_) for y in (
                (f"{x['n']}", f"{x['board_rel_change'] * 100:.0f}", f"{x['depth_change_mm'] * 1e3:+.0f}") if x
                else ("0", "&#8211;", "&#8211;"))])
        g, m_ = dl.get("gp:board"), dl.get("mlp:board")
        gs, ms_ = sn.get("gp:board"), sn.get("mlp:board")
        ph_, phs = dl.get("physical"), sn.get("physical")
        body = (f"<p>Trained on all {DR['n_train_presses']} presses of sphere2, the Gaussian process places the contacts "
                f"of the presses made a day later within {g['loc_mm']['median']:.2f} mm (median, Table 13). Its force error "
                f"is {g['force_N']['median']:.3f} N."
                + (f" Trained on the first {sn['N']} presses, it placed the night's own held-out contacts within "
                   f"{gs['loc_mm']['median']:.2f} mm and {gs['force_N']['median']:.3f} N (Section 6)." if gs else "")
                + ((f" The MLP ensemble gives {m_['loc_mm']['median']:.2f} mm and {m_['force_N']['median']:.3f} N a day "
                    f"later" + (f", against {ms_['loc_mm']['median']:.2f} mm and {ms_['force_N']['median']:.3f} N the same "
                                f"night." if ms_ else ".")) if m_ else "")
                + (f" The physical model gives {ph_['loc_mm']['median']:.2f} mm, against {phs['loc_mm']['median']:.2f} mm."
                   if ph_ and phs else "")
                + f" The viewer's estimator, with the magnets fitted to the night's field at rest, gives "
                f"{DR['viewer']['loc_mm']['median']:.2f} mm"
                + (f", against {PR['viewer']['loc_mm']['median']:.2f} mm." if PR else ".")
                + "</p>"
                f"<p>The lab repeated {len(grid_d)} of sphere2's {n_grid_s} grid presses and made its reference press "
                f"{n_ref} times, from {hm(td0)} to {hm(td1)} on 2026-10-23 (QUEUE.md R21)."
                + (f" The presses came a median {hours:.0f} h after the same presses of sphere2." if hours else "")
                + " At 12:45 the touch-offs sph3t had met the sphere within 0.06 mm of sphere2's, so the tool was the "
                "night's (Section 8). The robot aimed the presses with sph3t's registration. The ground truth here uses "
                "sphere2's registration, as the other sections do."
                + (f" Paired by target and level, the two runs' registered contact points lie a median "
                   f"{st['point_apart_mm']['median']:.2f} mm apart." if st and "point_apart_mm" in st else "")
                + " The physical model and the snap of the predicted points keep sphere2's fit of the field at rest, as a "
                "calibration made one night and used the next day would.</p>")
        if st:
            both = [L for L in sorted(set(sl) & set(rl), key=float) if float(L) >= 1.0]
            lo_, hi_ = min(sl, key=float), max(sl, key=float)
            rd = [v["depth_change_mm"] * 1e3 for v in rl.values()]
            day_pct = ands([f"{sl[L]['board_rel_change'] * 100:.0f}" for L in both])
            night_pct = ands([f"{rl[L]['board_rel_change'] * 100:.0f}" for L in both])
            body += ("<p>"
                     + (f"At {ands([f'{float(L):g}' for L in both])} N, the board change a day later differs from the "
                        f"night's at the same target by {day_pct} % of its size (medians, Table 14, Fig. 7a). Between the "
                        f"night's own repeats of the reference press it differed by {night_pct} %. " if both else "")
                     + f"The difference is {sl[lo_]['board_rel_change'] * 100:.0f} % at {float(lo_):g} N and "
                     f"{sl[hi_]['board_rel_change'] * 100:.0f} % at {float(hi_):g} N.</p>"
                     f"<p>A day later the pad met the sphere earlier. At the same force its registered indentation is a "
                     f"median {-st['depth_change_mm']['median'] * 1e3:.0f} µm smaller"
                     + (f", while between the night's own repeats it grew by {min(rd):.0f} to {max(rd):.0f} µm." if rd else ".")
                     + "</p>")
        dfe = DY.get("FE", {}).get("series")
        dS = DY.get("S", {}).get("by_region", {})
        if dfe:
            body = body[:-4] + (f" Beyond the FE pad the repeated presses travel {dfe['a_mm']:.2f} mm + {dfe['c_mm_N']:.2f} "
                                f"mm/N, with a residual of {dfe['rms_mm'] * 1e3:.0f} µm rms over {dfe['n']} holds. The night's "
                                f"presses travelled {sr['a_mm']:.2f} mm + {sr['c_mm_N']:.2f} mm/N (Section 3). The slope is "
                                f"the night's, and the offset is {sr['a_mm'] - dfe['a_mm']:.2f} mm smaller. From the onset at "
                                f"0.1 N to 2 N, the day's first reference press travelled 0.651 mm and the night's first "
                                f"0.709 mm (the lab's track_refs.py, its message of 16:06)."
                                + (f" With its magnets fitted to the day's field at rest, the viewer's estimator gives "
                                   f"{dS['face']['force_scale']:.2f} of the HEX-E's force on the face, against "
                                   f"{fs['face']:.2f} the night before." if dS.get("face", {}).get("force_scale") is not None
                                   else "")
                                + "</p>")
        body += (P.table(["Predictor", "Location error, a day later (mm)", "Force error, a day later (N)",
                          "Location error, same night (mm)", "Force error, same night (N)"], prow, num_cols=(1, 2, 3, 4),
                         caption=f"Predictors trained on sphere2's presses and scored on the presses a day later (all "
                                 f"{DR['n_train_presses']} training presses) and on the night's held-out presses (first "
                                 f"{sn.get('N', '?')}). Medians over the holds.", n=13)
                 + (P.table(["Level (N)", "Pairs a day later", "Difference of the board change (%)",
                             "Indentation change (µm)", "Pairs of reference holds, same night",
                             "Difference of the board change, same night (%)", "Indentation change, same night (µm)"],
                            lrow, num_cols=(0, 1, 2, 3, 4, 5, 6),
                            caption="Board change and indentation at the same target and force level (medians by level).",
                            n=14) if lrow else "")
                 + F["fig"](7, MEDIA / "20261023-sphere2_day_repeat.png",
                            "Repeat of sphere2's presses a day later. (a) Difference of the board change at the same "
                            "target and force level, each pair of holds a point, with the medians as bars: the presses a day "
                            "later against the night's, and the night's repeats of the reference press against its first. "
                            "(b) Median location error of the predictors trained on sphere2, by force level: on the presses "
                            "a day later (solid) and on the night's held-out presses (dashed)."))
        S_.append(("day", "Repeat of the sphere grid a day later", body))

    # ---- limitations, future work
    rs_ = R["ROD"]["FE"]["series"] if R.get("ROD") else None
    body = ("<p>In the ground truth, the arm, the tool's parts and the indenter's mount are taken as rigid. "
            + (f"The rod's extra indentation (Section 8) bounds the part that lies in series with the pad. At 3 N the "
               f"pad's true depth is then at most {rs_['a_mm'] + rs_['c_mm_N'] * 3.0:.2f} mm less than the registered one. "
               if rs_ else
               f"By the extra indentation of Section 3, the pad's true depth at 3 N is about "
               f"{sr['a_mm'] + sr['c_mm_N'] * 3.0:.2f} mm less than the registered one. ")
            + "The tangential location of the contact changes much less if that motion lies along "
            "the normal; a rotation of the pad under load would also move it tangentially. "
            f"The held-out set has {PR['n_test_presses'] if PR else 0} presses. The training presses come from the "
            "same night, the same pad and the same seating of the tool"
            + (", and the day repeat of Section 9 used them as well.</p>" if R.get("DAY") else ".</p>"))
    S_.append(("limits", "Limitations", body))
    fw = [
        "Press the DIP link by hand in its fork, and the sphere's block on the 2x4, and watch for motion (QUEUE.md R16). "
        "Then press a rigid part of the tool, the HEX-E's adapter or the fork, on the sphere at 1 to 5 N, with a dial "
        "indicator on the sphere's block. Travel of the block places the compliance on the indenter's side. Travel "
        f"beyond the HEX-E's data sheet with a still block places it in the arm or the mount. Travel near the data "
        f"sheet's leaves the DIP link's clamp for the "
        + (f"rod's {R['ROD']['FE']['series']['c_mm_N']:.2f} mm/N." if R.get("ROD") else f"{sr['c_mm_N']:.2f} mm/N."),
        "Fit the magnets' offsets and the pad's pose in its seat to the presses at 3 N and over: the physical model of "
        + F["code"]("ur5_tip_calibration_predict.py") + " without its priors on the offsets. If the mean location error "
        f"falls below 0.5 mm from {np.linalg.norm(bm['mean']) if bm else 0:.1f} mm, the offset is in the model's magnets; "
        "if it stays, it is in the registration.",
        "Add $\\Delta = a + c\\,f_n$ to the ground truth (" + F["code"]("ur5_tip_calibration_analysis.py") + ", sphere "
        "moved back along the normal) and repeat the FE comparison per region. A ratio of measured over FE near 1 in "
        "every region confirms a compliance in series; a ratio that still changes with the region points to the FE "
        "model's surface.",
        "Compare the twists' torsion about the normal with the sphere pads' friction torque at the same normal force, by "
        "turning the sphere about the normal in the rig of " + F["code"]("ur5_tip_calibration_pads.py") + ". The torsion is "
        "half the difference between the outbound and the return holds at each angle. A torque outside the pads' range at "
        "the measured friction coefficient falsifies the pads' torsion.",
    ] + ([] if R.get("ROD", {}).get("PADS") else [
        "Repeat the sphere-pad comparison on the rod run (rod1g) with the rig's capsule indenter. If the pads' residual of "
        "$\\Delta$ stays within 10 µm of the FE pad's for the line contact as well, the pads' depth law holds for both "
        "indenter shapes."]) + [
        "Press the FE pad's printed twin with a 5 mm sphere on a bench under a dial indicator, the pad clamped on a rigid "
        "block, at 0.5 to 5 N. Travel near the sphere's local part of Section 8 confirms that the FE pad is too stiff under "
        "a sphere. Travel near the FE pad's own places that part in the robot's tool.",
        "Run a grid in random order on another night, so that clock time and absolute temperature stop following the "
        "target; then the temperature inputs can be scored without the leak.",
    ]
    S_.append(("future", "Future work", "<ol>" + "".join(f"<li>{x}</li>" for x in fw) + "</ol>"))
    return S_, lede, summary


# ------------------------------------------------------------------------------------------------ the page
def build(session, data=None, day_data=None):
    R = load(session, data, day_data)
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig_fe(R, MEDIA / "20261022-sphere2_fe_against_hexe.png")
    fig_board(R, MEDIA / "20261022-sphere2_board_against_hexe.png")
    fig_map(R, MEDIA / "20261022-sphere2_estimates_on_the_pad.png")
    if R["PR"]:
        fig_predict(R, MEDIA / "20261022-sphere2_predictor_learning_curves.png")
    if R["PADS"]:
        fig_pads(R, MEDIA / "20261022-sphere2_sphere_pads_against_hexe.png")
    if R.get("ROD"):
        fig_rod(R, MEDIA / "20261023-rod1_against_sphere2.png")
    if R.get("DAY"):
        fig_day(R, MEDIA / "20261023-sphere2_day_repeat.png")
    F = dict(fig=P.fig, table=P.table, link=P.link, code=P.code)
    S, lede, summary = sections(R, F)
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    byline = (f"Workstation analysis of the lab's run {P.esc(session)}, built {built} from "
              f"{P.code('data/' + (data or session) + '/')}"
              + (f", {P.code('data/' + ROD_DATA.name + '/')}" if R.get("ROD") else "")
              + (f" and {P.code('data/' + R['DAY']['dir'].name + '/')}" if R.get("DAY") else "") + f" by {P.code('scripts/ur5_tip_calibration_results_page.py')}. "
              f"Requests to the lab: {P.link(QUEUE, 'QUEUE.md')}; status of the work: {P.link(STATUS_PAGE, 'status page')}.")
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "UR5 fingertip ground truth",
                 "H1": "Pad compliance, board response and contact predictors of the magnetic fingertip on the UR5",
                 "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-22", "LEDE": lede, "BYLINE": byline,
                 "FILM": "", "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary)
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", default="20261022-210852-sphere2")
    ap.add_argument("--data", default=None, help="the analysis folder under data/ (default: the session's own)")
    ap.add_argument("--day-data", default=None, help=f"the day repeat's analysis folder (default {DAY_DATA.name})")
    a = ap.parse_args()
    build(a.session, a.data, a.day_data)
