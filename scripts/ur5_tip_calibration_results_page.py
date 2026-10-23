#!/usr/bin/env python3
"""Page of the fingertip's ground truth and contact predictors from a UR5 calibration session (2026-10-22): the pad's
compliance against its FE model, the board's response against the HEX-E, the pivot check, the viewer's estimator against
the ground truth, and predictors of the contact point and the normal force trained on the session's presses.

Reads the analysis's output of the session in docs/experiments/20261022-ur5_tip_calibration/data/<session>/ and the
session's own folder in lab/<session>/. The outputs come from

    .venv/bin/python scripts/ur5_tip_calibration_analysis.py lab/<session> --gravity-from lab/20261022-195420-first2
    .venv/bin/python scripts/ur5_tip_calibration_fe.py data/<session>/holds.npz --plan lab/<session>/plan.json --out data/<session>/fe.json
    .venv/bin/python scripts/ur5_tip_calibration_predict.py data/<session>

(paths under docs/experiments/20261022-ur5_tip_calibration/). Writes the figures to media/ and the page
docs/experiments/20261022-ur5_tip_calibration/20261022-ur5_tip_ground_truth_and_predictors.html.

    .venv/bin/python scripts/ur5_tip_calibration_results_page.py [--session 20261022-210852-sphere2]
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
REGION_STYLE = {"face": ("o", "#1f5fbf"), "edge": ("s", "#c0392b"), "flank": ("^", "#2e8b57")}
RNAME = {"face": "face", "edge": "edges", "flank": "flanks"}
MODELS = [("physical", "Physical model, compliance fitted", "#2e8b57", "-"),
          ("gp:board", "Gaussian process on the board", "#1f5fbf", "-"),
          ("ridge:board", "Ridge regression on the board", "#7f7f7f", "--"),
          ("physical+gp:board", "Physical model and a Gaussian process on its residual", "#c0392b", "-")]
STEPS = ["board", "+temperature change", "+time since rest", "+orientation (leaks the target)"]
STEP_TEXT = {"board": "Board", "+temperature change": "Board, temperature change",
             "+time since rest": "Board, temperature change, time since rest",
             "+orientation (leaks the target)": "Board, temperature change, time since rest, orientation"}


def f1(x, n=2):
    return "&#8211;" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{n}f}"


def ands(parts):
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def hm(t):
    return datetime.datetime.fromtimestamp(t).strftime("%H:%M")


# ------------------------------------------------------------------------------------------------ data
def load(session):
    sd, ld = DATA / session, LAB / session
    R = dict(session=session, sd=sd, ld=ld)
    R["S"] = json.loads((sd / "summary.json").read_text())
    Hz = np.load(sd / "holds.npz", allow_pickle=False)
    R["H"] = {k: Hz[k] for k in Hz.files}
    R["FE"] = json.loads((sd / "fe.json").read_text())
    R["PR"] = json.loads((sd / "predict/predict.json").read_text()) if (sd / "predict/predict.json").exists() else None
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
    lede = (f"At the sphere's registered positions in the UR5 run {R['session'].split('-', 2)[-1]}, the force computed "
            f"with the pad's finite-element (FE) model is {k_fe:.1f} times the force measured by the HEX-E (median on the "
            f"face). The force estimated from the board is {min(fs.values()):.2f} to {max(fs.values()):.2f} times the "
            f"HEX-E's, so the pad itself is not {FE['s_best']:g} times softer than its model. The extra indentation, "
            f"{sr['a_mm']:.2f} mm plus {sr['c_mm_N']:.2f} mm per newton, occurs in parts whose deformation is not sensed "
            f"by the board. From 1 N up, the viewer's estimator places contacts {min(loc_hi):.1f} to {max(loc_hi):.1f} mm "
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
    summary = (f"UR5 run {R['session'].split('-', 2)[-1]}: measured force is {fe_r['face']['ratio_median_s1']:.2f} of the FE "
               f"pad's at the registered positions while the board reads {min(fs.values()):.2f}-{max(fs.values()):.2f} of "
               f"the HEX-E's force; extra indentation {sr['a_mm']:.2f} mm + {sr['c_mm_N']:.2f} mm/N outside the pad; "
               f"estimator {min(loc_hi):.1f}-{max(loc_hi):.1f} mm from 1 N"
               + (f"; GP on {last['N']} presses {last['gp:board']['loc_mm']['median']:.2f} mm, "
                  f"{last['gp:board']['force_N']['median']:.2f} N on held-out presses" if last else "")
               + (f", {PR['turns']['gp:board']['loc_mm']['median']:.2f} mm on rolls and twists." if PR and PR.get("turns") else "."))

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
        ("Predictor, training presses $N$", "A model trained on the holds of the first $N$ presses in time order and "
         "scored on held-out presses (every fourth grid press, apart from the reference presses). Each score is the "
         "median over the held-out holds."),
    ]
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
            f"is from {hm(R['t1'])}. Its pad touch-offs were fitted to a registration of the tool and the sphere (Table 1). "
            f"The registration moved the tool centre point by {tcp_mm:.2f} mm and {tcp_deg:.2f}° from the CAD and homing values. "
            f"The run itself, its restarts and the HEX-E's link are described on the "
            + (F["link"](LAB_PAGE, "page of the lab runs") if LAB_PAGE.exists() else "page of the lab runs (not built yet)")
            + f"; the protocol is on the {F['link'](PROTOCOL_PAGE, 'protocol page')}.{ref_txt}</p>"
            + F["table"](["Quantity", "Value"], rows, num_cols=(1,), caption="Registration and data of the session.", n=1)
            + (f"<p>The presses {ands(exc)} are left out of the ground truth. In each of them the HEX-E carried 0.3 to "
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
            for lab, key in (("Gaussian process", "gp"), ("Ridge regression", "ridge"), ("Physical model and Gaussian process", "physical+gp")):
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
        keys = (("viewer", "Viewer's estimator, untrained"), ("gp:board", "Gaussian process"),
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
                     ("gp:board", "Gaussian process, board"), ("ridge:board", "Ridge regression, board"),
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
        S_.append(("predict", "Predictors of the contact point and the normal force", body))

    # ---- limitations, future work
    body = ("<p>In the ground truth, the arm, the tool's parts and the indenter's mount are taken as rigid. By the extra "
            f"indentation of Section 3, the pad's true depth at 3 N is about {sr['a_mm'] + sr['c_mm_N'] * 3.0:.2f} mm less "
            "than the registered one. The tangential location of the contact changes much less if that motion lies along "
            "the normal; a rotation of the pad under load would also move it tangentially. "
            f"The held-out set has {PR['n_test_presses'] if PR else 0} presses. The training presses come from the "
            "same night, the same pad and the same seating of the tool.</p>")
    S_.append(("limits", "Limitations", body))
    fw = [
        "Press the DIP link by hand in its fork, and the sphere's block on the 2x4, and watch for motion (QUEUE.md R16). "
        "Then press a rigid part of the tool, the HEX-E's adapter or the fork, on the sphere at 1 to 5 N, with a dial "
        "indicator on the sphere's block. Travel of the block places the compliance on the indenter's side. Travel "
        f"beyond the HEX-E's data sheet with a still block places it in the arm or the mount. Travel near the data "
        f"sheet's leaves the DIP link's clamp for the {sr['c_mm_N']:.2f} mm/N.",
        "Fit the magnets' offsets and the pad's pose in its seat to the presses at 3 N and over: the physical model of "
        + F["code"]("ur5_tip_calibration_predict.py") + " without its priors on the offsets. If the mean location error "
        f"falls below 0.5 mm from {np.linalg.norm(bm['mean']) if bm else 0:.1f} mm, the offset is in the model's magnets; "
        "if it stays, it is in the registration.",
        "Add $\\Delta = a + c\\,f_n$ to the ground truth (" + F["code"]("ur5_tip_calibration_analysis.py") + ", sphere "
        "moved back along the normal) and repeat the FE comparison per region. A ratio of measured over FE near 1 in "
        "every region confirms a compliance in series; a ratio that still changes with the region points to the FE "
        "model's surface.",
        "Compare the twists' torsion about the normal with the sphere pads' friction torque at the same normal force "
        "(contact model overview). The torsion is half the difference between the outbound and the return holds at each "
        "angle. A torque outside the pads' range at the measured friction coefficient falsifies the pads' torsion.",
        "Run a grid in random order on another night, so that clock time and absolute temperature stop following the "
        "target; then the temperature inputs can be scored without the leak.",
    ]
    S_.append(("future", "Future work", "<ol>" + "".join(f"<li>{x}</li>" for x in fw) + "</ol>"))
    return S_, lede, summary


# ------------------------------------------------------------------------------------------------ the page
def build(session):
    R = load(session)
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig_fe(R, MEDIA / "20261022-sphere2_fe_against_hexe.png")
    fig_board(R, MEDIA / "20261022-sphere2_board_against_hexe.png")
    fig_map(R, MEDIA / "20261022-sphere2_estimates_on_the_pad.png")
    if R["PR"]:
        fig_predict(R, MEDIA / "20261022-sphere2_predictor_learning_curves.png")
    F = dict(fig=P.fig, table=P.table, link=P.link, code=P.code)
    S, lede, summary = sections(R, F)
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    byline = (f"Workstation analysis of the lab's run {P.esc(session)}, built {built} from "
              f"{P.code('data/' + session + '/')} by {P.code('scripts/ur5_tip_calibration_results_page.py')}. "
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
    build(ap.parse_args().session)
