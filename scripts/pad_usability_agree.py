#!/usr/bin/env python3
"""Scores of experiment U2 of the 2026-10-14 pad-usability study: agreement of six MuJoCo fingertip models with Drake
hydroelastic on the eight deployed plan replays (servo-limited and working plant), the same at the RL trainer's numerics,
HOM closed-loop agreement, Drake's step convergence and the block bed at the trainer's numerics. Writes the summary
JSON and the two figures of the comparison.

Terms
  turn          change of the tool axis' angle from the vertical between the end of the grip (t = 0.8 s) and the end of
                the replay (10.28 s), deg (reorient_backends.score, turn_end_deg)
  kept          held_end: the tool ends less than 20 mm below its grip height and at least two fingers touch it
  placement     one hand D1-D8 and one jitter seed (0: tool on its nominal pose; 1-4: x, y sd 2 mm, yaw sd 2 deg); the 24
                Drake placements are seeds 0-2, seeds 3-4 add 16 for the held count of 40
  within 3 deg  the model keeps the tool and its turn is within 3 deg of Drake's on the same placement
                (simulator_agreement_figure.TOL_DEG; Drake's own outcome does not enter)
  |dturn|       absolute difference of the model's turn from Drake's on the same placement, deg, all 24 placements
  grip ratio    summed fingertip normal force on the tool at the end of the hold (t = 7.6 s, F_pads_hold_N), model over
                Drake, on placements where both keep the tool
  cost          median wall time of one physics step on one core (us_per_step_median of each row, then the median over
                the rows), and the same per simulated second (ms of CPU per s simulated)
  HOM agreement the closed-loop turn ends within 5 deg of Drake's with the same kept outcome (hom_turn3, governor on)
  onset arm     twist-onset torque over mu times the pinch force on the two-pad block bed (T2), mm
  creep         pre-slip sliding speed of the pinched tool held at half its slip force (T1, um/s) or its spin rate at half
                the onset torque (T2, deg/s)

Statistics: Clopper-Pearson 95 % intervals on counts; exact McNemar test (two-sided binomial on the discordant pairs)
of each model's within-3 deg indicator against the 1 mm pads on the same plant; bootstrap 95 % interval of the median
|dturn| (10,000 resamples of the 24 placements, generator seed 0).

    .venv/bin/python scripts/pad_usability_agree.py
Reads docs/experiments/20261014-pad_usability/agree/*.jsonl and agree/bed/*.jsonl, and the 2026-10-06 working-plant
and bed rows; writes
agree/summary_U2.json, agree/20261014-u2_agreement_cost.png and agree/20261014-u2_turn_pairs.png.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AG = ROOT / "docs/experiments/20261014-pad_usability/agree"
HT = ROOT / "docs/experiments/20261006-hom_turn3"
SA = ROOT / "docs/experiments/20261006-simulator_agreement"
FB = ROOT / "docs/experiments/20261006-fingertip_backends"
SERVO, WORK = "kp0.72_kv0_fr0.18_fl0_dp0.0144", "kp4_kv0_fr1_fl0_dp0.08"
PLANT_LABEL = {SERVO: "Servo-limited plant (0.72 N m/rad, 0.18 N m)", WORK: "Working plant (4 N m/rad, 1 N m)"}
TOL, TOL_HOM = 3.0, 5.0
MODELS = ["pads", "padsT", "pads2", "pt4", "pt", "ptmc"]
LABEL = {"pads": "1 mm pads", "padsT": "1 mm pads, tool only", "pads2": "2 mm pads", "pt4": "Condim 4",
         "pt": "Point contact", "ptmc": "Multiccd", "pt4_2.6": "Condim 4, 2.6 mm", "drake": "Drake hydroelastic"}
C_SERVO, C_WORK, C_REF = "#BE7514", "#4A7FC4", "#666666"     # house chart series (retro_style --s1, --s2, --ref)


def jl(path):
    path = Path(path)
    return [json.loads(line) for line in path.open() if line.strip()] if path.exists() else []


def ok(r):
    return r.get("status", "complete") in ("complete", "ok")


def key(r):
    return r["hand"], int(r["seed"])


# ---------------------------------------------------------------------------------- statistics

def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def mcnemar_exact(a, b):
    """Two-sided exact McNemar test of paired booleans a (reference) and b; returns (b10, b01, p)."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = 1.0 if n10 + n01 == 0 else float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)
    return n10, n01, p


def boot_median(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    med = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return float(np.median(x)), float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


# ---------------------------------------------------------------------------------- rows

def drake_rows(plant, dt=1.0):
    src = AG / "servo_drake.jsonl" if plant == SERVO else HT / "plans_drake.jsonl"
    return {key(r): r for r in jl(src) if ok(r) and r.get("plant") == plant and float(r.get("dt_ms", 1.0)) == dt}


def model_rows(plant, model, numerics="bed", spin=None):
    """Rows of one model on one plant: this study's files, the 2026-10-06 files for the working plant's pads and point
    contact."""
    if plant == WORK and model == "pads":
        files = [HT / "plans_mujoco.jsonl"]
    elif plant == WORK and model == "pt":
        files = [SA / "plans_mujoco_pt.jsonl"]
    elif plant == WORK:
        files = [AG / "work_mujoco.jsonl"]
    elif numerics == "rl":
        files = [AG / "servo_mujoco_rl.jsonl"]
    elif spin is not None:
        files = [AG / "servo_mujoco_pt4spin.jsonl"]
    else:
        files = [AG / "servo_mujoco.jsonl"]
    out = {}
    for f in files:
        for r in jl(f):
            if r.get("model") != model or r.get("plant") != plant or r.get("numerics", "bed") != numerics:
                continue
            if spin is not None and abs(float(r.get("spin_m") or 0) - spin) > 1e-9:
                continue
            if float(r.get("dt_ms", 1.0 if numerics == "bed" else 2.0)) != (1.0 if numerics == "bed" else 2.0):
                continue
            out[key(r)] = r
    return out


def kept(r):
    return bool(r is not None and r.get("status", "complete") == "complete" and r.get("held_end"))


def score(plant, model, numerics="bed", spin=None, ref=None):
    """Agreement of one model with Drake on one plant (see the module docstring for the terms)."""
    dk = drake_rows(plant)
    mr = model_rows(plant, model, numerics, spin)
    keys24 = sorted(dk)
    pairs = [(k, dk[k], mr.get(k)) for k in keys24]
    have = [(k, d, m) for k, d, m in pairs if m is not None and m.get("status") == "complete"]
    within = [kept(m) and abs(m["turn_end_deg"] - d["turn_end_deg"]) <= TOL for _k, d, m in pairs]
    alt = [m is not None and kept(m) == kept(d) and (not kept(d) or abs(m["turn_end_deg"] - d["turn_end_deg"]) <= TOL)
           for _k, d, m in pairs]
    dturn = [abs(m["turn_end_deg"] - d["turn_end_deg"]) for _k, d, m in have]
    grip = [m["F_pads_hold_N"] / d["F_pads_hold_N"] for _k, d, m in have
            if kept(m) and kept(d) and d.get("F_pads_hold_N") and m.get("F_pads_hold_N") is not None]
    rows40 = [r for r in mr.values()]
    held40 = sum(kept(r) for r in rows40)
    us = [r["us_per_step_median"] for r in rows40 if r.get("us_per_step_median")]
    dt_ms = 2.0 if numerics == "rl" else 1.0
    k = int(sum(within))
    out = {"plant": plant, "model": model, "numerics": numerics, "spin_m": spin, "n_pairs": len(pairs),
           "n_model_rows": len(rows40), "within3": k, "within3_ci95": clopper_pearson(k, len(pairs)),
           "within3_alt_rule": int(sum(alt)), "held24": int(sum(kept(m) for _k, _d, m in pairs)),
           "held40": int(held40), "n_ejected": sum(1 for r in rows40 if r.get("status") == "ejected"),
           "n_failed": sum(1 for r in rows40 if r.get("status") == "failed"),
           "drake_held24": int(sum(kept(d) for _k, d, _m in pairs)),
           "median_abs_dturn_deg": boot_median(dturn) if dturn else None,
           "grip_ratio_median": float(np.median(grip)) if grip else None,
           "grip_ratio_range": (float(np.min(grip)), float(np.max(grip))) if grip else None, "n_grip": len(grip),
           "us_per_step_median": float(np.median(us)) if us else None,
           "ms_cpu_per_s_sim": float(np.median(us)) / dt_ms if us else None,
           "within_by_key": {f"{h}|{s}": bool(w) for (h, s), w in zip(keys24, within)},
           "pairs": [[k_[0], k_[1], d["turn_end_deg"], kept(d), None if m is None else m.get("turn_end_deg"), kept(m)]
                     for k_, d, m in pairs]}
    if ref is not None:
        out["mcnemar_vs_pads"] = mcnemar_exact([ref["within_by_key"][kk] for kk in out["within_by_key"]],
                                               list(out["within_by_key"].values()))
    return out


def drake_cost(plant):
    us = [r["us_per_step_median"] for r in drake_rows(plant).values() if r.get("us_per_step_median")]
    return float(np.median(us)) if us else None


def per_finger_grip(plant, model="pads"):
    """Summed fingertip force at the hold divided by the fingers touching, over the kept rows (N per finger)."""
    v = [r["F_pads_hold_N"] / max(1, r["pads_hold"]) for r in model_rows(plant, model).values() if kept(r)]
    return (float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)), len(v)) if v else None


def hom_score(plant):
    if plant == SERVO:
        rows = [r for r in jl(AG / "hom_servo.jsonl") if r.get("plant") == plant]
    else:
        rows = [r for r in jl(HT / "turn3.jsonl") if r.get("plant") == plant and r.get("prm_governor") == 1.0]
    mj = {key(r): r for r in rows if r["sim"] == "mujoco" and ok(r) and int(r["seed"]) <= 2}
    dk = {key(r): r for r in rows if r["sim"] == "drake" and ok(r) and int(r["seed"]) <= 2}
    pairs = [(k, dk[k], mj[k]) for k in sorted(dk) if k in mj]
    agree = [d["held"] == m["held"] and abs(m["turn_end_deg"] - d["turn_end_deg"]) <= TOL_HOM for _k, d, m in pairs]
    flips = sum(d["held"] != m["held"] for _k, d, m in pairs)
    k = int(sum(agree))
    return {"plant": plant, "n_pairs": len(pairs), "within5_same_held": k,
            "ci95": clopper_pearson(k, len(pairs)) if pairs else None, "held_flips": int(flips),
            "drake_held": int(sum(d["held"] for _k, d, _m in pairs)), "mujoco_held": int(sum(m["held"] for _k, _d, m in pairs)),
            "median_abs_dturn_deg": float(np.median([abs(m["turn_end_deg"] - d["turn_end_deg"]) for _k, d, m in pairs]))
            if pairs else None,
            "drake_turn_median_deg": float(np.median([d["turn_end_deg"] for _k, d, _m in pairs])) if pairs else None,
            "pairs": [[k_[0], k_[1], d["turn_end_deg"], d["held"], m["turn_end_deg"], m["held"], d.get("frozen_deg"),
                       m.get("frozen_deg")] for k_, d, m in pairs]}


def drake_step_check():
    out = []
    for r in jl(AG / "drake_dt05.jsonl"):
        if not ok(r):
            continue
        ref = drake_rows(r["plant"]).get(key(r))
        if ref is None:
            continue
        d = r["turn_end_deg"] - ref["turn_end_deg"]
        out.append({"plant": r["plant"], "hand": r["hand"], "seed": int(r["seed"]), "turn_1ms": ref["turn_end_deg"],
                    "turn_05ms": r["turn_end_deg"], "dturn_deg": d, "kept_1ms": kept(ref), "kept_05ms": kept(r),
                    "converged": abs(d) < 1.0 and kept(ref) == kept(r),
                    "us_per_step_05ms": r.get("us_per_step_median"), "us_per_step_1ms": ref.get("us_per_step_median")})
    return out


def pads_step_check():
    """1 mm pads at 0.5 ms on the placements they miss at 1 ms (D3, D8; servo plant) against their 1 ms rows and Drake."""
    p1 = model_rows(SERVO, "pads")
    dk = drake_rows(SERVO)
    out = []
    for r in jl(AG / "servo_mujoco_dt05_check.jsonl"):
        if not ok(r):
            continue
        k = key(r)
        out.append({"hand": k[0], "seed": k[1], "turn_05ms": r["turn_end_deg"], "turn_1ms": p1[k]["turn_end_deg"],
                    "turn_drake_1ms": dk[k]["turn_end_deg"] if k in dk else None, "kept_05ms": kept(r),
                    "within3_of_drake_05ms": bool(k in dk and kept(r)
                                                  and abs(r["turn_end_deg"] - dk[k]["turn_end_deg"]) <= TOL)})
    return out


def bed_table():
    """Block-bed rows: this study's (agree/bed) and the 2026-10-06 impratio-100 and Drake rows, one entry per
    (task, model, N, dt); relative changes of the onset arm and creep against the impratio-100 pads at the same N and dt
    (at 2 ms the reference is this study's rerun)."""
    rows = {}
    for t in ("T1", "T2", "T5"):
        for src, tag in ((FB / f"tip_{t}.jsonl", "20261006"), (AG / "bed" / f"tip_{t}.jsonl", "20261014")):
            for r in jl(src):
                if r.get("model") not in ("mj_tpu2p7_pads1", "mj_tpu2p7_pads1_ir10", "mj_tpu2p7_pads2", "drake_tpu2p7",
                                          "mj_tpu2p7_pads1_ir1e4"):
                    continue
                if r.get("status", "complete") == "failed":
                    continue
                rows[(t, r["model"], r.get("N"), float(r["dt_ms"]), tag)] = r
    out = []
    for (t, model, N, dt, tag), r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0, kv[0][3],
                                                                        kv[0][4])):
        e = {"task": t, "model": model, "N": N, "dt_ms": dt, "source": tag}
        if t == "T1":
            e.update(mu_eff=r.get("mu_eff"), creep_um_s=1e3 * r["creep_mm_s"] if r.get("creep_mm_s") is not None else None,
                     us_per_step=r.get("us_per_step_median"))
        elif t == "T2":
            e.update(rbar_onset_mm=r.get("rbar_onset_mm"), tau_onset_mNm=1e3 * r["tau_onset_Nm"]
                     if r.get("tau_onset_Nm") else None, creep_deg_s=r.get("creep_deg_s"),
                     rbar_kin_mm=r.get("rbar_kin_mm"))
        else:
            e.update(phi_end_deg=r.get("phi_end_deg"), N_at_45_N=r.get("N_at_45_N"), slip_end_mm=r.get("slip_end_mm"))
        ref = (rows.get((t, "mj_tpu2p7_pads1", N, dt, "20261006")) if dt in (1.0, 5.0)
               else rows.get((t, "mj_tpu2p7_pads1", N, dt, "20261014")))
        if ref is not None and model != "mj_tpu2p7_pads1":
            for fld in ("rbar_onset_mm", "creep_mm_s", "creep_deg_s", "mu_eff", "N_at_45_N", "phi_end_deg"):
                if r.get(fld) is not None and ref.get(fld):
                    e[f"{fld}_rel_to_ir100"] = r[fld] / ref[fld]
        out.append(e)
    return out


def widest_spread(models=("pads", "pads2", "pt4", "pt", "ptmc")):
    """Servo-plant Drake placement with the largest range of end turns across Drake and the filmed models."""
    dk = drake_rows(SERVO)
    mr = {m: model_rows(SERVO, m) for m in models}
    best = None
    for k, d in dk.items():
        v = [d["turn_end_deg"]] + [mr[m][k]["turn_end_deg"] for m in models if k in mr[m]
                                   and mr[m][k].get("status") == "complete"]
        spread = max(v) - min(v)
        if best is None or spread > best[1]:
            best = (k, spread, v)
    return {"hand": best[0][0], "seed": best[0][1], "spread_deg": best[1], "turns": best[2]} if best else None


# ---------------------------------------------------------------------------------- figures

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for p in ("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"):
        if Path(p).exists():
            font_manager.fontManager.addfont(p)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
            break
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15, "xtick.labelsize": 15,
                         "ytick.labelsize": 15, "legend.fontsize": 14, "axes.edgecolor": "#000",
                         "axes.linewidth": 0.8, "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "mathtext.fontset": "stix"})
    return plt


def fig_agreement_cost(S, path):
    """Within-3 deg count of 24 with its interval (left) and CPU time per simulated second (right), one row per
    model and numerics, both plants."""
    plt = _style()
    rows = [("pads", "bed"), ("padsT", "bed"), ("pads2", "bed"), ("pt4", "bed"), ("pt", "bed"), ("ptmc", "bed"),
            ("pads", "rl"), ("pt4", "rl"), ("pt", "rl")]
    names = [LABEL[m] + (", trainer numerics" if n == "rl" else "") for m, n in rows] + ["Drake hydroelastic"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 6.2), sharey=True, gridspec_kw=dict(width_ratios=[1.25, 1.0],
                                                                                         wspace=0.06))
    y = np.arange(len(names))[::-1]
    for i, (m, n) in enumerate(rows):
        for plant, col, mk, dy in ((SERVO, C_SERVO, "o", 0.19), (WORK, C_WORK, "s", -0.19)):
            s = S["scores"].get(f"{plant}|{m}|{n}")
            if s is None:
                continue
            k, (lo, hi), npairs = s["within3"], s["within3_ci95"], s["n_pairs"]
            a1.plot([lo * npairs, hi * npairs], [y[i] + dy] * 2, color=col, lw=2.0, solid_capstyle="butt")
            a1.plot(k, y[i] + dy, mk, ms=9, color=col, mec="white", mew=1.2, zorder=3)
            a1.text(25.0, y[i] + dy, f"{k}", va="center", ha="left", fontsize=14, color="#000")
            c = s["ms_cpu_per_s_sim"]
            if c:
                a2.plot(c, y[i] + dy, mk, ms=9, color=col, mec="white", mew=1.2, zorder=3)
    for plant, col, mk, dy in ((SERVO, C_SERVO, "o", 0.19), (WORK, C_WORK, "s", -0.19)):
        c = S["drake_ms_cpu_per_s_sim"].get(plant)
        if c:
            a2.plot(c, y[-1] + dy, mk, ms=9, color=col, mec="white", mew=1.2, zorder=3)
    a1.text(12, y[-1], "reference", va="center", ha="center", fontsize=14, color=C_REF, style="italic")
    a1.axvline(20, color="#000", lw=0.9, ls=(0, (4, 3)))
    a1.set_xlim(0, 27.5)
    a1.set_xticks([0, 6, 12, 18, 20, 24])
    a1.set_xlabel("Placements within 3° of Drake, tool kept (of 24)")
    a2.set_xscale("log")
    a2.set_xlim(4, 4000)
    a2.set_xticks([10, 100, 1000])
    a2.set_xticklabels(["10", "100", "1,000"])
    a2.xaxis.set_minor_formatter(__import__("matplotlib").ticker.NullFormatter())
    a2.set_xlabel("CPU time per simulated second (ms)")
    a1.set_yticks(y)
    a1.set_yticklabels(names)
    for a in (a1, a2):
        a.grid(axis="x", color="#ccc", lw=0.6)
        a.set_axisbelow(True)
        a.tick_params(axis="y", length=0)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=9, color=C_SERVO, mec="white", label=PLANT_LABEL[SERVO]),
         Line2D([], [], marker="s", ls="", ms=9, color=C_WORK, mec="white", label=PLANT_LABEL[WORK]),
         Line2D([], [], color="#000", lw=0.9, ls=(0, (4, 3)), label="Hypothesis threshold, 20 of 24")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.26, right=0.985, top=0.98, bottom=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_turn_pairs(S, path):
    """End turn of every Drake-paired placement, model against Drake, one panel per model and plant."""
    plt = _style()
    cols = ["pads", "padsT", "pads2", "pt4", "pt", "ptmc"]
    fig, axs = plt.subplots(2, 6, figsize=(12.0, 5.6), sharex=True, sharey=True,
                            gridspec_kw=dict(wspace=0.08, hspace=0.16))
    lim = (-100, 100)
    for i, (plant, col) in enumerate(((SERVO, C_SERVO), (WORK, C_WORK))):
        for j, m in enumerate(cols):
            a = axs[i, j]
            s = S["scores"].get(f"{plant}|{m}|bed")
            xs = np.linspace(*lim, 3)
            a.fill_between(xs, xs - TOL, xs + TOL, color=C_REF, alpha=0.35, lw=0)
            a.plot(xs, xs, color=C_REF, lw=0.9)
            if s is None:
                a.text(0, 0, "not run", ha="center", va="center", style="italic")
                continue
            for _h, _s, xd, _kd, ym, km in s["pairs"]:
                if ym is None:
                    continue
                a.plot(xd, ym, "o", ms=6.5, mfc=col if km else "white", mec=col, mew=1.3)
            a.text(0.04, 0.96, f"{s['within3']}/24", transform=a.transAxes, ha="left", va="top", fontsize=15)
            if i == 0:
                a.set_title(LABEL[m], fontsize=15)
            a.set_xlim(*lim)
            a.set_ylim(*lim)
            a.set_xticks([-60, 0, 60])
            a.set_yticks([-60, 0, 60])
            a.grid(color="#e3e3e3", lw=0.5)
            a.set_axisbelow(True)
        axs[i, 0].set_ylabel(("Servo plant" if plant == SERVO else "Working plant") + "\nmodel turn (°)")
    for a in axs[1]:
        a.set_xlabel("Drake turn (°)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], marker="o", ls="", ms=7, mfc="#444", mec="#444", label="Model keeps the tool"),
         Line2D([], [], marker="o", ls="", ms=7, mfc="white", mec="#444", label="Model drops the tool"),
         Patch(color=C_REF, alpha=0.35, label="Within 3° of Drake")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.94, bottom=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------- main

def main():
    S = {"scores": {}, "drake_ms_cpu_per_s_sim": {}, "drake_us_per_step": {}}
    for plant in (SERVO, WORK):
        ref = score(plant, "pads")
        for m in MODELS:
            S["scores"][f"{plant}|{m}|bed"] = score(plant, m, ref=ref) if m != "pads" else ref
        c = drake_cost(plant)
        S["drake_us_per_step"][plant] = c
        S["drake_ms_cpu_per_s_sim"][plant] = c
    ref = S["scores"][f"{SERVO}|pads|bed"]
    for m in ("pads", "pt", "pt4"):
        S["scores"][f"{SERVO}|{m}|rl"] = score(SERVO, m, "rl", ref=ref)
    ref_rl = S["scores"][f"{SERVO}|pads|rl"]
    for m in ("pt", "pt4"):
        s_ = S["scores"][f"{SERVO}|{m}|rl"]
        s_["mcnemar_vs_pads_same_numerics"] = mcnemar_exact([ref_rl["within_by_key"][kk] for kk in s_["within_by_key"]],
                                                             list(s_["within_by_key"].values()))
    S["scores"][f"{SERVO}|pt4_2.6|bed"] = score(SERVO, "pt4", spin=0.0026, ref=ref)
    seed0 = [r for r in model_rows(SERVO, "pads").values() if int(r["seed"]) == 0]
    S["gate_seed0_pads_kept"] = [sum(kept(r) for r in seed0), len(seed0)]
    S["per_finger_grip_N"] = {p: per_finger_grip(p) for p in (SERVO, WORK)}
    S["hom"] = {p: hom_score(p) for p in (SERVO, WORK)}
    S["drake_step"] = drake_step_check()
    S["pads_step"] = pads_step_check()
    S["bed"] = bed_table()
    S["widest_spread"] = widest_spread()
    out = AG / "summary_U2.json"
    out.write_text(json.dumps(S, indent=1, default=float))
    print(f"gate: pads keep the tool at seed 0 on {S['gate_seed0_pads_kept'][0]}/{S['gate_seed0_pads_kept'][1]} hands")
    print(f"{'plant':6s} {'model':8s} {'num':4s} within3 (CI)          alt  held24 held40 drake24 |dturn| med (CI)     "
          f"grip ratio      us/step  McNemar(b10,b01,p)")
    for k, s in S["scores"].items():
        if s["n_model_rows"] == 0:
            print(k, "no rows")
            continue
        md = s["median_abs_dturn_deg"]
        gr = s["grip_ratio_median"]
        print(f"{'servo' if s['plant'] == SERVO else 'work':6s} {k.split('|')[1]:8s} {s['numerics']:4s} "
              f"{s['within3']:2d}/{s['n_pairs']} ({s['within3_ci95'][0] * s['n_pairs']:4.1f}-{s['within3_ci95'][1] * s['n_pairs']:4.1f})  "
              f"{s['within3_alt_rule']:2d}  {s['held24']:2d}/24  {s['held40']:2d}/{s['n_model_rows']}  "
              f"{s['drake_held24']:2d}  {md[0]:5.1f} ({md[1]:4.1f}-{md[2]:5.1f})  "
              f"{(gr or float('nan')):5.2f} {s['grip_ratio_range'] or ''}  {s['us_per_step_median'] or float('nan'):7.1f}"
              f"  {s.get('mcnemar_vs_pads', '')}")
    print("Drake us/step:", S["drake_us_per_step"])
    print("per-finger grip (median, q25, q75, n):", S["per_finger_grip_N"])
    for p, h in S["hom"].items():
        print("HOM", p, {k: v for k, v in h.items() if k != "pairs"})
    for r in S["drake_step"]:
        print("drake dt", r)
    for r in S["pads_step"]:
        print("pads dt", r)
    for e in S["bed"]:
        print("bed", e)
    print("widest spread", S["widest_spread"])
    fig_agreement_cost(S, AG / "20261014-u2_agreement_cost.png")
    fig_turn_pairs(S, AG / "20261014-u2_turn_pairs.png")
    print("wrote", out.relative_to(ROOT), "and the two figures")


if __name__ == "__main__":
    main()
