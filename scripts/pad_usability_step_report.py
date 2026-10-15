#!/usr/bin/env python3
"""Scores, tables and figures of experiment U3 of the 2026-10-14 pad-usability study: the largest stable and the largest
accurate physics step of the 1 mm sphere pads, 2 mm pads, point contact and condim 4 on the real_v1 hand.

Terms
  spectral threshold  the step h at which the spectral radius of MuJoCo's one-step map (mjd_transitionFD at a rest
                      state, refsafe off) first exceeds 1 + 1e-3, bisected to 0.01 ms (scripts/pad_usability_step.py
                      threshold); slow modes within 0.05 of 1 (creep, slip under gravity) are excluded
  normal-row bound    2 t_c (sqrt(1 + q) - 1), q = d0 + (1 - d0) / n, n loaded tool contacts of a fingertip, smallest
                      over the fingertips
  friction-row bound  2 / (b lambda'), b the friction rows' damping coefficient, lambda' the largest eigenvalue of
                      A (A + R)^-1 over the loaded friction rows; d0 t_c when impratio is large
  joint-space model   the threshold of a linear model of the step built from the rest state (servo springs, damping,
                      every loaded contact row), pad_usability_step.jointspace_threshold
  within 3 deg of 1 ms  a plan replay at step h keeps the tool and ends within 3 deg of the same model's 1 ms replay on
                      the same placement (refsafe on, MuJoCo's default)
  usable step         the largest step with at least 36 of 40 placements within 3 deg of 1 ms and no ejection
  ejection            the tool faster than 2 m/s or more than 0.30 m from its start, or a MuJoCo bad-state warning
  CPU per simulated s wall time of mj_step summed over the replay, divided by the simulated time, ms per s

    .venv/bin/python scripts/pad_usability_step_report.py
Reads step/threshold.jsonl and step/plans_dt.jsonl under docs/experiments/20261014-pad_usability/, the Drake rows of U2
(agree/servo_drake.jsonl) and 2026-10-06 (20261006-hom_turn3/plans_drake.jsonl); writes step/summary_U3.json,
step/20261014-u3_step_sweep.png and step/20261014-u3_spectral.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pad_usability_agree as AG  # noqa: E402

STEP = ROOT / "docs/experiments/20261014-pad_usability/step"
REL = "docs/experiments/20261014-pad_usability/step/"
SERVO, WORK = AG.SERVO, AG.WORK
PLANTS = [(SERVO, "Servo-limited plant"), (WORK, "Working plant")]
MODELS = [("pads", "1 mm pads", "#2a78d6", "o"), ("pads2", "2 mm pads", "#eb6834", "s"),
          ("pt", "Point contact", "#1baf7a", "^"), ("pt4", "Condim 4", "#eda100", "D")]
DTS = [1.0, 2.0, 4.0, 5.0, 10.0]
NEED = 36


def sweep():
    R = [r for r in AG.jl(STEP / "plans_dt.jsonl")]
    by = {}
    for r in R:
        by.setdefault((r["model"], r["plant"], float(r["dt_ms"])), {})[(r["hand"], int(r["seed"]))] = r
    drake = {pl: AG.drake_rows(pl) for pl, _ in PLANTS}
    out = []
    for m, label, _, _ in MODELS:
        for pl, pname in PLANTS:
            base = by[(m, pl, 1.0)]
            for dt in DTS:
                cur = by[(m, pl, dt)]
                ok1 = [k for k, r in cur.items() if k in base and r.get("held_end") and base[k].get("held_end")
                       and abs(r["turn_end_deg"] - base[k]["turn_end_deg"]) <= 3.0]
                okd = [k for k, r in cur.items() if k in drake[pl] and r.get("held_end")
                       and abs(r["turn_end_deg"] - drake[pl][k]["turn_end_deg"]) <= 3.0]
                lo, hi = AG.clopper_pearson(len(ok1), len(cur), alpha=0.10)
                cpu = [r["cpu_ms_per_sim_s"] for r in cur.values() if r.get("cpu_ms_per_sim_s")]
                out.append(dict(model=m, label=label, plant=pl, plant_name=pname, dt_ms=dt, n=len(cur),
                                within3_1ms=len(ok1), ci90=[40 * lo, 40 * hi], held=sum(bool(r.get("held_end")) for r in cur.values()),
                                within3_drake=len(okd), n_drake=len(drake[pl]),
                                ejected=sum(bool(r.get("ejected_flag")) for r in cur.values()),
                                nan=sum(r.get("status") == "nan" for r in cur.values()),
                                cpu_ms_per_sim_s=float(np.median(cpu)) if cpu else None,
                                solver_niter=float(np.median([r["solver_niter_mean"] for r in cur.values()]))))
    usable = {}
    for m, label, _, _ in MODELS:
        for pl, _ in PLANTS:
            good = [s["dt_ms"] for s in out if s["model"] == m and s["plant"] == pl and s["within3_1ms"] >= NEED
                    and s["ejected"] == 0 and s["nan"] == 0]
            usable[f"{m}|{pl}"] = max(good) if good else None
    return out, usable


def spectral():
    T = [r for r in AG.jl(STEP / "threshold.jsonl") if r.get("status") == "ok"]
    hand, bed = [], []
    for r in T:
        p = r["pred"]
        row = dict(state=r["state"], model=r["model"], h_ms=r["threshold_ms"],
                   n_by_finger={k: v["n"] for k, v in p.get("fingers", {}).items()}, d0=p.get("d0"), tc_ms=p.get("tc_ms"),
                   normal_ms=p.get("pred_normal_full_ms"), friction_ms=p.get("pred_friction_ms"),
                   jointspace_ms=p.get("pred_jointspace_ms"), mode=(r.get("mode") or {}).get("mode"),
                   loaded_by_kind=p.get("loaded_by_kind"),
                   confirm=[(round(c["h_ms"], 3), c["bounded"], c["why"]) for c in r.get("confirm", [])])
        (bed if r["state"] == "bed" else hand).append(row)
    return hand, bed


def fig_sweep(S, path):
    plt = AG._style()
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 7.6), dpi=110, sharex=True,
                            gridspec_kw=dict(height_ratios=[1.35, 1.0], hspace=0.12, wspace=0.28))
    for j, (pl, pname) in enumerate(PLANTS):
        ax, bx = axs[0, j], axs[1, j]
        for m, label, col, mk in MODELS:
            ss = [s for s in S if s["model"] == m and s["plant"] == pl]
            x = [s["dt_ms"] for s in ss]
            ax.plot(x, [s["within3_1ms"] for s in ss], "-", color=col, lw=2, marker=mk, ms=7, label=label,
                    markeredgecolor="white", markeredgewidth=1.0)
            bx.plot(x, [s["cpu_ms_per_sim_s"] for s in ss], "-", color=col, lw=2, marker=mk, ms=7,
                    markeredgecolor="white", markeredgewidth=1.0)
        ax.axhline(NEED, color="#000", lw=1, ls="--")
        ax.set_ylim(0, 41.5)
        ax.set_yticks([0, 10, 20, 30, 36, 40])
        ax.set_title(pname)
        bx.set_yscale("log")
        bx.set_xscale("log")
        bx.set_xticks(DTS)
        bx.set_xticklabels(["1", "2", "4", "5", "10"])
        bx.set_xlabel("Physics step (ms)")
        bx.minorticks_off()
        bx.set_xticks(DTS)
        for a in (ax, bx):
            a.grid(axis="y", color="#ddd", lw=0.8)
            a.set_axisbelow(True)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
    axs[0, 0].set_ylabel("Within 3° of 1 ms, kept (of 40)")
    axs[1, 0].set_ylabel("CPU per simulated s (ms)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.15)
    fig.savefig(path)
    plt.close(fig)


def fig_spectral(hand, bed, path):
    plt = AG._style()
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.8), dpi=110, gridspec_kw=dict(wspace=0.3))
    ax = axs[0]
    for m, label, col, mk in MODELS + [("mesh", "Block mesh (RL state)", "#1baf7a", "v")]:
        pts = [(r["normal_ms"], r["h_ms"]) for r in hand if r["model"] == m and r["normal_ms"] and r["h_ms"]
               and ("postoff" in r["state"] or not r["state"].startswith("scale:"))]
        if pts:
            ax.scatter(*zip(*pts), color=col, marker=mk, s=60, edgecolor="white", linewidth=0.8, label=label, zorder=3)
    lim = (3.5, 9.0)
    ax.plot(lim, lim, color="#000", lw=1)
    ax.fill_between(lim, [0.9 * v for v in lim], [1.1 * v for v in lim], color="#000", alpha=0.07, lw=0)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Normal-row bound (ms)")
    ax.set_ylabel("Spectral threshold (ms)")
    ax.set_title("Hand states")
    ax.legend(loc="upper left", frameon=False, fontsize=12.5, handletextpad=0.2)
    bx = axs[1]
    for kind, label, col, mk in (("mj_pads1", "Pad pairs", "#2a78d6", "o"), ("mj_point3", "Point pairs", "#1baf7a", "^")):
        pts = [(r["friction_ms"], r["h_ms"]) for r in bed if r["model"].startswith(kind) and r["friction_ms"]]
        if pts:
            bx.scatter(*zip(*pts), color=col, marker=mk, s=60, edgecolor="white", linewidth=0.8, label=label, zorder=3)
    lim = (0.5, 21.0)
    bx.plot(lim, lim, color="#000", lw=1)
    bx.fill_between(lim, [0.9 * v for v in lim], [1.1 * v for v in lim], color="#000", alpha=0.07, lw=0)
    bx.set_xlim(*lim)
    bx.set_ylim(*lim)
    bx.set_xlabel("Friction-row bound (ms)")
    bx.set_title("Bed pinch, tool hanging")
    bx.legend(loc="upper left", frameon=False, fontsize=12.5, handletextpad=0.2)
    for a in axs:
        a.grid(color="#eee", lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.15)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    S, usable = sweep()
    hand, bed = spectral()
    bed_n = len(bed)
    bed_fr = sum(1 for r in bed if r["friction_ms"] and abs(r["h_ms"] - r["friction_ms"]) <= 0.1 * r["friction_ms"])
    bed_nr = sum(1 for r in bed if r["normal_ms"] and abs(r["h_ms"] - r["normal_ms"]) <= 0.1 * r["normal_ms"])
    hand_eval = [r for r in hand if "postoff" in r["state"] or not r["state"].startswith("scale:")]
    hand_nr = sum(1 for r in hand_eval if r["normal_ms"] and abs(r["h_ms"] - r["normal_ms"]) <= 0.1 * r["normal_ms"])
    hand_js = sum(1 for r in hand_eval if r["jointspace_ms"] and abs(r["h_ms"] - r["jointspace_ms"]) <= 0.03 * r["jointspace_ms"])
    summ = dict(id="U3", sweep=S, usable_step_ms=usable, spectral_hand=hand, spectral_bed=bed,
                counts=dict(bed_pairs=bed_n, bed_within10_friction=bed_fr, bed_within10_normal=bed_nr,
                            hand_states=len(hand_eval), hand_within10_normal=hand_nr, hand_within3_jointspace=hand_js),
                rows=[REL + "threshold.jsonl", REL + "plans_dt.jsonl"],
                figures=[REL + "20261014-u3_step_sweep.png", REL + "20261014-u3_spectral.png"])
    (STEP / "summary_U3.json").write_text(json.dumps(summ, indent=1) + "\n")
    fig_sweep(S, STEP / "20261014-u3_step_sweep.png")
    fig_spectral(hand, bed, STEP / "20261014-u3_spectral.png")
    for s in S:
        print(f"{s['label']:13s} {s['plant_name'][:5]} {s['dt_ms']:4.0f} ms: {s['within3_1ms']:2d}/40 "
              f"[{s['ci90'][0]:.1f}, {s['ci90'][1]:.1f}] kept {s['held']:2d} Drake {s['within3_drake']:2d}/24 "
              f"eject {s['ejected']} cpu {s['cpu_ms_per_sim_s']:.1f} ms/s")
    print("usable", usable)
    print("counts", summ["counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
