#!/usr/bin/env python3
"""The blind twists of 2026-11-16 against two simulations (owner, 2026-11-16: "i want to end the presentation strong, with
some preliminary evidence that we can get approximately close to real in simulation first"). Four ground-clear grasps
(grid2dgr 4, 8, 12, 13) at layout (105, 70) mm on the 40 mm sphere, each:
  pipeline    the simulation that scores the landscapes and draws the talk's films: hom_tasks.rollout on the CPU, the
              SCS0009 at 0.72 N m/rad that the load can backdrive, HOM converting at 0.72, the scenes' stiff contact
              (logs/20261116-sim_real_reversal/blind_pipeline.py, seeds 0-2);
  calibrated  the model fitted to the afternoon's trials: gears the load cannot backdrive, index and middle yaw gains
              0.35, HOM at the bench's 0.55, soft tangential contact on the real object (stem and block), HOM's targets
              under the 0.84 N m torque budget, predicted before the trials ran (rev_predict.py --kp-yaw-im 0.35
              --torque-budget 0.84 --trace; data/20261116-blind_twists/20261116-predictions_budget084.json, 19:51);
  hand        the trials under the budget at 19:58-20:06 (runs/, scored by scripts/hardware/hom_trial_tracking.py).
The tracking score is 1 - err/ref from 2.0 s, as hom_tasks.metrics defines it; grasps 4 and 8 stopped early on corvus's
drift guard (the sphere slid 20 mm in the hand), so their hand scores cover 2.0-6.0 s. Grasp 13 ran on the hand at 13:03,
before the calibration, so its prediction is not blind. Writes data/20261116-blind_twists/20261116-blind_compare.json,
media/20261116-blind_twists_sim_vs_hand.png (bench trials folder) and figures.json blind_twists (co-design folder).

    .venv/bin/python scripts/hardware/hom_blind_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401
from sr2hand.geometry.rotations import quat_integrate, to_scipy  # noqa: E402

BENCH = ROOT / "docs/experiments/20261115-hom_bench_trials"
REV = ROOT / "logs/20261116-sim_real_reversal"
OUT = BENCH / "data/20261116-blind_twists/20261116-blind_compare.json"
FIG = BENCH / "media/20261116-blind_twists_sim_vs_hand.png"
FIGS_JSON = ROOT / "docs/experiments/20261110-codesign_formulation/media/20261110-figures.json"
HAND = {13: "20261116-195825", 4: "20261116-200049", 12: "20261116-200348", 8: "20261116-200600"}
IN_CALIBRATION = {13}                     # ran on the hand at 13:03, among the trials the model was fitted to


def hand_trace(stamp):
    """The twist about z_P from the record's first tracked pose and the commanded twist, every 0.1 s."""
    rows = [json.loads(x) for x in (BENCH / "runs" / f"{stamp}_twist_d40_trial000.jsonl").read_text().splitlines()
            if x.strip()]
    rows = [r for r in rows if "t" in r and r.get("obj_quat") is not None]
    t = np.array([r["t"] for r in rows], float)
    Q = np.array([r["obj_quat"] for r in rows], float)
    W = np.array([(r.get("ref") or {}).get("w_ref") or [0.0, 0.0, 0.0] for r in rows], float)
    h = np.diff(t, append=t[-1] + 0.02)
    cmd = np.degrees(to_scipy(quat_integrate(W * h[:, None])).as_rotvec()[:, 2])
    tw = np.degrees((to_scipy(Q) * to_scipy(Q[0]).inv()).as_rotvec()[:, 2])
    keep = np.arange(0, len(t), 5)
    return {"t": t[keep].round(3).tolist(), "twist_deg": tw[keep].round(2).tolist(), "cmd_deg": cmd[keep].round(2).tolist()}


VARIANTS = (("pads_soft", "predict_budget084_kp035_trace.json", "1 mm sphere pads, soft tangential contact (impratio 1)"),
            ("point_soft", "predict_budget084_kp035_pt_trace.json", "one tip mesh, point contact, impratio 1"),
            ("pads_stiff", "predict_budget084_kp035_pads_imp100_trace.json", "1 mm sphere pads, impratio 100"),
            ("point_stiff", "predict_budget084_kp035_pt_imp100_trace.json", "one tip mesh, point contact, impratio 100"))


def trace_rms(hand, traces):
    """The RMS difference (deg) of each simulated twist trace from the hand's, from 0 s to the hand's last sample,
    averaged over the traces."""
    th, yh = np.array(hand["t"]), np.array(hand["twist_deg"])
    out = []
    for s in traces:
        ts, ys = np.array(s["t"]), np.array(s["twist_deg"])
        tt = np.arange(0.0, min(ts[-1], th[-1]), 0.1)
        out.append(float(np.sqrt(np.mean((np.interp(tt, ts, ys) - np.interp(tt, th, yh)) ** 2))))
    return float(np.mean(out)) if out else None


def build():
    tracking = {t["stamp"]: t for t in json.loads((BENCH / "data/20261116-trial_tracking.json").read_text())["trials"]}
    cal = json.loads((REV / "predict_budget084_kp035_trace.json").read_text())
    pipe = json.loads((REV / "blind_pipeline.json").read_text())
    out = {"grasps": {}, "sources": {"calibrated": cal["meta"], "pipeline": pipe["meta"]}}
    for g, stamp in HAND.items():
        tr = tracking[stamp]
        c = [r for r in cal["rows"] if r["grasp"] == f"grid2dgr_c329_g{g}" and "error" not in r]
        p = [r for r in pipe["rows"] if r["grasp"] == g]
        out["grasps"][str(g)] = {
            "blind": g not in IN_CALIBRATION,
            "hand": {"stamp": stamp, "track": tr["track"]["score"], "window_s": [tr["window"]["t0_s"], tr["window"]["t1_s"]],
                     "partial": tr["window"]["partial"], "ended": tr["ended"]["short"],
                     "twist_task_end_deg": tr["track"]["twist_task_end_deg"], "trace": hand_trace(stamp)},
            "calibrated": {"track": [r["track"] for r in c], "twist_end": [r["twist_end"] for r in c],
                           "held": sum(r["held"] for r in c), "traces": [r["trace"] for r in c]},
            "pipeline": {"track": [r["track"] for r in p], "twist_end": [r["twist_end"] for r in p],
                         "held": sum(r["held"] for r in p), "diverged": sum("error" in r for r in p),
                         "traces": [r["trace"] for r in p if r.get("trace")]}}
    mae = {k: float(np.mean([abs(np.mean(v[k]["track"]) - v["hand"]["track"]) for v in out["grasps"].values()]))
           for k in ("pipeline", "calibrated")}
    out["mean_abs_error"] = {k: round(v, 3) for k, v in mae.items()}
    # the calibrated servo with another contact (owner, 2026-11-16 21:1x: "can we now compare with a point contact
    # model"): the fingertip as one mesh with point contact in place of the pads, and the scenes' stiff tangential
    # contact (impratio 100) in place of the soft one (impratio 1), each as rev_predict.py --scene runs it
    out["variants"] = {}
    for key, fname, what in VARIANTS:
        d = json.loads((REV / fname).read_text())
        per = {}
        for g in HAND:
            rs = [r for r in d["rows"] if r["grasp"] == f"grid2dgr_c329_g{g}" and "error" not in r]
            per[str(g)] = {"track": float(np.mean([r["track"] or 0.0 for r in rs])), "held": sum(r["held"] for r in rs),
                           "twist_end": float(np.mean([r["twist_end"] for r in rs])),
                           "rise_max": float(np.mean([r["rise_max"] for r in rs])),
                           "trace_rms_deg": trace_rms(out["grasps"][str(g)]["hand"]["trace"],
                                                      [r["trace"] for r in rs if r.get("trace")])}
        out["variants"][key] = {"what": what, "scene": d["meta"]["scene"], "grasps": per,
                                "mean_abs_error": round(float(np.mean([abs(per[str(g)]["track"]
                                                                           - out["grasps"][str(g)]["hand"]["track"])
                                                                       for g in HAND])), 3),
                                "trace_rms_deg_mean": round(float(np.mean([per[str(g)]["trace_rms_deg"]
                                                                           for g in HAND])), 1)}
    out["pipeline_trace_rms_deg_mean"] = round(float(np.mean([trace_rms(v["hand"]["trace"], v["pipeline"]["traces"])
                                                              for v in out["grasps"].values()])), 1)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    return out


def figure(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 16, "axes.titlesize": 17, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13})
    INK, GREY, CAL, NEW = "#1f2937", "#9ca3af", "#d97706", "#7c3aed"
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.0, 5.6), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.28})
    a.plot([0, 0.85], [0, 0.85], "--", color=GREY, lw=1.2, zorder=1, label="simulation = hand")
    place = {"13": (-0.015, 0.03, "right"), "4": (0.015, 0.03, "left"), "12": (-0.015, -0.05, "right"),
             "8": (0.015, 0.03, "left")}              # each label's offset from its calibrated point, and its side
    for g, v in sorted(d["grasps"].items(), key=lambda kv: int(kv[0])):
        y = v["hand"]["track"]
        p, c = np.array(v["pipeline"]["track"]), np.array(v["calibrated"]["track"])
        a.annotate("", xy=(c.mean(), y), xytext=(p.mean(), y),
                   arrowprops={"arrowstyle": "-|>", "color": GREY, "lw": 1.2, "shrinkA": 7, "shrinkB": 7})
        a.errorbar(p.mean(), y, xerr=[[p.mean() - p.min()], [p.max() - p.mean()]], fmt="o", ms=10, mfc="white",
                   mec=GREY, mew=1.8, ecolor=GREY, capsize=3, zorder=3)
        a.errorbar(c.mean(), y, xerr=[[c.mean() - c.min()], [c.max() - c.mean()]], fmt="o", ms=10, color=CAL,
                   mec=INK, mew=0.8, ecolor=CAL, capsize=3, zorder=4)
        lab = f"grasp {g}" + ("" if v["blind"] else " (fitted)") + (" *" if v["hand"]["partial"] else "")
        dx, dy, ha = place[g]
        a.text(c.mean() + dx, y + dy, lab, fontsize=13, color=INK, ha=ha)
    a.plot([], [], "o", ms=10, mfc="white", mec=GREY, mew=1.8, label="before calibration (the design pipeline)")
    a.plot([], [], "o", ms=10, color=CAL, mec=INK, mew=0.8, label="calibrated, predicted before the trial")
    a.set_xlim(-0.02, 0.85)
    a.set_ylim(-0.04, 0.85)
    a.set_xlabel("Simulated tracking score (3 seeds)")
    a.set_ylabel("The hand's tracking score")
    me = d["mean_abs_error"]
    a.set_title(f"Mean error {me['pipeline']:.2f} before, {me['calibrated']:.2f} after", loc="left")
    a.legend(loc="upper left", frameon=False)
    a.text(0.0, -0.20, "* the hand's run stopped at 6 s, the sphere slid 20 mm in the hand", transform=a.transAxes,
           fontsize=12, color="#4b5563")
    v = d["grasps"]["12"]
    h = v["hand"]["trace"]
    b.plot(h["t"], h["cmd_deg"], "--", color=NEW, lw=2.2, label="command")
    for k, tr in enumerate(v["pipeline"]["traces"]):
        b.plot(tr["t"], tr["twist_deg"], color=GREY, lw=1.6, label="before calibration" if k == 0 else None)
    for k, tr in enumerate(v["calibrated"]["traces"]):
        b.plot(tr["t"], tr["twist_deg"], color=CAL, lw=2.0, label="calibrated, predicted" if k == 0 else None)
    b.plot(h["t"], h["twist_deg"], color=INK, lw=2.8, label="the hand")
    b.set_xlabel("Time (s)")
    b.set_ylabel("Twist of the sphere about the vertical (deg)")
    b.set_title("Grasp 12, predicted before it ran", loc="left")
    b.legend(loc="upper left", frameon=False)
    for ax in (a, b):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(color="#eef0f3")
        ax.set_axisbelow(True)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)


def main():
    d = build()
    figure(d)
    summ = {"image": str(FIG.relative_to(ROOT)), "mean_abs_error": d["mean_abs_error"],
            "variants": {k: {"mean_abs_error": v["mean_abs_error"], "trace_rms_deg_mean": v["trace_rms_deg_mean"],
                             "track": {g: round(x["track"], 3) for g, x in v["grasps"].items()}}
                         for k, v in d["variants"].items()},
            "pipeline_trace_rms_deg_mean": d["pipeline_trace_rms_deg_mean"],
            "grasps": {g: {"blind": v["blind"], "hand": v["hand"]["track"], "hand_partial": v["hand"]["partial"],
                           "hand_window_s": v["hand"]["window_s"],
                           "hand_twist_deg": v["hand"]["trace"]["twist_deg"][-1],   # from the first sample, as simulated
                           "calibrated": round(float(np.mean(v["calibrated"]["track"])), 3),
                           "pipeline": round(float(np.mean(v["pipeline"]["track"])), 3),
                           "pipeline_twist_deg": round(float(np.mean([x for x in v["pipeline"]["twist_end"]
                                                                      if x is not None])), 1),
                           "calibrated_twist_deg": round(float(np.mean(v["calibrated"]["twist_end"])), 1)}
                       for g, v in d["grasps"].items()}}
    fi = json.loads(FIGS_JSON.read_text())
    fi["blind_twists"] = summ
    FIGS_JSON.write_text(json.dumps(fi, indent=1, default=float))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
