#!/usr/bin/env python3
"""The press trials of 2026-11-16 against simulation, per fingertip contact model (autopilot hand-20261116-press_sim_real,
2026-11-17; owner, 2026-11-16 23:0x: "sure, send off a background autopilot run to churn on the press task
calibration"). Two trials at layout (70, 110) mm on the 25 mm cylinder with the sphere's stem and tag block, both lifted
40 mm by the arm at 1.6 s: the generalist press (13:12, runs/20261116-131243_*, the pinch 15.9 deg below the cylinder's
axis; the cylinder fell after the release) and the pinch raised to the axis (13:19, runs/20261116-131932_*; held). The
simulations run on the eight press scenes of logs/20261116-sim_real_reversal/press_stem.py (1 mm sphere pads or point
contact; impratio 1, the soft tangential contact of the twist's calibration, or 100; noslip 0 or 30 iterations):
  closed loop  HOM on the calibrated servo as the bench server ran it (press_predict.py run, seeds 0-2), with and without
               the station's refusal of frames outside its servo range (--station);
  replay       the targets that reached the hand's servos, applied open loop from the hand's start pose
               (press_predict.py replay), with the table's friction 1.0 or 0.35;
the closed loop and the replay also at the contacts' friction 0.5 (press_predict.py --mu 0.5; 1.0 stands for the grip
tape, and the twist's reversal fitted 0.5 with stiff contact as well as 1.0 with soft contact);
  pipeline     the design pipeline's model (press_predict.py pipeline: HOM and servo at 0.72, pads at impratio 100, the
               bare cylinder on its post).
Per trial and run: the tracking score of scripts/hardware/hom_trial_tracking.py for the press (track = max(0, 1 -
err/ref), err the RMS angle between the cylinder's axis and its 2.0 s pose turned by -theta_ref about x_P over 2.0 to
5.5 s, ref the RMS of theta_ref, 13.8 deg) and its difference from the hand's; the press angle (the turn of the axis
about x_P from its first pose, positive in theta_ref's sense) and the RMS of its difference from the hand's, at the
capture times of the hand's tracked poses from 0 to 5.5 s (trace RMS) and from 2.0 to 5.5 s (task trace RMS). Reads the
rows copied beside the output (data/20261116-press/sim/); writes data/20261116-press/20261116-press_contact_compare.json
and media/20261116-press_sim_vs_hand.png (bench trials folder).

    .venv/bin/python scripts/hardware/hom_press_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401
from sr2hand.geometry.rotations import quat_to_mat  # noqa: E402

BENCH = ROOT / "docs/experiments/20261115-hom_bench_trials"
SIM = BENCH / "data/20261116-press/sim"
OUT = BENCH / "data/20261116-press/20261116-press_contact_compare.json"
FIG = BENCH / "media/20261116-press_sim_vs_hand.png"
TRIALS = {"lat0": ("20261116-131932", "pinch at the axis, 13:19"), "generalist": ("20261116-131243", "generalist, 13:12")}
T_TASK, T_END = 2.0, 5.5
SCENES = [f"cyl025_stem_block_{c}_imp{i}{n}.xml" for c in ("pads", "pt") for i in (1, 100) for n in ("", "_ns30")]
LABEL = {"pads_imp1": "pads, soft tangential", "pads_imp1_ns30": "pads, soft, noslip", "pads_imp100": "pads, stiff",
         "pads_imp100_ns30": "pads, stiff, noslip", "pt_imp1": "point, soft tangential", "pt_imp1_ns30": "point, soft, noslip",
         "pt_imp100": "point, stiff", "pt_imp100_ns30": "point, stiff, noslip"}


def key(scene):
    return scene.replace("cyl025_stem_block_", "").replace(".xml", "")


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def press_angle(a0, A):
    """The turn (deg) of axes A about x_P from a0, positive in theta_ref's sense (press_predict.press_angle)."""
    return -np.degrees(np.arctan2(a0[1] * A[:, 2] - a0[2] * A[:, 1], a0[1] * A[:, 1] + a0[2] * A[:, 2]))


def hand(plan):
    """The hand's press angle and height at its tracked poses' capture times (s from the trial's first state), each pose
    once, with the commanded theta_ref at the server's periods, and how the trial moved and ended."""
    stamp = TRIALS[plan][0]
    rows = read_jsonl(BENCH / "runs" / f"{stamp}_press_d25_trial000.jsonl")
    t0 = float(rows[0]["t0_ms"])
    per = [r for r in rows if "kind" not in r]
    seen, cap = set(), []
    for r in per:
        if r.get("obj_quat") is None or r.get("t_capture_ms") is None or r["t_capture_ms"] in seen:
            continue
        seen.add(r["t_capture_ms"])
        cap.append(((float(r["t_capture_ms"]) - t0) / 1e3, r["obj_quat"], r["obj_pos_mm"]))
    cap.sort(key=lambda x: x[0])
    t = np.array([c[0] for c in cap])
    A = quat_to_mat(np.array([c[1] for c in cap], float))[:, :, 2]
    P = np.array([c[2] for c in cap], float)
    th = press_angle(A[0], A)
    tp = np.array([float(r["t"]) for r in per])
    ref = np.array([np.nan if (r.get("ref") or {}).get("theta_ref") is None else r["ref"]["theta_ref"] for r in per])
    heading = np.degrees(np.unwrap(np.arctan2(A[:, 0], A[:, 1])))
    k16 = int(np.searchsorted(t, 1.6))
    low = np.flatnonzero(P[:, 2] < P[0, 2] - 5.0)
    return {"stamp": stamp, "t": t.round(3).tolist(), "press_deg": th.round(2).tolist(),
            "z_mm": (P[:, 2] - P[0, 2]).round(2).tolist(), "t_ref": tp.round(3).tolist(),
            "theta_ref_deg": [None if not np.isfinite(x) else round(float(x), 2) for x in ref],
            "slide_x_mm_at_1p6": round(float(P[k16, 0] - P[0, 0]), 2),
            "turn_about_z_deg_at_1p6": round(float(abs(heading[k16] - heading[0])), 1),
            "t_low5_s": round(float(t[low[0]]), 2) if len(low) else None}


def trace_rms(h, ts, ys, t0=0.0, t1=T_END):
    """RMS (deg) of a simulated press angle (ts, ys) against the hand's at the hand's capture times in [t0, t1]."""
    th, yh = np.array(h["t"]), np.array(h["press_deg"])
    sel = (th >= t0 - 1e-9) & (th <= min(t1, ts[-1]) + 1e-9)
    return float(np.sqrt(np.mean((np.interp(th[sel], ts, ys) - yh[sel]) ** 2)))


def sim_trace(r):
    """(t, press angle) of a simulated run: every period for the replays, every 0.1 s otherwise."""
    if r.get("trace_all"):
        ta = r["trace_all"]
        A = quat_to_mat(np.array(ta["quat"], float))[:, :, 2]
        return np.array(ta["t"]), press_angle(A[0], A)
    tr = r["trace"]
    return np.array(tr["t"]), np.array(tr["press_deg"])


def at(ts, ys, t):
    return round(float(np.interp(t, ts, ys)), 2) if ts[-1] >= t - 1e-9 else None


def group(rows, h, hand_track):
    """Summary of the runs of one trial and one variant (module docstring)."""
    tr = [r.get("track") or 0.0 for r in rows]
    traces = [sim_trace(r) for r in rows if r.get("trace") or r.get("trace_all")]
    out = {"runs": len(rows), "track": round(float(np.mean(tr)), 3), "track_min": round(min(tr), 3),
           "track_max": round(max(tr), 3), "track_error": round(float(np.mean(tr)) - hand_track, 3),
           "held_t_end": sum(bool(r.get("held_t_end")) for r in rows), "held_end": sum(bool(r.get("held_end")) for r in rows),
           "t_low5_s": [r.get("t_low5_s") for r in rows],
           "press_max_deg": round(float(np.mean([r.get("press_max_deg") or 0.0 for r in rows])), 2),
           "press_3p5_deg": round(float(np.mean([at(t, y, 3.5) for t, y in traces if at(t, y, 3.5) is not None])), 2),
           "press_5p5_deg": round(float(np.mean([at(t, y, 5.5) for t, y in traces if at(t, y, 5.5) is not None])), 2),
           "trace_rms_deg": round(float(np.mean([trace_rms(h, t, y) for t, y in traces])), 2),
           "task_trace_rms_deg": round(float(np.mean([trace_rms(h, t, y, T_TASK) for t, y in traces])), 2),
           "frames_refused": [r.get("frames_refused") for r in rows],
           "gap_median_deg": ({k: round(float(np.mean([r["gap_median_deg"][k] for r in rows if r.get("gap_median_deg")])), 1)
                               for k in rows[0]["gap_median_deg"]} if rows[0].get("gap_median_deg") else None)}
    if rows[0].get("joint_rms_deg"):
        out["joint_rms_deg"] = rows[0]["joint_rms_deg"]
    if rows[0].get("trace_all"):
        ta = rows[0]["trace_all"]
        t = np.array(ta["t"])
        A = quat_to_mat(np.array(ta["quat"], float))[:, :, 2]
        P = np.array(ta["pos_mm"])
        k = int(np.searchsorted(t, 1.6))
        heading = np.degrees(np.unwrap(np.arctan2(A[:, 0], A[:, 1])))
        out["slide_x_mm_at_1p6"] = round(float(P[k, 0] - P[0, 0]), 2)
        out["turn_about_z_deg_at_1p6"] = round(float(abs(heading[k] - heading[0])), 1)
    return out


def build():
    tracking = {t["stamp"]: t for t in json.loads((BENCH / "data/20261116-trial_tracking.json").read_text())["trials"]}
    cal = read_jsonl(SIM / "press_calibrated.jsonl")
    rep = read_jsonl(SIM / "press_replay.jsonl")
    pipe = read_jsonl(SIM / "press_pipeline.jsonl")
    out = {"hand": {}, "closed_loop": {}, "closed_loop_station": {}, "closed_loop_mu_0p5": {}, "replay": {},
           "replay_table_mu_0p35": {}, "replay_mu_0p5": {}, "pipeline": {}, "sources": {"rows": str(SIM.relative_to(ROOT)), "scripts": "logs/20261116-sim_real_reversal/"
                                       "press_stem.py, press_predict.py (copies in the rows' folder)"}}
    for plan, (stamp, what) in TRIALS.items():
        tk = tracking[stamp]
        h = hand(plan)
        h.update({"what": what, "track": tk["track"]["score"], "err_rms_deg": tk["track"]["err_rms_deg"],
                  "ref_rms_deg": tk["track"]["ref_rms_deg"], "ended": tk["ended"]["short"],
                  "gap_median_deg": {k: v["median_deg"] for k, v in tk["joint_gap"]["joints"].items()},
                  "press_3p5_deg": at(np.array(h["t"]), np.array(h["press_deg"]), 3.5),
                  "press_5p5_deg": at(np.array(h["t"]), np.array(h["press_deg"]), 5.5)})
        out["hand"][plan] = h
        mu1 = (lambda r: r.get("mu", 1.0) == 1.0)
        for name, rows, sel in (("closed_loop", cal, lambda r: not r["station"] and mu1(r)),
                                ("closed_loop_station", cal, lambda r: r["station"] and mu1(r)),
                                ("closed_loop_mu_0p5", cal, lambda r: not r["station"] and r.get("mu") == 0.5),
                                ("replay", rep, lambda r: r.get("table_mu") is None and mu1(r)),
                                ("replay_table_mu_0p35", rep, lambda r: r.get("table_mu") == 0.35 and mu1(r)),
                                ("replay_mu_0p5", rep, lambda r: r.get("table_mu") is None and r.get("mu") == 0.5)):
            for sc in SCENES:
                rs = [r for r in rows if r["plan"] == plan and r["scene"] == sc and sel(r) and "trace" in r]
                if rs:
                    out[name].setdefault(key(sc), {})[plan] = group(rs, h, h["track"])
        rs = [r for r in pipe if r["plan"] == plan and "trace" in r]
        out["pipeline"].setdefault("pipeline", {})[plan] = group(rs, h, h["track"])
    for name in ("closed_loop", "closed_loop_station", "closed_loop_mu_0p5", "replay", "replay_table_mu_0p35",
                 "replay_mu_0p5", "pipeline"):
        for v in out[name].values():
            v["mean_abs_track_error"] = round(float(np.mean([abs(v[p]["track_error"]) for p in TRIALS])), 3)
            v["mean_trace_rms_deg"] = round(float(np.mean([v[p]["trace_rms_deg"] for p in TRIALS])), 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    return out, {"cal": cal, "rep": rep, "pipe": pipe}


def figure(d, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17,
                         "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15, "mathtext.fontset": "cm"})
    # the series colours of the twist and pick-up comparisons, the command moved off purple, which deuteranopes cannot
    # tell from the point contact's blue (OKLab dE x100 0.4); every pair below separates by 13.7 or more under
    # protanopia, deuteranopia and tritanopia and by 16.3 or more in normal vision
    INK, CMD, PADS, PT, PIPE = "#1f2937", "#be185d", "#d97706", "#2563eb", "#9ca3af"
    shown = [("pads_imp1", PADS, "-", "1 mm sphere pads, soft tangential contact (impratio 1)", "pads, soft"),
             ("pads_imp100_ns30", PADS, "--", "pads, stiff tangential contact and noslip", "pads, stiff"),
             ("pt_imp1", PT, "-", "point contact, soft tangential contact", "point, soft"),
             ("pt_imp100_ns30", PT, "--", "point contact, stiff and noslip", "point, stiff")]
    CMD_LABEL = r"command, $\theta_{\mathrm{ref}}$"
    MU05_LABEL = "point contact, soft, friction 0.5 (replay only)"
    YLIM = (-50.0, 30.0)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.36, "wspace": 0.08})
    for col, plan in enumerate(TRIALS):
        h = d["hand"][plan]
        for row, (model, src) in enumerate((("closed_loop", rows["cal"]), ("replay", rows["rep"]))):
            ax = axes[row, col]
            for x in (1.6, T_TASK, T_END):
                ax.axvline(x, color="#d1d5db", lw=1.2, ls=":", zorder=0)
            ax.text(1.62, YLIM[1] - 2.0, "lift", fontsize=13, color="#4b5563", va="top")
            tr = np.array(h["t_ref"])
            yr = np.array([np.nan if v is None else v for v in h["theta_ref_deg"]], float)
            ax.plot(tr, yr, "--", color=CMD, lw=2.0, label=CMD_LABEL)
            if model == "closed_loop":
                p = next(r for r in rows["pipe"] if r["plan"] == plan and r["seed"] == 0)
                t, y = sim_trace(p)
                ax.plot(t, y, color=PIPE, lw=2.0, label="design pipeline's model (servo 0.72, stiff pads, bare cylinder)")
            off = []
            for k, c, ls, lab, short in shown:
                sc = f"cyl025_stem_block_{k}.xml"
                cand = [r for r in src if r["plan"] == plan and r["scene"] == sc and r["seed"] == 0
                        and r.get("mu", 1.0) == 1.0
                        and (not r["station"] if model == "closed_loop" else r.get("table_mu") is None)]
                if not cand:
                    continue
                t, y = sim_trace(cand[0])
                lo = np.flatnonzero(y < YLIM[0])
                if len(lo):                                  # below the axis: drawn to the edge, marked, listed
                    k0 = lo[0]
                    ax.plot(t[:k0 + 1], np.maximum(y[:k0 + 1], YLIM[0]), ls, color=c, lw=2.2, label=lab)
                    ax.plot([t[k0]], [YLIM[0]], "v", color=c, ms=10, clip_on=False, zorder=5)
                    off.append(f"{short}: from {t[k0]:.1f} s, lowest {y.min():.0f} deg")
                else:
                    ax.plot(t, y, ls, color=c, lw=2.2, label=lab)
            if model == "replay":                            # the replay that follows both trials closest
                r5 = next(r for r in src if r["plan"] == plan and r["scene"] == "cyl025_stem_block_pt_imp1.xml"
                          and r.get("mu") == 0.5 and r.get("table_mu") is None)
                t, y = sim_trace(r5)
                ax.plot(t, y, ":", color=PT, lw=2.8, label=MU05_LABEL)
            if off:                                          # in the panel's empty corner: right for lat0, left else
                right = plan == "lat0"
                ax.text(0.98 if right else 0.02, 0.04, "Off the plot (seed 0):\n" + "\n".join(off),
                        transform=ax.transAxes, ha="right" if right else "left", va="bottom", fontsize=14,
                        color="#374151", bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#d1d5db"})
            ax.plot(h["t"], h["press_deg"], color=INK, lw=3.0, label="the hand (tracked)")
            ax.set_ylim(*YLIM)
            ax.set_xlim(0.0, 6.0)
            who = "HOM in the loop" if model == "closed_loop" else "the hand's targets replayed"
            name, clock = h["what"].split(", ")
            ax.set_title(f"{name[0].upper() + name[1:]} ({clock})\n{who[0].upper() + who[1:]}", loc="left")
            if row == 1:
                ax.set_xlabel("Time (s)")
            if col == 0:
                ax.set_ylabel(r"Press angle about $x_P$ (deg)")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.grid(color="#eef0f3")
            ax.set_axisbelow(True)
    hs, ls_ = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in ls_:
                hs.append(hh)
                ls_.append(ll)
    order = [ls_.index(x) for x in ["the hand (tracked)", CMD_LABEL] + [s[3] for s in shown] + [MU05_LABEL]
             + ["design pipeline's model (servo 0.72, stiff pads, bare cylinder)"] if x in ls_]
    fig.legend([hs[i] for i in order], [ls_[i] for i in order], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.09))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)


def main():
    d, rows = build()
    figure(d, rows)
    summ = {"image": str(FIG.relative_to(ROOT)), "json": str(OUT.relative_to(ROOT)),
            "hand": {p: {k: v[k] for k in ("track", "ended", "press_3p5_deg", "press_5p5_deg", "slide_x_mm_at_1p6",
                                          "turn_about_z_deg_at_1p6", "t_low5_s")} for p, v in d["hand"].items()}}
    for name in ("closed_loop", "closed_loop_station", "closed_loop_mu_0p5", "replay", "replay_table_mu_0p35",
                 "replay_mu_0p5", "pipeline"):
        summ[name] = {k: {"mean_abs_track_error": v["mean_abs_track_error"], "mean_trace_rms_deg": v["mean_trace_rms_deg"],
                          **{p: {kk: v[p][kk] for kk in ("track", "held_t_end", "held_end", "press_3p5_deg",
                                                          "press_5p5_deg", "trace_rms_deg", "task_trace_rms_deg")}
                             for p in TRIALS}} for k, v in d[name].items()}
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
