#!/usr/bin/env python3
"""The pinched sphere hanging after the pick-up, on the hand and in simulation with each fingertip contact model (owner,
2026-11-16 21:4x: "can we replicate this for the sphere pickup from dropped? as you said, this is where it should
demonstrate the most difference"). The hand's first pick-up of 2026-11-16 (15:43-15:47, thumb against middle along x_P):
the lift stood the lying sphere up from 85 to 43 deg, and once its block left the table the sphere hung in the pinch at
43.1-41.9 deg for about a minute (data/20261116-pinch_pickup/, lift and slip telemetry). In the pinch the sphere can turn
only about the line through the two contacts, its contacts' normal: friction against spin about the normal (a patch of
pads has it, a point contact has none) is what holds it. The simulation (logs/20261116-sim_real_reversal/pickup_sim.py
--hang) starts from the hand's measured ball centre in the pinch at 42 deg, on the calibrated servo, holds the object 1.5
s while the squeeze builds to the plan's grip, then lets it go; per contact variant the tilt over the next 10 s. The
blind twists' error under the same contact settings is beside it (rev_predict.py rows). Writes
data/20261116-pinch_pickup/20261116-pickup_contact_compare.json, media/20261116-pickup_hang_contact.png (bench trials
folder) and figures.json pickup_hang (co-design folder).

    .venv/bin/python scripts/hardware/hom_pickup_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "docs/experiments/20261115-hom_bench_trials"
REV = ROOT / "logs/20261116-sim_real_reversal"
PICK = BENCH / "data/20261116-pinch_pickup"
OUT = PICK / "20261116-pickup_contact_compare.json"
FIG = BENCH / "media/20261116-pickup_hang_contact.png"
FIGS_JSON = ROOT / "docs/experiments/20261110-codesign_formulation/media/20261110-figures.json"
FREE_S = 16.0          # s into the hand's lift: the block off the table and the tilt settled (43.9 deg at 14 s, 43.1 at 16)
# (key, hang rows file, variant, twist rows file, label)
VARIANTS = (("pads_stiff_ns", "pickup_hang10_ns.json", "pads_stiff_ns", "predict_budget084_kp035_pads_imp100_ns30_trace.json",
             "pads, stiff tangential, no creep (noslip)"),
            ("pads_soft", "pickup_hang10.json", "pads_soft", "predict_budget084_kp035_trace.json",
             "pads, soft tangential (the twist's calibration)"),
            ("point_stiff_ns", "pickup_hang10_ns.json", "point_stiff_ns", "predict_budget084_kp035_pt_imp100_ns30_trace.json",
             "point contact, stiff, no creep"),
            ("point_soft", "pickup_hang10.json", "point_soft", "predict_budget084_kp035_pt_trace.json",
             "point contact, soft"))
HAND_TWIST = {"4": 0.379, "8": 0.0, "12": 0.326, "13": 0.378}   # data/20261116-trial_tracking.json, the budgeted trials


def hand_hang():
    rows = [json.loads(x) for x in (PICK / "20261116-154429_pinch_lift_telemetry.jsonl").read_text().splitlines() if x.strip()]
    slip = [json.loads(x) for x in (PICK / "20261116-154534_pinch_slip_telemetry.jsonl").read_text().splitlines() if x.strip()]
    t0 = rows[0]["t"]
    pts = [(x["t"] - t0 - FREE_S, x["obj"]["tilt"]) for x in rows + slip
           if (x.get("obj") or {}).get("tilt") is not None and x["t"] - t0 >= FREE_S]
    # the hang ends where the slip first opened a finger and the tilt fell (41.9 to 13.2 deg): the first sample under 40
    end = next((k for k in range(len(pts)) if pts[k][1] < 40.0), len(pts))
    t, y = np.array(pts[:end]).T
    return {"t": t.round(2).tolist(), "tilt": y.round(2).tolist(), "free_s_into_lift": FREE_S,
            "held_s": round(float(t[-1]), 1), "tilt_first": round(float(y[0]), 2), "tilt_last": round(float(y[-1]), 2)}


def twist_error(fname):
    d = json.loads((REV / fname).read_text())
    tr = {}
    for g in HAND_TWIST:
        rs = [r for r in d["rows"] if r["grasp"] == f"grid2dgr_c329_g{g}" and "error" not in r]
        tr[g] = float(np.mean([r["track"] or 0.0 for r in rs])) if rs else 0.0
    return round(float(np.mean([abs(tr[g] - HAND_TWIST[g]) for g in HAND_TWIST])), 3), {g: round(v, 3) for g, v in tr.items()}


def build():
    out = {"hand": hand_hang(), "variants": {}}
    for key, hf, var, tf, label in VARIANTS:
        r = next(x for x in json.loads((REV / hf).read_text()) if x["variant"] == var)
        held = [x for x in r["rows"] if x["phase"] == "held"]
        t0 = held[0]["t"] - 0.02
        mae, tr = twist_error(tf)
        # out of the pinch: the ball's centre 10 mm below its height at release (the tilt after that is the fallen
        # object's, standing or lying on the floor)
        fell = next((round(x["t"] - t0, 2) for x in held if x["z_mm"] < held[0]["z_mm"] - 10.0), None)
        at = {k: (None if fell is not None and float(k) >= fell else v) for k, v in r["summary"]["held_at"].items()}
        out["variants"][key] = {"label": label, "scene": r["scene"], "held_at": at, "fell_s": fell,
                                "t": [round(x["t"] - t0, 2) for x in held], "tilt": [x["tilt"] for x in held],
                                "z_mm": [x["z_mm"] for x in held], "twist_mean_abs_error": mae, "twist_track": tr}
    OUT.write_text(json.dumps(out) + "\n")
    return out


def figure(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 16, "axes.titlesize": 17, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13})
    INK, CAL, GREY = "#1f2937", "#d97706", "#6b7280"
    style = {"pads_stiff_ns": (CAL, "-", 2.6), "pads_soft": (CAL, ":", 2.2), "point_stiff_ns": (GREY, "-", 1.6),
             "point_soft": (GREY, ":", 1.6)}
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    for key, v in d["variants"].items():
        c, ls, lw = style[key]
        t, y = np.array(v["t"]), np.array(v["tilt"])
        if v["fell_s"] is not None:                     # out of the pinch: drawn up to the fall, then marked
            k = int(np.searchsorted(t, v["fell_s"]))
            ax.plot([t[k - 1]], [y[k - 1]], "x", color=c, ms=11, mew=2.4)
            ax.annotate(f"out of the pinch at {v['fell_s']:.1f} s", xy=(t[k - 1], y[k - 1]),
                        xytext=(t[k - 1] + 0.25, y[k - 1] + 6.0), fontsize=12, color=c,
                        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9})
            t, y = t[:k], y[:k]
        ax.plot(t, y, ls, color=c, lw=lw, label=f"{v['label']}; twist error {v['twist_mean_abs_error']:.2f}")
    h = d["hand"]
    ax.plot(h["t"], h["tilt"], color=INK, lw=3.2, label=f"the hand: {h['tilt_first']:.0f} to {h['tilt_last']:.0f} deg over "
                                                         f"{h['held_s']:.0f} s")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 60)
    ax.set_xlabel("Time after the block left the table (s)")
    ax.set_ylabel("The sphere's tilt from vertical (deg)")
    ax.set_title("The pinched sphere after the pick-up, on the hand and in simulation", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=12.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#eef0f3")
    ax.set_axisbelow(True)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)


FILMS = REV / "films"                    # pickup_sim.py --hang --film logs/20261116-sim_real_reversal/films
HAND_FILM = BENCH / "media/20261116-pinch_pickup_controlled_slip.mp4"
FILM = BENCH / "media/20261116-pickup_hang_hand_vs_sim.mp4"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def film():
    """The hand's lift and hang (its film's second camera, 9-19 s, cropped to the hand: the lift at 3x, the hang and
    the first slip steps at 2x) beside the simulated hang, 10 s at real time, with creep-free pads, the twist-calibrated
    pads and point contact, two by two."""
    import subprocess
    sims = [FILMS / f"pickup_hang_{v}.mp4" for v in ("pads_stiff_ns", "pads_soft", "point_stiff_ns")]
    if not HAND_FILM.exists() or not all(f.exists() for f in sims):
        return None
    label = "the hand, its film at 2-3x, the lift and then the hang at 42 deg"
    vf = (f"[0:v]fps=25,crop=560:420:1070:28,scale=640:480,drawbox=x=0:y=0:w=640:h=27:color=white:t=fill,"
          f"drawtext=fontfile={FONT}:text='{label}':x=8:y=5:fontsize=17:fontcolor=0x141414[h];"
          "[h][1:v]hstack[top];[2:v][3:v]hstack[bot];[top][bot]vstack,format=yuv420p[v]")
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", "9", "-t", "10", "-i", str(HAND_FILM)] + \
        sum((["-i", str(f)] for f in sims), []) + \
        ["-filter_complex", vf, "-map", "[v]", "-t", "10", "-c:v", "libx264", "-crf", "20", "-preset", "slow",
         "-movflags", "+faststart", str(FILM)]
    subprocess.run(cmd, check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "6.24", "-i", str(FILM), "-frames:v", "1", "-q:v", "3",
                    str(FILM.with_suffix(".jpg"))], check=True)
    return str(FILM.relative_to(ROOT))


def main():
    d = build()
    figure(d)
    summ = {"image": str(FIG.relative_to(ROOT)), "film": film(),
            "hand": {k: d["hand"][k] for k in ("held_s", "tilt_first", "tilt_last")},
            "variants": {k: {"label": v["label"], "held_at": v["held_at"], "fell_s": v["fell_s"],
                             "twist_mean_abs_error": v["twist_mean_abs_error"]} for k, v in d["variants"].items()}}
    fi = json.loads(FIGS_JSON.read_text())
    fi["pickup_hang"] = summ
    FIGS_JSON.write_text(json.dumps(fi, indent=1, default=float))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
