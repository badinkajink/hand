#!/usr/bin/env python3
"""Filmstrips from the bench tape, for showing what a hold and a drop actually look like.

The tracker writes one IR frame every 1/video_hz s beside each trace, at half scale. Only the
right-hand ~55 % of the frame contains the rig; the rest is the operator's desk, and it is
cropped away here rather than published. Each strip is one run, N frames evenly spaced over
the trace, annotated with elapsed time and -- where the tag was still resolving -- the measured
shaft tilt at that instant.

Three strips are produced:
  modes    one hold, one overshoot and one ejection, the last two on the same hand as the hold
           where possible, so the comparison is within a design and not across designs.
  pair     the same hand holding and dropping, for a figure that needs two rows.
  designs  one representative run per hand, ordered by simulated rank.
"""
import argparse, csv, glob, os, sys
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_v1_transfer_figures import bench                              # noqa: E402
from real_v1_transfer_ranking import DESIGN_ID, floor_contact, FLOOR_MM  # noqa: E402

OUT = "paper/figures"
CROP = (280, 50, 640, 350)      # right of the frame: hand, object, both tags, no operator
ACTION_S = 1.5                  # the turn is over by here; the rest of the run is the hold
NFRAME = 6                      # frames inside ACTION_S, plus one final frame of the outcome


def frames(run_tag, n=NFRAME):
    """The first ACTION_S seconds at the tape's own rate, then the last frame of the run.

    The reorientation is over inside about a second and a half; sampling evenly across the
    whole trace spends five of six panels on a shaft that has stopped moving."""
    d = glob.glob(f"logs/tracker/*-{run_tag}_track_frames")
    if not d:
        return [], []
    fs = sorted(glob.glob(d[0] + "/*.jpg"))
    if not fs:
        return [], []
    ts = [int(os.path.basename(f).split("_")[1].split(".")[0]) / 1000.0 for f in fs]
    early = [i for i, t in enumerate(ts) if t <= ACTION_S][:n]
    if len(early) > n:
        early = [early[round(i * (len(early) - 1) / (n - 1))] for i in range(n)]
    ks = early + [len(fs) - 1]
    return [ImageOps.autocontrast(Image.open(fs[k]).convert("L").crop(CROP), cutoff=(2, 18))
            for k in ks], [ts[k] for k in ks]


def trace(run_tag):
    """t -> measured tilt, so a panel can be captioned with what the instrument saw."""
    p = glob.glob(f"logs/tracker/*-{run_tag}_track.csv")
    if not p:
        return []
    return [(float(r["t"]), float(r["deg"])) for r in csv.DictReader(open(p[0])) if r.get("cos")]


def at(tr, t, tol=0.35):
    near = [(abs(x - t), d) for x, d in tr if abs(x - t) < tol]
    return min(near)[1] if near else None


def sheet(rows, path, ncol=NFRAME + 1):
    """One figure, one row per run. Panels are placed on a fixed grid so the frames keep their
    aspect ratio; matplotlib's automatic layout stretches them into postage stamps otherwise."""
    import matplotlib.pyplot as plt
    W, H = CROP[2] - CROP[0], CROP[3] - CROP[1]
    cell = 6.6 * 0.80 / ncol
    fig, ax = plt.subplots(len(rows), ncol,
                           figsize=(6.6, len(rows) * cell * H / W + 0.42), squeeze=False)
    fig.subplots_adjust(left=.20, right=.995, top=.88, bottom=.012, wspace=.03, hspace=.06)
    for i, (run, label, sub) in enumerate(rows):
        ims, ts = frames(run["tag"], ncol - 1)
        tr = trace(run["tag"])
        base = at(tr, ts[0]) if tr else None
        for j, (im, t) in enumerate(zip(ims, ts)):
            a = ax[i][j]
            a.imshow(np.asarray(im), cmap="gray", vmin=0, vmax=255)
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color("#bbb"); sp.set_linewidth(.6)
            if i == 0:
                a.set_title(f"{t:.1f} s", fontsize=7, pad=3)
            if j == len(ims) - 1:
                for sp in a.spines.values():
                    sp.set_color("#555"); sp.set_linewidth(1.1)
            d = at(tr, t)
            txt = "tag lost" if d is None else (
                f"{base - d:+.0f}$\\degree$" if base is not None else f"{d:.0f}$\\degree$")
            a.text(.035, .04, txt, transform=a.transAxes, fontsize=6.5, va="bottom",
                   color="#dddddd" if d is None else "w",
                   bbox=dict(fc="#000000aa", ec="none", pad=1.3))
            if j == 0:
                a.set_ylabel(f"{label}\n{sub}", fontsize=7.5, labelpad=6, rotation=0,
                             ha="right", va="center", linespacing=1.6)
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=220)
    plt.close(fig)


def pick(B, F):
    """Representative runs, chosen by the numbers rather than by eye."""
    def get(design, want, key):
        g = [t for t in B[design] if t["outcome"] == want and t["deg"] is not None]
        return sorted(g, key=key)[0] if g else None
    return dict(
        hold=("sv1_w0099", get("sv1_w0099", "HELD", lambda t: -t["deg"])),
        over=("sv1_w0099", get("sv1_w0099", "DROPPED", lambda t: -t["deg"])),
        eject=("sv1_u0308", get("sv1_u0308", "DROPPED", lambda t: -t["slip"])),
        good=("sv1_w6689", get("sv1_w6689", "HELD", lambda t: -t["deg"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    choices=["all", "modes", "pair", "designs"])
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "figure.facecolor": "white"})
    os.makedirs(OUT, exist_ok=True)
    B, F = bench(10), floor_contact()
    P = pick(B, F)

    def lab(dsg, run, kind):
        return f"{DESIGN_ID[dsg]}  {kind}", f"{run['deg']:+.0f}$\\degree$ turn\n{run['slip']:.0f} mm slip"

    if a.which in ("all", "modes"):
        sel = [("hold", "Held"), ("over", "Overshoot"), ("eject", "Ejection")]
        sheet([(P[k][1],) + lab(P[k][0], P[k][1], v) for k, v in sel if P[k][1]],
              f"{OUT}/fig_filmstrip_modes")

    if a.which in ("all", "pair"):
        sel = [("hold", "Held"), ("over", "Dropped")]
        sheet([(P[k][1],) + lab(P[k][0], P[k][1], v) for k, v in sel if P[k][1]],
              f"{OUT}/fig_filmstrip_pair")

    if a.which in ("all", "designs"):
        rows = []
        for dsg in sorted(B, key=lambda d: DESIGN_ID[d]):
            g = B[dsg]
            nh = sum(t["outcome"] == "HELD" for t in g)
            want = "HELD" if nh * 2 >= len(g) else "DROPPED"
            cand = [t for t in g if t["outcome"] == want and t["deg"] is not None] or \
                   [t for t in g if t["deg"] is not None]
            run = sorted(cand, key=lambda t: -t["deg"])[len(cand) // 2]
            rows.append((run, f"{DESIGN_ID[dsg]}  {want.title()}",
                         f"{nh}/{len(g)} held\n{run['deg']:+.0f}$\\degree$ turn"))
        sheet(rows, f"{OUT}/fig_filmstrip_designs")
    print(f"wrote filmstrips to {OUT}/")


if __name__ == "__main__":
    main()
