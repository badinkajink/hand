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
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_v1_transfer_figures import bench                              # noqa: E402
from real_v1_transfer_ranking import DESIGN_ID, EJECT_SLIP_MM           # noqa: E402

OUT = "paper/figures"
CROP = (310, 45, 640, 345)      # right of the frame: hand, object, both tags, no operator
ACTION_S = 1.5                  # the turn is over by here; the rest of the run is the hold
NFRAME = 6                      # frames inside ACTION_S, plus one final frame of the outcome
DPI = 450                       # the tape is 640x360; anything less throws half of it away

#: Tone mapping. The IR frames come off an auto-exposure that locked on a dim room with the
#: projector off, so more than 10 % of every frame is already at 255 and the shadows sit around
#: 20. The strips used to run PIL's autocontrast with an 18 % highlight cut, which clipped even
#: harder and turned sensor noise into visible grain. Instead the low end is mapped from a
#: percentile of the RUN -- not of the frame, so the panels do not flicker against each other --
#: and a gamma below 1 lifts the midtones without touching the highlights that are already gone.
LO_PCT, HI_PCT, GAMMA = 2.0, 99.5, 0.85


def frames(run_tag, n=NFRAME, t_peak=None):
    """The first ACTION_S seconds at the tape's own rate, the instant of the largest turn,
    then the last frame of the run.

    The reorientation is over inside about a second and a half, so sampling evenly across the
    whole trace spends five of six panels on a shaft that has stopped moving. The peak column
    is there because the row's label is the peak: on an overshoot the turn runs a few tenths
    past the window and the label would otherwise assert something the panels do not show."""
    fs, ts = tape(run_tag)
    if not fs:
        return [], []

    def spread(pool, k):
        return ([pool[round(i * (len(pool) - 1) / (k - 1))] for i in range(k)]
                if len(pool) > k else pool)

    early = [i for i, t in enumerate(ts) if t <= ACTION_S]
    kp = (min(range(len(ts)), key=lambda i: abs(ts[i] - t_peak))
          if t_peak is not None else None)
    if kp is None:
        ks = spread(early, n + 1)
    else:
        # The peak ALWAYS gets its own column, whether or not it falls inside the action
        # window, because the column is titled "peak turn" for every row at once. Letting it
        # be an ordinary frame on the rows whose peak came early put that title over a panel
        # showing something else.
        ks = spread([i for i in early if i != kp], n) + [kp]
    ks = ks + [len(fs) - 1]                     # always n + 2 panels, so the grid is fixed
    raw = [np.asarray(Image.open(fs[k]).convert("L").crop(CROP), dtype=float) for k in ks]
    lo, hi = np.percentile(np.concatenate([a.ravel() for a in raw]), [LO_PCT, HI_PCT])
    return [_tone(a, lo, hi) for a in raw], [ts[k] for k in ks]


def tape(run_tag):
    """(frame files, their timestamps) for one run, in order."""
    d = glob.glob(f"logs/tracker/*-{run_tag}_track_frames")
    fs = sorted(glob.glob(d[0] + "/*.jpg")) if d else []
    return fs, [int(os.path.basename(f).split("_")[1].split(".")[0]) / 1000.0 for f in fs]


def _tone(a, lo, hi):
    """Stretch, then a 3x3 median against the IR speckle, then a light unsharp to put the
    edges back. Denoise alone reads as smeared; the pair reads as a cleaner photograph."""
    x = np.clip((a - lo) / max(hi - lo, 1.0), 0, 1) ** GAMMA * 255.0
    x = ndimage.median_filter(x, size=3)
    return np.clip(x + 0.6 * (x - ndimage.gaussian_filter(x, 1.1)), 0, 255)


def trace(run_tag):
    """t -> measured tilt, so a panel can be captioned with what the instrument saw."""
    p = glob.glob(f"logs/tracker/*-{run_tag}_track.csv")
    if not p:
        return []
    return [(float(r["t"]), float(r["deg"])) for r in csv.DictReader(open(p[0])) if r.get("cos")]


def turn_label(run):
    """How a row states its rotation.

    PEAK, not net: on an overshoot the shaft passes the label's angle and the tag dies there,
    so the net figure would contradict the panels. An ejection is the exception -- its peak is
    the shaft tumbling out of the grasp on the way to the table, which is not a turn the hand
    made -- so those rows carry the net turn they actually produced."""
    eject = run["slip"] is not None and run["slip"] >= EJECT_SLIP_MM
    v, what = (run["deg"], "net") if eject else (peak_at(run["tag"])[0], "peak")
    return f"{v:+.0f}$\\degree$ {what} turn"


def peak_at(run_tag):
    """(largest rotation any TAPED frame reports, the time of that frame).

    Deliberately not the trace's own maximum. The tag is sampled at about 29 Hz and the tape
    writes about 3.7 frames a second, so the true peak almost always falls between two
    pictures -- and a row labelled with a number no panel shows is a row the reader cannot
    check. Reading the peak off the tape makes the label and the overlays the same fact."""
    tr, (_, ts) = trace(run_tag), tape(run_tag)
    if not tr or not ts:
        return None, None
    base = at(tr, ts[0])
    if base is None:
        return None, None
    seen = [(base - d, t) for t, d in ((t, at(tr, t)) for t in ts) if d is not None]
    return max(seen) if seen else (None, None)


def at(tr, t, tol=0.35):
    near = [(abs(x - t), d) for x, d in tr if abs(x - t) < tol]
    return min(near)[1] if near else None


def sheet(rows, path, ncol=NFRAME + 1):
    """One figure, one row per run. Panels are placed on a fixed grid so the frames keep their
    aspect ratio; matplotlib's automatic layout stretches them into postage stamps otherwise."""
    import matplotlib.pyplot as plt
    W, H = CROP[2] - CROP[0], CROP[3] - CROP[1]
    cell = 6.6 * 0.84 / ncol
    fig, ax = plt.subplots(len(rows), ncol,
                           figsize=(6.6, len(rows) * cell * H / W + 0.42), squeeze=False)
    fig.subplots_adjust(left=.16, right=.995, top=.88, bottom=.012, wspace=.03, hspace=.06)
    for i, (run, label, sub) in enumerate(rows):
        ims, ts = frames(run["tag"], ncol - 2, peak_at(run["tag"])[1])
        tr = trace(run["tag"])
        base = at(tr, ts[0]) if tr else None
        for j, (im, t) in enumerate(zip(ims, ts)):
            a = ax[i][j]
            a.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color("#bbb"); sp.set_linewidth(.6)
            if i == 0:
                # The action columns are within 0.1 s of each other across rows, so row 0's
                # clock labels them all. The last two are not: the peak lands wherever the
                # run put it, and the final frame is the end of that run.
                a.set_title(f"{t:.1f} s" if j < ncol - 2 else
                            ("peak turn" if j == ncol - 2 else "end of run"),
                            fontsize=7, pad=3)
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
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=DPI)
    plt.close(fig)


def pick(B):
    """Representative runs, chosen by the numbers rather than by eye."""
    def get(design, want, key):
        g = [t for t in B[design] if t["outcome"] == want and t["deg"] is not None]
        return sorted(g, key=key)[0] if g else None
    # A stall is the mode that hides: it slips like a hold and simply stops short, so the
    # run to show is the one nearest its hand's mean dropped turn, not an extreme.
    st = [t for t in B["sv1_u1364"] if t["outcome"] == "DROPPED" and t["deg"] is not None]
    mu = sum(t["deg"] for t in st) / len(st) if st else 0.0
    return dict(
        hold=("sv1_w0099", get("sv1_w0099", "HELD", lambda t: -t["deg"])),
        over=("sv1_w0099", get("sv1_w0099", "DROPPED", lambda t: -t["deg"])),
        stall=("sv1_u1364", min(st, key=lambda t: abs(t["deg"] - mu)) if st else None),
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
    B = bench(10)
    P = pick(B)

    def lab(dsg, run, kind):
        return (f"{DESIGN_ID[dsg]}  {kind}",
                f"{turn_label(run)}\n{run['slip']:.0f} mm slip")

    if a.which in ("all", "modes"):
        # Three rows, not four. The stall is a real mode and the table names it, but the
        # figure is the paper's one picture of what a trial looks like and it earns its space
        # by contrast: two rows of the SAME hand under the SAME plan, held then overshot, and
        # one row of the grasp letting go. A fourth row of a turn that merely stops early
        # dilutes that without showing anything the drops figure does not already say.
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
                         f"{nh}/{len(g)} held\n{turn_label(run)}"))
        sheet(rows, f"{OUT}/fig_filmstrip_designs")
    print(f"wrote filmstrips to {OUT}/")


if __name__ == "__main__":
    main()
