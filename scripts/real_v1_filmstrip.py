#!/usr/bin/env python3
"""Filmstrips from the bench tape, for showing what a hold and a drop actually look like.

The tracker writes one IR frame every 1/video_hz s beside each trace, at half scale. Only the
right-hand ~55 % of the frame contains the rig; the rest is the operator's desk, and it is
cropped away here rather than published. Each strip is one run, N frames evenly spaced over
the trace, annotated with elapsed time and -- where the tag was still resolving -- the measured
shaft tilt at that instant.

Three strips are produced:
  modes    the best-aligned hold, one held and one dropped trial of the same hand, and one
           hand whose drops eject
           where possible, so the comparison is within a design and not across designs.
  pair     the same hand holding and dropping, for a figure that needs two rows.
  designs  one representative run per hand, ordered by simulated rank.
"""
import argparse, csv, glob, json, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_v1_transfer_figures import bench                              # noqa: E402
from real_v1_transfer_ranking import DESIGN_ID                          # noqa: E402

OUT = "paper/figures"
CROP = (310, 45, 640, 345)      # right of the frame: hand, object, both tags, no operator
ACTION_S = 1.8                  # the turn is over by here; the rest of the run is the hold
NFRAME = 6                      # frames inside ACTION_S, plus one final frame of the outcome
DPI = 450                       # the tape is 640x360; anything less throws half of it away

#: Tone mapping. The IR frames come off an auto-exposure that locked on a dim room with the
#: projector off, so more than 10 % of every frame is already at 255 and the shadows sit around
#: 20. The strips used to run PIL's autocontrast with an 18 % highlight cut, which clipped even
#: harder and turned sensor noise into visible grain. Instead the low end is mapped from a
#: percentile of the RUN -- not of the frame, so the panels do not flicker against each other --
#: and a gamma below 1 lifts the midtones without touching the highlights that are already gone.
LO_PCT, HI_PCT, GAMMA = 2.0, 99.5, 0.85

def frames(run_tag, n=NFRAME):
    """The first ACTION_S seconds at the tape's own rate, then the last frame of the run.

    The reorientation is over inside about 1.8 s, so sampling evenly across the whole trace
    spends five of six panels on a shaft that has stopped moving."""
    fs, ts = tape(run_tag)
    if not fs:
        return [], []
    early = [i for i, t in enumerate(ts) if t <= ACTION_S]
    ks = ([early[round(i * (len(early) - 1) / n)] for i in range(n + 1)]
          if len(early) > n else early)
    ks = ks + [len(fs) - 1]
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
        ims, ts = frames(run["tag"], ncol - 2)
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
                # clock labels them all; the last is wherever that run ended.
                a.set_title(f"{t:.1f} s" if j < ncol - 1 else "end of run", fontsize=7, pad=3)
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
    """Representative runs, chosen by the numbers rather than by eye.

    No run is chosen, or labelled, by how far the tag says it turned. On a dropped trial the
    tag keeps reading rotation all the way to the bench -- on w0099 the last 20 deg arrive at
    170--220 deg/s while the cylinder sinks, against 10--60 deg/s for the commanded motion --
    and there is no independent signal in these sessions to say when the shaft actually left
    the hand, because the transfer runs recorded no servo load. Ranking on turn therefore
    picks whichever drop fell hardest. Ranking on slip and on the operator's verdict uses only
    what was measured."""
    def get(design, want, key):
        g = [t for t in B[design] if t["outcome"] == want and t["deg"] is not None]
        return sorted(g, key=key)[0] if g else None

    def aligned(design):
        """The best-measured hold of a hand: alignment, not turn. cos_hold is the mean over
        the last second of a trial that never fell, so unlike a drop's turn it is a settled
        number and ranking on it is safe."""
        g = [t for t in B[design] if t["outcome"] == "HELD" and t["cos_hold"] is not None]
        return max(g, key=lambda t: t["cos_hold"], default=None)

    def median(design, want):
        g = sorted([t for t in B[design] if t["outcome"] == want and t["slip"] is not None],
                   key=lambda t: t["slip"])
        return g[len(g) // 2] if g else None
    return dict(
        best=("sv1_u0060", aligned("sv1_u0060")),
        hold=("sv1_w0099", get("sv1_w0099", "HELD", lambda t: -t["deg"])),
        over=("sv1_w0099", median("sv1_w0099", "DROPPED")),
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
        return f"{DESIGN_ID[dsg]}  {kind}", f"{run['slip']:.0f} mm slip"

    if a.which in ("all", "modes"):
        # The rows are named by what happened, not by a mechanism. The first is the best
        # reorientation the bench produced, and it is a different shape of trajectory as well
        # as a better one -- u0060 carries the shaft to about 70 deg where w0099 stops near
        # 50, so the two hands' last panels do not look alike. The middle two are the same
        # hand under the same plan, held and dropped. The last is a hand whose cylinder
        # travels 54 mm across the bench instead of turning. Slip is the only number on the
        # labels because it is the only one that means the same thing on a held and a dropped
        # trial; the panels carry the tag's own angles and can be read.
        sel = [("best", "Held"), ("hold", "Held"), ("over", "Dropped"), ("eject", "Dropped")]
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
                         f"{nh}/{len(g)} held\n{run['deg']:+.0f}$\\degree$ net turn"))
        sheet(rows, f"{OUT}/fig_filmstrip_designs")
    print(f"wrote filmstrips to {OUT}/")


if __name__ == "__main__":
    main()
