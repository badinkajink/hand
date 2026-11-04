#!/usr/bin/env python3
"""Plot gantry_rom_sweep.py CSVs: StallGuard sg_result vs position, one panel
per axis, one line per pass. Lower sg = more load. Flags 1 mm bins where the
median sg across passes drops below DIP_FRACTION of that axis's overall
median (a tight spot that repeats regardless of direction / other-axis park).

    python3 gantry_rom_plot.py <csv> [<csv> ...]   -> <csv stem>.png per file
"""

import csv
import pathlib
import sys
from collections import defaultdict
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIP_FRACTION = 0.8


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r["cruise"] == "1"]
    by_joint = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["other_joint"], r["other_mm"], r["dir"])
        by_joint[int(r["joint"])][key].append((float(r["mm"]), int(r["sg"]), int(r["stall"])))
    return by_joint


BIN_MM = 2.0


def binned(points):
    """BIN_MM-wide medians -- raw sg swings +-20 sample to sample, so the
    position trend only shows after binning."""
    bins = defaultdict(list)
    for mm, sg, _ in points:
        bins[int(mm // BIN_MM)].append(sg)
    keys = sorted(bins)
    return [(k + 0.5) * BIN_MM for k in keys], [median(bins[k]) for k in keys]


def summarize(joint, passes):
    all_sg = [sg for p in passes.values() for _, sg, _ in p]
    base = median(all_sg)
    xs, ys = binned([q for p in passes.values() for q in p])
    tight = [(x, y) for x, y in zip(xs, ys) if y < DIP_FRACTION * base]
    stalls = sum(st for p in passes.values() for _, _, st in p)
    print(f"J{joint}: median sg {base:.0f} (min {min(all_sg)}), stall samples {stalls}")
    if tight:
        spans, start, prev = [], tight[0][0], tight[0][0]
        for b, _ in tight[1:]:
            if b != prev + BIN_MM:
                spans.append((start, prev))
                start = b
            prev = b
        spans.append((start, prev))
        worst = min(tight, key=lambda t: t[1])
        print(f"   tight (median < {DIP_FRACTION:.0%} of baseline): "
              + ", ".join(f"{a - BIN_MM / 2:.0f}-{b + BIN_MM / 2:.0f} mm" for a, b in spans)
              + f"; worst {worst[1]:.0f} at {worst[0]} mm")
    else:
        print("   no repeatable tight spots")
    return base


def main():
    for path in sys.argv[1:]:
        by_joint = load(path)
        fig, axes = plt.subplots(len(by_joint), 1, figsize=(10, 3.2 * len(by_joint)), squeeze=False)
        for ax, (joint, passes) in zip(axes[:, 0], sorted(by_joint.items())):
            base = summarize(joint, passes)
            for (oj, omm, d), p in sorted(passes.items()):
                xs, ys = binned(p)
                ax.plot(xs, ys, lw=1.2, ls="-" if d == "out" else "--", label=f"J{oj}@{omm} {d}")
            xs, ys = binned([q for p in passes.values() for q in p])
            ax.plot(xs, ys, color="black", lw=2.2, label="all passes")
            ax.axhline(DIP_FRACTION * base, color="grey", lw=0.8, ls=":")
            ax.set_title(f"J{joint}: StallGuard load signal (lower = more resistance)")
            ax.set_xlabel("mm from home")
            ax.set_ylabel("sg_result")
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        out = pathlib.Path(path).with_suffix(".png")
        fig.savefig(out, dpi=110)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
