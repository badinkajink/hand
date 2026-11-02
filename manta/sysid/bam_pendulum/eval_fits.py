#!/usr/bin/env python3
"""Score fitted BAM (Better Actuator Models) friction models on the single-pendulum logs.

    ../.venv-bam/bin/python eval_fits.py [--fits fits20k] [--val-kp 4] [--plot fits20k/compare.png]

Mean absolute position error (deg) of a full closed-loop rollout (simulated firmware control),
split into the training logs and the held-out P gain, then broken down by trajectory and hole.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import bam_scs0009

bam_scs0009.register()
from bam.model import load_model  # noqa: E402
from bam.simulate import Simulator  # noqa: E402


def rollout(model, log):
    pos, _, _ = Simulator(model).rollout_log(log, simulate_control=True)
    return np.array(pos)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fits", default="fits20k")
    ap.add_argument("--logdir", default="data_processed/single")
    ap.add_argument("--val-kp", type=int, default=4)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()

    logs = [json.load(open(f)) for f in sorted(glob.glob(f"{a.logdir}/*.json"))]
    fits = {os.path.basename(f)[:-5]: f for f in sorted(glob.glob(f"{a.fits}/m*.json"))}
    summary = {}
    for name, path in fits.items():
        model = load_model(path)
        err = defaultdict(list)
        for log in logs:
            real = np.array([e["position"] for e in log["entries"]])
            mae = np.degrees(np.mean(np.abs(rollout(model, log) - real)))
            split = "val" if log["kp"] == a.val_kp else "train"
            err[split].append(mae)
            err[(split, log["trajectory"])].append(mae)
            err[(split, log["hole_mm"])].append(mae)
        summary[name] = {k: float(np.mean(v)) for k, v in err.items()}

    trajs = sorted({log["trajectory"] for log in logs})
    holes = sorted({log["hole_mm"] for log in logs})
    print(f"mean absolute position error, deg   (validation = held-out P gain {a.val_kp})")
    print(f"{'model':6} {'train':>6} {'val':>6} | " + " ".join(f"{t[:13]:>13}" for t in trajs)
          + " | " + " ".join(f"{h:>4.0f}mm" for h in holes) + "   (val columns)")
    for name, s in summary.items():
        print(f"{name:6} {s['train']:6.2f} {s['val']:6.2f} | "
              + " ".join(f"{s[('val', t)]:13.2f}" for t in trajs) + " | "
              + " ".join(f"{s[('val', h)]:6.2f}" for h in holes))
    for name, path in fits.items():
        p = json.load(open(path))
        keep = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items() if k not in ("model", "actuator")}
        print(f"{name}: {keep}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        picks = [l for l in logs if l["kp"] == a.val_kp and l["hole_mm"] in (90.0, 30.0)]
        picks.sort(key=lambda l: (l["hole_mm"], l["trajectory"]))
        fig, axes = plt.subplots(len(picks) // 2, 2, figsize=(13, 2.3 * len(picks) // 2), sharex=True)
        models = {n: load_model(fits[n]) for n in ("m1", "m6") if n in fits}
        for ax, log in zip(axes.T.flat, picks):
            t = [e["timestamp"] for e in log["entries"]]
            ax.plot(t, np.degrees([e["goal_position"] for e in log["entries"]]), "k:", lw=1, label="goal")
            ax.plot(t, np.degrees([e["position"] for e in log["entries"]]), "k", lw=1.6, label="real")
            for (n, m), c in zip(models.items(), ("tab:orange", "tab:blue")):
                ax.plot(t, np.degrees(rollout(m, log)), c, lw=1.1, label=f"sim {n.upper()}")
            ax.set_title(f"{log['trajectory']}  {log['hole_mm']:.0f} mm  P={log['kp']} (held out)", fontsize=9)
            ax.set_ylabel("deg")
        axes.flat[0].legend(fontsize=8, ncol=4)
        for ax in axes[-1]:
            ax.set_xlabel("s")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=110)
        print("plot ->", a.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
