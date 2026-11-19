"""S4(b) of the identification queue (lab, 2026-11-18 18:48-18:53; lab/s4b_drag.py; data/s3/*_index_drag*_z*.jsonl):
the index pad's friction on the taped plate, while sliding and at the start of a slide, and whether it sticks and slips.

In each run the index's MCP held the pad face on the tape at about 0.5, 1 or 2 N, and the arm dragged the hand 5 mm up
and 5 mm down at 1 mm/s, four legs per cycle, two cycles. The raw file has the HEX-E at 100 Hz and the TCP at 50 Hz. The
HEX-E's change from the run's F0 turns into P with the header's R_PH; the force along the drag is its component along
the legs' axis_P (nearly z_P), and the normal force its x_P component. Per leg:
  plateau      medians of the force along the drag and of the normal force over the leg's travel from 2 mm to its end
  mu           |along| / normal at the plateau; and per cycle, without the HEX-E z's drift, half the difference of the
               downward and upward plateaus over their mean normal force
  reversal     on the legs that reverse the drag, the force along it against the TCP's travel from the leg's start,
               as 0.1 s means: the travel at
               which it covers 20 % and 80 % of the swing from the previous plateau to this one, and the slope between
               (N/mm, the tangential stiffness of the pad, the finger and the arm in series)
  overshoot    the largest 0.3 s mean in the leg's direction within its first 2 mm, minus the plateau, against the
               largest deviation of 0.3 s means over 2 mm of the plateau (the noise's own maximum): a static friction
               above the sliding friction shows as an overshoot above that. The first leg starts from the hold, after
               the pad has rested on the tape about 20 s (the MCP's steps): its overshoot is taken from the hold's level,
               with |along| / normal at its peak beside the plateau's
  roughness    the standard deviation of the 0.1 s means about a 1 s running median on the plateau, against the same
               on the hold before the first leg: a stick-slip shows as roughness above the hold's
Writes data/s3/20261118-s4b_friction.json and, with --fig, the friction loops (force along the drag against the travel).

    python3 sim/20261118-s4b_drag_friction.py [--fig FILE.png]
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RUNS = ["20261118-184813_index_drag0.5N_z", "20261118-184948_index_drag1N_z", "20261118-185120_index_drag2N_z"]
MEAN_S = 0.1


def run(stem):
    f = STUDY / "data/s3" / stem
    L = [json.loads(x) for x in Path(f"{f}.jsonl").read_text().splitlines() if x.strip()]
    h, legs = L[0], [x for x in L if x.get("phase") == "drag"]
    R = json.loads(Path(f"{f}_raw.jsonl").read_text().splitlines()[0])
    th = np.array(R["hexe_t"])
    FP = (np.array(R["hexe"])[:, :3] - np.array(h["F0_N"])) @ np.array(h["R_PH"]).T
    ax = np.array(legs[0]["axis_P"], float)
    ax /= np.linalg.norm(ax)
    n = max(1, int(round(MEAN_S / np.median(np.diff(th)))))
    k = np.ones(n) / n
    Ft = np.convolve(FP @ ax, k, "same")
    k3 = np.ones(3 * n) / (3 * n)
    Ft3 = np.convolve(FP @ ax, k3, "same")
    Fn = np.convolve(FP[:, 0], k, "same")
    tt = np.array([r[0] for r in R["tcp"]])
    zt = np.array([r[1][2] for r in R["tcp"]]) * 1e3
    z = np.interp(th, tt, zt)
    hold = (th > legs[0]["t0"] - 3.0) & (th < legs[0]["t0"] - 0.2)
    rough = lambda v: float(np.std(v - np.array([np.median(v[max(0, i - 50):i + 50]) for i in range(len(v))])))
    Ft0 = float(np.median(Ft[hold])) if hold.sum() > 10 else None
    out, prev, prev_dir = [], Ft0, None
    for lg in legs:
        s = (th >= lg["t0"]) & (th <= lg["t1"])
        trav = np.abs(z[s] - z[s][0])
        sgn = 1.0 if z[s][-1] > z[s][0] else -1.0                 # +1: the hand moves up along the drag axis
        pl = trav >= 2.0
        ft, fn = float(np.median(Ft[s][pl])), float(np.median(Fn[s][pl]))
        rec = {"cycle": lg["cycle"], "leg": lg["leg"], "dir": "up" if sgn > 0 else "down", "t0": lg["t0"],
               "plateau_along_N": round(ft, 3), "plateau_normal_N": round(fn, 3), "mu": round(abs(ft) / fn, 3),
               "roughness_N": round(rough(Ft[s][pl]), 3)}
        if prev is not None and rec["dir"] != prev_dir:
            swing = ft - prev
            frac = (Ft[s] - prev) / swing
            first = trav <= 2.0
            i20 = int(np.argmax(frac >= 0.2))
            i80 = int(np.argmax(frac >= 0.8))
            ok = frac[i80] >= 0.8 and frac[i20] >= 0.2 and i80 > i20
            rec["reversal"] = {"swing_N": round(swing, 3), "travel_20_mm": round(float(trav[i20]), 3) if ok else None,
                               "travel_80_mm": round(float(trav[i80]), 3) if ok else None,
                               "slope_N_per_mm": round(abs(0.6 * swing) / max(trav[i80] - trav[i20], 1e-3), 2) if ok
                               else None}
            sg = np.sign(swing)
            Fn3 = np.convolve(Fn, np.ones(3) / 3, "same")
            j = int(np.argmax(sg * (Ft3[s][first] - ft)))
            rec["overshoot_N"] = round(float(sg * (Ft3[s][first][j] - ft)), 3)
            rec["mu_at_peak"] = round(float(abs(Ft3[s][first][j]) / Fn3[s][first][j]), 3)
            rec["after_rest"] = prev_dir is None
            win = pl & (trav <= 4.0)
            rec["plateau_max_dev_N"] = round(float(np.max(sg * (Ft3[s][win] - ft))), 3)
        out.append(rec)
        prev, prev_dir = ft, rec["dir"]
    cyc = []
    for c in sorted({r["cycle"] for r in out}):
        up = [r for r in out if r["cycle"] == c and r["dir"] == "up"]
        dn = [r for r in out if r["cycle"] == c and r["dir"] == "down"]
        fu = np.mean([r["plateau_along_N"] for r in up])
        fd = np.mean([r["plateau_along_N"] for r in dn])
        nn = np.mean([r["plateau_normal_N"] for r in up + dn])
        cyc.append({"cycle": c, "friction_N": round(abs(fd - fu) / 2, 3), "normal_N": round(float(nn), 3),
                    "mu": round(abs(fd - fu) / 2 / nn, 3)})
    return {"run": stem, "normal_set_N": h["normal_N"], "legs": out, "cycles": cyc,
            "hold_roughness_N": round(rough(Ft[hold]), 3) if hold.sum() > 100 else None}, (z, Ft, Fn, th, legs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261118-s4b_friction.json"))
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    res, curves = [], []
    for stem in RUNS:
        r, c = run(stem)
        res.append(r)
        curves.append((stem, c))
        print(stem, "hold roughness", r["hold_roughness_N"], "| cycles", json.dumps(r["cycles"]))
        for g in r["legs"]:
            print("   ", g["cycle"], g["leg"], g["dir"], "plateau", g["plateau_along_N"], "normal", g["plateau_normal_N"], "mu",
                  g["mu"], "rough", g["roughness_N"], "rev", g.get("reversal"), "overshoot", g.get("overshoot_N"),
                  "plateau max dev", g.get("plateau_max_dev_N"), "mu at peak", g.get("mu_at_peak"))
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "),
                                       "script": "docs/experiments/20261117-sr2_system_identification/sim/"
                                                 "20261118-s4b_drag_friction.py", "runs": res}, indent=1) + "\n")
    print("wrote", a.out)
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
        for ax, (stem, (z, Ft, Fn, th, legs)), r in zip(axs, curves, res):
            s = (th >= legs[0]["t0"]) & (th <= legs[-1]["t1"])
            ax.plot(z[s] - z[s][0], Ft[s], color="tab:red", lw=1.0, label="force along the drag")
            ax.plot(z[s] - z[s][0], Fn[s], color="tab:blue", lw=1.0, label="normal force (x_P)")
            mu = np.mean([c["mu"] for c in r["cycles"]])
            ax.set_title(f"held at {r['normal_set_N']:g} N: sliding friction {mu:.2f} of the normal", fontsize=13)
            ax.set_xlabel("arm travel up the plate (mm)", fontsize=13)
            ax.set_ylabel("force (N)", fontsize=13)
            ax.tick_params(labelsize=12)
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=12, frameon=False)
        fig.tight_layout()
        fig.savefig(a.fig, dpi=110)
        print("wrote", a.fig)


if __name__ == "__main__":
    main()
