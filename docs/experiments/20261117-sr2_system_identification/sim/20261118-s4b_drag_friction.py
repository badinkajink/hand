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

Phase A of the night of 2026-11-18 (lab/s4b_drag.py --protocol; data/s3/20261118-191748_index_protocol.jsonl, raw in
logs/20261118-sr2_s3_press/) dragged the index pad 5 mm up or down after rests of 1, 10 and 60 s with the pad loaded, at
0.2 and 2 mm/s, at about 0.5, 1 and 2 N. Phases B and C repeated it with the middle pad on the tape and the thumb pad on
the tag plate (the plate's printed tag face); the thumb presses along -x_P, so the normal force is the x_P component
times the sign of the last hold's. Per leg (protocol): the level before it (median of the rest's last half: the
pad stuck, holding the last slide's shear), the plateau (median over the travel from 2.5 mm to the end), the swing
between them, and the overshoot (the largest 0.3 s mean beyond the plateau in the swing's direction within the first
2.5 mm, minus the plateau) against the noise (the plateau's own largest 0.3 s deviation over the same travel). The
sliding friction per N, speed and rest is half the difference of the down and up plateaus.

On 2026-11-19 the same protocol dragged the tips' PLA sides on the tape: the hand turned -92 deg, the yaw holding the
DIP link's side on the plate (lab/s4b_drag.py --joint yaw; the header's "joint", "surface" and "normal_axis_P", here
y_P). A free-air control ran first on the same arm path with the finger 14 mm off the plate. In free air the cable and
the HEX-E alone gave half the down-up difference along the drag of 0.005-0.11 N, growing with speed. For a run with a
control, the control's median force along the drag and along the normal, at the leg's speed and direction, is
subtracted from each leg before the friction is formed; the uncorrected values are kept beside. These runs go to
data/s3/20261119-pla_side_friction.json, so the pads' file keeps its three protocols. Controls at the poses of the
index's and middle's pad drags (run on 2026-11-19, the hand unturned) correct those drags in the same file, under
"pads_corrected"; the thumb's pose had none. A record is read from data/s3, or
beside its raw file in logs/<date>-sr2_s3_press until the lab copies it. Legs at 10 mm/s or faster (the 20 mm/s legs
of 2026-11-19, 0.5 m/s^2) slide steadily for about 0.1 s: they take 0.05 s means, and their plateau is the travel past
2.5 mm where the TCP runs at 0.8 of the leg's speed or more.

    python3 sim/20261118-s4b_drag_friction.py [--fig FILE.png] [--protocol-fig FILE.png]
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RUNS = ["20261118-184813_index_drag0.5N_z", "20261118-184948_index_drag1N_z", "20261118-185120_index_drag2N_z"]
PROTOCOLS = ["20261118-191748_index_protocol", "20261118-200305_middle_protocol", "20261118-204610_thumb_protocol"]
SURFACE = {"index": "tape", "middle": "tape", "thumb": "tag plate"}
CONTROLS = ["20261119-122906_index_protocol", "20261119-130711_index_protocol", "20261119-131125_middle_protocol",
            "20261119-134049_middle_protocol", "20261119-134601_index_protocol", "20261119-135018_middle_protocol"]
PAD_CONTROLS = {  # the pads' drags of 11-18 -> free-air controls run on 11-19 at the same poses (thumb: none)
    "20261118-191748_index_protocol": ["20261119-134601_index_protocol"],
    "20261118-200305_middle_protocol": ["20261119-135018_middle_protocol"]}
SIDE_PROTOCOLS = {  # the PLA sides on the tape: loaded run -> its free-air controls (their legs pooled)
    "20261119-123219_index_protocol": ["20261119-122906_index_protocol", "20261119-130711_index_protocol"],
    "20261119-131412_middle_protocol": ["20261119-131125_middle_protocol", "20261119-134049_middle_protocol"]}
ROOT = STUDY.parents[2]
MEAN_S = 0.1
FAST_MM_S, FAST_MEAN_S = 10.0, 0.05  # legs this fast: 0.05 s means, plateau where the TCP runs at 0.8 of the speed


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


def records(stem):
    """A --protocol run's record (data/s3, else beside the raw file) and its raw HEX-E and TCP samples."""
    logs = ROOT / f"logs/{stem[:8]}-sr2_s3_press"
    rec = STUDY / "data/s3" / f"{stem}.jsonl"
    rec = rec if rec.exists() else logs / f"{stem}.jsonl"
    L = [json.loads(x) for x in rec.read_text().splitlines() if x.strip()]
    raw = [json.loads(x) for x in (logs / f"{stem}_raw.jsonl").read_text().splitlines() if x.strip()]
    return L, raw


def protocol(stem, control=None):
    """Per leg of a --protocol run: level before, plateau, swing, overshoot and the noise's own largest deviation.
    With a free-air control, its median forces at the leg's speed and direction are subtracted before the friction."""
    L, raw = records(stem)
    h = L[0]
    segs = [x for x in L if x.get("phase") in ("rest", "leg")]
    th = np.concatenate([r["hexe_t"] for r in raw])
    FH = np.concatenate([np.array(r["hexe"])[:, :3] for r in raw])
    o = np.argsort(th)
    th, FH = th[o], FH[o]
    tt = np.concatenate([[q[0] for q in r["tcp"]] for r in raw])
    zt = np.concatenate([[q[1][2] for q in r["tcp"]] for r in raw]) * 1e3
    o = np.argsort(tt)
    tt, zt = tt[o], zt[o]
    FP = (FH - np.array(h["F0_N"])) @ np.array(h["R_PH"]).T
    nax = np.array(h.get("normal_axis_P", [1.0, 0.0, 0.0]), float)
    holds = [x for x in L if x.get("phase") == "hold"]
    fh = float(np.dot(holds[-1]["dF_P"], nax)) if holds else 0.0
    sgn = float(np.sign(fh)) if abs(fh) > 0.1 else 1.0
    legs = [x for x in segs if x["phase"] == "leg"]
    ax = np.array(legs[0]["axis_P"], float)
    ax /= np.linalg.norm(ax)
    lat = np.cross(nax, ax)
    lat /= np.linalg.norm(lat)

    def means(sec):
        n = max(1, int(round(sec / np.median(np.diff(th)))))
        return [np.convolve(FP @ u, np.ones(n) / n, "same") for u in (ax, nax, lat)]

    slow, fast = means(0.3), means(FAST_MEAN_S)
    z = np.interp(th, tt, zt)
    vz = np.interp(th, tt, np.gradient(zt, tt))
    out = []
    for k, lg in enumerate(legs):
        quick = lg["speed_mm_s"] >= FAST_MM_S
        F3, Fn_raw, Fl = fast if quick else slow
        Fn = sgn * Fn_raw
        rest = next(x for x in reversed(segs[:segs.index(lg)]) if x["phase"] == "rest")
        rm = (th >= (rest["t0"] + rest["t1"]) / 2) & (th <= rest["t1"])
        s = (th >= lg["t0"]) & (th <= lg["t1"])
        trav = np.abs(z[s] - z[s][0])
        before = float(np.median(F3[rm]))
        pl = (trav >= 2.5) & (np.abs(vz[s]) >= 0.8 * lg["speed_mm_s"]) if quick else trav >= 2.5
        plateau, normal = float(np.median(F3[s][pl])), float(np.median(Fn[s][pl]))
        swing = plateau - before
        sg = np.sign(swing)
        first = trav < 2.5
        over = float(np.max(sg * (F3[s][first] - plateau)))
        noise = float(np.max(sg * (F3[s][pl] - plateau)))
        j = int(np.argmax(sg * (F3[s][first] - plateau)))
        out.append({"N": lg["N"], "speed_mm_s": lg["speed_mm_s"], "rest_s": lg["rest_s"],
                    "dir": "up" if lg["to_mm"] > 0 else "down", "t0": lg["t0"], "before_N": round(before, 3),
                    "plateau_N": round(plateau, 3), "normal_N": round(normal, 3), "swing_N": round(swing, 3),
                    "overshoot_N": round(over, 3), "noise_max_N": round(noise, 3),
                    "overshoot_at_mm": round(float(trav[first][j]), 2),
                    "normal_raw_N": round(float(np.median(Fn_raw[s][pl])), 3),
                    "lateral_N": round(float(np.median(Fl[s][pl])), 3)})
    bias = {}
    if control:
        cl = [r for c in ([control] if isinstance(control, str) else control) for r in protocol(c)[0]["legs"]]
        for v in {r["speed_mm_s"] for r in cl}:
            for d in ("up", "down"):
                c = [r for r in cl if (r["speed_mm_s"], r["dir"]) == (v, d)]
                if c:
                    bias[(v, d)] = (float(np.median([r["plateau_N"] for r in c])),
                                    sgn * float(np.median([r["normal_raw_N"] for r in c])),
                                    float(np.median([r["lateral_N"] for r in c])))
    fric = []
    for N in sorted({r["N"] for r in out}):
        for v in sorted({r["speed_mm_s"] for r in out}):
            for rs in sorted({r["rest_s"] for r in out}):
                up = [r for r in out if (r["N"], r["speed_mm_s"], r["rest_s"], r["dir"]) == (N, v, rs, "up")]
                dn = [r for r in out if (r["N"], r["speed_mm_s"], r["rest_s"], r["dir"]) == (N, v, rs, "down")]
                if up and dn:
                    fk = (dn[0]["plateau_N"] - up[0]["plateau_N"]) / 2
                    nn = (dn[0]["normal_N"] + up[0]["normal_N"]) / 2
                    row = {"N": N, "speed_mm_s": v, "rest_s": rs, "friction_N": round(fk, 3),
                           "normal_N": round(nn, 3), "mu": round(fk / nn, 3) if nn > 0.1 else None}
                    if (v, "up") in bias and (v, "down") in bias:
                        bu, bd = bias[(v, "up")], bias[(v, "down")]
                        fc = fk - (bd[0] - bu[0]) / 2
                        nc = nn - (bd[1] + bu[1]) / 2
                        lc = (dn[0]["lateral_N"] - bd[2] + up[0]["lateral_N"] - bu[2]) / 2
                        row = {**row, "friction_N": round(fc, 3), "normal_N": round(nc, 3),
                               "mu": round(fc / nc, 3) if nc > 0.1 else None, "control_friction_N":
                               round((bd[0] - bu[0]) / 2, 3), "control_normal_N": round((bd[1] + bu[1]) / 2, 3),
                               "lateral_N": round(lc, 3),
                               "mu_with_lateral": round(float(np.hypot(fc, lc)) / nc, 3) if nc > 0.1 else None,
                               "friction_uncorrected_N": row["friction_N"], "normal_uncorrected_N": row["normal_N"],
                               "mu_uncorrected": row["mu"]}
                    fric.append(row)
    return {"run": stem, "finger": h["finger"], "joint": h.get("joint"),
            "surface": h.get("surface", SURFACE.get(h["finger"])), "normal_axis_P": nax.tolist(), "normal_sign": sgn,
            "control": control, "legs": out, "friction": fric}, (th, z, F3, legs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261118-s4b_friction.json"))
    ap.add_argument("--side-out", default=str(STUDY / "data/s3/20261119-pla_side_friction.json"))
    ap.add_argument("--fig", default=None)
    ap.add_argument("--protocol-fig", default=None)
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
    prot, pcurves = [], []
    for stem in PROTOCOLS:
        r, c = protocol(stem)
        prot.append(r)
        pcurves.append(c)
        print(stem, r["finger"], "on the", r["surface"])
        for g in r["legs"]:
            print(f"  {g['N']:3g} N {g['speed_mm_s']:3g} mm/s rest {g['rest_s']:3g} s {g['dir']:4s} before {g['before_N']:+.2f} "
                  f"plateau {g['plateau_N']:+.2f} normal {g['normal_N']:.2f} swing {g['swing_N']:+.2f} overshoot "
                  f"{g['overshoot_N']:.2f} (noise {g['noise_max_N']:.2f}) at {g['overshoot_at_mm']} mm")
        for f in r["friction"]:
            print("   friction", json.dumps(f))
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "),
                                       "script": "docs/experiments/20261117-sr2_system_identification/sim/"
                                                 "20261118-s4b_drag_friction.py", "runs": res, "protocols": prot},
                                      indent=1) + "\n")
    print("wrote", a.out)
    side = {"controls": [], "protocols": []}
    for stem in CONTROLS:
        r = protocol(stem)[0]
        side["controls"].append(r)
        print(stem, r["surface"], "(control)")
        for f in r["friction"]:
            print("   along the drag", json.dumps(f))
    for stem, ctl in SIDE_PROTOCOLS.items():
        r = protocol(stem, control=ctl)[0]
        side["protocols"].append(r)
        print(stem, r["finger"], r["joint"], "on the", r["surface"], "control", ctl)
        for f in r["friction"]:
            print("   friction", json.dumps(f))
    side["pads_corrected"] = []
    for stem, ctl in PAD_CONTROLS.items():
        r = protocol(stem, control=ctl)[0]
        side["pads_corrected"].append(r)
        print(stem, r["finger"], "pad on the", r["surface"], "control", ctl)
        for f in r["friction"]:
            print("   friction", json.dumps(f))
    Path(a.side_out).write_text(json.dumps({"what": "S4(b) on 2026-11-19: the tips' PLA sides on the taped plate, "
                                                    "with the free-air controls, and the pads' drags of 2026-11-18 "
                                                    "corrected by controls at their poses",
                                            "script": "docs/experiments/20261117-sr2_system_identification/sim/"
                                                      "20261118-s4b_drag_friction.py", **side}, indent=1) + "\n")
    print("wrote", a.side_out)
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
