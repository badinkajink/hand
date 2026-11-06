#!/usr/bin/env python3
"""The contact filter of the HOM controller at task speeds, with the magnetic fingertips' real errors (owner, 2026-11-06:
"how well would this work on real? can we check with our calibration data?").

The UR5 rolls of scripts/hom/hom_filter_real_check.py move the contact 1.5 to 2.1 mm at 0.05 to 0.07 mm/s, so they test
the filter on the tips' noise and bias but not on a contact that moves at the speeds of the tasks. Here each roll's
measured error, the board GP's estimate less the robot's contact point, projected on the roll's direction and sampled
at the controller's 50 Hz, is added to a contact that moves along that direction at a task speed for 1 s and then holds
for 1 s. The filter of Eqs. (4) to (6) runs on it in one dimension, as hom_review_figures.filt does: the estimate moves by
the reference speed plus k times the gap to the measurement, with k from Eq. (6) (k_min 5 1/s, k_max 100 1/s, alpha
20 mm/s). Settings: "k_min", no commanded roll or slide (the four tasks); "commanded", the reference speed equal to the
contact's own speed during the move, with k_max 100 1/s (the library's), 25 1/s ("commanded_k25") or k_min
("commanded_kmin": the gain does not rise). Each roll's error sequence is used from 40 s into the roll action so that
the press has settled; the speeds are those of the tasks' contacts: 2.3 mm/s (the twist ramp's contacts rolled up to
7 mm in 3 s), 7.4 mm/s (a 40 deg/s roll of the v4b pad's 10.55 mm apex radius) and 20 mm/s.

    .venv/bin/python scripts/hom/hom_filter_speed_check.py
Reads rows/20261106-filter_real_traces.npz; writes rows/20261106-filter_speed_check.json and
media/20261106-filter_speed.png (docs/experiments/20261029-hom_control_review/).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "docs/experiments/20261029-hom_control_review"
TRACES = D / "rows/20261106-filter_real_traces.npz"
OUT = D / "rows/20261106-filter_speed_check.json"
FIG = D / "media/20261106-filter_speed.png"
SPEEDS = (0.0, 2.3, 7.4, 20.0)           # mm/s
RATE = 50.0                              # Hz, the bench's controller rate
K_MIN, K_MAX, ALPHA = 5.0, 100.0, 20.0   # 1/s, 1/s, mm/s
T_MOVE, T_HOLD, T_SKIP = 1.0, 1.0, 0.5   # s; the first T_SKIP s of the sequence settle the filter


def gain(g_ref, k_max=K_MAX):
    return K_MIN + (k_max - K_MIN) * (1.0 - math.exp(-3.0 * abs(g_ref) / ALPHA))


SETTINGS = {"raw": None, "k_min": (False, K_MAX), "commanded": (True, K_MAX), "commanded_k25": (True, 25.0),
            "commanded_kmin": (True, K_MIN)}


def run(err, v, commanded, dt=1.0 / RATE, k_max=K_MAX):
    """1-D filter on truth + err for a contact that rests T_SKIP s, moves at v for T_MOVE s, then holds T_HOLD s."""
    n = len(err)
    t = np.arange(n) * dt
    x = np.clip(t - T_SKIP, 0.0, T_MOVE) * v
    moving = (t >= T_SKIP) & (t < T_SKIP + T_MOVE)
    z = x + err
    est = np.empty(n)                    # est[i]: the estimate for time i, made from the measurements before it
    e = z[0]
    for i in range(n):
        est[i] = e
        g = v if (commanded and moving[i]) else 0.0
        e = e + (g + gain(g, k_max) * (z[i] - e)) * dt   # ContactFilter.update: the point for the period's end
    hold = t >= T_SKIP + T_MOVE + 0.2
    return t, x, z, est, moving, hold


def errors_by_roll():
    """Each roll's error along its roll direction at 50 Hz (mm), from 40 s into the action to its end."""
    Z = np.load(TRACES, allow_pickle=False)
    out = []
    for k, key in enumerate(Z["keys"]):
        t, truth, raw, sdir = Z[f"{key}/t"], Z[f"{key}/truth"], Z[f"{key}/raw"], Z[f"{key}/s_dir"]
        e = (raw - truth) @ np.asarray(sdir, float)
        tt = np.arange(t[0], t[-1], 1.0 / RATE)
        e50 = np.interp(tt, t, e)
        keep = tt >= tt[0] + 40.0 if tt[-1] - tt[0] > 45.0 else np.ones_like(tt, bool)
        out.append(dict(key=str(key), id=str(Z["ids"][k]), seat=int(Z["seat"][k]), press_N=float(Z["press_N"][k]),
                        err=e50[keep]))
    return out


def main():
    rolls = errors_by_roll()
    n = int((T_SKIP + T_MOVE + T_HOLD) * RATE)
    res = {}
    for v in SPEEDS:
        for setting, spec in SETTINGS.items():
            cmd, kmx = spec if spec else (False, K_MAX)
            mv, hd = [], []
            for r in rolls:
                err = r["err"]
                for s0 in range(0, len(err) - n, n):            # every window of the sequence, back to back
                    t, x, z, est, moving, hold = run(err[s0:s0 + n], v, cmd, k_max=kmx)
                    y = z if setting == "raw" else est
                    mv.append(np.sqrt(np.mean((y[moving] - x[moving]) ** 2)))
                    hd.append(np.sqrt(np.mean((y[hold] - x[hold]) ** 2)))
            res[f"{v:g}"] = res.get(f"{v:g}", {})
            res[f"{v:g}"][setting] = dict(move_rms_median=round(float(np.median(mv)), 3),
                                          move_rms_p90=round(float(np.percentile(mv, 90)), 3),
                                          hold_rms_median=round(float(np.median(hd)), 3), windows=len(mv))
    noise = [float(np.std(np.diff(r["err"])) / math.sqrt(2.0)) for r in rolls]
    bias = [float(np.mean(r["err"])) for r in rolls]
    out = dict(what=__doc__.split("\n\n")[0], rate_hz=RATE, speeds_mm_s=list(SPEEDS), k_min=K_MIN, k_max=K_MAX,
               alpha_mm_s=ALPHA, t_move_s=T_MOVE, n_rolls=len(rolls),
               err_noise_mm=dict(median=round(float(np.median(noise)), 3), p90=round(float(np.percentile(noise, 90)), 3)),
               err_bias_mm=dict(median_abs=round(float(np.median(np.abs(bias))), 3),
                                p90_abs=round(float(np.percentile(np.abs(bias), 90)), 3)),
               lag_kmin_mm={f"{v:g}": round(v / K_MIN, 2) for v in SPEEDS}, results=res)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    figure(out, rolls)
    return 0


def figure(out, rolls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 12.5,
                         "mathtext.fontset": "stix"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.28, 3.7), gridspec_kw={"width_ratios": [1.25, 1]})
    # (a) one window of a 4 N roll's real error on a contact moving at 7.4 mm/s
    r = next(x for x in rolls if x["press_N"] == 4.0)
    n = int((T_SKIP + T_MOVE + T_HOLD) * RATE)
    err = r["err"][:n]
    for setting, col, lab in (("k_min", "#1f63b4", "filter, $k_{\\min}$ (the tasks)"),
                              ("commanded", "#d9730d", "roll commanded, $k_{\\max}$ 100"),
                              ("commanded_kmin", "#2e7d32", "roll commanded, $k$ = $k_{\\min}$")):
        cmd, kmx = SETTINGS[setting]
        t, x, z, est, moving, hold = run(err, 7.4, cmd, k_max=kmx)
        if setting == "k_min":
            a.plot(t, z, ".", ms=3, color="#9a9a9a", label="tip's estimate (real error)")
            a.plot(t, x, color="k", lw=2.0, label="contact point")
        a.plot(t, est, color=col, lw=1.8, label=lab)
    a.set_xlabel("time (s)")
    a.set_ylabel("along the pad (mm)")
    a.set_title("(a) A contact moving at 7.4 mm/s", fontsize=13)
    a.legend(loc="lower right", fontsize=10.5, frameon=True, framealpha=0.9, edgecolor="none")
    # (b) RMS error while moving against the speed
    sp = out["speeds_mm_s"]
    for setting, col, mk, lab in (("raw", "#9a9a9a", "s", "tip's estimate"), ("k_min", "#1f63b4", "o", "$k_{\\min}$"),
                                  ("commanded", "#d9730d", "^", "commanded, $k_{\\max}$ 100"),
                                  ("commanded_k25", "#b07aa1", "v", "commanded, $k_{\\max}$ 25"),
                                  ("commanded_kmin", "#2e7d32", "D", "commanded, $k_{\\min}$")):
        y = [out["results"][f"{v:g}"][setting]["move_rms_median"] for v in sp]
        b.plot(sp, y, marker=mk, color=col, lw=1.6, ms=7, label=lab)
    b.set_xlabel("contact speed (mm/s)")
    b.set_ylabel("RMS error while moving (mm)")
    b.set_title("(b) Error against speed", fontsize=13)
    b.legend(loc="upper left", fontsize=10.5, frameon=False)
    b.grid(color="#eeeeee")
    b.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIG, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", FIG.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
