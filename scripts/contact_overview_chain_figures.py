#!/usr/bin/env python3
"""Figures and numbers of the contact overview's first part, the chain from the printed fingertip to the simulated contact
(docs/overviews/contact_model.html, section "chain"; owner, 2026-10-26: the contact overview as the link from the real tip
through the FE to the MuJoCo sphere pads, with the gaps between them).

Since 2026-10-30 the part quotes the four-magnet v3 fingertips (owner, 2026-10-30: the figure of 2026-10-22 was "8 days old
on a fingertip we are not going to use"; one modulus, errors accepted at the extrema; forces before indentation). Its
data are the fit of scripts/v3_tip_contact_model.py (docs/experiments/20261030-v3_tip_contact_model/data/
20261030-contact_model_fit.json and 20261030-numbers.json); the session sphere2 of tip 0 remains only as a comparison.

Figure "models": the three models of one fingertip (the print as measured on the UR5, its finite elements, the pressure
law of Drake's field and the 1 mm sphere pads) and the simulated hand, with the comparison at each link.

Figure "indentation": (a, b) normal force against registered indentation at the face under the 5 mm sphere and the
12.5 mm rod, both v3 tips' face holds, with the pads and Drake's field at their joint modulus, the FE at its joint scale
and as computed, and the scenes' 10 MPa pads; (c) measured over model force by load level for the pads and the FE.

Numbers (contact_chain_numbers.json): the fits, the measured face compliance at 1 N (power law over the face holds), the
models' depth at 1 N, the FE's force at s = 1 over the measured, the force ratio by level, and the series stiffness of a
fingertip with the servo's 250 N/m (contact sensitivity page of 2026-10-13).

    .venv/bin/python scripts/contact_overview_chain_figures.py      (seconds; run scripts/v3_tip_contact_model.py and
                                                                     scripts/v3_tip_contact_model_figures.py first)
writes docs/overviews/media/contact_chain_models.png, contact_chain_indentation.png and contact_chain_numbers.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v3_tip_contact_model_figures as V  # noqa: E402

MEDIA = ROOT / "docs/overviews/media"
NUMBERS = MEDIA / "contact_chain_numbers.json"
SERVO_N_PER_M = 250.0          # each fingertip through the servos at 0.72 N m/rad (contact sensitivity page, 2026-10-13)
PAD_SCENE_N_PER_M = 34.7e3     # the scenes' pad contact on the tool at 1 N, same page
ORANGE, BLUE, GREY, BLACK = V.ORANGE, V.BLUE, V.GREY, V.BLACK
JOINT = ("v3id1_low", "v3id2_low", "v3id1_rod", "v3id2_rod")


def save(fig, name):
    import matplotlib.pyplot as plt
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", (MEDIA / name).relative_to(ROOT), flush=True)


def ratios(d, keys, model):
    """Measured over model force of every hold of `keys` at the joint modulus (pads, field) or scale (fe), with levels."""
    J = d["fits"]["joint"]
    r, lv = [], []
    for k in keys:
        h = d["sessions_detail"][k]["holds"]
        f = np.array(h["f_n"])
        with np.errstate(divide="ignore", invalid="ignore"):
            if model == "pads":
                m = f / (J["E_pads_MPa"] * np.array(h["Gp"]))
            elif model == "field":
                m = f / (J["E_field_MPa"] * np.array(h["Gf"]))
            else:
                fe = np.array(h["fe_fit"])
                m = np.where(fe > 0, f / np.where(fe > 0, fe, 1.0), np.nan)
        m[~np.isfinite(m) | (m <= 0)] = np.nan
        r.append(m)
        lv.append(np.array(h["level"]))
    return np.concatenate(r), np.concatenate(lv)


def by_level(r, lv, min_n=8):
    out = {}
    for L in sorted(set(lv.tolist())):
        v = r[(lv == L) & np.isfinite(r)]
        if len(v) >= min_n:
            out[f"{L:g}"] = dict(n=int(len(v)), median=float(np.median(v)), p25=float(np.percentile(v, 25)),
                                 p75=float(np.percentile(v, 75)), within25=float(np.mean(np.abs(np.log(v)) < np.log(1.25))))
    return out


def fig_indentation(d):
    plt = V.plt_setup()
    import matplotlib.gridspec as gridspec
    J = d["fits"]["joint"]
    Ep, Ef, sf = J["E_pads_MPa"], J["E_field_MPa"], J["s_fe"]
    fig = plt.figure(figsize=(10.4, 9.6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.82], hspace=0.78, wspace=0.08)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    a3 = fig.add_subplot(gs[1, :])
    for ax, keys, title, xmax in ((a1, V.LOW, "(a) 5 mm sphere, face", 2.6), (a2, V.ROD, "(b) 12.5 mm rod, face", 1.8)):
        F, Dp, T = V.face_holds(d, keys)
        for k, shade in zip(keys, ("0.62", "0.30")):
            m = T == k
            ax.scatter(Dp[m], F[m], s=9, color=shade, alpha=0.55, lw=0, zorder=1, label=f"Measured, {d['sessions'][k]['tip']}")
        dep, Gp, Gf, F1, Fs, _ = V.model_curves(d, keys)
        ax.plot(dep, Ep * Gp, color=ORANGE, lw=2.6, zorder=4, label=f"Pads, {Ep:.2f} MPa (fitted)")
        ax.plot(dep, Ef * Gf, color=BLUE, lw=2.0, ls=(0, (6, 3)), zorder=4, label=f"Drake’s field, {Ef:.2f} MPa (fitted)")
        ax.plot(dep, Fs, color=BLACK, lw=2.0, zorder=3, label=f"FE, {12.0 / sf:.1f} MPa (one scale, fitted)")
        ax.plot(dep, F1, color=BLACK, lw=1.4, ls=":", zorder=3, label="FE, 12 MPa (as computed)")
        ax.plot(dep, V.E_SCENES * Gp, color=ORANGE, lw=1.4, ls=":", zorder=3, label="Pads, 10 MPa (scenes)")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 8.6)
        ax.set_xlabel("Registered indentation (mm)")
        ax.set_title(title)
        ax.grid(color="0.9", lw=0.6)
    a1.set_ylabel("Normal force (N)")
    a2.set_yticklabels([])
    h, lab = a1.get_legend_handles_labels()
    a1.legend(h[2:] + h[:2], lab[2:] + lab[:2], loc="upper center", bbox_to_anchor=(1.04, -0.15), ncol=2, frameon=False,
              fontsize=12, handlelength=2.6, markerscale=2.0)
    a3.axhspan(0.8, 1.25, color="0.93", zorder=0)
    a3.axhline(1.0, color="0.5", lw=0.8, zorder=1)
    off = dict(pads=-0.04, fe=0.04)
    for ind, keys in (("low", V.LOW), ("rod", V.ROD)):
        for model, col, lab_ in (("pads", ORANGE, f"Pads at {Ep:.2f} MPa"), ("fe", BLACK, f"FE at {12.0 / sf:.1f} MPa")):
            r, lv = ratios(d, keys, model)
            bl = by_level(r, lv)
            x = np.array([float(L) for L in bl]) * np.exp(off[model] * 3)
            med = np.array([v["median"] for v in bl.values()])
            lo = np.array([v["p25"] for v in bl.values()])
            hi = np.array([v["p75"] for v in bl.values()])
            a3.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o" if ind == "low" else "s", color=col,
                        mfc=col if ind == "low" else "white", mec=col, ms=7, lw=1.2, capsize=2.5, ls="-" if ind == "low" else "--",
                        zorder=3, label=f"{lab_}, {'5' if ind == 'low' else '12.5'} mm {'sphere' if ind == 'low' else 'rod'}")
    a3.set_xscale("log")
    a3.set_yscale("log")
    a3.set_xticks([0.25, 0.5, 1, 2, 3.5, 5, 8])
    a3.set_xticklabels(["0.25", "0.5", "1", "2", "3.5", "5", "8"])
    a3.set_yticks([0.4, 0.5, 0.6, 0.8, 1, 1.25, 1.6, 2, 2.5])
    a3.set_yticklabels(["0.4", "0.5", "0.6", "0.8", "1", "1.25", "1.6", "2", "2.5"])
    a3.minorticks_off()
    a3.set_ylim(0.38, 2.7)
    a3.set_xlabel("Load level of the press (N)")
    a3.set_ylabel("Measured over model force")
    a3.set_title("(c) Force at the registered indentation, both v3 tips")
    a3.grid(color="0.9", lw=0.6)
    a3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=12)
    save(fig, "contact_chain_indentation.png")


def numbers(d, N):
    J = d["fits"]["joint"]
    rp, lv = ratios(d, JOINT, "pads")
    rf, _ = ratios(d, JOINT, "field")
    re_, _ = ratios(d, JOINT, "fe")
    mid = (lv >= 1.0) & (lv <= 2.0)
    fin = lambda v: v[np.isfinite(v)]  # noqa: E731
    within = lambda v: float(np.mean(np.abs(np.log(fin(v))) < np.log(1.25)))  # noqa: E731
    lo, ro = N["low"], N["rod"]
    srv = 1e3 / SERVO_N_PER_M                                     # mm/N
    out = dict(
        source=dict(fit="docs/experiments/20261030-v3_tip_contact_model/data/20261030-contact_model_fit.json",
                    numbers="docs/experiments/20261030-v3_tip_contact_model/data/20261030-numbers.json",
                    page="docs/experiments/20261030-v3_tip_contact_model/20261030-v3_fingertip_contact_models.html"),
        joint=J, fits={k: d["fits"][k] for k in ("v3_low", "v3_rod", "v3_tall", "v3id1_low_rod", "v3id2_low_rod",
                                                  "old_sphere2", "old_rod1")},
        holds_v3=int(sum(d["sessions"][k]["n"] for k in d["sessions"] if k.startswith("v3"))),
        by_level=dict(pads=by_level(rp, lv), field=by_level(rf, lv), fe=by_level(re_, lv)),
        mid=dict(median_pads=float(np.median(fin(rp[mid]))), within25_pads=within(rp[mid]), within25_field=within(rf[mid])),
        fe_level_range=[min(v["median"] for v in by_level(re_, lv).values()), max(v["median"] for v in by_level(re_, lv).values())],
        pads_level_range=[min(v["median"] for v in by_level(rp, lv).values()), max(v["median"] for v in by_level(rp, lv).values())],
        measured=dict(sphere=dict(depth_1N_mm=lo["measured_depth_1N_mm"], compliance_1N_mm_N=lo["measured_compliance_1N_mm_N"],
                                  face_holds=lo["face_holds"]),
                      rod=dict(depth_1N_mm=ro["measured_depth_1N_mm"], compliance_1N_mm_N=ro["measured_compliance_1N_mm_N"],
                               face_holds=ro["face_holds"])),
        depth_1N_mm=dict(sphere=lo["depth_1N_mm"], rod=ro["depth_1N_mm"]),
        compliance_1N_mm_N=dict(sphere=lo["compliance_1N_mm_N"], rod=ro["compliance_1N_mm_N"]),
        fe_s1_over_measured=dict(sphere=lo["fe_s1_over_measured_face"], rod=ro["fe_s1_over_measured_face"]),
        rbar_mm=dict(sphere=lo["rbar_mm"], rod=ro["rbar_mm"]), twists=N["twists"], spring=d.get("spring", {}).get("joint"),
        depth_resid_um={k: d["sessions_detail"][k]["depth_resid_mm"]["pads"]["median"] * 1e3 for k in d["sessions_detail"]},
        servo=dict(servo_N_per_m=SERVO_N_PER_M, pad_scene_N_per_m=PAD_SCENE_N_PER_M,
                   series_scene_N_per_m=1.0 / (1.0 / SERVO_N_PER_M + 1.0 / PAD_SCENE_N_PER_M),
                   series_measured_rod_N_per_m=1e3 / (srv + ro["measured_compliance_1N_mm_N"]),
                   series_measured_sphere_N_per_m=1e3 / (srv + lo["measured_compliance_1N_mm_N"])))
    if out["spring"]:
        out["spring"] = {k: v for k, v in out["spring"].items() if k != "grid"}
    return out


def fig_models(num):
    plt = V.plt_setup()
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0", fc=fc, ec="black", lw=1.0))
        ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=13, weight="bold")
        ax.text(x + w / 2, y + h - 6.4, body, ha="center", va="top", fontsize=12, linespacing=1.4)

    def arrow(x0, y0, x1, y1, rad=0.0, lw=1.5, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="<|-|>", lw=lw, color="black", ls=ls, mutation_scale=12,
                                    connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

    model, gap = "#eef3fa", "#fbf1e4"
    d = num["depth_1N_mm"]["sphere"]
    w, h, y = 21.4, 21.5, 32.5
    xs = [0.6, 26.4, 52.2, 78.0]
    box(xs[0], y, w, h, "Print on the UR5", "Two v3 tips,\nHEX-E force,\nrobot’s depth\n"
        f"1 N: {num['measured']['sphere']['depth_1N_mm']:.2f} mm", model)
    box(xs[1], y, w, h, "Finite elements", f"Walls, skins,\ngyroid core,\nsolid 12 MPa\n1 N: {d['fe_s1']:.2f} mm", model)
    box(xs[2], y, w, h, "Pressure law", f"Drake’s field,\n1 mm sphere pads;\nscenes: 10 MPa\n"
        f"1 N: {d['pads_10MPa']:.2f} mm", model)
    box(xs[3], y, w, h, "Simulated hand", "Servos: 250 N/m\nper fingertip,\nin series with\nthe contact", model)
    mids = []
    for k in range(3):
        x0, x1 = xs[k] + w + 0.7, xs[k + 1] - 0.7
        arrow(x0, y + h / 2, x1, y + h / 2)
        mids.append((x0 + x1) / 2)
    gw, gh, gy = 31.0, 22.5, 1.0
    fe = num["fe_s1_over_measured"]
    gaps = [("Print against FE", f"At the measured depth the\nFE force is {fe['rod']:.1f}–{fe['sphere']:.1f} times the\n"
             "measured; magnets move\n0.71–0.74 of the FE’s"),
            ("FE against the law", "One modulus: approach\n13–24 % RMS off;\nthe print spreads a load,\n"
             "a local law does not"),
            ("Pads against Drake", "Spin-onset torque 0.94;\nhand’s turn within 3° on\n22–23 of 24 placements;\n"
             "creep 9–155 times at 1 N")]
    for xm, gx, (t, b) in zip(mids, (0.6, 34.5, 68.4), gaps):
        box(gx, gy, gw, gh, t, b, gap)
        ax.plot([xm, gx + gw / 2], [y - 0.9, gy + gh + 0.9], color="0.4", lw=1.0, ls=":")
    c0, c2 = xs[0] + w / 2, xs[2] + w / 2
    arrow(c0, y + h + 0.7, c2, y + h + 0.7, rad=-0.16, lw=1.2, ls="--")
    J = num["joint"]
    bl = num["by_level"]["pads"]
    ax.text((c0 + c2) / 2, y + h + 5.6, f"Pads fitted to the presses: {J['E_pads_MPa']:.2f} MPa under the sphere and the rod;\n"
            f"measured force {bl['0.25']['median']:.2f} times the pads’ at 0.25 N, {bl['8']['median']:.2f} times at 8 N",
            ha="center", va="bottom", fontsize=12, style="italic", linespacing=1.35)
    save(fig, "contact_chain_models.png")


def main():
    d = V.load()
    N = json.loads(V.NUMBERS.read_text())
    fig_indentation(d)
    num = numbers(d, N)
    fig_models(num)
    NUMBERS.write_text(json.dumps(num, indent=1, default=float) + "\n")
    print("wrote", NUMBERS.relative_to(ROOT))


if __name__ == "__main__":
    main()
