#!/usr/bin/env python3
"""Figures and numbers of the contact overview's first part, the chain from the printed fingertip to the simulated contact
(docs/overviews/contact_model.html, section "chain"; owner, 2026-10-26: the contact overview as the link from the real tip
through the FE to the MuJoCo sphere pads, with the gaps between them).

Figure "models": the three models of one fingertip (the print as measured on the UR5, its finite elements, the pressure
law of Drake's field and the 1 mm sphere pads) and the simulated hand, with the comparison at each link. Its numbers are
quoted from the pages that measured them (the docstring of chain_section in contact_overview_page.py lists them).

Figure "indentation": normal force against indentation at the face under the 5 mm sphere of the UR5 session sphere2.
Dots are the HEX-E's force at the registered depth of the 204 face holds. The model curves are medians over the face's
press targets of each model's force with the sphere at that depth along the target's normal: the FE pad
(ur5_tip_calibration_fe.FEPad, compliance as computed), the 812 sphere pads on the FE surface points at 10, 5.4 and
0.90 MPa (ur5_tip_calibration_pads.Pads, builder v2), and the pads of recipe M3 (152.9 MPa behind 8.45 N/mm, fitted to the
FE pad's depth). Panel (b) adds to the FE pad and to the 10 MPa pads the extra indentation a + c F fitted over all 357
holds (fe.json and pads.json of the session).

Numbers (contact_chain_numbers.json): the measured compliance at 1 N under the sphere (sphere2) and under the 12.5 mm rod
(rod1, joint registration), from a power-law fit of registered depth against force over each session's face holds; the
FE pad's and the pads' compliance at 1 N from the curves; and the series stiffness of a fingertip with the servo's
250 N/m (contact sensitivity page of 2026-10-13).

    .venv/bin/python scripts/contact_overview_chain_figures.py      (about 1 min, under 1 GB)
writes docs/overviews/media/contact_chain_models.png, contact_chain_indentation.png and contact_chain_numbers.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))

MEDIA = ROOT / "docs/overviews/media"
UR5 = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data"
S2 = UR5 / "20261022-210852-sphere2"
ROD = UR5 / "20261023-104526-rod1-joint"
NUMBERS = MEDIA / "contact_chain_numbers.json"
SERVO_N_PER_M = 250.0          # each fingertip through the servos at 0.72 N m/rad (contact sensitivity page, 2026-10-13)
PAD_SCENE_N_PER_M = 34.7e3     # the scenes' pad contact on the tool at 1 N, same page
ORANGE, BLUE, GREY = "#BE7514", "#4A7FC4", "#79858E"


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                         "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "mathtext.fontset": "dejavuserif"})
    return plt


def save(fig, name):
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / name, dpi=150, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("wrote", (MEDIA / name).relative_to(ROOT), flush=True)


def face_targets(H, kind="sphere"):
    """Indices of one hold per press target on the face (the first), and all face press holds over 0.2 N."""
    m = (H["kind"] == "press") & (H["region"] == "face") & np.isfinite(H["f_n"]) & (H["f_n"] > 0.2)
    idx = np.nonzero(m)[0]
    seen, first = set(), []
    for i in idx:
        key = H["id"][i]
        if key not in seen:
            seen.add(key)
            first.append(i)
    return np.array(first), idx


def compliance_at(F, d, F0=1.0):
    """d(depth)/dF at F0 from a fit depth = a + b F^p over the holds (mm/N)."""
    from scipy.optimize import curve_fit
    f = lambda x, a, b, p: a + b * np.power(x, p)  # noqa: E731
    (a, b, p), _ = curve_fit(f, F, d, p0=(0.2, 0.4, 0.7), maxfev=20000)
    return float(b * p * F0 ** (p - 1)), dict(a_mm=float(a), b_mm=float(b), p=float(p))


def curves():
    import ur5_tip_calibration_fe as FE
    import ur5_tip_calibration_pads as PD
    H = np.load(S2 / "holds.npz", allow_pickle=False)
    first, allf = face_targets(H)
    pads, fe = PD.Pads(), FE.FEPad()
    R = 5.0
    depth = np.round(np.arange(0.0, 2.41, 0.02), 3)
    G = np.zeros((len(first), len(depth)))     # pads' push along n per MPa
    Fe = np.zeros((len(first), len(depth)))    # FE pad's push, s = 1
    for j, i in enumerate(first):
        n, c0, d0 = H["n"][i], H["c_P"][i], float(H["depth"][i])
        for k, dd in enumerate(depth):
            c = c0 + (d0 - dd) * n
            f, _, _ = pads.law(c, R)
            G[j, k] = max(float(f @ n), 0.0)
            F1, _ = fe.force(c, 1.0, R)
            Fe[j, k] = max(float(-F1 @ n), 0.0)
    return H, first, allf, depth, G, Fe


def fig_indentation(H, first, allf, depth, G, Fe):
    plt = plt_setup()
    P = json.loads((S2 / "pads.json").read_text())
    FJ = json.loads((S2 / "fe.json").read_text())
    E_fe, E_meas, E_tpu = P["E_FE_MPa"], P["E_meas_MPa"], 10.0
    E3, k3 = P["M3"]["E_loc_MPa"], P["M3"]["k_s_N_mm"]
    a_fe, c_fe = FJ["series"]["a_mm"], FJ["series"]["c_mm_N"]          # fitted over all 357 holds, as the text quotes
    a_p, c_p = P["delta_E_TPU"]["law"]["fit"]["fit"]
    med = lambda A: np.median(A, axis=0)  # noqa: E731
    Fq = np.linspace(0.0, 7.5, 301)

    def inverse(dep, F):
        """Depth at force Fq from a monotone force(depth) curve."""
        Fm = np.maximum.accumulate(F)
        ok = np.r_[True, np.diff(Fm) > 1e-9]
        return np.interp(Fq, Fm[ok], dep[ok], right=np.nan)

    # each model's depth at a force, per target, then the median over the face's targets
    d_fe = np.nanmedian([inverse(depth, Fe[j]) for j in range(len(first))], axis=0)
    d_pad = {E: np.nanmedian([inverse(depth, E * G[j]) for j in range(len(first))], axis=0) for E in (E_tpu, E_fe, E_meas)}
    d_m3 = np.nanmedian([inverse(depth, E3 * G[j]) + Fq / k3 for j in range(len(first))], axis=0)
    Fm, dm = H["f_n"][allf], H["depth"][allf]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 5.4), sharey=True)
    for ax in (a1, a2):
        ax.scatter(dm, Fm, s=14, color="0.35", alpha=0.55, lw=0, label=f"Measured, {len(allf)} face holds", zorder=1)
        ax.set_xlim(0, 2.3)
        ax.set_ylim(0, 6.0)
        ax.set_xlabel("Indentation (mm)")
        ax.grid(color="0.9", lw=0.6)
    a1.set_ylabel("Normal force (N)")
    a1.plot(d_fe, Fq, color="black", lw=2.2, label="FE pad", zorder=4)
    a1.plot(d_pad[E_tpu], Fq, color=ORANGE, lw=2.2, label="Pads, 10 MPa (scenes)", zorder=3)
    a1.plot(d_pad[E_fe], Fq, color=ORANGE, lw=2.0, ls="--", label=f"Pads, {E_fe:.1f} MPa (fit to FE)", zorder=3)
    a1.plot(d_m3, Fq, color=BLUE, lw=2.0, ls=(0, (5, 3, 1, 3)), label=f"Pads, {E3:.0f} MPa + {k3:.2f} N/mm spring", zorder=5)
    a1.set_title("(a) The models as computed", fontsize=13)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=12, handlelength=2.4)
    a2.plot(d_fe + a_fe + c_fe * Fq, Fq, color="black", lw=2.2,
            label=f"FE pad, {a_fe:.2f} mm + {c_fe:.2f} mm/N added", zorder=4)
    a2.plot(d_pad[E_tpu] + a_p + c_p * Fq, Fq, color=ORANGE, lw=2.2,
            label=f"Pads, 10 MPa, {a_p:.2f} mm + {c_p:.2f} mm/N added", zorder=3)
    a2.plot(d_pad[E_meas], Fq, color=ORANGE, lw=2.0, ls=":", label=f"Pads, {E_meas:.2f} MPa alone", zorder=3)
    a2.set_title("(b) Fitted to the measured presses", fontsize=13)
    a2.legend(handles=a2.get_legend_handles_labels()[0][1:], loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=12, handlelength=2.4)
    fig.tight_layout()
    save(fig, "contact_chain_indentation.png")

    # compliance at 1 N: the curves' slope and the measured holds' power-law fit
    def slope(dcurve, F0=1.0):
        k = int(np.argmin(np.abs(Fq - F0)))
        return float((dcurve[k + 4] - dcurve[k - 4]) / (Fq[k + 4] - Fq[k - 4]))
    c_meas_s, fit_s = compliance_at(Fm, dm)
    HR = np.load(ROD / "holds.npz", allow_pickle=False)
    _, allr = face_targets(HR, "rod")
    c_meas_r, fit_r = compliance_at(HR["f_n"][allr], HR["depth"][allr])
    num = dict(
        sphere2_face_holds=int(len(allf)), sphere2_face_targets=int(len(first)),
        compliance_1N_mm_N=dict(measured_sphere=c_meas_s, measured_rod=c_meas_r, fe_sphere=slope(d_fe),
                                pads_10MPa_sphere=slope(d_pad[E_tpu]), pads_E_meas_sphere=slope(d_pad[E_meas]),
                                pads_M3_sphere=slope(d_m3)),
        measured_fit_sphere=fit_s, measured_fit_rod=dict(fit_r, n=int(len(allr))),
        measured_depth_1N_mm=float(fit_s["a_mm"] + fit_s["b_mm"]),
        depth_1N_mm=dict(fe=float(np.interp(1.0, Fq, d_fe)), pads_10MPa=float(np.interp(1.0, Fq, d_pad[E_tpu])),
                         pads_E_FE=float(np.interp(1.0, Fq, d_pad[E_fe])), pads_E_meas=float(np.interp(1.0, Fq, d_pad[E_meas])),
                         pads_M3=float(np.interp(1.0, Fq, d_m3))),
        moduli_MPa=dict(E_FE=E_fe, E_meas=E_meas, E_M3=E3, k_M3_N_mm=k3),
        extra_all_holds=dict(fe=[a_fe, c_fe], pads_10MPa=[a_p, c_p], rms_mm=[FJ["series"]["rms_mm"],
                                                                             P["delta_E_TPU"]["law"]["rms_mm"]]))
    srv = 1.0 / SERVO_N_PER_M * 1e3                                  # mm/N
    num["servo"] = dict(servo_N_per_m=SERVO_N_PER_M, pad_scene_N_per_m=PAD_SCENE_N_PER_M,
                        series_scene_N_per_m=1.0 / (1.0 / SERVO_N_PER_M + 1.0 / PAD_SCENE_N_PER_M),
                        series_measured_rod_N_per_m=1e3 / (srv + c_meas_r),
                        series_measured_sphere_N_per_m=1e3 / (srv + c_meas_s))
    return num


def fig_models(num):
    plt = plt_setup()
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
    d = num["depth_1N_mm"]
    w, h, y = 21.4, 21.5, 32.5
    xs = [0.6, 26.4, 52.2, 78.0]
    box(xs[0], y, w, h, "Print on the UR5", "Printed 85A pad\nHEX-E force,\nrobot\u2019s depth\n"
        f"1 N: {num['measured_depth_1N_mm']:.2f} mm", model)
    box(xs[1], y, w, h, "Finite elements", f"Walls, skins,\ngyroid core,\nsolid 12 MPa\n1 N: {d['fe']:.2f} mm", model)
    box(xs[2], y, w, h, "Pressure law", f"Drake\u2019s field,\n1 mm sphere pads;\nscenes: 10 MPa\n"
        f"1 N: {d['pads_10MPa']:.2f} mm", model)
    box(xs[3], y, w, h, "Simulated hand", "Servos: 250 N/m\nper fingertip,\nin series with\nthe contact", model)
    mids = []
    for k in range(3):
        x0, x1 = xs[k] + w + 0.7, xs[k + 1] - 0.7
        arrow(x0, y + h / 2, x1, y + h / 2)
        mids.append((x0 + x1) / 2)
    gw, gh, gy = 31.0, 22.5, 1.0
    gaps = [("Print against FE", "At the measured depth the\nFE force is 5.1 times the\nmeasured; board force\n"
             "0.78\u20131.31 of the HEX-E"),
            ("FE against the law", "One modulus: approach\n13\u201324 % RMS off; a stiff\nmodulus and 9.7 N/mm in\n"
             "series: 5\u20136 % RMS"),
            ("Pads against Drake", "Spin-onset torque 0.94;\nhand\u2019s turn within 3\u00b0 on\n22\u201323 of 24 placements;\n"
             "creep 9\u2013155 times at 1 N")]
    for xm, gx, (t, b) in zip(mids, (0.6, 34.5, 68.4), gaps):
        box(gx, gy, gw, gh, t, b, gap)
        ax.plot([xm, gx + gw / 2], [y - 0.9, gy + gh + 0.9], color="0.4", lw=1.0, ls=":")
    # the pads compared with the presses directly, over the top
    c0, c2 = xs[0] + w / 2, xs[2] + w / 2
    arrow(c0, y + h + 0.7, c2, y + h + 0.7, rad=-0.16, lw=1.2, ls="--")
    ex = num["extra_all_holds"]
    ax.text((c0 + c2) / 2, y + h + 5.6, f"Pads against the presses: {num['moduli_MPa']['E_meas']:.2f} MPa alone, or\n10 MPa "
            f"with {ex['pads_10MPa'][1]:.2f} mm/N in series: {ex['rms_mm'][1] * 1e3:.0f} \u00b5m RMS", ha="center", va="bottom", fontsize=12, style="italic", linespacing=1.35)
    save(fig, "contact_chain_models.png")


def main():
    H, first, allf, depth, G, Fe = curves()
    num = fig_indentation(H, first, allf, depth, G, Fe)
    fig_models(num)
    NUMBERS.write_text(json.dumps(num, indent=1) + "\n")
    print("wrote", NUMBERS.relative_to(ROOT))
    print(json.dumps(num, indent=1))


if __name__ == "__main__":
    main()
