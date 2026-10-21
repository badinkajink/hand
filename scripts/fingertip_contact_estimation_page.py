#!/usr/bin/env python3
"""Result page of the model-based contact estimation study (2026-10-20): the measurement model of the five-chip board under
the magnetic pad, its calibration at rest, the FE contact basis, the estimator's errors on simulated presses, its behaviour
without a press, and the live browser viewer.

Reads docs/experiments/20261020-fingertip_contact_estimation/ (results.json and data/20261020-uncertainty_maps.npz from
scripts/fingertip_contact_estimation_study.py; media/ from scripts/fingertip_contact_film.py and the viewer), draws the
figures into media/, renders the mathematics with scripts/texsvg.py, applies scripts/retro_style.py and writes
docs/experiments/20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html.

    .venv/bin/python scripts/fingertip_contact_estimation_page.py
"""
from __future__ import annotations

import base64
import html
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = ROOT / "docs/experiments/20261020-fingertip_contact_estimation"
MEDIA = D / "media"
PAGE = D / "20261020-model_based_contact_estimation.html"
TEMPLATE = ROOT / "scripts/fingertip_contact_estimation_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
FE_PAGE = ROOT / "docs/experiments/20261019-v2_tip_fe_checks/20261019-v2_fingertip_stiffness_checks.html"
STATUS_PAGE = ROOT / "docs/experiments/20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html"
ADDR = ("0x0C", "0x11", "0x12", "0x13", "0x10")
U = ("U1", "U2", "U3", "U4", "U5")


def esc(s):
    return html.escape(str(s), quote=True)


def rel(p):
    import os
    return os.path.relpath(p, D)


def link(p, text):
    return f'<a href="{esc(rel(p))}">{text}</a>'


def code(p):
    return f"<code>{esc(p)}</code>"


def img_uri(png, width=1100, quality=88):
    from PIL import Image
    im = Image.open(png).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def fig(n, png, caption, width=1100):
    return (f'<figure id="fig{n}"><img alt="Figure {n}" src="{img_uri(png, width)}">'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def film(n, mp4, poster, caption):
    v = "data:video/mp4;base64," + base64.b64encode(Path(mp4).read_bytes()).decode()
    return (f'<figure id="fig{n}"><video controls muted loop playsinline preload="metadata" poster="{img_uri(poster, 1100)}" '
            f'src="{v}" style="width:100%"></video><figcaption>Figure {n}. {caption}</figcaption></figure>')


def table(head, body, num_cols=(), caption=None, n=None):
    cap = f"<caption>Table {n}. {caption}</caption>" if caption else ""
    th = "".join(f'<th{" class=n" if k in num_cols else ""}>{h}</th>' for k, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td{" class=n" if k in num_cols else ""}>{c}</td>' for k, c in enumerate(r)) + "</tr>"
                 for r in body)
    return f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


# ------------------------------------------------------------------------------------------------ figures

def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                         "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12})
    return plt


def fig_rest(res, out):
    plt = plt_setup()
    c = res["calibration"]
    meas, mod = np.array(c["measured_uT"]) / 1e3, np.array(c["model_uT"]) / 1e3
    perm = c["best"]["perm"]
    fig, ax = plt.subplots(figsize=(11, 4.0), dpi=100)
    x = np.arange(15)
    ax.bar(x - 0.2, meas.ravel(), 0.4, color="#303030", label="Measured")
    ax.bar(x + 0.2, mod.ravel(), 0.4, color="#d08040", label="Model, best fit")
    ax.set_xticks(x, [a for i in range(5) for a in "xyz"])
    for i in range(5):
        ax.text(3 * i + 1, -0.16, f"{ADDR[i]} at {U[perm[i]]}", transform=ax.get_xaxis_transform(), ha="center", fontsize=13)
        if i:
            ax.axvline(3 * i - 0.5, color="#999", lw=0.8)
    ax.axhline(0, color="#000", lw=0.8)
    ax.set_ylabel("Field at rest (mT), chip axes")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_maps(npz, out):
    plt = plt_setup()
    z = np.load(npz)
    ys, zs = z["ys"], z["zs"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.4), dpi=100, sharex=True, sharey=True)
    for r, (key, lbl) in enumerate((("as_found", "Magnets as found (35 % of N52, mixed poles)"),
                                    ("n52", "N52, poles one way"))):
        S = z[key]
        for c, (k, name, unit, vmax) in enumerate(((0, "Pressing force 1-sigma", "N", None),
                                                   (None, "Position 1-sigma, sqrt(sd_y sd_z)", "mm", None))):
            v = S[..., 0] if k == 0 else np.sqrt(S[..., 3] * S[..., 4])
            ax = axs[r, c]
            hi = np.nanpercentile(v, 95)
            im = ax.pcolormesh(zs, ys, v, shading="nearest", cmap="viridis", vmin=0, vmax=hi)
            ax.scatter(z["chips"][:, 2], z["chips"][:, 1], marker="s", s=80, facecolor="none", edgecolor="#fff", lw=1.5)
            ax.scatter(z["magnets"][:, 2], z["magnets"][:, 1], marker="o", s=60, facecolor="none", edgecolor="#f04040", lw=1.5)
            ax.set_title(f"{lbl}\n{name} ({unit})", fontsize=12)
            ax.set_aspect("equal")
            cb = fig.colorbar(im, ax=ax, shrink=0.85)
            cb.set_label(unit)
            if r == 1:
                ax.set_xlabel("z, along the finger (mm)")
            if c == 0:
                ax.set_ylabel("y, across (mm)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_quiet(res, out):
    plt = plt_setup()
    q = res["no_contact"]
    S, G = np.array(q["series"]), np.array(q["g_series"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.2), dpi=100, sharex=True)
    a1.plot(S[:, 0], S[:, 1], color="#303030", lw=1, label="Explained chi-square of the best contact")
    a1.axhline(q["threshold"], color="#c00000", lw=1.2, ls="--", label=f"Threshold {q['threshold']:.0f}")
    a1.set_ylabel("Chi-square")
    a1.legend(loc="upper left")
    a1.grid(alpha=0.3)
    for k, (lbl, col) in enumerate((("x", "#c00000"), ("y", "#008000"), ("z", "#0040ff"))):
        a2.plot(S[:, 0], G[:, k], color=col, lw=1, label=f"Background change along {lbl}")
    a2.set_ylabel("Background (uT)")
    a2.set_xlabel("Time (s)")
    a2.legend(loc="upper left", ncol=3)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ page

def build():
    res = json.loads((D / "results.json").read_text())
    MEDIA.mkdir(exist_ok=True)
    fig_rest(res, MEDIA / "20261020-rest_fit.png")
    fig_maps(D / "data/20261020-uncertainty_maps.npz", MEDIA / "20261020-uncertainty_maps.png")
    fig_quiet(res, MEDIA / "20261020-no_press.png")
    c, fe, q = res["calibration"], res["fe"], res["no_contact"]
    b = c["best"]
    syn = {(s["magnets"], s["window"]): s for s in res["synthetic"]}
    n52 = syn[("N52, poles one way", "50 ms")]
    af = syn[("as found", "50 ms")]
    fmin, fmax = min(c["field_mT"]), max(c["field_mT"])
    perm = b["perm"]
    assign = ", ".join(f"{ADDR[i]} at {U[perm[i]]}" for i in range(5))
    mp = res["maps"]

    lede = (f"A measurement model of the five-chip board under the magnetic pad, calibrated from its reading at rest, turns "
            f"the 15 field components into a contact point and a contact force without a robot. The rest fit identifies which "
            f"chip address sits at which position, the chips' axes, the magnets' poles and their strength ({b['k']:.2f} of "
            f"N52 for the magnets in the pad today; {b['rms_mT']:.2f}&#8202;mT RMS against {fmin:.1f}&#8211;{fmax:.1f}&#8202;mT). "
            f"An FE basis of {fe['load_cases']} patch loads maps forces to magnet motions. On simulated presses with N52 "
            f"magnets and 50&#8202;ms of samples, the RMS errors are {n52['rms']['y_mm']:.2f}&#8202;mm in position and "
            f"{n52['rms']['f_press_N']:.2f}&#8202;N in pressing force; with today's magnets {af['rms']['y_mm']:.1f}&#8211;"
            f"{af['rms']['z_mm']:.1f}&#8202;mm and {af['rms']['f_press_N']:.2f}&#8202;N. A browser viewer shows the estimate, "
            f"the FE deformation and the pad pressure at 30 frames per second. No estimate has been compared with a "
            f"reference force yet.")

    terms = ('<dl class="terms">'
             "<dt>Tip frame</dt><dd>$x$ along the pad's normal toward the palmar face, $y$ across the finger, $z$ along it "
             "from the mount toward the tip end, in mm (the FE's frame). The pad's face spans about $y=-6$ to $6$ and "
             "$z=1$ to $20$&#8202;mm.</dd>"
             "<dt>Reading, reading change</dt><dd>The 15 field components the five chips report, each in its own axes, "
             "in &#181;T; the reading change $\\Delta\\mathbf b$ is the reading minus a baseline taken at rest.</dd>"
             "<dt>Noise per sample</dt><dd>Standard deviation of one conversion with the pad at rest: 1.65&#8202;&#181;T "
             "on a chip's $x$ and $y$ axes and 2.57&#8202;&#181;T on $z$ (GAIN_SEL 1, RES 1, DIG_FILT 5; a conversion "
             "every 7.0&#8211;7.1&#8202;ms). The viewer averages the 50&#8202;ms before each frame, about 7 conversions, "
             "which divides the noise by $\\sqrt 7$.</dd>"
             "<dt>Magnet strength $k$</dt><dd>A magnet's moment relative to a 3&#8202;&#215;&#8202;2&#8202;mm N52 cylinder "
             "(remanence 1.45&#8202;T).</dd>"
             "<dt>Rest residual</dt><dd>RMS over the 15 components of the measured minus the modelled reading at rest, "
             "in mT.</dd>"
             "<dt>Sensitivity $S(\\mathbf u)$</dt><dd>The reading change per newton of a patch load centred at face point "
             "$\\mathbf u=(y,z)$: three columns, for pressing along $-x$ and shearing along $+y$ and $+z$.</dd>"
             "<dt>Pressing force, shear</dt><dd>$f_\\mathrm{press}$, $f_y$, $f_z$: the force the object applies to the pad, "
             "in N.</dd>"
             "<dt>Explained chi-square $e$</dt><dd>The decrease of the weighted squared residual when the best contact is "
             "added to the nuisance terms (Eq. 5). The viewer shows a contact when $e>25$.</dd>"
             "<dt>Model 1-sigma</dt><dd>Square root of a diagonal entry of the inverse Fisher information at the estimate "
             "(Eq. 6): the spread the noise alone causes when the model is exact.</dd>"
             "<dt>RMS error</dt><dd>Root mean square of estimate minus truth over 300 simulated presses: $y$, $z$ uniform "
             "over the face, $f_\\mathrm{press}$ uniform in 0.3&#8211;3&#8202;N, shears normal with 0.3&#8202;N standard "
             "deviation, a background change of 5&#8202;&#181;T and a strength change of 0.1&#8202;% per press.</dd>"
             "<dt>Background change $\\Delta\\mathbf g$, strength change $\\varepsilon$</dt><dd>A uniform field change in "
             "the tip frame (the finger turning in Earth's field), in &#181;T; a relative change of the magnets' strength "
             "(N52 loses about 0.12&#8202;% per &#176;C).</dd></dl>")

    eq1 = (r"$$\mathbf B(\mathbf p)=\sum_{j=1}^{5}\sum_{q} w_q\,\frac{\mu_0}{4\pi}\,"
           r"\frac{3\,\hat{\mathbf r}_{jq}(\hat{\mathbf r}_{jq}\!\cdot\mathbf m_j)-\mathbf m_j}{\lVert\mathbf r_{jq}\rVert^{3}},\qquad"
           r"\mathbf r_{jq}=\mathbf p-\mathbf c_j-\boldsymbol\rho_q,\qquad \mathbf m_j=k\,s_j\,\frac{B_r V}{\mu_0}\,\mathbf e_x$$")
    eq2 = (r"$$\mathbf b_c=R_c\big(\mathbf B(\mathbf p_c)+\mathbf g\big),\qquad \mathbf z_c=s\,\mathbf e_x,\qquad"
           r"\mathbf x_c=\cos(\theta_0+\phi_c)\,\mathbf e_y+\sin(\theta_0+\phi_c)\,\mathbf e_z,\qquad \mathbf y_c=\mathbf z_c\times\mathbf x_c$$")
    eq3 = (r"$$\min_{s_j,\;\pi,\;s,\;\theta_0,\;k,\;\boldsymbol\delta,\;\mathbf g}\;\sum_{i=1}^{5}\Big\lVert \mathbf b_i^{\,\mathrm{rest}}"
           r"-R_{\pi(i)}\big(\mathbf B(\mathbf p_{\pi(i)}-\boldsymbol\delta)+\mathbf g\big)\Big\rVert^{2},\qquad \lvert g_x\rvert,"
           r"\lvert g_y\rvert,\lvert g_z\rvert\le 100\ \mu\mathrm T$$")
    eq4 = (r"$$\Delta\mathbf b=S(\mathbf u)\,\mathbf f+U\,\Delta\mathbf g+\mathbf b^{\mathrm{rest}}\varepsilon+\mathbf n,\qquad"
           r" S(\mathbf u)=\frac{\partial\mathbf b}{\partial\mathbf q}\,G(\mathbf u),\qquad U=\begin{bmatrix}R_1\\ \vdots\\ R_5\end{bmatrix},"
           r"\qquad \mathbf n\sim\mathcal N(0,\Sigma)$$")
    eq5 = (r"$$N=\big[\,U\ \ \mathbf b^{\mathrm{rest}}\,\big],\quad P=I-N(N^{\top}WN)^{-1}N^{\top}W,\quad"
           r" e(\mathbf u)=\Delta\mathbf b^{\top}WPS\,\big(S^{\top}P^{\top}WPS\big)^{-1}S^{\top}P^{\top}W\,\Delta\mathbf b,\quad"
           r"\hat{\mathbf u}=\arg\max_{\mathbf u}e(\mathbf u)$$")
    eq6 = (r"$$\hat{\mathbf f}=\big(S^{\top}P^{\top}WPS\big)^{-1}S^{\top}P^{\top}W\,\Delta\mathbf b,\qquad"
           r" J=\big[\,PS\ \ \ P\,\partial_yS\,\hat{\mathbf f}\ \ \ P\,\partial_zS\,\hat{\mathbf f}\,\big],\qquad"
           r"\operatorname{cov}(\hat{\mathbf f},\hat{\mathbf u})\approx\big(J^{\top}WJ\big)^{-1}$$")
    eq7 = (r"$$k_i=\frac{E^{*}a_i}{h_i},\qquad \sum_i k_i\,[\delta-g_i]_+=f_{\mathrm{press}},\qquad p_i=\frac{k_i\,[\delta-g_i]_+}{a_i}$$")

    t1 = table(["Assignment of 0x0C 0x11 0x12 0x13 0x10", "Poles (centre third)", "$\\theta_0$ (&#176;)", "$k$", "Rest residual (mT)"],
               [[" ".join(U[u] for u in r["perm"]), " ".join("+" if p > 0 else "&#8722;" for p in r["poles"]),
                 f"{r['theta0']:.1f}", f"{r['k']:.3f}", f"{r['rms_mT']:.3f}"] for r in c["per_assignment"][:6]],
               num_cols=(2, 3, 4), n=1,
               caption=f"The best fit of each assignment of chip addresses to positions, of {c['n_hypotheses']} fitted "
                       f"hypotheses (poles in the FE row's order; a rest residual with $k$ at its bound of 0.1 is a failed fit).")
    keys = ("y_mm", "z_mm", "f_press_N", "f_y_N", "f_z_N")
    t2 = table(["Magnets", "Samples", "$y$ (mm)", "$z$ (mm)", "$f_\\mathrm{press}$ (N)", "$f_y$ (N)", "$f_z$ (N)", "Shown as contact"],
               [[s["magnets"][0].upper() + s["magnets"][1:], s["window"]] + [f"{s['rms'][k]:.2f} ({s['sigma_median'][k]:.2f})" for k in keys]
                + [f"{s['detected_at_threshold']} of {s['n']}"] for s in res["synthetic"]],
               num_cols=(2, 3, 4, 5, 6, 7), n=2,
               caption="RMS error of the estimator on 300 simulated presses, with the median model 1-sigma in brackets, and "
                       "the presses shown as contacts ($e>25$).")

    cmd = ("<pre class=\"cmd\">~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/wrench_viewer.py\n"
           "    # open http://localhost:8790; keys b (new baseline), c (recalibrate at rest), r (record the stream)\n"
           "~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/wrench_viewer.py --replay REC.csv --http 8791\n"
           "~/miniconda3/bin/python scripts/fingertip_contact_estimation_study.py      # the numbers of this page\n"
           "~/miniconda3/envs/femtip/bin/python scripts/fingertip_wrench_fe.py         # the FE basis, 27 s, 2.0 GB</pre>")

    S = []
    S.append(("terms", "Terms and metrics", terms))
    S.append(("model", "Measurement model of the board under the pad",
              f"<p>The board's five MLX90393 chips sit 7&#8202;mm apart in a plus, U1 at the centre and U2 to U5 toward the "
              f"tip end, toward $+y$, toward the mount and toward $-y$, on a sensing plane 0.42&#8202;mm below the pad's back "
              f"face. The five 3&#8202;&#215;&#8202;2&#8202;mm magnets sit at the positions of the FE row {code(fe['case'])} "
              f"of {link(FE_PAGE, 'the v2 fingertip checks')}, {fe['depth_mm']}&#8202;mm below the palmar face. Each magnet "
              f"is a uniformly magnetized cylinder, evaluated by a volume quadrature of 72 point dipoles (Eq. 1): on its "
              f"axis at the chips' distance of 5.37&#8202;mm this gives 19.970&#8202;mT, as the closed form does, where one "
              f"point dipole gives 21.068&#8202;mT.</p>{eq1}"
              f"<p>Here $\\mathbf c_j$ is magnet $j$'s centre, $\\boldsymbol\\rho_q$ and $w_q$ the quadrature's points and "
              f"weights, $s_j=\\pm1$ its pole and $V$ its volume. A chip reports the field in its own axes (Eq. 2). Its $z$ "
              f"axis is the pad's normal up to a sign $s$, and its $x$ axis lies at $\\theta_0+\\phi_c$ from $\\mathbf e_y$, "
              f"where $\\phi_c$ is the part's rotation in the board's pick-and-place file (U1 0&#176;, U2 180&#176;, U3 "
              f"0&#176;, U4 90&#176;, U5 270&#176;) and $\\theta_0$, the orientation of the part's axes in its footprint, is "
              f"common to the five chips. A permutation $\\pi$ assigns the stream's chips (the board answers at 0x0C and "
              f"0x10&#8211;0x13) to positions.</p>{eq2}"))
    S.append(("rest", "Calibration from the reading at rest",
              f"<p>With the pad untouched the reading is the magnets' field plus Earth's field: 15 numbers the model must "
              f"reproduce. The fit (Eq. 3) searches the poles of the four corner magnets relative to the centre one (16 "
              f"patterns), the sign $s$, the assignment $\\pi$ (24) and $\\theta_0$ from four quadrants, and fits the strength "
              f"$k$, the chips' offset $\\boldsymbol\\delta$ from the pad and a uniform field $\\mathbf g$ by least squares. A "
              f"first pass on the normal component and the in-plane magnitude of each chip's field, which do not depend on "
              f"the chips' rotations, ranks the discrete hypotheses for the full fit.</p>{eq3}"
              f"<p>For the pad as found (magnets that are not N52, inserted with random poles) the best fit places "
              f"{assign}, with $\\theta_0={b['theta0']:.1f}$&#176;, two diagonally opposite corner magnets reversed, "
              f"$k={b['k']:.2f}$ and the chips offset by ({b['dx']:+.2f}, {b['dy']:+.2f}, {b['dz']:+.2f})&#8202;mm, at "
              f"{b['rms_mT']:.2f}&#8202;mT RMS against fields of {fmin:.1f}&#8211;{fmax:.1f}&#8202;mT per chip (Table 1, "
              f"Figure 2). The board turned by 180&#176;, which this pole pattern alone cannot distinguish, fits at "
              f"{c['turned_rms_mT']:.2f}&#8202;mT: the pick-and-place rotations do not follow the turn. The uniform field "
              f"ends at its bound, so about 0.4&#8202;mT of the reading is model error. The full search takes about a minute; "
              f"with the board's chip frames kept, the viewer refits poles, strength and offset in 6&#8211;9&#8202;s.</p>"
              f"{t1}"
              + fig(2, MEDIA / "20261020-rest_fit.png",
                    "The reading at rest per chip axis, measured with the pad as found and modelled by the best fit (labels: "
                    "address, fitted position, axis).")))
    S.append(("estimator", "Contact model and estimator",
              f"<p>A contact at face point $\\mathbf u$ with force $\\mathbf f$ moves each magnet rigidly by $G(\\mathbf u)"
              f"\\mathbf f$, taken from the FE of concept b's pad: the gyroid infill's homogenized stiffness, the back fixed, "
              f"$E^*={fe['E_star_MPa']:.1f}$&#8202;MPa from the pressure law and an approach of {fe['approach_1N_um']:.0f}"
              f"&#8202;&#181;m at 1&#8202;N. The basis has {fe['patch_centres']} patch centres on a 1&#8202;mm grid with three "
              f"loads each (Gaussian traction of $\\sigma={fe['sigma_mm']}$&#8202;mm, 1&#8202;N), {fe['load_cases']} load "
              f"cases on one factorization of {fe['n_dof']:,} degrees of freedom (27&#8202;s, 2.0&#8202;GB), and keeps every "
              f"load's displacement at {fe['display_nodes']} face nodes for the viewer. The field model turns the motions "
              f"into the sensitivity $S(\\mathbf u)$, interpolated onto a 0.25&#8202;mm grid (Eq. 4).</p>{eq4}"
              f"<p>A background change $\\Delta\\mathbf g$ reaches every chip through its rotation, and a strength change "
              f"$\\varepsilon$ scales the rest reading. At each grid point the force and the four nuisance values enter "
              f"linearly and follow from weighted least squares ($W=\\Sigma^{{-1}}$, the noise per axis); the contact point "
              f"maximizes the explained chi-square (Eq. 5) and is refined continuously within one grid step, and the Fisher "
              f"information gives its uncertainty (Eq. 6). A contact must press ($f_\\mathrm{{press}}>0$). One estimate "
              f"takes {n52['frame_ms']:.1f}&#8202;ms.</p>{eq5}{eq6}"))
    S.append(("simulated", "Errors on simulated presses",
              f"<p>Each configuration was tested on 300 presses generated by the model itself with the measured noise, so "
              f"Table 2 measures the estimator and the information in the reading, not the model's accuracy against the pad. "
              f"With N52 magnets and 50&#8202;ms of samples the RMS errors are {n52['rms']['y_mm']:.2f}&#8202;mm, "
              f"{n52['rms']['f_press_N']:.3f}&#8202;N pressing and {n52['rms']['f_y_N']:.3f} and {n52['rms']['f_z_N']:.3f}"
              f"&#8202;N in shear; the force errors exceed the model 1-sigma by 1.7&#8211;2.7 times, the position errors "
              f"by 1.0&#8211;1.9 times. Over the face the model 1-sigma for a 1&#8202;N press has a median of "
              f"{mp['as found']['f_press_N'][1]:.3f}&#8202;N and {mp['as found']['y_mm'][1]:.2f}&#8202;mm with the magnets as "
              f"found and {mp['N52, poles one way']['f_press_N'][1]:.3f}&#8202;N and {mp['N52, poles one way']['y_mm'][1]:.2f}"
              f"&#8202;mm with N52 (Figure 3); the outer rows across the pad, beyond the chips, are 2&#8211;3 times "
              f"worse.</p>{t2}"
              + fig(3, MEDIA / "20261020-uncertainty_maps.png",
                    "Model 1-sigma of the pressing force and of the position for a 1&#8202;N press over the face, with "
                    "50&#8202;ms of samples. White squares: chips; red circles: magnets.")))
    S.append(("quiet", "Behaviour without a press",
              f"<p>Over 30&#8202;s of the board with the pad untouched ({q['frames']} frames of 50&#8202;ms), the residual "
              f"after the nuisance terms has a median chi-square of {q['chi2_median']:.1f} against {q['dof']} degrees of "
              f"freedom, so the noise model holds to about 10&#8202;%. The best spurious contact explains a median "
              f"chi-square of {q['e_median']:.1f} and a 99th percentile of {q['e_p99']:.1f}; {q['false_contacts']} frame of "
              f"{q['frames']} exceeds the threshold of {q['threshold']:.0f}. A step of about 5&#8202;&#181;T in the reading "
              f"at {q['step_s']:.0f}&#8202;s, the finger turning slightly, appears as a background change along $z$ from "
              f"{q['g_before'][2]:.1f} to {q['g_after'][2]:.1f}&#8202;&#181;T and not as a contact (Figure 4).</p>"
              + fig(4, MEDIA / "20261020-no_press.png",
                    "The estimator over 30&#8202;s without a press: the explained chi-square of the best contact against the "
                    "threshold (top) and the background change it assigns (bottom).")))
    S.append(("viewer", "Live viewer in the browser",
              f"<p>{code('src/morphohand/driver/reskin5x/host/wrench_viewer.py')} reads the board, calibrates at rest and "
              f"serves the page at http://localhost:8790. The page draws the face deformed by the FE basis at the estimated "
              f"force (the displacement scaled for display), coloured by the contact pressure of Winkler pads on the face "
              f"nodes under a sphere pressed at the estimated point (Eq. 7, the stiffness rule of the pad builder v2 with "
              f"$h_i$ the node's depth below the FE nodes' centroid). It also draws the force arrow, the 1-sigma position "
              f"ellipse, the magnets in their fitted poles and the chips with their fitted addresses. Figure 1 replays two "
              f"synthetic presses through the server: 2&#8202;N at (1, 12)&#8202;mm returns 2.03&#8202;N at (1.00, 12.12)"
              f"&#8202;mm, and (1.5, 0.4, &#8722;0.3)&#8202;N at (&#8722;2, 7)&#8202;mm returns (1.58, 0.39, &#8722;0.37)"
              f"&#8202;N at (&#8722;1.32, 6.72)&#8202;mm, 0.68&#8202;mm off across against a 1-sigma of 0.61&#8202;mm.</p>{eq7}"
              + fig(5, MEDIA / "20261020-viewer_board_live.png",
                    "The viewer on the board at 19:20 on 2026-10-20, with the magnets as found.")
              + cmd))
    S.append(("limits", "Limitations",
              "<p>No estimate has been compared with a reference force or position; the errors above are the model against "
              "itself. The rest residual of 0.43&#8202;mT, about 10&#8202;% of the field, shows how well the magnets' geometry "
              "is known, and the sensitivity can be wrong by a similar or larger fraction. The FE is linear elastic with the "
              "gyroid's homogenized stiffness, without the stiffening of large presses or the infill's local buckling. The "
              "estimator assumes one contact; a press spread over the pad is reported near its centre of load, and the outer "
              "millimetre of the face lies outside the FE grid. The background term covers uniform fields only, so a steel "
              "tool near the pad reads as a contact. The pressure patch assumes a sphere of a chosen radius on Winkler pads, "
              "and the deformation drawn is the FE's response to a Gaussian patch, not a contact solution. The magnets in the "
              "pad today are not N52 and their poles are mixed; the N52 rows assume the same geometry with $k=1$ and all "
              "poles one way.</p>"))
    S.append(("future", "Future work",
              "<ul>"
              "<li>Recalibrate with the N52 magnets (key c in the viewer, or "
              "<code>fingertip_wrench_model.calibrate</code> on a <code>reskin5x_check.py --out</code> capture). The fit should "
              "give $k$ near 1, all poles one way and a rest residual below 0.43&#8202;mT; a $k$ well below 1 or a larger "
              "residual means the magnets' depth or the pad's seat on the board differs from the FE row.</li>"
              "<li>Ground truth without the UR5: masses of 50&#8211;500&#8202;g on a printed 3&#8202;mm stylus at marked points "
              "of a printed grid jig give the pressing force and position. Errors beyond 2&#8211;3 model sigmas falsify the "
              "model at that point.</li>"
              "<li>The UR5 session with the HEX-E sensor: a grid of points, forces and shear directions; refit $k$, "
              "$\\boldsymbol\\delta$, $\\theta_0$, $E^*$ (which scales $S$ by $1/E^*$) and per-magnet strengths on part of it, "
              "and compare the held-out errors with the model 1-sigma. A learned residual correction only if these physical "
              "parameters leave a structured error.</li>"
              "<li>Large presses: the FE row's rod responses at 0.5, 1 and 3&#8202;N give the stiffening as a force-magnitude "
              "correction, to be checked against the masses.</li>"
              "<li>Temperature: warm the pad at rest (a hand around it) and check that $\\varepsilon$ takes up the drift "
              "without a contact.</li></ul>"
              f"<p>Status of the printed-fingertip work: {link(STATUS_PAGE, 'status page')}.</p>"))

    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "Model-based contact estimation",
                 "H1": "Model-based estimation of contact position and force on the magnetic fingertip",
                 "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-20", "LEDE": lede,
                 "BYLINE": f"Built {time.strftime('%Y-%m-%d %H:%M')} from results.json ({res['when']}); the field readings "
                           f"are measured, the presses simulated.",
                 "FILM": film(1, MEDIA / "20261020-viewer_synthetic_presses.mp4", MEDIA / "20261020-viewer_synthetic_presses.png",
                              "The browser viewer replaying two synthetic presses made by the model with the measured noise "
                              "(Section 7): the estimated force arrow, the 1-sigma ellipse, the FE deformation drawn 20 times "
                              "and the pad pressure on a 300&#8202;kPa scale."),
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    build()
