#!/usr/bin/env python3
"""Result page of the 2026-10-21 comparison of spherical and disc N52 magnets in pad B00 on the ReSkin 5X board.

Owner, 2026-10-21: "my coworker suggested using spherical n52 magnets ... does it make magnetic field sensing easier?";
then "that doesnt seem right that the field would be too strong"; then "i ordered some spherical magnets and we can test
this out when time comes. just make sure the research is recorded somewhere to refer to".

Reads docs/experiments/20261021-spherical_magnets/20261021-spherical_magnet_check.json (scripts/spherical_magnet_check.py),
draws the two figures into media/, renders the mathematics with scripts/texsvg.py, tags the page for scripts/docs_index.py,
applies scripts/retro_style.py and writes docs/experiments/20261021-spherical_magnets/20261021-spherical_and_disc_magnets.html.

    .venv/bin/python scripts/spherical_magnet_check.py && .venv/bin/python scripts/spherical_magnets_page.py
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = ROOT / "docs/experiments/20261021-spherical_magnets"
DATA = D / "20261021-spherical_magnet_check.json"
PAGE = D / "20261021-spherical_and_disc_magnets.html"
TEMPLATE = ROOT / "scripts/spherical_magnets_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
EST_PAGE = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html"
V2_PAGE = ROOT / "docs/experiments/20261019-v2_tip_fe_checks/20261019-v2_fingertip_stiffness_checks.html"
MLX_PAGE = ROOT / "docs/experiments/20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html"
B3_PAGE = ROOT / "docs/experiments/20261020-ur5_calibration_mount_v2/20261020-dip_link_b_v3_and_ur5_mount_v2.html"
STATUS_PAGE = ROOT / "docs/experiments/20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html"
OVERVIEW = ROOT / "docs/overviews/printed_fingertips.html"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # reference blue ramp, steps 250-650 (ordinal on white)
NBSP = "&#8202;"


def esc(s):
    return html.escape(str(s), quote=True)


def relp(p):
    return os.path.relpath(p, D)


def img_uri(png, width=1100):
    from PIL import Image
    im = Image.open(png).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def table(head, body, num_cols=(), caption=None, n=None):
    cap = f"<caption>Table {n}. {caption}</caption>" if caption else ""
    th = "".join(f'<th{" class=n" if k in num_cols else ""}>{h}</th>' for k, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td{" class=n" if k in num_cols else ""}>{c}</td>' for k, c in enumerate(r)) + "</tr>"
                 for r in body)
    return f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def figure(png, n, caption, width=1100):
    return (f'<figure id="fig{n}"><img src="{img_uri(png, width)}" alt="Figure {n}" style="max-width:100%">'
            f'<figcaption>Figure {n}. {caption}</figcaption></figure>')


def mm(x, nd=2):
    return f"{x:.{nd}f}{NBSP}mm"


def options(J):
    """The rows of Table 4 and Figure 2: (layout, magnet, record) with the record's force resolution at the contact point."""
    S = {r["case"]: r for r in J["sensing"]}
    f5, f4 = S["bm_five_d75"], S["bm_four_d75"]
    return [
        ("Five", "3 x 2 mm disc (pad B00)", f5["cylinder_3x2_exact"]),
        ("Five", "3 mm sphere at the discs' centres", f5["sphere_3"]),
        ("Five", "3 mm sphere, top at the discs' tops", J["sphere_3_top_at_cylinder_top"]),
        ("Five", "2.5 mm sphere", f5["sphere_2.5"]),
        ("Five", "2 mm sphere", f5["sphere_2"]),
        ("Five", "4 mm sphere", f5["sphere_4"]),
        ("Five", "Any magnet at the range limit", f5["range_limited"]),
        ("Four", "3 x 2 mm disc", f4["cylinder_3x2_exact"]),
        ("Four", "3 mm sphere", f4["sphere_3"]),
        ("Four", "4 mm sphere", f4["sphere_4"]),
        ("Four", "Any magnet at the range limit", f4["range_limited"]),
    ]


def draw_axis(J, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    R = J["range"]
    plt.rcParams.update({"font.size": 15, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=110)
    z = np.linspace(1.0, 14.5, 600)
    pole = R["pole_field_T"] * 1e3
    xl = 14.8                                                     # label column, right of the curves
    for k, Dm in enumerate((2.0, 3.0, 4.0, 5.0, 6.0)):
        a = Dm / 2
        zz = z[z >= a]
        ax.plot(zz, pole * (a / zz) ** 3, color=RAMP[k], lw=2)
        lab = "3 mm sphere and\n3 x 2 mm disc (dashed)" if Dm == 3.0 else f"{Dm:g} mm sphere"
        ax.text(xl, pole * (a / 14.5) ** 3, lab, fontsize=13, va="center", ha="left")
    Br, Rr, L = 1.45, 1.5, 2.0
    zz = z[z >= 1.05]
    f = lambda u: u / np.sqrt(u * u + Rr * Rr)  # noqa: E731
    ax.plot(zz, 1e3 * Br / 2 * (f(zz + L / 2) - f(zz - L / 2)), color="#000", lw=2, ls="--")
    for y, lab, dy in ((R["saturation_onset_mT"], "Saturation onset, 50 mT", 1.25),
                       (R["usable_mT"], f"Usable, {R['usable_mT']:.1f} mT", 0.8)):
        ax.axhline(y, color="#777", lw=1.2, ls=":")
        ax.text(xl, y * dy, lab, fontsize=13, va="center", color="#333")
    for x, lab in ((R["U1_axial_mm"], "U1 under the centre magnet\n(pad B00, depth 7.5 mm)"), (R["chip_to_apex_mm"], "Pad\napex")):
        ax.axvline(x, color="#999", lw=1.0)
        ax.text(x + 0.15, 2.2e3, lab, fontsize=13, va="top")
    ax.set_yscale("log")
    ax.set_ylim(0.2, 3e3)
    ax.set_xlim(1.0, 19.5)
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
    ax.set_xlabel("Distance from the magnet's centre along its axis (mm)")
    ax.set_ylabel("Axial field, N52 (mT)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="major", color="#e6e6e6", lw=0.8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_options(J, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = options(J)
    plt.rcParams.update({"font.size": 15, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=110)
    ys = list(range(len(rows)))[::-1]
    usable = J["range"]["usable_mT"]
    for y, (lay, mag, r) in zip(ys, rows):
        v = r["centre_force_N"]
        over = r["B_max_component_mT"] > usable
        limit = mag.startswith("Any")
        if over:
            ax.text(0.001, y, f"{r['B_max_component_mT']:.0f} mT on a chip, over the range", va="center", fontsize=13, color="#333")
        else:
            ax.barh(y, v, height=0.62, color="#ffffff" if limit else (RAMP[3] if "B00" in mag else RAMP[1]),
                    edgecolor=RAMP[3], hatch="//" if limit else None, lw=1.0)
            ax.text(v + 0.001, y, f"{v:.3f} N", va="center", fontsize=13)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{lay}: {mag}" for lay, mag, _ in rows], fontsize=13)
    ax.set_xlabel("Force resolution at the contact point (N)")
    ax.set_xlim(0, 0.105)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="x", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def build():
    J = json.loads(DATA.read_text())
    R = J["range"]
    S = {r["case"]: r for r in J["sensing"]}
    f5, f4 = S["bm_five_d75"], S["bm_four_d75"]
    me = {r["case"]: r for r in J["model_error"]}
    tq = J["neighbour_torque"]["3"]
    disc5, sph5, top5 = f5["cylinder_3x2_exact"], f5["sphere_3"], J["sphere_3_top_at_cylinder_top"]
    disc4, sph44 = f4["cylinder_3x2_exact"], f4["sphere_4"]
    media = D / "media"
    media.mkdir(parents=True, exist_ok=True)
    draw_axis(J, media / "20261021-axial_field_vs_distance.png")
    draw_options(J, media / "20261021-force_resolution_options.png")
    Bs = [m["B_mT"][0] for m in tq["magnets"]]
    Ts = [m["torque_N_mm_per_rad"] for m in tq["magnets"]]
    press = R["press_fraction_pct_per_N"]

    lede = (f"A spherical N52 magnet resolves contact force about as well as the 3{NBSP}x{NBSP}2{NBSP}mm disc of the same volume. "
            f"On pad B00's five-magnet layout a 3{NBSP}mm sphere at the discs' centres gives {sph5['centre_force_N']:.4f}{NBSP}N against "
            f"the disc's {disc5['centre_force_N']:.4f}{NBSP}N, and on the four-magnet layout {f4['sphere_3']['centre_force_N']:.4f} against "
            f"{disc4['centre_force_N']:.4f}{NBSP}N. Spheres of 4{NBSP}mm and more put {f5['sphere_4']['B_max_component_mT']:.0f}&#8211;"
            f"{f5['sphere_5']['B_max_component_mT']:.0f}{NBSP}mT on a chip at the five-magnet centres, where the MLX90393 reads "
            f"{R['usable_mT']:.1f}{NBSP}mT with margin, because an N52 sphere's pole field is {R['pole_field_T']:.2f}{NBSP}T at every size. "
            f"Four 4{NBSP}mm spheres on the four-magnet layout read {sph44['B_max_component_mT']:.1f}{NBSP}mT and resolve "
            f"{sph44['centre_force_N']:.3f}{NBSP}N, the one sphere option better than pad B00, if their pocket walls pass. "
            f"The owner ordered spheres on 2026-10-21 to test this on the board.")
    summary = (f"A 3 mm N52 sphere resolves {sph5['centre_force_N']:.4f} N on pad B00 against {disc5['centre_force_N']:.4f} N for the "
               f"3 x 2 mm disc of equal volume; spheres of 4 mm and more exceed the MLX90393's range at the five-magnet centres, and four "
               f"4 mm spheres on the four-magnet layout resolve {sph44['centre_force_N']:.3f} N if their walls pass.")

    Sx = []
    Sx.append(("terms", "Terms and metrics", f"""
<dl class="terms">
<dt>Force resolution</dt><dd>Cram&#233;r&#8211;Rao lower bound on the standard deviation of the normal force estimated from one sample
of the five chips (N), for a 1{NBSP}N normal press at a point of the pad's face, with the shear forces, the contact position along and
across the finger and a uniform background field estimated alongside (<code>tpu_tip_magnet_sensing.analyse</code>). Chip noise is the
datasheet's at the gain the field needs (1.2&#8211;1.9{NBSP}&#181;T per axis); the board measured 1.4 times that on 2026-10-20, which
scales every value here by 1.4. Contact point: the centre of the face, over U1.</dd>
<dt>Largest field component</dt><dd>The largest of the 15 rest readings (five chips, three axes), mT; it chooses the gain.</dd>
<dt>Usable field</dt><dd>{R['usable_mT']:.1f}{NBSP}mT: the range of the lowest gain (GAIN_SEL 0, RES 1, {R['lowest_gain_range_mT']:.1f}{NBSP}mT)
divided by the 1.25 margin of <code>v2_tip_fe_checks.sense</code>. Saturation onset: {R['saturation_onset_mT']:.0f}{NBSP}mT, where the
chip's integrated magnetic concentrator starts to saturate (MLX90393 datasheet, ABA-0xx parts).</dd>
<dt>Moment</dt><dd>$m = B_r V / \\mu_0$, with remanence $B_r = 1.45$ T for N52 and $V$ the magnet's volume; equal volumes give equal moments.</dd>
<dt>Depth</dt><dd>Distance from the pad's apex to the magnet's top face, mm (7.5{NBSP}mm in pad B00).</dd>
<dt>Dipole error</dt><dd>Relative difference between the point-dipole field of a magnet's moment and its exact field, %.</dd>
<dt>Press fraction</dt><dd>Change of a chip's reading under a 1{NBSP}N press at the contact point, as a percentage of its rest reading.</dd>
<dt>U1&#8211;U5</dt><dd>The board's five MLX90393 chips: U1 at the centre, U2&#8211;U5 7{NBSP}mm from it in a plus.</dd>
</dl>"""))

    dd = {(r["t_mm"], r["z_mm"]): r["dipole_err_pct"] for r in R["disc_dipole_error"]}
    zs = sorted({r["z_mm"] for r in R["disc_dipole_error"]})
    tab_disc = table(["Disc (mm)", "Length/diameter"] + [f"{z:g}{NBSP}mm" for z in zs],
                     [[f"3 x {t:g}", f"{t / 3:.2f}"] + [f"{dd[(t, z)]:+.1f}" for z in zs] for t in (1.0, 2.0, 2.6, 3.0)],
                     num_cols=range(1, 2 + len(zs)), n=1,
                     caption="Dipole error (%) of 3 mm discs on their axis, by distance from the centre. A sphere's is zero at every distance.")
    Sx.append(("field", "Field of a sphere and of a disc", f"""
<p>Outside a uniformly magnetized sphere the field is that of a point dipole at its centre, at every distance (Petruska and Abbott,
<i>IEEE Trans. Magn.</i> 49(2):811, 2013):</p>
$$\\mathbf{{B}}(\\mathbf{{r}}) = \\frac{{\\mu_0}}{{4\\pi r^3}}\\left(3\\hat{{\\mathbf{{r}}}}\\,(\\hat{{\\mathbf{{r}}}}\\cdot\\mathbf{{m}}) - \\mathbf{{m}}\\right),
\\qquad B_{{\\mathrm{{axis}}}}(z) = \\tfrac{{2}}{{3}}\\,B_r\\left(\\frac{{a}}{{z}}\\right)^{{3}}$$
<p>with $a$ the sphere's radius and $z$ the distance from its centre. A disc of radius $R$ and length $L$ magnetized along its axis has
the axial field</p>
$$B_{{\\mathrm{{axis}}}}(z) = \\frac{{B_r}}{{2}}\\left[\\frac{{z + L/2}}{{\\sqrt{{(z+L/2)^2 + R^2}}}} - \\frac{{z - L/2}}{{\\sqrt{{(z-L/2)^2 + R^2}}}}\\right],$$
<p>which approaches the dipole value with distance; discs with length/diameter near 0.87 come closest. At U1, {mm(R['U1_axial_mm'])}
from the centre magnet of pad B00, the 3{NBSP}x{NBSP}2{NBSP}mm disc's axial field is {dd[(2.0, R['U1_axial_mm'])]:.1f}{NBSP}% under its
dipole value (Table 1). Over the five chips its rest field differs from the dipole field by
{min(me['bm_five_d75']['dipole']['rest_err_pct_per_chip']):.1f}&#8211;{max(me['bm_five_d75']['dipole']['rest_err_pct_per_chip']):.1f}{NBSP}%,
and its change under a 1{NBSP}N press by {me['bm_five_d75']['dipole']['press_err_pct']:.0f}{NBSP}% at 7.5{NBSP}mm depth and
{me['bm_five_d85']['dipole']['press_err_pct']:.0f}{NBSP}% at 8.5{NBSP}mm. The exact disc field here is a volume quadrature of point
dipoles at 16{NBSP}x{NBSP}64{NBSP}x{NBSP}16 points. It equals the analytic axial field to the printed digits and the 24{NBSP}x{NBSP}96{NBSP}x{NBSP}24
quadrature within {me['bm_five_d75']['check']['rest_err_uT_max']:.2f}{NBSP}&#181;T. The contact estimator
(<code>fingertip_wrench_model.Model</code>, 72 points per disc) agrees with it to
{max(me['bm_five_d75']['quad72']['press_err_pct'], me['bm_five_d85']['quad72']['press_err_pct']):.3f}{NBSP}%, so the sphere's exact
dipole gives the estimator nothing it lacks.</p>
{tab_disc}
<p>A literature search on 2026-10-21 found no magnet-in-elastomer tactile sensor built on spherical magnets; the open-source MLX90393
fingertips and the tri-axis design studies found use discs.</p>"""))

    sizes = R["sphere_axis"]
    tab_size = table(["Sphere (mm)", "Volume (mm&#179;)", "Moment / 3 x 2 mm disc", f"Axial field at {R['U1_axial_mm']:.2f}{NBSP}mm (mT)",
                      f"Distance for {R['usable_mT']:.1f}{NBSP}mT (mm)", "Distance for 50 mT (mm)", "Fits the five-magnet spacing"],
                     [[f"{r['D_mm']:g}", f"{r['volume_mm3']:.1f}", f"{r['volume_mm3'] / 14.14:.2f}", f"{r['field_at_U1_mT']:.1f}",
                       f"{r['distance_for_usable_mm']:.2f}", f"{r['distance_for_50mT_mm']:.2f}",
                       "Yes" if 5.0 - r["D_mm"] >= 1.8 else "No"] for r in sizes],
                     num_cols=range(0, 6), n=2,
                     caption=f"N52 spheres on their axis: the field at U1's distance in pad B00 and the distance each needs from a chip. "
                             f"The pad's apex is {R['chip_to_apex_mm']:.2f} mm from the chips.")
    Sx.append(("range", "Chip range and magnet size", f"""
<p>An N52 sphere's field at its pole is $\\tfrac{{2}}{{3}}B_r = {R['pole_field_T']:.2f}$ T whatever its size, and on its axis the field
falls as $(a/z)^3$, so a larger sphere at a fixed distance gives the field of a smaller sphere moved closer. The chip reads the magnets'
rest field, and a press changes that field by little: {min(press):.2f}&#8211;{max(press):.2f}{NBSP}% per newton on the five chips of
pad B00 ({press[0]:.2f}{NBSP}% on U1). The rest field must therefore stay inside the chip's range. At U1's {mm(R['U1_axial_mm'])} a
3{NBSP}mm sphere gives {sizes[2]['field_at_U1_mT']:.1f}{NBSP}mT on its axis and a 4{NBSP}mm sphere {sizes[3]['field_at_U1_mT']:.1f}{NBSP}mT
alone. The four neighbours add the rest of the {f5['sphere_4']['B_max_component_mT']:.0f}{NBSP}mT of Table 3 (Table 2, Figure 1).</p>
<p>A larger magnet kept inside the range has to sit farther from the chips, where a given travel changes the reading by a smaller
fraction (about $3\\,\\delta z / z$ for an axial step $\\delta z$). Magnets nearer the face also move more per newton (7.5{NBSP}&#181;m/N at
6.5{NBSP}mm depth, 5.2 at 7.5{NBSP}mm), so the two effects partly cancel. With this chip a design gains from magnet size only until the
largest component reaches the usable field. Pad B00's exact discs use {R['five_disc_exact_max_mT']:.1f} of {R['usable_mT']:.1f}{NBSP}mT, and a
magnet with the moment that fills it at the same centres would resolve {f5['range_limited']['centre_force_N']:.3f}{NBSP}N.</p>
{figure(media / '20261021-axial_field_vs_distance.png', 1, f"Axial field of N52 spheres of 2&#8211;6{NBSP}mm and of the 3{NBSP}x{NBSP}2{NBSP}mm disc against the distance from the magnet's centre, with the chip's saturation onset and usable field. The 3{NBSP}mm sphere and the disc have equal volumes; on the axis the disc's field is {dd[(2.0, R['U1_axial_mm'])]:.1f}{NBSP}% under the sphere's at U1 and {dd[(2.0, 10.0)]:.1f}{NBSP}% under at 10{NBSP}mm.")}
{tab_size}"""))

    opts = options(J)
    tab_opt = table(["Layout", "Magnet", "Largest field component (mT)", "Gain (GAIN_SEL, RES)", "Force resolution at the contact point (N)",
                     "Median over the face (N)", "Position across / along (mm)"],
                    [[lay, mag, f"{r['B_max_component_mT']:.1f}", "&#8211;" if r["gain_sel"] is None else f"{r['gain_sel']}, {r['res']}",
                      f"{r['centre_force_N']:.4f}" + (" (over the range)" if r["gain_sel"] is None else ""),
                      f"{r['median_force_N']:.4f}", f"{r['centre_sig_y_mm']:.2f} / {r['centre_sig_z_mm']:.2f}"] for lay, mag, r in opts],
                    num_cols=(2, 4, 5), n=3,
                    caption="Magnet options in pad B00 at 7.5 mm depth (five-magnet layout: centre plus corners at y &#177;2.5, z 5.46 and 15.26 mm; four-magnet layout: y &#177;3, z 7.36 and 14.36 mm). Discs are modelled exactly, spheres as dipoles; the range-limit rows scale the moment until the largest component is the usable field.")
    Sx.append(("options", "Spheres in pad B00", f"""
<p>Each sphere takes the place of a disc at the disc's centre and moves with the disc's rigid motion from the finite-element rows of
<code>docs/experiments/20261019-v2_tip_fe_checks/sense_fe.jsonl</code> (cases <code>bm_five_d65</code>&#8211;<code>d85</code>,
<code>bm_four_d70</code>, <code>d75</code>). Table 3 and Figure 2 give the options. The 3{NBSP}mm sphere improves the five-magnet pad by
{100 * (1 - sph5['centre_force_N'] / disc5['centre_force_N']):.0f}{NBSP}%; it is 1{NBSP}mm taller than the disc. With its top kept at the
disc's top (centre 0.5{NBSP}mm deeper, motions halfway to the 8.5{NBSP}mm row) it puts {top5['B_max_component_mT']:.1f}{NBSP}mT on a chip and
resolves {top5['centre_force_N']:.3f}{NBSP}N. Spheres of 2 and 2.5{NBSP}mm lose by the ratio of their moments. The five-magnet layout's corner
pairs are 5.0{NBSP}mm apart, which admits spheres of at most 3.2{NBSP}mm with 1.8{NBSP}mm walls.</p>
<p>The four-magnet layout puts its magnets between the chips and reads {disc4['B_max_component_mT']:.1f}{NBSP}mT with discs, so 4{NBSP}mm
spheres fit its range: {sph44['B_max_component_mT']:.1f}{NBSP}mT and {sph44['centre_force_N']:.4f}{NBSP}N, against
{disc5['centre_force_N']:.4f}{NBSP}N for pad B00 now. A 4{NBSP}mm sphere is 2{NBSP}mm taller than the disc: at the disc's centre its top comes
1{NBSP}mm nearer the palmar face, where the four-magnet pad left 1.76{NBSP}mm of wall at 7.0{NBSP}mm depth, and with its top at the disc's
top the largest component is {R['four_sphere4_top_at_disc_top_max_mT']:.1f}{NBSP}mT and its bottom comes 2{NBSP}mm nearer the rail grooves.
A disc of the same moment, {R['sphere4_equivalent_discs_mm'][0][0]:g}{NBSP}x{NBSP}{R['sphere4_equivalent_discs_mm'][0][1]:.2f}{NBSP}mm, gives
nearly the same field in a height of {R['sphere4_equivalent_discs_mm'][0][1]:.1f}{NBSP}mm.</p>
{figure(media / '20261021-force_resolution_options.png', 2, "Force resolution at the contact point for the options of Table 3. Hatched bars: the moment that fills the usable field at the same centres, any shape. Rows without a bar put more than the usable field on a chip.")}
{tab_opt}"""))

    Sx.append(("fabrication", "Assembly of spheres in printed slots", f"""
<p>A disc's slot fixes its axis, so a disc can only go in with its pole reversed, which the calibration finds (two corners of the
first pad of 2026-10-20). A sphere's axis is not visible and nothing in the slot fixes it. With the five moments parallel, each magnet
sits in its neighbours' field, {min(abs(b) for b in Bs):.1f}&#8211;{max(abs(b) for b in Bs):.1f}{NBSP}mT against its own moment at the
corners and the centre. This field turns a loose 3{NBSP}mm sphere away from the parallel arrangement with
{min(Ts):.2f}&#8211;{max(Ts):.2f}{NBSP}N{NBSP}mm per radian of tilt. The print pause leaves air above each magnet in its slot, so a sphere
needs a seat that grips it, or an aligning field while the layers above it print. The calibration then fits each magnet's
direction (two angles) in addition to its pole.</p>"""))

    Sx.append(("dipole_bound", "Point dipoles in the earlier force resolutions", f"""
<p><code>tpu_tip_magnet_sensing.analyse</code>, which gives the force resolutions of the earlier fingertip pages, models every magnet as
a point dipole. For the 3{NBSP}x{NBSP}2{NBSP}mm discs of pad B00 it therefore gives the 3{NBSP}mm sphere's value,
{sph5['centre_force_N']:.4f}{NBSP}N, where the exact disc gives {disc5['centre_force_N']:.4f}{NBSP}N. The largest component is
{f5['cylinder_3x2']['B_max_component_mT']:.2f} against {disc5['B_max_component_mT']:.2f}{NBSP}mT. On the four-magnet layout the two
agree ({f4['cylinder_3x2']['centre_force_N']:.4f} and {disc4['centre_force_N']:.4f}{NBSP}N). <code>spherical_magnet_check.sense_cylinder_exact</code>
gives the exact value for any row of <code>sense_fe.jsonl</code>.</p>"""))

    Sx.append(("limits", "Limitations", """
<p>The spheres move with the discs' finite-element motions; no finite-element case has a spherical inclusion. The pocket-wall and
rail-groove checks were not run on sphere slots. All values are model values at the datasheet noise and nominal N52 remanence;
nothing here was measured.</p>"""))

    Sx.append(("future", "Future work", f"""
<ul>
<li>Sphere slots: add a sphere seat to <code>metric_magnet_prints.upright_slots</code> and run its wall check (<code>walls_pass</code>,
1.8{NBSP}mm) and <code>v3b_pad_b00_magnets_xl.r4</code> (2.0{NBSP}mm to the rail grooves and screw notches) for four 4{NBSP}mm spheres
at depths 7.5&#8211;8.5{NBSP}mm and for five 3{NBSP}mm spheres with their tops at the discs' tops. No passing depth on the four-magnet
layout rules the 4{NBSP}mm option out.</li>
<li>Finite elements with spherical inclusions: a case in <code>v2_tip_fe_checks.CASES</code> with sphere magnets, run with
<code>--sense</code>, then <code>spherical_magnet_check.py</code> on its row. Sphere travel more than 10{NBSP}% from the disc's at the same
centre would change Table 3.</li>
<li>Board test when the ordered spheres arrive: a pad with sphere slots on the board, its rest reading fitted by
<code>scripts/reskin5x_pad_fit.py</code> with each magnet's direction free (the discs fitted to 0.38{NBSP}mT RMS); then the same presses
for a disc pad and a sphere pad under <code>src/morphohand/driver/reskin5x/host/pad_shift_check.py</code> and a kitchen-scale reference.
The model predicts a force resolution {100 * (1 - sph5['centre_force_N'] / disc5['centre_force_N']):.0f}{NBSP}% finer for
3{NBSP}mm spheres than for discs at the same centres. A sphere pad that resolves worse than the disc pad, or a direction scatter over
5{NBSP}degrees in the fit, argues against spheres in printed pads.</li>
<li>A 4{NBSP}x{NBSP}{R['sphere4_equivalent_discs_mm'][0][1]:.1f}{NBSP}mm disc on the four-magnet layout, through the same wall checks: the same
moment as a 4{NBSP}mm sphere at the disc's height.</li>
<li>Recompute the force resolutions quoted on the <a href="{relp(V2_PAGE)}">v2 fingertip checks</a>, the
<a href="{relp(B3_PAGE)}">DIP-link b v3 page</a>, the status page's DIP table and the <a href="{relp(OVERVIEW)}">fingertip overview</a>
with <code>sense_cylinder_exact</code>.</li>
</ul>"""))

    Sx.append(("files", "Files", f"""
<p>Folder <code>docs/experiments/20261021-spherical_magnets/</code>: <code>20261021-spherical_magnet_check.json</code>
(scripts/spherical_magnet_check.py), <code>media/</code> (the figures) and this page (scripts/spherical_magnets_page.py). Related pages:
<a href="{relp(EST_PAGE)}">model-based contact estimation</a>, <a href="{relp(MLX_PAGE)}">the MLX90393 fingertip readout</a>,
<a href="{relp(STATUS_PAGE)}">the status of the printed-fingertip work</a>.</p>"""))

    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(Sx, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(Sx, 1))
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "Spherical fingertip magnets", "H1": "Spherical and disc N52 magnets in the B00 fingertip pad",
                 "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-21",
                 "LEDE": lede, "BYLINE": f"Built {time.strftime('%Y-%m-%d %H:%M')} from the study's JSON; model values, nothing measured.",
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None) for m in TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = TEX_RE.sub(lambda m: next(svgs), page)
    page = docs_index.tag(page, topic="fingertip", summary=summary, date="2026-10-21")
    page = retro_style.apply(page)
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    build()
