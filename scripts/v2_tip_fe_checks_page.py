#!/usr/bin/env python3
"""Result page of the 2026-10-19 night's checks of the v2 fingertips: finite elements of concept a's tip and concept b's
pad against the calibrated 11/22 print, the grid check, under-extrusion, the G-code audit of the 85A plates and the
pads-against-Drake control on the block.

Reads docs/experiments/20261019-v2_tip_fe_checks/ (fe.jsonl, meta.jsonl, cells.jsonl from scripts/v2_tip_fe_checks.py;
print_audit.json from scripts/tpu_print_audit.py), the calibration study's fe/ rows and its iCEM replays
(scripts/fe_contact_calibration_summary.d1), draws the figure into media/, renders the mathematics with scripts/texsvg.py,
applies scripts/retro_style.py and writes docs/experiments/20261019-v2_tip_fe_checks/20261019-v2_fingertip_stiffness_checks.html.

    .venv/bin/python scripts/v2_tip_fe_checks_page.py
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = ROOT / "docs/experiments/20261019-v2_tip_fe_checks"
CAL = ROOT / "docs/experiments/20261018-fe_contact_calibration"
PAGE = D / "20261019-v2_fingertip_stiffness_checks.html"
TEMPLATE = ROOT / "scripts/v2_tip_fe_checks_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
CAL_PAGE = CAL / "20261018-fe_contact_calibration.html"
DIP_PAGE = ROOT / "docs/experiments/20261018-dip_link_board_mount/20261018-dip_link_magnetometer_board_mount.html"
PROC_PAGE = ROOT / "docs/experiments/20261018-fingertip_print_procedure/20261018-siraya85a_fingertip_print_procedure.html"
STATUS_PAGE = ROOT / "docs/experiments/20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html"
ROD = "cx12.5_z0"
SC_CELLS = ["cylinder:25:0.125", "cylinder:16:0.625", "cylinder:10:0.75", "cylinder:16:-0.125"]
VOX_MM3 = 0.05 ** 3
CORE0 = 0.326                     # solid fraction of the 30 % slicer gyroid cell (h 0.05 mm), the calibration's core
LABEL = {
    "cal4_base": "Calibrated print of 10-18: four magnets 7.0&#8202;mm deep",
    "v2_5": "Five magnets 7.5&#8202;mm deep (v2 layout)",
    "a_v2": "Concept a's tip: v2 magnets and the two key channels",
    "b_v2": "Concept b's pad: v2 magnets, cut 3.55&#8202;mm short at the back",
    "v2_none": "No magnets",
    "cal4_fine": "Calibrated print on the fine grid",
    "v2_5_skin07": "Skins 0.7&#8202;mm in place of 0.8&#8202;mm",
    "v2_5_uex05": "5&#8202;% under-extrusion: beads 0.4275&#8202;mm, perimeters 0.855&#8202;mm",
    "v2_5_uex10": "10&#8202;% under-extrusion: beads 0.405&#8202;mm, perimeters 0.81&#8202;mm",
    "v2_5_gyr10": "Gyroid beads alone 10&#8202;% narrower",
    "v2_5_wall081": "Perimeters alone 10&#8202;% thinner (0.81&#8202;mm)",
}


def rows(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def esc(s):
    return html.escape(str(s), quote=True)


def relp(p):
    return os.path.relpath(p, D)


def img_uri(png, width=1100, quality=88):
    from PIL import Image
    im = Image.open(png).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def video_uri(path):
    return "data:video/mp4;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def films_html(items, caption, n):
    """Clips side by side under label bars: items = [(label, mp4, poster png)]."""
    cells = "".join(f'<div style="flex:1 1 0;min-width:240px"><div style="background:#000;color:#fff;font-size:15px;padding:2px 6px">'
                    f'{esc(lab)}</div><video controls muted loop playsinline preload="metadata" poster="{img_uri(poster, 700)}" '
                    f'src="{video_uri(mp4)}" style="width:100%"></video></div>' for lab, mp4, poster in items)
    return (f'<figure id="film{n}"><div style="display:flex;gap:8px;flex-wrap:wrap">{cells}</div>'
            f'<figcaption>Figure {n}. {caption}</figcaption></figure>')


def table(head, body, num_cols=(), caption=None, n=None):
    cap = f"<caption>Table {n}. {caption}</caption>" if caption else ""
    th = "".join(f'<th{" class=n" if k in num_cols else ""}>{h}</th>' for k, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td{" class=n" if k in num_cols else ""}>{c}</td>' for k, c in enumerate(r)) + "</tr>"
                 for r in body)
    return f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def fe_cases():
    """{case: {probe: row}} with the calibration's 11/22 base-grid rows as cal4_base, and TPU volumes from the voxel counts."""
    T = {}
    for r in rows(CAL / "fe" / "fe.jsonl"):
        if r["tip"] == "ell11_g30" and r["grid"] == "base":
            T.setdefault("cal4_base", {})[r["probe"]] = r
    for r in rows(D / "fe.jsonl"):
        T.setdefault(r["case"], {})[r["probe"]] = r
    vol = {}
    for r in rows(CAL / "fe" / "meta.jsonl"):
        if r["tip"] == "ell11_g30" and r["grid"] == "base":
            a = r["aux_voxels"]
            vol["cal4_base"] = (a["wall"] + a["skin"] + a["core"] * CORE0) * VOX_MM3
    for r in rows(D / "meta.jsonl"):
        a = r["aux_voxels"]
        vol[r["case"]] = (a["wall"] + a["skin"] + a["core"] * (r.get("core_solid_fraction") or CORE0)) * VOX_MM3
    return T, vol


def approach(T, case, probe=ROD, F="1"):
    return T[case][probe]["per_force"][F]["approach_um"]


def rms_vs(T, case, ref):
    d = [T[case][p]["per_force"][F]["approach_um"] / T[ref][p]["per_force"][F]["approach_um"] - 1
         for p in T[ref] if p in T[case] for F in ("0.5", "1", "3")]
    return 100 * float(np.sqrt(np.mean(np.square(d)))), len(d)


def draw(T, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "font.family": "serif"})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), dpi=100, sharey=True)
    Fs = [0.5, 1.0, 3.0, 6.0]
    sets = [("Concept geometry", [("cal4_base", "Calibrated print (10-18)", "k", "-"), ("a_v2", "Concept a's tip", "tab:blue", "--"),
                                  ("b_v2", "Concept b's pad", "tab:red", "-."), ("v2_none", "No magnets", "0.5", ":")]),
            ("Under-extrusion", [("v2_5", "As sliced", "k", "-"), ("v2_5_uex05", "5 % narrower beads", "tab:orange", "--"),
                                 ("v2_5_uex10", "10 % narrower beads", "tab:red", "-."),
                                 ("v2_5_gyr10", "Gyroid beads alone 10 %", "tab:green", ":"),
                                 ("v2_5_wall081", "Perimeters alone 10 %", "tab:purple", ":")])]
    for ax, (title, cases) in zip(axs, sets):
        for c, lab, col, ls in cases:
            if c not in T or ROD not in T[c]:
                continue
            y = [T[c][ROD]["per_force"][f"{F:g}"]["approach_um"] for F in Fs]
            ax.plot(Fs, y, ls, color=col, marker="o", ms=4, lw=1.6, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(Fs)
        ax.set_xticklabels([f"{F:g}" for F in Fs])
        ax.set_yticks([100, 150, 200, 300, 400, 600, 800])
        ax.set_yticklabels(["100", "150", "200", "300", "400", "600", "800"])
        ax.minorticks_off()
        ax.set_xlabel("Normal load (N)")
        ax.set_title(title, fontsize=14)
        ax.grid(True, which="both", lw=0.4, color="0.85")
        ax.legend(fontsize=12, frameon=False, loc="upper left")
    axs[0].set_ylabel("Approach under the 12.5 mm rod (µm)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def build():
    T, vol = fe_cases()
    audit = json.loads((D / "print_audit.json").read_text())["plates"]
    cells = {round(r["w_mm"], 4): r for r in rows(D / "cells.jsonl")}
    import fe_contact_calibration_summary as SU
    d1 = SU.d1()
    media = D / "media"
    fig1 = media / "20261019-v2_tip_approach.png"
    draw(T, fig1)

    ref = "cal4_base"
    a1 = lambda c: approach(T, c)                      # noqa: E731
    pct = lambda c, r=ref: 100 * (a1(c) / a1(r) - 1)    # noqa: E731
    dv = lambda c, r="v2_5": 100 * (vol[c] / vol[r] - 1)  # noqa: E731
    rms_a, n_a = rms_vs(T, "a_v2", ref)
    rms_b, _ = rms_vs(T, "b_v2", ref)
    rms_f, _ = rms_vs(T, "cal4_fine", ref)
    fit = audit["fit_sizes_20261019"]
    good = audit["good_20261018"]
    tp = audit["tip_pad_20261019"]
    mA = [v for k, v in fit["mass_g_by_object"].items() if k.startswith("A")]
    mB = [v for k, v in fit["mass_g_by_object"].items() if k.startswith("B")]
    blk = d1["block4_g30|ready|M1|pads"]
    e11 = d1["ell11_g30|ready|M2|pads"]
    own = d1["ell11_g30|own|M2|pads"]
    ratio10 = pct("v2_5_uex10", "v2_5") / -dv("v2_5_uex10")
    ratio05 = pct("v2_5_uex05", "v2_5") / -dv("v2_5_uex05")

    lede_parts = [f"The calibration of 2026-10-18 applies to both v2 fingertips: concept a's 11/22 tip lies within {rms_a:.1f}&#8202;% RMS "
                  f"of the calibrated print in finite elements, b's shorter pad is {-pct('b_v2'):.1f}&#8202;% stiffer."]
    _sj = D / "sensing.json"
    if _sj.exists():
        _sr = {r["case"]: r for r in json.loads(_sj.read_text())["rows"]}
        if "a_v2" in _sr and "b_v2" in _sr:
            lede_parts.append(f"With the 11/22 tip b resolves {_sr['b_v2']['gain']['centre_force_N']:.3f}&#8202;N against a's "
                              f"{_sr['a_v2']['gain']['centre_force_N']:.3f}&#8202;N, as with the block.")
    _m3 = d1.get("ell11_g30|ready|M3|pads")
    if _m3 and _m3.get("complete"):
        lede_parts.append(f"With the series spring in both simulators, pads and Drake end "
                          f"{_m3['pads_vs_drake']['median_rms_angle_all_deg']:.1f}&#176; RMS apart instead of "
                          f"{e11['pads_vs_drake']['median_rms_angle_all_deg']:.1f}&#176;: the missing spring caused most of the 10-18 split.")
    _hv = rows(ROOT / "docs/experiments/20261019-v2_finger/cells/hom.jsonl")
    _hb = rows(ROOT / "docs/experiments/20261019-v2_finger/cells_base/hom.jsonl")
    if _hv and _hb:
        _h = lambda rr, v: sum(1 for r in rr if r["variant"] == v and r.get("status") == "ok" and r["held"])  # noqa: E731
        lede_parts.append(f"The 9.75&#8202;mm longer finger keeps the cylinder in {_h(_hv, 'ell11_g30_v2a')} of 108 HOM runs against "
                          f"{_h(_hb, 'ell11_g30')}.")
    lede_parts.append(f"Under-extrusion softens a tip about 2&#8202;% per 1&#8202;% of missing TPU.")
    lede = " ".join(lede_parts)

    S = []
    S.append(("terms", "Terms and metrics", f"""
<dl class="terms">
<dt>Approach</dt><dd>Travel of a rigid, frictionless indenter along the palmar normal from first touch of the tip's surface at a
given normal load, in &#181;m, from linear finite elements (scripts/fe_contact_calibration_fe.py's model: two 0.45&#8202;mm perimeters and
four 0.2&#8202;mm skins of solid TPU at 12&#8202;MPa over the homogenized gyroid core, bonded magnets, the back and mount faces fixed). A
larger approach is a softer tip.</dd>
<dt>Rod</dt><dd>The 12.5&#8202;mm-radius cylinder across the finger at the contact point (probe <code>{ROD}</code>), the bench and bed tool
(the 25&#8202;mm screwdriver shaft).</dd>
<dt>Probe set</dt><dd>The calibration's 18 probes (a 2&#8202;mm sphere at seven points, rods of radius 5, 8 and 12.5&#8202;mm across the finger
at three heights, rods of radius 8 and 12.5&#8202;mm along it) at 0.5, 1 and 3&#8202;N: 54 approaches per case.</dd>
<dt>RMS change</dt><dd>Root mean square over the 54 approaches of a case's approach over the reference's minus one, in %.</dd>
<dt>Pressure arm $\\bar r$</dt><dd>Pressure-weighted mean distance of the contact pressure from its centre under the rod at 1&#8202;N, in mm;
the full-slip torque about the normal is $\\mu F \\bar r$.</dd>
<dt>Under-extrusion</dt><dd>Every bead narrower than the slicer's 0.45&#8202;mm on the same toolpath, by the share given: the perimeters
thinner by that share and the gyroid cell rebuilt with narrower beads and homogenized again (scripts/tpu_tip_fem_cells.slicer_cell);
skins and the solid modulus unchanged.</dd>
<dt>TPU volume</dt><dd>Perimeters, skins and the core's solid fraction counted on the finite elements' 0.05&#8202;mm voxel grid, in
mm&#179;; its change stands for the change in a part's mass.</dd>
<dt>Retraction, long travel</dt><dd>From the sliced G-code: a retraction is a filament pull-back before a travel; a long travel is a
retracted travel longer than 20&#8202;mm, nearly always between parts, where strings form across open air.</dd>
<dt>Part mass</dt><dd>Filament extruded inside a part's object label (straight and arc moves, 1.75&#8202;mm filament, 1.26&#8202;g/cm&#179;), in g.</dd>
<dt>Same branch, RMS angle, force difference</dt><dd>The calibration study's consistency test: the pads and Drake replay the same
logged servo commands; a pair is on the same branch when the tool axes end within 15&#176;; the RMS angle is the root mean square of the
angle between their tool axes over the replay, in degrees; the force differences are the RMS differences of each finger's normal and
friction force on the tool, in N. Bounds: at least 5 of 6 runs with the same held outcome, a median RMS angle of at most 3&#176; or the
pads' seed spread, and median force differences of at most 0.25&#8202;N normal and 0.35&#8202;N friction.</dd>
<dt>Force resolution</dt><dd>The readout model's Cram&#233;r-Rao bound (one standard deviation) of the normal force at a contact point
from one 141&#8202;Hz sample of the five MLX90393 chips, with the contact position and a uniform background field also unknown, in N.</dd>
<dt>Cell, held, HOM, pinned iCEM, servo plant</dt><dd>As in the readiness study of 10-16. Cell: one cylinder (10, 16 or 25&#8202;mm in
diameter, 100&#8202;mm long, 25&#8202;g, on a 3&#8202;mm post) and one finger layout of the scale study's symmetric tripods, $t$ from &#8722;1 (wide)
to 1 (compact). Held: at the end the cylinder is less than 20&#8202;mm below its height after the grip and at least two fingers each press
on it with its weight; three loaded: all three. HOM: the relative contact-velocity turn controller of arXiv 2609.25619 at 100&#8202;Hz.
Pinned iCEM: the scale study's sampling planner (64 rollouts of 0.5&#8202;s replanned every 20&#8202;ms, a cost that keeps three pads loaded).
Servo plant: position actuators of gain 0.72&#8202;N&#8202;m/rad and torque limit 0.18&#8202;N&#8202;m.</dd>
</dl>"""))

    def geo_row(c):
        r0 = T[c][ROD]["per_force"]
        rm, _ = rms_vs(T, c, ref)
        return [LABEL[c], f"{vol[c]:.0f}", f"{r0['0.5']['approach_um']:.1f}", f"{r0['1']['approach_um']:.1f}",
                f"{r0['3']['approach_um']:.1f}", f"{pct(c):+.1f}" if c != ref else "0", f"{rm:.1f}" if c != ref else "&#8211;",
                f"{r0['1']['r_bar_mm']:.3f}"]

    geo = [c for c in ("cal4_base", "v2_5", "a_v2", "b_v2", "v2_none", "cal4_fine", "v2_5_skin07") if c in T]
    S.append(("geometry", "The v2 fingertips against the calibrated print", f"""
<p>The calibration fitted Drake's field and the pads to finite elements of the 11/22 half-ellipsoid printed with four 3&#8202;&#215;&#8202;2&#8202;mm
magnets 7.0&#8202;mm deep. The v2 tips have five magnets 7.5&#8202;mm deep (corners at $y = \\pm 2.5$&#8202;mm, 4.9&#8202;mm along the finger either
side of a centre magnet), concept a cuts two dovetail key channels 1.9&#8202;mm into the back face over the 5.1&#8202;mm nearest the mount, and
concept b cuts the tip 3.55&#8202;mm short at the back (13.45 of 17.0&#8202;mm deep) and rests that face on its cover. Both share the 11/22
palmar surface and its 14.8&#8202;&#215;&#8202;21.41&#8202;mm outline; a's tip looks larger on the plate because it keeps the full depth
(Table 3 gives the masses). Each change moves the approach by less than the calibration's 5&#8202;% approach bound: the magnets
add {pct('v2_none', 'v2_5'):.1f}&#8202;% of stiffness, a's channels change it by {abs(pct('a_v2', 'v2_5')):.2f}&#8202;%, and b's shorter body makes the
pad {-pct('b_v2'):.1f}&#8202;% stiffer under the rod ({rms_b:.1f}&#8202;% RMS over the probe set). The pressure arm stays at
{T['a_v2'][ROD]['per_force']['1']['r_bar_mm']:.3f}&#8202;mm in every case, since the palmar surface sets it. The fine grid (0.2&#8202;mm over the
contact region; 202,668 against 137,322 degrees of freedom) changes the calibrated tip's rod approach by
{pct('cal4_fine'):+.1f}&#8202;% and the probe set by {rms_f:.1f}&#8202;% RMS, inside the 5&#8202;% the calibration's capsule grid check allowed.
The as-sliced skins of the upright prints (0.7&#8202;mm, seven 0.1&#8202;mm layers) change it by {pct('v2_5_skin07', 'v2_5'):+.2f}&#8202;%.</p>
{table(["Case", "TPU volume (mm&#179;)", "Approach at 0.5&#8202;N (&#181;m)", "At 1&#8202;N", "At 3&#8202;N", "Change at 1&#8202;N (%)",
        "RMS change, 54 approaches (%)", "$\\bar r$ at 1&#8202;N (mm)"],
       [geo_row(c) for c in geo], num_cols=(1, 2, 3, 4, 5, 6, 7), n=1,
       caption="Finite elements of the 11/22 tip under the 12.5&#8202;mm rod (fe.jsonl, model values, solid 12&#8202;MPa). Changes against "
               "the calibrated print of 10-18 on the base grid.")}"""))

    def uex_row(c):
        dvv = dv(c)
        dp = pct(c, "v2_5")
        bead = T[c][ROD].get("bead_mm", 0.45)
        cell = cells.get(round(bead, 4))
        return [LABEL[c], f"{cell['rho']:.3f}" if cell else f"{CORE0:.3f}", f"{dvv:+.1f}", f"{a1(c):.1f}", f"{dp:+.1f}",
                f"{dp / -dvv:.1f}" if dvv < -0.05 else "&#8211;"]

    uex = [c for c in ("v2_5_uex05", "v2_5_uex10", "v2_5_gyr10", "v2_5_wall081") if c in T]
    S.append(("print", "Under-extrusion and the stiffness of a print", f"""
<p>Wet filament, ooze before a travel and a partial clog all lay down narrower beads. In the finite elements, beads 5 and 10&#8202;%
narrower on the v2 tip leave {-dv('v2_5_uex05'):.1f} and {-dv('v2_5_uex10'):.1f}&#8202;% less TPU and let the rod sink
{pct('v2_5_uex05', 'v2_5'):.1f} and {pct('v2_5_uex10', 'v2_5'):.1f}&#8202;% further at 1&#8202;N (Table 2, Figure 1): {ratio05:.1f} and
{ratio10:.1f}&#8202;% of approach per 1&#8202;% of missing TPU. The perimeters and the gyroid share the effect about equally. A part whose
mass falls 2.5&#8202;% short of its siblings therefore leaves the calibration's 5&#8202;% approach bound, and weighing to 0.01&#8202;g resolves
0.4&#8202;% of a 2.85&#8202;g tip. Steam voids from wet filament lower the solid modulus instead of the bead width; in these linear elements
the approach then grows about in proportion to the modulus lost.</p>
{table(["Case", "Gyroid solid fraction", "TPU change (%)", "Approach at 1&#8202;N (&#181;m)", "Approach change (%)",
        "Approach change per 1&#8202;% TPU lost"],
       [uex_row(c) for c in uex], num_cols=(1, 2, 3, 4, 5), n=2,
       caption="Under-extruded v2 tips (five magnets) against the as-sliced v2 tip "
               f"({a1('v2_5'):.1f}&#8202;&#181;m at 1&#8202;N, {vol['v2_5']:.0f}&#8202;mm&#179; of TPU). The cells of the narrower gyroid: "
               "cells.jsonl (homogenized at 0.05&#8202;mm voxels; the 0.45&#8202;mm cell reproduces the 10-09 table).")}
<figure id="fig1"><img src="{img_uri(fig1)}" alt="Approach against load for the concept geometries and for under-extrusion">
<figcaption>Figure 1. Approach under the 12.5&#8202;mm rod against the normal load. Left, the v2 concepts against the calibrated print:
the curves lie within 3&#8202;% of each other. Right, under-extrusion on the v2 tip: 10&#8202;% narrower beads move the whole curve up by
about a fifth.</figcaption></figure>"""))

    def plate_row(key, p):
        masses = sorted(p["mass_g_by_object"].values())
        return [esc(p["label"]), str(p["objects"]), f"{p['retractions_per_layer']:.1f}", f"{p['travels_over_20mm_per_layer']:.1f}",
                f"{masses[0]:.2f}&#8211;{masses[-1]:.2f}", f"{p['slicer_total_g']:.2f}", esc(p.get("slicer_time") or "")]

    st = fit["settings"]
    S.append(("stringing", "Stringing on the XL plates", f"""
<p>All three plates print on the settings of Siraya's XL profile for tool 2: {st['temperature'].split(',')[1]}&#8202;&#176;C,
{st['filament_retract_length'].split(',')[1]}&#8202;mm retraction at {st['retract_speed'].split(',')[1]}&#8202;mm/s, no wipe, no lift, travel at
{st['travel_speed']}&#8202;mm/s with avoid-crossing-perimeters on. Siraya's 85A user guide gives 200&#8211;225&#8202;&#176;C and 1&#8211;3&#8202;mm of
retraction, so the profile runs hot and pulls back little, which strings on any plate. The fit-size plate adds to it: eight parts make
{fit['retractions_per_layer']:.0f} retractions and {fit['travels_over_20mm_per_layer']:.0f} long travels per layer against
{good['retractions_per_layer']:.0f} and {good['travels_over_20mm_per_layer']:.0f} on the four-part print of 10-18 (Table 3). Between
about 16:20 and 18:40 on 10-19 the spool sat outside the dryer on the A1 mini and the P1S, and Siraya's guide calls for 4&#8211;6&#8202;h at
50&#8211;65&#8202;&#176;C when 85A has taken up moisture. Strings form outside the parts; a part's stiffness depends on the TPU inside it (Section 3), which a
scale measures against the slicer's per-part masses. The A tips weigh {min(mA):.3f}&#8211;{max(mA):.3f}&#8202;g and the B pads {min(mB):.3f}&#8211;{max(mB):.3f}&#8202;g
in the G-code, the same to 0.004&#8202;g across the four sizes.</p>
{table(["Plate", "Parts", "Retractions per layer", "Long travels per layer", "Part mass (g)", "Plate filament (g)", "Slicer time"],
       [plate_row(k, audit[k]) for k in ("good_20261018", "tip_pad_20261019", "fit_sizes_20261019")],
       num_cols=(1, 2, 3, 4, 5), n=3,
       caption="The 85A plates' G-code (print_audit.json, scripts/tpu_print_audit.py; 213 layers each). Part masses exclude magnets; "
               "the tip and pad of the tip-and-pad plate each get five 0.106&#8202;g magnets at the pauses.")}"""))

    def con_row(lab, v):
        s = v["pads_vs_drake"]
        ang = s.get("median_rms_angle_same_deg")
        pp = v["pass_parts"]
        alla = s.get("median_rms_angle_all_deg")
        return [lab, f"{s['n']}", f"{s['held_agree']} of {s['n']}", f"{s['same_branch']} of {s['n']}",
                f"{ang:.1f}" if ang is not None else "&#8211;", f"{alla:.1f}" if alla is not None else "&#8211;",
                f"{v['angle_bound_deg']:.1f}",
                (f"{s['median_rms_dN_same_N']:.2f} / {s['median_rms_dT_same_N']:.2f}" if s.get("median_rms_dN_same_N") is not None
                 else "&#8211;"),
                (", ".join(k for k, ok in (("angle", pp["angle"]), ("force", pp["force"]), ("held", pp["held"])) if not ok)
                 or "none").capitalize()]

    capd = d1.get("cap_g30|ready|M1|pads")
    if capd and capd.get("complete"):
        sc_ = capd["pads_vs_drake"]
        cap_txt = (f"The capsule, replayed on 2026-10-20 after the watchdog stopped its first run, ends on the same branch in "
                   f"{sc_['same_branch']} of {sc_['n']} pairs, {sc_['median_rms_angle_all_deg']:.1f}&#176; RMS apart over all pairs, with "
                   f"normal forces {sc_['median_rms_dN_same_N']:.2f}&#8202;N apart.")
    else:
        cap_txt = "The capsule's replays stopped at 1 of 6 when the memory watchdog killed the job at 20:57 on 10-19."
    S.append(("control", "Pads against Drake with the block as control", f"""
<p>On 10-18 the Drake replays of the block and the capsule were cut for time, so the 11/22 tip's failed consistency had no control.
On 2026-10-19 Drake replayed the block's readiness commands at M1 (the modulus fitted to the finite elements' approach), all six. On the
same branch in {blk['pads_vs_drake']['same_branch']} of 6 pairs, the block's pads and Drake differ by a median
{blk['pads_vs_drake']['median_rms_angle_same_deg']:.1f}&#176; RMS against a {blk['angle_bound_deg']:.1f}&#176; bound and by
{blk['pads_vs_drake']['median_rms_dN_same_N']:.2f}&#8202;N in normal force against 0.25&#8202;N (Table 4). The block therefore misses the
force bound as the 11/22 tip does. On the branch the two tips differ: the block stays on Drake's in all 6 pairs, the 11/22 tip leaves
it in {e11['pads_vs_drake']['n'] - e11['pads_vs_drake']['same_branch']} of 6 (all on the 25&#8202;mm cell) and in all 3 runs of its own
plan. The force difference appears on both tip shapes, so a calibration of the normal response does not remove it; the branch split
belongs to the 11/22 tip. {cap_txt}</p>
{table(["Tip, calibration, commands", "Pairs", "Held agreement", "Same branch", "Median RMS angle, same branch (&#176;)",
        "Median RMS angle, all pairs (&#176;)", "Angle bound (&#176;)", "Normal / friction force difference (N)", "Fails"],
       [con_row("Block, M1, readiness plans", blk), con_row("Half-ellipsoid 11/22, M2, readiness plans", e11),
        con_row("Half-ellipsoid 11/22, M2, its own plan", own)]
       + ([con_row("Capsule r 8.5&#8202;mm, M1, readiness plans", capd)] if capd and capd.get("complete") else []),
       num_cols=(1, 4, 5, 6), n=4,
       caption="Pads against Drake on the same logged servo commands (the calibration study's replays/icem.jsonl and its summary code; "
               "model values). The block's Drake rows are of 10-19; the 11/22 rows are those of 10-18.")}"""))

    # ------------------------------------------------------------------ queued runs, shown when their rows exist
    tn = [4]

    def nxt():
        tn[0] += 1
        return tn[0]

    sj = D / "sensing.json"
    if sj.exists():
        sr = {r["case"]: r for r in json.loads(sj.read_text())["rows"]}
        ev = {c: json.loads((ROOT / f"docs/experiments/20261018-dip_link_board_mount/v2/{c}/evaluation.json").read_text())["sensing"]
              for c in ("a", "b")}
        blk = {"a_v2": ev["a"]["block5_force_resolution_30_N"], "b_v2": ev["b"]["design_point"]["force_resolution_gain_N"]}
        lab = {"a_v2": "Concept a: the whole 11/22 tip, chips 0.37&#8202;mm behind its back face",
               "b_v2": "Concept b: the 11/22 pad cut 3.55&#8202;mm short, chips 0.42&#8202;mm behind its cut face"}
        srows = [[lab[c], f"{r['magnet_travel_um_per_N']:.2f}", f"{r['B_max_component_mT']:.1f}", f"{r['gain_sel']}",
                  f"{r['gain']['centre_force_N']:.3f}", f"{r['gain']['median_force_N']:.3f}",
                  f"{r['gain']['range_force_N'][0]:.3f}&#8211;{r['gain']['range_force_N'][1]:.3f}", f"{blk[c]:.3f}"]
                 for c, r in sorted(sr.items())]
        ra, rb = sr.get("a_v2"), sr.get("b_v2")
        txt = ""
        if ra and rb:
            txt = (f"With the 11/22 tip, concept a resolves {ra['gain']['centre_force_N']:.3f}&#8202;N at the contact point and concept b "
                   f"{rb['gain']['centre_force_N']:.3f}&#8202;N, against {blk['a_v2']:.3f} and {blk['b_v2']:.3f}&#8202;N for the five-magnet "
                   f"block on the DIP page: with the 11/22 tip b resolves {ra['gain']['centre_force_N'] / rb['gain']['centre_force_N']:.1f} times finer than a, as "
                   f"it does with the block. b also locates the contact to {rb['gain']['centre_sig_y_mm']:.2f} by "
                   f"{rb['gain']['centre_sig_z_mm']:.2f}&#8202;mm (one standard deviation across and along the finger) against a's "
                   f"{ra['gain']['centre_sig_y_mm']:.2f} by {ra['gain']['centre_sig_z_mm']:.2f}&#8202;mm. Its magnets move less per newton "
                   f"({rb['magnet_travel_um_per_N']:.1f} against {ra['magnet_travel_um_per_N']:.1f}&#8202;&#181;m/N, the shorter pad being "
                   f"stiffer behind them), but its chips sit 3.5&#8202;mm closer to them, where the field changes faster. ")
        t_s = nxt()
        S.append(("sensing", "Force resolution of the 11/22 tips in concepts a and b", f"""
<p>The DIP study compared a's and b's force sensing on the five-magnet block, because the 11/22 half-ellipsoid had no finite
elements with magnets (v2/a|b/evaluation.json: "only its static field, gain and chip-to-magnet distance"). The rows of 2026-10-20
give the 11/22 tip in each concept the readout model's inputs: the five magnets' rigid motion under the rod at the contact point and under
unit normal and shear patch loads at 15 points of the face (y = &#8722;3, 0, 3&#8202;mm; 4&#8202;mm either side of the contact point
along the finger; scripts/v2_tip_fe_checks.py <code>--sense</code>). The bound is the readout model's
(scripts/tpu_tip_magnet_sensing.analyse): the Cram&#233;r-Rao bound of the normal force from one 141&#8202;Hz sample of the five
chips, with the contact position and a uniform background field unknown, at the chip noise of the gain that the static field
allows. {txt}</p>
{table(["Tip and board", "Magnet travel (&#181;m/N)", "Largest field component (mT)", "GAIN_SEL", "Force resolution at the contact point (N)",
        "Median over the 15 points (N)", "Range over the 15 points (N)", "Five-magnet block, DIP page (N)"], srows,
       num_cols=(1, 2, 3, 4, 5, 6, 7), n=t_s,
       caption="Readout bound of the v2 tips (sensing.json from sense_fe.jsonl; model values, solid 12&#8202;MPa, N52 magnets at "
               "1.45&#8202;T).")}"""))

    m3 = d1.get("ell11_g30|ready|M3|pads")
    m3o = d1.get("ell11_g30|own|M3|pads")
    cap = d1.get("cap_g30|ready|M1|pads")
    if m3 and m3["pads_vs_drake"]["n"]:
        rows_m3 = [con_row("Half-ellipsoid 11/22, M2, readiness plans (10-18)", e11),
                   con_row("Half-ellipsoid 11/22, M3 with the spring in both, readiness plans", m3)]
        if m3o and m3o["pads_vs_drake"]["n"]:
            rows_m3.append(con_row("Half-ellipsoid 11/22, M3 with the spring in both, its own plan", m3o))
        sm = m3["pads_vs_drake"]
        own_txt = ""
        if m3o and m3o["pads_vs_drake"]["n"]:
            so = m3o["pads_vs_drake"]
            own_txt = (f"; on the tip's own plan {so['same_branch']} of {so['n']} runs end on the same branch (0 of 3 at M2), the median RMS "
                       f"angle falls from {own['pads_vs_drake']['median_rms_angle_all_deg']:.1f} to {so['median_rms_angle_all_deg']:.1f}&#176;"
                       f" and the own-plan criterion {'passes' if m3o['pass'] else 'still fails'}")
        t_m = nxt()
        d6 = SU.d1(seeds=(0, 1, 2, 3, 4, 5))
        six = ""
        k6 = [("Block, M1", d6.get("block4_g30|ready|M1|pads")), ("Half-ellipsoid 11/22, M2", d6.get("ell11_g30|ready|M2|pads")),
              ("Half-ellipsoid 11/22, M3 with the spring in both", d6.get("ell11_g30|ready|M3|pads"))]
        if all(v and v["pads_vs_drake"]["n"] > 6 for _, v in k6):
            t6 = nxt()
            r6 = [[lab, f"{v['pads_vs_drake']['n']}", f"{v['pads_vs_drake']['same_branch']} of {v['pads_vs_drake']['n']}",
                   f"{v['pads_vs_drake']['median_rms_angle_all_deg']:.1f}", f"{v['pads_vs_drake']['median_rms_dN_same_N']:.2f}",
                   f"{v['pads_seed_spread'].get('median_rms_angle_same_deg') or float('nan'):.1f}"] for lab, v in k6]
            six = (f"<p>Three more placements (seeds 3&#8211;5) on the same commands give the counts over six (Table {t6}).</p>"
                   + table(["Tip and calibration, readiness plans", "Pairs", "Same branch", "Median RMS angle, all pairs (&#176;)",
                            "Normal force difference (N)", "Pads' seed spread (&#176;)"], r6, num_cols=(1, 3, 4, 5), n=t6,
                           caption="Pads against Drake over placements 0&#8211;5 (fe_contact_calibration_summary.d1 with six seeds; "
                                   "model values)."))
        S.append(("spring", "The series spring in both simulators", f"""
<p>Calibration M3 puts the 11/22 tip's modulus (Drake 65.0, pads 53.0&#8202;MPa) behind a 9.7&#8202;N/mm series spring. On 10-18 only the
pads' D1&#8211;D8 replays had the spring, so the consistency test ran at M2, the stiff modulus alone. On 2026-10-20 both simulators replayed the
readiness commands with the spring: a slide joint behind each tip, held by a joint equality of that stiffness in the pads
(fe_contact_calibration_sim.add_spring) and by a PrismaticSpring with critical damping in Drake
(curved_tips_cell_drake.CellDrake, <code>k_s</code>). With the spring, {sm['same_branch']} of {sm['n']} readiness pairs end on the
same branch (3 of 6 at M2), and the median RMS angle over all pairs falls from {e11['pads_vs_drake']['median_rms_angle_all_deg']:.1f}
to {sm['median_rms_angle_all_deg']:.1f}&#176;{own_txt} (Table {t_m}). The readiness plans now miss the criterion on the normal force
alone ({sm['median_rms_dN_same_N']:.2f}&#8202;N against 0.25&#8202;N), as the block does; the pairs still on different branches are on the
25&#8202;mm cell. The angle bound widens to the pads' own seed spread, {m3['angle_bound_deg']:.1f}&#176; at M3: the pads' turns on the
25&#8202;mm cell differ by that much between placements. Most of the split of 10-18 therefore came from M2 standing in for the spring.</p>
{table(["Tip, calibration, commands", "Pairs", "Held agreement", "Same branch", "Median RMS angle, same branch (&#176;)",
        "Median RMS angle, all pairs (&#176;)", "Angle bound (&#176;)", "Normal / friction force difference (N)", "Fails"], rows_m3, num_cols=(1, 4, 5, 6), n=t_m,
       caption="Pads against Drake with and without the series spring (replays/icem.jsonl; model values).")}
{six}"""))
    bed = [r for r in rows(CAL / "bed" / "T2.jsonl") if r.get("status") == "complete"]
    if any(r["arm"] == "ir1000" for r in bed):
        names = {"block4_g30": "Block", "cap_g30": "Capsule r 8.5&#8202;mm", "ell16_g30": "Half-ellipsoid 16/32&#8202;mm",
                 "ell11_g30": "Half-ellipsoid 11/22&#8202;mm"}
        cr = {(r["tip"], r["calib"], r["arm"], r["N"]): r.get("creep_deg_s") for r in bed}
        brows = []
        for tip in ("block4_g30", "cap_g30", "ell16_g30", "ell11_g30"):
            for c in ("M1", "M2"):
                vals = [cr.get((tip, c, arm, 1.0)) for arm in ("pads", "ir1000", "drake")]
                if vals[1] is None:
                    continue
                brows.append([names[tip], c] + [f"{v:.3f}" if v is not None else "&#8211;" for v in vals])
        t_c = nxt()
        S.append(("creep", "Creep of the pads at impratio 1000", f"""
<p>Under a held twist at half the onset torque the pads creep where Drake sticks (the calibration page's bed, case T2). MuJoCo's soft
friction rows creep in proportion to their impedance, so a tenfold impratio should cut the pads' creep about tenfold. Table {t_c}
gives the creep at 1&#8202;N per tip for the pads at impratio 100 (the planner's), at 1000, and for Drake.</p>
{table(["Tip", "Calibration", "Pads, impratio 100 (&#176;/s)", "Pads, impratio 1000 (&#176;/s)", "Drake (&#176;/s)"], brows,
       num_cols=(2, 3, 4), n=t_c, caption="Creep under a held twist at half the onset torque, 1&#8202;N per tip (bed/T2.jsonl; model values).")}"""))
    V2 = ROOT / "docs/experiments/20261019-v2_finger"
    hv, hb = rows(V2 / "cells" / "hom.jsonl"), rows(V2 / "cells_base" / "hom.jsonl")
    if hv:
        iv, ib = rows(V2 / "cells" / "icem.jsonl"), rows(V2 / "cells_base" / "icem.jsonl")
        ready = rows(ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness/cells/icem.jsonl")

        def homstat(rr, var, cell):
            x = [r for r in rr if r["variant"] == var and r.get("status") == "ok"
                 and f"{r['shape']}:{r['d_mm']:g}:{r['t']:g}" == cell]
            h = sorted(r["turn_end_deg"] for r in x if r["held"])
            return (f"{len(h)} of {len(x)}", f"{h[len(h) // 2]:.1f}" if h else "&#8211;") if x else ("&#8211;", "&#8211;")

        def icstat(rr, var, cell):
            x = [r for r in rr if r["variant"] == var and r.get("status") == "ok"
                 and f"{r['shape']}:{r['d_mm']:g}:{r['t']:g}" == cell]
            if not x:
                return "&#8211;"
            r = x[0]
            return f"{'held' if r['held'] else 'dropped'}, {r['turn_end_deg']:+.1f}&#176;, {r.get('fingers_loaded_end')} loaded"

        crow = []
        for base, lab in (("ell11_g30", "Half-ellipsoid 11/22"),
                          ("block5_g30", "Block, five magnets")):
            for cell in SC_CELLS:
                hb_ = homstat(hb, base, cell)
                hv_ = homstat(hv, f"{base}_v2a", cell)
                ib_ = icstat(ib, base, cell) if ib else icstat(ready, base, cell)
                iv_ = icstat(iv, f"{base}_v2a", cell)
                crow.append([lab, cell.replace("cylinder:", "Cylinder ").replace(":", " mm, layout t "), hb_[0], hb_[1], hv_[0], hv_[1], ib_, iv_])
        vs = json.loads((ROOT / "assets/mjcf/experimental/20261019-v2_finger/ell11_g30_v2a/README.json").read_text())
        t_v = nxt()
        rhom = {(r["variant"], r["tag"], r["shape"], r["d_mm"], r["cand"], r["seed"]): r for r in
                rows(ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness/cells/hom.jsonl") if r.get("status") == "ok"}
        same = sum(1 for r in hb if r.get("status") == "ok" and (k := (r["variant"], r["tag"], r["shape"], r["d_mm"], r["cand"], r["seed"]))
                   in rhom and rhom[k]["held"] == r["held"] and abs(rhom[k]["turn_end_deg"] - r["turn_end_deg"]) < 1e-6)
        nb = sum(1 for r in hb if r.get("status") == "ok")
        held = lambda rr, var: sum(1 for r in rr if r["variant"] == var and r.get("status") == "ok" and r["held"])  # noqa: E731
        lim = lambda rr: float(np.mean([r.get("tau_at_limit_frac", 0) for r in rr if r.get("status") == "ok"]))   # noqa: E731

        def ticks(rr, part):
            x = [r for r in rr if r.get("status") == "ok"]
            return sum((r.get("nonpad_contact_ticks") or {}).get(part, 0) for r in x) / max(len(x), 1)

        def turn16(rr, var):
            x = [r for r in rr if r["variant"] == var and r.get("status") == "ok" and r["shape"] == "cylinder" and r["d_mm"] == 16
                 and abs(r["t"] - 0.625) < 1e-9]
            return x[0]["turn_end_deg"] if x else float("nan")
        cell10 = lambda rr, var: sum(1 for r in rr if r["variant"] == var and r.get("status") == "ok" and r["held"]  # noqa: E731
                                     and r["d_mm"] == 10)
        v2txt = (f"The CAD finger's rows reproduce those of 10-16 in {same} of {nb} HOM runs, so the two fingers differ only in the DIP "
                 f"link. With the 11/22 tip the v2 finger keeps the cylinder in {held(hv, 'ell11_g30_v2a')} of 108 HOM runs against "
                 f"{held(hb, 'ell11_g30')} on the CAD finger; with the five-magnet block in {held(hv, 'block5_g30_v2a')} against "
                 f"{held(hb, 'block5_g30')}, the block losing the 10&#8202;mm cylinder ({cell10(hv, 'block5_g30_v2a')} of 27 against "
                 f"{cell10(hb, 'block5_g30')}). Pinned iCEM keeps the cylinder on three pads in all {len([r for r in iv + ib if r.get('status') == 'ok'])} "
                 f"plans; on the 16&#8202;mm cell, the first bench task's, the 11/22 tip turns it {turn16(iv, 'ell11_g30_v2a'):+.1f}&#176; on "
                 f"the v2 finger against {turn16(ib, 'ell11_g30'):+.1f}&#176; on the CAD finger. The longer lever costs force: at the PIP "
                 f"servo the pads press 27&#8202;% less per newton-metre (26.61 against 36.36&#8202;mm), the servos spend "
                 f"{100 * lim(hv):.1f}&#8202;% of the HOM ticks at their 0.18&#8202;N&#8202;m limit against {100 * lim(hb):.1f}&#8202;%, and parts "
                 f"other than the pads touch the cylinder in more ticks per run (the board {ticks(hv, 'board'):.0f} against "
                 f"{ticks(hb, 'board'):.0f}, the DIP link {ticks(hv, 'pip_frame'):.0f} against {ticks(hb, 'pip_frame'):.0f}).")
        flm = []
        for tag, lab in (("base", "CAD finger"), ("v2", "v2 finger, 9.75 mm longer")):
            fj = V2 / tag / "media" / "films.jsonl"
            if fj.exists():
                r = [json.loads(l) for l in open(fj)][-1]
                mp4 = V2 / tag / "media" / r["clip"]
                poster = mp4.with_name(mp4.stem + "_end.png")
                flm.append((f"{lab}: {'held' if r['held'] else 'dropped'}, turn {r['turn_end_deg']:+.1f} deg", mp4, poster))
        film_fig = ""
        if flm:
            film_fig = films_html(flm, "HOM on the 16&#8202;mm cylinder (cell t 0.625, the first bench task's cell), the half-ellipsoid "
                                       "11/22 tip on each finger at its best candidate grasp, seed 0; pads touching the cylinder red, "
                                       "the board green. Real time.", 2)
        S.append(("v2finger", "The first bench task on the longer v2 finger", f"""
<p>The v2 DIP link puts the tip 9.75&#8202;mm further from the PIP axis (pad centre 36.36&#8202;mm past it) and weighs
{vs['dip_body']['mass_g']:.1f}&#8202;g with its clamp plate and the board, against 5.6&#8202;g for the CAD finger's link and board. The
readiness study's control tasks ran on both fingers with the same code, scenes, servo plant and planners
(scripts/v2_finger_cells.py; the CAD finger's rows with <code>V2_FINGER_BASE=1</code>): HOM on the nine candidate grasps of each of the four
cells at three seeds, and pinned iCEM on each cell's best HOM grasp (Table {t_v}). {v2txt}</p>
{table(["Tip", "Cell", "HOM held, CAD finger", "Median held turn (&#176;)", "HOM held, v2 finger", "Median held turn (&#176;)",
        "Pinned iCEM, CAD finger", "Pinned iCEM, v2 finger"], crow, num_cols=(2, 3, 4, 5), n=t_v,
       caption="Control tasks of the readiness study on the CAD finger and on the v2 finger (cells.jsonl rows of each; model "
               "values, servo plant kp 0.72&#8202;N&#8202;m/rad, 0.18&#8202;N&#8202;m).")}
{film_fig}"""))
    cc = D / "contact_class"
    if cc.exists() and any(cc.glob("*.csv")):
        rr_ = []
        for f in sorted(cc.glob("*.csv")):
            last = f.read_text().strip().splitlines()[-1].split(",")
            rr_.append([esc(f.stem), last[1], last[2]])
        for f in sorted(cc.glob("*cpu*.json")):
            j = json.loads(f.read_text())
            rr_.append([esc(f.stem), f"{j.get('final_cos', float('nan')):+.3f}", "held" if j.get("held_end") else "lost"])
        t_k = nxt()
        S.append(("contactclass", "Tool contact class in the renderer", f"""
<p>Runs trained before 2026-09-20 saw a tool with MuJoCo's default contact (solref 0.02&#8202;s, solimp 0.9&#8211;0.95), while the
renderer and the CPU rollout built the tool with the scene's class, and 11 of 16 such policies dropped the tool at the trained pose.
<code>scripts/rl_render_rollout.py</code> and <code>scripts/policy_cpu_rollout.py</code> now take the run's class
(<code>--object-contact run</code>, the evaluation suite's rule). Table {t_k} gives the final signed cosine and height of D7's
2026-09-20 policy (results/rl/20260920-0101-d7_clip_s0) under each class.</p>
{table(["Rollout", "Final cos / CPU final cos", "Final height (m) / CPU outcome"], rr_, num_cols=(1,), n=t_k,
       caption="D7's pre-fix policy rendered with the run's class and with the scene's (contact_class/).")}"""))

    S.append(("limits", "Limitations", """
<p>The finite elements are linear and small-strain, with frictionless contact, the gyroid homogenized, NinjaFlex 85A's 12&#8202;MPa for
Siraya's unmeasured modulus, bonded magnets, and the back and mount faces fixed; concept a's channels are left free (the keys press
their flanks in the print) and concept b's rail grooves and screw-head notches beside the cut face are left out. Under-extrusion is
modelled as uniformly narrower beads; real defects cluster at seams and travel starts. The G-code audit counts moves; it does not
see strings. The control rests on one block calibration (M1) and the readiness commands, three seeds per cell.</p>"""))

    _iv = rows(ROOT / "docs/experiments/20261019-v2_finger/cells/icem.jsonl")
    _t = [r["turn_end_deg"] for r in _iv if r.get("status") == "ok" and r["variant"] == "ell11_g30_v2a" and r["d_mm"] == 16]
    turn16_v2 = f"{_t[0]:+.1f}&#176;" if _t else "(pending)"
    S.append(("future", "Future work", f"""
<ul>
<li>Weigh the fit-size parts (A00&#8211;A20: {np.mean(mA):.2f}&#8202;g each in the G-code; B00&#8211;B15: {np.mean(mB):.2f}&#8202;g) and the
magnet tip and pad of the 55&#8202;min plate ({tp['mass_g_by_object'].get('20261018-dip_link_v2a_print_tip_ell11_5_channels_85a_t2_g30', 0):.2f}
and {tp['mass_g_by_object'].get('20261018-dip_link_v2b_print_pad_ell11_5_85a_t2_g30', 0):.2f}&#8202;g plus 0.53&#8202;g of magnets) to
0.01&#8202;g. A part 2.5&#8202;% or more under its siblings sinks 5&#8202;% further than they do (Table 2) and does not stand for the calibrated
tip on the bench.</li>
<li>Dry the spool at 55&#8202;&#176;C for 6&#8202;h and print one A tip alone, then weigh it against the A tips of 10-19. Fewer strings after drying
point to moisture; as many strings point to the profile's 240&#8202;&#176;C and 0.6&#8202;mm retraction, which a print of the same tip at
225&#8202;&#176;C with 1&#8202;mm retraction and wipe on would test, with an indentation, since the nozzle temperature changes the bond between
beads.</li>
<li>Indent one v2 tip under the 25&#8202;mm rod at 0.5, 1 and 3&#8202;N (readiness procedure, step &#8216;indentation&#8217;) to fix the bench factor
&#955;; with a and b within 2.5&#8202;% of the calibrated print, one measurement fixes &#955; for both concepts.</li>
<li>Export the first bench plan for the v2 finger with the 11/22 tip: pinned iCEM on the 16&#8202;mm cell held the cylinder on three
pads and turned it {turn16_v2} on the v2 finger (Section on the longer finger); <code>scripts/sim2real_fingertip_bench.py export</code>
pointed at <code>assets/mjcf/experimental/20261019-v2_finger/ell11_g30_v2a</code> as scripts/v2_finger_cells.py points the cells, then the
plant-condition sweep, the abort test and <code>scripts/real_v1_trajectory_clearance.py</code> before the bench. A plan that keeps the
cylinder in fewer than the 60 of 81 plant conditions of the block's 10-16 plan would argue for re-planning on the measured plant.</li>
<li>The force bound of the consistency test (0.25&#8202;N normal) fails on the block at M1 and on the 11/22 tip at M3 alike (0.54 and
0.37&#8202;N). The pinch-brake test of the contact overview's Future work measures which simulator's force sharing the printed tip
follows; until then the angle and branch criteria carry the comparison.</li>
<li>Put the 11/22 force resolutions (Section on sensing) into the DIP study's evaluations (scripts/dip_link_v2.py, a and b
<code>docs</code> stages) when the DIP page is next rebuilt, in place of the block's.</li>
<li>Runs still queued at the page's build (logs/20261019-overnight/q_all.txt, one job at a time): the capsule's Drake replays, the
impratio-1000 bed arm, the v2 tip on the fine grid, the D7 contact-class renders, seeds 3&#8211;5 of the 11/22 tip at M2 and M3 and of the
block at M1, and films of the 16&#8202;mm task on both fingers. Their sections appear when the page is rebuilt with their rows.</li>
</ul>"""))

    S.append(("files", "Files", f"""
<p>Folder <code>docs/experiments/20261019-v2_tip_fe_checks/</code>: <code>fe.jsonl</code>, <code>meta.jsonl</code> and <code>cells.jsonl</code>
(scripts/v2_tip_fe_checks.py), <code>print_audit.json</code> (scripts/tpu_print_audit.py), this page (scripts/v2_tip_fe_checks_page.py).
The block's Drake replays are rows of <code>docs/experiments/20261018-fe_contact_calibration/replays/icem.jsonl</code>. Related pages:
<a href="{relp(CAL_PAGE)}">the calibration of Drake and the pads to the 85A finite elements</a>, <a href="{relp(DIP_PAGE)}">the DIP link
and its board mount</a>, <a href="{relp(PROC_PAGE)}">the 85A print procedure</a>, <a href="{relp(STATUS_PAGE)}">the status of the
printed-fingertip work</a>.</p>"""))

    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    title = "Checks of the v2 11/22 fingertip and the 9.75 mm longer DIP link"
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "v2 fingertip checks", "H1": title,
                 "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-19",
                 "LEDE": lede, "BYLINE": f"Built {time.strftime('%Y-%m-%d %H:%M')} from the study's rows; model values, nothing measured.",
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None) for m in TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    build()
