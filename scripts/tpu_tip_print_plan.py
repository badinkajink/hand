#!/usr/bin/env python3
"""Prints to send, their sphere-pad calibration, the expected gap between simulation and the printed tip, and the
embedded magnet's signal (2026-10-12): data and HTML sections for scripts/tpu_tip_fem_page.py.

Inputs: docs/experiments/20261009-tpu_tip_fem/magnet_runs.jsonl (scripts/tpu_tip_magnet.py, the whole tip with and
without a magnet), the page's own rows for the insert (Data in scripts/tpu_tip_fem_page.py) and the bed's pad rows.
Magnet field: point dipole of an axially magnetized N52 disc (scripts/tpu_tip_design.py), read by an MLX90393 lying
flat behind the back face (chip z along the tip's x). Noise: the MLX90393 datasheet's Figures 2-3 (GAIN_SEL 7) at about
3 ms per axis, the conversion time that reads all three axes at 100 Hz.

    .venv/bin/python scripts/tpu_tip_print_plan.py stl      # print files and sheets for the plan's tips
    .venv/bin/python scripts/tpu_tip_print_plan.py table    # the plan as text
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_tip_design as TD  # noqa: E402

D = ROOT / "docs/experiments/20261009-tpu_tip_fem"
ROWS = D / "magnet_runs.jsonl"
PRINT_DIR = D / "print"
GAPS = (1.0, 4.0)                    # back face to the chip's sensing point, mm: in a pocket of the mount, behind a 3 mm plate
NOISE_UT = dict(xy=2.3, z=3.5)       # MLX90393 RMS noise per axis at ~3 ms per axis (100 Hz for XYZ), GAIN_SEL 7
RATE_HZ = 100.0
B_RANGE_MT = 4.8                     # full scale at GAIN_SEL 7 (datasheet: at least +-4.8 mT at the highest gain)
PLAN_MAG, PLAN_DEPTH = "d4.76x1.59", 5.0   # chosen 2026-10-12 from the depth runs: least change of E* and least force noise
RECIPE_LBL = {"w95_20": "95A, gyroid 20&#8202;%", "w95_10": "95A, gyroid 10&#8202;%", "w85_50": "85A, gyroid 50&#8202;%",
              "w85_30": "85A, gyroid 30&#8202;%"}
RECIPE_TXT = {"w95_20": "95A, gyroid 20 %", "w95_10": "95A, gyroid 10 %", "w85_50": "85A, gyroid 50 %", "w85_30": "85A, gyroid 30 %"}
INSERT_OF = {"w95_20": "g20_across", "w95_10": "g10_across", "w85_50": "s85_50_across", "w85_30": "s85_30_across"}
MAG_LBL = {"d4.76x1.59": "4.76 &#215; 1.59&#8202;mm", "d3.18x1.59": "3.18 &#215; 1.59&#8202;mm", "none": "none"}


def rows():
    out = {}
    if ROWS.exists():
        for line in open(ROWS):
            r = json.loads(line)
            out[(r["recipe"], r["magnet"], r["depth_mm"])] = r
    return out


def field(row, gap):
    """Field at the chip at rest and its change per newton of normal load (at 1 N) and of shear along z and y (µT/N),
    with the 1-sigma force noise at RATE_HZ (N) for each load."""
    _, mg = TD.pocket(row["magnet"], row["depth_mm"])
    pm = np.array([0.5 * (mg["x_lo"] + mg["x_hi"]), 0.0, TD.Z_C]) * 1e-3
    ps = np.array([TD.X_BACK - gap, 0.0, TD.Z_C]) * 1e-3
    m0 = TD.magnet_moment(row["magnet"]) * np.array([1.0, 0.0, 0.0])
    B0 = TD.dipole_B(ps, pm, m0)

    def dB(t_um, w_mrad, per=1.0):
        t = np.asarray(t_um) * 1e-6 / per
        w = np.asarray(w_mrad) * 1e-3 / per
        return (TD.dipole_B(ps, pm + t, m0 + np.cross(w, m0)) - B0) * 1e6

    sig = np.array([NOISE_UT["z"], NOISE_UT["xy"], NOISE_UT["xy"]])      # chip z along the tip's x
    out = dict(gap_mm=gap, B0_mT=float(np.linalg.norm(B0) * 1e3), r_mm=float(np.linalg.norm(ps - pm) * 1e3))
    loads = dict(normal=(row["per_force"]["1"]["magnet_t_um"], row["per_force"]["1"]["magnet_w_mrad"]),
                 shear_z=(row["shear_z_magnet_t_um_per_N"], row["shear_z_magnet_w_mrad_per_N"]),
                 shear_y=(row["shear_y_magnet_t_um_per_N"], row["shear_y_magnet_w_mrad_per_N"]))
    for k, (t, w) in loads.items():
        S = dB(t, w)
        out[f"S_{k}_uT_per_N"] = S.tolist()
        out[f"S_{k}_norm"] = float(np.linalg.norm(S))
        out[f"sigF_{k}_N"] = float(1.0 / math.sqrt(float(((S / sig) ** 2).sum()))) if np.any(S) else float("inf")
    return out


def plan_rows(R=None):
    """The plan: one magnet tip per print at the depth chosen below, and the no-magnet control of the reference."""
    R = R if R is not None else rows()
    plan = []
    for rec, mag, dep, role in (("w95_20", PLAN_MAG, PLAN_DEPTH, "Reference, easy to print"),
                                ("w85_50", PLAN_MAG, PLAN_DEPTH, "Closest to the pads' twist arm"),
                                ("w85_30", PLAN_MAG, PLAN_DEPTH, "Softer, larger magnet signal"),
                                ("w95_10", PLAN_MAG, PLAN_DEPTH, "Softer in 95A"),
                                ("w95_20", "none", None, "Control without magnet")):
        r = R.get((rec, mag, dep))
        plan.append(dict(recipe=rec, magnet=mag, depth=dep, role=role, row=r))
    return plan


INFILL = {"w95_20": 20, "w95_10": 10, "w85_50": 50, "w85_30": 30}
TPU = {"w95_20": "TPU 95A", "w95_10": "TPU 95A", "w85_50": "TPU 85A", "w85_30": "TPU 85A"}
GEOMS = [(None, None), ("d4.76x1.59", 3.0), ("d4.76x1.59", 1.5), ("d4.76x1.59", 5.0), ("d3.18x1.59", 3.0)]


def geom_name(magnet, depth):
    if magnet in (None, "none"):
        return "20261012-sr2_tip_whole"
    return f"20261012-sr2_tip_{magnet}_depth{depth:g}".replace(".", "p")


def sheet_name(recipe, magnet, depth):
    return f"20261012-sr2_tip_{recipe}_" + ("nomagnet" if magnet in (None, "none") else f"{magnet}_depth{depth:g}".replace(".", "p"))


def write_prints(plan=None):
    """One STL per geometry (the whole block, and the block with each magnet pocket) and one print sheet per tip of the plan,
    into docs/experiments/20261009-tpu_tip_fem/print/."""
    PRINT_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for mag, dep in GEOMS:
        m = TD.mesh(mag, dep)
        m.export(PRINT_DIR / f"{geom_name(mag, dep)}.stl")
        out.append((geom_name(mag, dep), len(m.faces), round(float(m.volume), 1), bool(m.is_watertight)))
    for p in plan or plan_rows():
        mag = None if p["magnet"] == "none" else p["magnet"]
        rec = dict(stl=f"{geom_name(mag, p['depth'])}.stl", material=TPU[p["recipe"]], role=p["role"],
                   slicer=dict(walls=2, line_width_mm=0.45, layer_mm=TD.LAYER, first_layer_mm=TD.FIRST_LAYER, top_layers=4,
                               bottom_layers=4, infill="gyroid", infill_percent=INFILL[p["recipe"]],
                               orientation="back face on the bed, palmar face up (layers across the load)", supports="none"))
        if mag is not None:
            rec.update(TD.print_sheet(mag, p["depth"]))
        r = p.get("row")
        if r:
            rec["model"] = dict(E_s_MPa=r["E_s"], pad_modulus_E_star_MPa=round(r["E_best_MPa"], 2),
                                approach_um={k: round(v["delta_um"], 1) for k, v in r["per_force"].items()},
                                twist_arm_mm={k: round(v["arm_mm"], 3) for k, v in r["per_force"].items()},
                                source="docs/experiments/20261009-tpu_tip_fem/magnet_runs.jsonl")
        json.dump(rec, open(PRINT_DIR / f"{sheet_name(p['recipe'], p['magnet'], p['depth'])}.json", "w"), indent=1)
    write_readme(plan or plan_rows())
    return out


def write_readme(plan):
    sh = TD.print_sheet(PLAN_MAG, PLAN_DEPTH)
    lines = ["# SR2 fingertips to print (2026-10-12)", "",
             "Written by `scripts/tpu_tip_print_plan.py stl`; the page that explains them is",
             "`docs/experiments/20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html`.", "",
             "Every tip is the whole block (17 x 14.8 x 21.4 mm, palmar edges rounded r 2.7 mm, end edge r 3.0 mm, back square),",
             "printed with its back face on the bed and the palmar face up: 2 walls of 0.45 mm, 4 top and 4 bottom layers of",
             "0.2 mm, gyroid infill, no supports.", "",
             "| Print | Material | Infill | STL | Sheet | Pad modulus E* (MPa) |", "|---|---|---|---|---|---|"]
    for p in plan:
        mag = None if p["magnet"] == "none" else p["magnet"]
        r = p.get("row")
        e = f"{r['E_best_MPa']:.1f}" if r else "pending"
        lines.append(f"| {RECIPE_TXT[p['recipe']]}{'' if mag else ', no magnet'} | {TPU[p['recipe']]} | {INFILL[p['recipe']]} % | "
                     f"{geom_name(mag, p['depth'])}.stl | {sheet_name(p['recipe'], p['magnet'], p['depth'])}.json | {e} |")
    lines += ["", f"Magnet: N52 disc {sh['magnet_diameter_mm']} x {sh['magnet_thickness_mm']} mm, axially magnetized, north toward the "
              f"palmar face, its top {sh['magnet_top_depth_mm']:g} mm under the contact point. Pocket {sh['pocket_diameter_mm']:.2f} mm "
              f"across and {sh['pocket_height_mm']:.2f} mm deep. Pause before the layer at {sh['pause_before_layer_at_mm']:g} mm "
              f"(layer {sh['pause_layer_number']} at 0.2 mm layers), press the magnet in, fix it with a drop of cyanoacrylate, resume.",
              "", "Other pocket depths (1.5 and 3 mm) and a 3.18 x 1.59 mm magnet are in the other STLs; their pause heights come from",
              "`scripts/tpu_tip_design.py sheet --magnet <name> --depth <mm>`.", "",
              "After printing: `scripts/tpu_tip_calibrate.py fit` turns an indentation and a pull test into E_s, E* and the friction",
              "coefficient, and prints the command that writes the pad scene at those values."]
    open(PRINT_DIR / "README.md", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "table"
    if cmd == "stl":
        for r in write_prints():
            print(*r)
    else:
        R = rows()
        for p in plan_rows(R):
            r = p["row"]
            if r is None:
                print(p["recipe"], p["magnet"], p["depth"], "not run yet")
                continue
            s = field(r, GAPS[0]) if p["magnet"] != "none" else {}
            print(p["recipe"], p["magnet"], p["depth"], f"E* {r['E_best_MPa']:.2f}", f"approach {r['per_force']['1']['delta_um']:.1f} um",
                  f"B0 {s.get('B0_mT', float('nan')):.2f} mT", f"S_n {s.get('S_normal_norm', float('nan')):.1f} uT/N",
                  f"sigF_n {s.get('sigF_normal_N', float('nan')):.3f} N")


# ------------------------------------------------------------------------------------------ page sections
# H is the page module (scripts/tpu_tip_fem_page.py) at run time: its figure/table helpers keep one numbering.

def _P():
    import contact_overview_page as P
    return P


PRINT_COL = {"w95_20": "var(--s-x)", "w85_50": "var(--s-sh)", "w85_30": "var(--s-res)", "w95_10": "var(--s-z)"}
DEPTHS = (1.5, 3.0, 5.0)


def chip_positions():
    return np.linspace(-6.0, 6.0, 49)


def svg_section(depth=PLAN_DEPTH, magnet=PLAN_MAG, gap=1.0):
    """Section of the whole tip at y = 0 in the print's orientation (palmar face up), with the magnet, its pocket's
    walls and skins, the gyroid core, the screwdriver and the magnetometer."""
    P = _P()
    W, H = 860, 450
    s = 15.0                                         # px per mm
    oz, oy = 262.0, 112.0                            # z = 0 and x = X_FACE in px
    X = lambda x: oy + (TD.X_FACE - x) * s           # noqa: E731
    Z = lambda z: oz + z * s                         # noqa: E731
    out = P._svg_open(W, H, "Section through the whole printed tip at its centre plane: palmar face up as printed, gyroid core, "
                            "perimeter walls and skins, the magnet in its pocket under the contact point, the screwdriver above and "
                            "the magnetometer below the back face")
    out.append('<defs><clipPath id="tipclip"><path d="' + _tip_path(X, Z) + '"/></clipPath>'
               '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
               '<path d="M0,0 L10,5 L0,10 Z" style="fill:var(--ink2)"/></marker></defs>')
    pk, mg = TD.pocket(magnet, depth)
    out.append(f'<path d="{_tip_path(X, Z)}" style="fill:#f4efe6;stroke:none"/>')
    a = 5.2
    k = 2 * math.pi / a
    lines = []
    for n in range(-8, 9):
        zz = np.linspace(-1, TD.Z_END + 1, 200)
        xx = -np.degrees(np.arctan(np.sin(k * zz))) / 360 * a + n * a / 2
        if xx.max() < TD.X_BACK - 1 or xx.min() > TD.X_FACE + 1:
            continue
        lines.append("M" + " L".join(f"{Z(z):.1f},{X(x):.1f}" for z, x in zip(zz, xx)))
    out.append(f'<g clip-path="url(#tipclip)"><path d="{" ".join(lines)}" style="fill:none;stroke:#cdb892;stroke-width:2.4"/></g>')
    wall, skin = 0.9, 0.8
    band = 'style="fill:#b9a27a;stroke:none"'
    out.append(f'<g clip-path="url(#tipclip)">'
               f'<rect x="{Z(0):.1f}" y="{X(TD.X_FACE):.1f}" width="{wall * s:.1f}" height="{17 * s:.1f}" {band}/>'
               f'<rect x="{Z(TD.Z_END - wall):.1f}" y="{X(TD.X_FACE):.1f}" width="{wall * s:.1f}" height="{17 * s:.1f}" {band}/>'
               f'<rect x="{Z(0):.1f}" y="{X(TD.X_FACE):.1f}" width="{TD.Z_END * s:.1f}" height="{skin * s:.1f}" {band}/>'
               f'<rect x="{Z(0):.1f}" y="{X(TD.X_BACK + skin):.1f}" width="{TD.Z_END * s:.1f}" height="{skin * s:.1f}" {band}/>'
               f'<rect x="{Z(TD.Z_C - pk["r"] - wall):.1f}" y="{X(pk["x_hi"] + skin):.1f}" width="{2 * (pk["r"] + wall) * s:.1f}" '
               f'height="{(pk["x_hi"] - pk["x_lo"] + 2 * skin) * s:.1f}" {band}/></g>')
    out.append(f'<rect x="{Z(TD.Z_C - pk["r"]):.1f}" y="{X(pk["x_hi"]):.1f}" width="{2 * pk["r"] * s:.1f}" '
               f'height="{(pk["x_hi"] - pk["x_lo"]) * s:.1f}" style="fill:#ffffff;stroke:none"/>')
    mh = 0.5 * (mg["x_hi"] + mg["x_lo"])
    out.append(f'<rect x="{Z(TD.Z_C - mg["r"]):.1f}" y="{X(mg["x_hi"]):.1f}" width="{2 * mg["r"] * s:.1f}" '
               f'height="{(mg["x_hi"] - mh) * s:.1f}" style="fill:#b23a3a"/>'
               f'<rect x="{Z(TD.Z_C - mg["r"]):.1f}" y="{X(mh):.1f}" width="{2 * mg["r"] * s:.1f}" height="{(mh - mg["x_lo"]) * s:.1f}" '
               f'style="fill:#3a5fb2"/>')
    out.append(f'<path d="{_tip_path(X, Z)}" style="fill:none;stroke:var(--ink);stroke-width:1.6"/>')
    cx, cy, R = Z(TD.Z_C), X(TD.X_FACE + 12.5), 12.5 * s
    th = np.linspace(math.radians(62), math.radians(118), 60)
    arc = "M" + " L".join(f"{cx + R * math.cos(t):.1f},{cy + R * math.sin(t):.1f}" for t in th)
    out.append(f'<path d="{arc} Z" style="fill:#d9dee3;stroke:var(--ink2);stroke-width:1.4"/>')
    y_p = X(pk["x_hi"])
    out.append(f'<line x1="{Z(-1.0):.1f}" x2="{Z(TD.Z_END + 1.0):.1f}" y1="{y_p:.1f}" y2="{y_p:.1f}" '
               f'style="stroke:#b23a3a;stroke-width:1.4;stroke-dasharray:6 4"/>')
    sh = TD.print_sheet(magnet, depth)
    yc = X(TD.X_BACK - gap)
    out.append(f'<rect x="{Z(TD.Z_C) - 22:.1f}" y="{yc:.1f}" width="44" height="13" rx="2" style="fill:#2b2f33"/>'
               f'<rect x="{Z(TD.Z_C) - 52:.1f}" y="{yc + 13:.1f}" width="104" height="6" style="fill:#3f7a46"/>')
    xd = Z(TD.Z_C - mg["r"] - 2.2)
    out.append(f'<line x1="{xd:.1f}" x2="{xd:.1f}" y1="{X(TD.X_FACE):.1f}" y2="{X(mg["x_hi"]):.1f}" style="stroke:var(--ink);stroke-width:1.2"/>'
               f'<line x1="{xd - 6:.1f}" x2="{xd + 6:.1f}" y1="{X(TD.X_FACE):.1f}" y2="{X(TD.X_FACE):.1f}" style="stroke:var(--ink);stroke-width:1.2"/>'
               f'<line x1="{xd - 6:.1f}" x2="{xd + 6:.1f}" y1="{X(mg["x_hi"]):.1f}" y2="{X(mg["x_hi"]):.1f}" style="stroke:var(--ink);stroke-width:1.2"/>'
               f'<text x="{xd - 9:.1f}" y="{0.5 * (X(TD.X_FACE) + X(mg["x_hi"])) + 5:.1f}" text-anchor="end" style="fill:var(--ink)">{depth:g} mm</text>')

    def lab(x0, y0, x1, y1, txt, anchor="start"):
        out.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" style="stroke:var(--ink3);stroke-width:1"/>'
                   f'<text x="{x1 + (5 if anchor == "start" else -5):.1f}" y="{y1 + 5:.1f}" text-anchor="{anchor}" '
                   f'style="fill:var(--ink)">{txt}</text>')
    right, left = Z(TD.Z_END) + 30, Z(0) - 36
    lab(Z(TD.Z_C + 2.6), X(TD.X_FACE + 0.7), right, 34, "Screwdriver, R 12.5 mm")
    lab(Z(TD.Z_END - 2.4), X(TD.X_FACE - 0.4), right, 92, "Palmar face, 4 layers")
    lab(Z(TD.Z_END - 0.45), X(3.0), right, 160, "Walls, 2 &#215; 0.45 mm")
    lab(Z(TD.Z_C + mg["r"]), X(mh), right, 214, "N52 magnet, north up")
    lab(Z(16.0), X(-2.6), right, 272, "Gyroid core, 20 %")
    lab(Z(TD.Z_C + 3.6), yc + 7, right, 400, "MLX90393 magnetometer")
    lab(Z(-0.2), y_p, left, y_p - 24, f"Pause at {sh['pause_before_layer_at_mm']:g} mm", "end")
    out.append(f'<text x="{left - 5:.1f}" y="{y_p - 1:.1f}" text-anchor="end" style="fill:var(--ink)">and insert</text>')
    lab(Z(-0.2), X(TD.X_BACK + 0.4), left, X(TD.X_BACK) + 26, "Back face on the bed", "end")
    xa = Z(0) - 20
    out.append(f'<line x1="{xa:.1f}" y1="{X(TD.X_BACK) - 6:.1f}" x2="{xa:.1f}" y2="{X(TD.X_FACE) + 60:.1f}" '
               f'style="stroke:var(--ink2);stroke-width:1.6;marker-end:url(#arrow)"/>'
               f'<text x="{xa - 8:.1f}" y="{X(-1.0):.1f}" text-anchor="end" style="fill:var(--ink2)">Layers</text>'
               f'<text x="{xa - 8:.1f}" y="{X(-1.0) + 19:.1f}" text-anchor="end" style="fill:var(--ink2)">stack up</text>')
    out.append(f'<text x="{Z(0):.1f}" y="{X(TD.X_BACK) + 52:.1f}" style="fill:var(--ink3)">Mount side</text>'
               f'<text x="{Z(TD.Z_END):.1f}" y="{X(TD.X_BACK) + 52:.1f}" text-anchor="end" style="fill:var(--ink3)">End</text>')
    out.append("</svg>")
    return "".join(out)


def _tip_path(X, Z):
    """Outline of the section y = 0: square except the palmar end edge (r R_END)."""
    r = TD.R_END
    pts = [(TD.X_BACK, 0.0), (TD.X_FACE, 0.0), (TD.X_FACE, TD.Z_END - r)]
    for t in np.linspace(0, math.pi / 2, 24)[1:]:
        pts.append((TD.X_FACE - r + r * math.cos(t), TD.Z_END - r + r * math.sin(t)))
    pts += [(TD.X_BACK, TD.Z_END)]
    return "M" + " L".join(f"{Z(z):.1f},{X(x):.1f}" for x, z in pts) + " Z"


def chip_curve(row, gaps=None):
    """Normal-force noise (N, 1 sigma at RATE_HZ) and static field (mT) against the chip's distance behind the back face
    (mm; negative: inside the tip's back)."""
    gaps = chip_positions() if gaps is None else gaps
    out = []
    for g in gaps:
        f = field(row, float(g))
        out.append((float(g), f["sigF_normal_N"], f["sigF_shear_y_N"], f["B0_mT"]))
    return out


def svg_results(R):
    P = _P()
    W, H = 860, 380
    out = P._svg_open(W, H, "Magnet displacement under 1 N against the magnet's depth for four prints, and the normal-force noise "
                            "of the magnetometer against its distance behind the back face")
    # (a) displacement under 1 N normal load against depth
    fx, fy = P._panel(out, 80, 54, 300, 250, (1.0, 5.5), (0, 50), [1.5, 3, 5], [0, 10, 20, 30, 40, 50],
                      "Magnet depth below the face (mm)", "Magnet travel under 1 N (&#181;m)")
    out.append('<text x="20" y="22" style="fill:var(--ink)">(a) Normal load at the contact</text>')
    for rec, col in PRINT_COL.items():
        pts = [(d, -R[(rec, "d4.76x1.59", d)]["per_force"]["1"]["magnet_t_um"][0]) for d in DEPTHS if (rec, "d4.76x1.59", d) in R]
        if len(pts) > 1:
            P._path(out, fx, fy, pts, col, width=2.2)
        for d, v in pts:
            P._marker(out, fx(d), fy(v), col)
        r = R.get((rec, "d3.18x1.59", 3.0))
        if r:
            P._marker(out, fx(3.0), fy(-r["per_force"]["1"]["magnet_t_um"][0]), col, hollow=True)
    # (b) force noise against the chip's distance behind the back face, 4.76 mm magnet at 3 mm
    gx, gy = P._panel(out, 520, 54, 300, 250, (-6, 6), (0.03, 3), [-6, -3, 0, 3, 6], [0.03, 0.1, 0.3, 1, 3],
                      "Chip behind the back face (mm)", "Normal-force noise, 1&#963; (N)", logy=True)
    out.append(f'<text x="460" y="22" style="fill:var(--ink)">(b) Magnetometer at 100 Hz, magnet {PLAN_DEPTH:g} mm deep</text>')
    sat = None
    for rec, col in PRINT_COL.items():
        r = R.get((rec, PLAN_MAG, PLAN_DEPTH))
        if not r:
            continue
        cur = chip_curve(r)
        cur = [c for c in cur if 0.03 <= c[1] <= 3.0]
        ok = [(g, sf) for g, sf, _, b in cur if b <= B_RANGE_MT]
        bad = [(g, sf) for g, sf, _, b in cur if b > B_RANGE_MT]
        P._path(out, gx, gy, ok, col, width=2.2)
        if bad:
            P._path(out, gx, gy, bad + ok[:1], col, dashed=True, width=1.6)
            sat = max(g for g, _ in bad) if sat is None else max(sat, max(g for g, _ in bad))
    if sat is not None:
        out.append(f'<line x1="{gx(sat):.1f}" x2="{gx(sat):.1f}" y1="{gy(3):.1f}" y2="{gy(0.03):.1f}" '
                   f'style="stroke:var(--ink3);stroke-dasharray:3 4"/>'
                   f'<text x="{gx(sat) - 6:.1f}" y="{gy(1.6):.1f}" text-anchor="end" style="fill:var(--ink3)">Over 4.8 mT</text>')
    out.append("</svg>")
    legend = [(RECIPE_LBL[k], c, False, "circle") for k, c in PRINT_COL.items()] + \
             [("3.18 mm magnet", "var(--ink2)", False, None)]
    return "".join(out), [(RECIPE_LBL[k], c, False, "circle") for k, c in PRINT_COL.items()]


def arm_err(H, X, row):
    a = row["per_force"]["3"]["arm_mm"]
    pa = X.pads_arm(row["E_best_MPa"])
    return (pa / a - 1) * 100 if pa else None, pa, a


SHORT = {"w95_20": "95A&#160;20&#8202;%", "w95_10": "95A&#160;10&#8202;%", "w85_50": "85A&#160;50&#8202;%", "w85_30": "85A&#160;30&#8202;%"}


def plan_section(H, X):
    R = rows()
    body = []
    for p in plan_rows(R):
        r, mag = p["row"], p["magnet"]
        gname, sname = geom_name(mag, p["depth"]), sheet_name(p["recipe"], mag, p["depth"])
        tip = SHORT[p["recipe"]] + (",<br>no magnet" if mag == "none" else "")
        files = f"<a href='print/{gname}.stl'>STL</a>, <a href='print/{sname}.json'>sheet</a>"
        if r is None:
            body.append([tip] + ["running"] * 4 + [files])
            continue
        ae, pa, a = arm_err(H, X, r)
        f = field(r, GAPS[0]) if mag != "none" else None
        body.append([tip, H.f1(r["E_best_MPa"]), H.f1(r["per_force"]["1"]["delta_um"]), f"{H.f2(a)} / {H.f2(pa)}",
                     H.f2(f["sigF_normal_N"]) if f else "&#8211;", files])
    sh = TD.print_sheet(PLAN_MAG, PLAN_DEPTH)
    return H.table(["Tip", "\\(E^*\\)<br>(MPa)", "Approach<br>at 1&#8202;N (&#181;m)", "Twist arm at 3&#8202;N,<br>tip / pads (mm)",
                    "Force<br>noise (N)", "Files"], body,
                   caption="Tips to print. Whole blocks printed palmar face up with two 0.45&#8202;mm walls, four 0.2&#8202;mm top and bottom "
                           "layers and gyroid infill; an N52 magnet 4.76 &#215; 1.59&#8202;mm, north toward the face, its top "
                           f"{PLAN_DEPTH:g}&#8202;mm under the "
                           f"contact point, pressed in at a pause before the layer at {sh['pause_before_layer_at_mm']:g}&#8202;mm "
                           "(<code>magnet_runs.jsonl</code>; files in <code>print/</code>).",
                   note="\\(E^*\\): the pads&#8217; modulus that best reproduces the tip&#8217;s approach under the screwdriver at 0.5, 1 and "
                        "3&#8202;N, at \\(E_s\\) = 30&#8202;MPa for 95A and 12&#8202;MPa for 85A. Twist arm: the tip&#8217;s full-slip arm against the "
                        "pads&#8217; at \\(E^*\\) (bed task T2). Force noise: one standard deviation of the normal force read from one "
                        "100&#8202;Hz sample of an MLX90393 1&#8202;mm behind the back face.", text_cols=(0, 5), key="plan")


def plan_text(H, X):
    R = rows()
    g0, gp = R.get(("w95_20", "none", None)), R.get(("w95_20", PLAN_MAG, PLAN_DEPTH))
    sp = R.get(("w85_30", PLAN_MAG, PLAN_DEPTH))
    if not (g0 and gp):
        return "<p>Runs in progress.</p>"
    fp = field(gp, GAPS[0])
    txt = ("<p>Five tips cover the choice (Table&#160;[[tab:plan]]): 95A at 20&#8202;% gyroid, the reference, which prints most "
           "easily; 85A at 50&#8202;%, as stiff, whose twist arm the pads reproduce best; 85A at 30&#8202;% and 95A at 10&#8202;%, softer, for a "
           "larger magnet signal; and the reference without a magnet, which separates the magnet&#8217;s effect on the contact. "
           f"Without its mount slot the whole 95A tip at 20&#8202;% infill has a pad modulus of {H.f1(g0['E_best_MPa'])}&#8202;MPa, the "
           f"insert&#8217;s 13.0 within 3&#8202;%. The magnet&#8217;s top lies {PLAN_DEPTH:g}&#8202;mm under the face, the depth that changes the "
           f"contact least and gives the least force noise of the three tried: it raises the reference&#8217;s modulus to "
           f"{H.f1(gp['E_best_MPa'])}&#8202;MPa, so a scene of a magnet tip takes the magnet tip&#8217;s modulus, and an MLX90393 1&#8202;mm "
           f"behind the back face reads the magnet with {H.f1(fp['S_normal_norm'])}&#8202;&#181;T/N, a force noise of "
           f"{H.f2(fp['sigF_normal_N'])}&#8202;N per 100&#8202;Hz sample.")
    if sp:
        fs = field(sp, GAPS[0])
        txt += (f" The softer 85A print at 30&#8202;% has a modulus of {H.f1(sp['E_best_MPa'])}&#8202;MPa and a noise of "
                f"{H.f2(fs['sigF_normal_N'])}&#8202;N.")
    return txt + "</p>"


def magnet_table(H, X, R):
    body = []
    for rec in PRINT_COL:
        for mag, dep in (("d4.76x1.59", 1.5), ("d4.76x1.59", 3.0), ("d4.76x1.59", 5.0), ("d3.18x1.59", 3.0)):
            r = R.get((rec, mag, dep))
            if r is None:
                continue
            lab = SHORT[rec] + f", {dep:g}&#8202;mm" + (",<br>3.18&#8202;mm magnet" if mag == "d3.18x1.59" else "")
            f = field(r, GAPS[0])
            body.append([lab, H.f1(r["E_best_MPa"]), H.f1(-r["per_force"]["1"]["magnet_t_um"][0]),
                         f"{H.f1(r['shear_z_magnet_t_um_per_N'][2])} / {H.f1(r['shear_y_magnet_t_um_per_N'][1])}",
                         H.f1(f["S_normal_norm"]), f"{H.f2(f['sigF_normal_N'])} / {H.f2(f['sigF_shear_y_N'])}"])
    return H.table(["Tip, magnet depth", "\\(E^*\\)<br>(MPa)", "Travel<br>at 1&#8202;N (&#181;m)", "Shear travel,<br>along / across (&#181;m/N)",
                    "Signal<br>(&#181;T/N)", "Force noise,<br>normal / across (N)"], body,
                   caption="Magnet tips under the screwdriver: 4.76 &#215; 1.59&#8202;mm magnets unless marked (<code>magnet_runs.jsonl</code>, "
                           "<code>scripts/tpu_tip_magnet.py</code>).",
                   note="Travel: of the magnet&#8217;s centre toward the back face under 1&#8202;N on the face; shear travel: under 1&#8202;N of shear "
                        "on a 1&#8202;N contact that sticks, along the finger and across it. Signal and force noise: an MLX90393 1&#8202;mm behind "
                        "the back face under the contact point, one 100&#8202;Hz sample (2.3&#8202;&#181;T RMS on its in-plane axes, "
                        "3.5&#8202;&#181;T on its normal axis).", key="magnet")


def magnet_section(H, X):
    R = rows()
    gp = R.get(("w95_20", PLAN_MAG, PLAN_DEPTH))
    g0 = R.get(("w95_20", "none", None))
    out = [H.figure(svg_section(), "Section through the whole tip at its centre plane, as printed: back face on the bed, palmar "
                                   "face up. Two perimeter walls and four solid layers enclose the gyroid core and the magnet&#8217;s pocket; "
                                   "the print pauses at the dashed line for the magnet, north toward the face; the magnetometer sits "
                                   "1&#8202;mm behind the back face under the contact point. True scale; the gyroid&#8217;s beads are drawn at "
                                   "the 20&#8202;% period of 5.2&#8202;mm.", key="magsec")]
    if not (gp and g0):
        return "".join(out) + "<p>Runs in progress.</p>"
    sc = [R[(k, PLAN_MAG, PLAN_DEPTH)] for k in PRINT_COL if (k, PLAN_MAG, PLAN_DEPTH) in R]
    trav = [-r["per_force"]["1"]["magnet_t_um"][0] for r in sc]
    fp, fp4 = field(gp, GAPS[0]), field(gp, GAPS[1])
    dep = []
    for d in DEPTHS:
        r = R.get(("w95_20", PLAN_MAG, d))
        if r:
            dep.append((d, (r["E_best_MPa"] / g0["E_best_MPa"] - 1) * 100, field(r, GAPS[0])["sigF_normal_N"]))
    rng = lambda v, f: f(min(v)) if f(min(v)) == f(max(v)) else f"{f(min(v))}&#8211;{f(max(v))}"  # noqa: E731
    deps = (f"with the magnet at {', '.join(f'{d:g}' for d, _, _ in dep[:-1])} and {dep[-1][0]:g}&#8202;mm the pad modulus rises by "
            f"{', '.join(H.f0(e) for _, e, _ in dep[:-1])} and {H.f0(dep[-1][1])}&#8202;% and the force noise is "
            f"{', '.join(H.f2(n) for _, _, n in dep[:-1])} and {H.f2(dep[-1][2])}&#8202;N")
    txt = ("<p>eFlesh embeds N52 disc magnets in a printed microstructure through lip-sealed pockets, closed after a print pause, "
           "and reads them with the MLX90393 boards of ReSkin and AnySkin; its fingertip magnets are 3/16 &#215; 1/16&#8202;in "
           "(4.76 &#215; 1.59&#8202;mm), and its cut cells are 0.001&#8211;0.002 times the filament&#8217;s modulus. The gyroid core of the tips "
           "here is 0.03&#8211;0.23 times it (Table&#160;[[tab:cells]]), so the same magnet moves tens of micrometres per newton, not "
           f"hundreds. With its top {PLAN_DEPTH:g}&#8202;mm under the face it moves {rng(trav, H.f1)}&#8202;&#181;m toward "
           "the back face under 1&#8202;N across the four prints, and an MLX90393 1&#8202;mm behind the 95A 20&#8202;% tip reads it with a noise "
           f"of {H.f2(fp['sigF_normal_N'])}&#8202;N per 100&#8202;Hz sample, {H.f2(fp4['sigF_normal_N'])}&#8202;N with the chip behind a "
           "3&#8202;mm plate (Figure&#160;[[fig:magres]], Table&#160;[[tab:magnet]]); averaging ten samples divides the noise by 3.2. "
           "Shear along the finger moves the magnet a third as far as shear across it, because the mount face holds the block at "
           "one end.</p>"
           "<p>The magnet stiffens the face above it, and a deeper magnet sits closer to the chip, so depth helps both: on the "
           f"95A 20&#8202;% tip, {deps}. A chip set into the back of the tip raises "
           "the signal further until the static field passes the 4.8&#8202;mT that the MLX90393 reads at its highest gain "
           "(Figure&#160;[[fig:magres]]b).</p>")
    out.append(txt)
    svg, leg = svg_results(R)
    out.append(H.figure(svg, "(a) Travel of a 4.76 &#215; 1.59&#8202;mm magnet toward the back face under 1&#8202;N on the face, against the "
                             "depth of its top below the face (filled), and of a 3.18 &#215; 1.59&#8202;mm magnet at 3&#8202;mm (hollow). "
                             "(b) Normal-force noise of one 100&#8202;Hz MLX90393 sample against the chip&#8217;s distance behind the back "
                             f"face (negative: set into the tip), magnet {PLAN_DEPTH:g}&#8202;mm deep; dashed where the static field "
                             "exceeds 4.8&#8202;mT.", legend=leg, key="magres"))
    out.append(magnet_table(H, X, R))
    return "".join(out)


def gap_section(H, X):
    R = rows()
    g0, g3 = R.get(("w95_20", "none", None)), R.get(("w95_20", PLAN_MAG, PLAN_DEPTH))
    T = X.T
    errs = [T[n]["E_best_max_err_pct"] for n in T]
    ae = [arm_err(H, X, p["row"])[0] for p in plan_rows(R) if p["row"]]
    ae = [-v for v in ae if v is not None]
    arm_txt = (f"{H.f0(min(ae))}&#8211;{H.f0(max(ae))}" if ae else "4&#8211;19")
    face_z = g3["shear_z_face_um_per_N"] if g3 else None
    face_y = g3["shear_y_face_um_per_N"] if g3 else None
    dE = (g3["E_best_MPa"] / g0["E_best_MPa"] - 1) * 100 if g0 and g3 else None
    body = [
        ["Solid modulus \\(E_s\\)", "Contact stiffness, \\(E^*\\), magnet travel",
         "26&#8211;67&#8202;MPa on 95A datasheets: a factor of 2.6 in stiffness, \\(\\pm\\)12&#8202;% in the pads&#8217; twist arm "
         "(\\(\\propto E^{-1/4}\\))", "Indentation or the magnet&#8217;s field under known loads"],
        ["Friction coefficient", "Slip force, twist torque", "Scenes use 1.0 on the TPU tips; not measured on a print",
         "Pull test: slip force over \\(2N\\)"],
        ["One modulus for a line contact", "Approach over 0.5&#8211;3&#8202;N",
         f"{H.f0(min(errs))}&#8211;{H.f0(max(errs))}&#8202;%: the tip&#8217;s force grows as \\(\\delta^{{1.07}}\\), the pads&#8217; as "
         "\\(\\delta^{1.5}\\)", "None within the pads"],
        ["Load on the walls", "Twist arm at 3&#8202;N", f"Pads {arm_txt}&#8202;% shorter than the tip on the plan&#8217;s tips "
         "(inserts: Table&#160;[[tab:recipes]])",
         "Spin onset at 1 and 3&#8202;N"],
        ["Magnet", "Pad modulus over the magnet", f"+{H.f0(dE)}&#8202;% at {PLAN_DEPTH:g}&#8202;mm on 95A 20&#8202;%" if dE is not None else "&#8211;",
         "The magnet tip&#8217;s own \\(E^*\\)"],
        ["Tangential compliance", "Tool travel before slip",
         (f"The tip deflects {H.f0(face_z)}&#8202;&#181;m/N along the finger and {H.f0(face_y)}&#8202;&#181;m/N across it at the contact; "
          "the pads&#8217; fingertip is rigid") if face_z else "&#8211;", "None within the pads at the trainer&#8217;s impratio"],
        ["Printed infill", "Stiffness", "Solid fraction 4&#8211;12&#8202;% above the slicer&#8217;s; gyroid position under the contact "
         "1.7&#8202;% at 20&#8202;%, 7&#8202;% at 5&#8202;%", "Indentation of each print"],
        ["Large deformation", "Force at large approach", "&#8722;6&#8202;% at 200&#8202;&#181;m, &#8722;11&#8202;% at 350&#8202;&#181;m (5&#8202;% tip); "
         "the plan&#8217;s tips stay under 130&#8202;&#181;m at 3&#8202;N", "&#8211;"],
        ["TPU rate dependence", "Approach under a held load", "Not modelled", "Indent at two speeds, hold 10&#8202;s"],
        ["Scene setting", "Every contact model so far", "Pads at 10&#8202;MPa against \\(E^*\\) of 6&#8211;17&#8202;MPa for these prints",
         "<code>make_pad_morphology_run.py --E</code>"],
    ]
    tab = H.table(["Source", "What it changes", "Size", "Measurement that closes it"], body,
                  caption="Differences between a printed tip and its simulation, largest first.", text_cols=(0, 1, 2, 3), key="gap")
    txt = ("<p>Before any measurement, the stiffness of a printed tip is known within the factor of 2.6 that separates the 95A "
           "datasheets, and its friction coefficient not at all; the model takes \\(E_s\\) = 30&#8202;MPa for 95A and the scenes "
           "\\(\\mu\\) = 1.0. One indentation and one pull "
           "test on the first print fix both (next section). The model error that remains is the single modulus&#8217;s "
           f"{H.f0(min(errs))}&#8211;{H.f0(max(errs))}&#8202;% in approach over 0.5&#8211;3&#8202;N, the twist arm, which the pads give "
           f"{arm_txt}&#8202;% short because the walls carry the face&#8217;s edges, and the tip&#8217;s own tangential compliance, which no "
           "pad model in the trainer represents (Table&#160;[[tab:gap]]).</p>")
    return txt + tab


def calib_section(H, X):
    items = [
        ("Indentation.", "Press the screwdriver shaft, or a 25&#8202;mm rod, across the palmar face at the contact point to 0.5, 1 and "
         "3&#8202;N and record the approach from first touch; the model&#8217;s approaches for each print are in "
         "<code>scripts/tpu_tip_calibrate.py predict</code>. The ratio of predicted to measured approach scales \\(E_s\\), and "
         "\\(E^*\\) with it."),
        ("Field under known loads.", "With the magnetometer in place, load the face with weights at the contact point and record "
         "the field change; the model&#8217;s signal per newton scales as \\(1/E_s\\) and gives a second estimate."),
        ("Pull.", "Pinch the tool between two printed tips at a set force \\(N\\) per tip and pull it along its axis until it slips: "
         "\\(\\mu = F/2N\\)."),
        ("Twist.", "Twist the tool in the same pinch at 1 and 3&#8202;N per tip: the onset torque over \\(2\\mu N\\) is the twist arm, to "
         "compare with the tip&#8217;s and the pads&#8217; arms in the plan."),
    ]
    lst = "<ol>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ol>"
    return ("<p>Each print maps to one pad modulus \\(E^*\\), fitted to its approach under the screwdriver with the datasheet "
            "\\(E_s\\) (Table&#160;[[tab:law]] for the inserts, Table&#160;[[tab:plan]] for the whole tips). Nothing else is calibrated: \\(E_s\\), the "
            "friction coefficient and the rate dependence come from no measurement of a print. Four bench measurements on each "
            "printed tip close that:</p>" + lst +
            "<p><code>scripts/tpu_tip_calibrate.py fit --tip w95_20 --magnet d4.76x1.59@5 --indent 0.5:&#8230; 1:&#8230; 3:&#8230; "
            "--pull N:F</code> returns \\(E_s\\), \\(E^*\\) and \\(\\mu\\) and the command that writes the pad scene at those values "
            "(<code>scripts/make_pad_morphology_run.py --E --mu</code>); <code>--save</code> keeps the result beside the print "
            "files.</p>")
