#!/usr/bin/env python3
"""Page on the layout of the indenter plate for the UR5 calibration cell (owner, 2026-10-28: "the big ass indenter plate i
printed wont actually work though, since the big sphere is in the way of the side-to-side tests ... spec out a plan so that
each indenter and motion per indenter has enough space? and that they all still fit on the bambu print plate"). Reads the
rows that scripts/indenter_plate_layout.py wrote to docs/experiments/20261028-indenter_plate_layout/data/ (replay, reach),
its renders and STL, draws the figures into media/ and writes
docs/experiments/20261028-indenter_plate_layout/20261028-indenter_plate_layout.html.

    .venv/bin/python scripts/indenter_plate_layout_page.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import ur5_tip_calibration_page as P  # noqa: E402   (esc, fig, film, table, plt_setup, TEX_RE)

D = ROOT / "docs/experiments/20261028-indenter_plate_layout"
DATA, MEDIA, CAD = D / "data", D / "media", D / "cad"
STEM = "20261028-indenter_plate_layout"
PAGE = D / f"{STEM}.html"
TEMPLATE = ROOT / "scripts/indenter_plate_layout_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
EXP = ROOT / "docs/experiments"
L = dict(array=EXP / "20261027-indenter_array/README.md", tall=EXP / "20261026-tall_indenter/README.md",
         holder=EXP / "20261027-fingertip_holder/README.md", queue=EXP / "20261022-ur5_tip_calibration/QUEUE.md",
         v3=EXP / "20261027-v3_tip_calibration_and_crosstalk/README.md",
         status=EXP / "20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html",
         overview=ROOT / "docs/overviews/printed_fingertips.html")
FINAL = "v2d"
SUITES = ("sphere5", "tall5", "sphere12", "rod", "facing", "side", "keys")
SUITE_NAME = dict(sphere5="Sphere suite on the 5 mm sphere", tall5="Tall suite on the 5 mm sphere",
                  sphere12="Sphere suite on the 12.5 mm sphere", rod="Rod suite on the half rod",
                  facing="Facing hovers over the pad-up tip", side="Side-by-side hovers beside the pad-down tip",
                  keys="Homing on the three keys", tall12="Tall suite on the 12.5 mm sphere")
PART_NAME = {"sphere5": "5 mm sphere", "sphere12": "12.5 mm sphere", "rod": "half rod", "holder column": "holder's column",
             "holder -y down block": "pad-down station", "holder +y up block": "pad-up station", "held tips": "held tips",
             "base": "base", "2x4s, table": "2x4s or table", "key 1": "key 1", "key 2": "key 2", "key 3": "key 3"}
GROUP_NAME = dict(robot="wrist", hex_e="HEX-E", mount="mount", finger="finger", pad="pad")


def link(p, text):
    return f'<a href="{P.esc(os.path.relpath(p, D))}">{text}</a>'


def code(s):
    return f"<code>{P.esc(s)}</code>"


def run(layout, suite):
    f = DATA / f"{STEM}_{layout}_{suite}_quick.json"
    return json.loads(f.read_text()) if f.exists() else None


def causes(d):
    """{(group, part) or kind: count} of a run's flags; joint flags as 'joints'."""
    c = {}
    for r in d["rows"]:
        seen = set()
        for w in r["flags"]:
            if " mm to the " in w:
                g, part = w.split()[0], w.split(" mm to the ")[1]
                key = f"{GROUP_NAME.get(g, g)} near the {PART_NAME.get(part, part)}"
            else:
                key = "joints outside their windows"
            if key not in seen:
                c[key] = c.get(key, 0) + 1
                seen.add(key)
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def cause_text(d, k=3):
    c = causes(d)
    if not c:
        return "none"
    return "; ".join(f"{v} {key}" for key, v in list(c.items())[:k])


# ------------------------------------------------------------------------------------------------ figures
def fig_reach(out, final):
    """Joint reach of the four suites whose joints depend on where their indenter stands: the share of actions inside
    every window (joint 6 past +45 deg counted with the owner present) over the plate, at one height."""
    plt = P.plt_setup()
    import matplotlib.colors as mc
    panels = (("tall5", 160.0, "sphere5"), ("sphere5", 160.0, "sphere5"), ("side", 153.0, "holder"), ("facing", 153.0, "holder"))
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 10.5), constrained_layout=True)
    axes = axs.ravel()
    for ax, (s, h, ind) in zip(axes, panels):
        d = json.loads((DATA / f"{STEM}_reach_{s}.json").read_text())
        rows = [r for r in d["rows"] if abs(r["h"] - h) < 1e-6] or d["rows"]
        xs = sorted({r["x"] for r in rows})
        ys = sorted({r["y"] for r in rows})
        M = np.full((len(ys), len(xs)), np.nan)
        for r in rows:
            M[ys.index(r["y"]), xs.index(r["x"])] = (r["ok"] + r["attended"]) / r["n"]
        im = ax.imshow(M, origin="lower", cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto",
                       extent=(xs[0] - 20, xs[-1] + 20, ys[0] - 25, ys[-1] + 25))
        for r in rows:
            ax.text(r["x"], r["y"], f"{r['fail']}", ha="center", va="center", fontsize=14,
                    color="w" if (r["ok"] + r["attended"]) / r["n"] < 0.8 else "k")
        c = final["g"][ind]["c"]
        ax.plot(c[0], c[1], marker="*", ms=18, mec="k", mfc="#e8a35c", zorder=5)
        ax.set_title(f"{SUITE_NAME[s]}\n{rows[0]['n']} actions", fontsize=15)
        ax.set_xlabel("x (mm), away from the robot")
        ax.set_ylabel("y (mm)")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, location="bottom")
    cb.set_label("share of actions inside the joint windows", fontsize=14)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_renders(out, layout):
    """The renders of a layout in a grid, each under a label bar."""
    from PIL import Image, ImageDraw, ImageFont
    picks = [("side", "plate"), ("side", "plate_top"), ("facing", "plate"), ("tall5", "plate"), ("sphere12", "plate"),
             ("rod", "plate")]
    tiles = []
    for s, cam in picks:
        f = MEDIA / f"{STEM}_{layout}_{s}_{cam}.png"
        if not f.exists():
            continue
        im = Image.open(f).convert("RGB")
        im = im.resize((720, round(im.height * 720 / im.width)), Image.LANCZOS)
        bar = Image.new("RGB", (im.width, 44), (255, 255, 255))
        dr = ImageDraw.Draw(bar)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 28)
        except OSError:
            font = ImageFont.load_default()
        label = dict(side="Side by side, g 15 mm", facing="Facing, W 2 mm", tall5="Tall suite, flank press",
                     sphere12="12.5 mm sphere, press", rod="Half rod, press")[s] + (", from above" if cam == "plate_top" else "")
        dr.text((10, 7), label, fill=(0, 0, 0), font=font)
        t = Image.new("RGB", (im.width, im.height + 44), (255, 255, 255))
        t.paste(bar, (0, 0))
        t.paste(im, (0, 44))
        tiles.append(t)
    if not tiles:
        return False
    nc = 2
    nr = (len(tiles) + nc - 1) // nc
    w, h = tiles[0].size
    grid = Image.new("RGB", (nc * w + (nc - 1) * 8, nr * h + (nr - 1) * 8), (255, 255, 255))
    for k, t in enumerate(tiles):
        grid.paste(t, ((k % nc) * (w + 8), (k // nc) * (h + 8)))
    grid.save(out)
    return True


# ------------------------------------------------------------------------------------------------ text
def suite_table(layouts, n):
    head = ["Suite", "Actions"] + [f"Clear on {lab}" for _, lab in layouts] + [f"What stops the rest on {layouts[-1][1]}"]
    body = []
    for s in SUITES:
        ds = [run(lay, s) for lay, _ in layouts]
        if any(d is None for d in ds):
            continue
        cells = [SUITE_NAME[s], str(ds[0]["n"])]
        for d in ds:
            cells.append(f"{d['n'] - d['flagged']}" + (f" ({d['attended']} with the owner)" if d.get("attended") else ""))
        cells.append(P.esc(cause_text(ds[-1])))
        body.append(cells)
    return P.table(head, body, num_cols=(1, 2, 3), n=n,
                   caption="Actions of each suite that clear every margin and joint window on each layout. Joint 6 past "
                           "+45 deg counts as clear with the owner present, as the morning plans run today.")


def spec_table(g, n):
    a = g["params"]
    rows = [["Base", f"{a['base'][1] - a['base'][0]:g} x {a['base'][3] - a['base'][2]:g} mm, {a['base'][4]:g} mm thick",
             "x &#8722;110 to 110, y &#8722;118 to 118", "six wood screws into the 2x4s"]]
    for i, k in enumerate(g["keys"]):
        rows.append([f"Key {i + 1}", f"yaw {k['yaw']:+g}&#176;", f"x {k['c'][0]:g}, y {k['c'][1]:g}",
                     f"top {k['pin_top']:.1f} mm; mount v3's wedge key"])
    s5, s12, r, h = g["sphere5"], g["sphere12"], g["rod"], g["holder"]
    rows.append(["5 mm sphere", f"R {s5['R']:g} mm, top {s5['top']:g} mm", f"x {s5['c'][0]:g}, y {s5['c'][1]:g}",
                 f"neck r {s5['neck_r']:g} mm, 15&#176; cone to r {s5['col_r']:g} mm"])
    rows.append(["12.5 mm sphere", f"R {s12['R']:g} mm, top {s12['top']:g} mm", f"x {s12['c'][0]:g}, y {s12['c'][1]:g}",
                 f"neck r {s12['neck_r']:g} mm, 15&#176; cone to r {s12['col_r']:g} mm"])
    rows.append(["Half rod", f"R {r['R']:g} mm, 20 mm long along y, top {r['top']:g} mm", f"x {r['c'][0]:g}, y {r['c'][1]:g}",
                 f"on a 25 x 20 mm block, cone to r {r['col_r']:g} mm"])
    ap = h["apex"]
    rows.append(["Fingertip holder", f"column r {h['col_r']:g} mm, blocks' undersides {h['z_bot']:g} mm",
                 f"x {h['c'][0]:g}, y {h['c'][1]:g}",
                 f"&#8722;y station pad-down (apex {ap[0][2]:.1f} mm), +y pad-up (apex {ap[1][2]:.1f} mm)"])
    return P.table(["Part", "Size", "Centre in A (mm)", "Notes"], rows, n=n,
                   caption="The parts of the proposed plate (frame A: x away from the robot, z up from the plate's "
                           "underside, origin at the base's centre).")


def sections(N):
    A, F = N["array"], N["final"]
    gF = N["gF"]
    S = []
    terms = r"""<dl class="terms">
<dt>Frame A</dt><dd>The plate's frame. x points away from the robot, along the tool's axis when the mount sits on key 2.
z points up from the plate's underside, and the origin is the base's centre. The checks put A where the low sphere's frame
stood at v3 ID 1's sphere registration of 2026-10-27, on the 2x4s. The robot's base stands at about (&#8722;600, 115) mm
in A.</dd>
<dt>Suite</dt><dd>The lab's plans for one indenter, as v3 ID 1 runs them today (Table 1). Each plan's indenter frame moves
to its indenter on the plate by a translation. On the 12.5 mm sphere the plans' radius becomes 12.5 mm, with the top
kept.</dd>
<dt>Replay</dt><dd>The moves of <code>scripts/ur5_tip_calibration_holder_paths.py</code>. They are the travel between
actions, the approach, the depth at the largest force, and the turns of slides, rolls and twists in 0.5&#176; steps.
Touch-offs stop at first contact. The runs on this page check each action's own moves at the end of every straight step,
and the travel's joint moves at their ends (<code>--quick</code>).</dd>
<dt>Clear</dt><dd>An action clears when every checked pose keeps the joints and the clearances below. Joint 5 stays in 30
to 100&#176; and joint 6 in &#8722;65 to +60&#176;. $|\sin q_5|$ stays at 0.5 or more, the elbow 20&#176; from straight,
and the wrist 150 mm from joint 1's axis. Every joint stays 10&#176; from its limit. The clearances to the plate, the held
tips, the 2x4s and the table are 8 mm for the robot's links and 6 mm for the HEX-E. They are 3 mm for the mount (2.5 mm for
touch-offs), 0.8 mm for the finger and 0.3 mm for the pad. The pad pressing its own indenter's tip is left out, because it
deforms there. Joint 6 past +45&#176; counts as clear with the owner present, as for today's morning plans.</dd>
<dt>Sweep and free height</dt><dd>A suite's sweep is the lowest point of the robot's links and the tool over each 2 mm
square of the plan view, over every checked pose. A spot's free height is the lowest sweep of the other suites over a disc
around it. An indenter standing there must stay 8 mm under it, the robot's margin.</dd>
<dt>Reach map</dt><dd>The joints alone, without clearances, with a suite's indenter moved over a grid of positions. Each
cell gives the share of the suite's actions inside every joint window.</dd>
</dl>"""
    S.append(("terms", "Terms and metrics", terms))

    rows = [["5 mm sphere", "sphere suite", "pass 1 touch-offs; stage 2 grid presses, slides, twists, rolls; the morning plan",
             "337"],
            ["5 mm sphere", "tall suite", "pass 1 touch-offs; edge and flank presses and slides; the morning plan; the top edge",
             "168"],
            ["12.5 mm sphere", "sphere suite, R 12.5 mm", "as the sphere suite", "337"],
            ["Half rod", "rod suite", "pass 1 touch-offs and pivots; stage 2 presses, slides around the rod, rolls, twists",
             "124"],
            ["Holder, +y station pad-up", "facing hovers", "W 0.5 to 45 mm over the held apex, twists &#8722;20 to +20&#176;, "
             "&#177;3 mm offsets", "58"],
            ["Holder, &#8722;y station pad-down", "side-by-side hovers", "g 13 to 40 mm beside the held tip, s 0 to 5 mm, "
             "twists &#8722;30 to +20&#176;", "162"],
            ["Keys", "homing", "the mount seated on each key and 12 mm out of it", "6"]]
    mo = ("<p>Each indenter's motions are the plans that the lab runs on it today (Table 1). The 12.5 mm sphere has no plans "
          "of its own yet, so the sphere suite stands in for it with the radius changed.</p>")
    S.append(("motions", "Motions of each indenter",
              mo + P.table(["Indenter", "Suite", "Motions", "Actions"], rows, num_cols=(3,), n=1,
                           caption="The motions that each indenter of the plate must allow.")))

    r = N["reach"]
    reach = (f"<p>Where an indenter stands relative to the robot decides whether its plans keep joints 5 and 6 inside their "
             f"windows (Figure 2). The tall suite's plans were built for the tall indenter, at about (100, 110) mm in A. With "
             f"its 5 mm sphere at y = +110 mm, {r['tall_best']} of the 168 tall actions clear the joints from x = 40 mm back. "
             f"At y = &#8722;110 mm only {r['tall_worst_lo']} to {r['tall_worst_hi']} clear. There the flank presses take "
             f"joint 5 under 30&#176; or toward the wrist's singularity. The sphere suite clears best near y = 0 to +60 mm, "
             f"with {r['sph_best']} of 337. It loses {r['sph_y110']} at y = +110 mm. The rod suite clears all 124 actions at "
             f"every position tried, and the hovers clear at most positions. Between 120 and 180 mm, an indenter's height "
             f"changes these counts by a few actions.</p>")
    reach += P.fig(2, MEDIA / "reach.png",
                   "Joint reach over the plate, the robot to the left. Each cell is one position of the suite's indenter; "
                   "the 5 mm sphere's top is at 160 mm and the holder's blocks at 153 mm. The colour gives the share of the "
                   "suite's actions inside every joint window, and the number the actions outside. The star marks the "
                   "indenter's place in the proposed layout.")
    S.append(("reach", "Joint reach of each suite over the plate", reach))

    na, nt = A["side"]["n"] - A["side"]["flagged"], A["tall5"]["flagged"]
    pr = (f"<p>The plate printed on 2026-10-27 ({link(L['array'], 'indenter array')}) has the 5 mm sphere and the rod in "
          f"front at 120 mm. The 12.5 mm sphere and the holder stand at the back at 180 mm. For the side-by-side map its "
          f"&#8722;y station's tip must hang pad-down, so the checks cut that station's pillar. Then only {na} of the 162 "
          f"side-by-side hovers clear, all at g 13 to 15 mm (Figure 3). Past 15 mm the mount and finger run into the 12.5 mm "
          f"sphere, and the HEX-E reaches the 5 mm sphere and the rod. The tall suite loses {nt} of its 168 actions to the "
          f"joint windows. Its 5 mm sphere stands at (&#8722;15, &#8722;42) mm, where the reach map is worst.</p>")
    pr += P.fig(3, MEDIA / f"{STEM}_array_plan.png",
                "The printed plate in plan view, one panel per suite. The colour is the suite's sweep: the lowest point of "
                "the robot and the tool over each 2 mm square. Red outlines mark the parts that the suite comes too near. "
                "Dashed: the P1S's 256 x 256 mm build plate. R5 and R12.5 are the spheres and k1 to k3 the keys; the "
                "triangles mark the held tips, pointing down for the pad-down one.")
    S.append(("printed", "The printed plate", pr))

    s5, s12, rd, h = gF["sphere5"], gF["sphere12"], gF["rod"], gF["holder"]
    lay = (f"<p>The proposed layout keeps the printed plate's base, keys and indenters and moves them (Table 2, Figure 4). "
           f"The 5 mm sphere goes to the back right at ({s5['c'][0]:g}, {s5['c'][1]:g}) mm, {s5['top']:g} mm tall, where the "
           f"tall suite's joints clear. The holder goes to the back left at ({h['c'][0]:g}, {h['c'][1]:g}) mm. Its pad-down "
           f"station faces the &#8722;y edge, so the side-by-side poses fall past the plate. On their way from the robot to "
           f"the back corners, the wrist and the HEX-E come down to {N['free_front']:.0f} to {N['free_front_hi']:.0f} mm "
           f"over the front of the plate. The front row therefore stands lower than on the printed plate. The 12.5 mm sphere "
           f"goes to the front right at ({s12['c'][0]:g}, {s12['c'][1]:g}) mm and {s12['top']:g} mm tall. That is under "
           f"the {N['free_s12']:.0f} mm that the back-right sphere's suites leave there. Its own suite moves the mount up to "
           f"85 mm sideways and back from the sphere, at up to 40 mm over its top. In the back middle, between the 5 mm "
           f"sphere and the holder, {N['v2c_s12_bad']} of its 337 actions put the mount near the holder. The rod goes to "
           f"the front left at ({rd['c'][0]:g}, {rd['c'][1]:g}) mm and drops to {rd['top']:g} mm, under the side-by-side "
           f"wrist's {N['free_rod']:.0f} mm.</p>")
    lay += spec_table(gF, 2)
    lay += P.fig(4, MEDIA / f"{STEM}_{FINAL}_plan.png", "The proposed plate in plan view, one panel per suite, as in Figure 3.")
    ns = N["side_gaps"]
    lay += (f"<p>On the proposed plate, every action that does not clear fails on its own indenter or on a joint window "
            f"(Table 3). The side-by-side map clears every gap from 13 to 40 mm at twist 0. Twists of &#177;10&#176; clear "
            f"from {ns['pm10']} mm, and &#8722;20 to &#8722;30&#176; from {ns['m20']} mm. The facing hovers clear all 58. "
            f"The tall suite keeps {F['tall5']['n'] - F['tall5']['flagged']} of 168, against "
            f"{A['tall5']['n'] - A['tall5']['flagged']} on the printed plate.</p>")
    lay += suite_table([("array", "the printed plate"), (FINAL, "the proposed plate")], 3)
    if N["renders"]:
        lay += P.fig(5, MEDIA / "renders.png",
                     "The proposed plate in the cell scene of <code>scripts/ur5_cell_render.py</code>, seen from the plate's "
                     "&#8722;y side, the robot to the left. The robot stands at one clear action of each suite: side by side at g 15 mm, from the side "
                     "and from above; facing at W 2 mm; a flank press of the tall suite; a press on the 12.5 mm sphere; a "
                     "press on the rod.")
    S.append(("layout", "The proposed layout", lay))

    pc = N["parts"]
    (x0, y0, z0), (x1, y1, z1) = pc["bounds_mm"]
    prn = (f"<p>The proposed plate is one body, {x1 - x0:.0f} x {y1 - y0:.0f} x {z1 - z0:.0f} mm. It is {pc['volume_cm3']:.0f} "
           f"cm&#179; solid, {pc['mass_g_solid']:.0f} g of PLA at full infill, and fits the P1S's 256 x 256 x 256 mm. The "
           f"pad-down station's block hangs from the holder's column on its arm and 45&#176; gusset. Its underside, where "
           f"the held tip screws on, overhangs 21 mm and needs support under that block alone. The pad-up station keeps "
           f"its pillar to the base. STL: {code(str(Path(pc['stl'])))}; geometry: "
           f"{code(str((CAD / f'{STEM}_{FINAL}_parts.json').relative_to(ROOT)))}.</p>")
    S.append(("print", "Printing and mounting", prn))

    fw = f"""<ul>
<li>Replay every sample of every move on the proposed plate, the travel's straight moves included
(<code>scripts/indenter_plate_layout.py replay --layout {FINAL} --suite S</code> without <code>--quick</code>). An action
that clears here and fails there names a travel that the quick runs skip.</li>
<li>Build plans for the 12.5 mm sphere, as the rod's were built: face and near-face points and the tilts its radius allows.
The sphere suite's edge and flank presses run the finger or mount into that sphere itself, on either plate.</li>
<li>Place the plate with A where the low sphere stands today, as the checks assume. Elsewhere, move A in
<code>LAYOUTS</code> and rerun the reach maps, which change with the plate's place.</li>
<li>Print the pad-down station's support and check by eye that the held tip hangs level. The side-by-side poses assume its
apex 41 mm under the pad-up apex.</li>
</ul>"""
    S.append(("future", "Future work", fw))

    nf = lambda s: F[s]["n"] - F[s]["flagged"]                       # noqa: E731
    lede = (f"On the plate printed on 2026-10-27, the side-by-side crosstalk map fails. With the &#8722;y tip pad-down, "
            f"only {na} of 162 side hovers clear: the 12.5 mm sphere and the front row stand where the robot's tool and "
            f"wrist pass. Its 5 mm sphere also costs the tall suite {nt} of 168 actions to the joint windows. On the "
            f"proposed plate the 5 mm sphere moves to the back right and the holder to the back left. The 12.5 mm sphere "
            f"and the rod move to the front, at 95 and 100 mm. There {nf('side')} side hovers, {nf('tall5')} tall actions "
            f"and {nf('sphere5')} of 337 sphere actions clear. The plate prints in one piece on the P1S.")
    summary = (f"The printed indenter plate blocks the side-by-side map ({na} of 162 hovers clear) and the tall suite ({nt} of "
               f"168 actions out of the joint windows). A layout with the 5 mm sphere back right, the holder back left, and "
               f"the 12.5 mm sphere and the rod in front at 95 and 100 mm clears {nf('side')} side hovers and "
               f"{nf('tall5')} tall actions, and fits the P1S plate.")
    return S, lede, summary


def free_heights(layout):
    """Free height (mm) over the plate from the back corners' suites (sphere5, tall5, side, facing) of a layout."""
    Z, lo, cell = None, None, None
    for s in ("sphere5", "tall5", "side", "facing"):
        f = DATA / f"{STEM}_{layout}_{s}_quick.npz"
        if f.exists():
            z = np.load(f)
            Z = z["zmin"] if Z is None else np.minimum(Z, z["zmin"])
            lo, cell = z["lo"], float(z["cell"])
    xs = lo[0] + (np.arange(Z.shape[0]) + 0.5) * cell
    ys = lo[1] + (np.arange(Z.shape[1]) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    def free(cx, cy, r):
        m = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
        v = Z[m]
        return float(np.min(v)) if np.isfinite(v).any() else float("inf")
    return free


def build():
    sys.argv = sys.argv[:1]
    import indenter_plate_layout as PL
    MEDIA.mkdir(parents=True, exist_ok=True)
    gA, gF = PL.plate("array"), PL.plate(FINAL)
    N = dict(gA=gA, gF=gF, array={s: run("array", s) for s in SUITES}, final={s: run(FINAL, s) for s in SUITES})
    rd = {s: json.loads((DATA / f"{STEM}_reach_{s}.json").read_text()) for s in ("tall5", "sphere5")}
    t = [r for r in rd["tall5"]["rows"] if r["h"] == 160.0]
    sp = [r for r in rd["sphere5"]["rows"] if r["h"] == 160.0]
    N["reach"] = dict(tall_best=max(r["ok"] + r["attended"] for r in t if r["y"] == 110.0 and r["x"] >= 40),
                      tall_worst_lo=min(r["ok"] + r["attended"] for r in t if r["y"] == -110.0),
                      tall_worst_hi=max(r["ok"] + r["attended"] for r in t if r["y"] == -110.0),
                      sph_best=max(r["ok"] + r["attended"] for r in sp),
                      sph_y110=max(r["fail"] for r in sp if r["y"] == 110.0))
    free = free_heights("v2a")
    N["free_front"] = min(free(x, y, 30) for x in (-40, -20, 0) for y in (-25, 0, 25, 50, 75))
    N["free_front_hi"] = max(free(x, y, 30) for x in (-40, -20, 0) for y in (-25, 0, 25, 50, 75))
    N["free_s12"] = free(gF["sphere12"]["c"][0], gF["sphere12"]["c"][1], 30)
    v2c = run("v2c", "sphere12")
    N["v2c_s12_bad"] = sum(1 for r in v2c["rows"] if any("holder" in w or "held tips" in w for w in r["flags"]))
    N["free_rod"] = free(gF["rod"]["c"][0], gF["rod"]["c"][1], 28)
    N["parts"] = json.loads((CAD / f"{STEM}_{FINAL}_parts.json").read_text())
    import re as _re
    ok = [tuple(int(v) for v in _re.match(r"hov-s-g(\d+)-s(\d+)-t([+-]\d+)", r_["id"]).groups())
          for r_ in N["final"]["side"]["rows"] if not r_["flags"]]
    N["side_gaps"] = dict(pm10=min(g for g, _, t in ok if abs(t) == 10 and sum(1 for g2, _, t2 in ok if g2 == g and t2 == -t)),
                          m20=min(g for g, _, t in ok if t <= -20))
    fig_reach(MEDIA / "reach.png", dict(g=gF))
    N["renders"] = fig_renders(MEDIA / "renders.png", FINAL)
    S, lede, summary = sections(N)
    mp4 = MEDIA / f"{STEM}_{FINAL}_film.mp4"
    film_html = P.film(1, mp4, MEDIA / f"{STEM}_{FINAL}_film_poster.png",
                       "One clear action of each suite on the proposed plate, with its travel and its moves: two side-by-side "
                       "hovers, a facing hover, a flank slide of the tall suite, a slide on the 5 mm sphere, a press on the "
                       "12.5 mm sphere and a slide on the rod. Seen from the plate's &#8722;y side, the robot to the left.") \
        if mp4.exists() else ""
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {hd}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {hd}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    byline = ("The lab's plans for v3 ID 1 replayed on the printed plate and on a proposed layout, mount v7. Tool "
              "<code>scripts/indenter_plate_layout.py</code>; page by <code>scripts/indenter_plate_layout_page.py</code>; rows in "
              + link(DATA, "data/") + ". The cell's queue: " + link(L["v3"], "the v3 tips' README") + "; the parts: "
              + link(L["array"], "indenter array") + ", " + link(L["tall"], "tall indenter") + ", "
              + link(L["holder"], "fingertip holder") + ".")
    page = TEMPLATE.read_text()
    title = "Layout of the indenter plate for the UR5 calibration cell"
    for k, v in {"TITLE": title, "H1": title, "EYEBROW": "Printed fingertips &#183; 2026-10-28", "LEDE": lede,
                 "BYLINE": byline, "FILM": film_html, "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = P.TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary, date="2026-10-28")
    PAGE.write_text(page)
    print(f"{PAGE}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    build()
