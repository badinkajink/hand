#!/usr/bin/env python3
"""Page: MuJoCo model of the SR2 finger assembled from its CAD parts, with sphere collision geometry (2026-10-13).

Owner, 2026-10-13: "can you update the current finger model with the real CAD models? ... the attachment points (where
the joints connect) are the 10.2mm diameter hole (11.2mm diameter chamfer) to a slotted 2.6mm radius (5.2mm diameter)
hole ... this should enable us to update the collision mesh (using sphere packing, eg morphit ...) ... ideally you
build in a mujoco webasm viewer in the html docpage so i can inspect in browser". On the tip: "since we have the 1mm
sphere packed fingertips, we should keep the two separate. but i provided the fused one as a reference on how to
couple them".

Data: assets/mjcf/experimental/20261013-cad_v1/ (assembly.json from scripts/build_cad_v1_finger.py, interference.json
from scripts/cad_v1_interference.py, spheres_metrics.json from scripts/cad_v1_spheres.py), media from
scripts/cad_v1_finger_media.py, the viewer from scripts/mjwasm_embed.py.

    .venv/bin/python scripts/cad_v1_finger_page.py
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import contact_overview_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import build_cad_v1_finger as B  # noqa: E402
import mjwasm_embed  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261013-cad_v1_finger")
OUT = os.path.join(D, "20261013-cad_v1_finger_model.html")
TPL = os.path.join(ROOT, "scripts/cad_v1_finger_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
MODEL = str(B.OUT)
REL_MODEL = "assets/mjcf/experimental/20261013-cad_v1"
HS = "&#8202;"
FIG, TAB, REFS = [0], [0], {}
EXTRA_CSS = """
.tw th{white-space:normal !important;vertical-align:bottom;text-transform:none}
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
figure.films video{width:100%;height:auto}
figure.pic img{width:100%;height:auto;display:block}
figure.diagram > svg{width:100%;height:auto}
.eq{margin:.8em 0;text-align:center}
""" + texsvg.BLOCK_CSS + "\n"


def mm(x, nd=2):
    return f"{x:.{nd}f}{HS}mm"


def data_uri(path, mime):
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def figure(body, caption, key=None, cls="pic"):
    FIG[0] += 1
    if key:
        REFS[f"fig:{key}"] = FIG[0]
    return f'<figure class="{cls}">{body}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def img(name, caption, key):
    p = os.path.join(D, "media", name)
    mime = "image/png" if name.endswith(".png") else "image/jpeg"
    return figure(f'<img src="{data_uri(p, mime)}" alt="">', caption, key)


def film(stem, caption, key):
    p = os.path.join(D, "media", f"{stem}.mp4")
    poster = os.path.join(D, "media", f"{stem}.jpg")
    v = (f'<video controls muted loop playsinline preload="metadata" poster="{data_uri(poster, "image/jpeg")}" '
         f'src="{data_uri(p, "video/mp4")}"></video>')
    return figure(v, caption, key=key, cls="films")


def table(head, body, caption, note=None, text_cols=(0,), key=None):
    TAB[0] += 1
    if key:
        REFS[f"tab:{key}"] = TAB[0]
    cls = lambda i: "" if i in text_cols else " class=num"  # noqa: E731
    out = [f"<p class='tcap'><b>Table&#160;{TAB[0]}.</b> {caption}</p>",
           "<div class='tw'><table><thead><tr>" + "".join(f"<th{cls(i)}>{h}</th>" for i, h in enumerate(head)) + "</tr></thead><tbody>"]
    for r in body:
        out.append("<tr>" + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table></div>")
    if note:
        out.append(f"<p class='tnote'>{note}</p>")
    return "".join(out)


def eq(tex):
    return f'<div class="eq">\\[{tex}\\]</div>'


def render_tex(t):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None) for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    return P.TEX_RE.sub(lambda m: next(svgs), t), len(items)


class Data:
    def __init__(self):
        self.asm = json.load(open(os.path.join(MODEL, "assembly.json")))
        self.itf = json.load(open(os.path.join(MODEL, "interference.json")))
        self.sph = json.load(open(os.path.join(MODEL, "spheres_metrics.json")))
        self.sph_tight = json.load(open(os.path.join(MODEL, "spheres_tight_metrics.json")))
        self.bench = json.load(open(os.path.join(MODEL, "collision_bench.json")))
        self.A = B.derive()
        self.rep = self.asm["report"]
        self.km = self.rep["kinematics_mm"]
        import mujoco
        self.m_cad = mujoco.MjModel.from_xml_path(os.path.join(MODEL, "hand.xml"))
        self.m_rv1 = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets/mjcf/real_v1/real_hand.xml"))

    def stops(self):
        """Joint stops from the single-joint sweeps: per joint, per colliding pair, the angles that collide."""
        out = {}
        for j, s in self.itf["sweeps"].items():
            q = np.array(s["q_deg"])
            rows = s["overlap_mm3"]
            for k in rows[0]:
                v = np.array([r[k] for r in rows])
                hit = q[v > self.itf["threshold_mm3"]]
                if len(hit):
                    out.setdefault(j, {})[k] = hit
        return out


GLOSSARY = [
    ("Bearing bore", "The 10.2&#8202;mm bore, 3.5&#8202;mm deep with a 0.5&#8202;mm chamfer to 11.2&#8202;mm, in the parent part of a joint. "
     "It holds an MR105 bearing (5&#8202;&#215;&#8202;10&#8202;&#215;&#8202;4&#8202;mm) flush in a 4.0&#8202;mm wall, the bearing wall."),
    ("Slotted bore", "The 5.2&#8202;mm bore in the child part of a joint, with a slot that removes 40&#176; of its circumference. The shaft "
     "through the bearing passes through it."),
    ("Joint axis", "The common axis of the parent&#8217;s bearing bore and the child&#8217;s slotted bore."),
    ("Clevis", "A child part with two arms that straddle the parent: one arm carries the slotted bore, the other is screwed to the "
     "parent&#8217;s servo horn."),
    ("Servo seat", "Where the SCS0009 sits in its parent: output coaxial with the bearing bore, base on the inner face of the bearing "
     "wall, the two tab holes (2.3&#8202;mm, 28.5&#8202;mm apart) on the parent&#8217;s two 1.8&#8202;mm pilot holes."),
    ("Horn stack", "Axial distance from the top of the servo&#8217;s round boss to the child&#8217;s horn-side arm, in mm: the room the horn "
     "takes. The horn is not among the STLs."),
    ("Yaw centre", "The point on the yaw axis nearest the MCP axis. It is the mount position of real_v1 and the origin of each finger in "
     "the model."),
    ("Penetration", "Volume of one part&#8217;s 0.3&#8202;mm voxels lying more than one voxel inside another part, in mm&#179;, the larger of "
     "the two directions; in the joint sweeps the area of boundary voxels so placed, in mm&#178;. Flush faces, such as a servo in a "
     "pocket of its own width, give zero; a collision is a penetration above 0.1."),
    ("Joint stop", "The joint angle at which two CAD parts first penetrate, all other joints at zero."),
    ("Coverage", "Fraction of 20&#8202;000 points sampled uniformly inside a body&#8217;s meshes that lie inside at least one of its "
     "collision spheres."),
    ("Protrusion", "Distance from points on the surface of the sphere union that lie outside the meshes to the nearest mesh surface, in mm; "
     "p95 is its 95th percentile. A protrusion makes a contact where the part has no material."),
    ("Surface fit", "Fraction of 20&#8202;000 points sampled on a body&#8217;s mesh surface that lie within 1&#8202;mm of the surface of its sphere "
     "union. A point farther in is a place an object can press into the part without contact."),
    ("Pads", "Spheres of 0.75&#8202;mm radius at 1&#8202;mm spacing on the palmar, end and side faces of the TPU tip, from "
     "<code>scripts/fingertip_geometry.py</code> through <code>reorient_backends.replace_tips</code>, the repository&#8217;s default "
     "fingertip. K is the stiffness of one pad, E&#8202;A/h with E&#8202;=&#8202;10&#8202;MPa; d0 is its solimp."),
    ("real_v1", "The capsule hand model in use, <code>assets/mjcf/real_v1/real_hand.xml</code>."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def lede(X: Data):
    km = X.km
    sph = X.sph
    s_txt = ""
    if sph:
        n = sum(r["spheres"] for r in sph["bodies"].values())
        bench = {r["model"]: r["us_per_step"] for r in X.bench["rows"]}
        add = bench["CAD finger, balanced spheres and pads"] - bench["CAD finger, pads only"]
        s_txt = (f" The collision geometry is {n} MorphIt spheres per finger, larger than the parts by up to "
                 f"{max(r['protrusion_max_mm'] for r in sph['parts'].values()):.0f}&#8202;mm, and the "
                 f"{X.asm['pads']['n_pads']['index']} pads of the TPU tip; the spheres add {add:.0f}&#8202;&#181;s to a "
                 f"{bench['CAD finger, pads only']:.0f}&#8202;&#181;s step of the screwdriver scene.")
    return (f"The real_v1 finger is three capsules. This model replaces them with the printed carriage, links and TPU tip and the "
            f"three SCS0009 servos, assembled from the part STLs by matching each 10.2&#8202;mm bearing bore to the next part&#8217;s "
            f"5.2&#8202;mm slotted bore. The assembly gives {mm(km['yaw_to_mcp'])} from the yaw axis to the MCP axis and "
            f"{mm(km['pip_to_tip_end'])} from the PIP axis to the tip, as in real_v1. It seats every servo on its pilot holes to "
            f"0.000&#8202;mm, and the parts stop within 1&#176; of the measured ranges of motion, except that the MCP link reaches the "
            f"carriage from &#8722;79&#176; of yaw. The MCP link STL spaces the MCP and PIP axes {mm(km['mcp_to_pip'])} apart, against "
            f"20.75&#8202;mm in real_v1, so the finger is 10.05&#8202;mm longer.{s_txt}")


def byline():
    return ("Assembly <code>scripts/build_cad_v1_finger.py</code> with <code>src/morphohand/geometry/cad_features.py</code> &#183; "
            "stops <code>scripts/cad_v1_interference.py</code> &#183; spheres <code>scripts/cad_v1_spheres.py</code> (MorphIt) &#183; "
            "media <code>scripts/cad_v1_finger_media.py</code> &#183; viewer <code>scripts/mjwasm_embed.py</code> &#183; builder "
            "<code>scripts/cad_v1_finger_page.py</code> &#183; model <code>assets/mjcf/experimental/20261013-cad_v1/</code>")


def parts_section(X: Data):
    A = X.A
    rows = []
    names = {"carriage": ("Carriage", "carriage.stl"), "yaw_link": ("Yaw link", "mcp_mount_YAW-link.stl"),
             "mcp_link": ("MCP link", "pip_mount_MCP_link.stl"), "dip_link": ("DIP link", "fingertip_mount_DIP_link.stl")}

    def pt(p):
        return "(" + ", ".join(f"{v:.2f}" for v in p) + ")"

    for k, (title, f) in names.items():
        pi, ci = A["parent_info"].get(k), A["child_info"].get(k)
        bear = f"{pt(pi['bore'].point)}, axis {pt(pi['bore'].axis)}" if pi else "none"
        slot = f"{pt(ci['slot'].point)}, axis {pt(ci['slot'].axis)}" if ci else "none"
        pil = (f"{np.linalg.norm(B.perp(pi['pilots'][0].point - pi['pilots'][1].point, pi['a'])):.2f}" if pi else "&#8211;")
        rows.append([f"{title} <code>{f}</code>", bear, slot, pil])
    t = table(["Part (file in <code>assets/cad_v1/</code>)", "Bearing bore centre and axis (part frame, mm)",
               "Slotted bore centre and axis (part frame, mm)", "Pilot-hole pitch (mm)"], rows,
              "Bores found in each part. Coordinates are in each STL&#8217;s own frame, the carriage&#8217;s after removing its export "
              "rotation. The pilot-hole pitch matches the 28.50&#8202;mm between the servo&#8217;s tab holes.", key="bores",
              text_cols=(0, 1, 2))
    sv = X.rep["servo"]
    return (
        "<p>Seven STLs describe one finger: the carriage that rides on the gantry, the yaw link, the MCP link, the DIP link, the TPU tip, "
        "the SCS0009 servo, and the DIP link and tip fused in their assembled position. Every part is in millimetres and in its own "
        f"frame. The assembly reads three kinds of feature from them, shown in Figure&#160;[[fig:parts]] and listed in "
        f"Table&#160;[[tab:bores]]: the bearing bores, the slotted bores, and the servo seats. The servo STL gives the output spline "
        f"(4.0&#8202;mm), the case base, the tab holes {mm(sv['tab_pitch'])} apart with their lower face {mm(sv['tab_face'])} above the base, "
        f"the top of its round boss at {mm(sv['boss_top'])} and the end of the spline at {mm(sv['height'])}; the spline lies "
        f"{mm(sv['spline_offset_from_tab_centre'])} from the midpoint of the tab holes.</p>"
        + img("parts.png", "The CAD parts with the bores the assembly reads: red, the 10.2&#8202;mm bearing bore; blue, the 5.2&#8202;mm "
              "slotted bore; green, the servo&#8217;s output spline. Each line is drawn along the bore&#8217;s axis.", "parts")
        + t)


def method_section(X: Data):
    rot = X.rep["parts"]["carriage_export_rotation_deg"]
    return (
        "<p>A CAD export tessellates a cylinder into flat chords. Two neighbouring chords share an edge parallel to the axis, both normals "
        "are perpendicular to that edge, and the angle \\(\\theta\\) between the normals with the chord width \\(w\\) fixes the radius:</p>"
        + eq(r"r = \frac{w}{2\,\sin(\theta/2)}")
        + "<p>The axis lies at distance \\(r\\) from the shared edge along the bisector of the two normals, toward them for a bore. Grouping "
        "these pairs by axis direction, axis line and radius finds every bore and boss of a part without growing patches across tangent "
        "fillets. Rays cast parallel to an axis at fixed radii then give the faces along it: the bearing wall, the servo cavity and the arms "
        "of a clevis (<code>cad_features.axial_intervals</code>).</p>"
        "<p>Five rules then fix every transform. The joint axis is where the parent&#8217;s bearing bore and the child&#8217;s slotted bore "
        "coincide. Along that axis the inner face of the child&#8217;s slotted arm lies on the outer face of the bearing wall, which is flush "
        "with the bearing. Each servo sits with its output on the bearing axis, its base on the inner face of the bearing wall, and its tab "
        "holes on the two pilot holes. At zero joint angles the finger is straight: the yaw link runs from the yaw axis to the MCP axis "
        "straight down, the MCP link continues it to the PIP axis, and the DIP link continues parallel to the tip&#8217;s pad face. The "
        "carriage hangs with its long arm up toward the gantry, as in the owner&#8217;s CAD views (Figure&#160;[[fig:assembly]]). The TPU tip "
        "takes its pose on the DIP link from <code>fingertip_and_DIP_link.stl</code>, registered by vertex ICP.</p>"
        f"<p>The carriage STL is rotated {rot:.3f}&#176; about its x axis: its bores and its largest planar faces carry the same tilt, so the "
        "builder rotates it back before reading it. Real_v1&#8217;s conventions are kept: the finger hangs along &#8722;z from the yaw "
        "centre at the real_v1 mounts, the yaw axis is +x, the flexion axes are +y on the index and middle fingers and &#8722;y on the "
        "thumb, the pads face the opposing finger, and body names, joint names, qpos layout and actuators match "
        "<code>real_hand.xml</code>. The thumb is the same parts turned 180&#176; about the vertical.</p>")


def checks_section(X: Data):
    rep, km = X.rep, X.km
    sv = rep["servos"]
    j = rep["joints"]
    resid = max(max(v["tab_hole_to_pilot_axis_mm"]) for v in sv.values())
    pip_horn = j["pip"]["horn_side_inner_face_axial_mm"]
    rest = X.itf["rest"]
    rest_max = max(rest.values())
    rows = [
        ["Yaw axis to MCP axis", mm(km["yaw_to_mcp"]), "20.75&#8202;mm in real_v1 and the paper&#8217;s appendix"],
        ["MCP axis to PIP axis", mm(km["mcp_to_pip"]), "20.75&#8202;mm in real_v1"],
        ["PIP axis to the end of the tip", mm(km["pip_to_tip_end"]), "37.16&#8202;mm in real_v1"],
        ["Finger axis to the pad face", mm(-km["tip_bounds"][0][0]), "10.55&#8202;mm, real_v1&#8217;s capsule radius"],
        ["Servo tab holes to pilot-hole axes, largest of six", f"{resid:.3f}{HS}mm", "0"],
        ["Servo tab face against the pilot holes&#8217; far end", f"{sv['mcp']['tab_face_axial_mm']:.2f} and "
         f"{sv['mcp']['pilot_hole_far_ends_axial_mm'][0]:.2f}{HS}mm", "equal"],
        ["Horn stack, yaw and MCP joints (disc horn)", f"{j['yaw']['boss_to_horn_side_mm']:.1f} and {j['mcp']['boss_to_horn_side_mm']:.1f}{HS}mm",
         "equal at the two joints of one design"],
        ["Horn stack, PIP joint (arm horn in a 2.0&#8202;mm pocket)", f"{pip_horn - j['pip']['servo_boss_top_axial_mm'] + 4.2:.1f}{HS}mm",
         "outer face of the pocket"],
        ["Tip on the DIP link, registration residual", f"{rep['parts']['tip_registration_residual_mm']['tip']:.5f}{HS}mm", "0"],
        ["Penetration at zero joint angles, largest of 28 part pairs", f"{rest_max:.3f}{HS}mm&#179;", "0"],
    ]
    return ("<p>The checks in Table&#160;[[tab:checks]] test the rules against the parts. Both pairs of servo tab holes and pilot holes are "
            "independent of the rules that place them, and the horn stacks of the two disc-horn joints are measured on different parts. "
            "The servo base and the tab face are fixed by the bearing wall and land on the pilot holes&#8217; far end as well.</p>"
            + table(["Quantity", "Model", "Reference"], rows, "Assembly checks.", key="checks", text_cols=(0, 2)))


def compare_section(X: Data):
    km = X.km
    import mujoco
    mc, mr = X.m_cad, X.m_rv1

    def bm(m, name):
        return 1000 * m.body_mass[m.body(name).id]

    reach_c = km["yaw_to_mcp"] + km["mcp_to_pip"] + km["pip_to_tip_end"]
    rows = [["Yaw axis to MCP axis (mm)", "20.75", f"{km['yaw_to_mcp']:.2f}"],
            ["MCP axis to PIP axis (mm)", "20.75", f"{km['mcp_to_pip']:.2f}"],
            ["PIP axis to the end of the tip (mm)", "37.16", f"{km['pip_to_tip_end']:.2f}"],
            ["Yaw centre to the end of the tip, straight (mm)", "78.66", f"{reach_c:.2f}"]]
    for b, t in (("index_mount", "carriage and yaw servo"), ("index_yaw_frame", "yaw link and MCP servo"),
                 ("index_mcp_frame", "MCP link and PIP servo"), ("index_pip_frame", "DIP link"), ("index_tip", "TPU tip")):
        rows.append([f"Mass, {t} (g)", f"{bm(mr, b):.2f}", f"{bm(mc, b):.2f}"])
    ng = lambda m: sum(1 for i in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("index") or m.geom_bodyid[i] in [m.body(f"index_{x}").id for x in ("yaw_frame", "mcp_frame", "pip_frame", "tip")])  # noqa: E731
    rows.append(["Colliding geoms per finger", f"{sum(1 for i in range(mr.ngeom) if mr.geom_bodyid[i] in [mr.body(f'index_{x}').id for x in ('yaw_frame', 'mcp_frame', 'pip_frame', 'tip')])}",
                 f"{sum(1 for i in range(mc.ngeom) if mc.geom_contype[i] and mc.geom_bodyid[i] in [mc.body(f'index_{x}').id for x in ('mount', 'yaw_frame', 'mcp_frame', 'pip_frame', 'tip')])}"])
    del ng
    return (img("assembly.png", "The assembled model beside the owner&#8217;s CAD views of the same finger (from "
                "<code>assets/cad_v1/diagram.png</code>). The owner&#8217;s right-hand view is flexed at the PIP joint.", "assembly")
            + "<p>Figure&#160;[[fig:compare]] draws real_v1&#8217;s capsule finger on the same mount. The two agree to the PIP axis; "
            "from there the CAD finger is 10.05&#8202;mm longer, all of it in the MCP link. The servos stand off the dorsal side of every "
            "link, up to 31&#8202;mm from the finger axis at the yaw horn, where the capsules give 10.55&#8202;mm all round. Real_v1 masses "
            "come from its capsules at 1000&#8202;kg/m&#179;; the model&#8217;s from the CAD volumes at solid PLA (1240&#8202;kg/m&#179;) "
            "and TPU (1210&#8202;kg/m&#179;) density and 13.1&#8202;g per servo (Table&#160;[[tab:compare]]).</p>"
            + img("compare.png", "CAD finger (purple, servos black, TPU tip tan) and real_v1&#8217;s capsule finger (translucent teal) on "
                  "the same mount at three joint configurations, seen along the flexion axes.", "compare")
            + table(["Quantity", "real_v1", "CAD model"], rows, "Lengths, masses and colliding geoms of the index finger. Colliding "
                    "geoms of the model are its link spheres and pads; the carriage counts as part of the finger.", key="compare"))


def rom_chart(X: Data):
    sweeps = X.itf["sweeps"]
    rom = {"yaw": (-85, 85), "mcp": (-15, 92), "pip": (-18, 92)}
    title = {"yaw": "Yaw", "mcp": "MCP", "pip": "PIP"}
    pair_name = lambda k: k.replace("|", " / ").replace("_", " ")  # noqa: E731
    colors = ["#c0392b", "#2c6fbb", "#27864a", "#8e44ad", "#d35400"]
    Wp, Hp, ml, mb, mt = 268, 220, 52, 44, 30
    W = 3 * Wp + 10
    parts = [f'<svg viewBox="0 0 {W} {Hp + 70}" xmlns="http://www.w3.org/2000/svg" font-family="Times, serif" font-size="14">']
    all_pairs = []
    ymax = 1.0
    for j in ("yaw", "mcp", "pip"):
        for k in sweeps[j]["overlap_mm3"][0]:
            v = max(r[k] for r in sweeps[j]["overlap_mm3"])
            if v > X.itf["threshold_mm3"]:
                ymax = max(ymax, v)
                if k not in all_pairs:
                    all_pairs.append(k)
    ytop = 10 ** math.ceil(math.log10(ymax))
    for c, j in enumerate(("yaw", "mcp", "pip")):
        x0 = c * Wp + ml
        pw, ph = Wp - ml - 12, Hp - mb - mt
        q = np.array(sweeps[j]["q_deg"])
        lo, hi = q.min(), q.max()
        X_ = lambda a: x0 + (a - lo) / (hi - lo) * pw  # noqa: E731
        Y_ = lambda v: mt + ph - (np.log10(max(v, 0.1)) + 1) / (math.log10(ytop) + 1) * ph  # noqa: E731
        a, b = rom[j]
        parts.append(f'<rect x="{X_(a):.1f}" y="{mt}" width="{X_(b) - X_(a):.1f}" height="{ph}" fill="#e9eef5"/>')
        parts.append(f'<rect x="{x0}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#000" stroke-width="1"/>')
        for e in range(-1, int(math.log10(ytop)) + 1):
            y = Y_(10 ** e)
            parts.append(f'<line x1="{x0 - 4}" x2="{x0}" y1="{y:.1f}" y2="{y:.1f}" stroke="#000"/>')
            if c == 0:
                parts.append(f'<text x="{x0 - 7}" y="{y + 5:.1f}" text-anchor="end">{10 ** e:g}</text>')
        for t in np.arange(math.ceil(lo / 30) * 30, hi + 1, 30):
            parts.append(f'<line x1="{X_(t):.1f}" x2="{X_(t):.1f}" y1="{mt + ph}" y2="{mt + ph + 4}" stroke="#000"/>')
            parts.append(f'<text x="{X_(t):.1f}" y="{mt + ph + 19}" text-anchor="middle">{t:.0f}</text>')
        parts.append(f'<text x="{x0 + pw / 2:.1f}" y="{mt + ph + 38}" text-anchor="middle">{title[j]} angle (deg)</text>')
        parts.append(f'<text x="{x0 + pw / 2:.1f}" y="{mt - 10}" text-anchor="middle">{title[j]} sweep, other joints at 0</text>')
        for i, k in enumerate(all_pairs):
            v = np.array([r[k] for r in sweeps[j]["overlap_mm3"]])
            if v.max() <= X.itf["threshold_mm3"]:
                continue
            pts = " ".join(f"{X_(a_):.1f},{Y_(b_):.1f}" for a_, b_ in zip(q, v))
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[i % 5]}" stroke-width="2"/>')
    parts.append(f'<text x="14" y="{mt + (Hp - mb - mt) / 2:.1f}" transform="rotate(-90 14 {mt + (Hp - mb - mt) / 2:.1f})" '
                 'text-anchor="middle">Penetration (mm&#178;)</text>')
    ly = Hp + 40
    x = ml
    for i, k in enumerate(all_pairs):
        parts.append(f'<line x1="{x}" x2="{x + 22}" y1="{ly}" y2="{ly}" stroke="{colors[i % 5]}" stroke-width="2"/>')
        parts.append(f'<text x="{x + 28}" y="{ly + 5}">{pair_name(k)}</text>')
        x += 40 + 9 * len(pair_name(k))
    parts.append(f'<rect x="{x}" y="{ly - 8}" width="22" height="14" fill="#e9eef5"/><text x="{x + 28}" y="{ly + 5}">measured range of motion</text>')
    parts.append("</svg>")
    return "".join(parts)


def rom_section(X: Data):
    st = X.stops()
    g = X.itf["grid"]

    def rng(a):
        neg, pos = a[a < 0], a[a >= 0]
        out = []
        if len(neg):
            out.append(f"{neg.min():.0f} to {neg.max():.0f}&#176;")
        if len(pos):
            out.append(f"{pos.min():.0f} to {pos.max():.0f}&#176;")
        return " and ".join(out)

    yaw_car = st.get("yaw", {}).get("carriage|yaw_link", np.array([]))
    yaw_mcp = st.get("yaw", {}).get("carriage|mcp_link", np.array([]))
    mcp_yaw = st.get("mcp", {}).get("yaw_link|mcp_link", np.array([]))
    txt = (f"<p>The parts collide only at the ends of the measured ranges of motion (Figure&#160;[[fig:rom]]). With the other joints at "
           f"zero, the yaw link meets the carriage at {rng(yaw_car)} of yaw, against the measured &#177;85&#176;; the MCP link meets the "
           f"carriage at {rng(yaw_mcp)}, inside the measured range on the negative side; the yaw link meets the MCP link at "
           f"{rng(mcp_yaw)} of MCP flexion, against the measured 92&#176;. The PIP joint is free over its measured &#8722;18 to 92&#176;. "
           f"Over the MCP&#8202;&#215;&#8202;PIP grid at 3&#176; steps, {100 * g['colliding_fraction']:.1f}&#8202;% of "
           f"{len(g['mcp_deg']) * len(g['pip_deg'])} configurations collide, all at the MCP stop; the tip and the DIP link never reach "
           "the MCP link or its servo. Stops that fall at both ends of the measured ranges also check the zero angles of the "
           "assembly.</p>")
    return txt + figure(rom_chart(X), "Penetration between CAD parts against joint angle, one joint at a time with the other joints at "
                        "zero, on a log scale; zero penetration is not drawn. The shaded band is the range of motion measured on the hand "
                        "(real_v1 and the paper&#8217;s Table&#160;I).", key="rom", cls="diagram")


def sphere_table(s, key, caption):
    names = {"carriage": "Carriage", "servo_yaw": "Yaw servo", "yaw_link": "Yaw link", "servo_mcp": "MCP servo",
             "mcp_link": "MCP link", "servo_pip": "PIP servo", "dip_link": "DIP link"}
    rows = []
    for p, r in s["parts"].items():
        rows.append([names.get(p, p), r["spheres"], f"{r['coverage']:.2f}", f"{r['protrusion_p95_mm']:.1f}", f"{r['protrusion_max_mm']:.1f}",
                     f"{r['fit_1mm']:.2f}", f"{r['seconds']:.0f}"])
    n_tot = sum(r["spheres"] for r in s["parts"].values())
    rows.append(["One finger", n_tot, "", f"{max(r['protrusion_p95_mm'] for r in s['parts'].values()):.1f}",
                 f"{max(r['protrusion_max_mm'] for r in s['parts'].values()):.1f}", "", f"{sum(r['seconds'] for r in s['parts'].values()):.0f}"])
    return table(["Part", "Spheres", "Coverage", "Protrusion p95 (mm)", "Protrusion max (mm)", "Surface fit within 1&#8202;mm",
                  "Fit time (s)"], rows, caption, key=key), n_tot


def collision_section(X: Data):
    pads = X.asm["pads"]
    s, t = X.sph, X.sph_tight
    bench = {r["model"]: r for r in X.bench["rows"]}
    sc = json.load(open(os.path.join(MODEL, "sphere_contacts.json")))
    yaw = sc["sweeps"]["yaw"].get("index_mcp_frame / index_mount")
    mcp = sc["sweeps"]["mcp"].get("index_mcp_frame / index_mount")
    tab_b, n_b = sphere_table(s, "spheres", "Balanced MorphIt spheres (the model&#8217;s collision geometry) per part, measured "
                              "against the part&#8217;s CAD mesh. Fit time includes loading the mesh and the inside tests.")
    tab_t, n_t = sphere_table(t, "spheres_tight", "Tight MorphIt spheres per part (<code>spheres_tight.json</code>), measured the same "
                              "way.")
    us = lambda k: bench[k]["us_per_step"]  # noqa: E731
    rows = [[k, bench[k]["ngeom_colliding"], f"{bench[k]['us_per_step']:.1f}", f"{bench[k]['steps_per_s'] / 1000:.1f}"] for k in bench]
    out = [
        "<p>Each link and each servo is packed by MorphIt (Nechyporenko et al., 2025; code <a href=\"https://github.com/HIRO-group/MorphIt-1\">"
        "HIRO-group/MorphIt-1</a>, commit 5f65299, MIT licence), which moves sphere centres and radii by gradient descent on losses for "
        "coverage, overlap, protrusion past the surface and distance to the surface. Its balanced preset, MorphIt-B, fills the volume "
        "and lets spheres stand off thin plates; its surface preset, MorphIt-S, weights protrusion 1000 to 1 over coverage and keeps "
        "spheres inside the part, with a hollow interior. The owner&#8217;s choice for this model: &#8220;it&#8217;s ok if the collision "
        "mesh is bigger than the visual mesh &#8230; speed is more important anyway&#8221;. The model therefore uses MorphIt-B at "
        f"{n_b} spheres per finger, {s['iterations']} iterations, seed {s['seed']} (Table&#160;[[tab:spheres]]). It covers 96 to 98&#8202;% "
        f"of every part and stands up to {max(r['protrusion_max_mm'] for r in s['parts'].values()):.1f}&#8202;mm off the surface. MorphIt-S "
        f"at {n_t} spheres per finger keeps every part within {max(r['protrusion_max_mm'] for r in t['parts'].values()):.1f}&#8202;mm of "
        "its surface (Table&#160;[[tab:spheres_tight]]); the builder takes either file with <code>--spheres</code>. Link and servo are "
        "packed apart and share their body&#8217;s spheres: the servo sits flush in its pocket, and a signed distance to the two meshes "
        "together misjudges points near the shared faces.</p>",
        tab_b, tab_t,
        img("spheres.png", "Collision geometry of the index finger at the real_v1 &#8216;open&#8217; angles: CAD meshes, the balanced "
            "MorphIt spheres with the tip pads, both, and the dorsal side.", "spheres"),
        f"<p>Spheres of neighbouring bodies overlap at every bearing, so they do not collide, as MuJoCo does for a body and its parent. "
        "Of the other pairs within one finger, only the carriage and the MCP link meet over the measured ranges of motion (section "
        "&#8220;Joint stops of the CAD parts&#8221;), so the model excludes the rest; the joint ranges stand in for the stops. The "
        f"remaining pair touches from {yaw[0]} to {yaw[1]}&#176; of yaw, where the parts meet from &#8722;86 to &#8722;79&#176;, and at "
        f"{mcp[0]}&#176; of MCP flexion; the PIP sweep and all {sc['grid']['n']} configurations of the MCP&#8202;&#215;&#8202;PIP grid "
        "at 3&#176; have none. Fingers collide with each other, the tool, the floor and the palm plate as in real_v1.</p>",
        "<p>Table&#160;[[tab:bench]] gives the cost of the extra geometry on the screwdriver scene at its &#8216;open&#8217; key, where the "
        f"fingers touch nothing: the balanced spheres add {us('CAD finger, balanced spheres and pads') - us('CAD finger, pads only'):.0f}"
        f"&#8202;&#181;s per step to the {us('CAD finger, pads only'):.0f}&#8202;&#181;s that the {pads['n_pads']['index'] * 3} pads "
        f"cost, and the tight spheres {us('CAD finger, tight spheres and pads') - us('CAD finger, pads only'):.0f}&#8202;&#181;s. Before "
        "the exclusions the balanced spheres cost more than the tight ones, because their larger boxes overlap the pads of the same "
        "finger&#8217;s tip.</p>",
        table(["Scene", "Colliding geoms", "Step time (&#181;s)", "Steps per second (thousands)"], rows,
              "Wall time of one 2&#8202;ms step of the screwdriver scene at its &#8216;open&#8217; key, MuJoCo "
              f"{X.bench['mujoco']} on one core of this workstation, best of three runs of 1000 steps "
              "(<code>scripts/cad_v1_collision_bench.py</code>).", key="bench", text_cols=(0,)),
        f"<p>The tip keeps the repository&#8217;s pad model. <code>reorient_backends.replace_tips(root, \"tpu2.7\", \"pads\", ...)</code> "
        f"runs on a copy of real_v1&#8217;s <code>scene_screwdriver_medium.xml</code> with the CAD fingers in it, with "
        f"<code>fingertip_geometry</code> pointed at the CAD insert in that process only: {pads['block_mm'][2]:.2f}&#8202;mm long "
        f"instead of 22.0&#8202;mm. It gives {pads['n_pads']['index']} pads per tip, K&#8202;=&#8202;{pads['pad_K']:.1f}&#8202;N/m and "
        f"d0&#8202;=&#8202;{pads['pad_d0']['index']:.3f} against the screwdriver. The same pads are copied into <code>finger.xml</code> "
        "and <code>hand.xml</code>. The DIP link and the tip stay separate bodies; the fused STL only fixes where the tip sits. The "
        f"scene&#8217;s keys are raised {pads['palm_raised_mm']:.2f}&#8202;mm with the palm, so the longer fingers start where "
        "real_v1&#8217;s did; its fitted grasp key <code>open_ik</code> is dropped.</p>",
        img("hand.png", "The three fingers on the real_v1 mounts at the &#8216;open&#8217; key: CAD meshes, and the collision spheres "
            "with the pads.", "hand"),
    ]
    return "".join(out)


def viewer_section(X: Data):
    v1 = mjwasm_embed.embed(os.path.join(MODEL, "finger.xml"), element_id="finger", height_px=620,
                            title="One finger, finger.xml")
    v2 = mjwasm_embed.embed(os.path.join(MODEL, "hand.xml"), element_id="hand", height_px=560, title="Three fingers, hand.xml",
                            include_runtime=False)
    return ("<p>The two viewers below run MuJoCo 3.14.0 compiled to WebAssembly on the model files themselves, inlined in this page. "
            "Drag to orbit, right-drag to pan, scroll to zoom. The sliders set the joint angles; Play steps the physics with the position "
            "actuators following the sliders. Geom group 2 holds the CAD meshes, group 3 the MorphIt spheres, group 4 the tip pads; the "
            "checkboxes show and hide them.</p>"
            + v1 + v2
            + "<p>The same files open in MuJoCo&#8217;s native viewer from the repository root:</p>"
            f"<p><code>.venv/bin/python -m mujoco.viewer --mjcf={REL_MODEL}/finger.xml</code><br>"
            f"<code>.venv/bin/python -m mujoco.viewer --mjcf={REL_MODEL}/hand.xml</code><br>"
            f"<code>.venv/bin/python -m mujoco.viewer --mjcf={REL_MODEL}/scene_screwdriver_medium.xml</code></p>"
            "<p>The native viewer shows groups 0 to 2 by default; keys 3 and 4 toggle the spheres and the pads.</p>"
            + film("sweep", "One joint sweep shown three ways: the CAD meshes, the collision spheres and pads, and real_v1&#8217;s capsules, "
                   "all at the same joint angles: flexion of MCP and PIP to their upper limits and back to their lower limits, then yaw "
                   "from &#8722;85 to 85&#176;.", "sweep"))


def limitations(X: Data):
    items = [
        "Masses come from the CAD volumes at solid PLA and TPU density and the SCS0009&#8217;s listed 13.1&#8202;g; printed parts with "
        "infill weigh less. The parts have not been weighed.",
        "The servo horns, shafts, bearings, screws and cables are not in the STLs and not in the model. The horn stack of "
        "5.2&#8202;mm is empty space in the model.",
        "The carriage hangs from the yaw axis with its long arm up, as in the owner&#8217;s CAD views; how it meets the gantry is not "
        "modelled. Real_v1&#8217;s palm plate is solid, so carriage-to-plate contact is excluded.",
        "The thumb is the index finger&#8217;s parts turned 180&#176; about the vertical; mirrored parts would need their own STLs.",
        "The grasp key <code>open_ik</code> of real_v1&#8217;s screwdriver scene presses the pads into the tool on this longer finger and is "
        "dropped; the scene keeps <code>open</code>, with the palm raised 10.05&#8202;mm.",
        "The joint stops come from the part geometry alone; the servos&#8217; own limits are in the driver&#8217;s table.",
    ]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def open_items(X: Data):
    items = [
        ("Which MCP link is mounted.", "The STL spaces the MCP and PIP axes 30.80&#8202;mm apart, real_v1 20.75&#8202;mm. Measure the "
         "mounted MCP link between the bearing bore and the slotted bore centres; if it is 20.75&#8202;mm, export that part over "
         "<code>assets/cad_v1/pip_mount_MCP_link.stl</code> and rerun <code>scripts/build_cad_v1_finger.py</code>, which reads any part "
         "with the same bores."),
        ("Weighed masses.", "Weigh the carriage, the three links and the tip, set <code>DENSITY</code> and <code>SERVO_MASS_KG</code> in "
         "<code>build_cad_v1_finger.py</code>, rebuild; the pads&#8217; d0 changes with the tip mass and is recomputed by the build."),
        ("Grasp key for the longer finger.", "Run <code>scripts/fit_real_v1_pose.py</code> on "
         f"<code>{REL_MODEL}/scene_screwdriver_medium.xml</code> to write <code>open_ik</code>; the fit fails if no pose holds the "
         "screwdriver on three pads at the real_v1 mounts."),
        ("Policies on the CAD finger.", "Train the servo-plant reorientation (<code>scripts/make_pad_morphology_run.py --model pads</code> on "
         "a run built from this scene) and compare held cosine and contact counts with the capsule finger; the link spheres add contacts "
         "with the tool and between fingers that the capsules lack."),
    ]
    return "<ol>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ol>"


def main():
    X = Data()
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "GLOSSARY": glossary()}
    v["PARTS"] = parts_section(X)
    v["METHOD"] = method_section(X)
    v["CHECKS"] = checks_section(X)
    v["COMPARE"] = compare_section(X)
    v["ROM"] = rom_section(X)
    v["COLLISION"] = collision_section(X)
    v["VIEWER"] = viewer_section(X)
    v["LIMITS"] = limitations(X)
    v["OPEN"] = open_items(X)
    v["LEDE"] = lede(X)
    v["FOOTER"] = ("<p>Rebuild, from the repository root: <code>.venv/bin/python scripts/build_cad_v1_finger.py --spheres "
                   f"{REL_MODEL}/spheres.json</code>, <code>scripts/cad_v1_interference.py</code>, <code>scripts/cad_v1_finger_media.py</code>, "
                   "<code>scripts/cad_v1_finger_page.py</code>. Parts: <code>assets/cad_v1/</code>. Model, data and meshes: "
                   f"<code>{REL_MODEL}/</code>.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    miss = sorted(set(re.findall(r"\[\[((?:fig|tab):[a-z0-9_]+)\]\]", t)) - set(REFS))
    if miss:
        raise SystemExit(f"unresolved references: {miss}")
    t = re.sub(r"\[\[((?:fig|tab):[a-z0-9_]+)\]\]", lambda m: str(REFS[m.group(1)]), t)
    t, n = render_tex(t)
    t = retro_style.apply(t)
    os.makedirs(D, exist_ok=True)
    open(OUT, "w").write(t)
    print(f"formulas {n}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
