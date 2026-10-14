#!/usr/bin/env python3
"""Page: MuJoCo model of the SR2 finger assembled from its CAD parts, with sphere collision geometry (2026-10-13).

Owner, 2026-10-13: "can you update the current finger model with the real CAD models? ... the attachment points (where
the joints connect) are the 10.2mm diameter hole (11.2mm diameter chamfer) to a slotted 2.6mm radius (5.2mm diameter)
hole ... this should enable us to update the collision mesh (using sphere packing, eg morphit ...) ... ideally you
build in a mujoco webasm viewer in the html docpage so i can inspect in browser". On the tip: "since we have the 1mm
sphere packed fingertips, we should keep the two separate. but i provided the fused one as a reference on how to
couple them". On the spheres: "it's ok if the collision mesh is bigger than the visual mesh. dont need to have a perfect
representation, spoeed is more important anyway". On the MCP link (22:55): the mounted link is the STL part.

Data, in assets/mjcf/experimental/20261013-cad_v1/: assembly.json (scripts/build_cad_v1_finger.py), interference.json
(scripts/cad_v1_interference.py), spheres_metrics.json and spheres_tight_metrics.json (scripts/cad_v1_spheres.py),
sphere_contacts.json (scripts/cad_v1_sphere_contacts.py), collision_bench.json (scripts/cad_v1_collision_bench.py).
Media from scripts/cad_v1_finger_media.py, the viewers from scripts/mjwasm_embed.py.

    .venv/bin/python scripts/cad_v1_finger_page.py
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

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
code{overflow-wrap:anywhere}
pre{overflow-x:auto;white-space:pre}
""" + texsvg.BLOCK_CSS + "\n"
PART_NAME = {"carriage": "carriage", "yaw_link": "yaw link", "mcp_link": "MCP link", "dip_link": "DIP link", "tip": "TPU tip",
             "servo_yaw": "yaw servo", "servo_mcp": "MCP servo", "servo_pip": "PIP servo"}
MEASURED = {j: (round(math.degrees(B.ROM[j][0])), round(math.degrees(B.ROM[j][1]))) for j in B.ROM}  # real_v1's ranges, deg


def num(x, nd=2):
    """A number with a typographic minus and no negative zero."""
    s = f"{float(x) + 0.0:.{nd}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return s.replace("-", "&#8722;")


def mm(x, nd=2):
    return f"{num(x, nd)}{HS}mm"


def deg(x, signed=False):
    return ("+" if signed and float(x) > 0 else "") + f"{num(x, 0)}&#176;"


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}{HS}%"


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


def pair_name(k):
    a, b = k.split("|")
    s = f"{PART_NAME.get(a, a)} / {PART_NAME.get(b, b)}"
    return s[0].upper() + s[1:]


class Data:
    def __init__(self):
        import mujoco
        self.mj = mujoco
        j = lambda f: json.load(open(os.path.join(MODEL, f)))  # noqa: E731
        self.asm, self.itf, self.sph, self.sph_tight = j("assembly.json"), j("interference.json"), j("spheres_metrics.json"), j("spheres_tight_metrics.json")
        self.bench, self.sc = j("collision_bench.json"), j("sphere_contacts.json")
        self.A = B.derive()
        self.rep = self.asm["report"]
        self.km = self.rep["kinematics_mm"]
        self.m_cad = mujoco.MjModel.from_xml_path(os.path.join(MODEL, "hand.xml"))
        self.m_rv1 = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets/mjcf/real_v1/real_hand.xml"))
        self.longer = self.km["yaw_to_mcp"] + self.km["mcp_to_pip"] + self.km["pip_to_tip_end"] - B.REAL_V1_REACH_MM
        self.st = self.stops()
        # dorsal extent (+x in the finger frame, q = 0) of every part
        self.dorsal = {}
        for body, parts in B.BODY_PARTS.items():
            for p in parts:
                m = self.A["meshes"][B.PART_SOURCE.get(p, p)].copy()
                m.apply_transform(self.A["T_F"][p])
                self.dorsal[p] = float(m.vertices[:, 0].max())
        self.bench_rows = {r["model"]: r for r in self.bench["rows"]}

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

    def stop(self, joint, pair, side):
        a = self.st.get(joint, {}).get(pair, np.array([]))
        a = a[a < 0] if side < 0 else a[a > 0]
        return (a.max() if side < 0 else a.min()) if len(a) else None

    def grid_hits(self):
        g = self.itf["grid"]
        tot = np.array(g["total_overlap_mm3"])
        idx = np.argwhere(tot > self.itf["threshold_mm3"])
        return [(g["mcp_deg"][i], g["pip_deg"][k]) for i, k in idx], len(g["mcp_deg"]) * len(g["pip_deg"])

    def rv1_pads_per_tip(self):
        import cad_v1_collision_bench as CB
        with tempfile.TemporaryDirectory() as td:
            m = self.mj.MjModel.from_xml_path(str(CB.real_v1_with_pads(Path(td))))
        return sum(1 for g in range(m.ngeom) if m.geom(g).name.startswith("index_pad"))

    def tip_end_offset_mm(self):
        """Largest distance between a CAD fingertip end and real_v1's at the screwdriver scene's 'open' key."""
        mj = self.mj
        out = []
        ms = [mj.MjModel.from_xml_path(os.path.join(MODEL, "scene_screwdriver_medium.xml")),
              mj.MjModel.from_xml_path(os.path.join(ROOT, "assets/mjcf/real_v1/scenes/scene_screwdriver_medium.xml"))]
        ends = []
        for m in ms:
            d = mj.MjData(m)
            mj.mj_resetDataKeyframe(m, d, [m.key(i).name for i in range(m.nkey)].index("open"))
            mj.mj_forward(m, d)
            e = {}
            for f in ("thumb", "index", "middle"):
                b = m.body(f"{f}_pip_frame").id
                e[f] = d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0, 0, -self.km["pip_to_tip_end"] / 1000])
            ends.append(e)
        for f in ends[0]:
            out.append(1000 * np.linalg.norm(ends[0][f] - ends[1][f]))
        return max(out)


GLOSSARY = [
    ("SR2, real_v1", "SR2 is the three-finger self-reconfigurable hand of <code>paper/hand_iros26-4.pdf</code>; real_v1 is its "
     "MuJoCo model in use, <code>assets/mjcf/real_v1/real_hand.xml</code>, whose fingers are three capsules and a tip sphere of "
     "10.55&#8202;mm radius."),
    ("Carriage, gantry", "Each finger hangs from a carriage that a stepper-driven gantry moves in the plane of the palm; the carriage "
     "holds the yaw servo and the yaw bearing."),
    ("Yaw, MCP and PIP joints", "The finger&#8217;s three revolute joints, named as in real_v1: yaw swings the finger sideways about x; MCP "
     "(metacarpophalangeal) and PIP (proximal interphalangeal) flex it about y. Angles are in degrees; at zero the finger is straight."),
    ("Yaw link, MCP link, DIP link", "The parts that the yaw, MCP and PIP joints turn. The DIP link holds the TPU tip; the finger has no "
     "DIP joint."),
    ("TPU, PLA, SCS0009", "The fingertip is printed in thermoplastic polyurethane (TPU), the links in polylactic acid (PLA). The SCS0009 "
     "is the Feetech serial-bus servo at each joint."),
    ("Bearing bore", "The 10.2&#8202;mm bore, 3.5&#8202;mm deep with a 0.5&#8202;mm chamfer to 11.2&#8202;mm, in the parent part of a joint. "
     "It holds an MR105 bearing (5&#8202;&#215;&#8202;10&#8202;&#215;&#8202;4&#8202;mm) flush in a 4.0&#8202;mm wall, the bearing wall."),
    ("Slotted bore", "The 5.2&#8202;mm bore in the child part of a joint, with a slot that removes 40&#176; of its circumference. The shaft "
     "through the bearing passes through it."),
    ("Joint axis, axial position", "The common axis of the parent&#8217;s bearing bore and the child&#8217;s slotted bore. An axial position "
     "is a distance along the axis from the bearing bore&#8217;s centre, positive toward the servo, in mm."),
    ("Clevis", "A child part with two arms that straddle the parent: one arm has the slotted bore, the other is screwed to the "
     "parent&#8217;s servo horn (the horn-side arm)."),
    ("Pilot holes, tab face", "The two 1.8&#8202;mm holes in the parent for the screws through the servo&#8217;s mounting tabs; the "
     "pilot-hole pitch is the distance between their axes. The tab face is the face of the tabs nearest the servo&#8217;s base."),
    ("Horn stack", "Axial distance from the top of the servo&#8217;s round boss to the inner face of the child&#8217;s horn-side arm, in mm: "
     "the room for the horn, which is not among the STLs."),
    ("Yaw centre", "The point on the yaw axis nearest the MCP axis: the mount position of real_v1 and the origin of each finger in the "
     "model. The finger frame has x along the yaw axis toward the dorsal side and z up."),
    ("Registration residual", "Largest distance between corresponding vertices after a part is registered by ICP (iterative closest "
     "point) onto its copy in the fused STL, in mm."),
    ("Penetration volume, penetration area", "Two parts are voxelized at 0.3&#8202;mm. The penetration volume is the volume of one part&#8217;s "
     "voxels lying more than one voxel inside the other, in mm&#179;, the larger of the two directions; the penetration area is the area "
     "of boundary voxels so placed, in mm&#178;. Flush faces, such as a servo in a pocket of its own width, give zero. A collision is a "
     "volume above 0.1&#8202;mm&#179; or an area above 0.1&#8202;mm&#178;."),
    ("Joint stop", "The joint angle at which two CAD parts first collide, with the other joints at zero."),
    ("MorphIt-B, MorphIt-S", "The balanced and surface presets of the MorphIt sphere packer: MorphIt-B weights its coverage loss 1500 and "
     "its protrusion loss 1, MorphIt-S weights coverage 0.01 and protrusion 1000 (<code>src/config.py</code> of MorphIt)."),
    ("Coverage", "Fraction of 20&#8202;000 points sampled uniformly inside a part&#8217;s mesh that lie inside at least one of its spheres."),
    ("Protrusion", "Distance to the mesh from points on the surface of the sphere union that lie outside the mesh, in mm; p95 is its 95th "
     "percentile. A protrusion makes a contact where the part has no material."),
    ("Exposed surface", "Fraction of 20&#8202;000 points sampled on a part&#8217;s surface that lie more than 1&#8202;mm outside its sphere "
     "union: places where an object can press 1&#8202;mm into the part without a contact."),
    ("Pads", "Spheres of 0.75&#8202;mm radius at 1&#8202;mm spacing on the palmar, end and side faces of the TPU tip, built by "
     "<code>scripts/fingertip_geometry.py</code> through <code>reorient_backends.replace_tips</code>, the repository&#8217;s default "
     "fingertip. \\(K = E A / h\\) is the stiffness of one pad (median over the pads), with \\(E = 10\\) MPa, \\(A\\) the pad&#8217;s share of "
     "the tip&#8217;s surface and \\(h = 8.5\\) mm the depth from the palmar face to the block&#8217;s centroid; \\(d_0\\) is the pad&#8217;s solimp "
     "impedance, set so that a pad pressed on the screwdriver has stiffness \\(K\\)."),
    ("Colliding geom", "A MuJoCo geom with non-zero contype or conaffinity. Geom groups 0 to 5 are MuJoCo&#8217;s display classes, toggled in "
     "the viewers; qpos is the vector of joint positions."),
    ("Step time", "Wall time of one 2&#8202;ms step of the screwdriver scene on one core of this workstation, best of three runs of 1000 "
     "steps, in &#181;s."),
    ("Open and close conditions", "&#8216;Open&#8217; holds the scene&#8217;s &#8216;open&#8217; key (yaw &#8722;24.6&#176;, MCP 1.4&#176;, PIP "
     "44.2&#176; on the index finger), where no finger touches the tool. &#8216;Close&#8217; starts there and raises every MCP and PIP command by "
     "0.9&#8202;rad over 1&#8202;s, clamped to the joint range, then holds for 1&#8202;s. The &#8216;open&#8217; key of <code>hand.xml</code> and "
     "<code>finger.xml</code> is a different pose, MCP and PIP at 32&#176;, as in <code>real_hand.xml</code>."),
    ("Tip force share", "Fraction of the normal force between the fingers and the tool, summed over the 1000 steps of a condition, that acts "
     "on the <code>&lt;f&gt;_tip</code> bodies, which carry the pads."),
    ("Held cosine", "Mean cosine between the tool&#8217;s axis and its goal direction over the hold at the end of an episode "
     "(<code>held_cos_tail</code> in <code>src/morphohand/rl/trajectory_health.py</code>)."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def lede(X: Data):
    km, s = X.km, X.sph
    n = sum(r["spheres"] for r in s["parts"].values())
    br = X.bench_rows
    dmax = max(abs(X.stop("yaw", "carriage|yaw_link", -1) - MEASURED["yaw"][0]), abs(MEASURED["yaw"][1] - X.stop("yaw", "carriage|yaw_link", 1)),
               abs(MEASURED["mcp"][1] - X.stop("mcp", "yaw_link|mcp_link", 1)))
    y_sph = X.sc["sweeps"]["yaw"]["index_mcp_frame / index_mount"]
    return (f"A MuJoCo model of the SR2 finger, assembled from the STLs of its seven parts, keeps real_v1&#8217;s joint names, axes and "
            f"mounts and its yaw-to-MCP ({mm(km['yaw_to_mcp'])}) and PIP-to-tip ({mm(km['pip_to_tip_end'])}) lengths. Its MCP link, the "
            f"part mounted on the hand, spaces the MCP and PIP axes {mm(km['mcp_to_pip'])} apart against real_v1&#8217;s 20.75&#8202;mm, so "
            f"the finger is {mm(X.longer)} longer. The parts collide within {dmax:.0f}&#176; of the measured yaw and MCP flexion limits, "
            f"except that the MCP link meets the carriage from {deg(X.stop('yaw', 'carriage|mcp_link', -1))} of yaw. The collision geometry "
            f"is {n} MorphIt spheres per finger and the {X.asm['pads']['n_pads']['index']} pads of the TPU tip. The spheres stand up to "
            f"{mm(max(r['protrusion_max_mm'] for r in s['parts'].values()), 1)} off the parts and stop the yaw at {deg(y_sph[1])}, and they "
            f"raise the step time of the screwdriver scene from {br['CAD finger, pads only']['open']['us_per_step']:.1f} to "
            f"{br['CAD finger, balanced spheres and pads']['open']['us_per_step']:.1f}&#8202;&#181;s with no finger in contact.")


def byline():
    return ("Assembly <code>scripts/build_cad_v1_finger.py</code> with <code>src/morphohand/geometry/cad_features.py</code> &#183; "
            "stops <code>scripts/cad_v1_interference.py</code> &#183; spheres <code>scripts/cad_v1_spheres.py</code> (MorphIt) &#183; "
            "contacts and cost <code>scripts/cad_v1_sphere_contacts.py</code>, <code>scripts/cad_v1_collision_bench.py</code> &#183; "
            "media <code>scripts/cad_v1_finger_media.py</code> &#183; viewer <code>scripts/mjwasm_embed.py</code> &#183; builder "
            "<code>scripts/cad_v1_finger_page.py</code> &#183; model <code>assets/mjcf/experimental/20261013-cad_v1/</code>")


def compare_section(X: Data):
    km = X.km
    mc, mr = X.m_cad, X.m_rv1

    def bm(m, name):
        return 1000 * m.body_mass[m.body(name).id]

    reach_c = km["yaw_to_mcp"] + km["mcp_to_pip"] + km["pip_to_tip_end"]
    rows = [["Yaw axis to MCP axis (mm)", "20.75", num(km["yaw_to_mcp"])],
            ["MCP axis to PIP axis (mm)", "20.75", num(km["mcp_to_pip"])],
            ["PIP axis to the end of the tip (mm)", "37.16", num(km["pip_to_tip_end"])],
            ["Yaw centre to the end of the tip, straight (mm)", num(B.REAL_V1_REACH_MM), num(reach_c)]]
    for b, t in (("index_mount", "carriage and yaw servo"), ("index_yaw_frame", "yaw link and MCP servo"),
                 ("index_mcp_frame", "MCP link and PIP servo"), ("index_pip_frame", "DIP link"), ("index_tip", "TPU tip")):
        rv = f"{bm(mr, b):.2f}" + (" (placeholder)" if b == "index_mount" else "")
        rows.append([f"Mass, {t} (g)", rv, f"{bm(mc, b):.2f}"])
    rv1_link = sum(1 for i in range(mr.ngeom) if mr.geom_bodyid[i] in [mr.body(f"index_{x}").id for x in ("yaw_frame", "mcp_frame", "pip_frame")])
    cad_link = sum(1 for i in range(mc.ngeom) if mc.geom_group[i] == 3 and mc.geom(i).name.startswith("index_"))
    rows.append(["Link collision geoms per finger", f"{rv1_link} capsules", f"{cad_link} spheres"])
    rows.append(["Tip pads per finger", str(X.rv1_pads_per_tip()), str(X.asm["pads"]["n_pads"]["index"])])
    servo_x = max(v for p, v in X.dorsal.items() if p.startswith("servo"))
    link_p = max((p for p in X.dorsal if not p.startswith("servo")), key=lambda p: X.dorsal[p])
    return (img("assembly.png", "The model at zero joint angles beside the owner&#8217;s CAD views of the same finger "
                "(<code>assets/cad_v1/diagram.png</code>), seen from +y and from &#8722;y; the owner&#8217;s &#8722;y view is flexed at the PIP "
                "joint.", "assembly")
            + f"<p>The model reproduces the owner&#8217;s CAD views (Figure&#160;[[fig:assembly]]). Figure&#160;[[fig:compare]] draws "
            "real_v1&#8217;s capsule finger on the same mount. The two fingers agree from the yaw axis to the MCP axis and from the PIP axis to "
            f"the tip; the CAD MCP link is {mm(km['mcp_to_pip'] - 20.75)} longer, and on 2026-10-13 the owner confirmed that this STL is the link "
            f"mounted on the hand. On the dorsal side the servos reach {mm(servo_x, 1)} from the finger axis and the {PART_NAME[link_p]} "
            f"{mm(X.dorsal[link_p], 1)}; the capsules have a radius of 10.55&#8202;mm. Table&#160;[[tab:compare]] compares lengths, masses and "
            "collision geometry.</p>"
            + img("compare.png", "CAD finger (purple, servos black, TPU tip tan) and real_v1&#8217;s capsule finger (translucent teal) on "
                  "the same mount at three joint configurations, seen along the flexion axes.", "compare")
            + table(["Quantity", "real_v1", "CAD model"], rows,
                    "Lengths, masses and collision geometry of the index finger. Real_v1&#8217;s masses come from its capsules and tip "
                    "sphere at 1000&#8202;kg/m&#179;, and its mount has a 1&#8202;g placeholder mass. The model&#8217;s come from the CAD "
                    "volumes at solid PLA (1240&#8202;kg/m&#179;) and TPU (1210&#8202;kg/m&#179;) density and 13.1&#8202;g per servo. The "
                    "pads are those that <code>reorient_backends.replace_tips</code> builds on each model&#8217;s tip block.",
                    key="compare"))


def rom_chart(X: Data):
    sweeps = X.itf["sweeps"]
    thr = X.itf["threshold_mm3"]
    title = {"yaw": "Yaw", "mcp": "MCP", "pip": "PIP"}
    colors = ["#c0392b", "#2c6fbb", "#27864a", "#8e44ad", "#d35400"]
    Wp, Hp, ml, mb, mt = 268, 220, 52, 44, 30
    W = 3 * Wp + 10
    parts = [f'<svg viewBox="0 0 {W} {Hp + 70}" xmlns="http://www.w3.org/2000/svg" font-family="Times, serif" font-size="14">']
    all_pairs, ymax = [], 1.0
    for j in ("yaw", "mcp", "pip"):
        for k in sweeps[j]["overlap_mm3"][0]:
            v = max(r[k] for r in sweeps[j]["overlap_mm3"])
            if v > thr:
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
        a, b = MEASURED[j]
        parts.append(f'<rect x="{X_(a):.1f}" y="{mt}" width="{X_(b) - X_(a):.1f}" height="{ph}" fill="#e9eef5"/>')
        parts.append(f'<rect x="{x0}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#000" stroke-width="1"/>')
        for e in range(-1, int(math.log10(ytop)) + 1):
            y = Y_(10 ** e)
            parts.append(f'<line x1="{x0 - 4}" x2="{x0}" y1="{y:.1f}" y2="{y:.1f}" stroke="#000"/>')
            if c == 0:
                parts.append(f'<text x="{x0 - 7}" y="{y + 5:.1f}" text-anchor="end">{10 ** e:g}</text>')
        for t in np.arange(math.ceil(lo / 30) * 30, hi + 1, 30):
            parts.append(f'<line x1="{X_(t):.1f}" x2="{X_(t):.1f}" y1="{mt + ph}" y2="{mt + ph + 4}" stroke="#000"/>')
            parts.append(f'<text x="{X_(t):.1f}" y="{mt + ph + 19}" text-anchor="middle">{num(t, 0)}</text>')
        parts.append(f'<text x="{x0 + pw / 2:.1f}" y="{mt + ph + 38}" text-anchor="middle">{title[j]} angle (deg)</text>')
        parts.append(f'<text x="{x0 + pw / 2:.1f}" y="{mt - 10}" text-anchor="middle">{title[j]} sweep, other joints at 0</text>')
        for i, k in enumerate(all_pairs):
            v = np.array([r[k] for r in sweeps[j]["overlap_mm3"]])
            on = v > thr
            # one polyline per run of colliding angles; a single colliding angle is a dot
            for run in np.split(np.arange(len(v)), np.flatnonzero(np.diff(on.astype(int))) + 1):
                if not on[run[0]]:
                    continue
                if len(run) == 1:
                    parts.append(f'<circle cx="{X_(q[run[0]]):.1f}" cy="{Y_(v[run[0]]):.1f}" r="2.5" fill="{colors[i % 5]}"/>')
                    continue
                pts = " ".join(f"{X_(q[t]):.1f},{Y_(v[t]):.1f}" for t in run)
                parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[i % 5]}" stroke-width="2"/>')
    parts.append(f'<text x="14" y="{mt + (Hp - mb - mt) / 2:.1f}" transform="rotate(-90 14 {mt + (Hp - mb - mt) / 2:.1f})" '
                 'text-anchor="middle">Penetration area (mm&#178;)</text>')
    ly = Hp + 40
    x = ml
    for i, k in enumerate(all_pairs):
        parts.append(f'<line x1="{x}" x2="{x + 22}" y1="{ly}" y2="{ly}" stroke="{colors[i % 5]}" stroke-width="2"/>')
        parts.append(f'<text x="{x + 28}" y="{ly + 5}">{pair_name(k)}</text>')
        x += 40 + 8 * len(pair_name(k))
    parts.append(f'<rect x="{x}" y="{ly - 8}" width="22" height="14" fill="#e9eef5"/><text x="{x + 28}" y="{ly + 5}">Measured range of motion</text>')
    parts.append("</svg>")
    return "".join(parts)


def rom_section(X: Data):
    hits, n = X.grid_hits()
    worst = X.itf["grid"]["worst_by_pair_mm3"]
    mcps = sorted({a for a, _ in hits})
    yaw_neg, yaw_pos = X.stop("yaw", "carriage|yaw_link", -1), X.stop("yaw", "carriage|yaw_link", 1)
    mcp_stop = X.stop("mcp", "yaw_link|mcp_link", 1)
    car_mcp = X.stop("yaw", "carriage|mcp_link", -1)
    dmax = max(abs(yaw_neg - MEASURED["yaw"][0]), abs(MEASURED["yaw"][1] - yaw_pos), abs(MEASURED["mcp"][1] - mcp_stop))
    pip_free = "pip" not in X.st
    out_of_range = all(a > MEASURED["mcp"][1] for a in mcps)
    txt = (f"<p>The parts collide within {dmax:.0f}&#176; of the measured limits (Figure&#160;[[fig:rom]]). With the other joints at zero, the "
           f"yaw link meets the carriage at {deg(yaw_neg)} and {deg(yaw_pos, True)} of yaw against the measured &#177;{MEASURED['yaw'][1]}&#176;, and "
           f"the yaw link meets the MCP link at {deg(mcp_stop)} of MCP flexion against the measured {MEASURED['mcp'][1]}&#176;. The MCP link "
           f"meets the carriage from {deg(car_mcp)} of yaw, {abs(car_mcp - MEASURED['yaw'][0]):.0f}&#176; inside the measured range."
           + (f" The PIP joint is free over its measured {deg(MEASURED['pip'][0])} to {deg(MEASURED['pip'][1])}." if pip_free else "")
           + f" Over the MCP&#8202;&#215;&#8202;PIP grid at 3&#176;, {len(hits)} of {n} configurations collide, "
           + (f"all at MCP {', '.join(deg(a) for a in mcps)}, outside the measured range" if out_of_range else f"at MCP {', '.join(deg(a) for a in mcps)}")
           + f"; the only colliding pair there is the {' and the '.join(PART_NAME[p] for p in next(iter(worst)).split('|'))}.</p>")
    return txt + figure(rom_chart(X), "Penetration area between CAD parts against joint angle, one joint at a time with the other joints at "
                        "zero, on a log scale; values at or below 0.1&#8202;mm&#178; are not drawn. The shaded band is real_v1&#8217;s joint "
                        "range, the range of motion measured on the hand.", key="rom", cls="diagram")


def sphere_table(s, key, caption):
    names = {"carriage": "Carriage", "servo_yaw": "Yaw servo", "yaw_link": "Yaw link", "servo_mcp": "MCP servo",
             "mcp_link": "MCP link", "servo_pip": "PIP servo", "dip_link": "DIP link"}
    rows = []
    for p, r in s["parts"].items():
        n = f"{r['spheres']}" + (f" ({r['clipped']} clipped)" if r.get("clipped") else "")
        rows.append([names.get(p, p), n, f"{r['coverage']:.2f}", f"{r['protrusion_p95_mm']:.1f}", f"{r['protrusion_max_mm']:.1f}",
                     f"{100 * r['gap_1mm']:.0f}", f"{r['seconds']:.0f}"])
    n_tot = sum(r["spheres"] for r in s["parts"].values())
    rows.append(["One finger (total; largest part protrusion)", n_tot, "", f"{max(r['protrusion_p95_mm'] for r in s['parts'].values()):.1f}",
                 f"{max(r['protrusion_max_mm'] for r in s['parts'].values()):.1f}", "", f"{sum(r['seconds'] for r in s['parts'].values()):.0f}"])
    return table(["Part", "Spheres", "Coverage", "Protrusion p95 (mm)", "Protrusion max (mm)", "Exposed surface (%)", "Fit time (s)"],
                 rows, caption, key=key), n_tot


def collision_section(X: Data):
    pads = X.asm["pads"]
    s, t = X.sph, X.sph_tight
    br = X.bench_rows
    sc = X.sc
    yaw = sc["sweeps"]["yaw"].get("index_mcp_frame / index_mount")
    mcp = sc["sweeps"]["mcp"].get("index_mcp_frame / index_mount")
    car_mcp = X.st["yaw"]["carriage|mcp_link"]
    grid_mcps = sorted({a for cells in sc["grid"].get("cells", {}).values() for a, _ in cells})
    tab_b, n_b = sphere_table(s, "spheres", "Balanced MorphIt spheres (MorphIt-B, the model&#8217;s collision geometry) per part, measured "
                              "against the part&#8217;s CAD mesh. Fit time is the MorphIt run, including loading the mesh.")
    tab_t, n_t = sphere_table(t, "spheres_tight", "Tight MorphIt spheres (MorphIt-S, <code>spheres_tight.json</code>) per part, measured the "
                              "same way.")
    cov = [r["coverage"] for p, r in s["parts"].items() if p != "dip_link"]
    dip, dip0 = s["parts"]["dip_link"], s["unclipped"]["dip_link"]
    past = dip0["past_pad_faces_mm"]["fitted"]
    us = lambda k, c: br[k][c]["us_per_step"]  # noqa: E731
    bal, tight, pads_only = "CAD finger, balanced spheres and pads", "CAD finger, tight spheres and pads", "CAD finger, pads only"
    close = br[bal]["close"]
    first = close["first_tool_contact_s"]
    others = sorted((v, k) for k, v in first.items() if k != "tip")
    rows = []
    for k, r in br.items():
        share = r["close"]["tip_force_share"]
        rows.append([k[0].upper() + k[1:], r["ngeom_colliding"], f"{r['open']['us_per_step']:.1f}", f"{r['close']['us_per_step']:.1f}",
                     r["close"]["ncon_max"], "" if share is None else f"{share:.3f}"])
    rv1_touch = br["real_v1 capsules and pads"]["close"]["first_tool_contact_s"]
    out = [
        "<p>Each link and each servo is packed by MorphIt (Nechyporenko et al., 2025; code <a href=\"https://github.com/HIRO-group/MorphIt-1\">"
        "HIRO-group/MorphIt-1</a>, commit 5f65299, MIT licence), which moves sphere centres and radii by gradient descent on weighted losses for "
        "coverage, overlap, protrusion past the surface and distance to the surface. The owner put collision speed before fidelity on "
        f"2026-10-13, so the model uses the balanced preset, MorphIt-B: {n_b} spheres per finger, {s['iterations']} iterations, seed "
        f"{s['seed']} (Table&#160;[[tab:spheres]]). They cover {100 * min(cov):.0f} to {100 * max(cov):.0f}&#8202;% of each part&#8217;s volume "
        f"other than the DIP link&#8217;s and stand up to {mm(max(r['protrusion_max_mm'] for r in s['parts'].values()), 1)} off its surface. "
        f"The surface preset, MorphIt-S, at {n_t} spheres per finger stays within "
        f"{mm(max(r['protrusion_max_mm'] for r in t['parts'].values()), 1)} of the surface and leaves the interior hollow "
        "(Table&#160;[[tab:spheres_tight]]); the builder takes either file with <code>--spheres</code>. Link and servo are packed apart and "
        "share their body&#8217;s spheres, because the servo sits flush in its pocket and a signed distance computed on the two meshes "
        "together has the wrong sign near their shared faces.</p>",
        "<p>Below the top of the tip the DIP link is a 3&#8202;mm dorsal plate flush with the TPU&#8217;s end and side faces, and above the tip "
        "a palmar ledge lies flush with the pad face. As fitted, its spheres reached "
        f"{mm(past['end'])} past the tip&#8217;s end face, {mm(past['palmar'])} past its palmar face and {mm(past['side'])} past a side face, "
        "so an object pressed on the pads met rigid spheres first. <code>clip_to_pads</code> in <code>scripts/cad_v1_spheres.py</code> "
        "replaces each DIP-link sphere by the largest ball inside both the sphere and the half-spaces "
        f"{mm(s['pad_clearance_mm'], 1)} behind the tip&#8217;s palmar and end faces, and behind its side faces for spheres centred below the "
        f"tip&#8217;s top. It shrinks {dip['clipped']} of {dip['fitted']} balanced spheres; the DIP link&#8217;s coverage falls from "
        f"{dip0['coverage']:.2f} to {dip['coverage']:.2f}, and its exposed surface grows from {pct(dip0['gap_1mm'])} to {pct(dip['gap_1mm'])}.</p>",
        tab_b, tab_t,
        img("spheres.png", "Collision geometry of the index finger with MCP and PIP at 32&#176;: CAD meshes, the balanced MorphIt spheres "
            "with the tip pads, both, and the dorsal side.", "spheres"),
        "<p>Spheres of neighbouring bodies overlap at every bearing; MuJoCo skips contacts between a body and its parent, so the overlaps make "
        "no contacts. Of the other body pairs within one finger, only the carriage and the MCP link meet over the measured ranges of motion, "
        f"so the model excludes the rest. That pair&#8217;s spheres touch from {deg(yaw[0])} to {deg(yaw[1])} of yaw, where the parts meet "
        f"from {deg(car_mcp.max())} to the yaw limit, and at {deg(mcp[0])} of MCP flexion, where the parts do not meet. The PIP sweep "
        f"has no sphere contact, and over the MCP&#8202;&#215;&#8202;PIP grid at 3&#176; the spheres touch only at MCP "
        f"{', '.join(deg(a) for a in grid_mcps)}, as the parts do. Fingers collide with each other, the tool and the floor; as in real_v1, no "
        "finger body collides with the palm plate.</p>",
        f"<p>Table&#160;[[tab:bench]] gives the step time of the screwdriver scene in the two conditions. Holding the &#8216;open&#8217; key, "
        f"where no finger touches the tool, the balanced spheres raise the step time from {us(pads_only, 'open'):.1f}&#8202;&#181;s with pads "
        f"only to {us(bal, 'open'):.1f}&#8202;&#181;s, and the tight spheres to {us(tight, 'open'):.1f}&#8202;&#181;s. In the close ramp the "
        f"balanced spheres cost {us(bal, 'close'):.1f}&#8202;&#181;s per step with up to {close['ncon_max']} contacts. The pads touch the tool "
        f"first, after {first['tip']:.2f}&#8202;s; the ramp then curls the DIP links until their dorsal plates face the tool, which they "
        f"touch from {others[0][0]:.2f}&#8202;s, and the finger bodies other than the tips carry {pct(1 - close['tip_force_share'])} of the "
        "finger-to-tool normal force."
        + (" Real_v1&#8217;s shorter fingers do not reach the tool in either condition." if not rv1_touch else "") + "</p>",
        table(["Scene", "Colliding geoms", "Step time, open (&#181;s)", "Step time, close (&#181;s)", "Contacts, close (largest)",
               "Tip force share, close"], rows,
              f"Step time of the screwdriver scene under four collision models of the fingers, MuJoCo {X.bench['mujoco']} on one core of "
              "this workstation, best of three runs of 1000 steps (<code>scripts/cad_v1_collision_bench.py</code>). Contacts include the "
              "tool&#8217;s two contacts with the floor; the tip force share is empty where no finger touches the tool.",
              key="bench", text_cols=(0,)),
        f"<p>The tip keeps the repository&#8217;s pad model. <code>reorient_backends.replace_tips(root, \"tpu2.7\", \"pads\", ...)</code> "
        f"runs on a copy of real_v1&#8217;s <code>scene_screwdriver_medium.xml</code> with the CAD fingers in it, with "
        f"<code>fingertip_geometry</code> pointed at the CAD insert in that process only: {mm(pads['block_mm'][2])} long instead of "
        f"22.00&#8202;mm. It gives {pads['n_pads']['index']} pads per tip, \\(K = {pads['pad_K']:.1f}\\) N/m and "
        f"\\(d_0 = {pads['pad_d0']['index']:.3f}\\) against the screwdriver. The same pads are copied into <code>finger.xml</code> and "
        "<code>hand.xml</code>. The DIP link and the tip stay separate bodies; the fused STL fixes where the tip sits. The scene&#8217;s keys "
        f"and the palm of <code>hand.xml</code> are raised {mm(pads['palm_raised_mm'])}, which puts the fingertips within "
        f"{mm(X.tip_end_offset_mm(), 1)} of real_v1&#8217;s at the scene&#8217;s &#8216;open&#8217; key.</p>",
        img("hand.png", "The three fingers on the real_v1 mounts at the &#8216;open&#8217; key of <code>hand.xml</code> (MCP and PIP at "
            "32&#176;): CAD meshes, and the collision spheres with the pads.", "hand"),
    ]
    return "".join(out)


def viewer_section(X: Data):
    v1 = mjwasm_embed.embed(os.path.join(MODEL, "finger.xml"), element_id="finger", height_px=620,
                            title="One finger, finger.xml")
    v2 = mjwasm_embed.embed(os.path.join(MODEL, "hand.xml"), element_id="hand", height_px=560, title="Three fingers, hand.xml",
                            include_runtime=False, keyframe="open")
    cmds = "\n".join(f".venv/bin/python -m mujoco.viewer --mjcf={REL_MODEL}/{f}" for f in ("finger.xml", "hand.xml", "scene_screwdriver_medium.xml"))
    return ("<p>The two viewers below run MuJoCo 3.14.0 compiled to WebAssembly on the model files themselves, inlined in this page. "
            "Drag to orbit and right-drag to pan; click the view, then scroll to zoom. The sliders set the joint angles, and Play steps the "
            "physics with the position actuators following the sliders. In all three model files geom group 2 holds the CAD meshes, group 3 "
            "the MorphIt spheres and group 4 the tip pads; the checkboxes show and hide them.</p>"
            + v1 + v2
            + "<p>The same files open in MuJoCo&#8217;s native viewer from the repository root:</p>"
            + f"<pre>{cmds}</pre>"
            + "<p>The native viewer shows groups 0 to 2; keys 3 and 4 toggle the spheres and the pads. The scene opens at zero joint positions, "
            "where the fingers stand below the floor; its &#8216;open&#8217; key is loaded from the viewer&#8217;s Simulation panel.</p>"
            + film("sweep", "One joint sweep shown three ways at the same joint angles: the CAD meshes, the collision spheres and pads, and "
                   "real_v1&#8217;s capsules. MCP and PIP flex to their upper limits and extend to their lower limits; then the yaw runs from "
                   "&#8722;85 to 85&#176; while MCP and PIP flex to 32&#176;.", "sweep"))


def parts_section(X: Data):
    A = X.A
    rows = []
    names = {"carriage": ("Carriage", "carriage.stl"), "yaw_link": ("Yaw link", "mcp_mount_YAW-link.stl"),
             "mcp_link": ("MCP link", "pip_mount_MCP_link.stl"), "dip_link": ("DIP link", "fingertip_mount_DIP_link.stl")}

    def pt(p):
        return "(" + ", ".join(num(v) for v in p) + ")"

    for k, (title, f) in names.items():
        pi, ci = A["parent_info"].get(k), A["child_info"].get(k)
        bear = f"{pt(pi['bore'].point)}, axis {pt(pi['bore'].axis)}" if pi else "None"
        slot = f"{pt(ci['slot'].point)}, axis {pt(ci['slot'].axis)}" if ci else "None"
        pil = num(np.linalg.norm(B.perp(pi["pilots"][0].point - pi["pilots"][1].point, pi["a"]))) if pi else "&#8211;"
        rows.append([f"{title} <code>{f}</code>", bear, slot, pil])
    t = table(["Part (file in <code>assets/cad_v1/</code>)", "Bearing bore centre and axis (part frame, mm)",
               "Slotted bore centre and axis (part frame, mm)", "Pilot-hole pitch (mm)"], rows,
              "Bores found in each part. Coordinates are in each STL&#8217;s own frame, the carriage&#8217;s after removing its export "
              f"rotation. The pilot-hole pitch matches the {num(X.rep['servo']['tab_pitch'])}&#8202;mm between the servo&#8217;s tab holes.",
              key="bores", text_cols=(0, 1, 2))
    sv = X.rep["servo"]
    return (
        "<p>Seven STLs describe one finger: the carriage that rides on the gantry, the yaw link, the MCP link, the DIP link, the TPU tip, "
        "the SCS0009 servo, and the DIP link and tip fused in their assembled position. Every part is in millimetres and in its own "
        "frame. The builder extracts three kinds of feature from them, shown in Figure&#160;[[fig:parts]] and listed in "
        "Table&#160;[[tab:bores]]: the bearing bores, the slotted bores and the servo seats. The servo STL gives the output spline "
        f"(4.0&#8202;mm), the case base, the tab holes {mm(sv['tab_pitch'])} apart with their tab face {mm(sv['tab_face'])} above the base, "
        f"the top of its round boss at {mm(sv['boss_top'])} and the end of the spline at {mm(sv['height'])}; the spline lies "
        f"{mm(sv['spline_offset_from_tab_centre'])} from the midpoint of the tab holes.</p>"
        + img("parts.png", "The CAD parts with the bores the builder extracts: red, the 10.2&#8202;mm bearing bore; blue, the 5.2&#8202;mm "
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
        "<p>These rules fix every transform. The joint axis is where the parent&#8217;s bearing bore and the child&#8217;s slotted bore "
        "coincide. Along that axis the inner face of the child&#8217;s slotted arm lies on the outer face of the bearing wall, which is flush "
        "with the bearing. Each servo sits with its output on the bearing axis, its base on the inner face of the bearing wall, and its tab "
        "holes turned toward the midpoint of the pilot holes. At zero joint angles the finger is straight: the yaw link runs from the yaw axis "
        "to the MCP axis straight down, the MCP link continues it to the PIP axis, and the DIP link continues parallel to the tip&#8217;s pad "
        "face; the carriage hangs with its long arm up toward the gantry, as in the owner&#8217;s CAD views (Figure&#160;[[fig:assembly]]). "
        "The TPU tip takes its pose on the DIP link from <code>fingertip_and_DIP_link.stl</code>, registered by vertex ICP.</p>"
        f"<p>The carriage STL is rotated {rot:.3f}&#176; about its x axis: its bores and its largest planar faces are tilted by the same "
        "angle, so the builder rotates it back before reading it. Real_v1&#8217;s conventions are kept: the finger hangs along &#8722;z from "
        "the yaw centre at the real_v1 mounts, the yaw axis is +x, the flexion axes are +y on the index and middle fingers and &#8722;y on "
        "the thumb, the pads face the opposing finger, and body names, joint names, qpos layout and actuators match "
        "<code>real_hand.xml</code>. The thumb is the same parts turned 180&#176; about the vertical.</p>")


def checks_section(X: Data):
    rep, km = X.rep, X.km
    sv = rep["servos"]
    j = rep["joints"]
    resid = max(max(v["tab_hole_to_pilot_axis_mm"]) for v in sv.values())
    rest_max = max(X.itf["rest"].values())
    reg = rep["parts"]["tip_registration_residual_mm"]
    rows = [
        ["Yaw axis to MCP axis", mm(km["yaw_to_mcp"]), "20.75&#8202;mm in real_v1 and in <code>paper/hand_iros26-4.pdf</code>"],
        ["MCP axis to PIP axis", mm(km["mcp_to_pip"]), "20.75&#8202;mm in real_v1; the STL is the mounted part"],
        ["PIP axis to the end of the tip", mm(km["pip_to_tip_end"]), "37.16&#8202;mm in real_v1"],
        ["Finger axis to the pad face", mm(-km["tip_bounds"][0][0]), "10.55&#8202;mm, real_v1&#8217;s tip radius"],
        ["Servo tab holes to pilot-hole axes, largest of six", mm(resid, 3), "0"],
        ["Servo tab face, axial position", mm(sv["mcp"]["tab_face_axial_mm"]),
         f"Far end of the pilot holes, {mm(sv['mcp']['pilot_hole_far_ends_axial_mm'][0])}"],
        ["Horn stack at the yaw, MCP and PIP joints", f"{num(j['yaw']['boss_to_horn_side_mm'], 1)}, {num(j['mcp']['boss_to_horn_side_mm'], 1)} "
         f"and {mm(j['pip']['boss_to_horn_side_mm'], 1)}", "Equal at the yaw and MCP joints"],
        ["Registration residual of the DIP link and the tip on the fused STL", f"{num(reg['dip'], 3)} and {mm(reg['tip'], 5)}", "0"],
        ["Penetration volume at zero joint angles, largest of 28 part pairs", f"{num(rest_max, 3)}{HS}mm&#179;", "0"],
    ]
    return ("<p>The rules fix only the direction of the pilot-hole midpoint, so the distance from all six tab holes to the pilot-hole axes "
            f"tests the {mm(rep['servo']['spline_offset_from_tab_centre'])} spline offset and the {mm(rep['servo']['tab_pitch'])} pitch "
            f"(Table&#160;[[tab:checks]]). With its base on the bearing wall, each servo&#8217;s tab face lies at "
            f"{mm(sv['mcp']['tab_face_axial_mm'])}, the far end of both pilot holes. The yaw and MCP joints, built on different parts, give "
            "the same horn stack.</p>"
            + table(["Quantity", "Model", "Reference"], rows, "Assembly checks.", key="checks", text_cols=(0, 2)))


def limitations(X: Data):
    j = X.rep["joints"]
    y = X.sc["sweeps"]["yaw"]["index_mcp_frame / index_mount"]
    items = [
        "Masses come from the CAD volumes at solid PLA and TPU density and the SCS0009&#8217;s listed 13.1&#8202;g; printed parts with "
        "infill weigh less. The parts have not been weighed.",
        "The servo horns, shafts, bearings, screws and cables are not in the STLs and not in the model. At the PIP joint the DIP "
        f"link&#8217;s horn-side arm lies {mm(j['pip']['boss_to_horn_side_mm'], 1)} above the servo&#8217;s boss, so a horn thicker than that "
        "would push the DIP link out along the PIP axis.",
        "The model squares the carriage to the yaw axis, removing the STL&#8217;s rotation about x; how the carriage meets the gantry is not "
        "modelled.",
        "The thumb is the index finger&#8217;s parts turned 180&#176; about the vertical; mirrored parts would need their own STLs.",
        f"Between {deg(X.stop('yaw', 'carriage|yaw_link', 1), True)} and the yaw limit of +{MEASURED['yaw'][1]}&#176; the yaw link passes into the "
        "carriage, because MuJoCo does not collide a body with its parent. On the negative side the spheres stop the yaw at "
        f"{deg(y[1])}, against {deg(X.stop('yaw', 'carriage|mcp_link', -1))} for the parts and the measured {deg(MEASURED['yaw'][0])}.",
        "Zero joint angles are the pose with each child&#8217;s long direction continuing its parent&#8217;s; the STLs do not encode the "
        "servos&#8217; mechanical zero.",
        "<code>hand.xml</code> and the scene keep real_v1&#8217;s <code>&lt;f&gt;_len</code> slide for its qpos layout. Its range "
        "&#8220;0 0&#8221; compiles as unlimited, along the pad normal, held only by its damping of 2000&#8202;N&#183;s/m; frozen scenes from "
        "<code>create_rigid_morphology_xml</code> remove it. <code>finger.xml</code> has no such joint.",
        "The scene drops real_v1&#8217;s fitted grasp key <code>open_ik</code>, which presses the pads into the tool on this longer finger.",
    ]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def open_items(X: Data):
    inert = X.asm["inertia"]
    servo_g = 1000 * B.SERVO_MASS_KG
    pred = [1000 * inert["mount"]["mass_kg"] - servo_g, 1000 * inert["yaw_frame"]["mass_kg"] - servo_g,
            1000 * inert["mcp_frame"]["mass_kg"] - servo_g, 1000 * inert["pip_frame"]["mass_kg"], 1000 * inert["tip"]["mass_kg"]]
    pip_room = X.rep["joints"]["pip"]["boss_to_horn_side_mm"]
    items = [
        ("Masses of the printed parts.", "Weigh the carriage, the three links, the tip and one servo. At solid density the build predicts "
         f"{', '.join(f'{g:.2f}' for g in pred[:4])} and {pred[4]:.2f}&#8202;g and {servo_g:.1f}&#8202;g per servo. If the five parts weigh "
         "in one ratio to their predictions, one infill factor on <code>DENSITY</code> in <code>scripts/build_cad_v1_finger.py</code> fits "
         "them; ratios that differ by more than 10&#8202;% need a density per part, which the builder does not have yet."),
        ("Thickness of the PIP horn.", f"Measure the PIP servo&#8217;s horn where it crosses the boss. The DIP link&#8217;s horn-side arm lies "
         f"{mm(pip_room, 1)} above the boss; a thicker horn moves the DIP link out along the PIP axis by the excess, and the flush-seat rule "
         "for that joint then needs the offset."),
        ("Grasp key for the longer finger.", f"Run <code>scripts/fit_real_v1_pose.py --scene {REL_MODEL}/scene_screwdriver_medium.xml "
         "--write</code> to write <code>open_ik</code>, the key the reorientation runs start from. The fit fails if no pose holds the "
         "screwdriver on the three tips&#8217; pads at the real_v1 mounts."),
        ("Reorientation policy on the CAD finger.", "Three scripts need changes first: <code>scripts/make_pad_morphology_run.py</code> fails "
         "on this scene with &#8220;repeated name tpu_pos in mesh&#8221;, because <code>replace_tips</code> appends the tip meshes to a scene "
         "that already has them; with <code>--mu</code> unset it reads friction 1.0 from <code>thumb_tipvis</code> instead of the pads&#8217; "
         "2.4; and <code>scripts/apply_measured_plant.py</code> finds no capsules and leaves the CAD masses as they are. Then train the "
         "servo-plant reorientation on a frozen copy of this scene and compare its held cosine with the capsule finger&#8217;s run "
         "<code>results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip_tpu2.7pads1</code>; a held cosine lower by more than that "
         "run&#8217;s seed spread means the link spheres&#8217; contacts change the task."),
        ("Clearance gates on the CAD geometry.", "<code>scripts/real_v1_trajectory_clearance.py</code> and the design search&#8217;s mount "
         "ranges were built on capsule fingers of 10.55&#8202;mm radius; the CAD fingers reach "
         f"{mm(max(X.dorsal.values()), 1)} on the dorsal side. Port the gate to this hand&#8217;s spheres and rerun it on the plans on the "
         "CB1; a plan that clears the capsules and collides here means the capsule gate does not protect the printed fingers."),
    ]
    return "<ol>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ol>"


def footer():
    steps = [
        ".venv/bin/python scripts/build_cad_v1_finger.py                 # meshes and assembly.json",
        "~/.claude/bin/resguard.sh run --mem 3G --cpu 500 -- .venv/bin/python scripts/cad_v1_spheres.py           # MorphIt-B",
        "~/.claude/bin/resguard.sh run --mem 3G --cpu 500 -- .venv/bin/python scripts/cad_v1_spheres.py --tight   # MorphIt-S",
        f".venv/bin/python scripts/build_cad_v1_finger.py --spheres {REL_MODEL}/spheres.json",
        "~/.claude/bin/resguard.sh run --mem 1G --cpu 300 -- .venv/bin/python scripts/cad_v1_interference.py",
        ".venv/bin/python scripts/cad_v1_sphere_contacts.py",
        ".venv/bin/python scripts/cad_v1_collision_bench.py",
        "MUJOCO_GL=egl .venv/bin/python scripts/cad_v1_finger_media.py",
        ".venv/bin/python scripts/cad_v1_finger_page.py",
    ]
    return ("<p>Rebuild from the repository root, in this order:</p><pre>" + "\n".join(steps) + "</pre>"
            "<p>MorphIt is the clone in <code>/home/humanoid/Programs/MorphIt</code> (HIRO-group/MorphIt-1 at 5f65299) with its own venv; "
            "<code>scripts/cad_v1_spheres.py</code> drives it through <code>hand_trials/pack_link.py</code> and measures fidelity with "
            "<code>hand_trials/sphere_fidelity.py</code>, two scripts that are not tracked in that clone. Parts: <code>assets/cad_v1/</code>. "
            f"Model, data and meshes: <code>{REL_MODEL}/</code>. Tests: <code>tests/test_cad_v1_finger.py</code>.</p>")


def main():
    X = Data()
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "GLOSSARY": glossary()}
    # sections in page order, so figures and tables number in reading order
    v["COMPARE"] = compare_section(X)
    v["ROM"] = rom_section(X)
    v["COLLISION"] = collision_section(X)
    v["VIEWER"] = viewer_section(X)
    v["PARTS"] = parts_section(X)
    v["METHOD"] = method_section(X)
    v["CHECKS"] = checks_section(X)
    v["LIMITS"] = limitations(X)
    v["OPEN"] = open_items(X)
    v["LEDE"] = lede(X)
    v["FOOTER"] = footer()
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
