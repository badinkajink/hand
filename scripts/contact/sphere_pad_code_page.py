#!/usr/bin/env python3
"""Implementation page of the sphere-pad fingertip (2026-11-11), a subpage of the contact model overview
(docs/overviews/contact_model.html, sections 'mechanics' and 'pads').

Owner, 2026-11-11: "can you build an accompaniment to the sphere packed contact model, similar to the hom/hom-implementation
docpages, that just details specifically how we implement the dense sphere packing for an arbitrary surface, with links
out to the collapsed code".

Reads docs/experiments/20261111-sphere_pad_implementation/rows/20261111-{steps,scene,press}.json and media/20261111-*
(scripts/contact/sphere_pad_code_data.py steps|scene|press|film). Mathematics and the algorithm through
scripts/docs/texsvg.py; code excerpts through scripts/docs/code_blocks.py.

    .venv/bin/python scripts/contact/sphere_pad_code_page.py
"""
from __future__ import annotations

import base64
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import code_blocks as CB  # noqa: E402
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = ROOT / "docs/experiments/20261111-sphere_pad_implementation"
MEDIA, ROWS = D / "media", D / "rows"
OUT = D / "20261111-sphere_pad_implementation.html"
TEMPLATE = ROOT / "scripts/contact/sphere_pad_code_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
OVERVIEW = "../../overviews/contact_model.html"
CODESIGN = "../20261014-tip_codesign/20261014-sphere_pad_accuracy_and_fingertip_codesign.html"
CURVED = "../20261015-curved_tips/20261015-capsule_and_ellipsoid_fingertips.html"
V3FIT = "../20261030-v3_tip_contact_model/20261030-v3_fingertip_contact_models.html"
HOMCODE = "../20261029-hom_control_review/20261030-hom_controller_code_walkthrough.html"
TG = "scripts/fingertip/tip_codesign_geometry.py"
RB = "scripts/contact/reorient_backends.py"
FG = "scripts/fingertip/fingertip_geometry.py"
SAMP = "src/sr2hand/geometry/sampling.py"
RIG = "scripts/hom/hom_contact_rig.py"
V3H = "scripts/hardware/build_cad_v3_hand.py"
DATA = "scripts/contact/sphere_pad_code_data.py"
FIG, FILM, TAB = [0], [0], [0]
ALGS: list[str] = []


def esc(s):
    return html.escape(str(s), quote=True)


def fill(s, **kw):
    """Replace <<key>> tokens (the prose keeps LaTeX braces, so str.format is out)."""
    for k, v in kw.items():
        s = s.replace(f"<<{k}>>", str(v))
    left = re.findall(r"<<(\w+)>>", s)
    if left:
        raise KeyError(f"unfilled tokens {left}")
    return s


def data_uri(p, mime):
    return f"data:{mime};base64,{base64.b64encode(Path(p).read_bytes()).decode()}"


def img(path, caption, width=None):
    FIG[0] += 1
    w = f' style="max-width:{width}px"' if width else ""
    mime = "image/jpeg" if Path(path).suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return (f'<figure><img src="{data_uri(path, mime)}" alt=""{w}>'
            f'<figcaption>Figure {FIG[0]}. {caption}</figcaption></figure>')


def film(path, caption, poster=None):
    FILM[0] += 1
    pa = f' poster="{data_uri(poster, "image/jpeg")}"' if poster and Path(poster).exists() else ""
    return (f'<figure><video controls muted loop playsinline preload="metadata"{pa} '
            f'src="{data_uri(path, "video/mp4")}"></video><figcaption>Film {FILM[0]}. {caption}</figcaption></figure>')


def table(head, body, caption, num_from=1):
    TAB[0] += 1
    h = "".join(f'<th class="{"n" if i >= num_from else ""}">{c}</th>' for i, c in enumerate(head))
    b = "".join("<tr>" + "".join(f'<td class="{"n" if i >= num_from else ""}">{c}</td>' for i, c in enumerate(r))
                + "</tr>" for r in body)
    return (f'<div class="tw"><table><caption>Table {TAB[0]}. {caption}</caption><thead><tr>{h}</tr></thead>'
            f'<tbody>{b}</tbody></table></div>')


def alg(caption, body):
    """An algorithm as a LaTeX algorithmic block (rendered by texsvg after the page's inline mathematics)."""
    ALGS.append(f"\\begin{{algorithmic}}[1]\n{body}\n\\end{{algorithmic}}")
    return (f'<figure class="alg"><figcaption><b>Algorithm&#160;{len(ALGS)}.</b> {caption}</figcaption>'
            f"\x00ALG{len(ALGS) - 1}\x00</figure>")


def render(page_text, holder):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(page_text)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    holder.append(len(items))
    return TEX_RE.sub(lambda m: next(svgs), page_text)


def blk(path, qualname, title, **kw):
    return CB.block(path, CB.find_def(path, qualname), title, **kw)


def LK(path, qualname, text=None):
    """A link to a def's lines on GitHub, named in code font."""
    return CB.link(path, CB.find_def(path, qualname), f"<code>{text or qualname}</code>")


def n0(x):
    return f"{x:,.0f}"


# ------------------------------------------------------------------------------------------------------ sections
def sec_terms():
    terms = [
        ("Sphere pad", "the set of small MuJoCo contact spheres that stands for a soft fingertip. Each sphere $i$ is an "
                       "ordinary MuJoCo contact with its own stiffness; together they sample the pressure of the "
                       "hydroelastic layer."),
        ("v4b fingertip", "the printed pad of the SR2 hand's cad_v2 finger, half an ellipsoid. Its key in "
                          "<code>tip_codesign_geometry</code> is <code>tcd_ell7.4x8.5x11x22</code>: semi-axes 7.4 mm "
                          "across, 8.5 mm deep, 11 mm toward the tip and 22 mm toward the PIP joint."),
        ("Tip frame", "the fingertip body's frame: $x$ along the palmar normal, $y$ across the finger (the PIP axis), "
                      "$z$ along the finger toward the PIP joint. The v4b apex lies at $(10.55, 0, 0)$ mm."),
        ("Inside test", "a function that returns, for a batch of points, whether each lies in the fingertip body "
                        "(<code>inside_design</code>)."),
        ("Samples", r"points on the exact surface with an outward normal $n_k$ and an area $a_k$: the pieces of the "
                    r"fine hull's triangles, moved onto the surface."),
        ("Voxel", r"a cube of side $s$ = 1 mm in a grid aligned with the tip frame; a point $p$ lies in voxel "
                  r"$\lfloor p/s \rfloor$."),
        (r"Area share $a_i$", "the surface area given to sphere $i$, in mm<sup>2</sup>: the sum of its samples' areas."),
        (r"Surface point $p_i$, normal $n_i$", r"the area-weighted mean of sphere $i$'s samples moved back onto the "
                                                r"surface, and their area-weighted mean normal, normalized."),
        (r"Centre $c_i$, radius $r_s$", r"the sphere's centre, $p_i - r_s n_i$ clamped into the body shrunk by $r_s$; "
                                         r"$r_s$ = 0.75 mm (<code>PAD_RS</code>)."),
        (r"Volume centroid $c_V$", "the centroid of the solid fingertip, from its fine hull."),
        (r"Depth $h_i$", r"$n_i \cdot (p_i - c_V)$, at least 1 mm: the distance along the normal from the surface point "
                         r"to the volume centroid."),
        (r"Modulus $E$", r"the hydroelastic modulus, 2.5 MPa in the scenes since 2026-10-30 (<code>E_TPU</code>)."),
        (r"Stiffness $K_i$", r"$E a_i / h_i$ in N/m: the spring of sphere $i$ at rest."),
        (r"Inverse weight $w_b$", r"MuJoCo's translational inverse inertia of body $b$ at the model's reference pose, "
                                  r"<code>body_invweight0[b, 0]</code>, in kg<sup>-1</sup>. The contact between bodies "
                                  r"$b_1$ and $b_2$ is computed with $w = w_{b_1} + w_{b_2}$."),
        (r"$t_c$, $t_r$, $d_0$", r"solref's time constant, the relaxation time $t_r = 2 t_c$ (0.04 s at 2.5 MPa), and "
                                 r"solimp's impedance, the same at both ends of its width ($d_{\min} = d_{\max} = d_0$)."),
        ("Approach", r"how far the plate's face has moved past the dome's apex, in &micro;m."),
        ("Penetration", "how far the plate's face lies inside one sphere (MuJoCo's contact distance with its sign "
                        "changed)."),
        ("Scallop depth", "the distance from a point of the exact surface to the nearest sphere's surface: the dimple "
                          "between neighbouring spheres."),
        ("Continuous law", r"the Winkler foundation that the spheres sample, with the same depth field $h$. Its "
                           r"pressure is $E u / h(x)$ for a local overlap $u$ along the plate's normal, integrated over "
                           r"the projected area of the exact surface."),
        ("Eq. (10)", r"the contact overview's static stiffness of one MuJoCo contact row (Section 7)."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in terms) + "</dl>"


def build():
    st = json.loads((ROWS / "20261111-steps.json").read_text())
    sc = json.loads((ROWS / "20261111-scene.json").read_text())
    pr = json.loads((ROWS / "20261111-press.json").read_text())
    th = sc["thumb"]
    P = {r["F"]: r for r in pr["presses"]}
    Cm = {c["F"]: c for c in pr["comp"]}
    ratio = {F: Cm[F]["F_pads_eq10"] / Cm[F]["F_law"] for F in Cm}
    dmax = max(r["ratio_maxdev"] for r in pr["presses"])
    w_tip, w_tool = th["inv_tip"], th["inv_tool"]
    k_table, k_tip = (w_tip + w_tool) / w_tip, (w_tip + w_tool) / (2 * w_tip)
    nums = dict(
        n=n0(st["pads"]), rays=n0(st["n_rays"]), hv=n0(st["hull_vertices"]), hf=n0(st["hull_faces"]),
        dn=n0(st["dense_n"]), da=n0(st["dense_area_mm2"]), pa=f"{st['piece_area_mm2_median']:.4f}",
        mn=n0(st["masked_n"]), ma=n0(st["masked_area_mm2"]), vox=n0(st["voxels"]), sl=n0(st["slivers"]),
        mv=f"{100 * st['samples_moved_frac']:.0f}", pjm=f"{st['proj_um_median']:.1f}", pjx=f"{st['proj_um_max']:.0f}",
        cm=n0(st["clamp_moved"]), cmed=f"{st['clamp_um_median']:.1f}", c100=n0(st["clamp_over_100um"]),
        c100b=n0(st["clamp_over_100um_at_back_or_mount"]), cmax=f"{st['clamp_um_max'] / 1000:.2f}",
        nnc=f"{st['nn_centres_mm']['median']:.2f}", nnm=f"{st['nn_mm']['median']:.2f}",
        nn5=f"{st['nn_mm']['p5']:.2f}", nn95=f"{st['nn_mm']['p95']:.2f}",
        amin=f"{st['area_mm2']['min']:.2f}", amax=f"{st['area_mm2']['max']:.2f}", amed=f"{st['area_mm2']['median']:.2f}",
        hmin=f"{st['h_mm']['min']:.2f}", hmax=f"{st['h_mm']['max']:.2f}", hap=f"{st['h_mm']['apex']:.2f}",
        cvx=f"{st['cV_mm'][0]:.2f}", cvz=f"{st['cV_mm'][2]:.2f}",
        kmin=f"{st['K_Npm']['min']:.0f}", kmax=f"{st['K_Npm']['max']:.0f}", kmed=f"{st['K_Npm']['median']:.0f}",
        ksum=f"{st['K_Npm']['sum_N_per_mm']:.0f}", kdome=f"{st['K_Npm']['dome_sum_N_per_mm']:.0f}",
        ndome=n0(st["dome_pads"]), tdev=f"{st['top_dev_um']['dome_absmax']:.0f}",
        scm=f"{st['scallop_mm']['median']:.2f}", sc95=f"{st['scallop_mm']['p95']:.2f}",
        scx=f"{st['scallop_mm']['max']:.2f}", xtop=f"{st['x_top_mm']:.3f}",
        gap=f"{(st['x_apex_mm'] - st['x_top_mm']) * 1000:.0f}", yrow=f"{st['y_row_mm']:.2f}",
        wtip=f"{w_tip:.2f}", wtool=f"{w_tool:.1f}", d0a=f"{th['d0_min']:.3f}", d0b=f"{th['d0_max']:.3f}",
        d0m=f"{th['d0_median']:.3f}", perr=f"{max(sc[f]['pos_err_um'] for f in ('thumb', 'index', 'middle')):.2f}",
        kerr=f"{max(sc[f]['K_rel_err_max'] for f in ('thumb', 'index', 'middle')) * 1e6:.0f}",
        ktab=f"{k_table:.1f}", ktip=f"{k_tip:.1f}", heavy=f"{pr['heavy']['ratio_med']:.2f}",
        dev=f"{dmax * 1e6:.1f}", a025=f"{P[0.25]['delta_um']:.0f}", a4=f"{P[4.0]['delta_um']:.0f}",
        t025=P[0.25]["n_touch"], t4=P[4.0]["n_touch"], r025=f"{ratio[0.25]:.2f}", r4=f"{ratio[4.0]:.2f}",
        lmin=f"{min(100 * (Cm[F]['delta_law_um'] / P[F]['delta_um'] - 1) for F in P):.0f}",
        lmax=f"{max(100 * (Cm[F]['delta_law_um'] / P[F]['delta_um'] - 1) for F in P):.0f}",
    )

    def F_(s):
        return fill(s, **nums)

    # ---------------------------------------------------------------------------------------- 2. the algorithm
    algo = alg("Sphere pads for a convex fingertip surface. Lines 1 to 12 are geometry and run once per fingertip "
               "shape (<code>TipShape.pads</code>); lines 13 to 17 set the MuJoCo parameters and run once per scene "
               "(<code>replace_tips</code>).", r"""\Require an inside test of a convex body, a point \(o\) inside it, spacing \(s = 1\) mm, sphere radius \(r_s = 0.75\) mm, modulus \(E\), the tool body
\State \(q_j \gets\) the surface point on the ray from \(o\) along the \(j\)-th of 300{,}000 Fibonacci directions, by 42 bisections of the inside test
\State \((V, F) \gets\) the convex hull of the points \(q_j\)
\State \((p_k, n_k, a_k) \gets\) the hull's triangles cut into pieces of at most 0.01 mm\(^2\); each centroid bisected back onto the surface
\State keep the samples with \(n_{k,x} > -0.5\) and \(n_{k,z} < 0.5\) \Comment{no spheres on the back and mount faces}
\State group the samples by voxel \(\lfloor p_k / s \rfloor\); drop each voxel with less than \(0.3\,s^2\) of surface
\State give each sample to the remaining voxel whose area-weighted mean point is nearest
\For{each remaining voxel \(i\)}
\State \(a_i \gets\) the area of its samples;\quad \(n_i \gets\) their normalized area-weighted mean normal
\State \(p_i \gets\) their area-weighted mean point, bisected back onto the surface along the ray from \(o\)
\State \(c_i \gets p_i - r_s n_i\), clamped into the body shrunk by \(r_s\)
\State \(h_i \gets \max\bigl(n_i\cdot(p_i - c_V),\ 1\ \mathrm{mm}\bigr)\);\quad \(K_i \gets E\,a_i / h_i\)
\EndFor
\State \(t_c \gets t_r(E)/2\);\quad compile the scene and read \(w \gets w_{\mathrm{tip}} + w_{\mathrm{tool}}\)
\For{each sphere \(i\)}
\State \(d_{0,i} \gets 1 - 1/(t_c^2\,K_i\,w)\)
\State write a sphere geom of radius \(r_s\) at \(c_i\) with solref \((t_c, 1)\) and solimp \((d_{0,i}, d_{0,i}, 0.001, 0.5, 2)\)
\EndFor""")
    steps_rows = [
        ["1. Rays to the exact surface", LK(TG, "TipShape.hull_fine", "hull_fine"), F_("<<rays>> points")],
        ["2. Fine hull", LK(TG, "TipShape.hull_fine", "hull_fine"), F_("<<hv>> vertices, <<hf>> triangles")],
        ["3. Samples", LK(TG, "TipShape.dense", "dense"), F_("<<dn>> over <<da>> mm<sup>2</sup>")],
        ["4. Faces that get spheres", LK(TG, "TipShape.pads", "pads"), F_("<<mn>> samples over <<ma>> mm<sup>2</sup>")],
        ["5. Voxels met by the surface; dropped", LK(TG, "_voxel_pick"), F_("<<vox>>; <<sl>>")],
        ["6. Samples moved to a neighbouring voxel", LK(TG, "TipShape.pads", "pads"), F_("<<mv>> %")],
        ["7&ndash;9. Spheres; their centres' spacing", LK(TG, "TipShape.pads", "pads"),
         F_("<<n>>; median <<nnc>> mm")],
        ["10. Centres clamped; more than 0.1 mm", LK(TG, "TipShape._clamp", "_clamp"), F_("<<cm>>; <<c100>>")],
        ["11. Depth $h_i$; stiffness $K_i$", LK(TG, "TipShape.pads", "pads"),
         F_("<<hmin>>&ndash;<<hmax>> mm; <<kmin>>&ndash;<<kmax>> N/m")],
        ["13&ndash;17. Impedance $d_{0,i}$ in the talk's scene", LK(RB, "replace_tips"), F_("<<d0a>>&ndash;<<d0b>>")],
    ]
    s_alg = F_(r"""<p>Algorithm 1 is the whole construction. Each sphere stands for one piece of the surface cut by
a 1 mm voxel grid, and their spacing follows from that cut alone (Section 4). The geometry is computed once per fingertip shape
by <code>TipShape.pads</code>. The MuJoCo parameters are set once per scene by <code>replace_tips</code>, because they
need the inverse weights of the compiled scene. The counts of each step on the v4b fingertip with $E$ = 2.5 MPa
are in Table 1.</p>""") + algo + table(["Step of Algorithm 1", "Code", "v4b fingertip"], steps_rows,
                                     "The steps of Algorithm 1 on one v4b fingertip.", num_from=9)

    # ---------------------------------------------------------------------------------------- 3. surface and samples
    s_surf = F_(r"""<p>Each fingertip shape is given to the builder as an inside test and a point inside the body. The v4b test
(<code>CurvedTip.inside_design</code>) intersects the envelope box, 17.0 mm deep, 14.8 mm wide and 21.41 mm long, with the
half-ellipsoid. Behind the ellipsoid's equator the section runs straight back to the back face. Rays leave the box centre
in <<rays>> Fibonacci directions. On each ray the surface is found by 42 bisections of the inside test between 0 and 30
mm, to 7 fm. The convex hull of these points has <<hv>> vertices and <<hf>> triangles (<code>hull_fine</code>).</p>
<p>The hull's triangles are cut into pieces of at most 0.01 mm<sup>2</sup> (<code>_subsample</code>). Each piece keeps
its triangle's normal and its own area. Its centroid lies on a chord, inside the curved surface. It is bisected back
onto the exact surface along the ray from the box centre (<code>dense</code>). The v4b fingertip gives <<dn>> samples
over <<da>> mm<sup>2</sup>, with a median area of <<pa>> mm<sup>2</sup>. On the curved parts the hull's triangles are
already smaller than the target, and only the flat faces' large triangles are cut.</p>
<p>Radial bisection finds one point per ray, so every ray from the interior point must cross the surface once. Every
convex body meets this, and the hull would bridge a concave part of a surface. All fingertip keys of
<code>tip_codesign_geometry</code> are convex bodies: the filleted boxes, the capsules, the half- and quarter-ellipsoids,
the spheres, and their as-printed hulls (<a href="{CODESIGN}">co-design page</a>, <a href="{CURVED}">curved-tip
page</a>).</p>""")
    s_surf = s_surf.replace("{CODESIGN}", CODESIGN).replace("{CURVED}", CURVED)
    s_surf += blk(TG, "CurvedTip.inside_design", "Code: the v4b body as an inside test, <code>CurvedTip.inside_design</code>")
    s_surf += blk(TG, "TipShape._bisect", "Code: the surface point on each ray, <code>TipShape._bisect</code>")
    s_surf += blk(SAMP, "fibonacci", "Code: the ray directions, <code>sampling.fibonacci</code>")
    s_surf += blk(TG, "TipShape.hull_fine", "Code: lines 1 and 2, the fine hull, <code>TipShape.hull_fine</code>")
    s_surf += blk(TG, "_subsample", "Code: the hull's triangles cut into pieces, <code>_subsample</code>")
    s_surf += blk(TG, "TipShape.dense", "Code: line 3, the samples on the exact surface, <code>TipShape.dense</code>")

    # ---------------------------------------------------------------------------------------- 4. voxels
    s_vox = F_(r"""<p>Samples with $n_x \le -0.5$ or $n_z \ge 0.5$ get no sphere. They make up the back face, which
rests on the DIP link's back plate, and the mount face. <<mn>> samples over <<ma>> mm<sup>2</sup> remain. Each is
assigned to its voxel $\lfloor p/s \rfloor$ of the 1 mm grid, and the v4b surface meets <<vox>> voxels. A voxel that the
surface only clips has less than $0.3\,s^2$ = 0.3 mm<sup>2</sup> of surface. The <<sl>> such voxels are dropped.</p>
<p>Each sample then joins the remaining voxel whose area-weighted mean point is nearest. <<mv>> % of the samples move to
a neighbouring voxel's sphere this way, those of the dropped voxels among them. The samples of sphere $i$ give its area
share $a_i$ and its normal $n_i$, their area-weighted mean normal. Their area-weighted mean point lies inside the curved
surface, by a median of <<pjm>> &micro;m and at most <<pjx>> &micro;m. It is bisected back onto the surface along the
ray from the box centre, which gives the surface point $p_i$.</p>
<p>Figure 1(a) shows the pieces on the dome. Within one voxel layer of the dome's height $x$, each piece is a 1 mm
square. Where the surface passes from one layer to the next, the pieces are split, merged or long. The surface points are
therefore spaced <<nn5>> to <<nn95>> mm apart (5th to 95th percentile of the distance to the nearest one, median
<<nnm>> mm). The areas range from <<amin>> to <<amax>> mm<sup>2</sup>, with a median of <<amed>> mm<sup>2</sup>.</p>""")
    s_vox += img(MEDIA / "20261111-voxel_pads.png", F_(
        r"""The v4b fingertip's pieces and spheres. (a) The dome seen from the palm. Each colour patch is the set of
samples given to one sphere, and the dots are the spheres' surface points $p_i$. The dashed lines are the surface's
heights $x$ = 5 to 10 mm, where the voxel layer changes. (b) The section through one row of centres, $y$ = <<yrow>> mm,
at the distal end, to scale: the exact outline and each sphere's section in that plane. Each sphere touches the outline
at its own surface point."""))
    s_vox += blk(TG, "TipShape.pads", "Code: lines 4 to 12, <code>TipShape.pads</code>")
    s_vox += blk(TG, "_voxel_pick", "Code: one area-weighted mean per occupied voxel, <code>_voxel_pick</code>")
    s_vox += blk(TG, "CurvedTip.pads", "Code: the curved shapes' entry, <code>CurvedTip.pads</code>")

    # ---------------------------------------------------------------------------------------- 5. centres
    s_cen = F_(r"""<p>Each centre is set $r_s$ = 0.75 mm inside the surface along the normal, $c_i = p_i - r_s n_i$.
The sphere then touches the surface at $p_i$, tangent to it there. The centres are then clamped into the body shrunk by
$r_s$ (<code>_clamp</code>). Each centre must lie on the inner side of every facet plane of the 0.5 mm hull, moved inward
by $r_s$. <<cm>> of the <<n>> centres moved, by a median of <<cmed>> &micro;m. These small moves come from the hull's
facets, which are chords and lie a few micrometres inside the curved surface. <<c100>> centres moved by 0.1 to <<cmax>>
mm, and <<c100b>> of them lie in the last voxel layer before the back face or the mount face. Their pieces' mean point
lies within 0.75 mm of that face, so an unclamped sphere would stand out past it.</p>
<p>On the dome the spheres' outermost points lie within <<tdev>> &micro;m of the exact surface. Between neighbouring
spheres the exact surface lies above the spheres, by a median of <<scm>> mm. The scallop depth is <<sc95>> mm at the
95th percentile and <<scx>> mm at most. Seen from the tool, the pad is the fingertip's surface with dimples of 0.1 to 0.2
mm (Figure 1b). The centres are <<nnc>> mm apart (median distance to the nearest centre), closer than the surface points
because they lie on a smaller, inset surface.</p>""")
    s_cen += blk(TG, "TipShape._clamp", "Code: line 10's clamp, <code>TipShape._clamp</code>")
    s_cen += blk(TG, "TipShape._planes", "Code: the hull's facet planes, <code>TipShape._planes</code>")

    # ---------------------------------------------------------------------------------------- 6. stiffness
    s_k = F_(r"""<p>Drake's hydroelastic field for a compliant convex body rises from zero at the surface to the modulus
$E$ at the body's deepest point. To first order, a thin piece of area $a$ at depth $h$ above that point is a spring of
stiffness $E a / h$. A layer of such springs is a Winkler foundation (<a href="{OV}#mechanics">contact overview</a>,
contact mechanics, step 2). Builder v2 gives each sphere the depth of its own surface point below the volume centroid
$c_V$, along its normal:</p>
$$ h_i = \max\bigl(n_i \cdot (p_i - c_V),\ 1\ \mathrm{mm}\bigr), \qquad K_i = \frac{E\,a_i}{h_i}. $$
<p>On the v4b fingertip $c_V$ lies at (<<cvx>>, 0, <<cvz>>) mm. The depth ranges from <<hmin>> to <<hmax>> mm and is
<<hap>> mm at the apex. At $E$ = 2.5 MPa, $K_i$ ranges from <<kmin>> to <<kmax>> N/m, with a median of <<kmed>> N/m. The
stiffness per unit area, $E/h_i$, changes smoothly over the fingertip. Neighbouring spheres differ by up to a factor of
two, because the voxel grid cuts pieces of unequal area (Figures 2 and 3). The <<ndome>> spheres of the dome ($n_x$ &gt;
0.5) add up to <<kdome>> N/mm, and all <<n>> to <<ksum>> N/mm.</p>
<p>The modulus is <code>reorient_backends.E_TPU</code>, 2.5 MPa since 2026-10-30. It was fitted with these spheres to the
UR5 presses of three printed v3 fingertips, 2.47 MPa over 3,568 static holds (<a href="{V3}">fit page</a>). Builder v1 reproduces
runs before 2026-10-15: it gives every sphere of a finger one stiffness, the median of $E a_i / h$ with $h$ = 8.5 mm.</p>
<p>The contact bed's sphere cap (Section 9) scales each sphere's area by $((R - r_s)/R)^2$. Its centres lie on a sphere
$r_s$ smaller than the fingertip. Seen from the tool, their contact points are therefore packed more densely than the
surface pieces (<a href="{OV}#pads">contact overview</a>, Eq. 8). <code>TipShape.pads</code> takes $a_i$ on the exact surface
without that factor. Section 8 measures what this leaves on the v4b dome.</p>""")
    s_k = s_k.replace("{OV}", OVERVIEW).replace("{V3}", V3FIT)
    s_k += img(MEDIA / "20261111-pad_stiffness_render.png", F_(
        r"""The <<n>> spheres of one v4b fingertip coloured by their stiffness $K_i$ at 2.5 MPa, seen from the palm
(left) and from the side (right), the finger pointing down. The bands of lower stiffness are the split pieces where the
surface passes from one voxel layer to the next (Figure 1a)."""))
    s_k += img(MEDIA / "20261111-pad_distributions.png", F_(
        r"""Distributions over the <<n>> spheres of the v4b fingertip. The panels give the spacing of the surface
points to the nearest other one, the area share $a_i$, the depth $h_i$ and the stiffness $K_i$ at 2.5 MPa. Blue: the <<ndome>> spheres of the
dome ($n_x$ &gt; 0.5). Grey: the side walls and the end face."""))
    s_k += blk(TG, "TipShape.volume_centroid", "Code: the volume centroid $c_V$, <code>TipShape.volume_centroid</code>")
    s_k += CB.block(RB, CB.find_span(RB, "# Pad modulus (owner, 2026-10-30", until="PAD_BUILDERS = "),
                    "Code: the modulus, the relaxation time, the spacing and the builders, in <code>reorient_backends</code>")
    s_k += blk(RB, "pad_set", "Code: the spheres of one fingertip by key, <code>pad_set</code>")

    # ---------------------------------------------------------------------------------------- 7. MuJoCo
    s_mj = F_(r"""<p>Each sphere becomes a MuJoCo sphere geom of radius 0.75 mm in the fingertip body. It has condim 3,
a normal force and two friction directions. Its friction coefficient is the scene's, and it has no mass. With priority 1
its own solref and solimp apply to every contact it makes. The fingertip's mesh stays as a visual geom without contacts
(<code>&lt;finger&gt;_tipvis</code>).</p>
<p>At rest, a MuJoCo contact row with solref $(t_c, 1)$ and a constant impedance $d_0$ is a spring
(<a href="{OV}#pads">contact overview</a>, Eq. 10):</p>
$$ f = K\,r, \qquad K = \frac{1}{(1 - d_0)\,t_c^2\,w}, \qquad w = w_{\mathrm{tip}} + w_{\mathrm{tool}}, $$
<p>where $f$ is the normal force and $r$ the penetration. The builder fixes $t_c$ and solves for each sphere's
impedance:</p>
$$ d_{0,i} = 1 - \frac{1}{t_c^2\,K_i\,w}. $$
<p>The relaxation time is set per modulus by <code>pad_tr</code>: 0.02 s at 10 MPa, scaled by $\sqrt{10\ \mathrm{MPa}/E}$,
so 0.04 s at 2.5 MPa. At this relaxation time each sphere's $d_0$ equals its value at 10 MPa. <code>replace_tips</code> writes the spheres,
compiles the scene once and reads $w$ from <code>body_invweight0</code>. It then writes solref $(t_c, 1)$ and solimp
$(d_{0,i}, d_{0,i}, 0.001, 0.5, 2)$ on every sphere. The build stops with an error at any sphere whose $d_0$ falls below 0.05.</p>
<p>One scene of the talk's landscapes was checked against the builder: the 25 mm cylinder on the layout
$x_{\mathrm{sep}}$ = 92.5 mm, $y_{\mathrm{sep}}$ = 102.5 mm, SCS0009 plant. In it the fingertip body's inverse weight is <<wtip>>
kg<sup>-1</sup> and the 25 g object's <<wtool>> kg<sup>-1</sup>, and $d_0$ runs from <<d0a>> to <<d0b>> (median
<<d0m>>). The scene's 3 &times; <<n>> spheres equal the builder's to <<perr>> &micro;m in position and to
<<kerr>> &times; 10<sup>-6</sup> in stiffness.</p>
<p>The inverse weight in $d_{0,i}$ is that of the body named as the tool. Against another body $b$ the same sphere is a
spring of $K_i\,(w_{\mathrm{tip}} + w_{\mathrm{tool}})/(w_{\mathrm{tip}} + w_b)$. In the talk's scene a v4b sphere is
<<ktab>> times stiffer against the static table ($w_b$ = 0) than against the object. Against another fingertip it is
<<ktip>> times stiffer. In the press of Section 8, a 100 g plate pressed with the 25 g plate's $d_0$ made every sphere
<<heavy>> times stiffer.</p>""")
    s_mj = s_mj.replace("{OV}", OVERVIEW)
    s_mj += blk(RB, "pad_tr", "Code: the relaxation time at modulus $E$, <code>pad_tr</code>")
    s_mj += CB.block(RB, CB.find_span(RB, "for i, c in enumerate(C):", until='g.set("conaffinity", "0")',
                                      within="replace_tips"),
                     "Code: line 16's geoms, in <code>replace_tips</code>")
    s_mj += CB.block(RB, CB.find_span(RB, "m0 = mujoco.MjModel.from_xml_string(ET.tostring(root", before=1,
                                      until="pad_builder=builder)", within="replace_tips"),
                     "Code: lines 13 to 17, $d_{0,i}$ from the compiled inverse weights, in <code>replace_tips</code>")
    s_mj += blk(RB, "replace_tips", "Code: the whole of <code>replace_tips</code>")
    s_mj += blk(V3H, "add_pads", "Code: the cad_v3 hand's spheres through <code>replace_tips</code>, "
                "<code>build_cad_v3_hand.add_pads</code>")

    # ---------------------------------------------------------------------------------------- 8. press
    prow = []
    for F in sorted(P):
        r, c = P[F], Cm[F]
        prow.append([f"{F:g}", f"{r['delta_um']:.0f}", f"{c['delta_law_um']:.0f}", str(r["n_touch"]),
                     f"{ratio[F]:.3f}", f"{r['ratio_maxdev'] * 1e6:.1f}"])
    s_pr = F_(r"""<p>One v4b fingertip was fixed with its <<n>> spheres and the parameters of Section 7. A rigid 25 g plate
on a slide joint along $x$ was pressed on it with 0.25 to 4 N, for 2 s at each force. The solver settings were the
scenes': a 1 ms step, implicitfast, elliptic cones, Newton and impratio 100, here without gravity. At rest the contact
forces added up to the plate's force. Each touching sphere's force equalled $K_i$ times its penetration to within
<<dev>> &times; 10<sup>-6</sup> of it (Table 2, Figure 4 right), as Eq. (10) states.</p>
<p>The approach was <<a025>> &micro;m at 0.25 N and <<a4>> &micro;m at 4 N. <<t025>> spheres touched the plate at
0.25 N and <<t4>> at 4 N. The continuous law with the same modulus and depth field needed <<lmin>> to <<lmax>> % more
approach for the same force. At equal approach the pads pushed <<r025>> times the law's force at 0.25 N and <<r4>> times
at 4 N. The first sphere touches the plate <<gap>> &micro;m below the apex, at $x$ = <<xtop>> mm. Film 1 shows the
patch growing over the press.</p>""")
    s_pr += table(["Plate force (N)", "Approach, pads (&micro;m)", "Approach, law (&micro;m)", "Spheres touching",
                   "Force ratio", "Eq. (10) error (10<sup>-6</sup>)"], prow,
                  "A flat 25 g plate pressed on one v4b fingertip in MuJoCo, at rest. Force ratio: the pads' force over "
                  "the law's at the same approach. Eq. (10) error: the largest $|f_i/(K_i r_i) - 1|$ over the touching "
                  "spheres, with $f_i$ a sphere's force and $r_i$ its penetration.")
    s_pr += img(MEDIA / "20261111-flat_press.png", F_(
        r"""The flat press. Left: plate force against approach for the continuous law, the pads' springs (Eq. 10) and
MuJoCo at rest. Right: each touching sphere's force against its stiffness $K_i$ times its penetration, at the five
forces; the line is equality."""))
    s_pr += film(MEDIA / "20261111-flat_press.mp4", F_(
        r"""The v4b fingertip pressed by the flat plate from 0 to 4 N in 3 s and held for 1 s, in MuJoCo with the
scenes' solver settings. Left: the fingertip seen through the plate, each touching sphere coloured by its force. Right:
the plate force against the approach, with the pads' springs and the continuous law of Figure 4. During the ramp the
plate trails the static curve by the contacts' damping."""), poster=MEDIA / "20261111-flat_press_poster.jpg")
    s_pr += blk(DATA, "press_model", "Code: the press model, <code>sphere_pad_code_data.press_model</code>")
    s_pr += blk(DATA, "cmd_press", "Code: the presses, the heavier plate and the continuous law, "
                "<code>sphere_pad_code_data.cmd_press</code>")

    # ---------------------------------------------------------------------------------------- 9. other samplers
    orow = [
        ["Convex CAD shapes (<code>tcd_*</code>), the v4b fingertip", "voxel pieces of the exact surface (Algorithm 1)",
         "area of the piece", r"$n_i \cdot (p_i - c_V)$", LK(TG, "TipShape.pads", "TipShape.pads")],
        ["Printed TPU block (<code>tpu&lt;r&gt;</code>), real_v1 and cad_v1 hands",
         "analytic grid at spacing $s$ on faces, fillets and corners", "area of the patch",
         r"$n_i \cdot (p_i - c_{\mathrm{box}})$; v1 8.5 mm", LK(FG, "pad_spheres")],
        ["Sphere cap of the contact bed", "Fibonacci lattice on a 35&deg; cap at radius $R - r_s$",
         r"cap area over count, times $((R - r_s)/R)^2$", r"$R$ = 10.55 mm", LK(SAMP, "cap_pad")],
    ]
    s_oth = r"""<p>Two other fingertips have their own samplers (Table 3). The printed TPU block of the real_v1 and cad_v1
hands is a rounded box. Its surface is sampled analytically at spacing $s$: a grid on each flat face, arcs on each
filleted edge and Fibonacci points on each corner (<code>fingertip_geometry.surface_samples</code>). Each sample becomes
one sphere with its patch's exact area, and the centres are clipped into the box shrunk by $r_s$. Builder v2 measures
each depth to the block's centre $c_{\mathrm{box}}$. The contact bed's
fingertip is a sphere of radius $R$ = 10.55 mm. Its spheres sit on a Fibonacci lattice of a 35&deg; cap at radius
$R - r_s$, and the cap's area is shared equally among them (<code>sampling.cap_pad</code>). The bed sets $d_0$ with its
own copy of line 15 (<code>hom_contact_rig</code>). The quarter-ellipsoid keys (<code>tcd_qell</code>) of the curved-tip
study take $h_i$ as the TPU's thickness between $p_i$ and the rigid back plane along $n_i$, at least 1 mm.</p>"""
    s_oth += table(["Fingertip", "Points", "Area share", "Depth", "Code"], orow,
                   "The three sphere-pad samplers.", num_from=9)
    s_oth += blk(FG, "surface_samples", "Code: the TPU block's analytic samples, "
                 "<code>fingertip_geometry.surface_samples</code>")
    s_oth += blk(FG, "pad_spheres", "Code: the TPU block's spheres, <code>fingertip_geometry.pad_spheres</code>")
    s_oth += blk(SAMP, "cap_pad", "Code: the bed's cap, <code>sampling.cap_pad</code>")
    s_oth += CB.block(RIG, CB.find_span(RIG, "# second pass: per-sphere stiffness", before=1,
                                        until='self.info["diagApprox"] = float(diag)'),
                      "Code: the bed's $d_0$ from the compiled inverse weights, in <code>hom_contact_rig</code>")

    # ---------------------------------------------------------------------------------------- 10. future, 11. files
    s_fut = F_(r"""<ul>
<li>Scale each area share in <code>TipShape.pads</code> by the ratio of the inset surface's area to the exact surface's
over its piece, the overview's Eq. (8) for a general surface. Then rerun <code>sphere_pad_code_data.py press</code>.
The pads' force should lie within 2 % of the law's from 0.25 to 4 N; it is <<r025>> to <<r4>> times the law's now. Refit the modulus
with <code>scripts/fingertip/v3_tip_contact_model.py</code>. The fitted 2.5 MPa absorbs the density if the refit
modulus rises by the same ratio; otherwise the scenes' forces under the UR5 presses change.</li>
<li>Count the spheres' contacts with bodies other than the tool in the HOM rollouts (<code>hom_tasks.Plant.contacts</code>)
and the force they take. If the table or another fingertip takes more than 5 % of a finger's force, give those pairs
their own $d_0$ through MuJoCo <code>&lt;pair&gt;</code> elements and compare the rollouts' held outcome.</li>
<li>For a concave fingertip surface, such as a grooved print or a scanned pad, the samples must come from its own
mesh. Take the
mesh's triangles in place of <code>hull_fine</code> and project onto the mesh by closest point in place of the radial
bisections. On the v4b pad's mesh (<code>ur5_cb3_cell.part_F("pad")</code>) the spheres should then agree with the
analytic key's to within 0.05 mm.</li>
</ul>""")
    s_files = f"""<p>References:</p><ul>
<li>[1] R. Elandt, E. Drumwright, M. Sherman and A. Ruina, "A pressure field model for fast, robust approximation of net
contact force and moment between nominally rigid objects", IROS 2019.</li>
<li>[2] MuJoCo documentation, Computation, soft constraints and the solref and solimp parameters.
<a href="https://mujoco.readthedocs.io/en/stable/computation/index.html">mujoco.readthedocs.io/en/stable/computation</a>.
</li></ul>
<p>Files:</p><ul>
<li>This page: <code>{OUT.relative_to(ROOT)}</code>, built by <code>scripts/contact/sphere_pad_code_page.py</code>
(template <code>scripts/contact/sphere_pad_code_page.template.html</code>). Parent page: <a href="{OVERVIEW}">the
fingertip contact model overview</a>.</li>
<li>Numbers, figures and film: <code>~/.claude/bin/resguard.sh run --mem 3G --cpu 300 -- .venv/bin/python
scripts/contact/sphere_pad_code_data.py steps</code> (then <code>scene</code>, <code>press</code>, <code>film</code>);
rows <code>{(ROWS / '20261111-steps.json').relative_to(ROOT)}</code>, <code>20261111-scene.json</code>,
<code>20261111-press.json</code> and <code>20261111-pads.npz</code>.</li>
<li>Geometry: <code>{TG}</code> (convex CAD shapes), <code>{FG}</code> (the TPU block), <code>{SAMP}</code> (Fibonacci
lattices, the bed's cap).</li>
<li>MuJoCo parameters: <code>{RB}</code> (<code>E_TPU</code>, <code>pad_tr</code>, <code>pad_set</code>,
<code>replace_tips</code>). Callers: <code>scripts/hardware/build_cad_v2_finger.py</code>, <code>{V3H}</code>,
<code>scripts/hom/hand_object_scale_sim.py</code>, <code>scripts/contact/make_pad_morphology_run.py</code>.</li>
<li>The scene checked: <code>logs/20261102-hom_servo_plant/scenes/cyl025.0/x092.5_y102.5_pads_bv2__kp0.72_kv0_fr0.18_fl0_dp0.0144.xml</code>
(on the workstation; <code>logs/</code> is not in git).</li>
<li>The HOM controller's code walkthrough, the model for this page: <a href="{HOMCODE}">{HOMCODE.rsplit('/', 1)[1]}</a>.</li>
</ul>"""

    sections = [
        ("terms", "Terms and metrics", sec_terms()),
        ("algorithm", "Sphere pads for a convex fingertip surface", s_alg),
        ("surface", "The exact surface from an inside test", s_surf),
        ("voxels", "Pieces of the surface from a voxel grid", s_vox),
        ("centres", "Sphere centres inset from the surface", s_cen),
        ("stiffness", "Sphere stiffness from the hydroelastic layer", s_k),
        ("mujoco", "MuJoCo solref and solimp from the stiffness", s_mj),
        ("press", "A flat plate pressed on the v4b fingertip", s_pr),
        ("samplers", "Samplers of the TPU block and the contact bed", s_oth),
        ("future", "Future work", s_fut),
        ("files", "References and files", s_files),
    ]
    lede = F_(r"""A convex fingertip surface becomes MuJoCo sphere pads in two parts. The exact surface is cut into
pieces by a 1 mm voxel grid. Each piece becomes a sphere of radius 0.75 mm that touches the surface at the piece's
area-weighted centre. Each sphere is then given the stiffness $E a_i / h_i$ of its piece of the hydroelastic layer,
through MuJoCo's solimp. The v4b fingertip is covered by <<n>> spheres with centres <<nnc>> mm apart. Their stiffness at
2.5 MPa is <<kmin>> to <<kmax>> N/m. Under a flat plate in MuJoCo every touching sphere pushed with its spring's force to within
<<dev>> parts per million. The pad pushed <<r025>> to <<r4>> times the continuous law's force at the same approach.""")
    summary = F_(r"""How the sphere pads of a convex fingertip are built: a 1 mm voxel grid cuts the exact surface into
pieces, each piece becomes a 0.75 mm sphere touching the surface, and each sphere gets the stiffness E a/h of its piece of
the hydroelastic layer through MuJoCo's solimp. The v4b fingertip gets <<n>> spheres at <<kmin>> to <<kmax>> N/m; under a
flat plate each sphere's force equalled its spring to <<dev>> parts per million, and the pad pushed <<r025>> to <<r4>>
times the continuous law.""")
    byline = (f'Implementation page, 2026-11-11. Parent page: <a href="{OVERVIEW}">fingertip contact model</a> (its '
              f'sections on contact mechanics and sphere pads). Fingertip: the v4b half-ellipsoid of the cad_v2 hand at '
              f'2.5 MPa. Builder: <code>scripts/contact/sphere_pad_code_page.py</code>.')
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>'
                     for k, (sid, hd, body) in enumerate(sections, 1))
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "Sphere-pad implementation", "H1": "Implementation of the sphere-pad contact model on a "
                 "convex fingertip surface", "EYEBROW": "Fingertip contact model", "LEDE": lede, "BYLINE": byline,
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS + texsvg.BLOCK_CSS + CB.CSS + ".alg{margin:1em auto;max-width:100%}"}.items():
        page = page.replace("{{" + k + "}}", v)
    nf = []
    page = render(page, nf)
    svgs = texsvg.render([(b, "block") for b in ALGS], cache_path=TEX_CACHE)
    page = re.sub(r"\x00ALG(\d+)\x00", lambda m_: svgs[int(m_.group(1))], page)
    page = docs_index.tag(page, topic="contact", summary=summary, date="2026-11-11")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {nf[0]} formulas, {len(ALGS)} algorithms",
          flush=True)


if __name__ == "__main__":
    build()
