#!/usr/bin/env python3
"""Caps over the tip end of DIP link concept b v3 (2026-10-20 evening): a PLA end plate over the finger's T-shaped tip end
(the tongue and the cover, 21 x 5.1 mm, and the pad's flat end, 4.4 x 7.5 mm, all at z_F -98.46), two ears under the two
distal board screws, and a 2 mm pin into the pad's tip end; cap TW adds side walls along the pad's distal flanks.

Owner, 2026-10-20 19:48: "the finger is getting popped out a lot ... a separate fingertip that goes ontop of and fastens to
both the dip link (top of the magnetometer plate) and the top of the fingertip. im imagining a kind of T-shaped plate ... you
could trick out this T-plate and extrude it down to mold around the top of the fingertip (and around the screws too) for
additional side-to-side retaining"; 20:36, on the proposal (ears under the distal board screws instead of two holes in the
3.6 mm tongue, a pin instead of a thread or an insert in the TPU, two caps on one plate): "sure we can go with your
recommendation".

The pad rests against the column at its mount face, so the end plate holds it along the finger and the pin across it; nothing
is threaded into the TPU. The ears take the two distal board screws (M2 x 8 in place of M2 x 5, the same thread in the tongue)
and stand ear_gap_plain off the pad's flank on cap TP. Cap TW's walls are 6 mm tall like the cover's, from 0.2 mm past the
U3/U5 windows to the ears, where the ears (3 mm) take over beside the screw heads; their inner faces lie on the pad's flank
(wall clearance 0.00, the cover's W00). FE bounds of both caps: scripts/v2_tip_fe_checks.py cases bm_five_d75_capend and
bm_five_d75_cap. The pad that takes the pin: scripts/v3b_pad_b00_magnets_xl.py --pin-hole.

Commands (frame F of dip_link_concept_b; outputs under docs/experiments/20261018-dip_link_board_mount/v3/b/):
  build   both caps as meshes (manifold3d): cad/*_tip_cap_{TP,TW}.stl, print/*_tip_cap_{TP,TW}_pla_bambu.stl, and the checks
          (overlaps with every part, gaps to the pad, the PIP sweep against the MCP link and the PIP servo, STL round trip)
          in 20261020-dip_link_v3b_tip_caps.json
  step    the same solids as B-rep in freecadcmd: cad/*_tip_cap_{TP,TW}.step
  render  media/20261020-dip_link_v3b_tip_caps.png

    unset VIRTUAL_ENV PYTHONPATH
    ~/.claude/bin/resguard.sh run --mem 4G --cpu 200 -- taskset -c 14,15 env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/envs/femtip/bin/python -u scripts/dip_b_tip_cap.py build step render
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("DIPB_PARAMS", str(REPO / "docs/experiments/20261018-dip_link_board_mount/params/20261020-v3_concept_b.json"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import dip_link_concept_b as B  # noqa: E402

V3 = REPO / "docs/experiments/20261018-dip_link_board_mount/v3/b"
PREFIX = "20261020-dip_link_v3b"
WORK = REPO / "logs/20261020-dip_link/v3/b/tip_cap"
RECORD = V3 / f"{PREFIX}_tip_caps.json"
CAP = dict(
    t_end=1.2,            # end plate along z, beyond the tip end
    stem_margin=0.8,      # the plate's rim around the pad's flat end
    ear_t=3.0,            # ears along x, palmar of the cover's front face: the head 3.0 mm higher, M2 x 8 for M2 x 5
    ear_z1=-93.40,        # ears' proximal end, 0.5 mm past the screw heads (z_F -95.81 +- 1.9)
    ear_gap_plain=0.5,    # cap TP: the ears this far outside the pad's flank, so they hold nothing there
    wall_h=6.0,           # cap TW: walls as tall as the cover's (x_F 2.90 to -3.10)
    wall_t=1.2,           # cap TW: wall thickness
    wall_clear=0.0,       # cap TW: inner face to the pad's flank (the cover's W00)
    window_gap=0.2,       # cap TW: the walls start this far past the U3/U5 windows
    pin_d=2.0, pin_l=2.7, pin_x=1.40, pin_chamfer=0.3,   # into the pad's tip end, 1.5 mm in front of its back face
    hole_d=2.0, hole_l=3.0,                               # the pad's hole for it, 2.2 mm from the nearest pocket (R4: 2.0)
    screw_l=8.0,
)
VARIANTS = {"TP": dict(walls=False, label="Cap TP: end plate, ears and pin"),
            "TW": dict(walls=True, label="Cap TW: end plate, ears, pin and side walls")}
ZE = B.TIP["z_end"]                                   # -98.46
X_EQ = B.TIP["x_palmar"] + B.ELL["ax"]                # -2.05: the pad's straight flank ends, its half-ellipsoid begins
RHO_PLA = 1.24


def wall_z0():
    """The walls' proximal end: window_gap past the distal edge of the U3 and U5 windows."""
    w = B.cover_windows()
    return round(min(w["U3"][2], w["U5"][2]) - CAP["window_gap"], 3)


def end_face():
    """Half-width and depth of the pad's flat tip end: |y| <= w_E over the straight flank, and the half-ellipsoid reaching
    d_E palmar of the flank's end x_F -2.05."""
    k = math.sqrt(max(1.0 - ((ZE - B.Z_CONTACT) / B.ELL["az"]) ** 2, 0.0))
    return B.ELL["ay"] * k, B.ELL["ax"] * k


def zcyl(K, x, y, r0, z0, z1, r1=None):
    """A cylinder (or a frustum to r1) along +z from z0 to z1."""
    r1 = r0 if r1 is None else r1
    if K.name == "mesh":
        m3 = B._m3()
        return m3.Manifold.cylinder(float(z1 - z0), float(r0), float(r1), 48).translate([float(x), float(y), float(z0)])
    V = K.App.Vector
    if abs(r1 - r0) < 1e-9:
        return K.Part.makeCylinder(r0, z1 - z0, V(x, y, z0), V(0, 0, 1))
    return K.Part.makeCone(r0, r1, z1 - z0, V(x, y, z0), V(0, 0, 1))


def flank(z, gap):
    """|y| of a face `gap` outside the pad's flank at z, measured normal to the flank's outline in the palmar view (the
    outline turns toward the tip end, where a gap along y alone would shrink to 0.26 of itself)."""
    zz = min(max(z, ZE), B.TIP["z_mount"])
    t = zz - B.Z_CONTACT
    a = B.ELL["azp"] if t >= 0 else B.ELL["az"]
    q = max(1.0 - (t / a) ** 2, 1e-9)
    slope = B.ELL["ay"] * (-t / a ** 2) / math.sqrt(q)          # d|y|/dz of the outline
    return B.pad_half_width(zz) + gap * math.sqrt(1.0 + slope ** 2)


def cap_part(K, walls):
    """One cap in F. Blocks first (ears and, on TW, walls), cut by one prism of the pad's outline grown by the variant's gap
    (so the inner faces come from one cutter), then the end plate and the pin, then the screw holes."""
    import numpy as np
    xs, ze = B.X_SLOT, ZE
    gap = CAP["wall_clear"] if walls else CAP["ear_gap_plain"]
    blocks = [K.box((xs - CAP["ear_t"], s0, ze - 0.05), (xs, s1, CAP["ear_z1"])) for s0, s1 in ((-B.Y_RIM, -1.0), (1.0, B.Y_RIM))]
    zw0, zw1 = CAP["ear_z1"] - 0.3, wall_z0()
    if walls:
        zs = np.linspace(zw1, zw0, 60)                       # proximal to distal
        for s in (-1, 1):
            outer = [(s * (flank(z, gap) + CAP["wall_t"]), z) for z in zs]
            inner = [(s * 1.0, z) for z in zs[::-1]]
            blocks.append(K.prism_x(outer + inner, xs - CAP["wall_h"], xs))
    zc = np.linspace(ze - 1.0, zw1 + 1.0, 160)
    foot = [(flank(z, gap), z) for z in zc] + [(-flank(z, gap), z) for z in zc[::-1]]
    sides = K.cut(K.fuse(blocks[0], blocks[1:]), [K.prism_x(foot, xs - CAP["wall_h"] - 1.0, xs + 0.5)])
    wE, dE = end_face()
    m = CAP["stem_margin"]
    a = np.linspace(0.0, math.pi, 49)
    stem = ([((wE + m), -0.05 + xs - CAP["ear_t"] + 0.1), ((wE + m), X_EQ)]
            + [((wE + m) * math.cos(t), X_EQ - (dE + m) * math.sin(t)) for t in a[1:-1]]
            + [(-(wE + m), X_EQ), (-(wE + m), -0.05 + xs - CAP["ear_t"] + 0.1)])
    plate = K.fuse(K.box((xs - CAP["ear_t"], -B.Y_RIM, ze - CAP["t_end"]), (B.X_TONGUE_BACK, B.Y_RIM, ze)),
                   [K.prism_z(stem, ze - CAP["t_end"], ze)])
    r, L, c = CAP["pin_d"] / 2, CAP["pin_l"], CAP["pin_chamfer"]
    pin = K.fuse(zcyl(K, CAP["pin_x"], 0.0, r, ze - 0.05, ze + L - c), [zcyl(K, CAP["pin_x"], 0.0, r, ze + L - c - 0.01, ze + L,
                                                                             r - c)])
    cap = K.fuse(plate, [sides, pin])
    holes = [K.cylx(y, z, B.SCREW_CLEAR / 2, xs - CAP["ear_t"] - 1.0, xs + 1.0) for y, z in screws_distal()]
    return K.finish(K.cut(cap, holes))


def screws_distal():
    return [(y, z) for y, z in B.screw_yz() if z < B.Z_CONTACT]


def screws_new():
    """The four board screws with the cap on: the proximal pair as now (M2 x 5, head on the cover), the distal pair M2 x
    screw_l with the head on the ears; their tips stay at x_F 7.90."""
    m3 = B._m3()
    out = []
    for y, z in B.screw_yz():
        x_head = B.X_SLOT - (CAP["ear_t"] if z < B.Z_CONTACT else 0.0)
        length = CAP["screw_l"] if z < B.Z_CONTACT else B.SCREW_L
        out.append(B.cyl_x(y, z, B.HEAD_D / 2, x_head - B.HEAD_H, x_head) + B.cyl_x(y, z, B.SCREW_D / 2, x_head - 0.01, x_head + length))
    return m3.Manifold.batch_boolean(out, m3.OpType.Add)


def distal_pair():
    """The two distal screws alone, M2 x screw_l with the head on the ears."""
    m3 = B._m3()
    xh = B.X_SLOT - CAP["ear_t"]
    return m3.Manifold.batch_boolean([B.cyl_x(y, z, B.HEAD_D / 2, xh - B.HEAD_H, xh) + B.cyl_x(y, z, B.SCREW_D / 2, xh - 0.01,
                                                                                               xh + CAP["screw_l"])
                                      for y, z in screws_distal()], m3.OpType.Add)


def pad_hole():
    """The pad's hole for the pin (F)."""
    return zcyl(B.MeshB(), CAP["pin_x"], 0.0, CAP["hole_d"] / 2, ZE - 1.0, ZE + CAP["hole_l"])


def round_trip(man, path):
    """Write an STL, read it back into manifold3d; the volume it keeps (the 2026-10-20 fork lost 725 mm3 this way)."""
    import trimesh
    tm = B.to_trimesh(man)
    tm.export(path)
    back = trimesh.load(path, force="mesh")
    back.merge_vertices()
    m2 = B.to_manifold(back)
    return dict(volume_mm3=round(float(man.volume()), 3), reread_mm3=round(float(m2.volume()), 3),
                reread_genus=int(m2.genus()), watertight=bool(back.is_watertight))


def cmd_build():
    import numpy as np
    import trimesh
    import dip_link_print_project as PP
    import dip_link_v2 as L2
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    K = B.MeshB()
    hw = B.hardware()
    pad = B.to_manifold(B.load_cad("pad_ell11_5"))
    pad_h = pad - pad_hole()
    scr = screws_new()
    others = {"pad_with_hole": pad_h, "cover": B.to_manifold(B.load_cad("cover")),
              "dip_link": B.to_manifold(B.load_cad("dip_link")), "board": hw["fr4"] + hw["board_parts"],
              "magnets": hw["magnets_ell11_5"], "screws_with_cap": scr}
    rec = dict(owner="2026-10-20 19:48 'the finger is getting popped out a lot ... a kind of T-shaped plate ... extrude it down to "
                     "mold around the top of the fingertip (and around the screws too)'; 20:36 'sure we can go with your "
                     "recommendation'", params=CAP, wall_z0_F=wall_z0(), end_face=dict(zip(("half_width_mm", "depth_mm"),
                                                                                         [round(v, 3) for v in end_face()])),
               pad_hole_overlap_magnets_mm3=round(float((pad_hole() ^ hw["magnets_ell11_5"]).volume()), 4),
               pad_hole_to_magnets_mm=round(float(pad_hole().min_gap(hw["magnets_ell11_5"], 5.0)), 3),
               screws_with_cap_to_pad_mm=round(float(scr.min_gap(pad_h, 3.0)), 3),     # the proximal heads in their notches
               distal_screws_to_pad_mm=round(float(distal_pair().min_gap(pad_h, 3.0)), 3),
               screws_with_cap_overlap_pad_mm3=round(float((scr ^ pad_h).volume()), 4), variants={})
    for vid, v in VARIANTS.items():
        cap = cap_part(K, v["walls"])
        stl = V3 / "cad" / f"{PREFIX}_tip_cap_{vid}.stl"
        rt = round_trip(cap, stl)
        ov = {k: round(float((cap ^ o).volume()), 4) for k, o in others.items()}
        ov["pad_without_hole"] = round(float((cap ^ pad).volume()), 3)
        gaps = {k: round(float(cap.min_gap(o, 3.0)), 3) for k, o in (("pad_with_hole", pad_h), ("cover", others["cover"]),
                                                                       ("dip_link", others["dip_link"]), ("board", others["board"]))}
        # the ears' and walls' faces to the pad's flank (the cap above the plate, without the pin): TP stands ear_gap_plain off,
        # TW touches
        side = (cap ^ B.box((-20, -20, ZE + 0.3), (20, 20, -80.0))) - B.box((CAP["pin_x"] - 1.6, -1.6, ZE), (CAP["pin_x"] + 1.6, 1.6,
                                                                                                            ZE + CAP["pin_l"] + 0.5))
        gaps["sides_to_pad"] = round(float(side.min_gap(pad_h, 3.0)), 3)
        tm = B.to_trimesh(cap)
        pts = B.sample_part(tm)
        v_c, i_c, q_c, part_c = B.clearance_exact(pts, reach=1.6, workers=2)
        bam = L2.bambu_stl(tm, np.eye(3), V3 / "print" / f"{PREFIX}_tip_cap_{vid}_pla_bambu.stl")
        rec["variants"][vid] = dict(
            label=v["label"], walls=v["walls"], stl=PP.rel(stl), bambu=bam, round_trip=rt,
            volume_mm3=rt["volume_mm3"], mass_solid_g=round(rt["volume_mm3"] * RHO_PLA / 1e3, 3),
            bounds_F=np.round(tm.bounds, 3).tolist(), width_mm=round(float(tm.bounds[1][1] - tm.bounds[0][1]), 3),
            overlap_mm3=ov, gaps_mm=gaps,
            pip_sweep=dict(min_mm=round(v_c, 3), exact=q_c is not None, pip_deg=q_c, part=part_c, reach_mm=1.6,
                           note="least distance to the MCP link and the PIP servo over PIP -18..92 deg; a lower bound when "
                                "every point stays beyond reach"))
        print(vid, json.dumps(rec["variants"][vid], default=float), flush=True)
    pip_to_tip = round(B.TIP["z_end"] - CAP["t_end"], 3)
    rec["finger_length"] = dict(pip_axis_z_F=-51.55, tip_end_z_F=ZE, with_cap_z_F=pip_to_tip,
                                pip_to_tip_mm=round(-51.55 - ZE, 2), pip_to_tip_with_cap_mm=round(-51.55 - pip_to_tip, 2))
    rec["screws"] = dict(distal="M2 x 8 (brass or A2 stainless), head on the ear, tip at x_F 7.90 as the M2 x 5 now",
                         proximal="M2 x 5 as now")
    problems = []
    for vid, r in rec["variants"].items():
        if r["round_trip"]["reread_mm3"] < 0.999 * r["round_trip"]["volume_mm3"] or not r["round_trip"]["watertight"]:
            problems.append(f"{vid}: STL round trip {r['round_trip']}")
        bad = {k: x for k, x in r["overlap_mm3"].items() if k != "pad_without_hole" and x > 0.05}
        if bad:
            problems.append(f"{vid}: overlaps {bad}")
        if r["pip_sweep"]["min_mm"] < 0.5:
            problems.append(f"{vid}: PIP sweep {r['pip_sweep']}")
        if r["width_mm"] > 21.0 + 1e-6:
            problems.append(f"{vid}: width {r['width_mm']}")
    if rec["pad_hole_to_magnets_mm"] < 2.0:
        problems.append(f"pin hole {rec['pad_hole_to_magnets_mm']} mm from a magnet")
    if rec["screws_with_cap_overlap_pad_mm3"] > 0.0:
        problems.append("screw heads overlap the pad")
    rec.update(problems=problems, when=time.strftime("%Y-%m-%d %H:%M"), script="scripts/dip_b_tip_cap.py")
    old = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    rec["step"] = old.get("step")
    RECORD.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print("problems:", problems, f"[{time.time() - t0:.0f} s]", flush=True)


def fc_main():
    """Inside freecadcmd: both caps as B-rep, STEP and STL into v3/b/cad, a log line per part."""
    import MeshPart
    K = B.BrepB()
    log = {}
    for vid, v in VARIANTS.items():
        s = cap_part(K, v["walls"])
        step = V3 / "cad" / f"{PREFIX}_tip_cap_{vid}.step"
        s.exportStep(str(step))
        msh = MeshPart.meshFromShape(Shape=s, LinearDeflection=0.01, AngularDeflection=0.1, Relative=False)
        msh.write(str(WORK / f"tip_cap_{vid}_from_step.stl"))
        log[vid] = dict(step=str(step.relative_to(REPO)), volume_mm3=round(s.Volume, 3), valid=bool(s.isValid()),
                        solids=len(s.Solids), faces=len(s.Faces))
        print("FC", vid, json.dumps(log[vid]), flush=True)
    (WORK / "fc_tip_cap.json").write_text(json.dumps(log, indent=1))
    print("FC done", flush=True)


def cmd_step():
    WORK.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    env["TIPCAP_FREECAD"] = "1"
    r = subprocess.run(["freecadcmd", str(Path(__file__).resolve())], env=env, cwd=str(REPO), capture_output=True, text=True)
    out = [l for l in r.stdout.splitlines() if l.startswith("FC")]
    print("\n".join(out), flush=True)
    if r.returncode != 0 or "FC done" not in r.stdout:
        print(r.stderr[-3000:])
        raise SystemExit("freecadcmd failed")
    log = json.loads((WORK / "fc_tip_cap.json").read_text())
    rec = json.loads(RECORD.read_text())
    for vid, l in log.items():
        l["mesh_volume_mm3"] = rec["variants"][vid]["volume_mm3"]
        l["volume_diff_pct"] = round(100 * (l["volume_mm3"] / l["mesh_volume_mm3"] - 1), 3)
    rec["step"] = log
    RECORD.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    bad = [k for k, l in log.items() if not l["valid"] or l["solids"] != 1 or abs(l["volume_diff_pct"]) > 1.0]
    if bad:
        raise SystemExit(f"STEP problems: {bad} {log}")


def cmd_render():
    """Three renders of the tip with the caps (pyvista), composed with labels: TP on the finger, TW on the finger, TW pulled
    off along the finger axis with the distal screws pulled out."""
    import numpy as np
    import trimesh
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    hw = B.hardware()
    pad = B.to_trimesh(B.to_manifold(B.load_cad("pad_ell11_5")) - pad_hole())
    base = {"dip_link": B.load_cad("dip_link"), "cover": B.load_cad("cover"), "fr4": B.to_trimesh(hw["fr4"]),
            "board_parts": B.to_trimesh(hw["board_parts"]), "magnets": B.to_trimesh(hw["magnets_ell11_5"]), "pad": pad}
    cols = dict(B.COLORS, cap="#e07b24")
    scr_prox = [B.cyl_x(y, z, B.HEAD_D / 2, B.X_SLOT - B.HEAD_H, B.X_SLOT) + B.cyl_x(y, z, B.SCREW_D / 2, B.X_SLOT - 0.01,
                                                                                    B.X_SLOT + B.SCREW_L)
                for y, z in B.screw_yz() if z > B.Z_CONTACT]

    def distal_screws(dx):
        xh = B.X_SLOT - CAP["ear_t"] - dx
        return [B.cyl_x(y, z, B.HEAD_D / 2, xh - B.HEAD_H, xh) + B.cyl_x(y, z, B.SCREW_D / 2, xh - 0.01, xh + CAP["screw_l"])
                for y, z in screws_distal()]

    def shot(meshes, path, focal, off, zoom, up=(0, 0, 1)):
        p = B._plotter((1300, 1100))
        for k, m in meshes:
            p.add_mesh(B._pv(m), color=cols[k], opacity=0.55 if k == "pad" else 1.0, smooth_shading=True,
                       split_sharp_edges=True, specular=0.2)
        p.camera.position = tuple(np.asarray(focal) + np.asarray(off))
        p.camera.focal_point = tuple(focal)
        p.camera.up = up
        p.camera.zoom(zoom)
        p.screenshot(str(path))
        p.close()

    WORK.mkdir(parents=True, exist_ok=True)
    off = (-60.0, -48.0, -38.0)
    panels = []
    for vid, view in (("TP", "on"), ("TW", "on"), ("TW", "exploded"), ("TW", "end")):
        cap = trimesh.load(V3 / "cad" / f"{PREFIX}_tip_cap_{vid}.stl", force="mesh")
        if view == "exploded":
            cap.apply_translation([0.0, 0.0, -14.0])
        scr = [B.to_trimesh(s) for s in scr_prox + distal_screws(16.0 if view == "exploded" else 0.0)]
        meshes = [(k, m) for k, m in base.items()] + [("cap", cap)] + [("screws", s) for s in scr]
        path = WORK / f"render_{vid}_{view}.png"
        if view == "end":                                    # from beyond the tip, dorsal up
            shot(meshes, path, (0.0, 2.0, -98.0), (0.0, 0.0, -90.0), 2.1, up=(1, 0, 0))
        else:
            shot(meshes, path, (0.0, 0.0, -102.0 if view == "exploded" else -92.0), off, 1.15 if view == "exploded" else 1.7)
        title = {"on": VARIANTS[vid]["label"].split(":")[0] + " on the finger",
                 "exploded": "Cap TW pulled off, distal screws out",
                 "end": "Cap TW from beyond the tip"}[view]
        panels.append((path, title))
    dpi = 100
    fig = plt.figure(figsize=(13.0, 12.6), dpi=dpi)
    for i, (path, title) in enumerate(panels):
        ax = fig.add_axes([0.5 * (i % 2), 0.53 - 0.44 * (i // 2), 0.5, 0.42])
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_axis_off()
        ax.set_title(title, fontsize=17, family="serif")
    keys = ["cap", "pad", "cover", "dip_link", "fr4", "magnets", "screws"]
    labels = ["Cap (PLA)", "85A pad B00, pin hole", "Cover (PLA)", "DIP link (PLA)", "ReSkin 5X board",
              "Magnets 3 x 2 mm", "M2 screws, distal pair x 8"]
    fig.legend([Patch(facecolor=cols[k], edgecolor="#333333") for k in keys], labels, loc="lower center", ncol=3, fontsize=16,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    out = V3 / "media" / f"{PREFIX}_tip_caps.png"
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    print("->", out.relative_to(REPO), flush=True)


def main():
    cmds = sys.argv[1:] or ["build", "step", "render"]
    for c in cmds:
        t0 = time.time()
        {"build": cmd_build, "step": cmd_step, "render": cmd_render}[c]()
        print(f"== {c} [{time.time() - t0:.0f} s]", flush=True)


if os.environ.get("TIPCAP_FREECAD"):
    fc_main()
elif __name__ == "__main__":
    main()
