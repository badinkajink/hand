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

Second revision (owner, 2026-10-21, after the pad pushed sideways off the cover W00 and snapped the walls' proximal ends,
0.66 mm fins between the proximal screws' counterbores and the mount face: "make the side walls on the bottom of the cassette
actually clamp a little (so 0 clearance or even some very maginal interference that squishes the finger in", "boost the bottom
screws up a little bit", "more solid/better connected"; "i anticipate these same issues for the tip caps, so you should beef
those up as well, particularly the one that curves around the ellipse further--tiny piece with sharp angles"):
  covers I05, I10, I15  dip_link_concept_b's cover_walls "solid" (outer face straight up the 21 mm edge with a 1 mm chamfer, the
          walls ending at the U3/U5 windows, the proximal screws' heads on solid wall 3 mm up (M2 x 8) and no wall above those
          seats toward the mount face) at 0.05, 0.10 and 0.15 mm interference per side; one, two and three notches at the
          distal edge right of centre (the W set's are left of it)
  TP2     TP with a 1.6 mm plate, 0.5 mm chamfers on the ears' outer and proximal top edges and a 0.4 mm fillet at the pin's root
  TW2     the thin walls replaced by solid blocks from the pad's flank (0.05 mm interference) to the 21 mm edge, 5 mm tall, the
          distal screws' heads on top (M2 x 10), the same chamfers, plate and pin
  pad B00R  B00 with the pin's hole (its mouth chamfered 0.5 mm for the fillet) and the proximal notches 3 mm deeper for the
          raised heads (scripts/v3b_pad_b00_magnets_xl.py --pin-hole --raised-seats)
  covers2 / step2 / render2: the second revision's covers and caps with their checks
          (20261021-dip_link_v3b_solid_covers_and_caps2.json), STEP, and media/20261021-dip_link_v3b_solid_covers_and_caps2.png

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


def revolved(K, x, y, prof):
    """A solid of revolution about the line (x, y, z): prof is its outline [(r, z)] from the axis outward and back; one
    profile, so a fillet or a chamfer joins the cylinder without coincident faces."""
    if K.name == "mesh":
        m3 = B._m3()
        return m3.Manifold.revolve(m3.CrossSection([[(float(r), float(z)) for r, z in prof]]), 48).translate([float(x), float(y), 0.0])
    V = K.App.Vector
    pts = [V(x + r, y, z) for r, z in prof]
    face = K.Part.Face(K.Part.makePolygon(pts + [pts[0]]))
    return face.revolve(V(x, y, 0), V(0, 0, 1), 360)


def pad_hole(chamfer=0.0):
    """The pad's hole for the pin (F), its mouth chamfered `chamfer` at 45 deg."""
    K = B.MeshB()
    if chamfer <= 0:
        return zcyl(K, CAP["pin_x"], 0.0, CAP["hole_d"] / 2, ZE - 1.0, ZE + CAP["hole_l"])
    r = CAP["hole_d"] / 2
    return revolved(K, CAP["pin_x"], 0.0, [(0.0, ZE - 1.0), (r + chamfer + 1.0, ZE - 1.0), (r, ZE + chamfer),
                                           (r, ZE + CAP["hole_l"]), (0.0, ZE + CAP["hole_l"])])


# ------------------------------------------------------------------------------------------- second revision (2026-10-21)
PREFIX2 = "20261021-dip_link_v3b"
RECORD2 = V3 / f"{PREFIX2}_solid_covers_and_caps2.json"
CAP2 = dict(t_end=1.6, chamfer=0.5, pin_fillet=0.4, hole_chamfer=0.5, block_h=5.0, block_clear=-0.05)
VARIANTS2 = {"TP2": dict(side="ears", label="Cap TP2: end plate, ears and pin"),
             "TW2": dict(side="blocks", label="Cap TW2: end plate, solid side blocks and pin")}
COVERS2 = {"I05": -0.05, "I10": -0.10, "I15": -0.15}     # wall clearance per side; negative: interference


class b_state:
    """Context: dip_link_concept_b's DESIGN and P keys, pad set and groove fit set for its duration, then restored."""

    def __init__(self, design=None, params=None, tips=None, groove_fit=None):
        self.new = (design or {}, params or {}, tips, groove_fit)

    def __enter__(self):
        d, q, tips, gf = self.new
        self.old = ({k: B.DESIGN.get(k) for k in d}, {k: B.P.get(k) for k in q}, B.TIPS, B.GROOVE_FIT)
        B.DESIGN.update(d)
        B.P.update(q)
        if tips is not None:
            B.TIPS = tips
        if gf is not None:
            B.GROOVE_FIT = gf
        return self

    def __exit__(self, *exc):
        d, q, B.TIPS, B.GROOVE_FIT = self.old
        for src, dst in ((d, B.DESIGN), (q, B.P)):
            for k, v in src.items():
                if v is None:
                    dst.pop(k, None)
                else:
                    dst[k] = v
        return False


def side_dims(v):
    """(height along x palmar of the cover, proximal end z_F, gap to the pad's flank) of a second-revision cap's sides."""
    if v["side"] == "blocks":
        return CAP2["block_h"], wall_z0(), CAP2["block_clear"]
    return CAP["ear_t"], CAP["ear_z1"], CAP["ear_gap_plain"]


def cap_part2(K, v):
    """A second-revision cap in F: TP2's ears or TW2's solid blocks (from the pad's flank, grown by the variant's gap, out to
    the cover's edge), 45 deg chamfers along their outer and proximal top edges, a 1.6 mm end plate, the pin with a fillet at
    its root, and the distal screws' holes; their heads sit on the sides' top faces."""
    import numpy as np
    xs, ze = B.X_SLOT, ZE
    te, ch = CAP2["t_end"], CAP2["chamfer"]
    H, zt, gap = side_dims(v)
    blocks = [K.box((xs - H, s0, ze - 0.05), (xs, s1, zt)) for s0, s1 in ((-B.Y_RIM, -1.0), (1.0, B.Y_RIM))]
    zc = np.linspace(ze - 1.0, zt + 1.0, 160)
    foot = [(flank(z, gap), z) for z in zc] + [(-flank(z, gap), z) for z in zc[::-1]]
    cuts = [K.prism_x(foot, xs - H - 1.0, xs + 0.5)]
    cuts += [K.prism_z([(s * (B.Y_RIM - ch - 1.0), xs - H - 1.0), (s * (B.Y_RIM + 1.0), xs - H + ch + 1.0),
                        (s * (B.Y_RIM + 1.0), xs - H - 1.0)], ze - te - 1.0, zt + 1.0) for s in (-1, 1)]
    cuts.append(K.prism_y([(xs - H - 1.0, zt - ch - 1.0), (xs - H + ch + 1.0, zt + 1.0), (xs - H - 1.0, zt + 1.0)],
                          -B.Y_RIM - 1.0, B.Y_RIM + 1.0))
    sides = K.cut(K.fuse(blocks[0], blocks[1:]), cuts[:1])
    wE, dE = end_face()
    m = CAP["stem_margin"]
    a = np.linspace(0.0, math.pi, 49)
    top = xs - H + 0.05
    stem = ([((wE + m), top), ((wE + m), X_EQ)] + [((wE + m) * math.cos(t), X_EQ - (dE + m) * math.sin(t)) for t in a[1:-1]]
            + [(-(wE + m), X_EQ), (-(wE + m), top)]) if top > X_EQ - (dE + m) else None
    plate = K.box((xs - H, -B.Y_RIM, ze - te), (B.X_TONGUE_BACK, B.Y_RIM, ze))
    if stem is not None:
        plate = K.fuse(plate, [K.prism_z(stem, ze - te, ze)])
    r, L, c, f = CAP["pin_d"] / 2, CAP["pin_l"], CAP["pin_chamfer"], CAP2["pin_fillet"]
    pin = revolved(K, CAP["pin_x"], 0.0, [(0.0, ze - 0.05), (r + f, ze - 0.05), (r, ze + f), (r, ze + L - c), (r - c, ze + L),
                                          (0.0, ze + L)])
    cap = K.cut(K.fuse(plate, [sides]), cuts[1:])
    cap = K.fuse(cap, [pin])
    holes = [K.cylx(y, z, B.SCREW_CLEAR / 2, xs - H - 1.0, xs + 1.0) for y, z in screws_distal()]
    return K.finish(K.cut(cap, holes))


def solid_cover(K, clear, cid=None):
    """Cover with cover_walls "solid" at wall clearance `clear` (F); with an ID, its notches at the distal edge right of the
    centre line (one, two, three for I05, I10, I15)."""
    with b_state(design=dict(cover_walls="solid"), params=dict(wall_clear=clear)):
        cov = B.cover_part(K)
    if cid is None:
        return cov
    k = list(COVERS2).index(cid)
    notches = [K.box((B.X_SLOT - 0.2, 2.0 + 1.6 * j, ZE - 0.1), (B.X_COMP + 0.2, 3.0 + 1.6 * j, ZE + 0.8)) for j in range(k + 1)]
    return K.finish(K.cut(cov, notches))


def pad_rev2():
    """Pad B00R in F: the five-magnet 11/22 pad at groove fit 0.00 with the proximal notches seat_raise deeper and the pin's
    hole, its mouth chamfered."""
    with b_state(design=dict(cover_walls="solid"), tips=B.PAD_SETS["v2_five"], groove_fit=0.0):
        pad, _ = B.make_pad("ell11_5")
    return pad - pad_hole(CAP2["hole_chamfer"])


def screws2(H):
    """The four board screws with a solid-wall cover and a second-revision cap: the proximal pair M2 x (5 + seat_raise) on
    their raised seats, the distal pair M2 x (5 + H) on the cap's sides; every tip at x_F 7.90."""
    m3 = B._m3()
    with b_state(design=dict(cover_walls="solid")):
        sr = B.rp("seat_raise")
    out = []
    for y, z in B.screw_yz():
        up = sr if z > B.Z_CONTACT else H
        xh = B.X_SLOT - up
        out.append(B.cyl_x(y, z, B.HEAD_D / 2, xh - B.HEAD_H, xh) + B.cyl_x(y, z, B.SCREW_D / 2, xh - 0.01, B.X_SLOT + B.SCREW_L))
    return m3.Manifold.batch_boolean(out, m3.OpType.Add), sr


def cmd_caps2_and_covers():
    """Both second-revision caps and the three solid-wall covers as meshes, Bambu STLs and checks."""
    import numpy as np
    import dip_link_print_project as PP
    import dip_link_v2 as L2
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    K = B.MeshB()
    hw = B.hardware()
    link = B.to_manifold(B.load_cad("dip_link"))
    board = hw["fr4"] + hw["board_parts"]
    mags = hw["magnets_ell11_5"]
    pad_r = pad_rev2()
    pad_b00 = B.to_manifold(B.load_cad("pad_ell11_5")) - pad_hole()          # B00P: the 1.5 mm notches
    rec = dict(owner="2026-10-21: 'make the side walls on the bottom of the cassette actually clamp a little (so 0 clearance or "
                     "even some very maginal interference that squishes the finger in'; 'boost the bottom screws up a little bit'; "
                     "'the side walls on the cassette actually snapped off just now (the bottom fin beneath the hole)'; 'beef those "
                     "up as well, particularly the one that curves around the ellipse further'", params=dict(CAP2, **{
                         k: B.REV[k] for k in ("seat_raise", "wall_chamfer")}), covers={}, caps={},
               pad_B00R=dict(volume_mm3=round(float(pad_r.volume()), 2), hole_to_magnets_mm=round(float(pad_hole(
                   CAP2["hole_chamfer"]).min_gap(mags, 5.0)), 3)))
    for cid, clear in COVERS2.items():
        cov = solid_cover(K, clear, cid)
        stl = V3 / "cad" / f"{PREFIX2}_cover_{cid}.stl"
        rt = round_trip(cov, stl)
        scr, sr = screws2(CAP["ear_t"])
        tm = B.to_trimesh(cov)
        bam = L2.bambu_stl(tm, PP.R_CLAMP, V3 / "print" / f"{PREFIX2}_cover_{cid}_pla_bambu.stl")
        r = dict(wall_clear_mm=clear, stl=PP.rel(stl), bambu=bam, round_trip=rt, volume_mm3=rt["volume_mm3"],
                 mass_solid_g=round(rt["volume_mm3"] * RHO_PLA / 1e3, 3), width_mm=round(float(tm.bounds[1][1] - tm.bounds[0][1]), 3),
                 interference_with_B00R_mm3=round(float((cov ^ pad_r).volume()), 3),
                 overlap_mm3=dict(dip_link=round(float((cov ^ link).volume()), 4), board=round(float((cov ^ board).volume()), 4),
                                  screws=round(float((cov ^ scr).volume()), 4)),
                 notches=list(COVERS2).index(cid) + 1)
        if cid == "I05":
            pts = B.sample_part(tm)
            v_c, i_c, q_c, part_c = B.clearance_exact(pts, reach=1.6, workers=2)
            r["pip_sweep"] = dict(min_mm=round(v_c, 3), exact=q_c is not None, pip_deg=q_c, part=part_c, reach_mm=1.6)
            prox = B._m3().Manifold.batch_boolean([B.cyl_x(y, z, B.HEAD_D / 2, B.X_SLOT - sr - B.HEAD_H, B.X_SLOT - sr)
                                                   for y, z in B.screw_yz() if z > B.Z_CONTACT], B._m3().OpType.Add)
            rec["raised_heads"] = dict(seat_above_cover_mm=sr, screw="M2 x 8", to_B00R_mm=round(float(prox.min_gap(pad_r, 3.0)), 3),
                                       overlap_B00R_mm3=round(float((prox ^ pad_r).volume()), 4),
                                       overlap_B00_notch_1p5_mm3=round(float((prox ^ pad_b00).volume()), 3))
        rec["covers"][cid] = r
        print(cid, json.dumps(r, default=float), flush=True)
    cov05 = solid_cover(K, COVERS2["I05"])
    for vid, v in VARIANTS2.items():
        cap = cap_part2(K, v)
        H, zt, gap = side_dims(v)
        stl = V3 / "cad" / f"{PREFIX2}_tip_cap_{vid}.stl"
        rt = round_trip(cap, stl)
        scr, _ = screws2(H)
        others = dict(cover_I05=cov05, dip_link=link, board=board, magnets=mags, screws=scr)
        ov = {k: round(float((cap ^ o).volume()), 4) for k, o in others.items()}
        ov["pad_B00R"] = round(float((cap ^ pad_r).volume()), 3)
        ov["pad_B00P"] = round(float((cap ^ pad_b00).volume()), 3)
        side = (cap ^ B.box((-20, -20, ZE + 0.3), (20, 20, -80.0))) - B.box((CAP["pin_x"] - 1.6, -1.6, ZE), (CAP["pin_x"] + 1.6, 1.6,
                                                                                                            ZE + CAP["pin_l"] + 0.5))
        distal = B._m3().Manifold.batch_boolean([B.cyl_x(y, z, B.HEAD_D / 2, B.X_SLOT - H - B.HEAD_H, B.X_SLOT - H)
                                                 for y, z in screws_distal()], B._m3().OpType.Add)
        gaps = dict(sides_to_B00R=round(float(side.min_gap(pad_r, 3.0)), 3), distal_heads_to_B00R=round(float(distal.min_gap(pad_r, 3.0)), 3),
                    cover_I05=round(float(cap.min_gap(cov05, 3.0)), 3))
        tm = B.to_trimesh(cap)
        pts = B.sample_part(tm)
        v_c, i_c, q_c, part_c = B.clearance_exact(pts, reach=1.6, workers=2)
        bam = L2.bambu_stl(tm, np.eye(3), V3 / "print" / f"{PREFIX2}_tip_cap_{vid}_pla_bambu.stl")
        rec["caps"][vid] = dict(label=v["label"], side=v["side"], side_height_mm=H, side_proximal_end_z_F=zt, gap_to_flank_mm=gap,
                                distal_screws=f"M2 x {int(round(5 + H))}", stl=PP.rel(stl), bambu=bam, round_trip=rt,
                                volume_mm3=rt["volume_mm3"], mass_solid_g=round(rt["volume_mm3"] * RHO_PLA / 1e3, 3),
                                bounds_F=np.round(tm.bounds, 3).tolist(), width_mm=round(float(tm.bounds[1][1] - tm.bounds[0][1]), 3),
                                overlap_mm3=ov, gaps_mm=gaps,
                                pip_sweep=dict(min_mm=round(v_c, 3), exact=q_c is not None, pip_deg=q_c, part=part_c, reach_mm=1.6))
        print(vid, json.dumps(rec["caps"][vid], default=float), flush=True)
    tip = round(ZE - CAP2["t_end"], 3)
    rec["finger_length"] = dict(tip_end_with_cap_z_F=tip, pip_to_tip_with_cap_mm=round(-51.55 - tip, 2))
    problems = []
    for grp in ("covers", "caps"):
        for k, r in rec[grp].items():
            if r["round_trip"]["reread_mm3"] < 0.999 * r["round_trip"]["volume_mm3"] or not r["round_trip"]["watertight"]:
                problems.append(f"{k}: STL round trip {r['round_trip']}")
            bad = {q: x for q, x in r["overlap_mm3"].items() if not q.startswith("pad_") and x > 0.05}
            if bad:
                problems.append(f"{k}: overlaps {bad}")
            if r["width_mm"] > 21.0 + 1e-6:
                problems.append(f"{k}: width {r['width_mm']}")
            if "pip_sweep" in r and r["pip_sweep"]["min_mm"] < 0.5:
                problems.append(f"{k}: PIP sweep {r['pip_sweep']}")
    if rec["raised_heads"]["overlap_B00R_mm3"] > 0:
        problems.append(f"raised heads overlap B00R {rec['raised_heads']}")
    if rec["pad_B00R"]["hole_to_magnets_mm"] < 2.0:
        problems.append(f"B00R hole {rec['pad_B00R']['hole_to_magnets_mm']} mm from a magnet")
    rec.update(problems=problems, when=time.strftime("%Y-%m-%d %H:%M"), script="scripts/dip_b_tip_cap.py")
    old = json.loads(RECORD2.read_text()) if RECORD2.exists() else {}
    rec["step"] = old.get("step")
    RECORD2.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print("problems:", problems, f"[{time.time() - t0:.0f} s]", flush=True)


def round_trip(man, path):
    """Write an STL, read it back into manifold3d; the volume it keeps (the 2026-10-20 fork lost 725 mm3 this way)."""
    import trimesh
    tm = B.to_trimesh(man)
    tm.update_faces(tm.nondegenerate_faces())                  # zero-area slivers where two chamfers meet at a corner
    tm.remove_unreferenced_vertices()
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


def fc_main2():
    """Inside freecadcmd: the solid-wall covers and the second-revision caps as B-rep, STEP into v3/b/cad."""
    K = B.BrepB()
    log = {}
    parts = [(cid, lambda cid=cid, c=c: solid_cover(K, c, cid), f"{PREFIX2}_cover_{cid}") for cid, c in COVERS2.items()]
    parts += [(vid, lambda v=v: cap_part2(K, v), f"{PREFIX2}_tip_cap_{vid}") for vid, v in VARIANTS2.items()]
    for pid, build, name in parts:
        s = build()
        step = V3 / "cad" / f"{name}.step"
        s.exportStep(str(step))
        log[pid] = dict(step=str(step.relative_to(REPO)), volume_mm3=round(s.Volume, 3), valid=bool(s.isValid()),
                        solids=len(s.Solids), faces=len(s.Faces))
        print("FC", pid, json.dumps(log[pid]), flush=True)
    (WORK / "fc_rev2.json").write_text(json.dumps(log, indent=1))
    print("FC done", flush=True)


def cmd_step2():
    WORK.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    env["TIPCAP_FREECAD"] = "2"
    r = subprocess.run(["freecadcmd", str(Path(__file__).resolve())], env=env, cwd=str(REPO), capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith("FC")), flush=True)
    if r.returncode != 0 or "FC done" not in r.stdout:
        print(r.stderr[-3000:])
        raise SystemExit("freecadcmd failed")
    log = json.loads((WORK / "fc_rev2.json").read_text())
    rec = json.loads(RECORD2.read_text())
    for pid, l in log.items():
        grp = "covers" if pid in COVERS2 else "caps"
        l["mesh_volume_mm3"] = rec[grp][pid]["volume_mm3"]
        l["volume_diff_pct"] = round(100 * (l["volume_mm3"] / l["mesh_volume_mm3"] - 1), 3)
    rec["step"] = log
    RECORD2.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    bad = [k for k, l in log.items() if not l["valid"] or l["solids"] != 1 or abs(l["volume_diff_pct"]) > 1.0]
    if bad:
        raise SystemExit(f"STEP problems: {bad} {log}")


def cmd_render2():
    """Four renders, composed with labels: cover I05 with its four screws; the finger with TW2 and with TP2 (pad B00R
    translucent); and the proximal end of I05 with the raised seat and the pad's deeper notch."""
    import numpy as np
    import trimesh
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    hw = B.hardware()
    cols = dict(B.COLORS, cap="#e07b24")
    cover = trimesh.load(V3 / "cad" / f"{PREFIX2}_cover_I05.stl", force="mesh")
    base = {"dip_link": B.load_cad("dip_link"), "fr4": B.to_trimesh(hw["fr4"]), "board_parts": B.to_trimesh(hw["board_parts"]),
            "magnets": B.to_trimesh(hw["magnets_ell11_5"]), "pad": B.to_trimesh(pad_rev2())}

    def shot(meshes, path, focal, off, zoom, up=(0, 0, 1)):
        p = B._plotter((1300, 1100))
        for k, m in meshes:
            p.add_mesh(B._pv(m), color=cols[k], opacity=0.5 if k == "pad" else 1.0, smooth_shading=True,
                       split_sharp_edges=True, specular=0.2)
        p.camera.position = tuple(np.asarray(focal) + np.asarray(off))
        p.camera.focal_point = tuple(focal)
        p.camera.up = up
        p.camera.zoom(zoom)
        p.screenshot(str(path))
        p.close()

    WORK.mkdir(parents=True, exist_ok=True)
    panels = []
    scr_tp, _ = screws2(CAP["ear_t"])
    scr_tw, _ = screws2(CAP2["block_h"])
    p0 = WORK / "r2_cover.png"
    with b_state(design=dict(cover_walls="solid")):
        scr_cover = B.hardware()["screws"]                   # without a cap: the distal pair M2 x 5 on the cover's face
    shot([("cover", cover), ("screws", B.to_trimesh(scr_cover))], p0, (0.0, 0.0, -86.0), (-62.0, -46.0, 26.0), 1.45)
    panels.append((p0, "Cover I05: solid walls, proximal heads 3 mm up"))
    for vid, scr in (("TW2", scr_tw), ("TP2", scr_tp)):
        cap = trimesh.load(V3 / "cad" / f"{PREFIX2}_tip_cap_{vid}.stl", force="mesh")
        path = WORK / f"r2_{vid}.png"
        meshes = list(base.items()) + [("cover", cover), ("cap", cap), ("screws", B.to_trimesh(scr))]
        shot(meshes, path, (0.0, 0.0, -90.0), (-62.0, -48.0, -30.0), 1.35)
        panels.append((path, f"Cap {vid} on the finger, pad B00R"))
    p3 = WORK / "r2_seat.png"
    shot([(k, m) for k, m in base.items() if k != "dip_link"] + [("cover", cover), ("screws", B.to_trimesh(scr_tw))], p3,
         (0.5, 7.5, -80.0), (-38.0, 26.0, -16.0), 2.3)
    panels.append((p3, "Raised seat and the pad's deeper notch"))
    dpi = 100
    fig = plt.figure(figsize=(13.0, 12.6), dpi=dpi)
    for i, (path, title) in enumerate(panels):
        ax = fig.add_axes([0.5 * (i % 2), 0.53 - 0.44 * (i // 2), 0.5, 0.42])
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_axis_off()
        ax.set_title(title, fontsize=17, family="serif")
    keys = ["cap", "cover", "pad", "dip_link", "fr4", "magnets", "screws"]
    labels = ["Cap (PLA)", "Cover I05 (PLA)", "85A pad B00R", "DIP link (PLA)", "ReSkin 5X board", "Magnets 3 x 2 mm",
              "M2 screws"]
    fig.legend([Patch(facecolor=cols[k], edgecolor="#333333") for k in keys], labels, loc="lower center", ncol=4, fontsize=16,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    out = V3 / "media" / f"{PREFIX2}_solid_covers_and_caps2.png"
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    print("->", out.relative_to(REPO), flush=True)


def main():
    cmds = sys.argv[1:] or ["build", "step", "render"]
    for c in cmds:
        t0 = time.time()
        {"build": cmd_build, "step": cmd_step, "render": cmd_render,
         "covers2": cmd_caps2_and_covers, "step2": cmd_step2, "render2": cmd_render2}[c]()
        print(f"== {c} [{time.time() - t0:.0f} s]", flush=True)


if os.environ.get("TIPCAP_FREECAD") == "1":
    fc_main()
elif os.environ.get("TIPCAP_FREECAD") == "2":
    fc_main2()
elif __name__ == "__main__":
    main()
