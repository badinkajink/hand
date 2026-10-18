#!/usr/bin/env python3
"""Final DIP link for the ReSkin 5X board (2026-10-18): concept a's window plate and dorsal clamp (judged against
concepts b and c in docs/experiments/20261018-dip_link_board_mount/decision.json), built on the tips the owner prints at
the 2.9 mm magnet seat.

Owner, 2026-10-18: "we are going to redesign the DIP linkage entirely so that the board can be as close to the printed
tip as possible, while trying to keep the contact geometry the same ... it's important that the board does not wiggle
at all--youll want something to fasten/sandwich it in place"; the seat coupon chose 2.9 mm, so the printed tips have the
11/22 half-ellipsoid's magnets 7.5 mm under the apex and the five-magnet block's corner magnets 7.4 mm apart (layout
five_z7p4, commit 85969c42).

The geometry of the DIP link, the clamp plate, the board model, the fasteners and the tips' key channels is
scripts/dip_link_concept_a.py's at its parameters; this script redirects that generator to final/, takes the tips from
the 2.9 mm print sheets of docs/experiments/20261016-sim2real_fingertip_readiness/print/, and adds the sensing estimate
at the as-printed magnets, the channels' distance to the magnet pockets (R4), a pin-fit coupon, a dimensioned drawing,
the XL print settings, the parts list and the assembly sequence.

    E=~/miniconda3/envs/femtip/bin/python
    RG="$HOME/.claude/bin/resguard.sh run --mem 2.5G --cpu 200 -- taskset -c 2-3 env NUMPY_MADVISE_HUGEPAGE=0"
    $RG $E scripts/dip_link_board_mount.py build       # parts, tips with channels, coupon -> final/cad, final/print
    $RG env DLF_CMD=step DLF_REPO=$PWD freecadcmd scripts/dip_link_board_mount.py   # STEP of the printed parts
    $RG $E scripts/dip_link_board_mount.py check       # clearance over PIP -18..92 deg, walls, masses, R4 distances
    $RG .venv/bin/python scripts/dip_link_board_mount.py sense    # force resolution and fields at the as-printed magnets
    $RG env MUJOCO_GL=egl .venv/bin/python scripts/dip_link_board_mount.py render
    $RG $E scripts/dip_link_board_mount.py drawing     # dimensioned drawing with the critical fits
    $E scripts/dip_link_board_mount.py docs            # print settings, parts list, assembly sequence
    $E scripts/dip_link_board_mount.py eval            # final/evaluation.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:                                   # freecadcmd executes the file without __file__
    REPO = Path(os.environ["DLF_REPO"])
os.environ.setdefault("DIPA_REPO", str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import dip_link_concept_a as A  # noqa: E402

STUDY = REPO / "docs/experiments/20261018-dip_link_board_mount"
FINAL = STUDY / "final"
CAD, PRINT, MEDIA = FINAL / "cad", FINAL / "print", FINAL / "media"
WORK = REPO / "logs/20261018-dip_link/final"          # context meshes, render caches, the fused DIP link STEP (untracked)
PFX = "20261018-dip_link_final"
READY = REPO / "docs/experiments/20261016-sim2real_fingertip_readiness"
SEAT = 2.9
SHEETS = {   # the 2.9 mm print set (commit 85969c42), 30 % gyroid sheets; the 50 % sheets share the geometry
    "block4": "20261016-siraya85a_block_four_d3x2_depth5_seat2p9_g30",
    "block5": "20261016-siraya85a_block_five_z7p4_d3x2_depth5_seat2p9_g30",
    "cap8.5": "20261016-siraya85a_tcd_cap8p5_four_d3x2_depth5_seat2p9_g30",
    "ell16": "20261016-siraya85a_tcd_ell7p4x8p5x16x32_four_d3x2_depth5p5_seat2p9_g30",
    "ell11": "20261016-siraya85a_tcd_ell7p4x8p5x11x22_four_d3x2_depth7p5_seat2p9_g30",
}
TIP_NAME = {"block4": "Block, four magnets", "block5": "Block, five magnets", "cap8.5": "Capsule r 8.5 mm",
            "ell16": "Half-ellipsoid 16/32 mm", "ell11": "Half-ellipsoid 11/22 mm"}
TIPS_CACHE = FINAL / f"{PFX}_tips.json"


def fname(stem, ext="stl"):
    """Files of the final design: tips in their print frame under print/, other STL and STEP under cad/, JSON in final/."""
    if stem.startswith("tip_") and stem.endswith("_print"):
        d = PRINT
    elif ext in ("stl", "step"):
        d = CAD
    else:
        d = FINAL
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{PFX}_{stem}.{ext}"


def spec_final():
    """spec.json with its tips replaced by the 2.9 mm print set (magnet positions, STL, outlines from the sheets). The
    tips are computed once (femtip env: the outlines need shapely) and cached beside the evaluation."""
    sp = json.loads((STUDY / "spec.json").read_text())
    if TIPS_CACHE.exists():
        sp["tips"] = json.loads(TIPS_CACHE.read_text())
        return sp
    import dip_link_spec as DS
    DS.TIPS = dict(SHEETS)
    tips = json.loads(json.dumps(DS.tips(), default=float))
    FINAL.mkdir(parents=True, exist_ok=True)
    TIPS_CACHE.write_text(json.dumps(tips, indent=1))
    sp["tips"] = tips
    return sp


# concept a's generator, pointed at final/ and at the 2.9 mm tips
A.OUT, A.MEDIA, A.CTX, A.PFX = FINAL, MEDIA, WORK, PFX
A.fname = fname
if os.environ.get("DLF_CMD") != "step":
    A.spec = spec_final


def ns(**kw):
    return argparse.Namespace(**kw)


# ----------------------------------------------------------------------------------------------- build
def pin_coupon():
    """PLA coupon: four pins of the clamp plate's form (diameters 2.25, 2.30, 2.35, 2.40 mm, 0.2 mm lead-in, 2.0 mm
    tall) on a 3.0 mm plate, to choose the pin that enters the board's plated 2.40 mm holes without play; a chamfered
    corner marks the 2.25 mm end. Same orientation as the clamp plate (plate on the bed, pins up)."""
    K = A.MeshK()
    ds = (2.25, 2.30, 2.35, 2.40)
    pitch, h0, hp = 6.0, 3.0, 2.0
    W = pitch * len(ds)
    plate = K.box((0.0, 0.0, 0.0), (W, 8.0, h0))
    cham = K.prism([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)], -0.1, h0 + 0.1, "xyz")
    plate = K.sub(plate, cham)
    pins = []
    for k, d in enumerate(ds):
        y, x = 4.0, pitch * (k + 0.5)
        r = d / 2
        pins.append(K.cyl("z", (x, y), r, h0 - 0.05, h0 + hp - 0.2, seg=64))
        pins.append(K.cyl("z", (x, y), r - 0.2, h0 + hp - 0.2, h0 + hp, r1=r, seg=64))
    return K.union([plate] + pins), list(ds)


def cmd_build(a):
    import trimesh
    for d in (CAD, PRINT, MEDIA, WORK):
        d.mkdir(parents=True, exist_ok=True)
    spec_final()
    A.cmd_build(a)
    rec = json.loads((FINAL / f"{PFX}_parts.json").read_text())
    rec["tips_source"] = {k: f"{READY.relative_to(REPO)}/print/{v}.json" for k, v in SHEETS.items()}
    rec["seat_mm"] = SEAT
    man, ds = pin_coupon()
    tm = A.to_trimesh(man)
    dst = PRINT / f"{PFX}_pin_fit_coupon.stl"
    tm.export(dst)
    rec["pin_fit_coupon"] = dict(file=str(dst.relative_to(REPO)), pin_d_mm=ds, volume_mm3=round(float(tm.volume), 1),
                                 watertight=bool(tm.is_watertight))
    (FINAL / f"{PFX}_parts.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"coupon: {tm.volume:.1f} mm3, watertight {tm.is_watertight}", flush=True)


# ----------------------------------------------------------------------------------------------- STEP (freecadcmd)
def freecad_step():
    """STEP of the DIP link's new features, the clamp plate and the tip cutter (exact solids, concept a's kernel), the
    fused DIP link zipped, and the two block tips with their channels as faceted solids, zipped."""
    import Mesh
    import Part
    A.freecad_step()
    rec = json.loads((FINAL / f"{PFX}_step.json").read_text())
    fused = WORK / f"{PFX}_dip_link.step"
    if fused.exists():
        z = CAD / f"{PFX}_dip_link.step.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(fused, fused.name)
        rec["dip_link"]["file_zip"] = str(z.relative_to(REPO))
        rec["dip_link"]["zip_bytes"] = z.stat().st_size
    for key in ("block4", "block5"):
        stl = PRINT / f"{PFX}_tip_{key}_print.stl"
        m = Mesh.Mesh(str(stl))
        sh = Part.Shape()
        sh.makeShapeFromMesh(m.Topology, 0.01)
        so = Part.makeSolid(Part.Shell(sh.Faces)) if not sh.Solids else sh.Solids[0]
        dst = WORK / f"{PFX}_tip_{key}_print.step"
        so.exportStep(str(dst))
        z = CAD / f"{PFX}_tip_{key}_print.step.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(dst, dst.name)
        rec[f"tip_{key}"] = dict(file_zip=str(z.relative_to(REPO)), valid=bool(so.isValid()), faces=len(so.Faces),
                                 volume_mm3=round(so.Volume, 2), mesh_volume_mm3=round(m.Volume, 2), zip_bytes=z.stat().st_size)
        print(f"tip {key}: valid {so.isValid()}, {len(so.Faces)} faces, {so.Volume:.1f} mm3 (mesh {m.Volume:.1f})", flush=True)
    rec["note_tips"] = ("The curved tips are given as STL only: their faceted STEP holds the same facets in 20-30 MB per tip. "
                        "In CAD, subtract cad/" + f"{PFX}_tip_cutter.step from a tip's solid to add the channels.")
    (FINAL / f"{PFX}_step.json").write_text(json.dumps(rec, indent=1))


# ----------------------------------------------------------------------------------------------- checks
def disc_distance(P, centre, x0, x1, R):
    """Least distance (mm) from points P (n, 3) to a solid disc magnet with its axis along x."""
    r = np.hypot(P[:, 1] - centre[1], P[:, 2] - centre[2])
    dx = np.maximum.reduce([x0 - P[:, 0], np.zeros(len(P)), P[:, 0] - x1])
    dr = np.maximum(r - R, 0.0)
    return np.hypot(dx, dr)


def cmd_check(a):
    import trimesh
    import dip_link_geometry as G
    A.cmd_check(ns(reuse=a.reuse, arms=False, skip_clearance=False))
    path = FINAL / f"{PFX}_check.json"
    res = json.loads(path.read_text())
    # R4: the key channels against each as-printed tip's magnets (3 x 2 mm discs; the pocket adds 0.2 mm of air above)
    cut = A.to_trimesh(A.tip_cutter(A.MeshK()))
    Pc = G.surface_points(cut, 0.1)
    sp = spec_final()
    r4 = {}
    for key, t in sp["tips"].items():
        best = (np.inf, None)
        for m in t["magnets"]["items"]:
            x0, x1 = m["x_F"]
            d = disc_distance(Pc, m["centre_F"], x0 - 0.2, x1, 1.5)
            i = int(np.argmin(d))
            if d[i] < best[0]:
                best = (float(d[i]), np.round(Pc[i], 2).tolist())
        r4[key] = dict(min_mm=round(best[0], 2), at_F=best[1], need_mm=2.0, ok=bool(best[0] >= 2.0),
                       magnets_x_F=t["magnets"]["items"][0]["x_F"], top_depth_mm=t["magnets"]["top_depth_mm"])
        print(f"R4 {key}: channels {best[0]:.2f} mm from a magnet", flush=True)
    res["channels_to_magnets_R4"] = r4
    tw = {}
    cutter = A.tip_cutter(A.MeshK())
    for key in ("block5",):                                   # concept a's check covers block4, cap8.5, ell16, ell11
        tF = A.to_trimesh(A.from_trimesh(A.tip_F(key)[0]) - cutter)
        tw[key] = {f"z{z:g}": A.section_min_wall(tF, (0, 0, z), (0, 0, 1)) for z in (-67.6, -69.5, -71.8)}
    res["tip_walls_near_channels"].update(tw)
    path.write_text(json.dumps(res, indent=1, default=float))


def cmd_sense(a):
    """Concept a's readout model at the final's sensing plane and at the as-printed magnets. Force resolution: the
    readiness finite elements of the 85A block (four magnets; five magnets with the corner magnets moved from 7.0 to the
    printed 7.4 mm spacing along the finger, their motions per newton kept from the 7.0 mm layout); static field, gain
    and chip-to-magnet distance for every printed tip at its sheet's magnet positions."""
    import sim2real_fingertip_sensing as SS
    import tpu_tip_design as TD
    import tpu_tip_magnet_sensing as MS
    L = A.levels()
    rows = [json.loads(l) for l in open(READY / "fe/fem.jsonl")]
    gaps = {"design": L["gap"], "chips_0.80": L["gap"] + (A.P["chip_h"] - 0.80), "chips_1.00": L["gap"] - (1.00 - A.P["chip_h"]),
            "plate_now": 3.42}
    sp = spec_final()
    z5 = sorted({round(v, 2) for m in sp["tips"]["block5"]["magnets"]["items"] for v in [A.Z_MOUNT - m["centre_F"][2]]})
    out = []
    for r0 in rows:
        if r0["layout"].endswith("_w1"):
            continue
        r = json.loads(json.dumps(r0))
        label = r["layout"]
        if r["layout"].startswith("five"):
            lo, hi = min(z5), max(z5)                          # the printed corner rows (from the mount face)
            for mg in r["magnets"]:
                if abs(mg[3] - 7.36) < 0.01:
                    mg[3] = lo
                elif abs(mg[3] - 14.36) < 0.01:
                    mg[3] = hi
            label = "five_z7p4"
        ys, zs, Gp = MS.patch_grid(r)
        zc = min(zs, key=lambda z: abs(z - TD.Z_C))
        i0, j0 = ys.index(0.0), zs.index(zc)
        for name, gap in gaps.items():
            an = MS.analyse(r, "board", gap)
            ch = MS.chips(r, "board", gap)
            Bc = MS.B0(r, ch) * 1e3
            bmax = float(np.abs(Bc).max())
            g, rng = SS.gain_for(bmax)
            both = [q["sigF_n_unk_bg"] for q in an["points"]]
            Sg = {k: np.array([[MS.signal(r, Gp[(y, z)][k], ch) for z in zs] for y in ys]) for k in ("normal", "shear_y", "shear_z")}
            J = np.concatenate([np.stack([Sg["normal"][i0, j0], Sg["shear_y"][i0, j0], Sg["shear_z"][i0, j0],
                                          MS.deriv(Sg["normal"][:, j0], ys, i0), MS.deriv(Sg["normal"][i0, :], zs, j0)], 1),
                                np.tile(np.eye(3), (len(ch), 1))], 1)
            sig = np.tile(MS.NOISE, len(ch))
            feq = {}
            for k, ax in enumerate("xyz"):
                e = np.zeros(3)
                e[k] = 1e-6
                dB = (MS.B0(r, [q + e for q in ch]) - MS.B0(r, [q - e for q in ch])).ravel() / 2 * 1e6
                est = np.linalg.lstsq(J / sig[:, None], dB / sig, rcond=None)[0]
                feq[ax] = dict(normal=round(abs(float(est[0])), 4), shear=round(float(np.hypot(est[1], est[2])), 4))
            out.append(dict(layout=label, fe_layout=r0["layout"], recipe=r["recipe"], magnets=len(r["magnets"]), case=name,
                            gap_mm=round(gap, 3), centre_force_N=round(an["centre"]["sigF_n_unk_bg"], 4),
                            median_force_N=round(float(np.median(both)), 4),
                            range_force_N=[round(min(both), 4), round(max(both), 4)],
                            B_chips_mT=np.round(Bc, 3).tolist(), B_U1_mT=round(float(np.linalg.norm(Bc[0])), 3),
                            B_max_component_mT=round(bmax, 3), gain_sel=g, headroom=None if rng is None else round(rng / bmax, 2),
                            false_force_N_per_um=feq, magnets_z_from_mount_mm=[m[3] for m in r["magnets"]]))
            print(f"{label:10s} {r['recipe']} {name:10s} gap {gap:.3f}: centre {an['centre']['sigF_n_unk_bg']:.4f} N, "
                  f"B max {bmax:.2f} mT (GAIN_SEL {g})", flush=True)
    static = {}
    chip_uv = [(0, 0), (0, 7), (7, 0), (0, -7), (-7, 0.051)]
    for key, t in sp["tips"].items():
        mg = t["magnets"]
        Dm, tm_ = mg["size_mm"]
        Vm = math.pi * (Dm / 2e3) ** 2 * (tm_ / 1e3)
        mom = np.array([MS.BR * Vm / (4e-7 * math.pi), 0.0, 0.0])
        mags_t = [np.array([-m["centre_F"][0] - 2.05, m["centre_F"][1], A.Z_MOUNT - m["centre_F"][2]]) * 1e-3
                  for m in mg["items"]]
        rowk = {}
        for name, gap in (("design", L["gap"]), ("plate_now", 3.42)):
            chips_t = [np.array([TD.X_BACK - gap, u, TD.Z_C + v]) * 1e-3 for u, v in chip_uv]
            B = np.array([sum(MS.dipole(c, pm, mom) for pm in mags_t) for c in chips_t]) * 1e3
            chipsF = [np.array([A.X_BACK + gap, u, A.Z_C - v]) for u, v in chip_uv]
            dc = [round(float(min(np.linalg.norm(c - np.array(m["centre_F"])) for m in mg["items"])), 2) for c in chipsF]
            ds = [round(float(min(disc_distance(c[None, :], m["centre_F"], m["x_F"][0], m["x_F"][1], Dm / 2)[0]
                                  for m in mg["items"])), 2) for c in chipsF]
            bmax = float(np.abs(B).max())
            g, rng = SS.gain_for(bmax)
            rowk[name] = dict(gap_mm=round(gap, 3), B_chips_mT=np.round(B, 3).tolist(), B_U1_mT=round(float(np.linalg.norm(B[0])), 3),
                              B_max_component_mT=round(bmax, 3), gain_sel=g,
                              chip_to_magnet_centre_mm=dc, chip_to_magnet_surface_mm=ds)
        rowk["magnets_top_depth_mm"] = mg["top_depth_mm"]
        rowk["magnets_F"] = [m["centre_F"] for m in mg["items"]]
        static[key] = rowk
        print(f"static {key:7s}: U1 {rowk['design']['B_U1_mT']:.2f} mT (GAIN_SEL {rowk['design']['gain_sel']}), chip to magnet "
              f"{min(rowk['design']['chip_to_magnet_surface_mm']):.2f} mm (surface)", flush=True)
    (FINAL / f"{PFX}_sense.json").write_text(json.dumps(dict(
        rows=out, static_fields=static, gaps_mm={k: round(v, 3) for k, v in gaps.items()},
        model="scripts/tpu_tip_magnet_sensing.py analyse (as concept a): point dipoles N52 1.45 T, magnet motions of the "
              "readiness finite elements of the 85A block (docs/experiments/20261016-sim2real_fingertip_readiness/fe/fem.jsonl), "
              "five MLX90393 at GAIN_SEL 7 noise, one 141 Hz sample, Cramer-Rao bound with the contact position and a uniform "
              "background field unknown",
        as_printed="tips of the 2.9 mm print set: four-magnet block as before (rows 7.36 and 14.36 mm from the mount face); "
                   f"five-magnet block five_z7p4 (corner rows at {min(z5):.2f} and {max(z5):.2f} mm, their motions per newton from "
                   "the five_d3x2_d5 finite elements); static fields of all five tips at their sheets' magnets (the 11/22 "
                   "half-ellipsoid's 7.5 mm under the apex)",
        when=time.strftime("%Y-%m-%d %H:%M")), indent=1, default=float))


def cmd_render(a):
    A.cmd_render(a)


# ----------------------------------------------------------------------------------------------- drawing
def _dim(ax, p0, p1, text, off, horizontal=True, fs=13, side=1):
    """A linear dimension between p0 and p1 (data units), drawn off by `off` across the measured direction."""
    (x0, y0), (x1, y1) = p0, p1
    if horizontal:
        y = max(y0, y1) + off if side > 0 else min(y0, y1) - off
        ax.annotate("", xy=(x0, y), xytext=(x1, y), arrowprops=dict(arrowstyle="<|-|>", lw=0.9, color="k", shrinkA=0, shrinkB=0,
                                                                     mutation_scale=9))
        for x, yy in ((x0, y0), (x1, y1)):
            ax.plot([x, x], [yy, y], color="k", lw=0.5)
        ax.text((x0 + x1) / 2, y + (0.12 if side > 0 else -0.12), text, ha="center", va="bottom" if side > 0 else "top",
                fontsize=fs)
    else:
        x = max(x0, x1) + off if side > 0 else min(x0, x1) - off
        ax.annotate("", xy=(x, y0), xytext=(x, y1), arrowprops=dict(arrowstyle="<|-|>", lw=0.9, color="k", shrinkA=0, shrinkB=0,
                                                                     mutation_scale=9))
        for xx, y in ((x0, y0), (x1, y1)):
            ax.plot([xx, x], [y, y], color="k", lw=0.5)
        ax.text(x + (0.12 if side > 0 else -0.12), (y0 + y1) / 2, text, ha="left" if side > 0 else "right", va="center",
                fontsize=fs)


def fits():
    """The critical fits and tolerances of the final design, from concept a's parameters."""
    P, L = A.P, A.levels()
    return [
        ("Window plate", f"{L['t_w']:.2f} mm = chip height - {P['chip_proud']:.2f}",
         "Print at the measured MLX90393 package height less 0.05 mm, so the chips stand 0.05 mm proud into the TPU"),
        ("Chip windows", f"3.00 + 2 x {P['clear_chip']:.2f} = {3 + 2 * P['clear_chip']:.2f} mm square",
         f"C1-C5, R1-R2 windows {P['clear_passive']:.2f} mm around each part"),
        ("Locating pins", f"{P['pin_d']:.2f} mm in the board's {P['pin_hole_d']:.2f} mm plated holes",
         f"0.025 mm radial; {P['pin_into_plate']:.2f} mm into reamed {P['pin_hole_d']:.2f} mm holes of the window plate; +y pin "
         f"relieved to +-{P['diamond_half']:.2f} mm in y (locates z only); pin chosen on the coupon"),
        ("M1.6 x 4 countersunk", f"head {P['cs_dk']:.1f} mm, 90 deg seat in the window plate",
         f"clearance {P['cs_d'] + 0.1:.1f} mm in the clamp plate; brass nut {P['nut_s']:.1f} mm A/F pressed into a {P['nut_m']:.1f} mm pocket"),
        ("M2 x 5 cheese head", f"pilot {P['m2_pilot']:.1f} mm, {P['m2_depth']:.1f} mm deep",
         f"thread-forming into the shoulders; ears {P['ear_gap']:.2f} mm short of the shoulders so the preload goes into the board"),
        ("Dovetail keys", f"root {P['key_root']:.1f}, head {P['key_head']:.1f}, depth {P['key_depth']:.1f}, length {P['key_len']:.1f} mm",
         f"at y +-{P['key_yc']:.1f} mm; tip channels {P['fit_flank']:.2f} mm narrower per flank, {P['fit_floor']:.2f} mm deeper, "
         f"{P['fit_len']:.2f} mm longer"),
        ("Detents", f"{P['detent_h']:.2f} mm high at z {P['detent_z']:.2f}", f"in a {2 * P['groove_r']:.2f} mm groove of each channel"),
        ("Rim and board", f"|y| <= {P['half_w']:.1f} mm (21.0 mm), board |y| <= 10.0", "the FR-4 edge 0.5 mm inside the rim (R5)"),
        ("Tab", f"cut {P['tab_len']:.1f} mm past the main square (v = -15.0 mm)", "J1 removed; four wires on its signal pads"),
        ("PLA tolerance", "holes print 0.1-0.2 mm undersize", "ream pin holes to 2.40 mm and pilots to 1.70 mm; pins from the coupon"),
    ]


def cmd_drawing(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    from matplotlib.patches import Polygon as MPoly
    P, L = A.P, A.levels()
    parts = [("Tip, 85A block", fname("tip_block4_F"), "#e8c88a"), ("DIP link", fname("dip_link"), "#d6d6cc"),
             ("Board", fname("board"), "#1f7a3a"), ("Clamp plate", fname("clamp_plate"), "#5b8fd9"),
             ("Brass screws, nuts", fname("fasteners"), "#c8a33c")]
    meshes = [(n, trimesh.load(p), c) for n, p, c in parts]
    it = A.board_items()
    hd = sorted([(y, z) for y, z, _ in it["holes"] if z < A.Z_C])
    hp = sorted([(y, z) for y, z, _ in it["holes"] if z > A.Z_C])
    plt.rcParams.update({"font.family": "serif", "font.size": 13})
    fig = plt.figure(figsize=(13.0, 15.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.95], hspace=0.32, wspace=0.18)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    axT = fig.add_subplot(gs[2, :])

    def draw(ax, orig, nrm, h_ax, v_ax):
        for n, tm, c in meshes:
            s = tm.section(plane_origin=orig, plane_normal=nrm)
            if s is None:
                continue
            for poly in s.discrete:
                ax.add_patch(MPoly(poly[:, [h_ax, v_ax]], closed=True, fc=c, ec="k", lw=0.5))
        ax.set_aspect("equal")
        ax.grid(True, lw=0.3, alpha=0.4)

    # A: section y = 0 at U1: the stack
    draw(axA, (0, 0.0, 0), (0, 1, 0), 0, 2)
    axA.set_xlim(4.0, 13.5)
    axA.set_ylim(A.Z_C - 3.2, A.Z_C + 3.4)
    axA.set_xlabel("x (mm), dorsal to the right")
    axA.set_ylabel("z (mm)")
    axA.set_title("A. Section y = 0 through U1: the stack", fontsize=14)
    zA = A.Z_C + 2.2
    _dim(axA, (A.X_BACK, zA), (L["x_cf"], zA), f"{L['t_w']:.2f}", 0.15)
    _dim(axA, (L["x_cf"], zA), (L["x_bb"], zA), f"{P['fr4']:.2f}", 0.15)
    _dim(axA, (L["x_bb"], zA), (L["x_c1"], zA), f"{P['clamp_t']:.2f}", 0.15)
    axA.axvline(X_S := A.X_BACK + L["gap"], color="k", lw=1.0, ls=":")
    axA.annotate(f"Sensing plane {L['gap']:.2f} mm\nbehind the back face", xy=(X_S, A.Z_C - 0.6), xytext=(8.6, A.Z_C - 2.6),
                 fontsize=13, arrowprops=dict(arrowstyle="->", lw=0.9))
    axA.annotate(f"Chip top {P['chip_proud']:.2f} mm proud\nof the window plate", xy=(L["chip_top"], A.Z_C + 0.9),
                 xytext=(4.2, A.Z_C + 3.0), fontsize=13, arrowprops=dict(arrowstyle="->", lw=0.9))
    axA.text(4.25, A.Z_C - 3.0, "Tip back face x 6.45", fontsize=13)

    # B: section z = -86.06 through the distal holes (M1.6 countersunk screws, nuts)
    draw(axB, (0, 0, hd[0][1]), (0, 0, 1), 1, 0)
    axB.set_xlim(-11.5, 11.5)
    axB.set_ylim(3.0, 13.5)
    axB.set_xlabel("y (mm)")
    axB.set_ylabel("x (mm), dorsal up")
    axB.set_title(f"B. Section z = {hd[0][1]:.2f}: M1.6 screws and nuts", fontsize=14)
    _dim(axB, (-P["half_w"], L["x_c1"]), (P["half_w"], L["x_c1"]), "21.0 (rim)", 0.7)
    _dim(axB, (-10.0, L["x_cf"]), (10.0, L["x_cf"]), "20.0 (board)", 3.4)
    _dim(axB, (-A.Y_HALF, A.X_BACK), (A.Y_HALF, A.X_BACK), "14.8 (tip)", 2.4, side=-1)
    axB.annotate(f"Nut {P['nut_s']:.1f} A/F,\n{P['nut_m']:.1f} deep", xy=(hd[1][0], L["x_c1"] - 0.6), xytext=(1.0, 12.4),
                 fontsize=13, arrowprops=dict(arrowstyle="->", lw=0.9))
    axB.annotate("90 deg countersink,\nhead 3.0", xy=(hd[0][0] + 0.9, A.X_BACK + 0.4), xytext=(-11.2, 4.6), fontsize=13,
                 arrowprops=dict(arrowstyle="->", lw=0.9))

    # C: section z through the proximal holes (pins)
    draw(axC, (0, 0, hp[0][1]), (0, 0, 1), 1, 0)
    axC.set_xlim(-11.5, 11.5)
    axC.set_ylim(3.0, 13.5)
    axC.set_xlabel("y (mm)")
    axC.set_ylabel("x (mm), dorsal up")
    axC.set_title(f"C. Section z = {hp[0][1]:.2f}: pins and dovetail keys", fontsize=14)
    axC.annotate(f"Pin {P['pin_d']:.2f} in the board's\n{P['pin_hole_d']:.2f} hole, {P['pin_into_plate']:.2f} into\nreamed {P['pin_hole_d']:.2f}",
                 xy=(hp[0][0], L["x_cf"] - 0.2), xytext=(-11.2, 10.6), fontsize=13, arrowprops=dict(arrowstyle="->", lw=0.9))
    axC.annotate(f"Diamond pin:\nflats +-{P['diamond_half']:.2f} in y", xy=(hp[1][0], L["x_cf"] + 0.3), xytext=(3.2, 11.6),
                 fontsize=13, arrowprops=dict(arrowstyle="->", lw=0.9))
    axC.annotate(f"Ear gap {P['ear_gap']:.2f}\nto the shoulder", xy=(8.5, L["x_ear0"] - P["ear_gap"] / 2), xytext=(4.0, 4.0),
                 fontsize=13, arrowprops=dict(arrowstyle="->", lw=0.9))

    # D: the dovetail key and channel at z -70 (key and channel outlines from concept a's polygon)
    axD.set_title("D. Dovetail key and tip channel, z -70.00", fontsize=14)
    kp = np.array(A.key_poly(P, P["key_yc"]))
    cp = np.array(A.key_poly(P, P["key_yc"], flank=P["fit_flank"], floor=P["fit_floor"], x_top=A.X_BACK + 0.3))
    axD.add_patch(MPoly(cp[:, [0, 1]][:, ::-1] if False else cp, closed=True, fc="#e8c88a", ec="k", lw=0.8, ls="--"))
    axD.add_patch(MPoly(kp, closed=True, fc="#d6d6cc", ec="k", lw=0.9))
    allp = np.vstack([kp, cp])
    axD.set_xlim(allp[:, 0].min() - 1.6, allp[:, 0].max() + 1.6)
    axD.set_ylim(allp[:, 1].min() - 1.2, allp[:, 1].max() + 1.2)
    axD.set_aspect("equal")
    axD.grid(True, lw=0.3, alpha=0.4)
    axD.set_xlabel("first axis of the key polygon (mm)")
    axD.set_ylabel("second axis (mm)")
    axD.text(0.02, 0.04, f"Key (grey): root {P['key_root']:.1f}, head {P['key_head']:.1f}, depth {P['key_depth']:.1f}\n"
                         f"Channel (dashed): {P['fit_flank']:.2f} narrower per flank,\n{P['fit_floor']:.2f} deeper; "
                         f"detent {P['detent_h']:.2f} in a {2 * P['groove_r']:.2f} groove",
             transform=axD.transAxes, fontsize=13, va="bottom")

    # table of fits
    axT.axis("off")
    rows = fits()
    tb = axT.table(cellText=[[a_, b_, c_] for a_, b_, c_ in rows], colLabels=["Feature", "Nominal", "Fit and tolerance"],
                   colWidths=[0.16, 0.30, 0.54], loc="upper center", cellLoc="left")
    tb.auto_set_font_size(False)
    tb.set_fontsize(12)
    tb.scale(1.0, 1.9)
    for (r_, c_), cell in tb.get_celld().items():
        cell.get_text().set_wrap(True)
        if r_ == 0:
            cell.set_text_props(weight="bold")
    axT.set_title("Critical fits and tolerances (mm)", fontsize=14)
    fig.suptitle("Final DIP link: window plate, board and dorsal clamp (concept a at the 2.9 mm-seat tips)", fontsize=16, y=0.995)
    dst = MEDIA / f"{PFX}_drawing.png"
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(dst.relative_to(REPO), flush=True)


# ----------------------------------------------------------------------------------------------- documents
def print_settings():
    sheets = {k: json.loads((READY / "print" / f"{v}.json").read_text()) for k, v in SHEETS.items()}
    tips = {}
    for k, sh in sheets.items():
        tips[k] = dict(file=str((PRINT / f"{PFX}_tip_{k.replace('.', 'p')}_print.stl").relative_to(REPO)),
                       sheet=f"{READY.relative_to(REPO)}/print/{SHEETS[k]}.json", orientation=sh["orientation"],
                       seat_mm=sh.get("seat_mm"), pauses=[dict(z_mm=p_.get("z_mm") or p_.get("pause_print_z"),
                                                               magnets=p_.get("magnets") or p_.get("n_magnets"))
                                                          for p_ in sh.get("pauses", [])])
    return dict(
        printer="Original Prusa XL 5T, 0.4 mm nozzles (PrusaSlicer profile 'Original Prusa XL - 5T Input Shaper 0.4 nozzle')",
        parts=dict(
            dip_link=dict(file=str(fname("dip_link").relative_to(REPO)), tool="tool 1", material="PLA (the links: PLA, 15 % infill)",
                          orientation="upright, the tip end up: the bridge's mount face is a top surface; the window plate, keys "
                                      "and shoulders rise from it as vertical walls",
                          layer_mm=0.10, perimeters=3, infill="15 % gyroid as the links; 100 % in the shoulders (modifier "
                          "box |y| 6.0-10.5 mm around the M2 pilots)", supports="only under the clevis arms and the bridge's face "
                          "toward the PIP axis (paint-on or support enforcers)", seam="aligned, on the dorsal side (away from the "
                          "window plate's palmar face)", nozzle_C=215, bed_C=60, after="ream the two pin holes to 2.40 mm and the "
                          "two M2 pilots to 1.70 mm; check the window plate's palmar face with a straightedge (0.05 mm)"),
            clamp_plate=dict(file=str(fname("clamp_plate").relative_to(REPO)), tool="tool 1", material="PLA or PETG, solid",
                             orientation="back face (x 10.9) on the bed: the board face is a top surface, the pins print "
                                         "vertical, the nut pockets open on the bed side", layer_mm=0.10, perimeters=4,
                             infill="100 %", supports="none", ironing="on the board face", nozzle_C=215, bed_C=60,
                             after="press the two M1.6 nuts in (warm with a soldering iron if tight)"),
            pin_fit_coupon=dict(file=str((PRINT / f"{PFX}_pin_fit_coupon.stl").relative_to(REPO)), tool="tool 1",
                                material="the clamp plate's material", orientation="plate on the bed, pins up", layer_mm=0.10,
                                perimeters=4, infill="100 %", supports="none",
                                use="print before the clamp plate: the pin (2.25, 2.30, 2.35 or 2.40 mm, the chamfered corner at "
                                    "2.25) that enters the board's holes without play sets pin_d"),
            tips=dict(tool="tool 2", material="Siraya Tech TPU 85A", profile="Siraya Tech TPU 85A @XL 0.4mm Nozzle: 240 C, bed "
                      "35 C, first layer 0.2 mm at 50 mm/s, fan off for 4 layers, 100 % from layer 6, extrusion multiplier 0.96, "
                      "retraction 0.6 mm (the print procedure page)", infill="gyroid 30 % (50 % for the stiff variants), two "
                      "perimeters, two around each pocket, four top and bottom layers", seat_mm=SEAT,
                      orientation="as the 2.9 mm sheets: the blocks on their back face (the key channels open to the bed and "
                                  "widen upward; their 1.9 mm floors bridge 3.5 mm), the curved tips upright on the mount face "
                                  "(channels vertical)",
                      pauses="the sheets' pauses (block 12.4 mm; curved tips 9.2 and 16.2 mm), magnets north pole toward the "
                             "palmar face", per_tip=tips,
                      slicing="re-slice each channelled STL with the sheet's settings (its PrusaSlicer project replaced by this "
                              "STL); check that no layer bridges a channel before its floor")),
        when=time.strftime("%Y-%m-%d %H:%M"))


def parts_list():
    return [
        dict(item="DIP link", material="PLA", qty=1, file=str(fname("dip_link").relative_to(REPO)),
             step=str((CAD / f"{PFX}_dip_link.step.zip").relative_to(REPO)) + " (fused, arms as the STL's facets); "
                  + str(fname("dip_link_new_features", "step").relative_to(REPO)) + " (new features, exact)"),
        dict(item="Clamp plate", material="PLA or PETG, solid", qty=1, file=str(fname("clamp_plate").relative_to(REPO)),
             step=str(fname("clamp_plate", "step").relative_to(REPO))),
        dict(item="Pin-fit coupon", material="as the clamp plate", qty=1,
             file=str((PRINT / f"{PFX}_pin_fit_coupon.stl").relative_to(REPO))),
        dict(item="85A tip with two key channels, 2.9 mm seats (one per tip family)", material="Siraya Tech TPU 85A", qty=1,
             file=str((PRINT / f"{PFX}_tip_block4_print.stl").relative_to(REPO)) + " (block4; block5, cap8p5, ell16, ell11 "
                  "beside it)", step=str(fname("tip_cutter", "step").relative_to(REPO)) + " (the channels, to subtract in CAD)"),
        dict(item="N52 disc magnet 3 x 2 mm, axially magnetized", qty="4 (5 for the five-magnet block)"),
        dict(item="ReSkin 5X board, tab cut to 4.9 mm, J1 removed, four wires", material="FR-4 0.6 mm", qty=1),
        dict(item="M1.6 x 4 countersunk screw, DIN 963 / ISO 2009", material="brass", qty=2),
        dict(item="M1.6 hex nut, ISO 4032", material="brass", qty=2),
        dict(item="M2 x 5 cheese-head screw, DIN 84", material="brass", qty=2),
        dict(item="Wire, AWG 32-36, four conductors to the inline connector on the MCP link side", qty=1),
        dict(item="Hot glue or Kapton tape for the wires' strain relief", qty=1),
    ]


def assembly():
    seq = [
        "Measure the five chips' package height h and the FR-4 thickness on the owner's board; set chip_h = h and fr4 in "
        "scripts/dip_link_concept_a.py P (the window plate prints at h - 0.05 mm) and rebuild: scripts/dip_link_board_mount.py "
        "build, check.",
        "Print the pin-fit coupon; set pin_d to the pin that enters the board's proximal holes without play; rebuild.",
        "Print the DIP link and the clamp plate (print_settings.json); ream the window plate's two pin holes to 2.40 mm and "
        "the shoulders' pilots to 1.70 mm.",
        "Desolder J1, cut the tab 4.9 mm from the main square (v = -15.0 mm), solder four wires (VDD, GND, SDA, SCL; AWG "
        "32-36) to the signal pads, leading to the +y side.",
        "Lay the board component side down on the window plate: chips in their windows, R1-R2 and the solder joints in the "
        "bridge's pocket, the wires in the channel and up through the gap beside the tab.",
        "Put on the clamp plate (pins through the board into the window plate) and drop two M1.6 nuts into its back pockets.",
        "With the tip off, drive two M1.6 x 4 countersunk screws from the palmar side through the window plate and the board's "
        "distal holes into the nuts; snug, about 0.05 N m.",
        "Drive two M2 x 5 screws through the clamp plate's ears into the shoulders; snug, the ears still clear of the shoulders.",
        "Fix the wires in the channel with a dab of hot glue or Kapton tape so a cable pull loads the DIP link; route the cable "
        "along the clamp plate's proximal edge to y > 17.4 mm and across the PIP joint outside the slotted arm.",
        "Print the tip with its channels (2.9 mm seats, magnets north pole toward the palmar face at the pauses).",
        "Slide the tip along +z onto the keys until its mount face meets the bridge and both detents engage; read the zero-load "
        "field. Tip exchange: pull along -z; the board's screws stay in.",
    ]
    first_print = [
        "Board pocket: the board lies flat on the window plate with every chip in its window and its top 0.05 mm proud "
        "(feeler gauge or a straightedge with a 0.05 mm shim); no part touches a window's edge.",
        "Pins: the clamp plate's pins enter the board's proximal holes and the window plate without play and without force.",
        "Clamp preload: with the four screws snug the board does not move under a fingernail push at its edges; the ears stay "
        "clear of the shoulders (0.3 mm gap visible).",
        "Tip fit: the tip slides onto the keys with a firm push, both detents click, the mount face meets the bridge, and the "
        "back face lies on the window plate without a visible gap; the tip comes off by hand along -z.",
        "Zero-load field: the five chips read a steady field with the tip mounted; re-read after each remount (repeatable to "
        "the board's noise) and after 50 presses at 3 N (the specification's retention check).",
    ]
    return dict(sequence=seq, first_print_checks=first_print)


def cmd_docs(a):
    (FINAL / f"{PFX}_print_settings.json").write_text(json.dumps(print_settings(), indent=1))
    (FINAL / f"{PFX}_parts_list.json").write_text(json.dumps(parts_list(), indent=1))
    (FINAL / f"{PFX}_assembly.json").write_text(json.dumps(assembly(), indent=1))
    print("docs ->", FINAL.relative_to(REPO), flush=True)


# ----------------------------------------------------------------------------------------------- evaluation
def cmd_eval(a):
    P, L = A.P, A.levels()
    chk = json.loads((FINAL / f"{PFX}_check.json").read_text())
    sen = json.loads((FINAL / f"{PFX}_sense.json").read_text())
    parts = json.loads((FINAL / f"{PFX}_parts.json").read_text())
    stp = json.loads((FINAL / f"{PFX}_step.json").read_text()) if (FINAL / f"{PFX}_step.json").exists() else {}

    def fr(layout, recipe, case):
        for r in sen["rows"]:
            if r["layout"] == layout and r["recipe"] == recipe and r["case"] == case:
                return r
    four = fr("four_d3x2_d5", "w85_30", "design")
    five = fr("five_z7p4", "w85_30", "design")
    clear = chk["clearance"]
    rigid = {k: v for k, v in clear.items() if k in ("dip_link_new", "clamp_plate", "board", "fasteners")}
    worst = min(rigid.items(), key=lambda kv: kv[1]["min_mm"])
    ev = dict(
        title="Final DIP link for the ReSkin 5X board: concept a's window plate and dorsal clamp at the 2.9 mm-seat tips",
        date="2026-10-18", status="model; nothing printed; chip height, FR-4 thickness, pin fit and clamp retention pending",
        generator="scripts/dip_link_board_mount.py (on scripts/dip_link_concept_a.py)", decision="../decision.json",
        stack_x_F=dict(tip_back_face=A.X_BACK, chip_tops=round(L["chip_top"], 3), window_plate=[A.X_BACK, round(L["x_cf"], 3)],
                       board_fr4=[round(L["x_cf"], 3), round(L["x_bb"], 3)], clamp_plate=[round(L["x_bb"], 3), round(L["x_c1"], 3)],
                       sensing_plane=round(A.X_BACK + L["gap"], 3)),
        gap_mm=round(L["gap"], 3), gap_current_plate_mm=3.42,
        chip_to_magnet_mm={k: dict(surface_min=min(v["design"]["chip_to_magnet_surface_mm"]),
                                   centre_min=min(v["design"]["chip_to_magnet_centre_mm"]),
                                   U1_surface=v["design"]["chip_to_magnet_surface_mm"][0])
                           for k, v in sen["static_fields"].items()},
        force_resolution_N=dict(
            four_30=four["centre_force_N"], five_z7p4_30=five["centre_force_N"],
            four_50=fr("four_d3x2_d5", "w85_50", "design")["centre_force_N"],
            five_z7p4_50=fr("five_z7p4", "w85_50", "design")["centre_force_N"],
            chips_0p80_to_1p00_four_30=[fr("four_d3x2_d5", "w85_30", "chips_1.00")["centre_force_N"],
                                        fr("four_d3x2_d5", "w85_30", "chips_0.80")["centre_force_N"]],
            current_plate_four_30=fr("four_d3x2_d5", "w85_30", "plate_now")["centre_force_N"],
            current_plate_five_30=fr("five_z7p4", "w85_30", "plate_now")["centre_force_N"],
            false_force_four_30_N_per_um=four["false_force_N_per_um"]),
        static_fields={k: dict(B_U1_mT=v["design"]["B_U1_mT"], B_max_component_mT=v["design"]["B_max_component_mT"],
                               gain_sel=v["design"]["gain_sel"], magnets_top_depth_mm=v["magnets_top_depth_mm"])
                       for k, v in sen["static_fields"].items()},
        clearance=dict(per_part={k: dict(min_mm=v.get("min_mm"), pip_deg=v.get("pip_deg"), part=v.get("part"))
                                 for k, v in clear.items()},
                       worst_rigid=dict(part=worst[0], **{k: worst[1][k] for k in ("min_mm", "pip_deg", "part", "at_F")}),
                       requirement="0.5 mm rigid, 1.0 mm cable (R7); exact method of the specification, PIP -18..92 deg"),
        to_pip_servo=chk["to_pip_servo"], channels_to_magnets_R4=chk["channels_to_magnets_R4"],
        walls=chk["walls"], tip_walls_near_channels=chk["tip_walls_near_channels"], mass_g=chk["mass_g"],
        width_over_tip_length_y=chk["width_over_tip_length_y"], overlap_mm3=chk["overlap_mm3"],
        arms_symmetric_difference_mm3=chk["arms_symmetric_difference_mm3"],
        parts=parts["parts"], tips=parts["tips"], pin_fit_coupon=parts.get("pin_fit_coupon"), step=stp,
        fits=[dict(feature=f_, nominal=n_, fit=t_) for f_, n_, t_ in fits()],
        files=sorted(str(p.relative_to(REPO)) for p in FINAL.rglob("*") if p.is_file()),
        when=time.strftime("%Y-%m-%d %H:%M"))
    (FINAL / "evaluation.json").write_text(json.dumps(ev, indent=1, default=float))
    print("eval ->", (FINAL / "evaluation.json").relative_to(REPO), flush=True)


def main():
    if os.environ.get("DLF_CMD") == "step":
        freecad_step()
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build", "check", "sense", "render", "drawing", "docs", "eval"):
        sp_ = sub.add_parser(c)
        if c == "check":
            sp_.add_argument("--reuse", action="store_true", help="reuse clearance rows whose part STL is unchanged")
    a = ap.parse_args()
    {"build": cmd_build, "check": cmd_check, "sense": cmd_sense, "render": cmd_render, "drawing": cmd_drawing,
     "docs": cmd_docs, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__" or os.environ.get("DLF_CMD") == "step":
    main()
