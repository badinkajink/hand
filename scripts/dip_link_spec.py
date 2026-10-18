#!/usr/bin/env python3
"""Specification of the DIP link that clamps the ReSkin 5X board against the printed fingertip (2026-10-18).

Owner, 2026-10-18: "we are going to redesign the DIP linkage entirely so that the board can be as close to the printed tip
as possible, while trying to keep the contact geometry the same. although that might change as well ... it's important
that the board does not wiggle at all--youll want something to fasten/sandwich it in place".

The builder reads the SR2 finger's CAD parts and the keep-out they sweep over the PIP range (scripts/dip_link_geometry.py),
the printed 85A tips, the finite-element rows, the readout model, the board sweep and the HOM cells of
docs/experiments/20261016-sim2real_fingertip_readiness/, the driver's calibrated joint ranges and the ReSkin 5X
fabrication files; it computes the clearance of each board placement over the PIP range and the force resolution at each
board distance, and writes into docs/experiments/20261018-dip_link_board_mount/

  spec.json                                every constraint with its number and its source
  spec.md                                  the requirements in short form, generated from spec.json
  media/20261018-dip_link_constraints.png  the keep-out over the PIP range beside the tip, and the board drawing

    ~/.claude/bin/resguard.sh run --mem 2G --cpu 200 -- taskset -c 12-15 env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/envs/femtip/bin/python scripts/dip_link_spec.py [--no-clearance]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import dip_link_geometry as G  # noqa: E402

READY = REPO / "docs/experiments/20261016-sim2real_fingertip_readiness"
OUT = G.STUDY
MEDIA = OUT / "media"
FIG = MEDIA / "20261018-dip_link_keepout_side.png"
FIG_BOARD = MEDIA / "20261018-reskin5x_board_and_tip.png"
SPEC = OUT / "spec.json"
SPEC_MD = OUT / "spec.md"
CLEAR_CACHE = G.LOGS / "clearance_boxes.json"
SERVOS_PY = REPO / "src/morphohand/driver/manta/host/manta_hand/servos.py"
BUILDER = REPO / "scripts/build_cad_v1_finger.py"

TIPS = {  # printed 85A tips of the readiness study (sheets at 30 % gyroid; the 50 % sheets share the geometry)
    "block4": "20261016-siraya85a_block_four_d3x2_depth5_seat2p7_g30",
    "block5": "20261016-siraya85a_block_five_d3x2_depth5_seat2p7_g30",
    "cap8.5": "20261016-siraya85a_tcd_cap8p5_four_d3x2_depth5_seat2p7_g30",
    "ell16": "20261016-siraya85a_tcd_ell7p4x8p5x16x32_four_d3x2_depth5p5_seat2p7_g30",
    "ell11": "20261016-siraya85a_tcd_ell7p4x8p5x11x22_four_d3x2_depth7_seat2p7_g30",
}
MASS_KEY = {"block4": "block4_g30", "block5": "block5_g30", "cap8.5": "cap_g30", "ell16": "ell16_g30", "ell11": "ell11_g30"}

CHIP_H = 0.9          # nominal MLX90393 package height (0.80-1.00 mm)
FR4 = 0.6
PLACEMENTS = {        # x_F of the FR-4's component face, and whether the components point toward the tip (-x)
    "plate": dict(comp_face=9.45 + CHIP_H, toward_tip=True, gap=3.42,
                  text="the readiness scenes: the board on the dorsal face of the DIP link's 3.0 mm plate, chips toward the "
                       "plate"),
    "touch": dict(comp_face=G.TIP["x_back"] + CHIP_H, toward_tip=True, gap=0.42,
                  text="chip tops against the tip's back face, the FR-4 behind them"),
    "recessed": dict(comp_face=G.TIP["x_back"], toward_tip=True, gap=0.42 - CHIP_H,
                     text="the FR-4's component face on the tip's back face, the chips and capacitors in 1.0 mm pockets of "
                          "the TPU"),
    "flipped": dict(comp_face=G.TIP["x_back"] + FR4, toward_tip=False, gap=FR4 + CHIP_H - 0.42,
                    text="the FR-4's bare face on the tip's back face, the chips and J1 facing away from the tip"),
}
BOARD_V = (-10.10, 10.05)          # the board's main square along v (outline)
TAB_V0 = -10.10
J1_V = (-18.094, -13.141)
J1_U = (-2.949, 3.046)
J1_H = 2.9


def v_to_z(v):
    return -np.asarray(v, float) + G.Z_CONTACT


def placement_x(p, height):
    """x_F interval of a layer `height` mm on the component side of placement p, and of the FR-4."""
    P = PLACEMENTS[p]
    c = P["comp_face"]
    if P["toward_tip"]:
        return (c - height, c), (c, c + FR4)
    return (c, c + height), (c - FR4, c)


def r3(v):
    return None if v is None else round(float(v), 3)


# ------------------------------------------------------------------------------------------------- the current parts

def current_dip_link():
    from morphohand.geometry import cad_features as cf
    m = G.load_part("dip_link")
    lm = G.line_material
    holes = []
    for c in cf.cylinders(m):
        if c.kind == "hole" and c.radius * 2 <= 6.5 and c.cover_deg >= 180:
            ax = np.round(np.abs(c.axis), 3)
            holes.append(dict(d_mm=round(2 * c.radius, 2), axis="xyz"[int(np.argmax(ax))], centre_F=np.round(c.point, 2).tolist(),
                              length_mm=round(c.length, 2)))
    holes.sort(key=lambda h: (h["d_mm"], h["centre_F"][2]))
    dens = G.assembly()["link_density_kg_m3"]["dip_link"]
    return dict(
        file="assets/cad_v1/fingertip_mount_DIP_link.stl", bounds_F=np.round(m.bounds, 3).tolist(),
        volume_mm3=round(float(m.volume), 1), mass_g=round(float(m.volume) * dens * 1e-6, 2),
        mass_src=f"CAD volume at its sliced density {dens} kg/m^3 (PLA, 15 % infill; assembly.json)",
        back_plate=dict(x=lm(m, "x", {"y": 0, "z": -78.16}), y=lm(m, "y", {"x": 8.0, "z": -78.16}),
                        z=lm(m, "z", {"x": 8.0, "y": 0}),
                        text="3.0 mm plate behind the tip's back face over its whole length; a 3.3 mm hole at z -82.21 for "
                             "a screw into the CAD insert"),
        distal_lip=dict(x=lm(m, "x", {"y": 0, "z": -86.7}), z=lm(m, "z", {"x": 5.0, "y": 0})[:1],
                        text="a 2.0 mm ridge reaching 2.4 mm palmar of the back face at the tip's end; it engages a groove "
                             "of the CAD insert"),
        tongue=dict(x=lm(m, "x", {"y": 0, "z": -70.0}), y=lm(m, "y", {"x": -6.0, "z": -70.0}), z=lm(m, "z", {"x": -6.0, "y": 0}),
                    text="a 2.6 x 9.8 mm tongue entering the insert's mount face 4.8 mm, 3.2-5.8 mm from the palmar face"),
        bridge=dict(z_at_x0=lm(m, "z", {"x": 0.0, "y": 0}), z_at_x6=lm(m, "z", {"x": 6.0, "y": 0}),
                    y_at_x6_z64=lm(m, "y", {"x": 6.0, "z": -64.0}),
                    text="the cross-bar joining the clevis arms below the MCP link; its inner face is a cylinder of radius "
                         "12.75 mm about the PIP axis, 0.05 mm outside the MCP link's end (radius 12.70 mm)"),
        clevis=dict(horn_side_y=lm(m, "y", {"x": 0.0, "z": -55.0}), slotted_y=lm(m, "y", {"x": 0.0, "z": -60.0})[-1:],
                    horn_arm_low_z=lm(m, "z", {"x": 0.0, "y": -20.15})[:1],
                    text="horn-side arm y -23.25..-17.05 (outer layer 2.0 mm, a 2.0 mm slot for the 1.6 mm horn, inner "
                         "layer 2.2 mm with the 6.4 mm hub recess); slotted arm y 13.40..17.40 with the 5.2 mm slotted bore"),
        holes=holes)


PART_ROLE = {
    "carriage": ("carriage.stl", "gantry carriage with the yaw servo and the yaw bearing"),
    "yaw_link": ("mcp_mount_YAW-link.stl", "yaw link with the MCP servo and the MCP bearing"),
    "mcp_link": ("pip_mount_MCP_link.stl", "MCP link (30.80 mm, MCP to PIP) with the PIP servo and the PIP bearing (MR105)"),
    "servo_pip": ("feetech scs0009.stl", "Feetech SCS0009 servo at the PIP joint, its output on the PIP axis"),
    "dip_link": ("fingertip_mount_DIP_link.stl", "DIP link: the clevis on the PIP joint and the tip mount; the part redesigned"),
    "tip": ("fingertip.stl", "the CAD TPU insert, 17.0 x 14.8 x 21.41 mm in place (its STL box is 22.0 mm long)"),
}


def parts_table():
    out = {}
    for k, (fn, role) in PART_ROLE.items():
        m = G.load_part(k)
        out[k] = dict(file=f"assets/cad_v1/{fn}", role=role, bounds_F=np.round(m.bounds, 2).tolist(),
                      volume_mm3=round(float(m.volume), 1))
    out["fused"] = dict(file="assets/cad_v1/fingertip_and_DIP_link.stl",
                        role="the DIP link with the insert in place; the builder registers the tip's pose on the DIP link from it")
    out["diagram"] = dict(file="assets/cad_v1/diagram.png", role="the owner's drawing of the yaw and MCP links")
    return out


def cad_insert():
    m = G.load_part("tip")
    lm = G.line_material
    return dict(
        file="assets/cad_v1/fingertip.stl (the CAD insert; the printed 85A tips are whole blocks without these features)",
        bounds_F=np.round(m.bounds, 3).tolist(), volume_mm3=round(float(m.volume), 1),
        tongue_pocket_z=lm(m, "z", {"x": -6.0, "y": 0}), groove_and_screw_z_at_x5=lm(m, "z", {"x": 5.0, "y": 0}),
        screw_hole=dict(d_mm=2.75, axis="x", centre_F=[3.95, 0.0, -82.21], depth_mm=5.0,
                        text="an M3 screw through the plate's 3.3 mm hole into the TPU"),
        back_proximal_round=dict(radius_mm=5.0, axis_F=[1.45, None, -72.3]))


def tip_outlines(stl, orient):
    """Back-face and mount-face outlines of a printed tip, in F: bounds and area."""
    import trimesh
    m = trimesh.load(REPO / stl, force="mesh")
    out = {}
    if orient.startswith("back face on the bed"):        # x_p = 21.41 - z_t, z_p = x_t + 8.5
        planes = {"back": ([0, 0, 0.05], [0, 0, 1]), "mount": ([21.36, 0, 0], [1, 0, 0])}
        to_F = lambda P: np.column_stack([-(P[:, 2] - 8.5) - 2.05, P[:, 1], -67.30 - (21.41 - P[:, 0])])  # noqa: E731
    else:                                                 # upright: x_p = x_t, z_p = z_t
        planes = {"back": ([-8.45, 0, 0], [1, 0, 0]), "mount": ([0, 0, 0.05], [0, 0, 1])}
        to_F = lambda P: np.column_stack([-P[:, 0] - 2.05, P[:, 1], -67.30 - P[:, 2]])  # noqa: E731
    for k, (o, n) in planes.items():
        s = m.section(plane_origin=o, plane_normal=n)
        p2, _ = s.to_2D()
        V = to_F(np.asarray(s.vertices))
        out[k] = dict(area_mm2=round(float(p2.area), 1), bounds_F=np.round([V.min(0), V.max(0)], 2).tolist())
    return out


def tips():
    proc = json.loads((READY / "bench/procedure.json").read_text())
    masses = [s for s in proc["steps"] if s["id"] == "mass"][0]["predicts"]
    out = {}
    for k, name in TIPS.items():
        sh = json.loads((READY / "print" / f"{name}.json").read_text())
        mg = sh["magnet"]
        mags = []
        for p in sh["pockets"]:
            top = 8.5 - p["magnet_top_depth_mm"]
            c = G.tip_design_to_F([top - mg["thickness_mm"] / 2, p["y_mm"], p["z_mm"]])
            mags.append(dict(centre_F=np.round(c, 2).tolist(), x_F=[round(-top - 2.05, 2), round(-(top - mg["thickness_mm"]) - 2.05, 2)],
                             tpu_wall_min_mm=p["wall_min_mm"], tpu_over_magnet_mm=p["tpu_over_magnet_mm"]))
        back_gap = G.TIP["x_back"] - max(m["x_F"][1] for m in mags)
        out[k] = dict(sheet=f"{READY.relative_to(REPO)}/print/{name}.json", stl=sh["stl"], label=sh["tip_label"],
                      shape=sh["tip"], print_orientation=sh["orientation"],
                      magnets=dict(count=mg["count"], size_mm=[mg["diameter_mm"], mg["thickness_mm"]],
                                   top_depth_mm=mg["top_depth_mm"], polarity=mg["orientation"], items=mags,
                                   tpu_behind_magnets_mm=round(back_gap, 2)),
                      chip_field_touch=sh["chip_field"], outlines=tip_outlines(sh["stl"], sh["orientation"]),
                      mass_g=masses.get(MASS_KEY[k]))
    return out


# -------------------------------------------------------------------------------------------------------- sensing

def sensing():
    """Force resolution at the contact point for each board placement (the readiness model at the placement's gap), and
    the force a one-position estimator reads per micrometre of board displacement relative to the magnets."""
    import sim2real_fingertip_sensing as SS
    import tpu_tip_design as TD
    import tpu_tip_magnet_sensing as MS
    rows = [json.loads(l) for l in open(READY / "fe/fem.jsonl")]
    out = []
    for r in rows:
        if r["layout"].endswith("_w1"):
            continue
        ys, zs, Gp = MS.patch_grid(r)
        zc = min(zs, key=lambda z: abs(z - TD.Z_C))
        travel = [float(np.linalg.norm(t)) for t, _ in r["per_force"]["1"]["magnets"]]
        for name, P in PLACEMENTS.items():
            gap = P["gap"]
            a = MS.analyse(r, "board", gap)
            ch = MS.chips(r, "board", gap)
            B = MS.B0(r, ch) * 1e3
            bmax = float(np.abs(B).max())
            g, rng = SS.gain_for(bmax)
            both = [p["sigF_n_unk_bg"] for p in a["points"]]
            # the readout's estimator at the contact point: normal and two shear forces, contact position and a uniform
            # background field unknown (tpu_tip_magnet_sensing.analyse); a rigid board shift enters as a field change dB
            i0, j0 = ys.index(0.0), zs.index(zc)
            Sg = {k: np.array([[MS.signal(r, Gp[(y, z)][k], ch) for z in zs] for y in ys]) for k in ("normal", "shear_y", "shear_z")}
            J = np.concatenate([np.stack([Sg["normal"][i0, j0], Sg["shear_y"][i0, j0], Sg["shear_z"][i0, j0],
                                          MS.deriv(Sg["normal"][:, j0], ys, i0), MS.deriv(Sg["normal"][i0, :], zs, j0)], 1),
                                np.tile(np.eye(3), (len(ch), 1))], 1)
            sig = np.tile(MS.NOISE, len(ch))
            feq = []
            for k in range(3):
                e = np.zeros(3)
                e[k] = 1e-6
                dB = (MS.B0(r, [p + e for p in ch]) - MS.B0(r, [p - e for p in ch])).ravel() / 2 * 1e6
                est = np.linalg.lstsq(J / sig[:, None], dB / sig, rcond=None)[0]
                feq.append([round(abs(float(est[0])), 4), round(float(np.hypot(est[1], est[2])), 4)])
            out.append(dict(layout=r["layout"], recipe=r["recipe"], magnets=len(r["magnets"]), placement=name,
                            gap_mm=round(gap, 3), E_star_MPa=round(r["E_best_MPa"], 2),
                            magnet_travel_um_per_N=round(float(np.mean(travel)), 2),
                            centre_force_N=round(a["centre"]["sigF_n_unk_bg"], 4), median_force_N=round(float(np.median(both)), 4),
                            range_force_N=[round(min(both), 4), round(max(both), 4)],
                            B_U1_mT=round(float(np.linalg.norm(B[0])), 2), B_max_mT=round(bmax, 2), gain_sel=g,
                            headroom=None if rng is None else round(rng / bmax, 2),
                            false_force_N_per_um={a: dict(normal=v[0], shear=v[1]) for a, v in zip(("x", "y", "z"), feq)}))
    return dict(rows=out, model="scripts/tpu_tip_magnet_sensing.py (point dipoles, N52 1.45 T; FE magnet motions of "
                                "docs/experiments/20261016-sim2real_fingertip_readiness/fe/fem.jsonl; MLX90393 noise at "
                                "GAIN_SEL 7, 141 Hz; Cramer-Rao bound with the contact position and a uniform background "
                                "field unknown); gain choice as scripts/sim2real_fingertip_sensing.py",
                note="the FE fixed the tip's back and mount faces; a gap below zero puts the chips 0.48 mm inside the "
                     "back face (recessed placement)")


# ------------------------------------------------------------------------------------------------------ clearance

def boxes():
    """Boxes fixed to the DIP link, per placement: the board with its tallest parts, the tab by length, J1, and a dorsal
    clamp plate behind the FR-4 by thickness and by how far it runs toward the PIP joint."""
    zb = sorted(v_to_z(BOARD_V))
    B = {}
    for p, P in PLACEMENTS.items():
        parts_x, fr4_x = placement_x(p, 1.0)
        lo_x, hi_x = min(parts_x[0], fr4_x[0]), max(parts_x[1], fr4_x[1])
        B[f"{p}|board"] = ((lo_x, -10.0, zb[0]), (hi_x, 10.0, zb[1]))
        for L in ((3.0, 6.0, 9.0) if p == "plate" else (3.0, 4.9, 6.0, 9.0)):
            B[f"{p}|tab{L:g}"] = ((lo_x, -5.0, zb[1]), (hi_x, 5.0, zb[1] + L))
        if p != "plate":
            jx, _ = placement_x(p, J1_H)
            jz = sorted(v_to_z(J1_V))
            B[f"{p}|J1"] = ((jx[0], J1_U[0], jz[0]), (jx[1], J1_U[1], jz[1]))
    back = PLACEMENTS["touch"]["comp_face"] + FR4
    for t in (2.0, 3.0, 4.0, 6.0, 8.0):
        B[f"touch|plate{t:g}"] = ((back, -10.5, G.TIP["z_end"]), (back + t, 10.5, zb[1]))
    for ztop in (-66.0, -64.0, -62.0, -60.0, -58.0):
        B[f"touch|plate2_to{ztop:g}"] = ((back, -10.5, zb[1]), (back + 2.0, 10.5, ztop))
    # the plate placement's boxes as the readiness sweep drew them (1.6 mm, U1 +-10 mm), for the comparison
    B["readiness|board"] = ((9.45, -10.0, -88.16), (11.05, 10.0, -68.16))
    for L in (3.0, 6.0, 9.0):
        B[f"readiness|tab{L:g}"] = ((9.45, -5.0, -68.16), (11.05, 5.0, -68.16 + L))
    return B


def clearance(recompute=True):
    B = boxes()
    if not recompute and CLEAR_CACHE.exists():
        res = json.loads(CLEAR_CACHE.read_text())
        if set(res) == set(B):
            return B, res
    t0 = time.time()
    res = G.boxes_clearance_exact(B)
    print(f"{len(B)} boxes, {time.time() - t0:.0f} s", flush=True)
    CLEAR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CLEAR_CACHE.write_text(json.dumps(res, indent=1))
    return B, res


def keepout_summary():
    K = G.keepout()
    m = G.load_part("dip_link")
    from morphohand.geometry import cad_features as cf
    P = cf.voxelize_solid(m, 0.25).centres()
    d = K.query(P)
    i = int(np.argmin(d))
    yspan = {}
    for name in K.parts:
        b = G.load_part(name).bounds
        yspan[name] = [round(float(b[0][1]), 2), round(float(b[1][1]), 2)]
    return dict(file=str(G.KEEPOUT.relative_to(REPO)), q_range_deg=list(K.q_range), parts=K.parts, pitch_mm=K.pitch,
                lo_F=K.lo.tolist(), shape=list(K.occ.shape), occupied_mm3=round(float(K.occ.sum()) * K.pitch ** 3),
                accuracy_mm=0.3,
                current_dip_link_min_mm=round(float(d[i]), 2), current_dip_link_min_at_F=np.round(P[i], 2).tolist(),
                parent_y_span_F=yspan,
                use="dip_link_geometry.keepout().query(points) gives the signed distance (mm, negative inside) of points "
                    "fixed to the DIP link to the MCP link and PIP servo swept over the PIP range, to 0.3 mm; "
                    "dip_link_geometry.points_clearance_exact and boxes_clearance_exact give the exact value")


# ---------------------------------------------------------------------------------------------------- other inputs

def joint_ranges():
    src = BUILDER.read_text()
    m = re.search(r'^ROM = \{"yaw": \(([-\d.]+), ([-\d.]+)\), "mcp": \(([-\d.]+), ([-\d.]+)\), "pip": \(([-\d.]+), ([-\d.]+)\)\}',
                  src, re.M)
    rom = [math.degrees(float(v)) for v in m.groups()]
    drv = {}
    txt = SERVOS_PY.read_text()
    for f, body in re.findall(r"^\s+(\d): \{(.*?)\},\s*$", txt, re.M):
        mm = re.search(r'"fe1": \(\d+, [-\d.]+, \(([-\d.]+), ([-\d.]+)\)\), "fe2": \(\d+, [-\d.]+, \(([-\d.]+), ([-\d.]+)\)\)', body)
        if mm:
            drv[f"finger {f}"] = dict(mcp_deg=[float(mm.group(1)), float(mm.group(2))], pip_deg=[float(mm.group(3)), float(mm.group(4))])
    return dict(model_deg=dict(yaw=[round(rom[0], 2), round(rom[1], 2)], mcp=[round(rom[2], 2), round(rom[3], 2)],
                               pip=[round(rom[4], 2), round(rom[5], 2)]),
                model_src="scripts/build_cad_v1_finger.py ROM (real_v1's ranges)",
                driver_calibrated_deg=drv,
                driver_src="src/morphohand/driver/manta/host/manta_hand/servos.py FINGER_JOINTS (the workstation's copy; the "
                           "CB1's running driver may differ), fe1 = MCP, fe2 = PIP",
                design_range_pip_deg=list(G.PIP_RANGE_DEG),
                note="every calibrated PIP range lies inside the model's -18..92 deg, which the keep-out sweeps")


def readiness():
    I = json.loads((READY / "scenes/interference.json").read_text())
    S = json.loads((READY / "summary.json").read_text())
    rows = {}
    for tab in (9, 6, 3, 0):
        r = I.get(f"block4_g30|tab{tab}", {}).get("index")
        if r:
            mc = r["min_clearance_mm"]
            rows[f"tab{tab}"] = dict(board_to_mcp_link=mc.get("board|mcp_link"), board_to_servo_pip=mc.get("board|servo_pip"),
                                     tab_to_mcp_link=mc.get("board_tab|mcp_link"), tab_to_servo_pip=mc.get("board_tab|servo_pip"),
                                     tab_into_dip_link=r["board_to_dip_link_mm"].get("board_tab"))
    C = S["cells"]
    per = {}
    for k, v in C.items():
        if not k.startswith("hom|") or v.get("board_runs") is None or v["variant"] == "block_e10":
            continue
        per.setdefault(v["variant"], []).append(v["board_runs"])
    br = [b for v in per.values() for b in v]
    runs = [v["runs"] for k, v in C.items() if k.startswith("hom|") and v.get("board_runs") is not None and v["variant"] != "block_e10"]
    peak = [v.get("peak_tau_Nm") for v in C.values() if v.get("peak_tau_Nm") is not None]
    grip = [v["grip_N"] for v in S["policies"].values() if v.get("grip_N") and v.get("n_held")]
    return dict(interference_index=rows, interference_src="docs/experiments/20261016-sim2real_fingertip_readiness/scenes/interference.json",
                hom_board_contact=dict(runs_per_cell=[min(br), max(br)], of=runs[0], cells=len(br), total=[sum(br), sum(runs)],
                                       per_variant={k: [min(v), max(v)] for k, v in sorted(per.items())},
                                       text="runs of 27 per HOM cell in which the board touched the cylinder; the contact is at "
                                            "the board's 2.6 mm overhang on each side of the 14.8 mm finger"),
                hom_peak_joint_torque_Nm=max(peak) if peak else None, policy_grip_N=[round(min(grip), 1), round(max(grip), 1)])


FASTENERS = dict(
    board_holes=dict(d_mm=2.40, plated=True, fits="M2 (0.20 mm radial clearance), M1.6 (0.40 mm); M2.5 does not pass"),
    catalogue=[
        dict(item="M2 cheese head, ISO 1207 / DIN 84", head_d_mm=3.8, head_h_mm=1.3),
        dict(item="M2 socket head, ISO 4762", head_d_mm=3.8, head_h_mm=2.0),
        dict(item="M1.6 cheese head, DIN 84", head_d_mm=3.0, head_h_mm=1.0),
        dict(item="M1.6 socket head, ISO 4762", head_d_mm=3.0, head_h_mm=1.6),
        dict(item="M2 hex nut, ISO 4032", across_flats_mm=4.0, h_mm=1.6),
        dict(item="M1.6 hex nut, ISO 4032", across_flats_mm=3.2, h_mm=1.3),
        dict(item="M2 x 3 brass heat-set insert (typical: 3.6 mm outside, 3.2 mm hole, 3.0 mm long)", outer_d_mm=3.6,
             hole_d_mm=3.2, length_mm=3.0),
        dict(item="M2 thread-forming into PLA", pilot_d_mm=[1.6, 1.8], note="the CAD's servo-tab pilots are 1.8 mm"),
    ],
    catalogue_src="ISO/DIN nominal dimensions and a typical insert; the owner's screw and insert stock is pending",
    magnetic="No ferromagnetic fastener within 15 mm of a chip: brass, nylon or titanium screws, brass inserts; austenitic "
             "stainless (A2/A4) can be weakly magnetic after cold forming. A magnetized or magnetizable part next to the chips "
             "adds a field that depends on the magnets' field and on its own history; the readout model has only the tip's "
             "magnets and a uniform background field")


def fastener_holes():
    """Holes up to 6.5 mm in the four PLA parts of the finger, by diameter (the fasteners the CAD uses)."""
    from morphohand.geometry import cad_features as cf
    out = {}
    for p in ("carriage", "yaw_link", "mcp_link", "dip_link"):
        for c in cf.cylinders(G.load_part(p)):
            if c.kind == "hole" and 2 * c.radius <= 6.5 and c.cover_deg >= 270:
                d = f"{2 * c.radius:.2f}"
                out.setdefault(d, []).append(p)
    meaning = {"1.60": "the four horn-arm screw holes of the DIP link (y -20.15, z -58.05..-66.05)",
               "1.80": "servo-tab pilot holes, M2 thread-forming (carriage, yaw link, MCP link)",
               "2.30": "the horn's centre screw (DIP link, y -22.45)", "2.40": "M2 clearance (carriage rail block)",
               "3.30": "M3 clearance: the DIP plate's screw into the CAD insert", "5.20": "the slotted bore for the 5 mm shaft"}
    return {d: dict(parts=sorted(set(v)), count=len(v), use=meaning.get(d, "")) for d, v in sorted(out.items(), key=lambda kv: float(kv[0]))}


PRINTER = dict(
    machine="Original Prusa XL 5T (five toolheads, 0.4 mm nozzles, 360 x 360 x 360 mm)",
    presets="Prusa XL 5T IS 0.4, 0.20mm STRUCTURAL (links); Siraya Tech TPU 85A on tool 2 (tips)",
    materials="PLA or PETG for rigid parts; TPU 85A and PLA or PETG can share one print (five tools)",
    links_as_built="PLA at 15 % infill (owner, 2026-10-13)",
    rules=["walls at least 0.9 mm (two 0.45 mm perimeters)", "clamp and seat regions solid (100 % infill or perimeters "
           "throughout)", "seat faces for the board printed as top or bottom surfaces, not on supports",
           "horizontal holes print about 0.1-0.2 mm undersize: give clearance or ream", "layer height 0.1-0.2 mm in the "
           "board seat so a 0.6 mm pocket is three to six layers"])


def component_windows(board, margin=0.2):
    """Area of the populated parts' footprints, grown by `margin`, inside the block tip's back face (|u| <= 7.4 mm,
    v from the mount face to the end): the windows a support shim needs at the touch placement."""
    v_lo, v_hi = -(G.TIP["z_mount"] - G.Z_CONTACT), -(G.TIP["z_end"] - G.Z_CONTACT)
    area = 0.0
    for k, p in board["parts"].items():
        if not p["populated"] or p["kind"] == "J1":
            continue
        su, sv = p["size_uv"]
        cu, cv = p["centre_uv"]
        du = max(0.0, min(cu + su / 2 + margin, 7.4) - max(cu - su / 2 - margin, -7.4))
        dv = max(0.0, min(cv + sv / 2 + margin, v_hi) - max(cv - sv / 2 - margin, v_lo))
        area += du * dv
    back = 14.8 * (v_hi - v_lo)
    return dict(area_mm2=area, back_mm2=back, share=area / back, margin_mm=margin)


# ---------------------------------------------------------------------------------------------- the specification

def build(recompute_clearance=True):
    t0 = time.time()
    board = G.board()
    print("board", flush=True)
    dip = current_dip_link()
    ins = cad_insert()
    print("current parts", flush=True)
    tp = tips()
    print("tips", flush=True)
    sens = sensing()
    print("sensing", flush=True)
    B, clr = clearance(recompute_clearance)
    ko = keepout_summary()
    jr = joint_ranges()
    rd = readiness()
    fh = fastener_holes()
    print("inputs", round(time.time() - t0), "s", flush=True)

    def worst(name):
        v = [(c[0], p, c[1]) for p, c in clr[name].items() if c[0] is not None]
        return min(v) if v else (None, None, None)

    clear_rows = []
    for name, (lo, hi) in B.items():
        w = worst(name)
        clear_rows.append(dict(box=name, lo_F=[round(v, 3) for v in lo], hi_F=[round(v, 3) for v in hi],
                               min_mm=w[0], part=w[1], at_pip_deg=w[2],
                               per_part={p: c[0] for p, c in clr[name].items()}))
    C = {r["box"]: r for r in clear_rows}

    def tab_max(p, need=0.5):
        ok = [float(n.split("tab")[1]) for n in C if n.startswith(f"{p}|tab")
              and (C[n]["min_mm"] is None or C[n]["min_mm"] >= need)]
        return max(ok) if ok else 0.0

    S = {(r["layout"], r["recipe"], r["placement"]): r for r in sens["rows"]}

    def f(layout, recipe, placement, key="centre_force_N"):
        return S[(layout, recipe, placement)][key]

    touch_free = [r for r in clear_rows if r["box"].startswith("touch|plate") and "_to" not in r["box"]]
    plate_ok = [float(r["box"].split("plate")[1]) for r in touch_free if r["min_mm"] is None or r["min_mm"] >= 0.5]
    prox = [r for r in clear_rows if r["box"].startswith("touch|plate2_to")]
    prox_ok = [float(r["box"].split("_to")[1]) for r in prox if r["min_mm"] is None or r["min_mm"] >= 0.5]
    mags4 = tp["block4"]["magnets"]
    loads = dict(design_normal_N=10.0, design_tangential_N=5.0, cable_pull_N=1.0,
                 basis=f"HOM cells on the CAD hand reach {rd['hom_peak_joint_torque_Nm']} N m at a joint (summary.json cells, "
                       f"peak_tau_Nm), {rd['hom_peak_joint_torque_Nm'] / 0.02661:.1f} N at the 26.61 mm from the PIP axis to the "
                       f"contact point; the D6 policies grip the screwdriver with {rd['policy_grip_N'][0]}-{rd['policy_grip_N'][1]} N "
                       "in total (summary.json policies, grip_N)")
    ff_touch = S[("four_d3x2_d5", "w85_30", "touch")]["false_force_N_per_um"]
    ff_touch50 = S[("four_d3x2_d5", "w85_50", "touch")]["false_force_N_per_um"]

    win = component_windows(board)
    req = [
        dict(id="R1", topic="Board distance",
             text="The chips' package tops lie against the printed tip's back face with no rigid layer between the TPU and "
                  "the five chips (the touch placement: sensing plane 0.42 +- 0.10 mm behind the back face).",
             threshold="sensing plane at most 0.52 mm behind the back face",
             gain=f"force resolution at the contact point, 85A 30 % gyroid: four magnets {f('four_d3x2_d5', 'w85_30', 'plate')} N "
                  f"(current plate) -> {f('four_d3x2_d5', 'w85_30', 'touch')} N; five magnets "
                  f"{f('five_d3x2_d5', 'w85_30', 'plate')} -> {f('five_d3x2_d5', 'w85_30', 'touch')} N (model)",
             source="sensing rows; print sheets' chip_field assumes this placement"),
        dict(id="R2", topic="Tip support",
             text="The tip's back face rests on a rigid flat surface over its area outside the component windows, and its "
                  "mount face on a rigid face, as the finite elements assumed (back and mount faces fixed). Around the chips "
                  "a rigid shim at the chip-top height (0.80-1.00 mm, nominal 0.90) or the FR-4 itself (recessed placement) "
                  "supports the back face; windows clear each part by 0.2 mm.",
             threshold=f"back face supported everywhere but the windows ({win['area_mm2']:.0f} mm2, {100 * win['share']:.0f} % of "
                       f"the block's {win['back_mm2']:.0f} mm2); support flat to 0.05 mm",
             source="scripts/sim2real_fingertip_fe.py boundary condition"),
        dict(id="R3", topic="Board retention",
             text="The board is clamped normal to its plane between rigid parts by screw preload and located in its plane by "
                  "rigid features (pins in two of its 2.40 mm holes, or pocket walls) with at most 0.05 mm clearance. Its "
                  "position depends on neither the tip nor TPU friction. No clamp force bears on U1-U5, C1-C5, R1-R2 or J1.",
             threshold=f"play and slip of the board relative to the DIP link below 1 um under the design loads (elastic "
                       f"motion is calibrated with the tip); 1 um of board motion toward the magnets reads as "
                       f"{ff_touch['x']['normal']:.3f} N of normal force and 1 um across as {ff_touch['y']['shear']:.3f} N of "
                       f"shear (four-magnet 30 % block, touch placement; {ff_touch50['x']['normal']:.3f} and "
                       f"{ff_touch50['y']['shear']:.3f} N at 50 %)",
             source="sensing rows, false_force_N_per_um"),
        dict(id="R4", topic="Tip retention",
             text="The tip is retained in all three axes by rigid features of the DIP link, keeps its back face against "
                  "the chips or shim, and comes off without unclamping the board. No retention feature or fastener comes "
                  "within 2.0 mm of a magnet pocket or through the palmar half of the tip.",
             threshold="tip-to-board position repeatable to 0.1 mm across remounts (the zero-load field is re-read after "
                       "each mount); no slip under the design loads",
             source=f"magnet pockets: the TPU around them is {min(m['tpu_wall_min_mm'] for t in tp.values() for m in t['magnets']['items']):.2f}"
                    f"-{max(m['tpu_wall_min_mm'] for t in tp.values() for m in t['magnets']['items']):.2f} mm in the print sheets"),
        dict(id="R5", topic="Contact geometry",
             text="The outer surface of each printed tip is unchanged (block, capsule r 8.5 mm, half-ellipsoids 7.4 x 8.5 x "
                  "16/32 and 11/22 mm): all share the flat back face at x_F 6.45 and the mount face at z_F -67.30. Over the "
                  "tip's length nothing protrudes beyond the tip's outer surface palmar of its back plane (retention features "
                  "may enter pockets or grooves of the tip, R4), and nothing is wider than the board (|y_F| <= 10.0 mm) "
                  "except a rim that covers the board's edge, at most 0.5 mm per side.",
             threshold="assembly width over the tip's length at most 21.0 mm; the board's FR-4 edge not exposed",
             source=f"HOM: the board's 2.6 mm overhang touched the cylinder in {rd['hom_board_contact']['runs_per_cell'][0]}-"
                    f"{rd['hom_board_contact']['runs_per_cell'][1]} of 27 runs per cell"),
        dict(id="R6", topic="Joint interface",
             text="The clevis arms of fingertip_mount_DIP_link.stl are kept as they are: the horn-side arm (y_F -23.25.."
                  "-17.05: the 2.0 mm slot for the 1.6 mm horn, the 6.4 mm hub recess, the 2.3 mm centre hole, the four "
                  "1.6 mm horn-arm holes at z_F -58.05, -61.05, -63.55, -66.05) and the slotted arm (y_F 13.40..17.40, the "
                  "5.2 mm slotted bore on the PIP axis). Everything between them is redesigned.",
             threshold="arms identical to the STL within 0.01 mm", source="assets/cad_v1/fingertip_mount_DIP_link.stl"),
        dict(id="R7", topic="Clearance over the joint range",
             text="Over PIP -18 to 92 deg every new part (bridge, board, tab or cable exit, clamp, fasteners, strain relief) "
                  "stays clear of the MCP link and the PIP servo; the yaw link, MCP servo and carriage stay 24 mm or more away "
                  "(readiness sweep).",
             threshold="0.5 mm for rigid parts, 1.0 mm for the cable, by dip_link_geometry.points_clearance_exact",
             source=f"keep-out {ko['file']}"),
        dict(id="R8", topic="Connector and cable",
             text=f"At the touch placement J1 (2.9 mm tall on the component side) overlaps the MCP link by "
                  f"{-C['touch|J1']['min_mm']:.2f} mm within the PIP range and the whole 9 mm tab by {-C['touch|tab9']['min_mm']:.2f} "
                  f"mm; a tab of {tab_max('touch'):g} mm keeps 0.5 mm. So the tab is cut to that length and four wires (VDD, GND, "
                  "SDA, SCL) are soldered to it, or the board is flipped. The cable is anchored to the DIP link next to the "
                  "board so no cable force reaches the board, crosses the PIP joint where its length change over the range "
                  "is small or taken up by a loop, and ends in an inline connector on the MCP link side, so the board comes "
                  "off with its pigtail.",
             threshold=f"tab at most {tab_max('touch'):g} mm at the touch placement; cable length change over the range "
                       "within its loop: r x 1.92 rad at a crossing r mm from the PIP axis (9.6 mm at r 5)",
             source="clearance rows; board J1 geometry from the Gerber files"),
        dict(id="R9", topic="Assembly and replacement",
             text="The tip and the board are each replaceable with hand tools without removing the DIP link from the finger; "
                  "the board's location survives a tip exchange; the thumb uses the same part (the same finger turned "
                  "180 deg about the vertical).",
             threshold="tip exchange without touching the board's screws", source="owner request"),
        dict(id="R10", topic="Magnetics",
             text=FASTENERS["magnetic"] + ". The board comes no closer to the PIP servo than now.",
             threshold="no ferromagnetic part within 15 mm of a chip; board to PIP servo at least 6.3 mm over the range",
             source="readiness sweep: board 6.33 mm from the PIP servo"),
        dict(id="R11", topic="Printability",
             text="Rigid parts in PLA or PETG on the Prusa XL with 0.4 mm nozzles; walls 0.9 mm or more; the board seat and "
                  "clamp faces flat without supports; solid infill in the clamp. TPU 85A features may be co-printed.",
             threshold="board seat flat to 0.05 mm", source="printer"),
        dict(id="R12", topic="Mass",
             text=f"The new DIP link with clamp and fasteners weighs at most 2 g more than the current DIP link "
                  f"({dip['mass_g']} g) with the board ({board['mass_g']} g).",
             threshold=f"at most {dip['mass_g'] + board['mass_g'] + 2.0:.1f} g without the tip", source="assembly.json"),
    ]
    concepts = [
        dict(name="Sandwich through the board's holes, screws from the component side",
             text="Dorsal backing plate (part of the DIP link) with brass inserts; the board on it located by two printed pins; "
                  "a 0.9 mm shim frame over the component side with windows; M1.6 or low-head M2 screws through the shim and "
                  "the board's holes into the inserts. The heads (3.0-3.8 mm) sit at |y| 6.1-9.9 mm and overlap the tip's "
                  "14.8 mm width by 0.9-1.3 mm: the tip needs four notches at its back edges, or narrower heads.",
             pros="preload on all four corners; board and shim stay when the tip comes off", cons="tip notches; screw heads beside the tip"),
        dict(name="Sandwich with screws from the dorsal side into the frame",
             text="Screws through the backing plate and the board into bosses of a component-side frame beside the tip; the "
                  "bosses (3.2-5.6 mm) overlap the tip's width by 1.0-2.2 mm.", pros="heads on the dorsal side",
             cons="bosses beside the tip widen the contact footprint or need tip notches"),
        dict(name="Edge clamp without the holes",
             text="The board slides into a pocket of the backing plate; lips over its outer 1.4 mm (|u| 8.6-10.0 mm, clear of "
                  "U3 and U5, which reach |u| 8.5 mm) and a screwed bar across the proximal margin preload it.",
             pros="no screws near the chips", cons="lips add 0.8-1.0 mm per side; preload depends on the lip's stiffness"),
        dict(name="Rigid frame co-printed with the tip",
             text="A PLA or PETG frame printed with the 85A tip at its back edges and mount face; the board is sandwiched "
                  "between this frame and the DIP link's backing plate, located there by pins.",
             pros="one set of screws locates tip and board", cons="the board's clamp opens at each tip exchange; TPU-PLA "
                  "interface strength unmeasured"),
    ]
    decisions = [
        "Tab: cut the connector tab and solder four wires to it, or keep J1 and flip the board (chips away from the tip: "
        f"sensing plane {PLACEMENTS['flipped']['gap']:.2f} mm, four-magnet 30 % resolution {f('four_d3x2_d5', 'w85_30', 'flipped'):.3f} N "
        f"against {f('four_d3x2_d5', 'w85_30', 'touch'):.3f} N at the touch placement).",
        f"Recessed chips: pockets in the TPU's back face for the chips and capacitors put the sensing plane 0.48 mm inside the "
        f"back face (four-magnet 30 % resolution {f('four_d3x2_d5', 'w85_30', 'recessed'):.3f} N; static field at U1 "
        f"{S[('four_d3x2_d5', 'w85_30', 'recessed')]['B_U1_mT']} mT, GAIN_SEL {S[('four_d3x2_d5', 'w85_30', 'recessed')]['gain_sel']}); "
        "the tip's outer surface stays, its back face changes.",
        "Tip notches: whether the tip's back edges may be notched for screw heads or bosses beside the board's holes.",
        "Thinner TPU behind the magnets (moving the board into the tip's volume) changes the tip's stiffness and needs new "
        "finite elements; this specification keeps the tips' 17.0 mm depth.",
        "Fasteners: the screws, inserts and nuts in stock (M1.6, M2; brass, nylon or titanium).",
    ]
    pending = [
        "Board thickness and part heights on the owner's board (FR-4 0.6 mm nominal; chips 0.80-1.00 mm; J1 about 2.9 mm).",
        "Flatness of a printed 85A tip's back face (printed on the bed) and its mount face.",
        "The cable: wire gauge and outer diameter, and where it leaves the MCP link.",
        "The PIP servo's own cable route (not in the CAD).",
        "The field of the PIP servo's motor at the chips over the PIP range (bench: five chips read with the tip unloaded "
        "while the PIP joint moves through its range).",
        "Retention and creep of the board clamp: zero-load field before and after 50 presses at 3 N and after 1 h at 5 N.",
    ]
    terms = {
        "F": "finger frame of the index finger at zero joint angles (scripts/build_cad_v1_finger.py), mm: x along the yaw "
             "axis toward the dorsal side, y along the PIP axis, z up; the straight finger hangs along -z",
        "PIP angle": "q in degrees, positive in flexion (the tip moves toward -x); range -18 to 92 deg (model)",
        "Touch placement": "the board with its chips' package tops on the tip's back face",
        "Gap": "distance from the tip's back face to the chips' sensing plane, 0.42 mm under the package top, mm; negative "
               "inside the tip",
        "Force resolution": "Cramer-Rao bound (one standard deviation, N) of the normal force at the contact point from one "
                            "141 Hz sample of the five chips, with the contact position and a uniform background field "
                            "unknown (scripts/tpu_tip_magnet_sensing.py)",
        "False force": "normal force or shear (N) that the readout's estimator (normal and shear force, contact "
                              "position and background field unknown) reads when the board moves 1 um relative to the "
                              "magnets, along x (toward them), y or z",
        "Keep-out": "the volume that the MCP link and the PIP servo sweep in the DIP link's frame over the PIP range",
        "Clearance": "least signed distance (mm) from the swept parts' surface to a part fixed to the DIP link; negative "
                     "values overlap, by that depth",
        "Design loads": "10 N normal and 5 N tangential at the contact point, and a 1 N pull on the cable",
    }
    spec = dict(
        title="DIP link with a clamped five-chip magnetometer board: design specification",
        date="2026-10-18", status="specification; no part designed or printed; model values unless marked as measured",
        request=["we are going to redesign the DIP linkage entirely so that the board can be as close to the printed tip as "
                 "possible, while trying to keep the contact geometry the same. although that might change as well.",
                 "it's important that the board does not wiggle at all--youll want something to fasten/sandwich it in place"],
        script="scripts/dip_link_spec.py (geometry and checks: scripts/dip_link_geometry.py)",
        when=time.strftime("%Y-%m-%d %H:%M"),
        terms=terms,
        frames=dict(F=terms["F"], pip_axis_point_F=G.PIP_C.tolist(), pip_hinge_axis_F=[0, 1, 0],
                    contact_point_F=list(G.TIP["contact"]), tip_envelope_F=G.TIP,
                    tip_design_to_F="x_F = -x_t - 2.05, y_F = y_t, z_F = -67.30 - z_t (scripts/tpu_tip_design.py frame: "
                                    "palmar face x_t = +8.5, back x_t = -8.5, mount face z_t = 0)",
                    board_to_F="u = y_F, v = -(z_F + 78.16), U1 over the contact point, U2 toward the tip's end, the tab "
                               "toward the PIP joint; at the touch placement the FR-4's component face is at x_F 7.35",
                    thumb="the thumb is the same finger turned 180 deg about z; all three fingers use one part",
                    transforms_F_from_stl={k: G.assembly()["T_F_from_part_design_mm"][k] for k in
                                           ("dip_link", "tip", "mcp_link", "servo_pip")}),
        parts=parts_table(),
        current=dict(dip_link=dip, cad_insert=ins, readiness=rd,
                     board_placement=dict(PLACEMENTS["plate"], src="scripts/sim2real_fingertip_scenes.py board_boxes")),
        tips=tp, board=board, placements=PLACEMENTS, sensing=sens, clearance=dict(rows=clear_rows, keepout=ko,
                                                                                  accuracy_mm=0.2,
                                                                                  method="dip_link_geometry.boxes_clearance_exact: surface points of the MCP link and PIP servo at 0.25 mm, PIP every 0.5 deg; matches the readiness sweep (plate board 6.332 against 6.334 mm, tabs within 0.006 mm)"),
        summary=dict(tab_max_touch_mm=tab_max("touch"), tab_max_flipped_mm=tab_max("flipped"),
                     tab_max_recessed_mm=tab_max("recessed"), dorsal_plate_ok_mm=plate_ok,
                     dorsal_plate_2mm_reaches_z_F=max(prox_ok) if prox_ok else None,
                     magnets_block4_x_F=mags4["items"][0]["x_F"], tpu_behind_magnets_mm={k: t["magnets"]["tpu_behind_magnets_mm"] for k, t in tp.items()}),
        joint_ranges=jr, loads=loads, fasteners=dict(FASTENERS, cad_holes=fh), printer=PRINTER,
        mass=dict(dip_link_g=dip["mass_g"], board_g=board["mass_g"], tips_g={k: t["mass_g"] for k, t in tp.items()},
                  finger_model_g=85.2, owner_estimate_g=120),
        requirements=req, concepts=concepts, owner_decisions=decisions, pending=pending,
        files=dict(spec_json=str(SPEC.relative_to(REPO)), spec_md=str(SPEC_MD.relative_to(REPO)),
                   figures=[str(FIG.relative_to(REPO)), str(FIG_BOARD.relative_to(REPO))],
                   keepout=str(G.KEEPOUT.relative_to(REPO)),
                   reskin_download=str(G.RESKIN_DL.relative_to(REPO)) + " (untracked)"))
    return spec


# ---------------------------------------------------------------------------------------------------------- figure

def figure(spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle
    K = G.keepout()
    plt.rcParams.update({"font.family": "serif", "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17,
                         "legend.fontsize": 15, "xtick.labelsize": 15, "ytick.labelsize": 15})
    INK, MUTED = "#0b0b0b", "#52514e"
    C_BOARD, C_J1, C_MAG = "#2a78d6", "#eb6834", "#1baf7a"
    fig, ax = plt.subplots(figsize=(10.0, 11.6))

    # (a) side view
    jy = np.flatnonzero(np.abs(K.lo[1] + K.pitch * np.arange(K.occ.shape[1])) <= 10.5)
    proj = K.occ[:, jy, :].any(axis=1)
    ext = [K.lo[0] - K.pitch / 2, K.lo[0] + K.pitch * (K.occ.shape[0] - 0.5), K.lo[2] - K.pitch / 2,
           K.lo[2] + K.pitch * (K.occ.shape[2] - 0.5)]
    from matplotlib.colors import to_rgba

    def mask_image(mask, color):
        img = np.zeros(mask.T.shape + (4,))
        img[mask.T] = to_rgba(color)
        return img

    ax.imshow(mask_image(proj, "#cfcfcf"), origin="lower", extent=ext, interpolation="nearest", zorder=1)
    q0 = G.build_keepout(q_range=(0.0, 0.0))
    p0 = q0.occ[:, jy, :].any(axis=1)
    ax.imshow(mask_image(p0, "#6f6f6f"), origin="lower", extent=ext, interpolation="nearest", zorder=2)

    def section_y0(name, **kw):
        m = G.load_part(name)
        s = m.section(plane_origin=[0, 0, 0], plane_normal=[0, 1, 0])
        if s is None:
            return
        for ent in s.entities:
            pts = s.vertices[ent.points]
            ax.add_patch(Polygon(pts[:, [0, 2]], closed=True, **kw))

    import trimesh
    blk = trimesh.load(REPO / spec["tips"]["block4"]["stl"], force="mesh")
    blk.apply_transform(np.array([[0, 0, -1, 6.45], [0, 1, 0, 0], [1, 0, 0, -88.71], [0, 0, 0, 1.0]]))   # print frame -> F
    sb = blk.section(plane_origin=[0, 0, 0], plane_normal=[0, 1, 0])
    for ent in sb.entities:
        ax.add_patch(Polygon(sb.vertices[ent.points][:, [0, 2]], closed=True, facecolor="#e8d9b5", edgecolor=MUTED, lw=1.0,
                             zorder=3))
    section_y0("dip_link", facecolor="none", edgecolor=INK, lw=1.4, zorder=4)
    for m in spec["tips"]["block4"]["magnets"]["items"]:
        x0, x1 = m["x_F"]
        z = m["centre_F"][2]
        ax.add_patch(Rectangle((x0, z - 1.5), x1 - x0, 3.0, facecolor=C_MAG, edgecolor="none", zorder=5))
    # board at the touch placement
    zb = sorted(v_to_z(BOARD_V))
    ax.add_patch(Rectangle((7.35, zb[0]), FR4, zb[1] - zb[0], facecolor=C_BOARD, edgecolor="none", zorder=6))
    for v in (-7.0, 0.0, 7.0):
        ax.add_patch(Rectangle((6.45, v_to_z(v) - 1.5), CHIP_H, 3.0, facecolor=C_BOARD, edgecolor="none", zorder=6))
    ax.add_patch(Rectangle((7.35, zb[1]), FR4, 9.0, facecolor="none", edgecolor=C_BOARD, lw=1.6, ls="--", zorder=6))
    jz = sorted(v_to_z(J1_V))
    ax.add_patch(Rectangle((7.35 - J1_H, jz[0]), J1_H, jz[1] - jz[0], facecolor=C_J1, edgecolor="none", alpha=0.9, zorder=7))
    ax.add_patch(Rectangle((9.45, zb[0]), 1.6, zb[1] - zb[0], facecolor="none", edgecolor=C_BOARD, lw=1.2, ls=":", zorder=6))
    ax.plot([0], [-51.55], "o", color="white", ms=8, mec=INK, mew=1.5, zorder=8)
    ax.annotate("PIP axis", (0, -51.55), (-20.5, -36.5), fontsize=16, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=1.0))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#cfcfcf", label="Swept by the MCP link and PIP\nservo, PIP \u221218\u00b0 to 92\u00b0"),
               Patch(facecolor="#6f6f6f", label="The same parts at PIP 0\u00b0"),
               Patch(facecolor="#e8d9b5", edgecolor=MUTED, label="Printed tip (block)"),
               Line2D([], [], color=INK, lw=1.4, label="Current DIP link"),
               Patch(facecolor=C_MAG, label="Magnets (y = \u00b13 mm)"),
               Patch(facecolor=C_BOARD, label="Board, touch placement"),
               Line2D([], [], color=C_BOARD, lw=1.6, ls="--", label="Its 9 mm connector tab"),
               Patch(facecolor=C_J1, label="J1 on that tab"),
               Line2D([], [], color=C_BOARD, lw=1.2, ls=":", label="Board on the current plate")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.09), frameon=False, fontsize=15, handlelength=1.6,
              labelspacing=0.45, ncol=2, columnspacing=1.2)
    ax.set_xlim(-22, 28)
    ax.set_ylim(-92, -34)
    ax.set_aspect("equal")
    ax.set_xlabel("x in F (mm), dorsal to the right")
    ax.set_ylabel("z in F (mm)")
    ax.set_title("Side view at y = 0, keep-out over |y| \u2264 10.5 mm", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=100, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # (b) the board, component side, with the tip's footprint
    fig, bx = plt.subplots(figsize=(10.0, 9.0))
    b = spec["board"]
    for (p0_, p1_) in b["outline"]["segments_uv"]:
        bx.plot([p0_[0], p1_[0]], [p0_[1], p1_[1]], color=INK, lw=1.6)
    for h in b["mounting_holes"]:
        bx.add_patch(Circle(h["uv"], h["d_mm"] / 2, facecolor="white", edgecolor=INK, lw=1.4))
        bx.add_patch(Circle(h["uv"], 3.8 / 2, facecolor="none", edgecolor=C_J1, lw=1.2, ls="--"))
    for k, p in b["parts"].items():
        su, sv = p["size_uv"]
        cu, cv = p["centre_uv"]
        if p["kind"] == "MLX90393":
            bx.add_patch(Rectangle((cu - su / 2, cv - sv / 2), su, sv, facecolor=C_BOARD, edgecolor="none"))
            bx.text(cu, cv, k, ha="center", va="center", color="white", fontsize=13)
        elif p["kind"] == "J1":
            bx.add_patch(Rectangle((p["body_u"][0], p["body_v"][0]), p["size_uv"][0], p["size_uv"][1], facecolor=C_J1,
                                   edgecolor="none", alpha=0.9))
            bx.text(cu, cv, "J1", ha="center", va="center", color=INK, fontsize=14)
        else:
            bx.add_patch(Rectangle((cu - su / 2, cv - sv / 2), su, sv, facecolor=MUTED if p["populated"] else "none",
                                   edgecolor=MUTED, lw=0.8))
    tz = (-(G.TIP["z_mount"] - G.Z_CONTACT), -(G.TIP["z_end"] - G.Z_CONTACT))
    bx.add_patch(Rectangle((-7.4, tz[0]), 14.8, tz[1] - tz[0], facecolor="#e8d9b5", edgecolor=MUTED, lw=1.2, alpha=0.45,
                           zorder=0))
    for m in spec["tips"]["block4"]["magnets"]["items"]:
        u, v = m["centre_F"][1], -(m["centre_F"][2] - G.Z_CONTACT)
        bx.add_patch(Circle((u, v), 1.5, facecolor="none", edgecolor=C_MAG, lw=2.0))
    bx.annotate("Tip's back face\n(14.8 x 21.41 mm)", (-7.4, 9.5), (-17.5, 13.0), fontsize=16, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    bx.annotate("Hole 2.40 mm; dashed:\nM2 head 3.8 mm", (8.0, 7.9), (5.0, 13.5), fontsize=16, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    bx.annotate("Magnets (four)", (3.0, 3.5), (11.0, 4.0), fontsize=16, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
    bx.annotate("Tab toward\nthe PIP joint", (5.0, -14.0), (8.0, -16.5), fontsize=16, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    bx.set_xlim(-18, 18)
    bx.set_ylim(-20.5, 16.5)
    bx.set_aspect("equal")
    bx.set_xlabel("u = y in F (mm)")
    bx.set_ylabel("v (mm), toward the tip's end")
    bx.set_title("ReSkin 5X board, component side, with the tip's back face", loc="left")
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_BOARD, dpi=100, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------- markdown

def md(spec):
    S = {(r["layout"], r["recipe"], r["placement"]): r for r in spec["sensing"]["rows"]}
    C = {r["box"]: r for r in spec["clearance"]["rows"]}
    sm = spec["summary"]
    b = spec["board"]
    rd = spec["current"]["readiness"]

    def c(name):
        r = C.get(name)
        if r is None or r["min_mm"] is None:
            return "clear (> 4 mm)"
        part = {"mcp_link": "MCP link", "servo_pip": "PIP servo"}[r["part"]]
        return f"{r['min_mm']:+.2f} ({part}, PIP {r['at_pip_deg']:g} deg)"

    L = []
    L.append(f"# {spec['title'][0].upper() + spec['title'][1:]}")
    L.append("")
    L.append(f"2026-10-18. Generated by `{spec['script'].split(' ')[0]}` from `spec.json` (every number with its source). "
             "Model values unless marked as measured; nothing here is designed or printed yet.")
    L.append("")
    four_t = S[("four_d3x2_d5", "w85_30", "touch")]
    four_p = S[("four_d3x2_d5", "w85_30", "plate")]
    five_t = S[("five_d3x2_d5", "w85_30", "touch")]
    five_p = S[("five_d3x2_d5", "w85_30", "plate")]
    j1 = C["touch|J1"]["min_mm"]
    L.append(
        f"The new DIP link puts the five chips' package tops on the printed tip's back face, removing the 3.0 mm plate between "
        f"them; the readout model then resolves {four_t['centre_force_N']:.3f} N at the contact point with four magnets and "
        f"{five_t['centre_force_N']:.3f} N with five (85A, 30 % gyroid), against {four_p['centre_force_N']:.3f} and "
        f"{five_p['centre_force_N']:.3f} N now. One micrometre of board motion toward the magnets reads as "
        f"{four_t['false_force_N_per_um']['x']['normal']:.3f} N, so the board is clamped by screw preload and located by rigid "
        f"features, independent of the tip. At this placement J1 overlaps the MCP link by {-j1:.2f} mm within the PIP range and "
        f"only a tab of {sm['tab_max_touch_mm']:g} mm or less keeps 0.5 mm clearance, so the tab is cut and the cable soldered, "
        "or the board is flipped. The board's four holes lie 0.6 mm inside the tip's side planes, which shapes every clamp.")
    L.append("")
    L.append("## Terms and frames")
    L.append("")
    for k, v in spec["terms"].items():
        L.append(f"- {k}: {v}.")
    L.append(f"- Board frame: (u, v) of the ReSkin 5X files, U1 at the origin; on the finger {spec['frames']['board_to_F']}.")
    L.append("- Placements: plate (now), touch, recessed and flipped, defined under the force-resolution table.")
    L.append("")
    L.append("## Requirements")
    L.append("")
    L.append("| ID | Topic | Requirement | Threshold |")
    L.append("|---|---|---|---|")
    for r in spec["requirements"]:
        L.append(f"| {r['id']} | {r['topic']} | {r['text']} | {r['threshold']} |")
    L.append("")
    L.append("## Board distance and force resolution")
    L.append("")
    L.append("Force resolution at the contact point (N) by magnet count and gyroid infill, and the static field at U1 (mT), per "
             "placement; magnet motions from the readiness study's finite elements of the 85A block.")
    L.append("")
    L.append("| Placement | Gap (mm) | Four, 30 % | Five, 30 % | Four, 50 % | Five, 50 % | Field at U1, four / five 30 % (mT) | GAIN_SEL, four / five 30 % |")
    L.append("|---|---|---|---|---|---|---|---|")
    for p in ("plate", "touch", "recessed", "flipped"):
        a, bb = S[("four_d3x2_d5", "w85_30", p)], S[("five_d3x2_d5", "w85_30", p)]
        c50, d50 = S[("four_d3x2_d5", "w85_50", p)], S[("five_d3x2_d5", "w85_50", p)]
        L.append(f"| {p.capitalize()} | {a['gap_mm']:.2f} | {a['centre_force_N']:.3f} | {bb['centre_force_N']:.3f} | "
                 f"{c50['centre_force_N']:.3f} | {d50['centre_force_N']:.3f} | {a['B_U1_mT']:.1f} / {bb['B_U1_mT']:.1f} | "
                 f"{a['gain_sel']} / {bb['gain_sel']} |")
    L.append("")
    for p, P in spec["placements"].items():
        L.append(f"- {p.capitalize()}: {P['text']}.")
    L.append("")
    ff = four_t["false_force_N_per_um"]
    L.append(f"Below GAIN_SEL 7 the datasheet gives no noise, so those resolutions are lower bounds. The readout's estimator "
             f"(normal and shear force, contact position and background field unknown) reads one micrometre of rigid board "
             f"motion relative to the magnets as {ff['x']['normal']:.3f} N of normal force along x (toward the magnets), "
             f"{ff['y']['shear']:.3f} N of shear along y and {ff['z']['shear']:.3f} N of shear along z (four magnets, 30 %, "
             f"touch); the magnets move {four_t['magnet_travel_um_per_N']:.1f} um per newton of normal load.")
    L.append("")
    L.append("## Clearance over the PIP range")
    L.append("")
    L.append(f"Least clearance (mm) of boxes fixed to the DIP link from the MCP link and the PIP servo, PIP -18 to 92 deg, "
             f"negative where they overlap; exact to 0.2 mm. The readiness sweep gave the plate board 6.33 mm and the 9, 6 "
             f"and 3 mm tabs -0.79, 0.63 and 3.48 mm; this method gives {C['readiness|board']['min_mm']:.2f}, "
             f"{C['readiness|tab9']['min_mm']:.2f}, {C['readiness|tab6']['min_mm']:.2f} and {C['readiness|tab3']['min_mm']:.2f}.")
    L.append("")
    L.append("| Part | Plate | Touch | Recessed | Flipped |")
    L.append("|---|---|---|---|---|")
    for lab, key in (("Board with its parts", "board"), ("Tab 3 mm", "tab3"), ("Tab 4.9 mm (J1's signal pads kept)", "tab4.9"),
                     ("Tab 6 mm", "tab6"), ("Tab 9 mm (whole)", "tab9"), ("J1, 2.9 mm tall", "J1")):
        L.append(f"| {lab} | " + " | ".join(c(f"{p}|{key}") if f"{p}|{key}" in C else "-" for p in ("plate", "touch", "recessed", "flipped")) + " |")
    L.append("")
    pl = [r for r in spec["clearance"]["rows"] if r["box"].startswith("touch|plate") and "_to" not in r["box"]]
    L.append("A dorsal backing plate behind the FR-4 at the touch placement (x_F from 7.95 mm, |y| <= 10.5 mm, over the board's "
             "length): " + ", ".join(f"{r['box'].split('plate')[1]} mm thick {c(r['box'])}" for r in pl) + ". The same plate 2 mm "
             "thick extended toward the PIP joint: " + ", ".join(
                 f"to z {r['box'].split('_to')[1]} mm {c(r['box'])}" for r in spec["clearance"]["rows"] if r["box"].startswith("touch|plate2_to")) + ".")
    L.append("")
    ko = spec["clearance"]["keepout"]
    L.append(f"The swept volume is stored in `{ko['file']}` ({ko['pitch_mm']} mm cells); `dip_link_geometry.keepout().query(points)` "
             "gives a candidate's signed distance to it within 0.3 mm and `points_clearance_exact` the exact value. The parent "
             f"parts span y_F {ko['parent_y_span_F']['mcp_link'][0]}..{ko['parent_y_span_F']['mcp_link'][1]} mm (MCP link) and "
             f"{ko['parent_y_span_F']['servo_pip'][0]}..{ko['parent_y_span_F']['servo_pip'][1]} mm (PIP servo), so a cable "
             "outside the slotted arm (y_F > 17.4 mm) crosses the joint clear of them.")
    L.append("")
    L.append(f"![Side view of the keep-out]({FIG.relative_to(OUT)})")
    L.append("")
    L.append("Figure 1: the side view at y = 0 with the volume the MCP link and the PIP servo sweep over PIP -18 to 92 deg "
             "within |y| <= 10.5 mm (light grey) and the same parts at PIP 0 deg (dark grey), the printed block tip, the "
             "current DIP link's section, the four magnets, the board at the touch placement with its 9 mm tab and J1, and "
             "the board on the current plate.")
    L.append("")
    L.append("## The board")
    L.append("")
    holes = ", ".join(f"({h['uv'][0]:+.2f}, {h['uv'][1]:+.2f})" for h in b["mounting_holes"])
    L.append(f"ReSkin 5X from `{b['source'].split(' ')[0]}` (5X_ReSkin.Zip, read as data; copy in `{b['local_copy']}`, untracked). "
             f"FR-4 {b['thickness_mm']} mm; outline u {b['outline']['u'][0]:.2f}..{b['outline']['u'][1]:.2f} mm, v "
             f"{b['outline']['v_main'][0]:.2f}..{b['outline']['v_main'][1]:.2f} mm, tab u -5.00..5.00 mm, v "
             f"{b['outline']['tab_v'][0]:.2f}..{b['outline']['tab_v'][1]:.2f} mm. Four plated 2.40 mm holes at (u, v) {holes} mm. "
             "U1 at the origin, U2-U5 7.00 mm away in a plus (U5 at v +0.05); every part on the component side: the five "
             "MLX90393 (3 x 3 mm, 0.80-1.00 mm tall, sensing spot 0.42 +- 0.10 mm under the top), C1-C5 0402 (0.5 mm), "
             "R1-R2 0402 at v -11.40 (0.35 mm), C6-C9 unpopulated pads at u -3.56, v -8.7..-12.0, and J1, a JST SH "
             f"side-entry header (body u {b['parts']['J1']['body_u'][0]:.2f}..{b['parts']['J1']['body_u'][1]:.2f}, v "
             f"{b['parts']['J1']['body_v'][0]:.2f}..{b['parts']['J1']['body_v'][1]:.2f} mm, about 2.9 mm tall, signal pads at v "
             "-13.70, mating face toward the tab's end). The bare side is flat. The holes' inner edges reach |u| 6.80 mm, inside "
             "the tip's 7.40 mm half-width; U3 and U5 reach |u| 8.50 mm, beyond it.")
    L.append("")
    L.append(f"![The board with the tip's back face]({FIG_BOARD.relative_to(OUT)})")
    L.append("")
    L.append("Figure 2: the board's component side in its own frame (u = y_F, v toward the tip's end) with the tip's back "
             "face (tan), the four magnets (green circles), the chips (blue), the capacitors and resistors (grey; C6-C9 "
             "unpopulated, outlined), J1 (orange) and an M2 head of 3.8 mm at each hole (orange, dashed).")
    L.append("")
    L.append("## The printed tips and magnets")
    L.append("")
    L.append("| Tip | Magnets, top under the apex | Magnet x_F (mm) | TPU behind the magnets (mm) | Back-face area (mm2) | Field at U1, touch (mT) | Mass with magnets (g) |")
    L.append("|---|---|---|---|---|---|---|")
    for k, t in spec["tips"].items():
        m = t["magnets"]
        L.append(f"| {t['label'].capitalize()} | {m['count']} x {m['size_mm'][0]:g} x {m['size_mm'][1]:g} mm, {m['top_depth_mm']:g} mm | "
                 f"{m['items'][0]['x_F'][0]:.2f}..{m['items'][0]['x_F'][1]:.2f} | {m['tpu_behind_magnets_mm']:.2f} | "
                 f"{t['outlines']['back']['area_mm2']:.0f} | {t['chip_field_touch']['B_U1_mT']:.1f} | {t['mass_g']:.2f} |")
    L.append("")
    L.append("Four magnets sit at y_F +-3.0 mm and z_F -74.66 and -81.66 mm, the fifth at the contact point (z_F -78.16). Every "
             "tip is printed whole: no tongue pocket, lip groove or screw hole, so its attachment is part of this design. "
             "The current DIP link's tongue (x_F -7.35..-4.75, to z_F -72.1) would run into the proximal magnet pockets "
             "(z_F -73.16 and below).")
    L.append("")
    L.append("## Joint ranges, loads, fasteners and printer")
    L.append("")
    jr = spec["joint_ranges"]
    drv = "; ".join(f"{k} {v['pip_deg'][0]:g}..{v['pip_deg'][1]:g} deg" for k, v in jr["driver_calibrated_deg"].items())
    L.append(f"- PIP range: model {jr['model_deg']['pip'][0]:g}..{jr['model_deg']['pip'][1]:g} deg; driver calibration {drv}.")
    L.append(f"- Design loads: {spec['loads']['design_normal_N']:g} N normal and {spec['loads']['design_tangential_N']:g} N "
             f"tangential at the contact point, {spec['loads']['cable_pull_N']:g} N on the cable. {spec['loads']['basis']}.")
    cad = "; ".join(f"{d} mm, {v['use']}" for d, v in spec["fasteners"]["cad_holes"].items() if v["use"])
    L.append(f"- Fastener holes in the CAD: {cad}. The board's holes take M2 (0.20 mm radial clearance) or M1.6 (0.40 mm); "
             "R10 sets the materials. Catalogue sizes: M2 cheese head 3.8 x 1.3 mm, M1.6 cheese head 3.0 x 1.0 mm, M2 nut 4.0 "
             "mm across flats x 1.6 mm, M2 x 3 brass insert 3.6 mm outside in a 3.2 mm hole.")
    L.append(f"- Printer: {spec['printer']['machine']}; {spec['printer']['materials']}; links as built: "
             f"{spec['printer']['links_as_built']}.")
    L.append(f"- Mass: DIP link {spec['mass']['dip_link_g']} g, board {spec['mass']['board_g']} g.")
    L.append("")
    L.append("## Candidate clamps")
    L.append("")
    for cpt in spec["concepts"]:
        L.append(f"- {cpt['name']}: {cpt['text']} For: {cpt['pros']}. Against: {cpt['cons']}.")
    L.append("")
    L.append("## Owner decisions")
    L.append("")
    for d in spec["owner_decisions"]:
        L.append(f"- {d}")
    L.append("")
    L.append("## Pending measurements")
    L.append("")
    for p in spec["pending"]:
        L.append(f"- {p}")
    L.append("")
    txt = "\n".join(L)
    for a, b in ((r"(\d) um\b", "\\1 \u00b5m"), (r"\b1 um\b", "1 \u00b5m"), (r"\bum per\b", "\u00b5m per"),
                 (r"(\d) deg\b", "\\1\u00b0"), (r"\+-", "\u00b1"), (r"<=", "\u2264"), (r"\bmm2\b", "mm\u00b2"),
                 (r"Cramer-Rao", "Cram\u00e9r-Rao")):
        txt = re.sub(a, b, txt)
    return txt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-clearance", action="store_true", help="reuse logs/20261018-dip_link/clearance_boxes.json")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    spec = build(recompute_clearance=not a.no_clearance)
    OUT.mkdir(parents=True, exist_ok=True)
    SPEC.write_text(json.dumps(spec, indent=1, default=float) + "\n")
    SPEC_MD.write_text(md(spec))
    if not a.no_figure:
        figure(spec)
    print(f"wrote {SPEC.relative_to(REPO)}, {SPEC_MD.relative_to(REPO)}" + ("" if a.no_figure else f", {FIG.name}, {FIG_BOARD.name}"))


if __name__ == "__main__":
    main()
