#!/usr/bin/env python3
"""Printable v2 of the DIP-link designs a and b (2026-10-18).

Owner, 2026-10-18 18:45: "we should just go for 5x 3x2 magnets"; "i do like the half ellipsoids most, the 11/22 one
especially"; "if we have to lengthen the fingertip to get everything to fit, including the qwiic port and the cable,
that's fine"; "i am a big fan of self-forming threads on pla. for m3 screws, i use 2.75mm diameter holes, m2 i use 1.8".
Output: docs/experiments/20261018-dip_link_board_mount/v2/.

  lengthen   Whether the board's own Qwiic connector J1 can stay plugged in. The DIP link is lengthened between the
             bridge and the tip by L, so that the tip, the board and everything fixed to them move distally by L; J1
             with its mated SHR-04V-S housing, 3 mm of cable and the uncut 9 mm tab (scripts/dip_link_board_mount.py
             j1_parts_board), tab proximal, is checked against the MCP link and the PIP servo over PIP -18..92 deg
             (dip_link_geometry.points_clearance_exact). The least L (0.25 mm) at which both clear by 0.5 mm, per board
             placement (concept a: component face x_F 7.35; concept b: its slot plane), and the L at which the tab distal
             fits inside a tip lengthened at its end; PIP-to-tip length 37.16 mm + L. -> v2/20261018-v2_lengthening.json

    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 400 -- taskset -c 8-11 env NUMPY_MADVISE_HUGEPAGE=0"
    $RG ~/miniconda3/envs/femtip/bin/python scripts/dip_link_v2.py lengthen
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import scipy.spatial  # noqa: F401  (the environment's libstdc++ before manifold3d)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("DIPA_REPO", str(REPO))
import numpy as np  # noqa: E402

STUDY = REPO / "docs/experiments/20261018-dip_link_board_mount"
V2 = STUDY / "v2"
DATE = "20261018"
PIP_TO_TIP = 37.16                  # PIP axis (z_F -51.55) to the tip's end (z_F -88.71), mm
Z_C, Z_END, Z_MOUNT, Y_BAND = -78.16, -88.71, -67.30, 10.5
NEED = 0.5                          # R7, rigid parts
MAX_LENGTHENING = 10.5              # owner, 2026-10-18 18:45, through the job's step 8: "if it adds at most about 10 mm"


def connector_parts(x_comp, theta, dz=0.0):
    """J1, its plug and 3 mm of cable (one trimesh) and the uncut tab (another), the board rotated by theta about U1 in its
    plane (0: tab proximal; 180: tab distal), its component face at x_comp with the parts toward the tip, moved distally
    by dz."""
    import trimesh
    import dip_link_board_mount as DLF
    th = math.radians(theta)
    meshes = {}
    for name, lo, hi in DLF.j1_parts_board():
        bm = DLF.box_mesh(lo, hi)
        u, v, w = (np.asarray(bm.vertices)[:, k] for k in range(3))
        ur, vr = u * math.cos(th) - v * math.sin(th), u * math.sin(th) + v * math.cos(th)
        V = np.column_stack([x_comp - w, ur, Z_C - vr - dz])
        tm = trimesh.Trimesh(V, np.asarray(bm.faces), process=False)
        if tm.volume < 0:
            tm.invert()
        meshes[name] = tm
    conn = trimesh.util.concatenate([m for k, m in meshes.items() if k != "tab_fr4"])
    return conn, meshes["tab_fr4"]


def clearance(conn, tab):
    import dip_link_geometry as G
    Pc = G.surface_points(conn, 0.25)
    v, i, q, part = G.points_clearance_exact(Pc, reach=3.0)
    Pt = G.surface_points(tab, 0.25)
    vt, _, qt, pt = G.points_clearance_exact(Pt, reach=3.0)
    return dict(connector_mm=round(v, 3), connector_at=dict(pip_deg=q, part=part, at_F=np.round(Pc[i], 2).tolist()),
                tab_mm=round(vt, 3), tab_at=dict(pip_deg=qt, part=pt), least_mm=round(min(v, vt), 3))


def least_lengthening(x_comp, step=0.25, hi=16.0, log=print):
    """The least L (a multiple of step) at which the tab-proximal connector and the tab clear by NEED, with every coarse
    L above it clearing too: a 1 mm scan, then the step inside the last failing millimetre."""
    rows = []
    t0 = time.time()
    for L in np.arange(0.0, hi + 1e-9, 1.0):
        c = clearance(*connector_parts(x_comp, 0, L))
        rows.append(dict(L_mm=float(L), **c))
        log(f"  x_comp {x_comp:5.2f}  L {L:5.2f}: connector {c['connector_mm']:+.3f} ({c['connector_at']['part']}, PIP "
            f"{c['connector_at']['pip_deg']}), tab {c['tab_mm']:+.3f} [{time.time() - t0:.0f} s]")
    ok = [r["least_mm"] >= NEED for r in rows]
    first = next((k for k in range(len(rows)) if all(ok[k:])), None)
    if first is None:
        return None, rows
    if first == 0:
        return 0.0, rows
    best = rows[first]["L_mm"]
    for L in np.arange(rows[first - 1]["L_mm"] + step, rows[first]["L_mm"] - 1e-9, step):
        c = clearance(*connector_parts(x_comp, 0, L))
        rows.append(dict(L_mm=float(L), **c))
        log(f"  x_comp {x_comp:5.2f}  L {L:5.2f}: connector {c['connector_mm']:+.3f}, tab {c['tab_mm']:+.3f}")
        if c["least_mm"] >= NEED:
            best = float(L)
            break
    rows.sort(key=lambda r: r["L_mm"])
    return best, rows


def distal_lengthening(x_comp, cover=1.0):
    """The tab distal: how far the connector reaches past the tip's end (mm), so the tip lengthened at its end by that
    and `cover` mm more holds it; and the band check (|y| <= 10.5)."""
    conn, tab = connector_parts(x_comp, 180)
    V = np.vstack([conn.vertices, tab.vertices])
    beyond = float(max(0.0, Z_END - V[:, 2].min()))
    return dict(beyond_tip_end_mm=round(beyond, 2), lengthening_mm=round(beyond + cover, 2), cover_mm=cover,
                half_width_mm=round(float(np.abs(V[:, 1]).max()), 2), inside_band=bool(np.abs(V[:, 1]).max() <= Y_BAND))


def cmd_lengthen(a):
    """Each placement to v2/lengthening/<name>.json (one process per placement runs in parallel), then the merge of every
    placement's file into v2/20261018-v2_lengthening.json."""
    (V2 / "lengthening").mkdir(parents=True, exist_ok=True)
    for name, x in a.placements:
        t0 = time.time()
        print(f"{name}: component face x_F {x}", flush=True)
        L, rows = least_lengthening(x, log=lambda s: print(s, flush=True))
        dist = distal_lengthening(x)
        rec = dict(component_face_x_F=x, tab_proximal_L_mm=L, tab_distal=dist, scan=rows,
                   pip_to_tip_tab_proximal_mm=None if L is None else round(PIP_TO_TIP + L, 2),
                   pip_to_tip_tab_distal_mm=round(PIP_TO_TIP + dist["lengthening_mm"], 2), seconds=round(time.time() - t0, 1),
                   when=time.strftime("%Y-%m-%d %H:%M"))
        (V2 / "lengthening" / f"{name}.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
        print(f"{name}: tab proximal L {L} mm; tab distal {dist['lengthening_mm']} mm [{time.time() - t0:.0f} s]", flush=True)
    merge()


def merge():
    out = dict(definition=("L: lengthening of the DIP link between its bridge and the tip's mount face, so that the tip, the "
                           "board and every part fixed to them move distally by L; J1, its mated SHR-04V-S housing, 3 mm of "
                           "cable and the uncut 9 mm tab, tab proximal, must clear the MCP link and the PIP servo by 0.5 mm "
                           "over PIP -18..92 deg (points_clearance_exact, 0.25 mm samples, 0.5 deg). Tab distal: the tip "
                           "lengthened at its end until the connector lies inside it with 1 mm to spare."),
               pip_to_tip_now_mm=PIP_TO_TIP, need_mm=NEED, when=time.strftime("%Y-%m-%d %H:%M"),
               placements={f.stem: json.loads(f.read_text()) for f in sorted((V2 / "lengthening").glob("*.json"))})
    choice = {}
    for name, r in out["placements"].items():          # owner 18:45: take the shorter if it adds at most about 10 mm
        Lp, Ld = r["tab_proximal_L_mm"], r["tab_distal"]["lengthening_mm"]
        best = min([(Lp, "tab proximal, the DIP link lengthened") if Lp is not None else (99.0, ""),
                    (Ld, "tab distal, the tip lengthened at its end")])
        choice[name] = dict(option=best[1] if best[0] <= MAX_LENGTHENING else "J1 removed, the header relocated as in a's final",
                            lengthening_mm=best[0] if best[0] <= MAX_LENGTHENING else 0.0,
                            pip_to_tip_mm=round(PIP_TO_TIP + (best[0] if best[0] <= MAX_LENGTHENING else 0.0), 2))
    out["choice"] = choice
    out["rule"] = f"the shorter of the two lengthenings if it is at most {MAX_LENGTHENING} mm (owner: 'about 10 mm'), else a's relocated header"
    dst = V2 / f"{DATE}-v2_lengthening.json"
    dst.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("->", dst.relative_to(REPO), json.dumps(choice), flush=True)


# ----------------------------------------------------------------------------------------------- five-magnet tips
PRINT = V2 / "print"
SEAT = 2.9
FIVE_TRIES = {   # tip: (magnets' top depth under the apex, layouts tried in order; five_y<a>_z<p>: corners at y +-a mm, p apart
    # along the finger, and the centre; the 11/22's upright slots need the rows ~9.8 mm apart for 1.8 mm between the corner and
    # centre slots, and at y +-3 mm the distal corner slots then come within 0.9-1.3 mm of the narrowing distal surface
    "tcd_ell7.4x8.5x11x22": (7.5, ["five_y2p5_z9p8_s0p5", "five_y2p5_z9p8_s1", "five_y2p7_z9p8_s1", "five_y3_z9p8_s1p5",
                                   "five_y2p5_z10_s1", "five_y2p5_z10_s1p5", "five_y2p3_z10_s1"]),
    "block": (5.0, ["five_z7p4"]),
}


def five_layout(name):
    """Magnet centres (y, z) in tpu_tip_design's frame (metric_magnet_prints.layout), proximal first: five_y<a>_z<p> puts the
    corners at y +-a and the contact point's z +-p/2, and the fifth magnet at the contact point; a suffix _s<d> moves the
    group d mm toward the mount face; other names go to metric_magnet_prints.layout."""
    import metric_magnet_prints as MP
    import tpu_tip_design as TD
    if not name.startswith("five_y"):
        return MP.layout(name)
    parts = name[6:].split("_")                      # y<a>, z<p>[, s<shift>]: the group shifted toward the mount face
    a = float(parts[0].replace("p", "."))
    p = float(parts[1][1:].replace("p", "."))
    sh = float(parts[2][1:].replace("p", ".")) if len(parts) > 2 else 0.0
    zc = TD.Z_C - sh
    return sorted([(sy * a, zc + sz * p / 2) for sy in (-1, 1) for sz in (-1, 1)] + [(0.0, zc)], key=lambda c: (c[1], c[0]))
TIP_KEY = {"tcd_ell7.4x8.5x11x22": "ell11_5", "block": "block5"}


def tip_record(sheet, stl):
    """The tip record of dip_link_spec.tips() for a v2 sheet: magnets in F, the STL, shape and orientation."""
    import dip_link_geometry as G
    mg = sheet["magnet"]
    mags = []
    for p in sheet["pockets"]:
        top = 8.5 - p["magnet_top_depth_mm"]
        c = G.tip_design_to_F([top - mg["thickness_mm"] / 2, p["y_mm"], p["z_mm"]])
        mags.append(dict(centre_F=np.round(c, 2).tolist(), x_F=[round(-top - 2.05, 2), round(-(top - mg["thickness_mm"]) - 2.05, 2)],
                         tpu_wall_min_mm=p["wall_min_mm"], tpu_over_magnet_mm=p["tpu_over_magnet_mm"]))
    return dict(sheet=str((PRINT / f"{sheet['name']}.json").relative_to(REPO)), stl=str(Path(stl).relative_to(REPO)),
                label=sheet["tip_label"], shape=sheet["tip"], print_orientation=sheet["orientation"],
                magnets=dict(count=mg["count"], size_mm=[mg["diameter_mm"], mg["thickness_mm"]], top_depth_mm=mg["top_depth_mm"],
                             polarity=mg["orientation"], items=mags,
                             tpu_behind_magnets_mm=round(6.45 - max(m["x_F"][1] for m in mags), 2)),
                chip_field_touch=sheet["chip_field"], mass_g=None)


def cmd_tips(a):
    """Five 3 x 2 mm magnets at the 2.9 mm seat in the 11/22 half-ellipsoid and in the block, every pocket wall at least
    1.8 mm (metric_magnet_prints.walls_pass: to the outer surface and between pockets, slots included): the first layout of
    FIVE_TRIES that passes, its STL in the print frame and its 85A sheet at 30 % gyroid (sim2real_fingertip_prints
    write_and_check, sliced) under v2/print/, and the tip records for the generators -> v2/20261018-v2_tips.json. The
    sheets here have no attachment features; the concepts' channelled tips are sliced by their own commands."""
    import metric_magnet_prints as MP
    import sim2real_fingertip_prints as SP
    MP.DATE = SP.DATE = DATE
    SP.OUT, SP.WORK = PRINT, REPO / "logs/20261018-dip_link/v2/print"
    pauses0 = MP.pauses

    def pauses(pockets, tip):                         # the group's middle magnet is the centre one, wherever the group sits
        rows = pauses0(pockets, tip)
        zs = sorted({round(p["z_mm"], 2) for p in pockets})
        for r in rows:
            if tip != "block" and {round(pockets[i]["z_mm"], 2) for i in r["magnets"]} == {zs[len(zs) // 2]}:
                r["message"] = r["message"].replace("(proximal) ", "(centre) ").replace("(distal) ", "(centre) ")
        return rows
    MP.pauses = pauses
    PRINT.mkdir(parents=True, exist_ok=True)
    SP.use_template()
    rec = {}
    for tip, (depth, tries) in FIVE_TRIES.items():
        rows = []
        for lay in tries:
            t0 = time.time()
            body, part, pockets, gaps, mesh = MP.build(tip, five_layout(lay), SEAT, depth)
            ok = MP.walls_pass(pockets, gaps)
            rows.append(dict(layout=lay, walls_ok=bool(ok), wall_min_mm=min(p["wall_min_mm"] for p in pockets),
                             gap_min_mm=min(g["gap_mm"] for g in gaps), seconds=round(time.time() - t0, 1)))
            print(f"{tip} {lay}: walls ok {ok}, least wall to the surface {rows[-1]['wall_min_mm']}, least gap between "
                  f"pockets {rows[-1]['gap_min_mm']} [{rows[-1]['seconds']} s]", flush=True)
            if ok:
                break
        if not ok:
            rec[TIP_KEY[tip]] = dict(tip=tip, tried=rows, chosen=None)
            continue
        stl = PRINT / f"{MP.name_of(tip, lay, SEAT, depth).replace(f'{DATE}-', f'{DATE}-siraya85a_', 1)}.stl"
        part.export(stl)
        name = SP.name_of(tip, lay, SEAT, depth, 2, 30)
        ns = argparse.Namespace(seat=SEAT, no_slice=a.no_slice)
        pk = json.loads(json.dumps(pockets, default=float))
        rep = SP.write_and_check(part, pk, gaps, mesh, name, tip, lay, depth, 2, 30, body, five_layout(lay), ns)
        rep["stl"] = str(stl.relative_to(REPO))
        rep["seat_status"] = "the owner's seat (2026-10-18: 2.9 mm)"
        (PRINT / f"{name}.json").write_text(json.dumps(rep, indent=1, default=float))
        rec[TIP_KEY[tip]] = dict(tip=tip, tried=rows, chosen=lay, depth_mm=depth, record=tip_record(rep, stl))
    dst = V2 / f"{DATE}-v2_tips.json"
    dst.write_text(json.dumps(dict(seat_mm=SEAT, wall_rule_mm=MP.WALL_MIN, tips=rec, when=time.strftime("%Y-%m-%d %H:%M")),
                              indent=1, default=float) + "\n")
    print("->", dst.relative_to(REPO), flush=True)


# ----------------------------------------------------------------------------------------------- concept a, v2
PARAMS_A = STUDY / "params/20261018-v2_concept_a.json"
A_DIR = V2 / "a"
A_WORK = REPO / "logs/20261018-dip_link/v2/a"


def setup_a():
    """Concept a's generators at v2's parameter file, writing into v2/a/ (work files under logs/)."""
    import dip_link_params as DP
    concept, vals, des, doc = DP.load(PARAMS_A)
    DP.apply_a(vals, des)
    DP.redirect_a(A_DIR, doc["outputs"]["prefix"], A_WORK)
    A, DLF = DP.modules_a()
    return A, DLF, vals, des, doc


def a_step():
    """Exact STEP of the printed parts in freecadcmd (dip_link_params freecad_step_a: dip_link_board_mount.freecad_step)."""
    import subprocess
    env = dict(os.environ, DLP_CMD="step", DLP_PARAMS=str(PARAMS_A), DLP_OUT=str(A_DIR), DLP_WORK=str(A_WORK), DLP_REPO=str(REPO))
    r = subprocess.run(["freecadcmd", str(REPO / "scripts/dip_link_params.py")], env=env, cwd=str(REPO), capture_output=True,
                       text=True)
    print("\n".join(l[:300] for l in r.stdout.splitlines() if l.strip() and "%" not in l and "*" not in l)[-3000:], flush=True)
    if r.returncode or "valid" not in r.stdout:
        print(r.stderr[-3000:])
        raise SystemExit("freecadcmd failed")


V2_LABELS = {"dip_link": "DIP link, 8.5 mm longer, slotted for J1", "clamp_plate": "Clamp plate", "board": "Board, whole tab",
             "qwiic": "J1, plug and cable start", "fasteners": "M2 screws", "tip": "Tip, 85A", "servo": "PIP servo",
             "mcp": "MCP link"}


def a_render():
    """Assembled (dorsal side, PIP 0, with the MCP link and the PIP servo), exploded, the DIP link from the palmar side and
    the connector close up, through dip_link_concept_a's MuJoCo renderer; labels at the lengthened tip's positions."""
    A, DLF, vals, des, doc = setup_a()
    C_ = A.COLORS
    L, P = A.levels(), A.P
    f = lambda s_: str(DLF.fname(s_))                     # noqa: E731
    ctx = lambda s_: str(A_WORK / f"{s_}.stl")            # noqa: E731
    tip = P["asm_tip"].replace(".", "p")
    qcol = "0.95 0.95 0.92 1"
    asm = [("dip_link", f("dip_link"), C_["dip_link"], (0, 0, 0)), ("clamp_plate", f("clamp_plate"), C_["clamp_plate"], (0, 0, 0)),
           ("board", f("board"), C_["board"], (0, 0, 0)), ("fasteners", f("fasteners"), C_["fasteners"], (0, 0, 0)),
           ("qwiic", f("qwiic"), qcol, (0, 0, 0)), ("tip", f(f"tip_{tip}_F"), C_["tip"], (0, 0, 0)),
           ("servo", ctx("context_servo_pip"), C_["servo"], (0, 0, 0)), ("mcp", ctx("context_mcp_link"), C_["mcp"], (0, 0, 0))]
    routed = bool(P.get("cable_route")) and DLF.fname("cable").exists()
    ccol = "0.20 0.35 0.85 1"                              # the Qwiic cable's blue
    if routed:
        asm.insert(5, ("cable", f("cable"), ccol, (0, 0, 0)))
    MED = A_DIR / "media"
    pf = DLF.PFX
    jb = {n: (np.array(lo), np.array(hi)) for n, lo, hi in A.j1_boxes_F()}
    cam = ((5.0, 0.0, A.Z_C + 5.0), 150, 228, -20)
    img = A._render(asm, cam, MED / f"{pf}_iso.png")
    DLF._annotate_final(img, cam, [
        ("Clamp plate", (L["x_c1"], -3.0, A.Z_C - 2.0), (0.03, 0.62)),
        ("M2 screw into a shoulder (1.8 mm hole)", (L["x_c1"] + P["m2_head_k"], -P["m2_y"], P["m2_z"]), (0.03, 0.44)),
        ("J1 with its plug, on the whole tab", ((jb["plug"][0][0] + jb["plug"][1][0]) / 2, 0.0, jb["plug"][1][2]), (0.97, 0.36)),
        (f"DIP link {vals['neck_len']:g} mm longer than the final", (L["x_cf"], 12.0, A.Z_MOUNT + 4.0), (0.97, 0.55)),
        ("Tip, 85A, five magnets", (-2.0, A.Y_HALF, A.Z_C - 2.0), (0.97, 0.78)),
        ("PIP servo", (21.95, -6.0, -52.0), (0.03, 0.20))],
        title="Concept a, v2, assembled at PIP 0 deg, dorsal side")
    ex = [("dip_link", f("dip_link"), C_["dip_link"], (0, 0, 0)),
          ("cs", ctx("fasteners_cs_screw"), C_["fasteners"], (-12, 0, 0)),
          ("tip", f(f"tip_{tip}_F"), C_["tip"], (0, 0, -18)),
          ("board", f("board"), C_["board"], (9, 0, 0)), ("qwiic", f("qwiic"), qcol, (9, 0, 0)),
          ("clamp_plate", f("clamp_plate"), C_["clamp_plate"], (18, 0, 0)),
          ("m2", ctx("fasteners_m2_screw"), C_["fasteners"], (27, 0, 0))]
    if routed:
        ex.insert(5, ("cable", f("cable"), ccol, (9, 0, 0)))
    cam = ((9.0, 0.0, A.Z_C), 165, 110, -12)
    img = A._render(ex, cam, MED / f"{pf}_exploded.png")
    hd = sorted([(y, z) for y, z, _ in A.board_items()["holes"] if z < A.Z_C])
    DLF._annotate_final(img, cam, [
        ("DIP link: window plate, keys, the slot for J1", (L["x_cf"], -6.0, A.Z_C - 5.0), None),
        ("M2 x 4 countersunk screws from the palmar side", (A.X_BACK - 12 + 0.3, hd[0][0], hd[0][1]), None),
        ("Tip with two key channels", (-1.0, -A.Y_HALF, A.Z_C - 2.0 - 18), None),
        ("Board with J1, chips toward the tip", (L["x_bb"] + 9, -6.0, A.Z_C - 6.0), None),
        ("Clamp plate: 1.8 mm holes, the screws form their thread", (L["x_c1"] + 18, -6.0, A.Z_C - 6.0), None),
        ("M2 x 5 screws into the shoulders", (L["x_c1"] + 27 + P["m2_head_k"], -P["m2_y"], P["m2_z"]), None)],
        title="Concept a, v2, exploded", callout=True)
    cam = ((4.0, 0.0, A.Z_C + 3.0), 120, 25, 28)
    img = A._render([("dip_link", f("dip_link"), C_["dip_link"], (0, 0, 0))], cam, MED / f"{pf}_dip_link_palmar.png")
    DLF._annotate_final(img, cam, [
        ("Bridge: the tip's mount face rests here", (-6.0, -12.0, A.Z_MOUNT), None),
        ("Dovetail keys with detents", (A.X_BACK - P["key_depth"], -P["key_yc"], A.Z_MOUNT - 3.7), None),
        ("Slot for J1, its plug and the cable", ((jb["j1_header"][0][0] + jb["j1_header"][1][0]) / 2, 0.0,
                                                 jb["j1_header"][1][2]), None),
        ("U1 window", (A.X_BACK, 0.0, A.Z_C - 1.2), None)],
        title="Concept a, v2: DIP link from the palmar side", callout=True)
    if routed:                                             # the cable's route, seen from the slotted arm's side
        seg = A.cable_segments()
        _, b1, b2 = seg[2]
        _, b3, A_ = seg[4]
        cam = ((6.0, 12.0, -60.0), 95, 250, -14)
        img = A._render([m for m in asm if m[0] not in ("servo", "mcp")] +
                        [("servo", ctx("context_servo_pip"), "0.25 0.25 0.27 0.35", (0, 0, 0)),
                         ("mcp", ctx("context_mcp_link"), "0.62 0.71 0.81 0.30", (0, 0, 0))], cam, MED / f"{pf}_cable_route.png")
        DLF._annotate_final(img, cam, [
            ("Plug in J1; the cable turns toward the arm", (b1[0], b1[1], b1[2]), (0.03, 0.90)),
            ("Ledge along the bridge's dorsal corner, open to the back and the top", ((b1[0] + b2[0]) / 2, 9.0, b1[2]),
             (0.03, 0.62)),
            ("Turn into the slotted arm's outer face", (b3[0], b3[1], b3[2]), (0.97, 0.74)),
            (f"Anchor {A.PIP[2] - A_[2]:.1f} mm below the PIP axis", (A_[0], A_[1], A_[2]), (0.97, 0.45)),
            ("PIP axis", (0.0, A.ARM_OUT, A.PIP[2]), (0.97, 0.22))],
            title="Concept a, v2: the Qwiic cable's route over the plug and up the slotted arm", callout=False)
        a_cable_section(A, DLF, vals)
    print("renders ->", MED.relative_to(REPO), flush=True)


def cable_span(A=None, q_step=0.5, anchor=None):
    """Length of the cable's free span from its anchor on the slotted arm (concept a's, or `anchor`) to a tie on the MCP
    link's side plate (the plate at y 13.4..17.4 begins at z -34.4 at PIP 0): the tie at (0, z -33.0) of the MCP link, over
    PIP -18..92 deg."""
    import dip_link_geometry as G
    A_ = np.asarray(anchor, float) if anchor is not None else A.cable_segments()[4][2]
    B = np.array([[0.0, A_[1], -33.0]])
    qs = np.arange(-18.0, 92.0 + 1e-9, q_step)
    Ls = np.array([float(np.linalg.norm((A_ - G.rot_pip(B, q)[0])[[0, 2]])) for q in qs])
    return dict(anchor_F=np.round(A_, 2).tolist(), tie_on_mcp_link_F_at_pip0=B[0].round(2).tolist(),
                span_mm=dict(min=round(float(Ls.min()), 2), max=round(float(Ls.max()), 2),
                             at_pip_minus18=round(float(Ls[0]), 2), at_pip0=round(float(Ls[int(18 / q_step)]), 2),
                             at_pip92=round(float(Ls[-1]), 2), longest_at_pip_deg=float(qs[int(Ls.argmax())])),
                note="laid with the longest span plus 1 mm, the cable never pulls on the joint; between the anchor and the tie "
                     "it lies outside y 17.40, where no part of the MCP link or the PIP servo reaches at any PIP angle")


def save_panels(fig, axs, dst):
    """Each axes of fig saved on its own, with its title, tick and axis labels, as <dst stem>_panel<k>.png beside dst (for
    pages whose column is narrower than the whole figure)."""
    for lg in fig.legends:                       # the figure's legend lies under both panels; the page's caption names it
        lg.set_visible(False)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for k, ax in enumerate(axs, 1):
        bb = ax.get_tightbbox(r).transformed(fig.dpi_scale_trans.inverted()).padded(0.08)
        fig.savefig(dst.with_name(f"{dst.stem}_panel{k}.png"), dpi=100, bbox_inches=bb)
    for lg in fig.legends:
        lg.set_visible(True)


def a_cable_section(A, DLF, vals):
    """Side views of the cable's route: left, the x-z section through the groove on the slotted arm's outer face with the
    MCP link and the PIP servo at four PIP angles; right, the y-z section through the groove over the plug, the same."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    from matplotlib.patches import Patch, Polygon as MPoly
    import dip_link_geometry as G
    pf = DLF.PFX
    D = trimesh.load(DLF.fname("dip_link"))
    Bd = trimesh.load(DLF.fname("board"))
    Q = trimesh.load(DLF.fname("qwiic"))
    ctx = {n: G.load_part(n) for n in ("mcp_link", "servo_pip")}
    C = A.cable_polyline()
    r = A.P["cable_d"] / 2
    seg = A.cable_segments()
    qcol = {-18.0: "#c0392b", 0.0: "#2c3e50", 45.0: "#16a085", 92.0: "#8e44ad"}
    plt.rcParams.update({"font.family": "serif", "font.size": 15})
    fig, axs = plt.subplots(1, 2, figsize=(17.0, 10.0))
    views = [(axs[0], (0, 1, 0), A.P["cable_face_y"] - 0.6, (0, 2), "x in F (mm), dorsal to the right",
              f"Section y = {A.P['cable_face_y'] - 0.6:.1f} mm, in the slotted arm's face groove", (-24, 24), (-72, -26)),
             (axs[1], (1, 0, 0), seg[2][1][0], (1, 2), "y in F (mm), toward the slotted arm to the right",
              f"Section x = {seg[2][1][0]:.1f} mm, through the groove over the plug", (-6, 22), (-74, -48))]
    for ax, nrm, at, ij, xl, title, xlim, ylim in views:
        o = np.zeros(3)
        o[int(np.argmax(nrm))] = at
        for tm, fc in ((D, "#d6d6cc"), (Bd, "#1f7a3a"), (Q, "#f2f2ea")):
            s = tm.section(plane_origin=o, plane_normal=nrm)
            if s is not None:
                for poly in s.discrete:
                    ax.add_patch(MPoly(poly[:, list(ij)], closed=True, fc=fc, ec="k", lw=0.5))
        for q, c in qcol.items():
            for n, tm in ctx.items():
                t2 = tm.copy()
                t2.vertices = G.rot_pip(np.asarray(t2.vertices), q)
                s = t2.section(plane_origin=o, plane_normal=nrm)
                if s is not None:
                    for poly in s.discrete:
                        ax.add_patch(MPoly(poly[:, list(ij)], closed=True, fc="none", ec=c, lw=1.3))
        ax.plot(C[:, ij[0]], C[:, ij[1]], "-", color="#2f5bd8", lw=9, alpha=0.55, solid_capstyle="round")
        ax.plot(C[:, ij[0]], C[:, ij[1]], "-", color="#1b2f80", lw=1.0)
        if ij == (0, 2):
            ax.plot([0], [A.PIP[2]], "k+", ms=14, mew=2)
            A_ = seg[4][2]
            ax.annotate(f"Anchor, {A.PIP[2] - A_[2]:.1f} mm below the PIP axis", xy=(A_[0], A_[2]), xytext=(-22, -64),
                        fontsize=14, arrowprops=dict(arrowstyle="->", lw=1.0))
            ax.annotate("PIP axis", xy=(0.0, A.PIP[2]), xytext=(8, -44), fontsize=14, arrowprops=dict(arrowstyle="->", lw=1.0))
            ax.annotate("MCP link's side plate", xy=(0.0, -34.4), xytext=(-22, -30), fontsize=14,
                        arrowprops=dict(arrowstyle="->", lw=1.0))
        else:
            ax.annotate("Plug in J1", xy=(seg[0][1][1], seg[0][1][2] - 1.0), xytext=(-5, -72), fontsize=14,
                        arrowprops=dict(arrowstyle="->", lw=1.0))
            ax.annotate("Slotted arm", xy=(15.4, -60.0), xytext=(13, -50), fontsize=14, arrowprops=dict(arrowstyle="->", lw=1.0))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.set_xlabel(xl)
        ax.set_ylabel("z in F (mm)")
        ax.set_title(title, fontsize=15)
    fig.legend([Patch(fc="#d6d6cc", ec="k"), Patch(fc="#2f5bd8", alpha=0.55)] + [Patch(fc="none", ec=c, lw=1.3)
                                                                                for c in qcol.values()],
               ["DIP link", f"Cable, {A.P['cable_d']:g} mm bundle"] + [f"MCP link and PIP servo at PIP {q:+.0f} deg"
                                                                     for q in qcol],
               loc="lower center", ncol=3, frameon=False, fontsize=14)
    fig.suptitle(f"Concept a, v2: the Qwiic cable's route (DIP link {vals['neck_len']:g} mm longer)", fontsize=18)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    dst = A_DIR / "media" / f"{pf}_cable_section.png"
    fig.savefig(dst, dpi=100)
    save_panels(fig, axs, dst)
    plt.close(fig)
    print(dst.relative_to(REPO), flush=True)


def a_section():
    """Section y = 0 through U1 and J1 at PIP 0 and at full flexion (92 deg): the stack, J1 with its plug and cable in the
    bridge's slot, the tip with its magnets, and the MCP link and the PIP servo in outline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    from matplotlib.patches import Polygon as MPoly
    import dip_link_geometry as G
    A, DLF, vals, des, doc = setup_a()
    pf = DLF.PFX
    tip = A.P["asm_tip"].replace(".", "p")
    parts = [("Tip, 85A (11/22, five magnets)", DLF.fname(f"tip_{tip}_F"), "#e8c88a"),
             ("DIP link", DLF.fname("dip_link"), "#d6d6cc"), ("Board", DLF.fname("board"), "#1f7a3a"),
             ("Clamp plate", DLF.fname("clamp_plate"), "#5b8fd9"),
             ("J1, plug" + ("" if A.P.get("cable_route") else ", cable start"), DLF.fname("qwiic"), "#f2f2ea"),
             ("Screws", DLF.fname("fasteners"), "#c8a33c")]
    if A.P.get("cable_route") and DLF.fname("cable").exists():
        parts.append(("Qwiic cable (2.0 mm bundle)", DLF.fname("cable"), "#2f5bd8"))
    meshes = [(n, trimesh.load(p_), c) for n, p_, c in parts]
    ctx = {n: G.load_part(n) for n in ("mcp_link", "servo_pip")}
    plt.rcParams.update({"font.family": "serif", "font.size": 15})
    fig, axs = plt.subplots(1, 2, figsize=(17.0, 11.0))
    for ax, q in zip(axs, (0.0, 92.0)):
        for n, tm, c in meshes:
            sec = tm.section(plane_origin=(0, 0.0, 0), plane_normal=(0, 1, 0))
            if sec is None:
                continue
            for poly in sec.discrete:
                ax.add_patch(MPoly(poly[:, [0, 2]], closed=True, fc=c, ec="k", lw=0.5))
        for n, tm in ctx.items():
            t2 = tm.copy()
            t2.vertices = G.rot_pip(np.asarray(t2.vertices), q)          # parent-fixed points seen from the DIP link
            sec = t2.section(plane_origin=(0, 0.0, 0), plane_normal=(0, 1, 0))
            if sec is None:
                continue
            for poly in sec.discrete:
                ax.add_patch(MPoly(poly[:, [0, 2]], closed=True, fc="none", ec="#3a6fb0" if n == "mcp_link" else "#b04a2a",
                                   lw=1.2))
        for m in A.spec()["tips"][A.P["asm_tip"]]["magnets"]["items"]:
            if abs(m["centre_F"][1]) < 1.6:
                (x0, x1), z = m["x_F"], m["centre_F"][2]
                ax.add_patch(MPoly([[x0, z - 1.5], [x1, z - 1.5], [x1, z + 1.5], [x0, z + 1.5]], closed=True, fc="#b03a2e",
                                   ec="k", lw=0.5))
        ax.plot([0], [-51.55], "k+", ms=14, mew=2)
        ax.set_xlim(-14, 24)
        ax.set_ylim(A.Z_END - 2.0, -40.0)
        ax.set_aspect("equal")
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.set_xlabel("x in F (mm), dorsal to the right")
        ax.set_ylabel("z in F (mm)")
        ax.set_title(f"PIP {q:.0f} deg: MCP link (blue) and PIP servo (red) in outline", fontsize=16)
    jb = {n: (np.array(lo), np.array(hi)) for n, lo, hi in A.j1_boxes_F()}
    axs[0].annotate("J1 on the tab, in the bridge's slot", xy=((jb["j1_header"][0][0] + jb["j1_header"][1][0]) / 2,
                                                              (jb["j1_header"][0][2] + jb["j1_header"][1][2]) / 2),
                    xytext=(9.0, A.Z_C + 2.0), fontsize=15, arrowprops=dict(arrowstyle="->", lw=1.0))
    if "wire1" in jb:
        axs[0].annotate("Plug and the cable's\nfirst 3 mm", xy=(5.9, jb["wire1"][1][2] - 0.5), xytext=(10.0, -46.5),
                        fontsize=15, arrowprops=dict(arrowstyle="->", lw=1.0))
    else:
        axs[0].annotate("Plug; the cable turns\ntoward the slotted arm", xy=(5.9, jb["plug"][1][2] + 0.6),
                        xytext=(10.0, -46.5), fontsize=15, arrowprops=dict(arrowstyle="->", lw=1.0))
    axs[0].annotate("PIP axis", xy=(0.0, -51.55), xytext=(-12.5, -43.5), fontsize=15, arrowprops=dict(arrowstyle="->", lw=1.0))
    from matplotlib.patches import Patch
    fig.legend([Patch(fc=c, ec="k") for _, _, c in parts] + [Patch(fc="#b03a2e", ec="k")],
               [n for n, _, _ in parts] + ["Magnets (y = 0)"], loc="lower center", ncol=4, frameon=False, fontsize=15)
    fig.suptitle("Concept a, v2: section y = 0 through U1 and J1", fontsize=18)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    dst = A_DIR / "media" / f"{pf}_section.png"
    fig.savefig(dst, dpi=100)
    plt.close(fig)
    print(dst.relative_to(REPO), flush=True)


def print_tips_a(A, DLF, out, pf, work):
    """Each tip of a's design (A.spec()["tips"]) with the two key channels cut and its magnet pauses: the print STL that
    dip_link_board_mount.cmd_build wrote to DLF.PRINT, sliced with its sheet's pockets (sim2real_fingertip_prints
    write_and_check) into out/."""
    import trimesh
    import metric_magnet_prints as MP
    import sim2real_fingertip_prints as SP
    MP.DATE = SP.DATE = DATE
    SP.OUT, SP.WORK = Path(out), Path(work)
    SP.use_template()
    for key, t in A.spec()["tips"].items():
        stl = DLF.PRINT / f"{pf}_tip_{key.replace('.', 'p')}_print.stl"
        if "sheet" not in t or not stl.exists():
            continue
        sheet = json.loads((REPO / t["sheet"]).read_text())
        part = trimesh.load(stl, force="mesh")
        name = f"{pf}_print_tip_{key.replace('.', 'p')}_channels_85a_t2_g30"
        ns = argparse.Namespace(seat=sheet.get("seat_mm", SEAT), no_slice=False)
        rep = SP.write_and_check(part, json.loads(json.dumps(sheet["pockets"])), sheet["pocket_gaps"], sheet["mesh"], name,
                                 sheet["tip"], sheet["layout"], sheet["magnet"]["top_depth_mm"], 2, 30, part,
                                 five_layout(sheet["layout"]), ns)
        rep["stl"] = str(stl.relative_to(REPO)) if REPO in stl.parents else str(stl)
        rep["note"] = "the sheet's tip with the two key channels of concept a cut (dip_link_concept_a.tip_cutter)"
        (Path(out) / f"{name}.json").write_text(json.dumps(rep, indent=1, default=float))


def a_print():
    """The PLA parts on one plate (dip_link_print_project.build_project: the link with its support enforcers and shoulder
    modifiers, the clamp plate, the pin-fit coupon), the attachment fit coupon at v2's keys, and each channelled tip with
    its pauses (sim2real_fingertip_prints.write_and_check on the sheet's pockets), sliced and checked."""
    import trimesh
    import dip_link_print_project as PP
    import metric_magnet_prints as MP
    import sim2real_fingertip_prints as SP
    A, DLF, vals, des, doc = setup_a()
    pf = DLF.PFX
    out = A_DIR / "print"
    # the print project's window-plate box is the final's (z -88.71..-67.30): moved with the tip
    box0 = PP.box_F_to_print

    def box_shifted(lo, hi, Rt):
        if abs(lo[2] + 88.71) < 1e-9 and abs(hi[2] + 67.30) < 1e-9:
            lo, hi = (lo[0], lo[1], A.Z_END), (hi[0], hi[1], A.Z_MOUNT)
        return box0(lo, hi, Rt)
    PP.box_F_to_print = box_shifted
    PP.PFX = pf
    a_bambu(A, DLF, out)
    PP.build_fit_coupon(out)
    print_tips_a(A, DLF, out, pf, A_WORK / "print")
    print("print ->", out.relative_to(REPO), flush=True)


def a_bambu(A, DLF, out):
    """a's DIP link (tessellated from the step stage's exact fused solid when it exists), clamp plate and pin-fit coupon as
    print-oriented PLA STLs for Bambu Studio (owner, 2026-10-19 14:15); the XL project of these parts is removed."""
    pf = DLF.PFX
    for stale in out.glob(f"{pf}_print_link_clamp_coupon_pla_t3*"):
        stale.unlink()
    exact = A_WORK / f"{pf}_dip_link.step"           # the fused exact solid of the step stage, tessellated: one closed body
    src = step_to_stl(exact, A_WORK / f"{pf}_dip_link_exact.stl") if exact.exists() else DLF.fname("dip_link")
    a_bambu_files(src, DLF.fname("clamp_plate"), out / f"{pf}_pin_fit_coupon.stl", out, pf, str(PARAMS_A.relative_to(REPO)))


def a_bambu_files(link_src, clamp_src, pin_src, out, pf, params):
    """Concept a's PLA parts as Bambu STLs into out/ with <pf>_pla_bambu.json: their settings and, for the link, the support
    regions and the shoulders' 100 % infill boxes of the XL project (dip_link_print_project.link_volumes) in the STL's own
    coordinates. Needs dip_link_concept_a at the design's parameters (dip_link_params.apply_a)."""
    import trimesh
    import dip_link_print_project as PP
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(printer="Bambu Lab P1S through Bambu Studio, PLA", params=params, parts={}, when=time.strftime("%Y-%m-%d %H:%M"))
    link = trimesh.load(link_src, force="mesh")
    rec["parts"]["dip_link"] = bambu_stl(link, PP.R_LINK, out / f"{pf}_dip_link_pla_bambu.stl")
    rec["parts"]["dip_link"]["source"] = str(Path(link_src).relative_to(REPO)) if REPO in Path(link_src).parents else str(link_src)
    m, Rt = PP.to_print(link, PP.R_LINK)
    enf, mods = PP.link_volumes(m, Rt)
    rec["parts"]["clamp_plate"] = bambu_stl(trimesh.load(clamp_src, force="mesh"), PP.R_CLAMP,
                                            out / f"{pf}_clamp_plate_pla_bambu.stl")
    if pin_src and Path(pin_src).exists():
        rec["parts"]["pin_fit_coupon"] = bambu_stl(trimesh.load(pin_src, force="mesh"), None,
                                                   out / f"{pf}_pin_fit_coupon_pla_bambu.stl")
    rec["settings"] = {
        "dip_link": dict(BAMBU_PLA["dip_link"], supports="tree supports on the build plate only, painted (or enforced) in the "
                         "three boxes below: under both clevis arms' ends and under the bridge's face toward the PIP axis; none "
                         "in the chip windows, the J1 slot or the cable groove",
                         support_boxes_mm={n: [np.round(lo, 2).tolist(), np.round(hi, 2).tolist()] for n, lo, hi in enf},
                         infill_100pct_boxes_mm={n: [np.round(lo, 2).tolist(), np.round(hi, 2).tolist()] for n, lo, hi in mods},
                         note="boxes in the STL's coordinates (X, Y centred, Z from the plate); the shoulders take the M2 "
                              "screws' 1.8 mm holes, so a 100 % rectilinear infill modifier over each"),
        "clamp_plate": dict(orientation="on its back face (X = z_F, Y = y_F, Z = -x_F)", layer_height_mm=0.1, walls=4,
                            infill="100 % rectilinear", supports="none", brim="none", ironing="top surfaces"),
        "pin_fit_coupon": dict(orientation="as modelled, pins up", layer_height_mm=0.1, walls=4, infill="100 % rectilinear",
                               supports="none", brim="none")}
    for k, v in rec["parts"].items():
        print(f"bambu {k}: {v}", flush=True)
    (out / f"{pf}_pla_bambu.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    return rec


def a_docs():
    """v2/a/evaluation.json: the lengthening, the Qwiic connection, clearance, sensing, walls, mass, the print projects,
    the parts list and the assembly sequence, from the stage outputs."""
    A, DLF, vals, des, doc = setup_a()
    pf = DLF.PFX
    rd = lambda n: json.loads((A_DIR / f"{pf}_{n}.json").read_text())       # noqa: E731
    chk, sen, parts = rd("check"), rd("sense"), rd("parts")
    step = rd("step") if (A_DIR / f"{pf}_step.json").exists() else {}
    lg = json.loads((V2 / f"{DATE}-v2_lengthening.json").read_text())
    tips = json.loads((V2 / f"{DATE}-v2_tips.json").read_text())
    prj = {f.stem: json.loads(f.read_text()) for f in sorted((A_DIR / "print").glob("*.json"))}
    four = next(r for r in sen["rows"] if r["layout"] == "five_z7p4" and r["recipe"] == "w85_30" and r["case"] == "design")
    clear = chk["clearance"]
    L = A.levels()
    routed = bool(A.P.get("cable_route"))
    pip_to_tip = round(37.16 + vals["neck_len"], 2)
    span = cable_span(A) if routed else None
    ev = dict(
        title=(f"DIP link concept a, v2: the board's own Qwiic connector kept on a DIP link {vals['neck_len']:g} mm longer"
               + (", the Qwiic cable routed out past the slotted arm" if routed else "") + ", M2 self-forming screws, "
               "five-magnet tips"),
        date="2026-10-19" if routed else "2026-10-18", params=str(PARAMS_A.relative_to(REPO)),
        generator="scripts/dip_link_v2.py a (dip_link_params.apply_a on scripts/dip_link_concept_a.py and "
                  "scripts/dip_link_board_mount.py)",
        owner=["'we should just go for 5x 3x2 magnets'", "'i do like the half ellipsoids most, the 11/22 one especially'",
               "'if we have to lengthen the fingertip to get everything to fit, including the qwiic port and the cable, that's "
               "fine'", "'for m3 screws, i use 2.75mm diameter holes, m2 i use 1.8'"]
              + (["2026-10-19: 'how are you planning to route the cable out though?'; 'i would like to minimize range of "
                  "motion loss, if any, on the back extension of the DIP joint'"] if routed else []),
        changes_from_final=[
            f"DIP link {vals['neck_len']:g} mm longer between its bridge and the tip (PIP to the tip's end {pip_to_tip} mm, "
            "37.16 mm in the final); the clevis arms and the bridge's top are the final's"
            + ("; the slotted arm loses its lower dorsal corner and gains a groove on its outer face for the cable, and the "
               "bridge's dorsal-top corner between the J1 slot and the arm is cut away over the cable" if routed else ""),
            "the board keeps its whole 9 mm tab and its Qwiic connector J1, tab proximal; the bridge has a slot for J1, the "
            "plug and the cable; no pigtail, no header on the clamp plate",
            "the two distal board screws are M2 x 4 countersunk (DIN 965), forming their thread in 1.8 mm holes of the clamp "
            "plate (no nuts); the two ear screws M2 x 5 into 1.8 mm holes of the shoulders (1.7 mm in the final)",
            "tips with five 3 x 2 mm magnets at the 2.9 mm seat: the 11/22 half-ellipsoid (corners y +-2.5 mm, rows 9.8 mm "
            "apart, the group 0.5 mm toward the mount face) and the block (five_z7p4), both with the key channels"],
        lengthening=lg["placements"]["a"] | dict(choice=lg["choice"]["a"]) | (dict(
            cable_route=f"{vals['neck_len']:g} mm: the routed cable needs 1.0 mm from the MCP link over PIP -18..92 deg; with a "
                        "2.0 mm bundle and 2.0 mm bends the groove over the plug cleared 0.14 mm at 8.5, 1.05 mm at 9.5 and 1.51 "
                        "mm at 10.0 mm (logs/20261019-dip_link_cable/route_rings.log); 9.75 mm is concept b's lengthening, so "
                        "both designs give one finger length") if routed else {}),
        cable=(dict(route="out of the plug toward the PIP joint, a 2.0 mm turn to +y over the plug, a groove along the DIP "
                          "link's dorsal face (open to the back, so the cable is laid in with its far plug on), through the "
                          "slotted arm's lower dorsal corner, a 2.0 mm turn up into the arm's outer face, a groove half the "
                          "bundle deep up the face to an anchor below the PIP axis. Between the J1 slot and the arm the groove "
                          "is open to the top as well: the bridge's arc lies 0.07-1.4 mm above it over x 4.70-7.30, too thin "
                          "for a roof, so the cable lies on a ledge of the bridge's dorsal corner",
                    bundle_d_mm=A.P["cable_d"], bend_r_mm=A.P["cable_r"], groove_clearance_mm=A.P["cable_clear"],
                    length_plug_to_anchor_mm=round(float(np.linalg.norm(np.diff(A.cable_polyline(), axis=0), axis=1).sum()), 2),
                    clearance=clear.get("qwiic_cable"), free_span=span,
                    range_of_motion="none lost: the cable runs inside the DIP link or outside y 17.40 to its anchor, and the "
                                    "free span to the MCP link is longest at the PIP angle given, so a cable laid at that "
                                    "length never pulls on the joint at -18 deg (back extension) or 92 deg")
               if routed else None),
        clearance=dict(per_part={k: dict(min_mm=v.get("min_mm"), pip_deg=v.get("pip_deg"), part=v.get("part"))
                                 for k, v in clear.items()},
                       requirement="R7: 0.5 mm rigid, 1.0 mm cable; exact method of the specification over PIP -18..92 deg; "
                                   "values of 3 mm or more are lower bounds (only points within 3-4 mm of the links are turned)"),
        to_pip_servo=chk["to_pip_servo"], walls=chk["walls"], tip_walls_near_channels=chk["tip_walls_near_channels"],
        channels_to_magnets_R4=chk["channels_to_magnets_R4"], overlap_mm3=chk["overlap_mm3"], mass_g=chk["mass_g"],
        width_over_tip_length_y=chk["width_over_tip_length_y"],
        sensing=dict(block5_force_resolution_30_N=four["centre_force_N"],
                     block5_force_resolution_50_N=next(r["centre_force_N"] for r in sen["rows"] if r["layout"] == "five_z7p4"
                                                       and r["recipe"] == "w85_50" and r["case"] == "design"),
                     gap_mm=round(L["gap"], 3),
                     static_fields={k: dict(B_U1_mT=v["design"]["B_U1_mT"], B_max_component_mT=v["design"]["B_max_component_mT"],
                                            gain_sel=v["design"]["gain_sel"],
                                            chip_to_magnet_surface_mm=min(v["design"]["chip_to_magnet_surface_mm"]))
                                    for k, v in sen["static_fields"].items()},
                     note="force resolution from the readiness finite elements of the five-magnet block (corners moved to the "
                          "printed 7.4 mm); the five-magnet 11/22 has no finite elements yet, so only its static field, gain "
                          "and chip-to-magnet distance"),
        tips={k: dict(layout=v["chosen"], depth_mm=v["depth_mm"], sheet=v["record"]["sheet"]) for k, v in tips["tips"].items()},
        step=step, parts=parts["parts"],
        print_projects={k: dict(project=v.get("project"),
                                print_time=(v.get("slice") or {}).get("print_time") or (v.get("slice") or {}).get("check", {}).get("print_time"),
                                filament_g=(v.get("slice") or {}).get("filament_g"),
                                pauses_mm=[q["print_z"] for q in v.get("pauses", [])], checks_pass=(v.get("slice") or {}).get("checks_pass"))
                        for k, v in prj.items() if not k.endswith("_pla_bambu")},
        print_pla_bambu=prj.get(f"{pf}_pla_bambu"),
        parts_list=[
            dict(item="DIP link, v2", material="PLA (Bambu Studio: print-oriented STL)", qty=1,
                 file=f"v2/a/print/{pf}_dip_link_pla_bambu.stl",
                 step=f"v2/a/cad/{pf}_dip_link_new_features.step (new features, exact) and {pf}_dip_link.step.zip (fused)"),
            dict(item="Clamp plate, v2 (1.8 mm holes, no nut pockets, no Qwiic pocket)", material="PLA, solid (Bambu Studio)",
                 qty=1, file=f"v2/a/print/{pf}_clamp_plate_pla_bambu.stl", step=f"v2/a/cad/{pf}_clamp_plate.step"),
            dict(item="Pin-fit coupon", material="as the clamp plate", qty=1, file=f"v2/a/print/{pf}_pin_fit_coupon_pla_bambu.stl"),
            dict(item="Attachment fit coupon: PLA key sections and 85A channel sections at 0, 0.05, 0.10 mm per flank", qty=1,
                 file=f"v2/a/print/{pf}_fit_coupon_keys_pla_t3.3mf and {pf}_fit_coupon_channels_85a_t2.3mf"),
            dict(item="85A tip with key channels and five magnets: 11/22 half-ellipsoid (or the block)", material="Siraya Tech "
                 "TPU 85A, 30 % gyroid", qty=1, file=f"v2/a/print/{pf}_print_tip_ell11_5_channels_85a_t2_g30.3mf"),
            dict(item="N52 disc magnet 3 x 2 mm, axially magnetized", qty=5),
            dict(item="ReSkin 5X board as supplied (J1 kept, tab not cut)", material="FR-4 0.6 mm", qty=1),
            dict(item="M2 x 4 countersunk screw, DIN 965 / ISO 7046", material="brass or stainless A2", qty=2),
            dict(item="M2 x 5 cheese-head screw, DIN 84", material="brass", qty=2),
            dict(item="Qwiic cable (JST SH 4-pin) from J1 to the controller", qty=1)],
        assembly=[
            "Print the pin-fit coupon and the attachment fit coupon first; choose the pin and the channel interference.",
            "Print the DIP link, the clamp plate and the pin coupon in PLA on the Bambu printer (print-oriented STLs, settings "
            "in v2/README.md) and the 85A tip with its pauses on the XL (press each magnet to the floor, north pole toward the "
            "palmar side).",
            "Plug the Qwiic cable into J1 before mounting the board: the plug enters J1 along the tab, from its end.",
            "Lay the board component side down on the window plate: chips in their windows, J1 and the plug in the bridge's "
            "slot, the cable out of the slot's top toward the PIP joint.",
            "Put on the clamp plate (pins through the board's proximal holes into the window plate).",
            "With the tip off, drive the two M2 x 4 countersunk screws from the palmar side through the window plate and the "
            "board's distal holes into the clamp plate's 1.8 mm holes; they form their thread; snug.",
            "Drive the two M2 x 5 screws through the clamp plate's ears into the shoulders' 1.8 mm holes; snug, the ears clear "
            "of the shoulders."]
        + (["Lay the cable into its groove from the dorsal side: bend it toward the slotted arm right behind the plug, press "
            "it along the ledge, around the arm's lower corner and up the groove on the arm's outer face to the anchor. A "
            "strip of Kapton tape along the open run keeps the cable on its ledge; a drop of CA glue or a strip of Kapton tape "
            "at the anchor keeps a pull off J1.",
            f"Tie the cable to the MCP link's side plate {span['span_mm']['max'] + 1.0:.0f} mm from the anchor (its longest "
            f"span, at PIP {span['span_mm']['longest_at_pip_deg']:.0f} deg, plus 1 mm), so it never pulls on the joint."] if routed else
           ["Route the cable from the slot's top along the link's dorsal side to y > 17.4 mm and across the PIP joint outside "
            "the slotted arm; fix it to the DIP link before the joint so a pull does not reach J1."])
        + ["Slide the tip along +z onto the keys until both detents engage; read the zero-load field."],
        first_print_checks=["J1 with the plug mated fits the slot without touching its walls; the cable leaves the slot "
                            "freely", "The M2 screws thread into the 1.8 mm holes by hand and hold the board without play",
                            "Over the full PIP range by hand, nothing touches the MCP link or the PIP servo"
                            + ((f" (the plug's nearest approach is {clear['qwiic_connector']['min_mm']} mm at PIP "
                                f"{clear['qwiic_connector']['pip_deg']:g} deg)" if clear["qwiic_connector"].get("pip_deg")
                                is not None else f" (the plug stays {clear['qwiic_connector']['min_mm']} mm or more from "
                                                 "both at every angle)") if clear.get("qwiic_connector") else "")]
                           + (["The cable stays in its groove over the full PIP range, back extension (-18 deg) included, "
                               "and the joint turns as freely with the cable tied as without it"] if routed else []),
        open_items=[f"Mass {chk['mass_g']['total_without_tip']} g without the tip (R12 7.6 g): the {vals['neck_len']:g} mm "
                    "longer bridge is solid between the clevis arms", "The window plate's web beside each distal M2 "
                    "countersink is 0.72 mm (R11 0.8 mm for two lines); M1.6 countersunk screws keep the final's 0.80 mm",
                    "Force resolution of the five-magnet 11/22 needs its finite elements"]
                   + (["The bundle is modelled as 2.0 mm round; measure the cable's four wires and the groove's fit on the "
                       "first print"] if routed else ["The cable's route past the PIP joint beyond its first 3 mm"]),
        when=time.strftime("%Y-%m-%d %H:%M"))
    (A_DIR / "evaluation.json").write_text(json.dumps(ev, indent=1, default=float) + "\n")
    print("->", (A_DIR / "evaluation.json").relative_to(REPO), flush=True)


# ----------------------------------------------------------------------------------------------- coupons for the owner's fits
COUPONS = V2 / "coupons"
SWEEP_FLANKS = (0.0, 0.05, 0.10, 0.15, 0.20)     # interference per flank of the 85A channel on the PLA key, mm
SWEEP_DETENTS = (0.15, 0.25, 0.35)               # detent height on the PLA key, mm
GAUGE_CLEAR = (0.15, 0.20, 0.30)                 # clearance of the window plate's chip windows, mm


def _notches_y(K, x0, x1, y0, z0, z1, n, pitch=1.4, w=0.6):
    return K.union([K.box((x0, y0 + pitch * j, z0), (x1, y0 + pitch * j + w, z1)) for j in range(n)])


def coupon_keys(A, K):
    """Concept a's PLA key with its detent at each SWEEP_DETENTS height, each on the key coupon's backing plate and stop
    flange (dip_link_print_project.fit_coupon_meshes), the flange's far end marked by 1, 2 or 3 notches."""
    import trimesh
    P = dict(A.P)
    yc, Zm, XB = P["key_yc"], A.Z_MOUNT, A.X_BACK
    units = []
    for k, h in enumerate(SWEEP_DETENTS):
        key = A.keys(K, dict(P, detent_h=h))[1]
        unit = K.union([key, K.box((XB - 0.01, yc - 4.5, Zm - 6.0), (XB + 2.0, yc + 4.5, Zm + 0.01)),
                        K.box((XB - 3.0, yc - 4.5, Zm), (XB + 2.0, yc + 4.5, Zm + 2.0))])
        unit = K.sub(unit, _notches_y(K, XB - 3.1, XB - 2.4, yc - 3.6, Zm + 1.3, Zm + 2.1, k + 1))
        units.append(A.to_trimesh(unit).apply_translation((0.0, 11.0 * k, 0.0)))
    return trimesh.util.concatenate(units)


def coupon_channels(A, K):
    """One 85A channel block per SWEEP_FLANKS value (the tip's back region with tip_cutter's channel and detent groove), its
    far end marked by 1 to 5 notches."""
    P = dict(A.P)
    yc, Zm, XB = P["key_yc"], A.Z_MOUNT, A.X_BACK
    out = []
    for k, fl in enumerate(SWEEP_FLANKS):
        blk = K.box((XB - 4.0, yc - 4.5, Zm - 7.0), (XB, yc + 4.5, Zm))
        cut = A.tip_cutter(K, dict(P, fit_flank=fl))
        out.append((fl, A.to_trimesh(K.sub(K.sub(blk, cut), _notches_y(K, XB - 4.1, XB - 3.4, yc - 3.6 - 0.7, Zm - 7.1,
                                                                         Zm - 6.4, k + 1)))))
    return out


def coupon_carrier(A, K, clear_chip=None, box=None):
    """Everything of concept a's DIP link that the board, J1, the plug, the cable and the clamp plate touch, without the
    clevis arms or the bridge's palmar half: the DIP link (cable groove included) cut to x -1..12, |y| <= 11 and z up to the
    bridge's flat (-63.10), at a chip-window clearance clear_chip."""
    P0 = dict(A.P)
    if clear_chip is not None:
        A.P["clear_chip"] = clear_chip
    try:
        D = A.dip_link(K)
    finally:
        A.P.update(P0)
    lo, hi = box or ((-1.0, -11.0, A.Z_END - 1.0), (12.0, 11.0, A.P["bridge_flat"]))
    return A.to_trimesh(K.inter(D, K.box(lo, hi)))


def coupon_gauge(A, K, clear_chip):
    """The window plate alone with its five chip windows, passive windows and keys, standing on a 2 mm stub of the bridge:
    the DIP link cut to x 4..7.4, |y| <= 11, z up to Z_MOUNT + 2."""
    return coupon_carrier(A, K, clear_chip, ((4.0, -11.0, A.Z_END - 1.0), (7.4, 11.0, A.Z_MOUNT + 2.0)))


def cmd_coupons(a):
    """v2/coupons/: the fits the owner calibrates before a whole DIP link is printed (owner, 2026-10-19: 'calibrate the
    magnetometer board fit and the snap fits, dont need to print the finger'). Concept a: (1) the snap-fit sweep, PLA keys
    at three detent heights (tool 3) and 85A channel blocks at five interferences (tool 2); (2) the board carrier at the
    design's chip clearance with a clamp plate; (3) window gauges at three chip clearances. Each project sliced and checked
    by dip_link_print_project.build_project."""
    import dip_link_print_project as PP
    A, DLF, vals, des, doc = setup_a()
    K = A.MeshK()
    COUPONS.mkdir(parents=True, exist_ok=True)
    pf = f"{DATE}-dip_link_v2a_coupon"
    PP.PFX = pf
    PP.COUPON_FLANKS = SWEEP_FLANKS
    rep = dict(owner="2026-10-19: 'calibrate the magnetometer board fit and the snap fits, dont need to print the finger'",
               params=str(PARAMS_A.relative_to(REPO)), projects={})
    # (1) snap-fit sweep
    keys = coupon_keys(A, K)
    stl = {"key_coupon": COUPONS / f"{pf}_keys_detents_pla.stl"}
    keys.export(stl["key_coupon"])
    PP.SETTINGS["key_coupon"] = dict(PP.PLA_OBJECT, perimeters="3", fill_density="100%", fill_pattern="rectilinear")
    PP.LABEL["key_coupon"] = "PLA keys, detents 0.15 / 0.25 / 0.35 mm (1 / 2 / 3 notches)"
    for fl, m in coupon_channels(A, K):
        k = f"channel_block_{fl:.2f}".replace(".", "p")
        stl[k] = COUPONS / f"{pf}_channel_{k[14:]}_85a.stl"
        m.export(stl[k])
        PP.SETTINGS[k] = dict(PP.SETTINGS["tip"])
        PP.LABEL[k] = f"85A channel block, {fl:.2f} mm per flank"
    r = PP.build_project({"key_coupon": str(stl["key_coupon"])}, COUPONS, f"{pf}_keys_detents_pla_t3",
                         source="scripts/dip_link_v2.py coupons")
    rep["projects"]["keys"] = r["project"]
    r = PP.build_project({k: str(v) for k, v in stl.items() if k.startswith("channel_block")}, COUPONS,
                         f"{pf}_channels_5_flanks_85a_t2", source="scripts/dip_link_v2.py coupons")
    rep["projects"]["channels"] = r["project"]
    # (2) board carrier with a clamp plate, printed as the DIP link prints (X = -x_F, Y = y_F, Z = -z_F)
    import trimesh
    car = coupon_carrier(A, K)
    car.vertices = np.asarray(car.vertices) @ PP.R_LINK.T
    stl_car = COUPONS / f"{pf}_board_carrier_clear0p20.stl"
    car.export(stl_car)
    PP.SETTINGS["board_carrier"] = dict(PP.PLA_OBJECT, perimeters="3", fill_density="20%", fill_pattern="gyroid",
                                        top_solid_layers="6", bottom_solid_layers="5")
    PP.LABEL["board_carrier"] = "Board carrier, chip clearance 0.20 mm"
    pin = A_DIR / "print" / f"{DLF.PFX}_pin_fit_coupon.stl"            # the clamp plate's pins at several diameters
    r = PP.build_project({"board_carrier": str(stl_car), "clamp_plate": str(DLF.fname("clamp_plate")),
                          "pin_fit_coupon": str(pin) if pin.exists() else None}, COUPONS,
                         f"{pf}_board_carrier_clamp_pins_pla_t3", source="scripts/dip_link_v2.py coupons")
    rep["projects"]["board_carrier"] = r["project"]
    # (3) window gauges
    gst = {}
    for c in GAUGE_CLEAR:
        g = coupon_gauge(A, K, c)
        g.vertices = np.asarray(g.vertices) @ PP.R_LINK.T
        key = f"gauge_{c:.2f}".replace(".", "p")
        gst[key] = COUPONS / f"{pf}_window_gauge_clear{key[6:]}.stl"
        g.export(gst[key])
        PP.SETTINGS[key] = dict(PP.PLA_OBJECT, perimeters="2", fill_density="100%", fill_pattern="rectilinear")
        PP.LABEL[key] = f"Window gauge, chip clearance {c:.2f} mm"
    r = PP.build_project({k: str(v) for k, v in gst.items()}, COUPONS, f"{pf}_window_gauges_pla_t3",
                         source="scripts/dip_link_v2.py coupons")
    rep["projects"]["window_gauges"] = r["project"]
    rep["when"] = time.strftime("%Y-%m-%d %H:%M")
    (COUPONS / f"{pf}s.json").write_text(json.dumps(rep, indent=1))
    print("coupons ->", COUPONS.relative_to(REPO), flush=True)


def cmd_a(a):
    for stage in a.stages:
        t0 = time.time()
        print(f"== a: {stage}", flush=True)
        if stage == "build":
            A, DLF, *_ = setup_a()
            DLF.cmd_build(argparse.Namespace())
        elif stage == "step":
            a_step()
        elif stage == "check":
            A, DLF, *_ = setup_a()
            DLF.cmd_check(argparse.Namespace(reuse=False))
        elif stage == "sense":
            A, DLF, *_ = setup_a()
            DLF.cmd_sense(argparse.Namespace())
        elif stage == "render":
            a_render()
            a_section()
        elif stage == "print":
            a_print()
        elif stage == "docs":
            a_docs()
        print(f"== a: {stage} [{time.time() - t0:.0f} s]", flush=True)


# ----------------------------------------------------------------------------------------------- concept b, v2
PARAMS_B = STUDY / "params/20261018-v2_concept_b.json"
B_DIR = V2 / "b"


def run_b(*cmd, python=None):
    """scripts/dip_link_concept_b.py at v2's parameter file (its outputs name v2/b/ and the work folder under logs/)."""
    import subprocess
    env = dict(os.environ, DIPB_PARAMS=str(PARAMS_B))
    r = subprocess.run([str(python or Path.home() / "miniconda3/envs/femtip/bin/python"), "-u",
                        str(REPO / "scripts/dip_link_concept_b.py"), *cmd], env=env, cwd=str(REPO), capture_output=True, text=True)
    print("\n".join(l[:260] for l in r.stdout.splitlines() if l.strip() and "*" not in l)[-3500:], flush=True)
    if r.returncode:
        print(r.stderr[-3000:])
        raise SystemExit(f"dip_link_concept_b.py {' '.join(cmd)} failed")


def b_module():
    os.environ["DIPB_PARAMS"] = str(PARAMS_B)
    import dip_link_concept_b as B
    return B


def print_pads_b(B, out, work):
    """Each pad of b's design with its magnet pauses: the print STL that dip_link_concept_b prep wrote, sliced with its
    sheet's pockets into out/; the block pads print on their cut back face, so their pockets and pauses move down with it,
    each pause on the 0.2 mm layer grid before the pocket's bridge layer."""
    import trimesh
    import metric_magnet_prints as MP
    import sim2real_fingertip_prints as SP
    pf, out = B.PREFIX, Path(out)
    MP.DATE = SP.DATE = DATE
    SP.OUT, SP.WORK = out, Path(work)
    SP.use_template()
    for key, t in B.tip_records().items():
        stl = out / f"{pf}_pad_{key.replace('.', 'p')}_print.stl"
        if key not in B.TIPS or "sheet" not in t or not stl.exists():
            continue
        sheet = json.loads((REPO / t["sheet"]).read_text())
        name_, orient, _ = B.TIPS[key]
        part = trimesh.load(stl, force="mesh")
        pk = json.loads(json.dumps(sheet["pockets"]))
        if orient == "back":
            dz = B.TIP["x_back"] - B.X_SLOT
            for q in pk:
                for f_ in ("floor_mm", "top_mm", "magnet_top_print_mm"):
                    if f_ in q:
                        q[f_] = round(q[f_] - dz, 3)
                # the pause goes before the pocket's bridge layer, the first 0.2 mm layer sliced (at its mid-height) above
                # the pocket's top; the lowered pause is off the layer grid when dz is not a multiple of 0.2 mm
                q["pause_layer"] = int(math.ceil((q["top_mm"] + 0.1) / 0.2 - 1e-9))
                q["pause_print_z"] = round(0.2 * q["pause_layer"], 3)
        name = f"{pf}_print_pad_{key.replace('.', 'p')}_85a_t2_g30"
        ns = argparse.Namespace(seat=sheet.get("seat_mm", SEAT), no_slice=False)
        rep = SP.write_and_check(part, pk, sheet["pocket_gaps"], sheet["mesh"], name, sheet["tip"], sheet["layout"],
                                 sheet["magnet"]["top_depth_mm"], 2, 30, part, five_layout(sheet["layout"]), ns)
        rep["stl"] = str(stl.relative_to(REPO)) if REPO in stl.parents else str(stl)
        rep["note"] = (f"the sheet's tip cut at the slot plane x_F {B.X_SLOT:.2f} with the rail grooves and the head notches"
                       + ("; printed on the cut back face, so the pockets and the pause sit "
                          f"{B.TIP['x_back'] - B.X_SLOT:.2f} mm lower than on the whole block" if orient == "back" else ""))
        (out / f"{name}.json").write_text(json.dumps(rep, indent=1, default=float))


def cmd_pads(a):
    """Concept b's pads at the parameter file in $DIPB_PARAMS (with $DIPB_OUT, $DIPB_WORK, $DIPB_PREFIX as
    scripts/dip_link_params.py export sets them), sliced with their pauses into --out."""
    import dip_link_concept_b as B
    print_pads_b(B, a.out, a.work)
    print("pads ->", a.out, flush=True)


BAMBU_PLA = {     # Bambu Studio settings per PLA part (owner, 2026-10-19 14:15: PLA on the Bambu printer, TPU on the XL)
    "dip_link": dict(orientation="tip end up, the clevis arms' rounded ends on the plate (X = -x_F, Y = y_F, Z = -z_F)",
                     layer_height_mm=0.1, walls=3, top_layers=6, bottom_layers=5, infill="15 % gyroid",
                     supports="tree supports on the build plate only, painted (or enforced) under both clevis arms' ends and "
                              "under the bridge's face toward the PIP axis; none in the J1 slot or the cable groove",
                     brim="5 mm brim (the arms' ends touch the plate along short lines)"),
    "cover": dict(orientation="on its back face, the rails up (X = z_F, Y = y_F, Z = -x_F)", layer_height_mm=0.1, walls=4,
                  infill="100 % rectilinear", supports="none", brim="none", ironing="top surfaces"),
    "rails": dict(orientation="on its back face, the rails up", layer_height_mm=0.1, walls=3, infill="100 % rectilinear",
                  supports="none", brim="none")}


STEP_TO_STL = r"""
import os
import MeshPart
import Part
sh = Part.read(os.environ["DLV2_STEP"])
msh = MeshPart.meshFromShape(Shape=sh, LinearDeflection=0.02, AngularDeflection=0.15, Relative=False)
msh.write(os.environ["DLV2_STL"])
print("tessellated", len(sh.Solids), "solid(s)", round(sh.Volume, 2), "mm3", msh.CountFacets, "facets", flush=True)
"""


def step_to_stl(step, stl):
    """An exact STEP tessellated in FreeCAD (MeshPart, 0.02 mm linear and 0.15 rad angular deflection, as
    dip_link_concept_b.fc_build writes its STL files)."""
    import subprocess
    script = Path(stl).parent / "step_to_stl.py"
    script.write_text(STEP_TO_STL)
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    r = subprocess.run(["freecadcmd", str(script)], env=dict(env, DLV2_STEP=str(step), DLV2_STL=str(stl)), cwd=str(REPO),
                       capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith("tessellated")), flush=True)
    if r.returncode or not Path(stl).exists():
        print(r.stderr[-2000:])
        raise SystemExit("freecadcmd tessellation failed")
    return Path(stl)


def bambu_stl(tm, R, dst):
    """A PLA part as a print-oriented STL for Bambu Studio: turned into its print frame (dip_link_print_project.to_print),
    lying on z = 0, vertices merged and zero-area faces dropped so the slicer sees one watertight body."""
    import trimesh
    import dip_link_print_project as PP
    m, Rt = PP.to_print(tm, R)
    m = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.faces), process=True)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    bodies = [b for b in m.split(only_watertight=False) if abs(b.volume) > 0.5]     # zero-volume slivers of the booleans
    m = trimesh.util.concatenate(bodies) if len(bodies) > 1 else bodies[0]
    m.merge_vertices()
    if m.volume < 0:
        m.invert()
    m.export(dst)
    return dict(file=str(dst.relative_to(REPO)), size_mm=np.round(m.extents, 2).tolist(), volume_mm3=round(float(m.volume), 1),
                watertight=bool(m.is_watertight), bodies=len(m.split(only_watertight=False)),
                rotation_print_from_F=np.asarray(Rt[0]).tolist())


def b_bambu_files(link_src, cover_src, out, pf, params, rails=None):
    """Concept b's PLA parts (the DIP link, the cover and, given as a mesh in F, the rail strip) as Bambu STLs into out/ with
    <pf>_pla_bambu.json and their settings (BAMBU_PLA)."""
    import trimesh
    import dip_link_print_project as PP
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(printer="Bambu Lab P1S through Bambu Studio, PLA", params=params, parts={},
               settings={k: BAMBU_PLA[k] for k in ("dip_link", "cover") + (("rails",) if rails is not None else ())},
               when=time.strftime("%Y-%m-%d %H:%M"))
    rec["parts"]["dip_link"] = bambu_stl(trimesh.load(link_src, force="mesh"), PP.R_LINK, out / f"{pf}_dip_link_pla_bambu.stl")
    rec["parts"]["cover"] = bambu_stl(trimesh.load(cover_src, force="mesh"), PP.R_CLAMP, out / f"{pf}_cover_pla_bambu.stl")
    if rails is not None:
        rec["parts"]["rails"] = bambu_stl(rails, PP.R_CLAMP, out / f"{pf}_fit_coupon_rails_pla_bambu.stl")
    for k, v in rec["parts"].items():
        print(f"bambu {k}: {v}", flush=True)
    (out / f"{pf}_pla_bambu.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    return rec


def b_print():
    """The DIP link, the cover and the rail strip as print-oriented PLA STLs for Bambu Studio with their settings (owner,
    2026-10-19: PLA on the Bambu printer), the 85A groove blocks of the rail fit coupon and the two pads with their pauses
    as PrusaSlicer projects for the XL's tool 2."""
    import trimesh
    import dip_link_print_project as PP
    import metric_magnet_prints as MP
    import sim2real_fingertip_prints as SP
    B = b_module()
    pf, out = B.PREFIX, B_DIR / "print"
    out.mkdir(parents=True, exist_ok=True)
    PP.PFX = pf
    for stale in out.glob(f"{pf}_*_pla_t3*"):         # the XL projects of the PLA parts (before 2026-10-19 14:15)
        stale.unlink()
    # the rail fit coupon: a 10 mm length of the cover with its two rails (PLA) and three 85A blocks with the pad's grooves
    # at 0, 0.05 and 0.10 mm interference per flank, each block notched once per 0.05 mm
    K = B.MeshB()
    z0, z1 = B.Z_CONTACT - 5.0, B.Z_CONTACT + 5.0
    strip = K.box((B.X_SLOT, -B.Y_RIM, z0), (B.X_COMP, B.Y_RIM, z1))
    rails = [K.prism_z([(s * y, (B.X_SLOT + 0.1) if x > B.X_SLOT else x) for y, x in B.rail_section()], z0, z1) for s in (-1, 1)]
    b_bambu_files(B.CAD_DIR / f"{pf}_dip_link.stl", B.CAD_DIR / f"{pf}_cover.stl", out, pf, str(PARAMS_B.relative_to(REPO)),
                  rails=B.to_trimesh(K.fuse(strip, rails)))
    blocks = []
    fit0 = B.GROOVE_FIT
    for k, fit in enumerate((0.0, 0.05, 0.10)):
        B.GROOVE_FIT = fit
        blk = K.box((B.X_SLOT - 4.0, -B.TIP["y_half"], z0 + 0.5), (B.X_SLOT, B.TIP["y_half"], z1 - 0.5))
        cut = []
        for s in (-1, 1):
            sec = [(s * y, B.X_SLOT + 0.5 if x > B.X_SLOT else x) for y, x in B.rail_section(-B.GROOVE_FIT, B.GROOVE_FLOOR)]
            cut.append(B.prism(sec, z0, z1))
        cut += [K.box((B.X_SLOT - 4.1, -B.TIP["y_half"] - 0.1 + 1.2 * j, z1 - 1.0), (B.X_SLOT - 3.4, -B.TIP["y_half"] + 0.6 + 1.2 * j,
                                                                                     z1)) for j in range(k + 1)]
        tm = B.to_trimesh(K.cut(blk, cut))
        V = np.column_stack([np.asarray(tm.vertices)[:, 2], np.asarray(tm.vertices)[:, 1], B.X_SLOT - np.asarray(tm.vertices)[:, 0]])
        V[:, 0] += 12.0 * k
        blocks.append(trimesh.Trimesh(V, np.asarray(tm.faces), process=False))
    B.GROOVE_FIT = fit0
    stl_blocks = out / f"{pf}_fit_coupon_grooves_85a.stl"
    trimesh.util.concatenate(blocks).export(stl_blocks)
    PP.LABEL["tip"] = "85A blocks with the pad's grooves, 0 / 0.05 / 0.10 mm per flank"
    PP.build_project({"tip": stl_blocks}, out, f"{pf}_fit_coupon_grooves_85a_t2", source=str(PARAMS_B.relative_to(REPO)))
    print_pads_b(B, out, REPO / "logs/20261018-dip_link/v2/b/print")
    print("print ->", out.relative_to(REPO), flush=True)


STEP_READ = r"""
import json, os, sys, tempfile, zipfile
import Part
out = {}
for f in json.loads(os.environ["DLV2_FILES"]):
    path, tmp = f, None
    if f.endswith(".zip"):
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(f) as zf:
            zf.extractall(tmp)
            path = os.path.join(tmp, zf.namelist()[0])
    sh = Part.read(path)
    out[os.path.basename(f)] = dict(valid=bool(sh.isValid()), solids=len(sh.Solids), faces=len(sh.Faces),
                                    volume_mm3=round(sh.Volume, 2), bytes=os.path.getsize(f))
    print(os.path.basename(f), out[os.path.basename(f)], flush=True)
open(os.environ["DLV2_OUT"], "w").write(json.dumps(out, indent=1))
print("read back", len(out), flush=True)
"""


def b_step():
    """Every STEP file of v2/b/cad read back in FreeCAD (Part.read; the zipped ones unzipped first): validity, solids,
    faces, volume -> v2/b/<prefix>_step.json."""
    import subprocess
    B = b_module()
    files = sorted(str(f) for f in B.CAD_DIR.glob("*.step*"))
    work = REPO / "logs/20261018-dip_link/v2/b"
    work.mkdir(parents=True, exist_ok=True)
    script = work / "step_read.py"
    script.write_text(STEP_READ)
    dst = B_DIR / f"{B.PREFIX}_step.json"
    env = dict(os.environ, DLV2_FILES=json.dumps(files), DLV2_OUT=str(dst))
    r = subprocess.run(["freecadcmd", str(script)], env=env, cwd=str(REPO), capture_output=True, text=True)
    print("\n".join(l[:300] for l in r.stdout.splitlines() if l.strip() and "%" not in l and "#" not in l)[-3000:], flush=True)
    if r.returncode or "read back" not in r.stdout:
        print(r.stderr[-3000:])
        raise SystemExit("freecadcmd read-back failed")
    rec = json.loads(dst.read_text())
    dst.write_text(json.dumps({Path(k).name: v for k, v in rec.items()} | dict(
        method="freecadcmd Part.read of each file (zips unzipped), FreeCAD 0.19", when=time.strftime("%Y-%m-%d %H:%M")),
        indent=1) + "\n")


def b_section():
    """Section y = 0 through U1 and J1 at PIP 0 and at full flexion (92 deg), as a_section: the stack, J1 with its plug
    and cable in the neck's slot, the pad with its magnets, the MCP link and the PIP servo in outline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    from matplotlib.patches import Polygon as MPoly
    import dip_link_geometry as G
    B = b_module()
    pf = B.PREFIX
    hw = B.hardware()
    tip = "ell11_5"
    parts = [("Pad, 85A (11/22, five magnets)", B.load_cad(f"pad_{tip}"), "#e8c88a"),
             ("DIP link", B.load_cad("dip_link"), "#d6d6cc"), ("Cover", B.load_cad("cover"), "#5b8fd9"),
             ("Board", B.to_trimesh(hw["fr4"]), "#1f7a3a"), ("Chips and parts", B.to_trimesh(hw["board_parts"]), "#202020"),
             ("J1 and its plug" if B.cable_on() else "J1, plug, cable start", B.to_trimesh(hw["j1"]), "#f2f2ea"),
             ("Screws", B.to_trimesh(hw["screws"]), "#c8a33c")]
    if B.cable_on():
        parts.append((f"Qwiic cable ({B.cp('cable_d'):g} mm bundle)", B.cable_mesh(), "#2f5bd8"))
    ctx = {n: G.load_part(n) for n in ("mcp_link", "servo_pip")}
    plt.rcParams.update({"font.family": "serif", "font.size": 15})
    fig, axs = plt.subplots(1, 2, figsize=(17.0, 12.0))
    for ax, q in zip(axs, (0.0, 92.0)):
        for n, tm, c in parts:
            sec = tm.section(plane_origin=(0, 0.0, 0), plane_normal=(0, 1, 0))
            if sec is None:
                continue
            for poly in sec.discrete:
                ax.add_patch(MPoly(poly[:, [0, 2]], closed=True, fc=c, ec="k", lw=0.5))
        for n, tm in ctx.items():
            t2 = tm.copy()
            t2.vertices = G.rot_pip(np.asarray(t2.vertices), q)
            sec = t2.section(plane_origin=(0, 0.0, 0), plane_normal=(0, 1, 0))
            if sec is None:
                continue
            for poly in sec.discrete:
                ax.add_patch(MPoly(poly[:, [0, 2]], closed=True, fc="none", ec="#3a6fb0" if n == "mcp_link" else "#b04a2a",
                                   lw=1.2))
        for c, x0, x1 in B.magnets(tip):
            if abs(c[1]) < 1.6:
                ax.add_patch(MPoly([[x0, c[2] - 1.5], [x1, c[2] - 1.5], [x1, c[2] + 1.5], [x0, c[2] + 1.5]], closed=True,
                                   fc="#b03a2e", ec="k", lw=0.5))
        ax.plot([0], [-51.55], "k+", ms=14, mew=2)
        ax.set_xlim(-14, 24)
        ax.set_ylim(B.TIP["z_end"] - 2.0, -40.0)
        ax.set_aspect("equal")
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.set_xlabel("x in F (mm), dorsal to the right")
        ax.set_ylabel("z in F (mm)")
        ax.set_title(f"PIP {q:.0f} deg: MCP link (blue) and PIP servo (red) in outline", fontsize=16)
    jb = {n: (np.array(lo), np.array(hi)) for n, lo, hi in B.j1_boxes_b()}
    hd = jb["j1_header"]
    axs[0].annotate("J1 on the whole tab, in the neck's slot", xy=((hd[0][0] + hd[1][0]) / 2, (hd[0][2] + hd[1][2]) / 2),
                    xytext=(10.5, B.Z_CONTACT + 14.4), fontsize=15, va="top", arrowprops=dict(arrowstyle="->", lw=1.0))
    axs[0].annotate("Plug; the cable turns\ntoward the back, then\nto the slotted arm" if B.cable_on() else
                    "Plug and the cable's\nfirst 3 mm; the slot\nopens through the\nbridge's top",
                    xy=((jb["plug"][0][0] + jb["plug"][1][0]) / 2, jb["plug"][1][2] - 0.5), xytext=(10.5, -64.5), fontsize=15, va="top",
                    arrowprops=dict(arrowstyle="->", lw=1.0))
    axs[0].annotate("Slot open at the neck's\nunderside: the board\nslides in along +z", xy=(2.2, B.TIP["z_mount"]),
                    xytext=(10.5, B.TIP["z_mount"] - 6.0), fontsize=15, va="top", arrowprops=dict(arrowstyle="->", lw=1.0))
    axs[0].annotate("PIP axis", xy=(0.0, -51.55), xytext=(-12.5, -43.5), fontsize=15, arrowprops=dict(arrowstyle="->", lw=1.0))
    from matplotlib.patches import Patch
    fig.legend([Patch(fc=c, ec="k") for _, _, c in parts] + [Patch(fc="#b03a2e", ec="k")],
               [n for n, _, _ in parts] + ["Magnets (y = 0)"], loc="lower center", ncol=4, frameon=False, fontsize=15)
    fig.suptitle("Concept b, v2: section y = 0 through U1 and J1", fontsize=18)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    dst = B_DIR / "media" / f"{pf}_section_j1.png"
    fig.savefig(dst, dpi=100)
    plt.close(fig)
    print(dst.relative_to(REPO), flush=True)


def b_cable_section(B, clr=None):
    """Side views of b's cable route: left, the x-z section y = 0.05 through J1, the plug and the turn toward the back,
    with the MCP link and the PIP servo at four PIP angles and the cable's nearest approach; right, the y-z section through
    the run to the slotted arm and the turn up its outer face."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon as MPoly
    import dip_link_geometry as G
    pf = B.PREFIX
    hw = B.hardware()
    D = B.load_cad("dip_link")
    meshes = [(D, "#d6d6cc"), (B.to_trimesh(hw["fr4"]), "#1f7a3a"), (B.to_trimesh(hw["j1"]), "#f2f2ea")]
    ctx = {n: G.load_part(n) for n in ("mcp_link", "servo_pip")}
    C = B.cable_polyline()
    seg = B.cable_segments()
    xr = B.cp("cable_run_x")
    qcol = {-18.0: "#c0392b", 0.0: "#2c3e50", 45.0: "#16a085", 92.0: "#8e44ad"}
    plt.rcParams.update({"font.family": "serif", "font.size": 15})
    fig, axs = plt.subplots(1, 2, figsize=(17.0, 10.0))
    views = [(axs[0], (0, 1, 0), 0.05, (0, 2), "x in F (mm), dorsal to the right",
              "Section y = 0.05 mm, through J1, the plug and the turn toward the back", (-14, 14), (-80, -50)),
             (axs[1], (1, 0, 0), xr, (1, 2), "y in F (mm), toward the slotted arm to the right",
              f"Section x = {xr:.2f} mm, through the run to the slotted arm", (-6, 22), (-74, -48))]
    for ax, nrm, at, ij, xl, title, xlim, ylim in views:
        o = np.zeros(3)
        o[int(np.argmax(nrm))] = at
        for tm, fc in meshes:
            s = tm.section(plane_origin=o, plane_normal=nrm)
            if s is not None:
                for poly in s.discrete:
                    ax.add_patch(MPoly(poly[:, list(ij)], closed=True, fc=fc, ec="k", lw=0.5))
        for q, c in qcol.items():
            for n, tm in ctx.items():
                t2 = tm.copy()
                t2.vertices = G.rot_pip(np.asarray(t2.vertices), q)
                s = t2.section(plane_origin=o, plane_normal=nrm)
                if s is not None:
                    for poly in s.discrete:
                        ax.add_patch(MPoly(poly[:, list(ij)], closed=True, fc="none", ec=c, lw=1.3))
        k = int(np.argmax(nrm))
        on = np.abs(C[:, k] - at) < 1.05                    # the part of the centre line within the bundle's radius
        Cs = np.where(on[:, None], C, np.nan)
        ax.plot(Cs[:, ij[0]], Cs[:, ij[1]], "-", color="#2f5bd8", lw=9, alpha=0.55, solid_capstyle="round")
        ax.plot(Cs[:, ij[0]], Cs[:, ij[1]], "-", color="#1b2f80", lw=1.0)
        if ij == (0, 2):
            ax.plot([0], [-51.55], "k+", ms=14, mew=2)
            ax.annotate("PIP axis", xy=(0.0, -51.55), xytext=(-12, -53.5), fontsize=14, arrowprops=dict(arrowstyle="->", lw=1.0))
            jb = {n: (np.array(lo), np.array(hi)) for n, lo, hi in B.j1_boxes_b()}
            ax.annotate("Plug in J1", xy=(jb["plug"][0][0] + 0.5, 0.5 * (jb["plug"][0][2] + jb["plug"][1][2])),
                        xytext=(-12.5, -72), fontsize=14, arrowprops=dict(arrowstyle="->", lw=1.0))
            if clr and clr.get("qwiic_cable", {}).get("at_F"):
                p_ = clr["qwiic_cable"]
                ax.plot([p_["at_F"][0]], [p_["at_F"][2]], "o", ms=7, mfc="none", mec="#c0392b", mew=1.8)
                ax.annotate(f"Nearest approach {p_['min_mm']:.2f} mm\n(MCP link, PIP {p_['pip_deg']:.0f} deg)",
                            xy=(p_["at_F"][0], p_["at_F"][2]), xytext=(4.0, -56.5), fontsize=14,
                            arrowprops=dict(arrowstyle="->", lw=1.0))
            ax.annotate("Turn toward the back", xy=(seg[1][2][0], seg[1][2][2]), xytext=(6.5, -73.5), fontsize=14,
                        arrowprops=dict(arrowstyle="->", lw=1.0))
        else:
            ax.annotate("Turn into the slotted\narm's outer face", xy=(seg[-2][2][1], seg[-2][2][2]), xytext=(3.0, -52.5),
                        fontsize=14, arrowprops=dict(arrowstyle="->", lw=1.0))
            ax.annotate("Run along +y to the arm", xy=(8.0, seg[-3][1][2]), xytext=(-4.5, -71.5), fontsize=14,
                        arrowprops=dict(arrowstyle="->", lw=1.0))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.set_xlabel(xl)
        ax.set_ylabel("z in F (mm)")
        ax.set_title(title, fontsize=15)
    fig.legend([Patch(fc="#d6d6cc", ec="k"), Patch(fc="#1f7a3a", ec="k"), Patch(fc="#f2f2ea", ec="k"),
                Patch(fc="#2f5bd8", alpha=0.55)] + [Patch(fc="none", ec=c, lw=1.3) for c in qcol.values()],
               ["DIP link", "Board", "J1 and its plug", f"Cable, {B.cp('cable_d'):g} mm bundle"]
               + [f"MCP link and PIP servo at PIP {q:+.0f} deg" for q in qcol], loc="lower center", ncol=4, frameon=False,
               fontsize=14)
    fig.suptitle(f"Concept b, v2: the Qwiic cable's route (DIP link {B.NECK:g} mm longer)", fontsize=18)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    dst = B_DIR / "media" / f"{pf}_cable_section.png"
    fig.savefig(dst, dpi=100)
    save_panels(fig, axs, dst)
    plt.close(fig)
    print(dst.relative_to(REPO), flush=True)


def b_walls(B):
    """Least walls of the printed parts in sections through their thin features (dip_link_concept_a.section_min_wall)."""
    import dip_link_concept_a as A
    body, cover = B.load_cad("dip_link_new_body"), B.load_cad("cover")
    zj = 0.5 * (B.Z_CONTACT + 13.14 + B.Z_CONTACT + 18.09)                   # J1's header, mid-height
    neck = ((B.TIP["x_palmar"] - 0.1, -B.Y_RIM - 0.1, B.TIP["z_mount"]), (B.X_SPINE_BACK + 0.1, B.Y_RIM + 0.1, -60.0))
    tongue = ((B.X_SEAT - B.RIM_H - 0.1, -B.Y_RIM - 0.1, B.TIP["z_end"]), (B.X_TONGUE_BACK + 0.1, B.Y_RIM + 0.1, B.TIP["z_mount"]))
    sec = {f"neck_z{zj:.2f}_around_J1": (body, (0, 0, zj), (0, 0, 1), neck),
           "neck_y0": (body, (0, 0.0, 0), (0, 1, 0), neck),
           f"neck_y{-3.6:g}": (body, (0, -3.6, 0), (0, 1, 0), neck),
           f"tongue_z{B.Z_CONTACT + 8.1:.2f}_proximal_screws": (body, (0, 0, B.Z_CONTACT + 8.1), (0, 0, 1), tongue),
           f"tongue_z{B.Z_CONTACT - 7.9:.2f}_distal_screws": (body, (0, 0, B.Z_CONTACT - 7.9), (0, 0, 1), tongue),
           f"tongue_x{(B.X_SEAT + B.X_TONGUE_BACK) / 2:.2f}": (body, ((B.X_SEAT + B.X_TONGUE_BACK) / 2, 0, 0), (1, 0, 0), tongue),
           f"cover_x{(B.X_SLOT + B.X_COMP) / 2:.2f}": (cover, ((B.X_SLOT + B.X_COMP) / 2, 0, 0), (1, 0, 0), None)}
    if B.cable_on():                                 # the bridge over the cable's groove: along the run and across it
        seg = B.cable_segments()
        z1 = seg[1][2][2]
        bridge = ((B.TIP["x_palmar"] - 0.1, B.Y_BRIDGE[0], -70.0), (B.X_SPINE_BACK + 0.1, B.Y_BRIDGE[1], -58.0))
        for y in (1.0, 6.0, 11.0):
            sec[f"bridge_y{y:g}_over_the_groove"] = (body, (0, y, 0), (0, 1, 0), bridge)
        sec[f"bridge_z{z1 + 1.6:.2f}_over_the_groove"] = (body, (0, 0, z1 + 1.6), (0, 0, 1), bridge)
    out = {}
    for k, (tm, o, n, b3) in sec.items():
        out[k] = A.section_min_wall(tm, o, n, box3=b3)
        print(f"wall {k}: {out[k]}", flush=True)
    return out


def b_r4(B):
    """R4: the pad's rail grooves and head notches against its magnet pockets (the voids of the pad's mesh), per pad."""
    import trimesh
    g_lo = np.array([B.X_SLOT - B.RAIL_H - B.GROOVE_FLOOR, B.RAIL_Y - B.RAIL_TOP / 2 + B.GROOVE_FIT, B.RAIL_Z[0] - 0.2])
    g_hi = np.array([B.X_SLOT, B.RAIL_Y + B.RAIL_TOP / 2 - B.GROOVE_FIT, B.RAIL_Z[1] + 0.2])

    def box_dist(P, lo, hi):
        return np.linalg.norm(np.maximum(np.maximum(lo - P, P - hi), 0.0), axis=1)
    out = {}
    for key in B.TIPS:
        pm = trimesh.load(B.WORK / f"pad_{key}.stl")
        voids = [b for b in pm.split(only_watertight=False) if abs(b.volume) < 100 and len(b.faces) > 20]
        Vv = np.vstack([b.vertices for b in voids])
        dg = float(min(box_dist(Vv, g_lo, g_hi).min(), box_dist(Vv * np.array([1, -1, 1]), g_lo, g_hi).min()))
        dn = []
        for y, z in B.screw_yz():             # to the notch's cylinder (radius HEAD_NOTCH_D/2, x X_SLOT - HEAD_NOTCH_H .. X_SLOT)
            dr = np.maximum(np.hypot(Vv[:, 1] - y, Vv[:, 2] - z) - B.HEAD_NOTCH_D / 2, 0.0)
            dx = np.maximum(B.X_SLOT - B.HEAD_NOTCH_H - Vv[:, 0], 0.0)
            dn.append(float(np.hypot(dr, dx).min()))
        out[key] = dict(groove_to_pocket_mm=round(dg, 2), notch_to_pocket_mm=round(min(dn), 2), need_mm=2.0,
                        ok=bool(min(dg, min(dn)) >= 2.0), pocket_dorsal_x_F=round(float(Vv[:, 0].max()), 2),
                        tpu_behind_pockets_mm=round(B.X_SLOT - float(Vv[:, 0].max()), 2))
        print("R4", key, out[key], flush=True)
    return out


def project_summary(v):
    """Print time, filament, pauses and the slice checks of a print project's record: the PLA projects
    (dip_link_print_project) give them under 'slice', the pads (sim2real_fingertip_prints) under 'slice' -> 'check', with a
    pass when every pocket's pause comes before its bridge layer, no bead lies in a pocket and the walls pass."""
    sl = v.get("slice") or {}
    ck = sl.get("check") or {}
    ok = sl.get("checks_pass")
    if ok is None and ck.get("pockets"):
        ok = bool(v.get("walls_ok") and all(q.get("pause_before_bridge") and q.get("beads_in_void_max", 0) == 0
                                            for q in ck["pockets"]))
    return dict(project=v.get("project"), print_time=sl.get("print_time") or ck.get("print_time") or sl.get("time"),
                filament_g=sl.get("filament_g") or ck.get("filament_g"),
                pauses_mm=[q.get("print_z") for q in v.get("pauses", [])], checks_pass=ok)


def b_docs():
    """v2/b/evaluation.json: the lengthening, the Qwiic connection, clearance, sensing, walls, R4, mass, the print projects,
    the parts list and the assembly sequence, from the stage outputs."""
    B = b_module()
    pf = B.PREFIX
    W = B.WORK
    rd = lambda f: json.loads((W / f).read_text())                         # noqa: E731
    fc, clr, sens, stat = rd("fc_build.json"), rd("clearance.json"), rd("sensing.json"), rd("static_fields.json")
    prep = rd("prep.json")
    lg = json.loads((V2 / f"{DATE}-v2_lengthening.json").read_text())
    tips = json.loads((V2 / f"{DATE}-v2_tips.json").read_text())
    step = json.loads((B_DIR / f"{pf}_step.json").read_text()) if (B_DIR / f"{pf}_step.json").exists() else {}
    prj = {f.stem: json.loads(f.read_text()) for f in sorted((B_DIR / "print").glob("*.json"))}
    hw = B.hardware()
    d5 = next(r for r in sens if r["layout"] == "five_d3x2_d5" and abs(r["t_behind_mm"] - B.T_BEHIND) < 1e-6
              and r["placement"] == "concept_b")
    t5 = next(r for r in sens if r["layout"] == "five_d3x2_d5" and r["t_behind_mm"] == 10.0 and r["placement"] == "touch")
    fe5 = next(json.loads(l) for l in open(B.FE_ROWS) if json.loads(l)["layout"] == "five_d3x2_d5"
               and abs(json.loads(l)["t_behind_mm"] - B.T_BEHIND) < 1e-6)
    fe5w = next(json.loads(l) for l in open(B.FE_ROWS) if json.loads(l)["layout"] == "five_d3x2_d5"
                and json.loads(l)["t_behind_mm"] == 10.0)
    walls = b_walls(B)
    r4 = b_r4(B)
    # mass: solid volumes, PLA 1.24 g/cm3 (the link also at the current link's sliced 0.948), brass screws, the board
    vol = {k: v["volume_mm3"] for k, v in fc.items() if "volume_mm3" in v}
    m_screws = float(hw["screws"].volume()) * B.RHO["brass"] / 1e3
    mass = dict(dip_link_solid=round(vol["dip_link"] * B.RHO["pla"] / 1e3, 2),
                dip_link_at_sliced_density=round(vol["dip_link"] * B.RHO["pla_15"] / 1e3, 2),
                cover_solid=round(vol["cover"] * B.RHO["pla"] / 1e3, 2), screws_brass=round(m_screws, 2), board=0.64)
    mass["total_without_pad"] = round(mass["dip_link_at_sliced_density"] + mass["cover_solid"] + mass["screws_brass"] + 0.64, 2)
    mass["limit_R12"] = 7.6
    link_prj = next((v for k, v in prj.items() if "link_cover" in k), None)
    if link_prj and link_prj.get("slice"):
        mass["sliced_filament_link_and_cover_g"] = link_prj["slice"].get("filament_g")
    rigid = {k: v for k, v in clr.items()
             if k not in ("arms_kept", "board_to_servo", "chips_to_servo", "qwiic_cable") and not k.startswith("pad")}
    worst = min(rigid.items(), key=lambda kv: kv[1]["min_mm"])
    j1b = {n: (lo, hi) for n, lo, hi in B.j1_boxes_b()}
    routed = B.cable_on()
    rings = B_DIR / "20261019-dip_link_v2b_cable_routes.json"
    rings = json.loads(rings.read_text())["rows"] if rings.exists() else []
    ar = [r for r in rings if r["route"] == "a_route"]
    span = cable_span(anchor=B.cable_segments()[-1][2]) if routed else None
    bambu = prj.get(f"{pf}_pla_bambu", {}).get("parts", {})
    ev = dict(
        title=(f"DIP link concept b, v2: the board in a cassette behind five-magnet pads on a DIP link {B.NECK:g} mm longer, "
               "the board's own Qwiic connector kept" + (", the Qwiic cable routed out past the slotted arm" if routed else "")
               + ", M2 self-forming screws"),
        date="2026-10-19" if routed else "2026-10-18", params=str(PARAMS_B.relative_to(REPO)),
        generator="scripts/dip_link_v2.py b (scripts/dip_link_concept_b.py at the parameter file)",
        status="design; model values; nothing printed or measured",
        owner=["'we should just go for 5x 3x2 magnets'", "'i do like the half ellipsoids most, the 11/22 one especially'",
               "'if we have to lengthen the fingertip to get everything to fit, including the qwiic port and the cable, that's "
               "fine'", "'for m3 screws, i use 2.75mm diameter holes, m2 i use 1.8'"]
              + (["2026-10-19: 'how are you planning to route the cable out though?'; 'i would like to minimize range of "
                  "motion loss, if any, on the back extension of the DIP joint'; 14:15: PLA on the Bambu printer through Bambu "
                  "Studio, TPU on the Prusa XL 5T"] if routed else []),
        terms=dict(
            F="mm, x toward the dorsal side, y along the PIP axis, z up; the straight finger hangs along -z, the PIP axis passes "
              "through (0, y, -51.55)",
            slot_plane="x_F of the pad's back face, the cover's front face and the chips' package tops",
            clearance="least signed distance from a part fixed to the DIP link to the MCP link and the PIP servo over PIP -18 to "
                      "92 deg, mm (dip_link_geometry.points_clearance_exact, 0.5 deg steps); R7 asks 0.5 mm for rigid parts",
            force_resolution="Cramer-Rao bound of the normal force at the contact point from one 141 Hz sample of the five "
                             "chips, the contact position and a uniform background field unknown, N",
            E_star="pressure-law modulus fitted to the finite-element approach of the screwdriver cylinder (R 12.5 mm), MPa",
            magnet_travel="mean rigid translation of the five magnets per newton of normal load at the contact point, um/N",
            wall="least wall thickness in a planar section, twice the distance from the section's medial ridge to its "
                 "boundary, mm; R11 asks 0.9 mm (0.8 mm for two 0.4 mm lines)",
            R4="least distance from the pad's rail grooves and screw-head notches to a magnet pocket, mm; R4 asks 2.0 mm"),
        changes_from_final=[
            f"DIP link {B.NECK:.2f} mm longer: a neck as wide as the cassette (|y| <= {B.Y_RIM}) between the bridge's underside "
            f"(z -67.30, unchanged) and the pad's mount face (z {B.TIP['z_mount']:.2f}); PIP to the pad's end "
            f"{lg['choice']['b']['pip_to_tip_mm']} mm (37.16 mm in the final)",
            f"the slot plane moved to x_F {B.X_SLOT:.2f} (t_behind {B.T_BEHIND} mm behind the block's magnets, "
            f"{B.X_SLOT + 1.05:.2f} mm behind the 11/22's), so the pad's grooves and notches keep R4's 2.0 mm from the 11/22's "
            "pockets; the cover, board and tongue moved dorsally with it",
            "the board keeps its whole 9 mm tab and its Qwiic connector J1, tab proximal; J1, the plug and the cable lie in a "
            "slot that runs from the neck's underside through the bridge's top; no wires are soldered, so the tongue has no "
            "wire exit and no tie block, and no header"]
        + ([f"a groove open to the back takes the Qwiic cable out of the plug toward the back, to +y at x {B.cp('cable_run_x'):g} "
            "mm and through the slotted arm's lower dorsal corner, and half the bundle deep up the arm's outer face to an "
            "anchor below the PIP axis"] if routed else [])
        + [
            f"the seat pocket is open at the tongue's distal end, so the board slides in along +z, J1 first, until the main "
            "square meets the proximal datum wall",
            f"four M2 x {B.SCREW_L:g} cheese-head screws (DIN 84) through cover and board form their thread in "
            f"{B.P['screw_thread_d']:g} mm holes of the tongue; no nuts, no nut pockets",
            "pads cut from the five-magnet 11/22 half-ellipsoid and the five-magnet block of v2/print"],
        levels_x_F=dict(pad_back=round(B.X_SLOT, 2), cover=[round(B.X_SLOT, 2), round(B.X_COMP, 2)],
                        board=[round(B.X_COMP, 2), round(B.X_SEAT, 2)], tongue=[round(B.X_SEAT, 2), round(B.X_TONGUE_BACK, 2)],
                        spine_back=B.X_SPINE_BACK, sensing_plane=round(B.X_SLOT + B.SPOT, 2)),
        lengthening=lg["placements"]["b_x3.85"] | dict(choice=lg["choice"]["b"], note=(
            f"the scan at the component face x_F 3.85 (this design: {B.X_COMP:.2f}); the slot plane x_F 1.35-3.85 all need "
            "9.75 mm")) | (dict(cable_route=(
                f"{B.NECK:g} mm: the cable turned toward the back out of the plug clears the MCP link by "
                f"{clr['qwiic_cable']['min_mm']} mm (R7 1.0 mm); with a 0.5 mm straight lead first, 0.676 mm. Concept a's "
                "turn to +y right over the plug, at b's plug (wires at x_F 2.40, a's at 5.90), would clear "
                + ", ".join(f"{r['clearance_mm']:.2f} mm at {r['neck_len']:g} mm with "
                            + ("no lead" if r["lead"] == 0 else f"a {r['lead']:g} mm lead") for r in ar) +
                " (v2/b/20261019-dip_link_v2b_cable_routes.json)")) if routed and clr.get("qwiic_cable") else {}),
        cable=(dict(route="out of the plug toward the PIP joint, at once a 2.0 mm turn toward the back over the board's tab "
                          f"end, a 2.0 mm turn to +y, a run along +y at x {B.cp('cable_run_x'):g} mm to the slotted arm, through "
                          "the arm's lower dorsal corner, a 2.0 mm turn up into the arm's outer face and a groove half the "
                          "bundle deep up the face to concept a's anchor below the PIP axis. The groove is open to the back, "
                          f"and over the plug and the leg toward the back (x up to {B.cable_open_top_x():.2f} mm) also to "
                          "the top, where the bridge's top lies less than 0.8 mm above it; the cable is laid in with its far "
                          "plug on",
                    open_top_to_x_mm=round(B.cable_open_top_x(), 2),
                    bundle_d_mm=B.cp("cable_d"), bend_r_mm=B.cp("cable_r"), lead_mm=B.cp("cable_lead"),
                    groove_clearance_mm=B.cp("cable_clear"), run_x_mm=B.cp("cable_run_x"),
                    length_plug_to_anchor_mm=round(B.cable_length(), 2), clearance=clr.get("qwiic_cable"), free_span=span,
                    range_of_motion="none lost: the cable runs inside the DIP link or outside y 17.40 to its anchor, and the "
                                    "free span to the MCP link is longest at the PIP angle given, so a cable laid at that "
                                    "length never pulls on the joint at -18 deg (back extension) or 92 deg")
               if routed else None),
        qwiic=dict(kept="J1 (JST SH SM04B-SRSS-TB) on the whole tab, tab proximal, mating face toward the PIP joint",
                   boxes_F={n: [list(np.round(lo, 2)), list(np.round(hi, 2))] for n, (lo, hi) in j1b.items()},
                   clearance=clr.get("qwiic_connector")),
        clearance=dict(per_part={k: dict(min_mm=v["min_mm"], exact=v.get("exact"), pip_deg=v.get("pip_deg"), part=v.get("part"))
                                 for k, v in clr.items()},
                       min_rigid_mm=worst[1]["min_mm"], min_rigid_part=worst[0],
                       note="values of 1.6 mm or more are lower bounds (only points within reach of the links are turned); "
                            "arms_kept are the unchanged clevis arms of fingertip_mount_DIP_link.stl"),
        walls=walls, r4=r4,
        sensing=dict(
            design_point=dict(layout="five_d3x2_d5 with its corners at the printed 7.4 mm (five_z7p4)", t_behind_mm=B.T_BEHIND,
                              force_resolution_gain_N=d5["gain_noise"]["centre_force_N"],
                              force_resolution_gain7_N=d5["gain7_noise"]["centre_force_N"],
                              gain_sel=d5["gain_noise"]["gain_sel"], B_U1_mT=d5["gain_noise"]["B_U1_mT"],
                              E_star_MPa=round(fe5["E_best_MPa"], 2), approach_1N_um=round(fe5["per_force"]["1"]["delta_um"], 1),
                              magnet_travel_um_per_N=d5["gain_noise"]["magnet_travel_um_per_N"]),
            whole_block_touch=dict(force_resolution_gain_N=t5["gain_noise"]["centre_force_N"],
                                   E_star_MPa=round(fe5w["E_best_MPa"], 2),
                                   approach_1N_um=round(fe5w["per_force"]["1"]["delta_um"], 1)),
            static_fields={k: dict(B_U1_mT=v["B_U1_mT"], B_max_component_mT=v["B_max_component_mT"], gain_sel=v["gain_sel"],
                                   chip_to_magnet_surface_mm=min(v["chip_to_magnet_surface_mm"]),
                                   chip_to_magnet_centre_mm=min(v["chip_to_magnet_centre_mm"])) for k, v in stat.items()},
            chip_to_magnet_mm={k: B.chip_magnet_distances(k) for k in B.TIPS},
            note="force resolution of the block pad from the coarse finite elements of the five-magnet block cut at the slot "
                 "plane (scripts/dip_link_concept_b.py fe, concept_b/fe/fem_coarse.jsonl); the five-magnet 11/22 has no finite "
                 "elements, so only its static field, gain and chip-to-magnet distance"),
        mass_g=mass, prep=prep, step=step,
        print_projects={k: project_summary(v) for k, v in prj.items() if not k.endswith("_pla_bambu")},
        print_pla_bambu=prj.get(f"{pf}_pla_bambu"),
        parts_list=[
            dict(item=f"DIP link, v2 ({B.NECK:g} mm longer, J1 slot, " + ("cable groove, " if routed else "")
                      + "1.8 mm screw holes)", material="PLA (Bambu Studio: print-oriented STL)", qty=1,
                 file=(bambu.get("dip_link") or {}).get("file", f"v2/b/cad/{pf}_dip_link.stl"),
                 step=f"v2/b/cad/{pf}_dip_link_new_body.step (new body, exact) and {pf}_dip_link.step.zip (fused with the arms)"
                      + (f"; {pf}_cable_groove_cutter.step" if routed else "")),
            dict(item="Cover (web as thick as the chips, two dovetail rails)", material="PLA, solid (Bambu Studio)", qty=1,
                 file=(bambu.get("cover") or {}).get("file", f"v2/b/cad/{pf}_cover.stl"), step=f"v2/b/cad/{pf}_cover.step"),
            dict(item="Pad, 85A, five magnets: 11/22 half-ellipsoid (or the block)", material="Siraya Tech TPU 85A, 30 % gyroid",
                 qty=1, file=f"v2/b/print/{pf}_pad_ell11_5_print.stl",
                 step=f"v2/b/cad/{pf}_pad_ell11_5.step.zip (faceted)"),
            dict(item="Rail fit coupon: a PLA strip with the two rails and 85A blocks with the grooves at 0, 0.05, 0.10 mm per "
                      "flank", qty=1, file=f"v2/b/print/{pf}_fit_coupon_rails_pla_bambu.stl (Bambu) and "
                                           f"{pf}_fit_coupon_grooves_85a_t2.3mf (XL, tool 2)"),
            dict(item="N52 disc magnet 3 x 2 mm, axially magnetized", qty=5),
            dict(item="ReSkin 5X board as supplied (J1 kept, tab not cut)", material="FR-4 0.6 mm", qty=1),
            dict(item=f"M2 x {B.SCREW_L:g} cheese-head screw, DIN 84 / ISO 1207", material="brass (R10: no ferromagnetic "
                 "fastener within 15 mm of a chip; A2 stainless can be weakly magnetic after cold forming)", qty=4),
            dict(item="Qwiic cable (JST SH 4-pin) from J1 to the controller", qty=1)],
        assembly=[
            "Print the rail fit coupon first and choose the groove interference (0, 0.05 or 0.10 mm per flank).",
            "Print the DIP link and the cover in PLA on the Bambu printer (print-oriented STLs, settings in v2/README.md) and "
            "the 85A pad with its pauses on the XL (press each magnet to the floor, north pole toward the palmar side).",
            "Plug the Qwiic cable into J1. Thread the cable up through the slot from the neck's underside until it leaves the "
            "bridge's top.",
            "Slide the board proximally along the tongue's seat, chips toward the palmar side and J1 first into the slot, until "
            "the main square meets the proximal datum wall; push it against the -y datum wall.",
            "Lay the cover on the board (windows over the parts, rails palmar) and drive the four M2 screws from the palmar side "
            "through cover and board into the tongue's 1.8 mm holes; they form their thread; tighten crosswise, snug."]
        + (["Lay the cable into its groove: bend it toward the back right at the plug into the groove that is open to the top "
            "over the plug, then toward the slotted arm, pressing it in from the back along the groove, around the arm's lower "
            "corner and up the groove on the arm's outer face to the anchor. A strip of Kapton tape over the open groove keeps "
            "the first turn down; a drop of CA glue or a strip of Kapton tape at the anchor keeps a pull off J1.",
            f"Tie the cable to the MCP link's side plate {span['span_mm']['max'] + 1.0:.0f} mm from the anchor (its longest "
            f"span, at PIP {span['span_mm']['longest_at_pip_deg']:.0f} deg, plus 1 mm), so it never pulls on the joint."]
           if routed else
           ["Route the cable from the bridge's top along the link's dorsal side to y > 17.4 mm and across the PIP joint outside "
            "the slotted arm; fix it to the DIP link before the joint so a pull does not reach J1."])
        + ["Press the pad onto the cover: notches over the screw heads, mount face against the neck, until both rails snap in.",
           "Read the zero-load field of the five chips."],
        first_print_checks=[
            "J1 with the plug mated slides through the slot from the neck's underside without touching its walls",
            "The M2 screws thread into the 1.8 mm holes by hand and clamp the board without play",
            "The pad's grooves snap onto the rails and the pad does not lift off the cover under a 3 N press",
            f"Over the full PIP range by hand, nothing touches the MCP link or the PIP servo (the least clearance is "
            f"{worst[1]['min_mm']} mm, {worst[0]})"]
        + (["The cable stays in its groove over the full PIP range, back extension (-18 deg) included, and the joint turns as "
            "freely with the cable tied as without it"] if routed else []),
        open_items=[f"Mass {mass['total_without_pad']} g without the pad at the link's sliced density (R12 7.6 g): the neck is "
                    f"a {B.NECK:g} mm block, 21 mm wide and 20 mm deep",
                    "Force resolution of the five-magnet 11/22 pad needs its finite elements"]
                   + (["The bundle is modelled as 2.0 mm round and turns right at the plug's rear face; measure the cable's "
                       "four wires and the groove's fit on the first print"] if routed else
                      ["The cable's route past the PIP joint beyond its first 3 mm"]),
        files=dict(cad=sorted(str(f.relative_to(REPO)) for f in B.CAD_DIR.glob("*")),
                   print=sorted(str(f.relative_to(REPO)) for f in (B_DIR / "print").glob("*")),
                   media=sorted(str(f.relative_to(REPO)) for f in (B_DIR / "media").glob("*.png"))),
        when=time.strftime("%Y-%m-%d %H:%M"))
    (B_DIR / "evaluation.json").write_text(json.dumps(ev, indent=1, default=float) + "\n")
    print("->", (B_DIR / "evaluation.json").relative_to(REPO), flush=True)


def cmd_b(a):
    for stage in a.stages:
        t0 = time.time()
        print(f"== b: {stage}", flush=True)
        if stage == "build":
            run_b("prep")
            run_b("cad")
        elif stage == "check":
            run_b("clearance", "--force")
        elif stage == "sense":
            run_b("sensing", python=REPO / ".venv/bin/python")
        elif stage == "render":
            run_b("render")
            b_section()
            B = b_module()
            if B.cable_on():
                b_cable_section(B, json.loads((B.WORK / "clearance.json").read_text()))
        elif stage == "step":
            b_step()
        elif stage == "print":
            b_print()
        elif stage == "docs":
            b_docs()
        print(f"== b: {stage} [{time.time() - t0:.0f} s]", flush=True)


# ----------------------------------------------------------------------------------------------- summary
def cmd_summary(a):
    """v2/20261018-v2_summary.json: what to print first and each design's numbers, from v2/a and v2/b (evaluation.json and
    the print projects' records)."""
    ev = {k: json.loads((V2 / k / "evaluation.json").read_text()) for k in ("a", "b")}
    prj = {k: {f.stem: project_summary(json.loads(f.read_text())) for f in sorted((V2 / k / "print").glob("*.json"))}
           for k in ("a", "b")}
    lg = json.loads((V2 / f"{DATE}-v2_lengthening.json").read_text())

    def item(k, stem, what):
        r = prj[k][stem]
        return dict(what=what, project=r["project"], print_time=r["print_time"], filament_g=r["filament_g"],
                    pauses_mm=r["pauses_mm"], checks_pass=r["checks_pass"])

    def bambu(k, pfx, part, what):                 # a print-oriented PLA STL for Bambu Studio (owner, 2026-10-19 14:15)
        rec = json.loads((V2 / k / "print" / f"{pfx}_pla_bambu.json").read_text())
        r = rec["parts"][part]
        return dict(what=what, project=r["file"], print_time="Bambu Studio", filament_g=None, pauses_mm=[],
                    checks_pass=bool(r["watertight"] and r["bodies"] == 1), settings=rec["settings"][part])
    import dip_link_params as DP
    neck = {k: DP.load(f)[1]["neck_len"] for k, f in (("a", PARAMS_A), ("b", PARAMS_B))}
    pa, pb = "20261018-dip_link_v2a", "20261018-dip_link_v2b"
    order = [
        item("a", f"{pa}_fit_coupon_keys_pla_t3", "a: PLA key sections with detents (tool 3)"),
        item("a", f"{pa}_fit_coupon_channels_85a_t2", "a: 85A channel blocks at 0, 0.05, 0.10 mm per flank (tool 2); choose the "
                                                       "channel interference"),
        bambu("b", pb, "rails", "b: PLA strip with the cover's two rails (Bambu)"),
        item("b", f"{pb}_fit_coupon_grooves_85a_t2", "b: 85A blocks with the pad's grooves at 0, 0.05, 0.10 mm per flank "
                                                      "(tool 2); choose the groove interference"),
        bambu("a", pa, "dip_link", "a: DIP link (PLA, Bambu)"), bambu("a", pa, "clamp_plate", "a: clamp plate (PLA, Bambu)"),
        bambu("a", pa, "pin_fit_coupon", "a: pin-fit coupon (PLA, Bambu)"),
        item("a", f"{pa}_print_tip_ell11_5_channels_85a_t2_g30", "a: 11/22 half-ellipsoid tip with the key channels and five "
                                                                 "magnets (85A, tool 2)"),
        bambu("b", pb, "dip_link", "b: DIP link (PLA, Bambu)"), bambu("b", pb, "cover", "b: cover (PLA, Bambu)"),
        item("b", f"{pb}_print_pad_ell11_5_85a_t2_g30", "b: 11/22 half-ellipsoid pad with five magnets (85A, tool 2)")]
    alt = [item("a", f"{pa}_print_tip_block5_channels_85a_t2_g30", "a: five-magnet block tip with the key channels"),
           item("b", f"{pb}_print_pad_block5_85a_t2_g30", "b: five-magnet block pad")]
    ca, cb = ev["a"]["clearance"]["per_part"], ev["b"]["clearance"]["per_part"]
    designs = dict(
        a=dict(pip_to_tip_mm=round(PIP_TO_TIP + neck["a"], 2), lengthening_mm=neck["a"],
               least_clearance_mm=dict(dip_link=ca["dip_link_new"]["min_mm"], j1_plug=ca["qwiic_connector"]["min_mm"],
                                       cable=(ca.get("qwiic_cable") or {}).get("min_mm")),
               block5_force_resolution_N=ev["a"]["sensing"]["block5_force_resolution_30_N"],
               static_field_U1_mT={k: v["B_U1_mT"] for k, v in ev["a"]["sensing"]["static_fields"].items()},
               gain_sel={k: v["gain_sel"] for k, v in ev["a"]["sensing"]["static_fields"].items()},
               chip_to_magnet_surface_mm={k: v["chip_to_magnet_surface_mm"] for k, v in ev["a"]["sensing"]["static_fields"].items()},
               mass_without_tip_g=ev["a"]["mass_g"]["total_without_tip"], open_items=ev["a"]["open_items"]),
        b=dict(pip_to_tip_mm=round(PIP_TO_TIP + neck["b"], 2), lengthening_mm=neck["b"],
               least_clearance_mm=dict(dip_link=cb["dip_link_new_body"]["min_mm"], j1_plug=cb["qwiic_connector"]["min_mm"],
                                       cable=(cb.get("qwiic_cable") or {}).get("min_mm")),
               block5_force_resolution_N=ev["b"]["sensing"]["design_point"]["force_resolution_gain_N"],
               static_field_U1_mT={k: v["B_U1_mT"] for k, v in ev["b"]["sensing"]["static_fields"].items()},
               gain_sel={k: v["gain_sel"] for k, v in ev["b"]["sensing"]["static_fields"].items()},
               chip_to_magnet_surface_mm={k: v["chip_to_magnet_surface_mm"] for k, v in ev["b"]["sensing"]["static_fields"].items()},
               mass_without_tip_g=ev["b"]["mass_g"]["total_without_pad"], open_items=ev["b"]["open_items"]))
    out = dict(
        title="Printable v2 of the DIP-link designs a and b: what to print first",
        terms=dict(pip_to_tip="PIP axis (z_F -51.55) to the tip's end, mm; 37.16 mm on the current hand (the hand model needs "
                              "the new length)",
                   least_clearance="least signed distance to the MCP link and the PIP servo over PIP -18 to 92 deg, mm; R7 "
                                   "asks 0.5 mm",
                   force_resolution="Cramer-Rao bound of the normal force at the contact point, five-magnet block, 30 % "
                                    "gyroid, at the gain the field needs, N",
                   static_field_U1="field at the centre chip U1 from the tip's magnets at rest, mT; gain_sel is the highest "
                                   "MLX90393 gain whose range covers 1.25 times the largest component"),
        print_first=order, alternatives=alt,
        why_this_order=("The four coupons (about 35 min) set the interference of a's channels and b's grooves before a tip or "
                        "pad takes magnets. a comes before b: it is the judge's final with the owner's 18:45 changes, and its "
                        "tip is the whole 11/22 shape with two channels, while b's pad is the 11/22 cut 3.95 mm behind its "
                        "magnets. The block versions of both are the alternatives."),
        designs=designs, settings="each project carries its printer settings (Prusa XL 5T, PLA on tool 3 at 0.1 mm layers, "
                                  "Siraya 85A on tool 2 at 0.2 mm layers, 30 % gyroid, M601 pauses at the 2.9 mm seats)",
        files=dict(a="docs/experiments/20261018-dip_link_board_mount/v2/a/", b="docs/experiments/20261018-dip_link_board_mount/v2/b/",
                   tips="docs/experiments/20261018-dip_link_board_mount/v2/print/",
                   lengthening=f"docs/experiments/20261018-dip_link_board_mount/v2/{DATE}-v2_lengthening.json"),
        when=time.strftime("%Y-%m-%d %H:%M"))
    dst = V2 / f"{DATE}-v2_summary.json"
    dst.write_text(json.dumps(out, indent=1, default=float) + "\n")
    for r in order + alt:
        print(f"{r['what'][:70]:70s} {str(r['print_time']):>12s} {r['filament_g']} g pauses {r['pauses_mm']} pass {r['checks_pass']}")
    print("->", dst.relative_to(REPO), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("lengthen")
    s.add_argument("--placements", nargs="+", default=["a=7.35", "b=1.35", "b_x2.35=2.35", "b_x3.85=3.85"],
                   type=lambda t: (t.split("=")[0], float(t.split("=")[1])),
                   help="name=x_F of the board's component face")
    s = sub.add_parser("tips")
    s.add_argument("--no-slice", action="store_true")
    sub.add_parser("merge")
    s = sub.add_parser("a", help="concept a v2 under v2/a/: build, step, check, sense, render, print")
    s.add_argument("stages", nargs="+", choices=["build", "step", "check", "sense", "render", "print", "docs"])
    s = sub.add_parser("b", help="concept b v2 under v2/b/: build, step, check, sense, render, print, docs")
    s.add_argument("stages", nargs="+", choices=["build", "step", "check", "sense", "render", "print", "docs"])
    sub.add_parser("summary", help="v2/20261018-v2_summary.json: what to print first, each design's numbers")
    sub.add_parser("coupons", help="v2/coupons/: concept a's snap-fit sweep, board carrier with a clamp plate, window gauges")
    s = sub.add_parser("pads", help="b's pads at $DIPB_PARAMS, sliced with their pauses (for dip_link_params.py export)")
    s.add_argument("--out", required=True)
    s.add_argument("--work", required=True)
    a = ap.parse_args()
    {"lengthen": cmd_lengthen, "tips": cmd_tips, "merge": lambda a: merge(), "a": cmd_a, "b": cmd_b,
     "summary": cmd_summary, "pads": cmd_pads, "coupons": cmd_coupons}[a.cmd](a)


if __name__ == "__main__":
    main()
