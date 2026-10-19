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
    MED = A_DIR / "media"
    pf = DLF.PFX
    jb = {n: (np.array(lo), np.array(hi)) for n, lo, hi in A.j1_boxes_F()}
    cam = ((5.0, 0.0, A.Z_C + 5.0), 150, 228, -20)
    img = A._render(asm, cam, MED / f"{pf}_iso.png")
    DLF._annotate_final(img, cam, [
        ("Clamp plate", (L["x_c1"], -3.0, A.Z_C - 2.0), (0.03, 0.62)),
        ("M2 screw into a shoulder (1.8 mm hole)", (L["x_c1"] + P["m2_head_k"], -P["m2_y"], P["m2_z"]), (0.03, 0.44)),
        ("J1 with its plug, on the whole tab", ((jb["plug"][0][0] + jb["plug"][1][0]) / 2, 0.0, jb["plug"][1][2]), (0.97, 0.36)),
        ("DIP link 8.5 mm longer than the final", (L["x_cf"], 12.0, A.Z_MOUNT + 4.0), (0.97, 0.55)),
        ("Tip, 85A, five magnets", (-2.0, A.Y_HALF, A.Z_C - 2.0), (0.97, 0.78)),
        ("PIP servo", (21.95, -6.0, -52.0), (0.03, 0.20))],
        title="Concept a, v2, assembled at PIP 0 deg, dorsal side")
    ex = [("dip_link", f("dip_link"), C_["dip_link"], (0, 0, 0)),
          ("cs", ctx("fasteners_cs_screw"), C_["fasteners"], (-12, 0, 0)),
          ("tip", f(f"tip_{tip}_F"), C_["tip"], (0, 0, -18)),
          ("board", f("board"), C_["board"], (9, 0, 0)), ("qwiic", f("qwiic"), qcol, (9, 0, 0)),
          ("clamp_plate", f("clamp_plate"), C_["clamp_plate"], (18, 0, 0)),
          ("m2", ctx("fasteners_m2_screw"), C_["fasteners"], (27, 0, 0))]
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
    print("renders ->", MED.relative_to(REPO), flush=True)


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
             ("Clamp plate", DLF.fname("clamp_plate"), "#5b8fd9"), ("J1, plug, cable start", DLF.fname("qwiic"), "#f2f2ea"),
             ("Screws", DLF.fname("fasteners"), "#c8a33c")]
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
    axs[0].annotate("Plug and the cable's\nfirst 3 mm", xy=(5.9, jb["wire1"][1][2] - 0.5), xytext=(10.0, -46.5), fontsize=15,
                    arrowprops=dict(arrowstyle="->", lw=1.0))
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
    PP.build_project({"dip_link": DLF.fname("dip_link"), "clamp_plate": DLF.fname("clamp_plate"),
                      "pin_fit_coupon": DLF.PRINT / f"{pf}_pin_fit_coupon.stl"}, out, f"{pf}_print_link_clamp_coupon_pla_t3",
                     source=str(PARAMS_A.relative_to(REPO)))
    PP.build_fit_coupon(out)
    MP.DATE = SP.DATE = DATE
    SP.OUT, SP.WORK = out, A_WORK / "print"
    SP.use_template()
    for key, t in A.spec()["tips"].items():
        sheet = json.loads((REPO / t["sheet"]).read_text())
        part = trimesh.load(DLF.PRINT / f"{pf}_tip_{key.replace('.', 'p')}_print.stl", force="mesh")
        name = f"{pf}_print_tip_{key.replace('.', 'p')}_channels_85a_t2_g30"
        ns = argparse.Namespace(seat=SEAT, no_slice=False)
        body = part
        rep = SP.write_and_check(part, json.loads(json.dumps(sheet["pockets"])), sheet["pocket_gaps"], sheet["mesh"], name,
                                 sheet["tip"], sheet["layout"], sheet["magnet"]["top_depth_mm"], 2, 30, body,
                                 five_layout(sheet["layout"]), ns)
        rep["stl"] = str((DLF.PRINT / f"{pf}_tip_{key.replace('.', 'p')}_print.stl").relative_to(REPO))
        rep["note"] = "the v2 sheet's geometry with the two key channels of concept a cut (dip_link_concept_a.tip_cutter)"
        (out / f"{name}.json").write_text(json.dumps(rep, indent=1, default=float))
    print("print ->", out.relative_to(REPO), flush=True)


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
    ev = dict(
        title="DIP link concept a, v2: the board's own Qwiic connector kept on a DIP link 8.5 mm longer, M2 self-forming "
              "screws, five-magnet tips",
        date="2026-10-18", params=str(PARAMS_A.relative_to(REPO)), generator="scripts/dip_link_v2.py a (dip_link_params.apply_a "
              "on scripts/dip_link_concept_a.py and scripts/dip_link_board_mount.py)",
        owner=["'we should just go for 5x 3x2 magnets'", "'i do like the half ellipsoids most, the 11/22 one especially'",
               "'if we have to lengthen the fingertip to get everything to fit, including the qwiic port and the cable, that's "
               "fine'", "'for m3 screws, i use 2.75mm diameter holes, m2 i use 1.8'"],
        changes_from_final=[
            f"DIP link {vals['neck_len']:.1f} mm longer between its bridge and the tip (PIP to the tip's end "
            f"{lg['choice']['a']['pip_to_tip_mm']} mm, 37.16 mm in the final); the clevis arms and the bridge's top are the final's",
            "the board keeps its whole 9 mm tab and its Qwiic connector J1, tab proximal; the bridge has a slot for J1, the "
            "plug and the cable; no pigtail, no header on the clamp plate",
            "the two distal board screws are M2 x 4 countersunk (DIN 965), forming their thread in 1.8 mm holes of the clamp "
            "plate (no nuts); the two ear screws M2 x 5 into 1.8 mm holes of the shoulders (1.7 mm in the final)",
            "tips with five 3 x 2 mm magnets at the 2.9 mm seat: the 11/22 half-ellipsoid (corners y +-2.5 mm, rows 9.8 mm "
            "apart, the group 0.5 mm toward the mount face) and the block (five_z7p4), both with the key channels"],
        lengthening=lg["placements"]["a"] | dict(choice=lg["choice"]["a"]),
        clearance=dict(per_part={k: dict(min_mm=v.get("min_mm"), pip_deg=v.get("pip_deg"), part=v.get("part"))
                                 for k, v in clear.items()},
                       requirement="R7: 0.5 mm rigid; exact method of the specification over PIP -18..92 deg; values of 3 mm "
                                   "or more are lower bounds (only points within 3-4 mm of the links are turned)"),
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
                        for k, v in prj.items()},
        parts_list=[
            dict(item="DIP link, v2", material="PLA", qty=1, file=f"v2/a/cad/{pf}_dip_link.stl",
                 step=f"v2/a/cad/{pf}_dip_link_new_features.step (new features, exact) and {pf}_dip_link.step.zip (fused)"),
            dict(item="Clamp plate, v2 (1.8 mm holes, no nut pockets, no Qwiic pocket)", material="PLA or PETG, solid", qty=1,
                 file=f"v2/a/cad/{pf}_clamp_plate.stl", step=f"v2/a/cad/{pf}_clamp_plate.step"),
            dict(item="Pin-fit coupon", material="as the clamp plate", qty=1, file=f"v2/a/print/{pf}_pin_fit_coupon.stl"),
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
            "Print the PLA plate (DIP link, clamp plate, pin coupon) and the 85A tip with its pauses (press each magnet to the "
            "floor, north pole toward the palmar side).",
            "Plug the Qwiic cable into J1 before mounting the board: the plug enters J1 along the tab, from its end.",
            "Lay the board component side down on the window plate: chips in their windows, J1 and the plug in the bridge's "
            "slot, the cable out of the slot's top toward the PIP joint.",
            "Put on the clamp plate (pins through the board's proximal holes into the window plate).",
            "With the tip off, drive the two M2 x 4 countersunk screws from the palmar side through the window plate and the "
            "board's distal holes into the clamp plate's 1.8 mm holes; they form their thread; snug.",
            "Drive the two M2 x 5 screws through the clamp plate's ears into the shoulders' 1.8 mm holes; snug, the ears clear "
            "of the shoulders.",
            "Route the cable from the slot's top along the link's dorsal side to y > 17.4 mm and across the PIP joint outside "
            "the slotted arm; fix it to the DIP link before the joint so a pull does not reach J1.",
            "Slide the tip along +z onto the keys until both detents engage; read the zero-load field."],
        first_print_checks=["J1 with the plug mated fits the slot without touching its walls; the cable leaves the slot "
                            "freely", "The M2 screws thread into the 1.8 mm holes by hand and hold the board without play",
                            "Over the full PIP range by hand, nothing touches the MCP link or the PIP servo (the plug's "
                            "nearest approach is 0.52 mm at PIP 73.5 deg)"],
        open_items=[f"Mass {chk['mass_g']['total_without_tip']} g without the tip (R12 7.6 g): the 8.5 mm longer bridge is "
                    "solid between the clevis arms", "The window plate's web beside each distal M2 countersink is 0.72 mm "
                    "(R11 0.8 mm for two lines); M1.6 countersunk screws keep the final's 0.80 mm",
                    "Force resolution of the five-magnet 11/22 needs its finite elements",
                    "The cable's route past the PIP joint beyond its first 3 mm"],
        when=time.strftime("%Y-%m-%d %H:%M"))
    (A_DIR / "evaluation.json").write_text(json.dumps(ev, indent=1, default=float) + "\n")
    print("->", (A_DIR / "evaluation.json").relative_to(REPO), flush=True)


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
    a = ap.parse_args()
    {"lengthen": cmd_lengthen, "tips": cmd_tips, "merge": lambda a: merge(), "a": cmd_a}[a.cmd](a)


if __name__ == "__main__":
    main()
