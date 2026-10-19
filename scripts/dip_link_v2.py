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
    a = ap.parse_args()
    {"lengthen": cmd_lengthen, "tips": cmd_tips, "merge": lambda a: merge()}[a.cmd](a)


if __name__ == "__main__":
    main()
