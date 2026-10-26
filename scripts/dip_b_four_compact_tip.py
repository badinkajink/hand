#!/usr/bin/env python3
"""The 11/22 tip with four 3 x 2 mm magnets in a compact square centred on the board's centre chip U1, the tip that pad B05R4
is cut from.

Owner, 2026-10-26: "can you prepare a 3mf for the prusa xl 5t with the same 3 fingertips, but with only 4 magnet pouches
instead of [5]?", with the advice "Keep the four compact and symmetric about the centre chip. The field that's left after
cancelling grows with the magnets' spacing, roughly with its square." U1 lies under the contact point (y_F = 0,
z_F = -78.16 mm), so the square is centred there; the five-magnet tip of v2 has its centre magnet 0.5 mm proximal of it.
The pitch is the least the upright slots allow: the layouts of TRIES in order, the first whose pocket walls all pass
metric_magnet_prints.walls_pass (1.8 mm to the outer surface and between pockets). four_y<a>_z<p> puts the magnets at
y = +-a and the contact point's z +-p/2. Depth 7.5 mm under the apex and the 2.9 mm seat are those of the five-magnet tip, so
B05R4 differs from B05R in its magnets alone.

    unset VIRTUAL_ENV
    ~/.claude/bin/resguard.sh run --mem 4G -- env NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 \\
        ~/miniconda3/envs/femtip/bin/python -u scripts/dip_b_four_compact_tip.py

Outputs: the tip's print STL and its 85A sheet (sliced and checked as the v2 tips, scripts/sim2real_fingertip_prints.py) in
docs/experiments/20261018-dip_link_board_mount/v4/b/print/, and the record v4/b/20261026-dip_link_v4b_four_compact_tip.json.
The pad and its XL project come from scripts/v3b_pad_b00_magnets_xl.py --magnets four_compact (pad set v4_four of
dip_link_concept_b.PAD_SETS).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import scipy.spatial  # noqa: F401  (the environment's libstdc++ before manifold3d)
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import dip_link_geometry as G  # noqa: E402
import metric_magnet_prints as MP  # noqa: E402
import sim2real_fingertip_prints as SP  # noqa: E402
import tpu_tip_design as TD  # noqa: E402

DATE = "20261026"
V4 = REPO / "docs/experiments/20261018-dip_link_board_mount/v4/b"
PRINT = V4 / "print"
TIP = "tcd_ell7.4x8.5x11x22"
SEAT, DEPTH = 2.9, 7.5
TRIES = ["four_y2p45_z4p9", "four_y2p5_z4p9", "four_y2p5_z5", "four_y2p5_z5p4"]
U1_F = (0.0, -78.16)                     # the centre chip's (y_F, z_F): dip_link_concept_b's Z_CONTACT
OWNER = ("2026-10-26: 'can you prepare a 3mf for the prusa xl 5t with the same 3 fingertips, but with only 4 magnet pouches "
         "instead of [5]?'; 'Keep the four compact and symmetric about the centre chip. The field that's left after cancelling "
         "grows with the magnets' spacing, roughly with its square.'")


def four_layout(name):
    """Magnet centres (y, z) in tpu_tip_design's frame, proximal first: four_y<a>_z<p> -> y = +-a, z = Z_C +- p/2."""
    a, p = (float(s[1:].replace("p", ".")) for s in name[5:].split("_"))
    return sorted([(sy * a, TD.Z_C + sz * p / 2) for sy in (-1, 1) for sz in (-1, 1)], key=lambda c: (c[1], c[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-slice", action="store_true")
    a = ap.parse_args()
    MP.DATE = SP.DATE = DATE
    SP.OUT, SP.WORK = PRINT, REPO / f"logs/{DATE}-dip_link/four_compact_tip"
    SP.WORK.mkdir(parents=True, exist_ok=True)
    PRINT.mkdir(parents=True, exist_ok=True)
    SP.use_template()
    rows, lay = [], None
    for name in TRIES:
        t0 = time.time()
        body, part, pockets, gaps, mesh = MP.build(TIP, four_layout(name), SEAT, DEPTH)
        ok = MP.walls_pass(pockets, gaps)
        rows.append(dict(layout=name, walls_ok=bool(ok), wall_min_mm=min(p["wall_min_mm"] for p in pockets),
                         gap_min_mm=min(g["gap_mm"] for g in gaps), seconds=round(time.time() - t0, 1)))
        print(f"{name}: walls ok {ok}, least wall to the surface {rows[-1]['wall_min_mm']}, least gap between pockets "
              f"{rows[-1]['gap_min_mm']} [{rows[-1]['seconds']} s]", flush=True)
        if ok:
            lay = name
            break
    if lay is None:
        raise SystemExit("no layout of TRIES passes the walls")
    stl = PRINT / f"{MP.name_of(TIP, lay, SEAT, DEPTH).replace(f'{DATE}-', f'{DATE}-siraya85a_', 1)}.stl"
    part.export(stl)
    sheet_name = SP.name_of(TIP, lay, SEAT, DEPTH, 2, 30)
    pk = json.loads(json.dumps(pockets, default=float))
    rep = SP.write_and_check(part, pk, gaps, mesh, sheet_name, TIP, lay, DEPTH, 2, 30, body, four_layout(lay),
                             argparse.Namespace(seat=SEAT, no_slice=a.no_slice))
    rep["stl"] = str(stl.relative_to(REPO))
    rep["seat_status"] = "the owner's seat (2026-10-18: 2.9 mm)"
    (PRINT / f"{sheet_name}.json").write_text(json.dumps(rep, indent=1, default=float))
    # the magnets in F, their distances to U1 (equal when the square is centred on it) and the pitch
    top = 8.5 - DEPTH                                        # the magnets' palmar face in the tip frame (palmar face x = 8.5)
    cF = [G.tip_design_to_F([top - 1.0, y, z]) for y, z in four_layout(lay)]
    du1 = [round(math.hypot(c[1] - U1_F[0], c[2] - U1_F[1]), 3) for c in cF]
    ys, zs = sorted({round(c[1], 3) for c in cF}), sorted({round(c[2], 3) for c in cF})
    rec = dict(owner=OWNER, tip=TIP, seat_mm=SEAT, depth_mm=DEPTH, tried=rows, chosen=lay,
               centres_tip_frame=[[round(y, 3), round(z, 3)] for y, z in four_layout(lay)],
               centres_F=[np.round(c, 3).tolist() for c in cF], u1_F=list(U1_F), distance_to_u1_in_plane_mm=du1,
               pitch_mm=dict(across=round(ys[1] - ys[0], 3), along=round(zs[1] - zs[0], 3)),
               stl=rep["stl"], sheet=str((PRINT / f"{sheet_name}.json").relative_to(REPO)),
               pad_set_stem=str(stl.with_suffix("").relative_to(REPO)), wall_min_mm=rep.get("wall_min_mm"),
               pauses=rep.get("pauses"), slice=rep.get("slice"), when=time.strftime("%Y-%m-%d %H:%M"),
               script="scripts/dip_b_four_compact_tip.py")
    dst = V4 / f"{DATE}-dip_link_v4b_four_compact_tip.json"
    dst.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps({k: rec[k] for k in ("chosen", "pitch_mm", "distance_to_u1_in_plane_mm", "wall_min_mm", "pad_set_stem")}),
          flush=True)
    print("->", dst.relative_to(REPO), flush=True)


if __name__ == "__main__":
    main()
