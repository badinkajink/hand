#!/usr/bin/env python3
"""The 11/22 tips of pads 05R and 05R4 with snap-fit magnet slots, the tips that pads 05S and 05S4 are cut from.

Owner, 2026-10-26: "i want to try having the pockets have a little more bite to them ... have the pocket come up past the
diameter of the disc ... if we just extend the pocket up a little, we should still be able to press the magnet in, and it
should help retain it a good bit better." The slots of metric_magnet_prints.upright_slots close the 2.9 mm seat with its lower
half circle and run straight up, 2.9 mm apart, from the seat's centre; the 3.0 mm magnet sits 0.05 mm above that centre.
Here the circle runs --snap mm above its centre and the walls rise from there (metric_magnet_prints.SNAP): at 0.5 mm the
throat is 2.72 mm, so the magnet is pressed through it and held under it. Layouts, depth and seat are those of 05R
(dip_link_v2's five_y2p5_z9p8_s0p5) and 05R4 (dip_b_four_compact_tip's four_y2p5_z5).

    unset VIRTUAL_ENV
    ~/.claude/bin/resguard.sh run --mem 4G -- env NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 \\
        ~/miniconda3/envs/femtip/bin/python -u scripts/dip_b_snap_fit_tips.py

Outputs in docs/experiments/20261018-dip_link_board_mount/v4/b/print/: each tip's print STL (..._seat2p9_snap0p5.stl) and its
85A sheet, and the record v4/b/20261026-dip_link_v4b_snap_fit_tips.json. The pads and their XL projects come from
scripts/v3b_pad_b00_magnets_xl.py --magnets five_snap / four_snap (pad set v4_snap of dip_link_concept_b.PAD_SETS, which
names the 0.5 mm tips).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import scipy.spatial  # noqa: F401  (the environment's libstdc++ before manifold3d)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import dip_b_four_compact_tip as FC  # noqa: E402
import dip_link_v2 as V2  # noqa: E402
import metric_magnet_prints as MP  # noqa: E402
import sim2real_fingertip_prints as SP  # noqa: E402

LAYOUTS = {"five_y2p5_z9p8_s0p5": V2.five_layout, "four_y2p5_z5": FC.four_layout}
OWNER = ("2026-10-26: 'i want to try having the pockets have a little more bite to them ... have the pocket come up past the "
         "diameter of the disc ... if we just extend the pocket up a little, we should still be able to press the magnet in'")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snap", type=float, default=0.5, help="how far the seat's circle runs above its centre, mm")
    ap.add_argument("--no-slice", action="store_true")
    a = ap.parse_args()
    MP.DATE = SP.DATE = FC.DATE
    MP.SNAP = a.snap
    SP.OUT, SP.WORK = FC.PRINT, REPO / f"logs/{FC.DATE}-dip_link/snap_fit_tips"
    SP.WORK.mkdir(parents=True, exist_ok=True)
    SP.use_template()
    tag = f"_snap{a.snap:g}".replace(".", "p")
    rec = dict(owner=OWNER, snap_mm=a.snap, tip=FC.TIP, seat_mm=FC.SEAT, depth_mm=FC.DEPTH, tips={},
               script="scripts/dip_b_snap_fit_tips.py")
    for lay, fn in LAYOUTS.items():
        t0 = time.time()
        centres = fn(lay)
        body, part, pockets, gaps, mesh = MP.build(FC.TIP, centres, FC.SEAT, FC.DEPTH)
        if not MP.walls_pass(pockets, gaps):
            raise SystemExit(f"{lay}: the pocket walls fail the {MP.WALL_MIN} mm rule")
        stem = MP.name_of(FC.TIP, lay, FC.SEAT, FC.DEPTH).replace(f"{FC.DATE}-", f"{FC.DATE}-siraya85a_", 1) + tag
        stl = FC.PRINT / f"{stem}.stl"
        part.export(stl)
        pk = json.loads(json.dumps(pockets, default=float))
        rep = SP.write_and_check(part, pk, gaps, mesh, f"{stem}_g30", FC.TIP, lay, FC.DEPTH, 2, 30, body, centres,
                                 argparse.Namespace(seat=FC.SEAT, no_slice=a.no_slice))
        rep["stl"] = str(stl.relative_to(REPO))
        rep["snap_mm"] = a.snap
        rep["seat_status"] = "the owner's seat (2026-10-18: 2.9 mm)"
        (FC.PRINT / f"{stem}_g30.json").write_text(json.dumps(rep, indent=1, default=float))
        rec["tips"][lay] = dict(stem=str(stl.with_suffix("").relative_to(REPO)), throat_mm=pockets[0]["throat_mm"],
                                wall_min_mm=min(p["wall_min_mm"] for p in pockets),
                                gap_min_mm=min(g["gap_mm"] for g in gaps), pauses=rep.get("pauses"),
                                slice=rep.get("slice"), seconds=round(time.time() - t0, 1))
        print(lay, json.dumps({k: rec["tips"][lay][k] for k in ("throat_mm", "wall_min_mm", "gap_min_mm", "seconds")}),
              flush=True)
    rec["when"] = time.strftime("%Y-%m-%d %H:%M")
    dst = FC.V4 / f"{FC.DATE}-dip_link_v4b_snap_fit_tips.json"
    dst.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print("->", dst.relative_to(REPO), flush=True)


if __name__ == "__main__":
    main()
