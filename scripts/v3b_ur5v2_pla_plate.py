#!/usr/bin/env python3
"""Bambu Lab P1S PLA plates of 2026-10-20: the UR5e calibration mount v2 with its two indenters and three covers of DIP
link b v3 at three wall clearances (plate "calibration"), and DIP link b v3 (plate "link").

Owner, 2026-10-20, after printing v1 of the mount and v2 of the DIP links: the UR5e calibration comes first, the fingers go
on the hand in a week or more. The covers are cover W00, W10 and W20: concept b v3's cover with its side walls 0.00, 0.10
and 0.20 mm outside the pad's flank (design 0.10), told apart by one, two and three notches in the distal edge; they fit
the DIP link b as printed (v2) as well as v3. Object settings as on the plate of 2026-10-19 (scripts/v2_ur5_pla_plate.py):
the DIP link at 0.1 mm with tree supports from the plate, the covers solid at 0.1 mm, the mount and the indenters at
0.15 mm with five walls and 40 % gyroid; a 5 mm brim on every part (Bambu Studio's command line ignores a brim set per
object), which also keeps the thin covers flat. Output: docs/experiments/20261020-ur5_calibration_mount_v2/print/.

    unset VIRTUAL_ENV
    ~/.claude/bin/resguard.sh run --mem 5G --cpu 400 -- taskset -c 14-18 env UR5M_REV=v2 NUMPY_MADVISE_HUGEPAGE=0 \\
        OMP_NUM_THREADS=1 ~/miniconda3/envs/femtip/bin/python -u scripts/v3b_ur5v2_pla_plate.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("UR5M_REV", "v2")
os.environ["DIPB_PARAMS"] = str(REPO / "docs/experiments/20261018-dip_link_board_mount/params/20261020-v3_concept_b.json")
import dip_fit_print_plates as D  # noqa: E402
import dip_link_print_project as PP  # noqa: E402
import v2_ur5_pla_plate as V  # noqa: E402

OUT = REPO / "docs/experiments/20261020-ur5_calibration_mount_v2/print"
WORK = REPO / "logs/20261020-ur5_calibration_mount_v2/plates"
V3 = REPO / "docs/experiments/20261018-dip_link_board_mount/v3/b"
UR5 = REPO / "docs/experiments/20261020-ur5_calibration_mount_v2/cad"
UP = "20261020-ur5_calibration_mount_v2"
COVERS = {"W00": 0.0, "W10": 0.10, "W20": 0.20}        # wall clearance per cover, mm


def covers():
    """The fit covers as Bambu STLs (print orientation) in v3/b/print, with ID notches in the distal edge."""
    import trimesh
    import dip_link_concept_b as B
    import dip_link_v2 as L2
    K = B.MeshB()
    out = {}
    c0 = B.P.get("wall_clear")
    for k, (cid, c) in enumerate(COVERS.items()):
        B.P["wall_clear"] = c
        cov = B.cover_part(K)
        notches = [K.box((B.X_SLOT - 0.2, -3.0 + 1.6 * j, B.TIP["z_end"] - 0.1), (B.X_COMP + 0.2, -2.0 + 1.6 * j,
                                                                                    B.TIP["z_end"] + 0.8)) for j in range(k + 1)]
        tm = B.to_trimesh(K.cut(cov, notches))
        dst = V3 / "print" / f"20261020-dip_link_v3b_cover_{cid}_pla_bambu.stl"
        rec = L2.bambu_stl(tm, PP.R_CLAMP, dst)
        out[cid] = dict(rec, wall_clear_mm=c, notches=k + 1)
        print(cid, out[cid], flush=True)
    B.P["wall_clear"] = c0
    (V3 / "print" / "20261020-dip_link_v3b_covers_W00_W10_W20.json").write_text(json.dumps(dict(
        covers=out, settings=V.SOLID, note="concept b v3's cover at three wall clearances; the design has 0.10 mm"), indent=1,
        default=float) + "\n")
    return out


R_ADAPTER = [[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]     # tool face on the bed, the dowel up
PLATES = {
    "calibration": [
        ("U2", "UR5 standoff-fork v2", UR5 / f"{UP}_standoff_fork.stl", np.eye(3), V.MOUNT),
        ("U1", "UR5 adapter v2", UR5 / f"{UP}_adapter.stl", R_ADAPTER, V.MOUNT),
        ("U4", "UR5 sphere indenter v2", UR5 / f"{UP}_indenter_sphere.stl", np.eye(3), V.INDENTER),
        ("U5", "UR5 rod indenter v2", UR5 / f"{UP}_indenter_rod.stl", np.eye(3), V.INDENTER),
        ("W00", "b v3 cover W00", V3 / "print/20261020-dip_link_v3b_cover_W00_pla_bambu.stl", None, V.SOLID),
        ("W10", "b v3 cover W10", V3 / "print/20261020-dip_link_v3b_cover_W10_pla_bambu.stl", None, V.SOLID),
        ("W20", "b v3 cover W20", V3 / "print/20261020-dip_link_v3b_cover_W20_pla_bambu.stl", None, V.SOLID)],
    "link": [
        ("B3", "b v3 DIP link", V3 / "print/20261020-dip_link_v3b_dip_link_pla_bambu.stl", None, V.LINK),
        ("C10", "b v3 cover W10", V3 / "print/20261020-dip_link_v3b_cover_W10_pla_bambu.stl", None, V.SOLID)],
}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    covers()
    summary = {}
    for plate, parts in PLATES.items():
        V.PARTS[:] = parts
        V.OUT, V.WORK, V.PFX = OUT, WORK / plate, f"20261020-p1s_pla_{plate}"
        V.TITLE = {"calibration": "UR5 mount v2, indenters, b v3 covers W00-W20", "link": "DIP link b v3, cover W10"}[plate]
        V.OWNER = ("2026-10-20 (owner): the UR5e calibration first; the 30 mm spigot off the adapter; the finger on the tool "
                   "axis; an exterior mold in place of the key; homing to the indenters; concept b with side walls on the "
                   "cassette, a column under the pad and an open cable channel")
        try:
            V.main()
        except SystemExit as e:
            print("plate", plate, "problems:", e, flush=True)
        rec = json.loads((OUT / f"{V.PFX}.json").read_text())
        summary[plate] = dict(project=rec["project"], time=rec["time"], filament_g=rec["filament_g"], problems=rec["problems"],
                              parts=[o["id"] for o in rec["objects"]])
    (OUT / "20261020-p1s_pla_plates.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1), f"[{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
