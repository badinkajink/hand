#!/usr/bin/env python3
"""Bambu Lab P1S PLA plates of 2026-10-20: the UR5e calibration mount v2 with its two indenters and three covers of DIP
link b v3 at three wall clearances (plate "calibration"), and DIP link b v3 (plate "link").

Owner, 2026-10-20, after printing v1 of the mount and v2 of the DIP links: the UR5e calibration comes first, the fingers go
on the hand in a week or more. The covers are cover W00, W10 and W20: concept b v3's cover with its side walls 0.00, 0.10
and 0.20 mm outside the pad's flank on each side, told apart by one, two and three notches in the distal edge; they fit
the DIP link b as printed (v2) as well as v3. The owner tried them with pad B00 ("i think w00 works fine", 2026-10-20), so
the design and the link plate take W00. Object settings as on the plate of 2026-10-19 (scripts/v2_ur5_pla_plate.py):
the DIP link at 0.1 mm with tree supports from the plate, the covers solid at 0.1 mm, the mount and the indenters at
0.15 mm with five walls and 40 % gyroid; a 5 mm brim on every part (Bambu Studio's command line ignores a brim set per
object), which also keeps the thin covers flat. Output: docs/experiments/20261020-ur5_calibration_mount_v2/print/.
Arguments name the plates to build (default: fork_and_indenters and link; the calibration plate of 15:40 only when named).
Plate "fork_and_indenters" (19:10): the standoff-fork with walls around both legs of the DIP link and the two indenters with
keyed homing posts, at Bambu Studio's 0.20 mm Standard (2 walls, 15 % grid; owner). Plate "tip_caps" (20:36): caps TP and
TW over b v3's tip end (scripts/dip_b_tip_cap.py), solid at 0.1 mm, into v3/b/print/. Plate "solid_covers_caps2" (2026-10-21):
the solid-wall covers I05, I10, I15 and the caps TP2, TW2, solid at 0.1 mm, into v3/b/print/. Plate "heights" (2026-10-21
16:20): covers I15 with walls 4, 3 and 2 mm tall and caps with side blocks 4, 3 and 2 mm tall at 0.15 mm (scripts/dip_b_v4.py
heights), into v4/b/print/; plate "link_v4": DIP link b v4 (recessed bridge, smaller column), into v4/b/print/.

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
V4 = REPO / "docs/experiments/20261018-dip_link_board_mount/v4/b"
UR5 = REPO / "docs/experiments/20261020-ur5_calibration_mount_v2/cad"
V3M = REPO / "docs/experiments/20261021-ur5_calibration_mount_v3"
UP3 = "20261021-ur5_calibration_mount_v3"
UP = "20261020-ur5_calibration_mount_v2"
COVERS = {"W00": 0.0, "W10": 0.10, "W20": 0.20}        # wall clearance per cover, mm
# owner, 2026-10-20 19:10: "we have a super dense infill on these mount parts. i think across the board it can be a standard
# sparse 15% grid infill since none of these are super fine or small parts": Bambu Studio's 0.20 mm Standard for the mount's
# parts (the DIP link and its cover keep theirs: walls under 1 mm print nearly solid anyway)
STANDARD = dict(layer_height="0.2", wall_loops="2", top_shell_layers="5", bottom_shell_layers="3",
                sparse_infill_density="15%", sparse_infill_pattern="grid", enable_support="0")


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
        covers=out, settings=V.SOLID, note="concept b v3's cover at three wall clearances; the design has 0.00 mm (W00, owner "
                                           "2026-10-20: 'i think w00 works fine')"), indent=1,
        default=float) + "\n")
    return out


R_ADAPTER = [[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]     # tool face on the bed, the dowel up
PLATES = {
    "calibration": [
        ("U2", "UR5 standoff-fork v2", UR5 / f"{UP}_standoff_fork.stl", np.eye(3), V.MOUNT_SOLID),
        ("U1", "UR5 adapter v2", UR5 / f"{UP}_adapter.stl", R_ADAPTER, V.MOUNT),
        ("U3", "UR5 clamp plate", UR5 / f"{UP}_clamp_plate.stl", V.R_UR5["clamp_plate"], dict(V.MOUNT_SOLID, wall_loops="4")),
        ("U4", "UR5 sphere indenter v2", UR5 / f"{UP}_indenter_sphere.stl", np.eye(3), V.INDENTER),
        ("U5", "UR5 rod indenter v2", UR5 / f"{UP}_indenter_rod.stl", np.eye(3), V.INDENTER),
        ("W00", "b v3 cover W00", V3 / "print/20261020-dip_link_v3b_cover_W00_pla_bambu.stl", None, V.SOLID),
        ("W10", "b v3 cover W10", V3 / "print/20261020-dip_link_v3b_cover_W10_pla_bambu.stl", None, V.SOLID),
        ("W20", "b v3 cover W20", V3 / "print/20261020-dip_link_v3b_cover_W20_pla_bambu.stl", None, V.SOLID)],
    "link": [
        ("B3", "b v3 DIP link", V3 / "print/20261020-dip_link_v3b_dip_link_pla_bambu.stl", None, V.LINK),
        ("C00", "b v3 cover W00", V3 / "print/20261020-dip_link_v3b_cover_W00_pla_bambu.stl", None, V.SOLID)],
    # 19:10: the parts that changed after the owner's 18:55 notes (the fork's walls around both legs, the keyed homing), at
    # STANDARD; the adapter and the clamp plate of the calibration plate are unchanged
    "fork_and_indenters": [
        ("U2", "UR5 standoff-fork v2, walls around the legs", UR5 / f"{UP}_standoff_fork.stl", np.eye(3), STANDARD),
        ("U4", "UR5 sphere indenter v2, keyed post", UR5 / f"{UP}_indenter_sphere.stl", np.eye(3), STANDARD),
        ("U5", "UR5 rod indenter v2, keyed post", UR5 / f"{UP}_indenter_rod.stl", np.eye(3), STANDARD)],
    # 20:36: the two caps over the tip end of b v3 (scripts/dip_b_tip_cap.py), end plate on the bed, solid at 0.1 mm like the
    # cover (1.2 mm walls, a 2 mm pin)
    "tip_caps": [
        ("TP", "b v3 tip cap TP, plate, ears, pin", V3 / "print/20261020-dip_link_v3b_tip_cap_TP_pla_bambu.stl", None, V.SOLID),
        ("TW", "b v3 tip cap TW, with side walls", V3 / "print/20261020-dip_link_v3b_tip_cap_TW_pla_bambu.stl", None, V.SOLID)],
    # 2026-10-21: the solid-wall covers at 0.05-0.15 mm interference (proximal heads 3 mm up) and the second caps
    # (scripts/dip_b_tip_cap.py covers2), solid at 0.1 mm like the covers before them; the covers' 3-4 mm walls take six
    # perimeters (with four, 100 % zig-zag left them at a fill ratio of 0.85 against the W covers' 0.875)
    "solid_covers_caps2": [
        (cid, f"b v3 cover {cid}, solid walls, {n} notch{'es' if n > 1 else ''}", V3 / f"print/20261021-dip_link_v3b_cover_{cid}_pla_bambu.stl",
         None, dict(V.SOLID, wall_loops="6")) for n, cid in enumerate(("I05", "I10", "I15"), 1)] + [
        ("TP2", "b v3 tip cap TP2", V3 / "print/20261021-dip_link_v3b_tip_cap_TP2_pla_bambu.stl", None, V.SOLID),
        ("TW2", "b v3 tip cap TW2, solid blocks", V3 / "print/20261021-dip_link_v3b_tip_cap_TW2_pla_bambu.stl", None, V.SOLID)],
    # 2026-10-21 16:20: the height sweep at the owner's 0.15 mm (scripts/dip_b_v4.py heights), the height engraved on each part
    "heights": [
        (f"I15H{h}", f"b cover I15, walls {h} mm, '{h}' on the wall tops", V4 / f"print/20261021-dip_link_v4b_cover_I15H{h}_pla_bambu.stl",
         None, dict(V.SOLID, wall_loops="6")) for h in (4, 3, 2)] + [
        (f"TW15H{h}", f"b tip cap, blocks {h} mm at 0.15 mm, '{h}' on the plate", V4 / f"print/20261021-dip_link_v4b_tip_cap_TW15H{h}_pla_bambu.stl",
         None, dict(V.SOLID, wall_loops="6")) for h in (4, 3, 2)],   # four walls left TW15H2 at a fill ratio of 0.845
    # DIP link b v4: the bridge's palmar edge recessed to a smaller column (params/20261021-v4_concept_b.json), as the v3 link
    "link_v4": [("B4", "b v4 DIP link, recessed bridge", V4 / "print/20261021-dip_link_v4b_dip_link_pla_bambu.stl", None, V.LINK)],
    # 2026-10-21 17:54: mount v3 for M6 x 20 (scripts/ur5_calibration_mount.py with UR5M_REV=v3): the adapter 15 mm thick, the
    # heads captive in the foot with key holes from above; STANDARD as the fork of 19:10, six walls on the adapter so the
    # plastic under each M6 head is solid down to the HEX-E, four on the fork for its M3 seats
    "mount_v3": [
        ("U1", "UR5 adapter v3, 15 mm, M6 x 20", V3M / f"cad/{UP3}_adapter.stl", R_ADAPTER, dict(STANDARD, wall_loops="6")),
        ("U2", "UR5 standoff-fork v3, captive M6 heads", V3M / f"cad/{UP3}_standoff_fork.stl", np.eye(3),
         dict(STANDARD, wall_loops="4"))],
}
DEFAULT_PLATES = ("fork_and_indenters", "link", "tip_caps", "solid_covers_caps2", "heights", "link_v4", "mount_v3")   # "calibration" when named
PLATE_OUT = {"tip_caps": (V3 / "print", "20261020-dip_link_v3b_p1s_pla_tip_caps"),     # other plates: OUT, 20261020-p1s_pla_<plate>
             "solid_covers_caps2": (V3 / "print", "20261021-dip_link_v3b_p1s_pla_solid_covers_caps2"),
             "heights": (V4 / "print", "20261021-dip_link_v4b_p1s_pla_heights"),
             "link_v4": (V4 / "print", "20261021-dip_link_v4b_p1s_pla_link"),
             "mount_v3": (V3M / "print", f"{UP3}_p1s_pla")}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    covers()
    sf = OUT / "20261020-p1s_pla_plates.json"
    summary = json.loads(sf.read_text()) if sf.exists() else {}
    for plate, parts in PLATES.items():
        if plate not in (sys.argv[1:] or DEFAULT_PLATES):
            continue
        V.PARTS[:] = parts
        out_dir, pfx = PLATE_OUT.get(plate, (OUT, f"20261020-p1s_pla_{plate}"))
        V.OUT, V.WORK, V.PFX = out_dir, WORK / plate, pfx
        V.TITLE = {"calibration": "UR5 mount v2 (v1 fork, no key), indenters, b v3 covers W00-W20",
                   "link": "DIP link b v3, cover W00",
                   "fork_and_indenters": "UR5 mount v2: fork with walls around the legs, indenters with keyed posts",
                   "tip_caps": "DIP link b v3: caps TP and TW over the tip end",
                   "solid_covers_caps2": "DIP link b v3: solid-wall covers I05, I10, I15 and caps TP2, TW2",
                   "heights": "DIP link b: covers I15 with walls 4, 3, 2 mm and caps with blocks 4, 3, 2 mm at 0.15 mm",
                   "link_v4": "DIP link b v4: the bridge's palmar edge recessed to a smaller column",
                   "mount_v3": "UR5 mount v3: adapter and standoff-fork for M6 x 20, keys from above"}[plate]
        V.OWNER = ("2026-10-20 (owner): the UR5e calibration first; the 30 mm spigot off the adapter; the finger on the tool "
                   "axis; an exterior mold in place of the key; homing to the indenters; concept b with side walls on the "
                   "cassette, a column under the pad and an open cable channel"
                   + ("; 18:55: walls around both legs that clock the link, no set screw, a keyed homing slot and shaft; "
                      "19:10: standard 15 % grid infill" if plate == "fork_and_indenters" else "")
                   + ("; 19:48: 'the finger is getting popped out a lot ... a kind of T-shaped plate'; 20:36: 'sure we can go "
                      "with your recommendation' (ears under the distal board screws, a pin, caps with and without side walls)"
                      if plate == "tip_caps" else "")
                   + ("; 2026-10-21: 'make the side walls on the bottom of the cassette actually clamp a little', 'boost the bottom "
                      "screws up a little bit', 'more solid/better connected', 'beef those [tip caps] up as well'"
                      if plate == "solid_covers_caps2" else "")
                   + ("; 2026-10-21 16:20: 'the 0.15mm inset is still perfectly fine ... let's try to lower the heights on both the "
                      "top cover and the cassette side walls'" if plate == "heights" else "")
                   + ("; 2026-10-21 16:20: 'recess/sculpt this edge of the dip link the same way we did for the column ... make the "
                      "underlying column even smaller radius too'" if plate == "link_v4" else "")
                   + ("; 2026-10-21 17:54: 'i only have 20mm long m6 screws. so we need to pop everything up a little. there is "
                      "not much thread on the hex e ft sensor ... make the holes accessible from the top plate'"
                      if plate == "mount_v3" else ""))
        try:
            V.main()
        except SystemExit as e:
            print("plate", plate, "problems:", e, flush=True)
        rec = json.loads((V.OUT / f"{V.PFX}.json").read_text())
        summary[plate] = dict(project=rec["project"], time=rec["time"], filament_g=rec["filament_g"], problems=rec["problems"],
                              parts=[o["id"] for o in rec["objects"]])
    sf.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1), f"[{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
