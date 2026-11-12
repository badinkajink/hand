#!/usr/bin/env python3
"""The mount that holds the ChArUco board of scripts/hardware/charuco_board_print.py on the SR2 hand for the hand-eye
poses (owner, 2026-11-12: at the camera arm's fork). The camera arm (`d435_plate`, solid 000 of
assets/cad_v3/step/hand_v3_anthro_config.step, extracted to logs/20261104-cad_v3/extract/hand/000.stl) runs under the base
plate toward -x_P and ends past it in a notch for a 20 x 20 mm extrusion, with an M5 clearance hole beside the notch in
its 4 mm flange. The mount is a 20 x 20 mm post in the notch with a flange on top of the arm, bolted through that hole
(M5 bolt and nut), and 30 mm of post below the arm. Its -x_P face and its -y_P face each have a pair of M3 pilot holes
(self-tapping screws into PLA, 2.75 mm), 12 mm apart like the pair at the middle of the board's top edge, so the board
hangs from either face, facing -x_P or -y_P.

Frames: the notch and hole come from logs/20261104-cad_v3/extract/parts.json (solid 000: the notch's fillets of 1.2 mm
and the hole of 2.7 mm radius) in the CAD frame A, and P from assets/cad_v3/frames.json. `build` (femtip) writes the STL,
flange down, and checks the mount and the hung board against every solid of the hand's assembly in its modelled pose:
the shared volume with the camera arm (the post sits in the notch with POST_CLEAR per side, the flange on the arm's top)
and the least gap to every other solid.

    ~/.claude/bin/resguard.sh run --mem 3G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/envs/femtip/bin/python scripts/hardware/charuco_board_mount_print.py build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for _paths
import _paths  # noqa: E402,F401
import charuco_board_print as CB  # noqa: E402
import object_tracking_print as OP  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXTRACT = ROOT / "logs/20261104-cad_v3/extract"
FRAMES = ROOT / "assets/cad_v3/frames.json"
OUT = CB.OUT
STL = OUT / f"{CB.DATE}-charuco_board_mount_white.stl"
REPORT = OUT / f"{CB.DATE}-charuco_board_mount.json"
# the camera arm's end in A (mm): the notch's walls (x_A -14.2 open to 6.2, y_A 271.1 to 291.5), the arm's top z_A 0,
# its 10 mm and 4 mm thicknesses, the M5 clearance hole at (15.8, 281.5)
NOTCH_A = ((-14.2, 6.2), (271.1, 291.5))
ARM_TOP_A, ARM_THICK = 0.0, 10.0
HOLE_A, HOLE_D = (15.8, 281.5), 5.5
POST, POST_CLEAR, BELOW = 20.0, 0.2, 30.0     # mm: post section, its clearance per side in the 20.4 mm notch, length below the arm
POST_R = 1.5                                  # mm: the post's vertical edges, rounded past the notch's 1.2 mm fillets
FLANGE_T, FLANGE_PAD = 6.0, 7.0               # mm: flange thickness; its reach past the post on +-x_P and past the hole on +y_P
PILOT_D, PILOT_DEPTH = 2.75, 9.0              # mm: M3 pilot holes for self-tapping screws into PLA
PAIR_DX = CB.MOUNT_HOLE_DX
Z_HOLES = {"-x": 10.0, "-y": 20.0}            # mm above the post's bottom: the -x and -y pairs at different heights


def P_from_A():
    f = json.loads(FRAMES.read_text())["P"]
    return np.asarray(f["R_P_from_A"], float), np.asarray(f["t_P_from_A_mm"], float)


def geometry():
    """The notch, hole and arm top in P (mm)."""
    R, t = P_from_A()
    corners = np.array([[x, y, ARM_TOP_A] for x in NOTCH_A[0] for y in NOTCH_A[1]]) @ R.T + t
    lo, hi = corners.min(0), corners.max(0)
    hole = np.array([*HOLE_A, ARM_TOP_A]) @ R.T + t
    return {"notch_lo": lo[:2], "notch_hi": hi[:2], "top": float(lo[2]), "hole": hole[:2]}


def mount():
    """The mount in P, and the board hung from each face in P, as manifolds."""
    import manifold3d as M
    g = geometry()
    c = (g["notch_lo"] + g["notch_hi"]) / 2
    z0 = g["top"] - ARM_THICK - BELOW                                     # the post's bottom
    # rounded vertical edges clear the notch's 1.2 mm fillets, which a square corner meets despite the wall clearance
    section = M.CrossSection.square([POST, POST], True).offset(-POST_R, M.JoinType.Miter).offset(
        POST_R, M.JoinType.Round, 2.0, 32)
    post = M.Manifold.extrude(section, g["top"] - z0).translate([c[0], c[1], z0])
    fx = (c[0] - POST / 2 - FLANGE_PAD, c[0] + POST / 2 + FLANGE_PAD)
    fy = (c[1] - POST / 2, g["hole"][1] + FLANGE_PAD)
    flange = M.Manifold.cube([fx[1] - fx[0], fy[1] - fy[0], FLANGE_T]).translate([fx[0], fy[0], g["top"]])
    body = (post + flange) - M.Manifold.cylinder(FLANGE_T + 2, HOLE_D / 2, HOLE_D / 2, 48).translate(
        [g["hole"][0], g["hole"][1], g["top"] - 1])
    pilots, boards = [], {}
    for face, zh in Z_HOLES.items():
        z = z0 + zh
        for s in (-1, 1):
            if face == "-x":   # along +x_P from the -x face
                R = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
                p = (c[0] - POST / 2 - 1.0, c[1] + s * PAIR_DX / 2, z)
            else:              # along +y_P from the -y face
                R = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
                p = (c[0] + s * PAIR_DX / 2, c[1] - POST / 2 - 1.0, z)
            pilots.append(M.Manifold.cylinder(PILOT_DEPTH + 1.0, PILOT_D / 2, PILOT_D / 2, 32).transform(
                np.hstack([R, np.asarray(p)[:, None]])))
        # the board on that face: its top edge HOLE_IN above the pair, its back on the face
        PW, PH = CB.board_wh()[0] + 2 * CB.MARGIN, CB.board_wh()[1] + 2 * CB.MARGIN
        top = z + CB.HOLE_IN
        if face == "-x":
            boards[face] = M.Manifold.cube([CB.THICK, PW, PH]).translate([c[0] - POST / 2 - CB.THICK, c[1] - PW / 2,
                                                                         top - PH])
        else:
            boards[face] = M.Manifold.cube([PW, CB.THICK, PH]).translate([c[0] - PW / 2, c[1] - POST / 2 - CB.THICK,
                                                                         top - PH])
    body = body - M.Manifold.batch_boolean(pilots, M.OpType.Add)
    return body, boards, g


def hand_solids():
    """{idx: (label, manifold in P)} of the hand assembly's solids (logs/20261104-cad_v3/extract/hand/*.stl)."""
    import manifold3d as M
    import trimesh
    R, t = P_from_A()
    parts = {p["idx"]: p for p in json.loads((EXTRACT / "parts.json").read_text())["parts"] if p["file"] == "hand"}
    out, bad = {}, []
    for idx, p in parts.items():
        tm = trimesh.load(ROOT / p["stl"], force="mesh")
        tm.merge_vertices()
        m = M.Manifold(M.Mesh(vert_properties=np.asarray(tm.vertices @ R.T + t, np.float32),
                              tri_verts=np.asarray(tm.faces, np.uint32)))
        if m.status() == M.Error.NoError:
            out[idx] = (p["label"], m)
        else:
            bad.append((idx, p["label"], str(m.status())))
    return out, bad


def cmd_build(a):
    body, boards, g = mount()
    solids, bad = hand_solids()
    arm = solids.pop(0)[1]
    rep = {"what": "mount of the ChArUco board on the camera arm's fork", "script": "scripts/hardware/charuco_board_mount_print.py",
           "notch_P_mm": [np.round(g["notch_lo"], 2).tolist(), np.round(g["notch_hi"], 2).tolist()],
           "arm_top_z_P_mm": round(g["top"], 2), "hole_P_mm": np.round(g["hole"], 2).tolist(),
           "solids_checked": len(solids) + 1, "solids_not_manifold": bad, "overlap_with_arm_mm3": round(abs((body ^ arm).volume()), 6),
           "gap_to_arm_mm": round(body.min_gap(arm, 5.0), 3)}
    for name, item in [("mount", body)] + [(f"board facing {f}_P", b) for f, b in boards.items()]:
        gaps = sorted(((item.min_gap(m, 60.0), idx, lab) for idx, (lab, m) in solids.items()))
        ov = max(abs((item ^ m).volume()) for _lab, m in solids.values())
        rep[name] = {"overlap_mm3": round(ov, 6),
                     "nearest": [{"solid": int(i), "label": lab, "gap_mm": round(float(d), 2)} for d, i, lab in gaps[:3]]}
        print(name, rep[name], flush=True)
    rep["board_with_arm_gap_mm"] = {f: round(b.min_gap(arm, 60.0), 2) for f, b in boards.items()}
    # print orientation: flange down, the post up
    T = np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, g["top"] + FLANGE_T]])
    t = OP.to_trimesh(body.transform(T))
    rep["stl"] = {"file": str(STL.relative_to(ROOT)), "watertight": bool(t.is_watertight), "triangles": int(len(t.faces)),
                  "bounds_mm": np.round(t.bounds, 2).tolist(), "volume_mm3": round(float(t.volume), 1)}
    t.export(STL)
    ok = (t.is_watertight and rep["overlap_with_arm_mm3"] < 1e-3
          and all(rep[k]["overlap_mm3"] < 1e-3 for k in rep if k.startswith(("mount", "board facing"))))
    rep["ok"] = bool(ok)
    REPORT.write_text(json.dumps(rep, indent=1))
    print("arm: overlap", rep["overlap_with_arm_mm3"], "mm3, gap", rep["gap_to_arm_mm"], "mm; boards to the arm",
          rep["board_with_arm_gap_mm"], "->", STL.relative_to(ROOT), "ok" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    a = ap.parse_args()
    {"build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
