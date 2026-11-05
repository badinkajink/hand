#!/usr/bin/env python3
"""Scratch: measure the realsense-ros D435 mesh (MuJoCo Menagerie realsense_d435i, derived from realsense2_description
d435.dae) to check the nominal lens offsets used in sensors_analysis.py: the lens centres relative to the casing's box,
and any holes in the casing's back and bottom faces (rear M3 pair, 1/4-20). Prints mm in the mesh's own frame."""
import sys
from pathlib import Path

import numpy as np
import trimesh

MEN = Path("/home/humanoid/Programs/mujoco_menagerie/realsense_d435i/assets")
NAMES = {0: "IR_Lens", 1: "IR_Emitter_Lens", 2: "IR_Rim", 3: "IR_Lens", 4: "Cameras_Gray", 5: "Black_Acrylic",
         6: "Black_Acrylic", 7: "RGB_Pupil", 8: "Metal_Casing"}


def load(i):
    m = trimesh.load(MEN / f"d435i_{i}.obj", process=False, force="mesh")
    return m


def main():
    cas = load(8)
    s = 1000.0 if cas.extents.max() < 1.0 else 1.0            # metres -> mm
    cb = cas.bounds * s
    print("casing bounds (mm):", np.round(cb, 2).tolist(), "extents", np.round(cas.extents * s, 2).tolist())
    ctr = cb.mean(axis=0)
    ext = cas.extents * s
    ax_w, ax_h, ax_d = int(np.argmax(ext)), None, None
    print("casing centre", np.round(ctr, 2))
    parts = {}
    for i in range(8):
        m = load(i)
        b = m.bounds * s
        c = b.mean(axis=0)
        parts[i] = (c, b)
        print(f"{i} {NAMES[i]:16s} centre-casing {np.round(c - ctr, 2)}  extents {np.round(m.extents * s, 2)}")
        if i in (0, 3, 7, 2):                                  # split multi-component meshes
            comps = m.split(only_watertight=False)
            if len(comps) > 1:
                for k, cc in enumerate(sorted(comps, key=lambda q: q.bounds.mean(axis=0)[ax_w])):
                    cbk = cc.bounds * s
                    print(f"     comp {k}: centre-casing {np.round(cbk.mean(axis=0) - ctr, 2)} extents "
                          f"{np.round(cc.extents * s, 2)} faces {len(cc.faces)}")
    # holes in the casing: section planes just inside each face, report interior loops that look circular
    for axis in range(3):
        for side in (0, 1):
            lvl = cb[side, axis] + (0.6 if side == 0 else -0.6)
            n = np.zeros(3)
            n[axis] = 1.0
            o = ctr.copy() / s
            o[axis] = lvl / s
            try:
                sec = cas.section(plane_origin=o, plane_normal=n)
            except Exception as e:                                  # noqa: BLE001
                print("section failed", axis, side, e)
                continue
            if sec is None:
                continue
            loops = [np.asarray(d.discrete(sec.vertices) if hasattr(d, "discrete") else sec.vertices[d.points])
                     for d in sec.entities]
            circ = []
            for L in loops:
                L = L * s
                cc = L.mean(axis=0)
                r = np.linalg.norm(L - cc, axis=1)
                if len(L) > 8 and r.mean() < 6 and r.std() < 0.15 * r.mean():
                    circ.append((np.round(cc - ctr, 2).tolist(), round(float(2 * r.mean()), 2)))
            print(f"section axis {axis} side {side} at {lvl:.2f}: {len(loops)} loops; circular (centre-casing, d):",
                  circ[:12])


if __name__ == "__main__":
    sys.exit(main())
