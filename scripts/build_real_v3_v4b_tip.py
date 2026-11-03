#!/usr/bin/env python3
"""real_v3 with the v4b magnetic-sensing fingertip (the tip on the UR5 calibration rig since 2026-10-23).

Takes the STEP-derived hand with the TPU block tip (assets/mjcf/real_v3/real_hand_block.xml, written by
scripts/build_real_v3_from_step.py on 2026-10-30) and swaps every finger's distal parts for the v4b set:

  * <f>_pip_frame: the DIP link b v4 (board mount), the 3T cassette and the TW15H3 top cap trimmed at x_F -0.10 with
    its four M2 button heads, as collision spheres inscribed in each part (315 per finger) plus visual meshes; mass and
    inertia of the three PLA parts (9.26 g, was 5.83 g for the block-era distal link).
  * <f>_tip: moves from 26.4 mm to 36.36 mm below the PIP axis (the v4b link is 9.75 mm longer) and carries the
    11/22 half-ellipsoid TPU pad (key tcd_ell7.4x8.5x11x22: 7.4 mm across, 8.5 mm deep, 11 mm distal and 22 mm proximal
    of the apex, apex 10.55 mm palmar of the finger axis) as a sphere grid whose outer envelope is the pad surface;
    pad mass 3.65 g (was 4.82 g).
  * <f>_pad_site: at the pad's apex, z = outward palmar normal; in-plane half-extents 5 mm along the finger (the dome
    falls 0.6 mm below the apex plane 4 mm distal of it) by 6 mm across, for the plan's flat-face IK.

Source of the parts: assets/mjcf/real_v3/v4b/ (copied from main, assets/mjcf/experimental/20261030-cad_v2, commit
72f382c3, scripts/build_cad_v2_finger.py; pad spheres from main's scripts/tip_codesign_geometry.py). Those parts are in
the cad_v1 finger frame F shifted to the PIP axis, which is real_v3's <f>_pip_frame for index and middle to 0.6 mm /
0.65 deg (surface registration of the v4b DIP link against real_v3's distal link above the pad; the rest is the design
change); the thumb is the same mirrored by 180 deg about z, as in both models.

Not modelled: the magnetometer board, its cable and the pad's four magnets (a few grams, no geometry).

    .venv-cpu/bin/python scripts/build_real_v3_v4b_tip.py [--pitch 1.5]
Writes assets/mjcf/real_v3/real_hand.xml.
"""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "assets/mjcf/real_v3"
SRC, OUT, V4B = DIR / "real_hand_block.xml", DIR / "real_hand.xml", DIR / "v4b"
FINGERS = ("thumb", "index", "middle")
TIP_BODY_M = 0.03636
APEX_X = 0.01055
SITE_HALF = (0.005, 0.006)          # along the finger, across it (m)
PIP_INERTIAL = dict(pos=(0.002967, -0.000912, -0.022687), mass=0.009263,
                    I=(2.5794e-06, 1.7135e-06, 1.1852e-06, -1.08e-08, 1.736e-07, 2.37e-08))
TIP_INERTIAL = dict(pos=(-0.002908, 0.0, 0.001001), mass=0.003654,
                    I=(1.631e-07, 1.569e-07, 9.03e-08, 0.0, 5.1e-09, 0.0))
PAD_FRICTION = "2.4 0.2 0.02"       # as real_v3's pads; cpu_grasp.scene gives the object priority over it


def f6(v):
    return " ".join(f"{x:.7f}".rstrip("0").rstrip(".") if abs(x) > 5e-8 else "0" for x in v)


def inertial(d, palmar):
    """Index / middle values are for palmar -x; the thumb (palmar +x) is the 180 deg turn about z."""
    m = palmar < 0
    p = np.array(d["pos"]) * (1 if m else np.array([-1, -1, 1]))
    I = np.array(d["I"]) * (1 if m else np.array([1, 1, 1, 1, -1, -1]))
    return {"pos": f6(p), "mass": f"{d['mass']:.6f}", "fullinertia": " ".join(f"{x:.4e}" for x in I)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch", type=float, default=1.5, help="pad sphere spacing, mm (1.5, 2 or 1)")
    a = ap.parse_args()
    key = {1.5: "s1.5_r1", 2.0: "s2_r1", 1.0: "s1_r0.75"}[a.pitch]
    pads = json.loads((V4B / "pad_spheres.json").read_text())["sets"][key]
    C = np.array(pads["centres"]); rs = pads["radius_m"]          # palmar +x
    rigid = json.loads((V4B / "spheres.json").read_text())["pip_frame"]   # palmar -x (frame F)

    root = ET.parse(SRC).getroot()
    asset = root.find("asset")
    for k in ("dip_link", "cassette", "cap"):
        ET.SubElement(asset, "mesh", name=f"v4b_{k}", file=f"v4b/v4b_{k}.stl")
    for s in ("pos", "neg"):
        ET.SubElement(asset, "mesh", name=f"v4b_pad_{s}", file=f"v4b/v4b_pad_{s}.stl")
    ET.SubElement(asset, "material", name="v4b_link", rgba="0.25 0.27 0.3 1")
    ET.SubElement(asset, "material", name="v4b_pad", rgba="0.85 0.55 0.35 1")
    n_pip = 0
    for f in FINGERS:
        pf = root.find(f".//body[@name='{f}_pip_frame']")
        tip = pf.find(f"body[@name='{f}_tip']")
        site = tip.find("site")
        palmar = 1.0 if float(site.get("pos").split()[0]) > 0 else -1.0
        flip = np.array([-palmar, -palmar, 1.0])                     # F (palmar -x) -> this finger
        q = {} if palmar < 0 else {"quat": "0 0 0 1"}
        # distal link: old spheres out, v4b parts in
        for g in list(pf.findall("geom")):
            if g.get("name", "").startswith(f"{f}_dist_s"):
                pf.remove(g)
        pf.find("inertial").attrib.update(inertial(PIP_INERTIAL, palmar))
        at = list(pf).index(tip)
        new = [ET.Element("geom", {"class": "visual", "mesh": f"v4b_{k}", "material": "v4b_link", **q})
               for k in ("dip_link", "cassette", "cap")]
        for k in ("dip_link", "cassette", "cap"):
            for i, (x, y, z, r) in enumerate(rigid[k]):
                new.append(ET.Element("geom", {"name": f"{f}_v4b_{k}_s{i}", "type": "sphere",
                                               "pos": f6(np.array([x, y, z]) * flip), "size": f"{r:.6f}",
                                               "material": "ell_mat"}))
        n_pip = len(new) - 3
        for i, e in enumerate(new):
            pf.insert(at + i, e)
        # tip body: the half-ellipsoid pad
        tip.set("pos", f"0 0 {-TIP_BODY_M:.5f}")
        for g in list(tip):
            if g.tag in ("geom", "site"):
                tip.remove(g)
        tip.find("inertial").attrib.update(inertial(TIP_INERTIAL, palmar))
        ET.SubElement(tip, "geom", {"class": "visual", "mesh": f"v4b_pad_{'pos' if palmar > 0 else 'neg'}",
                                    "material": "v4b_pad"})
        for i, c in enumerate(C * np.array([palmar, 1.0, 1.0])):
            ET.SubElement(tip, "geom", {"name": f"{f}_pad_s{i}", "type": "sphere", "pos": f6(c), "size": f"{rs:.6f}",
                                        "material": "pad_mat", "friction": PAD_FRICTION})
        # site at the apex; local z = palmar normal, local x = along the finger (proximal)
        h = math.sqrt(0.5)
        ET.SubElement(tip, "site", {"name": f"{f}_pad_site", "type": "box", "pos": f6([palmar * APEX_X, 0, 0]),
                                    "quat": f6([h, 0, palmar * h, 0]),
                                    "size": f"{SITE_HALF[0]:.6f} {SITE_HALF[1]:.6f} 0.0003", "rgba": "1 0.2 0.2 0.5"})
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(OUT)
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(OUT))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(C)} pad spheres ({a.pitch:g} mm, r {rs * 1000:g} mm) and {n_pip} "
          f"rigid spheres per finger; {m.ngeom} geoms, {m.nbody} bodies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
