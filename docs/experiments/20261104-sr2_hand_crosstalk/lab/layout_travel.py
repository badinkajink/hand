"""Each finger's travel at a layout move, from the D435 beside the hand (lab/d435_rgbd.py), against the plan. The owner,
19:1x on 11-05: "maybe you can do some intelligent geometry processing with the d435 to roughly estimate travel of each
finger".

The camera stands on the thumb's side, about 0.3 to 0.45 m from the fingers, turned some 8 deg from palm x (the
references below). A carriage move (palm x) shows mostly as a change of depth, an X-stage move (palm y) as a sideways
shift. For each finger the program takes its depth patch before the move (the pixels in a box around where it should
be, within its depth window) and registers it on the frame after: over shifts out to 40 px and 20 mm past the planned
move each way, the shift at which the most of its pixels find a depth within 3 mm. That shift, through the camera's
turn and its perspective, is the finger's travel in palm x and palm y. Registration needs neither the finger's whole
view nor one surface: at twist the index
showed 8 % of its pixels past the thumb and still read its travel within 1.4 mm. Each case's share of matched pixels
is printed too (every planned axis moved, the carriage left behind, the X stage left behind, nothing moved), so the
margin can be seen. A planned move of 8 mm or more read under a third of its size, or the wrong way: exit 3.

Checks on 11-05 (hom1), the plan in brackets: generalist read the index's and middle's X stages -19.9 and +19.6 mm
(-20, +20); twist read the carriages -16.0, +16.3 and +17.6 (-15, +15, +15) and the X stages -21.4 and +17.6 (-20, +20);
cover read the carriages +26.1, -26.5 and -26.7 (+25, -25, -25). Pairs of frames with nothing moving read 0 to 4 mm on
every axis, so every planned move of theirs read as not moved.

    /usr/bin/python3 layout_travel.py RGBD_DIR BEFORE AFTER FROM_LAYOUT TO_LAYOUT   (BEFORE, AFTER: HH:MM:SS, frames at
        the zero pose before the move and in the baseline after it)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

LAYOUTS = {   # palm-frame mounts (mm), as make_layouts.LAYOUTS (this program runs without mujoco)
    "home": {"thumb": (-80.0, -55.0), "index": (80.0, 85.0), "middle": (80.0, -85.0)},
    "g12": {"thumb": (-42.5, 0.0), "index": (42.5, 40.0), "middle": (42.5, -40.0)},
    "rv04_mid": {"thumb": (-35.0, 0.0), "index": (35.0, 40.0), "middle": (35.0, -40.0)},
    "sv1_w6689": {"thumb": (-54.3, -33.7), "index": (37.9, 42.1), "middle": (39.0, -43.0)},
    "sv1_w0099": {"thumb": (-43.8, -2.3), "index": (41.1, 35.7), "middle": (55.2, -44.2)},
    "sv1_u0308": {"thumb": (-37.8, 4.3), "index": (38.2, 39.1), "middle": (46.9, -76.0)},
    "hom_roll_85_170": {"thumb": (-42.5, 0.0), "index": (42.5, 85.0), "middle": (42.5, -85.0)},
    "hom_press_70_150": {"thumb": (-35.0, 0.0), "index": (35.0, 75.0), "middle": (35.0, -75.0)},
    "hom_generalist_70_110": {"thumb": (-35.0, 0.0), "index": (35.0, 55.0), "middle": (35.0, -55.0)},
    "hom_twist_100_70": {"thumb": (-50.0, 0.0), "index": (50.0, 35.0), "middle": (50.0, -35.0)},
    "hom_cover_50_75": {"thumb": (-25.0, 0.0), "index": (25.0, 37.5), "middle": (25.0, -37.5)},
}
# References from the press layout's zero pose (hom1, 19:22:33 to :37, three frames within 2 px and 1 mm): each finger's
# column and the median depth of its housings. Index and middle, both at palm x 35, stand 22 mm apart in depth over their
# 150 mm of palm y, so the camera is turned TILT about the vertical from palm x: palm x runs along (sin, 0, cos) of the
# camera frame (x right, y down, z ahead) and palm y along (-cos, 0, sin), the index (+y) on the left. The thumb's
# column moved left when its X stage moved +55 mm in palm y at run2's g12 move (12:35).
REF = "hom_press_70_150"
U_REF = {"index": 600.0, "thumb": 774.0, "middle": 941.0}
Z_REF = {"index": 433.0, "thumb": 318.0, "middle": 411.0}
TILT = np.arctan2(433.0 - 411.0, 150.0)
EX = np.array([np.sin(TILT), 0.0, np.cos(TILT)])
EY = np.array([-np.cos(TILT), 0.0, np.sin(TILT)])
V0, V1, HALF_W, HALF_Z, TOL = 250, 480, 45, 30.0, 3.0
K = {}


def frame(rgbd, hms):
    t = int(hms[:2]) * 3600 + int(hms[3:5]) * 60 + int(hms[6:8])
    fs = sorted(rgbd.glob("*_depth.png"))
    best = min(fs, key=lambda f: abs(int(f.name[:2]) * 3600 + int(f.name[2:4]) * 60 + int(f.name[4:6]) - t))
    return np.asarray(Image.open(best)).astype(float) * K["depth_scale"] * 1000.0, best.name[:12]


def expect(f, lay):
    """expected image column and depth of the finger's housings at a layout, from the press references"""
    d = (LAYOUTS[lay][f][0] - LAYOUTS[REF][f][0]) * EX + (LAYOUTS[lay][f][1] - LAYOUTS[REF][f][1]) * EY
    z = Z_REF[f] + d[2]
    return U_REF[f] + d[0] * K["fx"] / z, z


def patch(f, lay, db):
    """the finger's pixels before the move: rows, columns, depths"""
    u_e, z_e = expect(f, lay)
    lo = z_e - HALF_Z
    if f != "thumb":                                         # the thumb stands nearer: keep its pixels out
        lo = max(lo, expect("thumb", lay)[1] + 35.0)
    u0 = max(0, int(u_e - HALF_W))
    sub = db[V0:V1, u0:int(u_e + HALF_W)]
    vv, uu = np.nonzero((sub > lo) & (sub < z_e + HALF_Z))
    return vv + V0, uu + u0, sub[vv, uu]


def share(vv, uu, d0, da, du, dzs):
    """for one column shift: the share of the patch's pixels matched within TOL, for every depth shift in dzs"""
    u = uu + du
    ok = (u >= 0) & (u < da.shape[1])
    after = np.full(len(u), np.nan)
    after[ok] = da[vv[ok], u[ok]]
    after[after == 0] = np.nan
    return (np.abs(after[:, None] - d0[:, None] - dzs[None, :]) < TOL).mean(axis=0)


def to_palm(du, dz, u, z):
    """an image shift (du px, dz mm) of a patch at column u and depth z, in palm x and y (mm). The sideways travel
    takes the perspective into account: a finger off the image's centre that comes nearer moves outward in the image
    (20:4x on 11-05: the middle's 25 mm toward the camera at cover moved it 14 px outward)"""
    lat = ((u + du - K["ppx"]) * (z + dz) - (u - K["ppx"]) * z) / K["fx"]
    return lat * EX[0] + dz * EX[2], lat * EY[0] + dz * EY[2]


def from_palm(dx, dy, u, z):
    """the image shift (du px, dz mm) of a patch at column u and depth z that a palm move (dx, dy) gives"""
    m = dx * EX + dy * EY
    x = (u - K["ppx"]) * z / K["fx"] + m[0]
    return int(round(x * K["fx"] / (z + m[2]) + K["ppx"] - u)), float(m[2])


def main():
    rgbd, before, after, src, dst = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    K.update(json.loads((rgbd / "intrinsics.json").read_text()))
    db, nb = frame(rgbd, before)
    da, na = frame(rgbd, after)
    print(f"{src} -> {dst}: frames {nb} and {na}")
    bad = []
    for f in ("thumb", "index", "middle"):
        vv, uu, d0 = patch(f, src, db)
        if len(vv) < 200:
            print(f"  {f}: too few pixels before ({len(vv)})")
            bad.append(f)
            continue
        u, z = float(np.median(uu)), float(np.median(d0))
        dx = LAYOUTS[dst][f][0] - LAYOUTS[src][f][0]
        dy = LAYOUTS[dst][f][1] - LAYOUTS[src][f][1]
        du_p, dz_p = from_palm(dx, dy, u, z)                    # the search reaches 40 px and 20 mm past the plan
        r_u, r_z = abs(du_p) // 2 * 2 + 40, np.ceil(abs(dz_p)) + 20.0
        DUS, DZS = np.arange(-r_u, r_u + 1, 2), np.arange(-r_z, r_z + 0.5, 1.0)
        grid = np.array([share(vv, uu, d0, da, du, DZS) for du in DUS])
        i, k = np.unravel_index(int(np.argmax(grid)), grid.shape)
        gx, gy = to_palm(int(DUS[i]), float(DZS[k]), u, z)
        cases, seen = {}, set()
        for name, (cx, cy) in (("all moved", (dx, dy)), ("carriage left", (0.0, dy)), ("X stage left", (dx, 0.0)),
                               ("nothing moved", (0.0, 0.0))):
            if (cx, cy) in seen:
                continue
            seen.add((cx, cy))
            du, dz = from_palm(cx, cy, u, z)
            j, kk = int(np.argmin(np.abs(DUS - du))), int(np.argmin(np.abs(DZS - dz)))
            cases[name] = float(grid[max(0, j - 1):j + 2, max(0, kk - 2):kk + 3].max())
        verdict = []
        for name, got, plan in (("carriage", gx, dx), ("X stage", gy, dy)):
            if abs(plan) >= 8.0:
                ok = abs(got) >= abs(plan) / 3.0 and np.sign(got) == np.sign(plan)
                verdict.append(f"{name} {'moved' if ok else 'DID NOT MOVE'}")
                if not ok:
                    bad.append(f"{f} {name}")
        print(f"  {f}: best shift {int(DUS[i]):+d} px, {DZS[k]:+.0f} mm ({grid[i, k]:.0%} of {len(vv)} pixels): "
              f"carriage {gx:+.1f} mm (plan {dx:+.1f}), X stage {gy:+.1f} mm (plan {dy:+.1f}); "
              f"{', '.join(verdict) or 'no move of 8 mm or more planned'}. Cases: "
              + ", ".join(f"{n} {v:.0%}" for n, v in cases.items()))
    print("ALL MOVED" if not bad else f"CHECK: {', '.join(bad)}")
    raise SystemExit(3 if bad else 0)


if __name__ == "__main__":
    main()
