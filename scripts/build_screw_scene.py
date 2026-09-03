"""Turn the shaft into a screw and the floor into its seat.

Every gaiting result so far has the tool standing on its own flat end face on a flat plane. That
makes the ground a pure friction brake -- resisting torque mu*N*r, nothing else -- and it leaves
the shaft free to walk in x and y, which is why `drift_mm` had to be reported at all. A screw is
not that. A screw sits in a seat, and the seat does three things a plane does not:

  it CENTRES     a cone in a matching cone has one equilibrium, so lateral drift cannot
                 accumulate and a set-down that misses by a few millimetres is pulled in
  it WEDGES      the normal force on a cone flank is the axial load over sin(alpha), so the
                 same press buys more resisting torque than it does on a plane
  it CONSTRAINS  the seat resists tipping, which on a plane was only the 14.0 deg static limit

so "does the gait still turn it" is a real question and not a formality, and the answer decides
whether ground-supported gaiting is a screwing primitive or a spinning-a-rod primitive.

WHAT IT BUILDS. The object keeps its cylinder and gains a 45 deg frustum tip (a mesh, so the
cone is exact -- MuJoCo treats a mesh as its convex hull and a frustum is convex). The floor
gains a matching countersink. A conical HOLE is not convex, so the seat is built the way a real
recess is: as facets. A ring of tilted boxes forms the cone wall, a disc closes the bottom, and
a flat annulus of boxes is the table top around the mouth.

    uv run --extra rl python scripts/build_screw_scene.py \
        --scene results/phase1/real_v1/rv05_manual_stored/frozen_scene.xml
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _quat_from_cols(x, y, z) -> str:
    R = np.column_stack([x, y, z])
    t = float(np.trace(R))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    q = np.array(q) / np.linalg.norm(q)
    return " ".join(f"{v:.9g}" for v in q)


def _frustum_obj(path: Path, r_top: float, r_bot: float, h: float, n: int = 32) -> None:
    """A frustum with its wide end at z=0 and its narrow end at z=-h, written as an OBJ.

    Convex, so MuJoCo's hull is the shape itself and the cone angle is exact rather than
    stepped. A stack of cylinders would only touch the seat on its step edges.
    """
    v, f = [], []
    for k in range(n):
        a = 2 * np.pi * k / n
        v.append((r_top * np.cos(a), r_top * np.sin(a), 0.0))
        v.append((r_bot * np.cos(a), r_bot * np.sin(a), -h))
    v.append((0.0, 0.0, 0.0))          # top centre
    v.append((0.0, 0.0, -h))           # bottom centre
    ct, cb = 2 * n + 1, 2 * n + 2
    for k in range(n):
        a0, b0 = 2 * k + 1, 2 * k + 2
        a1, b1 = (2 * ((k + 1) % n)) + 1, (2 * ((k + 1) % n)) + 2
        f.append((a0, b0, b1))
        f.append((a0, b1, a1))
        f.append((ct, a1, a0))
        f.append((cb, b0, b1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"v {x:.7g} {y:.7g} {z:.7g}\n" for x, y, z in v)
                    + "".join(f"f {i} {j} {k}\n" for i, j, k in f))


def _annulus(parent, name: str, n: int, r_in: float, r_out: float,
             z_in: float, z_out: float, t: float, material: str) -> None:
    """A ring of n boxes whose top faces tile the annulus (r_in,z_in)-(r_out,z_out).

    Boxes overlap slightly at the inner radius (their width is set at r_out) rather than leaving
    gaps -- a gap in a seat wall is a place for the tip to catch, and it would show up as a
    spurious ratchet in the gait.
    """
    dr, dz = r_out - r_in, z_out - z_in
    L = float(np.hypot(dr, dz))
    w = 2.0 * r_out * np.tan(np.pi / n)
    rc, zc = 0.5 * (r_in + r_out), 0.5 * (z_in + z_out)
    for k in range(n):
        a = 2 * np.pi * k / n
        ex = np.array([-np.sin(a), np.cos(a), 0.0])                       # along the ring
        ey = np.array([np.cos(a) * dr / L, np.sin(a) * dr / L, dz / L])   # up the slope
        ez = np.cross(ex, ey)                                             # surface normal
        if ez[2] < 0:
            ey, ez = -ey, -ez
        c = np.array([rc * np.cos(a), rc * np.sin(a), zc]) - ez * (t / 2)
        ET.SubElement(parent, "geom", {
            "name": f"{name}{k}", "type": "box", "material": material,
            "size": f"{w / 2:.6g} {L / 2:.6g} {t / 2:.6g}",
            "pos": " ".join(f"{v:.6g}" for v in c),
            "quat": _quat_from_cols(ex, ey, ez)})


def build(scene: Path, out: Path, obj: str = "screwdriver_medium",
          half_angle_deg: float = 45.0, r_flat: float = 0.0025,
          clearance: float = 0.0005, relief: float = 0.0015,
          facets: int = 32, table_r: float = 0.15,
          wall_t: float = 0.02, socket_xy=(0.0, 0.0)) -> dict:
    root = ET.parse(scene).getroot()
    wb = root.find("worldbody")
    body = next(b for b in wb.iter("body") if b.get("name") == obj)
    g = body.find("geom")
    r_obj, half = (float(v) for v in g.get("size").split())
    ta = np.tan(np.radians(half_angle_deg))

    # The tip: a frustum from the shaft radius down to a small flat, at `half_angle_deg` to the
    # axis. Its height follows from the radii -- the angle is the thing being specified.
    h_tip = (r_obj - r_flat) / ta
    obj_dir = out.parent
    mesh = (obj_dir / f"{out.stem}_tip.obj").resolve()
    _frustum_obj(mesh, r_obj, r_flat, h_tip, facets)
    asset = root.find("asset")
    ET.SubElement(asset, "mesh", {"name": "screw_tip", "file": str(mesh)})
    ET.SubElement(asset, "material", {"name": "seat_mat", "rgba": "0.35 0.33 0.30 1"})
    tip = ET.SubElement(body, "geom", {
        "type": "mesh", "mesh": "screw_tip", "material": "object_mat",
        # local -z is the shaft's DOWN end: the carry finishes with local +z on world +z.
        "pos": f"0 0 {-half:.6g}", "quat": "1 0 0 0",
        "friction": g.get("friction", "2.4 0.2 0.02"), "density": g.get("density", "500")})
    tip.tail = None

    # The seat: a countersink of the same angle, `clearance` wider, cut into a table whose top
    # is z = 0 -- the same height the flat-floor gait stands the shaft on, so the two scenes put
    # the pad ring at the same place on the shaft and the comparison is not confounded.
    R_m = r_obj + clearance
    r_seat = r_flat + clearance
    depth = (R_m - r_seat) / ta
    seat = ET.SubElement(wb, "body", {"name": "screw_seat",
                                      "pos": f"{socket_xy[0]:.6g} {socket_xy[1]:.6g} 0"})
    _annulus(seat, "seat_wall", facets, r_seat, R_m, -depth, 0.0, wall_t, "seat_mat")
    _annulus(seat, "seat_top", max(8, facets // 2), R_m, table_r, 0.0, 0.0, wall_t, "seat_mat")
    ET.SubElement(seat, "geom", {
        "name": "seat_floor", "type": "cylinder", "material": "seat_mat",
        "size": f"{r_seat:.6g} {wall_t / 2:.6g}",
        # RELIEVED. With matched cones the tip's own flat reaches the seat's flat within half a
        # millimetre, so without relief the load lands on the apex and the flanks -- the part
        # that centres and wedges -- never take it. Real countersinks are relieved for exactly
        # this reason.
        "pos": f"0 0 {-depth - relief - wall_t / 2:.6g}"})
    # The world floor drops below the seat, so a missed insertion still lands on something and
    # the run reports a drop rather than a fall through the scene.
    for f in wb.iter("geom"):
        if f.get("name") == "floor":
            f.set("pos", f"0 0 {-depth - relief - wall_t - 0.005:.6g}")

    # Where the shaft comes to rest: matched cones touch all along the flank, so the cylinder's
    # bottom face sits where the seat's radius equals the shaft's, z = r_obj - R_m.
    seat_z = float(half + (r_obj - R_m))
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding="unicode")
    return {"scene": str(out), "tip_mesh": str(mesh), "r_obj": r_obj, "half": half,
            "half_angle_deg": half_angle_deg, "h_tip_mm": round(h_tip * 1000, 2),
            "socket_depth_mm": round(depth * 1000, 2), "mouth_r_mm": round(R_m * 1000, 2),
            "seat_r_mm": round(r_seat * 1000, 2), "clearance_mm": clearance * 1000,
            "relief_mm": relief * 1000, "capture_r_mm": round((R_m - r_flat) * 1000, 2),
            "tip_len": round(h_tip, 5),
            "seat_z": round(seat_z, 5), "socket_xy": list(socket_xy)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path,
                    default=ROOT / "results/phase1/real_v1/rv05_manual_stored/frozen_scene.xml")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--object-body", default="screwdriver_medium")
    ap.add_argument("--half-angle", type=float, default=45.0)
    ap.add_argument("--r-flat", type=float, default=0.0025)
    ap.add_argument("--clearance", type=float, default=0.0005)
    ap.add_argument("--facets", type=int, default=32)
    ap.add_argument("--relief", type=float, default=0.0015)
    ap.add_argument("--socket-xy", default="0.12,0",
                    help="m. Not under the palm: the tool is picked up at the "
                         "origin and the seat is somewhere else, so the chain has "
                         "to TRANSPORT it. 0.12 clears the palm plate's 0.085 "
                         "half-width, which at 0.06 spawns the shaft through it.")
    ap.add_argument("--table-r", type=float, default=0.15)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.scene.with_name(args.scene.stem + "_screw.xml")
    info = build(args.scene, out, obj=args.object_body, half_angle_deg=args.half_angle,
                 r_flat=args.r_flat, clearance=args.clearance, facets=args.facets,
                 table_r=args.table_r, relief=args.relief,
                 socket_xy=tuple(float(v) for v in args.socket_xy.split(",")))
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(out))
    info["ngeom"] = int(m.ngeom)
    print(json.dumps(info, indent=2))
    if args.out_json:
        args.out_json.write_text(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
