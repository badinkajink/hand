#!/usr/bin/env python3
"""Sensors on the cad_v3 hand (task "sensors" of the cad_v3 workflow, 2026-11-04): the D435 plate and the static AprilTag
mount in the hand assembly's frame A (mm), the D435's optical frames for an assumed mounting, and the camera's view.
Reads logs/20261104-cad_v3/extract/parts.json only (no meshes); writes sensors.json beside this file.

Identification (measured with match_downloads.py): solid 000 "Part 2" = ~/Downloads/d435_plate.stl (2026-08-30 20:34),
solid 005 "Part 006" = ~/Downloads/static_tag_mount.stl (2026-08-30 20:34). The D435 body itself is not in the export.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PARTS = {p["name"]: p for p in json.load(open(REPO / "logs/20261104-cad_v3/extract/parts.json"))["parts"]
         if p["file"] == "hand"}

# ---- Intel RealSense D435, nominal geometry (mm) -------------------------------------------------------------------
# Body 90 x 25 x 25 (Intel D400 datasheet). Lens offsets as in realsense-ros _d435.urdf.xacro (nominal, "approximate;
# the camera node publishes the calibrated extrinsics"): depth origin = left IR imager, 17.5 mm left of the bottom
# 1/4-20 mount (the mount is on the body's lateral centre), at mid-height, 4.2 mm behind the front glass, glass 0.1 mm
# inside the front plate; colour 15 mm further left of the depth origin; right IR 50 mm right of it; 1/4-20 14.9 mm
# behind the depth origin. Checked against the realsense-ros mesh (Menagerie realsense_d435i) in d435_mesh_check.py.
D435 = dict(width=90.0, height=25.0, depth=25.05, depth_x=-17.5, color_x=-32.5, right_ir_x=32.5,
            zero_depth_behind_front=4.3, screw_behind_depth_origin=14.9, rear_m3_x=(-22.5, 22.5))
FOV = dict(depth=(87.0, 58.0), color=(69.0, 42.0))       # horizontal x vertical, degrees (Intel datasheet)


def body_points():
    """Points of the D435 in its body frame: origin at the body's geometric centre, axes as the optical frames
    (x to the camera's right, y down, z out of the lenses)."""
    zf = D435["depth"] / 2.0                               # front face
    zo = zf - D435["zero_depth_behind_front"]              # optical centres (depth and colour)
    return dict(depth=np.array([D435["depth_x"], 0.0, zo]), color=np.array([D435["color_x"], 0.0, zo]),
                right_ir=np.array([D435["right_ir_x"], 0.0, zo]),
                screw=np.array([0.0, D435["height"] / 2.0, zo - D435["screw_behind_depth_origin"]]),
                m3_left=np.array([D435["rear_m3_x"][0], 0.0, -zf]), m3_right=np.array([D435["rear_m3_x"][1], 0.0, -zf]),
                front_centre=np.array([0.0, 0.0, zf]), back_centre=np.array([0.0, 0.0, -zf]))


def T(R, p):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


def plane(part, normal, near=None, min_area=0.0):
    """The planar face of a solid with this outward normal (and centre nearest `near`)."""
    out = [f for f in PARTS[part]["planes"] if np.dot(f["normal"], normal) > 0.999 and f["area"] >= min_area]
    if near is not None:
        out.sort(key=lambda f: np.linalg.norm(np.subtract(f["centre"], near)))
    return out


def fork():
    """The d435_plate's far end, read from its faces: the 20.4 mm slot, the two tines and the 5.4 mm hole."""
    p = PARTS["000"]
    inner_lo = plane("000", [0, 1, 0], near=[-4, 271, -5])[0]       # tine face toward the slot, +y normal
    inner_hi = plane("000", [0, -1, 0], near=[-4, 291, -5])[0]      # the other tine face, -y normal
    end = plane("000", [-1, 0, 0], near=[6, 281, -2])[0]            # closed end of the slot
    hole = [c for c in p["cylinders"] if abs(c["r"] - 2.7) < 0.05][0]
    tine_bottom = min(f["centre"][2] for f in plane("000", [0, 0, -1]) if f["centre"][1] > 260)
    return dict(slot_y=(inner_lo["centre"][1], inner_hi["centre"][1]),
                slot_width=inner_hi["centre"][1] - inner_lo["centre"][1],
                slot_x=(p["bbox"][0], end["centre"][0]), slot_depth=end["centre"][0] - p["bbox"][0],
                tine_z=(tine_bottom, 0.0), tine_x=(p["bbox"][0], 4.6),
                hole_xy=hole["point"][:2], hole_d=2 * hole["r"], end_y=p["bbox"][4])


def tag_plate():
    p = PARTS["005"]
    neg = plane("005", [0, -1, 0], min_area=1000)[0]
    pos = plane("005", [0, 1, 0], min_area=1000)[0]
    bb = p["bbox"]
    return dict(x=(bb[0], bb[3]), z_plate=(bb[2], -7.0), y_neg=neg["centre"][1], y_pos=pos["centre"][1],
                area_neg=neg["area"], area_pos=pos["area"])


def camera_on_post(z_c, f):
    """Assumed mounting: a 20 x 20 mm extrusion standing in the fork's slot (centred), the owner's d435_mount_80-20
    (4 mm plate, rails straddling the extrusion, two M3 holes 45 mm apart) on the extrusion's -y face, the D435's back
    on the bracket, upright (1/4-20 down), lenses facing -y_A (toward the hand). z_c is the body centre's height."""
    post_c = np.array([0.5 * (f["slot_x"][0] + f["slot_x"][1]), 0.5 * (f["slot_y"][0] + f["slot_y"][1])])
    post_face_y = post_c[1] - 10.0
    back_y = post_face_y - 4.0                               # bracket plate thickness
    centre = np.array([post_c[0], back_y - D435["depth"] / 2.0, z_c])
    R = np.column_stack([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])  # columns: body (optical) x, y, z in A
    return centre, R, post_c


def tag_frames(tp, cz=-32.0):
    """apriltag frame of the 40 mm tag (origin at its centre, x right, y down, z into the tag in the tag's own image),
    mounted with its x axis up as the 2026-08-30 probe measured (pupil_apriltags on probe_ir.png: tag x = image up).
    For the +y face (facing +y_A): x = +z_A, y = -x_A, z = -y_A. For the -y face: x = +z_A, y = +x_A, z = +y_A."""
    cx = 0.5 * (tp["x"][0] + tp["x"][1])
    pos = T(np.column_stack([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]]), [cx, tp["y_pos"], cz])
    neg = T(np.column_stack([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]]), [cx, tp["y_neg"], cz])
    return pos, neg


def part_pose(name):
    """The solid's own frame (its Onshape part frame) in A, from the STEP placement (FreeCAD quaternion x, y, z, w)."""
    pl = PARTS[name]["placement"]
    R = Rotation.from_quat(pl["quat_xyzw"]).as_matrix()
    R[np.abs(R) < 1e-9] = 0.0
    return T(R, pl["pos"])


def frames(centre, R):
    bp = body_points()
    out = {}
    for k in ("depth", "color"):
        out[k] = T(R, centre + R @ bp[k])
    return out, {k: centre + R @ v for k, v in bp.items()}


def view_metrics(Tc, pts, fov):
    """Angles of points from the optical axis (horizontal in the image plane, vertical) and range (mm)."""
    R, p = Tc[:3, :3], Tc[:3, 3]
    res = {}
    for k, q in pts.items():
        c = R.T @ (np.asarray(q) - p)
        h = math.degrees(math.atan2(c[0], c[2]))
        v = math.degrees(math.atan2(c[1], c[2]))
        inside = c[2] > 0 and abs(h) <= fov[0] / 2 and abs(v) <= fov[1] / 2
        res[k] = dict(cam_mm=[round(x, 1) for x in c], range_mm=round(float(np.linalg.norm(c)), 1),
                      h_deg=round(h, 1), v_deg=round(v, 1), in_fov=bool(inside))
    return res


def main():
    f = fork()
    tp = tag_plate()
    tips = {"thumb (Finger v3 <3>, solid 012)": PARTS["012"]["com"],
            "Finger v3 <1> (solid 037)": PARTS["037"]["com"],
            "Finger v3 <2> (solid 091)": PARTS["091"]["com"]}
    tip_c = np.mean([np.array(v) for v in tips.values()], axis=0)
    tag_c = np.array([0.5 * (tp["x"][0] + tp["x"][1]), tp["y_pos"], 0.5 * (tp["z_plate"][0] + tp["z_plate"][1])])
    z_hi = f["tine_z"][0] - 46.6 / 2.0                       # bracket (46.6 mm tall) just below the tines
    variants = {}
    for name, zc in (("highest", z_hi), ("tips_level", float(tip_c[2]))):
        centre, R, post_c = camera_on_post(zc, f)
        Fr, bp = frames(centre, R)
        pts = dict(tips, **{"fingertip centroid": tip_c.tolist(), "tag centre (+y face)": tag_c.tolist()})
        variants[name] = dict(
            body_centre_A=[round(x, 2) for x in centre], post_centre_xy_A=[round(x, 2) for x in post_c],
            T_A_from_depth_optical=[round(float(x), 4) for x in Fr["depth"].ravel()],
            T_A_from_color_optical=[round(float(x), 4) for x in Fr["color"].ravel()],
            depth_origin_A=[round(float(x), 2) for x in Fr["depth"][:3, 3]],
            color_origin_A=[round(float(x), 2) for x in Fr["color"][:3, 3]],
            axis_vs_minus_zA_deg=round(math.degrees(math.acos(-Fr["depth"][2, 2])), 2),
            depth_view=view_metrics(Fr["depth"], pts, FOV["depth"]),
            color_view=view_metrics(Fr["color"], pts, FOV["color"]),
            body_points_A={k: [round(float(x), 2) for x in v] for k, v in bp.items()})
    T_tag_pos, T_tag_neg = tag_frames(tp)
    Td = np.array(variants["highest"]["T_A_from_depth_optical"]).reshape(4, 4)
    T_cam_tag = np.linalg.inv(Td) @ T_tag_pos
    out = dict(fork=f, tag_plate=tp, fingertips_A={k: [round(x, 2) for x in v] for k, v in tips.items()},
               fingertip_centroid_A=[round(x, 2) for x in tip_c], tag_centre_pos_face_A=tag_c.tolist(),
               d435=D435, fov=FOV, variants=variants,
               T_A_from_tag_pos_face=T_tag_pos.round(4).ravel().tolist(),
               T_A_from_tag_neg_face=T_tag_neg.round(4).ravel().tolist(),
               T_depth_from_tag_highest=T_cam_tag.round(4).ravel().tolist(),
               T_A_from_part={"000": part_pose("000").round(4).ravel().tolist(),
                              "005": part_pose("005").round(4).ravel().tolist()})
    (HERE / "analysis.json").write_text(json.dumps(out, indent=1))
    print("T_depth_from_tag (highest):\n", T_cam_tag.round(3))
    print("part 000:\n", part_pose("000").round(3), "\npart 005:\n", part_pose("005").round(3))


if __name__ == "__main__":
    main()
