#!/usr/bin/env python3
"""Scratch: assemble sensors.json (task "sensors" of the cad_v3 workflow) from analysis.json and validation.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
an = json.load(open(HERE / "analysis.json"))
va = json.load(open(HERE / "validation.json"))
hi, lo = an["variants"]["highest"], an["variants"]["tips_level"]
f = an["fork"]


def views(v):
    return {cam: {k: dict(range_mm=m["range_mm"], right_deg=m["h_deg"], down_deg=m["v_deg"], in_fov=m["in_fov"])
                  for k, m in v[f"{cam}_view"].items()} for cam in ("depth", "color")}


data = dict(
    frame="A = frame of 'hand v3 (anthro config).step' (mm): z_A up toward the robot, base plate z 0..3",
    camera_parts=[
        dict(solid="hand/000", label="Part 2", name="d435_plate",
             identified_by="the same solid as ~/Downloads/d435_plate.stl (owner's export, 2026-08-30 20:34) turned "
                           "180 deg about z: surface distance max 0.46 mm, mean 0.13 mm (match_downloads.py in this folder)",
             T_A_from_part=an["T_A_from_part"]["000"], bbox_A=[-14.2, 111.0, -10.0, 25.8, 295.5, 0.0],
             shape="4 mm plate (z_A -4..0) under the base plate along its +x edge, from y_A 111 to 295.5; the last "
                   "52.6 mm overhang the base plate's +y edge (y_A 242.9)",
             screws="M4 clearance holes (d 4.4) at (6.5, 117.0), (-1.5, 173.8), (21.3, 230.4), on the base plate's "
                    "holes; the robot mount has M4 nut pockets above the first two",
             fork=dict(slot_y_A=f["slot_y"], slot_x_A=f["slot_x"], slot_width_mm=round(f["slot_width"], 2),
                       open_side="-x_A", tines="4 mm thick at y_A 267.1..271.1 and 291.5..295.5, x_A -14.2..4.6, "
                                                "z_A -10..0", hole_d_mm=f["hole_d"], hole_xy_A=f["hole_xy"],
                       reading="a 20.4 x 20.4 mm pocket fits a 20 x 20 mm extrusion with 0.2 mm per side; the 5.4 mm "
                               "hole fits M5 (2020 T-nut size) and lies on the slot's centreline. The owner's "
                               "d435_mount_80-20.stl (same evening) takes a D435 on two holes 45 mm apart and has two "
                               "rails about 21 mm apart (read from its render, mount8020_views.png) that straddle a 20 mm extrusion. "
                               "Inferred, not measured on the hardware.")),
        dict(solid=None, label="D435 body", name="Intel RealSense D435",
             note="not in the export: no solid of the six STEP files is 90 x 25 x 25 mm; the hand file has 33 B-rep "
                  "part definitions and 127 instances (121 solids + 6 sub-assemblies) and no tessellated bodies"),
    ],
    tag_parts=[
        dict(solid="hand/005", label="Part 006", name="static_tag_mount",
             identified_by="the same solid as ~/Downloads/static_tag_mount.stl (2026-08-30 20:34) turned +90 deg "
                           "about z: surface distance max 0.35 mm, mean 0.09 mm",
             T_A_from_part=an["T_A_from_part"]["005"], bbox_A=[0.1, 103.3, -57.0, 45.1, 120.0, 18.0],
             plate="2.0 mm thick, x_A 0.1..45.1, y_A 103.3..105.3, z_A -57..-7: a 45.0 x 50.0 mm face on -y "
                   "(2444 mm^2, flat up to the post) and 45.0 x 48.0 mm clear on +y (a 2 mm fillet and a gusset above "
                   "z_A -9)",
             fastening="one screw at (6.5, 117.0) through its 3.4 mm hole and the d435_plate's tab; its post "
                       "(x_A 11.2..14.2, up to z_A 18) lies 0.2 mm off the +x side faces of the base plate and the "
                       "robot mount at x_A 11.0, which sets its yaw"),
    ],
    camera_mount_assumption=(
        "ASSUMED, not in the CAD: a vertical 20 x 20 mm extrusion standing in the fork's slot (centre x_A -4.0, "
        "y_A 281.3), the owner's d435_mount_80-20 (4 mm plate) on its -y face, the D435's back on the bracket by its "
        "two rear M3 holes, upright (1/4-20 down), lenses facing -y_A toward the hand, the bracket just below the "
        "tines: body centre (-4.0, 254.78, -33.3). Lower mounts move the frames along z_A only (variant tips_level: "
        "z_A -137.85)."),
    T_A_from_depth_optical=hi["T_A_from_depth_optical"],
    T_A_from_color_optical=hi["T_A_from_color_optical"],
    depth_optical_axes_A=dict(x=[-1, 0, 0], y=[0, 0, -1], z=[0, -1, 0], origin=hi["depth_origin_A"]),
    color_optical_axes_A=dict(x=[-1, 0, 0], y=[0, 0, -1], z=[0, -1, 0], origin=hi["color_origin_A"]),
    camera_axis_vs_minus_zA_deg=hi["axis_vs_minus_zA_deg"],
    sees_fingertips=False,
    sees_fingertips_detail=(
        "Under the assumed mount the optical axis is horizontal (90 deg from -z_A) and 107 mm to the +x side of the "
        "hand's centre line (x_A -111). No fingertip is in either field of view: the tips are 28.6 to 50.0 deg right "
        "of the axis and 32 to 47 deg below it at 181 to 271 mm (fingertip centroid 213.7 mm, 44.2 deg right, 38.0 "
        "deg down). The tag is 3.7 deg left of the axis at 141.5 mm. Hung at the fingertips' height (z_A -137.85) the "
        "camera sees one tip (Finger v3 <2>, x_A -59, 152 mm) and the +x half of a grasped object; the thumb and "
        "Finger v3 <1> tips stay 49.6 and 50.0 deg right. Aimed at the fingertip centroid from the highest mount "
        "(44.2 deg yaw toward -x_A, 29.3 deg pitch down) all three tips would lie within 14 deg of the image centre."),
    distance_lens_to_fingertips_mm=dict(highest=hi["depth_view"]["fingertip centroid"]["range_mm"],
                                        tips_level=lo["depth_view"]["fingertip centroid"]["range_mm"]),
    views=dict(highest=views(hi), tips_level=views(lo),
               table="the table is in the lower part of the image: with the axis horizontal, a table d mm below the "
                     "lens appears beyond 1.8 d of horizontal range (depth half-angle 29 deg)"),
    variants=dict(highest=dict(body_centre_A=hi["body_centre_A"], T_A_from_depth_optical=hi["T_A_from_depth_optical"],
                               T_A_from_color_optical=hi["T_A_from_color_optical"]),
                  tips_level=dict(body_centre_A=lo["body_centre_A"], T_A_from_depth_optical=lo["T_A_from_depth_optical"],
                                  T_A_from_color_optical=lo["T_A_from_color_optical"])),
    d435_model=dict(source="Intel D400 datasheet (body 90 x 25 x 25 mm, depth origin = left IR imager, depth zero 4.2 "
                           "mm behind the front glass) and realsense-ros _d435.urdf.xacro nominal offsets",
                    body_frame="origin at the body's centre, axes as the optical frames", **an["d435"],
                    fov_deg=an["fov"],
                    note="nominal offsets; the factory extrinsics (rs-enumerate-devices -c) replace them on the "
                         "real camera. Not re-measured from the realsense-ros mesh: resguard refused mesh jobs "
                         "(swap 15.5 GB > 15 GB) from 17:3x on 2026-11-04."),
    tag=dict(size_mm=40.0, family="tag36h11", id=6, size_definition="black-square edge = pupil_apriltags tag_size",
             pose_A=an["T_A_from_tag_pos_face"], centre_A=an["tag_centre_pos_face_A"], normal_A=[0.0, 1.0, 0.0],
             frame="apriltag convention: origin at the tag centre, x right, y down, z into the tag in the tag's own "
                   "image; mounted with x up (+z_A), y = -x_A, z = -y_A, as the 2026-08-30 probe shows (tag x along "
                   "image up) and tags.py assumes (--up-axis x)",
             face="+y face (y_A 105.3), facing the thumb end and the assumed camera; the CAD does not mark which face "
                  "has the tag", printable_area="not modelled; the plate face is 45 x 50 mm, the printed sheet "
                                                     "(gen_tags.py) has a 60 mm patch, and the 2026-08-30 probe shows "
                                                     "it trimmed to about a 2.5 mm white border",
             alternative_face=dict(pose_A=an["T_A_from_tag_neg_face"], normal_A=[0.0, -1.0, 0.0],
                                   centre_A=[an["tag_centre_pos_face_A"][0], an["tag_plate"]["y_neg"], -32.0]),
             purpose="in the repo, id 6 was the static reference for an external D435 on a tripod (real_v1 bench, "
                     "2026-08-30/31): world up and a height datum, at 320-500 mm. On the cad_v3 hand it rides with "
                     "the hand; under the assumed camera mount the hand's own D435 sees it face-on at 141 mm"),
    validation=dict(method="pupil_apriltags on the MuJoCo renders from the computed optical frames (pinhole, square "
                           "pixels, centred principal point)", **va),
)
(HERE / "sensors.json").write_text(json.dumps(data, indent=1))
print("wrote", HERE / "sensors.json")
