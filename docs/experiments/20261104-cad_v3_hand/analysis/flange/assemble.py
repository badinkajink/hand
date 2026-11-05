#!/usr/bin/env python3
"""Compose flange.json from flange_raw.json (flange.py), step_check.json (step_check.py) and wrist3_features.json
(probe_wrist3*.py).

    .venv/bin/python logs/20261104-cad_v3/flange/assemble.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

OUT = Path(__file__).resolve().parent
RAW = json.load(open(OUT / "flange_raw.json"))
STEP = json.load(open(OUT / "step_check.json"))
WR3 = json.load(open(OUT / "wrist3_features.json"))


def clock(alpha_deg, from_tool=True):
    """Clock position of a direction at alpha (deg from x_E about +z_E) with the tool connector (-y_E) at 6 o'clock;
    seen from the tool side (z_E toward the viewer) or from the robot side (mirrored)."""
    h = 3.0 - alpha_deg / 30.0
    if not from_tool:
        h = 12.0 - h
    h = h % 12.0
    return 12.0 if round(h) == 0 else float(round(h))


def main():
    ad = RAW["adaptor_in_A"]
    c_A = RAW["c_A"]
    ks = RAW["clockings"]
    R_A_from_P = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # columns: x_P = -y_A, y_P = x_A, z_P = z_A
    near_axis = {}
    desc = []
    for ck in ks:
        k = ck["k"]
        d = ck["gantry_direction_from_flange_axis_E_deg"]
        rel = ck["connector_minus_gantry_direction_deg"]
        th = d["thumb"]
        ax_names = {0: "+x_E", 90: "+y_E", 180: "-x_E", -90: "-y_E", 270: "-y_E", -180: "-x_E"}
        a0 = min(ax_names, key=lambda a: abs(((th - a + 180) % 360) - 180))
        near_axis[k] = (ax_names[a0], round(abs(((th - a0 + 180) % 360) - 180), 1))
        R = np.array(ck["T_E_from_A"])[:3, :3]
        RP = R @ R_A_from_P
        ck["R_E_from_P_inferred"] = np.round(RP, 6).tolist()
        ck["x_P_angle_E_deg"] = round(math.degrees(math.atan2(RP[1, 0], RP[0, 0])), 3)
        ck["clock_tool_side"] = {g: clock(a) for g, a in d.items()}
        ck["clock_robot_side"] = {g: clock(a, False) for g, a in d.items()}
        ck["mjcf_pos_m"] = [round(v / 1000.0, 7) for v in ck["pos_A_origin_in_E_mm"]]
        ck["mjcf_quat_wxyz"] = ck["quat_wxyz_E_from_A"]
    descriptions = {
        0: "k = 0 (x_A at 24 deg from x_E): the thumb gantry points along -y_E turned 24 deg toward +x_E (thumb at -66 "
           "deg), toward the tool connector's side; the connector sits 24 deg from the thumb's direction, on the middle "
           "finger's side. By eye: the tool connector points almost at the thumb gantry.",
        1: "k = 1 (x_A at 114 deg): the thumb points along +x_E turned 24 deg toward +y_E (thumb at 24 deg); the "
           "connector sits 21 deg from the middle-finger gantry's direction, on the thumb's side, and 114 deg from the "
           "thumb. By eye: the connector points almost at the middle-finger gantry (the one away from the camera "
           "parts).",
        2: "k = 2 (x_A at 204 deg): the thumb points along +y_E turned 24 deg toward -x_E (thumb at 114 deg), away from "
           "the connector; the connector sits 21 deg from the index gantry's direction, on the middle finger's side. By "
           "eye: the connector points almost at the index gantry (the one on the camera parts' side), and the thumb is "
           "on the far side of the wrist.",
        3: "k = 3 (x_A at 294 deg): the thumb points along -x_E turned 24 deg toward -y_E (thumb at -156 deg); the "
           "connector points between the thumb (66 deg away) and the index gantry (69 deg away), and the middle-finger "
           "gantry is on the opposite side of the wrist (159 deg away). By eye: no gantry in line with the connector; "
           "the middle finger faces away from it.",
    }
    for ck in ks:
        k = ck["k"]
        ck["by_eye"] = descriptions[k]
        ck["thumb_near_axis"] = dict(axis=near_axis[k][0], off_deg=near_axis[k][1])

    evidence = [
        dict(source="ur_description ur5.urdf (rh56_controller/magpie_force_control/src/ur_description; "
                    "ros-industrial universal_robot 6507bb67)",
             finding="tool0 = flange (rpy 0,-pi/2,-pi/2) composed with flange-tool0 (rpy pi/2,0,pi/2) = wrist_3_link "
                     "exactly (rotation = identity); scripts/ur5/ur5_cb3_cell.py check: URDF tool0 against the CB3 DH model "
                     "(d6 82.3 mm) 0.0 mm and 0.0 deg at 5 random joint sets, so tool0 is the controller's flange frame "
                     "with a zero TCP.",
             what="measured (numerically, this session)"),
        dict(source="ur_description meshes/ur5/visual/wrist3.dae (Blender 2.79 export, 2018-07-11), placed as in the "
                    "URDF (origin z -81.8 mm, rpy pi/2,0,0), read with ElementTree",
             finding="on the flange face z_E = 0: four 5.0 mm holes (M6 tap drill, 8.0 mm deep) on a 50.000 mm circle at "
                     "45.0, 135.0, 225.0, 315.0 deg from x_E (circle fits, rms 1e-4 mm); the 31.5 mm recess 6.0 mm deep "
                     "and the 63.0 mm face; the tool connector boss (7 mm across, end face at r 43.0 mm, axis 19.5 mm "
                     "behind the face) at -90.0 deg, on -y_E. The mesh has no dowel hole.",
             what="measured (mesh geometry)", data=WR3),
        dict(source="Universal Robots, UR5/CB3 User Manual, Version 3.1 (rev. 18026), US version, Figure 3.2 'The tool "
                    "output flange, ISO 9409-1-50-4-M6' (p. I-16), "
                    "https://s3-eu-west-1.amazonaws.com/ur-support-site/16073/UR5_User_Manual_en_US.pdf",
             finding="the 6 H7 pin hole lies on the 50 mm circle on the same radial line as the Lumberg RKMV 8-354 tool "
                     "connector; the four M6 holes (6.0 deep) are 45 deg from that line and 90 deg apart; the connector's "
                     "axis is 19.5 mm behind the face (the mesh: 19.48). Text, p. I-14: 'four M6 thread holes ... the "
                     "6 hole is provided for use with a pin'.",
             what="read from the drawing (rendered page)"),
        dict(source="Universal Robots forum, 'Base coordinate system documentation', reply by user jbm, 2016-08-01 "
                    "(https://forum.universal-robots.com/t/base-coordinate-system-documentation/140/2); role not stated",
             finding="'Tool coordinate system / Tool feature: Right-hand coordinate system, where Y is away from the "
                     "Tool connector, and Z is normal to the tool flange.' So the connector is on -y_E.",
             what="quoted"),
        dict(source="docs/experiments/20261022-ur5_tip_calibration/lab/README.md, 'The geometry as measured' (lab UR5, "
                    "2026-10-22)",
             finding="a calibration mount located by its dowel in the HEX-E QC tool plate, with x_M toward that dowel, "
                     "was measured at -92.15 deg from x_E (palmar side along -y_E). The stack's dowel is therefore on "
                     "-y_E within 2.15 deg, the side of the flange's own pin hole if the HEX-E QC keeps its dowel in line "
                     "with the flange's.",
             what="measured on the robot; its bearing on the bare flange is inferred"),
    ]
    T_by_k = [np.array(ck["T_E_from_A"]).ravel().round(6).tolist() for ck in ks]
    out = dict(
        when="2026-11-04",
        frames=dict(
            A="hand assembly 'hand v3 (anthro config).step' (mm): z_A up toward the robot, base plate z 0..3, robot "
              "mount z 3..28, adaptor z 6..28",
            E="tool0 of the UR5 CB3 (controller flange frame, zero TCP; = URDF tool0 = wrist_3_link): origin at the "
              "centre of the flange face, z_E out of the flange toward the tool, y_E away from the tool connector",
            P="the driver's palm frame {P} (inferred mapping, not part of the CAD): x_P = -y_A, y_P = x_A, z_P = z_A"),
        plate_hole_residuals_mm=[h["residual_mm"] for h in ad["plate_side_holes"]],
        plate_holes_A_mm=[h["adaptor_hole_A_mm"] for h in ad["plate_side_holes"]],
        plate_hole_mate_unique=dict(
            proper_self_maps=len(ad["plate_side_hole_pattern_self_maps"]["proper_rotations"]),
            next_best_rms_mm=ad["plate_side_hole_pattern_self_maps"]["next_best_proper_rms_mm"],
            note="the isosceles trapezoid maps onto itself only by the identity among rotations (next best 7.78 mm "
                 "rms); its mirror symmetry about x_A = -111 would need the adaptor turned over, which puts the M6 "
                 "counterbores against the flange"),
        flange_centre_A_mm=c_A,
        flange_face_z_A_mm=ad["flange_face"]["z_A_mm"],
        tool_z_in_A=[0.0, 0.0, -1.0],
        m6_angles_A_deg=[h["angle_A_deg"] % 360 for h in ad["m6_holes"]],
        m6_bolt_circle=ad["m6_bolt_circle"],
        m6_angles_E_deg=[45.0, 135.0, 225.0, 315.0],
        dowel_and_connector_E_deg=-90.0,
        convention_evidence=evidence,
        T_E_from_A_by_k=T_by_k,
        k_rule="k = 0 is the clocking in which x_A's image in the flange plane makes the smallest angle with x_E "
               "(24.0 deg); k = 1, 2, 3 turn the hand by a further 90, 180, 270 deg about +z_E, so x_A lies at "
               "theta_k = 24 + 90 k deg from x_E, measured about +z_E. With z_A = -z_E, a direction at angle a in A "
               "lies at theta_k - a in E. On the robot: y_E points away from the tool connector, x_E is 90 deg "
               "clockwise from y_E seen from the robot side and counterclockwise seen from the tool side.",
        k_descriptions=[descriptions[k] for k in range(4)],
        theta_k_deg=[ck["theta_deg"] for ck in ks],
        clockings=ks,
        offsets=RAW["offsets"],
        standalone_frames=RAW["standalone_frames"],
        step_text_check=dict(
            adaptor_file_vs_freecad_max_mm=max(v["vs_freecad_max_mm"] for k, v in STEP["adaptor"].items()
                                               if k[:1] == "r" and k[1:2].isdigit()),
            robot_mount_file_vs_freecad_max_mm=max(v["vs_freecad_max_mm"] for k, v in STEP["robot_mount"].items()
                                                   if k[:1] == "r" and k[1:2].isdigit()),
            hand_adaptor_in_A_vs_freecad_max_mm=max(v["max_line_distance_mm"] for v in
                                                    STEP["hand"]["adaptor"][0]["checks"].values()),
            hand_robot_mount_in_A_vs_freecad_max_mm=max(v["max_line_distance_mm"] for v in
                                                        STEP["hand"]["robot_mount"][0]["checks"].values()),
            convention="child to parent = A(item_2) A(item_1)^-1 of the ITEM_DEFINED_TRANSFORMATION",
            note="own parser of the STEP text (no FreeCAD): hole axes of the adaptor and the robot mount in their files "
                 "and, through the assembly's instance transforms, in A"),
        adaptor_details=dict(
            counterbores=ad["m6_counterbores"], side_normals_A_deg=ad["adaptor_side_normals_A_deg"],
            half_width_mm=ad["adaptor_half_width_mm"], pocket_half_width_mm=ad["pocket_half_width_mm"],
            flange_face_area_mm2=ad["flange_face"]["area_mm2"], robot_mount_top_z_A_mm=ad["robot_mount_top_z_A_mm"],
            highest_point_of_hand_z_A_mm=ad["highest_point_of_hand_z_A_mm"],
            spigot_or_dowel="none: the flange face is flat apart from the four 6.4 mm and four 4.4 mm holes (area "
                            "3115.3 mm^2 = the rounded square less the eight holes)",
            play_inferred=dict(m6_radial_clearance_mm=0.2, rotation_about_tool_axis_deg=round(math.degrees(0.2 / 25.0), 2),
                               pocket_clearance_per_side_mm=0.3,
                               note="from nominal sizes: M6 (6.0) in 6.4 mm holes on a 25 mm radius; the adaptor "
                                    "(60.4 mm) in the 61.0 mm pocket"),
            m6_screw_length=dict(grip_mm=15.0, cb3_thread_depth_mm=6.0,
                                 note="head seat at z_A = 13, flange face at 28: an ISO 4762 M6 x 20 enters the flange "
                                      "5 mm; the CB3 drawing gives 6.0 mm of thread, and UR's article warns against "
                                      "bolts more than 8 mm into the flange")),
    )
    json.dump(out, open(OUT / "flange.json", "w"), indent=1)
    print("wrote", OUT / "flange.json")
    for ck in ks:
        print(ck["k"], ck["theta_deg"], ck["thumb_near_axis"], "clock tool side", ck["clock_tool_side"],
              "robot side", ck["clock_robot_side"], "x_P at", ck["x_P_angle_E_deg"])
        print("   T_E_from_A", np.array(ck["T_E_from_A"]).round(4).tolist())


if __name__ == "__main__":
    main()
