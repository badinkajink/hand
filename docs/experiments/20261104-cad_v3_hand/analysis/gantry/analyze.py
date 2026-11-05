#!/usr/bin/env python3
"""Gantry kinematics of the SR2 hand CAD (file "hand") and the map from the CAD frame A to the simulator's palm frame
P. Reads parts.json (B-rep axes and faces), travel.json / travel2.json (sweeps) and, when present, register.json (ICP
cross-check of the finger registration); writes gantry.json. No meshes are loaded.

    .venv/bin/python logs/20261104-cad_v3/gantry/analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import classify  # noqa: E402

REPO = HERE.parents[2]
PARTS = classify.PARTS
NAME = {"g1": "index", "g2": "middle", "gt": "thumb"}
GID = {"thumb": 0, "index": 1, "middle": 2}
JOINTS = {"thumb": (0, 1), "index": (2, 3), "middle": (4, 5)}      # kinematics.STEPPER_JOINTS: (firmware x, y)
TRAVEL = {0: 110.0, 1: 60.0, 2: 60.0, 3: 60.0, 4: 60.0, 5: 60.0}   # kinematics.FULL_EXTENSION_MM
NOMINAL = {"thumb": (-50.0, 0.0), "index": (50.0, 55.0), "middle": (50.0, -55.0)}   # kinematics.FINGER_GEOMETRY
NOMINAL_BOX = {"thumb": (60.0, 110.0), "index": (60.0, 60.0), "middle": (60.0, 60.0)}
R_PA = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])     # Rz(+90 deg): x_P = -y_A, y_P = x_A
THUMB_SHIFT = np.array([0.0, 2.5, 0.0])                                   # thumb gantry onto the base plate's holes
THUMB_STAY = ["060", "051", "054"]       # thumb X motor mount, motor and shaft: already on the base plate's hole row


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def foot(p1, a1, p2, a2):
    n = unit(np.cross(a1, a2))
    t = np.linalg.solve(np.column_stack([a1, -a2, n]), p2 - p1)
    return p1 + t[0] * a1, p2 + t[1] * a2


def cyl(p, r, tol=0.06):
    return [c for c in p["cylinders"] if abs(c["r"] - r) < tol]


def finger_brep(inst):
    """Yaw centre, F axes and CAD joint angles of one Finger v3 instance from its B-rep axes and faces."""
    parts = [p for p in PARTS.values() if classify.sub(p) == inst]
    g = lambda v: [p for p in parts if abs(p["volume"] - v) < 3]     # noqa: E731
    car, yl, ml, tip = g(23913)[0], g(10683)[0], g(21293)[0], g(4822)[0]
    b, m, pp = cyl(car, 5.1)[0], cyl(yl, 5.1)[0], cyl(ml, 5.1)[0]
    p1, a1 = np.asarray(b["point"]), unit(b["axis"])
    p2, a2 = np.asarray(m["point"]), unit(m["axis"])
    yc, mc = foot(p1, a1, p2, a2)
    lo, hi = classify.bbox(car)
    servo = [s for s in g(6673) if np.all(np.asarray(s["com"]) > lo) and np.all(np.asarray(s["com"]) < hi)][0]
    xF = a1 if (np.asarray(servo["com"]) - yc) @ a1 > 0 else -a1          # bearing wall -> servo: dorsal
    zF = unit(np.array([0, 0, 1.0]) - xF[2] * xF)
    yF = np.cross(zF, xF)
    R_AF = np.column_stack([xF, yF, zF])
    a2F = R_AF.T @ a2
    a2F = a2F if a2F[1] > 0 else -a2F
    yaw = float(np.arctan2(a2F[2], a2F[1]))
    Ry = Rotation.from_rotvec([-yaw, 0, 0]).as_matrix()
    p3, a3 = np.asarray(pp["point"]), unit(pp["axis"])
    v = (p3 - mc) - ((p3 - mc) @ a2) * a2
    vy = Ry @ (R_AF.T @ v)
    mcp = float(np.arctan2(-vy[0], -vy[2]))
    pl = sorted(tip["planes"], key=lambda f: -f["area"])
    pairs = [(f, h) for i, f in enumerate(pl) for h in pl[i + 1:]
             if np.dot(f["normal"], h["normal"]) < -0.9999
             and abs(abs((np.asarray(f["centre"]) - np.asarray(h["centre"])) @ np.asarray(f["normal"])) - 17.0) < 0.05]
    f, h = max(pairs, key=lambda e: e[0]["area"] + e[1]["area"])
    back = f if f["area"] > h["area"] else h
    nb = Ry @ (R_AF.T @ np.asarray(back["normal"]))
    pip = float(np.arctan2(-nb[2], nb[0])) - mcp
    tilt = float(np.degrees(np.arcsin(xF[2])))
    return dict(carriage=car["name"], yaw_link=yl["name"], mcp_link=ml["name"], tip=tip["name"], yaw_servo=servo["name"],
                yaw_centre=yc, mcp_centre=mc, pip_centre=mc + v, R_AF=R_AF, yaw=yaw, mcp=mcp, pip=pip,
                yaw_axis_rise_deg=tilt, yaw_to_mcp=float(np.linalg.norm(mc - yc)), mcp_to_pip=float(np.linalg.norm(v)))


def main():
    roles, finger_of = classify.classify()
    t2 = HERE / "travel2.json"
    T = json.loads(t2.read_text()) if t2.exists() else None
    T1 = json.loads((HERE / "travel.json").read_text())
    reg = json.loads((HERE / "register.json").read_text()) if (HERE / "register.json").exists() else None
    out = {"units": "mm, deg", "frames": {
        "A": "file 'hand' of the owner's Onshape export of 2026-11-04 (assets/cad_v2/full_cad), z_A up toward the robot",
        "P": "simulator palm frame = driver {P}: z_P up, fingers along -z_P, z_P = 0 on the yaw-centre plane",
        "F": "cad_v1 finger frame: origin at the yaw centre, x_F along the yaw axis toward the dorsal side, z_F up"},
        "sources": {"parts": "logs/20261104-cad_v3/extract/parts.json",
                    "travel": "travel2.json" if T else "travel.json (first pass)",
                    "registration": "B-rep axes from parts.json" + ("; ICP cross-check register.json" if reg else "")}}
    fingers = {}
    for g in ("g1", "g2", "gt"):
        name = NAME[g]
        inst = finger_of[g]
        fb = finger_brep(inst)
        r = roles[g]
        if T:
            q = T["gantries"][g]
            xr, yr = q["x_S1_vs_F"], q["y_S2r_vs_S1"]
        else:                                   # first pass: keep the contacts with this gantry's own fixed parts
            q = T1["gantries"][g]
            own = set(n for k in ("frame", "x_rail", "x_screw", "x_motor", "x_coupler", "x_shaft", "x_bearing",
                                  "x_mount") for n in r[k]) | {"000", "001", "002", "004", "005"}
            up = [e for e in q["x_S1_vs_F"]["upper_next"] if e[1] in own][0]
            lo = [e for e in q["x_S1_vs_F"]["lower_next"] if e[1] in own][0]
            xr = dict(lower=lo[2], upper=up[2], lower_pair=lo[:2], upper_pair=up[:2])
            yr = q["y_S2_vs_S1"]
        # stepper (motor) ends: the X motor's side along x_A and the Y motor's side along y_A
        xm = 0.5 * np.add(*classify.bbox(PARTS[r["x_motor"][0]]))[0]
        ym = 0.5 * np.add(*classify.bbox(PARTS[r["y_motor"][0]]))[1]
        frame_c = 0.5 * np.add(*classify.bbox(PARTS[r["frame"][0]]))
        beam_c = 0.5 * np.add(*classify.bbox(PARTS[r["beam"][0]]))
        x_home_sign = 1.0 if xm > frame_c[0] else -1.0
        y_home_sign = 1.0 if ym > beam_c[1] else -1.0
        jx, jy = JOINTS[name]
        s_home = xr["upper"] if x_home_sign > 0 else xr["lower"]
        t_home = yr["upper"] if y_home_sign > 0 else yr["lower"]
        s_box = sorted([s_home, s_home - x_home_sign * TRAVEL[jx]])
        t_box = sorted([t_home, t_home - y_home_sign * TRAVEL[jy]])
        yc = fb["yaw_centre"]
        travel = {
            "x_stage": dict(axis_A=[1, 0, 0], firmware_joint=f"J{jx}", carries="the beam with the Y axis and the finger",
                            parts={k: r[k] for k in ("beam", "x_block", "x_nut", "x_rail", "x_screw", "x_motor",
                                                      "x_coupler", "x_bearing", "x_mount", "frame")},
                            stop_minus=dict(disp=xr["lower"], yaw_centre_x_A=yc[0] + xr["lower"], parts=xr.get("lower_pair")),
                            stop_plus=dict(disp=xr["upper"], yaw_centre_x_A=yc[0] + xr["upper"], parts=xr.get("upper_pair")),
                            stroke=xr["upper"] - xr["lower"], motor_end="+x_A" if x_home_sign > 0 else "-x_A",
                            home_disp=s_home, firmware_mm_increases_toward="-x_A" if x_home_sign > 0 else "+x_A",
                            box_disp=s_box, box_yaw_centre_x_A=[yc[0] + s_box[0], yc[0] + s_box[1]],
                            margin_far_end=(xr["upper"] - s_box[1]) if x_home_sign < 0 else (s_box[0] - xr["lower"])),
            "y_stage": dict(axis_A=[0, 1, 0], firmware_joint=f"J{jy}", carries="the finger (carriage, yaw servo, nut)",
                            parts={k: r[k] for k in ("y_block", "y_rail", "y_screw", "y_motor", "y_coupler", "y_bearing")},
                            stop_minus=dict(disp=yr["lower"], yaw_centre_y_A=yc[1] + yr["lower"], parts=yr.get("lower_pair")),
                            stop_plus=dict(disp=yr["upper"], yaw_centre_y_A=yc[1] + yr["upper"], parts=yr.get("upper_pair")),
                            stroke=yr["upper"] - yr["lower"], motor_end="+y_A" if y_home_sign > 0 else "-y_A",
                            home_disp=t_home, firmware_mm_increases_toward="-y_A" if y_home_sign > 0 else "+y_A",
                            box_disp=t_box, box_yaw_centre_y_A=[yc[1] + t_box[0], yc[1] + t_box[1]],
                            margin_far_end=(yr["upper"] - t_box[1]) if y_home_sign < 0 else (t_box[0] - yr["lower"])),
        }
        box_c_A = np.array([yc[0] + 0.5 * sum(s_box), yc[1] + 0.5 * sum(t_box), yc[2]])
        geo_c_A = np.array([yc[0] + 0.5 * (xr["lower"] + xr["upper"]), yc[1] + 0.5 * (yr["lower"] + yr["upper"]), yc[2]])
        cad_mm = {f"J{jx}": abs(0.0 - s_home), f"J{jy}": abs(0.0 - t_home)}
        R_AF = fb["R_AF"]
        # simulator joint angles: index/middle axes yaw +x_F, mcp/pip +y_F; thumb yaw +x_P = -x_F, mcp/pip -y_P = +y_F
        sim = dict(yaw=np.degrees(fb["yaw"]) * (-1.0 if name == "thumb" else 1.0), mcp=np.degrees(fb["mcp"]),
                   pip=np.degrees(fb["pip"]))
        fingers[name] = dict(
            instance=inst, gantry_instance=classify.GANTRIES[g],
            parts=dict(carriage=fb["carriage"], yaw_servo=fb["yaw_servo"], yaw_link=fb["yaw_link"], mcp_link=fb["mcp_link"],
                       tip=fb["tip"]),
            yaw_centre_A_mm=yc, mcp_centre_A_mm=fb["mcp_centre"], pip_centre_A_mm=fb["pip_centre"],
            yaw_to_mcp_mm=fb["yaw_to_mcp"], mcp_to_pip_mm=fb["mcp_to_pip"],
            axes_A={"x_F": R_AF[:, 0], "y_F": R_AF[:, 1], "z_F": R_AF[:, 2]},
            yaw_axis_rise_toward_dorsal_deg=fb["yaw_axis_rise_deg"],
            cad_joint_angles_deg=dict(about_F=dict(yaw_xF=np.degrees(fb["yaw"]), mcp_yF=np.degrees(fb["mcp"]),
                                                   pip_yF=np.degrees(fb["pip"])), simulator=sim),
            travel_A_mm=travel, box_centre_A_mm=box_c_A, stroke_centre_A_mm=geo_c_A,
            cad_pose_stepper_mm=cad_mm)
    # thumb gantry onto the base plate's holes (+2.5 mm in y_A): its yaw centre and boxes move with it
    corr = {k: dict(v) for k, v in fingers.items()}
    corr["thumb"] = dict(fingers["thumb"])
    corr["thumb"]["box_centre_A_mm"] = fingers["thumb"]["box_centre_A_mm"] + THUMB_SHIFT
    corr["thumb"]["yaw_centre_A_mm"] = fingers["thumb"]["yaw_centre_A_mm"] + THUMB_SHIFT
    corr["thumb"]["stroke_centre_A_mm"] = fingers["thumb"]["stroke_centre_A_mm"] + THUMB_SHIFT

    def place(F):
        """P from A: rotation R_PA; translation puts y_P = 0 midway between the index and middle box centres,
        x_P = 0 midway between the thumb box centre and the index/middle box centres, z_P = 0 on the yaw plane."""
        c = {k: R_PA @ F[k]["box_centre_A_mm"] for k in F}
        z = np.mean([F[k]["yaw_centre_A_mm"][2] for k in F])
        ty = -0.5 * (c["index"][1] + c["middle"][1])
        tx = -0.5 * (c["thumb"][0] + 0.5 * (c["index"][0] + c["middle"][0]))
        t = np.array([tx, ty, -z])
        rows = {}
        for k in F:
            bc = R_PA @ F[k]["box_centre_A_mm"] + t
            gc = R_PA @ F[k]["stroke_centre_A_mm"] + t
            yc_P = R_PA @ F[k]["yaw_centre_A_mm"] + t
            nom = np.array(NOMINAL[k])
            ts = F[k]["travel_A_mm"]
            rows[k] = dict(box_centre_P_mm=bc, box_minus_nominal_mm=bc[:2] - nom,
                           stroke_centre_P_mm=gc, cad_pose_yaw_centre_P_mm=yc_P,
                           box_size_mm=dict(commanded_xP_yP=[ts["y_stage"]["box_disp"][1] - ts["y_stage"]["box_disp"][0],
                                                             ts["x_stage"]["box_disp"][1] - ts["x_stage"]["box_disp"][0]],
                                            stop_to_stop_xP_yP=[ts["y_stage"]["stroke"], ts["x_stage"]["stroke"]],
                                            nominal_xP_yP=list(NOMINAL_BOX[k])),
                           axes_P={a: R_PA @ np.asarray(v) for a, v in F[k]["axes_A"].items()})
        return t, rows

    t_cad, rows_cad = place(fingers)
    t_cor, rows_cor = place(corr)
    lsq = {}
    cA = {k: R_PA @ fingers[k]["box_centre_A_mm"] for k in fingers}
    tl = np.mean([np.array(NOMINAL[k]) - cA[k][:2] for k in fingers], axis=0)
    lsq = dict(t_xy=tl, residual={k: cA[k][:2] + tl - np.array(NOMINAL[k]) for k in fingers})
    for k in fingers:
        fingers[k]["P_with_thumb_on_base_plate_holes"] = rows_cor[k]
        fingers[k]["P_cad_as_is"] = rows_cad[k]
        fingers[k]["box_centre_P_mm"] = rows_cor[k]["box_centre_P_mm"]
        fingers[k]["box_size_mm"] = rows_cor[k]["box_size_mm"]
    out["fingers"] = fingers
    out["R_P_from_A"] = R_PA
    out["t_P_from_A_mm"] = t_cor
    out["t_P_from_A_mm_cad_as_is"] = t_cad
    out["t_P_from_A_note"] = ("x_P = -y_A + t_x, y_P = x_A + t_y, z_P = z_A + t_z. t_y puts y_P = 0 midway between the "
                              "index and middle box centres, t_x puts x_P = 0 midway between the thumb box centre and the "
                              "index/middle box centres (the driver's convention, XY_space.png), t_z puts the yaw "
                              "centres at z_P = 0. Primary t uses the thumb gantry moved +2.5 mm in y_A onto the base "
                              "plate's holes; t_P_from_A_mm_cad_as_is uses the CAD as exported.")
    out["least_squares_fit_to_nominal_cad_as_is"] = lsq
    out["thumb_gantry_correction"] = dict(
        shift_A_mm=THUMB_SHIFT, parts_that_move="every part of 'gantry_v3_thumb <1>' and 'Finger v3 <3>' except "
        + ", ".join(THUMB_STAY), parts_that_stay=THUMB_STAY)
    out["yaw_plane_z_A_mm"] = float(np.mean([fingers[k]["yaw_centre_A_mm"][2] for k in fingers]))
    out["yaw_centre_z_A_mm"] = {k: float(fingers[k]["yaw_centre_A_mm"][2]) for k in fingers}
    zy = out["yaw_plane_z_A_mm"]

    def zmin(names):
        n = min(names, key=lambda q: PARTS[q]["bbox"][2])
        return float(PARTS[n]["bbox"][2]), n

    def down_faces(n):
        return sorted({round(f["centre"][2], 3) for f in PARTS[n]["planes"] if f["normal"][2] < -0.999})

    fixed_roles = ("frame", "x_rail", "x_screw", "x_motor", "x_coupler", "x_shaft", "x_bearing", "x_mount")
    beam_roles = ("beam", "x_block", "x_nut", "y_rail", "y_screw", "y_motor", "y_coupler", "y_shaft", "y_bearing")
    per = {}
    for g in ("g1", "g2", "gt"):
        r = roles[g]
        fr = r["frame"][0]
        fixed = [n for k in fixed_roles for n in r[k]]
        inside = [n for n in fixed if n not in r["x_motor"] + r["x_mount"] + r["x_coupler"] + r["x_shaft"]]
        beam = [n for k in beam_roles for n in r[k]]
        z_fix, n_fix = zmin(fixed)
        z_in, n_in = zmin(inside)
        z_bm, n_bm = zmin(beam)
        fdn = down_faces(fr)
        bdn = down_faces(r["beam"][0])
        car = fingers[NAME[g]]["parts"]["carriage"]
        per[NAME[g]] = dict(
            frame_plate_underside_A=max(z for z in fdn if z <= -3.9), frame_downward_faces_A=fdn,
            frame_lowest_A=float(PARTS[fr]["bbox"][2]),
            lowest_fixed_part_A=dict(z=z_fix, part=n_fix, label=PARTS[n_fix]["label"],
                                     what=[k for k in fixed_roles if n_fix in r[k]][0]),
            lowest_fixed_part_within_frame_footprint_A=dict(z=z_in, part=n_in,
                                                            what=[k for k in fixed_roles if n_in in r[k]][0]),
            beam_downward_faces_A=bdn, beam_main_underside_A=-28.5 if -28.5 in bdn else None,
            lowest_moving_gantry_part_A=dict(z=z_bm, part=n_bm, what=[k for k in beam_roles if n_bm in r[k]][0]),
            carriage_bottom_A=float(PARTS[car]["bbox"][2]), carriage_top_A=float(PARTS[car]["bbox"][5]))
        for k, v in list(per[NAME[g]].items()):
            if isinstance(v, float):
                per[NAME[g]][k.replace("_A", "_above_yaw_plane")] = v - zy
            elif isinstance(v, dict) and "z" in v:
                v["z_above_yaw_plane"] = v["z"] - zy
    levels = {
        "flange_face": (28.0, "UR5 flange face (adaptor top)"),
        "robot_mount_underside": (float(PARTS["001"]["bbox"][2]), "robot mount underside"),
        "base_plate_underside": (float(PARTS["002"]["bbox"][2]), "base plate underside"),
        "frame_plate_underside": (per["index"]["frame_plate_underside_A"], "gantry frame plate underside"),
        "x_block_underside": (float(PARTS[roles["g1"]["x_block"][0]]["bbox"][2]), "X rail blocks underside"),
        "frame_lowest": (per["index"]["frame_lowest_A"], "frame lowest (X screw bearing blocks)"),
        "beam_main_underside": (-28.5, "beam underside over the finger travel"),
        "beam_bearing_blocks": (-51.5, "beam Y-screw bearing blocks"),
        "beam_lowest": (per["index"]["lowest_moving_gantry_part_A"]["z"], "beam motor end, Y motors"),
        "yaw_plane": (zy, "yaw-centre plane"),
        "carriage_bottom": (per["index"]["carriage_bottom_A"], "finger carriage bottom"),
    }
    out["heights"] = dict(levels_A_mm=levels, levels_above_yaw_plane_mm={k: v[0] - zy for k, v in levels.items()},
                          per_finger=per,
                          sensor_bracket_005=dict(z_A=[float(PARTS["005"]["bbox"][2]), float(PARTS["005"]["bbox"][5])],
                                                  lowest_above_yaw_plane=float(PARTS["005"]["bbox"][2]) - zy),
                          note="z_A of the lowest point of each group (bounding-box extremes of the B-rep solids) and "
                               "of downward planar faces; *_above_yaw_plane = z_A - yaw_plane_z_A = z_P")
    out["heights_A_mm"] = {k: v[0] for k, v in levels.items()}

    # the thumb gantry against the index/middle gantries along y_A, from the B-rep extremes (planar end faces)
    def ymax(names):
        return max(PARTS[n]["bbox"][4] for n in names)

    def ymin(names):
        return min(PARTS[n]["bbox"][1] for n in names)
    pair_stage = [n for g in ("g1", "g2") for k in ("beam", "x_block") for n in roles[g][k]]
    thumb_stage = [n for k in ("beam", "x_block") for n in roles["gt"][k]]
    pair_frames = [roles[g]["frame"][0] for g in ("g1", "g2")]
    thumb_frame = roles["gt"]["frame"][0]
    tab_half = 7.5                                    # rail-end tabs span the rail centre +- 7.5 mm (faces of 074)
    pair_tab_ymax = 112.5 + tab_half
    thumb_rail_c = 0.5 * (PARTS[roles["gt"]["x_rail"][0]]["bbox"][1] + PARTS[roles["gt"]["x_rail"][0]]["bbox"][4])
    thumb_tab_ymin = thumb_rail_c - tab_half
    rows = {}
    for tag, dy in (("cad_as_is", 0.0), ("thumb_on_base_plate_holes", float(THUMB_SHIFT[1]))):
        rows[tag] = {
            "frames_gap": ymin([thumb_frame]) + dy - ymax(pair_frames),
            "beams_and_X_blocks_gap": ymin(thumb_stage) + dy - ymax(pair_stage),
            "pair_X_blocks_vs_thumb_frame_rail_end_tabs": thumb_tab_ymin + dy - ymax([n for g in ("g1", "g2") for n in roles[g]["x_block"]]),
            "thumb_X_block_vs_pair_frame_rail_end_tabs": ymin(roles["gt"]["x_block"]) + dy - pair_tab_ymax}
    holes = dict(thumb_frame_corner_holes_y_A=sorted({round(c["point"][1], 3) for c in PARTS[thumb_frame]["cylinders"]
                                                      if abs(c["r"] - 2.2) < 0.01 and c["concave"]}),
                 base_plate_holes_for_thumb_frame_y_A=[142.875, 238.375],
                 thumb_x_motor_mount_centre_y_A=0.5 * (PARTS["060"]["bbox"][1] + PARTS["060"]["bbox"][4]),
                 thumb_x_coupler_centre_y_A=0.5 * (PARTS["061"]["bbox"][1] + PARTS["061"]["bbox"][4]),
                 base_plate_motor_mount_hole_y_A=230.375)
    out["cross_gantry_clearance_y_A_mm"] = dict(gaps=rows, evidence=holes, note=(
        "negative = overlap. Stages pass each other in x_A, so an overlap in y_A at shared z_A is a collision at some "
        "pair of X positions (the first sweep found them: thumb X stage blocked 1.6 mm from its CAD pose by the middle "
        "frame's rail-end tab). The base plate's holes for the thumb frame sit 2.5 mm further +y_A than the frame's "
        "holes in the assembly, and the thumb's X motor (on the base plate's hole row) is 2.5 mm off its coupler and "
        "screw; moving the thumb gantry +2.5 mm makes both consistent and clears every overlap."))
    flange_c_A = np.array([-111.0, 116.0, 28.0])
    out["flange_axis_P_mm"] = R_PA @ flange_c_A + t_cor
    out["identification_evidence"] = [
        "R_P_from_A = Rz(+90 deg) puts every stage's home (driver: thumb bottom-left, index top-right, middle bottom-"
        "right) at the end that carries its stepper motor, for all six axes; the mirror map that also puts the thumb at "
        "-x_P homes the thumb's J0 at the end away from its motor",
        "the X motors of the index/middle gantries sit outboard (+x_A on gantry_v3 <1>, -x_A on <2>, the only sides "
        "with motor-mount holes in the base plate); the thumb's X motor sits at -x_A (the +x_A end carries the sensor "
        "plate 000)",
        "each finger's dorsal direction (yaw axis from the bearing wall to the yaw servo) is -y_A on the pair and +y_A "
        "on the thumb, i.e. +x_P and -x_P under Rz(+90 deg), as the simulator places them (thumb turned 180 deg)",
        "XY_space.png read from the finger side (u = -x_A, v = y_A) gives the driver's {P} as (x_P, y_P) = "
        "(-(v - 80), -(u + 25)) = (-y_A, x_A) + const",
        "fingertip centres of mass, with the CAD joint angles undone, sit at (-2.046, 0, -26.508) mm from the PIP "
        "centre in all three fingers, the cad_v1 tip body offset"]
    if reg:
        out["icp_cross_check"] = {k: reg["fingers"][fingers[k]["instance"]] for k in fingers}
    (HERE / "gantry.json").write_text(json.dumps(out, indent=1, default=lambda o: np.round(np.asarray(o, float), 4).tolist()
                                                 if isinstance(o, (np.ndarray, np.floating)) else o))
    print("wrote", HERE / "gantry.json")
    for k in ("thumb", "index", "middle"):
        f = fingers[k]
        print(k, f["instance"], f["gantry_instance"], "yaw centre A", np.round(f["yaw_centre_A_mm"], 3),
              "box centre P", np.round(f["box_centre_P_mm"], 3), "CAD as-is", np.round(rows_cad[k]["box_centre_P_mm"], 3),
              "stepper", {j: round(v, 3) for j, v in f["cad_pose_stepper_mm"].items()},
              "angles", {j: round(v, 3) for j, v in f["cad_joint_angles_deg"]["simulator"].items()})
        for ax in ("x_stage", "y_stage"):
            s = f["travel_A_mm"][ax]
            print("   ", ax, s["firmware_joint"], "stops", round(s["stop_minus"]["disp"], 3), s["stop_minus"]["parts"],
                  round(s["stop_plus"]["disp"], 3), s["stop_plus"]["parts"], "stroke %.3f" % s["stroke"], "motor end",
                  s["motor_end"], "box", np.round(s["box_disp"], 3), "far margin %.3f" % s["margin_far_end"])
    print("t_P_from_A", np.round(t_cor, 4), "as-is", np.round(t_cad, 4), "lsq", np.round(lsq["t_xy"], 4),
          {k: np.round(v, 3) for k, v in lsq["residual"].items()})


if __name__ == "__main__":
    main()
