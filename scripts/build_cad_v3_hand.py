#!/usr/bin/env python3
"""cad_v3: the SR2 hand as built, on the lab's UR5 (CB3), in MuJoCo.

Owner, 2026-11-04: "it's time to go sim2real for real now on the sr2 hand, meaning we are putting the hardware hand on the
ur5 ... we do need to fully understand it (and probably have a visual mesh version to look pretty for renders which we do
not actually use for control)"; "1) generate a full visual fidelity model of this setup in mujoco 2) generate a simplified
geometry version of this model that is more or less just an update to the current model we are using with the corrected
geometry/offsets, with, crucially, the right transforms known. the ur5 mount adaptor will probably be mounted in one of
four possible orientations (i did not design it with the little clocking nub), and i will correct that when time comes.
this can be cad v3".

Sources. The owner's Onshape export of 2026-11-04 (assets/cad_v3/step, copied from assets/cad_v2/full_cad), read solid by
solid by scripts/cad_v3_assets.py extract (logs/20261104-cad_v3/extract: one mesh, its hole axes and mass properties per
solid). The analysis of 2026-11-04 (logs/20261104-cad_v3/{parts,gantry,flange,sensors,ur5,current}, archived in
docs/experiments/20261104-cad_v3_hand/analysis) named the 171 solids, found each gantry's travel between its hard stops,
the flange axis from the adaptor's holes, and the camera and tag mounts. `frames` gathers those results in
assets/cad_v3/frames.json and checks the key numbers again from the extracted solids.

Frames (mm in frames.json, m in the MJCF):
  A  the CAD assembly "hand v3 (anthro config)": z_A up toward the robot, base plate z_A 0..3, flange face z_A 28.
  P  the palm frame of the simulator and of the driver's {P}: x_P from the thumb toward the index/middle pair, y_P from the
     middle finger toward the index, z_P up, origin on the plane of the three yaw-axis centres, midway between the
     travel-box centres. P = A turned +90 deg about z_A: x_P = -y_A + t_x, y_P = x_A + t_y, z_P = z_A + t_z.
  E  tool0 of the UR5 CB3 (the controller's flange frame; the URDF's wrist_3_link): z_E out of the flange into the hand.
     The CB3 flange's four M6 holes lie at 45 + 90 j deg from x_E and the tool connector on -y_E. The adaptor's M6 holes
     lie at -21 + 90 i deg from x_A, so the adaptor bolts on in four clockings k: x_A lies at theta_k = 24 + 90 k deg
     from x_E about +z_E, with z_A = -z_E.

The models (assets/mjcf/experimental/20261104-cad_v3):
  scene_screwdriver_medium.xml        the simplified model: cad_v2's scene (same 31 qpos, 15 actuators, names and
                                      fingers) with the finger mounts at the CAD's travel-box centres and the palm's real
                                      envelope as 11 static boxes on palm_pose (palm_plate first).
  scene_screwdriver_medium_full.xml   the same with the visual meshes of every part (group 2), the moving gantry bridges
                                      (three slide joints appended after the fingers' and tied to <f>_y by equalities),
                                      the AprilTag and the v4b pads.
  ur5_k<k>.xml, ur5_full_k<k>.xml     the hand on the UR5 CB3 for clocking k: the object (qpos 0-6), the arm's six joints in
                                      place of the palm's pose joints (7-12), the hand (13-30; the full model's bridges
                                      31-33); six arm and nine finger actuators.

    .venv/bin/python scripts/build_cad_v3_hand.py frames      # assets/cad_v3/frames.json, inventory.json (no meshes)
    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python scripts/build_cad_v3_hand.py meshes
    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python scripts/build_cad_v3_hand.py scenes
    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python scripts/build_cad_v3_hand.py check
    .venv/bin/python scripts/build_cad_v3_hand.py tcp --k 0 [--point x y z]   # UR TCP of a point of P
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from morphohand.geometry import rotations as rot  # noqa: E402

CAD3 = ROOT / "assets/cad_v3"
FRAMES = CAD3 / "frames.json"
INVENTORY = CAD3 / "inventory.json"
OUT = ROOT / "assets/mjcf/experimental/20261104-cad_v3"
MESH = OUT / "meshes"
V2 = ROOT / "assets/mjcf/experimental/20261030-cad_v2"
V2_SCENE = V2 / "scene_screwdriver_medium.xml"
WORK = ROOT / "logs/20261104-cad_v3"
EXTRACT = WORK / "extract"
PARTS_JSON = EXTRACT / "parts.json"
UR5_DRAFT = WORK / "ur5"                       # the analysis' UR5 CB3 MJCF; its meshes are copied to OUT/meshes/ur5
D435_SRC = Path.home() / "Programs/mujoco_menagerie/realsense_d435i"
TAG_PNG = WORK / "sensors/tag36h11_id6_quiet2.png"
FINGERS = ("thumb", "index", "middle")
K_ALL = (0, 1, 2, 3)
R_PA = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])     # x_P = -y_A, y_P = x_A, z_P = z_A
# the moving parts of each gantry in the hand file (logs/20261104-cad_v3/parts/inventory.json, rigid_body)
GANTRY = {"thumb": "gantry_v3_thumb <1>", "index": "gantry_v3 <1>", "middle": "gantry_v3 <2>"}
FINGER_INSTANCE = {"thumb": "Finger v3 <3>", "index": "Finger v3 <2>", "middle": "Finger v3 <1>"}
# visual materials by component kind (the STEP files carry no materials; colours are chosen for the renders)
MATERIAL = {
    "base_plate": "v3_plate", "robot_mount": "v3_mount", "ur5_adaptor": "v3_adaptor",
    "gantry_frame": "v3_print", "gantry_frame_thumb": "v3_print", "x_stage_beam": "v3_print",
    "x_stage_beam_thumb": "v3_print", "x_stepper_mount": "v3_print", "x_stepper_mount_thumb": "v3_print",
    "sensor_arm": "v3_sensor", "sensor_bracket": "v3_sensor",
    "nema8_stepper": "v3_stepper", "mgn7_rail_100": "v3_steel", "mgn7_rail_150": "v3_steel",
    "mgn7_carriage": "v3_steel", "shaft_coupler": "v3_alu", "lead_nut": "v3_brass",
}
RGBA = {"v3_plate": "0.78 0.79 0.80 1", "v3_mount": "0.30 0.32 0.36 1", "v3_adaptor": "0.62 0.64 0.68 1",
        "v3_print": "0.20 0.22 0.25 1", "v3_sensor": "0.86 0.86 0.84 1", "v3_stepper": "0.07 0.07 0.08 1",
        "v3_steel": "0.70 0.72 0.75 1", "v3_alu": "0.82 0.83 0.85 1", "v3_brass": "0.80 0.62 0.28 1",
        "v3_box": "0.45 0.55 0.70 0.55", "v3_tag": "1 1 1 1"}
SCREW_R_MM = 2.5                               # 5 mm lead screws (T5 x 1): drawn as plain cylinders
PAD_TIP = "tcd_ell7.4x8.5x11x22"               # the v4b pad (CLAUDE.md: the tip calibrated on the UR5)


# ------------------------------------------------------------------------------------------------- small helpers
def jload(p):
    return json.loads(Path(p).read_text())


def fmt(v, nd=6):
    return " ".join(f"{float(x):.{nd}g}" for x in np.ravel(v))


def quat_str(R):
    return fmt(rot.quat_canonical(rot.mat_to_quat(np.asarray(R, float))), 9)


def tf(R, t):
    return rot.tf(np.asarray(R, float), np.asarray(t, float))


def frames():
    return jload(FRAMES)


def T_P_from_A(F):
    return tf(R_PA, F["P"]["t_P_from_A_mm"])


def T_E_from_A(F, k):
    return np.array(F["flange"]["T_E_from_A_by_k"][k], float).reshape(4, 4)


def T_E_from_P(F, k):
    return T_E_from_A(F, k) @ rot.tf_inv(T_P_from_A(F))


# ------------------------------------------------------------------------------------------------- frames
def derive_flange(parts):
    """The flange axis, face and the four clockings from the adaptor's M6 holes (the 6.4 mm holes along z_A)."""
    ad = next(p for p in parts if p["file"] == "hand" and abs(p["volume"] - 64184.6) < 5)
    holes = [c for c in ad["cylinders"] if c["concave"] and abs(c["r"] - 3.2) < 0.01 and abs(abs(c["axis"][2]) - 1) < 1e-6]
    xy = np.array([c["point"][:2] for c in holes])
    c = xy.mean(0)
    ang = np.degrees(np.arctan2(*(xy - c).T[::-1]))
    rad = np.linalg.norm(xy - c, axis=1)
    a = float(np.mod(ang, 90.0).mean())
    theta0 = (45.0 + a + 45.0) % 90.0 - 45.0          # x_A in E for the clocking nearest x_E, in (-45, 45]
    face = float(ad["bbox"][5])
    cA = np.array([c[0], c[1], face])
    Ts = []
    for k in K_ALL:
        R = rot.rot_z(math.radians(theta0 + 90.0 * k)) @ rot.rot_x(math.pi)
        Ts.append(tf(R, -R @ cA))
    return dict(centre_A_mm=cA.round(4).tolist(), m6_angles_A_deg=sorted(np.round(ang, 3).tolist()),
                m6_circle_d_mm=float(2 * rad.mean()), m6_angles_E_deg=[45.0, 135.0, 225.0, 315.0],
                connector_E_deg=-90.0, theta_k_deg=[round(theta0 + 90.0 * k, 6) for k in K_ALL],
                T_E_from_A_by_k=[T.round(9).ravel().tolist() for T in Ts])


def cmd_frames(args):
    parts = jload(PARTS_JSON)["parts"]
    G = jload(WORK / "gantry/gantry.json")
    Fl = jload(WORK / "flange/flange.json")
    S = jload(WORK / "sensors/sensors.json")
    inv = jload(WORK / "parts/inventory.json")
    flange = derive_flange(parts)
    # the analysis' transforms agree with the derivation here
    agent = np.array(Fl.get("T_E_from_A_by_k") or Fl["data"]["T_E_from_A_by_k"], float).reshape(4, 4, 4)
    dev = float(np.abs(agent - np.array(flange["T_E_from_A_by_k"]).reshape(4, 4, 4)).max())
    if dev > 1e-3:
        raise SystemExit(f"flange: this derivation and logs/20261104-cad_v3/flange differ by {dev}")
    shift = np.array(G.get("thumb_gantry_correction", {}).get("shift_A_mm", [0, 0, 0]), float)
    stay = set(G.get("thumb_gantry_correction", {}).get("parts_that_stay", []))
    fingers = {}
    for f in FINGERS:
        g = G["fingers"][f]
        tr = g["travel_A_mm"]
        sh = shift if f == "thumb" else np.zeros(3)
        dx = float(np.mean(tr["x_stage"]["box_disp"]))           # bridge along x_A at the box centre, from the CAD pose
        dy = float(np.mean(tr["y_stage"]["box_disp"]))           # carriage along y_A on the bridge
        yaw_A = np.array(g["yaw_centre_A_mm"], float) + sh
        box_A = yaw_A + np.array([dx, dy, 0.0])
        fingers[f] = dict(
            finger_instance=FINGER_INSTANCE[f], gantry_instance=GANTRY[f],
            yaw_centre_cad_A_mm=yaw_A.round(4).tolist(), box_centre_A_mm=box_A.round(4).tolist(),
            bridge_disp_mm=round(dx, 4), carriage_disp_mm=round(dy, 4),
            box_size_P_mm=[60.0, 110.0] if f == "thumb" else [60.0, 60.0],
            stops_disp_mm={s: [round(tr[s]["stop_minus"]["disp"], 4), round(tr[s]["stop_plus"]["disp"], 4)]
                           for s in ("x_stage", "y_stage")},
            box_disp_mm={s: [round(x, 4) for x in tr[s]["box_disp"]] for s in ("x_stage", "y_stage")},
            firmware={s: tr[s]["firmware_joint"] for s in ("x_stage", "y_stage")},
            motor_end={s: tr[s]["motor_end"] for s in ("x_stage", "y_stage")},
            stage_parts={s: sorted({n for v in tr[s]["parts"].values() for n in v}) for s in ("x_stage", "y_stage")},
            cad_joint_angles_deg=g.get("cad_joint_angles_deg", {}).get("simulator"),
            cad_pose_stepper_mm=g.get("cad_pose_stepper_mm"))
    # P: y_P = 0 midway between the index and middle box centres, x_P = 0 midway between the thumb's and the pair's,
    # z_P = 0 on the yaw-centre plane (the driver's convention; logs/20261104-cad_v3/gantry/gantry.json)
    bi, bm, bt = (np.array(fingers[f]["box_centre_A_mm"]) for f in ("index", "middle", "thumb"))
    t = np.array([0.5 * (bt[1] + 0.5 * (bi[1] + bm[1])), -0.5 * (bi[0] + bm[0]), -bi[2]])
    TPA = tf(R_PA, t)
    for f in FINGERS:
        b = rot.tf_apply(TPA, np.array(fingers[f]["box_centre_A_mm"]))
        fingers[f]["box_centre_P_mm"] = b.round(4).tolist()
        nominal = {"thumb": (-50.0, 0.0), "index": (50.0, 55.0), "middle": (50.0, -55.0)}[f]
        fingers[f]["box_minus_nominal_mm"] = (b[:2] - np.array(nominal)).round(4).tolist()
    F = dict(
        title="cad_v3 frames: the SR2 hand as built (owner's CAD of 2026-11-04) on the UR5 CB3",
        written=time.strftime("%Y-%m-%d %H:%M"), units="mm, deg",
        sources=dict(cad="assets/cad_v3/step (the owner's Onshape export of 2026-11-04, assets/cad_v2/full_cad)",
                     solids="scripts/cad_v3_assets.py extract -> logs/20261104-cad_v3/extract/parts.json",
                     analysis="docs/experiments/20261104-cad_v3_hand/analysis (gantry.json, flange.json, sensors.json)"),
        frames=dict(
            A="CAD assembly 'hand v3 (anthro config)': z_A up toward the robot, base plate z_A 0..3, flange face 28",
            P="simulator palm frame = driver {P}: x_P thumb -> pair, y_P middle -> index, z_P up, origin on the "
              "yaw-centre plane midway between the travel-box centres",
            E="tool0 of the UR5 CB3 (controller flange frame, URDF wrist_3_link): z_E out of the flange into the hand; "
              "M6 holes at 45 + 90 j deg from x_E, tool connector on -y_E"),
        P=dict(R_P_from_A=R_PA.tolist(), t_P_from_A_mm=t.round(4).tolist(),
               origin_in_A_mm=rot.tf_apply(rot.tf_inv(TPA), np.zeros(3)).round(4).tolist(),
               thumb_gantry_shift_A_mm=shift.tolist(), thumb_gantry_parts_not_shifted=sorted(stay),
               thumb_gantry_on_base_plate_holes="confirmed by the owner, 2026-11-04",
               note=G.get("t_P_from_A_note", "")),
        flange=flange,
        fingers=fingers,
        levels_P_mm={k: round(v, 3) for k, v in G["heights"]["levels_above_yaw_plane_mm"].items()},
        sensors=dict(camera_on_hand=False,
                     camera_note="Owner, 2026-11-04: the D435 is off the hand, on a tripod on the table; the d435_plate "
                                 "bracket arm stays on the hand. The optical frames below are the earlier assumed pose.",
                     tag_face="confirmed by the owner, 2026-11-04: the tag faces opposite the thumb's palmar face "
                              "(-x_P = +y_A)",
                     camera_parts=S["camera_parts"], camera_mount_assumption=S.get("camera_mount_assumption"),
                     T_A_from_depth_optical=S["T_A_from_depth_optical"],
                     T_A_from_color_optical=S["T_A_from_color_optical"], tag=S["tag"], tag_parts=S["tag_parts"]),
    )
    # tool0 -> P for each clocking, and the UR TCP of P's origin
    F["T_E_from_P_by_k"] = []
    F["tcp_P_origin_by_k"] = []
    for k in K_ALL:
        T = T_E_from_P(F, k)
        F["T_E_from_P_by_k"].append(T.round(9).ravel().tolist())
        F["tcp_P_origin_by_k"].append([round(x, 6) for x in rot.tf_to_ur(T)])   # m and rotation vector, as PolyScope
    CAD3.mkdir(parents=True, exist_ok=True)
    FRAMES.write_text(json.dumps(F, indent=1) + "\n")
    keep = ("file", "name", "label", "sub_assembly", "kind", "component", "role", "finger", "rigid_body", "material",
            "density_kg_m3", "mass_g", "mass_basis", "visual_detail_droppable", "volume_brep_mm3", "volume_mesh_mm3",
            "com_mm", "bbox_mm", "note")
    INVENTORY.write_text(json.dumps([{k: r.get(k) for k in keep} for r in inv], indent=1) + "\n")
    print("wrote", FRAMES.relative_to(ROOT), "and", INVENTORY.relative_to(ROOT))
    for f in FINGERS:
        print(f"  {f:6s} box centre P {fingers[f]['box_centre_P_mm']} (nominal offset {fingers[f]['box_minus_nominal_mm']})")
    for k in K_ALL:
        print(f"  k={k}: theta {flange['theta_k_deg'][k]:7.2f} deg; TCP of P's origin {F['tcp_P_origin_by_k'][k]}")


# ------------------------------------------------------------------------------------------------- geometry in P
def part_index():
    parts = jload(PARTS_JSON)["parts"]
    inv = {(r["file"], r["name"]): r for r in jload(INVENTORY)}
    return {p["name"]: dict(p, **{"inv": inv[("hand", p["name"])]}) for p in parts if p["file"] == "hand"}


def shift_of(F, name, finger=None):
    """The thumb gantry's correction onto the base plate's holes (frames.json P.thumb_gantry_shift_A_mm) for a solid."""
    P = F["P"]
    if finger == "thumb" and name not in P["thumb_gantry_parts_not_shifted"]:
        return np.array(P["thumb_gantry_shift_A_mm"], float)
    return np.zeros(3)


def finger_of_part(p):
    rb = p["inv"]["rigid_body"]
    for f, g in GANTRY.items():
        if rb.startswith(g) or FINGER_INSTANCE[f] in rb:
            return f
    if p["inv"]["kind"] in ("gantry_frame_thumb", "x_stepper_mount_thumb"):
        return "thumb"
    return None


def body_of_part(p):
    """palm, bridge:<f>, mount:<f>, or None (the CAD's finger links: cad_v2's fingers are used)."""
    rb = p["inv"]["rigid_body"]
    if rb == "palm":
        return "palm"
    for f, g in GANTRY.items():
        if rb == f"{g} x stage":
            return f"bridge:{f}"
        if rb.startswith(f"{g} y stage"):
            return f"mount:{f}" if p["inv"]["kind"] in ("mgn7_carriage", "lead_nut") else None
    return None


def palm_solid_owner(F, name, parts):
    """Which finger's gantry a palm solid belongs to (for the thumb shift)."""
    for f in FINGERS:
        if name in F["fingers"][f]["stage_parts"]["x_stage"] + F["fingers"][f]["stage_parts"]["y_stage"]:
            return f
    p = parts[name]
    return finger_of_part(p)


def yaw_cad_P(F, f):
    """Finger f's yaw centre in the CAD pose, in P (mm); the parts that ride with the finger keep their offset from it."""
    return rot.tf_apply(T_P_from_A(F), np.array(F["fingers"][f]["yaw_centre_cad_A_mm"], float))


def to_P(F, v_A, name=None, finger=None, extra_A=None):
    v = np.asarray(v_A, float) + shift_of(F, name, finger)
    if extra_A is not None:
        v = v + extra_A
    return rot.tf_apply(T_P_from_A(F), v)


def bbox_P(F, names, parts, disp=None):
    """Axis-aligned box in P (mm) around solids given by name, each moved by disp[name] (mm, A) if given."""
    pts = []
    for n in names:
        b = np.array(parts[n]["bbox"], float).reshape(2, 3)
        own = palm_solid_owner(F, n, parts)
        d = np.zeros(3) if disp is None else np.asarray(disp.get(n, 0.0), float)
        corners = np.array([[b[i, 0], b[j, 1], b[l, 2]] for i in (0, 1) for j in (0, 1) for l in (0, 1)])
        pts.append(to_P(F, corners + d, n, own))
    P = np.vstack(pts)
    return P.min(0), P.max(0)


def palm_boxes(F, parts):
    """The simplified palm: static boxes in P (mm), palm_plate first. Each moving gantry bridge is represented by the box
    it sweeps over its whole travel, so the boxes hold for every finger layout."""
    def sweep(f):
        fx = F["fingers"][f]
        names = [n for n in fx["stage_parts"]["x_stage"] if body_of_part(parts[n]) == f"bridge:{f}"]
        lo, hi = fx["stops_disp_mm"]["x_stage"]
        a = bbox_P(F, names, parts, {n: np.array([lo, 0, 0]) for n in names})
        b = bbox_P(F, names, parts, {n: np.array([hi, 0, 0]) for n in names})
        return np.minimum(a[0], b[0]), np.maximum(a[1], b[1])

    frames_ = [n for n, p in parts.items() if p["inv"]["kind"] in ("gantry_frame", "gantry_frame_thumb")]
    lo, hi = bbox_P(F, frames_, parts)
    bridges = {f: sweep(f) for f in FINGERS}
    # palm_plate: the three frames' footprint from the bridges' bearing blocks (their lowest faces inside the frames,
    # F levels_P_mm beam_bearing_blocks) up to the base plate
    plate_lo = np.array([lo[0], lo[1], F["levels_P_mm"]["beam_bearing_blocks"]])
    plate_hi = np.array([hi[0], hi[1], F["levels_P_mm"]["base_plate_underside"]])
    boxes = [("palm_plate", plate_lo, plate_hi, "the gantry frames' footprint from the bridges' bearing blocks to the "
                                                "base plate (the bridges sweep it)")]
    for f in FINGERS:
        blo, bhi = bridges[f]
        # the part of the swept bridge outside palm_plate: the y-stepper end below and beyond the frames
        boxes.append((f"gantry_{f}_bridge", blo, bhi, f"{f} bridge with its y stepper, swept over the bridge's travel"))
    for f in FINGERS:
        fx = F["fingers"][f]
        drive = [n for n in fx["stage_parts"]["x_stage"] if parts[n]["inv"]["kind"] in
                 ("nema8_stepper", "shaft_coupler", "x_stepper_mount", "x_stepper_mount_thumb")
                 and body_of_part(parts[n]) == "palm"]
        lo_, hi_ = bbox_P(F, drive, parts)
        boxes.append((f"gantry_{f}_xdrive", lo_, hi_, f"{f} x-axis stepper, coupler and mount (fixed)"))
    for kind, name, what in (("base_plate", "hand_base_plate", "base plate"),
                             ("robot_mount", "hand_robot_mount", "robot mount plate with the UR5 adaptor in its pocket"),
                             ("sensor_arm", "hand_camera_arm", "D435 bracket arm (d435_plate)"),
                             ("sensor_bracket", "hand_tag_plate", "AprilTag plate (static_tag_mount)")):
        ns = [n for n, p in parts.items() if p["inv"]["kind"] == kind]
        lo_, hi_ = bbox_P(F, ns, parts)
        boxes.append((name, lo_, hi_, what))
    return boxes


def d435_box_P(F):
    """The D435's 90 x 25 x 25 mm body around its depth optical frame (assumed pose; frames.json sensors): the depth
    origin (left imager) lies 17.5 mm toward -x_opt of the body centre... per sensors.json d435_model."""
    T = np.array(F["sensors"]["T_A_from_depth_optical"], float).reshape(4, 4)
    # body centre in the optical frame: +17.5 mm along x_opt (toward the right imager), 0 along y, 8.225 mm behind the
    # glass along -z_opt... the body spans x_opt -27.5..62.5, y_opt -12.5..12.5, z_opt -20.85..4.2 (glass at z 4.2)
    lo, hi = np.array([-27.5, -12.5, -20.85]), np.array([62.5, 12.5, 4.2])
    c = np.array([[lo[0] if i else hi[0], lo[1] if j else hi[1], lo[2] if l else hi[2]]
                  for i in (0, 1) for j in (0, 1) for l in (0, 1)])
    P = rot.tf_apply(T_P_from_A(F) @ T, c)
    return P.min(0), P.max(0)


# ------------------------------------------------------------------------------------------------- meshes
def cmd_meshes(args):
    import trimesh
    F = frames()
    parts = part_index()
    MESH.mkdir(parents=True, exist_ok=True)
    groups, screws, log = {}, [], []
    for name, p in sorted(parts.items()):
        body = body_of_part(p)
        if body is None or p["inv"]["visual_detail_droppable"]:
            continue
        kind = p["inv"]["kind"]
        f = body.split(":")[1] if ":" in body else palm_solid_owner(F, name, parts)
        fx = F["fingers"].get(f) if f else None
        if body.startswith("bridge:"):
            extra = np.array([fx["bridge_disp_mm"], 0.0, 0.0])          # the bridge at the box centre
        else:
            extra = None
        if kind.startswith("lead_screw"):
            b = np.array(p["bbox"], float).reshape(2, 3)
            ax = int(np.argmax(b[1] - b[0]))
            c = b.mean(0)
            e = np.zeros(3)
            e[ax] = 0.5 * (b[1, ax] - b[0, ax])
            ends = np.array([c - e, c + e])
            screws.append(dict(body=body, name=name, ends=to_P(F, ends, name, f, extra)))
            continue
        m = trimesh.load(EXTRACT / "hand" / f"{name}.stl", process=False)
        v = to_P(F, m.vertices, name, f, extra)
        if body.startswith("mount:"):
            v = v - yaw_cad_P(F, f)                                 # into the <f>_mount frame (yaw centre, P's axes)
        key = (body, MATERIAL.get(kind, "v3_print"))
        groups.setdefault(key, []).append(trimesh.Trimesh(v / 1000.0, m.faces, process=False))
        log.append((name, kind, body))
    written = {}
    for (body, mat), ms in groups.items():
        mesh = trimesh.util.concatenate(ms)
        fn = f"v3_{body.replace(':', '_')}_{mat[3:]}.stl"
        mesh.export(MESH / fn)
        written[fn] = dict(body=body, material=mat, faces=int(len(mesh.faces)))
    # the D435 (menagerie realsense_d435i, Apache-2.0) and the tag quad
    d = MESH / "d435i"
    d.mkdir(exist_ok=True)
    for src in sorted((D435_SRC / "assets").glob("*.obj")):
        shutil.copy2(src, d / src.name)
    shutil.copy2(D435_SRC / "LICENSE", d / "LICENSE")
    write_tag_obj(MESH / "v3_tag.obj")
    if TAG_PNG.exists():
        shutil.copy2(TAG_PNG, MESH / "v3_tag36h11_id6.png")
    # the UR5 CB3 meshes of the analysis (ur_description, BSD)
    u = MESH / "ur5"
    if (UR5_DRAFT / "meshes/visual").exists():
        shutil.copytree(UR5_DRAFT / "meshes/visual", u / "visual", dirs_exist_ok=True)
    meta = dict(written=written, screws=[dict(body=s["body"], name=s["name"], ends_P_mm=s["ends"].round(4).tolist())
                                         for s in screws], parts=log)
    (OUT / "meshes.json").write_text(json.dumps(meta, indent=1) + "\n")
    tot = sum(w["faces"] for w in written.values())
    print(f"wrote {len(written)} meshes ({tot} triangles), {len(screws)} screws as cylinders -> {MESH.relative_to(ROOT)}")


def write_tag_obj(path, size_mm=45.0):
    """A unit square with texture coordinates, centred, in the xy plane facing +z; the tag plate's 45 mm face."""
    h = size_mm / 2000.0
    path.write_text("\n".join([
        "# tag36h11 id 6 on the static tag mount's face, textured (cad_v3)",
        f"v {-h} {-h} 0", f"v {h} {-h} 0", f"v {h} {h} 0", f"v {-h} {h} 0",
        "vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1", "vn 0 0 1",
        "f 1/1/1 2/2/1 3/3/1", "f 1/1/1 3/3/1 4/4/1", ""]))


# ------------------------------------------------------------------------------------------------- scenes
def el(parent, tag, **kw):
    return ET.SubElement(parent, tag, {k.rstrip("_"): str(v) for k, v in kw.items() if v is not None})


def rel_meshes(root, src_dir: Path):
    """Make the scene's mesh files relative to OUT/meshes (cad_v2's resolve against its own meshdir)."""
    for m in root.find("asset").findall("mesh"):
        f = Path(m.get("file"))
        absf = (src_dir / f).resolve() if not f.is_absolute() else f
        m.set("file", str(Path(os_rel(absf, MESH.resolve()))))


def os_rel(a: Path, b: Path):
    import os
    return os.path.relpath(a, b)


def add_defaults(root, full):
    d = root.find("default")
    v = ET.SubElement(d, "default", {"class": "v3_visual"})
    el(v, "geom", type="mesh", group="2", contype="0", conaffinity="0", mass="0")
    b = ET.SubElement(d, "default", {"class": "v3_box"})
    el(b, "geom", type="box", group="3" if full else "0", mass="0", material="v3_box" if full else "palm_mat")
    s = ET.SubElement(d, "default", {"class": "v3_stage"})
    el(s, "joint", type="slide", damping="2000", armature="0.0001")


def add_materials(root):
    a = root.find("asset")
    for k, v in RGBA.items():
        if k == "v3_tag":
            continue
        el(a, "material", name=k, rgba=v, specular="0.3", shininess="0.4")


def boxes_xml(palm, boxes, first_is_plate=True):
    """palm_plate replaced in place (it stays the first geom), the others inserted after it."""
    plate = palm.find("geom[@name='palm_plate']")
    idx = list(palm).index(plate)
    for i, (name, lo, hi, what) in enumerate(boxes):
        c, hs = (lo + hi) / 2000.0, (hi - lo) / 2000.0
        if name == "palm_plate":
            plate.attrib.clear()
            plate.set("name", "palm_plate")
            plate.set("class", "v3_box")
            plate.set("pos", fmt(c))
            plate.set("size", fmt(hs))
            continue
        g = ET.Element("geom", {"name": name, "class": "v3_box", "pos": fmt(c), "size": fmt(hs)})
        palm.insert(idx + i, g)


def set_mounts(root, F):
    palm = root.find(".//body[@name='palm_pose']")
    for f in FINGERS:
        fx = F["fingers"][f]
        b = palm.find(f"body[@name='{f}_mount']")
        c = np.array(fx["box_centre_P_mm"]) / 1000.0
        b.set("pos", fmt(c))
        sx, sy = np.array(fx["box_size_P_mm"]) / 2000.0
        for j in b.findall("joint"):
            if j.get("name") == f"{f}_x":
                j.set("range", f"{-sx:.6g} {sx:.6g}")
            if j.get("name") == f"{f}_y":
                j.set("range", f"{-sy:.6g} {sy:.6g}")
        s = palm.find(f"site[@name='workspace_{f}']")
        if s is not None:
            s.set("pos", fmt([c[0], c[1], F["levels_P_mm"]["beam_bearing_blocks"] / 1000.0 - 0.0006]))
            s.set("size", fmt([sx, sy, 0.0005]))


def add_cameras(palm, F):
    """MuJoCo cameras at the D435's optical frames (assumed pose): a MuJoCo camera looks along its -z with y up, an
    optical frame along +z with y down, so the camera frame is the optical frame turned 180 deg about x."""
    for which, key, fovy in (("depth", "T_A_from_depth_optical", 58.0), ("color", "T_A_from_color_optical", 42.0)):
        T = T_P_from_A(F) @ np.array(F["sensors"][key], float).reshape(4, 4)
        R = T[:3, :3] @ rot.rot_x(math.pi)
        el(palm, "camera", name=f"d435_{which}", pos=fmt(T[:3, 3] / 1000.0), quat=quat_str(R), fovy=f"{fovy:g}")


def add_full_visuals(root, F):
    """Visual meshes of every part (palm, bridges, y stages), the lead screws, the D435 and the tag."""
    meta = jload(OUT / "meshes.json")
    asset = root.find("asset")
    palm = root.find(".//body[@name='palm_pose']")
    for fn, w in sorted(meta["written"].items()):
        el(asset, "mesh", name=fn[:-4], file=fn)
    # bridges: one body each, after the fingers, its slide tied to <f>_y
    bridge_body = {}
    for f in FINGERS:
        fx = F["fingers"][f]
        b = el(palm, "body", name=f"{f}_bridge")
        lo, hi = fx["box_disp_mm"]["x_stage"]
        half = (hi - lo) / 2000.0
        el(b, "joint", name=f"{f}_bridge", **{"class": "v3_stage"}, axis="0 1 0", range=f"{-half:.6g} {half:.6g}")
        el(b, "inertial", pos="0 0 0.04", mass="0.2", diaginertia="2e-4 2e-4 2e-4")
        bridge_body[f] = b
    eq = root.find("equality")
    if eq is None:
        eq = ET.Element("equality")
        root.insert(list(root).index(root.find("actuator")), eq)
    for f in FINGERS:
        el(eq, "joint", name=f"eq_{f}_bridge", joint1=f"{f}_bridge", joint2=f"{f}_y", polycoef="0 1 0 0 0")
    for fn, w in sorted(meta["written"].items()):
        body = w["body"]
        if body == "palm":
            tgt = palm
        elif body.startswith("bridge:"):
            tgt = bridge_body[body.split(":")[1]]
        else:
            tgt = root.find(f".//body[@name='{body.split(':')[1]}_mount']")
        el(tgt, "geom", name=fn[:-4], **{"class": "v3_visual"}, mesh=fn[:-4], material=w["material"])
    for s in meta["screws"]:
        body = s["body"]
        e = np.array(s["ends_P_mm"], float) / 1000.0
        if body.startswith("mount:"):
            e = e - yaw_cad_P(F, body.split(":")[1]) / 1000.0
        tgt = palm if body == "palm" else bridge_body.get(body.split(":")[1], palm)
        el(tgt, "geom", name=f"v3_screw_{s['name']}", **{"class": "v3_visual"}, type="cylinder", fromto=fmt(e.ravel()),
           size=f"{SCREW_R_MM / 1000:.4g}", material="v3_steel")
    # the tag (the D435 is off the hand since 2026-11-04; add_d435 and add_cameras stay for a camera put back on it)
    add_tag(root, palm, F)


def add_d435(root, palm, F):
    """The menagerie D435i meshes placed so that the mesh's left-imager lens lies on the depth optical origin and its
    front faces along the optical axis. The menagerie mesh frame: x across the camera, y toward the back(?), found from
    the lens meshes themselves."""
    import trimesh
    d = MESH / "d435i"
    lens = {}
    for i in range(9):
        lens[i] = trimesh.load(d / f"d435i_{i}.obj", force="mesh", process=False)
    # IR lenses (material IR_Lens): meshes 0 and 3; RGB pupil: 7; casing: 8
    c0, c3, c7 = (lens[i].bounds.mean(0) for i in (0, 3, 7))
    case = lens[8].bounds
    left, right = (c0, c3) if np.linalg.norm(c0 - c7) < np.linalg.norm(c3 - c7) else (c3, c0)
    x_opt = right - left                                      # left imager -> right imager = +x_opt
    x_opt /= np.linalg.norm(x_opt)
    # +z_opt out of the front face: from the casing's centre toward the lenses, snapped to the nearest mesh axis (the body
    # is 90 x 25 x 25 mm, so the smallest extent does not name the depth axis)
    n = 0.5 * (left + right) - case.mean(0)
    n -= (n @ x_opt) * x_opt
    i = int(np.argmax(np.abs(n)))
    n = np.sign(n[i]) * np.eye(3)[i]
    y_opt = np.cross(n, x_opt)
    R_mesh_opt = np.column_stack([x_opt, y_opt, n])           # optical axes in mesh coordinates
    T_mesh_opt = tf(R_mesh_opt, left)
    T_P_opt = T_P_from_A(F) @ np.array(F["sensors"]["T_A_from_depth_optical"], float).reshape(4, 4)
    T_P_opt[:3, 3] /= 1000.0
    T_P_mesh = T_P_opt @ rot.tf_inv(T_mesh_opt)
    asset = root.find("asset")
    mats = {0: "IR_Lens", 1: "IR_Emitter_Lens", 2: "IR_Rim", 3: "IR_Lens", 4: "Cameras_Gray", 5: "Black_Acrylic",
            6: "Black_Acrylic", 7: "RGB_Pupil", 8: "Metal_Casing"}
    rgba = {"Black_Acrylic": "0.07 0.07 0.07 1", "Cameras_Gray": "0.30 0.30 0.30 1", "IR_Emitter_Lens": "0.29 0.67 0.33 1",
            "IR_Lens": "0.036 0.036 0.036 1", "IR_Rim": "0.80 0.81 0.80 1", "Metal_Casing": "0.85 0.85 0.86 1",
            "RGB_Pupil": "0.09 0.003 0.009 1"}
    for m, c in rgba.items():
        el(asset, "material", name=f"d435_{m}", rgba=c)
    b = el(palm, "body", name="d435", pos=fmt(T_P_mesh[:3, 3]), quat=quat_str(T_P_mesh[:3, :3]))
    for i in range(9):
        el(asset, "mesh", name=f"d435i_{i}", file=f"d435i/d435i_{i}.obj")
        el(b, "geom", name=f"d435_vis{i}", **{"class": "v3_visual"}, mesh=f"d435i_{i}", material=f"d435_{mats[i]}")


def add_tag(root, palm, F):
    """The tag quad on the plate's face (+y_A face, nominal), its texture the tag image."""
    tg = F["sensors"]["tag"]
    T = np.array(tg["pose_A"], float).reshape(4, 4)             # apriltag frame: x right, y down, z into the tag
    T_P = T_P_from_A(F) @ T
    # the quad faces +z of its mesh; the tag's visible side faces -z_tag (out of the plate): turn 180 deg about x
    R = T_P[:3, :3] @ rot.rot_x(math.pi)
    p = T_P[:3, 3] / 1000.0 + R[:, 2] * 0.0002                 # 0.2 mm proud of the plate face
    asset = root.find("asset")
    if (MESH / "v3_tag36h11_id6.png").exists():
        el(asset, "texture", name="v3_tag", type="2d", file="v3_tag36h11_id6.png")
        el(asset, "material", name="v3_tag", texture="v3_tag", rgba="1 1 1 1")
    else:
        el(asset, "material", name="v3_tag", rgba="1 1 1 1")
    el(asset, "mesh", name="v3_tag", file="v3_tag.obj", inertia="shell")   # a flat quad has no volume
    el(palm, "geom", name="v3_tag", **{"class": "v3_visual"}, mesh="v3_tag", material="v3_tag", pos=fmt(p), quat=quat_str(R))


def dropin_root(F, full):
    root = ET.parse(V2_SCENE).getroot()
    root.set("model", "cad_v3_scene_screwdriver_medium" + ("_full" if full else ""))
    rel_meshes(root, V2 / "meshes")
    root.find("compiler").set("meshdir", "meshes")
    root.find("compiler").set("texturedir", "meshes")
    add_defaults(root, full)
    add_materials(root)
    parts = part_index()
    palm = root.find(".//body[@name='palm_pose']")
    boxes_xml(palm, palm_boxes(F, parts))
    # the boxes are massless; palm_pose keeps cad_v2's palm_plate mass and inertia (0.0918 kg, 170 x 180 x 3 mm at z 25 mm),
    # which the pose actuators were tuned for (logs/20261104-cad_v3/current/spec.md)
    palm.insert(0, ET.Element("inertial", {"pos": "0 0 0.025", "mass": "0.0918",
                                           "diaginertia": "0.000247922 0.000221103 0.000468945"}))
    set_mounts(root, F)
    if full:
        add_full_visuals(root, F)
        add_pads(root)
        for key in root.iter("key"):                           # the bridges' qpos follow <f>_y
            q = key.get("qpos").split()
            key.set("qpos", " ".join(q + [q[14], q[20], q[26]]))
    return root


def add_pads(root):
    """The v4b pads with replace_tips, as every cad_v2 user does at load time (sphere pads in group 4, the pad's
    envelope as the tip's visual)."""
    import reorient_backends as RB
    asset = root.find("asset")
    for m in list(asset.findall("mesh")):              # cad_v2 keeps cad_v1's unused TPU block meshes under these names
        if m.get("name") in ("tpu_pos", "tpu_neg"):
            asset.remove(m)
    root.find("compiler").set("meshdir", str(MESH.resolve()))
    root.find("compiler").set("texturedir", str(MESH.resolve()))
    RB.replace_tips(root, PAD_TIP, "pads", RB.MU_SCENE if hasattr(RB, "MU_SCENE") else 2.4, OUT, tool=RB.OBJ)
    root.find("compiler").set("meshdir", "meshes")
    root.find("compiler").set("texturedir", "meshes")
    for m in root.find("asset").findall("mesh"):
        f = Path(m.get("file"))
        if f.is_absolute():
            m.set("file", os_rel(f, MESH.resolve()))


def write(root, path):
    ET.indent(root, "  ")
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")
    print("wrote", path.relative_to(ROOT))


def arm_root(F, k, full):
    """The hand on the UR5 CB3 for clocking k: the drop-in scene's palm_pose (without its pose joints) on tool0."""
    hand = dropin_root(F, full)
    ur = ET.parse(UR5_DRAFT / "ur5_cb3.xml").getroot()
    root = ET.Element("mujoco", {"model": f"cad_v3_ur5_k{k}" + ("_full" if full else "")})
    comp = copy.deepcopy(hand.find("compiler"))
    root.append(comp)
    for tag in ("option", "visual"):
        root.append(copy.deepcopy(hand.find(tag)))
    g = root.find("visual").find("global")
    g.set("offwidth", "1920")
    g.set("offheight", "1440")
    dflt = copy.deepcopy(hand.find("default"))
    for c in ur.find("default"):
        dflt.append(copy.deepcopy(c))
    root.append(dflt)
    asset = copy.deepcopy(hand.find("asset"))
    for c in ur.find("asset"):
        c = copy.deepcopy(c)
        if c.tag == "mesh":
            c.set("file", "ur5/" + c.get("file"))
        asset.append(c)
    root.append(asset)
    wb = el(root, "worldbody")
    for c in hand.find("worldbody"):
        if c.tag == "body" and c.get("name") == "palm_pose":
            continue
        if c.tag == "geom" and c.get("name") == "floor":
            c = copy.deepcopy(c)
            c.set("pos", "0 0 0")
        wb.append(copy.deepcopy(c))
    # the lab's mat: its top stands 32.8 mm above the robot's mounting plane (lab README); the object lies on it
    el(wb, "geom", name="mat", type="box", pos="0.45 -0.25 0.0164", size="0.30 0.25 0.0164", rgba="0.25 0.27 0.30 1")
    obj = wb.find("body[@name='screwdriver_medium']")
    if obj is not None:
        obj.set("pos", "0.45 -0.25 0.0453")
    base = copy.deepcopy(ur.find("worldbody").find("body"))
    wb.append(base)
    tool0 = base.find(".//body[@name='tool0']")
    palm = copy.deepcopy(hand.find(".//body[@name='palm_pose']"))
    for j in palm.findall("joint"):
        palm.remove(j)
    T = T_E_from_P(F, k)
    palm.set("pos", fmt(T[:3, 3] / 1000.0, 9))
    palm.set("quat", quat_str(T[:3, :3]))
    tool0.append(palm)
    el(tool0, "site", name="tool0_flange", type="cylinder", size="0.0315 0.0005", rgba="0.9 0.1 0.1 0.6", group="4")
    contact = copy.deepcopy(hand.find("contact"))
    for c in ur.find("contact"):
        contact.append(copy.deepcopy(c))
    for b in ("wrist_3_link", "wrist_2_link"):
        el(contact, "exclude", body1="palm_pose", body2=b)
    root.append(contact)
    if hand.find("equality") is not None:
        root.append(copy.deepcopy(hand.find("equality")))
    act = el(root, "actuator")
    for a in ur.find("actuator"):
        act.append(copy.deepcopy(a))
    for a in hand.find("actuator"):
        if not a.get("name", "").startswith("a_palm_"):
            act.append(copy.deepcopy(a))
    kf = el(root, "keyframe")
    open_q = hand.find("keyframe").find("key[@name='open']").get("qpos").split()
    open_c = hand.find("keyframe").find("key[@name='open']").get("ctrl").split()
    obj_q = ["0.45", "-0.25", "0.0453", "0.70711", "0.70711", "0", "0"]
    for key in ur.find("keyframe"):
        q = key.get("qpos").split()
        el(kf, "key", name=key.get("name"), qpos=" ".join(obj_q + q + open_q[13:]),
           ctrl=" ".join(q + open_c[6:]))
    return root


def cmd_scenes(args):
    F = frames()
    OUT.mkdir(parents=True, exist_ok=True)
    write(dropin_root(F, False), OUT / "scene_screwdriver_medium.xml")
    write(dropin_root(F, True), OUT / "scene_screwdriver_medium_full.xml")
    for k in K_ALL:
        write(arm_root(F, k, False), OUT / f"ur5_k{k}.xml")
        write(arm_root(F, k, True), OUT / f"ur5_full_k{k}.xml")


# ------------------------------------------------------------------------------------------------- checks
def cmd_check(args):
    import mujoco
    F = frames()
    out = {}
    ref = mujoco.MjModel.from_xml_path(str(V2_SCENE))
    names = lambda m, t, n: [mujoco.mj_id2name(m, t, i) for i in range(n)]     # noqa: E731
    for fn in ["scene_screwdriver_medium.xml", "scene_screwdriver_medium_full.xml"] + \
              [f"ur5_k{k}.xml" for k in K_ALL] + [f"ur5_full_k{k}.xml" for k in K_ALL]:
        t0 = time.time()
        m = mujoco.MjModel.from_xml_path(str(OUT / fn))
        d = mujoco.MjData(m)
        row = dict(nq=m.nq, nv=m.nv, nu=m.nu, nbody=m.nbody, ngeom=m.ngeom, nmesh=m.nmesh, compile_s=round(time.time() - t0, 2))
        if fn.startswith("scene"):
            same = names(m, mujoco.mjtObj.mjOBJ_JOINT, ref.njnt) == names(ref, mujoco.mjtObj.mjOBJ_JOINT, ref.njnt)
            row["joints_as_cad_v2"] = bool(same)
            row["actuators_as_cad_v2"] = names(m, mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu) == \
                names(ref, mujoco.mjtObj.mjOBJ_ACTUATOR, ref.nu)
            row["first_palm_geom"] = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                                       int(m.body_geomadr[m.body("palm_pose").id]))
        key = 0
        mujoco.mj_resetDataKeyframe(m, d, key)
        mujoco.mj_forward(m, d)
        row["contacts_at_key0"] = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2), round(float(c.dist) * 1000, 2))
                                   for c in d.contact[:d.ncon]]
        if fn.startswith("ur5"):
            k = int(fn.split("_k")[1][0])
            Tw_tool = tf(d.site("tool0").xmat.reshape(3, 3), d.site("tool0").xpos * 1000)
            Tw_palm = tf(d.body("palm_pose").xmat.reshape(3, 3), d.body("palm_pose").xpos * 1000)
            T = rot.tf_inv(Tw_tool) @ Tw_palm
            row["T_E_from_P_error_mm"] = float(np.abs(T - T_E_from_P(F, k)).max())
            # the adaptor's M6 holes, through the model's tool0 -> P and frames.json's P -> A, land on the flange's
            TPA = T_P_from_A(F)
            c = np.array(F["flange"]["centre_A_mm"])
            holes_A = [c + 25.0 * np.array([math.cos(math.radians(a)), math.sin(math.radians(a)), 0])
                       for a in F["flange"]["m6_angles_A_deg"]]
            hE = [rot.tf_apply(T @ TPA, h) for h in holes_A]
            row["m6_in_E"] = [[round(float(np.hypot(h[0], h[1])), 4), round(float(np.degrees(np.arctan2(h[1], h[0]))), 3),
                               round(float(h[2]), 4)] for h in hE]
        out[fn] = row
        print(fn, {k_: v for k_, v in row.items() if k_ not in ("contacts_at_key0",)}, "contacts", len(row["contacts_at_key0"]))
    (OUT / "check.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", (OUT / "check.json").relative_to(ROOT))


def cmd_tcp(args):
    F = frames()
    T = T_E_from_P(F, args.k)
    p = rot.tf_apply(T, np.array(args.point, float))
    Tp = T.copy()
    Tp[:3, 3] = p
    print(f"k={args.k}: point {args.point} mm of P lies at {np.round(p, 3)} mm in tool0; UR TCP (m, rotation vector) "
          f"{np.round(rot.tf_to_ur(Tp), 6).tolist()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("frames")
    sp.add_parser("meshes")
    sp.add_parser("scenes")
    sp.add_parser("check")
    t = sp.add_parser("tcp")
    t.add_argument("--k", type=int, default=0, choices=K_ALL)
    t.add_argument("--point", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    sp.add_parser("all")
    a = ap.parse_args()
    if a.cmd in ("frames", "all"):
        cmd_frames(a)
    if a.cmd in ("meshes", "all"):
        cmd_meshes(a)
    if a.cmd in ("scenes", "all"):
        cmd_scenes(a)
    if a.cmd in ("check", "all"):
        cmd_check(a)
    if a.cmd == "tcp":
        cmd_tcp(a)


if __name__ == "__main__":
    main()
