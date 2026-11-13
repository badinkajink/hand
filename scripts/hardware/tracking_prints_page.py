#!/usr/bin/env python3
"""Builds docs/experiments/20261113-tracking_prints/20261113-tracking_prints_and_calibration_board.html: the print files
of the object seating and tracking plan (docs/notes/20261105-object_seating_and_tracking/) with renders, section
drawings, two MuJoCo viewers and links to every STL and Bambu project, and the cameras' calibrations (owner,
2026-11-13: "do I need to print the checkerboard? I already have a printed paper one ... Is there a docpage with all
this visualized and ready for me to analyze (with links out to the stl/3mfs where appropriate)").

Data: the print reports in assets/tracking/2026110{7,12}-*/ (tracking_objects_print.py, charuco_board_print.py,
charuco_board_mount_print.py, object_tracking_print.py), the stems' reach rows
docs/experiments/20261105-object_tracking_sim/rows/stem_sv25_sv55db.json (object_tracking_rollouts.py stem), and the
media of scripts/hardware/tracking_prints_media.py (figures in media/; the views' rows and the viewers' models in
logs/20261113-tracking_prints/, on the workstation).

    ~/miniconda3/bin/python scripts/hardware/tracking_prints_page.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import docs_index  # noqa: E402
import fp_review_page as P  # noqa: E402
import mjwasm_embed  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

table, img, esc = P.table, P.img, P.esc
DATE = "20261113"
D = ROOT / f"docs/experiments/{DATE}-tracking_prints"
MEDIA = D / "media"
OUT = D / f"{DATE}-tracking_prints_and_calibration_board.html"
TEMPLATE = ROOT / "scripts/hardware/tracking_prints_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
VIEW = ROOT / f"logs/{DATE}-tracking_prints/viewer"
GH = "https://github.com/badinkajink/hand/blob/main/"
PLAN = "docs/notes/20261105-object_seating_and_tracking/20261105-object_seating_and_tracking_plan.md"
SIM_PAGE = "docs/experiments/20261105-object_tracking_sim/20261105-object_tracking_sim.html"
OBJ = "assets/tracking/20261112-objects"
BRD = "assets/tracking/20261112-charuco_board"
TAGP = "assets/tracking/20261107-tag_prints"
STEM_ROWS = "docs/experiments/20261105-object_tracking_sim/rows/stem_sv25_sv55db.json"


def rel(path):
    """A link target from this page's folder to a repository file."""
    return "../../../" + path


def flink(path, name=None, github=True):
    """A file's link: local (relative), and its copy on GitHub for files in git."""
    name = name or Path(path).name
    out = f'<a href="{rel(path)}">{esc(name)}</a>'
    return out + (f' (<a href="{GH}{path}">GitHub</a>)' if github else "")


def load():
    J = {k: json.loads((ROOT / p).read_text()) for k, p in (
        ("obj", f"{OBJ}/20261112-objects.json"), ("brd", f"{BRD}/20261112-charuco_board.json"),
        ("mnt", f"{BRD}/20261112-charuco_board_mount.json"), ("tag", f"{TAGP}/20261107-tag_prints.json"),
        ("stem", STEM_ROWS))}
    J["views"] = json.loads((ROOT / f"logs/{DATE}-tracking_prints/views.json").read_text())
    return J


def hours(s):
    return f"{s / 3600:.1f} h"


# ------------------------------------------------------------------------------------------------ sections
def sec_terms():
    terms = [
        ("$P$", "the palm frame of <code>assets/cad_v3/README.md</code>: $x_P$ from the thumb toward the index and middle "
                "fingers, $y_P$ from the middle finger toward the index, $z_P$ up toward the flange."),
        ("$O$", "the object frame of the HOM scenes, at the object's centre, $y_O$ up; the cylinder's axis lies along "
                "$z_O$."),
        ("$B$", "the tag block's frame, at the block's centre, with $z_B$ along the block's axis and the stem "
                "($z_B = y_O$)."),
        ("$W$, $E$, $C$", "the UR5's base frame, its tool flange (tool0), and a camera's frame in OpenCV's convention ($x$ "
                          "right, $y$ down, $z$ along the view)."),
        ("$T_{AB}$", "the pose of frame $B$ in frame $A$, a 4 &times; 4 transform in mm."),
        ("Camera 0 and camera 260", "the plan's two D435 colour cameras, on their sides (1080 &times; 1920 px). They "
                                    "stand 0.3 m from the tag block in the task, at its height, at azimuths 0&deg; and "
                                    "260&deg; from $x_P$."),
        ("Tag block", "the hexagonal prism below each object, 30 mm across flats and 17 mm tall, with a 13 mm tag36h11 "
                      "AprilTag on each of its six side faces."),
        ("Stem, seat, key", "the 4 mm stem runs from the block's top to the object. Its upper end is a 45&deg; cone and "
                            "a flat seat 10 mm across. A key of 2.4 &times; 5 &times; 4 mm on the seat enters a slot "
                            "in the object."),
        ("Reach below the object", "how far the lowest finger point lies below the object's surface at the open and "
                                   "touch poses of a grasp (mm), from the HOM scenes of the rollouts."),
        ("Clearance to the block", "the stem's length less the reach below the object (mm); the plan's value is "
                                   "15 mm."),
        ("Corner RMS", "the RMS distance (mm) between the tag corners found by the detector on a drawing of a print "
                       "file's black cells and the corners of the tracker's map."),
        ("Shared volume, gap", "the volume two solids of a CAD assembly have in common (mm<sup>3</sup>; 0 means they "
                               "touch at most) and the least distance between two solids (mm)."),
        ("Turn", "a rotation of the hand about $z_P$ through $P$'s origin, from its orientation in the task (deg)."),
        ("View angle", "the angle between the board's outward normal and the direction from the board's centre to a "
                       "camera (deg): 0&deg; is head-on, above 90&deg; the camera faces the board's back."),
        ("Lens calibration", "a camera's focal lengths, principal point and distortion coefficients, fitted to "
                             "views of a flat board at many tilts."),
        ("Hand-eye calibration", "each camera's pose in $W$ and the board's pose in $E$, fitted to views of the board "
                                 "in about 15 arm poses."),
        ("Print time, filament, loads", "Bambu Studio's estimate for a sliced project: hours, grams of each filament, and "
                                        "the number of filament changes from the AMS."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in terms) + "</dl>"


def sec_files(J):
    O, B, T = J["obj"], J["brd"], J["tag"]
    op = O["parts"]

    def stls(name, folder=OBJ, colours=("white", "black")):
        date = Path(folder).name[:8]
        return "<br>".join(flink(f"{folder}/{date}-{name}_{c}.stl") for c in colours)

    rows = [
        ("Cylinder's tag block and stem (ids 10 to 15)", stls("cylinder_block"), f"{op['cylinder_block']['height_mm']:.1f}",
         "glued to the cylinder's flat and slot"),
        ("Sphere's tag block and stem (ids 16 to 21)", stls("sphere_block"), f"{op['sphere_block']['height_mm']:.1f}",
         "glued to the lower hemisphere's flat and slot"),
        ("Cylinder, end tags 22 and 23", stls("cylinder"), "100.0", "printed standing; grip tape on its side"),
        ("Sphere, lower hemisphere", stls("sphere_lower", colours=("white",)), "19.4", "three dowels into the upper half"),
        ("Sphere, upper hemisphere", stls("sphere_upper", colours=("white",)), "20.0", ""),
        ("Dowel, print three", stls("sphere_pin", colours=("white",)), "7.6", "2.9 mm in 3.1 mm holes, 4 mm deep"),
        ("ChArUco board 7 &times; 5", "<br>".join(flink(f"{BRD}/20261112-charuco_board_7x5_{c}.stl")
                                                  for c in ("white", "black")), "4.0",
         "two M3 self-tapping screws into the mount"),
        ("Board mount", flink(f"{BRD}/20261112-charuco_board_mount_white.stl"), "46.0",
         "camera arm's notch; one M5 bolt and nut"),
        ("Tag test plate (step 1, 2026-11-07)", stls("tag_test_plate", TAGP), "3.0", "free-standing"),
        ("Tag block of step 3 (2026-11-07, ids 10 to 15, 4.2 mm bore)", stls("tag_block_ids10-15", TAGP), "17.0",
         "turntable shaft"),
    ]
    t1 = table(["Part", "Print files", "Height as printed (mm)", "Joined to"], rows,
               "Print files. Each two-colour part is a white body and its black cells in one frame: load both files "
               "as one object with two parts and give the black part the black filament. Every file lies in its print "
               "orientation on z = 0.", num_from=2)
    proj = [
        (flink(f"{OBJ}/20261112-objects_p1s.3mf", github=False), "all nine object parts",
         hours(O["bambu"]["prediction_s"]), f"{O['bambu']['used_g']['2']:.1f}", f"{O['bambu']['used_g']['3']:.1f}",
         str(O["bambu"]["filament_loads"])),
        (flink(f"{BRD}/20261112-charuco_board_p1s.3mf", github=False), "the board",
         hours(B["bambu"]["prediction_s"]), f"{B['bambu']['used_g']['2']:.1f}", f"{B['bambu']['used_g']['3']:.1f}",
         str(B["bambu"]["filament_loads"])),
        (flink(f"{TAGP}/20261107-tag_prints_p1s.3mf", github=False), "test plate and step-3 block",
         hours(T["bambu"]["prediction_s"]), f"{T['bambu']['used_g']['2']:.1f}", f"{T['bambu']['used_g']['3']:.1f}",
         str(T["bambu"]["filament_loads"])),
    ]
    t2 = table(["Bambu Studio project", "Parts", "Print time", "White (g)", "Black (g)", "Loads"], proj,
               "Bambu Studio projects for the P1S with the AMS: white PLA on filament 2, black on filament 3, 0.20 mm "
               "layers, textured PEI plate, the slicer's default 15 % gyroid infill. The projects are on the "
               "workstation only (<code>.gitignore</code>); each print script's <code>bambu</code> command rebuilds "
               "its project. The mount prints in white alone and has no project.", num_from=2)
    hw = ("<p>The mount is fixed with one M5 &times; 16 bolt and a nut, and the board with two M3 self-tapping screws "
          "of 10 to 12 mm. The stems and the dowels are glued with cyanoacrylate or two-part epoxy. The cylinder's "
          "side is covered with about 0.8 mm of grip tape. The board's corner holes also fit M3 screws, for fixing it "
          "to another plate.</p>")
    return t1 + t2 + hw


def sec_objects(J, viewer):
    o = J["obj"]
    seat = o["seat"]
    op = o["parts"]
    f1 = img(MEDIA / f"{DATE}-objects.png",
             "The cylinder and the sphere on their tag blocks, assembled and with the parts apart. The cylinder's "
             "underside shows the flat and the slot; the sphere's hemispheres show the dowels' holes. Rendered from "
             "the print files, moved into the object frame.", jpeg=True)
    n1 = P.FIG[0]
    f2 = img(MEDIA / f"{DATE}-sections.png",
             "Sections of the print files through the stem's axis. Top: whole objects; bottom: the joint, with the seat "
             "plane dashed. Cream marks the object, blue the block and stem, grey a dowel and black the tags' cells; "
             "all of it is white PLA but the cells. The planes $x_O = 0$ and $z_O = 0$ cut the key along its 5 mm "
             "and its 2.4 mm side. Dimensions in mm.")
    n2 = P.FIG[0]
    p1 = (f"<p>Each tag block is printed in one piece with its stem, upright, so the stem lies on the block's axis "
          f"(Figure {n1}). The stem ends in a 45&deg; cone, a flat seat 10 mm across and a key. The seat is glued to a "
          f"flat cut into the object's underside, {seat['cylinder']['flat_depth_mm']:.2f} mm deep on the cylinder and "
          f"{seat['sphere']['flat_depth_mm']:.2f} mm deep on the sphere. The key enters a slot 0.1 mm wider on each "
          f"side and 0.5 mm deeper, which fixes the object's turn about the stem (Figure {n2}). In the CAD assembly the "
          f"block and the object share {o['assembly_overlap_mm3']['cylinder']:g} mm<sup>3</sup> on the cylinder and "
          f"{o['assembly_overlap_mm3']['sphere']:g} mm<sup>3</sup> on the sphere.</p>")
    p2 = ("<p>The cylinder is printed standing, so its section comes out round, at 23.4 mm; 0.8 mm of grip tape brings "
          "it to the HOM scenes' 25 mm. Its end faces bear tags 22 and 23. In a bundle adjustment of the assembled "
          "object these two tags are measured against the block's six. An error in the glue joint then enters the "
          "object's tag map and drops out of the tracked pose. The sphere is two hemispheres on three dowels and stays "
          "bare (owner, 2026-11-12). Each tag face of the print files decodes as its own id, and its "
          f"corners lie {max(c['corner_rms_mm'] for p in op.values() for c in p.get('check', [])):.3f} mm RMS from "
          "the tracker's map (<code>object_tracking_mjcf.object_map</code>).</p>")
    mass = [("Cylinder", f"{op['cylinder']['solid_pla_g']:.1f}", f"{op['cylinder_block']['solid_pla_g']:.1f}",
             f"{op['cylinder_block']['stem_solid_g']:.2f}"),
            ("Sphere", f"{op['sphere_lower']['solid_pla_g'] + op['sphere_upper']['solid_pla_g'] + 3 * op['sphere_pin']['solid_pla_g']:.1f}",
             f"{op['sphere_block']['solid_pla_g']:.1f}", f"{op['sphere_block']['stem_solid_g']:.2f}")]
    t = table(["Object", "Body (g)", "Block and stem (g)", "Stem alone (g)"], mass,
              "Masses of the print files at 100 % infill (PLA, 1.24 g/cm<sup>3</sup>). The sphere's body is both "
              "hemispheres and the three dowels.")
    p3 = (f"<p>The HOM objects weigh 25 g with stem, block and tape. At 100 % infill the cylinder's body alone weighs "
          f"{op['cylinder']['solid_pla_g']:.1f} g, so the body's infill sets the mass. Weigh the printed block with its "
          f"stem, and on the cylinder the tape, before the bodies are printed; then choose the body's infill for the "
          f"remainder (plan, step 4).</p>")
    v = (viewer + '<p class="vcap">Viewer 1. The two objects in MuJoCo. Drag to turn, right-drag to pan, click and '
         "scroll to zoom. The sliders move the blocks down, the sphere's upper half and the dowels up.</p>")
    return p1 + f1 + f2 + p2 + v + t + p3


def sec_stems(J):
    rows = J["stem"]["rows"]
    stem = J["stem"]["stem_mm"]
    out = []
    for shape, tasks in (("cylinder", ("pinpush", "press")), ("sphere", ("roll", "twist"))):
        for task in tasks:
            for set_name, pick in (("plan's five layouts", lambda r: r["xy"] is not None),
                                   ("measured servo, top 5", lambda r: r["xy"] is None)):
                rr = [r for r in rows if r["task"] == task and pick(r)]
                if not rr:
                    continue
                reach = [max(r["below_bottom_open_mm"], r["below_bottom_touch_mm"]) for r in rr]
                k = max(range(len(rr)), key=lambda i: reach[i])
                where = (f"({rr[k]['xy'][0]:.0f}, {rr[k]['xy'][1]:.0f})" if rr[k]["xy"] is not None
                         else esc(rr[k]["role"].split(",")[0].replace("grid2dff_v4b_sv55db ", "grid ")))
                out.append((shape.capitalize(), task.replace("pinpush", "pin-push").capitalize(), set_name,
                            str(len(rr)), f"{reach[k]:.2f}", where, f"{stem[shape]:.1f}", f"{stem[shape] - reach[k]:.1f}"))
    t = table(["Object", "Task", "Grasps", "Rows", "Largest reach (mm)", "At layout (mm)", "Stem (mm)",
               "Clearance (mm)"], out,
              "Reach of the fingers below the object at the open and touch poses, and the clearance left to the "
              "block's top. The plan's five layouts are the task-conflict study's S3 picks; the measured servo's are "
              "the five best layouts per task by tracking score on the SCS0009's identified plant (grids of "
              "2026-11-10, no pin-push grid).", num_from=3)
    p = ("<p>The stems put the block's top 15 mm below the lowest finger point of both sets of grasps (owner, "
         "2026-11-12: stems for both layout sets). The cylinder's stem is set by the press: at the roll specialist's "
         "layout (85, 170) its middle finger reaches 20.6 mm below the cylinder. The measured servo's best layouts "
         "reach 18.5 mm. On the sphere they reach 10.7 mm against 10.2 mm, so its stem grew from the plan's 25.2 mm "
         "to 25.7 mm. The table covers the static grasp poses. The rollouts of 2026-11-05 (<a href=\""
         f"{rel(SIM_PAGE)}\">simulation page</a>) replay the tasks: on the cylinder the stem cleared every finger by "
         "1.6 mm or more. The sphere rolls in the HOM tripod, and its stem touched a finger at three of the five "
         "layouts. The sphere therefore runs at the generalist (70, 110) and the pin-push specialist (75, 55) "
         "only.</p>")
    return p + t


def sec_board(J, viewer):
    B, M = J["brd"], J["mnt"]
    f3 = img(MEDIA / f"{DATE}-board.png", "The ChArUco board as printed, drawn from the print files' sections: "
             "the plate with its M3 holes and the black cells. The screws into the mount go through the pair of holes "
             "at the middle of the top edge, 12 mm apart. With the pair at the bottom edge the board hangs either way "
             "up.")
    p1 = (f"<p>The board is OpenCV's ChArUco board of 7 &times; 5 squares of 21 mm (Figure {P.FIG[0]}). Its 14 mm "
          f"markers are ids 0 to 16 of the 5 &times; 5 dictionary. With an 8 mm white margin the plate measures 163 "
          f"&times; 121 &times; 4 mm. Both square counts are odd, so OpenCV 4.6 on corvus and 4.10 on the workstation "
          f"draw the same pattern. The black print file was drawn at 10 px/mm and passed to OpenCV's ChArUco "
          f"detector. All {B['check']['corners_found']} corners were found, at most "
          f"{max(B['check']['corner_err_mm']):.4f} mm from the corners found on OpenCV's own drawing of the board.</p>")
    m = M["mount"]["nearest"][0]
    by = M["board facing -y_P"]["nearest"][0]
    bx = M["board facing -x_P"]["nearest"][0]
    f4 = img(MEDIA / f"{DATE}-mount.png", "The mount in the camera arm's notch (blue), and the board on the mount's "
             "&minus;$x_P$ and &minus;$y_P$ faces, on the hand's CAD assembly seen from the &minus;$x_P$, &minus;$y_P$ "
             "side. The flange's M5 hole lies over the arm's.", jpeg=True)
    p2 = (f"<p>The mount is a 20 &times; 20 mm post in the 20.4 mm notch at the end of the hand's camera arm (Figure "
          f"{P.FIG[0]}). Its 6 mm flange rests on the arm's top and is bolted through the arm's M5 hole. The post "
          f"reaches 30 mm below the arm. Its &minus;$x_P$ face and its &minus;$y_P$ face each have a pair of 2.75 mm pilot holes for M3 "
          f"self-tapping screws, 12 mm apart like the board's top pair. The board hangs from one face at a time. "
          f"Against the 121 solids of the hand's CAD assembly the mount shares "
          f"{M['overlap_with_arm_mm3']:g} mm<sup>3</sup> with the arm and none with any other solid. With the fingers "
          f"in the CAD's modelled pose, the mount is {m['gap_mm']:.1f} mm from the nearest other solid. The board facing "
          f"&minus;$y_P$ is {by['gap_mm']:.1f} mm from its nearest solid, and the board facing &minus;$x_P$ "
          f"{bx['gap_mm']:.1f} mm.</p>")
    v = (viewer + '<p class="vcap">Viewer 2. The hand (a display copy of the CAD assembly), the camera arm in blue, the '
         "mount and the board facing &minus;$y_P$. Geom group 3 shows the board on the &minus;$x_P$ face instead; the "
         "two cannot be mounted together.</p>")
    return p1 + f3 + p2 + f4 + v


def sec_views(J):
    V = J["views"]
    rows = V["rows"]
    turns = V["turns_deg"]
    get = {(r["camera_deg"], r["face"], r["turn_deg"]): r for r in rows}
    body = []
    for az in (260.0, 0.0):
        for face in ("-y", "-x"):
            cells = []
            for t in turns:
                a = get[(az, face, t)]["view_deg"]
                cells.append(f"<b>{a:.0f}</b>" if a <= 60 else f"{a:.0f}")
            body.append((f"Camera {az:.0f}", "&minus;$" + face[1] + "_P$", *cells))
    shift = [r["shift_mm"] for r in rows if r["face"] == "-y"]
    dist = sorted({r["distance_m"] for r in rows})
    f5 = img(MEDIA / f"{DATE}-views.png",
             "The board on the &minus;$y_P$ face in the views of camera 260 (top) and camera 0 (bottom). In each panel "
             "the hand is turned about $z_P$ by the labelled angle and moved sideways, which puts the board on the "
             "camera's axis. The cameras' poses and lenses are those of the 2026-11-05 simulation (portrait, 69&deg; "
             "high).", jpeg=True)
    t = table(["Camera", "Board face"] + [f"{t:+.0f}&deg;" for t in turns], body,
              f"View angle (deg) of the board for each turn of the hand, the board's centre on the camera's axis "
              f"{dist[0]:.2f} to {dist[-1]:.2f} m away. Bold: 60&deg; or less. The renders show which of these views the "
              f"fingers block.", num_from=2)
    p1 = ("<p>Camera 0 stands on the $+x_P$ side of the block and camera 260 on the &minus;$y_P$ side. In the task "
          f"orientation the board on the &minus;$y_P$ face is {get[(260.0, '-y', 0.0)]['view_deg']:.0f}&deg; off its "
          f"normal from camera 260 and {get[(0.0, '-y', 0.0)]['view_deg']:.0f}&deg; from camera 0, edge-on. The board on the &minus;$x_P$ face faces away from both cameras. For the hand-eye poses the arm "
          f"moves the board in front of each camera and turns the hand about $z_P$ (Figure {P.FIG[0]}, Table "
          f"{P.TAB[0]}). In the renders and the table the hand stays at the task's height and moves sideways by "
          f"{min(shift):.0f} to {max(shift):.0f} mm, which puts the board's centre on the camera's axis.</p>")
    a = {k: get[k]["view_deg"] for k in get}
    p2 = ("<p>The &minus;$y_P$ face works for both cameras. The whole board is in camera 260's view at turns of "
          f"0&deg; and +45&deg; ({a[(260.0, '-y', 0.0)]:.0f}&deg; and {a[(260.0, '-y', 45.0)]:.0f}&deg; off its "
          "normal). At &minus;45&deg; a finger covers part of it. The whole board is in camera 0's view at +90&deg; "
          f"and +135&deg; ({a[(0.0, '-y', 90.0)]:.0f}&deg; and {a[(0.0, '-y', 135.0)]:.0f}&deg;). At +45&deg; the "
          "fingers cover part of it. $z_P$ points against the flange's axis, so a turn of +90&deg; about $z_P$ is &minus;90&deg; on "
          "joint 6 from the task's pose. The lab keeps joint 6 at or below +45&deg; without the owner, and every path "
          "of the arm is checked with the cell model (<code>scripts/ur5/ur5_cb3_cell.py</code>). For the &minus;$x_P$ "
          "face the turns would be +45&deg; to +90&deg; for camera 260 and +135&deg; to +180&deg; for camera 0.</p>")
    return p1 + f5 + t + p2


def sec_calibration():
    eq1 = "$$T_{CB}^{(i)} = T_{CW}\\,T_{WE}^{(i)}\\,T_{EB}, \\qquad i = 1, \\dots, N$$"
    eq2 = "$$T_{CP} = T_{CW}\\,T_{WE}\\,T_{EP}, \\qquad T_{PO} = T_{CP}^{-1}\\,T_{CO}$$"
    p1 = ("<p>Each colour camera is calibrated twice for the tracker. In the lens calibration the focal lengths, the "
          "principal point and the distortion coefficients are fitted. On 2026-11-09 the D435s' depth maps put the tag block's "
          "faces 3.6 to 6.9 mm nearer than the tag solve, and the colour cameras' factory model has no distortion. "
          "The two causes are told apart by solving that block again with a lens model that includes distortion "
          "(plan, step 3; Future work). Each camera's pose in the arm's base comes from the hand-eye calibration. In each "
          "of $N$ arm poses the camera measures the board's pose $T_{CB}^{(i)}$, and the joint readings give the "
          "flange's pose $T_{WE}^{(i)}$. Two poses are unknown: the base in the camera, $T_{CW}$, and the board on the "
          "flange, $T_{EB}$. OpenCV's <code>calibrateRobotWorldHandEye</code> solves for both from</p>" + eq1 +
          "<p>In every frame of a trial the palm's pose in the camera then follows from the joint readings, with "
          "$T_{EP}$ from <code>sr2hand.robot.tool_chain.SR2Hand</code>. The object's pose in the palm follows from "
          "the tag block's pose $T_{CO}$:</p>" + eq2)
    p2 = ("<p>With the board fixed to the table and both cameras fixed, each camera has one view. If the board is in "
          "both views, the cameras' poses relative to the board, and so to each other, follow from them. They give "
          "neither the lens "
          "distortion, which needs the board at many tilts across the image, nor $T_{CW}$, which needs the board "
          "moved by the arm. The lens calibration is done without the arm. The board is held by hand in front of each "
          "camera for 20 or more views, which reach the image's corners and tilt the board by up to about 45&deg;. "
          "For the hand-eye calibration the board is fixed to the hand, and the arm turns it through about 15 poses "
          "about at least two axes.</p>")
    rows = [
        ("Flatness", "a 4 mm PLA plate", "glued over its whole back to glass, aluminium or acrylic; paper clamped at "
         "its edges bows"),
        ("Square size", "21 mm in CAD; measure across all seven squares with calipers", "set by the printer; measure "
         "across all squares along both edges"),
        ("Views with part of the board hidden", "used: each corner is named by its markers", "lost: the detector "
         "needs every inner corner"),
        ("Order of the corners", "fixed by the markers", "check that the first corner stays at the same corner of the "
         "board; a view numbered from the other end enters the fit as a 180&deg; turn"),
        ("Fixing to the hand", "two M3 screws into the mount", "a clamp on the mount's post or another static part, "
         "facing &minus;$y_P$"),
    ]
    t = table(["Requirement", "Printed ChArUco board", "Paper checkerboard"], rows,
              "The printed board and a paper checkerboard in the two calibrations.", num_from=9)
    p3 = ("<p>A paper checkerboard can replace the printed board if it is flat and measured, and for the hand-eye poses "
          "fixed to the hand. For the lens alone the board's overall scale does not enter. A uniform error in the "
          "square size changes the fitted distance of each view and leaves the focal lengths and the distortion as "
          "they are. A print stretched more along one edge than the other biases the focal lengths. In the hand-eye "
          "fit the board's translations are compared with the arm's, which are metric, so the square size enters "
          "there. The printed board adds partial views: near the fingers and at the image's edges a plain "
          "checkerboard view is lost whenever one inner corner is hidden.</p>")
    return p1 + p2 + p3 + t


def sec_earlier():
    return ("<p>Two parts were printed on 2026-11-07. The tag test plate of step 1 has tag36h11 and tag25h9 tags of "
            "10 to 16 mm. The tag block of step 3 has a 4.2 mm bore for the turntable's shaft. The lab measured that block on 2026-11-09 "
            "with two D435s at 0.28 to 0.29 m. Per camera and frame its pose varied by 0.14 to 0.16 mm and 0.11&deg; "
            "RMS. The owner measured 29.95 to 30.02 mm across its three pairs of flats with calipers. With the map "
            "<code>HexBlock(across_flats=29.985, tag_size=13.33)</code> the bench corners reproject at 0.33 and 0.38 px, "
            "against 0.77 and 0.75 px with the CAD map; the frames are in "
            "<code>logs/20261109-tag_block_bench/cams/</code>. The tracker drops tags seen more than 65&deg; from their "
            "normal (<code>sr2hand.tracking.pose.MAX_VIEW_DEG</code>), and <code>pose.bundle_adjust</code> fits each "
            "tag's size and takes the scale from measured lengths (plan, step 3).</p>")


def sec_limits():
    return ("<p>No part on this page has been printed or measured. The checks are on the print files: the decode of "
            "each tag face on a drawing, the shared volumes and gaps in CAD, and the slicer's estimates. The hand in "
            "the renders and in Viewer 2 has its fingers in the CAD's modelled pose, and the views omit the arm, the "
            "cables and the camera mounts. The view angles and the renders do not check the arm's reach or its "
            "joint limits for the turned poses. The stems' clearance comes from the HOM simulation of 2026-11-05 and "
            "from static grasp poses on the measured servo's grids.</p>")


def sec_future():
    items = [
        "Print the objects, weigh each block with its stem and the cylinder's tape, and set the bodies' infill so "
        "that each object weighs 25 g (plan, step 4).",
        "Calibrate each colour camera's lens with distortion from 20 or more views of a flat board, and solve the "
        "tag block of 2026-11-09 again with the fitted model. If its depth moves by less than 1 mm, the depth maps' "
        "3.6 to 6.9 mm offset comes from the depth stream.",
        "Plan about 15 hand-eye poses per camera in <code>scripts/ur5/ur5_cb3_cell.py</code>, with the board on the "
        "&minus;$y_P$ face. The turns about $z_P$ are 0&deg; to +45&deg; for camera 260 and +90&deg; to +135&deg; "
        "for camera 0. Each pose adds a tilt of up to &plusmn;20&deg; about a second axis. If joint 6 cannot reach "
        "&minus;90&deg; from the task's pose, a board facing $+x_P$ on another part of the hand is needed for camera 0.",
        "Design the flat seat on its pedestal, the catch tray and the turntable adapter with its 20&deg; wedge "
        "(plan, Parts). The seat's height is set by the arm's pose of the hand in the task. The adapter needs the "
        "STS3032 horn's dimensions.",
    ]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def sec_scripts():
    rows = [("scripts/hardware/tracking_objects_print.py", "the objects' print files, their check and Bambu project"),
            ("scripts/hardware/charuco_board_print.py", "the board's print files, check and Bambu project"),
            ("scripts/hardware/charuco_board_mount_print.py", "the mount and its check against the hand's CAD"),
            ("scripts/hardware/object_tracking_print.py", "the tag test plate and step-3 block of 2026-11-07, and the "
             "shared two-colour and Bambu steps"),
            ("scripts/hardware/object_tracking_mjcf.py", "the objects, blocks and tag maps used by the tracker and the "
             "simulation"),
            ("scripts/hardware/tracking_prints_media.py", "this page's renders, drawings and viewer models"),
            ("scripts/hardware/tracking_prints_page.py", "this page"),
            (PLAN, "the plan")]
    return "<ul>" + "".join(f'<li><a href="{GH}{a}"><code>{a}</code></a>: {b}</li>' for a, b in rows) + "</ul>"


# ------------------------------------------------------------------------------------------------ page
def lede_and_summary(J):
    stem = J["stem"]["stem_mm"]
    lede = (f"The two HOM objects, their tag blocks and a 7 &times; 5 ChArUco board with its mount on the hand are "
            f"ready to print. Each block prints in one piece with its stem, {stem['cylinder']:.1f} mm on the cylinder "
            f"and {stem['sphere']:.1f} mm on the sphere. Its top then lies 15 mm below the lowest finger point of both "
            f"layout sets. All 14 tags decode from the print files, 0.022 mm RMS from the tracker's map. The board "
            f"hangs on the mount's &minus;$y_P$ face. It is in camera 260's view with the hand turned 0 to +45&deg; "
            f"about $z_P$, and in camera 0's at +90 to +135&deg;. A paper checkerboard can replace the printed board if it is "
            f"flat and measured, and fixed to the hand for the hand-eye poses.")
    summary = (f"Print files of the HOM cylinder and sphere on tag-block stems ({stem['cylinder']:.1f} and "
               f"{stem['sphere']:.1f} mm, 15 mm below the lowest finger point of both layout sets; 14 tags decode at "
               f"0.022 mm RMS) and of a 7 x 5 ChArUco board with its mount on the SR2 hand's camera arm. The board on "
               f"the mount's -y face is seen by camera 260 with the hand turned 0 to +45 deg about z_P and by camera 0 "
               f"at +90 to +135 deg; a flat, measured paper checkerboard fixed to the hand can replace it.")
    return lede, summary


def render_tex(page):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = texsvg.render(items, cache_path=TEX_CACHE)
    it = iter(svgs)
    return P.TEX_RE.sub(lambda m: next(it), page), len(items)


def main():
    J = load()
    v_obj = mjwasm_embed.embed(VIEW / "objects.xml", element_id="viewer_objects", height_px=520,
                               title="Cylinder and sphere on their tag blocks",
                               camera={"azimuth": 120.0, "elevation": -12.0, "distance": 0.36, "lookat": [0.0, 0.0, 0.07]})
    v_hand = mjwasm_embed.embed(VIEW / "hand.xml", element_id="viewer_hand", height_px=560, include_runtime=False,
                                title="Board and mount on the SR2 hand",
                                camera={"azimuth": 50.0, "elevation": -15.0, "distance": 0.72,
                                        "lookat": [-0.08, 0.03, -0.01]})
    # the viewers carry no math: keep them out of the TeX pass by placeholders
    holders = {"VIEWER_OBJECTS": v_obj, "VIEWER_HAND": v_hand}
    sections = [
        ("terms", "Terms and metrics", sec_terms()),
        ("files", "Print files and hardware", sec_files(J)),
        ("objects", "Objects on their tag blocks", sec_objects(J, "{{VIEWER_OBJECTS}}")),
        ("stems", "Stem length and finger clearance", sec_stems(J)),
        ("board", "ChArUco board and its mount on the hand", sec_board(J, "{{VIEWER_HAND}}")),
        ("views", "The board seen by the two cameras", sec_views(J)),
        ("calibration", "Lens and hand-eye calibration, and the paper checkerboard", sec_calibration()),
        ("earlier", "Tag test plate and step-3 block of 2026-11-07", sec_earlier()),
        ("limits", "Limitations", sec_limits()),
        ("future", "Future work", sec_future()),
        ("scripts", "Scripts", sec_scripts()),
    ]
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|kg|px|N\b|ms\b|s\b|g\b|m\b|h\b)", "\\1&nbsp;\\2", s_))
    body = [(sid, hd, unit(b)) for sid, hd, b in sections]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(body, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{b}</section>' for k, (sid, hd, b) in enumerate(body, 1))
    lede, summary = lede_and_summary(J)
    byline = ("Print files of 2026-11-07 and 2026-11-12, checked in CAD and in the slicer; nothing printed or measured "
              "yet. Plan: <a href=\"" + rel(PLAN) + "\"><code>" + PLAN + "</code></a>. Builder: "
              "<code>scripts/hardware/tracking_prints_page.py</code>; figures and viewers: "
              "<code>scripts/hardware/tracking_prints_media.py</code>.")
    page = TEMPLATE.read_text()
    for k_, v in {"TITLE": "Tracking prints and board",
                  "H1": "Tracked objects, tag blocks and calibration board for the SR2 hand on the UR5",
                  "EYEBROW": "Hand and bench, 2026-11-13", "LEDE": unit(lede), "BYLINE": byline, "TOC": toc,
                  "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k_ + "}}", v)
    page, n = render_tex(page)
    for k_, v in holders.items():
        page = page.replace("{{" + k_ + "}}", v)
    page = docs_index.tag(page, topic="hardware", summary=summary, date="2026-11-13")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {n} formulas", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
