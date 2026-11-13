#!/usr/bin/env python3
"""Figures and viewer models of the page on the printed tracking objects and the calibration board
(docs/experiments/20261113-tracking_prints/, built by scripts/hardware/tracking_prints_page.py).

Parts come from the print files (assets/tracking/20261112-objects/ and 20261112-charuco_board/), moved from their print
frames into the object frame O or the palm frame P by the transforms of tracking_objects_print.py and
charuco_board_mount_print.py. The hand is the cad_v3 extract (logs/20261104-cad_v3/extract/hand: 121 solids in the CAD
frame A, fingers in the CAD's modelled pose). The cameras are the plan's: portrait D435s 0.3 m from the block in the
task, at its height, at azimuths 0 and 260 deg from x_P, aimed as in object_tracking_sim.py (`visibility`). The palm's
height in the task is the generalist's pin-push rollout of 2026-11-05 plus the 40 mm lift.

    drawings  section drawings of the joints and a drawing of the board (femtip: manifold3d, matplotlib)
    renders   MuJoCo renders of the objects, the mount and the cameras' views, and logs/20261113-tracking_prints/
              views.json (camera stack)
    viewers   the MJCF models of the page's two viewers in logs/20261113-tracking_prints/viewer (camera stack)

    ~/.claude/bin/resguard.sh run --mem 2G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/envs/femtip/bin/python scripts/hardware/tracking_prints_media.py drawings
    ~/.claude/bin/resguard.sh run --mem 4G -- env NUMPY_MADVISE_HUGEPAGE=0 MUJOCO_GL=egl \\
        ~/miniconda3/bin/python scripts/hardware/tracking_prints_media.py renders
    ~/.claude/bin/resguard.sh run --mem 3G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/bin/python scripts/hardware/tracking_prints_media.py viewers
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for _paths
import _paths  # noqa: E402,F401
import charuco_board_mount_print as MP  # noqa: E402
import charuco_board_print as CB  # noqa: E402
import object_tracking_mjcf as OM  # noqa: E402
import tracking_objects_print as TO  # noqa: E402
from sr2hand.geometry.rotations import mat_to_quat, quat_to_mat, rot_z  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATE = "20261113"
D = ROOT / f"docs/experiments/{DATE}-tracking_prints"
MEDIA = D / "media"
LOGS = ROOT / f"logs/{DATE}-tracking_prints"
VIEW = LOGS / "viewer"
OBJ = TO.OUT
BRD = CB.OUT
EXTRACT = MP.EXTRACT
TRAJ = ROOT / "logs/20261105-object_tracking_sim/traj/flat_pinpush_162_s23.npz"
LIFT_M = 0.040
CAM_AZ = (260.0, 0.0)
TABLE_TURNS = (-90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0)   # deg about z_P through P's origin
VIEW_CASES = ((260.0, (-45.0, 0.0, 45.0, 90.0)), (0.0, (0.0, 45.0, 90.0, 135.0)))
BOARD_DIST = 0.40                              # m from the camera to the board's centre, along the axis's heading
FONT = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
# display frame of the objects: x_W = x_O, y_W = -z_O, z_W = y_O (y_O up, the cylinder's axis along y_W)
R_WO = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
WHITE, BLACK, HAND, ARM = "0.93 0.93 0.91 1", "0.06 0.06 0.06 1", "0.60 0.62 0.66 1", "0.38 0.52 0.72 1"


# ------------------------------------------------------------------------------------------------ part frames
def stl(name, colour="white"):
    return OBJ / f"{TO.DATE}-{name}_{colour}.stl"


def block_T(shape):
    """(R, t) of the block-and-stem print file in O (mm)."""
    o = OM.OBJECTS[shape]
    c = OM.block_centre_O(o["r_mm"], o["stem_mm"])
    return OM.R_OB, c - OM.R_OB @ np.array([0.0, 0.0, OM.BLOCK.height / 2])


def object_parts(shape):
    """[(label, stl, R, t, colour, group)] of an object's print files in O (mm); group names the moving part."""
    out = []
    Rb, tb = block_T(shape)
    for c in ("white", "black"):
        out.append((f"{shape} block", stl(f"{shape}_block", c), Rb, tb, c, "block"))
    if shape == "cylinder":
        L = OM.OBJECTS["cylinder"]["length_mm"]
        for c in ("white", "black"):
            out.append(("cylinder", stl("cylinder", c), TO.R_PRINT["cylinder"].T, np.array([0.0, 0.0, -L / 2]), c, "body"))
        return out
    out.append(("lower hemisphere", stl("sphere_lower"), TO.R_PRINT["sphere_lower"].T, np.zeros(3), "white", "body"))
    out.append(("upper hemisphere", stl("sphere_upper"), TO.R_PRINT["sphere_upper"].T, np.zeros(3), "white", "upper"))
    Rp = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])            # the pin's z along +y_O
    for k, a in enumerate(TO.PIN_ANGLES):
        t = np.array([TO.PIN_R * math.cos(math.radians(a)), -TO.PIN_DEPTH + 0.2, TO.PIN_R * math.sin(math.radians(a))])
        out.append((f"dowel {k + 1}", stl("sphere_pin"), Rp, t, "white", "pins"))
    return out


def board_T(face):
    """(R, t) of the board's print files (face up, z 0 to 4) in P (mm), hung from the mount's face '-x' or '-y'."""
    g = MP.geometry()
    c = (g["notch_lo"] + g["notch_hi"]) / 2
    z0 = g["top"] - MP.ARM_THICK - MP.BELOW
    top = z0 + MP.Z_HOLES[face] + CB.HOLE_IN
    PH = CB.board_wh()[1] + 2 * CB.MARGIN
    if face == "-x":                       # x_b -> -y_P, y_b -> +z_P, z_b (the face's normal) -> -x_P
        R = np.array([[0, 0, -1.0], [-1.0, 0, 0], [0, 1.0, 0]])
        t = np.array([c[0] - MP.POST / 2, c[1], top - PH / 2])
    else:                                  # x_b -> +x_P, y_b -> +z_P, z_b -> -y_P
        R = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        t = np.array([c[0], c[1] - MP.POST / 2, top - PH / 2])
    return R, t


def mount_T():
    """(R, t) of the mount's print file (flange down) in P (mm): the inverse of charuco_board_mount_print's T."""
    g = MP.geometry()
    return np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, g["top"] + MP.FLANGE_T])


def board_parts(face):
    R, t = board_T(face)
    return [(f"board facing {face}_P", BRD / f"{CB.DATE}-charuco_board_7x5_{c}.stl", R, t, c, f"board{face}")
            for c in ("white", "black")]


def hand_parts():
    """[(label, stl, R, t)] of the hand's solids in P (mm)."""
    R, t = MP.P_from_A()
    parts = [p for p in json.loads((EXTRACT / "parts.json").read_text())["parts"] if p["file"] == "hand"]
    return [(p["label"], ROOT / p["stl"], R, t, p["idx"]) for p in parts]


def board_face_P(face):
    """Centre and outward normal of the board's printed face in P (mm)."""
    R, t = board_T(face)
    return R @ np.array([0.0, 0.0, CB.THICK]) + t, R[:, 2]


# ------------------------------------------------------------------------------------------------ MJCF
def mjcf(geoms, cameras=(), floor=None, extra_assets="", bodies=""):
    """A static MJCF of mesh geoms [(name, stl, R_world, t_world_m, rgba, group)] and fixed cameras
    [(name, pos, xyaxes, fovy)]; meshes in mm, scaled to m."""
    meshes, seen, rows = [], {}, []
    for name, path, R, t, rgba, grp in geoms:
        key = str(Path(path).resolve())
        if key not in seen:
            seen[key] = f"m{len(seen)}"
            meshes.append(f'<mesh name="{seen[key]}" file="{key}" scale="0.001 0.001 0.001" inertia="shell"/>')
        q = mat_to_quat(np.asarray(R, float))
        rows.append(f'<geom name="{name}" type="mesh" mesh="{seen[key]}" pos="{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}" '
                    f'quat="{q[0]:.7f} {q[1]:.7f} {q[2]:.7f} {q[3]:.7f}" rgba="{rgba}" group="{grp}" '
                    f'contype="0" conaffinity="0" mass="0"/>')
    cams = [f'<camera name="{n}" pos="{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" xyaxes="{" ".join(f"{v:.6f}" for v in xy)}" '
            f'fovy="{fy:.3f}"/>' for n, p, xy, fy in cameras]
    fl = (f'<geom name="table" type="plane" size="1 1 0.01" pos="0 0 {floor:.4f}" rgba="0.86 0.86 0.86 1" '
          f'contype="0" conaffinity="0"/>' if floor is not None else "")
    return f"""<mujoco model="tracking_prints">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual>
    <global offwidth="2400" offheight="2400" fovy="30"/>
    <quality shadowsize="8192" offsamples="8"/>
    <headlight ambient="0.30 0.30 0.30" diffuse="0.32 0.32 0.32" specular="0.04 0.04 0.04"/>
    <rgba haze="0.86 0.88 0.91 1"/>
    <map znear="0.002" zfar="20"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.93 0.94 0.96" rgb2="0.80 0.82 0.86" width="64" height="64"/>
    {chr(10).join(meshes)}
    {extra_assets}
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="0 0 2" dir="-0.35 0.45 -1" diffuse="0.48 0.48 0.48" specular="0.08 0.08 0.08" castshadow="true"/>
    <light name="fill" directional="true" pos="0 0 2" dir="0.5 -0.3 -0.6" diffuse="0.18 0.18 0.18" specular="0 0 0" castshadow="false"/>
    {fl}
    {chr(10).join(rows)}
    {chr(10).join(cams)}
    {bodies}
  </worldbody>
</mujoco>
"""


def rgba(c):
    return WHITE if c == "white" else BLACK


def placed(parts, R_W, t_W_mm, explode=None):
    """Mesh geoms of parts (in a part frame F, mm) placed in the world by (R_W, t_W_mm); explode moves each group by a
    vector in F (mm)."""
    out = []
    for k, (label, path, R, t, c, grp) in enumerate(parts):
        dv = np.zeros(3) if explode is None else np.asarray(explode.get(grp, (0.0, 0.0, 0.0)), float)
        out.append((f"{label} {c} {k}".replace(" ", "_"), path, R_W @ R, (R_W @ (t + dv) + t_W_mm) / 1000.0, rgba(c), 0))
    return out


# ------------------------------------------------------------------------------------------------ images
def font(px):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT, px)
    except OSError:
        return ImageFont.load_default()


def label(img, text, px=30):
    """Lines of text at the top left on a white band."""
    from PIL import ImageDraw
    lines = [text] if isinstance(text, str) else list(text)
    d = ImageDraw.Draw(img, "RGBA")
    f = font(px)
    w = max(d.textlength(t, font=f) for t in lines)
    d.rectangle([6, 6, 22 + w, 14 + len(lines) * (px + 6)], fill=(255, 255, 255, 215))
    for k, t in enumerate(lines):
        d.text((14, 10 + k * (px + 6)), t, fill=(0, 0, 0), font=f)
    return img


def tile(frames, cols, gap=8):
    from PIL import Image
    w, h = frames[0].size
    rows = math.ceil(len(frames) / cols)
    out = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), (255, 255, 255))
    for k, f in enumerate(frames):
        out.paste(f, ((k % cols) * (w + gap), (k // cols) * (h + gap)))
    return out


def outline(img):
    from PIL import ImageDraw
    ImageDraw.Draw(img).rectangle([0, 0, img.size[0] - 1, img.size[1] - 1], outline=(0, 0, 0), width=1)
    return img


def render(m, d, cam, w, h, ss=2):
    """One frame of camera cam (a name, or a free camera dict) at w x h, rendered at ss times and downsampled."""
    import mujoco
    from PIL import Image
    r = mujoco.Renderer(m, height=h * ss, width=w * ss)
    try:
        if isinstance(cam, str):
            r.update_scene(d, camera=cam)
        else:
            c = mujoco.MjvCamera()
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            c.lookat[:] = cam["lookat"]
            c.distance, c.azimuth, c.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
            r.update_scene(d, camera=c)
        img = Image.fromarray(r.render())
    finally:
        r.close()
    return img.resize((w, h), Image.LANCZOS)


def compile_xml(xml):
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def save(img, name):
    MEDIA.mkdir(parents=True, exist_ok=True)
    p = MEDIA / name
    img.save(p, optimize=True)
    print("wrote", p.relative_to(ROOT), img.size, flush=True)


# ------------------------------------------------------------------------------------------------ renders
def fig_objects():
    """Both objects on their blocks, assembled and taken apart, seen from below the tags' height."""
    explode = {"cylinder": {"block": (0.0, -22.0, 0.0)},
               "sphere": {"block": (0.0, -22.0, 0.0), "upper": (0.0, 22.0, 0.0), "pins": (0.0, 9.0, 0.0)}}
    frames = []
    for shape in ("cylinder", "sphere"):
        parts = object_parts(shape)
        for ex in (False, True):
            m, d = compile_xml(mjcf(placed(parts, R_WO, np.zeros(3), explode[shape] if ex else None)))
            look = (0.0, 0.0, -0.030 if not ex else -0.036)
            cam = {"lookat": look, "distance": (0.30 if shape == "cylinder" else 0.25) + (0.03 if ex else 0.0),
                   "azimuth": 118.0, "elevation": 14.0 if ex else -12.0}
            img = render(m, d, cam, 700, 760)
            what = ("Cylinder" if shape == "cylinder" else "Sphere") + (", parts apart" if ex else ", assembled")
            frames.append(outline(label(img, what)))
    save(tile(frames, 2), f"{DATE}-objects.png")


def task_world():
    """(palm position in the task, m) of the generalist's pin-push rollout, the arm's lift added."""
    z = np.load(TRAJ, allow_pickle=True)
    return np.asarray(z["palm"], float) + np.array([0.0, 0.0, LIFT_M])


def cameras(palm):
    """{az: T_WC (OpenCV, m)} and the portrait pinhole, aimed as in object_tracking_sim.cmd_visibility."""
    import object_tracking_sim as S
    seated = np.array([0.0, 0.0, 0.065])
    target = seated + np.array([0.0, 0.0, LIFT_M])
    out = {}
    for az in CAM_AZ:
        a = math.radians(az)
        hand0 = palm - np.array([0.0, 0.0, LIFT_M]) + np.array([S.HAND_TAG_R * math.cos(a), S.HAND_TAG_R * math.sin(a),
                                                                 S.HAND_TAG_Z]) / 1000.0
        out[az] = S.camera_pose(az, 0.0, target, 0.5 * (seated + hand0 + np.array([0.0, 0.0, LIFT_M])))
    return out, S.portrait()


def mj_camera(name, T_WC, cam):
    """A fixed MJCF camera at the OpenCV pose T_WC: MuJoCo's x is OpenCV's x, its y is OpenCV's -y."""
    fovy = math.degrees(2 * math.atan(cam.height / 2 / cam.K[1, 1]))
    return (name, T_WC[:3, 3], np.r_[T_WC[:3, 0], -T_WC[:3, 1]], fovy)


def hand_geoms(R_W, t_W_mm, skip=()):
    out = []
    for label_, path, R, t, idx in hand_parts():
        if idx in skip:
            continue
        out.append((f"hand_{idx}", path, R_W @ R, (R_W @ t + t_W_mm) / 1000.0, ARM if idx == 0 else HAND, 0))
    return out


def mount_geoms(R_W, t_W_mm):
    R, t = mount_T()
    return [("mount", TO.OUT.parent / "20261112-charuco_board" / f"{CB.DATE}-charuco_board_mount_white.stl",
             R_W @ R, (R_W @ t + t_W_mm) / 1000.0, WHITE, 0)]


def fig_mount():
    """The mount in the camera arm's notch and the board on each face, the hand in its CAD pose."""
    I = np.eye(3)
    g = MP.geometry()
    c = np.r_[(g["notch_lo"] + g["notch_hi"]) / 2, g["top"]] / 1000.0
    views = [("Mount in the camera arm's notch", None, {"lookat": c + np.array([0.0, 0.0, -0.025]), "distance": 0.22,
                                                         "azimuth": 40.0, "elevation": -25.0}),
             ("Board facing −x", "-x", {"lookat": (-0.08, 0.03, -0.01), "distance": 0.66, "azimuth": 50.0,
                                         "elevation": -15.0}),
             ("Board facing −y", "-y", {"lookat": (-0.08, 0.03, -0.01), "distance": 0.66, "azimuth": 50.0,
                                         "elevation": -15.0})]
    frames = []
    for title, face, cam in views:
        geoms = hand_geoms(I, np.zeros(3)) + mount_geoms(I, np.zeros(3))
        if face:
            geoms += placed(board_parts(face), I, np.zeros(3))
        m, d = compile_xml(mjcf(geoms))
        frames.append(outline(label(render(m, d, cam, 620, 620), title)))
    save(tile(frames, 3), f"{DATE}-mount.png")


def board_pose(palm, T_WC, face, turn):
    """The hand turned by turn deg about z_P and moved sideways (x, y) from the task so that the board's face centre
    lies in the camera's vertical plane through its optical axis, BOARD_DIST from the camera; the palm's height stays
    the task's. Returns (palm position, Rz, face centre, outward normal) in the world (m)."""
    cen, n = board_face_P(face)
    Rz = rot_z(math.radians(turn))
    c0 = Rz @ cen / 1000.0 + palm
    a = T_WC[:3, 2].copy()
    a[2] = 0.0
    a /= np.linalg.norm(a)
    goal = T_WC[:3, 3] + BOARD_DIST * a
    shift = np.array([goal[0] - c0[0], goal[1] - c0[1], 0.0])
    return palm + shift, Rz, c0 + shift, Rz @ n


def view_angles(palm, T, cam):
    """{(az, face, turn): (deg between the face's outward normal and the direction to the camera, share of the face's
    corners inside the image, distance to the camera (m), sideways move of the hand from the task (mm))}, the board
    placed by board_pose (its own geometry; the hand's occlusion is in the renders)."""
    out = {}
    for face in ("-y", "-x"):
        R, t = board_T(face)
        hw, hh = CB.board_wh()[0] / 2 + CB.MARGIN, CB.board_wh()[1] / 2 + CB.MARGIN
        corners = np.array([[sx * hw, sy * hh, CB.THICK] for sx in (-1, 1) for sy in (-1, 1)]) @ R.T + t
        for turn in TABLE_TURNS:
            for az, T_WC in T.items():
                pw, Rz, cw, nw = board_pose(palm, T_WC, face, turn)
                eye = T_WC[:3, 3]
                v = eye - cw
                ang = math.degrees(math.acos(np.clip(nw @ v / np.linalg.norm(v), -1, 1)))
                pc = (corners @ Rz.T / 1000.0 + pw - eye) @ T_WC[:3, :3]
                inside = [(p[2] > 0) and 0 <= cam.K[0, 0] * p[0] / p[2] + cam.K[0, 2] <= cam.width
                          and 0 <= cam.K[1, 1] * p[1] / p[2] + cam.K[1, 2] <= cam.height for p in pc]
                out[(az, face, turn)] = (round(ang, 1), float(np.mean(inside)), round(float(np.linalg.norm(v)), 3),
                                         round(float(np.linalg.norm(pw - palm)) * 1000.0, 1))
    return out


def fig_views():
    """What the two cameras see of the board facing -y_P, the hand at the task's height, turned about z_P and moved
    sideways to put the board on the camera's axis."""
    palm = task_world()
    T, cam = cameras(palm)
    VA = view_angles(palm, T, cam)
    frames = []
    for az, turns in VIEW_CASES:
        for turn in turns:
            pw, Rz, _c, _n = board_pose(palm, T[az], "-y", turn)
            tW = pw * 1000.0
            geoms = hand_geoms(Rz, tW) + mount_geoms(Rz, tW) + placed(board_parts("-y"), Rz, tW)
            m, d = compile_xml(mjcf(geoms, [mj_camera(f"cam{int(az)}", T[az], cam)]))
            img = render(m, d, f"cam{int(az)}", 405, 720)
            ang = VA[(az, "-y", turn)][0]
            img = label(img, [f"Camera {az:.0f}°, hand turned {turn:+.0f}°", f"board {ang:.0f}° off its normal"], 25)
            frames.append(outline(img))
    save(tile(frames, 4), f"{DATE}-views.png")
    rep = {"palm_task_m": np.round(palm, 4).tolist(), "board_dist_m": BOARD_DIST, "turns_deg": list(TABLE_TURNS),
           "cameras_deg": list(CAM_AZ),
           "camera_pos_m": {str(int(az)): np.round(T[az][:3, 3], 4).tolist() for az in CAM_AZ},
           "rows": [{"camera_deg": az, "face": face, "turn_deg": turn, "view_deg": a, "corners_in_image": ins,
                     "distance_m": dist, "shift_mm": sh}
                    for (az, face, turn), (a, ins, dist, sh) in sorted(VA.items())]}
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "views.json").write_text(json.dumps(rep, indent=1))
    for r in rep["rows"]:
        print(r)


def cmd_renders(a):
    for k in a.only or ("objects", "mount", "views"):
        {"objects": fig_objects, "mount": fig_mount, "views": fig_views}[k]()
    return 0


# ------------------------------------------------------------------------------------------------ drawings
def to_manifold(path, R, t):
    import manifold3d as M
    import trimesh
    tm = trimesh.load(path, force="mesh")
    tm.merge_vertices()
    m = M.Manifold(M.Mesh(vert_properties=np.asarray(tm.vertices, np.float32), tri_verts=np.asarray(tm.faces, np.uint32)))
    if m.status() != M.Error.NoError:
        raise SystemExit(f"{path}: {m.status()}")
    return m.transform(np.hstack([np.asarray(R, float), np.asarray(t, float)[:, None]]))


# the section planes in O: (rotation that takes the plane's normal to -z' and its axes to x' and y', name)
CUTS = {"x0": (np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]), "z_O", "y_O"),
        "z0": (np.eye(3), "x_O", "y_O")}
COL = {"body": "#ece6d6", "block": "#c9d6e6", "upper": "#ece6d6", "pins": "#a9a9a9", "black": "#111111"}


def sections(shape, cut):
    """[(group, colour key, [rings])] of an object's parts cut by a plane through the stem's axis (mm)."""
    R, _a, _b = CUTS[cut]
    out = []
    for label_, path, Rp, tp, c, grp in object_parts(shape):
        m = to_manifold(path, R @ Rp, R @ tp)
        rings = [np.asarray(p) for p in m.slice(0.0).to_polygons()]
        if rings:
            out.append((grp, "black" if c == "black" else grp, rings))
    return out


def draw_rings(ax, rings, colour):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    verts, codes = [], []
    for r in rings:
        verts += r.tolist() + [r[0].tolist()]
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(r) - 1) + [MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=colour, edgecolor="#000000", lw=0.7))


def dim(ax, p0, p1, text, off=(0, 0), ha="center", va="bottom", fs=14):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", lw=0.9, color="#000000", shrinkA=0, shrinkB=0))
    ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], text, ha=ha, va=va, fontsize=fs,
            bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))


def fig_sections():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 14, "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "mathtext.fontset": "cm"})
    panels = [("cylinder", "x0", "Cylinder, cut at $x_O = 0$"), ("cylinder", "z0", "Cylinder, cut at $z_O = 0$"),
              ("sphere", "x0", "Sphere, cut at $x_O = 0$")]
    fig, axs = plt.subplots(2, 3, figsize=(10.8, 10.8), gridspec_kw={"height_ratios": [1.55, 1.0]})
    rep = {}
    for j, (shape, cut, title) in enumerate(panels):
        S = sections(shape, cut)
        r_print, depth, seat_y = TO.seat(shape)
        for ax, zoom in ((axs[0, j], False), (axs[1, j], True)):
            for grp, ck, rings in S:
                draw_rings(ax, rings, COL[ck])
            ax.set_aspect("equal")
            o = OM.OBJECTS[shape]
            if zoom:
                ax.set_xlim(-9, 11)
                ax.set_ylim(seat_y - 7.5, seat_y + 7.5)
            else:
                ax.set_xlim(-24, 24)
                ax.set_ylim(-o["r_mm"] - o["stem_mm"] - OM.BLOCK.height - 3, o["r_mm"] + 2)
            ax.set_xlabel(f"${CUTS[cut][1][0]}_{{O}}$ (mm)")
            ax.set_ylabel("$y_O$ (mm)")
            ax.grid(alpha=0.2, lw=0.5)
        axs[0, j].set_title(title, fontsize=15)
        a = axs[1, j]
        # the joint: seat, cone, key and slot (the key's 5 mm along z_O, its 2.4 mm along x_O)
        a.axhline(seat_y, color="#666666", lw=0.6, ls="--")
        key_w = TO.KEY[1] if cut == "x0" else TO.KEY[0]
        slot_w = key_w + 2 * TO.KEY_CLEAR
        dim(a, (-TO.SEAT_D / 2, seat_y - 4.2), (TO.SEAT_D / 2, seat_y - 4.2), f"seat {TO.SEAT_D:.0f}", off=(0, -0.5),
            va="top")
        dim(a, (-key_w / 2, seat_y + TO.KEY[2] + 0.9), (key_w / 2, seat_y + TO.KEY[2] + 0.9),
            f"key {key_w:g} in {slot_w:g}", off=(0, 0.3))
        dim(a, (6.4, seat_y), (6.4, seat_y + TO.KEY[2]), f"key {TO.KEY[2]:g}", off=(0.3, 0), ha="left", va="center")
        dim(a, (6.4, seat_y - TO.CONE_H), (6.4, seat_y), f"cone {TO.CONE_H:g}", off=(0.3, 0), ha="left",
            va="center")
        rep[f"{shape}_{cut}"] = {"seat_y_O_mm": round(seat_y, 3), "flat_depth_mm": round(depth, 3),
                                 "key_mm": key_w, "slot_mm": slot_w}
    fig.tight_layout()
    MEDIA.mkdir(parents=True, exist_ok=True)
    p = MEDIA / f"{DATE}-sections.png"
    fig.savefig(p, dpi=100, facecolor="white")
    plt.close(fig)
    print("wrote", p.relative_to(ROOT), rep, flush=True)


def fig_board():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 15})
    I, z = np.eye(3), np.zeros(3)
    white = to_manifold(BRD / f"{CB.DATE}-charuco_board_7x5_white.stl", I, z)
    black = to_manifold(BRD / f"{CB.DATE}-charuco_board_7x5_black.stl", I, z)
    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    plate = [np.asarray(p) for p in white.slice(CB.THICK - 1.5).to_polygons()]
    draw_rings(ax, plate, "#ffffff")
    draw_rings(ax, [np.asarray(p) for p in black.slice(CB.THICK - 0.3).to_polygons()], "#000000")
    W, H = CB.board_wh()[0] / 2 + CB.MARGIN, CB.board_wh()[1] / 2 + CB.MARGIN
    dim(ax, (-W, H + 7), (W, H + 7), f"{2 * W:.0f} mm", off=(0, 0.8))
    dim(ax, (W + 7, -H), (W + 7, H), f"{2 * H:.0f} mm", off=(1.2, 0), ha="left", va="center")
    sq = CB.SQUARE
    x0, y0 = -CB.board_wh()[0] / 2, -CB.board_wh()[1] / 2
    dim(ax, (x0, y0 - 5), (x0 + sq, y0 - 5), f"square {sq:g}", off=(0, -1.2), va="top", fs=15)
    dim(ax, (-CB.MOUNT_HOLE_DX / 2, H + 1.5), (CB.MOUNT_HOLE_DX / 2, H + 1.5), "", fs=15)
    ax.text(CB.MOUNT_HOLE_DX / 2 + 3, H - CB.HOLE_IN, f"M3 pair, {CB.MOUNT_HOLE_DX:g} apart", ha="left", va="center",
            fontsize=15, bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xlim(-W - 6, W + 28)
    ax.set_ylim(-H - 14, H + 14)
    ax.set_aspect("equal")
    ax.axis("off")
    p = MEDIA / f"{DATE}-board.png"
    fig.savefig(p, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p.relative_to(ROOT), flush=True)


def cmd_drawings(a):
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig_sections()
    fig_board()
    return 0


# ------------------------------------------------------------------------------------------------ viewers
def cluster(tm, g):
    """The mesh with its vertices merged on a grid of g mm (each cell's mean) and degenerate or repeated faces dropped:
    a display copy of a CAD solid."""
    import trimesh
    q = np.floor(tm.vertices / g).astype(np.int64)
    _u, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = inv.max() + 1
    v = np.zeros((n, 3))
    np.add.at(v, inv, tm.vertices)
    v /= np.bincount(inv, minlength=n)[:, None]
    f = inv[tm.faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    _k, idx = np.unique(np.sort(f, 1), axis=0, return_index=True)
    return trimesh.Trimesh(v, f[np.sort(idx)], process=False)


def cmd_viewers(a):
    import trimesh
    VIEW.mkdir(parents=True, exist_ok=True)
    # objects: the two assemblies side by side, each moving part on a slide joint
    bodies = []
    geoms_assets = {}
    for shape, x in (("cylinder", -0.034), ("sphere", 0.036)):
        parts = object_parts(shape)
        groups = {}
        for p in parts:
            groups.setdefault(p[5], []).append(p)
        inner = []
        for grp, ps in groups.items():
            gs = []
            for label_, path, R, t, c, _g in ps:
                key = str(Path(path).resolve())
                geoms_assets.setdefault(key, f"m{len(geoms_assets)}")
                q = mat_to_quat(R_WO @ R)
                tw = R_WO @ t / 1000.0
                gs.append(f'<geom type="mesh" mesh="{geoms_assets[key]}" pos="{tw[0]:.6f} {tw[1]:.6f} {tw[2]:.6f}" '
                          f'quat="{q[0]:.7f} {q[1]:.7f} {q[2]:.7f} {q[3]:.7f}" rgba="{rgba(c)}" contype="0" '
                          f'conaffinity="0" mass="0"/>')
            if grp == "body":
                inner += gs
                continue
            axis, rng = {"block": ("0 0 -1", 0.035), "upper": ("0 0 1", 0.030), "pins": ("0 0 1", 0.012)}[grp]
            jn = {"block": f"{shape} block down", "upper": "upper hemisphere up", "pins": "dowels up"}[grp]
            inner.append(f'<body name="{shape}_{grp}"><joint name="{jn}" type="slide" axis="{axis}" range="0 {rng}" '
                         f'limited="true"/><inertial pos="0 0 0" mass="0.01" diaginertia="1e-6 1e-6 1e-6"/>'
                         + "".join(gs) + "</body>")
        bodies.append(f'<body name="{shape}" pos="{x} 0 0.11">' + "".join(inner) + "</body>")
    assets = "".join(f'<mesh name="{n}" file="{k}" scale="0.001 0.001 0.001" inertia="shell"/>'
                     for k, n in geoms_assets.items())
    xml = mjcf([], extra_assets=assets, bodies="\n".join(bodies), floor=0.0)
    (VIEW / "objects.xml").write_text(xml)
    print("wrote", (VIEW / "objects.xml").relative_to(ROOT), flush=True)
    # the hand with the mount and the two boards: the CAD solids merged into two display meshes
    R, t = MP.P_from_A()
    arm, rest = [], []
    for label_, path, Rh, th, idx in hand_parts():
        tm = trimesh.load(path, force="mesh")
        tm.apply_transform(np.r_[np.c_[Rh, th], [[0, 0, 0, 1.0]]])
        (arm if idx == 0 else rest).append(tm)
    sizes = {}
    for name, ms, g in (("camera_arm", arm, 0.6), ("hand", rest, 1.6)):
        tm = cluster(trimesh.util.concatenate(ms), g)
        tm.export(VIEW / f"{name}.stl")
        sizes[name] = (len(tm.faces), (VIEW / f"{name}.stl").stat().st_size)
    print("display meshes (faces, bytes)", sizes, flush=True)
    I = np.eye(3)
    geoms = [("camera arm", VIEW / "camera_arm.stl", I, np.zeros(3), ARM, 0),
             ("hand", VIEW / "hand.stl", I, np.zeros(3), HAND, 0)]
    geoms += mount_geoms(I, np.zeros(3))
    for face, grp in (("-y", 0), ("-x", 3)):
        geoms += [(n, p, Rb, tb, rgb, grp) for n, p, Rb, tb, rgb, _g in placed(board_parts(face), I, np.zeros(3))]
    (VIEW / "hand.xml").write_text(mjcf(geoms))
    print("wrote", (VIEW / "hand.xml").relative_to(ROOT), flush=True)
    for x in ("objects.xml", "hand.xml"):
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(VIEW / x))
        print(x, "compiles:", m.ngeom, "geoms,", m.njnt, "joints", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("drawings")
    r = sub.add_parser("renders")
    r.add_argument("--only", nargs="*", choices=("objects", "mount", "views"))
    sub.add_parser("viewers")
    a = ap.parse_args()
    return {"drawings": cmd_drawings, "renders": cmd_renders, "viewers": cmd_viewers}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
