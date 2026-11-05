#!/usr/bin/env python3
"""Builds docs/experiments/20261104-cad_v3_hand/20261104-cad_v3_hand_on_ur5.html: the SR2 hand as built on the UR5 CB3, its
MuJoCo models and the transform from the flange to the palm (owner, 2026-11-04: "1) generate a full visual fidelity model of
this setup in mujoco 2) generate a simplified geometry version of this model ... with, crucially, the right transforms
known. the ur5 mount adaptor will probably be mounted in one of four possible orientations ... this can be cad v3").

Data: assets/cad_v3/frames.json and inventory.json (scripts/build_cad_v3_hand.py frames), the MJCF folder's check.json and
meshes.json, the folder's 20261104-clocking_from_photo.json, analysis/tag_in_photo.json and media/20261104-film.json
(scripts/cad_v3_film.py). The design page's helpers draw the tables and figures (scripts/fp_review_page.py).

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python scripts/cad_v3_page.py figs   # the drawings
    .venv/bin/python scripts/cad_v3_page.py                                                            # the page
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_cad_v3_hand as B  # noqa: E402
import docs_index  # noqa: E402
import fp_review_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
from morphohand.geometry import rotations as rot  # noqa: E402

fill, table, img, film, esc = P.fill, P.table, P.img, P.film, P.esc
D = ROOT / "docs/experiments/20261104-cad_v3_hand"
MEDIA = D / "media"
OUT = D / "20261104-cad_v3_hand_on_ur5.html"
TEMPLATE = ROOT / "scripts/cad_v3_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
OUTLINES = MEDIA / "20261104-outlines_A.json"
K = B.AS_BUILT_K
FINGERS = B.FINGERS
COLOR = {"thumb": "#b03a2e", "index": "#1f4e9c", "middle": "#5b8c3a"}
CAD2 = "../20261030-cad_v2_finger/"


def load():
    F = B.frames()
    return dict(F=F, inv=json.loads(B.INVENTORY.read_text()), check=json.loads((B.OUT / "check.json").read_text()),
                meshes=json.loads((B.OUT / "meshes.json").read_text()),
                clock=json.loads((D / "20261104-clocking_from_photo.json").read_text()),
                tag=json.loads((D / "analysis/tag_in_photo.json").read_text()),
                film=json.loads((MEDIA / "20261104-film.json").read_text()))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def facing(F, k):
    """For clocking k: the angle from +y_E (the HEX-E's port) to each finger's box centre seen from the flange axis, in E."""
    T = B.T_E_from_A(F, k)
    out = {}
    for f in FINGERS:
        c = rot.tf_apply(T, np.array(F["fingers"][f]["box_centre_A_mm"]))
        out[f] = math.degrees(math.atan2(c[1], c[0]))
    return out


def faces_text(ang, target):
    off = {f: wrap180(a - target) for f, a in ang.items()}
    f = min(off, key=lambda x: abs(off[x]))
    if abs(off[f]) <= 30.0:
        return f"{f}, {abs(off[f]):.0f} deg off"
    near = sorted(off, key=lambda x: abs(off[x]))[:2]
    return f"between {near[0]} and {near[1]}"


# ------------------------------------------------------------------------------------------------- figures
def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
                         "legend.fontsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "mathtext.fontset": "cm"})
    return plt


def save(fig, name):
    import matplotlib.pyplot as plt
    fig.savefig(MEDIA / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", (MEDIA / name).relative_to(ROOT), flush=True)


def section_paths(path, z):
    """Closed outlines of a part's horizontal section at height z_A: lists of (x, y) in mm, from its extracted STL."""
    import trimesh
    m = trimesh.load(path, process=False)
    m.merge_vertices()
    s = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    return [np.asarray(p)[:, :2].round(3).tolist() for p in s.discrete]


def cmd_outlines():
    inv = {r["kind"]: r["name"] for r in json.loads(B.INVENTORY.read_text()) if r["file"] == "hand"}
    Ex = B.EXTRACT / "hand"
    J = dict(robot_mount=section_paths(Ex / f"{inv['robot_mount']}.stl", 27.5),
             base_plate=section_paths(Ex / f"{inv['base_plate']}.stl", 1.5),
             sensor_arm=section_paths(Ex / f"{inv['sensor_arm']}.stl", -2.0),
             tag_plate=section_paths(Ex / f"{inv['sensor_bracket']}.stl", -30.0))
    OUTLINES.write_text(json.dumps(J) + "\n")
    print("wrote", OUTLINES.relative_to(ROOT), {k: len(v) for k, v in J.items()}, flush=True)


def fig_clockings(C):
    """The hand under the flange for the four clockings, seen from the tool side with x_E right and y_E up."""
    plt = plt_setup()
    from matplotlib.patches import Circle
    F = C["F"]
    O = json.loads(OUTLINES.read_text())
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 9.8), constrained_layout=True, sharex=True, sharey=True)
    for k, ax in zip(B.K_ALL, axs.ravel()):
        T = B.T_E_from_A(F, k)

        def xy(p):
            p = np.asarray(p, float)
            return rot.tf_apply(T, np.c_[p, np.zeros(len(p))])[:, :2]
        for path in O["base_plate"]:
            q = xy(path)
            ax.fill(q[:, 0], q[:, 1], fc="#dcc7ae", ec="#7a5c3a", lw=0.8, zorder=1)
        for path in O["sensor_arm"] + O["tag_plate"]:
            q = xy(path)
            ax.fill(q[:, 0], q[:, 1], fc="#9a9a9a", ec="#333333", lw=0.6, zorder=2)
        for path in O["robot_mount"]:
            q = xy(path)
            ax.plot(q[:, 0], q[:, 1], color="black", lw=0.8, zorder=3)
        ax.add_patch(Circle((0, 0), 35.0, fc="#e6e6e6", ec="black", lw=1.0, zorder=4))
        for f in FINGERS:
            c = rot.tf_apply(T, np.array(F["fingers"][f]["box_centre_A_mm"]))
            ax.plot(c[0], c[1], "o", ms=11, mfc=COLOR[f], mec="white", mew=1.2, zorder=6)
            # the label beside the dot, on the side away from the vertical arrows
            sx = 1.0 if c[0] >= 0 else -1.0
            ax.text(c[0] + sx * 12, c[1], f, ha="left" if sx > 0 else "right", va="center", color=COLOR[f], zorder=7,
                    bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85))
        ax.annotate("", xy=(0, 150), xytext=(0, 35), arrowprops=dict(arrowstyle="-|>", color="#1a1a1a", lw=2.2), zorder=6)
        ax.text(0, 166, "HEX-E port", ha="center", va="center", zorder=7, bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))
        ax.annotate("", xy=(0, -150), xytext=(0, -35), arrowprops=dict(arrowstyle="-|>", color="#6a1b9a", lw=2.2), zorder=6)
        ax.text(0, -166, "tool connector", ha="center", va="center", color="#6a1b9a", zorder=7,
                bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))
        built = "; the hand as built" if k == K else ""
        ax.set_title(f"$k$ = {k}, $\\theta_k$ = {F['flange']['theta_k_deg'][k]:.2f} deg{built}")
        ax.set_aspect("equal")
        ax.set_xlim(-195, 195)
        ax.set_ylim(-195, 195)
        ax.grid(alpha=0.25, zorder=0)
        if k == K:
            for s in ax.spines.values():
                s.set_linewidth(2.5)
    for ax in axs[1]:
        ax.set_xlabel("$x_E$ (mm)")
    for ax in axs[:, 0]:
        ax.set_ylabel("$y_E$ (mm)")
    save(fig, "20261104-clockings_schematic.png")


def fig_travel(C):
    """The travel boxes in P against the driver's, with the plates."""
    plt = plt_setup()
    from matplotlib.patches import Rectangle
    F = C["F"]
    O = json.loads(OUTLINES.read_text())
    TPA = B.T_P_from_A(F)

    def xy(p):
        p = np.asarray(p, float)
        return rot.tf_apply(TPA, np.c_[p, np.zeros(len(p))])[:, :2]
    fig, ax = plt.subplots(figsize=(9.2, 8.8), constrained_layout=True)
    for path in O["base_plate"]:
        q = xy(path)
        ax.fill(q[:, 0], q[:, 1], fc="#eadfce", ec="#7a5c3a", lw=0.8, zorder=1)
    for path in O["robot_mount"]:
        q = xy(path)
        ax.plot(q[:, 0], q[:, 1], color="#555555", lw=0.7, zorder=2)
    for path in O["sensor_arm"] + O["tag_plate"]:
        q = xy(path)
        ax.fill(q[:, 0], q[:, 1], fc="#b5b5b5", ec="#444444", lw=0.6, zorder=2)
    nominal = {"thumb": (-50.0, 0.0), "index": (50.0, 55.0), "middle": (50.0, -55.0)}
    for f in FINGERS:
        fx = F["fingers"][f]
        c = np.array(fx["box_centre_P_mm"])[:2]
        sx, sy = fx["box_size_P_mm"]
        ax.add_patch(Rectangle((c[0] - sx / 2, c[1] - sy / 2), sx, sy, fc=COLOR[f], alpha=0.16, ec=COLOR[f], lw=2.0, zorder=4))
        n = np.array(nominal[f])
        ax.add_patch(Rectangle((n[0] - sx / 2, n[1] - sy / 2), sx, sy, fc="none", ec=COLOR[f], lw=1.3, ls="--", zorder=4))
        ax.plot(*c, "o", color=COLOR[f], ms=7, zorder=6)
        ax.plot(*n, "x", color=COLOR[f], ms=8, mew=1.8, zorder=6)
        ax.text(c[0], c[1] + 5, f"{f}\n({c[0]:.1f}, {c[1]:.1f})", ha="center", va="bottom", color=COLOR[f], zorder=7,
                bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.75))
    a = rot.tf_apply(TPA, np.array(F["flange"]["centre_A_mm"]))
    ax.plot(a[0], a[1], "+", color="black", ms=16, mew=2.2, zorder=7)
    ax.annotate("flange axis", (a[0], a[1]), xytext=(a[0] - 48, a[1] + 26), arrowprops=dict(arrowstyle="-", lw=0.8), zorder=7)
    ax.plot(0, 0, "s", color="black", ms=6, zorder=7)
    ax.annotate("origin of $P$", (0, 0), xytext=(-48, -22), arrowprops=dict(arrowstyle="-", lw=0.8), zorder=7)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_P$ (mm)")
    ax.set_ylabel("$y_P$ (mm)")
    ax.grid(alpha=0.25, zorder=0)
    ax.set_xlim(-185, 145)
    ax.set_ylim(-180, 180)
    save(fig, "20261104-travel_boxes.png")


def fig_stack(C):
    """The stack along the tool axis: a side view of the full model with its heights in z_P."""
    import mujoco
    import cad_v3_film as CF
    from PIL import Image
    plt = plt_setup()
    F = C["F"]
    w, h = 1320, 1200
    span_z, centre_z = 0.340, 0.045                     # m of z_P in view, and its centre

    def add(spec):
        cam = spec.worldbody.add_camera(name="side", pos=[0, 0, 1], fovy=span_z)
        cam.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
    m = CF.load(B.OUT / f"ur5_full_k{K}.xml", w, h, floor=False, extra=add)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("over_object").id)
    d.qpos[:3] = (0.0, 0.0, -5.0)
    for f in FINGERS:
        for j in ("yaw", "mcp", "pip"):
            d.qpos[m.jnt_qposadr[m.joint(f"{f}_{j}").id]] = 0.0
    mujoco.mj_forward(m, d)
    palm = d.body("palm_pose")
    Rw = palm.xmat.reshape(3, 3)
    c = m.cam("side").id
    # looking along +y_P with x_P to the right and z_P up: the thumb on the left, the pair on the right
    m.cam_pos[c] = palm.xpos + Rw @ np.array([0.0, -0.8, centre_z])
    m.cam_quat[c] = rot.quat_canonical(rot.mat_to_quat(Rw @ np.column_stack([[1, 0, 0], [0, 0, 1], [0, -1, 0]])))
    m.cam_mode[c], m.cam_bodyid[c] = 0, 0
    # the arm above wrist 3 is left out
    for g in range(m.ngeom):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        if bn in ("wrist_2_link", "wrist_1_link", "forearm_link", "upper_arm_link", "shoulder_link", "base_link_inertia"):
            m.geom_rgba[g, 3] = 0.0
            m.geom_matid[g] = -1
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, h, w, max_geom=12000)
    try:
        r.update_scene(d, "side", scene_option=CF.scene_option())
        im = np.array(r.render())
    finally:
        r.close()
    Image.fromarray(im).save(MEDIA / "20261104-side_render.png")
    half_x = 0.5 * span_z * w / h * 1000.0
    z0, z1 = (centre_z - span_z / 2) * 1000.0, (centre_z + span_z / 2) * 1000.0
    fig, ax = plt.subplots(figsize=(9.8, 8.0), constrained_layout=True)
    ax.imshow(im, extent=[-half_x, half_x, z0, z1], zorder=1)
    L = F["levels_P_mm"]
    rows = [(F["fingers"]["thumb"] and L["flange_face"] + 35.0, "UR5 flange face"),
            (L["flange_face"], "HEX-E tool face, adaptor face"),
            (L["robot_mount_underside"], "robot mount underside"),
            (L["base_plate_underside"], "base plate underside"),
            (L["frame_lowest"], "gantry frame, lowest point"),
            (L["beam_main_underside"], "beam underside"),
            (L["beam_bearing_blocks"], "beam bearing blocks"),
            (L["beam_lowest"], "beam motor end"),
            (0.0, "yaw plane, $z_P$ = 0"),
            (L["carriage_bottom"], "carriage bottom")]
    x_end = half_x + 8
    ys = [r_[0] for r_ in rows]
    # label heights spread so that the text does not overlap, tied to their levels by thin lines
    tgt = np.array(ys, float)
    lab = tgt.copy()
    for _ in range(200):
        for i in range(len(lab) - 1):
            gap = lab[i] - lab[i + 1]
            if gap < 15.5:
                lab[i] += (15.5 - gap) / 2
                lab[i + 1] -= (15.5 - gap) / 2
    for (y, text), yl in zip(rows, lab):
        ax.plot([-half_x, half_x + 2], [y, y], color="#c0392b", lw=0.8, zorder=3)
        ax.plot([half_x + 2, x_end + 8], [y, yl], color="#c0392b", lw=0.8, zorder=3, clip_on=False)
        ax.text(x_end + 10, yl, f"{y:.1f}  {text}", va="center", ha="left", fontsize=14, clip_on=False)
    ax.set_xlim(-half_x, half_x)
    ax.set_ylim(z0, z1)
    ax.set_xlabel("$x_P$ (mm)")
    ax.set_ylabel("$z_P$ (mm)")
    save(fig, "20261104-stack_heights.png")


def cmd_figs():
    C = load()
    if not OUTLINES.exists():
        cmd_outlines()
    fig_clockings(C)
    fig_travel(C)
    fig_stack(C)
    return 0


# ------------------------------------------------------------------------------------------------- prose
def num(v, nd=2):
    return f"{v:.{nd}f}"


def tuple_mm(v, nd=2):
    return "(" + ", ".join(f"{(x if abs(x) >= 0.5 * 10 ** -nd else 0.0):.{nd}f}" for x in v) + ")"


def sec_terms(C):
    F = C["F"]
    t = F["P"]["t_P_from_A_mm"]
    terms = [
        ("Frame $A$", "the frame of the owner's CAD assembly \"hand v3 (anthro config)\", in mm. Its $z_A$ axis points up "
         "toward the robot. The base plate spans $z_A$ 0 to 3, the robot mount 3 to 28, and the gantries hang below."),
        ("Frame $P$", "the palm frame of the simulator and of the hand driver. Its $x_P$ axis points from the thumb toward "
         f"the index and middle pair, its $y_P$ axis from the middle finger toward the index, and $z_P$ up. Its origin lies on "
         f"the plane of the three yaw-axis centres, midway between the travel-box centres. In coordinates, $x_P = -y_A + "
         f"{t[0]:.2f}$, $y_P = x_A + {t[1]:.2f}$ and $z_P = z_A + {t[2]:.2f}$ mm."),
        ("Frame $E$", "tool0 of the UR5 CB3, the flange frame that the controller reports with a zero TCP. Its $z_E$ axis "
         "points out of the flange into the hand. The robot's tool connector lies on $-y_E$."),
        ("Frame $H$", "the tool face of the OptoForce HEX-E, a force-torque sensor that stays on the flange. It stands 35.0 mm "
         "out along $z_E$, turned -2.15 deg about it."),
        ("Clocking $k$", "one of the four ways, $k$ = 0 to 3, to bolt the hand's adaptor to the HEX-E. The adaptor has no "
         "dowel, and its four M6 holes repeat every 90 deg. In clocking $k$, the axis $x_A$ lies at $\\theta_k$ = 21.85 + 90 $k$ "
         "deg from $x_E$ about $+z_E$, and $z_A = -z_E$."),
        ("HEX-E port", "the M8 cable port on the side of the HEX-E, along $+y_E$ (owner, 2026-10-22). A right-angle plug is "
         "screwed onto it."),
        ("UR TCP", "the tool centre point as PolyScope takes it: a position in m and a rotation vector in rad, both in $E$."),
        ("Stage, bridge, carriage", "each finger rides on two lead-screw stages. The x stage moves a bridge along $x_A$ "
         "($y_P$; firmware J0, J2, J4). The y stage moves the finger's carriage along $y_A$ ($x_P$; J1, J3, J5)."),
        ("Stroke, travel box", "the stroke is a stage's length between its two hard stops in the CAD. The travel box is the "
         "rectangle of yaw-axis centres that the firmware commands. It is 60 by 110 mm for the thumb and 60 by 60 mm for the others, "
         "measured from the home stop at the stage's stepper end."),
        ("Driver's positions", "the box centres that the hand driver and the simulator have used until now: thumb (-50, 0), "
         "index (50, 55) and middle (50, -55) mm in $P$."),
        ("Yaw plane", "the plane $z_P$ = 0 through the three yaw-axis centres. All heights on this page are $z_P$ in mm."),
        ("Full and simplified models", "the full model draws every part as a visual mesh, and it is for renders. The "
         "simplified model is cad_v2's scene with the corrected finger mounts and the palm's envelope as static boxes, for "
         "simulation and control."),
        ("Reprojection error", "the distance in pixels between a point picked in a photo and the same CAD point projected "
         "through the fitted camera, as an rms over the points. In the photo from above, 1 px is about 0.14 mm near the flange."),
        ("Port misfit", "the rms distance in pixels from the picks along the real plug's axis to the model's port segment "
         "for a clocking, at the height within the HEX-E that fits best."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in terms) + "</dl>"


def sec_models(C):
    ch, me = C["check"], C["meshes"]
    fi = C["film"]
    rows = []
    names = {"scene_screwdriver_medium.xml": "Simplified, hand only", "scene_screwdriver_medium_full.xml": "Full, hand only",
             f"ur5_k{K}.xml": f"Simplified on the UR5, $k$ = {K}", f"ur5_full_k{K}.xml": f"Full on the UR5, $k$ = {K}"}
    for fn, lab in names.items():
        r = ch[fn]
        rows.append([f"<code>{fn}</code>", lab, r["nq"], r["nu"], r["nbody"], r["ngeom"], r["nmesh"], f"{r['compile_s']:.2f}"])
    tb = table(["File", "Model", "nq", "nu", "Bodies", "Geoms", "Meshes", "Compile (s)"], rows,
               "The scenes of <code>assets/mjcf/experimental/20261104-cad_v3</code> as compiled by MuJoCo 3.6. nq is the length "
               "of the position vector and nu the number of actuators.", num_from=2)
    tris = sum(w["faces"] for w in me["written"].values())
    drop = ch["scene_screwdriver_medium.xml"]["contacts_at_key0"]
    obj = sum(1 for c in drop if c[1] is None)
    idx = sum(1 for c in drop if c[1] and c[1].startswith("index"))
    mid = sum(1 for c in drop if c[1] and c[1].startswith("middle"))
    over = ch[f"ur5_full_k{K}.xml"]["over_object"]
    s = (f'<p>In the simplified scene, cad_v2\'s 31 positions, 15 actuators and every joint and actuator name are kept. The '
         f'shared generator and the scripts that read those names therefore load it unchanged. Its finger mounts move to the centres of '
         f'the CAD\'s travel boxes (Section SEC[travel]). The palm becomes 11 static boxes on <code>palm_pose</code>, with '
         f'<code>palm_plate</code> first. These boxes cover the gantry frames, each bridge\'s swept volume, the x drives, both '
         f'plates, the bracket arm and the tag plate.</p>'
         f'<p>The full scene adds {len(me["written"])} merged part meshes with {tris:,} triangles, six lead screws as cylinders, '
         f'the tag and the v4b fingertip\'s pads. Its three gantry bridges slide on joints of their own, tied to '
         f'<code>&lt;f&gt;_y</code> by equality constraints. The finger links are silver, as the owner painted them. The '
         f'fingertips and servos are black.</p>'
         f'<p>At the open keyframe the simplified scene has {len(drop)} contacts, as cad_v2 has. They are the object on the floor '
         f'({obj}) and the index and middle fingertips on the floor ({idx} and {mid} spheres). No contact joins two parts of '
         f'the hand.</p>'
         f'<p>On the UR5, the arm\'s six joints replace the palm\'s pose joints, and the world is the robot\'s base frame. The '
         f'mat stands for the lab\'s cutting-mat table, with its top at 32.8 mm. The keyframe <code>sr2_night</code> is the arm\'s '
         f'logged pose of the night of 11-04. The keyframe <code>over_object</code> comes from the analytic inverse '
         f'kinematics. It puts the origin of $P$ 150 mm over the object with the palm level. Its position error is '
         f'{over["palm_position_error_mm"]:.0e} mm, and it has no contact.</p>')
    fg = (film(MEDIA / "20261104-cad_v3_on_ur5.mp4",
               f"The full model on the UR5 at clocking $k$ = {K}. The arm moves the hand from <code>lab_home</code> to "
               f"<code>over_object</code>. Each gantry then goes round the rim of its travel box, and the fingers close. The "
               f"frames are kinematic ({fi['frames']} frames, {fi['seconds']:.1f} s). Two geoms touch in {fi['contact_frames']} of "
               f"them (<code>scripts/cad_v3_film.py film</code>).")
          + img(MEDIA / "20261104-model_views.jpg",
                "Renders of the models. In the collision geometry, the palm's boxes are blue and the fingers' sphere pads orange."))
    return s + tb + fg


def sec_chain(C):
    F = C["F"]
    t = F["P"]["t_P_from_A_mm"]
    c = F["flange"]["centre_A_mm"]
    T = B.T_E_from_P(F, K)
    rows = []
    for k in B.K_ALL:
        Tk = B.T_E_from_P(F, k)
        tcp = F["tcp_P_origin_by_k"][k]
        ang = facing(F, k)
        lab = f"{k}" + (" (as built)" if k == K else "")
        rows.append([lab, f"{F['flange']['theta_k_deg'][k]:.2f}", tuple_mm(Tk[:3, 3]), tuple_mm(tcp[3:], 4),
                     faces_text(ang, 90.0), faces_text(ang, -90.0)])
    tb = table(["$k$", "$\\theta_k$ (deg)", "Origin of $P$ in $E$ (mm)", "Rotation vector (rad)", "HEX-E port faces",
                "Tool connector faces"], rows,
               "The UR TCP of the origin of $P$ for each clocking, and the gantry that the HEX-E's port and the robot's tool "
               "connector point at, seen from the flange axis. "
               "<code>scripts/build_cad_v3_hand.py tcp --k K --point X Y Z</code> gives the TCP of any point of $P$.", num_from=9)
    mat = " \\\\ ".join(" & ".join(f"{x:.3f}" if j < 3 else f"{x:.2f}" for j, x in enumerate(row)) for row in T[:3])
    return (f'<p>The transform from the palm to the flange frame is a product of three rigid transforms. Each comes from the '
            f'CAD or from a measurement of the sensor\'s seat.</p>'
            f'$$ {{}}^{{E}}T_{{P}}(k) = {{}}^{{E}}T_{{H}} \\; {{}}^{{H}}T_{{A}}(k) \\; \\left({{}}^{{P}}T_{{A}}\\right)^{{-1}} $$'
            f'<p>The first factor is the HEX-E on the flange. Its height is from the data sheet. Its turn is the dowel\'s angle '
            f'measured with the calibration mount in October, -92.15 deg from $x_E$, the mean of two seatings '
            f'(&plusmn;0.58 deg).</p>'
            f'$$ {{}}^{{E}}T_{{H}} = \\mathrm{{Trans}}(0, 0, 35.0)\\, R_z(-2.15^\\circ) $$'
            f'<p>The second factor is the adaptor on the HEX-E. The adaptor\'s four plate holes coincide with the robot mount\'s '
            f'holes, and their isosceles trapezoid fits in one way only. The tool axis then lies at the centre of its M6 holes, '
            f'$c_A$ = ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) mm in $A$ (a 50.000 mm circle, rms 4e-13 mm). The holes lie at '
            f'$-21 + 90\\,i$ deg from $x_A$.</p>'
            f'$$ {{}}^{{H}}T_{{A}}(k) = \\mathrm{{Trans}}(-R_k c_A)\\, R_k, \\qquad R_k = R_z(24^\\circ + 90^\\circ k)\\, R_x(180^\\circ) $$'
            f'<p>The third factor is the fixed map of the CAD frame to the palm, from the travel boxes and the yaw axes.</p>'
            f'$$ {{}}^{{P}}T_{{A}} = \\mathrm{{Trans}}({t[0]:.2f},\\, {t[1]:.2f},\\, {t[2]:.2f})\\, R_z(90^\\circ) $$'
            f'<p>The product at the clocking as built, $k$ = {K}, is below, in mm. The tool axis passes {math.hypot(T[0, 3], T[1, 3]):.2f} '
            f'mm from the origin of $P$, and the flange stands {T[2, 3]:.2f} mm above the yaw plane. The earlier UR5e arm scene used '
            f'100 mm for the wrist stack.</p>'
            f'$$ {{}}^{{E}}T_{{P}}({K}) = \\begin{{bmatrix}} {mat} \\\\ 0 & 0 & 0 & 1 \\end{{bmatrix}} $$'
            f'<p>In the MuJoCo scenes, <code>palm_pose</code> is placed on tool0 by this transform, with an error of '
            f'{C["check"][f"ur5_k{K}.xml"]["T_E_from_P_error_mm"]:.0e} mm. The adaptor\'s M6 holes fall on the HEX-E\'s, 25.0 mm from '
            f'the axis and 35.0 mm out. The UR5 CB3 model comes from the lab\'s <code>ur_description</code>. Its '
            f'tool0 agrees with the CB3\'s nominal DH table to 2.5e-13 mm at five random joint sets.</p>' + tb)


def sec_clocking(C):
    F, J, T = C["F"], C["clock"], C["tag"]
    fit, rim, port = J["fit"], J["rim_lower"], J["port"]
    by = port["by_k"]
    q6lo, q6hi = J["q6_log_range_deg"]
    rows = [[f"{r['k']}" + (" (as built)" if r["k"] == K else ""), f"{r['port_azimuth_A_deg']:.2f}", f"{r['from_camera_azimuth_deg']:.1f}",
             "Yes" if abs(r["from_camera_azimuth_deg"]) < 90 else "No", f"{r['best_z_A_mm']:.1f}", f"{r['rms_px']:.1f}"] for r in by]
    tb = table(["$k$", "Azimuth of $+y_E$ (deg)", "From the camera (deg)", "Camera's side", "Best $z_A$ (mm)", "Misfit (px)"], rows,
               "The HEX-E's port in the owner's photo for each clocking. The azimuth is measured in $A$ about $+z_A$ from $x_A$. The "
               "next two columns give its angle from the camera's azimuth, and whether the port would face the camera. Heights "
               "are searched over the HEX-E's span, $z_A$ 28 to 63 mm. The misfit is the port misfit.", num_from=1)
    others = [r["rms_px"] for r in by if r["k"] != K]
    k1 = next(r for r in by if r["k"] == K)
    return (img(MEDIA / "20261104-clockings_schematic.png",
                "The hand under the flange for the four clockings, seen from the tool side, with $z_E$ toward the viewer. The HEX-E "
                "is the grey disc, the base plate is tan, and the black lines are the robot mount's top face. Dots mark the "
                "centres of the travel boxes.")
            + f'<p>The hand as built is at clocking $k$ = {K}. The HEX-E\'s plug and the arm\'s wrist housing in the owner\'s '
            f'photo fit no other clocking. The photo was taken on 2026-11-04 from above the plate\'s $+x_A$ edge. A reading of '
            f'the same photo by eye had given $k$ = 2, from the plug\'s rough direction. The fit below replaces it.</p>'
            f'<p>A pinhole camera was fitted to {fit["n_holes"]} hole centres on the robot mount\'s top face, picked by hand on '
            f'zoomed crops. The principal point is at the image centre and the focal length is free. The reprojection error is '
            f'{fit["rms_px"]:.2f} px rms and {fit["max_px"]:.2f} px at most, with a focal length of {fit["f_px"]:.0f} px. The camera '
            f'stands at {tuple_mm(fit["camera_in_A_mm"], 0)} mm in $A$. Seen from the flange axis, its azimuth is '
            f'{fit["view_azimuth_A_deg"]:.1f} deg from $x_A$.</p>'
            f'<p>Points off the fitted plane test the fit and the stack\'s height. The wrist housing\'s lower rim, 7 mm behind the '
            f'flange face, falls within {rim["front"]["error_px"]:.1f} px of its picked height at the front. Its sides fall '
            f'within {rim["left"]["error_px"]:.1f} and {rim["right"]["error_px"]:.1f} px. So the 35.0 mm of the HEX-E from its data '
            f'sheet agrees with the photo to about 1 mm.</p>'
            + img(MEDIA / "20261104-clocking_overlay.jpg",
                  "The owner's photo with the fit. Magenta circles are the holes projected through the fitted camera. White "
                  "ellipses are the HEX-E's lower edge and the wrist housing's two rims. Each coloured line is $+y_E$ at the "
                  "port's height for one clocking. The plug lies on the line of $k$ = 1.")
            + f'<p>Four picks along the plug\'s axis give the port misfit in Table TAB[port]. The misfit is {k1["rms_px"]:.1f} px at '
            f'$k$ = {K} and {min(others):.1f} to {max(others):.1f} px at the others. At $k$ = 2 the segment comes within 47.6 px '
            f'only at the HEX-E\'s lower edge, and the port would then lie behind the cylinder, 132.9 deg from the camera\'s '
            f'azimuth. The first pick, placed on the 35 mm cylinder, gives the port at azimuth {port["base_azimuth_A_deg"]:.1f} '
            f'deg in $A$, against {k1["port_azimuth_A_deg"]:.2f} deg at $k$ = {K}. It also gives the port\'s height, '
            f'{port["base_z_E_mm"]:.1f} mm from the flange, which the models now use. The plug\'s cable leaves toward $-x_E$, as the '
            f'owner noted on 2026-10-22. Its image slope is {port["relief_slope_picked"]:.2f}, against '
            f'{port["relief_slope_by_k"][K]:.2f} in the model at $k$ = {K}.</p>'
            + tb
            + f'<p>The arm gives a second reading that does not use the plug. The wrist housing\'s direction in $A$ depends only on '
            f'joint 6 and $k$, and the outline of the next link also on joint 5. The log of the arm starts at 18:37 on 11-04, '
            f'after the photo. Over the night it shows joint 6 between {q6lo:.1f} and {q6hi:.1f} deg. With the last logged pose '
            f'(joint 5 at {J["arm_q_deg_night"][4]:.1f} deg, joint 6 at {J["arm_q_deg_night"][5]:.1f} deg), the render\'s outlines '
            f'at $k$ = {K} fall on the housing and the blue cap in the photo. At the other three clockings the housing falls '
            f'elsewhere.</p>'
            + img(MEDIA / "20261104-clocking_renders.jpg",
                  f"The owner's photo and the outlines of the render at $k$ = {K} over it. Below them, the full model is rendered from "
                  f"the fitted camera at each clocking, with the arm at its last logged pose.")
            + f'<p>On the robot, the clocking can be checked by eye from Table TAB[tcp]. At $k$ = {K} the HEX-E\'s plug points '
            f'between the thumb and index gantries, and the robot\'s tool connector points at the middle gantry. To change it, set '
            f'<code>AS_BUILT_K</code> in <code>scripts/build_cad_v3_hand.py</code> and use <code>ur5_k&lt;k&gt;.xml</code>.</p>')


def sec_travel(C):
    F = C["F"]
    nominal = {"thumb": (-50.0, 0.0), "index": (50.0, 55.0), "middle": (50.0, -55.0)}
    rows = []
    for f in FINGERS:
        fx = F["fingers"][f]
        c = fx["box_centre_P_mm"]
        d = fx["box_minus_nominal_mm"]
        st = fx["stops_disp_mm"]
        sx, sy = fx["box_size_P_mm"]
        rows.append([f.capitalize(), tuple_mm(c[:2]), f"({nominal[f][0]:g}, {nominal[f][1]:g})", tuple_mm(d),
                     f"{sx:g} x {sy:g}", f"{st['y_stage'][1] - st['y_stage'][0]:.2f} x {st['x_stage'][1] - st['x_stage'][0]:.2f}"])
    tb = table(["Finger", "CAD box centre (mm)", "Driver's position (mm)", "Difference (mm)", "Box (mm)", "Stroke (mm)"], rows,
               "Travel boxes in $P$, $x_P$ by $y_P$. The strokes are measured between the hard stops in the CAD. The boxes are "
               "anchored at the home stops.", num_from=1)
    tt, ti, tm = (np.array(F["fingers"][f]["box_centre_P_mm"]) for f in FINGERS)
    return (f'<p>The CAD\'s travel boxes are centred 6.18 mm further from the palm centre along $x_P$ than the driver\'s positions, '
            f'and 2.40 to 3.40 mm off along $y_P$. The spacing between the thumb and the pair is {ti[0] - tt[0]:.2f} mm, against '
            f'100 mm in the driver. The spacing between index and middle is {ti[1] - tm[1]:.2f} mm, against 110 mm.</p>'
            + tb
            + img(MEDIA / "20261104-travel_boxes.png",
                  "Travel boxes in the palm frame. Filled boxes with dots are the CAD's, and dashed boxes with crosses are the "
                  "driver's. The tan shape is the base plate, the grey lines the robot mount's top face, and the grey parts the "
                  "bracket arm and tag plate.")
            + '<p>The stops were found three ways: from the solids\' faces, from a mesh ray cast, and from the analysis\'s sweep. '
              'All agree within 0.007 mm. Every axis homes at the end that has its stepper, and the firmware commands 110 or 60 mm '
              'from home. That leaves 4.80 mm on the x stages and 2.58 mm on the y stages before the far stops. No stage limits '
              'another: each stage\'s stops are the same with the other stage at either end. At the index\'s corner of the hand '
              '(x stage at home, y stage at its far stop), the carriage passes 3.2 mm from the tag bracket.</p>'
              '<p>The thumb gantry sits on the base plate\'s holes (owner, 2026-11-04), 2.5 mm along $+y_A$ from where the export '
              'places it. As exported, the thumb\'s beam and rail block overlap the pair\'s by 1.6 mm. On the holes, every overlap '
              'clears by at least 0.9 mm.</p>'
              '<p>Three findings identify the fingers. All six axes home at their motor end only under this map of $A$ to $P$. The '
              'servos\' signs of 2026-08-29 reproduce on 3 of 3 fingers, against 0 of 3 for the mirror map. Each CAD carriage fits the '
              'cad_v1 mesh at 0.36 mm rms, against 1.8 mm for a mirrored one.</p>')


def sec_heights(C):
    F = C["F"]
    L = F["levels_P_mm"]
    return (img(MEDIA / "20261104-stack_heights.png",
                f"The full model in side view (orthographic, along $+y_P$, fingers straight, gantries at their box centres), with the "
                f"heights of the stack in $z_P$. The thumb is on the left.")
            + f'<p>The flange stands 138.90 mm above the yaw plane, and the adaptor\'s face {L["flange_face"]:.1f} mm. The bearing '
            f'blocks under the beams, at {L["beam_bearing_blocks"]:.1f} mm, are the lowest surface over the finger travel; cad_v2\'s '
            f'<code>palm_plate</code> had its underside at 23.5 mm. The beams\' motor ends reach {L["beam_lowest"]:.1f} mm, and the '
            f'carriages hang to {L["carriage_bottom"]:.1f} mm. The y lead screw under each beam lies at $z_P$ 29.1 to 34.1 mm, inside '
            f'the palm\'s boxes.</p>')


def sec_inventory(C):
    inv = [r for r in C["inv"] if r["file"] == "hand"]
    groups = [("Robot mount", ("robot_mount",)), ("Stepper motors, NEMA 8", ("nema8_stepper",)),
              ("Base plate", ("base_plate",)), ("Gantry frames", ("gantry_frame", "gantry_frame_thumb")),
              ("MGN7 rails", ("mgn7_rail_100", "mgn7_rail_150")), ("Beams", ("x_stage_beam", "x_stage_beam_thumb")),
              ("Servos, SCS0009", ("scs0009_servo",)), ("Shaft couplers", ("shaft_coupler",)),
              ("Finger carriages", ("finger_carriage",)), ("Adaptor", ("ur5_adaptor",)),
              ("MCP, yaw and DIP links", ("mcp_link", "yaw_link", "dip_link")),
              ("Lead screws", ("lead_screw_100", "lead_screw_150")), ("MGN7 blocks, lead nuts", ("mgn7_carriage", "lead_nut")),
              ("Bearings, rings, shafts", ("mr105_bearing", "ring_5x10x5", "joint_shaft", "stepper_shaft")),
              ("Bracket arm, tag plate, stepper mounts", ("sensor_arm", "sensor_bracket", "x_stepper_mount", "x_stepper_mount_thumb")),
              ("Fingertips (TPU block), horns", ("fingertip_tpu_block", "horn_disc", "horn_arm"))]
    rows, seen = [], set()
    for lab, kinds in groups:
        rs = [r for r in inv if r["kind"] in kinds]
        seen |= set(kinds)
        mats = ", ".join(sorted({(r["material"] or "").split(" (")[0] for r in rs}))
        mats = {"Base plate": "Hardboard in the owner's photos; counted as PLA", "Adaptor": "Not recorded; counted as PLA"}.get(lab, mats)
        rows.append([lab, mats[0].upper() + mats[1:], len(rs), f"{sum(r['mass_g'] or 0 for r in rs):.1f}"])
    left = [r for r in inv if r["kind"] not in seen]
    if left:
        raise SystemExit(f"inventory: kinds not grouped: {sorted({r['kind'] for r in left})}")
    total = sum(r["mass_g"] or 0 for r in inv)
    drop = [r for r in inv if r["visual_detail_droppable"]]
    tb = table(["Component", "Material", "Solids", "Mass (g)"], rows,
               "The hand file's solids by component. Masses are at solid density for the printed parts (PLA 1240, TPU 1210 "
               "kg/m&sup3;) and from data sheets for the bought parts.", num_from=2)
    allrows = len(C["inv"])
    return (f'<p>The six STEP files have {allrows} solids; the hand file has {len(inv)} of them. All are named "Part n" in the '
            f'export, so each was named from its volume, faces and holes (<code>assets/cad_v3/inventory.json</code>, one row per '
            f'solid). The solids weigh {total / 1000:.2f} kg at solid density, without the HEX-E\'s 0.26 kg. The prints are not solid: '
            f'the carriage slices to 14.15 g at 15 % infill against 29.65 g solid. The export has no screws, cables or boards. The '
            f'hand\'s weight therefore needs a measurement. The visual model leaves out {len(drop)} small solids '
            f'({sum(r["mass_g"] or 0 for r in drop):.1f} g) of bearings, rings and shafts.</p>'
            f'<p>The finger parts in the export are the cad_v1 parts, the original DIP link and the TPU block tip. The models use '
            f'cad_v2\'s fingers with the v4b tip, the tip that was calibrated on the UR5.</p>' + tb)


def sec_camera(C):
    F, T = C["F"], C["tag"]
    tg = F["sensors"]["tag"]
    c_P = rot.tf_apply(B.T_P_from_A(F), np.array(tg["centre_A"]))
    ph, re_ = T["photo"][0], T["render"][0]
    return (f'<p>The D435 is off the hand and stands on a tripod on the table (owner, 2026-11-04), so the models have no camera. Its '
            f'bracket arm <code>d435_plate</code> stays on the hand: it runs under the base plate\'s $+x_A$ edge and ends 52.6 mm '
            f'past it in a fork for a 20 x 20 mm extrusion. The tripod\'s pose has not been measured.</p>'
            f'<p>The tag plate <code>static_tag_mount</code> has tag36h11 id 6, a 40 mm black square on a 45 x 50 mm face. The '
            f'face looks along $+y_A = -x_P$, opposite the thumb\'s palmar face (owner, 2026-11-04). Its centre lies at '
            f'{tuple_mm(c_P, 1)} mm in $P$. The tag\'s axes in the AprilTag convention are $x$ along $+z_P$, $y$ along $-y_P$ and $z$ '
            f'along $+x_P$.</p>'
            f'<p>The owner\'s side photo shows the same axes. <code>pupil_apriltags</code> finds id 6 there, with the tag\'s $x$ axis '
            f'{abs(ph["x_from_image_up_deg"]):.1f} deg from the image\'s up and its $y$ axis {abs(ph["y_from_image_right_deg"]):.1f} '
            f'deg from the image\'s right. At the fitted focal length, its 40 mm edge reads {ph["side_px"]:.0f} px at '
            f'{ph["t_cam_tag_m"][2] * 1000:.0f} mm. In the full model the tag is drawn from the same image. From a camera 300 mm away '
            f'on its normal, the detector reads {re_["t_cam_tag_m"][2] * 1000:.1f} mm with the same axes.</p>'
            + img(MEDIA / "20261104-tag_side.jpg",
                  "The owner's side photo, and the model's outlines over it, with the camera placed from the tag's detected pose. The "
                  "arm is left out because the photo predates the log of its joints. The tag and the base plate's edge fall on the "
                  "real ones. The fingers and gantries differ because the model has them at its open keyframe."))


def sec_limits():
    items = [
        "The clocking was read from one photo. It relies on the HEX-E's port lying along $+y_E$, as the owner placed it on "
        "2026-10-22. The arm's reading also relies on joint 6 being near its logged value when the photo was taken.",
        "The adaptor has no dowel. With M6 bolts in 6.4 mm holes it can sit up to 0.2 mm off the axis and turned up to 0.46 deg, "
        "which the model does not include.",
        "The controller's reported flange differs from the nominal DH model by 0.82 to 1.77 mm (median 1.73 mm) and 0.38 to 0.41 "
        "deg on the lab's 27 recorded poses. The UR5 scenes use the nominal model.",
        "Cables, the fingertip boards and the mux boards on the thumb are not modelled. On the night of 11-04 the index stalled at "
        "mcp 45 and pip 80.3 deg. In the full model, the finger's moving links stand 30 mm from the nearest gantry part at that pose.",
        "The carriages' yaw bores are tilted 0.93 deg against their rail-block holes in the CAD. The models keep cad_v1's finger, "
        "which removed this tilt as an export artefact.",
        "The palm's inertia in the simulator is cad_v2's (0.0918 kg), and the arm's actuators are not tuned for the hand's mass.",
    ]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def sec_future():
    items = [
        f"Look at the hand on the robot: at $k$ = {K} the HEX-E's plug points between the thumb and index gantries, and the tool "
        "connector at the middle gantry. A RealSense view of the flange shows both. Any other direction falsifies the reading: set "
        "<code>AS_BUILT_K</code> and rerun <code>scripts/build_cad_v3_hand.py frames scenes check</code>.",
        "Measure each finger's yaw axis in $P$ with the gantries at home, by touching a known point with the fingertip under the "
        "UR5's TCP. The CAD predicts (-86.18, -57.40), (86.18, 88.40) and (86.18, -88.40) mm. If the measurement agrees within 1 mm, "
        "move <code>FINGER_GEOMETRY</code> and <code>REAL_V1_MOUNTS</code> to the CAD's centres. A difference near 6 mm along $x_P$ "
        "would keep the driver's numbers.",
        "Jog J2 on the bench and see which gantry moves. The index is expected to be the gantry beside the tag plate.",
        "Measure the yaw axis's tilt on one carriage with an angle gauge. A tilt of 0.93 deg would put the tilt back in the finger "
        "model; a level axis confirms that it is an export artefact.",
        "Weigh the hand, or read the HEX-E's $F_z$ and torques with the hand level, for the UR5's payload and centre of gravity. "
        "The count here is 2.98 kg at solid density, without screws, cables or boards.",
        "Measure the tripod camera's pose against tag id 6 with <code>scripts/real_v1_tracker_service.py</code>, once the D435 is "
        "plugged in on its final mount.",
        "Wire <code>SCALE_HAND=cad_v3</code> into the HOM and RL scripts, if the owner chooses to. Then rerun one HOM landscape, to "
        "measure how far the 6 mm move of the mounts shifts its best layouts.",
    ]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def sec_files():
    rows = [("assets/cad_v3/step/", "the owner's six STEP files"),
            ("assets/cad_v3/frames.json", "frames, transforms, travel boxes, heights, tag"),
            ("assets/cad_v3/inventory.json", "one row per solid"),
            ("assets/cad_v3/README.md", "how to use and rebuild the frames"),
            ("assets/mjcf/experimental/20261104-cad_v3/", "the ten MuJoCo scenes, meshes, check.json"),
            ("scripts/build_cad_v3_hand.py", "frames, meshes, scenes, checks, the UR TCP of a point"),
            ("scripts/cad_v3_assets.py", "the FreeCAD extraction of the STEP files"),
            ("scripts/cad_v3_film.py", "the clocking from the photo, stills and the film"),
            ("docs/experiments/20261104-cad_v3_hand/analysis/curl_check.py", "clearances of the index's stalled pose in the full model"),
            ("scripts/cad_v3_page.py", "this page and its drawings"),
            ("docs/experiments/20261104-cad_v3_hand/analysis/", "the analysis's scripts and results: parts, gantry, flange, "
             "sensors, UR5, current model and four checks"),
            ("docs/experiments/20261104-cad_v3_hand/20261104-clocking_from_photo.json", "the camera fit and the port against $k$")]
    return "<ul>" + "".join(f"<li><code>{a}</code>: {b}</li>" for a, b in rows) + "</ul>"


def lede_and_summary(C):
    F, J = C["F"], C["clock"]
    T = B.T_E_from_P(F, K)
    by = J["port"]["by_k"]
    k1 = next(r for r in by if r["k"] == K)
    others = [r["rms_px"] for r in by if r["k"] != K]
    lede = (f"The SR2 hand on the UR5 CB3 is modelled in MuJoCo at full visual fidelity and as a simplified drop-in for cad_v2. The "
            f"origin of the palm frame lies at {tuple_mm(T[:3, 3])} mm in the flange frame, in clocking $k$ = {K} of the adaptor. "
            f"That clocking comes from the owner's photo. A camera fitted to {J['fit']['n_holes']} plate holes "
            f"({J['fit']['rms_px']:.1f} px rms) puts the HEX-E's plug {k1['rms_px']:.1f} px from the model's port at $k$ = {K}. At the "
            f"other three clockings the distance is {min(others):.0f} to {max(others):.0f} px. The fingers' travel boxes are centred 6.2 mm further out "
            f"along $x_P$ than the driver's positions, and 2.4 to 3.4 mm off along $y_P$.")
    summary = (f"The SR2 hand on the UR5 CB3 is modelled in MuJoCo, full and simplified, with the flange-to-palm transform from the "
               f"CAD: the palm origin lies at {tuple_mm(T[:3, 3])} mm in tool0 in clocking k = {K}, read from the owner's photo "
               f"({k1['rms_px']:.1f} px port misfit against {min(others):.0f} to {max(others):.0f} px). The travel boxes are centred "
               f"about 6 mm outside the driver's positions.")
    return lede, summary


def render(page):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = texsvg.render(items, cache_path=TEX_CACHE)
    it = iter(svgs)
    return P.TEX_RE.sub(lambda m: next(it), page), len(items)


def cmd_page():
    C = load()
    sections = [
        ("terms", "Terms and metrics", sec_terms(C)),
        ("models", "MuJoCo models of the hand on the UR5", sec_models(C)),
        ("chain", "Transform from the flange to the palm", sec_chain(C)),
        ("clocking", "Clockings of the adaptor and the clocking as built", sec_clocking(C)),
        ("travel", "Finger travel against the driver's positions", sec_travel(C)),
        ("heights", "Heights along the tool axis", sec_heights(C)),
        ("inventory", "Part inventory", sec_inventory(C)),
        ("camera", "Camera and AprilTag", sec_camera(C)),
        ("limits", "Limitations", sec_limits()),
        ("future", "Future work", sec_future()),
        ("files", "Files", sec_files()),
    ]
    num_ = {sid: k for k, (sid, _h, _b) in enumerate(sections, 1)}
    sub = (lambda s: re.sub(r"SEC\[(\w+)\]", lambda m_: str(num_[m_.group(1)]), s))
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|kg|px|N\b|ms\b|s\b|g\b)", "\\1&nbsp;\\2", s_))
    body = [(sid, hd, unit(sub(b))) for sid, hd, b in sections]
    # table references by their captions' order
    html_all = "\n".join(b for _s, _h, b in body)
    caps = re.findall(r"<caption>Table (\d+)\. ([^<]{0,40})", html_all)
    ref = {}
    for n, cap in caps:
        if cap.startswith("The HEX-E's port in"):
            ref["port"] = n
        if cap.startswith("The UR TCP of"):
            ref["tcp"] = n
    tabs = (lambda s: re.sub(r"TAB\[(\w+)\]", lambda m_: ref[m_.group(1)], s))
    body = [(sid, hd, tabs(b)) for sid, hd, b in body]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(body, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{b}</section>' for k, (sid, hd, b) in enumerate(body, 1))
    lede, summary = lede_and_summary(C)
    byline = ("Model and analysis from the owner's Onshape export of 2026-11-04 and two photos of the hand on the robot. No bench "
              "measurement yet. Builder: <code>scripts/cad_v3_page.py</code>.")
    page = TEMPLATE.read_text()
    for k_, v in {"TITLE": "SR2 hand on UR5", "H1": "Model and transforms of the SR2 hand on the UR5 CB3",
                  "EYEBROW": "Hand and bench, 2026-11-04", "LEDE": unit(lede), "BYLINE": byline, "TOC": toc, "SECTIONS": secs,
                  "FOOTER": str(OUT.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k_ + "}}", v)
    page, n = render(page)
    page = docs_index.tag(page, topic="hardware", summary=summary, date="2026-11-04")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT}: {OUT.stat().st_size / 1e6:.2f} MB, {n} formulas", flush=True)
    return 0


def main():
    a = sys.argv[1:]
    if a and a[0] == "figs":
        return cmd_figs()
    if a and a[0] == "outlines":
        return cmd_outlines()
    return cmd_page()


if __name__ == "__main__":
    sys.exit(main())
