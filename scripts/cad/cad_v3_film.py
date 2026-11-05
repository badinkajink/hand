#!/usr/bin/env python3
"""Stills and the film of the cad_v3 models, and the clocking of the hand read from the owner's photo of 2026-11-04, for the
result page (scripts/cad/cad_v3_page.py, docs/experiments/20261104-cad_v3_hand/). All the frames are kinematic: qpos is set and
mj_forward run, with no dynamics.

    export MUJOCO_GL=egl
    R="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- .venv/bin/python scripts/cad/cad_v3_film.py"
    $R clocking     # camera fit on the plate's holes, port against k, renders from the photo's camera; JSON and figures
    $R stills       # the hand (full, boxes, drop-in) and the UR5
    $R tag          # a frontal render of the tag, and (after analysis/tag_in_photo.py) the model from the side photo's camera
    $R film         # the film: arm to over_object, gantries through their travel, fingers closing; contact report

The tag in the owner's side photo is read by docs/experiments/20261104-cad_v3_hand/analysis/tag_in_photo.py under
~/miniconda3/bin/python, which has pupil_apriltags.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import build_cad_v3_hand as B  # noqa: E402
from sr2hand.geometry import rotations as rot  # noqa: E402

D = ROOT / "docs/experiments/20261104-cad_v3_hand"
MEDIA = D / "media"
SCENES = B.OUT
K = B.AS_BUILT_K
PHOTO_ABOVE = MEDIA / "20261104-owner_photo_above.jpg"          # the owner's photos as sent, 4032 x 3024
PHOTO_SIDE = MEDIA / "20261104-owner_photo_side.jpg"
CLOCKING_JSON = D / "20261104-clocking_from_photo.json"
FILM = MEDIA / "20261104-cad_v3_on_ur5.mp4"
FILM_JSON = MEDIA / "20261104-film.json"
FINGERS = B.FINGERS
FONT = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"


# ------------------------------------------------------------------------------------------------- studio look
def load(path, width=1280, height=720, floor=True, extra=None):
    """A scene with a white background and a pale floor; its simulation numbers stay as they are. extra(spec) may add a
    camera before the compile."""
    import mujoco
    spec = mujoco.MjSpec.from_file(str(path))
    for t in spec.textures:
        if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            t.rgb1, t.rgb2 = [1.0, 1.0, 1.0], [0.97, 0.97, 0.98]
        elif t.name == "groundplane":
            t.rgb1, t.rgb2, t.markrgb = [0.80, 0.81, 0.82], [0.76, 0.77, 0.78], [0.66, 0.67, 0.69]
    for mat in spec.materials:
        if mat.name == "groundplane":
            mat.reflectance = 0.0
    v = spec.visual
    v.global_.offwidth, v.global_.offheight = max(width, 640), max(height, 480)
    v.quality.shadowsize = 8192
    v.map.shadowclip = 0.25                           # the shadow map's cover; its default leaves saw teeth on the links
    v.quality.offsamples = 8
    v.headlight.ambient = [0.42, 0.42, 0.42]
    v.headlight.diffuse = [0.42, 0.42, 0.42]
    v.headlight.specular = [0.10, 0.10, 0.10]
    for g in spec.geoms:
        if g.name == "floor":
            g.size = [8.0, 8.0, 0.1]
    if extra is not None:
        extra(spec)
    m = spec.compile()
    if not floor:
        for g in range(m.ngeom):
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) in ("floor", "mat"):
                m.geom_rgba[g, 3] = 0.0
                m.geom_matid[g] = -1
    return m


def scene_option(groups=(0, 1, 2)):
    import mujoco
    opt = mujoco.MjvOption()
    for g in range(6):
        opt.geomgroup[g] = 1 if g in groups else 0
    for g in range(6):
        opt.sitegroup[g] = 0
    return opt


def free_cam(lookat, dist, az, el):
    import mujoco
    c = mujoco.MjvCamera()
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = dist, az, el
    return c


def still(path, key, lookat=None, dist=0.6, az=135, el=-20, w=1000, h=640, groups=(0, 1, 2), floor=True):
    """A render of a keyframe from a free camera, as an array."""
    import mujoco
    m = load(path, w, h, floor=floor)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(key).id)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, h, w, max_geom=12000)
    try:
        r.update_scene(d, free_cam(lookat if lookat is not None else d.body("palm_pose").xpos + [0.0, 0.0, -0.015], dist, az, el),
                       scene_option=scene_option(groups))
        return np.array(r.render())
    finally:
        r.close()


# ------------------------------------------------------------------------------------------------- stills
def cmd_stills(a):
    MEDIA.mkdir(parents=True, exist_ok=True)
    full = SCENES / "scene_screwdriver_medium_full.xml"
    arm = SCENES / f"ur5_full_k{K}.xml"
    # the hand-only scenes hang the palm over the floor at the open keyframe
    tile([still(full, "open", dist=0.50, az=40, el=-18),
          still(full, "open", dist=0.50, az=40, el=24, floor=False),
          still(SCENES / "scene_screwdriver_medium.xml", "open", dist=0.50, az=40, el=-18),
          still(full, "open", dist=0.50, az=40, el=-18, groups=(0, 1, 3)),
          still(arm, "over_object", lookat=[-0.10, -0.26, 0.27], dist=1.35, az=116, el=-13),
          still(arm, "sr2_night", dist=0.72, az=62, el=-8)],
         ["Full model, from above", "Full model, from below", "Simplified model", "Collision geometry of the simplified model",
          f"Full model on the UR5, over_object, k = {K}", f"Full model on the UR5, sr2_night, k = {K}"],
         2, font_px=36, out=MEDIA / "20261104-model_views.jpg")


# ------------------------------------------------------------------------------------------------- film
def smooth(t):
    """Minimum-jerk blend of t in [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def box_path(sx, sy):
    """The corners of a finger's travel box, from its centre round the rectangle and back: (x, y) in m about the centre."""
    return np.array([[0, 0], [sx, 0], [sx, sy], [-sx, sy], [-sx, -sy], [sx, -sy], [sx, 0], [0, 0]], float)


def along(P, s):
    """The point at arc-length fraction s of polyline P."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)] / seg.sum()
    i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
    f = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-12)
    return P[i] + f * (P[i + 1] - P[i])


# closing posture, rad (yaw, mcp, pip): the three tips about 20 mm apart under the palm (checked by the film's contact report)
CLOSED = {"thumb": (0.0, 0.50, 0.40), "index": (-0.58, 0.50, 0.40), "middle": (0.58, 0.50, 0.40)}
CAM_WIDE = dict(lookat=np.array([-0.10, -0.26, 0.27]), dist=1.50, az=116.0, el=-13.0)
CAM_NEAR = dict(lookat=np.array([-0.22, -0.48, 0.205]), dist=0.60, az=112.0, el=14.0)
CAM_TIPS = dict(lookat=np.array([-0.22, -0.48, 0.160]), dist=0.42, az=128.0, el=12.0)


def blend_cam(a, b, s):
    return dict(lookat=a["lookat"] + s * (b["lookat"] - a["lookat"]), dist=a["dist"] * (b["dist"] / a["dist"]) ** s,
                az=a["az"] + s * (b["az"] - a["az"]), el=a["el"] + s * (b["el"] - a["el"]))


def timeline(m):
    """Yields (qpos, camera dict, label, tag) per frame at 25 fps."""
    import mujoco
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("over_object").id)
    q_over = d.qpos.copy()
    mujoco.mj_resetDataKeyframe(m, d, m.key("lab_home").id)
    q_home = d.qpos.copy()
    adr = {j: m.jnt_qposadr[m.joint(j).id] for f in FINGERS
           for j in (f"{f}_x", f"{f}_y", f"{f}_yaw", f"{f}_mcp", f"{f}_pip", f"{f}_bridge")}
    fps = 25
    q = q_home.copy()
    # the arm
    n = 5 * fps
    for i in range(n):
        q = q_home + smooth(i / (n - 1)) * (q_over - q_home)
        yield q.copy(), CAM_WIDE, "The arm moves the hand from lab_home to 150 mm over the object", "arm"
    # the camera closes in
    n = 2 * fps
    for i in range(n):
        yield q.copy(), blend_cam(CAM_WIDE, CAM_NEAR, smooth(i / (n - 1))), "The hand over the object, clocking k = 1", "cam"
    # the fingers straighten: at the open key's yaw the index and middle tips meet at the inner end of their boxes
    q0 = q.copy()
    n = int(1.5 * fps)
    for i in range(n):
        s = smooth(i / (n - 1))
        for f in FINGERS:
            for j in ("yaw", "mcp", "pip"):
                a = adr[f"{f}_{j}"]
                q[a] = q0[a] * (1.0 - s)
        yield q.copy(), CAM_NEAR, "The fingers straighten", "fingers0"
    # the gantries, one at a time, round the rim of each travel box
    size = {"thumb": "60 x 110", "index": "60 x 60", "middle": "60 x 60"}
    for f in ("thumb", "index", "middle"):
        lo_x, hi_x = m.jnt_range[m.joint(f"{f}_x").id]
        lo_y, hi_y = m.jnt_range[m.joint(f"{f}_y").id]
        P = box_path(hi_x, hi_y)
        n = int(4.5 * fps)
        for i in range(n):
            x, y = along(P, float(smooth(i / (n - 1))))
            q[adr[f"{f}_x"]], q[adr[f"{f}_y"]] = x, y
            q[adr[f"{f}_bridge"]] = y
            yield q.copy(), CAM_NEAR, f"{f.capitalize()} gantry round its {size[f]} mm travel box", f"gantry_{f}"
    # the fingers close
    n = fps
    for i in range(n):
        yield q.copy(), blend_cam(CAM_NEAR, CAM_TIPS, smooth(i / (n - 1))), "The fingers close", "cam"
    q0 = q.copy()
    n = 3 * fps
    for i in range(n):
        s = smooth(i / (n - 1))
        for f in FINGERS:
            for j, tgt in zip(("yaw", "mcp", "pip"), CLOSED[f]):
                a = adr[f"{f}_{j}"]
                q[a] = q0[a] + s * (tgt - q0[a])
        yield q.copy(), CAM_TIPS, "The fingers close", "fingers"
    for i in range(fps):
        yield q.copy(), CAM_TIPS, "The fingers close", "hold"


def hand_contacts(m, d):
    """Contacts between robot and hand parts or among hand parts, excluding the object's and the floor's."""
    import mujoco
    out = set()
    for c in d.contact[:d.ncon]:
        a, b = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"#{g}" for g in (c.geom1, c.geom2))
        out.add((a, b))
    return out


def label_bar(frame, text, font):
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im)
    W, H = im.size
    bar = 44
    dr.rectangle([0, H - bar, W, H], fill=(255, 255, 255))
    dr.line([0, H - bar, W, H - bar], fill=(0, 0, 0), width=1)
    dr.text((14, H - bar + 9), text, fill=(0, 0, 0), font=font)
    return np.asarray(im)


def cmd_film(a):
    import mujoco
    from PIL import Image, ImageFont
    w, h = 1280, 720
    m = load(SCENES / f"ur5_full_k{K}.xml", w, h)
    d = mujoco.MjData(m)
    font = ImageFont.truetype(FONT, 26)
    r = mujoco.Renderer(m, h, w, max_geom=12000)
    opt = scene_option()
    FILM.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25",
                            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow",
                            "-movflags", "+faststart", str(FILM)], stdin=subprocess.PIPE)
    contacts, hit_frames, n, tips = {}, 0, 0, []
    poster = None
    try:
        for q, cam, text, tag in timeline(m):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            hc = hand_contacts(m, d)
            if hc:
                hit_frames += 1
                for pair in hc:
                    contacts.setdefault(str(pair), [n, n])[1] = n
            if tag in ("fingers", "hold"):
                pts = {f: d.body(f"{f}_tip").xpos.copy() for f in FINGERS}
                tips.append([float(np.linalg.norm(pts["thumb"] - pts["index"])), float(np.linalg.norm(pts["thumb"] - pts["middle"])),
                             float(np.linalg.norm(pts["index"] - pts["middle"]))])
            r.update_scene(d, free_cam(cam["lookat"], cam["dist"], cam["az"], cam["el"]), scene_option=opt)
            frame = label_bar(r.render(), text, font)
            enc.stdin.write(frame.tobytes())
            if a.frames and tag in ("arm", "gantry_thumb", "gantry_index", "gantry_middle", "fingers", "hold") and n % a.frames == 0:
                Image.fromarray(frame).save(a.frame_dir / f"f{n:04d}.png")
            if tag == "gantry_index" and poster is None:
                poster = frame.copy()
            n += 1
    finally:
        r.close()
        enc.stdin.close()
        enc.wait()
    Image.fromarray(poster).convert("RGB").save(FILM.with_suffix(".jpg"), quality=88)
    rep = dict(frames=n, fps=25, seconds=n / 25.0, width=w, height=h, scene=f"ur5_full_k{K}.xml", contact_frames=hit_frames,
               contacts=contacts, closed_rad=CLOSED, tip_distance_mm_end=[round(1000 * x, 1) for x in tips[-1]])
    FILM_JSON.write_text(json.dumps(rep, indent=1) + "\n")
    print(f"wrote {FILM.relative_to(ROOT)}: {n} frames, {FILM.stat().st_size / 1e6:.1f} MB; frames with a contact: {hit_frames}; "
          f"tips at the end {rep['tip_distance_mm_end']} mm", flush=True)
    for k_, v in sorted(contacts.items())[:30]:
        print("  contact", k_, v)


# ------------------------------------------------------------------------------------------------- clocking from the photo
# Plate holes in the owner's photo from above the +x_A edge (full-resolution px, read on zoomed crops): the hex pockets on the
# robot mount's top face, z_A 28, keyed by their (x_A, y_A) in mm from extract/parts.json
HOLES = {(-4.5, 4.5): (632, 1391), (6.5, 16.0): (769, 1414), (-4.5, 100.0): (1450, 1141), (6.5, 117.0): (1617, 1141),
         (-1.5, 173.8): (1959, 993), (-35.5, 238.38): (2181, 783), (-217.5, 4.5): (181, 676), (-228.5, 16.0): (252, 634),
         (-89.0, 16.0): (512, 1021), (-133.0, 16.0): (419, 881), (-105.5, 4.5): (379, 992), (-116.5, 4.5): (355, 956),
         (-186.5, 238.38): (1662, 463)}
# The plug's axis in the photo: boss, knurled ring, plug body, body's end; the housing's lower rim: front, left, right
PLUG = [(1355, 650), (1402, 662), (1490, 678), (1540, 690)]
RIM = dict(front=(1154, 594), left=(866, 503), right=(1436, 485))
RELIEF = [(1455, 700), (1375, 735)]                               # the cable's strain relief, from the plug's body outward
UR5_LOG = ROOT / "docs/experiments/20261104-sr2_hand_crosstalk/data/20261104-night/ur5_pose.jsonl"
PHOTO_W, PHOTO_H = 4032, 3024


def fit_camera():
    from scipy.optimize import least_squares
    X = np.array([[x, y, 28.0] for x, y in HOLES])
    u = np.array(list(HOLES.values()), float)
    best = None
    for seed in range(30):
        rng = np.random.default_rng(seed)
        p0 = np.r_[rng.normal(size=3) * 2.0, rng.normal(size=2) * 200.0, 500.0 + rng.random() * 800.0, 3000.0]
        res = least_squares(lambda p: (project(p, X) - u).ravel(), p0, method="lm", max_nfev=4000)
        if best is None or res.cost < best.cost:
            best = res
    return best.x, project(best.x, X) - u


def project(p, X):
    R = rot.exp_so3(p[:3])
    c = np.atleast_2d(X) @ R.T + p[3:6]
    return np.c_[p[6] * c[:, 0] / c[:, 2] + PHOTO_W / 2, p[6] * c[:, 1] / c[:, 2] + PHOTO_H / 2]


def seg_dist(P, a, b):
    """Distance of points P from the segment a-b (px)."""
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def port_misfit(p, F):
    """For each k: the height z_A (over the HEX-E's 28..63) at which the model's port segment, radius 35 to 70 mm from the axis
    along +y_E, lies nearest the picks of the plug, and the rms distance there."""
    c = np.array(F["flange"]["centre_A_mm"])[:2]
    picks = np.array(PLUG, float)
    out = []
    for k in B.K_ALL:
        th = math.radians(F["flange"]["theta_k_deg"][k])
        y_E = np.array([math.sin(th), -math.cos(th)])
        best = (1e9, None)
        for z in np.arange(28.0, 63.01, 0.25):
            a, b = project(p, np.array([[*(c + 35.0 * y_E), z], [*(c + 70.0 * y_E), z]]))
            e = float(np.sqrt(np.mean(seg_dist(picks, a, b) ** 2)))
            if e < best[0]:
                best = (e, float(z))
        az = math.degrees(math.atan2(y_E[1], y_E[0]))
        C_A = -rot.exp_so3(p[:3]).T @ p[3:6]
        off = (az - math.degrees(math.atan2(*(C_A[:2] - c)[::-1])) + 180.0) % 360.0 - 180.0
        out.append(dict(k=k, port_azimuth_A_deg=round(az, 2), from_camera_azimuth_deg=round(off, 1), best_z_A_mm=best[1],
                        rms_px=round(best[0], 1)))
    return out


def port_base(p, F):
    """Azimuth in A and height of the port's base from its first pick, the point on the HEX-E's 35 mm cylinder."""
    from scipy.optimize import least_squares
    c = np.array(F["flange"]["centre_A_mm"])[:2]
    u = np.array(PLUG[0], float)
    res = least_squares(lambda x: (project(p, np.array([[c[0] + 35.0 * math.cos(x[0]), c[1] + 35.0 * math.sin(x[0]), x[1]]]))[0] - u),
                        [0.5, 50.0])
    return math.degrees(res.x[0]), float(res.x[1])


def photo_cam(p, T_W_A):
    """Pose in the world of a MuJoCo camera at the photo's camera, for hand frame A at T_W_A (mm): position in m, wxyz."""
    R = rot.exp_so3(p[:3])
    C_A = -R.T @ p[3:6]
    R_A_cam = R.T @ np.diag([1.0, -1.0, -1.0])            # MuJoCo's camera looks along its -z with y up
    pos = rot.tf_apply(T_W_A, C_A) / 1000.0
    return pos, rot.quat_canonical(rot.mat_to_quat(T_W_A[:3, :3] @ R_A_cam))


def renders_from_photo(p, F, w=1008, h=756):
    """One render per clocking from the photo's camera, the arm at the night's logged pose (group 2 visuals, no floor)."""
    import mujoco
    from PIL import Image
    fovy = 2.0 * math.degrees(math.atan(0.5 * PHOTO_H / p[6]))
    out = []
    for k in B.K_ALL:
        def add(spec):
            spec.worldbody.add_camera(name="photo", pos=[0, 0, 1], fovy=fovy)
        m = load(SCENES / f"ur5_full_k{k}.xml", w, h, floor=False, extra=add)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("sr2_night").id)
        d.qpos[:3] = (0.0, 0.0, -5.0)                     # the object is not in the photo
        mujoco.mj_forward(m, d)
        T_W_E = rot.tf(d.site("tool0").xmat.reshape(3, 3), d.site("tool0").xpos * 1000.0)
        T_W_A = T_W_E @ B.T_E_from_A(F, k)
        c = m.cam("photo").id
        m.cam_pos[c], m.cam_quat[c] = photo_cam(p, T_W_A)
        m.cam_mode[c], m.cam_bodyid[c] = 0, 0
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, h, w, max_geom=12000)
        try:
            r.update_scene(d, "photo", scene_option=scene_option())
            out.append(np.array(r.render()))
        finally:
            r.close()
        # the wrist 2 axis in A at the logged q6, toward its cap
        j = m.joint("wrist_2_joint").id
        a_A = T_W_A[:3, :3].T @ d.xaxis[j]
        print(f"k = {k}: wrist 2 axis in A {np.round(a_A, 3)}", flush=True)
    return out


def tile(images, labels, cols, font_px=30, out=None, quality=88):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(FONT, font_px)
    bar = font_px + 14
    w, h = images[0].shape[1], images[0].shape[0]
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * w, rows * (h + bar)), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (im, lb) in enumerate(zip(images, labels)):
        x, y = (i % cols) * w, (i // cols) * (h + bar)
        dr.text((x + 8, y + 5), lb, fill=(0, 0, 0), font=font)
        sheet.paste(Image.fromarray(im), (x, y + bar))
    sheet.save(out, quality=quality)
    print("wrote", Path(out).relative_to(ROOT), sheet.size, flush=True)


def cmd_clocking(a):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    F = B.frames()
    p, res = fit_camera()
    rms, mx = float(np.sqrt((res ** 2).sum(1).mean())), float(np.abs(res).max())
    R = rot.exp_so3(p[:3])
    C_A = -R.T @ p[3:6]
    c = np.array(F["flange"]["centre_A_mm"])[:2]
    fit = dict(f_px=round(float(p[6]), 1), rms_px=round(rms, 2), max_px=round(mx, 2), n_holes=len(HOLES),
               camera_in_A_mm=C_A.round(1).tolist(), view_azimuth_A_deg=round(math.degrees(math.atan2(*(C_A[:2] - c)[::-1])), 1))
    # the wrist housing's lower rim (z_E -7, radius 37.5 mm) through the fit
    th = np.linspace(0.0, 2.0 * math.pi, 720)
    ring = np.c_[c[0] + 37.5 * np.cos(th), c[1] + 37.5 * np.sin(th), np.full(th.shape, 63.0 + 7.0)]
    q = project(p, ring)
    rim_pred = dict(front=q[np.argmax(q[:, 1])], left=q[np.argmin(q[:, 0])], right=q[np.argmax(q[:, 0])])
    rim = {s: dict(pick=list(RIM[s]), model=np.round(rim_pred[s], 1).tolist(),      # front: its height; sides: their extent
                   error_px=round(float(abs(rim_pred[s][1] - RIM[s][1]) if s == "front" else abs(rim_pred[s][0] - RIM[s][0])), 1))
           for s in RIM}
    port = port_misfit(p, F)
    base_az, base_z = port_base(p, F)
    night = [float(x) for x in B.SR2_NIGHT_Q_DEG]
    q6 = [math.degrees(json.loads(l)["q"][5]) for l in UR5_LOG.read_text().splitlines() if '"q"' in l]
    relief = []
    for k in B.K_ALL:
        t_ = math.radians(F["flange"]["theta_k_deg"][k])
        x_E = np.array([math.cos(t_), math.sin(t_)])
        y_E = np.array([math.sin(t_), -math.cos(t_)])
        a_ = c + y_E * (35.0 + B.HEX_E["relief_at_mm"])
        s_, e_ = project(p, np.array([[*a_, base_z], [*(a_ - x_E * B.HEX_E["relief_mm"]), base_z]]))
        relief.append(round(float((e_[1] - s_[1]) / (e_[0] - s_[0])), 2))
    rel = np.array(RELIEF, float)
    J = dict(photo=PHOTO_ABOVE.name, size_px=[PHOTO_W, PHOTO_H], fit=fit, rim_lower=rim,
             port=dict(picks_px=PLUG, by_k=port, base_azimuth_A_deg=round(base_az, 1), base_z_A_mm=round(base_z, 1),
                       base_z_E_mm=round(63.0 - base_z, 1), relief_slope_by_k=relief, relief_picks_px=RELIEF,
                       relief_slope_picked=round(float((rel[1, 1] - rel[0, 1]) / (rel[1, 0] - rel[0, 0])), 2)),
             q6_log_range_deg=[round(min(q6), 2), round(max(q6), 2)],
             holes=[dict(x_A=x, y_A=y, u=u[0], v=u[1], du=round(float(e[0]), 1), dv=round(float(e[1]), 1))
                    for (x, y), u, e in zip(HOLES, HOLES.values(), res)],
             arm_q_deg_night=night, k=K)
    CLOCKING_JSON.write_text(json.dumps(J, indent=1) + "\n")
    print(json.dumps({k_: J[k_] for k_ in ("fit", "rim_lower", "port")}, indent=0), flush=True)
    # figure: the photo with the holes, the HEX-E rims and +y_E at each k
    im = ImageOps.exif_transpose(Image.open(PHOTO_ABOVE)).convert("RGB")
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, 60)
    for q_ in project(p, np.array([[x, y, 28.0] for x, y in HOLES])):
        dr.ellipse([q_[0] - 13, q_[1] - 13, q_[0] + 13, q_[1] + 13], outline=(255, 0, 255), width=4)
    for z, r_ in ((28.0, 35.0), (70.0, 37.5), (97.5, 37.5)):
        ring = np.c_[c[0] + r_ * np.cos(th), c[1] + r_ * np.sin(th), np.full(th.shape, z)]
        dr.line([tuple(x) for x in project(p, ring)], fill=(255, 255, 255), width=3)
    cols = [(230, 60, 40), (40, 200, 70), (70, 150, 255), (250, 220, 0)]
    z_port = base_z
    for k in B.K_ALL:
        t_ = math.radians(F["flange"]["theta_k_deg"][k])
        y_E = np.array([math.sin(t_), -math.cos(t_)])
        s, e = project(p, np.array([[*c, z_port], [*(c + 88.0 * y_E), z_port]]))
        dr.line([tuple(s), tuple(e)], fill=cols[k], width=9)
        dr.text(tuple(e + np.array([8, -34])), f"k = {k}", fill=cols[k], font=font, stroke_width=3, stroke_fill=(0, 0, 0))
    im.crop((0, 200, 2500, 1800)).resize((1500, 960), Image.LANCZOS).save(MEDIA / "20261104-clocking_overlay.jpg", quality=88)
    print("wrote the overlay", flush=True)
    # figure: the photo, the outlines of the render at k = 1 over it, and a render per clocking from the photo's camera
    from scipy import ndimage as ndi
    photo = np.asarray(ImageOps.exif_transpose(Image.open(PHOTO_ABOVE)).convert("RGB").resize((1008, 756), Image.LANCZOS))
    rs = renders_from_photo(p, F)
    g = np.asarray(Image.fromarray(rs[K]).convert("L"), float)
    ph = photo.copy()
    ph[np.hypot(ndi.sobel(g, 0), ndi.sobel(g, 1)) > 70] = (255, 40, 40)
    crop = (lambda x: x[:520, :760])
    tile([crop(photo), crop(ph)] + [crop(r) for r in rs],
         ["Owner's photo", "Outlines of the render at k = 1 over the photo", "Render at k = 0", "Render at k = 1",
          "Render at k = 2", "Render at k = 3"], 2, out=MEDIA / "20261104-clocking_renders.jpg")


def cmd_tag(a):
    """A frontal render of the full model's tag for analysis/tag_in_photo.py, then, when its JSON exists, the model seen from
    the side photo's camera (placed from the tag's pose in that photo)."""
    import mujoco
    from PIL import Image, ImageOps
    F = B.frames()
    T_A_tag = np.array(F["sensors"]["tag"]["pose_A"], float).reshape(4, 4)
    w, h, fovy = 1280, 960, 40.0
    f_px = 0.5 * h / math.tan(math.radians(fovy) / 2.0)
    dist_mm = 300.0
    # the camera on the tag's outward normal, its image up along the tag's x (as the plate hangs)
    cam_in_tag = np.array([0.0, 0.0, -dist_mm])
    R_cam_tag = rot.rot_z(math.pi / 2)                    # tag x -> image up (camera -y), tag y -> image right (camera +x)
    R_tag_cam = R_cam_tag.T
    T_A_cam = T_A_tag @ rot.tf(R_tag_cam, cam_in_tag)

    def add(spec):
        spec.worldbody.add_camera(name="photo", pos=[0, 0, 1], fovy=fovy)
    m = load(SCENES / f"ur5_full_k{K}.xml", w, h, floor=False, extra=add)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("sr2_night").id)
    d.qpos[:3] = (0.0, 0.0, -5.0)
    mujoco.mj_forward(m, d)
    T_W_A = rot.tf(d.site("tool0").xmat.reshape(3, 3), d.site("tool0").xpos * 1000.0) @ B.T_E_from_A(F, K)

    def set_cam(T_A_c):
        c = m.cam("photo").id
        m.cam_pos[c] = rot.tf_apply(T_W_A, T_A_c[:3, 3]) / 1000.0
        m.cam_quat[c] = rot.quat_canonical(rot.mat_to_quat(T_W_A[:3, :3] @ T_A_c[:3, :3] @ np.diag([1.0, -1.0, -1.0])))
        m.cam_mode[c], m.cam_bodyid[c] = 0, 0
        mujoco.mj_forward(m, d)
    set_cam(T_A_cam)
    r = mujoco.Renderer(m, h, w, max_geom=12000)
    try:
        r.update_scene(d, "photo", scene_option=scene_option())
        Image.fromarray(r.render()).save(MEDIA / "20261104-tag_render.png")
    finally:
        r.close()
    (MEDIA / "20261104-tag_render.json").write_text(json.dumps(dict(f_px=f_px, distance_mm=dist_mm, width=w, height=h), indent=1) + "\n")
    print("wrote the tag's render; run analysis/tag_in_photo.py next", flush=True)
    jp = D / "analysis/tag_in_photo.json"
    if not jp.exists():
        return
    J = json.loads(jp.read_text())
    det = J["photo"][0]
    T_cam_tag = rot.tf(np.array(det["R_cam_tag"]), np.array(det["t_cam_tag_m"]) * 1000.0)
    T_A_c = T_A_tag @ rot.tf_inv(T_cam_tag)
    wp, hp = 1008, 756
    fovy_p = 2.0 * math.degrees(math.atan(0.5 * PHOTO_H / J["f_px"]))

    def add2(spec):
        spec.worldbody.add_camera(name="photo", pos=[0, 0, 1], fovy=fovy_p)
    m = load(SCENES / f"ur5_full_k{K}.xml", wp, hp, floor=False, extra=add2)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("sr2_night").id)
    d.qpos[:3] = (0.0, 0.0, -5.0)
    mujoco.mj_forward(m, d)
    T_W_A = rot.tf(d.site("tool0").xmat.reshape(3, 3), d.site("tool0").xpos * 1000.0) @ B.T_E_from_A(F, K)
    set_cam(T_A_c)
    # the arm is left out: this photo predates the log of its joints
    for g in range(m.ngeom):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        if bn.endswith("_link") or bn == "base_link_inertia":
            m.geom_rgba[g, 3] = 0.0
            m.geom_matid[g] = -1
    r = mujoco.Renderer(m, hp, wp, max_geom=12000)
    try:
        r.update_scene(d, "photo", scene_option=scene_option())
        ren = np.array(r.render())
    finally:
        r.close()
    from scipy import ndimage as ndi
    photo = np.asarray(ImageOps.exif_transpose(Image.open(PHOTO_SIDE)).convert("RGB").resize((wp, hp), Image.LANCZOS))
    g = np.asarray(Image.fromarray(ren).convert("L"), float)
    ph = photo.copy()
    ph[np.hypot(ndi.sobel(g, 0), ndi.sobel(g, 1)) > 70] = (255, 40, 40)
    tile([photo, ph], ["Owner's side photo", "Outlines of the render, placed by the tag, over the photo"], 2, font_px=36,
         out=MEDIA / "20261104-tag_side.jpg")
    print("camera in A (mm)", T_A_c[:3, 3].round(1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("clocking")
    sp.add_parser("stills")
    sp.add_parser("tag")
    f = sp.add_parser("film")
    f.add_argument("--frames", type=int, default=0, help="save every nth frame to --frame-dir")
    f.add_argument("--frame-dir", type=Path, default=Path("logs/20261104-cad_v3/film_frames"))
    a = ap.parse_args()
    if a.cmd == "clocking":
        cmd_clocking(a)
    elif a.cmd == "stills":
        cmd_stills(a)
    elif a.cmd == "tag":
        cmd_tag(a)
    elif a.cmd == "film":
        if a.frames:
            a.frame_dir.mkdir(parents=True, exist_ok=True)
        cmd_film(a)


if __name__ == "__main__":
    main()
