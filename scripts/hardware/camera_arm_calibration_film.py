#!/usr/bin/env python3
"""Film and overlay stills of the D435 camera calibration of 2026-11-13 in the cad_v3 MuJoCo scene of the SR2 hand on
the UR5 (assets/mjcf/experimental/20261104-cad_v3/ur5_full_k1.xml, the hand as built). Every frame is kinematic:
qpos is set and mj_kinematics run, with contacts off.

Added to the scene, from docs/experiments/20261113-camera_arm_calibration/data/fits/: the paper ChArUco board on tool0
(the hand-eye fit through ur_kinematics.fk, the model's own kinematics), both cameras at their fitted poses in the base
with their fitted focal lengths and principal points, and the hex block and the table from
20261113-block_and_table.json (camera_arm_calibration_page.py data). The board is drawn with its ChArUco pattern as a
texture. The hand stands as it was during the session (HAND below): the gantries at the station's home layout, the
fingers limp with servo torque off since 11-06, at the angles read from the servos on 11-13 with the arm within 6 deg
of the session's untilted poses.

    film     the arm through the 17 recorded poses in their order, the overview with both cameras' viewing pyramids on
             the left and the two cameras' renders on the right -> media/20261113-calibration_replay.mp4 and a poster
    overlay  poses 0, 3 and 12: each camera's frame (logs/, corvus only), undistorted with its fitted lens, blended
             with the render from the fitted camera -> media/20261113-overlay_pNN.jpg. The owner sat in view of
             836612071918, left of x = 860 px: that part of its frames is blurred. At every pose the ChArUco corners
             are detected in the render and in the undistorted frame -> data/fits/20261113-render_overlap.json.

    MUJOCO_GL=egl uv run --no-project --python 3.10 --with mujoco --with numpy --with scipy \
        --with opencv-python-headless python scripts/hardware/camera_arm_calibration_film.py film|overlay
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
from sr2hand.geometry import rotations as rot  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

D = ROOT / "docs/experiments/20261113-camera_arm_calibration"
DATA, FITS, MEDIA = D / "data", D / "data/fits", D / "media"
LOGS = ROOT / "logs/20261113-camera_arm_calibration"
XML = ROOT / "assets/mjcf/experimental/20261104-cad_v3/ur5_full_k1.xml"
ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
COL = {"832412070344": (1.0, 0.55, 0.1, 1.0), "836612071918": (0.15, 0.45, 1.0, 1.0)}
SQ = 15.0
# The station's home layout (palm mm, lab/make_sysid_plan.py HOME, held with --keep-home since 11-05) less each finger's
# travel-box centre in P (assets/cad_v3/frames.json) gives the model's <f>_x, <f>_y; <f>_bridge follows <f>_y.
HOME_PALM_MM = {"thumb": (-80.0, -55.0), "index": (80.0, 85.0), "middle": (80.0, -85.0)}
# Servo angles in the station's degrees from each calibrated zero (lab/scs0009_registers.py on the CB1, 11-13 14:4x,
# torque register 2: limp). The model's yaw is minus the station's for index and middle (run_plan.YAW_SIGN).
SERVO_DEG = {"thumb": dict(yaw=6.16, mcp=88.48, pip=0.00), "index": dict(yaw=0.29, mcp=-49.80, pip=-2.05),
             "middle": dict(yaw=7.91, mcp=69.73, pip=0.00)}
YAW_SIGN = {"thumb": 1.0, "index": -1.0, "middle": -1.0}


def j(p):
    return json.loads(Path(p).read_text())


def quat_of(R):
    return rot.mat_to_quat(np.asarray(R))


def build():
    lens = j(FITS / "20261113-lens_board_k3_arm.json")["cams"]
    he = j(FITS / "20261113-handeye_fk_fitted_lens.json")
    bt = j(FITS / "20261113-block_and_table.json")
    sp = mujoco.MjSpec.from_file(str(XML))
    tool0 = sp.body("tool0")
    # board: OpenCV board frame (x along columns, y along rows, z into the board), front face at z = 0
    T_EP = np.array(he["T_EP"])
    b = tool0.add_body(name="calib_board", pos=T_EP[:3, 3] / 1000, quat=quat_of(T_EP[:3, :3]))
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.068, 0.063, 0.0015], pos=[0.0525, 0.0525, 0.0017],
               rgba=[0.97, 0.97, 0.95, 1], group=2)
    img, half = board_image()
    tex = sp.add_texture(name="charuco_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, width=img.shape[1],
                         height=img.shape[0], nchannel=3)
    tex.data = img.tobytes()
    mat = sp.add_material(name="charuco_mat")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "charuco_tex"
    lo, hi, z0, z1 = (52.5 - half) / 1000, (52.5 + half) / 1000, -0.0003, -0.0001
    vert = [lo, lo, z0, hi, lo, z0, hi, hi, z0, lo, hi, z0, lo, lo, z1, hi, lo, z1, hi, hi, z1, lo, hi, z1]
    uv = [0, 0, 1, 0, 1, 1, 0, 1] * 2
    face = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
    sp.add_mesh(name="charuco_quad", uservert=vert, usertexcoord=uv, userface=face)
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="charuco_quad", material="charuco_mat", group=2,
               contype=0, conaffinity=0)
    # cameras: OpenCV frame (x right, y down, z forward) -> MuJoCo camera frame (x right, y up, looks along -z)
    flip = np.diag([1.0, -1.0, -1.0])
    for sn, c in he["cams"].items():
        T_CB = np.array(c["T_CB"])
        R_BC, t_BC = T_CB[:3, :3].T, -T_CB[:3, :3].T @ T_CB[:3, 3]
        cb = sp.worldbody.add_body(name=f"cam_{sn}", pos=t_BC / 1000, quat=quat_of(R_BC))
        cb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.045, 0.0125, 0.0125], pos=[0.0, 0.0, -0.014],
                    rgba=list(COL[sn][:3]) + [1.0], group=2)
        l = lens[sn]["lens"]
        cam = cb.add_camera(name=f"c{sn}", quat=quat_of(flip))
        cam.resolution = [1920, 1080]
        cam.focal_pixel = [l["fx"], l["fy"]]
        cam.principal_pixel = [959.5 - l["cx"], 539.5 - l["cy"]]   # MuJoCo: offset of the centre from the principal point (measured, 11-13)
        cam.sensor_size = [1920 * 1e-6, 1080 * 1e-6]
    # hex block on its stand, and the table
    Bk = np.array(bt["T_B_block"])
    kb = sp.worldbody.add_body(name="hex_block", pos=Bk[:3, 3] / 1000, quat=quat_of(Bk[:3, :3]))
    r = bt["across_flats"] / np.sqrt(3) / 1000
    ang = np.radians(30 + 60 * np.arange(6))
    ring = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
    verts = np.concatenate([np.c_[ring, np.full(6, -bt["height"] / 2000)], np.c_[ring, np.full(6, bt["height"] / 2000)]])
    sp.add_mesh(name="hex_mesh", uservert=verts.ravel().tolist())
    kb.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hex_mesh", rgba=[0.93, 0.93, 0.93, 1], group=2)
    zt = bt["table_z_mm"] / 1000
    sp.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.009, (Bk[2, 3] / 1000 - bt["height"] / 2000 - zt) / 2],
                          pos=[Bk[0, 3] / 1000, Bk[1, 3] / 1000, (Bk[2, 3] / 1000 - bt["height"] / 2000 + zt) / 2],
                          rgba=[0.9, 0.9, 0.9, 1], group=2)
    sp.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[1.2, 1.0, 0.01], pos=[-0.25, -0.5, zt - 0.01],
                          rgba=[0.72, 0.6, 0.45, 1], group=2)
    m = sp.compile()
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1440
    return m, lens, he


def board_image(px_mm=16, margin_mm=11.5):
    """The board's ChArUco pattern (RGB, row 0 at the board's y = -margin) and the half width of the drawn square, mm."""
    import cv2
    A = cv2.aruco
    board = A.CharucoBoard((7, 7), SQ, 0.7 * SQ, A.getPredefinedDictionary(A.DICT_5X5_100))
    n = int(7 * SQ * px_mm)
    pat = board.generateImage((n, n), marginSize=0, borderBits=1)
    m = int(margin_mm * px_mm)
    pat = cv2.copyMakeBorder(pat, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
    return np.ascontiguousarray(np.repeat(pat[:, :, None], 3, 2)), (7 * SQ / 2 + margin_mm)


def hand_qpos():
    fr = j(ROOT / "assets/cad_v3/frames.json")["fingers"]
    out = {}
    for f, (x, y) in HOME_PALM_MM.items():
        cx, cy = fr[f]["box_centre_P_mm"][:2]
        out[f"{f}_x"], out[f"{f}_y"], out[f"{f}_bridge"] = (x - cx) / 1000, (y - cy) / 1000, (y - cy) / 1000
        s = SERVO_DEG[f]
        out[f"{f}_yaw"] = np.radians(YAW_SIGN[f] * s["yaw"])
        out[f"{f}_mcp"], out[f"{f}_pip"] = np.radians(s["mcp"]), np.radians(s["pip"])
    return out


def set_arm(m, d, q):
    for n, v in zip(ARM, q):
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = v
    for n, v in hand_qpos().items():
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = v
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)


def pyramids(scn, m, d, lens, depth=0.36):
    """Each camera's viewing pyramid to `depth` m, drawn as lines into the scene."""
    for sn, c in lens.items():
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"cam_{sn}")
        R, t = d.xmat[i].reshape(3, 3), d.xpos[i]
        l = c["lens"]
        cs = [((u - l["cx"]) / l["fx"] * depth, (v - l["cy"]) / l["fy"] * depth, depth)
              for u, v in ((0, 0), (1919, 0), (1919, 1079), (0, 1079))]
        pts = [t + R @ np.array(p) for p in cs]
        segs = [(t, p) for p in pts] + [(pts[k], pts[(k + 1) % 4]) for k in range(4)]
        for a, b in segs:
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                np.array(COL[sn], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 3.0, np.array(a, float), np.array(b, float))
            scn.ngeom += 1


def renderer(m, w, h):
    r = mujoco.Renderer(m, h, w, max_geom=20000)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [0, 1, 1, 0, 0, 0]
    return r, opt


def overview_cam():
    c = mujoco.MjvCamera()
    c.lookat[:] = [-0.25, -0.52, 0.12]                          # behind the two cameras, looking along +y at the hand
    c.distance, c.azimuth, c.elevation = 0.95, 90.0, -24.0
    return c


def film():
    m, lens, he = build()
    d = mujoco.MjData(m)
    recs = [json.loads(l) for l in open(DATA / "poses.jsonl")]
    Q = np.array([r["arm_before"]["q"] for r in recs])
    big, opt = renderer(m, 1280, 720)
    small, _ = renderer(m, 640, 360)
    oc = overview_cam()
    frames_dir = MEDIA / "tmp_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    fps, seg, hold = 30, 1.2, 0.6
    k = 0
    for i in range(len(Q)):
        a, b = Q[max(i - 1, 0)], Q[i]
        n_move = int(seg * fps) if i else 0
        for f in range(n_move + int(hold * fps)):
            s = min(f / max(n_move, 1), 1.0) if n_move else 1.0
            s = s * s * (3 - 2 * s)
            set_arm(m, d, a + s * (b - a))
            big.update_scene(d, oc, scene_option=opt)
            pyramids(big.scene, m, d, lens)
            left = big.render()
            right = []
            for sn in sorted(lens):
                small.update_scene(d, f"c{sn}", scene_option=opt)
                right.append(small.render())
            img = np.concatenate([left, np.concatenate(right, 0)], 1)
            import cv2
            cv2.putText(img, f"pose {i}: {recs[i]['label']}", (16, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            for kk, sn in enumerate(sorted(lens)):
                cv2.putText(img, sn, (1296, 30 + 360 * kk), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            tuple(int(255 * v) for v in COL[sn][:3]), 2)
            cv2.imwrite(str(frames_dir / f"f{k:05d}.jpg"), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            if i == 0 and f == 0:
                cv2.imwrite(str(MEDIA / "20261113-calibration_replay_poster.jpg"), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
            k += 1
    out = MEDIA / "20261113-calibration_replay.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "f%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", "-preset", "slow", "-movflags", "+faststart",
                    str(out)], check=True)
    for p in frames_dir.iterdir():
        p.unlink()
    frames_dir.rmdir()
    print(out, f"{out.stat().st_size / 1e6:.1f} MB, {k} frames")


BLUR = {"836612071918": (0, 860, 0, 780)}                     # x0, x1, y0, y1 of the region blurred (the owner)


def charuco_corners(gray):
    """{id: (u, v)} of the board's ChArUco corners in an image."""
    import cv2
    A = cv2.aruco
    board = A.CharucoBoard((7, 7), SQ, 0.7 * SQ, A.getPredefinedDictionary(A.DICT_5X5_100))
    cc, ci, _, _ = A.CharucoDetector(board).detectBoard(gray)
    return {} if ci is None else {int(a): b for a, b in zip(ci.ravel(), cc.reshape(-1, 2))}


def overlay(poses=(0, 3, 12)):
    import cv2
    m, lens, he = build()
    d = mujoco.MjData(m)
    recs = [json.loads(l) for l in open(DATA / "poses.jsonl")]
    r, opt = renderer(m, 1920, 1080)
    rows = []
    for rec in recs:
        p = rec["pose"]
        set_arm(m, d, rec["arm_before"]["q"])
        tiles = []
        for sn in sorted(lens):
            l = lens[sn]["lens"]
            K = np.array([[l["fx"], 0, l["cx"]], [0, l["fy"], l["cy"]], [0, 0, 1.0]])
            Dd = np.array([l["k1"], l["k2"], l["p1"], l["p2"], l["k3"]])
            real = cv2.imread(str(LOGS / f"p{p:02d}" / f"{sn}_color.png"))
            und = cv2.undistort(real, K, Dd, None, K)
            r.update_scene(d, f"c{sn}", scene_option=opt)
            ren = np.ascontiguousarray(r.render()[:, :, ::-1])
            cr = charuco_corners(cv2.cvtColor(ren, cv2.COLOR_BGR2GRAY))
            cu = charuco_corners(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY))
            ids = sorted(set(cr) & set(cu))
            e = np.array([cr[i] - cu[i] for i in ids]).reshape(-1, 2)
            rows.append(dict(pose=p, cam=sn, n=len(ids), rms_px=float(np.sqrt((e ** 2).sum(1).mean())) if len(ids) else None,
                             mean_px=e.mean(0).tolist() if len(ids) else None))
            if p in poses:
                if sn in BLUR:
                    x0, x1, y0, y1 = BLUR[sn]
                    und[y0:y1, x0:x1] = cv2.GaussianBlur(cv2.resize(cv2.resize(und[y0:y1, x0:x1], None, fx=1 / 60, fy=1 / 60),
                                                                    (x1 - x0, y1 - y0)), (0, 0), 25)
                tiles.append(cv2.addWeighted(und, 0.5, ren, 0.5, 0))
        if tiles:
            img = np.concatenate([cv2.resize(t, (960, 540)) for t in tiles], 1)
            cv2.imwrite(str(MEDIA / f"20261113-overlay_p{p:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (FITS / "20261113-render_overlap.json").write_text(json.dumps(dict(
        what="ChArUco corners in the render from each fitted camera minus those in the undistorted frame, px",
        hand=dict(home_palm_mm=HOME_PALM_MM, servo_deg=SERVO_DEG), rows=rows), indent=1))
    for sn in sorted(lens):
        v = [x["rms_px"] for x in rows if x["cam"] == sn and x["rms_px"] is not None]
        mu = np.array([x["mean_px"] for x in rows if x["cam"] == sn and x["mean_px"] is not None])
        print(f"{sn}: render vs frame corners RMS {np.median(v):.2f} px median over {len(v)} poses (max {max(v):.2f}); "
              f"mean offset {np.round(mu.mean(0), 2).tolist()} px")


if __name__ == "__main__":
    MEDIA.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "film"
    if what in ("overlay", "all"):
        overlay()
    if what in ("film", "all"):
        film()
