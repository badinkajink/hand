#!/usr/bin/env python3
"""Film and overlay stills of the D435 camera calibration of 2026-11-13 in the cad_v3 MuJoCo scene of the SR2 hand on
the UR5 (assets/mjcf/experimental/20261104-cad_v3/ur5_full_k1.xml, the hand as built). Every frame is kinematic:
qpos is set and mj_kinematics run, with contacts off.

Added to the scene, from docs/experiments/20261113-camera_arm_calibration/data/fits/: the paper ChArUco board on tool0
(the hand-eye fit through ur_kinematics.fk, the model's own kinematics), both cameras at their fitted poses in the base
with their fitted focal lengths and principal points, and the hex block and the table from
20261113-block_and_table.json (camera_arm_calibration_page.py block). The hand's joints stay at the model's zero.

    film     the arm through the 17 recorded poses in their order, the overview with both cameras' viewing pyramids on
             the left and the two cameras' renders on the right -> media/20261113-calibration_replay.mp4 and a poster
    overlay  poses 0, 3 and 12: each camera's frame (logs/, corvus only), undistorted with its fitted lens, blended
             with the render from the fitted camera -> media/20261113-overlay_pNN.jpg. The owner sat in view of
             836612071918, left of x = 860 px: that part of its frames is blurred.

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
    for r in range(7):
        for c in range(7):
            if (r + c) % 2 == 0:
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[SQ / 2000, SQ / 2000, 0.0001],
                           pos=[(c + 0.5) * SQ / 1000, (r + 0.5) * SQ / 1000, -0.0001], rgba=[0.05, 0.05, 0.05, 1], group=2)
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
        cam.principal_pixel = [l["cx"] - 959.5, 539.5 - l["cy"]]
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


def set_arm(m, d, q):
    for n, v in zip(ARM, q):
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


def overlay(poses=(0, 3, 12)):
    import cv2
    m, lens, he = build()
    d = mujoco.MjData(m)
    recs = [json.loads(l) for l in open(DATA / "poses.jsonl")]
    r, opt = renderer(m, 1920, 1080)
    for p in poses:
        set_arm(m, d, recs[p]["arm_before"]["q"])
        tiles = []
        for sn in sorted(lens):
            l = lens[sn]["lens"]
            K = np.array([[l["fx"], 0, l["cx"]], [0, l["fy"], l["cy"]], [0, 0, 1.0]])
            Dd = np.array([l["k1"], l["k2"], l["p1"], l["p2"], l["k3"]])
            real = cv2.imread(str(LOGS / f"p{p:02d}" / f"{sn}_color.png"))
            if sn in BLUR:
                x0, x1, y0, y1 = BLUR[sn]
                real[y0:y1, x0:x1] = cv2.GaussianBlur(cv2.resize(cv2.resize(real[y0:y1, x0:x1], None, fx=1 / 60, fy=1 / 60),
                                                                 (x1 - x0, y1 - y0)), (0, 0), 25)
            und = cv2.undistort(real, K, Dd, None, K)
            r.update_scene(d, f"c{sn}", scene_option=opt)
            ren = r.render()[:, :, ::-1]
            tiles.append(cv2.addWeighted(und, 0.5, ren, 0.5, 0))
        img = np.concatenate([cv2.resize(t, (960, 540)) for t in tiles], 1)
        cv2.imwrite(str(MEDIA / f"20261113-overlay_p{p:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(MEDIA / f"20261113-overlay_p{p:02d}.jpg")


if __name__ == "__main__":
    MEDIA.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "film"
    if what in ("overlay", "all"):
        overlay()
    if what in ("film", "all"):
        film()
