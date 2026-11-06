#!/usr/bin/env python3
"""Steps 1-3 of the object seating and tracking plan (docs/notes/20261105-object_seating_and_tracking/) in simulation,
the camera side. Images are rendered with MuJoCo and passed through a model of the D435's colour sensor, and the tags
are found by pupil_apriltags and solved by src/sr2hand/tracking, the code meant for the bench. Run with
~/miniconda3/bin/python (MuJoCo 3.4 for rendering, OpenCV 4.10, pupil_apriltags); the HOM rollouts come from
scripts/hardware/object_tracking_rollouts.py in the repository's .venv.

Camera. The D435's colour stream as a pinhole (sr2hand.tracking.pose.d435_colour: fx 1397 px at 1920 x 1080, scaled at
1280 x 720). Each image is rendered at twice the resolution with 4x multisampling and averaged over 2 x 2 pixels, lit by
a light at the camera (the plan's LED panel beside each camera: ambient 0.25, diffuse 0.65), with matte materials
(white 0.92, black 0.03).
Sensor. Gaussian blur of SENSOR.blur_px (lens and demosaicing), an RGGB mosaic, shot noise of SENSOR.e_per_dn electrons
per digital number and read noise of SENSOR.read_dn, 8-bit quantization and OpenCV's demosaicing to grey. A white
face seen head-on is about 200 of 255.

  tagsize     step 1: a plate with tag36h11 and tag25h9 at 10, 12, 13, 14 and 16 mm, at 0.25, 0.3, 0.4 and 0.5 m,
              0-70 deg from the normal, 1080p and 720p; 6 poses jittered by 2 mm and 1 deg per condition and 8 noise
              draws per pose. Per tag: the share of frames decoded, the corner error against the projection of the
              true corners (RMS, px) and the corner jitter (the spread over the noise draws at one pose, RMS, px).
Rows: docs/experiments/20261105-object_tracking_sim/rows/<command>.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for _paths
import _paths  # noqa: E402,F401
import object_tracking_mjcf as OM  # noqa: E402
from sr2hand.geometry.rotations import mat_to_quat, rot_axis_angle, tf, tf_apply, tf_inv  # noqa: E402
from sr2hand.tracking import pose as TP  # noqa: E402
from sr2hand.tracking import tags as TG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/experiments/20261105-object_tracking_sim"
ROWS = OUT / "rows"
MEDIA = OUT / "media"
LOGS = ROOT / "logs/20261105-object_tracking_sim"
SS = 2                                   # supersampling of the render
GROUP_HIDE = 5                       # geoms moved here are neither drawn nor hit by rays (floor, the study's post)
GROUP_PADS = 4                       # pads and collision spheres: inside the fingers' visual meshes, not drawn


@dataclass
class Sensor:
    blur_px: float = 0.6
    e_per_dn: float = 10.0
    read_dn: float = 1.5

    def apply(self, rgb, rng):
        """uint8 grey image from a float RGB image in [0, 255]."""
        img = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), self.blur_px) if self.blur_px > 0 else rgb
        h, w = img.shape[:2]
        mos = np.empty((h, w), np.float32)
        mos[0::2, 0::2] = img[0::2, 0::2, 0]
        mos[0::2, 1::2] = img[0::2, 1::2, 1]
        mos[1::2, 0::2] = img[1::2, 0::2, 1]
        mos[1::2, 1::2] = img[1::2, 1::2, 2]
        sd = np.sqrt(np.maximum(mos, 0.0) / self.e_per_dn + self.read_dn ** 2)
        mos = np.clip(np.round(mos + rng.standard_normal(mos.shape).astype(np.float32) * sd), 0, 255).astype(np.uint8)
        return cv2.cvtColor(mos, cv2.COLOR_BayerRG2GRAY)


SENSOR = Sensor()


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """OpenCV camera pose T_WC (x right, y down, z forward) at eye looking at target (m)."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return tf(np.stack([x, y, z], 1), eye)


class Renderer:
    """Renders an MjModel through its camera 'otrack_cam' at a given OpenCV pose and pinhole, supersampled."""

    def __init__(self, m, cam: TP.Camera, ss=SS):
        import mujoco
        self.mj, self.m, self.cam, self.ss = mujoco, m, cam, ss
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, cam.width * ss)
        m.vis.global_.offheight = max(m.vis.global_.offheight, cam.height * ss)
        m.vis.quality.offsamples = 4
        m.vis.headlight.ambient[:] = 0.25
        m.vis.headlight.diffuse[:] = 0.65
        m.vis.headlight.specular[:] = 0.0
        m.vis.headlight.active = 1
        if m.nlight:
            m.light_active[:] = 0
        m.vis.map.znear = 0.002 / max(m.stat.extent, 1e-6)
        m.vis.map.zfar = 5.0 / max(m.stat.extent, 1e-6)
        self.cid = m.camera("otrack_cam").id
        m.cam_fovy[self.cid] = math.degrees(2 * math.atan((cam.height / 2.0) / cam.K[1, 1]))
        self.r = mujoco.Renderer(m, cam.height * ss, cam.width * ss)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 1
        self.opt.geomgroup[:] = 1
        self.opt.geomgroup[GROUP_PADS] = self.opt.geomgroup[GROUP_HIDE] = 0   # a camera sees the visual meshes

    def render(self, d, T_WC):
        m = self.m
        R = T_WC[:3, :3] @ np.diag([1.0, -1.0, -1.0])     # MuJoCo's camera looks along its -z with y up
        m.cam_pos[self.cid] = T_WC[:3, 3]
        m.cam_quat[self.cid] = mat_to_quat(R)
        self.mj.mj_kinematics(m, d)
        self.mj.mj_camlight(m, d)
        self.r.update_scene(d, camera=self.cid, scene_option=self.opt)
        img = self.r.render().astype(np.float32)
        if self.ss > 1:
            img = cv2.resize(img, (self.cam.width, self.cam.height), interpolation=cv2.INTER_AREA)
        return img

    def close(self):
        self.r.close()


def add_camera(root):
    ET.SubElement(root.find("worldbody"), "camera", name="otrack_cam", pos="0 0 1", quat="1 0 0 0", fovy="42")


def project_world(cam, T_WC, P_W_m):
    """Pixels of world points (m) in a camera at T_WC (m)."""
    T = tf_inv(T_WC)
    T_mm = T.copy()
    T_mm[:3, 3] *= 1000.0
    return TP.project(cam, T_mm, np.asarray(P_W_m) * 1000.0)


# ------------------------------------------------------------------------------------------------ step 1: tag size

PLATE_SIZES = (10.0, 12.0, 13.0, 14.0, 16.0)
PLATE_FAMILIES = ("tag36h11", "tag25h9")
PLATE_PITCH = 30.0


def plate_maps():
    """{family: TagMap} of the test plate in its frame (mm): z out of the plate toward the camera, x right, y up; ids
    0-4 of each family left to right in PLATE_SIZES, tag36h11 in the upper row."""
    out = {}
    for row, fam in enumerate(PLATE_FAMILIES):
        tm = TG.TagMap(fam, PLATE_SIZES[0])
        for k, s in enumerate(PLATE_SIZES):
            one = TG.plate_map(fam, s, [k], [np.array([(k - 2) * PLATE_PITCH, (0.5 - row) * PLATE_PITCH, 0.0])],
                               normal=(0, 0, 1), right=(1, 0, 0))
            tm.corners[k], tm.frames[k] = one.corners[k], one.frames[k]
        out[fam] = tm
    return out


def plate_model():
    import mujoco
    root = ET.fromstring('<mujoco model="tag_plate"><asset/><worldbody/></mujoco>')
    mat = OM.add_matte(root)
    wb = root.find("worldbody")
    body = ET.SubElement(wb, "body", name="plate", pos="0 0 0")
    ET.SubElement(body, "geom", type="box", size=f"{0.085:.4f} {0.035:.4f} 0.0015", pos="0 0 -0.0015",
                  rgba="0.92 0.92 0.92 1", material=mat)
    maps = plate_maps()
    for fam, tm in maps.items():
        # each tag of the family at its own size
        for k, s in enumerate(PLATE_SIZES):
            one = TG.TagMap(fam, s)
            one.corners[k], one.frames[k] = tm.corners[k], tm.frames[k]
            OM.add_tag_sites(body, one, f"{fam}_", material=mat)
    add_camera(root)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), maps


def cmd_tagsize(a):
    import mujoco
    m, maps = plate_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    dets = {fam: TP.detector(fam) for fam in PLATE_FAMILIES}
    rng = np.random.default_rng(0)
    rows = []
    t0 = time.time()
    for res in a.res:
        cam = TP.d435_colour(*{1080: (1920, 1080), 720: (1280, 720)}[res])
        rd = Renderer(m, cam)
        for dist in a.dist:
            for ang in a.angles:
                acc = {}
                for k_pose in range(a.poses):
                    j = rng.uniform(-1, 1, 3) * 0.002 if k_pose else np.zeros(3)
                    ja = math.radians(ang + (rng.uniform(-1, 1) if k_pose else 0.0))
                    target = j
                    eye = target + dist * np.array([math.sin(ja), 0.0, math.cos(ja)])
                    up = rot_axis_angle([0, 0, 1], math.radians(rng.uniform(-1, 1) if k_pose else 0.0)) @ [0, 1, 0]
                    T_WC = look_at(eye, target, up)
                    img = rd.render(d, T_WC)
                    truth = {fam: {i: project_world(cam, T_WC, tm.corners[i] / 1000.0) for i in tm.ids()}
                             for fam, tm in maps.items()}
                    vis = {fam: {i: (float(np.linalg.norm(eye - tm.frames[i][0] / 1000.0)),
                                     float(np.degrees(np.arccos(np.clip(
                                         (eye - tm.frames[i][0] / 1000.0) @ tm.frames[i][3]
                                         / np.linalg.norm(eye - tm.frames[i][0] / 1000.0), -1, 1)))))
                                 for i in tm.ids()} for fam, tm in maps.items()}
                    for k_n in range(a.noise):
                        g = SENSOR.apply(img, rng)
                        for fam in PLATE_FAMILIES:
                            found = TP.detect(dets[fam], g)
                            for i in maps[fam].ids():
                                e = acc.setdefault((fam, i), {"found": 0, "n": 0, "err": [], "per_pose": {},
                                                              "dist": [], "angle": []})
                                e["n"] += 1
                                if k_n == 0:
                                    e["dist"].append(vis[fam][i][0])
                                    e["angle"].append(vis[fam][i][1])
                                if i in found:
                                    e["found"] += 1
                                    err = found[i] - truth[fam][i]
                                    e["err"].append(err)
                                    e["per_pose"].setdefault(k_pose, []).append(found[i])
                for (fam, i), e in acc.items():
                    jit = [np.std(np.stack(v), axis=0, ddof=1) for v in e["per_pose"].values() if len(v) >= 3]
                    E = np.concatenate(e["err"]) if e["err"] else np.zeros((0, 2))
                    rows.append({"res": res, "dist_m": dist, "angle_deg": ang, "family": fam,
                                 "size_mm": PLATE_SIZES[i], "frames": e["n"], "decoded": e["found"],
                                 "rate": round(e["found"] / e["n"], 4),
                                 "tag_dist_m": round(float(np.mean(e["dist"])), 4),
                                 "tag_angle_deg": round(float(np.mean(e["angle"])), 2),
                                 "px_per_cell": round(cam.fx * PLATE_SIZES[i] / TG.n_cells(fam) / (1000 * np.mean(e["dist"])), 2),
                                 "corner_err_px": round(float(np.sqrt((E ** 2).sum(1).mean())), 4) if len(E) else None,
                                 "corner_jitter_px": round(float(np.sqrt(np.mean(np.concatenate(jit) ** 2))), 4)
                                 if jit else None})
                print(f"{res}p {dist:.2f} m {ang:4.0f} deg  " + "  ".join(
                    f"{r['family'][3:]}/{r['size_mm']:g}:{r['rate']:.2f}" for r in rows[-10:]),
                    f"{time.time() - t0:.0f} s", flush=True)
        rd.close()
    ROWS.mkdir(parents=True, exist_ok=True)
    out = {"what": "step 1, tag size test in simulation", "sensor": vars(SENSOR), "poses": a.poses, "noise": a.noise,
           "camera_fx_1080": 1397.0, "rows": rows}
    (ROWS / "tagsize.json").write_text(json.dumps(out, indent=0))
    print("wrote", ROWS / "tagsize.json")
    return 0


# ------------------------------------------------------------------------------------------------ step 2: cameras

ROLLOUT_ROWS = ROWS / "rollouts.jsonl"
RAY_GROUPS = np.array([1, 1, 1, 0, 0, 0], np.uint8)
CAM_DIST = 0.30                      # m from the block's centre in the task (plan)
CAM_AZ = np.arange(0.0, 360.0, 10.0)
CAM_EL = (0.0, 10.0, 20.0, 30.0, 40.0)
MAX_VIEW_DEG = 60.0
EDGE_PX = 10.0
HAND_TAG_MM = 40.0
HAND_TAG_R, HAND_TAG_Z = 110.0, 44.0  # mm in P: on the hand's outline at the height of the tag mount's tag 6


def load_traj(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def hide_walls(m):
    """The pocket's walls hidden and out of the rays: the flat seat is the pocket seat's floor alone."""
    for g in range(m.ngeom):
        if (m.geom(g).name or "").startswith("otrack_wall"):
            m.geom_group[g] = GROUP_HIDE
            m.geom_rgba[g, 3] = 0.0


def scene_model(tr, flat=True):
    """The rollout's scene in this process's MuJoCo, posed for its layout, with the camera; the floor and the study's
    post hidden (and the pocket's walls for a flat seat), the pads and collision spheres moved out of the rays'
    groups."""
    import mujoco
    spec = mujoco.MjSpec.from_file(str(ROOT / str(tr["scene"])))
    spec.worldbody.add_camera(name="otrack_cam", pos=[0, 0, 1], quat=[1, 0, 0, 0], fovy=42.0)
    m = spec.compile()
    for k, f in enumerate(("thumb", "index", "middle")):
        m.body_pos[m.body(f"{f}_mount").id] = tr["mounts"][k]
    m.body_pos[m.body("palm_pose").id] = tr["palm"]
    for g in range(m.ngeom):
        nm = m.geom(g).name or ""
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or nm == "post":
            m.geom_group[g] = GROUP_HIDE
            m.geom_rgba[g, 3] = 0.0
        elif m.geom_group[g] in (0, 3) and m.body(m.geom_bodyid[g]).name.split("_")[0] in ("thumb", "index", "middle"):
            m.geom_group[g] = GROUP_PADS
    if flat:
        hide_walls(m)
    return m


def set_frame(m, d, tr, i):
    import mujoco
    d.qpos[:] = tr["qpos"][i]
    if "seat" in tr and len(tr["seat"]):
        mid = m.body_mocapid[m.body("otrack_seat").id]
        d.mocap_pos[mid] = tr["seat"][i][:3]
        d.mocap_quat[mid] = tr["seat"][i][3:]
    mujoco.mj_kinematics(m, d)


def table_offset(tr, i):
    """The table's displacement in the hand's frame (m): the seat's, which the arm's lift moves down."""
    return tr["seat"][i][:3] - tr["seat"][0][:3]


def object_T(tr, i):
    """T_WO (m) from the free joint, the first joint of the scenes."""
    from sr2hand.geometry.rotations import quat_to_mat
    q = tr["qpos"][i]
    return tf(quat_to_mat(q[3:7]), q[:3])


def block_tags_world(tr, i, shape):
    """The block's TagMap in the world (m) at record i."""
    T = object_T(tr, i) @ np.diag([1.0, 1.0, 1.0, 1.0])
    T_OB = OM.T_OB(float(tr["r_mm"]), float(tr["stem_mm"]))
    T_mm = T.copy()
    T_mm[:3, 3] *= 1000.0
    return OM.block_map(shape).transformed(T_mm @ T_OB)


def tag_samples(tm, tid, inset=0.4, lift=0.1):
    """(9, 3) points of a tag (mm, its map's frame): corners, edge middles and centre, inset along the tag and lifted
    off the face."""
    c, r, dn, n = tm.frames[tid]
    h = tm.size / 2.0 - inset
    pts = [c + a * h * r + b * h * dn for a in (-1, 0, 1) for b in (-1, 0, 1)]
    return np.array(pts) + lift * n


def hand_tag(palm_m, cam_pos_m, az_deg):
    """The camera's 40 mm hand tag (TagMap in the world, mm): on the hand's outline toward the camera's azimuth at
    HAND_TAG_Z in P, facing the camera."""
    a = math.radians(az_deg)
    c = np.asarray(palm_m) * 1000.0 + np.array([HAND_TAG_R * math.cos(a), HAND_TAG_R * math.sin(a), HAND_TAG_Z])
    n = np.asarray(cam_pos_m) * 1000.0 - c
    n /= np.linalg.norm(n)
    down = np.array([0.0, 0.0, -1.0]) - n * (-n[2])
    down /= np.linalg.norm(down)
    right = np.cross(n, down)
    tm = TG.TagMap("tag36h11", HAND_TAG_MM)
    tm.add(7, c, right, down)
    return tm


def portrait():
    """The D435's colour stream turned on its side: 1080 x 1920, 42.2 deg across and 69.0 deg high."""
    c = TP.d435_colour(1920, 1080)
    K = np.array([[c.K[0, 0], 0.0, (1080 - 1) / 2.0], [0.0, c.K[1, 1], (1920 - 1) / 2.0], [0.0, 0.0, 1.0]])
    return TP.Camera(K, np.zeros(5), 1080, 1920)


def camera_pose(az, el, target_m, look_m):
    """T_WC (m) of a candidate in the table's frame: CAM_DIST from target at azimuth az and elevation el (deg) above
    it, looking at look_m."""
    a, e = math.radians(az), math.radians(el)
    eye = np.asarray(target_m) + CAM_DIST * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return look_at(eye, look_m)


def visible(m, d, cam, T_WC, tm, ids, max_view=MAX_VIEW_DEG):
    """Bitmask of the tags of tm (mm, world) seen whole by the camera: within max_view of their normal, every sample
    point inside the image by EDGE_PX and no geom of the ray groups in front of it."""
    import mujoco
    eye = T_WC[:3, 3]
    pts, owner = [], []
    good = {}
    for k, tid in enumerate(ids):
        c, _r, _dn, n = tm.frames[tid]
        v = eye * 1000.0 - c
        if np.degrees(np.arccos(np.clip(v @ n / np.linalg.norm(v), -1, 1))) > max_view:
            continue
        P = tag_samples(tm, tid) / 1000.0
        uv = project_world(cam, T_WC, P)
        if (uv[:, 0] < EDGE_PX).any() or (uv[:, 1] < EDGE_PX).any() or (uv[:, 0] > cam.width - 1 - EDGE_PX).any() \
                or (uv[:, 1] > cam.height - 1 - EDGE_PX).any():
            continue
        if (tf_apply(tf_inv(T_WC), P)[:, 2] <= 0).any():
            continue
        good[k] = True
        pts.append(P)
        owner += [k] * len(P)
    if not pts:
        return 0
    P = np.concatenate(pts)
    vec = P - eye
    L = np.linalg.norm(vec, axis=1)
    vec = (vec / L[:, None]).ravel()
    gid = np.zeros(len(P), np.int32)
    dist = np.zeros(len(P))
    mujoco.mj_multiRay(m, d, eye, vec, RAY_GROUPS, 1, -1, gid, dist, len(P), 10.0)
    hit = (dist >= 0) & (dist < L - 2e-4)
    mask = 0
    owner = np.array(owner)
    for k in good:
        if not hit[owner == k].any():
            mask |= 1 << k
    return mask


def flat_rows(config):
    rows = [json.loads(x) for x in ROLLOUT_ROWS.read_text().splitlines() if x.strip()]
    return [r for r in rows if r["config"] == config and r.get("status") == "ok"]


def cmd_visibility(a):
    """For every rollout of a.config, every candidate camera (CAM_AZ x CAM_EL, CAM_DIST from the block's centre in the
    task, table-fixed, so the arm's lift moves the hand past it) and every frame at a.hz: the block's tags seen whole
    (bitmask), and the candidate's hand tag seen whole. Rows: logs/.../visibility_<config>_part<k>of<n>.npz; the chosen
    pair in rows/visibility_<config>.json (cmd_pairs)."""
    import mujoco
    rows = flat_rows(a.config)[a.part::a.parts]
    cam = portrait()
    cands = [(az, el) for el in CAM_EL for az in CAM_AZ]
    masks, hands, meta = [], [], []
    t0 = time.time()
    models = {}
    for k, r in enumerate(rows):
        tr = load_traj(ROOT / r["traj"])
        shape = "sphere" if r["task"] in ("roll", "twist") else "cylinder"
        key = str(tr["scene"])
        m = scene_model(tr, flat=a.config.startswith("flat"))
        d = mujoco.MjData(m)
        lift = float(r.get("lift_mm", 40.0)) / 1000.0
        T0 = block_tags_world(tr, 0, shape)
        b0 = np.mean([T0.frames[i][0] for i in T0.ids()], 0) / 1000.0
        target = np.array([0.0, 0.0, 0.065 + lift])            # the blocks' centre in the task, in the table frame
        seated = np.array([0.0, 0.0, 0.065])
        t = tr["t"]
        step = max(1, int(round(1.0 / a.hz / (t[1] - t[0]))))
        idx = np.arange(0, len(t), step)
        M = np.zeros((len(idx), len(cands)), np.uint8)
        Hh = np.zeros((len(idx), len(cands)), bool)
        for j, i in enumerate(idx):
            set_frame(m, d, tr, i)
            off = table_offset(tr, i)
            tm = block_tags_world(tr, i, shape)
            ids = tm.ids()
            for c, (az, el) in enumerate(cands):
                hand0 = np.asarray(tr["palm"]) + np.array([HAND_TAG_R * math.cos(math.radians(az)),
                                                           HAND_TAG_R * math.sin(math.radians(az)), HAND_TAG_Z]) / 1000.0
                # aimed at the middle of the seated block and the hand tag in the task, in the table's frame
                T_WC = camera_pose(az, el, target, 0.5 * (seated + hand0 + np.array([0.0, 0.0, lift])))
                T_WC[:3, 3] += off
                M[j, c] = visible(m, d, cam, T_WC, tm, ids)
                ht = hand_tag(tr["palm"], T_WC[:3, 3], az)
                Hh[j, c] = visible(m, d, cam, T_WC, ht, [7], max_view=75.0) == 1
        masks.append(M)
        hands.append(Hh)
        meta.append({"task": r["task"], "call": r["call"], "role": r["role"], "seed": r["seed"],
                     "t": t[idx].round(3).tolist(), "block0_m": b0.round(5).tolist()})
        print(f"{k + 1}/{len(rows)} {r['task']} {r['role']} s{r['seed']}: frames {len(idx)}, "
              f"best single camera {np.mean(np.array([bin(x).count('1') for x in M.ravel()]).reshape(M.shape) >= 2, 0).max():.2f}"
              f"  {time.time() - t0:.0f} s", flush=True)
    out = LOGS / f"visibility_{a.config}_part{a.part}of{a.parts}.npz"
    np.savez_compressed(out, masks=np.array(masks, dtype=object), hands=np.array(hands, dtype=object),
                        cands=np.array(cands), meta=json.dumps(meta))
    print("wrote", out)
    return 0


POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)
MIN_SEP_DEG = 60.0


def cmd_pairs(a):
    """The two cameras from the visibility rows: over the task's frames (t >= 2.0 s) of every rollout, the share in
    which the pair sees at least two distinct block tags whole (the plan's criterion) and the share in which each
    camera sees its hand tag; pairs at least MIN_SEP_DEG apart in azimuth, ranked by the first share, then the
    second; a pair whose cameras both see their hand tags in 99 % of the task's frames ranks before any other (the
    plan: the hand's tags must be visible to both cameras). Also the seated frames (t < 1.6 s) and each candidate
    alone. Out: rows/visibility_<config>.json."""
    masks, hands, meta = [], [], []
    parts = sorted(LOGS.glob(f"visibility_{a.config}_part*of*.npz"))
    for f in parts:
        z = np.load(f, allow_pickle=True)
        masks += list(z["masks"])
        hands += list(z["hands"])
        meta += json.loads(str(z["meta"]))
        cands = z["cands"]
    print(len(parts), "parts,", len(meta), "rollouts")
    task_sel = [np.array(m_["t"]) >= 2.0 for m_ in meta]
    seat_sel = [np.array(m_["t"]) < 1.6 for m_ in meta]
    M = np.concatenate([mk[s_] for mk, s_ in zip(masks, task_sel)])          # (frames, cands)
    Hh = np.concatenate([h[s_] for h, s_ in zip(hands, task_sel)])
    Ms = np.concatenate([mk[s_] for mk, s_ in zip(masks, seat_sel)])
    rid = np.concatenate([np.full(int(s_.sum()), k) for k, s_ in enumerate(task_sel)])
    single = (POP[M] >= 2).mean(0)
    hand1 = Hh.mean(0)
    nc = len(cands)
    best = []
    for i in range(nc):
        for j in range(i + 1, nc):
            sep = abs((cands[i][0] - cands[j][0] + 180.0) % 360.0 - 180.0)
            if sep < MIN_SEP_DEG:
                continue
            cov = float((POP[M[:, i] | M[:, j]] >= 2).mean())
            hnd = float((Hh[:, i] & Hh[:, j]).mean())
            best.append((cov, hnd, i, j))
    best.sort(key=lambda x: (-(x[1] >= 0.99), -x[0], -x[1]))      # both hand tags seen first (the plan), then coverage
    cov, hnd, i, j = best[0]
    ok = POP[M[:, i] | M[:, j]] >= 2
    per = []
    for k, m_ in enumerate(meta):
        sel = rid == k
        per.append({**{x: m_[x] for x in ("task", "call", "role", "seed")}, "task_frames": int(sel.sum()),
                    "covered": float(ok[sel].mean()) if sel.any() else None,
                    "one_tag_or_more": float((POP[M[sel, i] | M[sel, j]] >= 1).mean()) if sel.any() else None})
    out = {"config": a.config, "criterion": "at least two distinct block tags seen whole by the pair (task frames)",
           "pair": [list(map(float, cands[i])), list(map(float, cands[j]))], "covered": cov, "hand_tags_both": hnd,
           "covered_seated": float((POP[Ms[:, i] | Ms[:, j]] >= 2).mean()),
           "single": [{"az": float(c[0]), "el": float(c[1]), "covered": float(single[k]), "hand": float(hand1[k]),
                       "covered_seated": float((POP[Ms[:, k]] >= 2).mean())} for k, c in enumerate(cands)],
           "top_pairs": [{"a": list(map(float, cands[x[2]])), "b": list(map(float, cands[x[3]])), "covered": x[0],
                          "hand_tags_both": x[1]} for x in best[:20]],
           "per_rollout": per, "frames_task": int(len(M)), "frames_seated": int(len(Ms))}
    (ROWS / f"visibility_{a.config}.json").write_text(json.dumps(out, indent=0))
    print("pair", out["pair"], f"covered {cov:.4f}, hand tags {hnd:.4f}, seated {out['covered_seated']:.4f}")
    for x in out["top_pairs"][:8]:
        print("  ", x)
    return 0


# ------------------------------------------------------------------------------------------------ step 3: turntable

PRINT_SD = {"in_plane_mm": 0.10, "normal_mm": 0.05, "turn_deg": 0.30, "tilt_deg": 0.20}   # assumed print errors per tag
BLOCK_H = 40.0                   # mm: the block's centre above the turntable along its axis
WEDGE_DEG = 20.0
READOUT_S, EXPOSURE_S = 0.025, 0.005    # assumed rolling-shutter readout of the frame and exposure of a row


def perturbed_map(tm, rng, sd=PRINT_SD):
    """The block as printed: each tag moved and turned on its face by the assumed print errors."""
    out = TG.TagMap(tm.family, tm.size)
    for i, (c, r, dn, n) in tm.frames.items():
        R = (rot_axis_angle(n, math.radians(rng.normal() * sd["turn_deg"]))
             @ rot_axis_angle(r, math.radians(rng.normal() * sd["tilt_deg"]))
             @ rot_axis_angle(dn, math.radians(rng.normal() * sd["tilt_deg"])))
        c2 = c + r * rng.normal() * sd["in_plane_mm"] + dn * rng.normal() * sd["in_plane_mm"] + n * rng.normal() * sd["normal_mm"]
        out.add(i, c2, R @ r, R @ dn)
    return out


def block_model(printed):
    """The tag block alone (mocap body 'block', its frame B) on a turntable disc."""
    import mujoco
    root = ET.fromstring('<mujoco model="tag_block"><asset/><worldbody/></mujoco>')
    mat = OM.add_matte(root)
    # the faces 0.2 mm inside the tags, so a tag moved or tilted by the print errors stays above its face
    OM.prism_mesh(root.find("asset"), "block", TG.HexBlock(across_flats=OM.BLOCK.across_flats - 0.4))
    wb = root.find("worldbody")
    b = ET.SubElement(wb, "body", name="block", mocap="true", pos="0 0 0.04")
    ET.SubElement(b, "geom", type="mesh", mesh="block", rgba="0.92 0.92 0.92 1", material=mat)
    ET.SubElement(b, "geom", type="cylinder", size="0.002 0.012", pos="0 0 -0.0205", rgba="0.85 0.85 0.85 1",
                  material=mat)
    OM.add_tag_sites(b, printed, "t", material=mat)
    ET.SubElement(wb, "geom", type="cylinder", size="0.06 0.005", pos="0 0 -0.005", rgba="0.35 0.35 0.38 1",
                  material=mat)
    add_camera(root)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def table_pose(theta_deg, wedge_deg):
    """T_WB (m) of the block on the turntable at angle theta, on a wedge of wedge_deg."""
    Rz = rot_axis_angle([0, 0, 1], math.radians(theta_deg))
    Rw = rot_axis_angle([1, 0, 0], math.radians(wedge_deg))
    return tf(Rz @ Rw, Rz @ Rw @ np.array([0.0, 0.0, BLOCK_H / 1000.0]))


def put_block(m, d, T_WB):
    import mujoco
    mid = m.body_mocapid[m.body("block").id]
    d.mocap_pos[mid] = T_WB[:3, 3]
    d.mocap_quat[mid] = mat_to_quat(T_WB[:3, :3])
    mujoco.mj_kinematics(m, d)


def T_mm(T_m):
    T = np.array(T_m, float).copy()
    T[:3, 3] *= 1000.0
    return T


def angle_about(R, axis):
    """Signed angle (deg) of rotation R about a unit axis (its rotation vector's component along the axis)."""
    from sr2hand.geometry.rotations import log_so3
    return math.degrees(float(log_so3(R) @ axis))


def circle_fit(P):
    """(centre, normal, radius, residuals) of points P (N, 3) fitted by a plane and a circle in it."""
    c0 = P.mean(0)
    _u, _s, Vt = np.linalg.svd(P - c0)
    n, e1, e2 = Vt[2], Vt[0], Vt[1]
    xy = np.stack([(P - c0) @ e1, (P - c0) @ e2], 1)
    A = np.c_[2 * xy, np.ones(len(xy))]
    sol, *_ = np.linalg.lstsq(A, (xy ** 2).sum(1), rcond=None)
    cx, cy = sol[:2]
    rad = math.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0))
    radial = np.linalg.norm(xy - [cx, cy], axis=1) - rad
    out_of_plane = (P - c0) @ n
    return c0 + cx * e1 + cy * e2, n, rad, np.sqrt(radial ** 2 + out_of_plane ** 2)


def bench_metrics(Ts, enc_deg, circle=True):
    """The bench's own scoring of a turntable run without ground truth: the axis fitted to the solved rotations, the
    solved angle about it against the encoder's (offset removed, RMS deg), and the block centre's distance from the
    circle fitted to it (RMS mm); on the level table (circle False) the centre stays on the axis, and its scatter about
    the mean is given instead."""
    R0 = Ts[0][:3, :3]
    from sr2hand.geometry.rotations import log_so3
    w = np.array([log_so3(T[:3, :3] @ R0.T) for T in Ts[1:]])
    big = np.linalg.norm(w, axis=1) > math.radians(10)
    ax = (w[big] / np.linalg.norm(w[big], axis=1)[:, None])
    ax = ax * np.sign(ax @ ax[0])[:, None]
    axis = ax.mean(0)
    axis /= np.linalg.norm(axis)
    ang = np.array([0.0] + [angle_about(T[:3, :3] @ R0.T, axis) for T in Ts[1:]])
    ref = np.asarray(enc_deg) - enc_deg[0]
    if np.corrcoef(ang[: min(20, len(ang))], ref[: min(20, len(ang))])[0, 1] < 0:
        ang = -ang
    err = (ang - ref + 180.0) % 360.0 - 180.0
    err -= err.mean()
    P = np.array([T[:3, 3] for T in Ts])
    _c, _n, rad, res = circle_fit(P) if circle else (None, None, 0.0, np.linalg.norm(P - P.mean(0), axis=1))
    return {"angle_rms_deg": float(np.sqrt((err ** 2).mean())), "angle_max_deg": float(np.abs(err).max()),
            "circle_radius_mm": float(rad), "circle_rms_mm": float(np.sqrt((res ** 2).mean())),
            "circle_max_mm": float(np.abs(res).max())}


def render_rolling(rd, m, d, pose_at, t0, cam, bands=8, samples=2):
    """A rolling-shutter frame (portrait: the sensor's rows are the image's columns, read left to right over
    READOUT_S, each exposed for EXPOSURE_S): bands of columns rendered at their own times, each averaged over
    `samples` instants of its exposure."""
    img = np.zeros((cam.height, cam.width, 3), np.float32)
    edges = np.linspace(0, cam.width, bands + 1).astype(int)
    for b in range(bands):
        acc = None
        for k in range(samples):
            tb = t0 + READOUT_S * (b + 0.5) / bands + EXPOSURE_S * (k + 0.5) / samples
            put_block(m, d, pose_at(tb))
            im = rd.render(d, rd.T_WC)
            acc = im if acc is None else acc + im
        img[:, edges[b]:edges[b + 1]] = (acc / samples)[:, edges[b]:edges[b + 1]]
    return img


def cmd_turntable(a):
    import mujoco
    rng = np.random.default_rng(a.seed)
    cad = OM.block_map("cylinder")
    printed = perturbed_map(cad, rng)
    m = block_model(printed)
    d = mujoco.MjData(m)
    cam = portrait()
    rd = Renderer(m, cam)
    det = TP.detector("tag36h11")
    out = {"what": "step 3, tag block on a turntable in simulation", "print_sd": PRINT_SD, "sensor": vars(SENSOR),
           "seed": a.seed, "camera": {"width": cam.width, "height": cam.height, "fx": cam.fx, "dist_m": CAM_DIST,
                                      "elevation_deg": a.el},
           "map_error_cad_mm": TP.map_error(cad, printed)}
    t0 = time.time()
    # bundle adjustment from a.views views of the static block
    views, T0s, Ttrue = [], [], []
    for k in range(a.views):
        th = rng.uniform(0, 360)
        T_WB = table_pose(th, 0.0)
        put_block(m, d, T_WB)
        el = math.radians(rng.uniform(-10, 45))
        az = math.radians(rng.uniform(0, 360))
        dist = rng.uniform(0.22, 0.40)
        c = T_WB[:3, 3]
        eye = c + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        up = rot_axis_angle(c - eye, math.radians(rng.uniform(-10, 10))) @ np.array([0.0, 0.0, 1.0])
        T_WC = look_at(eye, c + rng.normal(size=3) * 0.01, up)
        g = SENSOR.apply(rd.render(d, T_WC), rng)
        obs = TP.detect(det, g, ids=set(cad.ids()))
        T_CB, info = TP.solve_body(obs, cad, cam)
        if T_CB is None or len(info["tags"]) < 2:
            continue
        views.append({i: obs[i] for i in info["tags"]})
        T0s.append(T_CB)
        Ttrue.append(T_mm(tf_inv(T_WC) @ T_WB))
    refined, ba = TP.bundle_adjust(views, cad, cam, T0s)
    out["ba"] = {"views": len(views), "rms_px": ba["rms_px"], "map_error_refined_mm": TP.map_error(refined, printed),
                 "tags_per_view": float(np.mean([len(v) for v in views]))}
    print("bundle adjustment:", out["ba"], "CAD map error", out["map_error_cad_mm"], f"{time.time() - t0:.0f} s",
          flush=True)
    # the turntable through 360 deg in 5 deg steps, level and on the wedge
    runs = {}
    for wedge in (0.0, WEDGE_DEG):
        c = table_pose(0.0, 0.0)[:3, 3]
        e = math.radians(a.el)
        eye = c + CAM_DIST * np.array([math.cos(e), 0.0, math.sin(e)])
        rd.T_WC = T_WC = look_at(eye, c)
        for name, tm in (("cad", cad), ("refined", refined)):
            runs[(wedge, name)] = {"T": [], "true": [], "enc": [], "tracked": 0, "frames": 0, "tags": []}
        for th in np.arange(0.0, 360.0, 5.0):
            T_WB = table_pose(th, wedge)
            put_block(m, d, T_WB)
            img = rd.render(d, T_WC)
            enc = round(th / 360.0 * 4096) * 360.0 / 4096
            for k in range(a.noise):
                g = SENSOR.apply(img, rng)
                obs = TP.detect(det, g, ids=set(cad.ids()))
                for name, tm in (("cad", cad), ("refined", refined)):
                    R_ = runs[(wedge, name)]
                    R_["frames"] += 1
                    T_CB, info = TP.solve_body(obs, tm, cam)
                    if T_CB is None:
                        continue
                    R_["tracked"] += 1
                    R_["T"].append(T_CB)
                    R_["true"].append(T_mm(tf_inv(T_WC) @ T_WB))
                    R_["enc"].append(enc)
                    R_["tags"].append(len(info["tags"]))
    out["turntable"] = []
    for (wedge, name), R_ in runs.items():
        from sr2hand.geometry.rotations import pose_error
        dp, dr = pose_error(np.array(R_["T"]), np.array(R_["true"]))
        bm = bench_metrics(R_["T"], R_["enc"], circle=wedge > 0)
        row = {"wedge_deg": wedge, "map": name, "frames": R_["frames"], "tracked": R_["tracked"],
               "tags_mean": float(np.mean(R_["tags"])), "tags_min": int(np.min(R_["tags"])),
               "rot_rms_deg": float(np.degrees(np.sqrt((dr ** 2).mean()))), "rot_max_deg": float(np.degrees(dr.max())),
               "pos_rms_mm": float(np.sqrt((dp ** 2).mean())), "pos_max_mm": float(dp.max()), **bm}
        out["turntable"].append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    # spinning at a.spin deg/s with the rolling shutter, 30 frames/s over one turn, the refined map
    if a.spin:
        w = a.spin
        c = table_pose(0.0, 0.0)[:3, 3]
        e = math.radians(a.el)
        rd.T_WC = T_WC = look_at(c + CAM_DIST * np.array([math.cos(e), 0.0, math.sin(e)]), c)
        sp = {"frames": 0, "tracked": 0, "err_mid": [], "err_row": [], "rot_mid": [], "rot_row": []}
        from sr2hand.geometry.rotations import pose_error
        for f in range(int(round(360.0 / w * 30))):
            t_f = f / 30.0

            def pose_at(t):
                return table_pose(w * t, a.spin_wedge)
            img = render_rolling(rd, m, d, pose_at, t_f, cam)
            g = SENSOR.apply(img, rng)
            obs = TP.detect(det, g, ids=set(cad.ids()))
            sp["frames"] += 1
            T_CB, info = TP.solve_body(obs, refined, cam)
            if T_CB is None:
                continue
            sp["tracked"] += 1
            # the frame's middle, and the instant the block's centre column was read
            u = project_world(cam, T_WC, [pose_at(t_f)[:3, 3]])[0, 0]
            t_mid = t_f + READOUT_S / 2 + EXPOSURE_S / 2
            t_row = t_f + READOUT_S * float(np.clip(u / cam.width, 0, 1)) + EXPOSURE_S / 2
            for key, tt_ in (("mid", t_mid), ("row", t_row)):
                dp, dr = pose_error(T_CB, T_mm(tf_inv(T_WC) @ pose_at(tt_)))
                sp["err_" + key].append(float(dp))
                sp["rot_" + key].append(float(np.degrees(dr)))
        out["spin"] = {"deg_per_s": w, "wedge_deg": a.spin_wedge, "readout_s": READOUT_S, "exposure_s": EXPOSURE_S,
                       "frames": sp["frames"], "tracked": sp["tracked"],
                       **{f"{k}_rms": float(np.sqrt(np.mean(np.square(v)))) for k, v in sp.items() if isinstance(v, list)},
                       **{f"{k}_max": float(np.max(v)) for k, v in sp.items() if isinstance(v, list)}}
        print("spin:", out["spin"], flush=True)
    rd.close()
    out["refined_map"] = refined.to_json()
    out["printed_map"] = printed.to_json()
    (ROWS / "turntable.json").write_text(json.dumps(out, indent=0))
    print("wrote", ROWS / "turntable.json", f"{time.time() - t0:.0f} s")
    return 0


# ------------------------------------------------------------------------------------------------ steps 2-3: tracking

def hand_tags_P(pair, palm_m, lift_m):
    """{camera index: TagMap in P (mm)} of the hand tags: tag 7 for the first camera, 8 for the second, each
    HAND_TAG_MM on the hand's outline toward its camera at HAND_TAG_Z, facing the camera's position in the task."""
    out = {}
    target = np.array([0.0, 0.0, 0.065 + lift_m])
    seated = np.array([0.0, 0.0, 0.065])
    for k, (az, el) in enumerate(pair):
        hand0 = np.asarray(palm_m) + np.array([HAND_TAG_R * math.cos(math.radians(az)),
                                               HAND_TAG_R * math.sin(math.radians(az)), HAND_TAG_Z]) / 1000.0
        T_WC = camera_pose(az, el, target, 0.5 * (seated + hand0 + np.array([0.0, 0.0, lift_m])))
        eye_P = (T_WC[:3, 3] - np.array([0.0, 0.0, lift_m]) - np.asarray(palm_m)) * 1000.0   # in P during the task
        c = np.array([HAND_TAG_R * math.cos(math.radians(az)), HAND_TAG_R * math.sin(math.radians(az)), HAND_TAG_Z])
        n = eye_P - c
        n /= np.linalg.norm(n)
        down = np.array([0.0, 0.0, -1.0]) - n * (-n[2])
        down /= np.linalg.norm(down)
        tm = TG.TagMap("tag36h11", HAND_TAG_MM)
        tm.add(7 + k, c, np.cross(n, down), down)
        out[k] = tm
    return out


def track_scene(tr, hand_maps, flat=True):
    """The rollout's scene with the hand tags (white plates and black cells on palm_pose) and the camera."""
    import mujoco
    src = ROOT / str(tr["scene"])
    root = ET.parse(src).getroot()
    mat = OM.add_matte(root)
    palm = root.find(".//body[@name='palm_pose']")
    for k, tm in hand_maps.items():
        for tid, (c, r, dn, n) in tm.frames.items():
            Rm = np.stack([r, dn, -n], 1)                  # right-handed (object_tracking_mjcf.add_tag_sites)
            ET.SubElement(palm, "geom", type="box", contype="0", conaffinity="0", group="2", mass="0",
                          pos=OM.fmt((c - 1.0 * n) / 1000.0), quat=OM.quat_of(Rm),
                          size=OM.fmt(np.array([tm.size / 2 + 8.0, tm.size / 2 + 8.0, 1.0]) / 1000.0),
                          rgba="0.92 0.92 0.92 1", material=mat)
        OM.add_tag_sites(palm, tm, f"htag{k}_", material=mat)
    add_camera(root)
    dst = src.with_name(src.stem + f"__track{os.getpid()}.xml")
    ET.ElementTree(root).write(dst)
    try:
        m = mujoco.MjModel.from_xml_path(str(dst))
    finally:
        dst.unlink()
    for k, f in enumerate(("thumb", "index", "middle")):
        m.body_pos[m.body(f"{f}_mount").id] = tr["mounts"][k]
    m.body_pos[m.body("palm_pose").id] = tr["palm"]
    for g in range(m.ngeom):
        nm = m.geom(g).name or ""
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or nm == "post":
            m.geom_group[g] = GROUP_HIDE
            m.geom_rgba[g, 3] = 0.0
        elif m.geom_group[g] in (0, 3) and m.body(m.geom_bodyid[g]).name.split("_")[0] in ("thumb", "index", "middle"):
            m.geom_group[g] = GROUP_PADS
    if flat:
        hide_walls(m)
    return m


def draw(gray, obs, uv_true=None, colour=(40, 200, 40)):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, c in obs.items():
        cv2.polylines(img, [np.round(c + 0.5).astype(np.int32).reshape(-1, 1, 2)], True, colour, 2)
        cv2.putText(img, str(i), tuple(np.round(c.mean(0) - [8, -8]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (30, 30, 230), 2)
    return img


def cmd_track(a):
    """Both cameras of the chosen pair render each rollout at a.fps; per frame the object's pose in P is solved from
    the block's tags and each camera's hand tag (T_PO = T_CP^-1 T_CO, both cameras in one solve) and compared with
    the simulator's. Rows: rows/track_<config>_part<k>of<n>.json (per rollout) and logs/.../track/<rollout>.npz (per frame);
    films media/<date>-track_<task>_<call>.mp4 for a.film rollouts."""
    import mujoco
    from sr2hand.geometry.rotations import pose_error
    vis = json.loads((ROWS / f"visibility_{a.config}.json").read_text())
    pair = [tuple(x) for x in vis["pair"]]
    rows = [r for r in flat_rows(a.config) if r["seed"] in a.seeds and (not a.tasks or r["task"] in a.tasks)]
    rows = rows[a.part::a.parts]
    cam = portrait()
    out = []
    (LOGS / "track").mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    dets = TP.detector("tag36h11")
    rng = np.random.default_rng(7)
    t0 = time.time()
    for r in rows:
        tr = load_traj(ROOT / r["traj"])
        shape = "sphere" if r["task"] in ("roll", "twist") else "cylinder"
        lift = float(r.get("lift_mm", 40.0)) / 1000.0
        hand = hand_tags_P(pair, tr["palm"], lift)
        m = track_scene(tr, hand, flat=a.config.startswith("flat"))
        d = mujoco.MjData(m)
        rd = Renderer(m, cam)
        bmap_O = OM.block_map(shape).transformed(OM.T_OB(float(tr["r_mm"]), float(tr["stem_mm"])))
        target = np.array([0.0, 0.0, 0.065 + lift])
        seated = np.array([0.0, 0.0, 0.065])
        T_tab = []
        for k, (az, el) in enumerate(pair):
            hand0 = np.asarray(tr["palm"]) + hand[k].frames[7 + k][0] / 1000.0
            T_tab.append(camera_pose(az, el, target, 0.5 * (seated + hand0 + np.array([0.0, 0.0, lift]))))
        t = tr["t"]
        times = np.arange(0.0, t[-1], 1.0 / a.fps)
        T_WP = tf(np.eye(3), np.asarray(tr["palm"]))
        rec = {"t": [], "err_mm": [], "err_deg": [], "n_tags": [], "ncam": [], "err1_mm": [], "err1_deg": [],
               "kin_mm": [], "kin_deg": [], "hand_deg": []}
        frames = []
        film = a.film and (r["task"], r["call"]) in {(x.split(":")[0], int(x.split(":")[1])) for x in a.film}
        for tf_ in times:
            i = int(np.argmin(np.abs(t - tf_)))
            set_frame(m, d, tr, i)
            off = table_offset(tr, i)
            T_PO_true = T_mm(tf_inv(T_WP) @ object_T(tr, i))
            obs_all, T_PC, imgs, obs_blk, T_PC_kin, hand_err = [], [], [], [], [], []
            for k in range(2):
                T_WC = T_tab[k].copy()
                T_WC[:3, 3] += off
                g = SENSOR.apply(rd.render(d, T_WC), rng)
                obs = TP.detect(dets, g)
                hid = 7 + k
                T_CP = None
                T_PC_true = T_mm(tf_inv(T_WP) @ T_WC)            # the kinematic route: the camera's pose in P known
                if hid in obs:
                    T_CP, _ = TP.solve_body({hid: obs[hid]}, hand[k], cam)
                    hand_err.append(float(np.degrees(pose_error(tf_inv(T_CP), T_PC_true)[1])))
                ob = {i_: c for i_, c in obs.items() if i_ in bmap_O.corners}
                obs_blk.append(ob)
                T_PC_kin.append(T_PC_true)
                obs_all.append(ob if T_CP is not None else {})
                T_PC.append(tf_inv(T_CP) if T_CP is not None else np.eye(4))
                if film:
                    imgs.append(draw(g, obs))
            rec["hand_deg"].append(float(np.mean(hand_err)) if hand_err else np.nan)
            # the kinematic route, both cameras in one solve from one camera's start
            kin = None
            for k in range(2):
                if obs_blk[k]:
                    T_CO, inf = TP.solve_body(obs_blk[k], bmap_O, cam)
                    if T_CO is not None and (kin is None or len(inf["tags"]) > kin[1]):
                        kin = (T_PC_kin[k] @ T_CO, len(inf["tags"]))
            if kin is None:
                rec["kin_mm"].append(np.nan)
                rec["kin_deg"].append(np.nan)
            else:
                T_PO_k, _ = TP.solve_multi(obs_blk, bmap_O, [cam, cam], T_PC_kin, kin[0])
                dpk, drk = pose_error(T_PO_k, T_PO_true)
                rec["kin_mm"].append(float(dpk))
                rec["kin_deg"].append(float(np.degrees(drk)))
            # one camera's solve as the start, then both cameras in one solve
            best = None
            for k in range(2):
                if obs_all[k]:
                    T_CO, inf = TP.solve_body(obs_all[k], bmap_O, cam)
                    if T_CO is not None and (best is None or len(inf["tags"]) > best[2]):
                        best = (k, T_PC[k] @ T_CO, len(inf["tags"]))
            rec["t"].append(float(t[i]))
            if best is None:
                for key in ("err_mm", "err_deg", "err1_mm", "err1_deg"):
                    rec[key].append(np.nan)
                rec["n_tags"].append(0)
                rec["ncam"].append(0)
            else:
                dp1, dr1 = pose_error(best[1], T_PO_true)
                T_PO, inf = TP.solve_multi(obs_all, bmap_O, [cam, cam], T_PC, best[1])
                dp, dr = pose_error(T_PO, T_PO_true)
                rec["err_mm"].append(float(dp))
                rec["err_deg"].append(float(np.degrees(dr)))
                rec["err1_mm"].append(float(dp1))
                rec["err1_deg"].append(float(np.degrees(dr1)))
                rec["n_tags"].append(int(sum(len(o) for o in obs_all)))
                rec["ncam"].append(int(inf["n_cams"]))
            if film:
                h = 960
                tiles = [cv2.resize(im, (h * cam.width // cam.height, h), interpolation=cv2.INTER_AREA) for im in imgs]
                fr = np.concatenate(tiles, 1)
                e = rec["kin_mm"][-1]
                txt = (f"t {t[i]:.2f} s   error {e:.2f} mm, {rec['kin_deg'][-1]:.2f} deg (camera poses from the arm)"
                       if np.isfinite(e) else f"t {t[i]:.2f} s   not tracked")
                cv2.rectangle(fr, (0, 0), (fr.shape[1], 34), (255, 255, 255), -1)
                cv2.putText(fr, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                frames.append(fr)
        rd.close()
        E = np.array(rec["err_mm"])
        D = np.array(rec["err_deg"])
        tt = np.array(rec["t"])
        task_ = tt >= 2.0
        ok = np.isfinite(E)
        row = {"task": r["task"], "role": r["role"], "call": r["call"], "seed": r["seed"], "score": r["score"],
               "frames": int(len(E)), "tracked": float(ok.mean()), "tracked_task": float(ok[task_].mean()),
               "pos_rms_mm": float(np.sqrt(np.nanmean(E[task_] ** 2))) if ok[task_].any() else None,
               "rot_rms_deg": float(np.sqrt(np.nanmean(D[task_] ** 2))) if ok[task_].any() else None,
               "pos_p99_mm": float(np.nanpercentile(E[task_], 99)) if ok[task_].any() else None,
               "rot_p99_deg": float(np.nanpercentile(D[task_], 99)) if ok[task_].any() else None,
               "two_cams": float(np.mean(np.array(rec["ncam"])[task_] == 2)),
               "pos1_rms_mm": float(np.sqrt(np.nanmean(np.array(rec["err1_mm"])[task_] ** 2))) if ok[task_].any() else None,
               "rot1_rms_deg": float(np.sqrt(np.nanmean(np.array(rec["err1_deg"])[task_] ** 2))) if ok[task_].any() else None,
               "hand_tag_rot_rms_deg": float(np.sqrt(np.nanmean(np.array(rec["hand_deg"])[task_] ** 2))),
               **kin_stats(np.array(rec["kin_mm"]), np.array(rec["kin_deg"]), task_)}
        np.savez_compressed(LOGS / "track" / f"{a.config}_{r['task']}_{r['call']}_s{r['seed']}.npz",
                            **{k: np.array(v) for k, v in rec.items()})
        if film and frames:
            fp = MEDIA / f"20261105-track_{r['task']}_{r['call']}.mp4"
            write_mp4(frames, fp, a.fps)
            row["film"] = str(fp.relative_to(ROOT))
        out.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, f"{time.time() - t0:.0f} s",
              flush=True)
        (ROWS / f"track_{a.config}_part{a.part}of{a.parts}.json").write_text(
            json.dumps({"pair": pair, "fps": a.fps, "rows": out}, indent=0))
    return 0


def kin_stats(E, D, sel):
    """Errors of the kinematic route over the selected frames: tracked share, RMS and 99th percentile."""
    ok = np.isfinite(E) & sel
    if not ok.any():
        return {"kin_tracked_task": 0.0}
    return {"kin_tracked_task": float(ok.sum() / sel.sum()), "kin_pos_rms_mm": float(np.sqrt(np.mean(E[ok] ** 2))),
            "kin_rot_rms_deg": float(np.sqrt(np.mean(D[ok] ** 2))), "kin_pos_p99_mm": float(np.percentile(E[ok], 99)),
            "kin_rot_p99_deg": float(np.percentile(D[ok], 99)), "kin_pos_max_mm": float(E[ok].max()),
            "kin_rot_max_deg": float(D[ok].max())}


def write_mp4(frames, path, fps):
    """H.264 through ffmpeg (yuv420p, even sizes)."""
    import subprocess
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "medium",
           str(path)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        pr.stdin.write(np.ascontiguousarray(f[:h2, :w2]).tobytes())
    pr.stdin.close()
    pr.wait()


# ------------------------------------------------------------------------------------------------ films

def label(img, top, bottom=None, size=0.8):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (255, 255, 255), -1)
    cv2.putText(out, top, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 2, cv2.LINE_AA)
    if bottom:
        cv2.rectangle(out, (0, out.shape[0] - 34), (out.shape[1], out.shape[0]), (255, 255, 255), -1)
        cv2.putText(out, bottom, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, size * 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def cmd_film(a):
    """Side-by-side clips of rollouts from their trajectories (config:task:call:seed), a side view at 640 x 480 and
    25 frames/s, the HOM scene's own lights; out: media/<name>.mp4 and a poster .jpg."""
    import mujoco
    rows = [json.loads(x) for x in ROLLOUT_ROWS.read_text().splitlines() if x.strip()]
    clips = []
    for spec in a.clips:
        cfg, task, call, seed = spec.split(":")
        r = next(x for x in rows if x["config"] == cfg and x["task"] == task and x["call"] == int(call)
                 and x["seed"] == int(seed))
        tr = load_traj(ROOT / r["traj"])
        spec_ = mujoco.MjSpec.from_file(str(ROOT / str(tr["scene"])))
        m = spec_.compile()
        for k, f in enumerate(("thumb", "index", "middle")):
            m.body_pos[m.body(f"{f}_mount").id] = tr["mounts"][k]
        m.body_pos[m.body("palm_pose").id] = tr["palm"]
        for g in range(m.ngeom):
            if (m.geom(g).name or "") == "post" and cfg != "base":
                m.geom_rgba[g, 3] = 0.0
        if cfg.startswith("flat"):
            hide_walls(m)
        m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 960
        d = mujoco.MjData(m)
        rd = mujoco.Renderer(m, 480, 640)
        cv = mujoco.MjvCamera()
        cv.lookat[:] = [0.0, 0.0, 0.095]
        cv.distance, cv.azimuth, cv.elevation = a.distance, a.azimuth, a.elevation
        opt = mujoco.MjvOption()
        opt.sitegroup[:] = 1
        frames = []
        t = tr["t"]
        name = {"base": "study (post)", "flat": "flat seat", "built": "pocket seat", "stem": "held to 1.6 s"}.get(cfg, cfg)
        for tf_ in np.arange(0.0, t[-1], 1.0 / 25):
            i = int(np.argmin(np.abs(t - tf_)))
            d.qpos[:] = tr["qpos"][i]
            if "seat" in tr and len(tr["seat"]):
                mid = m.body_mocapid[m.body("otrack_seat").id]
                d.mocap_pos[mid] = tr["seat"][i][:3]
                d.mocap_quat[mid] = tr["seat"][i][3:]
            mujoco.mj_forward(m, d)
            rd.update_scene(d, cv, scene_option=opt)
            im = cv2.cvtColor(rd.render(), cv2.COLOR_RGB2BGR)
            frames.append(label(im, f"{name}, {task} {r['role']} layout", f"t {t[i]:.2f} s   score {r['score']:.2f}", 0.7))
        rd.close()
        clips.append(frames)
    n = max(len(c) for c in clips)
    out = []
    for k in range(n):
        out.append(np.concatenate([c[min(k, len(c) - 1)] for c in clips], 1))
    MEDIA.mkdir(parents=True, exist_ok=True)
    fp = MEDIA / f"{a.name}.mp4"
    write_mp4(out, fp, 25)
    k = int(np.searchsorted(np.arange(n) / 25.0, a.poster_t))
    cv2.imwrite(str(fp.with_suffix(".jpg")), out[min(k, n - 1)], [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", fp.relative_to(ROOT), len(out), "frames")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tagsize")
    p.add_argument("--res", nargs="+", type=int, default=[1080, 720])
    p.add_argument("--dist", nargs="+", type=float, default=[0.25, 0.3, 0.4, 0.5])
    p.add_argument("--angles", nargs="+", type=float, default=[0, 15, 30, 40, 45, 50, 55, 60, 65, 70])
    p.add_argument("--poses", type=int, default=6)
    p.add_argument("--noise", type=int, default=8)
    p = sub.add_parser("visibility")
    p.add_argument("--config", default="flat")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p = sub.add_parser("pairs")
    p.add_argument("--config", default="flat")
    p = sub.add_parser("turntable")
    p.add_argument("--views", type=int, default=40)
    p.add_argument("--noise", type=int, default=3)
    p.add_argument("--el", type=float, default=20.0, help="camera elevation over the block (deg)")
    p.add_argument("--spin", type=float, default=40.0, help="deg/s of the spinning run (0: none)")
    p.add_argument("--spin-wedge", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("track")
    p.add_argument("--config", default="flat")
    p.add_argument("--seeds", nargs="+", type=int, default=[20])
    p.add_argument("--tasks", nargs="*", default=[])
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--film", nargs="*", default=[], help="task:call of the rollouts to film")
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p = sub.add_parser("film")
    p.add_argument("--clips", nargs="+", required=True, help="config:task:call:seed")
    p.add_argument("--name", required=True)
    p.add_argument("--azimuth", type=float, default=0.0)
    p.add_argument("--elevation", type=float, default=5.0)
    p.add_argument("--distance", type=float, default=0.30)
    p.add_argument("--poster-t", type=float, default=2.5)
    a = ap.parse_args()
    return {"tagsize": cmd_tagsize, "visibility": cmd_visibility, "pairs": cmd_pairs, "turntable": cmd_turntable,
            "track": cmd_track, "film": cmd_film}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
