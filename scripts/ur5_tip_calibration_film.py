#!/usr/bin/env python3
"""Film of a fingertip calibration session on the UR5 (CB3): the session's joint record (robot.bin) replayed in MuJoCo on
the URDF's link meshes with the tool stack (OptoForce HEX-E, UR5 mount v3, DIP link b v4 with cover, cap and pad) and the
indenter on its block and the plywood, in two views side by side (the cell, and the pad on the sphere), each under a label
bar; the stage, the action and the HEX-E's push are printed on the close view. Stills of chosen moments go to the page.

    MUJOCO_GL=egl .venv/bin/python scripts/ur5_tip_calibration_film.py logs/20261022-tip_calibration/<session>
        [--speed 40] [--fps 25] [--out docs/experiments/20261022-ur5_tip_calibration/media/<name>.mp4]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_cb3_cell as CELL  # noqa: E402

C = CELL.C
MEDIA = ROOT / "docs/experiments/20261022-ur5_tip_calibration/media"
COL = {"robot": "0.80 0.82 0.84 1", "joint": "0.30 0.52 0.78 1", "hex_e": "0.62 0.63 0.66 1", "adapter": "0.80 0.62 0.36 1",
       "fork": "0.36 0.56 0.85 1", "clamp_plate": "0.25 0.45 0.75 1", "dip_link": "0.90 0.90 0.86 1",
       "cover": "0.55 0.58 0.62 1", "cap": "0.45 0.48 0.52 1", "pad": "0.92 0.70 0.45 1", "indenter": "0.85 0.85 0.82 1",
       "block": "0.72 0.58 0.40 1", "board": "0.80 0.68 0.50 1", "table": "0.93 0.93 0.93 1"}
ROBOT_MESH = {"base_link_inertia": "base", "shoulder_link": "shoulder", "upper_arm_link": "upperarm",
              "forearm_link": "forearm", "wrist_1_link": "wrist1", "wrist_2_link": "wrist2", "wrist_3_link": "wrist3"}


def quat(R):
    """(w, x, y, z) of a rotation matrix."""
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def mesh_obj(tm, path):
    import trimesh
    sm = trimesh.graph.smooth_shade(tm, angle=np.radians(30.0), facet_minarea=None)
    Path(path).write_text(trimesh.exchange.obj.export_obj(sm, include_normals=True, include_texture=False))
    return str(path)


def build_scene(work, T_BI, T_EM, kind, w, h, block_mm):
    """MJCF: robot links and the tool as mocap bodies (their poses set every frame), the cell fixed."""
    import trimesh
    _, meshes = CELL.urdf_model()
    assets, bodies = [], []
    for ln, stem in ROBOT_MESH.items():
        T, f = meshes[ln]
        m = CELL._load(f)
        m.apply_transform(T)
        p = mesh_obj(m, work / f"{stem}.obj")
        assets.append(f'<mesh name="{stem}" file="{p}" scale="0.001 0.001 0.001"/>')
        rgba = COL["joint"] if ln in ("shoulder_link", "wrist_1_link", "wrist_2_link") else COL["robot"]
        bodies.append(f'<body name="{ln}" mocap="true"><geom type="mesh" mesh="{stem}" rgba="{rgba}" contype="0" conaffinity="0"/></body>')
    tool = []
    hm = CELL.hex_e_mesh(None)
    for name, m in [("hex_e", hm)] + [(n, CELL.finger_mesh(n)) for n in CELL.TOOL_PARTS]:
        m = m.copy()
        m.apply_transform(T_EM)
        p = mesh_obj(m, work / f"tool_{name}.obj")
        assets.append(f'<mesh name="t_{name}" file="{p}" scale="0.001 0.001 0.001"/>')
        tool.append(f'<geom type="mesh" mesh="t_{name}" rgba="{COL.get(name, COL["robot"])}" contype="0" conaffinity="0"/>')
    bodies.append(f'<body name="tool" mocap="true">{"".join(tool)}</body>')
    ind = CELL._load(CELL.INDENTER_STL[kind])
    ind.apply_transform(T_BI)
    cell = [("indenter", ind)]
    b = CELL.BOARD
    hb, hh, t = block_mm, b["half_mm"], b["thick_mm"]
    if hb > 0:
        bx, by = b["block_half_mm"]
        blk = trimesh.creation.box(extents=[2 * bx, 2 * by, hb])
        blk.apply_translation([0.0, 0.0, -hb / 2])
        blk.apply_transform(T_BI)
        cell.append(("block", blk))
    brd = trimesh.creation.box(extents=[2 * hh, 2 * hh, t])
    brd.apply_translation([0.0, 0.0, -hb - t / 2])
    brd.apply_transform(T_BI)
    cell.append(("board", brd))
    geoms = []
    for name, m in cell:
        p = mesh_obj(m, work / f"cell_{name}.obj")
        assets.append(f'<mesh name="c_{name}" file="{p}" scale="0.001 0.001 0.001"/>')
        geoms.append(f'<geom type="mesh" mesh="c_{name}" rgba="{COL[name]}" contype="0" conaffinity="0"/>')
    geoms.append(f'<geom type="plane" size="1.2 1.2 0.01" rgba="{COL["table"]}" contype="0" conaffinity="0"/>')
    xml = f"""<mujoco><compiler inertiafromgeom="false"/>
<visual><global offwidth="{w}" offheight="{h}" fovy="30"/><quality shadowsize="4096" offsamples="8"/>
<headlight ambient="0.40 0.40 0.40" diffuse="0.50 0.50 0.50" specular="0.05 0.05 0.05"/></visual>
<asset><texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/>{''.join(assets)}</asset>
<worldbody><light pos="0.4 -0.6 1.5" dir="-0.2 0.3 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
{''.join(geoms)}{''.join(bodies)}</worldbody></mujoco>"""
    return xml


def set_pose(m, d, q, T_EM):
    import mujoco
    LF = CELL.link_frames(q)
    for ln in ROBOT_MESH:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ln)
        k = m.body_mocapid[bid]
        T = LF[ln]
        d.mocap_pos[k] = T[:3, 3] * 1e-3
        d.mocap_quat[k] = quat(T[:3, :3])
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tool")
    k = m.body_mocapid[bid]
    T = LF["tool0"]
    d.mocap_pos[k] = T[:3, 3] * 1e-3
    d.mocap_quat[k] = quat(T[:3, :3])
    mujoco.mj_forward(m, d)


def label_bar(text, width, h=44):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (width, h), "white")
    dr = ImageDraw.Draw(im)
    dr.text((12, (h - 26) // 2), text, fill="black", font=ImageFont.truetype("DejaVuSerif.ttf", 22))
    dr.line([(0, h - 1), (width, h - 1)], fill="black", width=1)
    return im


def timeline(sd):
    """(start, end, id, stage, kind, tare (6) or None, press start or None) of every action."""
    ev = [json.loads(l) for l in open(sd / "events.jsonl") if l.strip()]
    spans = []
    open_ = {}
    for e in ev:
        if e.get("event") == "start":
            open_[e["id"]] = (e["t"], e.get("stage"), e.get("kind"))
        elif e.get("event") == "end" and e["id"] in open_:
            t0, st, kd = open_.pop(e["id"])
            rec = e.get("rec") or {}
            spans.append((t0, e["t"], e["id"], st, kd, rec.get("tare"), rec.get("t_move0")))
    return spans


STAGE_TEXT = {"preflight": "Streams checked", "homing": "Homing: the mount's socket on the indenter's key (freedrive)",
              "gravity": "Gravity sweep: the HEX-E's axes and the payload", "mount": "Touch-offs on the mount column's face",
              "pad": "Touch-offs on the pad", "pivot": "Pivot: the face centre from four turns", "grid": "Calibration presses",
              "shear": "Shear: sliding on the sphere", "roll": "Roll about the contact", "twist": "Twist about the normal",
              "final": "Reference press", "final_gravity": "Gravity sweep again"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--speed", type=float, default=40.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", default="960x720")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-seconds", type=float, default=None, help="film only this much of the session (s of session time)")
    ap.add_argument("--still", type=float, nargs="*", default=None, help="render frames at these session times (s) and stop")
    a = ap.parse_args()
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    sd = Path(a.session)
    plan = json.loads((sd / "plan.json").read_text())
    reg = json.loads((sd / "registration.json").read_text())
    T_BI = np.array(reg["T_BI"] if reg.get("T_BI") else plan["sim"]["T_BI"], float)
    T_EM = np.array(plan["geometry"]["T_EM"], float)
    block = float(plan.get("checks", {}).get("block_mm", 38.0))
    CELL.BOARD["block_mm"] = block
    rb = np.fromfile(sd / "robot.bin", "<f8").reshape(-1, 15)
    ft = np.fromfile(sd / "ft.bin", "<f8").reshape(-1, 9)
    spans = timeline(sd)
    w, h = (int(v) for v in a.size.split("x"))
    work = Path(tempfile.mkdtemp(prefix="tipcal_film_"))
    m = mujoco.MjModel.from_xml_string(build_scene(work, T_BI, T_EM, plan["indenter"], w, h, block))
    d = mujoco.MjData(m)
    top = C.apply(T_BI, np.array(plan["geometry"]["indenter"]["centre"]) + [0, 0, plan["geometry"]["indenter"]["radius"]]) * 1e-3
    base_mid = (np.array([0.0, 0.0, 0.25]) + top) / 2
    cams = []
    far = C.apply(T_BI, [60.0, 0.0, 0.0]) * 1e-3 - top          # the far side of the indenter, away from the robot
    az_far = math.degrees(math.atan2(-far[1], -far[0]))         # MuJoCo puts the camera at lookat - distance * direction
    for lookat, dist, az, el in (((base_mid + top) / 2, 1.30, -35.0, -20.0),
                                 (top + np.array([0.0, 0.0, 0.004]), 0.24, az_far + 35.0, -14.0)):
        c = mujoco.MjvCamera()
        c.lookat[:] = lookat
        c.distance, c.azimuth, c.elevation = dist, az, el
        cams.append(c)
    t0, t1 = rb[0, 0], rb[-1, 0]
    if a.max_seconds:
        t1 = min(t1, t0 + a.max_seconds)
    times = np.arange(t0, t1, a.speed / a.fps) if a.still is None else t0 + np.array(a.still)
    font = ImageFont.truetype("DejaVuSerif.ttf", 20)
    bars = [label_bar("The cell: UR5, HEX-E and the mount over the sphere", w),
            label_bar("The pad on the 5 mm sphere", w)]
    fr = work / "frames"
    fr.mkdir()
    rt = rb[:, 0]
    with mujoco.Renderer(m, h, w) as ren:
        for k, t in enumerate(times):
            i = min(int(np.searchsorted(rt, t)), len(rb) - 1)
            set_pose(m, d, rb[i, 2:8], T_EM)
            imgs = []
            for c in cams:
                ren.update_scene(d, camera=c)
                imgs.append(Image.fromarray(ren.render()))
            sp = next((s for s in spans if s[0] <= t <= s[1]), None)
            j = min(int(np.searchsorted(ft[:, 0], t)), len(ft) - 1)
            dr = ImageDraw.Draw(imgs[1])
            txt = STAGE_TEXT.get(sp[3], sp[3]) if sp else ""
            if sp:
                txt += f"\n{sp[2]}   {(t - t0) / 60:5.1f} min of the session"
                if sp[5] is not None and sp[6] is not None and t >= sp[6]:
                    f = ft[max(0, j - 12):j + 1, 3:6].mean(axis=0) - np.array(sp[5][:3])
                    txt += f"\nHEX-E change from the rest {np.linalg.norm(f):4.2f} N"
            dr.multiline_text((14, 12), txt, fill="black", font=font, spacing=6)
            frame = Image.new("RGB", (2 * w, h + bars[0].height), "white")
            frame.paste(bars[0], (0, 0))
            frame.paste(bars[1], (w, 0))
            frame.paste(imgs[0], (0, bars[0].height))
            frame.paste(imgs[1], (w, bars[0].height))
            frame.save(fr / f"f{k:05d}.png")
            if k % 250 == 0:
                print(f"frame {k}/{len(times)}", flush=True)
    if a.still is not None:
        for k, t in enumerate(a.still):
            dst = Path(a.out or MEDIA).with_suffix("") .parent / f"{Path(a.out or 'still').stem}_{int(t)}.png"
            Image.open(fr / f"f{k:05d}.png").save(dst)
            print("wrote", dst)
        shutil.rmtree(work)
        return
    out = Path(a.out or MEDIA / f"{time.strftime('%Y%m%d')}-simulated_session.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(fr / "f%05d.png"), "-vf",
                    "scale=1280:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", str(out)], check=True)
    Image.open(fr / f"f{len(times) // 2:05d}.png").save(out.with_suffix(".png"))
    shutil.rmtree(work)
    print(f"{len(times)} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
