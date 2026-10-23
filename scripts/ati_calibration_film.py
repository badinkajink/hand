#!/usr/bin/env python3
"""Film of an ATI calibration session (scripts/ati_calibration_session.py, real or --sim): the session's joint record
(robot.bin) replayed in MuJoCo on the UR5's link meshes, with the tool (HEX-E, mount v3's adapter, the ATI base, the
Mini45, the ATI top, the hook cross), the line and mass hanging from the eye in use (events.jsonl), and the cell of the tip
calibration (table, 2x4, plywood, both indenters); two views side by side (the cell, and the tool close up), each under a
label bar with the block, the eye and the mass's state.

    MUJOCO_GL=egl .venv/bin/python scripts/ati_calibration_film.py <session> [--speed 24] [--fps 25] [--out mp4]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ati_calibration_plan as PL  # noqa: E402
import ur5_cb3_cell as CELL  # noqa: E402
import ur5_tip_calibration_film as TF  # noqa: E402

C = CELL.C
OUT = ROOT / "docs/experiments/20261023-ati_ft_calibration/media"
COL = dict(TF.COL, base="0.55 0.78 0.50 1", ati="0.80 0.81 0.84 1", top="0.55 0.78 0.50 1", cross="0.88 0.55 0.30 1",
           line="0.10 0.10 0.10 1", mass="0.40 0.42 0.47 1", other="0.85 0.85 0.82 1")


def scene(work, plan, w, h):
    import trimesh
    rec = json.loads(PL.PARTS.read_text())
    T_BI, kind = PL.load_cell()
    T_EM = np.array(plan["geometry"]["T_EM"], float)
    zM = rec["z_M"]
    assets, bodies, tool = [], [], []
    _, meshes = CELL.urdf_model()
    for ln, stem in TF.ROBOT_MESH.items():
        T, f = meshes[ln]
        m = CELL._load(f)
        m.apply_transform(T)
        TF.mesh_obj(m, work / f"{stem}.obj")
        assets.append(f'<mesh name="{stem}" file="{work / f"{stem}.obj"}" scale="0.001 0.001 0.001"/>')
        rgba = COL["joint"] if ln in ("shoulder_link", "wrist_1_link", "wrist_2_link") else COL["robot"]
        bodies.append(f'<body name="{ln}" mocap="true"><geom type="mesh" mesh="{stem}" rgba="{rgba}" contype="0" conaffinity="0"/></body>')
    parts = [("hex_e", CELL.hex_e_mesh(None), 0.0), ("adapter", CELL.finger_mesh("adapter"), 0.0)]
    for nm, key, z in (("base", "ati_base", zM["adapter_top"]), ("top", "ati_top", zM["ati_origin"]),
                       ("cross", "hook_cross", zM["top_face"])):
        parts.append((nm, trimesh.load(ROOT / rec["parts"][key]["file"], force="mesh"), z))
    ati = trimesh.creation.cylinder(radius=22.5, height=15.7, sections=64)
    parts.append(("ati", ati, zM["ati_mount_face"] + 7.85))
    for nm, m, z in parts:
        m = m.copy()
        m.apply_translation([0.0, 0.0, z])
        m.apply_transform(T_EM)
        TF.mesh_obj(m, work / f"tool_{nm}.obj")
        assets.append(f'<mesh name="t_{nm}" file="{work / f"tool_{nm}.obj"}" scale="0.001 0.001 0.001"/>')
        tool.append(f'<geom type="mesh" mesh="t_{nm}" rgba="{COL.get(nm, COL["robot"])}" contype="0" conaffinity="0"/>')
    bodies.append(f'<body name="tool" mocap="true">{"".join(tool)}</body>')
    L, r, hm = plan["mass"]["line_mm"], 19.0, 60.0                  # a 0.5 kg steel cylinder about 38 x 60 mm
    bodies.append(f'<body name="mass" mocap="true"><geom type="capsule" fromto="0 0 0 0 0 {-L * 1e-3:.4f}" size="0.0006" '
                  f'rgba="{COL["line"]}" contype="0" conaffinity="0"/><geom type="cylinder" pos="0 0 {-(L + hm / 2) * 1e-3:.4f}" '
                  f'size="{r * 1e-3:.4f} {hm / 2 * 1e-3:.4f}" rgba="{COL["mass"]}" contype="0" conaffinity="0"/></body>')
    cell = []
    for T, k in [(T_BI, kind)] + list(CELL.OTHER):
        m = CELL._load(CELL.INDENTER_STL[k])
        m.apply_transform(T)
        cell.append((f"ind_{k}", m, COL["indenter"]))
    b = CELL.BOARD
    hb, hh, t = b["block_mm"], b["half_mm"], b["thick_mm"]
    if hb > 0:
        bx, by = b["block_half_mm"]
        blk = trimesh.creation.box(extents=[2 * bx, 2 * by, hb])
        blk.apply_translation([0.0, 0.0, -hb / 2])
        blk.apply_transform(T_BI)
        cell.append(("block", blk, COL["block"]))
    brd = trimesh.creation.box(extents=[2 * hh, 2 * hh, t])
    brd.apply_translation([0.0, 0.0, -hb - t / 2])
    brd.apply_transform(T_BI)
    cell.append(("board", brd, COL["board"]))
    geoms = []
    for nm, m, col in cell:
        TF.mesh_obj(m, work / f"cell_{nm}.obj")
        assets.append(f'<mesh name="c_{nm}" file="{work / f"cell_{nm}.obj"}" scale="0.001 0.001 0.001"/>')
        geoms.append(f'<geom type="mesh" mesh="c_{nm}" rgba="{col}" contype="0" conaffinity="0"/>')
    geoms.append(f'<geom type="plane" size="1.2 1.2 0.01" rgba="{COL["table"]}" contype="0" conaffinity="0"/>')
    return f"""<mujoco><compiler inertiafromgeom="false"/>
<visual><global offwidth="{w}" offheight="{h}" fovy="30"/><quality shadowsize="4096" offsamples="8"/>
<headlight ambient="0.40 0.40 0.40" diffuse="0.50 0.50 0.50" specular="0.05 0.05 0.05"/></visual>
<asset><texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/>{''.join(assets)}</asset>
<worldbody><light pos="0.4 -0.6 1.5" dir="-0.2 0.3 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
{''.join(geoms)}{''.join(bodies)}</worldbody></mujoco>"""


def timeline(sd):
    """Steps (t0, t1, block, eye, station, role) and mass intervals (t_on, t_off, eye)."""
    steps, mass, open_ = [], [], {}
    on = None
    for line in open(sd / "events.jsonl"):
        e = json.loads(line)
        if e.get("event") == "start" and e["id"] != "preflight":
            open_[e["id"]] = e
        elif e.get("event") == "end" and e["id"] in open_:
            s = open_.pop(e["id"])
            steps.append((s["t"], e["t"], s.get("block"), s.get("eye"), s.get("station"), s.get("role")))
        elif e.get("event") == "mass":
            if e["on"]:
                on = (e["t_mass"], e["eye"])
            elif on is not None:
                mass.append((on[0], e["t_mass"], on[1]))
                on = None
    if on is not None:
        mass.append((on[0], 1e18, on[1]))
    return steps, mass


def main(sd, speed=24.0, fps=25, out=None, w=640, h=480):
    import mujoco
    from PIL import Image
    sd = Path(sd)
    plan = json.loads((sd / "plan.json").read_text())
    PL.load_cell()
    R = np.fromfile(sd / "robot.bin", "<f8")
    R = R[: len(R) // 15 * 15].reshape(-1, 15)
    tq, Q = R[:, 0], R[:, 2:8]
    steps, mass = timeline(sd)
    eyes = {e["name"]: np.array(e["p_M"], float) for e in plan["geometry"]["eyes"]}
    T_EM = np.array(plan["geometry"]["T_EM"], float)
    work = Path(tempfile.mkdtemp(prefix="ati_film_"))
    m = mujoco.MjModel.from_xml_string(scene(work, plan, w, h))
    d = mujoco.MjData(m)
    mid = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mass")]
    frames = work / "frames"
    frames.mkdir()
    t_all = np.arange(tq[0], tq[-1], speed / fps)
    p0 = np.array(plan["p0_mm"]) * 1e-3
    cams = []
    away = np.degrees(np.arctan2(-p0[1], -p0[0]))               # MuJoCo puts the camera at lookat - distance * direction:
    for look, dist, az, el in ((np.array([p0[0] / 2, p0[1] * 0.75, 0.22]), 1.55, 75.0, -18.0),   # the cell from the front
                               (p0 + [0.0, 0.0, -0.05], 0.60, away + 40.0, -16.0)):           # the tool, the robot behind it
        c = mujoco.MjvCamera()
        c.lookat[:], c.distance, c.azimuth, c.elevation = look, dist, az, el
        cams.append(c)
    with mujoco.Renderer(m, h, w) as ren:
        for k, t in enumerate(t_all):
            q = np.array([np.interp(t, tq, Q[:, j]) for j in range(6)])
            TF.set_pose(m, d, q, T_EM)
            hang = next((e for t0, t1, e in mass if t0 <= t <= t1), None)
            if hang is not None:
                T_BM = CELL.link_frames(q)["tool0"] @ T_EM
                d.mocap_pos[mid] = C.apply(T_BM, eyes[hang][None])[0] * 1e-3
            else:
                d.mocap_pos[mid] = [5.0, 5.0, -5.0]
            mujoco.mj_forward(m, d)
            st = next((s for s in steps if s[0] <= t <= s[1]), None)
            if st is None:
                text = "Moving"
            else:
                _, _, b, eye, stn, role = st
                what = {"reference": "reference station, no mass", "load": "mass hung", "unload": "mass taken off",
                        "via": "passing the loading station"}.get(role, "rest")
                text = f"Block {b}, eye {eye or '-'}: {stn}, {what}" + (f"; mass on {hang}" if hang else "")
            imgs = []
            for c in cams:
                ren.update_scene(d, camera=c)
                imgs.append(Image.fromarray(ren.render()))
            bar = TF.label_bar(f"{text}   ({speed:.0f}x)", 2 * w)
            canvas = Image.new("RGB", (2 * w, h + bar.height), "white")
            canvas.paste(bar, (0, 0))
            canvas.paste(imgs[0], (0, bar.height))
            canvas.paste(imgs[1], (w, bar.height))
            canvas.save(frames / f"{k:05d}.png")
    out = Path(out or OUT / "20261023-ati_calibration_sim_session.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow", str(out)], check=True)
    poster = out.with_suffix(".png")
    shutil.copy(frames / f"{len(t_all) // 3:05d}.png", poster)
    shutil.rmtree(work)
    print(f"{len(t_all)} frames, {len(t_all) / fps:.0f} s -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--speed", type=float, default=24.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.session, a.speed, a.fps, a.out)
