#!/usr/bin/env python3
"""Instrumented rollouts of the HOM controller for the code-walkthrough page (scripts/hom_code_page.py), 2026-10-30.

Runs hom_tasks.rollout on the CPU, on one layout of the 2-D grid with the v4b fingertip (SCALE_HAND=cad_v2), and keeps
every intermediate that hom.HomController.step reports per period and finger, plus the tip-frame positions of the
contact points, the pinch frame {M} and the period's wall time. Films and keyframes draw the generalized contact frames
(GCFs) over the scene. Examples: the twist of the 40 mm sphere and the pin-push of the 25 mm cylinder on the layout with
mounts 95 mm apart along palm x and 75 mm along palm y (grid call 280), each from its foundational pose (grasp 0).

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python scripts/hom_code_trace.py
Writes docs/experiments/20261029-hom_control_review/rows/20261030-code_<task>.json and media/20261030-code_*.{mp4,png}.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")
os.environ.setdefault("MUJOCO_GL", "egl")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fp_sr2_toy as FP  # noqa: E402
import hom_contact_rig as HR  # noqa: E402
import hom_gpu as HG  # noqa: E402
import hom_landscape as L  # noqa: E402

T, S, H = HG.T, HG.S, HG.H
FINGERS = HG.FINGERS
D = ROOT / "docs/experiments/20261029-hom_control_review"
ROWS, MEDIA = D / "rows", D / "media"
EXAMPLES = {"twist": ("twist", 40.0, 280, 0), "pinpush": ("pinpush", 25.0, 280, 0)}
KEYS = {"twist": (0.5, 1.4, 2.1), "pinpush": (0.5, 1.4, 2.5)}       # s: apart, squeezing, during the task
VIEWS = ((90.0, -12.0), (225.0, 25.0), (45.0, 50.0))                 # (azimuth, elevation): side, two from below
COL = {"x1": (0.85, 0.1, 0.1, 1.0), "y1": (0.1, 0.65, 0.1, 1.0), "z1": (0.1, 0.25, 0.9, 1.0),
       "ref": (0.95, 0.75, 0.0, 1.0), "move": (0.75, 0.0, 0.75, 1.0)}


def setup_of(task, d_mm, call, grasp):
    shape = T.SHAPE[task]
    FP.set_sizes(**({"sph_d_mm": d_mm} if shape == "sphere" else {"cyl_d_mm": d_mm}))
    xs, ys, rec = L.grid_poses(task, d_mm)[call]
    g = (rec.get("grasps") or [rec])[grasp]
    six = FP.layout2(xs, ys)
    mxy = HG.mount_xy(six)
    if shape == "sphere":
        obj, _tg, n, _pc, _ax = FP.geometry("tripod", np.asarray(g["v"], float), FP.mounts6(six))
        setup = HG.tripod_setup(task, mxy, (np.asarray(obj["centre"], float),
                                            {f: np.asarray(n[f], float) for f in FINGERS}), d_mm)
    else:
        setup = HG.pinch_setup(task, g["v"], six)
    return xs, ys, g, mxy, setup


def v3(x, k=4):
    return [round(float(a), k) for a in np.asarray(x).ravel()]


class Recorder(H.HomController):
    """HomController that keeps each period's measurement, command, info and wall time."""
    log: list = []

    def step(self, meas, cmd):
        t0 = time.perf_counter()
        out, info = super().step(meas, cmd)
        wall = time.perf_counter() - t0
        tipf = {}
        o = self.mr.obj_pose()[0]
        for f in self.order:
            pb, Rb = self.mr.body_pose(f)
            cop = meas.cop.get(f)
            tipf[f] = {"p1": Rb.T @ (info[f]["gcf"].p1 - pb), "p_act": Rb.T @ (info[f]["p_act"] - pb),
                       "cop": None if cop is None else Rb.T @ (np.asarray(cop[0]) - pb), "obj": Rb.T @ (o - pb)}
        M = None if self.M is None else (self.M[0].copy(), self.M[1].copy())
        Recorder.log.append({"meas": meas, "cmd": cmd, "out": out, "info": info, "tip": tipf, "M": M, "wall": wall})
        return out, info


def arrow(scn, a, b, rgba, width=0.0011):
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def draw_frames(scn, rec, task):
    """The GCF {L1} of each finger at its origin (x1 red, toward the object; y1 green; z1 blue) and the velocity of the
    finger's material point there that Eq. (2) planned (magenta, 1 mm per mm/s); for the twist the object's reference
    angular velocity through its centre (yellow, 1 mm per deg/s), for the pin-push the pinch axis, {M}'s x (yellow)."""
    info, meas = rec["info"], rec["meas"]
    for f in FINGERS:
        i = info[f]
        g = i["gcf"]
        for k, (name, ln) in enumerate((("x1", 0.022), ("y1", 0.014), ("z1", 0.014))):
            arrow(scn, g.p1, g.p1 + ln * g.R1[:, k], COL[name], 0.0016)
        v = g.R1 @ (i["A"][3:] @ i["u"])                              # J_p u, world
        if np.linalg.norm(v) > 5e-4:
            arrow(scn, g.p1, g.p1 + v, COL["move"], 0.0016)
    o = np.asarray(meas.obj_pos, float)
    R = H.quat_to_mat(meas.obj_quat)
    if task in ("twist", "roll"):
        w = R @ np.asarray(rec["cmd"].object_twist[:3], float)
        if np.linalg.norm(w) > 1e-6:
            arrow(scn, o, o + 1e-3 * np.degrees(w), COL["ref"], 0.0022)
    elif rec["M"] is not None:
        Rm, pm = rec["M"]
        c, x = o + R @ pm, R @ Rm[:, 0]
        arrow(scn, c - 0.04 * x, c + 0.04 * x, COL["ref"], 0.002)


class Film:
    """hom_tasks.Film with the finger links and the object see-through, the GCFs drawn and the support post faded
    once it is removed; also keeps the plant's qpos per frame."""

    def __init__(self, plant, setup, task, w=640, h=480):
        import mujoco
        self.mj, self.task = mujoco, task
        self.r = mujoco.Renderer(plant.m, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.0, 0.0, S.Z_OBJ + 0.005]
        self.cam.distance = 0.17 if setup["shape"] == "cylinder" else 0.16
        self.cam.azimuth, self.cam.elevation = (340.0, 12.0) if setup["shape"] == "cylinder" else (60.0, 20.0)
        m = plant.m
        for g in range(m.ngeom):
            bn = m.body(m.geom_bodyid[g]).name
            if any(bn.startswith(f + "_") for f in FINGERS) and "_pad" not in (m.geom(g).name or ""):
                m.geom_rgba[g, 3] = min(float(m.geom_rgba[g, 3]), 0.4)
            if m.geom_bodyid[g] == plant.tb:
                m.geom_rgba[g, 3] = 0.45
        self.post = plant.post
        self.frames, self.qpos = [], []

    def frame(self, plant, title, line):
        if plant.d.time >= T.T_LIFT:
            plant.m.geom_rgba[self.post, 3] = 0.12
        self.r.update_scene(plant.d, self.cam)
        draw_frames(self.r.scene, Recorder.log[-1], self.task)
        self.qpos.append((float(plant.d.time), plant.d.qpos.copy()))
        return HR.annotate_bottom(HR.annotate(self.r.render().copy(), title), line)

    def close(self):
        self.r.close()


def keyframes(plant, film, task, times, out, w=420, h=340):
    """The plant at the frames nearest the given times, each from three views, with the GCFs drawn."""
    import mujoco
    ren = mujoco.Renderer(plant.m, h, w)
    d = mujoco.MjData(plant.m)
    rows = []
    try:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, S.Z_OBJ + 0.005]
        cam.distance = 0.13
        for t in times:
            k = int(np.argmin([abs(tt - t) for tt, _ in film.qpos]))
            tk, q = film.qpos[k]
            d.qpos[:] = q
            mujoco.mj_forward(plant.m, d)
            i = int(round(tk / T.DT_C)) - 1                 # the period whose command was executing
            rec = Recorder.log[max(0, min(i, len(Recorder.log) - 1))]
            ims = []
            for az, el in VIEWS:
                cam.azimuth, cam.elevation = az, el
                ren.update_scene(d, cam)
                draw_frames(ren.scene, rec, task)
                ims.append(ren.render().copy())
            ph = rec["cmd_phase"] if "cmd_phase" in rec else ""
            rows.append(HR.annotate(np.concatenate(ims, 1), f"t = {tk:.2f} s, {ph}", size=16))
    finally:
        ren.close()
    from PIL import Image
    Image.fromarray(np.concatenate(rows, 0)).save(out)
    print("wrote", out.relative_to(ROOT), flush=True)


def finger_row(info, tip, f):
    i = info[f]
    g = i["gcf"]
    r = {"dist_mm": round(1e3 * g.dist, 4), "src": g.source, "p1": v3(g.p1, 6), "p0": v3(g.p0, 6), "R1": v3(g.R1, 5),
         "V": v3(i["V"], 5), "A": v3(i["A"], 5), "b": v3(i["b"], 6), "v_ref": v3(i["v_ref"], 5),
         "w": v3(i["weights"], 3), "u": v3(i["u"], 5), "v_plan": v3(i["v_plan"], 6), "force": i["force"],
         "theta": v3(i["theta"], 5), "tr0": v3(i["theta_ref_prev"], 5), "tr": v3(i["theta_ref"], 5),
         "gap_mm": round(1e3 * i["gap"], 4), "off_gap": v3(i["off_gap"], 5), "off_force": v3(i["off_force"], 5),
         "off_grav": v3(i["off_grav"], 5), "q_cmd": v3(i["q_cmd"], 5), "F": round(i["F"], 4),
         "k": None if i["k_filter"] is None else v3(i["k_filter"], 3), "n_act": v3(i["n_act"], 5),
         "p1_tip_mm": v3(1e3 * tip["p1"], 3), "act_tip_mm": v3(1e3 * tip["p_act"], 3),
         "cop_tip_mm": None if tip["cop"] is None else v3(1e3 * tip["cop"], 3), "obj_tip_mm": v3(1e3 * tip["obj"], 3)}
    return r


def run(name):
    import mujoco
    task, d_mm, call, grasp = EXAMPLES[name]
    xs, ys, g, mxy, setup = setup_of(task, d_mm, call, grasp)
    scene = HG.Scene(T.SHAPE[task], d_mm)
    bm, q_open = scene.prepare(mxy, setup)
    plant = T.Plant(scene.pads_path, setup)
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for m in (plant.m, m_pt):
        for f in FINGERS:
            m.body_pos[m.body(f"{f}_mount").id] = bm[f]
    S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = H.Mirror(m_pt, fs, os_)
    film = Film(plant, setup, task)
    Recorder.log = []
    real = T.H.HomController
    T.H.HomController = Recorder
    try:
        res, trace, frames = T.rollout(task, setup, plant, mirror, q_open, 0, T.PRM, H.HomParams(dt=T.DT_C), film)
    finally:
        T.H.HomController = real
    for rec, row in zip(Recorder.log, trace):
        rec["cmd_phase"] = row["phase"]
    MEDIA.mkdir(parents=True, exist_ok=True)
    mp4 = MEDIA / f"20261030-code_{name}.mp4"
    HR.write_mp4(frames, mp4, fps=25)
    from PIL import Image
    Image.fromarray(frames[len(frames) // 2]).convert("RGB").save(mp4.with_suffix(".jpg"), quality=85)
    keyframes(plant, film, task, KEYS[name], MEDIA / f"20261030-code_{name}_keyframes.png")
    film.close()
    periods = []
    for rec, row in zip(Recorder.log, trace):
        p = {"t": row["t"], "phase": row["phase"], "pos_mm": row["pos_mm"], "axis": row["axis"],
             "w_dps": row["w_dps"], "F_link": row["F_link"], "wall_ms": round(1e3 * rec["wall"], 3),
             "obj_twist": v3(rec["cmd"].object_twist, 6), "s": rec["cmd"].s,
             "M": None if rec["M"] is None else {"R": v3(rec["M"][0], 5), "p": v3(rec["M"][1], 6)},
             "fingers": {f: finger_row(rec["info"], rec["tip"][f], f) for f in FINGERS}}
        for k in ("w_ref_dps", "theta_ref_deg"):
            if k in row:
                p[k] = row[k]
        periods.append(p)
    lim = {f: {"lo": v3(mirror.lo[f], 5), "hi": v3(mirror.hi[f], 5), "kp": v3(mirror.kp[f], 3)} for f in FINGERS}
    m = plant.m
    pads = [g for g in range(m.ngeom) if "_pad" in (m.geom(g).name or "")]
    a0 = plant.act[FINGERS[0]][0]
    model = {"timestep": float(m.opt.timestep), "pads_per_finger": len(pads) // 3, "pad_r_mm": 1e3 * float(m.geom_size[pads[0]][0]),
             "pad_friction": v3(m.geom_friction[pads[0]], 3), "pad_solref": v3(m.geom_solref[pads[0]], 5),
             "obj_mass_g": 1e3 * float(m.body_mass[plant.tb]), "kp": float(m.actuator_gainprm[a0][0]),
             "forcerange": v3(m.actuator_forcerange[a0], 3), "damping": float(m.dof_damping[plant.vadr[FINGERS[0]][0]]),
             "cone": int(m.opt.cone), "impratio": float(m.opt.impratio)}
    out = {"task": task, "d_mm": d_mm, "call": call, "grasp": grasp, "x_sep_mm": xs, "y_sep_mm": ys,
           "pose_v": g["v"], "setup": {"palm": v3(setup["palm"], 6), "quat": v3(setup["quat"], 6),
                                       "n_out": {f: v3(setup["n_out"][f], 5) for f in FINGERS},
                                       "mounts": {f: v3(bm[f], 6) for f in FINGERS}, "r": setup["r"],
                                       "hl": setup["hl"]},
           "q_open": {f: v3(q_open[f], 5) for f in FINGERS}, "limits": lim, "model": model, "result": res,
           "hom": {k: v for k, v in vars(H.HomParams(dt=T.DT_C)).items() if k != "filter"},
           "filter": vars(H.FilterParams()), "prm": T.PRM, "periods": periods,
           "when": time.strftime("%Y-%m-%d %H:%M")}
    ROWS.mkdir(parents=True, exist_ok=True)
    path = ROWS / f"20261030-code_{name}.json"
    path.write_text(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"{name}: held {res.get('held')}, {len(periods)} periods, wrote {path.relative_to(ROOT)}", flush=True)


def main():
    for name in (sys.argv[1:] or EXAMPLES):
        run(name)


if __name__ == "__main__":
    main()
