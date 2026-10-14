#!/usr/bin/env python3
"""Film of the three-finger grip on the 16 mm cylinder (scale-study cell cylinder:16:0.625, servo-limited plant) with
three fingertip models side by side: 1 mm sphere pads, 2 mm sphere pads and the TPU block mesh with condim 4.

Each panel is the CPU MuJoCo plant of pad_usability_timing.make_plant (the same cell scene, candidate grasp and palm
pose), closed by the grip command at t = 0 and held for 1 s after the 0.8 s grip, rendered every 20 ms of simulated
time and played at 25 frames per second (half speed), seen from the index and middle side (the thumb presses from
behind the cylinder). Each TPU block is drawn translucent; pads touching the cylinder are drawn red, and on the mesh
panel each fingertip contact with the cylinder is drawn as a red dot. The label bar gives the model with its CPU cost
per 1 ms step in the held grip (median of docs/experiments/20261014-pad_usability/cost/cpu_timing.jsonl, state grip,
kind step), the simulated time and the fingertip contacts on the cylinder.

    MUJOCO_GL=egl taskset -c 17-19 .venv/bin/python scripts/pad_usability_film.py
Out: docs/experiments/20261014-pad_usability/cost/media/{grip_<model>.mp4, grip_models_cyl016.mp4, grip_models_cyl016.png}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import contact_bed_common as CB  # noqa: E402
import hand_object_scale_mpc as M  # noqa: E402
import hand_object_scale_sim as S  # noqa: E402
import pad_usability_timing as PT  # noqa: E402
import reorient_backends_films as F  # noqa: E402

COST = ROOT / "docs/experiments/20261014-pad_usability/cost"
MEDIA = COST / "media"
PANELS = [("pads", "1 mm pads"), ("pads2", "2 mm pads"), ("pt4", "Block mesh, condim 4")]
BLOCK_RGBA = (0.85, 0.55, 0.35, 0.28)
CAM = (0.15, 180.0, -8.0)        # distance (m), azimuth, elevation (deg): the index and middle side of the grasp


def step_cost(cell: str) -> dict:
    out = {}
    path = COST / "cpu_timing.jsonl"
    if path.exists():
        for ln in path.read_text().splitlines():
            r = json.loads(ln)
            if r.get("cell") == cell and r.get("kind") == "step" and r.get("state") == "grip" and r.get("status") == "ok":
                out[r["model"]] = r["step_us"]["median"]
    return out


def film(mdl, title, cell, shape, dd, ci, paths, meta, gg, cost, t_grip=0.8, t_hold=1.0, every=0.02):
    import mujoco
    p, _ = PT.make_plant(mdl, paths, meta, gg)
    m, d = p.m, p.d
    pads = [i for i in range(m.ngeom) if "_pad" in (m.geom(i).name or "")]
    for i in pads:
        m.geom_rgba[i] = (0.9, 0.1, 0.1, 0.0)
    for f in S.FINGERS:                           # every model draws the TPU block translucent over its contacts
        for nm in (f"{f}_tipvis", f"{f}_tipgeom"):
            try:
                g = m.geom(nm).id
            except KeyError:
                continue
            m.geom_matid[g] = -1
            m.geom_rgba[g] = BLOCK_RGBA
    rend, cam = F.renderer_for(m, np.asarray(meta["tool7"], float)[:3])
    cam.lookat[:] = np.asarray(meta["tool7"], float)[:3] + [0.0, 0.0, 0.006]
    cam.distance, cam.azimuth, cam.elevation = CAM
    grip = np.array([meta["grip"][f][k] for f in S.FINGERS for k in range(3)])
    p.reset(0)
    p.set_targets(grip)
    k_every = int(round(every / m.opt.timestep))
    frames, step = [], 0
    us = cost.get(mdl)
    while True:
        if step % k_every == 0:
            m.geom_rgba[pads, 3] = 0.0
            dots, n_tool = [], 0
            for i in range(d.ncon):
                g1, g2 = int(d.contact[i].geom[0]), int(d.contact[i].geom[1])
                for g, o in ((g1, g2), (g2, g1)):
                    if o in p.tool_geoms and g not in p.tool_geoms and g in p.finger_of:
                        n_tool += 1
                        if g in pads:
                            m.geom_rgba[g, 3] = 1.0
                        else:
                            dots.append(d.contact[i].pos.copy())
            rend.update_scene(d, cam)
            scn = rend.scene
            for pos in dots:                      # mesh contacts: a red dot at each contact point
                if scn.ngeom < scn.maxgeom:
                    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.0011, 0, 0]),
                                        pos, np.eye(3).reshape(-1), np.array([0.9, 0.1, 0.1, 1.0], np.float32))
                    scn.ngeom += 1
            lines = [f"{title}: {us:.0f} µs per step" if us else title,
                     f"{d.time:4.2f} s, {n_tool} contacts on the cylinder"]
            frames.append(M.big_label(rend.render().copy(), lines, size=32))
        if d.time >= t_grip + t_hold - 1e-9:
            break
        mujoco.mj_step(m, d)
        step += 1
    s = p.state()
    rend.close()
    return frames, {f: round(float(s["F"][f]), 2) for f in S.FINGERS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plant", default="kp0.72_kv0_fr0.18_fl0_dp0.0144")
    ap.add_argument("--cell", default="cylinder:16:0.625")
    ap.add_argument("--size", default="640x480")
    a = ap.parse_args()
    F.W, F.H = (int(v) for v in a.size.split("x"))
    S.PLANT = a.plant
    shape, dd, t = a.cell.split(":")
    dd, t = float(dd), float(t)
    cell, _, _ = next(c for c in S.cell_list("diag") if c[1] == shape and c[2] == dd and c[0].get("t") == t)
    ci = M.hom_best(M.OUT / "hom_postoff_servo/turn.jsonl")[(cell["tag"], shape, dd)]["cand"]
    plant0, meta, geo, sc, gg = M.setup(cell, shape, dd, ci)
    del plant0
    paths = {"pads": sc["pads"][0], "pt": sc["pt"][0], "pads2": PT.pads2_scene(cell, shape, dd)[0]}
    cost = step_cost(a.cell)
    MEDIA.mkdir(parents=True, exist_ok=True)
    clips = []
    tag = S.obj_tag(shape, dd).replace(".0", "")
    for mdl, title in PANELS:
        frames, F_end = film(mdl, title, cell, shape, dd, ci, paths, meta, gg, cost)
        frames += [frames[-1]] * 25
        out = MEDIA / f"grip_{mdl}_{tag}.mp4"
        CB.write_h264(frames, out, crf=20)
        clips.append(out)
        print(f"{mdl}: {len(frames)} frames, forces at the end {F_end} -> {out.relative_to(ROOT)}", flush=True)
    stem = MEDIA / f"grip_models_{tag}"
    n = F.tile_stream(clips, stem.with_suffix(".mp4"), stem.with_suffix(".jpg"), cols=len(clips), poster_at=0.72, crf=20)
    from PIL import Image
    Image.open(stem.with_suffix(".jpg")).save(stem.with_suffix(".png"))
    stem.with_suffix(".jpg").unlink()
    print(f"tiled {n} frames -> {stem.with_suffix('.mp4').relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
