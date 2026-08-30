#!/usr/bin/env python3
"""Render the trajectory the hardware will actually replay.

Everything visual we have of the real_v1 reorient is of the DENSE carry that
`probe_real_v1_carry.py` simulates. That is not what the bench runs. The exported
plan is three set-points that `plan.run_trajectory` interpolates, plus a CSV whose
per-joint timing differs from that chord, and `real_v1_trajectory_clearance.py`
shows the three paths are far enough apart to disagree about whether the fingers
collide (g24: dense carry clear, chord -5.3 mm, csv -4.2 mm).

So this renders the exported artifact itself, driving the deploy scene through the
same poses the control station will command, reusing the loaders in
`real_v1_trajectory_clearance.py` so there is exactly one definition of what the
plan means.

KINEMATIC REPLAY, NOT A SIMULATION. Joints are set with mj_forward, not stepped:
the point is to show the commanded path, and the servos are position-controlled
against a hard stop that the sim's compliance would blur. The object is therefore
NOT carried in this view -- for the physics, use probe_real_v1_carry.py --video.
The clearance number in the corner is the honest content: it is the pose the hand
is actually commanded into.

  python3 scripts/real_v1_render_deploy_plan.py --design g12 --path csv
  python3 scripts/real_v1_render_deploy_plan.py --all --frames 10
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco  # noqa: E402
from real_v1_trajectory_clearance import (  # noqa: E402
    DEPLOY, DISTMAX, FINGERS, JOINTS, chord_path, csv_path, densify,
    finger_geoms, min_clearance, qadr,
)


def render(design: str, which: str, frames: int, width: int, height: int,
           out: Path, substeps: int = 4):
    plan = json.loads((DEPLOY / f"{design}_plan.json").read_text())
    m = mujoco.MjModel.from_xml_path(plan["meta"]["scene"])
    d = mujoco.MjData(m)
    groups, adr = finger_geoms(m), qadr(m)
    pairs = list(itertools.combinations(FINGERS, 2))
    owner = {gi: f for f, gs in groups.items() for gi in gs}

    path = (chord_path(plan, 55) if which == "chord"
            else csv_path(plan, DEPLOY / f"{design}_traj.csv"))
    dense = densify(path, substeps)

    ren = mujoco.Renderer(m, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    # The turn is a rotation about the pinch axis (world X), so azimuth 0 -- looking
    # down -X at the Y-Z plane -- is the only view it reads in. At the default
    # azimuth the shaft is nearly end-on and 90 deg of turn looks like a growing disc.
    cam.azimuth, cam.elevation, cam.distance = 0, -8, 0.42

    idx = np.linspace(0, len(dense) - 1, frames).astype(int)
    tiles, worst_overall = [], (DISTMAX, "", None)
    for n, i in enumerate(idx):
        pose = dense[i]
        for f in FINGERS:
            for j in JOINTS:
                d.qpos[adr[(f, j)]] = np.deg2rad(pose[f][j])
        mujoco.mj_forward(m, d)
        clr, who, _ = min_clearance(m, d, groups, pairs, owner)
        if clr < worst_overall[0]:
            worst_overall = (clr, who, i / (len(dense) - 1))
        cam.lookat[:] = d.body("palm_pose").xpos
        ren.update_scene(d, cam)
        tiles.append((ren.render(), i / (len(dense) - 1), clr, who))

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise SystemExit("pillow is required to tile the filmstrip")

    cols = min(len(tiles), 5)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * height), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    for n, (img, u, clr, who) in enumerate(tiles):
        x, y = (n % cols) * width, (n // cols) * height
        sheet.paste(Image.fromarray(img), (x, y))
        # Per-frame clearance, because a filmstrip that only showed the pose would
        # hide the entire point: the collision is a few millimetres and invisible.
        tag = f"u={u:.2f}   {clr*1000:+.1f} mm {who}"
        draw.rectangle([x, y, x + width, y + 16], fill=(16, 16, 18))
        draw.text((x + 4, y + 4), tag,
                  fill=(255, 90, 90) if clr < 0.005 else (150, 230, 150))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    w, who, at = worst_overall
    print(f"[render] {design:10s} {which:5s} -> {out}  "
          f"(min {w*1000:+.1f} mm {who} at u={at:.2f})")
    return {"design": design, "path": which, "min_mm": round(w * 1000, 2),
            "pair": who, "at_u": round(at, 3), "png": str(out)}


def replay_physics(design: str, out_png: Path, out_mp4: Path | None,
                   frames: int, width: int, height: int, fps: int = 40):
    """Step the deploy scene under the EXPORTED csv and see whether the tool is carried.

    The kinematic view above shows the commanded path; this shows what the commanded
    path does to the object. It is the end-to-end question nobody had asked of the
    exported artifact: the plan was produced from a carry that worked, but the plan is
    a resampled, re-timed version of that carry, and only physics settles whether the
    resampling still reorients the tool.

    The plan carries everything needed to make this faithful: `replay_initial_qpos` is
    the pose the control station starts from and `replay_base_ctrl` the command it holds
    the palm at, so the fingers are the only thing the csv drives.
    """
    plan = json.loads((DEPLOY / f"{design}_plan.json").read_text())
    meta = plan["meta"]
    m = mujoco.MjModel.from_xml_path(meta["scene"])
    d = mujoco.MjData(m)
    d.qpos[:] = np.array(meta["replay_initial_qpos"], dtype=float)
    d.ctrl[:] = np.array(meta["replay_base_ctrl"], dtype=float)
    mujoco.mj_forward(m, d)

    import csv as _csv
    rows = list(_csv.DictReader(open(DEPLOY / f"{design}_traj.csv")))
    act = {(f, j): mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{f}_{j}")
           for f in FINGERS for j in JOINTS}
    pz = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_palm_pz")
    obj = "screwdriver_medium"
    oid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj)

    ren = mujoco.Renderer(m, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.azimuth, cam.elevation, cam.distance = 0, -8, 0.34

    dt = m.opt.timestep
    shots, vframes, trace = [], [], []
    grab = set(np.linspace(0, len(rows) - 1, frames).astype(int))
    for i, r in enumerate(rows):
        t_next = float(rows[min(i + 1, len(rows) - 1)]["t_s"])
        for (f, j), a in act.items():
            d.ctrl[a] = np.deg2rad(float(r[f"{f}_{j}_deg"]))
        if "palm_z_mm" in r and pz >= 0:
            d.ctrl[pz] = float(meta["replay_base_ctrl"][pz]) + float(r["palm_z_mm"]) / 1000.0
        n = max(1, int(round((t_next - float(r["t_s"])) / dt)))
        for _ in range(n):
            mujoco.mj_step(m, d)
        q = d.body(oid).xquat
        cos = 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)   # object +Z vs world +Z
        z = float(d.body(oid).xpos[2])
        ncon = sum(1 for k in range(d.ncon)
                   if oid in (m.geom_bodyid[d.contact[k].geom1],
                              m.geom_bodyid[d.contact[k].geom2]))
        trace.append({"t_s": float(r["t_s"]), "cos": round(cos, 4),
                      "z": round(z, 4), "obj_contacts": ncon})
        if out_mp4 is not None and i % 2 == 0:
            cam.lookat[:] = d.body(oid).xpos
            ren.update_scene(d, cam)
            vframes.append(ren.render())
        if i in grab:
            cam.lookat[:] = d.body(oid).xpos
            ren.update_scene(d, cam)
            shots.append((ren.render(), float(r["t_s"]), cos, z, ncon))

    from PIL import Image, ImageDraw
    cols = min(len(shots), 5)
    nrows = (len(shots) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, nrows * height), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    for n, (img, t, cos, z, nc) in enumerate(shots):
        x, y = (n % cols) * width, (n // cols) * height
        sheet.paste(Image.fromarray(img), (x, y))
        draw.rectangle([x, y, x + width, y + 16], fill=(16, 16, 18))
        # cos alone is not a verdict: a shaft standing on the table reads 1.0, so the
        # height and the contact count travel with it.
        draw.text((x + 4, y + 4), f"t={t:.2f}s  cos={cos:+.3f}  z={z*1000:.0f}mm  con={nc}",
                  fill=(150, 230, 150) if (cos > 0.8 and nc > 0) else (255, 190, 90))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    if out_mp4 is not None and vframes:
        import imageio.v2 as imageio
        imageio.mimsave(str(out_mp4), vframes, fps=fps)
    fin = trace[-1]
    print(f"[physics] {design}: final cos {fin['cos']:+.3f}  z {fin['z']*1000:.0f} mm  "
          f"object contacts {fin['obj_contacts']}  -> {out_png}")
    return {"design": design, "final": fin, "trace": trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--path", choices=("chord", "csv", "both"), default="both")
    ap.add_argument("--physics", action="store_true",
                    help="also STEP the exported csv in the deploy scene and report whether "
                         "the tool is actually carried (the kinematic view cannot say)")
    ap.add_argument("--mp4", action="store_true", help="write a video alongside the filmstrip")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--width", type=int, default=420)
    ap.add_argument("--height", type=int, default=340)
    ap.add_argument("--outdir", type=Path,
                    default=Path("docs/experiments/20260830-deploy_renders"))
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    designs = args.design
    if args.all or not designs:
        designs = sorted(p.name.replace("_plan.json", "")
                         for p in DEPLOY.glob("*_plan.json"))
    which = ("chord", "csv") if args.path == "both" else (args.path,)

    rows = []
    for dsg in designs:
        for w in which:
            rows.append(render(dsg, w, args.frames, args.width, args.height,
                               args.outdir / f"{dsg}_{w}.png"))
    if args.physics:
        for dsg in designs:
            rows.append(replay_physics(
                dsg, args.outdir / f"{dsg}_physics.png",
                (args.outdir / f"{dsg}_physics.mp4") if args.mp4 else None,
                args.frames, args.width, args.height))
    (args.outdir / "RENDERS.json").write_text(json.dumps(rows, indent=1))
    print(f"[render] index -> {args.outdir / 'RENDERS.json'}")


if __name__ == "__main__":
    main()
