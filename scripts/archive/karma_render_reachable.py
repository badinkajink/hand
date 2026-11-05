"""Render a KaRMA reachable set on the real_v1 hand it was computed for.

The metric's output is a set of object-centre voxels, each carrying the fraction of the 228
twist-invariant orientation bins reachable there. Upstream ships a viser app for this;
this renders the same thing offline, in MuJoCo, on the hand's own geometry, so the cloud
can be read against the fingers that produce it and against the depth the task's grasp
actually uses.

Voxels are drawn as translucent cubes coloured by rotational coverage, red for the least
and green for the most, matching the upstream viewer's convention. The hand is posed at
KaRMA's seed configuration.

    python scripts/karma_render_reachable.py --run <workdir>/<design>_thumb_index_scaled \\
        --mounts="-50,0,50,55,50,-55" --out fig.png
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _colour(t: float) -> str:
    """Red (least coverage) to green (most), the upstream viewer's ramp."""
    t = float(np.clip(t, 0.0, 1.0))
    r, g, b = (0.85, 0.25, 0.20) if t < 0.5 else (0.20, 0.70, 0.35)
    if t < 0.5:
        u = t / 0.5
        r, g, b = 0.85, 0.25 + 0.55 * u, 0.20
    else:
        u = (t - 0.5) / 0.5
        r, g, b = 0.85 - 0.65 * u, 0.80, 0.20 + 0.15 * u
    return f"{r:.3f} {g:.3f} {b:.3f} 0.55"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="a run dir with current.yaml")
    ap.add_argument("--mounts", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--cam-dist", type=float, default=0.30)
    ap.add_argument("--cam-az", type=float, default=55.0)
    ap.add_argument("--cam-elev", type=float, default=10.0)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    import mujoco
    from karma_replay_mujoco import hand_at_mounts, KARMA_Q_ORDER

    y = yaml.safe_load((args.run / "current.yaml").read_text())
    mcfg = yaml.safe_load((args.run / "metric.yaml").read_text())
    sm, seed, vox = y["summary"], y["seed"], y["voxels"]
    scale = float(sm["l_ref_m"]) / float(mcfg.get("l_ref_nominal_m", 0.2))
    voxel = float(mcfg["voxel_size_m"]) * scale
    sphere_r = float(mcfg["sphere_radius_m"]) * scale
    centre0 = np.array(seed["centre_world"], float)

    tree = hand_at_mounts([float(x) for x in args.mounts.split(",")])
    root = tree.getroot()
    world = root.find("worldbody")
    palm = world.find("./body[@name='palm_pose']")
    palm_z = float(palm.attrib["pos"].split()[2])
    for s in list(palm.findall("site")):
        palm.remove(s)
    for g in list(world.findall("geom")):
        if g.attrib.get("name") == "floor":
            world.remove(g)
    # An ElementTree element with no children is falsy, so `find(...) or SubElement(...)`
    # silently appends a duplicate; these have to be explicit `is None` checks.
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    gl = vis.find("global")
    if gl is None:
        gl = ET.SubElement(vis, "global")
    gl.set("offwidth", str(max(args.width, 1920)))
    gl.set("offheight", str(max(args.height, 1440)))

    # Voxel indices are relative to the seed voxel, in the seed's manipulability frame.
    frame = np.array(seed["seed_frame"], float)
    for key, v in vox.items():
        i, j, k = (int(x) for x in key.split("_"))
        p = centre0 + frame @ (np.array([i, j, k], float) * voxel)
        p = p + np.array([0.0, 0.0, palm_z])
        ET.SubElement(world, "geom", {
            "type": "box", "size": f"{voxel * 0.34:.6f} {voxel * 0.34:.6f} {voxel * 0.34:.6f}",
            "pos": f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}", "contype": "0", "conaffinity": "0",
            "rgba": _colour(float(v["rot_coverage"]) /
                            max(1e-9, max(w["rot_coverage"] for w in vox.values())))})
    s0 = centre0 + np.array([0.0, 0.0, palm_z])
    ET.SubElement(world, "geom", {
        "type": "sphere", "size": f"{sphere_r:.6f}", "contype": "0", "conaffinity": "0",
        "pos": f"{s0[0]:.6f} {s0[1]:.6f} {s0[2]:.6f}", "rgba": "0.95 0.95 0.98 0.9"})

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), {})
    data = mujoco.MjData(model)
    for n, j in enumerate(KARMA_Q_ORDER):
        data.qpos[model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]] = seed["q"][n]
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = s0
    cam.distance, cam.azimuth, cam.elevation = args.cam_dist, args.cam_az, args.cam_elev
    r = mujoco.Renderer(model, height=args.height, width=args.width)
    r.update_scene(data, cam)
    img = r.render()

    import imageio.v2 as imageio
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(args.out), img)
    print(f"{args.run.name}: {len(vox)} voxels, voxel {voxel * 1e3:.2f} mm, "
          f"sphere r {sphere_r * 1e3:.2f} mm -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
