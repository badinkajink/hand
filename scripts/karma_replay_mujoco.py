"""Replay a KaRMA rolling path on the same hand in MuJoCo, with contact physics.

KaRMA's search is kinematic: at each step it solves a rolling-contact QP and projects
back onto the two-contact manifold, so its output is a joint trajectory plus the sphere
pose it believes that trajectory produces. Nothing in the search integrates dynamics.

This script takes a computed result, walks the BFS parent links from the seed out to the
furthest reached voxel, and drives the real_v1 MJCF's position actuators along exactly
that joint trajectory with a free sphere of KaRMA's own test radius seeded at KaRMA's own
start pose. It then compares the sphere pose MuJoCo produces against the pose KaRMA
predicted, and reports contact count and tracking error along the way.

Gravity is off by default. KaRMA claims kinematic rolling feasibility with antipodal
force closure, not that the pinch survives its own weight, so gravity would test a claim
the metric does not make; ``--gravity`` turns it on for the harder reading.

    python scripts/karma_replay_mujoco.py --pkl <run>/current.pkl \\
        --hand assets/mjcf/experimental/.../hand_....xml --video out.mp4
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Pinocchio orders joints alphabetically, so a KaRMA q vector is index, middle, thumb.
KARMA_Q_ORDER = ["index_yaw", "index_mcp", "index_pip",
                 "middle_yaw", "middle_mcp", "middle_pip",
                 "thumb_yaw", "thumb_mcp", "thumb_pip"]


def load_path(pkl: Path, karma_root: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(q[k,9], centres[k,3], R[k,3,3], meta) from the seed to the furthest voxel.

    The pickle holds karma dataclasses, so the package has to be importable. It is not
    imported for its behaviour and pulls in no pinocchio on this path, so putting the
    KaRMA checkout on sys.path is enough -- the replay still runs in the project venv,
    where MuJoCo lives.
    """
    if karma_root is not None and str(karma_root) not in sys.path:
        sys.path.insert(0, str(karma_root))
    with open(pkl, "rb") as fh:
        blob = pickle.load(fh)
    res = blob["result"]

    seed = res.seed_centre
    far, best = None, -1.0
    for key, node in res.state_nodes.items():
        d = float(np.linalg.norm(node.centre_world - seed))
        if d > best:
            best, far = d, key

    chain, key = [], far
    while key is not None:
        node = res.state_nodes[key]
        chain.append(node)
        key = node.parent
    chain.reverse()

    q = np.array([n.q for n in chain])
    c = np.array([n.centre_world for n in chain])
    R = np.array([n.R_sphere for n in chain])
    meta = {"n_states": len(chain), "reach_mm": best * 1e3,
            "contact_pair": list(chain[-1].contact_pair),
            "n_voxels": res.n_voxels_reached, "l_ref_mm": res.l_ref_m * 1e3}
    return q, c, R, meta


BASE_MJCF = ROOT / "assets/mjcf/real_v1/real_hand.xml"


def hand_at_mounts(mounts_mm: list[float]) -> ET.ElementTree:
    """The base real_v1 hand re-mounted and morphology-frozen.

    Same construction as the URDF export: the six mount coordinates are written into the
    mount bodies and the morph slides deleted, so the MJCF the replay drives and the URDF
    KaRMA scored are the same hand by construction rather than by coincidence.
    """
    tree = ET.parse(BASE_MJCF)
    root = tree.getroot()
    for i, f in enumerate(("thumb", "index", "middle")):
        body = root.find(f".//body[@name='{f}_mount']")
        body.set("pos", f"{mounts_mm[2 * i] * 1e-3:.6f} {mounts_mm[2 * i + 1] * 1e-3:.6f} 0")
    for parent in root.iter():
        for j in list(parent.findall("joint")):
            if j.attrib.get("class") == "morph":
                parent.remove(j)
    for kf in list(root.findall("keyframe")):
        root.remove(kf)
    return tree


def build_scene(hand_xml: Path | ET.ElementTree, sphere_r: float, centre: np.ndarray,
                gravity: bool, pad_radius: float | None = None) -> str:
    """The hand with KaRMA's test sphere as a free body at the seed centre.

    KaRMA works in the palm frame with the base link at the origin; the MJCF palm sits at
    ``palm_pose``'s z, so the sphere's world position is the palm offset plus the KaRMA
    centre. The floor is removed: a floor is a contact KaRMA never modelled.
    """
    tree = hand_xml if isinstance(hand_xml, ET.ElementTree) else ET.parse(hand_xml)
    root = tree.getroot()

    if pad_radius is not None:
        # KaRMA models every finger link as a capsule of one radius, which at the paper's
        # nominal 11.8 mm is fatter than the real_v1 pad (10.55 mm). Widening the MJCF
        # geoms to KaRMA's radius replays the path on the geometry KaRMA actually solved,
        # separating "the rolling constraint does not survive contact physics" from
        # "the capsule model is not this hand".
        for body in root.iter("body"):
            for g in body.findall("geom"):
                if g.attrib.get("type") in ("capsule", "sphere") and "finger_mat" in \
                        g.attrib.get("material", ""):
                    g.set("size", f"{pad_radius:.6f}")

    world = root.find("worldbody")
    palm = world.find("./body[@name='palm_pose']")
    for site in list(palm.findall("site")):
        palm.remove(site)  # mount-workspace boxes: visualisation only, clutter up close
    palm_z = float(palm.attrib["pos"].split()[2])
    for geom in list(world.findall("geom")):
        if geom.attrib.get("name") == "floor":
            world.remove(geom)

    opt = root.find("option")
    if opt is not None and not gravity:
        opt.set("gravity", "0 0 0")

    # The offscreen framebuffer defaults to 640x480; anything larger has to be declared.
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    glob = vis.find("global")
    if glob is None:
        glob = ET.SubElement(vis, "global")
    glob.set("offwidth", "1920")
    glob.set("offheight", "1440")

    p = np.array(centre, float) + np.array([0.0, 0.0, palm_z])
    body = ET.SubElement(world, "body", {"name": "karma_sphere",
                                         "pos": f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"})
    ET.SubElement(body, "freejoint", {"name": "sphere_free"})
    ET.SubElement(body, "geom", {
        "name": "karma_sphere_geom", "type": "sphere", "size": f"{sphere_r:.6f}",
        "density": "500", "material": "object_mat",
        # The pads carry the tuned high-friction contact; match it so the replay tests
        # the rolling constraint rather than an arbitrarily slippery object.
        "friction": "2.4 0.2 0.02"})

    for kf in list(root.findall("keyframe")):
        root.remove(kf)
    return ET.tostring(root, encoding="unicode")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, required=True)
    ap.add_argument("--hand", type=Path, default=None,
                    help="a generated real_v1 MJCF; omit and pass --mounts instead")
    ap.add_argument("--mounts", default=None,
                    help="six palm-frame millimetres 'tx,ty,ix,iy,mx,my', the same "
                         "string handed to karma_export_urdf.py")
    ap.add_argument("--karma-root", type=Path, default=None,
                    help="the KaRMA checkout, needed to unpickle its result dataclasses")
    ap.add_argument("--sphere-radius-mm", type=float, required=True)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--gravity", action="store_true")
    ap.add_argument("--settle-s", type=float, default=0.30,
                    help="seconds held at the seed pose before the path is driven")
    ap.add_argument("--pad-radius-mm", type=float, default=None,
                    help="override the MJCF finger radius, e.g. KaRMA's own "
                         "scaled link_radius, so the replay uses the geometry "
                         "the metric solved")
    ap.add_argument("--squeeze-deg", type=float, default=0.0,
                    help="extra commanded flexion on the two contact fingers, degrees; "
                         "KaRMA's pinch has zero preload so some squeeze is needed before "
                         "the object is carried at all")
    ap.add_argument("--substeps", type=int, default=25,
                    help="control ticks interpolated between consecutive KaRMA states")
    ap.add_argument("--step-s", type=float, default=0.15,
                    help="seconds of simulation per KaRMA state")
    ap.add_argument("--cam-dist", type=float, default=0.15)
    ap.add_argument("--cam-elev", type=float, default=12.0,
                    help="positive looks UP from below the palm; the fingers hang under it")
    ap.add_argument("--cam-az", type=float, default=110.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    import mujoco

    q, centres, _R, meta = load_path(args.pkl, args.karma_root)
    if args.mounts:
        hand = hand_at_mounts([float(x) for x in args.mounts.split(",")])
    elif args.hand:
        hand = args.hand
    else:
        raise SystemExit("pass --hand or --mounts")
    xml = build_scene(hand, args.sphere_radius_mm * 1e-3, centres[0], args.gravity,
                      pad_radius=(args.pad_radius_mm * 1e-3
                                  if args.pad_radius_mm else None))
    model = mujoco.MjModel.from_xml_string(xml, {})
    data = mujoco.MjData(model)

    act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{j}")
           for j in KARMA_Q_ORDER]
    jnt = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in KARMA_Q_ORDER]
    qadr = [model.jnt_qposadr[i] for i in jnt]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "karma_sphere")
    sadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "sphere_free")]
    sgeom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "karma_sphere_geom")
    palm_z = float(ET.fromstring(xml).find("./worldbody/body[@name='palm_pose']")
                   .attrib["pos"].split()[2])

    # Start on the seed configuration exactly, then let the solver settle the contact.
    for k, a in enumerate(qadr):
        data.qpos[a] = q[0, k]
    data.ctrl[act] = q[0]
    mujoco.mj_forward(model, data)

    renderer = None
    frames = []
    if args.video is not None:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        # Frame the pinch, not the hand: the sphere is ~10 mm across on a 170 mm palm.
        cam.lookat[:] = [centres[0][0], centres[0][1], centres[0][2] + palm_z]
        cam.distance, cam.azimuth, cam.elevation = args.cam_dist, args.cam_az, args.cam_elev

    dt = model.opt.timestep
    fps = 40.0
    trace = []

    # KaRMA's contact is a zero-preload kinematic touch: the gap is driven to within
    # eps_g and no further, so nothing in the metric asks the fingers to press. A pinch
    # with no preload carries nothing, so the replay adds an explicit squeeze -- extra
    # commanded flexion on the two contact fingers' MCP and PIP, which is how grip force
    # is produced on this hand (commanded minus achieved).
    squeeze = np.zeros(9)
    contact_fingers = {link.split("_")[0] for link in meta["contact_pair"]}
    for k, name in enumerate(KARMA_Q_ORDER):
        f, j = name.split("_")
        if f in contact_fingers and j in ("mcp", "pip"):
            squeeze[k] = np.radians(args.squeeze_deg)

    def _record(predicted: np.ndarray) -> None:
        pos = data.xpos[sid].copy()
        pos[2] -= palm_z
        ncon = sum(1 for c in data.contact[:data.ncon] if sgeom in (c.geom1, c.geom2))
        force = 0.0
        for ci in range(data.ncon):
            c = data.contact[ci]
            if sgeom in (c.geom1, c.geom2):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, f6)
                force += abs(float(f6[0]))
        trace.append({"predicted_mm": (predicted * 1e3).round(3).tolist(),
                      "actual_mm": (pos * 1e3).round(3).tolist(),
                      "error_mm": round(float(np.linalg.norm(pos - predicted) * 1e3), 3),
                      "contacts": int(ncon), "normal_N": round(force, 3)})

    def _advance(seconds: float, target: np.ndarray) -> None:
        n = max(1, int(seconds / dt))
        for i in range(n):
            data.ctrl[act] = target + squeeze
            mujoco.mj_step(model, data)
            if renderer is not None and (i % max(1, int(1.0 / (fps * dt))) == 0):
                renderer.update_scene(data, cam)
                frames.append(renderer.render())

    _advance(args.settle_s, q[0])
    _record(centres[0])
    settle_mm = trace[-1]["error_mm"]
    # Interpolate between KaRMA states so the hand is driven quasi-statically. The metric's
    # own BFS sub-step is an eighth of a voxel; a whole state per control tick is a slam.
    for k in range(1, len(q)):
        for s in range(1, args.substeps + 1):
            _advance(args.step_s / args.substeps,
                     q[k - 1] + (q[k] - q[k - 1]) * (s / args.substeps))
        _record(centres[k])

    pred_disp = float(np.linalg.norm(centres[-1] - centres[0]) * 1e3)
    act_disp = float(np.linalg.norm(
        np.array(trace[-1]["actual_mm"]) - np.array(trace[0]["actual_mm"])))
    held = all(t["contacts"] >= 2 for t in trace)
    summary = {
        **meta,
        "sphere_radius_mm": args.sphere_radius_mm,
        "gravity": bool(args.gravity),
        "pad_radius_mm": args.pad_radius_mm,
        "squeeze_deg": args.squeeze_deg,
        "settle_error_mm": settle_mm,
        "predicted_travel_mm": round(pred_disp, 2),
        "actual_travel_mm": round(act_disp, 2),
        "travel_ratio": round(act_disp / pred_disp, 3) if pred_disp > 1e-9 else None,
        "final_error_mm": trace[-1]["error_mm"],
        "max_error_mm": round(max(t["error_mm"] for t in trace), 2),
        "min_contacts": min(t["contacts"] for t in trace),
        "median_normal_N": round(float(np.median([t["normal_N"] for t in trace])), 3),
        "two_contacts_throughout": held,
    }
    print(json.dumps(summary, indent=1))

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "trace": trace}, indent=1))
    if args.video is not None and frames:
        import imageio.v2 as imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(args.video), frames, fps=int(fps), quality=8)
        print(f"wrote {args.video} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
