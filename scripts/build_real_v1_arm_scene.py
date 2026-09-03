"""Put the real_v1 hand on a UR5e, so the palm's 6 DOF have to be paid for by a real arm.

Every result in this program so far was produced by a HARRY POTTER HAND: a palm body carrying
six position-actuated slide/hinge joints, gravity-compensated, that can be commanded to any pose
in its box. The chain probe leans on that hard -- the re-pose to vertical is a rigid end-effector
transfer, T_palm = T_obj_des . T_obj^-1 . T_palm, and the re-index afterwards translates the palm
under a shaft standing on the floor. Both are arm moves. Neither has ever been asked whether an
arm can make them.

This builds the scene that asks. The six pose joints are deleted, the palm is bolted to the
UR5e's tool flange, and the same commands must now be produced by six revolute joints with real
limits, a real reach envelope, and a wrist that can run out of range or pass through a
singularity. Nothing about the hand, the object, the floor or the contact parameters changes, so
a difference between the two scenes is the ARM's, not the task's.

It writes two files:

  <out>          the task scene: floor, object, pedestal, UR5e, hand. What the probe rolls out.
  <ik-out>       the UR5e alone with a `palm_site` at the same flange offset. Differential IK
                 runs HERE, not on the task scene: mink integrates over every joint a model
                 has, and the task scene's joints include nine finger DOF and the object's
                 free joint, which an IK solver would happily "use" to reach the target.

    uv run --extra rl --extra arm python scripts/build_real_v1_arm_scene.py \
        --morph-run results/phase1/real_v1/rv05_manual_stored
"""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MENAGERIE = Path("/home/humanoid/Programs/mujoco_menagerie/universal_robots_ur5e")
ARM_BODIES = ("shoulder_link", "upper_arm_link", "forearm_link",
              "wrist_1_link", "wrist_2_link", "wrist_3_link")

# The hand hangs off the flange with its fingers pointing away from the wrist. The UR5e's
# wrist_3 frame has +y out of the flange (that is where `attachment_site` sits, at 0 0.1 0) and
# the palm's fingers run along its own -z, so the palm frame is the wrist rotated +90 deg about
# x: palm_z = -y_wrist3 = into the flange, fingers out along +y_wrist3.
PALM_POS = "0 0.1 0"
PALM_QUAT = "0.70710678 0.70710678 0 0"


def _find(root, tag, **attrs):
    for e in root.iter(tag):
        if all(e.get(k) == v for k, v in attrs.items()):
            return e
    return None


def _indent(elem, level=0):
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
        if not (elem.tail or "").strip():
            elem.tail = pad
    elif level and not (elem.tail or "").strip():
        elem.tail = pad


def build(design_scene: Path, ur5e_xml: Path, out: Path, ik_out: Path,
          base_xyz=(-0.50, 0.0, 0.30), pedestal: bool = True,
          gravcomp: bool = False, arm_gravcomp: float = 1.0,
          meshdir: Path | None = None, model_name: str = "ur5e_real_v1") -> dict:
    hand = ET.parse(design_scene).getroot()
    ur = ET.parse(ur5e_xml).getroot()

    root = ET.Element("mujoco", {"model": model_name})
    ET.SubElement(root, "compiler", {
        "angle": "radian", "autolimits": "true",
        # Absolute, and deliberately not an <include>: including the menagerie file splices a
        # second <compiler> whose meshdir is relative to ITS directory, and the resulting mesh
        # paths depend on where the generated scene is written.
        "meshdir": str((meshdir or (ur5e_xml.parent / "assets")).resolve())})
    for tag in ("option", "visual"):
        src = hand.find(tag)
        if src is not None:
            root.append(copy.deepcopy(src))
    vis = root.find("visual")
    if vis is not None:                      # the arm scene is 1.5 m across; 640x480 crops it
        # `find(...) or SubElement(...)` is a trap here: an Element with no children is
        # FALSY, so that idiom appends a second <global> and the schema rejects the file.
        g = vis.find("global")
        if g is None:
            g = ET.SubElement(vis, "global")
        g.set("offwidth", "1600")
        g.set("offheight", "1200")

    # ---- defaults: the design's, plus the ur5e class nested inside it -----------------------
    dflt = copy.deepcopy(hand.find("default"))
    for c in ur.find("default"):
        dflt.append(copy.deepcopy(c))
    root.append(dflt)

    # ---- assets: both, no name collisions (checked) ------------------------------------------
    asset = ET.SubElement(root, "asset")
    names = set()
    for src in (hand.find("asset"), ur.find("asset")):
        for c in src:
            # keyed by (tag, name): MuJoCo namespaces assets by TYPE, and the hand scene
            # legitimately has a texture and a material both called "groundplane".
            n = (c.tag, c.get("name") or c.get("file"))
            if n in names:
                raise SystemExit(f"asset name collision on {n!r}; the two models must be merged "
                                 f"by hand, not by this script")
            names.add(n)
            asset.append(copy.deepcopy(c))

    # ---- worldbody ---------------------------------------------------------------------------
    wb = ET.SubElement(root, "worldbody")
    hwb = hand.find("worldbody")
    for c in hwb:
        if c.tag == "light" or (c.tag == "geom" and c.get("name") == "floor"):
            wb.append(copy.deepcopy(c))
    obj = next(c for c in hwb if c.tag == "body" and c.get("name") not in (None, "palm_pose")
               and c.find("freejoint") is not None)
    wb.append(copy.deepcopy(obj))
    if pedestal:
        # The arm has to stand on something, and where it stands is a real constraint: on the
        # floor the UR5e's own forearm fouls the table before the hand is over the shaft.
        ET.SubElement(wb, "geom", {
            "name": "pedestal", "type": "box", "material": "palm_mat",
            "pos": f"{base_xyz[0]} {base_xyz[1]} {base_xyz[2] / 2:.6g}",
            "size": f"0.09 0.09 {base_xyz[2] / 2:.6g}"})

    base = copy.deepcopy(_find(ur.find("worldbody"), "body", name="base"))
    base.set("pos", f"{base_xyz[0]:.6g} {base_xyz[1]:.6g} {base_xyz[2]:.6g}")
    if arm_gravcomp:
        # THE MENAGERIE UR5e HAS NO GRAVITY FEEDFORWARD. Its actuators are a bare P/D pair
        # (gainprm 2000, biasprm 0 -2000 -400) holding a 12 kg arm, so the shoulder sits 21 mrad
        # below its set-point under the arm's OWN weight and the palm droops 12.7 mm at the top
        # of the lift -- which alone breaks the seam (carry exit 29 deg instead of 5). A real
        # UR5e's servo loop carries a dynamic model and holds ~0.1 mm under payload, so
        # compensating the ARM's links is the closer model, not a cheat. The hand and whatever
        # it holds are deliberately NOT compensated: that is real payload the controller does
        # not know about.
        for b in ARM_BODIES:
            _find(base, "body", name=b).set("gravcomp", str(arm_gravcomp))
    palm = copy.deepcopy(_find(hwb, "body", name="palm_pose"))
    for j in [c for c in palm if c.tag == "joint"]:
        palm.remove(j)                       # the six Harry Potter DOF, deleted
    palm.set("pos", PALM_POS)
    palm.set("quat", PALM_QUAT)
    if gravcomp:
        palm.set("gravcomp", "1")
    else:
        palm.attrib.pop("gravcomp", None)    # the arm carries the hand's weight now
    ET.SubElement(palm, "site", {"name": "palm_site", "size": "0.004",
                                 "rgba": "0.9 0.3 0.2 0.6"})
    _find(base, "body", name="wrist_3_link").append(palm)
    wb.append(base)

    # ---- contact + actuators -----------------------------------------------------------------
    con = copy.deepcopy(hand.find("contact"))
    if con is not None:
        for link in ("wrist_3_link", "wrist_2_link"):
            for f in ("thumb", "index", "middle"):
                for seg in ("yaw_frame", "mcp_frame", "pip_frame", "tip"):
                    ET.SubElement(con, "exclude", {"body1": link, "body2": f"{f}_{seg}"})
        root.append(con)
    act = ET.SubElement(root, "actuator")
    for c in ur.find("actuator"):
        act.append(copy.deepcopy(c))
    kept = 0
    for c in hand.find("actuator"):
        if c.get("class") == "ctrl":         # the nine finger servos; the six palm ones go
            act.append(copy.deepcopy(c))
            kept += 1
    if kept != 9:
        raise SystemExit(f"expected 9 finger actuators, kept {kept}")

    _indent(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding="unicode")

    # ---- the IK model: the arm alone, with the palm frame as a site --------------------------
    ik = ET.parse(ur5e_xml).getroot()
    ik.set("model", "ur5e_palm_ik")
    c = ik.find("compiler")
    c.set("meshdir", str((meshdir or (ur5e_xml.parent / "assets")).resolve()))
    b = _find(ik.find("worldbody"), "body", name="base")
    b.set("pos", f"{base_xyz[0]:.6g} {base_xyz[1]:.6g} {base_xyz[2]:.6g}")
    ET.SubElement(_find(b, "body", name="wrist_3_link"), "site",
                  {"name": "palm_site", "pos": PALM_POS, "quat": PALM_QUAT, "size": "0.004"})
    _indent(ik)
    ik_out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(ik).write(ik_out, encoding="unicode")
    return {"scene": str(out), "ik": str(ik_out), "base_xyz": list(base_xyz)}


def solve_home(scene: Path, ik_xml: Path, design_scene: Path, seeds: int = 24,
               seed0: int = 0) -> tuple:
    """Find an arm configuration that reproduces the design's `open_ik` palm pose, and is sane.

    A 6R arm reaches most poses eight ways and differential IK returns whichever branch the seed
    is nearest. Twelve random seeds on this target all converge to 0.01 mm and they are NOT
    interchangeable: several put the elbow through the table, one leaves wrist_2 at its limit
    with no room to rotate. So the branches are enumerated, then scored in the FULL scene --
    every arm link clear of the floor, no contact anywhere, and joints away from their stops,
    because the seam's re-pose needs wrist range left over.
    """
    import mujoco
    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))
    from morphohand.tools.arm_ik import ArmIK, UR5E_JOINTS

    dm = mujoco.MjModel.from_xml_path(str(design_scene))
    dd = mujoco.MjData(dm)
    mujoco.mj_resetDataKeyframe(dm, dd, dm.key("open_ik").id)
    mujoco.mj_forward(dm, dd)
    R = dd.body("palm_pose").xmat.reshape(3, 3).copy()
    p = dd.body("palm_pose").xpos.copy()

    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    # Everything except the arm comes from the design's own open_ik, matched BY NAME: the two
    # models have different joint orders and an index-wise copy silently spawns the object
    # inside the palm.
    for j in range(dm.njnt):
        name = dm.joint(j).name
        if not name:
            continue
        try:
            tgt = m.joint(name)
        except KeyError:
            continue
        n = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}.get(
            mujoco.mjtJoint(dm.jnt_type[j]), 1)
        a0, a1 = dm.jnt_qposadr[j], m.jnt_qposadr[tgt.id]
        d.qpos[a1:a1 + n] = dd.qpos[a0:a0 + n]
    for a in range(dm.nu):
        try:
            d.ctrl[m.actuator(dm.actuator(a).name).id] = dd.ctrl[a]
        except KeyError:
            continue

    ik = ArmIK(ik_xml)
    rng = np.random.default_rng(seed0)
    cands, seen = [], []
    for i in range(seeds):
        s0 = (np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]) if i == 0
              else rng.uniform(-np.pi, np.pi, 6))
        q, ep, er = ik.solve(R, p, s0, iters=600)
        if ep > 1e-3 or er > 1e-3:
            continue
        q = np.array([(v + np.pi) % (2 * np.pi) - np.pi for v in q])   # nearest wrap
        if any(np.allclose(q, o, atol=1e-2) for o in seen):
            continue
        seen.append(q)
        adr = [m.jnt_qposadr[m.joint(j).id] for j in UR5E_JOINTS]
        for a, v in zip(adr, q):
            d.qpos[a] = float(v)
        mujoco.mj_forward(m, d)
        zmin = min(float(d.body(b).xpos[2]) for b in ARM_BODIES)
        rng_lo = m.jnt_range[[m.joint(j).id for j in UR5E_JOINTS], 0]
        rng_hi = m.jnt_range[[m.joint(j).id for j in UR5E_JOINTS], 1]
        margin = float(np.min(np.minimum(q - rng_lo, rng_hi - q)))
        cands.append({"q": q, "ncon": int(d.ncon), "zmin": zmin, "margin": margin,
                      "score": (d.ncon, -min(zmin, 0.25), -margin)})
    if not cands:
        raise SystemExit("no IK branch reproduces the design's open_ik palm pose")
    best = min(cands, key=lambda c: c["score"])
    return best, cands, d, m


def write_home(scene: Path, q, d, m) -> None:
    """Write the solved configuration into the scene as its `open_ik` keyframe."""
    import mujoco
    import numpy as np
    from morphohand.tools.arm_ik import UR5E_JOINTS
    for a, v in zip([m.jnt_qposadr[m.joint(j).id] for j in UR5E_JOINTS], q):
        d.qpos[a] = float(v)
    for i, j in enumerate(UR5E_JOINTS):
        d.ctrl[m.actuator(m.actuator_trnid[:, 0].tolist().index(m.joint(j).id)).id] = float(q[i])
    mujoco.mj_forward(m, d)
    root = ET.parse(scene).getroot()
    for k in root.findall("keyframe"):
        root.remove(k)
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", {"name": "open_ik",
                              "qpos": " ".join(f"{v:.9g}" for v in d.qpos),
                              "ctrl": " ".join(f"{v:.9g}" for v in d.ctrl)})
    _indent(root)
    ET.ElementTree(root).write(scene, encoding="unicode")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--morph-run", type=Path,
                    default=ROOT / "results/phase1/real_v1/rv05_manual_stored")
    ap.add_argument("--scene", type=Path, default=None,
                    help="a scene XML directly; defaults to the run's frozen_scene.xml")
    ap.add_argument("--ur5e", type=Path, default=MENAGERIE / "ur5e.xml")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ik-out", type=Path, default=None)
    ap.add_argument("--base", default="-0.50,0,0.30",
                    help="x,y,z of the UR5e base mount")
    ap.add_argument("--gravcomp", action="store_true",
                    help="keep the palm's gravity compensation. Off by default: on the arm the "
                         "hand's ~0.2 kg is real payload.")
    ap.add_argument("--arm-gravcomp", type=float, default=1.0,
                    help="gravity compensation on the UR5e's own links, 0..1. A real UR5e's "
                         "controller does this; the menagerie model does not, and without it "
                         "the arm droops 12.7 mm under its own weight.")
    ap.add_argument("--ik-seeds", type=int, default=24)
    ap.add_argument("--cl-assets", type=Path, default=None,
                    help="also copy the generated pair into a CL_Assets mujoco_assets dir, "
                         "together with the menagerie UR5e it needs")
    args = ap.parse_args()

    scene = args.scene or (args.morph_run / "frozen_scene.xml")
    out = args.out or (args.morph_run / "arm_scene.xml")
    ik_out = args.ik_out or (args.morph_run / "arm_ik.xml")
    info = build(scene, args.ur5e, out, ik_out,
                 base_xyz=tuple(float(v) for v in args.base.split(",")),
                 gravcomp=args.gravcomp, arm_gravcomp=args.arm_gravcomp,
                 model_name=f"ur5e_{args.morph_run.name}")

    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_path(str(out))
    mujoco.MjModel.from_xml_path(str(ik_out))
    print(f"{out}  nq={m.nq} nu={m.nu} nbody={m.nbody}")
    print(f"{ik_out}")
    best, cands, d, mm = solve_home(out, ik_out, scene, seeds=args.ik_seeds)
    print(f"{len(cands)} distinct IK branches reach the design's open_ik palm pose")
    for c in sorted(cands, key=lambda c: c["score"]):
        print(f"   q {np.round(c['q'], 3)}  contacts {c['ncon']}  "
              f"lowest link z {c['zmin']:+.3f}  joint margin {c['margin']:.3f}"
              f"{'   <- chosen' if c is best else ''}")
    write_home(out, best["q"], d, mm)
    print(f"home written into {out}")

    if args.cl_assets:
        # The lab menagerie gets a SELF-CONTAINED pair: its own copy of the UR5e, and scenes
        # whose meshdir points at that copy rather than at whichever mujoco_menagerie clone
        # happened to be on the machine that generated them.
        dst = args.cl_assets / "universal_robots_ur5e"
        if not dst.exists():
            shutil.copytree(args.ur5e.parent, dst)
        a = args.cl_assets / f"ur5e_{args.morph_run.name}_scene.xml"
        b = args.cl_assets / f"ur5e_{args.morph_run.name}_ik.xml"
        build(scene, args.ur5e, a, b,
              base_xyz=tuple(float(v) for v in args.base.split(",")),
              gravcomp=args.gravcomp, arm_gravcomp=args.arm_gravcomp,
              meshdir=dst / "assets", model_name=f"ur5e_{args.morph_run.name}")
        best2, _, d2, m2 = solve_home(a, b, scene, seeds=args.ik_seeds)
        write_home(a, best2["q"], d2, m2)
        mujoco.MjModel.from_xml_path(str(a))
        print(f"-> {a}\n-> {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
