"""Export a real_v1 MuJoCo hand to the URDF + YAML pair that KaRMA reads.

KaRMA (arXiv:2605.15548, github.com/mfpeticco/karma-hand-metric) scores a hand from
its URDF joint tree alone: it models every finger link as a procedural capsule built
from the joint origins, so the URDF needs correct kinematics and nothing else. This
script reads a real_v1 MJCF -- the base hand, a generated design, or a scene -- and
writes the URDF plus the per-hand ``robot_<name>.yaml`` KaRMA needs.

Two facts about the export are worth stating because they set what the score means.

``tip_length_m`` is the distance from the terminal joint origin to the *outer surface*
of the fingertip, not to the pad centre: KaRMA subtracts its own capsule radius from
this value to get the medial axis (``robot.py`` line 354). For real_v1 that is
LINK_DISTAL = 37.16 mm, the CAD number, since the MJCF pad is a sphere of radius
10.55 mm centred 26.61 mm below the PIP axis.

The palm z offset is dropped. KaRMA canonicalizes the base link to the world origin,
so a rigid translation of the whole hand cannot change the score, and every real_v1
design carries its own palm height from ``fit_real_v1_pose.py``. Only the mount x/y
separations survive into the URDF, which is exactly the six-dimensional design space.

Usage:

    python scripts/karma_export_urdf.py \\
        --mjcf assets/mjcf/real_v1/real_hand.xml \\
        --name rv_base --pair thumb-index \\
        --out-dir external/karma/robots
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINGERS = ("thumb", "index", "middle")
PAIRS = {
    "thumb-index": ("thumb", "index"),
    "thumb-middle": ("thumb", "middle"),
    "index-middle": ("index", "middle"),
}


# ── MJCF reading ──────────────────────────────────────────────────────────────


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.replace(",", " ").split()]


class Segment:
    """One movable link of a real_v1 finger, as read out of the MJCF."""

    def __init__(self, joint: str, origin: list[float], axis: list[float],
                 lo: float, hi: float, cap_len: float, cap_r: float):
        self.joint = joint
        self.origin = origin
        self.axis = axis
        self.lo = lo
        self.hi = hi
        self.cap_len = cap_len      # MJCF capsule medial length (m), 0 if none
        self.cap_r = cap_r


def read_hand(mjcf: Path) -> dict:
    """Pull mounts, joint chains and pad geometry out of a real_v1 MJCF."""
    root = ET.parse(mjcf).getroot()
    bodies = {b.attrib.get("name"): b for b in root.iter("body")}

    hand: dict = {"fingers": {}, "mounts": {}, "source": str(mjcf)}
    for f in FINGERS:
        mount = bodies.get(f"{f}_mount")
        if mount is None:
            raise SystemExit(f"{mjcf}: no body named {f}_mount -- is this a real_v1 hand?")
        mx, my, _mz = _floats(mount.attrib.get("pos", "0 0 0"))
        hand["mounts"][f] = (mx, my)

        # Walk down the finger, collecting hinge joints and their capsules. The
        # morph slides (`<f>_x`, `<f>_y`, `<f>_len`) are design parameters, already
        # baked into the body positions of a generated design and zero-travel in the
        # base hand, so they are skipped rather than exported as URDF joints.
        segs: list[Segment] = []
        body = mount
        pending = [0.0, 0.0, 0.0]
        pad_offset = None
        while body is not None:
            child = None
            for sub in body.findall("body"):
                child = sub
                break
            for j in body.findall("joint"):
                if j.attrib.get("type") == "slide" or j.attrib.get("class") == "morph":
                    continue
                lo, hi = _floats(j.attrib["range"])
                cap_len, cap_r = 0.0, 0.0
                for g in body.findall("geom"):
                    if g.attrib.get("type") == "capsule":
                        ft = _floats(g.attrib["fromto"])
                        cap_len = math.dist(ft[:3], ft[3:])
                        cap_r = float(g.attrib["size"])
                segs.append(Segment(
                    joint=j.attrib["name"],
                    origin=list(pending),
                    axis=_floats(j.attrib["axis"]),
                    lo=lo, hi=hi, cap_len=cap_len, cap_r=cap_r,
                ))
                pending = [0.0, 0.0, 0.0]
            if body.attrib.get("name", "").endswith("_tip"):
                # `pending` already carries this body's own offset, accumulated on the
                # step that descended into it, so the pad centre is pending itself.
                pad_offset = list(pending)
                for g in body.findall("geom"):
                    if g.attrib.get("type") == "sphere":
                        hand.setdefault("pad_r", float(g.attrib["size"]))
                break
            if child is None:
                break
            pos = _floats(child.attrib.get("pos", "0 0 0"))
            pending = [pending[i] + pos[i] for i in range(3)]
            body = child

        if len(segs) != 3:
            raise SystemExit(f"{mjcf}: {f} has {len(segs)} hinge joints, expected 3")
        if pad_offset is None:
            raise SystemExit(f"{mjcf}: {f} has no *_tip body")
        # The pad centre, measured from the PIP joint origin, which is the frame
        # KaRMA measures tip_length_m in.
        hand["fingers"][f] = segs
        hand[f"{f}_pad"] = pad_offset
    return hand


# ── URDF writing ──────────────────────────────────────────────────────────────

_INERTIAL = (
    '    <inertial><origin xyz="0 0 -0.01"/><mass value="0.02"/>'
    '<inertia ixx="4e-6" ixy="0" ixz="0" iyy="4e-6" iyz="0" izz="1e-6"/></inertial>\n'
)


def write_urdf(hand: dict, name: str, path: Path) -> None:
    out = [f'<?xml version="1.0"?>\n<robot name="{name}">\n']
    out.append('  <link name="palm">\n')
    out.append('    <inertial><origin xyz="0 0 0"/><mass value="0.2"/>'
               '<inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/></inertial>\n')
    out.append('  </link>\n')

    for f in FINGERS:
        segs = hand["fingers"][f]
        mx, my = hand["mounts"][f]
        parent = "palm"
        for i, s in enumerate(segs):
            child = f"{f}_link{i + 1}"
            ox, oy, oz = s.origin
            if i == 0:
                ox, oy = ox + mx, oy + my
            out.append(f'  <link name="{child}">\n{_INERTIAL}  </link>\n')
            out.append(
                f'  <joint name="{s.joint}" type="revolute">\n'
                f'    <parent link="{parent}"/>\n'
                f'    <child link="{child}"/>\n'
                f'    <origin xyz="{ox:.6f} {oy:.6f} {oz:.6f}" rpy="0 0 0"/>\n'
                f'    <axis xyz="{s.axis[0]:.0f} {s.axis[1]:.0f} {s.axis[2]:.0f}"/>\n'
                f'    <limit lower="{s.lo:.6f}" upper="{s.hi:.6f}" effort="10" velocity="6"/>\n'
                f'  </joint>\n'
            )
            parent = child
    out.append("</robot>\n")
    path.write_text("".join(out))


def write_yaml(hand: dict, name: str, pair: str, urdf_rel: str, path: Path,
               tip_mode: str = "surface") -> float:
    a, b = PAIRS[pair]
    lines = [
        f"# Generated by scripts/karma_export_urdf.py from {hand['source']}\n",
        f"robot_name: {name}\n",
        f"urdf_path: {urdf_rel}\n",
        "base_link: palm\n",
        "fingers:\n",
    ]
    tip_len = 0.0
    for role, f in (("thumb", a), ("index", b)):
        segs = hand["fingers"][f]
        lines.append(f"  {role}:\n    active_joints:\n")
        for s in segs:
            lines.append(f"    - {s.joint}\n")
        lines.append("    contact_links:\n")
        for i in (3, 2, 1):
            lines.append(f"    - {f}_link{i}\n")
    lines.append(f"seed:\n  thumb_link: {a}_link3\n  index_link: {b}_link3\n")
    lines.append("links:\n")
    for f in FINGERS:
        pad = hand[f"{f}_pad"]
        centre = abs(pad[2])
        # KaRMA subtracts its own capsule radius from tip_length_m to get the medial
        # axis, so the value it wants is the distance to the OUTER pad surface.
        tip_len = centre + hand["pad_r"] if tip_mode == "surface" else centre
        lines.append(f"  {f}_link3:\n    tip_length_m: {tip_len:.6f}\n")
    path.write_text("".join(lines))
    return tip_len


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mjcf", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mounts", default=None,
                    help="override the MJCF's mount positions with six palm-frame "
                         "millimetres, 'tx,ty,ix,iy,mx,my'. The finger geometry still "
                         "comes from --mjcf, so this is the base hand re-mounted -- which "
                         "is exactly what a real_v1 design is.")
    ap.add_argument("--pair", default="thumb-index", choices=sorted(PAIRS))
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="KaRMA robots/ directory; the URDF lands in <out-dir>/urdfs/")
    ap.add_argument("--tip-mode", default="surface", choices=("surface", "centre"),
                    help="surface = pad outer surface (correct for KaRMA); "
                         "centre = pad centre (sensitivity check only)")
    args = ap.parse_args(argv)

    hand = read_hand(args.mjcf)
    if args.mounts:
        v = [float(x) * 1e-3 for x in args.mounts.split(",")]
        if len(v) != 6:
            raise SystemExit("--mounts needs six comma-separated millimetre values")
        hand["mounts"] = {"thumb": (v[0], v[1]), "index": (v[2], v[3]), "middle": (v[4], v[5])}
    out = args.out_dir
    (out / "urdfs").mkdir(parents=True, exist_ok=True)
    urdf = out / "urdfs" / f"{args.name}.urdf"
    write_urdf(hand, args.name, urdf)

    tag = args.pair.replace("-", "_")
    cfg = out / f"robot_{args.name}_{tag}.yaml"
    tip = write_yaml(hand, f"{args.name}_{tag}", args.pair, f"urdfs/{args.name}.urdf", cfg,
                     tip_mode=args.tip_mode)

    mounts = hand["mounts"]
    spread = max(math.dist(mounts[p], mounts[q])
                 for p in FINGERS for q in FINGERS if p < q)
    print(f"{args.name}: mounts(mm) " + " ".join(
        f"{f}=({mounts[f][0] * 1e3:+.1f},{mounts[f][1] * 1e3:+.1f})" for f in FINGERS))
    print(f"  max pairwise mount span {spread * 1e3:.1f} mm, tip_length {tip * 1e3:.2f} mm")
    print(f"  wrote {urdf}\n  wrote {cfg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
