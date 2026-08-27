"""Perturb a FROZEN evaluation scene along one physical axis at a time.

Every sim2real question in this program is "the real thing differs from the scene by X — does
the policy survive?", and every fingertip question is "what if the tip were shaped differently?".
Both are edits to the frozen per-design scene, so they live here together rather than as two
regex piles inside two sweep scripts. Editing the frozen scene (not the source MJCF) keeps each
variant a standalone file that `deploy.make_env_cfg` can load unchanged, which is what makes
"same policy, one axis moved" a clean comparison.

THE MASS TRAP (same one `generate_sphere_packed_scene.py` documents). Finger links and the
screwdriver carry no explicit `<inertial>` in the frozen scenes — mass comes from geom volume x
density. So swapping a 5 mm capsule tip for an 8 mm one silently changes the link's mass and
inertia, and a "bigger tips grip better" result would be a mass result wearing a geometry
costume. Any mutator that touches a geom's SHAPE therefore pins the original body inertials
first via `compiled_inertials`, so the only thing that changed is the contact surface.
Mutators that change mass ON PURPOSE (`scale_object_density`) skip that pinning for their own
body, which is the point of them.

Conventions in these scenes, relied on below:
  - the object body is `screwdriver_medium`, a cylinder whose axis lies along world Y when flat;
  - the fingertip bodies are `<finger>_tip`, each holding exactly one collision geom;
  - the tip's parent phalanx runs along local +x, so the tip geom's local +x points "along the
    finger" and local +y points across it. Measured at the `open_ik` grasp keyframe, every tip's
    local +y sits within 0.96-1.00 of world Y, and the flat shaft's axis IS world -Y. So local +y
    is the SHAFT direction and local +x crosses it — which is why the shape names below are
    stated relative to the shaft (`cap_line` runs along it, `cap_cross` crosses it) rather than
    relative to the finger. Shaft-relative is the property that decides rolling vs holding.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

TIP_BODIES = ("thumb_tip", "index_tip", "middle_tip")
PROXIMAL_BODIES = ("thumb_mcp_frame", "index_mcp_frame", "middle_mcp_frame")
OBJECT_BODY = "screwdriver_medium"


# --------------------------------------------------------------------------- inertia pinning
def compiled_inertials(scene: Path | str) -> dict[str, dict[str, str]]:
    """Compile a scene and read back every body's inertial as writable XML attributes.

    Written into the mutated scene so MuJoCo stops deriving mass from geom volume for the
    bodies we reshape. See THE MASS TRAP above.
    """
    m = mujoco.MjModel.from_xml_path(str(scene))
    out: dict[str, dict[str, str]] = {}
    for bid in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name:
            continue
        out[name] = {
            "pos": " ".join(f"{v:.9g}" for v in m.body_ipos[bid]),
            "quat": " ".join(f"{v:.9g}" for v in m.body_iquat[bid]),
            "mass": f"{m.body_mass[bid]:.9g}",
            "diaginertia": " ".join(f"{v:.9g}" for v in m.body_inertia[bid]),
        }
    return out


def _pin_inertial(body: ET.Element, inertials: dict[str, dict[str, str]]) -> None:
    name = body.get("name")
    if name in inertials and body.find("inertial") is None:
        body.insert(0, ET.Element("inertial", inertials[name]))


def _bodies(root: ET.Element, names) -> list[ET.Element]:
    want = set(names)
    return [b for b in root.iter("body") if b.get("name") in want]


# --------------------------------------------------------------------------- tip shape family
# WHERE THE OBJECT ACTUALLY IS, measured on the shipped scene at `open_ik` rather than assumed:
# the shaft's nearest point sits at tip-local (+10.4, 0.0, -2.4) mm for index and middle, and
# (+6.4, 0.2, +5.0) for the thumb. So contact arrives off the tip's **+x end**, and a shape that
# only extends sideways is swallowed by the distal capsule (r=7.5 mm, its end cap centred on the
# tip body origin) and never touches the object at all. The first render of this family showed
# exactly that: six of eight shapes were buried and only the shipped one protruded.
#
# Hence REACH NORMALISATION. Every shape is translated along +x so its forward-most surface
# point sits at `reach` from the tip origin. `reach` defaults to the shipped tip's 11 mm
# (half-length 6 + radius 5), so `cap_cross` reproduces the shipped geometry with zero shift and
# every other shape presents its pad to the shaft from the same distance. Without this, a shape
# comparison is a reach comparison — the same bug as joint-space keyframe transfer across
# morphologies, in a new coordinate.
#
# NOTE ON THE SHIPPED TIP. `cap_cross` is a capsule lying along +x, so what meets the shaft is
# its hemispherical END CAP, not its cylindrical side: the measured contact point is 5.01 mm
# from the cap centre, i.e. exactly on the hemisphere. The hand we ship is therefore already
# making SPHERICAL point contact at r=5 mm. The alternatives below are the ways out of that.
SHIPPED_TIP = "cap_cross"
SHIPPED_REACH = 0.011          # shipped half-length 6 mm + radius 5 mm


def _parse_xyz(spec: str) -> tuple[float, float, float]:
    v = [float(t) for t in (spec or "0 0 0").split()]
    return v[0], v[1], v[2]


def _fmt_xyz(v) -> str:
    return " ".join(f"{c:.6f}" for c in v)


def _shift_first(spec: str | None, x: float) -> str:
    """Replace a body pos's x while keeping its y/z."""
    _, y, z = _parse_xyz(spec or "0 0 0")
    return _fmt_xyz((x, y, z))


def _set_capsule(body: ET.Element, length: float) -> None:
    """Redraw a body's single `fromto` capsule as `0 0 0 -> length 0 0`, radius untouched."""
    caps = [g for g in body.findall("geom")
            if g.get("type") == "capsule" and g.get("fromto") is not None]
    if len(caps) != 1:
        raise ValueError(f"{body.get('name')} has {len(caps)} fromto capsules, expected 1")
    caps[0].set("fromto", f"0 0 0 {length:.6f} 0 0")


def _shift_x(spec: dict[str, str], dx: float) -> dict[str, str]:
    if "fromto" in spec:
        v = [float(t) for t in spec["fromto"].split()]
        v[0] += dx
        v[3] += dx
        spec["fromto"] = " ".join(f"{t:.6f}" for t in v)
    else:
        p = [float(t) for t in (spec.get("pos") or "0 0 0").split()]
        p[0] += dx
        spec["pos"] = " ".join(f"{t:.6f}" for t in p)
    return spec


def _tip_family(shape: str, r: float, h: float) -> tuple[list[dict[str, str]], float]:
    """(geom specs in the tip frame, forward extent along +x before normalisation).

    +x is the approach direction, +y runs along the shaft, +z is the in-pad transverse axis.
    """
    if shape == "sphere":                          # point contact, isotropic — the null design
        return [{"type": "sphere", "size": f"{r:.6f}"}], r
    if shape == "cap_cross":                       # SHIPPED: end cap meets the shaft (a sphere)
        return [{"type": "capsule", "fromto": f"{-h:.6f} 0 0 {h:.6f} 0 0",
                 "size": f"{r:.6f}"}], h + r
    if shape == "cap_line":                        # capsule ALONG the shaft -> line contact
        return [{"type": "capsule", "fromto": f"0 {-h:.6f} 0 0 {h:.6f} 0",
                 "size": f"{r:.6f}"}], r
    if shape == "ellipsoid":                       # human-like pad: broad along the shaft
        return [{"type": "ellipsoid", "size": f"{r:.6f} {h:.6f} {r * 0.8:.6f}"}], r
    if shape == "pad_flat":                        # flat printed pad, +x face meets the shaft
        return [{"type": "box", "size": f"{r * 0.6:.6f} {h:.6f} {h:.6f}"}], r * 0.6
    if shape == "cylinder_line":                   # flat-ended roller: line contact + hard edges
        return [{"type": "cylinder", "fromto": f"0 {-h:.6f} 0 0 {h:.6f} 0",
                 "size": f"{r:.6f}"}], r
    # Grooved tips: two ridges straddling the shaft instead of meeting it at one point. Both
    # ridge pairs sit in the pad plane (constant x) and differ only in which way they run:
    #   groove_cradle - ridges ALONG the shaft, separated in z; the shaft seats in the channel
    #                   between them, held against sideways escape but still free to roll.
    #   groove_bite   - ridges ACROSS the shaft, separated in y; they obstruct the roll while
    #                   leaving the shaft free to slide along its own axis.
    # The pair IS the turn-vs-hold question posed as geometry, so both ship and the probe decides.
    if shape in ("groove_cradle", "groove_bite"):
        s = r * 0.6                                # ridge radius
        off = r * 0.9                              # half-separation of the two ridges
        if shape == "groove_cradle":
            return [{"type": "capsule", "size": f"{s:.6f}",
                     "fromto": f"0 {-h:.6f} {sgn * off:.6f} 0 {h:.6f} {sgn * off:.6f}"}
                    for sgn in (-1, 1)], s
        return [{"type": "capsule", "size": f"{s:.6f}",
                 "fromto": f"0 {sgn * off:.6f} {-h:.6f} 0 {sgn * off:.6f} {h:.6f}"}
                for sgn in (-1, 1)], s
    raise ValueError(f"unknown tip shape {shape!r}")


def tip_geoms(shape: str, r: float = 0.005, h: float = 0.006,
              reach: float = SHIPPED_REACH) -> list[dict[str, str]]:
    """Geom attribute dicts for one fingertip, reach-normalised. Raises on an unknown shape."""
    specs, extent = _tip_family(shape, r, h)
    return [_shift_x(s, reach - extent) for s in specs]


TIP_SHAPES = ("sphere", "cap_cross", "cap_line", "ellipsoid",
              "pad_flat", "cylinder_line", "groove_cradle", "groove_bite")


# --------------------------------------------------------------------------- mutators
class Scene:
    """A frozen scene held open for mutation. Chain mutators, then `write`."""

    def __init__(self, path: Path | str):
        self.src = Path(path)
        # insert_comments: a plain ET.parse silently drops every XML comment, and the scenes
        # this gets pointed at carry their design rationale (contact excludes, mount rationale)
        # in comments. A mutated scene that lost them is a scene nobody can review.
        self.tree = ET.parse(self.src, ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
        self.root = self.tree.getroot()
        self._inertials: dict[str, dict[str, str]] | None = None

    @property
    def inertials(self) -> dict[str, dict[str, str]]:
        if self._inertials is None:                 # compiled from the UNMUTATED source
            self._inertials = compiled_inertials(self.src)
        return self._inertials

    # -- contact model ------------------------------------------------------
    def set_solimp(self, dmin: float, dmax: float, width: float = 0.0005) -> "Scene":
        """Contact stiffness. dmax up = harder (less penetration) = further from a soft pad."""
        for g in self.root.iter("geom"):
            if g.get("solimp") is not None:
                g.set("solimp", f"{dmin} {dmax} {width}")
        return self

    def set_solref(self, timeconst: float, damping: float = 1.0) -> "Scene":
        for g in self.root.iter("geom"):
            if g.get("solref") is not None:
                g.set("solref", f"{timeconst} {damping}")
        return self

    # -- friction -----------------------------------------------------------
    def scale_friction(self, factor: float, *, tips: bool = True, obj: bool = True) -> "Scene":
        """Scale the SLIDING coefficient on the contacting surfaces.

        Only the sliding term moves: torsional/rolling friction are separate physical claims
        about the pad and get their own knob rather than riding along with a material change.
        """
        targets = []
        if tips:
            targets += [g for b in _bodies(self.root, TIP_BODIES) for g in b.iter("geom")]
        if obj:
            targets += [g for b in _bodies(self.root, [OBJECT_BODY]) for g in b.iter("geom")]
        for g in targets:
            fr = (g.get("friction") or "1.8 0.15 0.01").split()
            fr[0] = f"{float(fr[0]) * factor:.6g}"
            g.set("friction", " ".join(fr))
        return self

    def set_tip_friction(self, slide: float, torsion: float | None = None,
                         roll: float | None = None) -> "Scene":
        for b in _bodies(self.root, TIP_BODIES):
            for g in b.iter("geom"):
                fr = (g.get("friction") or "1.8 0.15 0.01").split()
                fr[0] = f"{slide:.6g}"
                if torsion is not None:
                    fr[1] = f"{torsion:.6g}"
                if roll is not None:
                    fr[2] = f"{roll:.6g}"
                g.set("friction", " ".join(fr))
        return self

    # -- object -------------------------------------------------------------
    def scale_object_density(self, factor: float) -> "Scene":
        """Heavier/lighter tool. Deliberately changes mass, so nothing is pinned here."""
        for b in _bodies(self.root, [OBJECT_BODY]):
            for g in b.iter("geom"):
                g.set("density", f"{float(g.get('density') or 1000.0) * factor:.6g}")
        return self

    def scale_object_radius(self, factor: float) -> "Scene":
        """Fatter/thinner shaft at FIXED mass — a grip-geometry change, not a load change."""
        for b in _bodies(self.root, [OBJECT_BODY]):
            _pin_inertial(b, self.inertials)
            for g in b.iter("geom"):
                size = [float(v) for v in (g.get("size") or "").split()]
                if len(size) == 2:                  # cylinder: (radius, half-length)
                    g.set("size", f"{size[0] * factor:.6g} {size[1]:.6g}")
        return self

    # -- proximal phalanx ---------------------------------------------------
    def shorten_proximal(self, *, pin_mass: bool = True) -> "Scene":
        """Trim each proximal collision capsule back to its own KINEMATIC length.

        The generator writes a fixed 50 mm capsule into `<f>_mcp_frame` no matter what the
        design's proximal length is; the length parameter only moves where `<f>_len_frame`
        attaches. On m05 that leaves 9-14 mm where the middle phalanx sits INSIDE the proximal
        capsule (25 mm on the short base) — not manufacturable, and load-bearing, because at grip
        force a share of the load runs through the sleeve rather than the pads
        (`docs/notes/finger_spec.md` §3a).

        The fix is to draw the proximal by the same rule the middle and distal phalanges already
        follow: `fromto 0 0 0 L 0 0` with L the kinematic length, read per finger off the child
        `<f>_len_frame`'s x-offset. The hemispherical end cap still extends r past the joint,
        exactly as the middle's and distal's do — this makes the proximal consistent, it does not
        invent a new convention.

        `pin_mass` (default) holds the link's compiled mass/inertia at the 50 mm value so the
        only thing that moved is the contact surface (THE MASS TRAP). A real shortened link is
        also lighter; that is a separate axis and gets its own point rather than riding along.
        """
        for b in _bodies(self.root, PROXIMAL_BODIES):
            child = next((c for c in b.findall("body")
                          if (c.get("name") or "").endswith("_len_frame")), None)
            if child is None:
                raise ValueError(f"{b.get('name')} has no _len_frame child to read length from")
            length = float((child.get("pos") or "0 0 0").split()[0])
            if pin_mass:
                _pin_inertial(b, self.inertials)
            for g in b.findall("geom"):
                if g.get("type") == "capsule" and g.get("fromto") is not None:
                    g.set("fromto", f"0 0 0 {length:.6f} 0 0")
        return self

    def set_proximal_length(self, length: float, *, pin_mass: bool = True) -> "Scene":
        """Set the proximal phalanx to `length` metres — capsule AND kinematics together.

        `shorten_proximal` fixes a DRAWING inconsistency and leaves the robot's kinematics
        alone. This changes the robot: it moves `<f>_len_frame` as well, so the whole finger
        gets shorter and every reach shell moves with it. The 25 mm proximal is the short base
        the m05 lineage is built on (`docs/notes/finger_spec.md` §2), and a hand built on it
        reaches ~25 mm less far — a scene mutated this way generally needs its keyframes
        re-solved and its palm re-seated before it can touch the object at all.
        """
        for b in _bodies(self.root, PROXIMAL_BODIES):
            child = next((c for c in b.findall("body")
                          if (c.get("name") or "").endswith("_len_frame")), None)
            if child is None:
                raise ValueError(f"{b.get('name')} has no _len_frame child")
            if pin_mass:
                _pin_inertial(b, self.inertials)
            for g in b.findall("geom"):
                if g.get("type") == "capsule" and g.get("fromto") is not None:
                    g.set("fromto", f"0 0 0 {length:.6f} 0 0")
            pos = (child.get("pos") or "0 0 0").split()
            child.set("pos", f"{length:.6f} {pos[1]} {pos[2]}")
        return self

    def set_link_lengths(self, proximal: float, distal: float, *, taper: float = 0.4,
                         pad_reach: float = 0.011, yaw_link: float = 0.0,
                         yaw_link_dir: tuple[float, float, float] = (0.0, 0.0, -1.0)) -> "Scene":
        """Set the two FLEXING link lengths, in the hardware's own terms.

        The hand has two flexion joints per finger, MCP and PIP, so it has two links — but the
        scene draws the MCP link as two capsules (`<f>_mcp_frame` then `<f>_len_frame`) with no
        joint between them, purely so the co-design `len` parameter had somewhere to slide.
        `set_proximal_length` moves only the first of those, which is why "the 25 mm hand" is
        really a 65 mm MCP->PIP link. This sets the links themselves:

          proximal = MCP axis -> PIP axis          (m05: 75.8 / 77.3 / 80.9 mm)
          distal   = PIP axis -> pad front surface (m05: 41.0 mm = 30 mm link + 11 mm pad reach)

        `taper` keeps the drawn cross-section stepping down the way the shipped hand does — the
        MCP link stays two capsules (r 10.0 then r 8.5) splitting `proximal` at that fraction. The
        fingertip pad is left alone: it is the r 5 mm convex tip the fingertip study selected, and
        `pad_reach` (6 mm half-length + 5 mm radius) is how far it sits past the distal link.

        `yaw_link` is the THIRD segment the hardware has and this scene does not: the link from
        the yaw joint out to the MCP joint, run along the finger's +x. These scenes put both
        joints at the same point.

        `yaw_axis` resets what the yaw joint rotates about, and the two only mean something
        together. The shipped scenes use (1,0,0) — the finger's OWN long axis — so yaw is a ROLL
        that turns the flexion plane, and a link laid along that same axis is kinematically
        inert: the MCP joint sits on the axis and yaw cannot move it. Pass (0,0,1) for a true
        abduction about the palm normal, where the link is perpendicular to the axis and yaw
        sweeps the MCP joint through an arc of that radius. That is a different robot, not a
        repositioned one — the whole a10/b33 lineage is roll-yaw — so it needs its keyframes
        re-solved and its policies retrained, not transferred.

        Mass is deliberately NOT pinned here (contrast the tip-shape mutators and THE MASS TRAP
        above): a shorter phalanx really is lighter, and pinning m05's inertia onto a 40 mm link
        would be the fiction, not the control.
        """
        for mcp_name in PROXIMAL_BODIES:
            f = mcp_name.split("_")[0]
            mcp = _bodies(self.root, (mcp_name,))[0]
            lenf = next(c for c in mcp.findall("body") if (c.get("name") or "").endswith("_len_frame"))
            pip = next(c for c in lenf.findall("body") if (c.get("name") or "").endswith("_pip_frame"))
            tip = next(c for c in pip.findall("body") if (c.get("name") or "").endswith("_tip"))

            p1, p2 = taper * proximal, (1.0 - taper) * proximal
            d_link = distal - pad_reach
            if min(p1, p2, d_link) <= 0:
                raise ValueError(f"{f}: proximal {proximal} / distal {distal} leaves a "
                                 f"non-positive segment (distal must exceed pad reach {pad_reach})")
            _set_capsule(mcp, p1)
            lenf.set("pos", _shift_first(lenf.get("pos"), p1))
            _set_capsule(lenf, p2)
            pip.set("pos", _shift_first(pip.get("pos"), p2))
            _set_capsule(pip, d_link)
            tip.set("pos", _shift_first(tip.get("pos"), d_link))

            if yaw_link:
                yawb = _bodies(self.root, (f"{f}_yaw_frame",))[0]
                jnt = next(j for j in yawb.findall("joint") if j.get("name") == f"{f}_yaw")
                ax = np.asarray(_parse_xyz(jnt.get("axis", "1 0 0")), dtype=float)
                dirn = np.asarray(yaw_link_dir, dtype=float)
                dirn = dirn / np.linalg.norm(dirn)
                if abs(float(np.dot(ax / np.linalg.norm(ax), dirn))) > 1e-6:
                    raise ValueError(
                        f"{f}: yaw_link_dir {yaw_link_dir} is not perpendicular to the yaw axis "
                        f"{tuple(ax)}. A link along the axis leaves the MCP joint ON it, which is "
                        f"the coincident-joint bug this parameter exists to fix.")
                x, y, z = _parse_xyz(mcp.get("pos", "0 0 0"))
                end = np.array([x, y, z]) + yaw_link * dirn
                mcp.set("pos", _fmt_xyz(end))
                # DRAW the link. Without a geom the segment exists in the kinematics and nowhere
                # else: the viewer shows the finger starting in mid-air below the palm, which
                # reads as the two joints being coincident even though they are 32 mm apart. It
                # also has to be here for the mass and the collision volume to be real, so the
                # placeholder inertial on the yaw frame goes and density carries it like every
                # other link.
                for old in yawb.findall("geom"):
                    yawb.remove(old)
                for placeholder in yawb.findall("inertial"):
                    yawb.remove(placeholder)
                yawb.append(ET.Element("geom", {
                    "type": "capsule", "fromto": f"0 0 0 {end[0]:.6f} {end[1]:.6f} {end[2]:.6f}",
                    "size": "0.010", "material": "finger_mat"}))
        return self

    def set_mounts(self, mounts: dict[str, tuple[float, float]]) -> "Scene":
        """Move each finger's mount to (x, y) in the palm frame.

        Mount x/y are two thirds of the nine co-design free variables, and freezing them at one
        hand's values while changing the link lengths is the same error as inheriting a palm pose
        or a joint-space keyframe: the fingers get retargeted and the thing they hang off does
        not. A hand with 70 mm fingers wants a tighter footprint than one with 117 mm fingers.
        """
        for finger, (x, y) in mounts.items():
            b = _bodies(self.root, (f"{finger}_mount",))[0]
            _, _, z = _parse_xyz(b.get("pos", "0 0 0"))
            b.set("pos", _fmt_xyz((x, y, z)))
        return self

    def set_tool_length(self, length: float) -> "Scene":
        """Set the screwdriver stand-in's length, radius untouched.

        Mass rides along on purpose (density x volume): a shorter shaft IS lighter, and the
        robustness sweep found the hand one-sidedly fragile to a lighter tool (x0.6 -> hold 0.41)
        while a heavier one was fine. So shortening the shaft to buy headroom spends margin on
        the axis that is already the weak one, and the two effects have to be read together.
        """
        for b in _bodies(self.root, (OBJECT_BODY,)):
            for g in b.findall("geom"):
                if g.get("type") == "cylinder":
                    r, _ = [float(t) for t in (g.get("size") or "0 0").split()]
                    g.set("size", f"{r:.6f} {length / 2:.6f}")
        return self

    # -- fingertips ---------------------------------------------------------
    def set_tip_shape(self, shape: str, r: float = 0.005, h: float = 0.006,
                      reach: float = SHIPPED_REACH) -> "Scene":
        """Replace every fingertip's collision geom with `shape`, at fixed link mass and reach.

        Material properties (friction/solimp/class) are carried over from the geom being
        replaced so a shape comparison is not secretly a friction comparison.
        """
        specs = tip_geoms(shape, r=r, h=h, reach=reach)
        for b in _bodies(self.root, TIP_BODIES):
            _pin_inertial(b, self.inertials)
            old = [g for g in b.findall("geom")]
            if not old:
                continue
            carry = {k: v for k, v in old[0].attrib.items()
                     if k in ("material", "friction", "solimp", "solref", "class", "rgba",
                              "condim", "priority", "margin", "gap")}
            for g in old:
                b.remove(g)
            for spec in specs:
                b.append(ET.Element("geom", {**carry, **spec}))
        return self

    # -- output -------------------------------------------------------------
    def write(self, out: Path | str) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.tree.write(out, encoding="unicode")
        return out


def mass_check(before: Path | str, after: Path | str, tol: float = 1e-6) -> dict[str, float]:
    """Bodies whose mass moved between two scenes, as a relative delta.

    Call it after any shape mutation: a non-empty result on a shape-only edit means the
    inertial pinning missed a body and the comparison is contaminated.
    """
    a = mujoco.MjModel.from_xml_path(str(before))
    b = mujoco.MjModel.from_xml_path(str(after))
    out: dict[str, float] = {}
    for bid in range(a.nbody):
        name = mujoco.mj_id2name(a, mujoco.mjtObj.mjOBJ_BODY, bid)
        nid = mujoco.mj_name2id(b, mujoco.mjtObj.mjOBJ_BODY, name) if name else -1
        if nid < 0:
            continue
        m0 = float(a.body_mass[bid])
        rel = abs(float(b.body_mass[nid]) - m0) / max(m0, 1e-12)
        if rel > tol:
            out[name] = round(rel, 6)
    return out


def tip_contact_geoms(scene: Path | str) -> dict[str, list[int]]:
    """Geom ids per fingertip body — the probes need them to attribute contacts to a finger."""
    m = mujoco.MjModel.from_xml_path(str(scene))
    out: dict[str, list[int]] = {}
    for name in TIP_BODIES:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        out[name] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    return out


def object_axis_world(scene: Path | str) -> np.ndarray:
    """Unit world-frame axis of the object cylinder in its spawn pose (the shaft direction)."""
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
    return np.array(d.xmat[bid]).reshape(3, 3)[:, 2]     # cylinder axis is local +z
