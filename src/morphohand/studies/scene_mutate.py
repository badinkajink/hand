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
