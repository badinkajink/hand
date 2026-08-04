#!/usr/bin/env python3
"""Rewrite a frozen scene's collision primitives as a dense sphere packing.

    uv run python scripts/generate_sphere_packed_scene.py \
        --scene results/phase1/landscape/m05_ik_cem/frozen_scene.xml \
        --pack hand --eps 0.01 \
        --out results/spherepack/20260803-m05_handpack_eps0.010/frozen_scene.xml

WHY A SEPARATE SCRIPT AND NOT AN EDIT TO THE SOURCE MJCF. The frozen per-design scene is what
RL and eval actually load, so rewriting *it* keeps capsule-vs-packed as a clean A/B pair of
files. It also sidesteps two landmines in the source tree: `scripts/perp_phalanx_variant.py`
hard-asserts exactly three capsule `fromto` strings, and `tests/test_perp_hand_scene_parity.py`
pins `perp_hand.xml` byte-for-byte against the scene.

THE TWO ARMS ARE SEPARATE ON PURPOSE (`--pack hand` / `--pack object`):

  hand    - gives the HAND a contact map; the object keeps its exact analytic surface, so
            rolling geometry is untouched. Contact count stays linear in hand spheres because
            sphere-vs-cylinder is one contact per pair.
  object  - gives the OBJECT a contact map in tool coordinates, which is what makes "where is
            the contact on the screwdriver, over time" a well-defined signal. Higher physics
            risk: a sphere-packed cylinder is a SCALLOPED cylinder and rolls differently from
            a smooth one. Do not trust an arm-object policy result before the free-roll probe.

Packing both at once is not offered: contact count would go quadratic for no gain in what we
can see.

THE MASS TRAP (this is why the script compiles the scene instead of just editing XML).
In the frozen scenes the finger links and the screwdriver carry no explicit `<inertial>` --
their mass comes from geom volume x density. Overlapping spheres each contribute full volume,
so one capsule -> 18 spheres inflates the link mass ~18x, and the resulting "sphere packing
broke the grasp" would be a mass result wearing a representation costume. So we compile the
UNPACKED scene, read `model.body_mass` / `body_inertia` / `body_ipos` / `body_iquat`, and write
those out as an explicit `<inertial>` on every body we touch. MuJoCo then ignores geom-derived
inertia for that body and mass is preserved exactly.

What this script deliberately does NOT do: renormalize contact stiffness. N spheres pressing
in parallel are ~N times stiffer than one capsule contact at the same penetration, which is a
real and separate confound. That correction is measured by `scripts/probe_contact_stiffness.py`
and applied via `--solref/--solimp`, so the geometry step stays honest and inspectable.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from morphohand.geometry.sphere_pack import (
    Sphere,
    pack_box,
    pack_capsule,
    pack_cylinder,
    pack_sphere,
)

FINGERS = ("thumb", "index", "middle")
LINKS = ("mcp_frame", "len_frame", "pip_frame", "tip")
HAND_BODIES = tuple(f"{f}_{l}" for f in FINGERS for l in LINKS)

# Geom attributes that describe appearance/contact and must ride along onto every sphere.
# `friction` matters most: the tips override the scene default with "2.4 0.2 0.02", and
# dropping it would silently make the fingertips slippery.
CARRY_ATTRS = ("material", "rgba", "friction", "class", "condim", "priority", "margin", "gap")


def _pack_geom(elem: ET.Element, eps: float, frac: float) -> list[Sphere] | None:
    """Sphere-pack one geom element. Returns None for shapes we leave alone (planes, meshes)."""
    gtype = elem.get("type", "sphere")
    size = [float(v) for v in (elem.get("size") or "").split()]

    if gtype == "capsule":
        ft = elem.get("fromto")
        if ft is not None:
            vals = tuple(float(v) for v in ft.split())
            return pack_capsule(vals, size[0], eps)  # type: ignore[arg-type]
        # pos/quat-style capsule: size = (radius, half_length) along local +Z
        half = size[1]
        return pack_capsule((0, 0, -half, 0, 0, half), size[0], eps)
    if gtype == "sphere":
        return pack_sphere(size[0], eps)
    if gtype == "cylinder":
        return pack_cylinder(size[0], size[1], eps, sphere_radius_frac=frac)
    if gtype == "box":
        return pack_box((size[0], size[1], size[2]), eps, sphere_radius_frac=frac)
    return None


def _compiled_inertials(scene: Path) -> dict[str, dict[str, str]]:
    """Compile the UNPACKED scene and read out each body's inertial, ready to write back."""
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


def _local_offset(elem: ET.Element) -> np.ndarray:
    """Geom-local origin. Sphere centres from the packer are in the geom's own frame; for a
    `fromto` capsule that frame is already the body frame, but a pos/size capsule carries its
    own offset that we must re-add."""
    if elem.get("fromto") is not None:
        return np.zeros(3)
    return np.array([float(v) for v in (elem.get("pos") or "0 0 0").split()])


def pack_scene(
    scene: Path,
    out: Path,
    *,
    which: str,
    eps: float,
    frac: float,
    include_palm: bool,
) -> dict:
    inertials = _compiled_inertials(scene)
    tree = ET.parse(scene)
    root = tree.getroot()

    if which == "hand":
        targets = set(HAND_BODIES) | ({"palm_pose"} if include_palm else set())
    else:
        # the object body is the one carrying a freejoint
        targets = {
            b.get("name", "")
            for b in root.iter("body")
            if b.find("freejoint") is not None or b.find("joint[@type='free']") is not None
        }
        targets.discard("")
        if not targets:
            raise SystemExit("--pack object: no body with a freejoint found in the scene")

    manifest: dict[str, int] = {}
    for body in root.iter("body"):
        name = body.get("name")
        if name not in targets:
            continue

        geoms = [g for g in body.findall("geom")]
        # Never pack a non-colliding decoration (visual stalks carry contype/conaffinity 0):
        # it would triple the geom count for shapes that can never produce a contact.
        packable = [
            g for g in geoms
            if not (g.get("contype") == "0" and g.get("conaffinity") == "0")
        ]
        if not packable:
            continue

        # Bake inertia BEFORE substituting geoms -- see the module docstring's mass trap.
        if body.find("inertial") is None and name in inertials:
            body.insert(0, ET.Element("inertial", inertials[name]))

        n_new = 0
        for g in packable:
            spheres = _pack_geom(g, eps, frac)
            if spheres is None:
                continue
            carried = {k: g.get(k) for k in CARRY_ATTRS if g.get(k) is not None}
            off = _local_offset(g)
            idx = list(body).index(g)
            body.remove(g)
            for j, s in enumerate(spheres):
                p = np.asarray(s.pos) + off
                attrs = {
                    "type": "sphere",
                    "size": f"{s.radius:.9g}",
                    "pos": " ".join(f"{v:.9g}" for v in p),
                    **carried,
                }
                body.insert(idx + j, ET.Element("geom", attrs))
            n_new += len(spheres)
        manifest[name] = n_new

    # A packed scene must never inherit geom-derived mass anywhere we touched.
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="unicode", xml_declaration=False)

    # Compile the RESULT and verify mass survived. A silent 18x mass change is exactly the
    # failure this script exists to prevent, so it is checked, not assumed.
    m_new = mujoco.MjModel.from_xml_path(str(out))
    m_old = mujoco.MjModel.from_xml_path(str(scene))
    drift = {}
    for bid in range(m_old.nbody):
        nm = mujoco.mj_id2name(m_old, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not nm or nm not in manifest:
            continue
        nid = mujoco.mj_name2id(m_new, mujoco.mjtObj.mjOBJ_BODY, nm)
        rel = abs(m_new.body_mass[nid] - m_old.body_mass[bid]) / max(m_old.body_mass[bid], 1e-12)
        drift[nm] = float(rel)

    report = {
        "source_scene": str(scene),
        "out_scene": str(out),
        "pack": which,
        "eps": eps,
        "sphere_radius_frac": frac,
        "spheres_per_body": manifest,
        "total_spheres": sum(manifest.values()),
        "ngeom_before": int(m_old.ngeom),
        "ngeom_after": int(m_new.ngeom),
        "max_rel_mass_drift": max(drift.values()) if drift else 0.0,
        "rel_mass_drift": drift,
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True, help="frozen scene XML to pack")
    ap.add_argument("--out", type=Path, required=True, help="output packed scene XML")
    ap.add_argument("--pack", choices=("hand", "object"), required=True)
    ap.add_argument("--eps", type=float, default=0.01,
                    help="surface tolerance, relative to each shape's radius (default 0.01)")
    ap.add_argument("--sphere-radius-frac", type=float, default=0.5,
                    help="shell sphere radius / shape radius, for cylinder+box only")
    ap.add_argument("--include-palm", action="store_true",
                    help="also pack the palm slab (hand arm only)")
    args = ap.parse_args()

    rep = pack_scene(
        args.scene, args.out,
        which=args.pack, eps=args.eps,
        frac=args.sphere_radius_frac, include_palm=args.include_palm,
    )
    print(f"packed {rep['pack']}  eps={rep['eps']}  "
          f"ngeom {rep['ngeom_before']} -> {rep['ngeom_after']}  "
          f"spheres={rep['total_spheres']}")
    for b, n in rep["spheres_per_body"].items():
        print(f"   {b:<20s} {n:4d}")
    drift = rep["max_rel_mass_drift"]
    status = "OK" if drift < 1e-6 else "!! MASS DRIFT"
    print(f"max relative mass drift: {drift:.3e}  [{status}]")
    if drift >= 1e-6:
        raise SystemExit("mass was not preserved -- do not use this scene")


if __name__ == "__main__":
    main()
