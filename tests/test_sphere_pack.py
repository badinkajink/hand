# pyright: reportMissingImports=false
"""The sphere packing must change the contact representation and nothing else.

Two things make a packed scene silently wrong, and both look like physics results rather than
bugs, so both are pinned here:

1. **Mass.** The frozen scenes carry no explicit ``<inertial>`` on the finger links or the
   object -- mass is computed from geom volume x density. Overlapping spheres each contribute
   full volume, so a naive substitution inflates link mass by roughly the sphere count. A
   "sphere packing broke the grasp" caused by an 18x heavier finger is indistinguishable from a
   real finding until someone checks.
2. **Shape.** The packed surface must stay within the requested tolerance of the original. A
   packing that is uniformly thin reads downstream as a looser grasp.

The committed viewable models under ``assets/mjcf/spherepack/`` are also checked against
regeneration, so they cannot drift from the sources they mirror.
"""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from morphohand.geometry.sphere_pack import (
    chain_spacing,
    pack_box,
    pack_capsule,
    pack_cylinder,
)

ROOT = Path(__file__).resolve().parents[1]
PACKED_DIR = ROOT / "assets" / "mjcf" / "spherepack"
VIEWABLE = {
    PACKED_DIR / "perp_hand_packed.xml": ROOT / "assets" / "mjcf" / "perp" / "perp_hand.xml",
    PACKED_DIR / "hand_packed.xml": ROOT / "assets" / "mjcf" / "baseline" / "hand.xml",
}
VIEWABLE_EPS = 0.01  # the eps the committed models were generated at


def _surface_error(spheres, sample_fn, n=4000, seed=0) -> float:
    """Max |distance from a true-surface sample to the packed union| / characteristic radius."""
    rng = np.random.default_rng(seed)
    pts = sample_fn(rng, n)
    centres = np.array([s.pos for s in spheres])
    radii = np.array([s.radius for s in spheres])
    # signed distance to the union: negative inside
    d = np.linalg.norm(pts[:, None, :] - centres[None, :, :], axis=-1) - radii[None, :]
    return float(np.abs(d.min(axis=1)).max())


def test_capsule_chain_holds_the_requested_tolerance():
    r, eps = 0.010, 0.01
    ft = (0.0, 0.0, 0.0, 0.05, 0.0, 0.0)
    spheres = pack_capsule(ft, r, eps)

    def sample(rng, n):
        # points on the true capsule's cylindrical surface
        t = rng.uniform(0.0, 0.05, n)
        th = rng.uniform(0.0, 2 * math.pi, n)
        return np.stack([t, r * np.cos(th), r * np.sin(th)], axis=1)

    # the packed surface straddles the true one by at most eps*r/2 either way
    assert _surface_error(spheres, sample) <= eps * r / 2 + 1e-9


def test_chain_spacing_straddles_the_true_surface():
    r, eps = 0.0075, 0.02
    R, d = chain_spacing(r, eps)
    assert R == pytest.approx(r * (1 + eps / 2))
    # valley radius halfway between two centres
    valley = math.sqrt(R**2 - (d / 2) ** 2)
    assert valley == pytest.approx(r * (1 - eps / 2))


def test_cylinder_shell_outer_surface_lands_on_the_true_radius():
    r, hl, eps = 0.0125, 0.05, 0.02
    spheres = pack_cylinder(r, hl, eps)
    outer = max(math.hypot(s.pos[0], s.pos[1]) + s.radius for s in spheres)
    assert outer == pytest.approx(r, rel=1e-9)
    # and it is a shell, not a solid: no sphere sits on the axis at mid-length
    assert not any(
        abs(s.pos[2]) < 1e-9 and math.hypot(s.pos[0], s.pos[1]) < 1e-9 for s in spheres
    )


def test_finer_tolerance_never_reduces_sphere_count():
    prev = 0
    for eps in (0.05, 0.02, 0.01, 0.005):
        n = len(pack_capsule((0, 0, 0, 0.05, 0, 0), 0.010, eps))
        assert n >= prev
        prev = n


def test_box_pack_is_a_shell_and_stays_inside_the_extents():
    he = (0.048, 0.036, 0.001)
    spheres = pack_box(he, 0.02)
    assert spheres
    for s in spheres:
        for i, half in enumerate(he):
            assert abs(s.pos[i]) <= half + 1e-9


@pytest.mark.parametrize("packed_path", sorted(VIEWABLE))
def test_viewable_model_preserves_every_body_mass(packed_path: Path):
    """The whole point of the inertia bake. A drift here is the mass trap firing."""
    source = VIEWABLE[packed_path]
    m_src = mujoco.MjModel.from_xml_path(str(source))
    m_pk = mujoco.MjModel.from_xml_path(str(packed_path))

    for bid in range(m_src.nbody):
        name = mujoco.mj_id2name(m_src, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name:
            continue
        pid = mujoco.mj_name2id(m_pk, mujoco.mjtObj.mjOBJ_BODY, name)
        assert pid >= 0, f"{name} missing from {packed_path.name}"
        assert m_pk.body_mass[pid] == pytest.approx(m_src.body_mass[bid], rel=1e-6), name
        np.testing.assert_allclose(
            m_pk.body_inertia[pid], m_src.body_inertia[bid], rtol=1e-6, atol=1e-12
        )


@pytest.mark.parametrize("packed_path", sorted(VIEWABLE))
def test_viewable_model_matches_regeneration(packed_path: Path, tmp_path: Path):
    """Committed viewable models must not drift from the sources they mirror."""
    from scripts.generate_sphere_packed_scene import pack_scene

    regen = tmp_path / packed_path.name
    pack_scene(
        VIEWABLE[packed_path], regen,
        which="hand", eps=VIEWABLE_EPS, frac=0.5, include_palm=False,
    )
    assert regen.read_text() == packed_path.read_text(), (
        f"{packed_path.name} is stale -- regenerate with "
        f"scripts/generate_sphere_packed_scene.py --eps {VIEWABLE_EPS}"
    )


@pytest.mark.parametrize("packed_path", sorted(VIEWABLE))
def test_viewable_model_finger_geoms_are_all_spheres(packed_path: Path):
    m = mujoco.MjModel.from_xml_path(str(packed_path))
    n_sphere = 0
    for gid in range(m.ngeom):
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) or ""
        if body.endswith(("_mcp_frame", "_len_frame", "_pip_frame", "_tip")):
            assert m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE, body
            n_sphere += 1
    assert n_sphere > 100, "packing did not run on the finger links"
