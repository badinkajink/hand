# pyright: reportMissingImports=false
"""A scene mutation must change exactly the property it names, and nothing else.

These studies work by handing the SAME trained policies a scene with one physical property
moved, so a mutator with a side effect does not fail loudly — it reports a confident number
about the wrong experiment. The two side effects that actually bite here are mass (finger links
and the tool derive mass from geom volume, so reshaping a tip silently reweighs the finger) and
reach (a shape that protrudes further meets the object sooner, and then "wins" on grip for a
reason that has nothing to do with its cross-section). Both are pinned below.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from morphohand.studies.scene_mutate import (
    SHIPPED_REACH,
    SHIPPED_TIP,
    TIP_BODIES,
    TIP_SHAPES,
    Scene,
    mass_check,
    tip_geoms,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results/phase1/landscape/m05_ik_cem/frozen_scene.xml"

pytestmark = pytest.mark.skipif(not FROZEN.exists(), reason="m05 frozen scene not present")


def _write(tmp_path, mutate, name="s.xml") -> Path:
    sc = Scene(FROZEN)
    mutate(sc)
    return sc.write(tmp_path / name)


@pytest.mark.parametrize("shape", TIP_SHAPES)
def test_tip_shape_preserves_every_body_mass(tmp_path, shape):
    """Reshaping a pad must not reweigh the finger it sits on."""
    out = _write(tmp_path, lambda s: s.set_tip_shape(shape), f"{shape}.xml")
    assert mass_check(FROZEN, out) == {}


@pytest.mark.parametrize("shape", TIP_SHAPES)
def test_tip_shapes_are_reach_normalised(shape):
    """Every shape presents its surface at the same distance from the tip origin.

    Without this the family is a reach sweep wearing a shape sweep's name — and worse, shapes
    that extend only sideways sit INSIDE the distal capsule and never touch the object at all,
    which is what the first render of the family showed.
    """
    forward = []
    for spec in tip_geoms(shape):
        r = float(spec["size"].split()[0])
        if "fromto" in spec:
            v = [float(t) for t in spec["fromto"].split()]
            forward.append(max(v[0], v[3]) + r)
        else:                                   # pos/size primitive: half-extent along local x
            x = float((spec.get("pos") or "0 0 0").split()[0])
            sizes = [float(t) for t in spec["size"].split()]
            forward.append(x + sizes[0])
    assert max(forward) == pytest.approx(SHIPPED_REACH, abs=1e-9)


def test_shipped_shape_reproduces_the_shipped_tip(tmp_path):
    """`cap_cross` at the default size IS what the m05 scene ships, so it is the baseline row."""
    out = _write(tmp_path, lambda s: s.set_tip_shape(SHIPPED_TIP))
    before = mujoco.MjModel.from_xml_path(str(FROZEN))
    after = mujoco.MjModel.from_xml_path(str(out))
    for body in TIP_BODIES:
        b0 = mujoco.mj_name2id(before, mujoco.mjtObj.mjOBJ_BODY, body)
        b1 = mujoco.mj_name2id(after, mujoco.mjtObj.mjOBJ_BODY, body)
        g0 = [g for g in range(before.ngeom) if before.geom_bodyid[g] == b0]
        g1 = [g for g in range(after.ngeom) if after.geom_bodyid[g] == b1]
        assert len(g0) == len(g1) == 1
        np.testing.assert_allclose(before.geom_size[g0[0]], after.geom_size[g1[0]], atol=1e-9)
        np.testing.assert_allclose(before.geom_pos[g0[0]], after.geom_pos[g1[0]], atol=1e-9)


def test_shape_change_carries_material_properties(tmp_path):
    """A shape comparison must not become a friction comparison."""
    out = _write(tmp_path, lambda s: s.set_tip_shape("pad_flat"))
    before = mujoco.MjModel.from_xml_path(str(FROZEN))
    after = mujoco.MjModel.from_xml_path(str(out))
    b0 = mujoco.mj_name2id(before, mujoco.mjtObj.mjOBJ_BODY, "index_tip")
    b1 = mujoco.mj_name2id(after, mujoco.mjtObj.mjOBJ_BODY, "index_tip")
    g0 = next(g for g in range(before.ngeom) if before.geom_bodyid[g] == b0)
    g1 = next(g for g in range(after.ngeom) if after.geom_bodyid[g] == b1)
    np.testing.assert_allclose(before.geom_friction[g0], after.geom_friction[g1], atol=1e-9)
    np.testing.assert_allclose(before.geom_solimp[g0], after.geom_solimp[g1], atol=1e-9)


def test_radius_change_holds_object_mass(tmp_path):
    """A fatter shaft is a grip-geometry change; making it a load change too would confound it."""
    out = _write(tmp_path, lambda s: s.scale_object_radius(1.2))
    assert mass_check(FROZEN, out) == {}


def test_density_change_is_the_one_that_moves_mass(tmp_path):
    """The deliberate mass knob must actually move mass, and only the object's."""
    out = _write(tmp_path, lambda s: s.scale_object_density(2.0))
    assert mass_check(FROZEN, out) == {"screwdriver_medium": pytest.approx(1.0, abs=1e-3)}


def test_friction_scale_touches_only_the_sliding_term(tmp_path):
    out = _write(tmp_path, lambda s: s.scale_friction(0.5))
    before = mujoco.MjModel.from_xml_path(str(FROZEN))
    after = mujoco.MjModel.from_xml_path(str(out))
    b = mujoco.mj_name2id(before, mujoco.mjtObj.mjOBJ_BODY, "screwdriver_medium")
    g0 = next(g for g in range(before.ngeom) if before.geom_bodyid[g] == b)
    g1 = next(g for g in range(after.ngeom) if after.geom_bodyid[g] == b)
    assert after.geom_friction[g1][0] == pytest.approx(before.geom_friction[g0][0] * 0.5)
    np.testing.assert_allclose(before.geom_friction[g0][1:], after.geom_friction[g1][1:], atol=1e-9)


def test_comments_survive_a_mutation(tmp_path):
    """Scenes carry their design rationale in comments; a rewrite that drops them is unreviewable."""
    src = tmp_path / "commented.xml"
    src.write_text(FROZEN.read_text().replace(
        "<worldbody>", "<worldbody>\n    <!-- load-bearing rationale -->", 1))
    out = Scene(src).set_tip_shape("sphere").write(tmp_path / "out.xml")
    assert "load-bearing rationale" in out.read_text()


def test_proximal_length_moves_the_joint_not_just_the_capsule(tmp_path):
    """`set_proximal_length` changes the ROBOT, so the middle phalanx has to move with the draw.

    `shorten_proximal` deliberately leaves kinematics alone (it only fixes the drawing); if the
    two ever collapsed into each other, a "short proximal" hand would still have the long hand's
    reach and every reach-shell verdict taken on it would be wrong.
    """
    out = _write(tmp_path, lambda s: s.set_proximal_length(0.025))
    after = mujoco.MjModel.from_xml_path(str(out))
    for finger in ("thumb", "index", "middle"):
        assert after.body(f"{finger}_len_frame").pos[0] == pytest.approx(0.025)


def test_proximal_length_shortens_the_reach_by_what_it_removed(tmp_path):
    """The point of the mutation is a shorter finger; measure it at the tip, not in the XML."""
    before = mujoco.MjModel.from_xml_path(str(FROZEN))
    removed = float(before.body("index_len_frame").pos[0]) - 0.025
    out = _write(tmp_path, lambda s: s.set_proximal_length(0.025))
    after = mujoco.MjModel.from_xml_path(str(out))
    bd, ad = mujoco.MjData(before), mujoco.MjData(after)
    mujoco.mj_forward(before, bd)
    mujoco.mj_forward(after, ad)
    drop = np.linalg.norm(bd.body("index_tip").xpos - bd.body("index_mount").xpos) - \
        np.linalg.norm(ad.body("index_tip").xpos - ad.body("index_mount").xpos)
    assert drop == pytest.approx(removed, abs=1e-6)


def test_proximal_length_holds_link_mass(tmp_path):
    """THE MASS TRAP: a shorter link is also a lighter one, and that is a separate claim."""
    out = _write(tmp_path, lambda s: s.set_proximal_length(0.025))
    assert mass_check(FROZEN, out) == {}
