# pyright: reportMissingImports=false
"""The perp hand files must stay interchangeable with the perp scene.

A hand XML that has drifted from its scene bakes one geometry and evaluates another, and
nothing at runtime reports the mismatch — the generated model just quietly describes a
different robot. These tests pin the parity and the generation contract.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from morphohand.sampling.morphology import (
    PERP_T_WORKSPACE,
    perp_compact_design,
    perp_mount_positions,
)
from morphohand.tools.morphology_xml import (
    MorphologyValues,
    create_rigid_hand_and_scene_xmls,
    create_rigid_morphology_xml,
)

PERP_DIR = Path(__file__).resolve().parents[1] / "assets" / "mjcf" / "perp"
HAND = PERP_DIR / "perp_hand.xml"
HAND_ACTUATED = PERP_DIR / "perp_hand_morphology_actuated.xml"
SCENE = PERP_DIR / "scenes" / "scene_screwdriver_medium_perp.xml"

KEYFRAMES = ("open", "closed", "press", "open_ik")
MORPH_JOINTS = (
    "thumb_x", "thumb_y", "thumb_len",
    "index_x", "index_y", "index_len",
    "middle_x", "middle_y", "middle_len",
)


def _model(path: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(path))


@pytest.mark.parametrize("path", [HAND, HAND_ACTUATED, SCENE])
def test_files_parse_with_elementtree_not_just_mujoco(path: Path):
    """MuJoCo's parser tolerates `--` inside an XML comment; ElementTree does not.

    The generator uses ElementTree, so a file that only MuJoCo accepts loads fine everywhere
    except the one place that matters.
    """
    ET.parse(path)
    _model(path)


@pytest.mark.parametrize("path", [HAND, HAND_ACTUATED])
def test_hand_has_no_palm_pose_joints(path: Path):
    """`_is_scene_model` keys off `palm_px`; a hand carrying it is treated as a scene and the
    keyframe qpos gets sliced at the wrong offset."""
    model = _model(path)
    names = {model.joint(i).name for i in range(model.njnt)}
    assert "palm_px" not in names
    assert model.nq == 18


def test_hand_and_scene_agree_on_every_keyframe():
    """Same body poses for every shared body, at every keyframe."""
    mh, ms = _model(HAND), _model(SCENE)
    dh, ds = mujoco.MjData(mh), mujoco.MjData(ms)

    shared = [
        ms.body(i).name
        for i in range(ms.nbody)
        if ms.body(i).name not in ("world", "screwdriver_medium")
    ]
    assert len(shared) > 10

    for key in KEYFRAMES:
        mujoco.mj_resetDataKeyframe(mh, dh, mh.key(key).id)
        mujoco.mj_forward(mh, dh)
        mujoco.mj_resetDataKeyframe(ms, ds, ms.key(key).id)
        mujoco.mj_forward(ms, ds)
        for name in shared:
            np.testing.assert_allclose(
                dh.body(name).xpos, ds.body(name).xpos, atol=1e-12,
                err_msg=f"body {name} differs at keyframe {key}",
            )


def test_morph_joint_ranges_match_across_all_three_files():
    models = {"hand": _model(HAND), "actuated": _model(HAND_ACTUATED), "scene": _model(SCENE)}
    for joint in MORPH_JOINTS:
        ranges = {k: tuple(m.jnt_range[m.joint(joint).id]) for k, m in models.items()}
        assert len(set(ranges.values())) == 1, f"{joint} ranges diverge: {ranges}"


def test_morph_ranges_match_the_sampled_workspace():
    """The MJCF rails and `PERP_T_WORKSPACE` are two copies of one fact; drift silently makes
    the sampler propose designs the model clamps, or refuse ones it allows."""
    model = _model(HAND)
    for finger, box in (("thumb", PERP_T_WORKSPACE.thumb),
                        ("index", PERP_T_WORKSPACE.index),
                        ("middle", PERP_T_WORKSPACE.middle)):
        lo_x, hi_x = model.jnt_range[model.joint(f"{finger}_x").id]
        lo_y, hi_y = model.jnt_range[model.joint(f"{finger}_y").id]
        assert lo_x == pytest.approx(box.x_min, abs=1e-9)
        assert hi_x == pytest.approx(box.x_max, abs=1e-9)
        assert lo_y == pytest.approx(box.y_min, abs=1e-9)
        assert hi_y == pytest.approx(box.y_max, abs=1e-9)


def test_actuated_file_drives_all_eighteen_dofs():
    model = _model(HAND_ACTUATED)
    assert model.nu == 18
    driven = {model.joint(model.actuator_trnid[a, 0]).name for a in range(model.nu)}
    assert set(MORPH_JOINTS) <= driven


def test_generation_from_hand_yields_a_loadable_rigid_pair(tmp_path: Path):
    """The whole reason the hand file exists: a real hand/scene PAIR, not two copies of a scene."""
    morph = perp_compact_design(pair_x_t=1.0)
    hand_out, scene_out = create_rigid_hand_and_scene_xmls(
        base_hand_xml_path=HAND,
        base_scene_xml_path=SCENE,
        morphology=morph,
        output_dir=tmp_path,
    )

    mh, msc = _model(hand_out), _model(scene_out)
    for model in (mh, msc):
        names = {model.joint(i).name for i in range(model.njnt)}
        assert not (names & set(MORPH_JOINTS)), "rigid model still has morph joints"
        assert model.nkey == len(KEYFRAMES)

    hand_bodies = {mh.body(i).name for i in range(mh.nbody)}
    scene_bodies = {msc.body(i).name for i in range(msc.nbody)}
    assert "screwdriver_medium" not in hand_bodies
    assert "screwdriver_medium" in scene_bodies

    # The design is actually baked in, not just stripped out.
    d = mujoco.MjData(msc)
    mujoco.mj_resetDataKeyframe(msc, d, msc.key("open").id)
    mujoco.mj_forward(msc, d)
    expected_x = perp_mount_positions(morph)["index"][0]
    assert d.body("index_mount").xpos[0] == pytest.approx(expected_x, abs=1e-9)


def test_every_keyframe_is_stripped_not_just_open(tmp_path: Path):
    """Regression: only `open` used to be rewritten, so any source with a second keyframe
    produced a model MuJoCo refuses to load ('invalid qpos size, expected 9, got 18')."""
    out = create_rigid_morphology_xml(
        base_xml_path=HAND,
        morphology=MorphologyValues(0, 0, 0, 0, 0, 0, 0, 0, 0),
        output_xml_path=tmp_path / "rigid_hand.xml",
    )
    model = _model(out)  # would raise before the fix
    assert model.nq == 9
    for key in KEYFRAMES:
        assert model.key(key).qpos.shape == (9,)


def test_generating_from_the_actuated_file_is_refused(tmp_path: Path):
    """It used to silently substitute the BASELINE hand's open angles, discarding the perp pose."""
    with pytest.raises(ValueError, match="morphology_actuated"):
        create_rigid_morphology_xml(
            base_xml_path=HAND_ACTUATED,
            morphology=MorphologyValues(0, 0, 0, 0, 0, 0, 0, 0, 0),
            output_xml_path=tmp_path / "bad.xml",
        )


def test_compact_design_endpoints_and_clipping():
    assert perp_compact_design() == MorphologyValues(0, 0, 0, 0, 0, 0, 0, 0, 0)

    full = perp_compact_design(thumb_t=1.0, pair_x_t=1.0, pair_y_t=1.0)
    mounts = perp_mount_positions(full)
    assert mounts["thumb"][0] == pytest.approx(-0.0125, abs=1e-9)
    assert mounts["index"] == pytest.approx((0.0125, 0.0125), abs=1e-9)
    assert mounts["middle"] == pytest.approx((0.0125, -0.0125), abs=1e-9)

    # The pair stays symmetric — an asymmetric pair tilts the pinch axis off Y.
    half = perp_compact_design(pair_y_t=0.5)
    assert half.index_y == pytest.approx(-half.middle_y, abs=1e-12)

    for bad in ((-0.1, 0, 0), (0, 1.5, 0), (0, 0, 2.0)):
        with pytest.raises(ValueError):
            perp_compact_design(*bad)
