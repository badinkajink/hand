"""The cad_v1 finger model (assets/mjcf/experimental/20261013-cad_v1/, scripts/build_cad_v1_finger.py).

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_cad_v1_finger.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
D = REPO / "assets/mjcf/experimental/20261013-cad_v1"
RV1 = REPO / "assets/mjcf/real_v1/real_hand.xml"
FINGERS = ("thumb", "index", "middle")


def load(name):
    m = mujoco.MjModel.from_xml_path(str(D / name))
    return m, mujoco.MjData(m)


@pytest.mark.parametrize("name", ["finger.xml", "hand.xml", "scene_screwdriver_medium.xml"])
def test_compiles_and_rests_without_finger_contacts(name):
    m, d = load(name)
    for k in range(m.nkey):
        mujoco.mj_resetDataKeyframe(m, d, k)
        mujoco.mj_forward(m, d)
        for i in range(d.ncon):
            b = {m.body(m.geom_bodyid[g]).name for g in (d.contact[i].geom1, d.contact[i].geom2)}
            assert b <= {"world", "screwdriver_medium"}, f"{name} key {k}: contact between {b}"


def test_joint_spacing_from_the_cad():
    m, d = load("hand.xml")
    mujoco.mj_forward(m, d)
    km = json.loads((D / "assembly.json").read_text())["report"]["kinematics_mm"]
    for f in FINGERS:
        yaw, mcp, pip, tip = (d.xpos[m.body(f"{f}_{b}").id] for b in ("yaw_frame", "mcp_frame", "pip_frame", "tip"))
        assert np.linalg.norm(mcp - yaw) * 1000 == pytest.approx(km["yaw_to_mcp"], abs=1e-3) == pytest.approx(20.75, abs=0.01)
        assert np.linalg.norm(pip - mcp) * 1000 == pytest.approx(km["mcp_to_pip"], abs=1e-3)
        assert np.linalg.norm(tip - pip) * 1000 == pytest.approx(26.61, abs=1e-3)  # real_v1's tip body
    assert km["pip_to_tip_end"] == pytest.approx(37.16, abs=0.01)


def test_names_axes_and_layout_match_real_v1():
    m, _ = load("hand.xml")
    r = mujoco.MjModel.from_xml_path(str(RV1))
    assert (m.nq, m.nv, m.nu) == (r.nq, r.nv, r.nu)
    for j in range(r.njnt):
        name = r.joint(j).name
        assert m.joint(name).qposadr[0] == r.joint(j).qposadr[0]
        np.testing.assert_allclose(m.jnt_axis[m.joint(name).id], r.jnt_axis[j], atol=1e-9)
        np.testing.assert_allclose(m.jnt_range[m.joint(name).id], r.jnt_range[j], atol=1e-6)
    for a in range(r.nu):
        assert m.actuator(a).name == r.actuator(a).name
    for f in FINGERS:
        assert m.body(f"{f}_tip").id > 0


def test_pads_and_link_spheres():
    m, _ = load("hand.xml")
    pads = json.loads((D / "assembly.json").read_text())["pads"]["n_pads"]
    spheres = json.loads((D / "spheres.json").read_text())
    for f in FINGERS:
        n = sum(1 for g in range(m.ngeom) if m.geom(g).name.startswith(f"{f}_pad"))
        assert n == pads[f] > 900
        n_link = sum(1 for g in range(m.ngeom) if m.geom_group[g] == 3 and m.geom(g).name.startswith(f"{f}_"))
        assert n_link == sum(len(v) for v in spheres.values())


def test_pads_face_the_opposing_finger():
    m, d = load("hand.xml")
    mujoco.mj_forward(m, d)
    for f, sign in (("thumb", 1.0), ("index", -1.0), ("middle", -1.0)):
        tip = d.xpos[m.body(f"{f}_tip").id]
        xs = [d.geom_xpos[g][0] - tip[0] for g in range(m.ngeom) if m.geom(g).name.startswith(f"{f}_pad")]
        assert sign * max(xs, key=lambda x: sign * x) > 0.0098  # palmar pads 9.8 mm out from the tip axis


def test_dip_spheres_stay_behind_the_pads():
    """The DIP link's spheres reach no farther than the pads on the tip's palmar, end and side faces (clip_to_pads)."""
    m, d = load("finger.xml")
    mujoco.mj_forward(m, d)
    pads = [g for g in range(m.ngeom) if m.geom(g).name.startswith("index_pad")]
    P, rp = np.array([d.geom_xpos[g] for g in pads]), m.geom_size[pads[0]][0]
    lo, hi = (P - rp).min(0), (P + rp).max(0)
    for g in range(m.ngeom):
        if not m.geom(g).name.startswith("index_pip_frame_s"):
            continue
        c, r = d.geom_xpos[g], m.geom_size[g][0]
        assert c[0] - r >= lo[0] - 1e-9 and c[2] - r >= lo[2] - 1e-9, m.geom(g).name  # palmar (-x) and end (-z) faces
        if c[2] < hi[2]:
            assert lo[1] - 1e-9 <= c[1] - r and c[1] + r <= hi[1] + 1e-9, m.geom(g).name  # side faces
