# pyright: reportMissingImports=false
"""Contact laws of the production fingertip pads (reorient_backends.replace_tips, the default 1 mm TPU-tip pads), on the
rig of scripts/pad_contact_rig.py: the tracked real_v1 screwdriver scene with the production pads, the hand rigid, gravity
off, only the thumb's pads colliding with the tool. Known forces and torques on the tool are compared with closed-form
values: each pad's spring E a_s / h, the pressure law integrated over the block's surface, the soft-friction creep law,
the Coulomb and torsional slip limits, and the stiffness's independence of the tool's mass when the pads are rebuilt.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_pad_contact_laws.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fingertip_geometry as G  # noqa: E402
import reorient_backends as RB  # noqa: E402
from pad_contact_rig import FILLET, MU, Rig, continuum, continuum_approach, realised_stiffness  # noqa: E402

ROUND = 5e-5                                     # the builder writes d0 with 6 significant digits


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    return Rig(tmp_path_factory.mktemp("pads"))


@pytest.mark.parametrize("F", [0.5, 1.0, 3.0])
def test_each_pad_is_the_median_pad_spring_and_the_forces_balance(rig, F):
    _, _, a, K = G.pad_spheres(FILLET, RB.PAD_S, RB.PAD_RS, 1.0, RB.E_TPU)
    assert rig.meta["pad_K"] == pytest.approx(float(np.median(K)), rel=1e-12)
    assert realised_stiffness(rig.jointed, "thumb") == pytest.approx(rig.meta["pad_K"], rel=ROUND)
    rig.press(F)
    c = rig.contacts()
    assert len(c) >= 20
    for f, pen, _, _ in c:
        assert f / pen == pytest.approx(realised_stiffness(rig.m, "thumb"), rel=1e-6)
    push = sum(f * abs(float(nrm @ rig.n)) for f, _, _, nrm in c)   # normals tilt with the tool's curvature
    assert push == pytest.approx(F, rel=1e-3)


@pytest.mark.parametrize("F, approach_tol, twist_tol", [(0.5, 0.15, 0.07), (1.0, 0.05, 0.05), (3.0, 0.06, 0.05)])
def test_approach_and_twist_limit_follow_the_integrated_pressure_law(rig, F, approach_tol, twist_tol):
    rig.press(F)
    a_ref = continuum_approach(F)
    assert rig.approach() == pytest.approx(a_ref, rel=approach_tol)
    assert rig.twist_limit() == pytest.approx(MU * continuum(a_ref)[1], rel=twist_tol)


@pytest.mark.parametrize("impratio", [10.0, 100.0])
def test_tangential_load_creeps_at_the_soft_friction_law_and_slides_above_mu_n(tmp_path, impratio):
    """Below the cone every pad row obeys R_t f_t = b v, so the tool creeps at v = F_t / (b sum 1/R_t) with
    b = 2 / (d tc) and R_t = (1 - d) / d A / impratio; above mu N it slides."""
    r = Rig(tmp_path, impratio=impratio)
    F = 1.0
    r.press(F)
    v_law = r.creep_law(0.5 * MU * F, len(r.contacts()))
    r.load(F, Ft=0.5 * MU * F)
    r.run(200)                                                  # 0.4 s; a 98 g tool settles into creep by 0.3 s
    assert r.tool_speed() == pytest.approx(v_law, rel=0.1)
    r.press(F)
    y0 = float(r.d.xpos[r.tool] @ r.t)
    r.load(F, Ft=1.2 * MU * F)
    r.run(50)
    assert float(r.d.xpos[r.tool] @ r.t) - y0 > 0.005          # > 5 mm in 0.1 s


def test_twist_turns_the_tool_only_above_the_torsional_slip_limit(rig):
    turned = {}
    for frac in (0.5, 1.2):
        rig.press(1.0)
        tau = frac * rig.twist_limit()
        q0 = rig.d.xquat[rig.tool].copy()
        rig.load(1.0, tau=tau)
        rig.run(100)                                             # 0.2 s
        turned[frac] = rig.turned_deg(q0)
    assert turned[0.5] < 0.5
    assert turned[1.2] > 3.0


@pytest.mark.parametrize("scale", [0.25, 4.0])
def test_rebuilt_pads_keep_their_stiffness_and_creep_for_any_tool_mass(rig, tmp_path, scale):
    r = Rig(tmp_path, mass_scale=scale)
    assert r.m.body_mass[r.tool] == pytest.approx(scale * rig.m.body_mass[rig.tool], rel=1e-9)
    assert realised_stiffness(r.jointed, "thumb") == pytest.approx(rig.meta["pad_K"], rel=ROUND)
    speeds = []
    for x in (rig, r):
        x.press(1.0)
        for f, pen, _, _ in x.contacts():
            assert f / pen == pytest.approx(rig.meta["pad_K"], rel=ROUND)
        x.load(1.0, Ft=0.5 * MU)
        x.run(200)
        speeds.append(x.tool_speed())
    assert speeds[1] == pytest.approx(speeds[0], rel=0.005)


@pytest.mark.parametrize("scale", [0.25, 4.0])
def test_pads_built_for_another_tool_mass_scale_stiffness_by_the_inverse_weight_ratio(rig, tmp_path, scale):
    r = Rig(tmp_path, stale_scale=scale)
    ratio = rig.inverse_weight() / r.inverse_weight()
    assert abs(ratio - 1) > 0.5                                 # x0.25: -61 %, x4: +65 % for this tip and tool
    assert realised_stiffness(r.jointed, "thumb") == pytest.approx(ratio * rig.meta["pad_K"], rel=ROUND)
    r.press(1.0)
    for f, pen, _, _ in r.contacts():
        assert f / pen == pytest.approx(ratio * rig.meta["pad_K"], rel=ROUND)


@pytest.mark.parametrize("E", [5e6, 2e7])
def test_pad_stiffness_scales_with_the_modulus(rig, tmp_path, E):
    r = Rig(tmp_path, E=E)
    assert r.meta["pad_K"] == pytest.approx(rig.meta["pad_K"] * E / RB.E_TPU, rel=1e-12)
    r.press(1.0)
    for f, pen, _, _ in r.contacts():
        assert f / pen == pytest.approx(rig.meta["pad_K"] * E / RB.E_TPU, rel=ROUND)
