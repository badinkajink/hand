"""Mechanics checks: surface conservation, solver mixing, equilibrium and redundancy."""

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from contact_surface.geometry import Surface
from contact_surface.metrics import aggregate, contacts
from contact_surface.mujoco_model import build, continuum, load_step, static_fixture
from distributed_contact import DEFAULTS


def fixture(**kw):
    c = dict(DEFAULTS, **kw)
    m, d, p, _ = build(c)
    return c, m, d, p


def test_cap_conservation_and_frozen_nominal():
    reference = Surface().parameters()
    assert reference["count"] == 205
    assert reference["stiffness"] == pytest.approx(817.2125281276)
    assert reference["solimp"][0] == pytest.approx(0.740828961644)
    for spacing, count, relaxation in [(2, 51, 0.02), (1, 205, 0.02), (0.5, 819, 0.03)]:
        p = Surface(spacing_mm=spacing, relaxation=relaxation).parameters()
        assert p["count"] == count
        assert p["count"] * p["nominal_area"] == pytest.approx(p["cap_area"])
        assert p["count"] * p["stiffness"] == pytest.approx(
            reference["count"] * reference["stiffness"]
        )
    with pytest.raises(ValueError, match="Unrepresentable"):
        Surface(spacing_mm=0.5, relaxation=0.02).parameters()


def test_geometry_matches_existing_chain_formula():
    import hom_contact_rig as original

    p = Surface().parameters()
    expected = np.round(original.fib_cap(205, math.pi / 4) * (0.01055 - 0.00075), 7)
    np.testing.assert_array_equal(Surface().centers(), expected)
    assert np.max(np.linalg.norm(expected, axis=1)) + 0.00075 == pytest.approx(
        p["radius"], abs=1e-7
    )


def test_actual_mixing_and_forward_static_balance():
    _, m, d, p = fixture()
    _, residual = static_fixture(m, d, p, 0.0002)
    rows = contacts(m, d, p)
    assert residual < 1e-7
    assert len(rows) > 5
    for row in rows:
        assert row["condim"] == 3
        assert row["solref_0"] == pytest.approx(p["solref"][0])
        assert row["solimp_0"] == pytest.approx(p["solimp"][0])
        assert row["friction_0"] == 1  # pad priority overrides the object's 0.3
        assert row["f_n"] == pytest.approx(row["predicted_static_force"], rel=1e-7)
        assert row["force_2"] > 0


def test_fixed_parameter_mass_scaling_and_forward_load():
    forces = []
    for scale in [0.5, 1.0, 2.0]:
        c, m, d, p = fixture(mass_scale=scale)
        static_fixture(m, d, p, 0.0002)
        forces.append(aggregate(contacts(m, d, p), p)["Fn_total"])
        mujoco.mj_resetData(m, d)
        d.qpos[2] -= 0.00001
        for _ in range(600):
            load_step(m, d, p, 1.0)
        total = aggregate(contacts(m, d, p), p)
        assert total["wrench_2"] == pytest.approx(1.0, abs=1e-5)
        assert total["aggregate_stiffness_error"] + 1 == pytest.approx(
            p["nominal_inverse_inertia"] / p["actual_inverse_inertia"], rel=1e-4
        )
    assert forces[2] / forces[0] == pytest.approx(4.0, rel=1e-8)


def test_redundant_contacts_preserved_but_free_acceleration_saturates():
    static, free = [], []
    for n in [1, 32]:
        _, m, d, p = fixture(redundant_count=n)
        static_fixture(m, d, p, 0.0002)
        rows = contacts(m, d, p)
        assert len(rows) == n
        assert len({r["constraint_address"] for r in rows}) == n
        static.append(sum(r["f_n"] for r in rows))
        d.qfrc_applied[:] = 0
        d.qacc_warmstart[:] = 0
        mujoco.mj_forward(m, d)
        free.append(sum(r["f_n"] for r in contacts(m, d, p)))
    assert static[1] / static[0] == pytest.approx(32.0, rel=1e-8)
    assert 1 < free[1] / free[0] < 1.4


def test_continuum_plane_quadrature_has_independent_closed_form():
    p = Surface().parameters()
    for delta in [0.00005, 0.0002, 0.0004]:
        exact = math.pi * p["modulus"] * delta**2 * (1 - 2 * delta / (3 * p["radius"]))
        assert continuum(delta, "plane", 0.0125, p)["force"] == pytest.approx(exact, rel=1e-4)


def test_fillet_areas_support_and_native_force_balance():
    from contact_surface.fillet import filleted_surface, support_point

    for spacing, relaxation in [(2.0, 0.02), (1.0, 0.02), (0.5, 0.03)]:
        centers, p = filleted_surface(spacing, relaxation)
        a, b, r = p["width"] / 2 - 0.006, p["length"] / 2 - 0.006, 0.006
        exact_area = 4 * a * b + 2 * math.pi * r * (a + b) + 2 * math.pi * r * r
        assert sum(s["nominal_area"] for s in p["sample_parameters"]) == pytest.approx(exact_area)
        for angle in [0, 30, 60, 85]:
            normal, point, _ = support_point(p, angle, 45)
            assert np.max(centers @ normal) + p["sphere_radius"] <= point @ normal + 1e-7
    _, m, d, p = fixture(
        tip_shape="filleted_prism", tilt_degrees=45.0, azimuth_degrees=0.0, mode="displacement"
    )
    _, residual = static_fixture(m, d, p, 0.0002)
    assert residual < 1e-7
    rows = contacts(m, d, p)
    assert any(row["region"] == "edge" for row in rows)
    assert sum(row["f_n"] for row in rows) > 0


@pytest.mark.parametrize("geometry", ["plane", "cylinder"])
def test_lambda_compensation_restores_native_normal_stiffness(geometry):
    from contact_surface.scaling import physical_coefficients

    total_forces = []
    for mass_scale in [0.25, 1.0, 4.0]:
        _, model, data, parameters = fixture(
            geometry=geometry,
            mass_scale=mass_scale,
            mapping_policy="lambda_compensated_direct",
        )
        _, acceleration = static_fixture(model, data, parameters, 0.0002)
        assert acceleration < 1e-7
        rows = contacts(model, data, parameters)
        for row in rows:
            assert row["stiffness_measured"] == pytest.approx(parameters["stiffness"], rel=1e-9)
            assert row["stiffness_predicted"] == pytest.approx(parameters["stiffness"], rel=1e-9)
            assert row["damping_predicted"] == pytest.approx(
                parameters["physical_damping_target"], rel=1e-9
            )
            assert row["solref_0"] < 0 and row["solref_1"] < 0
        total_forces.append(sum(row["f_n"] for row in rows))
        # Native independent load integration, starting unloaded.
        mujoco.mj_resetData(model, data)
        data.qpos[2] -= 0.00001
        for _ in range(1000):
            load_step(model, data, parameters, 1.0)
        result = aggregate(contacts(model, data, parameters), parameters)
        assert result["wrench_2"] == pytest.approx(1.0, abs=1e-5)
        assert abs(result["aggregate_stiffness_error"]) < 1e-4
        # Coefficients also recover the same material law without using measured forces.
        k, damping = physical_coefficients(
            parameters["solref"],
            parameters["solimp"],
            parameters["actual_inverse_inertia"],
            model.opt.timestep,
        )
        assert k == pytest.approx(parameters["stiffness"])
        assert damping / k == pytest.approx(parameters["relaxation"])
    np.testing.assert_allclose(total_forces, total_forces[0], rtol=1e-9)
