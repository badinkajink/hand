"""Check the mechanisms under test, including a deliberately failing material law."""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from contact_surface.metrics import contacts
from contact_surface.mujoco_model import build, static_fixture
from contact_surface.runtime import step1
from distributed_contact import DEFAULTS

HAS_EXACT = hasattr(mujoco.mjtEnableBit, "mjENBL_DIAGEXACT")


@pytest.mark.skipif(not HAS_EXACT, reason="diagexact requires MuJoCo >=3.9")
def test_exact_diagonal_does_not_remove_material_mass_scaling():
    force = []
    for scale in (0.25, 1, 4):
        m, d, p, _ = build(dict(DEFAULTS, mass_scale=scale, diagexact=True))
        static_fixture(m, d, p, 0.0002)
        rows = contacts(m, d, p)
        assert all(r["Lambda_solver"] == pytest.approx(r["Lambda_exact"]) for r in rows)
        force.append(sum(r["f_n"] for r in rows))
    assert force[1] / force[0] == pytest.approx(4)
    assert force[2] / force[1] == pytest.approx(4)


@pytest.mark.skipif(not HAS_EXACT, reason="diagexact requires MuJoCo >=3.9")
def test_runtime_translation_tracks_configuration_without_a_step_of_lag():
    m, d, p, _ = build(dict(DEFAULTS, diagexact=True, mapping_policy="runtime_selected_direct"))
    lambdas = []
    for x in (0.0, 0.006):
        d.qpos[:] = m.qpos0
        d.qpos[0] = x
        d.qpos[2] -= 0.0002
        d.qvel[:] = 0
        step1(m, d, p)
        mujoco.mj_step2(m, d)
        rows = contacts(m, d, p)
        lambdas.append(rows[0]["Lambda_exact"])
        for r in rows:
            assert r["Lambda_solver"] == pytest.approx(r["Lambda_exact"])
            assert r["stiffness_predicted"] == pytest.approx(r["k_i"], rel=1e-12)
            assert r["damping_predicted"] == pytest.approx(r["k_i"] * p["relaxation"], rel=1e-12)
    assert not np.isclose(*lambdas, rtol=0.01)


def test_compensation_at_rest_does_not_eliminate_acceleration_term():
    errors = []
    for impedance in (0.8, 0.0001):
        m, d, p, _ = build(
            dict(
                DEFAULTS,
                mapping_policy="runtime_selected_direct",
                impedance_override=impedance,
                redundant_count=1,
            )
        )
        d.qpos[2] -= 0.0002
        step1(m, d, p)
        mujoco.mj_step2(m, d)
        r = contacts(m, d, p)[0]
        target = r["k_i"] * r["penetration"]
        assert r["f_n"] == pytest.approx(target - r["acceleration"] / r["efc_R"], rel=1e-10)
        errors.append(abs(r["f_n"] / target - 1))
    assert errors[0] > 0.5
    assert errors[1] < 0.002


def test_friction_regularization_can_change_without_changing_physical_damping():
    for ratio in (10, 10000):
        m, d, p, _ = build(
            dict(
                DEFAULTS,
                mapping_policy="runtime_selected_direct",
                impedance_override=0.0001,
                impratio=ratio,
                tangent_damping_ratio=10,
            )
        )
        static_fixture(m, d, p, 0.0002)
        step1(m, d, p)
        for c in d.contact[: d.ncon]:
            i = c.efc_address
            if i < 0:
                continue
            B = d.efc_KBIP.reshape(d.nefc, 4)[:, 1]
            assert B[i] / d.efc_R[i] == pytest.approx(p["physical_damping_target"])
            assert B[i + 1] / d.efc_R[i + 1] == pytest.approx(10 * p["physical_damping_target"])
