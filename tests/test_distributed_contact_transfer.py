"""The articulated port retains physical coefficients on native hand constraints."""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from distributed_contact_transfer import build, command
from contact_surface.runtime import step1, mass_matrix
from contact_surface.scaling import physical_coefficients


def test_articulated_runtime_contact_coefficients_and_diagonal():
    if not hasattr(mujoco.mjtEnableBit, "mjENBL_DIAGEXACT"):
        pytest.skip("isolated current MuJoCo required")
    m, d, p, meta = build(dict(policy="runtime", spacing_mm=1.0, timestep=0.00005))
    assert m.nv == 15 and m.nu == 9
    assert m.npair == 410
    assert meta["ik_error_m"] < 1e-7
    for _ in range(1500):
        step1(m, d, p)
        mujoco.mj_step2(m, d)
    assert d.ncon > 10
    # Audit a fresh consistent pre-integration state, including moving fingers.
    step1(m, d, p)
    Minv = np.linalg.inv(mass_matrix(m, d))
    J = d.efc_J.reshape(d.nefc, m.nv)
    for c in d.contact[: d.ncon]:
        adr = c.efc_address
        if adr < 0:
            continue
        assert d.efc_diagA[adr] == pytest.approx(float(J[adr] @ Minv @ J[adr]), rel=1e-11)
        k, b = physical_coefficients(c.solref, c.solimp, d.efc_diagA[adr], m.opt.timestep)
        assert k == pytest.approx(p["stiffness"], rel=1e-12)
        assert b == pytest.approx(k * 0.03, rel=1e-12)
        assert c.solreffriction[1] == pytest.approx(c.solref[1] * 10.0 / m.opt.impratio)


def test_torque_input_has_no_solver_or_contact_feedback():
    cfg = {"torque_peak": 0.012}
    assert command(0.2, cfg) == 0.0
    assert command(0.375, cfg) == pytest.approx(0.012)
    assert command(0.525, cfg) == pytest.approx(-0.012)
    ts = np.linspace(0.3, 0.6, 1001)
    assert np.trapezoid([command(float(t), cfg) for t in ts], ts) == pytest.approx(0.0, abs=1e-12)
