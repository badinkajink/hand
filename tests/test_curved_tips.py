"""Curved fingertips of the 2026-10-15 study (tip_codesign_geometry.CurvedTip, scripts/curved_tips_*.py).

The whole 10.55 mm sphere (tcd_sph10.55, the real_v1 sphere tip) is the limit at which the curved-tip machinery must
reproduce the sphere tip: its Drake-field law against hydroelastic_arm_integral's analytic sphere, its builder-v2 pads
against the sphere's foundation E/R, and, once the bed rows exist, its bed rows against the 2026-10-14 sphere-tip rows.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_curved_tips.py -q
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tip_codesign_geometry as TG  # noqa: E402

R = 0.01055
BED = ROOT / "docs/experiments/20261015-curved_tips/bed"
SPHERE_ROWS = ROOT / "docs/experiments/20261014-tip_codesign/deterioration"


def _rows(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []


def test_capsule_7p4_is_the_full_round_block():
    """The 7.4 mm capsule fills the 14.8 mm width: it is the block family's member with both fillets at 7.4 mm."""
    rng = np.random.default_rng(0)
    lo = np.array([TG.X_BACK, -TG.WIDTH / 2, TG.Z_END]) - 1e-4
    hi = np.array([TG.X_FACE, TG.WIDTH / 2, TG.Z_MOUNT]) + 1e-4
    P = lo + rng.random((200_000, 3)) * (hi - lo)
    a, b = TG.shape("tcd_cap7.4"), TG.shape("tcd_rs7.4_re7.4")
    assert np.array_equal(a.inside_design(P), b.inside_design(P))


def test_curved_keys_keep_the_mount_and_reach():
    """Every curved member keeps the palmar reach at the contact point, a flat back face and the mount face."""
    for key in ("tcd_cap6", "tcd_cap8.5", "tcd_ell7.4x8.5x11x22", "tcd_ell7.4x5x11x22", "tcd_ell7.4x8.5x16x32"):
        sh = TG.shape(key)
        assert sh.inside_design(np.array([[TG.X_FACE - 1e-7, 0.0, 0.0]]))[0]
        assert not sh.inside_design(np.array([[TG.X_FACE + 1e-6, 0.0, 0.0]]))[0]
        assert sh.inside_design(np.array([[TG.X_BACK + 1e-7, 0.0, 0.0], [0.0, 0.0, TG.Z_MOUNT - 1e-7]])).all()


def test_sphere_volume_and_centroid():
    sh = TG.shape("tcd_sph10.55")
    c, vol = sh.volume_centroid
    assert abs(vol / (4 / 3 * math.pi * R ** 3) - 1) < 2e-3
    assert np.abs(c).max() < 1e-5


def test_sphere_pads_v2_foundation_is_the_radius():
    """Builder v2 on the whole sphere: every pad's local foundation depth is R, so K_i = E a_i / R."""
    sh = TG.shape("tcd_sph10.55")
    C, Nn, A, K = sh.pads(1e7, h="local", K="each", full=True)
    assert np.allclose(K, 1e7 * A / R, rtol=2e-3)


def test_sphere_drake_field_law_matches_the_analytic_sphere():
    """The Drake-field law of the sphere's 0.25 mm mesh against hydroelastic_arm_integral (E (1 - rho / R)) at 1 and 3 N:
    approach within 1 %, onset arm within 0.5 %."""
    import curved_tips_law as CL
    import hydroelastic_arm_integral as HA
    import tip_codesign_law as L
    f = CL.DrakeFieldWide(L.shape_of("tcd_sph10.55"), res=0.00025)
    for N in (1.0, 3.0):
        d, arm, *_ = L.solve(f, 0.0, N)
        dA, (_, armA) = HA.solve(N)
        assert abs(d / dA - 1) < 0.01
        assert abs(arm / armA - 1) < 0.005


def test_upright_print_of_the_capsule_is_its_design_on_the_wall():
    """Printed upright, the capsule's cylinder is made of perimeters: within 3 mm of the contact point the printed
    surface is the design's to 1 um."""
    sh = TG.shape("tcd_cap6_u0.1")
    y = np.linspace(-3, 3, 13)
    z = np.linspace(-3, 3, 13)
    Y, Z = np.meshgrid(y, z)
    xs_p = sh.fe_surface_x(Y, TG.Z_MOUNT * 1e3 - Z)
    xs_d = sh.design.fe_surface_x(Y, TG.Z_MOUNT * 1e3 - Z)
    assert np.nanmax(np.abs(xs_p - xs_d)) < 1e-3


def test_sphere_limit_bed_rows():
    """Once the bed has run (curved_tips_bed.py), the sphere's builder-v2 pads and its Drake mesh reproduce the
    2026-10-14 sphere-tip rows (mj_pads1: 126 pads on a 35 deg cap; drake_hydro: Drake's compliant sphere) at 1 N:
    approach within 5 %, onset arm within 5 % (pads) and 3 % (Drake), spin arm at 1 rad/s within 5 %."""
    rows = {f: _rows(BED / f"{f}.jsonl") for f in ("static", "T2", "spin")}
    ref = {f: _rows(SPHERE_ROWS / f"{f}.jsonl") for f in ("static", "T2", "spin")}
    if not all(any(r.get("tip") == "tcd_sph10.55" and r.get("status") == "complete" for r in rows[f]) for f in rows):
        pytest.skip("no bed rows of tcd_sph10.55 yet")

    def pick(rs, model, **kw):
        out = [r for r in rs if r.get("model") == model and r.get("status") == "complete"
               and all(r.get(k) == v for k, v in kw.items())]
        assert out, (model, kw)
        return out[-1]
    pairs = (("pads_tcd_sph10.55", "mj_pads1", 0.05, 0.05), ("drake_tcd_sph10.55", "drake_hydro", 0.05, 0.03))
    for mine, theirs, tol_a, tol_on in pairs:
        a = pick(rows["static"], mine, N=1.0, dt_ms=1.0)["approach_um"]
        b = pick(ref["static"], theirs, N=1.0, dt_ms=1.0)["approach_um"]
        assert abs(a / b - 1) < tol_a, (mine, a, b)
        a = pick(rows["T2"], mine, N=1.0, dt_ms=1.0)["rbar_onset_mm"]
        b = pick(ref["T2"], theirs, N=1.0, dt_ms=1.0)["rbar_onset_mm"]
        assert abs(a / b - 1) < tol_on, (mine, a, b)
        a = pick(rows["spin"], mine, N=1.0, omega=1.0, dt_ms=1.0)["rbar_spin_mm"]
        b = pick(ref["spin"], theirs, N=1.0, omega=1.0, dt_ms=1.0)["rbar_spin_mm"]
        assert abs(a / b - 1) < 0.05, (mine, a, b)
