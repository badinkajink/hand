# pyright: reportMissingImports=false
"""Contact laws of the production fingertip pads (reorient_backends.replace_tips, the default 1 mm TPU-tip pads), on the
rig of scripts/pad_contact_rig.py: the tracked real_v1 screwdriver scene with the production pads, the hand rigid, gravity
off, only the thumb's pads colliding with the tool. Known forces and torques on the tool are compared with closed-form
values: each pad's spring E a_s / h, the pressure law integrated over the block's surface, the soft-friction creep law,
the Coulomb and torsional slip limits, and the stiffness's independence of the tool's mass when the pads are rebuilt.
These laws are those of pad builder v1 (one stiffness per finger, h = 8.5 mm), so their rigs are pinned to it; the
tests of builder v2, the default since 2026-10-15, check that each pad's spring is E a_i / h_i with h_i its local
depth below the tip's volume centroid, on the tpu2.7 block and on a CAD-envelope tip.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_pad_contact_laws.py -q
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fingertip_geometry as G  # noqa: E402
import reorient_backends as RB  # noqa: E402
from pad_contact_rig import FILLET, MU, Rig, continuum, continuum_approach, realised_stiffness  # noqa: E402

ROUND = 5e-5                                     # the builder writes d0 with 6 significant digits
V1 = "v1"                                        # the builder whose laws the analytic tests below state


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    return Rig(tmp_path_factory.mktemp("pads"), builder=V1)


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
    r = Rig(tmp_path, impratio=impratio, builder=V1)
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
    r = Rig(tmp_path, mass_scale=scale, builder=V1)
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
    r = Rig(tmp_path, stale_scale=scale, builder=V1)
    ratio = rig.inverse_weight() / r.inverse_weight()
    assert abs(ratio - 1) > 0.5                                 # x0.25: -61 %, x4: +65 % for this tip and tool
    assert realised_stiffness(r.jointed, "thumb") == pytest.approx(ratio * rig.meta["pad_K"], rel=ROUND)
    r.press(1.0)
    for f, pen, _, _ in r.contacts():
        assert f / pen == pytest.approx(ratio * rig.meta["pad_K"], rel=ROUND)


@pytest.mark.parametrize("E", [5e6, 2e7])
def test_pad_stiffness_scales_with_the_modulus(rig, tmp_path, E):
    r = Rig(tmp_path, E=E, builder=V1)
    assert r.meta["pad_K"] == pytest.approx(rig.meta["pad_K"] * E / RB.E_TPU, rel=1e-12)
    r.press(1.0)
    for f, pen, _, _ in r.contacts():
        assert f / pen == pytest.approx(rig.meta["pad_K"] * E / RB.E_TPU, rel=ROUND)


# ------------------------------------------------------------------------------------------------ builder v2

def pad_stiffness(m, finger: str, tool: str = RB.OBJ) -> np.ndarray:
    """Static force per metre of penetration of each of `finger`'s pads (in pad-index order) against `tool`, N/m."""
    A = m.body_invweight0[m.body(f"{finger}_tip").id, 0] + m.body_invweight0[m.body(tool).id, 0]
    ids = sorted((int(m.geom(i).name[len(finger) + 4:]), i) for i in range(m.ngeom)
                 if m.geom(i).name.startswith(f"{finger}_pad"))
    return np.array([1.0 / ((1.0 - m.geom_solimp[i, 1]) * m.geom_solref[i, 0] ** 2 * A) for _, i in ids])


@pytest.fixture(scope="module")
def rig2(tmp_path_factory):
    return Rig(tmp_path_factory.mktemp("pads_v2"), builder="v2")


def test_v2_each_pad_is_e_a_over_its_local_depth_on_the_block(rig2):
    """tpu2.7 block: the local depth h_i = n_i . (p_i - c) from the block's own surface samples, c the box centre; h is
    the half-depth on the palmar face, the half-width on the sides, the half-length on the end and up to |HALF - r| + r
    on the corner spheres."""
    p, n, a = G.surface_samples(FILLET, RB.PAD_S, 1.0)
    keep = G.pad_mask(n, 1.0)
    p, n, a = p[keep], n[keep], a[keep]
    h = np.einsum("ij,ij->i", n, p - G.CENTRE)
    for axis, half in ((0, G.HALF[0]), (1, G.HALF[1]), (2, G.HALF[2])):
        face = np.abs(np.abs(n[:, axis]) - 1) < 1e-12
        assert face.any() and np.allclose(h[face], half, rtol=1e-12)
    corner = np.linalg.norm(G.HALF - FILLET) + FILLET                 # n . (HALF - r) + r <= |HALF - r| + r
    assert np.all(h <= corner + 1e-12) and h.max() > 0.98 * corner and h.min() == pytest.approx(G.HALF.min())
    K = pad_stiffness(rig2.jointed, "thumb")
    assert len(K) == len(a) == rig2.meta["n_pads"]["thumb"]
    np.testing.assert_allclose(K, RB.E_TPU * a / h, rtol=ROUND)
    assert rig2.meta["pad_builder"] == "v2"


def test_v2_pressed_pads_carry_their_own_springs_and_balance_the_push(rig2):
    rig2.press(1.0)
    K = pad_stiffness(rig2.m, "thumb")
    m, d, f6 = rig2.m, rig2.d, np.zeros(6)
    push = 0.0
    n_seen = 0
    for i in range(d.ncon):
        c = d.contact[i]
        names = [m.geom(int(g)).name for g in c.geom]
        pad = next((x for x in names if x.startswith("thumb_pad")), None)
        if pad is None or c.dist >= 0:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        assert f6[0] / -c.dist == pytest.approx(K[int(pad[9:])], rel=1e-6)
        push += f6[0] * abs(float(c.frame[:3] @ rig2.n))
        n_seen += 1
    assert n_seen >= 20
    assert push == pytest.approx(1.0, rel=1e-3)


def test_v2_cad_envelope_tip_uses_the_local_depth(tmp_path):
    """tcd_rs3_re3 (the as-built insert's design surface) on the screwdriver scene: every pad's spring is E a_i / h_i
    with the shape's pad areas, and on the flat palmar face h_i is the face's distance from the volume centroid, which
    a grid integral of the shape gives independently."""
    import xml.etree.ElementTree as ET
    import tip_codesign_geometry as TG
    from pad_contact_rig import SRC
    sh = TG.shape("tcd_rs3_re3")
    root = ET.parse(SRC).getroot()
    _, meta = RB.replace_tips(root, "tcd_rs3_re3", "pads", MU, tmp_path, tool=RB.OBJ)
    assert meta["pad_builder"] == "v2" and meta["tip_key"] == "tcd_rs3_re3"
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    C, N, A, K_shape = sh.pads(RB.E_TPU, RB.PAD_S, RB.PAD_RS, 1.0, h="local", K="each", full=True)
    K = pad_stiffness(m, "thumb")
    np.testing.assert_allclose(K, K_shape, rtol=ROUND)
    ax = [(lo + (np.arange(k) + 0.5) * (hi - lo) / k) for lo, hi, k in
          ((TG.X_BACK, TG.X_FACE, 113), (-TG.WIDTH / 2, TG.WIDTH / 2, 99), (TG.Z_END, TG.Z_MOUNT, 143))]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    c_x = float(P[sh.inside_design(P), 0].mean())
    face = N[:, 0] > 1 - 1e-9
    assert face.sum() > 50
    h_face = RB.E_TPU * A[face] / K[face]
    np.testing.assert_allclose(h_face, TG.X_FACE - c_x, rtol=2e-3)
