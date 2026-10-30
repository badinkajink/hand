"""Checks of the HOM controller (src/morphohand/control/hom.py) against closed forms and finite differences.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_hom_control.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from morphohand.control import hom as H  # noqa: E402
from morphohand.control.surfaces import Cylinder, HalfEllipsoid, Plane, Posed, RoundedBox, Sphere, unit  # noqa: E402

RNG = np.random.default_rng(0)


def brute_closest(surface_pts, p):
    i = int(np.argmin(np.linalg.norm(surface_pts - p, axis=1)))
    return surface_pts[i]


# ------------------------------------------------------------------------------------------------ surfaces

def test_sphere_queries():
    s = Sphere(0.02)
    p = np.array([0.03, -0.01, 0.02])
    d, q, n = s.closest(p)
    assert d == pytest.approx(np.linalg.norm(p) - 0.02)
    assert np.linalg.norm(q) == pytest.approx(0.02)
    assert n == pytest.approx(p / np.linalg.norm(p))
    S = s.shape_operator(q, n)
    t = unit(np.cross(n, [0.0, 0.0, 1.0]))
    assert t @ S @ t == pytest.approx(1 / 0.02)
    assert np.linalg.norm(S @ n) < 1e-12


def _box_samples(box: RoundedBox, n=60000):
    """Points on the rounded box's surface: random core points pushed out along random directions."""
    out = []
    for _ in range(n):
        v = unit(RNG.normal(size=3))
        core = np.clip(RNG.uniform(-1, 1, 3) * 3 * box.a, -box.a, box.a)
        # the support point of the core in direction v, then out by r
        sup = np.where(v > 0, box.a, -box.a) * (np.abs(v) > 1e-9) + core * (np.abs(v) <= 1e-9)
        out.append(box.c + sup + box.r * v)
    return np.array(out)


def test_rounded_box_closest_point():
    box = RoundedBox([0.0085, 0.0074, 0.011], 0.0027, centre=[0.002, 0.0, 0.0005])
    pts = _box_samples(box)
    for _ in range(40):
        p = box.c + RNG.normal(size=3) * 0.012
        d, q, n = box.closest(p)
        assert abs(box.closest(q)[0]) < 1e-12                       # q is on the surface
        if d > 0:
            assert np.linalg.norm(p - q) == pytest.approx(d, abs=1e-12)
            assert np.linalg.norm(p - brute_closest(pts, p)) >= d - 2e-4
        assert np.linalg.norm(n) == pytest.approx(1.0)


def test_rounded_box_shape_operator():
    box = RoundedBox([0.0085, 0.0074, 0.011], 0.0027)
    _, q, n = box.closest(np.array([0.05, 0.0, 0.0]))              # face
    assert np.abs(box.shape_operator(q, n)).max() == 0.0
    _, q, n = box.closest(np.array([0.05, 0.05, 0.0]))             # edge along z
    S = box.shape_operator(q, n)
    t = unit(np.cross(n, [0, 0, 1.0]))
    assert t @ S @ t == pytest.approx(1 / 0.0027)
    assert np.array([0, 0, 1.0]) @ S @ np.array([0, 0, 1.0]) == pytest.approx(0.0, abs=1e-9)
    _, q, n = box.closest(np.array([0.05, 0.05, 0.05]))            # corner
    t = unit(np.cross(n, [0, 0, 1.0]))
    assert t @ box.shape_operator(q, n) @ t == pytest.approx(1 / 0.0027)


def test_cylinder_queries():
    c = Cylinder(0.0125, 0.05)
    d, q, n = c.closest(np.array([0.02, 0.0, 0.01]))
    assert d == pytest.approx(0.0075) and n == pytest.approx([1, 0, 0])
    d, q, n = c.closest(np.array([0.0, 0.005, 0.06]))
    assert d == pytest.approx(0.01) and n == pytest.approx([0, 0, 1])
    d, q, n = c.closest(np.array([0.0, 0.01, 0.0]))               # inside: nearest is the side
    assert d == pytest.approx(-0.0025) and n == pytest.approx([0, 1, 0])
    S = c.shape_operator(np.array([0.0125, 0, 0]), np.array([1.0, 0, 0]))
    assert S[1, 1] == pytest.approx(1 / 0.0125) and S[2, 2] == pytest.approx(0.0)


def test_geodesic_step_sphere_is_great_circle():
    r = 0.02
    s = Sphere(r)
    q = np.array([r, 0.0, 0.0])
    v = np.array([0.0, 0.01, 0.0])                                 # 10 mm of arc
    e = s.geodesic_step(q, v)
    ang = 0.01 / r
    assert e == pytest.approx([r * math.cos(ang), r * math.sin(ang), 0.0], abs=1e-12)


def test_geodesic_step_cylinder():
    c = Cylinder(0.0125, 0.05)
    q = np.array([0.0125, 0.0, 0.0])
    e = c.geodesic_step(q, np.array([0.0, 0.005, 0.0]))            # around: a circle of radius r
    ang = 0.005 / 0.0125
    assert e == pytest.approx([0.0125 * math.cos(ang), 0.0125 * math.sin(ang), 0.0], abs=1e-12)
    e = c.geodesic_step(q, np.array([0.0, 0.0, 0.004]))            # along the axis: straight
    assert e == pytest.approx([0.0125, 0.0, 0.004], abs=1e-12)


def test_posed_surface():
    R = H.quat_to_mat([math.cos(0.3), 0.0, math.sin(0.3), 0.0])
    ps = Posed(Sphere(0.01), pos=[0.1, 0.0, 0.0], R=R)
    d, q, n = ps.closest(np.array([0.1, 0.0, 0.03]))
    assert d == pytest.approx(0.02) and n == pytest.approx([0, 0, 1.0])


def test_half_ellipsoid_reduces_to_sphere():
    """With every semi-axis r the half-ellipsoid answers as the sphere of radius r, inside and outside."""
    r = 0.01
    E, S = HalfEllipsoid(r, r, r, r), Sphere(r)
    rng = np.random.default_rng(3)
    for _ in range(50):
        p = rng.normal(size=3) * rng.uniform(0.002, 0.03)
        de, xe, ne = E.closest(p)
        ds, xs, ns = S.closest(p)
        assert abs(de - ds) < 1e-9 and np.allclose(xe, xs, atol=1e-9) and np.allclose(ne, ns, atol=1e-7)
        assert np.allclose(E.shape_operator(xe, ne), S.shape_operator(xs, ns), atol=1e-6)


def test_half_ellipsoid_closest_point_is_a_foot():
    """The 11/22 pad: the closest point lies on the half it belongs to, the offset is along the outward normal, and no
    sampled surface point is nearer."""
    E = HalfEllipsoid(0.0085, 0.0074, 0.011, 0.022)
    rng = np.random.default_rng(4)
    u, v = rng.uniform(0, 2 * math.pi, 4000), rng.uniform(-1, 1, 4000)
    dirs = np.stack([np.sqrt(1 - v * v) * np.cos(u), np.sqrt(1 - v * v) * np.sin(u), v], 1)
    surf = dirs * np.array([0.0085, 0.0074, 0.0])
    surf[:, 2] = np.where(dirs[:, 2] < 0, 0.011, 0.022) * dirs[:, 2]
    pts = [rng.normal(size=3) * 0.015 for _ in range(40)]
    for k in range(3):                                   # near an axis plane: the pole of the solver's equation
        for _ in range(4):
            p = rng.normal(size=3) * 0.012
            p[k] = 1e-9
            pts.append(p)
    for p in pts:
        d, x, n = E.closest(p)
        az = 0.011 if x[2] < 0 else 0.022
        assert abs((x[0] / 0.0085) ** 2 + (x[1] / 0.0074) ** 2 + (x[2] / az) ** 2 - 1.0) < 1e-9
        assert np.allclose(p - x, d * n, atol=1e-8)
        assert abs(d) <= np.linalg.norm(surf - p, axis=1).min() + 1e-6


def test_plane():
    p = Plane([0, 0, 1.0], 0.1)
    d, q, n = p.closest(np.array([0.3, 0.2, 0.15]))
    assert d == pytest.approx(0.05) and q[2] == pytest.approx(0.1)


# ------------------------------------------------------------------------------------------------ roll-slide kinematics

def _two_sphere_frames(R0, R1):
    """Object sphere at the origin, finger sphere on top; contact at (0, 0, R0); y along world y."""
    R_c0 = np.column_stack([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])           # x0 = +z, y0 = +y, z0 = -x
    R_c1 = np.column_stack([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]])           # x1 = -z, y1 = +y, z1 = +x
    K0, K1 = np.eye(2) / R0, np.eye(2) / R1
    return R_c0, R_c1, K0, K1


@pytest.mark.parametrize("R0,R1", [(0.02, 0.01), (0.0125, 0.0027), (0.05, 0.02)])
def test_roll_slide_rolling_sphere_on_sphere(R0, R1):
    """Pure rolling: the finger's centre moves at V along +y and the finger turns at -V/R1 about x; the contact point
    moves along y at R0 V / (R0 + R1) on both surfaces (derived in the docstring of test_roll_slide_sliding)."""
    V = 0.02
    R_c0, R_c1, K0, K1 = _two_sphere_frames(R0, R1)
    w_world = np.array([-V / R1, 0.0, 0.0])
    V_ref = np.concatenate([R_c1.T @ w_world, np.zeros(3)])
    g0, g1 = H.roll_slide_rates(V_ref, K0, K1, R_c1.T @ R_c0, np.zeros(3))
    g = R0 * V / (R0 + R1)
    assert g0 == pytest.approx([g, 0.0]) and g1 == pytest.approx([g, 0.0])


def test_roll_slide_sliding():
    """Pure sliding without rotation: the finger's centre moves at V along +y. On the object the contact point moves at
    R0 V/(R0+R1); on the finger at -R1 V/(R0+R1) (the contact direction turns while the finger does not)."""
    R0, R1, V = 0.02, 0.01, 0.01
    R_c0, R_c1, K0, K1 = _two_sphere_frames(R0, R1)
    V_ref = np.concatenate([np.zeros(3), R_c1.T @ np.array([0.0, V, 0.0])])
    g0, g1 = H.roll_slide_rates(V_ref, K0, K1, R_c1.T @ R_c0, np.zeros(3))
    assert g0 == pytest.approx([R0 * V / (R0 + R1), 0.0])
    assert g1 == pytest.approx([-R1 * V / (R0 + R1), 0.0])


def test_roll_slide_rolling_numeric():
    """The same rolling case integrated geometrically: the contact point found from the moved bodies after a small
    step matches g dt on both surfaces."""
    R0, R1, V, dt = 0.02, 0.01, 0.02, 1e-4
    c = np.array([0.0, 0.0, R0 + R1])
    c2 = c + np.array([0.0, V * dt, 0.0])
    c2 = c2 / np.linalg.norm(c2) * (R0 + R1)                       # stays on the rolling circle
    th = -V / R1 * dt                                              # finger rotation about x
    Rf = np.array([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])
    p_obj = R0 * unit(c2)                                          # new contact point on the object
    p_fing_body = Rf.T @ (p_obj - c2)                              # on the finger, in its (initial) body frame
    dg0 = (p_obj - np.array([0.0, 0.0, R0]))[1] / dt
    dg1 = (p_fing_body - np.array([0.0, 0.0, -R1]))[1] / dt
    R_c0, R_c1, K0, K1 = _two_sphere_frames(R0, R1)
    V_ref = np.concatenate([R_c1.T @ np.array([-V / R1, 0, 0]), np.zeros(3)])
    g0, g1 = H.roll_slide_rates(V_ref, K0, K1, R_c1.T @ R_c0, np.zeros(3))
    assert g0[0] == pytest.approx(dg0, rel=1e-3) and g1[0] == pytest.approx(dg1, rel=1e-3)


# ------------------------------------------------------------------------------------------------ MuJoCo: Eq. (1), (2)

XML = """
<mujoco>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="l1" pos="0 0 0.1">
      <joint name="j0" type="hinge" axis="0 0 1" range="-1 1"/>
      <joint name="j1" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
      <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.005" mass="0.02"/>
      <body name="tip" pos="0.05 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom name="tipgeom" type="sphere" pos="0.03 0 0" size="0.01" mass="0.01"/>
      </body>
    </body>
    <body name="obj" pos="0.08 0 0.06">
      <freejoint/>
      <geom name="objgeom" type="sphere" size="0.02" mass="0.025"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="j0" kp="4"/><position joint="j1" kp="4"/><position joint="j2" kp="4"/>
  </actuator>
</mujoco>
"""


@pytest.fixture(scope="module")
def mirror():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_string(XML)
    fs = H.FingerSpec("f", ["j0", "j1", "j2"], "tip", "tipgeom", "j2", Posed(Sphere(0.01), pos=[0.03, 0, 0]))
    os_ = H.ObjectSpec("obj", "objgeom", Posed(Sphere(0.02)))
    return H.Mirror(m, [fs], os_)


def _rel_finite(mr, q, u, pos, quat, Vb, dt=1e-6):
    """Relative twist of the finger's material point at {L1}'s origin w.r.t. the object's coincident material point,
    by finite differences of the moved bodies, in {L1}'s axes."""
    mr.set_state({"f": q}, pos, quat)
    g = mr.gcf_witness("f")
    pb, Rb = mr.body_pose("f")
    o, Ro = mr.obj_pose()
    a_f = Rb.T @ (g.p1 - pb)                                       # the point in the finger body's frame
    a_o = Ro.T @ (g.p1 - o)                                        # the same point attached to the object
    # move the joints and the object
    w = Ro @ Vb[:3]
    ang = np.linalg.norm(w) * dt
    if ang > 0:
        k = w / np.linalg.norm(w)
        dq = np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * k])
    else:
        dq = np.array([1.0, 0, 0, 0])
    import mujoco
    q2 = np.zeros(4)
    mujoco.mju_mulQuat(q2, dq, np.asarray(quat, float))
    pos2 = pos + (Ro @ Vb[3:]) * dt
    mr.set_state({"f": q + u * dt}, pos2, q2)
    pb2, Rb2 = mr.body_pose("f")
    o2, Ro2 = mr.obj_pose()
    v_rel = ((pb2 + Rb2 @ a_f) - (o2 + Ro2 @ a_o)) / dt
    # relative rotation: finger minus object, from the body rotations
    Rrel = (Rb2 @ Rb.T) @ (Ro2 @ Ro.T).T
    wv = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]]) / (2 * dt)
    return np.concatenate([g.R1.T @ wv, g.R1.T @ v_rel]), g


def test_eq1_rows_match_finite_differences(mirror):
    ctl = H.HomController(mirror)
    for _ in range(5):
        q = RNG.uniform(-0.6, 0.6, 3)
        u = RNG.normal(size=3)
        pos = np.array([0.08, 0.0, 0.06]) + RNG.normal(size=3) * 0.005
        quat = unit(RNG.normal(size=4))
        Vb = RNG.normal(size=6) * np.array([1, 1, 1, 0.02, 0.02, 0.02])
        fd, g = _rel_finite(mirror, q, u, pos, quat, Vb)
        mirror.set_state({"f": q}, pos, quat)
        A, b = ctl.eq1_rows("f", mirror.gcf_witness("f"), Vb)
        assert A @ u + b == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_gcf_convention(mirror):
    """x1 points from the finger toward the object, so a positive v_sep closes the gap."""
    q = np.array([0.0, 0.0, 0.0])
    mirror.set_state({"f": q}, np.array([0.08, 0.0, 0.06]), np.array([1.0, 0, 0, 0]))
    g = mirror.gcf_witness("f")
    assert g.dist > 0
    assert (g.p0 - g.p1) @ g.x1 == pytest.approx(g.dist, rel=1e-6)
    ctl = H.HomController(mirror)
    A, b = ctl.eq1_rows("f", g, np.zeros(6))
    u = np.linalg.lstsq(A[3:4], np.array([0.01]), rcond=None)[0]  # some u with v_sep = +10 mm/s
    mirror.set_state({"f": q + u * 1e-3}, np.array([0.08, 0.0, 0.06]), np.array([1.0, 0, 0, 0]))
    assert mirror.gcf_witness("f").dist < g.dist


def test_impedance_target_on_the_surface_from_a_lagging_reference():
    """'reference' impedance: with theta_ref lagging 1.6 mm behind the finger, the servo target's action point still
    sits on the measured surface and presses F sum_j (J^T x1)_j^2 / k_p,j into it; the earlier 'gap' mode adds the gap
    measured at the finger to the lagging reference and stops short by the lag."""
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_string(XML.replace("<mujoco>", '<mujoco><compiler angle="radian"/>', 1))
    fs = H.FingerSpec("f", ["j0", "j1", "j2"], "tip", "tipgeom", "j2", Posed(Sphere(0.01), pos=[0.03, 0, 0]))
    mirror = H.Mirror(m, [fs], H.ObjectSpec("obj", "objgeom", Posed(Sphere(0.02))))
    q = np.zeros(3)
    pos, quat = np.array([0.08, 0.0, 0.068]), np.array([1.0, 0.0, 0.0, 0.0])
    F = 0.5
    lag = np.array([0.0, -0.02, 0.0])                       # j1 turned away from the object: the tip 1.6 mm higher
    mirror.set_state({"f": q}, pos, quat)
    g = mirror.gcf_witness("f")
    Jp, _ = mirror.jac("f", g.p1)
    press = F * float(((Jp.T @ g.x1) ** 2 / mirror.kp["f"]).sum())
    behind = -g.x1 @ (Jp @ lag)
    assert g.dist == pytest.approx(0.002, abs=2e-4) and behind == pytest.approx(0.0016, abs=1e-4)
    advance = {}
    for mode in ("reference", "gap"):
        ctl = H.HomController(mirror, H.HomParams(use_filter=False, gravity_ff=False, dls=1e-6, impedance=mode))
        ctl.reset({"f": q + lag})
        out, _ = ctl.step(H.Measurement(q={"f": q}, obj_pos=pos, obj_quat=quat),
                          H.HandCommand(fingers={"f": H.FingerCommand(force=F)}))
        advance[mode] = g.x1 @ (Jp @ (out["f"] - q))       # how far the target moves the action point toward the object
    assert advance["reference"] == pytest.approx(g.dist + press, rel=1e-3)
    assert advance["gap"] == pytest.approx(g.dist + press - behind, rel=1e-3)


def test_reanchor_in_contact_beyond_the_cut():
    """HomParams.reanchor: a touching finger whose reference lies more than gap_max behind the surface is set to its
    measured angles, and the target presses F from the surface; without it the cut leaves the target short by the lag
    less the cut. Apart from the object the rule does nothing."""
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_string(XML.replace("<mujoco>", '<mujoco><compiler angle="radian"/>', 1))
    fs = H.FingerSpec("f", ["j0", "j1", "j2"], "tip", "tipgeom", "j2", Posed(Sphere(0.01), pos=[0.03, 0, 0]))
    mirror = H.Mirror(m, [fs], H.ObjectSpec("obj", "objgeom", Posed(Sphere(0.02))))
    q, quat = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    lag = np.array([0.0, -0.02, 0.0])                       # the reference 1.6 mm behind the finger
    F, cut = 0.5, 0.0005
    for z, touching in ((0.0705, True), (0.068, False)):   # the object 0.5 mm into the tip, or 2 mm below it
        pos = np.array([0.08, 0.0, z])
        mirror.set_state({"f": q}, pos, quat)
        g = mirror.gcf_witness("f")
        assert (g.dist <= 0.0) == touching
        Jp, _ = mirror.jac("f", g.p1)
        press = F * float(((Jp.T @ g.x1) ** 2 / mirror.kp["f"]).sum())
        behind = -g.x1 @ (Jp @ lag)
        assert g.dist + behind > cut
        advance = {}
        for ra in (False, True):
            ctl = H.HomController(mirror, H.HomParams(use_filter=False, gravity_ff=False, dls=1e-6, gap_max=cut,
                                                      reanchor=ra))
            ctl.reset({"f": q + lag})
            out, info = ctl.step(H.Measurement(q={"f": q}, obj_pos=pos, obj_quat=quat),
                                 H.HandCommand(fingers={"f": H.FingerCommand(force=F)}))
            advance[ra] = g.x1 @ (Jp @ (out["f"] - q))
            assert info["f"]["reanchored"] == (ra and touching)
        if touching:
            assert advance[False] == pytest.approx(cut + press - behind, abs=2e-6)
            assert advance[True] == pytest.approx(press, abs=2e-6)
        else:
            assert advance[True] == pytest.approx(advance[False], abs=1e-12)


def test_step_info_parts_sum_to_the_target(mirror):
    """The step's info holds the period's intermediates: the reference moves by dt u (inside the lead and the limits),
    and the servo target is that reference plus the gap, force and gravity offsets."""
    ctl = H.HomController(mirror, H.HomParams(use_filter=False))
    q = np.array([0.1, 0.2, 0.3])
    meas = H.Measurement(q={"f": q}, obj_pos=np.array([0.08, 0.0, 0.06]), obj_quat=np.array([1.0, 0, 0, 0]))
    cmd = H.HandCommand(fingers={"f": H.FingerCommand(force=1.0, v_ref=np.array([0, 0, 0, 0.01, 0, 0]))})
    out, info = ctl.step(meas, cmd)
    i = info["f"]
    lo, hi, lead = mirror.lo["f"], mirror.hi["f"], ctl.prm.lead      # this model's ranges are degrees (+-1, +-1.5 deg)
    step = np.clip(i["theta_ref_prev"] + ctl.prm.dt * i["u"], q - lead, q + lead)
    assert i["theta_ref"] == pytest.approx(np.clip(step, lo, hi), abs=1e-12)
    assert out["f"] == pytest.approx(np.clip(i["theta_ref"] + i["off_gap"] + i["off_force"] + i["off_grav"], lo, hi),
                                     abs=1e-12)
    assert i["A"] @ i["u"] + i["b"] == pytest.approx(i["v_plan"])


def test_eq2_unbounded_matches_closed_form():
    A = RNG.normal(size=(6, 3))
    b = RNG.normal(size=6) * 0.01
    v_ref = RNG.normal(size=6) * 0.01
    w = RNG.uniform(0.5, 2.0, 6)
    th = np.zeros(3)
    u = H.solve_eq2(A, b, v_ref, w, th, th, -10 * np.ones(3), 10 * np.ones(3), 0.01, None, w_damp=1e-3, u_max=100)
    Wh = np.diag(np.sqrt(w) * H.SCALE)
    M = np.vstack([Wh @ A, math.sqrt(1e-3) * np.eye(3)])
    r = np.concatenate([Wh @ (v_ref - b), np.zeros(3)])
    assert u == pytest.approx(np.linalg.lstsq(M, r, rcond=None)[0], rel=1e-6, abs=1e-9)


def test_eq2_respects_bounds():
    A = np.eye(6)[:, :3]
    u = H.solve_eq2(A, np.zeros(6), np.array([5.0, 0, 0, 0, 0, 0]), np.ones(6), np.zeros(3), np.array([0.99, 0, 0]),
                    -np.ones(3), np.ones(3), 0.01, None, u_max=2.0)
    assert u[0] == pytest.approx(1.0, abs=1e-6)                     # (1 - 0.99) / 0.01


def test_pusher_twist_is_rotation_about_pinch_axis():
    R_bm = H.quat_to_mat(unit(np.array([0.9, 0.1, -0.3, 0.2])))
    p_bm = np.array([0.01, -0.02, 0.005])
    V = H.pusher_body_twist(0.7, R_bm, p_bm)
    assert V[:3] == pytest.approx(0.7 * R_bm[:, 0])
    for lam in (-0.03, 0.0, 0.05):                                  # points on the pinch axis do not move
        x = p_bm + lam * R_bm[:, 0]
        assert np.linalg.norm(V[3:] + np.cross(V[:3], x)) < 1e-12


# ------------------------------------------------------------------------------------------------ filter

class _FakeMirror:
    """Object sphere at the origin, finger sphere above it, identity orientations; joint axis along y."""

    def __init__(self, R0, R1):
        self.R0, self.R1 = R0, R1
        self.c = np.array([0.0, 0.0, R0 + R1])

    def obj_pose(self):
        return np.zeros(3), np.eye(3)

    def body_pose(self, f):
        return self.c.copy(), np.eye(3)

    def joint_axis(self, f):
        return np.array([0.0, 1.0, 0.0])


def test_filter_smooths_static_noise():
    R0, R1 = 0.02, 0.01
    fs = H.FingerSpec("f", [], "tip", "tipgeom", "j", Posed(Sphere(R1)))
    os_ = H.ObjectSpec("obj", "objgeom", Posed(Sphere(R0)))
    flt = H.ContactFilter(fs, os_, H.FilterParams(k_min=5.0, k_max=100.0, alpha=0.02))
    mr = _FakeMirror(R0, R1)
    true = np.array([0.0, 0.0, R0])
    errs, raw = [], []
    for _ in range(400):
        noise = np.array([RNG.normal() * 1e-3, RNG.normal() * 1e-3, 0.0])
        p = true + noise
        p0 = R0 * unit(p)
        p1 = mr.c + R1 * unit(p - mr.c)
        g = H.Gcf(0.0, p1, p0, H.frame_from_normal(np.array([0, 0, -1.0]), np.array([0, 1.0, 0]), np.zeros(3)), "cop")
        w1, n1 = flt.update(mr, "f", g, np.zeros(6), 0.002)
        errs.append(np.linalg.norm(w1 - true))
        raw.append(np.linalg.norm(p1 - true))
    assert flt.k[0] == pytest.approx(5.0) and flt.k[1] == pytest.approx(5.0)
    assert np.mean(errs[100:]) < 0.35 * np.mean(raw[100:])


# ------------------------------------------------------------------------------------------------ bench adapter

def test_hom_bench_simulated_stream():
    """hom_bench on a simulated stream: 2 s of the roll task on the 40 mm sphere (approach, squeeze, lift) at 50 Hz in
    closed loop through the bench units (servo degrees, the object pose and contact estimates in the palm frame in mm).
    On the same plant states the controller run directly gives the same servo targets, a replay of the recorded stream
    gives them again, and a plan whose mounts are not the design's layout is refused."""
    pytest.importorskip("mujoco")
    import json
    from morphohand.control import hom_bench as B
    try:
        bench = B.HomBench.from_design("roll", 40.0)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, RuntimeError) as e:
        pytest.skip(f"CAD-hand cell unavailable: {e}")
    stream, _ = B.simulate(bench, seed=0, seconds=2.0)
    assert len(stream) == 100
    assert {s["phase"] for s in stream} == {"open", "close", "squeeze", "lift"}
    assert all(not s["clipped"] for s in stream)
    assert any(len(s["contacts"]) == 3 for s in stream)                        # the pads touch before the lift
    T = bench.T
    plant = T.Plant(bench.p_pads, bench.setup)
    plant.reset(bench.q_open, 0)
    ctl = H.HomController(bench.mirror, bench.prm, healthy=bench.q_open)
    sch = T.Schedule("roll", bench.setup, T.PRM)
    t, lifted = 0.0, False
    for smp in stream:
        s = plant.state()
        if t >= T.T_LIFT and not lifted:
            plant.post_off()
            lifted = True
        cmd, _ = sch(t, s)
        q_cmd, _ = ctl.step(H.Measurement(q=s["q"], obj_pos=s["pos"], obj_quat=s["quat"], cop=dict(s["cop"])), cmd)
        q_bench = B.sim_from_servo(smp["targets"])
        for f in B.FINGERS:
            assert q_bench[f] == pytest.approx(q_cmd[f], abs=1e-6)
        plant.set_targets(q_bench)                                             # the states of the simulated stream
        t += bench.prm.dt                                                      # accumulated as simulate() does
        plant.advance(t)
    for a, b in zip(B.replay(bench, stream), stream):
        for f in B.FINGERS:
            for sv in ("aa", "fe1", "fe2"):
                assert a["targets"][f][sv] == pytest.approx(b["targets"][f][sv], abs=1e-6)
    plan = {"design": bench.setup["design"]["tag"],
            "mounts_palm_mm": {f: (1e3 * bench.setup["mounts"][f][:2]).tolist() for f in B.FINGERS},
            "poses": [{"name": "open", "ramp_s": 0.0, "hold_s": 0.0,
                       "joints": {f: dict(zip(B.SIM_JOINTS, np.degrees(bench.q_open[f]).tolist())) for f in B.FINGERS}}]}
    pb = B.HomBench.from_plan(plan, "roll", 40.0)
    for f in B.FINGERS:
        assert pb.q_open[f] == pytest.approx(bench.q_open[f], abs=1e-12)
    plan["mounts_palm_mm"]["index"][0] += 2.0
    with pytest.raises(ValueError):
        B.HomBench.from_plan(plan, "roll", 40.0)
