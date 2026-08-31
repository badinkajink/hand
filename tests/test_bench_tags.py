"""The tag geometry, checked without a camera.

Every number this module produces goes into a bench record that is compared against a
simulated prediction, so the arithmetic has to be verifiable offline. The fixtures build
synthetic tag poses whose answer is known by construction.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from morphohand.bench import tags as T


def rot(axis: str, deg: float) -> np.ndarray:
    a, c, s = math.radians(deg), None, None
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


@pytest.fixture
def ident_frame():
    """A camera whose axes ARE the bench axes: +x bench = +x cam, up = +z cam. The reference
    tag is oriented so its local +x is up and its local +y is bench +x."""
    R = np.column_stack([np.array([0., 0., 1.]),    # tag local x -> world up
                         np.array([1., 0., 0.]),    # tag local y -> bench +x
                         np.array([0., -1., 0.])])
    t = np.array([[0.5], [0.2], [1.0]])             # metres, arbitrary
    return T.BenchFrame.latch([(R, t)], up_axis="x", plane_axis="y", heading_deg=0.0), R, t


def test_the_floor_datum_is_the_fingertip_measurement():
    """44.0 mm is not a tuned constant: it is 65.0 measured on the bench minus the 21.0 the
    simulator puts the same edge at. If either moves, this is where it shows."""
    assert T.SIM_TO_BENCH_Z_MM == pytest.approx(65.0 - 21.0)
    assert T.bench_z_mm(T.sim_z_mm(123.4)) == pytest.approx(123.4)
    # the bench scenes' 100 mm post, in the units the tape measure works in
    assert T.bench_z_mm(T.BENCH_POST_HEIGHT_SIM_MM) == pytest.approx(144.0)


def test_tag_offset_is_half_length_plus_the_vane():
    assert T.CYL_TAG_AXIAL_MM == pytest.approx(T.CYL_HALF_LEN_MM + 21.0)


def test_deg_from_up_is_unfolded():
    """A turn to the wrong pole must not score like a correct one."""
    assert T.deg_from_up(1.0) == pytest.approx(0.0)
    assert T.deg_from_up(0.0) == pytest.approx(90.0)
    assert T.deg_from_up(-1.0) == pytest.approx(180.0)


def test_upright_shaft_reads_cos_one_and_the_centre_below_the_tag(ident_frame):
    frame, _, _ = ident_frame
    # cylinder tag 1.30 m from the camera in its own +z; its local +x (the shaft axis
    # pointing out to the tag) aligned with world up
    Rc = np.column_stack([np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., -1., 0.])])
    tc = np.array([[0.5], [0.2], [1.3]])
    r = T.reading_from_tags(frame, Rc, tc, t=0.0, shaft_axis="x")
    assert r.cos_up == pytest.approx(1.0)
    assert r.deg_from_up == pytest.approx(0.0, abs=1e-6)
    # the tag sits 300 mm above the reference in world up (1.3 m vs 1.0 m along cam +z)
    assert r.tag_z_bench_mm == pytest.approx(T.REF_TAG_BENCH_MM[2] + 300.0)
    # and the centre is 71 mm BELOW the tag, because the tag is out along the shaft
    assert r.z_bench_mm == pytest.approx(r.tag_z_bench_mm - T.CYL_TAG_AXIAL_MM)


def test_flat_shaft_reads_cos_zero_and_the_centre_beside_the_tag(ident_frame):
    frame, _, _ = ident_frame
    Rc = np.eye(3)                       # tag local x -> cam +x, which is horizontal here
    tc = np.array([[0.5], [0.2], [1.0]])
    r = T.reading_from_tags(frame, Rc, tc, t=0.0, shaft_axis="x")
    assert r.cos_up == pytest.approx(0.0, abs=1e-9)
    assert r.deg_from_up == pytest.approx(90.0)
    assert r.z_bench_mm == pytest.approx(r.tag_z_bench_mm)      # no height change
    assert r.radial_mm == pytest.approx(T.CYL_TAG_AXIAL_MM)     # ...but 71 mm sideways


def test_shaft_axis_sign_moves_the_centre_the_other_way(ident_frame):
    """The failure this guards: with --shaft-axis backwards the centre lands 142 mm off, and
    every height in the run is wrong by that much while cos merely flips sign."""
    frame, _, _ = ident_frame
    Rc = np.column_stack([np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., -1., 0.])])
    tc = np.array([[0.5], [0.2], [1.3]])
    up = T.reading_from_tags(frame, Rc, tc, t=0.0, shaft_axis="x")
    dn = T.reading_from_tags(frame, Rc, tc, t=0.0, shaft_axis="-x")
    assert dn.cos_up == pytest.approx(-up.cos_up)
    assert dn.z_bench_mm - up.z_bench_mm == pytest.approx(2 * T.CYL_TAG_AXIAL_MM)


def test_xy_is_withheld_until_a_heading_is_calibrated():
    R = np.column_stack([np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., -1., 0.])])
    frame = T.BenchFrame.latch([(R, np.array([[0.], [0.], [1.]]))], up_axis="x", plane_axis="y")
    assert frame.heading_deg is None and frame.x_cam is None
    z, radial, xy = frame.locate(np.array([100.0, 0.0, 1000.0]))
    assert xy is None                       # not (0, 0), not a guess
    assert radial == pytest.approx(100.0)   # still a real horizontal distance


def test_heading_calibration_round_trips(ident_frame):
    """Stage the cylinder somewhere known, solve the one unknown, and the solved frame must
    put it back where it was staged."""
    frame, _, _ = ident_frame
    for truth in (0.0, 37.0, -125.0):
        frame.heading_deg = truth
        p_cam = frame.ref_t_cam_mm + 120.0 * frame.x_cam + 45.0 * frame.y_cam + 10.0 * frame.up_cam
        want = (T.REF_TAG_BENCH_MM[0] + 120.0, T.REF_TAG_BENCH_MM[1] + 45.0)
        frame.heading_deg = None
        solved = frame.heading_for(p_cam, want)
        assert solved == pytest.approx(truth, abs=1e-6)
        frame.heading_deg = solved
        _, _, xy = frame.locate(p_cam)
        assert xy == pytest.approx(want, abs=1e-6)


def test_heading_calibration_refuses_a_degenerate_point(ident_frame):
    frame, _, _ = ident_frame
    with pytest.raises(ValueError):
        frame.heading_for(frame.ref_t_cam_mm + 0.5 * frame.up_cam, (T.REF_TAG_BENCH_MM[0],
                                                                    T.REF_TAG_BENCH_MM[1]))


def test_drift_catches_a_bumped_camera(ident_frame):
    frame, R, _ = ident_frame
    assert frame.drift_deg(R, "x") == pytest.approx(0.0, abs=1e-9)
    assert frame.drift_deg(rot("y", 5.0) @ R, "x") == pytest.approx(5.0, abs=1e-6)


# ---- the trace summary -----------------------------------------------------------------
def mk(t, cos, z, x=0.0, y=0.0):
    return T.Reading(t=t, cos_up=cos, deg_from_up=T.deg_from_up(cos), z_bench_mm=z,
                     radial_mm=math.hypot(x, y), tag_z_bench_mm=z + 71.0,
                     xy_bench_mm=(x, y), margin=50.0, range_mm=900.0)


def test_summary_of_a_clean_turn():
    rs = [mk(0.0, 0.02, 144.0), mk(1.0, 0.55, 150.0), mk(2.0, 0.93, 158.0)]
    s = T.summarise(rs)
    assert not s.dropped and not s.wrong_pole
    assert s.cos_start == pytest.approx(0.02) and s.cos_final == pytest.approx(0.93)
    assert s.cos_peak == pytest.approx(0.93) and s.t_peak_s == pytest.approx(2.0)
    # + means it moved TOWARD vertical
    assert s.deg_turned == pytest.approx(T.deg_from_up(0.02) - T.deg_from_up(0.93))
    assert s.deg_turned > 0
    assert s.visibility == pytest.approx(1.0)


def test_a_drop_is_a_sustained_fall_not_one_low_sample():
    steady = [mk(0.1 * i, 0.5, 144.0) for i in range(20)]
    blip = list(steady)
    blip[7] = mk(0.7, 0.5, 60.0)                      # one bad detection
    assert T.summarise(blip).dropped is False
    fell = steady[:10] + [mk(1.0 + 0.1 * i, 0.5, 60.0) for i in range(10)]
    s = T.summarise(fell)
    assert s.dropped and s.drop_at_s == pytest.approx(1.0)
    assert s.z_drop_mm == pytest.approx(84.0)


def test_the_wrong_pole_is_flagged_and_the_peak_stays_signed():
    s = T.summarise([mk(0.0, 0.0, 144.0), mk(1.0, -0.8, 144.0)])
    assert s.wrong_pole and s.cos_peak == pytest.approx(0.0)
    assert s.deg_turned < 0                            # it went AWAY from vertical
    assert any("WRONG pole" in n for n in s.notes)


def test_visibility_and_gaps_are_reported():
    rs = [mk(0.0, 0.5, 144.0), mk(2.0, 0.5, 144.0)]    # a 2 s hole
    s = T.summarise(rs, total_frames=60)
    assert s.seen == 2 and s.visibility == pytest.approx(2 / 60)
    assert s.gaps == [(0.0, 2.0)]
    assert any("dropout" in n for n in s.notes)
    assert any("visible in only" in n for n in s.notes)


def test_slip_is_horizontal_only():
    """A cylinder standing up necessarily raises its centre; counting that as slip would
    make every successful turn look like a slipping one."""
    s = T.summarise([mk(0.0, 0.0, 144.0, 0.0, 0.0), mk(1.0, 1.0, 190.0, 3.0, 4.0)])
    assert s.slip_mm == pytest.approx(5.0)


def test_an_empty_trace_says_so_rather_than_scoring_zero():
    s = T.summarise([], total_frames=100)
    assert s.seen == 0 and s.cos_peak is None and s.deg_turned is None
    assert "never detected" in " ".join(s.notes)
    assert "nothing measured" in s.line()


def test_heading_comes_from_the_mounting_not_a_calibration(ident_frame):
    """The reference tag is bolted normal to the gantry x-axis, so its heading is +-90 and
    only the SIGN is unknown -- and the sign follows from the hand sitting at bench x = 0
    while the tag sits at x = +133.5. No staging, no calibration point."""
    frame, _, _ = ident_frame
    for truth in (90.0, -90.0):
        frame.heading_deg = truth
        # a shaft in the hand: 133.5 mm back along -x from the tag, a little off in y
        p_cam = frame.ref_t_cam_mm - 133.5 * frame.x_cam + 20.0 * frame.y_cam
        frame.heading_deg = None
        got, why = frame.heading_from_mounting(p_cam)
        assert got == pytest.approx(truth)
        assert "bench x" in why
        frame.heading_deg = got
        assert frame.locate(p_cam)[2][0] == pytest.approx(0.0, abs=1e-6)


def test_mounting_heading_refuses_only_the_genuinely_degenerate_point(ident_frame):
    """The two candidates mirror the offset about the tag, so one always wins -- the sign is a
    choice between two, not a test. It is undefined only for a shaft at the tag's own x."""
    frame, _, _ = ident_frame
    frame.heading_deg = 90.0
    p_cam = frame.ref_t_cam_mm + 40.0 * frame.y_cam + 12.0 * frame.up_cam   # no x offset at all
    frame.heading_deg = None
    with pytest.raises(ValueError, match="own x"):
        frame.heading_from_mounting(p_cam)
