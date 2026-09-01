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
    # 27 mm of vane since 2026-08-31, when it was rebuilt to carry the 40 mm tag; 21 before.
    assert T.CYL_TAG_AXIAL_MM == pytest.approx(T.CYL_HALF_LEN_MM + 27.0)
    assert T.CYL_TAG_SIZE_M == pytest.approx(0.040)


def test_rigid_reference_and_camera_mounting_is_machine_readable():
    assert T.REF_TAG_BENCH_MM[0] == pytest.approx(133.5)
    assert T.REF_TAG_NORMAL_BENCH_AXIS.startswith("x ")
    assert T.CAMERA_VIEW_BENCH_AXIS.startswith("x ")
    assert set(T.REF_TAG_PLANE_HORIZONTAL_BENCH_CANDIDATES) == {"+y", "-y"}
    assert T.REF_TAG_MOUNTING_FIXED is True
    frame = T.BenchFrame(np.array([0., 0., 1.]), np.array([1., 0., 0.]),
                         np.zeros(3))
    assert "opposes" in frame.to_json()["rigid_mounting"]["camera_view_bench_axis"]


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


def test_mounting_heading_refuses_a_shaft_parked_beside_the_reference_tag(ident_frame):
    """The regression from the first live arm, 2026-08-31.

    Staging had the shaft on its post next to the reference tag rather than in the hand. The
    two mounting candidates came out at bench x +114 and +153 -- mirrored about the tag's own
    +133.5 -- and the first version of this method just took the smaller and reported it as
    "the hand's side". Both are outside the hand, so the sign was a coin flip that would have
    been written into the trace as measured x/y. Neither inside the envelope must refuse.
    """
    frame, _, _ = ident_frame
    frame.heading_deg = 90.0
    # 19.5 mm off the tag's own x, i.e. candidates at 133.5 -+ 19.5
    p_cam = (frame.ref_t_cam_mm - 19.5 * frame.x_cam
             + 40.0 * frame.y_cam + 12.0 * frame.up_cam)
    frame.heading_deg = None
    with pytest.raises(ValueError, match=r"\+114 and \+153"):
        frame.heading_from_mounting(p_cam)


def test_mounting_heading_resolves_a_shaft_staged_in_the_hand(ident_frame):
    """The staging the automatic flow actually arms in: the shaft near the palm centre, so one
    candidate lands inside the hand envelope and the mirrored one lands far outside it."""
    frame, _, _ = ident_frame
    for truth in (90.0, -90.0):
        frame.heading_deg = truth
        p_cam = (frame.ref_t_cam_mm - 133.5 * frame.x_cam        # shaft at bench x ~ 0
                 + 25.0 * frame.y_cam + 12.0 * frame.up_cam)
        frame.heading_deg = None
        got, why = frame.heading_from_mounting(p_cam)
        assert got == pytest.approx(truth)
        assert "+267" in why or "-267" in why   # the mirrored candidate, named in the reason


def test_mounting_heading_is_undefined_at_the_reference_tags_own_x(ident_frame):
    """Both candidates collapse onto 133.5, which is outside the envelope, so it still refuses."""
    frame, _, _ = ident_frame
    frame.heading_deg = 90.0
    p_cam = frame.ref_t_cam_mm + 40.0 * frame.y_cam + 12.0 * frame.up_cam   # no x offset at all
    frame.heading_deg = None
    with pytest.raises(ValueError, match="a guess"):
        frame.heading_from_mounting(p_cam)


def test_hold_metric_is_the_pose_that_was_KEPT_not_the_best_instant(ident_frame):
    """The study's primary metric. A shaft flicked upright and then lost must not score like one
    carried there and held, which is exactly what `cos_peak` does."""
    frame, _, _ = ident_frame
    frame.heading_deg = 90.0

    def trace(cos_series, dt=0.05):
        out = []
        for i, c in enumerate(cos_series):
            r = T.Reading(t=i * dt, cos_up=c, deg_from_up=T.deg_from_up(c),
                          z_bench_mm=100.0, radial_mm=50.0, tag_z_bench_mm=171.0,
                          xy_bench_mm=(0.0, 0.0), margin=60.0, range_mm=400.0)
            out.append(r)
        return out

    # 2 s: a spike to vertical at 1 s, then settled back to horizontal and held there
    flicked = trace([0.0] * 20 + [1.0] + [0.05] * 19)
    kept = trace([0.0] * 20 + [0.9] * 20)
    a, b = T.summarise(flicked), T.summarise(kept)
    assert a.cos_peak == pytest.approx(b.cos_peak, abs=0.11)   # the peak cannot tell them apart
    assert a.cos_hold < 0.2 and b.cos_hold > 0.85              # the hold metric can
    assert a.hold_window_s == pytest.approx(T.HOLD_WINDOW_S)


def test_hold_metric_falls_back_to_the_last_sample_on_a_trace_shorter_than_the_window():
    frame_free = [T.Reading(t=0.0, cos_up=0.4, deg_from_up=T.deg_from_up(0.4), z_bench_mm=100.0,
                            radial_mm=50.0, tag_z_bench_mm=171.0, xy_bench_mm=None,
                            margin=60.0, range_mm=400.0)]
    assert T.summarise(frame_free).cos_hold == pytest.approx(0.4)


def test_a_centre_below_the_bench_floor_is_reported_as_a_shaft_axis_sign_error():
    """The 2026-08-31 17:46 run: the tag itself fell 44 mm while the computed centre 'fell' 102
    and ended 46 mm UNDER the bench. The centre is tag - 71*u, so a --shaft-axis pointing from
    the tag toward the centre puts it 2 x 71 x cos on the wrong side. Flat on the post the error
    is horizontal and invisible; it only appears once the tool stands up.

    The tag heights are the ones that trace recorded; the offset is deliberately read from
    CYL_TAG_AXIAL_MM rather than pinned at the 71 mm of that day, because `summarise` builds
    its note from the same constant. Pinning one and not the other would test a mismatch that
    cannot occur. The incident being reproduced is the SIGN, which no vane rebuild changes."""
    def reading(t, cos, tag_z):
        z = tag_z - T.CYL_TAG_AXIAL_MM * cos          # exactly how the tracker computes it
        return T.Reading(t=t, cos_up=cos, deg_from_up=T.deg_from_up(cos), z_bench_mm=z,
                         radial_mm=40.0, tag_z_bench_mm=tag_z, xy_bench_mm=None,
                         margin=60.0, range_mm=320.0)

    s = T.summarise([reading(0.0, 0.049, 59.3), reading(1.5, 0.60, 30.0),
                     reading(3.0, 0.855, 14.8)])
    assert s.below_floor_mm is not None and s.below_floor_mm < -40
    note = " ".join(s.notes)
    assert "--shaft-axis" in note and "below the bench floor" in note
    # and it names the height the other sign would have given, so the operator can sanity-check it
    other_sign = max(14.8 + T.CYL_TAG_AXIAL_MM * 0.855, 30.0 + T.CYL_TAG_AXIAL_MM * 0.60)
    assert f"{other_sign:.0f} mm" in note

    upright = T.summarise([reading(0.0, 0.0, 130.0), reading(3.0, 0.9, 200.0)])
    assert upright.below_floor_mm is None


def test_symmetric_folds_the_pole_but_never_the_axial_geometry(ident_frame):
    """A plain cylinder has no distinguished end, so which pole ends up on top is a property of
    how the operator seated it. The axial offset is NOT foldable: it is geometry, and it always
    runs centre -> tag."""
    frame, _, _ = ident_frame
    frame.heading_deg = 90.0
    # tag frame whose local +x points DOWN in the world: the vane sits on the lower end
    R = np.column_stack([-frame.up_cam, frame.plane_cam,
                         np.cross(-frame.up_cam, frame.plane_cam)])
    t = ((frame.ref_t_cam_mm + 30.0 * frame.plane_cam) / 1000.0).reshape(3, 1)
    signed = T.reading_from_tags(frame, R, t, t=0.0, shaft_axis="x")
    folded = T.reading_from_tags(frame, R, t, t=0.0, shaft_axis="x", symmetric=True)
    assert signed.cos_up == pytest.approx(-1.0, abs=1e-6)
    assert folded.cos_up == pytest.approx(1.0, abs=1e-6)
    assert folded.deg_from_up == pytest.approx(0.0, abs=1e-4)
    # the centre is 71 mm from the tag along the shaft, on the same side either way
    assert folded.z_bench_mm == pytest.approx(signed.z_bench_mm)
    assert folded.tag_z_bench_mm == pytest.approx(signed.tag_z_bench_mm)
    # and a folded trace can never trip the wrong-pole flag, because there is no wrong pole
    assert not T.summarise([folded]).wrong_pole


def test_a_trace_that_goes_dark_before_the_end_has_no_hold():
    """Two runs on 2026-08-31 lost the tag at 1.5 s of a 3.4 s trajectory and were scored
    0.359 and 0.386 -- numbers taken from frames in which the shaft was still turning. The
    hold window belongs to when RECORDING stopped, and a window with nothing in it is a void,
    not a low score."""
    rs = [mk(t, c, 80.0) for t, c in [(0.0, 0.05), (0.7, 0.40), (1.4, 0.70)]]
    s = T.summarise(rs, total_frames=90, trace_end_s=3.4)
    assert s.cos_hold is None
    assert any("never observed" in n for n in s.notes)
    assert s.cos_peak == pytest.approx(0.70)     # the rest of the trace is still real
    # ... and a run whose tag was visible to the end is scored exactly as before.
    kept = rs + [mk(3.2, 0.90, 80.0), mk(3.4, 0.92, 80.0)]
    assert T.summarise(kept, total_frames=90, trace_end_s=3.4).cos_hold == pytest.approx(0.91)


def test_the_hold_window_still_defaults_to_the_last_detection():
    """Traces recorded before `trace_end_s` existed must summarise the way they always did."""
    rs = [mk(t, c, 80.0) for t, c in [(0.0, 0.1), (1.0, 0.5), (1.5, 0.8), (2.0, 0.9)]]
    assert T.summarise(rs, total_frames=4).cos_hold == pytest.approx((0.5 + 0.8 + 0.9) / 3)
