"""The sim's hand design space and the driver's rails have to describe the same machine.

They were written months apart from the same drawing and never compared. These tests are the
comparison, so a change to either side that breaks it fails here rather than on the bench.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import pytest

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "src/morphohand/driver/manta/host"


@pytest.fixture(scope="module")
def mh():
    sys.path.insert(0, str(HOST))
    import manta_hand.plan as plan

    return plan


def test_driver_tables_import_without_hardware_deps(mh):
    """The calibration tables are the contract an offline planner validates against; needing
    pyserial/rustypot installed to read them would push that check onto the bench."""
    assert mh.FULL_EXTENSION_MM[0] > 0
    assert mh.FINGER_JOINTS[0]["aa"][0] == 0


def test_mount_origins_are_identical(mh):
    """The claim that the sim's palm frame IS the firmware's {P}: same origins, no rotation."""
    from morphohand.sampling.morphology import REAL_V1_MOUNTS

    for finger, (mx, my) in REAL_V1_MOUNTS.items():
        ox, oy = mh.FINGER_GEOMETRY[mh.FINGER_ID[finger]]["origin"]
        assert (round(mx * 1000, 6), round(my * 1000, 6)) == (ox, oy), finger


def test_palm_local_round_trip(mh):
    for finger in ("thumb", "index", "middle"):
        for pt in ((0.0, 0.0), (-42.5, 0.0), (42.5, 40.0), (12.3, -7.9)):
            back = mh.palm_from_local(finger, *mh.local_from_palm(finger, *pt))
            assert back == pytest.approx(pt, abs=1e-9)


def test_local_envelope_matches_the_gantry_it_guards(mh):
    """plan.local_envelope and kinematics._local_bounds invert the same transform; if they ever
    disagree, a plan validates offline and then raises mid-move."""
    from manta_hand.kinematics import _local_bounds

    for finger, fid in mh.FINGER_ID.items():
        got = [v for pair in mh.local_envelope(finger) for v in pair]
        want = [v for pair in _local_bounds(fid) for v in pair]
        assert got == pytest.approx(want)


def test_envelope_corners_map_inside_the_stepper_range(mh):
    """Every point plan.py calls reachable must produce a MOVEMM inside [0, travel] -- MOVEMM
    does not stall-check, so an off-by-one here grinds an axis into a hardstop."""
    from manta_hand.kinematics import STEPPER_JOINTS, axis_stepper_range

    for finger, fid in mh.FINGER_ID.items():
        (xlo, xhi), (ylo, yhi) = mh.local_envelope(finger)
        for lx in (xlo, xhi):
            for ly in (ylo, yhi):
                for j, mm in mh.stepper_mm(finger, lx, ly).items():
                    lo, hi = axis_stepper_range(j)
                    assert lo - 1e-6 <= mm <= hi + 1e-6, (finger, j, mm)
        assert set(mh.stepper_mm(finger, 0.0, 0.0)) == set(STEPPER_JOINTS[fid])


def test_sim_joint_limits_are_the_servo_contract(mh):
    """What identifies yaw/mcp/pip with aa/fe1/fe2: three distinct declared ranges, matched
    exactly. If a scene edit changes a limit, the identification stops being forced."""
    m = mujoco.MjModel.from_xml_path(str(ROOT / "assets/mjcf/real_v1/real_hand.xml"))
    declared = {"aa": (-85.0, 85.0), "fe1": (-15.0, 92.0), "fe2": (-18.0, 92.0)}
    for finger in mh.FINGER_ID:
        for sim_joint, servo in mh.SIM_JOINT_TO_SERVO.items():
            rng = m.jnt_range[m.joint(f"{finger}_{sim_joint}").id]
            got = tuple(math.degrees(v) for v in rng)
            assert got == pytest.approx(declared[servo], abs=0.01), f"{finger}_{sim_joint}"


def test_flexion_is_positive_toward_the_palm_centre(mh):
    """The half of the sign question that IS settled on paper: the scene mirrors mcp/pip per
    finger so positive is flexion inward on all three, matching the drawing's +flex column and
    the servos' asymmetric [-15,+92] ranges. The yaw axis is deliberately NOT mirrored, which
    is why aa's sign needs hardware."""
    m = mujoco.MjModel.from_xml_path(str(ROOT / "assets/mjcf/real_v1/real_hand.xml"))
    inward = {"thumb": +1.0, "index": -1.0, "middle": -1.0}   # palm-x direction of flexion
    for finger, want in inward.items():
        for sim_joint in ("mcp", "pip"):
            axis = m.jnt_axis[m.joint(f"{finger}_{sim_joint}").id]
            # the link hangs along -z, so a positive rotation about `axis` displaces the tip
            # by theta * (axis x p) = (-axis_y * L, 0, 0): the flexion direction in palm x is
            # -sign(axis_y). The scene mirrors axis_y per finger precisely so that both come
            # out pointing at the palm centre.
            assert -math.copysign(1.0, axis[1]) == want, f"{finger}_{sim_joint}"
        yaw = m.jnt_axis[m.joint(f"{finger}_yaw").id]
        assert list(yaw) == [1.0, 0.0, 0.0], finger


@pytest.mark.parametrize("design", ["g12", "g23", "g24", "rv04_mid"])
def test_shortlisted_deploy_plans_fit_the_real_hand(mh, design):
    path = ROOT / f"docs/experiments/20260829-real_v1_deploy/deploy/{design}_plan.json"
    if not path.exists():
        pytest.skip(f"{path.name} not exported")
    plan = mh.HandPlan.from_json(path)
    bad = plan.validate()
    assert bad == [], "\n".join(str(v) for v in bad)
    cmds = plan.stepper_commands()
    assert sum(c.startswith("MOVEMM") for c in cmds) == 6


def test_compact_corner_designs_are_reported_out_of_reach(mh):
    """The live finding: the +-30mm design box's compact corner is 4-6mm past the rails. This
    is not a bug to route around -- it is the thing that has to stay visible."""
    v = (mh.mount_violations("thumb", 30.0, 0.0, frame="local")
         + mh.mount_violations("index", -30.0, 0.0, frame="local")
         + mh.mount_violations("middle", -30.0, 0.0, frame="local"))
    assert [round(x.short, 1) for x in v] == [3.8, 4.0, 5.9]
    # ...and that a re-calibrated rail would recover it, which is what the ruler check decides.
    assert not mh.mount_violations("middle", -29.5, 0.0, frame="local", travel_mm={5: 59.8})
