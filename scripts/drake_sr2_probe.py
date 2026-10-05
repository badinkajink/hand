#!/usr/bin/env python3
"""Reproduce the SR² → Drake feasibility study without changing the project env.

    uv venv --python .venv/bin/python logs/20261001-drake-port/venv
    uv pip install --python logs/20261001-drake-port/venv/bin/python \
        drake==1.57.0 mujoco==3.6.0
    logs/20261001-drake-port/venv/bin/python scripts/drake_sr2_probe.py
    python3 scripts/drake_sr2_page.py

These are kinematic/solver integration checks, not grasp or hardware validation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, InverseKinematics,
    JacobianWrtVariable, Parser, Simulator, Solve,
)

from drake_sr2_hand import (
    ACTUATED, FINGER_NAMES, FINGERS, HAND, MORPH_NAMES, ROOT, build_hand,
)

OUT = ROOT / "docs/experiments/20261001-drake-port"
DESIGNS = {
    "nominal": np.zeros(6),
    "compact": np.array([.010, 0, -.010, -.010, -.010, .010]),
    "asymmetric": np.array([0, .012, .005, -.008, -.008, .006]),
}


class Warnings(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage().replace(str(ROOT) + "/", ""))


def import_audit():
    rows = []
    for source in [HAND, ACTUATED, HAND.parent / "scenes/scene_screwdriver_medium.xml"]:
        capture = Warnings()
        logger = logging.getLogger("drake")
        logger.addHandler(capture)
        try:
            builder = DiagramBuilder()
            plant, sg = AddMultibodyPlantSceneGraph(builder, time_step=.002)
            instance, = Parser(plant, sg).AddModels(str(source))
            plant.Finalize()
            actuators = []
            for i in plant.GetJointActuatorIndices(instance):
                a = plant.get_joint_actuator(i)
                gains = a.get_controller_gains()
                actuators.append(dict(name=a.name(), effort=a.effort_limit(),
                                      kp=gains.p, kd=gains.d))
            rows.append(dict(source=str(source.relative_to(ROOT)),
                             nq=plant.num_positions(), nv=plant.num_velocities(),
                             nu=plant.num_actuators(), actuators=actuators,
                             warnings=capture.messages))
        finally:
            logger.removeHandler(capture)
    return rows


def mujoco_configuration(model, morphology, fingers):
    data = mujoco.MjData(model)
    for name, value in zip(MORPH_NAMES + FINGER_NAMES, np.r_[morphology, fingers]):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    return data


def kinematic_checks(samples, seed):
    rng = np.random.default_rng(seed)
    hand = build_hand("locked")
    reference = hand.reference
    max_position = max_rotation = max_jacobian = max_mass = 0.0
    names = MORPH_NAMES + FINGER_NAMES
    drake_columns = [hand.plant.GetJointByName(n).velocity_start() for n in names]
    mj_columns = [int(reference.joint(n).dofadr[0]) for n in names]
    lo, hi = hand.limits(names)
    start = time.perf_counter()
    for sample in range(samples):
        values = lo if sample == 0 else hi if sample == 1 else rng.uniform(lo, hi)
        m, q = values[:6], values[6:]
        root = hand.new_context(m, q)
        pc = hand.plant_context(root)
        md = mujoco_configuration(reference, m, q)
        for f in FINGERS:
            body = hand.plant.GetBodyByName(f"{f}_tip")
            pose = hand.plant.EvalBodyPoseInWorld(pc, body)
            mb = reference.body(f"{f}_tip").id
            max_position = max(max_position, float(np.max(np.abs(
                pose.translation() - md.xpos[mb]))))
            max_rotation = max(max_rotation, float(np.max(np.abs(
                pose.rotation().matrix() - md.xmat[mb].reshape(3, 3)))))
            jac = hand.plant.CalcJacobianTranslationalVelocity(
                pc, JacobianWrtVariable.kV, body.body_frame(), np.zeros(3),
                hand.plant.world_frame(), hand.plant.world_frame())
            mj_jac = np.zeros((3, reference.nv))
            mujoco.mj_jacBody(reference, md, mj_jac, None, mb)
            max_jacobian = max(max_jacobian, float(np.max(np.abs(
                jac[:, drake_columns] - mj_jac[:, mj_columns]))))
        mj_mass = np.zeros((reference.nv, reference.nv))
        mujoco.mj_fullM(reference, mj_mass, md.qM)
        mass = hand.plant.CalcMassMatrix(pc)
        max_mass = max(max_mass, float(np.max(np.abs(
            mass[np.ix_(drake_columns, drake_columns)] - mj_mass[np.ix_(mj_columns, mj_columns)]))))
    assert max_position < 1e-10, max_position
    assert max_rotation < 1e-10, max_rotation
    assert max_jacobian < 1e-10, max_jacobian
    # Mass is an audit, not a contact/dynamics equivalence assertion.
    return dict(samples=samples, seed=seed, max_tip_position_error_m=max_position,
                max_tip_rotation_matrix_error=max_rotation, max_tip_jacobian_error=max_jacobian,
                max_mass_matrix_entry_error=max_mass,
                elapsed_s=time.perf_counter() - start)


def closest_distance(hand, root):
    query = hand.scene_graph.get_query_output_port().Eval(
        hand.scene_graph.GetMyContextFromRoot(root))
    return min(p.distance for p in query.ComputeSignedDistancePairwiseClosestPoints())


def solve_tips(hand, morphology, targets):
    root = hand.new_context(morphology)
    pc = hand.plant_context(root)
    ik = InverseKinematics(hand.plant, pc)
    for f, target in zip(FINGERS, targets):
        ik.AddPositionConstraint(hand.plant.GetBodyByName(f"{f}_tip").body_frame(),
                                 np.zeros(3), hand.plant.world_frame(),
                                 target - 1e-6, target + 1e-6)
    # Existing MJCF exclusions apply. No object, floor or rail geometry here.
    ik.AddMinimumDistanceLowerBoundConstraint(.001, .005)
    initial = hand.plant.GetPositions(pc).copy()
    scale = np.ones(hand.plant.num_positions()) / .4
    if hand.mode != "frozen":
        for n in MORPH_NAMES:
            scale[hand.plant.GetJointByName(n).position_start()] = 1 / .02
    ik.prog().AddQuadraticErrorCost(np.diag(scale**2), initial, ik.q())
    ik.prog().SetInitialGuess(ik.q(), initial)
    start = time.perf_counter()
    result = Solve(ik.prog())
    elapsed = time.perf_counter() - start
    assert result.is_success(), result.get_solution_result()
    hand.plant.SetPositions(pc, result.GetSolution(ik.q()))
    error = float(np.max(np.abs(hand.tips(root) - targets)))
    clearance = closest_distance(hand, root)
    assert error < 2e-6, error
    assert clearance >= .001 - 1e-6, clearance
    solution_m = (morphology if hand.mode == "frozen" else np.array([
        hand.plant.GetJointByName(n).get_translation(pc) for n in MORPH_NAMES]))
    if hand.mode == "locked":
        np.testing.assert_allclose(solution_m, morphology, atol=1e-12, rtol=0)
    solution_q = np.array([hand.plant.GetJointByName(n).get_angle(pc) for n in FINGER_NAMES])
    return dict(mode=hand.mode, nq=hand.plant.num_positions(), success=True,
                solver=result.get_solver_id().name(), solve_s=elapsed,
                max_tip_error_m=error, min_enabled_collision_distance_m=clearance,
                morphology_offsets_m=solution_m.tolist(), finger_angles_rad=solution_q.tolist(),
                target_tip_positions_m=targets.tolist())


def planning_checks():
    rows, freeze_errors = [], []
    locked = build_hand("locked")
    target_q = np.array([.15, .65, .4, -.12, .60, .45, .1, .5, .6])
    for design, m in DESIGNS.items():
        frozen = build_hand("frozen", m)
        a, b = frozen.new_context(fingers=target_q), locked.new_context(m, target_q)
        error = float(np.max(np.abs(frozen.tips(a) - locked.tips(b))))
        assert error < 1e-12, error
        freeze_errors.append(error)
        targets = frozen.tips(a)
        for hand in [frozen, locked]:
            rows.append(dict(design=design, **solve_tips(hand, m, targets)))
    # Allow six morphology coordinates to participate in an ordinary IK NLP.
    # This checks a joint morphology/posture solve, not grasp quality or global optimality.
    joint_design = build_hand("actuated")
    targets = locked.tips(locked.new_context(DESIGNS["compact"], target_q))
    rows.append(dict(design="joint morphology and posture", **solve_tips(
        joint_design, np.zeros(6), targets)))
    optimum = rows[-1]
    frozen = build_hand("frozen", optimum["morphology_offsets_m"])
    frozen_root = frozen.new_context(fingers=optimum["finger_angles_rad"])
    frozen_error = float(np.max(np.abs(frozen.tips(frozen_root) - targets)))
    assert frozen_error < 2e-6, frozen_error
    return dict(ik=rows, max_frozen_locked_tip_difference_m=max(freeze_errors),
                optimized_morphology_frozen_tip_error_m=frozen_error)


def lock_check(live_unlock=False):
    hand = build_hand("locked")
    m = DESIGNS["asymmetric"]
    root = hand.new_context(m)
    pc = hand.plant_context(root)
    hand.command(root, np.zeros(6), np.tile([.1, .6, .5], 3))
    simulator = Simulator(hand.diagram, root)
    simulator.AdvanceTo(.1)
    result = np.array([hand.plant.GetJointByName(n).get_translation(pc) for n in MORPH_NAMES])
    error = float(np.max(np.abs(result - m)))
    assert error < 1e-12, error
    if live_unlock:
        # Isolated subprocess: Drake 1.57 can terminate the entire process here.
        for n in MORPH_NAMES:
            hand.plant.GetJointByName(n).Unlock(pc)
        simulator.Initialize()
        simulator.AdvanceTo(.3)
    else:
        # A fresh context is a separate rollout, not a simulated mode transition.
        hand = build_hand("actuated")
        root = hand.new_context(m)
        pc = hand.plant_context(root)
        hand.command(root, np.zeros(6), np.tile([.1, .6, .5], 3))
        Simulator(hand.diagram, root).AdvanceTo(.2)
    moved = np.array([hand.plant.GetJointByName(n).get_translation(pc) for n in MORPH_NAMES])
    displacement = float(np.max(np.abs(moved - result)))
    assert displacement > 1e-7, displacement
    return dict(locked_duration_s=.1, locked_max_drift_m=error,
                unlocked_duration_s=.2, unlocked_max_displacement_m=displacement,
                separate_rollouts=not live_unlock,
                caveat="No object. Shipped gantry gains/damping are not a stepper identification.")


def actuator_check():
    hand = build_hand("locked")
    rows = []
    for idx in hand.plant.GetJointActuatorIndices(hand.model_instance):
        a = hand.plant.get_joint_actuator(idx)
        g = a.get_controller_gains()
        assert a.effort_limit() == 10 and g.p == 30 and g.d == .5
        rows.append(dict(name=a.name(), kp=g.p, kd=g.d, effort=a.effort_limit(),
                         rotor_inertia=a.default_rotor_inertia()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT / "probe.json")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--unlock-repro", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.unlock_repro:
        print(json.dumps(lock_check(live_unlock=True)))
        return
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    results = dict(
        date="2026-10-01", python=platform.python_version(), platform=platform.platform(),
        versions={n: importlib.metadata.version(n) for n in ("drake", "mujoco", "numpy")},
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (HAND, ACTUATED)},
        import_audit=import_audit(), kinematics=kinematic_checks(args.samples, seed=42),
        planning=planning_checks(), locking=lock_check(), actuators=actuator_check())
    child = subprocess.run([sys.executable, __file__, "--unlock-repro"],
                           capture_output=True, text=True, timeout=30)
    results["live_unlock"] = dict(returncode=child.returncode,
                                  stdout=child.stdout, stderr=child.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: results[k] for k in ("versions", "kinematics", "planning", "locking")},
                     indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
