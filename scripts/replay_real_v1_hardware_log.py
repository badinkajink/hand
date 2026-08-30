#!/usr/bin/env python3
"""Replay a real_v1 hardware command log in MuJoCo and emit an analysis-friendly NPZ.

This is a commanded-trajectory replay, not a reconstruction of the real object motion: the
prototype has no object pose sensor. The before/after servo snapshots are compared with their
expected commands when the run summary contains the measured per-finger joint signs.

Example:
  uv run --extra rl python scripts/replay_real_v1_hardware_log.py \
    --log logs/hardware/<run>.jsonl \
    --plan docs/experiments/20260829-real_v1_deploy/deploy/g12_plan.json \
    --out logs/hardware/<run>_sim.npz
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "src/morphohand/driver/manta/host"
sys.path.insert(0, str(HOST))

import mujoco  # noqa: E402

from manta_hand.plan import (  # noqa: E402
    FINGER_ID,
    SIM_JOINT_TO_SERVO,
    HandPlan,
)

FINGER_ORDER = ("thumb", "index", "middle")
JOINT_ORDER = ("yaw", "mcp", "pip")


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or "kind" not in row:
                raise ValueError(f"{path}:{line_number}: expected a logged object with 'kind'")
            rows.append(row)
    return rows


def _summary_path(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.stem}_SUMMARY.json")


def _resolve_scene(plan: HandPlan, override: Path | None) -> Path:
    raw = override or (Path(plan.meta["scene"]) if plan.meta.get("scene") else None)
    if raw is None:
        raise ValueError("plan has no meta.scene; pass --scene")
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(ROOT / raw)
    else:
        # Exported plans historically stored absolute repo paths. Recover after a checkout
        # move if the path still contains an assets/... suffix.
        try:
            asset_i = raw.parts.index("assets")
            candidates.append(ROOT.joinpath(*raw.parts[asset_i:]))
        except ValueError:
            pass
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"MuJoCo scene not found; tried: {', '.join(map(str, candidates))}")


def _pose_vector(pose: dict[str, dict[str, float]]) -> np.ndarray:
    return np.asarray([
        math.radians(float(pose[finger][joint]))
        for finger in FINGER_ORDER
        for joint in JOINT_ORDER
    ])


def _endpoint_servo_errors(rows: list[dict], summary: dict, plan: HandPlan) -> dict:
    signs = summary.get("settings", {}).get("joint_signs")
    if not signs:
        return {"available": False, "reason": "run summary does not record joint_signs"}

    commands = [row for row in rows if row.get("kind") == "command"]
    if not commands:
        return {"available": False, "reason": "log contains no command frames"}
    expected = {
        "before": next(pose.joints for pose in plan.poses if pose.name == "grip"),
        "after": commands[-1]["sim_joint_deg"],
    }
    telemetry = {
        row.get("phase"): row.get("data", {}).get("servos")
        for row in rows
        if row.get("kind") == "telemetry" and row.get("phase") in expected
    }
    phases: dict[str, dict] = {}
    for phase, target in expected.items():
        measured = telemetry.get(phase)
        if not measured:
            phases[phase] = {"available": False, "reason": "no servo snapshot"}
            continue
        errors = {}
        for finger in FINGER_ORDER:
            fid = str(FINGER_ID[finger])
            servo_values = measured.get(fid, measured.get(FINGER_ID[finger], {}))
            for joint in JOINT_ORDER:
                servo_joint = SIM_JOINT_TO_SERVO[joint]
                relative = float(servo_values[servo_joint])
                sign = float(signs[finger][joint])
                errors[f"{finger}_{joint}"] = relative / sign - float(target[finger][joint])
        values = np.asarray(list(errors.values()))
        phases[phase] = {
            "available": True,
            "rms_error_deg": float(np.sqrt(np.mean(values**2))),
            "max_abs_error_deg": float(np.max(np.abs(values))),
            "error_deg": errors,
        }
    return {
        "available": any(value.get("available") for value in phases.values()),
        "phases": phases,
    }


def replay(log_path: Path, plan_path: Path, output: Path, *, scene_override: Path | None,
           sample_hz: float) -> dict:
    plan = HandPlan.from_json(plan_path)
    schema_errors = plan.schema_errors()
    if schema_errors:
        raise ValueError("invalid plan: " + "; ".join(schema_errors))
    rows = _load_rows(log_path)
    commands = [row for row in rows if row.get("kind") == "command"]
    if not commands:
        raise ValueError(f"{log_path} contains no command frames")
    if any("monotonic_s" not in row or "sim_joint_deg" not in row for row in commands):
        raise ValueError("every command row needs monotonic_s and sim_joint_deg")
    command_times = np.asarray([float(row["monotonic_s"]) for row in commands])
    if np.any(np.diff(command_times) < 0):
        raise ValueError("command monotonic_s values are not ordered")
    command_times -= command_times[0]

    summary_path = _summary_path(log_path)
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if summary.get("design") not in (None, plan.design):
        raise ValueError(
            f"log summary design {summary['design']!r} does not match plan {plan.design!r}"
        )
    speed_ratio = float(summary.get("settings", {}).get("speed_ratio", 1.0))
    if not speed_ratio > 0:
        raise ValueError("summary speed_ratio must be positive")

    scene = _resolve_scene(plan, scene_override)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    for key_name in ("open_ik", "open"):
        try:
            key_id = model.key(key_name).id
            break
        except KeyError:
            continue
    else:
        raise ValueError(f"{scene} has neither an open_ik nor open keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.time = 0.0

    initial_qpos = plan.meta.get("replay_initial_qpos")
    base_ctrl = plan.meta.get("replay_base_ctrl")
    if initial_qpos is None or base_ctrl is None:
        raise ValueError(
            "plan predates state-complete replay metadata; re-export it with "
            "scripts/real_v1_export_plan.py"
        )
    if len(initial_qpos) != model.nq or len(base_ctrl) != model.nu:
        raise ValueError(
            f"replay metadata shape mismatch: qpos {len(initial_qpos)}/{model.nq}, "
            f"ctrl {len(base_ctrl)}/{model.nu}"
        )
    data.qpos[:] = initial_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = base_ctrl

    actuator_ids = [
        model.actuator(f"a_{finger}_{joint}").id
        for finger in FINGER_ORDER
        for joint in JOINT_ORDER
    ]
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joints) != 1:
        raise ValueError(f"scene must have one free object joint, found {len(free_joints)}")
    object_body_id = int(model.jnt_bodyid[int(free_joints[0])])

    stride = max(1, int(round((1.0 / sample_hz) / model.opt.timestep)))
    record = {name: [] for name in (
        "sim_time_s", "object_pos", "object_quat", "object_axis_cos",
        "qpos", "qvel", "ctrl", "finger_ctrl_rad",
    )}
    step_count = 0

    def snapshot() -> None:
        record["sim_time_s"].append(float(data.time))
        record["object_pos"].append(data.xpos[object_body_id].copy())
        record["object_quat"].append(data.xquat[object_body_id].copy())
        axis = data.xmat[object_body_id].reshape(3, 3)[:, 2]
        record["object_axis_cos"].append(float(axis[2]))
        record["qpos"].append(data.qpos.copy())
        record["qvel"].append(data.qvel.copy())
        record["ctrl"].append(data.ctrl.copy())
        record["finger_ctrl_rad"].append(data.ctrl[actuator_ids].copy())

    def advance(duration_s: float) -> None:
        nonlocal step_count
        end = data.time + max(0.0, duration_s)
        while data.time + model.opt.timestep * 0.5 < end:
            mujoco.mj_step(model, data)
            step_count += 1
            if step_count % stride == 0:
                snapshot()

    def set_pose(pose: dict[str, dict[str, float]]) -> None:
        data.ctrl[actuator_ids] = _pose_vector(pose)

    open_pose = next(pose for pose in plan.poses if pose.name == "open")
    grip_pose = next(pose for pose in plan.poses if pose.name == "grip")
    # The saved fitted qpos is the same `open` state commanded on hardware. Override finger
    # controls only; base_ctrl preserves the fitted palm set-points that have no hardware axis.
    set_pose(open_pose.joints)
    mujoco.mj_forward(model, data)
    snapshot()

    # Recreate the close performed before logging. Plans describe a linear control ramp.
    close_duration = grip_pose.ramp_s / speed_ratio
    close_frames = max(1, int(round(close_duration * 50.0)))
    for frame in range(1, close_frames + 1):
        u = frame / close_frames
        blended = {
            finger: {
                joint: (
                    open_pose.joints[finger][joint]
                    + (grip_pose.joints[finger][joint] - open_pose.joints[finger][joint]) * u
                )
                for joint in JOINT_ORDER
            }
            for finger in FINGER_ORDER
        }
        set_pose(blended)
        advance(close_duration / close_frames)
    advance(grip_pose.hold_s / speed_ratio)

    # Preserve actual host timing, including missed deadlines, instead of regenerating the
    # nominal plan schedule.
    set_pose(commands[0]["sim_joint_deg"])
    for previous_time, next_time, command in zip(command_times, command_times[1:], commands[1:]):
        advance(float(next_time - previous_time))
        set_pose(command["sim_joint_deg"])

    final_phase = commands[-1].get("phase")
    final_hold = next(
        (pose.hold_s / speed_ratio for pose in plan.poses if pose.name == final_phase),
        0.0,
    )
    advance(final_hold)
    if not record["sim_time_s"] or record["sim_time_s"][-1] != data.time:
        snapshot()

    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(values) for name, values in record.items()}
    np.savez_compressed(output, **arrays)
    result = {
        "schema_version": 1,
        "source_log": str(log_path.resolve()),
        "source_summary": str(summary_path.resolve()) if summary_path.exists() else None,
        "plan": str(plan_path.resolve()),
        "scene": str(scene),
        "design": plan.design,
        "command_frames": len(commands),
        "samples": len(arrays["sim_time_s"]),
        "duration_s": float(data.time),
        "object": {
            "initial_pos_m": arrays["object_pos"][0].tolist(),
            "final_pos_m": arrays["object_pos"][-1].tolist(),
            "initial_axis_cos": float(arrays["object_axis_cos"][0]),
            "final_axis_cos": float(arrays["object_axis_cos"][-1]),
            "min_z_m": float(np.min(arrays["object_pos"][:, 2])),
        },
        "endpoint_servo_tracking": _endpoint_servo_errors(rows, summary, plan),
        "interpretation": (
            "Commanded-trajectory simulation only; without measured object pose this does not "
            "reconstruct the real object's motion."
        ),
    }
    result_path = output.with_name(f"{output.stem}_SUMMARY.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log", type=Path, required=True, help="hardware JSONL run")
    parser.add_argument("--plan", type=Path, required=True, help="matching exported plan JSON")
    parser.add_argument("--scene", type=Path, help="override plan meta.scene")
    parser.add_argument("--out", type=Path, required=True, help="output .npz")
    parser.add_argument("--sample-hz", type=float, default=50.0)
    args = parser.parse_args()
    if args.sample_hz <= 0:
        parser.error("--sample-hz must be positive")
    result = replay(
        args.log, args.plan, args.out, scene_override=args.scene, sample_hz=args.sample_hz
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
