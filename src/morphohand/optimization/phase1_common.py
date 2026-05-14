from __future__ import annotations
# pyright: reportMissingImports=false

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio
import mujoco
import numpy as np

from .contact_targets import ContactTargetSet, score_contact_targets
from .force_closure import (
    extract_finger_contacts,
    force_closure_metrics,
)
from .phase1_trajectory import build_trajectory_interpolator


@dataclass
class Phase1EvalConfig:
    settle_steps: int = 240
    lift_steps: int = 220
    hold_steps: int = 140
    lift_delta_z: float = 0.05
    lift_ramp_steps: int = 80
    pivot_steps: int = 0
    pivot_ramp_steps: int = 80
    pivot_delta_rx: float = 0.0
    pivot_delta_ry: float = 0.0
    pivot_delta_rz: float = 0.0
    objective_weight_distance: float = 2.0
    objective_weight_contact: float = 0.4
    objective_weight_lift: float = 35.0
    objective_weight_velocity_penalty: float = 0.15
    objective_weight_xy_drift_penalty: float = 6.0
    objective_weight_drop_penalty: float = 12.0
    objective_weight_contact_persistence: float = 0.8
    objective_weight_min_finger_persistence: float = 2.0
    objective_weight_finger_persistence_imbalance_penalty: float = 1.0
    objective_weight_finger_yaw_drift_penalty: float = 0.8
    objective_weight_finger_flex_drift_penalty: float = 0.4
    objective_weight_cube_yaw_drift_penalty: float = 4.0
    objective_weight_cube_axis_tilt_penalty: float = 6.0
    objective_weight_cube_ang_drift_penalty: float = 2.0
    objective_weight_contact_target_reward: float = 0.0
    objective_weight_contact_target_distance_penalty: float = 0.0
    objective_weight_force_closure: float = 0.0
    force_closure_friction_mu: float = 0.5
    force_closure_cone_edges: int = 4
    force_closure_weight_balance: float = 0.5
    force_closure_weight_q1: float = 1.0


class Phase1GraspEvaluator:
    """Scores finger control vectors for a fixed morphology scene."""

    def __init__(
        self,
        scene_xml: Path,
        keyframe: str = "open",
        cfg: Phase1EvalConfig | None = None,
        backend: str = "mujoco",
        comfree_stiffness: float = 0.2,
        comfree_damping: float = 0.001,
        backend_nworld: int = 1,
        backend_nconmax: int = 200,
        backend_njmax: int = 2000,
        backend_sync_interval: int = 1,
        metric_sample_interval: int = 1,
        speed_mode: str = "accurate",
        metric_collection_mode: str = "sampled",
        contact_target_set: ContactTargetSet | None = None,
    ) -> None:
        self.scene_xml = Path(scene_xml)
        self.cfg = cfg or Phase1EvalConfig()
        self.contact_target_set = contact_target_set
        self.backend = backend
        self.comfree_stiffness = float(comfree_stiffness)
        self.comfree_damping = float(comfree_damping)
        self.backend_nworld = int(backend_nworld)
        self.backend_nconmax = int(backend_nconmax)
        self.backend_njmax = int(backend_njmax)
        self.speed_mode = speed_mode
        self.metric_collection_mode = metric_collection_mode
        self.backend_sync_interval = max(1, int(backend_sync_interval))
        self.metric_sample_interval = max(1, int(metric_sample_interval))

        if self.speed_mode == "accurate":
            self.backend_sync_interval = 1
            self.metric_sample_interval = 1
            self.metric_collection_mode = "sampled"
        elif self.speed_mode == "balanced":
            self.backend_sync_interval = max(4, self.backend_sync_interval)
            self.metric_sample_interval = max(4, self.metric_sample_interval)
            self.metric_collection_mode = "sampled"
        elif self.speed_mode == "aggressive":
            self.backend_sync_interval = max(16, self.backend_sync_interval)
            self.metric_sample_interval = max(16, self.metric_sample_interval)
            self.metric_collection_mode = "terminal"

        if self.metric_collection_mode not in {"sampled", "terminal"}:
            raise ValueError("metric_collection_mode must be 'sampled' or 'terminal'")
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)

        self._wp = None
        self._backend_mod = None
        self._mjwarp_mod = None
        self._backend_model: Any = None
        self._backend_data: Any = None
        self._backend_step_fn: Any = None
        self._backend_steps_since_sync = 0

        self.keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if self.keyframe_id < 0:
            raise ValueError(f"Keyframe '{keyframe}' not found in {self.scene_xml}")

        self.pose_actuator_names = [
            "a_palm_px",
            "a_palm_py",
            "a_palm_pz",
            "a_palm_rx",
            "a_palm_ry",
            "a_palm_rz",
        ]
        self.finger_actuator_names = [
            "a_thumb_yaw",
            "a_thumb_mcp",
            "a_thumb_pip",
            "a_index_yaw",
            "a_index_mcp",
            "a_index_pip",
            "a_middle_yaw",
            "a_middle_mcp",
            "a_middle_pip",
        ]
        self.finger_joint_names = [
            "thumb_yaw",
            "thumb_mcp",
            "thumb_pip",
            "index_yaw",
            "index_mcp",
            "index_pip",
            "middle_yaw",
            "middle_mcp",
            "middle_pip",
        ]

        self.pose_actuator_ids = self._actuator_ids(self.pose_actuator_names)
        self.finger_actuator_ids = self._actuator_ids(self.finger_actuator_names)
        self.finger_joint_qpos_ids = self._joint_qpos_ids(self.finger_joint_names)
        self.finger_yaw_qpos_ids = np.asarray(
            [self.finger_joint_qpos_ids[0], self.finger_joint_qpos_ids[3], self.finger_joint_qpos_ids[6]],
            dtype=np.int32,
        )
        self.finger_flex_qpos_ids = np.asarray(
            [
                self.finger_joint_qpos_ids[1],
                self.finger_joint_qpos_ids[2],
                self.finger_joint_qpos_ids[4],
                self.finger_joint_qpos_ids[5],
                self.finger_joint_qpos_ids[7],
                self.finger_joint_qpos_ids[8],
            ],
            dtype=np.int32,
        )

        self.object_body_name, self.cube_body_id = self._resolve_object_body()
        self.tip_body_ids = [
            self._require_body("thumb_tip"),
            self._require_body("index_tip"),
            self._require_body("middle_tip"),
        ]

        self._reset_to_keyframe()
        self.initial_ctrl = self.data.ctrl.copy()
        self.initial_palm_pz_target = float(self.initial_ctrl[self.pose_actuator_ids[2]])

        self.finger_ctrl_min, self.finger_ctrl_max = self._finger_ctrl_bounds()
        self.cube_half_extents = self._infer_cube_half_extents()
        self._setup_backend()

    def _setup_backend(self) -> None:
        if self.backend == "mujoco":
            return
        if self.backend not in {"mjwarp", "comfree-warp"}:
            raise ValueError(
                f"Unsupported backend '{self.backend}'. Expected one of: mujoco, mjwarp, comfree-warp"
            )

        try:
            import warp as wp  # pyright: ignore[reportMissingImports]
            import comfree_warp as cfwarp  # pyright: ignore[reportMissingImports]
            from comfree_warp import mujoco_warp as mjwarp  # pyright: ignore[reportMissingImports]
        except Exception as exc:
            raise RuntimeError(
                "Warp backend requested but imports failed. Install mujoco_warp/comfree_warp and warp."
            ) from exc

        wp.init()
        self._wp = wp
        self._mjwarp_mod = mjwarp
        if self.backend == "mjwarp":
            self._backend_mod = mjwarp
            self._backend_model = mjwarp.put_model(self.model)
            self._backend_step_fn = mjwarp.step
        else:
            self._backend_mod = cfwarp
            self._backend_model = cfwarp.put_model(
                self.model,
                comfree_stiffness=self.comfree_stiffness,
                comfree_damping=self.comfree_damping,
            )
            self._backend_step_fn = cfwarp.step

        self._backend_data = self._backend_mod.put_data(
            self.model,
            self.data,
            nworld=self.backend_nworld,
            nconmax=self.backend_nconmax,
            njmax=self.backend_njmax,
        )
        self._backend_steps_since_sync = 0
        # Warm up kernels once to avoid first-step overhead in timing.
        self._backend_step_fn(self._backend_model, self._backend_data)
        self._backend_step_fn(self._backend_model, self._backend_data)

    def _sync_backend_from_mujoco(self) -> None:
        if self.backend == "mujoco":
            return
        assert self._wp is not None and self._backend_data is not None
        wp = self._wp
        wp.copy(self._backend_data.ctrl, wp.array([self.data.ctrl.astype(np.float32)]))
        wp.copy(
            self._backend_data.xfrc_applied,
            wp.array([self.data.xfrc_applied.astype(np.float32)]),
        )

    def _sync_backend_full_state_from_mujoco(self) -> None:
        if self.backend == "mujoco":
            return
        assert self._wp is not None and self._backend_data is not None
        wp = self._wp
        self._sync_backend_from_mujoco()
        wp.copy(self._backend_data.qpos, wp.array([self.data.qpos.astype(np.float32)]))
        wp.copy(self._backend_data.qvel, wp.array([self.data.qvel.astype(np.float32)]))
        wp.copy(self._backend_data.act, wp.array([self.data.act.astype(np.float32)]))
        wp.copy(self._backend_data.time, wp.array([self.data.time], dtype=wp.float32))

    def _sync_mujoco_from_backend(self) -> None:
        if self.backend == "mujoco":
            return
        assert self._wp is not None
        assert self._mjwarp_mod is not None
        assert self._backend_data is not None
        self._wp.synchronize()
        self._mjwarp_mod.get_data_into(self.data, self.model, self._backend_data, world_id=0)
        self._backend_steps_since_sync = 0

    def _step_dynamics(self, force_sync: bool = False) -> None:
        if self.backend == "mujoco":
            mujoco.mj_step(self.model, self.data)
            return

        assert self._backend_step_fn is not None
        assert self._backend_model is not None
        assert self._backend_data is not None
        assert self._wp is not None
        assert self._mjwarp_mod is not None
        self._sync_backend_from_mujoco()
        self._backend_step_fn(self._backend_model, self._backend_data)
        self._backend_steps_since_sync += 1
        needs_sync = force_sync or self._backend_steps_since_sync >= self.backend_sync_interval
        if needs_sync:
            self._sync_mujoco_from_backend()

    def _actuator_ids(self, names: list[str]) -> np.ndarray:
        ids = []
        for name in names:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if idx < 0:
                raise ValueError(f"Actuator '{name}' not found in scene")
            ids.append(idx)
        return np.array(ids, dtype=np.int32)

    def _require_body(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if idx < 0:
            raise ValueError(f"Body '{name}' not found in scene")
        return int(idx)

    def _joint_qpos_ids(self, names: list[str]) -> np.ndarray:
        ids = []
        for name in names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id < 0:
                raise ValueError(f"Joint '{name}' not found in scene")
            ids.append(int(self.model.jnt_qposadr[jnt_id]))
        return np.asarray(ids, dtype=np.int32)

    def _resolve_object_body(self) -> tuple[str, int]:
        preferred = ["cube", "power_drill", "prism", "screwdriver", "object"]
        for name in preferred:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx >= 0:
                return name, int(idx)

        for body_id in range(1, int(self.model.nbody)):
            jadr = int(self.model.body_jntadr[body_id])
            jnum = int(self.model.body_jntnum[body_id])
            has_freejoint = jnum > 0 and int(self.model.jnt_type[jadr]) == int(mujoco.mjtJoint.mjJNT_FREE)
            if not has_freejoint:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            if name == "world":
                continue
            return name, int(body_id)

        raise ValueError("Unable to infer manipulated object body (expected named body like cube/power_drill or a freejoint body)")

    def _infer_cube_half_extents(self) -> np.ndarray:
        # Approximate object extents from its geoms in body-local coordinates.
        # This keeps distance/contact proxy metrics usable for cylinders/capsules/meshes too.
        half = np.zeros(3, dtype=np.float64)
        found = False
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) != self.cube_body_id:
                continue
            found = True

            gtype = int(self.model.geom_type[geom_id])
            gsize = np.asarray(self.model.geom_size[geom_id, :3], dtype=np.float64)
            if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                ext = gsize
            elif gtype in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
                ext = np.array([gsize[0], gsize[0], gsize[1]], dtype=np.float64)
            elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                ext = np.array([gsize[0], gsize[0], gsize[0]], dtype=np.float64)
            elif gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                ext = gsize
            else:
                # Fallback: bounding sphere radius for mesh/other geoms.
                r = float(self.model.geom_rbound[geom_id])
                ext = np.array([r, r, r], dtype=np.float64)

            gpos = np.asarray(self.model.geom_pos[geom_id, :3], dtype=np.float64)
            half = np.maximum(half, np.abs(gpos) + ext)

        if not found:
            raise ValueError("Object body has no geoms; cannot infer extents")

        return half

    def _finger_ctrl_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        mins = []
        maxs = []
        for actuator_id in self.finger_actuator_ids:
            mins.append(self.model.actuator_ctrlrange[actuator_id, 0])
            maxs.append(self.model.actuator_ctrlrange[actuator_id, 1])
        return np.asarray(mins), np.asarray(maxs)

    def _reset_to_keyframe(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe_id)
        mujoco.mj_forward(self.model, self.data)
        if self.backend != "mujoco" and self._backend_data is not None:
            self._sync_backend_full_state_from_mujoco()
            self._backend_steps_since_sync = 0

    def _build_full_ctrl(self, finger_ctrl: np.ndarray, lift: bool = False) -> np.ndarray:
        ctrl = self.initial_ctrl.copy()
        ctrl[self.finger_actuator_ids] = finger_ctrl
        if lift:
            ctrl[self.pose_actuator_ids[2]] = self.initial_palm_pz_target + self.cfg.lift_delta_z
        return ctrl

    def _build_pose_ctrl(
        self,
        finger_ctrl: np.ndarray,
        lift_scale: float,
        pivot_scale: float,
    ) -> np.ndarray:
        ctrl = self.initial_ctrl.copy()
        ctrl[self.finger_actuator_ids] = finger_ctrl
        ctrl[self.pose_actuator_ids[2]] = self.initial_palm_pz_target + lift_scale * self.cfg.lift_delta_z
        ctrl[self.pose_actuator_ids[3]] = self.initial_ctrl[self.pose_actuator_ids[3]] + pivot_scale * self.cfg.pivot_delta_rx
        ctrl[self.pose_actuator_ids[4]] = self.initial_ctrl[self.pose_actuator_ids[4]] + pivot_scale * self.cfg.pivot_delta_ry
        ctrl[self.pose_actuator_ids[5]] = self.initial_ctrl[self.pose_actuator_ids[5]] + pivot_scale * self.cfg.pivot_delta_rz
        return ctrl

    def _step_with_ctrl(self, ctrl: np.ndarray, steps: int) -> None:
        for _ in range(steps):
            self.data.ctrl[:] = ctrl
            self._step_dynamics(force_sync=False)
        self._sync_mujoco_from_backend()

    def _cube_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.xpos[self.cube_body_id].copy()
        rot = self.data.xmat[self.cube_body_id].reshape(3, 3).copy()
        return pos, rot

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _cube_yaw(self, rot: np.ndarray) -> float:
        # Body x-axis heading around world Z. Works for all current scenes and catches spin/twist.
        return float(np.arctan2(rot[1, 0], rot[0, 0]))

    def _cube_axis_tilt(self, rot_before: np.ndarray, rot_after: np.ndarray) -> float:
        # Cylinder's principal axis is local +Z.
        axis_before = np.asarray(rot_before[:, 2], dtype=np.float64)
        axis_after = np.asarray(rot_after[:, 2], dtype=np.float64)
        dot = float(np.clip(np.dot(axis_before, axis_after), -1.0, 1.0))
        return float(np.arccos(dot))

    def _cube_total_angle_drift(self, rot_before: np.ndarray, rot_after: np.ndarray) -> float:
        rel = np.asarray(rot_after @ rot_before.T, dtype=np.float64)
        trace = float(np.trace(rel))
        cos_ang = float(np.clip(0.5 * (trace - 1.0), -1.0, 1.0))
        return float(np.arccos(cos_ang))

    def _tip_positions(self) -> np.ndarray:
        return np.vstack([self.data.xpos[bid] for bid in self.tip_body_ids])

    def _tip_distances_to_cube(self) -> np.ndarray:
        cube_pos, cube_rot = self._cube_pose()
        tip_world = self._tip_positions()
        local = (tip_world - cube_pos) @ cube_rot
        clipped = np.clip(local, -self.cube_half_extents, self.cube_half_extents)
        closest_world = clipped @ cube_rot.T + cube_pos
        return np.linalg.norm(tip_world - closest_world, axis=1)

    def _cube_tip_contact_count(self) -> int:
        count = 0
        tip_bodies = set(self.tip_body_ids)
        ngeom = int(self.model.ngeom)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            if g1 < 0 or g2 < 0 or g1 >= ngeom or g2 >= ngeom:
                continue
            b1 = int(self.model.geom_bodyid[contact.geom1])
            b2 = int(self.model.geom_bodyid[contact.geom2])
            if b1 == self.cube_body_id and b2 in tip_bodies:
                count += 1
            elif b2 == self.cube_body_id and b1 in tip_bodies:
                count += 1
        return count

    def _finger_contact_flags(self) -> np.ndarray:
        flags = np.zeros(3, dtype=np.float64)
        ngeom = int(self.model.ngeom)
        tip_lookup = {
            self.tip_body_ids[0]: 0,
            self.tip_body_ids[1]: 1,
            self.tip_body_ids[2]: 2,
        }
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            if g1 < 0 or g2 < 0 or g1 >= ngeom or g2 >= ngeom:
                continue
            b1 = int(self.model.geom_bodyid[contact.geom1])
            b2 = int(self.model.geom_bodyid[contact.geom2])
            if b1 == self.cube_body_id and b2 in tip_lookup:
                flags[tip_lookup[b2]] = 1.0
            elif b2 == self.cube_body_id and b1 in tip_lookup:
                flags[tip_lookup[b1]] = 1.0
        return flags

    def _pose_scales_for_dynamic_step(self, dynamic_t: int) -> tuple[float, float]:
        """Map a dynamic-phase step index to (lift_scale, pivot_scale) for `_build_pose_ctrl`."""
        if dynamic_t < self.cfg.lift_steps:
            ramp_denom = max(1, int(self.cfg.lift_ramp_steps))
            return min(1.0, float((dynamic_t + 1) / ramp_denom)), 0.0
        if dynamic_t < self.cfg.lift_steps + self.cfg.pivot_steps:
            pivot_local = dynamic_t - self.cfg.lift_steps
            ramp_denom = max(1, int(self.cfg.pivot_ramp_steps))
            return 1.0, min(1.0, float((pivot_local + 1) / ramp_denom))
        return 1.0, (1.0 if self.cfg.pivot_steps > 0 else 0.0)

    def _ctrl_for_dynamic_step(
        self,
        dynamic_t: int,
        get_finger_ctrl: Callable[[int], np.ndarray],
    ) -> np.ndarray:
        lift_scale, pivot_scale = self._pose_scales_for_dynamic_step(dynamic_t)
        return self._build_pose_ctrl(
            get_finger_ctrl(dynamic_t),
            lift_scale=lift_scale,
            pivot_scale=pivot_scale,
        )

    def _compute_score_and_metrics(
        self,
        *,
        distances: np.ndarray,
        contact_count: int,
        z_before: float,
        peak_z: float,
        z_after_hold: float,
        cube_pos_before: np.ndarray,
        cube_pos_after: np.ndarray,
        cube_rot_before: np.ndarray,
        cube_rot_after: np.ndarray,
        cube_yaw_before: float,
        cube_yaw_after: float,
        cube_vel_norm: float,
        contact_active_steps: int,
        finger_contact_steps: np.ndarray,
        all_finger_contact_steps: float,
        total_dynamic_steps: int,
        finger_qpos_settle: np.ndarray,
        finger_qpos_after: np.ndarray,
        tip_positions_settle: np.ndarray | None = None,
        fc_metrics: Any = None,
    ) -> tuple[float, dict[str, float]]:
        lift_amount = peak_z - z_before
        mean_dist = float(np.mean(distances))
        cube_xy_drift = float(np.linalg.norm(cube_pos_after[:2] - cube_pos_before[:2]))
        cube_yaw_drift = float(np.abs(self._wrap_to_pi(cube_yaw_after - cube_yaw_before)))
        cube_axis_tilt = self._cube_axis_tilt(cube_rot_before, cube_rot_after)
        cube_ang_drift = self._cube_total_angle_drift(cube_rot_before, cube_rot_after)
        cube_z_drop_from_peak = float(max(0.0, peak_z - z_after_hold))
        contact_persistence = float(contact_active_steps / max(1, total_dynamic_steps))
        finger_contact_persistence = finger_contact_steps / max(1, total_dynamic_steps)
        min_finger_contact_persistence = float(np.min(finger_contact_persistence))
        finger_persistence_imbalance = float(np.max(finger_contact_persistence) - np.min(finger_contact_persistence))
        all_finger_contact_persistence = float(all_finger_contact_steps / max(1, total_dynamic_steps))

        finger_qpos_delta = np.abs(finger_qpos_after - finger_qpos_settle)
        finger_yaw_drift = float(np.mean(finger_qpos_delta[[0, 3, 6]]))
        finger_flex_drift = float(np.mean(finger_qpos_delta[[1, 2, 4, 5, 7, 8]]))

        contact_target_reward = 0.0
        contact_target_mean_distance = 0.0
        contact_target_assignment: list[int | None] = []
        if (
            self.contact_target_set is not None
            and self.contact_target_set.patches
            and tip_positions_settle is not None
        ):
            breakdown = score_contact_targets(
                self.contact_target_set,
                tip_positions_settle,
                cube_pos_before,
                cube_rot_before,
            )
            contact_target_reward = breakdown.total_reward
            contact_target_mean_distance = breakdown.mean_distance
            contact_target_assignment = breakdown.assignment

        fc_contribution = 0.0
        if fc_metrics is not None and np.isfinite(fc_metrics.score):
            fc_contribution = self.cfg.objective_weight_force_closure * fc_metrics.score

        score = (
            self.cfg.objective_weight_lift * lift_amount
            - self.cfg.objective_weight_distance * mean_dist
            + self.cfg.objective_weight_contact * float(contact_count)
            - self.cfg.objective_weight_velocity_penalty * cube_vel_norm
            - self.cfg.objective_weight_xy_drift_penalty * cube_xy_drift
            - self.cfg.objective_weight_cube_yaw_drift_penalty * cube_yaw_drift
            - self.cfg.objective_weight_cube_axis_tilt_penalty * cube_axis_tilt
            - self.cfg.objective_weight_cube_ang_drift_penalty * cube_ang_drift
            - self.cfg.objective_weight_drop_penalty * cube_z_drop_from_peak
            + self.cfg.objective_weight_contact_persistence * contact_persistence
            + self.cfg.objective_weight_min_finger_persistence * min_finger_contact_persistence
            - self.cfg.objective_weight_finger_persistence_imbalance_penalty * finger_persistence_imbalance
            - self.cfg.objective_weight_finger_yaw_drift_penalty * finger_yaw_drift
            - self.cfg.objective_weight_finger_flex_drift_penalty * finger_flex_drift
            + self.cfg.objective_weight_contact_target_reward * contact_target_reward
            - self.cfg.objective_weight_contact_target_distance_penalty * contact_target_mean_distance
            + fc_contribution
        )

        metrics = {
            "score": float(score),
            "mean_tip_distance": mean_dist,
            "tip_distance_thumb": float(distances[0]),
            "tip_distance_index": float(distances[1]),
            "tip_distance_middle": float(distances[2]),
            "cube_tip_contacts": float(contact_count),
            "cube_z_before_lift": z_before,
            "cube_z_peak": peak_z,
            "cube_z_after_hold": z_after_hold,
            "cube_lift": lift_amount,
            "cube_vel_norm": cube_vel_norm,
            "cube_xy_drift": cube_xy_drift,
            "cube_yaw_drift": cube_yaw_drift,
            "cube_axis_tilt": cube_axis_tilt,
            "cube_ang_drift": cube_ang_drift,
            "cube_z_drop_from_peak": cube_z_drop_from_peak,
            "contact_persistence": contact_persistence,
            "thumb_contact_persistence": float(finger_contact_persistence[0]),
            "index_contact_persistence": float(finger_contact_persistence[1]),
            "middle_contact_persistence": float(finger_contact_persistence[2]),
            "all_finger_contact_persistence": all_finger_contact_persistence,
            "min_finger_contact_persistence": min_finger_contact_persistence,
            "finger_persistence_imbalance": finger_persistence_imbalance,
            "finger_yaw_drift": finger_yaw_drift,
            "finger_flex_drift": finger_flex_drift,
            "contact_target_reward": float(contact_target_reward),
            "contact_target_mean_distance": float(contact_target_mean_distance),
        }
        if contact_target_assignment:
            for i, tip in enumerate(contact_target_assignment):
                metrics[f"contact_target_assignment_{i}"] = (
                    float(tip) if tip is not None else float("nan")
                )
        if fc_metrics is not None:
            metrics.update(fc_metrics.to_dict())
        return float(score), metrics

    def _run_dynamic_loop_sampled(
        self,
        get_finger_ctrl: Callable[[int], np.ndarray],
    ) -> tuple[float, int, np.ndarray, float]:
        """Run lift→pivot→hold with sampled metric collection.

        Returns (peak_z, contact_active_steps, finger_contact_steps, all_finger_contact_steps).
        """
        sample_stride = 1 if self.backend == "mujoco" else self.metric_sample_interval

        peak_z = float(self.data.xpos[self.cube_body_id, 2])
        contact_active_steps = 0
        finger_contact_steps = np.zeros(3, dtype=np.float64)
        all_finger_contact_steps = 0.0

        phases = (
            (self.cfg.lift_steps, 0),
            (self.cfg.pivot_steps, self.cfg.lift_steps),
            (self.cfg.hold_steps, self.cfg.lift_steps + self.cfg.pivot_steps),
        )
        for phase_steps, phase_offset in phases:
            local_t = 0
            while local_t < phase_steps:
                chunk = max(1, min(sample_stride, int(phase_steps - local_t)))
                for k in range(chunk):
                    dynamic_t = phase_offset + local_t + k
                    self.data.ctrl[:] = self._ctrl_for_dynamic_step(dynamic_t, get_finger_ctrl)
                    self._step_dynamics(force_sync=False)
                self._sync_mujoco_from_backend()
                peak_z = max(peak_z, float(self.data.xpos[self.cube_body_id, 2]))
                finger_flags = self._finger_contact_flags()
                finger_contact_steps += finger_flags * float(chunk)
                if float(np.min(finger_flags)) > 0.5:
                    all_finger_contact_steps += float(chunk)
                if float(np.sum(finger_flags)) >= 2.0:
                    contact_active_steps += int(chunk)
                local_t += chunk

        return peak_z, contact_active_steps, finger_contact_steps, all_finger_contact_steps

    def _run_dynamic_loop_terminal(
        self,
        get_finger_ctrl: Callable[[int], np.ndarray],
        z_before: float,
    ) -> tuple[float, int, np.ndarray, float]:
        """Run lift→pivot→hold without per-chunk sync; collect metrics from the terminal state only."""
        total_dynamic_steps = self.cfg.lift_steps + self.cfg.pivot_steps + self.cfg.hold_steps
        for dynamic_t in range(total_dynamic_steps):
            self.data.ctrl[:] = self._ctrl_for_dynamic_step(dynamic_t, get_finger_ctrl)
            self._step_dynamics(force_sync=False)
        self._sync_mujoco_from_backend()

        peak_z = max(z_before, float(self.data.xpos[self.cube_body_id, 2]))
        finger_flags = self._finger_contact_flags()
        finger_contact_steps = finger_flags * float(total_dynamic_steps)
        all_finger_contact_steps = float(total_dynamic_steps) if float(np.min(finger_flags)) > 0.5 else 0.0
        contact_active_steps = total_dynamic_steps if float(np.sum(finger_flags)) >= 2.0 else 0
        return peak_z, contact_active_steps, finger_contact_steps, all_finger_contact_steps

    def _evaluate_with_provider(
        self,
        settle_finger_ctrl: np.ndarray,
        get_finger_ctrl: Callable[[int], np.ndarray],
    ) -> tuple[float, dict[str, float]]:
        """Shared evaluation body for scalar and trajectory paths."""
        self._reset_to_keyframe()
        settle_ctrl = self._build_full_ctrl(settle_finger_ctrl, lift=False)
        self._step_with_ctrl(settle_ctrl, self.cfg.settle_steps)

        distances = self._tip_distances_to_cube()
        contact_count = self._cube_tip_contact_count()
        cube_pos_before = self.data.xpos[self.cube_body_id].copy()
        cube_rot_before = self.data.xmat[self.cube_body_id].reshape(3, 3).copy()
        cube_yaw_before = self._cube_yaw(cube_rot_before)
        finger_qpos_settle = self.data.qpos[self.finger_joint_qpos_ids].copy()
        tip_positions_settle = self._tip_positions().copy()
        z_before = float(self.data.xpos[self.cube_body_id, 2])

        fc_metrics = None
        if self.cfg.objective_weight_force_closure != 0.0:
            wrenches = extract_finger_contacts(
                self.data, self.model, self.tip_body_ids, self.cube_body_id
            )
            fc_metrics = force_closure_metrics(
                wrenches,
                object_com=cube_pos_before,
                mu=self.cfg.force_closure_friction_mu,
                n_edges=self.cfg.force_closure_cone_edges,
                weight_balance=self.cfg.force_closure_weight_balance,
                weight_q1=self.cfg.force_closure_weight_q1,
            )

        if self.metric_collection_mode == "terminal" and self.backend != "mujoco":
            peak_z, contact_active_steps, finger_contact_steps, all_finger_contact_steps = (
                self._run_dynamic_loop_terminal(get_finger_ctrl, z_before)
            )
        else:
            peak_z, contact_active_steps, finger_contact_steps, all_finger_contact_steps = (
                self._run_dynamic_loop_sampled(get_finger_ctrl)
            )

        z_after_hold = float(self.data.xpos[self.cube_body_id, 2])
        cube_pos_after = self.data.xpos[self.cube_body_id].copy()
        cube_rot_after = self.data.xmat[self.cube_body_id].reshape(3, 3).copy()
        cube_yaw_after = self._cube_yaw(cube_rot_after)
        finger_qpos_after = self.data.qpos[self.finger_joint_qpos_ids].copy()
        cube_vel_norm = float(np.linalg.norm(self.data.qvel[:6]))
        total_dynamic_steps = self.cfg.lift_steps + self.cfg.pivot_steps + self.cfg.hold_steps

        return self._compute_score_and_metrics(
            distances=distances,
            contact_count=contact_count,
            z_before=z_before,
            peak_z=peak_z,
            z_after_hold=z_after_hold,
            cube_pos_before=cube_pos_before,
            cube_pos_after=cube_pos_after,
            cube_rot_before=cube_rot_before,
            cube_rot_after=cube_rot_after,
            cube_yaw_before=cube_yaw_before,
            cube_yaw_after=cube_yaw_after,
            cube_vel_norm=cube_vel_norm,
            contact_active_steps=contact_active_steps,
            finger_contact_steps=finger_contact_steps,
            all_finger_contact_steps=all_finger_contact_steps,
            total_dynamic_steps=total_dynamic_steps,
            finger_qpos_settle=finger_qpos_settle,
            finger_qpos_after=finger_qpos_after,
            tip_positions_settle=tip_positions_settle,
            fc_metrics=fc_metrics,
        )

    def evaluate(self, finger_ctrl: np.ndarray) -> tuple[float, dict[str, float]]:
        finger_ctrl = np.clip(
            np.asarray(finger_ctrl, dtype=np.float64),
            self.finger_ctrl_min,
            self.finger_ctrl_max,
        )
        return self._evaluate_with_provider(finger_ctrl, lambda _t: finger_ctrl)

    def evaluate_trajectory(self, finger_ctrl_traj: np.ndarray) -> tuple[float, dict[str, float]]:
        """Evaluate a piecewise-linear finger control trajectory across grasp→lift→pivot→hold."""
        finger_ctrl_traj = np.asarray(finger_ctrl_traj, dtype=np.float64)
        if finger_ctrl_traj.ndim != 2 or finger_ctrl_traj.shape[1] != self.finger_actuator_ids.size:
            raise ValueError("Trajectory finger control size mismatch")
        interp = build_trajectory_interpolator(finger_ctrl_traj, self)
        return self._evaluate_with_provider(finger_ctrl_traj[0], interp.at_dynamic_step)

    def _rollout_with_provider(
        self,
        settle_finger_ctrl: np.ndarray,
        get_finger_ctrl: Callable[[int], np.ndarray],
    ) -> dict[str, np.ndarray]:
        self._reset_to_keyframe()
        settle_ctrl = self._build_full_ctrl(settle_finger_ctrl, lift=False)

        total_steps = int(
            self.cfg.settle_steps + self.cfg.lift_steps + self.cfg.pivot_steps + self.cfg.hold_steps
        )
        qpos = np.zeros((total_steps, self.model.nq), dtype=np.float64)
        qvel = np.zeros((total_steps, self.model.nv), dtype=np.float64)
        cube_z = np.zeros(total_steps, dtype=np.float64)
        contacts = np.zeros(total_steps, dtype=np.float64)

        for t in range(total_steps):
            if t < self.cfg.settle_steps:
                self.data.ctrl[:] = settle_ctrl
            else:
                self.data.ctrl[:] = self._ctrl_for_dynamic_step(t - self.cfg.settle_steps, get_finger_ctrl)
            self._step_dynamics(force_sync=False)
            if self.backend == "mujoco" or ((t + 1) % self.metric_sample_interval == 0) or (t == total_steps - 1):
                self._sync_mujoco_from_backend()
            qpos[t] = self.data.qpos
            qvel[t] = self.data.qvel
            cube_z[t] = self.data.xpos[self.cube_body_id, 2]
            contacts[t] = self._cube_tip_contact_count()

        return {"qpos": qpos, "qvel": qvel, "cube_z": cube_z, "contacts": contacts}

    def rollout(self, finger_ctrl: np.ndarray) -> dict[str, np.ndarray]:
        finger_ctrl = np.clip(
            np.asarray(finger_ctrl, dtype=np.float64),
            self.finger_ctrl_min,
            self.finger_ctrl_max,
        )
        return self._rollout_with_provider(finger_ctrl, lambda _t: finger_ctrl)

    def rollout_trajectory(self, finger_ctrl_traj: np.ndarray) -> dict[str, np.ndarray]:
        finger_ctrl_traj = np.asarray(finger_ctrl_traj, dtype=np.float64)
        if finger_ctrl_traj.ndim != 2 or finger_ctrl_traj.shape[1] != self.finger_actuator_ids.size:
            raise ValueError("Trajectory finger control size mismatch")
        interp = build_trajectory_interpolator(finger_ctrl_traj, self)
        return self._rollout_with_provider(finger_ctrl_traj[0], interp.at_dynamic_step)

    def _render_rollout_with_provider(
        self,
        settle_finger_ctrl: np.ndarray,
        get_finger_ctrl: Callable[[int], np.ndarray],
        output_path: Path,
        width: int,
        height: int,
        fps: int,
        frame_stride: int,
    ) -> Path:
        self._reset_to_keyframe()
        settle_ctrl = self._build_full_ctrl(settle_finger_ctrl, lift=False)

        max_width = int(self.model.vis.global_.offwidth)
        max_height = int(self.model.vis.global_.offheight)
        renderer = mujoco.Renderer(
            self.model,
            height=min(height, max_height),
            width=min(width, max_width),
        )
        total_steps = int(
            self.cfg.settle_steps + self.cfg.lift_steps + self.cfg.pivot_steps + self.cfg.hold_steps
        )
        frames: list[np.ndarray] = []

        for t in range(total_steps):
            if t < self.cfg.settle_steps:
                self.data.ctrl[:] = settle_ctrl
            else:
                self.data.ctrl[:] = self._ctrl_for_dynamic_step(t - self.cfg.settle_steps, get_finger_ctrl)
            self._step_dynamics(force_sync=False)
            if self.backend == "mujoco" or ((t + 1) % self.metric_sample_interval == 0) or (t == total_steps - 1):
                self._sync_mujoco_from_backend()
            if t % frame_stride == 0:
                renderer.update_scene(self.data)
                frames.append(renderer.render().copy())

        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix == ".gif":
            imageio.mimsave(output_path, frames, fps=fps)
        elif suffix in (".mp4", ".m4v", ".mov"):
            _write_mp4(output_path, frames, fps)
        else:
            raise ValueError(f"Unsupported rollout output extension: {output_path.suffix!r}")
        return output_path

    def render_rollout(
        self,
        finger_ctrl: np.ndarray,
        output_path: Path,
        width: int = 720,
        height: int = 540,
        fps: int = 25,
        frame_stride: int = 4,
    ) -> Path:
        finger_ctrl = np.clip(
            np.asarray(finger_ctrl, dtype=np.float64),
            self.finger_ctrl_min,
            self.finger_ctrl_max,
        )
        return self._render_rollout_with_provider(
            finger_ctrl, lambda _t: finger_ctrl, output_path, width, height, fps, frame_stride,
        )

    def render_rollout_trajectory(
        self,
        finger_ctrl_traj: np.ndarray,
        output_path: Path,
        width: int = 720,
        height: int = 540,
        fps: int = 25,
        frame_stride: int = 4,
    ) -> Path:
        finger_ctrl_traj = np.asarray(finger_ctrl_traj, dtype=np.float64)
        if finger_ctrl_traj.ndim != 2 or finger_ctrl_traj.shape[1] != self.finger_actuator_ids.size:
            raise ValueError("Trajectory finger control size mismatch")
        interp = build_trajectory_interpolator(finger_ctrl_traj, self)
        return self._render_rollout_with_provider(
            finger_ctrl_traj[0], interp.at_dynamic_step, output_path, width, height, fps, frame_stride,
        )


def _write_mp4(output_path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("No frames to write")
    h, w, _ = frames[0].shape
    # libx264 + yuv420p requires even dimensions; pad up by one row/col if needed.
    pad_w = w + (w % 2)
    pad_h = h + (h % 2)
    vf = "null" if (pad_w == w and pad_h == h) else f"pad={pad_w}:{pad_h}:0:0"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-vf", vf, "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-crf", "28", "-preset", "slow",
        "-movflags", "+faststart",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    finally:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with code {rc} while writing {output_path}")
