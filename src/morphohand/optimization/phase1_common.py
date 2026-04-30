from __future__ import annotations
# pyright: reportMissingImports=false

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

from .phase1_trajectory import TrajectoryInterpolator, build_trajectory_interpolator


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
    ) -> None:
        self.scene_xml = Path(scene_xml)
        self.cfg = cfg or Phase1EvalConfig()
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

    def _build_lift_ctrl(self, finger_ctrl: np.ndarray, lift_scale: float) -> np.ndarray:
        ctrl = self.initial_ctrl.copy()
        ctrl[self.finger_actuator_ids] = finger_ctrl
        ctrl[self.pose_actuator_ids[2]] = self.initial_palm_pz_target + lift_scale * self.cfg.lift_delta_z
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

    def _step_chunk_with_ctrl(self, ctrl: np.ndarray, steps: int) -> None:
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

    def evaluate(self, finger_ctrl: np.ndarray) -> tuple[float, dict[str, float]]:
        finger_ctrl = np.asarray(finger_ctrl, dtype=np.float64)
        finger_ctrl = np.clip(finger_ctrl, self.finger_ctrl_min, self.finger_ctrl_max)

        self._reset_to_keyframe()

        settle_ctrl = self._build_full_ctrl(finger_ctrl, lift=False)
        self._step_with_ctrl(settle_ctrl, self.cfg.settle_steps)

        distances = self._tip_distances_to_cube()
        contact_count = self._cube_tip_contact_count()
        cube_pos_before = self.data.xpos[self.cube_body_id].copy()
        cube_rot_before = self.data.xmat[self.cube_body_id].reshape(3, 3).copy()
        cube_yaw_before = self._cube_yaw(cube_rot_before)
        finger_qpos_settle = self.data.qpos[self.finger_joint_qpos_ids].copy()
        z_before = float(self.data.xpos[self.cube_body_id, 2])

        peak_z = z_before
        contact_active_steps = 0
        finger_contact_steps = np.zeros(3, dtype=np.float64)
        all_finger_contact_steps = 0.0
        total_dynamic_steps = self.cfg.lift_steps + self.cfg.pivot_steps + self.cfg.hold_steps

        if self.metric_collection_mode == "terminal" and self.backend != "mujoco":
            for t in range(self.cfg.lift_steps):
                ramp_denom = max(1, int(self.cfg.lift_ramp_steps))
                lift_scale = min(1.0, float((t + 1) / ramp_denom))
                self.data.ctrl[:] = self._build_pose_ctrl(finger_ctrl, lift_scale=lift_scale, pivot_scale=0.0)
                self._step_dynamics(force_sync=False)
            for t in range(self.cfg.pivot_steps):
                ramp_denom = max(1, int(self.cfg.pivot_ramp_steps))
                pivot_scale = min(1.0, float((t + 1) / ramp_denom))
                self.data.ctrl[:] = self._build_pose_ctrl(finger_ctrl, lift_scale=1.0, pivot_scale=pivot_scale)
                self._step_dynamics(force_sync=False)
            for _ in range(self.cfg.hold_steps):
                final_pivot = 1.0 if self.cfg.pivot_steps > 0 else 0.0
                self.data.ctrl[:] = self._build_pose_ctrl(finger_ctrl, lift_scale=1.0, pivot_scale=final_pivot)
                self._step_dynamics(force_sync=False)
            self._sync_mujoco_from_backend()

            z_after_terminal = float(self.data.xpos[self.cube_body_id, 2])
            peak_z = max(z_before, z_after_terminal)
            finger_flags = self._finger_contact_flags()
            finger_contact_steps = finger_flags * float(total_dynamic_steps)
            all_finger_contact_steps = float(total_dynamic_steps) if float(np.min(finger_flags)) > 0.5 else 0.0
            contact_active_steps = total_dynamic_steps if float(np.sum(finger_flags)) >= 2.0 else 0
        else:
            sample_stride = 1 if self.backend == "mujoco" else self.metric_sample_interval

            lift_t = 0
            while lift_t < self.cfg.lift_steps:
                remaining = self.cfg.lift_steps - lift_t
                chunk = max(1, min(sample_stride, remaining))
                for k in range(chunk):
                    t = lift_t + k
                    ramp_denom = max(1, int(self.cfg.lift_ramp_steps))
                    lift_scale = min(1.0, float((t + 1) / ramp_denom))
                    self.data.ctrl[:] = self._build_pose_ctrl(finger_ctrl, lift_scale=lift_scale, pivot_scale=0.0)
                    self._step_dynamics(force_sync=False)
                self._sync_mujoco_from_backend()
                peak_z = max(peak_z, float(self.data.xpos[self.cube_body_id, 2]))
                finger_flags = self._finger_contact_flags()
                finger_contact_steps += finger_flags * float(chunk)
                if float(np.min(finger_flags)) > 0.5:
                    all_finger_contact_steps += float(chunk)
                if float(np.sum(finger_flags)) >= 2.0:
                    contact_active_steps += int(chunk)
                lift_t += chunk

            pivot_t = 0
            while pivot_t < self.cfg.pivot_steps:
                remaining = self.cfg.pivot_steps - pivot_t
                chunk = max(1, min(sample_stride, remaining))
                for k in range(chunk):
                    t = pivot_t + k
                    ramp_denom = max(1, int(self.cfg.pivot_ramp_steps))
                    pivot_scale = min(1.0, float((t + 1) / ramp_denom))
                    self.data.ctrl[:] = self._build_pose_ctrl(finger_ctrl, lift_scale=1.0, pivot_scale=pivot_scale)
                    self._step_dynamics(force_sync=False)
                self._sync_mujoco_from_backend()
                peak_z = max(peak_z, float(self.data.xpos[self.cube_body_id, 2]))
                finger_flags = self._finger_contact_flags()
                finger_contact_steps += finger_flags * float(chunk)
                if float(np.min(finger_flags)) > 0.5:
                    all_finger_contact_steps += float(chunk)
                if float(np.sum(finger_flags)) >= 2.0:
                    contact_active_steps += int(chunk)
                pivot_t += chunk

            hold_t = 0
            while hold_t < self.cfg.hold_steps:
                remaining = self.cfg.hold_steps - hold_t
                chunk = max(1, min(sample_stride, remaining))
                for _ in range(chunk):
                    final_pivot = 1.0 if self.cfg.pivot_steps > 0 else 0.0
                    self.data.ctrl[:] = self._build_pose_ctrl(finger_ctrl, lift_scale=1.0, pivot_scale=final_pivot)
                    self._step_dynamics(force_sync=False)
                self._sync_mujoco_from_backend()
                peak_z = max(peak_z, float(self.data.xpos[self.cube_body_id, 2]))
                finger_flags = self._finger_contact_flags()
                finger_contact_steps += finger_flags * float(chunk)
                if float(np.min(finger_flags)) > 0.5:
                    all_finger_contact_steps += float(chunk)
                if float(np.sum(finger_flags)) >= 2.0:
                    contact_active_steps += int(chunk)
                hold_t += chunk

        z_after_hold = float(self.data.xpos[self.cube_body_id, 2])
        cube_pos_after = self.data.xpos[self.cube_body_id].copy()
        cube_rot_after = self.data.xmat[self.cube_body_id].reshape(3, 3).copy()
        cube_yaw_after = self._cube_yaw(cube_rot_after)
        finger_qpos_after = self.data.qpos[self.finger_joint_qpos_ids].copy()
        cube_vel_norm = float(np.linalg.norm(self.data.qvel[:6]))

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
        }
        return float(score), metrics

    def evaluate_trajectory(self, finger_ctrl_traj: np.ndarray) -> tuple[float, dict[str, float]]:
        """Evaluate a small trajectory of finger control keypoints.

        Supports time-varying finger control over the grasp→lift→pivot→hold sequence.
        This enables dynamic grip adjustment during reorientation maneuvers.

        Args:
            finger_ctrl_traj: Shape (phases, n_f). Phases typically = 4:
              - Index 0: settle phase control.
              - Index 1: target control at end of lift.
              - Index 2: target control at end of pivot.
              - Index 3: target control at end of hold.

        Returns:
            (score, metrics) tuple with full evaluation results.

        Note: Uses phase1_trajectory.TrajectoryInterpolator for clean phase-to-phase mapping.
        """
        finger_ctrl_traj = np.asarray(finger_ctrl_traj, dtype=np.float64)
        phases, n_f = finger_ctrl_traj.shape
        if n_f != self.finger_actuator_ids.size:
            raise ValueError("Trajectory finger control size mismatch")

        interp = build_trajectory_interpolator(finger_ctrl_traj, self)

        self._reset_to_keyframe()

        # settle
        settle_ctrl = self.initial_ctrl.copy()
        settle_ctrl[self.finger_actuator_ids] = finger_ctrl_traj[0]
        self._step_with_ctrl(settle_ctrl, self.cfg.settle_steps)

        distances = self._tip_distances_to_cube()
        contact_count = self._cube_tip_contact_count()
        cube_pos_before = self.data.xpos[self.cube_body_id].copy()
        cube_rot_before = self.data.xmat[self.cube_body_id].reshape(3, 3).copy()
        cube_yaw_before = self._cube_yaw(cube_rot_before)
        finger_qpos_settle = self.data.qpos[self.finger_joint_qpos_ids].copy()
        z_before = float(self.data.xpos[self.cube_body_id, 2])

        peak_z = z_before
        contact_active_steps = 0
        finger_contact_steps = np.zeros(3, dtype=np.float64)
        all_finger_contact_steps = 0.0
        total_dynamic_steps = self.cfg.lift_steps + self.cfg.pivot_steps + self.cfg.hold_steps

        sample_stride = 1 if self.backend == "mujoco" else self.metric_sample_interval

        # dynamic phases: lift, pivot, hold
        dynamic_t = 0
        # lift
        while dynamic_t < self.cfg.lift_steps:
            chunk = max(1, min(sample_stride, int(self.cfg.lift_steps - dynamic_t)))
            for k in range(chunk):
                t = dynamic_t + k
                lift_scale = min(1.0, float((t + 1) / max(1, int(self.cfg.lift_ramp_steps))))
                pivot_scale = 0.0
                finger_ctrl = interp.at_dynamic_step(t)
                ctrl = self.initial_ctrl.copy()
                ctrl[self.finger_actuator_ids] = finger_ctrl
                ctrl[self.pose_actuator_ids[2]] = self.initial_palm_pz_target + lift_scale * self.cfg.lift_delta_z
                self.data.ctrl[:] = ctrl
                self._step_dynamics(force_sync=False)
            self._sync_mujoco_from_backend()
            peak_z = max(peak_z, float(self.data.xpos[self.cube_body_id, 2]))
            finger_flags = self._finger_contact_flags()
            finger_contact_steps += finger_flags * float(chunk)
            if float(np.min(finger_flags)) > 0.5:
                all_finger_contact_steps += float(chunk)
            if float(np.sum(finger_flags)) >= 2.0:
                contact_active_steps += int(chunk)
            dynamic_t += chunk

        # pivot
        pivot_t = 0
        while pivot_t < self.cfg.pivot_steps:
            chunk = max(1, min(sample_stride, int(self.cfg.pivot_steps - pivot_t)))
            for k in range(chunk):
                t = pivot_t + k
                pivot_scale = min(1.0, float((t + 1) / max(1, int(self.cfg.pivot_ramp_steps))))
                finger_ctrl = interp.at_dynamic_step(int(self.cfg.lift_steps + t))
                ctrl = self.initial_ctrl.copy()
                ctrl[self.finger_actuator_ids] = finger_ctrl
                ctrl[self.pose_actuator_ids[2]] = self.initial_palm_pz_target + self.cfg.lift_delta_z
                ctrl[self.pose_actuator_ids[3]] = self.initial_ctrl[self.pose_actuator_ids[3]] + pivot_scale * self.cfg.pivot_delta_rx
                ctrl[self.pose_actuator_ids[4]] = self.initial_ctrl[self.pose_actuator_ids[4]] + pivot_scale * self.cfg.pivot_delta_ry
                ctrl[self.pose_actuator_ids[5]] = self.initial_ctrl[self.pose_actuator_ids[5]] + pivot_scale * self.cfg.pivot_delta_rz
                self.data.ctrl[:] = ctrl
                self._step_dynamics(force_sync=False)
            self._sync_mujoco_from_backend()
            peak_z = max(peak_z, float(self.data.xpos[self.cube_body_id, 2]))
            finger_flags = self._finger_contact_flags()
            finger_contact_steps += finger_flags * float(chunk)
            if float(np.min(finger_flags)) > 0.5:
                all_finger_contact_steps += float(chunk)
            if float(np.sum(finger_flags)) >= 2.0:
                contact_active_steps += int(chunk)
            pivot_t += chunk

        # hold
        hold_t = 0
        while hold_t < self.cfg.hold_steps:
            chunk = max(1, min(sample_stride, int(self.cfg.hold_steps - hold_t)))
            for k in range(chunk):
                t = hold_t + k
                finger_ctrl = interp.at_dynamic_step(int(self.cfg.lift_steps + self.cfg.pivot_steps + t))
                ctrl = self.initial_ctrl.copy()
                ctrl[self.finger_actuator_ids] = finger_ctrl
                ctrl[self.pose_actuator_ids[2]] = self.initial_palm_pz_target + self.cfg.lift_delta_z
                ctrl[self.pose_actuator_ids[3]] = self.initial_ctrl[self.pose_actuator_ids[3]] + self.cfg.pivot_delta_rx
                ctrl[self.pose_actuator_ids[4]] = self.initial_ctrl[self.pose_actuator_ids[4]] + self.cfg.pivot_delta_ry
                ctrl[self.pose_actuator_ids[5]] = self.initial_ctrl[self.pose_actuator_ids[5]] + self.cfg.pivot_delta_rz
                self.data.ctrl[:] = ctrl
                self._step_dynamics(force_sync=False)
            self._sync_mujoco_from_backend()
            peak_z = max(peak_z, float(self.data.xpos[self.cube_body_id, 2]))
            finger_flags = self._finger_contact_flags()
            finger_contact_steps += finger_flags * float(chunk)
            if float(np.min(finger_flags)) > 0.5:
                all_finger_contact_steps += float(chunk)
            if float(np.sum(finger_flags)) >= 2.0:
                contact_active_steps += int(chunk)
            hold_t += chunk

        z_after_hold = float(self.data.xpos[self.cube_body_id, 2])
        cube_pos_after = self.data.xpos[self.cube_body_id].copy()
        cube_rot_after = self.data.xmat[self.cube_body_id].reshape(3, 3).copy()
        cube_yaw_after = self._cube_yaw(cube_rot_after)
        finger_qpos_after = self.data.qpos[self.finger_joint_qpos_ids].copy()
        cube_vel_norm = float(np.linalg.norm(self.data.qvel[:6]))

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
        }
        return float(score), metrics

    def rollout_trajectory(self, finger_ctrl_traj: np.ndarray) -> dict[str, np.ndarray]:
        """Generate full (qpos, qvel, cube_z, contacts) rollout for a trajectory.

        Args:
            finger_ctrl_traj: Shape (phases, n_f).

        Returns:
            Dictionary with keys: qpos, qvel, cube_z, contacts.
        """
        finger_ctrl_traj = np.asarray(finger_ctrl_traj, dtype=np.float64)
        phases, n_f = finger_ctrl_traj.shape
        if n_f != self.finger_actuator_ids.size:
            raise ValueError("Trajectory finger control size mismatch")

        interp = build_trajectory_interpolator(finger_ctrl_traj, self)
        self._reset_to_keyframe()

        total_steps = int(self.cfg.settle_steps + self.cfg.lift_steps + self.cfg.pivot_steps + self.cfg.hold_steps)
        qpos = np.zeros((total_steps, self.model.nq), dtype=np.float64)
        qvel = np.zeros((total_steps, self.model.nv), dtype=np.float64)
        cube_z = np.zeros(total_steps, dtype=np.float64)
        contacts = np.zeros(total_steps, dtype=np.float64)

        for t in range(total_steps):
            if t < self.cfg.settle_steps:
                ctrl = self.initial_ctrl.copy()
                ctrl[self.finger_actuator_ids] = finger_ctrl_traj[0]
            else:
                dynamic_t = t - self.cfg.settle_steps
                if dynamic_t < self.cfg.lift_steps:
                    ramp_denom = max(1, int(self.cfg.lift_ramp_steps))
                    lift_scale = min(1.0, float((dynamic_t + 1) / ramp_denom))
                    pivot_scale = 0.0
                elif dynamic_t < (self.cfg.lift_steps + self.cfg.pivot_steps):
                    pivot_t = dynamic_t - self.cfg.lift_steps
                    ramp_denom = max(1, int(self.cfg.pivot_ramp_steps))
                    lift_scale = 1.0
                    pivot_scale = min(1.0, float((pivot_t + 1) / ramp_denom))
                else:
                    lift_scale = 1.0
                    pivot_scale = 1.0 if self.cfg.pivot_steps > 0 else 0.0
                finger_ctrl = interp.at_dynamic_step(dynamic_t)
                ctrl = self.initial_ctrl.copy()
                ctrl[self.finger_actuator_ids] = finger_ctrl
                ctrl[self.pose_actuator_ids[2]] = self.initial_palm_pz_target + lift_scale * self.cfg.lift_delta_z
                ctrl[self.pose_actuator_ids[3]] = self.initial_ctrl[self.pose_actuator_ids[3]] + pivot_scale * self.cfg.pivot_delta_rx
                ctrl[self.pose_actuator_ids[4]] = self.initial_ctrl[self.pose_actuator_ids[4]] + pivot_scale * self.cfg.pivot_delta_ry
                ctrl[self.pose_actuator_ids[5]] = self.initial_ctrl[self.pose_actuator_ids[5]] + pivot_scale * self.cfg.pivot_delta_rz
            self.data.ctrl[:] = ctrl
            self._step_dynamics(force_sync=False)
            if self.backend == "mujoco" or ((t + 1) % self.metric_sample_interval == 0) or (t == total_steps - 1):
                self._sync_mujoco_from_backend()
            qpos[t] = self.data.qpos
            qvel[t] = self.data.qvel
            cube_z[t] = self.data.xpos[self.cube_body_id, 2]
            contacts[t] = self._cube_tip_contact_count()

        return {
            "qpos": qpos,
            "qvel": qvel,
            "cube_z": cube_z,
            "contacts": contacts,
        }

    def render_rollout_gif_trajectory(
        self,
        finger_ctrl_traj: np.ndarray,
        output_gif: Path,
        width: int = 720,
        height: int = 540,
        fps: int = 25,
        frame_stride: int = 4,
    ) -> Path:
        """Render full trajectory rollout as GIF animation.

        Args:
            finger_ctrl_traj: Shape (phases, n_f).
            output_gif: Output path for GIF file.
            width, height: Video dimensions.
            fps: Frames per second.
            frame_stride: Render every Nth frame.

        Returns:
            Path to created GIF.
        """
        finger_ctrl_traj = np.asarray(finger_ctrl_traj, dtype=np.float64)
        phases, n_f = finger_ctrl_traj.shape
        if n_f != self.finger_actuator_ids.size:
            raise ValueError("Trajectory finger control size mismatch")

        interp = build_trajectory_interpolator(finger_ctrl_traj, self)
        self._reset_to_keyframe()

        max_width = int(self.model.vis.global_.offwidth)
        max_height = int(self.model.vis.global_.offheight)
        render_width = min(width, max_width)
        render_height = min(height, max_height)
        renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        total_steps = int(self.cfg.settle_steps + self.cfg.lift_steps + self.cfg.pivot_steps + self.cfg.hold_steps)
        frames: list[np.ndarray] = []

        for t in range(total_steps):
            if t < self.cfg.settle_steps:
                ctrl = self.initial_ctrl.copy()
                ctrl[self.finger_actuator_ids] = finger_ctrl_traj[0]
            else:
                dynamic_t = t - self.cfg.settle_steps
                if dynamic_t < self.cfg.lift_steps:
                    ramp_denom = max(1, int(self.cfg.lift_ramp_steps))
                    lift_scale = min(1.0, float((dynamic_t + 1) / ramp_denom))
                    pivot_scale = 0.0
                elif dynamic_t < (self.cfg.lift_steps + self.cfg.pivot_steps):
                    pivot_t = dynamic_t - self.cfg.lift_steps
                    ramp_denom = max(1, int(self.cfg.pivot_ramp_steps))
                    lift_scale = 1.0
                    pivot_scale = min(1.0, float((pivot_t + 1) / ramp_denom))
                else:
                    lift_scale = 1.0
                    pivot_scale = 1.0 if self.cfg.pivot_steps > 0 else 0.0
                finger_ctrl = interp.at_dynamic_step(dynamic_t)
                ctrl = self.initial_ctrl.copy()
                ctrl[self.finger_actuator_ids] = finger_ctrl
                ctrl[self.pose_actuator_ids[2]] = self.initial_palm_pz_target + lift_scale * self.cfg.lift_delta_z
                ctrl[self.pose_actuator_ids[3]] = self.initial_ctrl[self.pose_actuator_ids[3]] + pivot_scale * self.cfg.pivot_delta_rx
                ctrl[self.pose_actuator_ids[4]] = self.initial_ctrl[self.pose_actuator_ids[4]] + pivot_scale * self.cfg.pivot_delta_ry
                ctrl[self.pose_actuator_ids[5]] = self.initial_ctrl[self.pose_actuator_ids[5]] + pivot_scale * self.cfg.pivot_delta_rz
            self.data.ctrl[:] = ctrl
            self._step_dynamics(force_sync=False)
            if self.backend == "mujoco" or ((t + 1) % self.metric_sample_interval == 0) or (t == total_steps - 1):
                self._sync_mujoco_from_backend()
            if t % frame_stride == 0:
                renderer.update_scene(self.data)
                frames.append(renderer.render().copy())

        output_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_gif, frames, fps=fps)
        return output_gif

    def rollout(self, finger_ctrl: np.ndarray) -> dict[str, np.ndarray]:
        finger_ctrl = np.asarray(finger_ctrl, dtype=np.float64)
        finger_ctrl = np.clip(finger_ctrl, self.finger_ctrl_min, self.finger_ctrl_max)
        self._reset_to_keyframe()

        settle_ctrl = self._build_full_ctrl(finger_ctrl, lift=False)
        total_steps = self.cfg.settle_steps + self.cfg.lift_steps + self.cfg.pivot_steps + self.cfg.hold_steps
        qpos = np.zeros((total_steps, self.model.nq), dtype=np.float64)
        qvel = np.zeros((total_steps, self.model.nv), dtype=np.float64)
        cube_z = np.zeros(total_steps, dtype=np.float64)
        contacts = np.zeros(total_steps, dtype=np.float64)

        for t in range(total_steps):
            if t < self.cfg.settle_steps:
                self.data.ctrl[:] = settle_ctrl
            else:
                dynamic_t = t - self.cfg.settle_steps
                if dynamic_t < self.cfg.lift_steps:
                    ramp_denom = max(1, int(self.cfg.lift_ramp_steps))
                    lift_scale = min(1.0, float((dynamic_t + 1) / ramp_denom))
                    pivot_scale = 0.0
                elif dynamic_t < (self.cfg.lift_steps + self.cfg.pivot_steps):
                    pivot_t = dynamic_t - self.cfg.lift_steps
                    ramp_denom = max(1, int(self.cfg.pivot_ramp_steps))
                    lift_scale = 1.0
                    pivot_scale = min(1.0, float((pivot_t + 1) / ramp_denom))
                else:
                    lift_scale = 1.0
                    pivot_scale = 1.0 if self.cfg.pivot_steps > 0 else 0.0
                self.data.ctrl[:] = self._build_pose_ctrl(finger_ctrl, lift_scale=lift_scale, pivot_scale=pivot_scale)
            self._step_dynamics(force_sync=False)
            if self.backend == "mujoco" or ((t + 1) % self.metric_sample_interval == 0) or (t == total_steps - 1):
                self._sync_mujoco_from_backend()
            qpos[t] = self.data.qpos
            qvel[t] = self.data.qvel
            cube_z[t] = self.data.xpos[self.cube_body_id, 2]
            contacts[t] = self._cube_tip_contact_count()

        return {
            "qpos": qpos,
            "qvel": qvel,
            "cube_z": cube_z,
            "contacts": contacts,
        }

    def render_rollout_gif(
        self,
        finger_ctrl: np.ndarray,
        output_gif: Path,
        width: int = 720,
        height: int = 540,
        fps: int = 25,
        frame_stride: int = 4,
    ) -> Path:
        finger_ctrl = np.asarray(finger_ctrl, dtype=np.float64)
        finger_ctrl = np.clip(finger_ctrl, self.finger_ctrl_min, self.finger_ctrl_max)
        self._reset_to_keyframe()

        settle_ctrl = self._build_full_ctrl(finger_ctrl, lift=False)
        max_width = int(self.model.vis.global_.offwidth)
        max_height = int(self.model.vis.global_.offheight)
        render_width = min(width, max_width)
        render_height = min(height, max_height)
        renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        total_steps = self.cfg.settle_steps + self.cfg.lift_steps + self.cfg.pivot_steps + self.cfg.hold_steps
        frames: list[np.ndarray] = []

        for t in range(total_steps):
            if t < self.cfg.settle_steps:
                self.data.ctrl[:] = settle_ctrl
            else:
                dynamic_t = t - self.cfg.settle_steps
                if dynamic_t < self.cfg.lift_steps:
                    ramp_denom = max(1, int(self.cfg.lift_ramp_steps))
                    lift_scale = min(1.0, float((dynamic_t + 1) / ramp_denom))
                    pivot_scale = 0.0
                elif dynamic_t < (self.cfg.lift_steps + self.cfg.pivot_steps):
                    pivot_t = dynamic_t - self.cfg.lift_steps
                    ramp_denom = max(1, int(self.cfg.pivot_ramp_steps))
                    lift_scale = 1.0
                    pivot_scale = min(1.0, float((pivot_t + 1) / ramp_denom))
                else:
                    lift_scale = 1.0
                    pivot_scale = 1.0 if self.cfg.pivot_steps > 0 else 0.0
                self.data.ctrl[:] = self._build_pose_ctrl(finger_ctrl, lift_scale=lift_scale, pivot_scale=pivot_scale)
            self._step_dynamics(force_sync=False)
            if self.backend == "mujoco" or ((t + 1) % self.metric_sample_interval == 0) or (t == total_steps - 1):
                self._sync_mujoco_from_backend()
            if t % frame_stride == 0:
                renderer.update_scene(self.data)
                frames.append(renderer.render().copy())

        output_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_gif, frames, fps=fps)
        return output_gif
