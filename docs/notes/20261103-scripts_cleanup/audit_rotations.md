# Rotation and rigid-transform math in the hand repository (audit, 2026-11-03)

Scope: every `.py` directly in `scripts/` (652), `scripts/contact_surface/` (10, a package imported by the live `distributed_contact*.py`), `scripts/cluster/` (3, no rotation math) and `src/` (110, firmware excluded). Findings come from a regex pass over every line, an AST pass over every function name, and reading each candidate. Each entry reads `file:line name (convention; live|dead; N lines)`; live/dead for scripts is from `live.json`, every `src/` file and the `contact_surface` package count as live. Conventions: wxyz is MuJoCo's, Drake's and mjlab's order, xyzw is Newton's, warp's and scipy's default. All quaternions and matrices found are active rotations (R @ v turns a vector; q * p applies p first), except the four passive Rz(psi)^T copies in 2a.


## Counts per subsection (entries listed below; an inline site counts once per cited line)

| Subsection | Entries | Live | Dead | Files |
|---|---|---|---|---|
| 1a. Quaternion to rotation matrix (full 3 x 3) | 10 | 9 | 1 | 10 |
| 1b. Quaternion to one axis of R (body z, or the vertical cosine R[2,2] = 1 - 2(x^2 + y^2)) | 29 | 23 | 6 | 27 |
| 1c. Rotation matrix to quaternion | 10 | 8 | 2 | 10 |
| 1d. Quaternion product | 9 | 9 | 0 | 8 |
| 1e. Conjugate / inverse | 5 | 4 | 1 | 5 |
| 1f. Rotate a vector by a quaternion | 4 | 4 | 0 | 4 |
| 1g. Axis-angle to quaternion (including yaw quaternions) | 13 | 10 | 3 | 9 |
| 1h. Quaternion angle, distance, twist and yaw | 13 | 10 | 3 | 10 |
| 1i. Interpolation (slerp) | 5 | 5 | 0 | 5 |
| 1j. Normalisation | 8 | 7 | 1 | 8 |
| 1k. Random rotations | 1 | 1 | 0 | 1 |
| 1l. Component-order conversions (wxyz <-> xyzw) and other libraries' quaternions | 7 | 7 | 0 | 5 |
| 2a. Elementary rotations about x, y, z (helpers and inline literals) | 34 | 33 | 1 | 27 |
| 2b. Axis-angle (Rodrigues, exp map) to matrix | 9 | 9 | 0 | 7 |
| 2c. Matrix to axis-angle / rotation angle (log map, trace) | 17 | 14 | 3 | 15 |
| 2d. Rotation taking vector a onto vector b | 7 | 7 | 0 | 7 |
| 2e. Frames from axes: basis completion, look-at, Gram-Schmidt | 38 | 37 | 1 | 27 |
| 2f. Skew (hat) and vee | 21 | 21 | 0 | 14 |
| 2g. Projection onto SO(3) (nearest rotation by SVD) | 6 | 6 | 0 | 3 |
| 3a. Euler and RPY conversions | 5 | 5 | 0 | 4 |
| 3b. Yaw read from a matrix, atan2(R[1,0], R[0,0]) | 12 | 12 | 0 | 8 |
| 3c. Rotation vectors (UR TCP poses and fits) | 4 | 4 | 0 | 3 |
| 4a. 4 x 4 build, compose, invert, apply | 15 | 15 | 0 | 9 |
| 4b. Poses as (p, q) or (p, R): compose, invert, relative pose | 7 | 7 | 0 | 5 |
| 4c. Inline world <-> local frame changes | 7 | 6 | 1 | 7 |
| 5a. Kabsch, Umeyama, Procrustes | 5 | 5 | 0 | 3 |
| 5b. Nonlinear pose fits (local exp-map perturbations) | 6 | 6 | 0 | 3 |
| 5c. Rotation averaging | 2 | 2 | 0 | 2 |

## Library calls already in use, and the two in-repo shared modules

- MuJoCo: 34 rotation calls in 27 files, all wxyz: `mju_quat2Mat` 18, `mju_mat2Quat` 7, `mju_mulQuat` 5,
  `mju_quatZ2Vec` 3, `mju_negQuat` 1. `mju_rotVecQuat`, `mju_mulPose`, `mju_negPose`, `mju_trnVecPose`,
  `mju_axisAngle2Quat`, `mju_subQuat` and `mju_euler2Quat` are not used.
- scipy `Rotation`/`Slerp`: 5 files (fingertip_crosstalk.py, fingertip_crosstalk_holder.py, fingertip_crosstalk_page.py,
  ur5_inverted_rig.py, ur5_tip_calibration_holder_paths.py), always through `from_rotvec`, `from_euler`, `from_matrix`,
  `as_rotvec`, `as_matrix`, `random` and `Slerp`. No call uses `from_quat`, `as_quat` or `scalar_first`.
- trimesh: `transformations.rotation_matrix`/`translation_matrix`/`quaternion_matrix` (wxyz) and
  `geometry.align_vectors` in 9 files; `apply_transform` and `transform_points` throughout the CAD scripts.
- pydrake (wxyz): `RigidTransform`, `RotationMatrix.Make*Rotation`, `Quaternion(wxyz)` and `.ToQuaternion().wxyz()` in
  about 15 files; every read and write keeps w first.
- warp/Newton (xyzw): `wp.quat` and `wp.quat_rotate` in 3 files and `wp.transform` in Newton builders and kernels, with the
  reorderings listed in 1l.
- mjlab (wxyz): `quat_mul`, `quat_from_euler_xyz` in `src/morphohand/rl/lifting_command.py` and the dead rl_eval_object.py.
- In-repo shared code: `src/morphohand/rl/math.py` (torch `quat_mul`, `quat_rotate`, `quat_rotate_inv`; imported by
  terms_obs, imitation and mjlab_terms) and `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py` lines 77-174
  (numpy `skew`, `exp_so3`, `log_so3`, `Tm`, `inv`, `apply`, `ur_to_T`, `T_to_ur`, `rot`, `rot_from_to`, `perturb`,
  `turn_about`, `pose_error`, `mean_pose`; called as `C.*` from 31 files).
- Names defined in several files: `quat_mul` (4 files, 5 bodies, all Hamilton wxyz), `quat_rotate` (3, all wxyz), `rot`
  (5 files with three meanings: quaternion to matrix, axis-angle in radians, axis-angle in degrees), `skew` (3, identical).
  The four `compose` functions (hom_films, hom_plant_page, fingertip_calibration_cell_render, real_v1_bench_video_array)
  compose video frames, not transforms.


## 1. Quaternion algebra


### 1a. Quaternion to rotation matrix (full 3 x 3)

- `src/morphohand/control/hom.py:100` quat_to_mat (wxyz, active, no normalisation; live; 5 lines). Used by hom_tasks, hom_tasks_mpc, hom_code_trace, hom_bench
- `scripts/hom_grasp_search.py:75` rot (wxyz, active, no normalisation; live; 5 lines). Named `rot`
- `scripts/fe_contact_calibration_sim.py:317` quat_R (wxyz, active, normalises; live; 5 lines)
- `scripts/rl_contact_eval.py:719` _quat_mat (wxyz, batched, normalises; live; 7 lines)
- `scripts/reorient_primitive.py:205` _quat_to_mat (wxyz, batched, normalises; dead; 10 lines)
- `scripts/hom_gpu.py:807` _quat_mat (wxyz, torch batched, no normalisation; live; 6 lines)
- `scripts/contact_bed_newton.py:415` inline in tool_state (xyzw from Newton body_q, unpacked as x, y, z, w (correct); live; 4 lines)
- `scripts/ati_calibration_fit.py:114` inline in selftest (wxyz named (a, b, c, d), random unit quaternions; live; 4 lines)
- `src/morphohand/rl/terms_common.py:138` inline in _alignment_cos (wxyz torch, R(q) @ axis written out term by term; live; 5 lines)
- `src/morphohand/rl/terms_reward.py:315` inline in palm_brace_distance (wxyz torch, second copy of the same R(q) @ axis; live; 6 lines)

Replacement: `Rotation.from_quat(q[..., [1, 2, 3, 0]]).as_matrix()` (or `scalar_first=True` on scipy >= 1.14); `mujoco.mju_quat2Mat` for one quaternion where mujoco is already imported. Torch copies: one `quat_to_mat` in `morphohand/rl/math.py`.

### 1b. Quaternion to one axis of R (body z, or the vertical cosine R[2,2] = 1 - 2(x^2 + y^2))

- `scripts/fe_contact_calibration_summary.py:67` axis_of (wxyz, normalises, body z column; live; 5 lines)
- `scripts/pad_builder_check.py:284` _zaxis (wxyz, body z column; live; 4 lines)
- `scripts/real_v1_d1_continuation.py:219` vertical_cos (wxyz, R[2,2]; NumPy and JAX; live; 4 lines)
- `scripts/policy_cpu_rollout.py:142` tool_cos (wxyz, R[2,2] of d.xquat; live; 3 lines)
- `scripts/policy_cpu_rollout.py:315` tool_cos (wxyz, R[2,2]; live; 3 lines). Second copy in the same file
- `src/morphohand/rl/reward.py:144` pen_cube_axis_tilt (wxyz, arccos of R[2,2]; live; 6 lines)
- `scripts/hand_object_scale_ilqr.py:106` inline in pose_res (wxyz, normalised, body z column; live; 1 line)
- `scripts/hom_tasks_mpc.py:160` inline in cost (wxyz, body z column, batched; live; 1 line)
- `scripts/rl_contact_eval.py:406` inline in rollout (wxyz torch, R[2,2]; live; 1 line)
- `scripts/rl_contact_eval.py:408` inline in rollout (wxyz torch, body z column; live; 1 line)
- `scripts/hand_object_scale_mpc.py:169` inline in cost (wxyz, R[2,2], no normalisation; live; 1 line)
- `scripts/hand_object_scale_mpc_gpu.py:156` inline in cost_records (wxyz, R[2,2], normalised (the CPU twin above is not); live; 1 line)
- `scripts/policy_healthcheck.py:89` inline in main (wxyz (mjlab root_link_pose_w), R[2,2]; live; 1 line)
- `scripts/policy_eval_suite.py:462` inline in main (wxyz torch, R[2,2]; live; 1 line)
- `scripts/probe_obs_ablation.py:168` inline in rollout (wxyz torch, R[2,2]; live; 1 line)
- `scripts/rl_eval_reorient_metrics.py:67` inline in run_one (wxyz torch, R[2,2]; live; 1 line)
- `scripts/rl_demo_handoff_continuous.py:293` inline in main (wxyz, R[2,2]; live; 1 line)
- `scripts/rl_render_rollout.py:101` inline in main (wxyz, R[2,2]; live; 1 line)
- `scripts/real_v1_contact_gait_warp.py:91` inline in snapshot (wxyz, normalised, R[2,2]; live; 1 line)
- `src/morphohand/rl/live_a_runner.py:103` inline in _schedule_reward (wxyz torch, R[2,2]; live; 1 line)
- `src/morphohand/rl/actions.py:307` inline in apply_actions (wxyz torch, R[2,2]; live; 1 line)
- `scripts/rl_policy_replay.py:406` inline in tool_state (xyzw (Newton body_q), R[2,2]; components unpacked correctly; live; 1 line)
- `scripts/newton_turn.py:260` inline in snap (xyzw (Newton body_q), R[2,2]; components unpacked correctly; live; 1 line)
- `scripts/compliance_rate_sweep.py:94` inline in rate_point (wxyz torch, R[2,2]; dead; 1 line)
- `scripts/sim2real_robustness_sweep.py:144` inline in rate_point (wxyz torch, R[2,2]; dead; 1 line)
- `scripts/probe_grip_balance.py:77` inline in run_one (wxyz torch, R[2,2]; dead; 1 line)
- `scripts/probe_grip_force.py:67` inline in run_one (wxyz torch, R[2,2]; dead; 1 line)
- `scripts/rl_record_reorient_schedule.py:72` inline in main (wxyz torch, R[2,2]; dead; 1 line)
- `scripts/real_v1_render_deploy_plan.py:164` inline in replay_physics (wxyz, R[2,2]; dead; 1 line)

Besides these, 14 files read the same number from MuJoCo as `xmat[8]` or `R9[8]` after `mju_quat2Mat`. Replacement: one `quat_axis(q, k=2)` (column k of R) and `tilt_cos(q)` (its z component) in the shared module, and the same two in `morphohand/rl/math.py` for torch.

### 1c. Rotation matrix to quaternion

- `scripts/contact_bed_roll.py:387` mat2quat (wxyz, Shepperd four-branch, w >= 0; live; 19 lines)
- `scripts/build_screw_scene.py:38` _quat_from_cols (wxyz, Shepperd variant, returns an MJCF string; live; 17 lines)
- `scripts/ur5_tip_calibration_film.py:38` quat (wxyz, copysign method; live; 7 lines). Wrong for half turns (bug B2)
- `scripts/hom_chain.py:82` quat_from_R (wxyz, wrapper of mju_mat2Quat; live; 5 lines)
- `scripts/ur5_cell_render.py:529` quat (wxyz, wrapper of mju_mat2Quat, lazy mujoco import; live; 5 lines). Same name and signature as the film's `quat`, which is wrong at half turns
- `scripts/hom_contact_rig.py:792` inline in kinematic_cop (wxyz, mju_mat2Quat inline; live; 1 line)
- `scripts/hom_hand_brake.py:260` inline in mj_scene (wxyz, mju_mat2Quat inline; live; 1 line)
- `scripts/pad_contact_rig.py:104` inline in press (wxyz, mju_mat2Quat inline; live; 1 line)
- `scripts/reorient_primitive.py:757` inline in feasibility (wxyz, mju_mat2Quat inline; dead; 1 line)
- `scripts/graspgenx_eval_phase1.py:119` inline in patched_reset (wxyz, mju_mat2Quat inline; dead; 1 line)

Replacement: `Rotation.from_matrix(R).as_quat()[..., [3, 0, 1, 2]]` (orthonormalises, exact at half turns), or `mju_mat2Quat`.

### 1d. Quaternion product

- `src/morphohand/rl/math.py:11` quat_mul (wxyz, Hamilton, torch; live; 10 lines). The existing shared torch module (CODEBASE_AUDIT step 3)
- `scripts/hom_gpu.py:146` quat_mul (wxyz, Hamilton, numpy; live; 5 lines)
- `scripts/hom_gpu.py:815` _quat_mul (wxyz, Hamilton, torch; live; 6 lines). Second copy in the same file
- `scripts/hom_grasp_search.py:63` quat_mul (wxyz, Hamilton; live; 5 lines)
- `scripts/policy_cpu_rollout.py:62` quat_mul (wxyz, Hamilton; live; 4 lines)
- `scripts/real_v1_chain_hands.py:153` _qmul (wxyz, Hamilton, tuples; live; 7 lines)
- `scripts/reorient_backends.py:417` _yaw_quat (wxyz, yaw_z(yaw) * q, world-frame yaw; live; 6 lines). Called from 13 other scripts as `RB._yaw_quat`
- `scripts/plant_drop_gate.py:82` inline in run_trial (wxyz, q * dq, body-frame turn; live; 5 lines). Bug B1
- `scripts/newton_scaling.py:557` twist_deg_xyzw (xyzw, q1 * conj(q0) written with cross products, then the twist about u; live; 6 lines)

Library calls already in use: `mju_mulQuat` (hom_gpu.py:553, hom_tasks.py:435 and :702, real_v1_deploy_envelope.py:425, pads_cpu_gpu_consistency.py:156) and mjlab's `quat_mul` (src/morphohand/rl/lifting_command.py:75 and :85; rl_eval_object.py:199 and :231). Replacement: `(Ra * Rb)` on scipy Rotations, wrapped as `quat_mul(a, b)`.

### 1e. Conjugate / inverse

- `scripts/policy_cpu_rollout.py:74` quat_inv (wxyz; live; 2 lines)
- `src/morphohand/rl/math.py:32` quat_rotate_inv (wxyz torch, conjugate then rotate; live; 5 lines)
- `src/morphohand/rl/terms_obs.py:53` inline in object_pose_rel_palm (wxyz torch, clone and negate xyz; live; 2 lines)
- `scripts/newton_scaling.py:558` inline in twist_deg_xyzw (xyzw, negate xyz; live; 1 line)
- `scripts/pads_cpu_gpu_consistency.py:154` inline in __call__ (wxyz, mju_negQuat; dead; 1 line)

Replacement: `Rotation.inv()`; `quat_conj(q)` in the shared module.

### 1f. Rotate a vector by a quaternion

- `scripts/hom_tasks_mpc.py:110` quat_rotate (wxyz, numpy batched, v + 2w(r x v) + 2 r x (r x v); live; 5 lines)
- `scripts/policy_cpu_rollout.py:68` quat_rotate (wxyz, single; live; 4 lines)
- `src/morphohand/rl/math.py:23` quat_rotate (wxyz, torch; live; 7 lines)
- `scripts/distributed_contact_gpu.py:47` inline in guide (warp kernel: wp.quat(x, y, z, w) built from MJWarp wxyz, then wp.quat_rotate (correct); live; 2 lines)

Replacement: `Rotation.apply(v)`; `mju_rotVecQuat` for single vectors.

### 1g. Axis-angle to quaternion (including yaw quaternions)

- `scripts/hom_grasp_search.py:70` quat_axis (wxyz, normalises the axis; live; 3 lines)
- `scripts/hom_gpu.py:173` inline in pinch_setup (wxyz yaw, pre-multiplied (world); live; 1 line)
- `scripts/hom_gpu.py:551` inline in _init_state (wxyz yaw, pre-multiplied via mju_mulQuat; live; 1 line)
- `scripts/hom_gpu.py:912` inline in measure (wxyz torch, random axis; live; 1 line)
- `scripts/hom_tasks.py:433` inline in reset (wxyz yaw, pre-multiplied via mju_mulQuat; live; 1 line)
- `scripts/hom_tasks.py:700` inline in rollout (wxyz, random unit axis; live; 1 line)
- `scripts/real_v1_deploy_envelope.py:423` inline in execute (wxyz yaw, pre-multiplied; live; 1 line)
- `scripts/real_v1_chain_hands.py:203` inline in prepare (wxyz yaw, pre-multiplied; live; 1 line)
- `scripts/plant_drop_gate.py:81` inline in run_trial (wxyz yaw, post-multiplied; live; 1 line). Bug B1
- `scripts/rl_eval_object.py:197` inline in main (wxyz yaw (torch tensor fill); dead; 1 line)
- `scripts/rl_eval_object.py:226` inline in main (wxyz yaw; dead; 1 line)
- `scripts/rl_diagnose_policy.py:118` inline in main (wxyz yaw written into pose[3:7]; dead; 1 line)
- `scripts/contact_surface/fillet.py:111` inline in support_point (wxyz, turn by theta about (-sin phi, cos phi, 0) taking z onto the normal (checked); live; 6 lines)

Library: mjlab `quat_from_euler_xyz` (src/morphohand/rl/lifting_command.py:67 and :84). Replacement: `Rotation.from_rotvec(angle * unit(axis))`; a `quat_yaw(yaw)` helper plus `quat_mul(quat_yaw(yaw), q)` names the frame of the turn explicitly.

### 1h. Quaternion angle, distance, twist and yaw

- `scripts/curved_tips_traj_analysis.py:57` inline in compare (wxyz, 2 arccos(|<qa, qb>| / norms); live; 2 lines)
- `scripts/fe_contact_calibration_summary.py:92` inline in pair (wxyz, 2 arccos|<qa, qb>|; live; 1 line)
- `scripts/pad_builder_check.py:272` inline in divergence (wxyz, 2 arccos|<qa, qb>|; live; 1 line)
- `scripts/pad_builder_check.py:397` _rot_from (wxyz, 2 arccos|<q, q_k>| over a trace; live; 5 lines)
- `scripts/fe_contact_calibration_prior.py:264` ang (wxyz, 2 arccos|<q1, q2>|; dead; 3 lines)
- `scripts/fe_contact_calibration_prior.py:326` inline in cmd_replays (wxyz, 2 arccos|<q, q_i0>|; dead; 1 line)
- `scripts/distributed_contact.py:161` inline in static_run (wxyz, 2 arccos|w| reported as 'tilt'; live; 1 line). This is the whole rotation angle from identity, yaw included
- `scripts/contact_surface/slip.py:151` inline in run (wxyz, 2 arccos|w| reported as 'tilt'; live; 1 line). Same as above
- `src/morphohand/rl/reward.py:71` track_object_quat (wxyz, 1 - <q, q_ref>^2; live; 6 lines)
- `src/morphohand/rl/terms_reward.py:55` track_object_quat (wxyz torch, 1 - <q, q_ref>^2; live; 8 lines)
- `scripts/pads_cpu_gpu_consistency.py:157` inline in __call__ (wxyz, twist 2 atan2(v.u, w) of q * conj(q0); dead; 1 line)
- `scripts/contact_bed_roll.py:408` tool_angles (wxyz, roll = -2 atan2(q_y, q_w) of R R0^T; live; 9 lines)
- `src/morphohand/rl/reward.py:136` yaw_from_quat (wxyz, ZYX yaw atan2(2(wz + xy), 1 - 2(y^2 + z^2)); live; 4 lines)

Replacement: `(Ra.inv() * Rb).magnitude()` for the geodesic angle; a `quat_twist(q0, q1, axis)` for the signed turn about a world axis; `mat_to_euler(R, 'ZYX')[..., 0]` for yaw.

### 1i. Interpolation (slerp)

- `src/morphohand/rl/reference_trajectory.py:80` _slerp (wxyz, slerp with lerp fallback above dot 0.9995; live; 13 lines)
- `scripts/probe_real_v1_chain.py:76` _rot_lerp (matrices, body-frame geodesic via Rodrigues; live; 12 lines). No interpolation for a half turn (returns R0 until t = 1)
- `scripts/ur5_inverted_rig.py:1417` _slerp (matrices, scipy Slerp, positions lerped; live; 4 lines)
- `scripts/ur5_tip_calibration_holder_paths.py:167` slerp_T (matrices, scipy Slerp; live; 4 lines). Same three lines as the one above
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration_sim.py:259` seg_pose (4 x 4, log/exp of A^T B through the UR5 module; live; 9 lines)

Replacement: `scipy.spatial.transform.Slerp`; a `tf_interp(T0, T1, s)` for poses.

### 1j. Normalisation

- `scripts/hom_grasp_search.py:123` inline in tool_pose (wxyz; live; 1 line)
- `scripts/newton_turn.py:187` inline in build (wxyz, before reordering to xyzw; live; 1 line)
- `scripts/curved_tips_cell_drake.py:208` inline in reset (wxyz, before Drake Quaternion; live; 1 line)
- `scripts/reorient_backends.py:682` inline in reset (wxyz, before Drake Quaternion; live; 1 line)
- `scripts/pad_builder_check.py:142` inline in snap (wxyz, before mju_quat2Mat; live; 1 line)
- `scripts/real_v1_contact_gait_warp.py:90` inline in snapshot (wxyz; live; 1 line)
- `scripts/reorient_backends_gpu.py:110` inline in snap (wxyz; dead; 1 line)
- `scripts/hand_object_scale_mpc_gpu.py:154` inline in cost_records (wxyz, x and y only; live; 2 lines)

No dedicated helper exists; the copies in 1a and 1c normalise inside. Replacement: scipy normalises on `from_quat`; a `quat_normalize(q)` for code that writes qpos.

### 1k. Random rotations

- `scripts/ati_calibration_fit.py:112` inline in selftest (wxyz, normal(4) normalised (uniform on S3); live; 2 lines)

Library already in use: `Rotation.random` (scripts/fingertip_crosstalk.py:314). Replacement: `Rotation.random(n, rng)`.

### 1l. Component-order conversions (wxyz <-> xyzw) and other libraries' quaternions

- `scripts/newton_turn.py:80` quat_wxyz_to_xyzw (wxyz to xyzw; live; 2 lines)
- `scripts/rl_policy_replay.py:386` inline in replay_newton (wxyz to xyzw for Newton joint_q; live; 1 line)
- `scripts/rl_policy_replay.py:436` inline in replay_newton (xyzw (body_q) to wxyz for the MuJoCo film; live; 1 line)
- `scripts/same_state_timing.py:419` inline in run_newton (wxyz to xyzw for Newton joint_q; live; 1 line)
- `scripts/same_state_timing.py:295` inline in build_newton (MJCF wxyz to wp.quat(x, y, z, w); live; 1 line)
- `scripts/distributed_contact_gpu.py:47` inline in guide (MJWarp wxyz to wp.quat(x, y, z, w); live; 1 line)
- `scripts/contact_bed_newton.py:583` inline in render_film (recorded xyzw to MuJoCo wxyz for the film; live; 2 lines)

Drake quaternions are wxyz and are read consistently: `.ToQuaternion().wxyz()` at contact_bed_common.py:210, curved_tips_cell_drake.py:261, hom_chain.py:625, hom_contact_rig.py:868, hom_hand_drake.py:181, hom_turn3.py:226, pad_builder_check.py:134, reorient_backends_films.py:144; `[q.w(), q.x(), q.y(), q.z()]` at curved_tips_traj.py:192 and fe_contact_calibration_sim.py:314; `Quaternion(wxyz)` at curved_tips_cell_drake.py:208 and reorient_backends.py:682. trimesh's `quaternion_matrix([w, *xyz])` (wxyz) at graspgenx_eval_phase1.py:62 and graspgenx_view_grasps.py:33 (both dead). No scipy `as_quat`/`from_quat` call exists, so no scipy xyzw quaternion reaches a wxyz helper. Replacement: `wxyz_to_xyzw` / `xyzw_to_wxyz` in the shared module.

## 2. Rotation matrices


### 2a. Elementary rotations about x, y, z (helpers and inline literals)

- `scripts/ati_calibration_sim_session.py:44` rotz (Rz, degrees; live; 3 lines)
- `scripts/hom_chain.py:77` rotz (Rz, radians; live; 3 lines)
- `scripts/indenter_array.py:57` _Rz (Rz, degrees; live; 3 lines)
- `scripts/ur5_tip_calibration_holder_paths.py:62` rot_z (Rz, degrees; live; 3 lines)
- `scripts/ur5_tip_calibration_holder_apex_fit.py:61` rz (Rz, radians; live; 3 lines)
- `scripts/ur5_tip_calibration_holder_apex_fit.py:66` rx_ (Rx, radians; live; 3 lines)
- `scripts/probe_real_v1_carry.py:196` _rotx (Rx, radians; live; 3 lines)
- `scripts/probe_real_v1_chain.py:59` _rotx (Rx, radians; live; 3 lines). Also used as `pc._rotx` by real_v1_carry_construction and real_v1_deploy_envelope
- `scripts/real_v1_basketball_grasp.py:213` _rx (Rx, radians, batched; live; 5 lines)
- `scripts/real_v1_basketball_grasp.py:220` _ry (Ry, radians, batched; live; 5 lines)
- `scripts/ur5_inverted_rig.py:521` _flat_R (Ry, degrees from a table; live; 4 lines)
- `scripts/dip_link_geometry.py:77` rot_pip (Ry applied to points about the PIP centre, formula; live; 8 lines)
- `scripts/dip_link_onshape_export.py:158` pip_T (Ry 4 x 4 about the PIP centre; live; 9 lines)
- `scripts/hand_object_scale_kin.py:148` _yaw (Rx applied to (x, 0, z), formula; live; 3 lines)
- `scripts/palm_driver.py:75` fk (Rx, Ry, Rz in a dict, composed Rx Ry Rz (intrinsic XYZ); live; 10 lines)
- `scripts/ati_calibration_analysis.py:163` inline in score_truth (Rz; live; 1 line)
- `scripts/ati_stack_mount.py:225` inline in base (2-D rotation inside a 4 x 4; live; 1 line)
- `scripts/ati_stack_mount.py:391` inline in frames (Rz inside a 4 x 4; live; 1 line)
- `scripts/contact_overview_page.py:1529` inline in flat_pad (Rx; live; 1 line)
- `scripts/fingertip_calibration_figures.py:79` inline in pad (2-D rotation; live; 1 line)
- `scripts/hom_chain.py:72` inline in pinch_frame (2-D rotation formula on a vector; live; 1 line)
- `scripts/hom_contact_rig.py:706` inline in exp_cop (Rz; live; 1 line)
- `scripts/hom_contact_rig.py:708` inline in exp_cop (Ry; live; 1 line)
- `scripts/hom_code_page.py:425` inline in fig_pinpush (Rz(psi)^T, passive (the toy's psi undone); live; 1 line)
- `scripts/hom_code_page.py:606` inline in build (Rz(psi)^T, passive; live; 1 line)
- `scripts/hom_gpu.py:1217` inline in score (Rz(psi)^T, passive; live; 1 line)
- `scripts/hom_reanchor_check.py:62` inline in yaw_pitch (Rz(psi)^T, passive; live; 1 line)
- `scripts/real_v1_turn_probe.py:130` inline in feasibility (Rx; live; 1 line)
- `scripts/real_v1_turn_probe.py:162` inline in hook (Rx; live; 1 line)
- `scripts/real_v1_turn_workspace.py:66` inline in arcs (Rx; dead; 1 line)
- `scripts/ur5_cb3_cell.py:591` inline in tiphold_T_IF (Rx; live; 1 line)
- `scripts/ur5_cell_render.py:710` inline in build (Rz; live; 1 line)
- `scripts/contact_surface/geometry.py:59` inline in centers (Rz; live; 3 lines)
- `scripts/fingertip_holder.py:216` T_IF_moving (fixed frame with a turn about the vertical, written out (proper, det +1); live; 8 lines)

Library calls already in use: `trimesh.transformations.rotation_matrix` (build_cad_v1_finger.py:98, fingertip_holder.py:290, tpu_tip_design.py:155, ur5_cb3_cell.py:229 and :233), scipy `from_euler` with one axis (fingertip_crosstalk.py:51, 302, 304, 310, 386), Drake `RotationMatrix.MakeXRotation/MakeZRotation` (contact_bed_roll.py:299, hom_chain_study.py:391, hom_contact_rig.py:461 and :472, tpu_tip_rig.py:226). Replacement: `rot_x/rot_y/rot_z(angle)` in radians, as `Rotation.from_euler('x', a).as_matrix()`; the four passive copies become `rot_z(-psi)`.

### 2b. Axis-angle (Rodrigues, exp map) to matrix

- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:81` exp_so3 (rotation vector, radians; live; 7 lines). The UR5 family's shared copy (31 files call `C.*`)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:135` rot (axis and angle in DEGREES; live; 3 lines)
- `scripts/hand_object_scale_kin.py:430` rotmat (rotation vector, radians; live; 7 lines)
- `scripts/fp_sr2_toy.py:536` rot (unit axis, angles in radians, batched over angles; live; 5 lines)
- `scripts/real_v1_pusher_reach.py:39` rot (axis, angle in radians; live; 4 lines)
- `scripts/ur5_calibration_mount.py:1498` _rot (axis, angle in radians; live; 4 lines)
- `scripts/ur5_tip_calibration_suite_value.py:144` rotm (rotation vectors, batched; live; 8 lines)
- `scripts/probe_real_v1_chain.py:989` inline in _stage (unit axis, angle in radians; live; 2 lines)
- `scripts/probe_real_v1_chain.py:1034` inline in _upright (unit axis, angle in radians; live; 2 lines)

Library calls already in use: scipy `from_rotvec` (fingertip_crosstalk.py:89, 294, 335; fingertip_crosstalk_holder.py:72; fingertip_crosstalk_page.py:139). Replacement: `exp_so3(w) = Rotation.from_rotvec(w).as_matrix()`.

### 2c. Matrix to axis-angle / rotation angle (log map, trace)

- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:90` log_so3 (rotation vector; SVD projection first; half-turn branch; live; 16 lines). Correct at 180 degrees (checked)
- `scripts/rl_contact_eval.py:728` _rotvec (rotation vector, batched; no half-turn branch; live; 8 lines). Returns [0, 0, 0] for an exact half turn; the docstring limits it to few-degree steps, but `pinch_turn` also applies it to the net turn at :852
- `scripts/hand_object_scale_sim.py:429` _angle (angle of Ra^T Rb from the trace, degrees; live; 3 lines)
- `scripts/probe_real_v1_chain.py:575` inline in _rel (angle from the trace; live; 2 lines)
- `scripts/probe_real_v1_chain.py:1446` inline in chain (angle from the trace; live; 1 line)
- `scripts/probe_real_v1_chain.py:1471` inline in chain (angle from the trace; live; 1 line)
- `scripts/real_v1_arm_study.py:72` inline in droop (angle from the trace; live; 1 line)
- `scripts/real_v1_bench_study.py:117` inline in droop (angle from the trace; dead; 1 line)
- `scripts/pad_usability_step.py:775` inline in confirm (angle from the trace; dead; 1 line)
- `scripts/probe_fingertip_mechanics.py:207` inline in ramp_torque (angle from the trace; dead; 1 line)
- `scripts/ur5_tip_calibration_estimator_checks.py:105` inline in distortion (angle from the trace; live; 1 line)
- `scripts/ur5_tip_calibration_lab_page.py:155` inline in earth_field (angle from the trace; live; 1 line)
- `scripts/ur5_tip_calibration_motion_value.py:298` inline in part_shared (angle from the trace; live; 1 line)
- `scripts/ur5_tip_calibration_suite_value.py:246` inline in action_phases (angle from the trace; live; 1 line)
- `scripts/build_cad_v1_finger.py:205` inline in derive (angle from the trace; live; 1 line)
- `src/morphohand/tools/arm_ik.py:81` inline in solve (angle from the trace; live; 1 line)
- `src/morphohand/optimization/phase1_common.py:432` _cube_total_angle_drift (angle from the trace; live; 5 lines)

The arccos of the trace has an absolute error of about 1e-15/theta rad, which matters only below about 1e-6 rad; the scripts report degrees. Replacement: `log_so3(R) = Rotation.from_matrix(R).as_rotvec()` and `rot_angle(Ra, Rb) = (Ra.inv() * Rb).magnitude()`.

### 2d. Rotation taking vector a onto vector b

- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:140` rot_from_to (least rotation via exp(atan2); half turn about a perpendicular when a = -b; live; 12 lines). Correct
- `scripts/ati_calibration_plan.py:215` rot_to (I + K + K^2/(1 + c); 2pp^T - I when a = -b; live; 10 lines). Correct
- `scripts/ur5_tip_calibration_holder_paths.py:67` rot_from_to (I + K + K^2/(1 + c); returns -I when a = -b; live; 7 lines). Bug B3
- `scripts/probe_real_v1_chain.py:64` _align (Rodrigues; returns -I when a = -b; live; 10 lines). Bug B4 (no caller)
- `scripts/hom_contact_rig.py:177` _rot_z_to (+z onto v; diag(1, -1, -1) when v = -z; live; 14 lines). Correct
- `scripts/v3_tip_contact_model.py:137` rot_z_to (+z onto v; live; 14 lines). Line-for-line copy of the one above
- `scripts/build_cad_v1_finger.py:80` design_rotation (facet normal onto the nearest axis via trimesh rotation_matrix; live; 20 lines)

Library calls already in use: `mju_quatZ2Vec` (ur5_shear_impratio.py:955, ur5_tip_calibration_pads.py:142 and :164) and `trimesh.geometry.align_vectors` (ur5_calibration_mount.py, 8 calls). Replacement: `rot_between(a, b)`; scipy `Rotation.align_vectors(b[None], a[None])` returns the least rotation and a proper half turn on scipy 1.18 (checked), but its single-pair behaviour has changed across versions, so the explicit form of ur5_tip_calibration.rot_from_to is the safer body for a module that also runs on corvus's scipy 1.12.

### 2e. Frames from axes: basis completion, look-at, Gram-Schmidt

- `src/morphohand/control/hom.py:91` frame_from_normal (x given, y = axis projected, z = x cross y; fallback axis; live; 7 lines)
- `src/morphohand/control/hom.py:86` R0 (column frame from two axes; live; 3 lines)
- `src/morphohand/geometry/cad_features.py:138` perpendicular_basis (two perpendiculars to a; fallback by |a_x| < 0.9; live; 5 lines)
- `src/morphohand/geometry/cad_features.py:175` rotation_from_frames (R taking (src_a, src_b) onto (dst_a, dst_b), Gram-Schmidt; live; 8 lines)
- `scripts/hom_chain.py:89` tool_R (z given, x horizontal; fallback; live; 9 lines)
- `scripts/ur5_inverted_rig.py:1098` frames_for (batched, z = -u, x nearest a reference; fallback; live; 9 lines)
- `scripts/ur5_inverted_rig.py:1319` tangent_basis (tangent pair at a normal; fallback; live; 7 lines)
- `scripts/ur5_inverted_rig.py:295` T_AF (frame from face and end directions; live; 6 lines)
- `scripts/ur5_inverted_rig.py:136` fingertip_T_MF (frame from face and end directions; live; 6 lines). Same construction as T_AF
- `scripts/indenter_array.py:290` T_AF_press (frame from a normal and a turn; fallback; live; 13 lines)
- `scripts/fingertip_wrench_model.py:518` tangent_frames (batched tangent frames; fallback by |n.z| > 0.9; live; 8 lines)
- `scripts/fingertip_wrench_model.py:121` chip_rotation (chip axes as rows; live; 7 lines)
- `scripts/ur5_shear_skin.py:79` frame (normal plus projected direction, no fallback; live; 6 lines)
- `scripts/probe_real_v1_chain.py:1180` _tool_frame (tool axis plus projected x, fallback; live; 8 lines)
- `scripts/ur5_cell_render.py:296` lumber_frame (frame from a mean horizontal direction; live; 12 lines)
- `scripts/ur5_cell_render.py:607` look_at (look-at for an MJCF camera; live; 7 lines). No fallback when looking straight down
- `scripts/fingertip_calibration_cell_render.py:87` project (look-at projection; live; 10 lines). No fallback when looking straight down
- `scripts/dip_link_concept_a.py:1187` _project (camera from azimuth and elevation; live; 12 lines)
- `scripts/ur5_calibration_mount.py:2082` _project (camera from azimuth and elevation; live; 11 lines). Copy of dip_link_concept_a._project
- `scripts/hand_object_scale_sim.py:568` inline in gcf (contact frame from normal and flexion axis; fallback; live; 6 lines)
- `scripts/hom_control.py:92` inline in gcf_sphere_cylinder (contact frame from normal and flexion axis; fallback; live; 8 lines)
- `scripts/hom_turn3.py:118` inline in gcf (contact frame from normal and flexion axis; fallback; live; 6 lines)
- `scripts/hom_hand_brake.py:256` inline in mj_scene (tool frame from its axis (re-orthogonalised on the next line); live; 2 lines)
- `scripts/hom_paper_tasks.py:303` inline in exp2 (frame from two axes, both flipped to keep e2 up; live; 6 lines)
- `scripts/pad_contact_rig.py:102` inline in press (column frame, y = x cross z of the rig; live; 1 line)
- `scripts/build_cad_v1_finger.py:391` inline in derive (frame from x and z; live; 1 line)
- `scripts/dip_b_v4.py:307` inline in engrave (engraving frame from a normal; live; 4 lines)
- `scripts/dip_link_concept_a.py:210` inline in tube_mesh (tube frame; fallback by |t_x| < 0.9; live; 8 lines)
- `scripts/ur5_calibration_mount.py:782` inline in key_channel (key frame, det asserted; live; 7 lines)
- `scripts/ur5_inverted_rig.py:283` inline in table_T_BA (frame from two measured directions; live; 1 line)
- `scripts/ur5_inverted_rig.py:1274` inline in search_rod (frame from y and z; live; 1 line)
- `scripts/hom_grasp_search.py:177` inline in forces_ok (tangent with fallback; live; 1 line)
- `scripts/ati_calibration_plan.py:108` inline in hanging_points (perpendiculars to gravity with fallback; live; 4 lines)
- `scripts/v3_tip_contact_model_figures.py:247` inline in film (tangents to a normal, no fallback; live; 3 lines)
- `scripts/reorient_primitive.py:750` inline in feasibility (frame from the shaft axis with fallback; dead; 6 lines)
- `src/morphohand/optimization/force_closure.py:118` inline in _build_friction_cone_wrenches (friction-cone tangents with fallback; live; 4 lines)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration_sim.py:205` inline in __init__ (batched tangent bases; live; 3 lines)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:589` inline in fit_rod_registration (perpendicular with fallback; live; 1 line)

All built frames that were checked are right-handed (z = x cross y or an equivalent order). Replacement: `frame_from_axis(axis, hint, which='z')`, `frame_from_two(primary, secondary, order)`, `tangent_basis(n)` and `look_at(eye, target, up)` with one fallback rule.

### 2f. Skew (hat) and vee

- `src/morphohand/control/hom.py:107` skew (skew; live; 2 lines)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:77` skew (skew; live; 2 lines)
- `scripts/ati_stack_mount.py:480` _skew (skew; live; 2 lines)
- `scripts/fp_sr2_toy.py:538` inline in rot (skew literal; live; 1 line)
- `scripts/fp_sr2_toy.py:766` inline in grasp_matrix_smin (skew literal (grasp matrix); live; 1 line)
- `scripts/hand_object_scale_kin.py:435` inline in rotmat (skew literal; live; 1 line)
- `scripts/hom_contact_rig.py:189` inline in _rot_z_to (skew literal; live; 1 line)
- `scripts/hom_grasp_search.py:175` inline in forces_ok (skew literal (grasp matrix); live; 1 line)
- `scripts/probe_real_v1_chain.py:72` inline in _align (skew literal; live; 1 line)
- `scripts/probe_real_v1_chain.py:86` inline in _rot_lerp (skew literal; live; 1 line)
- `scripts/probe_real_v1_chain.py:989` inline in _stage (skew literal; live; 1 line)
- `scripts/probe_real_v1_chain.py:1034` inline in _upright (skew literal; live; 1 line)
- `scripts/real_v1_pusher_reach.py:41` inline in rot (skew literal; live; 1 line)
- `scripts/ur5_calibration_mount.py:1500` inline in _rot (skew literal; live; 1 line)
- `scripts/ur5_tip_calibration_holder_paths.py:72` inline in rot_from_to (skew literal; live; 1 line)
- `scripts/v3_tip_contact_model.py:149` inline in rot_z_to (skew literal; live; 1 line)
- `scripts/ur5_tip_calibration_suite_value.py:148` inline in rotm (skew written element by element, batched; live; 2 lines)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:97` inline in log_so3 (vee of R - R^T; live; 1 line)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:365` inline in fit_gravity (vee of C - C^T; live; 1 line)
- `scripts/rl_contact_eval.py:732` inline in _rotvec (vee, batched; live; 1 line)
- `scripts/probe_real_v1_chain.py:84` inline in _rot_lerp (vee; live; 1 line)

Neither scipy nor MuJoCo exports a skew matrix; keep `skew(v)` and `vee(S)` (batched) in the shared module.

### 2g. Projection onto SO(3) (nearest rotation by SVD)

- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:91` inline in log_so3 (U diag(1, 1, det) V^T inside log_so3; live; 2 lines)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:357` inline in fit_gravity (polar factor, all axes flipped when det < 0 (intended sign fit); live; 4 lines)
- `scripts/ur5_inverted_rig.py:953` inline in context (U V^T without a det check; live; 2 lines)
- `scripts/ur5_inverted_rig.py:969` inline in context (U V^T without a det check; live; 2 lines)
- `scripts/ur5_tip_calibration_lab_page.py:133` procrustes (U diag(1, 1, det) V^T; live; 3 lines)
- `scripts/ur5_tip_calibration_lab_page.py:841` procrustes (U diag(1, 1, det) V^T; live; 3 lines). Identical nested copy in the same file

Replacement: `project_to_so3(M)`; `Rotation.from_matrix` also returns the nearest rotation of a near-orthogonal matrix.

## 3. Euler angles, roll-pitch-yaw and rotation vectors


### 3a. Euler and RPY conversions

- `scripts/palm_driver.py:29` _euler_xyz (matrix to intrinsic X-Y-Z (R = Rx(a) Ry(b) Rz(c)), gantry hinges; live; 15 lines). Agrees with scipy 'XYZ' to 7e-16 (checked)
- `scripts/palm_driver.py:75` fk (intrinsic X-Y-Z composition (the inverse of the line above); live; 10 lines)
- `scripts/ur5_cb3_cell.py:68` rpy_R (URDF rpy: Rz(y) Ry(p) Rx(r) = extrinsic x-y-z; live; 2 lines). Agrees with scipy 'xyz' to 4e-16 (checked); also used by ur5_cell_render.urdf_visuals
- `scripts/ur5_inverted_rig.py:2269` inline in cmd_dry_check (scipy from_euler('xyz', deg), extrinsic, for --fit; live; 1 line). The only Euler triple typed by hand; its producer's order is not recorded in the code
- `src/morphohand/rl/reward.py:136` yaw_from_quat (wxyz quaternion to ZYX yaw; live; 4 lines)

Library calls already in use: mjlab `quat_from_euler_xyz` (lifting_command.py:67 and :84). MJCF `euler=` strings (contact_surface/mujoco_model.py:54, hom_chain_study.py:357) turn about one axis only, so the compiler's eulerseq does not matter. Replacement: `euler_to_mat(angles, seq)` and `mat_to_euler(R, seq)` with scipy's seq strings ('xyz' extrinsic = URDF rpy, 'XYZ' intrinsic = the palm gantry), plus `rpy_to_mat(r, p, y)`.

### 3b. Yaw read from a matrix, atan2(R[1,0], R[0,0])

- `src/morphohand/optimization/phase1_common.py:421` _cube_yaw (body x heading; live; 3 lines)
- `scripts/hom_paper_tasks.py:510` theta (heading; live; 3 lines)
- `scripts/ur5_cell_render.py:692` inline in build (heading in degrees; live; 1 line)
- `scripts/ur5_fixture_from_seating.py:56` inline in main (heading in degrees; live; 1 line)
- `scripts/ur5_probe_plan.py:315` inline in build (heading in degrees; live; 1 line)
- `scripts/ur5_tip_calibration_check.py:156` inline in homing (heading in degrees; live; 1 line)
- `scripts/ur5_tip_calibration_holder_apex_fit.py:81` inline in frame_one (heading; live; 1 line)
- `scripts/ur5_tip_calibration_holder_apex_fit.py:104` inline in frame_two (heading as a start value; live; 1 line)
- `scripts/ur5_tip_calibration_holder_apex_fit.py:160` inline in main (heading in degrees; live; 1 line)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:369` inline in fit_gravity (heading in degrees; live; 1 line)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:1778` inline in check_placement (heading in degrees; live; 1 line)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:1782` inline in check_placement (heading in degrees; live; 1 line)

These are headings of the x axis, not the yaw of a ZYX decomposition; they agree when the frame is level. Replacement: a named `heading(R)`.

### 3c. Rotation vectors (UR TCP poses and fits)

- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:126` ur_to_T (UR pose [m, rotation vector] to 4 x 4 in mm; live; 3 lines)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:131` T_to_ur (4 x 4 in mm to UR pose; live; 2 lines)
- `scripts/hand_object_scale_kin.py:422` rotvec_grid (grid of rotation vectors, radians; live; 6 lines)
- `scripts/rl_contact_eval.py:728` _rotvec (matrix to rotation vector (also in 2c); live; 8 lines)

Library calls already in use: scipy `from_rotvec`/`as_rotvec` in fingertip_crosstalk.py (lines 89, 294, 308, 311, 314, 335), fingertip_crosstalk_holder.py:72 and fingertip_crosstalk_page.py:139. Replacement: `tf_from_ur(pose, scale=1e3)` and `tf_to_ur(T, scale=1e-3)` over `exp_so3`/`log_so3`.

## 4. Homogeneous transforms and poses


### 4a. 4 x 4 build, compose, invert, apply

- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:108` Tm (Tm(R, t); live; 7 lines). De-facto shared for the UR5 family
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:117` inv (rigid inverse R^T, -R^T t; live; 3 lines)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:122` apply (apply to points, P R^T + t; live; 2 lines)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:154` perturb (T exp(d), right-multiplied (own frame); live; 3 lines)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:159` turn_about (turn about a world line; live; 4 lines)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:165` pose_error ((mm, deg) between two poses; live; 3 lines)
- `src/morphohand/geometry/cad_features.py:185` transform (transform(R, t) to 4 x 4; live; 5 lines). Imported as `cf.transform` by build_cad_v1_finger
- `scripts/fingertip_crosstalk.py:84` inverse ((R, t) inverse; live; 2 lines)
- `scripts/fingertip_crosstalk.py:88` perturb (R <- exp(dr) R, t <- t + dt, left-multiplied (world frame); live; 2 lines). Same name as the UR5 module's `perturb`, opposite frame
- `scripts/dip_link_onshape_export.py:60` T_D_from_F (rigid inverse with rounding of R; live; 9 lines)
- `scripts/dip_link_concept_a.py:312` _axes_T (axis permutation as 3 x 4; live; 6 lines)
- `scripts/ur5_calibration_mount.py:400` _axis_T (axis permutation as 4 x 4 (cyclic, det +1); live; 7 lines)
- `scripts/ur5_inverted_rig.py:1109` pose_B (batched compose T_BA T inv(T_EM); live; 7 lines)
- `scripts/ur5_cb3_cell.py:101` link_frames (forward kinematics as a chain of 4 x 4 with exp_so3; live; 22 lines)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration_sim.py:58` fk (forward kinematics (DH) as a chain of 4 x 4; live; 12 lines)

Inline `np.eye(4)` builds: 40 sites in 20 files (ati_stack_mount 4, graspgenx_make_morphohand 4 (dead), ur5_calibration_mount 4, ur5_cb3_cell 4, indenter_array 3, ur5_inverted_rig 3, build_cad_v1_finger 2, dip_link_onshape_export 2, ur5_cell_render 2, ur5_tip_calibration_holder_apex_fit 2, and one each in dip_b_v4, dip_link_params, eflesh_build, fingertip_holder, graspgenx_eval_phase1, metric_magnet_prints, ur5_tip_calibration_plan, cad_features, ur5_tip_calibration_sim, ur5_tip_calibration). General `np.linalg.inv` on rigid 4 x 4 transforms where the closed form is exact: 29 sites in 6 files (fingertip_crosstalk_holder.py 14: lines 134, 152, 257, 258, 274, 276, 282, 283, 310, 332, 333, 345, 348, 349; ur5_calibration_mount.py 8: 791, 1372, 1373, 1527, 1572, 1573, 1588, 3084; fingertip_holder.py 140, 332, 345; build_cad_v1_finger.py 261, 393; ur5_fixture_from_seating.py:55; ur5_tip_calibration_trimmed_cap_reach.py:148). trimesh `apply_transform`/`transform_points` and `transformations.translation_matrix` are library calls. Replacement: `tf`, `tf_inv`, `tf_apply`, `tf_compose`; `perturb_body(T, d)` and `perturb_world(T, d)` as two names.

### 4b. Poses as (p, q) or (p, R): compose, invert, relative pose

- `src/morphohand/rl/terms_obs.py:27` object_pose_rel_palm (wxyz torch: conj(q_palm) * q_obj, rotate(conj(q_palm), p - p_palm); live; 34 lines)
- `scripts/policy_cpu_rollout.py:164` obs (wxyz numpy: the same relative pose as terms_obs; live; 24 lines). Two copies in this file (also :319)
- `scripts/policy_cpu_rollout.py:319` obs (wxyz numpy: the same relative pose; live; 17 lines). Second copy; it omits the q -> -q sign fix of :179
- `src/morphohand/rl/imitation.py:29` fingertips_in_object_frame (wxyz torch: points into the object frame; live; 15 lines)
- `scripts/probe_real_v1_chain.py:90` _rigid_palm_pose ((R, p): palm pose that carries a rigid object to a goal; live; 13 lines)
- `src/morphohand/control/hom.py:254` pinch_frame ((R, p): pinch frame expressed in the object frame; live; 7 lines)
- `src/morphohand/control/hom.py:111` adjoint (6 x 6 adjoint of (R, p), twists ordered (w, v); live; 7 lines)

Replacement: `pose_mul`, `pose_inv`, `pose_rel(a, b)` on (p, q wxyz), `tf_from_pose`/`tf_to_pose`, and `adjoint(R, p)`; MuJoCo's `mju_mulPose`, `mju_negPose` and `mju_trnVecPose` cover single poses where mujoco is imported.

### 4c. Inline world <-> local frame changes

- `src/morphohand/control/hom.py:208` inline in _obj_normal (R^T (p - o); live; 1 line)
- `src/morphohand/optimization/phase1_common.py:444` inline in _tip_distances_to_cube ((p - o) @ R and back; live; 3 lines)
- `src/morphohand/optimization/phase1_strategy_diffmjx.py:94` inline in _soft_tip_distances ((p - o) @ R and back; live; 3 lines). Copy of phase1_common
- `src/morphohand/optimization/phase1_strategy_mjx_autodiff.py:98` inline in _surrogate_score ((p - o) @ R and back; live; 3 lines). Copy of phase1_common
- `scripts/cad_v1_interference.py:75` inline in overlap (P R^T + p and (X - p) @ R; live; 4 lines)
- `scripts/pad_contact_rig.py:129` inline in contacts (R^T (p - o); live; 1 line)
- `scripts/contact_map_render.py:89` inline in rollout (R^T (p - o); dead; 1 line)

A pattern search finds at least 47 such sites in 30 files (`R.T @ (p - o)`, `(p - o) @ R`, `P @ R.T + t`); `.T @ (` alone occurs 66 times in 33 files and `xmat...reshape(3, 3)` 97 times in 48 files, so the count is a floor. MuJoCo contact-frame rotations `frame.reshape(3, 3).T @ f[:3]` appear at distributed_contact_gpu.py:162, distributed_contact_newton.py:334, rl_contact_sensor_check.py:146 and ur5_shear_skin.py:178. Replacement: `to_local(R, p, x)` and `to_world(R, p, x)`; these are one-liners and need no library.

## 5. Rigid registration, fitting and rotation averaging


### 5a. Kabsch, Umeyama, Procrustes

- `scripts/build_cad_v1_finger.py:102` kabsch (Kabsch, det-corrected, returns 4 x 4; live; 6 lines)
- `scripts/build_cad_v1_finger.py:110` register_copy (PCA start plus vertex ICP with kabsch; live; 19 lines)
- `scripts/ur5_tip_calibration_estimator_checks.py:96` inline in distortion (Umeyama similarity, det-corrected; live; 6 lines)
- `scripts/ur5_tip_calibration_lab_page.py:133` procrustes (nearest rotation of a fitted 3 x 3 (Procrustes); live; 3 lines)
- `scripts/ur5_tip_calibration_lab_page.py:841` procrustes (nearest rotation; live; 3 lines). Identical copy in the same file

Replacement: `kabsch(A, B, w=None) -> (R, t)` (or `Rotation.align_vectors(B - cb, A - ca, weights)`, whose argument order is target first) and `umeyama(A, B) -> (s, R, t)`.

### 5b. Nonlinear pose fits (local exp-map perturbations)

- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:452` fit_registration (T_EM, T_MP, sphere centre by least squares over perturb(); live; 54 lines)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:521` fit_probe_registration (probe registration, same parameterisation; live; 53 lines)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:576` fit_rod_registration (rod registration, same parameterisation; live; 39 lines)
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:337` fit_gravity (gravity fit: polar factor of a least-squares 3 x 3; live; 40 lines)
- `scripts/ur5_tip_calibration_holder_apex_fit.py:88` frame_two (least squares over rz(a) rx(b); live; 21 lines)
- `scripts/fingertip_crosstalk.py:275` sharpie (least squares over rotation vectors (scipy), random starts; live; 87 lines)

These use the shared exp map correctly; they need only `exp_so3`/`perturb_body` from the new module.

### 5c. Rotation averaging

- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:170` mean_pose (one log/exp step about the first rotation; live; 5 lines)
- `scripts/ur5_inverted_rig.py:278` inline in table_T_BA (chordal mean, U V^T of the sum, no det check; live; 2 lines)

Replacement: `Rotation.mean(weights)` (chordal L2 mean).

## Confirmed bugs and inconsistencies

Each was confirmed by reading the code and by a numeric check (scripts `numcheck*.py` in the scratchpad; scipy 1.18.1 in
the femtip environment as the reference).

**B1. Yaw jitter about the body axis.** `scripts/plant_drop_gate.py:79-86` (`run_trial`, live by the import graph; also
run by `plant_ranking_test.py:79`, dead) multiplies the spawn quaternion as q * dq, so the turn is about the body's z
axis. The plans' spawn quaternion (`meta.replay_initial_qpos[3:7]` = (0.70711, 0.70711, 0, 0) in
`docs/experiments/20260902-residual-bench/deploy/rv05_manual_b85_plan.json` and `sv1_w6689_b060_plan.json`) lays the
shaft's body z along world -y, so `--jitter-yaw` (default 0.09 rad, 1 sigma) spins the screwdriver about its own shaft
and leaves the shaft's heading at -90.000 deg for every draw. The world-frame product dq * q, which
`reorient_backends._yaw_quat`, `hom_tasks.Plant.reset`, `hom_gpu`, `real_v1_deploy_envelope` and `real_v1_chain_hands`
use, turns the heading to -84.843 deg for a one-sigma draw. Fix: `quat_mul(quat_yaw(a), q)`.

**B2. Matrix-to-quaternion sign loss at half turns.** `scripts/ur5_tip_calibration_film.py:38-44` (`quat`, live) takes
each vector component as `copysign(sqrt(...), R[k,j] - R[j,k])`. For a half turn the antisymmetric part is zero, so the
relative signs of x, y, z are lost. Input: a flange pointing straight down with yaw -60 deg,
R = Rz(-60 deg) diag(1, -1, -1); output (0, 0.866, 0.5, 0), the quaternion of yaw +60 deg (120 deg error). Over 2000
random half turns, 32 % of float matrices and 75 % of 7-decimal-rounded matrices come out wrong (errors up to 153 and
180 deg); at 179.99 deg the error falls to 0.015 deg. The function sets `mocap_quat` of every robot link and the tool
(lines 114 and 119). `ur5_cell_render.py:529`, also named `quat` with the same signature, uses `mju_mat2Quat` and is
right. Fix: `mat_to_quat` (scipy `from_matrix`) or `mju_mat2Quat`.

**B3. A reflection returned as a rotation.** `scripts/ur5_tip_calibration_holder_paths.py:70-71` (`rot_from_to`, live)
returns `-np.eye(3)` when a = -b; its determinant is -1. At line 91 it is called as `rot_from_to(n_P, [1, 0, 0])`, so a
pad normal n_P = (-1, 0, 0) gives an improper T_IP and an improper flange target T_BE for the IK. The holder plans aim
at the face (n_P near +x), so the case is latent. The UR5 module's function of the same name
(`ur5_tip_calibration.py:140`) and `ati_calibration_plan.rot_to` return a half turn about a perpendicular there
(det +1, checked). Fix: use the module's `rot_from_to`.

**B4. Same reflection, no caller.** `scripts/probe_real_v1_chain.py:70-71` (`_align`) returns `-np.eye(3)` for
antiparallel inputs (det -1, checked); nothing in the repository calls it.

**Inconsistencies that are not wrong today.**
- Same names, different units: `rot(axis, deg)` takes degrees in `ur5_tip_calibration.py:135`; `rot(u, ang)` in
  `fp_sr2_toy.py:536`, `rot(axis, a)` in `real_v1_pusher_reach.py:39` and `_rot` in `ur5_calibration_mount.py:1498` take
  radians; `rot(q)` in `hom_grasp_search.py:75` converts a quaternion. `rotz(deg)` (`ati_calibration_sim_session.py:44`),
  `_Rz(deg)` (`indenter_array.py:57`) and `rot_z(deg)` (`ur5_tip_calibration_holder_paths.py:62`) take degrees;
  `rotz(psi)` (`hom_chain.py:77`) and `rz(a)` (`ur5_tip_calibration_holder_apex_fit.py:61`) take radians.
- Same name, opposite frames: `perturb` right-multiplies in the pose's own frame in `ur5_tip_calibration.py:154` and
  left-multiplies in the world frame in `fingertip_crosstalk.py:88`.
- 'tilt' means two quantities: 2 arccos|w| (the whole rotation from identity, yaw included) in `distributed_contact.py:161`
  and `contact_surface/slip.py:151`; arccos of |R[2,2]| or of R[2,2] elsewhere (e.g. `probe_real_v1_gait.py:180`). The sign also
  differs: `probe_real_v1_pivot.py:75` (dead) reads |R[2,2]|, `probe_real_v1_carry.py:267` reads R[2,2].
- Passive copies: `hom_code_page.py:425` and `:606`, `hom_gpu.py:1217` and `hom_reanchor_check.py:62` write
  [[c, s, 0], [-s, c, 0], [0, 0, 1]] = Rz(psi)^T on purpose ("the toy's psi undone"); a shared `rot_z` needs `-psi` there.
- No normalisation before quat-to-matrix in `hom.quat_to_mat`, `hom_grasp_search.rot`, `hom_gpu._quat_mat`, the Newton
  inline copy and `hand_object_scale_mpc.py:169` (its GPU twin normalises). Their inputs come from simulators that keep
  |q| = 1 to float precision, so no error was observed.
- Numerical limits: `rl_contact_eval._rotvec` returns [0, 0, 0] for an exact half turn; `probe_real_v1_chain._rot_lerp`
  does not interpolate a half turn; the four look-at helpers in 2e divide by |forward x up| with no fallback (NaN for a
  camera looking straight down; none of the current camera tables does).
- `ur5_inverted_rig.py:2269`: `--fit` angles are read as scipy 'xyz' (extrinsic about x_P, y_P, z_P). For the few-degree
  registration offsets the order changes the result by the product of the angles (about 0.07 deg at 2 deg x 2 deg).
- `same_state_timing.py:243-246` (`strip_meshes`) carries only `pos` and `quat` of a mesh geom into Newton; a mesh geom
  oriented with `euler`, `axisangle`, `xyaxes` or `zaxis` would lose its turn. No such geom exists under `assets/mjcf`.

**Checked and correct (maximum error against scipy).** Quaternion to matrix: hom.quat_to_mat, hom_grasp_search.rot,
fe_contact_calibration_sim.quat_R, rl_contact_eval._quat_mat, reorient_primitive._quat_to_mat (<= 8e-16). Products:
hom_gpu.quat_mul, hom_grasp_search.quat_mul, policy_cpu_rollout.quat_mul, real_v1_chain_hands._qmul,
reorient_backends._yaw_quat (<= 3e-16). Vector rotation: hom_tasks_mpc.quat_rotate, policy_cpu_rollout.quat_rotate
(1.6e-15). newton_scaling.twist_deg_xyzw (3e-14 deg). Exp map: ur5_tip_calibration.exp_so3, hand_object_scale_kin.rotmat,
fp_sr2_toy.rot, real_v1_pusher_reach.rot, ur5_calibration_mount._rot, ur5_tip_calibration_suite_value.rotm (<= 9e-16).
ur5_tip_calibration.log_so3 (5e-12, right at 180 deg). reference_trajectory._slerp (3e-16). Matrix to quaternion:
contact_bed_roll.mat2quat (3e-16), build_screw_scene._quat_from_cols (5e-10). The a-to-b rotations away from a = -b
(<= 7e-14). palm_driver's Euler formula against scipy 'XYZ' (7e-16) and ur5_cb3_cell.rpy_R against scipy 'xyz' (4e-16).


## Recommended shared module

One file, `src/morphohand/geometry/rotations.py`, importing numpy and scipy only. Constraints found in the audit:

- The femtip environment (scipy 1.18.1, numpy 2.5.3) has no mujoco and runs `ur5_inverted_rig.py` and the other ur5_*
  scripts, so nothing may import mujoco at module import.
- The lab computer corvus runs `/usr/bin/python3` 3.10 with numpy 1.26.4 and scipy 1.12.0
  (`docs/experiments/20261022-ur5_tip_calibration/lab/README.md`, line 40). scipy 1.12 has no `scalar_first`, so the
  module converts wxyz <-> xyzw by indexing (`q[..., [1, 2, 3, 0]]`), which works on every installed scipy (1.12 to 1.18).
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py` is shipped to corvus as a standalone file and already acts
  as the UR5 family's transform module (31 files call `C.Tm`, `C.inv`, `C.apply`, `C.exp_so3`, `C.log_so3` and others).
  Keep the new module a single file with no package-relative imports so it can sit beside it, and have
  `ur5_tip_calibration.py` re-export its names.
- Torch versions stay in `src/morphohand/rl/math.py` (which already holds `quat_mul`, `quat_rotate`,
  `quat_rotate_inv`), with the same names and the same wxyz order.
- Cost: scipy's single-quaternion `from_quat(...).as_matrix()` takes 4.5 us against 1.6 us hand-written; a batch of
  10,000 takes 150 us.

Conventions fixed in the module: quaternions wxyz, Hamilton product, unit length, w >= 0 on output; active rotations;
angles in radians (callers convert degrees); arrays batched over leading axes; scipy Euler sequence strings.

    # quaternions (wxyz)
    to_scipy(q) -> Rotation;  from_scipy(r) -> q          # the only place the component order changes
    wxyz_to_xyzw(q);  xyzw_to_wxyz(q)                     # Newton, warp
    quat_normalize(q);  quat_canonical(q)                 # w >= 0
    quat_to_mat(q);  mat_to_quat(R)                        # from_matrix: exact at half turns
    quat_mul(a, b);  quat_conj(q)
    quat_apply(q, v);  quat_apply_inv(q, v)
    quat_from_axis_angle(axis, angle);  quat_from_rotvec(w);  quat_to_rotvec(q)
    quat_yaw(yaw)                                          # turn about world z; compose as quat_mul(quat_yaw(a), q)
    quat_axis(q, k=2);  tilt_cos(q)                        # column k of R; R[2, 2]
    quat_angle(a, b)                                       # geodesic angle, (ra.inv() * rb).magnitude()
    quat_twist(q0, q1, axis)                               # signed turn of q1 relative to q0 about a world axis
    quat_slerp(a, b, t);  quat_random(n, rng)
    # matrices
    rot_x(a);  rot_y(a);  rot_z(a)
    exp_so3(w);  log_so3(R);  rot_axis_angle(axis, angle)
    rot_angle(Ra, Rb);  heading(R)
    rot_between(a, b)                                      # least rotation, half turn when a = -b
    frame_from_axis(axis, hint=(1, 0, 0), which="z");  frame_from_two(primary, secondary, order="zx")
    tangent_basis(n);  look_at(eye, target, up=(0, 0, 1))
    skew(v);  vee(S);  project_to_so3(M)
    euler_to_mat(angles, seq);  mat_to_euler(R, seq);  rpy_to_mat(r, p, y)   # 'xyz' = URDF rpy, 'XYZ' = palm gantry
    # rigid transforms
    tf(R=None, t=None);  tf_inv(T);  tf_apply(T, P);  tf_compose(*Ts)
    tf_from_pose(p, q);  tf_to_pose(T)
    tf_from_ur(pose, scale=1e3);  tf_to_ur(T, scale=1e-3)  # UR rotation vectors
    tf_interp(T0, T1, s)
    perturb_body(T, d);  perturb_world(T, d);  turn_about(T, c, axis, angle)
    pose_mul(pa, pb);  pose_inv(p);  pose_rel(pa, pb)      # (pos, wxyz) tuples
    to_local(R, p, x);  to_world(R, p, x);  adjoint(R, p)
    # registration
    kabsch(A, B, w=None) -> (R, t);  umeyama(A, B) -> (s, R, t);  mean_rotation(Rs, w=None)

