# Audit of hand-written geometry and numerical helpers, MorphoHand repository, 2026-11-03

This audit lists 165 hand-written copies by file and line (143 live, 22 dead), grouped into about 30 families, plus
inline idioms counted by pattern. Most families have a library call (scipy, trimesh) or one existing module as their
replacement. Two segment-segment distances and seven Spearman copies give wrong numbers on inputs that are easy to
produce. Three function names (`tangential`, `line_distance`, `fib_cap`) each mean two or three different things.
The largest families are bisection loops (30), moving averages (14), Spearman correlations (14), Fibonacci lattices
(10) and the cap-pad stiffness formula (7 sites).

Files read: 773 Python files, the 652 files at the top level of `scripts/`, 10 in `scripts/contact_surface/`, 3 in
`scripts/cluster/` and 108 under `src/` (without `driver/manta/firmware`, the manta `.venv` and `scripts/archive/`).
Every function, nested functions and methods included, was indexed with Python's `ast`, and the source text was
searched for about 60 content patterns. Each family below was then read by hand. Liveness comes from the parent's
`live.json`; `src/` and `scripts/contact_surface/` count as live (the package is imported by the live
`reorient_backends.py`). The three cluster scripts had no findings. No mesh, voxel grid or simulation was loaded. Four
small numpy checks in this folder back the bugs: `check_segseg.py`, `check_spearman.py`, `check_sdf.py` and
`check_plant_rho.py`, the last run under `resguard.sh run --mem 1G` (117 MB, 2.2 s). The scanners are `geoscan.py`
and `funcs.py`.

Each entry reads `file:line function (live|dead; N lines; notes)`. Installed versions in `.venv`: numpy 2.4.4,
scipy 1.17.1, trimesh 4.11.5, mujoco 3.6.0; shapely only in the femtip environment.

## 1. Vector basics

### 1.1 Normalising helpers (5 named, all live; 165 inline lines in 93 files)

- `scripts/build_cad_v1_finger.py:70 unit` (live; 3 lines; no zero guard, NaN on a zero vector)
- `scripts/hom_paper_tasks.py:57 unit` (live; 2 lines; no zero guard)
- `src/morphohand/control/surfaces.py:20 unit` (live; 3 lines; returns the input unchanged below 1e-15)
- `src/morphohand/bench/tags.py:339 _unit` (live; 6 lines; raises ValueError below 1e-12)
- `src/morphohand/optimization/force_closure.py:47 ContactWrench.__post_init__` (live; 7 lines; raises below 1e-12)
- Inline: 165 lines in 93 files divide by `np.linalg.norm(...)`; 27 of them have an eps, `max()` or `clip()` guard on
  the same line, and 34 lines in 20 files normalise rows with `keepdims=True`. Torch code uses
  `torch.nn.functional.normalize` (`hom_gpu.py:910`) or `+ 1e-9` (`rl_eval_object.py:136`, dead).

Replacement: `trimesh.util.unitize(v, check_valid=True)` for arrays whose zero rows must be found (it returns a
validity mask), or a shared `vec.unit(v, eps=1e-12, zero="raise"|"keep"|"nan")` so that the zero policy is chosen
at the call. The four named copies use three policies today.

### 1.2 Angles between vectors and angles from cosines

- `scripts/ur5_tip_calibration_cb1r_checks.py:67 angle` (live; 3 lines; rows of two arrays, arccos of the clipped
  normalised dot with a 1e-12 guard; reused as `K.angle` by `ur5_tip_calibration_hard_extrapolation.py` and
  `ur5_tip_calibration_rod_tips.py`)
- `src/morphohand/bench/tags.py:139 deg_from_up` (live; 7 lines; clipped acos; its docstring forbids folding with abs)
- `src/morphohand/bench/tags.py:244 BenchFrame.drift_deg` (live; 4 lines; inline clipped acos)
- `src/morphohand/optimization/phase1_common.py:425 _cube_axis_tilt` (live; 6 lines; clipped)
- `src/morphohand/rl/reward.py:144 pen_cube_axis_tilt` (live; 6 lines; clipped)
- `src/morphohand/rl/terms_obs.py:63 target_axis_misalignment` (live; 10 lines; torch acos of
  `terms_common._alignment_cos`, which clamps at line 144)
- `scripts/probe_real_v1_gait.py:180 _axis_tilt` (live; 4 lines; folds with `abs(ax[2])`)
- `scripts/hom_paper_tasks.py:523 wield.tilt` (live; 2 lines; folds with `abs`)
- `scripts/probe_real_v1_chain.py:373 chain._tilt` (live; 2 lines; signed)
- Degrees from a recorded cosine: `scripts/d6_cal_60m_page.py:111 deg` (live; 2), `scripts/hands_tranche_page.py:66
  deg` (live; 2), `scripts/tip_gait_data.py:24 deg` (dead; 2), `scripts/calibrated_plant_chain_data.py:44 deg` (dead;
  3; 90 minus the angle), `scripts/real_v1_plant_table.py:35 deg` (dead; 3; 90 minus), `scripts/turn_mechanism_data.py:50
  turn_deg` (dead; 3; 90 minus), `scripts/hom_turn_rl_contact_page.py:1071 films_paragraph.turn` (live; 2; 90 minus)
- Turn between two cosines: `scripts/newton_turn.py:340 run.turn` (live; 2), `scripts/reorient_backends.py:434
  score.turn` (live; 2), `scripts/diffmjx_audit_page.py:142 tilt_deg` (live; 2; no clip, `math.acos` raises above 1)
- Inline: 135 `arccos`/`acos` calls in 81 files; 106 clip on the same line. The other 29 were read one by one: each is
  clipped a line earlier or receives a value inside [-1, 1] (Fibonacci `1 - 2k/n`, a mean of cosines, `min(1, ...)`).
  No code computes `arctan2(|a x b|, a . b)`.

Replacement: one `vec.angle(a, b)` as `arctan2(norm(cross(a, b)), dot(a, b))`, which needs no clip and keeps its
accuracy near 0 and 180 degrees, plus `vec.deg_from_cos(c)` for recorded cosines. `trimesh.geometry.vector_angle`
accepts unit pairs only. Rotation angles from matrices or quaternions are left to the rotation audit.

### 1.3 Projection onto a plane, tangential components (9 copies, 8 live)

- `scripts/build_cad_v1_finger.py:75 perp` (live; 3 lines; vector; assumes a unit axis)
- `scripts/fingertip_crosstalk.py:112 tangential` (live; 2 lines; returns the vector `v - (v.n) n`)
- `scripts/ur5_tip_calibration_analysis.py:364 tangential` (live; 2 lines; same body as the previous)
- `scripts/fingertip_magnet_transfer_sim.py:69 tangential` (live; 3 lines; arguments `(P, Pt, N)`, returns the norm)
- `scripts/ur5_tip_calibration_estimator_checks.py:44 tangential` (live; 3 lines; same as the previous)
- `scripts/ur5_tip_calibration_results_page.py:154 tangential_err` (live; 4 lines; same as the previous)
- `scripts/newton_scaling.py:538 Sim.fidelity.rel` (live; 3 lines)
- `scripts/pads_cpu_gpu_consistency.py:115 Metric.rel` (dead; 3 lines; same as the previous)
- `scripts/v3_tip_contact_model.py:198 r_bar` (live; 4 lines; weighted radial distance)
- Inline: the `v - (v @ n) * n` idiom on about 53 lines in 37 files.

Replacement: `vec.reject(v, n)` (vector) and `vec.reject_norm(v, n)` (length), vectorised with `einsum`.
`trimesh.points.project_to_plane` returns 2-D plane coordinates, a different contract.

### 1.4 Tangent bases from a normal (7 named, all live; 4 inline)

- `src/morphohand/geometry/cad_features.py:138 perpendicular_basis` (live; 5 lines; seed switch at |a_x| < 0.9)
- `scripts/fingertip_wrench_model.py:518 tangent_frames` (live; 8 lines; batched; reference z, y near the pole)
- `scripts/ur5_inverted_rig.py:1319 tangent_basis` (live; 7 lines; projected axis with a fallback)
- `scripts/ur5_inverted_rig.py:1098 frames_for` (live; 9 lines; batched frames with a fallback)
- `src/morphohand/control/hom.py:91 frame_from_normal` (live; 7 lines; projected axis with a fallback)
- `scripts/ur5_shear_skin.py:79 frame` (live; 6 lines; no fallback, NaN when v is parallel to n; its caller passes the
  slide direction, which is tangential)
- `scripts/probe_real_v1_chain.py:1180 chain._tool_frame` (live; 8 lines)
- Inline: `scripts/ur5_cb3_cell.py:300 _tube`, `scripts/dip_link_concept_a.py:205 tube_mesh`,
  `scripts/hom_control.py:65 gcf_sphere_cylinder`, `scripts/probe_perp_thumb_engage.py:110 surface_point` (dead)

Replacement: `vec.tangent_basis(n, ref=None)`, batched, with one fallback seed. These build frames, so the rotation
audit may list them too.

### 1.5 Pinhole projection, world point to pixel (4 copies, 3 live)

- `scripts/dip_link_concept_a.py:1187 _project` (live; 12 lines; MuJoCo free camera)
- `scripts/ur5_calibration_mount.py:2082 _project` (live; 11 lines; the same body)
- `scripts/fingertip_calibration_cell_render.py:87 project` (live; 10 lines; look-at camera)
- `scripts/real_v1_carry_construction.py:362 Studio.project` (dead; 21 lines)

Replacement: one `camera.project(P, cam, fovy, w, h)`; no library offers it short of a renderer.

## 2. Distances, closest points, intersections and signed distances

### 2.1 Point to segment (4 copies, 2 live)

- `scripts/real_v1_basketball_grasp.py:256 _seg_dist` (live; 5 lines; broadcast; 1e-12 guard)
- `scripts/ur5_cb3_cell.py:158 _seg_dist_2d` (live; 5 lines; 2-D, points by segments; 1e-18 guard)
- `scripts/karma_seed_geometry.py:39 _seg_dist` (dead; 4 lines)
- `scripts/probe_thumb_reach.py:81 dist_to_segment` (dead; 5 lines; also returns the closest point)

Replacement: `dist.point_segment(P, a, b)` for 2-D and 3-D; numpy, scipy and trimesh have no such routine.

### 2.2 Segment to segment (3 copies, all live; two wrong on zero-length segments)

- `scripts/hand_object_scale_kin.py:226 seg_seg` (live; 16 lines; vectorised; Ericson 5.1.9 branches for zero-length
  segments; agrees with brute force on 300 random pairs to 7e-6 and on the degenerate cases)
- `src/morphohand/rl/terms_common.py:198 _seg_seg_dist` (live; 21 lines; torch; wrong when the second segment has zero
  length, bug 1)
- `scripts/policy_cpu_rollout.py:78 seg_seg` (live; 12 lines; scalar; NaN when either segment has zero length, bug 2)

Replacement: move `hand_object_scale_kin.seg_seg` to the shared module and give the torch twin its degenerate
branches. For geoms that are already in a MuJoCo model, `mujoco.mj_geomDistance` returns the distance itself (40 lines
in 16 files use it).

### 2.3 Point to line and line to line (5 copies, 4 live), and a name used three ways

- `src/morphohand/geometry/cad_features.py:279 line_distance` (live; 7 lines; line to line; the parallel branch
  assumes a unit `a1`; its one caller, `build_cad_v1_finger.py:322`, passes unit axes)
- `scripts/build_cad_v1_finger.py:242 derive.common_perp` (live; 8 lines; line-line distance plus the common normal)
- `scripts/sim2real_fingertip_manifest.py:91 probe.axis_dist` (live; 4 lines; point to line)
- `scripts/real_v1_pusher_reach.py:104 main.perp` (live; 3 lines; point to line)
- `scripts/build_link_length_variants.py:75 yaw_mcp_offset` (dead; 14 lines; point to line)
- Same name, other meanings: `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:271 Geometry.line_distance`
  (live; 10 lines; least distance of the dense pad samples from a line within `half_len`) and
  `scripts/ur5_tip_calibration_rod_registration.py:35 line_distance` (live; 13 lines; signed surface-to-line distance
  by alternating projection).

Replacement: `dist.point_line` and `dist.line_line` (the latter from `cad_features.line_distance`, normalising its
directions); rename the two surface-to-line functions, for example `surface_line_gap`.

### 2.4 Point to triangle, point to mesh, inside tests

- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:188 closest_on_triangles` (live; 26 lines; Ericson
  5.1.5, vectorised over triangles). The only copy. The driver imports numpy alone because it runs on the lab
  computer, so it stays custom; elsewhere `trimesh.triangles.closest_point(triangles, points)` is the same routine.
- 2-D barycentric point-in-triangle tests, the same eight lines twice: `scripts/ur5_inverted_rig.py:1922
  _on_face.inside` (live; 8 lines; tolerance 1e-9) and `src/morphohand/driver/reskin5x/host/ur5_tip_calibration_sim.py:456
  World.wall_contact` (live; 27 lines; inline, no tolerance). Library: `trimesh.triangles.points_to_barycentric`.
- Point to mesh: `cKDTree` over surface samples on 51 lines in 33 files (the usual choice); `trimesh.proximity.closest_point`, `signed_distance` or `nearest.on_surface` on 7 lines in 5
  files; `mujoco.mj_geomDistance` on 40 lines in 16 files.
- `scripts/tip_codesign_geometry.py:661 compare_cad` (live; 17 lines) calls `trimesh.proximity.signed_distance` on
  60,000 unchunked samples per mesh (lines 672-673) and then takes `abs()`; `main` calls it at line 697. The same
  numbers come from `closest_point` in chunks of 4,096 points, or from
  `scipy.spatial.distance.directed_hausdorff` on the two sample sets (item 7 of the bug list).
- Signed distance from surface samples with normals (k = 8 nearest): `scripts/ur5_cb3_cell.py:648 held_tip_sdf`
  (live; 22 lines; sign of the summed offsets), `scripts/ur5_inverted_rig.py:857 _tree_sdf` (live; 5 lines; the same
  rule), `scripts/build_cad_v2_finger.py:68 pack` (live; 51 lines; inside by a 5-of-8 vote, a second rule).
- Swept clearance about the PIP axis (k-d tree distance, sign from `SolidGrid.contains`), four copies:
  `scripts/dip_link_geometry.py:546 points_clearance_exact` (live; 34 lines; the original),
  `scripts/dip_link_concept_b.py:1303 clearance_exact` (live; 45), `scripts/dip_link_board_mount.py:858
  points_clearance_curve` (live; 27; per angle), `scripts/dip_link_viewer.py:327 Geometry.clearance_at` (live; 13).
- Inside tests come in four forms: trimesh `contains`; `cad_features.SolidGrid.contains` on voxels; ray parity in
  `v2_finger_scenes.py`; `scripts/ur5_inverted_rig.py:1985 _winding` (live; 13 lines; generalised winding number;
  each chunk of 256 points allocates three arrays of 256 x n_triangles x 3 floats, and the second caller at line 2485
  lowers the chunk to 32).

Replacement: `sdf.sampled_sdf(tree, P, N, Q, k=8, rule="sum"|"vote")` in the shared module, one inside test chosen per
use, and the swept-clearance routine kept in `dip_link_geometry` with a per-angle option.

### 2.5 Ray to triangle (2 copies, both live)

- `src/morphohand/geometry/cad_features.py:118 _ray_all` (live; 18 lines; Moller-Trumbore, every crossing; reused by
  `dip_link_geometry.ray_intervals`)
- `scripts/fingertip_holder.py:112 _first_hit_x` (live; 20 lines; the same lines inline, first hit along +x)

Replacement: make `cad_features._ray_all` public and call it. `trimesh.ray.ray_triangle.RayMeshIntersector` is the
library alternative.

### 2.6 Signed distances of primitives

- `scripts/ur5_cb3_cell.py:357 sd_box`, `:364 sd_cyl_z`, `:371 sd_cyl_y` (live; 5 lines each), `:378
  sd_capped_cone` (live; 14 lines; Quilez's formula), `:460 _convex_sdf` (live; 5 lines; a lower bound outside). 21
  scripts import the module as `CELL`. `sd_capped_cone` and `sd_cyl_z` agree with a brute-force profile distance
  to 4e-15 on 4,000 random points (`check_sdf.py`).
- `scripts/ur5_inverted_rig.py:864 hull_sdf` (live; 9 lines; `_convex_sdf` with cached planes, as its docstring says)
- `scripts/sim2real_fingertip_scenes.py:327 box_sd` (live; 6 lines; `sd_box` for an oriented box)
- `scripts/dip_link_concept_b.py:1987 cmd_eval.box_dist` (live; 3 lines) and `scripts/dip_link_v2.py:1404
  b_r4.box_dist` (live; 2 lines): both equal `max(sd_box, 0)`
- `scripts/dip_link_board_mount.py:396 disc_distance` (live; 6 lines) and `scripts/dip_link_viewer.py:92
  disc_distance` (live; 4 lines): the same body, equal to `max(sd_cylinder along x, 0)`
- `scripts/indenter_array.py:180 array_sdf` (live; 41 lines) and `scripts/indenter_plate_layout.py:117 part_sdfs`
  (live; 43 lines): the same composition of the plate; `array_sdf` is the minimum over the parts plus the held tips.
- `src/morphohand/control/surfaces.py` `Sphere`, `Cylinder`, `RoundedBox`, `HalfEllipsoid`, `Plane` (live; scalar
  closest point, signed distance and normal), with torch ports at `scripts/hom_gpu.py:617-707` (documented as ports).
  `scripts/hom_control.py:65 gcf_sphere_cylinder` (live; 35 lines) repeats `Cylinder.closest` and adds a frame.

Replacement: no numpy or scipy routine covers analytic SDFs. Move the five `ur5_cb3_cell` functions to
`src/morphohand/geometry/sdf.py`, add a general `sd_cylinder(P, axis, r, lo, hi, centre)` and an optional pose, and
write the unsigned versions as `np.maximum(sd, 0)`.

Sign conventions were compared across copies: every SDF is negative inside (`sd_box`, `box_sd`, `_convex_sdf`,
`surfaces.*`, `KeepOut.signed`, `held_tip_sdf`, `Geometry.surface_distance`), and both hull builders wind their faces
outward. No disagreement was found.

## 3. Mesh and solid properties

### 3.1 Triangle normals and areas by hand

- `scripts/tip_codesign_geometry.py:99 _subsample` (live; 22 lines; areas and normals, then a deterministic n x n
  subdivision; keep)
- `scripts/tip_codesign_geometry.py:193 TipShape._planes` (live; 10 lines)
- `scripts/tip_codesign_law.py:102 DrakeField.__init__` (live; 11), `scripts/curved_tips_stiffness.py:292
  NearField.__init__` (live; 12), `scripts/v3_tip_contact_model.py:94 Field.__init__` (live; 14): facet normals with a
  1e-14 filter for degenerate facets
- `scripts/sim2real_fingertip_scenes.py:309 sample_mesh_points` (live; 16 lines) and
  `scripts/fingertip_contact_estimation_study.py:150 surface_points` (live; 12 lines): areas plus random barycentric
  sampling
- `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:221 Geometry.__init__` (live; numpy-only driver)

Replacement: `trimesh.triangles.normals(triangles)` (returns normals and a validity mask) and
`trimesh.triangles.area`; `trimesh.sample.sample_surface(mesh, n, seed=...)` for the two samplers (already used in 12
files), with `trimesh.triangles.points_to_barycentric` for interpolated normals.

### 3.2 Volume and centroid

- `scripts/tip_codesign_geometry.py:323 TipShape.volume_centroid` (live; 6 lines; signed tetrahedra about C0); the
  same file builds `trimesh.Trimesh` meshes elsewhere, so `.volume` and `.center_mass` replace it.
- Tetrahedral FE meshes: `scripts/tpu_tip_fem_plate_check.py:118 _volume` (live; 3 lines) and
  `scripts/contact_bed_candidates.py:382 add_flex` (live; inline at 385); one line each, `abs(det) / 6`.

### 3.3 Convex hull with outward winding (2 copies, both live)

- `scripts/tip_codesign_geometry.py:85 _hull` (live; 12 lines; vectorised flip)
- `scripts/fingertip_geometry.py:154 hull_mesh` (live; 18 lines; Python loop over the faces)

Replacement: `trimesh.convex.convex_hull(points)`, already called at `scripts/tpu_tip_design.py:150` and
`scripts/ur5_inverted_rig.py:803`. Five 2-D outlines call `scipy.spatial.ConvexHull` themselves, which is fine.

### 3.4 Inertia

- Parallel-axis sums `I + m (d.d E - d d^T)`, 7 sites in 4 files: `scripts/build_cad_v1_finger.py:472 inertial` (live;
  22 lines), `scripts/build_cad_v2_finger.py:121 mass_props` (live; 14), `scripts/v2_finger_scenes.py:73 inertial`
  (live; 19), `scripts/sim2real_fingertip_scenes.py:150 tip_mass_props` (live; 28; two sites) and `:195 add_mass`
  (live; 23; two sites).
- Primitive formulas. Solid cylinder: `scripts/hom_contact_rig.py:58-59` (module constants),
  `scripts/sim2real_fingertip_scenes.py:174` (inline), `scripts/contact_surface/mujoco_model.py:49` (inline, literal
  radius and length), `scripts/probe_roll_fidelity.py:65 _cyl_inertia` (dead; 6). Box or bar:
  `scripts/sim2real_fingertip_scenes.py:210` and `scripts/v2_finger_scenes.py:83` (inline),
  `scripts/hom_contact_rig.py:115 bar_inertia` (live; 3), `scripts/apply_measured_plant.py:121` (rod, inline). Sphere:
  `scripts/hom_contact_rig.py:243` and `scripts/newton_pinch_probe.py:152` (inline).

Replacement: `trimesh.inertia.cylinder_inertia` and `sphere_inertia`; a box formula and `combine_inertia([(m, c, I),
...])` in a shared mass-properties module, since no library mixes primitive parts with mesh parts.

### 3.5 Polygon section properties and orientation

- `scripts/dip_b_v4.py:164 section_props` (live; 24 lines; holes found by ring orientation)
- `scripts/ur5_calibration_mount.py:1855 section_props` (live; 36 lines; shapely union and an optional shell)
- `scripts/ur5_inverted_rig.py:782 cmd_seat_beam.props` (live; 10 lines; exterior ring only)
- Shoelace sign before an extrusion: `scripts/ati_stack_mount.py:291 _prism_z` (live; 5), `scripts/dip_link_concept_a.py:337
  MeshK.prism` (live; 7), `scripts/dip_link_concept_b.py:630 MeshB._extrude` (live; 10),
  `scripts/ur5_calibration_mount.py:437 MeshK.prism_xz` (live; 10)

Replacement: `trimesh.path.polygons.second_moments(polygon, return_centered=True)` with shapely's area and centroid;
`shapely.geometry.polygon.orient(poly, 1.0)` for the orientation.

### 3.6 Mesh builders and writers

- `box_mesh`, 3 copies: `scripts/ur5_cell_render.py:248` (live; 6 lines; sorts lo and hi),
  `scripts/dip_link_print_project.py:110` (live; 4), `scripts/dip_link_board_mount.py:265` (live; 4). Replacement:
  `trimesh.creation.box(bounds=[lo, hi])`.
- OBJ writers: `scripts/fingertip_geometry.py:174 write_obj` (live; 10), `scripts/pad_usability_backends.py:64
  _write_obj_atomic` (live; 8), `scripts/tip_codesign_geometry.py:385 TipShape.mesh_paths` (live; inline at 399),
  `scripts/build_screw_scene.py:57 _frustum_obj` (live; 24; inline), `scripts/tpu_tip_fem_polyfem.py:73 plate_obj`
  (dead; 5). `scripts/ur5_cell_render.py:521 write_obj` (live; 5) writes smoothed normals, a different output.
  Replacement: `trimesh.Trimesh(V, F, process=False).export(path)` behind one atomic-write helper.
- STL: no hand-written writer exists; every STL goes through trimesh export.
- Tubes: `scripts/dip_link_concept_a.py:205 tube_mesh` (live; 28; parallel-transport frames), `scripts/ur5_cb3_cell.py:300
  _tube` (live; 12; points only), `scripts/dip_link_concept_c.py:545 tube` (dead; 18; manifold3d). Replacement for the
  mesh: `trimesh.creation.sweep_polygon`.
- CAD-kernel wrappers (manifold3d meshes and FreeCAD B-reps): `MeshK` and `BrepK` at `scripts/dip_link_concept_a.py:325`
  and `:379` and at `scripts/ur5_calibration_mount.py:409` and `:484` (`box`, `union`, `sub`, `inter` identical; `cyl`
  the same contract with different helper names and segment counts); `MeshB` and `BrepB` at
  `scripts/dip_link_concept_b.py:615` and `:547`; free `box`/`prism`/`cyl` wrappers in `dip_fit_print_plates.py:174-183`,
  `dip_link_onshape_export.py:77`, `dip_link_params.py:824`, `ati_stack_mount.py:291`, `cad_v2_assets.py:361` and the
  dead `dip_link_concept_c.py:216-455`. Replacement: one `cad_kernel` module with both backends.
- Bounding boxes use `.min`/`.max`/`np.ptp` or trimesh `bounds` throughout; nothing to consolidate.

## 4. Sampling and grids

### 4.1 Fibonacci lattices (10 implementations, all live)

- Whole sphere: `scripts/ati_calibration_plan.py:208 fib_sphere` (5 lines), `scripts/ur5_inverted_rig.py:1083
  fib_dirs` (4; the same body), `scripts/ur5_inverted_rig.py:503 _sphere_pts` (5; scaled and offset),
  `scripts/tip_codesign_geometry.py:74 _fib` (6)
- Cap: `scripts/hom_contact_rig.py:166 fib_cap` (7; imported by `hom_chain`, `hom_hand_brake`, `hom_chain_study`,
  `contact_surface/geometry` and `contact_overview_page`), `scripts/hom_chain_figures.py:17 fib_cap` (6; identical
  copy), `scripts/v3_tip_contact_model.py:124 fib_cap` (11; same name, arguments `(R, cos_min, spacing)`, returns
  directions and areas)
- Octant: `scripts/fingertip_geometry.py:45 _corner_dirs` (15; Python loop over 8n points)
- Inline: `scripts/curved_tips_stiffness.py:237 Probe.__init__` (24) and `scripts/fe_contact_calibration_fit.py:95
  Probe.__init__` (36; an extended copy of the same class)
- Two golden-angle spellings appear, pi (3 - sqrt 5) and pi (1 + sqrt 5) = 4 pi - pi (3 - sqrt 5). The second gives
  the mirror image of the first lattice. No code compares point sets across the two.

Replacement: `sampling.fibonacci(n, cap=None, octant=False)`; numpy and scipy have none, and
`trimesh.sample.sample_surface_sphere` is random.

### 4.2 Pads on a spherical cap (7 sites)

The count `round(A / s^2)`, the per-sphere stiffness `K_s = E / R * A_s * ((R - rs) / R)^2` and the `fib_cap`
directions recur in: `scripts/hom_contact_rig.py:193 mj_xml` (live; inline at 209-218), `scripts/hom_chain.py:129
chain_scene` (live; inline at 170-176), `scripts/hom_hand_brake.py:207 mj_scene` (live; inline at 233-238),
`scripts/hom_chain_study.py:333 one_contact` (live; inline at 345-348; no `max(1, n)` and no range check on the
impedance), `scripts/contact_overview_page.py:1513 cap_pad` (live; 6), `scripts/hom_chain_figures.py:42 pad` (live;
13), and `scripts/contact_surface/geometry.py:21 Surface.parameters` (live; 33 lines; the version that refuses an
impedance outside [0.0001, 0.9999]).

Replacement: `contact_surface.geometry.Surface` at every site.

### 4.3 Surface sampling

`trimesh.sample.sample_surface` and `sample_surface_even` are called in 12 files. The two hand-rolled samplers are
listed under 3.1. Two deterministic samplers are kept on purpose: `tip_codesign_geometry._subsample` and
`scripts/dip_link_geometry.py:448 surface_points` (live; 31 lines).

### 4.4 Half-ellipsoid pad

- Closest point: `src/morphohand/control/surfaces.py:153 ellipsoid_closest` and `HalfEllipsoid` (live), with a torch
  port at `scripts/hom_gpu.py:617`, `:641` and `:653` (documented).
- Support function: `scripts/fp_sr2_toy.py:287 support` (live; 8 lines) and `scripts/hom_gpu.py:653 Batch._ell_support`
  (live; 6 lines); both take az below the centre plane and azp above it.
- Pad builders: `scripts/tip_codesign_geometry.py:529 CurvedTip.pads` and `:333 TipShape.pads`,
  `scripts/fingertip_geometry.py:138 pad_spheres` (the block); `reorient_backends.replace_tips` reuses them.
- The 11/22 section restated with the literals 7.4, 8.5, 11 and 22 mm: `scripts/dip_b_tip_cap.py:95 end_face` (5),
  `scripts/dip_b_tip_cap_sketch.py:41 half_w` (5) and `:103 end_face` (7), `scripts/dip_link_concept_b.py:978
  pad_half_width` (5) and `:912 bridge_x_palmar` (5), `scripts/dip_b_magnet_layouts_figure.py:34 outline` (7).

Replacement: `section(z)` and `support(u)` methods on `surfaces.HalfEllipsoid`, built from the cad_v2 tip key.

### 4.5 Sphere packings

`src/morphohand/geometry/sphere_pack.py` is used by `scripts/generate_sphere_packed_scene.py` (live) and
`scripts/probe_roll_fidelity.py` (dead). No script re-implements its packings of primitives. Two other packers fill
CAD meshes, a different task: `scripts/cad_v1_spheres.py` (MorphIt) and `scripts/build_cad_v2_finger.py:68 pack`
(inscribed spheres). The hex lattice of `scripts/contact_bed_candidates.py:350 cap_mesh` builds a flex mesh.

## 5. Small numerical helpers

### 5.1 Rank correlation (14 copies, 8 live)

- Tie-averaged ranks, agreeing with scipy: `scripts/bench_plan_page.py:299 spearman` (live; 18 lines; 0.0 for a
  constant input), `scripts/fingertip_backends_page.py:122 spearman` (live; 20 lines; None for a constant input or n <
  3)
- Ordinal ranks, ties broken by row order (bug 3): `scripts/fp_review_page.py:147 spearman` (live; 4) and `:467 rank`
  (live; 2), `scripts/fp_review_figures.py:372 spearman` (live; 7), `scripts/hom_plant_transfer.py:79 rank` (live; 5;
  mergesort) and `:86 spearman` (live; 2), `scripts/real_v1_screen_chain_page.py:46 spearman` (live; 9),
  `scripts/a_quality_predictor.py:81 spearman` (dead; 12), `scripts/plant_ranking_test.py:41 spearman` (dead; 9),
  `scripts/probe_a_reorientability.py:108 analyze.spearman` (dead; 6)
- Wrappers of `scipy.stats.spearmanr`: `scripts/real_v1_transfer_study.py:138 spearman` (live; 4),
  `scripts/overview_figures.py:107 _spearman_auc` (live; 9; AUC by pairwise counting),
  `scripts/hand_object_scale_landscape.py:147` (dead; 5), `scripts/real_v1_search_report.py:137` (dead; 7),
  `scripts/workshop_briefing_page.py:52` (dead; 4)

Replacement: `scipy.stats.spearmanr(a, b, nan_policy="omit").statistic` behind one wrapper that returns NaN for n < 3
or a constant input; AUC as `scipy.stats.mannwhitneyu(pos, neg).statistic / (n_pos * n_neg)`.

### 5.2 Bootstrap and binomial intervals

- One level of resampling: `scripts/servo_plant_policies_page.py:1066 boot_ci` (live; 7),
  `scripts/real_v1_transfer_study.py:144 bootstrap_rho` (live; 13; resamples plans), `scripts/pad_usability_agree.py:87
  boot_median` (dead; 5), `scripts/karma_analysis.py:49 auc_ci` (dead; 13) and inline at `karma_analysis.py:181`,
  `:305`, `:356` (dead), `scripts/workshop_briefing_page.py:180 rho_block` (dead; inline at 200)
- Cluster or two-level resampling: `scripts/servo_plant_policies_page.py:1044 boot_delta` (live; 20),
  `scripts/ur5_tip_calibration_suite_value.py:664 ratio_band` (live; 18), `scripts/pad_usability_planner_report.py:67
  boot_cluster` (dead; 8), and the same per-cell by per-seed index draw in `scripts/tip_codesign_analysis.py:512 stage3`
  (live; lines 524-525) and `scripts/curved_tips_analysis.py:240 stage3` (live; lines 246-247)
- Binomial: `scripts/bench_plan_page.py:295 binom_at_least` (live; 2) equals `scipy.stats.binom.sf(k - 1, n, p)`;
  `scripts/pad_usability_rl_report.py:47 wilson` (dead; 6; returns the interval in counts, not as a proportion) and
  `scripts/pad_usability_agree.py:71 clopper_pearson` (dead; 5) equal `scipy.stats.binomtest(k, n).proportion_ci(method=
  "wilson")` and `method="exact"`.

Replacement: `scipy.stats.bootstrap(..., method="percentile")` for the one-level cases (`paired=True` for
`bootstrap_rho`); one custom `cluster_bootstrap` for the hierarchical draws, which scipy does not offer.

### 5.3 Smoothing (14 moving averages, 13 live, five edge rules; 2 smoothsteps)

- Centred, padded with the end values: `scripts/ur5_shear_skin.py:56 smooth` (live; 6),
  `scripts/servo_plant_policies_page.py:1555 _smooth` (live; 5; one sample longer for an even window, only 9 is used),
  `scripts/rl_plot_reorient.py:89 smooth` (dead; 7), `scripts/contact_model_policies_page.py:617 _tb_lines` (live;
  inline). Library: `scipy.ndimage.uniform_filter1d(x, k, mode="nearest")`.
- Centred, the window shrinking at the ends: `scripts/ur5_tip_calibration_lab_page.py:318 smooth15` (live; 6)
- Centred, zero-padded `np.convolve(..., mode="same")`: `scripts/ur5_tip_calibration_lab_page.py:352 ref_rows` (live;
  inline at 359; bug 6), `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py:379 fit_onset` (live; inline at
  387; drops the last win/2 samples), `scripts/fingertip_contact_estimation_study.py:103 no_contact` (live; inline at
  116; slices the edges off), `scripts/hands_tranche_page.py:196 plot_curves` (live; `mode="valid"`)
- Trailing: `scripts/fp_review_figures.py:96 moving` (live; 8), `src/morphohand/driver/reskin5x/host/reskin5x_mux_rest.py:47
  smoothed_T` (live; 6), `src/morphohand/driver/reskin5x/host/pad_shift_check.py:38 trailing_mean` (live; 6; by time
  span)
- Mean over a time window: `scripts/contact_bed_pull.py:97 smooth_v_at` and `scripts/contact_bed_twist.py:100
  smooth_at` (live; 3 lines each; the same body)
- Smoothstep: `scripts/contact_surface/slip.py:21 smooth` and `scripts/ur5_cell_render.py:382 _smooth` (live; 3 each)
- No file uses `scipy.signal` (Savitzky-Golay, Butterworth, filtfilt); `scipy.ndimage.gaussian_filter` appears once,
  in `hom_landscape.smooth_map`.

Replacement: `signal.moving_average(x, k, edges="nearest"|"shrink")` on `uniform_filter1d`, `signal.trailing_mean(t, x,
span)` and `signal.smoothstep`.

### 5.4 Finite differences

`np.gradient` (4 sites) and `np.diff(...) / dt` (6 lines in 4 files) are library calls already.
`scripts/three_finger_soak_page.py:179 fig_steps` and `src/morphohand/driver/reskin5x/host/reskin5x_mux_steps.py:38
steps` compute the same central step `|x[k+1] - x[k-1]|` on one line each. Nothing to consolidate.

### 5.5 Interpolation

- Log-log with extrapolation along the end segments: `scripts/tpu_tip_calibrate.py:56 approach_at` (live; 7),
  `scripts/sim2real_fingertip_bench.py:39 _loglog` (live; 6), `scripts/curved_tips_stiffness.py:323 from_curve` (live;
  13; adds a monotone filter); the page's JavaScript calculator repeats it (docstring of `approach_at`). Library:
  `np.exp(scipy.interpolate.interp1d(np.log(xs), np.log(ys), fill_value="extrapolate")(np.log(x)))`.
- Hand loop: `scripts/bench_plan_page.py:394 curve_spread.at` (live; 5) equals `np.interp(x, xp, fp, left=np.nan,
  right=np.nan)`.
- Sample-and-hold lookups with two conventions: `scripts/bench_replay_calibration.py:231 compare.at` (live; 2; first
  sample at or after t), `scripts/reorient_backends.py:428 score.at` (live; 2; the same), and
  `scripts/distributed_contact_videos.py:469 state_at` (live; 7; last sample at or before t). Library: `interp1d(kind=
  "next")` or `kind="previous"`, or one helper with the side as an argument.
- Colour ramp: `scripts/contact_bed_common.py:139 heat` and `scripts/hom_chain.py:1061 heat` (live; 9 lines each; the
  same body). Library: `matplotlib.colors.LinearSegmentedColormap.from_list`.
- `np.interp`, `interp1d`, `CubicSpline` and related calls appear on 65 lines in 31 files already.

### 5.6 Fits

- Circle: `scripts/cad_v2_assets.py:146 circle` (live; 6 lines; algebraic Kasa fit), the only copy;
  `trimesh.nsphere.fit_nsphere` minimises the radial error instead.
- Line with standard errors: `scripts/ur5_tip_calibration_fe.py:74 line_fit` (live; 7) equals
  `scipy.stats.linregress` (slope, intercept and both standard errors).
- Other fits call `np.linalg.lstsq`, `np.polyfit`, `least_squares` or `curve_fit` (112 lines in 61 files), each for its
  own model. No duplicated plane fit or sphere fit was found. Rigid registrations (Kabsch, Procrustes) are left to the
  rotation audit.

### 5.7 Angle wrapping

- Named: `src/morphohand/optimization/phase1_common.py:418 _wrap_to_pi` (live; 2)
- Inline: 34 lines in 26 files use `(a + pi) % 2pi - pi`, `atan2(sin a, cos a)` or `(x + 180) % 360 - 180`.
- Nearest equivalent of a joint angle to a reference, `q_ref + wrap(q - q_ref)`, 8 live sites:
  `scripts/ati_calibration_plan.py:253`, `scripts/indenter_array.py:260`, `scripts/tall_indenter.py:264`,
  `scripts/ur5_inverted_rig.py:439`, `scripts/ur5_tip_calibration_plan.py:543`, `scripts/ur5_tip_calibration_r24_pass3.py:172`,
  `scripts/ur5_tip_calibration_tall_pass.py:112`, `scripts/ur5_tip_calibration_trimmed_cap_reach.py:88`
- Incremental unwrap of a turning angle: `scripts/hom_contact_rig.py:385 MjRig.tool_state` and `:537
  DrakeRig.tool_state`, `scripts/contact_bed_newton.py:410 NewtonRig.tool_state`, `scripts/hom_paper_tasks.py:516
  wield.turned` (live), `scripts/pad_mass_calibration.py:151 Tool.theta` (dead)

Replacement: `angles.wrap(a, period=2pi)`, `angles.nearest(q, q_ref)` and `angles.unwrap_step(prev, new)`; `np.unwrap`
covers whole sequences only.

### 5.8 Root finding

30 functions in 23 files run a hand-written bisection loop (27 live, 3 dead; 5 enclosing functions that match only
through a nested one are not counted). Most invert a monotone force-depth law:
`hom_contact_rig.py:153 winkler_law`, `contact_overview_page.py:1521 flat_pad`, `v3_tip_contact_model.py:268
bisect_depth`, `pad_contact_rig.py:169 continuum_approach`, `tpu_tip_fem_analysis.py:41 law_delta`,
`ur5_tip_calibration_pads.py:242 main.delta_law`, `ur5_shear_impratio.py:924 cmd_creep.delta_law`,
`tpu_tip_magnet.py:129 best_modulus.d_law`, `hom_chain.py:732 brake_force_for`, `hom_chain_figures.py:57 solve`, and
others. `tip_codesign_geometry.py:204 TipShape._bisect`, `:213 _outline`, `:417` and `:548 fe_surface_x` and `:567
_outline_z` bisect many rays at once. The full list is in `bisect.txt` in this folder.

Replacement: `scipy.optimize.brentq` for scalars and `scipy.optimize.elementwise.find_root` (present in scipy 1.17.1)
for the vectorised ray bisections.

### 5.9 Unit conversions

Conversions use `np.degrees`, `np.radians` or literal factors; the `mm()` helpers are HTML formatters. Nothing to
consolidate.

### 5.10 Other duplicated numerical helpers

- Magnetic dipole field: `scripts/mlx90393_model.py:140 dipole` (live; 5), `scripts/tpu_tip_magnet_sensing.py:49
  dipole` (live; 5; the same body), `scripts/tpu_tip_design.py:166 dipole_B` (live; 6),
  `scripts/fingertip_wrench_model.py:112 dipole_field` (live; 7; vectorised over magnets). One `dipole_field` would do.
- Drake's compliant-convex pressure `p = E (1 - g)` with `g = max_k (X - c) . n_k / d_k`: `scripts/tip_codesign_law.py:102
  DrakeField` (facets with n_x > -0.2), `scripts/curved_tips_stiffness.py:292 NearField` (all facets, restricted near
  the probe), `scripts/v3_tip_contact_model.py:94 Field` (surface triangles plus the base plane, normals oriented by
  the vertex normals), and the formula again at `scripts/fe_contact_calibration_fit.py:224 Field.p_at`. The facet sets
  differ by design; one class with a facet filter would cover all four.
- Pinch slip torque per unit friction: `scripts/tip_codesign_law.py:87 _arm` (live; 12) and
  `scripts/hydroelastic_arm_integral.py:19 _forces` (live; 19) share six lines.

## Confirmed bugs and inconsistencies

1. `src/morphohand/rl/terms_common.py:198 _seg_seg_dist` returns the distance from `p1` when the second segment has
   zero length. Lines 210-211 set `s = 0` because `denom = 0`, and `t = f / eps = 0`. Input `p1 = (0, 0, 0)`, `q1 =
   (2, 0, 0)`, `p2 = q2 = (1, 1, 0)` gives 1.414; the true distance is 1.0 (`check_segseg.py`, the torch code
   transcribed to numpy). It feeds `finger_pair_clearance` (line 228). In `real_hand.xml` consecutive chain bodies are
   20.75 mm apart, so real_v1 does not trigger it; a morphology with two coincident chain frames would.
2. `scripts/policy_cpu_rollout.py:78 seg_seg` divides by `e` at line 84 and by `a` at lines 86 and 88 without guards.
   It returns NaN for a zero-length segment on either side; for example the second segment as a point gives NaN where
   the true distance is 1.0. Its caller `main.clearance` (line 307) folds the result with Python `min`, which ignores
   a NaN in the second argument.
3. Seven Spearman copies use ordinal ranks, so ties are broken by row order (`fp_review_page.py:147`,
   `fp_review_figures.py:372`, `hom_plant_transfer.py:79/86`, `real_v1_screen_chain_page.py:46`, and the dead
   `a_quality_predictor.py:81`, `plant_ranking_test.py:41`, `probe_a_reorientability.py:108`). For x = [0, 1/3, 1/3,
   2/3, 1, 1], y = [0.2, 0.1, 0.5, 0.4, 0.9, 0.3] they return 0.543, and 0.486 for the same rows reversed; scipy gives
   0.530. A constant x = [1, 1, 1, 1, 1] with y = [0.1, 0.5, 0.3, 0.9, 0.7] returns rho = 0.80; scipy returns NaN,
   `bench_plan_page` 0.0 and `fingertip_backends_page` None (`check_spearman.py`). On the published plant-transfer
   rows (`20261102-plant_transfer.json`, 369 to 527 layouts per pair, up to 70 tied values) the copy and scipy differ
   by at most 0.003, for example 0.587 against 0.588 for twist and -0.100 against -0.099 for press
   (`check_plant_rho.py`). Small samples with many ties, such as hold fractions over a few designs in
   `real_v1_screen_chain_page`, are where the difference grows.
4. Three names each mean two or three different things. `tangential` returns a vector in
   `fingertip_crosstalk.py:112` and `ur5_tip_calibration_analysis.py:364`, and a distance in
   `fingertip_magnet_transfer_sim.py:69` and `ur5_tip_calibration_estimator_checks.py:44`. `line_distance` is
   line-to-line in `cad_features.py:279`, a sampled surface-to-line minimum in `ur5_tip_calibration.py:271`, and a
   signed surface-to-line distance in `ur5_tip_calibration_rod_registration.py:35`. `fib_cap(n, cap)` returns
   directions in `hom_contact_rig.py:166`, while `fib_cap(R, cos_min, spacing)` returns directions and areas in
   `v3_tip_contact_model.py:124`.
5. The zero-vector policies differ: NaN in `build_cad_v1_finger.py:70` and `hom_paper_tasks.py:57`, the zero vector
   in `surfaces.py:20`, an exception in `tags.py:339` and `force_closure.py:47`. As a result
   `build_cad_v1_finger.py:245 derive.common_perp` returns a NaN direction for two coaxial lines (its distance, 0, is
   right), and `hom_paper_tasks.py:408` gives a NaN `n_top` when the tool axis is vertical; that task holds the tool
   horizontal, so the case is latent.
6. In `scripts/ur5_tip_calibration_lab_page.py`, `smooth15` (line 318) normalises the window at the edges, while
   `ref_rows` (line 359) smooths the push with a zero-padded `np.convolve(mode="same")`. The last 7 smoothed samples
   fall to between 8/15 and 14/15 of the local mean. If the smoothed push therefore never passes 1.95 N,
   `np.argmax(push > 1.95)` returns 0 and the depth is computed from `s[0]`. This was not checked against the recorded
   presses.
7. `scripts/tip_codesign_geometry.py:672-673 compare_cad` runs `trimesh.proximity.signed_distance` on 60,000 unchunked
   points per mesh and keeps only `abs()`. The same call on about 670,000 points reached 24 GB on 2026-10-14. This is a
   memory risk, not a wrong number.
8. `scripts/probe_real_v1_gait.py:180 _axis_tilt` and `scripts/hom_paper_tasks.py:523 wield.tilt` fold the tilt with
   `abs()`, which `src/morphohand/bench/tags.py:139` documents against. For the symmetric cylinder of
   `probe_real_v1_gait` the fold does not change a verdict.
9. `scripts/diffmjx_audit_page.py:142 tilt_deg` calls `math.acos` without a clip; a recorded cosine plus its gain above
   1 would raise ValueError during the page build. Not observed.

Checked and found consistent: all 29 unclipped `arccos` calls; SDF signs across copies; `sd_capped_cone` and
`sd_cyl_z` against brute force (4e-15); the two golden-angle spellings (mirror images); the half-selection rule of the
two half-ellipsoid support functions; the tie-averaging Spearman copies against scipy.

## Recommended layout of the shared helpers

`src/morphohand/geometry/` (next to the existing `cad_features.py` and `sphere_pack.py`):

- `vec.py`: `unit(v, eps, zero=...)`, `angle(a, b)`, `deg_from_cos(c)`, `reject(v, n)`, `reject_norm(v, n)`,
  `tangent_basis(n, ref=None)`, `project_pinhole(P, cam, fovy, w, h)`
- `dist.py`: `point_segment`, `segment_segment` (from `hand_object_scale_kin`; the torch twin stays in
  `rl/terms_common.py` with the same branches), `point_line`, `line_line` (from `cad_features.line_distance`),
  `closest_on_triangles` (numpy only, so the lab driver can import it), `barycentric_inside`, `ray_triangles` (from
  `cad_features._ray_all`)
- `sdf.py`: `sd_box(P, lo, hi, pose=None)`, `sd_cylinder(P, axis, r, lo, hi, centre)`, `sd_capped_cone`,
  `sd_convex(P, pts)` with cached planes, `sampled_sdf(tree, P, N, Q, k=8, rule="sum"|"vote")`; `surfaces.py` stays
  the scalar closest-point library and gains `HalfEllipsoid.section` and `.support`
- `mesh.py`: an outward convex hull (wrapping `trimesh.convex.convex_hull`), `voxelize_solid` with axis and vote
  options (absorbing `v2_finger_scenes.column_parity`), `section_props` (trimesh `second_moments`),
  `combine_inertia`, `box_inertia` (cylinder and sphere from `trimesh.inertia`), `write_obj_atomic`
- `sampling.py`: `fibonacci(n, cap=None, octant=False)` with one golden-angle convention; the cap-pad layout stays
  `contact_surface.geometry.Surface`; the deterministic samplers `tip_codesign_geometry._subsample` and
  `dip_link_geometry.surface_points` move here
- `cad_kernel.py`: `MeshK` and `BrepK` merged from `dip_link_concept_a`, `ur5_calibration_mount` and
  `dip_link_concept_b`

`src/morphohand/numerics/` (new):

- `stats.py`: `spearman` (scipy wrapper with one NaN policy), `auc` (Mann-Whitney), `bootstrap_ci`
  (`scipy.stats.bootstrap`), `cluster_bootstrap` (custom), `binom_ci` (`scipy.stats.binomtest`)
- `signal.py`: `moving_average(x, k, edges)`, `trailing_mean(t, x, span)`, `value_at(t, ts, side)`, `loglog_interp`,
  `smoothstep`
- `angles.py`: `wrap`, `nearest`, `unwrap_step`
- root finding needs no module: call `scipy.optimize.brentq` and `scipy.optimize.elementwise.find_root`

Two physics helpers recur and would sit beside the contact code rather than in geometry: the Drake compliant-convex
field (`DrakeField`, `NearField`, `Field`) and the dipole field (four copies).
