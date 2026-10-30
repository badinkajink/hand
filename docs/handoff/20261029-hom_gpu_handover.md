# Hand-over of the HOM GPU evaluator and the foundational-pose sampling work, 2026-10-29

Status: handed over to hand-14 at 22:50 on 10-29, after the owner moved this work to that session at 19:3x. Written by
the window of session 8955e04e (colorado login), which wrote scripts/hom_gpu.py and the two 10-29 pages. Not armed.
State: everything below is committed and pushed. The GPU port and the HOM review page are in a159c0faa (merge
58cf9f990). The design page with its rows and figures is in c39d07868. Queues 5 and 6 ended at 20:28 and 20:53 with exit
0. No job of this window runs, and it makes no further edits to the HOM scripts or pages.
Read first:
- docs/experiments/20261029-foundational_pose_review/20261029-foundational_pose_hand_design.html, Sections 6-8 (the GPU
  evaluator, its checks, sensing across the Sobol layouts) and 12 (future work)
- docs/experiments/20261029-hom_control_review/20261029-hom_control_and_fingertip_sensing.html, Table 3 (GPU against CPU)
  and Tables 4-5 (sensing variants, 30 seeds)
- the module docstrings of scripts/hom_gpu.py and scripts/hom_tasks.py
- memory project_hom_fp_review_2026_10_29.md (directory /home/humanoid/.claude/projects/-home-humanoid-Programs-hand/memory/,
  shared by the logins)

## Contents of scripts/hom_gpu.py

The script runs the HOM controller of src/morphohand/control/hom.py on MuJoCo-Warp for many layouts, seeds and sensing
variants at once. A batch is one task on one object at one control period. A GPU world is one layout with its grasp, one
seed and one variant. The compiled scenes of two layouts differ only in the positions of the three mount bodies (checked
field by field). One model is therefore used for every layout, with one row of `body_pos` per world. The palm and the support
post are mocap bodies, because MuJoCo-Warp computes the poses of world-welded geoms once (memory
feedback_mjwarp_static_geoms_mocap.md). The controller is `HomController.step` rewritten in torch for all worlds and
fingers. A second MuJoCo-Warp data, the mirror, is set to the measured state; its kinematics give the tip poses, the
joint axes and anchors, and the gravity torques. Eq. (2) is solved exactly by trying the 27 patterns of active joint-rate
bounds. The scores come from `hom_tasks.metrics` on per-period rows of the CPU runner's format. The script imports
scripts/hom_tasks.py (plant constants, schedules, variants, metrics) and scripts/fp_hom_check.py (the 2-D layouts and the
task study's grasp rule).

Commands, each run as `.venv/bin/python scripts/hom_gpu.py <command>`:
- `run --task T [T ...] --layouts L [--d MM ...] [--max N] [--seeds S ...] [--variants V ...] [--batch 1024]
  [--gen-seed 0] --out NAME [--rows-dir DIR] [--append]`: the evaluator. Each batch of at most `--batch` worlds runs in
  its own spawned process (memory feedback_mjwarp_batch_per_process.md).
- `cpu-check --task T --layouts L [--n 40] [--offset 0] [--seeds S ...] [--variant V] --out NAME`: the CPU runner
  (`hom_tasks.rollout`) on every k-th layout of a population, with the GPU's layouts, grasps and open poses.
- `compare --cpu ROWS --gpu ROWS`: per task and variant, the share of runs with the same held outcome, the medians and
  the Spearman correlation of the score.
- `check-mounts`: confirms that mount positions built from offsets reproduce another layout's compiled scene.

Other scripts import `Scene`, `layout_list`, `variant_world`, `Batch` (`run`, `score`), `cpu_rollout` and `load_rows`;
scripts/fp_roll_sensing_check.py is a short example.

Tasks: `pinpush`, `press`, `roll` and `twist`. The fifth task of hom_tasks.py, `expa`, is not ported.

Variants are keys of `hom_tasks.VARIANTS`. The GPU runs those whose HomParams changes are limited to `dt`,
`contact_force_min` and `force_loop_gain` (`OVER_OK`): base, rate50, sense, floop, notip0, notip, tip, tip3, tipbias,
tipfl, tipflb, cal_notip, cal_tip and cal_tipfl. `variant_world` raises an error for nofilter, si, ell5, ell100, gapq and
surface. The variants l30 and l60 pass that test but would run the base grasp, because `layout_list("design")` does not
read the grasps of `design --lever`; do not run them on the GPU. Measurement delays of up to `DMAX` = 4 control periods
are supported, 80 ms at 50 Hz. The draws made once per rollout (calibration offsets, each fingertip's gain and offset)
equal the CPU's for the same seed: numpy `default_rng(1000 + seed)`, drawn in the CPU's order. The noise of each control
period comes from a torch generator seeded with `--gen-seed` plus the batch's first world index, so it differs from the
CPU's.

Layout sets (`--layouts`); the row files are in docs/experiments/20261029-foundational_pose_review/rows/:

| `--layouts` | Layouts read from | Layouts | Grasp |
|---|---|---|---|
| design | docs/experiments/20261021-hom_framework/design.json, through `hom_tasks.build` | one per task and size | the task study's; sizes from `hom_tasks.SIZES` or `--d` |
| 2d | rrt2d.jsonl, through `fp_hom_check.pick_layouts` | 212 of the 364 symmetric tripods | the task study's equator grasp on the 40 mm sphere; roll and twist only |
| 2dfp | rrt2d.jsonl, rrt2d_press.jsonl | 364 (press 340) | the toy's foundational poses |
| rrt6d | rrt6d.jsonl, rrt6d_press.jsonl | 2,500 (press 2,340) | the toy's foundational poses |
| sobol8k | sobol8k.jsonl, sobol8k_press.jsonl | 4,680 of 8,192 (press 4,358) | the toy's foundational poses |
| sobol6d | sobol6d.jsonl | 607 of 1,024 | the toy's foundational poses; never run, and no press pose file exists |

The toy's foundational poses come from scripts/fp_sr2_toy.py. The tripod on a 40 mm sphere is the roll's and the twist's
grasp. The pinch on a 25 mm cylinder is the pin-push's. The press pose is the same pinch with the middle finger 30 to 45
deg below the cylinder's equator.

Rows go to `<rows dir>/<out>.jsonl`, one JSON line per rollout, and `<out>_timing.jsonl`, one line per batch. The batch's
child process appends and fsyncs its rows when the batch ends. A killed batch therefore loses all of its rollouts, and
there is no resume per layout. A run without `--append` deletes both files first. A row has:
- the layout key: `d` (six mount coordinates, mm), `kin`, `kin_pinch` and, for the press, `kin_press` (the toy's
  kinematic scores) in the toy's sets; `x_sep_mm`, `y_sep_mm` and `kin` in 2d; `design`, `x_sep_mm` and `y_sep_mm` in
  design;
- `task`, `d_mm`, `seed`, `variant`, `status` ("ok") and `engine` ("mjwarp");
- the output of `hom_tasks.metrics`: `held`, `dropped`, `fingers_end`, `link_force_max_N` and `metrics_dt`; for the pinch
  tasks `theta_rmse_deg`, `theta_max_deg`, `theta_min_deg`, `strokes`, `gain_min`, `off_axis_max_deg`,
  `pusher_slip_max_mm`, `pinch_slide_max_mm`, `pusher_contact_frac` and `F_mean`; for the sphere tasks `w_rmse_dps`,
  `w_rmse_ratio`, `steps`, `drift_max_mm`, `centre_err_max_mm` and `centre_travel_ref_mm`; for the twist also
  `ramp_max_deg`, `ramp_three_pads_max_deg` and `ramp_end_deg`.

No per-period trace is kept. The CPU rows of hom_tasks.py also have `hom`, the HomParams. The GPU rows do not, and
`metrics_dt` gives their control period. A timing line has `task`, `d_mm`, `dt`, `batch`, `worlds`, `build_s`, `run_s`,
`score_s`, `layouts`, `gpu`, `max_mem_gb` and `when`. Its `max_mem_gb` counts torch's allocations only.

At about 2,000 worlds per batch on the RTX 4070 Ti SUPER, a rollout takes 24-25 ms in the twist and 15-16 ms in the roll.
It takes 25-26 ms in the pin-push and 21-22 ms in the press, at 100 Hz and at 50 Hz alike. Batches of 40 worlds take
0.6-0.75 s per rollout. Building a batch takes 3-7 s, and starting its process about 8 s. The CPU runner takes about
1.4 s per rollout on one core. 2,048 cylinder worlds use about 12 GB of the 16 GB GPU. Tonight's queues ran
`~/.claude/bin/resguard.sh run --mem 10G --cpu 300 --` with `--batch 2048`, `MUJOCO_GL=egl`, `NUMPY_MADVISE_HUGEPAGE=0`
and one `WARP_CACHE_PATH` per concurrent process. The exact commands are in logs/20261029-hom_gpu/queue1.sh to
queue6.sh (gitignored), next to their logs.

## How the object's shape and size enter

On the CPU (scripts/hom_tasks.py), `SHAPE` gives each task's shape: cylinders for the pin-push and the press, spheres for
the roll and the twist. `SIZES` gives the task study's diameters: 10, 16, 25 and 40 mm cylinders and 30, 40, 50 and
63 mm spheres. `build(task, d_mm)` reads the grasp from design.json under `dkey` (pinpush_cylinder25, sphere40 and so
on). It compiles the grasp's cell with `hand_object_scale_sim.cell_scenes(cell, shape, d_mm)`. It then sets the object
body's quaternion to `SPHERE_Q` (body z along palm -y). A cylinder's geom gets radius d/2 and half-length
`CYL_LEN / 2` = 50 mm. The setup dict passes `shape`, `d_mm`, `r` and `hl` on. `sr2_specs(setup)` makes the controller's
object surface from them, `surfaces.Sphere(r)` or `Cylinder(r, hl)`, which `hom.Mirror` uses for the contact frames. The
CPU's witness points come from `mj_geomDistance` between the block's convex mesh and the object.

On the GPU, `Scene(shape, d_mm)` compiles the same `cell_scenes` for one reference layout (x_sep 92.5 mm, y_sep
102.5 mm). It checks that the object geom's size is r, or r and 50 mm for a cylinder. It keeps r and hl for the torch
geometry: the analytic sphere, the cylinder's closest point (side, caps and rims), and the block-cylinder witness. That
witness is found by 24 alternating projections between the block's core box and the solid cylinder. A batch has one
shape and one size, because the object is compiled into the model. Each world sets its own mount positions
(`body_pos`), its palm offset (the mocap body `palm_pose` at `setup["palm"]`) and the object's start pose
(`setup["quat"]` in `qpos`).

In both runners the object weighs 25 g at every size (`hand_object_scale_sim.MASS`), with MuJoCo's inertia for that mass
and shape. Its centre starts at `Z_OBJ` = 120 mm. Seeds above 0 shift it by up to 1 mm in x and y and turn it by up to
2 deg about z (`Plant.reset`, repeated in the GPU's `_init_state`). The toy's sets use fixed objects (`D_FP`): the 25 mm
cylinder, 100 mm long (`fp_sr2_toy.CYL_D`, `CYL_L`), and the 40 mm sphere (`SPH_D`, also the 2d set's). The pinch and
press poses turn the cylinder about the vertical by the pose's yaw psi (`Rz(psi) SPHERE_Q`). `score()` turns the
object's axis back by psi before the pinch angle is measured. `--d` is accepted with a toy set. The scene then has
another object than the one the poses were solved for, so the toy must be solved at that size first.

## Validation status and known gaps

| Check | Rows (CPU; GPU) | Result |
|---|---|---|
| Twist and roll, 212 2-D layouts, equator grasp, 100 Hz, seeds 0-2 | hom2d_shard*.jsonl; hom2d_gpu_{twist,roll}.jsonl | same held outcome in 99.5 % of 636 rollouts per task; Spearman over rollouts 0.986 (twist ramp) and 0.993 (roll error) |
| Eleven sensing variants, the task study's grasps, 4 sizes, seeds 0-2, 50 Hz | HOM review rows/sensing.jsonl; sensing_gpu_val.jsonl | same held in 93 % of 528 runs (pin-push 89 %, press 87 %, roll 99 %, twist 95 %); medians 4.9 and 4.7 deg, 11.0 and 10.8 deg, 0.70 and 0.70, 0.53 and 0.53 |
| Pin-push and press, the same grasps, seeds 0-9, 100 Hz | cpu_val_base.jsonl; gpu_val_base.jsonl | same held in 90 % and 95 % of 40 runs; median RMSE per grasp within 0.36 deg |
| Twist and roll from the foundational poses, 40 Sobol layouts, seed 0 | cpu_check_sobol_{twist,roll}.jsonl; hom6d_gpu_{twist,roll}.jsonl | same held in 95 % and 88 %; Spearman 0.99 (ramp) and 0.96 (roll error) over the layouts both held |
| Pin-push from the foundational poses, 40 2-D layouts, seeds 0-1 | cpu_check_2dfp_pinpush.jsonl; hom2dfp_gpu_pinpush.jsonl | same held in 88 % (seed 0) and 75 % (seed 1); seeds 0 and 1 of one runner agree in 75 % (CPU) and 78 % (GPU) |

Known gaps:
1. Pin-push repeatability. On the toy's poses the pin-push's friction drive is seed-sensitive. On about a quarter of the
   2-D layouts, whether it keeps the cylinder depends on the start's jitter of up to 1 mm and 2 deg. Over the Sobol
   layouts, new seeds keep the pin-push's order at Spearman 0.65 and the press's at 0.82. With three seeds per layout,
   single layouts are ranked roughly, and the design page's four-task front has that noise.
2. The 0.01 s metrics step. Until a159c0faa, `hom_tasks.metrics` integrated the angular velocity and the roll's
   reference path with a fixed 0.01 s. Every 50 Hz row therefore reported half its twist and step angles. Rows scored
   since have `metrics_dt`. `hom_rows.rescale_metrics` corrects older rows when they are loaded. It drops the roll's
   centre error of older 50 Hz rows, which cannot be rescaled. Every reader of HOM rows must load them through it; the
   three builders and `hom_gpu.load_rows` do. The GPU rows without `metrics_dt` (hom2d_gpu_*, hom6d_gpu_{twist,roll},
   hom6d_rrt_gpu_{twist,roll}, written in the afternoon) are 100 Hz base runs and need no change.
3. Physics. The GPU runs in float32 with a solver tolerance of 1e-6 and 20 iterations, and its pads collide only with
   the object. Its cylinder witness and its per-period noise also differ from the CPU's. A single rollout's score can
   differ, most in the cylinder tasks. Held shares and rankings agree as in the table above.
4. `compare` keys rows by (task, size, variant, seed). It is correct only for the design set, which has one layout per
   task and size. The population checks are functions of scripts/fp_review_page.py: `pinch_validation`, `seed_check`,
   `sphere_check` and `roll_check`.
5. Subsets of layouts. `layout_list` reads whole populations, and `--max` takes the first n layouts. A run on chosen
   layouts, such as the four-task front, needs a row file in the toy's format added to `FP_ROWS`, or a filter on the key.
6. HomParams other than `dt`, `contact_force_min` and `force_loop_gain` cannot be changed on the GPU. Those three change
   only through named variants of `hom_tasks.VARIANTS`.
7. The GPU makes no films. Films come from the CPU runner (scripts/fp_hom_film.py, `hom_tasks.py run --film`).
8. A batch of 2,048 cylinder worlds needs about 13 GB of free GPU memory. Check `nvidia-smi` first, or use `--batch 1024`.

## Pages, builders and the data they read

| Page | Builder and figure scripts | Data read |
|---|---|---|
| docs/experiments/20261029-hom_control_review/20261029-hom_control_and_fingertip_sensing.html | scripts/hom_review_page.py and its template; figures scripts/hom_review_figures.py (schematic, planar, brake, filter, sensing) | in its rows/: sensing.jsonl (CPU, 528 runs), sensing_gpu_val.jsonl, cpu_val_base.jsonl and gpu_val_base.jsonl (Table 3), sensing_gpu.jsonl (Tables 4-5 and the sensing figure); the 10-21 task study's rows and posters; docs/experiments/20261026-fingertip_calibration/media/20261026-error_by_level.png |
| docs/experiments/20261029-foundational_pose_review/20261029-foundational_pose_hand_design.html | scripts/fp_review_page.py and its template; figures scripts/fp_review_figures.py (landscape, efficiency, embed, sobol, dyn2d, dyn6d, four2d, four6d, sense6d); films scripts/fp_hom_film.py | in its rows/: the toy's rows (rrt2d, rrt6d, sobol6d, sobol8k, the three press files, the summaries, embed6d.json, embed_sobol8k.json), every GPU row file with its timing file, the CPU checks (hom2d_shard*, hom_twist_check, cpu_check_*), figures.json, film_two_layouts.json and film_pinpush_two_layouts.json; the HOM review's cpu_val_base and gpu_val_base |
| docs/experiments/20261021-hom_framework/20261021-hom_tasks_across_object_sizes.html and 20261021-hom_controller_method_and_implementation.html | scripts/hom_framework_page.py and its template | docs/experiments/20261021-hom_framework/: design.json, rows/*.jsonl (the CPU task study, filter_sweep.jsonl, mpc_*.jsonl), rl_assessment.json, media/ |

Rebuild the figures first, then the page, with `.venv/bin/python`. scripts/fp_review_figures.py writes its numbers to
rows/figures.json, which scripts/fp_review_page.py reads. Every page build runs `retro_style` and `docs_index`, which
rewrite docs/experiments/INDEX.md (rows 28-29 for the 10-29 pages). The four pages are local files only; none has an
artifact. The HOM pages cite the papers as [1] and [2] and make no personal references (owner, 10-21).

## Tonight's runs and their purpose

Folder `hom_control_review` is docs/experiments/20261029-hom_control_review/rows/; folder `fp_review` is
docs/experiments/20261029-foundational_pose_review/rows/.

| Rows | Runs | Purpose |
|---|---|---|
| hom_control_review: sensing_gpu_val.jsonl | 528: the CPU's sensing runs, on the GPU | Table 3 (validation) |
| hom_control_review: cpu_val_base.jsonl, gpu_val_base.jsonl | 80 each: pin-push and press, 4 sizes, seeds 0-9, 100 Hz | Table 3 (validation) |
| hom_control_review: sensing_gpu.jsonl | 5,280: 4 tasks, 4 sizes, 11 variants, 30 seeds, 50 Hz | Tables 4-5, the sensing figure |
| fp_review: rrt2d_press, rrt6d_press, sobol8k_press | the press pose per toy candidate (`fp_sr2_toy.py press`) | the press runs' grasps |
| fp_review: hom2dfp_gpu_{twist,roll,pinpush,press} | the 2-D candidates from their foundational poses, seeds 0-2, 100 Hz | the four-task 2-D figure; the comparison of grasps against hom2d_gpu_{twist,roll} |
| fp_review: cpu_check_2dfp_pinpush | 40 layouts, seeds 0-1, CPU | pin-push repeatability |
| fp_review: hom6d_rrt_gpu_{pinpush,press}, hom6d_gpu_{pinpush,press} | RRT 2,500 and Sobol 4,680 layouts, seeds 0-2, 100 Hz | the four-task analysis with the afternoon's twist and roll rows (Spearman matrix, front, t-SNE) |
| fp_review: cpu_check_sobol_{twist,roll} | 40 Sobol layouts, seed 0, CPU | GPU check on the foundational poses |
| fp_review: hom6d_gpu_sense_{task} | 4,680 Sobol layouts; rate50, cal_notip, cal_tip and cal_tipfl; seeds 0-2; 50 Hz | sensing errors across the layouts (Section 8) |
| fp_review: hom6d_gpu_sense_{task}_s345 | rate50 with seeds 3-5 | the seed baseline for the reordering |
| fp_review: cpu_check_roll_poor | 12 roll layouts with poor tracking; rate50, notip0, notip, cal_notip; CPU | the roll's tracking under tracker noise and delay |
| fp_review: film_pinpush_two_layouts.json; fp_review media/20261029-hom_pinpush_two_layouts.mp4 and .jpg | CPU films of a held and a dropped pin-push | Section 6 of the design page |
| fp_review: figures.json, embed_sobol8k.json | scripts/fp_review_figures.py | the page's numbers and the t-SNE map |

The results, as the pages state them:
- With calibration errors and the measured fingertips, each task's order of the 4,680 Sobol layouts keeps a Spearman
  correlation of 0.45-0.72. New seeds keep 0.65-0.98. Of the best tenth, 26-58 % keep their place under calibration
  errors, against 66-90 % with new seeds.
- 178 of 4,358 layouts held every run of all four tasks. They cluster with the thumb and index mounts toward the palm's
  centre in x; 48 are on the four-task front.
- From the foundational poses the pin-push held in every seed on 94 % of the 2-D layouts with x_sep of 60 mm or less,
  and on 2 % beyond.
- From the foundational poses the roll tracked better with the tracker's noise and 40 ms delay than with the exact pose:
  1.41 against 2.76 per step on 12 CPU layouts. The cause is not known.
- Calibration errors raised the pin-push's pad force by 6.1 N over 7.4 N; the force loop removes the rise.
- On the HOM page, with calibration errors, the pinch force is off by a median 1.05 N in the pin-push and 1.10 N in the
  press over 30 seeds. With the force loop the errors are 0.31 N and 0.48 N.

## The sampling MPC

The sampling MPC on the HOM tasks is scripts/hom_tasks_mpc.py. Its planner is the `Planner` class of
scripts/hand_object_scale_mpc.py with the pinned iCEM settings of scripts/hand_object_scale_planners.py (2026-10-13/14).
A run starts as a HOM run: open pose, approach, commanded forces and lift under HOM until t = 2.0 s. From the task phase
on, the MPC replans every 10 ms.

The plan is the nine servo targets, three joints on each of three fingers, as a piecewise-linear spline with three knots
over a 0.5 s horizon. At each replanning the nominal plan is shifted to the new knot times. Candidates add Gaussian knot
noise of 0.08 rad, correlated across knots by AR(1) with coefficient 0.7, and are clipped to the servo ranges. Pinned
iCEM runs two iterations of 32 candidates and reuses the two best elites in the next replanning. The first knot is fixed
to the command being applied (`pin0`), which removed most of the command jitter in the scale study. Between replannings
the servo targets follow the spline at the 1 ms physics step. Candidates are rolled out with `mujoco.rollout` on the
plant's own pads model, compiled with one contact sensor per finger.

The cost is summed over 10 ms samples of the horizon, with the last sample weighted 10 times, plus a penalty on the
change between knots. For the pin-push and the press, two terms are the pinch-angle error over 2 deg and the rotation
off the pinch axis over 5 deg, each squared. A third is each finger's force shortfall below 0.2 N, as a fraction. For
the roll and the twist, two terms are the angular-velocity error over 10 deg/s and the centre's path error over 5 mm,
each squared. Their force term uses a floor of 0.5 N. A drop of more than 15 mm adds 10 per sample.

On 6 CPU threads (`--nthread 6`), one replanning took 0.3-2.8 s; the median per task was 1.3-2.4 s. The control period
is 10 ms. A rollout's task phase needed 290-580 replannings and took 2-21 min. The HOM controller takes about 1.1 ms per
control period on the CPU. scripts/hand_object_scale_mpc_gpu.py has a GPU version of the planner's rollouts
(`GPUPlanner`). With it, 1,024 plans of 0.5 s are rolled out in 1.78 s, 3.5 us per world-step. Its costs agree with the
CPU's at r 0.994. It was built for the scale study's turn and is not connected to hom_tasks_mpc.py.

Its results of 2026-10-22 are in docs/experiments/20261021-hom_framework/rows/mpc_<task>.jsonl and in the section
"MPC baseline" of 20261021-hom_tasks_across_object_sizes.html. There are 6 MPC and 6 HOM runs per task: pin-push and
press at 25 and 40 mm, roll and twist at 40 and 63 mm, seeds 0-2. Each controller held 21 of 24. Pinned iCEM tracked the
pinch angle to 0.5-0.6 deg, against HOM's 2.5-3.0 deg in the pin-push and 8.7-11.2 deg in the press. Under it the
pads pressed with 14-22 N each, and the finger links with up to 43 N. On the spheres it did no better than HOM: the
roll's centre left its path by 8-13 mm, against 3-6 mm.

## Steps planned next

From the pages' Future work, with the code each step needs:
1. Thirty seeds per task on the 48 layouts of the four-task front, keeping those whose rank agrees between seed halves at
   Spearman above 0.8. It needs a subset of layouts in hom_gpu.py (gap 5). It is about 5,800 rollouts, a few minutes of
   GPU.
2. Force-loop tuning on the design grasps: gains from 0.0025 to 0.02 m/s per newton and a cap on the closing speed,
   30 seeds per point, with calibration errors. It needs one variant per gain in `hom_tasks.VARIANTS` (or a `--set` flag
   in hom_gpu.py), and a HomParams field for the cap in hom.py with a test.
3. The press at fixed middle-finger latitudes of 37.5 and 45 deg on the Sobol layouts. It needs a fixed-latitude option
   in `fp_sr2_toy.py press`, then `hom_gpu.py run --task press`.
4. The roll's tracking: trace one layout's commanded and achieved twist at 50 Hz under rate50 and notip, then with the
   delay alone and the noise alone. It needs two new variants and the CPU runner, which records the trace.
5. Steps that need new poses or tools: the screwdriver's pinch brake and wield as foundational poses in fp_sr2_toy.py;
   the t-SNE map against a uniform null with scripts/paper_morphology_diversity.py.
6. Steps that need the bench or the owner: three layouts from the front, run on the gantries with tags; the
   fingertip's latency and the zero of a moving tip, measured on the UR5; one tip on the index for a bench pin-push.
7. The HOM page's simulation steps: a recorded press session replayed through the estimator in hom_bench.py, and the
   calibration-error variants on the SCS0009 plant.
8. Publishing the 10-29 pages as artifacts, if the owner wants them; that runs on the personal login.

## Memory files written today

All are in /home/humanoid/.claude/projects/-home-humanoid-Programs-hand/memory/:
- project_hom_fp_review_2026_10_29.md: the two pages, the GPU evaluator and tonight's results.
- feedback_mjwarp_static_geoms_mocap.md: MuJoCo-Warp computes the poses of world-welded geoms once, so the palm and the
  post are mocap bodies.
- feedback_mjwarp_batch_per_process.md: one spawned process per batch; a finished batch's captured graphs kept its GPU
  memory.
- project_hom_metrics_50hz_rescale.md: the 0.01 s integration step and `hom_rows.rescale_metrics`.
- feedback_pgrep_self_match.md: a paragraph added on waiters. `pgrep -f "[q]ueue1.sh"` returned a leftover `bash -c`
  wrapper whose argv contains the pattern, so a waiter checks a PID with `kill -0`.
