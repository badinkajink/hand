Status: active (window 8955e04e, not armed)
Task: owner 2026-11-12: "test how map-elites and/or another outer loops optimizer does"; "evaluate some of the other
inner loop controls we tried? Karma, uhas? If we have to train our own uhas, that's fine, esp since we have so many
demonstrations already"; then "in an ideal world we can show these other methods (... published ...) are both worse and
slower than ... gpu parallelized HOM. We can let the other methods have the sphere packed model and warp".
Settings: the talk's (SR2 cad_v2, v4b pads at 2.5 MPa, SCS0009 plant 0.72 N m/rad, MuJoCo-Warp 3.6, scenes
logs/20261102-hom_servo_plant/scenes), grids rows/grid2dff_v4b_sv25r, J = hom_landscape.track_score.
Read first: docs/experiments/20261029-foundational_pose_review/20261109-hom_search_methods.html Section 12;
scripts/hom/hom_search_qd.py; scripts/hom/karma_sr2.py; UHAS paper docs/uhas/paper/sections/03_method.tex and
appendix.tex (observations, PPO table, rewards); UHAS code docs/uhas/UHAS_sim/sphere_ctrl_isaaclab/.../tasks/
(UHAS_inhand/multi_manipulation_env.py, utils/sphere_torch_utils.py); memory project_uhas_integration.md (August).
State 12:40: outer loop done (eb98da82e): replays on the four J grids, 200 runs per method. By 6,000 rollouts BO came
within 0.05 of the best J in 94-100 % of runs (0.63-0.85 of the EA's median rollouts), EA64 88-100 %, CMA-ES 80-100 %,
MAP-Elites 49-99 %, random 52-99 %. KaRMA: external/karma cloned (c8d3698); one layout 163 s on one core, 0.42 GB; HOM
1.7-2.8 s per layout on the GPU (48 rollouts with build and scoring). logs/20261112-karma_sr2/queue.sh running since
12:38 (sph40 145 layouts, then sph25 128, 8 jobs, ~95 min); rows docs/experiments/20261112-inner_loop_benchmark/rows/.
State 12:50: UHAS port written and checked: scripts/hom/uhas_sr2_cik.py (sphere, deformation, cascade IK; FK equals
MuJoCo's to 0.0000 mm), scripts/hom/uhas_sr2.py (HOM's grasp and lift, then the policy; per-world resets from a pool of
HOM handovers; RSL-RL PPO with UHAS's table). logs/20261112-uhas_sr2/train_twist.sh running since 12:44: twist, 2,048
worlds, pool 6,756 of 8,192 held, 1.71 s per iteration, 120 min (~4,150 iterations, 136 M transitions), then eval of the
109 test layouts (rows/uhas_twist_<run>_test.jsonl). KaRMA sph40 61/145 at 12:48 (runs 150-580 s beside the GPU job).
Plan: one result page docs/experiments/20261112-inner_loop_benchmark/20261112-inner_loop_benchmark.html (builder
scripts/hom/inner_loop_benchmark_page.py): HOM vs KaRMA vs UHAS (and per-layout PPO = UHAS's single-hand baseline),
each judged by J of the hand it would pick and by wall time per layout on this box.
Claimed: scripts/hom/hom_search_qd.py, karma_sr2.py, uhas_sr2*.py, inner_loop_benchmark_page.py,
docs/experiments/20261112-inner_loop_benchmark/, logs/20261112-*.
Do not: GPU jobs beside a running GPU queue; git lfs push --all; change the talk decks without the owner.
