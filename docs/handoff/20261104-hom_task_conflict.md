Status: active
Task: Owner, 2026-11-03 20:3x: "how can we accommodate tasks with conflicting morphology objectives? can we linearly
interpolate? if not, is there some rule we can establish (pareto front) for when blending, however we do it, is
optimal, vs when we really should switch morphologies". The SR2 hand's gantries move the finger bases, so the layout
can change between tasks. This job measures what one layout for the four HOM tasks loses against one layout per task,
whether a layout between two tasks' best ones works, and states the rule with the gantries' travel time.
Read first: scripts/hom_task_conflict.py (docstring: the quantities and the rule); the scope JSONs
docs/experiments/20261103-hom_task_conflict/20261103-conflict_<plant>.json; docs/experiments/20261103-hom_score_refine/README.md
and scripts/hom_grasp_refine.py (S1, S2, S3, evaluate, the page job's elitist fix); the plant page
docs/experiments/20261029-foundational_pose_review/20261102-hom_plant_modulus.html (style); ~/.claude/writing_reference.md;
memory project_hom_codesign_levels_2026_11_03.md.
State: 11-03 20:5x. Scope from the grids (S1 at seeds 0-2, 561 layouts of the 5 mm lattice): on the SCS0009 one
layout (x_sep 70, y_sep 110 mm) gives a mean task score of 0.738, against 0.803 with one layout per task (J_4/J_1
1.09); it reaches 0.913 of each task's best on average and 0.835 on its worst task. J_2 0.765, J_3 0.789. J_4/J_1 on
the other grids: STS3032 1.10, the same at friction 0.4 1.17, work plant 1.12. The straight line between two tasks'
best layouts passes through layouts that another layout beats on both tasks by 0.10-0.66. Neighbouring layouts 5 mm
apart differ by up to 0.5 on one task, so seed noise inflates these numbers, the specialists' scores most (each the
best of 561): validate first. Gantry travel between the specialists (25-115 mm apart): 2.8-18.4 s.
Next:
1. Paths: after the scripts switch the scripts are under scripts/hom/ and the package is sr2hand (`git log -1 -- scripts/hom`).
   Add `validate --plant P` to hom_task_conflict.py: every layout of the grid on every task it reaches, with its S1 grasp,
   at seeds 20-24, through hom_grasp_refine.setenv and evaluate (hom_gpu.batch_child; rows
   docs/experiments/20261103-hom_task_conflict/rows/<plant>/val/<task>_d<d>.jsonl with their _timing lines), launched
   through resguard (--mem 5G; retry on exit 3 as logs/20261103-hom_score_refine/queue.sh's job() does). About 10,000
   rollouts per plant; run sv25, k21 and k21mu0p4.
2. Add `--val` to scope: choose the specialists, blends, K-sets and pair curves at seeds 0-2 and score them at seeds
   20-24; also the choice made at seeds 20-24 alone. Add a smoothed variant (each layout's score the mean over its 3 x 3
   lattice block), standing for seed noise and the gantries' placement error. Write per plant J_K, J_T/J_1, the pair
   quantities and the break-even run length N T_e = T_s / (J_K / J_1 - 1) to the JSON.
3. Blending through the grasp and HOM's parameters, on sv25: the four specialists, the J_1 layouts (raw and normalized),
   the max-min layout and the K = 2 and 3 layouts (at most 10). For each task a layout reaches, write <task>_layouts.json
   in hom_grasp_refine's format under docs/experiments/20261103-hom_task_conflict/rows_refine/sv25/ and run its levels
   grasp and full and validate there, from a wrapper in hom_task_conflict.py that sets hom_grasp_refine.ROWS_ROOT
   (layouts() reads an existing sample file; no edit to hom_grasp_refine.py). Recompute J_K at the validated S2 and S3:
   does each task's own grasp, forces and modes on one shared layout close the gap to the refined specialists?
4. Object sizes in the six gantry coordinates (CPU only): rows/sizes_v4bff under
   docs/experiments/20261029-foundational_pose_review/ (pin-push and press at 10, 16, 25 and 40 mm, twist and roll at
   30, 40, 50 and 63 mm, on the 8,192 Sobol layouts; work plant at 10 MPa). If one layout set is scored at every size,
   run the scope with (task, size) as the objectives: how many layouts does a range of object sizes need?
5. Page: scripts/hom_task_conflict_page.py from scripts/hom_score_refine_page.py's components (topic design), output
   docs/experiments/20261103-hom_task_conflict/<date>-hom_task_conflict.html: Terms and metrics; method (J_K, pair
   curves, dominance margin, the rule, in LaTeX); the SCS0009 (payoff table of specialists by task, J_K against K, maps
   of each task's score over x_sep and y_sep with the specialists and the blend marked, the pair fronts with the
   straight segment); the other plants; the refined layouts; sizes; the rule with the gantry times and episode
   lengths; limitations; future work. Films of the J_1 layout and each task's specialist doing the task (CPU clips
   through resguard, hom_plant_page.compose). Read every figure at its displayed size; prose_lint within targets;
   banned words searched.
6. Record: README.md in the study folder (runbook), memory, commit and push.
Claimed: scripts/hom_task_conflict.py, scripts/hom_task_conflict_page.py, docs/experiments/20261103-hom_task_conflict/.
Waiting on owner: none.
Do not: publish artifacts; run beside another GPU job; edit scripts/hom_grasp_refine.py, hom_gpu.py or hom_tasks.py
(other jobs' files); change the grid rows under docs/experiments/20261029-foundational_pose_review/rows/.
