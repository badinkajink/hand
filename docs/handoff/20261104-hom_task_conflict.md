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
best of 561): validate first. 22:4x, performance only (direction 9): coverage at a level alpha of each task's best.
SCS0009 lattice: alpha*_1 0.83 (the best generalist's worst task 0.835), alpha*_2 0.89, alpha*_3 0.93; generalists 30
at alpha 0.6, 7 at 0.7, 1 at 0.8, none at 0.9 (K* 3, three mutually incompatible tasks). alpha*_1 on the other
plants: STS3032 0.87, friction 0.4 0.79, work plant 0.85; the STS3032 at 0.9 has no incompatible pair yet needs K* 2.
Size grid (16 tasks, 8,182 Sobol layouts): alpha*_1 0.51, alpha*_3 0.80, alpha*_7 0.90; K* 7 at 0.9 and 11 at 0.95,
each equal to the largest mutually incompatible set; K* 3 against a bound of 2 at 0.7. One layout over the four twist
sizes (30-63 mm) reaches 0.60 of each size's best, over the four press sizes (10-40 mm) 0.88.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [ ] 1. Paths: after the scripts switch the scripts are under scripts/hom/ and the package is sr2hand (`git log -1 -- scripts/hom`).
    Add `validate --plant P` to hom_task_conflict.py: every layout of the grid on every task it reaches, with its S1 grasp,
    at seeds 20-24, through hom_grasp_refine.setenv and evaluate (hom_gpu.batch_child; rows
    docs/experiments/20261103-hom_task_conflict/rows/<plant>/val/<task>_d<d>.jsonl with their _timing lines), launched
    through resguard (--mem 5G; retry on exit 3 as logs/20261103-hom_score_refine/queue.sh's job() does). About 10,000
    rollouts per plant; run sv25, k21 and k21mu0p4.
- [ ] 2. Add `--val` to scope: choose the specialists, blends, K-sets and pair curves at seeds 0-2 and score them at seeds
    20-24; also the choice made at seeds 20-24 alone. Add a smoothed variant (each layout's score the mean over its 3 x 3
    lattice block), standing for seed noise and the gantries' placement error. Write per plant J_K, J_T/J_1, the pair
    quantities and the break-even run length N T_e = T_s / (J_K / J_1 - 1) to the JSON.
- [ ] 3. Blending through the grasp and HOM's parameters, on sv25: the four specialists, the J_1 layouts (raw and normalized),
    the max-min layout and the K = 2 and 3 layouts (at most 10). For each task a layout reaches, write <task>_layouts.json
    in hom_grasp_refine's format under docs/experiments/20261103-hom_task_conflict/rows_refine/sv25/ and run its levels
    grasp and full and validate there, from a wrapper in hom_task_conflict.py that sets hom_grasp_refine.ROWS_ROOT
    (layouts() reads an existing sample file; no edit to hom_grasp_refine.py). Recompute J_K at the validated S2 and S3:
    does each task's own grasp, forces and modes on one shared layout close the gap to the refined specialists?
- [ ] 4. Object sizes in the six gantry coordinates (CPU only): rows/sizes_v4bff under
    docs/experiments/20261029-foundational_pose_review/ (pin-push and press at 10, 16, 25 and 40 mm, twist and roll at
    30, 40, 50 and 63 mm, on the 8,192 Sobol layouts; work plant at 10 MPa). If one layout set is scored at every size,
    run the scope with (task, size) as the objectives: how many layouts does a range of object sizes need?
- [ ] 5. Page: scripts/hom_task_conflict_page.py from scripts/hom_score_refine_page.py's components (topic design), output
    docs/experiments/20261103-hom_task_conflict/<date>-hom_task_conflict.html: Terms and metrics; method (J_K, pair
    curves, dominance margin, the rule, in LaTeX); the SCS0009 (payoff table of specialists by task, J_K against K, maps
    of each task's score over x_sep and y_sep with the specialists and the blend marked, the pair fronts with the
    straight segment); the other plants; the refined layouts; sizes; the rule with the gantry times and episode
    lengths; limitations; future work. Films of the J_1 layout and each task's specialist doing the task (CPU clips
    through resguard, hom_plant_page.compose). Read every figure at its displayed size; prose_lint within targets;
    banned words searched.
- [ ] 6. Record: README.md in the study folder (runbook), memory, commit and push.
Standing directions (for every step):
- Compute tracking (owner 11-03 19:0x: "let's make sure we're timing this all for compute-tracking purposes and throughput measuring, including cpu vs gpu speedups"): run every GPU batch through hom_gpu.batch_child so each writes its _timing.jsonl line beside its rows; log each run's wall, draw and GPU-batch times as JSON lines; after the runs, `python3 scripts/hom_compute.py ledger` and `report` (add this study's rows to the ledger's roots if they are missing), and give the job's rollouts, batch hours and ms per rollout in its page section.
- Every search scores its start: candidate 0 of iteration 0 is the start itself (the page job's fix to hom_grasp_refine.draw_layout after the queue; check it is in before step 3), and a refined score is quoted beside its start's at the same seeds.
- Performance only (owner 11-03 22:3x: "we shouldnt consider time, only performance. for desired performance, we should be able to mathematically formulate the blend vs switch decision, given the current layout"; "there can be many capable generalists"; "not saying i expect you to solve that here with experiments, nor this project to answer that either. that's just one ultimate goal" (the specialist-generalist question at scale)). No time, gantry travel or score-per-second rule anywhere: steps 2 and 5's break-even run length and gantry times are dropped. The decision is the coverage formulation of scripts/hom_task_conflict.py's docstring (window commit of 22:4x): capable tasks C(x) at a desired level alpha of each task's best, generalists G and N_G, the decision at a current layout x0 (stay, move once to a generalist, or switch with the fewest extra layouts), K* as an exact set cover, alpha*_K, incompatible pairs and the largest mutually incompatible set. Report these at the validated scores (step 2) and at S2/S3 on shared layouts (step 3: does task-specific grasp and control enlarge G?). Step 4 is done by the window (20261103-conflict_sizes.json, 16 tasks on 8,182 Sobol layouts): check it, put it on the page as the breadth example, run nothing new. Launch no new grids for more sizes or task types.
- Related work for the page (window 220623e7, 11-03 23:0x; owner: "have other works thought about this problem? ... ideally the closer to robotics the better"). Gripper sets as a set cover: Xu, Wan, Koyama, Domae, Harada, Advanced Robotics 2021 (doi 10.1080/01691864.2020.1870047, arXiv 2003.04087). Multi-function fingers in place of tool changers: Honarpardaz, Tarkian, Olvander, Feng, RAS 87 (2017) review and RAS 113 (2019). Pareto structure of hand designs: Wang, Oh, Pollard, arXiv 2409.14158 (the foundational pose). Task sets specialise evolved hands: Yang et al., IEEE TEVC 2026 (doi 10.1109/TEVC.2026.3696420). Pareto sets of robot designs over two or three tasks: Xu, Spielberg, Zhao, Rus, Matusik, ICRA 2021 (arXiv 2107.05858). A design selected per terrain: Whitman, Travers, Choset, NeurIPS 2020 workshop; Hu et al. 2022 (GAN). Mode switching on one robot: Sihite et al., Nat. Commun. 2023 (M4). One solution per task over thousands of tasks: Mouret and Maguire, GECCO 2020 (arXiv 2003.04407). Generalist morphology from an information objective: Hejna, Abbeel, Pinto, ICLR 2021 (TAME, arXiv 2102.13100); Gupta et al., Nat. Commun. 2021 (DERL). Theory: Levins 1962 (Am. Nat., fitness sets: convex -> generalist, concave -> specialists) and Rueffler, Van Dooren, Metz 2006 (Am. Nat.); Shoval et al., Science 2012 (Pareto archetypes: optimal phenotypes lie between the single-task archetypes, which assumes performance falls with distance from each archetype; the HOM landscapes are multi-peaked at 5 mm, and that is why the straight segment fails); Standley et al., ICML 2020 (which tasks share a network under a budget: the same partition as K layouts); Kussell and Leibler, Science 2005, and Donaldson-Matasci, Bergstrom, Lachmann, Oikos 2010 (switching versus sensing when the next environment is unknown; the fitness value of a cue is bounded by its mutual information with the environment); Kolmogorov and Tikhomirov 1959 (epsilon-entropy: log2 K*(alpha) is the logarithm of a covering number). Read each paper before citing a claim from it.
<!-- /ap:steps -->
Claimed: scripts/hom_task_conflict.py, scripts/hom_task_conflict_page.py, docs/experiments/20261103-hom_task_conflict/.
Waiting on owner: none.
Do not: publish artifacts; run beside another GPU job; edit scripts/hom_grasp_refine.py, hom_gpu.py or hom_tasks.py
(other jobs' files); change the grid rows under docs/experiments/20261029-foundational_pose_review/rows/.
