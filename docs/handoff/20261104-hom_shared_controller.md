Status: active
Task: Owner, 2026-11-03: "is it possible to co-design HOM? are there hyperparameters / parameters that we can change
within it?"; HOM fixed "gives us a fixed reference to evaluate morphologies"; 18:1x: "set up different autopilot jobs to
continue promising work". Tune HOM's forces and contact modes once for all layouts (a shared controller, still one fixed
reference), rescore the SCS0009 grids with it, and compare that landscape with the default HOM's.
Read first: docs/handoff/20261104-hom_score_refine_page.md (State, its Go/no-go line); the score refine page and README
in docs/experiments/20261103-hom_score_refine/; scripts/hom_grasp_refine.py; scripts/hom_plant_transfer.py (the landscape
comparisons); scripts/hom_gpu.py cmd_run.
State: 11-04 09:3x (autopilot, colorado). Step 1 go (S3e - S2e pin-push +0.036, press +0.143). Level shared, validate
--shared, report --shared, prm in hom_grasp_refine.py; --prm in hom_gpu.py run (not committed yet). The sample stores its
random layouts in lattice order, so "first 10 random" would be the 10 narrowest: shared_layouts takes 10 of the 30 at
random (numpy seed 0) instead; say so on the page. Smoke 09:33-09:36 (rows_smoke/sv25/shared, rows_smoke/grid_shared):
--prm at default values = default grid (96 rollouts, max 0.0004); the search's start = grid S1 within the GPU spread.
Queue logs/20261104-hom_shared_controller/queue.sh launched 09:36:29 (nohup setsid; queue.log, Q_<step>.log/.done):
searches, grids rescored, validate --shared, report; expected done about 12:00. Do not edit hom_grasp_refine.py or
hom_gpu.py while it runs.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Go/no-go from the page job's line: if the validated S3 - S2 is under 0.03 in median on all four tasks, per-la… Done 11-04 09:26: Go 11-04 09:2x: validated S3e - S2e median twist +0.002, roll +0.019, pin-push +0.036, press +0.143 (pin-push and press above 0.03); scripts/hom/ absent, scripts stay at scripts/hom_*.py (package morphohand)
- [ ] 2. Add level `shared` to hom_grasp_refine.py: CEM over the forces and mode weights only (S3's ranges), each candidate run
    on a fixed set of 40 layouts (the sv25 sample's 30 top and its first 10 random) with their S1 grasps at seeds 0-2,
    scored by the mean over layouts (record the lowest too); 24 candidates, 8 iterations, about 25 min per task.
- [ ] 3. Add `--prm JSON` to hom_gpu.py's run command (every world gets that prm) and rescore each task's grid at the default
    object with the shared prm: rows grid2dff_v4b_sv25_shared (HOM_ROWS_TAG _sv25_shared), about 20 min per task.
- [ ] 4. Compare the default and shared landscapes as hom_plant_transfer.py does (Spearman of the best-of-16 maps, best tenth
    kept, same best grasp, each one's best layout on the other), and run the shared prm's best-of-16 picks of the sample
    at seeds 20-24 beside S1 and S3. Add a section to the score refine page; commit; push; memory and README.
Standing directions (for every step):
- Compute tracking (owner 11-03 19:0x: "let's make sure we're timing this all for compute-tracking purposes and throughput measuring, including cpu vs gpu speedups"): run every GPU batch through hom_gpu.batch_child so each writes its _timing.jsonl line beside its rows under docs/experiments/20261103-hom_score_refine/rows/; log each search iteration's wall, draw and GPU-batch times (and write them as JSON lines if you add a script); after the runs, `python3 scripts/hom_compute.py ledger` and `report`, and give the job's rollouts, batch hours and ms per rollout (run, batch, whole step) in its page section; for a new kind of rollout (another plant, the friction band, a new search), time the CPU runner on it with `scripts/hom_compute.py cpu-bench` (through resguard) and state the CPU against GPU speedup.
- Every search scores its start: candidate 0 of iteration 0 is the start itself (the page job's fix to hom_grasp_refine.draw_layout after the queue; check it is in before running), and a refined score is quoted beside its start's at the same seeds. On 11-03 the search without it ended below its start on 7 of 240 sv25 layouts (worst -0.169, pin-push).
<!-- /ap:steps -->
Claimed: scripts/hom_grasp_refine.py (level shared), scripts/hom_gpu.py (the --prm option),
docs/experiments/20261029-foundational_pose_review/rows/grid2dff_v4b_sv25_shared/, the page's shared-controller section.
Waiting on owner: none.
Do not: publish artifacts; run beside another GPU job; change the defaults in hom_tasks.PRM (the grids' reference).
