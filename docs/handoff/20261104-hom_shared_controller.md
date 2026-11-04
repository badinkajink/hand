Status: done
Task: Owner, 2026-11-03: "is it possible to co-design HOM? are there hyperparameters / parameters that we can change
within it?"; HOM fixed "gives us a fixed reference to evaluate morphologies"; 18:1x: "set up different autopilot jobs to
continue promising work". Tune HOM's forces and contact modes once for all layouts (a shared controller, still one fixed
reference), rescore the SCS0009 grids with it, and compare that landscape with the default HOM's.
Read first: docs/handoff/20261104-hom_score_refine_page.md (State, its Go/no-go line); the score refine page and README
in docs/experiments/20261103-hom_score_refine/; scripts/hom_grasp_refine.py; scripts/hom_plant_transfer.py (the landscape
comparisons); scripts/hom_gpu.py cmd_run.
State: done 11-04 12:4x (autopilot, colorado). Code 67e2b8fe4 (level shared, validate/report --shared, prm in
hom_grasp_refine.py; --prm in hom_gpu.py run). Queue logs/20261104-hom_shared_controller/queue.sh 09:36-11:52, no failed
step (copy queue_shared.sh beside the page). The 10 random search layouts are drawn from the sample's 30 with numpy
seed 0 (its "first 10" are the 10 narrowest). Searches at seeds 0-2, mean over 40 layouts: twist and roll keep HOM's
defaults, pin-push 0.644 -> 0.648, press 0.621 -> 0.776 (pinch 1.73 N, pusher roll weight 0.35). Validated, 30 best,
Sh - S1: 0.000, 0.000, +0.005, +0.19 (S3e - S1 +0.11, +0.05, +0.06, +0.20); press random 30: +0.26 of +0.38. Press
grid rho 0.47, best layout (70,150) -> (65,140). Twist/roll grids are reruns at the defaults: 5-13 % of rollouts move
> 0.02, rho 0.96/0.98 (the GPU's spread for a landscape). CPU: 3.0/2.0 s a rollout (pin-push/press), 68-69 cores per
GPU; 6/6 press rollouts within 0.05 of the GPU with every contact. Page Section 7 (prose within targets), README section,
memory; commits 324780eda (rows, grids) and a7993fbfc (page), pushed as a54ef4bce.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Go/no-go from the page job's line: if the validated S3 - S2 is under 0.03 in median on all four tasks, per-la… Done 11-04 09:26: Go 11-04 09:2x: validated S3e - S2e median twist +0.002, roll +0.019, pin-push +0.036, press +0.143 (pin-push and press above 0.03); scripts/hom/ absent, scripts stay at scripts/hom_*.py (package morphohand)
- [x] 2. Add level `shared` to hom_grasp_refine.py: CEM over the forces and mode weights only (S3's ranges), each cand… Done 11-04 10:51: Done 11-04 10:50: level shared (67e2b8fe4) run on the 4 tasks, 24 candidates x 8 its on 40 SCS0009 layouts with S1's grasps at seeds 0-2 (2,880 rollouts an iteration, 93-160 s); mean over the 40 layouts, HOM's defaults -> pick: twist 0.745 (defaults kept), roll 0.540 (defaults kept), pin-push 0.644 -> 0.648 (noise level), press 0.621 -> 0.776 (pinch 3 -> 1.73 N, pusher roll weight 1 -> 0.35, lowest layout 0.03 -> 0.31)
- [x] 3. Add `--prm JSON` to hom_gpu.py's run command (every world gets that prm) and rescore each task's grid at the … Done 11-04 11:54: Done 11-04 11:49: hom_gpu.py run --prm (67e2b8fe4); the 4 grids rescored with each task's shared set, rows grid2dff_v4b_sv25_shared (HOM_ROWS_TAG _sv25_shared), 10:50-11:49, 89,280 rollouts. Twist and roll keep HOM's defaults, so their grids are reruns: rho 0.96/0.98, best tenth kept 0.90/0.92, same best grasp 0.79/0.82 (the evaluator's run-to-run spread). Pin-push rho 0.96, best tenth 0.67; press rho 0.47, best tenth 0.28, median best-of-16 0.42 -> 0.78, best layout (70,150) -> (65,140)
- [x] 4. Compare the default and shared landscapes as hom_plant_transfer.py does (Spearman of the best-of-16 maps, bes… Done 11-04 12:05: Done 11-04 12:4x: Section 7 'Forces and contact modes shared by all layouts on the SCS0009' of docs/experiments/20261103-hom_score_refine/20261103-hom_score_refine.html (tables 11-13, figures 6-7, films 3-4 of press layouts (100,75) gain and (85,115) loss on the CPU plant); top-30 Sh - S1 twist 0.000, roll 0.000, pin-push +0.005, press +0.19 (S3e - S1 +0.11/+0.05/+0.06/+0.20); press grid rho 0.47, twist/roll reruns rho 0.96/0.98 (the GPU spread); compute 183,840 rollouts, 2.07 h; CPU 68-69 cores per GPU; commits 324780eda, a7993fbfc pushed (a54ef4bce); memory and README updated
Standing directions (for every step):
- Compute tracking (owner 11-03 19:0x: "let's make sure we're timing this all for compute-tracking purposes and throughput measuring, including cpu vs gpu speedups"): run every GPU batch through hom_gpu.batch_child so each writes its _timing.jsonl line beside its rows under docs/experiments/20261103-hom_score_refine/rows/; log each search iteration's wall, draw and GPU-batch times (and write them as JSON lines if you add a script); after the runs, `python3 scripts/hom_compute.py ledger` and `report`, and give the job's rollouts, batch hours and ms per rollout (run, batch, whole step) in its page section; for a new kind of rollout (another plant, the friction band, a new search), time the CPU runner on it with `scripts/hom_compute.py cpu-bench` (through resguard) and state the CPU against GPU speedup.
- Every search scores its start: candidate 0 of iteration 0 is the start itself (the page job's fix to hom_grasp_refine.draw_layout after the queue; check it is in before running), and a refined score is quoted beside its start's at the same seeds. On 11-03 the search without it ended below its start on 7 of 240 sv25 layouts (worst -0.169, pin-push).
<!-- /ap:steps -->
Claimed: scripts/hom_grasp_refine.py (level shared), scripts/hom_gpu.py (the --prm option),
docs/experiments/20261029-foundational_pose_review/rows/grid2dff_v4b_sv25_shared/, the page's shared-controller section.
Waiting on owner: whether the press's shared set (pinch 1.73 N, pusher roll weight 0.35) replaces HOM's press
defaults as the SCS0009 reference of later grids and searches; hom_tasks.PRM is unchanged.
Do not: publish artifacts; run beside another GPU job; change the defaults in hom_tasks.PRM (the grids' reference).
