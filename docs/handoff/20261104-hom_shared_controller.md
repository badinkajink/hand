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
hom_gpu.py while it runs. Code committed and pushed 67e2b8fe4. 09:55: twist search done (8 its, 140 s an iteration of
2,880 rollouts, peak 2.44 GB): no candidate beats HOM's defaults (mean 0.745 over the 40 layouts, lowest layout 0), so
the twist's rescored grid repeats the default one (the pipeline's null control). The defaults' 120 twist rollouts
re-run in the search's batches: 10 differ from the grid by > 0.02, one drops (110 x 110, seed 0; grid 94 deg). 10:07:
roll search done (93 s an iteration): pick = HOM's defaults again (0.540, lowest layout 0.190); candidates near the
defaults score 0.53-0.54. 10:33: pin-push search done (10:12-10:33 after the gate, 160 s an iteration): pick it 7
cand 10, 0.648 against the defaults' 0.644 (pinch 2.89 N, drive 1.56 N, pinch slide 0.82, pusher spin 0.87, roll 0.84);
per layout +0.004 median, 1 up and 1 down by 0.05. 10:50: press search done (10:35-10:50 after the gate; 110 s an
iteration): 0.621 -> 0.776 (lowest layout 0.03 -> 0.31), pinch 1.73 N, press 0.53 N, pusher roll 0.35, spin 0.81, pinch
slide 0.88: as S3's per-layout medians. Step 2 done. Grids rescoring from 10:50 (twist, roll, pin-push, press).
11:09: twist grid done (18 min). It reruns the default grid (same set, worlds and batches): 1,335 of 25,296 rollouts
differ by > 0.02, 146 by > 0.5, median 0; compare(): rho 0.963, best tenth kept 0.90, same best grasp 0.79, best layout
(100, 70) on both. The code change since 11-02 is the weights' storage only, so this is the GPU's run-to-run spread:
quote the twist and roll reruns as the noise floor of the landscape comparison. 11:52: queue done, no failed step
(report rows/sv25/20261104-shared_levels.json); step 3 done. Top 30 at seeds 20-24, Sh - S1: twist 0.000, roll 0.000,
pin-push +0.005, press +0.19 (S3e - S1 +0.11, +0.05, +0.06, +0.20; press 94 %, random 30: 0.26 of 0.38). Press grid
rho 0.47, best tenth 0.28, median best-of-16 0.42 -> 0.78, best layout (70,150) -> (65,140). CPU bench (pin-push,
press shared sets; cpu_bench.sh) 3.0 / 2.0 s a rollout, 68-69 cores per GPU; press CPU every contact 6/6 within 0.05
of the GPU. Films: press (100,75) gain and (85,115) loss, CPU reproduces both. ledger + report rerun. 12:2x: Section 7
of the page built (prose within targets); next commit + push, README outcome, memory, step 4, disarm. Page builder: sec_shared, shared-figs,
shared-films, terms, limits, files written (not run); lede, future work, README, memory after the report. After the
queue: shared_cpu_bench.py (logs folder; copy beside the page) on the idle box, then ledger + report.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Go/no-go from the page job's line: if the validated S3 - S2 is under 0.03 in median on all four tasks, per-la… Done 11-04 09:26: Go 11-04 09:2x: validated S3e - S2e median twist +0.002, roll +0.019, pin-push +0.036, press +0.143 (pin-push and press above 0.03); scripts/hom/ absent, scripts stay at scripts/hom_*.py (package morphohand)
- [x] 2. Add level `shared` to hom_grasp_refine.py: CEM over the forces and mode weights only (S3's ranges), each cand… Done 11-04 10:51: Done 11-04 10:50: level shared (67e2b8fe4) run on the 4 tasks, 24 candidates x 8 its on 40 SCS0009 layouts with S1's grasps at seeds 0-2 (2,880 rollouts an iteration, 93-160 s); mean over the 40 layouts, HOM's defaults -> pick: twist 0.745 (defaults kept), roll 0.540 (defaults kept), pin-push 0.644 -> 0.648 (noise level), press 0.621 -> 0.776 (pinch 3 -> 1.73 N, pusher roll weight 1 -> 0.35, lowest layout 0.03 -> 0.31)
- [x] 3. Add `--prm JSON` to hom_gpu.py's run command (every world gets that prm) and rescore each task's grid at the … Done 11-04 11:54: Done 11-04 11:49: hom_gpu.py run --prm (67e2b8fe4); the 4 grids rescored with each task's shared set, rows grid2dff_v4b_sv25_shared (HOM_ROWS_TAG _sv25_shared), 10:50-11:49, 89,280 rollouts. Twist and roll keep HOM's defaults, so their grids are reruns: rho 0.96/0.98, best tenth kept 0.90/0.92, same best grasp 0.79/0.82 (the evaluator's run-to-run spread). Pin-push rho 0.96, best tenth 0.67; press rho 0.47, best tenth 0.28, median best-of-16 0.42 -> 0.78, best layout (70,150) -> (65,140)
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
