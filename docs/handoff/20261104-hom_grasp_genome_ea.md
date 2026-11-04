Status: active
Task: Owner, 2026-11-03: "can't we score and refine a grasp based off how well the grasp (+morphology) executes the HOM
control? from what i understand we are doing pretty random sampling"; 18:1x: "set up different autopilot jobs to
continue promising work". Put the grasp in the evolutionary search's genome: a child inherits its parent's grasp,
re-solved on its own layout and mutated, and HOM scores that one grasp; compare with scripts/hom_ea.py's best-of-16
search at equal rollouts on the SCS0009.
Read first: docs/handoff/20261104-hom_score_refine_page.md (State, its Go/no-go line); the score refine page and README
in docs/experiments/20261103-hom_score_refine/; scripts/hom_ea.py and hom_landscape.ea_search; fp_sr2_toy.py (Problem,
grasp_set, clearances); the trends page docs/experiments/20261029-foundational_pose_review/20261030-hom_layout_trends.html
(the EA's budget study: pop 64 and 6,000 rollouts reach within 0.05 of the grid's best in 97-100 % of runs).
State: 11-04 12:3x (autopilot, colorado). Step 1: go (validated median S2e - S1 twist +0.083, press +0.032, roll
+0.019, pin-push +0.025). Step 2: scripts/hom_ea_grasp.py (f90255448), smoke on the roll in rows_smoke/ea_grasp/.
Step 3 under way: queue logs/20261104-hom_grasp_genome_ea/queue.sh started 12:27:44, detached (queue.log; Q_<step>.log
and .done per step; a stopped queue resumes from its rows: rerun the nohup line). Per task: eag_<t> (the grasp EA, three
seeds in lockstep, ~29 s per generation of 288 worlds, ~60 generations), ea_<t>_s0..2 (hom_ea.py, HOS_PLANT=servo,
HOM_ROWS_TAG=_sv25, rows ea_live_sv25/), val_<t>, cpu_<t>; then report. Expected done ~16:00. Do not edit
scripts/hom_ea_grasp.py while the queue runs (every batch child and pool worker re-imports it). Meanwhile: the page
section (builder scripts/hom_score_refine_page.py: ea-figs, ea-films, sec_ea). The roll's sv25 grid has one layout
within 0.05 of its best (0.654, call 249); hom_ea replayed on that grid reaches it by 6,000 in 66 % of 200 runs.
12:4x: twist grasp EA at gen 26 (2,600 rollouts per search), all three incumbents at 1.0. The resguard gate is closed
on swap (15.2 GB > 15 GB; desktop pages: VS Code, gnome-shell, Chrome, the owner's ReSkin viewer), so the queue's next
step will retry every 120 s until swap falls; the running step is unaffected. Builder: sec_ea, fig_ea, ea films, terms,
ref [5] Yang et al., files and algorithm rendering (render handles algorithmic blocks) written, not yet committed.
13:02: eag_twist done (2,068 s, peak 2.4 GB; 65 generations, median 32 s; GPU 1,672 s, draws 382 s). Its three
searches came within 0.05 of the grid's best (0.95; grid best 1.0) at 1,479, 1,410 and 747 rollouts (hom_ea replayed
on the stored grid: median 2,640, 96.5 % by 6,000); all three final incumbents are grasp steps at (85, 70), (105, 65),
(90, 70). From 13:02 resguard refuses ea_twist_s0 on swap 15.1 GB > 15 GB (RAM 14.6 GB free, PSI 0, load 0.3; the swap
is the desktop's); the queue retries every 120 s. Grid references (scratchpad grid_ref.json): best twist 1.0 (6 within
0.05), roll 0.654 (1), pin-push 0.744 (17), press 0.816 (3); hom_ea on the stored grid within 0.05 by 6,000: 96.5, 66.5,
100, 100 %. 13:28: memory PSI rose to 15 with another session's jobs (ec307d8f's gp_dense.py and
ur5_probe_calibration_analysis.py, launched 13:25-13:28); the watchdog killed those two and the ReSkin wrench viewer;
this queue had nothing running. 13:34:13 ea_twist_s0 started (gate open, swap 14.6 GB); rows on the SCS0009 plant
(kp0.72..., scenes logs/20261102-hom_servo_plant/scenes); gen 0 2,496 rollouts, incumbent (70, 110) 0.935. README
section (runbook, files) and queue_ea_grasp.sh beside it written; expected queue end ~17:00.
13:51: twist done. hom_ea s0/s1/s2: 6,288/6,240/~6,200 rollouts, incumbents (100, 70), (105, 105), (100, 70), within
0.05 at 5,472 and 1,728 (s2: see report). Validation (seeds 20-24): all six incumbents 1.0 (three-pad angle 90-106
deg); the grid's best layouts' S1 at 20-24: (100, 70) 1.0, (105, 105) 1.0, 362 0.975, 312 0.906. CPU at seed 20:
grasp EA 0.00 (85, 70: drop; GPU 0.998), 1.00, 1.00; layout EA 0.92, 0.37, 0.92 (GPU 1.0 each); 2.7-3.6 s a rollout.
14:30: roll done (grasp EA 13:52-14:18, 21 s a generation). Grasp EA: no search within 0.05 of 0.654 by 6,000;
incumbents (95, 60), (80, 55), (75, 55) at 0.583/0.592/0.583, validated 0.580/0.595/0.580. Layout EA: within 0.05 at
5,184, 5,424, 3,648; incumbents (85, 170) x2 validated 0.654, 0.653 and (90, 130) 0.607 -> 0.330 at seeds 20-24. CPU
seed 20: grasp EA 0.58, 0.59, 0.59 (GPU 0.58-0.60); layout EA 0.62, 0.50, 0.62. eag_pinpush from 14:30:50.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Go/no-go from the page job's line: if the validated S2 - S1 is under 0.03 in median on all four tasks, the gr… Done 11-04 12:14: Go 11-04 12:2x: validated median S2e - S1 on the 30 best SCS0009 layouts is twist +0.083, roll +0.019, pin-push +0.025, press +0.032 (page job's Go/no-go line); twist and press above 0.03, so the grasp-genome EA runs
- [x] 2. Write scripts/hom_ea_grasp.py: an individual is (lattice layout, grasp v). A child moves its parent's layout … Done 11-04 12:29: scripts/hom_ea_grasp.py (f90255448): (layout, grasp) individuals, hom_ea's gen 0 and selection, layout move + Problem.solve re-solve or grasp step at p 0.5, draw_layout's rule at 0.25 widths, fitness = 3-seed mean; search seeds in lockstep (one GPU batch per generation); run/validate/cpu/report smoke-tested on the roll (rows_smoke), resume by replay checked
- [ ] 3. Run it on sv25 at the default objects (twist and roll 40 mm sphere, pin-push and press 25 mm cylinder), three search
    seeds each, and hom_ea.py at the same budget on the same plant (HOS_PLANT=servo, HOS_SCENES, HOM_ROWS_TAG=_sv25);
    validate each run's incumbent at seeds 20-24.
- [ ] 4. Report the best validated score per method and the rollouts each needs to come within 0.05 of the grid's best; add a
    section and a figure to the score refine page; commit; push; memory and README.
Standing directions (for every step):
- Compute tracking (owner 11-03 19:0x: "let's make sure we're timing this all for compute-tracking purposes and throughput measuring, including cpu vs gpu speedups"): run every GPU batch through hom_gpu.batch_child so each writes its _timing.jsonl line beside its rows under docs/experiments/20261103-hom_score_refine/rows/; log each search iteration's wall, draw and GPU-batch times (and write them as JSON lines if you add a script); after the runs, `python3 scripts/hom_compute.py ledger` and `report`, and give the job's rollouts, batch hours and ms per rollout (run, batch, whole step) in its page section; for a new kind of rollout (another plant, the friction band, a new search), time the CPU runner on it with `scripts/hom_compute.py cpu-bench` (through resguard) and state the CPU against GPU speedup.
<!-- /ap:steps -->
Claimed: scripts/hom_ea_grasp.py, docs/experiments/20261103-hom_score_refine/rows/ea_grasp/, the page's EA section.
Waiting on owner: none.
Do not: publish artifacts; run beside another GPU job; change scripts/hom_ea.py's behaviour (its rows are the reference).
