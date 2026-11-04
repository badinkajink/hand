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
State: 11-04 12:2x (autopilot, colorado). Step 1: go (validated median S2e - S1 twist +0.083, press +0.032, roll
+0.019, pin-push +0.025). No hom_ea rows on sv25 at the default objects exist (ea_live_sv25 has other sizes only), so
hom_ea runs fresh there. Writing scripts/hom_ea_grasp.py; plan: the three search seeds run in lockstep, one GPU batch
per generation (3 x 32 children x 3 seeds = 288 worlds), rows <task>_d<d>.jsonl with the search seed in the key, gens
per seed <task>_d<d>_s<seed>_gens.jsonl.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Go/no-go from the page job's line: if the validated S2 - S1 is under 0.03 in median on all four tasks, the gr… Done 11-04 12:14: Go 11-04 12:2x: validated median S2e - S1 on the 30 best SCS0009 layouts is twist +0.083, roll +0.019, pin-push +0.025, press +0.032 (page job's Go/no-go line); twist and press above 0.03, so the grasp-genome EA runs
- [ ] 2. Write scripts/hom_ea_grasp.py: an individual is (lattice layout, grasp v). A child moves its parent's layout by
    U(-delta/2, delta/2) mm on the 5 mm lattice (hom_ea's mutation), re-solves the parent's v on the new layout with
    fp_sr2_toy.Problem.solve warm-started from it, then steps v by N(0, (0.25 x the feasible widths)^2) until feasible
    (hom_grasp_refine.draw_layout's rule); with probability 0.5 a child keeps the layout and mutates only the grasp.
    Fitness: the one grasp's mean score at seeds 0-2 (3 rollouts a child, against 48 for a best-of-16 layout).
    Population 64, budget 6,000 rollouts, elitist as hom_ea; rows and gens files as hom_ea's, under
    docs/experiments/20261103-hom_score_refine/rows/ea_grasp/.
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
