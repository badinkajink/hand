Status: active
Task: Owner, 2026-11-03 18:1x: "yea, let's send it on these tasks ... we can try out multiple scores, set up different
autopilot jobs to continue promising work. make sure things are documented well!" The tasks: refine each layout's grasp
on the HOM score (S2), and the grasp with HOM's commanded forces and contact modes (S3), against the best of 16 grasps
at HOM's defaults (S1). This job analyses the night's queue and builds the result page.
Read first: docs/experiments/20261103-hom_score_refine/README.md; the docstrings of scripts/hom_grasp_refine.py and
scripts/hom_score_refine_page.py; logs/20261103-hom_score_refine/queue.log; memory project_hom_codesign_levels_2026_11_03.md;
~/.claude/writing_reference.md; the plant page docs/experiments/20261029-foundational_pose_review/20261102-hom_plant_modulus.html
(style and the friction section).
State: 11-03 18:4x. Code 69d70295b (pushed): per-world forces and modes in hom_gpu, hom_tasks.mode_weights, the search
scripts/hom_grasp_refine.py. Queue started 18:26:53, expected done 03:00-04:30 on 11-04: sv25 (SCS0009 at 2.5 MPa, 60
layouts per task, S2 and S3 searches of 8 iterations, validation at seeds 20-24), then k21 and k21mu0p4 (STS3032 at
friction 1.0 and 0.4, the same 30 layouts, 6 iterations) and each friction's picks run at the other. 20:1x: sv25
level grasp done on all four tasks (twist 215 s per iteration of 4,320 rollouts, roll 138 s, pin-push 211 s, press
168 s); median S2 - S1 at seeds 0-2 over 60 layouts: twist 0.096, roll 0.040, pin-push 0.034, press 0.062 (validation
pending). Level full: twist's first iteration 5,757 rollouts in 266 s, median best 0.908 (below S2's 0.949: see
direction 10). Page builder scripts/hom_score_refine_page.py (figs, films,
page) was checked on rows_smoke/ (prose within targets); its result text is a draft to rewrite from the numbers.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [ ] 1. Wait for the queue: while `grep -q "score refine queue done" logs/20261103-hom_score_refine/queue.log` fails,
    `ap snooze <job> 45 "waiting for the HOM score queue"`. Then `grep -E "failed|refused" logs/20261103-hom_score_refine/queue.log`;
    for a failed step read its logs/20261103-hom_score_refine/Q_<step>.log, fix the cause, relaunch
    `nohup setsid bash logs/20261103-hom_score_refine/queue.sh >> logs/20261103-hom_score_refine/queue.log 2>&1 &`
    (finished steps are skipped, a stopped step resumes from its rows) and wait again.
- [ ] 2. Commit and push docs/experiments/20261103-hom_score_refine/rows/ (LFS; not rows_smoke/), with a message giving per
    task the median S1, S2 and S3 at seeds 20-24 on the 30 best SCS0009 layouts and rho(S1, S3) over 60.
- [ ] 3. `.venv/bin/python scripts/hom_score_refine_page.py figs`, then `... films` (CPU clips through resguard). Read every
    PNG and film poster at its displayed size and fix what is unreadable.
- [ ] 4. Rewrite the lede and the text of the result sections (sec_sv25, sec_prm, sec_sts, limitations, future work) from the
    numbers: outcome first, which tasks gain and by how much, whether the best layout or the ranking changes, what S3
    chose for forces and modes, what retuning at friction 0.4 is worth. Build the page
    (`.venv/bin/python scripts/hom_score_refine_page.py`), keep prose_lint within targets, search the banned words.
    Commit the page, builder, media and the refreshed indices; push.
- [ ] 5. Record: memory project_hom_codesign_levels_2026_11_03.md (results, page path) and its MEMORY.md line; one sentence in
    CLAUDE.md's HOM servo-plant bullet: HOM's forces and contact modes are per-world parameters (hom_tasks.mode_weights,
    a world's prm in hom_gpu) and scripts/hom_grasp_refine.py refines them; the outcome appended to the README.
- [ ] 6. Write a "Go/no-go:" line in State for the follow-up jobs (armed after this one, from 09:15): the validated median of
    S2 - S1 and S3 - S2 per task on the 30 best layouts, rho(S1, S3) per task, and the median of (S3 tuned at 0.4) - (S3
    tuned at 1.0, run at 0.4) per task.
- [ ] 8. Compute and throughput, after the queue has ended (the GPU free, the box otherwise idle; record resguard status): through resguard, `.venv/bin/python scripts/hom_compute.py gpu-bench` (batches of 64, 256, 1024 and the queue's size per task), `cpu-bench --n 6 --variants plant matched --procs 1`, then `cpu-bench --n 4 --variants plant --procs 4` (the throughput of four cores), then `ledger` and `report`. `scripts/hom_score_refine_page.py figs` draws media/20261103-throughput.png and the page's Compute and throughput section reads compute/; state the speedups in the lede only if they are new. Then add to scripts/hom_grasp_refine.py a JSON line per search iteration (draw, open poses, GPU batches, total seconds, rollouts) in <task>_d<d>_iters.jsonl for later runs, keeping the log line. Commit compute/ and the scripts.
Standing directions (for every step):
- Compute tracking (owner 11-03 19:0x: "let's make sure we're timing this all for compute-tracking purposes and throughput measuring, including cpu vs gpu speedups"): run every GPU batch through hom_gpu.batch_child so each writes its _timing.jsonl line beside its rows under docs/experiments/20261103-hom_score_refine/rows/; log each search iteration's wall, draw and GPU-batch times (and write them as JSON lines if you add a script); after the runs, `python3 scripts/hom_compute.py ledger` and `report`, and give the job's rollouts, batch hours and ms per rollout (run, batch, whole step) in its page section; for a new kind of rollout (another plant, the friction band, a new search), time the CPU runner on it with `scripts/hom_compute.py cpu-bench` (through resguard) and state the CPU against GPU speedup.
- In the compute step, run cpu-bench with --variants plant filter, never matched or solver: on 11-03 the GPU's solver settings (20 iterations, tolerance 1e-6) made CPU MuJoCo drop every twist and roll sphere and one matched pin-push rollout ran for minutes (scripts/hom_compute.py docstring). The CPU numbers of 11-03 19:1x were taken beside the running GPU queue (load about 5 on 20 cores); the clean rerun after the queue is the one to quote, with both in compute/cpu_bench.jsonl.
- Elitist picks (window 220623e7, 11-03 20:1x): a search scores draws around its start, never the start itself, so a level's best candidate can end below the grasp it started from. At level grasp on sv25 (seeds 0-2) this happened on twist 3 of 60 layouts (worst -0.102), pin-push 2 (worst -0.169), roll 2 (within 0.001) and press none; level full started from those raw S2 picks (picks()). After the queue ends, give in cmd_report and on the page each level's raw pick and its elitist pick: S2e is the better at seeds 0-2 of S1's grasp (s1 in <task>_layouts.json) and S2's pick; S3e is the best at seeds 0-2 of S1, S2 and S3. The val/ and val_from_*/ rows already hold all three at seeds 20-24. Quote the elitist medians as the result (the Go/no-go line too), give the count of layouts whose search ended below its start, define 'elitist pick' in Terms, and state once in the method that level full started from the raw S2 pick. Then, for later runs, make draw_layout return the start itself as candidate 0 of iteration 0, so picks() is elitist by construction and the start is re-scored in the same GPU batch (the night's rows are complete and are not replayed); the worst-case and shared-controller jobs use that search.
<!-- /ap:steps -->
Claimed: docs/experiments/20261103-hom_score_refine/, scripts/hom_score_refine_page.py, logs/20261103-hom_score_refine/,
scripts/hom_grasp_refine.py (fixes only, never while the queue runs).
Waiting on owner: none.
Do not: publish artifacts; run other GPU work while the queue runs; edit scripts/hom_gpu.py, hom_tasks.py or
hom_grasp_refine.py while the queue runs (its steps import them at every batch). The scripts switch
(docs/handoff/20261103-scripts_topic_folders.md, step 2) needs 45 minutes without commits under scripts/ and without
scripts/ processes once the lab is idle, and gives up at 09:00 on 11-04: from 07:15, if the lab is idle and the switch
has not landed, stop at a clean point, commit, and snooze until 09:15. After the switch the scripts are under
scripts/hom/ and the package is sr2hand.
