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
direction 10). 22:13: sv25 done (report rows/sv25/20261103-score_levels.json); k21 started. Window's look at seeds
20-24, 30 best layouts, median S1/S2/S3: twist 0.885/0.953/1.000, roll 0.560/0.577/0.604, pin-push 0.695/0.719/0.749,
press 0.702/0.754/0.906; random 30 gain more (S3 - S1 twist +0.27, press +0.38); rho(S1, S3) 0.57-0.76; S3 < S1 on 7
of the roll's 30 best; S3 lowered the press's pinch to a median 1.8 N (22 % at the 1.5 N floor) and w_roll_push to
0.38. Elitist picks change no top-30 median. Page builder scripts/hom_score_refine_page.py (figs, films,
page) was checked on rows_smoke/ (prose within targets); its result text is a draft to rewrite from the numbers.
11-04 01:4x (autopilot): queue done 01:20:11, no failed step. db9baaa57 pushed: rows (LFS) and the reports
rows/<plant>/20261104-score_levels.json, with elitist picks (hom_grasp_refine.elitist; fields s2e_*, s3e_*,
below_start_*). The old sv25 report 20261103-score_levels.json is replaced. Step 3 under way: builder plots S2e/S3e,
new fig_friction (k21 picks at 0.4 vs k21mu0p4 picks); friction film = twist layout 110 x 110 (call 362). Cone check
done (scratchpad, to go on the page): every tripod and pinch inward normal 0-20 deg above horizontal, inside the 0.4
cone (21.8 deg) in both groups; static mu needed at HOM's forces twist 0.48 (1.0 picks) vs 0.47 (0.4 picks), roll
0.37 vs 0.39: the cone does not separate them. Pinch drops at 0.4 rise with the pinch latitude (rho 0.59 / 0.67).
01:50: step 3 done (figs 2x2, 15 clips). Step 8's benchmark chain runs detached from 01:47
(logs/20261103-hom_score_refine/compute_bench.sh, .log; gpu-bench --force because its pgrep matches this session's
argv); hom_compute report now per plant and per CPU session. Writing step 4's text in the builder meanwhile; do not
edit hom_grasp_refine.py or hom_compute.py until the chain ends.
02:2x: benchmark chain done 02:05:47 (all exit 0; compute/20261104-ledger.json, -throughput.json); hom_grasp_refine
patched (start = cand 0 of it 0; <task>_d<d>_iters.jsonl; smoke on roll ok, rows_smoke only); page built within prose
targets; README outcome, CLAUDE.md sentence, memory done. Next: commit and push, then mark 4, 5, 6, 8.
Go/no-go: (11-04 02:0x; elitist picks S2e/S3e, validated at seeds 20-24) on the 30 best SCS0009 layouts the median
S2e - S1 is twist +0.083, roll +0.019, pin-push +0.025, press +0.032 (twist above 0.03: the grasp-genome EA goes);
S3e - S2e is twist +0.002, roll +0.019, pin-push +0.036, press +0.143 (pin-push and press above 0.03: the shared
controller goes); rho(S1, S3e) over the 60 layouts per task is 0.75, 0.56, 0.71, 0.67; on the STS3032 the median of
(S3e tuned at 0.4) - (S3e tuned at 1.0, run at 0.4) over 30 layouts is +0.432, +0.513, +0.528, +0.181.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Wait for the queue: while `grep -q "score refine queue done" logs/20261103-hom_score_refine/queue.log` fails,… Done 11-04 01:29: queue done 01:20:11 on 11-04, no failed or refused step, no traceback, no candidate left out; reports rows/sv25/20261103-score_levels.json and rows/{k21,k21mu0p4}/20261104-score_levels.json
- [x] 2. Commit and push docs/experiments/20261103-hom_score_refine/rows/ (LFS; not rows_smoke/), with a message givin… Done 11-04 01:33: db9baaa57 pushed: rows (LFS, 127 files), reports rows/<plant>/20261104-score_levels.json with elitist picks; top-30 SCS0009 S1/S2/S3 twist 0.885/0.953/1.000, roll 0.560/0.577/0.604, pin-push 0.695/0.719/0.749, press 0.702/0.754/0.906; rho(S1,S3) over 60: 0.75/0.56/0.71/0.67
- [x] 3. `.venv/bin/python scripts/hom_score_refine_page.py figs`, then `... films` (CPU clips through resguard). Read… Done 11-04 01:46: figs as 2x2 grids (text >= 13 px at the 828 px column), S2e/S3e plotted, new friction_k21.png; 15 CPU clips (4 task films S1/S2e/S3e on SCS0009 + friction film twist 110x110: 1.0 pick at 1.0 and 0.4, 0.4 pick at 0.4), labels 24/23 px; every PNG and poster read at display size
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
- From hand-20261103-scripts_topic_folders, 00:1x on 11-04: the scripts switch will not land on 11-04. The owner's direction of 19:3x on 11-03 puts it after hand-20261103-fingertip_shape_rerun, which runs after the HOM chain. Skip this note's stop from 07:15 and keep working; the scripts stay at scripts/hom_*.py and the package stays morphohand until the switch.
- Window 220623e7, 01:2x on 11-04: the queue ended at 01:20:11 with no failed step. The cross-friction runs give the night's largest difference. Over the 30 layouts at seeds 20-24, the picks tuned at friction 1.0 score a median 0.000 at friction 0.4 on the twist and pin-push at S1, S2 and S3, 0.000, 0.031 and 0.000 on the roll, and 0.440, 0.491 and 0.425 on the press. The picks tuned at 0.4 score 0.079, 0.241 and 0.471 on the twist, 0.544, 0.606 and 0.616 on the roll, 0.506, 0.515 and 0.633 on pin-push and 0.810, 0.858 and 0.928 on the press. At 0.4 the friction-1.0 S1 grasps drop the object in 109, 37, 97 and 72 of 150 rollouts (twist, roll, pin-push, press), the 0.4 grasps in 51, 30, 38 and 21; 47 more roll rollouts of the 1.0 grasps end not held with no finger in contact. Run at 1.0, the 0.4 picks score 0.23 to 0.33 on the twist and 0.53 to 0.89 on the other tasks. Before the page names a cause, compare the two grasps' contact normals with the 0.4 friction cone in the rows. In sec_sts, film one layout's friction-1.0 S3 pick run at friction 0.4 beside its friction-0.4 S3 pick on the twist.
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
