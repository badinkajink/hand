Status: active
Task: Owner, 2026-11-03: "maybe do need to do some online sysid to measure friction, and/or to have robustness to a band
of physical parameters as a scoring parameter in hand selection"; "worst-case sampling makes sense, i saw it in spider
too"; 18:1x: "we can try out multiple scores, set up different autopilot jobs to continue promising work". Add a
worst-case score to scripts/hom_grasp_refine.py (each candidate's lowest score over a band of scene frictions) and
compare its picks with the nominal and per-friction picks of the 11-03 queue.
Read first: docs/handoff/20261104-hom_score_refine_page.md (State, its Go/no-go line); the page
docs/experiments/20261103-hom_score_refine/20261103-hom_score_refine.html and README.md beside it;
scripts/hom_grasp_refine.py; the friction section of
docs/experiments/20261029-foundational_pose_review/20261102-hom_plant_modulus.html; logs/20261102-hom_servo_plant/README.md
(scene caches scenes_mu0.4, scenes_mu0.7, scenes).
State: 11-03 18:4x, not started. STS3032 friction grids exist at 0.4, 0.7, 1.0 and 2.4 (rows grid2dff_v4b_k21mu0p4,
_k21mu0p7, _k21, _k21mu2p4); the 11-03 queue refined grasps, forces and modes at 1.0 and at 0.4 on the same 30 layouts
(rows/k21, rows/k21mu0p4) and ran each friction's picks at the other.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [ ] 1. Paths: if the scripts switch has landed (scripts/hom/ exists), the scripts are under scripts/hom/ and the package is
    sr2hand; if the switch job (`ap status`, its note's State) is migrating scripts/hom/, snooze until it moves on.
- [ ] 2. Add a band option to hom_grasp_refine.py: PLANTS entry k21mu0p7 (scenes_mu0.7, tag _k21mu0p7); `run --band
    k21mu0p4,k21mu0p7,k21` scores each candidate at seeds 0-2 on every plant of the band and takes the lowest 3-seed
    mean. A batch's plant comes from os.environ (HOS_PLANT, HOS_SCENES, HOM_ROWS_TAG) set by the parent before it spawns
    hom_gpu.batch_child, which reads them at import; the open pose does not depend on the friction. Rows under
    rows/band_k21mu/<level>/. Check once with --smoke on 2 layouts x 2 iterations that a plant's rows match a
    single-plant run of the same candidate within the evaluator's spread (0.01-0.04); no test files.
- [ ] 3. Run both levels with the worst-case score on the k21 sample (30 layouts, 6 iterations; three plants per candidate,
    so about 3 h a level: cut to 4 iterations if needed and say so), detached through resguard with the retry wrapper of
    logs/20261103-hom_score_refine/queue.sh; then validate the worst-case picks at seeds 20-24 on each of the three
    frictions (`validate --picks-from`).
- [ ] 4. Compare per task at each friction: worst-case picks, nominal picks (tuned at 1.0) and per-friction picks (tuned at
    0.4 or 1.0): the loss at 1.0 for the worst case, the gain at 0.4, the lowest score over the band. Add a section to
    the score refine page (scripts/hom_score_refine_page.py) with a table and a film of one layout's nominal and
    worst-case picks at friction 0.4; rebuild; commit; push.
- [ ] 5. Memory project_hom_codesign_levels_2026_11_03.md and the README beside the page.
Standing directions (for every step):
- Compute tracking (owner 11-03 19:0x: "let's make sure we're timing this all for compute-tracking purposes and throughput measuring, including cpu vs gpu speedups"): run every GPU batch through hom_gpu.batch_child so each writes its _timing.jsonl line beside its rows under docs/experiments/20261103-hom_score_refine/rows/; log each search iteration's wall, draw and GPU-batch times (and write them as JSON lines if you add a script); after the runs, `python3 scripts/hom_compute.py ledger` and `report`, and give the job's rollouts, batch hours and ms per rollout (run, batch, whole step) in its page section; for a new kind of rollout (another plant, the friction band, a new search), time the CPU runner on it with `scripts/hom_compute.py cpu-bench` (through resguard) and state the CPU against GPU speedup.
- Every search scores its start: candidate 0 of iteration 0 is the start itself (the page job's fix to hom_grasp_refine.draw_layout after the queue; check it is in before running), and a refined score is quoted beside its start's at the same seeds. On 11-03 the search without it ended below its start on 7 of 240 sv25 layouts (worst -0.169, pin-push).
<!-- /ap:steps -->
Claimed: scripts/hom_grasp_refine.py (the band option), the page's worst-case section,
docs/experiments/20261103-hom_score_refine/rows/band_*, logs/20261104-hom_worst_case/.
Waiting on owner: none.
Do not: publish artifacts; run beside another GPU job; change the S1/S2/S3 code paths or their rows.
