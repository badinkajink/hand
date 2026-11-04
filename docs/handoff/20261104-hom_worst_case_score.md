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
11-04 04:2x (autopilot, colorado): steps 1-2 done; band option committed 198ec826e (not pushed). Smoke checks
(logs/20261104-hom_worst_case/check*.log): start vs grids max 0.025; band vs single-plant 3-seed means median
0.0004-0.0012, max 0.24; an identical batch vs its repeat median 0.0001-0.0010, max 0.22 (the GPU's own spread on
marginal twist candidates; quote it on the page). Queue logs/20261104-hom_worst_case/queue.sh launched 04:22:06
(nohup setsid; log queue.log, Q_<step>.log/.done): val07 (the 11-03 picks at friction 0.7), band grasp + full (6 its),
band val, report; expected done about 09:00. Do not edit scripts/hom_grasp_refine.py while it runs.
06:3x: val07 done 04:27; band grasp level done 06:23:58 (6 its, 6,480 rollouts an iteration; twist 1,952 s, roll 1,311,
pin-push 2,061, press 1,648; peak 2.44 GB). Worst-case pick minus start at seeds 0-2, median: twist +0.169, roll +0.041,
pin-push +0.125, press +0.034; none below its start. Plant at the worst case of the picks: friction 0.4 for 22, 10, 21
and 5 of 30; 1.0 for 6, 14, 7 and 25 (press).
09:0x: queue done 08:54:06, no failed step (step 3 done); report rows/band_k21mu/20261104-band_levels.json. Lowest
score over the band, S3w less S3e tuned at 1.0: median +0.26, +0.32, +0.56, +0.21 (twist, roll, pin-push, press); less
S3e tuned at 0.4: +0.08, 0.00, 0.00, +0.02. At 1.0 S3w less the 1.0 picks -0.46, -0.05, -0.12, -0.02. Rho(nominal at
1.0, S3w lowest) 0.31, 0.32, 0.51, 0.27. cpu-bench k21mu0p7 + ledger + report done (logs/.../compute.sh). band-figs
done; band-films done, but on the CPU plant S3w of layout 362 drops at 0.4 (GPU 0.67 at seed 20): CPU replication
of all picks at seed 20 done 09:13 (logs/20261104-hom_worst_case/cpu_check.sh, 900 rollouts; rows copied to
rows/band_k21mu/20261104-cpu_check.jsonl): lowest of 0.4/1.0, S3w less the 1.0 picks, CPU every contact vs GPU:
pin-push +0.55/+0.54, press +0.10/+0.11, twist +0.00/+0.27 (with the GPU's filter on the CPU +0.28), roll +0.01/+0.47.
Film now 2x2 (S3w at 0.4 with the filter 0.67, every contact 0.00). 09:2x: sec_band, lede, limits, future work, files
written; page built (prose within targets); next: read it, commit, push; README + queue_band.sh, memory (step 5).
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Paths: if the scripts switch has landed (scripts/hom/ exists), the scripts are under scripts/hom/ and the pac… Done 11-04 04:00: scripts/hom/ absent; switch job waits after fingertip_shape_rerun (not migrating): scripts stay at scripts/hom_*.py, package morphohand
- [x] 2. Add a band option to hom_grasp_refine.py: PLANTS entry k21mu0p7 (scenes_mu0.7, tag _k21mu0p7); `run --band k2… Done 11-04 04:22: band option committed (BANDS band_k21mu, plant k21mu0p7, evaluate_band, validate/report --band); smoke 2 twist layouts x 2 its: start vs grids max 0.025 (18 rollouts); band vs single-plant 3-seed means median 0.0004-0.0012, max 0.24; identical batch vs its repeat median 0.0001-0.0010, max 0.22: within the GPU's run-to-run spread
- [x] 3. Run both levels with the worst-case score on the k21 sample (30 layouts, 6 iterations; three plants per candi… Done 11-04 08:55: Done 11-04 08:54: queue logs/20261104-hom_worst_case/queue.sh 04:22-08:54, no failed step; both levels 6 iterations (grasp 6,480 rollouts an iteration, 6,972 s for 4 tasks; full 8,640, 8,591 s), peak 2.44 GB; S1w/S2w/S3w at seeds 20-24 on 0.4, 0.7, 1.0 (450 rows a task and plant) and the 11-03 picks at 0.7; report rows/band_k21mu/20261104-band_levels.json. Lowest score over the band, S3w less S3e tuned at 1.0, median +0.26 twist, +0.32 roll, +0.56 pin-push, +0.21 press; less S3e tuned at 0.4: +0.08, +0.00, 0.00, +0.02; at 1.0 S3w less the 1.0 picks -0.46, -0.05, -0.12, -0.02
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
