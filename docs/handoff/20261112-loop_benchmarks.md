Status: active in window 8955e04e (autopilot stopped 17:22 with `ap disarm --stop`; the window took the task back)
State 11-13 13:35: page v3 at https://claude.ai/artifact/3m4bcPKugwFCtX5ZZLtd7f (local
docs/experiments/20261112-inner_loop_benchmark/20261113-sampling_mpc_vs_hom.html, builder scripts/hom/inner_loop_mpc_page.py),
commit bb8b61b24 pushed. New since 12:10: HOM's pose through a complementary filter (hom_gpu.POSE_FILTER; inner_loop_mpc
--pose-tau, --meas-every): tag tracker noise 9.6 -> 2.0 reversals/s at tau 0.2, 0.96 at 0.5 (exact 0.91), J 0.283 ->
0.310/0.315 (exact 0.375); open loop after the grasp J 0.342; 17 Hz tracker filtered J 0.299; with the fingertip model
still 7.4 reversals/s (CoP noise: filter it next). HOM-space DIAL (hs_anneal, 40 runs) J 0.415 at 0.63 reversals/s,
7.2 s per replanning for 16 hands; DIAL from a 40 ms old noisy state 0.408. HOM's commanded force 2 N: J 0.415.
Owner 13:2x asked about compute on corvus and a policy streamer: told HOM is CPU-only (2.0 ms per period here), corvus
i7-8700K/RTX 2070, wired ping 0.31-0.56 ms to 192.168.0.5 (no WiFi needed), UDP state up / targets down with a hold
and stop watchdog. Not yet done: measure HOM and the two-camera tag tracker on corvus; build the streamer; filter the
fingertip CoP. corvus's LFS store backed up here (11,948 objects verified) and deleted by another session.
State 11-13 12:10: page docs/experiments/20261112-inner_loop_benchmark/20261113-sampling_mpc_vs_hom.html (builder
scripts/hom/inner_loop_mpc_page.py) published at https://claude.ai/artifact/3m4bcPKugwFCtX5ZZLtd7f (colorado login),
commits 4fc181db7, ef59860cb, c35fc5d84 pushed. SCS0009 plant (scripts/hom/inner_loop_plant.py; every run before 11-13
10:30 was on the work plant, rows *_work10): HOM J 0.388 (grip 7.7 N, saturation 16 %, 0.63 rev/s); DIAL annealing 0.461
(4.7 N, 2.0 rev/s; one hand 1.17 s GPU per 0.1 s); PS real-time 0.335 (0.089 s per 0.1 s); DIAL with a point-contact
model 0.426 (replanning 18 % faster); HOM angle feedback lowers J (0.370 at 1/s to 0.322 at 8/s: three-pad share falls);
HOM-space DIAL (hs_anneal: Eq. 2 solved again around HOM's own relative twist, separation row x10; zero plan = HOM,
checked 0.772 on pair 0) smoke 0.569 at 16 samples, full runs queued. Films p0 and p10 in the page. Lane continues
(lane.txt): HOM sensing variants, HOM force sweep, planners from a 40 ms old noisy state, hs_anneal x2, iCEM, SPIDER,
SPIDER worst case, friction 0.4 block, PS, MPPI, CEM, then UHAS on the SCS0009 plant (uhas_train_eval.sh, ~22:00).
After each block: rebuild the page, commit, republish the same file path.
State 11-13 08:50 (work plant): MPC on the 20 pairs x seeds 0-1, exact model: mean J HOM 0.402 (not 0.634,
the last chunk's mean, as told to the owner on 11-12), DIAL anneal 0.582, iCEM 0.566, SPIDER 0.549, MPPI 0.506, CEM 0.499,
PS 0.480, SPIDER worst case 0.379, PS near real time 0.396, UHAS 0.389 (held 0.88); planners grip 16.5-23 N against 9.6,
reverse 1.65-3.15 per joint per s against 0.25; one hand 1.075 s GPU per 0.1 s replanning (near-real-time PS 0.086 s).
UHAS limits s0 (140 min): test J 0.033, held 0.63, against 0.204 and 0.76 published. queue4 runs the rest (step 13).
State 17:45: GPU queue done (all rc 0). UHAS from HOM's handover (156 min PPO): mean test J 0.32 vs HOM 0.25 per layout,
HOM's best layout still best (0.756 vs 0.714), rho 0.54; open-hand UHAS held 0 of 5,232; single-layout 0.015.
Usability (f32837c6a, inner_loop_usability.py, 1,744 worlds): grip 56 vs 10 N, servo saturation 83 vs 0 %, target
reversals 20 vs 0.4 per joint per s, drift 26 vs 8 mm. Films sent to the owner. KaRMA sph25 at 107/128 rows.
Task: owner 2026-11-12: "test how map-elites and/or another outer loops optimizer does"; "evaluate some of the other
inner loop controls we tried? Karma, uhas? If we have to train our own uhas, that's fine, esp since we have so many
demonstrations already"; then "in an ideal world we can show these other methods (... published ...) are both worse and
slower than ... gpu parallelized HOM. We can let the other methods have the sphere packed model and warp".
Settings: the talk's (SR2 cad_v2, v4b pads at 2.5 MPa, SCS0009 plant 0.72 N m/rad, MuJoCo-Warp 3.6, scenes
logs/20261102-hom_servo_plant/scenes), grids rows/grid2dff_v4b_sv25r, J = hom_landscape.track_score.
Read first: docs/experiments/20261029-foundational_pose_review/20261109-hom_search_methods.html Section 12 (outer loop,
done); scripts/hom/inner_loop_benchmark_page.py (the page to finish; its docstring lists the data); scripts/hom/karma_sr2.py,
uhas_sr2_cik.py, uhas_sr2.py, uhas_sr2_film.py (all committed, adb097ea3); UHAS paper docs/uhas/paper/sections/.
State 13:10: outer loop done and pushed (eb98da82e): on the four J grids, by 6,000 rollouts BO came within 0.05 of the best
J in 94-100 % of 200 runs (0.63-0.85 of the EA's median rollouts), EA64 88-100 %, CMA-ES 80-100 %, MAP-Elites 49-99 %,
random 52-99 %. KaRMA: logs/20261112-karma_sr2/queue.sh (sph40 145 layouts then sph25 128, 8 jobs; ends ~14:50); first
65 layouts: KaRMA-T rho 0.70 with J on the twist, its first choice (90, 90) mm J 0.33 against the best 0.76; 150-800 s per
layout beside the GPU job, 163 s alone. UHAS: action space vs HOM's handover grasps, median 4.6 mm at the pads (rows
uhas_express_twist_32x512.json). GPU queue logs/20261112-uhas_sr2/: train_twist.sh (handover start, 2,048 worlds, 120 min
budget, ~2.1 s per iteration, ends ~15:20, J proxy 0.019 at 17 min) -> eval of 109 test layouts -> after_twist.sh (film) ->
single_twist.sh (one layout (105, 70), 30 min, eval) -> open_twist.sh (open-hand start, 120 min, eval, film; ends ~18:20).
queue.log in that folder gets one line per step with its rc.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Wait for "queue done" in logs/20261112-karma_sr2/queue.log and "open film rc" in logs/20261112-uhas_sr2/queue… Done 11-12 21:08: Done 11-12 18:16: KaRMA queue done (sph40 rc 0 15:49, sph25 rc 0 18:16); UHAS queue all rc 0 (train, eval, film, single, open, films with forces 17:29-17:30, usability rc 0 17:35)
- [ ] 2. Build the page: .venv/bin/python scripts/hom/inner_loop_benchmark_page.py. Rewrite lede() so it states the outcome with numbers (time per layout of each evaluator; J of KaRMA's first choice on each task; UHAS's mean test J and held share against HOM's; the open-hand variant). Then python3 scripts/docs/prose_lint.py --top 20 on the page until within targets; search the text for the banned words of ~/.claude/CLAUDE.md.
- [ ] 3. Look at every figure and film poster at displayed size (media/20261112-karma_maps.png, 20261112-uhas_vs_hom.png, 20261112-film_hom_uhas_twist*.jpg) and fix what is unreadable.
- [ ] 4. Commit the page, its rows (docs/experiments/20261112-inner_loop_benchmark/rows/*), media and the docs indices it refreshed; push.
- [ ] 5. Publish the page as an artifact (first publish) and give the owner the local path with the URL; the docs index picks the URL up on the next build.
- [ ] 6. Report to the owner in the note's State and, when a window is open, in chat: outer loop (BO beats the EA in replay; MAP-Elites does not), KaRMA, UHAS, the costs. Offer next: live BO at the size study's sizes; UHAS on the other tasks; KaRMA on the full lattice.
- [ ] 7. Page: add the usability section from rows/usability_twist_20261112-1244_twist_s0.jsonl (scripts/hom/inner_loop_usability.py; medians over held rollouts, HOM against UHAS) and the three force-labelled films (media/20261112-film_hom_uhas_twist.mp4, _x50y110.mp4, _open.mp4); state that UHAS's mean test J beat HOM's (0.32 against 0.25 per layout) with 56 N grip, 83 % servo saturation and 20 target reversals per joint per second; summarize the two >2 MB evaluation rows into a committed file under 2 MB for the builder
- [ ] 9. MPC and UHAS reruns (owner 11-12 21:0x go): wait for 'queue done' in logs/20261112-mpc/queue.log (launched 21:07; order hom, uhas, ps, mppi, cem, icem, anneal, spider, spider_wc, ps_rt, UHAS limits s0 train/eval/usability/pairs, plant friction 0.4 for hom, uhas, uhas limits, anneal, spider, spider_wc, then UHAS limits s1 and published s1); every rc 0, else read logs/20261112-mpc/<step>.log and ~/.claude/resguard_events.jsonl, fix, rerun that step's command (the rows files resume)
- [ ] 10. Page section on the MPC: per controller on the 20 pairs x seeds 0-1 (rows/mpc_*_twist*.jsonl): mean J, held share, usability medians over held rollouts, GPU seconds per second of control (replan_s_mean / 0.1 s for one hand from rows/mpc_timing_twist.jsonl); the friction-0.4 table (spider vs spider_wc isolates the worst case); UHAS with limits against UHAS published at seeds 0-1; films of the best planner and of UHAS with limits on HOM's best pair (uhas_sr2_film.py --tag)
- [x] 12. Resguard at 01:38 on 11-13 (seen by the feed poller's session): the watchdog killed the queue step usability_… Done 11-13 08:47: Done 11-13 08:47: the kills (01:38 and again 08:44, PSI full 19.5-19.8 at 16 GB available) were numpy huge-page direct compaction (compact_stall 4.3 M, 99 % failed), not RSS; logs/20261112-mpc/queue4.sh sets NUMPY_MADVISE_HUGEPAGE=0, waits 1 min before each launch, retries rc 3 every 2 min and a kill once; the usability step then ran at memory pressure 0.00
- [x] 13. queue4 (launched 08:45, logs/20261112-mpc/queue4.sh, lines in queue.log): usability and pairs of limits_s0; f… Done 11-13 11:35: Superseded 11-13 10:30: queue4's runs were on the work plant; reruns on the SCS0009 plant go through logs/20261112-mpc/lane.sh (lines in lane.txt)
- [ ] 14. Lane logs/20261112-mpc/lane.sh on the SCS0009 plant (owner 11-13: films/page to analyze, HOM-space controls, MPC without the sphere pads, noisy pose, HOM real-time, a simple HOM improvement): wait for each line of lane.txt to appear in lane.done (queue.log has '<name> rc'); rebuild .venv/bin/python scripts/hom/inner_loop_mpc_page.py; commit the page, rows, media and indices; republish the page to its artifact URL (first publish 11-13 ~12:30). UHAS trainings on the SCS0009 plant are the lane's last two lines (logs/20261112-mpc/uhas_train_eval.sh, ~5.5 h).
- [ ] 15. Owner 11-13 13:2x: (a) measure HOM's controller time and the two-camera tag tracker (fps, CPU) on corvus after the owner's camera calibration; (b) a UDP streamer, corvus sends joint angles, tag poses and fingertip readings, the workstation returns servo targets with sequence numbers, corvus holds the last target and stops after ~3 missed periods; (c) filter the fingertip CoP as the pose is filtered (7.4 reversals/s remain with the tip model at tau 0.2), then rerun sv_pf_tip
Standing directions (for every step):
- Owner 11-12 17:40: KaRMA and UHAS are not fair comparisons; UHAS is less usable for many reasons, to be codified and quantified. The page reports UHAS's higher J and its usability numbers side by side.
- Owner 11-12 21:1x: the UHAS films look jittery and implausible; HOM is smoother than every method tried. Report smoothness with the scorecard (target reversals per joint per second, RMS target rate) for every controller, beside J
<!-- /ap:steps -->
Claimed: scripts/hom/hom_search_qd.py, karma_sr2.py, uhas_sr2*.py, inner_loop_benchmark_page.py,
docs/experiments/20261112-inner_loop_benchmark/, logs/20261112-*.
Waiting on owner: whether to run Bayesian optimization live as the outer loop (size study's sizes, ~3 h GPU).
Do not: GPU jobs beside the running queue (they share the card); git lfs push --all; change the talk decks without the
owner; edit uhas_sr2*.py or hom_gpu.py while a queue step that imports them is starting.
