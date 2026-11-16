Status: active (armed for the autopilot on colorado from window 8955e04e, 11-15 17:44; the owner left for the night)
State 11-16 07:45 (autopilot): step 40, cells 1-7 of 8 done. Press 18 mm (07:45, 43 min): rho 0.70 (0.60 at
  25 mm), single seed 0.55 vs ceiling 0.61, mean J HOM 0.343, planner 0.471; both controllers' best pair (85, 110)
  (HOM 0.75, planner 0.76); ten-best median x_sep 72.5 mm under both (75 and 90 at 25 mm), 3 of 10 shared. Section
  revised (63d5bce71). Pin-push 18 mm running (~08:30), then the size films (now the roll at 28 mm, best pairs 79 mm
  apart).
State 11-16 07:03 (autopilot): step 40, cells 1-6 of 8 done. Roll 28 mm (07:02, 34 min): rho 0.17 (0.38 at
  40 mm), single seed 0.15 vs ceiling 0.71, mean J HOM 0.068, planner 0.555; best HOM (70, 150) 0.28 (planner 0.25),
  planner (95, 75) 0.80 (HOM 0.00), 79 mm apart, the farthest so far; ten-best median x_sep HOM 110, planner 92.5 mm
  (77.5 and 112.5 at 40 mm), 1 of 10 shared. Press 18 mm running, then pin-push 18 (~08:40), size films.
State 11-16 06:28 (autopilot): step 40, cells 1-5 of 8 done. Twist 28 mm (06:28, 58 min): rho 0.62 (0.81 at
  40 mm), single seed 0.56 vs ceiling 0.69, mean J HOM 0.341, planner 0.523; best HOM (65, 80) 0.75 (planner 0.94),
  planner (80, 80) 0.97 (HOM 0.61); ten-best median x_sep HOM 80, planner 77.5 mm (100 and 90 at 40 mm), 4 of 10
  shared. HOM page section drafted in the builder (3a4a3747a; page not yet committed). Roll 28 mm running, then
  press 18, pin-push 18 (~09:00), size films.
State 11-16 05:31 (autopilot): step 40, cells 1-4 of 8 done, so every task has its large size. Pin-push 45 mm
  (05:30, 59 min): rho 0.41 (0.50 at 25 mm), single seed 0.36 vs ceiling 0.79, mean J HOM 0.297, planner 0.368;
  best HOM (105, 65) 0.50 (planner 0.49), planner (105, 125) 0.59 (HOM 0.49); ten-best median x_sep HOM 97.5,
  planner 100 mm (72.5 and 85 at 25 mm), 2 of 10 shared. Twist 28 mm running, then roll 28, press 18, pin-push 18
  (~08:50), size films.
State 11-16 04:33 (autopilot): step 40, cells 1-3 of 8 done. Press 45 mm (04:31, 45 min): rho 0.46 (0.60 at 25 mm),
  single seed 0.42 vs ceiling 0.66, mean J HOM 0.316, planner 0.523; best HOM (120, 100) 0.81 (planner 0.73), planner
  (95, 55) 0.76 (HOM 0.59); ten-best median x_sep HOM 100, planner 112.5 mm (75 and 90 at 25 mm), 5 of 10 shared.
  Pin-push 45 mm running (~05:15), then the four small sizes (~08:20), size films.
State 11-16 03:48 (autopilot): step 40, cells 1-2 of 8 done. Roll 67 mm (03:46, 39 min): rho 0.54 (0.38 at 40 mm),
  single seed 0.51 vs ceiling 0.86, mean J HOM 0.105, planner 0.190; best HOM (125, 75) 0.35 (planner 0.43), planner
  (140, 70) 0.57 (HOM 0.22); ten-best median x_sep HOM 115, planner 125 mm (77.5 and 112.5 at 40 mm), 5 of 10 shared.
  Press 45 mm running; the last lane line size_films (report + films of the cell whose best pairs lie farthest apart).
State 11-16 03:10 (autopilot): step 40, cell 1 of 8 done (twist 67 mm, 03:07): --d checked (HOM's runs at 67 mm
  reproduce the grid, rho 0.97); 100 pairs: rho 0.66 (0.81 at 40 mm), single seed 0.59 vs ceiling 0.80, mean J HOM 0.288,
  planner 0.402; best pair HOM (150, 55) 0.74 (planner 0.34 there), planner (95, 105) 0.88 (HOM 0.33); ten-best median
  x_sep HOM 132.5, planner 120 mm (100 and 90 at 40 mm). Planner replanning 2.8-2.9 s per replan at 67 mm; cell 63 min.
  Roll 67 mm running; grid rows (25 MB) stay out of git like the base grids.
State 11-16 02:09 (autopilot): steps 7 and 39 done; step 40 runs. Step 7: open-hand UHAS (SCS0009) held 0 of
  5,232 test rollouts, J 0.000, every training episode dropped; benchmark page 808ccdfdb, artifact 6Pen2iF2bpC4YTWh13eBrb
  v2. Step 39: HOM page v7 section 6 (9e9e63c84, artifact 83AjHSqqWZXRuif4nTFjCa v7): gain mostly tracking on all four
  tasks; roll rho 0.21 at single seeds vs ceiling 0.64; films roll (105, 70) g6 (index pad leaves at 3.0 s, rigid parts
  1.7-2.7 N) and pin-push (80, 65) g0 (even at seeds 0-1, HOM J 0 at seed 10). Step 40: lane lines sz_* since 02:04
  (twist 67 grid 26,067 worlds, ~20 min; cells 45-65 min, all eight by ~09:15). Report:
  .venv/bin/python scripts/hom/hom_size_rank.py --json docs/experiments/20261114-hom_inner_loop/data/20261116-size_rank.json
  --fig docs/experiments/20261114-hom_inner_loop/media/20261116-size_rank_xsep.png (base sizes: rho 0.81/0.38/0.50/0.60
  over the 100 pairs). Then the HOM page section (table per task and size, x_sep figure, films of the cell whose two
  best layouts differ most), commit, push, republish.
State 11-16 01:55 (autopilot): steps 7, 39, 40 open. Open-hand UHAS on the SCS0009 plant learned nothing (every
  episode dropped, J proxy 0.0 over 5,059 iterations; stopped at 81.8 min, 01:52); open_eval and open_film run now.
  Step 39: data/20261116-anneal_gain_regimes.json and media/20261116-anneal_gain_by_xsep.png written in
  docs/experiments/20261114-hom_inner_loop (numbers match the window's); section next. Step 40: inner_loop_mpc.py --d
  (rows, pairs, films tagged _d<d>; base names unchanged). Lane logs/20261115-uhas_sv/lane.txt lines 33-66 after
  open_film: gain_film_roll (_all pair 280), gain_film_pinpush (pair 192), then per (task, size) sz_grid (hom_gpu grid
  into grid2dff_v4b_sv25r/<task>_d<d>.jsonl), sz_pairs (rank), sz_hom, sz_f2 in the order twist 67, roll 67, press 45,
  pin-push 45, twist 28, roll 28, press 18, pin-push 18 (~50 min each; first four by ~05:50, all by ~09:40).
State 11-16 00:33 (autopilot): only step 7 open. Done since 23:49: steps 33 and 36 (films relabelled, HOM v6 and MPC
  v11 republished, hand 7b1186f08), 37 and 38 (contact 121cc8b: intro film with the reconfiguration, morphology and
  dynamics openers, posters, plain film paths). The open-hand UHAS rerun on the SCS0009 plant trains till ~02:45
  (results/uhas_sr2/20261116-*_twist_open_s0), then open_eval and open_film (logs/20261115-uhas_sv/lane.txt, ends
  ~03:00). Then: rebuild scripts/hom/inner_loop_benchmark_page.py (open-hand row, lede sentence, film _open, check
  its poster), commit, push, republish https://claude.ai/artifact/6Pen2iF2bpC4YTWh13eBrb with page_films.py files,
  mark 7 done, Status: done, ap disarm.
State 11-15 23:49 (autopilot): step 33 rows done and pushed (8bc605d4b, page 118d44ba6): full annealed sampling 0.408
  -> 0.421 through the filter; gap to HOM -0.003..+0.111 filtered vs -0.028..+0.125 raw. Not yet republished: the
  refilm_* lines (23:48 to ~00:30) overwrite the films the MPC page links; after them delete
  docs/experiments/20261114-hom_inner_loop/media/*-film*.mp4, rebuild HOM and MPC pages, commit, republish HOM v6 and
  MPC v11 (page_films.py files), mark 33 and 36 done. Then open_train/eval/film (~02:50) for step 7.
State 11-15 23:29 (autopilot): step 33 partial: filtered pose raised real-time PS 0.255 -> 0.307 and real-time
  annealed sampling 0.353 -> 0.369 (HOM's tracker 0.283 -> 0.310); gap to HOM -0.028..+0.070 raw, -0.003..+0.059
  filtered; full annealed sampling (plf_anneal) until ~23:50. MPC builder updated (Table 25 column, paragraph).
State 11-15 23:26 (autopilot): step 15 done (contact-point filter: reversals 7.4 -> 2.7/s, J 0.318 -> 0.220, held 95
  -> 57 %; MPC builder has the paragraph, not yet built/committed). Single-layout UHAS (SCS0009): J 0.35 at (105, 70)
  vs HOM 0.76, on the benchmark page (1484222c0, not yet republished). Step 33: plf_ps_rt done 23:25 (rows
  mpc_ps_twist_d4nf0.2_r10h10k2i1), plf_rtD/plf_anneal running till ~23:50; then refilm_* (~40 min), open_* (~02:50).
State 11-15 22:48 (autopilot): steps 2-6 and 28 done. Pushed: hand 8272fe12a, 061fb258e (benchmark page, artifact
  6Pen2iF2bpC4YTWh13eBrb), 6f72b77e9 (HOM v5, MPC v10, charts, figures.json), 227f6021f (inner_loop_mpc --plan-filter,
  --cop-tau, hom_gpu cop filter, annealed-sampling labels); contact ffd70bf (v2 deck: rename, notes, plain film paths).
  HOM v5 and MPC v10 republished with films as files. Queue logs/20261115-uhas_sv/lane.txt (32 lines): single_train
  (till ~23:17), single_eval, pf_tip_cop0.2/0.5 (step 15 c), plf_ps_rt/plf_rtD/plf_anneal (step 33), refilm_* x13 (step
  36: films relabelled), open_train/eval/film (~02:50). MPC builder already reads _d4nf0.2 and _tip_pf0.2c* rows (built
  locally, not committed). When rows land: rebuild MPC page (prose flags 302/1000, trim), HOM page after refilm (delete
  docs/experiments/20261114-hom_inner_loop/media/*-film*.mp4 first), benchmark page after single/open; republish.
State 11-15 22:35 (autopilot), report for the owner (step 6): benchmark page
  docs/experiments/20261112-inner_loop_benchmark/20261112-inner_loop_benchmark.html, https://claude.ai/artifact/6Pen2iF2bpC4YTWh13eBrb
  (8272fe12a, 061fb258e). Outer loop (search methods page, Section 12): by 6,000 rollouts BO came within 0.05 of the best
  J in 94-100 % of 200 replayed runs at 0.63-0.85 of the EA's median rollouts; EA64 88-100 %, CMA-ES 80-100 %,
  MAP-Elites 49-99 %, random 52-99 %. KaRMA: first choice 0.18-0.61 below each task's best J (twist 0.55 vs 0.76, press
  0.01 vs 0.62), rho with J 0.09-0.52, 379-620 s per layout on one core against HOM's 1.7-2.8 s of GPU. UHAS (SCS0009,
  142 min PPO on the twist): mean test J 0.50 per layout against HOM's 0.25, held 69 vs 75 %; first choice (40, 80)
  scores 0.18 under HOM; targets 8,154 vs 136 deg/s RMS, 37.8 vs 2.6 reversals per joint per second, saturation 84 vs
  20 %, drift 31 vs 12 mm; limit terms: 0.39, 47.6 reversals/s, 58 % saturation. Cost for the twist's 527 layouts: HOM
  25 min, KaRMA 11.3 h on 8 cores, UHAS 142 min + 21 min, a policy per layout 11 days. The 11-12 numbers (0.32 vs 0.25,
  56 N grip) were on the work plant and are replaced. Open-hand and single-layout UHAS rerun on the SCS0009 plant in the
  queue (~02:00), then step 7 finishes. Next on offer: live BO at the size study's sizes (~3 h GPU); UHAS on the roll,
  pin-push and press; KaRMA on the full lattice (~11 h on 8 cores).
State 11-15 22:33 (autopilot): lane done 22:24 (rows 8b89f4c0a); queue logs/20261115-uhas_sv running (UHAS films
  done, express, then grid films, single, open; ~02:00). Steps 2, 3 done. Benchmark page built with films (linked from
  media/ since 2ec0394ec; publish with page_films.py files). Four grids: twist rho 0.76, 1 of 10 outside, fresh +0.059
  (-0.044 to +0.163); roll 0.23, grid's first = planner's best, +0.336; pin-push 0.64, 6 outside, -0.013; press 0.65,
  9 outside, +0.067. figures.json and the three v2 charts regenerated (uncommitted); contact deck rebuilt with the
  rename, four-grid/iLQR-pad/five-placement notes and plain film paths (direction 34; no '.mp4?v=' left), uncommitted.
  Next: commit benchmark page (step 4), publish (5); HOM/MPC pages after the grid films; commit hand, rebuild deck,
  commit + push contact.
State 11-15 22:23 (autopilot): lane at alltop_f2_twist, then finish; queue logs/20261115-uhas_sv/lane.sh waits for
  it (UHAS films, express, grid films, single and open UHAS reruns). Step 28 drafted, not yet built with the twist:
  hom_inner_loop_page.py (sec_grids four grids, five placements, iLQR pad term, future work), inner_loop_mpc_page.py
  (pad sentence, future work, task phase 2.0 s: T_LIFT 1.6 + 0.4, the pages said 2.4 s), both renamed DIAL -> annealed
  sampling on string spans (owner 21:5x; DIAL-MPC cited once); codesign_section_figures.py (grids, five, ilqr pad,
  renamed chart labels and rows_s keys) and contact build_talk.py (keys, grid_note four grids, ilqr_note pad,
  five_note, rename) not yet run. Step 33 patch ready: logs/20261115-uhas_sv/patch_plan_filter.py (apply when no
  queue step importing inner_loop_mpc starts; then --plan-delay 4 --plan-noise --plan-filter 0.2 runs).
State 11-15 21:45 (autopilot): step 2 done (page built, lede, lint 17 words/sentence, no banned words). The 11-12 UHAS
  rows, films and scorecard were work plant, so the page uses the SCS0009 runs of 11-13 (20261113-2115 published, -2347
  limits): per test layout UHAS 0.502 vs HOM 0.247, held 69 vs 75 %; scorecard reversals 37.8 vs 2.6 /s, saturation 84 vs
  20 %, 8,154 vs 136 deg/s. Summary of the 5 MB test rows: rows/uhas_twist_test_summary.json. Queue
  logs/20261115-uhas_sv/lane.sh (pid 2457595, waits for "lane done" of the follow-ups lane, ~22:30): UHAS films (SCS0009),
  express x3, grid films twist/roll/pin-push (HOM page), then single-layout (30 min) and open-hand (120 min) UHAS reruns;
  ~3.5 h, ends ~02:00. Step 28 under way: hom_inner_loop_page.py sec_grids (four grids; roll rho 0.23, grid's first =
  DIAL's best; pin-push 6 of 10 outside, fresh -0.013; press as before), five placements (drops 26.5->24.5, 31->27 %, 0
  drops on changed grasps), iLQR pad term (twist +0.113, mean 0.434 vs full DIAL 0.426). Owner 21:4x: slides may change
  where numbers change (direction 31).
State 11-15 17:44: for the autopilot. Lane logs/20261115-followups runs all_f2_pinpush (all_hom_pinpush done
  16:21), then the pin-push and twist grids; it ends with `finish` (finish.sh commits the rows), about 21:00-23:00. While
  it runs: no GPU work; `ap snooze` (max 240 min) and check queue.log for "finish rc 0". Then step 28 (HOM page v5, MPC
  page v10, deck notes; this login, colorado, owns both artifacts). Steps 2-7: the 11-12 KaRMA/UHAS page
  (20261112-inner_loop_benchmark.html) was never built or committed (no file, no git history; its media exist with
  _work10 names); build it per the steps, CPU only, any time. Step 15 (a) and (b) are done (11-13 tracker throughput
  page, the UDP link); (c) the fingertip CoP filter is open. Step 29 done; step 30 waits on the owner (objects, tag maps,
  masses, live read-back). Bench state: station running since 17:42, homed by adoption, servo torque off; J4's motor
  was unplugged (replugged 17:38); gantry page docs/experiments/20261115-gantry_travel/20261115-gantry_travel_dip_tags.html
  (https://claude.ai/artifact/DbxLVrunj4EAWHC3YJJuUc); corvus at 96bfe1111.
State 11-15 16:41: owner 16:0x: CB1 go (patch applied ~16:12 with git apply, backups ~/backups/20261115-stream_read_back/,
  station restarted, unhomed, torque off since 16:38 for the tags); pin-push layout (75, 56.8) (plans rewritten and
  verified, SIL 0.373/0.764/0.364/0.421); DIP-link tags 14 mm ids 24-26 (assets/tracking/20261115-dip_link_tags,
  2edae956b) and the gantry travel tools (docs/experiments/20261115-gantry_travel: lab look/pose/sweep, fit checked on a
  synthetic sweep); make_bench had no hold 14:15-16:22 (fixed 2edae956b). Owner's transfer concerns: pad-law refit on
  the v3 IDs 1-3 presses (92964cd48, docs/experiments/20261115-pad_law_refit/data): linear law 1.58 -> 0.44 by level;
  spring or softening laws flatten 1-5 N but do not transfer sphere <-> rod; rig compliance unmeasured. corvus at
  001fd3b38. Lane: all_f2_roll done 16:05, pin-push and twist grids running.
State 11-15 14:16: owner 13:0x "sure go ahead" (step 28: lane logs/20261115-followups, five placements done, iLQR pad
  term twist +0.113, DIAL grids ~22-23 h) and "make sure we are ready to start collecting data and calibrating from real
  trials of HOM" (step 29). Bench: runbook docs/experiments/20261115-hom_bench_trials/README.md; hom_server serve/sil/
  nominal (UDP = in-process loop; 14 plans on cad_v2 at 0.55 N m/rad, J 0.008-0.935, roll layout fails on cylinders);
  lab/hom_trial.py; tracker interpolates the arm pose at capture time; CB1 read-back patch staged (cb1/, not applied,
  HOM drops the sphere without measured joints); corvus at b729cd928. Waiting on owner: CB1 go, pin-push layout,
  tag-map views, masses. Scorer and replay calibrator: background builder.
State 11-15 12:52: iLQR done (lane logs/20261115-ilqr, ended 12:50): 20 pairs per task at seeds 0-1, J 0.383/0.449/
  0.276/0.491 (mean 0.399; HOM 0.281, full DIAL 0.425), within full DIAL's interval on roll/pin-push/press, -0.160 on
  the twist (three-pad share 0.45 vs 0.60); 95x HOM's GPU per twist rollout (full DIAL 209x); one hand 0.29 s per
  replanning; 4 iterations no better on _tune; repeat |dJ| 0.006-0.091. Cylinder runs need one 6-hand batch per process
  (each new batch adds a 3.2 GB collision buffer; 16 hands OOM). Pages built (HOM 265, MPC 294 flags per 1000);
  figures.json controllers_track has press_grid and ilqr; contact build_talk.py has grid_note/ilqr_note (not built).
  12:54: hand 6ccfed345 pushed; HOM page v4 https://claude.ai/artifact/83AjHSqqWZXRuif4nTFjCa, MPC page v9
  https://claude.ai/artifact/3m4bcPKugwFCtX5ZZLtd7f; contact 0efd93e (guide and notes: press grid, iLQR) pushed;
  ledger 26 and 27 done. Open: steps 2-7, 15 (older); future work on the HOM page (pad residual in iLQR, full DIAL on
  the other three grids, five grasp placements) waits on the owner.
State 11-15 11:57: press grid done (lane logs/20261115-press_grid, rows c93a1a9c0): HOM vs full DIAL rho 0.65 over 369
  layouts; criterion met (9 of full DIAL's ten best outside HOM's 50 best; 7 of ten within the grid's 100 best); at
  seeds 10-11 full DIAL's ten +0.067 (+0.017 to +0.115) over the grid's ten under DIAL, 0.384 below under HOM (pads all
  in contact 52 vs 97 % of the task). One hand: residual 0.25 s, DIAL 2x32 0.08 s per 0.1 s replanning. HOM page
  builder updated (press grid, fresh seeds, films p3/p10, J = 0 causes); built once, not committed. iLQR (step 27):
  patch applied 11:49; Joseph-form backward pass fixed NaN gains (first smoke J 0, then 0.317/0.136); lane
  logs/20261115-ilqr (il2_<task> on _tune, one_il2_twist --chunk 1) from 11:54; twist 0.276 (HOM 0.253, DIAL 2x64
  0.387). MPC page builder: iLQR row in TUNE, ilqr_text paragraph, refs [6]-[8]. Next: results, both pages, commit,
  push, republish HOM v4 and MPC v9.
State 11-15 10:20: owner 09:4x "sure go ahead" (full DIAL on all 369 press layouts). Lane logs/20261115-press_grid
  (lane.txt; 09:49): all_hom_press done 09:54; all_f2_press (738 worlds, ends about 11:35); then alltop (each
  controller's ten best and the grid's ten at seeds 10-11), one_res/one_c1 (--chunk 1 replanning), films_press (HOM and
  full DIAL at seed 10), finish.sh (commits the rows). inner_loop_mpc.py gained pairs --rule all and top --src (atomic,
  09:50). HOM page builder (not yet built): sections pressgrid and the J-floor/power table in metrics (owner 10:0x: "the
  scores are very low, so statistical power is also lower"), one-hand timing in fusion, conditional future work. Owner
  10:0x on iLQR/DDP/FDDP: scripts/hom/inner_loop_ilqr.py written and CPU-checked; apply logs/20261115-press_grid/
  patch_ilqr.py to inner_loop_mpc.py after the lane's last import, then smoke-test on _tune (step 27).
State 11-15 06:40: lane done 06:24 (var_*, fu_*, tm_* rc 0; lane.sh idle, polling lane.txt; "END" stops it). Step 25:
  HOM page text for the variants (2 of 13 raised J: the shared pick's forces +0.214 on the press, a 50 ms preview +0.013
  on the roll; lead 0.15 rad and the force loop lowered J on every task run) and the fusion (DIAL residual res_anneal
  2x32: +0.065/+0.133/+0.114 over HOM on twist/roll/press, within full DIAL's interval on pin-push and press, 46x HOM's
  GPU per twist rollout against 28x for DIAL 2x32 alone and 209x for full DIAL); four task films checked. tm_full rows
  fill the MPC tuning table's GPU column (2.37-4.64 s/s); tm_chunk16/64 rows are batch-throughput checks no page reads.
  figures.json controllers_track gains per-task residual numbers and residual_cost; talk v2 notes and guide get one
  residual sentence (about 24 days as the search's inner loop). Waiting on owner: f2 on the full press grid (1.6 GPU h).
State 11-15 05:05: top_* and patch_* done (04:45). Fresh seeds 10-11 on each controller's ten best (rank-run J at
  seeds 0-1): under f2 HOM's ten lost 0.010-0.036, f2's own ten up to 0.129 (press); sets then within the bootstrap
  interval on twist/pin-push/press (+0.021 press), -0.291 on the roll. HOM dropped the press cylinder 11/34 at 10-11 vs
  2/34 (f2 none). Fused pick (f2's best of HOM's ten) = f2's own first choice on twist and pin-push; -0.040 roll, -0.177
  press. Repeat test (patch vs rank runs, same layout and seed): HOM |dJ| 0.0009/0.0041, f2 0.082/0.071 (one sampling
  generator per batch). Patch smoothness HOM 0.82/0.66, f2 1.03/0.70. Hand 58eecfc99 (pushed); MPC page v7, HOM page v2;
  contact 1b5256f (deck notes + guide, pushed). Lane on var_* (from 04:46), then fu_*, tm_*, finish_end ~07:30; then step
  25's variants/fusion text check and republish.
State 11-15 03:30: ranking runs done on the four tasks (HOM; c1 = twist pick DIAL 2x32 H10; rt30 = DIAL 2x16 H30
  every 0.2 s; g1 = gradient 2x32; f2 = DIAL 4x64 at a 2 ms step; HOM at seeds 10-11). Mean J: HOM 0.264, f2 0.400 at 209x
  HOM's GPU per rollout, real-time planners 0.151-0.225 at 26-28x. HOM vs f2 rho 0.81/0.38/0.50/0.60; under f2 HOM's 50
  best layouts score 0.15-0.33 above 50 random; best of HOM's ten within 0.035 of f2's best on twist/roll/pin-push, 0.134
  short on the press (f2's press best at y_sep ~105 mm vs HOM's 65, where HOM drops at new seeds; HOM's drop rate at
  seeds 10-11 17 % vs 8 %, press 31 vs 10 %). Commits 637be87ea, 1ca6479f3 (pushed); contact ea6a5fa (deck v2 + guide,
  pushed). MPC page v6 https://claude.ai/artifact/3m4bcPKugwFCtX5ZZLtd7f; HOM page v1
  https://claude.ai/artifact/83AjHSqqWZXRuif4nTFjCa (docs/experiments/20261114-hom_inner_loop/). Lane (lane.txt): films,
  top_* (fresh-seed confirmation), finish_mid, patch_* (7x7 neighbourhood of HOM's best twist/press layout, HOM and f2),
  var_*, fu_*, tm_*, finish_end, about 07:30. finish.sh never adds docs/overviews/printed_fingertips.html (poller's).
State 11-14 22:55: commit ca590f710 (not pushed). Sweeps logs/20261114-mpc_tasks/tune_rt.sh (done) and tune_rt2.sh
  (larger budgets, ends ~23:35); then lane logs/20261114-mpc_tasks/lane.sh (93 lines, lane.txt): 0.3 s-horizon timing
  and tuning, the _rank runs (hom, c1 = twist pick, c7, f2 = DIAL 4x64 at 2 ms, g1 = gradient 2x32), films per task,
  HOM variants on _rank20 (preview, shared picks, Eq. (2) damping, u_max, lead, force loop, axis feedback), fused
  planners (res_anneal, hs_anneal), throughput timing at 16 and 64 hands; ~6 h. Builders: inner_loop_mpc_page.py
  (sections tasks, rank100), new scripts/hom/hom_inner_loop_page.py (draft built), codesign_section_figures.py cost
  --track (Steel measured 3.32x; DIAL rows 28.7x and 401x HOM per twist rollout: 15 and 215 days) and controllers
  (v2 deck figure, after the rank runs). Contact repo build_talk.py: d-speed notes and lead updated (not committed).
  Tuning-pair results: the roll needs a 0.3 s horizon (DIAL 4x16..4x64 at H30: 0.18-0.26; H<=20: <=0.011; HOM 0.096);
  gradient 2x32: roll 0.089, pin-push 0.077; HOM's grid seed rho 0.78/0.65/0.65/0.33 (twist/roll/pinpush/press).
State 11-14 22:40: owner 22:0x: "is this objective score across multiple tasks? we should run them if not"; may
  improve HOM (better than DLS? fuse methods? gradient-based?); "emergent trends in the hyperparameters such as impedance
  forces, twist weights"; run the ranking test on real-time DIAL; "keep things clean, video documented, and actually
  documented"; then 22:1x: update the talk v2 deck and guide (speed timings, Steel, comparison to other methods, evidence
  that HOM is the right choice: "stability across metrics, tasks, solutions? ... explore this"). Done: inner_loop_mpc.py
  --task (roll, pinpush, press; cost from task_refs: cylinder axis error, roll centre path, pinch point), --pairs
  (_rank: HOM's 50 best + 50 random per task; _rank20; _tune: 6 others per task), method grad (FD gradient + line
  search). Found: the twist's real-time pick fails on the roll and pin-push (J 0.00/0.01 vs HOM 0.33/0.43 on 2 pairs);
  full DIAL 4x64 reaches 0.73 on the roll (HOM 0.33). Sweeps logs/20261114-mpc_tasks/tune_rt.sh, tune_rt2.sh (configs
  c1-c7, f1-f6, g1-g2 on the _tune pairs). 11-11 level-hom picks (rows/sv25_track): forces below the defaults (twist
  1.89 N, roll 2.18, press pinch 1.76 vs 2.5/2.5/3.0); roll force rho -0.55 with x_sep, +0.56 with y_sep.
State 11-14 20:00: lane lines 124-126 done (10:57-11:02). On the 2x32 pick: rate-change cost w1 1.06 -> 0.52
  reversals at J -0.021 (-0.055 to +0.012); with the 3 deg dead band 0.45 at -0.027; w2 with the dead band 0.42 at -0.057
  (-0.107 to -0.012). Pick + w1 J 0.390, HOM 0.388 (+0.002). New page section "Mean J and ranking of the pairs under HOM
  and the planners": real-time pick vs HOM +0.024 (-0.044 to +0.087), DIAL 4x64 +0.074 (+0.019 to +0.138) at 11.7 s of
  GPU per s; Spearman rho over the 20 pairs HOM vs DIAL 4x64 0.89 (pick vs DIAL 0.89, iCEM 0.87; seed 0 vs 1: HOM 0.99,
  DIAL 0.97). Under the tag tracker (40 ms, noise) HOM reverses 9.6/s (1.96 with the pose filter), the pick 1.25/s; at
  friction 0.4 HOM 0.081 vs DIAL 0.035. Owner 11-14 19:4x: "no point to using HOM besides throughput". Page version 5.
State 11-14 10:55: lane done (123 lines, 02:18). MPC page rebuilt and republished, version 4 of
  https://claude.ai/artifact/3m4bcPKugwFCtX5ZZLtd7f (commit 2a37d966f): contact over five plants (HOM 0.388 -> 0.275 from
  pads to point contact, monotone in torsion), real-time pick 2x32 at 2 ms J 0.411, latency, reversal options (no
  sampling option helped; rate-change cost w1 1.30 -> 0.71 reversals at J -0.023), UHAS on the SCS0009 plant (test split
  0.261 vs HOM 0.085 at 37.8 reversals/s). Ledger 9, 10, 14, 16, 17 done. The benchmark page (steps 2-7) is still the
  work-plant draft in logs/20261112-mpc/; the MPC page's link to it is broken until it is rebuilt on the SCS0009 plant.
State 11-13 16:00: owner 15:5x asked to reduce the planners' target reversals, algorithmically and by cost shaping. Analysis: targets are
linear between replannings, so reversals happen only at replannings (max 10/s); MPPI/DIAL's temperature 0.03 std(c) gives
ESS 1.1 (each iteration moves the plan by ~0.96 of a sample's noise); with H = R (every real-time K=2 config) the shifted
plan holds the last target, so each replanning restarts from a zero rate. Commit 448ba4605: --warm extrap, --lam-rel,
--sigma, --rate-w, --acc-w, --rev-w, --err-db; page section 'Target reversals of the sampling planners'. Lane lines
94-107 (logs/20261112-mpc/lane_insert_1113c.txt): 11 options on DIAL 3x16 dt2 (J 0.431, 1.30 rev/s), PS rt extrap,
full DIAL with temperature 1 and rate-change cost 1; ~17:40-18:55.
State 11-13 15:50: 2x2 done (J, 40 runs; plant pads / point): HOM 0.388 / 0.275; PS rt pad model 0.335 / 0.196, point
model 0.332 / 0.247; DIAL 4x64 pad model 0.461 / 0.328, point model 0.426 / 0.442. Pad minus point model: DIAL +0.035
(-0.000 to +0.072) on pads, -0.114 (-0.180 to -0.056, 0 of 20) on point. Own model pads vs point plant +0.019 (n.s.).
Films media/20261113-film_mpc_twist_p0_s0_models.mp4 and _plant_pt.mp4 checked. Real time (one hand, s per 0.1 s):
1x64 1 ms 0.090, J 0.387; 3x16 at a 2 ms planner step 0.101 (1 % over), J 0.431; 2x64 2 ms 0.099; 2x32 0.082; 2x16
0.068; 2x64 1 ms 0.197; 4x64 h30 2 ms 0.627. Added before the pick: timing 3x12 and J runs 2x32, 2x16, 3x12 at 2 ms.
State 11-13 14:45: owner 14:2x asked for the planner with point-contact tips on a point-contact plant, the pad model on
that plant, HOM vs DIAL there, "quantifiably and with a clear trend"; then PS on the same ablation and "an annealed
sampling config that can run in realtime with minimal performance loss". Commit 0fa84188e: inner_loop_mpc.py
--plant-model pads|pt|pt_t<mm> (pt_t: condim 4 on the tip meshes, torsional coefficient on every geom; XML written once
beside the pt scene as ..._pt__..__tor<mm>mm.xml; the pads' pinch torque ~1.4-1.8 mm by the 10-01 rig), --plan-dt,
rows named _n<n>, timing --configs; uhas_sr2.UBatch(phys_dt); page builder sections "Annealed sampling at the real-time
budget" and "Fingertip contact model of the plant and of the planner" (shown once rows exist). Lane: 42 lines inserted
after sv_spider_wc (staged copy logs/20261112-mpc/lane_insert_1113b.txt): 2x2 (HOM, PS rt, DIAL 4x64 on pads/pt plants
and models), two films (p0, pads and pt plants), rt timing + 8 rt J runs, HOM and PS rt on pt_t0.5/1/2, then
logs/20261112-mpc/pick_rt.py writes rt_pick.txt (best real-time DIAL by J) that the sv_rtD_* lines read with $(cat):
DIAL rt on every plant with pad, point and own models, and from 40/140 ms old noisy states. Ends ~19:00; the old lane
(mu04, ps, mppi, cem, UHAS) follows. Then rebuild the page, commit page + rows + media + indices, republish.
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
- [x] 2. Build the page: .venv/bin/python scripts/hom/inner_loop_benchmark_page.py. Rewrite lede() so it states the ou… Done 11-15 21:40: Page built on the SCS0009 plant (the 11-12 UHAS rows were work plant): lede with HOM 1.7-2.8 s/layout, KaRMA 379-620 s and first choice 0.18-0.61 below each task's best J, UHAS 0.50 vs HOM 0.25 per test layout (held 69 vs 75 %), reversals 37.8 vs 2.6 /s, saturation 84 vs 20 %; prose 17 words/sentence, 0.8 % over 30, 156 flags per 1000; no banned words
- [x] 3. Look at every figure and film poster at displayed size (media/20261112-karma_maps.png, 20261112-uhas_vs_hom.p… Done 11-15 22:32: Figures redrawn for 860 px display (KaRMA maps 2x4 at 14 pt, 13.5 px text; UHAS vs HOM legend moved off the points); film posters of the SCS0009 films (HOM's best test layout, UHAS's first choice (40, 80), limit terms) checked at display size and readable; the open-hand film's poster is checked when its SCS0009 rerun lands (logs/20261115-uhas_sv lane, step 7)
- [x] 4. Commit the page, its rows (docs/experiments/20261112-inner_loop_benchmark/rows/*), media and the docs indices… Done 11-15 22:34: Committed and pushed 8272fe12a: the page (SCS0009 plant), builder, rows (test summary, progress, film rows, express x3), media (two figures, three SCS0009 films and posters), .gitignore for raw test rows, docs indices
- [x] 5. Publish the page as an artifact (first publish) and give the owner the local path with the URL; the docs inde… Done 11-15 22:35: Published docs/experiments/20261112-inner_loop_benchmark/20261112-inner_loop_benchmark.html as https://claude.ai/artifact/6Pen2iF2bpC4YTWh13eBrb (version 1, colorado login), with its three films as files
- [x] 6. Report to the owner in the note's State and, when a window is open, in chat: outer loop (BO beats the EA in r… Done 11-15 22:36: Reported in the note's State 22:35 and in chat: outer loop (BO 94-100 % within 0.05 by 6,000 rollouts at 0.63-0.85 of the EA's rollouts; MAP-Elites 49-99 %), KaRMA (first choice 0.18-0.61 below the best, 379-620 s per layout), UHAS (0.50 vs 0.25 per layout, 37.8 vs 2.6 reversals/s, 84 vs 20 % saturation), costs per 527 layouts; offered live BO, UHAS on the other tasks, KaRMA on the full lattice
- [x] 7. Page: add the usability section from rows/usability_twist_20261112-1244_twist_s0.jsonl (scripts/hom/inner_loo… Done 11-16 02:00: Done 11-16 02:05: usability section (Table 3: medians over held rollouts, HOM vs UHAS published/limits), three films linked from media/ plus the open-hand film; the 11-12 work-plant numbers (0.32 vs 0.25, 56 N grip, 83 % saturation, 20 reversals/s) replaced by the SCS0009 runs (UHAS 0.502 vs HOM 0.247 per test layout, 37.8 vs 2.6 reversals/s, 84 vs 20 % saturation); the two >2 MB evaluation rows summarized in rows/uhas_twist_test_summary.json (0.19 MB, raw rows gitignored). Open-hand UHAS on the SCS0009 plant: 0 of 5,232 test rollouts held, J 0.000, every training episode dropped (5,059 iterations, 82 min). Page 808ccdfdb, republished https://claude.ai/artifact/6Pen2iF2bpC4YTWh13eBrb v2 with its four films as files.
- [x] 9. MPC and UHAS reruns (owner 11-12 21:0x go): wait for 'queue done' in logs/20261112-mpc/queue.log (launched 21… Done 11-14 10:54: Superseded: the work-plant queue was replaced by the lane's sv_ lines on the SCS0009 plant (UHAS published and limits s0 trained 120 min each, done 11-14 02:18); results in the MPC page's UHAS section.
- [x] 10. Page section on the MPC: per controller on the 20 pairs x seeds 0-1 (rows/mpc_*_twist*.jsonl): mean J, held s… Done 11-14 10:54: Done: docs/experiments/20261112-inner_loop_benchmark/20261113-sampling_mpc_vs_hom.html, Table 1 and Figure 1 (servo plant).
- [x] 12. Resguard at 01:38 on 11-13 (seen by the feed poller's session): the watchdog killed the queue step usability_… Done 11-13 08:47: Done 11-13 08:47: the kills (01:38 and again 08:44, PSI full 19.5-19.8 at 16 GB available) were numpy huge-page direct compaction (compact_stall 4.3 M, 99 % failed), not RSS; logs/20261112-mpc/queue4.sh sets NUMPY_MADVISE_HUGEPAGE=0, waits 1 min before each launch, retries rc 3 every 2 min and a kill once; the usability step then ran at memory pressure 0.00
- [x] 13. queue4 (launched 08:45, logs/20261112-mpc/queue4.sh, lines in queue.log): usability and pairs of limits_s0; f… Done 11-13 11:35: Superseded 11-13 10:30: queue4's runs were on the work plant; reruns on the SCS0009 plant go through logs/20261112-mpc/lane.sh (lines in lane.txt)
- [x] 14. Lane logs/20261112-mpc/lane.sh on the SCS0009 plant (owner 11-13: films/page to analyze, HOM-space controls, … Done 11-14 10:54: Done 11-14 02:18: all 123 lane lines rc 0 (steel_stage1 killed twice by the watchdog at 18:48 and 19:01, the Steel session's job, rerun by that session). Results on the MPC page (2a37d966f).
- [x] 15. Owner 11-13 13:2x: (a) measure HOM's controller time and the two-camera tag tracker (fps, CPU) on corvus afte… Done 11-15 23:22: (a), (b) done 11-13/11-15 (tracker throughput page, UDP link); (c) done: hom_gpu POSE_FILTER cop_tau and inner_loop_mpc --cop-tau (227f6021f); with the fingertip model and the pose filter at 0.2 s, filtering the contact points at 0.2 s lowered HOM's reversals 7.4 -> 2.7 per joint per second (tag tracker alone 2.0), but J 0.318 -> 0.220 and held 95 -> 57 %; at 0.5 s 2.8/s, J 0.131, held 47 %. Rows mpc_hom_twist_tip_pf0.2c0.{2,5}; MPC page section on the pose filter (next republish)
- [x] 16. Owner 11-13 14:2x: contact-model ablation and real-time DIAL. When the lane's sv_plpt_*, sv_rt_*, sv_pltor*, … Done 11-14 10:54: Done 11-14 10:54: page rebuilt and republished (2a37d966f, artifact version 4). HOM J 0.388/0.351/0.335/0.308/0.275 from pads to point contact over torsion 2/1/0.5 mm; real-time DIAL own model 0.411/0.415/0.416/0.393/0.342; DIAL 4x64 pt model on pt plant 0.442. Real-time pick 2x32 at 2 ms (0.082 s per 0.1 s) J 0.411; 140 ms old noisy state 0.219.
- [x] 17. Owner 11-13 15:5x, target reversals: when the lane's sv_revA_*, sv_revC_*, sv_revX_*, sv_revP_ext, sv_revF_* … Done 11-14 10:54: Done 11-14 10:54: no sampling option lowered the real-time DIAL's reversals (temperature 1.0: 3.86 vs 1.30; continued slope J -0.172, 10 of 40 dropped); rate-change cost w1: 0.71 at J -0.023 (-0.068 to +0.019); dead band 3 deg: 1.02 at -0.002; full DIAL acc1 2.02 -> 1.54 at -0.002. Page section 11.
- [x] 18. Owner 11-14 10:3x 'update?': lane lines 124-126 (sv_revR_acc1, sv_revR_acc1_db3, sv_revR_acc2_db3: rate-chang… Done 11-14 19:50: Done 11-14 20:00: on the 2x32 pick the rate-change cost w1 gave 1.06 -> 0.52 reversals at J -0.021 (-0.055 to +0.012), with the 3 deg dead band 0.45 at -0.027 (-0.063 to +0.008), w2 + dead band 0.42 at -0.057 (-0.107 to -0.012); pick + w1 J 0.390 vs HOM 0.388 (+0.002). Page: reversal table rows 13-16, new section on mean J and ranking (Spearman rho HOM vs DIAL 4x64 0.89).
- [x] 19. Tasks and ranking: pick a real-time planner config on the _tune pairs of the four tasks (logs/20261114-mpc_ta… Done 11-15 06:38: Done 11-15 06:45: tuning on _tune (tune_h30_*), _rank runs of HOM, DIAL 2x32, DIAL 2x16 H30 every 0.2 s, gradient 2x32 and full DIAL on four tasks, fresh seeds 10-11, films per task; MPC page sections tasks and rank100 (v8)
- [x] 20. Gradient planner (method grad, FD + line search) at the pick's and the full budget on the _tune pairs and the… Done 11-15 03:22: Done 11-15 03:30: grad (FD central differences + line search) at 2x32 real time on the 100 pairs per task: mean J 0.225 over the four tasks (HOM 0.264), seed rho 0.82/0.80/0.50/0.62; on the MPC page's tasks table and the HOM page (637be87ea)
- [x] 21. HOM improvements page (new, docs/experiments/20261114-hom_improvements/): diagnosis of where HOM loses J (tra… Done 11-15 06:38: Done 11-15 06:45: as docs/experiments/20261114-hom_inner_loop/20261114-hom_inner_loop_stability_variants_fusion.html (v3): where HOM loses J (two factors of J), 13 variants (damping, rate bound, lead, preview, shared picks, force loop, orientation feedback), fused planners (res_anneal, hs_anneal), CEM force trends, films
- [x] 22. Parameter trends of HOM's refined forces and contact modes (11-11 level-hom picks, rows/sv25_track): figure o… Done 11-15 03:22: Done 11-15 03:30: section 'Forces and contact modes picked for HOM per layout' on docs/experiments/20261114-hom_inner_loop/20261114-hom_inner_loop_stability_variants_fusion.html (figure 20261114-hom_picks_trends.png, 2x2 since 11-15)
- [x] 23. Evidence for HOM beyond throughput: seed reproducibility, one parameter set across tasks vs per-task planner … Done 11-15 03:22: Done 11-15 03:30: HOM page sections 'HOM's best layouts under full DIAL' (search curves, layout map), stability (seed rho, drop rates at new placements), the two factors of J; the patch smoothness comparison is step 25
- [x] 24. Talk v2 deck and guide v2 (contact repo talks/20261117-hand_design): speed chart with Steel measured (3.3x pa… Done 11-15 03:22: Done 11-15 03:30: contact ea6a5fa (pushed): slide hom-mpc from the controller chart (cost per planner, best of HOM's ten under full DIAL), guide answer 'Why HOM and not sampling MPC?', speed charts with Steel 43 h and full DIAL 112 days at the 2 ms step (209x)
- [x] 25. When the lane's top_*, patch_*, var_*, fu_* and tm_* lines are done (logs/20261114-mpc_tasks/queue.log): MPC … Done 11-15 06:38: Done 11-15 06:45: MPC page v8 (HOM's first choice vs DIAL's at seeds 10-11; tuning table's GPU column from tm_full), HOM page v3 (variants grouped by effect: 2 of 13 raised J; DIAL residual on HOM's servo targets +0.065/+0.133/+0.114 on twist/roll/press, within full DIAL's interval on pin-push/press, 46x HOM's GPU; fresh-seed fusion), four task films checked; hand d71ffe3b7, contact bdafd32 (residual sentence in slide notes and guide)
- [x] 26. Owner 11-15 09:4x 'sure go ahead': full DIAL (anneal 4x64, --plan-dt 0.002) and HOM on every press layout (36… Done 11-15 12:54: Done 11-15 12:55: full DIAL and HOM on all 369 press layouts (seeds 0-1): rho 0.65; criterion met (9 of DIAL's ten best outside HOM's 50 best, 7 inside its 100 best); at seeds 10-11 DIAL's ten +0.067 (+0.017 to +0.115) over the grid's ten under DIAL, 0.384 below under HOM (three pads 52 vs 97 % of the task); one hand: residual 0.25 s, DIAL 2x32 0.08 s per 0.1 s; HOM page v4 (press grid, J = 0 causes), hand c93a1a9c0 + 6ccfed345, contact 0efd93e
- [x] 27. Owner 11-15 10:0x: 'did we also try ilqr/ddp/feasibleddp on top of the gradient search?' (no: only the FD gra… Done 11-15 12:54: Done 11-15 12:55: iLQR (inner_loop_ilqr.py; Joseph-form backward pass after NaN gains) on the 20 pairs per task at seeds 0-1: 0.383/0.449/0.276/0.491, mean 0.399 (HOM 0.281, full DIAL 0.425), within full DIAL's interval on roll/pin-push/press, -0.160 on the twist (three-pad share 0.45 vs 0.60); 95x HOM's GPU per twist rollout (full DIAL 209x); one hand 0.29 s; 4 iterations no gain; repeat 0.006-0.091. HOM page v4 section iLQR, MPC page v9 tuning rows; hand 63a4e8fd9 + 6ccfed345
- [x] 28. Owner 11-15 13:0x 'sure go ahead' on the HOM page's future work: lane logs/20261115-followups (finish.sh comm… Done 11-15 22:47: HOM page v5 (https://claude.ai/artifact/83AjHSqqWZXRuif4nTFjCa, docs/experiments/20261114-hom_inner_loop/...fusion.html): four grids (rho 0.23-0.76; misses on pin-push 6/10 and press 9/10; fresh seeds +0.336 roll, +0.067 press, twist +0.059 and pin-push -0.013 within zero), five placements (drops 26.5->24.5 %, 31->27 %, none on changed grasps), iLQR pad term (0.433; twist 0.383->0.496); MPC page v10 (https://claude.ai/artifact/3m4bcPKugwFCtX5ZZLtd7f); rename to annealed sampling; task phase 2.0 s; hand 6f72b77e9, contact ffd70bf (notes, guide, charts, plain film paths)
- [x] 29. Owner 11-15 13:0x: 'make sure we are ready to start collecting data and calibrating from real trials of HOM' … Done 11-15 17:43: Done 11-15 17:4x: software ready for real HOM trials and their calibration. CB1 read-back patch applied ~16:12 (git apply), station restarted 17:42 with a motionless start (cb1/20261115-servo_hold.py) and the home adopted (all six stages homed at their stops, J4's unplugged motor replugged 17:38); pin-push layout (75,56.8) plans verified (SIL 0.373/0.764/0.364/0.421); trial_score merge/score, sr2_hom_replay replay/fit checked in simulation; hom_trial.py plan loads plans inline; DIP-link tags 24-26 and the gantry travel page (docs/experiments/20261115-gantry_travel, stages 0.99-1.00 with both cameras, hand ~2 deg yaw off P). Physical items left: step 30.
- [?] 30. Waiting on owner (11-15 17:4x): before a first real HOM trial, the printed HOM objects (the print stalled 11-… Waiting on owner: Physical items for the owner: printed HOM objects, tag maps of their blocks, masses, grip tape, and the first live read-back stream with the owner present. Nothing here for the autopilot.
- [x] 33. Owner 11-15 22:0x: 'why didn't we try HOM with the filtered, measured pose? arguably annealed sampling / othe… Done 11-16 00:31: Planners from the 40 ms old noisy pose through hom_gpu.POSE_FILTER's filter at 0.2 s (inner_loop_mpc --plan-filter, 227f6021f): real-time PS 0.255 -> 0.307, real-time annealed sampling 0.353 -> 0.369, full annealed sampling 0.408 -> 0.421 on the twist's 40 runs; HOM's tag tracker 0.283 -> 0.310; gap to HOM -0.003..+0.111 through the filter vs -0.028..+0.125 raw. MPC page Table 25 (v11, https://claude.ai/artifact/3m4bcPKugwFCtX5ZZLtd7f; rows 8bc605d4b)
- [x] 36. Film labels: re-render the HOM and MPC pages' films with 'Annealed sampling' in place of 'DIAL' (lane logs/20… Done 11-16 00:31: 13 films re-rendered with 'Annealed sampling' labels (refilm_* rc 0, 23:52-00:29); HOM page v6 (https://claude.ai/artifact/83AjHSqqWZXRuif4nTFjCa) and MPC page v11 republished with their films as files; commit 7b1186f08
- [x] 37. Owner 11-15 23:0x: the introductory slide (v2-anchor, 'Three fingers turning a ball') shows the alternative a… Done 11-16 00:33: Superseded by step 38's three films: cd_abstract_reconfig on v2-anchor with the lead naming the two kinds of degrees of freedom (six stage positions set before the task; nine finger joints moved by HOM; 0.77 at (105, 70), 0.65 at (70, 110)); contact 121cc8b
- [x] 38. Owner 11-15 23:3x, the talk's films (hand repo docs/experiments/20261110-codesign_formulation/media/, commit … Done 11-16 00:33: Records cd_abstract_reconfig (intro), cd_abstract_morphology (morphology opener), cd_abstract_pads_zoom (dynamics opener); control opener keeps cd_abstract_twist; posters from the .jpg beside each film (also given to the deck's other single-file films); plain film paths, no .mp4?v= in either page; captions name each clip; contact 121cc8b pushed
- [x] 39. Owner 11-16 01:1x: "can we characterize in what regimes annealed sampling recovers much better performance th… Done 11-16 02:07: Done 11-16 02:08: HOM page v7 section 6 'Layouts on which full annealed sampling gains most on HOM' (9e9e63c84; https://claude.ai/artifact/83AjHSqqWZXRuif4nTFjCa v7, docs/experiments/20261114-hom_inner_loop/...fusion.html): Table 6 (rho, seed rho, single-seed rho with ceiling, J = 0 counts, e_ref, median held errors, error-x_sep rho, ten-best median x_sep), Table 7 (HOM's outcome classes with gain, tracking and three-pad parts, planner's ten best; plus all layouts), Figure 4 (x_sep profiles), Films 13-14, terms (gain, parts, classes, ceiling), future work (per-finger contact fractions), two lede sentences. The window's claims hold, with three restated from the data: roll HOM error 8-11 deg below 100 mm and 13-49 deg above (not 20-49); pin-push below 75 mm mean gain -0.014 with three-pad shares 0.92/0.98; pin-push 105-125 mm three-pad shares 0.09-0.29 (HOM) and 0.32-0.68. Films (seed 10): roll (105, 70) g6, at 3.0 s HOM's index pad left the sphere and its rigid parts pushed at 1.7 N (2.7 N later), sphere turned off the command (17.6 deg RMS vs 2.9 for the planner); pin-push (80, 65) g0 was even at seeds 0-1 (0.462/0.472) but at seed 10 HOM missed by 12.4 deg (J 0) against the planner's 5.6 deg (J 0.31), both keeping the cylinder in the pinch.
- [ ] 40. Owner 11-16 01:4x, the go for an overnight GPU lane: "maybe we can also evaluate on a wider range of sizes now too, now that we have narrowed down our controls, going from 4 tasks. --> however many is feasible to evaluate by the time i wake up tomorrow. as well as using/evaluating annealed sampling as the inner loop for search more comprehensively (this is just rank correlation evaluation again)". Controls: HOM and full annealed sampling (anneal 4x64, --plan-dt 0.002) on the benchmark's plant (inner_loop_plant: SCS0009, 2.5 MPa pads; set the plant explicitly). Sizes, those of the servo-plant size study (rows 20261110-live_sizes_sv25_summary.json): twist and roll 28 and 67 mm spheres, press 18 and 45 mm, pin-push 18 and 45 mm. Measured costs (window 8955e04e): full annealed sampling 5.6-10.4 s per run (rows mpc_anneal_<task>_all_pdt2: 1.5-3.1 h per whole grid at two seeds), HOM 0.2-0.4 s per run; HOM's grid (every layout x grasp set x seeds 0-2, as grid2dff_v4b_sv25r) 15-30 min per task and size (its timing rows). Per (task, size): HOM's grid at that size on the benchmark plant, then the 100 rank pairs (inner_loop_mpc.py pairs --rule rank: HOM's 50 best and 50 random) under HOM and full annealed sampling at seeds 0-1: about 35-65 min, so the eight take ~6.5 h. Budget: one serial lane, after the UHAS lane (logs/20261115-uhas_sv, ~03:00) and step 39's two films, resguard status before each launch, done by ~09:30; order it so that every task's first new size runs before any second size (twist 67, roll 67, press 45, pin-push 45, then 28, 28, 18, 18), so a cut-off still covers all four tasks. Code: inner_loop_mpc.py takes the size from uhas_sr2.D_MM; add a --d option (scene, pairs file, rows tag, BP.j_rows on the new grid's rows) that leaves the base sizes' file names unchanged. Report per (task, size): Spearman rho of the two controllers over the 100 pairs, each controller's seed-to-seed rho and the ceiling sqrt(rho_H rho_A), mean J of each, each controller's best layout among the pairs with the other's J there, the x_sep of each one's ten best, and the best layout's x_sep against size under each controller (HOM's landscapes gave a size slope near 1 mm/mm). At the base sizes give the same 100-pair numbers beside the whole-grid rho of step 39 (0.23-0.76) to calibrate. Morning: State in the note by 09:00 with the finished (task, size) cells; then a section on the HOM page with a table per task and size, a figure of the best layout's x_sep against size per controller, and films of the (task, size) whose two best layouts differ most; commit, push, republish with page_films.py files.
Standing directions (for every step):
- Owner 11-12 17:40: KaRMA and UHAS are not fair comparisons; UHAS is less usable for many reasons, to be codified and quantified. The page reports UHAS's higher J and its usability numbers side by side.
- Owner 11-12 21:1x: the UHAS films look jittery and implausible; HOM is smoother than every method tried. Report smoothness with the scorecard (target reversals per joint per second, RMS target rate) for every controller, beside J
- Owner 11-15 21:4x: 'we can update the slides if we have relevant results to update the slides. no need to bloat things out though.' In step 28, update the v2 slides themselves (contact repo talks/20261117-hand_design), not only the notes and guide, where the afternoon's results change a number or a claim on a slide; add no slides. This lifts the note's 'Do not: change the talk decks' for those edits.
- Owner 11-15 21:5x: call the planner 'annealed sampling' on the pages, slides and notes ('full annealed sampling' for the 4 x 64 budget), not DIAL or DIAL-MPC; cite DIAL-MPC once, at first use, as the method's source. Rename in the builders and text you touch in step 28.
- Owner 11-15 22:3x: 'the videos still dont render on the talk slides'. Every deck film is committed, pushed and H.264, but linked as media_v2/NAME.mp4?v=HASH; Safari's media loader likely fails on the query when the deck opens from disk. In build_talk.py murl(), give films (.mp4/.webm/.mov) the plain path and keep the hash query for images only; rebuild both v2 pages, check no '.mp4?v=' remains in the HTML, commit and push the contact repo.
- Owner 11-15 22:4x: pages link their films from media/ instead of inlining them (commit 'Pages link their films from media/'): retro_style does it at every builder's exit. When publishing or republishing a page as an artifact, pass its films as files: python3 scripts/docs/page_films.py files PAGE prints the map; without it the artifact's films do not play. Commit the page's media/ films with the page.
<!-- /ap:steps -->
Claimed: scripts/hom/hom_search_qd.py, karma_sr2.py, uhas_sr2*.py, inner_loop_benchmark_page.py,
docs/experiments/20261112-inner_loop_benchmark/, logs/20261112-*, scripts/hom/hom_inner_loop_page.py,
scripts/hom/inner_loop_mpc_page.py, docs/experiments/20261114-hom_inner_loop/, logs/20261115-followups/.
Waiting on owner: whether to run Bayesian optimization live as the outer loop (size study's sizes, ~3 h GPU).
Do not: GPU jobs beside the running queue (they share the card); git lfs push --all; change the talk decks without the
owner; edit uhas_sr2*.py or hom_gpu.py while a queue step that imports them is starting.
