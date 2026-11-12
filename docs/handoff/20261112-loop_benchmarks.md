Status: active (window 8955e04e; armed as a safety net for the queues that end after this window)
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
- [ ] 1. Wait for "queue done" in logs/20261112-karma_sr2/queue.log and "open film rc" in logs/20261112-uhas_sr2/queue.log; every rc must be 0. On a failure read that step's log (train_*.log, eval_*.log, film_*.log, batch_*.log), fix the cause and rerun the step's command from its queue script; a resguard kill shows in ~/.claude/resguard_events.jsonl.
- [ ] 2. Build the page: .venv/bin/python scripts/hom/inner_loop_benchmark_page.py. Rewrite lede() so it states the outcome with numbers (time per layout of each evaluator; J of KaRMA's first choice on each task; UHAS's mean test J and held share against HOM's; the open-hand variant). Then python3 scripts/docs/prose_lint.py --top 20 on the page until within targets; search the text for the banned words of ~/.claude/CLAUDE.md.
- [ ] 3. Look at every figure and film poster at displayed size (media/20261112-karma_maps.png, 20261112-uhas_vs_hom.png, 20261112-film_hom_uhas_twist*.jpg) and fix what is unreadable.
- [ ] 4. Commit the page, its rows (docs/experiments/20261112-inner_loop_benchmark/rows/*), media and the docs indices it refreshed; push.
- [ ] 5. Publish the page as an artifact (first publish) and give the owner the local path with the URL; the docs index picks the URL up on the next build.
- [ ] 6. Report to the owner in the note's State and, when a window is open, in chat: outer loop (BO beats the EA in replay; MAP-Elites does not), KaRMA, UHAS, the costs. Offer next: live BO at the size study's sizes; UHAS on the other tasks; KaRMA on the full lattice.
<!-- /ap:steps -->
Claimed: scripts/hom/hom_search_qd.py, karma_sr2.py, uhas_sr2*.py, inner_loop_benchmark_page.py,
docs/experiments/20261112-inner_loop_benchmark/, logs/20261112-*.
Waiting on owner: whether to run Bayesian optimization live as the outer loop (size study's sizes, ~3 h GPU).
Do not: GPU jobs beside the running queue (they share the card); git lfs push --all; change the talk decks without the
owner; edit uhas_sr2*.py or hom_gpu.py while a queue step that imports them is starting.
