Copies of the press simulations of 2026-11-16 23:0x-23:45 (autopilot hand-20261116-press_sim_real), from
logs/20261116-sim_real_reversal/ on the workstation (gitignored), where they run. The comparison with the hand is
`scripts/hardware/hom_press_compare.py`; its results are in the bench README, section "State on 2026-11-16 23:45".

- press_stem.py: the eight press scenes (the 25 mm cylinder with the sphere's stem and tag block, 30.1 g; pads or point
  contact, impratio 1 or 100, noslip 0 or 30) and the centre-of-mass check (press_stem_check.json)
- press_predict.py: `run` (HOM in the loop on the calibrated servo, seeds 0-2, `--station`, `--mu`), `replay` (the
  targets that reached the hand's servos, open loop, `--table-mu`, `--mu`), `pipeline` (the design pipeline's model),
  `frames` (a run's frames along the pinch axis, Mesa's software renderer)
- press_calibrated.jsonl (144 runs), press_replay.jsonl (48), press_pipeline.jsonl (6): one row per run

To rerun, from the repository's root (each run is about 0.5 GB and 2-6 s):

    cp docs/experiments/20261115-hom_bench_trials/data/20261116-press/sim/press_*.py logs/20261116-sim_real_reversal/
    R=logs/20261116-sim_real_reversal; export WARP_CACHE_PATH=$(mktemp -d) OMP_NUM_THREADS=1
    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 .venv/bin/python $R/press_stem.py --check
    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 ~/.claude/bin/resguard.sh run --mem 5G --cpu 700 -- \
        .venv/bin/python $R/press_predict.py run $R/press_calibrated.jsonl [--station] [--mu 0.5] --workers 6
    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 ~/.claude/bin/resguard.sh run --mem 5G --cpu 700 -- \
        .venv/bin/python $R/press_predict.py replay $R/press_replay.jsonl --table-mu -1 0.35 [--mu 0.5] --workers 6
    HOS_PLANT=servo HOM_ROWS_TAG=_sv25 ~/.claude/bin/resguard.sh run --mem 5G --cpu 700 -- \
        .venv/bin/python $R/press_predict.py pipeline $R/press_pipeline.jsonl --workers 6

The scripts import rev_clutch2.py and rev_rescore.py from the same folder, and through them rev_gear, rev_fric2,
rev_lat, rev_joint_sag, rev_stem and rev_closed (copies of all in `../../20261116-sweep/sim/scripts/`); from the
repository they import sr2_hom_replay, hom_bench_world, hom_bench and hom_trial_tracking.
