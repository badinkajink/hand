# manta/ — real-hand characterization and sim media for the manta-hand residual-RL project

Moved here from the local `manta-hand/` folder on 2026-11-02 (the paper draft stays local).

- `sysid/` — SCS0009 servo identification and hand checks, 2026-10-26 … 10-30.
  - `README.md` is the lab notebook (every test, in order); `TEST_PLAN.md` the plan it followed.
  - `bam_pendulum/` — single / double pendulum rig, raw logs (`data_raw/`), BAM (Better Actuator
    Models) fits (`fits_best/` = the ones used), `RESULTS.md`.
  - `hand_replay.py` — sim-to-real gate: replays the hand's full-range sweeps in MuJoCo with the
    fitted actuator (uses `../../assets/mjcf/real_v3`). `hand_friction_scales.json` = its per-joint
    friction calibration.
  - `zero_check.py`, `rs_grab.py`, `handoff_health_check_2026-10-30.txt` — joint-zero check with
    the RealSense on the CB1, and the hand's state at handoff.
- `sim_media/` — rendered sim trials (mp4 + still sheets) from `src/morphohand/cpu_grasp`.

Not copied (re-create them):
- `sysid/.venv-bam` — Python 3.12 venv with BAM installed from `sysid/bam_upstream`.
- `sysid/bam_upstream` — unmodified https://github.com/Rhoban/bam at commit `b9e1111`.
- `sysid/bam_pendulum/data_processed` — rebuild with `prepare_logs.py` from `data_raw/`.
- `sysid/bam_pendulum/fits20k/logs` — optimizer logs; the fitted models are kept.

Hand-side scripts run on the CB1 (irlab@10.99.99.2) against the `~/hand` driver copy (see
`sysid/README.md`, 2026-10-30: the `~/manta-hand/host` copy has a different middle-yaw zero).
