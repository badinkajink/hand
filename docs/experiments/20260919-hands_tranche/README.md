# Hands tranche 2026-09-19/20 — runbook

Serial, gate-aware queue of reorientation-policy finetunes across the deployed hands on the calibrated
plant. Results prose lives in the page (`20260919-reorient_policies_across_hands.html`, built by
`scripts/hands_tranche_page.py`), never here.

## Run / resume

    nohup setsid python3 scripts/hands_tranche_queue.py --queue docs/experiments/20260919-hands_tranche/queue.json \
        >> logs/20260919-hands_tranche.log 2>&1 &

The driver re-reads `queue.json` before each job (edit flags/status of *pending* jobs while it runs),
waits for `~/.claude/bin/resguard.sh status` to open the gate, trains through `resguard.sh run`, then
evaluates, renders, filmstrips, runs the CPU scene-variant probe and the chain probe, and appends an
fsynced row to `tranche_results.json`. Jobs left "running" by a dead driver go back to pending at
start. After the last job it runs the GPU transfer probes (`<id>_transfer.json`) and exits.

## Files

| file | what |
|---|---|
| `queue.json` | jobs, common flags, notes (design decisions with times) |
| `tranche_results.json` | one row per finished/failed job: eval summary + plausibility numbers |
| `<id>_eval.json`, `<id>_eval.png` | `policy_eval_suite.py`, 64 rollouts, run timing + action clip, plausibility block |
| `<id>_strip.png`, `<id>_trace.csv` | filmstrip and per-step trace of one 960x720 deterministic rollout (video in `videos/`, gitignored) |
| `<id>_transfer_cpu.json` | `policy_transfer_probe.py --cpu`: base, tipmesh, plate0, mu0.6, mu1.5, mass1.3, kp0.25 × 6 jittered CPU rollouts |
| `<id>_chain_pl25.json`, `<id>_chain_pl0.json` | `real_v1_chain_policy.py`: the policy driving the fingers inside the UR5e chain for 5 s, plate at 25 / 0 mm |
| `chain/`, `cpu_check/` | the D6 reference checkpoints under the same probes (2026-09-19 evening) |
| `20260919-tranche_curves.png` | tensorboard curves of every job with a run directory |

Per-job logs: `logs/20260919-hands_tranche-<id>.log` (train), `-<id>_eval.log`, `-<id>_render.log`.

## Decision log

- 19:56 queue armed; gate closed (swap 5.3 GB > 4 GB, idle VS Code pages; the user's own job running).
- 20:15 D6 grip finetune probed in the chain: turns to 0.91, drops at 3.2 s; the 17 N 60M checkpoint holds 3/4.
- 20:30 warm start changed to the 60M (+0.25) checkpoint, grip penalty -2 above 6 N.
- 21:38 all arms blind (user: the 66-dim obs will not transfer); swap gate raised to 8 GB by the user; first job started.
- 21:45 user: the tag pose can be used; blind warm start had not recovered in 22 iterations; both blind starts aborted, all arms sighted; first sighted job 21:52.
- Wake-ups: 00:53, 05:53, 10:53 on 2026-09-20 (session crons 6ccb78cf, b0efd918, 7249e298).
