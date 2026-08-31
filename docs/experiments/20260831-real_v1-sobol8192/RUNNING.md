# Sobol-8192 — IN FLIGHT, started 2026-08-30 late evening

`scripts/real_v1_sobol8192.sh`, launched detached; log at `logs/20260831-sobol8192.log`.

`sobol_designs` is prefix-stable, so this is a strict superset of the 4,096-hand population: the
first half keeps its tags, its vectors and its generated scenes, and only the new half costs
anything. **3,311 genuinely new hands are inside the measured rails** (6,629 of 8,198 reachable,
against 3,318 of 4,102 before).

| stage | what it does | state |
|---|---|---|
| A | sample 8,192 + 6 anchors, write the provenance manifest | done |
| A2 | keep only the hands the rails reach → `hardware_manifest.json`, `reachable.txt` | done, 6,629 |
| B | grasp screen, 16 shards | running (~1 h) |
| C | one table per grasp cell (`retention/`) | pending |
| D | the retention screen **swept across five clips**, not at one | pending (~1 h) |
| E | one (cell, clip) operating point per hand → `selected/` | pending |

Stage D is the difference from the 4,096 run: that population was screened at a single
`budget = 0.5` rad, and
[the re-screen](../20260830-real_v1-budget-rescreen/README.md) showed 39 of 269 passing
morphologies are invisible at that value. This one sweeps 0.50–1.30 from the start, and carries
the servo-command gate inside the screen.

**If it is still running, leave it** — every stage checkpoints into its `--out` and skips what is
already there, so it is safe to kill and re-run. **If it finished**, `selected/summary.json` has
the funnel and `scripts/real_v1_budget_confirm.sh` (pointed at `OUT=` this directory) is the
confirmation pass. Newly generated scene XMLs land in
`assets/mjcf/experimental/20260830-real_v1-sobol4096/` alongside the existing ones and are not yet
committed.

The `grasp_shard_*.json` files committed alongside this note are a **partial snapshot** taken
while stage B was running — they were swept into a commit by a `git add -A docs/`. They will be
complete when the run is; re-commit them then.

**Do not read stage-D single-trial cosines as results.** One rollout per cell per clip, twenty
chances per hand — the confirmation pass is the measurement.
