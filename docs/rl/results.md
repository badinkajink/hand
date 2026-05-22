# Results

This page is the cross-run scoreboard. Each row is one training tag with
deterministic eval numbers comparable to Phase 1 CEM under
`Phase1GraspEvaluator(speed_mode="accurate")` on the same morphology +
frozen scene.

## Status

> **No training runs completed yet.** The MVP plan, code scaffolding, and
> bring-up procedure are in place but a successful PPO run on
> `cube_mvp_v1` has not been executed. Once one is, fill the table below.

## Acceptance criteria (from the plan)

The MVP is considered successful when, evaluated over 64 deterministic
rollouts with cube xy jitter ±5 mm and yaw jitter ±10°:

- Median `lift_height ≥ 30 mm` (CEM hits 49.4 mm; bar = 60 % of CEM).
- Mean `min_finger_contact_persistence ≥ 0.8` over the hold phase.
- Drop rate < 10 %.
- `Phase1GraspEvaluator` score on the learned open-loop action sequence
  ≥ +20 (CEM hits +29.45). "Beats CEM" is stretch, not MVP — CEM is a
  brittle point estimate; matching it within 30 % under stochastic ICs
  counts as success.

## Scoreboard

| Tag | Date | Morph | Total steps | Median lift (mm) | Min persist | Drop rate | P1 score | Notes |
|---|---|---|---|---|---|---|---|---|
| _baseline_ CEM | run18_final | candidate_id=0 | n/a | **49.4** | **1.00** | n/a | **+29.45** | foundational CEM |
| `cube_mvp_v1` | _pending_ | candidate_id=0 | 200M | — | — | — | — | first PPO run |

## How to add a row

```bash
# 1. Run eval after training (entrypoint TBD: scripts/rl_eval_cube.py).
uv run python scripts/rl_eval_cube.py \
    --checkpoint results/rl/<tag>/checkpoints/model_latest.pt \
    --num-rollouts 64

# 2. The eval script writes a `metrics.json` next to the checkpoint.
#    Append a row to this table referencing it.
```

(The eval script is not yet implemented — see the open task list at the
end of the Phase 7 section of the plan.)

## Comparison plots

Once at least two runs exist, drop the lift/persistence/score curves into
`docs/rl/figs/` and reference them here. Plotly HTML exports embed
naturally in mkdocs-material via `<iframe>`.
