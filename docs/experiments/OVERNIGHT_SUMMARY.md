# Overnight Run Summary — Policy B v2 (in-hand reorientation)

_Run date: 2026-06-01 → 2026-06-02. Autonomous staged sweep + assessment._

## What ran

Off **Policy B v1** (`results/rl/20260601-1033-policyB_v1`), a two-stage finetune sweep
toward **smooth, quick, holds-vertical** in-hand reorientation:

- **Stage 1** — smoothness ramp + signed `target_axis_progress` only (no quick mechanisms).
  Swept final smoothness target 5× vs 10×. Warmstart from v1 `model_2033.pt`.
  → **5× won**; 10× over-penalized and stopped rotating.
- **Stage 2** — warmstart the Stage-1 5× ckpt, add the three "quick" mechanisms
  (alignment-success termination + bonus, per-step time cost, early-crossing speed bonus),
  with the smoothness base **anchored at the warmstart level** to protect grip. Swept 5× vs 10×.

Each run: 30M ts, 1024 envs, 1220 iters. Run dirs `results/rl/20260602-0024-policyB_v2_smooth{5x,10x}_quick`.
Logs `policyB_v2_{5x,10x}.log` (reused for Stage 2). Stage-1 metrics preserved in `STAGE1_RESULTS.txt`,
Stage-2 in `STAGE2_RESULTS.txt`.

## Headline metrics (final converged block)

| Metric | v1 | v2 Stage-1 best (5×) | v2 Stage-2 best (10×-quick) |
|---|---|---|---|
| Mean reward | 402.7 | 369.2 | 360.6 |
| Mean episode length (/200) | ~188 | ~173 | 188 |
| `target_axis_alignment` | 87.2 | 80.0 | 75.2 |
| `target_axis_progress` | +0.253 | **+0.499** | +0.331 |
| `tip_lost`/iter (grip loss) | 1.17 | 0.96–1.6 | **0.63** |
| raw `action_rate` jitter | 1.3 | 1.23 | **1.15** |
| raw `object_ang_acc` jitter | 50 | 24.5 | **9.0** |
| min object-center z (eval) | — | 0.111 m | **0.120 m** |

(The third "quick" variant, **s2-5×-quick**, is omitted from the headline because it
regressed — see below.)

## Honest assessment — did "quick" work? Mostly no.

- **s2-5×-quick** got ~10% shorter episodes (173→155) via alignment-success terminations,
  but at a real cost: jerkier hand (raw action_rate 1.23→1.56), worse grip (tip_lost
  1.6→2.58), and `target_axis_progress` **collapsed to ≈ 0**. That signature is
  **threshold-gaming** — snap just past the 0.9 alignment line to grab the bonus and end
  the episode, not robust reorientation. **Not shippable.**
- **s2-10×-quick** — the quick terms are near-inert here (bonuses ~0.004; episodes run to
  timeout). Its quality comes instead from warmstart + gentle 10× ramp **rescuing** the
  10× regime that collapsed in Stage 1. It is the **smoothest, grippiest, highest-holding**
  policy in the sweep while still rotating well. Just **not actually faster** than v1.
- **"Quick" is still an open problem:** terminating on success conflates "finished fast"
  with "failed fast," and the success bonus invites gaming. Decouple speed shaping from
  termination next time.

## Recommended final Policy B v2

**`results/rl/20260602-0024-policyB_v2_smooth10x_quick/tensorboard/model_1219.pt`**
— best smooth + grip + holds-vertical policy overall (raw jitter 1.15 / 9.0, tip_lost 0.63,
min-z 0.120 m, alignment 75, progress +0.33). Genuine in-hand reorientation, no floor-bracing.

Alternative if sustained rotation quality > grip: **Stage-1 5×**
(`results/rl/20260601-2310-policyB_v2_smooth5x/tensorboard/model_1219.pt`) — alignment 80,
progress +0.50.

Verification videos: `docs/rl/videos/reorient/policyB_v2_smooth{10x,5x}_quick.mp4`.
Comparison figure regenerated: `docs/rl/img/reorient_comparison.png`.

## What I changed and why

- **`scripts/rl_plot_reorient.py`** — added the four v2 runs (Stage-1 5×/10×, Stage-2
  5×/10×-quick) and an `alignment_success` termination panel so the comparison figure
  covers the v2 sweep.
- **`docs/rl/reorientation.md`** — appended "Phase Policy B v2 — Stage 2" with the v1 /
  Stage-1 / Stage-2 metrics table, video links, the threshold-gaming finding, and the
  recommended config.
- Rendered the two Stage-2 verification videos via `scripts/rl_render_reorient.py`
  (written in the Stage-1 pass — drives the env from each run's own `config.yaml`).
- _No training code behavior changed in this pass_ — Stage 2's training edits (softened
  quick rewards + anchored smoothness base) were already committed pre-launch (`be1b50f`).

## Concrete next steps for you

1. **Ship / sim2real eval s2-10×-quick** as Policy B v2 — it's the grip-safest, smoothest
   policy. Re-run the handoff demo (Policy A lift → Policy B reorient) with this ckpt.
2. **Ablate the quick terms off at 10×** (one run, `QUICK=0`, 10× ramp, same warmstart) to
   confirm they're removable — expectation: ≈ identical, proving 10×-quick's win is the
   smoothness ramp, not the quick mechanisms.
3. **Redesign "quick" if still wanted:** a no-terminate alignment-hold bonus (reward
   staying aligned for N steps) instead of terminate-on-cross, to kill the threshold-gaming
   seen at 5×.
4. **Housekeeping:** the many `*.log` files at repo root are scratch — prune when ready.

---
_Generated autonomously overnight. See `overnight_autonomy.log`, `overnight_claude_{1,2}.log`._
