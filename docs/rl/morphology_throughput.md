# Morphology-optimization throughput / resource quantification (2026-06-23)

Measured on the live A+B parallel retrain (RTX-class 16 GB GPU), 2048 envs/run, 24 steps/env/iter
(= 49,152 timesteps/iteration).

## Per-run and parallel throughput (measured)

| | steps/s | s/iter | GPU mem | notes |
|---|---|---|---|---|
| 1 run solo (2048 env) | ~9,220 | 5.3 | ~2.2 GB | reorient (B) |
| **2 runs parallel** | **~8,900 each → ~17,850 aggregate** | ~5.5 | ~2.2 GB each (5.7 GB total) | ~3% per-run slowdown = **near-linear scaling** |
| GPU utilization (2 runs) | — | — | **18–31 %** | **massively under-utilized** |

**Wall time (this run):** B = 20 M ts / ~8.9k ≈ **37 min** (406 iters); A = 30 M ts / ~8.9k ≈ **56 min** (610 iters).
**CEM grasp screen:** ~28 s/design (60 iters) to ~100 s (100 iters).

## Headroom — the GPU is NOT the bottleneck
At 2 parallel runs we use **5.7 / 16 GB (35 %) and 18–31 % compute**. Mem allows ~**7** concurrent
2048-env runs; compute allows even more. The bottleneck is **wall-clock per RL run (~37–56 min)**,
not the GPU. Two ways to spend the headroom:
- **More parallel runs** — evaluate several designs' A+B concurrently (4–6 fit comfortably).
- **Bigger `--num-envs`** (2048 → 4096/8192) — raises steps/s per run (faster wall-time per design)
  since we're far from compute-bound.

## The design-evaluation funnel (and night throughput)

The structural test is cheap; only promising designs need the expensive RL confirm.

1. **Screen — CEM grasp balance** (`sweep_thumb_grasp.py`): ~28–100 s/design, reads per-finger
   contact-persistence + imbalance. Sequential ⇒ **~36–130 designs/hour screened** (parallelizable
   further). This is the throughput multiplier — kills bad designs before any RL.
2. **Confirm — A+B retrain** on a screened winner. A∥B ≈ **56 min/design** at 2-way parallel; with
   the 4–6× GPU headroom, run **2–3 designs' A+B concurrently ⇒ ~2–3 designs/hour ⇒ ~16–24
   designs/8 h night** fully retrained + per-finger-evaluated.

**Net:** screen 100s of designs/night on the cheap grasp metric; deep-RL-evaluate ~16–24/night.

## Cost-cutters (warmstart)
- A warmstarts a01, B warmstarts B4 ⇒ both converge from a good prior; A can likely drop 30 M→20 M
  (~37 min) and B 20 M→15 M (~28 min), shrinking the confirm stage ~30 %.
- Eval (`probe_grip_balance` / handoff demo) is ~1–2 min, negligible.

## Reproduce
`scripts/train_reorient_on_morph.sh` (B) + `scripts/train_A_on_morph.sh` (A), each with its own
`WARP_CACHE_PATH=$(mktemp -d)` (gotcha #2), staggered ~75 s. steps/s + `Collection time` are in the
rsl_rl iteration block of each `*.trainer.log`.
