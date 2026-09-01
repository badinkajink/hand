# Open-loop transfer, first pass — 8 morphologies, 77 bench trials, 2026-09-01

Reproduce with `uv run --extra rl python scripts/real_v1_transfer_figures.py`.
Figure: `transfer_firstpass.png`. Per-trial data: `data.json`. Stats: `stats.txt`.

77 trials survive the scoring rules (see the module docstring in the script): the last ten per
morphology on the 40 mm / 77 mm vane, one operator-flagged mis-stage removed. **52 held, 6
dropped, 19 unresolved.**

## 1. The simulator does not rank the hands

| | sim cos | bench cos_hold | sim rank | bench rank |
|---|---|---|---|---|
| `sv1_w6689` | 0.827 | **+0.826** (sd 0.032, n=6) | 1 | 2 |
| `sv1_w2360` | 0.726 | **+0.797** (sd 0.061, n=9) | 2 | 3 |
| `sv1_u1364` | 0.711 | **+0.489** (sd 0.221, n=8) | 3 | **7** |
| `g12` | 0.627 | — (10/10 unresolved) | 4 | — |
| `sv1_u0060` | 0.597 | **+0.912** (sd 0.027, n=8) | 5 | **1** |
| `sv1_u0308` | 0.585 | +0.625 (sd 0.052, n=5) | 6 | 5 |
| `rv05_manual` | 0.568 | +0.553 (sd 0.049, n=10) | 7 | 6 |
| `sv1_w0099` | 0.501 | +0.721 (sd 0.110, n=6) | 8 | 4 |

Spearman **+0.36 (p = 0.43)**, pearson +0.19. On seven designs that is no relationship at all.
The two errors that matter run in opposite directions: `sv1_u0060`, which the simulator puts
fifth at 0.597, is the best hand on the bench by a clear margin (+0.912, and the tightest
spread of any design); `sv1_u1364`, which the simulator puts third at 0.711, is the worst
(+0.489).

**This is not the bench failing to resolve.** Six of the seven measurable designs have a
within-design sd between 0.027 and 0.061, against a between-design spread of 0.489 to 0.912.
The bench separates these hands cleanly (F = 2.3 on pooled within vs between, and that pooled
figure is inflated almost entirely by `sv1_u1364`). It separates them into a *different order*
than the simulator does. That is a transfer result, not a noise result.

## 2. The drops reorient first, then lose the shaft — the case for a residual controller

All six drops follow one shape: the hand turns the shaft to a mean peak of **+0.608**, which is
**37° of the 90° turn already achieved**, and then collapses to +0.109. Nothing about the
approach or the initial grasp fails. What fails is holding what was already won, and it fails
late, after the informative part of the motion.

An open-loop plan cannot react to that by construction — it is replaying set points and has no
term that knows the shaft is escaping. This is the cleanest argument the program has produced
for a residual policy: the residual does not need to discover the turn, which the plan already
performs, it needs only to notice slip and close on it. The failure is concentrated in exactly
the regime where feedback is cheap and the open-loop plan is blind.

`sv1_u1364`'s sd of 0.221 — four times any other design's — is the same phenomenon short of a
drop. Its trials land at 0.21, 0.30, 0.32, 0.33, 0.61, 0.66, 0.70, 0.78: not a distribution
around a mean, but a mixture of turns that held their grip and turns that slipped partway.

## 3. Simulated grip stability vs staying held: directionally right, not yet significant

Simulated contact force at the deployed clip against bench drop rate: **spearman −0.45
(p = 0.26)**, n = 8. The sign is the hoped-for one and the threshold behaviour is suggestive —
**every drop occurred at a simulated force ≤ 0.43 N, and no design at ≥ 0.48 N dropped at all**
— but `rv05_manual` sits at 0.40 N and never dropped in ten trials, so force is not sufficient
on its own. Contact *count* carries nothing (spearman +0.13): it is 3.0 for five of the eight
designs, and the design with the most simulated contacts (`sv1_u0308`, 3.5) has the worst bench
drop rate. Simulated retention (`ok`) is 1.00 for seven of eight and cannot discriminate either.

The one clean hit: `sv1_w0099` is the only design the simulator does not fully retain (0.75),
and it is one of only two that dropped on the bench. n = 1 design, so it is an observation, not
evidence.

## 4. What has to be fixed before the next sweep

**Twenty-five percent of trials have no measured ending.** The vane leaves the camera's view
mid-turn — at the last detection the tag sits a mean 9.5 mm above the bench floor, and in 24 of
24 cases below 60 mm. It is going down into the floor and out of frame, exactly as the operator
observed.

Going back to the 30 mm tag will not fix this, and the data says so directly: on `sv1_w6689`
the 30 mm tag lost 5 of 12 trials (42%) and the 40 mm tag lost 1 of 7 (14%). The 40 mm reprint
solved the decode problem it was meant to solve. What it did not solve — and slightly worsened,
by moving the tag centre from 71 mm to 77 mm — is the *sweep*: the tag traces an arc of radius
equal to the axial offset, so a longer vane reaches the floor sooner on a bigger turn.

**The fix is a shorter vane, not a smaller tag.** Keep the 40 mm tag and bring the axial offset
in toward 50 mm. That cuts the swept arc by a third while leaving decode margin untouched. It
is also the only change that helps `g12`, whose ten trials are *all* unresolved and which is
therefore absent from every ranking above despite peaking at a mean cos of 0.79 before its tag
dies — on the current evidence `g12` could be anywhere from the best hand in the set to a hand
that drops every time.

Unresolved rate does not track turn size across designs (spearman +0.39, p = 0.34), so this is
about where each hand carries the shaft, not simply how far it turns it. `g12` at 49° loses
every trial; `sv1_u0060` at 63° loses two of ten.
