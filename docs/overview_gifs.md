# MorphoHand — Visual Tour of Methods Tried

Last updated: 2026-05-13

This is a companion to [overview.md](overview.md) — same arc, but told through rollout GIFs from `results/`. Each section names the method, the question it was asked to answer, and the artifact directory it came from. Open the linked run dir for the full `summary.json` / `report.md` / CSVs.

---

## 1. Phase-1 sanity: foundational pose on a single scene

The earliest CEM runs. The question was *can the inner loop close a fist on a cube at all?* — short rollouts on the canonical scene, no morphology sweep.

Run dir: [`results/phase1/run_20260410_163959/`](../results/phase1/run_20260410_163959)

<table>
<tr>
<td align="center"><b>sanity</b><br><video src="videos/phase1/run_20260410_163959/sanity_phase1/best_rollout.mp4" width="320" autoplay loop muted playsinline></video></td>
<td align="center"><b>generated scene</b><br><video src="videos/phase1/run_20260410_163959/phase1_generated_scene/best_rollout.mp4" width="320" autoplay loop muted playsinline></video></td>
</tr>
</table>

---

## 2. MJX-autodiff lane vs CEM (prism y-sweep)

A gradient-based alternative to CEM, evaluated by sweeping the prism's y-offset to probe robustness to scene perturbation. The autodiff lane closed the grasp at all three offsets but lost to CEM on the standard scoring — recorded as the reason CEM stayed default for Phase 1.

Run dir: [`results/phase1/run_20260413_prism_y_sweep_mjx_autodiff/`](../results/phase1/run_20260413_prism_y_sweep_mjx_autodiff) · also [`run_20260413_phaseA_fp_conditioned/`](../results/phase1/run_20260413_phaseA_fp_conditioned)

<table>
<tr>
<td align="center"><b>y = 0.025</b><br><video src="videos/phase1/run_20260413_prism_y_sweep_mjx_autodiff/y0d0250/best_rollout.mp4" width="240" autoplay loop muted playsinline></video></td>
<td align="center"><b>y = 0.030</b><br><video src="videos/phase1/run_20260413_prism_y_sweep_mjx_autodiff/y0d0300/best_rollout.mp4" width="240" autoplay loop muted playsinline></video></td>
<td align="center"><b>y = 0.035</b><br><video src="videos/phase1/run_20260413_prism_y_sweep_mjx_autodiff/y0d0350/best_rollout.mp4" width="240" autoplay loop muted playsinline></video></td>
</tr>
<tr>
<td align="center"><b>FP-conditioned, foundational</b><br><video src="videos/phase1/run_20260413_phaseA_fp_conditioned/foundational_diffmjx_mvp/best_rollout.mp4" width="240" autoplay loop muted playsinline></video></td>
<td align="center"><b>FP-conditioned, open</b><br><video src="videos/phase1/run_20260413_phaseA_fp_conditioned/open_diffmjx_mvp/best_rollout.mp4" width="240" autoplay loop muted playsinline></video></td>
<td></td>
</tr>
</table>

---

## 3. Pollard-style multiscene morphology sweep (cube + prism)

500-sample sweep across cube and three prism variants, using CEM foundational poses as the inner loop. This is where morphology started doing work — the same controls behaved very differently across geometries, and feasibility gates began earning their keep.

Run dir: [`results/phase1/run_20260413_pollard_multiscene_smoke2/`](../results/phase1/run_20260413_pollard_multiscene_smoke2) · production: [`run_20260413_pollard_multiscene_500_run3/`](../results/phase1/run_20260413_pollard_multiscene_500_run3)

<table>
<tr>
<td align="center"><b>cube</b><br><video src="videos/phase1/run_20260413_pollard_multiscene_smoke2/cube/top_gifs/rank01_candidate0000.mp4" width="240" autoplay loop muted playsinline></video></td>
<td align="center"><b>prism1</b><br><video src="videos/phase1/run_20260413_pollard_multiscene_smoke2/prism1/top_gifs/rank01_candidate0002.mp4" width="240" autoplay loop muted playsinline></video></td>
<td align="center"><b>prism2</b><br><video src="videos/phase1/run_20260413_pollard_multiscene_smoke2/prism2/top_gifs/rank01_candidate0001.mp4" width="240" autoplay loop muted playsinline></video></td>
</tr>
<tr>
<td align="center"><b>prism3</b><br><video src="videos/phase1/run_20260413_pollard_multiscene_smoke2/prism3/top_gifs/rank01_candidate0006.mp4" width="240" autoplay loop muted playsinline></video></td>
<td align="center"><b>500-sample prism1 rank01</b><br><video src="videos/phase1/run_20260413_pollard_multiscene_500_run3/prism1/top_gifs/rank01_candidate0157.mp4" width="240" autoplay loop muted playsinline></video></td>
<td></td>
</tr>
</table>

---

## 4. Foundational-pose adaptation strategies (Run 5 family)

Same sweep, six different ways of feeding the foundational pose into per-morphology evaluation:

| Tag | Strategy |
|---|---|
| `run5a` | baseline — single FP, no adaptation |
| `run5b` | interval FP, every 50 samples |
| `run5c` | interval FP, every 25 samples |
| `run5d` | interval open-pose, every 50 samples |
| `run5e` | sparse — 5 FPs scattered across the sweep |
| `run5f` | local perturbation around current best |

`5e` (sparse-per-morph) and `5b/5c` (interval-initial-FP) ended up improving feasibility the most. Cube rank-1 from each is shown below — the differences are clearer in feasibility counts than in single-rollout video, but the visual variety is the point.

<table>
<tr>
<td align="center"><b>5a baseline</b><br><video src="videos/phase1/run5a_baseline/cube/top_gifs/rank01_candidate0173.mp4" width="200" autoplay loop muted playsinline></video></td>
<td align="center"><b>5b interval-FP 50</b><br><video src="videos/phase1/run5b_interval_fp_i50/cube/top_gifs/rank01_candidate0424.mp4" width="200" autoplay loop muted playsinline></video></td>
<td align="center"><b>5c interval-FP 25</b><br><video src="videos/phase1/run5c_interval_fp_i25/cube/top_gifs/rank01_candidate0424.mp4" width="200" autoplay loop muted playsinline></video></td>
</tr>
<tr>
<td align="center"><b>5d interval-open 50</b><br><video src="videos/phase1/run5d_interval_open_i50/cube/top_gifs/rank01_candidate0424.mp4" width="200" autoplay loop muted playsinline></video></td>
<td align="center"><b>5e sparse-5</b><br><video src="videos/phase1/run5e_sparse_5/cube/top_gifs/rank01_candidate0173.mp4" width="200" autoplay loop muted playsinline></video></td>
<td align="center"><b>5f local-perturb</b><br><video src="videos/phase1/run5f_local_perturb/cube/top_gifs/rank01_candidate0173.mp4" width="200" autoplay loop muted playsinline></video></td>
</tr>
</table>

---

## 5. Run 6 — multi-keyframe screwdriver (combined adaptation)

The first task that asked the same morphology to satisfy *three* keyframes at once: `open_flat`, `open_vertical`, `open_90vertical`. Headline finding was strong pose sensitivity — `open_flat` was easy, `open_90vertical` was hardest, and the same controls almost never worked across all three without retuning.

Run dir: [`results/phase1/run6_combined_1000/`](../results/phase1/run6_combined_1000)

<table>
<tr>
<td align="center"><b>open_flat (rank01)</b><br><video src="videos/phase1/run6_combined_1000/top5_gifs/rank01_open_flat.mp4" width="240" autoplay loop muted playsinline></video></td>
<td align="center"><b>open_vertical (rank01)</b><br><video src="videos/phase1/run6_combined_1000/top5_gifs/rank05_open_vertical.mp4" width="240" autoplay loop muted playsinline></video></td>
<td align="center"><b>open_90vertical (rank01)</b><br><video src="videos/phase1/run6_combined_1000/top5_gifs/rank04_open_90vertical.mp4" width="240" autoplay loop muted playsinline></video></td>
</tr>
</table>

---

## 6. Run 7 — sphere vs capsule fingertips (strict vertical)

A controlled A/B with identical keyframes and strict vertical constraints, varying only fingertip geometry. Spheres won on consistency (15.9% all-task feasible vs 3.6% for capsules), capsules produced higher outlier peaks. See [`run7_analysis_summary.md`](../results/phase1/run7_analysis_summary.md) for the full table.

Run dirs: [`run7_strict_spheres/`](../results/phase1/run7_strict_spheres) · [`run7_strict_capsules/`](../results/phase1/run7_strict_capsules)

<table>
<tr>
<th></th><th>open_flat</th><th>open_vertical</th><th>open_90vertical</th>
</tr>
<tr>
<th>spheres (rank01)</th>
<td><video src="videos/phase1/run7_strict_spheres/top5_gifs/rank01_open_flat.mp4" width="220" autoplay loop muted playsinline></video></td>
<td><video src="videos/phase1/run7_strict_spheres/top5_gifs/rank01_open_vertical.mp4" width="220" autoplay loop muted playsinline></video></td>
<td><video src="videos/phase1/run7_strict_spheres/top5_gifs/rank01_open_90vertical.mp4" width="220" autoplay loop muted playsinline></video></td>
</tr>
<tr>
<th>capsules (rank01)</th>
<td><video src="videos/phase1/run7_strict_capsules/top5_gifs/rank01_open_flat.mp4" width="220" autoplay loop muted playsinline></video></td>
<td><video src="videos/phase1/run7_strict_capsules/top5_gifs/rank01_open_vertical.mp4" width="220" autoplay loop muted playsinline></video></td>
<td><video src="videos/phase1/run7_strict_capsules/top5_gifs/rank01_open_90vertical.mp4" width="220" autoplay loop muted playsinline></video></td>
</tr>
</table>

---

## 7. Run 8 — power drill, forward → down reorientation

First real reorientation target: pivot the drill from forward to down while keeping fingers in contact. Several restarts (`_r2…_r4`, `_shortlen_*`) because generated-scene mesh paths needed `rebase_asset_file_paths()` — see the 2026-04-24 entry in [overview.md](overview.md).

Run dirs: [`run8_power_drill_forward_to_down_r4/`](../results/phase1/run8_power_drill_forward_to_down_r4) · [`_shortlen_r4/`](../results/phase1/run8_power_drill_forward_to_down_shortlen_r4)

<table>
<tr>
<td align="center"><b>r4 rank01</b><br><video src="videos/phase1/run8_power_drill_forward_to_down_r4/top5_gifs/rank01_open_flat.mp4" width="240" autoplay loop muted playsinline></video></td>
<td align="center"><b>r4 rank02</b><br><video src="videos/phase1/run8_power_drill_forward_to_down_r4/top5_gifs/rank02_open_flat.mp4" width="240" autoplay loop muted playsinline></video></td>
<td align="center"><b>shortlen_r4 rank01</b><br><video src="videos/phase1/run8_power_drill_forward_to_down_shortlen_r4/top5_gifs/rank01_open_flat.mp4" width="240" autoplay loop muted playsinline></video></td>
</tr>
</table>

---

## 8. Runs 8/9 — short-proximal drill baseline (CEM)

The same `power_drill_short_proximal` scene that later became a canonical benchmark target. Runs 8 and 9 are the CEM-only baselines on this scene before objective changes.

<table>
<tr>
<td align="center"><b>run8 CEM</b><br><video src="videos/phase1/run8_power_drill_short_proximal_open_flat_cem/best_rollout.mp4" width="320" autoplay loop muted playsinline></video></td>
<td align="center"><b>run9 CEM rerun</b><br><video src="videos/phase1/run9_power_drill_short_proximal_open_flat_cem_rerun/best_rollout.mp4" width="320" autoplay loop muted playsinline></video></td>
</tr>
</table>

---

## 9. Run 10 — contact-focused objective

Same scene as Run 9, but the objective was reweighted to encourage **proximal** contacts (not just any fingertip touch). Visually, fingers wrap further down the drill body.

Run dir: [`run10_power_drill_short_proximal_contact_focus/`](../results/phase1/run10_power_drill_short_proximal_contact_focus)

<p align="center"><video src="videos/phase1/run10_power_drill_short_proximal_contact_focus/best_rollout.mp4" width="420" autoplay loop muted playsinline></video></p>

---

## 10. Run 11 — trajectory optimization

Constant per-finger controls were replaced with piecewise-linear **trajectories** (Δt knots), optimized via CEM. This unlocks closing motions where the finger profile changes mid-rollout instead of a single set-point.

Run dir: [`run11_power_drill_short_proximal_traj4/`](../results/phase1/run11_power_drill_short_proximal_traj4)

<p align="center"><video src="videos/phase1/run11_power_drill_short_proximal_traj4/best_rollout.mp4" width="420" autoplay loop muted playsinline></video></p>

---

## 11. Run 12 — contact-focused trajectory (combined)

Run 10's contact-focused objective + Run 11's trajectory parameterization. This is the most recent single-scene result before the eval suite was stood up.

Run dir: [`run12_power_drill_short_proximal_contact_focused_traj/`](../results/phase1/run12_power_drill_short_proximal_contact_focused_traj)

<table>
<tr>
<td align="center"><b>current</b><br><video src="videos/phase1/run12_power_drill_short_proximal_contact_focused_traj/best_rollout.mp4" width="320" autoplay loop muted playsinline></video></td>
<td align="center"><b>previous take</b><br><video src="videos/phase1/run12_power_drill_short_proximal_contact_focused_traj.prev/best_rollout.mp4" width="320" autoplay loop muted playsinline></video></td>
</tr>
</table>

---

## 12. Eval suite — `baseline` vs `contact_map` across all objects

The cross-object benchmark harness re-evaluates each method under the **baseline** objective for apples-to-apples scoring. Two variants:

- **`full_baseline_vs_contact_map`** — live scenes
- **`full_frozen_baseline_vs_contact_map`** — frozen-scene protocol (joints baked; required for fair eval, see [feedback_frozen_scene_protocol.md](../../.claude/projects/-home-humanoid-Programs-hand/memory/feedback_frozen_scene_protocol.md))

Leaderboards: [live](../results/eval_suite/full_baseline_vs_contact_map/leaderboard.md) · [frozen](../results/eval_suite/full_frozen_baseline_vs_contact_map/leaderboard.md)

Headline: under the baseline objective, `contact_map` wins **4/8** benchmarks (mean Δ +0.32). Biggest win is `prism` (+3.26); biggest regression is `power_drill` (-1.17).

### Frozen-scene cross-comparison (baseline vs contact_map per object)

<table>
<tr><th>Object</th><th>baseline</th><th>contact_map</th></tr>
<tr>
<th>cube</th>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/cube__baseline.mp4" width="280" autoplay loop muted playsinline></video></td>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/cube__contact_map.mp4" width="280" autoplay loop muted playsinline></video></td>
</tr>
<tr>
<th>prism</th>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/prism__baseline.mp4" width="280" autoplay loop muted playsinline></video></td>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/prism__contact_map.mp4" width="280" autoplay loop muted playsinline></video></td>
</tr>
<tr>
<th>power_drill</th>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/power_drill__baseline.mp4" width="280" autoplay loop muted playsinline></video></td>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/power_drill__contact_map.mp4" width="280" autoplay loop muted playsinline></video></td>
</tr>
<tr>
<th>power_drill_short_proximal</th>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/power_drill_short_proximal__baseline.mp4" width="280" autoplay loop muted playsinline></video></td>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/power_drill_short_proximal__contact_map.mp4" width="280" autoplay loop muted playsinline></video></td>
</tr>
<tr>
<th>screwdriver_medium_flat</th>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/screwdriver_medium_flat__baseline.mp4" width="280" autoplay loop muted playsinline></video></td>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/screwdriver_medium_flat__contact_map.mp4" width="280" autoplay loop muted playsinline></video></td>
</tr>
<tr>
<th>screwdriver_medium_vertical</th>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/screwdriver_medium_vertical__baseline.mp4" width="280" autoplay loop muted playsinline></video></td>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/screwdriver_medium_vertical__contact_map.mp4" width="280" autoplay loop muted playsinline></video></td>
</tr>
<tr>
<th>screwdriver_medium_90vertical</th>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/screwdriver_medium_90vertical__baseline.mp4" width="280" autoplay loop muted playsinline></video></td>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/screwdriver_medium_90vertical__contact_map.mp4" width="280" autoplay loop muted playsinline></video></td>
</tr>
<tr>
<th>screwdriver_small_flat</th>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/screwdriver_small_flat__baseline.mp4" width="280" autoplay loop muted playsinline></video></td>
<td><video src="videos/eval_suite/full_frozen_baseline_vs_contact_map/gifs/screwdriver_small_flat__contact_map.mp4" width="280" autoplay loop muted playsinline></video></td>
</tr>
</table>

> The `small_flat` screwdriver currently produces negative scores under either method — that benchmark is still open. Everything else clears the lift gate.

---

## Where to look next

- All GIFs above live next to their `summary.json` / `report.md` / `top5_candidates.csv` — open the run dir to inspect controls, morphology, and metrics.
- For the textual / code-map version of the same story, see [overview.md](overview.md).
- For the strict numerical Run 7 breakdown, see [`run7_analysis_summary.md`](../results/phase1/run7_analysis_summary.md).
- For the eval-suite scoring rules and method definitions, see the suite's [leaderboard](../results/eval_suite/full_frozen_baseline_vs_contact_map/leaderboard.md) and [per_benchmark](../results/eval_suite/full_frozen_baseline_vs_contact_map/per_benchmark.md).
