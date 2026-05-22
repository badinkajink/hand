# Run 18 — Multi-object morphology sweep on the short-proximal hand

## Goal

Search a 2000-sample morphology space (thumb / index / middle × x, y, length
offsets — 9 dims) on a single shared short-proximal hand (capsule
fingertips, 0.025m proximal phalanges) and score every candidate against a
7-object eval set:

| Object | Scene | Keyframe | Notes |
|---|---|---|---|
| cube                       | scene_cube_short_proximal.xml                       | open_short_manual         | hand-authored seed (run via Lightning, then manual touch-up) |
| prism                      | scene_prism_short_proximal.xml                      | open_short_manual         | same |
| power_drill                | scene_power_drill_short_proximal.xml                | open_flat_gripping        | hand-authored from run17 (capsule-tip variant) |
| screwdriver_medium_flat    | scene_screwdriver_medium_flat_short_proximal.xml    | open_short_manual         | manual seed |
| screwdriver_medium_vertical| scene_screwdriver_medium_vertical_short_proximal.xml| open_short_manual         | palm body relocated to mirror open_vertical so the hand can reach the shaft |
| screwdriver_medium_90vert  | scene_screwdriver_medium_short_proximal.xml         | open_90vertical_manual    | manual touch-up of existing open_90vertical |
| screwdriver_small_flat     | scene_screwdriver_small_flat_short_proximal.xml     | open_short_manual         | manual seed (was duplicate-middle_mount-bug in source scene, fixed) |

All `_short_proximal` scenes are derived from their original by:
- Editing each `*_len_frame` body `pos="0.05 0 0"` → `pos="0.025 0 0"` (short proximal phalanx)
- Running `scripts/generate_capsule_tip_scene.py` in-place to replace
  sphere fingertips with capsules (`fromto -0.006 .. 0.006`, radius 0.005).
- Morph joints retained so the sweep can perturb x, y, len offsets on top.

## History of attempts (2026-05-20 → 2026-05-21)

### Phase A — Seed authoring

1. **Lightning Grasp pipeline**
   ([scripts/build_short_proximal_keyframes.py](build_short_proximal_keyframes.py))
   per object: build URDF from the current scene morphology, swap into
   `external/lightning-grasp`, generate ~200 candidate grasps, score in
   mujoco `init_pose` mode, pick best, write as `open_short` keyframe.
   Issues encountered:
   - **Primitive mesh OBJs had inverted face winding** → Lightning's
     `get_support_point_mask` returned 0 points → pipeline crashed. Fixed
     by switching to `trimesh.creation.box / cylinder`
     ([scripts/generate_primitive_meshes.py](generate_primitive_meshes.py)).
   - **Capsule tips** (5mm radius) are much smaller in y/z than the
     previous sphere group (effective 10mm); URDF builder updated to
     emit cylinder + 2 spheres approximating the capsule
     ([scripts/build_morphohand_urdf.py](build_morphohand_urdf.py) —
     `_tip_collision`). With capsule URDF Lightning produced fewer viable
     IK solutions, and the resulting keyframes failed the smoke test.

2. **Smoke test**
   ([scripts/smoke_eval_set_short_proximal.py](smoke_eval_set_short_proximal.py))
   verifies that holding the keyframe ctrl fixed for 200 steps and lifting
   5cm produces ≥2 fingers in real contact and ≥2.5cm lift. Lightning
   keyframes failed for most objects → user hand-authored `_manual`
   keyframes as positioning seeds (closed-hand poses near each object).

3. **screwdriver_medium_vertical fix:** the source scene's `palm_pose` body
   was at z=0.18, putting the palm 13cm above the screwdriver shaft. The
   `_manual` keyframe palm joints were edited to mirror the working
   `open_vertical` (palm at world z ≈ 0.23) so closure becomes geometrically possible.

### Phase B — Sweep architecture

4. **Run-18 driver**
   ([scripts/run18_multi_object_sweep.py](run18_multi_object_sweep.py))
   carries 7 tasks (one keyframe per scene) and one shared morphology
   population. For each (morphology, task): build rigid scene with morph
   baked, construct `Phase1GraspEvaluator`, score with each task's
   foundational ctrl. Foundational ctrl is the result of one `phase1_optimize_grasp.py`
   pass per task (100 iter × 52 pop) at the start.

5. **Drill pivot slowed** to `pivot_steps=240, pivot_ramp_steps=200`
   (from 180/120 in run17) to reduce impulse magnitude during the
   pivot-to-down task.

### Phase C — Ctrl adaptation

6. **Baseline run** (no per-morph adaptation): 2000 samples, ~270s sweep.
   Per-task tops looked weak — the foundational ctrl was optimised for the
   base morphology only, so most of the swept morphologies had a ctrl
   that didn't match their hand shape.

7. **Adapt run** (`--adapt-mode interval-initial-fp`,
   refresh every 40 candidates with 16×36 CEM): 42 min wall time, 50
   refreshes per task. Median `score_mean` improved 3× (+2.8 → +8.2),
   per-task bests improved across the board.

8. **Physics blowups**: 184/2000 candidates (most on small_flat) had
   non-physical object trajectories (z>1m, velocity>100 m/s) when the
   adapted ctrl drove a thin 8mm cylinder through contact instabilities.
   Filtered out by [scripts/run18_filter_blowups.py](run18_filter_blowups.py).
   Filtered results → `analysis_filtered/`.

### Phase D — Contact-map "whiff" debugging (this run18_final)

9. **Diagnosis**: top per-task scores on `screwdriver_small_flat` looked
   high (+20 to +28) but the per-grasp diagnostics showed many with
   `min_finger_contact_persistence = 0.0`. The reason: the
   contact-target patches use `radius: 0.008` (small_flat) and
   `_patch_reward(d, r)` decays smoothly out to ~3r ≈ 24mm. Fingers 22-36mm
   from a patch — i.e. *not on the screwdriver at all* — still earned
   near-max `contact_target_reward` (weight 10). The screwdriver got
   bumped up by physics, not gripped.
10. **Fix**: lower `contact_target_reward` weight 10 → **3.0** and raise
    `min_finger_persistence` weight 6 → **14.0**, so real fingertip
    contact dominates the score.
11. **Sparse adapt added alongside interval adapt** (run6 pattern):
    every candidate gets a small `sparse-per-morph` CEM (2 iter × 10 pop,
    σ=0.06) on top of the interval refresh (16 iter × 36 pop, every 40
    candidates).

## Final config (run18_final)

```bash
./scripts/run18_launch.sh
```

equivalent to:

```bash
python scripts/run18_multi_object_sweep.py \
  --tag run18_final --samples 2000 \
  --foundational-iterations 100 --foundational-population 52 \
  --adapt-mode interval-initial-fp \
  --fp-refresh-interval 40 \
  --interval-adapt-iterations 16 --interval-adapt-population 36 \
  --sparse-adapt-mode sparse-per-morph \
  --sparse-adapt-iterations 2 --sparse-adapt-population 10 \
  --contact-target-reward 3.0 \
  --contact-target-distance-penalty 20.0 \
  --min-finger-persistence 14.0
```

The launcher script chains:
1. **Sweep**          — `run18_multi_object_sweep.py`
2. **Blowup filter**  — `run18_filter_blowups.py`
3. **Analysis**       — `run18_analysis.py` (TSNE / hexbin / landscape-diff / Spearman / labeled TSNE / top-K)
4. **Top-K videos**   — `run18_render_top.py --adapt-before-render` (5 per task + 5 cross-set × 7 tasks)
5. **Text summary**   — `run18_text_summary.py` → `SUMMARY.txt`

Override any default with an env var, e.g.:
```bash
SAMPLES=500 TAG=run18_v2 CONTACT_TARGET_REWARD=5.0 ./scripts/run18_launch.sh
```

## Analysis outputs

Under `<run_dir>/analysis_filtered/`:

| File | Contents |
|---|---|
| `summary.json`                                  | Numeric summary + per-task best morphology |
| `labeled_tsne.png`                              | Background = all morphs by score_mean; overlays = top-hand per task + cross-set top + subset tops (cube+prism, screwdrivers, screwdrivers+drill). Overlay markers are 35× larger than background dots and individually labeled. |
| `per_task/<task>_tsne.png`                      | Per-task TSNE coloured by that task's score |
| `per_task/<task>_hex_<dx>_<dy>.png`             | MAP-Elites hexbin (max-score) over 4 dim pairs |
| `landscape/<task>_diff_<dx>_<dy>.png`           | `score_task - score_mean` over the same dim pairs (highlights specialisation regions) |
| `similarity/spearman_matrix.png` + `.csv`       | 7×7 Spearman rank correlation across tasks |
| `top_k/<task>.csv`, `top_k/_cross_set.csv`      | Top morphology IDs + dim values |
| `<run_dir>/videos_filtered/per_task/<task>/`    | top-K mp4s for that task |
| `<run_dir>/videos_filtered/cross_set/top_NN_on_<task>/` | top-K cross-set morphs rendered against each task |
| `<run_dir>/SUMMARY.txt`                         | Plain-text headline |

## Open issues (deferred for future work)

- **Capsule tip on thin cylinders** (small_flat) is fundamentally
  harder than spheres — fewer Lightning grasps, more physics blowups.
  Future work: investigate larger capsule tip radius (0.0075m to match
  the pip phalanx), or per-object friction tuning.
- **Per-candidate ctrl was not saved** in the sweep CSV; the top-K render
  re-runs a mini-CEM with `--adapt-before-render`. If reproducibility of
  the exact sweep ctrl is needed later, persist it per candidate.
- **TSNE perplexity** is fixed at 30.0 — for the 7-object eval set with
  ~2000 candidates this gives reasonable clustering, but for larger sweeps
  this should scale.
