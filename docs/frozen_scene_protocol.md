# Frozen-scene protocol (mandatory for grasp eval)

## Why this matters

Every scene under `assets/mjcf/` that uses the morphology-actuated hand has
the per-finger morphology DOFs (`thumb_x`, `thumb_y`, `thumb_len`,
`index_*`, `middle_*`) as **joints**, not baked transforms. They are not
actuated, so they sit at their keyframe values at reset — but they have no
explicit equality constraint either, so once the rollout starts they
**drift** under contact and inertial forces. The "fixed" morphology
silently changes during the experiment.

This invalidates results in two ways:

1. **Score noise** — the morphology that produced a given best grasp is
   not the same as the morphology used to evaluate it again or run the
   next CEM iteration. Scores depend on whatever drift trajectory the
   solver happened to take.
2. **Wrong conclusions** — methods that interact with the morph DOFs
   differently (e.g. by inducing different contact patterns) get
   different drift, so their *measured* performance reflects drift
   artifacts rather than method behaviour.

The fix is to bake the morphology into body transforms and remove the
morph joints up-front, producing a **frozen scene XML** that the
evaluator runs against.

## The protocol

For any experiment:

1. Before constructing a `Phase1GraspEvaluator`, freeze the source scene
   into the run output directory.
2. Pass the *frozen* path to the evaluator, never the base scene.
3. Same rule for oracle re-evaluation, GIF rendering, and any post-hoc
   analysis: always re-resolve to the frozen XML for that (scene, keyframe).

The canonical helper:

```python
from morphohand.sampling.scene import freeze_scene_for_eval

frozen_xml = freeze_scene_for_eval(
    scene_xml="assets/mjcf/baseline/scenes/scene_power_drill_short_proximal.xml",
    keyframe="open_flat",
    frozen_scene_xml=run_dir / "frozen_scenes" / "drill.frozen.xml",
)
# now use frozen_xml everywhere
```

Behaviour:
- If the source scene already has no morph joints (`thumb_x` etc.), the
  helper just copies it verbatim. The output path is "frozen and ready"
  regardless of starting point — callers don't need to branch.
- If the source has morph joints, the helper reads the qpos at the
  requested keyframe, extracts the morphology values from it, and bakes
  them into body transforms via `create_rigid_morphology_xml`. The
  resulting XML has 9 fewer joints and 9 fewer qpos.

## DOF count check

A correctly frozen scene drops 9 DOFs (the morph joints):

| Scene | base nq | frozen nq | base njnt | frozen njnt |
|---|---:|---:|---:|---:|
| `scene.xml` | 31 | 22 | 25 | 16 |
| `scene_prism.xml` | 31 | 22 | 25 | 16 |
| `scene_power_drill_short_proximal.xml` | 31 | 22 | 25 | 16 |
| `scene_screwdriver_medium.xml` | 31 | 22 | 25 | 16 |

Easy to sanity-check: `MjModel.nq` and `MjModel.njnt` should drop by 9.

## How the eval suite + comparison runners enforce this

`scripts/eval_suite.py` and `scripts/compare_methods.py` both:

1. Call `_set_frozen_scenes_dir(out_dir / "frozen_scenes")` at the top of
   `main()`.
2. Route every `Phase1GraspEvaluator(scene_xml=...)` construction through
   a `_frozen_scene_for(...)` resolver that caches per
   `(scene_xml, keyframe)` and refuses to run if the cache dir hasn't
   been set (raises a `RuntimeError` pointing back to this protocol).
3. Persist the frozen XMLs to `<run_tag>/frozen_scenes/` so future
   debugging starts from the exact file the experiment used.

`scripts/phase1_optimize_grasp.py` uses the same public helper via its
`_freeze_scene_xml` wrapper (it already followed the protocol; the wrapper
now just calls `freeze_scene_for_eval`).

## What changed in the numbers

Re-ran the full eval suite (8 benchmarks × 2 methods × 3 seeds, 24×40 = 960
evals/seed) **with the freezing rule applied** and compared to the prior
run that hit the raw scenes:

| Benchmark | Δ (drifty) | Δ (frozen) | base oracle drifty → frozen | contact_map oracle drifty → frozen |
|---|---:|---:|---|---|
| `cube` | +0.18 | +0.28 | 6.64 → 6.70 | 6.82 → 6.98 |
| `prism` | +3.16 | **+3.26** | 1.00 → 2.31 | 4.16 → 5.57 |
| `screwdriver_medium_flat` | **−0.28** | **+0.33** | 5.67 → 5.52 | 5.39 → 5.85 |
| `screwdriver_medium_vertical` | +0.01 | −0.11 | 5.70 → 5.83 | 5.71 → 5.72 |
| `screwdriver_medium_90vertical` | −0.04 | +0.07 | 6.24 → 6.20 | 6.21 → 6.27 |
| `screwdriver_small_flat` | −0.01 | −0.01 | −0.06 → −0.06 | −0.08 → −0.07 |
| `power_drill` | −0.33 | **−1.17** | 7.39 → 7.37 | 7.06 → 6.20 |
| `power_drill_short_proximal` | +0.30 | −0.05 | 8.46 → 7.89 | 8.76 → 7.84 |

Things to notice:

- **Headline holds**: the prism win is even cleaner (+3.26), and contact_map's mean Δ vs baseline is +0.32 (vs +0.37 unfrozen), wins 4/8.
- **Sign flip on `screwdriver_medium_flat`**: previously read as a contact_map loss (−0.28), with freezing it's a win (+0.33). The earlier "loss" was a morph-drift artifact.
- **Drill scores deflate when frozen**: both methods drop on `power_drill_short_proximal` (8.46 → 7.89 baseline, 8.76 → 7.84 contact_map). Drift had been helping the drill grasp — almost certainly because the morph DOFs were quietly rearranging the fingertips into a tighter contact pattern under load. This is exactly the kind of false-positive we want to eliminate.
- **`power_drill` baseline-vs-contact_map gap widens** to −1.17 from −0.33, suggesting the previous near-tie was also a drift artifact (different drift trajectories per method).
- **Variance increases on prism baseline** (0.17 std unfrozen → 1.10 std frozen), but stays tight for contact_map (1.96 → 0.08). The drift was masking how unreliable the baseline grasp actually is on this object — contact_map remains consistent.

Bottom line: the qualitative story from the previous run (contact_map wins big on prism, ties or marginally varies elsewhere, both fail on the small screwdriver) survives intact, but several per-benchmark numbers were lying. **Treat any pre-protocol-fix grasp eval result as suspect** and re-run if it informs a decision.

## How to make new code follow this

- Use `freeze_scene_for_eval` (do not roll your own joint baking).
- If you're writing a runner, register one frozen-scenes cache dir in your
  `main()` and route every evaluator through a resolver. Make the
  un-cached path raise loudly — silent fallthrough to the base scene is
  the entire failure mode this protocol exists to prevent.
- Contact-target YAMLs in `assets/contact_targets/` reference body-local
  coordinates of the manipulated object, which freezing preserves. No
  changes needed to the specs themselves.
