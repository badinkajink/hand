# Power Drill Run8 Recovery (2026-04-24)

## Objective
Recover and continue the power-drill run after scene generation failed to load the drill mesh during the multitask stage.

## Root Cause
Generated MJCF files were written under results paths, but asset file references (for example mesh file paths) were still relative to the original scene location. This broke mesh loading from generated scene directories.

## Fix Applied
File changed:
- src/morphohand/sampling/scene.py

Change summary:
- Added path rebasing for XML elements with a file attribute when writing generated scenes.
- For each asset file reference:
  - Resolve against source scene directory.
  - Rebase to output scene directory when possible.
  - Fall back to absolute path when relative rebasing is not possible.

## Validation
Local validation was run after the fix:
1. Generate a test scene from assets/mjcf/scene_power_drill.xml into results/phase1/tmp_pathcheck/scene_test.xml.
2. Load that generated scene with MuJoCo.
3. Result: model loaded successfully.

## Relaunch
Relaunched from sampling stage (skipping foundational):
- RUN_FOUNDATIONAL=0
- RUN_SHORT_PHALANGES=1
- BASE_TAG=run8_power_drill_forward_to_down_r2
- SHORT_TAG=run8_power_drill_forward_to_down_shortlen_r2

Launch log:
- results/phase1/run8_power_drill_r2_launch.log

## Current Status Snapshot
At monitoring time:
- Run directory present: results/phase1/run8_power_drill_forward_to_down_r2
- Generated scene count: 41
- Log growth over 8 seconds: +88 lines (261 -> 349)

Interpretation:
- The relaunched run is active and progressing.
- MuJoCo instability warnings (QACC Nan/Inf) are still frequent, but unlike the earlier failure, the run is no longer blocked by missing mesh file errors.

## Monitoring Commands
Use these from repository root:

- Tail launch log:
  tail -n 60 results/phase1/run8_power_drill_r2_launch.log

- Check output artifacts:
  ls -1 results/phase1/run8_power_drill_forward_to_down_r2 | rg 'all_candidates_multitask.csv|summary.json|top5_candidates.csv'

- Count generated scenes:
  find results/phase1/run8_power_drill_forward_to_down_r2/generated_mjcf -type f | wc -l

- Check process output via background terminal id:
  ec6cdc65-1e87-4bc4-ad84-6cd5a1ebc97e
