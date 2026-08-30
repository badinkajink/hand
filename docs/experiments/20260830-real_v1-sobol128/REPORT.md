# Real-v1 Sobol-128 open-loop morphology pilot

**Date:** 2026-08-30  
**Status:** pilot complete; two new hands merit export/hardware-clearance work  
**Task:** fixed-palm bench reorientation of the medium screwdriver using an open-loop joint
trajectory and only hardware-configurable real-v1 mount coordinates

## Result

The sampling route works well enough to continue. The 128 new hands produced multiple nominal
open-loop reorienters, and one new design (`sv1_u0100`) outperformed the tested g12 operating
point under the full combined-error ensemble. A different new design (`sv1_w0116`) is the clear
nominal winner but is substantially less robust.

This is not yet a hardware-ready winner declaration. The dynamic clearance gate covers the
MuJoCo links and tips, not the real servo bodies and mount side profiles. Positive simulated
clearance is necessary but not sufficient until those envelopes are measured or added to the
model.

## Population and the X bias

The pilot generated 128 scrambled Sobol designs (seed `20260830`) plus six anchors, including
g12. The six sampled coordinates are thumb/index/middle X and Y within `REAL_V1_WORKSPACE`.

- 96 samples are unmodified Sobol points.
- 32 samples form a mild outward-X stratum. Only index-X and middle-X use `u -> u^0.75`, moving
  them toward positive palm X; thumb-X and all Y coordinates remain unbiased.
- The biased stratum averages **+4.4 mm** index/middle X offset versus approximately **0.0 mm**
  in the uniform stratum.
- The bias is deliberately partial. The larger servo bodies are absent from the scene, so a
  guessed packaging preference must not erase unbiased coverage.

This bias increases pair-side packaging room relative to the thumb side. It does **not** directly
separate index from middle; their mutual collision is governed strongly by pair-Y and by the
trajectory. That is why every promoted design still passes a full dynamic clearance trace.

## Funnel

| gate | result |
|---|---:|
| generated | 128 new + 6 anchors |
| fitted grasp at one of four pilot placements | 59 new + 6 anchors |
| g12 deploy-recipe fixed cell, held alignment >= 0.7 | 4 new + 2 anchors |
| same cell and >= 5 mm dynamic simulated-link clearance | 2 new + 2 anchors |
| local operating-point sweep | 960 cells over 10 finalists |
| best safe cell confirmation | 10 repeats over 8 designs |
| combined-error evaluation | 40 half-error + 40 full-error draws over 6 designs |

The outward stratum was not more graspable: 14/32 (43.8%) versus 45/96 (46.9%) for uniform.
At the fixed g12 recipe, however, it contributed 2/14 hands above 0.7 versus 2/45 uniform hands.
That is an encouraging enrichment signal, but the counts are too small and operating-point
specific to claim that the bias improves reorientation probability.

## Confirmed nominal cells

`kept` and cosine are over 10 repeats. Clearance is the minimum dynamic finger-link/tip distance
over close, settle, turn, and hold. All listed cells use the built-style flat tip, 10 mm squeeze,
100 mm fixed-palm bench, and support at Y = -35 mm.

| design | source | straddle / thumb axial | k / angle | held cosine | kept | clearance |
|---|---|---:|---:|---:|---:|---:|
| **sv1_w0116** | outward-X | 40 / 10 mm | 0.55 / -70 deg | **0.918 +/- 0.001** | 10/10 | **11.66 mm** |
| sv1_w0099 | outward-X | 40 / 10 mm | 0.15 / -80 deg | 0.824 +/- 0.278 | 9/10 | 5.64 mm |
| **sv1_u0060** | uniform | 40 / 10 mm | 0.15 / -90 deg | **0.822 +/- 0.069** | 10/10 | **8.26 mm** |
| g12 | anchor | 32 / 10 mm | 0.05 / -60 deg | 0.792 +/- 0.266 | 9/10 | 5.29 mm |
| rv05_manual | anchor | 40 / 10 mm | 0.05 / -60 deg | 0.772 +/- 0.023 | 10/10 | 11.14 mm |
| **sv1_u0100** | uniform | 40 / 10 mm | 0.15 / -60 deg | **0.766 +/- 0.060** | 10/10 | **9.82 mm** |
| sv1_u0039 | uniform | 32 / 10 mm | 0.55 / -60 deg | 0.715 +/- 0.024 | 10/10 | 7.60 mm |
| sv1_w0064 | outward-X | 40 / 10 mm | 0.15 / -80 deg | 0.245 +/- 0.346 | 4/10 | 9.81 mm |

The single-rollout cell sweep overestimated `sv1_w0064` (0.876 became 0.245 over 10). This is a
useful confirmation that repeated evaluation is mandatory and that the pipeline must not rank
hands on a single nominal rollout.

## Combined-error results

A win is a retained tool with final cosine >= 0.7. Every row has 40 draws at each error level,
using the same draw sequence across designs. `kept` counts retention regardless of alignment.

| design | half-error win / kept | full-error win / kept | interpretation |
|---|---:|---:|---|
| **sv1_u0100** | **50.0% / 87.5%** | **45.0% / 75.0%** | strongest new robust hand |
| rv05_manual | 82.5% / 95.0% | 42.5% / 77.5% | strongest half-error anchor |
| sv1_u0060 | 40.0% / 70.0% | 25.0% / 70.0% | balanced nominal/clearance candidate |
| g12, newly tuned cell | 35.0% / 35.0% | 25.0% / 27.5% | nominal cell is retention-sensitive |
| sv1_w0116 | 30.0% / 32.5% | 20.0% / 30.0% | excellent nominally, brittle |
| sv1_w0099 | 35.0% / 45.0% | 17.5% / 25.0% | noisy and brittle |

The published g12 operating point previously scored 70% on the careful-bench distribution and
27% under full error. The newly tuned g12 cell here was selected for nominal cosine and clearance,
not robustness; its 25% full-error result is consistent with the old full-error number but its
retention is worse. The important comparison is that `sv1_u0100` reaches 45% full-error wins on
the same draws used for the other rows in this table.

## Mount coordinates and hardware envelope

All high-priority finalists below pass the current measured gantry-travel check.

| design | thumb XY | index XY | middle XY | X sep | Y sep |
|---|---:|---:|---:|---:|---:|
| g12 | (-42.5, 0.0) | (42.5, 40.0) | (42.5, -40.0) | 85.0 | 80.0 |
| sv1_w0116 | (-38.6, 9.8) | (36.8, 61.8) | (39.2, -57.5) | 76.6 | 119.2 |
| sv1_w0099 | (-43.8, -2.3) | (41.1, 35.7) | (55.2, -44.3) | 92.0 | 79.9 |
| sv1_u0060 | (-45.9, 14.3) | (26.1, 30.1) | (26.4, -67.5) | 72.2 | 97.6 |
| **sv1_u0100** | **(-40.4, 45.8)** | **(37.1, 31.6)** | **(36.9, -28.6)** | **77.5** | **60.2** |

Coordinates are palm-frame millimetres, the same frame used by the hardware command path.
`sv1_u0100` is notably asymmetric along the screwdriver axis: its thumb is displaced to
Y = +45.8 mm. The earlier compact-family search could not express this hand.

## What changed in the pipeline

- `real_v1_design_search.py` now provides seeded, prefix-stable Sobol sampling, the mixed
  outward-X stratum, stable IDs, a complete manifest, a grasp-only stage, separate generation
  directories, g12 anchoring, and carry clearance ranking.
- `probe_real_v1_carry.py` now measures clearance during the turn, re-squeeze, and hold as well
  as close/lift. Previously its claimed whole-schedule minimum silently excluded those phases.
- `real_v1_deploy_envelope.py` can ingest an external design table and sampler manifest, uses the
  correct fixed-palm bench schedule in cell mode, and records dynamic clearance for cell trials.
- Tiny negative coordinates are serialized in fixed-point notation; the previous scientific
  notation could be misparsed by the morphology generator as a command-line option.

The apparent g12 clearance contradiction found during the pilot was resolved: the old -90 degree,
4 mm squeeze, lifted search maneuver dynamically overlaps index/middle, while the exported safe
g12 plan is a different -70 degree, 10 mm squeeze, flat-tip, fixed-palm maneuver. Under that real
deploy recipe the dynamic checker reports the expected 8.7 mm g12 clearance.

## Artifacts

- [`generated_manifest.json`](generated_manifest.json): all 134 vectors, Sobol provenance,
  physical mounts, sampling class, and scene path.
- [`grasp_screen.json`](grasp_screen.json): joined grasp-stage results.
- [`deploy_recipe_cells_clearance.json`](deploy_recipe_cells_clearance.json): fixed g12-recipe
  screen with full dynamic clearance.
- [`finalist_cell_sweep.json`](finalist_cell_sweep.json): 960-cell local sweep.
- [`selected_safe_cells.json`](selected_safe_cells.json): selected clearance-safe operating point
  for each finalist.
- [`robustness_ensemble.json`](robustness_ensemble.json): 480 combined-error rollouts.
- [`videos/`](videos/): six 6.25-second baseline renders, including all promoted new designs.

## Decision and next step

Continue the morphology-sampling route, but do not scale directly to 4,096 full simulations yet.
The immediate hardware-facing promotion set is:

1. `sv1_u0100` — robust winner; export and add real housing-envelope clearance first.
2. `sv1_u0060` — best balance of repeated nominal behavior, retention, and clearance.
3. `sv1_w0116` — nominal/mechanism study; useful to understand why its large margin is brittle.
4. g12 — retain as the already-understood hardware baseline.

Before hardware motion, add conservative CAD boxes or measured swept envelopes for servo bodies,
horns, and brackets to the exact plans. Then export build sheets/firmware commands and replay the
four plans in sim with those bodies enabled. If `sv1_u0100` remains clear, it is the first new hand
to build. In parallel, the cheap geometry/grasp sampler can be extended to 4,096 points, but only
the diverse graspable subset should pay for the deploy cell sweep.

